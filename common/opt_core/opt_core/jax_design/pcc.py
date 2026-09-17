"""The persistent compilation cache + XLA autotune pin of a JAX design engine, keyed by the running stack.

Contract. A JAX process compiles its jitted design step on first use and, at every fresh compile, re-runs XLA's GEMM/conv autotuning —
so two fresh processes of the same program on the same GPU type hold (slightly) different executables and the same seed gives different
trajectories. The lever is two settings that must be in force BEFORE the first jax computation of the process:

  1. the persistent compilation cache — ``JAX_COMPILATION_CACHE_DIR=<dir>`` with the two store-everything thresholds at 0 (every
     executable is written, whatever its compile time or size), so the second process on skips the compile;
  2. the autotune pin — ``XLA_FLAGS`` gains ``--xla_gpu_dump_autotune_results_to=<dir>/xla_autotune_results.pb`` in the ONE process
     that populates the cache (the kit's ``warm`` verb, ``autotune="dump"``) and ``--xla_gpu_load_autotune_results_from=<same file>`` in
     every other process (``autotune="auto"``: load; when the file is absent, under the EXACT tier — ``exact=True``, the default of every
     call here, because this pin IS the cross-process bitwise mechanism of an exact JAX line — a refusal by name: warm once, then run the
     fleet, since a fleet of cold processes each dumping its own results would hold executables that differ box to box; under a
     tolerance-class line — ``exact=False`` — the process proceeds COLD, no autotune flag, named ``autotune=cold``, exit 0), so every
     process of an exact line holds the populating process's executable: trajectories are bit-exact across processes and hosts of one GPU
     type. The flag is APPENDED to whatever
     ``XLA_FLAGS`` already says — a library that assigns ``XLA_FLAGS`` wholesale at import silently drops the pin, which is why
     :func:`xla_flags_append` is the only writer here and why :func:`already_enabled` exists (the never-re-apply rule: a cache directory or
     an autotune flag already in force is reported, never overridden).

The cache belongs to one (jax, jaxlib, PJRT plugin, GPU product) stack: :func:`key` names it
``jax<version>-jaxlib<version>-<plugin label><version>-<gpu slug>`` from the installed distributions and the probed GPU name and, under
``exact=True``, RAISES when any part is unknown — a wrong-but-plausible key is a silent cache collision, so there is never a shared
``unknown`` bucket. Under ``exact=False`` an unknown part is ISOLATED instead: the raw version string that IS readable (a live
``sys.modules["jax"].__version__``) or ``unknown<token>`` (:func:`opt_core.jit_cache.isolation_token`: this process alone, a cold compile
nobody reuses), and :func:`key_facts` carries the word ``cache_key=unknown(<parts>)`` for the activation line. :func:`cache_dir` places
the files under ``<root>/<key>/...``; whether a pre-set directory is kept is :func:`opt_core.jit_cache.keep_or_key`'s rule.

Two routes, one plan (:func:`plan`): :func:`env` returns the variables a mode table exports (the environment route: nothing imported, the
engine reads them at its own jax import — the route of every command-line process); :func:`enable` applies the same variables to a live
environment for a resident driver that did not come through the environment route and, when jax is already imported, mirrors them into
``jax.config``. :func:`enable` is a recorded no-op when the same settings are already in force (a driver started under the mode table's
exports), refuses by name on a CONFLICTING setting, and refuses by name when a live jax cannot be shown to have uninitialised backends
(``XLA_FLAGS`` is read once, at backend initialisation — a pin applied after that would be recorded and silently ignored).
:func:`identity_key` is the evidence of WHICH executables a run used (sha256 of the autotune file's bytes + sha256 of the sorted names of the
cache entries): equal equality keys and equal seeds => equal trajectories. :func:`evidence_fields` turns the route's record into the fields
of the kit's one activation line (``opt_core.report.kv``); the sentence is the kit's.

Numerics class: exact (arithmetic unchanged — every process runs the populating process's executable). Standard library only; jax is
never imported here (a live jax is read from ``sys.modules``).
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Mapping, MutableMapping, Optional

from .. import gates

CACHE_DIR_VAR = "JAX_COMPILATION_CACHE_DIR"
STORE_ALL = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",        # every executable is written, whatever its compile time ...
             "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0"}          # ... or its size
JAX_CONFIG = {"jax_compilation_cache_dir": None,                        # the jax.config names of the three variables above (value: cache dir)
              "jax_persistent_cache_min_compile_time_secs": 0.0,
              "jax_persistent_cache_min_entry_size_bytes": 0}
XLA_FLAGS_VAR = "XLA_FLAGS"
AUTOTUNE_FILENAME = "xla_autotune_results.pb"
FLAG_LOAD = "--xla_gpu_load_autotune_results_from"
FLAG_DUMP = "--xla_gpu_dump_autotune_results_to"
AUTOTUNE_MODES = ("auto", "load", "dump", "off")
PLUGIN_DIST = "jax-cuda12-plugin"


class PccError(RuntimeError):
    """A refusal by name: the key is unresolvable, the mode is unknown, or the process is past the point where the pin can apply. The
    refusals that are the ENVIRONMENT's (unresolvable key part, cold cache under exact, a conflicting setting in force, initialised
    backends) carry ``cannot_run = True`` (``opt_core.gates.is_cannot_run``: the kit refuses the mode by name); a usage error does not."""


# ----------------------------------------------------------------------------------------------------------------- the key and the place

def gpu_slug(name: str) -> str:
    """``NVIDIA H100 80GB HBM3`` -> ``nvidia-h100-80gb-hbm3``: lower case, runs of anything but ``[a-z0-9]`` folded to one ``-``."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def plugin_label(plugin_dist: str) -> str:
    """``jax-cuda12-plugin`` -> ``cuda12plugin``: the PJRT plugin's distribution name without its ``jax`` prefix and separators."""
    parts = [p for p in re.split(r"[^a-z0-9]+", str(plugin_dist).lower()) if p and p != "jax"]
    return "".join(parts) or "plugin"


def _base(v) -> str:
    return str(v).partition("+")[0]


def _live_version(module_name: str) -> Optional[str]:
    """``__version__`` of an ALREADY IMPORTED module (``sys.modules``; nothing is imported here), else None."""
    m = sys.modules.get(module_name)
    v = getattr(m, "__version__", None) if m is not None else None
    if v is None and m is not None:                                       # jaxlib keeps it in jaxlib.version
        v = getattr(getattr(m, "version", None), "__version__", None)
    return str(v) if v else None


_MISSING = {"jax": "pcc.key: no installed 'jax' distribution and no jax_version given",
            "jaxlib": "pcc.key: no installed 'jaxlib' distribution and no jaxlib_version given",
            "plugin": "pcc.key: no installed '{plugin_dist}' distribution and no plugin_version given (pass plugin_dist= for another PJRT plugin)",
            "gpu": "pcc.key: no GPU name (nvidia-smi found no device) and no gpu_name given"}


def key_facts(jax_version: Optional[str] = None, jaxlib_version: Optional[str] = None, plugin_version: Optional[str] = None,
              gpu_name: Optional[str] = None, *, plugin_dist: str = PLUGIN_DIST) -> dict:
    """The run path's key of a tolerance-class line: ``{"key", "parts", "unknown", "word"}``. Every part resolved as :func:`key` resolves
    it; a part no distribution metadata / probe gives takes the live module's ``__version__`` when one is imported (recorded as read, not
    unknown), else the ISOLATED part ``unknown<token>`` (module contract). ``unknown`` lists the isolated parts and ``word`` is
    ``cache_key=unknown(<part>,…)`` (None when nothing is unknown) — the named uncertainty; the key never raises here."""
    from ..jit_cache import isolation_token  # noqa: PLC0415
    from ..report import word as _word       # noqa: PLC0415
    iso = "unknown" + isolation_token()
    parts = {"jax": jax_version or gates.dist_version("jax") or _live_version("jax"),
             "jaxlib": jaxlib_version or gates.dist_version("jaxlib") or _live_version("jaxlib"),
             "plugin": plugin_version or gates.dist_version(plugin_dist),
             "gpu": gpu_name or gates.nvidia_smi_probe(keys=("name",)).get("name")}
    unknown = [n for n in ("jax", "jaxlib", "plugin", "gpu") if not parts[n]]
    jv, lv, pv = (_base(parts[n]) if parts[n] else iso for n in ("jax", "jaxlib", "plugin"))
    gn = gpu_slug(parts["gpu"]) if parts["gpu"] else iso
    return {"key": f"jax{jv}-jaxlib{lv}-{plugin_label(plugin_dist)}{pv}-{gn}", "parts": parts, "unknown": unknown,
            "word": _word("cache_key", "unknown", *unknown) if unknown else None}


def key(jax_version: Optional[str] = None, jaxlib_version: Optional[str] = None, plugin_version: Optional[str] = None,
        gpu_name: Optional[str] = None, *, plugin_dist: str = PLUGIN_DIST, exact: bool = True) -> str:
    """``jax<version>-jaxlib<version>-<plugin label><version>-<gpu slug>``: the (jax, jaxlib, PJRT plugin, GPU product) stack a cache
    belongs to, e.g. ``jax0.4.30-jaxlib0.4.30-cuda12plugin0.4.30-nvidia-h100-80gb-hbm3``. Defaults read the installed distributions
    (``jax``, ``jaxlib``, ``plugin_dist``) and nvidia-smi's product name. ``exact=True``: raises :class:`PccError` (``cannot_run``) naming
    the missing part — no jax, no jaxlib, no plugin (a GPU cache keyed without its CUDA plugin version would collide across plugin
    builds), no GPU name. ``exact=False``: :func:`key_facts` ``["key"]`` (missing parts isolated, never shared)."""
    if not exact:
        return key_facts(jax_version, jaxlib_version, plugin_version, gpu_name, plugin_dist=plugin_dist)["key"]
    jv = jax_version or gates.dist_version("jax")
    if not jv:
        raise gates.cannot_run(PccError(_MISSING["jax"]))
    lv = jaxlib_version or gates.dist_version("jaxlib")
    if not lv:
        raise gates.cannot_run(PccError(_MISSING["jaxlib"]))
    pv = plugin_version or gates.dist_version(plugin_dist)
    if not pv:
        raise gates.cannot_run(PccError(_MISSING["plugin"].format(plugin_dist=plugin_dist)))
    gn = gpu_name or gates.nvidia_smi_probe(keys=("name",)).get("name")
    if not gn:
        raise gates.cannot_run(PccError(_MISSING["gpu"]))
    return f"jax{_base(jv)}-jaxlib{_base(lv)}-{plugin_label(plugin_dist)}{_base(pv)}-{gpu_slug(gn)}"


def cache_dir(root: str, cache_key: str, *parts: str) -> str:
    """``<root>/<key>/<parts...>`` (default parts: ``xla``). A kit that keeps one cache per input shape passes the shape as a part."""
    return os.path.join(root, cache_key, *(parts or ("xla",)))


def autotune_file_of(cache_dir_: str, autotune_file: Optional[str] = None) -> str:
    """The autotune results file: the given path, else ``<cache_dir>/xla_autotune_results.pb``."""
    return autotune_file or os.path.join(cache_dir_, AUTOTUNE_FILENAME)


# ------------------------------------------------------------------------------------------------------------------------- XLA_FLAGS

def xla_flags_split(flags: Optional[str]) -> list:
    """The flags of an ``XLA_FLAGS`` string, in order (whitespace-separated; empty entries dropped)."""
    return [f for f in str(flags or "").split() if f]


def _flag_name(flag: str) -> str:
    return flag.split("=", 1)[0]


def xla_flags_append(existing: Optional[str], *flags: str) -> str:
    """``existing`` with each of ``flags`` appended unless a flag of the same name (the text before ``=``) is already present. Never drops
    or rewrites an existing flag: appending is the only safe write to ``XLA_FLAGS``."""
    out = xla_flags_split(existing)
    names = {_flag_name(f) for f in out}
    for f in flags:
        if f and _flag_name(f) not in names:
            out.append(f)
            names.add(_flag_name(f))
    return " ".join(out)


def xla_flags_autotune(flags: Optional[str]) -> Optional[dict]:
    """``{"mode": "load"|"dump", "file": <path>}`` when ``flags`` already carries an autotune pin, else ``None``."""
    for f in xla_flags_split(flags):
        n, _, v = f.partition("=")
        if n == FLAG_LOAD:
            return {"mode": "load", "file": v}
        if n == FLAG_DUMP:
            return {"mode": "dump", "file": v}
    return None


def autotune_mode(autotune_file: str, requested: str = "auto", *, exact: bool = True) -> str:
    """``load`` | ``dump`` | ``off`` | ``cold``. ``auto`` / ``load`` = load when the file exists; when it is absent: under ``exact=True`` a
    refusal by name (:class:`PccError`, ``cannot_run``: the cache is not populated for this key — run the kit's ``warm`` verb once per
    key; a process never dumps implicitly, and a cold process cannot hold the populating process's executable), under ``exact=False``
    ``cold`` (no autotune flag: the process autotunes for itself, named ``autotune=cold`` on the line, exit 0). ``dump`` is what the ONE
    populating process (the warm verb) asks for explicitly."""
    if requested not in AUTOTUNE_MODES:
        raise PccError(f"pcc.autotune_mode: '{requested}' is not one of {', '.join(AUTOTUNE_MODES)}")
    if requested in ("auto", "load"):
        if os.path.isfile(autotune_file):
            return "load"
        if not exact:
            return "cold"
        raise gates.cannot_run(PccError(
            f"pcc: autotune results absent at {autotune_file} — the cache is not populated for this key: run the warm verb "
            f"(autotune='dump') once per (stack, GPU type) before the fleet; a process never dumps implicitly"))
    return requested


def autotune_flag(mode: str, autotune_file: str) -> Optional[str]:
    """The one ``XLA_FLAGS`` entry for ``mode`` (``None`` for ``off``)."""
    if mode == "load":
        return f"{FLAG_LOAD}={autotune_file}"
    if mode == "dump":
        return f"{FLAG_DUMP}={autotune_file}"
    if mode in ("off", "cold"):
        return None
    raise PccError(f"pcc.autotune_flag: '{mode}' is not load | dump | off | cold")


# --------------------------------------------------------------------------------------------------------------- the two routes

def plan(cache_dir_: str, *, autotune: str = "auto", autotune_file: Optional[str] = None,
         environ: Optional[Mapping[str, str]] = None, exact: bool = True) -> dict:
    """The record both routes share, computed against ``environ`` (default ``os.environ``) without touching it: ``cache_dir``, ``autotune``
    (resolved: ``load`` | ``dump`` | ``off`` | ``cold``), ``autotune_file``, ``exact`` (the tier the caller declared), ``words`` (``["autotune=cold"]``
    for a cold tolerance-class process, else ``[]``) and ``exports`` = the variables to put in force — ``JAX_COMPILATION_CACHE_DIR``, the two
    store-everything thresholds, and ``XLA_FLAGS`` = the existing flags with the pin appended (present only when there is something to say). Pure."""
    from ..report import word as _word  # noqa: PLC0415
    environ = os.environ if environ is None else environ
    at_file = autotune_file_of(cache_dir_, autotune_file)
    mode = autotune_mode(at_file, autotune, exact=exact)
    exports = {CACHE_DIR_VAR: cache_dir_, **STORE_ALL}
    flag = autotune_flag(mode, at_file)
    existing = environ.get(XLA_FLAGS_VAR)
    if flag is not None:
        exports[XLA_FLAGS_VAR] = xla_flags_append(existing, flag)
    elif existing:
        exports[XLA_FLAGS_VAR] = existing
    words = [_word("autotune", "cold")] if mode == "cold" else []
    return {"cache_dir": cache_dir_, "autotune": mode, "autotune_file": at_file, "exact": bool(exact), "words": words, "exports": exports}


def env(cache_dir_: str, *, autotune: str = "auto", autotune_file: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None, exact: bool = True) -> dict:
    """The variables to export for this cache (the environment route = :func:`plan` ``["exports"]``): the caller exports them."""
    return plan(cache_dir_, autotune=autotune, autotune_file=autotune_file, environ=environ, exact=exact)["exports"]


def in_force(environ: Optional[Mapping[str, str]] = None, jax_module=None) -> dict:
    """What is in force now: ``{"cache_dir_env", "cache_dir_config", "pin"}`` — the cache directory in the environment, the one in a live
    ``jax.config`` (``jax_module`` defaults to ``sys.modules.get("jax")``; jax is never imported here), the autotune pin in ``XLA_FLAGS``."""
    environ = os.environ if environ is None else environ
    jax = sys.modules.get("jax") if jax_module is None else jax_module
    live = None
    if jax is not None:
        try:
            live = getattr(jax.config, "jax_compilation_cache_dir", None) or None
        except Exception:  # noqa: BLE001
            live = None
    return {"cache_dir_env": environ.get(CACHE_DIR_VAR) or None, "cache_dir_config": live, "pin": xla_flags_autotune(environ.get(XLA_FLAGS_VAR))}


def already_enabled(environ: Optional[Mapping[str, str]] = None, jax_module=None) -> Optional[str]:
    """The never-re-apply rule: a reason naming what is already in force — a cache directory in the environment or in a live
    ``jax.config``, or an autotune pin in ``XLA_FLAGS`` — else ``None``."""
    f = in_force(environ, jax_module)
    reasons = []
    if f["cache_dir_env"]:
        reasons.append(f"{CACHE_DIR_VAR}={f['cache_dir_env']} (environment)")
    if f["cache_dir_config"]:
        reasons.append(f"jax_compilation_cache_dir={f['cache_dir_config']} (jax.config)")
    if f["pin"]:
        reasons.append(f"{XLA_FLAGS_VAR} autotune {f['pin']['mode']}={f['pin']['file']}")
    return "; ".join(reasons) or None


def backend_initialized(jax_module=None) -> Optional[bool]:
    """Whether the live jax has initialised its backends (``XLA_FLAGS`` is read only before this point): ``None`` when jax is not imported
    or the answer is not available from this jax version."""
    jax = sys.modules.get("jax") if jax_module is None else jax_module
    if jax is None:
        return None
    try:
        xb = sys.modules.get("jax._src.xla_bridge")
        if xb is not None and hasattr(xb, "backends_are_initialized"):
            return bool(xb.backends_are_initialized())
    except Exception:  # noqa: BLE001
        pass
    return None


def backend_state(jax_module=None) -> str:
    """``no_jax`` | ``uninitialized`` | ``initialized`` | ``unknown`` — :func:`backend_initialized` as the word the record carries."""
    jax = sys.modules.get("jax") if jax_module is None else jax_module
    if jax is None:
        return "no_jax"
    b = backend_initialized(jax)
    return "unknown" if b is None else ("initialized" if b else "uninitialized")


def enable(cache_dir_: str, *, autotune: str = "auto", autotune_file: Optional[str] = None,
           environ: Optional[MutableMapping[str, str]] = None, makedirs: bool = True, jax_module=None, exact: bool = True) -> dict:
    """The in-process route for a resident driver: puts :func:`plan`'s variables into ``environ`` (default ``os.environ``), creates the
    directory, and — when jax is already imported — mirrors the cache settings into ``jax.config``. Returns the plan record plus
    ``applied_via`` (``environ`` | ``environ+jax.config`` | ``already``: the same settings were in force, nothing written), ``backend_state``
    and ``created``. Refuses by name (:class:`PccError`, ``cannot_run``) on a CONFLICTING setting already in force (another cache directory,
    another pin: proceeding would run on a cache directory or an autotune file other than the one this record names), and when a live jax's
    backends are initialised (``XLA_FLAGS`` is read once at backend initialisation: the pin cannot apply — use the environment route). A
    live jax whose backend state cannot be read (``unknown``: this jax version does not say) is not a refusal: the settings are applied and
    the record carries the word ``backend_state=unknown``."""
    from ..report import word as _word  # noqa: PLC0415
    environ = os.environ if environ is None else environ
    jax = sys.modules.get("jax") if jax_module is None else jax_module
    rec = plan(cache_dir_, autotune=autotune, autotune_file=autotune_file, environ=environ, exact=exact)
    want_flag = autotune_flag(rec["autotune"], rec["autotune_file"])
    f = in_force(environ, jax)
    state = backend_state(jax)
    conflicts = []
    for what, have in (("environment " + CACHE_DIR_VAR, f["cache_dir_env"]), ("jax.config jax_compilation_cache_dir", f["cache_dir_config"])):
        if have and os.path.abspath(have) != os.path.abspath(cache_dir_):
            conflicts.append(f"{what}={have} (requested {cache_dir_})")
    if f["pin"] and (want_flag is None or f["pin"] != xla_flags_autotune(want_flag)):
        conflicts.append(f"{XLA_FLAGS_VAR} autotune {f['pin']['mode']}={f['pin']['file']} (requested {want_flag or 'off'})")
    if conflicts:
        raise gates.cannot_run(PccError("pcc.enable: a conflicting setting is already in force — " + "; ".join(conflicts)))
    same = (f["cache_dir_env"] or f["cache_dir_config"]) and (want_flag is None or f["pin"] == xla_flags_autotune(want_flag))
    if same:                                                              # the mode table's exports are in force: record, write nothing
        return {**rec, "applied_via": "already", "backend_state": state, "created": False}
    if state == "initialized":
        raise gates.cannot_run(PccError(
            "pcc.enable: live jax backend state is initialized; XLA_FLAGS is read once at backend initialisation, so the pin cannot "
            "apply — enable before importing jax's backends or use the environment route (the mode table's exports)"))
    if state == "unknown":                                                # this jax does not say: apply, and name it
        rec = {**rec, "words": list(rec["words"]) + [_word("backend_state", "unknown")]}
    created = False
    if makedirs and not os.path.isdir(cache_dir_):
        os.makedirs(cache_dir_, exist_ok=True)
        created = True
    for k, v in rec["exports"].items():
        environ[k] = v
    via = "environ"
    if jax is not None:
        for name, value in JAX_CONFIG.items():
            jax.config.update(name, cache_dir_ if value is None else value)
        via = "environ+jax.config"
    return {**rec, "applied_via": via, "backend_state": state, "created": created}


# ------------------------------------------------------------------------------------------------------------------------- evidence

def identity_key(cache_dir_: str, autotune_file: Optional[str] = None) -> dict:
    """Which executables a run used: ``autotune_sha256`` = sha256 of the autotune results file's BYTES (``None`` before the populating
    process exits); ``cache_listing_sha256`` = sha256 of the sorted NAMES of the cache entries (the autotune file excluded), newline-joined;
    ``n_cache_entries`` = their count. Record it beside the results: equal keys and equal seeds => bit-exact-equal trajectories."""
    at_file = autotune_file_of(cache_dir_, autotune_file)
    out = {"autotune_sha256": None, "cache_listing_sha256": None, "n_cache_entries": 0}
    if os.path.isfile(at_file):
        with open(at_file, "rb") as fh:
            out["autotune_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    if os.path.isdir(cache_dir_):
        skip = os.path.basename(at_file) if os.path.dirname(os.path.abspath(at_file)) == os.path.abspath(cache_dir_) else None
        names = sorted(n for n in os.listdir(cache_dir_) if n != skip)
        out["n_cache_entries"] = len(names)
        out["cache_listing_sha256"] = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return out


def evidence_fields(record: Optional[Mapping] = None) -> dict:
    """The fields of the kit's one activation line for this lever (pass to ``opt_core.report.kv(**fields)``) from the route's record
    (:func:`plan` or :func:`enable`; ``None`` = the lever is off): ``jax_cache`` (the directory or ``off``), ``autotune`` (``load`` | ``dump``
    | ``off`` | ``cold`` as the record resolved it), ``via`` (``environ`` | ``environ+jax.config`` | ``already`` | ``exports`` for a bare plan),
    ``autotune_sha256`` (12 hex digits or ``none``), ``cache_entries``. The record's ``words`` (``autotune=cold``, ``backend_state=unknown``)
    are the line's to append (:func:`opt_core.report.with_words`). A lever that is off prints ``jax_cache=off`` — never silence."""
    if not record:
        return {"jax_cache": "off", "autotune": "off", "via": None, "autotune_sha256": None, "cache_entries": 0}
    ident = identity_key(record["cache_dir"], record.get("autotune_file"))
    sha = ident["autotune_sha256"]
    return {"jax_cache": record["cache_dir"], "autotune": record["autotune"], "via": record.get("applied_via", "exports"),
            "autotune_sha256": sha[:12] if sha else None, "cache_entries": ident["n_cache_entries"]}
