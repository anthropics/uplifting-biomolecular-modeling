"""ccache — the compile-cache deployment lever (strategy F3.jit_cache_keyed), carried by EVERY kit mode (exact, fast, big) and never by the
stock line: JAX's persistent compilation cache placed in the DRIVER CHILD's environment (the package process imports no jax; the driver's jax
reads ``JAX_COMPILATION_CACHE_DIR`` and the two thresholds at import), so a process finds the executables an earlier process on the same stack
compiled — one per distinct residue count (the model, its 3 recycles inside, is one jitted program per length; tens of seconds of XLA
compilation each on an H100) — instead of compiling them again. The environment is the shared core's (``opt_core.capture.xla_cache.persistent_cache_env``: the directory plus
``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`` / ``JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1``, every compilation stored).

Placement (:func:`place`, in the package process before the driver starts), by the environment the package sees — first match wins:
  * ``JAX_COMPILATION_CACHE_DIR`` set by the caller: KEPT as given (``source=kept``; jax keys every entry by program, flags, jaxlib and
    device itself). On the stock line the same name stays a must-be-absent name (``stock/PINS.json`` stock_environment): the stock child never sees it.
  * ``AF2IG_OPT_JIT_ROOT`` (the kit's own word; ``configs/<card>.env`` gives it a default): ``<root>/<stack key>/jax`` (``source=AF2IG_OPT_JIT_ROOT``).
  * ``MODEL_OPT_JIT_ROOT`` (the tree's cross-kit word for the same root): likewise (``source=MODEL_OPT_JIT_ROOT``). When both are set the kit's own
    word wins and the activation line notes it if they differ.
  * neither: the package cache root, ``<AF2IG_OPT_CACHE_DIR | $XDG_CACHE_HOME/af2ig_opt | ~/.cache/af2ig_opt>/jit`` (``source=default``) — the
    lever is engaged in every kit mode with no variable set.
The stack key is ``jax<v>-jaxlib<v>-cuda12plugin<v>-sm<cc>`` (the installed distributions and the card's compute capability: a wheel or card
change is a cold directory by name, never a stale load). A directory that cannot be created or written steps aside BY NAME (``state=skipped
reason=cache_dir_unwritable:<dir>``; the run compiles as stock does), a core without ``opt_core.capture`` likewise
(``reason=opt_core.capture_unavailable``); ``MODEL_OPT_LEVERS_OFF=ccache`` drops it (``state=off reason=levers_off``). Never a refusal, never
a partial run: the lever has no numerics. Hazard, documented: keep the root at ONE path across boxes — jax couples XLA's per-fusion autotune
results cache to the cache directory, and a moved directory is at best a cold cache.
Numerics: none — an executable loaded from the cache is the one that was compiled (exact stays bitwise with the stock line under ``--det 1``,
cold cache or warm). Evidence (the LEVER line after the run, ``stack.applied``): ``evidence=child_env:_JAX_COMPILATION_CACHE_DIR=<dir>_source=<…>
_key=<stack key|adopted>_entries=<files before>-><files after>_stored=<after-before>`` — a cold process stores its executables (``stored`` > 0), a
warm one stores none and its first design of each length runs at the steady model time; the activation line carries ``jit=<on|kept>:<dir>`` /
``jit=skipped:<reason>`` / ``jit=off:levers_off``.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

from . import modes, registry

NAME = registry.CCACHE
PRESET_ENV = modes.DEPLOYMENT_SWITCH                       # JAX_COMPILATION_CACHE_DIR
ROOT_ENV, COMMON_ROOT_ENV = modes.ENV_JIT_ROOT, modes.ENV_JIT_ROOT_COMMON
SUBDIR = "jax"                                             # <root>/<stack key>/jax
DEFAULT_SUBDIR = "jit"                                     # under the package cache root (stack.cache_dir) when no root variable is set


def stack_key(upstream: Optional[dict], gpu: Optional[dict]) -> str:
    """``jax<v>-jaxlib<v>-cuda12plugin<v>-sm<cc>`` from the installed distributions (stack.upstream_versions) and the card nvidia-smi reports (stack.gpu_probe) — no jax import."""
    def v(name):
        return (upstream or {}).get(name) or "absent"
    cc = (gpu or {}).get("cc")
    return f"jax{v('jax')}-jaxlib{v('jaxlib')}-cuda12plugin{v('jax-cuda12-plugin')}-sm{str(cc).replace('.', '') if cc else 'none'}"


def entries(directory) -> Optional[int]:
    """Files under the cache directory now (jax's executables and its coupled XLA caches); None without a directory."""
    if not directory or not os.path.isdir(str(directory)):
        return None
    return sum(len(files) for _, _, files in os.walk(str(directory)))


def root_of(environ, default_root: str) -> Tuple[str, str, Optional[str]]:
    """(root, source, note): the kit's own word, else the tree's, else the package default; the note names a differing MODEL_OPT_JIT_ROOT that lost."""
    own, common = (environ.get(ROOT_ENV) or "").strip(), (environ.get(COMMON_ROOT_ENV) or "").strip()
    if own:
        note = (f"{COMMON_ROOT_ENV}={common} is set too: {ROOT_ENV}={own} wins (the kit's own word)"
                if common and os.path.abspath(common) != os.path.abspath(own) else None)
        return own, ROOT_ENV, note
    if common:
        return common, COMMON_ROOT_ENV, None
    return default_root, "default", None


XLA_CACHES_ENV, XLA_CACHES_FENCE = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "none"   # jax >= 0.4.36 couples XLA's per-fusion AUTOTUNE cache to the compilation cache directory (config jax_persistent_cache_enable_xla_caches, default 'xla_gpu_per_fusion_autotune_cache_dir'); under --det (autotune level 0) the kit fences it off: nothing measured, nothing loaded


def recipe_key(xla_flags: Optional[str], det_level: Optional[int]) -> str:
    """The numerics recipe the driver child compiles under, as a directory name: ``det<level>-<8 hex of the child's XLA_FLAGS words>`` under ``--det``,
    ``default`` at default numerics without XLA flags, ``default-<8 hex>`` with the caller's own flags. The cache and the program store live UNDER it
    (``<root>/<stack key>/<recipe>/{jax,programs}``): entries and autotune results written under one recipe are never read under another (with one shared
    jax directory a det-1 process would read the per-fusion autotune results det-0 processes wrote, and leave the stock line's bytes)."""
    words = sorted(w for w in (xla_flags or "").split() if w)
    tag = f"det{int(det_level)}" if det_level else "default"
    return f"{tag}-{hashlib.sha256(' '.join(words).encode()).hexdigest()[:8]}" if words else tag


def place(planned: bool, environ, upstream: Optional[dict], gpu: Optional[dict], default_root: str, det_level: Optional[int] = None, child_xla_flags: Optional[str] = None) -> dict:
    """Place the cache for the driver child (module docstring); returns the lever's state — ``env`` is what cli.py adds to the child's environment
    (empty unless ``state == "on"``). Creates the directory; nothing else is written here."""
    st = {"name": NAME, "state": "off", "reason": None, "source": None, "root": None, "key": stack_key(upstream, gpu), "dir": None, "env": {},
          "entries_before": None, "note": None, "recipe": recipe_key(child_xla_flags if child_xla_flags is not None else environ.get("XLA_FLAGS"), det_level),
          "xla_caches": XLA_CACHES_FENCE if det_level else "jax_default"}
    if not planned:                                                     # MODEL_OPT_LEVERS_OFF=ccache
        st["reason"] = "levers_off"
        return st
    try:
        from opt_core.capture import xla_cache as _core_xla_cache      # the shared seam (standard library only at import)
    except ImportError:
        st.update(state="skipped", reason="opt_core.capture_unavailable")
        return st
    preset = (environ.get(PRESET_ENV) or "").strip()
    if preset:                                                          # the caller's own directory: kept as given, the store-everything thresholds beside it
        env = {PRESET_ENV: preset}
        env.update(_core_xla_cache.PERSISTENT_CACHE_ENV)
        directory, source, root, note = preset, "kept", None, None   # the caller's directory is one namespace for every recipe the caller runs under it: the --det fence below still holds
    else:
        root, source, note = root_of(environ, default_root)
        root = os.path.abspath(os.path.expanduser(root))
        env = _core_xla_cache.persistent_cache_env(root, st["key"], os.path.join(st["recipe"], SUBDIR))   # <root>/<stack key>/<recipe>/jax
        directory = env[PRESET_ENV]
    st.update(source=source, root=root, dir=directory, note=note)
    try:
        os.makedirs(directory, exist_ok=True)
        writable = os.path.isdir(directory) and os.access(directory, os.W_OK | os.X_OK)
    except OSError as e:                                                # a root that cannot be created: the lever steps aside by name, the run compiles as stock does
        st.update(state="skipped", reason=f"cache_dir_unwritable:{directory}", note=f"{PRESET_ENV} not placed ({type(e).__name__}: {e})")
        return st
    if not writable:
        st.update(state="skipped", reason=f"cache_dir_unwritable:{directory}", note=f"{PRESET_ENV} not placed ({directory} is not writable)")
        return st
    if det_level:                                                       # --det: XLA's per-fusion autotune cache (and kernel cache) off in the child — an autotune-level-0 compile neither measures nor LOADS results
        env[XLA_CACHES_ENV] = XLA_CACHES_FENCE
    st.update(state="on", env=env, entries_before=entries(directory))
    return st


def word(st: dict) -> str:
    """The activation line's one token after ``jit=``: ``on:<dir>`` | ``kept:<dir>`` | ``skipped:<reason>`` | ``off:levers_off``."""
    if st.get("state") == "on":
        return f"{'kept' if st.get('source') == 'kept' else 'on'}:{st.get('dir')}"
    return f"{st.get('state')}:{st.get('reason')}"


def evidence(st: dict) -> str:
    """The LEVER line's evidence after the run: the directory placed, its source and key, the files under it before -> after and the difference (stored)."""
    before, after = st.get("entries_before"), entries(st.get("dir"))
    stored = (after - before) if isinstance(after, int) and isinstance(before, int) else "unknown"
    key = "adopted" if st.get("source") == "kept" else st.get("key")
    return f"child_env: {PRESET_ENV}={st.get('dir')} source={st.get('source')} key={key} recipe={st.get('recipe')} xla_caches={st.get('xla_caches')} entries={before}->{after} stored={stored}"


PROGRAMS_SUBDIR = "programs"                               # <root>/<stack key>/<recipe>/programs — beside the jax directory (L13, af2ig_opt.programs)


def programs_dir(st: dict, default_root: str) -> Tuple[Optional[str], Optional[str]]:
    """(directory, None) for the L13 program store — ``<root>/<stack key>/<recipe>/programs`` under the root the ccache lever placed (its own key), or under
    the package default root when the caller's JAX_COMPILATION_CACHE_DIR was kept or the cache is off — created here; (None, reason) when it
    cannot be created or written (the lever then steps aside by name)."""
    root = st.get("root") if st.get("state") == "on" and st.get("source") not in (None, "kept") else None
    root = os.path.abspath(os.path.expanduser(root or default_root))
    d = os.path.join(root, st.get("key") or "nokey", st.get("recipe") or "default", PROGRAMS_SUBDIR)
    try:
        os.makedirs(d, mode=0o755, exist_ok=True)               # never group/other-writable, whatever the umask: programs.refusal() rejects such a directory
        ok = os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK)
    except OSError:
        ok = False
    return (d, None) if ok else (None, f"cache_dir_unwritable:{d}")
