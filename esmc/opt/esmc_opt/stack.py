"""Activation: kit paths, pins, GPU equality, the mode's levers, and their application through the kits' OWN ``apply()``.

``activate(mode, variant, regime)`` resolves the mode (``modes.resolve``), gates (the kit tree present, the stock
pins, the upstream versions, a visible GPU), applies the datapath lever now (the
pipeline kit's ``apply('sdk', 'auto', weights_dir)``: its levers patch at the load), then ARMS the application of the
model-side lever: ESM C has no import-time activation point — the kits patch a LOADED model (``apply(client.model)``) — so a wrapper on
``esm.models.esmc.ESMC.from_pretrained`` (installed now if the module is imported, else at its import) loads the client through the
pipeline kit's own ``from_pretrained_sdk`` when that lever is on, then calls the fused kit's ``apply()`` on
``client.model``. Nothing here transcribes a kit value: every lever is the kit's own call with the kit's own defaults.

Late activation follows one rule: allowed before or after the upstream package is imported, refused by name once an ``ESMC`` client exists in the process (instances are counted by a constructor wrap from activation
on) or once a kit reports a lever already applied. Repeated ``activate()`` calls return the first report; a different mode, variant or
regime in the same process is refused. Every client built while the mode is active gets the levers (model#1, model#2, … — any
served size, one lever set per process). ``dry_run=True`` (``check``) resolves and gates and applies nothing.

Environment read by the package: ESMC_OPT (the mode, for a process that does not name it on the command line). The weights are the
HF cache under HF_HOME (upstream's own location). The package sets NO kit variable (ESMC_KIT is the kits' own marker, set by their
apply()).

The variant: named by the command line (``--variant``); a library caller that names none has it derived from the model it loads
(``ESMC.from_pretrained(<model name>)`` — the arming point), the model-side levers resolved for that variant at that moment.
"""
from __future__ import annotations

import functools
from esmc_opt._oom import is_oom
import importlib
import importlib.metadata as md
import inspect
import json
import os
import subprocess
import sys
import time
import weakref
from typing import Dict, List, Optional, Tuple

from . import modes, registry
from . import report as _report

__version__ = "0.4.0"
ENV_MODE = "ESMC_OPT"
PACKAGE_ENV = (ENV_MODE,)
PACKAGE_SWITCH_ENV = (ENV_MODE,)                                                    # the package's one SWITCH (a mode reaching a child process would change what it runs)
PINS_RELPATH = os.path.join("stock", "PINS.json")
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")
UPSTREAM_DISTS = ("esm", "torch", "flash_attn", "transformer_engine", "triton")   # versions on the ACTIVE line; the commit pin is esm's
CLIENT_MODULE = "esm.models.esmc.compatibility"                                    # the SDK client class the kits attach to
CLIENT_CLASS = "ESMC"
LOG = _report.PREFIX

_STATE: Dict[str, object] = {"report": None, "hook": None, "pipe": None, "model_index": 0}
_INSTANCES: Dict[str, object] = {"state": None}


class ActivationError(RuntimeError):
    pass


class PartialActivation(ActivationError):
    """A mode is all of its levers: a lever of the mode's set for this variant / regime / GPU class was NOT APPLIED to the loaded model
    (``levers_fallback``: its kit refused by name — it cannot run here: a compile / launch / import failure, an unsupported shape or dtype).
    The mode refuses by name — the NOT ACTIVE partial line (report.partial_refused_line) is printed, then this is raised from
    ``ESMC.from_pretrained``. Never a run under the mode's name on a subset of its levers. ``partial`` = the levers not applied."""
    def __init__(self, partial, detail: str):
        super().__init__(f"partial activation — {detail}")
        self.partial = list(partial)
        self.detail = detail


# --------------------------------------------------------------------------------------------------------- paths and pins
def tree_home() -> str:
    """The esmc/ tree this package lives in."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def kits_root() -> str:
    """The kits' package directory (esmc_opt/kits)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "kits")


def kit_dir(name: str) -> str:
    return os.path.join(kits_root(), registry.LEVERS[name]["dir"])


def kits_gate(levers) -> Tuple[bool, Optional[str]]:
    root = kits_root()
    if not os.path.isdir(root):
        return False, f"the kits package not found at {root} (esmc_opt/kits)"
    for n in levers:
        d = kit_dir(n)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "__init__.py")):
            return False, f"lever {n}: kit dir {d} missing or without __init__.py"
    return True, None


def pins() -> dict:
    return json.load(open(os.path.join(tree_home(), PINS_RELPATH)))


def dist_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def upstream_versions() -> Dict[str, Optional[str]]:
    return {n: dist_version(n) for n in UPSTREAM_DISTS}


DRIFT_DISTS = ("torch", "triton", "flash_attn", "transformer_engine")                  # the stack pins whose installed version is compared and, off the pin, NAMED (never refused)


def version_gate(p: dict, levers) -> Tuple[bool, Dict[str, Optional[str]], Optional[str], Dict[str, str]]:
    """Installed distributions against stock/PINS.json. The one refusal: esm (the stock) not installed — the levers patch the installed
    model. What "stock" means is decided by the upstream COMMIT (pins_gate), not here. Every stack pin off its version (torch, triton,
    flash_attn, transformer_engine: another version, or absent) is DRIFT: returned by name for the ACTIVE line (``drift=``), never a
    refusal — a lever that cannot run on what is installed steps aside by its own name at apply, and the KERNELS line says what the
    built model engages. Returns (ok, have, reason, drift)."""
    have = upstream_versions()
    if have["esm"] is None:
        return False, have, "the stock package esm is not installed (the levers patch the installed model; install the stock pin first)", {}
    want = {n: (p.get("pins") or {}).get(n) for n in DRIFT_DISTS}
    drift = {n: f"{have.get(n) or 'absent'}(pin_{want[n]})" for n in DRIFT_DISTS if want.get(n) and have.get(n) != want[n]}
    return True, have, None, drift


def _check_pins_module():
    import importlib.util
    path = os.path.join(tree_home(), CHECK_PINS_RELPATH)
    spec = importlib.util.spec_from_file_location("esmc_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                                           # type: ignore[union-attr]
    return mod


def pins_gate(p: dict) -> Tuple[bool, Dict[str, dict], Optional[str]]:
    """The commit-level pin check (stock/check_pins.py: pip's direct_url.json commit, or the stock/ archive sha256) for the esm SDK.
    Not at the pin -> refused. Returns (ok, detail, reason)."""
    try:
        cp = _check_pins_module()
        bad, detail = cp.check(p.get("upstream") or {})
    except Exception as e:  # noqa: BLE001
        return False, {}, f"the pin check could not run ({CHECK_PINS_RELPATH}: {e!r})"
    if bad:
        return False, detail, "upstream not at the pinned commit: " + "; ".join(bad)
    return True, detail, None


def nvidia_smi_probe() -> dict:
    """GPU equality without torch: name, compute capability and memory of GPU 0 from nvidia-smi (None when unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        return {"name": None, "cc": None, "sm": None, "mem_mib": None, "probe": f"nvidia-smi unavailable ({type(e).__name__})"}
    if out.returncode != 0 or not out.stdout.strip():
        return {"name": None, "cc": None, "sm": None, "mem_mib": None, "probe": f"nvidia-smi rc={out.returncode}"}
    parts = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    name, cc = parts[0], parts[1]
    mem = None
    if len(parts) > 2:
        try:
            mem = int(parts[2].split()[0])
        except ValueError:
            mem = None
    return {"name": name, "cc": cc, "sm": "sm" + cc.replace(".", ""), "mem_mib": mem, "probe": "nvidia-smi"}


def gpu_info() -> dict:
    return nvidia_smi_probe()


def snapshot_dir(variant: str) -> Optional[str]:
    """The variant's HF snapshot directory (config.json + the safetensors): under $HF_HOME, else the user cache — the directory the SDK
    resolves the model name to (refs/main → the pinned commit); None when it does not exist."""
    roots = [os.environ.get("HF_HOME"), os.path.join(os.path.expanduser("~"), ".cache", "huggingface")]
    repo = modes.VARIANT_HF_REPO[variant].replace("/", "--")
    for hf_home in [r for r in roots if r]:
        base = os.path.join(hf_home, "hub", f"models--{repo}")
        ref = os.path.join(base, "refs", "main")
        if os.path.isfile(ref):
            d = os.path.join(base, "snapshots", open(ref).read().strip())
            if os.path.isdir(d):
                return d
        snaps = os.path.join(base, "snapshots")
        if os.path.isdir(snaps):
            dirs = sorted(os.listdir(snaps))
            if dirs:
                return os.path.join(snaps, dirs[0])
    return None


def weights_dir(variant: Optional[str]) -> Optional[str]:
    """The weights directory the pipeline kit sizes its loader on: the variant's snapshot directory when it
    resolves (snapshot_dir), else None (no variant named yet, or no snapshot: the kit then sizes nothing — its own table)."""
    return snapshot_dir(variant) if variant else None


def stack_key(rep: Optional[dict] = None) -> str:
    """The JIT cache key of the running stack: torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g.
    torch2.11.0-cu130-sm90 — the torch and CUDA versions from the installed torch, the capability from `rep["gpu"]` (an activation report)
    or from nvidia-smi (no CUDA context is created); a persistent JIT / census cache directory keyed by it survives a stack change.
    Refuses by name (RuntimeError), never a silently malformed key, when the CUDA build or the GPU capability is not readable."""
    import importlib.metadata as _md
    v = _md.version("torch").split("+")[0]
    try:
        import torch
        cu = torch.version.cuda
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"stack_key: torch.version.cuda not readable ({type(e).__name__}: {e})") from e
    if not cu:
        raise RuntimeError("stack_key: torch.version.cuda is empty (a non-CUDA torch build) — no JIT cache key")
    cu = cu.replace(".", "")
    g = (rep or {}).get("gpu") or nvidia_smi_probe()
    sm = (g or {}).get("sm")
    if not sm:
        raise RuntimeError(f"stack_key: no GPU compute capability readable ({(g or {}).get('probe', 'no probe result')}) — no JIT cache key")
    return f"torch{v}-cu{cu}-{sm}"


# --------------------------------------------------------------------------------------------------------- model instances
def _client_class() -> Optional[type]:
    mod = sys.modules.get(CLIENT_MODULE)
    cls = getattr(mod, CLIENT_CLASS, None) if mod is not None else None
    return cls if isinstance(cls, type) else None


def _count_instances_of(cls: type, seed_from_gc: bool = False) -> bool:
    """Wrap ``cls.__new__`` so every instance created from now on is counted (a total and a weak set of the live ones). One wrap per
    class object; the original ``__new__`` is called unchanged. With ``seed_from_gc`` the instances that already exist are added
    once by a gc scan. Returns False when the class cannot be wrapped."""
    st = _INSTANCES["state"]
    if st is not None and st["cls"] is cls:
        return st["wrapped"]
    live = weakref.WeakSet()
    state = {"cls": cls, "wrapped": False, "live": live, "built": 0, "weakref": True, "gc_seeded": None}
    _INSTANCES["state"] = state
    orig_new = cls.__new__

    @functools.wraps(orig_new)
    def __new__(klass, *args, **kwargs):
        obj = orig_new(klass) if orig_new is object.__new__ else orig_new(klass, *args, **kwargs)
        state["built"] += 1
        try:
            live.add(obj)
        except TypeError:
            state["weakref"] = False
        return obj

    if "__new__" not in cls.__dict__:
        try:
            __new__.__signature__ = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            pass
    try:
        cls.__new__ = staticmethod(__new__)
    except (TypeError, AttributeError):
        state["live"] = None
        return False
    if seed_from_gc:
        import gc
        n = 0
        for o in gc.get_objects():
            if isinstance(o, cls):
                n += 1
                try:
                    live.add(o)
                except TypeError:
                    state["weakref"] = False
        state["gc_seeded"] = n
    state["wrapped"] = True
    return True


def instance_check() -> dict:
    """{"n": live instances (or the total built when they take no weak references), "source": "counter"|"gc"|"none"}."""
    st = _INSTANCES["state"]
    cls = _client_class()
    if st is not None and st["wrapped"] and st["cls"] is cls:
        n = len(st["live"]) if (st["live"] is not None and st["weakref"]) else st["built"]
        return {"n": n, "source": "counter", "built": st["built"]}
    if cls is None:
        return {"n": 0, "source": "none"}
    import gc
    return {"n": sum(1 for o in gc.get_objects() if isinstance(o, cls)), "source": "gc"}


def kit_levers_applied() -> List[str]:
    """Every lever whose kit module (already imported) reports itself applied — the kits' own state, read, never set."""
    out = []
    for name in registry.LEVER_ORDER:
        mod = sys.modules.get(registry.lever_module(name))
        if mod is None:
            continue
        st = getattr(mod, "_state", None)
        if isinstance(st, dict) and st.get("applied"):
            out.append(name)
        fn = getattr(mod, "is_applied", None)
        if callable(fn) and name not in out:
            try:
                if fn():
                    out.append(name)
            except Exception:  # noqa: BLE001
                pass
    return out


# --------------------------------------------------------------------------------------------------------- import hooks
class _OnImport:
    """A meta-path finder that runs ``callback(module)`` right after ``modname``'s body executes, once; nothing else is touched."""

    def __init__(self, modname: str, callback):
        self.modname, self.callback, self.armed = modname, callback, True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != self.modname:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            self.callback(module)
        spec.loader.exec_module = exec_module
        return spec


def _on_import(modname: str, callback) -> None:
    mod = sys.modules.get(modname)
    if mod is not None:
        callback(mod)
        return
    sys.meta_path.insert(0, _OnImport(modname, callback))


# --------------------------------------------------------------------------------------------------------- the levers
def _kit(name: str):
    return importlib.import_module(registry.lever_module(name))


def _apply_pipe(rep: dict) -> dict:
    """The pipeline kit's own apply('sdk', 'auto', weights_dir) — its levers (the device-side loader, the tokenizer fast path) patch at
    the load, so this runs before or after the upstream import alike. Returns the kit's record (or the refusal, named)."""
    kit = _kit("pipe")
    wd = weights_dir(rep["variant"])
    rec = {"lever": "pipe", "kit": registry.LEVERS["pipe"]["dir"], "weights_dir": wd}
    try:
        r = kit.apply("sdk", "auto", weights_dir=wd)
        rec.update({"applied": True, "levers": list(r.get("levers") or []), "record": r})
    except Exception as e:  # noqa: BLE001 — the kit's refusal is named, never hidden
        if is_oom(e): raise                                  # an out-of-memory propagates: no fallback applied
        rec.update({"applied": False, "refused": f"{type(e).__name__}: {str(e)[:300]}"})
    return rec


def _apply_model_lever(name: str, client, rep: dict) -> dict:
    """The fused kit's own apply(client.model) — no knobs; the kit prints its own line."""
    kit = _kit(name)
    model = getattr(client, "model", client)
    rec = {"lever": name, "kit": registry.LEVERS[name]["dir"]}
    t0 = time.perf_counter()
    try:
        r = kit.apply(model)
        rec.update({"applied": True, "record": _jsonable(r), "t_s": round(time.perf_counter() - t0, 4)})
    except Exception as e:  # noqa: BLE001 — the kit's refusal (KitRefused and friends) is named, never hidden
        if is_oom(e): raise                                  # an out-of-memory propagates: no fallback applied
        rec.update({"applied": False, "refused": f"{type(e).__name__}: {str(e)[:400]}", "t_s": round(time.perf_counter() - t0, 4)})
    return rec


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return json.loads(json.dumps(x, default=str))


def apply_to(client, trigger: str = "explicit", model_name=None) -> dict:
    """Apply the model-side lever of the active mode to a LOADED client (the wrapper on ESMC.from_pretrained calls this for every
    client built while the mode is active: model#1, model#2, …). ``model_name`` (the SDK name the client
    was loaded by) labels the APPLIED line with THIS model's variant; a model whose variant needs another lever set than the one the
    process is active with is refused by name (one lever set per process). Returns the per-model record and prints the APPLIED line
    after the kits' own lines."""
    rep = _STATE["report"]
    if not rep or not rep.get("active"):
        raise ActivationError("apply_to: no active mode (enable() first)")
    _STATE["model_index"] += 1
    variant = rep.get("variant")
    named = _variant_of_model_name(model_name) if model_name is not None else None
    if named is not None and named != variant:
        res = modes.resolve(rep["mode"], named, rep.get("regime") or modes.DEFAULT_REGIME)
        if tuple(res["levers"]) != tuple(rep.get("levers") or ()):
            raise ActivationError(f"model#{_STATE['model_index']} ({model_name}: variant {named}) needs the levers {','.join(res['levers'])}, "
                                  f"but this process is active with {','.join(rep.get('levers') or ())} (variant {variant}): one lever set per process")
        variant = named
    rec = {"model_index": _STATE["model_index"], "variant": variant, "trigger": trigger, "levers_applied": [], "levers_fallback": [], "kits": {}}
    t0 = time.perf_counter()
    pipe = _STATE.get("pipe")
    if pipe is not None:
        (rec["levers_applied"] if pipe.get("applied") else rec["levers_fallback"]).append("pipe")
        rec["kits"]["pipe"] = pipe
    for name in rep["levers"]:
        if name == "pipe":
            continue
        r = _apply_model_lever(name, client, rep)
        rec["kits"][name] = r
        (rec["levers_applied"] if r["applied"] else rec["levers_fallback"]).append(name)
    rec["t_apply_s"] = round(time.perf_counter() - t0, 4)
    rec["partial"] = partial_levers(rec)
    rep.setdefault("models", []).append({k: v for k, v in rec.items() if k != "kits"})
    rep["levers_applied"] = rec["levers_applied"]
    rep["levers_fallback"] = rec["levers_fallback"]
    rep["partial"] = rec["partial"]
    rep["kit_records"] = rec["kits"]
    print(_report.applied_line(rep, rec), flush=True)
    if rec["levers_fallback"]:                                                           # a lever of the set NOT applied: a mode is all of its levers or it refuses by name — never a model served under the mode's name on a subset
        refuse_partial(rec["levers_fallback"], rec["kits"])
    return rec                                                                           # every lever applied


def refuse_partial(partial, kit_records=None):
    """The mode's refusal when a lever of its set was not applied (``levers_fallback``): print the NOT ACTIVE partial line (each lever with
    its kit record's reason) and raise PartialActivation. Called by apply_to right after the APPLIED line."""
    detail = _report.partial_detail(partial, kit_records)
    print(_report.partial_refused_line(detail), flush=True)
    raise PartialActivation(partial, detail)


def partial_levers(rec: dict) -> List[str]:
    """The PARTIAL state of one client's application record: the levers of the mode not applied in full — every entry of
    ``levers_fallback`` (a kit's refusal of its lever, named) and every applied lever whose kit record names a fallback. Empty = the
    mode's composition applied in full (apply_to: a lever NOT applied → the NOT ACTIVE partial line, PartialActivation raised from the load)."""
    kits = rec.get("kits") or {}
    out = list(rec.get("levers_fallback") or [])
    out += [x for x in (rec.get("levers_applied") or []) if (kits.get(x.split("[", 1)[0]) or {}).get("fallback")]
    return out


def _install_hook() -> None:
    """Wrap ``ESMC.from_pretrained`` (a classmethod): the stock load — through the pipeline kit's own ``from_pretrained_sdk`` when that
    lever is on — then ``apply_to(client)``. Re-entrant: the kit's loader calls the original classmethod inside the wrapper."""
    def _patch(mod):
        cls = getattr(mod, CLIENT_CLASS, None)
        if not isinstance(cls, type) or getattr(cls, "_esmc_opt_wrapped", False):
            return
        orig = cls.__dict__["from_pretrained"]                                            # the classmethod object
        orig_fn = orig.__func__
        guard = {"depth": 0}

        @functools.wraps(orig_fn)
        def from_pretrained(klass, *args, **kwargs):
            rep = _STATE["report"]
            if guard["depth"] > 0 or not rep or not rep.get("active"):
                return orig_fn(klass, *args, **kwargs)
            guard["depth"] += 1
            t0 = time.perf_counter()
            name = args[0] if args else kwargs.get("model_name", modes.VARIANT_MODEL_NAME[rep["variant"]] if rep.get("variant") else None)
            try:
                pipe = _STATE.get("pipe")
                if pipe and pipe.get("applied") and _kit("pipe").serves_load(pipe.get("levers") or ()):
                    kwargs.pop("model_name", None)
                    kw = dict(kwargs)
                    if len(args) > 1:
                        kw["device"] = args[1]
                    if len(args) > 2:
                        kw["use_flash_attn"] = args[2]
                    import torch
                    dev = kw.pop("device", None)
                    dev = torch.device(dev) if dev is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    client = _kit("pipe").from_pretrained_sdk(name, device=dev, **kw)
                else:
                    client = orig_fn(klass, *args, **kwargs)
            finally:
                guard["depth"] -= 1
            rep["t_load_s"] = round(time.perf_counter() - t0, 4)
            if rep.get("variant") is None:                                                # a library caller named no variant: derived from the model name, the levers resolved for it now
                _resolve_variant_from_model_name(rep, name)
            from . import kernels as _kernels                                             # the accelerator proof of the model AS BUILT, before any lever touches it (the stock child proves at the same instant: driver.run)
            call = dict(kwargs)
            if len(args) > 1:                                                              # positional device / use_flash_attn (compatibility.py:154-158)
                call.update(dict(zip(("device", "use_flash_attn"), args[1:3])))
            rep["kernels"] = {k: v for k, v in _kernels.prove_once(client, mode=rep["mode"], via="kit", regime=rep.get("regime"), call=call,
                                                                   rule=modes.call_rule(rep["mode"], rep.get("regime"))).items() if k in ("words", "expected", "stack_token", "verdict", "line", "regime")}
            apply_to(client, trigger="from_pretrained", model_name=name)
            return client

        cls.from_pretrained = classmethod(from_pretrained)
        cls._esmc_opt_wrapped = True
        _count_instances_of(cls, seed_from_gc=True)
        _STATE["hook"] = f"{CLIENT_MODULE}.{CLIENT_CLASS}.from_pretrained"
    _on_import(CLIENT_MODULE, _patch)


# --------------------------------------------------------------------------------------------------------- activation
def _variant_of_model_name(model_name):
    """The variant an SDK model name (or a variant spelling) names, else None."""
    try:
        return modes.check_variant(model_name)
    except Exception:  # noqa: BLE001 — an unknown name labels nothing (the kits refuse an unserved model by shape)
        return None


def _resolve_variant_from_model_name(rep: dict, model_name) -> None:
    """The variant of an SDK model name (stock/PINS.json variants[].model_name) and the mode's lever set for it (modes.resolve); a name
    outside the table is refused by name (the levers are the variants')."""
    v = next((k for k, n in modes.VARIANT_MODEL_NAME.items() if n == model_name), None)
    if v is None:
        raise ActivationError(f"the model {model_name!r} is not one of this package's variants ({', '.join(f'{k}={n}' for k, n in modes.VARIANT_MODEL_NAME.items())})")
    res = modes.resolve(rep["mode"], v, rep.get("regime") or modes.DEFAULT_REGIME)
    rep.update({"variant": v, "levers": res["levers"], "levers_out": res["levers_out"]})


def status() -> dict:
    rep = _STATE["report"]
    return dict(rep) if rep else {"active": False, "reason": "enable() has not run"}


def _base_report(mode, variant, regime, trigger) -> dict:
    return {"active": False, "mode": mode, "variant": variant, "regime": regime, "trigger": trigger, "package_version": __version__,
            "levers": (), "levers_out": {}, "levers_applied": [], "levers_fallback": [], "partial": [], "upstream": upstream_versions(),
            "gpu": None, "kits_root": kits_root(), "tree_home": tree_home()}


def kit_env_present():
    """The kits' own switches present in the environment (registry.KIT_SWITCHES: exact names read from the kit bytes): under a named mode
    every such switch is refused by name (one mode table). A name the kits never read is ignored."""
    return sorted(k for k in registry.KIT_ENV_NAMES if k in os.environ)


def activate(mode: str, variant: Optional[str] = None, regime: Optional[str] = None, strict: bool = False, trigger: Optional[str] = None,
             dry_run: bool = False) -> dict:
    """The one activation (see the module docstring). Returns the report; ``strict`` raises ActivationError instead of an inactive report.
    ``variant`` None (a library caller): the gates run on the mode's whole lever set, the datapath lever applies unsized, and the variant
    is derived from the model name at ``ESMC.from_pretrained`` (the model-side levers resolved there)."""
    variant = (variant or "").strip().lower() or None
    if variant is not None and variant not in modes.VARIANTS and variant in set(modes.VARIANT_MODEL_NAME[k] for k in modes.VARIANTS):
        variant = modes.check_variant(variant)                                             # the SDK's model name names the size (esmc_6b -> 6b)
    regime = (regime or "").strip().lower() or modes.DEFAULT_REGIME
    try:
        m = modes.check_mode(mode)
    except modes.ResolveError as e:
        return _refuse(_base_report(mode, variant, regime, trigger), str(e), strict)
    from . import kernels as _kernels
    _kernels.listen()                                                                      # upstream's kernel warnings are captured from before its import
    prev = _STATE["report"]
    if prev is not None and not dry_run:
        same = (prev.get("mode"), prev.get("variant"), prev.get("regime")) == (m, variant, regime)
        if same:
            return dict(prev)
        return _refuse(_base_report(m, variant, regime, trigger), f"already activated as mode={prev.get('mode')} variant={prev.get('variant')} regime={prev.get('regime')} in this process", strict)
    rep = _base_report(m, variant, regime, trigger)
    rep["dry_run"] = bool(dry_run)
    env_mode = (os.environ.get(ENV_MODE) or "").strip().lower()
    if env_mode and env_mode != m:
        return _refuse(rep, f"{ENV_MODE}={env_mode!r} disagrees with the requested mode {m!r}", strict)
    if m in modes.STOCK_MODES:
        rep.update({"reason": f"mode {m}: {modes.describe(m)}, nothing applied"})
        if not dry_run:
            _STATE["report"] = rep
        rep.pop("dry_run", None)                 # the stock route has no dry-run form: the same NOT ACTIVE line as the real run
        _report.log_activation(rep)
        return dict(rep)
    present = kit_env_present()
    if present:
        return _refuse(rep, f"kit-internal switch set in the environment: {','.join(present)} — one mode table: the package composes the kits' switches itself; unset it", strict)
    try:
        if variant is not None:
            res = modes.resolve(m, variant, regime)
        else:                                                            # no variant named: the mode's whole lever set is gated now; the (variant, regime) resolution runs at the load
            res = {"mode": m, "variant": None, "regime": modes.check_regime(regime), "levers": tuple(modes.KIT_MODES[m]), "levers_out": {}}
            rep["variant_from"] = "the model name at ESMC.from_pretrained (no variant named)"
    except modes.ResolveError as e:
        return _refuse(rep, str(e), strict)
    rep.update({"variant": res["variant"], "regime": res["regime"], "levers": res["levers"], "levers_out": res["levers_out"]})
    ok, why = kits_gate(res["levers"])
    if not ok:
        return _refuse(rep, why, strict)
    try:
        p = pins()
    except Exception as e:  # noqa: BLE001
        return _refuse(rep, f"stock/PINS.json unreadable: {e!r}", strict)
    ok, have, why, drift = version_gate(p, res["levers"])
    rep["upstream"] = have
    rep["drift"] = drift                                                                    # the stack pins off their version, by name (the ACTIVE line's drift=); empty = at the pins
    if not ok:
        return _refuse(rep, why, strict)
    ok, detail, why = pins_gate(p)
    rep["pins"] = detail
    if not ok:
        return _refuse(rep, why, strict)
    gpu = gpu_info()
    rep["gpu"] = gpu
    if not gpu.get("name"):
        return _refuse(rep, f"no visible GPU ({gpu.get('probe')}); the levers are CUDA-only", strict)
    applied = kit_levers_applied()
    if applied:
        return _refuse(rep, f"a kit already reports a lever applied in this process ({', '.join(applied)}): activation must precede it", strict)
    ic = instance_check()
    rep["instances_at_activation"] = ic
    if ic["n"] > 0:
        return _refuse(rep, f"{ic['n']} {CLIENT_CLASS} instance(s) already exist in this process (counted by {ic['source']}); a client built before activation would run unpatched", strict)
    if dry_run:
        _report.log_activation(rep)
        return dict(rep)
    if "pipe" in res["levers"]:
        _STATE["pipe"] = _apply_pipe(rep)
        rep["pipe"] = {k: v for k, v in _STATE["pipe"].items() if k != "record"}
    rep["active"] = True
    _STATE["report"] = rep
    _install_hook()
    _report.register_exit_tally()
    _report.log_activation(rep)
    return dict(rep)


def _refuse(rep: dict, reason: str, strict: bool) -> dict:
    rep["active"] = False
    rep["reason"] = reason
    _report.log_activation(rep)
    if strict:
        raise ActivationError(reason)
    return dict(rep)
