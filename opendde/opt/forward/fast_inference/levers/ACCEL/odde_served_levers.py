# odde_served_levers (levers/ACCEL): the served-levers hook, with one post-runner hook point for odde_accel_v2:
#   post-runner : ODDE_DIT_ATTN=bf16 -> Tier-2 sampler attention (after the arm)
"""odde_served_levers -- make the kit's lever composition load inside the process that serves predictions (the stock `opendde pred` CLI, or
any driver that builds its runner through runner.batch_inference.get_default_runner) with no code edit.

Mechanism: a meta-path hook wraps runner.batch_inference.get_default_runner. AFTER the stock function has built the runner (every opendde module fully
imported, checkpoint loaded), the wrapper installs, in this order:
  1. DITFAST levers named in ODDE_ADDON_LEVERS (e.g. dit_hoist,dit_align)      -> odde_addon.install(runner.model, levers)
  2. the ARM-T add-on arm named by ODDE_ARM_U=1 | ODDE_ARM_Z=1                    -> odde_arm_t.install(arm, model=runner.model)
and prints ONE activation banner line per lever to stderr (+ a JSON line to $ODDE_SERVED_LEVERS_REPORT if set). Nothing is installed unless
ODDE_SERVED_LEVERS=1. ODDE_SERVED_DETERMINISTIC=1 additionally forces deterministic=True in get_default_runner (the DET recipe; requires
CUBLAS_WORKSPACE_CONFIG=:4096:8 exported before python) -- a test hook so the served worker itself can be byte-compared with the stock CLI.
"""
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, json, time, importlib.abc

__version__ = "0.2.0"
STATE = {"version": __version__, "active": False, "wrapped": False, "installs": [], "errors": {}, "deterministic_forced": False}
_TARGET = "runner.batch_inference"

def _flag(k):
    return os.environ.get(k, "0") not in ("", "0")

def requested():
    arm = "U" if _flag("ODDE_ARM_U") else ("Z" if _flag("ODDE_ARM_Z") else None)
    return {"addon_levers": [x for x in os.environ.get("ODDE_ADDON_LEVERS", "").split(",") if x.strip()], "arm": arm,
            "deterministic": _flag("ODDE_SERVED_DETERMINISTIC")}

def _banner(msg, rec=None):
    print(f"[odde_served_levers] {msg}", file=sys.stderr, flush=True)
    rp = os.environ.get("ODDE_SERVED_LEVERS_REPORT")
    if rp and rec is not None:
        try:
            with open(rp, "a") as f: f.write(json.dumps(dict(rec, t=time.time(), pid=os.getpid(), leg=os.environ.get("ODDE_SERVED_LEG")), default=str) + "\n")
        except Exception: pass  # noqa: BLE001

def install_on_runner(runner):
    """Install the requested levers on an already-built runner. Idempotent (a second call reports and returns). Returns STATE."""
    req = requested(); model = runner.model
    rec = {"event": "install", "requested": req, "levers": {}}
    # the kit's tuned tile cache is applied by the environment (report only; this module never installs it)
    rec["levers"]["kit.cueq_cache"] = os.environ.get("CUEQ_TRITON_CACHE_DIR") or False
    if req["addon_levers"]:
        try:
            import odde_addon
            already = getattr(odde_addon, "_ACTIVE", {})
            if already:
                rec["levers"]["ditfast"] = {"already_active": sorted(already)}
            else:
                info = odde_addon.install(model, req["addon_levers"])
                rec["levers"]["ditfast"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else "installed") for k, v in info.items()}
            _banner(f"ACTIVE DITFAST levers {req['addon_levers']} -> {rec['levers']['ditfast']}")
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            STATE["errors"]["ditfast"] = repr(e); rec["levers"]["ditfast"] = {"error": repr(e)}
            _banner(f"DITFAST install FAILED -> those levers stay off: {e!r}")
            if _flag("ODDE_SERVED_LEVERS_STRICT"): raise
    if req["arm"]:
        try:
            import odde_arm_t
            if odde_arm_t.COUNTS.get("installed"):
                rec["levers"]["odde_arm_t"] = {"already_installed": odde_arm_t.COUNTS.get("arm")}
            else:
                odde_arm_t.install(req["arm"], verbose=True, model=model)
                rec["levers"]["odde_arm_t"] = {"arm": odde_arm_t.COUNTS.get("arm"), "version": getattr(odde_arm_t, "__version__", "?"),
                                               "composition": odde_arm_t.COUNTS.get("composition"), "min_tokens": odde_arm_t.COUNTS.get("min_tokens")}
            _banner(f"ACTIVE odde_arm_t arm {rec['levers']['odde_arm_t']}")
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            STATE["errors"]["odde_arm_t"] = repr(e); rec["levers"]["odde_arm_t"] = {"error": repr(e)}
            _banner(f"odde_arm_t install FAILED -> stock trunk: {e!r}")
            if _flag("ODDE_SERVED_LEVERS_STRICT"): raise
    # OPENDDE_ACCEL_V2: Tier-2 sampler attention recast (numerics-changing; opt-in; never exact)
    try:
        import odde_accel_v2 as _A2
        da = _A2.install_dit_attn(model)
        if da.get("installed"): rec["levers"]["accel_v2.dit_attn(TIER-2)"] = {"mode": da.get("mode"), "n_modules": da.get("n_modules")}
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise
        STATE["errors"]["accel_v2.dit_attn"] = repr(e); rec["levers"]["accel_v2.dit_attn(TIER-2)"] = {"error": repr(e)}
        _banner(f"accel_v2 dit_attn install FAILED -> stock fp32 attention: {e!r}")
        if _flag("ODDE_SERVED_LEVERS_STRICT"): raise
    kr = os.environ.get("ODDE_SERVED_KEEP_RAW", "")
    if kr:      # test hook: copy the three raw dumped files per (sample, seed) out of the worker's scratch dir before it deletes them
        try:
            import glob as _glob, shutil as _sh
            dumper = runner.dumper; _orig_dump = dumper.dump
            def dump(*a, **k):
                r = _orig_dump(*a, **k)
                try:
                    pdb_id = k.get("pdb_id", a[1] if len(a) > 1 else None); seed = k.get("seed", a[2] if len(a) > 2 else None)
                    for f in _glob.glob(os.path.join(dumper.base_dir, "**", f"seed_{seed}", "predictions", "*"), recursive=True):
                        d = os.path.join(kr, str(pdb_id), f"seed_{seed}"); os.makedirs(d, exist_ok=True); _sh.copy(f, d)
                except Exception as e:  # noqa: BLE001
                    _banner(f"keep-raw copy failed: {e!r}")
                return r
            dumper.dump = dump; rec["levers"]["test.keep_raw"] = kr
        except Exception as e:  # noqa: BLE001
            rec["levers"]["test.keep_raw"] = {"error": repr(e)}
    STATE["installs"].append(rec); STATE["active"] = True
    _banner("SUMMARY " + json.dumps(rec["levers"], default=str), rec)
    return STATE

def _wrap_module(mod):
    if getattr(mod.get_default_runner, "_served_levers_wrapped", False): return
    orig = mod.get_default_runner
    def get_default_runner(*a, **k):
        if _flag("ODDE_SERVED_DETERMINISTIC"):
            k["deterministic"] = True; STATE["deterministic_forced"] = True
            if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
                _banner("WARNING: ODDE_SERVED_DETERMINISTIC=1 without CUBLAS_WORKSPACE_CONFIG exported before python; seed_everything sets it late")
            _banner("deterministic=True forced in get_default_runner (DET recipe)")
        runner = orig(*a, **k)
        if _flag("ODDE_SERVED_DETERMINISTIC"):
            # exactly what opendde.utils.seed.seed_everything(deterministic=True) sets per prediction -- applied once here so that a hosting process that
            # snapshots process-global torch state right after runner construction records the deterministic state.
            import torch
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            install_on_runner(runner)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            _banner(f"install_on_runner raised {e!r}")
            if _flag("ODDE_SERVED_LEVERS_STRICT"): raise
        return runner
    get_default_runner._served_levers_wrapped = True; get_default_runner.__wrapped__ = orig
    mod.get_default_runner = get_default_runner; STATE["wrapped"] = True

class _Hook(importlib.abc.MetaPathFinder):
    _busy = False
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET or _Hook._busy: return None
        _Hook._busy = True
        try:
            spec = None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"): continue
                spec = f.find_spec(name, path, target)
                if spec is not None: break
            if spec is None or spec.loader is None: return spec
            _orig_exec = spec.loader.exec_module
            def exec_module(module, _orig_exec=_orig_exec):
                _orig_exec(module)
                try: sys.meta_path.remove(self)
                except ValueError: pass
                _wrap_module(module)
            spec.loader.exec_module = exec_module
            return spec
        finally:
            _Hook._busy = False

def arm_hook():
    """Called from sitecustomize when ODDE_SERVED_LEVERS=1: wrap now if runner.batch_inference is already imported, else at its import."""
    if _TARGET in sys.modules: _wrap_module(sys.modules[_TARGET])
    elif not any(isinstance(f, _Hook) for f in sys.meta_path): sys.meta_path.insert(0, _Hook())
    _banner(f"v{__version__} armed (pid {os.getpid()}): requested {json.dumps(requested())}; levers install after get_default_runner() returns")

def stats():
    out = dict(STATE)
    try:
        import odde_arm_t; out["odde_arm_t"] = odde_arm_t.stats() if hasattr(odde_arm_t, "stats") else dict(odde_arm_t.COUNTS)
    except Exception: pass  # noqa: BLE001
    try:
        import odde_addon; out["ditfast"] = odde_addon.stats() if hasattr(odde_addon, "stats") else None
    except Exception: pass  # noqa: BLE001
    return out
