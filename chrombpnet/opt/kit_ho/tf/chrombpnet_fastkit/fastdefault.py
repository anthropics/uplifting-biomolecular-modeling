"""chrombpnet_fastkit.fastdefault — the route decision behind `fast` / `exact`: from the DETECTED GPU class (nvidia-smi, no framework import) and
the class / arch tables, resolve() decides the forward route (the K1 Triton forward | the stock graph in tf.function | the stock's own predict in
file order), the K1 conv precision, the native-dilation default, the tail form and the warm-up; r8_line() composes the kit's one status line from
that decision. No TensorFlow / torch import at module import (this module is also loaded by path, outside the package). No flag or environment
variable chooses a lever: the tables decide; the deterministic recipe's env (TF_DETERMINISTIC_OPS / TF_USE_DEFAULT_CONV_ALGO /
CHROMBPNET_DET_SUBPROCESS) selects the 'det' rows.
"""
import os
import hashlib as _hashlib, json, subprocess, importlib.util, importlib.metadata   # json: the arch-keyed table read (arch_tiles.json)
# the kit's one out-of-memory classifier (chrombpnet_fastkit/_oom.py), loaded as this file's sibling by path: this module is itself loaded by
# path (the package's `check`, the entry script's light probe), so no package context is assumed. A handler below that reroutes re-raises an OOM first.
_oom_spec = importlib.util.spec_from_file_location("chrombpnet_fastkit_oom", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_oom.py"))
_oom = importlib.util.module_from_spec(_oom_spec); _oom_spec.loader.exec_module(_oom); is_oom = _oom.is_oom

__all__ = ["detect_gpu", "resolve", "r8_line", "arch_tc_tile", "K1_CLASS_TABLE", "CC_DEFAULT_CLASS", "class_route", "route_mode", "NATIVE_DILATION_CLASS_TABLE", "NATIVE_DILATION_CLAUSE", "NATIVE_DILATION_BATCH_PIN", "TAIL_CLASS_TABLE", "NATIVE_DILATION_STOCK_ROUTE"]

# THE TAIL CLASS TABLE: the PADDED tail (every batch at the full batch shape, pad rows dropped) is the default on H100 / H200, where the tail rows come
# out bitwise equal to the stock's remainder batch; the STOCK tail (the stock's own remainder-batch shape) stays the default on B200 / A100 / L40S and
# on any unlisted class, said on the line. Why pad: an N < 64 job's remainder shape is a NEW kernel shape, and on the pinned stack a new shape costs a
# long driver PTX JIT that the shipped cache does not cover; the padded tail never touches it.
TAIL_CLASS_TABLE = {"H100": ("padded", "padded: tail rows bitwise equal to stock's on H100; no extra kernel shape for N<64"),
                    "H200": ("padded", "padded: tail rows bitwise equal to stock's on H200; no extra kernel shape for N<64"),
                    "B200": ("stock", "stock tail: stock's own remainder-batch shape on B200 (the padded tail is the H100 / H200 form) — an N<64 job pays the tail shape's JIT"),
                    "A100": ("stock", "stock tail: stock's own remainder-batch shape on A100 (the padded tail is the H100 / H200 form)"), "L40S": ("stock", "stock tail: stock's own remainder-batch shape on L40S (the padded tail is the H100 / H200 form)")}

# THE K1 CLASS TABLE: K1 where the torch stack imports AND the class is listed — H100, H200 (both modes), A100 (shipped numerics: the
# tensor-core convolutions; no bitwise Triton route under the deterministic recipe, so the package refuses `exact` by name there). L40S at shipped numerics and B200 (torch cu124 has no sm_100
# image) -> the TF forward with the reason on the line. The conv arithmetic follows the mode on every K1 class: tensor-core TF32 dots at shipped
# numerics (stock's own precision class), fp32 FFMA chains in stock's order under the deterministic recipe (bitwise) — resolve()["precision"].
K1_CLASS_TABLE = {"H100": "default (bitwise equal to stock under the deterministic recipe; faster than stock's own default)",
                  "H200": "default (bitwise equal to stock under the deterministic recipe; faster than stock's own default)",
                  "L40S": {"k1": False, "k1_by_mode": {"prod": False, "det": True},   # THE ROUTE IS PER (class, MODE)
                           "short": "per mode: shipped numerics -> stock's own graph (faster on this card; files bitwise equal to stock's default run); deterministic recipe -> K1 (bitwise equal to the L40S stock's own deterministic run)",
                           "basis_prod": "shipped numerics -> stock's own graph: files bitwise equal to stock's default run on every output; K1's output equals stock's deterministic run, not its default run, so K1 is not the default here",
                           "basis_det": "deterministic recipe -> K1: bitwise equal to the L40S stock's own deterministic run on every output file (h5, bigWig, metrics, preds.bed)"},
                  "A100": {"k1": True, "k1_by_mode": {"prod": True, "det": False},
                           "short": "per mode: shipped numerics -> K1 with tensor-core convolutions (about x4 stock's own forward on this card); deterministic recipe -> no bitwise Triton route (the sm_80 counts-head summation order varies with the rows per call): the package refuses `exact` by name on this class",
                           "basis_prod": "shipped numerics -> K1, tensor-core (TF32) convolutions: stock's own precision class, about x4 stock's TF32 cuDNN forward per region at 1,024 regions per call",
                           "basis_det": "deterministic recipe -> no K1: the fp32 kernels match the A100 stock's profile logits bit for bit at every batch shape but its log-counts only at some (the sm_80 counts-head summation order changes with the rows per call), so this class has no bitwise Triton route and the package refuses `exact` by name (--mode off --det 1 is the deterministic run there)"},
                  "B200": "no K1: torch cu124 carries no sm_100 image; the TensorFlow route runs"}
def _row_k1(v, mode):
    """A class table row -> is K1 the default under MODE? string rows: 'default …' = both modes; dict rows: k1_by_mode[mode] (else 'k1')."""
    if isinstance(v, str): return v.startswith("default")
    if isinstance(v, dict):
        bm = v.get("k1_by_mode")
        return bool(bm.get(mode)) if isinstance(bm, dict) else bool(v.get("k1"))
    return False
def _row_text(v, mode):
    if isinstance(v, str): return v
    if isinstance(v, dict): return v.get("basis_" + mode) or v.get("short") or v.get("why") or str(v)
    return str(v)
def class_route(cls, cap, mode="prod"):
    """The route for a class with no arch-keyed entry -> (k1_default, basis): the class table's row when the class has one; a class in neither table
    with a CUDA device present engages K1 (a cold JIT, the tile chosen by the device's shared memory — said on the line; not measured on that card);
    no device class detected (nvidia-smi unavailable) -> K1 cannot run: the TensorFlow route."""
    row = K1_CLASS_TABLE.get(cls)
    if row is not None: return _row_k1(row, mode), _row_text(row, mode)
    if cap: return True, f"class {cls!r} (cc {cap}) is in neither arch_tiles.json nor the class table: K1 engages with a cold JIT and the tile chosen by the device's shared memory — not measured on this card"
    return False, "no CUDA device class detected (nvidia-smi unavailable): K1 cannot run here — the TensorFlow route"
# THE NATIVE-DILATION CLASS TABLE:
# ON where the native-dilation graph is bitwise equal to stock's default run (H100, H200, B200: fp32 convolutions); OFF on A100 / L40S (stock's
# default there is TF32 and the graph is not bitwise equal to it); OFF on the TensorFlow 2.15 / cuDNN 8.9 stack (not bitwise equal to stock's default there).
NATIVE_DILATION_CLASS_TABLE = {"H100": True, "H200": True, "B200": True, "A100": False, "L40S": False}   # ON by default only where bitwise equal to stock's DEFAULT run (the fp32-conv classes); off on the stock-route classes A100 / L40S (their clause below says what native dilation gives there)
NATIVE_DILATION_STOCK_ROUTE = {"A100", "L40S"}   # the classes whose TensorFlow route keeps stock's own algorithms: native dilation stays off there (their stock default is TF32; the graph is not bitwise to it with native dilation)
# The native-dilation clause per class: printed on the kit's line beside the lever
NATIVE_DILATION_CLAUSE = {
    "H100": "bitwise equal to stock's default at every batch size (fp32 convs)", "H200": "bitwise equal to stock's default at every batch size", "B200": "bitwise equal to stock's default at every batch size",
    "A100": "bitwise equal to stock's fp32 mode at every batch size; within stock's own run-to-run spread vs its TF32 default; batch pinned <= 256 here (throughput falls past B=256 on this class)",
    "L40S": "bitwise equal to stock's fp32 mode at every batch size; equal to its default at B=16 only (per-process autotune pick), within its spread at the other batch sizes"}
NATIVE_DILATION_BATCH_PIN = {"A100": 256}   # the TF route's batch ceiling by class (the reason in the clause)
_REASON_OFF = {"A100": "A100 sm80: stock's default there is TF32, native dilation is not bitwise equal to it", "L40S": "L40S sm89: stock's default there is TF32, native dilation is not bitwise equal to it"}

_NVSMI = {}   # ONE nvidia-smi subprocess per process (the class probe and the cache loader's identity read share it)
def nvsmi_query():
    """{name, driver, cc} from ONE nvidia-smi call, memoized for the process; {'error': '<why>'} when nvidia-smi is unavailable."""
    if "q" in _NVSMI: return _NVSMI["q"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        parts = [x.strip() for x in out.split(",")]; _NVSMI["q"] = {"name": parts[0], "driver": parts[1] if len(parts) > 1 else None, "cc": parts[2] if len(parts) > 2 else None}
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; the unknown-class route below is for every other failure
        _NVSMI["q"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return _NVSMI["q"]
_CLASS_CC = {"H100": "9.0", "H200": "9.0", "A100": "8.0", "L40S": "8.9", "B200": "10.0"}
CC_DEFAULT_CLASS = {"9.0": "H100", "8.0": "A100", "8.9": "L40S", "10.0": "B200"}   # compute capability -> the class whose rows are that cc's DEFAULT (a card of the cc not named in the tables — H800 / GH200 on 9.0, A800 / A30 on 8.0 — is served these rows; the named classes stay the measured overrides: H200 differs from H100 only where its own tables say so)
def detect_gpu():
    """(class, name, compute_cap) from nvidia-smi — no framework import. class in {H100,H200,A100,L40S,B200,unknown}; a card not named in the
        tables is served the default class of its compute capability (CC_DEFAULT_CLASS), said in the name field.
        CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE=<class> is a TEST HOOK (the CPU tests' stand-in class; stated in the name field, never a user knob)."""
    _ov = os.environ.get("CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE", "").strip()
    if _ov: return _ov, f"class override {_ov} (CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE: a test hook, stated)", _CLASS_CC.get(_ov)
    try:
        q = nvsmi_query()
        if "error" in q: raise RuntimeError(q["error"])
        name, cap = q["name"], q["cc"]
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; the unknown-class route below is for every other failure
        return "unknown", f"nvidia-smi unavailable ({type(e).__name__})", None
    cls = next((c for c in ("H200", "H100", "A100", "L40S", "B200") if c in name), None)   # a card NAMED in the class tables: its own measured rows
    if cls is None:   # any other card is served the DEFAULT row of its compute capability (the broad key: the arch the kernels and numerics are keyed on), said in the name field — never left without a row for want of an exact name
        cls = CC_DEFAULT_CLASS.get(str(cap), "unknown")
        if cls != "unknown": name = f"{name} (class {cls} by cc {cap}: the cc's default row; the card is not in the named tables)"
    return cls, name, cap

IMAGE_TORCH_TREES = ("/opt/torch",)   # the pinned stack installs torch under /opt/torch, NOT on the default sys.path: tried when torch does not import from the default path

K1_INSTALL_LINE = "the torch stack (torch 2.4.1+cu124 / triton 3.0.0; under /opt/torch on the coexistence image, found by the kit; elsewhere: pip install torch==2.4.1 triton==3.0.0) — bpnet-lite 1.0.0 + tangermeme 1.4.1 ship VENDORED under torch/vendor/ (MIT), nothing else to install"

def swap_typing_extensions(vendor_dir=None, kit_root=None):
    """The ONE typing_extensions guard is the K1 package's _te_guard (its vendored typing_extensions is executed and installed HERE-first when the loaded
        or resolvable copy lacks the names torch 2.4.1 needs — e.g. an older copy already imported by TensorFlow). Loaded BY PATH (a leaf module: no
        package import, no torch) so it runs before any torch import; the package's own call at its import is idempotent. `vendor_dir` is unused (kept
        for the call sites). Returns the guard's record (action in kept/replaced/provided + detail) or a swap_failed record."""
    import sys, importlib.util
    _root = kit_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(_root, "torch", "chrombpnet_k1", "_te_guard.py")
    try:
        if not os.path.isfile(path): return {"swap_failed": "the K1 package's _te_guard.py is missing at %s" % path, "by": "chrombpnet_k1._te_guard (by path)"}
        spec = importlib.util.spec_from_file_location("_kit_te_guard", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        rec = dict(mod.ensure_typing_extensions()); rec["by"] = "chrombpnet_k1._te_guard (loaded by path)"; return rec
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; every other failure returns the swap_failed record (the K1 import may then fail and the mode refuse by name)
        return {"swap_failed": repr(e)[:160], "by": "chrombpnet_k1._te_guard (by path)"}

def _torch_stack(kit_root=None):
    """Is the K1 stack (torch + triton + bpnetlite, which imports tangermeme) importable? The kit's OWN vendored bpnet-lite / tangermeme
        (<kit>/torch/vendor/) go on sys.path HERE-first, so the K1 route needs nothing but torch/triton from the environment; if torch is not on the
        default sys.path the known torch tree (IMAGE_TORCH_TREES) is tried (said on the status line); the stack is then really imported. Absent or
        failing -> (False, details): resolve() reports forward='tf_function' with the reason + the install line and `k1_wanted` stays True, so the
        entry script refuses the mode by name (it never runs `fast` with K1 dropped). Returns (ok, details)."""
    import sys
    K1_PREREQ = ("torch", "triton", "bpnetlite", "tangermeme")   # bpnetlite imports tangermeme.predict at module import (bpnet-lite 1.0.0)
    def _missing(): return [m for m in K1_PREREQ if importlib.util.find_spec(m) is None]
    def _swap_typing_extensions(vendor_dir): return swap_typing_extensions(vendor_dir, kit_root=_root)   # the K1 package's guard, by path
    def _real_import():
        """The stack is IMPORTED, not find_spec'd — the failure text is the reason on the line; None when it imports."""
        try:
            import torch, triton, bpnetlite  # noqa
            return None
        except BaseException as e:
            if is_oom(e): raise                                       # an out-of-memory error propagates; every other failure is returned as the reason (K1 OFF; the mode then refuses by name)
            return "%s: %s" % (type(e).__name__, str(e)[:160])
    def _ok(): return not _missing()
    _root = kit_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vendor = os.path.join(_root, "torch", "vendor"); vendored = {}
    if os.path.isdir(vendor):
        if vendor not in sys.path: sys.path.insert(0, vendor); importlib.invalidate_caches()   # HERE-first: the kit's bytes win over any installed copy
        for d in sorted(os.listdir(vendor)):
            if d.endswith(".dist-info") and "-" in d: vendored[d.split("-")[0]] = d[:-len(".dist-info")].split("-", 1)[1]
    source = "default sys.path"; ok = _ok()
    if not ok:
        for tree in IMAGE_TORCH_TREES:
            if os.path.isdir(tree) and tree not in sys.path:
                sys.path.insert(0, tree); importlib.invalidate_caches()
                if _ok(): source = f"{tree} (image prerequisite; not on the default sys.path)"; ok = True; break
                sys.path.remove(tree)
    swapped = _swap_typing_extensions(vendor) if ok else None
    err = _real_import() if ok else None
    if err: ok = False
    v = {"source": source if ok else None, "missing": _missing(), "vendor_dir": vendor if os.path.isdir(vendor) else None, "vendored": vendored, "install_line": K1_INSTALL_LINE, "import_error": err, "typing_extensions_swapped": swapped}
    for m in ("torch", "triton"):
        try: v[m] = importlib.metadata.version(m)
        except Exception: v[m] = None
    return ok, v

def _tf_version():
    try: return importlib.metadata.version("tensorflow")
    except Exception: return None

CLASS_ARCH = {"H100": "cuda-90", "H200": "cuda-90", "A100": "cuda-80", "L40S": "cuda-89", "B200": "cuda-100"}   # the class -> arch key of arch_tiles.json (cuda-<major><minor>)

def arch_key(cap, cls=None):
    """'9.0' -> 'cuda-90' from the detected compute capability; the class map as the fallback."""
    try:
        if cap: m, n = str(cap).strip().split("."); return f"cuda-{int(m)}{int(n)}"
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; the class-map route below is for every other failure
    return CLASS_ARCH.get(cls or "", None)

def route_mode():
    """'det' when the deterministic recipe's env is set (TF_DETERMINISTIC_OPS=1 / TF_USE_DEFAULT_CONV_ALGO=1 / CHROMBPNET_DET_SUBPROCESS=1), else 'prod' — the same words as chrombpnet_k1's recipe_mode()."""
    return "det" if any(os.environ.get(k) == "1" for k in ("TF_DETERMINISTIC_OPS", "TF_USE_DEFAULT_CONV_ALGO", "CHROMBPNET_DET_SUBPROCESS")) else "prod"
def arch_tc_tile(kit_root, arch):
    """The tensor-core conv tile listed for `arch` (arch_tiles.json entries[arch].tc_tile); None when the arch has none (kernels_tc's own default then)."""
    if not kit_root or not arch: return None
    try:
        with open(os.path.join(kit_root, "torch", "chrombpnet_k1", "arch_tiles.json")) as f: return (json.load(f).get("entries", {}).get(arch) or {}).get("tc_tile")
    except (OSError, ValueError, TypeError): return None
def arch_default_route(kit_root, arch, mode=None):
    """THE ROUTE TABLE: the K1 package's arch-keyed default_route, read from the kit's OWN vendored bytes (torch/chrombpnet_k1/arch_tiles.json, HERE-first) —
        JSON only, no torch at import. Returns (route 'k1'|'stock', basis, source) or None when the file or the arch's entry is absent (class_route() then decides)."""
    if not kit_root or not arch: return None
    p = os.path.join(kit_root, "torch", "chrombpnet_k1", "arch_tiles.json")
    try:
        m = json.load(open(p)); e = (m.get("entries") or {}).get(arch)
        if not e: return None                                      # an unlisted arch: class_route() decides (a listed class's row, else K1 with a cold JIT)
        r = e.get("default_route")
        if not r: return None
        mode = mode or route_mode()
        if "route" not in r:   # the per-mode dict {'prod': {route, basis}, 'det': {route, basis}}
            r = r.get(mode) or {"route": "stock", "basis": f"no {mode} row for {arch}: the stock path by default"}
        return (r.get("route"), f"[{arch}/{mode}] {r.get('basis')}", "arch_tiles.json entries[%s].default_route[%s]" % (arch, mode))
    except Exception as ex:
        if is_oom(ex): raise                                          # an out-of-memory error propagates; the class-table route (None) below is for every other failure
        return None

def resolve(kit_root=None):
    """Decide the route from the detected class and the tables. Returns a dict: gpu_class, gpu_name, compute_cap, mode ('prod' | 'det'), forward ('k1' |
        'tf_function' | 'keras_predict_fileorder'), forward_reason, precision ('tf32' | 'ieee' on the K1 route, else None), tc_tile, k1_wanted, k1_cache_pins,
        native_dilation (bool) + native_dilation_reason, batch_pin, warmup (bool), tail + tail_reason, skipped (list of 'lever: reason'), torch_stack, tensorflow,
        kit_version, fixed_cost_s."""
    import time as _time; _t0 = _time.time(); cls, name, cap = detect_gpu(); _t_det = _time.time() - _t0; tfv = _tf_version() or ""
    skipped = []
    _mode = route_mode()   # the route is per (class, mode)
    _ar = arch_default_route(kit_root, arch_key(cap, cls), _mode); _table = "the class table (fallback: no arch-keyed entry)"
    if _ar is not None:
        _route, _basis, _table = _ar
        _k1_default = (_route == "k1")   # 'tf' / 'stock' = the stock-route branch (the stock's own predict() in prod; the tf.function form under the DET env)
    else:
        _k1_default, _basis = class_route(cls, cap, _mode); _table = "the class table (no arch-keyed entry)"
    # the K1 stack (torch + triton + bpnetlite) is imported/probed ONLY when the decision needs it — a K1-default class;
    # a table-decided TF route never imports torch (seconds of pure startup, buying nothing)
    _pins = k1_cache_pins(kit_root, arch_key(cap, cls)) if (kit_root and _k1_default) else None   # the cache-pins precondition (bytes only, no torch)
    _need_probe = _k1_default
    _t1 = _time.time()
    if _need_probe: torch_ok, tv = _torch_stack(kit_root)
    else: torch_ok, tv = False, {"probe": "skipped: the TF route by decision (v0.12.66: no torch import on a table-decided TF route)", "skipped": True}
    _held = bool(_pins) and not _pins["ok"] and torch_ok and _k1_default   # a pin mismatch matters only where the K1 stack IS present (a missing stack keeps its own OFF line + install hint)
    _t_probe = _time.time() - _t1
    if _held and kit_root and os.path.isdir(os.path.join(kit_root, "torch", "chrombpnet_k1")):   # a cache-pin miss is a missing shipped cache, not a missing lever — the same K1 forward runs, its kernels compiling in this process (a cold Triton JIT, seconds), said on ONE line
        fwd = "k1"
        why = f"[{cls}/{_mode}] K1 cold JIT: cache pins do not match this stack ({_pins['note']}; {_pins['cache_pin']} vs {_pins['kernels_sha256_16']}) — the kernels compile in this process; route [{_table}]: {_basis}; torch {tv.get('torch')} / triton {tv.get('triton')} present"
    else:
        if torch_ok and _k1_default and kit_root and os.path.isdir(os.path.join(kit_root, "torch", "chrombpnet_k1")):
            fwd, why = "k1", f"[{cls}/{_mode}] route [{_table}]: {_basis}; torch {tv.get('torch')} / triton {tv.get('triton')} present"
        else:
            fwd = "tf_function"
            if _k1_default and not torch_ok:
                _miss = ", ".join(tv.get("missing") or ["torch/triton"])
                if tv.get("import_error"): why = f"[{cls}/{_mode}] K1 route: OFF — the K1 stack failed to import ({tv['import_error']}); TF route"
                else: why = f"[{cls}/{_mode}] K1 route: OFF — {_miss} absent; TF route"
                skipped.append((f"K1 route: OFF — {_miss} absent; install: {K1_INSTALL_LINE}") if _miss else (f"K1 route: OFF — the K1 stack failed to import ({tv.get('import_error')}); install: {K1_INSTALL_LINE}"))
            elif not _k1_default:
                _det_env = os.environ.get("CHROMBPNET_DET_SUBPROCESS") == "1" or os.environ.get("TF_DETERMINISTIC_OPS") == "1"
                if _det_env: why = f"[{cls}/{_mode}] route [{_table}]: {cls} -> stock's graph; the deterministic recipe is set -> the tf.function form (bitwise equal to stock's deterministic run on this class): {_basis}"
                else: fwd = "keras_predict_fileorder"; why = f"[{cls}/{_mode}] route [{_table}]: {cls} -> stock's graph = stock's own predict() call over the units in input order (bitwise equal to stock's default by construction): {_basis}"
                skipped.append(f"K1: not the default on this class: {str(_basis)[:120]}"); why += " (no torch import on the TF route)"
            else: why = f"[{cls}/{_mode}] K1 route: OFF — this tree carries no torch/chrombpnet_k1 package (K1 cannot run); TF route"; skipped.append("K1 route: OFF — the tree carries no torch/chrombpnet_k1 package")
    # native dilation: device-keyed, not a user knob
    _route_table = _table
    if fwd == "k1": nd, nd_why = False, "n/a (the K1 forward has no dilation knob)"
    elif tfv.startswith("2.15"): nd, nd_why = False, "OFF: on the TensorFlow 2.15 / cuDNN 8.9 stack the native-dilation graph is not bitwise equal to stock's default run (measured there; it is on the pinned 2.8 stack) — stock's own dilation loop runs"; skipped.append("native_dilation: off on the TensorFlow 2.15 stack (not bitwise equal to stock's default run there)")
    elif cls in NATIVE_DILATION_CLASS_TABLE and NATIVE_DILATION_CLASS_TABLE[cls]: nd, nd_why = True, f"ON: {NATIVE_DILATION_CLAUSE.get(cls, cls + ' bitwise equal to stock')}"
    elif cls in NATIVE_DILATION_STOCK_ROUTE: nd, nd_why = False, f"OFF (default on a stock-route class: stock's own predict() with stock's own algorithms, bitwise equal to stock's default; native dilation on this class: {NATIVE_DILATION_CLAUSE.get(cls, cls)[:90]}…)"; skipped.append(f"native_dilation: off on {cls} (stock's default there is TF32; the kit keeps stock's own algorithms)")
    elif cls in _REASON_OFF: nd, nd_why = False, "OFF: " + _REASON_OFF[cls]; skipped.append("native_dilation: " + _REASON_OFF[cls])
    else: nd, nd_why = True, f"ON: class {cls!r} is not in the native-dilation table — engaged on its mechanism (cuDNN's dilated convolution in place of stock's space_to_batch form); bit-identity to stock's default is established on H100 / H200 only, not on this card"
    # NATIVE DILATION OFF UNDER THE DET RECIPE, by detection — cuDNN's default algorithm for a DILATED convolution (TF_USE_DEFAULT_CONV_ALGO=1) is the slow
    # path and its kernels are not in the shipped driver cache; the stock's own space_to_batch form is bitwise equal to the stock's deterministic run by
    # construction; the class default is overridden LOUDLY here (said on the line).
    _det_env = [k for k in ("TF_USE_DEFAULT_CONV_ALGO", "TF_DETERMINISTIC_OPS", "CHROMBPNET_DET_SUBPROCESS") if os.environ.get(k) == "1"]
    if nd and _det_env:
        _was = "requested ON (the class default) overridden"
        nd = False; nd_why = f"OFF under the DET recipe ({'/'.join(_det_env)}=1 present: cuDNN's default algorithm for a dilated conv is the slow path and its kernels are not in the shipped cache; stock's own space_to_batch form is bitwise equal to stock's deterministic run by construction); {_was}"
        skipped.append("native_dilation: OFF under the DET recipe")
    wu = True                                                       # the load-time warm-up (the shapes are fixed by the model: [batch, tail])
    tail, tail_why = TAIL_CLASS_TABLE.get(cls, ("stock", f"stock tail: stock's own remainder-batch shape on class {cls!r} (a padded last batch can change the convolution algorithm TensorFlow picks and with it the low bits; the padded form is the H100 / H200 form, bitwise equal to stock's there)"))
    if torch_ok and fwd == "k1": why = f"K1 route: ON — torch {tv.get('torch')} from {tv.get('source')}; " + (f"bpnet-lite {tv['vendored'].get('bpnet_lite')} + tangermeme {tv['vendored'].get('tangermeme')} vendored HERE-first ({tv['vendor_dir']}, MIT)" if tv.get("vendored") else "bpnet-lite/tangermeme from the environment") + f" | {why}"
    elif torch_ok and tv.get("source") and tv["source"] != "default sys.path": skipped.append(f"K1: torch stack from {tv['source']}")
    _pin = (NATIVE_DILATION_BATCH_PIN.get(cls) if (nd and fwd != "k1") else None)
    _prec = ("tf32" if _mode == "prod" else "ieee") if fwd == "k1" else None   # the K1 conv arithmetic follows the mode: tensor-core TF32 at shipped numerics (stock's own precision class), fp32 FFMA chains in stock's order under the recipe (bitwise)
    if fwd == "k1": why += ("; convs tf32 on the tensor cores (stock's shipped precision class)" if _prec == "tf32" else "; convs fp32 in stock's deterministic order (bitwise)")
    return {"gpu_class": cls, "gpu_name": name, "compute_cap": cap, "mode": _mode, "forward": fwd, "forward_reason": why, "precision": _prec, "tc_tile": arch_tc_tile(kit_root, arch_key(cap, cls)) if _prec == "tf32" else None, "k1_wanted": bool(_k1_default), "k1_cache_pins": _pins, "native_dilation": nd, "native_dilation_reason": nd_why, "batch_pin": _pin,
            "warmup": wu, "tail": tail, "tail_reason": tail_why, "skipped": skipped, "torch_stack": {"present": torch_ok, **tv}, "tensorflow": tfv or None, "kit_version": KIT_LINE_VERSION[1:], "fixed_cost_s": {"class_detection": round(_t_det, 3), "k1_stack_import_and_probe": round(_t_probe, 3)}}

def _te_note(res):
    s = ((res or {}).get("torch_stack") or {}).get("typing_extensions_swapped")
    if isinstance(s, dict) and s.get("action") in ("replaced", "provided"): return " | typing_extensions guard (chrombpnet_k1._te_guard): %s — %s" % (s.get("action"), str(s.get("detail"))[:160])
    return ""

def k1_cache_pins(kit_root, arch):
    """The cache-pins precondition: the shipped Triton cache for THIS arch (torch/triton_cache_of_record.json: the first entry whose arch == '<arch>-32'; the
        package's own loader may prefer a device-matched entry — the arch-only read here is the floor) must be pinned to the vendored package's OWN kernel bytes
        (JIT_IDENTITY.json kernels_sha256_16 == sha256(chrombpnet_k1/kernels.py)[:16]); else the package's apply() refuses the shipped cache and every fresh
        process pays a cold Triton JIT (seconds). Read from the kit's bytes, no torch; never raises."""
    out = {"arch": arch, "ok": False, "cache_dir": None, "cache_pin": None, "kernels_sha256_16": None, "note": ""}
    try:
        kp = os.path.join(kit_root or "", "torch", "chrombpnet_k1", "kernels.py")
        if not os.path.isfile(kp): out["note"] = "no vendored K1 package (torch/chrombpnet_k1/kernels.py absent)"; return out
        out["kernels_sha256_16"] = _hashlib.sha256(open(kp, "rb").read()).hexdigest()[:16]
        lp = os.path.join(kit_root, "torch", "triton_cache_of_record.json")
        if not os.path.isfile(lp): out["note"] = "no shipped cache list (torch/triton_cache_of_record.json absent): a cold JIT per fresh process"; return out
        ents = [e for e in (json.load(open(lp)).get("entries") or []) if e.get("arch") == f"{arch}-32"]
        if not ents: out["note"] = f"no shipped cache for {arch} in the cache-of-record list: a cold JIT per fresh process"; return out
        e = ents[0]; d = os.path.join(kit_root, "torch", e.get("dir", "")); out["cache_dir"] = e.get("dir")
        ip = os.path.join(d, "JIT_IDENTITY.json")
        if not os.path.isfile(ip): out["note"] = f"the shipped cache dir {e.get('dir')} carries no JIT_IDENTITY.json: the package's apply() refuses it"; return out
        out["cache_pin"] = (json.load(open(ip)) or {}).get("kernels_sha256_16")
        if out["cache_pin"] == out["kernels_sha256_16"]: out["ok"] = True; out["note"] = f"the shipped Triton cache {e.get('dir')} is pinned to the vendored kernels ({out['kernels_sha256_16']}): the package's loader applies it (HIT expected on a fresh process)"
        else: out["note"] = f"the shipped Triton cache {e.get('dir')} is pinned to other kernel bytes ({out['cache_pin']} vs the vendored kernels {out['kernels_sha256_16']}): the package's apply() refuses it -> a cold Triton JIT per fresh process (~5-7 s)"
    except Exception as ex:
        if is_oom(ex): raise                                          # an out-of-memory error propagates; the mismatch (-> cold JIT) route below is for every other failure
        out["note"] = f"pins unreadable ({type(ex).__name__}: {str(ex)[:80]}): treated as a mismatch"
    return out

def k1_first_process_terms(bias_h5, nobias_h5):
    """The K1 route's FIRST-PROCESS terms, timed as idempotent pre-warms BEFORE the package's apply() (the cost is moved into named terms, never
        removed; every output byte identical): cuda_context_init (torch.cuda.init + a 1-element tensor + synchronize = the CUDA context, the libcuda/libtorch_cuda
        page-in, the fatbin registration), weights_pagein (the two h5 files read once through the page cache), triton_key (Triton's package hash; lru-cached, so
        apply's own call is then free), backend_hash (the ptxas --version subprocess — a datum, not pre-payable). 'n/a (<reason>)' where a term cannot run; never a crash."""
    import time as _t
    out = {}
    try:
        t = _t.time(); import torch; torch.cuda.init(); _x = torch.zeros(1, device="cuda"); torch.cuda.synchronize(); out["cuda_context_init"] = round(_t.time() - t, 3)
    except Exception as e: out["cuda_context_init"] = f"n/a ({type(e).__name__}: {str(e)[:60]})"
    try:
        t = _t.time(); nb = 0
        for h in (bias_h5, nobias_h5):
            if h and os.path.isfile(h):
                with open(h, "rb") as fh:
                    while True:
                        b = fh.read(1 << 24)
                        if not b: break
                        nb += len(b)
        out["weights_pagein"] = round(_t.time() - t, 3) if nb else "n/a (no weight files given)"; out["weights_bytes"] = nb
    except Exception as e: out["weights_pagein"] = f"n/a ({type(e).__name__})"
    try:
        t = _t.time(); from triton.compiler.compiler import triton_key; triton_key(); out["triton_key"] = round(_t.time() - t, 3)
    except Exception as e: out["triton_key"] = f"n/a ({type(e).__name__}: {str(e)[:60]})"
    try:
        t = _t.time(); import triton
        from triton.runtime.driver import driver as _drv
        _target = _drv.active.get_current_target(); from triton.backends import backends as _bk
        _be = _bk["nvidia"].compiler(_target); _be.hash(); out["backend_hash"] = round(_t.time() - t, 3)
    except Exception as e: out["backend_hash"] = f"n/a ({type(e).__name__}: {str(e)[:60]})"
    return out

KIT_LINE_VERSION = "v0.12.73"   # the version tag on the kit's status line ("[chrombpnet_fastkit <version>] gpu=<class> cc=<cap> | forward=<route> (<reason>) | …"); resolve() returns it as kit_version


def r8_line(res, caches, warm):
    """The kit's one status line: GPU class and cc, the forward route and why, native dilation, tail form, caches, fixed costs, warm-up shapes and seconds, levers skipped and why."""
    c = " ".join(f"{k} {v}" for k, v in (caches or {}).items()) or "none"
    w = (f"shapes {warm.get('shapes')} x{warm.get('inputlen')} in {float(warm.get('s') or 0.0):.2f} s (per shape {warm.get('per_shape_s')}); deferred: {warm.get('deferred', 'none')}" if warm else "off")
    if res.get("forward") == "k1" and warm: w += f" | triton first-call: process {float(warm.get('s') or 0.0):.2f} s (the shipped K1 cache's first call in THIS process: a compile on this stack >= 2.5 s per kernel; a group hit ~0.01 s per kernel — the two-process census form, 2026-08-28)"
    _pt = res.get("process_terms_s") or {}
    _pts = (" | process terms (the user's own work; the stock pays the same): " + " ".join((f"{k} {v:.3f} s" if isinstance(v, (int, float)) else f"{k} {v}") for k, v in _pt.items())) if _pt else ""
    _k1t = res.get("k1_first_process_terms_s") or {}
    if _k1t: _pts += " | K1 first-process terms (pre-warms timed before apply; cost moved, not removed): " + " ".join((f"{k} {v:.3f} s" if isinstance(v, (int, float)) else f"{k} {v}") for k, v in _k1t.items() if k != "weights_bytes")
    fx = res.get("fixed_cost_s") or {}
    _num = {k: v for k, v in fx.items() if isinstance(v, (int, float))}
    fixed = (" ".join(f"{k} {v:.3f} s" for k, v in _num.items()) + f" = {sum(_num.values()):.3f} s total (kit-side stages beyond the user's work)") if _num else "n/a"
    return (f"[chrombpnet_fastkit v{res['kit_version']}] gpu={res['gpu_class']} cc={res.get('compute_cap')} | forward={res['forward']} ({res['forward_reason']}) | "
            f"native_dilation={'ON' if res['native_dilation'] else 'off'} ({res['native_dilation_reason']}) | tail={res.get('tail')} ({res.get('tail_reason')}){_te_note(res)} | caches: {c} | fixed cost paid at apply: {fixed}{_pts} | warm-up: {w} | skipped: {'; '.join(res['skipped']) or 'none'} | residue: none (no process-global numerics flag set; the native-dilation class patch is restored after the job)")
