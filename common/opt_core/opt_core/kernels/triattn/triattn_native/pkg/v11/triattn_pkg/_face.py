"""The package face: loads the router (dispatch/candidate.py, verbatim) by path, resolves the CUDA extensions (prebuilt for this stack, else
JIT, else a typed refusal), and forwards every call to `best`.

Layout (this directory): dispatch/ (router `best` + k13), triton/ (`tri`: k10 / k11 / k12), cuda/ (triattn_cuda: sm_90a wgmma kernel), cuda_c/
(triattn_mw, sm_90a), cuda_b/ (triattn_m1, sm_90a), cuda_80/ (triattn_sm80: the sm_80 member) -- each kernel directory's tree at its PINS commit,
unmodified; the router puts those directories on sys.path itself and imports the modules under their own top-level names (`triattn`, `kernels`,
`pins`, `candidate_tri`, `triattn_cuda`, `triattn_mw`, `triattn_m1`, `triattn_sm80`).  prebuilt/<stack tag>/<extension>.so + .json (build record) +
prebuilt/INDEX.json.  Which CUDA extension a process needs follows from the device: cc 9.0 routes reach cuda / cuda_c / cuda_b (sm_90a code),
cc 8.0 routes reach cuda_80 (sm_80 code); the router never sends a device to an extension built for another architecture.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import sys
import sysconfig
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_COMMIT = "29f5831ac67"                      # dispatch/candidate.py `best` of this generation (the cc-keyed router with the Gluon refusal)
PREBUILT_DIR = os.path.join(HERE, "prebuilt")

# CUDA kernel directories: router name -> (dir, python module, extension module name = <name>.so = its PyInit symbol, where the module keeps the
# loaded extension, architecture).  The modules' own _build() returns the object found in that slot instead of compiling.
CUDA_SEATS = {
    "cuda":    {"dir": "cuda",    "module": "triattn_cuda", "ext": "triattn_sm90_ext",    "slot": "_EXT",       "arch": "sm_90a", "needs_cutlass": True},
    "cuda_c":  {"dir": "cuda_c",  "module": "triattn_mw",   "ext": "triattn_mw_ext_g3x4", "slot": "_EXTS[3x4]", "arch": "sm_90a", "needs_cutlass": False},   # DEFAULT_GEOM 3x4 (MW_GEOM unset)
    "cuda_b":  {"dir": "cuda_b",  "module": "triattn_m1",   "ext": "triattn_m1_ext",      "slot": "_EXT",       "arch": "sm_90a", "needs_cutlass": True},    # flags 0 + its SAFE partner (TRIATTN_M1_FLAGS unset); assembled with ptxas 13.4 where the builder had it (build record "ptxas")
    "cuda_80": {"dir": "cuda_80", "module": "triattn_sm80", "ext": "triattn_sm80_ext",    "slot": "_EXT",       "arch": "sm_80",  "needs_cutlass": False},   # the sm_80 member: mma.sync / ldmatrix / cp.async; nvcc + ninja build it, no third-party headers
}

_LOCK = threading.Lock()
_ROUTER = None
_SEAT_STATE = {}                                  # router name -> {"binary": "prebuilt:<path>" | "jit" , "record": {...} | None} once resolved


def stack_tag() -> str:
    """torch<version>-<CPython SOABI>: the key of a prebuilt directory (a torch C++/CUDA extension is specific to exactly this)."""
    import torch
    return "torch%s-%s" % (torch.__version__, sysconfig.get_config_var("SOABI"))


def _router():
    global _ROUTER
    if _ROUTER is None:
        with _LOCK:
            if _ROUTER is None:
                name = "triattn_pkg_router"
                spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "dispatch", "candidate.py"))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)          # puts dispatch/, triton/, cuda/, cuda_c/, cuda_b/ of THIS directory on sys.path (its own HERE / NATIVE)
                _ROUTER = mod
    return _ROUTER


def __getattr__(name):                             # `Unsupported` = the router's class (triton/triattn/errors.py, re-exported by k11): resolved with the router
    if name == "Unsupported":
        return _router().Unsupported
    raise AttributeError(name)


def _read_pins():
    ns = {}
    with open(os.path.join(HERE, "dispatch", "pins.py")) as fh:
        exec(compile(fh.read(), "pins.py", "exec"), ns)
    return ns["PINS"]


PINS = _read_pins()


def _has_toolkit(entry) -> bool:
    try:
        from torch.utils import cpp_extension
    except Exception:  # noqa: BLE001
        return False
    if not cpp_extension.CUDA_HOME or not os.path.isfile(os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc")):
        return False
    if shutil.which("ninja") is None:
        try:
            import ninja  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
    if CUDA_SEATS[entry]["needs_cutlass"]:            # CUTLASS / CuTe headers (cuda, cuda_b)
        inc = os.path.join(os.environ.get("CUTLASS_PATH", "/opt/cutlass"), "include", "cute", "tensor.hpp")
        if not os.path.isfile(inc):
            return False
    return True


def _load_so(ext_name, path):
    import torch  # noqa: F401  (libtorch / libc10 / libcudart resolve from the process)
    loader = importlib.machinery.ExtensionFileLoader(ext_name, path)
    spec = importlib.util.spec_from_file_location(ext_name, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[ext_name] = mod
    return mod


def _inject(seat_module, slot, ext):
    if slot == "_EXT":
        seat_module._EXT = ext
    elif slot.startswith("_EXTS["):
        seat_module._EXTS[slot[len("_EXTS["):-1]] = ext
    else:
        raise AssertionError(slot)


def _resolve(name):
    """Make the CUDA kernel directory `name` callable in this process: prebuilt for this stack (default), else JIT if the toolkit is here, else Unsupported."""
    st = _SEAT_STATE.get(name)
    if st is not None:
        return st
    with _LOCK:
        st = _SEAT_STATE.get(name)
        if st is not None:
            return st
        entry = CUDA_SEATS[name]
        policy = os.environ.get("TRIATTN_PKG_PREBUILT", "auto")          # auto | always | never
        _router()                                                       # sys.path carries the kernel directories now
        try:
            M = importlib.import_module(entry["module"])
        except ImportError as e:                                        # the directory is absent from this tree (e.g. a generation sealed without it)
            raise _router().Unsupported("%s: module %s not importable from %s (%s)" % (name, entry["module"], os.path.join(HERE, entry["dir"]), e))
        so = os.path.join(PREBUILT_DIR, stack_tag(), entry["ext"] + ".so")
        rec_path = so[:-3] + ".json"
        if policy != "never" and os.path.isfile(so):
            record = json.load(open(rec_path)) if os.path.isfile(rec_path) else None
            if record and record.get("stack_tag") not in (None, stack_tag()):
                raise _router().Unsupported("%s: prebuilt %s was built for %s, this process is %s" % (name, so, record.get("stack_tag"), stack_tag()))
            _inject(M, entry["slot"], _load_so(entry["ext"], so))
            st = {"binary": "prebuilt:" + so, "record": record}
        elif policy == "always":
            raise _router().Unsupported("%s: no prebuilt binary for stack %s under %s (TRIATTN_PKG_PREBUILT=always forbids the JIT build)" % (name, stack_tag(), PREBUILT_DIR))
        elif _has_toolkit(name):
            st = {"binary": "jit", "record": None}                       # the kernel directory's own _build() compiles on the first call (minutes; cached by torch under TORCH_EXTENSIONS_DIR)
        else:
            raise _router().Unsupported("%s: no prebuilt binary for stack %s under %s and no CUDA toolkit to build one (needs nvcc + ninja%s); "
                                        "build it with build_prebuilt.py for this stack or run on a listed stack (prebuilt/INDEX.json)"
                                        % (name, stack_tag(), PREBUILT_DIR, " + CUTLASS headers at $CUTLASS_PATH" if entry["needs_cutlass"] else ""))
        _SEAT_STATE[name] = st
        return st


def route(q, k, v, bias, mask=None) -> str:
    """The kernel `triangle_attention` uses for these arguments: 'k13' | 'tri' | 'cuda' | 'cuda_c' | 'cuda_b' | 'cuda_80' (raises Unsupported naming the reason)."""
    return _router().route(q, k, v, bias, mask)


def triangle_attention(q, k, v, bias, mask=None, scale=None):
    R = _router()
    name = R.route(q, k, v, bias, mask)
    if name in CUDA_SEATS:
        _resolve(name)
    return R.best(q, k, v, bias, mask=mask, scale=scale)


def prebuilt_status() -> dict:
    """What this process has resolved so far per CUDA route, the stack tag, and the prebuilt index shipped in the package."""
    idx_path = os.path.join(PREBUILT_DIR, "INDEX.json")
    idx = json.load(open(idx_path)) if os.path.isfile(idx_path) else None
    try:
        tag = stack_tag()
    except Exception as e:  # noqa: BLE001
        tag = "unavailable: %s" % e
    return {"stack_tag": tag, "resolved": dict(_SEAT_STATE), "index": idx, "policy": os.environ.get("TRIATTN_PKG_PREBUILT", "auto")}
