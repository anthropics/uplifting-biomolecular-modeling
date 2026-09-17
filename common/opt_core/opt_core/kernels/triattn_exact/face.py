"""triattn_exact.face — the single Python face.

    from triattn_exact.face import triangle_attention
    out = triangle_attention(q, k, v, bias, mask=None, scale=None)          # library signature & semantics

Contract
- Same inputs as cuequivariance_torch.triangle_attention (including strided/non-contiguous views, 4-D B=1 forms),
  same output BITS, for every (library version, device class, dtype, head_dim, shape class, env state) cell that
  CELLS.json marks "proven". Everything else raises triattn_exact.Refused BY NAME — never a silent fallback, never
  a silent call into the library. The caller decides what to do with a refusal (typically: call the library).
- First call per (process, device, library version, route) runs a self-check against the LIVE library on a few
  small shapes inside the cell and refuses (and poisons the route for the process) on any mismatch.
- Forward only (inference). Autograd-requiring inputs are refused by name.

This file is deliberately boring. Routes (kernels) live in triattn_exact/<route>/ and expose
    attention(q, k, v, bias, mask=None, scale=None) -> out      and      supports(...) -> (bool, reason)
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from . import Refused
from . import _paths

log = logging.getLogger("triattn_exact")
_HERE = os.path.dirname(os.path.abspath(__file__))
_CELLS_PATH = _paths.cells_path()          # <parent>/CELLS.json or $TRIATTN_EXACT_CELLS (see LAYOUT.md)
_lock = threading.Lock()
_selfcheck_done: dict = {}      # (device_index, lib_version, route) -> True | Refused
_route_modules: dict = {}


# ----------------------------------------------------------------------------------------------------------------
# cell key derivation
# ----------------------------------------------------------------------------------------------------------------
DEVICE_CLASSES = {  # substring of torch.cuda.get_device_name() -> class name used in CELLS.json
    "H100 80GB HBM3": "H100_SXM",
    "H100 PCIe": "H100_PCIe",
    "H100 NVL": "H100_NVL",
    "H200": "H200",
    "A100-SXM4-80GB": "A100_SXM_80GB",
    "A100 80GB PCIe": "A100_PCIe_80GB",
    "A100-SXM4-40GB": "A100_SXM_40GB",
}


_devclass_cache: dict = {}


def device_class(dev: torch.device) -> str:
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    hit = _devclass_cache.get(idx)
    if hit is not None:
        return hit
    cls = _device_class_uncached(idx)
    _devclass_cache[idx] = cls
    return cls


def _device_class_uncached(dev) -> str:
    name = torch.cuda.get_device_name(dev)
    for sub, cls in DEVICE_CLASSES.items():
        if sub in name:
            return cls
    raise Refused(f"device '{name}' is not a characterised device class", {"device": name})


@functools.lru_cache(maxsize=None)
def _installed_ops():
    import importlib.metadata as md
    found = {}
    for pkg in ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12", "cuequivariance-torch"):
        try:
            found[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return found


@functools.lru_cache(maxsize=None)
def device_family(name_or_class: str) -> str:
    """GPU family key used by CELLS rows: SXM/PCIe/NVL variants of one chip fall in one family."""
    n = name_or_class.upper()
    if "H200" in n:
        return "H200"
    if "H100" in n or "GH200" in n:
        return "H100"
    if "A100" in n and "80" in n:
        return "A100_80GB"
    if "A100" in n:
        return "A100_40GB"
    return "UNKNOWN(" + name_or_class + ")"


def torch_version() -> str:
    return torch.__version__.split("+")[0]


def library_version() -> str:
    """CANONICAL version key used by the face, every route's supports() and CELLS.json: the plain 'X.Y.Z' of the
    installed cuequivariance-ops-torch package (the arithmetic lives there). The CUDA build flavour is ops_build()."""
    forced = os.environ.get("TRIATTN_EXACT_LIB_VERSION")
    found = _installed_ops()
    if forced:
        log.warning("triattn_exact: TRIATTN_EXACT_LIB_VERSION=%s overrides detection (installed: %s)", forced, found)
        return forced.split("+")[0]
    for pkg in ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12"):
        if pkg in found:
            return found[pkg]
    raise Refused("cuEquivariance ops package not installed: cannot determine which arithmetic to match "
                  "(set TRIATTN_EXACT_LIB_VERSION=X.Y.Z and TRIATTN_EXACT_OPS_BUILD=cu13 to assert it explicitly)", {"installed": found})


@functools.lru_cache(maxsize=None)
def ops_build() -> str:
    forced = os.environ.get("TRIATTN_EXACT_OPS_BUILD") or (os.environ.get("TRIATTN_EXACT_LIB_VERSION", "").split("+")[1:] or [None])[0]
    if forced:
        return forced
    found = _installed_ops()
    for pkg in ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12"):
        if pkg in found:
            return pkg.rsplit("-", 1)[-1]
    raise Refused("cuEquivariance ops package not installed (ops build unknown)", {"installed": found})


@dataclass(frozen=True)
class CallMeta:
    lib_version: str      # canonical 'X.Y.Z'
    ops_build: str        # 'cu13' | 'cu12'
    device_class: str
    dtype: str
    head_dim: int
    B: int
    N: int
    H: int
    S_q: int
    S_k: int
    has_mask: bool
    mask_dtype: str
    bias_dtype: str
    fallback_threshold_env: Optional[str]
    squeezed_4d: bool


def _normalize(q, k, v, bias, mask):
    """Accept the library's 4-D (B=1) forms; return 5-D views (no copies) + flag."""
    squeezed = False
    if q.dim() == 4:
        squeezed = True
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if bias is not None and bias.dim() == 4:
            bias = bias.unsqueeze(0)
        if mask is not None and mask.dim() == 4:
            mask = mask.unsqueeze(0)
    return q, k, v, bias, mask, squeezed


def describe(q, k, v, bias, mask, scale) -> CallMeta:
    """Validate everything the library validates (or faults on) and derive the cell key. Refuse BY NAME, before any
    route is imported or any kernel is launched, whenever the library would raise or fault — there is nothing to be
    equal to in those cases (wrong device, misaligned q/k/v storage, unsupported bias shapes)."""
    if bias is None:
        raise Refused("bias is required (library signature)", {})
    tensors = {"q": q, "k": k, "v": v, "bias": bias}
    if mask is not None:
        tensors["mask"] = mask
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise Refused(f"{name} is not a torch.Tensor ({type(t).__name__})", {})
    dev = q.device
    if dev.type != "cuda":
        raise Refused("q must be a CUDA tensor", {})
    for name, t in tensors.items():          # a CPU mask/bias with CUDA q would fault inside a kernel
        if t.device != dev:
            raise Refused(f"{name} is on {t.device}, q is on {dev}: all inputs must be on the same CUDA device", {})
    if not (q.dtype == k.dtype == v.dtype):
        raise Refused(f"q/k/v dtypes differ ({q.dtype}, {k.dtype}, {v.dtype})", {})
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise Refused(f"dtype {q.dtype} is not a library dtype", {})
    q5, k5, v5, b5, m5, squeezed = _normalize(q, k, v, bias, mask)
    if q5.dim() != 5 or k5.dim() != 5 or v5.dim() != 5:
        raise Refused(f"q/k/v must be 5-D [B,N,H,S,D] (or 4-D with B=1); got {tuple(q.shape)}", {})
    B, N, H, S_q, D = (int(x) for x in q5.shape)
    if tuple(k5.shape[:3]) != (B, N, H) or tuple(v5.shape[:3]) != (B, N, H) or int(k5.shape[4]) != D or int(v5.shape[4]) != D:
        raise Refused(f"k/v shapes {tuple(k5.shape)}/{tuple(v5.shape)} inconsistent with q {tuple(q5.shape)}", {})
    S_k = int(k5.shape[3])
    if int(v5.shape[3]) != S_k:
        raise Refused(f"k has S={S_k} but v has S={int(v5.shape[3])}", {})
    if b5.dim() != 5 or tuple(b5.shape) != (B, 1, H, S_q, S_k):      # the library rejects other bias shapes
        raise Refused(f"bias shape {tuple(bias.shape)} is not [B,1,H,S_q,S_k]=({B},1,{H},{S_q},{S_k})", {})
    if m5 is not None and (m5.dim() != 5 or tuple(m5.shape) != (B, N, 1, 1, S_k)):
        raise Refused(f"mask shape {tuple(mask.shape)} is not [B,N,1,1,S_k]=({B},{N},1,1,{S_k})", {})
    for name, t in (("q", q5), ("k", k5), ("v", v5)):                  # the library faults on misaligned q/k/v
        if t.stride(-1) != 1:
            raise Refused(f"{name} last dim is not contiguous (stride {t.stride(-1)})", {})
        if t.data_ptr() % 16 != 0:
            raise Refused(f"{name} storage address is not 16-byte aligned (data_ptr % 16 = {t.data_ptr() % 16}); "
                          "the library itself faults on such views", {})
        if any((st * t.element_size()) % 16 != 0 for st in t.stride()[:-1] if st != 0) and False:
            pass  # row strides need not be 16-B aligned for correctness; routes may refuse for their vector width
    return CallMeta(
        lib_version=library_version(), ops_build=ops_build(), device_class=device_class(dev), dtype=str(q.dtype).replace("torch.", ""),
        head_dim=D, B=B, N=N, H=H, S_q=S_q, S_k=S_k, has_mask=mask is not None,
        mask_dtype=str(mask.dtype).replace("torch.", "") if mask is not None else "none",
        bias_dtype=str(bias.dtype).replace("torch.", ""),
        fallback_threshold_env=os.environ.get("CUEQ_TRIATTN_FALLBACK_THRESHOLD"), squeezed_4d=squeezed)


# ----------------------------------------------------------------------------------------------------------------
# CELLS.json: which (version, device, dtype, D, shape-class, env) cells are proven, and which route serves them
# ----------------------------------------------------------------------------------------------------------------
_cells_cache = {"mtime": None, "rows": None}
_select_memo = {}   # (call signature, layout, cells mtime, env) -> ordered servable [(route, cell)] or a Refused


def _load_cells():
    try:
        mt = os.stat(_CELLS_PATH).st_mtime_ns
    except FileNotFoundError:
        raise Refused(f"CELLS.json not found at {_CELLS_PATH}", {})
    if _cells_cache["rows"] is None or _cells_cache["mtime"] != mt:
        with open(_CELLS_PATH) as f:
            doc = json.load(f)
        rows = []
        for c in doc.get("cells", []):
            if not isinstance(c, dict) or c.get("status") != "proven":
                continue
            why = _validate_cell(c)
            if why:
                log.error("triattn_exact: CELLS row %r skipped (malformed: %s) — it will not be served", c.get("id"), why)
                continue
            rows.append(c)
        _cells_cache["rows"] = rows
        _cells_cache["mtime"] = mt
    return _cells_cache["rows"]


_ROUTES = {"cuda_mma", "hopper", "tk", "smalls", "fallback", "breadth", "_prebuilt"}   # importable as triattn_exact.<route>

_PRED_TYPES = {
    "lib_version": (str, list), "ops_build": (str, list), "device_class": (str, list), "dtype": (str, list),
    "head_dim": (int, list), "H": (int, list, str), "B_max": (int,), "S_min": (int,), "S_max": (int,), "square": (bool,),
    "N_equals_S": (bool,), "bias_dtype": (str, list), "mask_dtype": (str, list), "kv_lengths_via_mask": (bool,),
    "fallback_threshold_env": (str, dict),
    "device_family": (str, list), "torch": (str, list), "form": (str, list),
}


def _validate_cell(c: dict) -> str:
    """Return '' if the row is in the face vocabulary, else a reason. Unknown predicate keys are NOT accepted here either
    (they would fail closed at match time anyway; rejecting at load makes the log useful)."""
    if not isinstance(c.get("id"), str):
        return "missing id"
    if c.get("route") not in _ROUTES:
        return f"route {c.get('route')!r} not one of {sorted(_ROUTES)}"
    w = c.get("when")
    if not isinstance(w, dict) or not w:
        return "missing/empty 'when'"
    for k, v in w.items():
        if k not in _PRED_TYPES:
            return f"unknown predicate {k!r}"
        if not isinstance(v, _PRED_TYPES[k]) or (isinstance(v, bool) and bool not in _PRED_TYPES[k] and int in _PRED_TYPES[k]):
            return f"predicate {k!r} has bad type {type(v).__name__}"
        if isinstance(v, list) and not v:
            return f"predicate {k!r} is an empty list"
    for req in ("lib_version", "dtype", "head_dim"):
        if req not in w:
            return f"required predicate {req!r} missing"
    if "device_class" not in w and "device_family" not in w:
        return "required predicate 'device_class' or 'device_family' missing"
    return ""


def _cell_matches(cell: dict, m: CallMeta) -> bool:
    """A cell row carries explicit predicates; all must hold. Unknown predicate keys never match (fail closed)."""
    preds = cell.get("when", {})
    for key, val in preds.items():
        if key == "lib_version":
            ok = m.lib_version in (val if isinstance(val, list) else [val])
        elif key == "ops_build":
            ok = m.ops_build in (val if isinstance(val, list) else [val])
        elif key == "device_class":
            ok = m.device_class in (val if isinstance(val, list) else [val])
        elif key == "dtype":
            ok = m.dtype in (val if isinstance(val, list) else [val])
        elif key == "head_dim":
            ok = m.head_dim in (val if isinstance(val, list) else [val])
        elif key == "H":
            ok = m.H in val if isinstance(val, list) else (val == "any" or m.H == val)
        elif key == "B_max":
            ok = m.B <= val
        elif key == "S_min":
            ok = min(m.S_q, m.S_k) >= val
        elif key == "S_max":
            ok = max(m.S_q, m.S_k) <= val
        elif key == "square":
            ok = (m.S_q == m.S_k) if val else True
        elif key == "N_equals_S":
            ok = (m.N == m.S_q) if val else True
        elif key == "bias_dtype":
            ok = m.bias_dtype in (val if isinstance(val, list) else [val])
        elif key == "mask_dtype":
            ok = (not m.has_mask) or (m.mask_dtype in (val if isinstance(val, list) else [val]))
        elif key == "kv_lengths_via_mask":
            ok = True   # evidence flag, checked separately when kv_lengths is actually used
        elif key == "fallback_threshold_env":
            # val: "unset" | "any" | {"max": int} meaning env must be set to an int <= max (kernel path forced)
            env = m.fallback_threshold_env
            if val == "unset":
                ok = env is None
            elif val == "any":
                ok = True
            elif isinstance(val, dict) and "max" in val:
                try:
                    ok = env is not None and int(env) <= int(val["max"])
                except ValueError:
                    ok = False
            else:
                ok = False
        elif key == "device_family":
            fam = device_family(m.device_class)
            if (fam not in val) if isinstance(val, list) else (fam != val):
                return False
        elif key == "torch":
            tv = torch_version()
            vals = val if isinstance(val, list) else [val]
            if not any(tv == x or tv.startswith(str(x).rstrip("*")) for x in vals):
                return False
        elif key == "form":
            pass  # informational (bias+mask | bias-only | kv_lengths); has_mask / kv_lengths_via_mask are the operative predicates
        else:
            return False
        if not ok:
            return False
    return True


# ---- route-source fingerprint guard ---------------------------------------------------------------------------------
# A proven CELLS.json row certifies specific route SOURCE (kernel + launcher) and may carry
# "route_fingerprint": {"sha256": ...} computed with this algorithm at certification time. If present and the source in
# THIS tree differs, refuse by name instead of serving an uncertified kernel under an old proof. The digest is taken over
# (logical path, file bytes) pairs with logical paths spelled csrc/<route>/... and triattn_exact/<route>/... (see
# _paths.route_source_files), so it does not depend on where the directories live. TRIATTN_EXACT_IGNORE_FINGERPRINT=1
# disables the guard (development use only; then the call is outside the certified cell).
_FP_ROUTE_DIRS = {
    "cuda_mma": ["csrc/cuda_mma", "triattn_exact/cuda_mma"], "fallback": ["triattn_exact/fallback"],
    "hopper": ["csrc/hopper", "triattn_exact/hopper"], "tk": ["csrc/tk", "triattn_exact/tk"], "smalls": ["csrc/smalls", "triattn_exact/smalls"],
    "breadth": ["csrc/cuda_mma/variants/breadth", "triattn_exact/breadth"],
}
_FP_EXCLUDE = {"variants", "tests", "microtests", "__pycache__"}
_FP_EXT = (".cu", ".cuh", ".h", ".hpp", ".py", ".json", ".ptx")
_fp_cache: dict = {}


def route_fingerprint(route: str) -> Optional[str]:
    if route in _fp_cache:
        return _fp_cache[route]
    dirs = _FP_ROUTE_DIRS.get(route)
    if dirs is None:
        return None
    import hashlib
    h = hashlib.sha256()
    for logical, path in _paths.route_source_files(dirs, _FP_EXCLUDE, _FP_EXT):
        with open(path, "rb") as f:
            b = f.read()
        h.update(logical.encode() + b"\0" + hashlib.sha256(b).hexdigest().encode() + b"\n")
    _fp_cache[route] = h.hexdigest()
    return _fp_cache[route]


def _device_sm(dev: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(dev)
    return f"sm_{major}{minor}"


def _bind_cell(cell: dict, m: CallMeta, dev: Optional[torch.device]) -> str:
    """Resolve ONE matching proven cell to the route module name that may serve it, enforcing binary/source fingerprints.
    Raises Refused (by name) when this cell cannot be served from this tree/device."""
    allow_jit = os.environ.get("TRIATTN_EXACT_ALLOW_JIT", "0") == "1"
    bfp = cell.get("binary_fingerprint") or {}
    if bfp and os.environ.get("TRIATTN_EXACT_FORCE_JIT", "0") != "1":
        arch = _device_sm(dev if dev is not None else torch.device("cuda", torch.cuda.current_device()))
        shas = bfp.get(arch) or bfp.get(arch + "a") or []
        if isinstance(shas, dict):            # {toolkit: sha} form
            shas = list(shas.values())
        if shas:
            try:
                pbmod = _route("_prebuilt." + cell["route"])
                pbmod.bind_certified(shas)    # every launch (incl. the self-check) must come from one of these cubins
            except (ImportError, AttributeError) as e:
                raise Refused(f"cell {cell.get('id')} certifies prebuilt binaries but triattn_exact._prebuilt.{cell['route']} is not importable "
                              f"({type(e).__name__}: {e})", {"cell": cell.get("id")})
            return "_prebuilt." + cell["route"]
        if not allow_jit:
            raise Refused(f"cell {cell.get('id')} certifies prebuilt {cell['route']} binaries for {sorted(bfp)} only; this device is {arch}. "
                          f"Set TRIATTN_EXACT_ALLOW_JIT=1 to JIT-compile the route from source with nvcc (development only)",
                          {"cell": cell.get("id"), "arch": arch})
    want = (cell.get("route_fingerprint") or {}).get("sha256")
    if want and os.environ.get("TRIATTN_EXACT_IGNORE_FINGERPRINT", "0") != "1":
        have = route_fingerprint(cell["route"])
        if have is not None and have != want:
            raise Refused(f"route '{cell['route']}' source in this tree ({have[:12]}) differs from the source certified for cell "
                          f"{cell.get('id')} ({want[:12]}): use a certified release of these files or re-certify the cell",
                          {"cell": cell.get("id"), "have": have, "want": want})
    return cell["route"]


def _dispatch_order(m: CallMeta, q, k, v, bias) -> list:
    """Route-name preference from the fitted dispatch table (dispatch.json in the package). ORDER ONLY — proof stays in
    CELLS.json; any problem with the table degrades to file order (never affects bits, only which proven route runs)."""
    try:
        from . import dispatch as _d
        ch = _d.routes_for(m.device_class, m.lib_version, _d.layout_signature(q, k, v, bias), m.B, m.N, m.H, m.S_q, m.S_k,
                           m.has_mask, getattr(torch, m.dtype) if isinstance(m.dtype, str) else m.dtype, m.head_dim, proven_only=True)
        out = []
        for c in ch:
            r = getattr(c, "route", None)
            if r and not getattr(c, "kwargs", None) and r not in out:   # default builds only (evidence is per shipped build)
                out.append(r)
        return out
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # noqa: BLE001
        log.debug("triattn_exact: dispatch table unavailable (%s: %s) — using CELLS.json order", type(e).__name__, e)
        return []


def select_routes(m: CallMeta, dev: Optional[torch.device] = None, q=None, k=None, v=None, bias=None) -> list:
    """-> ordered [(route module name, cell), ...] over ALL proven cells matching this call (Dispatch preference first, then
    CELLS.json file order), each already bound/fingerprint-checked. Cells that cannot be served from this tree are skipped with
    their refusal recorded; if none remains, Refused by name with every reason."""
    all_cells = _load_cells()
    memo_key = None
    try:
        lay = None
        if q is not None:
            from . import dispatch as _d
            lay = _d.layout_signature(q, k, v, bias)
        memo_key = (tuple(sorted((kk, str(vv)) for kk, vv in m.__dict__.items())), lay, _cells_cache.get("mtime"),
                    dev.index if dev is not None else None, os.environ.get("TRIATTN_EXACT_IGNORE_FINGERPRINT"),
                    os.environ.get("TRIATTN_EXACT_ALLOW_JIT"), os.environ.get("TRIATTN_EXACT_FORCE_JIT"))
        hit = _select_memo.get(memo_key)
        if hit is not None:
            if isinstance(hit, Refused):
                raise hit
            return hit
    except Refused:
        raise
    except Exception:  # noqa: BLE001  # hygiene: no-cuda (memo key built from python metadata only; optimisation)
        memo_key = None
    cells = [c for c in all_cells if _cell_matches(c, m)]
    if not cells:
        err = Refused("no proven cell covers this call", {"meta": m.__dict__})
        if memo_key is not None:
            _select_memo[memo_key] = err
        raise err
    pref = _dispatch_order(m, q, k, v, bias) if q is not None else []
    rank = {r: i for i, r in enumerate(pref)}
    cells = sorted(cells, key=lambda c: (rank.get(c.get("route"), len(rank)), 0))   # stable: file order within equal rank
    served, refusals = [], []
    for cell in cells:
        try:
            served.append((_bind_cell(cell, m, dev), cell))
        except Refused as e:
            refusals.append(f"{cell.get('id')}: {e.reason}")
    if not served:
        err = Refused("no matching proven cell can be served from this tree/device: " + " | ".join(refusals),
                      {"cells": [c.get("id") for c in cells]})
        if memo_key is not None:
            _select_memo[memo_key] = err
        raise err
    if memo_key is not None:
        if len(_select_memo) > 4096:
            _select_memo.clear()
        _select_memo[memo_key] = served
    return served


def select_route(m: CallMeta, dev: Optional[torch.device] = None) -> Tuple[str, dict]:
    """First servable (route, cell) — kept for tests and tooling; the face itself uses select_routes() with fall-through."""
    return select_routes(m, dev)[0]

def _route(name: str):
    base = name[len("_prebuilt."):] if name.startswith("_prebuilt.") else name
    if base not in _ROUTES or base == "_prebuilt":
        raise Refused(f"route {name!r} is not a known route", {})
    with _lock:
        if name not in _route_modules:
            import importlib
            try:
                _route_modules[name] = importlib.import_module("." + name, __package__)      # relative: the package may be re-rooted
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001 — a route that cannot import is refused by name, not an untyped error
                raise Refused(f"route {name!r} failed to import: {type(e).__name__}: {e}", {}) from e
        return _route_modules[name]


# ----------------------------------------------------------------------------------------------------------------
# first-call self-check against the live library
# ----------------------------------------------------------------------------------------------------------------
def _selfcheck(route_name: str, m: CallMeta, device: torch.device, dtype: torch.dtype):
    key = (device.index, m.lib_version, route_name, m.dtype, m.head_dim)
    state = _selfcheck_done.get(key)
    if state is True:
        return
    if isinstance(state, Refused):
        raise state
    if torch.cuda.is_current_stream_capturing():
        raise Refused("self-check inside CUDA-graph capture: the one-time self-check against the live library cannot run while "
                      "capturing; call the face once outside capture (warm-up) first", {"route": route_name})
    if os.environ.get("TRIATTN_EXACT_SKIP_SELFCHECK") == "1":
        log.warning("triattn_exact: self-check SKIPPED by TRIATTN_EXACT_SKIP_SELFCHECK=1 (route=%s, lib=%s)", route_name, m.lib_version)
        _selfcheck_done[key] = True
        return
    try:
        from cuequivariance_torch import triangle_attention as lib_fn
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # library not importable -> cannot self-check -> refuse by name
        err = Refused(f"self-check impossible: cuequivariance_torch not importable ({type(e).__name__}: {e})", {"route": route_name})
        _selfcheck_done[key] = err
        raise err
    mod = _route(route_name)
    g = torch.Generator(device=device); g.manual_seed(20260915)
    shapes = mod.SELFCHECK_SHAPES if hasattr(mod, "SELFCHECK_SHAPES") else [(1, 8, 4, 128, m.head_dim, True), (1, 3, 2, 191, m.head_dim, False), (2, 4, 4, 136, m.head_dim, True)]
    # the check runs with the CALL's dtype and head_dim (a D=64 or fp16 cell is checked as such); D=16 kernel cells start at S=201
    shapes = [(B, N, H, (max(S, 201) if m.head_dim == 16 else S), m.head_dim, use_mask) for (B, N, H, S, D, use_mask) in shapes]
    if route_name == "fallback":
        shapes = [(B, N, H, min(S, 100), D, use_mask) for (B, N, H, S, D, use_mask) in shapes]
    for (B, N, H, S, D, use_mask) in shapes:
        proj = torch.randn(B, N, S, 3 * H * D, device=device, dtype=dtype, generator=g)
        q, k, v = [proj[..., i * H * D:(i + 1) * H * D].unflatten(-1, (H, D)).permute(0, 1, 3, 2, 4) for i in range(3)]
        bias = torch.randn(B, S, S, H, device=device, dtype=torch.float32, generator=g).permute(0, 3, 1, 2).unsqueeze(1)
        mask = None
        if use_mask:
            lengths = torch.randint(1, S + 1, (B, N), device=device, generator=g)
            lengths[0, 0] = 0  # a fully-masked row
            mask = (torch.arange(S, device=device).view(1, 1, S) < lengths.unsqueeze(-1)).view(B, N, 1, 1, S)
        with torch.no_grad():
            ref = lib_fn(q, k, v, bias, mask=mask)
            try:
                ours = (mod.attention(q, k, v, bias, mask=mask, scale=None, lib_version=m.lib_version) if route_name == "fallback"
                        else mod.attention(q, k, v, bias, mask=mask, scale=None))
            except Refused:
                raise
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # build / load / launch failure inside the self-check -> typed refusal, route disabled
                err = Refused(f"self-check could not run route '{route_name}': {type(e).__name__}: {str(e)[:200]}", {"route": route_name})
                _selfcheck_done[key] = err
                raise err
        same = ref.shape == ours.shape and torch.equal(ref.view(torch.int16), ours.view(torch.int16))
        if not same:
            nbad = int((ref.view(torch.int16) != ours.view(torch.int16)).sum()) if ref.shape == ours.shape else -1
            err = Refused(f"SELF-CHECK MISMATCH vs live library on shape B{B} N{N} H{H} S{S} D{D} mask={use_mask}: "
                          f"{nbad} differing elements — route '{route_name}' disabled for this process", {"route": route_name, "lib": m.lib_version})
            _selfcheck_done[key] = err
            log.error(str(err))
            raise err
    _selfcheck_done[key] = True
    log.info("triattn_exact: self-check passed (route=%s, lib=%s, device=%s, %d shapes)", route_name, m.lib_version, m.device_class, len(shapes))


# ----------------------------------------------------------------------------------------------------------------
# the face
# ----------------------------------------------------------------------------------------------------------------
def _compile_opaque(fn):
    """Run the face EAGERLY even inside torch.compile'd models: Dynamo/Inductor would otherwise re-derive the fallback
    composition (and trace our launch glue), changing bits relative to the eager library (observed with torch 2.7) and
    turning Refused into InternalTorchDynamoError on older torch. A graph break here is the boring, exact choice."""
    try:
        import torch.compiler as _tc
        if hasattr(_tc, "disable"):
            return _tc.disable(fn)
    except Exception:  # noqa: BLE001  # hygiene: probe-once (decorator applied once at import; no CUDA work)
        pass
    try:
        import torch._dynamo as _dyn
        return _dyn.disable(fn)
    except Exception:  # noqa: BLE001  # hygiene: probe-once (decorator applied once at import; no CUDA work)
        return fn


@_compile_opaque
def triangle_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, scale: Optional[float] = None, return_aux: bool = False,
                       *, kv_lengths: Optional[torch.Tensor] = None):
    """Drop-in for cuequivariance_torch.triangle_attention (forward). Raises triattn_exact.Refused by name outside proven cells."""
    if return_aux:
        raise Refused("return_aux=True is not a proven cell (only `out` equality is certified)", {})
    kv_converted = False
    if kv_lengths is not None:
        if mask is not None:
            raise Refused("mask and kv_lengths given together (the library raises ValueError here)", {})
        # Semantics (public docs/frontend): kv_lengths = per-row valid prefix length, int32 [B,N,1,1,1] (or [N,1,1,1]).
        # We serve it ONLY through an equivalent dense prefix mask, and only in cells whose evidence includes the
        # library(kv_lengths) == library(prefix mask) identity (predicate kv_lengths_via_mask); otherwise refuse by name.
        if not isinstance(kv_lengths, torch.Tensor) or kv_lengths.device != q.device:
            raise Refused("kv_lengths must be a tensor on q.device", {})
        if kv_lengths.dtype != torch.int32:
            raise Refused(f"kv_lengths dtype {kv_lengths.dtype} is not int32 (library contract)", {})
        S_k = k.shape[-2]
        kl = kv_lengths
        while kl.dim() < 5:
            kl = kl.unsqueeze(0)
        if kl.shape[-3:] != (1, 1, 1):
            raise Refused(f"kv_lengths shape {tuple(kv_lengths.shape)} is not [B,N,1,1,1]", {})
        ar = torch.arange(S_k, device=q.device, dtype=torch.int32).view(1, 1, 1, 1, S_k)
        mask = ar < kl
        if q.dim() == 4:
            mask = mask.squeeze(0)
        kv_converted = True
        kv_live = (kl > 0)          # [B,N,1,1,1]; zero-length rows get the library's signed-zero treatment below
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, bias)):
        raise Refused("forward-only face: inputs require grad while grad mode is enabled (wrap in torch.no_grad or call the library)", {})
    if scale is not None and not (isinstance(scale, (int, float)) and math.isfinite(scale)):
        raise Refused(f"scale={scale!r} not a finite float", {})
    # Normalise exactly like the library (proven on the kernel path, both versions): any mask dtype is
    # read as bool (nonzero => valid; NB a float 0/-inf "additive-style" mask is therefore read inverted — by the library too),
    # and fp16/fp64 bias == bias.float(). bf16/fp32 bias pass through (routes serve both).
    if mask is not None and mask.dtype != torch.bool:
        mask = mask != 0
    # Defense in depth: stride-0 (expand()ed) dims are legal library inputs; some routes' bulk-copy engines
    # cannot express a 0 stride. Materialise such tensors here once (values unchanged ⇒ library output unchanged; routes
    # then see ordinary strided views). Size-1 dims with stride 0 are harmless and left alone.
    def _dezero(t):
        if t is None or not any(st == 0 and sz > 1 for st, sz in zip(t.stride(), t.shape)):
            return t
        return t.contiguous()
    q, k, v, bias, mask = _dezero(q), _dezero(k), _dezero(v), _dezero(bias), _dezero(mask)
    if bias is not None and bias.dtype in (torch.float16, torch.float64):
        bias = bias.float()
    m = describe(q, k, v, bias, mask, scale)
    candidates = select_routes(m, q.device, q, k, v, bias)
    q5, k5, v5, b5, m5, squeezed = _normalize(q, k, v, bias, mask)
    refusals = []
    out = None
    for route_name, cell in candidates:
        if kv_converted and not cell.get("when", {}).get("kv_lengths_via_mask", False):
            refusals.append(f"{cell.get('id')}: kv_lengths given but this cell carries no kv_lengths==prefix-mask evidence"); continue
        mod = _route(route_name)
        ok, reason = mod.supports(q, k, v, bias, mask, scale, m.lib_version, m.device_class)
        if not ok:
            refusals.append(f"route '{route_name}' refuses: {reason}"); continue
        try:
            _selfcheck(route_name, m, q.device, q.dtype)      # a mismatch poisons the route for the process (logged loudly inside)
        except Refused as e:                                    # self-check refused/failed: next proven route (still fail-closed, never the library)
            refusals.append(f"route '{route_name}' self-check: {e.reason}"); continue
        try:
            if route_name == "fallback":
                out = mod.attention(q5, k5, v5, b5, mask=m5, scale=scale, lib_version=m.lib_version)
            else:
                out = mod.attention(q5, k5, v5, b5, mask=m5, scale=scale)
        except Refused as e:                                    # fail CLOSED: next proven route, never the library
            refusals.append(f"route '{route_name}': {e.reason}"); out = None; continue
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:                                  # build/load/launch failure -> typed, by name (never untyped)
            refusals.append(f"route '{route_name}' failed: {type(e).__name__}: {str(e)[:200]}"); out = None; continue
        break
    if out is None:
        raise Refused("every proven route for this call refused: " + " | ".join(refusals), {"cells": [c.get("id") for _, c in candidates]})
    if kv_converted:
        # library(kv_lengths): live rows == dense-prefix-mask path bit-for-bit; ZERO-length rows == (mask-path value) * 0.0
        # (signed zeros; proven identity, see SPEC). Capture-safe, no sync.
        out = torch.where(kv_live, out, out * 0.0)
    return out          # 5-D [B,N,H,S,D] even for 4-D inputs — exactly what the library returns


__all__ = ["triangle_attention", "describe", "select_route", "select_routes", "CallMeta", "device_class", "library_version", "ops_build"]
