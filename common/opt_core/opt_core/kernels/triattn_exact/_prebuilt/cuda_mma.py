"""The cuda_mma route served from prebuilt cubins through the CUDA driver API.

Interface and semantics equal the source route (triattn_exact.cuda_mma):
    attention(q, k, v, bias, mask=None, scale=None) -> out        supports(q, k, v, bias, mask, scale, lib_version, device) -> (bool, reason)
Nothing is compiled here.  The kernel is the cubin recorded in the manifest for (route source of this tree or the certified sha set,
device arch, driver-loadable toolkit); the host-side work of the C++ launcher (cfg -> template instance, grid/block/dynamic smem,
parameter struct) is done below in Python, and everything upstream of the launch (input checks and typed refusals, bias staging,
config policy, tail split, V-finiteness pre-pass policy) is the source route's own Python, imported, so the bits are the same.

Parameter marshalling is one struct.pack_into into a thread-local buffer plus one cuLaunchKernel on torch's current stream; no syncs,
CUDA-graph capturable; no torch C++ ABI use.  Every failure is a typed triattn_exact.Refused.
Knobs for A/B work: TRIATTN_EXACT_PREBUILT_TOOLKIT=12.6|13.0, TRIATTN_EXACT_PREBUILT_SOURCE=<fp prefix>|any, TRIATTN_EXACT_DRIVER=ctypes|cuda-python;
per-call toolkit= / cfg= keyword arguments.
"""
from __future__ import annotations

import ctypes
import os
import struct
import threading

import torch

from .. import cuda_mma as cm                      # the source route's Python side (pure Python at import; nvcc runs only if ITS attention() is called)
from . import Refused, cubin_bytes, select_build, source_fingerprint
from . import driver as drv

ROUTE = "cuda_mma"
_OOM = getattr(torch.cuda, "OutOfMemoryError", MemoryError)

# ---- kernel ABIs: parameter struct PV3 passed by value (csrc/cuda_mma/triattn_v3.cu) ------------------------------------------------
#  cuda_mma.v3.2: q k v bias mask out | B N H S n_groups bias_fast | 18 x int64 strides | scale
#  cuda_mma.v3.3: q k v bias mask out | B N H S n_groups bias_fast q0 q_rows | vbad | 18 x int64 strides | scale   (+ separate vscan kernel)
_PV3 = {
    "cuda_mma.v3.2": struct.Struct("@6P6i18qf"),
    "cuda_mma.v3.3": struct.Struct("@6P8iP18qf"),
}
_VSCAN = struct.Struct("@P4qP2q3iP")                 # vscan_kernel(v, svB, svN, svH, svS, mask, smB, smN, N, H, S, out) — separate params
_VSCAN_OFFS = []
_o = 0
for _c in "PqqqqPqqiiiP":
    _al = 8 if _c in "Pq" else 4
    _o = (_o + _al - 1) // _al * _al
    _VSCAN_OFFS.append(_o); _o += _al
assert _VSCAN_OFFS[-1] + 8 == _VSCAN.size, (_VSCAN_OFFS, _VSCAN.size)


def _smem_bytes(abi, G, BM, NST, bias_bf16, has_mask, S):
    ntiles = (S + 63) // 64
    stage = G * 2 * 4096 + BM * 72 * (2 if bias_bf16 else 4)
    smem = NST * stage                                   # Q is register-resident in every built configuration (QSM = 0)
    if has_mask:
        smem += G * ntiles * 8 + (ntiles * 8 if abi == "cuda_mma.v3.3" else ntiles * 4)
    return smem


class _Tls(threading.local):
    def __init__(self):
        self.pbuf = ctypes.create_string_buffer(256)                 # >= sizeof(PV3) of any ABI
        self.pargs = (ctypes.c_void_p * 1)(ctypes.addressof(self.pbuf))
        self.vbuf = ctypes.create_string_buffer(_VSCAN.size + 8)
        base = ctypes.addressof(self.vbuf)
        self.vargs = (ctypes.c_void_p * len(_VSCAN_OFFS))(*[base + o for o in _VSCAN_OFFS])


_tls = _Tls()
_state_lock = threading.Lock()
_loaded = {}                # (device ordinal, selection key) -> (build, drv.Module)
_tree_fp = None
_certified_shas = None      # set by the face from the matching CELLS row (binary_fingerprint): only these cubins may serve


def bind_certified(shas):
    """Face hook: restrict every later launch in this process (self-check included) to cubins whose sha256 is in `shas`."""
    global _certified_shas
    new = frozenset(shas)
    if _certified_shas != new:
        with _state_lock:
            _certified_shas = new
            _loaded.clear()


def tree_fingerprint():
    """Route-source fingerprint of this tree; without a certified sha set the serving build must have been compiled from it."""
    global _tree_fp
    if _tree_fp is None:
        _tree_fp = source_fingerprint(ROUTE) or ""
    return _tree_fp or None


def _stream_of(dev_index):
    try:
        return torch._C._cuda_getCurrentRawStream(dev_index)      # same value as torch.cuda.current_stream().cuda_stream, cheaper
    except (AttributeError, RuntimeError, TypeError):
        return torch.cuda.current_stream(dev_index).cuda_stream


def _torch_cuda_major():
    try:
        return int(str(torch.version.cuda).split(".")[0])
    except (TypeError, ValueError):
        return None


def resolve(device, toolkit=None, allowed_shas=None):
    """-> (build, drv.Module) for this device, loading the cubin on first use.  Refused by name when nothing certified applies or the
    driver cannot load/resolve the module."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    be = drv.backend()
    dev = drv.Device.get(idx)
    if allowed_shas is None and _certified_shas is not None:
        allowed_shas = _certified_shas
    key_env = (toolkit, tuple(sorted(allowed_shas)) if allowed_shas is not None else None,
               os.environ.get("TRIATTN_EXACT_PREBUILT_TOOLKIT"), os.environ.get("TRIATTN_EXACT_PREBUILT_SOURCE"))
    ck = (idx, key_env)
    hit = _loaded.get(ck)
    if hit is not None:
        return hit
    with _state_lock:
        hit = _loaded.get(ck)
        if hit is not None:
            return hit
        build, why = select_build(ROUTE, dev.cc, be.driver_version(), tree_fingerprint=tree_fingerprint(), allowed_shas=allowed_shas,
                                  prefer_major=_torch_cuda_major(), toolkit=toolkit)
        if build is None:
            raise Refused(f"prebuilt cuda_mma: {why}", {"route": ROUTE, "cc": dev.cc})
        if build.get("abi") not in _PV3:
            raise Refused(f"prebuilt cuda_mma: unknown kernel ABI {build.get('abi')} in manifest", {"route": ROUTE})
        try:
            entries = [k["entry"] for k in build["kernels"].values()]
        except (KeyError, TypeError, AttributeError) as e:
            raise Refused(f"prebuilt cuda_mma: manifest entry malformed ({type(e).__name__})", {"route": ROUTE})
        mod = drv.load_module(idx, build["cubin_sha256"], lambda: cubin_bytes(build), entries)
        hit = (build, mod)
        _loaded[ck] = hit
        return hit


def _launch_main(be, mod, build, kinfo, abi, q5, k5, v5, b5, m5, out, scale, bias_fast, q0, q_rows, vbad_ptr, stream):
    B, N, H, S, D = q5.shape
    G, BM, NST = kinfo["G"], kinfo["BM"], kinfo["nst"]
    n_groups = (N + G - 1) // G
    if B * n_groups > 65535:
        raise Refused(f"B*ceil(N/G)={B * n_groups} > 65535 grid limit", cell=dict(B=B, N=N))
    entry = kinfo["entry"]
    smem = _smem_bytes(abi, G, BM, NST, b5.dtype == torch.bfloat16, m5 is not None, S)
    if smem > mod.max_dyn_smem[entry]:
        raise Refused(f"S={S}: dynamic shared memory {smem} B exceeds the device limit {mod.max_dyn_smem[entry]} B for this config", cell=dict(S=S))
    sbB = b5.stride(0) if b5.shape[0] == B else 0
    smB = (m5.stride(0) if m5.shape[0] == B else 0) if m5 is not None else 0
    smN = m5.stride(1) if m5 is not None else 0
    mptr = m5.data_ptr() if m5 is not None else 0
    t = _tls
    if abi == "cuda_mma.v3.3":
        _PV3[abi].pack_into(t.pbuf, 0, q5.data_ptr(), k5.data_ptr(), v5.data_ptr(), b5.data_ptr(), mptr, out.data_ptr(),
                            B, N, H, S, n_groups, bias_fast, q0, q_rows, vbad_ptr,
                            q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3), k5.stride(0), k5.stride(1), k5.stride(2), k5.stride(3),
                            v5.stride(0), v5.stride(1), v5.stride(2), v5.stride(3), sbB, b5.stride(2), b5.stride(3), b5.stride(4), smB, smN,
                            scale)
        grid = ((q_rows + BM - 1) // BM, H, B * n_groups)
    else:
        _PV3[abi].pack_into(t.pbuf, 0, q5.data_ptr(), k5.data_ptr(), v5.data_ptr(), b5.data_ptr(), mptr, out.data_ptr(),
                            B, N, H, S, n_groups, bias_fast,
                            q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3), k5.stride(0), k5.stride(1), k5.stride(2), k5.stride(3),
                            v5.stride(0), v5.stride(1), v5.stride(2), v5.stride(3), sbB, b5.stride(2), b5.stride(3), b5.stride(4), smB, smN,
                            scale)
        grid = ((S + BM - 1) // BM, H, B * n_groups)
    be.launch(mod.funcs[entry], grid, (kinfo["threads"], 1, 1), smem, stream, t.pargs)


def _launch_vscan(be, mod, build, v5, m5, vb, stream):
    B, N, H, S, D = v5.shape
    smB = m5.stride(0) if m5.shape[0] == B else 0
    t = _tls
    _VSCAN.pack_into(t.vbuf, 0, v5.data_ptr(), v5.stride(0), v5.stride(1), v5.stride(2), v5.stride(3), m5.data_ptr(), smB, m5.stride(1),
                     N, H, S, vb.data_ptr())
    be.launch(mod.funcs[build["kernels"]["vscan"]["entry"]], (B * N, H, 1), (128, 1, 1), 0, stream, t.vargs)


def _kinfo(build, cfg, has_mask, bias_bf16):
    try:
        nwr, nsplit, gw, nst = build["cfg_table"][str(cfg)]
    except KeyError:
        raise Refused(f"cfg {cfg} is not compiled into prebuilt build {build['blob']} (has {sorted(int(c) for c in build['cfg_table'])})", {"route": ROUTE})
    return build["kernels"][f"nwr{nwr}_ns{nsplit}_gw{gw}_nst{nst}_m{int(has_mask)}_b{int(bias_bf16)}"]


def _no_dynamo(fn):
    """Keep torch.compile from tracing into the launcher (opaque driver-API work): the call graph-breaks around it and a typed Refused
    propagates unchanged under every supported torch version."""
    dis = getattr(getattr(torch, "compiler", None), "disable", None) or getattr(getattr(torch, "_dynamo", None), "disable", None)
    return dis(fn) if dis is not None else fn


@_no_dynamo
def attention(q, k, v, bias, mask=None, scale=None, return_aux=False, *, kv_lengths=None, force_kernel=False, cfg=None, toolkit=None,
              allowed_shas=None, tail_split=None, **unexpected):
    """Forward triangle attention (library signature: ..., return_aux=False, *, kv_lengths=None) from the prebuilt cubin.  Every failure,
    including arguments this route does not serve (return_aux=True, kv_lengths, unknown keywords), is a typed triattn_exact.Refused;
    CUDA out-of-memory from the output/scratch allocation propagates as torch raises it."""
    try:
        if kv_lengths is not None:
            raise Refused("kv_lengths is not served by the cuda_mma route directly (the face lowers kv_lengths to a key mask); pass mask=", {"route": ROUTE})
        return _attention(q, k, v, bias, mask, scale, return_aux, force_kernel, cfg, toolkit, allowed_shas, tail_split, unexpected)
    except Refused:
        raise
    except _OOM:
        raise
    except (struct.error, KeyError, TypeError, ValueError, AttributeError, OSError, OverflowError) as e:
        raise Refused(f"prebuilt cuda_mma: internal {type(e).__name__}: {e}", {"route": ROUTE})


def _attention(q, k, v, bias, mask, scale, return_aux, force_kernel, cfg, toolkit, allowed_shas, tail_split, unexpected):
    if return_aux:
        raise Refused("return_aux=True is not served by the cuda_mma route (lse/max side outputs are not reproduced)")
    if unexpected:
        raise Refused(f"unsupported keyword arguments {sorted(unexpected)}")
    q5, k5, v5, b5, m5 = cm._check_inputs(q, k, v, bias, mask)
    B, N, H, S, D = q5.shape
    if not force_kernel and S <= cm.fallback_threshold():
        raise Refused(f"S={S} <= CUEQ_TRIATTN_FALLBACK_THRESHOLD={cm.fallback_threshold()}: the library takes its torch fallback path "
                      f"here (different arithmetic); kernel path only", cell=dict(S=S))
    if B * N > 65535:
        raise Refused(f"B*N={B*N} > 65535 grid limit of this launcher", cell=dict(B=B, N=N))
    build, mod = resolve(q5.device, toolkit=toolkit, allowed_shas=allowed_shas)
    abi = build["abi"]
    if cfg is None:
        cfg = cm.pick_cfg_id(B, N, H, S)
    kmain = _kinfo(build, cfg, m5 is not None, b5.dtype == torch.bfloat16)
    BM = kmain["BM"]
    # tail split (v3.3 ABI: q0/q_rows), following the source route's policy when its launcher provides one
    tail_rows, tail_cfg = 0, None
    if abi == "cuda_mma.v3.3" and hasattr(cm, "plan_rows") and (tail_split is None or tail_split):
        _, tail_rows, tail_cfg = cm.plan_rows(S, BM)
    b5 = cm._stage_bias(b5)
    bias_fast = cm._bias_fast(b5)
    out = torch.empty((B, N, H, S, D), dtype=torch.bfloat16, device=q5.device)
    scale_f = float(cm.default_scale(D) if scale is None else scale)
    idx = q5.device.index if q5.device.index is not None else torch.cuda.current_device()
    be = drv.backend()
    dev = mod.device
    with torch.cuda.device(idx):
        stream = _stream_of(idx)
        prev = be.ctx_get_current()
        if prev != dev.ctx:
            be.ctx_set_current(dev.ctx)
        try:
            vbad_ptr = 0
            if abi == "cuda_mma.v3.3" and m5 is not None:
                nqblk = (S + BM - 1) // BM + (1 if tail_rows else 0)
                if nqblk >= getattr(cm, "VSCAN_MIN_QBLOCKS", 4):
                    vb = torch.empty(B * N * H * ((S + 63) // 64), dtype=torch.uint8, device=q5.device)
                    _launch_vscan(be, mod, build, v5, m5, vb, stream)
                    vbad_ptr = vb.data_ptr()
            _launch_main(be, mod, build, kmain, abi, q5, k5, v5, b5, m5, out, scale_f, bias_fast, 0, S - tail_rows, vbad_ptr, stream)
            if tail_rows:
                ktail = _kinfo(build, tail_cfg, m5 is not None, b5.dtype == torch.bfloat16)
                _launch_main(be, mod, build, ktail, abi, q5, k5, v5, b5, m5, out, scale_f, bias_fast, S - tail_rows, tail_rows, vbad_ptr, stream)
        finally:
            if prev != dev.ctx and prev:
                be.ctx_set_current(prev)
    return out


# ---- capability + availability (proof lives in CELLS.json) ----------------------------------------------------------------------
LIB_VERSIONS_SERVED = cm.LIB_VERSIONS_SERVED
DEVICE_CLASSES_SERVED = cm.DEVICE_CLASSES_SERVED
CAPABILITY = "prebuilt cubin of the cuda_mma route (" + cm.CAPABILITY + "); needs only libcuda (driver >= the cubin's toolkit)"


def supports(q, k, v, bias, mask, scale, lib_version, device):
    ok, reason = cm.supports(q, k, v, bias, mask, scale, lib_version, device)
    if not ok:
        return ok, reason
    try:
        dev = q.device if isinstance(q, torch.Tensor) and q.is_cuda else torch.device("cuda", torch.cuda.current_device())
        build, _ = resolve(dev)
    except Refused as e:
        return False, e.reason
    except _OOM:
        raise
    except Exception as e:  # noqa: BLE001  # hygiene: no-cuda (probe-once path: loader failure -> a reason, never an exception)
        return False, f"prebuilt loader error: {type(e).__name__}: {e}"
    return True, f"{reason} [prebuilt {build['arch']} cuda{build['toolkit']['cuda']} cubin {build['cubin_sha256'][:12]} src {build['source_fingerprint'][:12]}]"


def loaded_build(device=None):
    """The manifest entry currently serving `device` (loads it if needed)."""
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return resolve(dev)[0]


# ---- toolkit-pinned and per-config entry points for A/B testing (all may live in one process) -------------------------------------
def attention_cuda13(q, k, v, bias, mask=None, scale=None, return_aux=False, **kw):
    """Library-signature entry pinned to the CUDA 13.0-built cubin (unknown/unserved keywords -> typed Refused, like attention())."""
    return attention(q, k, v, bias, mask, scale, return_aux, toolkit="13.0", **kw)


def attention_cuda12(q, k, v, bias, mask=None, scale=None, return_aux=False, **kw):
    """Library-signature entry pinned to the CUDA 12.6-built cubin."""
    return attention(q, k, v, bias, mask, scale, return_aux, toolkit="12.6", **kw)


def _mk_cfg_fn(c):
    def f(q, k, v, bias, mask=None, scale=None, return_aux=False, **kw):
        return attention(q, k, v, bias, mask, scale, return_aux, cfg=c, tail_split=False, **kw)
    f.__name__ = f"attention_cfg{c}"
    return f


for _c in range(11):
    globals()[f"attention_cfg{_c}"] = _mk_cfg_fn(_c)

__all__ = ["attention", "supports", "resolve", "loaded_build", "attention_cuda12", "attention_cuda13", "CAPABILITY"]
