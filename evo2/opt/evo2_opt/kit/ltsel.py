"""T9: cublasLt configuration selection for the bf16 GEMMs torch runs through cuBLAS (MLP l1 / l2 / l3, attention Wqkv / out_proj, the Hyena
blocks' out_filter_dense at B == 1); a configuration is used only after it reproduced torch's output bit for bit on the live operands.

A GEMM signature is (M, N, K, x layout, W layout, bias, device). Signatures with M < MIN_M (decode steps, short prompts) and operands that are not
256-byte aligned / not plain or transpose-contiguous always take torch's call. For the others:
  * forward 1 of a signature (its first `count` calls, `count` = the modules on that device that share it): torch's call, untouched;
  * the first call of forward 2: cublasLtMatmulAlgoGetHeuristic (the library torch itself calls) lists up to NREQ configurations; those that compute
    every output element with one sequential fp32 accumulation over k (split-K count 1, reduction scheme none) are run on THIS call's operands and
    admitted only if their output equals torch's output of the call bit for bit; each admitted one is timed beside torch's call (cold L2, torch
    first, TIMED_RUNS each; the winner once more with torch second) and the fastest becomes the candidate when it is at least 2 % faster both times
    (no candidate: the signature keeps torch's call); the call returns torch's output;
  * the second call of forward 2: the candidate runs beside torch's call on the same operands once more; a difference retires the signature to
    torch's call for the rest of the process (named once on stderr); the rest of forward 2 is torch's call;
  * forwards 3-6, the trial inside the model's own stream of kernels: the signature's calls run the candidate in forwards 3 and 6 and torch's call
    in forwards 4 and 5, a pair of CUDA events around each call, nothing synchronized; the first call of forward 7 reads the events and keeps the
    candidate only when its two forwards' summed GEMM time is at least 2 % below torch's two forwards' — torch's call otherwise; from then on the
    signature is settled: one cublasLtMatmul on the current stream (32 MiB workspace per device), or torch's call.
At most MAX_SHAPES distinct M per device keep their signatures; the least recently used M is dropped (its signatures start over when met again).
`settled(B, L, device)` tells whether every signature met at M = B * L on that device has reached its final state (the candidate kept after
its trial, or torch's call)."""
from __future__ import annotations

import collections
import ctypes
import glob
import os
import struct
import sys

import torch
import torch.nn.functional as F

MIN_M = 512
NREQ = 8
TIMED_RUNS = 8
MAX_SHAPES = 8
WS_BYTES = 32 << 20
ALIGN = 256
FASTER = 0.98                                  # kept only when at least 2 % faster than torch's own call, in both interleaved timings

CTR = collections.Counter()
_vp, _u64, _i64, _i32, _u32, _f32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32, ctypes.c_uint32, ctypes.c_float
_R_16BF, _R_32F, _COMPUTE_32F = 14, 0, 68                        # cudaDataType CUDA_R_16BF / CUDA_R_32F, cublasComputeType CUBLAS_COMPUTE_32F
_DESC_TRANSA, _DESC_TRANSB, _DESC_EPILOGUE, _DESC_BIAS_POINTER, _DESC_BIAS_DATA_TYPE = 3, 4, 7, 8, 26
_EPILOGUE_BIAS = 4
_PREF_MAX_WORKSPACE_BYTES = 1
_CFG_SPLITK_NUM, _CFG_REDUCTION_SCHEME = 2, 3
_lt = None
FLUSH_BYTES = 128 << 20                        # written before every timed run so each GEMM starts from a cold L2, as inside the forward
_handles: dict = {}                            # device str -> (handle, workspace tensor)
_flush: dict = {}                              # device str -> the L2-flush buffer (allocated for a selection, dropped after it)
_sig: dict = {}                                # signature -> _Entry
_shapes: dict = {}                             # device str -> OrderedDict M -> set of signatures (LRU order)
_counts: dict = {}                             # (N, K, W layout, bias, device) -> number of installed modules with that weight signature on that device
_installed: list = []                          # (module, had_instance_forward, previous) for remove()


TRIAL_FORWARDS = 4                             # candidate, torch, torch, candidate

class _Entry:
    __slots__ = ("state", "calls", "count", "desc", "lA", "lB", "lC", "algo", "M", "N", "trial_f0", "events")

    def __init__(self, count):
        self.state, self.calls, self.count = "seen", 0, count
        self.desc = self.lA = self.lB = self.lC = self.algo = None
        self.M = self.N = 0
        self.trial_f0 = 0                                        # the forward index (0-based, per signature) of the trial's first forward
        self.events = None                                       # trial: TRIAL_FORWARDS lists of (start, end) event pairs


# ---------------------------------------------------------------------------------------------------------------- the library
def _lib():
    global _lt
    if _lt is None:
        found = glob.glob(os.path.join(os.path.dirname(torch.__file__), os.pardir, "nvidia", "cublas", "lib", "libcublasLt.so.*[0-9]"))
        lt = ctypes.CDLL(found[0] if found else "libcublasLt.so.12")
        sig = {"cublasLtCreate": [ctypes.POINTER(_vp)], "cublasLtMatmulDescCreate": [ctypes.POINTER(_vp), _i32, _i32],
               "cublasLtMatmulDescDestroy": [_vp], "cublasLtMatmulDescSetAttribute": [_vp, _i32, _vp, ctypes.c_size_t],
               "cublasLtMatrixLayoutCreate": [ctypes.POINTER(_vp), _i32, _u64, _u64, _i64], "cublasLtMatrixLayoutDestroy": [_vp],
               "cublasLtMatmulPreferenceCreate": [ctypes.POINTER(_vp)], "cublasLtMatmulPreferenceDestroy": [_vp],
               "cublasLtMatmulPreferenceSetAttribute": [_vp, _i32, _vp, ctypes.c_size_t],
               "cublasLtMatmulAlgoGetHeuristic": [_vp, _vp, _vp, _vp, _vp, _vp, _vp, _i32, _vp, ctypes.POINTER(_i32)],
               "cublasLtMatmulAlgoConfigGetAttribute": [_vp, _i32, _vp, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
               "cublasLtMatmul": [_vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, ctypes.c_size_t, _vp]}
        for name, argtypes in sig.items():
            fn = getattr(lt, name); fn.argtypes = argtypes; fn.restype = _i32
        _lt = lt
    return _lt


def _handle(dev):
    key = str(dev)
    if key not in _handles:
        with torch.cuda.device(dev):
            h = _vp(); st = _lib().cublasLtCreate(ctypes.byref(h))
            if st != 0:
                raise RuntimeError(f"cublasLtCreate: status {st}")
            _handles[key] = (h, torch.empty(WS_BYTES, dtype=torch.uint8, device=dev))
    return _handles[key]


def _set(desc, attr, cval):
    st = _lib().cublasLtMatmulDescSetAttribute(desc, attr, ctypes.cast(ctypes.byref(cval), _vp), ctypes.sizeof(cval))
    if st != 0:
        raise RuntimeError(f"cublasLtMatmulDescSetAttribute({attr}): status {st}")


def _cfg_int(algo, attr):
    v = _i32(); w = ctypes.c_size_t()
    st = _lib().cublasLtMatmulAlgoConfigGetAttribute(ctypes.cast(algo, _vp), attr, ctypes.cast(ctypes.byref(v), _vp), 4, ctypes.byref(w))
    return v.value if st == 0 else None


def _layout(t):
    """'C' for a contiguous 2-D (r, c) matrix, 'T' for a transpose-contiguous one, None otherwise."""
    r, c = t.shape
    if t.stride(1) == 1 and (t.stride(0) == c or r == 1):
        return "C"
    if t.stride(0) == 1 and (t.stride(1) == r or c == 1):
        return "T"
    return None


# ---------------------------------------------------------------------------------------------------------------- the call
def _matmul(e, x2, W, bias, out, dev):
    h, ws = _handle(dev)
    if bias is not None:
        _set(e.desc, _DESC_BIAS_POINTER, _vp(bias.data_ptr()))                    # the descriptor is per signature; the bias is this call's
    alpha, beta = _f32(1.0), _f32(0.0)
    st = _lib().cublasLtMatmul(h, e.desc, ctypes.cast(ctypes.byref(alpha), _vp), _vp(W.data_ptr()), e.lA, _vp(x2.data_ptr()), e.lB,
                               ctypes.cast(ctypes.byref(beta), _vp), _vp(out.data_ptr()), e.lC, _vp(out.data_ptr()), e.lC, ctypes.cast(e.algo, _vp),
                               _vp(ws.data_ptr()), ctypes.c_size_t(WS_BYTES), _vp(torch.cuda.current_stream(dev).cuda_stream))
    if st != 0:
        raise RuntimeError(f"cublasLtMatmul: status {st}")


def _gemm(x2, W, bias, xl, wl, torch_call):
    """out (M, N) bf16 = x2 (M, K) @ W (N, K)^T (+ bias) — torch_call() is torch's own computation of exactly that."""
    M, K = x2.shape; N = W.shape[0]; dev = x2.device
    if M < MIN_M or x2.data_ptr() % ALIGN or W.data_ptr() % ALIGN or (bias is not None and (bias.data_ptr() % ALIGN or bias.dtype != torch.bfloat16 or not bias.is_contiguous())):
        CTR["t9_torch_call_small_or_unaligned"] += 1
        return torch_call()
    key = (M, N, K, xl, wl, bias is not None, str(dev))
    e = _sig.get(key)
    if e is None:
        e = _sig[key] = _Entry(_counts.get((N, K, wl, bias is not None, str(dev)), 1))
        _remember(key, M, dev)
    else:
        _touch(M, dev)
    e.calls += 1
    f = (e.calls - 1) // e.count                                                    # this signature's forward index, 0-based
    if e.state == "settled":
        return _lt_out(e, x2, W, bias, M, N, dev)
    if e.state == "torch" or f == 0:                                                # the signature's first forward: torch's call, untouched
        CTR["t9_torch_call"] += 1
        return torch_call()
    if e.state == "seen":
        return _select(e, key, x2, W, bias, torch_call)
    if e.state == "recheck":                                                        # the candidate's second meeting with live operands, beside torch's call
        ref = torch_call()
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        with torch.cuda.device(dev): _matmul(e, x2, W, bias, out, dev)
        CTR["t9_rechecked"] += 1
        if torch.equal(out, ref):
            e.state, e.trial_f0, e.events = "trial", f + 1, [[] for _ in range(TRIAL_FORWARDS)]
            return out
        _retire(e, key); CTR["t9_recheck_failed"] += 1
        print(f"[evo2-opt] T9: cublasLt configuration for GEMM {key[:6]} on {key[6]} differed from torch's output on re-check; that signature keeps torch's call", file=sys.stderr, flush=True)
        return ref
    # state "trial"
    k = f - e.trial_f0
    if k < 0:                                                                       # the rest of forward 2
        CTR["t9_torch_call"] += 1
        return torch_call()
    if k >= TRIAL_FORWARDS:                                                         # forward 7: the decision, then the settled path
        _decide(e, key)
        return _lt_out(e, x2, W, bias, M, N, dev) if e.state == "settled" else torch_call()
    stream = torch.cuda.current_stream(dev)
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    ev[0].record(stream)
    out = _lt_out(e, x2, W, bias, M, N, dev) if k in (0, 3) else torch_call()     # forwards 3, 6: the candidate; 4, 5: torch's call (a drift in clocks weighs on both alike)
    ev[1].record(stream); e.events[k].append(ev); CTR["t9_trial_call"] += 1
    return out


def _lt_out(e, x2, W, bias, M, N, dev):
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    if dev.index == torch.cuda.current_device():
        _matmul(e, x2, W, bias, out, dev)
    else:
        with torch.cuda.device(dev): _matmul(e, x2, W, bias, out, dev)
    CTR["t9_lt_call"] += 1
    return out


def _decide(e, key):
    """After the trial: the candidate's forwards against torch's forwards on this signature's summed in-forward GEMM time."""
    sums = []
    for pairs in e.events:
        total = 0.0
        for a, b in pairs:
            b.synchronize(); total += a.elapsed_time(b)
        sums.append(total)
    e.events = None
    cand, ref = sums[0] + sums[3], sums[1] + sums[2]
    if ref > 0.0 and cand < ref * FASTER:
        e.state = "settled"; CTR["t9_selected"] += 1
    else:
        _retire(e, key); CTR["t9_trial_kept_torch"] += 1


def _retire(e, key):
    _release(e); e.state = "torch"


def _timed(fn, dev):
    """GPU time of TIMED_RUNS calls of fn, each after a write of FLUSH_BYTES (the operands come from DRAM, as they do between the model's kernels)."""
    buf = _flush.get(str(dev))
    if buf is None:
        buf = _flush[str(dev)] = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    total = 0.0
    fn()
    for _ in range(TIMED_RUNS):
        buf.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(torch.cuda.current_stream(dev)); fn(); e1.record(torch.cuda.current_stream(dev))
        e1.synchronize(); total += e0.elapsed_time(e1)
    return total


def _select(e, key, x2, W, bias, torch_call):
    M, N, K, xl, wl, has_bias, _ = key; dev = x2.device; lt = _lib(); h, ws = _handle(dev)
    ref = torch_call()
    with torch.cuda.device(dev):
        desc = _vp()
        if lt.cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _R_32F) != 0:
            e.state = "torch"; CTR["t9_kept_torch"] += 1; return ref
        # torch's column-major call for out = x2 @ W^T: C (N, M) = op(A = W) @ op(B = x2); a contiguous operand is its transpose in column-major
        _set(desc, _DESC_TRANSA, _i32(1 if wl == "C" else 0)); _set(desc, _DESC_TRANSB, _i32(0 if xl == "C" else 1))
        if has_bias:
            _set(desc, _DESC_EPILOGUE, _u32(_EPILOGUE_BIAS)); _set(desc, _DESC_BIAS_POINTER, _vp(bias.data_ptr())); _set(desc, _DESC_BIAS_DATA_TYPE, _i32(_R_16BF))
        lA, lB, lC = _vp(), _vp(), _vp()
        okA = lt.cublasLtMatrixLayoutCreate(ctypes.byref(lA), _R_16BF, *((K, N, K) if wl == "C" else (N, K, N))) == 0
        okB = lt.cublasLtMatrixLayoutCreate(ctypes.byref(lB), _R_16BF, *((K, M, K) if xl == "C" else (M, K, M))) == 0
        okC = lt.cublasLtMatrixLayoutCreate(ctypes.byref(lC), _R_16BF, N, M, N) == 0
        pref = _vp(); okP = lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)) == 0
        res = (ctypes.c_uint8 * (96 * NREQ))(); nret = _i32(0)
        if okA and okB and okC and okP:
            wsv = _u64(WS_BYTES); lt.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.cast(ctypes.byref(wsv), _vp), 8)
            if lt.cublasLtMatmulAlgoGetHeuristic(h, desc, lA, lB, lC, lC, pref, NREQ, ctypes.cast(res, _vp), ctypes.byref(nret)) != 0:
                nret = _i32(0)
        if okP: lt.cublasLtMatmulPreferenceDestroy(pref)
        e.desc, e.lA, e.lB, e.lC, e.M, e.N = desc, lA, lB, lC, M, N
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        best = None
        for i in range(nret.value):
            if struct.unpack("<i", bytes(res[96 * i + 72: 96 * i + 76]))[0] != 0:       # cublasLtMatmulHeuristicResult_t.state
                continue
            algo = (ctypes.c_uint8 * 64).from_buffer_copy(bytes(res[96 * i: 96 * i + 64]))
            if _cfg_int(algo, _CFG_SPLITK_NUM) not in (0, 1) or _cfg_int(algo, _CFG_REDUCTION_SCHEME) != 0:   # one sequential accumulation over k only
                CTR["t9_skipped_splitk"] += 1; continue
            e.algo = algo
            try:
                _matmul(e, x2, W, bias, out, dev)
            except RuntimeError:
                CTR["t9_candidate_error"] += 1; continue
            if not torch.equal(out, ref):
                CTR["t9_candidate_not_equal"] += 1; continue
            t_ref = _timed(torch_call, dev)                                             # torch's call timed beside each candidate (same clocks), torch first
            t = _timed(lambda: _matmul(e, x2, W, bias, out, dev), dev)
            if t < t_ref * FASTER and (best is None or t / t_ref < best[0]):
                best = (t / t_ref, algo)
        if best is not None:                                                            # the winner once more beside torch's call, torch second
            e.algo = best[1]
            t = _timed(lambda: _matmul(e, x2, W, bias, out, dev), dev); t_ref = _timed(torch_call, dev)
            if not t < t_ref * FASTER:
                best = None
    _flush.pop(str(dev), None)
    if best is not None:
        e.algo = best[1]; e.state = "recheck"; CTR["t9_candidate"] += 1
    else:
        _release(e); e.state = "torch"; CTR["t9_kept_torch"] += 1
    return ref


# ---------------------------------------------------------------------------------------------------------------- residency (per device, LRU over M)
def _remember(key, M, dev):
    d = _shapes.setdefault(str(dev), collections.OrderedDict())
    if M not in d:
        d[M] = set()
        while len(d) > MAX_SHAPES:
            _, old_keys = d.popitem(last=False); CTR["t9_shape_evicted"] += 1
            for k in old_keys:
                ent = _sig.pop(k, None)
                if ent is not None: _release(ent)
    d[M].add(key); d.move_to_end(M)


def _touch(M, dev):
    d = _shapes.get(str(dev))
    if d is not None and M in d: d.move_to_end(M)


def _release(e):
    lt = _lib()
    for lay in (e.lA, e.lB, e.lC):
        if lay is not None: lt.cublasLtMatrixLayoutDestroy(lay)
    if e.desc is not None: lt.cublasLtMatmulDescDestroy(e.desc)
    e.desc = e.lA = e.lB = e.lC = e.algo = None


def settled(B: int, L: int, device) -> bool:
    """True when every GEMM signature met at M = B * L on `device` has its final path (a re-checked configuration, or torch's call)."""
    d = _shapes.get(str(torch.device(device)))
    keys = d.get(int(B) * int(L)) if d else None
    return bool(keys) and all(_sig[k].state in ("settled", "torch") for k in keys if k in _sig)


# ---------------------------------------------------------------------------------------------------------------- entry points
def linear(x, W, bias):
    """== F.linear(x, W, bias); bf16 CUDA operands with grad off go through the signature's kept configuration when it has one."""
    if torch.is_grad_enabled() or not x.is_cuda or x.dtype != torch.bfloat16 or W.dtype != torch.bfloat16 or not x.is_contiguous():
        CTR["t9_torch_call_other"] += 1
        return F.linear(x, W, bias)
    K = x.shape[-1]; x2 = x.view(-1, K); wl = _layout(W)
    if wl is None:
        CTR["t9_torch_call_other"] += 1
        return F.linear(x, W, bias)
    out = _gemm(x2, W, bias, "C", wl, lambda: F.linear(x2, W, bias))
    return out.view(*x.shape[:-1], W.shape[0])


def matmul_wt(z, W):
    """== torch.matmul(z, W.t()) for a (1, L, D) bf16 view z whose (L, D) matrix is transpose-contiguous (E50's call on the filter output at B == 1);
    any other z is torch's call."""
    if z.dim() == 3 and z.shape[0] == 1 and not torch.is_grad_enabled() and z.is_cuda and z.dtype == torch.bfloat16 and W.dtype == torch.bfloat16 \
            and _layout(z[0]) == "T" and _layout(W) == "C":
        z2 = z[0]
        return _gemm(z2, W, None, "T", "C", lambda: torch.matmul(z, W.t())[0]).unsqueeze(0)
    CTR["t9_torch_call_other"] += 1
    return torch.matmul(z, W.t())


def _module_forward(self, input):
    return linear(input, self.weight, self.bias)


def modules_of(model) -> list:
    """The nn.Linear modules T9 serves: every block's MLP l1 / l2 / l3 and the attention blocks' Wqkv / out_proj."""
    mods = []
    for b in model.blocks:
        mlp = getattr(b, "mlp", None)
        for n in ("l1", "l2", "l3"):
            l = getattr(mlp, n, None)
            if isinstance(l, torch.nn.Linear): mods.append(l)
        mha = getattr(b, "inner_mha_cls", None)
        for n in ("Wqkv", "out_proj"):
            l = getattr(mha, n, None)
            if isinstance(l, torch.nn.Linear): mods.append(l)
    return mods


def install(model, wrap=None) -> int:
    """Route modules_of(model) through `linear` (an instance-level forward; `wrap(kit_fn, stock_fn)` builds the installed callable, e.g. the
    route's gate). Returns the number of modules installed."""
    n = 0
    for b in model.blocks:                                                          # matmul_wt's signature (E50's call): one per Hyena block per forward
        ofd = getattr(b, "out_filter_dense", None)
        if isinstance(ofd, torch.nn.Linear) and _layout(ofd.weight) is not None:
            k = (ofd.weight.shape[0], ofd.weight.shape[1], _layout(ofd.weight), False, str(ofd.weight.device))
            _counts[k] = _counts.get(k, 0) + 1
    for mod in modules_of(model):
        wl = _layout(mod.weight)
        if wl is not None:
            k = (mod.weight.shape[0], mod.weight.shape[1], wl, mod.bias is not None, str(mod.weight.device))
            _counts[k] = _counts.get(k, 0) + 1
        kit_fn = _module_forward.__get__(mod)
        stock_fn = mod.__dict__.get("forward") or torch.nn.Linear.forward.__get__(mod)
        _installed.append((mod, "forward" in mod.__dict__, mod.__dict__.get("forward")))
        mod.forward = wrap(kit_fn, stock_fn) if wrap else kit_fn
        n += 1
    return n


def remove() -> None:
    for mod, had, prev in _installed:
        if had: mod.forward = prev
        else: mod.__dict__.pop("forward", None)
    _installed.clear(); _counts.clear()
    for e in _sig.values(): _release(e)
    _sig.clear(); _shapes.clear(); CTR.clear()


def is_unpatched() -> bool:
    return not _installed


def counters() -> dict:
    return {k: int(v) for k, v in CTR.items()}
