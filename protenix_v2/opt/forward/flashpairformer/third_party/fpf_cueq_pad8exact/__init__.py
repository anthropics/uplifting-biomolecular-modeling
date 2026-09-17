# SPDX-License-Identifier: Apache-2.0
"""fpf_cueq_pad8exact — FlashPairformer attention PROVIDER `cueq_pad8exact` (EXACT class) for the v0.4 BLK2 padded-core provider interface (PTX_BLK_PADDED=8).

What it does: calls the STOCK cuEquivariance triangle_attention (the same cudnn_sm80_fprop_sdpa_fp16_64x64xD<true,32> kernel Protenix already runs on cc 8.x / 9.x)
on the block core's P = ceil8(N)-PADDED q/k/v buffers with the padded keys excluded by -1e9 bias columns (softmax weight exactly 0 in fp32) and NO boolean mask.
On H100 this is BITWISE IDENTICAL to the stock call (checked per cell) and 12-17 % faster per call than the unpadded mask=None call
(x1.5 vs the stock call with Protenix's dense all-True bool mask, whose per-call mask/bias materialisation it also removes).  No new kernel: DISPATCH + GATE + COUNTERS only.
Why faster (same kernel template instance <true, 32> in both cases per the profiler): with S % 8 == 0 the kernel's K/V tiles and bias rows are fully populated / 16-byte aligned,
so the ragged-tail handling of the 705-wide rows disappears; we did not disassemble the kernel, so treat this as the plausible mechanism, not a measured one.

LABEL: EXACT on cc 8.x / 9.x ONLY (sm80 kernel).  On cc >= 10 with the cu13 cuEquivariance build an aligned unmasked call is routed to cudnn_sm100_fprop (different bits —
that is precisely the NOMASK hazard / the Tier-2 `cueq_sm100f` provider), so this provider REFUSES on cc >= 10.  N % 8 == 0: no-op (the stock call is already aligned; served as-is).

REGISTRY ENTRY (what ptx_trunk2_levers v0.4 imports):
    PROVIDER = {"fn": attn, "name": "cueq_pad8exact", "label": "EXACT", "kv_pad": 8, "bias_pad_fill": -1e9, "wants_kv_len": True, "zero_kv_pad": True,
                "min_tokens": 104, "max_tokens": None, "available": available, "describe": describe, "report": report}
CALL:  o = attn(q5, k5, v5, bias4, mask=None, scale=scale, kv_len=N, q_len=N)
       q5,k5,v5 : [1, P|I, H, P, D] bf16/fp16 views of the core's ZERO-INITIALISED padded buffers (strides (.., H*P*D, P*D, D, 1)); I may be a row chunk (rows of q,k,v sliced together)
       bias4    : [1, H, P, P] fp32 (preferred: identical to what stock passes) or bf16; columns >= kv_len hold bias_pad_fill written ONCE by the core — or 5-D [1,1,H,P,P]
       returns  : [1, I|P, H, P, D] (rows >= q_len / cols >= kv_len are don't-care)
       UNPADDED square calls (S % 8 != 0, no kv_len) are also accepted: served by a per-call pad COPY (bitwise identical, slower; counted as served_padcopy).
GATE (fail-closed; one decision line per process "[FPF] PAD8EXACT: ..."; census line at exit):
    G1 cc major in {8, 9} [FPF_PAD8X_CC_MAJORS]  — cc >= 10 REFUSED (sm100f hazard), cc 7.x refused (never checked)
    G2 cuequivariance_torch importable (any version with triangle_attention(q,k,v,bias,mask=,scale=); checked 0.8.0 and 0.11.1)
    G3 per call: bf16/fp16; last-dim stride 1; padded S % 8 == 0 (or square unpadded -> pad-copy); mask None or all-True; k/v rows == q rows;
       kv_len >= FPF_PAD8X_MIN_KV (104): below ~100 keys cuEq does not launch the cudnn kernel but a torch fallback whose bits change with padding (N=89: max|d| 0.016) -> Refused, stock serves
ENV: FPF_PAD8X=0 (off) | FPF_PAD8X_CC_MAJORS=8,9 | FPF_PAD8X_MIN_KV=104 | FPF_PAD8X_VERBOSE=1
HAZARDS: see HAZARDS.md (zero-filled buffers are MANDATORY: NaN/Inf in K/V pad rows poisons ALL valid outputs; -1e9 not -inf; provider owns the mask;
cu13 + cc>=10 refuses; the EXACT label is per (cuEq version, cc) as tested — a cuEq upgrade must re-run tests/cert_pad8exact.py).
Credit: kernel = NVIDIA cuEquivariance / cuDNN; padded-buffer core + interface = FlashPairformer integrator v0.4; found by the FPF-Blackwell/SM100 track's H100 vendor probe (2026-08-23).
"""
from __future__ import annotations
import os, math, time, atexit
import torch
import torch.nn.functional as F

__version__ = "0.1.0"
BIAS_PAD_FILL = -1e9
_STATS = {"calls": 0, "served_view": 0, "served_padcopy": 0, "served_aligned": 0, "refused": {}, "decision": None, "pad_hist": {}}
_STATE = {"disabled": os.environ.get("FPF_PAD8X", "1") == "0", "checked": False, "ok_static": False, "why": ""}
_VERBOSE = os.environ.get("FPF_PAD8X_VERBOSE", "0") == "1"
_CC_MAJORS = {int(x) for x in os.environ.get("FPF_PAD8X_CC_MAJORS", "8,9").split(",") if x}
_MIN_KV = int(os.environ.get("FPF_PAD8X_MIN_KV", "104"))
#   cuEq's small-N torch-fallback threshold is not exported and may move between versions -> a moved threshold fails CLOSED (Refused -> stock) instead of serving non-bitwise output.     # cuEq serves short sequences (observed: N=89) with a torch bmm/softmax FALLBACK, not the cudnn_sm80 kernel; padding changes
#   the fallback GEMM shapes -> NOT bitwise (max|d| 0.016 @89). First size checked on the CUDA kernel = 110 -> refuse kv_len < 104 (= padded S < 112). Stock serves those (tiny) calls.


class Refused(RuntimeError):
    """Raised when the gate refuses a call; .reason is a short token. The block core routes to stock."""
    def __init__(self, reason, detail=""):
        super().__init__(f"cueq_pad8exact refused: {reason} {detail}".strip()); self.reason = reason


try:
    import cuequivariance_torch as _cet
    _cue_tri = getattr(_cet, "triangle_attention", None)
    if _cue_tri is None:
        from cuequivariance_torch.primitives.triangle import triangle_attention as _cue_tri
except Exception as _e:
    _cet = None; _cue_tri = None; _STATE["disabled"] = True; _STATE["why"] = f"cuequivariance_torch import failed: {_e!r}"


def _say(msg):
    print(f"[FPF] PAD8EXACT: {msg}", flush=True)


def _refuse(reason, detail=""):
    _STATS["refused"][reason] = _STATS["refused"].get(reason, 0) + 1
    raise Refused(reason, detail)


def _static_check(device) -> bool:
    if _STATE["checked"]:
        return _STATE["ok_static"]
    _STATE["checked"] = True
    why = []; cc = None
    try:
        cc = tuple(torch.cuda.get_device_capability(device))
        if cc[0] not in _CC_MAJORS:
            why.append(f"cc {cc}: EXACT pad-8 is supported for the sm80 cuEq kernel on cc majors {sorted(_CC_MAJORS)} only" + (" (cc>=10: aligned unmasked calls reach cudnn_sm100 = different bits)" if cc[0] >= 10 else ""))
    except Exception as e:
        why.append(f"cc query failed {e!r}")
    try:
        import cuequivariance_torch as cet
        _STATS["versions"] = {"cuequivariance_torch": getattr(cet, "__version__", "?"), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(device) if cc else None, "cc": list(cc) if cc else None}
    except Exception as e:
        why.append(f"cuequivariance_torch import failed {e!r}")
    _STATE["ok_static"] = not why; _STATE["why"] = "; ".join(why)
    _STATS["decision"] = "ELIGIBLE" if not why else "DISABLED: " + _STATE["why"]
    _say(_STATS["decision"])
    return _STATE["ok_static"]


def available(device="cuda") -> bool:
    """Apply-time check for the registry (G1+G2, no tensors needed)."""
    return (not _STATE["disabled"]) and _cue_tri is not None and _static_check(torch.device(device))


def _pad8(n): return (n + 7) // 8 * 8


def _call(q, k, v, b5, scale, kv_len):
    """q,k,v [B,I,H,P,D]; b5 [B,1,H,P,P] fp32 or q.dtype; columns >= kv_len must hold BIAS_PAD_FILL (core contract; enforced here if the core did not tag the buffer)."""
    if kv_len < q.shape[3] and not getattr(b5, "_fpf_padfilled", False):
        b5 = b5.clone(); b5[..., kv_len:] = BIAS_PAD_FILL
    o = _cue_tri(q, k, v, b5, mask=None, scale=scale)
    return o[0] if isinstance(o, (tuple, list)) else o




def attn(q, k, v, bias, mask=None, scale=None, kv_len=None, q_len=None):
    """Provider entry (v0.4 signature). Raises Refused when the gate says no (the core routes to stock)."""
    _STATS["calls"] += 1
    if _STATE["disabled"] or _cue_tri is None:
        _refuse("disabled", _STATE["why"])
    if not (torch.is_tensor(q) and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype and v.dtype == q.dtype):
        _refuse("dtype_or_device")
    if not _static_check(q.device):
        _refuse("static_gate", _STATE["why"])
    squeeze4 = False
    if q.dim() == 4:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0); squeeze4 = True
    b5 = bias
    if b5.dim() == 3: b5 = b5.unsqueeze(0)
    if b5.dim() == 4: b5 = b5.unsqueeze(1) if b5.shape[0] == q.shape[0] else b5.unsqueeze(0)
    if q.dim() != 5 or k.dim() != 5 or v.shape != k.shape or q.shape[0] != 1 or k.shape[0] != 1 or q.shape[2] != k.shape[2] or q.shape[4] != k.shape[4]:
        _refuse("rank_or_shape", f"q{tuple(q.shape)} k{tuple(k.shape)}")
    B, I, H, S, D = q.shape
    if k.shape[3] != S or b5.dim() != 5 or b5.shape[-1] != S or b5.shape[-2] != S or b5.shape[-3] != H:
        _refuse("bias_or_kv_shape", f"k{tuple(k.shape)} bias{tuple(b5.shape)}")
    if b5.dtype not in (torch.float32, q.dtype):
        _refuse("bias_dtype", str(b5.dtype))
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        _refuse("stride")
    if mask is not None:
        try:
            all_true = bool(mask.to(torch.bool).all().item())          # one device sync per masked call; the v0.4 core passes mask=None
        except Exception:
            all_true = False
        if not all_true: _refuse("dense_mask", "only mask=None / all-True is exact-supported; padded keys are excluded by the bias columns")
    if scale is None: scale = 1.0 / math.sqrt(D)
    kv_len = int(kv_len) if kv_len is not None else S
    q_len = int(q_len) if q_len is not None else (I if I != S else S)
    if kv_len < _MIN_KV:
        _refuse("small_n_torch_fallback", f"kv_len={kv_len} < {_MIN_KV}: cuEq uses its torch fallback path there (padding not bitwise)")
    if k.shape[1] != I:
        _refuse("kv_rows_mismatch", f"q rows {I} vs k rows {k.shape[1]} (chunk rows of q,k,v together)")
    if q_len != kv_len and I == S:
        _refuse("q_len_ne_kv_len")
    if S % 8 == 0:
        if kv_len == S:                                   # already aligned problem (N % 8 == 0): the stock call IS the aligned call -> serve as-is (no-op lever)
            o = _cue_tri(q, k, v, b5, mask=None, scale=scale); o = o[0] if isinstance(o, (tuple, list)) else o
            _STATS["served_aligned"] += 1
            return o[0] if squeeze4 else o
        if kv_len > S or S - kv_len >= 8:
            _refuse("bad_kv_len", f"kv_len={kv_len} S={S}")
        o = _call(q, k, v, b5, scale, kv_len); _STATS["served_view"] += 1; _STATS["pad_hist"][str(S - kv_len)] = _STATS["pad_hist"].get(str(S - kv_len), 0) + 1
        if _VERBOSE and _STATS["served_view"] == 1:
            _say(f"first served call (padded-buffer path): q{tuple(q.shape)} kv_len={kv_len} bias={b5.dtype}")
        return o[0] if squeeze4 else o
    # ---- unpadded square caller (S % 8 != 0): pad-COPY path (drop-in; slower; bitwise identical)
    if I != S: _refuse("ragged_unpadded", "chunked callers must use the padded buffers")
    P = _pad8(S); pn = P - S
    qP = F.pad(q, (0, 0, 0, pn, 0, 0, 0, pn)); kP = F.pad(k, (0, 0, 0, pn, 0, 0, 0, pn)); vP = F.pad(v, (0, 0, 0, pn, 0, 0, 0, pn))
    bP = F.pad(b5, (0, pn, 0, pn)); bP[..., S:] = BIAS_PAD_FILL; bP._fpf_padfilled = True
    o = _call(qP, kP, vP, bP, scale, S)[:, :S, :, :S, :]
    _STATS["served_padcopy"] += 1; _STATS["pad_hist"][str(pn)] = _STATS["pad_hist"].get(str(pn), 0) + 1
    if _VERBOSE and _STATS["served_padcopy"] == 1:
        _say(f"first served call (pad-copy path): q{tuple(q.shape)} -> P={P}")
    return o[0] if squeeze4 else o


def attn_or(fallback_fn, name="stock"):
    """Drop-in wrapper: try attn, on Refused call fallback_fn(q,k,v,bias,mask=,scale=) and count it."""
    def f(q, k, v, bias, mask=None, scale=None, **kw):
        try:
            return attn(q, k, v, bias, mask=mask, scale=scale, **kw)
        except Refused as r:
            d = _STATS.setdefault("fallback_used", {}); d[f"{name}:{r.reason}"] = d.get(f"{name}:{r.reason}", 0) + 1
            return fallback_fn(q, k, v, bias, mask=mask, scale=scale)
    return f


triangle_attention = attn
fn = attn


def report() -> dict:
    return {"version": __version__, **_STATS, "state": {k: (dict(v) if isinstance(v, dict) else v) for k, v in _STATE.items()}}


def describe() -> dict:
    return {"name": "cueq_pad8exact", "version": __version__, "label": "EXACT", "kv_pad": 8, "bias_pad_fill": BIAS_PAD_FILL, "wants_kv_len": True, "zero_kv_pad": True,
            "kernel": "stock cuequivariance triangle_attention (cudnn_sm80_fprop_sdpa_fp16_64x64xD) on 8-aligned padded buffers", "cc_majors": sorted(_CC_MAJORS), "min_kv": _MIN_KV,
            "tested": {"cuequivariance_torch": ["0.8.0", "0.11.1"], "cc": ["9.0 (H100 80GB HBM3)"]}}


PROVIDER = {"fn": attn, "name": "cueq_pad8exact", "label": "EXACT", "kv_pad": 8, "bias_pad_fill": BIAS_PAD_FILL, "wants_kv_len": True, "zero_kv_pad": True,
            "min_tokens": _MIN_KV, "max_tokens": None, "available": available, "describe": describe, "report": report}


def _exit_line():
    if _STATS["calls"]:
        _say("census " + " ".join(f"{k}={_STATS[k]}" for k in ("calls", "served_view", "served_padcopy", "served_aligned")) + f" refused={_STATS['refused']} pad_hist={_STATS['pad_hist']}")
atexit.register(_exit_line)
