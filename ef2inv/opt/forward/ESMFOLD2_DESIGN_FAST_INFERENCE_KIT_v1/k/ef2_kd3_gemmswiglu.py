"""ef2_kd3_gemmswiglu — K-D3 lean's W12 projection with the SwiGLU as its epilogue: ONE Triton kernel computes, per [BM, BN] tile of the
hidden activation, both halves of lin = x̂ @ W12ᵀ (two fp32 accumulators over the W1 and the W2 columns of the same rows, K = d walked in
BK steps from zero, rounded to bf16 exactly where cuBLAS rounds its bf16 output) and applies K-D3's SwiGLU rounding chain
(bf16(silu(x1)) · x2 -> bf16) in registers, storing only hidden [M, h].  The [M, 2h] bf16 pre-activation — at h = 4·d the widest tensor
of the pair block, 8 pair-tensor passes to write and 8 to read back — is never written to memory in forward (grad forwards, checkpoint
recomputes and no-grad forwards of an agk-enabled trunk alike).  The W3 GEMM, the LayerNorm kernels, the residual add, the saved tensors,
K-D3 lean's backward (LN + W12 recomputed on cuBLAS, the SwiGLU backward in place) and the memory plan are K-D3's, unchanged.

Numerics class FAST — K-D3's own ('reordered accumulation, same rounding points'): every rounding point of K-D3 / stock is kept (lin rounded
to bf16 before the SiLU; SiLU output bf16; products bf16); the only freedom is the fp32 accumulation order of the d-term dot products inside
the Triton MMA tiles against cuBLAS's, i.e. lin may differ from cuBLAS's by one bf16 ulp on a small fraction of elements.
A card is given an entry below only where the kernel's outputs are bit-identical to K-D3's own at d = 256 under the
det recipe.

Cards.  Where K-D3 lean's forward serves — i.e. where no out-only forward kernel does (fast / big on compute capability 9.0 run that
forward on the t16 kernel, ef2_t16_transition, and never ask this lever: its LEVER line reads ``state=off reason=t16_serves``) — the
launch comes from a per-compute-capability table, ``_CFG_BY_CC``: sm_80 (A100) carries a tile tuned there; a capability without an
entry keeps K-D3's cuBLAS projection + SwiGLU kernel BY NAME (``state=stepped_aside reason=cc_untuned:sm_NN`` — never silent, never a
refusal).

Surface.  The patch point is the module-level name ``ef2_autograd_kernels._kd3_w12_swiglu`` that K-D3's ``TransitionRefround.forward``
reads per call on its own-kernels branch (lean only; process-wide; bind BEFORE any CUDA-graph capture of the trunk; idempotent; reversible):
    engage(cc=None) -> describe()      bind the hook when this card has an entry, else record why not (never raises for a card without one)
    live() / disable() / stats() / describe()
    hidden_for(xhat, w) -> Tensor|None the hook: hidden [M, h] bf16 = SwiGLU(x̂ W12ᵀ), or None — counted by name — for a weight entry it
                                       does not take (d not a power of two in [32, 256], h not a multiple of the tile, W12 not bf16 row-major);
                                       K-D3's cuBLAS projection + SwiGLU kernel then serve that call.
"""
from __future__ import annotations

import torch
from torch import Tensor

import ef2_autograd_kernels as agk

__all__ = ["engage", "disable", "live", "stats", "describe", "hidden_for", "w12_swiglu_fwd", "KERNELS_OK", "NAME", "VERSION"]

NAME = "ef2_kd3_gemmswiglu"
VERSION = "w12swiglu.1"

try:
    import triton
    import triton.language as tl
    from ef2_autograd_kernels import _k_exp, _k_div          # K-D3's exp / round-to-nearest division helpers (same libdevice route)

    @triton.jit
    def _k_tile_ids(M, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
        """program id -> (row tile, column tile), rows grouped GROUP tiles at a time so the programs that share an x̂ row tile (every
        column tile of it) and the whole W12 run close together (L2 reuse: x̂ is read from HBM ~once)."""
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        num_n: tl.constexpr = H // BN
        width: tl.constexpr = GROUP * num_n
        group_id = pid // width
        first_m = group_id * GROUP
        gsz = tl.minimum(num_m - first_m, GROUP)
        pid_m = first_m + ((pid % width) % gsz)
        pid_n = (pid % width) // gsz
        return pid_m, pid_n

    @triton.jit
    def _k_w12_swiglu_fwd(x_ptr, w_ptr, out_ptr, M, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          GROUP: tl.constexpr, NSTAGE: tl.constexpr):
        """hidden[M, H] = bf16( f32(bf16(silu(bf16(x̂ W1ᵀ)))) · f32(bf16(x̂ W2ᵀ)) ) ; x̂ [M, D] bf16 row-major, W12 = [W1; W2] [2H, D] bf16 row-major.
        One program = one [BM, BN] tile of hidden: two fp32 accumulators (the W1 and the W2 columns of the same rows), K walked in BK steps."""
        pid_m, pid_n = _k_tile_ids(M, H, BM, BN, GROUP)
        rows = pid_m.to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
        rmask = rows < M
        cols = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        acc1 = tl.zeros((BM, BN), dtype=tl.float32)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, D, BK, num_stages=NSTAGE):
            ks = (k0 + tl.arange(0, BK)).to(tl.int64)
            xk = tl.load(x_ptr + rows[:, None] * D + ks[None, :], mask=rmask[:, None], other=0.0)     # [BM, BK]
            w1 = tl.load(w_ptr + cols[None, :] * D + ks[:, None])                                    # [BK, BN] = W1[cols, ks]ᵀ
            w2 = tl.load(w_ptr + (cols[None, :] + H) * D + ks[:, None])                              # [BK, BN] = W2[cols, ks]ᵀ
            acc1 = tl.dot(xk, w1, acc1)
            acc2 = tl.dot(xk, w2, acc2)
        a = acc1.to(tl.bfloat16).to(tl.float32)                                                       # lin is a bf16 tensor (cuBLAS out)
        b = acc2.to(tl.bfloat16).to(tl.float32)
        sv = _k_div(a, 1.0 + _k_exp(-a))                                                             # ATen silu in opmath fp32
        s_bf = sv.to(tl.bfloat16).to(tl.float32)                                                     # F.silu output is bf16
        hv = s_bf * b                                                                                 # bf16*bf16 -> fp32 -> bf16
        tl.store(out_ptr + rows[:, None] * H + cols[None, :], hv.to(tl.bfloat16), mask=rmask[:, None])

    KERNELS_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    KERNELS_OK = False
    _IMPORT_ERR = _e

# The launch table, one entry per compute capability (d = 256, h = 1024): one program per [bm, bn] hidden tile — two fp32 [bm, bn]
# accumulators, K walked in bk steps with `stages`-deep software pipelining, `group` row tiles swizzled for L2 reuse, `warps` warps.
# sm_80 (torch 2.11 / triton 3.6): tensor-equal to cuBLAS W12 + the SwiGLU kernel at d = 256 (64 KB of shared memory: 4 stages of a 128x32 x̂
# tile + two 32x64 weight tiles).  No sm_90 entry: fast / big run K-D3 lean's forward on the t16 kernel there (ef2_t16_transition).
_CFG_SM80 = dict(bm=128, bn=64, bk=32, group=8, warps=4, stages=4)
_CFG_BY_CC = {(8, 0): _CFG_SM80}
CC_UNTUNED = "cc_untuned"
_STATE = dict(engaged=None, reason=None, reason_text=None, device=None, cc=None, cfg=None, prev=None)
STATS = {"calls": 0, "fallthrough_d": 0, "fallthrough_h": 0, "fallthrough_w12_layout": 0}


def _device_cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def cfg_for(cc=None):
    """(tile entry, None) for compute capability `cc` (default: the current device's), or (None, 'cc_untuned:sm_NN') when this kit carries
    no entry for it — (None, 'no_cuda') without a device, (None, 'kernels_unavailable') when Triton did not import."""
    if not KERNELS_OK:
        return None, "kernels_unavailable"
    cc = _device_cc() if cc is None else tuple(cc)
    if cc is None:
        return None, "no_cuda"
    e = _CFG_BY_CC.get(cc)
    return (dict(e), None) if e is not None else (None, f"{CC_UNTUNED}:sm_{cc[0]}{cc[1]}")


def tile_word(cfg=None) -> str:
    c = cfg or _STATE.get("cfg") or {}
    return f"{c['bm']}x{c['bn']}x{c['bk']}g{c['group']}w{c['warps']}s{c['stages']}" if c else "none"


def w12_swiglu_fwd(xhat: Tensor, w12: Tensor, cfg: dict | None = None) -> Tensor:
    """hidden [M, h] bf16 = SwiGLU(x̂ @ W12ᵀ) with K-D3's rounding points; x̂ [M, d] bf16 contiguous, W12 [2h, d] bf16 contiguous; `cfg` = a
    tile entry (default: the engaged one, else this card's, else sm_80's — the k/ test passes its own)."""
    c = cfg or _STATE.get("cfg") or cfg_for()[0] or _CFG_SM80
    M, D = xhat.shape; H = w12.shape[0] // 2
    out = torch.empty((M, H), dtype=torch.bfloat16, device=xhat.device)
    grid = (triton.cdiv(M, c["bm"]) * (H // c["bn"]),)
    _k_w12_swiglu_fwd[grid](xhat, w12, out, M, H=H, D=D, BM=c["bm"], BN=c["bn"], BK=c["bk"], GROUP=c["group"], NSTAGE=c["stages"], num_warps=c["warps"])
    return out


def _take(w: dict, d: int, c: dict) -> str | None:
    """None when the fused kernel serves K-D3's weight entry `w` at pair width `d` under tile `c`, else the fall-through word."""
    if d < 32 or d > 256 or (d & (d - 1)) != 0 or d % c["bk"] != 0:
        return "d"
    W12 = w["W12"]
    if (W12.shape[0] // 2) % c["bn"] != 0:
        return "h"
    if W12.dtype != torch.bfloat16 or not W12.is_contiguous():
        return "w12_layout"
    return None


def hidden_for(xhat: Tensor, w: dict):
    """The hook K-D3's forward calls (ef2_autograd_kernels._kd3_w12_swiglu): hidden [M, h] bf16 in one kernel, or None — counted by name in
    stats()['fallback'] — for an entry the kernel does not take; K-D3's cuBLAS W12 projection + SwiGLU kernel then serve the call."""
    c = _STATE["cfg"]
    why = _take(w, xhat.shape[1], c) if c is not None else "not_live"
    if why is not None:
        STATS[f"fallthrough_{why}"] = STATS.get(f"fallthrough_{why}", 0) + 1
        return None
    STATS["calls"] += 1
    return w12_swiglu_fwd(xhat, w["W12"], c)


def engage(cc=None, cfg: dict | None = None) -> dict:
    """Bind K-D3's forward hook to the fused kernel when this card has an entry (``cfg`` overrides it: the k/ test); idempotent.
    A card without an entry (or a stack whose Triton did not import) is RECORDED, not raised: live() is then False, K-D3's cuBLAS projection +
    SwiGLU kernel serve and describe() names the reason (the lever steps aside by name).  Returns describe()."""
    entry, reason = cfg_for(cc)
    if cfg is not None and KERNELS_OK:
        entry, reason = dict(cfg), None
    _STATE["device"] = torch.cuda.get_device_name().replace(" ", "_") if torch.cuda.is_available() else None
    _STATE["cc"] = _device_cc() if cc is None else tuple(cc)
    if entry is None:
        _STATE.update(engaged=False, reason=reason, reason_text=(repr(_IMPORT_ERR) if reason == "kernels_unavailable" else f"{NAME}: no launch entry for this device ({reason}); K-D3's own W12 projection + SwiGLU kernel serve"), cfg=None)
        return describe()
    if agk.__dict__.get("_kd3_w12_swiglu") is not hidden_for:
        _STATE["prev"] = agk.__dict__.get("_kd3_w12_swiglu")
        agk._kd3_w12_swiglu = hidden_for
    _STATE.update(engaged=True, reason=None, reason_text=None, cfg=entry)
    return describe()


def disable(model=None) -> None:
    """Restore K-D3's own forward chain (the hook as found); the counters stay."""
    if agk.__dict__.get("_kd3_w12_swiglu") is hidden_for:
        agk._kd3_w12_swiglu = _STATE.get("prev")
    _STATE.update(engaged=None, reason=None, reason_text=None, cfg=None, prev=None)


def live() -> bool:
    """The hook is bound to this module's kernel in this process (engage() said so)."""
    return bool(_STATE["engaged"]) and agk.__dict__.get("_kd3_w12_swiglu") is hidden_for


def stats() -> dict:
    """served = fused-kernel launches from K-D3's forward (grad + no-grad + recomputes); fallback = the named fall-through counters (non-zero only)."""
    fb = {k[len("fallthrough_"):]: int(v) for k, v in STATS.items() if k.startswith("fallthrough_") and v}
    return dict(served=int(STATS.get("calls", 0)), fallback=fb, installed=live())


def describe() -> dict:
    """The lever's record: state (``on`` | ``stepped_aside`` | ``off`` = never asked), reason (a word) + reason_text, version, device, cc word,
    the tile word and the counters."""
    state = "on" if live() else ("stepped_aside" if _STATE["engaged"] is False else "off")
    cc = _STATE.get("cc")
    return dict(name=NAME, state=state, reason=_STATE["reason"], reason_text=_STATE["reason_text"], version=VERSION, device=_STATE["device"],
                cc=(f"sm_{cc[0]}{cc[1]}" if cc else None), tile=(tile_word() if state == "on" else None), **stats())


def reset() -> None:
    """Test helper: forget the engagement (the hook restored) and zero the counters."""
    disable()
    for k in list(STATS):
        STATS[k] = 0
