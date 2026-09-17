"""lnl_fused.py — Triton fusion kernels for the pairformer's non-triangle pair ops, LayerNorm fused into the consuming linear (H100 / A100, bf16 pair stream under bf16 autocast).

1. ln_linear(x, ln_w, ln_b, W[NOUT,C], eps, write_y, transpose, planes)  ->  (y16 = bf16(LN(x)) written in [b,i,j] or transposed [b,j,i] row order | None,  out = bf16(y16 @ W^T) [.., NOUT],
   or plane-major [NOUT, ..] with planes=True)
   Used for AttentionPairBias  (B = to_b(ln_0(Z)); y not written; planes: the bias planes an attention kernel indexes without a gather)  and the TriangleAttention prologue
   (y = norm(pair) as bf16, transposed for the ending node; bias = to_b(y)).  This engine's form of the kernel (the plane-major store); launched with ONE fixed tile (_LN_LINEAR_TILE,
   the tile this kernel has run with on every measured card, cc 9.0 / 8.0) -- no candidate tile is timed at run time.
2. gate_transpose: the shared core's kernel (opt_core.kernels.lnl_fused.gate_transpose -- one copy, imported here for this module's callers).
All kernels: last dim C in {64, 128} (constexpr), rows flattened, CUDA only. Fallback decisions are made by the caller.
"""
import torch, triton, triton.language as tl
from opt_core.kernels.lnl_fused import gate_transpose  # noqa: E402,F401  -- 2.: the shared core's ONE copy of the gate/transpose kernel (same statement this module carried), re-exported for this module's callers


_LCFG = [triton.Config({"BM": 64}, num_warps=4, num_stages=2), triton.Config({"BM": 128}, num_warps=4, num_stages=2), triton.Config({"BM": 128}, num_warps=8, num_stages=2)]


@triton.jit
def _ln_linear_kernel(X, Y, OUT, LNW, LNB, W, M, MB, I, J, eps, C: tl.constexpr, NOUT: tl.constexpr, WRITE_Y: tl.constexpr, TRANSPOSE: tl.constexpr, BM: tl.constexpr, PLANES: tl.constexpr = False):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, C)
    r64 = rows.to(tl.int64)
    x = tl.load(X + r64[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + cols).to(tl.float32)
    b = tl.load(LNB + cols).to(tl.float32)
    y16 = (xc * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if WRITE_Y:
        if TRANSPOSE:
            bidx = r64 // (I * J); rem = r64 - bidx * (I * J); ii = rem // J; jj = rem - ii * J
            dst = bidx * (I * J) + jj * I + ii
        else:
            dst = r64
        tl.store(Y + dst[:, None] * C + cols[None, :], y16, mask=rmask[:, None])
    ncols = tl.arange(0, NOUT)
    wt = tl.load(W + ncols[None, :] * C + cols[:, None])            # [C, NOUT] = W^T
    o = tl.dot(y16, wt)                                             # fp32 [BM, NOUT]
    if PLANES:                                                      # plane-major [NOUT, M]: output channel n of every row contiguous (an attention bias operand [H, I, J] read by unit stride)
        tl.store(OUT + ncols[None, :].to(tl.int64) * M + r64[:, None], o.to(tl.bfloat16), mask=rmask[:, None])
    else:
        tl.store(OUT + r64[:, None] * NOUT + ncols[None, :], o.to(tl.bfloat16), mask=rmask[:, None])


_LN_LINEAR_TILE = {"BM": 64, "num_warps": 4, "num_stages": 2}                 # the ONE launch tile of this kernel (the tile it has run with on every measured card, cc 9.0 / 8.0: _LCFG's first entry); nothing is timed

def ln_linear(x, ln_w, ln_b, W, eps=1e-5, write_y=False, transpose=False, planes=False):
    """x [B, I, J, C] or [I, J, C] (contiguous, bf16/fp32); W [NOUT, C] bf16 with NOUT a power of two >= 16 (pad by the caller).
    Returns (y16 | None, out): y16 = bf16(LN(x)) as [B, I, J, C] (or [B, J, I, C] holding LN(x)[b,i,j] at [b,j,i] when transpose), out = bf16(y16 @ W^T) [B, I, J, NOUT] (never transposed);
    planes=True stores the same values plane-major, out [NOUT, B, I, J] (channel n of every row contiguous: an attention bias [.., H, I, J] taken as out[:H] without a gather)."""
    squeeze = x.dim() == 3
    x4 = x.unsqueeze(0) if squeeze else x
    B, I, J, C = x4.shape; NOUT = W.shape[0]
    xs = x4.contiguous().view(-1, C); M = xs.shape[0]
    out = torch.empty((NOUT, B, I, J) if planes else (B, I, J, NOUT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((B, J, I, C) if transpose else (B, I, J, C), device=x.device, dtype=torch.bfloat16) if write_y else out
    MB = 0 if M < 65536 else (1 if M < 400000 else 2)
    grid = (triton.cdiv(M, _LN_LINEAR_TILE["BM"]),)
    _ln_linear_kernel[grid](xs, y, out, ln_w, ln_b, W, M, MB, I, J, eps, C=C, NOUT=NOUT, WRITE_Y=bool(write_y), TRANSPOSE=bool(transpose), PLANES=bool(planes), **_LN_LINEAR_TILE)
    if squeeze:
        out = out[:, 0] if planes else out[0]; y = y[0] if write_y else None
    return (y if write_y else None), out


def census():
    """The fixed launch tile of this module's kernel (one entry; nothing is timed or stepped)."""
    return {"ln_linear": dict(_LN_LINEAR_TILE)}
