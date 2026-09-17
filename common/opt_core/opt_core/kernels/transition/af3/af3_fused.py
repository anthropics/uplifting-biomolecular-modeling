# af3_fused.py -- fused LayerNorm + SwiGLU + Linear transition kernel for xfold (AF3-PyTorch), H100, inference only.
# Derived from the support library's lnl_fused.py::fused_transition;
# generalised to non-power-of-two channel widths (c_single = 384) by padding the register tile to CP = next_pow2(C) with masked loads.
#
#   y   = LN(x) (fp32 statistics over the C real channels) -> bf16                       [stock: F.layer_norm fp32 out, cast to bf16 by the autocast matmul]
#   a|g = y @ W1^T | y @ W2^T  (bf16 MMA, fp32 acc) -> bf16                              [stock: torch.matmul(x, transition1.weight.T) under autocast -> bf16; a = first half, g = second half]
#   h   = bf16( bf16(silu(a)) * g )                                                       [stock: F.silu(a) * b on bf16 tensors (fp32 opmath, bf16 results)]
#   out = h @ W3^T (fp32 acc over the whole hidden dim) -> bf16                           [stock: transition2 under autocast]
# The HID-wide intermediate never touches HBM. Rounding points identical to stock-under-bf16-autocast; accumulation ORDER differs from cuBLAS
# -> same numerics class, not bitwise (tier 2). Deterministic (no atomics / split-K); autotuned tile choice is made once per (M-bucket, C, HID) OUTSIDE graph capture.
import torch
import triton
import triton.language as tl


@triton.jit
def _ft_kernel(X, Y, LNW, LNB, W1, W2, W3, M, MB, eps,
               C: tl.constexpr, CP: tl.constexpr, HID: tl.constexpr, RESIDUAL: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, CP)
    cmask = cols < C
    m2 = rmask[:, None] & cmask[None, :]
    r64 = rows.to(tl.int64)
    x = tl.load(X + r64[:, None] * C + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = tl.where(m2, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(LNB + cols, mask=cmask, other=0.0).to(tl.float32)
    y16 = (xc * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)      # padded columns are exactly 0
    acc = tl.zeros([BM, CP], dtype=tl.float32)
    for h0 in range(0, HID, BH):
        hcols = h0 + tl.arange(0, BH)
        w1t = tl.load(W1 + hcols[None, :] * C + cols[:, None], mask=cmask[:, None], other=0.0)   # [CP, BH] = W1[hchunk, :]^T
        w2t = tl.load(W2 + hcols[None, :] * C + cols[:, None], mask=cmask[:, None], other=0.0)
        a = tl.dot(y16, w1t)                                                               # fp32 [BM, BH]
        g = tl.dot(y16, w2t)
        a16 = a.to(tl.bfloat16).to(tl.float32)                                             # stock: bf16 GEMM outputs
        g16 = g.to(tl.bfloat16).to(tl.float32)
        s16 = (a16 / (1.0 + tl.exp(-a16))).to(tl.bfloat16).to(tl.float32)                 # silu in fp32 on the bf16 value, rounded to bf16 (torch opmath)
        h16 = (s16 * g16).to(tl.bfloat16)                                                  # bf16 * bf16 -> rounded to bf16
        w3t = tl.load(W3 + cols[None, :] * HID + hcols[:, None], mask=cmask[None, :], other=0.0)  # [BH, CP] = W3[:, hchunk]^T
        acc += tl.dot(h16, w3t)
    if RESIDUAL:
        outv = (x + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    else:
        outv = acc.to(tl.bfloat16)
    tl.store(Y + r64[:, None] * C + cols[None, :], outv, mask=m2)


_CFG_POW2 = [triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2), triton.Config({"BM": 64, "BH": 64}, num_warps=4, num_stages=2),
             triton.Config({"BM": 64, "BH": 128}, num_warps=4, num_stages=2), triton.Config({"BM": 128, "BH": 128}, num_warps=8, num_stages=1),
             triton.Config({"BM": 128, "BH": 64}, num_warps=4, num_stages=3), triton.Config({"BM": 128, "BH": 128}, num_warps=8, num_stages=2),
             triton.Config({"BM": 256, "BH": 64}, num_warps=8, num_stages=2)]
_CFG_PAD = [triton.Config({"BM": 32, "BH": 64}, num_warps=4, num_stages=1), triton.Config({"BM": 16, "BH": 64}, num_warps=4, num_stages=1),
            triton.Config({"BM": 32, "BH": 32}, num_warps=4, num_stages=1), triton.Config({"BM": 64, "BH": 64}, num_warps=8, num_stages=1),
            triton.Config({"BM": 64, "BH": 128}, num_warps=8, num_stages=2), triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2)]
_ft_pow2 = triton.autotune(configs=_CFG_POW2, key=["MB", "C", "HID", "RESIDUAL"])(_ft_kernel)
_ft_pad = triton.autotune(configs=_CFG_PAD, key=["MB", "C", "HID", "RESIDUAL"])(_ft_kernel)

SUPPORTED_C = (64, 128, 256, 384)


def fused_transition(x, ln_w, ln_b, W1, W2, W3, eps=1e-5, residual=False):
    """x [..., C] (bf16 or fp32, CUDA), W1/W2 [HID, C] bf16 contiguous, W3 [C, HID] bf16 contiguous -> bf16 [..., C].
    out = W3 @ (silu(W1 LN(x)) * (W2 LN(x)))  (+ x if residual)."""
    C = x.shape[-1]; HID = W1.shape[0]
    assert W1.shape == (HID, C) and W2.shape == (HID, C) and W3.shape == (C, HID), (W1.shape, W2.shape, W3.shape)
    assert HID % 64 == 0
    xs = x.contiguous().view(-1, C); M = xs.shape[0]
    y = torch.empty((M, C), device=x.device, dtype=torch.bfloat16)
    MB = 0 if M < 65536 else (1 if M < 400000 else 2)
    CP = triton.next_power_of_2(C)
    kern = _ft_pow2 if (CP == C and C <= 128) else _ft_pad
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)
    kern[grid](xs, y, ln_w, ln_b, W1, W2, W3, M, MB, eps, C=C, CP=CP, HID=HID, RESIDUAL=bool(residual))
    return y.view(x.shape)


def reference_transition(x, ln_w, ln_b, W1, W2, W3, eps=1e-5, dtype=torch.float64):
    F = torch.nn.functional
    xx = x.to(dtype)
    y = F.layer_norm(xx, (x.shape[-1],), ln_w.to(dtype), ln_b.to(dtype), eps)
    return F.linear(F.silu(F.linear(y, W1.to(dtype))) * F.linear(y, W2.to(dtype)), W3.to(dtype))
