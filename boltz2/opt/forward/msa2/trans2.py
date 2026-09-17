"""msa2.trans2 — fused Boltz-2 Transition for dim 64 / hidden 256 (the MSA transition `MSALayer.msa_transition` on m [B,S,N,64] and the template
pairformer's transition): ONE Triton kernel per call — LayerNorm(64) -> [fc1|fc2] (64->256 each) -> SiLU(a)*b -> fc3 (256->64) — row-tiled, the hidden
dimension processed in on-chip chunks so neither the fp32 LN output nor the [rows,256] hidden activations ever reach HBM (stock, chunked at N>384:
transition.py:67-78 = 1 LN pass + 8 x (2 casts + 2 GEMMs + silu + mul + GEMM + add) = ~40 passes over m-sized tensors per call).

Numerics classes (constexpr MODE):
  MODE=0 'fast'  : LN fp32 -> bf16 operand; fc1/fc2 bf16 x bf16 -> fp32; SiLU(a)*b in fp32 -> bf16 operand; fc3 partials accumulated in FP32 across the
                   hidden chunks; one bf16 rounding at the end. Differs from stock only by rounding placement (stock rounds fc1/fc2 outputs, the SiLU, the
                   product, every fc3 partial and every partial sum to bf16) — strictly fewer roundings. Tier 2.
  MODE=1 'mimic' : every stock rounding point reproduced (a,b -> bf16; silu(a) = a/(1+expf(-a)) in fp32 -> bf16; *b -> bf16; fc3 partial (K=chunk) -> bf16;
                   running sum bf16(x_out + partial)); hidden chunk = the caller's chunk_size
                   (32 at inference). Bitwise = stock IFF the tensor-core K-reduction
                   order of tl.dot equals cuBLAS's for K=64 / K=32 bf16 -> fp32 — an EMPIRICAL property checked per (shape class, card, library pin) by
                   tests/test_msa2_trans2.py::test_mimic_bitwise; the adapter serves MODE=1 in the exact tier only when that check has passed on the card
                   (else the unit reports state=skipped reason=not_bitwise and the stock statements serve).
Guards (adapter): eval, CUDA autocast bf16 active (the regime whose dtypes this reproduces: x bf16 in, bf16 out), x.shape[-1]==64, hidden==256, x bf16
contiguous rows. `do_not_specialize` on the row count: one compiled kernel per MODE/BM (no first-of-class JIT cliff per input size).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


try:
    from triton.language.extra import libdevice as _ld
except Exception:                                            # older layouts
    from triton.language.extra.cuda import libdevice as _ld


@triton.jit
def _silu_mul(a, b, MODE: tl.constexpr):
    if MODE == 1:
        # stock: x_chunk = silu(bf16(fc1)) * bf16(fc2); ATen silu on bf16 = float(x) / (1.0f + expf(-float(x))) in fp32 (opmath), rounded to bf16;
        # the product = float(silu16) * float(b16) rounded to bf16. expf = CUDA math library = libdevice __nv_expf; '/' must be IEEE div.rn.f32.
        a16 = a.to(tl.bfloat16).to(tl.float32)
        b16 = b.to(tl.bfloat16).to(tl.float32)
        s = tl.div_rn(a16, 1.0 + _ld.exp(-a16))
        s16 = s.to(tl.bfloat16).to(tl.float32)
        return (s16 * b16).to(tl.bfloat16)
    else:
        s = a / (1.0 + tl.exp(-a))
        return (s * b).to(tl.bfloat16)


@triton.jit(do_not_specialize=["M"])
def _trans2_kernel(X, OUT, LNW, LNB, W1, W2, W3, M, eps,
                   C: tl.constexpr, H: tl.constexpr, HC: tl.constexpr, BM: tl.constexpr, MODE: tl.constexpr, LN_IN_KERNEL: tl.constexpr):
    # X: [M, C] bf16 or fp32 row-major contiguous (loaded in its own dtype, LN statistics in fp32 = stock nn.LayerNorm on either); OUT: [M, C] bf16; LNW/LNB: [C] fp32; W1, W2: [H, C] bf16 (nn.Linear weight layout); W3: [C, H] bf16.
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = tl.arange(0, C)
    xp = X + rows[:, None] * C + cols[None, :]
    x = tl.load(xp, mask=rmask[:, None], other=0.0).to(tl.float32)                     # [BM, C]
    if LN_IN_KERNEL:
        # LayerNorm over C (biased variance, eps inside rsqrt: torch.nn.LayerNorm) — fast class (not torch's Welford tree: not bitwise)
        mean = tl.sum(x, axis=1) / C
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        lnw = tl.load(LNW + cols).to(tl.float32); lnb = tl.load(LNB + cols).to(tl.float32)
        xh = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]
        xb = xh.to(tl.bfloat16)                                                          # the autocast operand cast, once
    else:
        xb = x.to(tl.bfloat16)                                                           # X = the stock nn.LayerNorm's fp32 output; autocast's RNE cast
    hk = tl.arange(0, HC)
    if MODE == 1:
        out16 = tl.zeros((BM, C), dtype=tl.bfloat16)
    acc = tl.zeros((BM, C), dtype=tl.float32)
    for h0 in tl.static_range(0, H, HC):
        # W1[h0:h0+HC, :] is [HC, C]; the dot wants [C, HC] = its transpose: load with swapped index roles
        w1t = tl.load(W1 + (h0 + hk)[None, :] * C + cols[:, None])                          # [C, HC]  (element (c, k) = W1[h0+k, c])
        w2t = tl.load(W2 + (h0 + hk)[None, :] * C + cols[:, None])
        a = tl.dot(xb, w1t)                                                              # [BM, HC] fp32
        b = tl.dot(xb, w2t)
        g = _silu_mul(a, b, MODE)                                                        # [BM, HC] bf16
        w3t = tl.load(W3 + cols[None, :] * H + (h0 + hk)[:, None])                          # [HC, C]  (element (k, c) = W3[c, h0+k])
        if MODE == 1:
            part = tl.dot(g, w3t).to(tl.bfloat16)                                        # fc3 slice GEMM -> bf16 (cuBLAS epilogue rounding)
            if h0 == 0:
                out16 = part
            else:
                out16 = (out16.to(tl.float32) + part.to(tl.float32)).to(tl.bfloat16)     # x_out = x_out + partial (bf16 add = fp32 add, RNE)
        else:
            acc = tl.dot(g, w3t, acc)
    if MODE == 1:
        res = out16
    else:
        res = acc.to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * C + cols[None, :], res, mask=rmask[:, None])


_WCACHE = {}


def _weights(module):
    """bf16 contiguous copies of fc1/fc2/fc3 weights + fp32 LN affine, cached per module (inference: weights frozen). Autocast's cached_cast makes the
    same bf16 copies of the leaf weights; .to(bfloat16) is that conversion."""
    key = id(module)
    ent = _WCACHE.get(key)
    w1 = module.fc1.weight
    if ent is not None and ent["ref"] is module and ent["ptr"] == w1.data_ptr() and ent["dev"] == w1.device:
        return ent
    ent = {"ref": module, "ptr": w1.data_ptr(), "dev": w1.device,
           "w1": module.fc1.weight.detach().to(torch.bfloat16).contiguous(), "w2": module.fc2.weight.detach().to(torch.bfloat16).contiguous(),
           "w3": module.fc3.weight.detach().to(torch.bfloat16).contiguous(),
           "lnw": module.norm.weight.detach().float().contiguous(), "lnb": module.norm.bias.detach().float().contiguous(), "eps": float(module.norm.eps)}
    _WCACHE[key] = ent
    return ent


def supported(module, x, chunk_size) -> str | None:
    """None if trans2 can serve this Transition call, else the fallback word."""
    if x.shape[-1] != 64 or module.fc1.weight.shape != (256, 64) or module.fc3.weight.shape != (64, 256):
        return "dims"
    if x.dtype not in (torch.bfloat16, torch.float32):      # m is bf16 in an MSAModule call's first block and fp32 after it (the fp32 dropout-mask
        return "dtype"                                        # residual promotes it): the kernel loads either and computes LN in fp32 as stock does
    if not x.is_cuda:
        return "device"
    if getattr(module.norm, "weight", None) is None or getattr(module.norm, "bias", None) is None:
        return "ln_affine"
    return None


def transition64(module, x: torch.Tensor, chunk_size=None, mode: int = 0, BM: int = 64, HC: int = 32, num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """Transition(dim=64, hidden=256).forward(x, chunk_size) -> bf16, one kernel. mode 0 = fast class; mode 1 = stock-rounding mimic (hidden chunk =
    chunk_size or 32)."""
    C, H = 64, 256
    shp = x.shape
    if mode == 1:
        x = module.norm(x)                                    # the stock statement (torch's LayerNorm kernel, fp32 out under autocast): bitwise by identity
    x2 = x.reshape(-1, C)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    w = _weights(module)
    out = torch.empty((M, C), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out.reshape(shp)
    hc = HC if mode == 0 else (int(chunk_size) if chunk_size else H)   # mimic: the caller's hidden chunking (32 at N>384) or the unchunked path (one K=256 fc3 GEMM)
    grid = (triton.cdiv(M, BM),)
    _trans2_kernel[grid](x2, out, w["lnw"], w["lnb"], w["w1"], w["w2"], w["w3"], M, w["eps"], C=C, H=H, HC=hc, BM=BM, MODE=mode,
                         LN_IN_KERNEL=(mode == 0), num_warps=num_warps, num_stages=num_stages)
    return out.reshape(shp)
