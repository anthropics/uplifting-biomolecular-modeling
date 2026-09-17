"""af3t_opm.py — Triton kernels for the MSA module's OuterProductMean (xfold @22bdeed nn/primitives.py, PyTorch AlphaFold3), inference only,
bf16 activations under bf16 autocast (the trunk's MSA stream).  Written for this kit (af3t_*: kit-namespaced; nothing here imports the kit).

The stock statement (S msa rows, N tokens, c_m = 64, c = 32 outer channels, c_z = 128):

    x    = LayerNorm(msa)                                   [S, N, 64]   fp32 (autocast runs layer_norm in fp32)
    l, r = mask * Linear_l(x), mask * Linear_r(x)           [S, N, 32]   bf16 GEMMs (fp32 accumulate, bf16 out), fp32 after the mask product
    act  = einsum('acb,ade->dceb', l^T, r)                  [N, 32, 32, N] bf16  — N^2 * 1024 elements (1.4 GB at 832 tokens, 3 GB at 1216) + one
    act  = einsum('dceb,cef->dbf', act, W_out) + b_out      [N, N, 128]  fp32    strided copy of it for the second GEMM's operand layout
    out  = act^T / (eps + einsum('abc,adc->bdc', mask, mask))            fp32  (the norm GEMM runs in bf16 under autocast; eps + norm is a bf16 add)

The kernels (same rounding points as the statement above — LN in fp32, bf16 GEMM operands, fp32 accumulation, bf16 GEMM outputs, fp32 epilogue;
the fp32 summation ORDER inside the LayerNorm statistics, the c_m = 64 projections and the 1024-long output contraction differs: tolerance-class,
not bitwise; deterministic run to run):

  ln_proj2(msa, ln_w, ln_b, WT, mask) -> (LT [N, 32, S], R [S, N, 32]) bf16      ONE pass over the MSA stream: LayerNorm + both projections + the mask
      product, the left operand written transposed ([token, channel, row]: the layout the outer-product GEMM reads as its row-major A operand, so
      a CHUNK of left tokens is a contiguous row range — no operand copies), the right operand row-major.
  (the outer product itself is ONE cuBLAS bf16 GEMM per chunk of left tokens: T[b, c, d, e] = LT[b, c, :] . R[:, d, e] — issued by the caller)
  opm_out(T, W16, bias, norm16, eps, b0, pair | out)                              the output contraction over (c, e) = 1024 with W_out read as 32
      [32, 128] bf16 tiles (L2-resident), + b_out, / (eps + norm) with the statement's bf16 rounding of the denominator, and EITHER the fp32 update
      written (the module contract) OR the block's `pair += update` folded: pair rows read, added in fp32, written back in place as bf16 (the
      residual form is bitwise the statement `pair += update` applied to this kernel's own update).  T is read exactly once, contiguously; the
      [N, N, 1024]-element intermediate of the stock statement is never permuted or copied and exists only one chunk at a time.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_proj2_kernel(X, LNW, LNB, WT, MASK, LT, R, S, N, eps,
                     C: tl.constexpr, CO: tl.constexpr, BA: tl.constexpr):
    # one program = BA msa rows (a) of ONE token column b: x rows (a, b) are N*C elements apart, each C contiguous
    pid = tl.program_id(0)
    nA = tl.cdiv(S, BA)
    b = pid // nA
    ab = pid % nA
    a = ab * BA + tl.arange(0, BA)                       # [BA] msa row indices
    am = a < S
    cols = tl.arange(0, C)
    co = tl.arange(0, CO)
    row = a.to(tl.int64) * N + b                         # flattened (a, b) row index, int64 (S*N*C can pass 2^31)
    x = tl.load(X + row[:, None] * C + cols[None, :], mask=am[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt_rn(var + eps)
    w = tl.load(LNW + cols).to(tl.float32)
    bb = tl.load(LNB + cols).to(tl.float32)
    y16 = (xc * rstd[:, None] * w[None, :] + bb[None, :]).to(tl.bfloat16)      # autocast: the fp32 LN output cast to bf16 at the Linear
    m = tl.load(MASK + row, mask=am, other=0.0).to(tl.float32)                  # [BA] the msa mask of these rows
    wl = tl.load(WT + cols[:, None] * (2 * CO) + co[None, :])                   # [C, CO] bf16 = W_l^T
    wr = tl.load(WT + cols[:, None] * (2 * CO) + CO + co[None, :])              # [C, CO] bf16 = W_r^T
    l = tl.dot(y16, wl)                                                         # fp32 accumulate
    r = tl.dot(y16, wr)
    l16 = (l.to(tl.bfloat16).to(tl.float32) * m[:, None]).to(tl.bfloat16)       # bf16 GEMM output, fp32 mask product, bf16 at the einsum (the statement's chain)
    r16 = (r.to(tl.bfloat16).to(tl.float32) * m[:, None]).to(tl.bfloat16)
    # left operand transposed: LT[b, c, a]  (row-major [N*CO, S]: the outer-product GEMM's A operand, left-token chunks = contiguous row ranges)
    tl.store(LT + (b * CO + co[None, :]).to(tl.int64) * S + a[:, None], l16, mask=am[:, None])
    # right operand row-major: R[a, b, e]  (row-major [S, N*CO]: the GEMM's B operand)
    tl.store(R + row[:, None] * CO + co[None, :], r16, mask=am[:, None])


@triton.jit
def _opm_out_kernel(T, W, BIAS, NORM, DST, N, b0, ld_t, eps,
                    CO: tl.constexpr, F: tl.constexpr, BD: tl.constexpr, RESIDUAL: tl.constexpr):
    # one program = one left token b (local row block bl of the chunk's T) x BD right tokens d: out[b, d, :] = sum_{c,e} T[bl, c, d, e] W[c, e, :]
    bl = tl.program_id(0)
    dt = tl.program_id(1)
    d = dt * BD + tl.arange(0, BD)
    dm = d < N
    e = tl.arange(0, CO)
    f = tl.arange(0, F)
    acc = tl.zeros((BD, F), dtype=tl.float32)
    t_row = T + (bl * CO).to(tl.int64) * ld_t + (d[:, None] * CO + e[None, :])     # T row (bl, c=0), columns (d, e): BD*CO contiguous elements
    for c in range(0, CO):
        t = tl.load(t_row + c * ld_t, mask=dm[:, None], other=0.0)                  # [BD, CO] bf16
        w = tl.load(W + (c * CO + e)[:, None] * F + f[None, :])                     # [CO, F] bf16 = bf16(W_out)[c]
        acc = tl.dot(t, w, acc)                                                     # fp32 accumulate over (c, e)
    b = b0 + bl
    v = acc.to(tl.bfloat16).to(tl.float32) + tl.load(BIAS + f).to(tl.float32)[None, :]   # bf16 GEMM output + fp32 bias -> fp32
    nrm = tl.load(NORM + b.to(tl.int64) * N + d, mask=dm, other=1.0).to(tl.float32)      # bf16 norm (exact integer counts)
    den = (nrm + eps).to(tl.bfloat16).to(tl.float32)                                    # `eps + norm` is a bf16 add in the statement (python scalar + bf16 tensor)
    v = tl.math.div_rn(v, den[:, None] + tl.zeros((BD, F), dtype=tl.float32))           # fp32 IEEE division, as torch's
    dst = DST + (b.to(tl.int64) * N + d[:, None]) * F + f[None, :]
    if RESIDUAL:
        p = tl.load(dst, mask=dm[:, None], other=0.0).to(tl.float32)                    # pair[b, d, :] bf16: `pair += update` = bf16(fp32(pair) + update)
        tl.store(dst, (p + v).to(tl.bfloat16), mask=dm[:, None])
    else:
        tl.store(dst, v, mask=dm[:, None])                                              # the module contract: the fp32 update


def ln_proj2(msa, ln_w, ln_b, WT, mask, eps, LT=None, R=None, BA=64, num_warps=4):
    """msa [S, N, C] (bf16 | fp32, contiguous), ln_w / ln_b [C] fp32, WT [C, 2*CO] bf16 (= cat(W_l, W_r).T), mask [S, N] (any float dtype) ->
    LT [N, CO, S] bf16 (left operand, transposed), R [S, N, CO] bf16."""
    S, N, C = msa.shape
    CO = WT.shape[1] // 2
    if LT is None: LT = torch.empty((N, CO, S), device=msa.device, dtype=torch.bfloat16)
    if R is None: R = torch.empty((S, N, CO), device=msa.device, dtype=torch.bfloat16)
    grid = (N * triton.cdiv(S, BA),)
    _ln_proj2_kernel[grid](msa, ln_w, ln_b, WT, mask, LT, R, S, N, eps, C=C, CO=CO, BA=BA, num_warps=num_warps)
    return LT, R


def opm_out(T, W16, bias, norm16, eps, b0, nb, N, dst, residual, BD=64, num_warps=4):
    """T [>= nb*CO, N*CO] bf16 row-major (rows (b_local, c), cols (d, e)); W16 [CO*CO, F] bf16; bias [F] fp32; norm16 [N, N] bf16; dst = pair
    [N, N, F] bf16 (residual=True: += in place) or out [N, N, F] fp32; rows b0 .. b0+nb of dst are written."""
    CO = W16.shape[0]; F = W16.shape[1]; CO = int(round(CO ** 0.5))
    grid = (nb, triton.cdiv(N, BD))
    _opm_out_kernel[grid](T, W16, bias, norm16, dst, N, b0, T.stride(0), eps, CO=CO, F=F, BD=BD, RESIDUAL=bool(residual), num_warps=num_warps)
    return dst
