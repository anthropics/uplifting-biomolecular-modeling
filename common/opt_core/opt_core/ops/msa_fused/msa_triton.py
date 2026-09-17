"""msa_triton.py — Triton kernels behind opm_fused / pwa_fused (see __init__). Numerics class: TOLERANCE (bf16 operands exactly where stock's
autocast Linears use bf16, fp32 LayerNorm statistics / accumulation / softmax-free epilogues; reduction orders differ from cuBLAS)."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------------------------------------------- fused LN + k linears
@triton.jit(do_not_specialize=["R_total", "eps"])
def _ln_linear_kernel(X, LNW, LNB, W, O1, O2, O3, R_total, sx, eps,
                      C: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr, NT: tl.constexpr,
                      R: tl.constexpr, HAS_LNB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < R_total
    offs_c = tl.arange(0, C)
    x = tl.load(X + rows.to(tl.int64)[:, None] * sx + offs_c[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + offs_c).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :]
    if HAS_LNB:
        y = y + tl.load(LNB + offs_c).to(tl.float32)[None, :]
    yb = y.to(tl.bfloat16)                                                  # autocast: the Linear consumes the LN output rounded to bf16
    offs_n = tl.arange(0, NT)
    wt = tl.load(W + offs_n[None, :] * C + offs_c[:, None])                 # W^T tile [C, NT] from W [NT, C] bf16 (row n = output feature)
    acc = tl.dot(yb, wt)                                                    # [R, NT] fp32
    ob = acc.to(tl.bfloat16)
    r64 = rows.to(tl.int64)
    tl.store(O1 + r64[:, None] * N1 + offs_n[None, :], ob, mask=rmask[:, None] & (offs_n[None, :] < N1))
    if N2 > 0:
        tl.store(O2 + r64[:, None] * N2 + (offs_n[None, :] - N1), ob, mask=rmask[:, None] & (offs_n[None, :] >= N1) & (offs_n[None, :] < N1 + N2))
    if N3 > 0:
        tl.store(O3 + r64[:, None] * N3 + (offs_n[None, :] - N1 - N2), ob, mask=rmask[:, None] & (offs_n[None, :] >= N1 + N2))


def ln_linear(x, ln_weight, ln_bias, weights, eps: float = 1e-5, *, R: int = 64, num_warps: int = 4):
    """x [..., C] (bf16|fp32, rows contiguous) -> tuple of bf16 tensors [..., N_k] = Linear_k(LayerNorm(x)) for each W_k [N_k, C] in `weights`
    (no biases; k <= 3). LN in fp32 (weight/bias as given), output rounded to bf16, bf16 x bf16 products with fp32 accumulation, bf16 store =
    the roundings of stock's autocast path. One pass over x."""
    C = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, C)
    assert x2.stride(1) == 1
    Rt = x2.shape[0]
    ws = [w.detach() for w in weights]
    assert 1 <= len(ws) <= 3 and all(w.shape[1] == C for w in ws)
    Wcat = torch.cat([w.to(torch.bfloat16) for w in ws], 0).contiguous()          # [NT, C]
    ns = [w.shape[0] for w in ws] + [0, 0]
    NT = Wcat.shape[0]
    assert NT & (NT - 1) == 0 and C & (C - 1) == 0, (NT, C)
    outs = [torch.empty(lead + (n,), device=x.device, dtype=torch.bfloat16) for n in ns[:len(ws)]]
    optrs = outs + [outs[0]] * (3 - len(outs))
    grid = (triton.cdiv(Rt, R),)
    _ln_linear_kernel[grid](x2, ln_weight, ln_bias if ln_bias is not None else ln_weight, Wcat, optrs[0], optrs[1], optrs[2], Rt, x2.stride(0), float(eps),
                            C=C, N1=ns[0], N2=ns[1], N3=ns[2], NT=NT, R=R, HAS_LNB=ln_bias is not None, num_warps=num_warps)
    return tuple(outs)


# ----------------------------------------------------------------------------------------------------------------- OPM out-projection
@triton.jit(do_not_specialize=["NB", "ND", "sb", "sc", "sd", "snb", "sob"])
def _opm_out_kernel(OUTER, WT, BIAS, NORM, OUT, NB, ND, sb, sc, sd, snb, sob,
                    CH: tl.constexpr, CE: tl.constexpr, CZ: tl.constexpr, ZT: tl.constexpr, TD: tl.constexpr, CG: tl.constexpr, HAS_BIAS: tl.constexpr):
    """out[b, d, z-tile] = bf16( bf16(sum_{c,e} outer[b,c,d,e] * W[z, c*CE+e] + bias[z]) / norm[b, d] ) for one b, TD d's, ZT z's."""
    b = tl.program_id(1)
    d0 = tl.program_id(0) * TD
    z0 = tl.program_id(2) * ZT
    offs_d = d0 + tl.arange(0, TD)
    dmask = offs_d < ND
    offs_k = tl.arange(0, CE * CG)                         # CG head-dim slices per iteration: k = cl*CE + e
    cl = offs_k // CE; ek = offs_k - cl * CE
    offs_z = z0 + tl.arange(0, ZT)
    acc = tl.zeros([TD, ZT], dtype=tl.float32)
    base = OUTER + b.to(tl.int64) * sb
    for c0 in range(0, CH, CG):
        x = tl.load(base + (c0 + cl)[None, :] * sc + offs_d.to(tl.int64)[:, None] * sd + ek[None, :], mask=dmask[:, None], other=0.0)   # [TD, CG*CE]
        w = tl.load(WT + (c0 * CE + offs_k)[:, None] * CZ + offs_z[None, :])                                                          # [CG*CE, ZT]
        acc = tl.dot(x, w, acc)
    if HAS_BIAS:
        acc = acc + tl.load(BIAS + offs_z).to(tl.float32)[None, :]
    v = acc.to(tl.bfloat16).to(tl.float32)
    nrm = tl.load(NORM + b.to(tl.int64) * snb + offs_d, mask=dmask, other=1.0).to(tl.float32)
    v = v / nrm[:, None]
    tl.store(OUT + b.to(tl.int64) * sob + offs_d.to(tl.int64)[:, None] * CZ + offs_z[None, :], v.to(tl.bfloat16), mask=dmask[:, None])


def opm_out(outer, w_out_t, bias, norm, out, *, TD: int = 64, ZT: int | None = None, CG: int = 2, num_warps: int = 4, num_stages: int = 2):
    """outer: the einsum result, logical shape [NB, ND, CH, CE] with ANY strides (stock's is stored b,c,d,e) bf16; w_out_t = linear_out.weight.T
    contiguous [CH*CE, CZ] bf16; bias [CZ] or None; norm [NB, ND] (or [NB, ND, 1]) divisor; out [NB, ND, CZ] bf16 (rows contiguous) written."""
    NB, ND, CH, CE = outer.shape
    CZ = w_out_t.shape[1]
    assert w_out_t.shape[0] == CH * CE and w_out_t.is_contiguous() and w_out_t.dtype == torch.bfloat16
    assert outer.stride(3) == 1 and outer.dtype == torch.bfloat16
    assert out.shape == (NB, ND, CZ) and out.stride(2) == 1 and out.stride(1) == CZ
    nrm = norm.reshape(NB, ND) if norm.dim() == 3 else norm
    assert nrm.stride(1) == 1
    ZT = ZT or CZ
    assert CZ % ZT == 0 and CH % CG == 0
    grid = (triton.cdiv(ND, TD), NB, CZ // ZT)
    _opm_out_kernel[grid](outer, w_out_t, bias if bias is not None else w_out_t, nrm, out, NB, ND,
                          outer.stride(0), outer.stride(2), outer.stride(1), nrm.stride(0), out.stride(0),
                          CH=CH, CE=CE, CZ=CZ, ZT=ZT, TD=TD, CG=CG, HAS_BIAS=bias is not None, num_warps=num_warps, num_stages=num_stages)
    return out


# ----------------------------------------------------------------------------------------------------------------- PWA epilogue
@triton.jit(do_not_specialize=["R_total", "NI", "swm", "swi", "swh", "swc", "sg"])
def _pwa_out_kernel(G, WV, WT, OUT, R_total, NI, swm, swi, swh, swc, sg,
                    H: tl.constexpr, CC: tl.constexpr, CM: tl.constexpr, R: tl.constexpr):
    """rows r = (m, i) flattened: out[r, :] = bf16( sum_k bf16(sigmoid(g[r,k]) * wv[m,i,k//CC,k%CC]) * W[n, k] )."""
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < R_total
    HC: tl.constexpr = H * CC
    offs_k = tl.arange(0, HC)
    r64 = rows.to(tl.int64)
    m_idx = r64 // NI; i_idx = r64 - m_idx * NI
    g = tl.load(G + r64[:, None] * sg + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)                        # torch.sigmoid on a bf16 tensor -> bf16
    hk = offs_k // CC; ck = offs_k - hk * CC
    wv = tl.load(WV + m_idx[:, None] * swm + i_idx[:, None] * swi + hk[None, :] * swh + ck[None, :] * swc, mask=rmask[:, None], other=0.0).to(tl.float32)
    o = (gate * wv).to(tl.bfloat16)                                                            # g * wv in bf16
    offs_n = tl.arange(0, CM)
    wt = tl.load(WT + offs_k[:, None] * CM + offs_n[None, :])                                  # W^T [HC, CM] bf16
    acc = tl.dot(o, wt)
    tl.store(OUT + r64[:, None] * CM + offs_n[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


def pwa_out(g_logits, wv, w_out_t, *, R: int = 64, num_warps: int = 4):
    """g_logits [M, NI, H*CC] bf16 (rows contiguous), wv [M, NI, H, CC] any strides, w_out_t [H*CC, CM] bf16 contiguous -> [M, NI, CM] bf16."""
    M, NI, H, CC = wv.shape
    HC = H * CC
    assert g_logits.shape == (M, NI, HC) and g_logits.stride(2) == 1 and g_logits.stride(0) == NI * g_logits.stride(1)
    CM = w_out_t.shape[1]
    assert w_out_t.shape[0] == HC and w_out_t.is_contiguous() and w_out_t.dtype == torch.bfloat16
    out = torch.empty((M, NI, CM), device=wv.device, dtype=torch.bfloat16)
    Rt = M * NI
    grid = (triton.cdiv(Rt, R),)
    _pwa_out_kernel[grid](g_logits, wv, w_out_t, out, Rt, NI, wv.stride(0), wv.stride(1), wv.stride(2), wv.stride(3), g_logits.stride(1),
                          H=H, CC=CC, CM=CM, R=R, num_warps=num_warps)
    return out


# ----------------------------------------------------------------------------------------------------------------- PWA v2: head-major v
@triton.jit(do_not_specialize=["M", "NTOK", "eps"])
def _pwa_ln_vg_kernel(X, LNW, LNB, W, V, G, M, NTOK, eps,
                      C: tl.constexpr, H: tl.constexpr, CC: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr, HAS_LNB: tl.constexpr):
    """x = m_chunk [M, NTOK, C] rows; tile = MB m's x NB n's. v = Linear_mv(LN(x)) stored HEAD-MAJOR v[h, n, m, c] (so the pair-weighted average is
    one plain bmm per head with contiguous operands), g = Linear_mg(LN(x)) logits stored row-major [M, NTOK, H*CC]."""
    HC: tl.constexpr = H * CC
    pm = tl.program_id(0); pn = tl.program_id(1)
    r = tl.arange(0, MB * NB)
    ni = r // MB; mi = r - ni * MB                                    # row order: m fastest within an n
    mrow = pm * MB + mi; ncol = pn * NB + ni
    rmask = (mrow < M) & (ncol < NTOK)
    offs_c = tl.arange(0, C)
    rowoff = (mrow.to(tl.int64) * NTOK + ncol) * C
    x = tl.load(X + rowoff[:, None] + offs_c[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / C
    y = xc * (1.0 / tl.sqrt(var + eps))[:, None] * tl.load(LNW + offs_c).to(tl.float32)[None, :]
    if HAS_LNB:
        y = y + tl.load(LNB + offs_c).to(tl.float32)[None, :]
    yb = y.to(tl.bfloat16)
    offs_o = tl.arange(0, 2 * HC)
    wt = tl.load(W + offs_o[None, :] * C + offs_c[:, None])                      # [C, 2*HC]: cols 0..HC-1 = mv, HC.. = mg
    acc = tl.dot(yb, wt).to(tl.bfloat16)                                          # [MB*NB, 2*HC]
    # g logits, row-major [M, NTOK, HC]
    growoff = (mrow.to(tl.int64) * NTOK + ncol) * HC
    tl.store(G + growoff[:, None] + (offs_o[None, :] - HC), acc, mask=rmask[:, None] & (offs_o[None, :] >= HC))
    # v head-major [H, NTOK, M, CC]
    hk = offs_o // CC; ck = offs_o - hk * CC
    voff = hk[None, :].to(tl.int64) * NTOK * M * CC + ncol[:, None].to(tl.int64) * M * CC + mrow[:, None].to(tl.int64) * CC + ck[None, :]
    tl.store(V + voff, acc, mask=rmask[:, None] & (offs_o[None, :] < HC))


def pwa_ln_vg(m, ln_weight, ln_bias, w_mv, w_mg, eps=1e-5, *, H=8, CC=8, MB=8, NB=16, num_warps=8):
    """m [M, NTOK, C] bf16|fp32 contiguous -> (v_hm [H, NTOK, M, CC] bf16, g_logits [M, NTOK, H*CC] bf16)."""
    M, NTOK, C = m.shape
    assert m.is_contiguous()
    HC = H * CC
    assert w_mv.shape == (HC, C) and w_mg.shape == (HC, C)
    Wcat = torch.cat([w_mv.detach().to(torch.bfloat16), w_mg.detach().to(torch.bfloat16)], 0).contiguous()
    v = torch.empty((H, NTOK, M, CC), device=m.device, dtype=torch.bfloat16)
    g = torch.empty((M, NTOK, HC), device=m.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, MB), triton.cdiv(NTOK, NB))
    _pwa_ln_vg_kernel[grid](m, ln_weight, ln_bias if ln_bias is not None else ln_weight, Wcat, v, g, M, NTOK, float(eps),
                            C=C, H=H, CC=CC, MB=MB, NB=NB, HAS_LNB=ln_bias is not None, num_warps=num_warps)
    return v, g


@triton.jit(do_not_specialize=["M", "NTOK"])
def _pwa_out2_kernel(G, WV, WT, OUT, M, NTOK,
                     H: tl.constexpr, CC: tl.constexpr, CM: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    """out[m, i, :] = Linear_out( bf16(sigmoid(g[m,i,:]) * wv[h, i, m, c]) ), tile = MB m's x NB i's; wv head-major [H, NTOK, M, CC]."""
    HC: tl.constexpr = H * CC
    pm = tl.program_id(0); pn = tl.program_id(1)
    r = tl.arange(0, MB * NB)
    ii = r // MB; mi = r - ii * MB
    mrow = pm * MB + mi; icol = pn * NB + ii
    rmask = (mrow < M) & (icol < NTOK)
    offs_k = tl.arange(0, HC)
    growoff = (mrow.to(tl.int64) * NTOK + icol) * HC
    g = tl.load(G + growoff[:, None] + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    hk = offs_k // CC; ck = offs_k - hk * CC
    wvoff = hk[None, :].to(tl.int64) * NTOK * M * CC + icol[:, None].to(tl.int64) * M * CC + mrow[:, None].to(tl.int64) * CC + ck[None, :]
    wv = tl.load(WV + wvoff, mask=rmask[:, None], other=0.0).to(tl.float32)
    o = (gate * wv).to(tl.bfloat16)
    offs_n = tl.arange(0, CM)
    wt = tl.load(WT + offs_k[:, None] * CM + offs_n[None, :])
    acc = tl.dot(o, wt)
    orow = (mrow.to(tl.int64) * NTOK + icol) * CM
    tl.store(OUT + orow[:, None] + offs_n[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


def pwa_out2(g_logits, wv_hm, w_out_t, *, MB=8, NB=8, num_warps=4):
    """g_logits [M, NTOK, H*CC] bf16, wv_hm [H, NTOK, M, CC] bf16 contiguous, w_out_t [H*CC, CM] bf16 -> [M, NTOK, CM] bf16."""
    H, NTOK, M, CC = wv_hm.shape
    HC = H * CC; CM = w_out_t.shape[1]
    assert wv_hm.is_contiguous() and g_logits.shape == (M, NTOK, HC) and g_logits.is_contiguous()
    assert w_out_t.shape[0] == HC and w_out_t.is_contiguous() and w_out_t.dtype == torch.bfloat16
    out = torch.empty((M, NTOK, CM), device=wv_hm.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, MB), triton.cdiv(NTOK, NB))
    _pwa_out2_kernel[grid](g_logits, wv_hm, w_out_t, out, M, NTOK, H=H, CC=CC, CM=CM, MB=MB, NB=NB, num_warps=num_warps)
    return out
