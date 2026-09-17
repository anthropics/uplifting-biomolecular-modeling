"""Triton kernels of the ESM-family pair TriangleMultiplication line, re-tiled for other pair widths.

Structure (unchanged from the line these kernels derive from; see NOTICE):

    K1  LN_in(z) + [a|b] projections + sigmoid gates + mask   -> zero-padded channel-major bf16 planes ab[2*D, Np, Np]
    B   cuBLAS strided-batched bmm over the D channels (bf16 operands, fp32 accumulate) -> x[D, Np, Np] (bf16, or fp32 with lever x32)
    K3  LN_out(x^T) + out-projection + recomputed LN_in(z) output gate (+ residual) -> token-major output in z's dtype

What is new here (everything is a compile-time switch; nothing is read from the environment, nothing falls back silently):

    channel chunks   the pair width C (LN_in, the K1 projections' K, the K3 gate's K and the K3 output-channel loop) and the hidden width
                     CH (LN_out, the K3 out-projection's K) are walked as NCK x CK / NCH x CHK power-of-two chunks held resident, so
                     C, CH in {64, 128, 256, 384, 512} run natively (384 = 3 x 128: no padding of the reduction to 512).  With one chunk
                     (CK == C, CHK == CH) K1 / K3 are the derived-from kernels statement for statement.
    io dtype         z (and the output, and the residual) may be bf16 or fp32: LN statistics are fp32 either way, every MMA is
                     bf16 x bf16 -> fp32 accumulate, the gates / mask / residual add are fp32, the planes are bf16.  fp32 in/out is the
                     fp32-activations / bf16-tensor-core form for fp32 and TF32 trunks (a tolerance-class substitute of their fp32 GEMMs).
    descriptors      K1 reads the z tile and the weight blocks through tensor descriptors (kernels_desc._k1s_tma; cc 9.0, needs
                     tl.make_tensor_descriptor) or through pointer loads (_k1s here; cc 8.0 and any triton without the descriptor API) --
                     same arithmetic, same bytes out; the cell's k1.tma flag picks, the module with descriptors is imported only then.
    RESIDUAL         K3 adds z (fp32 add) or not: the no-residual epilogue is the forward-only entry cofolding trunks call.
    levers           sigmoid (tanh.approx gate), stagger (weight-block start offset per CTA), incnt / form (contraction form NT|TN with
                     the planes written in the orientation the form needs), x32 (fp32 contraction output into K3), stock_round.

Launch cells (BM / BN / num_warps / num_stages per kernel, chunk widths, contraction form) come from TILE_TABLES.json through the
package face; this module takes them as arguments and never chooses.
"""
import torch
import triton
import triton.language as tl

MAX_CHUNKS = 4
_STATE = dict(alloc=False)
STATS = dict(calls=0, k1=0, k3=0, trans_calls=0, tn_calls=0)


def has_descriptor_api():
    return hasattr(tl, "make_tensor_descriptor") and hasattr(triton, "set_allocator")


def has_bmm_out_dtype():
    try:
        import inspect
        return "out_dtype" in inspect.signature(torch.bmm).parameters
    except (TypeError, ValueError):          # C builtins without a signature: probe the overload text
        return "out_dtype" in (torch.bmm.__doc__ or "")


# =====================================================================================================================
# device helpers
# =====================================================================================================================
@triton.jit
def _sigmoid_tanh(x):
    # sigmoid(x) = 0.5 * tanh(x / 2) + 0.5 ; tanh.approx.f32: 1 MUFU op, |rel err| <= 2^-10.7
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x * 0.5], dtype=tl.float32, is_pure=True, pack=1)
    return 0.5 * t + 0.5


@triton.jit
def _gate(x, FASTSIG: tl.constexpr):
    if FASTSIG:
        return _sigmoid_tanh(x)
    else:
        return tl.sigmoid(x)


# ---- shared device helpers: two-pass LayerNorm over resident channel chunks, and the K1 gate/mask/store epilogue --------------------------
@triton.jit
def _ln_rows(z0, z1, z2, z3, w_ptr, b_ptr, ck, eps, C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr):
    """[BM, CK] fp32 chunks (tokens x channels; chunks k >= NCK are aliases of chunk 0 and unused) -> LN -> bf16 chunks."""
    s1 = tl.sum(z0, axis=1)
    if NCK > 1:
        s1 += tl.sum(z1, axis=1)
    if NCK > 2:
        s1 += tl.sum(z2, axis=1)
    if NCK > 3:
        s1 += tl.sum(z3, axis=1)
    mean = s1 / C
    zc_0 = z0 - mean[:, None]
    s2 = tl.sum(zc_0 * zc_0, axis=1)
    if NCK > 1:
        zc_1 = z1 - mean[:, None]
        s2 += tl.sum(zc_1 * zc_1, axis=1)
    if NCK > 2:
        zc_2 = z2 - mean[:, None]
        s2 += tl.sum(zc_2 * zc_2, axis=1)
    if NCK > 3:
        zc_3 = z3 - mean[:, None]
        s2 += tl.sum(zc_3 * zc_3, axis=1)
    var = s2 / C
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + ck)
    b = tl.load(b_ptr + ck)
    x0 = (zc_0 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    x1 = x0
    x2 = x0
    x3 = x0
    if NCK > 1:
        w = tl.load(w_ptr + CK + ck)
        b = tl.load(b_ptr + CK + ck)
        x1 = (zc_1 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if NCK > 2:
        w = tl.load(w_ptr + 2 * CK + ck)
        b = tl.load(b_ptr + 2 * CK + ck)
        x2 = (zc_2 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if NCK > 3:
        w = tl.load(w_ptr + 3 * CK + ck)
        b = tl.load(b_ptr + 3 * CK + ck)
        x3 = (zc_3 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    return x0, x1, x2, x3


@triton.jit
def _ln_cols(z0, z1, z2, z3, w_ptr, b_ptr, ck, eps, C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr):
    """[CK, BM] fp32 chunks (channels x tokens) -> LN over the channel axis -> bf16 chunks."""
    s1 = tl.sum(z0, axis=0)
    if NCK > 1:
        s1 += tl.sum(z1, axis=0)
    if NCK > 2:
        s1 += tl.sum(z2, axis=0)
    if NCK > 3:
        s1 += tl.sum(z3, axis=0)
    mean = s1 / C
    zc_0 = z0 - mean[None, :]
    s2 = tl.sum(zc_0 * zc_0, axis=0)
    if NCK > 1:
        zc_1 = z1 - mean[None, :]
        s2 += tl.sum(zc_1 * zc_1, axis=0)
    if NCK > 2:
        zc_2 = z2 - mean[None, :]
        s2 += tl.sum(zc_2 * zc_2, axis=0)
    if NCK > 3:
        zc_3 = z3 - mean[None, :]
        s2 += tl.sum(zc_3 * zc_3, axis=0)
    var = s2 / C
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + ck)
    b = tl.load(b_ptr + ck)
    x0 = (zc_0 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    x1 = x0
    x2 = x0
    x3 = x0
    if NCK > 1:
        w = tl.load(w_ptr + CK + ck)
        b = tl.load(b_ptr + CK + ck)
        x1 = (zc_1 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    if NCK > 2:
        w = tl.load(w_ptr + 2 * CK + ck)
        b = tl.load(b_ptr + 2 * CK + ck)
        x2 = (zc_2 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    if NCK > 3:
        w = tl.load(w_ptr + 3 * CK + ck)
        b = tl.load(b_ptr + 3 * CK + ck)
        x3 = (zc_3 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    return x0, x1, x2, x3


@triton.jit
def _k1_tile(pid0, pid1, r, mask_ptr, N, Np, BM: tl.constexpr, HAS_MASK: tl.constexpr, TRANS: tl.constexpr):
    """Token coordinates of a K1 program: (first token index of the tile's rows into z [tokens], load mask, plane offsets, store mask, mask values)."""
    if TRANS:
        j = pid1
        i0 = pid0 * BM
        ri = i0 + r
        lmask = (ri < N) & (j < N)
        j64 = j.to(tl.int64)
        tok = ri.to(tl.int64) * N + j64
        obase = j64 * Np + ri.to(tl.int64)
        smask = ri < Np
    else:
        i0 = pid1
        j0 = pid0 * BM
        rj = j0 + r
        lmask = (rj < N) & (i0 < N)
        i64 = i0.to(tl.int64)
        tok = i64 * N + rj.to(tl.int64)
        obase = i64 * Np + rj.to(tl.int64)
        smask = rj < Np
    if HAS_MASK:
        mv = tl.load(mask_ptr + tok, mask=lmask, other=0.0).to(tl.float32)
    else:
        mv = lmask.to(tl.float32)
    return tok, lmask, obase, smask, mv


@triton.jit
def _k1_gate_store(acc_g, acc_p, mv, lmask, smask, obase, n0, ncols, plane_stride, ab_ptr, HAS_MASK: tl.constexpr, FASTSIG: tl.constexpr):
    val = _gate(acc_g, FASTSIG) * acc_p
    if HAS_MASK:
        val = val * mv[:, None]
    val = tl.where(lmask[:, None], val, 0.0)
    nch = (n0 + ncols).to(tl.int64) * plane_stride
    tl.store(ab_ptr + obase[:, None] + nch[None, :], val.to(tl.bfloat16), mask=smask[:, None])


# =====================================================================================================================
# K1: LN_in + gated dual projection -> channel-major zero-padded planes.   program = one token tile of a pair row (TRANS=0) / column (TRANS=1)
#   TRANS = 0: tile = BM consecutive columns j of row i;  planes[n, i, j]              (grid: cdiv(Np, BM) x Np)
#   TRANS = 1: tile = BM consecutive rows i of column j;  planes[n, j, i]  (a^T, b^T)  (grid: cdiv(Np, BM) x Np)
# The C reduction is NCK chunks of CK channels (C == NCK * CK, CK a power of two <= 256), all resident: LN statistics are two-pass fp32
# over the chunks, each projection block accumulates NCK MMAs of K = CK.  _k1s reads z and the weights through pointer loads (any cc);
# kernels_desc._k1s_tma is the same kernel reading them through tensor descriptors (cc 9.0) -- same statements otherwise, same bytes out.
# =====================================================================================================================
@triton.jit
def _k1s(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, mask_ptr, ab_ptr,
         N, Np, NN, NC, plane_stride, eps,
         C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, D2: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
         HAS_MASK: tl.constexpr, TRANS: tl.constexpr, FASTSIG: tl.constexpr, STAGGER: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    r = tl.arange(0, BM)
    ck = tl.arange(0, CK)
    ck64 = ck.to(tl.int64)
    tok, lmask, obase, smask, mv = _k1_tile(pid0, pid1, r, mask_ptr, N, Np, BM, HAS_MASK, TRANS)
    zrow = z_ptr + (tok * C)[:, None]
    z0 = tl.load(zrow + ck64[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    z1 = z0
    z2 = z0
    z3 = z0
    if NCK > 1:
        z1 = tl.load(zrow + (CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    if NCK > 2:
        z2 = tl.load(zrow + (2 * CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    if NCK > 3:
        z3 = tl.load(zrow + (3 * CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    x0, x1, x2, x3 = _ln_rows(z0, z1, z2, z3, lnw_ptr, lnb_ptr, ck, eps, C, CK, NCK)
    ncols = tl.arange(0, BN)
    NB: tl.constexpr = D2 // BN
    for nb in tl.range(0, NB, num_stages=STAGES):
        if STAGGER:
            n0 = ((nb + pid0 + pid1) % NB) * BN
        else:
            n0 = nb * BN
        woff = (ck64 * D2)[:, None] + (n0 + ncols).to(tl.int64)[None, :]
        wg = tl.load(wgT_ptr + woff)
        wp = tl.load(wpT_ptr + woff)
        acc_g = tl.dot(x0, wg)
        acc_p = tl.dot(x0, wp)
        if NCK > 1:
            wg = tl.load(wgT_ptr + woff + CK * D2)
            wp = tl.load(wpT_ptr + woff + CK * D2)
            acc_g = tl.dot(x1, wg, acc_g)
            acc_p = tl.dot(x1, wp, acc_p)
        if NCK > 2:
            wg = tl.load(wgT_ptr + woff + 2 * CK * D2)
            wp = tl.load(wpT_ptr + woff + 2 * CK * D2)
            acc_g = tl.dot(x2, wg, acc_g)
            acc_p = tl.dot(x2, wp, acc_p)
        if NCK > 3:
            wg = tl.load(wgT_ptr + woff + 3 * CK * D2)
            wp = tl.load(wpT_ptr + woff + 3 * CK * D2)
            acc_g = tl.dot(x3, wg, acc_g)
            acc_p = tl.dot(x3, wp, acc_p)
        _k1_gate_store(acc_g, acc_p, mv, lmask, smask, obase, n0, ncols, plane_stride, ab_ptr, HAS_MASK, FASTSIG)


# =====================================================================================================================
# K3: channel-major x -> LN_out -> out-projection; LN_in(z) recomputed -> output gate; (+ residual); token-major store in out's dtype.
# program = BM consecutive tokens of pair row i; acc[BN out-channels, BM tokens] = W[BN, K] @ act[K, BM] accumulated over the K chunks
# (CH = NCH x CHK for the out-projection, C = NCK x CK for the gate); the output-channel loop walks C in BN blocks.
# =====================================================================================================================
@triton.jit
def _k3s(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wz_ptr, wg_ptr, out_ptr,
         N, Npx, x_plane_stride, eps,
         C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, CH: tl.constexpr, CHK: tl.constexpr, NCH: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr, FASTSIG: tl.constexpr, STAGGER: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    rows = pid_j * BM + tl.arange(0, BM)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    ch = tl.arange(0, CHK)
    ch64 = ch.to(tl.int64)
    cz = tl.arange(0, CK)
    cz64 = cz.to(tl.int64)
    xcol = (i * Npx + rows64)[None, :]
    # ---- x^T tile: NCH resident chunks [CHK, BM] -> LN_out
    xt0 = tl.load(x_ptr + (ch64 * x_plane_stride)[:, None] + xcol, mask=rmask[None, :], other=0.0).to(tl.float32)
    xt1 = xt0
    xt2 = xt0
    xt3 = xt0
    if NCH > 1:
        xt1 = tl.load(x_ptr + ((CHK + ch64) * x_plane_stride)[:, None] + xcol, mask=rmask[None, :], other=0.0).to(tl.float32)
    if NCH > 2:
        xt2 = tl.load(x_ptr + ((2 * CHK + ch64) * x_plane_stride)[:, None] + xcol, mask=rmask[None, :], other=0.0).to(tl.float32)
    if NCH > 3:
        xt3 = tl.load(x_ptr + ((3 * CHK + ch64) * x_plane_stride)[:, None] + xcol, mask=rmask[None, :], other=0.0).to(tl.float32)
    xln0, xln1, xln2, xln3 = _ln_cols(xt0, xt1, xt2, xt3, lnow_ptr, lnob_ptr, ch, eps, CH, CHK, NCH)
    # ---- z^T tile (gate input): NCK resident chunks [CK, BM] -> LN_in recomputed
    ztok = (i * N + rows64) * C
    zt0 = tl.load(z_ptr + cz64[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0).to(tl.float32)
    zt1 = zt0
    zt2 = zt0
    zt3 = zt0
    if NCK > 1:
        zt1 = tl.load(z_ptr + (CK + cz64)[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0).to(tl.float32)
    if NCK > 2:
        zt2 = tl.load(z_ptr + (2 * CK + cz64)[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0).to(tl.float32)
    if NCK > 3:
        zt3 = tl.load(z_ptr + (3 * CK + cz64)[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0).to(tl.float32)
    zln0, zln1, zln2, zln3 = _ln_cols(zt0, zt1, zt2, zt3, lniw_ptr, lnib_ptr, cz, eps, C, CK, NCK)
    # ---- output channels, BN at a time: p = W_o[nblk, :CH] @ LN_out(x) ; g = W_og[nblk, :C] @ LN_in(z)
    nr = tl.arange(0, BN)
    NB: tl.constexpr = C // BN
    for nb in range(0, NB):
        if STAGGER:
            n0 = ((nb + pid_j + i) % NB).to(tl.int32) * BN
        else:
            n0 = nb * BN
        n64 = (n0 + nr).to(tl.int64)
        wz = tl.load(wz_ptr + n64[:, None] * CH + ch64[None, :])                    # [BN, CHK]   (load -> dot per block keeps one weight
        acc_p = tl.dot(wz, xln0)                                                    #  block live at a time)
        if NCH > 1:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln1, acc_p)
        if NCH > 2:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (2 * CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln2, acc_p)
        if NCH > 3:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (3 * CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln3, acc_p)
        wg = tl.load(wg_ptr + n64[:, None] * C + cz64[None, :])                     # [BN, CK]
        acc_g = tl.dot(wg, zln0)
        if NCK > 1:
            wg = tl.load(wg_ptr + n64[:, None] * C + (CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln1, acc_g)
        if NCK > 2:
            wg = tl.load(wg_ptr + n64[:, None] * C + (2 * CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln2, acc_g)
        if NCK > 3:
            wg = tl.load(wg_ptr + n64[:, None] * C + (3 * CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln3, acc_g)
        if STOCK_ROUND:
            o = (acc_p.to(tl.bfloat16).to(tl.float32) * _gate(acc_g, FASTSIG)).to(tl.bfloat16).to(tl.float32)
        else:
            o = _gate(acc_g, FASTSIG) * acc_p
        optrs = n64[:, None] + ztok[None, :]
        if RESIDUAL:
            o = o + tl.load(z_ptr + optrs, mask=rmask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + optrs, o.to(out_ptr.dtype.element_ty), mask=rmask[None, :])


# =====================================================================================================================
# host side
# =====================================================================================================================
def ceil_to(n, m):
    return ((int(n) + int(m) - 1) // int(m)) * int(m)


def ensure_allocator():
    """Device-side tensor descriptors need a scratch allocator (once per process)."""
    if not _STATE["alloc"]:
        triton.set_allocator(lambda size, align, stream: torch.empty(int(size), dtype=torch.int8, device="cuda"))
        _STATE["alloc"] = True


def chunking(width, chunk=None):
    """(CK, NCK) for a channel width: the requested power-of-two chunk, else the width itself when it is a power of two <= 256, else 128."""
    width = int(width)
    if chunk is None:
        chunk = width if (width & (width - 1)) == 0 and width <= 256 else 128
    chunk = int(chunk)
    if chunk & (chunk - 1) or chunk > 256 or chunk < 16 or width % chunk or width // chunk > MAX_CHUNKS:
        raise ValueError("width %d cannot be walked as chunks of %d (power of two, 16..256, at most %d chunks)" % (width, chunk, MAX_CHUNKS))
    return chunk, width // chunk


def pack_weights(ln_in_w, ln_in_b, w_ag, w_ap, w_bg, w_bp, ln_out_w, ln_out_b, w_o, w_og, device=None):
    """The ten op weights -> the kernels' pack: LN affine fp32; W^T[in C, 2*D] bf16 with a|b concatenated (gate, proj); W_o [C, CH] and
    W_og [C, C] bf16 in their native [out, in] layout.  Weights of any float dtype are rounded to bf16 once here (the tensor-core operand form)."""
    dev = device if device is not None else w_o.device

    def f32(t):
        return t.detach().to(device=dev, dtype=torch.float32).contiguous()

    def b16(t):
        return t.detach().to(device=dev, dtype=torch.bfloat16)
    D, C = int(w_ap.shape[0]), int(w_ap.shape[1])
    CH = int(w_o.shape[1])
    if int(w_o.shape[0]) != C or tuple(w_og.shape) != (C, C) or D != CH:
        raise ValueError("weights disagree: w_ap %s w_o %s w_og %s" % (tuple(w_ap.shape), tuple(w_o.shape), tuple(w_og.shape)))
    return dict(ln_in_w=f32(ln_in_w), ln_in_b=f32(ln_in_b), ln_out_w=f32(ln_out_w), ln_out_b=f32(ln_out_b),
                wgT_in=b16(torch.cat([w_ag, w_bg], 0)).t().contiguous(), wpT_in=b16(torch.cat([w_ap, w_bp], 0)).t().contiguous(),
                wz=b16(w_o).contiguous(), wg_out=b16(w_og).contiguous(), C=C, CH=CH)


def launch_k1(z3, w, mask2, ab, N, Np, k1, eps, trans, levers, ck):
    """z3 [N, N, C] bf16|fp32 contiguous; ab [2*D, Np, Np] bf16 (written fully, zero pad); mask2 [N, N] float or None."""
    C = int(z3.shape[-1]); D2 = int(ab.shape[0])
    CK, NCK = chunking(C, ck)
    BM, BN, ST, WARPS = int(k1["BM"]), int(k1["BN"]), int(k1["num_stages"]), int(k1["num_warps"])
    use_tma = bool(k1.get("tma", False))
    if use_tma:
        ensure_allocator()
    if D2 % BN:
        raise ValueError("K1 BN %d does not divide 2*D = %d" % (BN, D2))
    grid = (triton.cdiv(Np, BM), Np)
    if use_tma:
        if not has_descriptor_api():
            raise RuntimeError("k1 cell asks for tensor descriptors; this triton has no tl.make_tensor_descriptor (select a pointer cell)")
        from . import kernels_desc
        kern = kernels_desc._k1s_tma
    else:
        kern = _k1s
    kern[grid](z3, w["ln_in_w"], w["ln_in_b"], w["wgT_in"], w["wpT_in"], mask2 if mask2 is not None else z3, ab,
               N, Np, N * N, N * C, Np * Np, eps,
               C=C, CK=CK, NCK=NCK, D2=D2, BM=BM, BN=BN, STAGES=ST, HAS_MASK=mask2 is not None, TRANS=bool(trans),
               FASTSIG=bool(levers.get("sigmoid", True)), STAGGER=bool(levers.get("stagger", False)), num_warps=WARPS)
    STATS["k1"] += 1


def launch_bmm(ab, x, form):
    """x_d = P0_d P1_d^T (NT) | P0_d^T P1_d (TN) over the D channel planes; cuBLAS strided-batched bf16 GEMM, fp32 accumulate; x bf16 or fp32."""
    D = int(x.shape[0])
    a, b = ab[:D], ab[D:]
    lhs, rhs = (a, b.transpose(1, 2)) if form == "NT" else (a.transpose(1, 2), b)
    if x.dtype == torch.bfloat16:
        torch.bmm(lhs, rhs, out=x)
    else:
        x.copy_(torch.bmm(lhs, rhs, out_dtype=torch.float32))
    return x


def launch_k3(x, z3, w, out, N, Np, k3, residual, stock_round, eps, levers, ck, chk):
    C = int(z3.shape[-1]); CH = int(x.shape[0])
    CK, NCK = chunking(C, ck)
    CHK, NCH = chunking(CH, chk)
    BM, BN, WARPS, ST = int(k3["BM"]), int(k3["BN"]), int(k3["num_warps"]), int(k3["num_stages"])
    if C % BN:
        raise ValueError("K3 BN %d does not divide C = %d" % (BN, C))
    grid = (triton.cdiv(N, BM), N)
    _k3s[grid](x, z3, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wz"], w["wg_out"], out,
               N, Np, Np * Np, eps, C=C, CK=CK, NCK=NCK, CH=CH, CHK=CHK, NCH=NCH, BM=BM, BN=BN,
               RESIDUAL=bool(residual), STOCK_ROUND=bool(stock_round), FASTSIG=bool(levers.get("sigmoid", True)),
               STAGGER=bool(levers.get("stagger", False)), num_warps=WARPS, num_stages=ST)
    STATS["k3"] += 1


def plane_form(outgoing, levers, cell_form=None):
    """(form, trans): the GEMM form and whether K1 writes transposed planes.  Base: outgoing NT / incoming TN on z-order planes.  incnt: the
    incoming direction on transposed planes contracts NT too.  cell_form ('NT'|'TN', the cell's measured faster form) with lever form: that
    form for either direction (incoming needs incnt to leave TN)."""
    if outgoing:
        form = cell_form if (levers.get("form", True) and cell_form in ("NT", "TN")) else "NT"
    elif not levers.get("incnt", True):
        form = "TN"
    else:
        form = cell_form if (levers.get("form", True) and cell_form in ("NT", "TN")) else "NT"
    trans = (form == "NT") != bool(outgoing)
    return form, trans


DEFAULT_LEVERS = dict(sigmoid=True, stagger=False, incnt=True, form=True, x32=False, stock_round=False)


def forward(z, mask, *, outgoing, pack, cell, eps=1e-5, residual=False, levers=None, out=None, pad=16):
    """z [N, N, C] (or [1, N, N, C]) CUDA, bf16 or fp32, C == pack['C']; mask [N, N] float/bool or None; pack = pack_weights(...);
    cell = {'k1': {BM, BN, num_warps, num_stages, tma}, 'k3': {BM, BN, num_warps, num_stages}, 'ck': int|None, 'chk': int|None, 'form': 'NT'|'TN'|None}.
    Returns update (residual=False) or z + update (residual=True) in z's dtype.  Raises (never substitutes) when a kernel cannot launch."""
    lv = dict(DEFAULT_LEVERS)
    if levers:
        lv.update(levers)
    squeeze = False
    if z.dim() == 4:
        if int(z.shape[0]) != 1:
            raise ValueError("a leading batch > 1 is not served by these kernels (got %s)" % (tuple(z.shape),))
        z = z[0]
        squeeze = True
        if mask is not None and mask.dim() == 3:
            mask = mask[0]
    if z.dim() != 3 or not z.is_cuda or z.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("expects a CUDA bf16 or fp32 [N, N, C] pair, got %s %s %s" % (tuple(z.shape), z.dtype, z.device))
    z3 = z.contiguous()
    N, N2, C = (int(s) for s in z3.shape)
    if N != N2:
        raise ValueError("pair must be square (got %d x %d)" % (N, N2))
    if C != int(pack["C"]):
        raise ValueError("pair width %d != the packed weights' %d" % (C, int(pack["C"])))
    D = int(pack["wz"].shape[1])
    Np = ceil_to(N, pad)
    m2 = None
    if mask is not None:
        m2 = mask
        if m2.dtype == torch.bool:
            m2 = m2.to(torch.float32)
        m2 = m2.reshape(N, N).contiguous()
    form, trans = plane_form(bool(outgoing), lv, cell.get("form"))
    ab = torch.empty((2 * D, Np, Np), dtype=torch.bfloat16, device=z3.device)
    x = torch.empty((D, Np, Np), dtype=torch.float32 if lv.get("x32") else torch.bfloat16, device=z3.device)
    launch_k1(z3, pack, m2, ab, N, Np, cell["k1"], eps, trans, lv, cell.get("ck"))
    launch_bmm(ab, x, form)
    del ab
    if out is None:
        out = torch.empty_like(z3)
    launch_k3(x, z3, pack, out, N, Np, cell["k3"], residual, lv.get("stock_round", False), eps, lv, cell.get("ck"), cell.get("chk"))
    STATS["calls"] += 1
    STATS["trans_calls"] += int(trans)
    STATS["tn_calls"] += int(form == "TN")
    return out.unsqueeze(0) if squeeze else out


def workspace_bytes(N, D, pad=16, x32=False):
    """Transient bytes of one call above z / out: the planes (2*D bf16) + x (D bf16 or fp32) at the padded extent."""
    Np = ceil_to(N, pad)
    return Np * Np * (2 * D * 2 + D * (4 if x32 else 2))


def reference_torch(z, mask, *, outgoing, weights, eps=1e-5, residual=False, dtype=torch.float64):
    """The op's module statements in torch at `dtype` (fp64 by default) -- the numerics reference of the served math (weights = the ten tensors)."""
    w = {k: v.to(dtype) for k, v in weights.items() if torch.is_tensor(v)}
    zz = z.to(dtype)
    if zz.dim() == 3:
        zz = zz.unsqueeze(0)
    x = torch.nn.functional.layer_norm(zz, (zz.shape[-1],), w["ln_in_w"], w["ln_in_b"], eps)
    m = None if mask is None else mask.to(dtype).reshape(1, zz.shape[1], zz.shape[2], 1)
    a = torch.sigmoid(x @ w["w_ag"].t()) * (x @ w["w_ap"].t())
    b = torch.sigmoid(x @ w["w_bg"].t()) * (x @ w["w_bp"].t())
    if m is not None:
        a = a * m
        b = b * m
    if outgoing:
        X = torch.einsum("bikd,bjkd->bijd", a, b)
    else:
        X = torch.einsum("bkid,bkjd->bijd", a, b)
    upd = torch.sigmoid(x @ w["w_og"].t()) * (torch.nn.functional.layer_norm(X, (X.shape[-1],), w["ln_out_w"], w["ln_out_b"], eps) @ w["w_o"].t())
    res = (zz + upd) if residual else upd
    return res if z.dim() == 4 else res[0]
