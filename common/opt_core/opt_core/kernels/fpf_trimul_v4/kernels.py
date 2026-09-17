"""Triton kernels for fpf_trimul_v4.  Fixed tiles from table.json (keyed cc|triton), no autotune at run time, no atomics, no split-K
-> run-to-run bit-exact and batch-invariant by construction.  A leading batch B (z [B, N, N, C]) is ONE launch set: K1 / K3 carry the batch element on
grid axis 2 (each program serves one (b, row i, token tile) with the single-plane arithmetic; b enters only as pointer offsets), the planes are laid out
a-planes of every b then b-planes of every b ([2, B, D, Np, Np]) so that a and b are contiguous [B*D, Np, Np] operands of ONE strided-batched cuBLAS GEMM,
and B = 1 is the [2D, Np, Np] layout and launch arithmetic byte for byte.  LayerNorm statistics fp32 (two-pass, tile resident in registers); every MMA bf16 x bf16 -> fp32 accumulate;
rounding points = stock cuEq TriMul under bf16 autocast (LN_in out -> bf16 | gated projection -> bf16 AFTER gating (+bias) on the fp32 accumulators | contraction: bf16 operands,
fp32 accumulate, bf16 out (cuBLAS) | LN_out out -> bf16 | gated output -> bf16 | residual add in fp32 -> input dtype).
v4.1.0 generalisation (module-agnostic `generic.trimul`): pair width C = c_z in {128, 256}; hidden D in {128, 256} (planes 2D, contraction over D channels); optional projection/gate/output
biases (HAS_BIAS constexpr: absent -> the exact v4.0.4 instruction stream); input z may be fp32 (IN_F32 constexpr, pointer-load K1 only; LN stats from the fp32 values; residual added in fp32,
stored fp32).  With C == D == 256, HAS_BIAS=False, IN_F32=False these kernels are the v4.0.4r1 kernels op for op (proved bit-exact on recorded stock activation dumps).
  K1  _k1c / _k1t : program = (token tile j0:j0+BM, row i, batch b); LN_in(z[b, i, j-tile, :]) once -> loop over 2D/BN column blocks: two K=C MMAs (gate|proj) -> (+b) -> sigmoid(g)*p*mask -> bf16
                    -> ab[a|b, b, n, i, j] (channel-major, zero pad)
      kdesc._k1d  : the same program on host TMA descriptors (cell impl 'tma2'); the incoming direction reads z transposed and writes the planes of pair column s as plane
                    ROW s, so both directions contract as a @ b^T (launch_k1 returns which layout it wrote; launch_bmm takes it)
  B   torch.bmm on the [B*D, Np, Np] plane views (cuBLAS strided-batched, bf16 in/out, fp32 acc): outgoing a @ b^T, incoming a^T @ b (transposed planes: a @ b^T) -> x[B*D, Np, Np]
  K3  _k3c        : program = (token tile, row i, batch b); x[b, :, i, j-tile] ([D, BM], channel-strided) -> LN_out ; z[b, i, j-tile, :]^T -> LN_in (gate input recomputed) ;
                    loop over C/BN output blocks: acc_p = Wo[nblk, :D] @ LN_out(x) ; acc_g = Wg[nblk, :C] @ LN_in(z) ; o = bf16(sigmoid(acc_g + bg) * (acc_p + bo)) (+ z) -> out[b, i, j, nblk]
      kdesc._k3d  : the same program on host TMA descriptors (cell impl 'tma'), weight blocks pipelined over the output-channel loop.
  Which kernel a call runs: the cell's impl word AND kdesc.serves (bf16 z, plane extent Np >= kdesc.N_MIN_DESC, the API importable); otherwise the pointer / _k1t kernels of the
  same cell numbers serve that call — the output bytes are the same either way (tests/test_cells_eq.py), the speed is not."""
import torch
from . import table as _table
from . import kdesc as _kdesc
import triton
import triton.language as tl


@triton.jit
def _k1c(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, bg_ptr, bp_ptr, mask_ptr, ab_ptr,
         N, Np, plane_stride, eps, NN, mask_bstride, bplane_hi,
         C: tl.constexpr, D2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, IN_F32: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)                       # 0 <= i < Np (pad rows i >= N are written as zeros)
    bidx = tl.program_id(2).to(tl.int64)                    # batch element: enters only as pointer / plane offsets (b = 0: the single-plane addresses)
    z_ptr += bidx * NN * C                                  # z[b] ([B, N, N, C] contiguous; NN = N*N)
    mask_ptr += bidx * mask_bstride                         # mask[b] (mask_bstride = N*N) or the one shared [N, N] mask (mask_bstride = 0)
    pbase = bidx * (D2 // 2)                                # a-plane index of channel n < D for this b: n + b*D ; b-plane index of channel n >= D: n + b*D + (B-1)*D (= bplane_hi)
    rows = pid_j * BM + tl.arange(0, BM)
    rows64 = rows.to(tl.int64)
    lmask = (rows < N) & (i < N)
    cols = tl.arange(0, C)
    cols64 = cols.to(tl.int64)
    zt = tl.load(z_ptr + ((i * N + rows64) * C)[:, None] + cols64[None, :], mask=lmask[:, None], other=0.0)
    zf = zt.to(tl.float32)                                                       # [BM, C]
    mean = tl.sum(zf, axis=1) / C
    zc = zf - mean[:, None]
    var = tl.sum(zc * zc, axis=1) / C
    rstd = tl.rsqrt(var + eps)
    lnw = tl.load(lnw_ptr + cols)
    lnb = tl.load(lnb_ptr + cols)
    x = (zc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)       # stock rounding point: LN output in the autocast dtype
    if HAS_MASK:
        mv = tl.load(mask_ptr + i * N + rows64, mask=lmask, other=0.0).to(tl.float32)
    obase = i * Np + rows64                                                       # [BM] element offsets inside plane 0
    smask = rows < Np
    ncols = tl.arange(0, BN)
    for nb in range(0, D2 // BN):
        n0 = nb * BN
        woff = cols64[:, None] * D2 + (n0 + ncols)[None, :]
        wg = tl.load(wgT_ptr + woff)                                             # [C, BN] bf16  (W^T pre-packed, unit stride along out-channels)
        acc_g = tl.dot(x, wg)                                                    # [BM, BN] fp32
        wp = tl.load(wpT_ptr + woff)
        acc_p = tl.dot(x, wp)
        if HAS_BIAS:
            acc_g = acc_g + tl.load(bg_ptr + n0 + ncols)[None, :]
            acc_p = acc_p + tl.load(bp_ptr + n0 + ncols)[None, :]
        val = tl.sigmoid(acc_g) * acc_p                                          # gate on fp32 accumulators, THEN round (cuEq order)
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)                                 # exact zeros in the pad region
        nch = n0 + ncols
        planes = nch.to(tl.int64) + pbase + (nch >= (D2 // 2)).to(tl.int64) * bplane_hi      # BN divides D: a block never straddles the a|b boundary
        optr = ab_ptr + obase[:, None] + (planes * plane_stride)[None, :]
        tl.store(optr, val.to(tl.bfloat16), mask=smask[:, None])


@triton.jit
def _k3c(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wz_ptr, wg_ptr, bo_ptr, bg_ptr, out_ptr,
         N, Npx, x_plane_stride, eps, NN,
         C: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr, HAS_BIAS: tl.constexpr, IN_F32: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    bidx = tl.program_id(2).to(tl.int64)                    # batch element: pointer offsets only
    x_ptr += bidx * CH * x_plane_stride                     # x[b] ([B, CH, Np, Np])
    z_ptr += bidx * NN * C                                  # z[b], out[b] ([B, N, N, C]; NN = N*N)
    out_ptr += bidx * NN * C
    rows = pid_j * BM + tl.arange(0, BM)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    ch = tl.arange(0, CH)
    cz = tl.arange(0, C)
    xt = tl.load(x_ptr + (ch.to(tl.int64) * x_plane_stride)[:, None] + (i * Npx + rows64)[None, :], mask=rmask[None, :], other=0.0)   # [CH, BM] bf16, channel-strided
    xf = xt.to(tl.float32)
    mean = tl.sum(xf, axis=0) / CH
    xc = xf - mean[None, :]
    var = tl.sum(xc * xc, axis=0) / CH
    rstd = tl.rsqrt(var + eps)
    low = tl.load(lnow_ptr + ch)
    lob = tl.load(lnob_ptr + ch)
    xln = (xc * rstd[None, :] * low[:, None] + lob[:, None]).to(tl.bfloat16)    # [CH, BM]  LN_out (stock rounding point)
    ztok = (i * N + rows64) * C                                                  # [BM] token offsets (token-major z / out)
    zt = tl.load(z_ptr + cz.to(tl.int64)[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0)                                  # [C, BM] = z tile transposed
    zf = zt.to(tl.float32)
    mean2 = tl.sum(zf, axis=0) / C
    zc2 = zf - mean2[None, :]
    var2 = tl.sum(zc2 * zc2, axis=0) / C
    rstd2 = tl.rsqrt(var2 + eps)
    liw = tl.load(lniw_ptr + cz)
    lib = tl.load(lnib_ptr + cz)
    zln = (zc2 * rstd2[None, :] * liw[:, None] + lib[:, None]).to(tl.bfloat16)  # [C, BM]  LN_in(z) recomputed (gate input)
    nr = tl.arange(0, BN)
    for nb in range(0, C // BN):
        n0 = nb * BN
        n64 = (n0 + nr).to(tl.int64)
        wz = tl.load(wz_ptr + n64[:, None] * CH + ch[None, :])                    # [BN, CH]  native nn.Linear layout [out, in]
        acc_p = tl.dot(wz, xln)                                                  # [BN, BM] fp32
        wg = tl.load(wg_ptr + n64[:, None] * C + cz[None, :])                     # [BN, C]
        acc_g = tl.dot(wg, zln)
        if HAS_BIAS:
            acc_p = acc_p + tl.load(bo_ptr + n0 + nr)[:, None]
            acc_g = acc_g + tl.load(bg_ptr + n0 + nr)[:, None]
        o = tl.sigmoid(acc_g) * acc_p
        if STOCK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)                                 # stock: cuEq kernel returns bf16; torch then adds the residual (fp32 opmath -> bf16)
        optrs = n64[:, None] + ztok[None, :]
        if RESIDUAL:
            o = o + tl.load(z_ptr + optrs, mask=rmask[None, :], other=0.0).to(tl.float32)
        if IN_F32:
            tl.store(out_ptr + optrs, o, mask=rmask[None, :])
        else:
            tl.store(out_ptr + optrs, o.to(tl.bfloat16), mask=rmask[None, :])


HAS_DESC = hasattr(tl, "make_tensor_descriptor") and _kdesc.HAS_TD     # the descriptor cells build: device-side descriptors (_k1t) and host-side ones (kdesc) come with the same tritons (>= 3.4)
DESC_IMPLS = _table.DESC_IMPLS                                # K1 / K3 cell impl words that need the tensor-descriptor API ('tma' = _k1t, 'tma2' = kdesc._k1d; K3 'tma' = kdesc._k3d)
_ALLOCATOR_SET = {"done": False}


def ensure_allocator():
    """Triton tensor-descriptor kernels need a global scratch allocator (idempotent; standard torch idiom)."""
    if not _ALLOCATOR_SET["done"] and hasattr(triton, "set_allocator"):
        triton.set_allocator(lambda size, align, stream: torch.empty(int(size), dtype=torch.int8, device="cuda"))
        _ALLOCATOR_SET["done"] = True


@triton.jit
def _k1t(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, bg_ptr, bp_ptr, mask_ptr, ab_ptr,
         N, Np, NN, plane_stride, eps, BNN, mask_bstride, bplane_hi,
         C: tl.constexpr, D2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """K1 with TMA tensor-descriptor loads of the (bf16) z tile and of the weight blocks + a software-pipelined weight-block loop (impl='tma').
    Identical arithmetic to _k1c (bit-exact-equal planes); rows past the end of row i read row i+1 tokens (finite; past the last row of z[b]: z[b+1]'s first
    tokens, or the descriptor's zero fill past the end of z) and are zeroed before the store.  The descriptor spans the whole batch ([B*N*N, C] rows, BNN = B*N*N)."""
    pid_j = tl.program_id(0)
    i = tl.program_id(1)
    b = tl.program_id(2)                                    # batch element: row / mask / plane offsets only
    j0 = pid_j * BM
    i64 = i.to(tl.int64)
    b64 = b.to(tl.int64)
    mask_ptr += b64 * mask_bstride
    pbase = b64 * (D2 // 2)
    rows = j0 + tl.arange(0, BM)
    rows64 = rows.to(tl.int64)
    lmask = (rows < N) & (i < N)
    cols = tl.arange(0, C)
    z_desc = tl.make_tensor_descriptor(z_ptr, shape=[BNN, C], strides=[C, 1], block_shape=[BM, C])
    wg_desc = tl.make_tensor_descriptor(wgT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[C, BN])
    wp_desc = tl.make_tensor_descriptor(wpT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[C, BN])
    r0 = b * NN + tl.minimum(i, N - 1) * N + j0
    zt = z_desc.load([r0, 0])
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, axis=1) / C
    zc = zf - mean[:, None]
    var = tl.sum(zc * zc, axis=1) / C
    rstd = tl.rsqrt(var + eps)
    lnw = tl.load(lnw_ptr + cols)
    lnb = tl.load(lnb_ptr + cols)
    x = (zc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)
    if HAS_MASK:
        mv = tl.load(mask_ptr + i64 * N + rows64, mask=lmask, other=0.0).to(tl.float32)
    obase = i64 * Np + rows64
    smask = rows < Np
    ncols = tl.arange(0, BN)
    for nb in tl.range(0, D2 // BN, num_stages=STAGES):
        n0 = nb * BN
        wg = wg_desc.load([0, n0])
        wp = wp_desc.load([0, n0])
        acc_g = tl.dot(x, wg)
        acc_p = tl.dot(x, wp)
        if HAS_BIAS:
            acc_g = acc_g + tl.load(bg_ptr + n0 + ncols)[None, :]
            acc_p = acc_p + tl.load(bp_ptr + n0 + ncols)[None, :]
        val = tl.sigmoid(acc_g) * acc_p
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)
        nch = n0 + ncols
        planes = nch.to(tl.int64) + pbase + (nch >= (D2 // 2)).to(tl.int64) * bplane_hi
        optr = ab_ptr + obase[:, None] + (planes * plane_stride)[None, :]
        tl.store(optr, val.to(tl.bfloat16), mask=smask[:, None])


def ceil_to(n, m):
    return ((n + m - 1) // m) * m


PAD = 16
SUPPORTED_C = (128, 256)          # pair width c_z (LN_in / gate-in / output width)
SUPPORTED_D = (128, 256)          # hidden width (a/b planes, LN_out, out-projection input)


def pack_generic(*, ln_in_w, ln_in_b, w_ag, w_ap, w_bg, w_bp, ln_out_w, ln_out_b, w_o, w_og,
                 b_ag=None, b_ap=None, b_bg=None, b_bp=None, b_o=None, b_og=None, cdt=torch.bfloat16):
    """Module-agnostic weight pack (once per weight set; cache it — see generic.pack_weights).  Math served (x = LN_in(z)):
         a = sigmoid(x @ w_ag^T + b_ag) * (x @ w_ap^T + b_ap) * mask ;  b = same with w_bg/w_bp ;  X = contraction(a, b) over the third token ;
         out = sigmoid(x @ w_og^T + b_og) * (LN_out(X) @ w_o^T + b_o)  (+ z if residual)
       shapes: ln_in_* [C]; w_a*/w_b* [D, C]; ln_out_* [D]; w_o [C, D]; w_og [C, C]; biases [D]/[C] or None (all None -> HAS_BIAS=False = the v4.0.4 kernels).
       GEMM weights are cast to bf16 exactly as autocast does (fp32 -> bf16 RN); LN affine parameters and biases stay fp32."""
    def f32(t): return t.detach().float().contiguous()
    def c(t): return t.detach().to(cdt)
    D, C = int(w_ap.shape[0]), int(w_ap.shape[1])
    assert tuple(w_ag.shape) == (D, C) and tuple(w_bg.shape) == (D, C) and tuple(w_bp.shape) == (D, C), "projection/gate weights must all be [D, C]"
    assert tuple(w_o.shape) == (C, D) and tuple(w_og.shape) == (C, C), "w_o must be [C, D], w_og [C, C]"
    assert ln_in_w.numel() == C and ln_in_b.numel() == C and ln_out_w.numel() == D and ln_out_b.numel() == D
    w = {"C": C, "D": D, "CH": D}
    w["ln_in_w"], w["ln_in_b"] = f32(ln_in_w), f32(ln_in_b)
    w["ln_out_w"], w["ln_out_b"] = f32(ln_out_w), f32(ln_out_b)
    g_in = torch.cat([w_ag.detach(), w_bg.detach()], 0)          # [2D, C]  (a|b concatenated, gate)
    p_in = torch.cat([w_ap.detach(), w_bp.detach()], 0)          # [2D, C]  (a|b concatenated, projection)
    w["wgT_in"] = c(g_in).t().contiguous()                        # [C, 2D]
    w["wpT_in"] = c(p_in).t().contiguous()
    w["wz"] = c(w_o).contiguous()                                 # [C, D]  native [out, in]
    w["wg_out"] = c(w_og).contiguous()                            # [C, C]
    biases = [b_ag, b_ap, b_bg, b_bp, b_o, b_og]
    w["has_bias"] = any(b is not None for b in biases)
    if w["has_bias"]:
        dev = w_ap.device
        zD = lambda b: f32(b) if b is not None else torch.zeros(D, dtype=torch.float32, device=dev)
        zC = lambda b: f32(b) if b is not None else torch.zeros(C, dtype=torch.float32, device=dev)
        w["bg_in"] = torch.cat([zD(b_ag), zD(b_bg)], 0).contiguous()   # [2D]
        w["bp_in"] = torch.cat([zD(b_ap), zD(b_bp)], 0).contiguous()
        w["b_o"], w["b_og"] = zC(b_o), zC(b_og)
    return w


def pack_weights(module, cdt=torch.bfloat16):
    """The stock TriangleMultiplicativeUpdate attribute names -> pack_generic (the only place those names appear)."""
    def bias(lin): return getattr(lin, "bias", None)
    return pack_generic(ln_in_w=module.layer_norm_in.weight, ln_in_b=module.layer_norm_in.bias,
                        w_ag=module.linear_a_g.weight, w_ap=module.linear_a_p.weight, w_bg=module.linear_b_g.weight, w_bp=module.linear_b_p.weight,
                        b_ag=bias(module.linear_a_g), b_ap=bias(module.linear_a_p), b_bg=bias(module.linear_b_g), b_bp=bias(module.linear_b_p),
                        ln_out_w=module.layer_norm_out.weight, ln_out_b=module.layer_norm_out.bias,
                        w_o=module.linear_z.weight, w_og=module.linear_g.weight, b_o=bias(module.linear_z), b_og=bias(module.linear_g), cdt=cdt)


MASK_SHARED, MASK_PER_B = 0, 1       # mask batch layout words for launch_k1: one [N, N] mask shared by every b (batch stride 0) | a [B, N, N] mask (batch stride N*N)
INT32_MAX = _table.INT32_MAX             # the TMA kernels (`_k1t`, kdesc `_k1d` / `_k3d`) index the whole batch's token rows [B*N*N, C] with an int32 descriptor coordinate
                                      # There is NO element-count ceiling otherwise: every z / mask / plane / out offset in _k1c, _k1t and _k3c is formed in int64 (program ids cast
                                      # with .to(tl.int64) before any product; channel / row aranges cast before multiplying a stride), so z, the planes and the output may each exceed
                                      # 2^31 elements (equality-tested at B 128 x N 384 x C 128 = 2.4e9 elements of z, 4.8e9 plane elements, bf16 and fp32).
CUDA_GRID_Z_MAX = 65535               # CUDA's limit on grid axis 2 = the batch extent of one launch set
BATCH_LIMITS = ("tma_rows>int32", "grid_batch>65535")   # the NAMES of the two limits of one launch set (batch_launch_limit); a batch beyond them is B single-plane launch sets (generic: a counted event) or a refusal here


class TrimulUnsupported(ValueError):
    """Raised (before any kernel launch) when the inputs are outside the served cell. .reason is a short machine-readable tag.  Defined here so the kernel-level entry and
    the module-agnostic entry (generic.TrimulUnsupported is this class) refuse with ONE type: a consumer's `except TrimulUnsupported` covers both."""
    def __init__(self, reason, detail=""):
        super().__init__("fpf_trimul_v4.generic: unsupported (%s) %s" % (reason, detail)); self.reason = reason

    @property
    def cannot_run(self):
        """True when the lever CANNOT RUN in this process (`none:<why>`: its SAFE cell failed to build; `probe-failed`: the warm numerics probe refused the shape class):
        the caller's mode refuses by that name — never a per-call structural refusal (c/d/dtype/mask/n/no-cell), which is False."""
        return str(self.reason).startswith("none") or self.reason == "probe-failed"


class BatchLimit(TrimulUnsupported):
    """z [B, N, N, C] exceeds a limit of ONE launch set — raised by trimul_v4_forward BEFORE any allocation or launch, `.reason` in BATCH_LIMITS.  When every batch element on
    its own is within the limits (B*N*N -> N*N rows), generic.trimul_packed serves such a batch as B single-plane launch sets and counts the event by this name instead."""
    def __init__(self, reason, detail=""):
        TrimulUnsupported.__init__(self, reason, "one launch set cannot serve this batch" + ((": " + detail) if detail else ""))
        self.detail = detail


def batch_launch_limit(B, N, k1, in_f32, k3=None):
    """None when z [B, N, N, C] is ONE launch set; else the name of the limit it exceeds (checked in this order): 'tma_rows>int32' — the cell's K1 (or K3) is a descriptor kernel
    (bf16 z) and B*N*N > INT32_MAX (the descriptor's row coordinate is int32; every other offset in K1/K3 is int64); 'grid_batch>65535' — B exceeds CUDA's grid axis 2."""
    B, N = int(B), int(N)
    desc = k1.get("impl", "ptr") in DESC_IMPLS or (k3 or {}).get("impl", "ptr") in DESC_IMPLS
    if desc and not in_f32 and B * N * N > INT32_MAX:
        return "tma_rows>int32"
    if B > CUDA_GRID_Z_MAX:
        return "grid_batch>65535"
    return None


LAUNCHES = {"k1d": 0, "k1t": 0, "k1c": 0, "k3d": 0, "k3c": 0}     # which kernel served each K1 / K3 launch in this process (decided per call on impl word, dtype and plane extent):
                                                                 # k1d / k3d = the host-descriptor cells (kdesc), k1t = the device-descriptor K1, k1c / k3c = the pointer kernels;
                                                                 # generic.COUNTS["kernels"] is this dict (printed on the exit COUNTS line)


def launch_k1(z, w, mask, ab, N, Np, k1, eps=1e-5, mask_layout=MASK_PER_B, outgoing=True, k3=None):
    """z [B, N, N, C] contiguous (bf16 | fp32); ab [2, B, D, Np, Np] bf16 (a-planes of every b, then b-planes; B = 1: the [2D, Np, Np] planes); mask None | [N, N] (MASK_SHARED) |
    [B, N, N] (MASK_PER_B), float, contiguous.  One launch, grid (token tiles, Np rows, B).  Returns the plane layout written: True = transposed (the descriptor K1, impl
    'tma2', on the incoming direction: launch_bmm(..., planes_t=True)), False = the [a|b, b, ch, i, k] layout of z's own (i, k) order (every other case)."""
    B, C = z.shape[0], z.shape[-1]; D = ab.shape[2]; D2 = 2 * D
    grid = (triton.cdiv(Np, k1["BM"]), Np, B)
    hb = bool(w.get("has_bias")); bg = w["bg_in"] if hb else w["ln_in_w"]; bp = w["bp_in"] if hb else w["ln_in_w"]
    in_f32 = z.dtype == torch.float32
    mask_bstride = 0 if (mask is None or mask_layout == MASK_SHARED) else N * N
    bplane_hi = (B - 1) * D                                      # extra plane offset of the b-planes: channel n >= D of batch b lives at plane n + b*D + (B-1)*D
    impl = k1.get("impl", "ptr")
    if impl in DESC_IMPLS and not in_f32 and B * N * N > INT32_MAX:   # trimul_v4_forward refuses by name before allocating; a direct launch_k1 call gets the same word, never a truncated coordinate
        raise BatchLimit("tma_rows>int32", "B*N*N = %d" % (B * N * N))
    if _kdesc.serves(z, Np, k1, k3 or {})[0]:                   # host-descriptor K1 (impl 'tma2'): bf16 z, Np >= kdesc.N_MIN_DESC; incoming planes transposed
        LAUNCHES["k1d"] += 1
        return _kdesc.launch_k1d(z, w, mask, ab, N, Np, k1, eps, mask_bstride, bool(outgoing))
    if impl in DESC_IMPLS and not in_f32:                       # TMA loads + pipelined weight blocks; requires tl.make_tensor_descriptor (triton >= 3.4); bf16 z only ('tma2' below its extent: same numbers)
        ensure_allocator()
        LAUNCHES["k1t"] += 1
        _k1t[grid](z, w["ln_in_w"], w["ln_in_b"], w["wgT_in"], w["wpT_in"], bg, bp, mask if mask is not None else z, ab, N, Np, N * N, Np * Np, eps, B * N * N, mask_bstride, bplane_hi,
                   C=C, D2=D2, BM=k1["BM"], BN=k1["BN"], STAGES=k1["num_stages"], HAS_MASK=mask is not None, HAS_BIAS=hb, num_warps=k1["num_warps"])
        return False
    LAUNCHES["k1c"] += 1
    _k1c[grid](z, w["ln_in_w"], w["ln_in_b"], w["wgT_in"], w["wpT_in"], bg, bp, mask if mask is not None else z, ab, N, Np, Np * Np, eps, N * N, mask_bstride, bplane_hi,
               C=C, D2=D2, BM=k1["BM"], BN=k1["BN"], HAS_MASK=mask is not None, HAS_BIAS=hb, IN_F32=in_f32, num_warps=k1["num_warps"], num_stages=k1["num_stages"])
    return False


def launch_bmm(ab, x, outgoing, planes_t=False):
    """ab [2, B, D, Np, Np] -> a = ab[0], b = ab[1] as contiguous [B*D, Np, Np] views (zero pad; all extents multiples of 16) -> ONE plain strided-batched GEMM into x [B, D, Np, Np]
    viewed [B*D, Np, Np].  B = 1: the [D, Np, Np] operands of the single-plane call.  planes_t: the planes are transposed (launch_k1's return on the incoming direction
    under the descriptor K1) -> the incoming contraction is a @ b^T too."""
    BD = x.shape[0] * x.shape[1]; Np = x.shape[-1]
    a = ab[0].view(BD, Np, Np); b = ab[1].view(BD, Np, Np); xv = x.view(BD, Np, Np)
    if outgoing or planes_t:
        torch.bmm(a, b.transpose(1, 2), out=xv)             # x[p,i,j] = sum_k a[p,i,k] b[p,j,k]   (transposed incoming planes: a[p,i,k] = a_in[p,k,i])
    else:
        torch.bmm(a.transpose(1, 2), b, out=xv)             # x[p,i,j] = sum_k a[p,k,i] b[p,k,j]


def launch_k3(x, z, w, out, N, Np, k3, residual=True, stock_round=True, eps=1e-5, k1=None):
    """x [B, D, Np, Np]; z, out [B, N, N, C] contiguous.  One launch, grid (token tiles, N rows, B).  impl 'tma' cells run kdesc._k3d when kdesc.serves (bf16 z, Np >= N_MIN_DESC),
    else the pointer kernel with the cell's numbers."""
    B, C = z.shape[0], z.shape[-1]; CH = x.shape[1]
    if _kdesc.serves(z, Np, k1 or {}, k3)[1]:
        LAUNCHES["k3d"] += 1
        _kdesc.launch_k3d(x, z, w, out, N, Np, k3, residual, stock_round, eps)
        return
    LAUNCHES["k3c"] += 1
    grid = (triton.cdiv(N, k3["BM"]), N, B)
    hb = bool(w.get("has_bias")); bo = w["b_o"] if hb else w["ln_in_w"]; bg = w["b_og"] if hb else w["ln_in_w"]
    _k3c[grid](x, z, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wz"], w["wg_out"], bo, bg, out, N, Np, Np * Np, eps, N * N,
               C=C, CH=CH, BM=k3["BM"], BN=k3["BN"], RESIDUAL=residual, STOCK_ROUND=stock_round, HAS_BIAS=hb, IN_F32=(z.dtype == torch.float32),
               num_warps=k3["num_warps"], num_stages=k3["num_stages"])


def resolve_cfg(cfg, C, D, has_bias):
    """cells row -> (k1, k3) launch cells for this (C, D, bias): the table's one statement (table.resolve_cfg: overrides keyed k1_C<C>_D<D>[_bias] | k1_C<C>_D<D> |
    k1_C<C>[_bias] | k1_C<C>, same for k3; first match wins, else the row's base k1/k3)."""
    return _table.resolve_cfg(cfg, C, D, has_bias)


def trimul_v4_forward(z, outgoing, mask, w, cfg, eps=1e-5, residual=True, stock_round=True, out=None, pad=PAD):
    """z: [N, N, C] or [B, N, N, C] contiguous, bf16 (or fp32: pointer-load K1, fp32 out); mask: None, [N, N] / [1, N, N] (one mask for every b) or [B, N, N] (float/bool);
    w = pack_generic(...) with w['C'] == C.  pad: plane leading extent Np = ceil_pad(N); must be a multiple of 8 (16-byte bf16 rows for cuBLAS); 16 = bytes (the module-signature
    wrapper), 8 admissible (v4.1.0 datapoint).  The whole batch is ONE launch set (K1, bmm, K3); every batch element gets the single-plane arithmetic.
    Returns z + update (residual=True) or the update, as a NEW tensor of z's dtype and shape."""
    assert z.dim() in (3, 4) and z.shape[-3] == z.shape[-2] and z.is_contiguous(), "z must be a contiguous [N, N, C] or [B, N, N, C] tensor"
    assert z.dtype in (torch.bfloat16, torch.float32), "z must be bf16 or fp32"
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    B, N, _, C = z4.shape
    D = int(w.get("D", w["wz"].shape[1]))                      # v4.0.4-style pack dicts (no 'D'/'C'/'has_bias' keys) keep working
    assert C == int(w.get("C", w["wz"].shape[0])), "z channel width %d != packed weights C %d" % (C, w.get("C"))
    assert C in SUPPORTED_C and D in SUPPORTED_D, "unsupported (c_z, D) = (%d, %d)" % (C, D)
    assert pad % 8 == 0 and pad >= 8, "pad must be a positive multiple of 8"
    k1, k3 = resolve_cfg(cfg, C, D, bool(w.get("has_bias")))
    lim = batch_launch_limit(B, N, k1, z.dtype == torch.float32, k3)
    if lim is not None:                                        # refused BY NAME before any allocation or launch (generic serves such a batch plane by plane, counted under this name)
        raise BatchLimit(lim, "B = %d, N = %d, k1 = %s" % (B, N, k1.get("impl", "ptr")))
    dev = z.device
    Np = ceil_to(N, pad)
    ab = torch.empty((2, B, D, Np, Np), dtype=torch.bfloat16, device=dev)
    x = torch.empty((B, D, Np, Np), dtype=torch.bfloat16, device=dev)
    m, layout = None, MASK_PER_B
    if mask is not None:
        m = mask
        if m.dtype == torch.bool:
            m = m.to(torch.float32)
        assert tuple(m.shape[-2:]) == (N, N) and m.numel() in (N * N, B * N * N), "mask must be [N, N], [1, N, N] or [B, N, N]"
        if m.numel() == N * N:
            m = m.reshape(N, N).contiguous(); layout = MASK_SHARED
        else:
            m = m.reshape(B, N, N).contiguous()
    planes_t = launch_k1(z4, w, m, ab, N, Np, k1, eps, layout, outgoing, k3)
    launch_bmm(ab, x, outgoing, planes_t)
    if out is None:
        out = torch.empty_like(z4)
    else:
        assert out.shape == z4.shape or (z.dim() == 3 and out.shape == z.shape), "out must have z's shape"
        out = out if out.dim() == 4 else out.unsqueeze(0)
    launch_k3(x, z4, w, out, N, Np, k3, residual, stock_round, eps, k1)
    return out if z.dim() == 4 else out[0]
