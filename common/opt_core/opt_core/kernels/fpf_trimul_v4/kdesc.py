"""The host-descriptor cells of fpf_trimul_v4: K1 ``impl='tma2'`` (:func:`_k1d`) and K3 ``impl='tma'`` (:func:`_k3d`).  Same math, rounding points and plane
layout contract as ``kernels._k1t`` / ``kernels._k3c`` (output bytes equal to those cells' on the tested matrix of tests/test_cells_eq.py — synthetic sizes and
recorded activations; an observed equality per stack, HAZARDS 17);
what differs is HOW the tiles move:
  * every global access is a TMA box built on the host (``triton.tools.tensor_descriptor.TensorDescriptor``): the z tile, the weight blocks (double/triple
    buffered over the 2D / C output-channel loop by ``num_stages``), the plane tile store (K1) and the x tile / residual / output tiles (K3) — no pointer
    tensors, no per-element masks (zero padding and row ends come from the box clipping);
  * K1 serves the INCOMING direction by reading z transposed (the tokens (r, s) of pair column s are one strided TMA box) and storing that column's planes as
    plane ROW s: the planes of both directions are then contracted by the same ``a @ b^T`` strided-batched GEMM (``kernels.launch_bmm(..., planes_t=True)``),
    instead of ``a^T @ b`` for incoming.
Served domain (else the row's ``_k1t`` / ``_k1c`` / ``_k3c`` cells serve the call, decided per call by :func:`serves`): z bf16, plane extent Np >= :data:`N_MIN_DESC`
(the launch-cost gate, see there), C / D in kernels.SUPPORTED_*, any mask layout, any B within kernels.batch_launch_limit ('tma_rows>int32' covers these cells: the z / plane /
out descriptors index B*N*N token rows and Np*Np plane columns with int32 coordinates; B*N*N <= INT32_MAX is checked before launch and :func:`serves` requires
Np*Np <= INT32_MAX, so a plane extent above 46,336 takes the replaced kernels).
Requires triton's host tensor-descriptor API (:data:`HAS_TD`; triton >= 3.4) — ``cells.has_desc()`` gates the table rows that name these cells."""
import torch
import triton
import triton.language as tl

from .table import INT32_MAX

try:                                                            # triton >= 3.4; older stacks serve the pointer cells (cells.has_desc() is False there)
    from triton.tools.tensor_descriptor import TensorDescriptor
    HAS_TD = True
except ImportError:                                             # a triton without the tools package / the host descriptor class
    TensorDescriptor = None
    HAS_TD = False

N_MIN_DESC = 512                  # smallest plane extent Np these cells serve.  Two bounds: >= K1's token tile BM (the last tile of a row is placed at Np - BM), and
                                  # the size below which the LAUNCH-side cost of host descriptors (~0.06-0.10 ms per call more than the pointer / device-descriptor
                                  # cells: descriptor objects + the launcher's tensor-map encodes) exceeds the kernels' device-side gain (~12 % of a 0.2-0.3 ms op at
                                  # Np 384-400) — a pair stack at that size is launch-bound, so the replaced cells (same output bytes) serve it


def names_desc(k1, k3):
    """(K1 names the host-descriptor cell 'tma2', K3 names the host-descriptor cell 'tma') for resolved cells k1 / k3."""
    return (k1 or {}).get("impl") == "tma2", (k3 or {}).get("impl") == "tma"


def serves(z, Np, k1, k3):
    """(k1_desc, k3_desc): whether this call runs the descriptor K1 / K3 — the cell names it (:func:`names_desc`), z is bf16, Np >= N_MIN_DESC and the API is here."""
    n1, n3 = names_desc(k1, k3)
    ok = HAS_TD and z.dtype == torch.bfloat16 and N_MIN_DESC <= int(Np) and int(Np) * int(Np) <= INT32_MAX      # plane column m*Np + k is an int32 coordinate too
    return (ok and n1 and int(Np) >= int(k1["BM"]), ok and n3)


@triton.jit
def _k1d(z_desc, ab_desc, mask_ptr, lnw_ptr, lnb_ptr, wg_desc, wp_desc, bg_ptr, bp_ptr,
         N, Np, D, zrow_m, zcol_m, mask_sm, mask_sk, mask_sb, bplane0, eps,
         C: tl.constexpr, D2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, W_STAGES: tl.constexpr,
         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, TRANS: tl.constexpr):
    """Program = (token tile k0..k0+BM of plane row m, m in [0, Np), batch b).  z_desc: 2-D box [BM, C] = the z tile of tokens (m, k):
         TRANS=0 (outgoing): z viewed [B*N*N, C], row b*N*N + m*N + k          (zrow_m = N, zcol_m = 0;  m = pair row i, k = pair column)
         TRANS=1 (incoming): z viewed [B*N, N*C],  row b*N + k, column m*C     (zrow_m = 0, zcol_m = C;  m = pair column s, k = pair row r)
       ab_desc: the planes [2, B, D, Np, Np] viewed [2*B*D, Np*Np], box [BN, BM]: a-plane of (b, ch) = row b*D + ch, b-plane = bplane0 + b*D + ch (bplane0 = B*D),
       element (m, k) = column m*Np + k.  mask fp32, element (b, m, k) at b*mask_sb + m*mask_sm + k*mask_sk.  wg/wp_desc: [C, 2D] bf16 (W^T), box [C, BN].
       LN_in(z tile) once in fp32 -> x bf16; per BN-channel block: gate / projection MMAs (fp32 acc) (+bias) -> sigmoid(g) * p * mask -> zero past N -> bf16 -> transposed
       [BN, BM] TMA store (channel-major planes, zero padding written)."""
    pid_k = tl.program_id(0)
    m = tl.program_id(1)
    b = tl.program_id(2)
    k0 = tl.minimum(pid_k * BM, Np - BM)                       # last tile re-covers the row end: no box ever wraps into the next plane row
    m64 = m.to(tl.int64); b64 = b.to(tl.int64)
    ks = k0 + tl.arange(0, BM)
    valid = (ks < N) & (m < N)
    cols = tl.arange(0, C)
    lnw = tl.load(lnw_ptr + cols); lnb = tl.load(lnb_ptr + cols)   # parameter loads first: their latency hides under the tile's TMA load
    if HAS_MASK:
        mv = tl.load(mask_ptr + b64 * mask_sb + m64 * mask_sm + ks.to(tl.int64) * mask_sk, mask=valid, other=0.0).to(tl.float32)
    if TRANS:
        zt = z_desc.load([b * N + k0, m * zcol_m])                # rows r = k (past N: next batch / zero fill), columns of pair column m (past N: zero fill)
    else:
        zt = z_desc.load([b * N * N + tl.minimum(m, N - 1) * zrow_m + k0, 0])   # rows past this pair row read the next row's tokens (finite, zeroed below)
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, axis=1) / C
    zc = zf - mean[:, None]
    var = tl.sum(zc * zc, axis=1) / C
    rstd = tl.rsqrt(var + eps)
    x = (zc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)     # stock rounding point
    col0 = m * Np + k0                                         # int32 plane column of (m, k0); Np*Np < 2^31
    ncols = tl.arange(0, BN)
    for nb in tl.range(0, D2 // BN, num_stages=W_STAGES):
        n0 = nb * BN
        wg = wg_desc.load([0, n0])                             # [C, BN] bf16 (W^T, unit stride along output channels)
        wp = wp_desc.load([0, n0])
        acc_g = tl.dot(x, wg)                                  # [BM, BN] fp32
        acc_p = tl.dot(x, wp)
        if HAS_BIAS:
            acc_g = acc_g + tl.load(bg_ptr + n0 + ncols)[None, :]
            acc_p = acc_p + tl.load(bp_ptr + n0 + ncols)[None, :]
        val = tl.sigmoid(acc_g) * acc_p                        # gate on the fp32 accumulators, THEN round (stock order)
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(valid[:, None], val, 0.0)               # exact zeros in the pad region and past N
        which = n0 // D                                        # 0 = a-planes, 1 = b-planes (BN divides D: a block never straddles a|b)
        prow = which * bplane0 + b * D + (n0 - which * D)
        ab_desc.store([prow, col0], tl.trans(val.to(tl.bfloat16)))              # [BN, BM]: channel-major tile


@triton.jit
def _k3d(x_desc, z_desc, zres_desc, out_desc, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, woT_desc, wgT_desc, bo_ptr, bg_ptr,
         N, Np, eps,
         C: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr, HAS_BIAS: tl.constexpr, W_STAGES: tl.constexpr):
    """Program = (token tile j0..j0+BM of pair row i, i in [0, N), batch b).  x_desc: planes x [B, D, Np, Np] viewed [B*D, Np*Np], box [D, BM] at (b*D, i*Np + j0);
       z_desc: z viewed [B*N*N, C], box [BM, C];  zres_desc: same view, box [BM, BN] (the residual chunk, re-read L2-hot);  out_desc: out viewed [B*N, N, C],
       box [1, BM, BN] (columns >= N clipped);  woT [D, C] / wgT [C, C] bf16 (W^T), boxes [D | C, BN].
       LN_out(x tile) and LN_in(z tile) once in fp32 -> bf16; per BN output-channel block: out-projection / gate MMAs (fp32 acc) (+bias) -> sigmoid(g) * p
       -> bf16 (stock rounding) -> + z in fp32 (residual) -> bf16 store."""
    pid_j = tl.program_id(0)
    i = tl.program_id(1)
    b = tl.program_id(2)
    j0 = pid_j * BM                                            # a multiple of BM: TMA column offsets stay 16-byte aligned; the last tile runs past N (reads land in
    tok0 = (b * N + i) * N + j0                                #   the next row / zero fill, the 3-D out box is clipped at N).  B*N*N < 2^31 checked by the caller.
    ch = tl.arange(0, D)
    cz = tl.arange(0, C)
    low = tl.load(lnow_ptr + ch); lob = tl.load(lnob_ptr + ch)  # parameter loads first: their latency hides under the tiles' TMA loads
    liw = tl.load(lniw_ptr + cz); lib = tl.load(lnib_ptr + cz)
    xt = x_desc.load([b * D, i * Np + j0])                     # [D, BM] bf16
    zt = z_desc.load([tok0, 0])                                # [BM, C]
    xf = xt.to(tl.float32)                                     # LayerNorm_out over the D channels (axis 0)
    mean = tl.sum(xf, axis=0) / D
    xc = xf - mean[None, :]
    var = tl.sum(xc * xc, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    y = (xc * rstd[None, :] * low[:, None] + lob[:, None]).to(tl.bfloat16)      # [D, BM]  (stock rounding point)
    yT = tl.trans(y)                                                           # [BM, D]  A operand of the out-projection
    zf = zt.to(tl.float32)                                     # LayerNorm_in recomputed (the gate's input)
    mean2 = tl.sum(zf, axis=1) / C
    zc2 = zf - mean2[:, None]
    var2 = tl.sum(zc2 * zc2, axis=1) / C
    rstd2 = tl.rsqrt(var2 + eps)
    xg = (zc2 * rstd2[:, None] * liw[None, :] + lib[None, :]).to(tl.bfloat16)    # [BM, C]
    ncols = tl.arange(0, BN)
    for nb in tl.range(0, C // BN, num_stages=W_STAGES):
        n0 = nb * BN
        wo = woT_desc.load([0, n0])                                            # [D, BN]
        wg = wgT_desc.load([0, n0])                                            # [C, BN]
        acc_p = tl.dot(yT, wo)                                                 # [BM, BN] fp32
        acc_g = tl.dot(xg, wg)
        if HAS_BIAS:
            acc_p = acc_p + tl.load(bo_ptr + n0 + ncols)[None, :]
            acc_g = acc_g + tl.load(bg_ptr + n0 + ncols)[None, :]
        o = tl.sigmoid(acc_g) * acc_p
        if STOCK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)                               # stock: the fused kernel returns bf16; the residual is added by torch afterwards
        if RESIDUAL:
            o = o + zres_desc.load([tok0, n0]).to(tl.float32)
        out_desc.store([b * N + i, j0, n0], tl.reshape(o.to(tl.bfloat16), (1, BM, BN)))


def _t_weights(w):
    """Per-pack constants of the descriptor cells, made once per pack and kept in it: the K3 weight operands woT [D, C] / wogT [C, C] bf16 (W^T, unit stride along
    output channels) and the weight descriptors of both kernels per BN (:func:`_w_descs`) — a weight tensor of a pack never changes, so its descriptor is built once."""
    t = w.get("_desc")
    if t is None:
        t = {"woT": w["wz"].t().contiguous(), "wogT": w["wg_out"].t().contiguous(), "descs": {}}
        w["_desc"] = t
    return t


def _w_descs(w, which, BN):
    """("k1", BN) -> (wg_desc [C, BN] over wgT_in, wp_desc over wpT_in);  ("k3", BN) -> (wo_desc [D, BN] over woT, wg_desc [C, BN] over wogT).  Cached in the pack."""
    t = _t_weights(w)
    d = t["descs"].get((which, BN))
    if d is None:
        C = int(w["wgT_in"].shape[0])
        if which == "k1":
            d = (TensorDescriptor.from_tensor(w["wgT_in"], [C, BN]), TensorDescriptor.from_tensor(w["wpT_in"], [C, BN]))
        else:
            D = int(t["woT"].shape[0])
            d = (TensorDescriptor.from_tensor(t["woT"], [D, BN]), TensorDescriptor.from_tensor(t["wogT"], [C, BN]))
        t["descs"][(which, BN)] = d
    return d


def launch_k1d(z4, w, mask, ab, N, Np, k1, eps, mask_sb, outgoing):
    """K1 impl 'tma2' on z4 [B, N, N, C] bf16 -> planes ab [2, B, D, Np, Np] (incoming: transposed planes, see the module docstring).  One launch, grid
    (token tiles, Np rows, B).  mask: None | fp32 [N, N] (mask_sb = 0) | [B, N, N] (mask_sb = N*N).  Returns True when the planes are transposed (incoming)."""
    B, _, _, C = z4.shape
    D = ab.shape[2]; BM, BN = int(k1["BM"]), int(k1["BN"])
    if B * N * N > INT32_MAX:                                    # z's token-row coordinate is int32 by the descriptor API (kernels.batch_launch_limit names it 'tma_rows>int32' first)
        from .kernels import BatchLimit
        raise BatchLimit("tma_rows>int32", "B*N*N = %d" % (B * N * N))
    if outgoing:
        z_desc = TensorDescriptor.from_tensor(z4.view(B * N * N, C), [BM, C]); zrow_m, zcol_m, msm, msk = N, 0, N, 1
    else:
        z_desc = TensorDescriptor.from_tensor(z4.view(B * N, N * C), [BM, C]); zrow_m, zcol_m, msm, msk = 0, C, 1, N
    ab_desc = TensorDescriptor.from_tensor(ab.view(2 * B * D, Np * Np), [BN, BM])
    wg_desc, wp_desc = _w_descs(w, "k1", BN)
    hb = bool(w.get("has_bias")); dummy = w["ln_in_w"]
    grid = (triton.cdiv(Np, BM), Np, B)
    _k1d[grid](z_desc, ab_desc, mask if mask is not None else dummy, w["ln_in_w"], w["ln_in_b"], wg_desc, wp_desc,
               w["bg_in"] if hb else dummy, w["bp_in"] if hb else dummy,
               N, Np, D, zrow_m, zcol_m, msm, msk, mask_sb, B * D, eps,
               C=C, D2=2 * D, BM=BM, BN=BN, W_STAGES=int(k1["num_stages"]), HAS_MASK=mask is not None, HAS_BIAS=hb, TRANS=not outgoing,
               num_warps=int(k1["num_warps"]))
    return not outgoing


def launch_k3d(x, z4, w, out, N, Np, k3, residual, stock_round, eps):
    """K3 impl 'tma' on x [B, D, Np, Np], z4 / out [B, N, N, C] bf16.  One launch, grid (token tiles, N rows, B)."""
    B, _, _, C = z4.shape
    D = x.shape[1]; BM, BN = int(k3["BM"]), int(k3["BN"])
    if B * N * N > INT32_MAX:                                    # the token-row coordinate (b*N + i)*N + j0 is int32 by the descriptor API; trimul_v4_forward refuses by name first
        from .kernels import BatchLimit
        raise BatchLimit("tma_rows>int32", "B*N*N = %d" % (B * N * N))
    x_desc = TensorDescriptor.from_tensor(x.view(B * D, Np * Np), [D, BM])
    z2 = z4.view(B * N * N, C)
    z_desc = TensorDescriptor.from_tensor(z2, [BM, C])
    zres_desc = TensorDescriptor.from_tensor(z2, [BM, BN])
    out_desc = TensorDescriptor.from_tensor(out.view(B * N, N, C), [1, BM, BN])
    wo_desc, wg_desc = _w_descs(w, "k3", BN)
    hb = bool(w.get("has_bias")); dummy = w["ln_out_w"]
    grid = (triton.cdiv(N, BM), N, B)
    _k3d[grid](x_desc, z_desc, zres_desc, out_desc, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], wo_desc, wg_desc,
               w["b_o"] if hb else dummy, w["b_og"] if hb else dummy, N, Np, eps,
               C=C, D=D, BM=BM, BN=BN, RESIDUAL=bool(residual), STOCK_ROUND=bool(stock_round), HAS_BIAS=hb, W_STAGES=int(k3["num_stages"]),
               num_warps=int(k3["num_warps"]))
