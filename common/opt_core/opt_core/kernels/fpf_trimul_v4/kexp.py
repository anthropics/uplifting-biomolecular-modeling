"""EXPERIMENTAL kernel variants for fpf_trimul_v4 (never served by trimul.fn unless promoted into kernels.py + table.json).
  _k3a : K3 in token-major orientation (tiles [tokens, channels]; x tile transposed once in registers; natural row-major z load / out store)
  _k3t : _k3a with TMA tensor-descriptor loads (x tile, z tile, weight blocks) and a software-pipelined weight-block loop (tl.range num_stages)
  _k1t : K1 with TMA tensor-descriptor loads (z tile, weight blocks) + pipelined weight-block loop
  _k1m : K1 with the gate|proj weight blocks merged into ONE MMA of width 2*BN (acc split in the epilogue)
Same arithmetic and rounding points as kernels._k1c/_k3c (LN fp32 two-pass on register tiles, bf16 MMAs fp32 acc, gate on fp32 acc, bf16 stores)."""
import torch
import triton
import triton.language as tl

HAS_DESC = hasattr(tl, "make_tensor_descriptor")


def ensure_allocator():
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(lambda size, align, stream: torch.empty(int(size), dtype=torch.int8, device="cuda"))


@triton.jit
def _k3a(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wzT_ptr, wgT_ptr, out_ptr,
         N, Npx, x_plane_stride, eps,
         C: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    rows = pid_j * BM + tl.arange(0, BM)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    ch = tl.arange(0, CH)
    ch64 = ch.to(tl.int64)
    cz = tl.arange(0, C)
    cz64 = cz.to(tl.int64)
    xt = tl.load(x_ptr + (ch64 * x_plane_stride)[:, None] + (i * Npx + rows64)[None, :], mask=rmask[None, :], other=0.0)   # [CH, BM]
    xf = xt.to(tl.float32)
    mean = tl.sum(xf, axis=0) / CH
    xc = xf - mean[None, :]
    var = tl.sum(xc * xc, axis=0) / CH
    rstd = tl.rsqrt(var + eps)
    low = tl.load(lnow_ptr + ch)
    lob = tl.load(lnob_ptr + ch)
    xln = (xc * rstd[None, :] * low[:, None] + lob[:, None]).to(tl.bfloat16)
    xlnT = tl.trans(xln)                                                          # [BM, CH]
    ztok = (i * N + rows64) * C
    zt = tl.load(z_ptr + ztok[:, None] + cz64[None, :], mask=rmask[:, None], other=0.0)                                    # [BM, C]
    zf = zt.to(tl.float32)
    mean2 = tl.sum(zf, axis=1) / C
    zc2 = zf - mean2[:, None]
    var2 = tl.sum(zc2 * zc2, axis=1) / C
    rstd2 = tl.rsqrt(var2 + eps)
    liw = tl.load(lniw_ptr + cz)
    lib = tl.load(lnib_ptr + cz)
    zln = (zc2 * rstd2[:, None] * liw[None, :] + lib[None, :]).to(tl.bfloat16)  # [BM, C]
    nc = tl.arange(0, BN)
    for nb in range(0, C // BN):
        n0 = nb * BN
        wz = tl.load(wzT_ptr + (ch64 * C)[:, None] + (n0 + nc)[None, :])          # [CH, BN]
        acc_p = tl.dot(xlnT, wz)                                                   # [BM, BN]
        wg = tl.load(wgT_ptr + (cz64 * C)[:, None] + (n0 + nc)[None, :])          # [C, BN]
        acc_g = tl.dot(zln, wg)
        o = tl.sigmoid(acc_g) * acc_p
        if STOCK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)
        optrs = ztok[:, None] + (n0 + nc).to(tl.int64)[None, :]
        if RESIDUAL:
            o = o + tl.load(z_ptr + optrs, mask=rmask[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptr + optrs, o.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _k3t(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wzT_ptr, wgT_ptr, out_ptr,
         N, Npx, NN, NpNp, eps,
         C: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1)
    j0 = pid_j * BM
    i64 = i.to(tl.int64)
    rows = j0 + tl.arange(0, BM)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    ch = tl.arange(0, CH)
    cz = tl.arange(0, C)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[CH, NpNp], strides=[NpNp, 1], block_shape=[CH, BM])
    z_desc = tl.make_tensor_descriptor(z_ptr, shape=[NN, C], strides=[C, 1], block_shape=[BM, C])
    wz_desc = tl.make_tensor_descriptor(wzT_ptr, shape=[CH, C], strides=[C, 1], block_shape=[CH, BN])
    wg_desc = tl.make_tensor_descriptor(wgT_ptr, shape=[C, C], strides=[C, 1], block_shape=[C, BN])
    xt = x_desc.load([0, i * Npx + j0])                                             # [CH, BM] (pad cols of x are exact zeros; OOB -> 0)
    xf = xt.to(tl.float32)
    mean = tl.sum(xf, axis=0) / CH
    xc = xf - mean[None, :]
    var = tl.sum(xc * xc, axis=0) / CH
    rstd = tl.rsqrt(var + eps)
    low = tl.load(lnow_ptr + ch)
    lob = tl.load(lnob_ptr + ch)
    xln = (xc * rstd[None, :] * low[:, None] + lob[:, None]).to(tl.bfloat16)
    xlnT = tl.trans(xln)
    zt = z_desc.load([i * N + j0, 0])                                               # [BM, C] rows j>=N read the next row's tokens (finite, never stored) or OOB zeros
    zf = zt.to(tl.float32)
    mean2 = tl.sum(zf, axis=1) / C
    zc2 = zf - mean2[:, None]
    var2 = tl.sum(zc2 * zc2, axis=1) / C
    rstd2 = tl.rsqrt(var2 + eps)
    liw = tl.load(lniw_ptr + cz)
    lib = tl.load(lnib_ptr + cz)
    zln = (zc2 * rstd2[:, None] * liw[None, :] + lib[None, :]).to(tl.bfloat16)
    ztok = (i64 * N + rows64) * C
    nc = tl.arange(0, BN)
    for nb in tl.range(0, C // BN, num_stages=STAGES):
        n0 = nb * BN
        wz = wz_desc.load([0, n0])
        wg = wg_desc.load([0, n0])
        acc_p = tl.dot(xlnT, wz)
        acc_g = tl.dot(zln, wg)
        o = tl.sigmoid(acc_g) * acc_p
        if STOCK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)
        optrs = ztok[:, None] + (n0 + nc).to(tl.int64)[None, :]
        if RESIDUAL:
            o = o + tl.load(z_ptr + optrs, mask=rmask[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptr + optrs, o.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _k1t(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, mask_ptr, ab_ptr,
         N, Np, NN, plane_stride, eps,
         C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr, HAS_MASK: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1)
    j0 = pid_j * BM
    i64 = i.to(tl.int64)
    rows = j0 + tl.arange(0, BM)
    rows64 = rows.to(tl.int64)
    lmask = (rows < N) & (i < N)
    cols = tl.arange(0, C)
    z_desc = tl.make_tensor_descriptor(z_ptr, shape=[NN, C], strides=[C, 1], block_shape=[BM, C])
    wg_desc = tl.make_tensor_descriptor(wgT_ptr, shape=[C, 2 * C], strides=[2 * C, 1], block_shape=[C, BN])
    wp_desc = tl.make_tensor_descriptor(wpT_ptr, shape=[C, 2 * C], strides=[2 * C, 1], block_shape=[C, BN])
    r0 = tl.minimum(i, N - 1) * N + j0
    zt = z_desc.load([r0, 0])                                                       # [BM, C]
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
    for nb in tl.range(0, (2 * C) // BN, num_stages=STAGES):
        n0 = nb * BN
        wg = wg_desc.load([0, n0])
        wp = wp_desc.load([0, n0])
        acc_g = tl.dot(x, wg)
        acc_p = tl.dot(x, wp)
        val = tl.sigmoid(acc_g) * acc_p
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)
        optr = ab_ptr + obase[:, None] + ((n0 + ncols).to(tl.int64) * plane_stride)[None, :]
        tl.store(optr, val.to(tl.bfloat16), mask=smask[:, None])


@triton.jit
def _k1m(z_ptr, lnw_ptr, lnb_ptr, w2T_ptr, mask_ptr, ab_ptr,
         N, Np, plane_stride, eps,
         C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, HAS_MASK: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    rows = pid_j * BM + tl.arange(0, BM)
    rows64 = rows.to(tl.int64)
    lmask = (rows < N) & (i < N)
    cols = tl.arange(0, C)
    cols64 = cols.to(tl.int64)
    zt = tl.load(z_ptr + ((i * N + rows64) * C)[:, None] + cols64[None, :], mask=lmask[:, None], other=0.0)
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, axis=1) / C
    zc = zf - mean[:, None]
    var = tl.sum(zc * zc, axis=1) / C
    rstd = tl.rsqrt(var + eps)
    lnw = tl.load(lnw_ptr + cols)
    lnb = tl.load(lnb_ptr + cols)
    x = (zc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)
    if HAS_MASK:
        mv = tl.load(mask_ptr + i * N + rows64, mask=lmask, other=0.0).to(tl.float32)
    obase = i * Np + rows64
    smask = rows < Np
    ncols = tl.arange(0, BN)
    n2 = tl.arange(0, 2 * BN)
    for nb in range(0, (2 * C) // BN):
        n0 = nb * BN
        w2 = tl.load(w2T_ptr + (cols64 * (4 * C))[:, None] + (nb * 2 * BN + n2)[None, :])    # [C, 2BN] = [gate block | proj block]
        acc = tl.dot(x, w2)                                                                     # [BM, 2BN]
        acc3 = tl.permute(tl.reshape(acc, (BM, 2, BN)), (0, 2, 1))                             # [BM, BN, 2]
        g, p = tl.split(acc3)
        val = tl.sigmoid(g) * p
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)
        optr = ab_ptr + obase[:, None] + ((n0 + ncols).to(tl.int64) * plane_stride)[None, :]
        tl.store(optr, val.to(tl.bfloat16), mask=smask[:, None])


def extra_weights(w, BN=None):
    if "wzT" not in w:
        w["wzT"] = w["wz"].t().contiguous()            # [CH, C_z]
        w["wgT_out"] = w["wg_out"].t().contiguous()    # [C_z in, C_z out]
    if BN is not None and ("w2T", BN) not in w:
        C = w["C"]; g, p = w["wgT_in"], w["wpT_in"]     # [C, 2C]
        blocks = []
        for n0 in range(0, 2 * C, BN):
            blocks += [g[:, n0:n0 + BN], p[:, n0:n0 + BN]]
        w[("w2T", BN)] = torch.cat(blocks, 1).contiguous()   # [C, 4C]
    return w


def launch_k3a(x, z, w, out, N, Np, c, residual=True, stock_round=True, eps=1e-5):
    C = z.shape[-1]; CH = x.shape[0]; extra_weights(w)
    _k3a[(triton.cdiv(N, c["BM"]), N)](x, z, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wzT"], w["wgT_out"], out, N, Np, Np * Np, eps,
                                        C=C, CH=CH, BM=c["BM"], BN=c["BN"], RESIDUAL=residual, STOCK_ROUND=stock_round, num_warps=c["num_warps"], num_stages=c["num_stages"])


def launch_k3t(x, z, w, out, N, Np, c, residual=True, stock_round=True, eps=1e-5):
    C = z.shape[-1]; CH = x.shape[0]; extra_weights(w); ensure_allocator()
    _k3t[(triton.cdiv(N, c["BM"]), N)](x, z, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wzT"], w["wgT_out"], out, N, Np, N * N, Np * Np, eps,
                                        C=C, CH=CH, BM=c["BM"], BN=c["BN"], STAGES=c["num_stages"], RESIDUAL=residual, STOCK_ROUND=stock_round, num_warps=c["num_warps"])


def launch_k1t(z, w, mask, ab, N, Np, c, eps=1e-5):
    C = z.shape[-1]; ensure_allocator()
    _k1t[(triton.cdiv(Np, c["BM"]), Np)](z, w["ln_in_w"], w["ln_in_b"], w["wgT_in"], w["wpT_in"], mask if mask is not None else z, ab, N, Np, N * N, Np * Np, eps,
                                          C=C, BM=c["BM"], BN=c["BN"], STAGES=c["num_stages"], HAS_MASK=mask is not None, num_warps=c["num_warps"])


def launch_k1m(z, w, mask, ab, N, Np, c, eps=1e-5):
    C = z.shape[-1]; extra_weights(w, c["BN"])
    _k1m[(triton.cdiv(Np, c["BM"]), Np)](z, w["ln_in_w"], w["ln_in_b"], w[("w2T", c["BN"])], mask if mask is not None else z, ab, N, Np, Np * Np, eps,
                                          C=C, BM=c["BM"], BN=c["BN"], HAS_MASK=mask is not None, num_warps=c["num_warps"], num_stages=c["num_stages"])
