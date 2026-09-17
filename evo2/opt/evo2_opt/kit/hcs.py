"""E62: the vortex-kernels route's gated short inner filter (vortex ``hcs_conv``: ``u = x1 * v`` in bf16, ``u.float()``, the fp32 depthwise
causal FIR of ``_hcs_depthwise_conv_kernel``, ``.to(bf16)``, ``x2 * z``) as the same bf16 product followed by ONE Triton kernel. Per element
the arithmetic is the stock's, in the stock's order: the bf16 product widened to fp32, the tap loop ``acc += w_k * u`` ascending from 0.0
with out-of-range taps loading 0.0, one fp32->bf16 rounding of the sum, one bf16 multiply by x2. Four kernels and three (B, D, L)
intermediates (one bf16, two fp32) become one kernel and none. With E64's channels-first projection output the block's featurizer FIR is
folded in as well (``gate_conv_fold``): the featurizer call hands the raw projection rows on, u = bf16(bf16(fir3 x1) * bf16(fir3 v)) is one
kernel and the conv-gate kernel computes x2 = bf16(fir3 x2) inline - E7's expression per element - so an HCS block never writes the
(B, 3D, L) featurizer output."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from evo2_opt.kit.base import _rne_bf16
from evo2_opt.kit.fir3rows import _fir3_acc
from evo2_opt.kit import i32 as _I32

LEVER = "E62_fused_hcs_gate_conv"
GEOMETRY = (16, 128, 4)          # BLOCK_D, BLOCK_L, num_warps: launch geometry only; per-element arithmetic unchanged
GEOMETRY_U = (2, 1024, 2)        # the folded u kernel
GEOMETRY_FOLD = (8, 128, 8)      # the folded conv-gate kernel
FIR3_TAPS = "_evo2_kit_fir3_taps"  # the attribute a deferred featurizer output carries: its (3D, 1, 3) featurizer filter


@triton.jit
def _hcs_gate_conv_kernel(u_ptr, x2_ptr, w_ptr, o_ptr, D, L, ub, ud, ul, sb, sd, sl, swd, swk, ob, od_, ol_,
                          FIR_LEN: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    tile_mask = mask_d[:, None] & mask_l[None, :]
    urow = pid_b * ub + offs_d[:, None] * ud
    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)
    for k in tl.static_range(FIR_LEN):
        w_k = tl.load(w_ptr + offs_d * swd + k * swk, mask=mask_d, other=0.0).to(tl.float32)      # weight.float(): bf16 -> fp32 is exact
        pos = offs_l - (FIR_LEN - 1) + k
        mask_pos = tile_mask & (pos[None, :] >= 0) & (pos[None, :] < L)
        u_tile = tl.load(u_ptr + urow + pos[None, :] * ul, mask=mask_pos, other=0.0).to(tl.float32)   # u.float()
        acc += w_k[:, None] * u_tile                                                               # the stock kernel's tap statement
    z = _rne_bf16(acc)                                                                             # z.to(u.dtype): fp32 -> bf16
    g = tl.load(x2_ptr + pid_b * sb + offs_d[:, None] * sd + offs_l[None, :] * sl, mask=tile_mask, other=0.0).to(tl.float32)
    out = (g * z).to(tl.bfloat16)                                                                  # x2 * z: one bf16 multiply
    tl.store(o_ptr + pid_b * ob + offs_d[:, None] * od_ + offs_l[None, :] * ol_, out, mask=tile_mask)


@triton.jit
def _hcs_u_kernel(x1_ptr, v_ptr, w1_ptr, wv_ptr, u_ptr, D, L, sb, sd, sl, ub, ud, ul, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    pb = tl.program_id(0)
    pd = tl.program_id(1)
    pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D
    mrow = md[:, None]
    a0 = tl.load(w1_ptr + od * 3 + 0, mask=md, other=0.0).to(tl.float32)
    a1 = tl.load(w1_ptr + od * 3 + 1, mask=md, other=0.0).to(tl.float32)
    a2 = tl.load(w1_ptr + od * 3 + 2, mask=md, other=0.0).to(tl.float32)
    b0 = tl.load(wv_ptr + od * 3 + 0, mask=md, other=0.0).to(tl.float32)
    b1 = tl.load(wv_ptr + od * 3 + 1, mask=md, other=0.0).to(tl.float32)
    b2 = tl.load(wv_ptr + od * 3 + 2, mask=md, other=0.0).to(tl.float32)
    fx1 = _rne_bf16(_fir3_acc(x1_ptr + pb * sb + od[:, None] * sd, ol, sl, L, a0, a1, a2, mrow))     # bf16(fir3 x1): the featurizer output row
    fv = _rne_bf16(_fir3_acc(v_ptr + pb * sb + od[:, None] * sd, ol, sl, L, b0, b1, b2, mrow))       # bf16(fir3 v)
    tl.store(u_ptr + pb * ub + od[:, None] * ud + ol[None, :] * ul, (fx1 * fv).to(tl.bfloat16), mask=mrow & (ol[None, :] < L))   # u = x1 * v (bf16)


@triton.jit
def _hcs_gate_conv_fold_kernel(u_ptr, x2_ptr, w2_ptr, w_ptr, o_ptr, D, L, ub, ud, ul, sb, sd, sl, swd, swk, ob, od_, ol_,
                               FIR_LEN: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    tile_mask = mask_d[:, None] & mask_l[None, :]
    urow = pid_b * ub + offs_d[:, None] * ud
    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)
    for k in tl.static_range(FIR_LEN):
        w_k = tl.load(w_ptr + offs_d * swd + k * swk, mask=mask_d, other=0.0).to(tl.float32)
        pos = offs_l - (FIR_LEN - 1) + k
        mask_pos = tile_mask & (pos[None, :] >= 0) & (pos[None, :] < L)
        u_tile = tl.load(u_ptr + urow + pos[None, :] * ul, mask=mask_pos, other=0.0).to(tl.float32)
        acc += w_k[:, None] * u_tile
    z = _rne_bf16(acc)
    c0 = tl.load(w2_ptr + offs_d * 3 + 0, mask=mask_d, other=0.0).to(tl.float32)
    c1 = tl.load(w2_ptr + offs_d * 3 + 1, mask=mask_d, other=0.0).to(tl.float32)
    c2 = tl.load(w2_ptr + offs_d * 3 + 2, mask=mask_d, other=0.0).to(tl.float32)
    g = _rne_bf16(_fir3_acc(x2_ptr + pid_b * sb + offs_d[:, None] * sd, offs_l, sl, L, c0, c1, c2, mask_d[:, None]))   # bf16(fir3 x2), inline
    tl.store(o_ptr + pid_b * ob + offs_d[:, None] * od_ + offs_l[None, :] * ol_, (g * z).to(tl.bfloat16), mask=tile_mask)


def takes(engine, hcs_conv_symbol, gate, dim_last, fir_length, groups, bias, inference_params, padding_mask, column_split_hyena, u, weight) -> bool:
    """True when vortex's parallel_fir would dispatch this call to hcs_conv (its own condition: gate, use_hcs_kernel, the kernel imported,
    fir_length < 128, groups) AND the call is the scoring shape this lever reproduces: channels-first (B, 3D, L) bf16 input, a depthwise
    (D, 1, fir_length) filter, no skip bias, no generation state, no padding mask, no column split. Anything else is the stock's call."""
    if not (gate and getattr(engine, "use_hcs_kernel", False) and hcs_conv_symbol is not None and fir_length < 128 and groups):
        return False
    if dim_last or bias is not None or inference_params is not None or padding_mask is not None or column_split_hyena:
        return False
    if u.dim() != 3 or u.dtype != torch.bfloat16 or weight.dim() != 3 or weight.dtype != torch.bfloat16:
        return False
    D = u.shape[1] // 3
    return u.shape[1] == 3 * D and tuple(weight.shape) == (D, 1, int(fir_length)) and weight.is_contiguous()


def gate_conv(x1, x2, v, weight, geometry=GEOMETRY):
    """x1, x2, v: (B, D, L) bf16 views of one (B, 3D, L) tensor; weight: (D, 1, FIR) bf16 contiguous -> (B, D, L) bf16 =
    hcs_conv(x1, x2, v, weight, None)."""
    u = x1 * v                                                                                     # the stock's bf16 product, unchanged
    Bn, Dn, Ln = (int(s) for s in u.shape)
    fir = int(weight.shape[-1])
    w2 = weight.reshape(Dn, fir)
    _I32.check(LEVER, f"the gated short-filter input (B={Bn}, D={Dn}, L={Ln})",
               (_I32.strided_extent(u.shape, u.stride()), _I32.strided_extent(x2.shape, x2.stride()), Bn * Dn * Ln),
               f"batch <= {_I32.EXTENT // max(1, _I32.strided_extent((Dn, Ln), (x2.stride(1), x2.stride(2))))} at L={Ln}")
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=u.device)
    bd, bl, nw = geometry
    grid = (Bn, triton.cdiv(Dn, bd), triton.cdiv(Ln, bl))
    _hcs_gate_conv_kernel[grid](u, x2, w2, out, Dn, Ln, u.stride(0), u.stride(1), u.stride(2), x2.stride(0), x2.stride(1), x2.stride(2),
                                w2.stride(0), w2.stride(1), out.stride(0), out.stride(1), out.stride(2), FIR_LEN=fir, BLOCK_D=bd, BLOCK_L=bl, num_warps=nw)
    return out


def defer_featurizer(u_bld, weight):
    """The featurizer call of an HCS block over the channels-first projection output: the raw (B, 3D, L) rows, carrying their featurizer
    filter, for gate_conv_fold in the block's gate call."""
    x = u_bld.permute(0, 2, 1)
    setattr(x, FIR3_TAPS, weight)
    return x


def deferred_taps(u):
    return getattr(u, FIR3_TAPS, None)


def gate_conv_fold(x2r, x1r, vr, taps, hidden_size, flip, weight):
    """x2r, x1r, vr: (B, D, L) raw projection rows (views, L contiguous); taps: the (3D, 1, 3) featurizer filter (rows x2 | x1 | v in the
    projection's channel order); weight: the (D, 1, FIR) inner filter -> (B, D, L) bf16 = hcs_conv(fir3 x1, fir3 x2, fir3 v, weight, None)."""
    Bn, Dn, Ln = (int(s) for s in x1r.shape)
    assert x1r.stride() == vr.stride() == x2r.stride(), (x1r.stride(), vr.stride(), x2r.stride())
    w3 = taps.reshape(3 * hidden_size, 3).contiguous()
    wx2, wx1, wv = w3[:hidden_size], w3[hidden_size:2 * hidden_size], w3[2 * hidden_size:]
    if flip:
        wx1, wx2 = wx2, wx1
    fir = int(weight.shape[-1])
    w2 = weight.reshape(Dn, fir)
    _I32.check(LEVER, f"the projection rows (B={Bn}, D={Dn}, L={Ln})", (_I32.strided_extent(x1r.shape, x1r.stride()), Bn * Dn * Ln),
               f"batch x padded length <= {_I32.EXTENT // max(1, 3 * Dn):,} at width {3 * Dn}")
    u = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1r.device)
    out = torch.empty_like(u)
    bd, bl, nw = GEOMETRY_U
    _hcs_u_kernel[(Bn, triton.cdiv(Dn, bd), triton.cdiv(Ln, bl))](x1r, vr, wx1, wv, u, Dn, Ln, x1r.stride(0), x1r.stride(1), x1r.stride(2),
                                                                 u.stride(0), u.stride(1), u.stride(2), BLOCK_D=bd, BLOCK_L=bl, num_warps=nw)
    bd, bl, nw = GEOMETRY_FOLD
    _hcs_gate_conv_fold_kernel[(Bn, triton.cdiv(Dn, bd), triton.cdiv(Ln, bl))](u, x2r, wx2, w2, out, Dn, Ln, u.stride(0), u.stride(1), u.stride(2),
                                                                              x2r.stride(0), x2r.stride(1), x2r.stride(2), w2.stride(0), w2.stride(1),
                                                                              out.stride(0), out.stride(1), out.stride(2), FIR_LEN=fir, BLOCK_D=bd, BLOCK_L=bl, num_warps=nw)
    return out
