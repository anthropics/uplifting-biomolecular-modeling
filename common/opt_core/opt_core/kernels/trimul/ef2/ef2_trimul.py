"""ef2_trimul — triangle-multiplication levers for the grad-mode pair stack of Biohub ESMFold2(-Experimental)
(``TriangleMultiplicativeUpdate`` inside ``PairUpdateBlock``): a fused forward + frozen-weight backward (FAST class) and a
tile table for the stock cuEquivariance kernels (EXACT class).

    import ef2_trimul
    h = ef2_trimul.enable(model, variant="fused")        # or variant="cueq_tiles"
    ...
    ef2_trimul.disable(model)

variant="fused" — FAST class (tier 2).  One autograd Function per triangle multiplication computes the block's residual
update ``pair + TriMul(pair, mask)``:
  fwd   LN_in (Triton; writes x_in as zero-padded rows AND unpadded rows) -> [value|gate] gated dual GEMM (the vendored
        Biohub Triton kernel) on the padded rows, a|b written channel-major (2C, B, Np, Np), Np = ceil8(N) -> the O(N^3)
        contraction as ONE cuBLAS strided-batched GEMM over the padded per-channel matrices (every GEMM dimension a multiple
        of 8 keeps cuBLAS on its aligned sm_90 kernels; an odd N runs an align2 sm_80 kernel at ~1/3 the throughput) ->
        LN_out (Triton, channel-major in / rows out) -> out-projection x sigmoid(out-gate) + residual (the vendored kernel).
  bwd   d pair ONLY (design use: every parameter is frozen — no weight / LN-affine / mask gradients are formed): the two
        vendored partial-gradient kernels, three cuBLAS dgrad GEMMs, two cuBLAS batched GEMMs for the contraction, and
        Triton LN_out / LN_in backward kernels (the latter adds the residual's gradient).  x_in / x_out are recomputed from
        the saved tensors (one LN pass each) rather than saved.  Saved for backward: the block input (the residual stream,
        alive anyway), a|b (2C x B Np^2 bf16), the contraction output (C x B Np^2 bf16), the padded mask.
  Rounding points (vs the cuEquivariance-backed stock path): identical in kind — x_in = bf16(LN fp32); a|b = bf16(fp32 value
  x sigmoid(fp32 gate) x mask); contraction bf16 x bf16 -> fp32 accumulate -> bf16; x_out = bf16(LN fp32) — except the last
  stage: new = bf16(pair + fp32 proj x sigmoid(fp32 gate)) is ONE rounding here and in ef2_autograd_kernels K-A2 (cuEq
  rounds the delta, then the sum).  Accumulation order differs (tiles, the zero-padded contraction, LN reductions).  The
  backward keeps LayerNorm row sums in fp32 where the reference chain stores bf16 intermediates.  Error vs an fp64 gold of
  the stock math is within the stock kernels' own error (k/test_ef2_trimul.py, <= 1.25x).
  The four gated-GEMM launches go through a per-compute-capability launch table (sm_90: the kit's persistent weight-resident
  TMA kernels for the dual GEMM fwd / bwd partials, tuned tiles for the vendored out-gate kernels; tensor-equal to the vendored
  launches — see `_GG_SM90`); a capability without an entry runs the vendored launches and ``h.gemm`` names it.
  bf16 CUDA pairs under grad take the fused path; anything else (fp32 pair, CPU, no-grad when only_under_grad, training-time
  row dropout) runs the module's own forward — counted in ``h.stats`` (served / fallback), never silent.
  Integration: composes with ``ef2_autograd_kernels.enable(model, trimul=None, transition=..., checkpoint=...)`` (a block
  whose agk config replaces the triangle multiplication itself is refused by name); patches
  TriangleMultiplicativeUpdate.forward and makes the block's row_drop the identity for its (residual-inclusive) outputs
  (re-applied if set_kernel_backend() rebuilds row_drop later).

variant="cueq_tiles" — EXACT class.  The stock path with set_kernel_backend('cuequivariance') runs four cuequivariance_ops
  Triton GEMM kernels per call whose tile shape comes from the library's tuning cache; the cache shipped for sm_90 has no
  entry for d_pair-256 bf16 shapes, so every call runs the library default (64x32x32, 4 warps).  Tile shape (M/N tiling,
  warps, stages; TILE_K is kept at the default 32) does not enter the kernels' per-element arithmetic on this pin, so tuned
  tiles give tensor-equal outputs AND input-gradients — bitwise PER PIN (torch 2.11 / triton 3.6 / cuequivariance 0.10;
  k/test_ef2_trimul.py::test_cueq_tiles_bitwise re-proves it, 257..700 tokens; only the mask-gradient partials, which nothing
  consumes, change).  Row counts below 65536 (N < 256 at batch 1) keep the library default (launch-bound; no measurable gain).  The tile is
  one entry PER COMPUTE CAPABILITY (torch.cuda.get_device_capability(): sm_90 and sm_80 carry one, each proven tensor-equal on its own
  card; a capability without an entry keeps the library default and says so — entries=0 reason=cc_untuned).  enable() writes tuned entries
  for every row-count bucket of this model's shapes into the library's IN-PROCESS table (nothing on disk); disable()
  restores it.  Integration: this is process-global library state — enable it BEFORE any CUDA-graph capture of the trunk
  (a captured graph replays whatever tiles were live at capture); it needs no module patch and is transparent to
  torch.compile (the custom op is unchanged; the table is read per call).
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl
from torch import Tensor

import transformers.models.esmfold2.modeling_esmfold2_common as C

__all__ = ["enable", "disable", "trimul_residual", "pair_trimul_residual", "VARIANTS", "Handle"]

VARIANTS = ("fused", "cueq_tiles")
PAD = 8                     # token-axis padding of the channel-major intermediates (cuBLAS alignment class)


# =============================================================================================== Triton kernels
# Conventions.  `mp` = the PADDED linear row index b*Np*Np + i*Np + j: contiguous for the channel-major intermediates
# (ab, o, d_o, d_ab are (channels, B, Np, Np)) and for padded row tensors ((B*Np*Np, C)); `row` = the matching row of
# the unpadded (B, N, N, C) tensors; pad slots (i >= N or j >= N) are `valid == False`: gathered as 0, written as 0 into
# padded outputs, skipped in unpadded outputs.  LayerNorm statistics are two-pass fp32 over the full row.
import transformers.models.esmfold2.kernels.fused_dual_gemm as _FDG          # vendored Biohub: gated dual GEMM fwd / bwd partials
import transformers.models.esmfold2.kernels.trimul_with_residual as _TWR    # vendored Biohub: out-gate dual GEMM + residual fwd / bwd partials


@triton.jit
def _rows_of(offs_mp, N, Np):
    npnp = Np * Np
    b = offs_mp // npnp
    rem = offs_mp - b * npnp
    i = rem // Np
    j = rem - i * Np
    valid = (i < N) & (j < N)
    row = (b * N + i) * N + j
    return valid, row


@triton.jit
def _ln_tile(x, gam, bet, eps, CDIM: tl.constexpr):
    """LayerNorm of full fp32 rows: returns (xhat, rstd, bf16(xhat*gam + bet))."""
    mean = tl.sum(x, axis=1) / CDIM
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / CDIM
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd[:, None]
    return xhat, rstd, (xhat * gam[None, :] + bet[None, :]).to(tl.bfloat16)


@triton.jit
def _ln_bwd_tile(dy, xhat, rstd, gam, CDIM: tl.constexpr):
    """d x of y = xhat*gam + bet given d y (fp32 full rows)."""
    dxh = dy * gam[None, :]
    m1 = tl.sum(dxh, axis=1) / CDIM
    m2 = tl.sum(dxh * xhat, axis=1) / CDIM
    return (dxh - m1[:, None] - xhat * m2[:, None]) * rstd[:, None]


@triton.jit
def _k_ln_in(z_ptr, gam_ptr, bet_ptr, xp_ptr, xu_ptr, N, Np, M_pad, eps, CDIM: tl.constexpr, TILE_M: tl.constexpr):
    """x = bf16(LN_in(z[row])): xp[mp] (padded rows, 0 at pad slots) and xu[row] (unpadded rows).  grid = (M_pad / TILE_M,)."""
    pid = tl.program_id(0)
    N = N.to(tl.int64); Np = Np.to(tl.int64); M_pad = M_pad.to(tl.int64)
    offs_mp = pid.to(tl.int64) * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    valid, row = _rows_of(offs_mp, N, Np)
    cols = tl.arange(0, CDIM)
    z = tl.load(z_ptr + row[:, None] * CDIM + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    gam = tl.load(gam_ptr + cols); bet = tl.load(bet_ptr + cols)
    _xh, _rs, xn = _ln_tile(z, gam, bet, eps, CDIM)
    tl.store(xu_ptr + row[:, None] * CDIM + cols[None, :], xn, mask=valid[:, None])
    xn = tl.where(valid[:, None], xn, 0.0).to(tl.bfloat16)
    tl.store(xp_ptr + offs_mp[:, None] * CDIM + cols[None, :], xn)


@triton.jit
def _k_ln_out(o_ptr, gam_ptr, bet_ptr, x_ptr, N, Np, M_pad, eps, CDIM: tl.constexpr, TILE_M: tl.constexpr):
    """x[row] = bf16(LN_out(o[:, mp])): channel-major padded in (transpose on load), unpadded rows out."""
    pid = tl.program_id(0)
    N = N.to(tl.int64); Np = Np.to(tl.int64); M_pad = M_pad.to(tl.int64)
    offs_mp = pid.to(tl.int64) * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    valid, row = _rows_of(offs_mp, N, Np)
    cols = tl.arange(0, CDIM); cols64 = cols.to(tl.int64)
    o = tl.load(o_ptr + cols64[None, :] * M_pad + offs_mp[:, None]).to(tl.float32)
    gam = tl.load(gam_ptr + cols); bet = tl.load(bet_ptr + cols)
    _xh, _rs, xn = _ln_tile(o, gam, bet, eps, CDIM)
    tl.store(x_ptr + row[:, None] * CDIM + cols[None, :], xn, mask=valid[:, None])


@triton.jit
def _k_ln_out_bwd(o_ptr, gam_ptr, dx_ptr, do_ptr, N, Np, M_pad, eps, CDIM: tl.constexpr, TILE_M: tl.constexpr):
    """d o[:, mp] = LN_out_bwd(d x_out[row]) (channel-major padded out, 0 at pad slots)."""
    pid = tl.program_id(0)
    N = N.to(tl.int64); Np = Np.to(tl.int64); M_pad = M_pad.to(tl.int64)
    offs_mp = pid.to(tl.int64) * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    valid, row = _rows_of(offs_mp, N, Np)
    cols = tl.arange(0, CDIM); cols64 = cols.to(tl.int64)
    o = tl.load(o_ptr + cols64[None, :] * M_pad + offs_mp[:, None]).to(tl.float32)
    gam = tl.load(gam_ptr + cols)
    xh, rs, _y = _ln_tile(o, gam, gam, eps, CDIM)
    dx = tl.load(dx_ptr + row[:, None] * CDIM + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    d_o = _ln_bwd_tile(dx, xh, rs, gam, CDIM)
    d_o = tl.where(valid[:, None], d_o, 0.0)
    tl.store(do_ptr + cols64[None, :] * M_pad + offs_mp[:, None], d_o.to(tl.bfloat16))


@triton.jit
def _k_ln_in_bwd(z_ptr, gam_ptr, dxp_ptr, dxu_ptr, dn_ptr, dz_ptr, N, Np, M_pad, eps, CDIM: tl.constexpr, TILE_M: tl.constexpr):
    """dz[row] = LN_in_bwd(dxp[mp] + dxu[row]) + dn[row]  (dxp: d x_in from the [value|gate] GEMM, padded rows; dxu: d x_in
    from the out-gate GEMM, unpadded rows; dn: d new through the residual)."""
    pid = tl.program_id(0)
    N = N.to(tl.int64); Np = Np.to(tl.int64); M_pad = M_pad.to(tl.int64)
    offs_mp = pid.to(tl.int64) * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    valid, row = _rows_of(offs_mp, N, Np)
    cols = tl.arange(0, CDIM)
    z = tl.load(z_ptr + row[:, None] * CDIM + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    gam = tl.load(gam_ptr + cols)
    xh, rs, _y = _ln_tile(z, gam, gam, eps, CDIM)
    dx = tl.load(dxp_ptr + offs_mp[:, None] * CDIM + cols[None, :]).to(tl.float32)
    dx = dx + tl.load(dxu_ptr + row[:, None] * CDIM + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    dz = _ln_bwd_tile(dx, xh, rs, gam, CDIM)
    dz = dz + tl.load(dn_ptr + row[:, None] * CDIM + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    tl.store(dz_ptr + row[:, None] * CDIM + cols[None, :], dz.to(tl.bfloat16), mask=valid[:, None])


# =============================================================================================== host side
_LN_TILE_M = 16
_LN_WARPS = 4


def _padded(n: int) -> int:
    return -(-n // PAD) * PAD


def _weights(engine: "C.TriangleMultiplicativeBlock", cache: dict) -> dict:
    """bf16 projection weights (cast once, cached per engine, refreshed when a parameter version changes); LN affine fp32."""
    key = id(engine)
    ver = tuple(int(p._version) for p in engine.parameters())
    ent = cache.get(key)
    if ent is not None and ent["ver"] == ver:
        return ent
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()
    f32 = lambda t: t.detach().float().contiguous()
    p_in, g_in = engine.split_kernel_weights()                   # value rows [0,2C) / gate rows [2C,4C) of proj_bundle.weight
    ent = dict(ver=ver, engine_ref=engine,                       # keep the engine alive so id() cannot be recycled while cached
               w_val=bf(p_in), w_gate=bf(g_in),
               g_in=f32(engine.norm_start.weight), b_in=f32(engine.norm_start.bias),
               g_out=f32(engine.norm_mix.weight), b_out=f32(engine.norm_mix.bias),
               w_p=bf(engine.proj_emit.weight), w_g=bf(engine.proj_gate.weight),   # (C, C) each
               flow=engine.flow)
    ent["w_in"] = torch.cat([ent["w_val"], ent["w_gate"]], dim=0).contiguous()      # (4C, C) = [value; gate], the dual-GEMM d x order
    cache[key] = ent
    return ent


def _grid_rows(M_pad: int):
    return (triton.cdiv(M_pad, _LN_TILE_M),)


def _ln_in(z, w, N, Np, M, M_pad, Cd):
    xp = torch.empty((M_pad, Cd), device=z.device, dtype=torch.bfloat16)
    xu = torch.empty((M, Cd), device=z.device, dtype=torch.bfloat16)
    _k_ln_in[_grid_rows(M_pad)](z, w["g_in"], w["b_in"], xp, xu, N, Np, M_pad, C._EPS, CDIM=Cd, TILE_M=_LN_TILE_M, num_warps=_LN_WARPS)
    return xp, xu


def _ln_out(o, w, N, Np, M, M_pad, Cd):
    x = torch.empty((M, Cd), device=o.device, dtype=torch.bfloat16)
    _k_ln_out[_grid_rows(M_pad)](o, w["g_out"], w["b_out"], x, N, Np, M_pad, C._EPS, CDIM=Cd, TILE_M=_LN_TILE_M, num_warps=_LN_WARPS)
    return x


# ----------------------------------------------------------------------------------------------- the gated-GEMM launch table
# The four GEMM-class launches of the fused path — [value|gate] dual GEMM fwd / bwd partials on the padded rows, out-gate GEMM +
# residual fwd / bwd partials on the unpadded rows — all have K = d_pair = 256.  The vendored kernels tile K in 32/64-wide steps
# and re-stream their operand tiles per 64-row x 64-column output tile: the dual-GEMM backward call moves ~2.2 GB of operand tiles
# through L2 at 431 tokens (23 k CTAs x 96 KB) — L2-bandwidth bound at 17-21 % of bf16 peak / 40-50 % of HBM, and no tensor-equal
# retile of that structure measures better than x1.04 (a 343-config sweep).  Per compute capability this table names, per
# launch, EITHER the kit's persistent kernels below (`_k_gg_dual_fwd` / `_k_gg_dual_bwd`: grid = column chunks x programs; a
# CTA loads its 64-column chunk of both weights ONCE (SMEM-resident) and walks the row tiles with TMA-fed multi-stage loads of the
# full-K x tile — one K=256 dot per operand — and TMA stores of the epilogue) OR the vendored kernel launched with a tuned tile in
# place of its one built-in config.  Per-element arithmetic is the vendored kernels' own in both routes (fp32 accumulation over K
# in the same 16-wide steps from zero, the same epilogue expressions and rounding points): every output and input-gradient is
# tensor-equal to today's launches — bitwise PER PIN (torch 2.11 / triton 3.6; k/test_ef2_trimul.py::test_gemm_table_bitwise
# re-proves it on the card at 257..800 tokens).  sm_90 (H100 80GB HBM3), per call at 431 / 800 tokens: dual bwd 482 -> 330 /
# 1574 -> 1063 us, dual fwd 283 -> 249 / 887 -> 815, out-gate bwd 299 -> 231 / 955 -> 733, out-gate fwd 218 -> 208 / 676 -> 639
# (60-67 % of HBM on the backward partials).  A capability without an entry launches the vendored kernels through their own
# wrappers — today's path, BY NAME: Handle.gemm = "vendored:cc_untuned:sm_NN" on the lever line (never silent, never a refusal).
_GG_SM90 = {
    "dual_fwd": dict(route="kit", TILE_M=64, TILE_N=64, num_warps=4, STAGES=3),
    "dual_bwd": dict(route="kit", TILE_M=64, TILE_N=64, num_warps=4, STAGES=4),
    "og_fwd": dict(route="vendored_tiles", TILE_M=64, TILE_N=128, TILE_K=64, GROUP_M=8, num_warps=4, num_stages=4),
    "og_bwd": dict(route="vendored_tiles", TILE_M=64, TILE_N=64, TILE_K=64, GROUP_M=8, num_warps=4, num_stages=3),
}
_GG_BY_CC = {(9, 0): _GG_SM90}
GEMM_TABLE = {"use": True}          # process switch for A/B instruments (k/ tests, benches): use=False -> the vendored launches on every card
_GG_STATE = {"sms": {}}


def _gg_table(cc: tuple | None = None):
    """(table, word): the launch table of compute capability `cc` (default: the current device's) and its lever word ("sm_90"), or
    (None, "vendored:cc_untuned:sm_NN") when this kit carries no entry — (None, "vendored:off") when GEMM_TABLE["use"] is False."""
    if not GEMM_TABLE["use"]:
        return None, "vendored:off"
    cc = _device_cc() if cc is None else tuple(cc)
    t = _GG_BY_CC.get(cc)
    return (t, f"sm_{cc[0]}{cc[1]}") if t is not None else (None, f"vendored:{CC_UNTUNED}:sm_{cc[0]}{cc[1]}")


def _gg_programs(dev: torch.device, nchunk: int, n_tiles: int) -> int:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    sms = _GG_STATE["sms"].get(i)
    if sms is None:
        sms = _GG_STATE["sms"][i] = torch.cuda.get_device_properties(i).multi_processor_count
    return max(1, min(sms // nchunk, n_tiles))


def _gg_scratch(size: int, align: int, stream):
    return torch.empty(max(int(size), 1), dtype=torch.int8, device="cuda")      # torch's caching allocator: CUDA-graph capturable


def _gg_tma_allocator() -> None:
    """Device-side TMA descriptors take their scratch from triton's allocator, a context variable on this pin (triton 3.6): set it
    in the launching thread every time — autograd's backward runs on its own thread, where a set from the forward is not seen."""
    triton.set_allocator(_gg_scratch)


@triton.jit
def _k_gg_dual_fwd(x_ptr, w1_ptr, w2_ptr, mask_ptr, out_ptr, M, NPROG,
                   N: tl.constexpr, K: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, HAS_MASK: tl.constexpr, STAGES: tl.constexpr):
    """out (N, M) channel-major bf16 = sigmoid(x w1^T) * (x w2^T) [* mask[m]]  — x (M, K) bf16 rows, w1 / w2 (N, K) bf16.
    grid = (N / TILE_N, NPROG): the CTA's weight chunk is loaded once; row tiles pid_p, pid_p + NPROG, ... are TMA-loaded full-K."""
    pid_n = tl.program_id(0); pid_p = tl.program_id(1)
    n0 = pid_n * TILE_N
    offs_n = (n0 + tl.arange(0, TILE_N)).to(tl.int64); offs_k = tl.arange(0, K).to(tl.int64)
    w1c = tl.load(w1_ptr + offs_n[None, :] * K + offs_k[:, None])            # (K, TILE_N): the gate weight chunk, transposed on load
    w2c = tl.load(w2_ptr + offs_n[None, :] * K + offs_k[:, None])            # the value weight chunk
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[TILE_M, K])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N, M], strides=[M.to(tl.int64), 1], block_shape=[TILE_N, TILE_M])
    num_mt = tl.cdiv(M, TILE_M)
    for mt in tl.range(pid_p, num_mt, NPROG, num_stages=STAGES):
        m0 = mt * TILE_M
        x = x_desc.load([m0, 0])                                             # rows >= M read as 0 (TMA bounds)
        gate = tl.dot(x, w1c); val = tl.dot(x, w2c)                          # fp32, K in 16-wide steps from zero = the vendored order
        delta = tl.sigmoid(gate) * val
        if HAS_MASK:
            offs_m = m0 + tl.arange(0, TILE_M)
            mk = tl.load(mask_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
            delta = delta * mk[:, None]
        out_desc.store([n0, m0], tl.trans(delta).to(out_ptr.dtype.element_ty))   # rows >= M clipped (TMA bounds)


@triton.jit
def _k_gg_dual_bwd(dab_ptr, x_ptr, w1_ptr, w2_ptr, mask_ptr, gc_ptr, M, NPROG,
                   N: tl.constexpr, K: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, HAS_MASK: tl.constexpr, STAGES: tl.constexpr):
    """gc (M, 2N) bf16 = [d value logits | d gate logits] of ab = sigmoid(x w1^T) * (x w2^T) * mask given d ab (N, M) channel-major:
    g = sigmoid(gate); go = d ab[n, m] (* mask[m]); d val = go * g; d gate = go * val * g * (1 - g)  (the vendored partials, verbatim;
    the mask-gradient partials the vendored kernel also writes are consumed by nothing on this path and are not formed)."""
    pid_n = tl.program_id(0); pid_p = tl.program_id(1)
    n0 = pid_n * TILE_N
    offs_n = (n0 + tl.arange(0, TILE_N)).to(tl.int64); offs_k = tl.arange(0, K).to(tl.int64)
    w1c = tl.load(w1_ptr + offs_n[None, :] * K + offs_k[:, None])
    w2c = tl.load(w2_ptr + offs_n[None, :] * K + offs_k[:, None])
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[TILE_M, K])
    dab_desc = tl.make_tensor_descriptor(dab_ptr, shape=[N, M], strides=[M.to(tl.int64), 1], block_shape=[TILE_N, TILE_M])
    gc_desc = tl.make_tensor_descriptor(gc_ptr, shape=[M, 2 * N], strides=[2 * N, 1], block_shape=[TILE_M, TILE_N])
    num_mt = tl.cdiv(M, TILE_M)
    for mt in tl.range(pid_p, num_mt, NPROG, num_stages=STAGES):
        m0 = mt * TILE_M
        x = x_desc.load([m0, 0])
        gate = tl.dot(x, w1c); val = tl.dot(x, w2c)
        g = tl.sigmoid(gate)
        go = tl.trans(dab_desc.load([n0, m0])).to(tl.float32)
        if HAS_MASK:
            offs_m = m0 + tl.arange(0, TILE_M)
            mk = tl.load(mask_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
            go = go * mk[:, None]
        d_val = (go * g).to(gc_ptr.dtype.element_ty)
        gc_desc.store([m0, n0], d_val)
        d_gate = (go * val * g * (1.0 - g)).to(gc_ptr.dtype.element_ty)
        gc_desc.store([m0, N + n0], d_gate)


def _gate_proj(xp, w, m_pad, M_pad, Cd):
    """ab (2C, M_pad) bf16 = sigmoid(x @ w_gate^T) * (x @ w_val^T) [* mask], written channel-major (the table's kernel, else the vendored fused kernel)."""
    Nout = 2 * Cd
    ab = torch.empty((Nout, M_pad), device=xp.device, dtype=torch.bfloat16)
    t = _gg_table()[0]
    e = t["dual_fwd"] if t is not None else None
    if e is not None and e["route"] == "kit":
        _gg_tma_allocator()
        nchunk = Nout // e["TILE_N"]; nprog = _gg_programs(xp.device, nchunk, triton.cdiv(M_pad, e["TILE_M"]))
        _k_gg_dual_fwd[(nchunk, nprog)](xp, w["w_gate"], w["w_val"], m_pad if m_pad is not None else w["g_in"], ab, M_pad, nprog, N=Nout, K=Cd,
                                         TILE_M=e["TILE_M"], TILE_N=e["TILE_N"], HAS_MASK=m_pad is not None, STAGES=e["STAGES"], num_warps=e["num_warps"])
        return ab
    grid = lambda meta: (triton.cdiv(M_pad, meta["TILE_M"]), Nout // meta["TILE_N"])
    _FDG._gated_dual_gemm_kernel[grid](xp, w["w_gate"], w["w_val"], m_pad if m_pad is not None else w["g_in"], ab, M_pad, Nout, Cd,
                                       HAS_MASK=m_pad is not None, TRANSPOSE_OUT=True, NEEDS_INT64=(M_pad * Nout >= 2**31 - 1))
    return ab


def _gate_proj_bwd(xp, w, m_pad, dab, M_pad, Cd):
    """d x_in (M_pad, C) bf16 of the gated dual GEMM: partials kernel (the table's, else the vendored) -> [d value | d gate] logits
    (bf16), then ONE cuBLAS GEMM against [w_val; w_gate].  `dab` is the (2C, M_pad) gradient of ab (one tensor: a = rows [0, C))."""
    Nout = 2 * Cd
    gc = torch.empty((M_pad, 2 * Nout), device=xp.device, dtype=torch.bfloat16)
    t = _gg_table()[0]
    e = t["dual_bwd"] if t is not None else None
    if e is not None and e["route"] == "kit":
        _gg_tma_allocator()
        nchunk = Nout // e["TILE_N"]; nprog = _gg_programs(xp.device, nchunk, triton.cdiv(M_pad, e["TILE_M"]))
        _k_gg_dual_bwd[(nchunk, nprog)](dab, xp, w["w_gate"], w["w_val"], m_pad if m_pad is not None else w["g_in"], gc, M_pad, nprog, N=Nout, K=Cd,
                                         TILE_M=e["TILE_M"], TILE_N=e["TILE_N"], HAS_MASK=m_pad is not None, STAGES=e["STAGES"], num_warps=e["num_warps"])
        return torch.mm(gc, w["w_in"])
    g_val = gc[:, :Nout]; g_gate = gc[:, Nout:]
    gmask = torch.empty((triton.cdiv(Nout, _FDG._BWD_TILE_N), M_pad) if m_pad is not None else (1,), device=xp.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M_pad, meta["TILE_M"]), Nout // meta["TILE_N"])
    _FDG._gated_dual_gemm_backward_kernel[grid](dab[:Cd], dab[Cd:], xp, w["w_gate"], w["w_val"], m_pad if m_pad is not None else w["g_in"],
                                                g_gate, g_val, gmask, M_pad, Nout, Cd, HALF_N=Cd, HAS_MASK=m_pad is not None,
                                                GRAD_OUT_TRANSPOSED=True, GRAD_OUT_SPLIT=True, NEEDS_INT64=(M_pad * 2 * Nout >= 2**31 - 1))
    return torch.mm(gc, w["w_in"])


def _tile_kw(e: dict) -> dict:
    return dict(TILE_M=e["TILE_M"], TILE_N=e["TILE_N"], TILE_K=e["TILE_K"], GROUP_M=e["GROUP_M"], num_warps=e["num_warps"], num_stages=e["num_stages"])


def _out_gate_residual(xu, x_out, w, z2d, N):
    """new (M, C) = z + sigmoid(x_in @ w_g^T) * (x_out @ w_p^T)  (vendored fused kernel, one rounding; the table's tile where it has one)."""
    t = _gg_table()[0]
    e = t["og_fwd"] if t is not None else None
    if e is not None and e["route"] == "vendored_tiles":
        M, K = xu.shape; Nc = w["w_g"].shape[0]
        out = z2d.new_empty((M, Nc))
        grid = (triton.cdiv(M, e["TILE_M"]), triton.cdiv(Nc, e["TILE_N"]))
        _TWR._gated_gemm_with_residual_kernel.fn[grid](xu, x_out, w["w_g"], w["w_p"], z2d, z2d, out, M, Nc, K, STRIDE_B=N * N, N_COL=N, PRECISION=0,
                                                        HAS_DROP_MASK=False, NEEDS_INT64=(M * K >= 2**31 - 1) or (M * Nc >= 2**31 - 1), **_tile_kw(e))
        return out
    return _TWR._gated_gemm_with_residual_fwd(xu, x_out, w["w_g"], w["w_p"], z2d, None, N, N, precision=0)


def _out_gate_residual_bwd(dn2d, xu, x_out, w, N):
    """(d x_in, d x_out) (M, C) bf16 of the out-gate stage: vendored partials kernel (the table's tile where it has one), then two cuBLAS GEMMs."""
    M, K = xu.shape; Nc = w["w_g"].shape[0]
    gg = torch.empty((M, Nc), device=xu.device, dtype=torch.bfloat16)
    gv = torch.empty((M, Nc), device=xu.device, dtype=torch.bfloat16)
    gdb = torch.empty((1,), device=xu.device, dtype=torch.float32)
    t = _gg_table()[0]
    e = t["og_bwd"] if t is not None else None
    if e is not None and e["route"] == "vendored_tiles":
        grid = (triton.cdiv(M, e["TILE_M"]), triton.cdiv(Nc, e["TILE_N"]))
        _TWR._gated_gemm_with_residual_backward_kernel.fn[grid](dn2d, xu, x_out, w["w_g"], w["w_p"], gdb, gg, gv, gdb, M, Nc, K,
                                                                STRIDE_B=N * N, N_COL=N, HAS_DROP_MASK=False,
                                                                NEEDS_INT64=(M * K >= 2**31 - 1) or (M * Nc >= 2**31 - 1), **_tile_kw(e))
    else:
        grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(Nc, meta["TILE_N"]))
        _TWR._gated_gemm_with_residual_backward_kernel[grid](dn2d, xu, x_out, w["w_g"], w["w_p"], gdb, gg, gv, gdb, M, Nc, K,
                                                         STRIDE_B=N * N, N_COL=N, HAS_DROP_MASK=False,
                                                         NEEDS_INT64=(M * K >= 2**31 - 1) or (M * Nc >= 2**31 - 1))
    return torch.mm(gg, w["w_g"]), torch.mm(gv, w["w_p"])


class _TriMulResidual(torch.autograd.Function):
    """new_pair = pair + TriMul(pair, mask) (frozen weights: backward returns d pair only)."""

    @staticmethod
    def forward(ctx, pair: Tensor, mask: Tensor | None, w: dict):
        B, N, N2, Cd = pair.shape
        assert N == N2 and Cd % 64 == 0, pair.shape
        Np = _padded(N); M = B * N * N; M_pad = B * Np * Np
        z = pair.contiguous(); z2d = z.view(M, Cd)
        m_pad = None
        if mask is not None:
            m_pad = torch.zeros((B, Np, Np), device=z.device, dtype=torch.float32)
            m_pad[:, :N, :N] = mask.reshape(B, N, N)
            m_pad = m_pad.view(-1)
        xp, xu = _ln_in(z2d, w, N, Np, M, M_pad, Cd)                          # x_in: padded rows (0 at pads) and unpadded rows
        ab = _gate_proj(xp, w, m_pad, M_pad, Cd).view(2 * Cd, B, Np, Np)     # a = ab[:C], b = ab[C:], channel-major, 0 at pads
        del xp
        a = ab[:Cd].view(Cd * B, Np, Np); b = ab[Cd:].view(Cd * B, Np, Np)
        if w["flow"] == "outgoing":            # o[c,i,j] = sum_k a[c,i,k] b[c,j,k]
            o = torch.bmm(a, b.transpose(1, 2))
        else:                                  # o[c,i,j] = sum_k a[c,k,i] b[c,k,j]
            o = torch.bmm(a.transpose(1, 2), b)
        x_out = _ln_out(o, w, N, Np, M, M_pad, Cd)
        new = _out_gate_residual(xu, x_out, w, z2d, N)
        ctx.save_for_backward(z, ab, o, *((m_pad,) if m_pad is not None else ()))
        ctx.has_mask = m_pad is not None; ctx.w = w; ctx.dims = (B, N, Np, M, M_pad, Cd)
        return new.view(B, N, N, Cd)

    @staticmethod
    def backward(ctx, dnew: Tensor):
        if ctx.has_mask:
            z, ab, o, m_pad = ctx.saved_tensors
        else:
            z, ab, o = ctx.saved_tensors; m_pad = None
        w = ctx.w; B, N, Np, M, M_pad, Cd = ctx.dims
        dn = dnew.contiguous().view(M, Cd)
        if dn.dtype != torch.bfloat16:
            dn = dn.to(torch.bfloat16)
        z2d = z.view(M, Cd)
        xp, xu = _ln_in(z2d, w, N, Np, M, M_pad, Cd)                          # recomputed (one LN pass) rather than saved
        x_out = _ln_out(o, w, N, Np, M, M_pad, Cd)
        dxu, dx_out = _out_gate_residual_bwd(dn, xu, x_out, w, N)             # d x_in (out-gate path), d x_out; unpadded rows
        del xu, x_out
        d_o = torch.empty_like(o)
        _k_ln_out_bwd[_grid_rows(M_pad)](o, w["g_out"], dx_out, d_o, N, Np, M_pad, C._EPS, CDIM=Cd, TILE_M=_LN_TILE_M, num_warps=_LN_WARPS)
        del dx_out
        a = ab[:Cd].view(Cd * B, Np, Np); b = ab[Cd:].view(Cd * B, Np, Np)
        dab = torch.empty_like(ab)
        da = dab[:Cd].view(Cd * B, Np, Np); db = dab[Cd:].view(Cd * B, Np, Np)
        if w["flow"] == "outgoing":            # O = A B^T :  dA = dO B ; dB = dO^T A
            torch.bmm(d_o, b, out=da); torch.bmm(d_o.transpose(1, 2), a, out=db)
        else:                                  # O = A^T B :  dA = B dO^T ; dB = A dO
            torch.bmm(b, d_o.transpose(1, 2), out=da); torch.bmm(a, d_o, out=db)
        del d_o
        dxp = _gate_proj_bwd(xp, w, m_pad, dab.view(2 * Cd, M_pad), M_pad, Cd)   # d x_in ([value|gate] path), padded rows
        del dab, xp
        dz = torch.empty_like(z2d)
        _k_ln_in_bwd[_grid_rows(M_pad)](z2d, w["g_in"], dxp, dxu, dn, dz, N, Np, M_pad, C._EPS, CDIM=Cd, TILE_M=_LN_TILE_M, num_warps=_LN_WARPS)
        return dz.view(B, N, N, Cd), None, None


def trimul_residual(pair: Tensor, mask: Tensor | None, engine: "C.TriangleMultiplicativeBlock", cache: dict) -> Tensor:
    """pair + TriMul(pair, mask) through the fused path (bf16 CUDA pair; frozen weights)."""
    assert pair.is_cuda and pair.dtype == torch.bfloat16, (pair.device, pair.dtype)
    if mask is not None and mask.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        mask = mask.to(torch.float32)
    return _TriMulResidual.apply(pair, mask, _weights(engine, cache))


def pair_trimul_residual(block: "C.PairUpdateBlock", pair: Tensor, mask: Tensor | None, variant: str = "fused", cache: dict | None = None) -> Tensor:
    """The block's two residual updates (outgoing then incoming) through the fused path — the bench entry point."""
    if variant not in VARIANTS:
        raise ValueError(f"ef2_trimul: variant {variant!r} unknown; variants are {VARIANTS}")
    if variant == "cueq_tiles":
        raise ValueError("ef2_trimul.pair_trimul_residual: 'cueq_tiles' is not a replacement path — it re-tiles the block's own "
                         "cuEquivariance kernels; run the block's modules with enable(model, variant='cueq_tiles') in effect")
    cache = block.__dict__.setdefault("_ef2_trimul_cache", {}) if cache is None else cache
    pair = trimul_residual(pair, mask, block.tri_mul_out._engine, cache)
    pair = trimul_residual(pair, mask, block.tri_mul_in._engine, cache)
    return pair


# =============================================================================================== exact: cuEquivariance tile table
# The stock cuEquivariance tri-mul (set_kernel_backend('cuequivariance')) runs four Triton GEMM kernels per call (gated dual
# GEMM fwd / dual-x fwd / the two backward "pregemm" partial kernels) whose tile shape comes from cuequivariance_ops' tuning
# cache; the cache shipped for sm_90 has no entry for d_pair-256 bf16 shapes, so every call runs the library default
# (TILE_M 64, TILE_N 32, TILE_K 32, 4 warps, 4 stages).  The kernels' per-element arithmetic does not depend on the tile
# shape ON THIS PIN (torch 2.11 / triton 3.6 / cuequivariance 0.10: Triton lowers `acc += tl.dot(a, b)` to MMA-with-accumulator, so
# the fp32 accumulation over K runs in the same 16-wide steps in the same order for any TILE_M / TILE_N / warps / stages), so a
# better tile is an EXACT lever, bitwise per pin: outputs and input-gradients tensor-equal (k/test_ef2_trimul.py::
# test_cueq_tiles_bitwise, 257..700 tokens; the mask-gradient partials, which nothing consumes, are the only tensors that change).
# A new pin re-proves it with that test.  enable(variant="cueq_tiles") writes entries for every row-count bucket of this
# model's shapes into the library's in-process cache (nothing on disk); disable() restores it.
_CUEQ_FN_FWD = "fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper"
_CUEQ_FN_BWD = "fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel_wrapper"
# TILE_K stays the library default's 32: the K loop (acc += dot over 32-wide K tiles) is then step-for-step the default's; only the
# M/N tiling, warps and pipelining change.  (TILE_K=64 measured tensor-equal too and ~3 % faster; not worth the extra premise.)  Only
# row counts >= 65536 (B*N*N, i.e. N >= 256 tokens at batch 1) get an entry: below that the step is launch-bound and the default tile
# measured within noise, so smaller shapes keep running the library default untouched.
# One entry per compute capability — the tile that measured fastest AND tensor-equal (out + d pair) on that card, re-proved there by
# k/test_ef2_trimul.py::test_cueq_tiles_bitwise: sm_90 (H100 80GB) 128x128, 8 warps, 4 stages: -27..-28 % vs the library default at
# 450 / 700 tokens; sm_80 (A100 80GB) 64x128, 4 warps, 4 stages: -19 / -21 % at 431 / 700 tokens (5 stages measures the same; the
# sm_90 tile gives -9 / -11 % there and 128x128 with 4 warps spills to 3x slower).  A capability with no entry keeps the library
# default BY NAME (entries=0 reason=cc_untuned on the lever's line) — never silent, never a refusal.
_CUEQ_TILES = dict(TILE_M=128, TILE_N=128, TILE_K=32, num_stages=4, num_warps=8)             # sm_90
_CUEQ_TILES_SM80 = dict(TILE_M=64, TILE_N=128, TILE_K=32, num_stages=4, num_warps=4)         # sm_80
_CUEQ_TILES_BY_CC = {(9, 0): _CUEQ_TILES, (8, 0): _CUEQ_TILES_SM80}
_CUEQ_ROWS_MIN = 65536
CC_UNTUNED = "cc_untuned"


def _device_cc() -> tuple:
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)


def _cueq_tiles_for(cc: tuple | None = None):
    """(tile entry, None) for compute capability `cc` (default: the current device's), or (None, 'cc_untuned') when this kit carries
    no entry for it — the caller then leaves the library's table untouched and names the reason."""
    cc = _device_cc() if cc is None else tuple(cc)
    cfg = _CUEQ_TILES_BY_CC.get(cc)
    return (cfg, None) if cfg is not None else (None, CC_UNTUNED)


def _cueq_tile_entries(d_pair: int, cfg: dict | None = None) -> dict:
    from cuequivariance_ops_torch.gated_gemm_torch import (forward_kernel_config_to_key as _fk, backward_kernel_config_to_key as _bk, Precision as _P)
    if cfg is None:
        cfg = _cueq_tiles_for()[0]
        if cfg is None:
            return {}
    ents = {}
    for M in range(_CUEQ_ROWS_MIN, 8192 * 32 + 512, 512):   # one representative per row-count bucket of the library's key function (buckets are >= 512 wide here)
        for n_out, two in ((2 * d_pair, False), (d_pair, True)):         # (dual GEMM: N=2C, one input) / (dual-x: N=C, two inputs)
            ents[(_CUEQ_FN_FWD, _fk(M, n_out, d_pair, torch.bfloat16, two, _P.DEFAULT.value))] = cfg
            ents[(_CUEQ_FN_BWD, _bk(M, n_out, d_pair, torch.bfloat16, two))] = cfg
    return ents


def _cueq_tiles_install(d_pair: int, cfg: dict | None = None) -> dict:
    from cuequivariance_ops.triton.cache_manager import get_cache_manager
    cm = get_cache_manager()
    saved = {}
    for (fn, key), cfg in _cueq_tile_entries(d_pair, cfg).items():
        cm.get(fn, key)                                      # loads the library's table for fn once
        table = cm.gpu_cache[fn]
        saved[(fn, key)] = table.get(key)
        table[key] = {"config": dict(cfg), "time": 0.0}
    return saved


def _cueq_tiles_restore(saved: dict) -> None:
    from cuequivariance_ops.triton.cache_manager import get_cache_manager
    cm = get_cache_manager()
    for (fn, key), prev in saved.items():
        table = cm.gpu_cache.get(fn)
        if table is None:
            continue
        if prev is None:
            table.pop(key, None)
        else:
            table[key] = prev


# =============================================================================================== patching
@dataclass
class Handle:
    variant: str = "fused"
    only_under_grad: bool = True
    cache: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"served": 0, "fallback": 0})
    cueq_saved: dict = field(default_factory=dict)      # cueq_tiles: the library cache entries replaced (restored by disable)
    cueq_reason: str = ""                                 # cueq_tiles: why no entry was written (cc_untuned:sm_NN), else ""
    gemm: str = ""                                        # fused: the gated-GEMM launch table in effect (sm_90) or vendored:cc_untuned:sm_NN / vendored:off


@torch._dynamo.disable
def _tmu_forward(self: "C.TriangleMultiplicativeUpdate", z: Tensor, mask: Tensor | None = None) -> Tensor:
    """Patched TriangleMultiplicativeUpdate.forward: returns the RESIDUAL-INCLUSIVE update pair + TriMul(pair) on the fused
    path (the owning block's row_drop is made the identity on that path), else the module's own forward.  Opaque to
    torch.compile (upstream compiles PairUpdateBlock.forward): the compiled block graph-breaks around this call, so the
    autograd Function, its Triton launches and the residual flag below always run eagerly — identically in a checkpoint's
    first forward and its recompute (a traced-then-retraced Function saved different tensor sets and tripped
    torch.utils.checkpoint's recompute check)."""
    h: Handle = self._ef2_trimul_handle
    use = (z.is_cuda and z.dtype == torch.bfloat16 and (torch.is_grad_enabled() or not h.only_under_grad)
           and not (self.training and getattr(self._ef2_trimul_block.row_drop, "_r", 0.0) > 0.0))
    if not use:
        h.stats["fallback"] += 1
        return self._ef2_trimul_orig_forward(z, mask=mask)
    _patch_row_drop(self._ef2_trimul_block.row_drop)        # PairUpdateBlock.set_kernel_backend() rebuilds row_drop: re-patch
    h.stats["served"] += 1
    out = trimul_residual(z, mask, self._engine, h.cache)
    out._ef2_trimul_residual_included = True
    return out


@torch._dynamo.disable
def _row_drop_forward(self: "C.DropoutResidual", residual: Tensor, delta: Tensor) -> Tensor:
    if getattr(delta, "_ef2_trimul_residual_included", False):
        return delta
    return self._ef2_trimul_orig_forward(residual, delta)


def _patch_row_drop(rd: "C.DropoutResidual") -> None:
    if not hasattr(rd, "_ef2_trimul_orig_forward"):
        rd._ef2_trimul_orig_forward = rd.forward
        rd.forward = types.MethodType(_row_drop_forward, rd)


def enable(model: torch.nn.Module, variant: str = "fused", only_under_grad: bool = True) -> Handle:
    """variant="fused" (FAST class): route every PairUpdateBlock's tri_mul_out / tri_mul_in through the fused residual update
    (instance-level, reversible; compose with ef2_autograd_kernels.enable(model, trimul=None, ...) — a block whose agk config
    replaces the triangle multiplication itself is refused by name).  variant="cueq_tiles" (EXACT class): keep the stock cuEquivariance tri-mul and give its Triton GEMM kernels
    the tile entry of this card's compute capability for this model's row counts (bitwise-identical outputs and input-gradients); a
    capability without an entry keeps the library default, ``h.cueq_reason`` naming why (entries=0)."""
    if variant not in VARIANTS:
        raise ValueError(f"ef2_trimul: variant {variant!r} unknown; variants are {VARIANTS}")
    h = Handle(variant=variant, only_under_grad=only_under_grad)
    blocks = [m for m in model.modules() if isinstance(m, C.PairUpdateBlock)]
    if isinstance(model, C.PairUpdateBlock) and model not in blocks:
        blocks = [model] + blocks
    if not blocks:
        raise RuntimeError("ef2_trimul.enable: no PairUpdateBlock in the model")
    if variant == "cueq_tiles":
        if not C.CUE_AVAILABLE:
            raise RuntimeError("ef2_trimul.enable(variant='cueq_tiles'): cuequivariance_torch is not importable")
        if not any(getattr(b.tri_mul_out, "_kernel_backend", None) == C.BACKEND_CUEQ or getattr(b.tri_mul_out._engine, "_use_kernels", False) for b in blocks):
            raise RuntimeError("ef2_trimul.enable(variant='cueq_tiles'): no PairUpdateBlock runs the cuequivariance backend "
                               "(model.set_kernel_backend('cuequivariance') first) — the tile table would be inert")
        d_pair = blocks[0].tri_mul_out._engine.norm_start.weight.shape[0]
        cc = _device_cc()
        cfg, why = _cueq_tiles_for(cc)
        if cfg is None:                                   # no entry for this card: the library default runs, named on the lever's line
            h.cueq_reason = f"{why}:sm_{cc[0]}{cc[1]}"
        else:
            h.cueq_saved = _cueq_tiles_install(d_pair, cfg)
        model.__dict__["_ef2_trimul_handle"] = h
        return h
    h.gemm = _gg_table()[1]                              # the gated-GEMM launch table of this card (sm_90), else the vendored launches by name
    for blk in blocks:
        agk_cfg = getattr(blk, "_agk_cfg", None)
        if agk_cfg is not None and getattr(agk_cfg, "trimul", None) is not None:
            raise RuntimeError(f"ef2_trimul.enable: block already runs ef2_autograd_kernels trimul={agk_cfg.trimul!r}; enable it with trimul=None")
        for tmu in (blk.tri_mul_out, blk.tri_mul_in):
            if not hasattr(tmu, "_ef2_trimul_orig_forward"):
                tmu._ef2_trimul_orig_forward = tmu.forward
            tmu._ef2_trimul_handle = h
            tmu.__dict__["_ef2_trimul_block"] = blk            # plain attribute, not a registered submodule (no cycle in modules())
            tmu.forward = types.MethodType(_tmu_forward, tmu)
        _patch_row_drop(blk.row_drop)
        blk.__dict__["_ef2_trimul_handle"] = h
    model.__dict__["_ef2_trimul_handle"] = h
    return h


def disable(model: torch.nn.Module) -> None:
    h = model.__dict__.get("_ef2_trimul_handle")
    if h is not None and h.cueq_saved:
        _cueq_tiles_restore(h.cueq_saved); h.cueq_saved = {}
    mods = list(model.modules())
    for m in mods:
        if isinstance(m, C.PairUpdateBlock):
            for sub in (m.tri_mul_out, m.tri_mul_in, m.row_drop):
                if hasattr(sub, "_ef2_trimul_orig_forward"):
                    sub.forward = sub._ef2_trimul_orig_forward
                    del sub._ef2_trimul_orig_forward
                for k in ("_ef2_trimul_handle", "_ef2_trimul_block"):
                    sub.__dict__.pop(k, None)
            m.__dict__.pop("_ef2_trimul_handle", None)
            m.__dict__.pop("_ef2_trimul_cache", None)
    model.__dict__.pop("_ef2_trimul_handle", None)


def describe(h: Handle | None) -> str:
    if h is None:
        return "stock"
    if h.variant == "cueq_tiles":
        return f"ef2_trimul variant=cueq_tiles entries={len(h.cueq_saved)}" + (f" reason={h.cueq_reason}" if h.cueq_reason else "")
    return f"ef2_trimul variant={h.variant} served={h.stats['served']} fallback={h.stats['fallback']} gemm={h.gemm or _gg_table()[1]}"
