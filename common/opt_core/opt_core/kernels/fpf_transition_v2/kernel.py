"""fpf_transition_v2 — the AF3-family SwiGLU transition (LayerNorm -> W_a|W_b -> silu(a)*b -> W_out) with the row mask and the
residual add folded in, ONE Triton kernel per call (the pair / template / MSA transitions of an AF3 trunk):

    out = [z +] [mask[:, None] *] ( W_o @ ( silu(W_a @ LN(z)) * (W_b @ LN(z)) ) )        per row of a 2-D view z = x2d [M, c] (bf16, CUDA)

Numerics (the stock bf16 trunk's rounding points): LayerNorm statistics in fp32 on the bf16 row, affine weight/bias applied as their
bf16-rounded values, LN output rounded to bf16; the two input GEMMs bf16 x bf16 with fp32 accumulation over K = c; their results a, b rounded
to bf16 (numerics='fast' skips this one rounding); h = bf16( (a * sigmoid(a)) * b ) in fp32 (the Liger silu*mul kernel's
expression; sigmoid = 1/(1+exp(-a)) in fp32 — instruction forms per numerics mode, see NUMERICS); the output GEMM accumulates h_chunk @ W_o_chunk^T over ascending
hidden chunks in fp32, rounded to bf16; u = update * mask (rounded to bf16 for 16-bit float masks like `bf16 * bf16`, kept fp32 for fp32/fp64
masks like type promotion does); out = bf16(fp32(z) + u) when residual=True (one rounding, like `z += u`).
Not bit-identical to the stock chain: LayerNorm's reduction tree and the output GEMM's summation order differ from ATen / cuBLAS (the result
differs from the stock chain by one bf16 ulp in <= 2e-3 of the elements, at the same distance from a float64 evaluation as the stock chain).

Structure: one program = BM consecutive rows.  Prologue: load the [BM, c] tile and the [BM] mask, LayerNorm in registers.  Loop over the
hidden dimension in chunks of BH (weights double-buffered by cp.async, streamed from L2): a|b = y @ Wa|b^T[:, chunk] (wgmma, fp32 acc) ->
h = silu(a) * b in registers -> acc += h @ W_o^T[chunk, :] (wgmma, fp32 acc in registers).  Epilogue: round, mask, re-read the residual rows
(L2-hot), CTA barrier, add, store.  No atomics, no split-K, fixed chunk order: run-to-run deterministic and independent of M / strides / grid.
Every row offset is int64 (M * row_stride may exceed 2^31).  M need not be a multiple of BM (masked rows); M = 0 launches nothing.
In place (out2d is x2d) is safe: a program re-reads its rows before any of its warps stores (barrier), and programs own disjoint rows.

API (pure functions; nothing is patched):
    pack_weights(ln_weight, ln_bias, w_a, w_b, w_o, eps, device) -> Packed      once per module (bf16 copies, layout in Packed's docstring)
    fused_transition(x2d, packed, *, mask1d=None, residual=False, out2d=None, binary_mask=None, numerics='exact') -> out2d
    served(c, hidden, device) -> (bool, reason)                                 the declared domain
    describe() -> dict
Anything outside the served domain raises Unsupported(reason) — there is no fallback path in this file.
"""
from __future__ import annotations

import dataclasses

import torch
import triton
import triton.language as tl

__version__ = "2.2.0"


class Unsupported(Exception):
    """Raised BY NAME for any input outside the served domain (reason: short machine-readable string)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------------------------------------------------------------------------
# (c, hidden) -> the module's own launch configuration for sm_90 (H100 / H200), torch 2.10 / triton 3.6; opt_core.attn.pair_fused launches
# with the row of its cell table instead (pair_fused_cells.json, variant 'v2'), these are the defaults of fused_transition() called directly.
# (a|b split in registers), 0 = two dots.
CONFIGS = {
    (128, 512):  {"BM": 64, "BH": 64, "num_warps": 4, "num_stages": 2, "IL": 1},     # pairformer / MSA-module pair / confidence: 112 KB smem, 2 CTAs/SM
    (64, 128):   {"BM": 64, "BH": 32, "num_warps": 4, "num_stages": 3, "IL": 0},     # template pair stack (n=2)
    (64, 256):   {"BM": 64, "BH": 64, "num_warps": 4, "num_stages": 2, "IL": 0},     # MSA transition (n=4)
    (256, 1024): {"BM": 128, "BH": 32, "num_warps": 8, "num_stages": 3, "IL": 0},    # other engines' pair width (served, not tuned)
    (384, 1536): {"BM": 16, "BH": 64, "num_warps": 4, "num_stages": 2, "IL": 0},     # single transition (served, not tuned)
}
SERVED_CC = {(9, 0)}                                  # measured device classes (H100 / H200); other capabilities are refused by name
MASK_DTYPES = {torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64}
BINARY_MASK_DTYPES = {torch.bool}                     # {0,1} by construction -> folded mask path without a caller promise
ROUND16_MASK_DTYPES = {torch.float16, torch.bfloat16}  # `bf16_update * mask` rounds to 16 bits for these; wider masks promote to fp32
LAST_LAUNCH = {"handle": None}                        # compiled-kernel handle of the most recent launch (diagnostics: n_regs / n_spills / shared)
# numerics modes (fused_transition(numerics=...)):
#   exact  — the stock chain's rounding points: a, b (the input GEMMs' fp32 accumulators) rounded to bf16 (round-to-nearest-even via the
#            Veltkamp split _round_bf16, FMA pipes), sigmoid(a) = 1/(1+exp(-a)) in fp32 via _sigmoid (ex2.approx.ftz + rcp.approx.ftz: the
#            MUFU ops tl.sigmoid uses, flush-to-zero forms).  Compared against the stock instruction forms over all
#            2^32 fp32 inputs the two are identical except (i) sigmoid inputs in (-88.72, -87.33), where tl.sigmoid returns a subnormal (< 1.18e-38)
#            and _sigmoid returns 0, and (ii) subnormal |a|, |b| < 1.18e-38, which _round_bf16 leaves unrounded; max relative error of the
#            sigmoid vs float64 is 3.2e-6 for both forms (tanh.approx-class shortcuts, 4.9e-4, are NOT used in any mode).
#   strict — cvt.rn.bf16x2 rounding and tl.sigmoid itself (ex2.approx.f32 + div.full.f32, the Liger kernel's lowering): removes (i) and (ii);
#            ~10 % slower (the conversion unit and the non-ftz fix-up instructions share the activation stage's bottleneck pipe).
#   fast   — a, b are NOT rounded to bf16 before the activation (one rounding fewer than stock: slightly more accurate vs float64, differs
#            from stock in ~20 % of output elements by an ulp); sigmoid via _sigmoid.  ~5 % faster than exact.
NUMERICS = {"exact": 0, "strict": 1, "fast": 2}


# --------------------------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _round_bf16(x):
    """fp32 -> nearest-even bf16 value (kept in fp32) with 3 FMA-pipe ops instead of two F2F conversions (the conversion unit is the same
    16-op/clk/SM pipe as MUFU and is this kernel's bottleneck): Veltkamp split with 2^16 + 1, t = RN(65537 x), hi = RN(t - RN(t - x)) —
    equal to RNE-to-8-significant-bits(x), ties included, for |x| < 2^100.  Explicit .rn keeps ptxas from contracting into an FMA."""
    return tl.inline_asm_elementwise("{ .reg .f32 t, u; mul.rn.f32 t, $1, 0f47800080; sub.rn.f32 u, t, $1; sub.rn.f32 $0, t, u; }",
                                     "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sigmoid(x):
    """1 / (1 + 2^(-x*log2e)): the stock expression through ex2.approx.ftz + rcp.approx.ftz (2 MUFU ops; the non-ftz forms tl.sigmoid lowers
    to cost extra ALU ops and give the same values here: subnormal cases only arise where 1 + e == 1 or e == inf either way)."""
    return tl.inline_asm_elementwise("{ .reg .f32 t; mul.ftz.f32 t, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; add.ftz.f32 t, t, 0f3F800000; rcp.approx.ftz.f32 $0, t; }",
                                     "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu_mul(a32, b32, NUM: tl.constexpr):
    """h = bf16( (a * sigmoid(a)) * b ) in fp32 from the two fp32 GEMM accumulators.  NUM (see NUMERICS): 0 exact — a, b rounded to bf16
    (stock's GEMM outputs are bf16 tensors) by _round_bf16, sigmoid by _sigmoid; 1 strict — the stock instruction forms: cvt-based rounding
    and tl.sigmoid (bit-identical to 0 for all normal-range values; slower); 2 fast — a, b not rounded (one rounding
    fewer than stock), sigmoid by _sigmoid."""
    if NUM == 2:
        af = a32
        bf = b32
    elif NUM == 1:
        af = a32.to(tl.bfloat16).to(tl.float32)
        bf = b32.to(tl.bfloat16).to(tl.float32)
    else:
        af = _round_bf16(a32)
        bf = _round_bf16(b32)
    if NUM == 1:
        sg = tl.sigmoid(af)
    else:
        sg = _sigmoid(af)
    return ((af * sg) * bf).to(tl.bfloat16)


@triton.jit
def _pair_transition_kernel(X, OUT, MASK, LNW, LNB, WA, WB, WO, M, s_x, s_o, s_m, eps,
                            C: tl.constexpr, CP: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr,
                            MASK_MODE: tl.constexpr, MASK_ROUND: tl.constexpr, HAS_RES: tl.constexpr, NUM: tl.constexpr,
                            NUM_STAGES: tl.constexpr, DBG: tl.constexpr):
    # MASK_MODE 0: no mask; 1: general multiply in the epilogue; 2: {0,1} mask folded into the GEMM input rows.
    # DBG (diagnostics only, results invalid): 1 every chunk reads hidden chunk 0 (no L2 weight streaming), 2 no activation math,
    # 4 no LayerNorm, 8 no x load, 16 predicated-off store.
    PADK: tl.constexpr = CP != C
    blk = tl.program_id(0)
    rm = blk * BM + tl.arange(0, BM)
    mrow = rm < M
    rm64 = rm.to(tl.int64)
    rc = tl.arange(0, CP)
    mcol = rc < C
    if PADK:
        xmask = mrow[:, None] & mcol[None, :]
    else:
        xmask = mrow[:, None]
    # ---- prologue: the row tile (LN input AND residual) and the row mask, fetched together so the epilogue has no dependent global load ----
    if DBG & 8:
        x = (rm64[:, None] + rc[None, :]).to(tl.bfloat16)
    else:
        x = tl.load(X + rm64[:, None] * s_x + rc[None, :], mask=xmask, other=0.0)                 # [BM, CP] bf16
    if MASK_MODE != 0:
        mv = tl.load(MASK + rm64 * s_m, mask=mrow, other=0).to(tl.float32)                         # [BM]
    # ---- LayerNorm: fp32 statistics, two-pass on the register tile, rsqrt(var + eps) ----
    if DBG & 4:
        y16 = x
    else:
        xf = x.to(tl.float32)
        mean = tl.sum(xf, axis=1) / C
        d = xf - mean[:, None]
        if PADK:
            d = tl.where(mcol[None, :], d, 0.0)
        var = tl.sum(d * d, axis=1) / C
        rstd = tl.rsqrt(var + eps)
        w = tl.load(LNW + rc).to(tl.float32)                                                        # packs are zero-padded to CP
        bb = tl.load(LNB + rc).to(tl.float32)
        y16 = ((d * rstd[:, None]) * w[None, :] + bb[None, :]).to(tl.bfloat16)                     # [BM, CP] bf16 (padded columns exactly 0)
    if MASK_MODE == 2:
        # {0,1} mask folded into the GEMM input: a zero row of y gives a = b = 0, h = silu(0) * 0 = 0, update = 0 — the same bits as
        # bf16(bf16(update) * 0); rows with mask 1 are untouched.  Exact ONLY for binary masks (launch() routes other masks to MASK_MODE 1).
        y16 = tl.where(mv[:, None] != 0.0, y16, 0.0).to(tl.bfloat16)
    # ---- hidden-chunk loop: everything stays on chip ----
    acc = tl.zeros((BM, CP), dtype=tl.float32)
    rh = tl.arange(0, BH)
    for jj in tl.range(0, NH, BH, num_stages=NUM_STAGES):
        if DBG & 1:
            j = jj * 0
        else:
            j = jj
        if IL == 1:
            r2 = 2 * j + tl.arange(0, 2 * BH)
            w2 = tl.load(WA + r2[None, :] * CP + rc[:, None])                                       # [CP, 2BH]: columns a_j, b_j, a_j+1, b_j+1, ...
            ab = tl.dot(y16, w2)                                                                    # [BM, 2BH] fp32
            a32, b32 = tl.split(tl.reshape(ab, (BM, BH, 2)))
        else:
            wa = tl.load(WA + (j + rh)[None, :] * CP + rc[:, None])                                 # [CP, BH] = Wa[j:j+BH, :]^T (K contiguous)
            wb = tl.load(WB + (j + rh)[None, :] * CP + rc[:, None])
            a32 = tl.dot(y16, wa)                                                                   # fp32 accumulation over K = C in one chain
            b32 = tl.dot(y16, wb)
        if DBG & 2:
            h16 = (a32 + b32).to(tl.bfloat16)
        else:
            h16 = _silu_mul(a32, b32, NUM)                                                          # [BM, BH] bf16
        wo = tl.load(WO + rc[None, :] * NH + (j + rh)[:, None])                                     # [BH, CP] = Wo[:, j:j+BH]^T (K contiguous)
        acc = tl.dot(h16, wo, acc)                                                                  # ascending hidden chunks, fp32
    # ---- epilogue: bf16(update) -> * mask -> + residual -> store ----
    o = acc.to(tl.bfloat16).to(tl.float32)
    if MASK_MODE == 1:
        o = o * mv[:, None]
        if MASK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)                                                    # bf16 * 16-bit mask rounds to bf16
    if PADK:
        omask = mrow[:, None] & mcol[None, :]
    else:
        omask = mrow[:, None]
    if HAS_RES:
        # Re-read the residual rows (L2-hot: this program read them in the prologue) rather than hold the tile in registers through the loop.
        # cache_modifier=".cg" also keeps the compiler from merging this load with the prologue's (that merge pins the fp32 tile in registers).
        xr = tl.load(X + rm64[:, None] * s_x + rc[None, :], mask=xmask, other=0.0, cache_modifier=".cg")
        tl.debug_barrier()                             # in-place safety (OUT may be X): all warps of this program re-read before any stores
        o = xr.to(tl.float32) + o
    if DBG & 16:
        omask = omask & (blk < 0)
    tl.store(OUT + rm64[:, None] * s_o + rc[None, :], o.to(tl.bfloat16), mask=omask)


# --------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Packed:
    """bf16 copies of one transition module's parameters, built once (contiguous, on `device`; CP = c rounded up to a power of two, zero pad):
        lnw, lnb : [CP]             LayerNorm weight / bias (ones / zeros when the module has none)
        wa, wb   : [hidden, CP]     = linear_a.weight, linear_b.weight (the stock [hidden, c] layouts, columns padded)
        wab      : [2*hidden, CP]   row 2i = wa[i], row 2i+1 = wb[i]        (read by IL=1 configs instead of wa, wb)
        wo       : [CP, hidden]     = linear_out.weight (the stock [c, hidden] layout, rows padded)
        eps : float ; c, cp, hidden : ints"""
    c: int
    cp: int
    hidden: int
    eps: float
    lnw: torch.Tensor
    lnb: torch.Tensor
    wa: torch.Tensor
    wb: torch.Tensor
    wab: torch.Tensor
    wo: torch.Tensor


def pack_weights(ln_weight, ln_bias, w_a, w_b, w_o, eps: float, device) -> Packed:
    hidden, c = w_a.shape
    if tuple(w_b.shape) != (hidden, c) or tuple(w_o.shape) != (c, hidden):
        raise Unsupported(f"weight-shapes-{tuple(w_a.shape)}-{tuple(w_b.shape)}-{tuple(w_o.shape)}")
    for name, t in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        if t is not None and tuple(t.shape) != (c,):
            raise Unsupported(f"{name}-shape-{tuple(t.shape)}-c{c}")
    cp = triton.next_power_of_2(c)
    dev = torch.device(device)
    with torch.no_grad():
        bf = lambda t: t.detach().to(device=dev, dtype=torch.bfloat16)
        lnw = torch.zeros(cp, dtype=torch.bfloat16, device=dev); lnb = torch.zeros(cp, dtype=torch.bfloat16, device=dev)
        lnw[:c] = bf(ln_weight) if ln_weight is not None else 1.0
        lnb[:c] = bf(ln_bias) if ln_bias is not None else 0.0
        wa = torch.zeros((hidden, cp), dtype=torch.bfloat16, device=dev); wa[:, :c] = bf(w_a)
        wb = torch.zeros((hidden, cp), dtype=torch.bfloat16, device=dev); wb[:, :c] = bf(w_b)
        wab = torch.stack([wa, wb], dim=1).reshape(2 * hidden, cp).contiguous()
        wo = torch.zeros((cp, hidden), dtype=torch.bfloat16, device=dev); wo[:c] = bf(w_o)
    return Packed(c=int(c), cp=int(cp), hidden=int(hidden), eps=float(eps), lnw=lnw, lnb=lnb, wa=wa, wb=wb, wab=wab, wo=wo)


def served(c: int, hidden: int, device) -> tuple:
    """The declared domain: (True, '') or (False, reason).  A pure function of (c, hidden, device class)."""
    if (int(c), int(hidden)) not in CONFIGS:
        return False, f"cell-{int(c)}x{int(hidden)}-not-configured"
    dev = torch.device(device)
    if dev.type != "cuda":
        return False, f"device-{dev.type}"
    cc = tuple(torch.cuda.get_device_capability(dev))
    if cc not in SERVED_CC:
        return False, f"arch-sm{cc[0]}{cc[1]}-not-measured"
    return True, ""


def _span(t: torch.Tensor):
    """[lo, hi) byte range a strided view can touch."""
    if t.numel() == 0:
        return (0, 0)
    lo = t.data_ptr(); hi = lo
    for size, stride in zip(t.shape, t.stride()):
        if stride >= 0:
            hi += (size - 1) * stride * t.element_size()
        else:
            lo += (size - 1) * stride * t.element_size()
    return (lo, hi + t.element_size())


def _check(x2d, packed: Packed, mask1d, out2d):
    if not (torch.is_tensor(x2d) and x2d.is_cuda):
        raise Unsupported("x-not-cuda")
    if x2d.dtype != torch.bfloat16:
        raise Unsupported(f"x-dtype-{x2d.dtype}")
    if x2d.dim() != 2 or x2d.shape[1] != packed.c:
        raise Unsupported(f"x-shape-{tuple(x2d.shape)}-c{packed.c}")
    M = x2d.shape[0]
    if M >= 2 ** 31 - 2 ** 12:
        raise Unsupported("rows", f"{M} rows: the row index is int32 in-kernel (row offsets are int64)")
    if x2d.stride(1) != 1 and packed.c > 1:
        raise Unsupported(f"x-col-stride-{x2d.stride(1)}")
    if M > 1 and x2d.stride(0) < packed.c:
        raise Unsupported(f"x-row-stride-{x2d.stride(0)}-overlapping-rows")
    ok, why = served(packed.c, packed.hidden, x2d.device)
    if not ok:
        raise Unsupported(why)
    if packed.wa.device != x2d.device:
        raise Unsupported(f"packed-on-{packed.wa.device}-x-on-{x2d.device}")
    if out2d is not None:
        if not (torch.is_tensor(out2d) and out2d.is_cuda and out2d.device == x2d.device):
            raise Unsupported("out-not-on-x-device")
        if out2d.dtype != torch.bfloat16 or tuple(out2d.shape) != tuple(x2d.shape):
            raise Unsupported(f"out-{out2d.dtype}-{tuple(out2d.shape)}")
        if out2d.stride(1) != 1 and packed.c > 1:
            raise Unsupported(f"out-col-stride-{out2d.stride(1)}")
        if M > 1 and out2d.stride(0) < packed.c:
            raise Unsupported(f"out-row-stride-{out2d.stride(0)}-overlapping-rows")
        xs, os_ = _span(x2d), _span(out2d)
        if M > 0 and xs[0] < os_[1] and os_[0] < xs[1]:                                          # storage overlap: only the exact alias is safe
            if not (out2d.data_ptr() == x2d.data_ptr() and (M <= 1 or out2d.stride(0) == x2d.stride(0))):
                raise Unsupported("out-partially-overlaps-x")
    if mask1d is not None:
        if not (torch.is_tensor(mask1d) and mask1d.device == x2d.device):
            raise Unsupported("mask-not-on-x-device")
        if mask1d.dim() != 1 or mask1d.shape[0] != M:
            raise Unsupported(f"mask-shape-{tuple(mask1d.shape)}-M{M}")
        if mask1d.dtype not in MASK_DTYPES:
            raise Unsupported(f"mask-dtype-{mask1d.dtype}")
        if out2d is not None and M > 0:
            ms, os_ = _span(mask1d), _span(out2d)
            if ms[0] < os_[1] and os_[0] < ms[1]:
                raise Unsupported("mask-overlaps-out")


def launch(x2d, packed: Packed, mask1d, residual: bool, out2d, cfg: dict, *, binary_mask=None, numerics: str = "exact", dbg: int = 0):
    """The launch with an explicit config (opt_core.attn.pair_fused calls this with its cell row; fused_transition = _check + CONFIGS + launch).
    cfg keys: BM, BH, num_warps, num_stages, IL (optional MASK_MODE forces the mask path, MAXNREG caps registers).  dbg: see the kernel."""
    M, C = x2d.shape
    if out2d is None:
        out2d = torch.empty((M, C), dtype=torch.bfloat16, device=x2d.device)
    if M == 0:
        return out2d
    BM, BH, NH, CP = int(cfg["BM"]), int(cfg["BH"]), packed.hidden, packed.cp
    if NH % BH != 0:
        raise Unsupported(f"hidden-{NH}-not-multiple-of-BH-{BH}")
    if numerics not in NUMERICS:
        raise Unsupported(f"numerics-{numerics}")
    if mask1d is None:
        mask_mode = 0
    elif binary_mask if binary_mask is not None else (mask1d.dtype in BINARY_MASK_DTYPES):
        mask_mode = 2
    else:
        mask_mode = 1
    if mask1d is not None and "MASK_MODE" in cfg:
        mask_mode = int(cfg["MASK_MODE"])
    mask_round = mask1d is not None and mask1d.dtype in ROUND16_MASK_DTYPES
    if mask1d is not None and mask1d.dtype == torch.bool:
        mask1d = mask1d.view(torch.uint8)
    mask_arg = mask1d if mask1d is not None else packed.lnw                                       # any valid pointer when unused
    s_m = mask1d.stride(0) if mask1d is not None else 0
    il = int(cfg.get("IL", 0))
    LAST_LAUNCH["handle"] = _pair_transition_kernel[(triton.cdiv(M, BM),)](
        x2d, out2d, mask_arg, packed.lnw, packed.lnb, packed.wab if il else packed.wa, packed.wb, packed.wo,
        M, x2d.stride(0), out2d.stride(0), s_m, packed.eps,
        C=C, CP=CP, NH=NH, BM=BM, BH=BH, IL=il, MASK_MODE=mask_mode, MASK_ROUND=bool(mask_round), HAS_RES=bool(residual), NUM=NUMERICS[numerics],
        NUM_STAGES=int(cfg["num_stages"]), DBG=int(dbg),
        num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"]), **({"maxnreg": int(cfg["MAXNREG"])} if cfg.get("MAXNREG") else {}))
    return out2d


def fused_transition(x2d, packed: Packed, *, mask1d=None, residual: bool = False, out2d=None, binary_mask=None, numerics: str = "exact"):
    """out2d = [x2d +] [mask1d[:, None] *] transition(x2d), one kernel.  x2d [M, c] bf16 CUDA with unit column stride and any row stride >= c;
    mask1d [M] (bool / integer / float dtype) or None; residual=True adds the input rows; out2d [M, c] bf16 view with unit column stride (may be
    x2d itself — in place is safe; a PARTIAL overlap with x2d is refused) or None (allocated contiguous).
    binary_mask: True = the caller promises every mask value is 0 or 1 (an AF3-family pair mask is) -> the mask is folded into the GEMM input
    rows (free; exact for {0,1}); False = general values, multiplied in the epilogue; None = True for bool masks, False for other dtypes.
    numerics: "exact" (default; stock's rounding points), "strict" (also stock's instruction forms for the bf16 rounding and the sigmoid —
    same values for all normal-range numbers, slower) or "fast" (a, b enter the activation unrounded: one rounding fewer than stock, faster);
    see NUMERICS.  Raises Unsupported(reason) for anything outside the served domain."""
    _check(x2d, packed, mask1d, out2d)
    return launch(x2d, packed, mask1d, residual, out2d, CONFIGS[(packed.c, packed.hidden)], binary_mask=binary_mask, numerics=numerics)


def describe() -> dict:
    return {"name": "fpf_transition_v2", "version": __version__, "triton": triton.__version__, "torch": torch.__version__,
            "cells": {f"{c}x{h}": dict(cfg) for (c, h), cfg in sorted(CONFIGS.items())}, "served_cc": sorted(SERVED_CC),
            "numerics": {"exact": "LN fp32 stats -> bf16; a,b = bf16(fp32-acc GEMM) [Veltkamp RNE]; h = bf16((a*sigmoid(a))*b) fp32 [ex2.approx.ftz+rcp.approx.ftz]; "
                                  "update = bf16(fp32-acc GEMM over ascending hidden chunks); u = update*mask (bf16-rounded for 16-bit masks); out = bf16(fp32(x) + u)",
                         "strict": "as exact with cvt.rn.bf16 rounding and tl.sigmoid (ex2.approx.f32 + div.full.f32, the stock Liger lowering)",
                         "fast": "as exact but a,b enter the activation as fp32 accumulators (not rounded to bf16)"},
            "mask_paths": {"binary": "folded into GEMM input rows (bool masks, or binary_mask=True)", "general": "epilogue multiply"}}
