"""ef2_msa_v2 — MSA-module levers for the ESMFold2 FULL model (MSAEncoder / MSAEncoderBlock of Biohub transformers@ef32577f,
modeling_esmfold2.py:1104-1205; OuterProductMean / MSAPairWeightedAveraging in modeling_esmfold2_common.py:2727-2824).

Where the MSA module sits: ``ESMFold2Model._run_one_loop`` calls ``self.msa_encoder(x_pair=z_init, x_inputs, <fresh row subsample of the MSA>)``
once per loop iteration (num_loops + 1 = 21 per fold at the library default): 21 x 4 MSAEncoderBlocks = 84 OuterProductMean, 63
MSAPairWeightedAveraging, 63 msa_transition, 168 TriMul and 84 PairTransition evaluations per fold, on m [1, L, M<=msa_max_depth, 128] bf16 and
z [1, L, L, 256] bf16 under torch.autocast(bf16).  The encoder never reads the evolving trunk state z (its pair input is z_init every
iteration); it is RNG-free (the row subsample and the LM dropout draw happen in _run_one_loop before it is called).

Levers (each its own flag of ``enable()``; every one is a per-INSTANCE patch, nothing at class or module level except `mh`'s wrapper of the
module-global ``maybe_subsample_msa`` name; stock source is never edited):

  m15  mtr_fused    msa_transition (PairTransition(128, x4): LN -> w12 -> silu(x1)*x2 -> w3) + the block's residual add in ONE Triton kernel per
                    row block of m: fp32 LN statistics in-kernel, bf16 x_hat, two K=128 dots (fp32 acc), the stock rounding points
                    (x12 -> bf16, silu -> bf16, product -> bf16, w3 out -> bf16, m + t -> bf16), W12^T / W3^T streamed from L2.  TIER-2
                    (LN statistics order, exp inside silu, K=128 / K=512 fp32 accumulation order vs cuBLAS).  Removes the x12 [L*M, 1024] /
                    hidden [L*M, 512] HBM round trips and the separate residual pass.
  m16  pwa_fused    MSAPairWeightedAveraging + the block's residual add as: (k1) ONE producer kernel LN(m) -> [Wv | Wgate] (K=128) -> sigmoid(gate),
                    writing v head-major [H, L*M, dh] and gate [L*M, H*dh] (bf16, stock rounding points); (k2) ONE bias kernel LN(z) -> 8 head
                    logits (+ key mask -1e5) kept fp32, softmax over j in fp32 -> bf16 [H, L, L]; (k3) attn . v as ONE cuBLAS bmm
                    [H, L, L] x [H, L, M*dh] -> bf16 (no permute copies: the producer wrote the layout bmm reads); (k4) ONE epilogue kernel
                    bf16(o * gate) -> Wout (K=H*dh) -> bf16 -> m + . -> bf16 written in place over m.  TIER-2 (LN statistics order; K=128 /
                    K=256 accumulation order vs cuBLAS; softmax over a contiguous row instead of a strided column).
  m17  opm_fused    OuterProductMean + the block's pair residual add as: (p) ONE producer kernel LN(m) -> W (K=128, N=64) -> * row mask, writing the
                    two cuBLAS operands A2 [(i c), M] / B2 [(j d), M] bf16 directly (no chunk / permute / reshape copies); (g) the SAME cuBLAS
                    problem as stock's einsum ([(i c), M] x [M, (j d)] -> bf16); (e) ONE epilogue kernel: K=1024 -> 256 projection (fp32
                    acc, single ascending chain) + bias -> bf16, / n_valid (div.rn) -> bf16, + pair -> bf16, written straight into the
                    new pair tensor.  TIER-2 (LN statistics order, K=128 producer accumulation order).
  mh   msa_hoist    EXACT class (bitwise by construction): when ``maybe_subsample_msa`` returned its inputs unchanged (MSA depth <= msa_max_depth:
                    no row subsample, no RNG draw) the encoder's inputs are the same tensors in every loop iteration of the fold, so its output
                    is computed on the first iteration and REUSED on the following ones (identity of every source tensor is checked by object,
                    version and shape; any change recomputes).  No effect when rows are subsampled (the library depth cap 1024 vs deeper
                    MSAs) — then every iteration sees a different row set and all 21 are computed as stock does.  Engagement is per fold and
                    data-dependent: STATS mh_fills / mh_hits count the folds it served, mh_subsample_rows the calls that drew rows.  When
                    engaged it holds ONE pair-sized bf16 tensor (the encoder output) for the fold; ``mh_clear()`` drops it between inputs.

Composition / ownership: this module owns ``MSAEncoderBlock.forward`` (per instance) so the three residual adds on m / pair live in the kernels'
epilogues; the block's pair tail (tri_mul_out, tri_mul_in, pair_transition) is run by ``pair_tail()`` exactly as ef2_opt's M1 block forward
runs it (``C._fused_trimul_with_residual`` when the model's kernel backend is 'fused', served through ef2_w4's TriMul wrapper under `tx` when live; ``self.pair_transition``), so the
TriMul / PairTransition levers of the other modules apply unchanged.  Install AFTER ef2_opt.install (it patches the block forwards for M1 and
wraps msa_encoder.forward for the encoder graphs; this module wraps on top of both) and BEFORE the first fold (graph capture is lazy:
``enable`` asserts no msa_encoder graph exists yet).  Capture-safe: no host syncs, no data-dependent host control flow inside the block
forward; static shapes per (L, M).  `mh` lives outside the captured region (it wraps the graphed encoder forward).

Scope: ``static_refusals()`` decides from the module configuration alone (dims, biases, chunking) whether each requested lever can serve THIS
model, before the first fold — ``enable()`` raises by name otherwise (the Fast variant has no msa_encoder: every lever of this module refuses
there).  The per-call ``_ok`` predicates re-check run-time facts only (CUDA, bf16 autocast, inference mode, contiguity, batch 1); a served call
that still took the stock statements is counted (STATS m15/m16/m17_fallthrough, expected 0).

Usage:  import ef2_msa_v2; ef2_msa_v2.enable(model, mtr_fused=True, pwa_fused=True, opm_fused=True, msa_hoist=True)
        ef2_msa_v2.install(model, "m15,m16,m17,mh"); ef2_msa_v2.describe(); ef2_msa_v2.stats(); ef2_msa_v2.disable(model); ef2_msa_v2.mh_clear()
Credits: ef2_msa (conventions, the OPM projection chain, the PWA rounding analysis), ef2_w4 T10 (one-kernel
transition structure), Biohub (model code).
"""
import sys, types, weakref, collections
import torch
import triton
import triton.language as tl

VERSION = "msa2.1.1"
STATS = collections.Counter()
_BF16 = torch.bfloat16
_STATE = dict(enabled=False, mtr_fused=False, pwa_fused=False, opm_fused=False, msa_hoist=False, models=[])
LEVER_FLAGS = dict(m15="mtr_fused", m16="pwa_fused", m17="opm_fused", mh="msa_hoist")     # the kit's lever names -> enable() keywords
LEVERS_FAST = ("m15", "m16", "m17")            # tolerance class
LEVERS_EXACT = ("mh",)                        # bitwise class


# =====================================================================================================================
# shared: in-register row LayerNorm (fp32 statistics) of a [BR, K] fp32 tile -> bf16 (the value a stock Linear consumes under bf16 autocast)
# =====================================================================================================================
@triton.jit
def _ln_tile_bf16(x, w_ptr, b_ptr, cols, eps, K: tl.constexpr, HAS_BIAS: tl.constexpr):
    mean = tl.sum(x, axis=1) / K
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / K
    rstd = 1.0 / tl.sqrt_rn(var + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :]
    if HAS_BIAS:
        y = y + tl.load(b_ptr + cols).to(tl.float32)[None, :]
    return y.to(tl.bfloat16)


# =====================================================================================================================
# m15: msa_transition + residual in one kernel
# =====================================================================================================================
@triton.jit
def _mtr_fused_kernel(X_ptr, OUT_ptr, LNW_ptr, LNB_ptr, W12T_ptr, W3T_ptr, R, eps,
                      K: tl.constexpr, HID: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr, HAS_LN_BIAS: tl.constexpr):
    """rows r of m [R, K] (bf16): out[r] = bf16( m[r] + bf16( W3 . bf16( bf16(silu(bf16(x1))) * bf16(x2) ) ) ),  [x1|x2] = bf16(LN(m[r]) @ W12^T)
    W12T [K, 2*HID] bf16 (cols 0..HID-1 -> x1, HID.. -> x2 = the stock chunk order), W3T [HID, K] bf16."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    rows = rows.to(tl.int64)                                             # R*K passes 2^31 at L*M >= 2^24
    cols = tl.arange(0, K)
    xr = tl.load(X_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0)
    x = xr.to(tl.float32)
    xn = _ln_tile_bf16(x, LNW_ptr, LNB_ptr, cols, eps, K, HAS_LN_BIAS)   # [BR, K] bf16
    acc = tl.zeros((BR, K), dtype=tl.float32)
    hh = tl.arange(0, BH)
    for h0 in tl.range(0, HID, BH):
        w1 = tl.load(W12T_ptr + cols[:, None] * (2 * HID) + (h0 + hh)[None, :])            # [K, BH]
        w2 = tl.load(W12T_ptr + cols[:, None] * (2 * HID) + (HID + h0 + hh)[None, :])
        x1 = tl.dot(xn, w1, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)          # Linear output rounding point (x12 is bf16)
        x2 = tl.dot(xn, w2, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        s = (x1 / (1.0 + tl.exp(-x1))).to(tl.bfloat16).to(tl.float32)                     # F.silu on bf16: fp32 opmath -> bf16
        hb = (s * x2).to(tl.bfloat16)                                                      # bf16 * bf16 -> fp32 -> bf16
        w3 = tl.load(W3T_ptr + (h0 + hh)[:, None] * K + cols[None, :])                     # [BH, K]
        acc = tl.dot(hb, w3, acc, out_dtype=tl.float32)
    y = acc.to(tl.bfloat16).to(tl.float32)                                                 # w3 output bf16
    out = (x + y).to(tl.bfloat16)                                                          # m + t  (bf16 + bf16: fp32 add, one rounding)
    tl.store(OUT_ptr + rows[:, None] * K + cols[None, :], out, mask=rmask[:, None])


_M15_CFG = dict(BR=64, BH=64, num_warps=4, num_stages=2)          # tile of the m15 kernel (rows per program, hidden chunk)


def _mtr_weights(tr):
    """(W12^T [128, 1024] bf16 contiguous, W3^T [512, 128] bf16 contiguous) cached on the instance, keyed by parameter identity/version."""
    w12, w3 = tr.ffn.w12.weight, tr.ffn.w3.weight
    key = (w12.data_ptr(), w12._version, w3.data_ptr(), w3._version)
    ent = getattr(tr, "_ef2msa2_wc", None)
    if ent is None or ent[0] != key:
        ent = (key, w12.detach().to(_BF16).t().contiguous(), w3.detach().to(_BF16).t().contiguous())
        tr._ef2msa2_wc = ent; STATS["m15_weight_pack"] += 1
    return ent[1], ent[2]


def mtr_fused(tr, m, out=None, cfg=None):
    """m [..., 128] bf16 contiguous -> m + msa_transition(m) (bf16), written into `out` (may alias m: rows are independent)."""
    cfg = cfg or _M15_CFG
    K = m.shape[-1]
    x2 = m.view(-1, K)
    assert x2.is_contiguous()
    W12T, W3T = _mtr_weights(tr)
    HID = W3T.shape[0]
    if out is None:
        out = torch.empty_like(m)
    o2 = out.view(-1, K)
    R = x2.shape[0]
    grid = (triton.cdiv(R, cfg["BR"]),)
    _mtr_fused_kernel[grid](x2, o2, tr.norm.weight, tr.norm.bias if tr.norm.bias is not None else tr.norm.weight, W12T, W3T, R, tr.norm.eps,
                            K=K, HID=HID, BR=cfg["BR"], BH=cfg["BH"], HAS_LN_BIAS=tr.norm.bias is not None,
                            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    STATS["m15_calls"] += 1
    return out


def _mtr_ok(tr, m):
    K = m.shape[-1]
    return (m.is_cuda and m.dtype == _BF16 and m.is_contiguous() and K == 128 and tr.norm.normalized_shape[0] == K
            and tr.ffn.w12.bias is None and tr.ffn.w3.bias is None and tr.ffn.w12.weight.shape[0] == 2 * tr.ffn.w3.weight.shape[1]
            and tr.ffn.w3.weight.shape[0] == K and (tr._chunk_size is None or m.shape[1] <= tr._chunk_size) and not torch.is_grad_enabled())


# =====================================================================================================================
# block forward (owns the residual adds) + pair tail (as ef2_opt M1)
# =====================================================================================================================
def _msa_trimul_call(C, blk, pb, direction, maskf):
    """One MSA-block fused TriMul+residual call.  When ef2_w4's weight-cast cache or its provider binding is live, the call goes through
    ef2_w4's PairUpdateBlock wrapper: the eight bf16 weight casts are cached per (block, direction) and the provider receives a STABLE weight
    identity, so its address-keyed weight packs are made once per block — the per-call casts this path used to hand down made the provider
    pack (and, on cc 8.0, retain) a fresh weight set on every call: +0.25 GiB per fold at 1400 tokens.  Values identical (same casts of the
    same parameters; same mask, eps, residual)."""
    W4 = sys.modules.get("ef2_w4")
    st = getattr(W4, "_STATE", None) if W4 is not None else None
    if st and st.get("enabled") and (st.get("weight_cache") or st.get("tx")) and hasattr(W4, "_pub_fused_trimul_w4"):
        return W4._pub_fused_trimul_w4(blk, pb, direction, maskf)
    eng = (blk.tri_mul_out if direction == "outgoing" else blk.tri_mul_in)._engine
    p_in_w, g_in_w = eng.split_kernel_weights()
    bf = lambda t: t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)  # noqa: E731
    return C._fused_trimul_with_residual(pb, direction, residual=pb, drop_mask=None, norm_in_weight=bf(eng.norm_start.weight), norm_in_bias=bf(eng.norm_start.bias),
                                        p_in_weight=bf(p_in_w), g_in_weight=bf(g_in_w), norm_out_weight=bf(eng.norm_mix.weight), norm_out_bias=bf(eng.norm_mix.bias),
                                        p_out_weight=bf(eng.proj_emit.weight), g_out_weight=bf(eng.proj_gate.weight), mask=maskf, eps=C._EPS)


def pair_tail(self, pair, pair_attention_mask):
    """tri_mul_out -> tri_mul_in -> pair_transition with their residuals, exactly as ef2_opt._msa_block_forward_fused runs them (M1: the fused
    TriMul+residual call when the model's kernel backend is 'fused' — through ef2_w4's TriMul wrapper (cached casts, the provider under `tx`) when live — else the stock modules)."""
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    fused_blk = getattr(self, "_ef2opt_fused", False)
    kb = getattr(self, "_ef2opt_backend_owner", None)
    fused_mode = (getattr(kb, "_kernel_backend", "fused") == "fused") if kb is not None else True
    if fused_blk and fused_mode and (not torch.is_grad_enabled()) and pair.is_cuda and C.TRITON_KERNELS_AVAILABLE:
        import ef2_opt
        if ef2_opt.CFG.enabled:
            orig_dtype = pair.dtype
            pb = pair.to(torch.bfloat16)
            maskf = pair_attention_mask.to(pb.dtype) if pair_attention_mask is not None else None
            for direction in ("outgoing", "incoming"):
                pb = _msa_trimul_call(C, self, pb, direction, maskf)       # cached casts + a stable provider weight identity (one pack per block)
            pair = pb.to(orig_dtype)
            ef2_opt.STATS["msa_fused_trimul_calls"] += 2
            pair = pair + self.pair_transition(pair)
            return pair
    pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
    pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
    pair = pair + self.pair_transition(pair)
    return pair


def _block_forward_v2(self, m, pair, msa_attention_mask, pair_attention_mask):
    """MSAEncoderBlock.forward with this module's levers (per instance).  m is updated IN PLACE by m16/m15 (m is private to MSAEncoder.forward)."""
    STATS["block_calls"] += 1
    opm = self.outer_product_mean
    if _STATE["opm_fused"] and _opm_ok(opm, m, msa_attention_mask, pair):
        pair = opm_fused(opm, m, msa_attention_mask, pair)                  # pair + opm(m)   (new tensor: the first block's pair aliases z_init)
    else:
        STATS["m17_fallthrough"] += int(bool(_STATE["opm_fused"]))
        pair = pair + opm(m, msa_attention_mask)
    if not self.is_final_block:
        pwa = self.msa_pair_weighted_averaging
        if _STATE["pwa_fused"] and _pwa_ok(pwa, m, pair, pair_attention_mask):
            m = pwa_fused(pwa, m, pair, pair_attention_mask, out=m)         # m + pwa(m, pair)  in place
        else:
            STATS["m16_fallthrough"] += int(bool(_STATE["pwa_fused"]))
            m = m + pwa(m, pair, pair_attention_mask)
        tr = self.msa_transition
        if _STATE["mtr_fused"] and _mtr_ok(tr, m):
            m = mtr_fused(tr, m, out=m)                                      # m + transition(m)  in place
        else:
            STATS["m15_fallthrough"] += int(bool(_STATE["mtr_fused"]))
            m = m + tr(m)
    pair = pair_tail(self, pair, pair_attention_mask)
    return m, pair


# =====================================================================================================================
# m16: PairWeightedAveraging pipeline
# =====================================================================================================================
@triton.jit
def _pwa_proj_kernel(X_ptr, LNW_ptr, LNB_ptr, WVT_ptr, WGT_ptr, V_ptr, G_ptr, R, eps,
                     K: tl.constexpr, HD: tl.constexpr, DH: tl.constexpr, BR: tl.constexpr, HAS_LN_BIAS: tl.constexpr):
    """rows r=(l,m) of m [R, K]: xn = bf16(LN(m[r])); v = bf16(xn @ Wv^T) written head-major V[h, r, d] (h = e // DH, d = e % DH);
    g = bf16(sigmoid(bf16(xn @ Wg^T))) written G[r, e].  WVT/WGT [K, HD] bf16."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    rows = rows.to(tl.int64)
    cols = tl.arange(0, K)
    x = tl.load(X_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    xn = _ln_tile_bf16(x, LNW_ptr, LNB_ptr, cols, eps, K, HAS_LN_BIAS)
    ee = tl.arange(0, HD)
    wv = tl.load(WVT_ptr + cols[:, None] * HD + ee[None, :])                                   # [K, HD]
    v = tl.dot(xn, wv, out_dtype=tl.float32).to(tl.bfloat16)
    hh = (ee // DH).to(tl.int64); dd = ee % DH
    tl.store(V_ptr + hh[None, :] * R * DH + rows[:, None] * DH + dd[None, :], v, mask=rmask[:, None])
    wg = tl.load(WGT_ptr + cols[:, None] * HD + ee[None, :])
    gl = tl.dot(xn, wg, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)                  # Wgate output bf16
    g = (1.0 / (1.0 + tl.exp(-gl))).to(tl.bfloat16)                                            # torch.sigmoid on bf16: fp32 opmath -> bf16
    tl.store(G_ptr + rows[:, None] * HD + ee[None, :], g, mask=rmask[:, None])


MASKVAL = tl.constexpr(float(torch.tensor(-1e5, dtype=torch.bfloat16)))    # masked_fill_(-1e5) on a bf16 tensor stores bf16(-1e5) = -99840.0


@triton.jit
def _pwa_bias_softmax_kernel(Z_ptr, LNW_ptr, LNB_ptr, WBT_ptr, MASK_ptr, LG_ptr, ATTN_ptr, L, eps,
                             C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, BJ: tl.constexpr, HAS_LN_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, NSTAGE: tl.constexpr):
    """one program per query row i: logits[h, j] = f32(bf16( bf16(LN(z[i, j])) @ WbT[:, h] )) (bf16(-1e5) where mask[i, j] == 0) for all j (BJ at a
    time, loads pipelined), kept in an fp32 scratch row LG[i, h, j]; then softmax over j in fp32 exactly as max -> sum(exp(x - max)) -> exp(x - max) / sum,
    written bf16 to ATTN[h, i, j] (the [H, L, L] layout the bmm consumes).  z [L, L, C] bf16; WbT [C, HP] bf16 (heads zero-padded to HP >= 16)."""
    i = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, C)
    hcol = tl.arange(0, HP)
    hm = hcol < H
    w = tl.load(WBT_ptr + cols[:, None] * HP + hcol[None, :])                                  # [C, HP] bf16
    mx = tl.full((HP,), float("-inf"), tl.float32)
    lg_row = LG_ptr + i * (HP * L)                                                               # scratch [L(i), HP, L(j)] fp32
    for j0 in tl.range(0, L, BJ, num_stages=NSTAGE):
        offs_j = j0 + tl.arange(0, BJ); jm = offs_j < L
        offs_j64 = offs_j.to(tl.int64)
        z = tl.load(Z_ptr + (i * L + offs_j64)[:, None] * C + cols[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        xn = _ln_tile_bf16(z, LNW_ptr, LNB_ptr, cols, eps, C, HAS_LN_BIAS)
        lg = tl.dot(xn, w, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)                # Linear(256->8) output bf16
        if HAS_MASK:
            keep = tl.load(MASK_ptr + i * L + offs_j64, mask=jm, other=1) != 0
            lg = tl.where(keep[:, None], lg, MASKVAL)
        lg = tl.where(jm[:, None], lg, float("-inf"))
        mx = tl.maximum(mx, tl.max(lg, axis=0))
        tl.store(lg_row + hcol[None, :] * L + offs_j64[:, None], lg, mask=jm[:, None] & hm[None, :])
    ssum = tl.zeros((HP,), tl.float32)
    for j0 in tl.range(0, L, BJ):
        offs_j = j0 + tl.arange(0, BJ); jm = offs_j < L
        lg = tl.load(lg_row + hcol[None, :] * L + offs_j[:, None], mask=jm[:, None] & hm[None, :], other=float("-inf"))
        ssum += tl.sum(tl.exp(lg - mx[None, :]), axis=0)
    for j0 in tl.range(0, L, BJ):
        offs_j = j0 + tl.arange(0, BJ); jm = offs_j < L
        offs_j64 = offs_j.to(tl.int64)
        lg = tl.load(lg_row + hcol[None, :] * L + offs_j[:, None], mask=jm[:, None] & hm[None, :], other=float("-inf"))
        p = tl.exp(lg - mx[None, :]) / ssum[None, :]
        tl.store(ATTN_ptr + (hcol[None, :].to(tl.int64) * L + i) * L + offs_j64[:, None], p.to(tl.bfloat16), mask=jm[:, None] & hm[None, :])


@triton.jit
def _pwa_out_kernel(O_ptr, G_ptr, M_ptr, WOT_ptr, R, K: tl.constexpr, HD: tl.constexpr, DH: tl.constexpr, BR: tl.constexpr):
    """rows r of m: x = bf16( f32(O[h, r, d]) * f32(G[r, e]) ); y = bf16(x @ Wout^T); m[r] = bf16( f32(m[r]) + f32(y) )  (in place).
    O head-major [H, R, DH] bf16 (= bf16(attn . v)), G [R, HD] bf16, WOT [HD, K] bf16."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    rows = rows.to(tl.int64)
    ee = tl.arange(0, HD); cols = tl.arange(0, K)
    hh = (ee // DH).to(tl.int64); dd = ee % DH
    o = tl.load(O_ptr + hh[None, :] * R * DH + rows[:, None] * DH + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    g = tl.load(G_ptr + rows[:, None] * HD + ee[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    x = (o * g).to(tl.bfloat16)
    wo = tl.load(WOT_ptr + ee[:, None] * K + cols[None, :])                                   # [HD, K]
    y = tl.dot(x, wo, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    mrow = tl.load(M_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    tl.store(M_ptr + rows[:, None] * K + cols[None, :], (mrow + y).to(tl.bfloat16), mask=rmask[:, None])


_M16_CFG = dict(BR=64, num_warps=4, num_stages=2, BJ=64, bias_warps=4, bias_stages=3, BRO=64, out_warps=4)   # tiles of the m16 kernels (producer / bias+softmax / epilogue)


def _pwa_weights(pwa):
    ps = (pwa.Wv.weight, pwa.Wgate.weight, pwa.Wout.weight, pwa.compute_bias[1].weight)
    key = tuple((p.data_ptr(), p._version) for p in ps)
    ent = getattr(pwa, "_ef2msa2_wc", None)
    if ent is None or ent[0] != key:
        wb = ps[3].detach().to(_BF16)                                        # [H, 256]
        HP = max(16, triton.next_power_of_2(wb.shape[0]))
        wbt = torch.zeros((wb.shape[1], HP), dtype=_BF16, device=wb.device); wbt[:, :wb.shape[0]] = wb.t()
        ent = (key, ps[0].detach().to(_BF16).t().contiguous(), ps[1].detach().to(_BF16).t().contiguous(), ps[2].detach().to(_BF16).t().contiguous(), wbt)   # WvT [128, HD], WgT [128, HD], WoutT [HD, 128], WbT [256, HP]
        pwa._ef2msa2_wc = ent; STATS["m16_weight_pack"] += 1
    return ent[1:]


def pwa_fused(pwa, m, pair, pair_attention_mask, out=None, cfg=None):
    """m [1, L, M, 128] bf16, pair [1, L, L, 256] bf16, pair_attention_mask [1, L, L] -> m + pwa(m, pair, mask) written into `out` (may alias m)."""
    cfg = cfg or _M16_CFG
    B, L, M, K = m.shape
    H, DH = pwa.n_heads, pwa.head_width
    HD = H * DH
    R = L * M
    WvT, WgT, WoT, WbT = _pwa_weights(pwa)
    HP = WbT.shape[1]
    x2 = m.view(R, K)
    dev = m.device
    # (k1) producer: LN + Wv|Wgate + sigmoid -> v head-major [H, R, DH], g [R, HD]
    v_t = torch.empty((H, R, DH), device=dev, dtype=_BF16); g = torch.empty((R, HD), device=dev, dtype=_BF16)
    ln = pwa.norm_single
    _pwa_proj_kernel[(triton.cdiv(R, cfg["BR"]),)](x2, ln.weight, ln.bias if ln.bias is not None else ln.weight, WvT, WgT, v_t, g, R, ln.eps,
                                                    K=K, HD=HD, DH=DH, BR=cfg["BR"], HAS_LN_BIAS=ln.bias is not None, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    # (k2) bias: LN(z) . Wb + key mask -> softmax over j (fp32) -> attn [H, L, L] bf16, one program per row i
    lnz = pwa.compute_bias[0]
    z2 = pair[0]
    assert z2.is_contiguous()
    mask2 = pair_attention_mask[0] if pair_attention_mask is not None else None
    if mask2 is not None and mask2.dtype == torch.bool:
        mask2 = mask2.view(torch.uint8)                                      # reinterpret, no copy
    if mask2 is not None and not mask2.is_contiguous():
        mask2 = mask2.contiguous()
    scratch = torch.empty((L, HP, L), device=dev, dtype=torch.float32)
    attn = torch.empty((H, L, L), device=dev, dtype=_BF16)
    _pwa_bias_softmax_kernel[(L,)](z2, lnz.weight, lnz.bias if lnz.bias is not None else lnz.weight, WbT, mask2 if mask2 is not None else z2, scratch, attn, L, lnz.eps,
                                   C=z2.shape[-1], H=H, HP=HP, BJ=cfg["BJ"], HAS_LN_BIAS=lnz.bias is not None, HAS_MASK=mask2 is not None, NSTAGE=cfg["bias_stages"], num_warps=cfg["bias_warps"])
    # (k3) attn . v : one bmm on the head-major layout -> O [H, R, DH] bf16
    o_t = torch.bmm(attn, v_t.view(H, L, M * DH)).view(H, R, DH)
    # (k4) epilogue: gate, Wout, residual -> m (in place)
    if out is None:
        out = m.clone()
    elif out.data_ptr() != m.data_ptr():
        out.copy_(m)
    o2 = out.view(R, K)
    _pwa_out_kernel[(triton.cdiv(R, cfg["BRO"]),)](o_t, g, o2, WoT, R, K=K, HD=HD, DH=DH, BR=cfg["BRO"], num_warps=cfg["out_warps"])
    STATS["m16_calls"] += 1
    return out


def _pwa_ok(pwa, m, pair, pair_attention_mask):
    if torch.is_grad_enabled() or not (m.is_cuda and m.dtype == _BF16 and m.dim() == 4 and m.shape[0] == 1 and m.is_contiguous()):
        return False
    H, DH = pwa.n_heads, pwa.head_width; HD = H * DH
    return (m.shape[-1] == 128 and pwa.Wv.weight.shape == (HD, 128) and pwa.Wout.weight.shape == (128, HD) and (HD & (HD - 1)) == 0 and HD <= 256
            and (DH & (DH - 1)) == 0 and pair.dtype == _BF16 and pair.shape[-1] == 256 and pair.shape[0] == 1 and pair_attention_mask is not None
            and pwa.Wv.bias is None and pwa.Wgate.bias is None and pwa.Wout.bias is None and pwa.compute_bias[1].bias is None
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == _BF16)


# =====================================================================================================================
# m17: OuterProductMean producer + cuBLAS + fused projection/normalise/residual epilogue
# =====================================================================================================================
@triton.jit
def _opm_ab_kernel(X_ptr, LNW_ptr, LNB_ptr, WT_ptr, MASK_ptr, A_ptr, B_ptr, R, MDEP, eps,
                   K: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr, HAS_LN_BIAS: tl.constexpr):
    """rows r=(l,m) of m [R, K], M = MDEP rows per l: ab = bf16( bf16(LN(m[r])) @ W^T ) * mask[r] (exact for mask in {0,1});
    A2[(l*CH + c), m] = ab[c], B2[(l*CH + d), m] = ab[CH + d]  (A2/B2 [(L*CH), MDEP] bf16 = the operands stock's einsum hands cuBLAS).  WT [K, 2*CH]."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    rows = rows.to(tl.int64)
    cols = tl.arange(0, K)
    x = tl.load(X_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    xn = _ln_tile_bf16(x, LNW_ptr, LNB_ptr, cols, eps, K, HAS_LN_BIAS)
    cc = tl.arange(0, 2 * CH)
    w = tl.load(WT_ptr + cols[:, None] * (2 * CH) + cc[None, :])                                 # [K, 2CH]
    ab = tl.dot(xn, w, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    mk = tl.load(MASK_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    ab = (ab * mk[:, None]).to(tl.bfloat16)
    l_idx = rows // MDEP; m_idx = rows % MDEP
    c_idx = (cc % CH).to(tl.int64)
    dst = (l_idx[:, None] * CH + c_idx[None, :]) * MDEP + m_idx[:, None]                        # [(l c), m]
    is_a = cc[None, :] < CH
    tl.store(A_ptr + dst, ab, mask=rmask[:, None] & is_a)
    tl.store(B_ptr + dst, ab, mask=rmask[:, None] & (cc[None, :] >= CH))


@triton.jit
def _opm_proj_res_kernel(WS_ptr, WT_ptr, BIAS_ptr, NV_ptr, PAIR_ptr, OUT_ptr, L, stride_ws_row, stride_p_i, stride_p_j, stride_nv_i,
                         C_HID: tl.constexpr, D_OUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CPK: tl.constexpr, NSTAGE: tl.constexpr, HAS_RES: tl.constexpr):
    """the projection + the block's residual: for one i (pid 0), BM j's (pid 1), BN n's (pid 2):
    acc[j, n] = sum_k ws[i*C + c(k), j*C + d(k)] * Wt[k, n] over k = c*C_HID + d ascending (single fp32 chain: CPK c-rows of C_HID d's per dot step);
    y = bf16(acc + bias); o = bf16(div_rn(y, nv[i, j])); out[i, j, n] = bf16( f32(pair[i, j, n]) + f32(o) )  (HAS_RES) else o.
    Each c-row of the ws tile is one contiguous [BM*C_HID] run (j-major, d-minor) -> vector loads."""
    i = tl.program_id(0).to(tl.int64); jb = tl.program_id(1); nb = tl.program_id(2)
    offs_j = jb * BM + tl.arange(0, BM); offs_n = nb * BN + tl.arange(0, BN)
    jmask = offs_j < L
    BK: tl.constexpr = CPK * C_HID
    kk = tl.arange(0, BK)
    cc = kk // C_HID; dd = kk % C_HID
    a_ptrs = WS_ptr + (i * C_HID) * stride_ws_row + cc[None, :] * stride_ws_row + offs_j[:, None] * C_HID + dd[None, :]
    b_ptrs = WT_ptr + kk[:, None] * D_OUT + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for c0 in tl.range(0, C_HID, CPK, num_stages=NSTAGE):
        a = tl.load(a_ptrs, mask=jmask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += CPK * stride_ws_row
        b_ptrs += BK * D_OUT
    bias = tl.load(BIAS_ptr + offs_n).to(tl.float32)
    y = (acc + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    nv = tl.load(NV_ptr + i * stride_nv_i + offs_j, mask=jmask, other=1.0).to(tl.float32)
    o = tl.div_rn(y, tl.broadcast_to(nv[:, None], (BM, BN))).to(tl.bfloat16)
    offs_j64 = offs_j.to(tl.int64)
    po = i * stride_p_i + offs_j64[:, None] * stride_p_j + offs_n[None, :]
    if HAS_RES:
        p = tl.load(PAIR_ptr + po, mask=jmask[:, None], other=0.0).to(tl.float32)
        o = (p + o.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT_ptr + po, o, mask=jmask[:, None])


_M17_CFG = dict(BR=128, num_warps=4, BM=64, BN=256, CPK=1, proj_warps=8, proj_stages=4)   # tiles of the m17 kernels (producer / projection epilogue)


def _opm_weights(opm):
    ps = (opm.W.weight, opm.Wout.weight, opm.Wout.bias)
    key = tuple((p.data_ptr(), p._version) for p in ps)
    ent = getattr(opm, "_ef2msa2_wc", None)
    if ent is None or ent[0] != key:
        ent = (key, ps[0].detach().to(_BF16).t().contiguous(), ps[1].detach().to(_BF16).t().contiguous(), ps[2].detach().to(_BF16).contiguous())   # WT [128, 64], WoutT [1024, 256], bias [256]
        opm._ef2msa2_wc = ent; STATS["m17_weight_pack"] += 1
    return ent[1:]


def opm_fused(opm, m, msa_attention_mask, pair=None, cfg=None):
    """m [1, L, M, 128] bf16, msa_attention_mask [1, L, M] float {0,1}, pair [1, L, L, 256] bf16 or None -> pair + opm(m) (new tensor) or opm(m)."""
    cfg = cfg or _M17_CFG
    B, L, M, K = m.shape
    CH = opm.d_hidden
    R = L * M
    WT, WoT, bias = _opm_weights(opm)
    D = WoT.shape[1]
    dev = m.device
    x2 = m.view(R, K)
    maskf = msa_attention_mask.to(torch.float32)                             # stock: mask_f = mask.to(a.dtype), a fp32
    A2 = torch.empty((L * CH, M), device=dev, dtype=_BF16); B2 = torch.empty((L * CH, M), device=dev, dtype=_BF16)
    ln = opm.norm
    mrow = maskf.reshape(R)
    _opm_ab_kernel[(triton.cdiv(R, cfg["BR"]),)](x2, ln.weight, ln.bias if ln.bias is not None else ln.weight, WT, mrow, A2, B2, R, M, ln.eps,
                                                  K=K, CH=CH, BR=cfg["BR"], HAS_LN_BIAS=ln.bias is not None, num_warps=cfg["num_warps"])
    ws = torch.matmul(A2, B2.t())                                            # [(i c), (j d)] bf16: the cuBLAS problem stock's einsum lowers to
    n_valid = (maskf @ maskf.transpose(-1, -2)).clamp(min=1.0)               # [1, L, L] (bf16 under autocast, as stock)
    res = pair is not None
    out = torch.empty((B, L, L, D), device=dev, dtype=_BF16)
    p0 = pair[0] if res else out[0]
    assert out[0].stride() == p0.stride()
    grid = (L, triton.cdiv(L, cfg["BM"]), D // cfg["BN"])
    _opm_proj_res_kernel[grid](ws, WoT, bias, n_valid[0], p0, out[0], L, ws.stride(0), out.stride(1), out.stride(2), n_valid.stride(1),
                               C_HID=CH, D_OUT=D, BM=cfg["BM"], BN=cfg["BN"], CPK=cfg["CPK"], NSTAGE=cfg["proj_stages"], HAS_RES=res, num_warps=cfg["proj_warps"])
    STATS["m17_calls"] += 1
    return out


def _opm_ok(opm, m, msa_attention_mask, pair):
    cfg = _M17_CFG
    return (not torch.is_grad_enabled() and m.is_cuda and m.dtype == _BF16 and m.is_contiguous() and m.dim() == 4 and m.shape[0] == 1 and m.shape[-1] == 128
            and opm._chunk_size is None and not opm.divide_outer_before_proj and opm.W.bias is None and opm.Wout.bias is not None
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == _BF16
            and opm.Wout.out_features % cfg["BN"] == 0 and opm.d_hidden % cfg["CPK"] == 0 and opm.d_hidden == 32
            and msa_attention_mask is not None and msa_attention_mask.dim() == 3 and pair.dtype == _BF16 and pair.is_contiguous() and pair.shape[-1] == opm.Wout.out_features)


# =====================================================================================================================
# MH: loop-invariant MSA-encoder output reuse (EXACT class)
# =====================================================================================================================
_MH = dict(last=None, cache=None, installed=False)


def _tver(t):
    """version counter of `t` (inference tensors carry none: the loop's MSA inputs are read-only locals of ESMFold2Model.forward)."""
    return None if t.is_inference() else t._version


def _tsig(t):
    return None if t is None else (weakref.ref(t), t.data_ptr(), _tver(t), tuple(t.shape), t.dtype)


def _tsig_same(sig, t):
    if sig is None or t is None:
        return sig is None and t is None
    r, ptr, ver, shape, dt = sig
    return r() is t and t.data_ptr() == ptr and _tver(t) == ver and tuple(t.shape) == shape and t.dtype == dt


def _mh_subsample_wrapper(orig):
    def maybe_subsample_msa(msa, msa_attention_mask, has_deletion, deletion_value, *, max_depth, enabled):
        out = orig(msa, msa_attention_mask, has_deletion, deletion_value, max_depth=max_depth, enabled=enabled)
        identity = out[0] is msa and out[1] is msa_attention_mask and out[2] is has_deletion and out[3] is deletion_value
        _MH["last"] = (dict(identity=identity, src=tuple(_tsig(t) for t in (msa, msa_attention_mask, has_deletion, deletion_value)), depth=int(msa.size(1)), max_depth=max_depth, enabled=enabled)
                       if _STATE["msa_hoist"] else None)
        STATS["mh_subsample_identity" if identity else "mh_subsample_rows"] += 1
        return out
    maybe_subsample_msa._ef2msa2_orig = orig
    return maybe_subsample_msa


def _mh_encoder_forward(self, *args, **kwargs):
    """msa_encoder.forward wrapper: reuse the previous output when the row subsample was the identity and every source / argument tensor is the
    same unmodified tensor as at the cached call (EXACT: the encoder is a deterministic RNG-free function of these)."""
    inner = self._ef2msa2_inner_forward
    last = _MH["last"]; _MH["last"] = None
    if not (_STATE["msa_hoist"] and last is not None and last["identity"] and not args and not torch.is_grad_enabled()):
        STATS["mh_miss_noident"] += int(bool(_STATE["msa_hoist"]))
        return inner(*args, **kwargs)
    x_pair, x_inputs = kwargs.get("x_pair"), kwargs.get("x_inputs")
    ent = _MH["cache"]

    def _same_src(a, b):                                                     # cached source signature vs this call's: same live tensor object, ptr, version, shape, dtype
        if a is None or b is None:
            return a is None and b is None
        ta = a[0]()
        return ta is not None and ta is b[0]() and a[1:] == b[1:]
    if (ent is not None and ent["model"] is self and _tsig_same(ent["x_pair"], x_pair) and _tsig_same(ent["x_inputs"], x_inputs)
            and len(ent["src"]) == len(last["src"]) and all(_same_src(a, b) for a, b in zip(ent["src"], last["src"]))
            and tuple(kwargs["msa_oh"].shape) == ent["msa_oh_shape"] and _tver(ent["out"]) == ent["out_version"]):
        STATS["mh_hits"] += 1
        return ent["out"]
    out = inner(*args, **kwargs)
    if torch.cuda.is_current_stream_capturing():                             # never cache under a capture (the decision would be frozen into the graph): computed, not held
        STATS["mh_fill_skipped_capturing"] += 1
        return out
    held = out.clone()                                                       # the encoder's output CLONED: a graphed encoder (ef2_opt eg / G4) returns a pool-resident static output that
    _MH["cache"] = dict(model=self, x_pair=_tsig(x_pair), x_inputs=_tsig(x_inputs), src=last["src"], msa_oh_shape=tuple(kwargs["msa_oh"].shape), out=held, out_version=_tver(held))
    STATS["mh_fills"] += 1                                                   # is valid only until another graph of the generation replays; the clone (one [L, L, 256] bf16 tensor per engaged
    _MH["depth"] = (last["depth"], last["max_depth"])                        # fold) is what the remaining recycles read (depth last["depth"] <= last["max_depth"]: no row subsample, no RNG draw)
    return out


def _install_mh(model):
    import transformers.models.esmfold2.modeling_esmfold2 as MOD
    enc = getattr(model, "msa_encoder", None)
    if enc is None:
        return False
    if not getattr(MOD.maybe_subsample_msa, "_ef2msa2_orig", None):
        MOD.maybe_subsample_msa = _mh_subsample_wrapper(MOD.maybe_subsample_msa)
    if getattr(enc, "_ef2msa2_inner_forward", None) is None:
        enc._ef2msa2_inner_forward = enc.forward                             # ef2_opt's graphed forward when eg is installed, else the stock bound method
        enc._ef2msa2_had_inst_fwd = "forward" in enc.__dict__
        enc.forward = types.MethodType(_mh_encoder_forward, enc)
        enc._host_decision_lever = "mh"                                     # read by ef2_opt's G5 install: a per-call host decision must never sit inside a whole-recycle graph
    _MH["installed"] = True
    return True


def _uninstall_mh(model):
    import transformers.models.esmfold2.modeling_esmfold2 as MOD
    orig = getattr(MOD.maybe_subsample_msa, "_ef2msa2_orig", None)
    if orig is not None:
        MOD.maybe_subsample_msa = orig
    enc = getattr(model, "msa_encoder", None)
    if enc is not None and getattr(enc, "_host_decision_lever", None) == "mh":
        del enc._host_decision_lever
    if enc is not None and getattr(enc, "_ef2msa2_inner_forward", None) is not None:
        if enc._ef2msa2_had_inst_fwd:
            enc.forward = enc._ef2msa2_inner_forward
        else:
            del enc.__dict__["forward"]
        enc._ef2msa2_inner_forward = None
    _MH.update(last=None, cache=None, installed=False)


def mh_clear():
    """Drop the hoist cache — the one pair-sized tensor mh holds — between inputs (the kit's per-input cache clear calls this; a stale entry can
    never hit anyway: identity is by tensor object). Returns whether an entry was held."""
    held = _MH.get("cache") is not None
    _MH.update(last=None, cache=None)
    STATS["mh_clears"] += int(held)
    return held


# =====================================================================================================================
# plumbing
# =====================================================================================================================
def _msa_blocks(model):
    enc = getattr(model, "msa_encoder", None)
    return [] if enc is None else list(enc.blocks)


def _assert_not_captured(model, what):
    enc = getattr(model, "msa_encoder", None)
    graphs = getattr(enc, "_ef2opt_graphs", None) if enc is not None else None
    assert not graphs, (f"ef2_msa_v2.{what}() called while ef2_opt holds {len(graphs)} captured msa_encoder CUDA graph(s): the change would not reach the replayed graph "
                        "(install after ef2_server.configure() and before the first fold, or ef2_opt.clear_graphs(model) first)")


def static_refusals(model, mtr_fused=False, pwa_fused=False, opm_fused=False, msa_hoist=False):
    """The requested levers this model's MSA encoder cannot serve, by the kit's lever name, decided from the module configuration alone (dims,
    biases, chunking) BEFORE the first fold — a lever engages or ``enable()`` raises by name; nothing falls back silently.  The per-call `_ok`
    predicates then only re-check run-time facts the kit guarantees (CUDA, bf16 autocast, inference mode, contiguity); a call that still fell
    through is counted in stats() as m1x_fallthrough (expected 0)."""
    why = {}
    blocks = _msa_blocks(model)
    if (mtr_fused or pwa_fused or opm_fused or msa_hoist) and not blocks:
        return {k: "model has no msa_encoder (Fast variant)" for k, v in dict(m15=mtr_fused, m16=pwa_fused, m17=opm_fused, mh=msa_hoist).items() if v}
    for bi, blk in enumerate(blocks):
        opm = blk.outer_product_mean
        if opm_fused:
            cfg = _M17_CFG
            ok = (opm.norm.normalized_shape[0] == 128 and opm.W.bias is None and opm.Wout.bias is not None and not opm.divide_outer_before_proj and opm.d_hidden == 32
                  and opm.Wout.out_features % cfg["BN"] == 0 and opm.d_hidden % cfg["CPK"] == 0 and opm._chunk_size is None)
            if not ok: why["m17"] = f"block {bi}: OuterProductMean config (d_msa {opm.norm.normalized_shape[0]}, d_hidden {opm.d_hidden}, d_pair {opm.Wout.out_features}, chunk {opm._chunk_size}, biases) is not the served (128, 32, 256k, None) cell"
        if not blk.is_final_block:
            pwa, tr = blk.msa_pair_weighted_averaging, blk.msa_transition
            if pwa_fused:
                H, DH = pwa.n_heads, pwa.head_width; HD = H * DH
                ok = (pwa.Wv.weight.shape == (HD, 128) and pwa.Wout.weight.shape == (128, HD) and (HD & (HD - 1)) == 0 and HD <= 256 and (DH & (DH - 1)) == 0 and H <= 16
                      and pwa.Wv.bias is None and pwa.Wgate.bias is None and pwa.Wout.bias is None and pwa.compute_bias[1].bias is None and pwa.compute_bias[0].normalized_shape[0] == 256)
                if not ok: why["m16"] = f"block {bi}: MSAPairWeightedAveraging config (heads {H} x {DH}, d_msa {pwa.Wv.weight.shape[1]}, d_pair {pwa.compute_bias[0].normalized_shape[0]}, biases) is not the served cell"
            if mtr_fused:
                K = tr.norm.normalized_shape[0]
                ok = (K == 128 and tr.ffn.w12.bias is None and tr.ffn.w3.bias is None and tr.ffn.w12.weight.shape[0] == 2 * tr.ffn.w3.weight.shape[1] and tr.ffn.w3.weight.shape[0] == K
                      and tr.ffn.w3.weight.shape[1] % _M15_CFG["BH"] == 0)
                if not ok: why["m15"] = f"block {bi}: msa_transition config (d {K}, hidden {tr.ffn.w3.weight.shape[1]}, biases) is not the served cell"
    return why


def enable(model, mtr_fused=False, pwa_fused=False, opm_fused=False, msa_hoist=False):
    """Install the block forward on model.msa_encoder.blocks[*] and switch the requested levers.  Refuses by name (RuntimeError) when the model's MSA
    encoder configuration is not the cell the kernels serve (static_refusals)."""
    _assert_not_captured(model, "enable")
    why = static_refusals(model, mtr_fused, pwa_fused, opm_fused, msa_hoist)
    if why:
        raise RuntimeError("ef2_msa_v2: lever(s) cannot engage on this model: " + "; ".join(f"{k}: {v}" for k, v in why.items()))
    import ef2_srcguard                                                       # the block forward / the hoist are written against ONE upstream source: refuse by name on another
    ef2_srcguard.check_many({name: flag for name, flag in (("m15", mtr_fused), ("m16", pwa_fused), ("m17", opm_fused), ("mh", msa_hoist))})
    _eo = __import__("sys").modules.get("ef2_opt")
    if msa_hoist and _eo is not None and getattr(_eo.CFG, "recycle_graph", False) and getattr(model, "_ef2opt_name", None) == "recycle":   # mh decides hit / miss on the host inside msa_encoder.forward: under a whole-recycle graph (ef2_opt G5) that
        raise RuntimeError("ef2_msa_v2: mh cannot engage on a model with ef2_opt's recycle graph (G5) installed: the hoist's per-call host decision "
                           "would be frozen into the captured recycle body - refusing the composition by name")   # decision would be captured once and replayed for later inputs
    blocks = _msa_blocks(model)
    _STATE.update(enabled=bool(blocks), mtr_fused=bool(mtr_fused), pwa_fused=bool(pwa_fused), opm_fused=bool(opm_fused), msa_hoist=bool(msa_hoist) and bool(blocks))
    if model not in _STATE["models"]: _STATE["models"].append(model)
    n = 0
    if mtr_fused or pwa_fused or opm_fused:
        for blk in blocks:
            if getattr(blk, "_ef2msa2_orig_forward", None) is None:
                blk._ef2msa2_orig_forward = blk.__dict__.get("forward")      # ef2_opt's M1 instance forward, or None (class forward)
                blk.forward = types.MethodType(_block_forward_v2, blk)
            n += 1
    if msa_hoist:
        _install_mh(model)
    else:
        _uninstall_mh(model)
    d = describe(); d["patched_blocks"] = n
    return d


def disable(model=None):
    for mdl in ([model] if model is not None else list(_STATE["models"])):
        _assert_not_captured(mdl, "disable")
        for blk in _msa_blocks(mdl):
            if "_ef2msa2_orig_forward" in blk.__dict__:
                orig = blk.__dict__.pop("_ef2msa2_orig_forward")
                if orig is not None: blk.forward = orig
                elif "forward" in blk.__dict__: del blk.__dict__["forward"]
        _uninstall_mh(mdl)
    _STATE.update(enabled=False, mtr_fused=False, pwa_fused=False, opm_fused=False, msa_hoist=False)
    return describe()


def parse_flags(spec):
    """'m15,m16,m17,mh' | 'm15+mh' | 'all' -> enable() keywords; an unknown name raises by name."""
    fl = set(x for x in str(spec).strip().lower().replace("+", ",").split(",") if x)
    if "all" in fl:
        fl = (fl - {"all"}) | set(LEVER_FLAGS)
    unknown = sorted(fl - set(LEVER_FLAGS))
    if unknown:
        raise ValueError(f"ef2_msa_v2.parse_flags: unknown lever(s) {unknown} (known: {sorted(LEVER_FLAGS)})")
    return {kw: (name in fl) for name, kw in LEVER_FLAGS.items()}


def install(model, spec="m15,m16,m17"):
    """``install(model, "m15,m16,m17,mh")`` == ``enable(model, **parse_flags(spec))``."""
    return enable(model, **parse_flags(spec))


def labels():
    return dict(m15="TIER-2 (one-kernel msa_transition: LN statistics order, exp in silu, K=128/512 fp32 accumulation order; stock rounding points)",
                m16="TIER-2 (PWA: LN statistics order, K=128/256 accumulation order, contiguous-row softmax; stock rounding points)",
                m17="TIER-2 (OPM: LN statistics order + K=128 producer order; cuBLAS contraction = stock's; single-chain projection; stock rounding points)",
                mh="EXACT (loop-invariant encoder output reused only when every input tensor is provably the same object, unmodified)")


def levers_on():
    """The kit's lever names this module has ON right now (install order)."""
    return [name for name, kw in LEVER_FLAGS.items() if _STATE.get(kw)]


def describe():
    return dict(version=VERSION, enabled=_STATE["enabled"], levers=levers_on(), m15_mtr_fused=_STATE["mtr_fused"], m16_pwa_fused=_STATE["pwa_fused"], m17_opm_fused=_STATE["opm_fused"],
                mh_msa_hoist=_STATE["msa_hoist"], mh_held=_MH.get("cache") is not None, labels=labels(), tiles=dict(m15=dict(_M15_CFG), m16=dict(_M16_CFG), m17=dict(_M17_CFG)))


def stats():
    d = dict(STATS)
    d["version"] = VERSION
    return d
