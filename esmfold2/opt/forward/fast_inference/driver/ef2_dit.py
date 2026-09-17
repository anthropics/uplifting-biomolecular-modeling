"""ef2_dit — the diffusion sampler's token path: a device-resident roll-out (exact tier) and a fused bf16 diffusion-transformer step (fast tier).
Install as the LAST step of ef2_server.configure() (it wraps the installed
`structure_head.sample` chain and `diffusion_module.forward`); every lever is its own flag and configure() passes every flag explicitly:

    exact (opt7x):     ef2_dit.install(model, rollout=True, graph=True, static_once=True, bias_f32=True, kabsch="torch", graph_budget_tokens=None,
                                       dit=False, gemm="fp32", cond="fp32", attn="sdpa", attn_precision="ieee", fused_ew=False)      # bitwise vs stock
    fast  (opt14_msa): ef2_dit.install(model, rollout=True, graph=True, static_once=True, bias_f32=True, kabsch="device", graph_budget_tokens=None,
                                       dit=True, gemm="bf16", cond="bf16", attn="flash", attn_precision="bf16", fused_ew=True)         # tolerance tier
    'dit' composes INSTEAD of ef2_mk_sampler's 'mk' (install(dit=True) refuses by name on a model with mk enabled).

ROLL-OUT levers (structure_head.sample; model math, RNG draws, step count, schedule unchanged) — registry name `ro` (+ `kd`):
  rollout      DiffusionStructureHead.sample as ONE CUDA graph per step boundary = [Kabsch tail of step k + x update] + [centre / random
               augmentation / noise of step k+1 + DiffusionModule(k+1) + Kabsch head H(k+1)], captured once per shape key and replayed S-1 times;
               every per-step scalar comes from device fp32 tables computed on the host exactly as the stock loop folds them (fp32(t_hat),
               fp32(eps), fp32(1.0/t_hat) [ATen divides a tensor by a Python float by multiplying with the DOUBLE reciprocal rounded once],
               fp32(eta*(sigma_t - t_hat))) indexed by an on-device step counter; all per-step RNG (rotation quaternion [B,4], translation
               [B,1,3], noise [B,A,3]) is drawn right after the initial x draw in stock ORDER and SHAPES (same Philox stream). Shape key =
               (B, atoms, steps, kwargs signature, the FILLED inference cache's signature — the atom encoder's varlen tensors are valid-atom-count
               shaped); LRU `lru` keys. EXACT (bitwise).
  graph        capture+replay the boundary (else the same boundary code runs eagerly per step: same numerics, host-launch bound).
  graph_budget_tokens  None: the kit's per-shape graph budget decides capture vs eager (ef2_opt._over_graph_budget('sampler', L) — i.e. the
               per-site sampler budget where ef2_opt has one); 0: capture every shape; N: capture up to N tokens.
  static_once  fold-constant graph inputs (z_trunk, s_inputs, s_trunk, relpos, ref_*, masks, the module's inference cache) are staged into the
               graph's static buffers ONCE per fold (ef2_opt's `sg` step graph copies them in at every step); rolls that run eagerly hold plain
               references (no clone). EXACT.
  bias_f32     ef2_opt CFG.bias_f32: an fp32 twin of each cached [B,H,L,L] pair-bias buffer refreshed in place once per fold, so the 12 blocks'
               SDPA reads fp32 directly instead of casting bf16->fp32 at every step (the cast's values ARE the twin's values). EXACT.
  kabsch       "torch": stock torch.linalg.svd (cusolver + its error-check host sync) runs EAGERLY between graph replays — exact tier;
               "device" (`kd`): a sync-free 3x3 SVD (Jacobi, fp64 in registers, one program per batch element) inside the graph => the whole
               roll-out is enqueued without a host sync — fast tier (R agrees with cusolver to ~1e-6; end-to-end CA-RMSD <= 0.006 A).
FUSED-STEP levers (diffusion_module.forward -> DiffusionConditioning s-path + the 12-block token DiffusionTransformer) — registry name `dit`:
  dit          per block: 4 GEMMs (q|k|v|g packed [768->3072], out, swish a|b packed [768->3072], lin_out) + 1 flash-attention kernel reading the
               per-fold pair bias in place (bf16, key mask folded in by upstream's fused pair-bias kernel; q-bias prologue, sigmoid(g) epilogue,
               head dim 48 as 32+16 slabs, base-2 online softmax with fp32 statistics, static per-L tile table) + 2 fused [sigmoid(gate)*y +
               residual -> next AdaLN] kernels + SwiGLU; all 12 blocks' AdaLN gate|shift and output gates from 2 GEMMs per step on LN(s) / s
               (s_scale folded into the weights); the two s-conditioning transitions packed the same way; per-fold hoists (s_proj(LN(s_inputs)),
               the 12 bias buffers) in static per-(B,L) buffers refreshed in place. Scope: SCOPE (num_diffusion_samples=1, batch 1); other calls
               run the previous forward by name, counted. Precision sub-flags, each independent:
                 gemm / cond  "fp32" cuBLAS fp32 | "tf32" cuBLAS TF32 scoped to these calls | "bf16" bf16 tensor-core inputs, fp32 accumulate, fp32
                              out wherever the value meets the residual stream / LN / gates | "tf32x3" / "bf16x3" fp32-faithful split (Triton)
                 attn_precision  "ieee" | "tf32" | "tf32x3" | "bf16" (dots; softmax statistics and accumulation fp32 in every mode); attn="sdpa"
                              = torch SDPA with the fp32 bias (stock kernel class)
                 fused_ew     Triton AdaLN / gate / SwiGLU / LN kernels (fp32 math) instead of the torch elementwise sequence
Judgement / lines: scope_note() (plan time), applied_report() (after the run, from STATS — like ef2_mk_sampler's), memory_report(model) (static
bytes per shape), describe(), stats(). No environment variable is read here; no graph is ever replayed concurrently (one stream, one pool).
"""
import collections, math, time, types
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

STATS = collections.Counter()
# DECLARED SCOPE (N4): the roll-out and the fused step engage for sample() calls with num_diffusion_samples == 1 and batch 1 (the kit's `pred`
# folds one design x one sample per call), no denoising early exit, no atom-repr request, inference cache on, CUDA tensors. Any other call runs
# the previously installed chain / forward by name and is COUNTED (STATS rollout_fallback_<reason>, dit_fallback_<reason>) so the run's verdict
# can name it (applied_report()).
SCOPE = dict(num_diffusion_samples=1, batch=1, denoising_early_exit=None, return_atom_repr=False, use_inference_cache=True, device="cuda")   # the ROLL-OUT's (ro)
SCOPE_STEP = dict(SCOPE, num_diffusion_samples="N", batch="N")     # the fused step (dit) and the device Kabsch (kd) serve any sample count of one input —
                                                                   # inside the roll-out at S == 1, inside the step-graph / stock sampler (where ro steps aside) at S > 1
VERSION = "dit0.4"


def lever_scope(name):
    """the scope record of one of the kit's lever names: ro -> SCOPE (S == 1, one input); kd / dit -> SCOPE_STEP (any S of one input)."""
    return SCOPE if name == "ro" else SCOPE_STEP
_CFG = {}
_ENABLED = []

# =====================================================================================================================
# Triton kernels
# =====================================================================================================================
if _HAS_TRITON:

    @triton.jit
    def _adaln_kernel(X, CS, CH, BCS, OUT, D: tl.constexpr, BD: tl.constexpr, eps, sx, scs, sch, so, OUT_BF16: tl.constexpr):
        """OUT[m,:] = sigmoid(CS[m,:] + BCS) * LN(X[m,:]) + CH[m,:]   (LN without affine, two-pass mean/var in fp32; one program per row)"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * sx + offs, mask=msk, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / D
        xc = tl.where(msk, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        cs = tl.load(CS + m * scs + offs, mask=msk, other=0.0).to(tl.float32) + tl.load(BCS + offs, mask=msk, other=0.0).to(tl.float32)
        ch = tl.load(CH + m * sch + offs, mask=msk, other=0.0).to(tl.float32)
        y = tl.sigmoid(cs) * (xc * rstd) + ch
        if OUT_BF16:
            tl.store(OUT + m * so + offs, y.to(tl.bfloat16), mask=msk)
        else:
            tl.store(OUT + m * so + offs, y, mask=msk)

    @triton.jit
    def _gate_res_adaln_kernel(X, Y, OG, BOG, XOUT, CS, CH, BCS, AOUT, D: tl.constexpr, BD: tl.constexpr, eps,
                               sx, sy, sog, sxo, scs, sch, sao, HAS_ADALN: tl.constexpr, OUT_BF16: tl.constexpr):
        """XOUT[m,:] = X[m,:] + sigmoid(OG[m,:] + BOG) * Y[m,:]  (fp32 residual stream);
           if HAS_ADALN: AOUT[m,:] = sigmoid(CS[m,:]+BCS) * LN(XOUT[m,:]) + CH[m,:]  (the next AdaLN, fused)"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * sx + offs, mask=msk, other=0.0).to(tl.float32)
        y = tl.load(Y + m * sy + offs, mask=msk, other=0.0).to(tl.float32)
        og = tl.load(OG + m * sog + offs, mask=msk, other=0.0).to(tl.float32) + tl.load(BOG + offs, mask=msk, other=0.0).to(tl.float32)
        xn = x + tl.sigmoid(og) * y
        tl.store(XOUT + m * sxo + offs, xn, mask=msk)
        if HAS_ADALN:
            mean = tl.sum(tl.where(msk, xn, 0.0), axis=0) / D
            xc = tl.where(msk, xn - mean, 0.0)
            var = tl.sum(xc * xc, axis=0) / D
            rstd = 1.0 / tl.sqrt(var + eps)
            cs = tl.load(CS + m * scs + offs, mask=msk, other=0.0).to(tl.float32) + tl.load(BCS + offs, mask=msk, other=0.0).to(tl.float32)
            ch = tl.load(CH + m * sch + offs, mask=msk, other=0.0).to(tl.float32)
            a = tl.sigmoid(cs) * (xc * rstd) + ch
            if OUT_BF16:
                tl.store(AOUT + m * sao + offs, a.to(tl.bfloat16), mask=msk)
            else:
                tl.store(AOUT + m * sao + offs, a, mask=msk)

    @triton.jit
    def _ln_kernel(X, W, B, OUT, D: tl.constexpr, BD: tl.constexpr, eps, sx, so, HAS_W: tl.constexpr, HAS_B: tl.constexpr, OUT_BF16: tl.constexpr):
        """OUT = LN(X) [* W] [+ B]  per row (fp32 math)"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * sx + offs, mask=msk, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / D
        xc = tl.where(msk, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / D
        y = xc * (1.0 / tl.sqrt(var + eps))
        if HAS_W:
            y = y * tl.load(W + offs, mask=msk, other=0.0).to(tl.float32)
        if HAS_B:
            y = y + tl.load(B + offs, mask=msk, other=0.0).to(tl.float32)
        if OUT_BF16:
            tl.store(OUT + m * so + offs, y.to(tl.bfloat16), mask=msk)
        else:
            tl.store(OUT + m * so + offs, y, mask=msk)

    @triton.jit
    def _swiglu_kernel(SW, OUT, HID: tl.constexpr, BH: tl.constexpr, ssw, so, OUT_BF16: tl.constexpr):
        """OUT[m,:] = silu(SW[m,:HID]) * SW[m,HID:2HID]  (fp32 math)"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BH)
        msk = offs < HID
        a = tl.load(SW + m * ssw + offs, mask=msk, other=0.0).to(tl.float32)
        b = tl.load(SW + m * ssw + HID + offs, mask=msk, other=0.0).to(tl.float32)
        h = a * tl.sigmoid(a) * b
        if OUT_BF16:
            tl.store(OUT + m * so + offs, h.to(tl.bfloat16), mask=msk)
        else:
            tl.store(OUT + m * so + offs, h, mask=msk)

    @triton.jit
    def _res_add_kernel(X, Y, OUT, D: tl.constexpr, BD: tl.constexpr, sx, sy, so):
        """OUT = X + Y (fp32) — the conditioning transitions' residual"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * sx + offs, mask=msk, other=0.0).to(tl.float32)
        y = tl.load(Y + m * sy + offs, mask=msk, other=0.0).to(tl.float32)
        tl.store(OUT + m * so + offs, x + y, mask=msk)

    @triton.jit
    def _fb_dot(a, b, acc, PREC: tl.constexpr):
        if PREC == 3:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        elif PREC == 2:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="tf32x3")
        elif PREC == 1:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="tf32")
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")

    @triton.jit
    def _fb_tile(QKVG, BIAS, k0, L, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                 D: tl.constexpr, HD: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr, PREC: tl.constexpr, MASKED: tl.constexpr):
        """one key tile [k0, k0+BKV): scores from the two head-dim slabs (32 + 16 = 48, no padding), + bias (pre-scaled by log2 e), online softmax
        in base 2, P@V accumulated in fp32. MASKED only for the ragged last tile."""
        LOG2E: tl.constexpr = 1.4426950408889634
        offs_k = k0 + tl.arange(0, BKV)
        if MASKED:
            k_valid = offs_k < L
            k0t = tl.load(QKVG + offs_k[None, :] * s_row + (D + h * HD + offs_d0)[:, None], mask=k_valid[None, :], other=0.0)
            k1t = tl.load(QKVG + offs_k[None, :] * s_row + (D + h * HD + offs_d1)[:, None], mask=k_valid[None, :], other=0.0)
            bias = tl.load(BIAS + offs_q[:, None] * s_bq + offs_k[None, :], mask=q_mask[:, None] & k_valid[None, :], other=0.0).to(tl.float32)
        else:
            k0t = tl.load(QKVG + offs_k[None, :] * s_row + (D + h * HD + offs_d0)[:, None])
            k1t = tl.load(QKVG + offs_k[None, :] * s_row + (D + h * HD + offs_d1)[:, None])
            bias = tl.load(BIAS + offs_q[:, None] * s_bq + offs_k[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        s = tl.zeros([BQ, BKV], dtype=tl.float32)
        s = _fb_dot(q0, k0t, s, PREC)
        s = _fb_dot(q1, k1t, s, PREC)
        s = s + bias * LOG2E
        if MASKED:
            s = tl.where(k_valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        if MASKED:
            p = tl.where(k_valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc0 = acc0 * alpha[:, None]; acc1 = acc1 * alpha[:, None]
        if MASKED:
            v0 = tl.load(QKVG + offs_k[:, None] * s_row + (2 * D + h * HD + offs_d0)[None, :], mask=k_valid[:, None], other=0.0)
            v1 = tl.load(QKVG + offs_k[:, None] * s_row + (2 * D + h * HD + offs_d1)[None, :], mask=k_valid[:, None], other=0.0)
        else:
            v0 = tl.load(QKVG + offs_k[:, None] * s_row + (2 * D + h * HD + offs_d0)[None, :])
            v1 = tl.load(QKVG + offs_k[:, None] * s_row + (2 * D + h * HD + offs_d1)[None, :])
        if PREC == 3:
            p = p.to(tl.bfloat16)
        acc0 = _fb_dot(p, v0, acc0, PREC)
        acc1 = _fb_dot(p, v1, acc1, PREC)
        return m_new, l_i, acc0, acc1

    @triton.jit
    def _flash_bias_kernel(QKVG, QB, BIAS, OUT, L, s_row, s_bh, s_bq, s_out, scale,
                           H: tl.constexpr, HD: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr, OUT_BF16: tl.constexpr,
                           BQ: tl.constexpr, BKV: tl.constexpr):
        """flash attention with pair bias, one batch element: rows of QKVG = [q_raw | k | v | g_raw] (H heads x HD=48 each), q bias QB [D], bias
        [H, L, L] (bf16 or fp32, read in place; the producer folded the key mask in); OUT[:, h*HD:(h+1)*HD] = softmax((q_raw+qb) k^T * scale + bias) v
        * sigmoid(g_raw). Head dim handled as 32 + 16 (no padding); base-2 online softmax with fp32 statistics and fp32 accumulation in every PREC
        (0 ieee fp32 dots, 1 tf32, 2 tf32x3, 3 bf16 tensor cores). grid = (cdiv(L, BQ), H)."""
        LOG2E: tl.constexpr = 1.4426950408889634
        pid_q = tl.program_id(0).to(tl.int64); h = tl.program_id(1).to(tl.int64)
        offs_q = pid_q * BQ + tl.arange(0, BQ)
        offs_d0 = tl.arange(0, 32); offs_d1 = 32 + tl.arange(0, 16)
        q_mask = offs_q < L
        q0 = tl.load(QKVG + offs_q[:, None] * s_row + (h * HD + offs_d0)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        q1 = tl.load(QKVG + offs_q[:, None] * s_row + (h * HD + offs_d1)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        q0 = (q0 + tl.load(QB + h * HD + offs_d0).to(tl.float32)[None, :]) * (scale * LOG2E)
        q1 = (q1 + tl.load(QB + h * HD + offs_d1).to(tl.float32)[None, :]) * (scale * LOG2E)
        if PREC == 3:
            q0 = q0.to(tl.bfloat16); q1 = q1.to(tl.bfloat16)
        m_i = tl.full([BQ], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BQ], dtype=tl.float32)
        acc0 = tl.zeros([BQ, 32], dtype=tl.float32); acc1 = tl.zeros([BQ, 16], dtype=tl.float32)
        BIAS_h = BIAS + h * s_bh
        L_full = (L // BKV) * BKV
        for k0 in range(0, L_full, BKV):
            m_i, l_i, acc0, acc1 = _fb_tile(QKVG, BIAS_h, k0, L, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                                            D, HD, BQ, BKV, PREC, False)
        if L_full < L:
            m_i, l_i, acc0, acc1 = _fb_tile(QKVG, BIAS_h, L_full, L, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                                            D, HD, BQ, BKV, PREC, True)
        g0 = tl.load(QKVG + offs_q[:, None] * s_row + (3 * D + h * HD + offs_d0)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        g1 = tl.load(QKVG + offs_q[:, None] * s_row + (3 * D + h * HD + offs_d1)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        inv = 1.0 / l_i
        o0 = acc0 * inv[:, None] * tl.sigmoid(g0); o1 = acc1 * inv[:, None] * tl.sigmoid(g1)
        p0 = OUT + offs_q[:, None] * s_out + (h * HD + offs_d0)[None, :]; p1 = OUT + offs_q[:, None] * s_out + (h * HD + offs_d1)[None, :]
        if OUT_BF16:
            tl.store(p0, o0.to(tl.bfloat16), mask=q_mask[:, None]); tl.store(p1, o1.to(tl.bfloat16), mask=q_mask[:, None])
        else:
            tl.store(p0, o0, mask=q_mask[:, None]); tl.store(p1, o1, mask=q_mask[:, None])

    @triton.jit
    def _mm_split_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                         PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """C = A @ B on tensor cores with fp32-faithful split emulation: PREC 2 = tf32x3, 4 = bf16x3 (hi/lo split, 3 MMAs); fp32 in, fp32 out."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM); num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
        b_ptrs = B + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            am = (rm[:, None] < M) & ((k0 + rk)[None, :] < K)
            bm = ((k0 + rk)[:, None] < K) & (rn[None, :] < N)
            a = tl.load(a_ptrs, mask=am, other=0.0).to(tl.float32)
            b = tl.load(b_ptrs, mask=bm, other=0.0).to(tl.float32)
            if PREC == 2:
                acc = tl.dot(a, b, acc, input_precision="tf32x3")
            else:
                ah = a.to(tl.bfloat16); al = (a - ah.to(tl.float32)).to(tl.bfloat16)
                bh = b.to(tl.bfloat16); bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(ah, bl, acc)
                acc = tl.dot(al, bh, acc)
                acc = tl.dot(ah, bh, acc)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _jacobi_rot(app, aqq, apq):
        """Jacobi rotation (c, s) annihilating apq of the symmetric 2x2 [[app, apq], [apq, aqq]] (fp64 scalars)."""
        small = tl.abs(apq) < 1e-300
        theta = (aqq - app) / (2.0 * tl.where(small, 1.0, apq))
        t = tl.where(theta >= 0, 1.0, -1.0) / (tl.abs(theta) + tl.sqrt(theta * theta + 1.0))
        c = 1.0 / tl.sqrt(t * t + 1.0)
        s = t * c
        c = tl.where(small, 1.0, c); s = tl.where(small, 0.0, s)
        return c, s

    @triton.jit
    def _kabsch3_kernel(Hp, Rp, DETp, NSWEEP: tl.constexpr):
        """per batch element b: H = Hp[b] (3x3 fp32, row-major) = U S Vh (SVD, S descending); R = U diag(1, 1, det(U Vh)) Vh -> Rp[b]; det -> DETp[b].
        Jacobi eigen-decomposition of H^T H in fp64, U columns from H v_i / s_i re-orthonormalised (u2 = +-(u0 x u1)). One program per b; scalar code."""
        b = tl.program_id(0).to(tl.int64)
        base = Hp + b * 9
        h00 = tl.load(base + 0).to(tl.float64); h01 = tl.load(base + 1).to(tl.float64); h02 = tl.load(base + 2).to(tl.float64)
        h10 = tl.load(base + 3).to(tl.float64); h11 = tl.load(base + 4).to(tl.float64); h12 = tl.load(base + 5).to(tl.float64)
        h20 = tl.load(base + 6).to(tl.float64); h21 = tl.load(base + 7).to(tl.float64); h22 = tl.load(base + 8).to(tl.float64)
        # S = H^T H (symmetric)
        a00 = h00 * h00 + h10 * h10 + h20 * h20
        a01 = h00 * h01 + h10 * h11 + h20 * h21
        a02 = h00 * h02 + h10 * h12 + h20 * h22
        a11 = h01 * h01 + h11 * h11 + h21 * h21
        a12 = h01 * h02 + h11 * h12 + h21 * h22
        a22 = h02 * h02 + h12 * h12 + h22 * h22
        v00 = a00 * 0.0 + 1.0; v01 = a00 * 0.0; v02 = a00 * 0.0
        v10 = a00 * 0.0; v11 = a00 * 0.0 + 1.0; v12 = a00 * 0.0
        v20 = a00 * 0.0; v21 = a00 * 0.0; v22 = a00 * 0.0 + 1.0
        for _ in tl.static_range(NSWEEP):
            # pair (0,1)
            c, s = _jacobi_rot(a00, a11, a01)
            n00 = c * c * a00 - 2.0 * s * c * a01 + s * s * a11
            n11 = s * s * a00 + 2.0 * s * c * a01 + c * c * a11
            n02 = c * a02 - s * a12
            n12 = s * a02 + c * a12
            a00 = n00; a11 = n11; a01 = a00 * 0.0; a02 = n02; a12 = n12
            t0 = c * v00 - s * v01; t1 = s * v00 + c * v01; v00 = t0; v01 = t1
            t0 = c * v10 - s * v11; t1 = s * v10 + c * v11; v10 = t0; v11 = t1
            t0 = c * v20 - s * v21; t1 = s * v20 + c * v21; v20 = t0; v21 = t1
            # pair (0,2)
            c, s = _jacobi_rot(a00, a22, a02)
            n00 = c * c * a00 - 2.0 * s * c * a02 + s * s * a22
            n22 = s * s * a00 + 2.0 * s * c * a02 + c * c * a22
            n01 = c * a01 - s * a12
            n21 = s * a01 + c * a12
            a00 = n00; a22 = n22; a02 = a00 * 0.0; a01 = n01; a12 = n21
            t0 = c * v00 - s * v02; t1 = s * v00 + c * v02; v00 = t0; v02 = t1
            t0 = c * v10 - s * v12; t1 = s * v10 + c * v12; v10 = t0; v12 = t1
            t0 = c * v20 - s * v22; t1 = s * v20 + c * v22; v20 = t0; v22 = t1
            # pair (1,2)
            c, s = _jacobi_rot(a11, a22, a12)
            n11 = c * c * a11 - 2.0 * s * c * a12 + s * s * a22
            n22 = s * s * a11 + 2.0 * s * c * a12 + c * c * a22
            n01 = c * a01 - s * a02
            n02 = s * a01 + c * a02
            a11 = n11; a22 = n22; a12 = a11 * 0.0; a01 = n01; a02 = n02
            t0 = c * v01 - s * v02; t1 = s * v01 + c * v02; v01 = t0; v02 = t1
            t0 = c * v11 - s * v12; t1 = s * v11 + c * v12; v11 = t0; v12 = t1
            t0 = c * v21 - s * v22; t1 = s * v21 + c * v22; v21 = t0; v22 = t1
        # eigenvalues a00, a11, a22 with eigenvector columns (v0*, v1*, v2*) -> sort descending (3-element network)
        l0 = a00; l1 = a11; l2 = a22
        sw = l1 > l0
        l0, l1 = tl.where(sw, l1, l0), tl.where(sw, l0, l1)
        x0, x1 = tl.where(sw, v01, v00), tl.where(sw, v00, v01); v00 = x0; v01 = x1
        x0, x1 = tl.where(sw, v11, v10), tl.where(sw, v10, v11); v10 = x0; v11 = x1
        x0, x1 = tl.where(sw, v21, v20), tl.where(sw, v20, v21); v20 = x0; v21 = x1
        sw = l2 > l1
        l1, l2 = tl.where(sw, l2, l1), tl.where(sw, l1, l2)
        x0, x1 = tl.where(sw, v02, v01), tl.where(sw, v01, v02); v01 = x0; v02 = x1
        x0, x1 = tl.where(sw, v12, v11), tl.where(sw, v11, v12); v11 = x0; v12 = x1
        x0, x1 = tl.where(sw, v22, v21), tl.where(sw, v21, v22); v21 = x0; v22 = x1
        sw = l1 > l0
        l0, l1 = tl.where(sw, l1, l0), tl.where(sw, l0, l1)
        x0, x1 = tl.where(sw, v01, v00), tl.where(sw, v00, v01); v00 = x0; v01 = x1
        x0, x1 = tl.where(sw, v11, v10), tl.where(sw, v10, v11); v10 = x0; v11 = x1
        x0, x1 = tl.where(sw, v21, v20), tl.where(sw, v20, v21); v20 = x0; v21 = x1
        # U columns: u_i = H v_i normalised (i = 0, 1), Gram-Schmidt u1 vs u0, u2 = +-(u0 x u1) oriented along H v2
        u00 = h00 * v00 + h01 * v10 + h02 * v20; u10 = h10 * v00 + h11 * v10 + h12 * v20; u20 = h20 * v00 + h21 * v10 + h22 * v20
        nrm = tl.sqrt(u00 * u00 + u10 * u10 + u20 * u20)
        ok0 = nrm > 1e-200
        u00 = tl.where(ok0, u00 / nrm, 1.0); u10 = tl.where(ok0, u10 / nrm, 0.0); u20 = tl.where(ok0, u20 / nrm, 0.0)
        u01 = h00 * v01 + h01 * v11 + h02 * v21; u11 = h10 * v01 + h11 * v11 + h12 * v21; u21 = h20 * v01 + h21 * v11 + h22 * v21
        dt = u00 * u01 + u10 * u11 + u20 * u21
        u01 = u01 - dt * u00; u11 = u11 - dt * u10; u21 = u21 - dt * u20
        nrm = tl.sqrt(u01 * u01 + u11 * u11 + u21 * u21)
        ok1 = nrm > 1e-200
        # fallback for a degenerate second direction: any unit vector orthogonal to u0
        f01 = -u10; f11 = u00; f21 = u00 * 0.0
        fn = tl.sqrt(f01 * f01 + f11 * f11)
        okf = fn > 1e-100
        f01 = tl.where(okf, f01 / tl.where(okf, fn, 1.0), 0.0); f11 = tl.where(okf, f11 / tl.where(okf, fn, 1.0), 0.0); f21 = tl.where(okf, f21, 1.0)
        u01 = tl.where(ok1, u01 / tl.where(ok1, nrm, 1.0), f01); u11 = tl.where(ok1, u11 / tl.where(ok1, nrm, 1.0), f11); u21 = tl.where(ok1, u21 / tl.where(ok1, nrm, 1.0), f21)
        c02 = u10 * u21 - u20 * u11; c12 = u20 * u01 - u00 * u21; c22 = u00 * u11 - u10 * u01
        w02 = h00 * v02 + h01 * v12 + h02 * v22; w12 = h10 * v02 + h11 * v12 + h12 * v22; w22 = h20 * v02 + h21 * v12 + h22 * v22
        sg = tl.where(c02 * w02 + c12 * w12 + c22 * w22 < 0, -1.0, 1.0)
        u02 = sg * c02; u12 = sg * c12; u22 = sg * c22
        # det(U @ Vh) = det(U) * det(V);  R = U diag(1,1,det) V^T
        detU = u00 * (u11 * u22 - u12 * u21) - u01 * (u10 * u22 - u12 * u20) + u02 * (u10 * u21 - u11 * u20)
        detV = v00 * (v11 * v22 - v12 * v21) - v01 * (v10 * v22 - v12 * v20) + v02 * (v10 * v21 - v11 * v20)
        det = detU * detV
        u02 = u02 * det; u12 = u12 * det; u22 = u22 * det
        r00 = u00 * v00 + u01 * v01 + u02 * v02; r01 = u00 * v10 + u01 * v11 + u02 * v12; r02 = u00 * v20 + u01 * v21 + u02 * v22
        r10 = u10 * v00 + u11 * v01 + u12 * v02; r11 = u10 * v10 + u11 * v11 + u12 * v12; r12 = u10 * v20 + u11 * v21 + u12 * v22
        r20 = u20 * v00 + u21 * v01 + u22 * v02; r21 = u20 * v10 + u21 * v11 + u22 * v12; r22 = u20 * v20 + u21 * v21 + u22 * v22
        ob = Rp + b * 9
        tl.store(ob + 0, r00.to(tl.float32)); tl.store(ob + 1, r01.to(tl.float32)); tl.store(ob + 2, r02.to(tl.float32))
        tl.store(ob + 3, r10.to(tl.float32)); tl.store(ob + 4, r11.to(tl.float32)); tl.store(ob + 5, r12.to(tl.float32))
        tl.store(ob + 6, r20.to(tl.float32)); tl.store(ob + 7, r21.to(tl.float32)); tl.store(ob + 8, r22.to(tl.float32))
        tl.store(DETp + b, det.to(tl.float32))


def _nw(D):
    return 4 if D <= 1024 else 8


def kabsch_rotation_device(H32):
    """H32 [B,3,3] fp32 -> (R [B,3,3], det [B]) with R = U diag(1,1,det(U Vh)) Vh of H's SVD — the rotation stock's _weighted_rigid_align builds,
    computed by one sync-free kernel (no cusolver, no error-check read-back)."""
    B = H32.shape[0]
    Hc = H32.contiguous()
    R = torch.empty_like(Hc); det = torch.empty(B, device=Hc.device, dtype=torch.float32)
    _kabsch3_kernel[(B,)](Hc, R, det, NSWEEP=8, num_warps=1)
    return R, det


# =====================================================================================================================
# GEMM dispatch (precision lever) and packed weights
# =====================================================================================================================
GEMM_PRECISIONS = ("fp32", "tf32", "tf32x3", "bf16x3", "bf16")
ATTN_PRECISIONS = ("ieee", "tf32", "tf32x3", "bf16")
_SPLIT_CFG = dict(BM=64, BN=128, BK=32, num_warps=4, num_stages=4)


class _W:
    """one packed weight: t32 = W^T contiguous [K, N] fp32 (concatenation of the stock Linear weights along N); t16 its bf16 copy (made lazily)."""
    __slots__ = ("t32", "t16", "K", "N")

    def __init__(self, w_out_in):                     # [N, K] like nn.Linear.weight (possibly a torch.cat of several)
        self.t32 = w_out_in.detach().t().contiguous().float()
        self.t16 = None
        self.K, self.N = self.t32.shape

    def bf16(self):
        if self.t16 is None:
            self.t16 = self.t32.to(torch.bfloat16)
        return self.t16


def _in_dtype(prec):
    return torch.bfloat16 if prec == "bf16" else torch.float32


def _mm(x, w, prec, out_dtype=torch.float32):
    """x [M, K] @ W^T -> [M, N] in the arithmetic `prec`. bf16: x must already be bf16 (its producer emitted it); out_dtype fp32 or bf16.
    Every other precision takes and returns fp32."""
    STATS["mm_" + prec] += 1
    if prec == "fp32":
        return torch.mm(x, w.t32)
    if prec == "tf32":
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return torch.mm(x, w.t32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    if prec == "bf16":
        if out_dtype == torch.bfloat16:
            return torch.mm(x, w.bf16())
        return torch.mm(x, w.bf16(), out_dtype=torch.float32)
    # split emulation on tensor cores (fp32 in / fp32 out)
    M, K = x.shape
    c = torch.empty(M, w.N, device=x.device, dtype=torch.float32)
    cfg = _SPLIT_CFG
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(w.N, cfg["BN"]),)
    _mm_split_kernel[grid](x, w.t32, c, M, w.N, K, x.stride(0), x.stride(1), w.t32.stride(0), w.t32.stride(1), c.stride(0), c.stride(1),
                           PREC=(2 if prec == "tf32x3" else 4), BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP_M=8,
                           num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return c


# ---- elementwise building blocks: Triton (fused_ew) or torch reference of the same dataflow
def _ln(x, w, b, eps, out_dtype, fused):
    M, D = x.shape
    if fused:
        out = torch.empty(M, D, device=x.device, dtype=out_dtype)
        _ln_kernel[(M,)](x, w if w is not None else x, b if b is not None else x, out, D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), so=out.stride(0),
                         HAS_W=w is not None, HAS_B=b is not None, OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_nw(D))
        return out
    return F.layer_norm(x.float(), (D,), w, b, eps).to(out_dtype)


def _adaln(x, cs, ch, bcs, eps, out_dtype, fused):
    M, D = x.shape
    if fused:
        out = torch.empty(M, D, device=x.device, dtype=out_dtype)
        _adaln_kernel[(M,)](x, cs, ch, bcs, out, D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), scs=cs.stride(0), sch=ch.stride(0), so=out.stride(0),
                            OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_nw(D))
        return out
    a_norm = F.layer_norm(x, (D,), None, None, eps)
    return (torch.sigmoid(cs + bcs) * a_norm + ch).to(out_dtype)


def _gate_res_adaln(x, y, og, bog, cs, ch, bcs, eps, out_dtype, fused):
    """returns (x_new fp32, a_next in out_dtype or None): x_new = x + sigmoid(og+bog)*y; a_next = AdaLN(x_new) with (cs, ch) when cs is given."""
    M, D = x.shape
    has = cs is not None
    if fused:
        xo = torch.empty(M, D, device=x.device, dtype=torch.float32)
        ao = torch.empty(M, D, device=x.device, dtype=out_dtype) if has else xo
        _gate_res_adaln_kernel[(M,)](x, y, og, bog, xo, cs if has else x, ch if has else x, bcs if has else bog, ao, D=D, BD=triton.next_power_of_2(D), eps=eps,
                                     sx=x.stride(0), sy=y.stride(0), sog=og.stride(0), sxo=xo.stride(0), scs=(cs.stride(0) if has else 0), sch=(ch.stride(0) if has else 0),
                                     sao=ao.stride(0), HAS_ADALN=has, OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_nw(D))
        return xo, (ao if has else None)
    xo = x + torch.sigmoid(og + bog) * y.float()
    if not has:
        return xo, None
    return xo, _adaln(xo, cs, ch, bcs, eps, out_dtype, False)


def _swiglu(sw, hid, out_dtype, fused):
    M = sw.shape[0]
    if fused:
        out = torch.empty(M, hid, device=sw.device, dtype=out_dtype)
        _swiglu_kernel[(M,)](sw, out, HID=hid, BH=triton.next_power_of_2(hid), ssw=sw.stride(0), so=out.stride(0), OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_nw(hid))
        return out
    a, b = sw.float().chunk(2, dim=-1)
    return (F.silu(a) * b).to(out_dtype)


def _res_add(x, y, fused):
    M, D = x.shape
    if fused:
        out = torch.empty(M, D, device=x.device, dtype=torch.float32)
        _res_add_kernel[(M,)](x, y, out, D=D, BD=triton.next_power_of_2(D), sx=x.stride(0), sy=y.stride(0), so=out.stride(0), num_warps=_nw(D))
        return out
    return x + y.float()


def _flash_cfg(L, prec):
    """static (BQ, BKV, num_warps, num_stages) per token count: a fixed table (no run-time autotuning: no per-shape tuning cost on a fold's clock, and
    the same tile order — the same numerics — for a given L in every process): small tiles for occupancy below ~600 tokens (16 heads x few
    query tiles), 64 x 128 above."""
    if prec in ("ieee", "tf32x3"):
        return (64, 32, 4, 2)
    if L <= 512:
        return (32, 64, 2, 2)
    return (64, 128, 4, 1)


def _attention(qkvg, bias, bq, H, HD, scale, prec, out_dtype, kind, key_mask=None):
    """qkvg [L, 4*D] rows = [q_raw | k | v | g_raw] (q_raw lacks the q bias bq); bias [H, L, L] (bf16 or fp32, key mask already folded in by the producer);
    returns ctx [L, D] = softmax((q_raw+bq) k^T * scale + bias) v * sigmoid(g_raw)."""
    L, D4 = qkvg.shape; D = D4 // 4
    if kind == "flash":
        assert HD == 48, "ef2_dit flash kernel is specialised for head_dim 48 (32 + 16 slabs)"
        out = torch.empty(L, D, device=qkvg.device, dtype=out_dtype)
        BQ, BKV, nw, ns = _flash_cfg(L, prec)
        _flash_bias_kernel[(triton.cdiv(L, BQ), H)](qkvg, bq, bias, out, L, qkvg.stride(0), bias.stride(0), bias.stride(1), out.stride(0), float(scale),
                                                      H=H, HD=HD, D=D, PREC={"ieee": 0, "tf32": 1, "tf32x3": 2, "bf16": 3}[prec], OUT_BF16=(out_dtype == torch.bfloat16),
                                                      BQ=BQ, BKV=BKV, num_warps=nw, num_stages=ns)
        STATS["attn_flash_" + prec] += 1
        return out
    # kind == "sdpa": the stock kernel family on the stock layout (fp32 q/k/v as stock; bf16 when prec == 'bf16')
    dt = torch.bfloat16 if prec == "bf16" else torch.float32
    q = (qkvg[:, :D].float() + bq).to(dt).view(L, H, HD).transpose(0, 1).unsqueeze(0)
    k = qkvg[:, D:2 * D].to(dt).view(L, H, HD).transpose(0, 1).unsqueeze(0)
    v = qkvg[:, 2 * D:3 * D].to(dt).view(L, H, HD).transpose(0, 1).unsqueeze(0)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(dt).unsqueeze(0))          # [1, H, L, HD]
    g = torch.sigmoid(qkvg[:, 3 * D:].float())
    ctx = (o[0].transpose(0, 1).reshape(L, D).float() * g).to(out_dtype)
    STATS["attn_sdpa_" + prec] += 1
    return ctx


# =====================================================================================================================
# per-model state: packed weights of the token transformer + conditioning
# =====================================================================================================================
class _DitState:
    def __init__(self, dm, cfg):
        self.cfg = dict(cfg)
        tt = dm.token_transformer
        self.nb = len(tt.attn_blocks)
        blk0 = tt.attn_blocks[0]
        self.H, self.HD, self.D = blk0.num_heads, blk0.head_dim, blk0.d_model
        self.scale = blk0.scale
        self.eps = blk0.adaln.eps
        self.hid = tt.transition_blocks[0].lin_out.in_features
        with torch.no_grad():
            self.w_qkvg, self.w_o, self.w_sw, self.w_lo, self.bq = [], [], [], [], []
            wg_cols, wog_cols, self.bcs_a, self.bcs_t, self.bog_a, self.bog_t = [], [], [], [], [], []
            for b in range(self.nb):
                A = tt.attn_blocks[b]; T = tt.transition_blocks[b]
                self.w_qkvg.append(_W(torch.cat([A.q_proj.weight, A.kv_proj.weight, A.g_proj.weight], 0)))      # [4D, D]
                self.bq.append(A.q_proj.bias.detach().float().contiguous())
                self.w_o.append(_W(A.out_proj.weight)); self.w_sw.append(_W(T.lin_swish.weight)); self.w_lo.append(_W(T.lin_out.weight))
                # AdaLN(a, s) = sigmoid(s_gate(LN(s)*s_scale)) * LN(a) + s_shift(LN(s)*s_scale): fold s_scale into the input dim of the two linears
                wg_cols += [A.adaln.s_gate.weight * A.adaln.s_scale[None, :], A.adaln.s_shift.weight * A.adaln.s_scale[None, :],
                            T.adaln.s_gate.weight * T.adaln.s_scale[None, :], T.adaln.s_shift.weight * T.adaln.s_scale[None, :]]
                self.bcs_a.append(A.adaln.s_gate.bias.detach().float().contiguous()); self.bcs_t.append(T.adaln.s_gate.bias.detach().float().contiguous())
                wog_cols += [A.out_gate.weight, T.output_gate.weight]
                self.bog_a.append(A.out_gate.bias.detach().float().contiguous()); self.bog_t.append(T.output_gate.bias.detach().float().contiguous())
            self.w_cs_all = _W(torch.cat(wg_cols, 0))          # [nb*4D, D]  -> per step CS_ALL = LN(s) @ this^T  [L, nb*4D]
            self.w_og_all = _W(torch.cat(wog_cols, 0))         # [nb*2D, D]  -> OG_ALL = s @ this^T           [L, nb*2D]
            cd = dm.conditioning
            self.s_trans = []
            for tr in cd.s_transitions:                        # TransitionLayer: norm(affine) -> a_proj|b_proj -> silu(a)*b -> out_proj
                self.s_trans.append(dict(ln_w=tr.norm.weight.detach().float().contiguous(), ln_b=tr.norm.bias.detach().float().contiguous(), eps=tr.norm.eps,
                                         w_ab=_W(torch.cat([tr.a_proj.weight, tr.b_proj.weight], 0)), w_out=_W(tr.out_proj.weight), hid=tr.out_proj.in_features))
            self.s_step_ln = (dm.s_step_norm.weight.detach().float().contiguous(), dm.s_step_norm.bias.detach().float().contiguous(), dm.s_step_norm.eps)
            self.w_s_tok = _W(dm.s_to_token.weight)
            self.tok_ln = (dm.token_norm.weight.detach().float().contiguous(), dm.token_norm.bias.detach().float().contiguous(), dm.token_norm.eps)
        # per-fold hoists live in STATIC per-shape buffers (a captured roll-out graph reads them by address): key (B, L) -> entry; refreshed in
        # place at each fold's eager step 0; a shape's entry is dropped only together with every graph (clear()).
        self.hoist = collections.OrderedDict()
        self.cur = None                 # the entry of the fold in flight: dict(s0=[B,L,D] fp32, pb=[nb x [B,H,L,L]], filled=bool)

    @property
    def pb(self):
        return self.cur["pb"]

    @property
    def s0(self):
        return self.cur["s0"] if (self.cur is not None and self.cur["filled_s0"]) else None


def _hoist_entry(st, B, L, device):
    """select (allocating on first use of a shape — never during capture) the static hoist entry of shape (B, L) as the fold in flight."""
    key = (B, L)
    ent = st.hoist.get(key)
    if ent is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ef2_dit: per-fold hoist buffers missing during capture (step 0 must run eagerly first)")
        while len(st.hoist) >= max(1, int(_CFG.get("lru", 2))):
            (oB, oL), _ent = st.hoist.popitem(last=False); STATS["hoist_evict"] += 1
            for m in _ENABLED:                                                  # a captured roll graph of that shape reads the evicted buffers: drop it too
                rl = getattr(m.structure_head, "_dit_rolls", None)
                if rl:
                    for sg in [k for k, r in rl.items() if getattr(r, "L", None) == oL and k[1] == oB]:
                        del rl[sg]; STATS["roll_evict_with_hoist"] += 1
            EO = _CFG.get("_ef2_opt")                                           # ef2_opt's STEP graphs (the sampler where ro steps aside, S > 1) read the hoists too —
            if EO is not None and hasattr(EO, "clear_sampler_graphs") and any(len(getattr(m.structure_head, "_ef2opt_step_graphs", None) or ()) for m in _ENABLED):
                EO.clear_sampler_graphs(reason="dit_hoist_evict"); STATS["step_graphs_dropped_with_hoist"] += 1   # (the fold loop releases the generation before a new shape's step 0, so this is insurance)
        ent = dict(s0=None, pb=[None] * st.nb, filled_s0=False)
        st.hoist[key] = ent; STATS["hoist_new_shape"] += 1
    else:
        st.hoist.move_to_end(key)
    ent["filled_s0"] = False
    st.cur = ent
    return ent


def _store_static(old, new):
    """refresh a static buffer in place (same address for captured graphs); allocate when absent or reshaped (eager step 0 only)."""
    if old is not None and old.shape == new.shape and old.dtype == new.dtype:
        old.copy_(new); return old
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("ef2_dit: static hoist buffer (re)allocation during capture")
    STATS["hoist_alloc"] += 1
    return new.clone() if new.is_contiguous() else new.contiguous()


def _pair_bias_all(st, tt, z, attention_mask):
    """the 12 blocks' pair bias for this fold: upstream's fused LN+projection kernel (routed through ef2_opt's row-blocked wrapper when installed),
    one bf16 cast of z shared by all blocks; written into the fold's static per-shape buffers."""
    from transformers.models.esmfold2 import modeling_esmfold2_common as M
    bsz, L = z.shape[0], z.shape[1]
    kernel_mask = attention_mask if attention_mask is not None else torch.ones(bsz, L, device=z.device, dtype=torch.bool)
    z_bf = z if z.dtype == torch.bfloat16 else z.to(torch.bfloat16)
    want_f32 = st.cfg["attn"] == "sdpa" and st.cfg["attn_precision"] != "bf16"
    ent = st.cur
    for b in range(st.nb):
        A = tt.attn_blocks[b]
        pnb = A.pair_norm.bias if A.pair_norm.bias is not None else torch.zeros_like(A.pair_norm.weight)
        bias = M._fused_pair_bias(z_bf, kernel_mask, A.pair_bias_proj.weight, num_heads=A.num_heads, pair_norm_w=A.pair_norm.weight, pair_norm_b=pnb)   # [B,H,L,L]
        ent["pb"][b] = _store_static(ent["pb"][b], bias.float() if want_f32 else bias)
        STATS["pb_compute"] += 1


def _s_path(st, dm, t, s_inputs, sigma, nds, fused):
    """DiffusionConditioning's s branch for this step: s0 (hoisted per fold) + noise embedding of t, then the two SwiGLU transitions in the `cond` precision."""
    cd = dm.conditioning
    prec = st.cfg["cond"]
    if st.s0 is None:
        s_in = s_inputs if s_inputs.shape[0] == t.shape[0] else s_inputs.repeat_interleave(nds, 0)
        st.cur["s0"] = _store_static(st.cur["s0"], cd.s_proj(cd.s_input_norm(s_in.to(dtype=torch.float32))))   # [B, L, 768] fp32, step-invariant: once per fold
        st.cur["filled_s0"] = True
        STATS["s0_hoist"] += 1
    t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
    n = cd.noise_proj(cd.noise_norm(cd.fourier(t_noise)))                                       # [B, 768] (tiny; stock modules)
    B, L, D = st.s0.shape
    s = _res_add(st.s0.reshape(B * L, D), n.repeat_interleave(L, 0) if B > 1 else n.expand(L, D), fused)      # s0 + n  [B*L, D] fp32
    for tr in st.s_trans:
        x = _ln(s, tr["ln_w"], tr["ln_b"], tr["eps"], _in_dtype(prec), fused)
        ab = _mm(x, tr["w_ab"], prec, out_dtype=_in_dtype(prec))                                # [BL, 2*hid]
        h = _swiglu(ab, tr["hid"], _in_dtype(prec), fused)
        o = _mm(h, tr["w_out"], prec, out_dtype=torch.float32)
        s = _res_add(s, o, fused)
    return s                                                                                      # [B*L, D] fp32


def _token_transformer_dit(st, tt, a, s, fused, B=1):
    """the 12 DiT blocks on a [B*L, D] fp32 (rows: the B = num_diffusion_samples samples of ONE input stacked), s [B*L, D] fp32; returns x [B*L, D]
    fp32 (before token_norm). Every GEMM / LN / gate is row-wise (batch-agnostic); the pair-bias attention runs PER SAMPLE — the B == 1 kernel on
    that sample's L rows against the fold's one [H, L, L] bias (the pair conditioning is the input's, shared by its samples exactly as stock's
    repeat_interleave shares it) — so a sample's arithmetic at B > 1 is the B == 1 arithmetic (B == 1 issues exactly the single-sample statements)."""
    cfg = st.cfg; gp, cp, ap = cfg["gemm"], ("fp32" if cfg["cond"] == "stock" else cfg["cond"]), cfg["attn_precision"]
    R, D = a.shape
    B = max(1, int(B)); L = R // B
    assert L * B == R, (R, B)
    gin = _in_dtype(gp)
    # conditioning of all blocks: 2 GEMMs per step
    sn = _ln(s, None, None, st.eps, _in_dtype(cp), fused)                                        # LN(s) without affine (s_scale folded into the weights)
    cs_all = _mm(sn, st.w_cs_all, cp, out_dtype=torch.float32)                                  # [L, nb*4D]  = [cs_a | ch_a | cs_t | ch_t] per block (pre-sigmoid gates)
    s_in = s if cp != "bf16" else s.to(torch.bfloat16)
    og_all = _mm(s_in, st.w_og_all, cp, out_dtype=torch.float32)                                # [L, nb*2D]  = [og_a | og_t] per block (pre-sigmoid)
    x = a
    xin = _adaln(x, cs_all[:, 0:D], cs_all[:, D:2 * D], st.bcs_a[0], st.eps, gin, fused)
    for b in range(st.nb):
        c0 = b * 4 * D; g0 = b * 2 * D
        qkvg = _mm(xin, st.w_qkvg[b], gp, out_dtype=gin)                                        # [L, 4D]  (q without its bias: added in the attention prologue)
        if B == 1:
            ctx = _attention(qkvg, st.pb[b][0], st.bq[b], st.H, st.HD, st.scale, ap, gin, cfg["attn"])
        else:                                                                                    # per sample: the B == 1 attention on rows [i*L, (i+1)*L) (S launches; S is small)
            ctx = torch.cat([_attention(qkvg[i * L:(i + 1) * L], st.pb[b][0], st.bq[b], st.H, st.HD, st.scale, ap, gin, cfg["attn"]) for i in range(B)], 0)
            STATS["attn_multi_sample_blocks"] += 1
        o = _mm(ctx if ctx.dtype == gin else ctx.to(gin), st.w_o[b], gp, out_dtype=torch.float32)     # [L, D] fp32
        x, xin = _gate_res_adaln(x, o, og_all[:, g0:g0 + D], st.bog_a[b], cs_all[:, c0 + 2 * D:c0 + 3 * D], cs_all[:, c0 + 3 * D:c0 + 4 * D], st.bcs_t[b], st.eps, gin, fused)
        sw = _mm(xin, st.w_sw[b], gp, out_dtype=gin)                                            # [L, 2*hid]
        h = _swiglu(sw, st.hid, gin, fused)
        o = _mm(h, st.w_lo[b], gp, out_dtype=torch.float32)
        if b + 1 < st.nb:
            c1 = (b + 1) * 4 * D
            x, xin = _gate_res_adaln(x, o, og_all[:, g0 + D:g0 + 2 * D], st.bog_t[b], cs_all[:, c1:c1 + D], cs_all[:, c1 + D:c1 + 2 * D], st.bcs_a[b + 1], st.eps, gin, fused)
        else:
            x, _ = _gate_res_adaln(x, o, og_all[:, g0 + D:g0 + 2 * D], st.bog_t[b], None, None, None, st.eps, gin, fused)
    return x


PAIR_PATH_HOOK = "_ef2_pair_path"       # an attribute a memory add-on may set on DiffusionConditioning (class or instance): callable(z_trunk, relative_position_encoding)
                                        # -> z, the conditioning pair path computed with the module's own weights (the XL add-on's x3: row-wise LN(cat) into one buffer,
                                        # one z_proj GEMM of the stock shape, row-chunked transitions). Absent: the stock statements below, unchanged.


def _z_cond(cd, z_trunk, relative_position_encoding):
    """DiffusionConditioning's z branch (once per fold, eager: step 0): the stock statements, or the module's PAIR_PATH_HOOK when an add-on
    installed one (counted: STATS z_cond_hook / z_cond_stock)."""
    hook = getattr(cd, PAIR_PATH_HOOK, None)
    if hook is not None:
        STATS["z_cond_hook"] += 1
        return hook(z_trunk, relative_position_encoding)
    STATS["z_cond_stock"] += 1
    z_rel = relative_position_encoding.to(dtype=torch.float32)
    z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)
    z = cd.z_proj(cd.z_input_norm(z))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for block in cd.z_transitions:
            z = z + block(z)
    return z


def _dm_forward_dit(self, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, s_trunk, z_trunk,
                    relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id, sigma_data=None, token_attention_mask=None,
                    num_diffusion_samples=1, return_token_repr=False, return_atom_repr=False, inference_cache=None):
    st = self._dit
    bsz = x_noisy.shape[0]
    why = None if _CFG.get("dit") else "disabled"
    if why is None:
        why = ("return_atom_repr" if return_atom_repr else "grad_enabled" if torch.is_grad_enabled() else "device" if not x_noisy.is_cuda else
               "batch" if bsz != max(1, int(num_diffusion_samples)) else                                        # the S samples of ONE input (bsz == S; s_trunk / z_trunk are None after step 0 — the cache holds them); a batch of inputs (bsz == n_inputs * S, n_inputs > 1) steps aside by name
               "no_inference_cache" if inference_cache is None else None)
    if why is not None:
        STATS["dit_fallback"] += 1; STATS["dit_fallback_" + why] += 1
        return self._dit_prev_forward(x_noisy=x_noisy, t_hat=t_hat, ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element,
                                      ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid, tok_idx=tok_idx, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                                      relative_position_encoding=relative_position_encoding, asym_id=asym_id, residue_index=residue_index, entity_id=entity_id,
                                      token_index=token_index, sym_id=sym_id, sigma_data=sigma_data, token_attention_mask=token_attention_mask,
                                      num_diffusion_samples=num_diffusion_samples, return_token_repr=return_token_repr, return_atom_repr=return_atom_repr,
                                      inference_cache=inference_cache)
    fused = bool(st.cfg["fused_ew"]) and _HAS_TRITON
    sigma = self.sigma_data if sigma_data is None else float(sigma_data)
    t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(bsz)
    fresh = "z" not in inference_cache
    if fresh:                                                           # step 0 of a fold (fresh inference cache): per-fold hoists are recomputed, eagerly,
        if torch.cuda.is_current_stream_capturing():                    # into the static buffers of this shape
            raise RuntimeError("ef2_dit: step 0 (fresh inference cache) must run eagerly before capture")
        _hoist_entry(st, bsz, int(s_inputs.shape[1]), x_noisy.device)
        STATS["folds_seen"] += 1
    elif st.cur is None:
        raise RuntimeError("ef2_dit: DiffusionModule step with a filled inference cache before any step 0 of this process")
    cd = self.conditioning
    if st.cfg["cond"] == "stock":                                       # the stock conditioning module verbatim (s and z)
        s, z = cd(t_hat=t, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, relative_position_encoding=relative_position_encoding, sigma_data=sigma,
                  num_diffusion_samples=num_diffusion_samples, inference_cache=inference_cache)
        s = s.reshape(-1, s.shape[-1])
    else:
        if "z" in inference_cache:
            z = inference_cache["z"]
        else:                                                            # the z branch (once per fold): stock statements or the conditioning module's pair-path hook
            z = _z_cond(cd, z_trunk, relative_position_encoding)
            inference_cache["z"] = z
        s = _s_path(st, self, t, s_inputs, sigma, num_diffusion_samples, fused)                  # [L, D] fp32
    if fresh:
        _pair_bias_all(st, self.token_transformer, z, token_attention_mask)
    denom = torch.sqrt(t * t + sigma * sigma)
    r_noisy = x_noisy / denom[:, None, None]
    a, q_skip, c_skip, p_skip, _ = self.atom_encoder(ref_pos=ref_pos, atom_attention_mask=ref_mask, ref_space_uid=ref_space_uid, ref_charge=ref_charge, ref_element=ref_element,
                                                     ref_atom_name_chars=ref_atom_name_chars, atom_to_token=tok_idx, r_l=r_noisy, s_i=s_trunk,
                                                     num_diffusion_samples=num_diffusion_samples, return_intermediates=False, inference_cache=inference_cache)
    L, D = a.shape[1], a.shape[2]
    cp = st.cfg["cond"] if st.cfg["cond"] != "stock" else "fp32"
    w, b_, eps = st.s_step_ln
    s_tok = _mm(_ln(s, w, b_, eps, _in_dtype(cp), fused), st.w_s_tok, cp, out_dtype=torch.float32)          # s_to_token(s_step_norm(s))
    x = _res_add(a.reshape(bsz * L, D), s_tok, fused)                                                       # [S*L, D]: the samples' token rows stacked (S == 1: [L, D])
    x = _token_transformer_dit(st, self.token_transformer, x, s, fused, bsz)
    w, b_, eps = st.tok_ln
    a_out = _ln(x, w, b_, eps, torch.float32, fused).view(bsz, L, D)
    r_update, _ = self.atom_decoder(a_i=a_out, q_l=q_skip, c_l=c_skip, p_lm=p_skip, atom_to_token=tok_idx, atom_attention_mask=ref_mask,
                                    num_diffusion_samples=num_diffusion_samples, return_intermediates=False)
    sigma2 = sigma * sigma
    t2 = t * t
    out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
    out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
    STATS["dit_steps"] += 1
    return {"x_denoised": out, "token_repr": a_out if return_token_repr else None, "atom_intermediates": None}


# =====================================================================================================================
# ROLL-OUT: DiffusionStructureHead.sample as one graph per step boundary
# =====================================================================================================================
def _host_tables(sched_list, gam_list, lam, eta):
    """per-step scalars exactly as the stock loop folds them (Python double arithmetic, rounded to fp32 once where a tensor op consumes them)."""
    import numpy as np
    S = len(sched_list) - 1
    t_hat = np.zeros(S, np.float32); eps = np.zeros(S, np.float32); recip = np.zeros(S, np.float32); coef = np.zeros(S, np.float32)
    for k in range(S):
        sigma_tm = float(sched_list[k]); sigma_t = float(sched_list[k + 1]); gamma = float(gam_list[k + 1])
        th = sigma_tm * (1.0 + gamma)
        t_hat[k] = np.float32(th)                                                     # torch.full((B,), th, dtype=fp32)
        eps[k] = np.float32(lam * max(th ** 2 - sigma_tm ** 2, 0.0) ** 0.5)          # eps_std * randn  -> mul(Tensor, double->float)
        recip[k] = np.float32(1.0 / th)                                               # x / th == x * fp32(1.0 / double(th)): ATen's div by a CPU scalar multiplies by the
                                                                                      # reciprocal computed in DOUBLE and rounded once (checked bitwise against ATen over the schedule's values;
                                                                                      # fp32(1)/fp32(th) differs by 1 ulp for ~27 % of divisors)
        coef[k] = np.float32(eta * (sigma_t - th))                                    # eta*(sigma_t - t_hat) * dos -> mul(Tensor, double->float)
    return t_hat, eps, recip, coef


def _out_of_scope(num_diffusion_samples, batch, return_atom_repr, early_exit, use_inference_cache, is_cuda):
    """None when a call is inside SCOPE, else the reason's name."""
    if num_diffusion_samples != SCOPE["num_diffusion_samples"]:
        return "num_diffusion_samples"
    if batch != SCOPE["batch"]:
        return "batch"
    if return_atom_repr:
        return "return_atom_repr"
    if early_exit is not None:
        return "denoising_early_exit"
    if not use_inference_cache:
        return "no_inference_cache"
    if not is_cuda:
        return "device"
    return None


def scope_note():
    """the one-line scope declaration for the kit's LEVER/NOTE lines."""
    return (f"ef2_dit: roll-out + fused step active at num_diffusion_samples={SCOPE['num_diffusion_samples']}, batch {SCOPE['batch']}, no early exit / atom repr; "
            "other sample() calls run the previous chain by name (counted rollout_fallback_<reason> / dit_fallback_<reason>)")


def applied_report():
    """for the run's verdict: did the installed levers engage on the folds that reached the sampler? (like ef2_mk_sampler's after-run judgement)"""
    s = stats()
    folds = s.get("rollout_folds", 0); fb = s.get("rollout_fallback", 0)
    out = dict(rollout_installed=bool(_CFG.get("rollout")), dit_installed=bool(_CFG.get("dit")), rollout_folds=folds, rollout_fallback=fb,
               fallback_reasons=sorted(k[len("rollout_fallback_"):] for k in s if k.startswith("rollout_fallback_")),
               dit_steps=s.get("dit_steps", 0), dit_fallback=s.get("dit_fallback", 0),
               dit_fallback_reasons=sorted(k[len("dit_fallback_"):] for k in s if k.startswith("dit_fallback_")),
               graph_captures=s.get("roll_captures", 0), graph_replays=s.get("roll_replays", 0), eager_boundaries=s.get("roll_eager_boundaries", 0))
    if folds + fb == 0:
        out["verdict"] = "no fold reached the sampler"
    elif fb == 0 and (not _CFG.get("dit") or s.get("dit_fallback", 0) == 0):
        out["verdict"] = "applied on every fold"
    else:
        out["verdict"] = f"previous chain used on {fb} sample() call(s) / {s.get('dit_fallback', 0)} step(s): outside the declared scope ({out['fallback_reasons'] + out['dit_fallback_reasons']})"
    return out


def memory_report(model=None):
    """bytes held by this module's static state, per shape: roll statics (fold-constant clones incl. z_trunk / inference cache, RNG tables, x/H
    buffers), the graph pool is torch's (not counted here), fused-step hoists (s0, 12 pair-bias buffers), ef2_opt bias_f32 twins."""
    def nbytes(o):
        if torch.is_tensor(o):
            return o.numel() * o.element_size()
        if isinstance(o, dict):
            return sum(nbytes(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return sum(nbytes(v) for v in o)
        return 0
    out = dict(rolls=[], hoists=[], bias_f32_twins_bytes=0)
    for m in ([model] if model is not None else list(_ENABLED)):
        sh = m.structure_head
        for sig, r in getattr(sh, "_dit_rolls", {}).items():
            const = (nbytes(r.static_kwargs) + nbytes(r.static_cache)) if getattr(r, "clones", True) else 0   # references to the fold's own tensors cost nothing
            tables = nbytes([r.rotq, r.trans, r.noise, r.t_hat_tab, r.eps_tab, r.recip_tab, r.coef_tab])
            small = nbytes([r.x_noisy, r.t_hat, r.x_c, r.mu_gt, r.xden, r.H, r.U, r.Vh, r.atom_mask, r.ctr, r.tok])
            out["rolls"].append(dict(B=sig[1], atoms=sig[2], steps=sig[3], fold_constants_bytes=const, rng_tables_bytes=tables, step_buffers_bytes=small,
                                    total_bytes=const + tables + small, graph=r.graph is not None))
        st = getattr(sh.diffusion_module, "_dit", None)
        if st is not None:
            for key, ent in st.hoist.items():
                out["hoists"].append(dict(B=key[0], L=key[1], s0_bytes=nbytes(ent["s0"]), pair_bias_bytes=nbytes(ent["pb"]), total_bytes=nbytes(ent["s0"]) + nbytes(ent["pb"])))
            out["packed_weights_bytes"] = sum(nbytes([w.t32, w.t16]) for w in (st.w_qkvg + st.w_o + st.w_sw + st.w_lo + [st.w_cs_all, st.w_og_all, st.w_s_tok] + [t["w_ab"] for t in st.s_trans] + [t["w_out"] for t in st.s_trans]))
        for blk in sh.diffusion_module.token_transformer.attn_blocks:
            for ent in getattr(blk, "_ef2opt_pb", {}).values():
                if len(ent) > 2 and torch.is_tensor(ent[2]):
                    out["bias_f32_twins_bytes"] += nbytes(ent[2])
            e = getattr(blk, "_dit_pb32", None)
            if e is not None:
                out["bias_f32_twins_bytes"] += nbytes(e[2])
    out["total_bytes"] = sum(r["total_bytes"] for r in out["rolls"]) + sum(h["total_bytes"] for h in out["hoists"]) + out["bias_f32_twins_bytes"] + out.get("packed_weights_bytes", 0)
    return out


def _rotation_from_quat(q):
    """the body of DiffusionStructureHead._random_rotations after its randn draw (identical tensor ops)."""
    n = q.shape[0]
    scale = torch.sqrt((q * q).sum(dim=1))
    signs = torch.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    return torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r), two_s * (i * j + k * r), 1 - two_s * (i * i + k * k),
                        two_s * (j * k - i * r), two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), dim=-1).reshape(n, 3, 3)


class _Roll:
    """static state of the graphed roll-out for one shape signature (kept across folds of that shape)."""

    def __init__(self, sh, sig, B, A, S, device, base_kwargs, cache, clone_constants=True):
        self.sig = sig; self.graph = None; self.pool = None
        self.clones = bool(clone_constants)     # True: fold constants CLONED into address-stable statics (a captured graph reads them); False (eager
                                                # boundaries above the graph budget): plain references to the fold's own tensors, re-bound per fold —
                                                # no copy of z_trunk / the conditioning cache (GBs at large token counts)
        f32 = dict(device=device, dtype=torch.float32)
        self.x_noisy = torch.zeros(B, A, 3, **f32); self.t_hat = torch.zeros(B, **f32)
        self.x_c = torch.zeros(B, A, 3, **f32); self.mu_gt = torch.zeros(B, 1, 3, **f32); self.xden = torch.zeros(B, A, 3, **f32)
        self.H = None; self.U = None; self.Vh = None                     # allocated from the eager step-0 tensors' layouts (set_layouts)
        self.atom_mask = torch.zeros(B, A, **f32)                             # every tensor the graph body reads is a roll static (refreshed per fold)
        self.ctr = torch.zeros(1, device=device, dtype=torch.long)
        self.t_hat_tab = torch.zeros(S, **f32); self.eps_tab = torch.zeros(S, **f32); self.recip_tab = torch.zeros(S, **f32); self.coef_tab = torch.zeros(S, **f32)
        self.rotq = torch.zeros(S, B, 4, **f32); self.trans = torch.zeros(S, B, 1, 3, **f32); self.noise = torch.zeros(S, B, A, 3, **f32)
        self.tok = None
        if self.clones:
            self.static_kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_kwargs.items()}
            self.static_cache = _tree_clone(cache)
        else:
            self.static_kwargs = dict(base_kwargs); self.static_cache = cache
        self.const_fold = None          # fold number whose constants the static buffers hold

    def set_layouts(self, H0, mu_gt0, x_c0):
        """static H / U / Vh / mu_gt / x_c with exactly the sizes AND strides of the tensors the stock ops produce (einsum output, cusolver's
        column-major U / Vh), so every downstream op (svd, U @ Vh, det, x_c @ R^T) sees the operand layouts it sees in the stock loop."""
        if self.H is None or self.H.stride() != H0.stride():
            self.H = torch.empty_strided(H0.size(), H0.stride(), device=H0.device, dtype=H0.dtype).zero_()
            U, _S, Vh = torch.linalg.svd(torch.eye(3, device=H0.device, dtype=torch.float32).expand(H0.shape[0], 3, 3).contiguous(), driver="gesvd")
            self.U = torch.empty_strided(U.size(), U.stride(), device=H0.device, dtype=torch.float32).zero_()
            self.Vh = torch.empty_strided(Vh.size(), Vh.stride(), device=H0.device, dtype=torch.float32).zero_()
        if self.mu_gt.stride() != mu_gt0.stride():
            self.mu_gt = torch.empty_strided(mu_gt0.size(), mu_gt0.stride(), device=H0.device, dtype=torch.float32).zero_()
        if self.x_c.stride() != x_c0.stride():
            self.x_c = torch.empty_strided(x_c0.size(), x_c0.stride(), device=H0.device, dtype=torch.float32).zero_()


def _tree_clone(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _tree_clone(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_tree_clone(v) for v in obj)
    if isinstance(obj, list):
        return [_tree_clone(v) for v in obj]
    return obj


def _tree_copy_(dst, src):
    if torch.is_tensor(dst):
        dst.copy_(src); return
    if isinstance(dst, dict):
        for k in dst:
            _tree_copy_(dst[k], src[k])
        return
    if isinstance(dst, (tuple, list)):
        for a, b in zip(dst, src):
            _tree_copy_(a, b)
        return
    assert dst == src, f"static non-tensor value changed: {dst!r} vs {src!r}"


def _tree_sig(obj):
    if torch.is_tensor(obj):
        return ("T", tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        return ("D", tuple((k, _tree_sig(v)) for k, v in obj.items()))
    if isinstance(obj, (tuple, list)):
        return ("L", tuple(_tree_sig(v) for v in obj))
    return ("V", obj)


def _kabsch_head(x_noisy, x_denoised, atom_mask):
    """_weighted_rigid_align up to (and including) H: returns x_c, mu_gt, H32 (stock op order; weights = mask = atom_mask)."""
    x = x_noisy.float(); x_gt = x_denoised.float()
    w = (atom_mask * atom_mask).unsqueeze(-1)
    denom = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    mu = (x * w).sum(dim=-2, keepdim=True) / denom
    mu_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denom
    x_c = x - mu
    xgt_c = x_gt - mu_gt
    H = torch.einsum("bni,bnj->bij", w * xgt_c, x_c)
    return x_c, mu_gt, H.float()


def _kabsch_R_torch(U, Vh, H_dtype=torch.float32):
    det = torch.linalg.det(U @ Vh)
    ones = torch.ones_like(det)
    return (U @ torch.diag_embed(torch.stack([ones, ones, det], dim=-1)) @ Vh).to(H_dtype)


def _weighted_rigid_align_device(x, x_gt, w, mask):
    """DiffusionStructureHead._weighted_rigid_align (a staticmethod: x, x_gt, w, mask) on the device Kabsch (lever kd): the stock statements
    up to H32 verbatim, then R = U diag(1,1,det(UVh)) Vh from kabsch_rotation_device instead of torch.linalg.svd(driver='gesvd') + det — no
    cusolver call, no host-side error check, one program per sample. It serves the calls the roll-out does not make itself: the step-graph /
    stock sampler's per-step alignment where ro steps aside (S > 1). CPU tensors or autograd run the stock method by name (STATS kabsch_head_stock)."""
    if (not x.is_cuda) or torch.is_grad_enabled() or x.dim() != 3:
        STATS["kabsch_head_stock"] += 1
        return _STOCK["weighted_rigid_align"](x, x_gt, w, mask)
    w = (mask * w).unsqueeze(-1)
    denom = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    mu = (x * w).sum(dim=-2, keepdim=True) / denom
    mu_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denom
    x_c = x - mu
    xgt_c = x_gt - mu_gt
    H = torch.einsum("bni,bnj->bij", w * xgt_c, x_c)
    R, _det = kabsch_rotation_device(H.float())
    STATS["kabsch_device"] += 1; STATS["kabsch_device_head"] += 1
    return x_c @ R.to(H.dtype).transpose(-1, -2) + mu_gt


_STOCK = {}


def _tail(roll, kabsch):
    """Kabsch tail of step k=ctr + the x update: returns x_{k+1} (before augmentation)."""
    if kabsch == "device":
        R, _det = kabsch_rotation_device(roll.H)
        STATS["kabsch_device"] += 1
    else:
        R = _kabsch_R_torch(roll.U, roll.Vh)
    x_al = roll.x_c @ R.transpose(-1, -2) + roll.mu_gt                      # _weighted_rigid_align's return
    x_al = x_al.to(dtype=roll.xden.dtype)
    recip = roll.recip_tab.index_select(0, roll.ctr)                        # [1]
    coef = roll.coef_tab.index_select(0, roll.ctr)
    dos = (x_al - roll.xden) * recip                                         # == (x_noisy - x_denoised) / t_hat_val   (bitwise, see _host_tables)
    return x_al + dos * coef                                                 # == x_noisy + eta*(sigma_t - t_hat) * dos


def _head(roll, j, x, atom_mask, dm, nds):
    """centre + augmentation + noise of step j (device index tensor), the DiffusionModule call, and the Kabsch head; writes the roll statics."""
    mask = atom_mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(dim=1, keepdim=True) / denom
    x = x - mean
    q = roll.rotq.index_select(0, j)[0]                                      # [B, 4]  (pre-drawn, stock order)
    r = _rotation_from_quat(q)
    x = torch.einsum("bmd,bds->bms", x, r)
    x = x + roll.trans.index_select(0, j)[0]                                 # [B, 1, 3]
    eps = roll.eps_tab.index_select(0, j)                                    # [1]
    x_noisy = x + roll.noise.index_select(0, j)[0] * eps                     # == x + eps_std * randn_like(x)
    roll.x_noisy.copy_(x_noisy)
    roll.t_hat.copy_(roll.t_hat_tab.index_select(0, j).expand(roll.t_hat.shape[0]))
    out = dm(**roll.static_kwargs, x_noisy=roll.x_noisy, t_hat=roll.t_hat, num_diffusion_samples=nds, return_token_repr=True, return_atom_repr=False,
             inference_cache=roll.static_cache)
    x_den = out["x_denoised"]
    x_c, mu_gt, H32 = _kabsch_head(roll.x_noisy, x_den, atom_mask)
    roll.xden.copy_(x_den); roll.x_c.copy_(x_c); roll.mu_gt.copy_(mu_gt); roll.H.copy_(H32)
    if roll.tok is None:
        roll.tok = torch.zeros_like(out["token_repr"])
    roll.tok.copy_(out["token_repr"])


def _boundary(roll, atom_mask, dm, nds, kabsch):
    """ONE graph body: tail of step ctr, head of step ctr+1, ctr += 1."""
    x = _tail(roll, kabsch)
    j = roll.ctr + 1
    _head(roll, j, x, atom_mask, dm, nds)
    roll.ctr.add_(1)


def _svd_into(roll):
    U, _S, Vh = torch.linalg.svd(roll.H, driver="gesvd")                     # stock call (cusolver gesvd; host error check) — eager, between replays
    roll.U.copy_(U); roll.Vh.copy_(Vh)
    STATS["kabsch_svd_eager"] += 1


def _sample_dit(self, z_trunk, s_inputs, s_trunk, relative_position_encoding, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid,
                tok_idx, asym_id, residue_index, entity_id, token_index, sym_id, token_attention_mask=None, num_diffusion_samples=1, num_sampling_steps=None,
                max_inference_sigma=256.0, noise_scale=None, step_scale=None, return_atom_repr=False, use_inference_cache=True, denoising_early_exit_rmsd=None):
    cfg = _CFG
    kw_all = dict(z_trunk=z_trunk, s_inputs=s_inputs, s_trunk=s_trunk, relative_position_encoding=relative_position_encoding, ref_pos=ref_pos, ref_charge=ref_charge,
                  ref_mask=ref_mask, ref_element=ref_element, ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid, tok_idx=tok_idx, asym_id=asym_id,
                  residue_index=residue_index, entity_id=entity_id, token_index=token_index, sym_id=sym_id, token_attention_mask=token_attention_mask,
                  num_diffusion_samples=num_diffusion_samples, num_sampling_steps=num_sampling_steps, max_inference_sigma=max_inference_sigma, noise_scale=noise_scale,
                  step_scale=step_scale, return_atom_repr=return_atom_repr, use_inference_cache=use_inference_cache, denoising_early_exit_rmsd=denoising_early_exit_rmsd)
    target_batch = s_inputs.shape[0] * num_diffusion_samples
    why = _out_of_scope(num_diffusion_samples, target_batch, return_atom_repr, denoising_early_exit_rmsd, use_inference_cache, s_inputs.is_cuda)
    if (not cfg.get("rollout")) or why:
        STATS["rollout_fallback"] += 1; STATS["rollout_fallback_" + (why or "disabled")] += 1
        return self._dit_inner_sample(**kw_all)
    STATS["rollout_folds"] += 1
    # the per-fold prologue of the chain this roll-out replaces, through ef2_opt's named API: fold epoch (SWA static masks), pair-bias epoch (every
    # pb-cached block recomputes its bias at this fold's eager step 0 — fold k+1 can never be served fold k's bias), numerics guard.
    eo = _CFG.get("_ef2_opt")
    if eo is not None:
        _fold_epoch, pb_epoch = eo.sampler_fold_prologue(pair_bias_epoch=True)
        if eo.pair_bias_cache_blocks() and pb_epoch == getattr(self, "_dit_last_pb_epoch", None):
            raise RuntimeError("ef2_dit rollout: the pair-bias epoch did not advance for a new fold (lever pb would serve a stale bias) — refusing")
        self._dit_last_pb_epoch = pb_epoch
    elif any(hasattr(blk, "_ef2opt_pb") for blk in self.diffusion_module.token_transformer.attn_blocks):
        raise RuntimeError("ef2_dit rollout: pair-bias cache installed on the blocks but ef2_opt's prologue API is unavailable — refusing")
    kabsch = cfg["kabsch"]
    with torch.inference_mode():
        n_atoms = tok_idx.shape[1]
        device = s_inputs.device
        inference_cache = {}
        steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        schedule = self.inference_noise_schedule(steps, device)
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
        lam = self.noise_scale if noise_scale is None else float(noise_scale)
        eta = self.step_scale if step_scale is None else float(step_scale)
        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)        # RNG draw #1 as stock
        atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()
        gammas = torch.where(schedule > self.gamma_min, torch.full_like(schedule, self.gamma_0), torch.zeros_like(schedule))
        sched_list = schedule.tolist(); gam_list = gammas.tolist()
        S = len(sched_list) - 1
        B, A = target_batch, n_atoms
        dm = self.diffusion_module
        base_kwargs = dict(ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element, ref_atom_name_chars=ref_atom_name_chars,
                           ref_space_uid=ref_space_uid, tok_idx=tok_idx, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                           relative_position_encoding=relative_position_encoding, asym_id=asym_id, residue_index=residue_index, entity_id=entity_id,
                           token_index=token_index, sym_id=sym_id, token_attention_mask=token_attention_mask)
        # ---- pre-draw every per-step RNG value in stock ORDER with stock SHAPES (identical Philox stream): per step rot [B,4], trans like x[:, 0:1, :], noise like x
        rotq = torch.empty(S, B, 4, device=device, dtype=torch.float32); trans = torch.empty(S, B, 1, 3, device=device, dtype=torch.float32)
        noise = torch.empty(S, B, A, 3, device=device, dtype=torch.float32)
        for k in range(S):
            rotq[k].copy_(torch.randn((B, 4), dtype=x.dtype, device=device))
            trans[k].copy_(torch.randn_like(x[:, 0:1, :]))
            noise[k].copy_(torch.randn_like(x))
        STATS["rng_predraw_steps"] += S
        t_hat_np, eps_np, recip_np, coef_np = _host_tables(sched_list, gam_list, lam, eta)
        own = cfg.get("graph_budget_tokens")                                   # None: the kit's per-shape budget decides (ef2_opt); else the roll-out's own token threshold
        if own is None:
            over_budget = bool(eo._over_graph_budget("sampler", int(z_trunk.shape[1])))
        else:
            over_budget = own > 0 and int(z_trunk.shape[1]) > own
            STATS["roll_budget_own_" + ("eager" if over_budget else "capture")] += 1
        use_graph = bool(cfg.get("graph", True)) and not over_budget
        # ---- step 0 head, eager (fills the module's inference cache and every per-fold hoist)
        mask = atom_mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x = x - mean
        r = _rotation_from_quat(rotq[0])
        x = torch.einsum("bmd,bds->bms", x, r)
        x = x + trans[0]
        x_noisy0 = x + noise[0] * torch.from_numpy(eps_np[0:1]).to(device)
        t_hat0 = torch.full((B,), float(t_hat_np[0]), device=device, dtype=torch.float32)
        out0 = dm(**base_kwargs, x_noisy=x_noisy0, t_hat=t_hat0, num_diffusion_samples=num_diffusion_samples, return_token_repr=True, return_atom_repr=False,
                  inference_cache=inference_cache)
        STATS["rollout_eager_steps"] += 1
        x_den0 = out0["x_denoised"]
        x_c0, mu_gt0, H0 = _kabsch_head(x_noisy0, x_den0, atom_mask)
        # ---- roll state for this shape: keyed on the padded shapes AND the filled inference cache's shapes (the atom encoder's varlen index
        # tensors are sized by the VALID atom count: two items of one padded shape can differ) — a graph and its statics serve one such key
        sig = ("roll", B, A, S, _tree_sig(base_kwargs), _tree_sig(inference_cache))
        rolls = self._dit_rolls
        roll = rolls.get(sig)
        new_roll = roll is None
        self._dit_fold_n = getattr(self, "_dit_fold_n", 0) + 1
        fold_n = self._dit_fold_n
        if new_roll:
            while len(rolls) >= max(1, int(cfg.get("lru", 2))):                   # evict the oldest shape's roll (its graph + private pool go with it)
                rolls.popitem(last=False); STATS["roll_evict"] += 1
            if STATS["roll_evict"]:
                import gc; gc.collect()
            roll = _Roll(self, sig, B, A, S, device, base_kwargs, inference_cache, clone_constants=use_graph)
            roll.L = int(z_trunk.shape[1])
            roll.const_fold = fold_n
            rolls[sig] = roll
            STATS["roll_new"] += 1
        elif roll.const_fold != fold_n:                                       # static_once: fold constants staged ONCE per fold
            if use_graph and not roll.clones:                                 # (a roll first used eagerly, now to be captured: give it its own statics)
                roll.static_kwargs = {k_: (v.clone() if torch.is_tensor(v) else v) for k_, v in base_kwargs.items()}
                roll.static_cache = _tree_clone(inference_cache); roll.clones = True; STATS["static_clone_late"] += 1
            elif roll.clones:
                for k_, v in base_kwargs.items():
                    sv = roll.static_kwargs[k_]
                    if torch.is_tensor(sv):
                        sv.copy_(v)
                    else:
                        assert sv == v
                _tree_copy_(roll.static_cache, inference_cache)
            else:                                                             # eager roll: re-bind the references to this fold's tensors
                roll.static_kwargs = dict(base_kwargs); roll.static_cache = inference_cache
            roll.const_fold = fold_n
            STATS["static_stage_fold"] += 1
        # per-fold tables and step-0 state into the statics
        roll.t_hat_tab.copy_(torch.from_numpy(t_hat_np).to(device)); roll.eps_tab.copy_(torch.from_numpy(eps_np).to(device))
        roll.recip_tab.copy_(torch.from_numpy(recip_np).to(device)); roll.coef_tab.copy_(torch.from_numpy(coef_np).to(device))
        roll.rotq.copy_(rotq); roll.trans.copy_(trans); roll.noise.copy_(noise)
        roll.set_layouts(H0, mu_gt0, x_c0)
        roll.atom_mask.copy_(atom_mask)
        roll.x_noisy.copy_(x_noisy0); roll.t_hat.copy_(t_hat0); roll.xden.copy_(x_den0); roll.x_c.copy_(x_c0); roll.mu_gt.copy_(mu_gt0); roll.H.copy_(H0)
        if roll.tok is None:
            roll.tok = torch.zeros_like(out0["token_repr"])
        roll.tok.copy_(out0["token_repr"])
        roll.ctr.zero_()
        k_next = 0
        if use_graph and roll.graph is None:
            # warm-up = the real boundary 0 -> 1, eagerly on a side stream (also JIT-compiles every kernel); then capture (records, does not execute)
            if kabsch == "torch":
                _svd_into(roll)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _boundary(roll, roll.atom_mask, dm, num_diffusion_samples, kabsch)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            rng_state = torch.cuda.get_rng_state(device)
            t0 = time.perf_counter()
            roll.pool = torch.cuda.graph_pool_handle() if _CFG.get("_pool") is None else _CFG["_pool"]
            _CFG["_pool"] = roll.pool
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=roll.pool):
                _boundary(roll, roll.atom_mask, dm, num_diffusion_samples, kabsch)
            torch.cuda.synchronize()
            if not torch.equal(rng_state, torch.cuda.get_rng_state(device)):        # the graph body holds no RNG op (all draws are pre-drawn); refuse by name otherwise
                raise RuntimeError("ef2_dit rollout: the CUDA RNG advanced during graph capture (an RNG op inside the step boundary) — refusing")
            roll.graph = g
            STATS["roll_captures"] += 1; STATS["roll_capture_s_x1000"] += int(1000 * (time.perf_counter() - t0))
            k_next = 1
        # ---- the roll-out: boundaries k -> k+1 for k = k_next .. S-2
        for k in range(k_next, S - 1):
            if not cfg.get("static_once", True):                               # ablation only: the old per-step re-staging cost
                for k_, v in base_kwargs.items():
                    sv = roll.static_kwargs[k_]
                    if torch.is_tensor(sv):
                        sv.copy_(v)
                _tree_copy_(roll.static_cache, inference_cache)
            if kabsch == "torch":
                _svd_into(roll)
            if use_graph:
                roll.graph.replay(); STATS["roll_replays"] += 1
            else:
                _boundary(roll, roll.atom_mask, dm, num_diffusion_samples, kabsch); STATS["roll_eager_boundaries"] += 1
        # ---- final tail (step S-1), eager
        if kabsch == "torch":
            _svd_into(roll)
        x = _tail(roll, kabsch).clone()
        token_repr = roll.tok.clone()
        if not roll.clones:                                                   # an EAGER roll holds plain references to this fold's pair rows and the module's
            roll.static_kwargs = None; roll.static_cache = None               # inference cache (no copies): drop them as sample() returns so nothing of the fold
            roll.const_fold = None; STATS["roll_eager_refs_dropped"] += 1     # outlives it through the confidence head / into the next fold (re-bound at the
            st_ = getattr(dm, "_dit", None)                                   # next fold's step 0); a captured roll keeps its cloned statics (its graph reads them).
            if st_ is not None and st_.hoist.pop((B, roll.L), None) is not None:   # The fused step's per-shape hoist entry (s0 + the 12 pair-bias buffers, [B,H,L,L]
                st_.cur = None; STATS["hoist_eager_released"] += 1           # each) goes too under an eager roll: no graph reads it by address, and the next fold
        return {"sample_atom_coords": x, "diff_token_repr": token_repr}      # of the shape re-allocates it at its eager step 0


# =====================================================================================================================
# bias_f32: hoist the per-block per-step bf16 -> fp32 pair-bias copy of the EXISTING consumers (ef2_opt pb cache, ef2_mk_sampler)
# =====================================================================================================================
def _install_bias_f32(model):
    """exact-safe: the consumers keep reading `bias.to(q.dtype)`; we hand them an fp32 tensor refreshed once per fold so that `.to` is the identity."""
    n = 0
    try:
        import ef2_mk_sampler as MK
        if not getattr(MK, "_dit_bias_f32", False):
            _orig = MK._pair_bias

            def _pair_bias_f32(st, blk, z, attention_mask, bsz, n_):
                b = _orig(st, blk, z, attention_mask, bsz, n_)
                if b.dtype == torch.float32 or b.dim() != 4 or b.shape[1] != blk.num_heads:
                    return b
                ent = getattr(blk, "_dit_pb32", None)                      # (epoch, src buffer id, static fp32 twin)
                if ent is None or ent[2].shape != b.shape:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("ef2_dit bias_f32: twin buffer missing during capture (step 0 must run eagerly first)")
                    blk._dit_pb32 = (MK._EPOCH["n"], id(b), b.float()); STATS["bias_f32_new_buffer"] += 1
                elif ent[0] != MK._EPOCH["n"] or ent[1] != id(b):
                    ent[2].copy_(b); blk._dit_pb32 = (MK._EPOCH["n"], id(b), ent[2]); STATS["bias_f32_refresh"] += 1
                else:
                    STATS["bias_f32_hits"] += 1
                return blk._dit_pb32[2]
            MK._pair_bias = _pair_bias_f32; MK._dit_bias_f32 = True
        n += 1
    except ImportError:
        pass
    eo = _CFG.get("_ef2_opt")
    if eo is not None and hasattr(eo.CFG, "bias_f32"):
        eo.CFG.bias_f32 = True; n += 1
    return n


# =====================================================================================================================
# install / clear / describe
# =====================================================================================================================
def clear(reason="shape"):
    """drop every roll graph + static buffer (synchronize first; fresh pool next time)."""
    import gc
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for m in list(_ENABLED):
        sh = m.structure_head
        if hasattr(sh, "_dit_rolls"):
            sh._dit_rolls.clear()
        st = getattr(sh.diffusion_module, "_dit", None)
        if st is not None:
            st.hoist.clear(); st.cur = None
    _CFG["_pool"] = None
    gc.collect()
    try:
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    except Exception:
        pass
    STATS["clears_" + reason] += 1


def install(model, *, rollout=True, graph=True, static_once=True, kabsch="torch", bias_f32=False, lru=2, graph_budget_tokens=None,
            dit=False, gemm="bf16", cond="bf16", attn="flash", attn_precision="bf16", fused_ew=True):
    """Install the levers on `model` (after ef2_server.configure()). Returns describe()."""
    import ef2_srcguard                                                       # the roll-out re-issues DiffusionStructureHead.sample (+ its augmentation / Kabsch helpers) and the fused step the
    ef2_srcguard.check_many({"ro": bool(rollout), "kd": kabsch == "device", "dit": bool(dit)})   # diffusion transformer's forwards: refuse by name on another upstream source
    assert kabsch in ("torch", "device")
    assert gemm in GEMM_PRECISIONS and (cond in GEMM_PRECISIONS or cond == "stock") and attn in ("flash", "sdpa") and attn_precision in ATTN_PRECISIONS
    if (dit or kabsch == "device" or fused_ew) and not _HAS_TRITON:
        raise RuntimeError("ef2_dit: triton is required for dit / kabsch='device'")
    if graph_budget_tokens is not None and int(graph_budget_tokens) < 0:
        raise ValueError("ef2_dit: graph_budget_tokens must be None (the kit's per-shape budget), 0 (capture every shape) or a token count")
    _CFG.update(rollout=bool(rollout), graph=bool(graph), static_once=bool(static_once), kabsch=kabsch, bias_f32=bool(bias_f32), lru=int(lru),
                graph_budget_tokens=(None if graph_budget_tokens is None else int(graph_budget_tokens)), dit=bool(dit), gemm=gemm, cond=cond, attn=attn, attn_precision=attn_precision, fused_ew=bool(fused_ew))
    try:
        import ef2_opt as EO
        _CFG["_ef2_opt"] = EO
        if not getattr(EO, "_dit_hooked", False):
            _orig = EO.clear_graphs

            def clear_graphs_dit(*a, **k):
                r = _orig(*a, **k); clear(reason="ef2_opt"); return r
            EO.clear_graphs = clear_graphs_dit; EO._dit_hooked = True
    except ImportError:
        _CFG["_ef2_opt"] = None
    sh = model.structure_head
    dm = sh.diffusion_module
    if dit and getattr(dm, "_mk_enabled", False):                  # the fused step supersedes ef2_mk_sampler's forward: the composition is 'dit INSTEAD of mk'
        raise RuntimeError("ef2_dit: dit=True on a model with ef2_mk_sampler enabled — the fused step supersedes mk's diffusion-module forward and mk's "
                           "counters would never fire; compose 'dit' instead of 'mk' (ef2_mk_sampler.disable(model) first) — refusing by name")
    out = {}
    if rollout and not getattr(sh, "_dit_rollout", False):
        sh._dit_inner_sample = sh.sample                      # the installed chain (ef2_opt pb epoch wrapper -> _sample_v2 -> ...): used for the calls the
                                                              # roll-out does not cover (num_diffusion_samples > 1, early exit, atom repr)
        if _CFG.get("_ef2_opt") is None:
            raise RuntimeError("ef2_dit rollout: ef2_opt (fold prologue / graph budget API) is not importable — refusing")
        sh._dit_rolls = collections.OrderedDict()
        sh.sample = types.MethodType(_sample_dit, sh)
        sh._dit_rollout = True
        out["rollout"] = True
    if kabsch == "device" and not getattr(sh, "_dit_wra_device", False):   # kd at every S: the structure head's OWN Kabsch alignment — the call the step-graph
        _STOCK.setdefault("weighted_rigid_align", sh._weighted_rigid_align)   # sampler / stock loop makes once per step where ro steps aside (S > 1) — on the device kernel
        sh._weighted_rigid_align = _weighted_rigid_align_device    # (a staticmethod upstream: instance attribute, no binding): no cusolver gesvd, no host error check
        sh._dit_wra_device = True
        out["kabsch_head_patched"] = True
    if bias_f32:
        out["bias_f32_patched"] = _install_bias_f32(model)
    if dit and not getattr(dm, "_dit_installed", False):
        dm._dit = _DitState(dm, _CFG)
        dm._dit_prev_forward = dm.forward
        dm.forward = types.MethodType(_dm_forward_dit, dm)
        dm._dit_installed = True
        out["dit"] = True
    elif dit:
        dm._dit.cfg.update(_CFG)
    if model not in _ENABLED:
        _ENABLED.append(model)
    out.update(describe())
    return out


LEVER_NAMES = ("ro", "kd", "dit")           # the kit's names: ro = rollout + graph + static_once + bias_f32; kd = kabsch='device'; dit = the fused step


def levers_on():
    """{ro, kd, dit: bool} as installed in this process (an installed module with no enabled model reads all False)."""
    on = bool(_ENABLED)
    return dict(ro=on and all(bool(_CFG.get(k)) for k in ("rollout", "graph", "static_once", "bias_f32")),
                kd=on and _CFG.get("kabsch") == "device", dit=on and bool(_CFG.get("dit")))


def knobs():
    """The precision sub-choices as installed: {gemm, cond, attn, attn_precision} (blank-free words for the dit LEVER line)."""
    return {k: str(_CFG.get(k)) for k in ("gemm", "cond", "attn", "attn_precision")}


def describe():
    return dict(version=VERSION, cfg={k: v for k, v in _CFG.items() if not k.startswith("_")}, levers=levers_on(), scope=dict(SCOPE), scope_step=dict(SCOPE_STEP), triton=_HAS_TRITON, enabled_models=len(_ENABLED))


def stats():
    return {k: int(v) for k, v in STATS.items()}
