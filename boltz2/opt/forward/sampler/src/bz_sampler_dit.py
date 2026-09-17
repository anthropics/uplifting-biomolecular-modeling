"""bz_sampler_dit.py — the `dit_fused` lever (FAST tier, precision class): the diffusion module's
24-layer token transformer (boltz.model.modules.transformersv2.DiffusionTransformer, the `token_transformer` instance of DiffusionModule) as a
fused bf16 tensor-core step (per layer: AdaLN -> attention with pair bias (16 heads x 48) gated by sigmoid(g) and by an s-conditioned
output gate -> conditioned SwiGLU transition, in Boltz-2's SwiGLU order: silu(second half) * first half).

Dataflow per denoiser step (M = multiplicity x tokens rows; D = 768; H = 16, HD = 48; HID = 1536; L = 24 layers):
  conditioning of ALL layers from 2 GEMMs:  CS = LN0(s) @ Wcs^T  [M, L*4D]  (per layer: adaln scale | adaln bias | transition-adaln scale | bias; the
                                             s_norm weight of each AdaLN folded into the
                                             columns),  OG = s @ Wog^T  [M, L*2D] (the two output gates, pre-sigmoid)
  per layer:  qkvg = xin @ Wqkvg^T (bf16 [M,4D]; q's bias added in the attention prologue)
              ctx  = flash attention with pair bias (Triton; bias bf16 [H,N,N] per layer with the key mask folded in once per sample(); online softmax
                     in base 2 with fp32 statistics; QK^T and PV on bf16 tensor cores (word attn=bf16) or tf32 (attn=tf32); * sigmoid(g) in the epilogue)
              o    = ctx @ Wo^T (fp32 out);  x, xin = [x + sigmoid(OG_a) * o ; transition AdaLN of the new x]  (one Triton kernel, fp32 residual stream)
              tri  = xin @ [Wswish | Wa_to_b]^T (bf16 [M, 3*HID]);  h = silu(tri[:, HID:2HID]) * tri[:, :HID] * tri[:, 2HID:]  (one kernel)
              o    = h @ Wb_to_a^T (fp32 out);  x, xin = [x + sigmoid(OG_t) * o ; the NEXT layer's AdaLN]  (one kernel)
  10 launches per layer (stock: ~45), GEMM inputs bf16 with fp32 accumulation (cuBLAS), residual stream and all LayerNorm / softmax statistics fp32.
Words (BOLTZ_SAMPLER_DIT=<word>[,<word>]): `weights=resident|sample` = the packed weights' lifetime (resident from the first sample() on — the
roll-out reads them in its captured step — or built at each sample() entry and dropped at its exit: the memory row, nothing resident outside
sample()). Precision words: `bf16` = gemm=bf16 + attn=bf16 (the lever); `gemm=tf32` / `attn=tf32` = the same dataflow with
TF32 tensor-core products on fp32 data (a higher-precision variant of the same dataflow). Tolerance class: outputs differ from stock (bf16
products) — never bitwise; compare against stock's seed-to-seed band.

Scope by name: the token transformer instance only (to_keys is None, pair bias present, B == 1 pair batch); anything else is served by the stock
forward and counted (`scope=<reason>`). The per-sample() pair-bias pack (bf16 [L,H,N,N] + key mask) is prepared by bz_sampler's sample() entry
(prepare()); a forward that finds no prepared pack for its shapes falls back to stock by name (`no_pack`).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False

STATS = {"installed": None, "calls": 0, "served": 0, "scope": {}, "packs": 0, "pack_gib": 0.0, "flash_cfg": {}, "gemm": None, "attn": None,
         "layers": None, "launches_per_layer": 10, "weights_mib": None, "weights": None, "weights_builds": 0, "weights_releases": 0,
         "out_dtype_probe": None, "mask_fold": "bias"}   # mask_fold: stock's (1 - token_pad_mask) * -inf term is folded into the packed bias once per sample() (prepare)
_CFG = {"gemm": "bf16", "attn": "bf16", "on": False, "weights": "resident"}   # weights: `resident` = the packed weights of the token transformer live from the first sample() on
                                                                              # (513 MiB, the roll-out's captured step reads them in place); `sample` = built at sample() entry
                                                                              # (prepare) and dropped at its exit (release): nothing of the lever is resident outside sample()
                                                                              # — the memory row's word (the item peak is the trunk's; one bf16 cast of 24 layers per item)
_DEBUG = os.environ.get("BOLTZ_SAMPLER_DEBUG", "0") == "1"


def _log(*a):
    if _DEBUG:
        print("[bz_sampler_dit]", *a, file=sys.stderr, flush=True)


if _HAVE_TRITON:

    @triton.jit
    def _adaln_kernel(X, CS, CH, BCS, OUT, D: tl.constexpr, BD: tl.constexpr, eps, sx, scs, sch, so, OUT_BF16: tl.constexpr):
        """OUT[m,:] = sigmoid(CS[m,:] + BCS) * LN(X[m,:]) + CH[m,:]   (AdaLN with the s-side already projected; fp32 math)"""
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
        a = tl.sigmoid(cs) * (xc * rstd) + ch
        if OUT_BF16:
            tl.store(OUT + m * so + offs, a.to(tl.bfloat16), mask=msk)
        else:
            tl.store(OUT + m * so + offs, a, mask=msk)

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
    def _swiglu_ab_kernel(T, OUT, HID: tl.constexpr, BH: tl.constexpr, st, so, OUT_BF16: tl.constexpr):
        """Boltz-2 ConditionedTransitionBlock inner product on the packed GEMM output T = [swish (2*HID) | a_to_b (HID)]:
        OUT[m,:] = silu(T[m, HID:2HID]) * T[m, 0:HID] * T[m, 2HID:3HID]   (SwiGLU: x, gates = chunk(2); silu(gates) * x; then * a_to_b(a); fp32 math)"""
        m = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BH)
        msk = offs < HID
        x = tl.load(T + m * st + offs, mask=msk, other=0.0).to(tl.float32)
        gt = tl.load(T + m * st + HID + offs, mask=msk, other=0.0).to(tl.float32)
        ab = tl.load(T + m * st + 2 * HID + offs, mask=msk, other=0.0).to(tl.float32)
        h = gt * tl.sigmoid(gt) * x * ab
        if OUT_BF16:
            tl.store(OUT + m * so + offs, h.to(tl.bfloat16), mask=msk)
        else:
            tl.store(OUT + m * so + offs, h, mask=msk)

    @triton.jit
    def _fb_dot(a, b, acc, PREC: tl.constexpr):
        if PREC == 3:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        elif PREC == 1:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="tf32")
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")

    @triton.jit
    def _fb_tile(QKVG, BIAS, k0, L, row0, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                 D: tl.constexpr, HD: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr, PREC: tl.constexpr, MASKED: tl.constexpr):
        """one key tile [k0, k0+BKV): scores from the two head-dim slabs (32 + 16 = 48, no padding), + bias (pre-scaled by log2 e), online softmax
        in base 2, P@V accumulated in fp32. MASKED only for the ragged last tile."""
        LOG2E: tl.constexpr = 1.4426950408889634
        offs_k = k0 + tl.arange(0, BKV)
        rows_k = row0 + offs_k
        if MASKED:
            k_valid = offs_k < L
            k0t = tl.load(QKVG + rows_k[None, :] * s_row + (D + h * HD + offs_d0)[:, None], mask=k_valid[None, :], other=0.0)
            k1t = tl.load(QKVG + rows_k[None, :] * s_row + (D + h * HD + offs_d1)[:, None], mask=k_valid[None, :], other=0.0)
            bias = tl.load(BIAS + offs_q[:, None] * s_bq + offs_k[None, :], mask=q_mask[:, None] & k_valid[None, :], other=0.0).to(tl.float32)
        else:
            k0t = tl.load(QKVG + rows_k[None, :] * s_row + (D + h * HD + offs_d0)[:, None])
            k1t = tl.load(QKVG + rows_k[None, :] * s_row + (D + h * HD + offs_d1)[:, None])
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
            v0 = tl.load(QKVG + rows_k[:, None] * s_row + (2 * D + h * HD + offs_d0)[None, :], mask=k_valid[:, None], other=0.0)
            v1 = tl.load(QKVG + rows_k[:, None] * s_row + (2 * D + h * HD + offs_d1)[None, :], mask=k_valid[:, None], other=0.0)
        else:
            v0 = tl.load(QKVG + rows_k[:, None] * s_row + (2 * D + h * HD + offs_d0)[None, :])
            v1 = tl.load(QKVG + rows_k[:, None] * s_row + (2 * D + h * HD + offs_d1)[None, :])
        if PREC == 3:
            p = p.to(tl.bfloat16)
        acc0 = _fb_dot(p, v0, acc0, PREC)
        acc1 = _fb_dot(p, v1, acc1, PREC)
        return m_new, l_i, acc0, acc1

    @triton.jit
    def _flash_bias_kernel(QKVG, QB, BIAS, OUT, L, s_row, s_bh, s_bq, s_out, scale,
                           H: tl.constexpr, HD: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr, OUT_BF16: tl.constexpr,
                           BQ: tl.constexpr, BKV: tl.constexpr):
        """flash attention with pair bias: rows of QKVG = [q_raw | k | v | g_raw] (H heads x HD=48 each), row block b of L rows per sample
        (grid z); q bias QB [D]; bias [H, L, L] shared by the samples (bf16 or fp32, read in place; the producer folded the key mask in);
        OUT[b*L + i, h*HD:(h+1)*HD] = softmax((q_raw+qb) k^T * scale + bias) v * sigmoid(g_raw). Head dim as 32 + 16 (no padding); base-2 online
        softmax with fp32 statistics and fp32 accumulation in every PREC (0 ieee fp32 dots, 1 tf32, 3 bf16 tensor cores). grid = (cdiv(L,BQ), H, Bm)."""
        LOG2E: tl.constexpr = 1.4426950408889634
        # MMA operands are the K / V tiles as loaded — no dtype conversion of a loaded tile in front of tl.dot (see install(): refused words)
        if PREC == 3:
            tl.static_assert(QKVG.dtype.element_ty == tl.bfloat16, "PREC=3 (bf16 products) requires bf16 activations")
        else:
            tl.static_assert(QKVG.dtype.element_ty == tl.float32, "PREC=1|0 (tf32 | ieee products) requires fp32 activations")
        pid_q = tl.program_id(0).to(tl.int64); h = tl.program_id(1).to(tl.int64); b = tl.program_id(2).to(tl.int64)
        row0 = b * L
        offs_q = pid_q * BQ + tl.arange(0, BQ)
        offs_d0 = tl.arange(0, 32); offs_d1 = 32 + tl.arange(0, 16)
        q_mask = offs_q < L
        rows_q = row0 + offs_q
        q0 = tl.load(QKVG + rows_q[:, None] * s_row + (h * HD + offs_d0)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        q1 = tl.load(QKVG + rows_q[:, None] * s_row + (h * HD + offs_d1)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
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
            m_i, l_i, acc0, acc1 = _fb_tile(QKVG, BIAS_h, k0, L, row0, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                                            D, HD, BQ, BKV, PREC, False)
        if L_full < L:
            m_i, l_i, acc0, acc1 = _fb_tile(QKVG, BIAS_h, L_full, L, row0, s_row, s_bq, h, offs_q, q_mask, q0, q1, m_i, l_i, acc0, acc1, offs_d0, offs_d1,
                                            D, HD, BQ, BKV, PREC, True)
        g0 = tl.load(QKVG + rows_q[:, None] * s_row + (3 * D + h * HD + offs_d0)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        g1 = tl.load(QKVG + rows_q[:, None] * s_row + (3 * D + h * HD + offs_d1)[None, :], mask=q_mask[:, None], other=0.0).to(tl.float32)
        inv = 1.0 / l_i
        o0 = acc0 * inv[:, None] * tl.sigmoid(g0); o1 = acc1 * inv[:, None] * tl.sigmoid(g1)
        p0 = OUT + rows_q[:, None] * s_out + (h * HD + offs_d0)[None, :]; p1 = OUT + rows_q[:, None] * s_out + (h * HD + offs_d1)[None, :]
        if OUT_BF16:
            tl.store(p0, o0.to(tl.bfloat16), mask=q_mask[:, None]); tl.store(p1, o1.to(tl.bfloat16), mask=q_mask[:, None])
        else:
            tl.store(p0, o0, mask=q_mask[:, None]); tl.store(p1, o1, mask=q_mask[:, None])


def _nw(D):
    return 4 if D <= 1024 else 8


def _flash_cfg(L, prec):
    """static (BQ, BKV, num_warps, num_stages) per token count (H100 table: small tiles for occupancy below ~600 tokens, 64x128 above)."""
    if prec == "ieee":
        return (64, 32, 4, 2)
    if L <= 512:
        return (32, 64, 2, 2)
    return (64, 128, 4, 1)


class _W:
    """one packed weight in nn.Linear layout [N, K] (a torch.cat of stock weights along N): the bf16 copy for gemm=bf16, the fp32 copy only for
    gemm=tf32 (memory: 24 layers x 7.7 M parameters); mm uses .t() views (TN GEMM, no transpose copy)."""
    __slots__ = ("w32", "w16", "N", "K")

    def __init__(self, w_out_in):
        w = w_out_in.detach().float().contiguous()
        self.N, self.K = w.shape
        self.w16 = w.to(torch.bfloat16) if _CFG["gemm"] == "bf16" else None
        self.w32 = w if _CFG["gemm"] != "bf16" else None


def _mm(x, w, prec, out_dtype=torch.float32):
    """x [M, K] @ W^T -> [M, N]. bf16: x is bf16, fp32 accumulate, out bf16 or fp32 (cuBLAS out_dtype). tf32: fp32 data, TF32 tensor-core products."""
    if prec == "bf16":
        if out_dtype == torch.bfloat16:
            return torch.mm(x, w.w16.t())
        return torch.mm(x, w.w16.t(), out_dtype=torch.float32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (prec == "tf32")
    try:
        y = torch.mm(x if x.dtype == torch.float32 else x.float(), w.w32.t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert torch.backends.cuda.matmul.allow_tf32 == prev            # the process-wide switch is restored: a TF32 product never leaks into another module's fp32 GEMM
    return y if out_dtype == torch.float32 else y.to(out_dtype)


class _State:
    """Packed weights of one DiffusionTransformer (token) instance + the per-sample() pair-bias pack."""

    def __init__(self, tt):
        lay = tt.layers
        self.L = len(lay)
        A0 = lay[0].pair_bias_attn
        self.H, self.HD = int(A0.num_heads), int(A0.head_dim)
        self.D = int(A0.c_s) if hasattr(A0, "c_s") else self.H * self.HD
        self.inf = float(A0.inf)
        self.scale = 1.0 / (self.HD ** 0.5)
        self.eps = float(lay[0].adaln.a_norm.eps)
        self.HID = int(lay[0].transition.b_to_a.in_features)
        D = self.D
        with torch.no_grad():
            self.w_qkvg, self.bq, self.w_o, self.w_tri, self.w_b2a = [], [], [], [], []
            cs_rows, og_rows, self.bcs_a, self.bcs_t, self.bog_a, self.bog_t = [], [], [], [], [], []
            for ly in lay:
                A = ly.pair_bias_attn; T = ly.transition
                assert isinstance(A.proj_z, torch.nn.Module) and not hasattr(A.proj_z, "weight"), "token layers carry precomputed pair bias (compute_pair_bias=False)"
                self.w_qkvg.append(_W(torch.cat([A.proj_q.weight, A.proj_k.weight, A.proj_v.weight, A.proj_g.weight], 0)))       # [4D, D]
                self.bq.append(A.proj_q.bias.detach().float().contiguous())
                self.w_o.append(_W(A.proj_o.weight))
                self.w_tri.append(_W(torch.cat([T.swish_gate[0].weight, T.a_to_b.weight], 0)))                                    # [2*HID + HID, D]
                self.w_b2a.append(_W(T.b_to_a.weight))                                                                              # [D, HID]
                # AdaLN(a, s) = sigmoid(s_scale(LNw(s))) * LN(a) + s_bias(LNw(s)), LNw(s) = LN0(s) * s_norm.weight: fold the weight into the input dim
                wa, wt = ly.adaln.s_norm.weight, T.adaln.s_norm.weight
                cs_rows += [ly.adaln.s_scale.weight * wa[None, :], ly.adaln.s_bias.weight * wa[None, :], T.adaln.s_scale.weight * wt[None, :], T.adaln.s_bias.weight * wt[None, :]]
                self.bcs_a.append(ly.adaln.s_scale.bias.detach().float().contiguous()); self.bcs_t.append(T.adaln.s_scale.bias.detach().float().contiguous())
                og_rows += [ly.output_projection_linear.weight, T.output_projection[0].weight]
                self.bog_a.append(ly.output_projection_linear.bias.detach().float().contiguous()); self.bog_t.append(T.output_projection[0].bias.detach().float().contiguous())
            self.w_cs = _W(torch.cat(cs_rows, 0))            # [L*4D, D]
            self.w_og = _W(torch.cat(og_rows, 0))            # [L*2D, D]
        self.pack = None                                     # dict(key, pb [L, H, N, N] (bf16 | fp32), N)

    def nbytes(self):
        n = 0
        for lst in (self.w_qkvg, self.w_o, self.w_tri, self.w_b2a, [self.w_cs, self.w_og]):
            for w in lst:
                n += (w.w32.numel() * 4 if w.w32 is not None else 0) + (w.w16.numel() * 2 if w.w16 is not None else 0)
        return n


_STATE = {"tt": None, "st": None}


def prepare(score_model, feats, diffusion_conditioning, multiplicity):
    """Per sample(): pack the 24 token layers' pair bias [B=1, N, N, L*H] -> [L, H, N, N] in the attention's input precision with stock's key-mask
    term (1 - mask) * -inf folded in (stock attentionv2: attn + (1 - mask[:, None, None]) * -self.inf). Called by bz_sampler at sample() entry."""
    if not _CFG["on"]:
        return
    st = _state_for(score_model.token_transformer)
    z = diffusion_conditioning["token_trans_bias"]                       # [B, N, N, L*H]
    mask = feats["token_pad_mask"]                                       # [B, N]
    B, N = int(z.shape[0]), int(z.shape[1])
    if B != 1:
        st.pack = None; return                                           # forward() falls back by name (bias_batch)
    dt = torch.bfloat16 if _CFG["attn"] == "bf16" else torch.float32
    with torch.no_grad():
        zv = z.view(B, N, N, st.L, st.H).permute(3, 0, 4, 1, 2)[:, 0]     # [L, H, N, N] view
        mterm = (1 - mask[0].float())[None, None, :] * -st.inf            # [1, 1, N]: stock's key-mask term
        pb = torch.empty((st.L, st.H, N, N), dtype=dt, device=z.device)
        for l in range(st.L):                                            # layer by layer: the same elementwise fp32 add + cast per element (same bytes as the whole-tensor
            pb[l] = (zv[l].to(dtype=torch.float32) + mterm).to(dt)       # form), with a [H, N, N] fp32 transient instead of an [L, H, N, N] one (2.2 GiB at 1200 tokens)
    st.pack = {"pb": pb, "N": N, "z_ptr": z.data_ptr(), "multiplicity": int(multiplicity)}
    STATS["packs"] += 1; STATS["pack_gib"] = round(pb.numel() * pb.element_size() / 2 ** 30, 3)
    _log(f"packed pair bias [L={st.L},H={st.H},N={N}] {dt} ({STATS['pack_gib']} GiB)")


def release():
    """sample() exit (bz_sampler): drop the per-sample() pair-bias pack; under weights=sample drop the packed weights too (the next prepare() rebuilds them)."""
    st = _STATE["st"]
    if st is not None:
        st.pack = None
        if _CFG["weights"] == "sample":
            _STATE["st"] = None; _STATE["tt"] = None
            STATS["weights_releases"] += 1


def _state_for(tt):
    if _STATE["tt"] is not tt:
        _STATE["st"] = _State(tt); _STATE["tt"] = tt
        STATS["layers"] = _STATE["st"].L; STATS["weights_mib"] = round(_STATE["st"].nbytes() / 2 ** 20); STATS["weights_builds"] += 1
        if STATS["weights_builds"] == 1:
            _log(f"packed weights for the token transformer: {STATS['weights_mib']} MiB (weights={_CFG['weights']})")
    return _STATE["st"]


def _scope(reason, tt, args, kwargs):
    STATS["scope"][reason] = STATS["scope"].get(reason, 0) + 1
    return tt.__class__.forward(tt, *args, **kwargs)


def fused_forward(tt, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
    """DiffusionTransformer.forward for the token instance: 24 fused layers (module doc). Same signature/return as stock."""
    STATS["calls"] += 1
    if not _CFG["on"]:
        return _scope("off", tt, (a, s), dict(bias=bias, mask=mask, to_keys=to_keys, multiplicity=multiplicity))
    if to_keys is not None or not tt.pair_bias_attn or bias is None:
        return _scope("not_token_attention", tt, (a, s), dict(bias=bias, mask=mask, to_keys=to_keys, multiplicity=multiplicity))
    st = _STATE["st"] if _STATE["tt"] is tt else None
    Bm, N, D = a.shape
    if st is None or st.pack is None or st.pack["N"] != N or D != st.D:
        return _scope("no_pack", tt, (a, s), dict(bias=bias, mask=mask, to_keys=to_keys, multiplicity=multiplicity))
    gp, ap = _CFG["gemm"], _CFG["attn"]
    gin = torch.bfloat16 if gp == "bf16" else torch.float32
    M = Bm * N
    x = a.reshape(M, D).float().contiguous()
    s2 = s.reshape(M, D).float().contiguous()
    eps = st.eps
    # ---- all layers' conditioning: 2 GEMMs ----
    sn = torch.empty(M, D, device=x.device, dtype=gin)
    _ln_kernel[(M,)](s2, s2, s2, sn, D=D, BD=triton.next_power_of_2(D), eps=eps, sx=s2.stride(0), so=sn.stride(0), HAS_W=False, HAS_B=False,
                     OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(D))
    cs = _mm(sn, st.w_cs, gp, out_dtype=torch.float32)                  # [M, L*4D]
    og = _mm(s2.to(gin) if gin != torch.float32 else s2, st.w_og, gp, out_dtype=torch.float32)     # [M, L*2D]
    xin = torch.empty(M, D, device=x.device, dtype=gin)
    _adaln_kernel[(M,)](x, cs[:, 0:D], cs[:, D:2 * D], st.bcs_a[0], xin, D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), scs=cs.stride(0),
                        sch=cs.stride(0), so=xin.stride(0), OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(D))
    pb = st.pack["pb"]
    BQ, BKV, nw, ns = _flash_cfg(N, ap)
    PREC = {"bf16": 3, "tf32": 1, "ieee": 0}[ap]
    STATS["flash_cfg"][str((N, ap))] = (BQ, BKV, nw, ns)
    HID = st.HID
    for l in range(st.L):
        c0 = l * 4 * D; g0 = l * 2 * D
        qkvg = _mm(xin, st.w_qkvg[l], gp, out_dtype=gin)                # [M, 4D]
        ctx = torch.empty(M, D, device=x.device, dtype=gin)
        _flash_bias_kernel[(triton.cdiv(N, BQ), st.H, Bm)](qkvg, st.bq[l], pb[l], ctx, N, qkvg.stride(0), pb.stride(1), pb.stride(2), ctx.stride(0), float(st.scale),
                                                              H=st.H, HD=st.HD, D=D, PREC=PREC, OUT_BF16=(gin == torch.bfloat16), BQ=BQ, BKV=BKV, num_warps=nw, num_stages=ns)
        o = _mm(ctx, st.w_o[l], gp, out_dtype=torch.float32)            # [M, D] fp32
        x_new = torch.empty_like(x); xin = torch.empty(M, D, device=x.device, dtype=gin)
        _gate_res_adaln_kernel[(M,)](x, o, og[:, g0:g0 + D], st.bog_a[l], x_new, cs[:, c0 + 2 * D:c0 + 3 * D], cs[:, c0 + 3 * D:c0 + 4 * D], st.bcs_t[l], xin,
                                     D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), sy=o.stride(0), sog=og.stride(0), sxo=x_new.stride(0),
                                     scs=cs.stride(0), sch=cs.stride(0), sao=xin.stride(0), HAS_ADALN=True, OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(D))
        x = x_new
        tri = _mm(xin, st.w_tri[l], gp, out_dtype=gin)                  # [M, 3*HID] = [swish 2*HID | a_to_b HID]
        h = torch.empty(M, HID, device=x.device, dtype=gin)
        _swiglu_ab_kernel[(M,)](tri, h, HID=HID, BH=triton.next_power_of_2(HID), st=tri.stride(0), so=h.stride(0), OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(HID))
        o = _mm(h, st.w_b2a[l], gp, out_dtype=torch.float32)            # [M, D] fp32
        x_new = torch.empty_like(x)
        if l + 1 < st.L:
            c1 = (l + 1) * 4 * D
            xin = torch.empty(M, D, device=x.device, dtype=gin)
            _gate_res_adaln_kernel[(M,)](x, o, og[:, g0 + D:g0 + 2 * D], st.bog_t[l], x_new, cs[:, c1:c1 + D], cs[:, c1 + D:c1 + 2 * D], st.bcs_a[l + 1], xin,
                                         D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), sy=o.stride(0), sog=og.stride(0), sxo=x_new.stride(0),
                                         scs=cs.stride(0), sch=cs.stride(0), sao=xin.stride(0), HAS_ADALN=True, OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(D))
        else:
            _gate_res_adaln_kernel[(M,)](x, o, og[:, g0 + D:g0 + 2 * D], st.bog_t[l], x_new, cs, cs, st.bcs_a[0], xin,
                                         D=D, BD=triton.next_power_of_2(D), eps=eps, sx=x.stride(0), sy=o.stride(0), sog=og.stride(0), sxo=x_new.stride(0),
                                         scs=cs.stride(0), sch=cs.stride(0), sao=xin.stride(0), HAS_ADALN=False, OUT_BF16=(gin == torch.bfloat16), num_warps=_nw(D))
        x = x_new
    STATS["served"] += 1
    return x.view(Bm, N, D).to(a.dtype)


def install(words, score_model_cls=None):
    """words: list like ['bf16'] or ['bf16', 'attn=tf32', 'gemm=tf32']. Installs nothing on modules yet: the token-transformer INSTANCE is patched at
    the first sample() (bz_sampler calls attach(score_model)), so class-level patches by other levers (the kit hoist) never displace it."""
    if not _HAVE_TRITON:
        raise RuntimeError("dit_fused: triton is not importable in this process")
    gemm, attn, attn_given, weights = "bf16", "bf16", False, "resident"
    for w in words:
        w = w.strip().lower()
        if w in ("bf16", "1", "on"):
            continue
        if w.startswith("weights="):                         # the packed weights' lifetime: resident (default) | sample (built / dropped per sample(): the memory row)
            weights = w.split("=", 1)[1]
            if weights not in ("resident", "sample"):
                raise RuntimeError(f"dit_fused: word {w!r} is not a value (weights=resident|sample)")
            continue
        if w == "tf32":                                      # the higher-precision word: TF32 GEMMs on fp32 data AND tf32 attention products
            gemm, attn = "tf32", ("tf32" if not attn_given else attn)
        elif w.startswith("gemm="):
            gemm = w.split("=", 1)[1]
        elif w.startswith("attn="):
            attn = w.split("=", 1)[1]; attn_given = True
        else:
            raise RuntimeError(f"dit_fused: word {w!r} is not a value (bf16 | tf32 | gemm=bf16|tf32 | attn=bf16|tf32|ieee | weights=resident|sample; activation dtype must equal the attention operand dtype)")
    if gemm not in ("bf16", "tf32") or attn not in ("bf16", "tf32", "ieee"):
        raise RuntimeError(f"dit_fused: gemm={gemm} attn={attn}: not a value")
    if gemm == "tf32" and attn == "bf16" and not attn_given:
        attn = "tf32"                                        # gemm=tf32 alone implies tf32 attention
    if (gemm == "bf16") != (attn == "bf16"):
        # the attention kernel's MMA operands are the loaded activation tiles AS LOADED: bf16 activations <-> bf16 products, fp32 activations <->
        # tf32 | ieee products. A dtype conversion of a loaded tile in front of tl.dot (fp32->bf16 or bf16->fp32) mis-compiles on triton 3.7 /
        # sm90 for several tile configurations (wrong values, NaN or an illegal address, configuration-dependent) — those combinations are refused.
        raise RuntimeError(f"dit_fused: gemm={gemm} with attn={attn} is not a supported combination (activation dtype must equal the attention operand dtype: "
                           "bf16 | tf32 | gemm=tf32,attn=ieee) — refused")
    STATS["out_dtype_probe"] = _has_out_dtype()
    if gemm == "bf16" and not STATS["out_dtype_probe"]:
        raise RuntimeError("dit_fused: torch.mm(out_dtype=) is not available on this torch build (needed for bf16 GEMMs with fp32 outputs)")
    _CFG.update(gemm=gemm, attn=attn, on=True, weights=weights)
    STATS.update(installed={"gemm": gemm, "attn": attn, "weights": weights}, gemm=gemm, attn=attn, weights=weights)
    return [f"dit_fused=gemm:{gemm},attn:{attn},weights:{weights}"]


def _has_out_dtype():
    try:
        a = torch.zeros(8, 8, device="cuda", dtype=torch.bfloat16)
        torch.mm(a, a, out_dtype=torch.float32)
        return True
    except Exception:  # noqa: BLE001
        return False


def attach(score_model):
    """Patch the token transformer INSTANCE of this DiffusionModule (idempotent)."""
    if not _CFG["on"]:
        return False
    tt = score_model.token_transformer
    if getattr(tt, "_bzs_dit", False):
        return True
    if _CFG["weights"] == "resident":
        _state_for(tt)                                       # weights=sample: prepare() builds the pack at sample() entry, release() drops it at exit
    import types
    tt.forward = types.MethodType(fused_forward, tt)
    tt._bzs_dit = True
    _log("attached fused forward to", type(tt).__name__, "layers:", len(tt.layers))
    return True


def report():
    return {"stats": dict(STATS), "cfg": dict(_CFG)}
