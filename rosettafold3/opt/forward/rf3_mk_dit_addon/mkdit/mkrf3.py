"""mkrf3.py — RF3 token DiffusionTransformer (24 x [AdaLN -> attention w/ hoisted pair bias -> gate/residual -> AdaLN -> SwiGLU transition -> gate/residual])
as persistent per-row-tile Triton kernels.  Every CTA owns BM token rows for the whole row-local chain; the only grid-wide dependency in a block
(all K/V rows must exist before attention) is the single launch boundary, so one 24-block call = 1 + 24 fused launches (+3 cuBLAS/LN launches for the
per-step S-side precompute) instead of ~100 launches per block.  Precision policy mirrors RF3 bf16-mixed (TF32 off): bf16 MMA operands, fp32 accumulate,
fp32 LayerNorm/softmax/residual stream, bf16 rounding at the same points as the stock autocast graph (see comments 'stock:').
Deterministic: fixed tiling, no atomics, no split-K."""
import math, os, torch
import triton, triton.language as tl

# ----------------------------------------------------------------------------------------------------------------- device helpers
@triton.jit
def _ln_stats(X, ld, rows, rmask, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, IS_RMS: tl.constexpr, eps):
    """two-pass mean / rstd over C features for BM rows; tiles are feature-major [CB, BM]; fp32."""
    s1 = tl.zeros([BM], dtype=tl.float32)
    if IS_RMS:
        mean = tl.zeros([BM], dtype=tl.float32)
    else:
        for c0 in range(0, C, CB):
            cs = c0 + tl.arange(0, CB)
            x = tl.load(X + rows[None, :] * ld + cs[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
            s1 += tl.sum(x, axis=0)
        mean = s1 / C
    s2 = tl.zeros([BM], dtype=tl.float32)
    for c0 in range(0, C, CB):
        cs = c0 + tl.arange(0, CB)
        x = tl.load(X + rows[None, :] * ld + cs[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        d = x - mean[None, :]
        s2 += tl.sum(d * d, axis=0)
    rstd = 1.0 / tl.sqrt(s2 / C + eps)
    return mean, rstd


@triton.jit
def _mm_wxT(acc, W, ldw, ns, nmask, X, ldx, rows, rmask, K: tl.constexpr, BK: tl.constexpr,
            XMODE: tl.constexpr, mean, rstd, lnw, lnb, LNW: tl.constexpr, LNB: tl.constexpr):
    """acc[BN, BM] += W[ns, 0:K] @ f(X[rows, 0:K])^T .  XMODE 0: X bf16 as-is; 1: X fp32 -> bf16; 2: fp32 LN-normalise (mean/rstd [*lnw] [+lnb]) -> bf16;
    3: X bf16 -> fp32 LN-normalise (stats given) [*lnw][+lnb] -> bf16 (kq-norm path)."""
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + ns[:, None] * ldw + ks[None, :], mask=nmask[:, None], other=0.0)
        x = tl.load(X + rows[None, :] * ldx + ks[:, None], mask=rmask[None, :], other=0.0)
        if XMODE == 2 or XMODE == 3:
            xf = (x.to(tl.float32) - mean[None, :]) * rstd[None, :]
            if LNW:
                xf = xf * tl.load(lnw + ks).to(tl.float32)[:, None]
            if LNB:
                xf = xf + tl.load(lnb + ks).to(tl.float32)[:, None]
            xb = xf.to(tl.bfloat16)
        elif XMODE == 1:
            xb = x.to(tl.bfloat16)
        else:
            xb = x
        acc = tl.dot(w, xb, acc)
    return acc


@triton.jit
def _phase_adaln(A, lda, GB, ldgb, gcol0, bcol0, XOUT, ldxo, rows, rmask, mean, rstd,
                 CA: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
    """XOUT[rows, :] = bf16( sigmoid(GB[rows, gcol0+n]) * LN(A)[rows, n] + GB[rows, bcol0+n] )   (stock: to_gain(Si) * ln_a(Ai) + to_bias(Si), then .to(bf16))"""
    for n0 in range(0, CA, BN):
        ns = n0 + tl.arange(0, BN)
        a = tl.load(A + rows[None, :] * lda + ns[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        aln = (a - mean[None, :]) * rstd[None, :]
        g = tl.load(GB + rows[None, :] * ldgb + (gcol0 + ns)[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        g = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)                       # stock: Sigmoid on the bf16 Linear output -> bf16
        b = tl.load(GB + rows[None, :] * ldgb + (bcol0 + ns)[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        xo = (g * aln + b).to(tl.bfloat16)
        tl.store(XOUT + rows[None, :] * ldxo + ns[:, None], xo, mask=rmask[None, :])


@triton.jit
def _phase_qkvg(X1, ldx, Wqkvg, QKVG, ldq, qoff, koff, voff, goff, rows, rmask, qscale,
                CA: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, KQN: tl.constexpr):
    """Q|K|V|G[rows] = X1[rows] @ Wqkvg^T (Wqkvg [4*CA, CA] bf16).  stock: four Linear (bf16 out); G -> sigmoid (bf16); Q -> /sqrt(c) (bf16) unless kq_norm."""
    dz = tl.zeros([BM], dtype=tl.float32)
    for which in tl.static_range(4):
        for c0 in range(0, CA, BN):
            col = c0 + tl.arange(0, BN); ns = which * CA + col; nmask = col < CA
            acc = tl.zeros([BN, BM], dtype=tl.float32)
            acc = _mm_wxT(acc, Wqkvg, CA, ns, nmask, X1, ldx, rows, rmask, CA, BK, 0, dz, dz, dz, dz, False, False)
            x = acc.to(tl.bfloat16)
            if which == 3:
                x = tl.sigmoid(x.to(tl.float32)).to(tl.bfloat16)
                tl.store(QKVG + goff + rows[None, :] * ldq + col[:, None], x, mask=rmask[None, :])
            elif which == 2:
                tl.store(QKVG + voff + rows[None, :] * ldq + col[:, None], x, mask=rmask[None, :])
            elif which == 1:
                tl.store(QKVG + koff + rows[None, :] * ldq + col[:, None], x, mask=rmask[None, :])
            else:
                if not KQN:
                    x = (x.to(tl.float32) * qscale).to(tl.bfloat16)               # stock: Q_IH / np.sqrt(self.c) on the bf16 tensor
                tl.store(QKVG + qoff + rows[None, :] * ldq + col[:, None], x, mask=rmask[None, :])


@triton.jit
def _phase_kqnorm(QKVG, ldq, off, rows, rmask, lnw, lnb, scale, eps, CA: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
    """in place: X[rows] = bf16( LayerNorm_fp32(X_bf16[rows]) * scale )  (stock: query/key_layer_norm over H*c under autocast -> fp32, / sqrt(c), cast at einsum)"""
    mean, rstd = _ln_stats(QKVG + off, ldq, rows, rmask, CA, BN, BM, False, eps)
    for n0 in range(0, CA, BN):
        ns = n0 + tl.arange(0, BN)
        p = QKVG + off + rows[None, :] * ldq + ns[:, None]
        x = tl.load(p, mask=rmask[None, :], other=0.0).to(tl.float32)
        y = (x - mean[None, :]) * rstd[None, :] * tl.load(lnw + ns).to(tl.float32)[:, None] + tl.load(lnb + ns).to(tl.float32)[:, None]
        tl.store(p, (y * scale).to(tl.bfloat16), mask=rmask[None, :])


@triton.jit
def _pre_and_qkvg(A, lda, GB, ldgb, gcol0, bcol0, X1, Wqkvg, QKVG, ldq, qoff, koff, voff, goff, rows, rmask, qscale, eps_a,
                  qlnw, qlnb, klnw, klnb, eps_kq,
                  CA: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, LNA_RMS: tl.constexpr, KQN: tl.constexpr):
    mean, rstd = _ln_stats(A, lda, rows, rmask, CA, BN, BM, LNA_RMS, eps_a)
    _phase_adaln(A, lda, GB, ldgb, gcol0, bcol0, X1, CA, rows, rmask, mean, rstd, CA, BN, BM)
    tl.debug_barrier()
    _phase_qkvg(X1, CA, Wqkvg, QKVG, ldq, qoff, koff, voff, goff, rows, rmask, qscale, CA, BN, BK, BM, KQN)
    if KQN:
        tl.debug_barrier()
        _phase_kqnorm(QKVG, ldq, qoff, rows, rmask, qlnw, qlnb, qscale, eps_kq, CA, BN, BM)
        _phase_kqnorm(QKVG, ldq, koff, rows, rmask, klnw, klnb, 1.0, eps_kq, CA, BN, BM)


# ----------------------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _k_pre(A, GB, ldgb, gcol0, bcol0, X1, Wqkvg, QKVG, qoff, koff, voff, goff, I, qscale, eps_a, qlnw, qlnb, klnw, klnb, eps_kq,
           CA: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, LNA_RMS: tl.constexpr, KQN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM); rmask = rows < I
    _pre_and_qkvg(A, CA, GB, ldgb, gcol0, bcol0, X1, Wqkvg, QKVG, CA, qoff, koff, voff, goff, rows, rmask, qscale, eps_a,
                  qlnw, qlnb, klnw, klnb, eps_kq, CA, BN, BK, BM, LNA_RMS, KQN)


@triton.jit
def _k_block(A, A1S, S_unused, Bt, GB, ldgb, LOP, ldlop, blk4, blk2, nb4, nb2,
             QKVG, pin_off, pout_off, qo, ko, vo, go, X1, ATT, HS, Wa, W1, W2, W3, Wqkvg_next,
             I, qscale, eps_a, eps_t, qlnw, qlnb, klnw, klnb, eps_kq,
             CA: tl.constexpr, NH: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, NT: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, BNK: tl.constexpr,
             LNA_RMS: tl.constexpr, KQN: tl.constexpr, NORES: tl.constexpr, HAS_NEXT: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM); rmask = rows < I
    ds = tl.arange(0, DHP); dmask = ds < DH
    dz = tl.zeros([BM], dtype=tl.float32)
    # ---------------- attention with hoisted pair bias, all heads for these query rows (stock: softmax(bf16(QK^T)+B) fp32, P->bf16, PV bf16, *G)
    for h in range(NH):
        hc = h * DH + ds
        q = tl.load(QKVG + pin_off + qo + rows[:, None] * CA + hc[None, :], mask=rmask[:, None] & dmask[None, :], other=0.0)
        m_i = tl.zeros([BM], dtype=tl.float32) + float("-inf")
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, DHP], dtype=tl.float32)
        for j0 in range(0, I, BNK):
            js = j0 + tl.arange(0, BNK); jmask = js < I
            k = tl.load(QKVG + pin_off + ko + js[:, None] * CA + hc[None, :], mask=jmask[:, None] & dmask[None, :], other=0.0)
            s = tl.dot(q, tl.trans(k))
            b = tl.load(Bt + h.to(tl.int64) * I * I + rows[:, None] * I + js[None, :], mask=rmask[:, None] & jmask[None, :], other=0.0)   # head-plane offset h*I*I in int64 (int32 wraps from I = 11,586)
            s = (s.to(tl.bfloat16).to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            s = tl.where(jmask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(QKVG + pin_off + vo + js[:, None] * CA + hc[None, :], mask=jmask[:, None] & dmask[None, :], other=0.0)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            m_i = m_new
        o = (acc / l_i[:, None]).to(tl.bfloat16).to(tl.float32)
        g = tl.load(QKVG + pin_off + go + rows[:, None] * CA + hc[None, :], mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        tl.store(ATT + rows[:, None] * CA + hc[None, :], (o * g).to(tl.bfloat16), mask=rmask[:, None] & dmask[None, :])
    tl.debug_barrier()
    # ---------------- to_a + sigmoid(lop1) gate + residual  (A1 = A + gate*to_a(att));  A1 -> A (in place) or A1S (NORES)
    for n0 in range(0, CA, BN):
        ns = n0 + tl.arange(0, BN); nmask = ns < CA
        acc2 = tl.zeros([BN, BM], dtype=tl.float32)
        acc2 = _mm_wxT(acc2, Wa, CA, ns, nmask, ATT, CA, rows, rmask, CA, BK, 0, dz, dz, dz, dz, False, False)
        x = acc2.to(tl.bfloat16).to(tl.float32)
        gate = tl.load(LOP + rows[None, :] * ldlop + (blk2 * CA + ns)[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
        out = (gate * x).to(tl.bfloat16).to(tl.float32)
        a = tl.load(A + rows[None, :] * CA + ns[:, None], mask=rmask[None, :], other=0.0)
        if NORES:
            tl.store(A1S + rows[None, :] * CA + ns[:, None], a + out, mask=rmask[None, :])
        else:
            tl.store(A + rows[None, :] * CA + ns[:, None], a + out, mask=rmask[None, :])
    tl.debug_barrier()
    # ---------------- conditioned transition on TIN (= A1, or the original A when NORES): AdaLN -> X1 ; silu(W1 x) * (W2 x) -> HS ; W3 -> gate -> residual
    mean2, rstd2 = _ln_stats(A, CA, rows, rmask, CA, BN, BM, LNA_RMS, eps_t)
    _phase_adaln(A, CA, GB, ldgb, (blk4 + 2) * CA, (blk4 + 3) * CA, X1, CA, rows, rmask, mean2, rstd2, CA, BN, BM)
    tl.debug_barrier()
    for n0 in range(0, NT * CA, BN):
        ns = n0 + tl.arange(0, BN); nmask = ns < NT * CA
        a1 = tl.zeros([BN, BM], dtype=tl.float32)
        a1 = _mm_wxT(a1, W1, CA, ns, nmask, X1, CA, rows, rmask, CA, BK, 0, dz, dz, dz, dz, False, False)
        a2 = tl.zeros([BN, BM], dtype=tl.float32)
        a2 = _mm_wxT(a2, W2, CA, ns, nmask, X1, CA, rows, rmask, CA, BK, 0, dz, dz, dz, dz, False, False)
        x1 = a1.to(tl.bfloat16).to(tl.float32); x2 = a2.to(tl.bfloat16).to(tl.float32)
        sx = (x1 * tl.sigmoid(x1)).to(tl.bfloat16).to(tl.float32)                 # stock: F.silu on the bf16 tensor -> bf16
        hval = (sx * x2).to(tl.bfloat16)
        tl.store(HS + rows[None, :] * (NT * CA) + ns[:, None], hval, mask=rmask[None, :])
    tl.debug_barrier()
    for n0 in range(0, CA, BN):
        ns = n0 + tl.arange(0, BN); nmask = ns < CA
        a3 = tl.zeros([BN, BM], dtype=tl.float32)
        a3 = _mm_wxT(a3, W3, NT * CA, ns, nmask, HS, NT * CA, rows, rmask, NT * CA, BK, 0, dz, dz, dz, dz, False, False)
        x3 = a3.to(tl.bfloat16).to(tl.float32)
        gate = tl.load(LOP + rows[None, :] * ldlop + ((blk2 + 1) * CA + ns)[:, None], mask=rmask[None, :], other=0.0).to(tl.float32)
        gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
        out = (gate * x3).to(tl.bfloat16).to(tl.float32)
        if NORES:
            a1v = tl.load(A1S + rows[None, :] * CA + ns[:, None], mask=rmask[None, :], other=0.0)
        else:
            a1v = tl.load(A + rows[None, :] * CA + ns[:, None], mask=rmask[None, :], other=0.0)
        tl.store(A + rows[None, :] * CA + ns[:, None], a1v + out, mask=rmask[None, :])
    # ---------------- fused prologue of the NEXT block on the same rows: AdaLN(A) -> X1 -> Q|K|V|G into the other QKVG buffer
    if HAS_NEXT:
        tl.debug_barrier()
        _pre_and_qkvg(A, CA, GB, ldgb, (blk4 + nb4) * CA, (blk4 + nb4 + 1) * CA, X1, Wqkvg_next, QKVG + pout_off, CA, qo, ko, vo, go, rows, rmask, qscale, eps_a,
                      qlnw, qlnb, klnw, klnb, eps_kq, CA, BN, BK, BM, LNA_RMS, KQN)


# ----------------------------------------------------------------------------------------------------------------- host side
def _bf(x): return x.detach().to(torch.bfloat16).contiguous()

class MKTokenTransformer:
    """Wraps a stock rf3 DiffusionTransformer (token level, Beta_II=None).  prepare() reads its weights; hoist(Z) builds the per-block pair-bias
    cache (stock ln_0/to_b under the caller's autocast, permuted to [H, I, I]); forward(A_I, S_I) runs the fused kernels."""
    def __init__(self, tok, BM=16, BN=128, BK=64, BNK=64, num_warps=4, sfold=True):
        self.tok = tok; self.BM, self.BN, self.BK, self.BNK, self.nw = BM, BN, BK, BNK, num_warps
        b0 = tok.blocks[0]; apb = b0.attention_pair_bias; ctb = b0.conditioned_transition_block
        self.NB = len(tok.blocks); self.CA = apb.c_a; self.NH = apb.n_head; self.DH = apb.c; self.DHP = triton.next_power_of_2(self.DH)
        self.NT = ctb.linear_1.out_features // ctb.linear_1.in_features
        self.CS = apb.ada_ln_1.to_bias.in_features
        self.KQN = bool(apb.kq_norm); self.NORES = bool(b0.no_residual_connection_between_attention_and_transition)
        ln_a = apb.ada_ln_1.ln_a; ln_s = apb.ada_ln_1.ln_s
        self.LNA_RMS = "RMS" in type(ln_a).__name__; self.LNS_RMS = "RMS" in type(ln_s).__name__
        self.eps_a = float(getattr(ln_a, "eps", 1e-5) or 1e-5); self.eps_t = float(getattr(ctb.ada_ln.ln_a, "eps", 1e-5) or 1e-5); self.eps_s = float(getattr(ln_s, "eps", 1e-5) or 1e-5)
        assert self.CA % BN == 0 and self.CA % BK == 0 and (self.NT * self.CA) % BN == 0, (self.CA, self.NT)
        assert getattr(ln_a, "weight", None) is None or ln_a.weight is None or not ln_a.elementwise_affine, "ln_a affine not supported"
        dev = apb.to_q.weight.device
        W = {"qkvg": [], "a": [], "w1": [], "w2": [], "w3": []}; GBW = []; GBb = []; LOPW = []; LOPb = []
        for b in tok.blocks:
            p = b.attention_pair_bias; t = b.conditioned_transition_block
            W["qkvg"].append(_bf(torch.cat([p.to_q.weight, p.to_k.weight, p.to_v.weight, p.to_g[0].weight], 0)))
            W["a"].append(_bf(p.to_a.weight)); W["w1"].append(_bf(t.linear_1.weight)); W["w2"].append(_bf(t.linear_2.weight)); W["w3"].append(_bf(t.linear_3.weight))
            for ada in (p.ada_ln_1, t.ada_ln):
                ws = ada.ln_s.weight.detach().float() if getattr(ada.ln_s, "weight", None) is not None else torch.ones(self.CS, device=dev)
                assert getattr(ada.ln_s, "bias", None) is None, "ln_s bias not supported"
                wg = ada.to_gain[0].weight.detach().float(); wb = ada.to_bias.weight.detach().float()
                if sfold:
                    wg = wg * ws[None, :]; wb = wb * ws[None, :]
                GBW += [wg, wb]; GBb += [ada.to_gain[0].bias.detach().float(), torch.zeros(self.CA, device=dev)]
            for lop in (p.linear_output_project[0], t.linear_output_project[0]):
                LOPW.append(lop.weight.detach().float()); LOPb.append(lop.bias.detach().float())
        self.sfold = sfold
        self.lns_w = [(_a.ln_s.weight.detach().float() if getattr(_a.ln_s, "weight", None) is not None else None) for b in tok.blocks for _a in (b.attention_pair_bias.ada_ln_1, b.conditioned_transition_block.ada_ln)]
        self.W = {k: torch.stack(v, 0).contiguous() for k, v in W.items()}           # [NB, ...] bf16
        self.GBW = _bf(torch.cat(GBW, 0)); self.GBb = _bf(torch.cat(GBb, 0)); self.LOPW = _bf(torch.cat(LOPW, 0)); self.LOPb = _bf(torch.cat(LOPb, 0))
        if self.KQN:
            self.kqn = [(p.query_layer_norm.weight.detach().float().contiguous(), p.query_layer_norm.bias.detach().float().contiguous(),
                         p.key_layer_norm.weight.detach().float().contiguous(), p.key_layer_norm.bias.detach().float().contiguous(), float(p.query_layer_norm.eps))
                        for p in (b.attention_pair_bias for b in tok.blocks)]
        self.qscale = 1.0 / math.sqrt(self.DH)
        self.Bt = None; self._scratch = {}
        self.n_params_M = sum(v.numel() for v in self.W.values()) / 1e6

    def hoist(self, Z_II):
        """per-block pair bias exactly as stock (to_b(ln_0(Z)) under the active autocast) -> [NB, H, I, I] bf16 contiguous"""
        outs = []
        for b in self.tok.blocks:
            p = b.attention_pair_bias
            B = p.to_b(p.ln_0(Z_II))                       # [.., I, I, H]  (bf16 under autocast)
            B = B.reshape(-1, B.shape[-3], B.shape[-2], B.shape[-1])
            assert B.shape[0] == 1, "diffusion batch > 1 not supported by the fused path"
            outs.append(B[0].permute(2, 0, 1).to(torch.bfloat16).contiguous())
        self.Bt = torch.stack(outs, 0).contiguous(); self.I = self.Bt.shape[-1]
        return self.Bt

    def _buf(self, I, dev):
        key = (I, str(dev))
        if key not in self._scratch:
            CA, NT = self.CA, self.NT
            self._scratch[key] = dict(QKVG=torch.empty(2, 4, I, CA, dtype=torch.bfloat16, device=dev), X1=torch.empty(I, CA, dtype=torch.bfloat16, device=dev),
                                      ATT=torch.empty(I, CA, dtype=torch.bfloat16, device=dev), HS=torch.empty(I, NT * CA, dtype=torch.bfloat16, device=dev),
                                      A=torch.empty(I, CA, dtype=torch.float32, device=dev), A1S=torch.empty(I, CA, dtype=torch.float32, device=dev))
        return self._scratch[key]

    def s_precompute(self, S):
        """per step: GB = [gain|bias pre-activations of both AdaLNs of every block] and LOP = [both output-gate pre-activations of every block] (cuBLAS, bf16 out)"""
        S2 = S.reshape(-1, S.shape[-1]).float()
        if self.LNS_RMS:
            Sn = S2 * torch.rsqrt(S2.pow(2).mean(-1, keepdim=True) + self.eps_s)
        else:
            Sn = torch.nn.functional.layer_norm(S2, (self.CS,), None, None, self.eps_s)
        GB = torch.addmm(self.GBb, Sn.to(torch.bfloat16), self.GBW.t())            # [I, NB*4*CA] bf16 (stock: to_gain/to_bias Linear on bf16(ln_s(S)))
        LOP = torch.addmm(self.LOPb, S2.to(torch.bfloat16), self.LOPW.t())        # [I, NB*2*CA] bf16 (stock: linear_output_project Linear on bf16(S))
        return GB, LOP

    def forward(self, A_I, S_I, Z_II=None, Beta_II=None):
        assert Beta_II is None
        shp = A_I.shape; CA = self.CA
        A2 = A_I.reshape(-1, CA); S2 = S_I.reshape(-1, S_I.shape[-1]); I = A2.shape[0]
        if self.Bt is None or self.Bt.shape[-1] != I:
            assert Z_II is not None; self.hoist(Z_II)
        dev = A2.device; buf = self._buf(I, dev)
        A = buf["A"]; A.copy_(A2)                                                    # fp32 residual stream (stock keeps A_I fp32)
        GB, LOP = self.s_precompute(S2)
        ldgb, ldlop = GB.stride(0), LOP.stride(0)
        BM, BN, BK, BNK, nw = self.BM, self.BN, self.BK, self.BNK, self.nw
        grid = (triton.cdiv(I, BM),)
        QKVG = buf["QKVG"]; pstride = QKVG.stride(0); qo, ko, vo, go = 0, QKVG.stride(1), 2 * QKVG.stride(1), 3 * QKVG.stride(1)
        W = self.W
        if self.KQN:
            qlnw, qlnb, klnw, klnb, eps_kq = self.kqn[0]
        else:
            qlnw = qlnb = klnw = klnb = self.GBb; eps_kq = 1e-5
        _k_pre[grid](A, GB, ldgb, 0, CA, buf["X1"], W["qkvg"][0], QKVG, qo, ko, vo, go, I, self.qscale, self.eps_a, qlnw, qlnb, klnw, klnb, eps_kq,
                     CA=CA, BN=BN, BK=BK, BM=BM, LNA_RMS=self.LNA_RMS, KQN=self.KQN, num_warps=nw)
        for b in range(self.NB):
            nxt = b + 1 < self.NB
            if self.KQN and nxt:
                qlnw, qlnb, klnw, klnb, eps_kq = self.kqn[b + 1]
            _k_block[grid](A, buf["A1S"], A, self.Bt[b], GB, ldgb, LOP, ldlop, 4 * b, 2 * b, 4, 2,
                           QKVG, (b % 2) * pstride, ((b + 1) % 2) * pstride, qo, ko, vo, go, buf["X1"], buf["ATT"], buf["HS"],
                           W["a"][b], W["w1"][b], W["w2"][b], W["w3"][b], W["qkvg"][b + 1] if nxt else W["qkvg"][b],
                           I, self.qscale, self.eps_a, self.eps_t, qlnw, qlnb, klnw, klnb, eps_kq,
                           CA=CA, NH=self.NH, DH=self.DH, DHP=self.DHP, NT=self.NT, BN=BN, BK=BK, BM=BM, BNK=BNK,
                           LNA_RMS=self.LNA_RMS, KQN=self.KQN, NORES=self.NORES, HAS_NEXT=nxt, num_warps=nw)
        return A.reshape(shp).clone()

    def launches_per_call(self):
        return {"s_precompute": 3 + (0 if self.LNS_RMS else 0), "fused": 1 + self.NB}
