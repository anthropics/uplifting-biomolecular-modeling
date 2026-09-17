"""mk2.py — RF3 token DiffusionTransformer as ONE persistent dataflow megakernel launch per 24-block call (Triton).
grid = #SMs persistent CTAs; a static global item list (block -> phase -> tile) is walked round-robin; dependencies between items are
counters in global memory (release/acquire atomics = scheduling only; every output element is produced by exactly one item with a fixed
K order -> deterministic, no atomics in numerics).  Phases per block b (R = row tiles of BMR=64 rows, N tiles of 128):
  STAT(r)      : LN stats of A rows -> X1 = AdaLN1(A) , X2 = AdaLN2(A) (bf16)          [RF3: transition reads the block INPUT (no_residual...=True)]
  QKVG(r,n)    : X1 @ Wqkvg^T tile -> Q|K|V|G (G sigmoid)                               24 tiles per r
  KQN(r)       : LayerNorm(Q)*1/sqrt(c), LayerNorm(K) in place (kq_norm=True)
  TR1(r,n)     : silu(X2 @ W1^T) * (X2 @ W2^T) tile -> HS                                12 tiles per r (independent of the attention chain)
  ATT(r,h)     : flash attention w/ hoisted [H,I,I] bias, gated -> ATT                   waits the KQN join of ALL rows
  OUT1(r,n)    : A_next = A + sigmoid(lop1) * (ATT @ Wa^T)                               6 tiles per r
  OUT2(r,n)    : A_next += sigmoid(lop2) * (HS @ W3^T)                                   6 tiles per r  -> DONE(b,r) gates STAT(b+1,r)
Precision policy = stock RF3 bf16-mixed (see mkrf3.py comments).  Cross-CTA buffers are read with cache_modifier='.cg'."""
import math, torch, triton, triton.language as tl
import mkrf3

K_STAT, K_QKVG, K_KQN, K_TR1, K_ATT, K_OUT1, K_DONE = 0, 1, 2, 3, 4, 5, 6
NKIND = 8

@triton.jit
def _wait(CNT, idx, target, POLL: tl.constexpr):
    if POLL:
        c = tl.load(CNT + idx, volatile=True)
        while c < target:
            c = tl.load(CNT + idx, volatile=True)
        tl.atomic_add(CNT + idx, 0, sem="acquire")
    else:
        c = tl.atomic_add(CNT + idx, 0, sem="acquire")
        while c < target:
            c = tl.atomic_add(CNT + idx, 0, sem="acquire")
    tl.debug_barrier()

@triton.jit
def _signal(CNT, idx):
    tl.debug_barrier()
    tl.atomic_add(CNT + idx, 1, sem="release")

@triton.jit
def _cidx(b, kind, r, R, NKIND: tl.constexpr):
    return (b * NKIND + kind) * R + r

# ------------------------------------------------------------------------------------------------ phases
@triton.jit
def _ph_stat(Acur, GB, ldgb, gcol, X1, X2, row0, I, eps, CA: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr):
    rows = row0 + tl.arange(0, BMR); rm = rows < I
    s1 = tl.zeros([BMR], dtype=tl.float32)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        a = tl.load(Acur + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg")
        s1 += tl.sum(a, axis=1)
    mean = s1 / CA
    s2 = tl.zeros([BMR], dtype=tl.float32)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        a = tl.load(Acur + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg")
        dlt = a - mean[:, None]; s2 += tl.sum(dlt * dlt, axis=1)
    rstd = 1.0 / tl.sqrt(s2 / CA + eps)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        a = tl.load(Acur + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg")
        aln = (a - mean[:, None]) * rstd[:, None]
        g1 = tl.load(GB + rows[:, None] * ldgb + (gcol + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        b1 = tl.load(GB + rows[:, None] * ldgb + (gcol + CA + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        g2 = tl.load(GB + rows[:, None] * ldgb + (gcol + 2 * CA + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        b2 = tl.load(GB + rows[:, None] * ldgb + (gcol + 3 * CA + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        g1 = tl.sigmoid(g1).to(tl.bfloat16).to(tl.float32); g2 = tl.sigmoid(g2).to(tl.bfloat16).to(tl.float32)
        tl.store(X1 + rows[:, None] * CA + cs[None, :], (g1 * aln + b1).to(tl.bfloat16), mask=rm[:, None])
        tl.store(X2 + rows[:, None] * CA + cs[None, :], (g2 * aln + b2).to(tl.bfloat16), mask=rm[:, None])

@triton.jit
def _gemm_tile(X, ldx, W, ldw, rows, rm, ncols, K: tl.constexpr, BK: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr):
    """acc[BMR, BN] = X[rows, 0:K] (bf16, cross-CTA -> .cg) @ W[ncols, 0:K]^T   (fp32 accumulate, fixed k order)"""
    acc = tl.zeros([BMR, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * ldx + ks[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg")
        w = tl.load(W + ncols[:, None] * ldw + ks[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    return acc

@triton.jit
def _ph_qkvg(X1, Wb, QKVGp, tstride, r, n, I, qscale, CA: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KQN: tl.constexpr, NPER: tl.constexpr):
    rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    ncols = n * BN + tl.arange(0, BN)                      # row index into Wqkvg [4*CA, CA]
    acc = _gemm_tile(X1, CA, Wb, CA, rows, rm, ncols, CA, BK, BMR, BN)
    which = n // NPER
    cols = (n - which * NPER) * BN + tl.arange(0, BN)
    x = acc.to(tl.bfloat16)
    if which == 3:
        x = tl.sigmoid(x.to(tl.float32)).to(tl.bfloat16)
    if which == 0:
        if not KQN:
            x = (x.to(tl.float32) * qscale).to(tl.bfloat16)
    tl.store(QKVGp + which * tstride + rows[:, None] * CA + cols[None, :], x, mask=rm[:, None])

@triton.jit
def _ln_inplace_bf16(P, rows, rm, w, bvec, scale, eps, CA: tl.constexpr, CB: tl.constexpr, BMR: tl.constexpr):
    s1 = tl.zeros([BMR], dtype=tl.float32)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        x = tl.load(P + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        s1 += tl.sum(x, axis=1)
    mean = s1 / CA
    s2 = tl.zeros([BMR], dtype=tl.float32)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        x = tl.load(P + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        dlt = x - mean[:, None]; s2 += tl.sum(dlt * dlt, axis=1)
    rstd = 1.0 / tl.sqrt(s2 / CA + eps)
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        ptr = P + rows[:, None] * CA + cs[None, :]
        x = tl.load(ptr, mask=rm[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        y = ((x - mean[:, None]) * rstd[:, None] * tl.load(w + cs)[None, :] + tl.load(bvec + cs)[None, :]) * scale
        tl.store(ptr, y.to(tl.bfloat16), mask=rm[:, None])

@triton.jit
def _ph_kqn(QKVGp, tstride, KQNW, row0, I, qscale, eps, CA: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr):
    rows = row0 + tl.arange(0, BMR); rm = rows < I
    _ln_inplace_bf16(QKVGp, rows, rm, KQNW, KQNW + CA, qscale, eps, CA, CB, BMR)
    tl.debug_barrier()
    _ln_inplace_bf16(QKVGp + tstride, rows, rm, KQNW + 2 * CA, KQNW + 3 * CA, 1.0, eps, CA, CB, BMR)

@triton.jit
def _ph_att(QKVGp, tstride, Btb, ATT, r, h, I, CA: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, BMR: tl.constexpr, BNK: tl.constexpr):
    rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    ds = tl.arange(0, DHP); dm = ds < DH; hc = h * DH + ds
    q = tl.load(QKVGp + rows[:, None] * CA + hc[None, :], mask=rm[:, None] & dm[None, :], other=0.0, cache_modifier=".cg")
    m_i = tl.zeros([BMR], dtype=tl.float32) + float("-inf"); l_i = tl.zeros([BMR], dtype=tl.float32); acc = tl.zeros([BMR, DHP], dtype=tl.float32)
    for j0 in range(0, I, BNK):
        js = j0 + tl.arange(0, BNK); jm = js < I
        k = tl.load(QKVGp + tstride + js[:, None] * CA + hc[None, :], mask=jm[:, None] & dm[None, :], other=0.0, cache_modifier=".cg")
        s = tl.dot(q, tl.trans(k))
        bias = tl.load(Btb + h.to(tl.int64) * I * I + rows[:, None] * I + js[None, :], mask=rm[:, None] & jm[None, :], other=0.0)   # head-plane offset h*I*I in int64 (int32 wraps from I = 11,586)
        s = (s.to(tl.bfloat16).to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)   # stock: bf16 logits + bf16 bias -> bf16, softmax fp32
        s = tl.where(jm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1)); alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(QKVGp + 2 * tstride + js[:, None] * CA + hc[None, :], mask=jm[:, None] & dm[None, :], other=0.0, cache_modifier=".cg")
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_new
    o = (acc / l_i[:, None]).to(tl.bfloat16).to(tl.float32)
    g = tl.load(QKVGp + 3 * tstride + rows[:, None] * CA + hc[None, :], mask=rm[:, None] & dm[None, :], other=0.0, cache_modifier=".cg").to(tl.float32)
    tl.store(ATT + rows[:, None] * CA + hc[None, :], (o * g).to(tl.bfloat16), mask=rm[:, None] & dm[None, :])

@triton.jit
def _ph_out1(ATT, Wab, LOP, ldlop, lcol, Acur, Anxt, r, n, I, CA: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    ncols = n * BN + tl.arange(0, BN)
    acc = _gemm_tile(ATT, CA, Wab, CA, rows, rm, ncols, CA, BK, BMR, BN)
    x = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(LOP + rows[:, None] * ldlop + (lcol + ncols)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    out = (gate * x).to(tl.bfloat16).to(tl.float32)
    a = tl.load(Acur + rows[:, None] * CA + ncols[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cg")
    tl.store(Anxt + rows[:, None] * CA + ncols[None, :], a + out, mask=rm[:, None])

@triton.jit
def _ph_tr1(X2, W1b, W2b, HS, r, n, I, CA: tl.constexpr, HID: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    ncols = n * BN + tl.arange(0, BN)
    a1 = _gemm_tile(X2, CA, W1b, CA, rows, rm, ncols, CA, BK, BMR, BN)
    a2 = _gemm_tile(X2, CA, W2b, CA, rows, rm, ncols, CA, BK, BMR, BN)
    x1 = a1.to(tl.bfloat16).to(tl.float32); x2 = a2.to(tl.bfloat16).to(tl.float32)
    sx = (x1 * tl.sigmoid(x1)).to(tl.bfloat16).to(tl.float32)
    tl.store(HS + rows[:, None] * HID + ncols[None, :], (sx * x2).to(tl.bfloat16), mask=rm[:, None])

@triton.jit
def _ph_out2(HS, W3b, LOP, ldlop, lcol, Anxt, r, n, I, CA: tl.constexpr, HID: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    ncols = n * BN + tl.arange(0, BN)
    acc = _gemm_tile(HS, HID, W3b, HID, rows, rm, ncols, HID, BK, BMR, BN)
    x = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(LOP + rows[:, None] * ldlop + (lcol + ncols)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    out = (gate * x).to(tl.bfloat16).to(tl.float32)
    ptr = Anxt + rows[:, None] * CA + ncols[None, :]
    a1 = tl.load(ptr, mask=rm[:, None], other=0.0, cache_modifier=".cg")
    tl.store(ptr, a1 + out, mask=rm[:, None])

# ------------------------------------------------------------------------------------------------ the megakernel
@triton.jit
def _mk24(Abuf, astride, X1, X2, QKVG, pstride, tstride, ATT, HS, CNT,
          GB, ldgb, LOP, ldlop, Bt, btstride, Wqkvg, wqs, Wa, was, W1, W2, w12s, W3, w3s, KQNW, kqs,
          I, R, NB, NITEMS, qscale, eps_a, eps_kq,
          CA: tl.constexpr, HID: tl.constexpr, NH: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr,
          BMR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BNK: tl.constexpr, CB: tl.constexpr, KQN: tl.constexpr, NKIND: tl.constexpr,
          DYN: tl.constexpr, POLL: tl.constexpr, BMS: tl.constexpr):
    NQ: tl.constexpr = 4 * CA // BN          # qkvg n-tiles per row tile (24)
    NPER: tl.constexpr = CA // BN            # n-tiles per CA (6)
    NT1: tl.constexpr = HID // BN            # tr1 n-tiles (12)
    SUB: tl.constexpr = BMR // BMS           # memory-bound phases (STAT, KQN) are split into BMS-row sub-items
    pid = tl.program_id(0); ncta = tl.num_programs(0)
    npb = R * (SUB + NQ + SUB + NT1 + NH + NPER + NPER)
    QCTR = CNT + NB * NKIND * R                       # dynamic work-queue head (scheduling atomic; numerics untouched)
    if DYN:
        g = tl.atomic_add(QCTR, 1)
        step = 0
    else:
        g = pid
        step = ncta
    while g < NITEMS:
        b = g // npb; l = g - b * npb
        pin = b % 2
        Acur = Abuf + pin * astride; Anxt = Abuf + (1 - pin) * astride
        QKVGp = QKVG + pin * pstride
        o1 = R * SUB; o2 = o1 + R * NQ; o3 = o2 + R * SUB; o4 = o3 + R * NT1; o5 = o4 + R * NH; o6 = o5 + R * NPER
        if l < o1:                                   # STAT(r, sub)
            r = l // SUB; sr = l - r * SUB
            if b > 0:
                _wait(CNT, _cidx(b - 1, 6, r, R, NKIND), NPER, POLL)
            _ph_stat(Acur, GB, ldgb, b * 4 * CA, X1, X2, r * BMR + sr * BMS, I, eps_a, CA, BMS, CB)
            _signal(CNT, _cidx(b, 0, r, R, NKIND))
        elif l < o2:                                 # QKVG(r, n)
            q = l - o1; r = q // NQ; n = q - r * NQ
            _wait(CNT, _cidx(b, 0, r, R, NKIND), SUB, POLL)
            _ph_qkvg(X1, Wqkvg + b * wqs, QKVGp, tstride, r, n, I, qscale, CA, BMR, BN, BK, KQN, NPER)
            _signal(CNT, _cidx(b, 1, r, R, NKIND))
        elif l < o3:                                 # KQN(r, sub)  (also the 'all Q|K|V|G tiles of row-tile r done' marker when KQN is off)
            q = l - o2; r = q // SUB; sr = q - r * SUB
            _wait(CNT, _cidx(b, 1, r, R, NKIND), NQ, POLL)
            if KQN:
                _ph_kqn(QKVGp, tstride, KQNW + b * kqs, r * BMR + sr * BMS, I, qscale, eps_kq, CA, BMS, CB)
            _signal(CNT, _cidx(b, 2, 0, R, NKIND))
        elif l < o4:                                 # TR1(r, n)
            q = l - o3; r = q // NT1; n = q - r * NT1
            _wait(CNT, _cidx(b, 0, r, R, NKIND), SUB, POLL)
            _ph_tr1(X2, W1 + b * w12s, W2 + b * w12s, HS, r, n, I, CA, HID, BMR, BN, BK)
            _signal(CNT, _cidx(b, 3, r, R, NKIND))
        elif l < o5:                                 # ATT(r, h)
            q = l - o4; r = q // NH; h = q - r * NH
            _wait(CNT, _cidx(b, 2, 0, R, NKIND), R * SUB, POLL)
            _ph_att(QKVGp, tstride, Bt + b.to(tl.int64) * btstride, ATT, r, h, I, CA, DH, DHP, BMR, BNK)   # block offset b*16*I*I in int64 (int32 wraps from I = 2,416)
            _signal(CNT, _cidx(b, 4, r, R, NKIND))
        elif l < o6:                                 # OUT1(r, n)
            q = l - o5; r = q // NPER; n = q - r * NPER
            _wait(CNT, _cidx(b, 4, r, R, NKIND), NH, POLL)
            _ph_out1(ATT, Wa + b * was, LOP, ldlop, b * 2 * CA, Acur, Anxt, r, n, I, CA, BMR, BN, BK)
            _signal(CNT, _cidx(b, 5, r, R, NKIND))
        else:                                        # OUT2(r, n)
            q = l - o6; r = q // NPER; n = q - r * NPER
            _wait(CNT, _cidx(b, 3, r, R, NKIND), NT1, POLL)
            _wait(CNT, _cidx(b, 5, r, R, NKIND), NPER, POLL)
            _ph_out2(HS, W3 + b * w3s, LOP, ldlop, b * 2 * CA + CA, Anxt, r, n, I, CA, HID, BMR, BN, BK)
            _signal(CNT, _cidx(b, 6, r, R, NKIND))
        if DYN:
            g = tl.atomic_add(QCTR, 1)
        else:
            g = g + step


class MK2TokenTransformer(mkrf3.MKTokenTransformer):
    """single-launch dataflow megakernel; weight prep / hoist / S-precompute inherited from MKTokenTransformer"""
    def __init__(self, tok, BMR=64, BN=128, BK=64, BNK=64, num_warps=8, ncta=None, sfold=True, dyn=True, poll=True, num_stages=3, bms=16):
        super().__init__(tok, BM=16, BN=BN, BK=BK, BNK=BNK, num_warps=num_warps, sfold=sfold)
        self.BMR = BMR; self.CB = 128; self.dyn = dyn; self.poll = poll; self.num_stages = num_stages; self.BMS = bms
        assert BMR % self.BMS == 0
        assert self.NORES, "mk2 implements the RF3 ordering (transition reads the block input); sequential-residual models need the STAT2 phase"
        assert self.CA % BN == 0 and (self.NT * self.CA) % BN == 0
        dev = self.W["qkvg"].device
        self.ncta = ncta or torch.cuda.get_device_properties(dev).multi_processor_count
        if self.KQN:
            self.KQNW = torch.stack([torch.stack([qw, qb, kw, kb], 0) for (qw, qb, kw, kb, _e) in self.kqn], 0).contiguous()   # [NB, 4, CA] fp32
            self.eps_kq = self.kqn[0][4]
        else:
            self.KQNW = torch.zeros(self.NB, 4, self.CA, device=dev); self.eps_kq = 1e-5
        self._buf2 = {}

    def _b2(self, I, dev):
        key = (I, str(dev))
        if key not in self._buf2:
            CA, HID, NB = self.CA, self.NT * self.CA, self.NB
            R = triton.cdiv(I, self.BMR)
            self._buf2[key] = dict(A=torch.empty(2, I, CA, dtype=torch.float32, device=dev), X1=torch.empty(I, CA, dtype=torch.bfloat16, device=dev), X2=torch.empty(I, CA, dtype=torch.bfloat16, device=dev),
                                   QKVG=torch.empty(2, 4, I, CA, dtype=torch.bfloat16, device=dev), ATT=torch.empty(I, CA, dtype=torch.bfloat16, device=dev), HS=torch.empty(I, HID, dtype=torch.bfloat16, device=dev),
                                   CNT=torch.zeros(NB * NKIND * R + 1, dtype=torch.int32, device=dev), R=R)
        return self._buf2[key]

    def forward(self, A_I, S_I, Z_II=None, Beta_II=None):
        assert Beta_II is None
        shp = A_I.shape; CA = self.CA
        A2 = A_I.reshape(-1, CA); S2 = S_I.reshape(-1, S_I.shape[-1]); I = A2.shape[0]
        if self.Bt is None or self.Bt.shape[-1] != I:
            assert Z_II is not None; self.hoist(Z_II)
        dev = A2.device; buf = self._b2(I, dev); R = buf["R"]
        buf["A"][0].copy_(A2); buf["CNT"].zero_()
        GB, LOP = self.s_precompute(S2)
        W = self.W; NB = self.NB
        HID = self.NT * CA
        SUB = self.BMR // self.BMS
        nitems = NB * R * (SUB + 4 * CA // self.BN + SUB + HID // self.BN + self.NH + 2 * (CA // self.BN))
        _mk24[(self.ncta,)](buf["A"], buf["A"].stride(0), buf["X1"], buf["X2"], buf["QKVG"], buf["QKVG"].stride(0), buf["QKVG"].stride(1), buf["ATT"], buf["HS"], buf["CNT"],
                            GB, GB.stride(0), LOP, LOP.stride(0), self.Bt, self.Bt.stride(0), W["qkvg"], W["qkvg"].stride(0), W["a"], W["a"].stride(0), W["w1"], W["w2"], W["w1"].stride(0), W["w3"], W["w3"].stride(0), self.KQNW, self.KQNW.stride(0),
                            I, R, NB, nitems, self.qscale, self.eps_a, self.eps_kq,
                            CA=CA, HID=HID, NH=self.NH, DH=self.DH, DHP=self.DHP, BMR=self.BMR, BN=self.BN, BK=self.BK, BNK=self.BNK, CB=self.CB, KQN=self.KQN, NKIND=NKIND,
                            DYN=self.dyn, POLL=self.poll, BMS=self.BMS, num_warps=self.nw, num_stages=self.num_stages)
        return buf["A"][NB % 2].reshape(shp).clone()

    def launches_per_call(self):
        return {"s_precompute": 3, "fused": 1, "memset": 1}


# ================================================================================================ MK3: hybrid (cuBLAS GEMMs + fused phase kernels)
@triton.jit
def _k3_stat(Acur, GB, ldgb, gcol, X1, X2, I, eps, CA: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr):
    _ph_stat(Acur, GB, ldgb, gcol, X1, X2, tl.program_id(0) * BMR, I, eps, CA, BMR, CB)

@triton.jit
def _k3_qkvg_epi(QKVGraw, QKVGp, tstride, KQNW, I, qscale, eps, CA: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr, KQN: tl.constexpr):
    """from the cuBLAS output [I, 4*CA] (bf16): Q -> LN*scale | K -> LN | V copy | G sigmoid  into QKVGp[4][I,CA]"""
    r = tl.program_id(0); rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        v = tl.load(QKVGraw + rows[:, None] * (4 * CA) + (2 * CA + cs)[None, :], mask=rm[:, None], other=0.0)
        tl.store(QKVGp + 2 * tstride + rows[:, None] * CA + cs[None, :], v, mask=rm[:, None])
        g = tl.load(QKVGraw + rows[:, None] * (4 * CA) + (3 * CA + cs)[None, :], mask=rm[:, None], other=0.0)
        tl.store(QKVGp + 3 * tstride + rows[:, None] * CA + cs[None, :], tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16), mask=rm[:, None])
        q = tl.load(QKVGraw + rows[:, None] * (4 * CA) + cs[None, :], mask=rm[:, None], other=0.0)
        k = tl.load(QKVGraw + rows[:, None] * (4 * CA) + (CA + cs)[None, :], mask=rm[:, None], other=0.0)
        if KQN:
            tl.store(QKVGp + rows[:, None] * CA + cs[None, :], q, mask=rm[:, None])
            tl.store(QKVGp + tstride + rows[:, None] * CA + cs[None, :], k, mask=rm[:, None])
        else:
            tl.store(QKVGp + rows[:, None] * CA + cs[None, :], (q.to(tl.float32) * qscale).to(tl.bfloat16), mask=rm[:, None])
            tl.store(QKVGp + tstride + rows[:, None] * CA + cs[None, :], k, mask=rm[:, None])
    if KQN:
        tl.debug_barrier()
        _ln_inplace_bf16(QKVGp, rows, rm, KQNW, KQNW + CA, qscale, eps, CA, CB, BMR)
        _ln_inplace_bf16(QKVGp + tstride, rows, rm, KQNW + 2 * CA, KQNW + 3 * CA, 1.0, eps, CA, CB, BMR)

@triton.jit
def _k3_att(QKVGp, tstride, Btb, ATT, I, CA: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, BMR: tl.constexpr, BNK: tl.constexpr):
    _ph_att(QKVGp, tstride, Btb, ATT, tl.program_id(0), tl.program_id(1), I, CA, DH, DHP, BMR, BNK)

@triton.jit
def _k3_gate_res(Xraw, ldx, LOP, ldlop, lcol, Ain, Aout, I, CA: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr):
    """Aout[rows] = Ain[rows] + bf16(sigmoid(lop) * x)   (x = bf16 cuBLAS output)"""
    r = tl.program_id(0); rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    for c0 in range(0, CA, CB):
        cs = c0 + tl.arange(0, CB)
        x = tl.load(Xraw + rows[:, None] * ldx + cs[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        gate = tl.sigmoid(tl.load(LOP + rows[:, None] * ldlop + (lcol + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        out = (gate * x).to(tl.bfloat16).to(tl.float32)
        a = tl.load(Ain + rows[:, None] * CA + cs[None, :], mask=rm[:, None], other=0.0)
        tl.store(Aout + rows[:, None] * CA + cs[None, :], a + out, mask=rm[:, None])

@triton.jit
def _k3_swiglu(H12, HS, I, HID: tl.constexpr, BMR: tl.constexpr, CB: tl.constexpr):
    r = tl.program_id(0); rows = r * BMR + tl.arange(0, BMR); rm = rows < I
    for c0 in range(0, HID, CB):
        cs = c0 + tl.arange(0, CB)
        x1 = tl.load(H12 + rows[:, None] * (2 * HID) + cs[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(H12 + rows[:, None] * (2 * HID) + (HID + cs)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        sx = (x1 * tl.sigmoid(x1)).to(tl.bfloat16).to(tl.float32)
        tl.store(HS + rows[:, None] * HID + cs[None, :], (sx * x2).to(tl.bfloat16), mask=rm[:, None])


class MK3TokenTransformer(MK2TokenTransformer):
    """hybrid: the four projections per block on cuBLAS (bf16, fp32 accumulate = stock class), everything else in 5 fused Triton kernels
    -> 9 launches/block + 3 per call.  Numerics identical in construction to MK2 (RF3-exact structure: kq-norm, transition on the block input)."""
    def __init__(self, tok, **kw):
        kw = dict(kw); kw.setdefault("BMR", 32); kw.setdefault("num_warps", 4)
        super().__init__(tok, **kw)
        self.W12 = torch.stack([torch.cat([self.W["w1"][b], self.W["w2"][b]], 0) for b in range(self.NB)], 0).contiguous()   # [NB, 2*HID, CA]

    def forward(self, A_I, S_I, Z_II=None, Beta_II=None):
        assert Beta_II is None
        shp = A_I.shape; CA = self.CA; HID = self.NT * CA
        A2 = A_I.reshape(-1, CA); S2 = S_I.reshape(-1, S_I.shape[-1]); I = A2.shape[0]
        if self.Bt is None or self.Bt.shape[-1] != I:
            assert Z_II is not None; self.hoist(Z_II)
        dev = A2.device; buf = self._b2(I, dev)
        if "RAW" not in buf:
            buf["RAW"] = torch.empty(I, 4 * CA, dtype=torch.bfloat16, device=dev); buf["OA"] = torch.empty(I, CA, dtype=torch.bfloat16, device=dev)
            buf["H12"] = torch.empty(I, 2 * HID, dtype=torch.bfloat16, device=dev); buf["O3"] = torch.empty(I, CA, dtype=torch.bfloat16, device=dev)
        A = buf["A"]; A[0].copy_(A2)
        GB, LOP = self.s_precompute(S2)
        BMR, CB = self.BMS, self.CB; R = triton.cdiv(I, BMR); nw = self.nw          # memory-bound phase kernels: 16-row programs
        QKVG = buf["QKVG"][0]; tstride = buf["QKVG"].stride(1)
        for b in range(self.NB):
            Acur = A[b % 2]; Anxt = A[(b + 1) % 2]
            _k3_stat[(R,)](Acur, GB, GB.stride(0), b * 4 * CA, buf["X1"], buf["X2"], I, self.eps_a, CA=CA, BMR=BMR, CB=CB, num_warps=nw)
            torch.mm(buf["X1"], self.W["qkvg"][b].t(), out=buf["RAW"])                                   # [I, 4CA] bf16 (cuBLAS, fp32 acc)
            _k3_qkvg_epi[(R,)](buf["RAW"], QKVG, tstride, self.KQNW[b], I, self.qscale, self.eps_kq, CA=CA, BMR=BMR, CB=CB, KQN=self.KQN, num_warps=nw)
            _k3_att[(triton.cdiv(I, 64), self.NH)](QKVG, tstride, self.Bt[b], buf["ATT"], I, CA=CA, DH=self.DH, DHP=self.DHP, BMR=64, BNK=self.BNK, num_warps=nw)
            torch.mm(buf["ATT"], self.W["a"][b].t(), out=buf["OA"])
            _k3_gate_res[(R,)](buf["OA"], CA, LOP, LOP.stride(0), b * 2 * CA, Acur, Anxt, I, CA=CA, BMR=BMR, CB=CB, num_warps=nw)
            torch.mm(buf["X2"], self.W12[b].t(), out=buf["H12"])                                       # [I, 2*HID]
            _k3_swiglu[(R,)](buf["H12"], buf["HS"], I, HID=HID, BMR=BMR, CB=CB, num_warps=nw)
            torch.mm(buf["HS"], self.W["w3"][b].t(), out=buf["O3"])
            _k3_gate_res[(R,)](buf["O3"], CA, LOP, LOP.stride(0), b * 2 * CA + CA, Anxt, Anxt, I, CA=CA, BMR=BMR, CB=CB, num_warps=nw)
        return A[self.NB % 2].reshape(shp).clone()

    def launches_per_call(self):
        return {"s_precompute": 3, "per_block": 9, "total": 3 + 9 * self.NB}
