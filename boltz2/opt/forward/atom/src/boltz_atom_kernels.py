"""boltz_atom_kernels.py — the Triton kernels of the Boltz-2 atom-attention levers (boltz 2.2.1, `boltz_atom.py` is the lever module; this file
carries kernels + their launchers + a statement-by-statement torch oracle of the same fused decomposition, nothing else).

Shapes (Boltz-2 checkpoint): atom width D=128, 4 heads x 32, query window W=32 atoms, key window H=128 atoms (the 128 atoms [32w-48, 32w+80) around
query window w, clipped at the ends of the atom axis), transition hidden 256 (SwiGLU over a 512-wide projection), encoder->token width 768.
Rows: R = B*m*M flattened (B inputs, m diffusion samples, M padded atoms, M % 32 == 0); the per-atom conditioning tensors (hoisted once per
sample() by boltz_atom.py) have Bc*M rows, Bc = B (crow = (row // (CMULT*M)) * M + row % M with CMULT = m) or B*m (CMULT = 1); the pair bias is
read IN PLACE from the conditioning tensor [B, K=M/32, 32, 128, L*4] (bf16 or fp32; layer l, head h at channel 4l+h) — never expanded, never copied.

Fast tier (tolerance class; every dot states its input precision — PREC in {"ieee","tf32","tf32x3"} on fp32 operands, or LOWP=1: bf16 operands,
fp32 accumulate; the row default is ieee = true fp32 products and fp32 sums, only the summation order differs from cuBLAS):
    _k_adaln_qkvg      AdaLN(a; cached sigmoid-scale, bias) -> [q|k|v|g] = b @ W1^T (+bq), g = sigmoid(g); prologues: PRO=1 encoder layer 1
                       (a = q_cond + r @ Wr^T), PRO=2 decoder layer 1 (a = q_skip + a_to_q[token(atom)]).
    _k_win_attn        one (32-query window, head): S = q k^T / sqrt(32) + bias + (1-mask)(-1e6) over the window's 128 keys, softmax, P v, * gate.
    _k_out_transition  o @ Wo^T, gated residual, AdaLN, SwiGLU transition (silu(gates)*x * a_to_b, @ Wba^T), gated residual; epilogues: EPI=1 the NEXT
                       layer's AdaLN + [q|k|v|g] (so a 3-layer transformer is 1 + 3x2 launches), EPI=2 encoder tail relu(a @ Wat^T) -> [R,768],
                       EPI=3 decoder tail LayerNorm(affine) @ Wpos^T -> [R,3].
    _k_segmean         encoder aggregation: a[token] = (sum of its atoms' rows, CSR order) * 1/(n+1e-6)  (the stock bmm with the mean matrix, as a
                       deterministic segmented sum; fast tier only — a real reduction, never bitwise to cuBLAS).
Exact tier (bitwise class; pure data movement):
    _k_gather_keys     single_to_keys as a gather: out[b,k,s,:] = x[b, 32k-48+s, :] inside the atom axis else +0.0, and every zero written as +0.0
                       (cuBLAS accumulates the one-hot product from +0.0 and so never emits -0.0; a gather would forward x's -0.0).
    _k_gather_rows     decoder broadcast bmm(one_hot[Bm,M,N], x[Bm,N,D]) as a gather by token index (-1 = empty row -> +0.0), zeros canonical +0.0.
No autotuning (graph-capturable, deterministic launch configuration; _CFG below, set_cfg() for experiments).
"""
from __future__ import annotations

import math

import torch

_TRITON = {"ok": None, "why": ""}


def triton_ok() -> bool:
    if _TRITON["ok"] is None:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
            _TRITON["ok"] = True
        except Exception as e:  # noqa: BLE001
            _TRITON["ok"] = False; _TRITON["why"] = f"{type(e).__name__}: {e}"
    return bool(_TRITON["ok"])


D = 128; HEADS = 4; HD = 32; WQ = 32; WK = 128; HID = 256; DTOK = 768
INV_SQRT_HD = 1.0 / math.sqrt(HD)
PRECS = ("ieee", "tf32", "tf32x3", "bf16")          # the GEMM-precision words (BOLTZ_ATOM_GEMM); bf16 = LOWP operands, fp32 accumulate

if triton_ok():
    import triton
    import triton.language as tl

    @triton.jit
    def _mm(a, b, PREC: tl.constexpr, LOWP: tl.constexpr):
        """The one dot of this file: fp32 accumulate always; operands bf16 (LOWP) or fp32 at the stated input precision (never Triton's tf32 default silently)."""
        if LOWP:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision=PREC)

    @triton.jit
    def _ln_rows(x, eps):
        """LayerNorm statistics over the last axis (width 128, fp32, biased variance as torch): returns (x - mean) * rsqrt(var + eps)."""
        mean = tl.sum(x, 1) / 128.0
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, 1) / 128.0
        return xc * (1.0 / tl.sqrt(var + eps))[:, None]

    @triton.jit
    def _adaln_qkvg_rows(x, crow, rows, rmask, S1, B1, W1T, BQ, QKV, G, eps,
                         PREC: tl.constexpr, LOWP: tl.constexpr, OBF: tl.constexpr, BM: tl.constexpr):
        """Shared body: b = S1[crow]*LN(x) + B1[crow]; [q|k|v|g] = b @ W1T (W1T = [Wq;Wk;Wv;Wg]^T, [128,512]); q += bq; g = sigmoid(g).
        Writes QKV[rows, 0:384] and G[rows, 0:128] (bf16 when OBF)."""
        d = tl.arange(0, 128)
        xn = _ln_rows(x, eps)
        s1 = tl.load(S1 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        b1 = tl.load(B1 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        bmod = s1 * xn + b1
        for n0 in tl.static_range(0, 512, 128):
            w = tl.load(W1T + d[:, None] * 512 + (n0 + d)[None, :])                      # [128 k, 128 n]
            acc = _mm(bmod, w, PREC, LOWP)
            if n0 == 0:
                acc = acc + tl.load(BQ + d)[None, :]
            if n0 == 384:
                acc = 1.0 / (1.0 + tl.exp(-acc))
                if OBF:
                    tl.store(G + rows[:, None] * 128 + d[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])
                else:
                    tl.store(G + rows[:, None] * 128 + d[None, :], acc, mask=rmask[:, None])
            else:
                if OBF:
                    tl.store(QKV + rows[:, None] * 384 + (n0 + d)[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])
                else:
                    tl.store(QKV + rows[:, None] * 384 + (n0 + d)[None, :], acc, mask=rmask[:, None])

    @triton.jit(do_not_specialize=["R", "M", "CMULT", "NTOK"])
    def _k_adaln_qkvg(X, Q0, POS, WR, QSKIP, A2Q, TOK, S1, B1, W1T, BQ, QKV, G,
                      R, M, CMULT, NTOK, eps,
                      PRO: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, OBF: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < R
        rows64 = rows.to(tl.int64)
        bm = rows // M; atom = rows % M
        crow = ((bm // CMULT) * M + atom).to(tl.int64)
        d = tl.arange(0, 128)
        if PRO == 0:
            x = tl.load(X + rows64[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        elif PRO == 1:
            # encoder layer 1: a = q_cond[crow] + r_noisy @ Wr^T  (Wr [128,3]); the stream a is written for the residual reads downstream
            x = tl.load(Q0 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            for k in tl.static_range(0, 3):
                pk = tl.load(POS + rows64 * 3 + k, mask=rmask, other=0.0).to(tl.float32)
                wk = tl.load(WR + d * 3 + k)
                x = x + pk[:, None] * wk[None, :]
            tl.store(X + rows64[:, None] * 128 + d[None, :], x, mask=rmask[:, None])
        else:
            # decoder layer 1: a = q_skip + a_to_q[bm, token(atom)]  (token index -1: an atom of no token -> adds nothing, as the all-zero one-hot row)
            x = tl.load(QSKIP + rows64[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            t = tl.load(TOK + crow, mask=rmask, other=-1)
            has = (t >= 0) & rmask
            tsafe = tl.where(has, t, 0).to(tl.int64)
            add = tl.load(A2Q + (bm.to(tl.int64) * NTOK + tsafe)[:, None] * 128 + d[None, :], mask=has[:, None], other=0.0).to(tl.float32)
            x = x + add
            tl.store(X + rows64[:, None] * 128 + d[None, :], x, mask=rmask[:, None])
        _adaln_qkvg_rows(x, crow, rows64, rmask, S1, B1, W1T, BQ, QKV, G, eps, PREC, LOWP, OBF, BM)

    @triton.jit(do_not_specialize=["M", "K", "CMULT", "BMULT", "LD", "LOFF"])
    def _k_win_attn(QKV, G, BIAS, PAD, OB, M, K, CMULT, BMULT, LD, LOFF, inv_sqrt_hd, inf,
                    PREC: tl.constexpr, LOWP: tl.constexpr, OBF: tl.constexpr):
        wi = tl.program_id(0); h = tl.program_id(1)
        row0 = wi * 32
        bm = row0 // M; w = (row0 % M) // 32
        bc = bm // CMULT                                     # conditioning row block (pad mask)
        bb = bm // BMULT                                     # bias batch index
        i = tl.arange(0, 32); j = tl.arange(0, 128); dd = tl.arange(0, 32)
        rows = (row0 + i).to(tl.int64)
        katom = w * 32 - 48 + j
        kin = (katom >= 0) & (katom < M)
        ksafe = tl.where(kin, katom, 0)
        krows = (bm * M + ksafe).to(tl.int64)
        q = tl.load(QKV + rows[:, None] * 384 + (h * 32 + dd)[None, :])
        k = tl.load(QKV + krows[:, None] * 384 + (128 + h * 32 + dd)[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(QKV + krows[:, None] * 384 + (256 + h * 32 + dd)[None, :], mask=kin[:, None], other=0.0)
        mval = tl.load(PAD + (bc * M + ksafe).to(tl.int64), mask=kin, other=0.0).to(tl.float32)   # to_keys(mask): 0 outside the atom axis
        boff = ((bb * K + w).to(tl.int64) * 32 + i.to(tl.int64))[:, None] * 128 + j.to(tl.int64)[None, :]
        bias = tl.load(BIAS + boff * LD + LOFF + h).to(tl.float32)
        s = _mm(q, tl.trans(k), PREC, LOWP)                                                     # [32,128]
        s = s * inv_sqrt_hd + bias
        s = s + (1.0 - mval)[None, :] * (-inf)
        mx = tl.max(s, 1)
        p = tl.exp(s - mx[:, None])
        l = tl.sum(p, 1)
        o = _mm(p, v, PREC, LOWP) / l[:, None]                                                  # [32,32]
        g = tl.load(G + rows[:, None] * 128 + (h * 32 + dd)[None, :]).to(tl.float32)
        o = o * g
        if OBF:
            tl.store(OB + rows[:, None] * 128 + (h * 32 + dd)[None, :], o.to(tl.bfloat16))
        else:
            tl.store(OB + rows[:, None] * 128 + (h * 32 + dd)[None, :], o)

    @triton.jit(do_not_specialize=["R", "M", "CMULT"])
    def _k_out_transition(OB, X, G1, S2, B2, G2, WOT, WSGT, WABT, WBAT,
                          NS1, NB1, NW1T, NBQ, QKV, G,                      # EPI=1: next layer's AdaLN + qkvg
                          WATT, QA,                                        # EPI=2: encoder tail
                          LNW, LNB, WPOS, RUP,                             # EPI=3: decoder tail
                          R, M, CMULT, eps, eps_out,
                          EPI: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, OBF: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < R
        rows64 = rows.to(tl.int64)
        bm = rows // M; atom = rows % M
        crow = ((bm // CMULT) * M + atom).to(tl.int64)
        d = tl.arange(0, 128); c = tl.arange(0, BC)
        o = tl.load(OB + rows64[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        wo = tl.load(WOT + d[:, None] * 128 + d[None, :])
        y = _mm(o, wo, PREC, LOWP)
        x = tl.load(X + rows64[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        g1 = tl.load(G1 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        a1 = x + g1 * y
        tl.store(X + rows64[:, None] * 128 + d[None, :], a1, mask=rmask[:, None])              # park a1 (re-read below: fewer live registers)
        xn = _ln_rows(a1, eps)
        s2 = tl.load(S2 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        b2v = tl.load(B2 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        b2 = s2 * xn + b2v
        z = tl.zeros((BM, 128), dtype=tl.float32)
        for c0 in tl.static_range(0, 256, BC):
            wx = tl.load(WSGT + d[:, None] * 512 + (c0 + c)[None, :])                          # x half of swish_gate: output cols [0,256)
            wg = tl.load(WSGT + d[:, None] * 512 + (256 + c0 + c)[None, :])                    # gates half: output cols [256,512)
            wab = tl.load(WABT + d[:, None] * 256 + (c0 + c)[None, :])
            xp = _mm(b2, wx, PREC, LOWP)
            gp = _mm(b2, wg, PREC, LOWP)
            ab = _mm(b2, wab, PREC, LOWP)
            t = (gp * (1.0 / (1.0 + tl.exp(-gp)))) * xp * ab                                    # silu(gates) * x * a_to_b(a)
            wba = tl.load(WBAT + (c0 + c)[:, None] * 128 + d[None, :])                          # [BC k, 128 n]
            z += _mm(t, wba, PREC, LOWP)
        a1 = tl.load(X + rows64[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        g2 = tl.load(G2 + crow[:, None] * 128 + d[None, :], mask=rmask[:, None], other=0.0)
        a2 = a1 + g2 * z
        tl.store(X + rows64[:, None] * 128 + d[None, :], a2, mask=rmask[:, None])
        if EPI == 1:
            _adaln_qkvg_rows(a2, crow, rows64, rmask, NS1, NB1, NW1T, NBQ, QKV, G, eps, PREC, LOWP, OBF, BM)
        elif EPI == 2:
            for n0 in tl.static_range(0, 768, 128):
                w = tl.load(WATT + d[:, None] * 768 + (n0 + d)[None, :])
                qa = _mm(a2, w, PREC, LOWP)
                qa = tl.maximum(qa, 0.0)
                tl.store(QA + rows64[:, None] * 768 + (n0 + d)[None, :], qa, mask=rmask[:, None])
        elif EPI == 3:
            xn3 = _ln_rows(a2, eps_out)
            xn3 = xn3 * tl.load(LNW + d)[None, :] + tl.load(LNB + d)[None, :]
            for cc in tl.static_range(0, 3):
                wr = tl.load(WPOS + cc * 128 + d)
                val = tl.sum(xn3 * wr[None, :], 1)
                tl.store(RUP + rows64 * 3 + cc, val, mask=rmask)

    @triton.jit(do_not_specialize=["M", "NTOK", "MULT"])
    def _k_segmean(QA, ORDER, OFF, WTOK, A, M, NTOK, MULT, BD: tl.constexpr):
        p = tl.program_id(0); cb = tl.program_id(1)
        bm = p // NTOK; t = p % NTOK; b = bm // MULT
        cols = cb * BD + tl.arange(0, BD)
        start = tl.load(OFF + b * (NTOK + 1) + t); end = tl.load(OFF + b * (NTOK + 1) + t + 1)
        acc = tl.zeros((BD,), dtype=tl.float32)
        for q in range(start, end):
            atom = tl.load(ORDER + b * M + q).to(tl.int64)
            acc += tl.load(QA + (bm.to(tl.int64) * M + atom) * 768 + cols)
        wt = tl.load(WTOK + b * NTOK + t)
        tl.store(A + (bm.to(tl.int64) * NTOK + t) * 768 + cols, acc * wt)

    @triton.jit(do_not_specialize=["M", "K", "DW", "TOTAL"])
    def _k_gather_keys(X, OUT, M, K, DW, TOTAL, BR: tl.constexpr, BD: tl.constexpr):
        """single_to_keys as a gather (exact tier): flat key-row r in [0, B*K*128): b = r // (K*128), k, s; source atom 32k-48+s."""
        pid = tl.program_id(0); pc = tl.program_id(1)
        r = pid.to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
        rm = r < TOTAL
        b = r // (K * 128); rem = r % (K * 128); k = rem // 128; s = rem % 128
        src = k * 32 - 48 + s
        ok = (src >= 0) & (src < M) & rm
        cols = pc * BD + tl.arange(0, BD); cm = cols < DW
        v = tl.load(X + (b * M + tl.where(ok, src, 0))[:, None] * DW + cols[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
        v = tl.where(v == 0.0, tl.zeros_like(v), v)                                            # +0.0 canonical: -0.0 -> +0.0, as the GEMM emits
        tl.store(OUT + r[:, None] * DW + cols[None, :], v, mask=rm[:, None] & cm[None, :])

    @triton.jit(do_not_specialize=["M", "NTOK", "DW", "TOTAL"])
    def _k_gather_rows(X, TOK, OUT, M, NTOK, DW, TOTAL, BR: tl.constexpr, BD: tl.constexpr):
        """bmm(one_hot[Bm,M,NTOK], x[Bm,NTOK,DW]) as a gather (exact tier): out[bm,i,:] = x[bm, tok[bm,i], :] or +0.0 when tok < 0."""
        pid = tl.program_id(0); pc = tl.program_id(1)
        r = pid.to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)                             # flat atom row in [0, Bm*M)
        rm = r < TOTAL
        bm = r // M
        t = tl.load(TOK + r, mask=rm, other=-1)
        ok = (t >= 0) & rm
        cols = pc * BD + tl.arange(0, BD); cm = cols < DW
        v = tl.load(X + (bm * NTOK + tl.where(ok, t, 0).to(tl.int64))[:, None] * DW + cols[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
        v = tl.where(v == 0.0, tl.zeros_like(v), v)
        tl.store(OUT + r[:, None] * DW + cols[None, :], v, mask=rm[:, None] & cm[None, :])


# ============================================================== launchers ==============================================================
_CFG = {"BM_A": 64, "WARPS_A": 4, "BM_C": 64, "BC": 64, "WARPS_C": 4, "WARPS_B": 2, "BD_SEG": 256}   # tuned on H100 at ~9k atoms (bf16); the step time is only mildly sensitive to these settings


def cfg() -> dict:
    return dict(_CFG)


def set_cfg(**kw) -> None:
    for k, v in kw.items():
        if k not in _CFG:
            raise KeyError(k)
        _CFG[k] = int(v)


def _prec_args(prec: str):
    if prec not in PRECS:
        raise ValueError(f"gemm precision {prec!r} not in {PRECS}")
    lowp = prec == "bf16"
    return ("ieee" if lowp else prec), lowp


def launch_adaln_qkvg(*, x, pro, q0=None, pos=None, wr=None, qskip=None, a2q=None, tok=None, lw, qkv, g, R, M, cmult, ntok, prec, eps=1e-5):
    PREC, LOWP = _prec_args(prec)
    BM = _CFG["BM_A"]
    dummy = x
    grid = (triton.cdiv(R, BM),)
    _k_adaln_qkvg[grid](x, q0 if q0 is not None else dummy, pos if pos is not None else dummy, wr if wr is not None else dummy,
                        qskip if qskip is not None else dummy, a2q if a2q is not None else dummy, tok if tok is not None else dummy,
                        lw["S1"], lw["B1"], lw["W1T"], lw["BQ"], qkv, g, R, M, cmult, ntok, eps,
                        PRO=pro, PREC=PREC, LOWP=LOWP, OBF=(qkv.dtype == torch.bfloat16), BM=BM, num_warps=_CFG["WARPS_A"])


def launch_win_attn(*, qkv, g, bias, ld, loff, bmult, pad, ob, R, M, K, cmult, prec, inf=1e6):
    PREC, LOWP = _prec_args(prec)
    grid = (R // 32, HEADS)
    _k_win_attn[grid](qkv, g, bias, pad, ob, M, K, cmult, bmult, ld, loff, INV_SQRT_HD, inf,
                      PREC=PREC, LOWP=LOWP, OBF=(ob.dtype == torch.bfloat16), num_warps=_CFG["WARPS_B"])


def launch_out_transition(*, ob, x, lw, epi, nlw=None, qkv=None, g=None, watt=None, qa=None, lnw=None, lnb=None, wpos=None, rup=None,
                          R, M, cmult, prec, eps=1e-5, eps_out=1e-5):
    PREC, LOWP = _prec_args(prec)
    BM, BC = _CFG["BM_C"], _CFG["BC"]
    dm = x
    nl = nlw if nlw is not None else lw
    grid = (triton.cdiv(R, BM),)
    obf = (qkv is not None and qkv.dtype == torch.bfloat16)
    _k_out_transition[grid](ob, x, lw["G1"], lw["S2"], lw["B2"], lw["G2"], lw["WOT"], lw["WSGT"], lw["WABT"], lw["WBAT"],
                            nl["S1"] if epi == 1 else dm, nl["B1"] if epi == 1 else dm, nl["W1T"] if epi == 1 else dm, nl["BQ"] if epi == 1 else dm,
                            qkv if qkv is not None else dm, g if g is not None else dm,
                            watt if watt is not None else dm, qa if qa is not None else dm,
                            lnw if lnw is not None else dm, lnb if lnb is not None else dm, wpos if wpos is not None else dm, rup if rup is not None else dm,
                            R, M, cmult, eps, eps_out,
                            EPI=epi, PREC=PREC, LOWP=LOWP, OBF=obf, BM=BM, BC=BC, num_warps=_CFG["WARPS_C"])


def launch_segmean(*, qa, order, off, wtok, a, Bm, M, ntok, mult):
    BD = _CFG["BD_SEG"]
    grid = (Bm * ntok, DTOK // BD)
    _k_segmean[grid](qa, order, off, wtok, a, M, ntok, mult, BD=BD, num_warps=4)


def gather_keys(x: torch.Tensor, K: int) -> torch.Tensor:
    """single_to_keys(x, indexing_matrix(K), W=32, H=128) as a gather: x [B, K*32, Dw] -> [B, K, 128, Dw] contiguous (exact tier)."""
    B, N, Dw = x.shape
    M = K * 32
    assert N == M, (N, K)
    x = x.contiguous()
    out = torch.empty((B, K, 128, Dw), dtype=x.dtype, device=x.device)
    if x.is_cuda and triton_ok():
        total = B * K * 128
        BD = 128 if Dw >= 128 else max(16, triton.next_power_of_2(Dw))
        BR = 32 if Dw >= 64 else 128
        grid = (triton.cdiv(total, BR), triton.cdiv(Dw, BD))
        _k_gather_keys[grid](x, out, M, K, Dw, total, BR=BR, BD=BD, num_warps=4)
        return out
    return gather_keys_torch(x, K)


def gather_rows(x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """bmm(one_hot, x) as a gather: x [Bm, NTOK, Dw], tok [Bm, M] int32 (-1 = empty row) -> [Bm, M, Dw] (exact tier)."""
    Bm, NT, Dw = x.shape
    M = tok.shape[1]
    assert tok.shape[0] == Bm
    x = x.contiguous(); tok = tok.contiguous()
    out = torch.empty((Bm, M, Dw), dtype=x.dtype, device=x.device)
    if x.is_cuda and triton_ok():
        total = Bm * M
        BD = 128 if Dw >= 128 else max(16, triton.next_power_of_2(Dw))
        BR = 32
        grid = (triton.cdiv(total, BR), triton.cdiv(Dw, BD))
        _k_gather_rows[grid](x, tok, out, M, NT, Dw, total, BR=BR, BD=BD, num_warps=4)
        return out
    return gather_rows_torch(x, tok)


# ======================================================== torch oracles (CPU/GPU) ======================================================
def key_index(K: int, device=None) -> torch.Tensor:
    """[K*128] source atom index of each key slot (32k-48+s), -1 outside [0, 32K)."""
    k = torch.arange(K, device=device).view(K, 1); s = torch.arange(128, device=device).view(1, 128)
    src = k * 32 - 48 + s
    src = torch.where((src >= 0) & (src < K * 32), src, torch.full_like(src, -1))
    return src.view(-1)


def gather_keys_torch(x: torch.Tensor, K: int) -> torch.Tensor:
    B, N, Dw = x.shape
    idx = key_index(K, x.device)
    ok = idx >= 0
    out = x[:, idx.clamp(min=0), :]
    out = torch.where(ok.view(1, -1, 1), out, torch.zeros((), dtype=x.dtype, device=x.device))
    out = torch.where(out == 0, torch.zeros((), dtype=x.dtype, device=x.device), out)      # +0.0 canonical
    return out.view(B, K, 128, Dw).contiguous()


def gather_rows_torch(x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    Bm, NT, Dw = x.shape
    ok = tok >= 0
    idx = tok.clamp(min=0).long()
    out = torch.gather(x, 1, idx.unsqueeze(-1).expand(Bm, tok.shape[1], Dw))
    out = torch.where(ok.unsqueeze(-1), out, torch.zeros((), dtype=x.dtype, device=x.device))
    out = torch.where(out == 0, torch.zeros((), dtype=x.dtype, device=x.device), out)
    return out.contiguous()
