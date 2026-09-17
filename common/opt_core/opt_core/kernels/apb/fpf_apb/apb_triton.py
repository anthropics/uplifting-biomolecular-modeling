"""apb_triton.py — flash attention with an additive pair bias shared across a sample batch (Triton).

o[s, n, h, :] = sum_m softmax_m( scale * q[s,n,h,:].k[s,m,h,:] + bias[h,n,m] ) v[s,m,h,:]   (optionally  o *= sigmoid(g[s,n,h,:]))

* q/k/v/g are addressed as [S, N, H, D] through explicit strides (unit stride on D): column blocks of one fused GEMM output or stock's
  per-projection views — no permute / contiguous copies are made here.
* bias [H, N, N] with any row pitch >= N (fp32 or bf16), read per (q-tile, k-tile) and shared by the S programs of that tile through L2
  (grid x = sample index fastest => the S CTAs of one (q-tile, head) are co-scheduled).
* head dim D handled as D1 + D2 (powers of two, e.g. 48 = 32 + 16) => no padded MMA columns; or padded (DP >= D, one dot) when D2 == 0.
* fp32 accumulation, fp32 running max / sum; exp2 with scale*log2(e) folded; MMA operand dtype = input dtype (bf16 / fp16) or, for fp32
  inputs, PREC = "tf32" | "tf32x3" | "ieee" (tl.dot input_precision).
Numerics class: TOLERANCE (tensor-core operands; reduction order differs from the cutlass fp32 kernel stock dispatches to).
"""
from __future__ import annotations
import math, os
import torch
import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _TD  # noqa: F401  (triton >= 3.3)
    _HAS_TMA = True
except Exception:                                                             # pragma: no cover
    _HAS_TMA = False


@triton.jit
def _rn_tf32(x):
    """fp32 -> nearest TF32 value (10-bit mantissa, ties away from zero) kept in an fp32 container = the `cvt.rna.tf32.f32` instruction cuBLAS TF32
    GEMMs apply to their operands (a plain tl.dot "tf32" truncates instead, ~3x further from fp64). The hardware conversion propagates NaN and
    +-inf. (0.2.2: replaces the integer form `(u + 0x1000) & 0xFFFFE000`, which gives the same bits for every finite input but mapped the tensor
    cores' canonical NaN 0x7FFFFFFF to -0.0 — a NaN in k/v came out as a finite, attention-free output.)"""
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rn_tf32_int(x):
    """The 0.2.1 integer form, kept for the old-vs-new identity test only (tests/test_atom_triton.py --legacy-rn)."""
    u = x.to(tl.uint32, bitcast=True)
    u = (u + 0x1000) & 0xFFFFE000
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _cast_opd(x, OPD: tl.constexpr):
    if OPD == 1:
        x = x.to(tl.bfloat16)
    elif OPD == 2:
        x = x.to(tl.float16)
    elif OPD == 3:
        x = _rn_tf32(x)
    elif OPD == 5:
        x = _rn_tf32_int(x)                                   # legacy (test only)
    return x


@triton.jit
def _apb_inner(q1, q2, kb, vb, bb, Bdesc, lo, hi, h, pid_m, offs_m, m_mask, offs_d1, offs_d2,
               skn, svn, sbm, N, c_qk, c_b, m_i, l_i, acc1, acc2,
               D: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               PREC: tl.constexpr, OPD: tl.constexpr, HAS_BIAS: tl.constexpr, BIAS_TMA: tl.constexpr, MASKED: tl.constexpr):
    """Online-softmax over the key tiles [lo, hi). MASKED: tiles may run past N (row-masked loads, -inf columns) — the ragged tail."""
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        if MASKED:
            k1 = tl.load(kb + offs_n[:, None] * skn + offs_d1[None, :], mask=n_mask[:, None] & (offs_d1[None, :] < D), other=0.0)
        elif D1 > D:
            k1 = tl.load(kb + offs_n[:, None] * skn + offs_d1[None, :], mask=offs_d1[None, :] < D, other=0.0)
        else:
            k1 = tl.load(kb + offs_n[:, None] * skn + offs_d1[None, :])
        k1 = _cast_opd(k1, OPD)
        qk = tl.dot(q1, tl.trans(k1), input_precision=PREC)
        if D2 > 0:
            if MASKED:
                k2 = tl.load(kb + offs_n[:, None] * skn + offs_d2[None, :], mask=n_mask[:, None], other=0.0)
            else:
                k2 = tl.load(kb + offs_n[:, None] * skn + offs_d2[None, :])
            k2 = _cast_opd(k2, OPD)
            qk = tl.dot(q2, tl.trans(k2), qk, input_precision=PREC)
        if HAS_BIAS:
            if BIAS_TMA:
                bias = Bdesc.load([h * N + pid_m * BLOCK_M, start_n])
            elif MASKED:
                bias = tl.load(bb + offs_m[:, None] * sbm + offs_n[None, :], mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            else:
                bias = tl.load(bb + offs_m[:, None] * sbm + offs_n[None, :], mask=m_mask[:, None], other=0.0)
            sc = qk * c_qk + bias.to(tl.float32) * c_b
        else:
            sc = qk * c_qk
        if MASKED:
            sc = tl.where(n_mask[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        if MASKED:
            v1 = tl.load(vb + offs_n[:, None] * svn + offs_d1[None, :], mask=n_mask[:, None] & (offs_d1[None, :] < D), other=0.0)
        elif D1 > D:
            v1 = tl.load(vb + offs_n[:, None] * svn + offs_d1[None, :], mask=offs_d1[None, :] < D, other=0.0)
        else:
            v1 = tl.load(vb + offs_n[:, None] * svn + offs_d1[None, :])
        v1 = _cast_opd(v1, OPD)
        pc = _cast_opd(p.to(v1.dtype), OPD)
        acc1 = tl.dot(pc, v1, acc1 * alpha[:, None], input_precision=PREC)
        if D2 > 0:
            if MASKED:
                v2 = tl.load(vb + offs_n[:, None] * svn + offs_d2[None, :], mask=n_mask[:, None], other=0.0)
            else:
                v2 = tl.load(vb + offs_n[:, None] * svn + offs_d2[None, :])
            v2 = _cast_opd(v2, OPD)
            acc2 = tl.dot(pc, v2, acc2 * alpha[:, None], input_precision=PREC)
    return m_i, l_i, acc1, acc2


@triton.jit(do_not_specialize=["N", "sbm", "c_qk", "c_b"])
def _apb_fwd(Q, K, V, G, B, O, Bdesc,
             sqs, sqn, sqh, sks, skn, skh, svs, svn, svh, sgs, sgn, sgh,
             sbh, sbm,
             sos, son, soh,
             N, c_qk, c_b,
             H: tl.constexpr, D: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             GATE: tl.constexpr, OUT_F32: tl.constexpr, PREC: tl.constexpr, OPD: tl.constexpr,
             HAS_BIAS: tl.constexpr, BIAS_TMA: tl.constexpr):
    s_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d1 = tl.arange(0, D1)
    m_mask = offs_m < N
    # ---- Q tile(s)
    qb = Q + s_idx * sqs + h * sqh
    q1 = tl.load(qb + offs_m[:, None] * sqn + offs_d1[None, :], mask=m_mask[:, None] & (offs_d1[None, :] < D), other=0.0)
    q1 = _cast_opd(q1, OPD)
    if D2 > 0:
        offs_d2 = D1 + tl.arange(0, D2)
        q2 = tl.load(qb + offs_m[:, None] * sqn + offs_d2[None, :], mask=m_mask[:, None], other=0.0)
        q2 = _cast_opd(q2, OPD)
    kb = K + s_idx * sks + h * skh
    vb = V + s_idx * svs + h * svh
    bb = B + h * sbh
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, D1], dtype=tl.float32)
    if D2 > 0:
        acc2 = tl.zeros([BLOCK_M, D2], dtype=tl.float32)
    if D2 == 0:
        q2 = q1; acc2 = acc1; offs_d2 = offs_d1                         # placeholders (unused when D2 == 0)
    N_full = (N // BLOCK_N) * BLOCK_N
    # full key tiles [0, N_full): no row masks anywhere; then the ragged tail tile (if any): masked loads + -inf columns
    m_i, l_i, acc1, acc2 = _apb_inner(q1, q2, kb, vb, bb, Bdesc, 0, N_full, h, pid_m, offs_m, m_mask, offs_d1, offs_d2,
                                      skn, svn, sbm, N, c_qk, c_b, m_i, l_i, acc1, acc2,
                                      D, D1, D2, BLOCK_M, BLOCK_N, PREC, OPD, HAS_BIAS, BIAS_TMA, False)
    if N_full < N:                                                          # runtime branch: N is not specialised (one compile per cell)
        m_i, l_i, acc1, acc2 = _apb_inner(q1, q2, kb, vb, bb, Bdesc, N_full, N, h, pid_m, offs_m, m_mask, offs_d1, offs_d2,
                                          skn, svn, sbm, N, c_qk, c_b, m_i, l_i, acc1, acc2,
                                          D, D1, D2, BLOCK_M, BLOCK_N, PREC, OPD, HAS_BIAS, BIAS_TMA, True)
    inv_l = 1.0 / l_i
    o1 = acc1 * inv_l[:, None]
    ob = O + s_idx * sos + h * soh
    if GATE:
        gb = G + s_idx * sgs + h * sgh
        g1 = tl.load(gb + offs_m[:, None] * sgn + offs_d1[None, :], mask=m_mask[:, None] & (offs_d1[None, :] < D), other=0.0).to(tl.float32)
        o1 = o1 * (1.0 / (1.0 + tl.exp2(-g1 * 1.4426950408889634)))
    if OUT_F32:
        tl.store(ob + offs_m[:, None] * son + offs_d1[None, :], o1, mask=m_mask[:, None] & (offs_d1[None, :] < D))
    else:
        tl.store(ob + offs_m[:, None] * son + offs_d1[None, :], o1.to(O.dtype.element_ty), mask=m_mask[:, None] & (offs_d1[None, :] < D))
    if D2 > 0:
        o2 = acc2 * inv_l[:, None]
        if GATE:
            g2 = tl.load(gb + offs_m[:, None] * sgn + offs_d2[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            o2 = o2 * (1.0 / (1.0 + tl.exp2(-g2 * 1.4426950408889634)))
        if OUT_F32:
            tl.store(ob + offs_m[:, None] * son + offs_d2[None, :], o2, mask=m_mask[:, None])
        else:
            tl.store(ob + offs_m[:, None] * son + offs_d2[None, :], o2.to(O.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _ld2d(base, offs_r, stride_r, offs_c, rmask, cmask, OPD: tl.constexpr):
    x = tl.load(base + offs_r[:, None] * stride_r + offs_c[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0)
    x = _cast_opd(x, OPD)
    return x


@triton.jit
def _softmax_step(sc, m_i, l_i):
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(sc - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    return p, m_new, l_i, alpha


@triton.jit
def _apb_fwd_sg2(Q, K, V, G, B, O,
             sqs, sqn, sqh, sks, skn, skh, svs, svn, svh, sgs, sgn, sgh,
             sbh, sbm,
             sos, son, soh,
             S, N, c_qk, c_b,
             H: tl.constexpr, D: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             GATE: tl.constexpr, OUT_F32: tl.constexpr, PREC: tl.constexpr, OPD: tl.constexpr,
             EVEN_N: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Two samples per program share every bias tile (halves the dominant bias stream); sample index clamps at S-1 for an odd S."""
    sa = tl.program_id(0) * 2
    sb = tl.minimum(sa + 1, S - 1)
    pid_m = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d1 = tl.arange(0, D1)
    d1_mask = offs_d1 < D
    m_mask = offs_m < N
    qa1 = _ld2d(Q + sa * sqs + h * sqh, offs_m, sqn, offs_d1, m_mask, d1_mask, OPD)
    qb1 = _ld2d(Q + sb * sqs + h * sqh, offs_m, sqn, offs_d1, m_mask, d1_mask, OPD)
    if D2 > 0:
        offs_d2 = D1 + tl.arange(0, D2)
        d2_mask = offs_d2 < D
        qa2 = _ld2d(Q + sa * sqs + h * sqh, offs_m, sqn, offs_d2, m_mask, d2_mask, OPD)
        qb2 = _ld2d(Q + sb * sqs + h * sqh, offs_m, sqn, offs_d2, m_mask, d2_mask, OPD)
    ka = K + sa * sks + h * skh; kb_ = K + sb * sks + h * skh
    va = V + sa * svs + h * svh; vb_ = V + sb * svs + h * svh
    bb = B + h * sbh
    m_a = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32); l_a = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_b = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32); l_b = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_a1 = tl.zeros([BLOCK_M, D1], dtype=tl.float32); acc_b1 = tl.zeros([BLOCK_M, D1], dtype=tl.float32)
    if D2 > 0:
        acc_a2 = tl.zeros([BLOCK_M, D2], dtype=tl.float32); acc_b2 = tl.zeros([BLOCK_M, D2], dtype=tl.float32)
    offs_nb = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_nb
        n_mask = offs_n < N
        if HAS_BIAS:
            if EVEN_N:
                bias = tl.load(bb + offs_m[:, None] * sbm + offs_n[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32) * c_b
            else:
                bias = tl.load(bb + offs_m[:, None] * sbm + offs_n[None, :], mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32) * c_b
        # ---- sample a
        k1 = _ld2d(ka, offs_n, skn, offs_d1, n_mask, d1_mask, OPD)
        qk = tl.dot(qa1, tl.trans(k1), input_precision=PREC)
        if D2 > 0:
            k2 = _ld2d(ka, offs_n, skn, offs_d2, n_mask, d2_mask, OPD)
            qk = tl.dot(qa2, tl.trans(k2), qk, input_precision=PREC)
        if HAS_BIAS:
            sc = qk * c_qk + bias
        else:
            sc = qk * c_qk
        if not EVEN_N:
            sc = tl.where(n_mask[None, :], sc, -float("inf"))
        p, m_a, l_a, alpha = _softmax_step(sc, m_a, l_a)
        v1 = _ld2d(va, offs_n, svn, offs_d1, n_mask, d1_mask, OPD)
        pc = _cast_opd(p.to(v1.dtype), OPD)
        acc_a1 = tl.dot(pc, v1, acc_a1 * alpha[:, None], input_precision=PREC)
        if D2 > 0:
            v2 = _ld2d(va, offs_n, svn, offs_d2, n_mask, d2_mask, OPD)
            acc_a2 = tl.dot(pc, v2, acc_a2 * alpha[:, None], input_precision=PREC)
        # ---- sample b (same bias tile)
        k1 = _ld2d(kb_, offs_n, skn, offs_d1, n_mask, d1_mask, OPD)
        qk = tl.dot(qb1, tl.trans(k1), input_precision=PREC)
        if D2 > 0:
            k2 = _ld2d(kb_, offs_n, skn, offs_d2, n_mask, d2_mask, OPD)
            qk = tl.dot(qb2, tl.trans(k2), qk, input_precision=PREC)
        if HAS_BIAS:
            sc = qk * c_qk + bias
        else:
            sc = qk * c_qk
        if not EVEN_N:
            sc = tl.where(n_mask[None, :], sc, -float("inf"))
        p, m_b, l_b, alpha = _softmax_step(sc, m_b, l_b)
        v1 = _ld2d(vb_, offs_n, svn, offs_d1, n_mask, d1_mask, OPD)
        pc = _cast_opd(p.to(v1.dtype), OPD)
        acc_b1 = tl.dot(pc, v1, acc_b1 * alpha[:, None], input_precision=PREC)
        if D2 > 0:
            v2 = _ld2d(vb_, offs_n, svn, offs_d2, n_mask, d2_mask, OPD)
            acc_b2 = tl.dot(pc, v2, acc_b2 * alpha[:, None], input_precision=PREC)
    # ---- epilogue (sample a, then b)
    _epilogue(acc_a1, l_a, O + sa * sos + h * soh, G + sa * sgs + h * sgh, offs_m, son, sgn, offs_d1, m_mask, d1_mask, GATE, OUT_F32)
    _epilogue(acc_b1, l_b, O + sb * sos + h * soh, G + sb * sgs + h * sgh, offs_m, son, sgn, offs_d1, m_mask, d1_mask, GATE, OUT_F32)
    if D2 > 0:
        _epilogue(acc_a2, l_a, O + sa * sos + h * soh, G + sa * sgs + h * sgh, offs_m, son, sgn, offs_d2, m_mask, d2_mask, GATE, OUT_F32)
        _epilogue(acc_b2, l_b, O + sb * sos + h * soh, G + sb * sgs + h * sgh, offs_m, son, sgn, offs_d2, m_mask, d2_mask, GATE, OUT_F32)


@triton.jit
def _epilogue(acc, l_i, ob, gb, offs_m, son, sgn, offs_d, m_mask, d_mask, GATE: tl.constexpr, OUT_F32: tl.constexpr):
    o = acc * (1.0 / l_i)[:, None]
    msk = m_mask[:, None] & d_mask[None, :]
    if GATE:
        g = tl.load(gb + offs_m[:, None] * sgn + offs_d[None, :], mask=msk, other=0.0).to(tl.float32)
        o = o * (1.0 / (1.0 + tl.exp2(-g * 1.4426950408889634)))
    if OUT_F32:
        tl.store(ob + offs_m[:, None] * son + offs_d[None, :], o, mask=msk)
    else:
        tl.store(ob + offs_m[:, None] * son + offs_d[None, :], o.to(ob.dtype.element_ty), mask=msk)


_LOG2E = 1.4426950408889634
_CFG = {}
LAST_LAUNCH = {}   # the most recent launch's resolved cell (lever report / tests)   # (D, opd) -> dict(BLOCK_M, BLOCK_N, num_warps, num_stages, D1, D2)


def apb_config(D: int, opd: str, in_fp32: bool = False, **override):
    """The launch cell for (head dim, operand dtype, fp32-input?); overridable for sweeps. Defaults chosen on H100 (see MICROBENCH).
    fp32 inputs (in-kernel downcast or tf32 operands) use one padded dot (D1 = next pow2 >= D, D2 = 0): the split-D form with fp32 loads
    miscompiles on triton 3.7 (wrong values / illegal address); 16-bit inputs use the split form (48 = 32 + 16: no padded MMA columns)."""
    key = (D, opd, bool(in_fp32))
    if override:
        _CFG[key] = dict(override)
    if key not in _CFG:
        p = 1 << (D - 1).bit_length()
        if D == 48 and not in_fp32:      # H100 sweep (tools/sweep_dit.py): 64x64 tile, one warpgroup, 2 stages, 48 = 32 + 16, bias tile via TMA descriptor
            _CFG[key] = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, D1=32, D2=16, SG=1, BIAS_TMA=True)
        elif D == 48:                     # fp32 inputs: padded single dot (split form miscompiles with fp32 loads on triton 3.7)
            _CFG[key] = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, D1=64, D2=0, SG=1, BIAS_TMA=True)
        else:
            _CFG[key] = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, D1=max(16, p), D2=0, SG=1, BIAS_TMA=True)
    return _CFG[key]


LAUNCH_OPTION_KEYS = ("num_warps", "num_stages", "num_ctas", "maxnreg")      # triton launch options a cfg may carry besides the cell's keys


def resolve_cfg(cfg: dict | None, D: int, opd: str, in_fp32: bool = False) -> dict:
    """The launch cell of one call: ``apb_config(D, opd, in_fp32)``'s cell with the caller's ``cfg`` entries laid over it KEY BY KEY --
    a partial cfg keeps the cell's other keys (``{"BIAS_TMA": False}`` alone is a valid cfg; before, any cfg replaced the whole cell and a
    partial one failed with KeyError 'D1'); ``None`` / ``{}`` = the cell unchanged; a full cfg = exactly that cfg, as before.  Keys that are
    neither the cell's nor a launch option raise ValueError by name.  Returns a fresh dict (the caller pops from it)."""
    base = apb_config(D, opd, in_fp32)
    if not cfg:
        return dict(base)
    unknown = sorted(set(cfg) - set(base) - set(LAUNCH_OPTION_KEYS))
    if unknown:
        raise ValueError(f"ptx_apb: unknown cfg keys {unknown} (cell keys {sorted(base)}, launch options {list(LAUNCH_OPTION_KEYS)})")
    return {**base, **cfg}


def _opd_code(dtype: torch.dtype, opd: str | None):
    """operand dtype selector: 0 = as loaded (bf16/fp16 inputs, or fp32 inputs with PREC), 1 = cast to bf16 in-kernel, 2 = cast to fp16 in-kernel."""
    if dtype in (torch.bfloat16, torch.float16):
        return 0, ("bf16" if dtype == torch.bfloat16 else "fp16"), "ieee"
    opd = opd or "bf16"
    if opd == "bf16":
        return 1, "bf16", "ieee"
    if opd == "fp16":
        return 2, "fp16", "ieee"
    if opd in ("tf32", "tf32x3", "ieee"):
        return 0, opd, opd
    if opd == "tf32rn":
        return 3, "tf32rn", "tf32"
    if opd == "tf32rn_legacy":
        return 5, "tf32rn_legacy", "tf32"
    raise ValueError(f"ptx_apb: unknown operand precision {opd!r} for fp32 inputs (bf16|fp16|tf32|tf32rn|tf32x3|ieee)")


def precast16(q, k, v, dtype=torch.float16, bufs=None):
    """fp32 [S,N,H,D]-addressable views -> contiguous 16-bit copies (3 copy kernels, ~8 us each @800 tokens). Feeding the 16-bit split-D kernel
    with these is 1.3-1.45x faster per call than casting fp32 tiles in-kernel (padded single dot) and gives bitwise the same result."""
    outs = []
    for i, t in enumerate((q, k, v)):
        b = bufs[i] if bufs is not None else torch.empty(t.shape, device=t.device, dtype=dtype)
        b.copy_(t)
        outs.append(b)
    return tuple(outs)


def apb_views(q, k, v, bias, g=None, *, scale=None, out=None, out_dtype=None, opd: str | None = None, cfg: dict | None = None, _no_bias: bool = False):
    """q,k,v(,g): tensors indexable as [S, N, H, D] (any strides, unit stride on D); bias [H,N,N] or [1,H,N,N] (row pitch >= N).
    Returns o [S, N, H, D] contiguous (out_dtype, default = q.dtype)."""
    assert q.dim() == 4 and k.shape == q.shape and v.shape == q.shape, (q.shape, k.shape, v.shape)
    S, N, H, D = q.shape
    if bias.dim() == 4:
        assert bias.shape[0] == 1, bias.shape
        bias = bias[0]
    assert bias.shape[0] == H and bias.shape[1] == N and bias.shape[2] >= N and bias.stride(2) == 1, (bias.shape, bias.stride())
    for t in (q, k, v) + ((g,) if g is not None else ()):
        assert t.stride(3) == 1, t.stride()
    assert k.dtype == q.dtype and v.dtype == q.dtype, (q.dtype, k.dtype, v.dtype)
    assert g is None or g.dtype in (q.dtype, torch.float32), g.dtype          # the gate logits may stay fp32 when q/k/v were pre-cast to 16 bit (read once, sigmoid in fp32)
    assert q.is_cuda and bias.is_cuda
    opd_code, opd_name, prec = _opd_code(q.dtype, opd)
    c = resolve_cfg(cfg, D, opd_name, q.dtype == torch.float32)                  # the cell, the caller's (possibly partial) cfg merged over it
    D1, D2 = c.pop("D1"), c.pop("D2")
    assert D1 + D2 >= D and (D2 == 0 or D1 + D2 == D), (D, D1, D2)
    out_dtype = out_dtype or q.dtype
    if out is None:
        out = torch.empty((S, N, H, D), device=q.device, dtype=out_dtype)
    assert out.shape == (S, N, H, D) and out.stride(3) == 1
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    BLOCK_M, BLOCK_N = c.pop("BLOCK_M"), c.pop("BLOCK_N")
    SG = c.pop("SG", 2 if S > 1 else 1)
    gg = g if g is not None else q
    tma = bool(c.pop("BIAS_TMA", False)) and not _no_bias
    extra = {}
    if SG == 2 and S > 1:
        grid = (triton.cdiv(S, 2), triton.cdiv(N, BLOCK_M), H)
        kern = _apb_fwd_sg2
        nargs = (S, N)
        extra = dict(EVEN_N=(N % BLOCK_N == 0))
        head = (q, k, v, gg, bias, out)
    else:
        grid = (S, triton.cdiv(N, BLOCK_M), H)
        kern = _apb_fwd
        nargs = (N,)
        bdesc = None
        realigned = False
        if tma:
            elt = bias.element_size()
            if (bias.stride(1) * elt) % 16 != 0 or bias.data_ptr() % 16 != 0 or bias.stride(0) != N * bias.stride(1):
                # a bias whose row pitch is not 16-byte aligned (N_token % 4 != 0 for fp32, % 8 for bf16, stored densely) cannot be a TMA
                # source and makes every plain tile load unaligned (measured 4-5x kernel time): re-store it once with the pitch rounded up to
                # 8 elements (one extra pass over the bias; inside a captured graph this copy is captured too, so replays re-read the source).
                P8 = (N + 7) // 8 * 8
                buf = torch.empty((H, N, P8), device=bias.device, dtype=bias.dtype)
                buf[:, :, :N].copy_(bias[:, :, :N])
                bias = buf[:, :, :N]
                realigned = True
            tma = _HAS_TMA
        if tma:
            from triton.tools.tensor_descriptor import TensorDescriptor
            bdesc = TensorDescriptor(bias, [H * N, N], [bias.stride(1), 1], [BLOCK_M, BLOCK_N])
        head = (q, k, v, gg, bias, out, bdesc)
        extra = dict(BIAS_TMA=tma)                                            # (EVEN_N is a constexpr of the sg2 variant only)
        LAST_LAUNCH["bias_realigned"] = realigned
    LAST_LAUNCH.update(kernel=kern.fn.__name__ if hasattr(kern, "fn") else str(kern), grid=grid, bias_tma=tma, bias_dtype=str(bias.dtype), in_dtype=str(q.dtype), opd=opd_name, prec=prec, cfg=dict(c, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D1=D1, D2=D2))
    kern[grid](*head,
                   q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                   gg.stride(0), gg.stride(1), gg.stride(2),
                   bias.stride(0), bias.stride(1),
                   out.stride(0), out.stride(1), out.stride(2),
                   *nargs, scale * _LOG2E, _LOG2E,
                   H=H, D=D, D1=D1, D2=D2, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                   GATE=g is not None, OUT_F32=out_dtype == torch.float32, PREC=prec, OPD=opd_code,
                   HAS_BIAS=not _no_bias, **extra, **c)
    return out


def dit_apb(qkvg, bias, S: int, N: int, *, H: int = 16, D: int = 48, scale=None, gate=True, out=None, out_dtype=None, opd=None, cfg=None):
    """The fused-projection signature: qkvg [S*N, 4*H*D] (q | k | v | g column blocks, q bias added, not pre-scaled) -> o [S*N, H*D]."""
    C = H * D
    assert qkvg.shape == (S * N, 4 * C) or (not gate and qkvg.shape == (S * N, 3 * C)), qkvg.shape
    x = qkvg.view(S, N, qkvg.shape[1])
    q = x[:, :, 0 * C:1 * C].unflatten(2, (H, D))
    k = x[:, :, 1 * C:2 * C].unflatten(2, (H, D))
    v = x[:, :, 2 * C:3 * C].unflatten(2, (H, D))
    g = x[:, :, 3 * C:4 * C].unflatten(2, (H, D)) if gate else None
    o4 = None if out is None else out.view(S, N, H, D)
    o = apb_views(q, k, v, bias, g, scale=scale, out=o4, out_dtype=out_dtype, opd=opd, cfg=cfg)
    return o.view(S * N, C)
