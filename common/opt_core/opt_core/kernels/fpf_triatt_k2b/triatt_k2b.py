"""triatt_k2b.py -- FPF TriAtt library kernel K2B = K2 (flash_triattn_k2.py: the flash_triattn.py kernels, 16-bit and fp32 inputs)
with three measured small wins frozen in, nothing else changed:

  (a) base-2 softmax domain is the DEFAULT for 16-bit (bf16/fp16) inputs: log2(e) is folded into the softmax scale and into
      the fp32 bias tile once per tile load, p = ex2(t - m) with the difference (t - m) formed BEFORE anything else (so an
      unchanged running max gives alpha == 1 exactly and a fully-masked row -- all logits == -1e9*log2e == m -- gives p == 1
      exactly = the uniform average of V, cuEquivariance/stock -1e9 semantics; no inf/NaN).  fp32 inputs (TF32 path) keep
      K2's natural-domain default.  Same rounding points as K2/cuEq: bf16 QK^T and P.V on tensor cores with fp32
      accumulation, fp32 statistics; error vs an fp64 reference is the same class as cuEq (ratio ~1.0).
  (b) ptxas register cap MAXNREG passed at launch (K2 compiles to 166 regs/thread = 3 CTAs/SM on H100; 128 -> 4 CTAs/SM).
      Register allocation only -- results are bit-identical to the uncapped compile (checked in the sweep that chose it).
  (c) num_stages re-chosen under that register budget (swept 2/3/4 on H100, triton 3.7 and 3.3.1).
  Tile shape / rows-per-program / warps / grid order (ORDER=0) / 16-bit lossless bias tiles / mask fast path = K2, unchanged.
Deterministic: no atomics, no runtime autotuning; config is the frozen CONFIG below (env K2B_CONFIG or config= kwarg override
exist for timing only).  Call convention = cuequivariance triangle_attention / flash_triangle_attention:
    attn(q, k, v, bias, mask=None, scale=None) -> out [B, N, H, SQ, D]   (see flash_triangle_attention docstring below).
"""
from __future__ import annotations

# Frozen choices (measured on H100 SXM, D=32, N=356..813, H=4/8; the measurements are summarised below CONFIG)
def _triton_ge(major: int, minor: int) -> bool:
    try:
        import triton as _t
        v = tuple(int(x) for x in _t.__version__.split("+")[0].split(".")[:2])
        return v >= (major, minor)
    except Exception:
        return False


CONFIG = {
    "BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "ORDER": 0,   # = K2 tile config (optimum of a 288-point tile grid)
    "num_stages": 3,                                                    # re-swept {2,3,4} x MAXNREG {none,128,144,160}: 3 wins on both stacks
    # ptxas register cap at launch (allocation only; EXP_MODE-0 compiles were bit-exact == uncapped K2 on both stacks).
    # Stack-keyed because the spill cost differs by compiler: triton 3.7: 128 regs (6 spill slots, 4 CTAs/SM) = x1.20 over
    # K2 @705 vs 144: x1.10;  triton 3.3.1: 128 regs spills 20 slots and gives only x1.06, 144 (6 slots) = x1.10 -> best.
    "MAXNREG": 128 if _triton_ge(3, 7) else 144,
    "MAXNREG_BY_TRITON": {">=3.7": 128, "<3.7 (measured 3.3.1)": 144},   # documentation of the rule above
    "EXP_MODE_16BIT": 1,                                                # 1 = base-2 domain (default for bf16/fp16 inputs); 0 = K2's natural domain
    "EXP_MODE_FP32": 0,                                                 # fp32 inputs (TF32 path): unchanged from K2 (uncapped, natural domain)
    "BIAS16_MIN_SK": 256,                                               # lossless 16-bit bias tiles from this S_kv on (= K2)
}
K2B_BASE = "flash_triattn.py (K2) sha256[:12]=d2775231db73"
# Measured (H100 80GB HBM3, bf16, D=32, synthetic model-like inputs, median of 30, K2 timed in the same run):
#   triton 3.7.0 / torch 2.12:  N=705 H=4 start 1.166 ms (K2 1.403, cuEq 0.10: 2.945)  H=8 start 2.262 ms (K2 2.691, cuEq 5.822)
#                              x over K2 = 1.15-1.21 over N=356..813, H=4/8, start/end (geomean 1.18); 154-175 TF/s at N>=705
#   triton 3.3.1 / torch 2.7.1: N=705 H=4 start 1.513 ms (K2 1.679, cuEq 0.8: 2.963) H=8 start 2.921 ms (K2 3.234, cuEq 5.845)
#                              x over K2 = 1.06-1.18 (geomean 1.10)
#   error vs fp64 reference relative to cuEq's own error: rms 0.996-0.998, max 1.00 (= K2's class); r2r bit-exact, batch- and
#   co-tenancy-invariant in every case; masked inputs (10% keys + one fully-masked row): same ratios, fully-masked row =
#   uniform mean of V (fp32 -1e9 semantics, bit-exact == K2).

_K2_DOC = """Library kernel K2: flash triangle attention with pair bias (forward), 16-bit and fp32 inputs -- the same Triton kernels as
opt_core/kernels/flash_triattn.py (identical arithmetic) with per-device frozen config tables in place of its per-capability rows
(PF_TRIATTN_TABLE json override; device class detected from torch.cuda.get_device_name) and env alias PF_FLASH_IP == OF3T_FLASH_IP.
Credit: the authors of that flash triangle-attention kernel.

fp32 inputs: the two dots run at the tl.dot input_precision constexpr IP ('tf32' = the stock OpenFold3 predict-preset class, which
runs torch.einsum under float32_matmul_precision=high; 'tf32x3'; 'ieee'), with an fp32 config table (_CONFIG_TABLE_F32) and the
input_precision kwarg; IP has no effect on 16-bit inputs. Everything else below holds for both input classes.


flash_triattn.py -- H100-native flash triangle attention with pair bias (forward / inference only), Triton.

Drop-in replacement for  cuequivariance_torch.primitives.triangle.triangle_attention(q, k, v, bias, mask=None, scale=None)
(return_aux=False path), as called by the stock model (its triangular/layers.py::cuequivariance_triangular_attn):

    out[b, i, h, q, :] = softmax_j( scale * q[b,i,h,q,:] . k[b,i,h,j,:] + bias[b,0,h,q,j]  (+ -1e9 where mask[b,i,0,0,j] == False) ) @ v[b,i,h,j,:]

Shapes:  q [B, N, H, SQ, D], k/v [B, N, H, SK, D] (bf16/fp16, arbitrary strides with stride(-1)==1),
         bias [B, 1, H, SQ, SK] (cast to fp32; shared across the row dimension N), mask [B, N, 1, 1, SK] bool (True = keep),
         out [B, N, H, SQ, D] contiguous, dtype = q.dtype  (exactly what cuEq returns for a 5-D input with return_aux=False;
         for <5-D inputs cuEq prepends singleton dims and returns the 5-D result -- mimicked here).

Numerics: bf16 tensor-core products with fp32 accumulation, fp32 online softmax, P cast to v.dtype for the PV product (the
cuEq / FlashAttention scheme); logits s = scale*q.k + bias formed in fp32 exactly as the reference orders them.  Exponent:
    exact_exp=True  (default)          : p = exp(s - m)  (tl.exp: (s-m)*log2e -> MUFU ex2)
    exact_exp=True, use_libdevice=True : p = libm-accurate expf(s - m) (slowest; reference variant)
    exact_exp=False                    : log2(e) folded into the scale and the bias tile (base-2-domain logits), p = ex2(t - m)
                                         (one FMUL per logit fewer; ~1-3% faster per call)
All three have the same error statistics against an fp64 reference.
Bias: one prep pass per call writes an fp32 copy with unit key stride (cuEq makes the same contiguous copy of the model's
permuted view) and, for S_kv >= 256, a 16-bit copy plus per-block flags certifying the fp32->16-bit cast was lossless; the
attention kernel then reads the 16-bit tiles (half the L2 traffic) and reconstructs the identical fp32 values, else the fp32
tiles.  The model computes the triangle bias under bf16 autocast and upcasts it (.float()), so the lossless path is the common
case in the model; either way the arithmetic is bitwise the same.
No atomics, fixed reduction order for a given config -> bitwise reproducible run-to-run (config choice is a deterministic
function of (D, N, H), never runtime-autotuned).

Grid: (q-tiles, row-groups of ROWS rows, B*H) (ORDER=0: q-tile index fastest so CTAs in flight share the K/V rows; ORDER=1:
row-group fastest so they share bias tiles); each program handles ROWS rows so the fp32 bias tile is loaded once per ROWS
QK^T products, with scalar per-row base pointers and shared offset tiles.  When a mask is given, each program first scans the
mask rows it owns (ROWS x S_kv bytes); if every key is unmasked it runs the mask-free loop (bitwise-identical result for such
rows, ~20% fewer instructions per logit), otherwise the masked loop.
"""

import math
import os
from typing import Optional, Tuple, Dict, Any

import torch
import triton
import triton.language as tl

try:  # accurate expf for the 'exact' variant
    from triton.language.extra import libdevice as _libdevice  # triton >= 3.0
    _HAS_LIBDEVICE = True
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _libdevice
        _HAS_LIBDEVICE = True
    except Exception:
        _libdevice = None
        _HAS_LIBDEVICE = False

__all__ = ["flash_triangle_attention", "attn", "CONFIG", "flash_supported", "reference_triangle_attention", "pick_config", "fast_launch_stats"]

_LOG2E = 1.4426950408889634
_MASK_NEG = 1.0e9  # cuEq / stock additive mask value


# --------------------------------------------------------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _bias_prep(
    Bias, Out32, Out16, Flags,
    sbb, sbh, sbq, sbk,
    H, SQ, SK, SKp, NQB,
    PB: tl.constexpr, PK: tl.constexpr, MAKE16: tl.constexpr,
):
    """One pass over the (arbitrarily strided) fp32 bias of head (b,h), query rows [pid_q*PB, +PB):
       Out32[b,0,h,q,k] = bias (row stride SKp, unit key stride, 16-B aligned);  if MAKE16 also Out16 = bias cast to the
       16-bit dtype and Flags[pid] = 1 iff that cast is lossless for every element handled by this program."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    rows = pid_q * PB + tl.arange(0, PB)
    rmask = rows < SQ
    src = Bias + b * sbb + h * sbh
    dst = pid_bh.to(tl.int64) * SQ * SKp
    nbad = tl.zeros([PB], dtype=tl.int32)
    for k0 in range(0, SK, PK):
        cols = k0 + tl.arange(0, PK)
        m2 = rmask[:, None] & (cols < SK)[None, :]
        w2 = rmask[:, None] & (cols < SKp)[None, :]          # also zero-fill the padding columns [SK, SKp): deterministic buffer
        x = tl.load(src + rows[:, None] * sbq + cols[None, :] * sbk, mask=m2, other=0.0).to(tl.float32)
        o = dst + rows[:, None] * SKp + cols[None, :]
        tl.store(Out32 + o, x, mask=w2)
        if MAKE16:
            x16 = x.to(Out16.dtype.element_ty)
            tl.store(Out16 + o, x16, mask=w2)
            nbad += tl.sum((x16.to(tl.float32) != x).to(tl.int32), 1)
    if MAKE16:
        ok = (tl.sum(nbad, 0) == 0).to(tl.int32)
        tl.store(Flags + pid_bh * NQB + pid_q, ok)


@triton.jit
def _row_step(
    acc, m_i, l_i,            # running state of this row: [BM, D] fp32, [BM], [BM]
    q,                        # [BM, D] query tile of this row (bf16/fp16)
    k_base, v_base, m_base,   # scalar pointers: K[b,i,h,0,0], V[b,i,h,0,0], Mask[b,i,0,0,0]
    k_lo, v_lo, m_lo,         # scalar element offsets of this key tile (start_n * stride)
    k_off, v_off, m_off,      # shared loop-invariant offset tiles: [BN, D], [BN, D], [BN] (int32)
    s_bias,                   # [BM, BN] fp32 bias tile (times log2e when EXP_MODE == 1); shared by the rows of the program
    kvm,                      # [BN] bool key-in-range (used only when MASK_N)
    qk_scale,                 # softmax scale (times log2e when EXP_MODE == 1)
    MASK_N: tl.constexpr, APPLY_MASK: tl.constexpr, EXP_MODE: tl.constexpr, IP: tl.constexpr,
):
    if MASK_N:
        k = tl.load(k_base + k_lo + k_off, mask=kvm[:, None], other=0.0)
        v = tl.load(v_base + v_lo + v_off, mask=kvm[:, None], other=0.0)
    else:
        k = tl.load(k_base + k_lo + k_off)
        v = tl.load(v_base + v_lo + v_off)
    s = tl.dot(q, tl.trans(k), input_precision=IP)  # [BM, BN] fp32 on tensor cores (IP applies to fp32 inputs only)
    s = s * qk_scale + s_bias                       # logits (natural-log domain for EXP_MODE 0/2, base-2 domain for 1)
    if APPLY_MASK:
        if MASK_N:
            mk = tl.load(m_base + m_lo + m_off, mask=kvm, other=1)
        else:
            mk = tl.load(m_base + m_lo + m_off)
        # cuEq/stock semantics: logit += -1e9 for masked keys (|logit| << ulp(1e9) so the fp32 sum IS -1e9); a select is
        # used so that a fully-masked row cannot overflow and yields the uniform average like the additive form does in fp32.
        if EXP_MODE == 1:
            s = tl.where(mk[None, :] != 0, s, -1.4426950408889634e9)
        else:
            s = tl.where(mk[None, :] != 0, s, -1.0e9)
    if MASK_N:
        s = tl.where(kvm[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    # NOTE: the difference (s - m) is formed BEFORE any scaling so that an unchanged running max gives alpha == 1 exactly and
    # a fully-masked row (all logits == -1e9 == m) gives p == 1 exactly (uniform average, cuEq semantics).  Folding the
    # log2e multiply into an FFMA (ex2(s*c - m*c)) breaks both properties when |m| is huge and was measured to produce
    # inf/NaN on fully-masked rows -- do not "optimise" this.
    if EXP_MODE == 0:
        alpha = tl.exp(m_i - m_new)                     # lowers to ex2((x)*log2e) on NVIDIA
        p = tl.exp(s - m_new[:, None])
    elif EXP_MODE == 1:
        alpha = tl.math.exp2(m_i - m_new)               # logits already in the base-2 domain
        p = tl.math.exp2(s - m_new[:, None])
    else:
        alpha = _libdevice.exp(m_i - m_new)             # libm-accurate expf (slowest; reference variant)
        p = _libdevice.exp(s - m_new[:, None])
    l_new = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(v.dtype), v, acc, input_precision=IP)
    return acc, m_new, l_new


@triton.jit
def _attend(
    q0, q1, q2, q3,
    kb0, kb1, kb2, kb3, vb0, vb1, vb2, vb3, mb0, mb1, mb2, mb3,
    bias_bh, b_off, k_off, v_off, m_off, offs_n,
    skk, svk, smk, sbq,
    SEQ_K, qk_scale2, bias_scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
    APPLY_MASK: tl.constexpr, EXP_MODE: tl.constexpr, IP: tl.constexpr,
):
    acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc1 = acc0; m1 = m0; l1 = l0
    acc2 = acc0; m2 = m0; l2 = l0
    acc3 = acc0; m3 = m0; l3 = l0
    kv_all = offs_n < BLOCK_N                       # all-true placeholder for full tiles
    # ---- full key tiles ------------------------------------------------------------------------------------------------
    hi = (SEQ_K // BLOCK_N) * BLOCK_N
    for start_n in range(0, hi, BLOCK_N):
        s_bias = tl.load(bias_bh + start_n + b_off).to(tl.float32) * bias_scale     # bias buffer has unit key stride
        k_lo = start_n * skk
        v_lo = start_n * svk
        m_lo = start_n * smk
        acc0, m0, l0 = _row_step(acc0, m0, l0, q0, kb0, vb0, mb0, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kv_all, qk_scale2,
                                 MASK_N=False, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
        if ROWS >= 2:
            acc1, m1, l1 = _row_step(acc1, m1, l1, q1, kb1, vb1, mb1, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kv_all, qk_scale2,
                                     MASK_N=False, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
        if ROWS >= 4:
            acc2, m2, l2 = _row_step(acc2, m2, l2, q2, kb2, vb2, mb2, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kv_all, qk_scale2,
                                     MASK_N=False, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
            acc3, m3, l3 = _row_step(acc3, m3, l3, q3, kb3, vb3, mb3, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kv_all, qk_scale2,
                                     MASK_N=False, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
    # ---- ragged tail tile (bounds-masked; skipped when SEQ_K % BLOCK_N == 0) --------------------------------------------
    if hi < SEQ_K:
        kvm = (hi + offs_n) < SEQ_K
        s_bias = tl.load(bias_bh + hi + b_off, mask=kvm[None, :], other=0.0).to(tl.float32) * bias_scale
        k_lo = hi * skk
        v_lo = hi * svk
        m_lo = hi * smk
        acc0, m0, l0 = _row_step(acc0, m0, l0, q0, kb0, vb0, mb0, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kvm, qk_scale2,
                                 MASK_N=True, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
        if ROWS >= 2:
            acc1, m1, l1 = _row_step(acc1, m1, l1, q1, kb1, vb1, mb1, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kvm, qk_scale2,
                                     MASK_N=True, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
        if ROWS >= 4:
            acc2, m2, l2 = _row_step(acc2, m2, l2, q2, kb2, vb2, mb2, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kvm, qk_scale2,
                                     MASK_N=True, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
            acc3, m3, l3 = _row_step(acc3, m3, l3, q3, kb3, vb3, mb3, k_lo, v_lo, m_lo, k_off, v_off, m_off, s_bias, kvm, qk_scale2,
                                     MASK_N=True, APPLY_MASK=APPLY_MASK, EXP_MODE=EXP_MODE, IP=IP)
    return acc0, m0, l0, acc1, m1, l1, acc2, m2, l2, acc3, m3, l3


@triton.jit
def _flash_triattn_fwd(
    Q, K, V, Bias32, Bias16, Flags, Mask, Out,
    sqb, sqi, sqh, sqq,
    skb, ski, skh, skk,
    svb, svi, svh, svk,
    sbb, sbh, sbq,              # strides of the prepared bias buffers [B,1,H,SQ,SKp] (unit key stride)
    smb, smi, smk,
    sob, soi, soh, soq,
    N_ROWS, SEQ_Q, SEQ_K, H, n_flags,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    EXP_MODE: tl.constexpr,     # 0: natural-domain logits, FFMA+ex2 (default); 1: log2e folded into scale/bias; 2: libdevice expf
    BIAS16: tl.constexpr,       # 1: use the 16-bit copy of the bias when Flags say the fp32->16-bit cast was lossless
    ORDER: tl.constexpr,        # 0: q-tile index fastest (CTAs in flight share rows' K/V); 1: row-group index fastest (share bias tiles)
    MSCAN: tl.constexpr,        # chunk size of the per-program mask-row scan
    NFLAG: tl.constexpr,        # upper bound on the number of bias-prep flags scanned
    IP: tl.constexpr,           # tl.dot input precision for fp32 inputs: 'tf32' (the stock OpenFold3 class: TF32 einsum), 'tf32x3', 'ieee'
):
    if ORDER == 0:
        pid_q = tl.program_id(0)
        pid_r = tl.program_id(1)
    else:
        pid_r = tl.program_id(0)
        pid_q = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)

    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_mc = tl.minimum(offs_m, SEQ_Q - 1)          # clamped query indices for loads (stores are masked)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    if EXP_MODE == 1:
        bias_scale = 1.4426950408889634
        qk_scale2 = qk_scale * 1.4426950408889634
    else:
        bias_scale = 1.0
        qk_scale2 = qk_scale

    # shared, loop-invariant offset tiles (int32) -- rows differ only by a scalar base pointer
    k_off = offs_n[:, None] * skk + offs_d[None, :]
    v_off = offs_n[:, None] * svk + offs_d[None, :]
    m_off = offs_n * smk
    b_off = offs_mc[:, None] * sbq + offs_n[None, :]
    q_off = offs_mc[:, None] * sqq + offs_d[None, :]
    bias32_bh = Bias32 + b * sbb + h * sbh
    bias16_bh = Bias16 + b * sbb + h * sbh

    r0 = pid_r * ROWS
    i0 = tl.minimum(r0 + 0, N_ROWS - 1).to(tl.int64)
    q0 = tl.load(Q + b * sqb + i0 * sqi + h * sqh + q_off)
    kb0 = K + b * skb + i0 * ski + h * skh
    vb0 = V + b * svb + i0 * svi + h * svh
    mb0 = Mask + b * smb + i0 * smi
    q1 = q0; kb1 = kb0; vb1 = vb0; mb1 = mb0
    q2 = q0; kb2 = kb0; vb2 = vb0; mb2 = mb0
    q3 = q0; kb3 = kb0; vb3 = vb0; mb3 = mb0
    if ROWS >= 2:
        i1 = tl.minimum(r0 + 1, N_ROWS - 1).to(tl.int64)
        q1 = tl.load(Q + b * sqb + i1 * sqi + h * sqh + q_off)
        kb1 = K + b * skb + i1 * ski + h * skh
        vb1 = V + b * svb + i1 * svi + h * svh
        mb1 = Mask + b * smb + i1 * smi
    if ROWS >= 4:
        i2 = tl.minimum(r0 + 2, N_ROWS - 1).to(tl.int64)
        q2 = tl.load(Q + b * sqb + i2 * sqi + h * sqh + q_off)
        kb2 = K + b * skb + i2 * ski + h * skh
        vb2 = V + b * svb + i2 * svi + h * svh
        mb2 = Mask + b * smb + i2 * smi
        i3 = tl.minimum(r0 + 3, N_ROWS - 1).to(tl.int64)
        q3 = tl.load(Q + b * sqb + i3 * sqi + h * sqh + q_off)
        kb3 = K + b * skb + i3 * ski + h * skh
        vb3 = V + b * svb + i3 * svi + h * svh
        mb3 = Mask + b * smb + i3 * smi

    # ---- per-program uniform decisions ------------------------------------------------------------------------------------
    # (1) 16-bit bias tiles only if EVERY prep program tested a lossless cast (then the fp32 values reconstructed in
    #     registers are identical to the fp32 bias -> bit-exact the same arithmetic, half the L2 traffic for the bias).
    use16 = False
    if BIAS16:
        foffs = tl.arange(0, NFLAG)
        fmin = tl.min(tl.load(Flags + foffs, mask=foffs < n_flags, other=1), 0)
        use16 = fmin != 0
    # (2) mask: scan the ROWS mask rows this program owns; all keys unmasked -> mask-free loop (bit-identical for such rows)
    all_unmasked = True
    if HAS_MASK:
        soff = tl.arange(0, MSCAN)
        mmin = tl.min(tl.load(mb0 + soff * smk, mask=soff < SEQ_K, other=1).to(tl.int32), 0)
        for ms in range(MSCAN, SEQ_K, MSCAN):
            mmin = tl.minimum(mmin, tl.min(tl.load(mb0 + (ms + soff) * smk, mask=(ms + soff) < SEQ_K, other=1).to(tl.int32), 0))
        if ROWS >= 2:
            for ms in range(0, SEQ_K, MSCAN):
                mmin = tl.minimum(mmin, tl.min(tl.load(mb1 + (ms + soff) * smk, mask=(ms + soff) < SEQ_K, other=1).to(tl.int32), 0))
        if ROWS >= 4:
            for ms in range(0, SEQ_K, MSCAN):
                mmin = tl.minimum(mmin, tl.min(tl.load(mb2 + (ms + soff) * smk, mask=(ms + soff) < SEQ_K, other=1).to(tl.int32), 0))
                mmin = tl.minimum(mmin, tl.min(tl.load(mb3 + (ms + soff) * smk, mask=(ms + soff) < SEQ_K, other=1).to(tl.int32), 0))
        all_unmasked = mmin != 0

    if all_unmasked:
        if use16:
            acc0, m0, l0, acc1, m1, l1, acc2, m2, l2, acc3, m3, l3 = _attend(
                q0, q1, q2, q3, kb0, kb1, kb2, kb3, vb0, vb1, vb2, vb3, mb0, mb1, mb2, mb3,
                bias16_bh, b_off, k_off, v_off, m_off, offs_n, skk, svk, smk, sbq, SEQ_K, qk_scale2, bias_scale,
                HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS, APPLY_MASK=False, EXP_MODE=EXP_MODE, IP=IP)
        else:
            acc0, m0, l0, acc1, m1, l1, acc2, m2, l2, acc3, m3, l3 = _attend(
                q0, q1, q2, q3, kb0, kb1, kb2, kb3, vb0, vb1, vb2, vb3, mb0, mb1, mb2, mb3,
                bias32_bh, b_off, k_off, v_off, m_off, offs_n, skk, svk, smk, sbq, SEQ_K, qk_scale2, bias_scale,
                HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS, APPLY_MASK=False, EXP_MODE=EXP_MODE, IP=IP)
    else:
        if use16:
            acc0, m0, l0, acc1, m1, l1, acc2, m2, l2, acc3, m3, l3 = _attend(
                q0, q1, q2, q3, kb0, kb1, kb2, kb3, vb0, vb1, vb2, vb3, mb0, mb1, mb2, mb3,
                bias16_bh, b_off, k_off, v_off, m_off, offs_n, skk, svk, smk, sbq, SEQ_K, qk_scale2, bias_scale,
                HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS, APPLY_MASK=True, EXP_MODE=EXP_MODE, IP=IP)
        else:
            acc0, m0, l0, acc1, m1, l1, acc2, m2, l2, acc3, m3, l3 = _attend(
                q0, q1, q2, q3, kb0, kb1, kb2, kb3, vb0, vb1, vb2, vb3, mb0, mb1, mb2, mb3,
                bias32_bh, b_off, k_off, v_off, m_off, offs_n, skk, svk, smk, sbq, SEQ_K, qk_scale2, bias_scale,
                HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ROWS=ROWS, APPLY_MASK=True, EXP_MODE=EXP_MODE, IP=IP)

    # ---- epilogue ----------------------------------------------------------------------------------------------------------
    row_ok = (offs_m < SEQ_Q)[:, None]
    o_off = offs_mc[:, None] * soq + offs_d[None, :]
    o_bh = Out + b * sob + h * soh
    tl.store(o_bh + i0 * soi + o_off, (acc0 / l0[:, None]).to(Out.dtype.element_ty), mask=row_ok)
    if ROWS >= 2:
        tl.store(o_bh + i1 * soi + o_off, (acc1 / l1[:, None]).to(Out.dtype.element_ty), mask=row_ok & ((r0 + 1) < N_ROWS))
    if ROWS >= 4:
        tl.store(o_bh + i2 * soi + o_off, (acc2 / l2[:, None]).to(Out.dtype.element_ty), mask=row_ok & ((r0 + 2) < N_ROWS))
        tl.store(o_bh + i3 * soi + o_off, (acc3 / l3[:, None]).to(Out.dtype.element_ty), mask=row_ok & ((r0 + 3) < N_ROWS))


# --------------------------------------------------------------------------------------------------------------------------
# Config selection (deterministic; no runtime autotuning so results are bit-exact reproducible run-to-run)
# --------------------------------------------------------------------------------------------------------------------------
_SUPPORTED_D = (16, 32, 64, 128)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)   # fp32 inputs run the dots at IP precision

# default table: (D) -> list of (max_seq, cfg); first entry whose max_seq >= SK wins.  Tuned on H100 SXM;
# override with env PTX_TRIATTN_CONFIG="BLOCK_M,BLOCK_N,ROWS,num_warps,num_stages[,ORDER]" or the `config=` kwarg.
_CONFIG_TABLE: Dict[int, list] = {
    # D=32 (the stock head dim): 64x32 tiles, 2 rows/program, 4 warps, 3 stages, q-tile-fastest grid.  Chosen by tile sweeps on
    # H100 SXM.  BIAS16: use lossless 16-bit bias tiles when the fp32 bias is exactly representable (S_kv>=256).
    16:  [(1 << 30, dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=CONFIG["num_stages"], ORDER=0, MAXNREG=CONFIG["MAXNREG"]))],
    32:  [(1 << 30, dict(BLOCK_M=CONFIG["BLOCK_M"], BLOCK_N=CONFIG["BLOCK_N"], ROWS=CONFIG["ROWS"], num_warps=CONFIG["num_warps"],
                         num_stages=CONFIG["num_stages"], ORDER=CONFIG["ORDER"], MAXNREG=CONFIG["MAXNREG"]))],
    64:  [(1 << 30, dict(BLOCK_M=64, BLOCK_N=64, ROWS=2, num_warps=4, num_stages=2, ORDER=0))],      # = K2 (not re-tuned here)
    128: [(1 << 30, dict(BLOCK_M=64, BLOCK_N=32, ROWS=1, num_warps=4, num_stages=2, ORDER=0))],      # = K2 (not re-tuned here)
}
# fp32-input table (the OpenFold3 predict preset runs the pair stack in fp32/TF32; D=32, H=4). Chosen by an op-level sweep on H100 SXM.
_CONFIG_TABLE_F32: Dict[int, list] = {
    32:  [(1 << 30, dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=2, ORDER=0))],
    16:  [(1 << 30, dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=2, ORDER=0))],
    64:  [(1 << 30, dict(BLOCK_M=64, BLOCK_N=32, ROWS=1, num_warps=4, num_stages=2, ORDER=0))],
    128: [(1 << 30, dict(BLOCK_M=32, BLOCK_N=32, ROWS=1, num_warps=4, num_stages=2, ORDER=0))],
}
_IP_DEFAULT: str = os.environ.get("PF_FLASH_IP", os.environ.get("OF3T_FLASH_IP", "tf32"))      # 'tf32' | 'tf32x3' | 'ieee'  (fp32 inputs only)
assert _IP_DEFAULT in ("tf32", "tf32x3", "ieee"), _IP_DEFAULT
_BIAS16_MIN_SK = int(os.environ.get("K2B_BIAS16_MIN_SK", str(CONFIG["BIAS16_MIN_SK"])))   # below this the call is launch-bound
_BIAS16_ENABLED = os.environ.get("K2B_BIAS16", "1") != "0"




# ---- PF addition: per-device frozen table override -------------------------------------------------------------------------
from opt_core.kernels import safe_settings as _safe     # the core's ONE cc-keyed resolution + safe-settings mechanism (stdlib at import)


class BuildFailed(RuntimeError):
    """This lever cannot run in the process: its SAFE launch cell failed to build too (after the picked cell failed to build). A kit turns it into
    its hard error naming the one-flag escape (`--mode off`, or its explicit opt-out for this lever)."""
    kind = "build_failed"


_NET = _safe.SafeNet("fpf_triatt_k2b", refused=BuildFailed)     # this lever's safety net (safe_state / settings_word / the ONE line)
_CELL: Dict[str, Any] = {"no_cell": False, "cc": None}         # cc of the device; no_cell: the K2B cells table named by PF_TRIATTN_TABLE has NO entry for it (the package default cell serves, cells_note=default:no_row)


def _safe_cfg(D: int = 32, dtype=None) -> Dict[str, Any]:
    """The SAFE launch cell of this kernel for the current compute capability (opt_core.kernels.safe_settings.SAFE_ROWS "k2b": the tuned cell's tile
    at 1 row/program, 1 stage, no register cap) — every head dim and dtype."""
    row = _safe.safe_row("k2b", _CELL["cc"])
    return dict(row["settings"])


def safe_state() -> Dict[str, Any]:
    """``{"on", "reason", "where", "note"}``: whether this process serves the SAFE cell (a build failure of the picked cell) / the no-entry note."""
    return _NET.snapshot()


def settings_word() -> Optional[str]:
    """The LEVER line's ``settings=`` fact: ``"safe:build_failed:<exception class>"`` once the safe cell serves, else None."""
    return _NET.word()


def cells_note() -> Optional[str]:
    """The LEVER line's ``cells_note=`` fact: ``"default:no_row"`` when the cells table names no entry for this compute capability (the package default cell serves), else None."""
    return _NET.note()


def _where() -> str:
    return _safe.where_word(_CELL["cc"], _safe.triton_mm())


def _pf_resolve_cells(tab, cc, name):
    """k2b_cells/v2: compute capability FIRST ('by_cc' -> 'entries'), device-name substring only as a secondary match ('by_name'); v1 tables
    (top-level device-name substring keys) still accepted.  Returns (entry_key | None, entry_dict | None, how)."""
    cc = tuple(int(x) for x in cc)
    if isinstance(tab.get("by_cc"), dict):
        ent = tab["by_cc"].get(f"{cc[0]}.{cc[1]}")
        if ent is not None:
            key = ent if isinstance(ent, str) else ent.get("entry")
            return key, (tab.get("entries") or {}).get(key), "cc"
        for sub, key in (tab.get("by_name") or {}).items():
            if sub in name:
                return key, (tab.get("entries") or {}).get(key), "name(secondary)"
        return None, None, "no cc/name match"
    for key, val in tab.items():
        if isinstance(val, dict) and not key.startswith("_") and key in name:
            return key, val, "name(v1)"
    return None, None, "no name match (v1 table)"


def _pf_load_device_table():
    """PF_TRIATTN_TABLE=<json>: per-arch launch cells (v1.1.1: keyed on compute capability first; one printed line per process)."""
    path = os.environ.get("PF_TRIATTN_TABLE")
    if not path or not os.path.exists(path):
        return None
    import json as _json
    try:
        tab = _json.load(open(path))
        if not torch.cuda.is_available():
            return None
        cc = tuple(torch.cuda.get_device_capability(0)); dev = torch.cuda.get_device_name(0)
        _CELL["cc"] = (int(cc[0]), int(cc[1]))
        key, val, how = _pf_resolve_cells(tab, cc, dev)
        if val is None:                      # no entry names this compute capability: the package default cell (the tuned settings serve everyone), ONE info line;
            _CELL["no_cell"] = True            # a BUILD failure of it at launch drops to the SAFE cell (safe_settings)
            _NET.inform("no row for cc %d.%d (%s, %r, %s)" % (cc[0], cc[1], how, dev, os.path.basename(path)), _where(), tuned=_safe.table_ccs(tab.get("by_cc") or {}))
            return None
        for dt_name, target in (("bf16", _CONFIG_TABLE), ("fp32", _CONFIG_TABLE_F32)):
            for d_str, entries in (val.get(dt_name) or {}).items():
                target[int(d_str)] = [(int(ms), {k: v for k, v in dict(cfg).items() if not str(k).startswith("_")}) for ms, cfg in entries]
        c32 = (_CONFIG_TABLE.get(32) or [(None, {})])[0][1]
        print(f"[triatt_k2b] K2B cells: cc={cc} name={dev!r} -> entry {key!r} via {how} (MAXNREG={c32.get('MAXNREG', 'uncapped')}; status={str(val.get('status', 'n/a'))[:60]}) [{os.path.basename(path)}]", flush=True)
        return key
    except Exception as _e:  # never fail the model because of a table file
        print(f"[triatt_k2b] PF_TRIATTN_TABLE ignored: {_e!r}", flush=True)
    return None

_PF_TABLE_KEY = _pf_load_device_table()


def _env_config() -> Optional[Dict[str, int]]:
    """K2B_CONFIG="BLOCK_M,BLOCK_N,ROWS,num_warps,num_stages[,ORDER[,MAXNREG]]" (timing run override; 0/absent MAXNREG = uncapped)."""
    s = os.environ.get("K2B_CONFIG")
    if not s:
        return None
    vals = [int(x) for x in s.split(",")]
    bm, bn, rows, nw, ns = vals[:5]
    order = vals[5] if len(vals) > 5 else 0
    maxnreg = vals[6] if len(vals) > 6 and vals[6] > 0 else None
    return dict(BLOCK_M=bm, BLOCK_N=bn, ROWS=rows, num_warps=nw, num_stages=ns, ORDER=order, MAXNREG=maxnreg)


# Environment knobs are resolved ONCE at import (process constants): a persistent worker cannot flip numerics or tiling
# mid-run by mutating os.environ. Pass explicit kwargs to flash_triangle_attention() to override per call.
_ENV_CONFIG: Optional[Dict[str, int]] = _env_config()
# K2B: the exponent domain default depends on the input dtype (CONFIG["EXP_MODE_16BIT"] / CONFIG["EXP_MODE_FP32"]); the
# K2 env knobs are honoured only when set explicitly (K2B_EXACT=1 forces the natural domain for every dtype).
_EXACT_ENV: Optional[str] = os.environ.get("K2B_EXACT")
_LIBDEVICE_DEFAULT: bool = os.environ.get("K2B_LIBDEVICE", "0") == "1"           # True (with exact) -> EXP_MODE 2 (libdevice expf)


def _default_exact(dtype: torch.dtype) -> bool:
    if _EXACT_ENV is not None:
        return _EXACT_ENV != "0"
    mode = CONFIG["EXP_MODE_FP32"] if dtype == torch.float32 else CONFIG["EXP_MODE_16BIT"]
    return mode == 0


def _exp_mode(exact_exp: bool, use_libdevice: bool) -> int:
    if exact_exp and use_libdevice:
        return 2
    return 0 if exact_exp else 1


def pick_config(D: int, SQ: int, SK: int, H: int, dtype: torch.dtype = torch.bfloat16) -> Dict[str, int]:
    if _ENV_CONFIG is not None:
        return dict(_ENV_CONFIG)
    if _NET.on:                                    # the SAFE cell serves this process (a build failure of the picked cell — safe_settings)
        return _safe_cfg(D, dtype)
    if dtype == torch.float32:
        for max_seq, c in _CONFIG_TABLE_F32.get(D, _CONFIG_TABLE[D]):
            if SK <= max_seq:
                return dict(c)
    for max_seq, c in _CONFIG_TABLE[D]:
        if SK <= max_seq:
            return dict(c)
    return dict(_CONFIG_TABLE[D][-1][1])


def set_default_config(D: int, cfg: Dict[str, int], max_seq: int = 1 << 30) -> None:
    """Replace the config table for head dim D with a single entry (used by the timing run after a sweep)."""
    _CONFIG_TABLE[D] = [(max_seq, dict(cfg))]


# --------------------------------------------------------------------------------------------------------------------------
# Python wrapper
# --------------------------------------------------------------------------------------------------------------------------
def _ensure_dims(t: torch.Tensor, n: int) -> torch.Tensor:
    while t.dim() < n:
        t = t.unsqueeze(0)
    return t


def _maybe_to(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t if t.dtype == dtype else t.to(dtype)


_CC_CACHE: Dict[Any, Tuple[int, int]] = {}


def _device_cc(device) -> Tuple[int, int]:
    key = (device.type, device.index)
    cc = _CC_CACHE.get(key)
    if cc is None:
        cc = tuple(torch.cuda.get_device_capability(device))
        _CC_CACHE[key] = cc
    return cc


# ---- low-overhead launch path -------------------------------------------------------------------------------------------
# triton's JITFunction launcher re-derives the argument specialisation and cache key in Python on every call (tens of us).
# The model calls this op ~1200x per trunk pass, and at small N the trunk is launch-bound, so after the first (JIT) launch of
# a given specialisation we keep the CompiledKernel and launch it directly.  The specialisation key below mirrors triton's
# (pointer 16-B alignment; int ==1 / %16==0), the first direct launch of every key is checked bit-exact against the JIT
# launcher, and any exception or mismatch permanently disables the fast path (-> plain JIT launches; same numerics).
_FAST_LAUNCH_ENABLED: bool = os.environ.get("K2B_FASTLAUNCH", os.environ.get("PTX_TRIATTN_FASTLAUNCH", "1")) != "0"
_FAST_RUNNERS: Dict[Any, Any] = {}
_KERNEL_INFO: list = []     # resource info (regs/spills/smem) of every compiled specialisation
_FAST_STATE: Dict[str, Any] = {"disabled": False, "verified_keys": 0, "fast_launches": 0, "jit_launches": 0, "why_disabled": None}


def fast_launch_stats() -> Dict[str, Any]:
    return dict(_FAST_STATE, cached_keys=len(_FAST_RUNNERS), kernel_info=list(_KERNEL_INFO))


def _spec_key(ptr_args, int_args):
    return (tuple((p.data_ptr() % 16 == 0) for p in ptr_args),
            tuple(((a % 16 == 0), (a == 1)) for a in int_args))


_ONES: Dict[Any, torch.Tensor] = {}


def _launch_generic(jit_fn, tag, grid, ptr_args, int_args, float_args, const_kwargs, num_warps, num_stages, verify_out_idx, maxnreg=None):
    """Launch `jit_fn`; after the first (JIT) launch of a specialisation keep the CompiledKernel and launch it directly.
    triton's JITFunction launcher re-derives the specialisation key in Python on every call (tens of us); the model calls this
    op ~1200x per trunk pass and small-N trunks are launch-bound.  The key mirrors triton's own specialisation (pointer 16-B
    alignment; int ==1 / %16==0); the first direct launch of every key is checked bit-exact against the JIT launcher and any
    exception or mismatch permanently disables the fast path (-> plain JIT launches; identical numerics)."""
    grid = tuple(grid) + (1,) * (3 - len(grid))       # CompiledKernel's direct runner needs a 3-tuple
    use_fast = (_FAST_LAUNCH_ENABLED and not _FAST_STATE["disabled"] and ptr_args[0].is_cuda
                and isinstance(jit_fn, triton.runtime.JITFunction))
    key = None
    if use_fast:
        key = (tag, ptr_args[0].device.index, tuple(p.dtype for p in ptr_args), tuple(sorted(const_kwargs.items())),
               num_warps, num_stages, maxnreg, _spec_key(ptr_args, int_args))
        ck = _FAST_RUNNERS.get(key)
        if ck is not None:
            try:
                ck[grid](*ptr_args, *int_args, *float_args, *const_kwargs.values())
                _FAST_STATE["fast_launches"] += 1
                return
            except Exception as e:  # API drift -> permanent fallback
                from opt_core.oom import is_oom                     # an out-of-memory is the caller's to see, never a reroute (the core's one classifier)
                if is_oom(e):
                    raise
                _FAST_STATE["disabled"] = True; _FAST_STATE["why_disabled"] = f"launch: {e!r}"[:300]
    launch_opts = dict(num_warps=num_warps, num_stages=num_stages)
    if maxnreg is not None:
        launch_opts["maxnreg"] = int(maxnreg)          # ptxas --maxnreg (triton >= 3.0 launch option); allocation only
    ck = jit_fn[grid](*ptr_args, *int_args, *float_args, **const_kwargs, **launch_opts)
    _FAST_STATE["jit_launches"] += 1
    try:
        info = {"kernel": tag, "n_regs": getattr(ck, "n_regs", None), "n_spills": getattr(ck, "n_spills", None),
                "shared": getattr(getattr(ck, "metadata", None), "shared", None),
                "const": {k: (int(v) if isinstance(v, bool) else v) for k, v in const_kwargs.items()}, "num_warps": num_warps, "num_stages": num_stages,
                "maxnreg": maxnreg}
        _FAST_STATE["last_kernel_info"] = info; _KERNEL_INFO.append(info)
    except Exception:
        pass
    if use_fast and not _FAST_STATE["disabled"] and ck is not None and key is not None:
        try:
            out = ptr_args[verify_out_idx]
            out2 = torch.empty_like(out)
            pa = list(ptr_args); pa[verify_out_idx] = out2
            ck[grid](*pa, *int_args, *float_args, *const_kwargs.values())
            if torch.equal(out2, out):
                _FAST_RUNNERS[key] = ck; _FAST_STATE["verified_keys"] += 1
            else:
                _FAST_STATE["disabled"] = True; _FAST_STATE["why_disabled"] = f"verification mismatch ({tag})"
        except Exception as e:
            from opt_core.oom import is_oom
            if is_oom(e):
                raise
            _FAST_STATE["disabled"] = True; _FAST_STATE["why_disabled"] = f"verify: {e!r}"[:300]


def _launch(grid, ptr_args, int_args, scale, const_kwargs, num_warps, num_stages, maxnreg=None):
    """ptr_args: (q,k,v,bias32,bias16,flags,mask_u8,out); int_args: strides/sizes in kernel-signature order."""
    _launch_generic(_flash_triattn_fwd, "attn", grid, ptr_args, int_args, (scale,), const_kwargs, num_warps, num_stages, len(ptr_args) - 1,
                    maxnreg=maxnreg)


def _launch_prep(grid, ptr_args, int_args, const_kwargs):
    """ptr_args: (bias, out32, out16, flags)."""
    _launch_generic(_bias_prep, "prep", grid, ptr_args, int_args, (), const_kwargs, 4, 2, 1)


def flash_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> Tuple[bool, str]:
    """Cheap (no sync) support check. Returns (ok, reason)."""
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda):
        return False, "not_cuda"
    dt = q.dtype
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
    if dt not in _SUPPORTED_DTYPES:
        return False, f"dtype_{dt}".replace("torch.", "")
    D = q.shape[-1]
    if D not in _SUPPORTED_D:
        return False, f"head_dim_{D}"
    if q.dim() > 5 or k.dim() != q.dim() or v.dim() != q.dim():
        return False, "rank"
    if k.shape[-2] < 1 or q.shape[-2] < 1:
        return False, "empty"
    if _device_cc(q.device)[0] < 8:
        return False, "cc<8"
    return True, "ok"


def reference_triangle_attention(q, k, v, bias, mask=None, scale=None, dtype=torch.float32, row_chunk: int = 16):
    """Materialised-logits reference (rows chunked to bound memory). dtype=float64 gives the gold reference."""
    q = _ensure_dims(q, 5); k = _ensure_dims(k, 5); v = _ensure_dims(v, 5); bias = _ensure_dims(bias, 5)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    B, N, H, SQ, D = q.shape
    out = torch.empty((B, N, H, SQ, D), dtype=dtype, device=q.device)
    bias_d = bias.to(dtype)
    if mask is not None:
        mask = _ensure_dims(mask.to(torch.bool), 5)
        mterm = (-_MASK_NEG) * (~mask).to(dtype)          # [B, N, 1, 1, SK]
    for i0 in range(0, N, row_chunk):
        i1 = min(N, i0 + row_chunk)
        qq = q[:, i0:i1].to(dtype) * scale
        kk = k[:, i0:i1].to(dtype)
        vv = v[:, i0:i1].to(dtype)
        a = torch.matmul(qq, kk.transpose(-1, -2))       # [B, r, H, SQ, SK]
        if mask is not None:
            a = a + mterm[:, i0:i1]
        a = a + bias_d
        a = torch.softmax(a, dim=-1)
        out[:, i0:i1] = torch.matmul(a, vv)
    return out


def flash_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
    *,
    exact_exp: Optional[bool] = None,
    use_libdevice: Optional[bool] = None,
    bias16: Optional[bool] = None,
    config: Optional[Dict[str, int]] = None,
    out_dtype: Optional[torch.dtype] = None,
    input_precision: Optional[str] = None,
) -> torch.Tensor:
    """Same signature/semantics as cuequivariance_torch.primitives.triangle.triangle_attention (return_aux=False).

    Returns a tensor of shape (B, N, H, SQ, D) (inputs of rank < 5 are treated as having leading singleton dims and the
    5-D result is returned, exactly like cuEq), dtype = q.dtype (after autocast), contiguous.
    """
    if return_aux:
        raise NotImplementedError("flash_triangle_attention: return_aux=True is not supported (inference forward only)")
    exact_req = exact_exp                          # K2B: resolved after the dtype is known (base-2 default for 16-bit inputs)
    if use_libdevice is None:
        use_libdevice = _LIBDEVICE_DEFAULT
    q = _ensure_dims(q, 5); k = _ensure_dims(k, 5); v = _ensure_dims(v, 5); bias = _ensure_dims(bias, 5)
    if mask is not None:
        mask = _ensure_dims(mask, 5)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

    B, N, H, SQ, D = q.shape
    SK = k.shape[3]
    if k.shape != (B, N, H, SK, D):
        raise ValueError(f"triangle_attention: input k must have shape (B, N, H, S_kv, hidden_dim) but got: {tuple(k.shape)}")
    if v.shape != (B, N, H, SK, D):
        raise ValueError(f"triangle_attention: input v must have shape (B, N, H, S_kv, hidden_dim) but got: {tuple(v.shape)}")
    if bias.shape != (B, 1, H, SQ, SK):
        raise ValueError(f"triangle_attention: input bias must have shape (B, 1, H, S_qo, S_kv) but got: {tuple(bias.shape)}")
    if mask is not None and mask.shape != (B, N, 1, 1, SK):
        raise ValueError(f"triangle_attention: input mask must have shape (B, N, 1, 1, S_kv) but got: {tuple(mask.shape)}")

    # autocast handling identical to cuEq: cast q,k,v to the autocast dtype
    if torch.is_autocast_enabled():
        adt = torch.get_autocast_dtype("cuda")
        q = _maybe_to(q, adt); k = _maybe_to(k, adt); v = _maybe_to(v, adt)
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)
    if q.dtype not in _SUPPORTED_DTYPES or D not in _SUPPORTED_D:
        raise ValueError(f"flash_triangle_attention: unsupported dtype/head_dim {q.dtype}/{D}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    exact_exp = _default_exact(q.dtype) if exact_req is None else bool(exact_req)
    if exact_exp and use_libdevice and not _HAS_LIBDEVICE:
        raise RuntimeError("libdevice expf requested but triton libdevice is unavailable")

    odt = q.dtype if out_dtype is None else out_dtype
    out = torch.empty((B, N, H, SQ, D), dtype=odt, device=q.device)
    if out.numel() == 0:
        return out
    if SK == 0:
        return out.fill_(float("nan"))

    # last-dim contiguity is required for vectorised loads; everything else is handled by strides (no copies)
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    if bias.dtype != torch.float32:
        bias = bias.float()
    if mask is not None:
        mask_u8 = mask.view(torch.uint8)  # same storage, no copy
        smb, smi, smk = mask_u8.stride(0), mask_u8.stride(1), mask_u8.stride(4)
        has_mask = True
    else:
        mask_u8 = out                      # dummy pointer, never dereferenced
        smb = smi = smk = 0
        has_mask = False

    cfg = dict(config) if config is not None else pick_config(D, SQ, SK, H, q.dtype)     # tile grid / constexprs / maxnreg are derived per launch in _attn (the safe cell may replace cfg)
    want16 = bool(cfg.get("BIAS16", _BIAS16_ENABLED and SK >= _BIAS16_MIN_SK)) if bias16 is None else bool(bias16)
    if q.dtype == torch.float32:
        want16 = False                     # 16-bit bias tiles are a bandwidth saving for 16-bit inputs only

    # ---- bias preparation: ONE pass over the (permuted, fp32) bias -> fp32 copy with unit key stride / 16-elt row pitch /
    # 16-B alignment (vectorised, L2-friendly tile loads; cuEq makes the same contiguous copy), plus -- when want16 -- a
    # 16-bit copy and per-program flags qualifying that the fp32 -> 16-bit cast was lossless (the model computes the triangle
    # bias under bf16 autocast and upcasts it with .float(), so this holds in the model; it is checked on device per call
    # and the kernel falls back to the fp32 tiles otherwise -> results are bit-identical either way).
    SKp = -(-SK // 16) * 16
    PB, PK = 32, 128
    nqb = triton.cdiv(SQ, PB)
    n_flags = B * H * nqb
    bias32 = torch.empty((B, 1, H, SQ, SKp), dtype=torch.float32, device=q.device)
    if want16:
        bias16_t = torch.empty((B, 1, H, SQ, SKp), dtype=q.dtype, device=q.device)
        flags = torch.empty((max(n_flags, 1),), dtype=torch.int32, device=q.device)
    else:
        bias16_t = bias32                 # dummies, never dereferenced (BIAS16=0)
        flags = torch.ones((1,), dtype=torch.int32, device=q.device) if _ONES.get(q.device) is None else _ONES[q.device]
        _ONES[q.device] = flags
        n_flags = 1
    _launch_prep((nqb, B * H), (bias, bias32, bias16_t, flags),
                 (bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, SQ, SK, SKp, nqb),
                 dict(PB=PB, PK=PK, MAKE16=want16))

    int_args = (q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                bias32.stride(0), bias32.stride(2), bias32.stride(3),
                smb, smi, smk,
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                N, SQ, SK, H, n_flags)

    def _attn(c: Dict[str, Any]) -> None:       # everything the launch cell decides: tile grid, the int32 offset bound, the constexpr set, the register cap
        bm, bn, rows, order = c["BLOCK_M"], c["BLOCK_N"], c["ROWS"], int(c.get("ORDER", 0))
        assert rows in (1, 2, 4), rows
        maxnreg = None if q.dtype == torch.float32 else c.get("MAXNREG", None)     # fp32 (TF32) path keeps K2's launch options (no register cap)
        n_q, n_r = triton.cdiv(SQ, bm), triton.cdiv(N, rows)
        grid = (n_q, n_r, B * H) if order == 0 else (n_r, n_q, B * H)
        if int(bn) * max(int(k.stride(3)), int(v.stride(3))) + D >= 2 ** 31 or SQ * SKp >= 2 ** 31:
            raise ValueError("flash_triangle_attention: tensor too large for int32 tile offsets")
        const_kwargs = dict(HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn, ROWS=rows, HAS_MASK=has_mask, EXP_MODE=_exp_mode(bool(exact_exp), bool(use_libdevice)),
                            BIAS16=1 if want16 else 0, ORDER=order, MSCAN=1024, NFLAG=max(1024, triton.next_power_of_2(n_flags)),
                            IP=(input_precision or _IP_DEFAULT))
        _launch(grid, (q, k, v, bias32, bias16_t, flags, mask_u8, out), int_args, float(scale), const_kwargs, c["num_warps"], c["num_stages"],
                maxnreg=maxnreg)

    _NET.run(_attn, cfg, lambda: _safe_cfg(D, q.dtype), where=_where(), explicit=config is not None)
    return out


# FPF timing run entry point (cuequivariance triangle_attention call convention)
attn = flash_triangle_attention
