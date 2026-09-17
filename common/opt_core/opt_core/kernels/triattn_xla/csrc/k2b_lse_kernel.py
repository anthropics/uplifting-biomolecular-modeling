"""k2b_lse_kernel.py -- the K2B flash triangle-attention forward (kernels/fpf_triatt_k2b/triatt_k2b.py::_flash_triattn_fwd, carried
unmodified there) with ONE addition for the differentiable row: the kernel also stores each query row's log-sum-exp (fp32, log2 units) --
the statistic a flash-attention backward reads instead of re-running the forward.  The body below is _flash_triattn_fwd verbatim (its
_attend / _row_step device functions are imported from the carried module, not copied) plus the parameter `Lse` and the marked epilogue;
the attention output it stores is the same arithmetic in the same order, so `Out` is bit-identical to the forward-only kernel's (asserted
on the recorded vectors).  Compiled ahead of time by csrc/build.py (step k2bl) into bin/k2b/sm_*/k2bl_fwd_*.cubin;
never imported at prediction time (it needs triton; the JAX images do not have it).  Licence / provenance: as the carried kernel (NOTICE)."""
import os
import sys

import triton
import triton.language as tl

_K2B_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "fpf_triatt_k2b")
if _K2B_DIR not in sys.path:
    sys.path.insert(0, _K2B_DIR)
import triatt_k2b as _K          # noqa: E402  (the carried kernel module, by its own loader convention)

_attend = _K._attend


@triton.jit
def _flash_triattn_fwd_lse(
    Q, K, V, Bias32, Bias16, Flags, Mask, Out, Lse,     # Lse: fp32 [B, N_ROWS, H, SEQ_Q] dense -- log2-units logsumexp of every query row
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
    # ---- added epilogue: the softmax statistics the backward needs, as ONE fp32 number per query row --------------------------
    # lse2 = log2(sum_k 2^(s2_k)) in the base-2 logit domain (EXP_MODE 1: m, l are already base-2), i.e. (m + ln l) * log2e in the
    # natural domain (EXP_MODE 0 / 2).  Dense [B, N_ROWS, H, SEQ_Q]: offset ((b*N_ROWS + i)*H + h)*SEQ_Q + q -- no new stride arguments.
    rowv = offs_m < SEQ_Q
    l_bh = Lse + (b * N_ROWS * H + h) * SEQ_Q
    if EXP_MODE == 1:
        e0 = m0 + tl.math.log2(l0); e1 = m1 + tl.math.log2(l1); e2 = m2 + tl.math.log2(l2); e3 = m3 + tl.math.log2(l3)
    else:
        e0 = (m0 + tl.log(l0)) * 1.4426950408889634; e1 = (m1 + tl.log(l1)) * 1.4426950408889634
        e2 = (m2 + tl.log(l2)) * 1.4426950408889634; e3 = (m3 + tl.log(l3)) * 1.4426950408889634
    tl.store(l_bh + i0 * H * SEQ_Q + offs_mc, e0, mask=rowv)
    if ROWS >= 2:
        tl.store(l_bh + i1 * H * SEQ_Q + offs_mc, e1, mask=rowv & ((r0 + 1) < N_ROWS))
    if ROWS >= 4:
        tl.store(l_bh + i2 * H * SEQ_Q + offs_mc, e2, mask=rowv & ((r0 + 2) < N_ROWS))
        tl.store(l_bh + i3 * H * SEQ_Q + offs_mc, e3, mask=rowv & ((r0 + 3) < N_ROWS))
