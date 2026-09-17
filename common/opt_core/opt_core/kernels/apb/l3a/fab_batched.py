# fab_batched.py -- lever L3a: SHARED-BIAS BATCHED pair-bias flash attention, stride-0 sample axis.
#
#   flash_bias_attn_batched(q, k, v, bias, ...)   M samples (or sequence-alignment rows) in ONE launch; ONE bias tensor without a sample axis
#   flash_bias_attn_v2(q, k, v, bias, ...)        the M = 1 entry with the Stage C options on (same signature as production flash_bias_attn)
#   pair_bias_attention(q, k, v, bias, ...)       dispatcher behind the ONE flag KOPT_ATTN_L3A = 0 | a | c | auto   (0 = production kernel looped per sample)
#
# Numerics: the per-program arithmetic is a copy, operation for operation, of production `opt_core.kernels.dtk_kernels._fab_fwd_kernel`
# (same BM/BN/num_warps/num_stages defaults, same order of scale / bias*log2e / key mask / where / max / exp2 / sum / dot, m_i initialised to -1e30,
# fp32 logits + running max + running sum + accumulator, P cast to the value dtype for P@V for 16-bit inputs, fused sigmoid-gate epilogue).  The only
# things that differ are ADDRESSES: the launch grid gains a sample axis (fastest) and the bias pointer ignores it.  So out[m] is intended to be
# BITWISE equal to production flash_bias_attn(q[m], k[m], v[m], bias[..., :N], key_mask[m], gate[m]); the proof is torch.equal on the GPU
# (gpu/gpu_rows_l3a.py); the CPU interpreter tests are a logic check.  The tile configuration depends on the dtype only -- never on M or G.
#
# Stage C options (each a constexpr).  MEASURED CONTRACT (offline sm_90 / sm_80 compiles, triton 3.3.1 and 3.7.1; GPU bit check in gpu/gpu_rows_l3a.py):
#   variant a (all off)  : op-for-op production -> bitwise at every N.
#   variant c (all on)   : bitwise expected at N % 64 == 0; UNKNOWN at N % 16 == 0 with N % 64 != 0 (peeled tail block); not bitwise at N % 16 != 0, where production ITSELF runs a different numeric program (its bias loads cannot vectorise, and the
#                          compiler then does the fp32 row-sum in the blocked layout instead of the MMA layout); PEEL turns the full tiles into the aligned-N
#                          program, which is the speed-up AND a different fp32 row-sum association (same error class, ~1 output ulp on a few elements).
#   ALIGN_B / EVICT alone, and the padded / skewed layout alone, never change bits.
#   ALIGN_B   (i)   the bias row / head / group strides are declared multiples of 16 elements (only set by the wrapper when true and the base pointer is
#                   16-byte aligned) -> with (ii) the compiler can prove 16-byte vector loads on bias rows at ragged N (row stride Np, see bias_layout.py)
#   PEEL      (ii)  key loop over FULL tiles without the n < N bounds masks / tl.where; only the last partial tile is masked
#   ROWSPLIT  (ii)  full query tiles skip the m < N row mask on q / bias / gate loads and the out store
#   (iii)           power-of-two stride skew lives in bias_layout.PairBiasLayout; the kernel just honours the row stride it is given
#   EVICT     (iv)  eviction_policy = evict_last on bias loads, evict_first on the q / gate loads and the out store
# No atomics, no split-K, fixed reduction order (run-to-run deterministic), CUDA-graph capturable (no host syncs, no allocation when out= is given),
# all sample / group / head / row base offsets int64.  Pad columns of a padded bias are NEVER read (keys are bounded by n < N, N = q.shape[-2]).
#
# Import cost: standard library only unless triton is importable (guarded); torch is imported inside the functions.
import math
import os

try:  # guarded: the package must import without torch / triton
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAVE_TRITON = False

_LOG2E = 1.4426950408889634

FLAG_ENV = "KOPT_ATTN_L3A"          # the ONE flag: 0 (off: production per-sample loop) | a (batched, Stage C off) | c (batched, Stage C on)
#                                     | auto (c where c is bitwise to production -- N % 64 == 0 and an aligned bias -- else a: bitwise everywhere)
AUTO_C_MODULUS = 64                 # auto picks variant c only when N is a multiple of this.  64 = BM = BN: no partial key / query tile exists, and the executed code is
#                                     production's aligned-N loop minus all-true selects (offline PTX evidence).  At N % 16 == 0 with N % 64 != 0 the peeled TAIL tile is a
#                                     separate straight-line block whose op mix equals production's ragged-N program, so bit-equality there is NOT established: lower this to
#                                     16 only after the GPU bit check passes at N in {400, 1008, 2000} (gpu/gpu_rows_l3a.py)


class KoptAttnUnavailable(RuntimeError):
    """triton (or torch) is not importable in this interpreter."""


class KoptAttnRefused(ValueError):
    """an argument combination this kernel refuses by name (never silently mis-computes)."""


def l3a_mode(default="0"):
    """value of the ONE flag KOPT_ATTN_L3A: '0' | 'a' | 'c' | 'auto' (also accepts 1 -> 'a', 2 / 'on' -> 'c')."""
    v = os.environ.get(FLAG_ENV, default).strip().lower()
    return {"": "0", "0": "0", "off": "0", "1": "a", "a": "a", "2": "c", "c": "c", "on": "c", "auto": "auto"}.get(v, "0")


def auto_variant(N, bias):
    """'c' where Stage C keeps production's bits, else 'a'.  Production runs its vector-load program (bias tiles through cp.async, row-sum in the MMA layout)
    exactly when its bias loads can vectorise: N % 16 == 0 and 16-element-aligned bias strides / base pointer.  There the peeled kernel is that same
    program minus all-true masks.  At ragged N production runs a different program (scalar bias loads, row-sum in the blocked layout) whose bits Stage C
    does not keep, so auto stays on 'a' (op-for-op production) there."""
    if int(N) % AUTO_C_MODULUS != 0:
        return "a"
    if bias is not None:
        st = bias.stride()
        if st[-1] != 1 or any(int(x) % 16 for x in st[:-1]) or bias.data_ptr() % 16:
            return "a"
    return "c"


def _fp32_prec():
    """same rule as production dtk_kernels._fp32_prec (env DTK_FP32_PREC; 'ieee' default)."""
    p = os.environ.get("DTK_FP32_PREC", "ieee").lower()
    return p if p in ("ieee", "tf32", "tf32x3") else "ieee"


def _pow2(x, lo=16):
    p = lo
    while p < x:
        p *= 2
    return p


def tile_config(N, D, lowp):
    """(BM, BN, num_warps, num_stages) keyed on (N bucket, D, dtype) and NEVER on M or G.  Today every bucket returns production's defaults
    (flash_bias_attn: 64/64/4/2 for 16-bit inputs, 32/32/4/2 for fp32) -- required for bitwise equality with the production per-sample call."""
    return (64, 64, 4, 2) if lowp else (32, 32, 4, 2)


STAGE_C_OFF = dict(peel=False, rowsplit=False, align=False, evict=False)
STAGE_C_ON = dict(peel=True, rowsplit=True, align=True, evict=True)


def _stage_c_flags(stage_c):
    """None / False -> all off; True / 'c' / 'on' -> all on; a dict or a comma string ('peel,align') selects options one by one."""
    if stage_c is None or stage_c is False:
        return dict(STAGE_C_OFF)
    if stage_c is True:
        return dict(STAGE_C_ON)
    if isinstance(stage_c, dict):
        f = dict(STAGE_C_OFF)
        for k_, v_ in stage_c.items():
            if k_ not in f:
                raise KoptAttnRefused("flash_bias_attn_batched: unknown Stage C option %r (known: %s)" % (k_, sorted(f)))
            f[k_] = bool(v_)
        return f
    if isinstance(stage_c, str):
        s = stage_c.strip().lower()
        if s in ("", "0", "off", "a", "none"):
            return dict(STAGE_C_OFF)
        if s in ("1", "on", "c", "all"):
            return dict(STAGE_C_ON)
        f = dict(STAGE_C_OFF)
        for tok in s.split(","):
            tok = tok.strip()
            if tok not in f:
                raise KoptAttnRefused("flash_bias_attn_batched: unknown Stage C option %r (known: %s)" % (tok, sorted(f)))
            f[tok] = True
        return f
    raise KoptAttnRefused("flash_bias_attn_batched: stage_c must be None / bool / str / dict")


if _HAVE_TRITON:

    @triton.jit
    def _fab_key_tile(q, K_b, V_b, B_b, KM_b, start_n, offs_m, offs_n, offs_d, m_ok, d_ok,
                      s_kn, s_kd, s_vn, s_vd, s_bq, s_bk,
                      N, qk_scale, mask_neg, m_i, l_i, acc,
                      BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                      HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, IN_PREC: tl.constexpr, PV_LOWP: tl.constexpr,
                      MASK_N: tl.constexpr, MASK_M: tl.constexpr, EV_BIAS: tl.constexpr):
        # ONE key tile of the online softmax.  MASK_N = MASK_M = True is production's loop body verbatim; the False variants drop bounds masks that are
        # all-true on the tile (values on valid lanes identical).
        cur_n = start_n + offs_n
        cur_n64 = cur_n.to(tl.int64)
        k_ptrs = K_b + cur_n64[:, None] * s_kn + offs_d[None, :] * s_kd
        if MASK_N:
            n_ok = cur_n < N
            k = tl.load(k_ptrs, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs, mask=tl.broadcast_to(d_ok[None, :], (BN, BD)), other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision=IN_PREC)          # fp32 [BM, BN]
        qk = qk * qk_scale                                             # scale * log2(e)
        if HAS_BIAS:
            b_ptrs = B_b + offs_m[:, None] * s_bq + cur_n64[None, :] * s_bk
            if MASK_N:
                b = tl.load(b_ptrs, mask=m_ok[:, None] & n_ok[None, :], other=0.0, eviction_policy=EV_BIAS).to(tl.float32)
            else:
                if MASK_M:
                    b = tl.load(b_ptrs, mask=tl.broadcast_to(m_ok[:, None], (BM, BN)), other=0.0, eviction_policy=EV_BIAS).to(tl.float32)
                else:
                    b = tl.load(b_ptrs, eviction_policy=EV_BIAS).to(tl.float32)
            qk = qk + b * 1.4426950408889634
        if HAS_MASK:
            if MASK_N:
                km = tl.load(KM_b + cur_n, mask=n_ok, other=0).to(tl.float32)
            else:
                km = tl.load(KM_b + cur_n).to(tl.float32)
            qk = qk + (1.0 - km)[None, :] * (mask_neg * 1.4426950408889634)
        if MASK_N:
            qk = tl.where(n_ok[None, :], qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = V_b + cur_n64[:, None] * s_vn + offs_d[None, :] * s_vd
        if MASK_N:
            v = tl.load(v_ptrs, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        else:
            v = tl.load(v_ptrs, mask=tl.broadcast_to(d_ok[None, :], (BN, BD)), other=0.0)
        if PV_LOWP:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=IN_PREC)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision=IN_PREC)
        m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _fab_key_loop(q, K_b, V_b, B_b, KM_b, offs_m, offs_n, offs_d, m_ok, d_ok,
                      s_kn, s_kd, s_vn, s_vd, s_bq, s_bk,
                      N, qk_scale, mask_neg,
                      BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                      HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, IN_PREC: tl.constexpr, PV_LOWP: tl.constexpr,
                      PEEL: tl.constexpr, MASK_M: tl.constexpr, EV_BIAS: tl.constexpr):
        m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, BD], dtype=tl.float32)
        if PEEL:
            n_full = (N // BN) * BN                                    # keys covered by full tiles; the tail tile (if any) starts here
            for start_n in range(0, n_full, BN):
                m_i, l_i, acc = _fab_key_tile(q, K_b, V_b, B_b, KM_b, start_n, offs_m, offs_n, offs_d, m_ok, d_ok,
                                              s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg, m_i, l_i, acc,
                                              BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, False, MASK_M, EV_BIAS)
            if n_full < N:
                m_i, l_i, acc = _fab_key_tile(q, K_b, V_b, B_b, KM_b, n_full, offs_m, offs_n, offs_d, m_ok, d_ok,
                                              s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg, m_i, l_i, acc,
                                              BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, True, True, EV_BIAS)
        else:
            for start_n in range(0, N, BN):                            # production's loop: bounds masks on every key tile
                m_i, l_i, acc = _fab_key_tile(q, K_b, V_b, B_b, KM_b, start_n, offs_m, offs_n, offs_d, m_ok, d_ok,
                                              s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg, m_i, l_i, acc,
                                              BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, True, True, EV_BIAS)
        acc = acc / l_i[:, None]
        return acc

    @triton.jit(do_not_specialize=["M"])          # ONE binary for every sample count M (no ==1 / %16 specialisation of M)
    def _fab_batched_kernel(Q, K, V, B, KM, G, O, GRP,
                            s_qm, s_qh, s_qn, s_qd,
                            s_km, s_kh, s_kn, s_kd,
                            s_vm, s_vh, s_vn, s_vd,
                            s_bg, s_bh, s_bq, s_bk,
                            s_mm,
                            s_gm, s_gn, s_gc,
                            s_om, s_on, s_oc,
                            M, N, qk_scale, mask_neg,
                            H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr,
                            HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr, HAS_GROUP: tl.constexpr,
                            IN_PREC: tl.constexpr, PV_LOWP: tl.constexpr,
                            PEEL: tl.constexpr, ROWSPLIT: tl.constexpr, ALIGN_B: tl.constexpr,
                            EV_BIAS: tl.constexpr, EV_STREAM: tl.constexpr):
        # 1-D grid of M * cdiv(N, BM) * H programs; the SAMPLE index is fastest, the query tile next, the head slowest: the M programs that read the
        # same [BM, N] bias slab are adjacent in launch order and share it through L2.  Every index below is int64.
        pid = tl.program_id(0).to(tl.int64)
        n_qt = (N + (BM - 1)) // BM
        m = pid % M
        t = pid // M
        pid_m = t % n_qt
        h = t // n_qt
        if ALIGN_B:
            s_bq = tl.multiple_of(s_bq, 16)
            s_bh = tl.multiple_of(s_bh, 16)
            s_bg = tl.multiple_of(s_bg, 16)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, BD)
        m_ok = offs_m < N
        d_ok = offs_d < D
        Q_b = Q + m * s_qm + h * s_qh
        K_b = K + m * s_km + h * s_kh
        V_b = V + m * s_vm + h * s_vh
        if HAS_GROUP:
            grp = tl.load(GRP + m).to(tl.int64)
            B_b = B + grp * s_bg + h * s_bh
        else:
            B_b = B + h * s_bh
        KM_b = KM + m * s_mm
        q_ptrs = Q_b + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd
        g_ptrs = G + m * s_gm + offs_m[:, None] * s_gn + (h * D + offs_d)[None, :] * s_gc
        o_ptrs = O + m * s_om + offs_m[:, None] * s_on + (h * D + offs_d)[None, :] * s_oc
        if ROWSPLIT:
            if (pid_m + 1) * BM <= N:                                   # FULL query tile: no row mask anywhere
                dmask = tl.broadcast_to(d_ok[None, :], (BM, BD))
                q = tl.load(q_ptrs, mask=dmask, other=0.0, eviction_policy=EV_STREAM)
                acc = _fab_key_loop(q, K_b, V_b, B_b, KM_b, offs_m, offs_n, offs_d, m_ok, d_ok,
                                    s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg,
                                    BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, PEEL, False, EV_BIAS)
                if HAS_GATE:
                    g = tl.load(g_ptrs, mask=dmask, other=0.0, eviction_policy=EV_STREAM).to(tl.float32)
                    acc = acc * (1.0 / (1.0 + tl.exp(-g)))
                tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=dmask, eviction_policy=EV_STREAM)
            else:                                                       # the last, partial query tile: production's masks
                q = tl.load(q_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0, eviction_policy=EV_STREAM)
                acc = _fab_key_loop(q, K_b, V_b, B_b, KM_b, offs_m, offs_n, offs_d, m_ok, d_ok,
                                    s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg,
                                    BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, PEEL, True, EV_BIAS)
                if HAS_GATE:
                    g = tl.load(g_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0, eviction_policy=EV_STREAM).to(tl.float32)
                    acc = acc * (1.0 / (1.0 + tl.exp(-g)))
                tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_ok[:, None] & d_ok[None, :], eviction_policy=EV_STREAM)
        else:
            q = tl.load(q_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0, eviction_policy=EV_STREAM)
            acc = _fab_key_loop(q, K_b, V_b, B_b, KM_b, offs_m, offs_n, offs_d, m_ok, d_ok,
                                s_kn, s_kd, s_vn, s_vd, s_bq, s_bk, N, qk_scale, mask_neg,
                                BM, BN, BD, HAS_BIAS, HAS_MASK, IN_PREC, PV_LOWP, PEEL, True, EV_BIAS)
            if HAS_GATE:
                g = tl.load(g_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0, eviction_policy=EV_STREAM).to(tl.float32)
                acc = acc * (1.0 / (1.0 + tl.exp(-g)))
            tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_ok[:, None] & d_ok[None, :], eviction_policy=EV_STREAM)

else:  # pragma: no cover
    _fab_batched_kernel = None


def _need():
    if not _HAVE_TRITON:
        raise KoptAttnUnavailable("opt_core.kernels.kopt_attn.fab_batched: triton is not importable here")


def flash_bias_attn_batched(q, k, v, bias=None, key_mask=None, gate=None, out=None, out_dtype=None, scale=None, mask_neg=-1e9,
                            group=None, BM=None, BN=None, num_warps=None, num_stages=None, in_prec=None, stage_c=None):
    """Pair-bias flash attention for M samples in ONE launch with ONE shared bias (no sample axis).

    q, k, v : [M, H, N, D] views, arbitrary strides (heads-last storage [M, N, H, D] is passed as x.permute(0, 2, 1, 3): no copies); unit stride on D
              expected (any stride is computed correctly, unit stride is what vectorises).
    bias    : [H, N, Nb] with Nb >= N (a padded PairBiasLayout layer [H, N, Np], or any [H, N, N] tensor / view; strides are honoured, columns >= N are
              never read) -- or [G, H, N, Nb] together with group = int32 [M] mapping sample -> bias group (seed x sample batches) -- or None.
    key_mask: [N] (shared) or [M, N] (per sample, e.g. one mask per alignment row), nonzero = valid, unit stride on N; or None.  Same soft-mask
              formula as production: logit += (1 - mask) * mask_neg.
    gate    : [M, N, H*D] pre-sigmoid gate logits or None.      out : [M, N, H*D] (allocated when None; out_dtype default = q.dtype).
    stage_c : None / False = production's masks on every tile (lever variant 'a'); True = Stage C options on (variant 'c'); str / dict = per option
              (peel, rowsplit, align, evict).  Every setting is intended to give the same bits.
    Returns out.  out[m] is intended BITWISE equal to production flash_bias_attn(q[m], k[m], v[m], bias[..., :N], key_mask(_m), gate[m]).
    BM / BN / num_warps / num_stages default to production's and never depend on M or G (a caller who overrides them leaves the bitwise contract)."""
    _need()
    import torch
    if q.dim() != 4:
        raise KoptAttnRefused("flash_bias_attn_batched: q must be [M, H, N, D] (got %s); use flash_bias_attn_v2 for [H, N, D]" % (tuple(q.shape),))
    M, H, N, D = q.shape
    if tuple(k.shape) != (M, H, N, D) or tuple(v.shape) != (M, H, N, D):
        raise KoptAttnRefused("flash_bias_attn_batched: k / v shapes %s / %s differ from q %s" % (tuple(k.shape), tuple(v.shape), tuple(q.shape)))
    if M < 1 or N < 1:
        raise KoptAttnRefused("flash_bias_attn_batched: empty problem M=%d N=%d" % (M, N))
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out_dtype = out_dtype or q.dtype
    if out is None:
        out = torch.empty((M, N, H * D), device=q.device, dtype=out_dtype)
    elif tuple(out.shape) != (M, N, H * D):
        raise KoptAttnRefused("flash_bias_attn_batched: out must be [M, N, H*D] = %s (got %s)" % ((M, N, H * D), tuple(out.shape)))
    BD = _pow2(D, 16)
    lowp = q.dtype in (torch.bfloat16, torch.float16)
    if in_prec is None:
        in_prec = _fp32_prec()
    if lowp:
        in_prec = "ieee"  # ignored by triton for 16-bit operands, keep a valid literal (as production does)
    cBM, cBN, cW, cS = tile_config(N, D, lowp)
    BM = cBM if BM is None else BM
    BN = cBN if BN is None else BN
    num_warps = cW if num_warps is None else num_warps
    num_stages = cS if num_stages is None else num_stages
    flags = _stage_c_flags(stage_c)

    # ---- bias / group
    has_group = False
    s_bg = s_bh = s_bq = s_bk = 0
    b = q
    grp = q
    if bias is not None:
        if bias.dim() == 3:
            if group is not None:
                raise KoptAttnRefused("flash_bias_attn_batched: group given but bias has no group axis (bias %s)" % (tuple(bias.shape),))
            if bias.shape[0] != H or bias.shape[1] != N or bias.shape[2] < N:
                raise KoptAttnRefused("flash_bias_attn_batched: bias must be [H, N, >=N] = [%d, %d, >=%d] (got %s)" % (H, N, N, tuple(bias.shape)))
            s_bh, s_bq, s_bk = bias.stride(0), bias.stride(1), bias.stride(2)
        elif bias.dim() == 4:
            if group is None:
                raise KoptAttnRefused("flash_bias_attn_batched: bias [G, H, N, Nb] needs group = int32 [M] (a per-sample bias [M, H, N, N] is what this "
                                      "lever removes; pass group=torch.arange(M) only if you really mean it)")
            if bias.shape[1] != H or bias.shape[2] != N or bias.shape[3] < N:
                raise KoptAttnRefused("flash_bias_attn_batched: bias must be [G, H, N, >=N] (got %s)" % (tuple(bias.shape),))
            if group.dim() != 1 or group.shape[0] != M or group.dtype not in (torch.int32, torch.int64) or group.stride(0) != 1:
                raise KoptAttnRefused("flash_bias_attn_batched: group must be a contiguous int32 / int64 [M] tensor")
            if group.device != q.device:
                raise KoptAttnRefused("flash_bias_attn_batched: group must live on the device of q (no host sync is done to move or range-check it)")
            has_group = True
            grp = group
            s_bg, s_bh, s_bq, s_bk = bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3)
        else:
            raise KoptAttnRefused("flash_bias_attn_batched: bias must be [H, N, Nb] or [G, H, N, Nb] (got %s)" % (tuple(bias.shape),))
        b = bias
    elif group is not None:
        raise KoptAttnRefused("flash_bias_attn_batched: group given without bias")

    # ---- key mask
    s_mm = 0
    km = q
    if key_mask is not None:
        if key_mask.dim() == 1 and key_mask.shape[0] == N:
            s_mm = 0
        elif key_mask.dim() == 2 and tuple(key_mask.shape) == (M, N):
            s_mm = key_mask.stride(0)
        else:
            raise KoptAttnRefused("flash_bias_attn_batched: key_mask must be [N] or [M, N] (got %s)" % (tuple(key_mask.shape),))
        if N > 1 and key_mask.stride(-1) != 1:
            raise KoptAttnRefused("flash_bias_attn_batched: key_mask needs unit stride on N (got %d)" % key_mask.stride(-1))
        km = key_mask
    g = out
    s_gm = s_gn = s_gc = 0
    if gate is not None:
        if tuple(gate.shape) != (M, N, H * D):
            raise KoptAttnRefused("flash_bias_attn_batched: gate must be [M, N, H*D] = %s (got %s)" % ((M, N, H * D), tuple(gate.shape)))
        g = gate
        s_gm, s_gn, s_gc = gate.stride(0), gate.stride(1), gate.stride(2)

    n_qt = (N + BM - 1) // BM
    n_prog = M * n_qt * H
    if n_prog >= 2 ** 31:
        raise KoptAttnRefused("flash_bias_attn_batched: grid M * cdiv(N, BM) * H = %d >= 2^31" % n_prog)

    # Stage C (i): declare the bias strides multiples of 16 elements only when that is TRUE and the base pointer is 16-byte aligned
    align_b = bool(flags["align"] and bias is not None and s_bk == 1 and s_bq % 16 == 0 and s_bh % 16 == 0 and (s_bg % 16 == 0)
                   and bias.data_ptr() % 16 == 0)
    ev_bias = "evict_last" if flags["evict"] else ""
    ev_stream = "evict_first" if flags["evict"] else ""

    _fab_batched_kernel[(n_prog,)](
        q, k, v, b, km, g, out, grp,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        s_bg, s_bh, s_bq, s_bk,
        s_mm,
        s_gm, s_gn, s_gc,
        out.stride(0), out.stride(1), out.stride(2),
        M, N, float(scale * _LOG2E), float(mask_neg),
        H=H, D=D, BD=BD, BM=BM, BN=BN,
        HAS_BIAS=bias is not None, HAS_MASK=key_mask is not None, HAS_GATE=gate is not None, HAS_GROUP=has_group,
        IN_PREC=in_prec, PV_LOWP=lowp,
        PEEL=bool(flags["peel"]), ROWSPLIT=bool(flags["rowsplit"]), ALIGN_B=align_b,
        EV_BIAS=ev_bias, EV_STREAM=ev_stream,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def flash_bias_attn_v2(q, k, v, bias=None, key_mask=None, gate=None, out=None, out_dtype=None, scale=None, mask_neg=-1e9,
                       BM=None, BN=None, num_warps=None, num_stages=None, in_prec=None, stage_c=True):
    """The M = 1 entry with the Stage C options on; drop-in for production flash_bias_attn: q, k, v [H, N, D] views, bias [H, N, Nb >= N] (a padded
    PairBiasLayout layer, or the plain [H, N, N]), key_mask [N], gate [N, H*D]; returns out [N, H*D]."""
    if q.dim() != 3:
        raise KoptAttnRefused("flash_bias_attn_v2: q must be [H, N, D] (got %s)" % (tuple(q.shape),))
    o = flash_bias_attn_batched(q[None], k[None], v[None], bias=bias, key_mask=key_mask,
                                gate=(gate[None] if gate is not None else None), out=(out[None] if out is not None else None),
                                out_dtype=out_dtype, scale=scale, mask_neg=mask_neg, group=None, BM=BM, BN=BN, num_warps=num_warps,
                                num_stages=num_stages, in_prec=in_prec, stage_c=stage_c)
    return out if out is not None else o[0]


def flash_bias_attn_looped(q, k, v, bias=None, key_mask=None, gate=None, out=None, out_dtype=None, scale=None, mask_neg=-1e9, group=None):
    """The reference path of the flag (KOPT_ATTN_L3A = 0): production dtk_kernels.flash_bias_attn, one launch per sample, same argument convention as
    flash_bias_attn_batched (a padded bias is sliced to [..., :N], which production reads through its strides)."""
    import torch
    from opt_core.kernels import dtk_kernels as _K
    M, H, N, D = q.shape
    if out is None:
        out = torch.empty((M, N, H * D), device=q.device, dtype=out_dtype or q.dtype)
    gl = group.tolist() if (group is not None and bias is not None and bias.dim() == 4) else None   # host sync: reference path only
    for m in range(M):
        bm = None
        if bias is not None:
            bm = (bias[gl[m]] if gl is not None else bias)[..., :N]
        kmm = None
        if key_mask is not None:
            kmm = key_mask if key_mask.dim() == 1 else key_mask[m]
        _K.flash_bias_attn(q[m], k[m], v[m], bias=bm, key_mask=kmm, gate=(gate[m] if gate is not None else None), out=out[m], scale=scale,
                           mask_neg=mask_neg)
    return out


def pair_bias_attention(q, k, v, bias=None, key_mask=None, gate=None, out=None, out_dtype=None, scale=None, mask_neg=-1e9, group=None, mode=None):
    """Dispatcher behind the ONE flag.  mode None -> env KOPT_ATTN_L3A: '0' production kernel looped per sample | 'a' batched (bitwise at every N) |
    'c' batched + Stage C (bitwise expected at N % 64 == 0, else same error class) | 'auto' = 'c' where it is bitwise, else 'a' (bitwise everywhere)."""
    mode = l3a_mode() if mode is None else mode
    if mode == "auto":
        mode = auto_variant(q.shape[-2], bias)
    if mode == "0":
        return flash_bias_attn_looped(q, k, v, bias=bias, key_mask=key_mask, gate=gate, out=out, out_dtype=out_dtype, scale=scale, mask_neg=mask_neg,
                                      group=group)
    return flash_bias_attn_batched(q, k, v, bias=bias, key_mask=key_mask, gate=gate, out=out, out_dtype=out_dtype, scale=scale, mask_neg=mask_neg,
                                   group=group, stage_c=(mode == "c"))


def check_finite(t, name="tensor"):
    """writer-side non-finite check (host sync; call outside CUDA-graph capture): raises FloatingPointError naming the tensor."""
    import torch
    n_bad = int((~torch.isfinite(t)).sum())
    if n_bad:
        raise FloatingPointError("%s: %d non-finite values" % (name, n_bad))
    return t
