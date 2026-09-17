# dtk_kernels.py -- Diffusion-Transformer Kernels (DTK) v0.2
# Triton kernels for the AF3-family diffusion module (token transformer + atom transformer):
#   flash_bias_attn   : softmax(q k^T * scale + bias [+ key mask]) v  [* sigmoid(gate)]  -- flash-style, static pre-laid bias
#   window_attn       : AF3 sequence-local atom attention (32 queries x 128 keys per block) as ONE block-sparse flash kernel
#   ln_modulate       : AdaLN   y = sigmoid(scale) * LayerNorm_noaffine(x) + shift        (scale/shift = s-side GEMM outputs)
#   swiglu            : silu(a) * b from the concatenated [.., 2h] GEMM output
#   gate_residual     : res + rowmask * sigmoid(g) * x   (adaLN-zero output gate, the transition's row mask, residual add)
#   seg_reduce        : per-token masked mean / sum of atom features over each token's contiguous atom run (atom -> token aggregation;
#                       `segments` derives the runs from the atom->token index once per input)
# All kernels: deterministic (no atomics, fixed reduction order), CUDA-graph capturable, no host syncs.
# Row kernels address an optional PERIODIC operand: a modulation / gate / factor given as [P, C] with P dividing R serves row r from its row
# r % P (a conditioning tensor that is one block of P rows broadcast over R/P leading copies is handed as the block, never materialised);
# gate_residual takes the gate pre-sigmoid (default) or post-sigmoid (sigmoid_gate=False) and an optional periodic 1-D row mask.
import math
import torch
import triton
import triton.language as tl

from opt_core.kernels import safe_settings as _safe

_LOG2E = 1.4426950408889634


import os


def _fp32_prec():
    """Precision of tl.dot for fp32 operands: 'ieee' (default; same class as fp32 SDPA / matmul precision 'highest') or 'tf32'
    (env DTK_FP32_PREC=tf32; a precision substitution -> table separately, never headline)."""
    p = os.environ.get("DTK_FP32_PREC", "ieee").lower()
    return p if p in ("ieee", "tf32", "tf32x3") else "ieee"


def _pow2(x, lo=16):
    p = lo
    while p < x:
        p *= 2
    return p


# Row-kernel launch settings (ln_modulate / swiglu / gate_residual: one program per row [x column block], no loop, so num_warps is the one launch
# setting). _ROW_WARPS = the capability-free default (measured on H100; cc 9.0 has no row and is served it); _ROW_WARPS_BY_CC
# ["<cc>|<triton major.minor>"] (a NAMED EXCEPTION for one known environment) else ["<cc>|*"] (the capability's row) serves a device of that
# capability instead (opt_core.kernels.safe_settings.row_for_device; row_warps()). Per kernel: (max block, num_warps) pairs over the kernel's
# constexpr tile width -- BC = next_pow2(C) (>= 32) for ln_modulate / gate_residual, BH = min(next_pow2(h), 1024) for swiglu --; the first pair
# whose max block >= the tile wins. num_warps changes the in-tile reduction tree of ln_modulate's statistics: part of the numerics, never tuned
# at run time.
#   "8.0|*"   A100-SXM4-80GB (cc 8.0), torch 2.13.0+cu130 / triton 3.7.1 -- the default's values, MEASURED there: device time per call (calls captured
#             in one CUDA graph, replayed) over warps 1/2/4/8/16 at rows R = 100 .. 64,000 of C = 768 (ln_modulate / gate_residual, BC 1024) and
#             swiglu h = 1536 (BH 1024), fp32 and bf16 rows: 4 warps is the fastest or within 2 % of it at every R in fp32 ln_modulate (R 64k:
#             235 us; 1 / 2 / 8 / 16 warps 236 / 237 / 249 / 377), bf16 swiglu (450 us; 451 / 455 / 462 / 498) and gate_residual (both dtypes,
#             +-1 %); bf16 ln_modulate rows prefer 1 warp at R 5-6k only (10.0 vs 13.6 us; R >= 12.8k within 4 %). Narrower tiles (BC 128-512,
#             R 8192 ladder) run 13-25 % faster with 1-2 warps in ln_modulate -- not keyed: no caller in the release tree launches them.
_ROW_WARPS = {"ln_modulate": ((1024, 4), (1 << 30, 8)), "swiglu": ((1 << 30, 4),), "gate_residual": ((1024, 4), (1 << 30, 8))}
_ROW_WARPS_BY_CC = {
    "8.0|*": {"ln_modulate": ((1024, 4), (1 << 30, 8)), "swiglu": ((1 << 30, 4),), "gate_residual": ((1024, 4), (1 << 30, 8))},
}
_rows_resolved = {}     # device slot -> the _ROW_WARPS[_BY_CC] row serving it (safe_settings.row_for_device memo)


_NET = _safe.SafeNet("dtk_kernels")          # names the default-row engagement of a capability without a row, once per process (cells_note())


def row_settings(device=None):
    """The row-kernel settings serving a device (default: the current CUDA device): its capability's _ROW_WARPS_BY_CC row, else _ROW_WARPS (no
    CUDA, a CPU device, cc 9.0 whose measurements _ROW_WARPS are, or a capability without a row — the last said ONCE:
    ``[opt_core/dtk_kernels] default settings (no row for cc <cc>, …); tuned rows exist for cc …``, ``cells_note() == 'default:no_row'``)."""
    row = _safe.row_for_device(_ROW_WARPS_BY_CC, device, _ROW_WARPS, cache=_rows_resolved)
    if row is _ROW_WARPS and _NET.note() is None:
        from opt_core.kernels import cell_words as _cw
        _cw.name_default_row(_NET, _ROW_WARPS_BY_CC, _safe.device_cc(device), _safe.triton_mm())
    return row


def cells_note(device=None):
    """``default:no_row`` once a CUDA device of a capability without a row (other than cc 9.0) was served _ROW_WARPS, else None."""
    row_settings(device)
    return _NET.note()


def row_warps(kernel, block, device=None):
    """num_warps for row kernel ``kernel`` ("ln_modulate" | "swiglu" | "gate_residual") at tile width ``block`` on ``device``'s capability."""
    return int(_safe.pick_by_block(row_settings(device).get(kernel) or _ROW_WARPS[kernel], block))


# --------------------------------------------------------------------------------------
# 1. flash attention with additive pair bias  (q,k,v: [H, N, D] any strides; bias [H, N, N]; mask [N] (1 = keep))
# --------------------------------------------------------------------------------------
@triton.jit
def _fab_fwd_kernel(Q, K, V, B, KM, G, O,
                    s_qh, s_qn, s_qd,
                    s_kh, s_kn, s_kd,
                    s_vh, s_vn, s_vd,
                    s_bh, s_bq, s_bk,
                    s_gn, s_gc,
                    s_on, s_oc,
                    N, qk_scale, mask_neg,
                    H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr,
                    HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr,
                    IN_PREC: tl.constexpr, PV_LOWP: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)              # int64 program indices: every row / head / key offset below is a 64-bit product
    h = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    m_ok = offs_m < N
    d_ok = offs_d < D
    q_ptrs = Q + h * s_qh + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd
    q = tl.load(q_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0)
    m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, BD], dtype=tl.float32)
    for start_n in range(0, N, BN):
        cur_n = start_n + offs_n
        n_ok = cur_n < N
        cur_n64 = cur_n.to(tl.int64)
        k_ptrs = K + h * s_kh + cur_n64[:, None] * s_kn + offs_d[None, :] * s_kd
        k = tl.load(k_ptrs, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision=IN_PREC)          # fp32 [BM, BN]
        qk = qk * qk_scale                                             # scale * log2(e)
        if HAS_BIAS:
            b_ptrs = B + h * s_bh + offs_m[:, None] * s_bq + cur_n64[None, :] * s_bk
            b = tl.load(b_ptrs, mask=m_ok[:, None] & n_ok[None, :], other=0.0).to(tl.float32)
            qk = qk + b * 1.4426950408889634
        if HAS_MASK:
            km = tl.load(KM + cur_n, mask=n_ok, other=0).to(tl.float32)
            qk = qk + (1.0 - km)[None, :] * (mask_neg * 1.4426950408889634)
        qk = tl.where(n_ok[None, :], qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = V + h * s_vh + cur_n64[:, None] * s_vn + offs_d[None, :] * s_vd
        v = tl.load(v_ptrs, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        if PV_LOWP:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=IN_PREC)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision=IN_PREC)
        m_i = m_new
    acc = acc / l_i[:, None]
    if HAS_GATE:
        g_ptrs = G + offs_m[:, None] * s_gn + (h * D + offs_d)[None, :] * s_gc
        g = tl.load(g_ptrs, mask=m_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
        acc = acc * (1.0 / (1.0 + tl.exp(-g)))
    o_ptrs = O + offs_m[:, None] * s_on + (h * D + offs_d)[None, :] * s_oc
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_ok[:, None] & d_ok[None, :])


def flash_bias_attn(q, k, v, bias=None, key_mask=None, gate=None, out=None, out_dtype=None,
                    scale=None, mask_neg=-1e9, BM=None, BN=None, num_warps=None, num_stages=2,
                    in_prec=None):
    """q,k,v: [H, N, D] (views with arbitrary strides OK, e.g. heads-last storage). bias: [H, N, N] (bf16/fp16/fp32) or None.
    key_mask: [N] (nonzero = valid) or None. gate: [N, H*D] pre-sigmoid gate logits or None.
    Returns out [N, H*D] (out_dtype default = q.dtype). Softmax statistics in fp32; P@V in v.dtype for 16-bit inputs
    (same class as SDPA/flash), fp32 for fp32 inputs. in_prec: None -> 'tf32' if torch allows tf32 matmul else 'ieee' (fp32 inputs);
    ignored for 16-bit inputs."""
    H, N, D = q.shape
    assert k.shape == (H, N, D) and v.shape == (H, N, D)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out_dtype = out_dtype or q.dtype
    if out is None:
        out = torch.empty((N, H * D), device=q.device, dtype=out_dtype)
    BD = _pow2(D, 16)
    lowp = q.dtype in (torch.bfloat16, torch.float16)
    if in_prec is None:
        in_prec = _fp32_prec()
    if lowp:
        in_prec = "ieee"  # ignored by triton for 16-bit operands, keep a valid literal
    if BM is None:
        BM = 64 if lowp else 32
    if BN is None:
        BN = 64 if lowp else 32
    if num_warps is None:
        num_warps = 4
    grid = (triton.cdiv(N, BM), H)
    b = bias if bias is not None else q
    km = key_mask if key_mask is not None else q
    g = gate if gate is not None else out
    _fab_fwd_kernel[grid](
        q, k, v, b, km, g, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        (b.stride(0) if bias is not None else 0), (b.stride(1) if bias is not None else 0), (b.stride(2) if bias is not None else 0),
        (g.stride(0) if gate is not None else 0), (g.stride(1) if gate is not None else 0),
        out.stride(0), out.stride(1),
        N, float(scale * _LOG2E), float(mask_neg),
        H=H, D=D, BD=BD, BM=BM, BN=BN,
        HAS_BIAS=bias is not None, HAS_MASK=key_mask is not None, HAS_GATE=gate is not None,
        IN_PREC=in_prec, PV_LOWP=lowp,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# --------------------------------------------------------------------------------------
# 2. AF3 sequence-local atom attention: query block b = atoms [32b, 32b+32), keys = atoms [32b-48, 32b+80) ∩ [0, Na)
#    q,k,v: [H, Na, D] strided views; bias_blk: [H, NB, 32, 128] (models store the atom pair bias in this blocked layout);
#    atom_mask [Na] (valid atoms); out [Na, H*D]
# --------------------------------------------------------------------------------------
@triton.jit
def _win_fwd_kernel(Q, K, V, B, AM, G, O,
                    s_qh, s_qn, s_qd, s_kh, s_kn, s_kd, s_vh, s_vn, s_vd,
                    s_bh, s_bb, s_bq, s_bk,
                    s_gn, s_gc, s_on, s_oc,
                    NA, qk_scale, mask_neg,
                    H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                    NQ: tl.constexpr, NK: tl.constexpr, BN: tl.constexpr,
                    HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr, IN_PREC: tl.constexpr, PV_LOWP: tl.constexpr):
    b_id = tl.program_id(0).to(tl.int64)               # int64 program indices: every atom-row / head / bias-block offset below is a 64-bit product
    h = tl.program_id(1).to(tl.int64)
    offs_m = b_id * NQ + tl.arange(0, NQ)
    offs_d = tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    m_ok = offs_m < NA
    d_ok = offs_d < D
    q = tl.load(Q + h * s_qh + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd, mask=m_ok[:, None] & d_ok[None, :], other=0.0)
    key0 = b_id * NQ + NQ // 2 - NK // 2
    m_i = tl.full([NQ], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([NQ], dtype=tl.float32)
    acc = tl.zeros([NQ, BD], dtype=tl.float32)
    for j0 in range(0, NK, BN):
        jj = j0 + offs_n                      # position inside the 128-key window
        kn = key0 + jj                        # absolute atom index (may be <0 or >= NA -> padding key)
        n_ok = (kn >= 0) & (kn < NA)
        k = tl.load(K + h * s_kh + kn[:, None] * s_kn + offs_d[None, :] * s_kd, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision=IN_PREC) * qk_scale
        bb = tl.load(B + h * s_bh + b_id * s_bb + tl.arange(0, NQ)[:, None] * s_bq + jj[None, :] * s_bk,
                     mask=(jj[None, :] < NK), other=0.0).to(tl.float32)
        qk = qk + bb * 1.4426950408889634
        valid = n_ok
        if HAS_MASK:
            am = tl.load(AM + kn, mask=n_ok, other=0)
            valid = valid & (am != 0)
        qk = qk + tl.where(valid, 0.0, mask_neg * 1.4426950408889634)[None, :]
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + h * s_vh + kn[:, None] * s_vn + offs_d[None, :] * s_vd, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        if PV_LOWP:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=IN_PREC)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision=IN_PREC)
        m_i = m_new
    acc = acc / l_i[:, None]
    if HAS_GATE:
        g = tl.load(G + offs_m[:, None] * s_gn + (h * D + offs_d)[None, :] * s_gc, mask=m_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
        acc = acc * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(O + offs_m[:, None] * s_on + (h * D + offs_d)[None, :] * s_oc, acc.to(O.dtype.element_ty), mask=m_ok[:, None] & d_ok[None, :])


def window_attn(q, k, v, bias_blk, atom_mask=None, gate=None, out=None, out_dtype=None, scale=None,
                mask_neg=-1e9, NQ=32, NK=128, BN=64, num_warps=4, num_stages=2, in_prec=None):
    H, NA, D = q.shape
    NB = bias_blk.shape[1]
    assert bias_blk.shape == (H, NB, NQ, NK), bias_blk.shape
    assert NB * NQ >= NA
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out_dtype = out_dtype or q.dtype
    if out is None:
        out = torch.empty((NA, H * D), device=q.device, dtype=out_dtype)
    BD = _pow2(D, 16)
    lowp = q.dtype in (torch.bfloat16, torch.float16)
    if in_prec is None:
        in_prec = _fp32_prec()
    if lowp:
        in_prec = "ieee"
    g = gate if gate is not None else out
    am = atom_mask if atom_mask is not None else q
    _win_fwd_kernel[(NB, H)](
        q, k, v, bias_blk, am, g, out,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        bias_blk.stride(0), bias_blk.stride(1), bias_blk.stride(2), bias_blk.stride(3),
        (g.stride(0) if gate is not None else 0), (g.stride(1) if gate is not None else 0), out.stride(0), out.stride(1),
        NA, float(scale * _LOG2E), float(mask_neg),
        H=H, D=D, BD=BD, NQ=NQ, NK=NK, BN=BN,
        HAS_MASK=atom_mask is not None, HAS_GATE=gate is not None, IN_PREC=in_prec, PV_LOWP=lowp,
        num_warps=num_warps, num_stages=num_stages)
    return out


# --------------------------------------------------------------------------------------
# 3. LayerNorm / AdaLN-modulate / SwiGLU / gate+residual  (row kernels, fp32 statistics)
# --------------------------------------------------------------------------------------
@triton.jit
def _ln_mod_kernel(X, SC, SH, W, Bv, Y, C, s_x, s_sc, s_sh, s_y, eps,
                   BC: tl.constexpr, HAS_MOD: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr, SIGMOID_SCALE: tl.constexpr,
                   RMS: tl.constexpr = False, PER=1, HAS_PER: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)                # int64 row index: row * pitch is a 64-bit product
    offs = tl.arange(0, BC)
    ok = offs < C
    if HAS_PER:
        mrow = row % PER                          # periodic modulation operand: [PER, C] serves row r from row r % PER
    else:
        mrow = row
    x = tl.load(X + row * s_x + offs, mask=ok, other=0.0).to(tl.float32)
    if RMS:
        xc = x                                   # RMSNorm: no mean subtraction (torch.nn.RMSNorm: x * rsqrt(mean(x^2) + eps))
    else:
        mean = tl.sum(x, 0) / C
        xc = tl.where(ok, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / C
    y = xc / tl.sqrt(var + eps)
    if HAS_W:
        w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
        y = y * w
    if HAS_B:
        bv = tl.load(Bv + offs, mask=ok, other=0.0).to(tl.float32)
        y = y + bv
    if HAS_MOD:
        sc = tl.load(SC + mrow * s_sc + offs, mask=ok, other=0.0).to(tl.float32)
        sh = tl.load(SH + mrow * s_sh + offs, mask=ok, other=0.0).to(tl.float32)
        if SIGMOID_SCALE:
            sc = 1.0 / (1.0 + tl.exp(-sc))
        y = sc * y + sh
    tl.store(Y + row * s_y + offs, y.to(Y.dtype.element_ty), mask=ok)


def ln_modulate(x, scale=None, shift=None, weight=None, bias=None, eps=1e-5, out_dtype=None, sigmoid_scale=True, out=None, rms=False,
                mod_period=None):
    """y = [sigmoid](scale) * NORM(x)[*weight+bias] + shift ; NORM = LayerNorm (rms=False) or RMSNorm (rms=True), fp32 statistics;
    x [R, C] (unit column stride; row stride free, e.g. a column slice of a wider GEMM output), scale/shift [R, C] views (row stride free),
    or -- mod_period=P, P dividing R -- scale/shift [P, C]: row r is modulated by their row r % P (a conditioning block broadcast over R/P
    leading copies, handed as the block)."""
    C = x.shape[-1]
    x2 = x if x.dim() == 2 else x.reshape(-1, C)
    assert x2.stride(-1) == 1
    R = x2.shape[0]
    out_dtype = out_dtype or x.dtype
    if out is None:
        out = torch.empty((R, C), device=x.device, dtype=out_dtype)
    has_mod = scale is not None
    sc = scale if has_mod else x2
    sh = shift if has_mod else x2
    has_per = has_mod and mod_period is not None
    per = int(mod_period) if has_per else 1
    if has_mod:
        assert sc.dim() == 2 and sh.dim() == 2 and sc.stride(-1) == 1 and sh.stride(-1) == 1 and sc.shape[-1] == C and sh.shape[-1] == C
        assert sc.shape[0] == sh.shape[0] == (per if has_per else R) and R % (per if has_per else R) == 0, (R, per, tuple(sc.shape))
    BC = _pow2(C, 32)
    nw = row_warps("ln_modulate", BC, x2.device)
    _ln_mod_kernel[(R,)](x2, sc, sh, weight if weight is not None else x2, bias if bias is not None else x2, out, C,
                         x2.stride(0), (sc.stride(0) if has_mod else 0), (sh.stride(0) if has_mod else 0), out.stride(0), eps,
                         BC=BC, HAS_MOD=has_mod, HAS_W=weight is not None, HAS_B=bias is not None, SIGMOID_SCALE=sigmoid_scale,
                         RMS=bool(rms), PER=per, HAS_PER=has_per, num_warps=nw)
    return out if x.dim() == 2 else out.reshape(x.shape[:-1] + (C,))


@triton.jit
def _swiglu_kernel(AB, Y, Hd, s_ab, s_y, BH: tl.constexpr, A_FIRST_SILU: tl.constexpr, Cf=None, s_c=0, HAS_C: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)                # int64 row index: row * pitch is a 64-bit product
    blk = tl.program_id(1)
    offs = blk * BH + tl.arange(0, BH)
    ok = offs < Hd
    a = tl.load(AB + row * s_ab + offs, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(AB + row * s_ab + Hd + offs, mask=ok, other=0.0).to(tl.float32)
    if A_FIRST_SILU:
        y = a * (1.0 / (1.0 + tl.exp(-a))) * b
    else:
        y = b * (1.0 / (1.0 + tl.exp(-b))) * a
    if HAS_C:
        c = tl.load(Cf + row * s_c + offs, mask=ok, other=0.0).to(tl.float32)
        y = y * c                                 # third factor: (silu(.) * .) * c, the product order of a SwiGLU gate times a value GEMM
    tl.store(Y + row * s_y + offs, y.to(Y.dtype.element_ty), mask=ok)


def swiglu(ab, out_dtype=None, a_first_silu=True, c=None, out=None):
    """ab [R, 2h] (unit column stride, row stride free) -> silu(a) * b  [R, h]  (a = first half if a_first_silu); c [R, h] (row stride free)
    an optional third factor: (silu(.) * .) * c -- e.g. ab and c the column blocks of ONE GEMM output [R, 3h]."""
    R, H2 = ab.shape
    Hd = H2 // 2
    assert ab.stride(-1) == 1
    if c is not None:
        assert c.shape == (R, Hd) and c.stride(-1) == 1, (tuple(c.shape), c.stride())
    if out is None:
        out = torch.empty((R, Hd), device=ab.device, dtype=out_dtype or ab.dtype)
    BH = 1024 if Hd >= 1024 else _pow2(Hd, 32)
    _swiglu_kernel[(R, triton.cdiv(Hd, BH))](ab, out, Hd, ab.stride(0), out.stride(0), BH=BH, A_FIRST_SILU=a_first_silu,
                                              Cf=c if c is not None else ab, s_c=(c.stride(0) if c is not None else 0), HAS_C=c is not None,
                                              num_warps=row_warps("swiglu", BH, ab.device))
    return out


@triton.jit
def _gate_res_kernel(X, Gt, R_, Y, C, s_x, s_g, s_r, s_y, BC: tl.constexpr, HAS_RES: tl.constexpr, HAS_GATE: tl.constexpr,
                     PER=1, HAS_PER: tl.constexpr = False, SIGMOID_GATE: tl.constexpr = True,
                     MK=None, s_mk=0, PM=1, HAS_MASK: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)                # int64 row index: row * pitch is a 64-bit product
    offs = tl.arange(0, BC)
    ok = offs < C
    x = tl.load(X + row * s_x + offs, mask=ok, other=0.0).to(tl.float32)
    if HAS_GATE:
        if HAS_PER:
            grow = row % PER                      # periodic gate operand: [PER, C] serves row r from row r % PER
        else:
            grow = row
        g = tl.load(Gt + grow * s_g + offs, mask=ok, other=0.0).to(tl.float32)
        if SIGMOID_GATE:
            g = 1.0 / (1.0 + tl.exp(-g))
        x = x * g
    if HAS_MASK:
        m = tl.load(MK + (row % PM) * s_mk).to(tl.float32)   # periodic row mask: [PM] serves row r from element r % PM
        x = x * m
    if HAS_RES:
        r = tl.load(R_ + row * s_r + offs, mask=ok, other=0.0).to(tl.float32)
        x = x + r
    tl.store(Y + row * s_y + offs, x.to(Y.dtype.element_ty), mask=ok)


def gate_residual(x, gate=None, res=None, out_dtype=None, out=None, gate_period=None, sigmoid_gate=True, rowmask=None, mask_period=None):
    """y = res + rowmask * [sigmoid](gate) * x ; x / res / out [R, C] row views with unit last stride; gate_period=P (P dividing R): gate
    [P, C], row r gated by its row r % P (a conditioning block broadcast over R/P leading copies, handed as the block); sigmoid_gate=False:
    the gate is given post-sigmoid (multiplied as is); rowmask [PM] 1-D (any float / bool dtype, free element stride) with mask_period=PM
    dividing R (row r scaled by element r % PM; default PM = R). Every operand optional; fp32 arithmetic, one rounding to out's dtype
    (default: res's dtype if given, else x's)."""
    R, C = x.shape
    assert x.stride(-1) == 1 or C == 1, x.stride()
    out_dtype = out_dtype or (res.dtype if res is not None else x.dtype)
    if out is None:
        out = torch.empty((R, C), device=x.device, dtype=out_dtype)
    if R == 0 or C == 0:
        return out
    g = gate if gate is not None else x
    r = res if res is not None else x
    has_per = gate is not None and gate_period is not None
    per = int(gate_period) if has_per else 1
    if gate is not None:
        assert g.dim() == 2 and (g.stride(-1) == 1 or C == 1) and g.shape[-1] == C and g.shape[0] == (per if has_per else R) and R % (per if has_per else R) == 0, (R, per, tuple(g.shape))
    if res is not None:
        assert r.shape == (R, C) and (r.stride(-1) == 1 or C == 1), (tuple(r.shape), r.stride())
    has_mask = rowmask is not None
    pm = 1
    mk = x
    if has_mask:
        mk = rowmask.to(torch.float32) if rowmask.dtype == torch.bool else rowmask
        pm = int(mask_period) if mask_period is not None else R
        assert mk.dim() == 1 and mk.shape[0] == pm and R % pm == 0, (R, pm, tuple(mk.shape))
    BC = _pow2(C, 32)
    _gate_res_kernel[(R,)](x, g, r, out, C, x.stride(0), (g.stride(0) if gate is not None else 0), (r.stride(0) if res is not None else 0),
                           out.stride(0), BC=BC, HAS_RES=res is not None, HAS_GATE=gate is not None, PER=per, HAS_PER=has_per,
                           SIGMOID_GATE=bool(sigmoid_gate), MK=mk, s_mk=(mk.stride(0) if has_mask else 0), PM=pm, HAS_MASK=has_mask,
                           num_warps=row_warps("gate_residual", BC, x.device))
    return out


# ---------------------------------------------------------------------------------------------------------------- atom -> token aggregation
# The all-atom models lay atoms out token by token: the atom->token index is non-decreasing over the atom axis, so token j's atoms are the
# contiguous run [start_j, start_j + count_j). `segments` derives the runs once per input (token j's run = the position span of its UNMASKED
# atoms; a masked atom inside a span weighs 0 in the sum and the count, the padded tail is never read; a token without unmasked atoms gets
# count 0 -> output 0); `seg_reduce` then reads every feature row once: out[s, j, :] = sum_t mask[a] * F[s, a, :] over a = start_j + t,
# t < count_j, divided by (sum mask + eps) for the mean. fp32 accumulation in a fixed order (deterministic — the scatter-add form is not),
# int64 addressing, the loop bound read from a DEVICE scalar (max atoms per token) so a captured CUDA graph stays correct when a later input
# with longer runs refreshes the segment tensors in place.
@triton.jit
def _seg_reduce_kernel(F, MK, ST, CT, MAXC, Y, N, C, A,
                       s_fs, s_fa, s_ys, s_yn, eps,
                       BT: tl.constexpr, BC: tl.constexpr, MEAN: tl.constexpr, HAS_MASK: tl.constexpr):
    pid_t = tl.program_id(0); pid_s = tl.program_id(1); pid_c = tl.program_id(2)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_c = pid_c * BC + tl.arange(0, BC)
    t_ok = offs_t < N
    c_ok = offs_c < C
    st = tl.load(ST + offs_t, mask=t_ok, other=0)
    ct = tl.load(CT + offs_t, mask=t_ok, other=0)
    maxc = tl.load(MAXC)
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    msum = tl.zeros((BT,), dtype=tl.float32)
    f_base = F + pid_s.to(tl.int64) * s_fs
    for i in range(0, maxc):
        live = t_ok & (i < ct)
        row = (st + i).to(tl.int64)
        if HAS_MASK:
            m = tl.load(MK + row, mask=live, other=0.0).to(tl.float32)
        else:
            m = live.to(tl.float32)
        xx = tl.load(f_base + row[:, None] * s_fa + offs_c[None, :], mask=live[:, None] & c_ok[None, :], other=0.0).to(tl.float32)
        acc += xx * m[:, None]
        msum += m
    if MEAN:
        acc = acc / (msum[:, None] + eps)
    y_ptrs = Y + pid_s.to(tl.int64) * s_ys + offs_t.to(tl.int64)[:, None] * s_yn + offs_c[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=t_ok[:, None] & c_ok[None, :])


def segments(atom_to_token_index, atom_mask, n_token):
    """(starts int32 [N], counts int32 [N], maxc int32 [1], sorted: bool) from a 1-D atom->token index [A] and atom mask [A] (device tensors).
    `sorted` = the unmasked atoms' indices are non-decreasing along the atom axis and inside [0, N) (then runs cannot interleave); False ->
    the runs are all-zero and the caller must not use them (serve the statement another way, by name). ONE host sync (that verdict): call once
    per input, never inside a CUDA-graph capture."""
    idx = atom_to_token_index.reshape(-1).long()
    mk = atom_mask.reshape(-1) > 0
    A = idx.numel(); N = int(n_token)
    pos = torch.arange(A, device=idx.device)
    iv = idx[mk]; pv = pos[mk]
    inrange = bool(((iv >= 0) & (iv < N)).all().item()) if iv.numel() else True
    ok = inrange and (bool((iv[1:] >= iv[:-1]).all().item()) if iv.numel() > 1 else True)
    if not ok:
        z = torch.zeros(N, dtype=torch.int32, device=idx.device)
        return z, z.clone(), torch.zeros(1, dtype=torch.int32, device=idx.device), False
    starts = torch.full((N,), A, dtype=torch.long, device=idx.device).scatter_reduce_(0, iv, pv, reduce="amin", include_self=True)
    ends = torch.full((N,), -1, dtype=torch.long, device=idx.device).scatter_reduce_(0, iv, pv, reduce="amax", include_self=True) + 1
    counts = (ends - starts).clamp_(min=0)
    starts = torch.where(counts > 0, starts, torch.zeros_like(starts))
    maxc = counts.max().reshape(1).to(torch.int32) if N > 0 else torch.zeros(1, dtype=torch.int32, device=idx.device)
    return starts.to(torch.int32), counts.to(torch.int32), maxc, True


def seg_reduce(feat, starts, counts, maxc, atom_mask=None, mean=True, eps=1e-9, out=None, out_dtype=None, BT=16, BC=128, num_warps=4):
    """feat [S, A, C] (unit channel stride; free sample / atom strides), starts / counts int32 [N], maxc int32 [1] (device), atom_mask [A]
    (unit stride, any float dtype) or None -> out [S, N, C] (out_dtype, default feat's): per-token masked mean (mean=True) or sum over each
    token's atom run, fp32 accumulation."""
    S, A, C = feat.shape
    assert feat.stride(-1) == 1 or C == 1, feat.stride()
    N = int(starts.shape[0])
    if out is None:
        out = torch.empty((S, N, C), device=feat.device, dtype=out_dtype or feat.dtype)
    if N == 0 or S == 0 or C == 0:
        return out
    assert out.stride(-1) == 1 or C == 1
    if atom_mask is not None:
        assert atom_mask.dim() == 1 and atom_mask.shape[0] == A and atom_mask.stride(0) == 1
    mk = atom_mask if atom_mask is not None else starts
    grid = (triton.cdiv(N, BT), S, triton.cdiv(C, BC))
    _seg_reduce_kernel[grid](feat, mk, starts, counts, maxc, out, N, C, A,
                             feat.stride(0), feat.stride(1), out.stride(0), out.stride(1), float(eps),
                             BT=BT, BC=BC, MEAN=bool(mean), HAS_MASK=atom_mask is not None, num_warps=num_warps)
    return out
