"""ATOM_WINDOW v0.1 — sequence-local atom attention (AlphaFold3 Alg. 5/6/7/24: the atom-attention encoder / decoder blocks of the diffusion
sampler) as two fused Triton kernels, fp32 in / fp32 out.

    window_starts(A, n_real, NQ, NK)  the engines' SHIFTED key windows as one int32 start per query block (may be negative when fewer real atoms
                                      than one key window exist: those keys are invalid, as the engines mark them) — the rule of the engines'
                                      atom_attention_block_utils.get_block_indices, re-derived so the keys are read IN PLACE instead of gathered.
    ln_qkvg(a, cond, W)               per atom row: LayerNorm(a) (no affine), the query-side and key-side AdaLN modulations from PER-ATOM
                                      conditioning rows [A, C] (row r of the [S*A, C] activation <-> conditioning row r % A: never expanded over
                                      the samples), then four C x C projections in one pass -> qkvg [S, A, 4C] = [ q*D^-1/2 | k | v | sigmoid(g) ]
                                      (column slices; k / v once per ATOM — the engines compute the key-side AdaLN and the k / v projections on the
                                      gathered NK-key windows, i.e. ~NK/NQ times per atom).
    window_attn(qkvg, a, inv, W_o)    per (query block, sample): keys = atoms [ks[i], ks[i]+NK) read in place from qkvg, logits = q k^T + pair
                                      bias [NB, H, NQ, NK] with -inf where the key is outside [0, n_real) or masked, fp32 softmax, o = p v,
                                      o *= sigmoid gate, W_o (+ b_o), the AdaLN-Zero gate and the block's residual: returns a + gate * W_o(o) —
                                      the whole attention half of a diffusion-transformer atom block.
    warmup(C, H, NQ, NK, ...)         compiles both kernels for one module geometry on a dummy problem (a model does it at construction, outside
                                      any timed forward).

Precision (`precision=`): "tf32rn" (default) rounds every dot operand to the nearest TF32 value — weights once on the host (weight_tf32, memoised
per parameter), activations in-kernel (cvt.rna) — then runs a TF32 tensor-core dot: the numerics class of cuBLAS TF32 GEMMs under
torch.set_float32_matmul_precision("high"), which is what the engines' fp32 atom-attention island runs (a plain Triton "tf32" dot truncates the
operands on sm_90 and lands ~3x further from an fp64 reference); "tf32x3" (3-pass, ~fp32 accuracy, ~2.3x the kernel time); "tf32"; "ieee".
exp / sigmoid lower to Triton's ex2.approx-based forms and the softmax normalisation to div.full (a few ulp — below the TF32 operand rounding of
the dots, inside the `ieee` word's fp32 bound); no tanh.approx. Every index product that scales with S*A*C is formed in int64; the atom count and the mask's sample
stride are not value-specialised (one compile per kernel and precision serves every structure size).

Operand contract (all CUDA, fp32 unless noted): a [S, A, C] (any sample stride, unit stride in C); gq / lsq / gk / lsk / gate [A, C] contiguous
(sigmoid(linear_g(LN c)), linear_s(LN c) of the query- and key-side AdaLNs, sigmoid(linear_ada_out(c)) of the AdaLN-Zero output gate); bias
[NB, H, NQ, NK] (unit stride in NK; NB = ceil(A / NQ)); ks int32 [NB]; n_real int32 [1] (real atoms, equal across samples); atom_mask
[S or 1, A]; W_* the torch Linear parameters ([C_out, C_in] row-major, optional biases). Geometry served: C = H * D <= 128 (one row tile and
one C x C weight tile per projection live in shared memory) with D in {16, 32, 64}; NQ a multiple of 16 and NK a power of two >= NQ compile, (NQ, NK) = (32, 128) is the tested
pair the levers serve; callers refuse anything else by name.
"""
import math

import torch
import triton
import triton.language as tl

__version__ = "0.1"


def window_starts(n_atom_padded, n_real, n_query, n_key, device):
    """The engines' shifted key windows: the start atom index of every query block's key window, int32 [NB] (NB = ceil(A / n_query)). Window i
    is centred on its query block and shifted inward so it never leaves [0, n_real); with n_real < n_key the start is negative and the keys
    below 0 / at or above n_real are invalid (window_attn masks them, as the engines' invalid-index mask does). `n_real`: a number or a 0-d /
    1-element tensor (kept on device: no sync)."""
    nb = math.ceil(n_atom_padded / n_query)
    centers = n_query // 2 + torch.arange(nb, device=device) * n_query
    start0 = centers - n_key // 2
    underflow = torch.relu(-start0)
    overflow = torch.relu((start0 + n_key - 1) - (n_real - 1))
    shift = torch.where(underflow > 0, underflow, -overflow)
    return (start0 + shift).to(torch.int32)


PRECISIONS = {"tf32rn": ("tf32", True), "tf32": ("tf32", False), "tf32x3": ("tf32x3", False), "ieee": ("ieee", False)}   # name -> (tl.dot input_precision, round operands RN first)
_WR = {}


def weight_tf32(w):
    """A copy of W rounded to the nearest TF32 value (ties away, = cvt.rna.tf32.f32) so the kernels' TF32 dots need no per-tile rounding of the
    weight operand; memoised per parameter (re-made if the parameter is modified in place or reallocated)."""
    key = id(w)
    hit = _WR.get(key)
    if hit is not None and hit[0] is w and hit[1] == w._version:
        return hit[2]
    wi = w.detach().to(torch.float32).contiguous().view(torch.int32)
    wr = ((wi + 0x1000) & -0x2000).view(torch.float32)          # add half an ulp of the 10-bit mantissa, clear the 13 dropped bits
    _WR[key] = (w, w._version, wr)
    return wr


@triton.jit
def _rn(x, RN: tl.constexpr):
    """fp32 -> the nearest TF32 value (cvt.rna: 10 explicit mantissa bits, ties away), kept in an fp32 register. The tensor core's own fp32->tf32
    step truncates; feeding it pre-rounded operands makes a "tf32" dot round-to-nearest like cuBLAS / CUTLASS TF32 GEMMs (the engines' class)."""
    if RN:
        return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return x


@triton.jit(do_not_specialize=["R", "NATOM"])
def _ln_qkvg_kernel(A, OUT, GQ, LSQ, GK, LSK, WQ, WK, WV, WG, BQ, BK, BV, BG,
                    R, NATOM, stride_ar, stride_or, eps, qscale,
                    C: tl.constexpr, BLOCK_R: tl.constexpr, HAS_BQ: tl.constexpr, HAS_BK: tl.constexpr, HAS_BV: tl.constexpr, HAS_BG: tl.constexpr,
                    IP: tl.constexpr, RN: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)          # int64 row ids: every offset product below is 64-bit
    rmask = rows < R
    cols = tl.arange(0, C)
    r64 = rows
    a = tl.load(A + r64[:, None] * stride_ar + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(a, axis=1) / C
    d = a - mean[:, None]
    var = tl.sum(d * d, axis=1) / C
    ah = d * (1.0 / tl.sqrt(var + eps))[:, None]
    atom = rows % NATOM
    coff = atom[:, None] * C + cols[None, :]
    xq = _rn(tl.load(GQ + coff, mask=rmask[:, None], other=0.0) * ah + tl.load(LSQ + coff, mask=rmask[:, None], other=0.0), RN)
    xk = _rn(tl.load(GK + coff, mask=rmask[:, None], other=0.0) * ah + tl.load(LSK + coff, mask=rmask[:, None], other=0.0), RN)
    # torch Linear weight W is [C_out, C_in] row-major; x @ W^T = dot(x, WT) with WT[kk, n] = W[n * C + kk] (K-major B, as TF32 MMA wants);
    # the weights arrive pre-rounded to TF32 (weight_tf32) and q / k / v leave rounded to TF32 when RN (window_attn's dots consume them as is)
    kk = tl.arange(0, C)[:, None]
    nn = tl.arange(0, C)[None, :].to(tl.int64)
    obase = OUT + r64[:, None] * stride_or + cols[None, :]
    woff = nn * C + kk                                                  # [C_in, C_out] element offsets of W^T, shared by the four projections
    w = tl.load(WQ + woff)
    q = tl.dot(xq, w, input_precision=IP)
    if HAS_BQ:
        q = q + tl.load(BQ + cols)[None, :]
    tl.store(obase, _rn(q * qscale, RN), mask=rmask[:, None])
    w = tl.load(WK + woff)
    k = tl.dot(xk, w, input_precision=IP)
    if HAS_BK:
        k = k + tl.load(BK + cols)[None, :]
    tl.store(obase + C, _rn(k, RN), mask=rmask[:, None])
    w = tl.load(WV + woff)
    v = tl.dot(xk, w, input_precision=IP)
    if HAS_BV:
        v = v + tl.load(BV + cols)[None, :]
    tl.store(obase + 2 * C, _rn(v, RN), mask=rmask[:, None])
    w = tl.load(WG + woff)
    g = tl.dot(xq, w, input_precision=IP)
    if HAS_BG:
        g = g + tl.load(BG + cols)[None, :]
    tl.store(obase + 3 * C, tl.sigmoid(g), mask=rmask[:, None])


LN_QKVG_TILE_BY_CC = {"8.0": (32, 8)}      # compute capability -> (BLOCK_R rows, num_warps) of the one-pass TF32 ln_qkvg launch when the caller names no tile:
                                            # 8.0: 32 rows x 8 warps (the 128-row tile needs 196608 B of shared memory, over 8.0's 166912 B opt-in limit;
                                            # measured 32 rows 0.13 ms / 64 rows 0.35 ms / 16 rows 0.14 ms per launch at 3200 atoms x 5 samples, outputs
                                            # bitwise across row tiles); any other capability: the 9.0-swept 128 rows x 8 warps; tf32x3 keeps 16 rows everywhere
_CC_MEMO = {}


def _cc_of(device):
    key = (device.type, device.index)
    if key not in _CC_MEMO:
        _CC_MEMO[key] = ("%d.%d" % torch.cuda.get_device_capability(device)) if device.type == "cuda" else "cpu"
    return _CC_MEMO[key]


def ln_qkvg_tile(cc, precision="tf32rn", num_warps=8):
    """(BLOCK_R, num_warps) of the ln_qkvg launch for a capability word when the caller names no tile (LN_QKVG_TILE_BY_CC; the cell for cc 8.0)."""
    if precision == "tf32x3":
        return 16, num_warps                 # the 3-pass dot needs the small tile on every card
    row = LN_QKVG_TILE_BY_CC.get(str(cc))
    if row is not None:
        return int(row[0]), int(row[1])
    return 128, num_warps                    # H100 sweep at ~6.5k atoms x 5 samples: 128 rows x 8 warps for one-pass TF32


def ln_qkvg(a, gq, lsq, gk, lsk, lin_q, lin_k, lin_v, lin_g, eps, qscale, out=None, precision="tf32rn", BLOCK_R=None, num_warps=8, num_stages=None):
    """a [S, A, C] fp32 (any sample stride, rows contiguous in C); conditioning tensors [A, C] fp32 contiguous; lin_* = the torch Linear modules
    (weight [C, C], bias or None).  Returns qkvg [S, A, 4C] fp32 = [ (W_q x_q + b_q) * qscale | W_k x_k + b_k | W_v x_k + b_v | sigmoid(W_g x_q + b_g) ]."""
    S, A_, C = a.shape
    assert a.stride(-1) == 1 and gq.shape == (A_, C)
    if BLOCK_R is None:
        BLOCK_R, num_warps = ln_qkvg_tile(_cc_of(a.device), precision, num_warps)
    a2 = a.reshape(S * A_, C) if a.is_contiguous() else a.contiguous().view(S * A_, C)
    if out is None:
        out = torch.empty((S, A_, 4 * C), device=a.device, dtype=torch.float32)
    R = S * A_
    grid = (triton.cdiv(R, BLOCK_R),)
    ws = [weight_tf32(l.weight) if PRECISIONS[precision][1] else l.weight for l in (lin_q, lin_k, lin_v, lin_g)]
    bs = [l.bias for l in (lin_q, lin_k, lin_v, lin_g)]
    _ln_qkvg_kernel[grid](a2, out, gq, lsq, gk, lsk, *ws, *[b if b is not None else ws[0] for b in bs], R, A_, a2.stride(0), out.stride(1), eps, qscale,
                          C=C, BLOCK_R=BLOCK_R, HAS_BQ=bs[0] is not None, HAS_BK=bs[1] is not None, HAS_BV=bs[2] is not None, HAS_BG=bs[3] is not None,
                          IP=PRECISIONS[precision][0], RN=PRECISIONS[precision][1], num_warps=num_warps, **({"num_stages": num_stages} if num_stages else {}))
    return out


@triton.jit(do_not_specialize=["NATOM", "stride_ms"])
def _window_attn_kernel(QKVG, AIN, OUT, BIAS, KS, NREAL, AMASK, GATE, WO, BO, NATOM,
                        stride_qs, stride_qr, stride_as, stride_ar, stride_os, stride_or,
                        stride_bb, stride_bh, stride_bq, stride_ms,
                        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, INF: tl.constexpr, HAS_BO: tl.constexpr,
                        IP: tl.constexpr, RN: tl.constexpr):
    i64 = tl.program_id(0).to(tl.int64)                                # int64 block / sample / row / column ids: every offset product below is 64-bit
    s64 = tl.program_id(1).to(tl.int64)
    qr = tl.arange(0, NQ).to(tl.int64)
    kc = tl.arange(0, NK).to(tl.int64)
    dc = tl.arange(0, D).to(tl.int64)
    cc = tl.arange(0, C).to(tl.int64)
    q_rows = i64 * NQ + qr
    q_in = q_rows < NATOM
    ks = tl.load(KS + i64).to(tl.int64)
    n_real = tl.load(NREAL)
    k_idx = ks + kc
    k_in = (k_idx >= 0) & (k_idx < NATOM)
    am_k = tl.load(AMASK + s64 * stride_ms + k_idx, mask=k_in, other=0.0)
    k_valid = k_in & (k_idx < n_real) & (am_k > 0.5)
    am_q = tl.load(AMASK + s64 * stride_ms + q_rows, mask=q_in, other=0.0)
    pair_ok = (am_q[:, None] > 0.5) & k_valid[None, :]
    mask_bias = tl.where(pair_ok, 0.0, -INF)
    q64 = q_rows
    k64 = k_idx
    qbase = QKVG + s64 * stride_qs + q64[:, None] * stride_qr
    kbase = QKVG + s64 * stride_qs + k64[:, None] * stride_qr
    acc = tl.zeros((NQ, C), dtype=tl.float32)
    for h in tl.static_range(H):
        h64 = tl.cast(h, tl.int64)
        hd = h64 * D                                                                                # this head's first column
        qh = tl.load(qbase + hd + dc[None, :], mask=q_in[:, None], other=0.0)                      # q / k / v arrive TF32-rounded from ln_qkvg when RN
        kh = tl.load(kbase + (C + hd) + dc[None, :], mask=k_in[:, None], other=0.0)
        vh = tl.load(kbase + (2 * C + hd) + dc[None, :], mask=k_in[:, None], other=0.0)
        sc = tl.dot(qh, tl.trans(kh), input_precision=IP)
        b = tl.load(BIAS + i64 * stride_bb + h64 * stride_bh + qr[:, None] * stride_bq + kc[None, :])
        sc = (sc + mask_bias) + b
        m = tl.max(sc, axis=1)
        p = tl.exp(sc - m[:, None])
        l = tl.sum(p, axis=1)
        p = _rn(p / l[:, None], RN)
        oh = tl.dot(p, vh, input_precision=IP)
        gh = tl.load(qbase + (3 * C + hd) + dc[None, :], mask=q_in[:, None], other=0.0)
        oh = _rn(oh * gh, RN)
        woT = tl.load(WO + cc[None, :] * C + (hd + dc)[:, None])             # [D, C]: WoT[kk, n] = Wo[n, h*D + kk] (K-major B); Wo pre-rounded when RN
        acc += tl.dot(oh, woT, input_precision=IP)
    if HAS_BO:
        acc = acc + tl.load(BO + cc)[None, :]
    gate = tl.load(GATE + q64[:, None] * C + cc[None, :], mask=q_in[:, None], other=0.0)
    a_in = tl.load(AIN + s64 * stride_as + q64[:, None] * stride_ar + cc[None, :], mask=q_in[:, None], other=0.0)
    tl.store(OUT + s64 * stride_os + q64[:, None] * stride_or + cc[None, :], a_in + gate * acc, mask=q_in[:, None])


def window_attn(qkvg, a, bias, ks, n_real, atom_mask, gate, wo, bo, n_heads, n_query=32, n_key=128, inf=1e9, out=None, precision="tf32rn", num_warps=None, num_stages=None):
    """qkvg [S, A, 4C] (ln_qkvg at the same precision); a [S, A, C] fp32 residual input; bias [NB, H, NQ, NK] fp32 (k contiguous); ks [NB] int32
    key-window starts (may be negative); n_real int32 tensor [1]; atom_mask [S or 1, A] fp32; gate [A, C]; wo [C, C], bo [C] or None.
    Returns a + gate * (W_o o + b_o) [S, A, C]."""
    S, A_, C4 = qkvg.shape
    C = C4 // 4
    NB = bias.shape[0]
    assert bias.shape == (NB, n_heads, n_query, n_key) and bias.stride(-1) == 1 and NB * n_query >= A_
    assert a.shape == (S, A_, C) and a.stride(-1) == 1 and qkvg.stride(-1) == 1
    assert atom_mask.shape[-1] == A_ and atom_mask.dim() == 2
    if out is None:
        out = torch.empty((S, A_, C), device=a.device, dtype=torch.float32)
    if num_warps is None:
        num_warps = 8 if precision == "tf32x3" else 4
    rn = PRECISIONS[precision][1]
    grid = (NB, S)
    _window_attn_kernel[grid](qkvg, a, out, bias, ks, n_real, atom_mask, gate, weight_tf32(wo) if rn else wo, bo if bo is not None else wo, A_,
                              qkvg.stride(0), qkvg.stride(1), a.stride(0), a.stride(1), out.stride(0), out.stride(1),
                              bias.stride(0), bias.stride(1), bias.stride(2), atom_mask.stride(0) if atom_mask.shape[0] > 1 else 0,
                              H=n_heads, D=C // n_heads, C=C, NQ=n_query, NK=n_key, INF=inf, HAS_BO=bo is not None, IP=PRECISIONS[precision][0], RN=rn,
                              num_warps=num_warps, **({"num_stages": num_stages} if num_stages else {}))
    return out


def warmup(C, n_heads, n_query, n_key, inf, has_bias, precision="tf32rn", device="cuda"):
    """Compile (or load from Triton's cache) both kernels for this module geometry on a dummy 2-block problem. has_bias = (q, k, v, g, o) booleans."""
    import time
    t0 = time.perf_counter()
    S, A_ = 1, 2 * n_query
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn((S, A_, C), device=device, generator=g)
    cond = [torch.rand((A_, C), device=device, generator=g) for _ in range(5)]
    lin = [torch.nn.Linear(C, C, bias=b).to(device) for b in has_bias[:4]]
    qkvg = ln_qkvg(a, cond[0], cond[1], cond[2], cond[3], *lin, 1e-5, 1.0, precision=precision)
    NB = A_ // n_query
    bias = torch.zeros((NB, n_heads, n_query, n_key), device=device)
    ks = torch.zeros((NB,), dtype=torch.int32, device=device); n_real = torch.full((1,), A_, dtype=torch.int32, device=device)
    am = torch.ones((1, A_), device=device)
    wo = torch.nn.Linear(C, C, bias=has_bias[4]).to(device)
    window_attn(qkvg, a, bias, ks, n_real, am, cond[4], wo.weight, wo.bias, n_heads, n_query, n_key, inf, precision=precision)
    torch.cuda.synchronize()
    return time.perf_counter() - t0
