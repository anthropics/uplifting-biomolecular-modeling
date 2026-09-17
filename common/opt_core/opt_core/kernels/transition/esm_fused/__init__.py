"""Row ``esm_fused_exact``: the ESM-family pair Transition's FUSED INFERENCE statement in ONE Triton kernel, bit for bit (exact class).

The statement (the ESM-family ``transformers`` fork's ``Transition`` under no-grad, bf16 rows, unchunked; C = 256, hidden H = 4 C):

    mean, rstd = per-row statistics as the fork's ``_ln_stats_kernel`` stores them (bf16 tree mean over 256 lanes-pairs, fp32 tree variance on
                 (x - mean)^2, rstd = 1/sqrt(var + eps); both rounded to bf16)
    x_hat      = ((x - mean) * rstd) * ln_w + ln_b                       stepwise IN bf16 (four bf16 roundings; ln_w / ln_b bf16)
    a | b      = x_hat @ W12[:, :H] | x_hat @ W12[:, H:]                 bf16 x bf16, fp32 accumulate over K = 256 in four ascending 64-chunks
    h          = bf16( silu(a) * b )                                     fp32 sigmoid / products, ONE bf16 rounding
    y          = bf16( h @ W3^T )                                        cuBLAS bf16 GEMM, fp32 accumulate over K = H ascending (torch.addmm's product)
    out        = bf16( x + y )                                           the addmm residual: the product is rounded to bf16 FIRST, then added (two roundings)

This kernel: one program per BM-row tile of x [M, 256] (bf16, contiguous).  Prologue = the statistics tree and the bf16 x_hat chain of the
ESM-family kits' own Triton parts (functions ``_ld_chunk``, ``_bfly``, ``_warp_tree``, ``_stock_tree_sum``, ``_stock_stats`` of the inference
kit's ef2_pair_v2.py and ``_xhat_chunk`` of its ef2_w4.py, carried below byte for byte: their association is fixed by code, not by the launch
layout, so they are bitwise at any BM / warp count).  Main loop over H in BH-chunks (software-pipelined weight loads): the a|b columns of the
chunk as ONE dot chain on the column-interleaved packing [Wa|Wb] (column 2i = a_i, 2i+1 = b_i: each output element is its own K chain, so the
packing does not change a bit), h = bf16(silu(a)*b), acc += h @ W3T-chunk (fp32, chained: the K association cuBLAS's non-split kernels use at
these shapes on the 9.0 class -- measured, see TRANSITION_CELLS.json rows.esm_fused_exact).  Epilogue ROUND-THEN-ADD: out = bf16(x + bf16(acc)).
Nothing H-wide touches HBM (the statement writes and re-reads a [M, H] bf16 hidden and the two bf16 statistics vectors).

Envelope: C = 256 (four resident 64-wide K chunks), H % (2*BH) == 0, bf16 rows, LayerNorm with weight and bias, cc 9.0 launch row below
(227 KB shared-memory class).  Forward only.  Deterministic (no atomics, no split-K).  ``out`` may alias ``x`` (each program reads its rows before
it stores them; rows are disjoint across programs).  Serves under torch.inference_mode and any autocast state (no torch op on the data path).

Host entry points: ``pack_weights(W)`` (once per Weights: [Wa|Wb] interleaved bf16 [256, 2H], W3T bf16 [H, 256], bf16 LayerNorm vectors),
``select_cfg(device)`` (the launch row for the device or RuntimeError with the reason), ``fused_transition(x2d, pack, residual=, out=, cfg=)``.
"""
import torch
import triton
import triton.language as tl

C_IN = 256
BK = 64
# launch rows: (name, predicate on (cc major, shared memory opt-in bytes), cfg), first match serves.  Footprint = x_hat 4 x [BM, 64] bf16 staged for
# the MMA (64 KB at BM 128) + NSTAGE x (W12 chunk 32 KB + W3T chunk 16 KB) at BH 32: 208 KB at 3 stages (the sm_90 row; H100 SXM opts in 227 KB);
# the sm_80 row keeps 64 rows resident and one weight stage (80 KB: two programs per SM on A100, the fastest of 24 tilings measured there).  The
# tiling does not enter the per-element operation order (every tiling measured gives identical bits); EPI does: the statement's library GEMM adds
# the residual after rounding the product on sm_90 and before rounding it on sm_80 (measured on each card), so each row names its epilogue.
ROWS = (
    ("sm90_bm128_bh32_w8_s3", lambda major, smem: major == 9 and smem >= 220 * 1024, {"BM": 128, "BH": 32, "num_warps": 8, "NSTAGE": 3}),
    ("sm80_bm64_bh32_w4_s1", lambda major, smem: major == 8 and smem >= 96 * 1024, {"BM": 64, "BH": 32, "num_warps": 4, "NSTAGE": 1, "EPI": 1}),
)


# ----------------------------------------------------------------------------------------------------------------- carried parts (byte for byte)
# From the ESM-family inference kit's ef2_pair_v2.py (lever t6s: the fork's LayerNorm statistics from a row-block program, bitwise by
# construction of the association; TREE 0 = permute + split + add, no layout-dependent reduction) and ef2_w4.py (lever T6: the fork's bf16
# x_hat chain).  Do not edit: tests compare these function texts with the kit files.
@triton.jit
def _ld_chunk(X, offs_m, mrow, kk: tl.constexpr, C_: tl.constexpr, BK: tl.constexpr):
    offs_k = kk + tl.arange(0, BK)
    return tl.load(X + offs_m[:, None] * C_ + offs_k[None, :], mask=mrow[:, None], other=0.0)


@triton.jit
def _bfly(v, HALF: tl.constexpr, TREE: tl.constexpr):
    """one butterfly level: v [BM, 2*HALF] indexed by lane-residue r = h*HALF + r2 -> [BM, HALF]: v[r2] + v[r2 + HALF]  (one add per pair;
    TREE 0: permute+split+add, TREE 1: size-2-axis tl.sum — same pairing, different data movement)."""
    if TREE == 1:
        return tl.sum(tl.reshape(v, [v.shape[0], 2, HALF]), axis=1)
    else:
        a, b = tl.split(tl.permute(tl.reshape(v, [v.shape[0], 2, HALF]), (0, 2, 1)))
        return a + b


@triton.jit
def _warp_tree(xc, TREE: tl.constexpr):
    """xc [BM, 64] (one warp's 64 elements of the stock layout: lane l holds elements 2l, 2l+1) -> [BM]: in-thread pair add then butterfly
    xor 16, 8, 4, 2, 1 — in xc's dtype (bf16 for the mean, fp32 for the variance)."""
    if TREE == 1:
        p = tl.sum(tl.reshape(xc, [xc.shape[0], 32, 2]), axis=2)
    else:
        e0, e1 = tl.split(tl.reshape(xc, [xc.shape[0], 32, 2]))
        p = e0 + e1                               # [BM, 32] by lane
    p = _bfly(p, 16, TREE); p = _bfly(p, 8, TREE); p = _bfly(p, 4, TREE); p = _bfly(p, 2, TREE)
    if TREE == 1:
        return tl.sum(p, axis=1)                  # [BM, 2] -> lanes 0,1: one add
    else:
        a, b = tl.split(p)
        return a + b


@triton.jit
def _stock_tree_sum(c0, c1, c2, c3, TREE: tl.constexpr):
    """the stock _ln_stats reduction of a 256-row split as 4 warp chunks: ((w0 + w2) + (w1 + w3))."""
    w0 = _warp_tree(c0, TREE); w1 = _warp_tree(c1, TREE); w2 = _warp_tree(c2, TREE); w3 = _warp_tree(c3, TREE)
    return (w0 + w2) + (w1 + w3)


@triton.jit
def _stock_stats(x0, x1, x2, x3, eps, TREE: tl.constexpr):
    """x0..x3: bf16 [BM,64] chunks of the row -> (mean_bf16, rstd_bf16) exactly as the stock kernel stores them (all-resident variant)."""
    s = _stock_tree_sum(x0, x1, x2, x3, TREE)                   # bf16 tree sum
    mean = s.to(tl.float32) / 256
    d0 = x0.to(tl.float32) - mean[:, None]; d1 = x1.to(tl.float32) - mean[:, None]
    d2 = x2.to(tl.float32) - mean[:, None]; d3 = x3.to(tl.float32) - mean[:, None]
    v = _stock_tree_sum(d0 * d0, d1 * d1, d2 * d2, d3 * d3, TREE)   # fp32 tree sum
    var = v / 256
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean.to(tl.bfloat16), rstd.to(tl.bfloat16)


@triton.jit
def _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, kk: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, HAS_LN_BIAS: tl.constexpr):
    offs_k = kk + tl.arange(0, BLOCK_SIZE_K)
    x = tl.load(X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk), mask=(offs_m[:, None] < M), other=0.0)
    ln_w = tl.load(LN_W_ptr + offs_k)
    if HAS_LN_BIAS:
        ln_b = tl.load(LN_B_ptr + offs_k)
        x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    else:
        x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :]
    return x_hat
# ----------------------------------------------------------------------------------------------------------------- end of carried parts


@triton.jit
def _esm_fused_exact_kernel(X, W12I, W3T, LNW, LNB, Out, M, eps,
                            BM: tl.constexpr, BH: tl.constexpr, C_: tl.constexpr, H: tl.constexpr, NSTAGE: tl.constexpr,
                            RESIDUAL: tl.constexpr, EPI: tl.constexpr, INTERLEAVED: tl.constexpr):
    """grid = (cdiv(M, BM),).  X / Out: [M, 256] bf16 contiguous (Out may alias X).  W12I: [256, 2H] bf16, column 2i = Wa column i, 2i+1 = Wb
    column i (INTERLEAVED 1) or [Wa | Wb] blocked (INTERLEAVED 0).  W3T: [H, 256] bf16.  LNW / LNB: [256] bf16.  EPI 0: out = bf16(x + bf16(acc))
    (the product rounded, then the residual added: the statement's addmm as its sm_90 GEMM rounds); EPI 1: out = bf16(fp32(x) + acc) (the residual
    added to the fp32 accumulator, one rounding: as its sm_80 GEMM rounds); the launch row names which."""
    BK: tl.constexpr = 64
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    mrow = offs_m < M
    # statistics: the fork's tree, bf16 mean / fp32 variance, both rounded to bf16 (as its stats kernel stores them)
    x0 = _ld_chunk(X, offs_m, mrow, 0, C_, BK); x1 = _ld_chunk(X, offs_m, mrow, 64, C_, BK)
    x2 = _ld_chunk(X, offs_m, mrow, 128, C_, BK); x3 = _ld_chunk(X, offs_m, mrow, 192, C_, BK)
    mean, rstd = _stock_stats(x0, x1, x2, x3, eps, 0)
    # x_hat: the fork's bf16 chain per 64-wide K chunk (kept on chip for the whole hidden loop)
    xh0 = _xhat_chunk(X, LNW, LNB, mean, rstd, offs_m, M, C_, 1, 0, BK, True)
    xh1 = _xhat_chunk(X, LNW, LNB, mean, rstd, offs_m, M, C_, 1, 64, BK, True)
    xh2 = _xhat_chunk(X, LNW, LNB, mean, rstd, offs_m, M, C_, 1, 128, BK, True)
    xh3 = _xhat_chunk(X, LNW, LNB, mean, rstd, offs_m, M, C_, 1, 192, BK, True)
    offs_k = tl.arange(0, BK)
    offs_c = tl.arange(0, C_)
    offs_h = tl.arange(0, BH)
    acc = tl.zeros((BM, C_), dtype=tl.float32)
    if INTERLEAVED:
        offs_2h = tl.arange(0, 2 * BH)
        for j in tl.range(0, H // BH, num_stages=NSTAGE):
            col = 2 * (j * BH) + offs_2h
            w0 = tl.load(W12I + (offs_k[:, None] + 0) * (2 * H) + col[None, :])
            w1 = tl.load(W12I + (offs_k[:, None] + 64) * (2 * H) + col[None, :])
            w2 = tl.load(W12I + (offs_k[:, None] + 128) * (2 * H) + col[None, :])
            w3 = tl.load(W12I + (offs_k[:, None] + 192) * (2 * H) + col[None, :])
            ab = tl.zeros((BM, 2 * BH), dtype=tl.float32)
            ab = tl.dot(xh0, w0, ab)
            ab = tl.dot(xh1, w1, ab)
            ab = tl.dot(xh2, w2, ab)
            ab = tl.dot(xh3, w3, ab)
            a_acc, b_acc = tl.split(tl.reshape(ab, [BM, BH, 2]))
            sig = tl.sigmoid(a_acc)
            silu_a = a_acc * sig
            swiglu = silu_a * b_acc
            h = swiglu.to(tl.bfloat16)
            rows_h = j * BH + offs_h
            w3t = tl.load(W3T + rows_h[:, None] * C_ + offs_c[None, :])
            acc = tl.dot(h, w3t, acc)
    else:
        for j in tl.range(0, H // BH, num_stages=NSTAGE):
            cola = j * BH + offs_h
            colb = H + j * BH + offs_h
            wa0 = tl.load(W12I + (offs_k[:, None] + 0) * (2 * H) + cola[None, :]); wb0 = tl.load(W12I + (offs_k[:, None] + 0) * (2 * H) + colb[None, :])
            wa1 = tl.load(W12I + (offs_k[:, None] + 64) * (2 * H) + cola[None, :]); wb1 = tl.load(W12I + (offs_k[:, None] + 64) * (2 * H) + colb[None, :])
            wa2 = tl.load(W12I + (offs_k[:, None] + 128) * (2 * H) + cola[None, :]); wb2 = tl.load(W12I + (offs_k[:, None] + 128) * (2 * H) + colb[None, :])
            wa3 = tl.load(W12I + (offs_k[:, None] + 192) * (2 * H) + cola[None, :]); wb3 = tl.load(W12I + (offs_k[:, None] + 192) * (2 * H) + colb[None, :])
            a_acc = tl.zeros((BM, BH), dtype=tl.float32)
            b_acc = tl.zeros((BM, BH), dtype=tl.float32)
            a_acc = tl.dot(xh0, wa0, a_acc); b_acc = tl.dot(xh0, wb0, b_acc)
            a_acc = tl.dot(xh1, wa1, a_acc); b_acc = tl.dot(xh1, wb1, b_acc)
            a_acc = tl.dot(xh2, wa2, a_acc); b_acc = tl.dot(xh2, wb2, b_acc)
            a_acc = tl.dot(xh3, wa3, a_acc); b_acc = tl.dot(xh3, wb3, b_acc)
            sig = tl.sigmoid(a_acc)
            silu_a = a_acc * sig
            swiglu = silu_a * b_acc
            h = swiglu.to(tl.bfloat16)
            rows_h = j * BH + offs_h
            w3t = tl.load(W3T + rows_h[:, None] * C_ + offs_c[None, :])
            acc = tl.dot(h, w3t, acc)
    optr = Out + offs_m[:, None] * C_ + offs_c[None, :]
    if RESIDUAL:
        xres = tl.load(X + offs_m[:, None] * C_ + offs_c[None, :], mask=mrow[:, None], other=0.0)
        if EPI == 0:
            y = acc.to(tl.bfloat16)
            o = xres + y                                              # bf16 + bf16: one rounding of the exact sum (== torch's opmath add)
        else:
            o = (xres.to(tl.float32) + acc).to(tl.bfloat16)
        tl.store(optr, o, mask=mrow[:, None])
    else:
        tl.store(optr, acc.to(tl.bfloat16), mask=mrow[:, None])


# ----------------------------------------------------------------------------------------------------------------- host side
def smem_optin(device=None):
    """Opt-in shared memory per block (bytes) of the device."""
    props = torch.cuda.get_device_properties(device)
    v = getattr(props, "shared_memory_per_block_optin", None)
    if not v:
        v = 232448 if props.major == 9 else (166912 if (props.major, props.minor) == (8, 0) else 101376)
    return int(v)


def select_cfg(device=None):
    """-> (cfg dict, row name) for the device, or RuntimeError naming why no launch row applies (the face turns it into a refusal by name)."""
    props = torch.cuda.get_device_properties(device)
    smem = smem_optin(device)
    for name, pred, cfg in ROWS:
        if pred(props.major, smem):
            return dict(cfg), name
    raise RuntimeError("no launch row for cc %d.%d with %d B shared memory (rows: %s)" % (props.major, props.minor, smem, ", ".join(r[0] for r in ROWS)))


def pack_weights(W, interleaved=True):
    """Kernel operands from the provider's canonical Weights (bf16 copies made once): W12I [256, 2H] (interleaved a|b columns, or blocked),
    W3T [H, 256], LayerNorm weight / bias as bf16 [256] (the statement's parameter dtype under bf16 autocast)."""
    wa, wb, wo = W.wa16, W.wb16, W.wo16                    # [H, C], [H, C], [C, H]
    H, C = int(wa.shape[0]), int(wa.shape[1])
    if interleaved:
        w12i = torch.stack((wa.t(), wb.t()), dim=2).reshape(C, 2 * H).contiguous()      # col 2i = Wa^T[:, i], 2i+1 = Wb^T[:, i]
    else:
        w12i = torch.cat((wa.t(), wb.t()), dim=1).contiguous()
    return {"W12I": w12i, "W3T": wo.t().contiguous(), "ln_w16": W.ln_w.to(torch.bfloat16).contiguous(), "ln_b16": W.ln_b.to(torch.bfloat16).contiguous(),
            "H": H, "C": C, "eps": float(W.eps), "interleaved": bool(interleaved)}


def admits_shape(c, hidden, bh=32):
    return int(c) == C_IN and int(hidden) > 0 and int(hidden) % (2 * int(bh)) == 0


def fused_transition(x2d, pack, residual=True, out=None, cfg=None, epi=None):
    """x2d [M, 256] bf16 contiguous -> out [M, 256] bf16 (= x + Transition(x) when residual, else Transition(x) without the residual).  ``epi``: the
    residual epilogue -- 0 = round the product to bf16, then add x (bf16 + bf16 -> bf16); 1 = add x to the fp32 accumulator, round once; None = the
    launch row's EPI (the statement's library GEMM rounds differently per card: measured, recorded per row)."""
    if cfg is None:
        cfg, _ = select_cfg(x2d.device)
    if epi is None:
        epi = cfg.get("EPI", 0)
    M, C = int(x2d.shape[0]), int(x2d.shape[1])
    H = pack["H"]
    if C != C_IN or pack["C"] != C_IN or not admits_shape(C, H, cfg["BH"]):
        raise RuntimeError("shape outside the kernel's envelope: c=%d hidden=%d (c must be 256, hidden a multiple of %d)" % (C, H, 2 * cfg["BH"]))
    if x2d.dtype != torch.bfloat16 or not x2d.is_contiguous():
        raise RuntimeError("x must be bf16 and contiguous")
    if out is None:
        out = torch.empty_like(x2d)
    if M == 0:
        return out
    grid = (triton.cdiv(M, cfg["BM"]),)
    _esm_fused_exact_kernel[grid](x2d, pack["W12I"], pack["W3T"], pack["ln_w16"], pack["ln_b16"], out, M, pack["eps"],
                                  BM=cfg["BM"], BH=cfg["BH"], C_=C_IN, H=H, NSTAGE=cfg["NSTAGE"], RESIDUAL=bool(residual), EPI=int(epi),
                                  INTERLEAVED=bool(pack["interleaved"]), num_warps=cfg["num_warps"])
    return out
