"""ef2_pair_v2 — pair-trunk Transition levers for the ESMFold2 kit.  The Transition (LN -> w12 -> silu(x1)*x2 -> w3 -> +residual,
C = 256, H = 1024) is the single largest op of the trunk: 22 per recycle x 20 recycles (+ lm_encoder / parcae_coda / confidence trunk) and one per
MSA-encoder block.  Five independent levers, all runtime patches (nothing under stock/ is edited):

  FAST tier (tolerance):
    t15     C.Transition.forward -> ONE Triton kernel per call: in-kernel fp32 LayerNorm statistics (x tile held as 4 bf16 [128,64] chunks, x_hat
            rounded once to bf16), packed a|b GEMM (interleaved [Wa|Wb] columns -> one N=64 wgmma chain per hidden chunk, A read once), h =
            bf16(silu(a)*b), fp32 accumulation of h @ W3^T over ascending hidden chunks, out = bf16(fp32(x) + acc).  Replaces the kit's fast path
            (W4 t10: stock _ln_stats launch + row-block kernel + addmm) — 2 launches and one |pair| HBM round trip fewer.
    t15msa  the MSA-encoder blocks' `pair + PairTransition(pair)` (left on the reference ATen/cuBLAS path by every kit mode) through the same
            kernel: the block's bound forward runs with pair_transition swapped for a zero shim, then the fused transition+residual is applied.
  EXACT tier (bitwise == stock under the deterministic recipe):
    t6s     [guarded: a bitwise self-check vs the stock kernel at install (synthetic rows drawn from a private generator: the process RNG is not
            touched) and again on the first real eager call; on mismatch disengaged BY NAME for the process, exact keeps the stock statistics]
            the stock _ln_stats kernel (one 256-thread program per ROW) replaced by a coalesced row-block kernel that reproduces its
            reduction tree bit for bit (bf16 warp-butterfly mean: in-thread pair, xor 16/8/4/2/1, (w0+w2)+(w1+w3); fp32 tree variance;
            1/sqrt; bf16 stores) at ~3x its bandwidth; W4's T6 row-block kernel and the addmm are launched exactly as W4 launches them.
    t6i     the residual addmm done in place: torch.addmm(x, hidden, w3t) first copies the pair into a fresh output and then runs the beta=1
            GEMM on it; x_2d aliases the pair (contiguous) and the pair is dead after Transition.forward in every caller (PairUpdateBlock /
            FoldingTrunk / MSA block reassign it), so x_2d.addmm_(hidden, w3t) is the identical cuBLAS call (same m,n,k, dtypes, strides,
            beta; cuBLAS GEMM permits C == D) minus the |pair|-sized memcpy.

    xtr     the MSA module's PairTransition modules (Full model: pair_transition, C = 256 / H = 1024, and msa_transition, C = 128 / H = 512),
            which every other exact-tier lever leaves on the reference statements (fp32 LayerNorm under autocast -> Linear w12 -> chunk ->
            silu(x1) * x2 -> Linear w3; the block adds the residual), served through the shared core's transition provider
            (opt_core.kernels.transition, core >= 0.5.34) BY WORD with the module's own LayerNorm output given (the LayerNorm statement is
            unchanged): C = 256 -> 'flash_sm90a' (the sealed sm_90a TMA/wgmma transition kernel; refused BY NAME on a stack without its
            prebuilt -> the next word) then 'v1' (fpf_transition: LN-given Triton GEMM chain, fp32 accumulate, the statements' bf16 rounding
            points); C = 128 -> 'v1' — on the 9.0 class (XTR_WORDS is keyed by class: a class whose rows are not measured bitwise has no word
            and the lever steps aside by name there).  The words are resolved once at install on an eager probe call (a refusal is recorded by
            name and the next word serves); packs are built at install (nothing is packed inside a graph capture).  Both rows reproduce the reference
            statements AND cuBLAS's bf16 results bit for bit at these shapes on the 9.0 class (measured: C=256 every N in 20..1536, C=128
            rows 20k..1.47M) -> exact class; a call outside the measured row window [XTR_ROWS] runs the stock statements,
            counted (xtr_outside_window), never silently substituted.

    t15 / t15msa bind the shared core's transition provider BY WORD (T15_BIND_WORD = 'esm_t15': this file's kernel carried into the core;
            bitwise identical and speed-equal to the kit-local launch, measured); a core older than the row or a refusal on this
            stack keeps the kit-local kernel with the reason on the LEVER line (t15 `bind=`); EF2_T15_BIND=module is the comparison arm.

install(model, trunk=, msa=, t6s=, t6i=, xtr=) AFTER ef2_opt.install (M1 rebinds the MSA block forwards) and ef2_msa_v2 (its block forward calls
self.pair_transition, which t15msa wraps) and before graphs are captured; uninstall() restores.  Inference only: grad enabled / CPU / non-bf16 / C != 256 (xtr: C not in XTR_WORDS) / chunked Transition fall through to the previous forward.  No static
activation buffers (outputs are allocated per call; caches hold weight packs keyed by parameter identity+version).  License Apache-2.0.
"""
import os
import types, collections
import torch
import triton
import triton.language as tl

import transformers.models.esmfold2.modeling_esmfold2_common as C
import transformers.models.esmfold2.modeling_esmfold2 as MOD

VERSION = "pair_v2.1.7"
STATS = collections.Counter()
_BF16 = torch.bfloat16
_STATE = dict(installed=False, orig_transition_forward=None, msa_blocks=[], cfg=None, row=None, t15=False, t15msa=False, t6s=False, t6i=False, t6s_selfcheck=None,
              xtr=False, orig_pair_transition_forward=None, xtr_words={}, xtr_modules=0,
              t15_bind=None, t15_bind_reason=None, t15_bind_core=None, t15_face=None, t15_face_refused=collections.Counter())
T15_BIND_WORD = "esm_t15"             # levers t15 / t15msa serve through the shared core's transition provider face BY WORD (opt_core.kernels.transition row esm_t15 =
T15_BIND_ENV = "EF2_T15_BIND"         # this file's Triton kernel carried into the core byte for byte); resolved once at install on an 8-row probe; a core older than the row /
                                      # a refusal on this stack keeps the kit-local kernel BY NAME (t15_bind=kit_module:<reason>); EF2_T15_BIND=module = the comparison arm
XTR_WORDS = {"9.0": {256: ("flash_sm90a", "v1"), 128: ("v1",)}}       # lever xtr: provider row words per compute-capability class and PairTransition width, in preference order (a refused word ->
                                                                       # the next, by name); a class without an entry (8.0: the rows are not measured bitwise against this engine's statements there —
                                                                       # the core's cc 8.0 cells record no bitwise transition row) has no word: the lever steps aside BY NAME on it (refused 'class')
XTR_ROWS = {256: (1, 2_359_296), 128: (1, 1_474_560)}                # lever xtr: the row windows [lo, hi] measured bitwise vs the statements on the 9.0 class (C=256: N 1..1536 squared; C=128: 1440 tokens x 1024 rows); outside -> the stock statements, counted

# ---------------------------------------------------------------------------------------------------------------------------------------------
# T15 kernel: one program per 128-row tile of the [M, 256] pair.  Prologue: x as 4 bf16 [BM,64] chunks, fp32 two-pass statistics, x_hat per
# chunk (one bf16 rounding).  Hidden loop (H/BH chunks, software-pipelined weight loads): ab = x_hat @ [Wa|Wb]_chunk as 4 chained K=64 dots on
# the interleaved packing (col 2i = x1_i, 2i+1 = x2_i) -> split -> h = bf16(silu(a)*b) -> acc += h @ W3T_chunk.  Epilogue: out = bf16(x + acc).
# ---------------------------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _ld_chunk(X, offs_m, mrow, kk: tl.constexpr, C_: tl.constexpr, BK: tl.constexpr):
    offs_k = kk + tl.arange(0, BK)
    return tl.load(X + offs_m[:, None] * C_ + offs_k[None, :], mask=mrow[:, None], other=0.0)


@triton.jit
def _xhat_c(xc_bf, mean, rstd, LNW, LNB, kk: tl.constexpr, BK: tl.constexpr):
    offs_k = kk + tl.arange(0, BK)
    lnw = tl.load(LNW + offs_k)
    lnb = tl.load(LNB + offs_k)
    return ((xc_bf.to(tl.float32) - mean[:, None]) * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)


@triton.jit
def _t15_kernel(X, W12I, W3T, LNW, LNB, Out, M, eps,
                BM: tl.constexpr, BH: tl.constexpr, C_: tl.constexpr, H: tl.constexpr, NSTAGE: tl.constexpr):
    BK: tl.constexpr = 64
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    mrow = offs_m < M
    x0 = _ld_chunk(X, offs_m, mrow, 0, C_, BK); x1 = _ld_chunk(X, offs_m, mrow, 64, C_, BK)
    x2 = _ld_chunk(X, offs_m, mrow, 128, C_, BK); x3 = _ld_chunk(X, offs_m, mrow, 192, C_, BK)
    s = tl.sum(x0.to(tl.float32), axis=1) + tl.sum(x1.to(tl.float32), axis=1) + tl.sum(x2.to(tl.float32), axis=1) + tl.sum(x3.to(tl.float32), axis=1)
    mean = s / C_
    d0 = x0.to(tl.float32) - mean[:, None]; d1 = x1.to(tl.float32) - mean[:, None]; d2 = x2.to(tl.float32) - mean[:, None]; d3 = x3.to(tl.float32) - mean[:, None]
    var = (tl.sum(d0 * d0, axis=1) + tl.sum(d1 * d1, axis=1) + tl.sum(d2 * d2, axis=1) + tl.sum(d3 * d3, axis=1)) / C_
    rstd = tl.rsqrt(var + eps)
    xh0 = _xhat_c(x0, mean, rstd, LNW, LNB, 0, BK); xh1 = _xhat_c(x1, mean, rstd, LNW, LNB, 64, BK)
    xh2 = _xhat_c(x2, mean, rstd, LNW, LNB, 128, BK); xh3 = _xhat_c(x3, mean, rstd, LNW, LNB, 192, BK)
    offs_k = tl.arange(0, BK)
    offs_c = tl.arange(0, C_)
    offs_h = tl.arange(0, BH)
    acc = tl.zeros((BM, C_), dtype=tl.float32)
    offs_2h = tl.arange(0, 2 * BH)
    for j in tl.range(0, H // BH, num_stages=NSTAGE):
        col0 = j * BH
        col = 2 * col0 + offs_2h
        wp = W12I + offs_k[:, None] * (2 * H) + col[None, :]
        ab = tl.dot(xh0, tl.load(wp))
        ab = tl.dot(xh1, tl.load(wp + 64 * (2 * H)), ab)
        ab = tl.dot(xh2, tl.load(wp + 128 * (2 * H)), ab)
        ab = tl.dot(xh3, tl.load(wp + 192 * (2 * H)), ab)
        a, b = tl.split(tl.reshape(ab, [BM, BH, 2]))
        h = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        w3 = tl.load(W3T + (col0 + offs_h)[:, None] * C_ + offs_c[None, :])
        acc = tl.dot(h, w3, acc)
    xres = tl.load(X + offs_m[:, None] * C_ + offs_c[None, :], mask=mrow[:, None], other=0.0).to(tl.float32)
    out = xres + acc
    tl.store(Out + offs_m[:, None] * C_ + offs_c[None, :], out.to(tl.bfloat16), mask=mrow[:, None])


T15_CFG = dict(BM=128, BH=32, num_warps=8, NSTAGE=3)          # the sm90 row (default for direct transition_v2 calls)

# Tile rows, keyed on CAPABILITY (probed opt-in shared memory per block + minimum compute capability for the code path), never on an arch
# allow-list: an untested arch that meets a row's capability gets that row, named `arch=untested:<cc>` on the line, and the kit's probe launch /
# first call decides "cannot run" — the engagement rule every kit lever follows.  Footprints are architecture dependent (sm90: x_hat 64 KB
# staged for wgmma + NSTAGE x 48 KB TMA/async weight ring; sm8x: cp.async ring of (NSTAGE-1) x 48 KB + layout-conversion scratch):
#   row sm90: BM128/BH32/8w/NSTAGE3 — needs cc >= 9.0 and >= 217088 B opt-in smem (212992 B used on sm90; 248-255 regs, 0 spills).
#   row sm80: BM64/BH32/4w/NSTAGE3 — needs cc >= 8.0 and >= 139264 B (135168 B used on sm80: cp.async-pipelined, mma.sync m16n8k16, 254 regs,
#             0 spills; 180224 B when compiled for sm90, where it is slower than t10 -> sm90 parts take the sm90 row; the sm90 tile shape does not
#             fit sm80's shared memory with NSTAGE3).
#   below 139264 B (the 99 KB sm86/sm89/sm120 classes): no row -> RuntimeError naming the lever (the class lever there stays t10/t1).
T15_ROWS = (
    dict(name="sm90", min_cc=(9, 0), min_smem_optin=217088, cfg=dict(BM=128, BH=32, num_warps=8, NSTAGE=3), tested_cc=((9, 0),)),
    dict(name="sm80", min_cc=(8, 0), min_smem_optin=139264, cfg=dict(BM=64, BH=32, num_warps=4, NSTAGE=3), tested_cc=((8, 0), (9, 0))),
)


def select_t15_cfg(device=None):
    """-> (cfg dict, info dict(t15_row, arch='tested:<cc>'|'untested:<cc>', cc, smem_optin, row_min_smem)) for the current CUDA device;
    raises RuntimeError naming the lever when the device's opt-in shared memory is below every row (cannot run)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
    cc = (int(props.major), int(props.minor))
    smem = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    for row in T15_ROWS:
        if cc >= row["min_cc"] and smem >= row["min_smem_optin"]:
            arch = ("tested:" if cc in row["tested_cc"] else "untested:") + f"{cc[0]}.{cc[1]}"
            return dict(row["cfg"]), dict(t15_row=row["name"], arch=arch, cc=f"{cc[0]}.{cc[1]}", smem_optin=smem, row_min_smem=row["min_smem_optin"], device=props.name)
    raise RuntimeError(f"ef2_pair_v2 t15: cannot run on {props.name} (sm{cc[0]}{cc[1]}, {smem} B opt-in shared memory per block): every T15 row needs "
                       f">= {min(r['min_smem_optin'] for r in T15_ROWS)} B — the transition lever for this class stays t10/t1")


def transition_v2(x2d, pack, cfg=None, out=None):
    """x2d [M,256] bf16 contiguous -> x + Transition(x) [M,256] bf16 (lever t15).  pack = pack_transition(norm, ffn)."""
    cfg = dict(T15_CFG, **(cfg or {}))
    M, K = x2d.shape
    assert K == 256 and x2d.is_contiguous() and x2d.dtype == _BF16
    H = pack["H"]
    assert H % cfg["BH"] == 0
    if out is None:
        out = torch.empty_like(x2d)
    _t15_kernel[(triton.cdiv(M, cfg["BM"]),)](x2d, pack["W12I"], pack["W3T"], pack["ln_w"], pack["ln_b"], out, M, 1e-5,
                                             BM=cfg["BM"], BH=cfg["BH"], C_=256, H=H, NSTAGE=cfg["NSTAGE"], num_warps=cfg["num_warps"])
    STATS["t15_calls"] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------
# T6S: the stock LayerNorm statistics, bit for bit, from a coalesced row-block kernel (lever t6s, exact tier).
# The stock `_ln_stats_kernel` (transformers .../kernels/fused_lnlin_swiglu.py) runs one 256-thread program per row: each lane holds 2
# adjacent bf16 elements; the mean is a bf16 tree (in-thread pair, warp butterfly xor 16/8/4/2/1, then (w0+w2)+(w1+w3) across the 4 warps)
# extended to fp32 and divided by 256; the variance is the same tree in fp32 on (x - mean)^2; rstd = 1/sqrt(var+eps); both stored bf16.
# _stock_tree_sum reproduces that association exactly (fp add is commutative, so only the pairing matters).  Bitwise identity vs the stock
# kernel is checked in every process, not assumed (t6s_selfcheck: at install on synthetic rows, again on the first real eager call).
# ---------------------------------------------------------------------------------------------------------------------------------------------
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
def _t6s_stats_kernel(X, Mean, Rstd, M, eps, BM: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    mrow = offs_m < M
    x0 = _ld_chunk(X, offs_m, mrow, 0, 256, 64); x1 = _ld_chunk(X, offs_m, mrow, 64, 256, 64)
    x2 = _ld_chunk(X, offs_m, mrow, 128, 256, 64); x3 = _ld_chunk(X, offs_m, mrow, 192, 256, 64)
    mean, rstd = _stock_stats(x0, x1, x2, x3, eps, 0)
    tl.store(Mean + offs_m, mean, mask=mrow)
    tl.store(Rstd + offs_m, rstd, mask=mrow)


T6S_CFG = dict(BM=64, num_warps=4)


def t6s_stats(x_2d, Mean=None, Rstd=None):
    """(Mean bf16 [M], Rstd bf16 [M]) bitwise == the stock _ln_stats_kernel's, ~3x its bandwidth."""
    M, K = x_2d.shape
    assert K == 256 and x_2d.is_contiguous() and x_2d.dtype == _BF16
    if Mean is None:
        Mean = torch.empty((M,), dtype=_BF16, device=x_2d.device); Rstd = torch.empty((M,), dtype=_BF16, device=x_2d.device)
    _t6s_stats_kernel[(triton.cdiv(M, T6S_CFG["BM"]),)](x_2d, Mean, Rstd, M, 1e-5, BM=T6S_CFG["BM"], num_warps=T6S_CFG["num_warps"])
    return Mean, Rstd


def stock_stats(x_2d):
    """the STOCK _ln_stats_kernel's (Mean, Rstd) bf16 [M], launched exactly as the stock host code (and W4's T6 path) launches it."""
    import transformers.models.esmfold2.kernels.fused_lnlin_swiglu as SW
    M, K = x_2d.shape
    Mean = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device); Rstd = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    block, num_warps = SW._ln_stats_settings(K)
    SW._ln_stats_kernel[(M,)](x_2d, x_2d.stride(0), Mean, Mean.stride(0), Rstd, Rstd.stride(0), K, 1e-5, BLOCK_SIZE=block, num_warps=num_warps)
    return Mean, Rstd


def t6s_selfcheck(x_2d, origin="first_call"):
    """Bitwise GUARD of lever t6s: the stock _ln_stats kernel and t6s_stats on the same rows, torch.equal on Mean and Rstd.  t6s's exactness is a
    property of how the installed Triton lowers the STOCK kernel's reductions, so a Triton or architecture change is caught here instead of
    silently turning `exact` into a tolerance tier: on mismatch t6s is DISENGAGED BY NAME for the process (state=skipped
    reason=guard:off:bitwise_check_failed:triton<ver>) and the exact forward keeps the stock statistics kernel.  Runs at install (``origin="install"``:
    synthetic rows, see selfcheck_rows) and on the first real eager Transition input (``"first_call"``); a later check never re-engages a
    disengaged lever.  Returns True (identical) / False (disengaged)."""
    sc = _STATE.get("t6s_selfcheck")
    if sc is not None and (origin == "install" or sc.get("first_call_done") or not sc["identical"]):
        return sc["identical"]
    ms, rs = stock_stats(x_2d)
    mt, rt = t6s_stats(x_2d)
    identical = bool(torch.equal(ms, mt)) and bool(torch.equal(rs, rt))
    ver = getattr(triton, "__version__", "?")
    rows = int(x_2d.shape[0]) + int((sc or {}).get("rows", 0))
    _STATE["t6s_selfcheck"] = dict(identical=identical and bool((sc or {}).get("identical", True)), rows=rows, triton=ver, origin=origin,
                                   first_call_done=(origin != "install") or bool((sc or {}).get("first_call_done")),
                                   word=("identical" if identical else f"bitwise_check_failed:triton{ver}"))
    STATS["t6s_selfcheck_identical" if identical else "t6s_selfcheck_failed"] += 1
    if not identical:
        _STATE["t6s"] = False                                    # disengaged by name for the rest of the process
    return identical


SELFCHECK_ROWS = 4096                                            # rows of the install-time check (x 256 bf16: 2 MiB)
SELFCHECK_SEED = 20260910                                        # the private generator's seed (the process's default generators are never touched)


def selfcheck_rows(device=None):
    """Synthetic [SELFCHECK_ROWS, 256] bf16 rows for the install-time t6s check, drawn from a PRIVATE torch.Generator (torch.randn(generator=)):
    six value regimes a pair activation takes (unit normal, large offset, tiny scale, wide scale, one-hot-like spikes, constant rows) so the
    bf16 mean tree and the fp32 variance tree are exercised where rounding differs."""
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    g = torch.Generator(device=dev); g.manual_seed(SELFCHECK_SEED)
    n = SELFCHECK_ROWS // 6
    x = torch.randn((SELFCHECK_ROWS, 256), generator=g, device=dev, dtype=torch.float32)
    x[n:2 * n] += 37.0; x[2 * n:3 * n] *= 1e-3; x[3 * n:4 * n] *= 300.0
    x[4 * n:5 * n] = torch.where(x[4 * n:5 * n] > 2.2, x[4 * n:5 * n] * 50.0, x[4 * n:5 * n] * 0.01); x[5 * n:] = 0.5
    return x.to(_BF16).contiguous()


def t6s_hidden(x_2d, W12, LN_W, LN_B):
    """= ef2_w4._lnlin_swiglu_fwd_rowblock (hidden [M, N] bf16 = silu(x1)*x2 of LN(x); the caller runs the addmm) with the stock _ln_stats
    launch replaced by t6s_stats; W4's T6 kernel is launched exactly as W4 launches it."""
    import ef2_w4
    M, K = x_2d.shape; K2, two_N = W12.shape; N = two_N // 2
    assert x_2d.is_contiguous() and K == K2 == 256 and two_N % 2 == 0
    out = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
    Mean, Rstd = t6s_stats(x_2d)
    t6 = ef2_w4._t6_cfg()
    grid = (triton.cdiv(M, t6["BLOCK_SIZE_M"]), t6["N_SPLIT"])
    ef2_w4._lnlin_swiglu_fwd_rowblock_kernel[grid](
        x_2d, W12, LN_W, LN_B if LN_B is not None else LN_W, out, Mean, Rstd, M, N,
        x_2d.stride(0), x_2d.stride(1), W12.stride(0), W12.stride(1), out.stride(0), out.stride(1),
        HAS_LN_BIAS=(LN_B is not None), BLOCK_SIZE_M=t6["BLOCK_SIZE_M"], BLOCK_SIZE_N=t6["BLOCK_SIZE_N"], BLOCK_SIZE_K=64, N_SPLIT=t6["N_SPLIT"],
        num_stages=t6["num_stages"], num_warps=t6["num_warps"])
    STATS["t6s_split_calls"] += 1
    return out


def _transition_forward_t6s(self, x):
    """C.Transition.forward for the EXACT tier (levers t6s / t6i): W4's T6+T5 path statement for statement (bf16-cast cached weights, stock
    statistics, row-block kernel, then the residual addmm under the caller's autocast); t6s swaps the stock _ln_stats launch for t6s_stats
    (bitwise), t6i runs the addmm in place (identical cuBLAS call, no |pair| copy)."""
    import ef2_w4
    if not ((_STATE.get("t6s") or _STATE.get("t6i")) and x.is_cuda and x.dtype == _BF16 and x.shape[-1] == 256 and not torch.is_grad_enabled() and self._can_use_fused_path(x)):
        return _STATE["orig_transition_forward"](self, x)
    ac = torch.is_autocast_enabled("cuda")
    if not (ac and torch.get_autocast_dtype("cuda") == _BF16):
        return _STATE["orig_transition_forward"](self, x)
    fused = self._fused_swiglu
    cache = self.__dict__.setdefault("_pair_v2_t6s_wcache", {})
    key = ef2_w4._pkey(fused.W12, fused.LN_W, fused.LN_B, self.ffn.w3.weight)
    if cache.get("key") != key:
        cache.update(key=key, W12=ef2_w4._bf16c(fused.W12.detach()), LN_W=ef2_w4._bf16c(fused.LN_W.detach()),
                     LN_B=(ef2_w4._bf16c(fused.LN_B.detach()) if fused.LN_B is not None else None), w3t=self.ffn.w3.weight.detach().t().to(_BF16))
        STATS["t6s_cache_fill"] += 1
    x_shape = x.shape
    with torch.amp.autocast("cuda", enabled=False):
        x_2d = x.contiguous().view(-1, x_shape[-1])
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            use_t6s = bool(_STATE.get("t6s"))
            sc = _STATE.get("t6s_selfcheck") or {}
            if use_t6s and not sc.get("first_call_done"):
                # first-real-call bitwise guard (the install-time check ran on synthetic rows); a capturing stream cannot host it (torch.equal
                # syncs), so a call under capture before it ran uses the stock statistics (exact either way) and the check runs at the next eager call
                use_t6s = (not torch.cuda.is_current_stream_capturing()) and t6s_selfcheck(x_2d, origin="first_call")
            if use_t6s:
                hidden = t6s_hidden(x_2d, cache["W12"], cache["LN_W"], cache["LN_B"])
            else:                               # t6i alone / t6s unchecked or disengaged: W4's T6 hidden exactly as W4 computes it (stock _ln_stats + row-block kernel)
                if _STATE.get("t6s") and not sc.get("first_call_done"):
                    STATS["t6s_unchecked_stock_calls"] += 1
                hidden = ef2_w4._lnlin_swiglu_fwd_rowblock(x_2d, cache["W12"], cache["LN_W"], cache["LN_B"])
        else:                                   # chunked callers: same statements per dim-1 slice as W4 (kit modes run unchunked)
            return _STATE["orig_transition_forward"](self, x)
    if _STATE.get("t6i") and x_2d.data_ptr() == x.data_ptr():
        # lever t6i (exact tier): x is contiguous so x_2d aliases the pair -> accumulate the GEMM straight into it: the same cuBLAS bf16 GEMM
        # (beta=1, C == D) that torch.addmm runs after first memcpy-ing x into a fresh output; the pair is dead after Transition in every caller
        # (PairUpdateBlock / FoldingTrunk reassign it), so the in-place update is unobservable except for the missing |pair|-sized copy.
        out = x_2d.addmm_(hidden, cache["w3t"])
        STATS["t6i_inplace"] += 1
    else:
        out = torch.addmm(x_2d, hidden, cache["w3t"])      # == W4 addmm_res / stock _addmm_residual under the ambient autocast (w3t already bf16: the cast is a no-op)
    return out.view(x_shape)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# weight packs (cached per parameter identity+version, W4-T5-style key)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _pkey(*params):
    return tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params if p is not None)


def pack_transition(norm, ffn):
    """norm: nn.LayerNorm(256); ffn: SwiGLUMLP (w12: Linear 256->2H packed [x1|x2], w3: Linear H->256).  bf16 weights = the values the stock
    fused / autocast paths consume (fp32 masters rounded to bf16); LN affine kept fp32 (stock rounds them to bf16)."""
    H = ffn.hidden_features
    W12T = ffn.w12.weight.detach().to(_BF16).t().contiguous()          # [256, 2H]: cols 0..H-1 -> x1 (silu branch), H..2H-1 -> x2
    W12I = torch.stack([W12T[:, :H], W12T[:, H:]], dim=2).reshape(W12T.shape[0], 2 * H).contiguous()   # interleaved: col 2i = x1_i, 2i+1 = x2_i
    W3T = ffn.w3.weight.detach().to(_BF16).t().contiguous()            # [H, 256]
    ln_w = norm.weight.detach().float().contiguous()
    ln_b = (norm.bias.detach().float().contiguous() if norm.bias is not None else torch.zeros_like(ln_w))
    return dict(W12I=W12I, W3T=W3T, ln_w=ln_w, ln_b=ln_b, H=H)


def _t15_face_pack(mod):
    """opt_core.kernels.transition Weights of a module for the esm_t15 row (w12 fused rows [0,H) = the silu branch), cached on the module by parameter identity + version."""
    cache = mod.__dict__.setdefault("_pair_v2_t15_face_pack", {})
    key = _pkey(mod.norm.weight, mod.norm.bias, mod.ffn.w12.weight, mod.ffn.w3.weight)
    if cache.get("key") != key:
        TR = _STATE["t15_face"]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            W = TR.pack(w_ab=mod.ffn.w12.weight.detach(), w_o=mod.ffn.w3.weight.detach(), ln_w=mod.norm.weight.detach(),
                        ln_b=(mod.norm.bias.detach() if mod.norm.bias is not None else torch.zeros_like(mod.norm.weight)), eps=mod.norm.eps, device=mod.ffn.w12.weight.device)
        cache.clear(); cache["key"] = key; cache["pack"] = W; STATS["t15_face_pack"] += 1
    return cache["pack"]


def _t15_bind_resolve(mod=None):
    """Resolve ONCE how t15 / t15msa serve in this process: the core's provider face by word (T15_BIND_WORD) when it imports, names the row and serves an
    8-row probe with this module's weights on this device; else the kit-local kernel with the reason by name.  EF2_T15_BIND=module forces the kit-local kernel."""
    if _STATE["t15_bind"] is not None:
        return _STATE["t15_bind"]
    want = os.environ.get(T15_BIND_ENV, "face").strip().lower()
    if want not in ("face", "module"):
        raise ValueError(f"ef2_pair_v2: {T15_BIND_ENV}={want!r} (face | module)")
    try:
        import opt_core; _STATE["t15_bind_core"] = getattr(opt_core, "__version__", "?")
    except ImportError:
        _STATE["t15_bind_core"] = None
    if want == "module":
        _STATE["t15_bind"], _STATE["t15_bind_reason"] = "kit_module", f"env:{T15_BIND_ENV}=module"
        return _STATE["t15_bind"]
    try:
        from opt_core.kernels import transition as TR
        if T15_BIND_WORD not in getattr(TR, "ROW_NAMES", ()):
            raise ImportError(f"row {T15_BIND_WORD} not in opt_core {_STATE['t15_bind_core']} kernels.transition")
    except ImportError as e:
        _STATE["t15_bind"], _STATE["t15_bind_reason"] = "kit_module", "core:" + str(e)[:80]
        return _STATE["t15_bind"]
    if mod is None:
        return None                                                     # resolved on the first eligible call instead (no module to probe with)
    _STATE["t15_face"] = TR
    try:
        W = _t15_face_pack(mod)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            TR.transition(torch.zeros((8, 256), dtype=_BF16, device=mod.ffn.w12.weight.device), W, word=T15_BIND_WORD, residual=True)
    except TR.Refusal as r:
        _STATE["t15_face"] = None; _STATE["t15_bind"], _STATE["t15_bind_reason"] = "kit_module", "refused:" + str(getattr(r, "kind", r))[:80]
        return _STATE["t15_bind"]
    except Exception as e:
        _STATE["t15_face"] = None; _STATE["t15_bind"], _STATE["t15_bind_reason"] = "kit_module", f"probe:{type(e).__name__}:{str(e)[:60]}"
        return _STATE["t15_bind"]
    _STATE["t15_bind"], _STATE["t15_bind_reason"] = "core_face", None
    return _STATE["t15_bind"]


def _t15_serve(mod, x2d):
    """x2d [M,256] bf16 -> x + Transition(x): through the core face by word when bound (a per-call refusal is named, counted, and the kit-local kernel serves)."""
    if _STATE["t15_bind"] is None:
        _t15_bind_resolve(mod)
    if _STATE["t15_bind"] == "core_face":
        TR = _STATE["t15_face"]
        try:
            with torch.amp.autocast("cuda", enabled=False):
                y, _sel = TR.transition(x2d, _t15_face_pack(mod), word=T15_BIND_WORD, residual=True)
            STATS["t15_face_calls"] += 1
            return y
        except TR.Refusal as r:
            STATS["t15_face_refused"] += 1; _STATE["t15_face_refused"][str(getattr(r, "kind", "refusal"))[:60]] += 1
    pack = _pack_for(mod)
    with torch.amp.autocast("cuda", enabled=False):
        return transition_v2(x2d, pack, _STATE["cfg"])


def t15_bind_words():
    """LEVER-line evidence of the binding: {'bind': 'core_face:esm_t15' | 'kit_module:<reason>', 'core': <opt_core version>} ({} before it is resolved)."""
    b = _STATE.get("t15_bind")
    if not b:
        return {}
    d = dict(bind=(f"core_face:{T15_BIND_WORD}" if b == "core_face" else f"kit_module:{_STATE.get('t15_bind_reason') or 'unresolved'}"), core=str(_STATE.get("t15_bind_core") or "none"))
    if _STATE["t15_face_refused"]:
        d["face_refused"] = ",".join(f"{k}={v}" for k, v in sorted(_STATE["t15_face_refused"].items()))
    return d


def _pack_for(mod):
    cache = mod.__dict__.setdefault("_pair_v2_pack", {})
    key = _pkey(mod.norm.weight, mod.norm.bias, mod.ffn.w12.weight, mod.ffn.w3.weight)
    if cache.get("key") != key:
        cache.clear(); cache["key"] = key; cache["pack"] = pack_transition(mod.norm, mod.ffn); STATS["t15_pack"] += 1
    return cache["pack"]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# patched forwards
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _eligible(x):
    return (_STATE["installed"] and x.is_cuda and x.dtype == _BF16 and x.shape[-1] == 256 and not torch.is_grad_enabled())


def _transition_forward_v2(self, x):
    """C.Transition.forward: x + ffn(norm(x)) in one kernel when eligible (bf16 CUDA pair, fused backend), else the previous forward."""
    if not (_eligible(x) and self._can_use_fused_path(x)):
        return _STATE["orig_transition_forward"](self, x)
    x2d = x.contiguous().view(-1, 256)
    out = _t15_serve(self, x2d)
    return out.view(x.shape)


def _pair_transition_residual_v2(pt, pair):
    """pair + PairTransition(pair) for MOD.PairTransition(256) (MSA encoder block) through the fused kernel; else the stock statements."""
    if not (_eligible(pair) and isinstance(pt, MOD.PairTransition) and pt.norm.normalized_shape[0] == 256):
        return pair + pt(pair)
    x2d = pair.contiguous().view(-1, 256)
    out = _t15_serve(pt, x2d)
    STATS["t15_msa_pair_transition_calls"] += 1
    return out.view(pair.shape)


def _msa_block_forward_v2(self, m, pair, msa_attention_mask, pair_attention_mask):
    """MSAEncoderBlock.forward (lever t15msa): the forward bound before install() (ef2_opt's M1 forward, or the class forward) runs with
    pair_transition swapped for a zero-returning shim, so its final `pair = pair + self.pair_transition(pair)` is a no-op add; the fused
    transition + residual is then applied to its result.  Every other statement of the block (OPM, PWA, msa_transition, TriMul out/in) is
    executed by that previous forward unchanged."""
    prev = self._pair_v2_prev_forward
    # Run the previous forward with pair_transition replaced by a zero-delta shim, then apply the fused transition+residual ourselves.
    pt = self.pair_transition
    self.pair_transition = _ZERO_SHIM
    try:
        m, pair = prev(m, pair, msa_attention_mask, pair_attention_mask)
    finally:
        self.pair_transition = pt
    pair = _pair_transition_residual_v2(pt, pair)
    return m, pair


def _xtr_pack(pt):
    """opt_core.kernels.transition Weights for a MOD.PairTransition (w12 rows [0,H) = the silu branch a, [H,2H) = b), cached on the module by parameter identity+version."""
    from opt_core.kernels import transition as TR
    cache = pt.__dict__.setdefault("_pair_v2_xtr_pack", {})
    key = _pkey(pt.norm.weight, pt.norm.bias, pt.ffn.w12.weight, pt.ffn.w3.weight)
    if cache.get("key") != key:
        H = pt.ffn.hidden_features
        w12 = pt.ffn.w12.weight.detach()
        cache.clear(); cache["key"] = key
        cache["pack"] = TR.pack(w_a=w12[:H], w_b=w12[H:], w_o=pt.ffn.w3.weight.detach(), ln_w=pt.norm.weight.detach(),
                                ln_b=(pt.norm.bias.detach() if pt.norm.bias is not None else torch.zeros_like(pt.norm.weight)), eps=pt.norm.eps, device=w12.device)
        STATS["xtr_pack"] += 1
    return cache["pack"]


def _cc_word():
    """'9.0' / '8.0' … of the current device (XTR_WORDS key); '' without CUDA."""
    return ("%d.%d" % tuple(torch.cuda.get_device_capability())) if torch.cuda.is_available() else ""


def _xtr_resolve(pt):
    """Resolve the provider word for this PairTransition's width ONCE (eager, at install): the first word of XTR_WORDS[C] the provider serves on a
    probe call of 8 rows; every refusal is recorded by name (xtr_words[C]['refused']).  None when no word serves (the lever steps aside for that width)."""
    from opt_core.kernels import transition as TR
    c = int(pt.norm.normalized_shape[0])
    ent = _STATE["xtr_words"].get(c)
    if ent is not None:
        return ent
    ent = dict(word=None, row=None, refused={}, bias=False)
    words = XTR_WORDS.get(_cc_word(), {}).get(c, ())
    if not words:
        ent["refused"]["class"] = "no_measured_word_cc%s_c%d" % (_cc_word().replace(".", ""), c)   # by name: this class / width has no bitwise-measured row
        _STATE["xtr_words"][c] = ent
        return ent
    if getattr(pt.ffn.w12, "bias", None) is not None or getattr(pt.ffn.w3, "bias", None) is not None:
        ent["bias"] = True                                            # a biased projection is not the provider's op: the statements run (named)
        _STATE["xtr_words"][c] = ent
        return ent
    W = _xtr_pack(pt)
    dev = pt.ffn.w12.weight.device
    x = torch.zeros((1, 8, c), dtype=_BF16, device=dev)
    n = torch.zeros((1, 8, c), dtype=_BF16, device=dev)
    for word in words:
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                _, sel = TR.transition(x, W, word=word, residual=False, x_ln=n)
        except TR.Refusal as r:
            ent["refused"][word] = r.kind
            continue
        ent["word"] = word; ent["row"] = sel.row + ((":" + sel.variant) if sel.variant else "")
        break
    _STATE["xtr_words"][c] = ent
    return ent


def _pair_transition_forward_xtr(self, x):
    """MOD.PairTransition.forward = ffn(norm(x)) (lever xtr): the module's own LayerNorm statement, then the provider row resolved for this width
    with the LayerNorm output given (residual=False: the MSA block adds it, unchanged); the previous forward when not eligible (grad / CPU /
    non-bf16 / chunked / a width without a serving word / rows outside the measured window), counted by name."""
    prev = _STATE["orig_pair_transition_forward"]
    c = int(x.shape[-1])
    ent = _STATE["xtr_words"].get(c)
    if ent is None or ent.get("word") is None or not (_STATE["xtr"] and x.is_cuda and x.dtype == _BF16 and not torch.is_grad_enabled()) \
            or (self._chunk_size is not None and x.shape[1] > self._chunk_size):
        STATS["xtr_fallthrough"] += 1
        return prev(self, x)
    rows = x.numel() // c
    lo, hi = XTR_ROWS.get(c, (1, 0))
    if not (lo <= rows <= hi):
        STATS["xtr_outside_window"] += 1
        return prev(self, x)
    from opt_core.kernels import transition as TR
    n = self.norm(x)                                                  # the stock statement (fp32 LayerNorm under the caller's autocast), unchanged
    if n.dtype != _BF16:
        n = n.to(_BF16)                                               # == the cast autocast applies at ffn.w12's input
    try:
        with torch.amp.autocast("cuda", enabled=False):
            y, sel = TR.transition(x, _xtr_pack(self), word=ent["word"], residual=False, x_ln=n)
    except TR.Refusal as r:                                          # resolved words do not refuse at these shapes; a refusal here is named and the statements run
        STATS["xtr_refused"] += 1; ent["refused"][ent["word"] + "@call"] = r.kind
        return prev(self, x)
    STATS["xtr_calls"] += 1; STATS["xtr_c%d_calls" % c] += 1
    return y


class _ZeroShim(torch.nn.Module):
    """pair_transition stand-in: returns an additive zero so `pair + self.pair_transition(pair)` leaves pair unchanged (one cheap broadcast add)."""
    def forward(self, x):
        STATS["t15_msa_shim_calls"] += 1
        return torch.zeros((), dtype=x.dtype, device=x.device)


_ZERO_SHIM = _ZeroShim()


def install(model=None, trunk=True, msa=True, t6s=False, t6i=False, xtr=False):
    """Levers (independent): trunk (t15) -> process-wide fused Transition kernel on C.Transition (every FoldingTrunk: folding_trunk, lm_encoder,
    parcae_coda, confidence trunk) [fast tier]; msa (t15msa) -> model.msa_encoder's blocks' PairTransition + residual through the same kernel
    [fast tier; a model without msa_encoder has nothing to wrap: msa_blocks=0, the lever is not for that variant]; t6s / t6i -> exact-tier
    Transition forward on C.Transition (used when trunk is off: t15 supersedes them).  Call after ef2_opt.install (M1 rebinds the MSA-block
    forwards; graphs are captured at the first fold, after this).  t6s runs its bitwise self-check here (t6s_selfcheck origin=install).  Idempotent."""
    import ef2_srcguard                                          # the exact forward / the MSA shim re-issue C.Transition.forward / PairTransition: refuse by name on another upstream source
    ef2_srcguard.check_many({"t15": trunk, "t15msa": msa, "t6s": t6s, "t6i": t6i, "xtr": xtr})
    if xtr and msa:
        raise ValueError("ef2_pair_v2.install: xtr with t15msa — one owner of the MSA blocks' PairTransition per process (xtr is the exact line's, t15msa / t16 the fast line's)")
    if trunk or msa:
        base, row = select_t15_cfg()                            # raises BY NAME when the device cannot run any T15 row
        _STATE["cfg"] = dict(base); _STATE["row"] = row
    else:
        _STATE["cfg"] = dict(T15_CFG); _STATE["row"] = None
    if t6s:
        t6s_selfcheck(selfcheck_rows(), origin="install")       # bitwise guard at install: identical -> engaged; else disengaged by name (t6s_selfcheck word)
    sc = _STATE.get("t6s_selfcheck")
    _STATE["t6s"] = bool(t6s) and not (sc is not None and not sc["identical"])     # a failed self-check keeps t6s off for the process (named in stats/describe)
    _STATE["t6i"] = bool(t6i)
    if trunk or t6s or t6i:
        if _STATE["orig_transition_forward"] is None:
            _STATE["orig_transition_forward"] = C.Transition.forward
        C.Transition.forward = _transition_forward_v2 if trunk else _transition_forward_t6s
    _STATE["installed"] = True
    n_msa = 0
    enc = getattr(model, "msa_encoder", None) if model is not None else None
    if msa and enc is not None:
        for blk in enc.blocks:
            if getattr(blk, "_pair_v2_prev_forward", None) is None:
                blk._pair_v2_prev_forward = blk.forward          # bound: ef2_opt's M1 forward if installed, else the class forward
                blk.forward = types.MethodType(_msa_block_forward_v2, blk)
                _STATE["msa_blocks"].append(blk)
                n_msa += 1
    _STATE["t15"] = bool(trunk); _STATE["t15msa"] = bool(msa) and n_msa > 0
    if (trunk or _STATE["t15msa"]) and model is not None and _STATE["t15_bind"] is None:   # the t15 binding resolved now (eager, before any graph capture) on the first servable module
        for mod in model.modules():
            ffn, norm = getattr(mod, "ffn", None), getattr(mod, "norm", None)
            if (isinstance(mod, (C.Transition, MOD.PairTransition)) and isinstance(norm, torch.nn.LayerNorm) and tuple(norm.normalized_shape) == (256,)
                    and getattr(ffn, "hidden_features", None) == 1024 and getattr(ffn.w12, "bias", None) is None and ffn.w12.weight.is_cuda):
                _t15_bind_resolve(mod)
                break
    n_xtr = 0
    if xtr and enc is not None:                                       # lever xtr: MOD.PairTransition class patch; words resolved + packs built now (eager, before any graph capture)
        for mod in enc.modules():
            if isinstance(mod, MOD.PairTransition) and int(mod.norm.normalized_shape[0]) in (256, 128):
                if _xtr_resolve(mod).get("word") is not None:
                    _xtr_pack(mod); n_xtr += 1
        if n_xtr and _STATE["orig_pair_transition_forward"] is None:
            _STATE["orig_pair_transition_forward"] = MOD.PairTransition.forward
            MOD.PairTransition.forward = _pair_transition_forward_xtr
    _STATE["xtr"] = bool(xtr) and n_xtr > 0; _STATE["xtr_modules"] = n_xtr
    row = _STATE.get("row") or {}
    return dict(version=VERSION, t15_row=row.get("t15_row"), arch=row.get("arch"), smem_optin=row.get("smem_optin"), device=row.get("device"), cfg=_STATE["cfg"],
                transition="C.Transition.forward", msa_blocks=n_msa, t6s=_STATE["t6s"], t6i=_STATE["t6i"],
                t6s_selfcheck=(sc or {}).get("word"), xtr=_STATE["xtr"], xtr_modules=n_xtr, xtr_words=xtr_words(), t15_bind=t15_bind_words())              # LEVER-line fields: t15_row=sm90|sm80 arch=tested:|untested:<cc> smem_optin=<B> bitwise_check=<word>


def uninstall(model=None):
    if _STATE["orig_transition_forward"] is not None:
        C.Transition.forward = _STATE["orig_transition_forward"]
        _STATE["orig_transition_forward"] = None
    if _STATE["orig_pair_transition_forward"] is not None:
        MOD.PairTransition.forward = _STATE["orig_pair_transition_forward"]
        _STATE["orig_pair_transition_forward"] = None
    _STATE["t6s"] = _STATE["t6i"] = _STATE["t15"] = _STATE["t15msa"] = _STATE["xtr"] = False
    _STATE["xtr_words"].clear(); _STATE["xtr_modules"] = 0
    _STATE["t15_bind"] = _STATE["t15_bind_reason"] = _STATE["t15_face"] = None; _STATE["t15_face_refused"].clear()
    for blk in _STATE["msa_blocks"]:
        prev = blk.__dict__.pop("_pair_v2_prev_forward", None)
        if prev is not None:
            blk.forward = prev
    _STATE["msa_blocks"].clear()
    _STATE["installed"] = False
    return dict(version=VERSION, installed=False)


def stats():
    d = dict(STATS)
    if _STATE.get("t6s_selfcheck") is not None:
        d["t6s_selfcheck"] = dict(_STATE["t6s_selfcheck"])
    if _STATE.get("xtr_words"):
        d["xtr_words"] = xtr_words()
    return d


def xtr_words():
    """{C: {'word': served word|None, 'row': row[:variant]|None, 'refused': {word: refusal kind}}} as resolved at install (LEVER-line evidence of xtr)."""
    return {c: dict(word=e.get("word"), row=e.get("row"), refused=dict(e.get("refused", {})), bias=bool(e.get("bias"))) for c, e in _STATE.get("xtr_words", {}).items()}


def levers_on():
    """{lever name: bool} for the five levers as they stand in this process (t6s reads False after a failed self-check)."""
    return {k: bool(_STATE.get(k)) and bool(_STATE["installed"]) for k in ("t15", "t15msa", "t6s", "t6i", "xtr")}


def describe():
    return dict(version=VERSION, levers=dict(t15="fast: fused Transition kernel (C.Transition)", t15msa="fast: MSA-block PairTransition through t15",
                                              t6s="exact: bitwise stock LN statistics kernel", t6i="exact: in-place residual addmm",
                                              xtr="exact: MSA-module PairTransition through the core transition provider by word (LN given)"),
                installed=_STATE["installed"], on=levers_on(), xtr=_STATE.get("xtr"), xtr_modules=_STATE.get("xtr_modules"), xtr_words=xtr_words(), t15_row=_STATE.get("row"), t6s=_STATE.get("t6s"), t6s_selfcheck=_STATE.get("t6s_selfcheck"), t6i=_STATE.get("t6i"),
                msa_blocks=len(_STATE["msa_blocks"]), cfg=_STATE.get("cfg"),
                rows=[(r["name"], f"cc>={r['min_cc'][0]}.{r['min_cc'][1]}", r["min_smem_optin"], r["cfg"]) for r in T15_ROWS])
