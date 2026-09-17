"""ef2_w4 — W4 inference-execution levers for the ESMFold2 pair trunk (Triton kernels + the TriMul provider binding), layered on top
of ef2_opt.  Nothing here changes model math, inputs, MSAs, loops,
steps, seeds or outputs; every lever is an execution change of an op whose stock implementation is Biohub's vendored 'fused'
Triton backend (Biohub/transformers@ef32577f, models/esmfold2/kernels/*).  Inference only (no grad); with grad enabled or on
CPU every patched entry point falls straight through to the stock function.

Levers (all OFF unless enabled through enable(...) or the EF2_W4 env; each is testable in isolation):

 W4-T1  transition_nolin   Pair Transition: inference-only clone of Biohub's _lnlin_swiglu_fwd_kernel that does NOT materialise the
                           [M, 2N] bf16 'lin' buffer (consumed only by backward; 2.0 GB per call at 705 tok).  Same tile config,
                           same K loop, same tl.dot accumulators, same fp32 SiLU, and the same bf16 casts torch.amp.custom_fwd applies
                           to the stock Function's inputs -> designed BITWISE (checked by the deterministic gate, not assumed).
 W4-T3  tiles              Re-tuned Triton tile configs (the built-in H100 table below or a json) for the three static-config
                           GEMM-class kernels; only configs with the stock K-tile are in the table (per-element fp32 accumulation
                           sequence unchanged -> measured bitwise at 544..872 tok).
 W4-T6  transition_rowblock Restructured inference Transition kernel (row-block program computes x_hat once, loops over N tiles;
                           same per-element arithmetic as stock -> designed bitwise; supersedes T1 when both are set).
 tx     trimul word        the pair TriMul (LN_in, projections + gates + mask, contraction, LN_out + gated out-projection, residual) served by the
                           shared core's TriMul provider (opt_core.kernels.trimul) bound by the MODE's tier word (exact | fast | big): the provider's
                           measured cell table decides the row per (capability, stack, width, size class, direction); a class whose cell names the stock
                           op, and a call the provider refuses, are served by the upstream fused TriMul BY NAME (printed once, counted); see enable_tx.
 W4-T10 transition_fused   whole pair Transition in one kernel (LN + w12 + SwiGLU + w3 + residual; Biohub rounding points); see _transition_fused_kernel.
 W4-T5  weight_cache       Cache the per-call fp32->bf16 weight casts (8 per fused TriMul call in PairUpdateBlock, 3-4 per Transition
                           under autocast): identical bf16 values, ~1400 tiny kernels per fold removed.  Cache key = parameter
                           data_ptr + version counter (refreshes if weights are reloaded / modified in place).
 probe=True                run the stock op next to every patched op once per new shape and record (max|diff|, bitwise).
 Platforms                 configs are chosen by the device's opt-in shared memory per block: >= 160 KB (sm_80 A100, sm_90 H100/H200, sm_100)
                           -> the tested H100 configs, unchanged; < 160 KB (sm_86/89/120: A10, L4, L40S, RTX class) -> T6 is replaced by T1, T3 keeps
                           the stock tiles (see _apply_device_policy), one printed line each.
                           enable() also test-launches each enabled W4 kernel once; a kernel that cannot launch on the device disables only
                           that lever with one printed '[ef2_w4] ... disabled ...' line (folds never fail because of a W4 kernel).

Usage:  import ef2_w4; ef2_w4.enable(model, tiles=True, weight_cache=True, transition_fused=True); ef2_w4.enable_tx("fast")
        (call before the kit captures CUDA graphs, or ef2_opt.clear_graphs(model) afterwards); ef2_w4.disable(); ef2_w4.stats()
"""
import os, sys, json, collections
import torch
import triton
import triton.language as tl

import transformers.models.esmfold2.modeling_esmfold2_common as C
import transformers.models.esmfold2.kernels.trimul_with_residual as TR
import transformers.models.esmfold2.kernels.fused_dual_gemm as DG
import transformers.models.esmfold2.kernels.fused_lnlin_swiglu as SW
import transformers.models.esmfold2.kernels.fused_ln_residual as LNR

VERSION = "w4.19"
STATS = collections.Counter()
PROBES = []
_STATE = dict(enabled=False, transition_nolin=False, transition_rowblock=False, tiles=False, weight_cache=False, tx=False, transition_fused=False,
              probe=False, orig=dict(), verified_shapes=set())
_BF16 = torch.bfloat16


# =====================================================================================================================
# W4-T1: inference-only LN+Linear+SwiGLU kernel without the 'lin' store (verbatim clone of Biohub _lnlin_swiglu_fwd_kernel minus Lin)
# =====================================================================================================================
@triton.jit
def _lnlin_swiglu_fwd_nolin_kernel(
    X_ptr, W_ptr, LN_W_ptr, LN_B_ptr, Out_ptr, Mean_ptr, Rstd_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_out_m, stride_out_n,
    HAS_LN_BIAS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mean = tl.load(Mean_ptr + offs_m, mask=offs_m < M, other=0.0)
    rstd = tl.load(Rstd_ptr + offs_m, mask=offs_m < M, other=0.0)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    wa_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    wb_ptrs = W_ptr + (offs_k[:, None] * stride_wk + (N + offs_n[None, :]) * stride_wn)

    a_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kk = k * BLOCK_SIZE_K
        k_remaining = K - kk
        k_mask = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        ln_w = tl.load(LN_W_ptr + kk + offs_k, mask=k_mask, other=0.0)
        if HAS_LN_BIAS:
            ln_b = tl.load(LN_B_ptr + kk + offs_k, mask=k_mask, other=0.0)
            x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
        else:
            x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :]
        wa = tl.load(wa_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        wb = tl.load(wb_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        a_acc = tl.dot(x_hat, wa, a_acc)
        b_acc = tl.dot(x_hat, wb, b_acc)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        wa_ptrs += BLOCK_SIZE_K * stride_wk
        wb_ptrs += BLOCK_SIZE_K * stride_wk

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    sig = tl.sigmoid(a_acc)
    silu_a = a_acc * sig
    swiglu = silu_a * b_acc
    out_ptrs = Out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, swiglu.to(Out_ptr.type.element_ty), mask=out_mask)


def _lnlin_swiglu_fwd_nolin(x_2d, W12, LN_W, LN_B):
    """same host code as SW._lnlin_swiglu_fwd (same _ln_stats launch, same config picker) minus the lin allocation."""
    assert x_2d.is_contiguous(), "X must be contiguous"
    M, K = x_2d.shape
    K2, two_N = W12.shape
    assert K2 == K and two_N % 2 == 0
    N = two_N // 2
    out = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
    Mean = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    Rstd = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    block, num_warps = SW._ln_stats_settings(K)
    SW._ln_stats_kernel[(M,)](x_2d, x_2d.stride(0), Mean, Mean.stride(0), Rstd, Rstd.stride(0), K, 1e-5, BLOCK_SIZE=block, num_warps=num_warps)
    cfg = SW._pick_fwd_config(K)      # stock picker (or the W4-T3 tile when tiles are enabled: hooked in _apply_triton_tiles)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    _lnlin_swiglu_fwd_nolin_kernel[grid](
        x_2d, W12, LN_W, LN_B if LN_B is not None else LN_W, out, Mean, Rstd, M, N, K,
        x_2d.stride(0), x_2d.stride(1), W12.stride(0), W12.stride(1), out.stride(0), out.stride(1),
        HAS_LN_BIAS=(LN_B is not None), BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"], BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"], num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])
    return out


# =====================================================================================================================
# W4-T6 (N1): restructured inference Transition kernel — one program per BLOCK_M row block computes x_hat ONCE (4 bf16 K-chunks kept
# on chip) and loops over all N tiles, instead of one program per (M,N) tile re-loading X and re-doing the LN arithmetic N/BLOCK_N times.
# Per-element arithmetic is the stock sequence: x_hat = ((x - mean) * rstd) * ln_w (+ ln_b) in bf16, tl.dot(bf16, bf16) accumulated in
# fp32 over the same BLOCK_K=64 chunks in the same order, fp32 SiLU * gate, bf16 store -> designed bitwise vs stock (checked by tests).
# =====================================================================================================================
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


@triton.jit
def _lnlin_swiglu_fwd_rowblock_kernel(
    X_ptr, W_ptr, LN_W_ptr, LN_B_ptr, Out_ptr, Mean_ptr, Rstd_ptr,
    M, N,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_out_m, stride_out_n,
    HAS_LN_BIAS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, N_SPLIT: tl.constexpr,
):
    """K is fixed at 256 = 4 x BLOCK_SIZE_K(64).  grid = (cdiv(M, BLOCK_M), N_SPLIT); program (pm, ps) handles rows pm*BLOCK_M.. and the
    N tiles ps, ps+N_SPLIT, ... (N_SPLIT>1 restores parallelism at small M)."""
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_s = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mean = tl.load(Mean_ptr + offs_m, mask=offs_m < M, other=0.0)
    rstd = tl.load(Rstd_ptr + offs_m, mask=offs_m < M, other=0.0)
    xh0 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 0 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh1 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 1 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh2 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 2 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh3 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 3 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    out_mask_m = offs_m[:, None] < M
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    for pn in tl.range(pid_s, num_pid_n, N_SPLIT):
        offs_n = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        nmask = offs_n[None, :] < N
        wa_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        wb_ptrs = W_ptr + (offs_k[:, None] * stride_wk + (N + offs_n[None, :]) * stride_wn)
        a_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        b_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        wa = tl.load(wa_ptrs, mask=nmask, other=0.0); wb = tl.load(wb_ptrs, mask=nmask, other=0.0)
        a_acc = tl.dot(xh0, wa, a_acc); b_acc = tl.dot(xh0, wb, b_acc)
        wa = tl.load(wa_ptrs + BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0); wb = tl.load(wb_ptrs + BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0)
        a_acc = tl.dot(xh1, wa, a_acc); b_acc = tl.dot(xh1, wb, b_acc)
        wa = tl.load(wa_ptrs + 2 * BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0); wb = tl.load(wb_ptrs + 2 * BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0)
        a_acc = tl.dot(xh2, wa, a_acc); b_acc = tl.dot(xh2, wb, b_acc)
        wa = tl.load(wa_ptrs + 3 * BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0); wb = tl.load(wb_ptrs + 3 * BLOCK_SIZE_K * stride_wk, mask=nmask, other=0.0)
        a_acc = tl.dot(xh3, wa, a_acc); b_acc = tl.dot(xh3, wb, b_acc)
        sig = tl.sigmoid(a_acc)
        silu_a = a_acc * sig
        swiglu = silu_a * b_acc
        out_ptrs = Out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        tl.store(out_ptrs, swiglu.to(Out_ptr.type.element_ty), mask=out_mask_m & nmask)


# =====================================================================================================================
# W4-T10  transition_fused — the WHOLE pair Transition (LN -> w12 -> silu(x1)*x2 -> w3 -> + residual) in ONE kernel (structure adapted from
# the FlashPairformer fpf_transition / glue transition_v3 kernels [credited], arithmetic = Biohub's rounding points):
#   per row block: LN stats (stock _ln_stats_kernel, unchanged) -> x_hat chunks exactly as T6 -> for each hidden chunk j (ascending): a,b fp32
#   accumulators over K=256 in 4 chained tl.dot (== T6 == stock kernel order), h = bf16(silu(a)*b in fp32) (the stock kernel's single rounding of
#   the SwiGLU output), acc += tl.dot(h, w3T[j-chunk]) (fp32, ascending hidden chunks: this order was measured == cuBLAS bf16 GEMM output
#   for [M,1024]x[1024,256] on H100), epilogue out = bf16(fp32(x) + acc)  (cuBLAS addmm beta=1: C is up-converted, added to the fp32 accumulator,
#   one rounding).  Whether the whole thing is bitwise vs stock (T6 kernel + cuBLAS addmm) is MEASURED (op test + det gate); if not it is Tier-2.
@triton.jit
def _transition_fused_kernel(
    X_ptr, W_ptr, W3T_ptr, LN_W_ptr, LN_B_ptr, Out_ptr, Mean_ptr, Rstd_ptr,
    M, N,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_w3h, stride_w3c, stride_out_m,
    HAS_LN_BIAS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, C: tl.constexpr, H_ROUND: tl.constexpr,
):
    pid_m = tl.program_id(axis=0).to(tl.int64)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mean = tl.load(Mean_ptr + offs_m, mask=offs_m < M, other=0.0)
    rstd = tl.load(Rstd_ptr + offs_m, mask=offs_m < M, other=0.0)
    xh0 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 0 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh1 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 1 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh2 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 2 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    xh3 = _xhat_chunk(X_ptr, LN_W_ptr, LN_B_ptr, mean, rstd, offs_m, M, stride_xm, stride_xk, 3 * BLOCK_SIZE_K, BLOCK_SIZE_K, HAS_LN_BIAS)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_c = tl.arange(0, C)
    acc = tl.zeros((BLOCK_SIZE_M, C), dtype=tl.float32)
    for pn in tl.range(0, N // BLOCK_SIZE_H):
        offs_n = pn * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        wa_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        wb_ptrs = W_ptr + (offs_k[:, None] * stride_wk + (N + offs_n[None, :]) * stride_wn)
        a_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)
        b_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)
        wa = tl.load(wa_ptrs); wb = tl.load(wb_ptrs)
        a_acc = tl.dot(xh0, wa, a_acc); b_acc = tl.dot(xh0, wb, b_acc)
        wa = tl.load(wa_ptrs + BLOCK_SIZE_K * stride_wk); wb = tl.load(wb_ptrs + BLOCK_SIZE_K * stride_wk)
        a_acc = tl.dot(xh1, wa, a_acc); b_acc = tl.dot(xh1, wb, b_acc)
        wa = tl.load(wa_ptrs + 2 * BLOCK_SIZE_K * stride_wk); wb = tl.load(wb_ptrs + 2 * BLOCK_SIZE_K * stride_wk)
        a_acc = tl.dot(xh2, wa, a_acc); b_acc = tl.dot(xh2, wb, b_acc)
        wa = tl.load(wa_ptrs + 3 * BLOCK_SIZE_K * stride_wk); wb = tl.load(wb_ptrs + 3 * BLOCK_SIZE_K * stride_wk)
        a_acc = tl.dot(xh3, wa, a_acc); b_acc = tl.dot(xh3, wb, b_acc)
        sig = tl.sigmoid(a_acc)
        silu_a = a_acc * sig
        swiglu = silu_a * b_acc
        if H_ROUND:
            h = swiglu.to(tl.bfloat16)                                                  # the stock kernel stores silu(x1)*x2 as bf16; cuBLAS reads bf16
        else:
            h = swiglu
        w3 = tl.load(W3T_ptr + offs_n[:, None] * stride_w3h + offs_c[None, :] * stride_w3c)      # [BH, C] = rows of w3^T (hidden-major)
        acc = tl.dot(h, w3, acc)
    out_mask = offs_m[:, None] < M
    xres = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_c[None, :] * stride_xk, mask=out_mask, other=0.0).to(tl.float32)
    out = xres + acc
    tl.store(Out_ptr + offs_m[:, None] * stride_out_m + offs_c[None, :], out.to(Out_ptr.type.element_ty), mask=out_mask)


# H100 sweep (9 configs x 7 sizes x 2 blocks): M128/H32/8 warps/3 stages = 1.25-1.31x over T6 (x1.53 vs stock kernel+addmm); H64 configs exceed 227 KB smem.
# NOT bitwise vs stock (cuBLAS addmm reduces the hidden dim in a different order): ~27% of output elements differ by 1 bf16 ulp, fp64-error ratio 0.94-0.96 (Tier-2).
T10_CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_H=32, num_warps=8, num_stages=3)
# Per compute capability (the H100 row above stays the default for every other class): the 3-stage M128 pipeline needs 172032 B of shared memory,
# above sm_80's 166912 B opt-in limit. A100-80GB sweep (8 configs x N 128..1300, torch 2.13.0+cu130 / triton 3.7.1): M64/H32/4 warps/3 stages is
# the fastest launchable config (x1.29 vs the stock transition at N=1024/1300; every launchable config produces identical bits).
T10_CFG_BY_CC = {"sm_80": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_H=32, num_warps=4, num_stages=3)}


def _t10_cfg():
    """T10's launch config for this device's compute capability (T10_CFG_BY_CC), else the H100 row."""
    return T10_CFG_BY_CC.get(_device_info().get("cc"), T10_CFG) if torch.cuda.is_available() else T10_CFG


def _transition_fused(x_2d, W12, W3T, LN_W, LN_B, cfg=None):
    """x_2d [M,256] bf16 contiguous; W12 [256, 2N] bf16 (= w12.weight.T as the stock module packs it); W3T [N, 256] bf16 (= w3.weight.T contiguous); returns x + transition(x) [M,256] bf16."""
    cfg = dict(_t10_cfg(), **(cfg or {}))
    assert x_2d.is_contiguous()
    M, K = x_2d.shape; K2, two_N = W12.shape; N = two_N // 2
    assert K == K2 == 256 and W3T.shape[0] == N and W3T.shape[1] == 256 and N % cfg["BLOCK_SIZE_H"] == 0
    out = torch.empty((M, 256), dtype=x_2d.dtype, device=x_2d.device)
    Mean = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device); Rstd = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    block, num_warps = SW._ln_stats_settings(K)
    SW._ln_stats_kernel[(M,)](x_2d, x_2d.stride(0), Mean, Mean.stride(0), Rstd, Rstd.stride(0), K, 1e-5, BLOCK_SIZE=block, num_warps=num_warps)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]),)
    _transition_fused_kernel[grid](x_2d, W12, W3T, LN_W, LN_B if LN_B is not None else LN_W, out, Mean, Rstd, M, N,
                                   x_2d.stride(0), x_2d.stride(1), W12.stride(0), W12.stride(1), W3T.stride(0), W3T.stride(1), out.stride(0),
                                   HAS_LN_BIAS=(LN_B is not None), BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_H=cfg["BLOCK_SIZE_H"], BLOCK_SIZE_K=64, C=256, H_ROUND=True,
                                   num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])
    return out


# H100 sweep (144 configs x 6 sizes, all bitwise vs stock): best mean = 1.26x vs the stock kernel (1.08x vs T1)
T6_CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, num_warps=8, num_stages=2, N_SPLIT=4)
# ---- platform awareness.  The tested configs above/below were tuned on H100 (227 KB opt-in shared memory per block) and need up to
# ~144 KB of shared memory; sm_86 / sm_89 / sm_120 parts (A10, L4, L40S, RTX *: 99-101 KB) cannot launch them (triton OutOfResources).
# On such devices a small-smem table is used instead; each entry keeps the K tiling / accumulation structure of the tested kernel so that
# the per-element fp32 arithmetic is unchanged (bitwise-vs-stock is nevertheless MEASURED per device class), and every
# W4 Triton kernel is test-launched once at enable(): a lever whose kernel cannot compile/launch on the device is routed to the stock op with
# one printed line instead of failing folds.
SMEM_LARGE_MIN = 160 * 1024          # >= : H100/H200/A100/B200 class -> tested configs unchanged
T6_CFG_SMALL_SMEM = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, num_warps=4, num_stages=2, N_SPLIT=4)      # ~70 KB
SMALL_SMEM_TILES = {                  # T3 on < 160 KB devices: transition tile within budget; stage-5 GEMM keeps the STOCK tile (the H100 tile needs ~144 KB)
    "transition": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 2, "num_warps": 4},
}
_DEV = dict(checked=False, name=None, cc=None, smem_optin=None, small=False, disabled={})


def _device_info():
    if not _DEV["checked"] and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        smem = int(getattr(p, "shared_memory_per_block_optin", 0) or 0)
        if smem == 0:   # older torch: derive from the architecture
            smem = {8: (166912 if p.minor == 0 else 101376), 9: 232448, 10: 232448, 12: 101376}.get(p.major, 101376)
        _DEV.update(checked=True, name=p.name, cc=f"sm_{p.major}{p.minor}", smem_optin=smem, small=(smem < SMEM_LARGE_MIN))
    return _DEV


def _t6_cfg():
    return T6_CFG_SMALL_SMEM if _device_info()["small"] else T6_CFG


def _lever_off(name):
    return name in _DEV["disabled"]


def _disable_lever(name, reason):
    if name not in _DEV["disabled"]:
        _DEV["disabled"][name] = reason
        d = _device_info()
        print(f"[ef2_w4] {name} disabled on {d['name']} ({d['cc']}, {d['smem_optin'] // 1024} KB smem/block): {reason} -> stock op used for this lever", flush=True)


def _lnlin_swiglu_fwd_rowblock(x_2d, W12, LN_W, LN_B, cfg=None):
    cfg = dict(_t6_cfg(), **(cfg or {}))
    assert x_2d.is_contiguous(), "X must be contiguous"
    M, K = x_2d.shape
    K2, two_N = W12.shape
    assert K2 == K == 256 and two_N % 2 == 0, (K, K2, two_N)
    N = two_N // 2
    out = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
    Mean = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    Rstd = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    block, num_warps = SW._ln_stats_settings(K)
    SW._ln_stats_kernel[(M,)](x_2d, x_2d.stride(0), Mean, Mean.stride(0), Rstd, Rstd.stride(0), K, 1e-5, BLOCK_SIZE=block, num_warps=num_warps)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]), cfg["N_SPLIT"])
    _lnlin_swiglu_fwd_rowblock_kernel[grid](
        x_2d, W12, LN_W, LN_B if LN_B is not None else LN_W, out, Mean, Rstd, M, N,
        x_2d.stride(0), x_2d.stride(1), W12.stride(0), W12.stride(1), out.stride(0), out.stride(1),
        HAS_LN_BIAS=(LN_B is not None), BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"], BLOCK_SIZE_K=64, N_SPLIT=cfg["N_SPLIT"],
        num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])
    return out


# =====================================================================================================================
# tx — the pair TriangleMultiplication through the shared core's TriMul provider (opt_core.kernels.trimul) bound by the MODE's TIER WORD.
# Replaces Biohub's 5-launch fused pipeline (LN_in kernel -> gated dual GEMM -> cuBLAS einsum -> LN_out transpose kernel -> gated out-GEMM + residual)
# by the row the provider's measured cell table (TRIMUL_CELLS.json) names for this call class under the word: `fast` / `big` -> the fastest measured
# kernel row of the cell on this capability and stack (tolerance class: same math, same weights, same mask convention, bf16 planes / fp32 accumulate,
# reordered fp32 arithmetic), `exact` -> an exact-class row ONLY where the table vouches it byte-identical on this stack; a class whose cell names the
# stock op (today: every class under `exact`) is served by the upstream fused TriMul BY NAME -- printed once per class, counted -- so the exact line's
# bytes are the upstream statement's.  The weights are the provider's ten canonical tensors re-viewed ONCE per (module, direction) from the bf16 casts
# the T5 cache holds (LN affine: the bf16-rounded values Biohub's fused path uses, held fp32); eligibility: CUDA bf16 square pair, no dropout mask,
# precision 0, residual is pair; a call the provider refuses BY NAME (its Refusal) takes the upstream module, counted and printed once per kind.
_IDPROBE = dict(done={}, enabled=os.environ.get("EF2_W4_IDENTITY_PROBE", "1") != "0",
                budget=int(os.environ.get("EF2_GRAPH_BUDGET_TOKENS", "0") or 0))    # the per-shape graph budget (ef2_opt's switch): above it the probe is skipped too — no O(N²·C) reference buffers at the largest shapes


def _identity_probe(name, out, ref_fn, tokens):
    """First served call per lever name (per process): run the stock op on the same input once and print IDENTICAL or TIER-2 with max|d| and the
    fraction of differing elements, keyed to the torch/triton versions (a future cuBLAS/Triton may flip a Tier-2 lever to identical or vice
    versa).  Skipped during CUDA-graph capture; EF2_W4_IDENTITY_PROBE=0 disables; skipped by name above the graph budget (EF2_GRAPH_BUDGET_TOKENS > 0
    and ``tokens`` — the pair's extent L, passed by the caller: tx ``pair.shape[1]``, T10 ``x.shape[-2]``, never read off the (possibly flattened)
    output — above it: the skip is recorded in the probe record and printed once per lever name)."""
    if not _IDPROBE["enabled"] or name in _IDPROBE["done"]:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        tokens = int(tokens)
        if _IDPROBE["budget"] > 0 and tokens > _IDPROBE["budget"]:
            _IDPROBE["done"][name] = dict(skipped=True, tokens=tokens, budget=_IDPROBE["budget"], shape=tuple(out.shape))
            print(f"[ef2_w4] identity probe {name} skipped on the first served call {tuple(out.shape)}: {tokens} tokens > EF2_GRAPH_BUDGET_TOKENS={_IDPROBE['budget']} "
                  f"(no reference buffers above the graph budget)", flush=True)
            return
        ref = ref_fn()
        same = bool(torch.equal(out, ref)); nd = int((out != ref).sum()); frac = nd / max(1, out.numel()); md = float((out.float() - ref.float()).abs().max())
        _IDPROBE["done"][name] = dict(identical=same, n_diff=nd, frac_diff=round(frac, 6), max_abs_diff=md, shape=tuple(out.shape))
        print(f"[ef2_w4] identity probe {name} vs stock op on the first served call {tuple(out.shape)}: "
              f"{'IDENTICAL' if same else 'TIER-2 (differs: %d elements = %.2f%%, max|d| %.4g)' % (nd, 100 * frac, md)} [torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name(out.device) if out.is_cuda else out.device}]", flush=True)
    except Exception as e:
        _IDPROBE["done"][name] = dict(error=repr(e)[:160])


TX_WORDS = ("exact", "fast", "big")                              # the provider's tier words (opt_core.kernels.trimul.TIER_WORDS): the MODE binds one -- exact -> `exact`, fast -> `fast`, big -> `big`
TX_WORD = "fast"                                                  # enable_tx's default word (an engineering call without a mode); the kit server passes the mode's word
TX_EAGER_PREFER = ("tx_sm90a",)                                   # under the BIG word, in EAGER use (never while a CUDA graph is capturing), on class 9.x, for call classes of
TX_EAGER_PREFER_MAX_N = 512                                       # N <= 512 tokens the provider is asked with prefer=("tx_sm90a",) -- the tier narrowed to this row FIRST, the cell's own order after
TX_EAGER_PREFER_WORDS = ("big",)                                # it (a cell that does not measure the row on this stack keeps the word's own answer, never the stock row `prefer` alone would
TX_EAGER_PREFER_CC = ("9.",)                                      # name).  Why: the cells' big winner at 257..512 tokens, row `native` (opt_core's native TriMul), is the faster DEVICE time
                                                                  # (0.43 vs 0.50-0.53 ms per call at N<=400, H100) but costs ~0.35-0.6 ms MORE HOST time per call in eager use; the memory
                                                                  # mode's graph-free trunk issues 1220 (Fast) / 2396 (full_msa) such calls per fold at 400 tokens and went host-bound (big
                                                                  # trunk 1.67 s with native vs 0.90 s with tx_sm90a per 400-token fold measured on one H100 with the row bound at
                                                                  # every size).  Inside a captured graph (the fast mode) the host cost is replayed away and the cell's winner stands; above
                                                                  # 512 tokens the calls are device-bound and the cell's winner stands.  fast / exact never take the preference.
_TX = dict(on=False, bound=False, word=None, why=None, mod=None, abi=None, cc=None, stack=None, has_cueq=None, row=None, cell=None, selection=None, canary=None,
           classes={}, rows={}, refused={}, printed=set(), stock_classes=0)


def _tx_pack(norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, key=None):
    """ESMFold2 (Biohub fused-call weights, already bf16) -> the provider's ten canonical tensors (opt_core.kernels.trimul WEIGHT_KEYS).  Biohub:
    a|b = p * sigmoid(g), p = proj_bundle signal rows [0:2D] (a rows 0:D, b rows D:2D), g = gate-logit rows; out = sigmoid(z_ln W_og^T) * (LN_out(x) W_o^T).
    LN affine: the bf16-ROUNDED values Biohub's fused path uses, held fp32.  Cached per weight identity (the T5-style key the PairUpdateBlock wrapper hands
    down); direct calls without a key (unit tests, the M1 MSA path) pack per call (views + two small casts, graph-safe)."""
    cache = _TX.setdefault("wcache", {})
    w = cache.get(key) if key is not None else None
    if w is None:
        def f32(t): return t.detach().to(_BF16).float().contiguous()
        def b16(t): return t.detach().to(_BF16).contiguous()
        D = int(p_in_weight.shape[0]) // 2
        w = dict(ln_in_w=f32(norm_in_weight), ln_in_b=f32(norm_in_bias), ln_out_w=f32(norm_out_weight), ln_out_b=f32(norm_out_bias),
                 w_ap=b16(p_in_weight[:D]), w_bp=b16(p_in_weight[D:]), w_ag=b16(g_in_weight[:D]), w_bg=b16(g_in_weight[D:]),
                 w_o=b16(p_out_weight), w_og=b16(g_out_weight))
        w = dict(tx_w=w, tx_cache={}, C=int(g_out_weight.shape[0]), CH=D)
        STATS["tx_weight_pack"] += 1
        if key is not None:
            if len(cache) > 256:
                cache.clear()
            cache[key] = w
    return w


def _tx_eligible(pair, residual, drop_mask, mask, precision):
    """The binding's call envelope (the provider decides the rest by name): CUDA bf16 square pair [B, L, L, C], no dropout mask, precision 0, residual is pair."""
    if not (_TX["on"] and pair.is_cuda and pair.dtype == _BF16 and drop_mask is None and precision == 0 and residual is pair and pair.dim() == 4):
        return False
    B, L1, L2, cz = pair.shape
    if L1 != L2:
        STATS["tx_ineligible_shape"] += 1; return False
    if mask is not None and tuple(mask.shape[-2:]) != (L1, L2):
        STATS["tx_ineligible_mask"] += 1; return False
    return True


def _tx_prefer(L, capturing):
    """The row preference this binding adds to the bound word for a call class, or None: ``TX_EAGER_PREFER`` under a word of ``TX_EAGER_PREFER_WORDS``
    (big) for N <= ``TX_EAGER_PREFER_MAX_N`` on a class of ``TX_EAGER_PREFER_CC`` while NO CUDA graph is capturing.  fast / exact / a row word,
    a capturing stream, N >= 513, another class: None -- the word alone, as before."""
    if (not capturing and _TX.get("word") in TX_EAGER_PREFER_WORDS and int(L) <= TX_EAGER_PREFER_MAX_N
            and str(_TX.get("cc") or "").startswith(TX_EAGER_PREFER_CC)):
        return TX_EAGER_PREFER
    return None


def prefer_word():
    """The blank-free word of this process's preference rule for the tx LEVER line (``prefer=``), or None when the bound word carries none."""
    if _TX.get("word") in TX_EAGER_PREFER_WORDS and str(_TX.get("cc") or "").startswith(TX_EAGER_PREFER_CC):
        return f"{'+'.join(TX_EAGER_PREFER)}:eager:N<={TX_EAGER_PREFER_MAX_N}"
    return None


def _tx_select(z4, L, direction, D, prefer=None):
    """The provider's own PURE selection for this call class under the bound word (+ the binding's row preference for the class, ``_tx_prefer``),
    memoised per (N, direction, precision, prefer): one `tx:` line per class names the word, the row, the preference and the cell.  ``rec['row']``
    is None when the cell names the stock op for the class (the upstream fused TriMul serves it BY NAME) -- under `exact` that is every class whose
    exact-class row the table does not vouch byte-identical on this stack.  ``rec['prefer']`` is the preference the SERVING calls of the class pass
    (None when none was asked, or when the cell does not serve the preferred row on this stack: the word's own answer stands, by name)."""
    TRI = _TX["mod"]
    prec = TRI.call_precision(z4)[0]
    prefer = tuple(prefer) if prefer else None
    key = (int(L), str(direction), prec, prefer)
    rec = _TX["classes"].get(key)
    if rec is not None:
        return rec
    f32z = prec.startswith("f32z_")
    sdt = prec.replace("f32z_", "") if f32z else ("fp32" if prec == "tf32" else prec)
    word = _TX["word"]
    kept, pnote = None, ""
    try:
        skw = dict(word=word, residency=("fp32" if f32z else None), backward=False, stack=_TX["stack"], tf32=(prec == "tf32"), abi=_TX["abi"], has_cueq=_TX["has_cueq"])
        sel = None
        if prefer:
            try:
                sel = TRI.select(_TX["cc"], sdt, int(z4.shape[-1]), int(D), int(L), direction, prefer=prefer, **skw)
            except TRI.Refusal as e:                           # the narrowed tier names nothing servable here: the word alone decides the class (by name on the `tx:` line)
                sel, pnote = None, f" prefer={'+'.join(prefer)}(eager,N<={TX_EAGER_PREFER_MAX_N}):refusal({str(getattr(e, 'kind', e)).split('(')[0]})->the_word_alone"
            else:
                if sel.row in prefer:                              # the cell measures and admits the preferred row on this stack: it serves, ahead of the cell's winner
                    kept, pnote = prefer, f" prefer={'+'.join(prefer)}(eager,N<={TX_EAGER_PREFER_MAX_N}):served"
                else:                                              # not measured / not admitted here: the word's own answer (the cell's order), never the stock row prefer narrows a tier to
                    pnote = f" prefer={'+'.join(prefer)}(eager,N<={TX_EAGER_PREFER_MAX_N}):not_served_by_the_cell({sel.row})->the_word_alone"; sel = None
        if sel is None:
            sel = TRI.select(_TX["cc"], sdt, int(z4.shape[-1]), int(D), int(L), direction, **skw)
        cell_row, cell, desc = sel.row, sel.cell, TRI.describe(sel)
        row = cell_row if cell_row not in TRI.STOCK_ROWS else None
        if row is not None and word == "exact" and row not in tuple(getattr(TRI, "EXACT_ROWS", ())):
            row, desc = None, desc + " [not an exact-class row: the upstream statement keeps the exact line]"
    except TRI.Refusal as e:                                   # a selection refuses only under prefer / form; named all the same
        cell_row, cell, desc, row, kept = None, None, "refusal:%s[row %s]->%s" % (getattr(e, "kind", e), getattr(e, "row", None), getattr(e, "fallback", None)), None, None
    rec = dict(row=row, cell_row=cell_row, cell=cell, describe=desc, N=int(L), direction=str(direction), precision=prec, calls=0, stock_calls=0,
               prefer=kept, prefer_asked=prefer)
    _TX["classes"][key] = rec
    if row is None:
        _TX["stock_classes"] += 1
    what = (f"served by row {row} (the provider steps aside by name inside the word; a call it refuses takes the upstream fused TriMul, named)" if row is not None else
            f"the cell names the stock op for this class ({cell_row}): the upstream fused TriMul serves it BY NAME")
    print(f"[ef2_w4] tx: word={word}{pnote} row={row if row is not None else 'upstream'} N={int(L)} {direction} {prec} -- {what}; cell {cell}; {desc}", flush=True)
    return rec


def _tx_mask(mask, B, L):
    if mask is None:
        return None
    m = mask
    if m.dtype == torch.bool:
        m = m.to(torch.float32)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    return m.expand(B, L, L)


def _tx_forward(pair, direction, mask, w):
    """The whole pair TriMul through the provider under the bound tier word; returns None when the cell names the stock op for this class or the
    provider refuses this call BY NAME (counted, printed once per kind) so the upstream fused TriMul serves it."""
    TRI = _TX["mod"]
    B, L, _, cz = pair.shape
    pc = pair.contiguous()
    capturing = bool(pc.is_cuda and torch.cuda.is_current_stream_capturing())
    rec = _tx_select(pc, L, direction, w["CH"], _tx_prefer(L, capturing))
    if rec["row"] is None:
        rec["stock_calls"] += 1; STATS["tx_class_stock"] += 1
        return None
    m = _tx_mask(mask, B, L)
    tw, cache = w["tx_w"], w["tx_cache"]
    pkw = dict(prefer=rec["prefer"]) if rec.get("prefer") else {}     # the class's row preference (big, eager, N<=512: tx_sm90a first) rides on the serving call; none otherwise
    out = None
    try:
        for b in range(B):
            mb = m[b].contiguous() if m is not None else None
            o = TRI.triangle_multiplication(pc[b:b + 1], mb, direction=direction, weights=tw, word=_TX["word"], residual=True, cache=cache, **pkw)
            if cache.pop("_z_cast", None) is not None:           # the face's cast memo is call-scoped here: an autocast-dtype copy of this call's pair
                STATS["tx_z_cast_popped"] += 1                    # input is never held in the per-module cache past the call (values unchanged)
            if B == 1:
                out = o.reshape(B, L, L, cz)
            else:                                              # B > 1 (the confidence head's per-sample pair trunk): ONE B x L x L x C output written per sample,
                if out is None:                                # never B per-sample outputs plus their concatenation (a second full-size copy of the batched
                    out = torch.empty((B, L, L, cz), dtype=o.dtype, device=o.device)   # pair tensor is what runs the memory mode out of memory first); copy_ is exact
                out[b].copy_(o.reshape(L, L, cz))
                del o
    except TRI.Refusal as e:                                   # the word's whole chain refuses this call by name: the upstream fused TriMul serves it
        kind = str(getattr(e, "kind", e)).split("(")[0]
        _TX["refused"][kind] = _TX["refused"].get(kind, 0) + 1
        STATS["tx_refused"] += 1
        if kind not in _TX["printed"]:
            _TX["printed"].add(kind)
            print(f"[ef2_w4] tx: word={_TX['word']}: the provider refuses {getattr(e, 'kind', e)} [row {getattr(e, 'row', rec['row'])}] at N={L} ({direction}); "
                  f"the upstream fused TriMul serves the calls of this kind BY NAME (named once; describe()['tx_state']['refused'] counts them)", flush=True)
        return None
    sel = cache.get("_last")
    served = getattr(sel, "row", rec["row"]) or rec["row"]
    if served in TRI.STOCK_ROWS:                               # the word's chain ended at the stock row for this class (every kernel row stepped aside): from here on the
        rec["row"] = None; _TX["stock_classes"] += 1           # upstream fused TriMul serves the class BY NAME (this one call took the provider's stock statement)
        print(f"[ef2_w4] tx: word={_TX['word']} N={L} {direction}: every kernel row of the cell stepped aside on this stack ({'+'.join(cache.get('_stepaside') or []) or '?'}); "
              f"the upstream fused TriMul serves this class BY NAME from here", flush=True)
    STATS["tx_calls"] += 1
    if pkw:
        STATS["tx_prefer_calls"] += 1
    rec["calls"] += 1; _TX["rows"][served] = _TX["rows"].get(served, 0) + 1
    if rec.get("served") != served:                            # the row that ACTUALLY served (a step-aside inside the word lands on the cell's next row: named here once per class)
        rec["served"] = served
        if served != rec["cell_row"]:
            print(f"[ef2_w4] tx: word={_TX['word']} N={L} {direction}: served by row {served} (cell row {rec['cell_row']} stepped aside by name: "
                  f"{'+'.join(cache.get('_stepaside') or []) or '?'})", flush=True)
    return out


def enable_tx(word=TX_WORD):
    """Bind the pair TriMul to the shared core's TriMul provider under ``word`` -- a tier word (`exact` | `fast` | `big`: the MODE's) or one of the
    provider's ROW_NAMES (exactly that row, for an A/B).  Installs W4's TriMul entry points when enable() has not, loads the provider, runs a canary
    (N=128, both directions) through the word when its cell names a kernel row, and records the binding (describe()['tx_state']; the tx LEVER line
    reads it).  On a class / stack where the word resolves to the stock op for every canary class the binding is live and every call class is named
    on its `tx:` line as served by the upstream fused TriMul BY NAME (``on`` False, ``bound`` True, ``why`` says so) -- the exact line today.
    Called by ef2_server.configure right after ef2_w4.enable."""
    _TX.update(on=False, bound=False, word=str(word), why=None, mod=None, classes={}, rows={}, refused={}, printed=set(), stock_classes=0, row=None, cell=None,
               selection=None, canary=None)
    _TX.pop("wcache", None)
    if not _STATE["enabled"]:                                  # a set without W4 levers: install the TriMul entry points alone (the transition patch falls through to stock)
        _save_orig()
        _STATE["enabled"] = True
        C.Transition.forward = _transition_forward_w4
        C.PairUpdateBlock._fused_trimul_with_residual = _pub_fused_trimul_w4
        C._fused_trimul_with_residual = _trimul_w4
    _STATE["tx"] = True
    try:
        from opt_core.kernels import trimul as TRI
        if _TX["word"] not in tuple(TRI.TIER_WORDS) + tuple(TRI.ROW_NAMES):
            _STATE["tx"] = False
            _TX["why"] = f"unknown word {_TX['word']!r} (tier words {','.join(TRI.TIER_WORDS)}; rows {','.join(TRI.ROW_NAMES)}): the upstream fused TriMul serves"; return dict(tx_state())
        _TX["mod"] = TRI
        if not torch.cuda.is_available():                      # the kit's no-device path (ESMFOLD2_OPT_FORCE=1 applies the set anyway): the word is bound, the provider's
            _TX.update(bound=True, why="no_cuda_device(canary deferred)")   # selection and canary are deferred to the first call on a device; named, like every W4 optimization there
            print(f"[ef2_w4] tx: word={_TX['word']} bound; no CUDA device -- provider selection and canary deferred (named)", flush=True)
            return dict(tx_state())
        cc = torch.cuda.get_device_capability()
        _TX.update(cc="%d.%d" % tuple(cc), stack=TRI.stack_word(), has_cueq=bool(TRI.cueq_present()))
        try:
            _TX["abi"] = TRI.tx_abi_tag()
        except Exception:                                      # an ops module that cannot name its ABI has no prebuilt binary either: the provider skips those rows by name
            _TX["abi"] = "unknown"
        g = torch.Generator(device="cuda").manual_seed(20260911)
        N, Cz = 128, 256
        z = torch.randn(1, N, N, Cz, device="cuda", dtype=torch.float32, generator=g).to(_BF16)
        msk = torch.ones(N, N, device="cuda", dtype=torch.float32)
        rw = lambda *shape: (torch.randn(*shape, device="cuda", dtype=torch.float32, generator=g) * 0.05)      # noqa: E731
        tw = dict(ln_in_w=torch.ones(Cz, device="cuda"), ln_in_b=torch.zeros(Cz, device="cuda"), ln_out_w=torch.ones(Cz, device="cuda"), ln_out_b=torch.zeros(Cz, device="cuda"),
                  w_ag=rw(Cz, Cz).to(_BF16), w_bg=rw(Cz, Cz).to(_BF16), w_ap=rw(Cz, Cz).to(_BF16), w_bp=rw(Cz, Cz).to(_BF16), w_o=rw(Cz, Cz).to(_BF16), w_og=rw(Cz, Cz).to(_BF16))
        cache = {}; served = []
        for d in ("outgoing", "incoming"):
            rec = _tx_select(z, N, d, Cz, _tx_prefer(N, False))  # the canary class resolves as the folds' eager calls of that class will (with the word's row preference)
            if rec["row"] is None:                             # the cell names the stock op for the canary class under this word (the exact word today): nothing to launch here
                served.append((d, None, rec["cell"])); continue
            o = TRI.triangle_multiplication(z, msk, direction=d, weights=tw, word=_TX["word"], residual=True, cache=cache, **(dict(prefer=rec["prefer"]) if rec.get("prefer") else {}))
            if not bool(torch.isfinite(o.float()).all()):
                raise RuntimeError(f"canary {d}: non-finite output [row {rec['row']}]")
            sel = cache.get("_last"); served.append((d, getattr(sel, "row", rec["row"]), getattr(sel, "cell", rec["cell"])))
        kernel = [s for s in served if s[1] is not None and s[1] not in TRI.STOCK_ROWS]
        rec = _TX["classes"].get((N, "incoming", "bf16", _tx_prefer(N, False))) or rec
        _TX.update(bound=True, on=bool(kernel), row=(kernel[-1][1] if kernel else None), cell=served[-1][2], selection=rec.get("describe"), canary=served)
        for r in _TX["classes"].values():                      # the canary's calls are not the model's: the per-class call counts start at the first fold
            r["calls"] = 0; r["stock_calls"] = 0
        _TX["rows"] = {}; _TX["stock_classes"] = 0; STATS["tx_calls"] = 0; STATS["tx_class_stock"] = 0
        del z, msk, tw, cache
        if kernel:
            print(f"[ef2_w4] tx: on -- word={_TX['word']} bound (the shared core's cell table decides the row per size class and direction; canary N=128 out+in "
                  f"served by row {_TX['row']}, cell {_TX['cell']}, finite; stack {_TX['stack']}; cc {_TX['cc']}; prebuilt ABI {_TX['abi']}); every call class names its row on a "
                  f"`tx:` line; classes whose cell names the stock op and calls the provider refuses are named and served by the upstream fused TriMul", flush=True)
        else:
            _TX["why"] = f"word_{_TX['word']}_names_the_stock_op_on_this_stack({_TX['stack']}):upstream_fused_trimul_by_name"
            print(f"[ef2_w4] tx: bound -- word={_TX['word']}: the provider's cell names the stock op for the canary classes on this stack ({_TX['stack']}, cc {_TX['cc']}): "
                  f"the upstream fused TriMul serves the pair TriMul BY NAME (every call class is still resolved through the word and named on its `tx:` line; "
                  f"a row the table vouches on this stack later is served with no change here)", flush=True)
    except Exception as e:                                     # an import / load error or a row's error on the canary: off BY NAME (the upstream fused TriMul serves); an OOM propagates
        from opt_core.oom import is_oom
        if is_oom(e): raise
        kind = getattr(e, "kind", None)
        _TX["why"] = (f"refused:{kind}[row {getattr(e, 'row', '?')}]->{getattr(e, 'fallback', '?')}" if kind is not None else f"{type(e).__name__}:{str(e)[:160]}")
        _TX["on"] = False; _STATE["tx"] = False
        print(f"[ef2_w4] tx: off by name -- word={_TX['word']} {_TX['why']}; the upstream fused TriMul serves", flush=True)
    return dict(tx_state())


def tx_state():
    """The binding's record for describe()['tx_state'] and the tx LEVER line: the bound word, the rows served so far with call counts (``rows``), one
    entry per call class (``classes``: N, direction, precision, the row served or None = the upstream fused TriMul by name, the cell and the provider's
    description), the canary's row / cell, this process's stack word and prebuilt ABI key, the refusal tally, the classes the stock op serves by name."""
    classes = [dict(N=r["N"], direction=r["direction"], precision=r["precision"], row=r["row"], served=r.get("served"), cell_row=r["cell_row"], cell=r["cell"],
                    calls=int(r["calls"]), stock_calls=int(r.get("stock_calls", 0)), prefer=("+".join(r["prefer"]) if r.get("prefer") else None),
                    prefer_asked=("+".join(r["prefer_asked"]) if r.get("prefer_asked") else None))
               for r in sorted(_TX["classes"].values(), key=lambda r: (r["N"], r["direction"], r["precision"], str(r.get("prefer_asked") or "")))]
    return dict(on=bool(_TX["on"]), bound=bool(_TX["bound"]), installed=bool(_TX["bound"]), word=_TX["word"], why=_TX["why"], abi=_TX["abi"], stack=_TX["stack"], cc=_TX["cc"],
                has_cueq=_TX["has_cueq"], row=_TX["row"], cell=_TX["cell"], selection=_TX["selection"], rows=dict(_TX["rows"]), classes=classes,
                calls=int(STATS.get("tx_calls", 0)), refused=dict(_TX["refused"]), class_stock_calls=int(STATS.get("tx_class_stock", 0)), stock_classes=int(_TX.get("stock_classes", 0)),
                weight_packs=int(STATS.get("tx_weight_pack", 0)), prefer_rule=prefer_word(), prefer_calls=int(STATS.get("tx_prefer_calls", 0)))


def _trimul_w4(pair, direction, residual, drop_mask, *, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias,
               p_out_weight, g_out_weight, mask=None, eps=1e-5, precision=0):
    stock = _STATE["orig"]["trimul_fn"]
    if _STATE["enabled"] and _STATE["tx"] and not torch.is_grad_enabled() and _tx_eligible(pair, residual, drop_mask, mask, precision):
        w = _tx_pack(norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, key=_TX.get("cur_key"))
        o = _tx_forward(pair, direction, mask, w)
        if o is not None:
            if _STATE["probe"] and ("tx_" + direction, tuple(pair.shape)) not in _TX.setdefault("probed", set()):
                ref = stock(pair, direction, residual, drop_mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight, g_in_weight=g_in_weight,
                            norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight, mask=mask, eps=eps, precision=precision)
                _TX["probed"].add(("tx_" + direction, tuple(pair.shape)))
                PROBES.append(("tx_" + direction, tuple(pair.shape), float((o.float() - ref.float()).abs().max()), bool(torch.equal(o, ref))))
                return o
            _identity_probe("tx/" + direction, o, lambda: stock(pair, direction, residual, drop_mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight, g_in_weight=g_in_weight,
                            norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight, mask=mask, eps=eps, precision=precision),
                            tokens=pair.shape[1])                               # the pair's extent L (B, L, L, C), the graph budget's unit
            return o
        STATS["tx_upstream_calls"] += 1
    return stock(pair, direction, residual, drop_mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight, g_in_weight=g_in_weight,
                 norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight, mask=mask, eps=eps, precision=precision)


# =====================================================================================================================
# W4-T5 weight-cast cache + patched PairUpdateBlock._fused_trimul_with_residual / Transition.forward
# =====================================================================================================================
def _pkey(*params):
    return tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params if p is not None)


def _bf16c(t):
    return t if t.dtype == _BF16 else t.to(_BF16)


def _pub_fused_trimul_w4(self, pair, direction, pair_attention_mask):
    """PairUpdateBlock._fused_trimul_with_residual with the 8 weight casts cached (T5); otherwise the stock call."""
    if not (_STATE["enabled"] and (_STATE["weight_cache"] or _STATE["tx"])) or torch.is_grad_enabled():
        return _STATE["orig"]["pub_fused_trimul"](self, pair, direction, pair_attention_mask)
    tri = self.tri_mul_out if direction == "outgoing" else self.tri_mul_in
    engine = tri._engine
    cache = self.__dict__.setdefault("_w4_wcache", {})
    key = _pkey(engine.norm_start.weight, engine.norm_start.bias, engine.proj_bundle.weight, engine.norm_mix.weight, engine.norm_mix.bias, engine.proj_emit.weight, engine.proj_gate.weight)
    ent = cache.get(direction)
    if ent is None or ent[0] != key:
        p_in_weight, g_in_weight = engine.split_kernel_weights()
        w = dict(norm_in_weight=_bf16c(engine.norm_start.weight), norm_in_bias=_bf16c(engine.norm_start.bias), p_in_weight=_bf16c(p_in_weight), g_in_weight=_bf16c(g_in_weight),
                 norm_out_weight=_bf16c(engine.norm_mix.weight), norm_out_bias=_bf16c(engine.norm_mix.bias), p_out_weight=_bf16c(engine.proj_emit.weight), g_out_weight=_bf16c(engine.proj_gate.weight))
        w = {k: v.detach() for k, v in w.items()}
        cache[direction] = (key, w); STATS["t5_trimul_cache_fill"] += 1
        ent = cache[direction]
    STATS["t5_trimul_calls"] += 1
    _TX["cur_key"] = (id(self), direction, key)          # stable identity for the tx pack cache (consumed by _trimul_w4 in this same call)
    try:
        return C._fused_trimul_with_residual(pair, direction, residual=pair, drop_mask=None, mask=pair_attention_mask, eps=C._EPS, **ent[1])
    finally:
        _TX["cur_key"] = None


def _transition_forward_w4(self, x):
    """C.Transition.forward inference fast path with T1 (no-lin kernel) and/or T5 (cached bf16 weight casts).  Mirrors the stock
    fast path exactly: FusedLNLinearSwiGLUFunction.forward runs under torch.amp.custom_fwd(cast_inputs=bf16) -> when autocast is
    active all floating CUDA tensor args are cast to bf16 and the body runs with autocast disabled; _addmm_residual runs under the
    ambient autocast (which casts w3.weight.t() to bf16)."""
    use = _STATE["enabled"] and (_STATE["transition_nolin"] or _STATE["transition_rowblock"] or _STATE["weight_cache"] or _STATE["transition_fused"]) and (not torch.is_grad_enabled()) and self._can_use_fused_path(x)
    if not use:
        return _STATE["orig"]["transition_forward"](self, x)
    fused = self._fused_swiglu
    ac = torch.is_autocast_enabled("cuda")
    ac_bf16 = ac and torch.get_autocast_dtype("cuda") == _BF16
    if ac and not ac_bf16:
        return _STATE["orig"]["transition_forward"](self, x)          # unexpected autocast dtype: stock
    if _STATE["weight_cache"]:
        cache = self.__dict__.setdefault("_w4_wcache", {})
        key = (_pkey(fused.W12, fused.LN_W, fused.LN_B, self.ffn.w3.weight), ac_bf16)
        if cache.get("key") != key:
            if ac_bf16:
                cache.update(W12=_bf16c(fused.W12.detach()), LN_W=_bf16c(fused.LN_W.detach()), LN_B=(_bf16c(fused.LN_B.detach()) if fused.LN_B is not None else None),
                             w3t=self.ffn.w3.weight.detach().t().to(_BF16))
            else:
                cache.update(W12=fused.W12.detach(), LN_W=fused.LN_W.detach(), LN_B=(fused.LN_B.detach() if fused.LN_B is not None else None), w3t=self.ffn.w3.weight.detach().t())
            cache["key"] = key; STATS["t5_transition_cache_fill"] += 1
        W12, LN_W, LN_B, w3t = cache["W12"], cache["LN_W"], cache["LN_B"], cache["w3t"]
    else:
        if ac_bf16:
            W12, LN_W, LN_B = _bf16c(fused.W12), _bf16c(fused.LN_W), (_bf16c(fused.LN_B) if fused.LN_B is not None else None)
        else:
            W12, LN_W, LN_B = fused.W12, fused.LN_W, fused.LN_B
        w3t = self.ffn.w3.weight.t()
    nolin = _STATE["transition_nolin"] and not _lever_off("T1"); rowblock = _STATE["transition_rowblock"] and W12.shape[0] == 256 and not _lever_off("T6")

    def pre_w3(t):
        x_shape = t.shape
        with torch.amp.autocast("cuda", enabled=False):
            x_2d = t.contiguous().view(-1, x_shape[-1])            # t is bf16 (fused path precondition) -> custom_fwd cast is a no-op
            if rowblock:
                out = _lnlin_swiglu_fwd_rowblock(x_2d, W12, LN_W, LN_B); STATS["t6_calls"] += 1
            elif nolin:
                out = _lnlin_swiglu_fwd_nolin(x_2d, W12, LN_W, LN_B); STATS["t1_calls"] += 1
            else:
                out = SW._lnlin_swiglu_fwd(x_2d, W12, LN_W, LN_B)[0]; STATS["t5_transition_calls"] += 1
        if _STATE["probe"]:
            key = ("transition", tuple(x_2d.shape), nolin, rowblock)
            if key not in _STATE["verified_shapes"]:
                _STATE["verified_shapes"].add(key)
                ref = fused(t)
                PROBES.append(("transition_rowblock" if rowblock else ("transition_nolin" if nolin else "transition_wcache"), tuple(x_2d.shape), float((out.float() - ref.reshape(out.shape).float()).abs().max()), bool(torch.equal(out, ref.reshape(out.shape)))))
        return out.view(*x_shape[:-1], out.shape[-1])

    def addmm_res(xx, hidden):
        x_shape = xx.shape
        out = torch.addmm(xx.contiguous().view(-1, x_shape[-1]), hidden.view(-1, hidden.shape[-1]), w3t)
        return out.view(x_shape)

    if _STATE["transition_fused"] and ac_bf16 and W12.shape[0] == 256 and x.dtype == _BF16 and not _lever_off("T10"):
        # T10: whole transition in one kernel (row blocks are independent -> chunking over dim 1 is unnecessary; output written once)
        # w3^T must be bf16 + contiguous for the kernel (the stock path leaves the cast to autocast inside addmm; T5 caches a bf16 copy already)
        if w3t.dtype == _BF16 and w3t.is_contiguous():
            w3t_c = w3t
        else:
            cache10 = self.__dict__.setdefault("_w4_w3t_c", {})
            k10 = (w3t.data_ptr(), tuple(w3t.shape), w3t.dtype, (w3t._version if not w3t.is_inference() else -1))
            w3t_c = cache10.get(k10)
            if w3t_c is None:
                cache10.clear(); w3t_c = cache10.setdefault(k10, w3t.detach().to(_BF16).contiguous())
        x_2d = x.contiguous().view(-1, x.shape[-1])
        with torch.amp.autocast("cuda", enabled=False):
            out2d = _transition_fused(x_2d, W12, w3t_c, LN_W, LN_B)
        STATS["t10_calls"] += 1
        _identity_probe("T10", out2d, lambda: _STATE["orig"]["transition_forward"](self, x).reshape(out2d.shape), tokens=x.shape[-2])   # x is the pair (B, L, L, C): L, not out2d's flattened B·L² rows
        if _STATE["probe"]:
            key = ("transition_fused", tuple(x_2d.shape))
            if key not in _STATE["verified_shapes"]:
                _STATE["verified_shapes"].add(key)
                ref = _STATE["orig"]["transition_forward"](self, x)
                PROBES.append(("transition_fused", tuple(x_2d.shape), float((out2d.float() - ref.reshape(out2d.shape).float()).abs().max()), bool(torch.equal(out2d, ref.reshape(out2d.shape)))))
        return out2d.view(x.shape)
    if self._chunk_size is None or x.shape[1] <= self._chunk_size:
        return addmm_res(x, pre_w3(x))
    out = torch.empty_like(x)
    for s in range(0, x.shape[1], self._chunk_size):
        e = min(s + self._chunk_size, x.shape[1])
        sl = x[:, s:e]
        out[:, s:e] = addmm_res(sl, pre_w3(sl))
    return out


# =====================================================================================================================
# W4-T3: tile registry
# =====================================================================================================================
_TILES = {}
# Built-in H100 table (measured on NVIDIA H100 80GB HBM3, torch 2.13.0+cu130, Triton 3.7.1; 11 sizes 544..872 tok):
# every entry keeps the stock K tile, so each output element is accumulated over the same K chunks in the same order -> measured
# BITWISE identical to the stock config at all sizes.  Kernel time vs stock: transition 0.76-0.78x, s2 dual-GEMM ~1.0x (stock is
# already the best of the sweep), s5 out-GEMM 0.97x.
H100_DEFAULT_TILES = {
    "transition": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
    "s5_outgemm": {"TILE_M": 64, "TILE_N": 128, "TILE_K": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 4},
}


def _tile_cfg(kind):
    if not _STATE["enabled"] or not _STATE["tiles"]:
        return None
    return _TILES.get(kind)


# Per compute capability (the H100 table above stays the default for every other >= 160 KB class). A100-80GB sweep (block-level, N 256..1300, stock K tile kept,
# torch 2.13.0+cu130 / triton 3.7.1): the H100 transition tile is neutral-to-slower on sm_80 (x0.976 at N=256); M128/N64/3 stages/4 warps is >= stock at every
# size (x1.002..1.02, bitwise); the stage-5 tile is the H100 one (x1.00..1.01, bitwise).
TILES_BY_CC = {"sm_80": {
    "transition": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 4},
    "s5_outgemm": {"TILE_M": 64, "TILE_N": 128, "TILE_K": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 4},
}}


def default_tiles():
    """The built-in tile table for this device: its compute capability's row (TILES_BY_CC), else SMALL_SMEM_TILES on < 160 KB devices, else the H100 table."""
    d = _device_info()
    return TILES_BY_CC.get(d.get("cc")) or (SMALL_SMEM_TILES if d["small"] else H100_DEFAULT_TILES)


def load_tiles(src):
    """src: True -> built-in table for this device (default_tiles: the cc row, the small-smem table, or the H100 table); dict; or a json path."""
    global _TILES
    if src is True:
        src = default_tiles()
    elif isinstance(src, str):
        src = json.load(open(src))
    _TILES = dict(src)
    return _TILES


def _apply_triton_tiles():
    o = _STATE["orig"]
    for kind, kern, key in (("s2_dualgemm", DG._gated_dual_gemm_kernel, "s2_cfgs"), ("s5_outgemm", TR._gated_gemm_with_residual_kernel, "s5_cfgs")):
        cfg = _TILES.get(kind)
        if cfg:
            kern.configs = [triton.Config({k: int(cfg[k]) for k in ("TILE_M", "TILE_N", "TILE_K", "GROUP_M")}, num_stages=int(cfg["num_stages"]), num_warps=int(cfg["num_warps"]))]
        else:
            kern.configs = list(o[key])
        kern.cache.clear()
    tcfg = _TILES.get("transition")
    if tcfg:
        SW._pick_fwd_config = (lambda K, _c=dict(tcfg), _orig=o["pick_fwd_config"]: dict(_c) if K == 256 else _orig(K))
    else:
        SW._pick_fwd_config = o["pick_fwd_config"]
    STATS["tiles_applied"] += 1


def _restore_triton_tiles():
    o = _STATE["orig"]
    if "pick_fwd_config" in o:
        SW._pick_fwd_config = o["pick_fwd_config"]
        DG._gated_dual_gemm_kernel.configs = list(o["s2_cfgs"]); DG._gated_dual_gemm_kernel.cache.clear()
        TR._gated_gemm_with_residual_kernel.configs = list(o["s5_cfgs"]); TR._gated_gemm_with_residual_kernel.cache.clear()


# =====================================================================================================================
def _save_orig():
    o = _STATE["orig"]
    if "transition_forward" not in o:
        o["transition_forward"] = C.Transition.forward
        o["pub_fused_trimul"] = C.PairUpdateBlock._fused_trimul_with_residual
        o["trimul_fn"] = C._fused_trimul_with_residual            # == TR.triangle_multiplicative_update_with_residual
        o["trimul_fn_TR"] = TR.triangle_multiplicative_update_with_residual
        o["pick_fwd_config"] = SW._pick_fwd_config
        o["s2_cfgs"] = list(DG._gated_dual_gemm_kernel.configs); o["s5_cfgs"] = list(TR._gated_gemm_with_residual_kernel.configs)


def enable(model=None, transition_nolin=False, tiles=None, weight_cache=False, probe=False, transition_rowblock=False, transition_fused=False):
    """Install W4 levers process-wide (module-level patches; `model` accepted for API symmetry).  Call BEFORE the kit captures CUDA
    graphs (or ef2_opt.clear_graphs(model) after) so the graphs contain the patched kernels."""
    _save_orig()
    _STATE.update(enabled=True, transition_nolin=bool(transition_nolin), transition_rowblock=bool(transition_rowblock),
                  tiles=bool(tiles), weight_cache=bool(weight_cache), transition_fused=bool(transition_fused), probe=bool(probe))
    C.Transition.forward = _transition_forward_w4
    C.PairUpdateBlock._fused_trimul_with_residual = _pub_fused_trimul_w4
    C._fused_trimul_with_residual = _trimul_w4          # used by PairUpdateBlock (all FoldingTrunks) and by the kit's M1 MSA path (tx binds the provider on it: enable_tx)
    _apply_device_policy()
    if _STATE["tiles"]:
        load_tiles(tiles if tiles not in (None, False) else True); _apply_triton_tiles()
    else:
        _restore_triton_tiles()
    _probe_levers()
    return describe()


def _policy_line(name, text):
    if name not in _DEV["disabled"]:
        _DEV["disabled"][name] = text
        d = _device_info()
        print(f"[ef2_w4] {name} on {d['name']} ({d['cc']}, {d['smem_optin'] // 1024} KB smem/block): {text}", flush=True)


def _apply_device_policy():
    """Device policy.  >= 160 KB opt-in shared memory (A100 / H100 / H200 / B200 class): tested configuration, unchanged.
    Smaller devices (sm_86 / sm_89 / sm_120 class, ~99-101 KB): the H100-tuned kernels either cannot launch (T6, T3 stage-5 tile) or bring no
    measured benefit, so:
      T6 requested -> T1 is used instead (the stock kernel's own tile config minus the backward-only store; bitwise vs stock on H100 and L40S),
      T3 -> stock tiles, T5 unchanged.  One line is printed per substitution."""
    if not torch.cuda.is_available():
        return
    d = _device_info()
    if not d["small"]:
        return
    if _STATE["transition_rowblock"]:
        _STATE["transition_rowblock"] = False; _STATE["transition_nolin"] = True
        _policy_line("T6", "row-block transition kernel needs > 99 KB shared memory; using T1 (exact no-lin transition kernel, stock tile config) instead")
    if _STATE["transition_fused"]:
        _STATE["transition_fused"] = False
        if not (_STATE["transition_nolin"] or _STATE["transition_rowblock"]):
            _STATE["transition_nolin"] = True
        _policy_line("T10", "fused transition kernel: its launch row is a 3-stage M128 pipeline needing 168 KB of shared memory, above this device's; using T1 instead")
    if _STATE["tiles"]:
        _STATE["tiles"] = False
        _policy_line("T3", "no tile table for this device class; stock Triton tiles kept")


def _probe_levers():
    """Test-launch every enabled W4 Triton kernel once on tiny inputs (per process); a kernel that cannot compile or launch on this device
    (e.g. shared-memory limits on sm_86/89/120) disables just that lever with one printed line; folds never fail because of a W4 kernel."""
    if not torch.cuda.is_available() or _STATE.get("probed"):
        return
    d = _device_info(); dev = torch.device("cuda")
    L, CZ, CH = 20, 256, 128
    g = torch.Generator(device="cuda"); g.manual_seed(1)
    x = torch.randn(L * L, CZ, device=dev, generator=g).to(_BF16)
    def probe(name, fn):
        try:
            fn(); torch.cuda.synchronize()
        except Exception as e:                       # triton OutOfResources / CompilationError / CUDA launch errors; a GPU out-of-memory error propagates
            from opt_core.oom import is_oom
            if is_oom(e): raise
            _disable_lever(name, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
    with torch.no_grad():
        W12 = torch.randn(CZ, 2 * 4 * CZ // 2, device=dev, generator=g).to(_BF16)        # (K=256, 2N) with N = 512 like the pair Transition (expansion 2)
        LN_W = torch.ones(CZ, device=dev, dtype=_BF16); LN_B = torch.zeros(CZ, device=dev, dtype=_BF16)
        if _STATE["transition_rowblock"]:
            probe("T6", lambda: _lnlin_swiglu_fwd_rowblock(x, W12, LN_W, LN_B))
        if _STATE["transition_nolin"]:
            probe("T1", lambda: _lnlin_swiglu_fwd_nolin(x, W12, LN_W, LN_B))
        if _STATE["tiles"]:
            def t3():
                cfg = SW._pick_fwd_config(256)
                SW._lnlin_swiglu_fwd(x, W12, LN_W, LN_B)                                   # transition tile (T3) through the stock launcher
                xs = torch.randn(L * L, CH, device=dev, generator=g).to(_BF16); w1 = torch.randn(CZ, CH, device=dev, generator=g).to(_BF16)
                res = torch.randn(L * L, CZ, device=dev, generator=g).to(_BF16)
                TR._gated_gemm_with_residual_fwd(xs, xs, w1, w1, res, None, L, L)          # stage-5 tile (T3) through the stock launcher
            probe("T3", t3)
            if _lever_off("T3"):
                _restore_triton_tiles(); _STATE["tiles"] = False
        if _STATE["transition_fused"]:
            def t10():
                W3T = torch.randn(4 * CZ // 2 * 2, CZ, device=dev, generator=g).to(_BF16)[: W12.shape[1] // 2].contiguous()
                _transition_fused(x, W12, W3T, LN_W, LN_B)
            probe("T10", t10)
            if _lever_off("T10"):
                _STATE["transition_fused"] = False
    _STATE["probed"] = True


def disable(model=None):
    o = _STATE["orig"]
    if "transition_forward" in o:
        C.Transition.forward = o["transition_forward"]
        C.PairUpdateBlock._fused_trimul_with_residual = o["pub_fused_trimul"]
        C._fused_trimul_with_residual = o["trimul_fn"]
    _restore_triton_tiles()
    _STATE.update(enabled=False, transition_nolin=False, transition_rowblock=False, tiles=False, weight_cache=False, tx=False, transition_fused=False, probe=False, probed=False)
    _TX.update(on=False, bound=False)
    return describe()


def describe():
    d = _device_info() if torch.cuda.is_available() else _DEV
    out = dict(version=VERSION, enabled=_STATE["enabled"], device=dict(name=d.get("name"), cc=d.get("cc"), smem_kb=(d.get("smem_optin") or 0) // 1024, small_smem_table=bool(d.get("small")), disabled=dict(d.get("disabled") or {})), transition_nolin=_STATE["transition_nolin"], transition_rowblock=_STATE["transition_rowblock"],
               tiles=(dict(_TILES) if _STATE["tiles"] else None), weight_cache=_STATE["weight_cache"], transition_fused=_STATE["transition_fused"], probe=_STATE["probe"])
    out["tx"] = bool(_STATE["tx"] and (_TX["on"] or (_TX["bound"] and not torch.cuda.is_available())))   # lever tx (probe ("w4", "tx")): a provider kernel row serves the pair TriMul by the
    out["tx_state"] = tx_state()                               # bound word in this process (on a device); bound-without-a-device counts as applied like every W4 lever (named); bound with
    return out                                                 # the word naming the stock op = False with tx_state()['why'] (the LEVER line's reason: the upstream statement serves by name)


def stats():
    d = dict(STATS)
    d["identity_probe"] = dict(_IDPROBE["done"])
    d["probe"] = [list(v) for v in PROBES]
    d["probe_all_bitwise"] = all(v[-1] for v in PROBES) if PROBES else None
    d["probe_max_abs_diff"] = max([v[-2] for v in PROBES] or [0.0])
    return d


