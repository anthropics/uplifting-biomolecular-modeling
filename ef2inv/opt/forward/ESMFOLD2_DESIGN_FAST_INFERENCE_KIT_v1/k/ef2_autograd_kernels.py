"""ef2_autograd_kernels — autograd-path fast kernels for Biohub ESMFold2(-Experimental) FoldingTrunk.


Background
----------
The vendored Triton kernels in ``transformers.models.esmfold2.kernels`` (Biohub fork) are gated by
``_fused_active()`` which requires ``not torch.is_grad_enabled()``, so every gradient-based design loop
(``ESMFold2ExperimentalModel.forward(res_type_soft=...)``) runs the reference path:
LN -> proj_bundle -> sigmoid gate -> mask -> fp32 chunked einsum -> LN -> proj_emit * sigmoid(proj_gate),
plus ``x + ffn(norm(x))`` transitions, each PairUpdateBlock wrapped in ``torch.utils.checkpoint``.

The vendored kernels DO ship backward implementations (LN fwd/bwd, gated dual GEMM fwd/bwd, batched
einsum fwd/bwd, output gated GEMM + residual fwd/bwd, LN+Linear+SwiGLU fwd/bwd).  This module

  (1) wires them into an autograd-capable PairUpdateBlock forward ("vendored" trimul mode),
  (2) replaces the weakest stage (stage 3, the O(N^3) triangle contraction, a 64x64x64 bf16 Triton
      tile kernel) by cuBLAS strided-batched GEMMs over the native (D, B, L, L) layout with NO permute
      copies in forward or backward ("bmm" trimul mode; fp32 accumulate, optional fp32 output),
  (3) exposes checkpoint policy for FoldingTrunk under grad ("block" = stock, "none" = keep activations),
  (4) keeps the reference numerics class: bf16 storage, fp32 LayerNorm statistics, fp32 GEMM accumulation.
      Nothing here changes TF32/precision flags.

Usage
-----
    import ef2_autograd_kernels as agk
    agk.enable(model, trimul="bmm", transition="fused", checkpoint="none")   # returns a handle
    ...
    agk.disable(model)

All patched behaviour is opt-in per model instance; unpatched instances (and no-grad calls when
``only_under_grad=True``) run the stock code.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import transformers.models.esmfold2.modeling_esmfold2_common as C

import ef2_t16_transition as _t16          # the carried sm_90a pair-transition forward kernel K-D3 lean can run its forward on (Config.transition_fwd="t16"); steps aside by name elsewhere

try:  # vendored Triton kernels (present whenever triton is importable)
    from transformers.models.esmfold2.kernels.fused_dual_gemm import fused_gated_dual_gemm_split
    from transformers.models.esmfold2.kernels.fused_ln_residual import (
        fused_ln_transpose,
        fused_ln_with_residual_link,
    )
    from transformers.models.esmfold2.kernels.trimul_einsum_triton import trimul_batched_einsum
    from transformers.models.esmfold2.kernels.trimul_with_residual import _gated_gemm_with_residual
    from transformers.models.esmfold2.kernels.fused_lnlin_swiglu import FusedLNLinearSwiGLUFunction
    from transformers.models.esmfold2.kernels import fused_dual_gemm as _FDG
    from transformers.models.esmfold2.kernels import trimul_with_residual as _TWR
    from transformers.models.esmfold2.kernels import fused_lnlin_swiglu as _LSW
    from transformers.models.esmfold2.kernels import fused_ln_residual as _FLR

    KERNELS_OK = True
except Exception as _e:  # pragma: no cover
    KERNELS_OK = False
    _IMPORT_ERR = _e

__all__ = ["enable", "disable", "BmmTriangleContract", "trimul_with_residual", "transition_fused", "Config"]

_HAS_BMM_OUT_DTYPE = None


def _bmm(a: Tensor, b: Tensor, out_dtype: torch.dtype | None = None) -> Tensor:
    """torch.bmm with optional fp32 output for bf16 inputs (cuBLAS computes in fp32 anyway)."""
    global _HAS_BMM_OUT_DTYPE
    if out_dtype is None or out_dtype == a.dtype:
        return torch.bmm(a, b)
    if _HAS_BMM_OUT_DTYPE is None:
        try:
            r = torch.bmm(a, b, out_dtype=out_dtype)
            _HAS_BMM_OUT_DTYPE = True
            return r
        except (TypeError, RuntimeError):
            _HAS_BMM_OUT_DTYPE = False
    if _HAS_BMM_OUT_DTYPE:
        return torch.bmm(a, b, out_dtype=out_dtype)
    return torch.bmm(a, b).to(out_dtype)  # fallback: same values as bf16-out path, then upcast


# ----------------------------------------------------------------------------------------------
# Stage 3: triangle contraction over the native (D, B, L, L) layout via cuBLAS strided-batched GEMM
# ----------------------------------------------------------------------------------------------
class BmmTriangleContract(torch.autograd.Function):
    """out[d,b,i,j] = sum_k a[d,b,i,k] * b[d,b,j,k]   (outgoing;  A @ B^T per (d,b) plane)
       out[d,b,i,j] = sum_k a[d,b,k,i] * b[d,b,k,j]   (incoming;  A^T @ B per plane)

    a, b: (D, B, L, L) bf16 (views of one buffer are fine; only the last dim must be unit-stride).
    Forward and backward are 1 and 2 cuBLAS bgemm calls respectively, all on transposed *views*
    (no .contiguous() copies).  Accumulation is fp32 inside cuBLAS; output dtype selectable.
    """

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, direction: str, out_dtype: torch.dtype | None):  # type: ignore[override]
        D, B, L1, L2 = a.shape
        A = a.reshape(D * B, L1, L2)
        Bm = b.reshape(D * B, L1, L2)
        if direction == "outgoing":
            out = _bmm(A, Bm.transpose(1, 2), out_dtype)
        elif direction == "incoming":
            out = _bmm(A.transpose(1, 2), Bm, out_dtype)
        else:
            raise ValueError(direction)
        ctx.save_for_backward(a, b)
        ctx.direction = direction
        return out.view(D, B, L1, L2)

    @staticmethod
    def backward(ctx, g: Tensor):  # type: ignore[override]
        a, b = ctx.saved_tensors
        D, B, L1, L2 = a.shape
        A = a.reshape(D * B, L1, L2)
        Bm = b.reshape(D * B, L1, L2)
        G = g.reshape(D * B, L1, L2)
        if G.dtype != A.dtype:
            G = G.to(A.dtype)
        if not (G.stride(-1) == 1 or G.stride(-2) == 1):
            G = G.contiguous()
        ga = gb = None
        if ctx.direction == "outgoing":  # O = A B^T :  dA = G B ; dB = G^T A
            if ctx.needs_input_grad[0]:
                ga = torch.bmm(G, Bm).view(D, B, L1, L2)
            if ctx.needs_input_grad[1]:
                gb = torch.bmm(G.transpose(1, 2), A).view(D, B, L1, L2)
        else:  # O = A^T B :  dA = B G^T ; dB = A G
            if ctx.needs_input_grad[0]:
                ga = torch.bmm(Bm, G.transpose(1, 2)).view(D, B, L1, L2)
            if ctx.needs_input_grad[1]:
                gb = torch.bmm(A, G).view(D, B, L1, L2)
        return ga, gb, None, None


def bmm_triangle_contract(a: Tensor, b: Tensor, direction: str, out_dtype: torch.dtype | None = None) -> Tensor:
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return BmmTriangleContract.apply(a, b, direction, out_dtype)  # type: ignore[return-value]
    D, B, L1, L2 = a.shape
    A = a.reshape(D * B, L1, L2)
    Bm = b.reshape(D * B, L1, L2)
    if direction == "outgoing":
        return _bmm(A, Bm.transpose(1, 2), out_dtype).view(D, B, L1, L2)
    return _bmm(A.transpose(1, 2), Bm, out_dtype).view(D, B, L1, L2)


# ----------------------------------------------------------------------------------------------
# K-A2: frozen-weight ("design mode") variants of the vendored autograd Functions.
# The vendored backward kernels always compute weight gradients (dW = grad^T @ x, an (M x N x K) GEMM per
# stage, plus the LN gamma/beta reductions) even when the weights are frozen, which is the case in every
# design loop.  These wrappers call the same Triton elementwise/backward kernels but skip every reduction whose
# consumer does not require grad (decided from ctx.needs_input_grad), so the input gradient is bit-identical to
# the vendored Functions' input gradient.
# ----------------------------------------------------------------------------------------------
import triton as _triton


class GatedDualGemmSplitFrozen(torch.autograd.Function):
    """(a, b_t) = split(fused_gated_dual_gemm(x, w1, w2, mask, transpose_out=True)); backward: dx (and dW only if needed)."""

    @staticmethod
    def forward(ctx, x, w1, w2, mask, trailing_shape, stacked_w21=None):
        out = _FDG._fused_gated_dual_gemm_fwd(x, w1, w2, mask=mask, transpose_out=True)
        N = w1.shape[0]; half_n = N // 2
        out_view = out.view((N,) + trailing_shape)
        a = out_view[:half_n]; b_t = out_view[half_n:]
        ctx.save_for_backward(x, w1, w2, *( (mask,) if mask is not None else () ))
        ctx.has_mask = mask is not None
        ctx.stacked_w = stacked_w21          # non-differentiable constant (frozen weights), not saved-for-backward
        return a, b_t

    @staticmethod
    def backward(ctx, grad_a, grad_b_t):
        if ctx.has_mask:
            x, w1, w2, mask = ctx.saved_tensors
        else:
            x, w1, w2 = ctx.saved_tensors; mask = None
        if not grad_a.is_contiguous(): grad_a = grad_a.contiguous()
        if not grad_b_t.is_contiguous(): grad_b_t = grad_b_t.contiguous()
        need_w = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        if need_w:
            d_x, d_w1, d_w2, d_mask = _FDG._fused_gated_dual_gemm_bwd(None, x, w1, w2, mask, grad_out_transposed=True, grad_out_split=(grad_a, grad_b_t))
            return d_x, d_w1, d_w2, None, None, None
        # --- input-grad-only path: same Triton bwd kernel for the per-element partials, then ONE GEMM for dx
        in_shape = x.shape; K = in_shape[-1]
        x_2d = x.contiguous().view(-1, K); M = x_2d.shape[0]; N = w1.shape[0]
        mask_flat = mask.contiguous().view(-1) if mask is not None else None
        grad_combined = torch.empty((M, 2 * N), device=x.device, dtype=torch.bfloat16)
        grad_val_acc = grad_combined[:, :N]; grad_gate_logits = grad_combined[:, N:]
        tiles_n_max = _triton.cdiv(N, _FDG._BWD_TILE_N)
        grad_mask_partials = torch.empty((tiles_n_max, M) if mask is not None else (1,), device=x.device, dtype=torch.float32)
        _dummy_mask = torch.zeros((), device=x.device, dtype=torch.bfloat16)
        NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * (2 * N) >= 2**31 - 1)
        def grid(meta):
            return (_triton.cdiv(M, meta["TILE_M"]), N // meta["TILE_N"])
        ga2 = grad_a.view(N // 2, M); gb2 = grad_b_t.view(N // 2, M)
        _FDG._gated_dual_gemm_backward_kernel[grid](
            ga2, gb2, x_2d, w1.contiguous(), w2.contiguous(),
            mask_flat if mask_flat is not None else _dummy_mask,
            grad_gate_logits, grad_val_acc, grad_mask_partials, M, N, K,
            HALF_N=N // 2, HAS_MASK=mask is not None, GRAD_OUT_TRANSPOSED=True, GRAD_OUT_SPLIT=True, NEEDS_INT64=NEEDS_INT64,
        )
        stacked_w = ctx.stacked_w if ctx.stacked_w is not None else torch.cat([w2, w1], dim=0)
        d_x = (grad_combined @ stacked_w).view(in_shape)
        return d_x, None, None, None, None, None


class GatedGemmResidualFrozen(torch.autograd.Function):
    """out = residual + sigmoid(x1 @ w1^T) * (x2 @ w2^T); backward returns d_x1, d_x2, d_residual (dW only if needed)."""

    @staticmethod
    def forward(ctx, x1, x2, w1, w2, residual, n_row, n_col):
        out = _TWR._gated_gemm_with_residual_fwd(x1, x2, w1, w2, residual, None, n_row, n_col, precision=0)
        ctx.save_for_backward(x1, x2, w1, w2); ctx.n_row = n_row; ctx.n_col = n_col
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x1, x2, w1, w2 = ctx.saved_tensors
        need_w = ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
        if need_w:
            d_x1, d_x2, d_w1, d_w2, d_res, _ = _TWR._gated_gemm_with_residual_bwd(grad_out.contiguous(), x1, x2, w1, w2, None, ctx.n_row, ctx.n_col)
            return d_x1, d_x2, d_w1, d_w2, d_res, None, None
        M, K = x1.shape; N = w1.shape[0]
        go = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        go2 = go.view(M, N)
        x1_c = x1 if x1.is_contiguous() else x1.contiguous(); x2_c = x2 if x2.is_contiguous() else x2.contiguous()
        w1_c = w1.contiguous(); w2_c = w2.contiguous()
        gg = torch.empty((M, N), device=x1.device, dtype=torch.bfloat16)
        gv = torch.empty((M, N), device=x1.device, dtype=torch.bfloat16)
        gdb = torch.empty((1,), device=x1.device, dtype=torch.float32)
        _dummy_mask = torch.zeros((), device=x1.device, dtype=torch.bfloat16)
        NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)
        def grid(meta):
            return (_triton.cdiv(M, meta["TILE_M"]), _triton.cdiv(N, meta["TILE_N"]))
        _TWR._gated_gemm_with_residual_backward_kernel[grid](
            go2, x1_c, x2_c, w1_c, w2_c, _dummy_mask, gg, gv, gdb, M, N, K,
            STRIDE_B=ctx.n_row * ctx.n_col, N_COL=ctx.n_col, HAS_DROP_MASK=False, NEEDS_INT64=NEEDS_INT64,
        )
        d_x1 = gg @ w1_c
        d_x2 = gv @ w2_c
        return d_x1, d_x2, None, None, go2, None, None


class LNLinearSwiGLUFrozen(torch.autograd.Function):
    """hidden = silu(x1)*x2, (x1,x2)=chunk(LN(X) @ W12); backward: dX only when W12/LN params are frozen."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, X, W12, LN_W, LN_B):
        x_shape = X.shape
        x_2d = X.contiguous().view(-1, x_shape[-1])
        out, lin, mean, rstd = _LSW._lnlin_swiglu_fwd(x_2d, W12, LN_W, LN_B)
        ctx.save_for_backward(x_2d, W12, LN_W, LN_B, mean, rstd, lin)
        ctx.x_shape = x_shape; ctx.has_ln_bias = LN_B is not None
        return out.view(*x_shape[:-1], out.shape[-1])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        x_2d, W12, LN_W, LN_B, mean, rstd, lin = ctx.saved_tensors
        dout_2d = dout.contiguous().view(-1, dout.shape[-1]); K = x_2d.shape[1]
        dlin = _LSW._swiglu_bwd_inplace(dout_2d, lin)
        need_w = ctx.needs_input_grad[1]
        need_ln = ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
        dW12 = None
        if need_w:
            x_norm = F.layer_norm(x_2d, (K,), LN_W, LN_B, eps=1e-5)
            dW12 = x_norm.transpose(0, 1) @ dlin
            del x_norm
        dx_norm = dlin @ W12.transpose(0, 1)
        output_mask = [True, bool(need_ln), bool(need_ln and ctx.has_ln_bias)]
        dX, dLN_W, dLN_B = torch.ops.aten.native_layer_norm_backward(dx_norm, x_2d, [K], mean.float(), rstd.float(), LN_W, LN_B, output_mask)
        return dX.view(ctx.x_shape), dW12, dLN_W, dLN_B


# ----------------------------------------------------------------------------------------------
# dx-only LayerNorm backward (frozen gamma/beta): launches the VENDORED _ln_bwd_kernel (identical compiled code ->
# bitwise-identical dx) but skips the two host-side reductions of the per-tile dgamma/dbeta partials
# (grad_w_partial.sum(0), grad_b_partial.sum(0): 2 extra kernel launches per LN backward, ~240 per design step).
# A separate dx-only Triton kernel was tried first: NOT bitwise-equal to the vendored dx (different reduction layout
# chosen by the compiler when the axis-0 partial sums are absent) -> rejected to keep the shipped agk3 digits.
# ----------------------------------------------------------------------------------------------
def _ln_bwd_dx_only(grad_y, x_view, w, mean, rstd, layout_int, grad_residual, M, D):
    """dx of LayerNorm, bitwise-equal to _FLR._ln_bwd(...)[0]; dgamma/dbeta partials are computed by the kernel
    (cheap, in-register) but never reduced."""
    import triton as _tt
    if layout_int == _FLR._LAYOUT_BND_BND:
        grad_x = torch.empty((M, D), device=x_view.device, dtype=x_view.dtype)
    else:
        grad_x = torch.empty((D, M), device=x_view.device, dtype=x_view.dtype)
    num_tiles = _tt.cdiv(M, _FLR._BWD_TILE_M)
    part = torch.empty((2, num_tiles, D), device=x_view.device, dtype=torch.float32)     # scratch for the partials
    has_residual = grad_residual is not None
    _dummy = torch.empty((), device=x_view.device, dtype=grad_y.dtype)
    _FLR._ln_bwd_kernel[(num_tiles,)](
        grad_y, x_view, w, mean, rstd, grad_x, part[0], part[1], grad_residual if has_residual else _dummy, M,
        D=D, LAYOUT=layout_int, HAS_RESIDUAL=has_residual, TILE_M=_FLR._BWD_TILE_M,
        num_warps=_FLR._BWD_NUM_WARPS, num_stages=_FLR._BWD_NUM_STAGES,
    )
    return grad_x
_HAS_LN_DX = True
LN_DX_ONLY = True     # module switch (unit test flips it to compare against the vendored 3-output backward)
_kd3_w12_swiglu = None   # K-D3 lean's forward hook: ef2_kd3_gemmswiglu.engage() binds its fused W12-projection + SwiGLU kernel here on a card with an entry (sm_80) — hidden in ONE kernel, the [M, 2h] pre-activation never written; None = cuBLAS W12 + the SwiGLU kernel. Read per call (TransitionRefround.forward, lean only); bind before any trunk graph capture.


class LNFrozen(torch.autograd.Function):
    """LayerNorm via the vendored Triton LN kernels with optional layout change and residual-link grad folding,
    skipping the gamma/beta reductions when they are frozen.  layouts: 'bijd->bijd' (with residual link) or 'dbij->bijd'."""

    @staticmethod
    def forward(ctx, x, w, b, eps, layout, with_link):
        layout_int = _FLR._layout_to_int(layout)
        out_shape, M, D, x_view = _FLR._reshape_for_layout(x, layout)
        out_bnd, mean, rstd = _FLR._ln_fwd(x_view, w, b, eps, layout_int, M, D)
        ctx.save_for_backward(x_view, w, mean, rstd)
        ctx.layout_int = layout_int; ctx.M = M; ctx.D = D; ctx.x_orig_shape = x.shape; ctx.with_link = with_link
        ln_out = out_bnd.view(*out_shape)
        if with_link:
            return ln_out, x.view_as(x)
        return ln_out

    @staticmethod
    def backward(ctx, grad_ln_out, grad_link=None):
        x_view, w, mean, rstd = ctx.saved_tensors
        grad_y = grad_ln_out.contiguous().view(ctx.M, ctx.D)
        grad_residual = grad_link.contiguous().view(ctx.M, ctx.D) if (ctx.with_link and grad_link is not None) else None
        if LN_DX_ONLY and _HAS_LN_DX and not (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            grad_x = _ln_bwd_dx_only(grad_y, x_view, w, mean, rstd, ctx.layout_int, grad_residual, ctx.M, ctx.D)
            return grad_x.view(*ctx.x_orig_shape), None, None, None, None, None
        grad_x, grad_w, grad_b = _FLR._ln_bwd(grad_y, x_view, w, mean, rstd, ctx.layout_int, grad_residual, ctx.M, ctx.D)
        if not ctx.needs_input_grad[1]: grad_w = None
        if not ctx.needs_input_grad[2]: grad_b = None
        return grad_x.view(*ctx.x_orig_shape), grad_w, grad_b, None, None, None


def trimul_with_residual_frozen(pair, engine, mask, w, einsum_out_dtype=None):
    """K-A2: same stages as trimul_with_residual but with the frozen-weight Functions (no dW work in backward)."""
    B, L1, L2, cz = pair.shape
    if not (torch.is_grad_enabled() and pair.requires_grad):
        return trimul_with_residual(pair, engine, mask, w, "bmm", einsum_out_dtype)
    x_in, residual_alias = LNFrozen.apply(pair, w["norm_in_w"], w["norm_in_b"], C._EPS, "bijd->bijd", True)
    if mask is not None and mask.dtype != torch.bfloat16:
        mask = mask.to(torch.bfloat16)
    a, b_t = GatedDualGemmSplitFrozen.apply(x_in, w["g_in"], w["p_in"], mask, (B, L1, L2), w["gp_in_stacked"])
    x = bmm_triangle_contract(a, b_t, engine.flow, out_dtype=einsum_out_dtype)
    x_out = LNFrozen.apply(x, w["norm_out_w"], w["norm_out_b"], C._EPS, "dbij->bijd", False)
    if x_out.dtype != torch.bfloat16:
        x_out = x_out.to(torch.bfloat16)
    out_2d = GatedGemmResidualFrozen.apply(x_in.reshape(-1, cz), x_out.reshape(-1, cz), w["g_out"], w["p_out"],
                                           residual_alias.reshape(-1, cz), L1, L2)
    return out_2d.view(B, L1, L2, cz)


def transition_fused_frozen(x, tr, w):
    hidden = LNLinearSwiGLUFrozen.apply(x, w["W12t"], w["LN_W"], w["LN_B"])
    x2 = x.contiguous().view(-1, x.shape[-1])
    out = torch.addmm(x2, hidden.view(-1, hidden.shape[-1]), w["W3t"])
    return out.view(x.shape)


# ----------------------------------------------------------------------------------------------
# Full TriMul + residual under autograd
# ----------------------------------------------------------------------------------------------
def _weights(engine: "C.TriangleMultiplicativeBlock", cache: dict, ln_params: str) -> dict:
    """bf16 copies of the projection weights (cast once, cached); LN params fp32 (reference) or bf16."""
    key = id(engine)
    ent = cache.get(key)
    ver = tuple(int(p._version) for p in engine.parameters())
    if ent is not None and ent["ver"] == ver:
        return ent
    p_in, g_in = engine.split_kernel_weights()
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()
    lnp = (lambda t: t.detach().float().contiguous()) if ln_params == "fp32" else bf
    ent = dict(
        ver=ver,
        norm_in_w=lnp(engine.norm_start.weight), norm_in_b=lnp(engine.norm_start.bias),
        p_in=bf(p_in), g_in=bf(g_in),
        norm_out_w=lnp(engine.norm_mix.weight), norm_out_b=lnp(engine.norm_mix.bias),
        p_out=bf(engine.proj_emit.weight), g_out=bf(engine.proj_gate.weight),
    )
    ent["gp_in_stacked"] = torch.cat([ent["p_in"], ent["g_in"]], dim=0).contiguous()   # (2N, K) = [w2; w1] order used by the dual-GEMM bwd
    ent["engine_ref"] = engine          # keep the engine alive so id(engine) cannot be recycled while cached
    cache[key] = ent
    return ent


def trimul_with_residual(pair: Tensor, engine: "C.TriangleMultiplicativeBlock", mask: Tensor | None,
                         w: dict, einsum_impl: str = "bmm", einsum_out_dtype: torch.dtype | None = None) -> Tensor:
    """pair_new = pair + TriMul(pair)  (row_drop r=0 / eval), autograd-capable, bf16 in/out.

    Stages: (1) LN_in with residual-link (bwd folds grad_residual into grad_x), (2) fused gated dual GEMM
    -> (a, b_t) in (D,B,L,L) layout, (3) triangle contraction [bmm: cuBLAS | vendored: Triton tile kernel],
    (4) LN_out reading the (D,B,L,L) layout and writing (B,L,L,D), (5) output gated GEMM + residual add.
    """
    assert pair.dtype == torch.bfloat16, pair.dtype
    B, L1, L2, cz = pair.shape
    direction = engine.flow
    x_in, residual_alias = fused_ln_with_residual_link(pair, w["norm_in_w"], w["norm_in_b"], pair,
                                                       eps=C._EPS, layout="bijd->bijd")
    if mask is not None and mask.dtype != torch.bfloat16:
        mask = mask.to(torch.bfloat16)
    a, b_t = fused_gated_dual_gemm_split(x_in, w["g_in"], w["p_in"], mask=mask)
    if einsum_impl == "bmm":
        x = bmm_triangle_contract(a, b_t, direction, out_dtype=einsum_out_dtype)
    elif einsum_impl == "vendored":
        x = trimul_batched_einsum(a, b_t, direction)
    else:
        raise ValueError(einsum_impl)
    if x.dtype != torch.bfloat16:
        # fp32 contraction output: LN in fp32 (reference-like), then bf16 for the gated GEMM
        x_out = fused_ln_transpose(x, w["norm_out_w"], w["norm_out_b"], eps=C._EPS, layout="dbij->bijd").to(torch.bfloat16)
    else:
        x_out = fused_ln_transpose(x, w["norm_out_w"], w["norm_out_b"], eps=C._EPS, layout="dbij->bijd")
    out_2d = _gated_gemm_with_residual(
        x_in.reshape(-1, cz), x_out.reshape(-1, cz), w["g_out"], w["p_out"],
        residual_alias.reshape(-1, cz), None, n_row=L1, n_col=L2, precision=0,
    )
    return out_2d.view(B, L1, L2, cz)


# ----------------------------------------------------------------------------------------------
# Transition: fused LN + w12 + SwiGLU (Triton fwd, cuBLAS/ATen bwd) + addmm residual
# ----------------------------------------------------------------------------------------------
def _transition_weights(tr: "C.Transition", cache: dict) -> dict:
    key = id(tr)
    ver = tuple(int(p._version) for p in tr.parameters())
    ent = cache.get(key)
    if ent is not None and ent["ver"] == ver:
        return ent
    ent = dict(
        ver=ver,
        W12t=tr.ffn.w12.weight.detach().t().contiguous().to(torch.bfloat16),   # (d, 2h)
        LN_W=tr.norm.weight.detach().to(torch.bfloat16).contiguous(),
        LN_B=(tr.norm.bias.detach().to(torch.bfloat16).contiguous() if tr.norm.bias is not None else None),
        W3t=tr.ffn.w3.weight.detach().t().contiguous().to(torch.bfloat16),      # (h, d)
    )
    ent["module_ref"] = tr             # keep the module alive so id(tr) cannot be recycled while cached
    cache[key] = ent
    return ent


def transition_fused(x: Tensor, tr: "C.Transition", w: dict) -> Tensor:
    assert x.dtype == torch.bfloat16
    hidden = FusedLNLinearSwiGLUFunction.apply(x, w["W12t"], w["LN_W"], w["LN_B"])  # [..., h] bf16
    x2 = x.contiguous().view(-1, x.shape[-1])
    out = torch.addmm(x2, hidden.view(-1, hidden.shape[-1]), w["W3t"])
    return out.view(x.shape)


# ----------------------------------------------------------------------------------------------
# Patching
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    trimul: str | None = "bmm"          # "bmm" | "bmm2" (frozen-weight fast bwd) | "vendored" | None (stock)
    transition: str | None = None       # "fused" | "fused2" (frozen-weight fast bwd) | "refround" (K-D3: stock rounding points) | "refround_lean" (K-D3, recompute LN+W12 in bwd; less memory) | None (stock)
    transition_fwd: str | None = None   # "t16" = K-D3 lean's FORWARD runs on ef2_t16_transition's one-kernel forward where that kernel is live (sm_90; K-D3's backward unchanged: it recomputes LN + W12 itself); None = K-D3's own forward kernels. A device / stack the kernel cannot serve keeps K-D3's forward BY NAME (ef2_t16_transition.describe)
    checkpoint: str = "block"           # "block" (stock) | "none"
    ln_params: str = "fp32"             # dtype of LN affine params handed to the Triton LN kernels
    einsum_out_dtype: str = "bf16"      # "bf16" (vendored-like) | "fp32" (reference-like LN_out input)
    only_under_grad: bool = True        # leave no-grad calls (confidence trunk, CLI folds) on the stock path
    cache: dict = field(default_factory=dict)


def _pub_forward(self: "C.PairUpdateBlock", pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
    cfg: Config | None = getattr(self, "_agk_cfg", None)
    use = (
        cfg is not None and KERNELS_OK and pair.is_cuda and pair.dtype == torch.bfloat16
        and (cfg.trimul is not None or cfg.transition is not None)
        and (torch.is_grad_enabled() or not cfg.only_under_grad)
        and not (self.training and getattr(self.row_drop, "_r", 0.0) > 0.0)
    )
    if not use:
        return self._agk_orig_forward(pair, pair_attention_mask=pair_attention_mask)
    od = torch.float32 if cfg.einsum_out_dtype == "fp32" else None
    if cfg.trimul is not None:
        for tri in (self.tri_mul_out, self.tri_mul_in):
            eng = tri._engine
            w = _weights(eng, cfg.cache, cfg.ln_params)
            if cfg.trimul == "bmm2":
                pair = trimul_with_residual_frozen(pair, eng, pair_attention_mask, w, einsum_out_dtype=od)
            else:
                pair = trimul_with_residual(pair, eng, pair_attention_mask, w, einsum_impl=cfg.trimul, einsum_out_dtype=od)
    else:
        pair = self.row_drop(pair, self.tri_mul_out(pair, mask=pair_attention_mask))
        pair = self.row_drop(pair, self.tri_mul_in(pair, mask=pair_attention_mask))
    if cfg.transition == "fused":
        pair = transition_fused(pair, self.pair_transition, _transition_weights(self.pair_transition, cfg.cache))
    elif cfg.transition == "fused2":
        pair = transition_fused_frozen(pair, self.pair_transition, _transition_weights(self.pair_transition, cfg.cache))
    elif cfg.transition == "refround":
        pair = transition_refround(pair, self.pair_transition, cfg.cache)
    elif cfg.transition == "refround_lean":      # same numerics as refround (bitwise), ~25% less activation memory per kept block, + 1 GEMM recompute per block in bwd; fwd="t16": the forward on the carried one-kernel transition where live
        pair = transition_refround(pair, self.pair_transition, cfg.cache, lean=True, fwd=cfg.transition_fwd)
    else:
        pair = self.pair_transition(pair)
    return pair


def _trunk_forward(self: "C.FoldingTrunk", pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
    cfg: Config | None = getattr(self, "_agk_cfg", None)
    mode = cfg.checkpoint if cfg is not None else "block"
    orig_dtype = pair.dtype
    fused_on = len(self.blocks) > 0 and getattr(self.blocks[0], "_kernel_backend", None) == C.BACKEND_FUSED
    if pair.is_cuda and fused_on and orig_dtype != torch.bfloat16:
        pair = pair.to(torch.bfloat16)
    nb = len(self.blocks)
    ckpt_fn = self.__dict__.get("_agk_ckpt_fn")             # a lever's checkpoint (ef2_trimul_nosave: no-save first pass); None = torch's non-reentrant
    for i, block in enumerate(self.blocks):
        fn = partial(block, pair_attention_mask=pair_attention_mask)
        if torch.is_grad_enabled() and _ckpt_this_block(mode, i, nb):
            pair = ckpt_fn(block, pair, pair_attention_mask) if ckpt_fn is not None else checkpoint(fn, pair, use_reentrant=False)
        else:
            pair = fn(pair)
    if pair.dtype != orig_dtype:
        pair = pair.to(orig_dtype)
    return pair


def _ckpt_this_block(mode: str, i: int, nb: int) -> bool:
    """checkpoint policy: 'block' (all, stock) | 'none' | 'every:k' (checkpoint blocks whose index % k != 0,
    i.e. keep activations of every k-th block) | 'first:m' (checkpoint the first m blocks, keep the last nb-m) |
    'keep:m' (keep activations of m blocks spread evenly, checkpoint the rest)."""
    if mode == "block":
        return True
    if mode == "none":
        return False
    kind, _, arg = mode.partition(":")
    m = int(arg) if arg else 0
    if kind == "every":      # keep 1 of every m blocks un-checkpointed
        return (i % max(m, 1)) != 0
    if kind == "first":      # checkpoint blocks [0, m)
        return i < m
    if kind == "ckpt":       # checkpoint exactly m blocks (the first m); keep the rest
        return i < m
    if kind == "keep":       # keep m blocks (evenly spread) without checkpoint
        if m <= 0: return True
        keep = {round(j * (nb - 1) / max(m - 1, 1)) for j in range(m)} if m > 1 else {nb - 1}
        return i not in keep
    raise ValueError(mode)


def enable(model: torch.nn.Module, trimul: str | None = "bmm", transition: str | None = None,
           checkpoint: str = "block", ln_params: str = "fp32", einsum_out_dtype: str = "bf16",
           only_under_grad: bool = True, trunks: str = "all", transition_fwd: str | None = None) -> Config:
    """Patch every FoldingTrunk / PairUpdateBlock in `model` (instance-level; reversible via disable()).  A second enable on a patched
    model installs the new Config WITHOUT rebinding the forwards (they read the module's `_agk_cfg` per call), so a wrapper another lever
    stacked on the patched trunk forward after the first enable (a CUDA-graph pool) stays outermost and engaged.
    ``transition_fwd="t16"`` (with ``transition="refround_lean"``): K-D3 lean's forward runs on ef2_t16_transition's kernel where it is live —
    engaged here once per process (before any capture) and the blocks' weight entries + operand descriptors built now, so nothing is packed inside
    a CUDA-graph capture; where the kernel cannot serve, K-D3's own forward runs and ef2_t16_transition.describe() names why."""
    if (trimul is not None or transition is not None) and not KERNELS_OK:
        raise RuntimeError(f"vendored Triton kernels unavailable: {_IMPORT_ERR!r}")
    if transition_fwd not in (None, "t16"):
        raise ValueError(f"ef2_autograd_kernels.enable: transition_fwd {transition_fwd!r} is not None | 't16'")
    if transition_fwd is not None and transition != "refround_lean":
        raise ValueError(f"ef2_autograd_kernels.enable: transition_fwd={transition_fwd!r} rides on transition='refround_lean' (asked {transition!r})")
    cfg = Config(trimul=trimul, transition=transition, checkpoint=checkpoint, ln_params=ln_params,
                 einsum_out_dtype=einsum_out_dtype, only_under_grad=only_under_grad, transition_fwd=transition_fwd)
    if transition_fwd == "t16":
        _t16.engage()                                 # idempotent; records (never raises) a device / stack that cannot serve
    for name, m in model.named_modules():
        if isinstance(m, C.FoldingTrunk):
            if trunks == "main" and "confidence" in name:
                continue
            m._agk_cfg = cfg
            if not hasattr(m, "_agk_orig_forward"):
                m._agk_orig_forward = m.forward
                m.forward = types.MethodType(_trunk_forward, m)
        if isinstance(m, C.PairUpdateBlock):
            if trunks == "main" and "confidence" in name:
                continue
            m._agk_cfg = cfg
            if not hasattr(m, "_agk_orig_forward"):
                m._agk_orig_forward = m.forward
                m.forward = types.MethodType(_pub_forward, m)
            if transition_fwd == "t16" and _t16.live() and _KD3_OK and m.pair_transition.ffn.w12.weight.is_cuda:
                _t16.pack_kd3(_transition_weights3(m.pair_transition, cfg.cache), C._EPS)   # K-D3's bf16 weights + the kernel's descriptors over them, before any capture
    model._agk_cfg = cfg
    return cfg


def disable(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "_agk_orig_forward"):
            m.forward = m._agk_orig_forward
            del m._agk_orig_forward
        if hasattr(m, "_agk_cfg"):
            del m._agk_cfg


def describe(cfg: Config | None) -> str:
    if cfg is None:
        return "stock"
    fwd = f" fwd={cfg.transition_fwd}" if getattr(cfg, "transition_fwd", None) else ""
    return f"trimul={cfg.trimul} transition={cfg.transition}{fwd} ckpt={cfg.checkpoint} ln={cfg.ln_params} eo={cfg.einsum_out_dtype}"


# ----------------------------------------------------------------------------------------------
# Exact usage-level lever: honour calculate_confidence=False
# ----------------------------------------------------------------------------------------------
def enable_skip_unused_confidence(model) -> None:
    """ESMFold2ExperimentalModel.forward accepts `calculate_confidence` only through **kwargs and ignores it: the
    confidence head (a full pairformer stack under no_grad; 24 ms @205 tok ... 303 ms @705 tok on H100) runs on every
    design step although the design loss never reads plddt/pae/iptm (the ESM cookbook passes
    calculate_confidence=False during optimisation and only True for the final scoring fold).
    This patch makes forward() skip the confidence head iff the caller passed calculate_confidence=False.  All other
    outputs are bitwise unchanged (the head consumes detached tensors and feeds nothing back); calls that do not pass
    the flag, or pass True, behave exactly as stock."""
    if hasattr(model, "_agk_orig_model_forward"):
        return
    orig = model.forward
    def fwd(*a, **kw):
        cc = kw.pop("calculate_confidence", None)
        if cc is False and getattr(model, "confidence_head", None) is not None:
            ch = model.confidence_head
            try:
                model.confidence_head = None       # nn.Module __setattr__ allows None for a registered submodule
                return orig(*a, **kw)
            finally:
                model.confidence_head = ch
        return orig(*a, **kw)
    model._agk_orig_model_forward = orig
    model.forward = fwd


def disable_skip_unused_confidence(model) -> None:
    if hasattr(model, "_agk_orig_model_forward"):
        model.forward = model._agk_orig_model_forward
        del model._agk_orig_model_forward


# ----------------------------------------------------------------------------------------------
# K-D3 "refround": Transition fwd+bwd with the STOCK rounding points
# ----------------------------------------------------------------------------------------------
# Stock (bf16 autocast, weights frozen):  x_hat = bf16( LN_fp32(x) ) ; x12 = bf16(x_hat @ W12ᵀ) ; s = bf16(silu(x1)) ;
# h = bf16(s * x2) ; y = bf16(h @ W3ᵀ) ; out = bf16(x + y).  K-D2 (vendored FusedLNLinearSwiGLU) instead keeps LN stats,
# LN affine and the SwiGLU epilogue in reduced/other precision (bf16 mean/rstd, silu·mul from fp32 accumulators, addmm) —
# 'rounding points changed'.  K-D3 reproduces every stock rounding point: LN by the vendored Triton LN kernel (fp32 stats,
# bf16 out), both GEMMs by the same cuBLAS calls the stock nn.Linear makes (same operand dtypes/layouts), SwiGLU fwd/bwd by
# two elementwise Triton kernels that round exactly where eager PyTorch rounds, residual add in bf16.  Remaining differences
# vs stock are fp32-ulp level (LN mean/var reduction order; libdevice exp) => class 'reordered accumulation, same rounding
# points'.  Speed comes from: no fp32 LN round trip + casts, one pass for silu*mul (fwd) and its backward (in place, no cat),
# no dW / dgamma / dbeta GEMMs, fewer autograd nodes.
try:
    import triton, triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld       # triton >= 3.0
        _EXP = "libdevice"
    except Exception:                                             # pragma: no cover
        try:
            from triton.language.extra.cuda import libdevice as _ld
            _EXP = "libdevice(cuda)"
        except Exception:
            _ld = None; _EXP = "tl.exp"
    _HAS_DIV_RN = tl.constexpr(bool(hasattr(tl, "math") and hasattr(tl.math, "div_rn")))
    _HAS_LD = tl.constexpr(_ld is not None)

    @triton.jit
    def _k_exp(x):
        if _HAS_LD:
            return _ld.exp(x)
        else:
            return tl.exp(x)

    @triton.jit
    def _k_div(a, b):
        if _HAS_DIV_RN:
            return tl.math.div_rn(a, b)
        else:
            return a / b

    @triton.jit
    def _swiglu_refround_fwd_kernel(lin_ptr, out_ptr, N, stride_l, stride_o, BLOCK: tl.constexpr):
        # lin = [M, 2N] bf16 (x1 | x2) ; out[M, N] = bf16( f32(bf16(silu(x1))) * f32(x2) )   -- exactly eager PyTorch's rounding
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK); m = cols < N
        a = tl.load(lin_ptr + row * stride_l + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(lin_ptr + row * stride_l + N + cols, mask=m, other=0.0).to(tl.float32)
        s = _k_div(a, 1.0 + _k_exp(-a))                 # ATen silu: x / (1 + exp(-x)) in opmath fp32
        s_bf = s.to(tl.bfloat16).to(tl.float32)          # F.silu output is a bf16 tensor
        h = s_bf * b                                      # bf16*bf16 -> opmath fp32 -> bf16
        tl.store(out_ptr + row * stride_o + cols, h.to(tl.bfloat16), mask=m)

    @triton.jit
    def _swiglu_refround_bwd_kernel(dh_ptr, lin_ptr, N, stride_d, stride_l, BLOCK: tl.constexpr):
        # in place: lin[:, :N] <- d_x1 , lin[:, N:] <- d_x2 ; with eager's rounding:
        #   d_s  = bf16(dh * x2) ; d_x2 = bf16(dh * s_bf) ; d_x1 = bf16( d_s * sig * (1 + x1*(1 - sig)) )   (ATen silu_backward)
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK); m = cols < N
        a = tl.load(lin_ptr + row * stride_l + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(lin_ptr + row * stride_l + N + cols, mask=m, other=0.0).to(tl.float32)
        dh = tl.load(dh_ptr + row * stride_d + cols, mask=m, other=0.0).to(tl.float32)
        s = _k_div(a, 1.0 + _k_exp(-a))
        s_bf = s.to(tl.bfloat16).to(tl.float32)
        d_s = (dh * b).to(tl.bfloat16).to(tl.float32)
        d_b = (dh * s_bf).to(tl.bfloat16)
        sig = _k_div(1.0, 1.0 + _k_exp(-a))
        d_a = (d_s * sig * (1.0 + a * (1.0 - sig))).to(tl.bfloat16)
        tl.store(lin_ptr + row * stride_l + cols, d_a, mask=m)
        tl.store(lin_ptr + row * stride_l + N + cols, d_b, mask=m)
    _KD3_OK = True
except Exception as _e3:   # pragma: no cover
    _KD3_OK = False; _EXP = f"unavailable: {_e3}"


class TransitionRefround(torch.autograd.Function):
    """out = x + W3( swiglu( W12( LN(x) ) ) ) with stock rounding points; frozen weights (returns dX only)."""

    @staticmethod
    def forward(ctx, x, w):
        shape = x.shape; d = shape[-1]
        x2d = x.contiguous().view(-1, d); M = x2d.shape[0]
        layout_int = _FLR._layout_to_int("bijd->bijd")
        lean = bool(w.get("lean", False))
        fwd = w.get("fwd") if lean else None                 # lean only: an out-only forward kernel (ef2_t16_transition where live) — one launch, no [M,2h] / [M,h] transients; the backward recomputes LN (+ its statistics) and W12 as lean always does
        h = w["W12"].shape[0] // 2
        if fwd is not None:
            out = fwd(x2d)                                   # [M, d] bf16 = x + W3(swiglu(W12(LN(x)))) in one kernel (fp32 LN statistics, bf16 operands, fp32 accumulation)
            ctx.save_for_backward(x2d)
        else:
            xhat, mean, rstd = _FLR._ln_fwd(x2d, w["LN_W32"], w["LN_B32"], C._EPS, layout_int, M, d)     # bf16 [M,d], fp32 stats
            gsw = _kd3_w12_swiglu if lean else None             # lean only: ef2_kd3_gemmswiglu's W12 projection with the SwiGLU as its epilogue where that lever engaged (a card with an entry: sm_80) — hidden [M, h] in one kernel, lin never written (lean saves no lin); None from it = an entry it does not take, counted by name there
            hidden = gsw(xhat, w) if gsw is not None else None
            if hidden is None:
                lin = F.linear(xhat, w["W12"])                   # [M, 2h] bf16  (same cuBLAS call as nn.Linear under autocast)
                hidden = torch.empty((M, h), dtype=lin.dtype, device=lin.device)
                BLOCK = triton.next_power_of_2(h)
                _swiglu_refround_fwd_kernel[(M,)](lin, hidden, h, lin.stride(0), hidden.stride(0), BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8)
            y = F.linear(hidden, w["W3"])                        # [M, d] bf16
            out = x2d + y                                        # bf16 add (stock: x + ffn(...))
            if lean:      # memory-lean: save only the block input + LN stats (M*(d*2+8) B) and recompute LN-out + W12 GEMM in backward
                ctx.save_for_backward(x2d, mean, rstd)
            else:         # fast: also keep lin (M*2h*2 B = 8x the input at h=4d) to avoid the recompute GEMM
                ctx.save_for_backward(x2d, mean, rstd, lin)
        ctx.lean = lean; ctx.fwd_kernel = fwd is not None
        ctx.w = w; ctx.shape = shape; ctx.M = M; ctx.d = d; ctx.h = h; ctx.layout_int = layout_int
        return out.view(shape)

    @staticmethod
    def backward(ctx, g):
        w = ctx.w
        if ctx.lean:
            if ctx.fwd_kernel:                               # the one-kernel forward kept no LN statistics: the recompute below IS K-D3's LN kernel on the same input, its mean / rstd are the ones lean's forward would have saved
                (x2d,) = ctx.saved_tensors
                xhat, mean, rstd = _FLR._ln_fwd(x2d, w["LN_W32"], w["LN_B32"], C._EPS, ctx.layout_int, ctx.M, ctx.d)
            else:
                x2d, mean, rstd = ctx.saved_tensors
                xhat, _m, _r = _FLR._ln_fwd(x2d, w["LN_W32"], w["LN_B32"], C._EPS, ctx.layout_int, ctx.M, ctx.d)   # same kernel, same input => same bf16 xhat
            lin = F.linear(xhat, w["W12"]); del xhat                                                          # same cuBLAS call => same lin
        else:
            x2d, mean, rstd, lin = ctx.saved_tensors
        g2d = g.contiguous().view(ctx.M, ctx.d)
        if g2d.dtype != torch.bfloat16: g2d = g2d.to(torch.bfloat16)
        d_hidden = torch.mm(g2d, w["W3"])                    # [M, h]  (LinearBackward: grad_output @ weight)
        BLOCK = triton.next_power_of_2(ctx.h)
        _swiglu_refround_bwd_kernel[(ctx.M,)](d_hidden, lin, ctx.h, d_hidden.stride(0), lin.stride(0), BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8)
        d_xhat = torch.mm(lin, w["W12"])                     # [M, d]  (lin now holds [d_x1 | d_x2])
        if LN_DX_ONLY and _HAS_LN_DX:
            dx_ln = _ln_bwd_dx_only(d_xhat, x2d, w["LN_W32"], mean, rstd, ctx.layout_int, None, ctx.M, ctx.d)      # same dx arithmetic, no dgamma/dbeta partials
        else:
            dx_ln, _, _ = _FLR._ln_bwd(d_xhat, x2d, w["LN_W32"], mean, rstd, ctx.layout_int, None, ctx.M, ctx.d)   # bf16, no dgamma/dbeta use
        dx = dx_ln + g2d                                     # two bf16 grads accumulate at the residual fork
        return dx.view(ctx.shape), None


def _transition_weights3(tr: "C.Transition", cache: dict) -> dict:
    key = ("kd3", id(tr)); ver = tuple(int(p._version) for p in tr.parameters())
    ent = cache.get(key)
    if ent is not None and ent["ver"] == ver:
        return ent
    ent = dict(ver=ver,
               W12=tr.ffn.w12.weight.detach().to(torch.bfloat16).contiguous(),        # [2h, d]  (autocast would cast the same way)
               W3=tr.ffn.w3.weight.detach().to(torch.bfloat16).contiguous(),          # [d, h]
               LN_W32=tr.norm.weight.detach().float().contiguous(),
               LN_B32=(tr.norm.bias.detach().float().contiguous() if tr.norm.bias is not None else torch.zeros_like(tr.norm.weight).float()),
               module_ref=tr)
    cache[key] = ent
    return ent


def transition_refround(x: Tensor, tr: "C.Transition", cache: dict, lean: bool = False, fwd: str | None = None) -> Tensor:
    """K-D3 (``lean``: LN + W12 recomputed in backward). ``fwd="t16"`` with ``lean``: the forward runs on ef2_t16_transition's kernel when that
    lever is live and the module has its served shape (else K-D3's own forward, counted by name there); the backward is K-D3's either way."""
    if not _KD3_OK:
        raise RuntimeError(f"K-D3 kernels unavailable ({_EXP})")
    assert tr.ffn.w12.bias is None and tr.ffn.w3.bias is None, "K-D3 assumes bias-free SwiGLUMLP (as in ESMFold2 Transition)"
    w = _transition_weights3(tr, cache)
    w["lean"] = bool(lean)
    w["fwd"] = _t16.forward_for(w, C._EPS) if (fwd == "t16" and lean) else None
    if torch.is_grad_enabled() and x.requires_grad:
        return TransitionRefround.apply(x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16), w)
    # no-grad: same forward without ctx
    with torch.no_grad():
        return TransitionRefround.forward(type("ctx", (), {"save_for_backward": lambda *a, **k: None})(), x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16), w)


# ----------------------------------------------------------------------------------------------
# D59 (joint rule): the kit must not change process-global numeric state. `enable()`/`disable()` are pure module rewiring;
# this helper lets resident runtimes assert it at every chained-step boundary:  g = agk.state_guard(); ...; g.check("step k")
# ----------------------------------------------------------------------------------------------
def state_guard(mode: str = "assert", label: str = "kit-enable"):
    import ef2_state_guard as _sg
    return _sg.StateGuard(mode=mode, label=label)


def assert_enable_is_state_neutral(model, **enable_kwargs) -> dict:
    """Snapshot process-global state, run enable(model, **kw) and disable(model), and assert the snapshot is unchanged
    after each.  Returns the snapshot.  Used by the unit test and the self-test."""
    import ef2_state_guard as _sg
    ref = _sg.snapshot()
    cfg = enable(model, **enable_kwargs)
    d1 = _sg.diff(_sg.snapshot(), ref)
    disable(model)
    d2 = _sg.diff(_sg.snapshot(), ref)
    assert not d1 and not d2, f"D59: enable/disable changed process-global state: after enable {d1}, after disable {d2}"
    return ref
