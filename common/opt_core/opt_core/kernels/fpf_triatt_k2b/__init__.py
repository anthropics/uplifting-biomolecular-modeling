"""FPF registry impls for `triatt_start` / `triatt_end` (the engine's TriangleAttention.forward contract) with ONLY the attention kernel swapped:
    stock fast_layernorm + stock OpenfoldLinear q/k/v/g/bias (module code)  ->  K2B Triton flash triangle attention  ->  stock sigmoid gate + linear_o (module code)
impl strings:  fpf_triatt_k2b:fn        (K2B, kernel candidate)      fpf_triatt_k2b:fn_k2   (previous K2, for A/B)
engine_stock_class: the engine's TriangleAttention served path (fast_layernorm; bf16 OpenfoldLinear; cuequivariance triangle_attention 0.8 fp32 bias; sigmoid gate; linear_o)
label: TIER2_SAME_CLASS (reordered fp32 online-softmax accumulation; base-2 exponent domain); LN: stock-call (fast_layernorm, untouched); r2r bit-exact; batch dims: looped per example
(v1.1 shape contract: output rank/shape identical to the stock cuEq path for 3-D and 4-D z; for a leading batch > 1 the result is per-example correct AND batch-invariant
— the full batch is returned under one extra singleton so the stock `[0]` yields it — instead of inheriting the upstream `[0]`-selects-example-0 hazard).
cells: D in {16,32,64,128}; any H; bf16/fp16 (base-2 domain) and fp32 inputs (natural domain, IP=tf32|tf32x3|ieee via PF_TRIATTN_FP32_DOT); c_in irrelevant (kernel starts after
the projections) => c=64 template stack (H=2,D=32) is a supported cell; below PF_TRIATTN_MIN_TOKENS query rows (default 0 here; adapters use 300) the stock cuEq call is used.
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _here)
import triatt_k2b as K2B
import flash_triattn_k2 as K2
MIN_TOKENS = int(os.environ.get("PF_TRIATTN_MIN_TOKENS", "0") or 0)
STATS = {"calls": 0, "kernel_calls": 0, "stock_calls": 0}

def fn(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    """Registry callable (K2B kernel): the engine-contract adapter ``_engine_adapter.fn``, imported on first call."""
    from ._engine_adapter import fn as _f
    return _f(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)

def fn_k2(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    """Registry callable (previous K2 kernel, for A/B): ``_engine_adapter.fn_k2``, imported on first call."""
    from ._engine_adapter import fn_k2 as _f
    return _f(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
attn_k2b = K2B.flash_triangle_attention      # raw kernel entry (cuEq triangle_attention signature)
attn_k2 = K2.flash_triangle_attention
