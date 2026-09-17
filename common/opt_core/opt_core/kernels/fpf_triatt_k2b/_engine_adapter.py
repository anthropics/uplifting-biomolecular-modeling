"""Engine-contract adapters of this unit: the registry callables ``fn`` (K2B) / ``fn_k2`` (previous K2) with the engine's
TriangleAttention.forward signature — the module's own LayerNorm, q/k/v/g/bias projections, sigmoid gate and output projection run as
module code with ONLY the attention kernel swapped for this unit's flash triangle attention. The engine's upstream package is imported
lazily, inside the call. The unit's ``__init__`` keeps thin wrappers of the same names and signatures that import this module on first
call, so importing the unit never imports this module."""
import sys

_P = sys.modules[__package__]          # the unit package: its kernel modules (K2B / K2), STATS counters and MIN_TOKENS gate, read at call time


def _make(kernel):
    def fn(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
        import protenix.model.triangular.layers as L
        STATS = _P.STATS; MIN_TOKENS = _P.MIN_TOKENS
        STATS["calls"] += 1
        orig = L.cuequivariance_triangular_attn
        def attn(q, k, v, bias, mask_b, scale):
            if q.shape[-2] < MIN_TOKENS or q.shape[-1] not in (16, 32, 64, 128):
                STATS["stock_calls"] += 1; return orig(q, k, v, bias, mask_b, scale)
            STATS["kernel_calls"] += 1
            # SHAPE CONTRACT — mirror cuEq exactly: inputs of rank < 5 get leading singleton dims, output is 5-D [B,N,H,S,D]; the stock caller then
            # applies `[0]`.  For B == 1 (the engine's inference path: z is [N,N,c] or [1,N,N,c]) that strips the singleton => identical shapes to stock.
            if q.dim() == 5 and q.shape[0] > 1:
                # leading batch > 1: stock's `[0]` would select example 0 (upstream hazard).  We compute every example separately (batch-invariant by
                # construction) and return the full batch under ONE extra leading singleton, so the caller's `[0]` yields [B,N,H,S,D] = per-example correct.
                import torch
                outs = [kernel(q[b:b+1], k[b:b+1], v[b:b+1], bias[b:b+1] if bias.shape[0] > 1 else bias,
                               mask=None if mask_b is None else (mask_b[b:b+1] if mask_b.shape[0] > 1 else mask_b), scale=scale) for b in range(q.shape[0])]
                return torch.cat(outs, 0).unsqueeze(0)
            return kernel(q, k, v, bias, mask=mask_b, scale=scale)            # 5-D tensor exactly like cuEq; stock `[0]` strips B==1
        L.cuequivariance_triangular_attn = attn
        try:
            f = getattr(type(module).forward, "_fpf_orig", type(module).forward)
            return f(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        finally:
            L.cuequivariance_triangular_attn = orig
    return fn
fn = _make(_P.K2B.flash_triangle_attention)
fn_k2 = _make(_P.K2.flash_triangle_attention)
