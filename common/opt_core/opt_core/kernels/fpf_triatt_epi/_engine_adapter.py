"""Engine-contract adapters of this unit: the registry callable ``fn`` (op ``triatt``: the engine's TriangleAttention.forward signature ->
the update the caller adds), the block-mode composition helper ``fn_block_residual`` and their shared STOCK prologue + STOCK attention step
``_stock_prologue_attention`` — the engine's own LayerNorm / bias / q,k,v,g projections and its cuEquivariance attention call, followed by this
unit's epilogue kernel. The engine's upstream package is imported lazily, inside the calls. ``epilogue`` keeps thin wrappers of the same names
and signatures that import this module on first call, so importing the unit never imports this module; the epilogue's own names
(``triatt_epilogue``, ``kernel_eligible``, ``_cache``, ``_STATS``) are read off the ``epilogue`` module at call time, so a lever that rebinds
``epilogue.triatt_epilogue`` is honoured exactly as before."""
from __future__ import annotations

import math
import torch

from . import epilogue as _E


def _stock_prologue_attention(module, x, mask, triangle_attention):
    """Exactly layers.Attention.forward up to (and excluding) _wrap_up, for the module's frame (x already transposed for the ending node by the caller).
    Returns (o [.., I, H, J, D] bf16 cuEq layout, g_lin [.., I, J, H*D] bf16 pre-sigmoid, x_ln)."""
    from protenix.model.utils import permute_final_dims
    import protenix.model.triangular.layers as TL
    mha = module.mha
    if mask is None:
        mask = x.new_ones(x.shape[:-1])
    x = module.layer_norm(x)                                                       # stock FusedLayerNorm (makes x contiguous internally)
    mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
    triangle_bias = permute_final_dims(module.linear(x), (2, 0, 1)).unsqueeze(-4)
    biases = [mask_bias, triangle_bias]
    q, k, v = mha._prep_qkv(x, x, apply_scale=False)                               # stock GEMMs + views (any patch on _prep_qkv, e.g. a fused q|k|v|g GEMM, is honoured automatically)
    if q.shape[-2] <= 16 or triangle_attention != "cuequivariance":
        return None, None, x, biases
    scale = 1.0 / math.sqrt(mha.c_hidden)
    o = TL.cuequivariance_triangular_attn(q, k, v, biases[1].float(), (biases[0] == 0).bool(), scale)[0]     # [.., I, H, J, D] contiguous (cuEq)
    g_pre = getattr(mha, "_t2g_gate", None)                                        # gate columns of a fused q|k|v|g GEMM (strided view) when a `_t2g_gate` patch is present
    if g_pre is not None and g_pre.shape[-1] == mha.no_heads * mha.c_hidden and tuple(g_pre.shape[:-1]) == tuple(x.shape[:-1]) and g_pre.dtype == o.dtype:
        g_lin = g_pre; mha._t2g_gate = None
    else:
        g_lin = mha.linear_g(x)                                                    # [.., I, J, H*D] bf16 (same GEMM as stock)
    return o, g_lin, x, biases


def fn(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    """FPF registry `triatt`: TriangleAttention.forward(x, mask, chunk_size, triangle_attention, inplace_safe) -> update (caller adds). Both starting and ending nodes."""
    stock_fwd = getattr(type(module).forward, "_fpf_orig", None)
    def _fallback():
        if stock_fwd is not None:
            return stock_fwd(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        import protenix.model.triangular.triangular as TT
        return TT.TriangleAttention.forward(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
    mha = module.mha
    if (chunk_size is not None or triangle_attention != "cuequivariance" or x.dtype != torch.bfloat16 or not x.is_cuda
            or mha.linear_g is None or not _E._HAS_TRITON or x.shape[-2] <= 16):
        _E._STATS["fallback"] = _E._STATS.get("fallback", 0) + 1                     # a precondition of the served class (dtype / backend / chunking / tiny N)
        _E._STATS.setdefault("fallback_by", {})["precondition"] = _E._STATS["fallback_by"].get("precondition", 0) + 1
        return _fallback()
    word = _E.stock_word(mha.linear_o.weight.shape[0], mha.no_heads, mha.c_hidden, x.device)
    if word:                                                                          # a cell the kernel does not run: the stock forward BY NAME (measured off / nothing admits it)
        _E._STATS["fallback"] = _E._STATS.get("fallback", 0) + 1
        _E._STATS.setdefault("fallback_by", {})[word] = _E._STATS["fallback_by"].get(word, 0) + 1
        return _fallback()
    _E._STATS["kernel"] = _E._STATS.get("kernel", 0) + 1
    if x.dim() > 3:                                                                  # leading batch dims: loop per example (each element == the un-batched result;
        lead = x.shape[:-3]                                                          #  stock's own batched path squeezes cuEq's batch with [0] and is NOT per-example)
        xs = x.reshape((-1,) + tuple(x.shape[-3:])); ms_ = None if mask is None else mask.reshape((-1,) + tuple(mask.shape[-2:]))
        outs = [_E.fn(module, xs[b], mask=None if ms_ is None else ms_[b], chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
                for b in range(xs.shape[0])]
        u = torch.stack(outs, 0)
        return u.reshape(tuple(lead) + tuple(u.shape[1:]))
    xx = x.transpose(-2, -3) if not module.starting else x
    mm = (mask.transpose(-1, -2) if (mask is not None and not module.starting) else mask)
    o, g_lin, x_ln, biases = _stock_prologue_attention(module, xx, mm, triangle_attention)
    if o is None:                                                                    # tiny-N torch fallback inside Attention.forward -> run stock wrap-up
        return _fallback()
    c = _E._cache(module, x)
    u = _E.triatt_epilogue(o, g_lin, c["wo16"], ending=False, residual=False, woT16=c["woT16"])   # [.., I, J, c] in the module's own frame (== stock mha output)
    if u.dim() != x.dim():
        u = u.reshape(xx.shape[:-1] + (u.shape[-1],))
    if not module.starting:
        u = u.transpose(-2, -3)                                                      # stock returns the transposed VIEW; caller adds it into (transposed-contiguous) z
    return u


def fn_block_residual(module, z, ending: bool, mask=None, triangle_attention="cuequivariance"):
    """Composition helper (block mode), IN PLACE on the un-transposed block tensor z [.., N, N, c] (contiguous):
         ending=False :  z += tri_att_start(z)                                                   (stock statement 3 of the inplace PairformerBlock path)
         ending=True  :  zT = z.transpose(-2,-3).contiguous(); zT += tri_att_end(zT); z = zT.transpose(-2,-3).contiguous()   (statements 4-6)
       NOTE (protenix 2.0.0): PairformerBlock builds tri_att_end as a plain TriangleAttention (starting=True!) — the column-wise role comes ONLY from the caller's
       transposes. So for ending=True the stock prologue must see x = z^T: we hand it the strided VIEW z.transpose(-2,-3) (FusedLayerNorm calls .contiguous()
       internally exactly as it does for stock's own strided inputs; LN/linear outputs are then contiguous in the transposed frame (i,j) = (col,row) of z, and the
       q/k/v/g GEMMs + cuEq see bit-exact the same operands as stock, which ran on the contiguous zT) and the epilogue scatters u[i,j] into z[j,i] (ENDING=True).
       Cost: the LN's internal .contiguous() is one transpose pass (stock pays it too, as z.transpose.contiguous()); the sibling prologue kernel removes it by
       reading z transposed directly. Returns z (same storage)."""
    mha = module.mha
    word = _E.stock_word(mha.linear_o.weight.shape[0], mha.no_heads, mha.c_hidden, z.device)
    if word:
        raise NotImplementedError(f"fn_block_residual: cell (c={mha.linear_o.weight.shape[0]}, H={mha.no_heads}, D={mha.c_hidden}) is not run by the kernel ({word}) -> use the stock block statements")
    x = z.transpose(-2, -3) if ending else z
    mm = mask if mask is None or not ending else mask.transpose(-1, -2)
    o, g_lin, _, _ = _stock_prologue_attention(module, x, mm, triangle_attention)
    if o is None:
        raise RuntimeError("fn_block_residual: stock attention fell back to torch path (N<=16 or backend) — use the stock block")
    cc = _E._cache(module, z)
    _E.triatt_epilogue(o, g_lin, cc["wo16"], z, ending=bool(ending), residual=True, woT16=cc["woT16"])
    return z
