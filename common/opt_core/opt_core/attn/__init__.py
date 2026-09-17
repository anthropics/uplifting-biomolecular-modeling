"""Attention backends and served-shape gates — the non-triangle attention primitives a kit's thin adapter calls.

    size_gate    a token/shape gate with total accounting: every call is served, gated (by a named bound) or a named
                 fallback; the census renders as one key=value fragment for the kit's activation-evidence / exit-tally line
    sdpa_bias    ONE fused scaled-dot-product-attention call with an additive pair bias (+ key mask, optional compute
                 dtype, optional backend pin); refuses by NAME (``Refused.event``) instead of degrading silently

Pure standard library at module level: ``sdpa_bias`` imports torch inside the call that needs it (a stack without torch
gets ``Refused(event='torch_missing')``, never an ImportError at ``import opt_core``). Nothing here names an engine: the
engine-specific glue (which module to patch, the stock mask constant, the layout) is the kit adapter's, passed as arguments.
"""
