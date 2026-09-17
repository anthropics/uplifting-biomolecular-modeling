"""Carried module ``af3_fused`` (byte-identical to its kit copy): the LayerNorm + SwiGLU + output-projection transition in one Triton kernel,
generalised to non-power-of-two channel widths by padding the register tile (row af3_fused).  Imports torch and triton at module level."""
