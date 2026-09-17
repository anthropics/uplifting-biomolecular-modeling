"""Carried module ``ef2_fused_ln`` (byte-identical to its kit): the ESM-family design loop's LayerNorm pieces; the face serves its dx-only
backward for frozen gamma/beta (row ef2_ln_bwd_dx, bitwise to the vendored backward kernel it replaces on row-major operands).  torch at module
level; the vendored helpers it can fall back to are imported inside functions (only where that image runs)."""
