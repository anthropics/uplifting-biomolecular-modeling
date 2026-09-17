"""The weight vocabulary of the triangle multiplication's fused providers — the names a kit's weight shim maps its module's tensors
to. One vocabulary for the single-GPU provider ladder (:mod:`opt_core.trimul`) and the row-block provider
(:mod:`opt_core.mem.rowpair.trimul_fused`); this module imports nothing, so naming the vocabulary carries no kernel unit along.

``WEIGHT_KEYS``: LayerNorm_in weight / bias ``[C_z]``; the a-gate, a-projection, b-gate, b-projection matrices ``[C_h, C_z]``; LayerNorm_out
weight / bias ``[C_h]``; the output projection ``[C_z, C_h]``; the output gate ``[C_z, C_z]``. ``BIAS_KEYS``: the optional biases of the six
linear maps (a module without biases passes none of them)."""

__all__ = ["WEIGHT_KEYS", "BIAS_KEYS"]

WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
BIAS_KEYS = ("b_ag", "b_ap", "b_bg", "b_bp", "b_o", "b_og")
