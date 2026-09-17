"""qk_rotary — fused q/k LayerNorm + rotary in one Triton launch (Triton compiles each class once per machine into its own cache; the fused kit warms the classes at apply). Entry: qk_rotary.ops(model)."""
from .qk_rotary import ops  # noqa: F401
