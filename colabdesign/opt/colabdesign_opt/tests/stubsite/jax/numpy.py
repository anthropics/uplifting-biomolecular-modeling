"""stand-in jax.numpy: BindCraft's colabdesign_utils imports it at module level and uses it inside loss closures only (never called here)."""


def __getattr__(name):
    raise NotImplementedError(f"stand-in jax.numpy has no {name}: the loss closures are not evaluated on the stand-in stack")
