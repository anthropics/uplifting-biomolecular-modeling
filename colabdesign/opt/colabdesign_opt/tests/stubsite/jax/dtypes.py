"""stand-in jax.dtypes (tests only)."""
import numpy as _np


def canonicalize_dtype(dt):
    dt = _np.dtype(dt)
    return _np.dtype("float32") if dt == _np.dtype("float64") else dt
