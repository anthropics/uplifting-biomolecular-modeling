import hashlib
import random
from collections.abc import Sequence

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def spawn_rng(global_rng: np.random.Generator | None = None) -> np.random.Generator:
    """Get a random number generator.
    This is designed to create a new RNG which is independent from the global RNG.

    Parameters
    ----------
    rng : np.random.Generator, optional
        A random number generator, by default None.

    Returns
    -------
    np.random.Generator
        A random number generator.
    """
    if global_rng is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(global_rng.integers(0, 1 << 30))


def hash_seq(seq: str) -> str:
    """Hash a sequence string to create a unique identifier."""
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def check_array(
    array: np.ndarray,
    name: str,
    dtype: type | np.dtype | Sequence[type | np.dtype] | None = None,
    shape: tuple[int, ...] | None = None,
):
    if dtype is not None:
        if isinstance(dtype, type | np.dtype):
            dtype = [dtype]
        assert any(np.issubdtype(array.dtype, dt) for dt in dtype), (
            f"Name {name}: Expected array dtype to be one of {dtype}, got {array.dtype}"
        )
    if shape is not None:
        assert array.ndim == len(shape), (
            f"Name {name}: Expected array shape to be {shape}, got {array.shape}"
        )
        _shape = tuple(v if v != -1 else array.shape[i] for i, v in enumerate(shape))
        assert array.shape == _shape, (
            f"Name {name}: Expected array shape to be {_shape}, got {array.shape}"
        )


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype | Sequence[torch.dtype] | None = None,
    shape: tuple[int, ...] | None = None,
):
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            dtype = (dtype,)
        assert tensor.dtype in dtype, (
            f"Name {name}: Expected tensor dtype to be one of {dtype}, got {tensor.dtype}"
        )
    if shape is not None:
        assert tensor.ndim == len(shape), (
            f"Name {name}: Expected tensor shape to be {len(shape)}, got {tensor.shape}"
        )
        _shape = tuple(v if v != -1 else tensor.shape[i] for i, v in enumerate(shape))
        assert tensor.shape == _shape, (
            f"Name {name}: Expected tensor shape to be {_shape}, got {tensor.shape}"
        )
