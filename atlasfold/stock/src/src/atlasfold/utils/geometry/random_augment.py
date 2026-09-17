import math
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def sum_array(array: ArrayT, dim: int, keepdim: bool = False) -> ArrayT:
    if isinstance(array, np.ndarray):
        return array.sum(dim, keepdims=keepdim)
    else:
        return array.sum(dim, keepdim=keepdim)


def stack_array(arrays: Sequence[ArrayT], dim: int) -> ArrayT:
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=dim)
    else:
        return torch.stack(arrays, dim=dim)  # type: ignore


def copysign(a: ArrayT, b: ArrayT) -> ArrayT:
    if isinstance(a, np.ndarray):
        return np.copysign(a, b)
    else:
        return torch.copysign(a, b)


def get_center(coords: ArrayT, mask: ArrayT) -> ArrayT:
    """Get the mean position of the masked coordinates.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (*, L, 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (*, L).

    Returns
    -------
    np.ndarray | torch.Tensor
        The mean position of the masked coordinates of shape (*, 1, 3).
    """
    if not mask.any():
        raise ValueError("Mask has no True values; cannot compute center.")

    if isinstance(coords, np.ndarray):
        assert isinstance(mask, np.ndarray)
        # Sanitize coords
        safe_coords = np.where(mask[..., None], coords, 0.0)

        total_mass = mask.sum(-1, keepdims=True, dtype=np.float32).clip(1)  # [..., 1]
        center = (
            np.sum(safe_coords, axis=-2, keepdims=True) / total_mass[..., None]
        )  # [..., 1, 3]
        return center
    else:
        assert isinstance(mask, torch.Tensor)
        assert coords.dtype == torch.float32
        # Sanitize coords
        safe_coords = coords.masked_fill(~mask[..., None], 0.0)

        total_mass = mask.sum(-1, keepdim=True).clamp(1)
        center = torch.sum(safe_coords, dim=-2, keepdim=True) / total_mass[..., None]
        return center


def do_centering(coords: ArrayT, mask: ArrayT, mask_to_zero: bool = True) -> ArrayT:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (*, L, 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (*, L).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.

    Returns
    -------
    np.ndarray | torch.Tensor
        The centered coordinates tensor of shape (*, L, 3).
    """
    assert coords.ndim == mask.ndim + 1
    assert coords.ndim >= 2

    if not mask.any():
        # If no positions are masked, return coords as is
        return coords

    # get_center now handles NaNs internally
    center_pos = get_center(coords, mask)  # shape (*, 1, 3)
    centered_coords = coords - center_pos

    if mask_to_zero:
        centered_coords[~mask] = 0.0

    return centered_coords  # type: ignore


def do_centering_atom14(
    coords: ArrayT, mask: ArrayT, mask_to_zero: bool = True
) -> ArrayT:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (*, L, 14, 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (*, L, 14).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.

    Returns
    -------
    np.ndarray | torch.Tensor
        The centered coordinates tensor of shape (*, L, 14, 3).
    """
    shape = coords.shape
    assert shape[-2:] == (14, 3)
    return do_centering(
        coords.reshape(*coords.shape[:-3], -1, 3),
        mask.reshape(*mask.shape[:-2], -1),
        mask_to_zero,
    ).reshape(shape)


def center_random_augmentation(
    coords: ArrayT,
    mask: ArrayT,
    s_trans: float = 1.0,
    rng: np.random.Generator | torch.Generator | None = None,
) -> ArrayT:
    """Centering and Random Augmentation (NumPy/Torch version)
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    if isinstance(coords, np.ndarray):
        assert isinstance(mask, np.ndarray)
        assert isinstance(rng, np.random.Generator | None)
        return _center_random_augmentation_npy(coords, mask, s_trans, rng)
    else:
        assert isinstance(mask, torch.Tensor)
        assert isinstance(rng, torch.Generator | None)
        return _center_random_augmentation_torch(coords, mask, s_trans, rng)


def center_random_augmentation_atom14(
    coords: ArrayT,
    mask: ArrayT,
    s_trans: float = 1.0,
    rng: np.random.Generator | torch.Generator | None = None,
) -> ArrayT:
    """Centering and Random Augmentation for atom14 coordinates.
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    assert coords.shape[-2:] == (14, 3)
    return center_random_augmentation(
        coords.reshape(*coords.shape[:-3], -1, 3),
        mask.reshape(*mask.shape[:-2], -1),
        s_trans,
        rng,
    ).reshape(coords.shape)


def _center_random_augmentation_npy(
    coords: np.ndarray,
    mask: np.ndarray,
    s_trans: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Centering and Random Augmentation (NumPy version)
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    coords = do_centering(coords, mask, mask_to_zero=False)

    rng = rng or np.random.default_rng()

    R = random_rotations_npy(coords.shape[:-2], dtype=np.float32, rng=rng)  # [..., 3, 3]

    # Matrix multiplication is safe (NaNs stay local to invalid atoms)
    coords = coords @ R

    if s_trans > 0.0:
        # Create random translation with same shape as coords[..., 0:1, :]
        trans_shape = list(coords.shape)
        trans_shape[-2] = 1  # The 'L' dimension becomes 1 for broadcasting

        noise = rng.normal(scale=s_trans, size=trans_shape).astype(coords.dtype)
        coords += noise

    # Mask out invalid positions to zero
    coords[~mask] = 0.0

    return coords


def _center_random_augmentation_torch(
    coords: torch.Tensor,
    mask: torch.Tensor,
    s_trans: float = 1.0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Centering and Random Augmentation
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """
    assert coords.dtype == torch.float32
    with torch.autocast(coords.device.type, enabled=False):
        coords = do_centering(coords, mask, mask_to_zero=False)

        R = random_rotations_torch(coords.shape[:-2], coords.device, rng)  # [..., 3, 3]
        coords = coords @ R

        if s_trans > 0.0:
            noise = torch.randn_like(coords[..., 0:1, :], generator=rng)
            coords.add_(noise, alpha=s_trans)

        # Mask out
        coords.masked_fill_(~mask[..., None], 0.0)

    return coords


def random_rotations_npy(
    shape: tuple[int, ...], dtype: type | np.dtype, rng: np.random.Generator
) -> np.ndarray:
    """Generate random rotations as 3x3 rotation matrices."""
    n = math.prod(shape)
    o = rng.normal(size=(n, 4)).astype(dtype)
    s = (o**2).sum(1)
    # Use broadcasting for division
    quaternions = o / copysign(np.sqrt(s), o[:, 0])[:, np.newaxis]
    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)


def random_rotations_torch(
    shape: tuple[int, ...],
    device: torch.device,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate random rotations as 3x3 rotation matrices."""
    n = math.prod(shape)
    o = torch.randn((n, 4), dtype=torch.float32, device=device, generator=rng)
    s = o.pow(2).sum(1)
    quaternions = o / copysign(torch.sqrt(s), o[:, 0])[:, None]
    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)


def quaternion_to_matrix(Q: ArrayT) -> ArrayT:
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = (Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3])
    two_s = 2.0 / sum_array(Q**2, -1)
    o = stack_array(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return o.reshape(Q.shape[:-1] + (3, 3))
