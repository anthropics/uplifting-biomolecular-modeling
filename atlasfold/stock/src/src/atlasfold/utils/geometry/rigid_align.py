import warnings
from typing import overload

import numpy as np
import torch


@overload
def rigid_align_atom14(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = ...,
    align_mode: str = ...,
) -> np.ndarray: ...


@overload
def rigid_align_atom14(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = ...,
    align_mode: str = ...,
) -> torch.Tensor: ...


def rigid_align_atom14(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    align_mode: str = "full",
    mask_to_zero: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Performs rigid alignment of atom14 coordinates to a target set.
    Dispatches to NumPy or PyTorch implementations based on input type.
    """
    if isinstance(coords, np.ndarray):
        assert isinstance(target, np.ndarray)
        return rigid_align_atom14_numpy(coords, target, mask, align_mode, mask_to_zero)
    elif isinstance(coords, torch.Tensor):
        assert isinstance(target, torch.Tensor)
        return rigid_align_atom14_torch(coords, target, mask, align_mode, mask_to_zero)
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


def rigid_align_atom14_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    align_mode: str = "full",
    mask_to_zero: bool = True,
) -> np.ndarray:
    shape = coords.shape
    if len(shape) < 3 or shape[-2:] != (14, 3):
        raise ValueError(f"Expected coords shape (*, L, 14, 3), got {shape}")

    if mask is None:
        mask = np.ones(shape[:-1], dtype=bool)

    # Create sub-mask based on align_mode
    align_mask = np.zeros_like(mask)
    mode = align_mode.lower()
    if mode == "full":
        align_mask = mask.copy()
    elif mode == "ca":
        align_mask[..., 1] = mask[..., 1]
    elif mode == "backbone":
        align_mask[..., 0:3] = mask[..., 0:3]
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    # Flatten dimensions L and 14 for transform calculation
    RT, t = get_rigid_transform_numpy(
        coords.reshape(*coords.shape[:-3], -1, 3),
        target.reshape(*coords.shape[:-3], -1, 3),
        align_mask.reshape(*align_mask.shape[:-2], -1),
    )

    # Flatten for safe matrix multiplication, then reshape back
    flat_coords = coords.reshape(*shape[:-3], -1, 3)
    flat_aligned = np.matmul(flat_coords, RT) + t[..., np.newaxis, :]
    aligned = flat_aligned.reshape(shape)

    if mask_to_zero:
        aligned[~mask[..., np.newaxis]] = 0.0

    return aligned


@torch.no_grad()
def rigid_align_atom14_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    align_mode: str = "full",
    mask_to_zero: bool = True,
) -> torch.Tensor:
    assert coords.dtype == torch.float32
    shape = coords.shape
    if shape[-2:] != (14, 3):
        raise ValueError(f"Expected coords shape (*, L, 14, 3), got {shape}")

    with torch.autocast(coords.device.type, enabled=False):
        # Create sub-mask based on align_mode
        align_mask = coords.new_zeros(coords.shape[:-1], dtype=torch.bool)
        mode = align_mode.lower()
        if mode == "full":
            align_mask = mask
        elif mode == "ca":
            align_mask[..., 1] = mask[..., 1]
        elif mode == "backbone":
            align_mask[..., 0:3] = mask[..., 0:3]
        else:
            raise ValueError(f"Unknown align_mode: {align_mode}")

        RT, t = get_rigid_transform_torch(
            coords.reshape(*coords.shape[:-3], -1, 3),  # (*, L*14, 3)
            target.reshape(*target.shape[:-3], -1, 3),  # (*, L*14, 3)
            align_mask.reshape(*align_mask.shape[:-2], -1),  # (*, L*14)
        )
        flat_coords = coords.reshape(*shape[:-3], -1, 3)  # (*, L*14, 3)
        flat_aligned = flat_coords @ RT + t[..., None, :]
        aligned = flat_aligned.reshape(shape)  # (*, L, 14, 3)

        if mask_to_zero:
            aligned.masked_fill_(~mask[..., None], 0.0)

    return aligned


@overload
def rigid_align(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    mask_to_zero: bool = True,
) -> np.ndarray: ...


@overload
def rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    mask_to_zero: bool = True,
) -> torch.Tensor: ...


def rigid_align(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    mask_to_zero: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Performs rigid alignment of a set of coordinates to a target set using SVD.

    This function computes the optimal rigid transformation (rotation and translation)
    that aligns `coords` to `target`, minimizing the mean squared error,
    with optional masking and numerical stability.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the target coordinates.
    mask : np.ndarray | torch.Tensor (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).


    Returns
    -------
    aligned_coords : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    if isinstance(coords, np.ndarray):
        return rigid_align_numpy(coords, target, mask, mask_to_zero)  # type: ignore
    elif isinstance(coords, torch.Tensor):
        return rigid_align_torch(coords, target, mask, mask_to_zero)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


def rigid_align_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    mask_to_zero: bool = True,
) -> np.ndarray:
    """
    Performs rigid alignment of a set of coordinates to a target set using SVD.

    This function computes the optimal rigid transformation (rotation and translation)
    that aligns `coords` to `target`, minimizing the mean squared error,
    with optional masking and numerical stability.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (*, N, 3) representing the target coordinates.
    mask : np.ndarray | None (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).

    Returns
    -------
    aligned_coords : np.ndarray
        Array of shape (*, N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    if mask is None:
        mask = np.ones(coords.shape[:-1], dtype=bool)

    if not np.any(mask):
        return coords

    RT, T = get_rigid_transform_numpy(coords, target, mask)

    # Apply transformation: coords @ RT + T
    aligned_coords = np.matmul(coords, RT) + T[..., np.newaxis, :]

    if mask_to_zero:
        aligned_coords[~mask[..., np.newaxis]] = 0.0

    return aligned_coords


def get_rigid_transform_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the rigid transformation that aligns `coords` to `target`.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (*, N, 3) representing the target coordinates.
    mask : np.ndarray
        Array of shape (*, N) indicating valid points.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : np.ndarray
        Array of shape (*, 3, 3) representing the rotation matrices.
    t : np.ndarray
        Array of shape (*, 3) representing the translation vectors.
    """
    dtype = coords.dtype

    if not np.any(mask):
        # Return identity transform
        shape_prefix = coords.shape[:-2]
        R = np.eye(3, dtype=dtype)
        t = np.zeros((3,), dtype=dtype)
        if shape_prefix:
            R = np.tile(R, (*shape_prefix, 1, 1))
            t = np.tile(t, (*shape_prefix, 1))
        return R, t

    L = coords.shape[-2]
    if L < 3:
        warnings.warn(
            f"Point cloud has only {L} points (< 3). "
            "Weighted rigid alignment may not produce a unique rotation.",
            stacklevel=2,
        )

    # Sanitize inputs: If there are NaNs in masked regions, they will propagate
    # during centroid calculation even if weights are zero (NaN * 0 = NaN).
    # We force masked regions to 0.0.
    mask_expanded = mask[..., np.newaxis]
    coords = np.where(mask_expanded, coords, 0.0)
    target = np.where(mask_expanded, target, 0.0)

    w = mask.astype(dtype)
    w_sum = np.sum(w, axis=-1) + eps

    # Expand weights for broadcasting: (..., N) -> (..., N, 1)
    w_expanded = w[..., np.newaxis]

    # Weighted centroids
    # Note: Inputs (coords, target) are already sanitized (0 in masked regions),
    # so summation here is safe even if original data had NaNs in masked regions.
    coords_center = np.sum(coords * w_expanded, axis=-2) / w_sum[..., np.newaxis]
    target_center = np.sum(target * w_expanded, axis=-2) / w_sum[..., np.newaxis]

    # Center coordinates
    coords_centered = coords - coords_center[..., np.newaxis, :]
    target_centered = target - target_center[..., np.newaxis, :]

    # Re-apply mask implicitly by multiplying weights (masked regions become 0 again)
    # This is redundant if sanitized, but ensures correctness if weights vary.
    H = np.matmul((coords_centered * w_expanded).swapaxes(-1, -2), target_centered)

    try:
        # SVD: H = U S Vh
        U, _, Vh = np.linalg.svd(H)

        # Fixed reflection removal (Kabsch algorithm)
        d = np.linalg.det(np.matmul(U, Vh))

        F = np.eye(3, dtype=dtype)
        if H.ndim > 2:
            batch_shape = H.shape[:-2]
            F = np.tile(F, (*batch_shape, 1, 1))

        if F.ndim == 2:
            F[2, 2] = np.sign(d)
        else:
            F[..., 2, 2] = np.sign(d)

        RT = np.matmul(U, np.matmul(F, Vh))

    except np.linalg.LinAlgError as e:
        warnings.warn(
            f"SVD failed during rigid alignment: {e}. Returning identity rotation.",
            stacklevel=2,
        )
        shape_prefix = coords.shape[:-2]
        RT = np.eye(3, dtype=dtype)
        if shape_prefix:
            RT = np.tile(RT, (*shape_prefix, 1, 1))

    # Compute translation: t = target_center - coords_center @ RT
    term2 = np.matmul(coords_center[..., np.newaxis, :], RT)
    t = target_center[..., np.newaxis, :] - term2
    t = t.squeeze(-2)

    return RT, t


@torch.no_grad()
def rigid_align_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    mask_to_zero: bool = True,
) -> torch.Tensor:
    """
    Performs rigid alignment of a set of coordinates to a target set using SVD in PyTorch.

    Parameters
    ----------
    coords : torch.Tensor
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Array of shape (*, N, 3) representing the target coordinates.
    mask : torch.Tensor | None (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).

    Returns
    -------
    aligned_coords : torch.Tensor
        Array of shape (*, N, 3) containing the aligned coordinates.
    """
    assert coords.dtype == torch.float32
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)

    if not mask.any():
        return coords

    with torch.autocast(coords.device.type, enabled=False):
        RT, T = get_rigid_transform_torch(coords, target, mask)
        aligned_coords = coords @ RT + T[..., None, :]

    if mask_to_zero:
        aligned_coords.masked_fill_(~mask[..., None], 0.0)

    return aligned_coords


@overload
def get_rigid_transform(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def get_rigid_transform(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def get_rigid_transform(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    eps: float = 1e-8,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Dispatch function to compute rigid transform for either NumPy or PyTorch inputs.
    """
    if isinstance(coords, np.ndarray):
        return get_rigid_transform_numpy(coords, target, mask, eps)
    elif isinstance(coords, torch.Tensor):
        return get_rigid_transform_torch(coords, target, mask, eps)
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


def get_rigid_transform_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the rigid transformation that aligns `coords` to `target`.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (*, N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Tensor of shape (*, N, 3) representing the target coordinates.
    mask : torch.Tensor
        Tensor of shape (*, N) indicating valid points.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : torch.Tensor
        Tensor of shape (*, 3, 3) representing the rotation matrices.
    t : torch.Tensor
        Tensor of shape (*, 3) representing the translation vectors.
    """
    assert coords.dtype == torch.float32
    assert target.dtype == torch.float32
    device = coords.device

    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=device)

    if not mask.any():
        # If there are no valid atoms, return identity transform
        R = torch.eye(3, dtype=torch.float32, device=device)
        R = R.tile(*coords.shape[:-2], 1, 1)
        t = torch.zeros(*coords.shape[:-2], 3, dtype=torch.float32, device=device)
        return R, t

    L = coords.shape[-2]
    if L < 3:
        warnings.warn(
            f"Point cloud has only {L} points (< 3). "
            "Weighted rigid alignment may not produce a unique rotation.",
            stacklevel=2,
        )

    with torch.autocast(coords.device.type, enabled=False):
        coords = coords.masked_fill(~mask[..., None], 0.0)
        target = target.masked_fill(~mask[..., None], 0.0)

        w = mask.float()
        w_sum = w.sum(dim=-1) + eps

        # Inputs are already sanitized (0.0 in masked regions), so these sums are safe.
        coords_center = (coords * w[..., None]).sum(dim=-2) / w_sum[..., None]
        target_center = (target * w[..., None]).sum(dim=-2) / w_sum[..., None]

        coords = coords - coords_center[..., None, :]
        target = target - target_center[..., None, :]

        H = (coords * w[..., None]).mT @ target

        try:
            U, _, Vh = torch.linalg.svd(H)

            # Fixed reflection removal
            F = torch.eye(3, dtype=torch.float32, device=device)
            F = F.tile(*H.shape[:-2], 1, 1)
            F[..., -1, -1] = torch.sign(torch.linalg.det(U @ Vh))

            # Transposed rotation matrix
            RT = U @ F @ Vh
        except RuntimeError as e:
            warnings.warn(
                f"SVD failed during rigid alignment: {e}. Returning identity rotation.",
                stacklevel=2,
            )
            RT = torch.eye(3, dtype=torch.float32, device=device)
            RT = RT.tile(*coords.shape[:-2], 1, 1)

        # Compute translation
        t = target_center[..., None, :] - coords_center[..., None, :] @ RT
        t = t.squeeze(-2)

    return RT, t
