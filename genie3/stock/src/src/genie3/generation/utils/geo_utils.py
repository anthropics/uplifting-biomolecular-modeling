"""
Geometry utilities.

Functions for protein geometry calculations and transformations.
"""

import torch
import torch.nn.functional as F
from typing import Dict

from genie3.generation.utils.affine_utils import T
from genie3.generation.utils.tensor_utils import batched_gather


def _compute_frenet_rotations(
    catom: torch.Tensor, latom: torch.Tensor, ratom: torch.Tensor, legacy: bool = False
) -> torch.Tensor:
    """
    Compute Frenet-Serret rotation matrices from three atoms.

    Constructs local coordinate frames using tangent (T), binormal (B),
    and normal (N) vectors from three consecutive atoms.

    Args:
        catom: Center atom positions [..., 3]
        latom: Left atom positions [..., 3]
        ratom: Right atom positions [..., 3]
        legacy: If True, use legacy frame orientation

    Returns:
        torch.Tensor: Rotation matrices [..., 3, 3] with columns [T, B, N]
    """
    cl = catom - latom
    cr = catom - ratom
    if legacy:
        T = F.normalize(-cr, dim=-1)
        B = F.normalize(torch.linalg.cross(cl, -cr), dim=-1)
        N = F.normalize(torch.linalg.cross(B, T), dim=-1)
    else:
        T = F.normalize(cl, dim=-1)
        B = F.normalize(torch.linalg.cross(cl, cr), dim=-1)
        N = F.normalize(torch.linalg.cross(T, B), dim=-1)
    return torch.stack([T, B, N], dim=-1)


def _compute_frenet_frames(
    token_mask: torch.Tensor,
    trans_mask: torch.Tensor,
    rot_mask: torch.Tensor,
    trans_atom_positions: torch.Tensor,
    rot_catom_positions: torch.Tensor,
    rot_latom_positions: torch.Tensor,
    rot_ratom_positions: torch.Tensor,
    legacy: bool = False,
) -> T:
    """
    Compute Frenet frames for a batch of protein structures.

    Constructs local coordinate frames for each token using Frenet-Serret
    formulas, with separate masks for translation and rotation components.

    Args:
        token_mask: Token validity mask [batch, n_token]
        trans_mask: Translation component mask [batch, n_token]
        rot_mask: Rotation component mask [batch, n_token]
        trans_atom_positions: Atom positions for translation [batch, n_token, 3]
        rot_catom_positions: Center atom positions [batch, n_token, 3]
        rot_latom_positions: Left atom positions [batch, n_token, 3]
        rot_ratom_positions: Right atom positions [batch, n_token, 3]
        legacy: If True, use legacy frame orientation

    Returns:
        T: Transformation object with rotation and translation
    """
    batch_size, n_token = token_mask.shape[0], token_mask.shape[1]

    # Compute rotations
    rots = []
    for i in range(batch_size):
        l = torch.sum(token_mask[i]).long()
        rots_ = torch.zeros(
            (n_token, 3, 3),
            dtype=rot_catom_positions.dtype,
            device=rot_catom_positions.device,
        )
        rots_[:l] = _compute_frenet_rotations(
            catom=rot_catom_positions[i, :l],
            latom=rot_latom_positions[i, :l],
            ratom=rot_ratom_positions[i, :l],
            legacy=legacy,
        )
        rots.append(rots_)

    # Construct frames
    trans = trans_atom_positions * trans_mask[..., None]
    rots = torch.stack(rots, dim=0) * rot_mask[..., None, None]

    return T(rots, trans)


def compute_noisy_structure_frames(
    batch: Dict[str, torch.Tensor], xl: torch.Tensor, legacy: bool = False
) -> T:
    """
    Compute structure frames from noisy atom positions.

    Constructs local coordinate frames using noisy atom positions,
    typically during diffusion sampling.

    Args:
        batch: Batch dictionary with token masks and frame indices
        xl: Noisy atom positions [batch, n_atom, 3]
        legacy: If True, use legacy frame orientation

    Returns:
        T: Transformation object with Frenet frames
    """

    # Gather atoms for rotation component computation
    rot_catom_positions = batched_gather(
        data=xl, inds=batch["token_frame_cindex_adj"], dim=-2, no_batch_dims=1
    )
    rot_latom_positions = batched_gather(
        data=xl, inds=batch["token_frame_lindex_adj"], dim=-2, no_batch_dims=1
    )
    rot_ratom_positions = batched_gather(
        data=xl, inds=batch["token_frame_rindex_adj"], dim=-2, no_batch_dims=1
    )

    # Compute frenet frames
    fi = _compute_frenet_frames(
        token_mask=batch["token_mask"],
        trans_mask=batch["token_mask"],
        rot_mask=batch["token_mask"],
        trans_atom_positions=xl,
        rot_catom_positions=rot_catom_positions,
        rot_latom_positions=rot_latom_positions,
        rot_ratom_positions=rot_ratom_positions,
        legacy=legacy,
    )

    return fi


def compute_conditional_structure_frames(
    batch: Dict[str, torch.Tensor], legacy: bool = False
) -> T:
    """
    Compute structure frames from conditional (ground truth) atom positions.

    Constructs local coordinate frames using ground truth atom positions
    for conditional tokens, respecting structural conditioning masks.

    Args:
        batch: Batch dictionary with gt_atom_positions and conditioning masks
        legacy: If True, use legacy frame orientation

    Returns:
        T: Transformation object with conditional Frenet frames
    """

    # Gather atoms for rotation component computation
    cond_xl = batch["gt_atom_positions"] * batch["cond_struct_mask"][..., None]
    rot_latom_positions = batched_gather(
        data=cond_xl, inds=batch["token_frame_lindex"], dim=-2, no_batch_dims=1
    )
    rot_ratom_positions = batched_gather(
        data=cond_xl, inds=batch["token_frame_rindex"], dim=-2, no_batch_dims=1
    )

    # Compute frenet frames
    cond_fi = _compute_frenet_frames(
        token_mask=batch["token_mask"],
        trans_mask=batch["cond_struct_mask"],
        rot_mask=batch["cond_struct_frame_mask"],
        trans_atom_positions=cond_xl,
        rot_catom_positions=cond_xl,
        rot_latom_positions=rot_latom_positions,
        rot_ratom_positions=rot_ratom_positions,
        legacy=legacy,
    )

    return cond_fi


def weighted_rigid_align(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Perform weighted rigid alignment using Kabsch algorithm.

    Aligns predicted coordinates to ground truth using SVD-based
    optimal rotation, with weighted masking support.

    Args:
        x: Predicted coordinates [..., n_points, 3]
        x_gt: Ground truth coordinates [..., n_points, 3]
        mask: Point weights [..., n_points]
        eps: Small epsilon for numerical stability
        verbose: If True, also return rotation matrix and centroids

    Returns:
        torch.Tensor: Aligned coordinates, or tuple (aligned, R, mu, mu_gt) if verbose
    """
    # Mean-centre positions
    mu = torch.sum(x * mask[..., None], dim=-2) / torch.sum(
        mask + eps, dim=-1, keepdim=True
    )
    mu_gt = torch.sum(x_gt * mask[..., None], dim=-2) / torch.sum(
        mask + eps, dim=-1, keepdim=True
    )
    x = x - mu[..., None, :]
    x_gt = x_gt - mu_gt[..., None, :]

    # Construct covariance matrix
    H = x_gt[..., None] * x[..., None, :]
    H = H * mask[..., None, None]
    H = torch.sum(H, dim=-3)

    dtype = H.dtype

    # TODO: Check why autocast did not work in test
    # Find optimal rotation from single value decomposition
    # SVD (cast to float because doesn't work with bf16/fp16)
    U, _, V = torch.linalg.svd(H.float())

    dets = torch.linalg.det(U @ V).to(dtype=dtype)
    U = U.to(dtype=dtype)
    V = V.to(dtype=dtype)

    # Remove reflection
    F = torch.eye(3, device=x.device, dtype=x.dtype).tile((*H.shape[:-2], 1, 1))
    F[..., -1, -1] = torch.sign(dets)
    R = U @ F @ V

    # Apply alignment
    x_align = x @ R.transpose(-1, -2) + mu_gt[..., None, :]

    if not verbose:
        return x_align.detach()
    else:
        return x_align.detach(), R, mu, mu_gt


def distance(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute distances between pairs of Euclidean coordinates.

    Args:
        p:
            [*, 2, 3] Input tensor where the last two dimensions have
            a shape of [2, 3], representing a pair of coordinates in
            the Euclidean space.

    Returns:
        [*] Output tensor of distances, where each distance is computed
        between the pair of Euclidean coordinates in the last two
        dimensions of the input tensor p.
    """
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5
