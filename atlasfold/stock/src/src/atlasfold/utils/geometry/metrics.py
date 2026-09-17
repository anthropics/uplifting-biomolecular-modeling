import torch

from .rigid_align import rigid_align, rigid_align_atom14


def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    return torch.norm(x[..., :, None, :] - y[..., None, :, :], dim=-1)


@torch.no_grad()
def compute_rmsd(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    align: bool = False,
):
    """Compute RMSD between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, N, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, N, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, N)
    align : bool, optional
        Whether to perform rigid alignment before computing RMSD

    Returns
    -------
    rmsd : torch.Tensor
        RMSD between predicted and ground truth coordinates, shape (*,)
    """
    if align:
        x_pred = rigid_align(x_pred, x_gt, mask)
    sq_diff = ((x_pred - x_gt) ** 2).sum(-1)  # (*, N)
    w = mask.float()
    return torch.sqrt((sq_diff * w).sum(-1) / w.sum(-1).clamp_min(1))


@torch.no_grad()
def compute_rmsd_atom14(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    align: bool = False,
):
    """Compute RMSD between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, L, 14, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, L, 14, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, L, 14)
    align : bool, optional
        Whether to perform rigid alignment before computing RMSD

    Returns
    -------
    rmsd : torch.Tensor
        RMSD between predicted and ground truth coordinates, shape (*,)
    """
    if align:
        x_pred = rigid_align_atom14(x_pred, x_gt, mask)
    sq_diff = ((x_pred - x_gt) ** 2).sum(-1)  # (*, L, 14)
    w = mask.float()  # (*, L, 14)
    w_sum = w.sum((-1, -2)).clamp_min(1.0)  # (*,)
    return torch.sqrt((sq_diff * w).sum((-1, -2)) / w_sum)


@torch.no_grad()
def compute_lddt(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, N, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, N, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, N)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt : torch.Tensor
        lDDT between predicted and ground truth coordinates, shape (*,)
    """
    pdist_gt = cdist(x_gt, x_gt)  # (*, N, N)
    pdist_pred = cdist(x_pred, x_pred)  # (*, N, N)
    error = torch.abs(pdist_gt - pdist_pred)
    score = torch.zeros_like(error)
    for threshold in [0.5, 1.0, 2.0, 4.0]:
        score += (error < threshold).float()
    score *= 0.25

    # Mask out pairs where either atom is unresolved
    pair_mask = mask[..., :, None] & mask[..., None, :]  # [*, N, N]

    # Exclude self-pairs properly for batched N-dimensional tensors
    N = pair_mask.shape[-1]
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=pair_mask.device)
    pair_mask &= diag_mask

    # Exclude pairs that are too far apart
    pair_mask &= pdist_gt < cutoff

    w = pair_mask.float()
    lddt = (score * w).sum((-1, -2)) / w.sum((-1, -2)).clamp_min(1)
    return lddt


@torch.no_grad()
def compute_lddt_ca(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute CA-lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, L, 14, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, L, 14, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, L, 14)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt-ca : torch.Tensor
        lDDT-Calpha between predicted and ground truth coordinates, shape (*,)
    """
    ca_gt = x_gt[..., 1, :]  # (*, L, 3)
    ca_pred = x_pred[..., 1, :]  # (*, L, 3)
    ca_mask = mask[..., 1]  # (*, L)
    return compute_lddt(ca_pred, ca_gt, ca_mask, cutoff)


@torch.no_grad()
def compute_lddt_fullatom(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute full-atom lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, L, 14, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, L, 14, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, L, 14)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt : torch.Tensor
        lDDT between predicted and ground truth coordinates, shape (*,)
    """
    x_pred_flat = x_pred.flatten(-3, -2)  # (*, L*14, 3)
    x_gt_flat = x_gt.flatten(-3, -2)  # (*, L*14, 3)
    mask_flat = mask.flatten(-2)  # (*, L*14)
    return compute_lddt(x_pred_flat, x_gt_flat, mask_flat, cutoff)
