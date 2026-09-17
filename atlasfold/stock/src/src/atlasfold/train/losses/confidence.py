"""Loss functions for confidence prediction heads, including experimentally resolved
prediction, pLDDT, and Predicted Aligned Error (PAE).
"""

import functools

import numpy as np
import torch
import torch.nn.functional as F

from atlasfold.common import residue_constants
from atlasfold.utils.torch_utils import get_one_hot_from_bins, index_select_dim


def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    return torch.norm(x[..., :, None, :] - y[..., None, :, :], dim=-1)


# Compute restype_atom_swap based on restype_ambiguous_atoms
@functools.lru_cache(maxsize=1)
def get_restype_atom_swap(device_type: str) -> torch.Tensor:
    restype_atom_swap: np.ndarray = np.zeros((21, 14), dtype=np.int64)
    for res_name, (group1, group2) in residue_constants.restype_ambiguous_atoms.items():
        restype = residue_constants.restype_orders[
            residue_constants.restype_3to1[res_name]
        ]
        atom14_order = residue_constants.restype_atom14_order[res_name]
        for atom_name1, atom_name2 in zip(group1, group2, strict=True):
            atom_idx1 = atom14_order[atom_name1]
            atom_idx2 = atom14_order[atom_name2]
            restype_atom_swap[restype, atom_idx1] = atom_idx2
            restype_atom_swap[restype, atom_idx2] = atom_idx1
    return torch.from_numpy(restype_atom_swap).to(device_type)


@functools.lru_cache(maxsize=1)
def get_restype_atom14_to_atom37(device_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    gather_indices = torch.from_numpy(residue_constants._gather_indices).to(device_type)
    gather_mask = torch.from_numpy(residue_constants._gather_mask).to(device_type)
    return gather_indices, gather_mask  # [21, 37] each


@functools.lru_cache(maxsize=1)
def get_restype_atom37_mask(device_type: str) -> torch.Tensor:
    mask = torch.from_numpy(residue_constants.restype_atom37_mask).to(device_type)
    return mask  # [21, 37]


class ExperimentallyResolvedPredictionLoss(torch.nn.Module):
    """
    Loss for predicting whether each atom is experimentally resolved or not.
    Based on the AlphaFold2 architecture.
    """

    def forward(
        self,
        logits: torch.Tensor,
        aatype: torch.Tensor,
        resolved_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the experimentally resolved loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, L, 37) containing the logits for each atom being
            experimentally resolved.
        aatype : torch.Tensor
            Tensor of shape (B, L) containing the amino acid type indices.
        resolved_mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for whether each atom
            is experimentally resolved or not.
        pad_mask : torch.Tensor
            Tensor of shape (B, L) containing boolean masks for padded residues.

        Returns
        -------
        resolved_loss : torch.Tensor
            The computed loss for experimentally resolved prediction of shape (B,).
        """
        # Get the resolved mask in atom37 format
        gather_indices, _ = get_restype_atom14_to_atom37(logits.device)  # [21, 37]
        indices = gather_indices[aatype]  # [B, L, 37]
        label = resolved_mask.gather(-1, indices).float()

        # Get atom37 padding mask
        restype_atom37_mask = get_restype_atom37_mask(logits.device)
        atom37_mask = restype_atom37_mask[aatype]  # Shape: [B, L, 37]
        atom37_mask[~pad_mask] = False  # Mask out padded residues

        # Compute Binary Cross Entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")

        # Reduce loss
        w = atom37_mask.float()  # [B, L, 37]
        loss_sum = (loss * w).sum(dim=(-1, -2))  # [B]
        w_sum = w.sum(dim=(-1, -2)).clamp(min=1)  # [B]
        loss_mean = loss_sum / w_sum  # [B]

        return loss_mean


class PLDDTLoss(torch.nn.Module):
    """Loss for predicting lDDT-Calpha."""

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the lDDT-Calpha loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, L, num_bins) containing pLDDT logits.
        bin_centers : torch.Tensor
            Tensor of shape (num_bins,) containing the centers of the bins for pLDDT
        x_pred : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for valid atoms
            in the GT structure.
        aatype : torch.Tensor
            Tensor of shape (B, L) containing the amino acid type indices.

        Returns
        -------
        plddt_loss : torch.Tensor
            The computed pLDDT loss of shape (B,).
        """
        with torch.no_grad():
            lddt = self.get_lddt_ca_score(x_pred, x_gt, mask)  # [B, L]

        lddt_bins = get_one_hot_from_bins(lddt, bin_centers)  # [B, L, num_bins]
        loss = -(lddt_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, L]

        w = mask[..., 1].float()  # [B, L]
        n_valid = w.sum(dim=-1).clamp(min=1)
        loss_mean = (loss * w).sum(dim=-1) / n_valid
        return loss_mean

    def get_lddt_ca_score(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Compute the ground truth LDDT score for each atom based on predicted and
        true coordinates.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates
        mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for valid atoms.

        Returns
        -------
        lddt_score : torch.Tensor
            Tensor of shape (B, L) containing the ground truth LDDT-Calpha.
        """

        # Extract CA coordinates and masks
        x_pred_ca = x_pred[..., 1, :]  # [B, L, 3]
        x_gt_ca = x_gt[..., 1, :]  # [B, L, 3]
        mask_ca = mask[..., 1]  # [B, L]
        del x_pred, x_gt, mask

        d_pred = cdist(x_pred_ca, x_pred_ca)  # [B, L, L]
        d_gt = cdist(x_gt_ca, x_gt_ca)  # [B, L, L]

        # Create pairwise mask for valid CA atoms
        pair_mask = mask_ca[..., :, None] & mask_ca[..., None, :]  # [B, L, L]
        pair_mask.diagonal(dim1=-2, dim2=-1).fill_(False)  # Exclude self-pairs
        pair_mask &= d_gt < 15.0  # Only consider pairs within 15A.

        # Compute LDDT Score
        e = torch.abs(d_gt - d_pred)  # [B, L, L]
        score = torch.zeros_like(e)
        for cutoff in [0.5, 1.0, 2.0, 4.0]:
            score += (e < cutoff).float()
        score *= 0.25
        score.masked_fill_(~pair_mask, 0.0)

        # Aggregate score
        lddt_score = score.sum(dim=-1) / pair_mask.sum(dim=-1).clamp(min=1)
        return lddt_score  # [B, L]


class PDELoss(torch.nn.Module):
    """Loss on Predicted Distance Error (PDE)."""

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
        cbeta_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PDE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, L, L, num_bins) containing PDE logits.
        bin_centers : torch.Tensor
            Tensor of shape (num_bins,) containing the centers of the bins for PDE.
        x_pred : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for valid atoms
            in the GT structure.
        cbeta_idx : torch.Tensor
            Tensor of shape (B, L) containing the index of the C-beta atom
            for each residue (or C-alpha for glycine).

        Returns
        -------
        pde_loss : torch.Tensor
            The computed PDE loss of shape (B,).
        """
        with torch.no_grad():
            e = self.get_distance_error(x_pred, x_gt, cbeta_idx)  # [B, L, L]

        e_bins = get_one_hot_from_bins(e, bin_centers)
        loss = -(e_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, L, L]

        mask_beta = index_select_dim(mask, dim=-1, index=cbeta_idx)  # [B, L]
        pair_mask = mask_beta[..., :, None] & mask_beta[..., None, :]
        pair_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        n_valid = pair_mask.sum(dim=(-1, -2)).clamp(min=1)
        loss_mean = (loss * pair_mask).sum(dim=(-1, -2)) / n_valid
        return loss_mean

    @staticmethod
    def get_distance_error(
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        cbeta_idx: torch.Tensor,
    ) -> torch.Tensor:
        x_pred_beta = index_select_dim(x_pred, dim=-2, index=cbeta_idx)
        x_gt_beta = index_select_dim(x_gt, dim=-2, index=cbeta_idx)

        d_pred = cdist(x_pred_beta, x_pred_beta)
        d_gt = cdist(x_gt_beta, x_gt_beta)
        return torch.abs(d_pred - d_gt)


class PAELoss(torch.nn.Module):
    """Loss on Predicted Aligned Error (PAE)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps: float = eps

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PAE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, L, L, num_bins) containing PAE logits.
        bin_centers : torch.Tensor
            Tensor of shape (num_bins,) containing the centers of the bins for PAE.
        x_pred : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for valid atoms
            in the GT structure.

        Returns
        -------
        pae_loss : torch.Tensor
            The computed PAE loss of shape (B,).
        """
        with torch.no_grad():
            e = self.get_alignment_error(x_pred, x_gt, mask)  # [B, L, L]

        # Compute Cross Entropy Error
        e_bins = get_one_hot_from_bins(e, bin_centers)
        loss = -(e_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, L, L]

        # === Compute validity masks ===
        frame_mask = mask[..., :3].all(dim=-1)  # All backbone atoms must be valid
        ca_mask = mask[..., 1]  # [B, L]
        mask_i = frame_mask & ca_mask
        mask_j = ca_mask
        pair_mask = mask_i[..., :, None] & mask_j[..., None, :]  # [B, L, L]

        # Reduce
        n_valid = pair_mask.sum(dim=(-1, -2)).clamp(min=1)  # [B,]
        loss_mean = (loss * pair_mask).sum(dim=(-1, -2)) / n_valid  # [B,]

        return loss_mean

    def get_alignment_error(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the ground truth alignment error for each pair of representative atoms.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, L, 14, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, L, 14) containing boolean masks for valid atoms
            in the GT structure.

        Returns
        -------
        alignment_error : torch.Tensor
            Tensor of shape (B, L, L) containing the ground truth alignment error for
            each pair of representative atoms.
        """
        # Extract backbone coordinates
        x_gt_backbone = x_gt[..., :3, :]  # [B, L, 3, 3]
        x_pred_backbone = x_pred[..., :3, :]  # [B, L, 3, 3]

        # Compute local frame points for GT and predicted structures
        xij_gt = self.get_local_frame_points(x_gt_backbone)  # [B, L, L, 3]
        xij_pred = self.get_local_frame_points(x_pred_backbone)  # [B, L, L, 3]

        # Compute Euclidean distance between alignments
        e = torch.sqrt((xij_pred - xij_gt).pow(2).sum(-1) + self.eps)

        # Apply mask to set pad atoms to zero
        mask_ca = mask[..., 1]  # [B, L]
        pair_mask = mask_ca[..., :, None] & mask_ca[..., None, :]  # [B, L, L]
        e.masked_fill_(~pair_mask, 0.0)
        return e

    @staticmethod
    def get_local_frame_points(x_backbone: torch.Tensor) -> torch.Tensor:
        x_n, x_ca, x_c = x_backbone.unbind(dim=-2)  # Each is [B, L, 3]
        w1 = x_n - x_ca
        w1 /= w1.norm(dim=-1, keepdim=True) + 1e-8

        w2 = x_c - x_ca
        w2 /= w2.norm(dim=-1, keepdim=True) + 1e-8

        # Build orthogonal frame basis (e1, e2, e3)
        e1 = w1 + w2
        e1 /= e1.norm(dim=-1, keepdim=True) + 1e-8

        e2 = w2 - w1
        e2 /= e2.norm(dim=-1, keepdim=True) + 1e-8

        e3 = torch.linalg.cross(e1, e2, dim=-1)

        # Project onto frame basis
        d_ca = x_ca.unsqueeze(-3) - x_ca.unsqueeze(-2)

        basis = torch.stack([e1, e2, e3], dim=-2)  # [B, L, 3, 3]
        local_coords = basis.unsqueeze(2) @ d_ca.unsqueeze(-1)
        return local_coords.squeeze(-1)
