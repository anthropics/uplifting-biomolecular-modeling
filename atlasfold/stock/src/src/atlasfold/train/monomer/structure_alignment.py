import functools

import numpy as np
import torch

from atlasfold.common import residue_constants
from atlasfold.utils.geometry.metrics import compute_rmsd
from atlasfold.utils.geometry.rigid_align import rigid_align_atom14
from atlasfold.utils.torch_utils import gather_dim


# Compute restype_atom_swap based on restype_ambiguous_atoms
@functools.lru_cache(maxsize=1)
def get_restype_atom_swap() -> torch.Tensor:
    restype_atom_swap = np.tile(np.arange(14), (21, 1))
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
    return torch.from_numpy(restype_atom_swap)


@functools.lru_cache(maxsize=1)
def get_restype_has_ambiguous_atoms() -> torch.Tensor:
    has_ambiguous_atoms = np.zeros(21, dtype=bool)
    for res_name in residue_constants.restype_ambiguous_atoms.keys():
        restype = residue_constants.restype_orders[
            residue_constants.restype_3to1[res_name]
        ]
        has_ambiguous_atoms[restype] = True
    return torch.from_numpy(has_ambiguous_atoms)


@torch.no_grad()
def get_aligned_gt_structure(
    x_gt: torch.Tensor,
    x_pred: torch.Tensor,
    aatype: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the aligned ground truth coordinates for the confidence prediction losses.

    Parameters
    ----------
    x_gt : torch.Tensor
        Tensor of shape (L, 14, 3) containing ground truth coordinates.
    x_pred : torch.Tensor
        Tensor of shape (L, 14, 3) containing predicted coordinates.
    aatype : torch.Tensor
        Tensor of shape (L) containing the amino acid type indices.
    mask : torch.Tensor
        Tensor of shape (L, 14) containing boolean masks for resolved atoms.

    Returns
    -------
    x_gt_aligned : torch.Tensor
        Tensor of shape (L, 14, 3) containing the aligned ground truth coordinates.
    mask_aligned : torch.Tensor
        Tensor of shape (L, 14) containing boolean masks for resolved atoms
        in the aligned ground truth structure.
    """
    # Rigid alignment of GT to predicted structure using backbone atoms
    x_gt_aligned_b = rigid_align_atom14(x_gt, x_pred, mask, align_mode="backbone")

    # Get the atom swap indices for each residue type
    restype_has_ambiguous_atoms = get_restype_has_ambiguous_atoms().to(x_gt.device)
    restype_atom_swap = get_restype_atom_swap().to(x_gt.device)

    swap_indices = restype_atom_swap[aatype]  # [L, 14]
    check_ambiguous = restype_has_ambiguous_atoms[aatype]  # [L]
    num_resolved_atoms = mask.sum(-1)  # [L]
    check_ambiguous = check_ambiguous & (num_resolved_atoms >= 3)

    swap_indices = swap_indices[check_ambiguous]  # [Lswap, 14]
    query = x_gt_aligned_b[check_ambiguous]  # [Lswap, 14, 3]
    query_swapped = gather_dim(query, -2, index=swap_indices[..., None])  # [Lswap, 14, 3]
    target = x_pred[check_ambiguous]  # [Lswap, 14, 3]

    m = mask[check_ambiguous]  # [Lswap, 14]
    m_swapped = gather_dim(m, -1, index=swap_indices)  # [Lswap, 14]

    # Compute per-residue RMSD with local-alignment
    err_orig = compute_rmsd(query, target, m, align=True)
    err_swap = compute_rmsd(query_swapped, target, m_swapped, align=True)

    # Choose the original or swapped GT structure based on which has lower local error
    swap_res = err_swap < err_orig  # [N]
    x_to_insert = torch.where(
        swap_res[:, None, None], query_swapped, query
    )  # [Lswap, 14, 3]
    mask_to_insert = torch.where(swap_res[:, None], m_swapped, m)  # [Lswap, 14]

    # Clone from the batched tensors that were NOT overwritten
    final_x_gt = x_gt_aligned_b.clone()
    final_mask = mask.clone()
    final_x_gt[check_ambiguous] = x_to_insert
    final_mask[check_ambiguous] = mask_to_insert

    # Restore the original batch dimensions
    return final_x_gt, final_mask
