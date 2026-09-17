"""Functions for computing confidence metrics from model outputs."""

from collections.abc import Sequence

import numpy as np
import torch


def compute_plddt(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted lDDT-Calpha from the lDDT logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 1:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, num_bins]
    plddt = (probs * bin_centers).sum(dim=-1)  # [*, L]
    plddt = plddt * mask.float()  # [*, L]
    return plddt  # [*, L]


def compute_pae(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted aligned error (PAE) from the PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_pae_from_probs(probs, bin_centers, mask)


def compute_pae_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute PAE from precomputed PAE-bin probabilities."""
    length = probs.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match probabilities length {length}."
        )
    if mask.ndim != probs.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with probabilities shape "
            f"{probs.shape}."
        )

    pae = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pae = pae * pair_mask.float()  # [*, L, L]
    return pae  # [*, L, L]


def compute_pde(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted distance error (PDE) from the PDE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    pde = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pde = pde * pair_mask.float()  # [*, L, L]
    return pde  # [*, L, L]


def compute_ptm(
    logits: torch.Tensor,  # [*, L, L, num_bins]
    bin_centers: torch.Tensor,  # [num_bins]
    mask: torch.Tensor,  # [*, L]
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the predicted TM-score from the PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_ptm_from_probs(probs, bin_centers, mask, eps)


def compute_ptm_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute pTM from precomputed PAE-bin probabilities."""
    length = probs.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match probabilities length {length}."
        )
    if mask.ndim != probs.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with probabilities shape "
            f"{probs.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]

    d0 = d0[..., None, None, None]  # [*, 1, 1, 1]
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[..., :, None] & mask[..., None, :]
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment.masked_fill(~mask, 0.0)
    return weighted.max(-1).values  # [*]


def compute_iptm(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the interface predicted TM-score from PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )
    if asym_id.shape[-1] != length:
        raise ValueError(
            f"asym_id shape {asym_id.shape} does not match logits length {length}."
        )
    if asym_id.ndim == mask.ndim - 1:
        asym_id = asym_id.unsqueeze(-2)
    elif asym_id.ndim != mask.ndim:
        raise ValueError(
            f"asym_id shape {asym_id.shape} is not compatible with mask shape "
            f"{mask.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_iptm_from_probs(probs, bin_centers, asym_id, mask, eps)


def compute_iptm_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ipTM from precomputed PAE-bin probabilities."""
    length = probs.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match probabilities length {length}."
        )
    if mask.ndim != probs.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with probabilities shape "
            f"{probs.shape}."
        )
    if asym_id.shape[-1] != length:
        raise ValueError(
            f"asym_id shape {asym_id.shape} does not match probabilities length {length}."
        )
    if asym_id.ndim == mask.ndim - 1:
        asym_id = asym_id.unsqueeze(-2)
    elif asym_id.ndim != mask.ndim:
        raise ValueError(
            f"asym_id shape {asym_id.shape} is not compatible with mask shape "
            f"{mask.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]

    d0 = d0[..., None, None, None]  # [*, 1, 1, 1]
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[..., :, None] & mask[..., None, :]
    asym_pair_mask = asym_id[..., :, None] != asym_id[..., None, :]
    pair_mask = pair_mask & asym_pair_mask
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment.masked_fill(~mask, 0.0)
    return weighted.max(-1).values  # [*]


def compute_chain_tm_scores_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chain pTM and pairwise interface ipTM on the current device.

    Parameters
    ----------
    probs
        PAE-bin probabilities with shape ``[B, N, L, L, num_bins]``.
    bin_centers
        PAE bin centers with shape ``[num_bins]``.
    asym_id
        Per-residue chain IDs with shape ``[B, L]``. Padding IDs are ignored.
    mask
        Residue mask with shape ``[B, 1, L]``.

    Returns
    -------
    chain_ptm
        Tensor with shape ``[B, N, C]`` where ``C`` is the maximum number of
        chains in the batch.
    interface_iptm
        Symmetric tensor with shape ``[B, N, C, C]``. Entries without a
        corresponding chain pair are NaN.
    """
    if probs.ndim != 5:
        raise ValueError(
            "Expected PAE probabilities with shape [B, N, L, L, num_bins], "
            f"got {probs.shape}."
        )
    batch_size, num_samples, length, pair_length, _ = probs.shape
    if pair_length != length:
        raise ValueError(f"PAE probabilities must be square, got {probs.shape}.")
    if asym_id.shape != (batch_size, length):
        raise ValueError(
            f"Expected asym_id shape {(batch_size, length)}, got {asym_id.shape}."
        )
    if mask.shape != (batch_size, 1, length):
        raise ValueError(
            f"Expected mask shape {(batch_size, 1, length)}, got {mask.shape}."
        )

    valid_mask = mask[:, 0].bool()
    chain_ids_by_batch = [
        torch.unique(asym_id[batch_idx][valid_mask[batch_idx]], sorted=True)
        for batch_idx in range(batch_size)
    ]
    max_chains = max((len(chain_ids) for chain_ids in chain_ids_by_batch), default=0)
    chain_ptm = torch.full(
        (batch_size, num_samples, max_chains),
        torch.nan,
        dtype=probs.dtype,
        device=probs.device,
    )
    interface_iptm = torch.full(
        (batch_size, num_samples, max_chains, max_chains),
        torch.nan,
        dtype=probs.dtype,
        device=probs.device,
    )

    for batch_idx, chain_ids in enumerate(chain_ids_by_batch):
        chain_indices = [
            torch.nonzero(
                valid_mask[batch_idx] & (asym_id[batch_idx] == chain_id),
                as_tuple=False,
            ).squeeze(-1)
            for chain_id in chain_ids
        ]

        for chain_idx, residue_indices in enumerate(chain_indices):
            chain_probs = probs[batch_idx][:, residue_indices]
            chain_probs = chain_probs[:, :, residue_indices]
            chain_mask = torch.ones(
                (num_samples, len(residue_indices)),
                dtype=torch.bool,
                device=probs.device,
            )
            chain_ptm[batch_idx, :, chain_idx] = compute_ptm_from_probs(
                chain_probs,
                bin_centers,
                chain_mask,
            )

        for chain_i, residue_indices_i in enumerate(chain_indices):
            for chain_j in range(chain_i + 1, len(chain_indices)):
                residue_indices_j = chain_indices[chain_j]
                pair_indices = torch.cat((residue_indices_i, residue_indices_j))
                pair_probs = probs[batch_idx][:, pair_indices]
                pair_probs = pair_probs[:, :, pair_indices]
                pair_length = len(pair_indices)
                pair_mask = torch.ones(
                    (num_samples, pair_length),
                    dtype=torch.bool,
                    device=probs.device,
                )
                pair_asym_id = torch.cat(
                    (
                        torch.zeros(
                            len(residue_indices_i),
                            dtype=asym_id.dtype,
                            device=probs.device,
                        ),
                        torch.ones(
                            len(residue_indices_j),
                            dtype=asym_id.dtype,
                            device=probs.device,
                        ),
                    )
                )
                score = compute_iptm_from_probs(
                    pair_probs,
                    bin_centers,
                    pair_asym_id,
                    pair_mask,
                )
                interface_iptm[batch_idx, :, chain_i, chain_j] = score
                interface_iptm[batch_idx, :, chain_j, chain_i] = score

    return chain_ptm, interface_iptm


def compute_fraction_disordered(
    rasa: np.ndarray,
    chain_lengths: Sequence[int],
    smoothing_window: int = 25,
    disorder_cutoff: float = 0.581,
) -> np.ndarray:
    """Compute the fraction of residues classified as disordered from RASA."""
    assert rasa.ndim == 2
    assert sum(chain_lengths) == rasa.shape[1]

    half_window = (smoothing_window - 1) // 2
    smoothing_kernel = np.full(smoothing_window, 1.0 / smoothing_window, dtype=np.float32)
    num_disordered = np.zeros(rasa.shape[0], dtype=np.int64)
    chain_start = 0
    for chain_length in chain_lengths:
        chain_end = chain_start + chain_length
        chain_rasa = rasa[:, chain_start:chain_end]
        for sample_idx in range(rasa.shape[0]):
            padded = np.pad(
                chain_rasa[sample_idx], (half_window, half_window), mode="reflect"
            )
            smoothed = np.convolve(padded, smoothing_kernel, mode="valid")
            num_disordered[sample_idx] += np.count_nonzero(smoothed > disorder_cutoff)
        chain_start = chain_end
    return num_disordered / rasa.shape[1]
