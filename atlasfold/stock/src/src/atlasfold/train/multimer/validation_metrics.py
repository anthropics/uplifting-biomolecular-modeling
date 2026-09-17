"""Validation metrics for protein multimer structure prediction."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch

from atlasfold.train.monomer.structure_alignment import (
    get_aligned_gt_structure as get_atom_aligned_gt_structure,
)
from atlasfold.utils.geometry.metrics import (
    compute_lddt,
    compute_lddt_ca,
    compute_lddt_fullatom,
    compute_rmsd,
    compute_rmsd_atom14,
)
from atlasfold.utils.geometry.rigid_align import rigid_align_atom14

CA_IDX = 1
CONTACT_CUTOFF = 8.0
DEFAULT_MAX_CHAIN_PERMUTATIONS = 100


def compute_distogram_contact_probability(
    logits: torch.Tensor,
    boundaries: torch.Tensor,
    cutoff: float = CONTACT_CUTOFF,
) -> torch.Tensor:
    """Compute contact probability from distogram logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    if probs.shape[-1] != boundaries.numel() + 1:
        raise ValueError(
            "distogram logits/bin mismatch: "
            f"{probs.shape[-1]} != {boundaries.numel()} + 1"
        )

    if boundaries.numel() == 0:
        contact_bins = torch.ones(1, device=logits.device, dtype=torch.bool)
    elif boundaries.numel() == 1:
        centers = boundaries.new_tensor([boundaries[0] * 0.5, boundaries[0] * 1.5])
        contact_bins = centers < cutoff
    else:
        step = boundaries[1] - boundaries[0]
        first_center = boundaries[0] - 0.5 * step
        middle_centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        last_center = boundaries[-1] + 0.5 * step
        centers = torch.cat(
            [first_center[None], middle_centers, last_center[None]], dim=0
        )
        contact_bins = centers < cutoff

    if not bool(contact_bins.any()):
        contact_bins[0] = True
    return probs[..., contact_bins].sum(dim=-1)


def compute_global_pde(
    pde_score: torch.Tensor,
    prob_contact: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute contact-probability-weighted global PDE for sample ranking."""
    if pde_score.ndim != 3:
        raise ValueError(f"Expected pde_score shape [N, L, L], got {pde_score.shape}")
    if prob_contact.shape != pde_score.shape[-2:]:
        raise ValueError(
            "prob_contact shape does not match pde_score residues: "
            f"{prob_contact.shape} != {pde_score.shape[-2:]}"
        )

    pair_mask = seq_mask.bool()[:, None] & seq_mask.bool()[None, :]
    prob_contact = prob_contact.float() * pair_mask.float()
    weighted_pde = torch.einsum("sij,ij->s", pde_score.float(), prob_contact)
    return weighted_pde / prob_contact.sum(dim=(-1, -2)).clamp_min(1e-6)


def _iter_chain_indices(
    batch: dict[str, torch.Tensor],
) -> list[dict[str, int | torch.Tensor]]:
    """Return contiguous protein chain residue indices from an unbatched feature dict."""
    seq_mask = batch["seq_mask"].bool()
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]

    chains = []
    seen_asym_ids = torch.unique(asym_id[seq_mask], sorted=True)
    for aid in seen_asym_ids.tolist():
        if aid == 0:
            continue
        indices = torch.nonzero(seq_mask & (asym_id == aid), as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        chains.append(
            {
                "asym_id": int(aid),
                "entity_id": int(entity_id[indices[0]].item()),
                "indices": indices,
            }
        )
    return chains


def _get_low_homology_chain_asym_ids(label: dict[str, torch.Tensor]) -> set[int]:
    if "low_homology_chain_asym_ids" not in label:
        return set()
    return {
        int(asym_id)
        for asym_id in label["low_homology_chain_asym_ids"].detach().cpu().tolist()
        if int(asym_id) > 0
    }


def _get_low_homology_interface_asym_ids(
    label: dict[str, torch.Tensor],
) -> set[tuple[int, int]]:
    if "low_homology_interface_asym_ids" not in label:
        return set()

    interface_pairs = label["low_homology_interface_asym_ids"].detach().cpu().tolist()
    low_homology_interfaces = set()
    for asym_i, asym_j in interface_pairs:
        asym_i, asym_j = int(asym_i), int(asym_j)
        if asym_i <= 0 or asym_j <= 0:
            continue
        low_homology_interfaces.add(
            (asym_i, asym_j) if asym_i <= asym_j else (asym_j, asym_i)
        )
    return low_homology_interfaces


def _compute_pair_lddt(
    pdist_pred: torch.Tensor,
    pdist_gt: torch.Tensor,
) -> torch.Tensor:
    score = torch.zeros_like(pdist_gt)
    for cutoff in (0.5, 1.0, 2.0, 4.0):
        score += (torch.abs(pdist_gt - pdist_pred) < cutoff).float()
    return score * 0.25


def _generate_chain_permutations(
    chains: list[dict[str, int | torch.Tensor]],
    max_permutations: int = DEFAULT_MAX_CHAIN_PERMUTATIONS,
) -> Iterable[list[int]]:
    """Generate target-chain to source-chain maps for identical chain groups."""
    target_to_source = list(range(len(chains)))

    entity_groups: dict[int, list[int]] = defaultdict(list)
    for i, chain in enumerate(chains):
        entity_groups[int(chain["entity_id"])].append(i)

    swappable_groups: list[list[int]] = []
    for group in entity_groups.values():
        if len(group) <= 1:
            continue
        lengths = {
            int(chains[i]["indices"].numel())  # type: ignore[union-attr]
            for i in group
        }
        if len(lengths) == 1:
            swappable_groups.append(group)

    num_permutations = math.prod(math.factorial(len(g)) for g in swappable_groups)
    if num_permutations > max_permutations:
        yielded = {tuple(target_to_source)}
        yield target_to_source
        attempts = 0
        max_attempts = max_permutations * 20
        while len(yielded) < max_permutations and attempts < max_attempts:
            attempts += 1
            mapping = list(target_to_source)
            for group in swappable_groups:
                perm = torch.randperm(len(group)).tolist()
                for target_i, source_pos in zip(group, perm, strict=True):
                    mapping[target_i] = group[source_pos]
            key = tuple(mapping)
            if key in yielded:
                continue
            yielded.add(key)
            yield mapping
        return

    if not swappable_groups:
        yield target_to_source
        return

    group_perms = [list(itertools.permutations(group)) for group in swappable_groups]
    for permuted_groups in itertools.product(*group_perms):
        mapping = list(target_to_source)
        for group, perm in zip(swappable_groups, permuted_groups, strict=True):
            for target_i, source_i in zip(group, perm, strict=True):
                mapping[target_i] = source_i
        yield mapping


def _apply_chain_permutation(
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    chains: list[dict[str, int | torch.Tensor]],
    target_to_source: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_perm = x_gt.clone()
    mask_perm = mask.clone()
    for target_i, source_i in enumerate(target_to_source):
        if target_i == source_i:
            continue
        target_idx = chains[target_i]["indices"]
        source_idx = chains[source_i]["indices"]
        x_perm[target_idx] = x_gt[source_idx]
        mask_perm[target_idx] = mask[source_idx]
    return x_perm, mask_perm


def _align_on_ca(
    x_gt: torch.Tensor,
    x_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ca_mask = mask[:, CA_IDX]
    if ca_mask.sum() < 3:
        aligned = x_gt.clone()
        aligned.masked_fill_(~mask[..., None], 0.0)
        return aligned
    return rigid_align_atom14(
        x_gt.float(),
        x_pred.float(),
        mask.bool(),
        align_mode="ca",
        mask_to_zero=True,
    )


@torch.no_grad()
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    permute_chains: bool = True,
    max_chain_permutations: int = DEFAULT_MAX_CHAIN_PERMUTATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align protein-only multimer GT to a prediction.

    The alignment is invariant to rigid transforms, same-entity whole-chain
    permutations, and residue-level atom14 ambiguity.
    """
    x_gt = label["coordinates"].float()
    mask = label["resolved_mask"].bool()

    best_x, best_mask = x_gt, mask
    if permute_chains:
        chains = _iter_chain_indices(batch)
        best_rmsd = float("inf")
        for target_to_source in _generate_chain_permutations(
            chains, max_permutations=max_chain_permutations
        ):
            x_perm, mask_perm = _apply_chain_permutation(
                x_gt, mask, chains, target_to_source
            )
            x_aligned = _align_on_ca(x_perm, x_pred, mask_perm)
            ca_mask = mask_perm[:, CA_IDX]
            if ca_mask.any():
                rmsd = compute_rmsd(
                    x_aligned[:, CA_IDX],
                    x_pred[:, CA_IDX],
                    ca_mask,
                    align=False,
                ).item()
            else:
                rmsd = float("inf")
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_x, best_mask = x_perm, mask_perm

    if "aatype_int" not in batch:
        return _align_on_ca(best_x, x_pred, best_mask), best_mask

    return get_atom_aligned_gt_structure(
        x_gt=best_x,
        x_pred=x_pred.float(),
        aatype=batch["aatype_int"],
        mask=best_mask,
    )


@torch.no_grad()
def _compute_single_sample_metrics(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> dict[str, float]:
    x_gt, mask = get_aligned_gt_structure(x_pred, batch, label)
    seq_mask = batch["seq_mask"].bool()
    atom_mask = mask & seq_mask[:, None]
    ca_mask = atom_mask[:, CA_IDX]

    metrics: dict[str, float] = {}
    metrics["complex/rmsd"] = compute_rmsd_atom14(x_pred, x_gt, atom_mask).item()
    metrics["complex/lddt"] = compute_lddt_fullatom(
        x_pred,
        x_gt,
        atom_mask,
        cutoff=15.0,
    ).item()
    metrics["complex/lddt-ca"] = compute_lddt_ca(
        x_pred,
        x_gt,
        atom_mask,
        cutoff=15.0,
    ).item()

    chains = _iter_chain_indices(batch)
    low_homology_chains = _get_low_homology_chain_asym_ids(label)
    chain_rmsds = []
    chain_lddts = []
    chain_lddt_cas = []
    for chain in chains:
        if int(chain["asym_id"]) not in low_homology_chains:
            continue
        idx = chain["indices"]
        chain_atom_mask = atom_mask[idx]
        chain_ca_mask = ca_mask[idx]
        if chain_ca_mask.sum() < 2:
            continue
        chain_rmsds.append(
            compute_rmsd(
                x_pred[idx, CA_IDX],
                x_gt[idx, CA_IDX],
                chain_ca_mask,
                align=True,
            ).item()
        )
        if chain_atom_mask.sum() > 1:
            chain_lddts.append(
                compute_lddt_fullatom(
                    x_pred[idx],
                    x_gt[idx],
                    chain_atom_mask,
                    cutoff=15.0,
                ).item()
            )
        chain_lddt_cas.append(
            compute_lddt(
                x_pred[idx, CA_IDX],
                x_gt[idx, CA_IDX],
                chain_ca_mask,
                cutoff=15.0,
            ).item()
        )
    if chain_rmsds:
        metrics["chain/rmsd"] = float(np.mean(chain_rmsds))
    if chain_lddts:
        metrics["chain/lddt"] = float(np.mean(chain_lddts))
    if chain_lddt_cas:
        metrics["chain/lddt-ca"] = float(np.mean(chain_lddt_cas))

    interface_lddts = []
    interface_lddt_cas = []
    low_homology_interfaces = _get_low_homology_interface_asym_ids(label)
    ca_pdist_gt = torch.cdist(x_gt[:, CA_IDX], x_gt[:, CA_IDX])
    ca_pdist_pred = torch.cdist(x_pred[:, CA_IDX], x_pred[:, CA_IDX])
    ca_pair_score = _compute_pair_lddt(ca_pdist_pred, ca_pdist_gt)

    for i, chain_i in enumerate(chains):
        idx_i = chain_i["indices"]
        for chain_j in chains[i + 1 :]:
            asym_i = int(chain_i["asym_id"])
            asym_j = int(chain_j["asym_id"])
            interface_key = (asym_i, asym_j) if asym_i <= asym_j else (asym_j, asym_i)
            if interface_key not in low_homology_interfaces:
                continue

            idx_j = chain_j["indices"]
            pair_mask = ca_mask[idx_i, None] & ca_mask[idx_j][None, :]
            pair_mask &= ca_pdist_gt[idx_i[:, None], idx_j[None, :]] < 15.0
            if pair_mask.any():
                score = ca_pair_score[idx_i[:, None], idx_j[None, :]]
                interface_lddt_cas.append(score[pair_mask].mean().item())

            pred_i = x_pred[idx_i].reshape(-1, 3)
            pred_j = x_pred[idx_j].reshape(-1, 3)
            gt_i = x_gt[idx_i].reshape(-1, 3)
            gt_j = x_gt[idx_j].reshape(-1, 3)
            atom_mask_i = atom_mask[idx_i].reshape(-1)
            atom_mask_j = atom_mask[idx_j].reshape(-1)
            atom_pair_mask = atom_mask_i[:, None] & atom_mask_j[None, :]
            atom_pdist_gt = torch.cdist(gt_i, gt_j)
            atom_pair_mask &= atom_pdist_gt < 15.0
            if atom_pair_mask.any():
                atom_pdist_pred = torch.cdist(pred_i, pred_j)
                atom_score = _compute_pair_lddt(atom_pdist_pred, atom_pdist_gt)
                interface_lddts.append(atom_score[atom_pair_mask].mean().item())
    if interface_lddts:
        metrics["interface/lddt"] = float(np.mean(interface_lddts))
    if interface_lddt_cas:
        metrics["interface/lddt-ca"] = float(np.mean(interface_lddt_cas))

    return metrics


def compute_validation_metric(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    rank_idx: int | None = None,
) -> dict[str, float]:
    """Compute aggregate validation metrics for multimer diffusion samples."""
    sample_metrics = [
        _compute_single_sample_metrics(x_pred_i, batch, label) for x_pred_i in x_pred
    ]
    metric_names = sorted(set().union(*(m.keys() for m in sample_metrics)))

    all_metrics: dict[str, float] = {}
    for name in metric_names:
        values = [m[name] for m in sample_metrics if name in m]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        all_metrics[f"avg/{name}"] = float(arr.mean())
        if "rmsd" in name:
            all_metrics[f"top/{name}"] = float(arr.min())
        else:
            all_metrics[f"top/{name}"] = float(arr.max())
        if rank_idx is not None and name in sample_metrics[rank_idx]:
            all_metrics[f"rank/{name}"] = float(sample_metrics[rank_idx][name])

    return all_metrics
