"""Training-time alignment for multimer confidence losses."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Any

import torch

# NOTE: For monomer training, alignment is done only at the atom level,
# without chain permutation.
from atlasfold.train.monomer.structure_alignment import (
    get_aligned_gt_structure as get_atom_aligned_gt_structure,
)
from atlasfold.utils.geometry.rigid_align import get_rigid_transform

CA_IDX = 1

logger = logging.getLogger(__name__)


# ============================================================
# Metadata for chain permutation alignment
# ============================================================
@dataclasses.dataclass(slots=True)
class ChainInfo:
    asym_id: int
    entity_id: int
    indices: torch.Tensor
    res_idx: torch.Tensor


@dataclasses.dataclass(slots=True)
class FullLabelMetadata:
    crop_to_full_idx: torch.Tensor
    asym_id: torch.Tensor
    entity_id: torch.Tensor
    res_idx: torch.Tensor
    resolved_mask: torch.Tensor
    target_positions: dict[int, torch.Tensor]
    chain_info: dict[int, ChainInfo]
    entity_to_sources: dict[int, list[int]]
    residue_lookup: dict[tuple[int, int], int]

    @property
    def has_swappable_target(self) -> bool:
        asym_ids = set(self.target_positions.keys())
        return any(
            len(sources) > 1 and any(aid in asym_ids for aid in sources)
            for sources in self.entity_to_sources.values()
        )


def prepare_alignment_metadata(full_label: dict[str, Any]) -> FullLabelMetadata:
    """Build metadata used by train-time chain permutation alignment."""
    crop_to_full_idx = full_label["crop_to_full_idx"]
    asym_id = full_label["asym_id"]
    entity_id = full_label["entity_id"]
    res_idx = full_label["res_idx"]
    resolved_mask = full_label["resolved_mask"]

    chain_info = _build_chain_info(asym_id, entity_id, res_idx)
    target_positions = _build_crop_positions_by_asym(crop_to_full_idx, asym_id)
    entity_to_sources = _build_entity_to_sources(chain_info)
    residue_lookup = _build_residue_lookup(asym_id, res_idx)

    return FullLabelMetadata(
        crop_to_full_idx=crop_to_full_idx,
        asym_id=asym_id,
        entity_id=entity_id,
        res_idx=res_idx,
        resolved_mask=resolved_mask,
        target_positions=target_positions,
        chain_info=chain_info,
        entity_to_sources=entity_to_sources,
        residue_lookup=residue_lookup,
    )


def _build_chain_info(
    asym_id: torch.Tensor,
    entity_id: torch.Tensor,
    res_idx: torch.Tensor,
) -> dict[int, ChainInfo]:
    chain_info: dict[int, ChainInfo] = {}
    for aid in torch.unique(asym_id, sorted=True).tolist():
        if aid == 0:
            continue
        indices = torch.nonzero(asym_id == aid, as_tuple=False).flatten()
        asym_id_int = int(aid)
        chain_info[asym_id_int] = ChainInfo(
            asym_id=asym_id_int,
            entity_id=int(entity_id[indices[0]].item()),
            indices=indices,
            res_idx=res_idx[indices],
        )
    return chain_info


def _build_crop_positions_by_asym(
    crop_to_full_idx: torch.Tensor,
    asym_id: torch.Tensor,
) -> dict[int, torch.Tensor]:
    positions: dict[int, list[int]] = defaultdict(list)
    for crop_pos, full_idx in enumerate(crop_to_full_idx.tolist()):
        target_asym_id = int(asym_id[full_idx].item())
        if target_asym_id > 0:
            positions[target_asym_id].append(crop_pos)
    return {
        target_asym_id: torch.as_tensor(pos, dtype=torch.long)
        for target_asym_id, pos in positions.items()
    }


def _build_entity_to_sources(
    chain_info: dict[int, ChainInfo],
) -> dict[int, list[int]]:
    entity_to_sources: dict[int, list[int]] = defaultdict(list)
    for asym_id, info in chain_info.items():
        entity_to_sources[info.entity_id].append(asym_id)
    for sources in entity_to_sources.values():
        sources.sort()
    return entity_to_sources


def _build_residue_lookup(
    asym_id: torch.Tensor,
    res_idx: torch.Tensor,
) -> dict[tuple[int, int], int]:
    return {
        (int(asym), int(res)): i
        for i, (asym, res) in enumerate(
            zip(asym_id.tolist(), res_idx.tolist(), strict=True)
        )
    }


# ============================================================
# Align full-complex GT to a cropped training mini-rollout
# ============================================================
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    feat: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, Any] | None,
    permutation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align full-complex GT to a cropped training mini-rollout.

    The chain choice follows the KFold/AlphaFold-Multimer train-time heuristic:
    choose one anchor chain, align that source chain to each possible cropped
    anchor, then greedily assign same-entity chains by centroid distance.


    Parameters
    ----------
    x_pred : torch.Tensor
        Tensor of shape (L, 14, 3) containing predicted coordinates.
    feat : dict[str, torch.Tensor]
        Dictionary containing input data.
    label : dict[str, torch.Tensor]
        Dictionary containing cropped ground truth data.
    full_label : dict[str, Any] | None
        Dictionary containing full ground truth data, or None if not available.
    permutation : bool, optional
        Whether to perform chain permutation alignment. Default is True.
        If False, only the atom-level alignment of the cropped GT to the prediction
        is performed.
    """
    with torch.no_grad(), torch.autocast(x_pred.device.type, enabled=False):
        return _get_aligned_gt_structure(x_pred, feat, label, full_label, permutation)


def _get_aligned_gt_structure(
    x_pred: torch.Tensor,
    feat: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, Any] | None,
    permutation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

    def _fallback() -> tuple[torch.Tensor, torch.Tensor]:
        aatype = feat["aatype_int"]
        x_gt, mask = label["coordinates"], label["resolved_mask"]
        return get_atom_aligned_gt_structure(x_gt, x_pred, aatype, mask=mask)

    if permutation is False:
        # Simple case: no chain permutation, just align the cropped GT to the prediction
        return _fallback()

    assert full_label is not None, (
        "full_label must be provided for chain permutation alignment"
    )

    metadata = full_label["alignment_metadata"]
    assert isinstance(metadata, FullLabelMetadata), (
        "full_label['alignment_metadata'] must be prepared by the multimer dataset"
    )

    # If there are no swappable chains, we can just align the cropped GT to the prediction
    if not metadata.has_swappable_target:
        return _fallback()

    # 1. Chain permutation alignment.
    L_valid = feat["seq_mask"].sum().item()

    crop_to_full_idx = metadata.crop_to_full_idx
    assert crop_to_full_idx.numel() == L_valid, (
        "full_label crop_to_full_idx length does not match valid sequence length: "
        f"{crop_to_full_idx.numel()} != {L_valid}"
    )

    x_pred_crop = x_pred[:L_valid]
    aatype = feat["aatype_int"][:L_valid]
    try:
        target_to_source_asym = _get_greedy_full_label_chain_mapping(
            crop_to_full_idx=crop_to_full_idx,
            full_label=full_label,
            x_pred_crop=x_pred_crop,
            metadata=metadata,
        )
    except (RuntimeError, ValueError, KeyError, IndexError) as e:
        logger.warning(
            "Greedy chain alignment failed; falling back to cropped label: %s", e
        )
        return _fallback()

    source_idx, source_found = _map_crop_to_full_indices(
        crop_to_full_idx=crop_to_full_idx,
        target_to_source_asym=target_to_source_asym,
        metadata=metadata,
    )
    device = x_pred.device
    source_idx = source_idx.to(device=device)
    source_found = source_found.to(device=device)
    x_gt_crop = full_label["coordinates"][source_idx]
    mask_crop = full_label["resolved_mask"][source_idx] & source_found[:, None]

    if mask_crop[:, CA_IDX].sum() < 3:
        logger.warning(
            "Not enough resolved CA atoms in the aligned GT structure; "
            "falling back to cropped label"
        )
        return _fallback()

    x_aligned, mask_aligned = get_atom_aligned_gt_structure(
        x_gt_crop, x_pred_crop, aatype, mask=mask_crop
    )

    out_x = torch.zeros_like(x_pred)
    out_mask = torch.zeros(x_pred.shape[:-1], device=device, dtype=torch.bool)
    out_x[:L_valid] = x_aligned
    out_mask[:L_valid] = mask_aligned
    return out_x, out_mask


def _map_target_positions_to_source_indices(
    target_positions: torch.Tensor,
    crop_to_full_idx: torch.Tensor,
    source_asym: int,
    lookup: dict[tuple[int, int], int],
    full_res_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_indices = []
    found = []
    for target_idx in crop_to_full_idx[target_positions].tolist():
        res_idx = int(full_res_idx[target_idx].item())
        source_idx = lookup.get((source_asym, res_idx))
        source_indices.append(int(target_idx) if source_idx is None else source_idx)
        found.append(source_idx is not None)
    return (
        torch.as_tensor(source_indices, dtype=torch.long),
        torch.as_tensor(found, dtype=torch.bool),
    )


def _map_crop_to_full_indices(
    crop_to_full_idx: torch.Tensor,
    target_to_source_asym: dict[int, int],
    metadata: FullLabelMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_asym_id = metadata.asym_id
    full_res_idx = metadata.res_idx
    lookup = metadata.residue_lookup

    source_indices = []
    found = []
    for target_idx in crop_to_full_idx.tolist():
        target_asym = int(full_asym_id[target_idx].item())
        source_asym = target_to_source_asym.get(target_asym, target_asym)
        res_idx = int(full_res_idx[target_idx].item())
        source_idx = lookup.get((source_asym, res_idx))
        source_indices.append(int(target_idx) if source_idx is None else source_idx)
        found.append(source_idx is not None)

    return (
        torch.as_tensor(source_indices, dtype=torch.long),
        torch.as_tensor(found, dtype=torch.bool),
    )


def _select_anchor_source_asym(
    chain_info: dict[int, ChainInfo],
    entity_to_sources: dict[int, list[int]],
    target_asym_ids: set[int],
    full_ca_mask: torch.Tensor,
) -> int | None:
    candidates = [
        asym_id
        for sources in entity_to_sources.values()
        if len(sources) > 1 and any(aid in target_asym_ids for aid in sources)
        for asym_id in sources
    ]
    if not candidates:
        return None

    def priority(asym_id: int) -> tuple[int, int, int, int]:
        info = chain_info[asym_id]
        n_resolved_ca = int(full_ca_mask[info.indices].sum().item())
        return (
            -len(entity_to_sources[info.entity_id]),
            n_resolved_ca,
            int(info.indices.numel()),
            asym_id,
        )

    return max(candidates, key=priority)


def _find_greedy_chain_assignment(
    distances: torch.Tensor,
    n_resolved: torch.Tensor,
) -> list[int]:
    """Greedily assign each cropped chain to an unused source chain."""
    distances = distances.clone()
    distances.nan_to_num_(nan=1e9, posinf=1e9, neginf=1e9)
    order = torch.argsort(n_resolved.max(dim=1).values, descending=True).tolist()
    assignment = [-1] * distances.shape[0]
    for target_i in order:
        source_i = int(torch.argmin(distances[target_i]).item())
        assignment[target_i] = source_i
        distances[:, source_i] = float("inf")
    return assignment


def _get_greedy_full_label_chain_mapping(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, Any],
    x_pred_crop: torch.Tensor,
    metadata: FullLabelMetadata,
) -> dict[int, int]:
    target_positions = metadata.target_positions
    target_asym_ids = set(target_positions.keys())
    if not target_asym_ids:
        return {}

    chain_info = metadata.chain_info
    full_ca_mask_cpu = metadata.resolved_mask[:, CA_IDX]
    entity_to_sources = metadata.entity_to_sources
    anchor_source = _select_anchor_source_asym(
        chain_info, entity_to_sources, target_asym_ids, full_ca_mask_cpu
    )
    if anchor_source is None:
        return {}

    anchor_entity = chain_info[anchor_source].entity_id
    anchor_targets = [
        asym_id
        for asym_id in entity_to_sources[anchor_entity]
        if asym_id in target_asym_ids
    ]
    if not anchor_targets:
        return {}

    device = x_pred_crop.device
    full_coords = full_label["coordinates"]
    full_mask = full_label["resolved_mask"]

    lookup = metadata.residue_lookup
    best_cost = float("inf")
    best_mapping: dict[int, int] = {}

    for anchor_target in anchor_targets:
        anchor_pos_cpu = target_positions[anchor_target]
        anchor_source_idx_cpu, anchor_found_cpu = _map_target_positions_to_source_indices(
            anchor_pos_cpu,
            crop_to_full_idx,
            anchor_source,
            lookup,
            metadata.res_idx,
        )
        anchor_pos = anchor_pos_cpu.to(device=device)
        anchor_source_idx = anchor_source_idx_cpu.to(device=device)
        anchor_found = anchor_found_cpu.to(device=device)
        anchor_mask = anchor_found & full_mask[anchor_source_idx, CA_IDX]
        if anchor_mask.sum() < 3:
            continue

        rt, trans = get_rigid_transform(
            full_coords[anchor_source_idx, CA_IDX],
            x_pred_crop[anchor_pos, CA_IDX],
            anchor_mask,
        )

        total_cost = 0.0
        mapping: dict[int, int] = {}
        for _entity, sources in entity_to_sources.items():
            if len(sources) <= 1:
                continue
            targets = [asym_id for asym_id in sources if asym_id in target_asym_ids]
            if not targets:
                continue

            distances = torch.full(
                (len(targets), len(sources)),
                float("inf"),
                device=x_pred_crop.device,
            )
            n_resolved = torch.zeros_like(distances)
            for target_i, target_asym in enumerate(targets):
                target_pos_cpu = target_positions[target_asym]
                target_pos = target_pos_cpu.to(device=device)
                x_pred_target = x_pred_crop[target_pos, CA_IDX]
                for source_i, source_asym in enumerate(sources):
                    source_idx_cpu, source_found_cpu = (
                        _map_target_positions_to_source_indices(
                            target_pos_cpu,
                            crop_to_full_idx,
                            source_asym,
                            lookup,
                            metadata.res_idx,
                        )
                    )
                    source_idx = source_idx_cpu.to(device=device)
                    source_found = source_found_cpu.to(device=device)
                    mask = source_found & full_mask[source_idx, CA_IDX]
                    n_resolved[target_i, source_i] = mask.sum()
                    if not mask.any():
                        continue

                    x_gt_source = full_coords[source_idx, CA_IDX] @ rt + trans
                    x_gt_centroid = (x_gt_source * mask[:, None]).sum(dim=0)
                    x_gt_centroid = x_gt_centroid / mask.sum().clamp_min(1)
                    x_pred_centroid = (x_pred_target * mask[:, None]).sum(dim=0)
                    x_pred_centroid = x_pred_centroid / mask.sum().clamp_min(1)
                    distances[target_i, source_i] = torch.norm(
                        x_pred_centroid - x_gt_centroid
                    )

            assignment = _find_greedy_chain_assignment(distances, n_resolved)
            for target_i, source_i in enumerate(assignment):
                if source_i < 0:
                    continue
                target_asym = targets[target_i]
                source_asym = sources[source_i]
                mapping[target_asym] = source_asym
                distance = distances[target_i, source_i]
                total_cost += distance.item() if torch.isfinite(distance) else 1e9

        if total_cost < best_cost:
            best_cost = total_cost
            best_mapping = mapping

    if best_cost == float("inf"):
        return {}
    return {
        target_asym: source_asym
        for target_asym, source_asym in best_mapping.items()
        if target_asym != source_asym
    }
