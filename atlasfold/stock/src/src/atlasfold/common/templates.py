"""Template feature preprocessing utilities.

The multimer template path follows AlphaFold-Multimer's per-chain template
setup: each template hit is aligned to one query chain, then chain-level
template tensors are concatenated over the cropped complex.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from atlasfold.common import protein, residue_constants

MAX_TEMPLATES = 20
N_ATOM = 0
CA_ATOM = 1
C_ATOM = 2
CB_ATOM = 4

TEMPLATE_RESIDUE_FEATURES: tuple[str, ...] = (
    "template.aatype",
    "template.pseudo_beta",
    "template.pseudo_beta_mask",
    "template.backbone_coords",
    "template.backbone_frame_mask",
)


@dataclasses.dataclass(slots=True)
class TemplateHit:
    """A single per-chain template alignment record."""

    template_id: str
    index: int
    entry_indices: np.ndarray
    template_indices: np.ndarray
    release_date: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> TemplateHit:
        return cls(
            template_id=data["template_id"],
            index=int(data["index"]),
            release_date=data.get("release_date"),
            entry_indices=np.asarray(data["entry_indices"], dtype=np.int64),
            template_indices=np.asarray(data["template_indices"], dtype=np.int64),
        )


def make_empty_template_features(
    num_templates: int,
    num_residues: int,
) -> dict[str, np.ndarray]:
    """Create a zero-filled template feature dictionary."""
    restype_dim = len(residue_constants.restypes)
    unknown_idx = residue_constants.restype_orders["X"]
    template_aatype = np.zeros(
        (num_templates, num_residues, restype_dim), dtype=np.float32
    )
    template_aatype[..., unknown_idx] = 1.0
    return {
        "template.mask": np.zeros((num_templates,), dtype=bool),
        "template.aatype": template_aatype,
        "template.pseudo_beta_mask": np.zeros((num_templates, num_residues), dtype=bool),
        "template.pseudo_beta": np.zeros(
            (num_templates, num_residues, 3), dtype=np.float32
        ),
        "template.backbone_coords": np.zeros(
            (num_templates, num_residues, 3, 3), dtype=np.float32
        ),
        "template.backbone_frame_mask": np.zeros(
            (num_templates, num_residues), dtype=bool
        ),
    }


def template_sequence_mismatch_stats(
    query_sequence: str,
    template: protein.Protein,
    hit: TemplateHit,
) -> tuple[float, int, int]:
    """Return aligned residue-type mismatch fraction, count, and aligned count."""
    entry_indices = hit.entry_indices - 1
    template_indices = hit.template_indices - 1
    valid = (
        (entry_indices >= 0)
        & (entry_indices < len(query_sequence))
        & (template_indices >= 0)
        & (template_indices < len(template))
    )
    if not np.any(valid):
        return 0.0, 0, 0

    unknown_idx = residue_constants.restype_orders["X"]
    query_aatype = np.asarray(
        [
            residue_constants.restype_orders.get(query_sequence[i], unknown_idx)
            for i in entry_indices[valid]
        ],
        dtype=np.int64,
    )
    template_aatype = np.asarray(
        [
            residue_constants.restype_orders.get(template.sequence[i], unknown_idx)
            for i in template_indices[valid]
        ],
        dtype=np.int64,
    )
    aligned_count = int(query_aatype.shape[0])
    mismatch_count = int(np.count_nonzero(query_aatype != template_aatype))
    return mismatch_count / aligned_count, mismatch_count, aligned_count


def compute_backbone_frame_mask(atom14_mask: np.ndarray) -> np.ndarray:
    """Return mask for residues with valid N, CA, and C coordinates."""
    return atom14_mask[..., N_ATOM] & atom14_mask[..., CA_ATOM] & atom14_mask[..., C_ATOM]


def compute_backbone_coords(atom14_positions: np.ndarray) -> np.ndarray:
    """Return backbone N, CA, C coordinates."""
    return atom14_positions[..., [N_ATOM, CA_ATOM, C_ATOM], :].astype(np.float32)


def compute_pseudo_beta(
    aatype_int: np.ndarray,
    atom14_positions: np.ndarray,
    atom14_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pseudo-beta coordinates and mask for template residues."""
    gly_idx = residue_constants.restype_orders["G"]
    pseudo_beta_atom = np.full(aatype_int.shape, CB_ATOM, dtype=np.int64)
    pseudo_beta_atom[aatype_int == gly_idx] = CA_ATOM

    gather_idx = pseudo_beta_atom[..., None, None]
    gather_idx = np.broadcast_to(gather_idx, pseudo_beta_atom.shape + (1, 3))
    pseudo_beta = np.take_along_axis(atom14_positions, gather_idx, axis=-2)[..., 0, :]

    mask_idx = pseudo_beta_atom[..., None]
    pseudo_beta_mask = np.take_along_axis(atom14_mask, mask_idx, axis=-1)[..., 0]
    pseudo_beta = np.where(pseudo_beta_mask[..., None], pseudo_beta, 0.0)
    return pseudo_beta.astype(np.float32), pseudo_beta_mask


def featurize_aligned_template(
    template: protein.Protein,
    hit: TemplateHit,
    query_length: int,
) -> dict[str, np.ndarray]:
    """Place one 1-based template alignment onto a query chain."""
    feat = make_empty_template_features(1, query_length)
    if hit.entry_indices.shape != hit.template_indices.shape:
        raise ValueError(
            f"Template hit {hit.template_id} has mismatched entry/template indices."
        )

    entry_indices = hit.entry_indices - 1
    template_indices = hit.template_indices - 1
    valid = (
        (entry_indices >= 0)
        & (entry_indices < query_length)
        & (template_indices >= 0)
        & (template_indices < len(template))
    )
    if not np.any(valid):
        return feat

    q_idx = entry_indices[valid]
    t_idx = template_indices[valid]
    aatype_int = np.asarray(
        [
            residue_constants.restype_orders.get(
                template.sequence[i], residue_constants.restype_orders["X"]
            )
            for i in t_idx
        ],
        dtype=np.int64,
    )

    atom14_positions = np.zeros((query_length, 14, 3), dtype=np.float32)
    atom14_mask = np.zeros((query_length, 14), dtype=bool)
    template_atom14_positions = np.nan_to_num(
        template.coordinates[t_idx], nan=0.0
    ).astype(np.float32)
    template_atom14_mask = template.atom_mask[t_idx]
    atom14_positions[q_idx] = template_atom14_positions
    atom14_mask[q_idx] = template_atom14_mask

    aatype_int_full = np.full(
        query_length,
        residue_constants.restype_orders["X"],
        dtype=np.int64,
    )
    aatype_int_full[q_idx] = aatype_int

    feat["template.mask"][0] = True
    feat["template.aatype"][0, q_idx] = np.eye(
        len(residue_constants.restypes), dtype=np.float32
    )[aatype_int]

    pseudo_beta, pseudo_beta_mask = compute_pseudo_beta(
        aatype_int_full,
        atom14_positions,
        atom14_mask,
    )
    frame_mask = compute_backbone_frame_mask(atom14_mask)
    backbone_coords = compute_backbone_coords(atom14_positions)
    backbone_coords = np.where(frame_mask[:, None, None], backbone_coords, 0.0)
    feat["template.pseudo_beta"][0] = pseudo_beta
    feat["template.pseudo_beta_mask"][0] = pseudo_beta_mask
    feat["template.backbone_coords"][0] = backbone_coords
    feat["template.backbone_frame_mask"][0] = frame_mask
    return feat


def pack_template_features(
    template_features: list[dict[str, np.ndarray]],
    *,
    num_templates: int,
    query_length: int,
) -> dict[str, np.ndarray]:
    """Pack a list of single-template feature dicts into fixed template slots."""
    packed = make_empty_template_features(num_templates, query_length)
    for i, feat in enumerate(template_features[:num_templates]):
        packed["template.mask"][i] = feat["template.mask"][0]
        for key in TEMPLATE_RESIDUE_FEATURES:
            packed[key][i] = feat[key][0]
    return packed


def crop_template_features(
    feat: dict[str, np.ndarray],
    crop_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Crop template features along the residue dimension."""
    cropped = {"template.mask": feat["template.mask"]}
    for key in TEMPLATE_RESIDUE_FEATURES:
        cropped[key] = feat[key][:, crop_indices]
    return cropped


def concat_chain_template_features(
    chain_feats: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate per-chain template features over the residue dimension."""
    if not chain_feats:
        raise ValueError("At least one chain template feature dictionary is required.")
    feat = {
        "template.mask": np.any(
            np.stack([chain_feat["template.mask"] for chain_feat in chain_feats]),
            axis=0,
        )
    }
    for key in TEMPLATE_RESIDUE_FEATURES:
        feat[key] = np.concatenate(
            [chain_feat[key] for chain_feat in chain_feats], axis=1
        )

    return feat


def pad_template_features(
    feat: dict[str, np.ndarray],
    max_length: int,
) -> dict[str, np.ndarray]:
    """Pad template features along the residue dimension to ``max_length``."""
    if not feat:
        return feat
    length = feat[TEMPLATE_RESIDUE_FEATURES[0]].shape[1]
    if length > max_length:
        raise ValueError(
            f"Template residue length {length} exceeds max_length {max_length}."
        )
    pad_len = max_length - length
    if pad_len == 0:
        return feat

    padded = {"template.mask": feat["template.mask"]}
    for key in TEMPLATE_RESIDUE_FEATURES:
        v = feat[key]
        pad_width = [(0, 0), (0, pad_len)] + [(0, 0)] * (v.ndim - 2)
        padded[key] = np.pad(v, pad_width, constant_values=0)
        if key == "template.aatype":
            padded[key][:, length:, residue_constants.restype_orders["X"]] = 1.0
    return padded
