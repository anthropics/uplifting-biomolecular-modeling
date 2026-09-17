"""
Post-processing utilities for generated structures.

Handles structure cleanup, validation, and output formatting.
"""

import os
import torch
import logging
from typing import Dict, Tuple, List, Any

from genie3.generation.utils.feat_utils import (
    debatchify_tensor_features,
    prepare_tensor_features,
    save_np_features_to_pdb,
)


def postprocess(
    source: str,
    outputs: Dict[str, torch.Tensor],
    batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    outdir: str,
):
    """
    Route postprocessing to appropriate handler based on source type.

    Args:
        source: Source type ('unconditional', 'motif', 'target', 'sidechain')
        outputs: Model outputs including generated structures
        batch: Input batch data
        outdir: Output directory for saving results
    """
    if source == "unconditional":
        _postprocess_unconditional(outputs, batch, outdir)
    elif source == "motif":
        _postprocess_motif(outputs, batch, outdir)
    elif source == "target":
        _postprocess_target(outputs, batch, outdir)
    elif source == "sidechain":
        _postprocess_sidechain(outputs, batch, outdir)
    else:
        logging.error(f"Invalid source for postprocessing: {source}")
        exit(0)


def _postprocess_unconditional(
    outputs: Dict[str, torch.Tensor],
    batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    outdir: str,
):
    """
    Postprocess unconditional generation outputs.

    Args:
        outputs: Model outputs with 'xl' (coordinates) and optionally 'ai' (sequence)
        batch: Input batch data
        outdir: Output directory
    """
    names = [elt[0] for elt in batch[1]]
    batch = prepare_tensor_features(batch[0], reduce_dim=0)
    batch["gt_atom_positions"] = outputs["xl"]
    if "ai" in outputs:
        batch["gt_restype"] = outputs["ai"]
    list_np_features = debatchify_tensor_features(batch)
    for i, np_features in enumerate(list_np_features):

        # Set up directory
        pdb_outdir = os.path.join(outdir, "pdbs")
        if not os.path.exists(pdb_outdir):
            os.makedirs(pdb_outdir, exist_ok=True)

        # Save
        save_np_features_to_pdb(
            residue_index=np_features["residue_index"],
            asym_id=np_features["asym_id"],
            atom_type=np_features["atom_type"],
            atom_symbol=np_features["atom_symbol"],
            atom_mask=np_features["token_struct_mask"],
            atom_positions=np_features["gt_atom_positions"],
            restype=np_features["gt_restype"],
            filepath=os.path.join(pdb_outdir, f"{names[i]}.pdb"),
        )


def _postprocess_motif(
    outputs: Dict[str, torch.Tensor],
    batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    outdir: str,
):
    """
    Postprocess motif scaffolding outputs.

    Args:
        outputs: Model outputs with 'xl' (coordinates)
        batch: Input batch data with motif information
        outdir: Output directory organized by problem
    """
    names = [elt[0] for elt in batch[1]]
    batch = prepare_tensor_features(batch[0], reduce_dim=0)
    batch["gt_atom_positions"] = outputs["xl"]
    list_np_features = debatchify_tensor_features(batch)

    for i, np_features in enumerate(list_np_features):

        # Set up directory
        problem = names[i].rsplit("_", 1)[0]
        pdb_outdir = os.path.join(outdir, problem, "pdbs")
        if not os.path.exists(pdb_outdir):
            os.makedirs(pdb_outdir, exist_ok=True)

        # Save
        save_np_features_to_pdb(
            residue_index=np_features["residue_index"],
            asym_id=np_features["asym_id"],
            atom_type=np_features["atom_type"],
            atom_symbol=np_features["atom_symbol"],
            atom_mask=np_features["token_struct_mask"],
            atom_positions=np_features["gt_atom_positions"],
            restype=np_features["gt_restype"],
            token_group=np_features["cond_group"],
            filepath=os.path.join(pdb_outdir, f"{names[i]}.pdb"),
            default_resname="UNK",
        )


def _postprocess_target(
    outputs: Dict[str, torch.Tensor],
    batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    outdir: str,
):
    """
    Postprocess target-conditioned generation outputs.

    Args:
        outputs: Model outputs with 'xl' (coordinates) and optionally 'ai' (sequence)
        batch: Input batch data with target information
        outdir: Output directory organized by problem
    """
    names = [elt[0] for elt in batch[1]]
    batch = prepare_tensor_features(batch[0], reduce_dim=0)
    batch["gt_atom_positions"] = outputs["xl"]
    if "ai" in outputs:
        batch["gt_restype"] = outputs["ai"]
    list_np_features = debatchify_tensor_features(batch)
    for i, np_features in enumerate(list_np_features):

        # Set up directory
        problem = names[i].rsplit("_", 1)[0]
        pdb_outdir = os.path.join(outdir, problem, "pdbs")
        if not os.path.exists(pdb_outdir):
            os.makedirs(pdb_outdir, exist_ok=True)

        # Save
        save_np_features_to_pdb(
            residue_index=np_features["residue_index"],
            asym_id=np_features["asym_id"],
            atom_type=np_features["atom_type"],
            atom_symbol=np_features["atom_symbol"],
            atom_mask=np_features["token_struct_mask"],
            atom_positions=np_features["gt_atom_positions"],
            restype=np_features["gt_restype"],
            filepath=os.path.join(pdb_outdir, f"{names[i]}.pdb"),
            default_resname="UNK",
        )


def _postprocess_sidechain(
    outputs: Dict[str, torch.Tensor],
    batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    outdir: str,
):
    """
    Postprocess sidechain packing outputs.

    Args:
        outputs: Model outputs with 'xl' (coordinates) and optionally 'ai' (sequence)
        batch: Input batch data
        outdir: Output directory
    """
    names = [elt[0] for elt in batch[1]]
    batch = prepare_tensor_features(batch[0], reduce_dim=0)
    batch["gt_atom_positions"] = outputs["xl"]
    if "ai" in outputs:
        batch["gt_restype"] = outputs["ai"]
    list_np_features = debatchify_tensor_features(batch)
    for i, np_features in enumerate(list_np_features):

        # Set up directory
        pdb_outdir = os.path.join(outdir, "pdbs")
        if not os.path.exists(pdb_outdir):
            os.makedirs(pdb_outdir, exist_ok=True)

        # Save
        save_np_features_to_pdb(
            residue_index=np_features["residue_index"],
            asym_id=np_features["asym_id"],
            atom_type=np_features["atom_type"],
            atom_symbol=np_features["atom_symbol"],
            atom_mask=np_features["token_struct_mask"],
            atom_positions=np_features["gt_atom_positions"],
            restype=np_features["gt_restype"],
            filepath=os.path.join(pdb_outdir, f"{names[i]}.pdb"),
        )
