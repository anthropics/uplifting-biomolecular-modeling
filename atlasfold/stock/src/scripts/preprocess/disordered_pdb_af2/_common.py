"""Shared helpers for disordered-PDB AlphaFold2 preprocessing."""

import csv
import json
import pathlib

import numpy as np

from atlasfold.common import protein

DATASET_NAME = "disordered_pdb_af2"


def dataset_dir(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / DATASET_NAME


def load_prediction(data_dir: pathlib.Path, seq_id: str) -> protein.Protein:
    path = dataset_dir(data_dir) / "pdb" / f"{seq_id}.pdb"
    if not path.is_file():
        raise FileNotFoundError(path)
    return protein.Protein.from_pdb(path, name=seq_id)


def load_scores(data_dir: pathlib.Path, seq_id: str, length: int) -> np.ndarray:
    path = dataset_dir(data_dir) / "scores" / f"{seq_id}.json"
    with path.open() as handle:
        scores = json.load(handle)
    plddt = np.asarray(scores.get("plddt"), dtype=np.float32)
    if plddt.shape != (length,):
        raise ValueError(
            f"Invalid pLDDT shape in {path}: {plddt.shape}, expected {(length,)}"
        )
    if not np.isfinite(plddt).all() or np.any((plddt < 0) | (plddt > 100)):
        raise ValueError(f"Invalid pLDDT values in {path}")
    return plddt


def restore_source_sequence(
    pred: protein.Protein, source_sequence: str, name: str
) -> protein.Protein:
    """Restore FASTA positions omitted from a ColabFold PDB, including X."""
    length = len(source_sequence)
    coordinates = np.full((length, 14, 3), np.nan, dtype=np.float32)
    b_factors = np.full((length, 14), np.nan, dtype=np.float32)
    if pred.residue_index is None or len(pred.residue_index) != len(pred):
        raise ValueError(f"Prediction has no usable residue indices: {name}")
    indices = pred.residue_index.astype(np.int64) - 1
    if np.any(indices < 0) or np.any(indices >= length):
        raise ValueError(f"Prediction residue index outside FASTA bounds: {name}")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"Duplicate prediction residue indices: {name}")
    indexed_sequence = "".join(source_sequence[index] for index in indices)
    if indexed_sequence != pred.sequence:
        raise ValueError(f"PDB residue indices do not match source FASTA: {name}")
    if pred.sequence != source_sequence.replace("X", ""):
        raise ValueError(f"PDB sequence does not equal FASTA with X removed: {name}")
    coordinates[indices] = pred.coordinates
    b_factors[indices] = pred.b_factors
    return protein.Protein.create(name, source_sequence, coordinates, b_factors)


def accepted_ids(data_dir: pathlib.Path) -> set[str]:
    path = dataset_dir(data_dir) / "gdt_report.csv"
    with path.open() as handle:
        return {
            row["id"] for row in csv.DictReader(handle) if row["status"] == "accepted"
        }
