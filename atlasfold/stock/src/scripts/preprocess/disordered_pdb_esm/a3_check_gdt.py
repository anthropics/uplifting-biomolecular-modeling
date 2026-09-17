"""Filter disordered-PDB ESMFold predictions by GDT-TS."""

import argparse
import logging
import pathlib
import shutil
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from atlasfold.common import protein
from atlasfold.data.fasta import read_fasta
from atlasfold.utils.geometry.rigid_align import rigid_align

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Base directory containing the PDB dataset and FASTA files",
    )
    return parser.parse_args()


def compute_gdt_ts(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Computes a standard GDT_TS approximation using basic Kabsch alignment."""
    if pred_coords.shape[0] != gt_coords.shape[0] or pred_coords.shape[0] == 0:
        logger.error(
            f"Length mismatch: pred ({pred_coords.shape[0]}) vs gt ({gt_coords.shape[0]})"
        )
        return 0.0

    # Align prediction onto ground truth framework
    mask: np.ndarray = ~np.isnan(gt_coords).any(-1)
    if not np.any(mask):
        return 0.0

    aligned_pred = rigid_align(pred_coords, gt_coords, mask, mask_to_zero=False)
    dists = np.linalg.norm(aligned_pred - gt_coords, axis=-1)
    dists = dists[mask]  # Consider only valid residues
    resolved_length = len(dists)

    if resolved_length == 0:
        return 0.0

    gdt_1 = np.sum(dists < 1.0) / resolved_length
    gdt_2 = np.sum(dists < 2.0) / resolved_length
    gdt_4 = np.sum(dists < 4.0) / resolved_length
    gdt_8 = np.sum(dists < 8.0) / resolved_length

    return float((gdt_1 + gdt_2 + gdt_4 + gdt_8) / 4.0 * 100.0)


def main():
    args = parse_args()

    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_esm/"
    gt_dir = data_dir / "npz/"
    pdb_dir = data_dir / "pdb_all/"
    out_dir = data_dir / "pdb/"
    out_dir.mkdir(exist_ok=True)

    # 1. Load ground truth coordinates
    gt_coords_dict: dict[str, np.ndarray] = {}
    for path in tqdm(list(gt_dir.glob("*.npz"))):
        pdb_id = path.stem
        with np.load(path) as data:
            gt_coords_dict[pdb_id] = data["coordinates"][:, 1, :]  # (L, 3) - CA atoms

    # 2. Read FASTA and map sequences to PDB IDs
    fasta_path = data_dir / "rcsb_sequences.fasta"
    sequences: list[tuple[str, str]] = read_fasta(fasta_path)
    seq_to_pdb_ids: dict[str, list[str]] = defaultdict(list)
    for pdb_id, seq in sequences:
        seq_to_pdb_ids[seq].append(pdb_id)

    # 3. Read Unique FASTA for prediction ID mapping
    unique_fasta_path = data_dir / "unique_sequences.fasta"
    uniq_sequences: list[tuple[str, str]] = read_fasta(unique_fasta_path)
    id_to_seq: dict[str, str] = {seq_id: seq for seq_id, seq in uniq_sequences}

    # 4. Process PDB files
    matched_count = 0
    for pdb_file in (pbar := tqdm(sorted(pdb_dir.rglob("*.pdb")))):
        uniq_id = pdb_file.stem.split("_sample")[0]
        seq = id_to_seq[uniq_id]

        pdb_ids = seq_to_pdb_ids.get(seq, [])
        if not pdb_ids:
            logger.warning(f"No matching PDB IDs found for sequence of {pdb_file.name}")
            continue

        # Load prediction protein
        try:
            pred_prot = protein.Protein.from_pdb(pdb_file)
            pred_coords = pred_prot.coordinates[:, 1, :]  # (L, 3) - CA atoms
        except Exception as e:
            logger.error(f"Failed to parse PDB file {pdb_file.name}: {e}")
            continue

        max_gdt = 0.0
        match_found = False

        for pdb_id in pdb_ids:
            if pdb_id not in gt_coords_dict:
                continue

            gt_coords = gt_coords_dict[pdb_id]
            gdt_score = compute_gdt_ts(pred_coords, gt_coords)
            max_gdt = max(max_gdt, gdt_score)

            if gdt_score >= 60.0:
                match_found = True
                break

        if match_found or max_gdt >= 60.0:
            out_path = out_dir / pdb_file.name
            shutil.copy(pdb_file, out_path)
            matched_count += 1
        pbar.set_postfix({"matched": matched_count})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
