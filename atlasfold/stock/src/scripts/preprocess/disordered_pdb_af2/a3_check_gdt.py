"""Filter disordered-PDB AlphaFold2 predictions by GDT-TS."""

import argparse
import csv
import functools
import logging
import multiprocessing
import os
import pathlib
from collections import defaultdict

import numpy as np
from _common import dataset_dir, load_prediction, load_scores, restore_source_sequence
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.utils.geometry.rigid_align import rigid_align

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    return parser.parse_args()


def compute_gdt_ts(pred_ca: np.ndarray, gt_ca: np.ndarray) -> float:
    if pred_ca.shape != gt_ca.shape or pred_ca.shape[0] == 0:
        return 0.0
    mask = np.isfinite(pred_ca).all(-1) & np.isfinite(gt_ca).all(-1)
    if not np.any(mask):
        return 0.0
    aligned = rigid_align(pred_ca, gt_ca, mask, mask_to_zero=False)
    distances = np.linalg.norm(aligned - gt_ca, axis=-1)[mask]
    return float(
        np.mean([np.mean(distances < cutoff) for cutoff in (1.0, 2.0, 4.0, 8.0)]) * 100.0
    )


def process_record(
    record: tuple[str, str],
    data_dir: pathlib.Path,
    sequence_to_gt_ids: dict[str, list[str]],
    threshold: float,
) -> tuple[str, int, str, float, str]:
    seq_id, sequence = record
    output_dir = dataset_dir(data_dir)
    status = "missing_prediction"
    max_gdt = float("nan")
    error = ""
    try:
        pred = load_prediction(data_dir, seq_id)
        restored = restore_source_sequence(pred, sequence, seq_id)
        load_scores(data_dir, seq_id, len(sequence))
        values = []
        for gt_id in sequence_to_gt_ids.get(sequence, []):
            gt_path = output_dir / "ground_truth_npz" / f"{gt_id}.npz"
            with np.load(gt_path) as data:
                gt_ca = data["coordinates"][:, 1, :]
            values.append(compute_gdt_ts(restored.coordinates[:, 1, :], gt_ca))
        max_gdt = max(values, default=0.0)
        status = "accepted" if max_gdt >= threshold else "rejected_gdt"
    except FileNotFoundError as exc:
        error = str(exc)
    except Exception as exc:
        status = "error"
        error = str(exc)
    return seq_id, len(sequence), status, max_gdt, error


def main() -> None:
    args = parse_args()
    output_dir = dataset_dir(args.data_dir)
    source_records = read_fasta(output_dir / "rcsb_sequences.fasta")
    unique_records = read_fasta(output_dir / "unique_sequences.fasta")
    sequence_to_gt_ids: dict[str, list[str]] = defaultdict(list)
    for gt_id, sequence in source_records:
        sequence_to_gt_ids[sequence].append(gt_id)
    wrapped = functools.partial(
        process_record,
        data_dir=args.data_dir,
        sequence_to_gt_ids=dict(sequence_to_gt_ids),
        threshold=args.threshold,
    )
    rows = []
    with multiprocessing.Pool(args.num_workers) as pool:
        iterator = pool.imap_unordered(wrapped, unique_records, chunksize=10)
        for row in tqdm(iterator, total=len(unique_records), desc="Computing GDT-TS"):
            rows.append(row)
            if row[-1] and row[2] == "error":
                logger.error("Failed %s: %s", row[0], row[-1])

    rows.sort(key=lambda row: row[0])

    report_path = output_dir / "gdt_report.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "length", "status", "max_gdt_ts", "error"))
        writer.writerows(rows)
    counts = defaultdict(int)
    for _, _, status, _, _ in rows:
        counts[status] += 1
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
