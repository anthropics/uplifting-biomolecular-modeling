"""Convert accepted disordered-PDB AF2 predictions to AtlasFold NPZ."""

import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

import numpy as np
from _common import (
    accepted_ids,
    dataset_dir,
    load_prediction,
    load_scores,
    restore_source_sequence,
)
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)
SUCCESS = 0
FAILED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    return parser.parse_args()


def worker(
    record: tuple[str, str], data_dir: pathlib.Path, npz_dir: pathlib.Path
) -> tuple[str, int, str]:
    seq_id, sequence = record
    try:
        pred = load_prediction(data_dir, seq_id)
        restored = restore_source_sequence(pred, sequence, seq_id)
        plddt = load_scores(data_dir, seq_id, len(sequence))
        atom_mask = np.isfinite(restored.coordinates).all(-1)
        full_plddt = np.broadcast_to(plddt[:, None], restored.b_factors.shape)
        restored.b_factors[atom_mask] = full_plddt[atom_mask]
        DataPipeline.save(restored, npz_dir / f"{seq_id}.npz")
        return seq_id, SUCCESS, ""
    except Exception as exc:
        return seq_id, FAILED, str(exc)


def main() -> None:
    args = parse_args()
    output_dir = dataset_dir(args.data_dir)
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    accepted = accepted_ids(args.data_dir)
    records = [
        record
        for record in read_fasta(output_dir / "unique_sequences.fasta")
        if record[0] in accepted
    ]
    wrapped = functools.partial(worker, data_dir=args.data_dir, npz_dir=npz_dir)
    failures = []
    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.imap_unordered(wrapped, records, chunksize=20)
        for seq_id, status, error in tqdm(
            results, total=len(records), desc="Converting AF2 predictions"
        ):
            if status == FAILED:
                failures.append((seq_id, error))
                logger.error("Failed %s: %s", seq_id, error)
    if failures:
        raise RuntimeError(f"Failed to convert {len(failures)} predictions")
    print(f"Converted {len(records)} predictions to {npz_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
