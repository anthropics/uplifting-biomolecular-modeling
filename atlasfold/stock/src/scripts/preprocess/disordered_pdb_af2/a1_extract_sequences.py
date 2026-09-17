"""Extract unresolved RCSB chains for the disordered-PDB AF2 dataset."""

import argparse
import io
import logging
import pathlib
from concurrent.futures import ProcessPoolExecutor

import lmdb
import numpy as np
from _common import dataset_dir
from tqdm import tqdm

from atlasfold.common import protein
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--source_fasta",
        type=pathlib.Path,
        help="Reuse an existing disordered-PDB rcsb_sequences.fasta.",
    )
    parser.add_argument(
        "--rcsb_lmdb",
        type=pathlib.Path,
        help="RCSB structure LMDB; defaults to <data_dir>/rcsb/structure.lmdb.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Optionally retain only chains at or below this sequence length.",
    )
    return parser.parse_args()


def process_protein_bytes(npz_bytes: bytes) -> protein.Protein | None:
    try:
        with io.BytesIO(npz_bytes) as handle:
            prot = DataPipeline.load(handle)
        ca_coordinates = prot.coordinates[:, 1, :]
        if np.isnan(ca_coordinates).all(-1).sum() >= 40:
            return prot
    except Exception as error:
        logger.error("Failed to inspect protein: %s", error)
    return None


def main() -> None:
    args = parse_args()
    structure_path = args.rcsb_lmdb or args.data_dir / "rcsb" / "structure.lmdb"
    output_dir = dataset_dir(args.data_dir)
    gt_dir = output_dir / "ground_truth_npz"
    gt_dir.mkdir(parents=True, exist_ok=True)

    if args.source_fasta is not None:
        from atlasfold.data.fasta import read_fasta

        records = read_fasta(args.source_fasta)
        if args.max_length is not None:
            records = [record for record in records if len(record[1]) <= args.max_length]
        env = lmdb.open(str(structure_path), readonly=True, lock=False, readahead=True)
        proteins = []
        with env.begin() as transaction:
            for name, expected_sequence in tqdm(
                records, desc="Loading selected RCSB chains"
            ):
                raw = transaction.get(name.encode())
                if raw is None:
                    raise KeyError(f"RCSB LMDB key not found: {name}")
                with io.BytesIO(raw) as handle:
                    prot = DataPipeline.load(handle)
                if prot.sequence != expected_sequence:
                    raise ValueError(f"RCSB sequence mismatch for {name}")
                proteins.append(prot)
        env.close()
        with (output_dir / "rcsb_sequences.fasta").open("w") as handle:
            for prot in proteins:
                handle.write(f">{prot.name}\n{prot.sequence}\n")
                np.savez_compressed(
                    gt_dir / f"{prot.name}.npz", coordinates=prot.coordinates
                )
        print(f"Reused {len(proteins)} disordered chains in {output_dir}")
        return

    raw_records = []
    env = lmdb.open(str(structure_path), readonly=True, lock=False, readahead=True)
    with env.begin() as transaction:
        cursor = transaction.cursor()
        total = transaction.stat()["entries"]
        for _, value in tqdm(cursor, total=total, desc="Loading RCSB LMDB"):
            raw_records.append(value)
    env.close()

    proteins = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = executor.map(process_protein_bytes, raw_records, chunksize=16)
        for prot in tqdm(results, total=len(raw_records), desc="Selecting chains"):
            if prot is not None:
                proteins.append(prot)
    proteins.sort(key=lambda item: item.name)

    with (output_dir / "rcsb_sequences.fasta").open("w") as handle:
        for prot in proteins:
            handle.write(f">{prot.name}\n{prot.sequence}\n")
            np.savez_compressed(gt_dir / f"{prot.name}.npz", coordinates=prot.coordinates)
    print(f"Saved {len(proteins)} disordered chains to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
