"""Convert accepted ESMFold PDB files to AtlasFold NPZ records."""

import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

from tqdm import tqdm

from atlasfold.common import protein
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF files.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def parse_pdb(name: str, pdb_path: pathlib.Path, out_dir: pathlib.Path):
    prot = protein.Protein.from_pdb(pdb_path, name=name)
    npz_path = out_dir / f"{name}.npz"
    DataPipeline.save(prot, npz_path)


def worker_fn(pdb_path: pathlib.Path, out_dir: pathlib.Path):
    # Output path
    file_name = pdb_path.name.split(".")[0].lower()
    try:
        return parse_pdb(file_name, pdb_path, out_dir)
    except Exception as e:
        logger.error(f"Failed to process ({pdb_path}): {e}")
        raise e


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_esm/"

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for multiprocessing
    worker_wrapped = functools.partial(worker_fn, out_dir=out_dir)

    pdb_dir: pathlib.Path = data_dir / "pdb"
    pdb_paths = list(pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_paths)} PDB files to process.")

    with multiprocessing.Pool(args.num_workers) as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(worker_wrapped, pdb_paths, chunksize=100),
                total=len(pdb_paths),
                desc="Processing RCSB mmCIF files",
            )
        )
    print("Processing completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
