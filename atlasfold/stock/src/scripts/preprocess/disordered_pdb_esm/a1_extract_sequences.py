"""Preprocess Disordered PDB mmCIF files for the ESMFold dataset."""

import argparse
import io
import logging
import pathlib
from concurrent.futures import ProcessPoolExecutor

import lmdb
import numpy as np
from tqdm import tqdm

from atlasfold.common import protein
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process Disordered PDB mmCIF files.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for save sequences",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for multiprocessing (default: all available CPUs)",
    )
    args = parser.parse_args()
    return args


def process_protein_bytes(npz_bytes):
    """Worker function to process a single protein's byte data.

    Returns the protein object if it meets the disorder criteria, else None.
    """
    if npz_bytes is None:
        return None

    try:
        with io.BytesIO(npz_bytes) as f:
            prot = DataPipeline.load(f)

        ca_coords = prot.coordinates[:, 1, :]  # (L, 3)
        n_disordered = np.isnan(ca_coords).all(-1).sum()

        if n_disordered >= 40:
            return prot
    except Exception as e:
        logger.error(f"Error processing protein data: {e}")

    return None


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    rcsb_dir = args.data_dir / "rcsb/"
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_esm"
    structure_path = rcsb_dir / "structure.lmdb"

    # Step 1: Read all raw bytes from LMDB sequentially (fast I/O operation)
    print("Reading data from LMDB...")
    raw_data_list = []
    env = lmdb.open(str(structure_path), readonly=True, lock=False, readahead=True)
    with env.begin() as txn:
        n_entries = txn.stat()["entries"]
        cursor = txn.cursor()
        for k, npz_bytes in tqdm(cursor, total=n_entries, desc="Loading LMDB"):
            if npz_bytes is None:
                raise ValueError(f"Key {k.decode('utf-8')} not found in LMDB.")
            raw_data_list.append(npz_bytes)
    env.close()

    # Step 2: Process the loaded payloads using multiprocessing
    disordered_proteins: list[protein.Protein] = []
    print(f"Processing data using {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # chunksize prevents excessive IPC overhead between processes
        results = executor.map(process_protein_bytes, raw_data_list, chunksize=16)

        for prot in (pbar := tqdm(results, total=len(raw_data_list))):
            if prot is not None:
                disordered_proteins.append(prot)
                pbar.set_postfix({"Disordered found": len(disordered_proteins)})

    # Print stats
    print(f"Total disordered proteins found: {len(disordered_proteins)}")

    if not disordered_proteins:
        print("No disordered proteins found matching the criteria.")
        return

    # Save outputs
    data_dir.mkdir(parents=True, exist_ok=True)
    disordered_proteins.sort(key=lambda p: p.name)

    # Write FASTA
    fasta_path = data_dir / "rcsb_sequences.fasta"
    with open(fasta_path, "w") as f:
        for prot in disordered_proteins:
            f.write(f">{prot.name}\n{prot.sequence}\n")

    # Save NPZ files
    npz_dir = data_dir / "npz/"
    npz_dir.mkdir(parents=True, exist_ok=True)
    for prot in tqdm(disordered_proteins, desc="Saving npz files"):
        npz_path = npz_dir / f"{prot.name}.npz"
        np.savez_compressed(npz_path, coordinates=prot.coordinates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
