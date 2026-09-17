"""concatenate multiple files into one file."""

import argparse
import logging
import multiprocessing
import os
import pathlib

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from atlasfold.common import metadata, protein

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--size_gb",
        type=int,
        default=1000,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def process_metadata(npz_path: pathlib.Path) -> tuple[str, dict | None, bool]:
    """Worker function to parse the monomer and extract metadata."""
    entry_id = npz_path.stem
    try:
        prot = protein.Protein.load_npz(npz_path)
    except Exception as e:
        return npz_path.name, {"error": str(e)}, False

    # Compute the average plddt
    bfactors = prot.b_factors
    bfactors_ca = bfactors[:, 1]
    plddt = np.nanmean(bfactors_ca).item()  # Convert to Python float
    plddt = round(plddt, 2)  # Round to 2 decimal places
    pred_record = metadata.PredictionRecord(model="AF2", plddt=plddt)

    m = metadata.Metadata(
        id=entry_id,
        num_residues=len(prot),
        pred=pred_record,
    )
    return entry_id, m.to_dict(), True


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name
    npz_dir: pathlib.Path = data_dir / "npz"

    if not npz_dir.exists():
        raise FileNotFoundError(f"Preprocessed NPZ directory not found at {npz_dir}")

    # Construct LMDB database
    print("Initializing LMDB database...")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,  # size in Bytes
    )
    subdirs = sorted([d for d in npz_dir.iterdir() if d.is_dir()])
    for subdir in tqdm(subdirs, desc="Constructing LMDB"):
        npz_paths = sorted(list(subdir.glob("*.npz")))
        if not npz_paths:
            continue
        with env.begin(write=True) as txn:
            for npz_path in tqdm(
                npz_paths,
                desc=f"Processing files in {subdir.name}",
                leave=False,
            ):
                with open(npz_path, "rb") as f:
                    value_bytes = f.read()
                entry_id = npz_path.stem
                txn.put(entry_id.encode(), value_bytes)

    # Close the LMDB environment
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")

    # Extract metadata
    print("Extracting metadata from NPZ files...")
    metadatas: list[dict] = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for subdir in tqdm(subdirs, desc="Extracting Metadata"):
            npz_paths = list(subdir.glob("*.npz"))
            if not npz_paths:
                continue

            shard_iterator = pool.imap_unordered(
                process_metadata, npz_paths, chunksize=50
            )
            for entry_id, meta_dict, success in tqdm(
                shard_iterator,
                total=len(npz_paths),
                desc=f"Processing files in {subdir.name}",
                leave=False,
            ):
                if not success:
                    logger.error(
                        f"Failed to process file {entry_id}: {meta_dict.get('error')}"
                    )
                    continue
                metadatas.append(meta_dict)
    print(f"Total entries processed for metadata extraction: {len(metadatas)}")

    # Save to msgpack file
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    metadatas.sort(key=lambda x: x["id"])  # Ensure manifest is sorted by ID
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadatas, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
