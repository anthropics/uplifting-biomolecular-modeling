"""Construct the disordered-PDB ESMFold LMDB and manifest."""

import argparse
import json
import logging
import multiprocessing
import os
import pathlib
from collections import defaultdict

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_MAPPING: dict[str, str] = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def worker_init(cluster_path: pathlib.Path):
    global GLOBAL_CLUSTER_MAPPING
    cluster_mapping = {}
    with open(cluster_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, seq = line.strip().split(",", 2)
            cluster_mapping[seq.replace("X", "A")] = cluster_id.upper()
    GLOBAL_CLUSTER_MAPPING = cluster_mapping


def process_metadata(npz_path: pathlib.Path) -> tuple[str, dict | None, bool]:
    global GLOBAL_CLUSTER_MAPPING
    assert GLOBAL_CLUSTER_MAPPING, (
        "Worker initialization failed: cluster mapping not loaded."
    )

    """Worker function to parse the monomer and extract metadata."""
    entry_id = npz_path.stem
    try:
        prot: protein.Protein = DataPipeline.load(npz_path)
    except Exception as e:
        return npz_path.name, {"error": str(e)}, False

    # Compute the average plddt
    bfactors = prot.b_factors
    bfactors_ca = bfactors[:, 1]
    plddt = np.nanmean(bfactors_ca).item()  # Convert to Python float
    plddt = round(plddt, 2)  # Round to 2 decimal places
    pred_record = metadata.PredictionRecord(model="ESMFold", plddt=plddt)

    cluster_id = GLOBAL_CLUSTER_MAPPING.get(prot.sequence)
    if cluster_id is None:
        logger.warning(f"Cluster ID not found for sequence {prot.sequence}")
        cluster_id = None

    m = metadata.Metadata(
        id=entry_id,
        num_residues=len(prot),
        pred=pred_record,
        cluster_id=cluster_id,
    )
    return entry_id, m.to_dict(), True


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_esm/"
    npz_dir: pathlib.Path = data_dir / "npz"

    if not npz_dir.exists():
        raise FileNotFoundError(f"Preprocessed NPZ directory not found at {npz_dir}")

    # Construct LMDB database
    print("Initializing LMDB database...")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=10 * 1024 * 1024 * 1024,  # size in Bytes
    )
    npz_paths = sorted(list(npz_dir.rglob("*.npz")))
    with env.begin(write=True) as txn:
        for npz_path in tqdm(
            npz_paths,
            desc=f"Processing files in {npz_dir.name}",
            leave=False,
        ):
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            entry_id = npz_path.stem
            txn.put(entry_id.encode(), value_bytes)

    # Close the LMDB environment
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")

    # Read cluster mapping
    cluster_path = data_dir / "rcsb_clusters.csv"

    # Extract metadata
    print("Extracting metadata from NPZ files...")
    metadatas: list[dict] = []
    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(cluster_path,),
    ) as pool:
        npz_paths = list(npz_dir.rglob("*.npz"))
        shard_iterator = pool.imap_unordered(process_metadata, npz_paths, chunksize=50)
        for entry_id, meta_dict, success in tqdm(
            shard_iterator,
            total=len(npz_paths),
            desc="Processing files",
            leave=False,
        ):
            if not success:
                logger.error(
                    f"Failed to process file {entry_id}: {meta_dict.get('error')}"
                )
                continue
            metadatas.append(meta_dict)
    print(f"Total entries processed for metadata extraction: {len(metadatas)}")

    cluster_size: dict[str, int] = defaultdict(int)
    for m in metadatas:
        cluster_id = m["cluster_id"]
        cluster_size[cluster_id] += 1
    for m in metadatas:
        m["cluster_size"] = cluster_size[m["cluster_id"]]

    # Save to msgpack file
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    metadatas.sort(key=lambda x: x["id"])  # Ensure manifest is sorted by ID
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadatas, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")

    manifest_json_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(metadatas, f, indent=2)
    print(f"Saved manifest (json) to {manifest_json_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
