"""concatenate multiple files into one file."""

import argparse
import json
import pathlib

import lmdb
import msgpack
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=1,
        help="LMDB map size in GB.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "cameo_val"
    npz_dir: pathlib.Path = data_dir / "npz"

    print("Creating LMDB database...")
    metadatas: list[dict] = []
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,  # size in GB
    )
    with env.begin(write=True) as txn:
        # Get chain ids
        npz_paths = sorted(npz_dir.glob("*.npz"))
        chain_ids = [npz_path.stem for npz_path in npz_paths]
        for chain_id in tqdm(chain_ids, desc="Processing entries"):
            npz_path = npz_dir / f"{chain_id}.npz"
            metadata_path = npz_dir / f"{chain_id}.json"
            # Read the npz file as bytes
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            # Put (key, value) pair into the transaction
            txn.put(chain_id.encode(), value_bytes)
            # Load metadata and append to list
            with open(metadata_path) as f:
                m = json.load(f)
            metadatas.append(m)
    env.close()

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(metadatas)}")

    # Save to msgpack file
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadatas, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    main()
