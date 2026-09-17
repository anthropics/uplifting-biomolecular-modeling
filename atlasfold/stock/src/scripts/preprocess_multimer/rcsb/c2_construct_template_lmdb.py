"""Construct LMDB database for processed template structures."""

import argparse
import pathlib

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Construct template LMDB database.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root or rcsb_multimer directory.",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=256,
        help="LMDB map size in GB.",
    )
    args = parser.parse_args()
    return args


def read_npz_metadata(npz_path: pathlib.Path) -> dict:
    with np.load(npz_path) as data:
        name = data["name"].item().decode("utf-8")
        sequence = data["sequence"].item().decode("utf-8")
    return {
        "id": name,
        "num_residues": len(sequence),
    }


def main():
    args = parse_args()
    data_dir = args.data_dir / "rcsb_multimer"
    npz_dir = data_dir / "templates" / "npz"
    if not npz_dir.exists():
        raise FileNotFoundError(f"Processed template NPZ directory not found: {npz_dir}")

    npz_paths = sorted(npz_dir.rglob("*.npz"))
    print(f"Found {len(npz_paths)} processed template NPZ files.")

    lmdb_path = data_dir / "template.lmdb"
    print("Creating template LMDB database...")
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,
    )

    metadatas: list[dict] = []
    with env.begin(write=True) as txn:
        for npz_path in tqdm(npz_paths, desc="Writing templates"):
            template_id = npz_path.stem
            with open(npz_path, "rb") as f:
                txn.put(template_id.encode(), f.read())
            metadatas.append(read_npz_metadata(npz_path))
    env.close()

    manifest_path = data_dir / "template_manifest.msgpack"
    metadatas.sort(key=lambda x: x["id"])
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadatas, f)

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Total templates written: {len(metadatas)}")


if __name__ == "__main__":
    main()
