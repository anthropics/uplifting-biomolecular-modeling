"""Pack the RCSB protein-multimer validation structures into LMDB."""

import argparse
import pathlib

import lmdb
import msgpack
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct the RCSB protein-multimer validation LMDB."
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root or rcsb_multimer_val.",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=64,
        help="LMDB map size in GB.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    val_dir = args.data_dir / "rcsb_multimer_val"
    npz_dir = val_dir / "npz"

    val_ids_path = val_dir / "validation_ids.txt"
    manifest_path = val_dir / "manifest.msgpack"
    if not val_ids_path.exists():
        raise FileNotFoundError(f"Validation IDs not found: {val_ids_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")

    with open(val_ids_path) as f:
        entry_ids = [line.strip().lower() for line in f if line.strip()]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("Duplicate entry IDs found in validation_ids.txt")
    print(f"Total validation entries to include: {len(entry_ids)}")

    with open(manifest_path, "rb") as f:
        metadatas = msgpack.unpackb(f.read(), raw=False)
    manifest_ids = [metadata["id"].lower() for metadata in metadatas]
    if manifest_ids != entry_ids:
        raise ValueError(
            "manifest.msgpack entry order does not match validation_ids.txt."
        )

    lmdb_path = val_dir / "structure.lmdb"
    print(f"Creating LMDB database at {lmdb_path}...")
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,
    )
    with env.begin(write=True) as txn:
        for entry_id in tqdm(entry_ids, desc="Writing validation entries"):
            npz_path = npz_dir / entry_id[1:3] / entry_id / f"{entry_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")
            with open(npz_path, "rb") as f:
                txn.put(entry_id.encode(), f.read())
    env.close()

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(entry_ids)}")


if __name__ == "__main__":
    main()
