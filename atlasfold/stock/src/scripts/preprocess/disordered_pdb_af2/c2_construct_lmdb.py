"""Construct the disordered-PDB AF2 LMDB and training manifest."""

import argparse
import io
import json
import pathlib
from collections import defaultdict

import lmdb
import msgpack
import numpy as np
from _common import dataset_dir
from tqdm import tqdm

from atlasfold.common import metadata
from atlasfold.train.monomer.dataset import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--size_gb", type=int, default=20)
    parser.add_argument("--plddt_threshold", type=float, default=None)
    return parser.parse_args()


def load_clusters(path: pathlib.Path) -> dict[str, str]:
    mapping = {}
    with path.open() as handle:
        next(handle)
        for line in handle:
            entity_id, cluster_id, _ = line.rstrip("\n").split(",", 2)
            mapping[entity_id] = cluster_id.upper()
    return mapping


def main() -> None:
    args = parse_args()
    output_dir = dataset_dir(args.data_dir)
    npz_paths = sorted((output_dir / "npz").glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files in {output_dir / 'npz'}")
    clusters = load_clusters(output_dir / "clusters.csv")

    lmdb_path = output_dir / "structure.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=args.size_gb * 1024**3)
    manifests = []
    with env.begin(write=True) as transaction:
        for path in tqdm(npz_paths, desc="Constructing LMDB"):
            if path.stem not in clusters:
                raise KeyError(f"Missing cluster for {path.stem}")
            raw = path.read_bytes()
            transaction.put(path.stem.encode(), raw)
            prot = DataPipeline.load(io.BytesIO(raw))
            plddt = round(float(np.nanmean(prot.b_factors[:, 1])), 2)
            manifests.append(
                metadata.Metadata(
                    id=path.stem,
                    num_residues=len(prot),
                    cluster_id=clusters[path.stem],
                    pred=metadata.PredictionRecord(model="AlphaFold2", plddt=plddt),
                ).to_dict()
            )
    env.close()

    cluster_sizes: dict[str, int] = defaultdict(int)
    for record in manifests:
        cluster_sizes[record["cluster_id"]] += 1
    for record in manifests:
        record["cluster_size"] = cluster_sizes[record["cluster_id"]]
    manifests.sort(key=lambda record: record["id"])
    with (output_dir / "manifest.msgpack").open("wb") as handle:
        msgpack.pack(manifests, handle)
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifests, handle, indent=2)

    if args.plddt_threshold is not None:
        filtered = [
            record
            for record in manifests
            if record["pred"]["plddt"] >= args.plddt_threshold
        ]
        suffix = f"{args.plddt_threshold:g}".replace(".", "p")
        with (output_dir / f"manifest_plddt{suffix}.msgpack").open("wb") as handle:
            msgpack.pack(filtered, handle)
        print(f"pLDDT-filtered entries: {len(filtered)}")
    print(f"Constructed {lmdb_path} with {len(manifests)} entries")


if __name__ == "__main__":
    main()
