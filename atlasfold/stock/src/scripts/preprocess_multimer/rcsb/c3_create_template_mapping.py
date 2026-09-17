"""Create entry-to-template mapping metadata from template hit NPZ files."""

import argparse
import logging
import multiprocessing
import os
import pathlib
import pickle
from datetime import datetime, timedelta

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Create template mapping metadata.")
    parser.add_argument(
        "--metadata_dir",
        type=pathlib.Path,
        required=True,
        help="Path to train_template_metadata directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root or rcsb_multimer directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def load_entry_mapping(
    path: pathlib.Path,
    max_template_release_date: datetime,
) -> dict:
    """Load one entry mapping NPZ.

    The source idx_map is 1-based. We expose it as explicit 1-based
    entry_indices/template_indices fields to avoid ambiguity about columns.
    """
    templates = []
    with np.load(path, allow_pickle=True) as data:
        for template_id in data.files:
            info = data[template_id].item()
            if np.max(info["idx_map"]) > 65535:
                raise ValueError(
                    f"{path}:{template_id} has idx_map values exceeding 65535."
                )
            idx_map = np.asarray(info["idx_map"], dtype=np.uint16)
            if idx_map.ndim != 2 or idx_map.shape[1] != 2:
                raise ValueError(
                    f"{path}:{template_id} has invalid idx_map shape {idx_map.shape}"
                )
            if idx_map.size > 0 and idx_map.min() < 1:
                raise ValueError(
                    f"{path}:{template_id} idx_map is expected to be 1-based."
                )
            release_date = str(info["release_date"])
            if datetime.fromisoformat(release_date) > max_template_release_date:
                continue

            templates.append(
                {
                    "template_id": template_id,
                    "index": int(info["index"]),
                    "release_date": release_date,
                    "entry_indices": idx_map[:, 0],
                    "template_indices": idx_map[:, 1],
                }
            )

    templates.sort(key=lambda x: (x["index"], x["template_id"]))
    return {
        "entry_id": path.stem,
        "templates": templates,
    }


def load_manifest_entity_key_map(
    manifest_path: pathlib.Path,
) -> dict[str, tuple[str, int, datetime]]:
    """Return source chain id -> ({pdb_id}_{entity_id}, num_residues, cutoff)."""
    with open(manifest_path, "rb") as f:
        metadatas = msgpack.unpackb(f.read(), raw=False)

    chain_id_to_entity_key: dict[str, tuple[str, int, datetime]] = {}
    for metadata in metadatas:
        pdb_id = str(metadata["id"]).lower()
        entry_release_date = datetime.fromisoformat(metadata["exp"]["release_date"])
        max_template_release_date = entry_release_date - timedelta(days=60)
        for chain in metadata["chains"]:
            chain_id = str(chain["id"])
            entity_id = chain.get("entity_id")
            if entity_id is None:
                logger.warning(f"Missing entity_id for manifest chain {chain_id}.")
                continue
            entity_key = f"{pdb_id}_{int(entity_id)}"
            chain_id_to_entity_key[chain_id] = (
                entity_key,
                int(chain["num_residues"]),
                max_template_release_date,
            )
    return chain_id_to_entity_key


def load_template_lengths(data_dir: pathlib.Path) -> dict[str, int]:
    manifest_path = data_dir / "template_manifest.msgpack"
    if not manifest_path.exists():
        logger.warning(
            f"Template manifest not found: {manifest_path}; "
            "template residue index validation will be skipped."
        )
        return {}
    with open(manifest_path, "rb") as f:
        metadatas = msgpack.unpackb(f.read(), raw=False)
    return {str(metadata["id"]): int(metadata["num_residues"]) for metadata in metadatas}


def select_mapping_paths(
    metadata_paths: list[pathlib.Path],
    chain_id_to_entity_key: dict[str, tuple[str, int, datetime]],
) -> tuple[list[tuple[pathlib.Path, str, int, datetime]], int]:
    path_by_chain_id = {path.stem: path for path in metadata_paths}
    path_by_chain_id.update({path.stem.lower(): path for path in metadata_paths})

    selected: list[tuple[pathlib.Path, str, int, datetime]] = []
    seen_entity_keys: dict[str, pathlib.Path] = {}
    missing_count = 0
    for (
        chain_id,
        (entity_key, query_length, max_template_release_date),
    ) in chain_id_to_entity_key.items():
        path = (
            path_by_chain_id.get(chain_id)
            or path_by_chain_id.get(chain_id.lower())
            or path_by_chain_id.get(entity_key)
            or path_by_chain_id.get(entity_key.lower())
        )
        if path is None:
            missing_count += 1
            continue
        if entity_key in seen_entity_keys:
            logger.warning(
                f"Duplicate template mapping key {entity_key}: "
                f"using {seen_entity_keys[entity_key].name}, skipping {path.name}."
            )
            continue
        seen_entity_keys[entity_key] = path
        selected.append((path, entity_key, query_length, max_template_release_date))

    manifest_chain_ids = set(chain_id_to_entity_key)
    manifest_chain_ids_lower = {chain_id.lower() for chain_id in manifest_chain_ids}
    manifest_entity_keys = {
        entity_key for entity_key, _, _ in chain_id_to_entity_key.values()
    }
    manifest_entity_keys_lower = {
        entity_key.lower() for entity_key in manifest_entity_keys
    }
    extra_paths = [
        path
        for path in metadata_paths
        if path.stem not in manifest_chain_ids
        and path.stem.lower() not in manifest_chain_ids_lower
        and path.stem not in manifest_entity_keys
        and path.stem.lower() not in manifest_entity_keys_lower
    ]
    if extra_paths:
        examples = ", ".join(path.stem for path in extra_paths[:5])
        logger.warning(
            f"{len(extra_paths)} template metadata files are not present in "
            f"manifest.msgpack; skipping. Examples: {examples}"
        )
    return selected, missing_count


def serialize_entry_mapping(
    task: tuple[pathlib.Path, str, int, datetime],
) -> tuple[str, bytes | None]:
    path, entity_key, query_length, max_template_release_date = task
    try:
        record = load_entry_mapping(path, max_template_release_date)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        raise e
        return entity_key, None

    if len(record["templates"]) == 0:
        return entity_key, None

    return entity_key, pickle.dumps(
        record["templates"],
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def clear_lmdb(txn: lmdb.Transaction) -> None:
    keys = [key for key, _ in txn.cursor()]
    for key in keys:
        txn.delete(key)


def main():
    args = parse_args()
    data_dir = args.data_dir / "rcsb_multimer/"
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(args.metadata_dir.glob("*.npz"))
    print(f"Found {len(metadata_paths)} template metadata NPZ files.")
    manifest_path = data_dir / "manifest.msgpack"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    chain_id_to_entity_key = load_manifest_entity_key_map(manifest_path)
    mapping_tasks, missing_count = select_mapping_paths(
        metadata_paths, chain_id_to_entity_key
    )
    if missing_count > 0:
        logger.warning(
            f"{missing_count} manifest chains do not have template metadata files."
        )
    print(f"Selected {len(mapping_tasks)} entity-level template mappings.")

    lmdb_path = data_dir / "template_mapping.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=20 * 1024 * 1024 * 1024,
    )
    with env.begin(write=True) as txn:
        clear_lmdb(txn)

    txn = env.begin(write=True)
    count = 0
    with multiprocessing.Pool(args.num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(serialize_entry_mapping, mapping_tasks, chunksize=100),
            total=len(mapping_tasks),
            desc="Creating template mapping",
        )
        for entry_id, value in pbar:
            if value is None:
                continue
            txn.put(entry_id.encode(), value)
            count += 1
            if count % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()
    env.close()

    print(f"Saved template mapping LMDB to {lmdb_path}")
    print(f"Total mappings written: {count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
