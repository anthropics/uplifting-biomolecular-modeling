"""Convert raw template chain NPZ files to AtlasFold atom14 NPZ files."""

import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

import numpy as np
from tqdm import tqdm

from atlasfold.common import protein, residue_constants
from atlasfold.train.monomer.dataset import DataPipeline as MonomerDataPipeline

SUCCESS = 0
FAILED = 1
SKIPPED_NON_PROTEIN = 2
SKIPPED_EMPTY = 3

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process template NPZ files.")
    parser.add_argument(
        "--template_dir",
        type=pathlib.Path,
        required=True,
        help="Path to raw template source directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root. Outputs are written under rcsb_multimer/templates.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def iter_template_paths(template_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in template_dir.rglob("*.npz")
        if path.name != "chain_id_to_moltype.npz"
    )


def load_template_as_protein(path: pathlib.Path) -> protein.Protein | None:
    """Load one raw template chain NPZ and convert it to atom14 Protein."""
    with np.load(path, allow_pickle=True) as data:
        molecule_type_id = data["molecule_type_id"]
        hetero = data["hetero"]
        protein_mask = (molecule_type_id == 0) & (~hetero)
        if not np.any(protein_mask):
            logger.warning(f"Template {path} contains no protein chains; skipping.")
            return None

        coord = data["coord"][protein_mask]
        chain_id = data["chain_id"][protein_mask]
        res_id = data["res_id"][protein_mask]
        res_name = data["res_name"][protein_mask]
        atom_name = data["atom_name"][protein_mask]

    residue_keys: list[tuple[str, int, str]] = []
    residue_to_index: dict[tuple[str, int, str], int] = {}
    for c_id, r_id, r_name in zip(chain_id, res_id, res_name, strict=True):
        key = (str(c_id), int(r_id), str(r_name))
        if key not in residue_to_index:
            residue_to_index[key] = len(residue_keys)
            residue_keys.append(key)

    if len(residue_keys) == 0:
        return None

    full_sequence = [r_name for _, _, r_name in residue_keys]
    sequence = residue_constants.convert_full_sequence_to_sequence(
        full_sequence, standardize_ambiguous=True
    )
    if len(sequence) < 10:
        logger.warning(f"Template {path} has sequence length < 10; skipping.")
        return None

    prot = protein.Protein.get_empty(name=path.stem, sequence=sequence)
    for c_id, r_id, r_name, a_name, xyz in zip(
        chain_id, res_id, res_name, atom_name, coord, strict=True
    ):
        residue_key = (str(c_id), int(r_id), str(r_name))
        residue_i = residue_to_index[residue_key]
        aa = sequence[residue_i]
        restype = residue_constants.restype_1to3.get(aa, "UNK")
        atom14_order = residue_constants.restype_atom14_order[restype]
        atom = str(a_name).upper().strip()
        if atom not in atom14_order:
            continue
        atom_i = atom14_order[atom]
        prot.coordinates[residue_i, atom_i, :] = xyz
        prot.b_factors[residue_i, atom_i] = 0.0

    return prot


def worker_fn(path: pathlib.Path, out_root: pathlib.Path) -> tuple[int, str]:
    template_id = path.stem
    out_dir = out_root / path.parent.name
    out_path = out_dir / f"{template_id}.npz"
    if out_path.exists():
        return SUCCESS, template_id
    try:
        prot = load_template_as_protein(path)
        if prot is None:
            return SKIPPED_NON_PROTEIN, template_id
        if len(prot) == 0:
            return SKIPPED_EMPTY, template_id
        out_dir.mkdir(parents=True, exist_ok=True)
        MonomerDataPipeline.save(prot, out_path)
        return SUCCESS, template_id
    except Exception as e:
        logger.error(f"Failed to process template {path}: {e}")
        return FAILED, template_id


def main():
    args = parse_args()
    template_dir = args.template_dir
    data_dir = args.data_dir / "rcsb_multimer/"
    out_root = data_dir / "templates" / "npz"
    out_root.mkdir(parents=True, exist_ok=True)

    template_paths = iter_template_paths(template_dir)
    print(f"Found {len(template_paths)} raw template NPZ files.")

    worker = functools.partial(worker_fn, out_root=out_root)
    flags: list[int] = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for flag, _ in tqdm(
            pool.imap_unordered(worker, template_paths, chunksize=100),
            total=len(template_paths),
            desc="Processing templates",
        ):
            flags.append(flag)

    print("Template processing completed.")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Failed to process: {flags.count(FAILED)}")
    print(f"  Skipped non-protein: {flags.count(SKIPPED_NON_PROTEIN)}")
    print(f"  Skipped empty: {flags.count(SKIPPED_EMPTY)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
