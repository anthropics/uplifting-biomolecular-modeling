import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

import gemmi
import numpy as np
import zstandard as zstd
from tqdm import tqdm

from atlasfold.common import protein, residue_constants
from atlasfold.train.monomer.dataset import DataPipeline

logger = logging.getLogger(__name__)
SUCCESS = 0
FAILED = 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process synthetic PDB files.")
    parser.add_argument(
        "--pdb_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the structure files directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def read_protein_structure(
    path: str | pathlib.Path, name: str | None = None
) -> protein.Protein:
    """Load a predicted protein structure from single monomer file.

    Parameters
    ----------
    path : Path
        Path to the structure file.

    Returns
    -------
    protein.Protein
        Protein object containing sequence and coordinates.
    """

    if name is None:
        name = pathlib.Path(path).name.split(".")[0]

    filetype = pathlib.Path(path).name.split(".", 1)[-1].lower()
    assert filetype in {"pdb", "pdb.gz", "pdb.zst", "cif", "cif.gz", "cif.zst"}, (
        f"Unsupported file type: {filetype}"
    )

    structure: gemmi.Structure
    if filetype.endswith("zst"):
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                structure = gemmi.read_structure_string(reader.read())
    else:
        structure = gemmi.read_structure(str(path))

    # Get sequence
    raw_chain = structure[0].subchains()[0]
    residue_dict = {}
    for res in raw_chain:
        # Get residue index
        # NOTE: use seqid.num instead of label_seq to handle pdb format.
        res_idx = res.seqid.num
        if res_idx is None:
            logger.warning(f"Residue {res.name} in {name} missing label_seq; skipping.")
            continue

        aa = residue_constants.restype_3to1.get(res.name, "X")
        res_type = residue_constants.restype_1to3[aa]
        res_atom14_order = residue_constants.restype_atom14_order[res_type]
        res_coords = np.full((14, 3), np.nan, dtype=np.float32)
        res_b_factors = np.full((14,), np.nan, dtype=np.float32)
        for atom in res:
            atom_name: str = atom.name.upper().strip()
            if atom_name not in res_atom14_order:
                if atom_name.startswith(("OXT", "H", "D")):
                    # Ignore loging for missing OXT and hydrogens
                    continue
                logger.warning(
                    f"Atom {atom_name} in residue {res.name} {res_idx} not in "
                    f"standard atom list for residue type {res_type}; skipping."
                )
                continue
            atom_i: int = res_atom14_order[atom_name]
            res_coords[atom_i, :] = atom.pos.tolist()
            res_b_factors[atom_i] = atom.b_iso
        residue_dict[res_idx] = (aa, res_coords, res_b_factors)

    # Sort residues by index and convert to arrays
    residue_indices = sorted(residue_dict.keys())
    start_idx = min(residue_indices)
    end_idx = max(residue_indices)
    L = end_idx - start_idx + 1

    aa_list = ["X"] * L
    coordinates = np.full((L, 14, 3), np.nan, dtype=np.float32)
    b_factors = np.full((L, 14), np.nan, dtype=np.float32)
    for res_idx in residue_indices:
        aa, res_coords, res_b_factors = residue_dict[res_idx]
        i = res_idx - start_idx
        aa_list[i] = aa
        coordinates[i, :, :] = res_coords
        b_factors[i, :] = res_b_factors

    sequence = "".join(aa_list)
    return protein.Protein.create(name, sequence, coordinates, b_factors)


def worker_fn(path: pathlib.Path, output_dir: pathlib.Path):
    """Worker function to process a single file."""
    out_path = output_dir / f"{path.name.split('.')[0]}.npz"
    if out_path.exists():
        return SUCCESS
    try:
        prot = read_protein_structure(path)
        npz_path = output_dir / f"{prot.name}.npz"
        DataPipeline.save(prot, npz_path)
        return SUCCESS
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return FAILED


def main():
    """Main function to process files in parallel."""
    args = parse_args()
    pdb_dir: pathlib.Path = args.pdb_dir
    data_dir: pathlib.Path = args.data_dir / args.name / "npz/"
    data_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_failed = 0
    for subdir in (pbar := tqdm(sorted(pdb_dir.iterdir()))):
        if not subdir.is_dir():
            logger.warning(f"Skipping non-directory {subdir} in pdb_dir.")
            continue

        shard_name = subdir.name
        save_dir = data_dir / shard_name
        save_dir.mkdir(parents=True, exist_ok=True)

        pdb_paths = list(subdir.glob("*.pdb*"))
        if len(pdb_paths) == 0:
            logger.warning("No files to process. Exiting.")

        parse_pdb_partial = functools.partial(worker_fn, output_dir=save_dir)
        with multiprocessing.Pool(args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(parse_pdb_partial, pdb_paths, chunksize=20),
                    total=len(pdb_paths),
                    desc="Processing synthetic data",
                    leave=False,
                )
            )
        n_success = results.count(SUCCESS)
        n_failed = results.count(FAILED)
        total_success += n_success
        total_failed += n_failed
        pbar.set_postfix(
            {
                "subdir": shard_name,
                "success": n_success,
                "failed": n_failed,
                "total_success": total_success,
                "total_failed": total_failed,
            }
        )


if __name__ == "__main__":
    main()
