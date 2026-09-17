"""Clean structure files for use with downstream Caliby workflows."""

import tempfile
from pathlib import Path

import atomworks.enums as aw_enums
import hydra
import numpy as np
from atomworks.io.parser import get_structure, read_any
from atomworks.io.utils.io_utils import to_cif_string
from biotite.structure.io.pdb import PDBFile
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from caliby.data import const
from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import get_sd_example


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks", config_name="clean_pdbs", version_base="1.3.2"
)
def main(cfg: DictConfig):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    use_parallel = cfg.num_workers > 1

    # Load in PDB files.
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Clean the PDB files.
    if use_parallel:
        parallel = Parallel(n_jobs=cfg.num_workers)
        jobs = [delayed(clean_pdb)(pdb_path, cfg.out_dir) for pdb_path in pdb_files]
        list(parallel(tqdm(jobs, total=len(jobs), desc="Cleaning RCSB mmCIF files")))
    else:
        for pdb_path in tqdm(pdb_files, total=len(pdb_files), desc="Cleaning RCSB mmCIF files"):
            clean_pdb(pdb_path, cfg.out_dir)


def _fix_blank_chain_ids(pdb_path: str) -> str:
    """If a PDB has blank chain IDs, load with biotite, assign 'A', and write to a temp PDB. Returns the path to use."""
    file_obj = read_any(pdb_path)
    atom_array = get_structure(file_obj, model=1)
    atom_array.chain_id[np.array([c.strip() == "" for c in atom_array.chain_id])] = "A"
    pdb_out = PDBFile()
    pdb_out.set_structure(atom_array)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    pdb_out.write(tmp)
    tmp.close()
    return tmp.name


def clean_pdb(pdb_path: str, out_dir: str) -> str:
    """Clean one PDB/CIF file and write a sanitized mmCIF copy.

    Args:
        pdb_path: Input PDB or CIF path.
        out_dir: Output directory for the cleaned file.

    Returns:
        Path to the cleaned mmCIF file.
    """
    # Try the normal path first; if the PDB has blank chain IDs, fix and retry.
    fixed_path = None
    try:
        example = get_sd_example(pdb_path, data_cfg=None)
    except ValueError as e:
        if "Chain identifier is empty" not in str(e):
            raise
        fixed_path = _fix_blank_chain_ids(pdb_path)
        example = get_sd_example(fixed_path, data_cfg=None)
        Path(fixed_path).unlink()
    atom_array = example["atom_array"]

    # Remove any unresolved atoms.
    atom_array = atom_array[atom_array.occupancy > 0]

    # Keep only protein chains.
    is_protein = np.isin(atom_array.chain_type, aw_enums.ChainTypeInfo.PROTEINS)
    atom_array = atom_array[is_protein]

    # Keep only resnames supported by Protpardelle-1c.
    keep_mask = np.isin(atom_array.res_name, const.PROTPARDELLE_SUPPORTED_RESNAMES)
    atom_array = atom_array[keep_mask]

    # Map each unique pair of (chain_id, transformation_id) to a sequential label 'A', 'B', ..., 'Z', 'AA', ...
    unique_pairs = []
    for c, t in zip(atom_array.chain_id, atom_array.transformation_id):
        key = (str(c), str(t))
        if key not in unique_pairs:
            unique_pairs.append(key)

    pair_to_label = {pair: _pair_index_to_label(i) for i, pair in enumerate(unique_pairs)}
    new_chain_ids = [pair_to_label[(str(c), str(t))] for c, t in zip(atom_array.chain_id, atom_array.transformation_id)]
    atom_array.chain_id = new_chain_ids

    # Write the PDB file.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = f"{out_dir}/{Path(pdb_path).stem}.cif"
    with open(out_file, "w") as f:
        f.write(to_cif_string(atom_array, include_nan_coords=False))
    return out_file


def _pair_index_to_label(idx: int) -> str:
    """
    Convert a zero-based index to chain label:
      0 -> 'A', 1 -> 'B', ..., 25 -> 'Z',
      26 -> 'AA', 27 -> 'AB', ...
    This is the usual "excel-style" base-26 without a zero digit.
    """
    if idx < 0:
        raise ValueError("idx must be >= 0")
    letters = []
    n = idx + 1  # convert to 1-based for the math
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


if __name__ == "__main__":
    main()
