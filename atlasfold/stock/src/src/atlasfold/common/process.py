import pathlib

import gemmi
import numpy as np

from atlasfold.common import protein, residue_constants


def read_pdb(
    path: str | pathlib.Path,
    chain_id: int | str | None = None,
    name: str | None = None,
) -> protein.Protein:
    """Load a predicted protein structure from single monomer file.

    Parameters
    ----------
    path : Path
        Path to the structure file.
    chain_id : int | str, optional
        Chain ID to extract (default: first chain).
    name : str, optional
        Name of the protein (default: derived from file name).

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
    structure: gemmi.Structure = gemmi.read_structure(str(path))
    model: gemmi.Model = structure[0]

    # Get chain
    if chain_id is not None:
        raw_chain = model[chain_id]
    else:
        raw_chain = model[0]

    # Get sequence
    aa_list, coords_list, biso_list, res_idx_list = [], [], [], []
    for res in raw_chain:
        aa1 = residue_constants.restype_3to1.get(res.name, "X")
        aa3 = residue_constants.restype_1to3[aa1]
        res_atom14_order = residue_constants.restype_atom14_order[aa3]
        res_coords = np.full((14, 3), np.nan, dtype=np.float32)
        res_b_factors = np.full((14,), np.nan, dtype=np.float32)
        for atom in res:
            atom_name: str = atom.name.upper().strip()
            if atom_name in res_atom14_order:
                atom_i: int = res_atom14_order[atom_name]
                res_coords[atom_i, :] = atom.pos.tolist()
                res_b_factors[atom_i] = atom.b_iso
        aa_list.append(aa1)
        coords_list.append(res_coords)
        biso_list.append(res_b_factors)
        res_idx_list.append(res.seqid.num)

    sequence = "".join(aa_list)
    coordinates = np.stack(coords_list, axis=0)  # [L, 14, 3]
    b_factors = np.stack(biso_list, axis=0)  # [L, 14]
    residue_index = np.array(res_idx_list, dtype=np.int32)
    return protein.Protein.create(name, sequence, coordinates, b_factors, residue_index)
