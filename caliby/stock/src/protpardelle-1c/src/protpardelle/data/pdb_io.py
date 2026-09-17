"""PDB file input/output.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

from typing import Any

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser
from einops import rearrange

from protpardelle.common import residue_constants
from protpardelle.common.protein import (
    PDB_CHAIN_IDS,
    PDB_MAX_CHAINS,
    Hetero,
    Protein,
    to_pdb,
)
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.utils import StrPath, get_logger, norm_path, tensor_to_ndarray

logger = get_logger(__name__)


def add_chain_gap(
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    chain_residx_gap: int = 200,
) -> torch.Tensor:
    """Add a residue index gap between chains.

    Iteratively shift residue index per chain, starting from last residue index from previous chain.
    E.g. if chain A has 5 residues and chain B has 3 residues,
    with chain_residx_gap = 200, the residue indices for chain B will be 206, 207, 208.

    Args:
        residue_index (torch.Tensor): The residue index tensor. (L,)
        chain_index (torch.Tensor): The chain index tensor. (L,)
        chain_residx_gap (int, optional): The gap to add between chains. Defaults to 200.

    Returns:
        torch.Tensor: The modified residue index tensor.
    """

    if torch.sum(chain_index) == 0:
        return residue_index

    for ci in range(1, int(torch.max(chain_index).item()) + 1):
        curr_chain_pos = torch.nonzero(chain_index == ci).flatten()
        prev_chain_residue_idx_end = residue_index[curr_chain_pos[0] - 1]
        curr_chain_residue_idx_start = residue_index[curr_chain_pos[0]]
        target_start = prev_chain_residue_idx_end + chain_residx_gap
        residx_delta = target_start - curr_chain_residue_idx_start
        # special case when chain_residx_gap is zero: keep original indices for each chain, reset to 1
        if chain_residx_gap == 0:
            residue_index[curr_chain_pos] = (
                residue_index[curr_chain_pos] - residue_index[curr_chain_pos[0]] + 1
            )
        else:
            residue_index[curr_chain_pos] = residue_index[curr_chain_pos] + residx_delta

    return residue_index


def read_pdb(
    pdb_path: StrPath, chain_id: str | None = None
) -> tuple[Protein, Hetero, dict[str, int]]:
    """Takes a PDB string and constructs a Protein object.

    Args:
        pdb_file (StrPath): The path to the PDB file.
        chain_id (str | None, optional): If chain_id is specified (e.g. A), then only
        that chain is parsed. Otherwise all chains are parsed. Defaults to None.

    Returns:
        tuple[Protein, Hetero, dict[str, int]]: A tuple containing the parsed Protein,
        Hetero, and a mapping of chain IDs to their indices.
    """

    pdb_path = norm_path(pdb_path)

    if pdb_path.suffix == ".cif":
        parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=False)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("protein", pdb_path)
    num_models = len(structure)
    if num_models > 1:
        logger.warning(
            "PDB file %s has %d models, only the first one will be used.",
            pdb_path,
            num_models,
        )
    if num_models == 0:
        logger.error("PDB file %s has no models.", pdb_path)
    model = next(structure.get_models())

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    hetero_atom_positions = (
        []
    )  # list of length len(ncaa) storing variable number of atom coordinates per array
    hetero_aatype = []  # list of aatypes (three letter)
    hetero_atom_types = []  # list of list of atom types per ncaa
    hetero_motif_mask = []  # indices of hetero_atom_positions that are motif positions
    hetero_not_motif_mask = (
        []
    )  # indices of hetero_atom_positions that are non-motif but ligand/metal positions (for clash loss)

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for ri, res in enumerate(chain):
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            # Or 'UNK' reserved for redesignable motif positions
            if res_shortname != "X" or res.resname == "UNK":  # canonical amino acids
                if res.resname == "UNK":
                    res_shortname = "G"
                restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num
                )
                pos = np.zeros((residue_constants.atom_type_num, 3))
                mask = np.zeros((residue_constants.atom_type_num,))
                res_b_factors = np.zeros((residue_constants.atom_type_num,))
                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    pos[residue_constants.atom_order[atom.name]] = atom.coord
                    mask[residue_constants.atom_order[atom.name]] = 1.0
                    res_b_factors[residue_constants.atom_order[atom.name]] = (
                        atom.bfactor
                    )
                if np.sum(mask) < 0.5:
                    # If no known atom positions are reported for the residue then skip it.
                    continue
                aatype.append(restype_idx)
                atom_positions.append(pos)
                atom_mask.append(mask)
                residue_index.append(res.id[1])
                chain_ids.append(chain.id)
                b_factors.append(res_b_factors)
            else:
                # If residue has amino acid backbone atoms, treat it as a noncanonical motif residue
                # Otherwise, treat as a ligand/metal for clash loss
                resemble_atoms = residue_constants.backbone_atoms.copy()
                for atom in res:
                    if atom.name in resemble_atoms:
                        resemble_atoms.remove(atom.name)
                if len(resemble_atoms) == 0:
                    hetero_motif_mask.append(ri)
                else:
                    hetero_not_motif_mask.append(ri)
                hetero_aatype.append(res.get_resname())

                pos = []
                h_atom_types = []
                for atom in res:
                    pos.append(atom.coord)
                    h_atom_types.append(atom.name)
                hetero_atom_positions.append(pos)
                hetero_atom_types.append(h_atom_types)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    prot = Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )
    het = Hetero(
        hetero_atom_positions=hetero_atom_positions,
        hetero_aatype=hetero_aatype,
        hetero_atom_types=hetero_atom_types,
        hetero_motif_mask=hetero_motif_mask,
        hetero_not_motif_mask=hetero_not_motif_mask,
    )

    return prot, het, chain_id_mapping


def load_feats_from_pdb(
    pdb_path: StrPath,
    chain_id: str | None = None,
    chain_residx_gap: int = 200,
    include_pos_feats: bool = False,
) -> tuple[dict[str, Any], Hetero]:
    """Load model input features from a PDB file or mmcif file.

    Args:
        pdb_path (StrPath): The path to PDB or mmcif file.
        chain_id (str | None, optional): Chain ID to load. Defaults to None.
        chain_residx_gap (int, optional): Residue index gap for chain breaks for PDBs
            with multiple chains. Defaults to 200.
        include_pos_feats (bool, optional): If True, include chain_id_mapping and residue_index_orig
            in feats for specifying specific positions. Defaults to False.

    Returns:
        tuple[dict[str, Any], Hetero]: A tuple of (feats, hetero) where feats is
            a dictionary of model input features and hetero is a Hetero object.
    """

    protein_obj, hetero_obj, chain_id_mapping = read_pdb(pdb_path, chain_id=chain_id)

    feats = {}
    feats["bb_coords"] = torch.as_tensor(
        protein_obj.atom_positions[:, residue_constants.backbone_idxs],
        dtype=torch.float,
    )
    for k, v in vars(protein_obj).items():
        feats[k] = torch.as_tensor(v, dtype=torch.float)
    feats["aatype"] = feats["aatype"].long()

    # For users to specify conditioning: keep track of original residx and mapping of chain ID to chain index
    if include_pos_feats:
        feats["residue_index_orig"] = feats["residue_index"].clone()
        feats["chain_id_mapping"] = chain_id_mapping

    # Always start first residx from 1 (shouldn't matter with relative pos encoding though)
    feats["residue_index"] = (
        feats["residue_index"] - torch.min(feats["residue_index"]) + 1
    )

    # Handle residue index for PDBs with multiple chains
    feats["residue_index"] = add_chain_gap(
        feats["residue_index"], feats["chain_index"], chain_residx_gap=chain_residx_gap
    )

    return feats, hetero_obj


def feats_to_pdb_str(
    atom_coords: torch.Tensor,
    aatype: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    chain_index: torch.Tensor | None = None,
    b_factors: torch.Tensor | None = None,
    chain_id_mapping: dict[str, int] | None = None,
    atom_lines_only: bool = True,
) -> str:
    """Convert features to PDB string.

    Args:
        atom_coords (torch.Tensor): Atom coordinates. (L, 37, 3)
        aatype (torch.Tensor | None, optional): Amino acid types. Defaults to None. (L,)
        atom_mask (torch.Tensor | None, optional): Atom mask. Defaults to None. (L, 37)
        residue_index (torch.Tensor | None, optional): Residue index. Defaults to None. (L,)
        chain_index (torch.Tensor | None, optional): Chain index. Defaults to None. (L,)
        b_factors (torch.Tensor | None, optional): B-factors. Defaults to None. (L, 37)
        chain_id_mapping (dict[str, int] | None, optional): Chain ID mapping. Defaults to None.
        atom_lines_only (bool, optional): If True, only include atom lines. Defaults to True.

    Raises:
        ValueError: If both atom_mask and aatype are None.

    Returns:
        str: PDB string representation of the features.
    """

    # Expects unbatched, cropped inputs. needs at least one of atom_mask, aatype
    # Uses all-GLY aatype if aatype not given: does not infer from atom_mask
    if (atom_mask is None) and (aatype is None):
        raise ValueError("At least one of atom_mask or aatype must be provided.")
    if atom_mask is None:
        assert aatype is not None
        atom_mask = atom37_mask_from_aatype(aatype)
    if aatype is None:
        assert atom_mask is not None
        seq_mask = atom_mask[:, 1]  # use CA atom to indicate residue presence
        aatype = seq_mask * residue_constants.restype_order["G"]

    if residue_index is None:
        residue_index = torch.arange(aatype.shape[-1]) + 1  # start residue index from 1
    if chain_index is None:
        chain_index = torch.ones_like(aatype)
    if b_factors is None:
        b_factors = torch.ones_like(atom_mask)

    prot = Protein(
        atom_positions=tensor_to_ndarray(atom_coords),
        atom_mask=tensor_to_ndarray(atom_mask),
        aatype=tensor_to_ndarray(aatype),
        residue_index=tensor_to_ndarray(residue_index),
        chain_index=tensor_to_ndarray(chain_index),
        b_factors=tensor_to_ndarray(b_factors),
    )
    pdb_str = to_pdb(prot, chain_id_mapping=chain_id_mapping)

    if atom_lines_only:
        pdb_lines = pdb_str.split("\n")
        atom_lines = [
            line
            for line in pdb_lines
            if len(line.split()) > 1 and line.split()[0] == "ATOM"
        ]
        pdb_str = "\n".join(atom_lines) + "\n"

    return pdb_str


def _bb_pdb_line(
    atomnum: int,
    atom: str,
    res: str,
    chain_idx: int,
    resnum: int,
    coords: torch.Tensor,
    elem: str,
    chain_id_mapping: dict[str, int] | None = None,
) -> str:
    """Format a single ATOM line for a PDB file."""

    if chain_id_mapping is None:
        if chain_idx < PDB_MAX_CHAINS:
            chain = PDB_CHAIN_IDS[chain_idx]
        else:
            raise ValueError(
                f"chain_idx {chain_idx} exceeds max PDB chains {PDB_MAX_CHAINS}."
            )
    else:
        id_chain_mapping = {v: k for k, v in chain_id_mapping.items()}
        chain = id_chain_mapping.get(chain_idx, "A")

    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
    occupancy = 1.0
    temp_factor = 20.0

    return (
        f"{'ATOM':<6}{atomnum:>5} {atom:^4} {res:<3} {chain:>1}{resnum:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
        f"{elem:>2}\n"
    )


def bb_coords_to_pdb_str(
    bb_coords: torch.Tensor,
    atoms: tuple[str, ...] = ("N", "CA", "C", "O"),
    aatype: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    chain_index: torch.Tensor | None = None,
    chain_id_mapping: dict[str, int] | None = None,
) -> str:
    """Convert backbone coordinates to PDB string.

    Args:
        bb_coords (torch.Tensor): Backbone coordinates of shape (L, 4, 3).
        atoms (tuple[str, ...], optional): Atom names to include. Defaults to ("N", "CA", "C", "O").
        aatype (torch.Tensor | None, optional): Amino acid types; if not specified, default to GLY. Defaults to None. (L,)
        residue_index (torch.Tensor | None, optional): Residue indices. Defaults to None. (L,)
        chain_index (torch.Tensor | None, optional): 0-indexed chain index for each residue, starting from A. Defaults to None. (L,)
        chain_id_mapping (dict[str, int] | None, optional): Chain ID mapping. Defaults to None.

    Returns:
        str: PDB string representation of the backbone coordinates.
    """

    L = bb_coords.shape[0]
    num_atoms = len(atoms)
    pdb_str_list: list[str] = []
    res_counter: dict[int, int] = {}
    chain_idx = 0
    for i in range(0, L, num_atoms):
        for j, atom in enumerate(atoms):
            residx = i // num_atoms  # 0-indexed residue index

            # Handle aatype
            if aatype is not None:
                # Save with aatype if specified
                restype = residue_constants.restypes[aatype[residx].item()]  # type: ignore
                res = residue_constants.restype_1to3[restype]
            else:
                # Otherwise, default to GLY
                res = "GLY"

            # Handle chain index
            chain_idx = (
                chain_index[residx].long().item() if chain_index is not None else 0
            )
            assert isinstance(chain_idx, int)
            if chain_idx not in res_counter:
                res_counter[chain_idx] = 1

            if residue_index is not None:
                resnum = residue_index[residx].long().item()
                assert isinstance(resnum, int)
            else:
                resnum = res_counter[chain_idx]

            pdb_line = _bb_pdb_line(
                atomnum=i + j + 1,
                atom=atom,
                res=res,
                chain_idx=chain_idx,
                resnum=resnum,
                coords=bb_coords[i + j],
                elem=atom[0],
                chain_id_mapping=chain_id_mapping,
            )
            pdb_str_list.append(pdb_line)

        res_counter[chain_idx] += 1

    return "".join(pdb_str_list)


def write_coords_to_pdb(
    atom_coords: torch.Tensor,
    output_path: StrPath,
    chain_id_mapping: dict[str, int] | None = None,
    **all_atom_feats: torch.Tensor,
) -> None:
    """Write atomic coordinates to a PDB file.

    Args:
        atom_coords (torch.Tensor): Atomic coordinates. (L, A, 3).
        output_path (StrPath): Path to the output PDB file.
        chain_id_mapping (dict[str, int] | None, optional): Mapping of chain IDs to indices. Defaults to None.
        **all_atom_feats (torch.Tensor): Additional features such as aatype, atom_mask, residue_index, chain_index, b_factors.

    Raises:
        ValueError: If the input tensor has an invalid shape.
    """

    L, A, _ = atom_coords.shape

    if A <= 4:
        coords_flat = rearrange(atom_coords, "l a c -> (l a) c")
        if A == 1:
            atoms = ("CA",)
        elif A == 3:
            atoms = ("N", "CA", "C")
        elif A == 4:
            atoms = ("N", "CA", "C", "O")
        else:
            raise ValueError("Invalid number of atoms for backbone/CA PDB.")
        feats = {k: v[:L] for k, v in all_atom_feats.items()}
        pdb_str = bb_coords_to_pdb_str(
            coords_flat,
            atoms,
            aatype=feats.get("aatype"),
            residue_index=feats.get("residue_index"),
            chain_index=feats.get("chain_index"),
            chain_id_mapping=chain_id_mapping,
        )
    else:
        feats = {k: v[:L] for k, v in all_atom_feats.items()}
        pdb_str = feats_to_pdb_str(
            atom_coords,
            aatype=feats.get("aatype"),
            atom_mask=feats.get("atom_mask"),
            residue_index=feats.get("residue_index"),
            chain_index=feats.get("chain_index"),
            b_factors=feats.get("b_factors"),
            chain_id_mapping=chain_id_mapping,
            atom_lines_only=True,
        )

    output_path = norm_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pdb_str)
