# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Process Structure Data."""

from xfold.constants import atom_types
from xfold.constants import residue_names
from xfold.constants import side_chains

import torch


NUM_DENSE = atom_types.DENSE_ATOM_NUM
NUM_AA = len(residue_names.PROTEIN_TYPES)
NUM_AA_WITH_UNK_AND_GAP = len(
    residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP
)
NUM_RESTYPES_WITH_UNK_AND_GAP = (
    residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
)


def _make_restype_rigidgroup_dense_atom_idx():
    """Create Mapping from rigid_groups to dense_atom indices."""
    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms):
    # (31, 8, 3)
    base_atom_indices = torch.zeros(
        (NUM_RESTYPES_WITH_UNK_AND_GAP, 8, 3), dtype=torch.int32
    )

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(
        residue_names.PROTEIN_TYPES_ONE_LETTER
    ):
        resname = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]

        dense_atom_names = atom_types.ATOM14[resname]
        # 0: backbone frame
        base_atom_indices[restype, 0, :] = torch.tensor([
            dense_atom_names.index(atom) for atom in ['C', 'CA', 'N']
        ], dtype=torch.int32)

        # 3: 'psi-group'
        base_atom_indices[restype, 3, :] = torch.tensor([
            dense_atom_names.index(atom) for atom in ['CA', 'C', 'O']
        ], dtype=torch.int32)

        for chi_idx in range(4):
            if side_chains.CHI_ANGLES_MASK[restype][chi_idx]:
                atom_names = side_chains.CHI_ANGLES_ATOMS[resname][chi_idx]
                base_atom_indices[restype, chi_idx + 4, :] = torch.tensor([
                    dense_atom_names.index(atom) for atom in atom_names[1:]
                ], dtype=torch.int32)
    dense_atom_names = atom_types.DENSE_ATOM['A']
    nucleic_rigid_atoms = torch.tensor([
        dense_atom_names.index(atom) for atom in ["C1'", "C3'", "C4'"]
    ], dtype=torch.int32)
    for nanum, _ in enumerate(residue_names.NUCLEIC_TYPES):
        # 0: backbone frame only.
        # we have aa + unk + gap, so we want to start after those
        resnum = nanum + NUM_AA_WITH_UNK_AND_GAP
        base_atom_indices[resnum, 0, :] = nucleic_rigid_atoms

    return base_atom_indices


RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX = _make_restype_rigidgroup_dense_atom_idx()


def _make_restype_pseudobeta_idx():
    """Returns indices of residue's pseudo-beta."""
    restype_pseudobeta_index = torch.zeros(
        (NUM_RESTYPES_WITH_UNK_AND_GAP,), dtype=torch.int32
    )
    for restype, restype_letter in enumerate(
        residue_names.PROTEIN_TYPES_ONE_LETTER
    ):
        restype_name = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
        atom_names = list(atom_types.ATOM14[restype_name])
        if restype_name in {'GLY'}:
            restype_pseudobeta_index[restype] = atom_names.index('CA')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('CB')
    for nanum, resname in enumerate(residue_names.NUCLEIC_TYPES):
        atom_names = list(atom_types.DENSE_ATOM[resname])
        # 0: backbone frame only.
        # we have aa + unk , so we want to start after those
        restype = nanum + NUM_AA_WITH_UNK_AND_GAP
        if resname in {'A', 'G', 'DA', 'DG'}:
            restype_pseudobeta_index[restype] = atom_names.index('C4')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('C2')
    return restype_pseudobeta_index


RESTYPE_PSEUDOBETA_INDEX = _make_restype_pseudobeta_idx()


def _make_aatype_dense_atom_to_atom37():
    """Map from dense_atom to atom37 per residue type."""
    restype_dense_atom_to_atom37 = [
    ]  # mapping (restype, dense_atom) --> atom37
    for rt in residue_names.PROTEIN_TYPES_ONE_LETTER:
        atom_names = list(
            atom_types.ATOM14_PADDED[residue_names.PROTEIN_COMMON_ONE_TO_THREE[rt]]
        )
        atom_names.extend([''] * (NUM_DENSE - len(atom_names)))
        restype_dense_atom_to_atom37.append(
            [(atom_types.ATOM37_ORDER[name] if name else 0)
             for name in atom_names]
        )
    # Add dummy mapping for restype 'UNK', '-' (gap), and nucleics [but not DN].
    for _ in range(2 + len(residue_names.NUCLEIC_TYPES_WITH_UNKNOWN)):
        restype_dense_atom_to_atom37.append([0] * NUM_DENSE)

    restype_dense_atom_to_atom37 = torch.tensor(
        restype_dense_atom_to_atom37, dtype=torch.int32
    )
    return restype_dense_atom_to_atom37


PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37 = _make_aatype_dense_atom_to_atom37()
