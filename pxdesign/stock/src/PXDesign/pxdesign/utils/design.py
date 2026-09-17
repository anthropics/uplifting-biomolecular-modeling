# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import biotite.structure as struc
import torch
from protenix.data import ccd

from pxdesign.data.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RNA_STD_RESIDUES,
    STD_RESIDUES_WITH_GAP,
)


def encoder(encode_def_list: list[str], input_list: list[str]) -> torch.Tensor:
    """
    Encode a list of input values into a binary format using a specified encoding definition list.

    Args:
        encode_def_list (list): A list of encoding definitions.
        input_list (list): A list of input values to be encoded.

    Returns:
        torch.Tensor: A tensor representing the binary encoding of the input values.
    """
    onehot_dict = {}
    num_keys = len(encode_def_list)
    for index, key in enumerate(encode_def_list):
        onehot = [0] * num_keys
        onehot[index] = 1
        onehot_dict[key] = onehot

    onehot_encoded_data = [onehot_dict[item] for item in input_list]
    onehot_tensor = torch.Tensor(onehot_encoded_data)
    return onehot_tensor


def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
    return encoder(list(STD_RESIDUES_WITH_GAP.keys()), restype_list)


def cano_seq_resname_with_mask(atom_array):
    """
    Assign to each atom the three-letter residue name (resname)
    corresponding to its place in the canonical sequences.
    Non-standard residues are mapped to standard ones.
    Residues that cannot be mapped to standard residues and ligands are all labeled as "UNK".

    Note: Some CCD Codes in the canonical sequence are mapped to three letters. It is labeled as one "UNK".

    Args:
        atom_array (AtomArray): Biotite AtomArray object

    Returns:
        AtomArray: Biotite AtomArray object with "cano_seq_resname" annotation added.
    """
    cano_seq_resname = []
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)

    for start, stop in zip(starts[:-1], starts[1:]):
        res_atom_nums = stop - start
        mol_type = atom_array.mol_type[start]
        resname = atom_array.res_name[start]

        if resname == "xpb":
            one_letter_code = "j"
        else:
            one_letter_code = ccd.get_one_letter_code(resname)

        if one_letter_code is None or len(one_letter_code) != 1:
            # Some non-standard residues cannot be mapped back to one standard residue.
            one_letter_code = "X" if mol_type == "protein" else "N"

        if mol_type == "protein":
            res_name_in_cano_seq = PROT_STD_RESIDUES_ONE_TO_THREE.get(
                one_letter_code, "UNK"
            )
        elif mol_type == "dna":
            res_name_in_cano_seq = "D" + one_letter_code
            if res_name_in_cano_seq not in DNA_STD_RESIDUES:
                res_name_in_cano_seq = "DN"
        elif mol_type == "rna":
            res_name_in_cano_seq = one_letter_code
            if res_name_in_cano_seq not in RNA_STD_RESIDUES:
                res_name_in_cano_seq = "N"
        else:
            # some molecules attached to a polymer like ATP-RNA. e.g.
            res_name_in_cano_seq = "UNK"

        cano_seq_resname.extend([res_name_in_cano_seq] * res_atom_nums)

    return cano_seq_resname


def make_random_mask(ref_tensor, mask_ratio=None):
    if mask_ratio == None:
        mask_ratio = random.random()
    mask = torch.zeros_like(ref_tensor)
    random_tensor = torch.rand_like(ref_tensor.to(torch.float32))
    mask[random_tensor >= mask_ratio] = 1
    return mask.to(torch.int64)
