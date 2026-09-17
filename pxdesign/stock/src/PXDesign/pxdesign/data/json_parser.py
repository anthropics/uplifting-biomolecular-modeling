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

import copy
import logging

import numpy as np
from protenix.data.json_parser import (
    DNA_1to3,
    PROTEIN_1to3,
    RNA_1to3,
    _build_polymer_atom_array,
    add_reference_features,
    build_ligand,
    lig_file_to_atom_info,
    rdkit_mol_to_atom_info,
)
from protenix.data.parser import MMCIFParser
from protenix.utils.file_io import load_gzip_pickle

logger = logging.getLogger(__name__)


def remove_unresolved_residue_in_atom_array(atom_array):
    coord_mask = atom_array.is_resolved.astype(bool)
    res_ids = atom_array.res_id
    chain_ids = atom_array.chain_id
    res_chain_ids_to_mask = set(zip(res_ids[coord_mask], chain_ids[coord_mask]))
    new_mask = np.array(
        [
            not (res_id, chain_id) in res_chain_ids_to_mask
            for res_id, chain_id in zip(res_ids, chain_ids)
        ]
    )
    atom_array = atom_array[~new_mask]
    return atom_array


def build_polymer_from_sequence(entity_info: dict):
    """
    build a polymer from a polymer info dict
    example: {
        "name": "polymer",
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",
        "count": 1
        }

    Args:
        item (dict): polymer info dict

    Returns:
        dict: {"atom_array": biotite_AtomArray_object}
    """
    poly_type, info = list(entity_info.items())[0]
    if poly_type == "proteinChain":
        ccd_seqs = [PROTEIN_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["ptmPosition"] - 1
                mtype = m["ptmType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        if glycans := info.get("glycans"):
            logging.warning(f"glycans not supported: {glycans}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        map_1to3 = DNA_1to3 if poly_type == "dnaSequence" else RNA_1to3
        ccd_seqs = [map_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["basePosition"] - 1
                mtype = m["modificationType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    raise ValueError(f"unknown modification type: {mtype}")
        chain_array = _build_polymer_atom_array(ccd_seqs)

    else:
        raise ValueError(
            "polymer type must be proteinChain, dnaSequence or rnaSequence"
        )
    chain_array = add_reference_features(chain_array)
    return {"atom_array": chain_array}


def build_polymer_from_bioassombly_dict(entity_info, remove_unresolved_residue):
    poly_type, info = list(entity_info.items())[0]
    path_file = entity_info[poly_type]["path"]
    if path_file.endswith(".pkl.gz"):
        bioassembly_dict = load_gzip_pickle(path_file)
    else:
        raise ValueError(f"Unsupported structure file {path_file}!")

    mask_chain_id = entity_info[poly_type]["json_chain_id"]
    atom_array = bioassembly_dict["atom_array"]

    if remove_unresolved_residue:
        atom_array = remove_unresolved_residue_in_atom_array(atom_array)

    chain_array_mask = atom_array.chain_id == mask_chain_id
    chain_array = atom_array[chain_array_mask]

    if "hotspot" in entity_info[poly_type]:
        hotspot = entity_info[poly_type]["hotspot"]
    else:
        hotspot = []
    hotspot = np.isin(chain_array.res_id, np.array(hotspot))

    if "noise_level" in entity_info[poly_type]:
        noise = np.full(len(chain_array), float(entity_info[poly_type]["noise_level"]))
    else:
        noise = np.full(len(chain_array), 0.00)

    conditional_label = np.full(len(chain_array), 1).astype(bool)

    chain_array.set_annotation("noise_level", noise)
    chain_array.set_annotation("conditional_label", conditional_label)
    chain_array.set_annotation("hotspot", hotspot)
    chain_array.set_annotation("coord_from_cif", chain_array.coord)
    chain_array.set_annotation(
        "coord_from_cif_is_resolved", chain_array.is_resolved.astype(bool)
    )
    chain_array = add_reference_features(chain_array)

    if "crop" in entity_info[poly_type] and entity_info[poly_type]["crop"] is not None:
        crop = entity_info[poly_type]["crop"]
        crop.replace(" ", "")
        crop = crop.split(",")
        save_list = []
        for pid in crop:
            if "-" in pid:
                s, e = pid.split("-")
                length = int(e) - int(s) + 1
                save_num = [i + int(s) for i in range(0, length)]
                save_list += save_num
            else:
                save_list.append(int(pid))
        crop_mask = np.isin(chain_array.res_id, np.array(save_list))
        chain_array = chain_array[crop_mask]

    # chain_array = chain_array[chain_array.atom_name != "OXT"]

    return {"atom_array": chain_array}


def build_polymer(entity_info: dict, remove_unresolved_residue: bool = True):

    poly_type, info = list(entity_info.items())[0]
    if (
        entity_info[poly_type]["sequence_type"] == "condition"
        and info.get("path", None) is not None
    ):
        return build_polymer_from_bioassombly_dict(
            entity_info, remove_unresolved_residue
        )

    assert entity_info[poly_type]["sequence_type"] in ["design", "condition"]
    assert "sequence" in info
    chain_array = build_polymer_from_sequence(entity_info=entity_info)["atom_array"]
    # Add hotspot if exists
    if "hotspot" in entity_info[poly_type]:
        hotspot = entity_info[poly_type]["hotspot"]
    else:
        hotspot = []
    hotspot = np.isin(chain_array.res_id, np.array(hotspot))
    chain_array.set_annotation("hotspot", hotspot)

    # Add noise: currently not used
    noise = np.full(len(chain_array), 0.00)
    chain_array.set_annotation("noise_level", noise)

    # Add condition label
    if entity_info[poly_type]["sequence_type"] == "design":
        conditional_label = np.full(len(chain_array), 0).astype(bool)
    else:
        assert entity_info[poly_type]["sequence_type"] == "condition"
        conditional_label = np.full(len(chain_array), 1).astype(bool)

    chain_array.set_annotation("conditional_label", conditional_label.copy())
    res_name = chain_array.res_name.copy()
    res_name[~conditional_label] = "xpb"
    chain_array.set_annotation("res_name", res_name)

    ## coord * 0 -> not from cif file
    chain_array.set_annotation("coord_from_cif", chain_array.coord * 0.0)
    chain_array.set_annotation(
        "coord_from_cif_is_resolved", np.full(len(chain_array), 0).astype(bool)
    )

    if "is_resolved" not in chain_array._annot:
        chain_array.set_annotation(
            "is_resolved", np.ones((len(chain_array),)).astype(bool)
        )

    if "crop" in entity_info[poly_type] and entity_info[poly_type]["crop"] is not None:
        crop = entity_info[poly_type]["crop"]
        crop.replace(" ", "")
        crop = crop.split(",")
        save_list = []
        for pid in crop:
            if "-" in pid:
                s, e = pid.split("-")
                length = int(e) - int(s) + 1
                save_num = [i + int(s) for i in range(0, length)]
                save_list += save_num
            else:
                save_list.append(int(pid))
        crop_mask = np.isin(chain_array.res_id, np.array(save_list))
        chain_array = chain_array[crop_mask]
    return {"atom_array": chain_array}


def add_entity_atom_array(single_job_dict: dict) -> dict:
    """
    add atom_array to each entity in single_job_dict
    args:
        single_job_dict (dict): input job dict
    returns:
        dict: deepcopy and updated job dict with atom_array
    """
    single_job_dict = copy.deepcopy(single_job_dict)
    sequences = single_job_dict["sequences"]
    smiles_ligand_count = 0
    for entity_info in sequences:
        if info := entity_info.get("proteinChain"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("dnaSequence"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("rnaSequence"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("condition_ligand"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("ligand"):
            atom_info = build_ligand(entity_info)
            if not info["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                assert smiles_ligand_count <= 99, "too many smiles ligands"
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with CCD code
                atom_info["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        elif info := entity_info.get("ion"):
            atom_info = build_ligand(entity_info)
        else:
            raise ValueError(
                "entity type must be proteinChain, dnaSequence, rnaSequence, ligand or ion"
            )
        info.update(atom_info)
    return single_job_dict
