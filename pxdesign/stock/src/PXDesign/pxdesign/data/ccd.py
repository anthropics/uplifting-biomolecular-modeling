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

import functools
import logging
from pathlib import Path
from typing import Union

from protenix.data.ccd import (
    _connect_inter_residue,
    add_inter_residue_bonds,
    biotite_load_ccd_cif,
    get_ccd_ref_info,
    get_component_atom_array,
    get_component_rdkit_mol,
    res_names_to_sequence,
)
from protenix.data.substructure_perms import get_substructure_perms

from pxdesign.configs.configs_data import data_configs
from pxdesign.data.constants import MOL_TYPE_MAP, STD_DESIGN

logger = logging.getLogger(__name__)

COMPONENTS_FILE = data_configs["ccd_components_file"]
RKDIT_MOL_PKL = Path(data_configs["ccd_components_rdkit_mol_file"])


@functools.lru_cache(maxsize=None)
def get_one_letter_code(ccd_code: str) -> Union[str, None]:
    """get one_letter_code from CCD components file.

    normal return is one letter: ALA --> A, DT --> T
    unknown protein: X
    unknown DNA or RNA: N
    other unknown: None
    some ccd_code will return more than one letter:
    eg: XXY --> THG

    Args:
        ccd_code (str): _description_

    Returns:
        str: one letter code
    """
    ccd_cif = biotite_load_ccd_cif()
    design_dic = {"xpb": "j"}
    if ccd_code not in ccd_cif:
        return None
    if ccd_code in design_dic.keys():
        return design_dic[ccd_code]
    one = ccd_cif[ccd_code]["chem_comp"]["one_letter_code"].as_item()
    if one == "?":
        return None
    else:
        return one


@functools.lru_cache(maxsize=None)
def get_mol_type(ccd_code: str) -> str:
    """get mol_type from CCD components file.

    based on _chem_comp.type
    http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html

    not use _chem_comp.pdbx_type, because it is not consistent with _chem_comp.type
    e.g. ccd 000 --> _chem_comp.type="NON-POLYMER" _chem_comp.pdbx_type="ATOMP"
    https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_struct_asym.pdbx_type.html

    Args:
        ccd_code (str): ccd code

    Returns:
        str: mol_type, one of {"protein", "rna", "dna", "ligand"}
    """
    ccd_cif = biotite_load_ccd_cif()

    if ccd_code in STD_DESIGN:
        return MOL_TYPE_MAP[ccd_code]

    if ccd_code not in ccd_cif:
        return "ligand"

    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        return "protein"
    if "DNA" in link_type:
        return "dna"
    if "RNA" in link_type:
        return "rna"
    return "ligand"
