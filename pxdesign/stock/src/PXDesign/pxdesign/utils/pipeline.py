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

# -*- coding: utf-8 -*-
"""
Utility helpers & constants for the design pipeline.
"""

import ast
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import get_residues
from protenix.data.constants import PROT_STD_RESIDUES_ONE_TO_THREE
from protenix.data.parser import MMCIFParser

from pxdesign.data.utils import CIFWriter, pdb_to_cif

# -------- Global subdir constants (used everywhere) --------
ORIG_SUBDIR = os.path.join("orig_designed")
AF2_SUBDIR = os.path.join("passing-AF2-IG-easy")
PTX_SUBDIR = os.path.join("passing-Protenix-basic")


# -------- Small general helpers --------


def convert_strlist_col(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stringified lists like '[0.3, 0.6]' to their numeric mean."""

    def _safe_mean(v):
        if isinstance(v, list):
            return float(np.mean(v)) if len(v) > 0 else np.nan
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return float(np.mean(parsed))
            except Exception:
                pass
        return v

    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(_safe_mean)
    return out


# -------- File post-processing --------


def add_terms_to_cif(input_cif: str, output_cif: str) -> None:
    """Insert license/disclaimer blocks into a CIF file."""
    terms1 = [
        "# By using this file you agree to the legally binding terms of use found at https://protenix-server.com/terms-of-service\n",
        "# Version: 0.1\n",
    ]
    terms2 = [
        "#\n",
        "loop_\n",
        "_audit_author.name\n",
        "_audit_author.pdbx_ordinal\n",
        '"ByteDance Seed" 1\n',
        "#\n",
        "loop_\n",
        "_pdbx_data_usage.details\n",
        "_pdbx_data_usage.id\n",
        "_pdbx_data_usage.type\n",
        "_pdbx_data_usage.url\n",
        ";NON-COMMERCIAL USE ONLY, BY USING THIS FILE YOU AGREE TO THE TERMS OF USE FOUND\n",
        "AT https://protenix-server.com/terms-of-service.\n",
        ";\n",
        "1 license    ?\n",
        ";THE INFORMATION IS NOT INTENDED FOR, HAS NOT BEEN VALIDATED FOR, AND IS NOT\n",
        "APPROVED FOR CLINICAL USE. IT SHOULD NOT BE USED FOR CLINICAL PURPOSE OR RELIED\n",
        "ON FOR MEDICAL OR OTHER PROFESSIONAL ADVICE. IT IS THEORETICAL MODELLING ONLY\n",
        'AND CAUTION SHOULD BE EXERCISED IN ITS USE. IT IS PROVIDED "AS-IS" WITHOUT ANY\n',
        "WARRANTY OF ANY KIND, WHETHER EXPRESSED OR IMPLIED. NO WARRANTY IS GIVEN THAT\n",
        "USE OF THE INFORMATION SHALL NOT INFRINGE THE RIGHTS OF ANY THIRD PARTY.\n",
        ";\n",
        "2 disclaimer ?\n",
    ]
    with open(input_cif, "r") as f:
        lines = f.readlines()
    out, inserted = [], False
    out.extend(terms1)
    for line in lines:
        out.append(line)
        if not inserted and line.strip().startswith("_entry.id"):
            out.extend(terms2)
            inserted = True
    with open(output_cif, "w") as f:
        f.writelines(out)


def replace_last_xpb_chain_sequence(
    input_cif_path: str, out_cif_path: str, new_sequence: str
) -> str:
    """Replace the sequence on the LAST chain where all residues are 'xpb'."""
    parser = MMCIFParser(input_cif_path)
    atom_array = parser.get_structure(
        altloc="first", model=1, bond_lenth_threshold=None
    )

    uniq_chain_ids, first_idx = np.unique(atom_array.chain_id, return_index=True)
    ordered_chain_ids = uniq_chain_ids[np.argsort(first_idx)]

    candidates = []
    for ch in ordered_chain_ids:
        ch_mask = atom_array.chain_id == ch
        if np.any(ch_mask) and np.all(atom_array.res_name[ch_mask] == "xpb"):
            candidates.append(ch)
    if not candidates:
        raise ValueError("No chain found where all residues are 'xpb'.")

    target_chain = candidates[-1]
    ch_mask = atom_array.chain_id == target_chain
    res_ids, _ = get_residues(atom_array[ch_mask])
    if len(res_ids) != len(new_sequence):
        raise ValueError(
            f"Length mismatch on chain {target_chain}: structure={len(res_ids)} vs seq={len(new_sequence)}"
        )

    for resid, one in zip(res_ids, new_sequence):
        if one not in PROT_STD_RESIDUES_ONE_TO_THREE:
            raise ValueError(f"Invalid amino acid letter '{one}' in new_sequence.")
        res3 = PROT_STD_RESIDUES_ONE_TO_THREE[one]
        atom_array.res_name[
            (atom_array.chain_id == target_chain) & (atom_array.res_id == resid)
        ] = res3

    writer = CIFWriter(atom_array=atom_array, entity_poly_type=parser.entity_poly_type)
    writer.save_to_cif(out_cif_path, include_bonds=True)
    return out_cif_path


# -------- Structure writers (project-specific) --------


def save_design_cif(
    task,
    base_dir: str,
    output_dir: str,
    output_subdir: str = "",
    rank_col: str = "rank",
) -> str:
    """Write original designed CIF and inject license blocks."""
    task_name = task["task_name"]
    src = os.path.join(
        base_dir,
        f"global_run_{task['run_idx']}",
        task_name,
        f"seed_{task['seed']}",
        "predictions",
        task["name"] + ".cif",
    )
    dst = os.path.join(output_dir, output_subdir, f"rank_{task[rank_col]}.cif")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    replace_last_xpb_chain_sequence(src, dst, task["sequence"])
    add_terms_to_cif(dst, dst)
    return dst


def save_af2_docked(
    task,
    base_dir: str,
    output_dir: str,
    output_subdir: str = "",
) -> str:
    """Copy AF2-docked PDB to <output>/<task>/<output_subdir>/rank_<k>.pdb"""
    task_name, sample_name, rank = task["task_name"], task["name"], task["rank"]
    matches = glob(
        os.path.join(
            base_dir,
            f"global_run_{task['run_idx']}",
            task_name,
            f"seed_{task['seed']}",
            "predictions",
            "af2_pred",
            f"*{sample_name}_seq{task['seq_idx']}_model*.pdb",
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expect 1 AF2 PDB for {sample_name}, got {len(matches)}"
        )
    src = matches[0]
    dst = os.path.join(output_dir, output_subdir, f"rank_{rank}.cif")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pdb_to_cif(src, dst, reset_res_id=False, pad_chain_id=True)
    add_terms_to_cif(dst, dst)
    return dst


def save_ptx_docked(
    task,
    base_dir: str,
    output_dir: str,
    output_subdir: str = "",
    is_large: bool = True,
) -> str:
    """Write Protenix-docked CIF (with hotspot-based chain permutation)."""
    sample_name, rank = task["name"], task["rank"]
    ptx_redocked_name = f"run_{task['run_idx']}_{task['name']}_seq{task['seq_idx']}"
    matches = glob(
        os.path.join(
            base_dir,
            "ptx_pred" if is_large else "ptx_mini_pred",
            ptx_redocked_name + "_seq0",
            "seed_*",
            "predictions",
            f"{ptx_redocked_name}*.cif",
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expect 1 PTX CIF for {sample_name}, got {len(matches)}"
        )
    src = matches[0]
    dst = os.path.join(output_dir, output_subdir, f"rank_{rank}.cif")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    add_terms_to_cif(src, dst)
    return dst


# -------- Misc helpers --------


def trim_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a curated subset of columns and rename AF2 metrics if present."""
    base_metrics = [
        "plddt",
        "ptm_binder",
        "ptm_target",
        "iptm",
        "ptm",
        "iptm_binder",
        "pred_design_rmsd",
    ]
    ptx_cols = ["ptx_" + m for m in base_metrics]
    ptx_mini_cols = ["ptx_mini_" + m for m in base_metrics]
    columns_to_keep = (
        [
            "rank",
            "task_name",
            "sequence",
            "af2_easy_success",
            "af2_opt_success",
            "ptx_success",
            "ptx_basic_success",
            "pLDDT",
            "pTM",
            "i_pTM",
            "pAE",
            "unscaled_i_pAE",
            "pLDDT_MONOMER",
            "pTM_MONOMER",
            "pAE_MONOMER",
            "bound_unbound_RMSD",
            "af2_binder_pred_design_rmsd",
            "af2_complex_pred_design_rmsd",
        ]
        + ptx_cols
        + ptx_mini_cols
        + ["alpha", "beta", "loop", "Rg", "chosen_struct_type", "chosen_struct_path"]
    )
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[existing_cols].copy()

    # ---- Rename AF2 columns to canonical names ----
    df = df.rename(
        columns={
            "pLDDT": "af2_plddt",
            "pTM": "af2_ptm",
            "i_pTM": "af2_iptm",
            "pAE": "af2_pAE",
            "unscaled_i_pAE": "af2_ipAE",
            "pLDDT_MONOMER": "af2_monomer_plddt",
            "pTM_MONOMER": "af2_monomer_ptm",
            "pAE_MONOMER": "af2_monomer_pAE",
            "bound_unbound_RMSD": "af2_bound_unbound_RMSD",
            "af2_easy_success": "AF2-IG-easy-success",
            "af2_opt_success": "AF2-IG-success",
            "ptx_success": "Protenix-success",
            "ptx_basic_success": "Protenix-basic-success",
        }
    )
    return df


def parse_hotspot_json(json_list):
    """Parse hotspots from the input design JSON for downstream PTX permutation."""
    result = {}
    for entry in json_list:
        name = entry["name"]
        if "condition" in entry:  # structure input
            chain_ids = entry["condition"]["filter"]["chain_id"]
            hotspot = entry.get("hotspot", {})
            chain_mapping = {
                chain_id: f"{chr(65 + i)}0" for i, chain_id in enumerate(chain_ids)
            }
            new_hotspot = {
                chain_mapping[chain_id]: residues
                for chain_id, residues in hotspot.items()
                if chain_id in chain_mapping
            }
        else:  # sequence input
            assert "sequences" in entry
            new_hotspot = {}
            for i, seq_entity in enumerate(entry["sequences"]):
                entity_type = list(seq_entity.keys())[0]
                new_hotspot[f"{chr(65 + i)}0"] = seq_entity[entity_type].get(
                    "hotspot", {}
                )
        result[name] = new_hotspot
    return result


def check_tool_weights() -> None:
    """
    Sanity check for required tool weights.
    Equivalent to the original shell script.
    """
    root = os.environ.get("TOOL_WEIGHTS_ROOT")
    if not root:
        raise RuntimeError(
            "Environment variable TOOL_WEIGHTS_ROOT is not set.\n"
            "Please set TOOL_WEIGHTS_ROOT or run download_tool_weights.sh."
        )

    root = Path(root)

    required_files = [
        # ---- AF2 ----
        root / "af2" / "params_model_1.npz",
        root / "af2" / "params_model_1_ptm.npz",
    ]

    print(f"Checking tool weights in: {root}")

    missing = [p for p in required_files if not p.is_file()]
    if missing:
        print()
        print("Missing required tool weights:")
        for p in missing:
            print(f"   {p}")
        print()
        print("Please run:")
        print("   bash download_tool_weights.sh")
        sys.exit(1)
