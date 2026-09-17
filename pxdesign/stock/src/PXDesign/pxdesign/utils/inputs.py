import argparse
import json
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from protenix.utils.file_io import load_gzip_pickle

from pxdesign.data.utils import CIFWriter
from pxdesign.utils.infer import convert_to_bioassembly_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def parse_yaml_to_json(yaml_path, json_path=None):
    """
    Parses the YAML config and converts it to the
    JSON structure required by PXDesign model.
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    # --- 1. Top Level Fields ---
    # Default task name uses filename if not provided
    default_name = os.path.splitext(os.path.basename(yaml_path))[0]
    task_name = cfg.get("task_name", default_name)

    # Binder length (Required)
    if "binder_length" not in cfg:
        raise ValueError("Missing required field: 'binder_length'")
    binder_length = int(cfg["binder_length"])

    # --- 2. Target Parsing ---
    target_cfg = cfg.get("target", {})
    if "file" not in target_cfg:
        raise ValueError("Missing required field: 'target.file'")

    target_file_path = target_cfg["file"]
    if not os.path.exists(target_file_path):
        raise FileNotFoundError(f"Target structure file not found: {target_file_path}")

    # Initialize containers
    chain_ids = []
    crop_dict = {}
    hotspot_dict = {}
    msa_dict_per_chain = {}

    # --- 3. Chains Parsing ---
    chains_cfg = target_cfg.get("chains", {})
    if not chains_cfg:
        raise ValueError("Missing required field: 'target.chains'")

    for chain_id, props in chains_cfg.items():
        chain_id = str(chain_id)
        chain_ids.append(chain_id)

        # Handle "A: all" or "A: null" shorthand
        if props is None or (
            isinstance(props, str) and props.lower() in ["all", "full"]
        ):
            props = {}

        # --- Crop Logic ---
        # User YAML: ["1-50", "80-100"] OR "1-100" OR "all"
        # Internal JSON: "1-50,80-100" OR None
        if "crop" in props:
            raw_crop = props["crop"]
            crop_val = None

            if isinstance(raw_crop, list):
                # Join list into comma-separated string
                crop_val = ",".join(str(x) for x in raw_crop)
            elif isinstance(raw_crop, str):
                if raw_crop.lower() in ["all", "full"]:
                    crop_val = None
                else:
                    crop_val = raw_crop

            if crop_val:
                crop_dict[chain_id] = crop_val

        # --- Hotspot Logic ---
        if "hotspots" in props:
            # YAML list is already a Python list
            hotspot_dict[chain_id] = props["hotspots"]

        # --- MSA Logic ---
        if "msa" in props and props["msa"]:
            msa_path = props["msa"]
            for fname in ["pairing.a3m", "non_pairing.a3m"]:
                if not os.path.exists(os.path.join(msa_path, fname)):
                    raise FileNotFoundError(
                        f"MSA file not found: {os.path.join(msa_path, fname)}"
                    )
            msa_config = {
                "precomputed_msa_dir": msa_path,  # Default to None (Auto)
                "pairing_db": "uniref100",
            }

            msa_dict_per_chain[chain_id] = msa_config

    # --- 4. Construct Internal JSON Structure ---
    json_task = {
        "name": task_name,
        "condition": {
            "structure_file": target_file_path,
            "filter": {
                "chain_id": chain_ids,
                "crop": crop_dict,
            },
            "msa": msa_dict_per_chain,
        },
        "hotspot": hotspot_dict,
        "generation": [
            {
                "type": "protein",
                "length": binder_length,
                "count": 1,
            }
        ],
    }

    if json_path is not None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump([json_task], f, cls=NpEncoder)

    return [json_task]


def check_yaml_file(yaml_path: str):
    print(f"Checking YAML file: {yaml_path}...")
    result = parse_yaml_to_json(yaml_path, None)
    print("✅ YAML file is valid.")


def process_input_file(input_path: str, out_dir: str = None) -> str:
    """
    Process the input file path to ensure it has the correct extension.
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check file extension
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in [".json", ".yaml"]:
        raise ValueError(
            f"Unsupported input file format: {ext}. "
            "Supported formats are: JSON, YAML."
        )

    # Convert YAML to JSON if necessary
    if ext == ".yaml":
        base, _ = os.path.splitext(os.path.basename(input_path))
        out_dir = out_dir or os.path.dirname(input_path)
        json_path = os.path.join(out_dir, f"{base}.json")

        parse_yaml_to_json(input_path, json_path)
        input_path = json_path

    return input_path


def dump_bioassembly_to_cif(
    bio_dict: Union[str, Path, dict],
    output_cif: Union[str, Path],
    dump_unresolved_atoms: bool = False,
):
    """
    Dump a bioassembly dict to CIF.
    """
    if isinstance(bio_dict, str) or isinstance(bio_dict, Path):
        bio_dict = load_gzip_pickle(bio_dict)
    atom_array = bio_dict["atom_array"]

    if not dump_unresolved_atoms:
        mask = atom_array.is_resolved
    else:
        mask = None

    entity_poly_type = bio_dict["entity_poly_type"]
    writer = CIFWriter(
        atom_array=atom_array,
        entity_poly_type=entity_poly_type,
        atom_array_output_mask=mask,
    )
    writer.save_to_cif(
        output_cif,
        entry_id=Path(output_cif).stem.split(".")[0],
        include_bonds=False,
    )

    return


def generate_pml_from_json_input(cif_file_path: str, json_file_path: str) -> dict:
    """
    generate PML script from input_dict. for visualizing in pymol.
    cif_file_path: path to the cif file of the target structure.
    json_file_path: path to the json file of the input dict.
    """
    cif_fname = os.path.basename(cif_file_path)
    cif_file_dir = os.path.dirname(cif_file_path)
    pymol_cmds = [
        "load " + cif_fname,
        "hide",
        "show cartoon",
        'cmd.util.cbc(selection="(elem C)")',
    ]
    json_task_dict = json.load(open(json_file_path, "r"))[0]

    cond_dict = json_task_dict.get("condition", {})
    if "filter" in cond_dict:
        crop_region = cond_dict["filter"].get("crop", {})
        crop_sele_list = []
        for chain_id, chain_crop in crop_region.items():
            chain_crop_sele = (
                f"(chain {chain_id} and resi " + chain_crop.replace(",", "+") + ")"
            )
            crop_sele_list.append(chain_crop_sele)
        if len(crop_sele_list) > 0:
            crop_sele_str = "select crop, " + " OR ".join(crop_sele_list)
            pymol_cmds.append(crop_sele_str)
            pymol_cmds.append("color marine, crop and elem C")

    hotspot_residues = json_task_dict.get("hotspot", {})
    hotspot_sele_list = []
    for chain_id, residues in hotspot_residues.items():
        hotspot_sele = (
            f"(chain {chain_id} and resi " + "+".join([str(x) for x in residues]) + ")"
        )
        hotspot_sele_list.append(hotspot_sele)
    if len(hotspot_sele_list) > 0:
        hotspot_sele_str = "select hotspot, " + " OR ".join(hotspot_sele_list)
        pymol_cmds.append(hotspot_sele_str)
        pymol_cmds.extend(["color pink, hotspot and elem C", "show sticks, hotspot"])

    pymol_cmds.append("color grey70, not (hotspot OR crop)")
    pml_script_path = os.path.join(cif_file_dir, f"{Path(cif_file_path).stem}.pml")
    with open(pml_script_path, "w") as f:
        f.write("\n".join(pymol_cmds))
    return


def dump_target_cif_from_input_file(file_path: str, out_dir: str) -> dict:
    """
    Parse target structure from input_dict.
    """
    if os.path.splitext(file_path)[1].lower() == ".json":
        json_path = file_path
    else:
        assert (
            os.path.splitext(file_path)[1].lower() == ".yaml"
        ), f"Input file must be JSON or YAML, but got {os.path.splitext(file_path)[1]}"
        json_path = os.path.join(out_dir, "tmp", f"{Path(file_path).stem}.json")
        parse_yaml_to_json(file_path, json_path)
    with open(json_path, "r") as f:
        json_task_dict = json.load(f)[0]
    bioassembly_dict = convert_to_bioassembly_dict(
        json_task_dict, os.path.join(out_dir, "tmp")
    )
    if isinstance(bioassembly_dict, str):
        bioassembly_dict = load_gzip_pickle(bioassembly_dict)

    output_cif = os.path.join(out_dir, f"{Path(file_path).stem}_parsed_target.cif")
    dump_bioassembly_to_cif(bioassembly_dict, output_cif)
    generate_pml_from_json_input(output_cif, json_path)
    return


# --- CLI Wrapper for Debugging ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", help="Path to input YAML file")
    parser.add_argument(
        "--output_json_file", help="Path to output JSON file", default=None
    )
    args = parser.parse_args()

    try:
        result = parse_yaml_to_json(args.yaml_file, args.output_json_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
