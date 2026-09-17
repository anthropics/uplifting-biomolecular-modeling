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
import json
import os
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray
from protenix.data.utils import save_atoms_to_cif
from protenix.utils.file_io import save_json
from protenix.utils.torch_utils import round_values


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    # Remove atom_coordinate
    full_confidence_dict.pop("atom_coordinate")
    # Remove atom_is_polymer
    full_confidence_dict.pop("atom_is_polymer")
    # Keep two decimal places
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


class DataDumper:
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
        )

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data based on the dataset name, sample name, and seed.
        """
        dump_dir = os.path.join(
            self.base_dir, dataset_name, sample_name, f"seed_{seed}"
        )
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump raw predictions from the model:
            structure: Save the predicted coordinates as CIF files.
            confidence: Save the confidence data as JSON files.
        """
        prediction_save_dir = os.path.join(dump_dir, "predictions")
        os.makedirs(prediction_save_dir, exist_ok=True)

        self._save_structure(
            pred_dict["coordinate"],
            prediction_save_dir,
            pdb_id,
            atom_array,
            entity_poly_type,
        )
        self._save_confidence(
            data=pred_dict, prediction_save_dir=prediction_save_dir, sample_name=pdb_id
        )
        self._mark_task_complete(dump_dir)

    def _mark_task_complete(self, dump_dir):
        success_file_path = os.path.join(dump_dir, f"SUCCESS_FILE")
        success_data = {"prediction": True}
        with open(success_file_path, "w") as f:
            json.dump(success_data, f)

    def check_completion(self, dataset_name, sample_name, seed):
        dump_dir = self._get_dump_dir(dataset_name, sample_name, seed)
        success_file_path = os.path.join(dump_dir, f"SUCCESS_FILE")  # json file
        return os.path.exists(success_file_path)

    def _save_structure(
        self,
        pred_coordinates,
        prediction_save_dir,
        sample_name,
        atom_array,
        entity_poly_type=None,
    ):
        N_sample = pred_coordinates.shape[0]
        for sample_idx in range(N_sample):
            output_fpath = os.path.join(
                prediction_save_dir, f"{sample_name}_sample_{sample_idx}.cif"
            )
            # fake b_factor
            atom_array.set_annotation(
                "b_factor", np.round(np.zeros(len(atom_array)).astype(float), 2)
            )
            if "occupancy" not in atom_array._annot:
                # fake occupancy
                atom_array.set_annotation(
                    "occupancy", np.round(np.ones(len(atom_array)), 2)
                )
            save_structure_cif(
                atom_array,
                pred_coordinates[sample_idx],
                output_fpath,
                entity_poly_type,
                sample_name,
                # save_wounresol=False,
            )

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
    ):
        N_sample = (
            len(data["summary_confidence"]) if "summary_confidence" in data else 0
        )
        if N_sample <= 0:
            return

        for idx, rank in enumerate(range(N_sample)):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{rank}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)


def save_structure_cif(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
):
    """
    Save the predicted structure to a CIF file.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates for the structure.
        output_fpath (str): The output file path for saving the CIF file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    save_atoms_to_cif(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
