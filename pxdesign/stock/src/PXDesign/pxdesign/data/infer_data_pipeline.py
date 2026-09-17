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

import json
import logging
import sys
import time
import traceback
import warnings
from copy import deepcopy
from typing import Any, Mapping

import protenix
import torch
from biotite.structure import AtomArray
from protenix.data.parser import MMCIFParser
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.file_io import load_gzip_pickle
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from pxdesign.data.json_to_feature import SampleDictToFeatures
from pxdesign.data.tokenizer import AtomArrayTokenizer
from pxdesign.data.utils import data_type_transform, make_dummy_feature
from pxdesign.utils.design import cano_seq_resname_with_mask, restype_onehot_encoded

sys.modules["meson"] = protenix

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(configs: Any) -> DataLoader:
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        use_msa=configs.use_msa,
    )
    # data = inference_dataset[0]
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        use_msa: bool = True,
    ) -> None:

        self.input_json_path = input_json_path
        self.use_msa = use_msa
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        dummy_feats = ["template", "msa"]
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        ## change fake to sequence/restype to mask sequence
        atom_array = self.change_fake_sequence_to_mask(atom_array, single_sample_dict)
        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms_new = atom_array[centre_atoms_indices]
        restype = cano_seq_resname_with_mask(centre_atoms_new)
        restype_onehot = restype_onehot_encoded(restype)

        assert restype_onehot.shape == data["input_feature_dict"]["restype"].shape
        data["input_feature_dict"]["restype"] = restype_onehot
        return data, atom_array, time_tracker

    def change_fake_sequence_to_mask(self, atom_array, json_dict):

        json_prot_sequences = []
        ## Get centre atom restype of protein chain
        prot_mask = atom_array.mol_type == "protein"
        sorted_prot_chain_id = sorted(list(set(atom_array[prot_mask].chain_id)))

        for seq in json_dict["sequences"]:
            if "proteinChain" in seq:
                sequence = seq["proteinChain"]["sequence_prev"]
                num = seq["proteinChain"]["count"]
                for i in range(int(num)):
                    json_prot_sequences.append(sequence)

        for seq, chain_id in zip(json_prot_sequences, sorted_prot_chain_id):
            for i in range(len(seq)):
                if seq[i] == "j":
                    res_mask = (atom_array.chain_id == chain_id) & (
                        atom_array.res_id == i + 1
                    )
                    atom_array.res_name[res_mask] = "xpb"

        return atom_array

    def make_mask_prot_seq(self, json_dict):
        """
        old version: read sequence data from json, includ designed seq & conditional seq
            and change j to G, x to XXX
        """
        for seq in json_dict["sequences"]:
            if "proteinChain" in seq:
                sequence = seq["proteinChain"]["sequence"]
                modified_sequence = sequence.replace("j", "G")
                modified_sequence = modified_sequence.replace("_", "G")
                seq["proteinChain"]["sequence_prev"] = seq["proteinChain"]["sequence"]
                seq["proteinChain"]["sequence"] = modified_sequence

            if "rnaChain" in seq:
                sequence = seq["proteinChain"]["sequence"]
                modified_sequence = sequence.replace("x", "G")
                seq["proteinChain"]["sequence_prev"] = seq["proteinChain"]["seuence"]
                seq["proteinChain"]["sequence"] = modified_sequence
        return json_dict

    def get_and_map_sequence_from_atom_array(
        self, atom_array, json_dict, chain_ids, path
    ):
        """
        add proteinChain, nucChain, Ligand or ions to json
        """

        bioassembly_dict = json_dict["condition"]["bioassembly_dict"]
        label_entity_id_to_sequence = bioassembly_dict.get("sequences", {})

        if "sequences" not in json_dict:
            json_dict["sequences"] = []

        for chain in chain_ids:
            chain_mask = atom_array.chain_id == chain
            chains = atom_array[chain_mask]
            chain_type = chains[0].mol_type
            label_entity_id = chains.label_entity_id[0]
            sequence = label_entity_id_to_sequence.get(label_entity_id, None)

            hotspot = None
            crop = None
            remove_unresolved_atoms = None
            msa_path = None

            if "hotspot" in json_dict and chain in json_dict["hotspot"].keys():
                hotspot = json_dict["hotspot"][chain]

            if (
                "remove_unresolved_atoms" in json_dict
                and chain in json_dict["remove_unresolved_atoms"].keys()
            ):
                remove_unresolved_atoms = json_dict["remove_unresolved_atoms"][chain]

            if (
                "crop" in json_dict["condition"]["filter"]
                and chain in json_dict["condition"]["filter"]["crop"].keys()
            ):
                crop = json_dict["condition"]["filter"]["crop"][chain]
            if (
                "msa" in json_dict["condition"]
                and chain in json_dict["condition"]["msa"]
            ):
                msa_path = json_dict["condition"]["msa"][chain]

            if chain_type == "protein":
                one_dict = {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                        "sequence_type": "condition",
                        "path": path,
                        "hotspot": hotspot,
                        "crop": crop,
                        "remove_unresolved_atoms": remove_unresolved_atoms,
                        "json_chain_id": chain,
                        "use_msa": self.use_msa,
                    }
                }
                if msa_path is not None:
                    one_dict["proteinChain"]["msa"] = msa_path
            elif chain_type == "dna":
                one_dict = {
                    "dnaSequence": {
                        "sequence": sequence,
                        "count": 1,
                        "sequence_type": "condition",
                        "path": path,
                        "hotspot": hotspot,
                        "crop": crop,
                        "remove_unresolved_atoms": remove_unresolved_atoms,
                        "json_chain_id": chain,
                        "use_msa": self.use_msa,
                    }
                }
                if msa_path is not None:
                    one_dict["dnaSequence"]["msa"] = msa_path
            elif chain_type == "rna":
                one_dict = {
                    "rnaSequence": {
                        "sequence": sequence,
                        "count": 1,
                        "sequence_type": "condition",
                        "path": path,
                        "crop": crop,
                        "remove_unresolved_atoms": remove_unresolved_atoms,
                        "hotspot": hotspot,
                        "json_chain_id": chain,
                        "use_msa": self.use_msa,
                    }
                }
                if msa_path is not None:
                    one_dict["rnaSequence"]["msa"] = msa_path
            elif chain_type == "ligand":
                one_dict = {
                    "condition_ligand": {
                        "ligand": "",
                        "count": 1,
                        "sequence_type": "condition",
                        "path": path,
                        "json_chain_id": chain,
                        "use_msa": self.use_msa,
                    }
                }
            else:
                print("Error: chain_type error")

            json_dict["sequences"].append(one_dict)

        return json_dict

    def make_gen_sequences(self, json_dict):
        """
        read json_dict and add designed sequences
        """

        if "sequences" not in json_dict:
            json_dict["sequences"] = []

        for gen_seq_dict in json_dict.get("generation", {}):
            assert "sequence" not in gen_seq_dict
            length = gen_seq_dict["length"]
            count = gen_seq_dict["count"]
            one_dict = {
                "proteinChain": {
                    "sequence": "j" * length,
                    "count": count,
                    "sequence_type": "design",
                    "use_msa": False,
                }
            }
            json_dict["sequences"].append(one_dict)
        return json_dict

    def get_and_map_ligand_from_ccd(self, json_dict):
        """
        for pocket only: add fake ligand chain from ccd
        """
        if "sequences" not in json_dict:
            json_dict["sequences"] = []
        for pid in json_dict["condition"]["ligands"]:
            ligand_name = pid["ligand"]
            count = pid["count"]
            one_dict = {
                "ligand": {
                    "ligand": ligand_name,
                    "count": count,
                    "sequence_type": "ccd",
                }
            }
            json_dict["sequences"].append(one_dict)

        return json_dict

    def make_cond_sequences(self, json_dict):
        """
        read json_dict and make condition sequences
        """
        atom_array = None

        if "condition" in json_dict:
            path_file = json_dict["condition"]["structure_file"]
            if path_file.endswith(".pkl.gz"):
                # Load bioassembly_dict
                bioassembly_dict = load_gzip_pickle(path_file)
            else:
                raise ValueError(f"Unsupported structure file {path_file}!")

            json_dict["condition"]["bioassembly_dict"] = bioassembly_dict
            atom_array = bioassembly_dict["atom_array"]

            if "filter" in json_dict["condition"]:
                filtered_chains = json_dict["condition"]["filter"]["chain_id"]
            else:
                filtered_chains = list(set(atom_array.chain_id))

            assert (
                len(filtered_chains) > 0
            ), "The number of chains mush be larger than one if `condition` is specified."
            all_chains = list(set(atom_array.chain_id))
            for c in filtered_chains:
                if c not in all_chains:
                    raise ValueError(
                        f"Chain {c} does not exist in the structure file (available: {all_chains}). "
                        f"Please check your json file!"
                    )
            self.check_input_validity(json_dict, filtered_chains)

            json_dict = self.get_and_map_sequence_from_atom_array(
                atom_array, json_dict, filtered_chains, path_file
            )

        return json_dict

    def check_input_validity(self, json_dict, available_chains):
        if "hotspot" in json_dict:
            for c in json_dict["hotspot"]:
                if c not in available_chains:
                    raise ValueError(
                        f"Hotspot specification on chain {c} is invalid (available: {available_chains}). Please check your json file!"
                    )

        if "crop" in json_dict["condition"]["filter"]:
            assert isinstance(json_dict["condition"]["filter"]["crop"], dict)
            for c in json_dict["condition"]["filter"]["crop"]:
                if c not in available_chains:
                    raise ValueError(
                        f"Crop specification on chain {c} is invalid (available: {available_chains}). Please check your json file!"
                    )

        if "msa" in json_dict["condition"]:
            for c in json_dict["condition"]["msa"]:
                if c not in available_chains:
                    raise ValueError(
                        f"MSA specification on chain {c} is invalid (available: {available_chains}). Please check your json file!"
                    )

    def __len__(self) -> int:
        return len(self.inputs)

    def process_sample_dict(self, single_sample_dict):
        sample_name = single_sample_dict["name"]
        logger.info(f"Featurizing {sample_name}...")
        logger.info(f"json dict with keys: {list(single_sample_dict.keys())}")

        assert any(
            k in single_sample_dict for k in ("condition", "sequences", "generation")
        )
        processed_sample_dict = deepcopy(single_sample_dict)
        processed_sample_dict["sequences"] = []
        assert "sequences" not in single_sample_dict.keys()
        if "condition" in single_sample_dict:
            # should not have any sequences
            processed_sample_dict = self.make_cond_sequences(processed_sample_dict)

        if "generation" in single_sample_dict:
            # append gen sequence to "sequences"
            processed_sample_dict = self.make_gen_sequences(processed_sample_dict)

        processed_sample_dict = self.make_mask_prot_seq(processed_sample_dict)
        return processed_sample_dict

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            processed_sample_dict = self.process_sample_dict(single_sample_dict)
            data, atom_array, _ = self.process_one(
                single_sample_dict=processed_sample_dict
            )
            data["sequences"] = processed_sample_dict["sequences"]
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"

        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, atom_array, error_message
