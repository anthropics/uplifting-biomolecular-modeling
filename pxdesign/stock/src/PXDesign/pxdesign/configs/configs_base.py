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

# pylint: disable=C0114,C0301
from protenix.config.extend_types import GlobalConfigValue, ValueMaybeNone

basic_configs = {
    "load_checkpoint_path": "",
    "load_strict": True,
    "deterministic": False,
    "model_name": "pxdesign_v0.1.0",  # train model name
}
model_configs = {
    # Model
    "c_s": 384,
    "c_z": 128,
    "c_s_inputs": 449,  # c_s_inputs == c_token + 32 + 32 + 1
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 16.0,
    "diffusion_batch_size": 48,
    "diffusion_chunk_size": ValueMaybeNone(4),  # chunksize of diffusion_batch_size
    "blocks_per_ckpt": ValueMaybeNone(
        1
    ),  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    # switch of kernels
    "use_memory_efficient_kernel": False,  # whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.
    "use_deepspeed_evo_attention": False,
    "use_flash": False,
    "use_lma": False,
    "use_xformer": False,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "loss_metrics_sparse_enable": True,  # the swicth for both sparse lddt metrics and sparse bond/smooth lddt loss
    "skip_amp": {
        "sample_diffusion": True,
        "confidence_head": True,
        "sample_diffusion_training": True,
        "loss": True,
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(
            64
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "sample_diffusion_chunk_size": ValueMaybeNone(
            10
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "lddt_metrics_sparse_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "lddt_metrics_chunk_size": ValueMaybeNone(
            1
        ),  # only works if loss_metrics_sparse_enable, can set as default 1
    },
    "inference_noise_scheduler": {
        "s_max": 160.0,
        "s_min": 4e-4,
        "rho": 7,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "sample_diffusion": {
        "gamma0": 1.0,
        "gamma_min": 0.01,
        "noise_scale_lambda": 1.003,
        "N_step": 400,
        "N_sample": 100,
        "eta_schedule": {"type": "piecewise_65", "min": 1.0, "max": 2.5},
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "N_cycle": 4,
        "condition_embedding_drop_rate": 0.0,
        "confidence_embedding_drop_rate": 0.0,
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_token": 768,
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "atom_encoder": {
                "n_blocks": 4,
                "n_heads": 4,
            },
            "transformer": {
                "n_blocks": 16,
                "n_heads": 16,
            },
            "atom_decoder": {
                "n_blocks": 4,
                "n_heads": 4,
            },
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
    },
}

design_model_configs = {
    "condition_embedder": {
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
        },
        "template_embedder": {
            "c_templ_in": 64 + 1,
            "c_z": GlobalConfigValue("c_z"),
        },
    },
    "fuse_condition": {
        "c_s": GlobalConfigValue("c_s"),
        "c_z": GlobalConfigValue("c_z"),
        "c_s_inputs": GlobalConfigValue("c_s_inputs"),
        "fuse_s": False,
        "fuse_z": True,
        "fuse_s_inputs": False,
        "s_clamp": 30000.0,
    },
    "sequence_head": {
        "pairformer": {
            "n_blocks": 4,
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "n_heads": 8,
            "dropout": 0.20,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "seq_encoder": {
            "min_bin": 2.0,
            "max_bin": 12.0,
            "no_bins": 20,
            "eps": 1e-6,
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "restype_num": 32 + 4,
        },
        "seq_single_decoder": {
            "c_s": GlobalConfigValue("c_s"),
            "c_s_out": 21,
            "dropout_rate": 0.2,
        },
        "seq_pair_decoder": {
            "c_z": GlobalConfigValue("c_z"),
            "num_token": 21,
            "c_z_out": 441,
            "dropout_rate": 0.2,
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
    },
    "design_diffusion_distogram": {
        "c_z": 768,
        "no_bins": GlobalConfigValue("no_bins"),
    },
    "design_distogram_head": {
        "c_z": GlobalConfigValue("c_z"),
        "no_bins": GlobalConfigValue("no_bins"),
    },
}
model_configs["model"].update(design_model_configs)

configs = {**basic_configs, **model_configs}
