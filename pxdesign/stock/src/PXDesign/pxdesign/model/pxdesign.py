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

from typing import Any, Optional

import torch
import torch.nn as nn
from protenix.model.generator import InferenceNoiseScheduler
from protenix.model.modules.diffusion import DiffusionModule
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import autocasting_disable_decorator

from pxdesign.model.embedders import DesignConditionEmbedder
from pxdesign.model.generator import sample_diffusion

logger = get_logger(__name__)


class ProtenixDesign(nn.Module):

    def __init__(self, configs) -> None:

        super().__init__()
        self.configs = configs
        self.c_z = configs.c_z
        self.c_s = configs.c_s
        self.c_s_inputs = configs.c_s_inputs

        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        self.design_condition_embedder = DesignConditionEmbedder(configs=configs)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        # self.design_distogram_head = DistogramHead(
        #     **configs.model.design_distogram_head
        # )
        # self.design_diffusion_distogram = DistogramHead(
        #     **configs.model.design_diffusion_distogram
        # )

    def get_condition_embedding(
        self,
        input_feature_dict: dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:

        # Design Condition
        # s_inputs: same as protenix, except that 'design tokens' are added
        # z: encodes 'conditional_templ' & 'conditional_templ_mask'
        s_inputs, z = self.design_condition_embedder(
            input_feature_dict=input_feature_dict,
            chunk_size=chunk_size,
        )
        s = s_inputs.new_zeros(size=s_inputs.shape[:-1] + (self.configs.c_s,))
        return s_inputs, s, z

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        if hasattr(self.configs.sample_diffusion, "eta_schedule"):
            _configs.update(
                {"step_scale_eta": self.configs.sample_diffusion.eta_schedule}
            )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        pred_dict = self._design_inference_loop(
            input_feature_dict=input_feature_dict,
            mode=mode,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return pred_dict

    def _design_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:

        N_token = input_feature_dict["residue_index"].shape[-1]

        pred_dict = {}
        s_inputs, s, z = self.get_condition_embedding(
            input_feature_dict=input_feature_dict,
            chunk_size=chunk_size,
        )
        # if token=5000, template_distogram occupies 14.53G memory
        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        torch.cuda.empty_cache()

        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        print(f"Design inference with {N_step} N_step")

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        print(f"Protenix-Design sample diffusion: {self.configs.sample_diffusion}")
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
        )

        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()

        return pred_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        mode: str = "inference",
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:

        assert mode == "inference", mode
        chunk_size = self.configs.infer_setting.chunk_size
        pred_dict = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            mode=mode,
            chunk_size=chunk_size,
        )

        return pred_dict
