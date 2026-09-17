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
import torch.nn.functional as F
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.modules.transformer import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )
        # Line2
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )
        if not self.training and a.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return s_inputs


class InputFeatureEmbedderDesign(InputFeatureEmbedder):
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        c_s_inputs: int = 449,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
            c_input (int, optional): input embedding dim. Defaults to 449.
        """
        super().__init__(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token)
        self.c_s_inputs = c_s_inputs
        # Line2
        # Design restype = 32 + 4 (need to be designed)
        # profile need to + 4 to fit the dim
        self.input_feature = {
            "restype": 32 + 4,
            # "profile": 32 + 4,
            # "deletion_mean": 1,
            "plddt": 1,
            "hotspot": 1,
            "add_feat1": 4,
            "add_feat2": 4,
        }
        self.input_map = nn.Linear(
            c_token + sum(self.input_feature.values()), c_s_inputs
        )

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 + 32 + 1 + 1 :=450]
            torch.Tensor: token embedding
                for design:
                [..., N_token, 384 (c_token) + 32 + 4 + 4 + 32 + 1 + 1 + 1 + 4 := 463]
        """
        add_feat = torch.zeros_like(input_feature_dict["deletion_mean"])
        add_feat = F.one_hot(add_feat.to(torch.int64), num_classes=4)
        input_feature_dict["add_feat1"] = add_feat
        input_feature_dict["add_feat2"] = add_feat

        # pLDDT mapping
        if "plddt" not in input_feature_dict:
            input_feature_dict["plddt"] = torch.zeros_like(
                input_feature_dict["deletion_mean"]
            )

        # hotspot mapping
        if "hotspot" not in input_feature_dict:
            input_feature_dict["hotspot"] = torch.zeros_like(
                input_feature_dict["deletion_mean"]
            )

        s_inputs = super().forward(input_feature_dict, inplace_safe, chunk_size)
        s_inputs = self.input_map(s_inputs)
        return s_inputs


class ConditionTemplateEmbedder(nn.Module):
    """
    Design module to encode conditional structure
    """

    def __init__(self, c_templ_in: int = 64 + 1, c_z: int = 128) -> None:
        super(ConditionTemplateEmbedder, self).__init__()
        self.c_templ_in = c_templ_in
        self.c_z = c_z
        self.embedder = nn.Embedding(self.c_templ_in, self.c_z)

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        conditional_templ = input_feature_dict["conditional_templ"]
        pair_mask = input_feature_dict["conditional_templ_mask"]

        conditional_templ = pair_mask * (1 + conditional_templ)
        # z_conditional_templ = self.embedder(conditional_templ) * pair_mask[..., None]
        z_conditional_templ = self.embedder(conditional_templ)
        return z_conditional_templ


class DesignConditionEmbedder(nn.Module):

    def __init__(self, configs) -> None:
        super(DesignConditionEmbedder, self).__init__()
        self.configs = configs
        self.input_embedder = InputFeatureEmbedderDesign(
            **configs.model.condition_embedder.input_embedder
        )
        self.condition_template_embedder = ConditionTemplateEmbedder(
            **configs.model.condition_embedder.template_embedder
        )

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:

        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z = self.condition_template_embedder(
            input_feature_dict
        )  # [..., N_token, N_token, c_z]

        return s_inputs, z


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        b_same_chain = (
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        rel_pos_index = (
            input_feature_dict["residue_index"][..., :, None]
            - input_feature_dict["residue_index"][..., None, :]
        )

        d_residue = torch.clip(
            input=rel_pos_index + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
            - input_feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = torch.clip(
            input=input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        if self.training:
            p = self.linear_no_bias(
                torch.cat(
                    [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                    dim=-1,
                ).float()
            )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
            return p
        else:
            del d_chain, d_token, d_residue, b_same_chain, b_same_residue
            origin_shape = a_rel_pos.shape[:-1]
            Ntoken = a_rel_pos.shape[-2]
            a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
            chunk_num = 1 if Ntoken < 3200 else 8
            a_rel_pos_chunks = torch.chunk(
                a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
            )
            a_rel_token_chunks = torch.chunk(
                a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
            )
            b_same_entity_chunks = torch.chunk(
                b_same_entity.reshape(-1, 1), chunk_num, dim=-2
            )
            a_rel_chain_chunks = torch.chunk(
                a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
            )
            start = 0
            p = None
            for i in range(len(a_rel_pos_chunks)):
                data = torch.cat(
                    [
                        a_rel_pos_chunks[i],
                        a_rel_token_chunks[i],
                        b_same_entity_chunks[i],
                        a_rel_chain_chunks[i],
                    ],
                    dim=-1,
                ).float()
                result = self.linear_no_bias(data)
                del data
                if p is None:
                    p = torch.empty(
                        (a_rel_pos.shape[-2], self.c_z),
                        device=a_rel_pos.device,
                        dtype=result.dtype,
                    )
                p[start : start + result.shape[0]] = result
                start += result.shape[0]
                del result
            del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
            p = p.reshape(*origin_shape, -1)
            if p.shape[-2] > 2000:
                torch.cuda.empty_cache()
            return p


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )
