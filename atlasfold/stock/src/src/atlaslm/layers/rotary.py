# Started from the implementation of RotaryEmbedding in the ESM-3 repository.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTE: this implementation is from LLaMA 2:
# https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/08639a72e17836184096ae6a7e2766f2a34c3e36/modeling_flash_llama.py#L114

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: int = 20000,
    ):
        super().__init__()
        self.dim: int = dim
        self.base: float = float(base)
        self.max_seqlen: int = max_seqlen
        self.init_buffers()

    def init_buffers(self, device: str | torch.device | None = None):
        self.inv_freq: torch.Tensor
        inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for the cos and sin tables
        cos, sin = self._get_cos_sin(self.max_seqlen)
        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    def _get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seqlen, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs).tile(1, 2)  # [seqlen, dim]
        sin = torch.sin(freqs).tile(1, 2)  # [seqlen, dim]
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: (*, seqlen, nheads, headdim)
        k: (*, seqlen, nheads, headdim)
        pos_id: (*, seqlen)
        """
        if pos_id is None:
            seqlen = q.shape[-3]
            cos = self._cos_cached[:seqlen]  # [seqlen, dim]
            sin = self._sin_cached[:seqlen]  # [seqlen, dim]
        else:
            cos = self._cos_cached[pos_id]  # [*, seqlen, headdim]
            sin = self._sin_cached[pos_id]  # [*, seqlen, headdim]
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q_ = self.apply_rotary_emb(q, cos, sin)
        k_ = self.apply_rotary_emb(k, cos, sin)
        return q_, k_

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        x: (*, seqlen, nheads, headdim)
        cos, sin: (seqlen, dim)
        """
        cos, sin = cos[..., None, :], sin[..., None, :]
        return x * cos + rotate_half(x) * sin
