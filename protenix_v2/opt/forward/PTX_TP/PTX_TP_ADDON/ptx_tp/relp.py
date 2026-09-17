# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
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
"""ptx_tp/relp.py -- row-block relative-position one-hot features (never an [N, N, 139] tensor beyond one
row block).

relp_rows(feats, i0, i1, r_max, s_max) reproduces protenix 2.0.0
embedders.RelativePositionEncoding.generate_relp (embedders.py 124-215) restricted to query rows
i0:i1 (GLOBAL token indices): integer comparisons/clips + one_hot + .float() => per-element exact.

get_relp_rows(input_feature_dict, i0, i1, r_max, s_max) resolves, in order:
  1. input_feature_dict['relp'] has a .rows(i0, i1) method (a single-card LazyRelp or TPLazyRelp)       -> use it;
  2. input_feature_dict['relp'] is a dense tensor [..., N, N, F]                                -> slice rows;
  3. otherwise                                                                                    -> relp_rows().
"""
from typing import Any, Dict

import torch
import torch.nn.functional as F

_INDEX_KEYS = ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")


def relp_rows(feats: Dict[str, Any], i0: int, i1: int, r_max: int = 32, s_max: int = 2) -> torch.Tensor:
    """relp[..., i0:i1, :, :] as float32 [.., i1-i0, N, 4*r_max + 2*s_max + 7] from the five [.., N] index
    vectors -- the stock generate_relp statements with the query axis restricted to rows i0:i1."""
    with torch.no_grad():
        asym_id, residue_index, entity_id, token_index, sym_id = (feats[k] for k in _INDEX_KEYS)
        ai, ri, ei, ti, si = (x[..., i0:i1, None] for x in (asym_id, residue_index, entity_id, token_index, sym_id))
        aj, rj, ej, tj, sj = (x[..., None, :] for x in (asym_id, residue_index, entity_id, token_index, sym_id))
        b_same_chain = (ai == aj).long()
        b_same_residue = (ri == rj).long()
        b_same_entity = (ei == ej).long()
        d_residue = torch.clip(input=ri - rj + r_max, min=0, max=2 * r_max) * b_same_chain + (1 - b_same_chain) * (
            2 * r_max + 1
        )
        a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))
        d_token = torch.clip(input=ti - tj + r_max, min=0, max=2 * r_max) * b_same_chain * b_same_residue + (
            1 - b_same_chain * b_same_residue
        ) * (2 * r_max + 1)
        a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))
        d_chain = torch.clip(input=si - sj + s_max, min=0, max=2 * s_max) * b_same_entity + (1 - b_same_entity) * (
            2 * s_max + 1
        )
        a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))
        relp = torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain], dim=-1).float()
    return relp


def get_relp_rows(input_feature_dict: Dict[str, Any], i0: int, i1: int, r_max: int = 32, s_max: int = 2) -> torch.Tensor:
    relp = input_feature_dict.get("relp", None) if isinstance(input_feature_dict, dict) else None
    if relp is not None and hasattr(relp, "rows") and callable(getattr(relp, "rows")):
        return relp.rows(i0, i1)  # a LazyRelp (communication-free, global row indices)
    if torch.is_tensor(relp) and relp.dim() >= 3:
        return relp[..., i0:i1, :, :]
    return relp_rows(input_feature_dict, i0, i1, r_max=r_max, s_max=s_max)


class LazyRelpRows:
    """Minimal stand-in for input_feature_dict['relp'] when neither a LazyRelp nor the dense tensor
    is present: holds the five index vectors; .rows(i0, i1) -> relp row block.  Interface-compatible with
    the single-card LazyRelp (.rows/.shape/.to)."""

    def __init__(self, feats: Dict[str, Any], r_max: int = 32, s_max: int = 2):
        self.feats = {k: feats[k] for k in _INDEX_KEYS}
        self.r_max, self.s_max = r_max, s_max
        n = self.feats["asym_id"].shape[-1]
        self.shape = tuple(self.feats["asym_id"].shape[:-1]) + (n, n, 4 * r_max + 2 * s_max + 7)
        self.dtype = torch.float32
        self.device = self.feats["asym_id"].device
        self.is_cuda = self.feats["asym_id"].is_cuda

    def rows(self, i0: int, i1: int) -> torch.Tensor:
        return relp_rows(self.feats, i0, i1, self.r_max, self.s_max)

    def dim(self):
        return len(self.shape)

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def to(self, *a, **k):
        return self

    def __repr__(self):
        return f"LazyRelpRows(shape={self.shape})"
