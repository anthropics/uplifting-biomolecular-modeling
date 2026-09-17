# Copyright 2026 Anthropic, PBC. Restates DiffusionHead._pair_conditioning of xfold/nn/diffusion_head.py (Copyright 2024 xfold authors; Copyright 2024 DeepMind Technologies Limited) in row chunks.
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
"""Row-chunked restatement of ``DiffusionHead._pair_conditioning`` — the ``big`` mode's lever ``paircond_chunk``
(``af3_torch_opt/big.py``).

``_pair_conditioning`` (``diffusion_head.py``) is the step-invariant pair conditioning of the diffusion sampler: the relative-position
encoding of the token grid (``featurization.create_relative_encoding``: three int64 one-hots ``[N, N, 66 | 66 | 6]`` and their concat
``[N, N, 139]``), ``use_conditioning * pair`` (an fp32 copy of the trunk's ``[N, N, 128]`` pair embedding), their fp32 concat ``[N, N, 267]``,
its LayerNorm, the ``267 -> 128`` projection and two transitions (LayerNorm + a ``128 -> 512`` gated linear unit each). Whole, the statement
holds of the order of 5 kB per token pair of transients at once (about 20 GB at 2,000 tokens, 47 GB at 3,000) — the statement that runs
out of memory first as inputs grow. Every one of those operations is per token pair over the channel dimension, so the same statements
evaluated for a block of ROWS ``[r0:r1, :, :]`` and written into one preallocated ``[N, N, 128]`` result give the same pair conditioning
with the transients of ``rows x N`` pairs instead of ``N x N``:

  * the relative encoding of a row block is ``create_relative_encoding`` with every LEFT (row) operand sliced to ``r0:r1`` — the same
    integer arithmetic; its one-hots are written as ``(value == class)`` in the pair dtype directly (``one_hot(v)[..., c] == (v == c)``:
    identical values, no int64 intermediates);
  * ``use_conditioning * pair``: ``use_conditioning`` is the Python bool ``True`` on the sampler's path, and ``True * x`` is ``x`` — the
    block is read in place instead of copied (``False`` keeps the multiply: zeros, as stock);
  * LayerNorm / Linear / the transitions run on the block as they run on the whole (leading dims are batch dims to each of them).

Class: tolerance — the per-pair arithmetic is unchanged, but a GEMM library may choose another tile or split for another leading size, so
byte equality with the whole statement on a GPU is not claimed (on CPU fp32 the kit's tests hold the two to ``1e-5``; the relative encoding
is integer-exact). ``rows >= N`` evaluates one block = the whole statement's own operations in the same order.
"""

import torch


def _one_hot_as(v: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
  """``torch.nn.functional.one_hot(v.to(int64), num_classes).to(dtype)`` without the int64 ``[..., num_classes]`` intermediate:
  the same 0 / 1 values, written in ``dtype``."""
  cls = torch.arange(num_classes, device=v.device, dtype=v.dtype)
  return (v[..., None] == cls).to(dtype=dtype)


def relative_encoding_rows(seq_features, r0: int, r1: int, max_relative_idx: int, max_relative_chain: int,
                           dtype: torch.dtype) -> torch.Tensor:
  """``featurization.create_relative_encoding(seq_features, max_relative_idx, max_relative_chain)[r0:r1].to(dtype)`` — the same
  statements (``featurization.py`` ``create_relative_encoding``) with every left operand sliced to rows ``r0:r1``: ``[r1 - r0, N, 139]``."""
  token_index = seq_features.token_index
  residue_index = seq_features.residue_index
  asym_id = seq_features.asym_id
  entity_id = seq_features.entity_id
  sym_id = seq_features.sym_id

  left_asym_id, right_asym_id = asym_id[r0:r1, None], asym_id[None, :]
  left_residue_index, right_residue_index = residue_index[r0:r1, None], residue_index[None, :]
  left_token_index, right_token_index = token_index[r0:r1, None], token_index[None, :]
  left_entity_id, right_entity_id = entity_id[r0:r1, None], entity_id[None, :]
  left_sym_id, right_sym_id = sym_id[r0:r1, None], sym_id[None, :]

  rel_feats = []
  # relative positions: one-hot of the clipped residue distance along a chain
  offset = left_residue_index - right_residue_index
  clipped_offset = torch.clip(offset + max_relative_idx, min=0, max=2 * max_relative_idx)
  asym_id_same = left_asym_id == right_asym_id
  final_offset = torch.where(asym_id_same, clipped_offset, (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset))
  rel_feats.append(_one_hot_as(final_offset, 2 * max_relative_idx + 2, dtype))

  # relative token index: one-hot of the clipped token distance inside a residue
  token_offset = left_token_index - right_token_index
  clipped_token_offset = torch.clip(token_offset + max_relative_idx, min=0, max=2 * max_relative_idx)
  residue_same = (left_asym_id == right_asym_id) & (left_residue_index == right_residue_index)
  final_token_offset = torch.where(residue_same, clipped_token_offset,
                                   (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset))
  rel_feats.append(_one_hot_as(final_token_offset, 2 * max_relative_idx + 2, dtype))

  # same entity id
  entity_id_same = left_entity_id == right_entity_id
  rel_feats.append(entity_id_same.to(dtype=dtype)[..., None])

  # relative chain id inside each symmetry class
  rel_sym_id = left_sym_id - right_sym_id
  clipped_rel_chain = torch.clip(rel_sym_id + max_relative_chain, min=0, max=2 * max_relative_chain)
  final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                (2 * max_relative_chain + 1) * torch.ones_like(clipped_rel_chain))
  rel_feats.append(_one_hot_as(final_rel_chain, 2 * max_relative_chain + 2, dtype))

  return torch.concatenate(rel_feats, dim=-1)


def pair_conditioning_rows(head, batch, embeddings, use_conditioning: bool, rows: int) -> torch.Tensor:
  """``DiffusionHead._pair_conditioning(batch, embeddings, use_conditioning)`` evaluated per block of ``rows`` rows of the token grid
  into one preallocated ``[N, N, C]`` result (``head``: the DiffusionHead — its ``pair_cond_initial_norm`` / ``pair_cond_initial_projection`` /
  ``pair_transition_0`` / ``pair_transition_1``). The number of blocks evaluated is added to ``head._paircond_blocks`` (the lever's census)."""
  pair = embeddings['pair']
  n = int(pair.shape[0])
  rows = max(1, int(rows))
  out = None
  blocks = 0
  for r0 in range(0, n, rows):
    r1 = min(r0 + rows, n)
    if use_conditioning is True:                 # True * x == x: the block is read in place (no fp32 copy of it)
      pair_embedding = pair[r0:r1]
    else:
      pair_embedding = use_conditioning * pair[r0:r1]
    rel_features = relative_encoding_rows(batch.token_features, r0, r1, max_relative_idx=32, max_relative_chain=2,
                                          dtype=pair_embedding.dtype)
    features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)
    del pair_embedding, rel_features
    pair_cond = head.pair_cond_initial_projection(head.pair_cond_initial_norm(features_2d))
    del features_2d
    pair_cond += head.pair_transition_0(pair_cond)
    pair_cond += head.pair_transition_1(pair_cond)
    if out is None:
      out = torch.empty((n, n) + tuple(pair_cond.shape[2:]), dtype=pair_cond.dtype, device=pair_cond.device)
    out[r0:r1].copy_(pair_cond)
    del pair_cond
    blocks += 1
  head._paircond_blocks = int(getattr(head, "_paircond_blocks", 0) or 0) + blocks
  return out
