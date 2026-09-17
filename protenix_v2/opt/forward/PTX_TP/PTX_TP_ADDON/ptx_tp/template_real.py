# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ptx_tp.template_real -- REAL (user-supplied) templates in rows mode.

Stock protenix 2.x builds on the host dense [T,N,N,39] distograms + [T,N,N,3] unit vectors + two [T,N,N] masks (Templates.as_protenix_dict).
Here rank 0 emits only PER-TOKEN quantities (aatype, pseudo-beta position + mask, CA position, backbone frame R + mask; a few MB at 31k) and every
rank rebuilds ITS OWN ROWS per 128-row block with the featurizer's own numpy expressions (row-sliced on the first index => element-wise identical
arithmetic => the same values as the dense construction), then moves the block to the device.
Inter-chain template pairs are masked by the TemplateEmbedder's asym multichain mask exactly as stock (monomer template => same-chain blocks only).
Every templated run prints: 'Templates are monomer-only (assembly information NOT supplied)'.
"""
from __future__ import annotations

import numpy as np
import torch

REAL_MARKER = "template_real_rows"
KEYS = ("template_aatype", "template_pb_pos", "template_pb_mask_1d", "template_ca_pos", "template_frame_R", "template_bb_mask_1d")
NOTICE = "Templates are monomer-only (assembly information NOT supplied)"


def per_token_from_templates(aatype, atom_positions, atom_mask, is_ligand=None) -> dict:
    """[T,N] aatype, [T,N,A,3] positions, [T,N,A] mask -> per-token dict (numpy), using the featurizer's own 1-D statements."""
    from protenix.data.template.template_utils import TemplateFeatures, RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX
    aatype = np.asarray(aatype); atom_positions = np.asarray(atom_positions); atom_mask = np.asarray(atom_mask)
    T, N = aatype.shape[:2]
    out = {"template_pb_pos": np.zeros((T, N, 3), np.float32), "template_pb_mask_1d": np.zeros((T, N), np.float32),
           "template_ca_pos": np.zeros((T, N, 3), np.float32), "template_frame_R": np.zeros((T, N, 3, 3), np.float32),
           "template_bb_mask_1d": np.zeros((T, N), np.float32), REAL_MARKER: np.array(1, dtype=np.int64)}
    epsilon = 1e-6
    for i in range(T):
        aat = aatype[i]
        mask = atom_mask[i]
        pos = atom_positions[i] * mask[..., None]
        pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(aat, pos, mask, is_ligand=is_ligand)
        out["template_pb_pos"][i] = pb_pos.astype(np.float32, copy=False)
        out["template_pb_mask_1d"][i] = pb_mask
        backbone_indices = RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[aat, 0]
        c_idx, ca_idx, n_idx = backbone_indices[:, 0], backbone_indices[:, 1], backbone_indices[:, 2]
        res_indices = np.arange(aat.shape[0])
        c_pos = pos[res_indices, c_idx].astype(np.float32, copy=False)
        ca_pos = pos[res_indices, ca_idx].astype(np.float32, copy=False)
        n_pos = pos[res_indices, n_idx].astype(np.float32, copy=False)
        m = (mask[res_indices, c_idx] * mask[res_indices, ca_idx] * mask[res_indices, n_idx]).astype(np.float32)
        v1 = c_pos - ca_pos
        v2 = n_pos - ca_pos
        v1_norm = np.sqrt(np.einsum("ij,ij->i", v1, v1))[:, np.newaxis] + epsilon
        e1 = v1 / v1_norm
        e2 = v2 - np.einsum("ij,ij->i", v2, e1)[:, np.newaxis] * e1
        e2_norm = np.sqrt(np.einsum("ij,ij->i", e2, e2))[:, np.newaxis] + epsilon
        e2 = e2 / e2_norm
        e3 = np.cross(e1, e2)
        out["template_frame_R"][i] = np.stack([e1, e2, e3], axis=-1)
        out["template_ca_pos"][i] = ca_pos
        out["template_bb_mask_1d"][i] = m
    return out


def rows_numpy(d_np: dict, t: int, g0: int, g1: int):
    """Rows g0:g1 of (dgram*pb2d [n,N,39], pb2d [n,N], unit_vector*bb2d [n,N,3], bb2d [n,N]) for template t (numpy fp32, featurizer arithmetic)."""
    from protenix.data.template.template_utils import TemplateFeatures
    from protenix.data.template.template_featurizer import Templates
    config = Templates._DGRAM_CONFIG
    I = slice(g0, g1)
    pb_pos, pb_mask = d_np["template_pb_pos"][t], d_np["template_pb_mask_1d"][t]
    cache_key = (config.min_bin, config.max_bin, config.num_bins)
    if cache_key not in TemplateFeatures._dgram_cache:
        TemplateFeatures.dgram_from_positions(np.zeros((2, 3), np.float32), config=config)
    lower_breaks, upper_breaks = TemplateFeatures._dgram_cache[cache_key]
    pos = pb_pos.astype(np.float32, copy=False)
    diff = pos[I][:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)[..., np.newaxis]
    dgram = ((dist2 > lower_breaks) & (dist2 < upper_breaks)).astype(np.float32)
    pb2d = pb_mask[I][:, None] * pb_mask[None, :]
    dgram = dgram * pb2d[..., None]
    epsilon = 1e-6
    R, ca, m = d_np["template_frame_R"][t], d_np["template_ca_pos"][t], d_np["template_bb_mask_1d"][t]
    dca = ca[np.newaxis, :, :] - ca[I][:, np.newaxis, :]
    uv = np.einsum("ilk,ijl->ijk", R[I], dca)
    uv_norm = np.sqrt(np.einsum("ijk,ijk->ij", uv, uv))[..., np.newaxis] + epsilon
    uv = uv / uv_norm
    bb2d = m[I][:, None] * m[None, :]
    uv = uv * bb2d[..., None]
    return dgram, pb2d, uv, bb2d


_NP_CACHE: dict = {}


def rows_torch(d: dict, t: int, g0: int, g1: int, device, dtype=torch.float32):
    key = id(d.get("template_pb_pos"))
    d_np = _NP_CACHE.get(key)
    if d_np is None:
        _NP_CACHE.clear()
        d_np = {k: (d[k].detach().cpu().numpy() if torch.is_tensor(d[k]) else np.asarray(d[k])) for k in KEYS if k in d}
        _NP_CACHE[key] = d_np
    dg, pb2d, uv, bb2d = rows_numpy(d_np, t, g0, g1)

    def cv(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
    return cv(dg), cv(pb2d), cv(uv), cv(bb2d)


def install_rows_hook(log=print) -> None:
    """Route ptx_tp.template.template_pair_rows through rows_torch when the feature dict carries REAL per-token template keys."""
    import ptx_tp.msa  # noqa: F401  (the msa <-> template modules import each other; msa must be imported first)
    from ptx_tp import template as H2T
    if getattr(H2T.template_pair_rows, "_real_hook", False):
        return
    _orig = H2T.template_pair_rows
    _said = {"n": 0}

    def template_pair_rows(d, template_id, g0, g1):
        if REAL_MARKER in d and "template_distogram" not in d:
            pbm = d["template_pb_mask_1d"][template_id]; bbm = d["template_bb_mask_1d"][template_id]
            if float(pbm.sum()) == 0.0 and float(bbm.sum()) == 0.0:
                return _orig(d, template_id, g0, g1)          # dummy slot -> the compact path's zero rows (identical values)
            if _said["n"] == 0:
                log(NOTICE); _said["n"] = 1
            dev = pbm.device if torch.is_tensor(pbm) else torch.device("cpu")
            return rows_torch(d, template_id, g0, g1, dev)
        return _orig(d, template_id, g0, g1)

    template_pair_rows._real_hook = True
    H2T.template_pair_rows = template_pair_rows


def featurizer_emit_real_rows(templates_obj) -> dict:
    """Templates (protenix featurizer object with REAL coordinates) -> per-token feature dict for rows mode (replaces as_protenix_dict's dense build)."""
    d = dict(templates_obj.as_data_dict())
    is_lig = getattr(templates_obj, "is_ligand", None)
    d.update(per_token_from_templates(templates_obj.aatype, templates_obj.atom_positions, templates_obj.atom_mask, is_ligand=is_lig))
    # compact keys for the 1-D masks (used by ptx_tp.template's rows path; identical products)
    d["template_pseudo_beta_mask_1d"] = d["template_pb_mask_1d"].copy()
    d["template_backbone_frame_mask_1d"] = d["template_bb_mask_1d"].copy()
    d["template_pair_dense_free"] = np.array(1, dtype=np.int64)
    return d
