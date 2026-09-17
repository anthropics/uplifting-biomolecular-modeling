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
"""Distributed summary_confidence / full_data for Protenix v2.

tp_full_data_and_summary(...) -> (summary_confidence: list[dict] per sample, full_data: list[dict])
with IDENTICAL keys, key order and semantics to
    protenix/model/sample_confidence.py  compute_full_data_and_summary (51-230 / 780-828).

  head_out["mode"] == "replicated" (a replicated layout):
      the STOCK compute_full_data_and_summary runs verbatim on every rank.
  head_out["mode"] == "block" (row shards):
      per sample the RowBlockReducer (ptx_tp.blockreduce) is finalized (collective), then rank 0
      assembles the summary dict statement-by-statement like _compute_full_data_and_summary:
      plddt / chain_plddt / has_clash / gpde family via the stock functions on replicated or
      gathered [N, N] fp32 inputs, pTM family from the reducer (finish="exact": stock's
      post-contraction statements on same-shape tensors; finish="rowsum": O(N) partials).
      full_data: the O(N^2) members (token_pair_pae / token_pair_pde / contact_probs) are NOT
      returned as tensors for the JSON dumper; they are written (fp16) to an .npz when npz_path
      is given, together with atom_plddt, chain ids and token/atom maps.  The returned full_data
      dicts keep atom_plddt (+ the small shared keys) so DataDumper's B-factor path works unchanged.

Summary JSON formatting note (runner/dumper.py + protenix/utils/file_io.py save_json): summary
values go through tensor.tolist() -> Python floats -> json.dump(indent=4) with NO rounding, i.e.
float32 values are printed with repr precision (e.g. 0.8532161116600037); byte-identical JSON
therefore requires bitwise-identical float32 results.  (full_data JSON, only written when
need_atom_confidence=True, is rounded to 2 decimals by round_values.)
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from protenix.model import sample_confidence as SC

from ptx_tp.blockreduce import RowBlockReducer  # noqa: F401  (type reference)


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


@torch.no_grad()
def tp_full_data_and_summary(configs, head_out: dict, *, layout,
                             token_asym_id: torch.Tensor, token_has_frame: torch.Tensor,
                             atom_coordinate: torch.Tensor, atom_to_token_idx: torch.Tensor,
                             atom_is_polymer: torch.Tensor, N_recycle: int,
                             contact_rows: Optional[torch.Tensor] = None,
                             contact_probs_full: Optional[torch.Tensor] = None,
                             return_full_data: bool = True,
                             interested_atom_mask: Optional[torch.Tensor] = None,
                             mol_id: Optional[torch.Tensor] = None,
                             elements_one_hot: Optional[torch.Tensor] = None,
                             npz_path: Optional[str] = None):
    """Returns (summary_confidence list, full_data list) on rank 0 (every rank in replicated mode);
    (None, None) on the other ranks.  Collective: every rank must call it."""
    mode = head_out["mode"]
    rank = _rank()

    if mode == "replicated":
        if contact_probs_full is None:
            assert contact_rows is not None
            contact_probs_full = contact_rows
        with torch.autocast(device_type="cuda" if atom_coordinate.is_cuda else "cpu", enabled=False):
            return SC.compute_full_data_and_summary(          # autocasting_disable_decorator(True)(...) in protenix.py
                configs=configs,
                pae_logits=head_out["pae"].float() if head_out["pae"].is_floating_point() else head_out["pae"],
                plddt_logits=head_out["plddt"].float(),
                pde_logits=head_out["pde"].float(),
                contact_probs=contact_probs_full.float(),
                token_asym_id=token_asym_id, token_has_frame=token_has_frame,
                atom_coordinate=atom_coordinate.float(), atom_to_token_idx=atom_to_token_idx,
                atom_is_polymer=atom_is_polymer, N_recycle=N_recycle,
                return_full_data=return_full_data, interested_atom_mask=interested_atom_mask,
                mol_id=mol_id, elements_one_hot=elements_one_hot)

    assert mode == "block", mode
    reducers = head_out["reducers"]
    N_sample = len(reducers)
    plddt_logits_all = head_out["plddt"].float()
    summaries, full_datas, npz_payload = [], [], {}
    for i, red in enumerate(reducers):
        stats = red.finalize(layout.bounds, collect_full=bool(return_full_data or npz_path))   # collective
        if rank != 0:
            continue
        with torch.autocast(device_type="cuda" if atom_coordinate.is_cuda else "cpu", enabled=False):
            s_i, f_i = _assemble_rank0(
                configs, stats, red, plddt_logits=plddt_logits_all[i: i + 1],
                token_asym_id=token_asym_id, token_has_frame=token_has_frame,
                atom_coordinate=atom_coordinate[i: i + 1].float(), atom_to_token_idx=atom_to_token_idx,
                atom_is_polymer=atom_is_polymer, N_recycle=N_recycle, return_full_data=return_full_data,
                interested_atom_mask=interested_atom_mask, mol_id=mol_id, elements_one_hot=elements_one_hot)
        summaries.extend(s_i)
        full_datas.extend(f_i)
        if npz_path:
            for k in ("token_pair_pae_f16", "token_pair_pde_f16"):
                if k in stats:
                    npz_payload.setdefault(k[:-4], []).append(stats[k].numpy())
            if "contact_probs_f16" in stats and "contact_probs" not in npz_payload:
                npz_payload["contact_probs"] = stats["contact_probs_f16"].numpy()
            npz_payload.setdefault("atom_plddt", []).append(f_i[0]["atom_plddt"].float().cpu().numpy() if f_i and "atom_plddt" in f_i[0]
                                                            else SC.logits_to_score(plddt_logits_all[i: i + 1], **SC.get_bin_params(configs.loss.plddt))[0].cpu().numpy())
        del stats
    if rank != 0:
        return None, None
    if npz_path:
        payload = {k: (np.stack(v, 0) if isinstance(v, list) else v) for k, v in npz_payload.items()}
        payload.update({
            "token_asym_id": token_asym_id.long().cpu().numpy(),
            "token_chain_index": reducers[0].chains.asym.cpu().numpy(),        # contiguous 0..C-1 (stock remap)
            "token_has_frame": token_has_frame.bool().cpu().numpy(),
            "atom_to_token_idx": atom_to_token_idx.long().cpu().numpy(),
            "atom_is_polymer": atom_is_polymer.to(torch.int8).cpu().numpy(),
        })
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        np.savez(npz_path, **payload)
        for f in full_datas:
            f["npz_path"] = npz_path
    return summaries, full_datas


def _assemble_rank0(configs, stats: dict, red, *, plddt_logits, token_asym_id, token_has_frame, atom_coordinate,
                    atom_to_token_idx, atom_is_polymer, N_recycle, return_full_data, interested_atom_mask,
                    mol_id, elements_one_hot):
    """Mirror of sample_confidence._compute_full_data_and_summary (96-230) for ONE sample."""
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = torch.zeros_like(token_asym_id).scatter_add(0, atom_to_token_idx, atom_is_ligand)
    token_is_ligand = token_is_ligand > 0  # noqa: F841  (chain_is_ligand handled inside the reducer's ChainIndex)

    full_data = {}
    full_data["atom_plddt"] = SC.logits_to_score(plddt_logits, **SC.get_bin_params(configs.loss.plddt))   # [1, N_atom]

    summary = {}
    summary["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100
    if red.finish == "exact":
        token_pair_pde = stats["token_pair_pde_f32"]        # [1, N, N]  (rank-ordered row concat == stock tensor)
        contact_probs = stats["contact_probs_f32"]          # [N, N]
        summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])
    else:
        summary["gpde"] = stats["gpde"]
    summary["ptm"] = stats["ptm"]
    summary["iptm"] = stats["iptm"]
    if red.finish == "exact":
        summary.update(SC.calculate_chain_based_gpde(token_pair_pde=token_pair_pde, contact_probs=contact_probs,
                                                     asym_id=token_asym_id))
    else:
        summary.update({"chain_gpde": stats["chain_gpde"], "chain_pair_gpde": stats["chain_pair_gpde"]})
    # calculate_chain_based_ptm return order: chain_ptm, chain_iptm, chain_pair_iptm, chain_pair_iptm_global
    summary.update({"chain_ptm": stats["chain_ptm"], "chain_iptm": stats["chain_iptm"],
                    "chain_pair_iptm": stats["chain_pair_iptm"],
                    "chain_pair_iptm_global": stats["chain_pair_iptm_global"]})
    summary.update(SC.calculate_chain_based_plddt(full_data["atom_plddt"], token_asym_id, atom_to_token_idx))
    summary["has_clash"] = SC.calculate_clash(atom_coordinate, token_asym_id, atom_to_token_idx, atom_is_polymer,
                                              configs.metrics.clash.af3_clash_threshold)
    summary["num_recycles"] = torch.tensor(N_recycle, device=atom_coordinate.device)
    summary["disorder"] = torch.zeros_like(summary["ptm"])
    summary["ranking_score"] = (0.8 * summary["iptm"] + 0.2 * summary["ptm"] + 0.5 * summary["disorder"]
                                - 100 * summary["has_clash"])
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        assert len(torch.unique(asym_ids)) == 1
        interested_asym_id = asym_ids[0].item()
        N_chains = token_asym_id.max().long().item() + 1
        pb_ranking_score = summary["chain_pair_iptm_global"][:, interested_asym_id,
                                                             torch.arange(N_chains) != interested_asym_id]
        summary["pb_ranking_score"] = pb_ranking_score[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            vdw_clash = SC.calculate_vdw_clash(pred_coordinate=atom_coordinate, asym_id=token_asym_id, mol_id=mol_id,
                                               is_polymer=atom_is_polymer, atom_token_idx=atom_to_token_idx,
                                               elements_one_hot=elements_one_hot,
                                               threshold=configs.metrics.clash.vdw_clash_threshold)
            flag = vdw_clash[:, interested_asym_id, :].reshape(atom_coordinate.shape[0], -1).max(dim=-1)[0]
            summary["has_vdw_pl_clash"] = flag
            summary["pb_ranking_score_vdw_penalized"] = summary["pb_ranking_score"] - 100 * flag

    summary = SC.break_down_to_per_sample_dict(summary, shared_keys=["num_recycles"])
    if not return_full_data:
        return summary, [{}]
    full_data["token_has_frame"] = token_has_frame.clone()
    full_data["token_asym_id"] = token_asym_id.clone()
    full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
    full_data["atom_is_polymer"] = atom_is_polymer.clone()
    full_data["atom_coordinate"] = atom_coordinate.clone()
    full_data = SC.break_down_to_per_sample_dict(
        full_data, shared_keys=["token_has_frame", "token_asym_id", "atom_to_token_idx", "atom_is_polymer"])
    return summary, full_data


# ----------------------------------------------------------------------------- JSON helpers (tests / drivers)
def summary_json_bytes(summary_sample: dict, indent: Optional[int] = 4) -> bytes:
    """Exactly what runner/dumper.py writes for one summary_confidence sample (save_json, indent=4)."""
    import json
    from protenix.utils.torch_utils import map_values_to_list
    data_json = map_values_to_list(dict(summary_sample).copy() if False else {k: v for k, v in summary_sample.items()})
    return (json.dumps(data_json, indent=indent) if indent is not None else json.dumps(data_json)).encode()


def full_data_json_bytes(full_sample: dict) -> bytes:
    """What the dumper writes for full_data (need_atom_confidence=True): pop coords/is_polymer, round 2 dp, indent=None."""
    import copy
    import json
    from protenix.utils.torch_utils import map_values_to_list, round_values
    d = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in full_sample.items()}
    d.pop("atom_coordinate", None)
    d.pop("atom_is_polymer", None)
    d.pop("npz_path", None)
    d = round_values(d)
    d = map_values_to_list(d)
    return json.dumps(d).encode()
