# Copyright 2026 AlQuraishi Laboratory (original OpenFold3 0.4.1 statements, Apache-2.0)
# OF3 adapter / row-chunked re-implementation: OF3-OFFLOAD add-on helper, 2026.
# The pTM-family row-block contraction is delegated to BLOCKREDUCE v0
# (of3o_blockreduce.py, beside this module) -- imported, not copied.
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
"""
of3o_confidence.py (v0.2) -- row-chunked, drop-in re-implementation of OpenFold3
0.4.1 inference confidence scoring, built as an OF3 adapter on the shared
BLOCKREDUCE v0 primitive (``of3o_blockreduce.RowBlockReducer``).

Mirrors ``openfold3.core.metrics.aggregate_confidence_ranking.get_confidence_scores``
(and everything it calls on the inference path) while never materialising more
than O(rows * N * bins) pair-logit temporaries on the compute device, nor the
[S, N_atom, N_atom(,3)] tensors of stock ``get_token_frame_atoms`` /
``compute_has_clash``.

Entry point
-----------
    get_confidence_scores_chunked(batch, outputs, config, compute_per_sample=False,
        rows=256, logits_device=None, *, frame_rows=128, clash_rows=4096,
        pair_output_device=None, tm_backend="auto", tm_finish="rowsum",
        gpde_mode="exact", samples_per_pass=None)

Returns a dict with exactly stock's keys / dtypes / shapes (plddt, pde,
[pde_probs], gpde, [contact_probs], pae, [pae_probs], iptm, ptm, disorder,
has_clash, sample_ranking_score, chain_pair_iptm, bespoke_iptm, chain_ptm), all
tensors on ``logits_device`` (default ``outputs["plddt_logits"].device``), i.e.
where stock would have put them with device-resident logits.  pae / pde /
distogram logits may live anywhere (e.g. pinned CPU); each row block
``[..., i0:i1, :, :]`` is moved to ``logits_device`` inside ONE merged pass.

DIVISION OF LABOUR
------------------
BLOCKREDUCE v0 (of3o_blockreduce; imported):
    ChainIndex (contiguous chain remap, masks), RowBlockReducer: enumeration of
    the TM contexts FULL / CHAIN a / PAIR (a,b), per-row-block contraction
    T_d[i, cols(d)] = sum_b softmax(pae_logits[i,j,:])_b * w_d[b], exact
    gather of each context's [N_d, N_d] matrix (multi-rank capable) and the
    finalize() driver; both finishing regimes ("rowsum": per-row partial sums,
    O(N * n_ctx) state; "exact": gathered [N_d, N_d] matrices).
This file (OF3-OFFLOAD):
    * _OF3RowBlockReducer(RowBlockReducer): injects OF3-exact bin weights
      (OF3 computes d0 with fp32 tensor ops; BLOCKREDUCE uses Python doubles ->
      1-ulp different weights for most N_d) and overrides the two finishing
      hooks with OF3's verbatim compute_ptm tail
      (sum(-1)/num_tokens ; masked pair-sum / count.clamp_min(eps) ;
       masked_fill(~has_frame, 0).max(-1)).
    * everything that is an OF3 statement rather than a contraction: pLDDT,
      PAE / PDE expectations (OF3: sum(p*centers); BLOCKREDUCE's p@centers is
      not bitwise-equal and is not used for outputs), distogram contact
      probabilities, gPDE, has_frame (row-chunked get_token_frame_atoms),
      has_clash (row-chunked cdist), disorder (stock, CPU/biotite),
      chain_has_frame / chain_is_ligand / bespoke_iptm bookkeeping (OF3 rules
      differ from the reference adapter's), sample_ranking_score, OF3 key naming, and the
      batch / per-sample iteration wrapper.
    * a native fallback for the TM family (_TMSet; same numerics) used when
      BLOCKREDUCE is not importable or token_mask contains zeros (BLOCKREDUCE's
      FULL context has no token mask).

EXACTNESS NOTES
---------------
Notation: S samples, N tokens, b bins, block = rows i0:i1 of dim -3.
(E1) Bitwise-equal to stock by construction (identical torch statements on row
     sub-blocks; every reduction is per output element with unchanged extent):
       plddt; pde, pae (softmax over b + 64-term expectation per (i,j));
       pde_probs / pae_probs / contact_probs; ptm, iptm, chain_ptm[c],
       chain_pair_iptm[(a,b)] (per-row T sums; full tm_i vector re-assembled,
       same masked_fill + max); bespoke_iptm; disorder (stock fn); has_frame
       (distance rows + per-row topk; tie-break among exactly equal distances is
       a deterministic function of row content on CPU, assumed per-slice on
       CUDA); sample_ranking_score;
       gpde with gpde_mode="exact" (DEFAULT): the stock statement
         sum(contact*pde, dim=[-2,-1]) / (sum(contact, dim=[-2,-1]) + eps)
       is applied to the re-assembled full [S,N,N] pde and [Sd,N,N] contact
       matrices, i.e. on tensors of stock's shape -> stock's reduction order.
       Cost: contact matrix Sd*N^2*4 B + transient product S*N^2*4 B.
     Assumption: ATen softmax(dim=-1) / sum(dim=-1) / topk(dim=-1) give per-row
     results independent of the number of leading rows (true for CPU kernels --
     checked bit-for-bit on the CPU kernels at rows in {7,33,64,200} --
     and for the standard CUDA kernels at b=64, M<32768; confirm once on GPU).
     BLOCKREDUCE applies softmax to the whole row block and then selects the
     context columns, stock selects then softmaxes: same per-(i,j) op.
(E2) gpde_mode="stream": O(rows*N) memory; the two stock torch.sum(dim=[-2,-1])
     statements are applied per row block and the [S] partials accumulated
     sequentially in fp32 (num = num + part_k) -> reduction order differs from
     stock; observed |diff| <= 4e-6 abs (2e-7 rel).  Nothing consumes gpde.
(E3) has_clash: integer clash counting is exact for identical distances, but
     per-block torch.cdist (mm-based Euclidean path, as stock) may round
     differently from full-matrix cdist; the 0/1 output can differ only if an
     inter-chain atom pair sits within float rounding of 1.1 A AND the count is
     exactly at the >100 / >50% boundary.  clash_rows=None -> stock function.
(E4) tm_finish: OF3's stock post-contraction statements are per-row reductions
     (ptm_ij.sum(-1) / n ; masked row sum / count ; max over rows), so BOTH
     BLOCKREDUCE regimes reproduce them bitwise here: "rowsum" (DEFAULT; per-row
     sums taken block-wise with the same per-row torch.sum, divided by the same
     fp32 count, max over framed rows; O(N * n_ctx) state) and "exact" (gathered
     [N_d, N_d] matrices + the OF3 finishing overrides; ~20 N^2 B per in-flight
     sample).  (For the reference adapter, whose single-device statements reduce whole
     matrices, "rowsum" is TIER-2 -- not so for OF3.)  Checked bit-for-bit for
     both regimes by the CPU test.
(E5) fp32 logits assumed (BLOCKREDUCE casts logits to fp32; stock would compute
     in the logits dtype).

MEMORY MODEL (compute device, fp32, per batch element; S' = samples_per_pass)
------------------------------------------------------------------------------
  merged row pass, per block:  ~ (3*S' + 2*Sd) * rows * N * b * 4 B
      (pae block + softmax + p*w product for the FULL context, pde block +
       softmax, distogram block + softmax; CHAIN/PAIR contexts are subsets)
  BLOCKREDUCE state:           tm_finish="rowsum" (default): O(N * (C + n_ctx))
      floats per in-flight sample (negligible).  tm_finish="exact": ~5 N^2 * 4 B
      per in-flight sample (T pieces ~3 N^2 + internal fp32 pde/contact rows
      2 N^2) = S' * 20 N^2 B, plus one transient [N_d, N_d] gather (<= 4 N^2 B).
      tm_backend="native": O(S' * N * n_ctx).
  frame mask:                  ~ 4 * S * frame_rows * N_atom * 3 * 4 B
  has_clash:                   ~ 3 * clash_rows * n_atoms(chain_j) * 4 B
  outputs:                     pae + pde = 2 * S * N^2 * 4 B on pair_output_device
                               (+ contact Sd*N^2*4 B and transient S*N^2*4 B for
                                gpde_mode="exact")
  The stock scorer materialises >= 3*S*N^2*b*4 B + S*N_atom^2*16 B instead.
"""

from __future__ import annotations

import logging
import math
from functools import partial

import torch

from openfold3.core.metrics.confidence import get_bin_centers, probs_to_expected_error
from openfold3.core.metrics.rasa import compute_disorder
from openfold3.core.metrics.sample_ranking import compute_has_clash
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_atom_index_offset,
)
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map

try:  # BLOCKREDUCE v0: of3o_blockreduce.py beside this module (the hook directory on PYTHONPATH)
    from of3o_blockreduce import CHAIN as _BR_CHAIN
    from of3o_blockreduce import FULL as _BR_FULL
    from of3o_blockreduce import PAIR as _BR_PAIR
    from of3o_blockreduce import ChainIndex, RowBlockReducer

    HAVE_BLOCKREDUCE = True
except ImportError:  # pragma: no cover - exercised when of3o_blockreduce is absent
    HAVE_BLOCKREDUCE = False
    RowBlockReducer = object  # type: ignore[assignment,misc]

# FAIL-CLOSED CARRY (this tree's port): the TM backend as run per element and every native fallback, read by the package through
# of3_offload.census() -> the exit tally (`offload.conf_census` of the run record); a `big` row with native_fallback_elements > 0 is PARTIAL.
CENSUS = {"backend_runs": {}, "native_fallback_elements": 0, "auto_resolved": {}, "have_blockreduce": HAVE_BLOCKREDUCE}

logger = logging.getLogger(__name__)

__all__ = [
    "get_confidence_scores_chunked",
    "HAVE_BLOCKREDUCE",
    "valid_frame_mask_chunked",
    "compute_has_clash_chunked",
]
__version__ = "0.2"


# =============================================================================
# helpers
# =============================================================================
def _row_blocks(n: int, rows: int):
    rows = max(1, int(rows))
    for i0 in range(0, n, rows):
        yield i0, min(n, i0 + rows)


def _to_dev(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device == device:
        return t
    return t.to(device=device, non_blocking=True)


def _of3_tm_preamble(n_d: int, device, dtype, bin_min, bin_max, no_bins):
    """OF3 compute_ptm preamble, verbatim, for a context of size n_d = mask_i.sum()."""
    num_tokens_considered = torch.tensor(n_d, device=device).clamp_min(1).to(dtype)
    clipped = torch.maximum(
        num_tokens_considered, torch.tensor(19.0, device=device, dtype=dtype)
    )
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
    bin_centers = get_bin_centers(bin_min, bin_max, no_bins, device, dtype)
    bin_weight = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    return num_tokens_considered, bin_weight


def _of3_finish_ptm(tm_num_rows: torch.Tensor, num_tokens_considered, has_frame):
    """OF3 compute_ptm tail (interface=False): rows = ptm_ij.sum(-1) already."""
    tm_i = tm_num_rows / num_tokens_considered
    tm_i = tm_i.masked_fill(~has_frame, 0.0)
    return tm_i.max(dim=-1).values


# =============================================================================
# native TM-family fallback (used when BLOCKREDUCE is absent / token_mask has 0s)
# =============================================================================
class _TMSet:
    """One stock `compute_ptm` invocation (set D = mask_i) evaluated block-wise."""

    def __init__(self, mask_i, asym_id, has_frame, device, dtype, bin_min, bin_max,
                 no_bins, want_ptm, want_iptm):
        mask_i = mask_i.to(device=device, dtype=torch.bool)
        if asym_id is not None:
            asym_id = asym_id[mask_i].to(device=device)
        self.mask, self.mask_cpu = mask_i, mask_i.to("cpu")
        self.m = int(self.mask_cpu.sum())
        self.ntc, self.bin_weight = _of3_tm_preamble(self.m, device, dtype, bin_min,
                                                     bin_max, no_bins)
        self.asym_sel = asym_id
        self.has_frame_sel = has_frame[:, mask_i].bool()
        self.want_ptm, self.want_iptm = want_ptm, want_iptm
        self.tm = self.tmi = None
        self.pos = 0

    def consume(self, blk, i0, i1, eps):
        n_sel = int(self.mask_cpu[i0:i1].sum())
        if n_sel == 0:
            return
        sel = self.mask[i0:i1]
        logits = blk[:, sel, ...]  # stock: logits[:, mask_i, ...]  (rows p0:p1)
        logits = logits[..., self.mask, :]  # stock: logits[..., mask_i, :]
        probs = torch.softmax(logits, dim=-1)
        del logits
        ptm_ij = torch.sum(probs * self.bin_weight, dim=-1)
        del probs
        p0, p1 = self.pos, self.pos + n_sel
        self.pos = p1
        if self.want_iptm:
            asym_id = self.asym_sel
            pair_mask = asym_id[p0:p1].unsqueeze(-1) != asym_id.unsqueeze(-2)
            tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
            if self.tmi is None:
                self.tmi = torch.empty((tm_i.shape[0], self.m), dtype=tm_i.dtype,
                                       device=tm_i.device)
            self.tmi[:, p0:p1] = tm_i
        if self.want_ptm:
            rs = ptm_ij.sum(dim=-1)
            if self.tm is None:
                self.tm = torch.empty((rs.shape[0], self.m), dtype=rs.dtype, device=rs.device)
            self.tm[:, p0:p1] = rs

    def result_ptm(self):
        return _of3_finish_ptm(self.tm, self.ntc, self.has_frame_sel)

    def result_iptm(self):
        tm_i = self.tmi.masked_fill(~self.has_frame_sel, 0.0)
        return tm_i.max(dim=-1).values


class _NativeTM:
    """TM family for a group of samples via _TMSet (all contexts, one row pass)."""

    def __init__(self, batch, has_frame, device, dtype, bin_kw, want_pair, want_chain):
        mk = dict(device=device, dtype=dtype, **bin_kw)
        token_mask, asym_id = batch["token_mask"], batch["asym_id"]
        self.full = _TMSet(token_mask, asym_id, has_frame, want_ptm=True, want_iptm=True, **mk)
        self.sets = [self.full]
        self.pair_sets, self.chain_sets = {}, []
        token_mask_b = batch["token_mask"].bool()
        if want_pair:
            asym_id_l = batch["asym_id"].long()
            self.unique_chains = torch.unique(asym_id_l).tolist()
            cm = [(asym_id_l == aid) & token_mask_b for aid in self.unique_chains]
            for i in range(len(cm)):
                for j in range(i + 1, len(cm)):
                    s = _TMSet(cm[i] | cm[j], asym_id_l, has_frame, want_ptm=False,
                               want_iptm=True, **mk)
                    self.pair_sets[(i, j)] = s
                    self.sets.append(s)
        if want_chain:
            for aid in batch["asym_id"].unique():
                s = _TMSet(token_mask_b & (batch["asym_id"] == aid), None, has_frame,
                           want_ptm=True, want_iptm=False, **mk)
                self.chain_sets.append(s)
                self.sets.append(s)

    def consume(self, i0, i1, pae_blk, pde_blk, contact_rows, eps):
        for s in self.sets:
            s.consume(pae_blk, i0, i1, eps)

    def finalize(self, num_samples, device, dtype):
        out = {"ptm": self.full.result_ptm(), "iptm": self.full.result_iptm()}
        if self.pair_sets or hasattr(self, "unique_chains"):
            C = len(self.unique_chains)
            cp = torch.zeros((num_samples, C, C), device=device, dtype=dtype)
            for (i, j), s in self.pair_sets.items():
                v = s.result_iptm()
                cp[:, i, j] = v
                cp[:, j, i] = v
            out["chain_pair"] = cp
        out["chain_ptm_list"] = [s.result_ptm() for s in self.chain_sets]
        return out


# =============================================================================
# BLOCKREDUCE-backed TM family (default)
# =============================================================================
class _OF3RowBlockReducer(RowBlockReducer):  # type: ignore[misc]
    """BLOCKREDUCE RowBlockReducer with (i) OF3-exact per-context bin weights and
    (ii) OF3's verbatim compute_ptm finishing statements."""

    def __init__(self, chains, r0, r1, device, *, of3_bin_kw, dtype=torch.float32, **kw):
        b = (of3_bin_kw["bin_min"], of3_bin_kw["bin_max"], of3_bin_kw["no_bins"])
        super().__init__(chains, r0, r1, device, pae_bins=b, pde_bins=b, **kw)
        self._ntc = {}
        for k, ctx in enumerate(self.ctxs):  # replace the reference adapter's w_d by OF3's
            ntc, w = _of3_tm_preamble(ctx.N_d, self.device, dtype, *b)
            ctx.w = w
            self._ntc[ctx.N_d] = ntc

    # T: [1, N_d, N_d] == stock ptm_ij for this context; has_frame: [N_d] bool
    def _finish_ptm_T(self, T, has_frame):
        num_tokens_considered = self._ntc[T.shape[-1]]
        return _of3_finish_ptm(T.sum(dim=-1), num_tokens_considered, has_frame)

    def _finish_iptm_T(self, T, has_frame, asym):
        asym_id = asym
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (T * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(self.eps)
        tm_i = tm_i.masked_fill(~has_frame, 0.0)
        return tm_i.max(dim=-1).values


class _BlockreduceTM:
    """TM family for a group of samples: one _OF3RowBlockReducer per sample."""

    def __init__(self, batch, has_frame, device, dtype, bin_kw, finish, eps, sample_ids):
        asym_id = _to_dev(batch["asym_id"], device)
        is_lig = _to_dev(batch["is_ligand"], device).bool()
        n = asym_id.shape[-1]
        self.reducers = []
        for s in sample_ids:
            ch = ChainIndex(asym_id, _to_dev(has_frame[s], device), is_lig)
            self.reducers.append(_OF3RowBlockReducer(
                ch, 0, n, device, of3_bin_kw=bin_kw, dtype=dtype, finish=finish, eps=eps,
                keep_value_rows=False))
        self.n = n
        self.unique_chains = torch.unique(batch["asym_id"].long()).tolist()

    def consume(self, i0, i1, pae_blk, pde_blk, contact_rows, eps):
        c = contact_rows.reshape(-1, *contact_rows.shape[-2:])
        for k, red in enumerate(self.reducers):
            red.consume(i0, i1, pae_blk[k], pde_blk[k], c[min(k, c.shape[0] - 1)])

    def finalize(self, num_samples, device, dtype):
        stats = [red.finalize([(0, self.n)], collect_full=False) for red in self.reducers]
        self.reducers = []
        out = {
            "ptm": torch.cat([st["ptm"] for st in stats]),
            "iptm": torch.cat([st["iptm"] for st in stats]),
            "chain_pair": torch.cat([st["chain_pair_iptm"] for st in stats]).to(dtype),
        }
        C = len(self.unique_chains)
        cptm = torch.cat([st["chain_ptm"] for st in stats]).to(dtype)  # [S', C]
        out["chain_ptm_list"] = [cptm[:, a] for a in range(C)]
        return out


# =============================================================================
# OF3 bespoke-ipTM bookkeeping (verbatim tail of stock compute_chain_pair_iptm)
# =============================================================================
def _bespoke_from_chain_pair(batch, chain_pair_iptm, unique_chains, has_frame, device):
    token_mask = _to_dev(batch["token_mask"], device).bool()
    asym_id = _to_dev(batch["asym_id"], device).long()
    is_ligand = _to_dev(batch["is_ligand"], device).bool()
    has_frame = _to_dev(has_frame, device)
    dtype = chain_pair_iptm.dtype
    num_chains = len(unique_chains)
    num_samples = chain_pair_iptm.shape[0]
    chain_masks = [(asym_id == aid) & token_mask for aid in unique_chains]

    chain_has_frame = torch.tensor(
        [(chain_mask & has_frame).any().item() for chain_mask in chain_masks],
        device=device,
    )
    chain_is_ligand = torch.tensor(
        [
            (chain_mask & is_ligand).sum() * 2 >= chain_mask.sum()
            for chain_mask in chain_masks
        ],
        device=device,
    )

    chain_mean_iptm = torch.zeros((num_samples, num_chains), device=device, dtype=dtype)
    for i in range(num_chains):
        values = [
            chain_pair_iptm[:, i, j]
            for j in range(num_chains)
            if j != i and chain_has_frame[i]
        ]
        values.extend(
            [
                chain_pair_iptm[:, j, i]
                for j in range(num_chains)
                if j != i and chain_has_frame[j]
            ]
        )
        if values:
            chain_mean_iptm[:, i] = torch.stack(values, dim=-1).mean(dim=-1)
        else:
            chain_mean_iptm[:, i] = 0.0

    bespoke_iptm = torch.zeros_like(chain_pair_iptm)
    for i in range(num_chains):
        for j in range(num_chains):
            if i == j:
                continue
            if chain_is_ligand[i]:
                bespoke_iptm[:, i, j] = chain_mean_iptm[:, i]
            elif chain_is_ligand[j]:
                bespoke_iptm[:, i, j] = chain_mean_iptm[:, j]
            else:
                bespoke_iptm[:, i, j] = 0.5 * (
                    chain_mean_iptm[:, i] + chain_mean_iptm[:, j]
                )

    chain_pair_iptm_map, bespoke_iptm_map = {}, {}
    for i in range(num_chains):
        for j in range(num_chains):
            if i >= j:
                continue
            key = f"({unique_chains[i]},{unique_chains[j]})"
            chain_pair_iptm_map[key] = chain_pair_iptm[:, i, j]
            bespoke_iptm_map[key] = bespoke_iptm[:, i, j]
    return {"chain_pair_iptm": chain_pair_iptm_map, "bespoke_iptm": bespoke_iptm_map}


# =============================================================================
# the merged row pass: pde, pae, contact / gpde, TM family
# =============================================================================
def pair_row_pass(batch, outputs, cc, *, rows, device, pair_output_device, has_frame,
                  tm_backend, tm_finish, gpde_mode, samples_per_pass, want_tm=True):
    pae_logits = outputs.get("pae_logits") if want_tm else None
    pde_logits = outputs["pde_logits"]
    dist_logits = outputs["distogram_logits"]
    n = pde_logits.shape[-3]
    lead = pde_logits.shape[:-3]  # (S,)
    S = int(math.prod(lead)) if len(lead) else 1
    dtype = pde_logits.dtype
    ptm_kw = dict(cc.ptm)
    bin_kw = {k: ptm_kw[k] for k in ("bin_min", "bin_max", "no_bins")}
    eps_tm = ptm_kw.get("eps", 1e-8)
    dg = dict(cc.distogram)
    eps_gpde = dg.get("eps", 1e-8)
    want_contact_full = gpde_mode == "exact" or dg.get("return_contact_probs", False)

    # stock compute_global_predicted_distance_error bin bookkeeping
    distogram_bin_ends = torch.linspace(dg["bin_min"], dg["bin_max"], dg["no_bins"] + 1,
                                        device=device)[1:]
    distogram_bins_8A = distogram_bin_ends <= 8.0

    pde = pae = pde_probs = pae_probs = contact_full = None
    gp_num = gp_den = None
    spp = S if not samples_per_pass else max(1, min(S, int(samples_per_pass)))
    groups = [list(range(g, min(S, g + spp))) for g in range(0, S, spp)]
    pde_l = pde_logits.reshape(S, n, n, -1)
    pae_l = pae_logits.reshape(S, n, n, -1) if pae_logits is not None else None
    tm_results = []

    for gi, sids in enumerate(groups):
        sl = slice(sids[0], sids[-1] + 1)
        tm = None
        if pae_l is not None:
            if tm_backend == "blockreduce":
                tm = _BlockreduceTM(batch, has_frame, device, dtype, bin_kw, tm_finish,
                                    eps_tm, sids)
            else:
                tm = _NativeTM(batch, _to_dev(has_frame[sl], device), device, dtype, bin_kw,
                               want_pair=bool(cc.sample_ranking.chain_pair_iptm.enabled),
                               want_chain=bool(cc.sample_ranking.chain_ptm.enabled))
        for i0, i1 in _row_blocks(n, rows):
            # --- distogram contact rows (stock statements) ------------------------
            dblk = _to_dev(dist_logits[..., i0:i1, :, :], device)
            probs = torch.softmax(dblk, dim=-1)
            contact_probs = torch.sum(probs[..., distogram_bins_8A], dim=-1)  # [Sd?, r, N]
            del dblk, probs
            if gi == 0 and want_contact_full:
                if contact_full is None:
                    contact_full = torch.empty((*contact_probs.shape[:-2], n, n),
                                               dtype=contact_probs.dtype, device=device)
                contact_full[..., i0:i1, :] = contact_probs
            # --- PDE rows (stock statements) --------------------------------------
            pblk = _to_dev(pde_l[sl, i0:i1], device)
            p = torch.softmax(pblk, dim=-1)
            e = probs_to_expected_error(p, **cc.pde)  # [S', r, N]
            if pde is None:
                pde = torch.empty((S, n, n), dtype=e.dtype, device=pair_output_device)
            pde[sl, i0:i1, :] = e
            if cc.pde.return_probs:
                if pde_probs is None:
                    pde_probs = torch.empty((S, n, n, p.shape[-1]), dtype=p.dtype,
                                            device=pair_output_device)
                pde_probs[sl, i0:i1] = p
            del p
            if gpde_mode == "stream":  # (E2) per-block partials of the two stock sums
                part_num = torch.sum(contact_probs * e, dim=[-2, -1])  # [S']
                part_den = torch.sum(contact_probs, dim=[-2, -1])  # contact lead shape
                if gp_num is None:
                    gp_num = torch.zeros((S,), dtype=part_num.dtype, device=device)
                gp_num[sl] += part_num  # sequential fp32 accumulation, row order
                if gi == 0:  # denominator is sample independent
                    gp_den = part_den if gp_den is None else gp_den + part_den
            del e
            # --- PAE rows (stock statements) + TM family ---------------------------
            if pae_l is not None:
                ablk = _to_dev(pae_l[sl, i0:i1], device)  # [S', r, N, b]
                pae_p = torch.softmax(ablk, dim=-1)
                ea = probs_to_expected_error(pae_p, **cc.pae)
                if pae is None:
                    pae = torch.empty((S, n, n), dtype=ea.dtype, device=pair_output_device)
                pae[sl, i0:i1, :] = ea
                if cc.pae.return_probs:
                    if pae_probs is None:
                        pae_probs = torch.empty((S, n, n, pae_p.shape[-1]), dtype=pae_p.dtype,
                                                device=pair_output_device)
                    pae_probs[sl, i0:i1] = pae_p
                del pae_p, ea
                tm.consume(i0, i1, ablk, pblk, contact_probs, eps_tm)
                del ablk
            del pblk, contact_probs
        if tm is not None:
            tm_results.append(tm.finalize(len(sids), device, dtype))

    out = {"pde": pde.reshape(*lead, n, n), "pde_probs": pde_probs,
           "pae_probs": pae_probs, "contact_full": contact_full}
    if pae is not None:
        out["pae"] = pae.reshape(*lead, n, n)
    # --- gPDE ------------------------------------------------------------------
    if gpde_mode == "exact":  # stock statement on stock-shaped full matrices (E1)
        pde_dev = _to_dev(out["pde"], device)
        out["gpde"] = torch.sum(contact_full * pde_dev, dim=[-2, -1]) / (
            torch.sum(contact_full, dim=[-2, -1]) + eps_gpde
        )
    else:
        out["gpde"] = (gp_num / (gp_den + eps_gpde)).reshape(lead)
    if dg.get("return_contact_probs", False):
        out["contact_probs"] = contact_full
    # --- TM family: concat sample groups ---------------------------------------
    if tm_results:
        out["ptm"] = torch.cat([r["ptm"] for r in tm_results])
        out["iptm"] = torch.cat([r["iptm"] for r in tm_results])
        if "chain_pair" in tm_results[0]:
            out["chain_pair"] = torch.cat([r["chain_pair"] for r in tm_results])
        ncs = len(tm_results[0]["chain_ptm_list"])
        out["chain_ptm_list"] = [torch.cat([r["chain_ptm_list"][a] for r in tm_results])
                                 for a in range(ncs)]
    return out


# =============================================================================
# has_frame: stock get_token_frame_atoms with the [N_atom, N_atom] distance /
# top-k replaced by a row-chunked evaluation of only the rows stock gathers.
# =============================================================================
def valid_frame_mask_chunked(batch, x, atom_mask, frame_rows=128, angle_threshold=25.0,
                             eps=1e-8, inf=1e9):
    n_token = batch["token_mask"].shape[-1]
    atom_asym_id = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["asym_id"],
    )
    start_atom_index = batch["start_atom_index"].long()
    start_atom_index = start_atom_index.expand(*x.shape[:-2], start_atom_index.shape[-1])
    sai_rows = batch["start_atom_index"].long().reshape(-1, n_token)
    if sai_rows.shape[0] > 1:
        assert bool((sai_rows == sai_rows[:1]).all()), "start_atom_index varies over leading dims"
    sai_rows = sai_rows[0]

    a_index = torch.empty(start_atom_index.shape, dtype=torch.long, device=x.device)
    c_index = torch.empty(start_atom_index.shape, dtype=torch.long, device=x.device)
    for t0, t1 in _row_blocks(n_token, frame_rows):
        ridx = sai_rows[t0:t1]
        pair_mask = atom_mask[..., ridx].unsqueeze(-1) * atom_mask[..., None, :]
        atom_asym_id_mask = atom_asym_id[..., ridx].unsqueeze(-1) == atom_asym_id[..., None, :]
        pair_mask = pair_mask * atom_asym_id_mask
        xr = x[..., ridx, :]
        d = torch.sum(eps + (xr[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5
        d = d * pair_mask + inf * (1 - pair_mask)
        _, closest_atom_index = torch.topk(d, k=3, dim=-1, largest=False)
        a_index[..., t0:t1] = closest_atom_index[..., 1]
        c_index[..., t0:t1] = closest_atom_index[..., 2]
        del pair_mask, atom_asym_id_mask, xr, d, closest_atom_index

    # ---------------- remainder verbatim from stock get_token_frame_atoms --------
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (1 - batch["is_atomized"])
    restype = batch["restype"]
    n_off, n_msk = get_token_atom_index_offset(atom_name="N", restype=restype)
    ca_off, ca_msk = get_token_atom_index_offset(atom_name="CA", restype=restype)
    c_off, c_msk = get_token_atom_index_offset(atom_name="C", restype=restype)
    c3p_off, c3p_msk = get_token_atom_index_offset(atom_name="C3'", restype=restype)
    c1p_off, c1p_msk = get_token_atom_index_offset(atom_name="C1'", restype=restype)
    c4p_off, c4p_msk = get_token_atom_index_offset(atom_name="C4'", restype=restype)
    frame_atoms = {
        "a": {
            "index": (a_index * batch["is_atomized"]
                      + (start_atom_index + n_off) * is_standard_protein
                      + (start_atom_index + c3p_off) * is_standard_nucleotide),
            "token_atom_mask": (batch["is_atomized"] + n_msk * is_standard_protein
                                + c3p_msk * is_standard_nucleotide),
        },
        "b": {
            "index": (start_atom_index * batch["is_atomized"]
                      + (start_atom_index + ca_off) * is_standard_protein
                      + (start_atom_index + c1p_off) * is_standard_nucleotide),
            "token_atom_mask": (batch["is_atomized"] + ca_msk * is_standard_protein
                                + c1p_msk * is_standard_nucleotide),
        },
        "c": {
            "index": (c_index * batch["is_atomized"]
                      + (start_atom_index + c_off) * is_standard_protein
                      + (start_atom_index + c4p_off) * is_standard_nucleotide),
            "token_atom_mask": (batch["is_atomized"] + c_msk * is_standard_protein
                                + c4p_msk * is_standard_nucleotide),
        },
    }
    for key in frame_atoms:
        idx = frame_atoms[key]["index"]
        frame_atoms[key].update(
            {
                "atom_positions": torch.gather(
                    x, dim=-2,
                    index=idx.unsqueeze(-1).expand(*(x.shape[:-2] + (idx.shape[-1], 3))).long(),
                ),
                "asym_id": torch.gather(
                    atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]), dim=-1,
                    index=idx.long(),
                ),
                "atom_mask": torch.gather(
                    atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]), dim=-1,
                    index=idx.long(),
                )
                * batch["token_mask"]
                * frame_atoms[key]["token_atom_mask"],
            }
        )
    u = frame_atoms["a"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    v = frame_atoms["c"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    uv = torch.einsum("...i,...i->...", u, v)
    u_norm = (eps + torch.sum(u**2, dim=-1)) ** 0.5
    v_norm = (eps + torch.sum(v**2, dim=-1)) ** 0.5
    cos_angle = uv / (u_norm * v_norm)
    cos_angle_min_bound = math.cos((180 - angle_threshold) * math.pi / 180)
    cos_angle_max_bound = math.cos(angle_threshold * math.pi / 180)
    valid_frame_mask_angle = (cos_angle < cos_angle_max_bound) * (cos_angle > cos_angle_min_bound)
    valid_frame_mask_angle = (
        valid_frame_mask_angle * batch["is_atomized"]
        + torch.ones_like(valid_frame_mask_angle) * (1 - batch["is_atomized"])
    ) * batch["token_mask"]
    valid_frame_mask_atom = (frame_atoms["a"]["atom_mask"] * frame_atoms["b"]["atom_mask"]
                             * frame_atoms["c"]["atom_mask"])
    valid_frame_mask_asym_id = (
        frame_atoms["a"]["asym_id"] == frame_atoms["b"]["asym_id"]
    ) * (frame_atoms["b"]["asym_id"] == frame_atoms["c"]["asym_id"])
    valid_frame_mask = valid_frame_mask_angle * valid_frame_mask_atom * valid_frame_mask_asym_id
    phi = (frame_atoms["a"]["atom_positions"], frame_atoms["b"]["atom_positions"],
           frame_atoms["c"]["atom_positions"])
    return phi, valid_frame_mask


# =============================================================================
# has_clash: stock loops, cdist evaluated clash_rows atom rows at a time (E3)
# =============================================================================
def compute_has_clash_chunked(asym_id, atom_positions_predicted, atom_mask, is_polymer,
                              threshold=1.1, violation_abs=100, violation_frac=0.5,
                              clash_rows=4096):
    device = atom_positions_predicted.device
    dtype = atom_positions_predicted.dtype
    unique_chains = torch.unique(asym_id).tolist()
    num_samples = atom_positions_predicted.size(0)
    polymer_chains = list(
        filter(lambda aid: ((asym_id != aid) | is_polymer).all(), unique_chains)
    )
    num_chains = len(polymer_chains)
    chain_masks = [(asym_id == aid) & atom_mask for aid in polymer_chains]
    has_clash = torch.zeros(num_samples, dtype=dtype, device=device)
    for s in range(num_samples):
        clashing = False
        for i in range(num_chains):
            ni = chain_masks[i].sum()
            if ni == 0:
                continue
            for j in range(i + 1, num_chains):
                nj = chain_masks[j].sum()
                if nj == 0:
                    continue
                chain_i = atom_positions_predicted[s, chain_masks[i], :]
                chain_j = atom_positions_predicted[s, chain_masks[j], :]
                num_clashes = 0
                for r0, r1 in _row_blocks(chain_i.shape[0], clash_rows):
                    distance = torch.cdist(chain_i[r0:r1], chain_j, p=2)
                    num_clashes += (distance < threshold).sum().item()
                    del distance
                if (num_clashes > violation_abs) or (
                    (num_clashes / min(ni, nj)) > violation_frac
                ):
                    has_clash[s] = 1.0
                    clashing = True
                    break
            if clashing:
                break
    return has_clash


# =============================================================================
# per-batch-element driver  (mirrors stock _get_confidence_scores)
# =============================================================================
def _get_confidence_scores_chunked(batch, outputs, config, rows, device, frame_rows,
                                   clash_rows, pair_output_device, tm_backend, tm_finish,
                                   gpde_mode, samples_per_pass):
    cc = config.confidence
    confidence_scores = {}
    pae_enabled = bool(config.architecture.heads.pae.enabled)

    # pLDDT: stock statement (O(S * N_atom * 50); not chunked)
    plddt_logits = _to_dev(outputs["plddt_logits"], device)
    confidence_scores["plddt"] = (
        probs_to_expected_error(torch.softmax(plddt_logits, dim=-1), **cc.plddt) * 100.0
    )

    valid_frame_mask = None
    if pae_enabled:
        _, valid_frame_mask = valid_frame_mask_chunked(
            batch=batch, x=outputs["atom_positions_predicted"], atom_mask=batch["atom_mask"],
            frame_rows=frame_rows,
        )
        valid_frame_mask = valid_frame_mask.bool()
        backend = tm_backend
        if backend == "auto":
            backend = "blockreduce" if HAVE_BLOCKREDUCE else "native"
            CENSUS["auto_resolved"][backend] = CENSUS["auto_resolved"].get(backend, 0) + 1
        if backend == "blockreduce" and not HAVE_BLOCKREDUCE:
            raise ImportError("tm_backend='blockreduce' requires BLOCKREDUCE v0 "
                              "(of3o_blockreduce.py) on PYTHONPATH")
        if backend == "blockreduce" and not bool(batch["token_mask"].bool().all()):
            logger.warning("token_mask contains zeros: BLOCKREDUCE FULL context has no "
                           "token mask -> using native TM backend for this element")
            CENSUS["native_fallback_elements"] += 1                       # FAIL-CLOSED CARRY: a counted event the package reads (never silent)
            backend = "native"
    else:
        backend = "native"
    CENSUS["backend_runs"][backend] = CENSUS["backend_runs"].get(backend, 0) + 1

    pp = pair_row_pass(batch, outputs, cc, rows=rows, device=device,
                       pair_output_device=pair_output_device, has_frame=valid_frame_mask,
                       tm_backend=backend, tm_finish=tm_finish, gpde_mode=gpde_mode,
                       samples_per_pass=samples_per_pass, want_tm=pae_enabled)
    confidence_scores["pde"] = pp["pde"]
    if cc.pde.return_probs:
        confidence_scores["pde_probs"] = pp["pde_probs"]
    confidence_scores["gpde"] = pp["gpde"]
    if cc.distogram.return_contact_probs:
        confidence_scores["contact_probs"] = pp["contact_probs"]

    if pae_enabled:
        confidence_scores["pae"] = pp["pae"]
        if cc.pae.return_probs:
            confidence_scores["pae_probs"] = pp["pae_probs"]

        # ---- full_complex_sample_ranking_metric (stock statements) --------------
        fc = cc.sample_ranking.full_complex
        iptm, ptm = pp["iptm"], pp["ptm"]
        atom_positions_predicted = outputs["atom_positions_predicted"]
        num_atoms_per_token = batch["num_atoms_per_token"]
        atom_mask = batch["atom_mask"].bool()
        token_mask = batch["token_mask"]
        asym_id = batch["asym_id"]
        is_polymer = batch["is_protein"] | batch["is_rna"] | batch["is_dna"]
        is_polymer_atomized = broadcast_token_feat_to_atoms(
            token_mask, num_atoms_per_token, is_polymer
        ).bool()
        asym_id_atomized = broadcast_token_feat_to_atoms(
            token_mask, num_atoms_per_token, asym_id
        ).bool()
        clash_kw = dict(asym_id=asym_id_atomized,
                        atom_positions_predicted=atom_positions_predicted,
                        atom_mask=atom_mask, is_polymer=is_polymer_atomized)
        if clash_rows is None:
            has_clash = compute_has_clash(**clash_kw)
        else:
            has_clash = compute_has_clash_chunked(clash_rows=clash_rows, **clash_kw)
        if torch.any(batch["is_protein"]):
            disorder = compute_disorder(batch=batch, outputs=outputs,
                                        disorder_threshold=fc.get("disorder_threshold", 0.581))
        else:
            disorder = torch.zeros(atom_positions_predicted.shape[:-2],
                                   device=atom_positions_predicted.device,
                                   dtype=atom_positions_predicted.dtype)
        scores = {}
        scores["iptm"] = iptm.detach().clone()
        scores["ptm"] = ptm.detach().clone()
        scores["disorder"] = disorder
        scores["has_clash"] = has_clash
        scores["sample_ranking_score"] = (
            (
                fc.get("iptm_weight", 0.8) * iptm
                + fc.get("ptm_weight", 0.2) * ptm
                + fc.get("disorder_weight", 0.5) * disorder
                - fc.get("has_clash_weight", 100.0) * has_clash
            )
            .detach()
            .clone()
        )
        confidence_scores.update(scores)

        unique_chains = torch.unique(batch["asym_id"].long()).tolist()
        if cc.sample_ranking.chain_pair_iptm.enabled:
            confidence_scores.update(
                _bespoke_from_chain_pair(batch, pp["chain_pair"], unique_chains,
                                         valid_frame_mask, device)
            )
        if cc.sample_ranking.chain_ptm.enabled:
            # stock compute_chain_ptm: keys are asym_id.item() in batch order of .unique()
            keys = [aid.item() for aid in batch["asym_id"].unique()]
            confidence_scores["chain_ptm"] = {
                k: v.detach().clone() for k, v in zip(keys, pp["chain_ptm_list"])
            }
    return confidence_scores


# =============================================================================
# public wrapper  (mirrors stock get_confidence_scores batch / sample iteration)
# =============================================================================
def get_confidence_scores_chunked(
    batch: dict,
    outputs: dict,
    config,
    compute_per_sample: bool = False,
    rows: int = 256,
    logits_device=None,
    *,
    frame_rows: int = 128,
    clash_rows: int | None = 4096,
    pair_output_device=None,
    tm_backend: str = "auto",
    tm_finish: str = "rowsum",
    gpde_mode: str = "exact",
    samples_per_pass: int | None = None,
) -> dict:
    """Drop-in replacement for stock `get_confidence_scores`.

    rows:               token rows per block for pae/pde/distogram logits (dim -3).
    logits_device:      compute device (default outputs["plddt_logits"].device); pae/
                        pde/distogram logits may live elsewhere (pinned CPU) and are
                        streamed block-wise.  All returned tensors live here.
    frame_rows:         tokens per block for the frame-atom distance rows.
    clash_rows:         atom rows per cdist block in has_clash (None -> stock function).
    pair_output_device: where the full [S,N,N] pae/pde matrices are assembled
                        (default = logits_device, as stock; "cpu" is the only
                        intentional placement deviation).
    tm_backend:         "auto" (BLOCKREDUCE if importable) | "blockreduce" | "native".
    tm_finish:          BLOCKREDUCE finish regime: "rowsum" (default; bitwise for
                        OF3, O(N) state) | "exact" (bitwise, ~20 N^2 B per in-flight
                        sample; kept for parity with the reference adapter).
    gpde_mode:          "exact" (stock statement on full matrices; bitwise) |
                        "stream" (O(rows*N) memory; reduction order differs, E2).
    samples_per_pass:   samples processed per sweep over the logits rows (default:
                        all).  Lower it to bound BLOCKREDUCE exact-mode state at the
                        cost of re-streaming the logits.
    """
    assert tm_backend in ("auto", "blockreduce", "native"), tm_backend
    assert tm_finish in ("exact", "rowsum"), tm_finish
    assert gpde_mode in ("exact", "stream"), gpde_mode
    if logits_device is None:
        logits_device = outputs["plddt_logits"].device
    device = torch.device(logits_device)
    pair_output_device = device if pair_output_device is None else torch.device(pair_output_device)

    atom_positions_predicted = outputs["atom_positions_predicted"]
    batch_size = atom_positions_predicted.size(0)
    num_samples = atom_positions_predicted.size(1)

    def slice_batch(t, i):
        if isinstance(t, torch.Tensor) and t.ndim >= 1:
            return t[i]
        return t

    def slice_sample(t, j):
        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] != 1:
            return t[j : j + 1]
        return t

    inner = partial(
        _get_confidence_scores_chunked, config=config, rows=rows, device=device,
        frame_rows=frame_rows, clash_rows=clash_rows, pair_output_device=pair_output_device,
        tm_backend=tm_backend, tm_finish=tm_finish, gpde_mode=gpde_mode,
        samples_per_pass=samples_per_pass,
    )

    per_batch_metrics = []
    for bi in range(batch_size):
        cur_batch_b = tensor_tree_map(
            lambda x: slice_batch(x, bi).squeeze(0),  # noqa: B023
            batch,
            strict_type=False,
        )
        cur_batch_b["atom_array"] = cur_batch_b["atom_array"][bi]
        cur_outputs_b = tensor_tree_map(
            lambda x: slice_batch(x, bi),  # noqa: B023
            outputs,
            strict_type=False,
        )
        if compute_per_sample and num_samples is not None and num_samples > 1:
            per_sample_metrics_list = []
            for sj in range(num_samples):
                cur_outputs_bs = tensor_tree_map(
                    lambda x: slice_sample(x, sj),  # noqa: B023
                    cur_outputs_b,
                    strict_type=False,
                )
                per_sample_metrics_list.append(inner(batch=cur_batch_b, outputs=cur_outputs_bs))
            cat_samples = partial(torch.concat, dim=0)
            metrics_b = dict_multimap(cat_samples, per_sample_metrics_list)
        else:
            metrics_b = inner(batch=cur_batch_b, outputs=cur_outputs_b)
        per_batch_metrics.append(metrics_b)

    metrics = dict_multimap(torch.stack, per_batch_metrics)
    return metrics
