"""OpenFold3's auxiliary heads + confidence scores with the pair representation row-sharded (``heads/head_modules.py``
``AuxiliaryHeadsAllAtom.forward``, ``heads/prediction_heads.py`` ``PairformerEmbedding`` / ``PredictedAlignedErrorHead`` /
``PredictedDistanceErrorHead`` / ``DistogramHead``, ``core/metrics/{confidence,sample_ranking,aggregate_confidence_ranking}.py``,
``runner.py`` ``_compute_confidence_scores``).

What never exists under this module: the ``[S, N, N, 64]`` PAE / PDE logits, the ``[1, N, N, 64]`` distogram logits, a per-sample
``[S, N, N, 128]`` pair representation, the ``[N_atom, N_atom]`` frame-search / clash matrices. The statements are the core's
(``opt_core.mem.rowpair``): the confidence pairformer's pair INPUT is built per row block from the trunk shard (``heads.embed_rows`` with
``embed_zij_rows`` below = ``PairformerEmbedding.embed_zij`` on GLOBAL rows; ``ROWPAIR_FREE_ZTRUNK=1`` overwrites the trunk shard's own
storage when S == 1), its pairformer runs through the ONE driver (``pairstack.pairformer_stack_rows``, per sample), the PAE logits are
produced per LOCAL row block (``heads.logit_rows``) and the PDE / distogram ``L + L^T`` per row block by column-slab exchange
(``heads.sym_logit_rows``); each block is soft-maxed and reduced on the spot: expected PAE / PDE rows, contact-probability rows, and the
per-context TM pieces (``confidence.ContextReducer`` in the ``block`` statement form = OpenFold3's ``compute_ptm`` order:
``ptm_ij = sum_b probs * w_D`` once per distinct context size over the block, rows / columns selected after). ``finalize`` assembles ONE
context's ``[S, n_D, n_D]`` matrix at a time on rank 0 and ``finish_context`` runs ``compute_ptm``'s closing statements on it verbatim
(row mean | interface mean, has-frame fill, max); the chain-pair / bespoke tables are ``compute_chain_pair_iptm``'s closing statements.
The token frames come from ``frames.nearest_atoms_rows`` (the start-atom rows of ``get_token_frame_atoms``' all-pairs search, then its
per-token statements), the clash rule from ``frames.count_pairs_within`` inside ``compute_has_clash``'s loop.

OUTPUT MATRICES (``pae`` / ``pde`` / ``contact_probs`` ``[S, N, N]``, what the writer stores) are gathered row-wise to rank 0
by row block straight onto rank 0's (pinned) HOST (``dist.gather_rows_to_rank0_host``); ranks > 0 hold ``[S, 0, 0]`` empties and never write.
REPLICATED BY DESIGN (named): the single representation, atom coordinates, pLDDT / experimentally-resolved logits (per-token heads, stock
calls), the per-token batch features. Refused by name (``ConfidenceRefused``): ``offload_inference``, the per-sample cutoff paths
(``per_sample_token_cutoff`` / ``per_sample_atom_cutoff`` hit with S > 1), batch size > 1, kernel flags.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from . import core as C
from . import pairstack as PS

C.register("heads", ("embed_rows", "logit_rows", "sym_logit_rows", "conf_rows", "ZTrunkPlan"))
C.register("confidence", ("ContextReducer",))
C.register("dist", ("gather_rows_to_rank0_host",))
C.register("frames", ("nearest_atoms_rows", "count_pairs_within", "frame_rows"))


class ConfidenceRefused(RuntimeError):
    pass


def check_trunk_shard(z_loc, lay) -> None:
    """``zij_trunk`` must be this rank's rows with batch size 1: ``[1, n_loc, N, C]``, or ``[1, 1, n_loc, N, C]`` once ``OpenFold3.forward`` added
    the sample dim (``model.py:656-664``) — refused by name otherwise."""
    N, C_z = lay.N, int(z_loc.shape[-1])
    if z_loc.dim() < 3 or int(z_loc.shape[-3]) != lay.R or int(z_loc.shape[-2]) != N or z_loc.numel() != lay.R * N * C_z:
        raise ConfidenceRefused(f"refused: zij_trunk {tuple(z_loc.shape)} is not this rank's rows [1, (1,) {lay.R}, {N}, C] (predict batch size 1)")


def rows4(z_rows):
    """``[w, N, C]``-ended rows with every leading dim 1 (``[1, 1, w, N, C]`` at roll-out time) viewed as ``[1, w, N, C]`` — a view, never a copy
    (the callers' row blocks and row-range slices of the shard are contiguous)."""
    return z_rows.reshape((1,) + tuple(z_rows.shape[-3:]))


def ztrunk_plan(z_loc, lay, passes: int = 1, log=None):
    """The core's placement of the trunk shard over the roll-out (``heads.ZTrunkPlan`` on ``output['zij_trunk']`` AS THE ROLL-OUT HOLDS IT —
    ``[1, 1, n_loc, N, C]`` with the sample dim, so ``plan.source()`` serves the diffusion conditioning row blocks in the shape the resident shard
    did; reads ``ROWPAIR_FREE_ZTRUNK`` / ``ROWPAIR_CONF_PARK_ZTRUNK`` — the tp line's constants (env.LINE_CONSTANTS), both 1
    on the tp line). ``model.rollout_rows`` makes ONE plan per roll-out spanning the sample loop's passes and parks at roll-out entry
    (``plan.park_now()``: the diffusion conditioning and every confidence pass then read trunk rows through ``plan.source()`` / ``plan.begin()``
    while the shard's device storage is released — zero restores; the host copy lives until the LAST pass's ``plan.end``); a caller without a
    roll-out gets a one-pass plan here."""
    check_trunk_shard(z_loc, lay)
    return C.seam("heads").ZTrunkPlan(z_loc.detach(), passes=int(passes), name="z_trunk", log=log)


def trunk_contact_rows(ah, zsrc, lay, config, S: int = 1):
    """The distogram head's contact probability rows ``sum_{b: end_b <= 8 A} softmax(DistogramHead(z_trunk))`` for this rank's rows, ``[1, n_loc, N]``
    — the trunk shard's only reader in the confidence stage besides the embedding. Taken from the DEVICE shard (``zsrc`` = the shard tensor, viewed
    ``[1, n_loc, N, C]``; ``heads.sym_logit_rows`` exchanges column slabs and refuses a parked source by name), so ``model.rollout_rows`` takes them
    ONCE at roll-out entry, before the shard is parked or embedded in place, and every pass's ``confidence_scores_rows`` reuses them (the trunk rows
    do not change between passes)."""
    dist_cfg = config.confidence.distogram
    ends = torch.linspace(dist_cfg["bin_min"], dist_cfg["bin_max"], dist_cfg["no_bins"] + 1, device=zsrc.device)[1:]
    rows, _ = C.seam("heads").conf_rows(lay.N, 64, None, int(S))
    return distogram_contact_rows(ah, rows4(zsrc) if torch.is_tensor(zsrc) else zsrc, lay, ends <= 8.0, rows=rows)


# ================================================================================================================ PairformerEmbedding on rows
def embed_zij_rows(pe, si_input, z_rows, x_pred, g0: int, g1: int):
    """``PairformerEmbedding.embed_zij`` for GLOBAL pair rows ``g0:g1``: ``si_input`` ``[N, c]``, ``z_rows`` ``[w, N, C]``, ``x_pred`` ``[S, N, 3]``
    -> ``[S, w, N, C]``. The per-token linears run on all N tokens (stock M) and are sliced; the distance one-hot is the rows' own."""
    zij = z_rows                                                             # prediction_heads.py:80-111 (no autocast wrapper of its own: the roll-out's cast_dtype context applies)
    zij = zij + pe.linear_i(si_input.unsqueeze(-2))[g0:g1, :, :] + pe.linear_j(si_input.unsqueeze(-3))
    bins = torch.linspace(pe.min_bin, pe.max_bin, pe.no_bin, device=zij.device, dtype=zij.dtype)
    squared_bins = bins ** 2
    upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([pe.inf])], dim=-1)
    dij = torch.sum((x_pred[..., g0:g1, None, :] - x_pred[..., None, :, :]) ** 2, dim=-1, keepdims=True)
    dij = ((dij > squared_bins) * (dij < upper)).type(x_pred.dtype)
    zij = zij + pe.linear_distance(dij)
    return zij


def pairformer_embedding_rows(pe, si_input, si, z2, x_pred, single_mask, pair_mask_loc, lay, chunk_size=None, inplace_safe=False, _mask_trans=True, inplace=False,
                              pairformer_dtype: torch.dtype = torch.float32, after_embed=None, **flags):
    """``PairformerEmbedding.pairformer_emb`` with the pair representation as this rank's rows. ``si_input`` ``[N, c]``, ``si`` ``[N, c_s]``,
    ``z2`` = the source ``heads.ZTrunkPlan.begin`` returned (the trunk shard ``[1, (1,) n_loc, N, C]``, READ per row block — with ``inplace`` its storage
    is CONSUMED and the returned ``zij`` is a view of it — or the parked shard serving ``.zrows`` from the host), ``x_pred`` ``[S, N, 3]``,
    ``single_mask`` ``[S, N]``, ``pair_mask_loc`` ``[n_loc, N]`` ->
    ``(si [S, N, c_s], zij [S, n_loc, N, C])``. The confidence pairformer then runs IN PLACE on ``zij`` sample by sample (``pairstack.pair_stack_``
    with ``reuse_storage``: the driver returns the caller's tensor; a returned tensor with another storage is copied back and named ``conf_pairstack=copy``)."""
    HD = C.seam("heads")
    S = int(x_pred.shape[0])
    zc = HD.embed_rows(lambda z_blk, g0, g1: embed_zij_rows(pe, si_input, rows4(z_blk)[0], x_pred, g0, g1), z2, lay, lead_out=(S,), inplace=bool(inplace),
                       out_dtype=z2.dtype, bins=int(pe.no_bin))                                   # z_blk: the source's row block [1, (1,) w, N, C] -> the statement's [w, N, C]
    if after_embed is not None:                                              # the z_trunk park may retire BEFORE this pass's pair stack (heads.ZTrunkPlan.retire)
        after_embed()
    si = si.unsqueeze(0).expand(S, *si.shape[-2:]).clone()
    use_kernels = any(bool(flags.get(k)) for k in ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels"))
    if use_kernels and S > 1:
        chunk_size = None
    s_out = []
    in_dtype = zc.dtype                                                      # prediction_heads.py:222-233: the pairformer under autocast(pairformer_dtype), si / zij back to in_dtype
    with torch.amp.autocast(device_type="cuda", dtype=pairformer_dtype):
        for s in range(S):                                                   # one pairformer pass per sample (independent samples; the ONE driver)
            si_s, z_s = PS.pairformer_stack_rows(pe.pairformer_stack, si[s:s + 1], zc[s:s + 1], single_mask[s:s + 1], pair_mask_loc.unsqueeze(0), lay,
                                                 chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans, **flags)
            s_out.append(si_s.to(dtype=in_dtype))
            copied = z_s.data_ptr() != zc[s:s + 1].data_ptr()
            if copied or z_s.dtype != in_dtype:                              # the driver returned a fresh tensor (reuse_storage off) or an autocast dtype change: copied back into the shard's storage
                zc[s:s + 1].copy_(z_s)
            C.fn("evidence", "record_schedule")(conf_pairstack=("copy" if copied else "inplace"))
            del z_s
    return torch.cat(s_out, dim=0), zc


# ================================================================================================================ token frames / clash (row statements)
def token_frame_atoms_rows(batch: dict, x: torch.Tensor, atom_mask: torch.Tensor, angle_threshold: float = 25.0, eps: float = 1e-8, inf: float = 1e9):
    """``get_token_frame_atoms(batch, x, atom_mask, ...)`` -> ``(phi, valid_frame_mask)`` with the all-pairs nearest-atom search replaced by
    ``frames.nearest_atoms_rows`` over the token START atoms (the only rows the stock statement reads back); everything after is the stock
    per-token statement sequence."""
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms, get_token_atom_index_offset
    atom_asym_id = broadcast_token_feat_to_atoms(token_mask=batch["token_mask"], num_atoms_per_token=batch["num_atoms_per_token"], token_feat=batch["asym_id"])
    start_atom_index = batch["start_atom_index"].long()
    start_atom_index = start_atom_index.expand(*x.shape[:-2], start_atom_index.shape[-1])
    closest = C.fn("frames", "nearest_atoms_rows")(x, start_atom_index, atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]),
                                                    atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]), k=3, eps=eps, inf=inf)
    a_index, c_index = closest[..., 1], closest[..., 2]
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (1 - batch["is_atomized"])
    restype = batch["restype"]
    off, msk = {}, {}
    for name in ("N", "CA", "C", "C3'", "C1'", "C4'"):
        off[name], msk[name] = get_token_atom_index_offset(atom_name=name, restype=restype)
    frame_atoms = {
        "a": {"index": a_index * batch["is_atomized"] + (start_atom_index + off["N"]) * is_standard_protein + (start_atom_index + off["C3'"]) * is_standard_nucleotide,
              "token_atom_mask": batch["is_atomized"] + msk["N"] * is_standard_protein + msk["C3'"] * is_standard_nucleotide},
        "b": {"index": start_atom_index * batch["is_atomized"] + (start_atom_index + off["CA"]) * is_standard_protein + (start_atom_index + off["C1'"]) * is_standard_nucleotide,
              "token_atom_mask": batch["is_atomized"] + msk["CA"] * is_standard_protein + msk["C1'"] * is_standard_nucleotide},
        "c": {"index": c_index * batch["is_atomized"] + (start_atom_index + off["C"]) * is_standard_protein + (start_atom_index + off["C4'"]) * is_standard_nucleotide,
              "token_atom_mask": batch["is_atomized"] + msk["C"] * is_standard_protein + msk["C4'"] * is_standard_nucleotide},
    }
    for key in frame_atoms:
        idx = frame_atoms[key]["index"].long()
        frame_atoms[key].update({
            "atom_positions": torch.gather(x, dim=-2, index=idx.unsqueeze(-1).expand(*(x.shape[:-2] + (idx.shape[-1], 3)))),
            "asym_id": torch.gather(atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]), dim=-1, index=idx),
            "atom_mask": torch.gather(atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]), dim=-1, index=idx) * batch["token_mask"] * frame_atoms[key]["token_atom_mask"],
        })
    u = frame_atoms["a"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    v = frame_atoms["c"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    uv = torch.einsum("...i,...i->...", u, v)
    u_norm = (eps + torch.sum(u ** 2, dim=-1)) ** 0.5
    v_norm = (eps + torch.sum(v ** 2, dim=-1)) ** 0.5
    cos_angle = uv / (u_norm * v_norm)
    cos_angle_min_bound = math.cos((180 - angle_threshold) * math.pi / 180)
    cos_angle_max_bound = math.cos(angle_threshold * math.pi / 180)
    valid_frame_mask_angle = (cos_angle < cos_angle_max_bound) * (cos_angle > cos_angle_min_bound)
    valid_frame_mask_angle = (valid_frame_mask_angle * batch["is_atomized"] + torch.ones_like(valid_frame_mask_angle) * (1 - batch["is_atomized"])) * batch["token_mask"]
    valid_frame_mask_atom = frame_atoms["a"]["atom_mask"] * frame_atoms["b"]["atom_mask"] * frame_atoms["c"]["atom_mask"]
    valid_frame_mask_asym_id = (frame_atoms["a"]["asym_id"] == frame_atoms["b"]["asym_id"]) * (frame_atoms["b"]["asym_id"] == frame_atoms["c"]["asym_id"])
    valid_frame_mask = valid_frame_mask_angle * valid_frame_mask_atom * valid_frame_mask_asym_id
    phi = (frame_atoms["a"]["atom_positions"], frame_atoms["b"]["atom_positions"], frame_atoms["c"]["atom_positions"])
    return phi, valid_frame_mask


def has_clash_rows(asym_id, atom_positions_predicted, atom_mask, is_polymer, threshold=1.1, violation_abs=100, violation_frac=0.5):
    """``compute_has_clash`` with each chain pair's ``(cdist < threshold).sum()`` counted by ``frames.count_pairs_within`` (whole when the pair
    matrix fits ``ROWPAIR_CLASH_MAX_GB``, else in row blocks; integer counts)."""
    count = C.fn("frames", "count_pairs_within")
    device, dtype = atom_positions_predicted.device, atom_positions_predicted.dtype
    unique_chains = torch.unique(asym_id).tolist()
    num_samples = atom_positions_predicted.size(0)
    polymer_chains = list(filter(lambda aid: ((asym_id != aid) | is_polymer).all(), unique_chains))
    chain_masks = [(asym_id == aid) & atom_mask for aid in polymer_chains]
    has_clash = torch.zeros(num_samples, dtype=dtype, device=device)
    for s in range(num_samples):
        clashing = False
        for i in range(len(chain_masks)):
            ni = chain_masks[i].sum()
            if ni == 0:
                continue
            for j in range(i + 1, len(chain_masks)):
                nj = chain_masks[j].sum()
                if nj == 0:
                    continue
                num_clashes = count(atom_positions_predicted[s, chain_masks[i], :], atom_positions_predicted[s, chain_masks[j], :], threshold)
                if (num_clashes > violation_abs) or ((num_clashes / min(ni, nj)) > violation_frac):
                    has_clash[s] = 1.0
                    clashing = True
                    break
            if clashing:
                break
    return has_clash


# ================================================================================================================ compute_ptm contexts + finish
class PTMContext(object):
    """One ``compute_ptm`` call's token subset: ``mask`` (bool ``[N]``: ``mask_i``), ``w`` (its ``bin_weight`` ``[no_bins]`` — contexts of equal
    size share one tensor, so the reducer's ``block`` form makes one pass per size), ``num`` (``num_tokens_considered``), ``asym`` (the
    ``asym_id`` the interface form pairs on, or None), ``key`` (``"full"`` | ``("chain", aid)`` | ``("pair", i, j)``)."""

    def __init__(self, key, mask, w, num, asym, chains=None):
        self.key, self.mask, self.w, self.num, self.asym = key, mask, w, num, asym
        self.chains = chains                                       # the context's chain-label tuple ((a,) | (a_i, a_j)); None = the full complex (the label reducer's key)


def ptm_contexts(batch_b: dict, cfg_ptm, device, dtype, want_chain_ptm: bool, want_chain_pair: bool):
    """The contexts ``_get_confidence_scores`` evaluates ``compute_ptm`` on: the full complex (``mask_i = token_mask``; pTM and ipTM),
    ``compute_chain_ptm``'s per-chain masks, ``compute_chain_pair_iptm``'s chain-pair unions — built from ONE label vector by the core's
    ``confidence.ptm_contexts_from_labels`` (three host syncs whatever the context count, instead of one
    ``mask_i.sum().item()`` per context and ``C²`` device masks). ``bin_weight`` per context is ``compute_ptm``'s OWN statement, here, fed the
    integer count (d0 from ``num_tokens_considered``; equal-size contexts share one weight tensor exactly as before); the per-context ``mask`` is
    built only under ``ROWPAIR_CONF_REDUCER=classic|both`` (the fast reducer selects by label). Returns ``(contexts, unique_chains, chain_masks, labels)``."""
    from openfold3.core.metrics.confidence import get_bin_centers
    token_mask = batch_b["token_mask"]
    asym_id = batch_b["asym_id"]
    asym_l = asym_id.long()
    w_by_n: Dict[int, torch.Tensor] = {}

    def make(key, n, chains, mask):                                          # 1 + C + C(C-1)/2 calls with host ints; mask is None under the fast reducer
        num_tokens_considered = torch.tensor(int(n), dtype=torch.int64, device=device).clamp_min(1).to(dtype)   # == mask_i.sum().clamp_min(1).to(dtype)
        n = int(n)
        if n not in w_by_n:
            clipped = torch.maximum(num_tokens_considered, torch.tensor(19.0, device=device, dtype=dtype))
            d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
            bin_centers = get_bin_centers(cfg_ptm["bin_min"], cfg_ptm["bin_max"], cfg_ptm["no_bins"], device, dtype)
            w_by_n[n] = 1.0 / (1.0 + (bin_centers / d0) ** 2)
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
        asym = asym_id if key == "full" else (asym_l if chains is not None and len(chains) == 2 else None)
        return PTMContext(key, mask, w_by_n[n], num_tokens_considered, asym, chains=chains)

    ctxs, unique_chains, labels, _n_by_chain = C.fn("confidence", "ptm_contexts_from_labels")(
        asym_id, token_mask, make, want_chain_ptm=want_chain_ptm, want_chain_pair=want_chain_pair)
    chain_masks = [(asym_l == aid) & token_mask.bool() for aid in unique_chains] if want_chain_pair else None   # C masks for chain_pair_tables (its consumer, unchanged)
    return ctxs, (unique_chains if want_chain_pair else None), chain_masks, labels


def finish_context(ctx: PTMContext, ptm_ij: torch.Tensor, has_frame: torch.Tensor, interface: bool, eps: float = 1e-8):
    """``compute_ptm`` after ``ptm_ij`` (``[S, n_D, n_D]``, rows / columns = ``mask_i`` in token order), verbatim: interface -> mean over j in
    another chain; else sum / ``num_tokens_considered``; rows without a frame -> 0; max over rows."""
    has_frame = has_frame[:, ctx.mask.to(has_frame.device)].bool()
    if interface:
        asym_id = ctx.asym.to(ptm_ij.device)[ctx.mask.to(ptm_ij.device)]
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(dim=-1) / ctx.num
    tm_i = tm_i.masked_fill(~has_frame.to(tm_i.device), 0.0)
    return tm_i.max(dim=-1).values


def finish_context_idx(ctx: PTMContext, ptm_ij: torch.Tensor, has_frame: torch.Tensor, interface: bool, idx: torch.Tensor, asym_full: torch.Tensor, eps: float = 1e-8):
    """:func:`finish_context` with the context's token INDEX (``idx``: ascending token positions = the reducer's ``index(k)``) in place of its boolean
    mask: ``has_frame.index_select(1, idx)`` for ``has_frame[:, mask]`` and ``asym_full.index_select(0, idx)`` for ``asym[mask]`` — identical elements in
    identical order (no boolean-mask host sync; runs under every reducer, the fast one included, whose contexts carry no mask)."""
    idx = idx.to(device=ptm_ij.device, dtype=torch.long)
    has_frame = has_frame.to(ptm_ij.device).index_select(1, idx).bool()
    if interface:
        asym_id = asym_full.to(ptm_ij.device).index_select(0, idx)
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(dim=-1) / ctx.num
    tm_i = tm_i.masked_fill(~has_frame.to(tm_i.device), 0.0)
    return tm_i.max(dim=-1).values


def chain_pair_tables(pair_vals: dict, unique_chains, chain_masks, has_frame, is_ligand, device, dtype):
    """``compute_chain_pair_iptm`` after the per-pair ipTM values (``sample_ranking.py:253-310``): the symmetric ``chain_pair_iptm`` table, the
    per-sample masked chain means over framed partners, the ligand-aware ``bespoke_iptm``, and the ``"(a,b)"``-keyed maps it returns.
    ``has_frame``: ``[num_samples, num_tokens]`` bool.  The stock tail's ``C²`` Python loops (two device copies + a ``chain_is_ligand[i]``
    host sync PER PAIR) are ONE stack / index_put / gather / where each; the values are
    the stock statements' (the masked mean's terms in the stock order ``(cp[i,j]·hf_i, cp[j,i]·hf_j)`` over ascending ``j != i``), and the two
    maps are views of ONE host copy of each ``[S, C, C]`` table (the writer reads them entry by entry: no per-entry device sync)."""
    num_chains, num_samples = len(unique_chains), int(has_frame.shape[0])
    C, S = num_chains, num_samples
    chain_pair_iptm = torch.zeros((S, C, C), device=device, dtype=dtype)
    if C > 1:
        iu = [(i, j) for i in range(C) for j in range(i + 1, C)]                # the loops' (i < j) order
        ii = torch.tensor([p[0] for p in iu], device=device); jj = torch.tensor([p[1] for p in iu], device=device)
        vals = torch.stack([pair_vals[p] for p in iu], dim=-1).to(device=device, dtype=dtype)   # [S, C(C-1)/2]
        chain_pair_iptm[:, ii, jj] = vals
        chain_pair_iptm[:, jj, ii] = vals
    is_ligand = is_ligand.to(device).bool()
    has_frame = has_frame.bool().to(device)                                  # [S, N]
    cm = torch.stack([m.to(device).bool() for m in chain_masks], dim=0) if C else torch.zeros((0, int(has_frame.shape[-1])), dtype=torch.bool, device=device)   # [C, N]
    chain_has_frame = (cm.unsqueeze(0) & has_frame.unsqueeze(1)).any(dim=-1)     # [S, C]  (sample_ranking.py:257-260: per SAMPLE)
    chain_is_ligand = (cm & is_ligand.unsqueeze(0)).sum(dim=-1) * 2 >= cm.sum(dim=-1)   # [C] bool, on the device (no per-chain host sync)
    chain_mean_iptm = torch.zeros((S, C), device=device, dtype=dtype)
    if C > 1:                                                                # sample_ranking.py:269-285: the masked mean over framed partners, per sample —
        js = torch.tensor([[j for j in range(C) if j != i] for i in range(C)], device=device)           # [C, C-1] ascending j != i
        iis = torch.arange(C, device=device).unsqueeze(-1).expand(C, C - 1)                            # [C, C-1]
        v_ij = chain_pair_iptm[:, iis, js]                                   # [S, C, C-1]  cp[i, j]
        v_ji = chain_pair_iptm[:, js, iis]                                   # [S, C, C-1]  cp[j, i]
        vals_t = torch.stack((v_ij, v_ji), dim=-1).reshape(S, C, 2 * (C - 1))                          # interleaved (cp[i,j], cp[j,i]) per j: the loop's `vals` order
        hf = chain_has_frame.to(dtype)                                       # [S, C]
        m_i = hf.unsqueeze(-1).expand(S, C, C - 1)                           # hf[:, i] beside cp[i, j]
        m_j = hf[:, js]                                                      # hf[:, j] beside cp[j, i]   [S, C, C-1]
        masks_t = torch.stack((m_i, m_j), dim=-1).reshape(S, C, 2 * (C - 1))
        for i in range(C):                                                   # per chain the stock statement on the stock-shaped [S, 2(C-1)] operands (one reduction each,
            v_i, k_i = vals_t[:, i].contiguous(), masks_t[:, i].contiguous() #  bitwise the loop form's sum order; C small launches instead of C² — the C² part is the gathers above)
            denom = k_i.sum(dim=-1).clamp(min=1)
            chain_mean_iptm[:, i] = (v_i * k_i).sum(dim=-1) / denom
    mi_i = chain_mean_iptm.unsqueeze(-1).expand(S, C, C)                     # mean_i at [s, i, j]
    mi_j = chain_mean_iptm.unsqueeze(-2).expand(S, C, C)                     # mean_j at [s, i, j]
    lig_i = chain_is_ligand.view(1, C, 1); lig_j = chain_is_ligand.view(1, 1, C)
    bespoke_iptm = torch.where(lig_i, mi_i, torch.where(lig_j, mi_j, 0.5 * (mi_i + mi_j)))          # if ligand[i]: mean_i elif ligand[j]: mean_j else the average
    if C:
        eye = torch.eye(C, dtype=torch.bool, device=device).unsqueeze(0)
        bespoke_iptm = bespoke_iptm.masked_fill(eye, 0.0)                    # i == j: skipped by the loop (stays 0)
    cp_host, bs_host = chain_pair_iptm.detach().to("cpu"), bespoke_iptm.detach().to("cpu")            # ONE host copy each: the maps' S-vectors are read entry by entry downstream
    cp_map, bs_map = {}, {}
    for i in range(C):
        for j in range(i + 1, C):
            key = f"({unique_chains[i]},{unique_chains[j]})"
            cp_map[key], bs_map[key] = cp_host[:, i, j], bs_host[:, i, j]
    return cp_map, bs_map


def rows_to_rank0_host(rows_loc: torch.Tensor, lay):
    """``[S, n_loc, N]`` per-row outputs -> ``[S, N, N]`` on rank 0's HOST (``dist.gather_rows_to_rank0_host``: column blocks, pinned);
    ``[S, 0, 0]`` on the other ranks."""
    S = int(rows_loc.shape[0])
    full = C.fn("dist", "gather_rows_to_rank0_host")(rows_loc.permute(1, 2, 0).contiguous(), lay)      # rows [R, N, S] of [N, N, S] -> rank 0's (pinned) host by row block | None
    return rows_loc.new_zeros((S, 0, 0), device="cpu") if full is None else full.permute(2, 0, 1).contiguous()   # [S, N, N] row-major, as the dense statement sums it


# ================================================================================================================ confidence scores from rows
def confidence_scores_rows(ah, batch, zc, zt2, plddt_logits, atom_positions_predicted, lay, config, contact_loc=None):
    """Everything ``_compute_confidence_scores`` -> ``get_confidence_scores`` returns, from row shards: ``zc`` ``[S, n_loc, N, C]`` (the
    confidence pair representation rows), ``zt2`` ``[1, n_loc, N, C]`` (the trunk rows the distogram head reads; None when ``contact_loc``
    ``[1, n_loc, N]`` — the contact-probability rows taken before the trunk shard was reused — is given), ``plddt_logits`` /
    ``atom_positions_predicted`` replicated. Rank 0 returns the stacked dict ``get_confidence_scores`` returns (``pae`` / ``pde`` / ``contact_probs`` on its host); ranks > 0
    the same keys with empty / zero placeholders (they never write)."""
    from openfold3.core.metrics.confidence import probs_to_expected_error
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
    from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map
    HD, CF = C.seam("heads"), C.seam("confidence")
    comm = C.comm()
    if int(atom_positions_predicted.shape[0]) != 1:
        raise ConfidenceRefused(f"refused: confidence scores on row shards take predict batches (batch size 1); got {int(atom_positions_predicted.shape[0])}")
    S, N = int(atom_positions_predicted.shape[1]), lay.N
    num_atoms = int(atom_positions_predicted.shape[-2])
    mem = config.settings.memory.eval
    if S > 1 and mem.get("per_sample_atom_cutoff", None) is not None and num_atoms > mem.per_sample_atom_cutoff:
        raise ConfidenceRefused("refused: the per-sample confidence loop (samples > 1 with more atoms than settings.memory.eval.per_sample_atom_cutoff) is not a "
                                "row-sharded domain of the tp line — its domain is per_sample_atom_cutoff: null (the launcher's pinned yaml, `pred --mode big "
                                "--n_gpu P` sets it); a rank started by hand takes that yaml")
    cconf = config.confidence
    device, dtype = zc.device, zc.dtype
    def slice_batch(t, i):                                                   # get_confidence_scores: batch element 0, sample dim of the features squeezed
        return t[i] if isinstance(t, torch.Tensor) and t.ndim >= 1 else t
    batch_b = tensor_tree_map(lambda x: slice_batch(x, 0).squeeze(0), batch, strict_type=False)
    atom_array = batch_b.get("atom_array") if batch_b.get("atom_array") is not None else batch.get("atom_array")
    batch_b["atom_array"] = atom_array[0] if atom_array is not None else None
    x_b = atom_positions_predicted[0]                                        # [S, N_atom, 3]
    scores = {"plddt": probs_to_expected_error(torch.softmax(plddt_logits[0], dim=-1), **cconf.plddt) * 100.0}
    pae_enabled = bool(config.architecture.heads.pae.enabled)
    rows, rows_src = HD.conf_rows(N, 64, None, S)
    comm.log(f"[confidence] N={N} S={S} rows/rank={lay.R} conf_rows={rows} ({rows_src}) pae={'on' if pae_enabled else 'off'}") if comm.verbose else None

    # ---- distogram contact rows: softmax(L + L^T)[bins <= 8 A] summed, L = distogram.linear(z_trunk)
    dist_cfg = cconf.distogram
    ends = torch.linspace(dist_cfg["bin_min"], dist_cfg["bin_max"], dist_cfg["no_bins"] + 1, device=device)[1:]
    bins_8A = ends <= 8.0
    if contact_loc is None:
        if zt2 is None:
            raise ConfidenceRefused("refused: no trunk rows and no precomputed contact rows for the distogram head")
        contact_loc = distogram_contact_rows(ah, zt2, lay, bins_8A, rows=rows)

    # ---- PDE rows: softmax(L + L^T) -> expected error, L = pde.linear(pde.layer_norm(zij))
    pde_loc = torch.empty(S, lay.R, N, device=device, dtype=dtype)
    for i0, i1, logits in HD.sym_logit_rows(lambda z: ah.pde.linear(ah.pde.layer_norm(z)), zc, lay, rows=rows, bins=64):
        pde_loc[:, i0:i1] = probs_to_expected_error(torch.softmax(logits, dim=-1), **cconf.pde)
        del logits

    # ---- PAE rows: softmax(L) -> expected error rows + the compute_ptm pieces of every context, L = pae.linear(pae.layer_norm(zij))
    if pae_enabled:
        _, has_frame = token_frame_atoms_rows(batch_b, x_b, batch_b["atom_mask"])
        has_frame = has_frame.bool()                                         # [S, N]
        ctxs, unique_chains, chain_masks, ctx_labels = ptm_contexts(batch_b, cconf.ptm, device, dtype, want_chain_ptm=bool(cconf.sample_ranking.chain_ptm.enabled),
                                                        want_chain_pair=bool(cconf.sample_ranking.chain_pair_iptm.enabled))
        red = C.fn("confidence", "context_reducer_for_layout")(ctxs, lay, form="block", labels=ctx_labels)   # ROWPAIR_CONF_REDUCER: fast (the tp line's default, tp.rank_env) | classic (the tag's per-context reducer) | both (side by side, refused by name on a mismatch)
        asym_long = batch_b["asym_id"].long()
        pae_loc = torch.empty(S, lay.R, N, device=device, dtype=dtype)
        for i0, i1, logits in HD.logit_rows(lambda z: ah.pae.linear(ah.pae.layer_norm(z)), zc, lay, rows=rows, bins=64):
            probs = torch.softmax(logits, dim=-1)
            del logits
            pae_loc[:, i0:i1] = probs_to_expected_error(probs, **cconf.pae)
            red.consume(i0, i1, probs)
            del probs

    # ---- output matrices to rank 0's host; gpde on rank 0 from the host matrices (fp32 sums, the stock statement)
    pde_full = rows_to_rank0_host(pde_loc, lay)
    del pde_loc
    contact_full = rows_to_rank0_host(contact_loc, lay)
    del contact_loc
    rank0 = comm.rank == 0
    if rank0:
        scores["pde"] = pde_full
        scores["gpde"] = (torch.sum(contact_full * pde_full, dim=[-2, -1]) / (torch.sum(contact_full, dim=[-2, -1]) + 1e-8)).to(device)
        if dist_cfg.get("return_contact_probs", False):
            scores["contact_probs"] = contact_full
    del contact_full
    if pae_enabled:
        pae_full = rows_to_rank0_host(pae_loc, lay)
        del pae_loc
        pair_vals = {}
        results = {}

        def finish(k, ctx, T):                                               # compute_ptm's close on context k's [S, n_D, n_D] matrix (rank 0)
            idx = red.index(k, T.device)                                     # the context's ascending token index (every reducer has it; no boolean-mask sync)
            if ctx.key == "full":
                results["iptm"] = finish_context_idx(ctx, T, has_frame, True, idx, asym_long)
                results["ptm"] = finish_context_idx(ctx, T, has_frame, False, idx, asym_long)
            elif ctx.key[0] == "chain":
                results.setdefault("chain_ptm", {})[ctx.key[1]] = finish_context_idx(ctx, T, has_frame, False, idx, asym_long).detach().clone()
            else:
                pair_vals[(ctx.key[1], ctx.key[2])] = finish_context_idx(ctx, T, has_frame, True, idx, asym_long)
            return None

        red.finalize(finish=finish)                                          # COLLECTIVE: one context matrix alive at a time on rank 0
        if rank0:
            scores["pae"] = pae_full
            frc = cconf.sample_ranking.full_complex
            iptm, ptm = results["iptm"], results["ptm"]
            token_mask, asym_id, num_atoms_per_token = batch_b["token_mask"], batch_b["asym_id"], batch_b["num_atoms_per_token"]
            atom_mask = batch_b["atom_mask"].bool()
            is_polymer = batch_b["is_protein"] | batch_b["is_rna"] | batch_b["is_dna"]
            is_polymer_atomized = broadcast_token_feat_to_atoms(token_mask, num_atoms_per_token, is_polymer).bool()
            asym_id_atomized = broadcast_token_feat_to_atoms(token_mask, num_atoms_per_token, asym_id)
            has_clash = has_clash_rows(asym_id=asym_id_atomized, atom_positions_predicted=x_b, atom_mask=atom_mask, is_polymer=is_polymer_atomized)
            if torch.any(batch_b["is_protein"].bool()) and batch_b.get("atom_array", None) is not None:
                from openfold3.core.metrics.sample_ranking import compute_disorder
                disorder = compute_disorder(batch=batch_b, outputs={"atom_positions_predicted": x_b}, disorder_threshold=frc.get("disorder_threshold", 0.581))
            else:
                if torch.any(batch_b["is_protein"].bool()):
                    comm.log("[confidence] disorder=skipped:no_atom_array (ranking score without the disorder term)")
                disorder = torch.zeros(x_b.shape[:-2], device=x_b.device, dtype=x_b.dtype)
            scores["iptm"], scores["ptm"] = iptm.detach().clone(), ptm.detach().clone()
            scores["disorder"], scores["has_clash"] = disorder, has_clash
            scores["sample_ranking_score"] = (frc.get("iptm_weight", 0.8) * iptm + frc.get("ptm_weight", 0.2) * ptm + frc.get("disorder_weight", 0.5) * disorder
                                              - frc.get("has_clash_weight", 100.0) * has_clash).detach().clone()
            if cconf.sample_ranking.chain_pair_iptm.enabled:
                scores["chain_pair_iptm"], scores["bespoke_iptm"] = chain_pair_tables(pair_vals, unique_chains, chain_masks, has_frame, batch_b["is_ligand"].bool(), device, dtype)
            if cconf.sample_ranking.chain_ptm.enabled:
                scores["chain_ptm"] = results.get("chain_ptm", {})
    if not rank0:
        z0 = torch.zeros(S, device=device, dtype=dtype)
        scores.update({"pde": pde_full, "gpde": z0.clone(), "iptm": z0.clone(), "ptm": z0.clone(), "disorder": z0.clone(), "has_clash": z0.clone(),
                       "sample_ranking_score": z0.clone(), "chain_pair_iptm": {}, "bespoke_iptm": {}, "chain_ptm": {}})
        if pae_enabled:
            scores["pae"] = pae_full
    return dict_multimap(torch.stack, [scores])


def distogram_contact_rows(ah, zt2, lay, bins_8A, rows=None):
    """``sum_{b: end_b <= 8 A} softmax(DistogramHead(z))[..., b]`` for this rank's rows: ``[1, n_loc, N]`` (``L + L^T`` by column-slab exchange)."""
    out = torch.empty(int(zt2.shape[0]), lay.R, lay.N, device=zt2.device, dtype=zt2.dtype)
    for i0, i1, logits in C.seam("heads").sym_logit_rows(lambda z: ah.distogram.linear(z), zt2, lay, rows=rows, bins=64):
        out[:, i0:i1] = torch.sum(torch.softmax(logits, dim=-1)[..., bins_8A], dim=-1)
        del logits
    return out


# ================================================================================================================ AuxiliaryHeadsAllAtom on rows
def aux_heads_rows(ah, batch, si_input, output, lay, config, use_zij_trunk_embedding=True, chunk_size=None, inplace_safe=False, offload_inference=False,
                   _mask_trans=True, reuse_ztrunk=None, ztrunk=None, contact_loc=None, pairformer_dtype: torch.dtype = torch.float32, **flags):
    """``AuxiliaryHeadsAllAtom.forward`` with ``output['zij_trunk']`` = this rank's rows ``[1, n_loc, N, C]``. Returns ``aux_out`` with the
    replicated per-token heads (``plddt_logits``, ``experimentally_resolved_logits``) and ``_tp_confidence`` = the finished dict
    ``_compute_confidence_scores`` returns (rank 0: real, incl. ``pae`` / ``pde`` on host; ranks > 0: placeholders); ``pae_logits`` /
    ``pde_logits`` / ``distogram_logits`` are never materialised.

    The trunk shard's placement is the core's ``heads.ZTrunkPlan`` (``ztrunk_plan``): ``ztrunk = (plan, i, last_use, device_needed_after)`` from
    ``model.rollout_rows`` (pass ``i`` of the roll-out's plan; ``last_use``: no later pass embeds the trunk rows (None = ``i`` is the plan's last
    pass); ``device_needed_after``: a statement after this pass reads the device shard whole),
    or None = a one-pass plan over this call (``reuse_ztrunk``: this call is the trunk shard's last reader — None / True — or not). Per pass the
    plan says ``inplace`` (``ROWPAIR_FREE_ZTRUNK=1``, one sample, last use: the pair representation overwrites the shard's storage), ``parked:<where>``
    (``ROWPAIR_CONF_PARK_ZTRUNK=1``: the shard sits on pinned host, its device storage released BEFORE the pass's ``[S, n_loc, N, C]`` pair
    representation is allocated, row blocks served from the host) or ``resident`` — the device then holds S shard-equivalents under either lever,
    1 + S resident; the words ride the schedule census (``conf_ztrunk=…``, the core's record). ``contact_loc``: the distogram head's contact rows
    taken beforehand (``trunk_contact_rows``; None = taken here from the device shard before the plan acts). The confidence pairformer runs in
    place on the pass's pair representation (``conf_pairstack=inplace``)."""
    from openfold3.core.model.heads import head_modules as _hm                # get_token_representative_atoms / broadcast_token_feat_to_atoms as the head module binds them
    PS.refuse_kernel_flags("AuxiliaryHeads", **flags)
    comm = C.comm()
    C.fn("evidence", "record_schedule")(conf_offload=("stock_on_na" if offload_inference else "off"))
    if offload_inference:                                                    # head_modules.py:140-141,194-259: stock moves the [S, N, N, *] pair rep / logits to the host above token_cutoff;
        comm.log("[confidence] offload_inference.confidence_heads=true (the runner yaml's shipped setting above its token_cutoff): not applicable under row "
                 "sharding — no [S, N, N, *] tensor exists on any rank (row blocks are reduced on the device that holds them; rank 0 assembles PAE / PDE "
                 "rows on its host) — named here and in the census (conf_offload=stock_on_na), nothing moved")   # never a refusal: the shipped preset keeps it on
    aux_out = {}
    out_dtype = output["atom_positions_predicted"].dtype
    si = output["si_trunk"]
    z_loc = output["zij_trunk"]                                               # [1, 1, n_loc, N, C] (_rollout's sample dim) or [1, n_loc, N, C]
    N = lay.N
    check_trunk_shard(z_loc, lay)
    if si_input.numel() != N * int(si_input.shape[-1]) or si.numel() != N * int(si.shape[-1]):
        raise ConfidenceRefused(f"refused: si_input {tuple(si_input.shape)} / si_trunk {tuple(si.shape)} carry a batch larger than 1")
    atom_positions_predicted = output["atom_positions_predicted"].to(dtype=si.dtype)   # [1, S, N_atom, 3]
    si_input = si_input.detach().reshape(N, -1).clone()
    si = si.detach().reshape(N, -1).clone()
    S = int(atom_positions_predicted.shape[1])
    if ztrunk is None:                                                       # a one-pass plan over this call: last use unless the caller says its trunk rows are read again (reuse_ztrunk=False)
        last_use = reuse_ztrunk is None or bool(reuse_ztrunk)
        zplan, zi, device_needed_after = ztrunk_plan(z_loc, lay, passes=1, log=comm.log if comm.verbose else None), 0, not last_use
    else:
        zplan, zi, last_use, device_needed_after = ztrunk
    if contact_loc is None:                                                  # the distogram head reads the DEVICE shard: its rows first, before the plan releases or overwrites it
        contact_loc = trunk_contact_rows(ah, zplan.source(), lay, config, S)
    if not use_zij_trunk_embedding:
        raise ConfidenceRefused("refused: use_zij_trunk_embedding=False (the training-time draw) under the tp line — inference embeds the trunk pair representation")
    ztrunk_after = z_loc[..., :0, :, :]                                     # the zero-row stand-in output["zij_trunk"] becomes once the plan has consumed the shard (a zero-element view: valid on a released storage)
    z2, embed_inplace = zplan.begin(zi, lead_out=(S,), last_use=last_use, out_dtype=z_loc.dtype)   # parked: the shard's device storage is released HERE (if not at roll-out entry), before the pass's pair representation exists
    C.fn("evidence", "record_schedule")(conf_samples=S)
    atom_positions_predicted = atom_positions_predicted.detach().clone()
    token_mask = batch["token_mask"]
    tm = token_mask.reshape(N).to(dtype=z2.dtype)
    pair_mask_loc = tm[lay.r0:lay.r1, None] * tm[None, :]                    # [n_loc, N]
    repr_x_pred, _ = _hm.get_token_representative_atoms(batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"])   # head_modules.py:183-187 (the representative-atom mask is discarded; the token mask is the single mask)
    num_samples = int(repr_x_pred.shape[-3])
    apply_per_sample = (not torch.is_grad_enabled() and num_samples > 1 and ah.per_sample_token_cutoff is not None
                        and int(repr_x_pred.shape[-2]) > ah.per_sample_token_cutoff)   # head_modules.py:188-193: stock's per-sample pairformer / PAE / PDE loops (a memory schedule, same statements)
    C.fn("evidence", "record_schedule")(conf_per_sample=int(apply_per_sample))   # the sharded head is per sample and per row block at every size: the switch changes nothing here, named
    S = num_samples
    single_mask = token_mask.reshape(1, N).expand(S, N).to(dtype=z2.dtype)  # head_modules.py:206 single_mask=token_mask; reshape_inputs expands it over the S samples (prediction_heads.py:207)
    si, zc = pairformer_embedding_rows(ah.pairformer_embedding, si_input, si, z2, repr_x_pred.reshape(S, N, 3), single_mask, pair_mask_loc, lay,
                                       chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans, inplace=embed_inplace, pairformer_dtype=pairformer_dtype,
                                       after_embed=lambda: zplan.retire(zi), **flags)
    si = si.reshape(*atom_positions_predicted.shape[:-2], N, si.shape[-1])   # [1, S, N, c_s] (stock batch_dims)
    # the confidence-pairformer -> heads SEAM: the A-block row mirror is not read by the heads and torch's cached-free pinned blocks are
    # dead weight; both leave the host BEFORE the PDE/PAE row tensors and rank 0's N x N pinned host matrices are allocated (every rank; the
    # z_trunk park stays: the heads read the trunk rows through the last pass). ROWPAIR_POOL_SHRINK=0 keeps both through the heads.
    C.fn("heads", "pinned_shrink")(comm.log, plan=zplan)
    C.mark("confidence")
    max_atom_per_token_mask = _hm.broadcast_token_feat_to_atoms(token_mask=token_mask, num_atoms_per_token=batch["num_atoms_per_token"], token_feat=token_mask,
                                                                max_num_atoms_per_token=ah.max_atoms_per_token)
    max_atom_per_token_mask = max_atom_per_token_mask.expand((*atom_positions_predicted.shape[:-2], -1))
    aux_out["plddt_logits"] = ah.plddt(s=si, max_atom_per_token_mask=max_atom_per_token_mask)
    aux_out["experimentally_resolved_logits"] = ah.experimentally_resolved(si, max_atom_per_token_mask)
    conf = confidence_scores_rows(ah, batch, zc, None, aux_out["plddt_logits"].to(dtype=out_dtype), output["atom_positions_predicted"], lay, config, contact_loc=contact_loc)
    del zc, z2
    zplan.end(zi, device_needed_next=bool(device_needed_after))            # parked: restores the device storage only when a later statement reads the shard whole; last pass: host copy dropped
    if zplan.consumed:                                                       # in place or released-not-restored: output["zij_trunk"] no longer holds trunk rows (keep write_latent_outputs off)
        output["zij_trunk"] = ztrunk_after
    aux_out = {k: v.to(dtype=out_dtype) for k, v in aux_out.items()}
    aux_out["_tp_confidence"] = conf
    comm.log(f"[confidence] done: ptm={float(conf['ptm'].flatten()[0]) if comm.rank == 0 and 'ptm' in conf else 'n/a'}") if comm.verbose else None
    return aux_out
