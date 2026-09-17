"""RF3's confidence head under ``--mode big --n_gpu P``: the head runs on this rank's ROWS of the trunk pair shard and its PAE / PDE logits
are consumed row block by row block — no ``[I, I, 64]`` logits tensor and no ``[I, I]`` fp32 matrix is resident on any device at any time.

RF3 (``rf3.model.layers.af3_auxiliary_heads.ConfidenceHead.forward``, shipped config ``use_af3_style_binning_and_final_layer_norms=True``,
``use_Cb_distances=False``) produces per diffusion sample ``pae_logits`` / ``pde_logits`` ``[1, I, I, 64]``; every consumer downstream
(``rf3.metrics.predicted_error.compute_ptm`` / ``ComputeIPTM``, ``rf3.utils.predicted_error.compile_af3_style_confidence_outputs``) is a
function of three ``[1, I, I]`` fp32 matrices and token-level masks:

    T[i, j]     = sum_b softmax(pae_logits[i, j, :])[b] * w[b],  w[b] = 1 / (1 + (bin_centers[b] / d0(I))^2)   (compute_ptm; ONE weight: RF3
                  normalises every TM statistic by the FULL token count I)  ->  ptm / iptm / iptm_protein_protein / iptm_protein_ligand /
                  iptm_ligand_ligand = max_i [ sum_j T[i, j] m[i, j] / (sum_j m[i, j] + 1e-6) ] for RF3's five masks m
    E_pae[i, j] = unbin_logits(pae_logits) = sum_b softmax(.)[b] * midpoints[b];  E_pde likewise  ->  chain / chain-pair mean and min PAE / PDE,
                  overall means, and the full PAE matrix RF3 writes to ``<name>_confidences.json``

so the head is bound onto ``opt_core.mem.rowpair`` as ROW statements:

    trunk shard    ``heads.ZTrunkPlan`` (the adapter's plan over the item's per-sample passes, ``rowpair._conf_plan``): per pass the trunk
    placement      shard is embedded IN PLACE (an fp32 shard at its last use: zero copies), PARKED (``ROWPAIR_CONF_PARK_ZTRUNK``: host copy,
                   device storage released BEFORE the fp32 pair input is allocated, rows served from the host; a bf16 shard under the
                   engine's autocast always takes this form: ``free_declined:dtype``) or RESIDENT (both levers off). Device bytes while a
                   pass's pair input lives: ONE shard-equivalent (512·R·I B fp32) parked / in place, that plus the trunk shard resident.

    pair input     ``heads.embed_rows``: per row block ``LN(z rows) + right[i] + left[j] + process_pred_distances(one_hot(bins(cdist rows)))``
                   (``trunk.outer_sum_rows``; RF3's ``discretize_distance_matrix``) -> the per-sample fp32 pair SHARD ``[1, R, I, c_z]``.
                   RF3's ``F.layer_norm(Z, Z.shape)`` is a GLOBAL LayerNorm (one mean, one variance over ``I * I * c_z`` elements): its two
                   moments are reduced over the shards (``dist.allreduce_`` of fp64 partial sums, two passes) and applied per row block —
                   per element the dense formula, the moment reduction order differs from the single-device kernel (``conf_ln=global``)
    pair stack     the caller's ``pair_stack_fn(z_shard [R, I, c_z], S) -> (z_shard, S)`` (the adapter's ``pairstack.pair_stack_`` over
                   ``head.pairformer``; RF3's block order — pair updates, then attention-pair-bias + transition of S on the updated pair — is
                   the driver's order)
    PAE logits     ``heads.logit_rows(predict_pae . layernorm_pae)`` -> per LOCAL row block: ``softmax`` -> the ``T`` rows of the block by
                   ``confidence.ContextReducer.pieces`` (ONE context: all I tokens, RF3's weight ``w``; ``form="block"``: ``(probs * w).sum(-1)``
                   IS compute_ptm's statement) -> compute_ptm's masked row mean ``sum_j T[i, j] m[i, j] / (sum_j m[i, j] + 1e-6)`` for RF3's
                   five masks on the block's rows (the masks' ROWS are built from ``asym_id`` / ``is_ligand``; no ``[I, I]`` mask exists) -> a
                   local ``[R, 5]`` table; and RF3's ``unbin_logits`` on the block -> local ``E_pae`` rows ``[R, I]``. Every TM statistic of
                   RF3 is ROW-SEPARABLE (a max over rows of a per-row masked mean), so ``T`` is never assembled anywhere
    PDE logits     ``heads.sym_logit_rows(predict_pde . layernorm_pde)`` (``L + L^T`` AFTER the linear = RF3's statement; the transpose term by
                   the column-slab exchange) -> ``unbin_logits`` -> local ``E_pde`` rows
    finish         the ``[R, 5]`` row means -> ``[I, 5]`` on rank 0 (``confidence.per_row_outputs_to_rank0``; ONE gather of 5 I floats) ->
                   compute_ptm's ``.max(dim=-1)`` over rows -> the five scalars; ``E_pae`` / ``E_pde`` assembled on PINNED HOST on rank 0 only
                   (``dist.gather_rows_to_rank0_host``) — the only ``[I, I]`` objects of the head, and only on the host of rank 0
    pLDDT / exp_resolved   RF3's statements on the (replicated) single representation, unchanged, on every rank
    probe          ``X_pred_L=None`` (RF3.py's recycle-0 early-stop probe; its consumer reads ``plddt_logits`` only): RF3's statement minus
                   the distance term (its own ``if X_pred_L is not None``) -> the pair stack -> pLDDT / exp_resolved; no logits rows, no
                   gathers; pair-derived keys ``None`` on every rank (``conf_probe=plddt_only``)

REPLICATED BY DESIGN (named): ``S_inputs_I``, ``S_trunk_I`` (and their global LayerNorms), the two ``[I, c_z]`` outer-sum operands, the
representative-atom coordinates ``[I, 3]``, ``asym_id`` / ``is_ligand``, the head weights, pLDDT / experimentally-resolved logits.
Per-rank memory per sample: the fp32 pair shard ``4 I^2 c_z / P`` B + ``E`` rows ``8 I^2 / P`` B + one logits row block ``rows * I * 64 * 4`` B
and its ``T`` rows / mask rows (+ the column slabs of one exchange window); rank 0 adds ``8 I^2`` B of pinned HOST. Nothing ``[I, I]`` on a device.

Refused by name (``RowpairRefused``): a P == 1 / replicated / no-group layout (the structural n_gpu=1 rule — at ``--n_gpu 1`` nothing here
runs), a head with ``use_af3_style_binning_and_final_layer_norms=False``, ``use_Cb_distances=True`` or ``layer_norm_along_feature_dimension=True``
(branches not bound; the shipped config sets none of them), a head in training mode, a core without ``dist.gather_rows_to_rank0_host``.

Numerics (tier-2 like every ``n_gpu > 1`` run): row statements are per-(i, j) maps (bitwise
iff the kernels are M-invariant); ``(probs * w).sum(-1)``, the masked row means and ``unbin_logits`` are the dense statements on row blocks (the
per-row sums run over the same I columns; the max over rows is order-free); the global LayerNorm's moments are re-associated (fp64 partial sums);
``cdist`` rows pin torch's compute path to the dense shape's.

API::

    RF3Fns(find_bin_midpoints, unbin_logits, discretize_distance_matrix)     the three RF3 free functions the statements call (passed, never imported here)
    run_confidence_sharded(head, *, z_trunk_shard, layout, S_inputs_I, S_trunk_I, X_pred_L, rep_atoms, pair_stack_fn, asym_id, is_ligand, rf3,
                           pae=(32.0, 64), pde=(32.0, 64), tm=(32.0, 64), rows=None, form="block")
        -> dict on every rank: ``plddt_logits`` / ``exp_resolved_logits`` (replicated, device) everywhere; on rank 0 also ``pae`` / ``pde``
           (``[1, I, I]`` fp32 pinned host = RF3's ``unbin_logits`` of the logits), ``tm`` = {ptm, iptm, iptm_protein_protein,
           iptm_protein_ligand, iptm_ligand_ligand: ``[1]`` fp32}; those keys are ``None`` on ranks > 0, and on every rank for the
           ``X_pred_L=None`` probe
    tm_bin_weight(find_bin_midpoints, I, device, max_distance=32.0, bin_count=64)   compute_ptm's ``denominator`` (its lines, no function in RF3)
    tm_mask_rows(asym_id, is_ligand, g0, g1)                                 rows ``[g0, g1)`` of RF3's five ``[I, I]`` masks (ComputePTM: ones; ComputeIPTM's four)
    tm_row_means(T_rows, mask_rows)                                          compute_ptm line 37 per mask on ``T`` rows ``[D, w, I]`` -> ``[D, w, 5]``
    tm_max(V)                                                                compute_ptm line 38: ``V [D, I, 5]`` -> ``{key: [D]}``
    compile_on_expected(PE, plddt_logits, pae, pde, *args, compile_fn=None, **kwargs)
        RF3's ``compile_af3_style_confidence_outputs`` VERBATIM on pre-computed expected matrices (``PE`` = the ``rf3.utils.predicted_error``
        module; its ``unbin_logits`` is rebound for the call to pass the two matrices through and to run the stock statement on pLDDT)
    metrics_from_tm(tms)                                                     ``{"ptm": {"ptm_<d>": v}, "iptm": {"iptm_<d>": v, "iptm_protein_protein_<d>": v, ...}}``
                                                                             — the keys ``ComputePTM`` / ``ComputeIPTM`` return, per sample d
    SCHEDULE census (``evidence.record_schedule``): ``conf=rows conf_finish=rowstats conf_form=<form> conf_gathers=0 conf_row_gathers=1
    conf_host_matrices=2 conf_ln=global conf_collect=host conf_probe=none|plddt_only`` (``conf_gathers`` counts ``[n_D, n_D]``-class
    device gathers: none)
"""
from __future__ import annotations

from collections import namedtuple
from typing import Callable, Optional, Sequence, Tuple

from . import _core

__all__ = ["RF3Fns", "TM_KEYS", "run_confidence_sharded", "tm_bin_weight", "tm_mask_rows", "tm_row_means", "tm_max", "compile_on_expected",
           "metrics_from_tm", "DIST_BINS", "CENSUS"]

RF3Fns = namedtuple("RF3Fns", ["find_bin_midpoints", "unbin_logits", "discretize_distance_matrix"])
TM_KEYS = ("ptm", "iptm", "iptm_protein_protein", "iptm_protein_ligand", "iptm_ligand_ligand")
# ConfidenceHead.forward, ``use_af3_style_binning_and_final_layer_norms`` branch (af3_auxiliary_heads.py:145-150): "published code is 3.25 to
# 50.75, with 39 bins" -> one_hot(num_classes=40) -> process_pred_distances = linearNoBias(40, c_z)
DIST_BINS = dict(min_distance=3.25, max_distance=50.75, num_bins=39, num_classes=40)
CENSUS = {"conf": "rows", "conf_finish": "rowstats", "conf_ln": "global", "conf_collect": "host", "conf_gathers": 0, "conf_row_gathers": 1,
          "conf_host_matrices": 2}


# ----------------------------------------------------------------------------------------------------------------------------- core access
def _rp(name: str = ""):
    """``opt_core.mem.rowpair[.name]`` through the kit's pinned-core loader."""
    return _core.load("mem.rowpair" + ("." + name if name else ""))


def _torch():
    import torch
    return torch


def _refused(msg: str):
    return _rp().RowpairRefused("tp_conf: " + msg)


# ----------------------------------------------------------------------------------------------------------------------- RF3 statements
def tm_bin_weight(find_bin_midpoints: Callable, I: int, device, max_distance: float = 32.0, bin_count: int = 64):
    """compute_ptm's weight (rf3/metrics/predicted_error.py:28-33; RF3 has no function boundary here, the lines are these):
    ``bin_centers = find_bin_midpoints(max_distance, bin_count)``; ``d0 = 1.24 (max(I, 19) - 15)^(1/3) - 1.8``; ``w = 1 / (1 + (bin_centers / d0)^2)``."""
    bin_centers = find_bin_midpoints(max_distance, bin_count, device=device)
    normalization_factor = 1.24 * (max(int(I), 19) - 15.0) ** (1 / 3) - 1.8
    denominator = 1 / (1 + (bin_centers / (normalization_factor)) ** 2)
    return denominator


def tm_mask_rows(asym_id, is_ligand, g0: int, g1: int) -> list:
    """Rows ``[g0, g1)`` (``[w, I]`` bool) of RF3's five ``[I, I]`` masks, in :data:`TM_KEYS` order: ``ComputePTM``'s ``to_calculate=None`` ->
    ``ones`` (compute_ptm:25-26) and ``ComputeIPTM.compute``'s four (predicted_error.py:96-110, operator precedence as written). The masks are
    elementwise expressions of per-token vectors, so their rows are the same expressions with the row operand sliced — no ``[I, I]`` mask exists."""
    torch = _torch()
    I = int(asym_id.shape[0])
    dev = asym_id.device
    ones = torch.ones((int(g1) - int(g0), I), dtype=torch.bool, device=dev)
    to_calculate = asym_id[None, :] != asym_id[g0:g1, None]
    protein_mask = is_ligand == 0
    ligand_mask = is_ligand == 1
    protein_protein_mask = (protein_mask[None, :] & protein_mask[g0:g1, None] * to_calculate)
    protein_ligand_mask = ((protein_mask[None, :] & ligand_mask[g0:g1, None]) | (ligand_mask[None, :] & protein_mask[g0:g1, None])) * to_calculate
    ligand_ligand_mask = ligand_mask[None, :] & ligand_mask[g0:g1, None] * to_calculate
    return [ones, to_calculate, protein_protein_mask, protein_ligand_mask, ligand_ligand_mask]


def tm_row_means(T_rows, mask_rows: Sequence) -> object:
    """compute_ptm line 37 per mask on ``T`` rows: ``(pae * to_calculate[None]).sum(dim=-1) / (to_calculate.sum(dim=-1) + 1e-6)`` with ``pae = T_rows
    [D, w, I]`` and ``to_calculate`` = the mask's rows ``[w, I]`` — per row the dense statement (the sum runs over the same I columns). ``[D, w, 5]``."""
    torch = _torch()
    cols = []
    for to_calculate in mask_rows:
        pae = (T_rows * to_calculate[None]).sum(dim=-1) / (to_calculate.sum(dim=-1) + 1e-6)
        cols.append(pae)
    return torch.stack(cols, dim=-1)


def tm_max(V) -> dict:
    """compute_ptm line 38: ``ptm = pae.max(dim=-1).values`` over the rows, per mask: ``V [D, I, 5]`` -> ``{key: [D] fp32}``."""
    out = {}
    for k, key in enumerate(TM_KEYS):
        ptm = V[..., k].max(dim=-1).values
        assert ptm.shape == (int(V.shape[0]),)
        out[key] = ptm
    return out


# --------------------------------------------------------------------------------------------------------------- the sharded head (per sample)
RETIRE_KEPT_PROBE = "kept:probe"                     # the census word of a pass whose trunk rows are read again after it (the recycle-0 early-stop probe)


def retire_after_embed(plan, pass_index: int, last_use: Optional[bool]) -> Optional[str]:
    """The trunk park's early retirement after pass ``pass_index``'s ``embed_rows`` (``heads.ZTrunkPlan.retire``): the word the
    plan answers (``pass<i>@embed`` dropped now | ``kept:not_last`` | ``nothing_parked``), or :data:`RETIRE_KEPT_PROBE` WITHOUT touching the plan when
    the caller declared the shard is read again after this pass (``last_use=False``: RF3's recycle-0 early-stop probe, ``X_pred_L=None`` — recycles
    1.. read and write the trunk shard whole after it and its pass ends ``device_needed_next=True``; a retired park there is refused by name by the
    core's ``end`` and the item is lost), or None on a core without ``retire`` (nothing recorded)."""
    if not hasattr(plan, "retire"):
        return None
    if last_use is False:
        return RETIRE_KEPT_PROBE
    return plan.retire(int(pass_index))


def run_confidence_sharded(head, *, z_trunk_shard, layout, S_inputs_I, S_trunk_I, X_pred_L, rep_atoms, pair_stack_fn: Callable,
                           asym_id, is_ligand, rf3: RF3Fns, pae: Tuple[float, int] = (32.0, 64), pde: Tuple[float, int] = (32.0, 64),
                           tm: Tuple[float, int] = (32.0, 64), rows: Optional[int] = None, form: str = "block",
                           plan, pass_index: int = 0, last_use: Optional[bool] = None, moments=None) -> dict:
    """ONE diffusion sample of ``ConfidenceHead.forward`` on this rank's rows (COLLECTIVE: every rank calls it, per sample, in the same order).

    ``head``: the ``ConfidenceHead`` module (its named sub-modules are the callables); ``z_trunk_shard``: this rank's rows ``[1, R, I, c_z]`` (or
    ``[R, I, c_z]``) of the trunk pair in the trunk's dtype — read per row block, never modified; ``layout``: the query's sharded ``Layout``;
    ``S_inputs_I [1, I, 449]``, ``S_trunk_I [1, I, c_s]``, ``X_pred_L [1, n_atoms, 3]``, ``rep_atoms [I]``, ``asym_id [I]``, ``is_ligand [I]``:
    replicated (identical on every rank — the adapter's entry sync); ``pair_stack_fn(z_shard [R, I, c_z] fp32, S [1, I, c_s] fp32) -> (z_shard, S)``:
    the head's pairformer blocks on the pre-sharded rows (in place allowed); ``rf3``: :class:`RF3Fns`; ``pae`` / ``pde``: ``(max_value, n_bins)`` of
    the confidence-loss config (``unbin_logits``' arguments); ``tm``: compute_ptm's ``(max_distance, bin_count)``; ``rows``: the head row block
    (``heads.conf_rows``: given -> ``ROWPAIR_CONF_ROWS`` -> 128 under the int32 rule); ``form``: the reducer statement form. ``X_pred_L=None`` is
    the early-stop probe (pLDDT / exp_resolved only). ``plan``: the item's ``heads.ZTrunkPlan`` over the trunk shard and
    ``pass_index`` / ``last_use``: this call is pass ``pass_index`` of it — the trunk shard's whole-shard read (the global LayerNorm's two
    moments) runs FIRST, then ``plan.begin`` places the shard for the pass (in place: the fp32 pair input overwrites the shard's own storage;
    parked: host copy + device storage RELEASED; resident) and only then is the pass's ``[1, R, I, c_z]`` fp32 pair input allocated — so a
    parked or in-place pass holds ONE shard-equivalent on the device, not two; the CALLER ends the pass (``plan.end``) once this returns (the
    pair input is dead then). ``moments``: the ``(mean, rstd)`` an earlier pass of the same plan measured (the trunk shard is the same tensor for
    every pass; a parked / consumed shard has no device rows to measure) — returned as ``out["moments"]``. Returns the dict of the module
    docstring (pair-derived keys on rank 0, ``None`` on ranks > 0)."""
    torch = _torch()
    F = torch.nn.functional
    RP = _rp()
    D = _rp("dist")
    heads = _rp("heads")
    conf = _rp("confidence")
    trunk = _rp("trunk")
    evidence = _rp("evidence")
    lay = D.require_sharded(layout, "tp_conf.run_confidence_sharded")
    gather_host = getattr(D, "gather_rows_to_rank0_host", None)
    if gather_host is None:
        raise _refused("opt_core.mem.rowpair.dist.gather_rows_to_rank0_host is absent (opt_core >= 0.4.3 rc2 required: the E_pae / E_pde matrices "
                       "are assembled on pinned host by the core's one output-assembly primitive)")
    if not getattr(head, "use_af3_style_binning_and_final_layer_norms", False):
        raise _refused("ConfidenceHead.use_af3_style_binning_and_final_layer_norms=False is not bound under n_gpu>1 (the pre-linear PDE "
                       "symmetrisation + residual branch); the shipped config sets it True")
    if getattr(head, "use_Cb_distances", False):
        raise _refused("ConfidenceHead.use_Cb_distances=True is not bound under n_gpu>1 (calc_Cb_distances rows); the shipped config sets it False")
    if getattr(head, "training", False):
        raise _refused("ConfidenceHead in training mode (row dropout) is not bound under n_gpu>1; inference only")
    probe = X_pred_L is None                                                      # RF3.py's recycle-0 early-stop probe: the head without coordinates;
    if form not in ("block", "sub"):                                              # its consumer reads plddt_logits only (should_early_stop_by_mean_plddt)
        raise _refused(f"form={form!r} not in ('block', 'sub')")
    zs = tuple(int(x) for x in z_trunk_shard.shape)                               # METADATA only — the shard's storage may already be released (parked by an
    if len(zs) not in (3, 4) or (len(zs) == 4 and zs[0] != 1) or zs[-3] != lay.R or zs[-2] != lay.N:   # earlier pass or at the roll-out entry): no view of it is taken here
        raise _refused(f"z_trunk_shard {zs} is not this rank's rows [R={lay.R}, I={lay.N}, c_z] (or [1, R, I, c_z]) of layout {lay!r}")
    P, rank = D.world()
    I, R, C = lay.N, lay.R, zs[-1]
    dev = z_trunk_shard.device
    if getattr(head, "layer_norm_along_feature_dimension", False):
        raise _refused("ConfidenceHead.layer_norm_along_feature_dimension=True is not bound under n_gpu>1 (the stock branch itself passes an "
                       "int as normalized_shape and raises); the shipped config leaves it False (the global LayerNorm)")

    with torch.no_grad():
        # ---- af3_auxiliary_heads.py:97-118 — stopgrad + fp32; the single-track global LayerNorms are replicated statements, verbatim
        S_trunk_I = S_trunk_I.detach().float()
        X_pred_L = X_pred_L.detach().float() if not probe else None
        S_inputs_I = S_inputs_I.detach().float()
        S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=tuple(S_trunk_I.shape))
        S_inputs_I = F.layer_norm(S_inputs_I, normalized_shape=tuple(S_inputs_I.shape))
        if moments is None:                                                       # F.layer_norm(Z, Z.shape): ONE mean / variance over I*I*c_z — the
            mean, rstd = _global_ln_moments(plan.source(), lay, eps=1e-5)             # whole-shard read in row blocks, BEFORE the plan takes the device tensor
                                                                                       # away (a shard parked at the roll-out entry serves the blocks from the
                                                                                       # host; an embedded / dropped shard is refused by name by source())
        else:
            mean, rstd = moments

        def ln_rows(x):                                                           # per element the dense formula (x - mean) * rstd
            return (x - mean) * rstd
        # ---- :121-127 — the two [1, I, c_z] operands of right[i] + left[j] (replicated, tiny)
        right = head.process_s_inputs_right(S_inputs_I)
        left = head.process_s_inputs_left(S_inputs_I)
        # ---- :133-134 — representative-atom coordinates (replicated); cdist rows pin the compute path torch takes on the dense [I, I] shape
        X_rep = X_pred_L.index_select(1, rep_atoms) if not probe else None
        cmode = "use_mm_for_euclid_dist" if I > 25 else "donot_use_mm_for_euclid_dist"
        ppd = head.process_pred_distances
        discretize = rf3.discretize_distance_matrix

        def embed_fn(blk, g0, g1):                                                # blk: this rank's trunk rows g0:g1 as the plan's source serves them —
            if blk.dim() == 3:                                                    # [rows, I, c_z] from RF3's 3-D shard (resident, parked or in place) or
                blk = blk.unsqueeze(0)                                            # [1, rows, I, c_z]; the pass's pair input is [1, R, I, c_z] (lead_out=(1,)),
            z = ln_rows(blk.float())                                              # so the block is embedded 4-D whatever the single reps' rank (RF3's are [I, c])
            z = z + trunk.outer_sum_rows(right, left, g0, g1)                     # right.unsqueeze(-2) + left.unsqueeze(-3), rows g0:g1
            if probe:                                                             # :132 `if X_pred_L is not None:` — the probe skips the distance term
                return z
            dist = torch.cdist(X_rep[:, g0:g1], X_rep, compute_mode=cmode)         # rows of torch.cdist(X_rep, X_rep)
            dist_one_hot = F.one_hot(discretize(dist, min_distance=DIST_BINS["min_distance"], max_distance=DIST_BINS["max_distance"],
                                                num_bins=DIST_BINS["num_bins"]), num_classes=DIST_BINS["num_classes"])
            return z + ppd(dist_one_hot.float())

        # ---- the trunk shard's placement for this pass (heads.ZTrunkPlan: in place | parked | resident), decided in the core; a park copies
        #      the shard out and RELEASES its device storage in here, before the pass's pair input exists
        src, inplace = plan.begin(int(pass_index), lead_out=(1,), last_use=last_use, out_dtype=torch.float32)   # in place | parked (device storage released HERE) | resident
        ztrunk_word = plan.words[int(pass_index)]
        if inplace:                                                               # an fp32 trunk shard, last use: the pair input IS its storage
            z_conf = heads.embed_rows(embed_fn, src, lay, rows=rows, lead_out=(1,), inplace=True, bins=DIST_BINS["num_classes"], out_dtype=torch.float32)
        else:                                                                     # the per-sample fp32 pair SHARD (never [1, I, I, c_z]), allocated
            z_conf = torch.empty((1, R, I, C), dtype=torch.float32, device=dev)  # after begin(): beside a RELEASED (parked) or a resident trunk shard
            heads.embed_rows(embed_fn, src, lay, rows=rows, lead_out=(1,), out=z_conf, bins=DIST_BINS["num_classes"], out_dtype=torch.float32)
        del src
        word = retire_after_embed(plan, int(pass_index), last_use)                # the trunk rows' last reader of the LAST sample pass ran —
        if word is not None:                                                      # its park retires here, BEFORE this pass's pair stack (u pinned host released; a no-op by
            evidence.record_schedule(conf_ztrunk_retire=f"{int(pass_index)}:{word}")   # word on non-last passes / in-place / resident forms; NEVER on the early-stop probe)
        # ---- :182-183 — the head's pairformer blocks on the pre-sharded rows (the adapter's driver; s carried, replicated); in place on the
        #      pair input's own storage (pairstack reuse_storage) — a driver that answers a fresh tensor is named (conf_pairstack=copy)
        zc0 = z_conf[0]
        z_rows, S_trunk_I = pair_stack_fn(zc0, S_trunk_I)
        if z_rows is not zc0:                                                     # the driver answered a fresh tensor: a second shard-sized buffer — refused by name
            raise _refused("conf_pairstack=copy: the confidence pair stack returned a tensor other than the pair input it was handed (the line's driver "
                           "runs in place on the pair input's storage, pairstack reuse_storage)")
        evidence.record_schedule(conf_pairstack="inplace", conf_ztrunk_pass=f"{int(pass_index)}:{ztrunk_word}")
        del z_conf, zc0
        z4 = z_rows.unsqueeze(0) if z_rows.dim() == 3 else z_rows                # [1, R, I, c_z]
        # ---- :208-211 — pLDDT / experimentally-resolved on the replicated single representation, verbatim
        plddt_logits = head.predict_plddt(head.layernorm_plddt(S_trunk_I))
        exp_resolved_logits = head.predict_exp_resolved(head.layernorm_exp_resolved(S_trunk_I))
        out = {"plddt_logits": plddt_logits, "exp_resolved_logits": exp_resolved_logits, "pae": None, "pde": None, "tm": None, "moments": (mean, rstd)}
        if probe:                                                                 # the early-stop probe reads pLDDT only: no logits rows, no gathers
            del z4, z_rows
            evidence.record_schedule(conf_form=form, conf_probe="plddt_only", **CENSUS)
            return out
        # ---- :207 — PAE logits per LOCAL row block -> softmax -> reducer (compute_ptm's statement) and unbin_logits (compile's statement)
        w = tm_bin_weight(rf3.find_bin_midpoints, I, dev, float(tm[0]), int(tm[1]))
        ctx = conf.Context("rf3_tm", N_d=I, w=w)                                 # ONE context: every token, RF3's weight
        red = conf.ContextReducer.for_layout([ctx], lay, form=form)             # its STATEMENT (pieces) is used; T is never accumulated
        asym_d, lig_d = asym_id.to(dev), is_ligand.to(dev)
        pae_fn = lambda zr: head.predict_pae(head.layernorm_pae(zr))  # noqa: E731
        pde_fn = lambda zr: head.predict_pde(head.layernorm_pde(zr))  # noqa: E731
        E_pae_loc = torch.empty((R, I), dtype=torch.float32, device=dev)
        E_pde_loc = torch.empty((R, I), dtype=torch.float32, device=dev)
        V_loc = torch.empty((R, len(TM_KEYS)), dtype=torch.float32, device=dev)  # per LOCAL row: the five masked row means
        r0 = lay.r0
        for i0, i1, lg in heads.logit_rows(pae_fn, z4, lay, rows=rows, bins=int(pae[1])):
            probs = torch.nn.Softmax(dim=-1)(lg).detach().float()               # compute_ptm:31
            for _k, _rows, T_rows in red.pieces(i0, i1, probs):                  # compute_ptm:34-36 on the block: (probs * w).sum(-1) -> [1, w, I]
                V_loc[i0:i1] = tm_row_means(T_rows, tm_mask_rows(asym_d, lig_d, r0 + i0, r0 + i1))[0]
                del T_rows
            del probs
            E_pae_loc[i0:i1] = rf3.unbin_logits(lg.permute(0, 3, 1, 2).float(), float(pae[0]), int(pae[1]))[0]   # compile:272-276
            del lg
        # ---- :203-205 — PDE = L + L^T after the linear (the symmetrised head; COLLECTIVE per exchange window)
        for i0, i1, lg in heads.sym_logit_rows(pde_fn, z4, lay, rows=rows, bins=int(pde[1])):
            E_pde_loc[i0:i1] = rf3.unbin_logits(lg.permute(0, 3, 1, 2).float(), float(pde[0]), int(pde[1]))[0]   # compile:277-281
            del lg
        del z4, z_rows
        # ---- finish (COLLECTIVE): the [R, 5] row means -> [I, 5] on rank 0 -> compute_ptm:38 max over rows; E rows -> rank 0 pinned host
        V = conf.per_row_outputs_to_rank0(V_loc, lay)
        del V_loc
        pae_host = gather_host(E_pae_loc, lay)
        del E_pae_loc
        pde_host = gather_host(E_pde_loc, lay)
        del E_pde_loc
    evidence.record_schedule(conf_form=form, conf_probe="none", **CENSUS)
    if rank == 0:
        out.update(pae=pae_host.unsqueeze(0), pde=pde_host.unsqueeze(0), tm=tm_max(V.unsqueeze(0)))
    return out


def _global_ln_moments(z_src, lay, eps: float = 1e-5):
    """Mean and ``rstd = 1/sqrt(var + eps)`` of the WHOLE pair tensor whose rows this rank holds (``F.layer_norm(Z, Z.shape)``'s two moments,
    biased variance): two passes over the local row blocks with fp64 partial sums, each all-reduced (``dist.allreduce_``) — identical on every
    rank. ``z_src`` is the shard tensor or a parked source serving ``.zrows(i0, i1)`` (the same rows, hence the same moments). Returned as fp32
    0-dim tensors on the shard's device."""
    torch = _torch()
    D = _rp("dist")
    shard = _rp("shard")
    heads = _rp("heads")
    n = 1
    for s in tuple(z_src.shape[:-3]) + (lay.N, lay.N, int(z_src.shape[-1])):
        n *= int(s)
    rb, _ = heads.conf_rows(lay.N, int(z_src.shape[-1]), None, 1, record=False)
    rows_of = (lambda i0, i1: z_src.zrows(i0, i1)) if hasattr(z_src, "zrows") else (lambda i0, i1: z_src[..., i0:i1, :, :])
    acc = torch.zeros((1,), dtype=torch.float64, device=z_src.device)
    for i0, i1, _g0, _g1 in shard.iter_row_blocks(lay, rb, unit=1):
        acc += rows_of(i0, i1).float().sum(dtype=torch.float64)
    D.allreduce_(acc, "sum")
    mean64 = acc[0] / n
    mean = mean64.to(torch.float32)
    acc2 = torch.zeros((1,), dtype=torch.float64, device=z_src.device)
    for i0, i1, _g0, _g1 in shard.iter_row_blocks(lay, rb, unit=1):
        d = rows_of(i0, i1).to(torch.float32, copy=True).sub_(mean)             # a NEW fp32 block (never in place on the shard: .float() of an fp32 shard IS the shard)
        acc2 += (d * d).sum(dtype=torch.float64)
        del d
    D.allreduce_(acc2, "sum")
    var64 = acc2[0] / n
    rstd = (1.0 / torch.sqrt(var64 + eps)).to(torch.float32)
    return mean, rstd


# ------------------------------------------------------------------------------------------------------------------ the consumers (rank 0)
class _Expected(object):
    """A pre-computed expected-value matrix handed to RF3's compile in place of logits: ``.permute(...).float()`` (what the caller applies before
    ``unbin_logits``) is itself; the rebound ``unbin_logits`` returns ``.value``."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def permute(self, *dims):
        return self

    def float(self):
        return self

    @property
    def device(self):
        return self.value.device

    @property
    def shape(self):
        return self.value.shape


def compile_on_expected(PE, plddt_logits, pae, pde, *args, compile_fn: Optional[Callable] = None, **kwargs):
    """``compile_af3_style_confidence_outputs(plddt_logits, pae_logits, pde_logits, ...)`` of module ``PE`` (``rf3.utils.predicted_error``) run
    VERBATIM with the expected PAE / PDE matrices ``[D, I, I]`` (host or device) in place of the logits: ``PE.unbin_logits`` is rebound for the
    duration of the call to pass an :class:`_Expected` through and to run the stock statement on anything else (pLDDT). ``compile_fn``: the
    stock function object when ``PE.compile_af3_style_confidence_outputs`` has itself been re-pointed by a kit lever."""
    stock_unbin = PE.unbin_logits
    fn = compile_fn if compile_fn is not None else PE.compile_af3_style_confidence_outputs

    def unbin_logits(logits, max_distance, num_bins):
        if isinstance(logits, _Expected):
            return logits.value
        return stock_unbin(logits, max_distance, num_bins)

    PE.unbin_logits = unbin_logits
    try:
        return fn(plddt_logits, _Expected(pae), _Expected(pde), *args, **kwargs)
    finally:
        PE.unbin_logits = stock_unbin


def metrics_from_tm(tms: Sequence[dict]) -> dict:
    """``ComputePTM.compute`` / ``ComputeIPTM.compute``'s return dicts from the per-sample ``tm`` dicts of :func:`run_confidence_sharded` (sample d =
    call order): ``{"ptm": {"ptm_<d>": float}, "iptm": {"iptm_<d>": float, "iptm_protein_protein_<d>": float, "iptm_protein_ligand_<d>": float,
    "iptm_ligand_ligand_<d>": float}}``."""
    out = {"ptm": {}, "iptm": {}}
    for d, tm in enumerate(tms):
        for key in TM_KEYS:
            v = tm[key]
            out["ptm" if key == "ptm" else "iptm"][f"{key}_{d}"] = float(v.reshape(-1)[0]) if hasattr(v, "reshape") else float(v)
    return out
