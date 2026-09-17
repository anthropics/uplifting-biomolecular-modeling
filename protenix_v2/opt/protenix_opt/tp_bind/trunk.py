"""Protenix-v2's recycling TRUNK (``Protenix.get_pairformer_output``, inference) bound onto the shared core's row-sharded trunk driver
(``opt_core.mem.rowpair.trunk.run_trunk_sharded``): the pair representation z is BORN as this rank's rows ``z[r0:r1, :, :]`` and stays a shard
through every recycling iteration — nothing ``N x N x c_z`` is ever whole on a rank; the row-block schedule, the z_init park, the RNG guard
and every collective are the core's.

What is Protenix's here (the per-(i, j) statements of ``protenix/model/protenix.py``'s trunk, written on GLOBAL rows ``[g0, g1)``):
    z_init rows      ``zinit1(s_init)[rows, None, :] + zinit2(s_init)[None, :, :]``  (``trunk.outer_sum_rows``)
                     ``+= relative_position_encoding.linear_no_bias(relp rows)``    (``relpos.relp_rows``: the lazy one-hot, a row slab only)
                     ``+= linear_no_bias_token_bond(token_bonds[rows, :, None])``    (``trunk.feature_rows``: the bond matrix may live on the host)
    recycling        ``z[rows] = z_init[rows] + linear_no_bias_z_cycle(layernorm_z_cycle(z[rows]))`` in place per row block (``recycle_shard_``);
                     with the inference MC-dropout drawn ON the recycling projection is multiplied by the P-invariant BLOCK-SEEDED keep mask
                     (a private ``torch.Generator`` per 128-row GLOBAL block seeded from ``(initial_seed, cycle, block)``, drawn at the block's
                     row count and sliced to the rows at hand: identical under any layout policy; the global RNG stream is untouched) —
                     census ``trunk_mc_dropout=block_seeded`` (a tier-2-by-mask statement: not torch's full-tensor Philox mask)
    single recycle   ``s = s_init + linear_no_bias_s(layernorm_s(s))`` (replicated)
    template         :func:`protenix_opt.tp_bind.template.template_update_` (rows, in place)
    MSA module       the carried unit's MSA seam (``ptx_tp.impl('msa').tp_msa_module``), its pair block also serves the template stack
    pairformer       the line's pairformer seam (``ptx_tp.impl('pairformer').tp_pairformer_stack``)
    distogram rows   ``compute_contact_prob(linear(z)[rows] + linear(z).T[rows])`` per row block (``heads.sym_logit_rows``: one all-to-all window
                     per block for the transpose term; fp32 softmax as stock)
Replicated by design (identical on every rank, named in the census ``trunk_bind_replicated``): ``s_inputs``, ``s_init``, ``s``, the per-token index
vectors, ``token_bonds`` (its ROWS move to the device per block). Row block: ``BLOCK_ROWS`` = 128 for init / recycle / template (an explicit
argument -> census source ``given``). The z_init shard is device-resident or host-PARKED (``ROWPAIR_PARK_ZINIT``; the carried unit's
``PTX_TP_ZINIT_RECOMPUTE`` resolving to recompute maps to the park and is printed as ``zinit_recompute->park_z_init``). Refused by name: a replicated / P == 1 layout (the engine's own trunk runs at ``--n_gpu 1``),
a live ``constraint_feature`` dict (a dense constraint pair embedding is not a row statement of this line)."""
import os
import time
from typing import Any, Mapping, Optional

import torch

from opt_core.mem.rowpair import RowpairRefused
from opt_core.mem.rowpair import heads as core_heads
from opt_core.mem.rowpair import trunk as core_trunk
from opt_core.mem.rowpair.dist import Layout, layout_here, require_sharded
from opt_core.mem.rowpair.evidence import record_schedule
from opt_core.mem.rowpair.shard import iter_row_blocks, owns_whole_storage

from . import relpos as bind_relpos
from . import template as bind_template

TAG = "protenix-opt"
BLOCK_ROWS = 128
REPLICATED = "s_inputs,s_init,s,index_vectors,token_bonds(host rows->device per block)"
ENV_OP_TIMERS = "PTX_TP_OP_TIMERS"                       # engineering switch (=1): the Pairformer stack fills per-op device-synchronised seconds (``timers`` /
                                                         # ``mem`` of tp_pairformer_stack) and this seam prints one ``TP-OPTIMERS cycle=<c> …`` line per cycle per rank
# the tp_route.BIND row this module serves (carried module -> names)
ROUTE_ROWS = {"ptx_tp.trunk": ("protenix_opt.tp_bind.trunk", (
    "get_pairformer_output_tp", "zinit_rows", "zinit_rows_block", "recycle_rows", "relp_rows", "relp_block", "contact_probs_rows"))}


# ----------------------------------------------------------------------------------------------------------------- row statements
def relp_rows(rpe, feats: Mapping[str, Any], r0: int, r1: int) -> torch.Tensor:
    """Rows ``[r0, r1)`` of the relative-position one-hot feature for ``rpe = model.relative_position_encoding`` (``r_max``, ``s_max``)."""
    return bind_relpos.relp_rows(feats, r0, r1, int(rpe.r_max), int(rpe.s_max))


def relp_block(feats: Mapping[str, Any], rpe, c0: int, c1: int) -> torch.Tensor:
    """relp rows ``[c0, c1)`` from whatever the dict carries under ``'relp'``: a rows-serving object (``.rows(c0, c1)``), a dense ``[N, N, F]``
    tensor (sliced), or absent (the row statement on the index features)."""
    relp = feats.get("relp") if hasattr(feats, "get") else None
    if relp is None:
        return relp_rows(rpe, feats, c0, c1)
    if hasattr(relp, "rows"):
        return relp.rows(c0, c1)
    return relp[..., c0:c1, :, :]


def zinit_rows_block(model, a_rows: torch.Tensor, b_all: torch.Tensor, feats: Mapping[str, Any], rpe, c0: int, c1: int, zc_rows_blk=None) -> torch.Tensor:
    """z_init rows ``[c0, c1)`` (GLOBAL): ``a[c0:c1, None, :] + b[None, :, :]``; ``+= rpe.linear(relp rows)``; ``+= token-bond rows`` (+ constraint rows)."""
    zb = core_trunk.outer_sum_rows(a_rows, b_all, c0, c1)
    zb += rpe.linear_no_bias(relp_block(feats, rpe, c0, c1))
    zb += model.linear_no_bias_token_bond(core_trunk.feature_rows(feats["token_bonds"], c0, c1, zb.device, row_dim=-2, non_blocking=True).unsqueeze(dim=-1))
    if zc_rows_blk is not None:
        zb += zc_rows_blk
    return zb


def zinit_rows(model, s_init: torch.Tensor, feats: Mapping[str, Any], layout: Layout, z_constraint_rows=None, row_block: int = BLOCK_ROWS) -> torch.Tensor:
    """This rank's z_init shard ``[R, N, c_z]``, produced per ``row_block`` rows (``trunk.init_pair_shard``)."""
    a, b, rpe, r0 = model.linear_no_bias_zinit1(s_init), model.linear_no_bias_zinit2(s_init), model.relative_position_encoding, layout.r0

    def rows_fn(g0: int, g1: int):
        return zinit_rows_block(model, a, b, feats, rpe, g0, g1, None if z_constraint_rows is None else z_constraint_rows[g0 - r0:g1 - r0])
    return core_trunk.init_pair_shard(layout, rows_fn, a, rows=row_block)


def recycle_rows(model, z_sh: torch.Tensor) -> torch.Tensor:
    """``linear_no_bias_z_cycle(layernorm_z_cycle(z rows))`` — row-local."""
    return model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z_sh))


def mc_dropout_rows(u_rows: torch.Tensor, g0: int, N: int, p: float, cycle: int) -> torch.Tensor:
    """The P-invariant block-seeded MC-dropout of the recycling projection rows ``u_rows [rows, N, c_z]`` = GLOBAL rows ``[g0, g0 + rows)``:
    for every 128-row global block ``k`` the rows touch, ``keep_k = rand([rows of block k, N, c_z], Generator seeded
    initial_seed % 2**62 + 1_000_003 (cycle + 1) + k) >= p`` drawn at the block's TRUE row count (``min(N, 128 (k + 1)) - 128 k``) and sliced to
    the rows at hand; ``(u.float() * keep / (1 - p)).to(u.dtype)``. The draw is a function of (seed, cycle, k) only, so the mask is identical
    under any layout policy and any row blocking."""
    rows = int(u_rows.shape[-3])
    g1 = int(g0) + rows
    out = torch.empty_like(u_rows)
    base = int(torch.initial_seed()) % (2 ** 62)
    for k in range(int(g0) // BLOCK_ROWS, (g1 - 1) // BLOCK_ROWS + 1):
        b0, b1 = BLOCK_ROWS * k, min(int(N), BLOCK_ROWS * (k + 1))
        gen = torch.Generator(device=u_rows.device)
        gen.manual_seed(base + 1_000_003 * (int(cycle) + 1) + k)
        keep = torch.rand((b1 - b0,) + tuple(u_rows.shape[-2:]), generator=gen, device=u_rows.device, dtype=torch.float32) >= p
        s0, s1 = max(int(g0), b0), min(g1, b1)
        out[..., s0 - g0:s1 - g0, :, :] = (u_rows[..., s0 - g0:s1 - g0, :, :].float() * keep[s0 - b0:s1 - b0] * (1.0 / (1.0 - p))).to(u_rows.dtype)
        del keep
    return out


def contact_probs_rows(model, z_sh: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Rows of ``compute_contact_prob(distogram_head(z))`` -> float32 ``[R, N]``: the symmetrised logits row block by row block."""
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    bins = sample_confidence.get_bin_params(model.configs.loss.distogram)
    contact = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)
    lin = model.distogram_head.linear
    out = torch.empty(tuple(z_sh.shape[:-1]), dtype=torch.float32, device=z_sh.device)
    n_bins = int(lin.weight.shape[0]) if hasattr(lin, "weight") else 64
    for i0, i1, logits in core_heads.sym_logit_rows(lin, z_sh, layout, bins=n_bins):
        out[..., i0:i1, :] = contact(distogram_logits=logits, **bins)
    return out


# ----------------------------------------------------------------------------------------------------------------- the trunk
def _say(layout: Layout):
    import ptx_tp                                                  # the carried unit's log line (rank-tagged)

    def say(msg: str) -> None:
        if layout.rank == 0:
            ptx_tp.log(msg)
    return say


def _park_zinit(layout: Layout, N: int, c_z: int, say) -> Optional[bool]:
    """``ROWPAIR_PARK_ZINIT`` decides (None) unless the unit's recompute lever resolves to recompute: then the park (named)."""
    shard_gb = layout.R * N * c_z * 2 / 1e9
    zr = os.environ.get("PTX_TP_ZINIT_RECOMPUTE", "auto").strip().lower()
    if zr == "1" or (zr == "auto" and shard_gb > float(os.environ.get("PTX_TP_ZINIT_RECOMPUTE_ABOVE_GB", "8"))):
        say(f"trunk: z shard {layout.R}x{N}x{c_z} = {shard_gb:.1f} GB/rank; zinit_recompute->park_z_init (the z_init shard is parked on the host, "
            f"the device holds one shard)")
        record_schedule(trunk_zinit_lever="zinit_recompute->park_z_init")
        return True
    say(f"trunk: z shard {layout.R}x{N}x{c_z} = {shard_gb:.1f} GB/rank; z_init shard per ROWPAIR_PARK_ZINIT; statements per {BLOCK_ROWS}-row block")
    return None


def get_pairformer_output_tp(model, input_feature_dict: Mapping[str, Any], N_cycle: int, inplace_safe: bool = True, chunk_size: Optional[int] = None,
                             mc_dropout: bool = False, layout: Optional[Layout] = None, tri_chunk="same"):
    """``Protenix.get_pairformer_output`` on this rank's rows -> ``(s_inputs, s, z_shard [R, N, c_z], layout)``."""
    import ptx_tp                                                  # the carried unit's seam registry (msa / pairformer seams)
    from ptx_tp import trunk as dispatch                            # the carried dispatcher's phase log (progress phases the line's events read)
    feats = input_feature_dict
    N = int(feats["residue_index"].shape[-1])
    layout = layout if layout is not None else layout_here(N, B=int(os.environ.get("PTX_TP_B", "128")))
    require_sharded(layout, "protenix trunk binding (get_pairformer_output_tp)")
    say = _say(layout)
    dispatch.phase("trunk_start", N=N, P=layout.P, R=layout.R, trunk_impl="rowpair", seams=ptx_tp.ledger(), reset_peak=True)

    s_inputs = model.input_embedder(feats, inplace_safe=False, chunk_size=chunk_size)              # replicated (per-token statement)
    if "constraint_feature" in feats:
        raise RowpairRefused("protenix trunk binding: input_feature_dict carries 'constraint_feature' — a dense [N, N, c_z] constraint pair embedding "
                             "is not a row statement of this line (protenix-v2 disables every constraint embedder; the unit drops the dict, "
                             "PTX_TP_DROP_CONSTRAINT=1)")
    if "constraint_feature_dropped" in feats:
        ce = model.constraint_embedder
        enabled = [n for n in ("pocket_embedder_config", "contact_embedder_config", "contact_atom_embedder_config", "substructure_embedder_config")
                   if getattr(ce, n, {}).get("enable", False)]
        if enabled:
            raise RowpairRefused(f"protenix trunk binding: constraint features were dropped but this model ENABLES {enabled}")
    s_init = model.linear_no_bias_sinit(s_inputs)
    a, b, rpe = model.linear_no_bias_zinit1(s_init), model.linear_no_bias_zinit2(s_init), model.relative_position_encoding
    p_drop = float(model.configs.mc_dropout_rate) if mc_dropout else 0.0
    if mc_dropout:
        say(f"TIER-2 NOTICE: mc_dropout at N={N} uses the P-invariant block-seeded mask (not torch's full-tensor Philox mask)")
    msa_mod = ptx_tp.impl("msa")
    pf_mod = ptx_tp.impl("pairformer")
    kw = dict(triangle_multiplicative=model.configs.triangle_multiplicative, triangle_attention=model.configs.triangle_attention,
              inplace_safe=inplace_safe, chunk_size=(chunk_size if tri_chunk == "same" else tri_chunk))
    feats_t = bind_template.slice_template_inputs_to_rows(feats, layout) if bind_template.template_embedder_is_active(model.template_embedder, feats) else None
    real_rows_fn = None
    if feats_t is not None and bind_template.REAL_MARKER in feats_t:
        from ptx_tp.template_real import rows_torch as real_rows_fn                             # the featuriser's per-token template rows statement
    state = {"cycle": 0, "blocks": [], "t0": time.time()}

    def init_rows_fn(g0: int, g1: int):
        return zinit_rows_block(model, a, b, feats, rpe, g0, g1)

    def on_cycle_start(cycle: int, is_final: bool) -> None:
        state["cycle"], state["t0"] = int(cycle), time.time()
        state["blocks"] = list(iter_row_blocks(layout, BLOCK_ROWS)) if mc_dropout else []

    def recycle_update_fn(z_rows):
        u = recycle_rows(model, z_rows)
        if not mc_dropout:
            return u
        b0, b1, g0, g1 = state["blocks"].pop(0)                                                 # the driver walks iter_row_blocks(layout, BLOCK_ROWS) in order
        if b1 - b0 != int(z_rows.shape[-3]):
            raise RowpairRefused(f"protenix trunk binding: recycle row block {b0}:{b1} does not match the block handed to the update ({int(z_rows.shape[-3])} rows)")
        return mc_dropout_rows(u, g0, layout.N, p_drop, state["cycle"])

    def template_fn(z_loc, cycle: int):
        return bind_template.template_update_(model.template_embedder, feats_t, z_loc, layout, pair_block=msa_mod.tp_pair_stack_block, add=True,
                                              rows=BLOCK_ROWS, real_rows_fn=real_rows_fn, log=say if cycle == 0 else None, **kw)

    def msa_fn(z_loc, cycle: int):
        z_new = msa_mod.tp_msa_module(model.msa_module, feats, z_loc, s_inputs, layout, **kw)
        if torch.is_tensor(z_new) and not owns_whole_storage(z_new):                                # the pair stack's storage release needs an owning shard
            owned = torch.empty_like(z_new, memory_format=torch.contiguous_format)
            owned.copy_(z_new)
            z_new = owned
        return z_new

    def single_recycle_fn(s, cycle: int):
        return s_init + model.linear_no_bias_s(model.layernorm_s(s))

    def pairstack_fn(s, z_loc, cycle: int):
        pkw = kw
        if os.environ.get(ENV_OP_TIMERS, "") == "1":                                                 # engineering: per-op seconds of the Pairformer stack (device-synchronised
            timers, mem = {}, {}                                                                     #   per op by the stack that fills them), one line per trunk cycle per rank
            pkw = dict(kw, timers=timers, mem=mem)
            t_stack = time.time()
        s, z_loc = pf_mod.tp_pairformer_stack(model.pairformer_stack, s, z_loc, layout, **pkw)
        if pkw is not kw:
            torch.cuda.synchronize()
            say("TP-OPTIMERS cycle=%d stack_s=%.2f blocks=%d %s peak_gib=%s" % (
                cycle, time.time() - t_stack, len(model.pairformer_stack.blocks), " ".join("%s=%.2f" % (k, v) for k, v in timers.items()),
                {k: round(v / 2 ** 30, 2) for k, v in mem.items()}))
        dispatch.phase("trunk_cycle_end", cycle=cycle, s=round(time.time() - state["t0"], 2))
        return s, z_loc

    record_schedule(trunk_bind="protenix_v2", trunk_mc_dropout="block_seeded" if mc_dropout else "off", trunk_bind_rows=BLOCK_ROWS,
                    trunk_bind_replicated=REPLICATED)
    out = core_trunk.run_trunk_sharded(
        layout, n_cycles=int(N_cycle), init_rows_fn=init_rows_fn, init_like=a, recycle_update_fn=recycle_update_fn, s_init=s_init,
        single_recycle_fn=single_recycle_fn, template_fn=template_fn if feats_t is not None else None, msa_fn=msa_fn, pairstack_fn=pairstack_fn,
        s_input=s_inputs, init_rows=BLOCK_ROWS, recycle_rows=BLOCK_ROWS, park_zinit=_park_zinit(layout, N, int(model.c_z), say), gather="none",
        on_cycle_start=on_cycle_start, log=say)
    say(f"trunk timings_s={out.record.get('timings_s')} park={out.record.get('park')}")
    return s_inputs, out.s, out.z, layout
