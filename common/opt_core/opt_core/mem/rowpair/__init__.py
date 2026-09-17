"""opt_core.mem.rowpair — the row-sharded pair stack: tensor parallelism for a torch pair-representation model's memory mode (``--mode big
--n_gpu P``). The pair representation ``z[*, N, N, C]`` is ROW-SHARDED over the P GPUs of one node (rank q holds rows ``[r0_q, r1_q)`` of
the row partition :class:`dist.Layout`) from the statement that creates it to the last statement that reads it: pair init / relative
position / token bonds, recycling, the template embedder, the MSA module, the pair stack, the distogram / confidence heads and the
diffusion conditioning run on this rank's rows; nothing ``N x N x C`` is whole on a rank. REPLICATED BY DESIGN (named per module and in
the schedule census): the single representation ``s``, the MSA representation ``m``, per-token features, atom tensors and the diffusion
module's atom attention; large read-once input features may stay on the host (:mod:`.msa_host`). Every rank runs the engine's own control
flow on its rows; engine sub-modules enter as callables. The INSTALL (which engine module is rebound onto which primitive, the kit's
``--n_gpu`` flag, its mode table row) is the kit's adapter; this package holds mechanism only — no engine name, no engine import, no
lever value.

Layers (torch and torch.distributed are reached through :mod:`._torch` lazily: importing this package imports nothing heavy)::

    rowpair             refusal words (:func:`refuse_unless_big`, :func:`refuse_unless_visible`), constants, :class:`RowpairRefused`
    rowpair.launch      the kit process spawns its own P workers (users never type torchrun): :func:`launch.run_sharded` (an in-process
                        entry callable per rank, ``torch.multiprocessing`` start method spawn) and :func:`launch.run_rank_processes` (one
                        command line per rank); NCCL/gloo group from the environment; rank <-> device binding; rank-0 result return;
                        ANY rank failure -> every rank torn down -> :class:`launch.RankFailed` (the kit exits non-zero, naming the event)
    rowpair.dist        L1 data movement (bit-preserving, no arithmetic on payloads): :class:`dist.Layout` (the P-invariant row grid),
                        ``init_from_env``, ``all_gather_rows``, ``gather_rows_to_rank0``, ``transpose_shards`` / ``transpose_blocks``
                        (all-to-all), ``alltoall_window``, ``ring_blocks`` (isend/irecv ring), ``broadcast_obj``, ``allreduce_checksum``
    rowpair.shard       ``shard_rows`` / ``unshard_rows`` of a whole tensor, ``local_rows`` (a row-local sub-module applied to the shard
                        in row blocks — composes :mod:`opt_core.mem.torch_rowchunk`)
    rowpair.trimul      L2 statements: the sharded triangle-multiplication contraction, outgoing (``x_ij = sum_k a_ik b_jk``: b row slabs
                        stream around the ring, full-k local matmul per tile, sub-chunked) and incoming (``x_ij = sum_k a_ki b_kj``: column
                        operands assembled by all-to-all windows, same tile loop) — projections / gates / epilogue are the caller's callables
    rowpair.triatt      L2 statements: triangle attention starting node (row-local given the all-gathered ``[N, N, H]`` triangle bias; query
                        row sub-blocks bound the logits transient) and ending node (the same statement on the transposed shard); the
                        attention kernel is the caller's callable (stock, SDPA, a fused kernel of the F1 family ...)
    rowpair.transition  row-local statements with no communication: pair transition / LayerNorm / dropout-free residual adds in row
                        blocks, outer-product-mean output rows (``a[:, rows] x b[:, all]``), attention-pair-bias with local query rows
    rowpair.confidence  head: ContextReducer (the one row-piece / gather mechanism of per-context TM matrices; RowBlockReducer = the
                        confidence statistic table on it: pTM/ipTM/per-chain/has_frame, exact | rowsum finish),
                        gather_single_rows / per_row_outputs_to_rank0
    rowpair.heads       L2: distogram / PAE / PDE logits and the confidence head's pair input produced row block by row block
                        (conf_rows, logit_rows, sym_logit_rows, embed_rows) — never [N, N, bins] / [S, N, N, C] whole
    rowpair.frames      atoms: row-blocked atom-pair statements of the confidence tail (nearest_atoms_rows, count_pairs_within) —
                        replicated by design, no [N_atom, N_atom]
    rowpair.ring        L1: p2p distributed transposes of z_loc[*, n_loc, N, C] (all-at-once | peer-streamed | in place | banded) and
                        the sub-block ring_pass — bit-preserving
    rowpair.trunk       L2 schedules + driver: the pair representation BORN sharded (init_pair_shard over row statements), recycling in
                        place (recycle_shard_), z_init parked on host (ShardPark), replication guards, run_trunk_sharded -> TrunkOut
    rowpair.rng         L2: rows [g0, g1) of a stock full-tensor CUDA Philox draw (trunc_normal_rows / trunc_normal_shard / dropout_rows),
                        bit-identical to the single-device call, generator left where the stock call leaves it
    rowpair.msa         L2: the MSA module on a sharded z, m replicated or token-sharded (m_layout=) — OPM output rows (opm_rows_budgeted), pair-weighted
                        averaging with local query-row logits (pwa_bias_rows / pwa_rows), msa_transition_rows, msa_block_/module_sharded
    rowpair.msa_host    placement: host-resident input features (raw MSA, token bonds): park_features / HostTensor / rows_to_device /
                        sync_host_features_ / residency census
    rowpair.pairstack   L2 driver: PairBlockFns / bind / pair_block_ / pair_stack_ / transition_update_ / mask_transposed — one pair
                        block or a stack of them on a PRE-SHARDED z (orientation of the ending attention owned here; never gathers)
    rowpair.template    L2: the template embedder on rows (slice_template_inputs_to_rows, TemplatePairRows, slot_census,
                        template_slot_groups, template_embed_rows) — no [T, N, N, *] on a rank
    rowpair.diffusion   L2: diffusion conditioning rows once per roll-out (pair_cond_rows), per-block pair bias rows (+ cache), the
                        diffusion transformer with local query rows (dit_block_sharded), the atom-attention band (band_plan /
                        pair_band_rows / band_lookup), noise replication (sync_replicated) — atoms replicated by design
    rowpair.sampler_hook L0: the diffusion sampler hook protocol (ROWPAIR_SAMPLER_HOOK; identical on every rank; no torch at import)
    rowpair.bcast       inputs: rank-0 featurisation + broadcast of a nested tensor dict, sync_tensordict_from_rank0, assert_replicated
    rowpair.rankdata    inputs: the ranks' input contract — ONE hash seed for every rank interpreter of a launch (ranks_env, layered on by
                        launch.rank_env), the cross-rank feature digest gate (feature_digest, ranks_agree, assert_ranks_agree) and rank-0
                        featurisation (broadcast_features: store rendezvous, status word, meta, tensors, receipt check)
    rowpair.ckpt        resume: per-rank shards + rank-0 replicated dict + RNG under a manifest written last; TrunkCheckpointer
    rowpair.census      measurement (opt-in, OPT_CORE_TP_CENSUS=1): per-stage CUDA memory census lines + stdlib readers (a + b/P fit)
    rowpair.evidence    ACTIVE/EXIT fields (n_gpu=, sharding=), per-rank peak fields, the family's LEVER line, the schedule census,
                        emit_rank0
    opt_core.testing.run_ranks   the CPU launcher: P rank-threads of one process on the ``threaded`` comm backend

Contract:
  * P == 1 IS THE ENGINE'S SINGLE-GPU PATH. ``run_sharded(1, entry, ...)`` calls ``entry`` in this process: no process group, no worker,
    no environment change; every L1 primitive given a P == 1 layout (or no initialised group) is the equality, and every L2 statement /
    schedule / driver refuses a P == 1 layout by name — the kit's P=1 big bytes do not change by installing this package.
  * REFUSALS ARE BY NAME (:class:`RowpairRefused`, a :class:`opt_core.mem.MemLeverRefused`): ``n_gpu > 1`` outside the memory mode
    (:data:`REFUSE_MODE`), fewer visible GPUs than ``n_gpu`` (:data:`REFUSE_VISIBLE`), a layout whose row grid leaves a rank with zero rows
    or falls back to replication (the shard plan is refused, never silently unsharded), a shape outside a primitive's domain. The kit maps
    a refusal to its NOT ACTIVE line and a non-zero exit; nothing here converts a refusal into a stock run.
  * NUMERICS. L1 is pure data movement (bit-exact). L2 statements keep every reduction WHOLE on one rank (the contraction over k of the
    triangle multiplication, the softmax over keys of the triangle attention, LayerNorm statistics, the expected values of the confidence
    bins): sharding changes only the M (row count) of each kernel launch and the tile schedule, never splits a sum across ranks — so at
    fixed P a sharded statement equals the dense statement bit for bit WHEN the stack's kernels are M-invariant (fp32 on CPU always is;
    cuBLAS / cuDNN heuristics may pick kernels by M). The cross-rank scalar aggregates of the confidence head (pTM / ipTM sums over row
    blocks owned by different ranks) DO reorder floating-point sums: the reducer's ``exact`` finish gathers each context's matrix to rank 0 and applies the dense
    statements on same-shape tensors (same order -> bit-exact when the per-element steps are shape-invariant); its ``rowsum`` finish reorders
    sums (|diff| ~ 1e-7, a STATED tolerance class, never asserted bit-exact). Hence the mode rule the
    the kit enforces with :func:`refuse_unless_big`: ``--n_gpu > 1`` is a resource axis of the memory mode (fast-class numerics, tested as
    a band with n >= 2 draws), refused under ``exact`` and ``fast``.
  * EVIDENCE. One activation-evidence line per lever per arm: the kit's parent process prints its ACTIVE / EXIT lines with
    :func:`evidence.fields`; worker ranks > 0 write their lines to per-rank logs (:mod:`launch`), rank 0's reach the arm's stderr.
  * Card-generic: device count, capability and memory are read from the runtime (``torch.cuda``), never assumed; block sizes and byte
    budgets derive from the device's actual memory (:mod:`opt_core.mem.budget`).
"""
from __future__ import annotations

from typing import Optional

from ..ngpu import (MEMORY_MODE, REFUSE_MODE, REFUSE_VISIBLE, NGpuRefused, active_fields, active_pairs, check_n_gpu, refuse_unless_big,
                    sharding_value, visible_refusal,
                    TP_LEVER, CERTIFIED_SM, TP_ARCH_NOTE)
from ..ngpu import refuse_unless_visible as _refuse_unless_visible_count

__all__ = ["RowpairRefused", "NGpuRefused", "SHARDING", "LEVER", "SUB_LEVERS", "sub_strategy", "CERTIFIED_SM", "MEMORY_MODE", "REFUSE_MODE", "REFUSE_VISIBLE", "refuse_unless_big",
           "refuse_unless_visible", "visible_gpus", "check_n_gpu", "active_fields", "active_pairs", "sharding_value", "visible_refusal"]

SHARDING = "rowpair"                     # this package's scheme name in ``sharding=`` (opt_core.mem.ngpu.SCHEMES)
LEVER = TP_LEVER                         # "F7.tensor_parallel": the strategy id of row-sharded pair TP (opt_core.mem.ngpu declares it)
SUB_LEVERS = {                           # impl-level sub-levers of the torch stack: key -> the strategy id its LEVER line names
    "trimul_ring": "F7.tp_trimul_ring_contract",        # ring-streamed, sub-chunked full-k contraction
    "triatt_bias_gather": "F7.tp_triatt_bias_gather",
    "a2a_transpose": "F7.tp_a2a_transpose_window",      # streamed all-to-all transposes
    "conf_blockreduce": "F7.tp_conf_blockreduce",           # exact-finish row-block confidence reducer
    "rank0_featurize_bcast": "F7.tp_rank0_featurize_bcast",
    "shard_park_host": "F7.tp_shard_park_host",              # parking a shard in pinned host memory
    "trunk_rows": "F7.tp_trunk_rows",                        # pair init / recycling as row statements on the shard (z born sharded)
    "msa_rows": "F7.tp_msa_rows",                            # MSA module on a sharded z: OPM output rows, PWA with local query-row logits
    "msa_host": "F7.tp_msa_host",                            # raw MSA / token bonds host-resident, selected rows per cycle (placement)
    "template_rows": "F7.tp_template_rows",                  # template embedder on rows (u slabs [T, R, N, c_t])
    "heads_rows": "F7.tp_heads_rows",                        # distogram / PAE / PDE logits and confidence pair input per row block
    "frame_rows": "F7.tp_frame_rows",                        # row-blocked token-frame atom search / clash counts (replicated statements)
    "diffusion_rows": "F7.tp_diffusion_rows",                # diffusion conditioning rows, local-query transformer, atom-attention band
}


def sub_strategy(key: str) -> str:
    """The ``strategy=`` value of sub-lever ``key``'s LEVER line: its strategy id (:data:`SUB_LEVERS`)."""
    return SUB_LEVERS[key]


def _declare_arch() -> None:
    """The torch stack's ONE arch declaration per sub-lever (:mod:`opt_core.arch`; ``F7.tensor_parallel`` itself is declared by
    :mod:`opt_core.mem.ngpu`, the framework-free producer): data movement over NCCL + the engine's own statements —
    no arch-specific kernel, so no floor among the registry's classes and no exclusion."""
    from ... import arch
    for lever in SUB_LEVERS.values():
        arch.declare(lever, certified=CERTIFIED_SM, note=TP_ARCH_NOTE)


_declare_arch()


class RowpairRefused(NGpuRefused):
    """A row-sharding precondition failed (``lever`` = ``rowpair`` or the sub-lever's name, ``reason`` = the refusal words). The n_gpu
    rules themselves raise the base :class:`opt_core.mem.ngpu.NGpuRefused`; the kit catches :class:`opt_core.mem.MemLeverRefused`."""

    def __init__(self, reason: str, lever: str = SHARDING):
        super().__init__(reason, lever)


def visible_gpus() -> int:
    """The number of CUDA devices visible to this process (``torch.cuda.device_count()``, which honours ``CUDA_VISIBLE_DEVICES`` and does
    not initialise a CUDA context); 0 when torch has no CUDA. torch is imported here, lazily."""
    from ._torch import torch
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 — a broken driver reads as zero visible devices; the refusal then names visible=0
        return 0


def refuse_unless_visible(n_gpu, visible: Optional[int] = None) -> int:
    """:func:`opt_core.mem.ngpu.refuse_unless_visible` with ``visible`` defaulting to :func:`visible_gpus` (the torch probe)."""
    p = check_n_gpu(n_gpu)
    if p == 1:
        return p
    return _refuse_unless_visible_count(p, visible_gpus() if visible is None else int(visible))
