"""tp — the ``--n_gpu P`` resource axis of ``--mode big``: the trunk's pair stacks row-sharded over ``P`` cards through the core's
``opt_core.mem.rowpair`` primitives (strategy ``F7.tensor_parallel``; lever ``rowpair_tp``).

Interface (``opt_core.mem.ngpu`` is the ONE producer of the token text and the refusal sentences; this module imports them):
  * ``--n_gpu`` explicit, default 1; ``--n_gpu 1`` = the single-card path under any mode (token ``n_gpu=1 sharding=none``).
  * ``--n_gpu P > 1`` only under ``--mode big`` (``off`` / ``exact`` / ``fast`` REFUSE BY NAME before anything runs, exit 3: sharded reductions
    reorder sums), only with ``P`` visible cards (``refused: n_gpu=P visible=K``, exit 3; never auto-sized), token ``n_gpu=P sharding=rowpair``.
  * The kit launches its own workers: the parent ``opendde-opt pred --mode big --n_gpu P`` runs no model; it starts ``P`` rank processes
    (``opt_core.mem.rowpair.launch.run_rank_processes``, one visible card each, ``ROWPAIR_*`` rendezvous) running the same command; rank 0
    writes the run's outputs and manifest into ``-o``, ranks 1.. write theirs under ``-o/.rowpair/rank<r>/``; every rank's transcript is
    relayed to the parent's stdout prefixed ``[r<k>]``; a rank that dies takes the run down BY NAME (``ROWPAIR event=...``, exit 1).

Line: mode ``big`` at ``P > 1`` resolves to ``BIG_TP`` (``modes.big_line``, one of ``modes.BIG_LINES``): the ``big`` levers with
every pair track — trunk, template, MSA-module, structural, diffusion conditioning, confidence — row-sharded across the ranks (neither the
offload unit nor the XL unit is on the path: host streaming and row sharding are alternatives, not layers) + ``rowpair_tp``
(at ``P == 1`` nothing is installed and ``rowpair_tp`` reads ``inert``).

Sharding (``P > 1``): the residue pair representation ``z [N, N, c_z]`` is this rank's ROWS ``z[r0:r1] [R, N, c_z]`` (``Layout.auto(N, P,
rank)``: the P-invariant B-block grid) from the statement that creates it to the heads — nothing ``N x N x c_z`` of the residue track is
whole on a rank's device:
  * trunk (``OpenDDE.get_pairformer_output``): ``z_init`` born as rows (the two single projections, the relative position encoding
    materialized for the rows, the token bonds of the rows); the ``[T, N, N, *]`` template pair features sliced to the rows once; per
    recycle the recycling embed row-local, the template embedder on rows (its pair stack on the shard), the MSA module on rows (pair-weighted
    averaging reads local rows — ``msa.pwa_rows``, its ``[S_chunk, N, H*c]`` transient bounded by ``ROWPAIR_PWA_SCHUNK``; the outer-product
    mean writes local rows — ``msa.opm_rows_budgeted``; its pair block on the shard), the 48-block pairformer stack on the shard; every
    ``PairformerStack`` / ``PairformerBlock`` call = the engine's own sub-module statements on this rank's rows with the core's contractions
    (``trimul_outgoing`` ring / ``trimul_incoming`` all-to-all, ``triatt_starting`` / ``triatt_ending`` with the all-gathered triangle bias,
    ``transition_rows``, ``apb_local_queries`` + all-gather of the ``s`` rows); the trunk returns a :class:`RowShard` (no gather).
  * structural-token stage (expansion, refiner): on this rank's rows of the STRUCTURAL pair space, fed the residue rows they read fetched in
    fixed row slabs around the ring (``ROWPAIR_HOSTGATHER_ROWS``; :mod:`opendde_opt.tp_struct`); the diffusion conditioning and the diffusion
    transformer's pair biases on this rank's structural rows (:mod:`opendde_opt.tp_diffusion`); the residue shard parked on the host meanwhile
    (``ROWPAIR_PARK_ZRES``).
  * distogram: logits of the rows (+ the transposed rows through the shard transpose), contact probabilities row-local — this rank's rows
    ``[R, N]`` kept for the confidence reducer; no ``[N, N]`` matrix on a device.
  * confidence head: pair init rows, per sample the distance one-hot of the rows, its 4-block stack on the shard, PAE rows, PDE rows of
    ``z + zᵀ`` consumed in row blocks by the core's ``confidence.RowBlockReducer``; rank 0 finishes the stock summaries and broadcasts them
    (``conf_logits=reducer:<finish>``).
Replicated by design (named in the LEVER census ``replicated_by_design=``): ``s``, ``s_inputs``, the MSA representation ``m`` only under
``ROWPAIR_MSA_M_LAYOUT=replicated`` (token-sharded by default), PWA's value projection per S-chunk, the triangle-attention bias ``[N, N, H]``
per call, atom tensors, the structural singles, the DiT token activations, noise / coordinates, the confidence summaries. A call shape the adapter does not shard is refused or named (``ROWPAIR event=unsharded_call``), never silently replicated. The
replicated INPUTS are compared across ranks once per (job, seed) batch before they move to the card: every rank digests its feature dict
minus :data:`PER_RANK_FEATS` and the core names the verdict (``[feats] rank <r> digest <d16> feats_ranks_equal=yes|no ranks=<P> digests=…``;
LEVER census ``feats_ranks_equal=`` / ``feats_digest=``); a differing rank refuses on every rank. They are equal by construction: under
``P > 1`` rank 0 ALONE featurises each query item and the core broadcasts the tree (``data_form=rank0_bcast``, :func:`_rank0_item`); ranks > 0
build only their own rows of the template pair features from the broadcast template coordinates (``tp_feats.rows_for_rank``). Every
block size the adapter chooses is fixed or env-given (``kit_schedule=`` in the census); the core's budget-derived schedules are agreed across
ranks by the core. Numerics: fast-class (the band of ``big``), never bitwise vs ``P = 1``.

STATS / evidence: ``kit_stats()`` = this rank's census (P, rank, layout facts of the last stack, stacks/blocks sharded, gathers) for the
lever's ONE ``LEVER name=rowpair_tp`` line (``report.lever_lines``: the core's ``rowpair.evidence`` fields lead); ``fallbacks()`` names a
stack that ran unsharded under ``P > 1``; ``notes()`` says when no pair stack was reached.
"""
from __future__ import annotations

import os
import sys
import threading

from opt_core.mem.rowpair import rankdata as _rankdata      # the core's rank-data words and calls (standard library at import; torch is reached lazily inside its functions)

from . import modes

try:                                                     # the core's per-stage memory census (opt_core.mem.rowpair.census; no-op unless OPT_CORE_TP_CENSUS=1).
    from opt_core.mem.rowpair import census as _census   # Instrumentation only: an older core without the module = no marks (never gates real work).
except ImportError:                                      # pragma: no cover
    _census = None


def _mark(stage: str) -> None:
    STATS["last_mark"] = stage
    if _census is not None:
        _census.mark(stage)
        _memstats(stage)


def host_peak_gib() -> float:
    """This process's peak resident host memory in GiB (``getrusage(RUSAGE_SELF).ru_maxrss``, KiB on Linux)."""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)


def _memstats(stage: str) -> None:
    """One ``MEMSTATS`` line per census mark on this rank (only with the census hook on and CUDA present): the caching allocator's peak
    ``allocated`` vs peak ``active`` (blocks freed by the program but still held for a stream event, e.g. a collective's operand) vs peak
    ``reserved`` and the inactive-split bytes — the numbers that say whether the DRIVER's view of a rank follows its tensors."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        st = torch.cuda.memory_stats()
        g = lambda k: round(st.get(k, 0) / 2 ** 30, 2)
        print(f"[{TAG}] MEMSTATS rank={STATS.get('rank')} stage={stage} allocated_peak_gib={g('allocated_bytes.all.peak')} active_peak_gib={g('active_bytes.all.peak')} "
              f"reserved_peak_gib={g('reserved_bytes.all.peak')} reserved_gib={g('reserved_bytes.all.current')} inactive_split_gib={g('inactive_split_bytes.all.current')} "
              f"inactive_split_peak_gib={g('inactive_split_bytes.all.peak')} alloc_retries={st.get('num_alloc_retries', 0)} ooms={st.get('num_ooms', 0)} "
              f"avoid_record_streams={os.environ.get('TORCH_NCCL_AVOID_RECORD_STREAMS', '')}"
              f"{' since=' + str(STATS.get('memstats_prev') or 'process_start') if os.environ.get('ODDE_TP_MEMSTATS_RESET', '') == '1' else ''}", file=sys.stderr, flush=True)
        if os.environ.get("ODDE_TP_MEMSTATS_RESET", "") == "1":                   # box-local diagnostic (tp_probe): each mark reads the peak SINCE the previous mark
            torch.cuda.reset_peak_memory_stats()
            STATS["memstats_prev"] = stage
    except Exception as e:   # instrumentation only: never takes the run down, says so
        print(f"[{TAG}] MEMSTATS rank={STATS.get('rank')} stage={stage} unavailable: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)


LEVER = modes.TP_LEVER                                  # "rowpair_tp"
LINE_NAME = modes.BIG_TP_LINE                         # "BIG_TP"
TAG = "opendde-opt"
TARGET = "opendde.model.modules.pairformer"
BLOCK_ATTR = "PairformerBlock.forward_source"          # the block statement the pin's stacks (and the MSA module's pair stack) call: pairformer.py:602-632; .forward = Fold-CP hook + delegate (:273-323)
RANK_SUBDIR = ".rowpair"                               # under -o: rank<r>/ output dirs of ranks 1.. and the launcher's per-rank logs
STATS = {"installed": False, "armed": False, "n_gpu": 1, "rank": None, "calls": 0, "stacks": 0, "blocks": 0, "direct_blocks": 0,
         "gathers": 0, "unsharded": 0, "unsharded_reasons": {}, "layout": None, "group": False,
         "stack_s": {},                                  # n_blocks -> [calls, seconds] of sharded stack calls (device-synchronised wall)
         "trunk_rows": 0, "presharded_calls": 0, "template_rows": 0, "template_feats_rows": 0, "msa_blocks_rows": 0,
         "trunk_exit": None, "conf_rows": 0, "conf_samples_rows": 0, "conf_logits": None, "distogram_rows": 0,
         "zcond_rows": 0, "dit_local_queries": 0, "struct_rows": 0, "zres_ring_rows": 0, "struct_presharded_calls": 0, "struct_stage": None, "struct_stage_s": None, "struct_bf16_calls": 0, "struct_pair_dtype": None, "zinit_where": None, "zinit_park_gb": 0.0, "zinit_park_s": 0.0, "recycle_rows_calls": 0, "zres_park": 0, "zres_unpark": 0, "zres_where": None, "zres_park_gb": 0.0, "zres_unpark_at": None, "last_mark": None, "template_rows_released_gb": 0.0, "schedule": {},
         "feat_host_keys": 0, "feat_host_gb": 0.0, "template_full_gb": 0.0, "feat_resident_gb": None, "template_rows_h2d_gb": 0.0, "zinit_blocks": 0,
         "template_rows_born": 0, "inputs_born_gb": 0.0, "inputs_full_gb": 0.0,
         "feats_batches": 0, "feats_unequal": 0, "feats_digest": None, "feats_ranks_equal": None,   # the cross-rank feature census (_feats_census): batches digested / differing, the last digest, yes|no
         "feats_bcast_calls": 0, "feats_error_items": 0, "feats_bcast_gib": 0.0, "feats_wait_s": 0.0, "feats_bcast_s": 0.0, "feats_rng": None}   # the rank0_bcast data form (_rank0_item): items handed from rank 0, error items, GiB / seconds moved, rendezvous wait, rank 0's RNG state carried|off|none
REPLICATED_BY_DESIGN = ("s", "s_inputs", "input_feats(msa/token feats on device; template pair feats = this rank's host rows)", "m[S,N,c_m](only under ROWPAIR_MSA_M_LAYOUT=replicated)", "pwa_v[S_chunk,N,H*c]", "triatt_bias[N,N,H]", "atoms",
                        "struct_singles(s_inputs_struct[N_st,c_s_inputs], s_struct[N_st,c_s], structural token features[N_st])", "refiner_triatt_bias[N_st,N_st,H]", "dit_tokens_a[N_st,768]", "noise/coordinates", "conf_summaries(broadcast from rank 0)")
ENV_PWA_SCHUNK = "ROWPAIR_PWA_SCHUNK"                   # MSA rows per pair-weighted-averaging call (bounds PWA's v/g/o [S_chunk, N, H*c] transient; exact)
PWA_SCHUNK_DEFAULT = 512
ENV_ATTN_ROWS = "ROWPAIR_ATTN_ROWS"                      # rows per triangle-attention / transition batch when the stock passes no chunk size
ATTN_ROWS_DEFAULT = 32
ENV_ALLOW_UNSHARDED = "ROWPAIR_ALLOW_UNSHARDED"          # opt-in: a pair-stack call shape the adapter does not shard RUNS unsharded (named event) instead of refusing
TRIMUL_INPLACE_CHUNK_DEFAULT = 256                       # opendde's TriangleMultiplicativeUpdate._inference_forward column chunk (triangular.py)
ENV_MSA_M_LAYOUT = "ROWPAIR_MSA_M_LAYOUT"                # replicated (default) | token_sharded: the MSA representation m's layout in the MSA module's row statements
KIT_SCHEDULE_KEYS = ('layout_B', 'layout_B_source', 'pwa_schunk', 'pwa_schunk_source', 'hostgather_rows', 'hostgather_rows_source', 'msa_transition', 'diff_noise', 'msa_m', 'conf_logits', 'conf_rows', 'conf_rows_source', 'contact_probs', 'struct_stage', 'dit_queries', 'atom_band', 'trunk_init', 'input_feats', 'inputs', 'diffz', 'struct_pair_dtype', 'zinit', 'recycle', 'template', 'zres', 'template_rows_after_trunk', 'struct_trimul_rb', 'nccl_timeout_s', 'transpose_inplace')   # STATS['schedule'] carries exactly these words (tests/test_tp_gate_names.py)
ENV_NOISE_SYNC = "ROWPAIR_DIFF_NOISE_SYNC"               # bcast (line default) | guard: replicated random draws (MSA sample, diffusion noise) across ranks
ENV_STRUCT_PAIR_DTYPE = "ODDE_TP_STRUCT_PAIR_DTYPE"    # bf16 | fp32: the structural pair shard + refiner pair stack dtype under n_gpu>1 (lever struct_pair_bf16; BIG_TP exports bf16)
ENV_TEMPL_ROWS = "ROWPAIR_TEMPL_ROWS"                    # rows per template-embedder block (v_t rows / the close); unset = the core ROWPAIR_ROWBLK_MB chooser; P-invariant
ENV_STRUCT_TRIMUL_RB = "ODDE_TP_STRUCT_TRIMUL_RB"          # rows per streamed tri-mult b slab in the STRUCTURAL refiner (P-invariant; BIG_TP exports 128; 0/unset = the core's budgeted RB)
ENV_PARK_ZRES = "ROWPAIR_PARK_ZRES"                      # 1 (BIG_TP export): the residue z shard parked on the host during the structural stage + roll-out (RowShard.park)
ENV_PARK_ZINIT = "ROWPAIR_PARK_ZINIT"                    # 1 (BIG_TP export): z_init parked in pinned host memory during the trunk; 0: resident on the card (the core's ShardPark flag)
ENV_INIT_ROWS = "ROWPAIR_INIT_ROWS"                      # rows per z_init block (the core's trunk_init_rows chooser: this count > ROWPAIR_ROWBLK_MB > 512 MiB; P-invariant)
RUNNER_TARGET = "runner.inference"                       # the stock runner module: its to_device moves the feature dict to the card (the [T, N, N, *] template pair features stay on host under P>1)
GETITEM_TARGET = "opendde.data.inference.infer_dataloader"   # upstream's inference dataset: InferenceDataset.__getitem__ featurises one query item (process_one) — rank 0 alone under P>1 (_rank0_item)
HOST_FEATS_KEY = "_rowpair_host_feats"                    # marker in the feature dict: ("rows", r0, R) = the template pair features are this rank's HOST row slabs (they move at trunk entry)
INPUT_ROWS_KEY = "_rowpair_input_rows"                    # the featurizer's marker (== tp_feats.ROWS_KEY, which imports numpy; the tests pin the equality): int64 [r0, R, N]; never moves to the card
ENV_LAYOUT_B = "ROWPAIR_LAYOUT_B"                        # explicit layout block size (default: the core's largest valid candidate of 128/64/32/16)
ENV_MSA_TRANS_SHARD = "ROWPAIR_MSA_TRANS_SHARD"          # 1 (default): the MSA transition token-sharded + all-gather; 0: whole-m statement on every rank (named)
ENV_RING_ROWS = "ROWPAIR_HOSTGATHER_ROWS"               # row-slab height of the residue-row ring the structural stage reads (tp_struct; fixed -> identical collective counts on every rank)
_LOCK = threading.Lock()
_LAYOUTS: dict = {}


class TpRefused(RuntimeError):
    """``--n_gpu`` refused by name (the core's sentence); the CLI exits 3 with it."""


# ----------------------------------------------------------------------------------------------------------------- interface
def check(n_gpu, mode: str | None, visible: int | None = None) -> int:
    """``P`` accepted for ``mode`` on this box, or :class:`TpRefused` with the core's sentence: ``P > 1`` needs ``--mode big`` (``off``,
    ``exact``, ``fast`` refuse it by name before anything runs) and ``P`` visible cards (``visible`` None = the torch probe of the core).
    ``P == 1`` passes under every mode."""
    from opt_core.mem import ngpu as _ngpu                     # imported here: the entry routes refuse an older core by name before this runs (_producers)
    try:
        p = _ngpu.refuse_unless_big(n_gpu, mode or "")
        if p > 1 and not in_rank_process():                    # the parent probes P cards; a rank process sees its one card by construction
            from opt_core.mem import rowpair
            p = rowpair.refuse_unless_visible(p, visible)
    except _ngpu.NGpuRefused as e:                              # ⊃ RowpairRefused
        raise TpRefused(getattr(e, "reason", None) or str(e)) from None
    except ValueError as e:
        raise TpRefused(str(e)) from None
    return p


def fields(n_gpu) -> str:
    """``n_gpu=P sharding=rowpair`` / ``n_gpu=1 sharding=none`` for the ACTIVE and EXIT lines (the core's text)."""
    from opt_core.mem import ngpu as _ngpu
    return _ngpu.active_fields(int(n_gpu or 1), "rowpair")


def rank_world() -> int:
    """The launcher's world size in this process's environment (``ROWPAIR_WORLD``; 1 outside a rank process)."""
    try:
        return int(os.environ.get("ROWPAIR_WORLD") or 1)
    except ValueError:
        return 1


def mismatch(requested: int, active: int, where: str) -> str | None:
    """``n_gpu_mismatch requested=P active=Q (<where>)`` when the axis a process reports differs from the ``--n_gpu`` it was asked for
    (fail-closed: a run that folded on fewer cards is never a pass), else None."""
    if int(requested) != int(active):
        return f"n_gpu_mismatch requested={int(requested)} active={int(active)} ({where})"
    return None


def in_rank_process() -> bool:
    """True inside a rank process the launcher started (``ROWPAIR_WORLD > 1``)."""
    from opt_core.mem.rowpair import launch
    return launch.world_size() > 1


def rank() -> int:
    from opt_core.mem.rowpair import launch
    return launch.rank()


def register_line() -> modes.Line:
    """``BIG_TP``: the mode table's own row (modes.big_line at BIG_OFFLOAD_STAGES / BIG_MEM_LEVERS; composed at the package's
    size-gate switch like every big line)."""
    return modes.LINES[LINE_NAME]


def enter_rank_process(n_gpu: int) -> None:
    """Inside a rank process: ``big`` resolves to ``BIG_TP`` for this process (the mode word stays ``big`` on every line)."""
    modes.MODE_LINES["big"] = LINE_NAME
    STATS["n_gpu"], STATS["rank"] = int(n_gpu), rank()
    _apply_rank_threads()


def _apply_rank_threads() -> None:
    """The per-rank CPU-thread cap: ``ROWPAIR_RANK_THREADS`` — the line exports ``auto`` (cores / ranks) — applied
    ONCE in this rank process (``dist.apply_rank_threads``: torch threads + the OpenMP / MKL / OpenBLAS variables + the core's census words
    ``rank_threads=<n> rank_threads_source=…``). Called at rank entry and again at the first trunk entry: the line's exports reach this
    process's environment during activation, after ``enter_rank_process``; an unset word touches nothing (``STATS["rank_threads"]`` None)."""
    if STATS.get("rank_threads") is not None:
        return
    from opt_core.mem.rowpair import dist as _dist
    if hasattr(_dist, "apply_rank_threads"):
        STATS["rank_threads"] = _dist.apply_rank_threads(os.environ.get("ROWPAIR_RANK_THREADS"), local_world=int(STATS.get("n_gpu") or 1))[0]


def rank_out_dir(out_dir: str, r: int) -> str:
    return out_dir if r == 0 else os.path.join(out_dir, RANK_SUBDIR, f"rank{r}")


def run_ranks(n_gpu: int, argv_of, out_dir: str, on_line=None, run_timeout_s: float | None = None):
    """The parent's whole job at ``P > 1``: start the rank processes (one visible card each), relay their lines, return the core's
    per-rank records. Raises the core's ``RankFailed`` when a rank dies or times out."""
    from opt_core.mem.rowpair import launch
    log_dir = os.path.join(out_dir, RANK_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)
    kw = dict(argv_of=argv_of, mode="big", log_dir=log_dir, isolate_devices=True, on_line=on_line, what="opendde-opt pred",
              env=dict(os.environ, **{launch.ENV_TAG: TAG}))                  # the ranks' base environment = the parent's, tagged: the core's launch lines read `[opendde-opt] RANKENV …`
    if run_timeout_s:
        kw["run_timeout_s"] = float(run_timeout_s)
    return launch.run_rank_processes(int(n_gpu), None, **kw)


# ----------------------------------------------------------------------------------------------------------------- install (rank side)
def install() -> None:
    """Patch ``PairformerStack.forward`` and ``PairformerBlock.forward_source`` at the module's import (the core's per-site patch; on the pin
    the stacks reach their blocks through ``forward_source`` — pairformer.py:602-632 — and ``PairformerBlock.forward`` is the Fold-CP hook + a
    delegate to it, :273-323). Idempotent. At ``n_gpu == 1`` (no rank environment) nothing is patched: the engine's statements run unchanged."""
    with _LOCK:
        if STATS["installed"] or STATS["armed"]:
            return
        if not in_rank_process():
            STATS["n_gpu"] = 1
            return
        from opt_core import autoload
        from . import tp_feats as _tf
        _noise_sync()                                                               # malformed ROWPAIR_DIFF_NOISE_SYNC values refused by name at install (the policy itself is decided per call by det level)
        STATS["patches"] = [autoload.patch_attr_at_import(TARGET, "PairformerStack.forward", _make_stack_forward, tag=TAG, name=LEVER),
                            autoload.patch_attr_at_import(TARGET, BLOCK_ATTR, _make_block_forward, tag=TAG, name=LEVER + "_block"),
                            autoload.patch_attr_at_import(MODEL_TARGET, "OpenDDE.get_pairformer_output", _make_trunk_forward, tag=TAG, name=LEVER + "_trunk"),
                            autoload.patch_attr_at_import(MODEL_TARGET, "OpenDDE.compute_distogram_contact_probs", _make_distogram_rows, tag=TAG, name=LEVER + "_distogram"),
                            autoload.patch_attr_at_import(CONF_TARGET, "ConfidenceHead.forward", _make_conf_forward, tag=TAG, name=LEVER + "_conf"),
                            autoload.patch_attr_at_import(EMB_TARGET, "RelativePositionEncoding.forward", _make_relp_forward_rows_only, tag=TAG, name=LEVER + "_relp_whole")]
        STATS["input_patches"] = [autoload.patch_attr_at_import(RUNNER_TARGET, "to_device", _make_to_device_host_feats, tag=TAG, name=LEVER + "_host_feats"),
                                  autoload.patch_attr_at_import(GETITEM_TARGET, "InferenceDataset.__getitem__", _make_getitem_rank0, tag=TAG, name=LEVER + "_rank0_feats")] + \
                                 _tf.install_patches(autoload, TAG, LEVER)         # the featurizer bears this rank's ROWS of the template pair features (tp_feats)
                                                                                    # the runner's feature move (installed when the stock runner drives the model; a caller that feeds the
                                                                                    # model itself hands whole HOST tensors and the trunk entry cuts them — whole tensors ON THE CARD refuse)
                                                                                    # no generate_relp patch: upstream's inference relp is lazy on the pin (opendde.py:1906 lazy_relp=True)
        _sync()


def _sync() -> None:
    ps = STATS.get("patches") or []
    if ps:
        STATS["installed"] = all(p.state == "installed" for p in ps)
        STATS["armed"] = (not STATS["installed"]) and any(p.state in ("armed", "installed") for p in ps)


def _group():
    """``(P, rank)`` (``dist.world()``) with the process group initialised from the rank environment (once)."""
    from opt_core.mem.rowpair import dist as _dist
    if not STATS["group"]:
        import atexit
        P, r, dev = _dist.init_from_env()
        STATS["group"], STATS["n_gpu"], STATS["rank"] = True, int(P), int(r)
        atexit.register(_dist.destroy)                       # the group is torn down at exit (after the EXIT line's peak gather)
    return _dist.world()


def layout_of(N: int, P: int, r: int):
    """THE row layout of ``N`` tokens over ``P`` ranks as seen by rank ``r``: the core's ``Layout.auto`` (refusing forms only; ``ROWPAIR_LAYOUT_B``
    = one explicit candidate). Pure arithmetic — the featurizer (``tp_feats``, before any process group exists) and the trunk call this one
    function, so the rows born in the data pipeline ARE the trunk's rows."""
    from opt_core.mem.rowpair import dist as _dist
    b = _layout_b()
    return _dist.Layout.auto(int(N), int(P), int(r), lever=LEVER) if b is None else _dist.Layout.auto(int(N), int(P), int(r), candidates=(b,), lever=LEVER)


def _layout(N: int):
    P, r = _group()
    key = (int(N), P, r)
    lay = _LAYOUTS.get(key)
    if lay is None:
        lay = layout_of(N, P, r)
        _LAYOUTS[key] = lay
    STATS["layout"] = lay.facts()
    return lay


# ----------------------------------------------------------------------------------------------------------------- engine statements
def _lead(z, keep: int = 3):
    """``(x, n_lead)``: ``x`` with size-1 leading dims stripped down to ``keep`` dims, plus how many were stripped."""
    n = 0
    while z.dim() > keep and int(z.shape[0]) == 1:
        z = z[0]
        n += 1
    return z, n


def _relead(x, n):
    for _ in range(n):
        x = x.unsqueeze(0)
    return x


def _trimul_fns(m):
    """The engine's TriangleMultiplication{Outgoing,Incoming} (triangular.py:478-560) as the core's ``TriMulFns``: gated a / b projections of
    LayerNorm'd ORIGINAL pair values under the mask, LayerNorm-out + ``linear_z`` of the contraction, the output gate from the original values.
    A fused trimul kernel installed on the module's ``forward`` (cuEquivariance / the fast line's ARM binds) is BYPASSED here by construction: the
    statements below are the module's torch sub-layers — and the torch FALLBACK of the core's fused row-block provider the result
    is wrapped in (``tp_kernels.trimul_provider``: fpf_trimul_rows K1 / K3 at this trunk's width 384 and the template stack's 64)."""
    from opt_core.mem.rowpair import trimul as _tm

    def proj(zb, mb, is_a):                                                      # every callable returns z's dtype (the ring schedule sizes its sub-block messages by the shard's dtype; an
        zl = m.layer_norm_in(zb)                                                    # autocast-typed operand would desync byte counts across ranks)
        g, p = (m.linear_a_g, m.linear_a_p) if is_a else (m.linear_b_g, m.linear_b_p)
        return (mb.to(dtype=zl.dtype) * m.sigmoid(g(zl)) * p(zl)).to(dtype=zb.dtype)

    def out(x):
        return m.linear_z(m.layer_norm_out(x)).to(dtype=x.dtype)

    def gate(zb):
        return m.sigmoid(m.linear_g(m.layer_norm_in(zb))).to(dtype=zb.dtype)
    stock = _tm.TriMulFns(proj=proj, out=out, gate=gate, C_h=int(m.linear_a_p.out_features))
    from . import tp_kernels as _tpk
    return _tpk.trimul_provider(m, stock) or stock                                  # the core's fused row-block provider around these statements (declined units run them, counted)


def _trimul_inplace_chunk(m) -> int:
    """The engine's in-place triangle-multiplication column chunk (``TriangleMultiplicativeUpdate._inference_forward(_inplace_chunk_size=256)``,
    triangular.py) — the core's outgoing ring grid follows it (``grid=stock``)."""
    import inspect
    try:
        v = inspect.signature(m._inference_forward).parameters["_inplace_chunk_size"].default
    except (AttributeError, KeyError, ValueError, TypeError):
        v = None
    return int(v) if isinstance(v, int) and v > 0 else TRIMUL_INPLACE_CHUNK_DEFAULT


def _triatt_fns(t, triangle_attention: str, chunk_size=None, kernel: bool = True):
    """The engine's TriangleAttention (triangular.py:600-700; starting form — the core driver hands the ending node the rows of zᵀ and maskᵀ)
    as the core's ``TriAttFns``: its LayerNorm, its triangle-bias projection, and the attention of a row batch (the mask bias of those rows,
    the whole triangle bias as ``[1, H, N, N]``). ``kernel`` (the trunk, template and MSA-module pair stacks — lever ``tp_triatt``): the row
    batch runs on the core's ONE row-block kernel dispatch (``tp_kernels.attention_dispatch``: the core's triangle-attention kernel for the line's tier word; the
    module's own torch statement its named per-call fallback over ``chunk_size`` query rows). ``kernel=False`` (the structural refiner's pair
    stack, ``tp_struct.refine_rows``): the module's own ``mha`` — through its ``_chunk`` over ``chunk_size`` query rows when the stock passes
    one — on the runner's ``triangle_attention`` word (cuequivariance | torch), as on one card."""
    from opendde.model.utils import permute_final_dims
    from opt_core.mem.rowpair import triatt as _ta
    if not kernel:
        def attend(x_rows, mask_rows, tb_full, _span):
            mask_bias = (t.inf * (mask_rows.to(dtype=x_rows.dtype) - 1))[..., :, None, None, :]
            tb = permute_final_dims(tb_full, (2, 0, 1)).unsqueeze(-4)
            if chunk_size is not None and int(x_rows.shape[0]) > int(chunk_size):
                return t._chunk(x_rows, [mask_bias, tb], int(chunk_size), triangle_attention=triangle_attention, inplace_safe=False)
            return t.mha(q_x=x_rows, kv_x=x_rows, biases=[mask_bias, tb], triangle_attention=triangle_attention)
        return _ta.TriAttFns(ln=t.layer_norm, bias=t.linear, attend=attend)
    from . import tp_kernels as _tpk
    run = _tpk.attention_dispatch(t, chunk_size)

    def attend(x_rows, mask_rows, tb_full, _span):                               # noqa: F811 — the kernel route's attend
        mask_bias = (t.inf * (mask_rows.to(dtype=x_rows.dtype) - 1))[..., :, None, None, :]
        tb = permute_final_dims(tb_full, (2, 0, 1)).unsqueeze(-4)
        return run(x_rows, mask_bias, tb)
    return _ta.TriAttFns(ln=t.layer_norm, bias=t.linear, attend=attend)


def _dist_env_flag(name: str) -> bool:
    """A 0/1 core flag as the core parses it (``dist.env_flag``: one parser for every ROWPAIR_* flag)."""
    from opt_core.mem.rowpair import dist as _dist
    return bool(_dist.env_flag(name)) if hasattr(_dist, "env_flag") else (os.environ.get(name, "0").strip() == "1")


def _log(msg: str) -> None:
    """``[opendde-opt] msg`` on this rank's stderr, flushed (the adapter's event lines; the core's census text passes through it)."""
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _struct_pair_dtype() -> str:
    """``bf16`` | ``fp32``: the structural pair shard's dtype (lever ``struct_pair_bf16``; :data:`ENV_STRUCT_PAIR_DTYPE`, unset = fp32; any other
    value refuses by name)."""
    raw = (os.environ.get(ENV_STRUCT_PAIR_DTYPE) or "fp32").strip().lower()
    if raw not in ("bf16", "fp32"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_STRUCT_PAIR_DTYPE}={raw!r} (bf16 or fp32)")
    return raw


def _attn_rows(chunk_size, lay) -> int:
    """Rows per triangle-attention / transition row batch on this rank: the stock's chunk size when it passes one (its dynamic chunk keyed on
    N: a replicated value, identical on every rank), else ``ROWPAIR_ATTN_ROWS`` (default 32) — fixed, never derived from this rank's bytes."""
    if chunk_size is not None:
        return max(1, int(chunk_size))
    v = int(os.environ.get(ENV_ATTN_ROWS) or ATTN_ROWS_DEFAULT)
    if v < 1:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_ATTN_ROWS}={v} (must be >= 1)")
    return min(v, int(lay.Rmax))


def _pair_fns(blk, lay, *, triangle_attention: str, chunk_size=None, extra_attn_bias=None, trimul_rb=None, triatt_kernel: bool = True):
    """One ``PairformerBlock`` (pairformer.py:58-270) as the core's ``PairBlockFns`` (``pairstack.bind``): its two triangle multiplications,
    its two triangle attentions, its pair transition (no mask), and — when the block carries the single track (``c_s > 0``) — its
    attention-pair-bias on LOCAL query rows (all-gathered) and single transition. Dropout is the equality (eval)."""
    from opt_core.mem.rowpair import pairstack as _ps, transition as _tr
    if blk.training:
        raise RuntimeError(f"{LEVER}: PairformerBlock in training mode; the sharded statement is inference-only")
    apb = single = None
    if getattr(blk, "c_s", 0) > 0:
        def apb(s, z_sh, layout):
            return s + _tr.apb_local_queries(_apb_rows_of(blk.attention_pair_bias, extra_attn_bias, layout), s, z_sh, layout, gather=True)

        def single(s):
            return s + blk.single_transition(s)
    return _ps.bind(trimul_out=_trimul_fns(blk.tri_mul_out), trimul_in=_trimul_fns(blk.tri_mul_in),
                    triatt_start=_triatt_fns(blk.tri_att_start, triangle_attention, chunk_size, kernel=triatt_kernel), triatt_end=_triatt_fns(blk.tri_att_end, triangle_attention, chunk_size, kernel=triatt_kernel),
                    transition=lambda x, _mask_u=None: blk.pair_transition(x), chunk=_attn_rows(chunk_size, lay), apb=apb, single_transition=single,
                    trimul_kw=dict(inplace_chunk=_trimul_inplace_chunk(blk.tri_mul_out), **({"RB": int(trimul_rb)} if trimul_rb else {})),   # RB: an explicit, P-invariant streamed-slab row count (the structural refiner's; None = the core's budgeted RB)
                    stats=STATS.setdefault("contract_stats", {}))


def _apb_rows_of(p, extra_attn_bias, lay):
    """``attn_fn(q_rows, s_full, z_shard) -> [R, c_s]``: AttentionPairBias (transformer.py:550-615, ``has_s=False``) for this rank's
    query rows: the pair bias from the local rows of ``z``, keys/values from all of ``s``; an ``extra_attn_bias`` handed whole ``[.., N, N]``
    or as this rank's rows ``[.., R, N]`` (``tp_diffusion._extra_rows`` decides) adds its local query rows."""
    from opendde.model.utils import permute_final_dims

    def attn_fn(_q_rows, s_full, z_sh):
        a = p.layernorm_a(s_full)
        q = a[..., lay.r0:lay.r1, :]
        bias = permute_final_dims(p.linear_nobias_z(p.layernorm_z(z_sh)), [2, 0, 1])          # [H, R, N]
        if extra_attn_bias is not None:
            from . import tp_diffusion as _td
            eb, off = _td._extra_rows(extra_attn_bias, lay)                             # whole [.., N, N] (offset 0) or this rank's rows [.., R, N] (offset r0): one decision, tp_diffusion's
            while len(eb.shape) < len(bias.shape) - 1:
                eb = eb.unsqueeze(dim=0)
            if len(eb.shape) == len(bias.shape) - 1:
                eb = eb.unsqueeze(dim=-3)
            eb = eb[..., lay.r0 - off:lay.r1 - off, :]
            bias = bias + eb.to(dtype=bias.dtype, device=bias.device)
        bias = p._align_bias_to_query(bias, q, n_pair_dims=2)
        return p.attention(q_x=q, kv_x=a, attn_bias=bias)
    return attn_fn


def _presharded(z):
    """The trunk layout when ``z`` is this rank's row shard of the trunk's pair track (``[R, N, C]`` under an active :data:`_TRUNK` layout),
    else None (a full ``[N, N, C]`` tensor: shard at entry, gather at exit)."""
    lay = _TRUNK.get("lay")
    if lay is None or z is None or z.dim() < 3:
        return None
    return lay if (int(z.shape[-3]) == lay.R and int(z.shape[-2]) == lay.N and lay.R != lay.N) else None


def _run_sharded(blocks, s, z, pair_mask, *, triangle_attention: str, extra_attn_bias=None, chunk_size=None):
    """``(s, z)`` after ``blocks`` through the core's ONE pair-block driver (``pairstack.pair_stack_``) on this rank's rows. A ``z`` that
    arrives as this rank's shard (``[R, N, C]`` under the active layout: trunk, template, MSA, confidence) stays sharded — no gather; a full
    ``[N, N, C]`` ``z`` (a stack outside the residue track, e.g. the structural refiner when the structural stage runs on the device) is
    sharded at entry and all-gathered at exit (``gathers`` in the census). Leading size-1 dims kept."""
    from opt_core.mem.rowpair import pairstack as _ps, shard as _shard
    z3, nz = _lead(z)
    if z3.dim() != 3:
        raise _Unsharded(f"pair tensor {tuple(z.shape)}: a leading batch > 1")
    pre = _presharded(z3)
    N = pre.N if pre is not None else int(z3.shape[0])
    lay = pre if pre is not None else _layout(N)
    m2 = pair_mask
    while m2 is not None and m2.dim() > 2 and int(m2.shape[0]) == 1:
        m2 = m2[0]
    if m2 is not None and (m2.dim() != 2 or tuple(m2.shape) != (N, N)):
        raise _Unsharded(f"pair mask {tuple(pair_mask.shape)} vs N={N}")
    s2, ns = (None, 0) if s is None else _lead(s, keep=2)
    if s2 is not None and (s2.dim() != 2 or int(s2.shape[0]) != N):
        raise _Unsharded(f"single tensor {tuple(s.shape)} vs N={N}")
    import time
    import torch
    if z3.is_cuda:
        torch.cuda.synchronize(z3.device)
    t0 = time.perf_counter()
    if torch.are_deterministic_algorithms_enabled():                          # det-scoped: under the deterministic recipe the recomputed z / s agree bitwise across
        if pre is None:                                                       # ranks (checked, refused by name on a mismatch); outside it the tier-2 statement covers it
            _replicated(z3, f"z@stack_entry[{len(blocks)}x{N}]")
        if s2 is not None:
            _replicated(s2, f"s@stack_entry[{len(blocks)}x{N}]")
    z_sh = (z3 if z3.is_contiguous() else z3.contiguous()) if pre is not None else _shard.shard_rows(z3, lay, dim=0).contiguous()
    mask_sh = None if m2 is None else m2[lay.r0:lay.r0 + lay.R].to(dtype=z_sh.dtype).contiguous()     # None = all ones (the driver's default; no [N, N] allocation)
    if s2 is not None and not s2.is_contiguous():
        s2 = s2.contiguous()
    STATS["calls"] += 1
    fns = [_pair_fns(blk, lay, triangle_attention=triangle_attention, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias) for blk in blocks]
    prof = None
    if os.environ.get("ODDE_TP_PROF_STACK"):                                    # box-local diagnostic (tp_probe): one named stack call under torch.profiler
        from . import tp_probe as _probe
        prof = _probe.stack_profile(len(fns), N, lay.R, STATS.get("rank"))
    if prof is None:
        z_sh, s2 = _ps.pair_stack_(fns, z_sh, mask_sh, lay, s=s2, transition_mask=False)
    else:
        with prof:
            z_sh, s2 = _ps.pair_stack_(fns, z_sh, mask_sh, lay, s=s2, transition_mask=False)
    STATS["blocks"] += len(fns)
    if pre is not None:                                                       # the residue track stays sharded
        z_out = z_sh
        STATS["presharded_calls"] += 1
    else:
        z_out = _shard.unshard_rows(z_sh, lay, dim=0)
        STATS["gathers"] += 1
        del z_sh
    if z_out.is_cuda:
        torch.cuda.synchronize(z_out.device)
    rec = STATS["stack_s"].setdefault(f"{len(blocks)}x{N}", [0, 0.0])
    rec[0] += 1
    rec[1] = round(rec[1] + time.perf_counter() - t0, 3)
    return (None if s is None else _relead(s2, ns)), _relead(z_out, nz)


# ------------------------------------------------------------------------ the trunk's pair track kept sharded across the recycles (persistent seam)
_TRUNK = {"lay": None}                                    # the active trunk layout while OpenDDE.get_pairformer_output runs on row shards (rank processes)
MODEL_TARGET = "opendde.model.opendde"


def _template_rows(te, feats, z_sh, lay, *, triangle_attention: str, triangle_multiplicative: str, inplace_safe: bool, chunk_size):
    """== TemplateEmbedder.forward (pairformer.py) on this rank's rows through the core's ``template.template_embed_rows``: per template slot
    ``v_t`` rows = ``linear_z(LN_z(z rows)) + linear_a(features rows under the multichain mask)`` produced per ROW BLOCK (never a shard-sized
    ``LN_z(z)`` or feature slab), the slot's pair stack on the ``[R, N, c]`` row shard through the module call (the kit's hooks stay in force;
    the shard is recognised: no gather) with ``LN_v``, and the close ``z rows += linear_u(relu(sum_t / (1e-7 + T)))`` ADDED IN PLACE per row
    block (the stock's accumulation order over slots). Returns ``z_sh`` (the same tensor). Block rows: ``ROWPAIR_TEMPL_ROWS`` (a count,
    P-invariant) else the core's ``ROWPAIR_ROWBLK_MB`` chooser; recorded as ``templ_rows`` in the core schedule census."""
    import torch
    import torch.nn.functional as F
    from opt_core.mem.rowpair import template as _tmpl
    if "template_aatype" not in feats or te.n_blocks < 1:
        return z_sh
    r0, N = lay.r0, lay.N
    n_cls = len(sys.modules[TARGET].STD_RESIDUES_WITH_GAP)
    asym = feats["asym_id"]
    sliced = bool(feats.get(TEMPLATE_ROWS_KEY))                                 # pair features already cut to this rank's rows [T, R, N, *] (trunk entry), or full [T, N, N, *]
    T = int(feats["template_aatype"].shape[0])

    def frows(key, t, g0, g1):                                                  # GLOBAL rows g0:g1 of slot t's [N, N(, *)] feature
        x = feats[key][t]
        return x[g0 - r0:g1 - r0] if sliced else x[g0:g1]

    def unit_rows(z_rows, t, span):                                             # v_t rows [1, rows, N, c]
        g0, g1 = span
        n = g1 - g0
        mc = (asym[g0:g1, None] == asym[None, :]).to(z_rows.dtype)             # multichain mask rows
        aat = F.one_hot(feats["template_aatype"][t], num_classes=n_cls).to(z_rows.dtype)
        at = torch.cat([frows("template_distogram", t, g0, g1) * mc[..., None], (frows("template_pseudo_beta_mask", t, g0, g1) * mc).unsqueeze(-1),
                        aat[None, :, :].expand(n, N, n_cls), aat[g0:g1, None, :].expand(n, N, n_cls),      # expand_at_dim(aatype, -3) / (-2)
                        frows("template_unit_vector", t, g0, g1) * mc[..., None], (frows("template_backbone_frame_mask", t, g0, g1) * mc).unsqueeze(-1)], dim=-1)
        return (te.linear_no_bias_z(te.layernorm_z(z_rows)) + te.linear_no_bias_a(at)).unsqueeze(0)

    def pair_stack(u, _mask):                                                   # u [1, R, N, c]: the slot's pair stack on the row shard + LN_v
        _, v = te.pairformer_stack(None, u[0], None, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                   inplace_safe=inplace_safe, chunk_size=chunk_size)          # the shard is recognised (_presharded): no gather
        return te.layernorm_v(v).unsqueeze(0)

    def finish(tt):                                                             # tt [T, rows, N, c] in slot order -> the close on these rows
        u = 0
        for ti in range(int(tt.shape[-4])):                                     # the stock's accumulation order: u = u + LN_v(v_t)
            u = u + tt[..., ti, :, :, :]
        return te.linear_no_bias_u(te.relu(u / (1e-7 + T)))

    STATS["template_rows"] += T
    rows = int(os.environ.get(ENV_TEMPL_ROWS) or 0) or None
    return _tmpl.template_embed_rows(z_sh, lay, n_templ=T, c_t=int(te.linear_no_bias_z.out_features), unit_rows_fn=unit_rows, pair_stack_fn=pair_stack,
                                     finish_fn=finish, rows=rows, add=True, log=_log)


def _pwa_fns(pwa):
    """MSAPairWeightedAveraging (pairformer.py:555-599) as the core's ``pwa_bias_rows`` / ``pwa_rows`` callables: the pair logits of local rows
    (``linear_no_bias_z(layernorm_z(z_rows))`` heads-first; no mask term in the stock statement), the value / gate projections of an S-chunk of
    ``m`` (``[S_chunk, N, H*c]`` — the transient ``ROWPAIR_PWA_SCHUNK`` bounds), the weights × values einsum in the stock's layout times the
    gate rows, ``linear_no_bias_out`` on the gathered slab."""
    import torch
    H, c = int(pwa.n_heads), int(pwa.c)

    def prep_fn(z_rows, g0, g1):
        return pwa.linear_no_bias_z(pwa.layernorm_z(z_rows)).permute(2, 0, 1)                 # [rows, N, H] -> [H, rows, N]

    def values_fn(m_chunk):
        mn = pwa.layernorm_m(m_chunk)
        v = pwa.linear_no_bias_mv(mn).reshape(*mn.shape[:-1], H, c)                          # [chunk, N, H, c]
        g = torch.sigmoid(pwa.linear_no_bias_mg(mn))                                          # [chunk, N, H*c]
        return v, g

    def attend_fn(w, state, g0, g1):
        v, g = state
        wv = torch.einsum("hij,mjhc->mihc", w, v)                                             # == '...ijh,...mjhc->...mihc' with heads first
        return (g[:, g0:g1].reshape(*wv.shape) * wv).reshape(*wv.shape[:-2], H * c)          # [chunk, q, H*c]

    def out_fn(o_full):
        return pwa.linear_no_bias_out(o_full)
    return prep_fn, values_fn, attend_fn, out_fn


def _msa_rows(mm, feats, z_sh, s_inputs, lay, *, triangle_attention: str, triangle_multiplicative: str, inplace_safe: bool, chunk_size):
    """== MSAModule.forward (pairformer.py:1403-1500; statement order: pair-weighted averaging + transition on ``m`` first, then the outer-product
    mean into ``z``, then the pair block) through the core's ``msa.msa_module_sharded`` with ``z`` this rank's rows and ``m`` this rank's
    TOKEN BLOCK ``[S, R, c_m]`` (``ROWPAIR_MSA_M_LAYOUT=token_sharded``, the default: the replicated sample — rank 0's draw on every rank —
    is cut at entry; the OPM column operand is one all-gather of the small ``[S, R, c]`` projection) or replicated (``=replicated``): per block ``m += PWA(m, z rows)``
    (``msa.pwa_rows``, S-chunks of ``min(msa_chunk_size, ROWPAIR_PWA_SCHUNK)``), ``m += transition`` (``msa.msa_transition_rows``, the stock's
    S-chunks inside), ``z += OPM(m)`` rows in place (``msa.opm_rows_budgeted``: the rank-agreed row block), the pair block on the shard."""
    import torch
    from opt_core.mem.rowpair import diffusion as _diff, msa as _msa, pairstack as _ps, shard as _shard
    m = mm._prepare_msa_sample(input_feature_dict=feats, s_inputs=s_inputs, z_token_dim=lay.N)
    if m.dtype != torch.float32 and z_sh.dtype == torch.float32:
        m = m.float()
    m = _diff.sync_replicated(m, "msa_sample", mode=_noise_sync())              # bcast (default): rank 0's MSA sample replaces every rank's own draw, then the guard; every rank still draws (RNG streams stay aligned)
    tok = _m_layout() == "token_sharded"
    if tok:                                                                     # m TOKEN-SHARDED: this rank's token block [S, R, c_m]; the replicated sample is released here
        m = m[..., lay.r0:lay.r0 + lay.R, :].contiguous()
    STATS["guards"] = STATS.get("guards", 0) + 1
    C = int(z_sh.shape[-1])
    schunk = _pwa_schunk()

    def msa_update_of(blk):
        st = blk.msa_stack
        cs = min(int(st.msa_chunk_size or 1 << 30), schunk)
        prep_fn, values_fn, attend_fn, out_fn = _pwa_fns(st.msa_pair_weighted_averaging)

        def transition(m_, _t0, _t1):                                                        # the stock's S-chunked transition of m (per-MSA-row statement)
            out = torch.empty_like(m_)
            for i0 in range(0, int(m_.shape[-3]), cs):
                out[i0:i0 + cs] = st.transition_m(m_[i0:i0 + cs])
            return out

        def msa_update(m_, z_):
            bias = _msa.pwa_bias_rows(prep_fn, z_, lay)                                          # [H, R, N]
            m_ += _msa.pwa_rows(m_, bias, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=out_fn, s_chunk=cs, **_m_layout_kw(_msa.pwa_rows))
            del bias
            m_ += _msa.msa_transition_rows(transition, m_, lay, shard_tokens=True if tok else _msa_trans_shard(), token_dim=-2, **_m_layout_kw(_msa.msa_transition_rows))
            return m_
        return msa_update

    def opm_of(blk):
        opm = blk.outer_product_mean_msa

        def opm_fn(m_, z_):
            ln = opm.layer_norm(m_)
            a, b = opm.linear_1(ln), opm.linear_2(ln)                                             # [S, N, c] (replicated m) or [S, R, c] (token-sharded m)
            if tok:
                b = _shard.unshard_rows(b, lay, dim=-2)                                           # the column operand of every rank's rows: ONE all-gather of the small [S, R, c] projection
            S = int(m_.shape[-3])

            def outer_fn(a_rows, b_full):
                o = torch.einsum("sic,sjd->ijcd", a_rows, b_full)
                return opm.linear_out(o.reshape(*o.shape[:-2], -1)) / (float(S) + opm.eps)
            return _msa.opm_rows_budgeted(a, b, lay, outer_fn, C_z=C, out=z_, add=True, **({"a_local": True} if tok else {}))
        return opm_fn

    def pair_block_of(blk):
        def pair_block(z_):
            fns = _pair_fns(blk.pair_stack, lay, triangle_attention=triangle_attention, chunk_size=chunk_size)
            z_out, _ = _ps.pair_block_(fns, z_, None, lay, transition_mask=False)
            STATS["direct_blocks"] += 1
            STATS["presharded_calls"] += 1
            STATS["blocks"] += 1
            return z_out
        return pair_block

    blocks = [dict(opm=opm_of(blk), pair_block=pair_block_of(blk), msa_update=msa_update_of(blk), opm_first=False) for blk in mm.blocks]

    def between(i):
        STATS["msa_blocks_rows"] += 1
    z_sh = _msa.msa_module_sharded(m, z_sh, lay, blocks, between_blocks=between)
    del m
    return z_sh


def _make_trunk_forward(orig):
    """``OpenDDE.get_pairformer_output`` under ``P > 1``: the pair track is born sharded (z_init rows: the two single projections, the relative
    position encoding materialized for the rows, the token bonds of the rows), stays sharded through every recycle (recycling embed row-local,
    template embedder rows, MSA module rows, the pairformer stack on the shard) and is all-gathered ONCE after the recycles for the heads."""
    def get_pairformer_output(self, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None):
        if not in_rank_process():
            return orig(self, input_feature_dict, N_cycle, inplace_safe=inplace_safe, chunk_size=chunk_size)
        import time
        import torch
        from opt_core.mem.rowpair import shard as _shard, trunk as _trunk
        s_inputs = self.input_embedder(input_feature_dict, inplace_safe=False, chunk_size=chunk_size)
        s_init = self.linear_no_bias_sinit(s_inputs)
        if s_init.dim() != 2:
            _note_unsharded(f"trunk: s_init {tuple(s_init.shape)} has a batch dim")
            return orig(self, input_feature_dict, N_cycle, inplace_safe=inplace_safe, chunk_size=chunk_size)
        N = int(s_init.shape[-2])
        lay = _layout(N)
        r0, r1 = lay.r0, lay.r0 + lay.R
        _guard_weights_once(self)
        t0 = time.perf_counter()
        STATS["feat_resident_gb"] = _feats_device_gb(input_feature_dict)                # the feature dict's DEVICE bytes before the pair track exists (host-resident keys excluded)
        z_init = _zinit_rows(self, input_feature_dict, s_init, lay)                     # [R, N, c_z] born per row block: never a second shard-sized addend, never a whole [N, N, *] operand
        zinit = _trunk.ShardPark(z_init, park=None, name="z_init", log=_log)      # ROWPAIR_PARK_ZINIT (BIG_TP exports 1): the shard PARKED in pinned host memory, its device
        del z_init                                                              # storage released; row blocks stream back inside the recycle statement (a CPU shard stays resident)
        STATS["zinit_where"], STATS["zinit_park_gb"], STATS["zinit_park_s"] = zinit.where, (round(zinit.nbytes / 1e9, 3) if zinit.park else 0.0), zinit.park_s
        z = None                                                                # cycle 0 reads a zeros ROW BLOCK (stock: z = zeros_like(z_init)); the shard is allocated once by the recycle
        s = torch.zeros_like(s_init)
        tm, ta = self.configs.triangle_multiplicative, self.configs.triangle_attention
        if self.template_embedder.n_blocks > 0:
            _slice_template_feats(input_feature_dict, lay, device=s_init.device)   # the [T, N, N, *] template pair features (host-resident under the runner) -> this rank's rows on the card ONCE
        _TRUNK["lay"] = lay
        STATS["trunk_rows"] += 1
        _apply_rank_threads()                                                   # the line's ROWPAIR_RANK_THREADS export is in the environment by now (no-op once applied / when unset)
        _record_schedule(lay)
        _mark("trunk_entry")
        dump_s = os.environ.get("ROWPAIR_HANG_DUMP_S", "").strip()              # diagnostic knob (off by default): dump every thread's Python stack to stderr
        if dump_s:                                                              # every <s> seconds while the sharded trunk runs — a rank desync is located by the two dumps
            import faulthandler
            faulthandler.dump_traceback_later(float(dump_s), repeat=True, file=sys.stderr)
        try:
            for _cycle in range(N_cycle):
                with torch.set_grad_enabled(False):
                    if s_init.is_cuda:                                         # per-recycle heartbeat per rank (a desync is located by the last cycle each rank reached)
                        torch.cuda.synchronize(s_init.device)
                    print(f"[{TAG}] ROWPAIR event=trunk_cycle rank={STATS.get('rank')} cycle={_cycle} N={N} R={lay.R} r0={r0} "
                          f"alloc_gib={(torch.cuda.memory_allocated() / 2**30 if s_init.is_cuda else 0):.2f} t={time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
                    z = _trunk.recycle_shard_(z, zinit, lambda zb: self.linear_no_bias_z_cycle(self.layernorm_z_cycle(zb)), lay)   # z = z_init + linear(LN(z_prev)) IN PLACE per row block
                    STATS["recycle_rows_calls"] += 1
                    if self.template_embedder.n_blocks > 0:
                        z = _template_rows(self.template_embedder, input_feature_dict, z, lay, triangle_attention=ta, triangle_multiplicative=tm,
                                           inplace_safe=inplace_safe, chunk_size=chunk_size)     # z += the template term IN PLACE per row block (core template_embed_rows)
                    _mark("after_template")
                    z = _msa_rows(self.msa_module, input_feature_dict, z, s_inputs, lay, triangle_attention=ta, triangle_multiplicative=tm,
                                  inplace_safe=inplace_safe, chunk_size=chunk_size)
                    _mark("after_msa")
                    s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                    s, z = self.pairformer_stack(s, z, pair_mask=None, triangle_multiplicative=tm, triangle_attention=ta,
                                               inplace_safe=inplace_safe, chunk_size=chunk_size)      # the shard is recognised: no gather
                    _mark("after_pairstack")
        finally:
            _TRUNK["lay"] = None
            zinit.release()                                                     # the parked z_init's host (and device) bytes return
            _template_rows_to_host(input_feature_dict)                          # the template pair row slabs leave the card: no stage after the trunk reads them
            if dump_s:
                faulthandler.cancel_dump_traceback_later()
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        STATS["trunk_s"] = round(STATS.get("trunk_s", 0.0) + time.perf_counter() - t0, 3)
        _mark("no_gather")
        _TRUNK["configs"] = self.configs                                         # the heads' bin parameters (confidence reducer) come from the model configs
        _ensure_model_wraps()                                                    # the structural stage's entry is wrapped NOW (a lever that replaced it after import is wrapped too)
        return s_inputs, s, RowShard(z.contiguous(), lay)                      # the residue pair track LEAVES the trunk as this rank's rows (no gather)
    get_pairformer_output._rowpair_tp = True
    return get_pairformer_output


# ------------------------------------------------------------------------ the trunk's exit: the residue pair track stays this rank's rows
CONF_TARGET = "opendde.model.modules.confidence"
EMB_TARGET = "opendde.model.modules.embedders"


def _make_relp_forward_rows_only(orig):
    """``RelativePositionEncoding.forward`` on a lazy feature under ``n_gpu > 1`` would materialize the whole ``[N, N, 139]`` one-hot: refused by
    name (every consumer on this line projects ROWS: the trunk's pair init, the diffusion conditioning rows); an eager tensor argument (a caller
    that built rows itself) passes through."""
    def forward(self, relp_feature):
        emb = sys.modules.get(EMB_TARGET)
        lazy_cls = getattr(emb, "LazyRelativePositionEncodingFeatures", None) if emb is not None else None
        if lazy_cls is not None and isinstance(relp_feature, lazy_cls):
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: RelativePositionEncoding.forward would materialize the whole [N, N, 139] relative-position one-hot on this rank "
                                 f"under n_gpu>1 (consumers project rows: relp.materialize(row_slice=...) + linear_no_bias)")
        return orig(self, relp_feature)
    forward._rowpair_tp = True
    return forward
TEMPLATE_ROWS_KEY = "_rowpair_template_rows"             # marker in the feature dict: the template pair features hold this rank's rows (sliced once at trunk entry)
TEMPLATE_PAIR_FEATS = ("template_distogram", "template_unit_vector", "template_pseudo_beta_mask", "template_backbone_frame_mask")   # [T, N, N(, *)]
PER_RANK_FEATS = TEMPLATE_PAIR_FEATS + (INPUT_ROWS_KEY, HOST_FEATS_KEY, TEMPLATE_ROWS_KEY)   # the feature-dict keys that differ across ranks BY DESIGN under P > 1 (this rank's rows of the
                                                          # template pair features + the row markers): left out of the cross-rank feature digest (feats_digest); every other key is replicated


# ----------------------------------------------------------------------------------------------------------------- rank data: the model-input features across ranks
FEATS_WHAT = "feats"                                      # the core's name of what is digested (``rankdata.assert_ranks_agree(what=)``): the census line reads ``[feats] rank <r> digest <d16> feats_ranks_equal=…``
FEATS_EQUAL = _rankdata.agree_word(FEATS_WHAT, None).partition("=")[0]   # LEVER census field ``yes`` | ``no`` | ``n/a`` (= the core's ``feats_ranks_equal``): every batch's digest agreed on every rank
FEATS_DIGEST = "feats_digest"                             # LEVER census field: this rank's digest (``rankdata.DIGEST_HEX`` digits) of the last batch's model-input features minus PER_RANK_FEATS
DATA_FORM = "rank0_bcast"                                 # the line's data form under P > 1 (one of the core's ``rankdata.DATA_FORMS``; LAUNCH / LEVER word ``data_form=``): rank 0 featurises every
                                                          # query item and the core broadcasts the tree; ranks > 0 build only their own template pair rows (tp_feats.rows_for_rank)
ERROR_ITEM = "erroritem"                                  # rank 0's status word (a core control word: nothing travels) when upstream's per-item guard turned a failing item into an
                                                          # error item — every rank returns the same error item and upstream's error path runs on every rank in step
ATOM_ARRAY_KEY = "_rowpair_atom_array"                    # where the item's AtomArray rides in the broadcast tree (a pickled leaf beside the feature tensors); never a feature-dict key


def feats_digest(fd) -> str:
    """The core's feature digest (``opt_core.mem.rowpair.rankdata.feature_digest``: sha256 over dtype / shape / bytes of every tensor leaf, keys
    sorted, non-tensor leaves canonical or excluded by name) of the model-input feature dict ``fd`` minus :data:`PER_RANK_FEATS`: equal on two
    ranks exactly when both hand the model the same replicated inputs."""
    return _rankdata.feature_digest(fd, exclude=PER_RANK_FEATS)


def _feats_census(fd) -> None:
    """Under ``P > 1``, once per (job, seed) batch at the runner's feature move, before any tensor reaches the card: this rank's :func:`feats_digest`
    compared with every other rank's by the core (``rankdata.assert_ranks_agree(mode="census")``): ONE stderr line per rank in the core's words,
    ``[opendde-opt] [feats] rank <r> digest <d16> feats_ranks_equal=yes|no ranks=<P> digests=<d16>,…``, and the LEVER census fields
    ``feats_ranks_equal`` / ``feats_digest`` / ``feats_batches`` / ``feats_unequal`` / ``feats_excluded``; then the contract of the ``rank0_bcast``
    data form (:func:`_rank0_item`): every rank hands the model rank 0's features, so a digest that differs on any rank is a refusal on EVERY rank
    (the core's ``refused: feats_ranks_differ: …``; upstream's per-item guard turns it into a failed item and a non-zero exit of every rank). A
    process whose environment names no rank world (``rank_world() == 1``: a caller driving the move itself) has nothing to compare and prints nothing."""
    if rank_world() <= 1:
        return
    _group()                                                                     # from the rank environment alone: the group exists from here on (the move and the trunk reuse it)
    digest = feats_digest(fd)
    equal = _rankdata.assert_ranks_agree(digest, what=FEATS_WHAT, mode="census", log=_log)   # the core's census line on this rank's stderr; never raises in census mode
    STATS["feats_batches"] += 1
    STATS["feats_unequal"] += 1 if equal is False else 0
    STATS[FEATS_DIGEST] = digest[:_rankdata.DIGEST_HEX]
    STATS[FEATS_EQUAL] = "no" if STATS["feats_unequal"] else _rankdata.agree_word(FEATS_WHAT, equal).partition("=")[2]   # yes | n/a (nothing was compared: no active group) — the core's word, never re-rendered here
    if equal is False:                                                           # the rank0_bcast form's contract: every rank hands the model rank 0's features — every rank saw the same
        raise _rankdata.differ_refusal(FEATS_WHAT, STATS.get("rank"), digest)     # digests, so every rank refuses by the core's name (``refused: feats_ranks_differ: …``); none is left in a collective


def data_form_fields() -> str:
    """The LAUNCH line's word for who featurises under P > 1: the core's ``data_form=rank0_bcast`` (``rankdata.data_form_word``)."""
    return _rankdata.data_form_word(DATA_FORM)


def _bcast_record(fb) -> None:
    """This rank's share of one ``rankdata.broadcast_features`` call in the census (GiB and seconds moved, rendezvous wait, RNG carry)."""
    STATS["feats_bcast_gib"] = round(STATS["feats_bcast_gib"] + fb.nbytes / 2 ** 30, 6)
    STATS["feats_bcast_s"] = round(STATS["feats_bcast_s"] + fb.bcast_s, 2)
    STATS["feats_wait_s"] = round(STATS["feats_wait_s"] + fb.wait_s, 2)
    if STATS["feats_rng"] != "carried":                                          # sticky: `carried` once any call carried rank 0's RNG state (an error item carries nothing and must not mask it)
        STATS["feats_rng"] = fb.rng


BCAST_MALFORMED = f"refused: {FEATS_WHAT}_bcast_malformed:"   # the receive-side refusal words: the core's receipt check (rankdata.check_received) prints exactly these for the
                                                            # tree's top level; the core exposes no public builder for them, so the prefix is composed ONCE here from FEATS_WHAT
                                                            # and the kit's own receive-side checks (item shape, item index, template precursors) raise :func:`_malformed`


def _malformed(key: str, why: str):
    """``refused: feats_bcast_malformed: <key> (<why>)`` — one grep finds every receive-side refusal, the core's and the kit's."""
    from opt_core.mem.rowpair import RowpairRefused
    return RowpairRefused(f"{BCAST_MALFORMED} {key} ({why})", lever="n_gpu")


def _make_getitem_rank0(orig):
    """``InferenceDataset.__getitem__`` in a rank process (P > 1): the ``rank0_bcast`` data form (:func:`_rank0_item`); the stock method at P = 1."""
    def __getitem__(self, index):
        if not in_rank_process():
            return orig(self, index)
        return _rank0_item(self, index, orig)
    return __getitem__


def _rank0_item(ds, index: int, orig):
    """One query item under P > 1. Rank 0 runs the stock statements (``orig``: ``process_one`` — tokenisation, MSA / template featurisation, ligand
    conformers — inside upstream's own per-item guard, so a failing item is an error item, not a raise; its featurizer bears rank 0's template pair
    ROWS as on every rank process, tp_feats) and hands the item to every other rank through the core (``rankdata.broadcast_features``: the ranks
    meet at the store while rank 0 works — no collective pending — then a status word, the tree's meta, its tensors, a receipt check, and rank 0's
    host RNG state carried onto the receivers so every rank enters the model with the RNG state a featurising rank would have). Ranks > 0 never
    featurise: they receive the tree minus rank 0's own rows (:data:`PER_RANK_FEATS`) and build ONLY their rows of the four template pair features
    from the template precursors it carries (``tp_feats.rows_for_rank``). An error item on rank 0 is the same error item on every rank (status
    :data:`ERROR_ITEM`; nothing travels; ranks > 0 record rank 0's message, truncated to the core's status length). The group is created here from
    the rank environment: after upstream built its dataloader (whose sampler therefore read no group and keeps every item on every rank), before
    any model work. The item is read in the rank's main process (upstream's ``num_workers: 0``); a DataLoader worker is refused by name."""
    import torch.utils.data
    from opt_core.mem.rowpair import RowpairRefused
    if torch.utils.data.get_worker_info() is not None:                            # a loader worker would join the rendezvous as a second `rank r`
        raise RowpairRefused(f"{LEVER}: refused: {DATA_FORM} reads every item in the rank process itself (num_workers=0, upstream's inference default); "
                             f"this item is being read in DataLoader worker {torch.utils.data.get_worker_info().id}")
    P, r = _group()
    call = STATS["feats_bcast_calls"]
    STATS["feats_bcast_calls"] += 1
    key = f"{FEATS_WHAT}/{call}"                                                  # one store key per item (the core keeps every key of a run)
    if r == 0:
        try:                                                                     # ONE guard over the stock statements AND the tree this rank hands on: whatever raises here goes out
            data, atom_array, error_message = orig(ds, index)                    # as the `failed:` status first, so the receivers refuse by name at once instead of waiting for teardown
            if error_message:                                                    # upstream's error item: the receivers learn it through the status word; upstream's error path runs on every rank
                tree, status = None, f"{ERROR_ITEM} {int(index)} " + " ".join(str(error_message).split())[:_rankdata.STATUS_MAX]
            else:
                fd = data["input_feature_dict"]
                tree, status = {**{k: v for k, v in data.items() if k != "input_feature_dict"},
                                "input_feature_dict": {k: v for k, v in fd.items() if k not in PER_RANK_FEATS},   # rank 0's own template pair rows and row marker stay home
                                ATOM_ARRAY_KEY: atom_array}, None
        except BaseException as e:                                               # nothing upstream's per-item guard let through raises; the statements around it and the tree build might
            _rankdata.broadcast_features(None, key=key, status=_rankdata.status_word(e), what=FEATS_WHAT)   # raises ``refused: feats_rank0_failed: …`` here as on the receivers
            raise                                                                # not reached while a group is active (the call above raises); kept so no path continues past a failure
        fb = _rankdata.broadcast_features(tree, key=key, status=status, what=FEATS_WHAT)   # carry_rng: rank 0's post-featurisation host RNG state rides along with a tree
        if status is not None:
            STATS["feats_error_items"] += 1
        _bcast_record(fb)
        return data, atom_array, error_message                                   # rank 0's own objects, untouched
    fb = _rankdata.broadcast_features(None, key=key, what=FEATS_WHAT)               # waits at the store for rank 0's word, then receives (or refuses by rank 0's name)
    _bcast_record(fb)
    if fb.status.split()[0] == ERROR_ITEM:                                       # rank 0's item failed inside upstream's guard: the same error item here (rank 0's message, truncated)
        STATS["feats_error_items"] += 1
        _w, idx0, msg = (fb.status.split(" ", 2) + ["", ""])[:3]
        if str(idx0) != str(int(index)):
            raise _malformed("sample_index", f"rank 0 reported an error item for item {idx0} while this rank's sampler asked for item {int(index)}")
        sample = ds.inputs[index] if 0 <= int(index) < len(ds.inputs) else {}
        return {"sample_name": sample.get("name", f"job_{index}"), "sample_index": index}, None, f"rank 0: {msg}"
    from . import tp_feats as _tf
    data = fb.feats
    if not isinstance(data, dict) or not isinstance(data.get("input_feature_dict"), dict) or ATOM_ARRAY_KEY not in data:
        raise _malformed("input_feature_dict" if isinstance(data, dict) and ATOM_ARRAY_KEY in data else ATOM_ARRAY_KEY if isinstance(data, dict) else "<tree>",
                         f"rank 0's item {int(index)} arrived as {sorted(map(str, data))[:8] if isinstance(data, dict) else type(data).__name__}; "
                         f"an item carries an input_feature_dict mapping and {ATOM_ARRAY_KEY}")
    if int(data.get("sample_index", index)) != int(index):
        raise _malformed("sample_index", f"rank 0 handed item {data.get('sample_index')} while this rank's sampler asked for item {int(index)}")
    atom_array = data.pop(ATOM_ARRAY_KEY)
    _tf.rows_for_rank(data["input_feature_dict"])                               # this rank's rows of the template pair features + the row marker, from the received precursors
    return data, atom_array, ""


class RowShard:
    """The residue pair representation after the trunk under ``n_gpu > 1``: this rank's rows ``z [R, N, c_z]`` and the row layout. It is what
    ``OpenDDE.get_pairformer_output`` returns in a rank process; its consumers are wrapped: the structural-token stage receives the full
    residue ``z`` as its row shard (:mod:`opendde_opt.tp_struct`), the distogram and the confidence head run on the rows (:func:`_make_distogram_rows`,
    :func:`_make_conf_forward`); ``contact_rows`` holds this rank's rows ``[R, N]`` of the contact probabilities once the distogram stage ran (the
    confidence reducer consumes them). PARKING (``ROWPAIR_PARK_ZRES=1``, the BIG_TP line's export; opt_core ``trunk.ShardPark``): the structural
    stage calls :meth:`park` once it has fetched the residue rows it reads — the shard moves to pinned host memory and its device storage is
    released for the structural stage and the diffusion roll-out (neither reads residue ``z`` again); the first reader afterwards (the distogram /
    confidence rows: the ``z`` property) streams it back once. ``numel()`` / ``shape`` describe the local rows; ``cpu()`` refuses by name (a
    trunk checkpoint of one rank's rows is not a trunk checkpoint)."""
    __slots__ = ("_z", "lay", "contact_rows", "_park", "_released")

    def __init__(self, z, lay):
        self._z, self.lay, self.contact_rows, self._park, self._released = z, lay, None, None, None

    def release_device(self, why: str) -> float:
        """Drop this shard's device rows for good (the STRUCTURAL shard once the diffusion conditioning consumed it: nothing after the
        roll-out's priming reads structural ``z`` — the heads run on the residue shard). Returns the GB released; a later ``z`` read is refused
        BY NAME (never a silent recompute). Idempotent; a parked shard releases its park too."""
        gb = 0.0
        if self._park is not None:
            park, self._park = self._park, None
            park.release()
        if self._z is not None:
            gb = self._z.numel() * self._z.element_size() / 1e9
            self._released = (tuple(self._z.shape), str(why), STATS.get("last_mark"))
            self._z = None
        return gb

    @property
    def z(self):
        if self._z is None and self._park is None and self._released is not None:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: pair shard {self._released[0]} was released at {self._released[2]} ({self._released[1]}) and is read again "
                                 f"at {STATS.get('last_mark')} (ODDE_TP_KEEP_ZSTRUCT=1 keeps the structural shard resident)")
        if self._park is not None:                                              # parked: the first reader after the structural stage + roll-out unparks (one H2D)
            park, self._park = self._park, None
            self._z = park.full()
            park.release()
            STATS["zres_unpark"] += 1
            STATS["zres_unpark_at"] = STATS.get("last_mark")
            _log(f"ROWPAIR event=zres_unpark rank={STATS.get('rank')} at={STATS.get('last_mark')} gb={self._z.numel() * self._z.element_size() / 1e9:.2f}")
        return self._z

    @property
    def parked(self) -> bool:
        return self._park is not None

    def park(self) -> bool:
        """Park the residue shard on the host (``ROWPAIR_PARK_ZRES=1``; a CPU shard stays resident): device storage released now, streamed back by
        the next ``z`` read. Returns whether a park happened. Idempotent."""
        if self._park is not None or not _dist_env_flag("ROWPAIR_PARK_ZRES"):
            return False
        from opt_core.mem.rowpair import trunk as _trunk
        self._park = _trunk.ShardPark(self._z, park=True, name="z_res", log=_log)
        STATS["zres_park"] += 1
        STATS["zres_where"], STATS["zres_park_gb"] = self._park.where, (round(self._park.nbytes / 1e9, 3) if self._park.park else 0.0)
        _log(f"ROWPAIR event=zres_park rank={STATS.get('rank')} where={self._park.where} gb={STATS['zres_park_gb']} at={STATS.get('last_mark')}")
        return bool(self._park.park)

    @property
    def shape(self):
        if self._z is None and self._park is None and self._released is not None:
            return self._released[0]
        return self._z.shape if self._park is None else self._park.shape

    def numel(self) -> int:
        import math
        return int(math.prod(self.shape))

    def cpu(self):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: a trunk checkpoint (ODDE_OFFLOAD_CKPT_DIR / ODDE_OFFLOAD_RESUME) of a row shard under n_gpu>1")


def _pwa_schunk() -> int:
    from opt_core.mem.rowpair import dist as _dist
    v = _dist.env_int(ENV_PWA_SCHUNK, PWA_SCHUNK_DEFAULT) if hasattr(_dist, "env_int") else int(os.environ.get(ENV_PWA_SCHUNK) or PWA_SCHUNK_DEFAULT)
    if v < 1:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_PWA_SCHUNK}={v} (must be >= 1)")
    return v


def _det_level_on() -> bool:
    """The deterministic recipe is active in this rank process (det.py: ``--deterministic true`` on the stock + ``ODDE_SERVED_DETERMINISTIC``)."""
    if os.environ.get("ODDE_SERVED_DETERMINISTIC", "").strip() == "1":
        return True
    try:
        import torch
        return bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        return False


def _noise_sync() -> str:
    """The replicated-draw sync policy of this line, BY DET LEVEL unless ``ROWPAIR_DIFF_NOISE_SYNC`` pins it: ``--det 0`` -> ``bcast`` (rank 0
    authoritative at every sync point: the MSA sample per cycle, the per-denoiser-call diffusion state, every noise draw — det-0 replicated
    kernels are not bitwise across ranks at large N, so each rank adopts rank 0's tensor); ``--det 1`` -> ``guard`` (strict: every rank's own
    draw / state proven bitwise identical; a mismatch is a real defect and refuses by name). ``guard`` at det 0 is the named strict opt-in;
    ``off`` is refused on this line. The sampler exit adopts rank 0's coordinates at both levels (``diff_rank_spread_A`` recorded)."""
    v = (os.environ.get(ENV_NOISE_SYNC) or "").strip().lower()
    if not v:
        return "guard" if _det_level_on() else "bcast"
    if v not in ("bcast", "guard"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_NOISE_SYNC}={v!r} (bcast | guard on this line)")
    return v


def _layout_b():
    """``ROWPAIR_LAYOUT_B``: an explicit row-block size for the layout grid (``Layout.auto`` with that single candidate; refused by name when the
    grid would replicate or leave a rank empty) instead of the largest of the core's candidates — a named schedule knob (``layout_B_source=env``
    in the census), never derived from this rank's state."""
    v = os.environ.get(ENV_LAYOUT_B, "").strip()
    if not v:
        return None
    if not v.isdigit() or int(v) < 1:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_LAYOUT_B}={v!r} (a positive integer)")
    return int(v)


def _m_layout() -> str:
    """``ROWPAIR_MSA_M_LAYOUT``: the MSA representation ``m [S, N, c_m]`` is TOKEN-SHARDED (default ``token_sharded``: this rank holds its token
    block; the core's row statements gather only the non-local operand) or stays REPLICATED on every rank (``replicated``, named
    ``msa_m=replicated`` in the census). Any other value is refused by name."""
    v = (os.environ.get(ENV_MSA_M_LAYOUT) or "token_sharded").strip()
    if v not in ("replicated", "token_sharded"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_MSA_M_LAYOUT}={v!r} (replicated | token_sharded)")
    return v


def _m_layout_kw(fn) -> dict:
    """``m_layout=`` for a core MSA row statement that accepts it; a core without the argument runs replicated ``m`` only, and asking it for
    ``token_sharded`` is refused by name (never silently replicated)."""
    import inspect
    want = _m_layout()
    if "m_layout" in inspect.signature(fn).parameters:
        return {"m_layout": want}
    if want != "replicated":
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_MSA_M_LAYOUT}={want} but the pinned core's {fn.__name__} has no m_layout argument")
    return {}


def _msa_trans_shard() -> bool:
    """The MSA transition of the replicated ``m``: token-sharded on this rank's token block + one all-gather (``ROWPAIR_MSA_TRANS_SHARD=1``, the
    default: the sharded statement) or the whole-``m`` statement on every rank (``=0``: replicated compute, named ``msa_transition=replicated``
    in the core's schedule census). Any other value is refused by name."""
    v = os.environ.get(ENV_MSA_TRANS_SHARD, "1")
    if v not in ("0", "1"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_MSA_TRANS_SHARD}={v!r} (0 | 1)")
    return v == "1"


def _hostgather_rows(lay) -> int:
    """Row-slab height of the residue-row ring (``tp_struct.fetch_rows_ring``): ``ROWPAIR_HOSTGATHER_ROWS`` or the layout's block ``B`` — a
    FIXED value (never derived from this rank's free bytes), so every rank runs the same number of ring exchanges."""
    v = int(os.environ.get(ENV_RING_ROWS) or 0) or int(lay.B)
    if v < 1:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_RING_ROWS}={v} (must be >= 1)")
    return v


def _record_schedule(lay) -> None:
    """The adapter's own schedule facts (beside the core's ``evidence.schedule_fields``): every block size here is fixed or env-given."""
    # diff_noise: per denoiser call the INPUT STATE x_noisy = rank 0's (bcast, det 0) or proven bitwise (guard, det 1); sampler exit: rank 0's coordinates + diff_rank_spread_A
    STATS["schedule"] = {"layout_B": int(lay.B), "layout_B_source": "env" if os.environ.get(ENV_LAYOUT_B, "").strip() else "auto", "pwa_schunk": _pwa_schunk(), "pwa_schunk_source": "env" if os.environ.get(ENV_PWA_SCHUNK) else "fixed",
                         "hostgather_rows": _hostgather_rows(lay), "hostgather_rows_source": "env" if os.environ.get(ENV_RING_ROWS) else "layout_B",
                         "msa_transition": "token_sharded" if _msa_trans_shard() else "replicated", "diff_noise": ("bcast_rank0_state" if _noise_sync() == "bcast" else "guard_strict") + ("" if os.environ.get(ENV_NOISE_SYNC, "").strip() else f"(det{int(_det_level_on())}_default)"),
                         "msa_m": _m_layout(), "conf_logits": f"reducer:{_conf_finish()}", "conf_rows": _conf_rows(lay), "conf_rows_source": "env" if os.environ.get(ENV_CONF_ROWS) else "fixed",
                         "contact_probs": "rows", "struct_stage": "rowshard(expand_rows+refine_rows)+zcond_rows",
                         "dit_queries": "local_rows", "atom_band": "rows",
                         "trunk_init": "row_blocks(trunk_init_rows)", "input_feats": _input_feats_word(), "inputs": _inputs_word(),
                         "diffz": "replaced_by_rowpair", "struct_pair_dtype": _struct_pair_dtype(), "zinit": ("host_parked" if _dist_env_flag("ROWPAIR_PARK_ZINIT") else "device"), "recycle": "rows_inplace", "template": "embed_rows_inplace", "zres": ("host_parked_struct_rollout" if _dist_env_flag("ROWPAIR_PARK_ZRES") else "device"), "template_rows_after_trunk": "host", "struct_trimul_rb": (os.environ.get("ODDE_TP_STRUCT_TRIMUL_RB") or "budget"), "nccl_timeout_s": (os.environ.get("ROWPAIR_NCCL_TIMEOUT_S") or "core_default"), "transpose_inplace": (os.environ.get("ROWPAIR_TRANSPOSE_INPLACE") or "0")}                      # the offload unit's diffusion-pair hoist is served by the row-sharded roll-out (tp_diffusion: z_cond rows once per roll-out)


def _zinit_rows(model, feats, s_init, lay):
    """``z_init`` of ``get_pairformer_output`` (opendde.py: the two single projections' outer sum + the relative position encoding + the token-bond
    projection) BORN as this rank's shard through the core's ``trunk.init_pair_shard``: per block of ``trunk_init_rows`` global rows ``[g0, g1)``
    the statement reads the lazy relp's ROWS (``relp.materialize(row_slice)``: ``[rows, N, 139]``, never ``[N, N, 139]``) and the token bonds'
    rows; the block transient is the only extra memory (``ROWPAIR_INIT_ROWS`` / ``ROWPAIR_ROWBLK_MB``, P-invariant, in the core schedule census)."""
    from opt_core.mem.rowpair import trunk as _trunk
    zi = model.linear_no_bias_zinit1(s_init)                                     # [N, c_z] each: the per-token halves of the outer sum
    zj = model.linear_no_bias_zinit2(s_init)
    relp = feats["relp"]
    bonds = feats["token_bonds"]
    rpe = model.relative_position_encoding

    def rows_fn(g0, g1):
        relp_rows = relp.materialize(row_slice=slice(g0, g1)) if hasattr(relp, "materialize") else relp[g0:g1]
        blk = _trunk.outer_sum_rows(zi, zj, g0, g1) + rpe.linear_no_bias(relp_rows)
        del relp_rows
        STATS["zinit_blocks"] += 1
        return blk + model.linear_no_bias_token_bond(_trunk.feature_rows(bonds, g0, g1, zi.device, row_dim=-2).unsqueeze(dim=-1))
    n_relp = int(getattr(rpe.linear_no_bias, "in_features", 0) or 0)
    return _trunk.init_pair_shard(lay, rows_fn, like=zi, channels=3 * int(zi.shape[-1]) + n_relp)   # transient per block: the block, the outer sum, the relp projection + its one-hot rows


def _inputs_word() -> str:
    """Census word ``inputs``: the N²-shaped input features per rank — the template pair features BORN as rows (rows / N, GB born vs the GB
    the dense ``[T, N, N, 44]`` fp32 tensors cost per rank), ``token_bonds`` replicated by design (4·N² B), the MSA features ``[S, N]``
    replicated, the atom-level ``bond_mask`` transient left to the stock featurizer (read by no consumer; dropped by ``drop_bond_mask``)."""
    lay = STATS.get("layout") or {}
    born = "born_rows" if STATS.get("template_rows_born") else ("host_rows_cut_at_move" if STATS.get("feat_host_keys") else "none_seen")
    return (f"template_pair={born}(R={lay.get('R', '?')},N={lay.get('N', '?')}):{STATS.get('inputs_born_gb', 0.0):.3f}GB/full={STATS.get('inputs_full_gb', 0.0):.3f}GB "
            f"token_bonds=replicated(4*N^2B) msa=replicated([S,N]) bond_mask=stock_transient")


def _input_feats_word() -> str:
    """Census word for the template pair features' path: ``template_pair_host_rows`` (the runner's move holds this rank's host row slabs) or
    ``template_pair_rows_at_trunk_entry`` (the runner module is not in this process: a caller feeds the model whole host tensors and the trunk
    entry cuts them; whole tensors on the card are refused by name there)."""
    ps = STATS.get("input_patches") or []
    return "template_pair_host_rows" if ps and all(p.state == "installed" for p in ps) else "template_pair_rows_at_trunk_entry"


def _feats_device_gb(feats) -> float:
    """GB of CUDA tensors held by the feature dict (the replicated-by-design input features on this rank; host-resident keys excluded)."""
    import torch
    n = 0
    for v in feats.values():
        if torch.is_tensor(v) and v.is_cuda:
            n += v.numel() * v.element_size()
    return round(n / 1e9, 3)


def _make_to_device_host_feats(orig):
    """The runner's ``to_device(data, device)`` under ``P > 1``: the ``[T, N, N(, *)]`` template pair features of ``input_feature_dict`` never
    reach a rank's card whole — at the move this rank keeps ITS ROWS ``[T, R, N(, *)]`` of them on the HOST (``trunk.feature_rows``; the
    layout is the trunk's ``Layout.auto(N, P, rank)``) and every other feature moves as the stock statement does; the trunk entry moves the row
    slabs to the card (:func:`_slice_template_feats`). Marker :data:`HOST_FEATS_KEY`; census ``feat_host_keys`` / ``template_full_gb`` (the
    replicated bytes that stayed off the card) / ``feat_host_gb`` (the row slabs held). First, once per batch, the cross-rank feature census
    (:func:`_feats_census`: every rank's digest of the features it is about to move, compared and named)."""
    def to_device(obj, device, non_blocking=False):
        import torch
        from opt_core.mem.rowpair import trunk as _trunk
        if not in_rank_process() or not isinstance(obj, dict):
            return orig(obj, device, non_blocking=non_blocking)
        fd = obj.get("input_feature_dict") if isinstance(obj.get("input_feature_dict"), dict) else obj
        already = fd.get(HOST_FEATS_KEY) is not None                            # idempotent: row slabs held by an earlier call pass through untouched
        if not already:
            _feats_census(fd)                                                   # the cross-rank digest of this batch's features as the featurizer handed them on (P > 1; one line per rank)
        held = {k: fd[k] for k in TEMPLATE_PAIR_FEATS if torch.is_tensor(fd.get(k)) and fd[k].dim() >= 3}
        if not held:
            return orig(obj, device, non_blocking=non_blocking)
        rest = {k: v for k, v in fd.items() if k not in held and k != INPUT_ROWS_KEY}   # the featurizer's row marker never moves to the card
        moved = orig(rest if fd is obj else {**obj, "input_feature_dict": rest}, device, non_blocking=non_blocking)
        out_fd = moved if fd is obj else moved["input_feature_dict"]
        from opt_core.mem.rowpair import RowpairRefused
        from . import tp_feats as _tf
        born = None if already else _tf.born_rows(fd)                            # the featurizer bore this rank's rows (marker [r0, R, N])
        lay = None
        full = rows_b = 0
        for k, v in held.items():
            hv = v if v.device.type == "cpu" else v.cpu()
            if born is not None:                                                 # rows born in the data pipeline: cross-check against THIS rank's layout, keep on the host
                r0, R, N = born
                lay = lay or _layout(N)
                if (int(lay.r0), int(lay.R)) != (r0, R):
                    raise RowpairRefused(f"{LEVER}: refused: template rows {k} born for rows [{r0}, {r0 + R}) but this rank's layout is [{lay.r0}, {lay.r0 + lay.R}) of N={N}")
                if hv.dim() < 3 or int(hv.shape[1]) != R or int(hv.shape[2]) != N:
                    raise RowpairRefused(f"{LEVER}: refused: row-born template feature {k} {tuple(hv.shape)} is not [T, R={R}, N={N}(, *)]")
                full += _tf.full_pair_bytes(1, N) // sum(_tf.PAIR_CHANNELS.values()) * _tf.PAIR_CHANNELS[k] * int(hv.shape[0])   # what the dense key would have cost
            elif not already:
                N = int(hv.shape[1])
                if hv.dim() < 3 or int(hv.shape[2]) != N:
                    raise RowpairRefused(f"{LEVER}: refused: template feature {k} {tuple(hv.shape)} is not [T, N, N(, *)]")
                if os.environ.get(ENV_ALLOW_UNSHARDED) != "1":                  # a WHOLE [T, N, N, *] built in this rank process = the featurizer is not patched: never silently sliced
                    raise RowpairRefused(f"{LEVER}: refused: template feature {k} {tuple(hv.shape)} was built WHOLE on this rank's host (the featurizer "
                                         f"{_tf.TEMPL_TARGET}.Templates.as_opendde_dict is not patched in this process); {ENV_ALLOW_UNSHARDED}=1 opts in to cutting it here")
                _note_unsharded(f"template feature {k} built whole on the host, cut to rows at the move")
                lay = lay or _layout(N)
                full += hv.numel() * hv.element_size()
                hv = _trunk.feature_rows(hv, lay.r0, lay.r0 + lay.R, None, row_dim=1 - hv.dim()).contiguous()   # this rank's rows, on the host
            out_fd[k] = hv
            rows_b += hv.numel() * hv.element_size()
        out_fd[HOST_FEATS_KEY] = fd[HOST_FEATS_KEY] if already else ("rows", int(lay.r0), int(lay.R))
        out_fd.pop(_tf.ROWS_KEY, None)
        if not already:
            STATS["feat_host_keys"] += len(held)
            STATS["template_full_gb"] = round(full / 1e9, 6)
            STATS["feat_host_gb"] = round(rows_b / 1e9, 6)
            if born is not None:
                STATS["template_rows_born"] += len(held)
            STATS["inputs_born_gb"] = round(rows_b / 1e9, 6)
            STATS["inputs_full_gb"] = round(full / 1e9, 6)
        return moved
    return to_device


def _slice_template_feats(feats, lay, device=None) -> None:
    """The template pair features of the feature dict -> this rank's rows ``[T, R, N(, *)]`` on ``device``: the HOST row slabs the runner held
    (:func:`_make_to_device_host_feats`, marker :data:`HOST_FEATS_KEY`) move to the card as they are; whole ``[T, N, N(, *)]`` tensors (a caller
    that bypassed the runner) are cut to the rows through the core's ``trunk.feature_rows`` (a contiguous row slab is what moves; the full
    tensors are released). Idempotent (marker :data:`TEMPLATE_ROWS_KEY`). ``template_aatype [T, N]`` stays whole (row and column one-hots)."""
    if "template_aatype" not in feats:
        return
    import torch
    from opt_core.mem.rowpair import RowpairRefused, trunk as _trunk
    if feats.get(TEMPLATE_ROWS_KEY):                                            # rows already cut (an earlier trunk on this dict): host rows move back to the card as they are
        h2d = 0
        for k in TEMPLATE_PAIR_FEATS:
            tt = feats.get(k)
            if device is not None and torch.is_tensor(tt) and tt.device != torch.device(device) and str(device) != "cpu":
                feats[k] = tt.to(device, non_blocking=False)
                h2d += tt.numel() * tt.element_size()
        STATS["template_rows_h2d_gb"] = round(STATS["template_rows_h2d_gb"] + h2d / 1e9, 3)
        return
    r0, r1 = lay.r0, lay.r0 + lay.R
    held = feats.get(HOST_FEATS_KEY)
    if held is not None and (tuple(held[:1]) != ("rows",) or int(held[1]) != r0 or int(held[2]) != lay.R):
        raise RowpairRefused(f"{LEVER}: refused: host template rows {held} do not match this rank's layout rows [{r0}, {r1})")
    n, h2d = 0, 0
    for k in TEMPLATE_PAIR_FEATS:
        t = feats.get(k)
        if held is not None:
            if t is None or not torch.is_tensor(t) or t.dim() < 3 or int(t.shape[1]) != lay.R or int(t.shape[2]) != lay.N:
                raise RowpairRefused(f"{LEVER}: refused: host template rows {k} {None if t is None else tuple(t.shape)} are not [T, R={lay.R}, N={lay.N}(, *)]")
            rows_t = t if device is None else t.to(device, non_blocking=False)
        else:
            if t is None or not torch.is_tensor(t) or t.dim() < 3 or int(t.shape[1]) != lay.N or int(t.shape[2]) != lay.N:
                raise RowpairRefused(f"{LEVER}: refused: template feature {k} {None if t is None else tuple(t.shape)} is not [T, N={lay.N}, N(, *)]")
            if t.is_cuda and os.environ.get(ENV_ALLOW_UNSHARDED) != "1":         # a whole [T, N, N, *] on the card = the runner's move was not patched: never silently resident
                raise RowpairRefused(f"{LEVER}: refused: template feature {k} {tuple(t.shape)} reached the card WHOLE (the runner's to_device is not "
                                     f"patched in this process: {RUNNER_TARGET}); {ENV_ALLOW_UNSHARDED}=1 opts in to cutting it on the card")
            if t.is_cuda:
                _note_unsharded(f"template feature {k} whole on the card")
            rows_t = _trunk.feature_rows(t, r0, r1, device, row_dim=1 - t.dim(), non_blocking=False).contiguous()
        if rows_t.device != t.device:
            h2d += rows_t.numel() * rows_t.element_size()
        feats[k] = rows_t
        n += 1
    feats[TEMPLATE_ROWS_KEY] = True
    feats.pop(HOST_FEATS_KEY, None)
    STATS["template_rows_h2d_gb"] = round(STATS["template_rows_h2d_gb"] + h2d / 1e9, 3)
    STATS["template_feats_rows"] += n


def _struct_trimul_rb():
    """The structural refiner's streamed tri-mult slab rows: ``ODDE_TP_STRUCT_TRIMUL_RB`` (a multiple of 8, P-invariant; ``0`` /
    unset = the core's budgeted ``RB``); malformed values refuse by name. Census word ``struct_trimul_rb``."""
    v = (os.environ.get(ENV_STRUCT_TRIMUL_RB) or "").strip()
    if v in ("", "0"):
        return None
    if not v.isdigit() or int(v) < 8 or int(v) % 8:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_STRUCT_TRIMUL_RB}={v!r} (a multiple of 8, >= 8; 0 for the core's budgeted RB)")
    return int(v)


def _template_rows_to_host(feats) -> None:
    """After the last trunk cycle the template pair ROW slabs ``[T, R, N(, *)]`` leave the card (back to host tensors in the feature dict; the next
    trunk on the same dict moves them in again): no stage after the trunk reads them — the structural stage reads the structural features, the
    diffusion conditioning the structural pair rows, the distogram / confidence head the residue rows (tests/test_tp_seams_cpu.py proves the
    downstream stages equal the dense pipeline with the slabs gone)."""
    import torch
    if not feats.get(TEMPLATE_ROWS_KEY):
        return
    freed = 0
    for k in TEMPLATE_PAIR_FEATS:
        tt = feats.get(k)
        if torch.is_tensor(tt) and tt.is_cuda:
            freed += tt.numel() * tt.element_size()
            feats[k] = tt.to("cpu")
    STATS["template_rows_released_gb"] = round(STATS["template_rows_released_gb"] + freed / 1e9, 3)


def _make_expand_rows(inner):
    """``OpenDDE.expand_to_structural_tokens`` fed from a :class:`RowShard`: the structural stage runs on THIS RANK'S STRUCTURAL ROWS
    (:func:`tp_struct.expand_to_structural_tokens_rows`: the expansion reads the K residue rows its structural rows need, fetched around the ring;
    the refiner runs through the core pair-block driver; ``z_struct`` leaves as a ``RowShard`` of the structural layout and the structural extra
    attention bias as this rank's rows) — the residue ``z`` is never gathered, on host or device. Expansion disabled = the stock's equality: the
    residue ``RowShard`` is the diffusion stage's pair source."""
    def expand_to_structural_tokens(self, input_feature_dict, s_inputs, s, z, inplace_safe=False, chunk_size=None, lazy_relp=False):   # the pin's signature (opendde.py:422-431; the loop passes lazy_relp=True, chunk_size=structural_refiner_chunk_size)
        if not isinstance(z, RowShard) or not self.enable_structural_token_expansion:
            return inner(self, input_feature_dict, s_inputs, s, z, inplace_safe=inplace_safe, chunk_size=chunk_size, lazy_relp=lazy_relp)
        from . import tp_struct as _ts
        STATS["trunk_exit"] = "rowshard"
        _mark("diffusion_start")                                                # the structural stage (expansion, refiner) starts here on this line; then the diffusion conditioning
        return _ts.expand_to_structural_tokens_rows(self, input_feature_dict, s_inputs, s, z, chunk_size=chunk_size, inplace_safe=inplace_safe, lazy_relp=lazy_relp)
    expand_to_structural_tokens._rowpair_tp = True
    return expand_to_structural_tokens



def _guard_weights_once(model) -> None:
    """Once per rank process: a checksum of the model's parameters proven identical on every rank (``trunk.guard_replicated``: one all-reduce;
    a per-rank weight desync is refused by name instead of folding garbage)."""
    if STATS.get("weights_guarded"):
        return
    import torch
    from opt_core.mem.rowpair import trunk as _trunk
    with torch.no_grad():
        chk = torch.stack([p.detach().double().sum() for p in model.parameters()])
    _trunk.guard_replicated(chk, "model_parameters")
    STATS["weights_guarded"] = True


def _n_sample_per_call(model) -> int:
    """Samples per denoiser call in the stock sampler: ``min(N_sample, sample_diffusion_chunk_size)`` (generator.sample_diffusion chunks the
    samples by ``configs.infer_setting.sample_diffusion_chunk_size``)."""
    cfg = model.configs
    n = int(getattr(cfg.sample_diffusion, "N_sample", 1) if hasattr(cfg, "sample_diffusion") else 1)
    ch = getattr(getattr(cfg, "infer_setting", None), "sample_diffusion_chunk_size", None)
    return max(1, min(n, int(ch)) if ch else n)


def _make_diffusion_cache_rows(inner):
    """``OpenDDE.prepare_diffusion_cache_for_sampling`` under ``n_gpu > 1``: the diffusion pair conditioning of THIS RANK'S structural rows
    (``tp_diffusion.prime``: z_cond rows, per-block pair-bias rows, the schedule) and the atom encoder's cache with the token-pair term from the
    row band (``tp_diffusion.prepare_cache_sharded``) — nothing ``[N_st, N_st, *]`` on a device. ``z`` is the structural pair source the
    structural stage produced (the offload unit's host pair, a tensor, or a RowShard)."""
    def prepare_diffusion_cache_for_sampling(self, *, input_feature_dict, z, foldcp_mesh=None, diffusion_z_spec=None, **extra):
        if STATS["n_gpu"] <= 1 or not STATS.get("group"):
            return inner(self, input_feature_dict=input_feature_dict, z=z, foldcp_mesh=foldcp_mesh, diffusion_z_spec=diffusion_z_spec, **extra)
        if foldcp_mesh is not None:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: a Fold-CP mesh under n_gpu>1 (the row-sharded diffusion stage is this line's only multi-GPU path)")
        import torch
        from . import tp_diffusion as _td
        _mark("struct_stage_done")                                               # expansion + refiner on the structural row shard done; the diffusion conditioning of the rows starts
        n_struct = int(z.lay.N) if isinstance(z, RowShard) else int(z.shape[-2])
        lay = _td.struct_layout(n_struct)
        _td.shard_extra_bias(input_feature_dict, lay)                            # the structural extra attention bias [N_st, N_st] -> this rank's rows BEFORE the blocks bind it (nothing after the roll-out reads it whole)
        ctx = _td.prime(self, input_feature_dict, z, lay=lay, n_sample=_n_sample_per_call(self))
        _mark("diffusion_primed")
        cache = _td.prepare_cache_sharded(self, input_feature_dict, ctx)
        if isinstance(z, RowShard) and os.environ.get(ENV_KEEP_ZSTRUCT, "").strip() != "1":   # the structural shard is consumed — z_cond rows + the atom encoder's band term are
            ctx.zsrc = None                                                       # built; nothing after this point reads structural z (the heads run on the RESIDUE shard). Its device
            gb = z.release_device("consumed by the diffusion conditioning rows")  # rows (768 B x R_st x N_st, bf16) leave the roll-out, distogram and confidence stages; a later read
            STATS["zstruct_released_gb"] = round(STATS.get("zstruct_released_gb", 0.0) + gb, 3)   # is refused by name. ODDE_TP_KEEP_ZSTRUCT=1 keeps the shard resident (diagnostic).
            STATS["zstruct_release"] = "after_prime"
        else:
            STATS["zstruct_release"] = "kept" if isinstance(z, RowShard) else "n/a"
        torch.cuda.empty_cache() if torch.cuda.is_available() else None          # the priming transients' cached segments returned before the roll-out (reserved tracks allocated)
        _mark("diffusion_resident")                                               # the roll-out's resident floor: W + z_cond rows (+ the atom cache); the bias cache fills inside the roll-out
        return cache
    prepare_diffusion_cache_for_sampling._rowpair_tp = True
    return prepare_diffusion_cache_for_sampling


def _make_sample_rows(inner):
    """``OpenDDE.run_sample_diffusion_stage`` under ``n_gpu > 1``: the stock sampler loop with the denoiser's DiffusionTransformer on LOCAL query
    rows (``tp_diffusion.run_sample_diffusion_stage``; coordinates proven replicated), then the roll-out's device residents released."""
    def run_sample_diffusion_stage(self, **kw):
        from . import tp_diffusion as _td
        cache = kw.get("cache") or {}
        if not isinstance(cache.get("pair_z"), _td.StructDiffusion):
            return inner(self, **kw)
        from opt_core.mem.rowpair import RowpairRefused
        for key in ("atom_window_spec", "diffusion_attn_bias"):                    # the pin's stage hands these to the sampler (opendde.py:1244-1264); this line's local-row
            if cache.get(key) is not None:                                          # denoiser does not consume them: a non-None one is refused by name, never swallowed
                raise RowpairRefused(f"{LEVER}: refused: run_sample_diffusion_stage with cache[{key!r}] set (a Fold-CP / atom-window sampling shape this line does not shard)")
        if getattr(self, "_maybe_foldcp_mesh", None) is not None and self._maybe_foldcp_mesh() is not None:
            raise RowpairRefused(f"{LEVER}: refused: run_sample_diffusion_stage under an upstream Fold-CP mesh (foldcp_group): the row-sharded line and upstream's 1 x P are exclusive")
        sd_cfg = getattr(getattr(self, "configs", None), "sample_diffusion", None)      # training-free guidance (sample_diffusion.guidance.enable: generator.sample_diffusion's
        g_cfg = sd_cfg.get("guidance") if (sd_cfg is not None and hasattr(sd_cfg, "get")) else None   # tfg.step drives the denoiser through its own inner loops) is not this line's roll-out:
        if g_cfg is not None and bool(g_cfg.get("enable", False) if hasattr(g_cfg, "get") else getattr(g_cfg, "enable", False)):   # refused by name, never run silently sharded
            raise RowpairRefused(f"{LEVER}: refused: sample_diffusion.guidance.enable (training-free guidance, tfg.step) under the row-sharded roll-out")
        ctx = cache["pair_z"]
        try:
            out = _td.run_sample_diffusion_stage(self, inner, ctx, **kw)
        finally:
            _td.release(ctx)
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None      # the roll-out's residents and bias cache returned before the heads
        _mark("diffusion_peak")
        return out
    run_sample_diffusion_stage._rowpair_tp = True
    return run_sample_diffusion_stage


def _diffusion_fields() -> list:
    from . import tp_diffusion as _td
    from . import tp_struct as _ts
    return list(_td.fields()) + list(_ts.fields())

def _ensure_model_wraps() -> None:
    """The model-level consumers of the :class:`RowShard` are this adapter's wrappers around WHATEVER is bound now (the offload unit re-binds
    ``expand_to_structural_tokens`` when the runner is built, after the import-time patch). Checked at every trunk exit; counted."""
    m = sys.modules.get(MODEL_TARGET)
    cls = getattr(m, "OpenDDE", None) if m is not None else None
    if cls is None:
        return
    for name, maker in (("expand_to_structural_tokens", _make_expand_rows), ("compute_distogram_contact_probs", _make_distogram_rows),
                        ("run_post_confidence_outputs_stage", _make_post_conf), ("prepare_diffusion_cache_for_sampling", _make_diffusion_cache_rows),
                        ("run_sample_diffusion_stage", _make_sample_rows)):
        f = getattr(cls, name, None)
        if f is not None and not getattr(f, "_rowpair_tp", False):
            setattr(cls, name, maker(f))
            STATS["model_rewraps"] = STATS.get("model_rewraps", 0) + 1



def upstream_stage_cast(t):
    """The rows of a :class:`RowShard` inside one of upstream's autocast-disabled stages (``autocasting_disable_decorator(True)``: autocast off and
    every floating tensor ARGUMENT cast to fp32 — ``opendde/utils/torch_utils.py:166-186``). The shard is not a tensor argument, so the decorator
    leaves it alone; this applies the same cast to its rows where the stage reads them: fp32 when autocast is disabled in the calling frame (the
    bf16 base's rows are bf16 there), unchanged otherwise — on fp32 rows a no-op."""
    import torch
    if torch.is_tensor(t) and torch.is_floating_point(t) and t.dtype != torch.float32 and not torch.is_autocast_enabled():
        return t.to(dtype=torch.float32)
    return t


def _make_distogram_rows(orig):
    """``OpenDDE.compute_distogram_contact_probs`` on a :class:`RowShard`: distogram logits of this rank's rows (``DistogramHead``: ``linear(z)``
    plus its transpose — the transposed rows come through the shard transpose of the small logits), contact probabilities row-local
    (``sample_confidence.compute_contact_prob``). Returns this rank's rows ``[R, N]`` (kept on the shard for the confidence reducer); the
    ``[N, N]`` matrix is never assembled on a device."""
    def compute_distogram_contact_probs(self, pair_z, pair_z_spec=None):
        if not isinstance(pair_z, RowShard):
            return orig(self, pair_z, pair_z_spec=pair_z_spec)
        from opendde.model import sample_confidence as SC
        from opt_core.mem.rowpair import dist as _dist, shard as _shard
        lay, z_sh = pair_z.lay, upstream_stage_cast(pair_z.z)                             # the stage runs autocast-disabled with fp32 arguments (opendde.py:1490): the rows likewise
        l_rows = self.distogram_head.linear(z_sh).contiguous()                           # [R, N, bins]
        l_rows = l_rows + _dist.transpose_shards(l_rows, lay)                               # + logits^T rows (head.py: logits + logits.transpose(-2, -3))
        c_rows = SC.compute_contact_prob(distogram_logits=l_rows, **SC.get_bin_params(self.configs.confidence.distogram))   # [R, N] row-local
        del l_rows
        STATS["distogram_rows"] += 1
        _mark("distogram")
        pair_z.contact_rows = c_rows.contiguous()                                            # [R, N]: this rank's rows only (the confidence reducer consumes them; the [N, N] matrix the
        return pair_z.contact_rows                                                           # engine writes is collected on rank 0's HOST by the reducer — never whole on a device)
    compute_distogram_contact_probs._rowpair_tp = True
    return compute_distogram_contact_probs


def _make_conf_forward(orig):
    """``ConfidenceHead.forward`` (confidence.py:204-384 with memory_efficient_forward :490-639) on a :class:`RowShard` ``z_trunk``: the pair
    init on rows (``s1`` term along j, ``s2`` term along this rank's i rows), per sample a clone of the rows, the representative-atom distance
    one-hot of the rows, the head's pairformer stack on the shard (recognised as pre-sharded: no gather; ``s`` all-gathered per block), PAE
    logits of the rows, PDE logits of the rows of ``z + zᵀ`` (the transposed rows through the shard transpose), pLDDT / resolved from the
    replicated ``s``. The per-sample PAE / PDE logits are computed in row blocks (``ROWPAIR_CONF_ROWS``) and CONSUMED on this rank by the core's
    ``confidence.RowBlockReducer`` (bin expectations, TM-weighted per-context row statistics, contact-weighted PDE sums); nothing ``[N, N, bins]``
    exists anywhere. The reducers are finished by the wrapped ``OpenDDE.run_post_confidence_outputs_stage`` (:func:`_make_post_conf`), which
    produces the stock ``summary_confidence`` / ``full_data`` on rank 0 (``ROWPAIR_CONF_FINISH=exact``: the stock formulas on gathered per-context
    O(N²)-scalar matrices, bitwise-class; ``rowsum``: O(N) row sums, reordered) and broadcasts the summaries to every rank."""
    def forward(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, z_trunk_spec=None, triangle_multiplicative="torch",
                triangle_attention="torch", inplace_safe=False, chunk_size=None, compute_plddt=True, compute_pae=True, compute_pde=True,
                compute_resolved=True):
        if not isinstance(z_trunk, RowShard):
            return orig(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, z_trunk_spec=z_trunk_spec,
                        triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe,
                        chunk_size=chunk_size, compute_plddt=compute_plddt, compute_pae=compute_pae, compute_pde=compute_pde, compute_resolved=compute_resolved)
        import torch
        from opendde.model.utils import broadcast_token_to_atom, one_hot
        from opt_core.mem.rowpair import dist as _dist, shard as _shard
        lay, z_src = z_trunk.lay, z_trunk.z                                                # the residue shard as the trunk left it (bf16 on the bf16 base); the fp32 rows the head runs on are
                                                                                            # made ONE sample at a time below (no resident fp32 cast / z_rows / clone triple)
        r0, r1 = lay.r0, lay.r0 + lay.R
        s_inputs = s_inputs.detach()
        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk.detach(), min=-512, max=512))
        if s_trunk.dim() != 2 or int(s_trunk.shape[0]) != lay.N:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: confidence head s_trunk {tuple(s_trunk.shape)} vs N={lay.N} (a batch dim is not sharded)")
        x_rep_atom_mask = self._select_distogram_rep_atom_mask(input_feature_dict=input_feature_dict, n_token=s_trunk.shape[-2])
        x_rep = x_pred_coords[..., x_rep_atom_mask, :]
        N_sample = x_rep.size(-3)
        p1, p2 = self.linear_no_bias_s1(s_inputs), self.linear_no_bias_s2(s_inputs)[r0:r1]  # the two single projections the stock adds to z ([N, c], [R, c]); their [R, N, c] broadcast sum is
                                                                                            # formed per row block inside _sample_pair_rows (same addends, same order per element: bitwise the stock's)

        def _sample_pair_rows():
            """This sample's confidence pair rows, fp32 under the head's autocast-disabled frame (upstream_stage_cast: opendde.py:1505): ONE new
            [R, N, c] tensor = cast(z rows) + (s1[j] + s2[i]) — the dense statement ``z_sh + (p1[None] + p2[:, None])`` element for
            element (the addend is formed first, then added), without the resident fp32 cast, the resident sum and the per-sample clone."""
            zc_ = upstream_stage_cast(z_src)
            if zc_ is z_src:                                                                # fp32 base: the cast is a no-op — a private copy (the pair stack updates its rows in place)
                zc_ = z_src.detach().clone()
            zc_ = zc_.detach()
            for i0 in range(0, int(zc_.shape[0]), step):
                i1 = min(int(zc_.shape[0]), i0 + step)
                zc_[i0:i1] += (p1[None, :, :] + p2[i0:i1, None, :])
            return zc_
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]
        extra = input_feature_dict.get("structural_pair_attn_bias", None)
        plddt_preds, resolved_preds, reducers = [], [], []
        from opt_core.mem.rowpair import confidence as _conf
        finish, step = _conf_finish(), _conf_rows(lay)
        contact_rows = getattr(z_trunk, "contact_rows", None)                       # this rank's rows [R, N] of the contact probabilities (the distogram stage ran on the rows)
        if (compute_pae or compute_pde) and contact_rows is None:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: the confidence reducer needs the distogram stage's contact probability rows (RowShard.contact_rows unset: "
                                 f"compute_distogram_contact_probs did not run on this shard)")
        cfg = _TRUNK.get("configs")
        if cfg is None:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"{LEVER}: refused: the model configs were not recorded at the trunk exit (bins of the PAE / PDE heads)")
        from opendde.model import sample_confidence as _sc
        bins = {k: tuple(_sc.get_bin_params(getattr(cfg.confidence, k))[n] for n in ("min_bin", "max_bin", "no_bins")) for k in ("pae", "pde")}
        is_lig_atom = input_feature_dict["is_ligand"].long()
        tok_is_lig = torch.zeros_like(input_feature_dict["asym_id"], dtype=torch.long).scatter_add(0, atom_to_token_idx.long(), is_lig_atom) > 0   # sample_confidence.py:193-197
        chains = _conf.ChainIndex(input_feature_dict["asym_id"], input_feature_dict["has_frame"], tok_is_lig)
        _TRUNK["lay"] = lay
        try:
            for i in range(N_sample):
                zc = _sample_pair_rows()
                sc = s_trunk.clone() if inplace_safe else s_trunk
                with torch.amp.autocast("cuda", enabled=False):
                    xi = x_rep[..., i, :, :].to(torch.float32)
                    d_rows = torch.cdist(xi[..., r0:r1, :], xi)                            # [R, N] representative-atom distances of this rank's rows
                zc += self.linear_no_bias_d(one_hot(x=d_rows, lower_bins=self.lower_bins, upper_bins=self.upper_bins).to(dtype=self.linear_no_bias_d.weight.dtype))
                zc += self.linear_no_bias_d_wo_onehot(d_rows.unsqueeze(dim=-1).to(dtype=self.linear_no_bias_d_wo_onehot.weight.dtype))
                del d_rows
                sc, zc = self.pairformer_stack(sc, zc, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                               inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra)      # pre-sharded: no gather of z
                if compute_pae or compute_pde:
                    zc = zc.to(torch.float32)
                if compute_plddt or compute_resolved:
                    sc = sc.to(torch.float32)
                with torch.amp.autocast("cuda", enabled=False):
                    if compute_pae or compute_pde:                                          # the [R, N, bins] logits never leave this rank: consumed in row blocks by the
                        red = _conf.RowBlockReducer(chains, r0, r1, zc.device, pae_bins=bins["pae"], pde_bins=bins["pde"], finish=finish)   # core's exact-finish reducer
                        zc_c = zc.contiguous()
                        for i0, i1, zt_blk in _dist.transpose_blocks(zc_c, lay, step=step):   # rows of zᵀ block-streamed (one all-to-all per block; no transposed shard)
                            blk = zc_c[i0:i1]
                            pae_c = self.linear_no_bias_pae(self.pae_ln(blk))
                            pde_c = self.linear_no_bias_pde(self.pde_ln((blk + zt_blk).contiguous()))
                            red.consume(r0 + i0, r0 + i1, pae_c, pde_c, contact_rows[i0:i1].to(device=zc.device, dtype=torch.float32))
                            del blk, zt_blk, pae_c, pde_c
                        reducers.append(red)
                        del zc_c
                    if compute_plddt or compute_resolved:
                        a = broadcast_token_to_atom(x_token=sc, atom_to_token_idx=atom_to_token_idx)
                        if compute_plddt:
                            plddt_preds.append(torch.einsum("...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx]))
                        if compute_resolved:
                            resolved_preds.append(torch.einsum("...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx]))
                        del a
                del zc, sc
                STATS["conf_samples_rows"] += 1
        finally:
            _TRUNK["lay"] = None
        STATS["conf_rows"] += 1
        STATS["conf_logits"] = f"reducer:{finish}"
        _mark("confidence")
        if reducers:                                                                # finished by the post-confidence stage (OpenDDE.run_post_confidence_outputs_stage, wrapped)
            _PENDING["conf"] = {"reducers": reducers, "lay": lay, "finish": finish, "n": lay.N}
        return (torch.stack(plddt_preds, dim=-3) if compute_plddt else None, None, None,   # PAE / PDE logits: None (never materialised [N, N, bins]; pred_dict keeps no 'pae' / 'pde')
                torch.stack(resolved_preds, dim=-3) if compute_resolved else None)
    forward._rowpair_tp = True
    return forward


_PENDING = {}                                            # the confidence head's per-sample reducers awaiting the post-confidence stage of the same prediction
ENV_CONF_FINISH = "ROWPAIR_CONF_FINISH"                  # exact (default) | rowsum — the core reducer's finish
ENV_CONF_ROWS = "ROWPAIR_CONF_ROWS"                      # rows per PAE / PDE logits block in the head (fixed; default 128)
ENV_RANK_THREADS = "ROWPAIR_RANK_THREADS"                # the core launcher's per-rank CPU-thread cap (auto = cores / ranks; the line exports auto), applied once per rank
                                                         # process by _apply_rank_threads; the core records rank_threads=<n> in its own census words
ENV_KEEP_ZSTRUCT = "ODDE_TP_KEEP_ZSTRUCT"                # =1: keep the structural pair shard resident after the diffusion conditioning consumed it (diagnostic)
CONF_ROWS_DEFAULT = 128


def _conf_finish() -> str:
    v = os.environ.get(ENV_CONF_FINISH) or "exact"
    if v not in ("exact", "rowsum"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_CONF_FINISH}={v!r} (exact | rowsum)")
    return v


def _conf_rows(lay) -> int:
    v = int(os.environ.get(ENV_CONF_ROWS) or CONF_ROWS_DEFAULT)
    if v < 1:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: {ENV_CONF_ROWS}={v} (must be >= 1)")
    return min(v, int(lay.Rmax))


def _make_post_conf(inner):
    """``OpenDDE.run_post_confidence_outputs_stage`` (opendde.py:1303-1360) when the confidence head left row-block reducers (:data:`_PENDING`):
    the reducers are finished collectively (every rank), rank 0 assembles the stock ``summary_confidence`` / ``full_data`` lists from the reduced
    statistics with the stock's own finishing statements (:func:`_summary_rank0` == ``sample_confidence._compute_full_data_and_summary`` term by
    term), and the summaries are broadcast so every rank's writer holds them; ``full_data`` (the ``[N, N]`` expected-value matrices, rank 0 only,
    fp16-collected by the reducer) stays on rank 0 (ranks >= 1 hold ``[{}] * N_sample``, named). A prediction without pending reducers (the head
    ran the stock statement) runs the stock stage."""
    def run_post_confidence_outputs_stage(self, *, pred_dict, input_feature_dict, pair_input_feature_dict, N_cycle):   # the pin's signature (opendde.py:1590: no pair_z)
        pend = _PENDING.pop("conf", None)
        if pend is None:
            return inner(self, pred_dict=pred_dict, input_feature_dict=input_feature_dict, pair_input_feature_dict=pair_input_feature_dict, N_cycle=N_cycle)
        import torch
        from opt_core.mem.rowpair import dist as _dist
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.add_shape_complementarity_predictions(pred_dict=pred_dict, input_feature_dict=pair_input_feature_dict, coordinate=pred_dict["coordinate"], label_dict=None)
        feats = pair_input_feature_dict                                       # the residue-level feature dict (the head ran on the residue pair rows)
        dev = pred_dict["plddt"].device
        need_full = bool(getattr(self.configs, "need_atom_confidence", False))
        summaries, full = [], []
        for i, red in enumerate(pend["reducers"]):
            stats = red.finalize(pend["lay"].bounds, collect_full=need_full)   # collective
            if _dist.world()[1] == 0:
                s_i, f_i = _summary_rank0(self.configs, stats, red, plddt_logits=pred_dict["plddt"][i:i + 1], token_asym_id=feats["asym_id"].to(dev),
                                          token_has_frame=feats["has_frame"].to(dev), atom_coordinate=pred_dict["coordinate"][i:i + 1],
                                          atom_to_token_idx=feats["atom_to_token_idx"].to(dev), atom_is_polymer=1 - input_feature_dict["is_ligand"],
                                          N_recycle=N_cycle, return_full_data=need_full)
                summaries.extend(s_i)
                full.extend(f_i)
            del stats
        if _dist.world()[1] == 0:
            payload = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in s.items()} for s in summaries]
        else:
            payload = None
        payload = _dist.broadcast_obj(payload, src=0)                          # O(samples x chains²) scalars
        if _dist.world()[1] != 0:
            summaries = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()} for s in payload]
            full = [{} for _ in summaries]
        pred_dict["summary_confidence"], pred_dict["full_data"] = summaries, full
        pred_dict.pop("per_sample_contact_probs", None)
        pred_dict.pop("contact_probs", None)                                        # this rank's ROWS (not an engine output; the [N, N] matrix lives in full_data on rank 0 when requested)
        STATS["conf_summaries"] = STATS.get("conf_summaries", 0) + len(summaries)
        return pred_dict
    run_post_confidence_outputs_stage._rowpair_tp = True
    return run_post_confidence_outputs_stage


def _summary_rank0(configs, stats, red, *, plddt_logits, token_asym_id, token_has_frame, atom_coordinate, atom_to_token_idx, atom_is_polymer,
                   N_recycle, return_full_data):
    """== ``sample_confidence._compute_full_data_and_summary`` (sample_confidence.py:37-224) for ONE sample, with every PAE / PDE-derived term taken
    from the reducer's statistics (``ptm iptm chain_ptm chain_iptm chain_pair_iptm chain_pair_iptm_global``; ``gpde`` / ``chain_*gpde`` from the
    gathered ``[N, N]`` expected-PDE and contact matrices under ``exact``, from the reduced sums under ``rowsum``) and every atom-level term by the
    stock's own functions on the replicated inputs (pLDDT, chain pLDDT, clash, ranking score). ``full_data``: the stock keys, the ``[N, N]``
    matrices from the reducer's fp16 CPU collection."""
    import torch
    from opendde.model import sample_confidence as SC
    full_data, summary = {}, {}
    atom_plddt = SC.logits_to_score(plddt_logits, **SC.get_bin_params(configs.confidence.plddt))       # [1, N_atom]
    summary["plddt"] = atom_plddt.mean(dim=-1) * 100
    if red.finish == "exact":
        token_pair_pde, contact_probs = stats["token_pair_pde_f32"], stats["contact_probs_f32"]        # [1, N, N], [N, N] (rank-ordered rows == the stock tensors)
        summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])
    else:
        summary["gpde"] = stats["gpde"]
    summary["ptm"], summary["iptm"] = stats["ptm"], stats["iptm"]
    if red.finish == "exact":
        summary.update(SC.calculate_chain_based_gpde(token_pair_pde=token_pair_pde, contact_probs=contact_probs, asym_id=token_asym_id))
    else:
        summary.update({"chain_gpde": stats["chain_gpde"], "chain_pair_gpde": stats["chain_pair_gpde"]})
    summary.update({"chain_ptm": stats["chain_ptm"], "chain_iptm": stats["chain_iptm"], "chain_pair_iptm": stats["chain_pair_iptm"],
                    "chain_pair_iptm_global": stats["chain_pair_iptm_global"]})
    summary.update(SC.calculate_chain_based_plddt(atom_plddt, token_asym_id, atom_to_token_idx))
    summary["has_clash"] = SC.calculate_clash(atom_coordinate, token_asym_id, atom_to_token_idx, atom_is_polymer, configs.metrics.clash.af3_clash_threshold)
    summary["num_recycles"] = torch.tensor(N_recycle, device=atom_coordinate.device)
    summary["disorder"] = torch.zeros_like(summary["ptm"])
    summary["ranking_score"] = 0.8 * summary["iptm"] + 0.2 * summary["ptm"] + 0.5 * summary["disorder"] - 100 * summary["has_clash"]
    summaries = SC.break_down_to_per_sample_dict(summary, shared_keys=["num_recycles"])
    if not return_full_data:
        return summaries, [{}]
    full_data["atom_plddt"] = atom_plddt
    for key, sk in (("token_pair_pde", "token_pair_pde_f16"), ("token_pair_pae", "token_pair_pae_f16")):
        if sk in stats:
            full_data[key] = stats[sk].float().unsqueeze(0)                   # [1, N, N] (fp16-collected expected values; named precision)
    if "contact_probs_f16" in stats:
        full_data["contact_probs"] = stats["contact_probs_f16"].float()
    full_data["token_has_frame"] = token_has_frame.clone()
    full_data["token_asym_id"] = token_asym_id.clone()
    full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
    full_data["atom_is_polymer"] = atom_is_polymer.clone()
    full_data["atom_coordinate"] = atom_coordinate.clone()
    return summaries, SC.break_down_to_per_sample_dict(full_data, shared_keys=["contact_probs", "token_has_frame", "token_asym_id", "atom_to_token_idx", "atom_is_polymer"])


class _Unsharded(RuntimeError):
    """A call shape the adapter does not shard (named; the stock statement runs and the lever reads PARTIAL)."""


def _replicated(t, name: str) -> None:
    """``t`` (a replicated input of a sharded stack) is bitwise identical on every rank (opt_core.mem.rowpair.dist.allreduce_checksum:
    one all_gather of a checksum); a mismatch is the core's ``RowpairRefused`` naming the tensor — the run dies by name, never averages
    diverged ranks. Counted in ``STATS["guards"]``."""
    from opt_core.mem.rowpair import dist as _dist
    _dist.allreduce_checksum(t.detach(), name)
    STATS["guards"] = STATS.get("guards", 0) + 1


def _ensure_block_patch() -> None:
    """``PairformerBlock.forward_source`` is this adapter's wrapper: a kit lever that re-binds the class attribute after the import-time patch
    (ARM Z's transpose-free block statement, installed when the runner is built) is wrapped again, so the MSA module's pair blocks shard too
    (the displaced statement stays the ``n_gpu == 1`` path of the wrapper). Checked at every stack entry; counted (``block_rewraps``;
    ``direct_blocks`` counts the wrapper's sharded calls — a bind that is never reached shows as zero)."""
    m = sys.modules.get(TARGET)
    cls = getattr(m, "PairformerBlock", None) if m is not None else None
    f = getattr(cls, BLOCK_ATTR.split(".")[1], None) if cls is not None else None
    if f is not None and not getattr(f, "_rowpair_tp", False):
        setattr(cls, BLOCK_ATTR.split(".")[1], _make_block_forward(f))
        STATS["block_rewraps"] = STATS.get("block_rewraps", 0) + 1


def _note_unsharded(reason: str) -> None:
    """A pair-stack call shape the adapter does not shard (a leading batch > 1, a mask / single tensor that does not match N): REFUSED by name,
    unless ``ROWPAIR_ALLOW_UNSHARDED=1`` opts into running it unsharded on every rank (a named event + the ``unsharded`` census count +
    the exit-time fallback line; never silent)."""
    if os.environ.get(ENV_ALLOW_UNSHARDED) != "1":
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"{LEVER}: refused: a pair-stack call the adapter does not shard ({reason}); {ENV_ALLOW_UNSHARDED}=1 runs such calls unsharded (named)")
    STATS["unsharded"] += 1
    key = reason.split(":")[0][:60]
    STATS["unsharded_reasons"][key] = STATS["unsharded_reasons"].get(key, 0) + 1
    print(f"[{TAG}] ROWPAIR event=unsharded_call reason={reason!r}", file=sys.stderr, flush=True)


def _make_stack_forward(orig):
    """``PairformerStack.forward`` under ``P > 1``: the whole stack on row shards, one all-gather at its exit."""
    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None,
                extra_attn_bias=None):
        if not in_rank_process():
            return orig(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                        inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
        _ensure_block_patch()
        STATS["stacks"] += 1
        try:
            return _run_sharded(list(self.blocks), s, z, pair_mask, triangle_attention=triangle_attention, extra_attn_bias=extra_attn_bias, chunk_size=chunk_size)
        except _Unsharded as e:
            _note_unsharded(str(e))
            return orig(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                        inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
    forward._rowpair_tp = True
    return forward


def _make_block_forward(orig):
    """``PairformerBlock.forward_source`` reached outside this adapter's stack statement (the MSA module's pair block) under ``P > 1``: shard,
    one block, all-gather."""
    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None,
                extra_attn_bias=None):
        if not in_rank_process():
            return orig(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                        inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
        STATS["direct_blocks"] += 1
        try:
            return _run_sharded([self], s, z, pair_mask, triangle_attention=triangle_attention, extra_attn_bias=extra_attn_bias, chunk_size=chunk_size)
        except _Unsharded as e:
            _note_unsharded(str(e))
            return orig(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                        inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
    forward._rowpair_tp = True
    return forward


# ----------------------------------------------------------------------------------------------------------------- evidence
def kit_stats() -> dict:
    _sync()
    out = {k: v for k, v in STATS.items() if k not in ("patches", "contract_stats")}
    out["unsharded_reasons"] = dict(STATS["unsharded_reasons"])
    from . import tp_kernels as _tpk
    out["tp_kernels"] = _tpk.stats()                                            # the row-block kernel's census (tp_kernels: the TRIATT word + the core dispatch's counts)
    return out


def evidence_pairs() -> list:
    """The LEVER line's evidence, the core's ``n_gpu`` / ``sharding`` (+ layout facts of this rank) first."""
    from opt_core.mem.rowpair import evidence as _ev
    pairs = list(_ev.fields(STATS["n_gpu"]))
    lay = STATS.get("layout")
    if lay:
        pairs += [("N", lay["N"]), ("P", lay["P"]), ("B", lay["B"]), ("rank", lay["rank"]), ("rows", f"{lay['r0']}:{lay['r1']}"), ("R", lay["R"])]
    from . import tp_kernels as _tpk
    pairs += [("triatt_kernel", _tpk.TRIATT["kernel"] or "-"), ("trimul_kernels", _tpk.TRIMUL["kernels"] or "-")]   # the words bound (tp_kernels; counts on their own LEVER lines)
    pairs += [("stacks", STATS["stacks"]), ("direct_blocks", STATS["direct_blocks"]), ("blocks", STATS["blocks"]), ("gathers", STATS["gathers"]),
              ("block_rewraps", STATS.get("block_rewraps", 0)), ("guards", STATS.get("guards", 0)),
              ("trunk_rows", STATS["trunk_rows"]), ("trunk_exit", STATS["trunk_exit"]),
              ("presharded_calls", STATS["presharded_calls"]), ("msa_blocks_rows", STATS["msa_blocks_rows"]), ("template_rows", STATS["template_rows"]),
              ("template_feats_rows", STATS["template_feats_rows"]), ("conf_rows", STATS["conf_rows"]), ("conf_samples_rows", STATS["conf_samples_rows"]),
              ("conf_logits", STATS["conf_logits"]), ("distogram_rows", STATS["distogram_rows"]), ("zcond_rows", STATS["zcond_rows"]), ("dit_local_queries", STATS["dit_local_queries"]),
              ("feat_resident_gb", STATS["feat_resident_gb"]), ("template_full_gb", STATS["template_full_gb"]), ("template_rows_h2d_gb", STATS["template_rows_h2d_gb"]), ("template_rows_born", STATS["template_rows_born"]),
              ("inputs_born_gb", STATS["inputs_born_gb"]), ("inputs_full_gb", STATS["inputs_full_gb"]),
              (FEATS_EQUAL, STATS[FEATS_EQUAL]), (FEATS_DIGEST, STATS[FEATS_DIGEST]), ("feats_batches", STATS["feats_batches"]), ("feats_unequal", STATS["feats_unequal"]),
              ("feats_excluded", ";".join(PER_RANK_FEATS)),            # the feature-dict keys the digest leaves out (this rank's rows by design)
              ("data_form", DATA_FORM), ("feats_bcast_calls", STATS["feats_bcast_calls"]), ("feats_error_items", STATS["feats_error_items"]),
              ("feats_bcast_gib", STATS["feats_bcast_gib"]), ("feats_wait_s", STATS["feats_wait_s"]), ("feats_bcast_s", STATS["feats_bcast_s"]), ("feats_rng", STATS["feats_rng"]),
              ("host_peak_gib", host_peak_gib()),                          # this rank process's peak resident host memory (ranks > 0 carry no featurisation)
              *_diffusion_fields(),
              ("model_rewraps", STATS.get("model_rewraps", 0)), ("trunk_s", STATS.get("trunk_s", 0.0)), ("kit_schedule", dict(STATS["schedule"]) or None),
              ("core_schedule", dict(_ev.schedule_fields()) or None),                 # the core's record_schedule words of this rank's calls so far (trimul RA/RB/tiles, ring steps, opm/pwa rows, ...)
              ("replicated_by_design", ";".join(REPLICATED_BY_DESIGN)),
              ("unsharded", STATS["unsharded"]), ("stack_s", {k: v[1] for k, v in STATS["stack_s"].items()} or None),
              ("stack_calls", {k: v[0] for k, v in STATS["stack_s"].items()} or None)]
    return pairs


def fallbacks(planned) -> list:
    """Named events at exit: a row-block kernel fallback outside its declared reasons (tp_kernels); a pair stack that ran unsharded under ``P > 1``
    (a call shape the adapter does not shard)."""
    if LEVER not in planned or STATS["n_gpu"] <= 1:
        return []
    from . import tp_kernels as _tpk
    out = list(_tpk.undeclared())                                                # a row-block kernel fallback outside the declared reasons: named, the exit rule fails closed
    if STATS["unsharded"]:
        out.append(f"{LEVER}: {STATS['unsharded']} pair-stack call(s) ran unsharded ({','.join(sorted(STATS['unsharded_reasons']))})")
    return out


def notes(planned) -> list:
    if LEVER not in planned:
        return []
    if STATS["n_gpu"] <= 1:
        return [f"{LEVER}: n_gpu=1 — the adapter installs nothing; the engine's single-card statements run"]
    if not STATS["calls"]:
        return [f"{LEVER}: no pair stack was reached in this process"]
    return []