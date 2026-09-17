"""The TP ADAPTER of the memory mode: ``pred --mode big --n_gpu P`` with ``P > 1`` runs the kit's own ``pred`` as P rank processes (one per
GPU of the node, launched by :mod:`opt_core.mem.rowpair.launch`) and, inside every rank, binds the engine's statements to the release tree's
row-sharded pair family (:mod:`opt_core.mem.rowpair`, strategy ``F7.tensor_parallel``, token ``sharding=rowpair``). At ``--n_gpu 1`` this
module installs NOTHING (:func:`install` returns before touching the model, ``PATCHES.names() == []``): ``n_gpu=1`` is the single-GPU line
byte for byte.

What is sharded: rank ``r`` holds rows ``r0:r1`` of every pair-shaped tensor ``[B, N, N, C]`` (:class:`opt_core.mem.rowpair.dist.Layout`,
built by ``Layout.auto`` under ``require_sharded``; census ``rows=``), from the statement that creates it to the statement that consumes it —
no ``N x N x C`` tensor is whole on any device at any point of a fold (census ``mode_s=born_rows,no_gather,...``):

* born rows (:mod:`.rowpair_heads`): the pair-state init (rows of the stock CUDA Philox draw: ``opt_core.mem.rowpair.rng``), the relative-position
  features, the LM -> pair projection and its dropout rows, the pair mask rows;
* the Parcae loop (``ESMFold2Model._run_one_loop``) on rows: injection, ``lm_encoder`` / ``folding_trunk`` PairUpdateBlocks, ``parcae_readout``,
  ``parcae_coda``; the MSA module's pair reads/writes on rows (:mod:`.rowpair_msa`, variant ``full_msa``; the MSA representation ``m`` is
  replicated by design);
* inside a PairUpdateBlock on a shard: triangle multiplication outgoing / incoming = the stock statements issued per row block by the family
  driver (``opt_core.mem.rowpair.trimul.trimul_update_``: a from this rank's rows, b sub-blocks projected just in time and ring-passed, the
  output written per (row block, column block) with the residual added in place; incoming through banded transposes, z^T never whole);
  the pair transition runs per row block in place (``pairstack.transition_update_``); LayerNorms are row-local as they stand;
* heads (:mod:`.rowpair_heads`): the distogram on rows (``z + transpose_shard(z)``; the writer's processor takes rows), the diffusion module's
  pair conditioning on rows once per roll-out and its token transformer with LOCAL query rows (``opt_core.mem.rowpair.diffusion``), the
  confidence head on rows (pTM / ipTM from row-local reductions, PAE assembled in rank 0's HOST memory by ``dist.gather_rows_to_rank0_host``);
* atom attention runs banded (:mod:`.atom_swa`: no ``[N_atom, N_atom]`` mask).

REPLICATED BY DESIGN (whole on every rank, named ``replicated=`` in the install line): the token inputs / single representation, the ESM-C
hidden states, the MSA representation, atoms and the diffusion sample, the structure ``x_pred`` (broadcast from rank 0 before the
confidence head reads it). The single-GPU line's levers whose statements this line re-issues on rows are named, not counted
(``XL_LEVERS_REPLACED`` -> the xl gate's ``replaced_by_rowpair``; ``KIT_LEVERS_REPLACED``: the MK sampler hoist -> ``levers_replaced``).

Refused BY NAME at install (``opt_core.mem.rowpair.RowpairRefused``): no process group / world != P; a per-rank weight checksum mismatch
(``esmfold2.weights``); the ``cuequivariance`` backend; a training-mode model. An input BELOW THE SHARDING FLOOR (``Layout.auto``:
no row grid of N over P ranks without an empty rank — tiny inputs only) is not refused: it is folded WHOLE with the single-GPU statements
on every rank (every callable this lever rebound takes its kept original for that fold; no collective is issued; rank 0's fold is the one
written), named once per fold (``… input below the sharding floor (N=…) — folded unsharded …``; census ``sharding=none:below_floor``).

NUMERICS (why ``n_gpu>1`` is big-only, tier 2, held as a band): the triangle-multiplication tiles and the row-blocked GEMMs differ from the
single-GPU kernels in extent and summation order (per-element statements are the engine's; nothing is claimed bitwise); the confidence
head's per-chain ipTM sums are reduced across ranks in rank order. RANDOMNESS: every rank runs the kit's own ``pred`` with the SAME explicit
seed; pair-shaped random draws are rows of the stock full-tensor Philox stream and leave the default generator where the stock call leaves
it, so the replicated diffusion sampler draws the same noise on every rank; the production guard ``rng_state@structure_module`` refuses by
name unless the CPU and CUDA default-generator states are identical on every rank right before the structure module draws its noise; the
det-scoped guard ``features@S1`` (``--det >= 1``) refuses unless the loop's inputs are bitwise identical across ranks (outside the recipe the
comparison is recorded as ``features_replicated=``). THE MODEL'S INPUT across ranks (:mod:`.rowpair_feats`): the P rank processes start
with one ``PYTHONHASHSEED`` (the launcher's ``[esmfold2-opt] RANKENV hashseed=<v> source=default|inherited [parent='<value>'] ranks=P`` line); rank 0 featurises
each item and every other rank receives its tensors over the group (the builder's ``prepare_input`` rebound at install; install line
``data_form=rank0_bcast``; per call ``rowpair feats: data_form=rank0_bcast … digest=<hex16> feats_ranks_equal=yes …``; a rank whose digest
differs from rank 0's refuses on every rank, ``refused: feats_ranks_differ``; rank 0's featurisation error reaches every rank,
``refused: feats_rank0_failed``).
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import sys
import time
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from opt_core.mem import ngpu                                          # the ONE producer of the n_gpu words (check_n_gpu, tokens)
from opt_core.mem.patchset import PatchSet                            # the family's install record (framework-free)
from opt_core.mem.rowpair import RowpairRefused, refuse_unless_visible   # every refusal of this module is BY NAME through the family's class
from opt_core.mem.rowpair import evidence as EV
from opt_core.mem.rowpair import launch as RL

from . import report as _report
from . import rowpair_feats as RF
from .stack import MODEL_MODULE

LEVER = "rowpair"                                                      # this adapter's lever name in the kit (the strategy id is opt_core.mem.rowpair.LEVER = F7.tensor_parallel)
COMMON_MODULE = MODEL_MODULE + "_common"                               # transformers.models.esmfold2.modeling_esmfold2_common: PairUpdateBlock / FoldingTrunk / TriangleMultiplicativeBlock
W4_MODULE = "ef2_w4"                                                   # the kit driver's W4 lever module (lever tx: the pair TriMul entry point the row form re-binds; ef2_w4.tx_state)
XL_MODULE = "ef2_xl"                                                   # the XL memory add-on (its x2 OWN handoff decides how the shard is passed to the trunks)
SHARDED_TRUNKS = ("folding_trunk", "lm_encoder", "parcae_coda", "confidence_head.folding_trunk")   # the FoldingTrunks whose PairUpdateBlocks run on rows
DIFF_RANK_SPREAD_MAX_A = 1.0                                          # sampler exit: max|x_rank - x_rank0| above this refuses by name (expected ~1e-3 A non-det, 0 det)
KIT_LEVERS_REPLACED = {"mk": "the row-sharded diffusion forward (rowpair_heads.dm_forward_rows: pair conditioning rows + local-query token transformer)"}
#   the kit levers whose instance forward this line replaces under n_gpu > 1 (cli settles them as levers_replaced + gated, never partial)
NOT_FOR_ROUTE: Dict[str, str] = {                                     # registry lever names of the single-GPU sets that are NOT ON THE n_gpu > 1 ROUTE, each with its
    #   mechanistic reason token. The package leaves them out of a rank process's configure() (modes: `levers_off` += tp.not_for_route(P)) and
    #   names each on its LEVER line as `state=not_for_route:n_gpu>1 reason=<token>` — declared, never silent; the all-or-refuse verdict counts
    #   the route's set. Every other lever of the set engages under sharding as installed (the atom path, fz, t15 / t15msa, m15, kd, the W4 set,
    #   the caches) or runs its eager form by the kit's own graph budget (EAGER_UNDER_SHARDING).
    "ls": "recycle_loop_resharded:rowpair._run_one_loop_sharded_issues_the_recycle_on_row_shards(ef2_opt's_static_loop_I/O_stages_the_graphed_trunk's_inputs;the_sharded_trunks_run_eager)",
    "rg": "recycle_graph_not_capturable_on_shards:the_sharded_recycle_issues_NCCL_ring/all_to_all_traffic_with_host-decided_block_plans_and_replicated-state_guards(host_reads);its_pool_would_pin_the_shard_transients",
    #   `dit` (ef2_dit's fused step) IS installed at n_gpu > 1: it is the engine of the sampler's WHOLE route (rowpair_heads.dm_forward_whole,
    #   fed the conditioned pair gathered from the rows when that fits under the trunk's peak); the rows route (dm_forward_rows) never calls it.
    "m16": "pwa_on_rows:opt_core.mem.rowpair.msa.pwa_rows(bias+softmax_complete_per_local_query_row_block,out_rows_all_gathered);m16's_bias/softmax_kernel_takes_one_L(square_[H,L,L]_logits)",
    "m17": "opm_on_rows:opt_core.mem.rowpair.msa.opm_rows_budgeted(a[rows]_x_b[all]_per_row_block,in_place);m17's_projection+residual_epilogue_takes_one_L(square_pair)",
    "trimul": "msa_trimul_on_rows:opt_core.mem.rowpair.trimul.trimul_update_(per-row-block_projections+ring_contraction_from_the_module's_weights);the_engine_forward_is_not_called_on_a_shard",
    "glue": "msa_trimul_on_rows:rides_on_trimul's_engine_forward(not_called_on_a_shard)",
    "mh": "msa_encoder_on_rows:the_sharded_loop_calls_rowpair_msa.msa_encoder_rows(the_hoist_wraps_the_whole-pair_encoder_forward)",
    "disto": "distogram_rows:x10_on_rows_owns_the_distogram_move",
}
EAGER_UNDER_SHARDING: Dict[str, str] = {                              # levers that stay INSTALLED and ENGAGED under n_gpu > 1 without their CUDA graph: the graph would capture the
    #   row-sharded statements' collectives, host-decided block plans and replicated-state guards (host reads), and a private pool would pin the
    #   shard-sized transients (a pool growing with the shard). Same kernels, eager — named `graphs_off=` on the rowpair LEVER line.
    "tg": "folding_trunk,lm_encoder,parcae_coda,confidence_head.folding_trunk: eager forwards on row shards",
    "eg": "lm_encoder / msa_encoder graphs: the encoders run on rows eager",
    "sg": "structure_head.sample: the sampler-site graph budget is 1 token on a rank process (ef2_opt's named `graph budget -> eager` event)",
    "ro": "ef2_dit's roll-out runs its EAGER boundaries (pre-drawn RNG in upstream's order, device step tables, constants staged once, Kabsch as configured) "
          "around dm_forward_rows: the boundary graph is off by the same sampler-site budget",
}
SAMPLER_SITE_BUDGET_SHARDED = 1                                       # ef2_opt.CFG.graph_budget_tokens_sampler on a rank process: every sampler shape is `over budget` -> the
#   graphed sampler (sg) and ef2_dit's roll-out (ro) take their eager forms by the kit's one capture gate, printed once per shape by ef2_opt


FOR_ROUTE: Tuple[str, ...] = ("injrows", "confrows", "confbf16", "pdeskip", "confmem", "biasfree", "zbf16")
#   the registry levers this route ADDS to the set at P > 1, in install order (esmfold2_opt.rowchunk: statements the sharded pair stack issues on
#   whole row shards, reissued per row block of the rank's shard — the recycle inject, the confidence head's prologue / heads / TM reduction, the
#   pair state the confidence head reads, the sampler's pair-bias buffers' lifetime). modes.route_add puts them in `levers` (the ACTIVE / DRY-RUN
#   line names them; the ablation variable subtracts any one by name); install_rank installs the resolved subset after install() and
#   stack.settle_route judges them from the installer's record (a member that did not bind: partial, refused by name).


def for_route(P: int) -> Tuple[str, ...]:
    """The registry levers this route adds to the set at ``P`` > 1 (``FOR_ROUTE``, install order); empty at ``P == 1``."""
    return tuple(FOR_ROUTE) if int(P) > 1 else ()


def not_for_route(P: int) -> Dict[str, str]:
    """{registry lever name: reason token} the n_gpu = ``P`` route leaves out of a rank process's set (empty at P == 1: the single-GPU line is
    untouched). The package subtracts these from configure()'s set and names each on its LEVER line (``state=not_for_route:n_gpu>1``)."""
    return dict(NOT_FOR_ROUTE) if int(P) > 1 else {}
XL_LEVERS_REPLACED = ("x3", "x6", "x7", "x8", "x2b")             # the single-GPU line's XL levers whose statements this line re-issues on ROWS (diffusion pair
#   conditioning, LM->pair, relative position, pair-state init, the whole-pair frees): their wrappers never run under n_gpu > 1; cli's xl gate names them
#   (replaced_by_rowpair), not counts them. x2 (loop), x10 (distogram to host, on rows) and x4 (ESMC-6B host offload after the LM pass) compose and stay gated.
REPLICATED = ("x_inputs", "lm_activations", "pair_mask_nxn", "token_bonds_nxn", "msa_m", "atoms", "diffusion_a", "x_pred")   # named: whole on every rank BY DESIGN (rowpair_heads docstring); the atom attention builds no [N_atom, N_atom] mask on any rank (atom_swa: banded on the class and on every U1 instance forward)
D_PAIR = 256                                                           # the row-sharded TriMul launch is written for C == CH == 256 (the model's d_pair)
RANK_ENV = {"ROWPAIR_TRIMUL_GRID": "rank",                        # the TriMul ring's b sub-block grid: uniform RB sub-blocks from each rank start (the core's default `stock` also splits at
            #   the in-place statement's 256-row column chunks and pads every wire unit to RB: more ring bytes and twice the tiles for the same result)
            "ROWPAIR_RANK_THREADS": "auto",                          # opt_core >= 0.5.213: each rank caps its CPU threads to cores/ranks at entry (no OpenMP oversubscription in per-rank CPU statements; older cores ignore the word)
            "ROWPAIR_DIFF_BIAS": "ln_proj_fp32",                     # the rows route's pair-bias producer (opt_core >= 0.5.218.1 serves c_z 256): LayerNorm + projection + head-major write in one
            #                                                        pass, fp32 IEEE dot = the statement's values (fp32-exact), faster, no [rows, N, 16] transient; ln_proj = bf16 MMA
            #                                                        (bf16-class, faster still); engine = the statement. Older cores refuse c 256 by name -> the statement per block.
            "ROWPAIR_DIFF_BIAS_CACHE_GB": "auto",                    # the rows route's sampler budgets (pair-bias row cache, attention work) from the free bytes AGREED across ranks, not the
            "ROWPAIR_DIFF_WORK_GB": "auto",                          # core's fixed 24 GB / 8 GB defaults: near the reach edge a fixed cap allocates into memory that is not there
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",                 # the rank processes' settings (a user's own value wins). AVOID_RECORD_STREAMS: collective outputs are
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # held by reference instead pinned_stream, so freed transfer buffers are reusable at once;
                                                                      # expandable_segments: the ring / all-to-all / row-block transients of varying size map into growable
                                                                      # segments instead of pinning one cached block per distinct size (reserved tracks allocated; `alloc_conf=` census)
RUN_TIMEOUT_S = None                                                             # the launcher's bound on the whole n_gpu run (None: unbounded; the family's finish_grace / NCCL timeouts apply)
NCCL_TIMEOUT_S = 1800.0                                               # the process group's collective timeout (seconds)

BELOW_FLOOR_WORD = "none:below_floor"                                    # the census token of a fold below the sharding floor (folded whole, unsharded)
PATCHES = PatchSet(LEVER)                                              # what install() rebound in THIS process (names() == [] at n_gpu=1)
STATE: Dict[str, object] = {"P": 1, "rank": 0, "installed": False, "backend": None, "refused": None, "layouts": {}, "folds": [],
                            "install_line": None, "x_pred_mismatch": 0, "group": False}


# ================================================================================================================ the axis' process roles
def is_rank_process() -> bool:
    """True inside a rank process of an n_gpu > 1 run (the namespaced rank environment of :mod:`opt_core.mem.rowpair.launch` is set)."""
    return RL.world_size() > 1


def is_output_rank() -> bool:
    """Rank 0 (or a single-GPU process) writes the run's files; ranks > 0 compute every fold and write nothing."""
    return RL.is_output_rank()


def launch(argv: Sequence[str], mode: str, P: int, *, stream=None) -> int:
    """The LAUNCHER side of ``pred --n_gpu P`` (P > 1; called by cli.cmd_pred in a process that is not itself a rank): P rank processes of
    this same command line, rank r on visible GPU r, rank 0's transcript pumped to ``stream`` (stderr); returns the run's exit code — 0 when
    every rank exited 0, else the failing rank's own code when it is one of the kit's (usage 2 / not-active 3), else 1 — after the family's
    ``RANK-FAILED`` line. Loads no model and imports no torch here."""
    stream = stream or sys.stderr
    P = ngpu.refuse_unless_big(P, mode)
    if P == 1:
        raise RowpairRefused("launch: n_gpu=1 has no launcher (the single-GPU line runs in-process)")
    refuse_unless_visible(P)                                          # fewer than P visible GPUs: refused by name here, before any process starts (never shrunk)
    cmd = [sys.executable, "-m", "esmfold2_opt", "pred", *list(argv)]
    print(f"{_report.PREFIX} rowpair launch: {EV.fields_text(P)} ranks={P} isolate_devices=1 cmd=python -m esmfold2_opt pred ... "
          f"(rank r binds visible GPU r; rank 0 writes; transcripts <log_dir>/rank<r>.log)", file=stream, flush=True)
    t0 = time.time()
    try:
        recs = RL.run_rank_processes(P, cmd, mode=mode, isolate_devices=True, run_timeout_s=RUN_TIMEOUT_S, nccl_timeout_s=NCCL_TIMEOUT_S,
                                     env={**os.environ, **{k: v for k, v in RANK_ENV.items() if k not in os.environ}, RL.ENV_TAG: _report.TAG},   # the launch's RANKENV line carries the kit's tag
                                     on_line=lambda s: (stream.write(s if s.endswith("\n") else s + "\n"), stream.flush()),
                                     what="esmfold2 rowpair rank")
    except RL.RankFailed as exc:
        print(EV.rank_failed_line(_report.TAG, exc), file=stream, flush=True)
        rc = exc.exitcode if exc.exitcode in (_report.EXIT_USAGE, _report.EXIT_NOT_ACTIVE) else _report.EXIT_FAIL
        print(f"{_report.PREFIX} rowpair launch: FAILED event={exc.event} rank={exc.rank} exitcode={exc.exitcode} -> exit {rc} "
              f"(log {exc.log})", file=stream, flush=True)
        return rc
    walls = ",".join(f"r{r['rank']}:{r['wall_s']:.0f}s" for r in recs)
    print(f"{_report.PREFIX} rowpair launch: ranks_ok={sum(1 for r in recs if r['ok'])}/{P} walls={walls} wall_s={time.time() - t0:.0f} "
          f"logs={os.path.dirname(recs[0]['log'])}", file=stream, flush=True)
    return max(int(r["rc"]) for r in recs)


def install_rank(model, P: int, builder=None) -> dict:
    """Inside a rank process, after the model is loaded and the kit's configure() ran: join the run's process group (NCCL on the rank's
    GPU; :func:`opt_core.mem.rowpair.dist.init_from_env` reads the rank environment) and install the sharded pair stack on ``model`` — and,
    given the run's ``builder`` (``ESMFold2InputBuilder``), rank 0's featurisation for every rank on it (:mod:`.rowpair_feats`).
    ``P == 1``: returns at once (nothing joined, nothing installed, the builder untouched)."""
    P = ngpu.check_n_gpu(P)
    if P == 1:
        return install(model, 1, builder=builder)
    import faulthandler
    import signal
    if not faulthandler.is_enabled():
        faulthandler.enable(file=sys.stderr, all_threads=True)               # a rank that dies hard leaves its Python stack in its transcript
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)   # `kill -USR1 <rank pid>` dumps every thread's stack (hang triage)
    from opt_core.mem.rowpair import dist as RD
    t0 = time.time()
    if not RD.is_dist():
        RD.init_from_env(timeout_s=NCCL_TIMEOUT_S)                            # joins the group and arms the core's exit guard (a teardown stuck after the main thread
        STATE["group"] = True                                                  # ended is ended by name within ROWPAIR_EXIT_GRACE_S, exiting with the status finish() registered)
        import atexit
        atexit.register(finish)                                                # the group is left on every path out of the interpreter (idempotent with the kit's own call)
        STATE["warm_s"] = _warm_collectives()                                  # every collective form the sharded stack uses, once, now — the library's communicators,
                                                                               # channels and buffers exist BEFORE the model's large allocations (no lazy set-up mid-fold)
    print(f"{_report.PREFIX} rowpair rank: {EV.fields_text(P)} rank={RL.rank()} group_joined_s={time.time() - t0:.1f} "
          f"store={'file' if os.environ.get(RL.ENV_STORE) else 'tcp'} log_dir={os.environ.get(RL.ENV_LOG_DIR, '-')} pid={os.getpid()}",
          file=sys.stderr, flush=True)
    # The sharded pair init carries a rows replica of torch's REJECTION-form trunc_normal_ only and refuses BY NAME on a torch whose
    # _no_grad_trunc_normal_ is the inverse-CDF form (uniform_ + erfinv_): on such a torch (newer than the kit's pin) rowchunk.tn_shim
    # swaps in a whole-tensor rejection sampler so the route can start — the init RNG stream then differs from that torch's own. Inert
    # on the pinned torch (needed() is False: nothing is patched); named trunc_normal=native|shim on this lever's LEVER line.
    from .rowchunk import tn_shim as _tn
    STATE["trunc_normal"] = "shim" if _tn.apply().get("applied") else "native"
    out = install(model, P, builder=builder)
    # The route's row-chunking levers (FOR_ROUTE; esmfold2_opt.rowchunk). MUST follow install(): install() re-owns the pair frames, so a
    # lever bound earlier would be shadowed (bound, reported, never run). The resolved set's members are installed by name; a member that
    # cannot bind raises RowpairRefused (this route's by-name refusal: NOT ACTIVE, exit 3) — never a fold without it under the mode's name.
    from .rowchunk import install as _rowchunk
    out["rowchunk"] = _rowchunk.install(model, P, _route_levers_wanted())
    try:
        from . import stack as _stack
        _stack.settle_route(out["rowchunk"], model)                       # the deferred levers join applied / partial in the activation report; their LEVER lines print here
    except ImportError:
        pass
    return out


def _route_levers_wanted() -> Tuple[str, ...]:
    """The FOR_ROUTE members the active mode's resolved set names (all of them when no mode is active in this process: a direct
    install_rank call); the ablation variable subtracts members by name before this (modes.resolve)."""
    try:
        from . import stack as _stack
        res = (_stack.status() or {}).get("resolution")
    except Exception:  # noqa: BLE001
        res = None
    if res is None or not getattr(res, "levers", None):
        return tuple(FOR_ROUTE)
    return tuple(n for n in FOR_ROUTE if n in set(res.levers))


def _trimul_rows_word() -> str:
    """The LEVER line's ``trimul_rows=`` word: which statements a row block's triangle multiplication runs — ``provider:<core lever>`` (the
    core's fused row-block kernels above the size gate, :func:`_with_trimul_provider`; the served / declined counts by reason are the core's own
    ``LEVER name=F2.trimul_rows`` line at exit) or ``statements:<why>``."""
    try:
        from opt_core.mem.rowpair import trimul_fused as RTF
    except ImportError as e:
        return f"statements:provider_absent:{type(e).__name__}"
    mt = (os.environ.get(ENV_TRIMUL_MIN_TOKENS, "") or "").strip()
    gate = f"min_tokens={mt}:env" if mt else "min_tokens=0:cc>=9|core_default"
    return f"provider:{RTF.LEVER}:{os.environ.get(RTF.ENV_KERNELS, 'fpf_v4')}:{gate}"


def _warm_collectives() -> float:
    """One tiny call of each collective form used below (all_reduce, all_gather, all_to_all, the neighbour isend/irecv ring, an object
    gather) so the collective library allocates its communicators and buffers at join time, while the card is nearly empty."""
    import torch
    import torch.distributed as tdist
    from opt_core.mem.rowpair import dist as RD
    t0 = time.time()
    P, r = RD.world()
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    x = torch.ones(8, device=dev)
    tdist.all_reduce(x)
    out = [torch.empty_like(x) for _ in range(P)]
    tdist.all_gather(out, x)
    y = torch.empty(8 * P, device=dev)
    tdist.all_to_all_single(y, torch.ones(8 * P, device=dev))
    nxt, prv = (r + 1) % P, (r - 1) % P
    recv = torch.empty_like(x)
    for w in tdist.batch_isend_irecv([tdist.P2POp(tdist.isend, x, nxt), tdist.P2POp(tdist.irecv, recv, prv)]):
        w.wait()
    _all_gather_object((r, P), torch)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return time.time() - t0


def finish(exit_code: Optional[int] = None) -> None:
    """End of a rank's run: leave the process group this module joined (idempotent; a no-op at n_gpu=1). The kit calls it on every path out
    of the rank's command and it is registered ``atexit`` (a rank that exits with the group joined can block in the collective library's teardown).
    ``exit_code`` is the kit's verdict for this rank: registered with the core (``dist.exit_code``) so a teardown that sticks still ends the
    process with it. The leave itself is the core's bounded ``dist.destroy()``: silent when the library's teardown returns in time; when it
    overruns (a peer diverged mid-collective) the communicators are aborted or abandoned within ``ROWPAIR_ABORT_TIMEOUT_S`` under one
    ``RANKLEAVE`` line and the process ends through the core's hard exit (its EXIT tally printed) instead of the library's blocking destructors."""
    clean = sys.exc_info()[0] is None
    STATE.setdefault("finished_clean", clean)
    if TRIMUL_PROVIDER.get("bound") and not STATE.get("trimul_rows_line"):     # the core provider's ONE activation-evidence line (served / declined by reason), once per rank
        STATE["trimul_rows_line"] = True
        try:
            from opt_core.mem.rowpair import trimul_fused as RTF
            RTF.emit_line(_report.TAG, rank=STATE.get("rank"))
        except Exception as e:                                                 # formatting only; never holds up the leave
            print(f"{_report.PREFIX} trimul_rows line unavailable: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    from opt_core.mem.rowpair import dist as RD
    if exit_code is not None:
        STATE["exit_code"] = int(exit_code)
    RD.exit_code(STATE.get("exit_code", 0 if STATE["finished_clean"] else RD.EXIT_TEARDOWN_STUCK))
    if STATE.get("group"):
        try:
            verdict = RD.destroy()                                            # no device synchronize first: on a wedged stream it would block like the library's own teardown
            if verdict not in ("destroyed", "none"):
                STATE["group_abandoned"] = verdict                            # named in the transcript by the core's RANKLEAVE line; outputs are complete
        finally:
            STATE["group"] = False


# ================================================================================================================ install (P > 1 only)
def _modules():
    return importlib.import_module(COMMON_MODULE), importlib.import_module(MODEL_MODULE)


def _trunk_backend(model) -> Optional[str]:
    blocks = getattr(getattr(model, "folding_trunk", None), "blocks", None)
    if not blocks:
        raise RowpairRefused("install: model.folding_trunk has no blocks")
    return getattr(blocks[0], "_kernel_backend", None)


def _w4():
    return sys.modules.get(W4_MODULE)


def _t9_ready() -> str:
    """The fused line's sharded trimul re-binds the entry point W4's lever tx serves (``_fused_trimul_with_residual``): ef2_w4 must be enabled with tx
    bound (else refused by name). Returns the install line's one-token TriMul word ``tx:<tier word>``."""
    W4 = _w4()
    if W4 is None or not W4._STATE.get("enabled"):
        raise RowpairRefused("install: kernel backend 'fused' without the kit's W4 module enabled (ef2_w4): n_gpu>1 on the fused line needs lever tx bound")
    st = W4.tx_state() if hasattr(W4, "tx_state") else {}
    if not (W4._STATE.get("tx") and st.get("bound")):
        raise RowpairRefused("install: lever tx (the shared core's TriMul provider binding) is not bound on this line/device: the fused Biohub trimul has no row-sharded form in this kit "
                             f"(ef2_w4 tx: {st.get('why')})")
    return "tx:" + "".join(str(st.get("word")).split())                        # one token


def install(model, n_gpu, builder=None) -> dict:
    """Install the row-sharded pair stack on ``model`` for ``n_gpu = P > 1`` (this rank's process group must exist: :func:`install_rank`),
    and on ``builder`` (the run's ``ESMFold2InputBuilder``, when given) rank 0's featurisation for every rank: its ``prepare_input`` becomes
    :func:`.rowpair_feats.prepare_input_rank0` over what the instance held (install line ``data_form=rank0_bcast``; without a builder every
    rank featurises and the forward's entry prints the per-rank digest census, ``data_form=per_rank``).
    ``P == 1``: NOTHING is installed or read — the return says ``n_gpu=1 sharding=none``. Returns :func:`report`."""
    P = ngpu.check_n_gpu(n_gpu)
    if P == 1:
        STATE.update(P=1, rank=0, installed=False, backend=None)
        return report()
    from opt_core.mem.rowpair import dist as RD
    if STATE["installed"]:
        raise RowpairRefused("install: already installed in this process (one model per rank process)")
    if not RD.is_dist():
        raise RowpairRefused(f"install: n_gpu={P} without a process group (install_rank joins it; a bare install() at P>1 is refused)")
    world, rank = RD.world()
    if world != P:
        raise RowpairRefused(f"install: n_gpu={P} but the process group has world={world}")
    msa_rows = None
    if getattr(model, "msa_encoder", None) is not None:                      # variant full_msa: the MSA encoder's pair writes on rows (rowpair_msa.py)
        try:
            from . import rowpair_msa as msa_rows
        except ImportError:
            raise RowpairRefused("install: this checkpoint has an MSA encoder (variant full_msa) and esmfold2_opt.rowpair_msa is not in this kit — "
                                 "n_gpu>1 supports variants fast | full_nomsa here") from None
    _guard_weights_replicated(model)                                                    # a per-rank weight desync refuses by name (one allreduce), never folds garbage
    if getattr(model, "training", False):
        raise RowpairRefused("install: the model is in train() mode; n_gpu>1 runs inference only (eval(): identical statements and RNG on every rank)")
    d_pair = int(model.parcae_input_norm.normalized_shape[0])
    backend = _trunk_backend(model)
    CMN, MOD = _modules()
    if backend == CMN.BACKEND_CUEQ:
        raise RowpairRefused("install: kernel backend 'cuequivariance' has no row-sharded trimul in this kit (fused | reference: the stock statements through opt_core trimul_update_)")
    XL = sys.modules.get(XL_MODULE)
    xl_own = bool(XL is not None and XL._CFG.get("own"))                    # x2 OWN: FoldingTrunk.forward takes [pair] and frees block inputs early — the shard is handed over the same way
    xl_loopfree = bool(XL is not None and XL._CFG.get("loopfree"))          # x2: the add-on's loop (with frees) is installed — re-issued on rows below, its per-fold counter kept
    if backend == CMN.BACKEND_FUSED:
        if d_pair != D_PAIR:
            raise RowpairRefused(f"install: d_pair={d_pair}; the sharded TriMul launch is for d_pair={D_PAIR}")
        trimul_impl = _t9_ready()
    else:
        trimul_impl = "reference:einsum-split"
    ch = getattr(model, "confidence_head", None)
    STATE["compose"] = _composition_census(model, CMN)                        # the single-GPU levers found installed (read BEFORE any rebinding): composing on rows / bypassed BY NAME
    # ---- the sharded modules run EAGER: the fast line's CUDA-graph wrappers (levers tg / eg) decide by the token count of a square pair
    #      argument; a row shard [1, R, N, C] is not square, so the generic wrapper would CAPTURE the sharded lm_encoder — collectives inside a
    #      capture, and a private graph pool that keeps every byte the capture touched (a pool growing with the shard until the
    #      reservation reaches the card). Their eager forwards are restored here for the run (the kit's budget gate already runs
    #      the single-GPU line eager above EF2_GRAPH_BUDGET_TOKENS; nothing changes at n_gpu=1).
    eager_forced = []

    def eager_of(mod, label):
        """The module's eager forward: the fast line's graph wrapper on it (if any) is bypassed for the run and named."""
        if getattr(mod, "_ef2opt_graphed", False) and "forward" in vars(mod):
            eager_forced.append(label)
            return mod._ef2opt_eager_forward
        return mod.forward
    for label in ("folding_trunk", "lm_encoder"):                            # called by the loop below through self.<name>(...): one patch = the eager forward
        mod = getattr(model, label, None)
        if mod is not None:
            fwd = eager_of(mod, label)
            if label in eager_forced:                                         # (a bound method is a new object per access: membership, not `is`)
                _patch(mod, "forward", fwd, f"model.{label}")
    sh = getattr(model, "structure_head", None)                               # the fast line's graphed roll-out (lever sg) / ef2_dit's roll-out graph (lever ro): a step /
    EO = sys.modules.get("ef2_opt")                                           # boundary graph would capture the row-sharded diffusion forward (collectives, host-read block plans,
    DIT = sys.modules.get("ef2_dit")                                          # replicated-state guards). The kit's ONE capture gate decides it: the sampler-site graph budget is
    STATE["sampler"] = "stock"                                                # SAMPLER_SITE_BUDGET_SHARDED token on this rank, so ef2_opt's _sample_v2 AND ef2_dit's _sample_dit run
    if EO is not None and hasattr(getattr(EO, "CFG", None), "graph_budget_tokens_sampler"):   # their EAGER forms (ef2_opt prints `graph budget: sampler ... -> eager` once per shape)
        STATE["sampler_budget_prev"] = EO.CFG.graph_budget_tokens_sampler
        EO.CFG.graph_budget_tokens_sampler = SAMPLER_SITE_BUDGET_SHARDED
        STATE["sampler_budget"] = SAMPLER_SITE_BUDGET_SHARDED
    dit_rollout = bool(DIT is not None and sh is not None and "sample" in vars(sh) and getattr(vars(sh)["sample"], "__func__", None) is getattr(DIT, "_sample_dit", None))
    if dit_rollout:                                                           # lever ro: ef2_dit's roll-out STAYS the sampler — eager boundaries around dm_forward_rows (its
        if STATE.get("sampler_budget") != SAMPLER_SITE_BUDGET_SHARDED:        # in-place diffusion_state sync lands in the roll's x_noisy static, which its Kabsch head / tail read);
            raise RowpairRefused("install: ef2_dit's roll-out is the sampler but ef2_opt's sampler-site graph budget is not reachable "
                                 "(ef2_opt.CFG.graph_budget_tokens_sampler): its boundary graph would capture the row-sharded diffusion forward — refused")
        eager_forced.append("structure_head.sample(ef2_dit.rollout:eager_boundaries)")   # kd (device Kabsch) rides on it unchanged; out-of-scope calls take ef2_opt's chain, itself
        STATE["sampler"] = "ef2_dit.rollout:eager"                            # eager by the same budget
    elif sh is not None and "sample" in vars(sh) and hasattr(sh, "_ef2opt_eager_sample"):   # no roll-out installed: the eager sampler serves, as before
        eager_forced.append("structure_head.sample")
        _patch(sh, "sample", sh._ef2opt_eager_sample, "model.structure_head")
        STATE["sampler"] = "ef2_opt.eager"
    # ---- rebinding (originals kept in PATCHES; instance attributes shadow the class / ef2_opt wrappers they wrap; one patch per attribute)
    _patch(model, "forward", types.MethodType(_forward_rows, model), "model")                 # the pair representation BORN as rows (rowpair_heads)
    _patch(model, "_run_one_loop", types.MethodType(_run_one_loop_sharded, model), "model")
    _patch(model.parcae_coda, "forward", _coda_forward(eager_of(model.parcae_coda, "parcae_coda")), "model.parcae_coda")
    if ch is not None and getattr(ch, "folding_trunk", None) is not None:
        _patch(ch, "forward", _conf_head_rows(ch, eager_of(ch.folding_trunk, "confidence_head.folding_trunk")), "model.confidence_head")
    sh = getattr(model, "structure_head", None)
    dm = getattr(sh, "diffusion_module", None) if sh is not None else None
    if dm is not None:
        inst = vars(dm).get("forward")
        inst_fn = getattr(inst, "__func__", inst)
        if inst is None:
            STATE["dm_forward_replaced"] = "stock"
        elif DIT is not None and inst_fn is getattr(DIT, "_dm_forward_dit", None):       # lever dit's fused step takes the WHOLE pair bias [B, H, N, N]: kept as the inner forward,
            STATE["dm_forward_replaced"] = "ef2_dit.fused_step"                           # the whole route's engine (rowpair_heads.dm_forward_whole); the rows route never calls it
        elif getattr(dm, "_mk_state", None) is not None:
            STATE["dm_forward_replaced"] = "mk_sampler"
        else:
            STATE["dm_forward_replaced"] = "instance:" + getattr(inst_fn, "__name__", "?")
        _patch(dm, "forward", _dm_forward(dm), "model.structure_head.diffusion_module")   # z_cond rows + local-query token transformer (rowpair_heads)
    STATE["msa_census"] = {}
    if msa_rows is not None:                                                              # rowpair_msa's entry: rebinds model.msa_encoder for row inputs; its census
        STATE["msa_rows"] = msa_rows.install_msa_rows(model, None, STATE["msa_census"]) or "installed"   # keys (msa=rows, msa_m=replicated, ...) join the lines
        STATE["msa_model"] = model                                                                         # uninstall() puts the MSA blocks' PairTransition chunks back (rowpair_msa.uninstall_msa_rows)
    try:
        from . import atom_swa                                                            # atom_swa: banded atom sliding-window attention (exact attended set)
    except ImportError:
        atom_swa = None
    STATE["atom_swa"] = atom_swa.install(model, P)["path"] if atom_swa is not None else "absent:stock_dense_swa_masks"   # banded | stock_flash; absent = O(N_atom^2) replicated masks
    STATE["atom_swa_fields"] = list(atom_swa.census_fields().items()) if atom_swa is not None else [("atom_swa", STATE["atom_swa"])]
    STATE["graphs_off"] = eager_forced
    if backend == CMN.BACKEND_FUSED:
        PATCHES.replace(CMN, "_fused_trimul_with_residual", _trimul_fused_dispatch)          # W4's _trimul_w4 at this name is kept as the original and serves every non-sharded call
    PATCHES.replace(CMN.TriangleMultiplicativeBlock, "forward", _tmb_forward_dispatch)      # under EVERY backend: the MSA encoder's trimul modules are these
    PATCHES.replace(CMN.Transition, "forward", _transition_dispatch)                        # pair transitions on a shard run per row block in place
    #   modules whatever the trunk's backend (rows reach them through rowpair_msa; passthrough outside the sharded context) — msa_trimul=reference_rows
    if builder is not None:                                                                  # rank 0 featurises, every other rank receives its tensors and the ranks' digests must agree
        _patch(builder, "prepare_input", RF.prepare_input_rank0(builder.prepare_input), "builder")   # (rowpair_feats): OUTERMOST on the builder instance, over the kit's feature cache and
    #   SMILES seed guard (stack.apply_to) — one hand-over + one digest agreement per prepare_input call on every rank, whatever a rank-local cache holds
    STATE.update(P=P, rank=rank, installed=True, backend=backend or "reference", trimul=trimul_impl, layouts={}, folds=[], xl_own=xl_own, xl_loopfree=xl_loopfree,
                 lm_encoder=getattr(model, "lm_encoder", None) is not None, coda_layers=len(model.parcae_coda.blocks),
                 conf_trunk=bool(ch is not None and getattr(ch, "folding_trunk", None) is not None))
    line = EV.lever_line(_report.TAG, "on", P, name=LEVER, with_schedule=False, rank=rank, trimul=trimul_impl, backend=STATE["backend"],
                         trunks=",".join(t for t in SHARDED_TRUNKS if t != "confidence_head.folding_trunk" or STATE["conf_trunk"]),
                         replicated=",".join(REPLICATED), graphs_off=",".join(STATE.get("graphs_off") or []) or "none", pair_rng="philox_rows|rank_rows(cpu)",
                         mode_s="born_rows,no_gather,distogram_rows,zcond_rows,dit_local_q|dit_whole_when_fits,conf_rows", dm_forward_replaced=STATE.get("dm_forward_replaced", "none"), alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset"),
                         msa_rows=("on" if STATE.get("msa_rows") else "n/a"), xl_own=int(xl_own), **{RF.WORD_FORM: RF.STATE["form"]}, patches=len(PATCHES),
                         trimul_rows=_trimul_rows_word(), empty_cache=(os.environ.get(ENV_EMPTY_CACHE, "") or EMPTY_CACHE_DEFAULT),
                         sampler_route=(os.environ.get(ENV_SAMPLER, "") or "auto"), dit_rows=(os.environ.get("EF2_ROWPAIR_DIT_ROWS", "") or "big"), dit_bias=os.environ.get("ROWPAIR_DIFF_BIAS", "unset"),
                         rank_threads=os.environ.get("ROWPAIR_RANK_THREADS", "unset"), trunc_normal=STATE.get("trunc_normal", "native"),
                         sampler=STATE.get("sampler", "stock"), sampler_graph_budget=STATE.get("sampler_budget", "unset"),
                         **dict(STATE.get("compose") or {}),
                         **{k: v for k, v in list(STATE.get("atom_swa_fields") or []) + list((STATE.get("msa_census") or {}).items())})
    STATE["install_line"] = line
    print(line, file=sys.stderr, flush=True)
    return report()


def _composition_census(model, CMN) -> Dict[str, str]:
    """Which single-GPU levers of the new sets this rank found INSTALLED and how the row form meets them — words for the rowpair LEVER line:
    ``compose_transition`` (the C.Transition class forward the per-row-block dispatcher wraps: ef2_pair_v2's t15 kernel / W4's / stock),
    ``compose_atom`` (instance forwards on the diffusion module's atom encoder / decoder: called replicated per rank as installed),
    ``compose_msa_trimul`` (instance forwards on the MSA TriMul engines — ef2_hoist's trimul / glue: BYPASSED on rows by name, the row form issues
    opt_core trimul_update_ from the module's weights), ``compose_loop`` (an instance ``_run_one_loop`` this line's sharded loop shadows: ef2_opt's
    ls / rg), ``route_off`` (NOT_FOR_ROUTE levers nevertheless found installed: replaced on rows BY NAME — the package normally leaves them out)."""
    out: Dict[str, str] = {}
    tf = getattr(CMN.Transition, "forward", None)
    P2 = sys.modules.get("ef2_pair_v2")
    if P2 is not None and tf is getattr(P2, "_transition_forward_v2", None):
        out["compose_transition"] = "t15:per_row_block"
    elif sys.modules.get("ef2_transition_cute") is not None and tf is getattr(sys.modules["ef2_transition_cute"], "_transition_forward_cute", None):
        out["compose_transition"] = "t16:per_row_block"                # the 9.0 class's CuTe transition (its PairTransition class forward serves the MSA blocks' rows through transition_residual_rows_)
    else:
        out["compose_transition"] = getattr(tf, "__module__", "?").rsplit(".", 1)[-1] + "." + getattr(tf, "__name__", "?") + ":per_row_block"
    sh = getattr(model, "structure_head", None)
    dm = getattr(sh, "diffusion_module", None) if sh is not None else None
    words = []
    for label in ("atom_encoder", "atom_decoder"):
        mod = getattr(dm, label, None) if dm is not None else None
        inst = vars(mod).get("forward") if mod is not None else None
        if inst is not None:
            fn = getattr(inst, "__func__", inst)
            words.append(f"{label}={getattr(fn, '__module__', '?').rsplit('.', 1)[-1]}.{getattr(fn, '__name__', '?')}")
    out["compose_atom"] = ("instance:" + "+".join(words) + ":replicated_per_rank") if words else "stock:replicated_per_rank"
    enc = getattr(model, "msa_encoder", None)
    names = set()
    if enc is not None:
        for blk in getattr(enc, "blocks", []):
            for tm in (getattr(blk, "tri_mul_out", None), getattr(blk, "tri_mul_in", None)):
                for eng in ([tm] + list(tm.modules())) if tm is not None else []:
                    if isinstance(eng, CMN.TriangleMultiplicativeBlock) and "forward" in vars(eng):
                        fn = getattr(vars(eng)["forward"], "__func__", vars(eng)["forward"])
                        names.add(getattr(fn, "__name__", "?"))
    out["compose_msa_trimul"] = ("instance:" + ",".join(sorted(names)) + ":bypassed_on_rows(opt_core.trimul_update_)") if names else "none"
    inst = vars(model).get("_run_one_loop")
    if inst is not None:
        fn = getattr(inst, "__func__", inst)
        out["compose_loop"] = f"{getattr(fn, '__module__', '?').rsplit('.', 1)[-1]}.{getattr(fn, '__name__', '?')}:shadowed_by_run_one_loop_sharded"
    else:
        out["compose_loop"] = "stock:shadowed_by_run_one_loop_sharded"
    found = []
    DIT, V2, HO, EO = (sys.modules.get(n) for n in ("ef2_dit", "ef2_msa_v2", "ef2_hoist", "ef2_opt"))
    try:
        if DIT is not None and (DIT.levers_on() or {}).get("dit"): found.append("dit")
        if V2 is not None: found += [n for n in ("m16", "m17", "mh") if n in (V2.levers_on() or [])]
        if HO is not None: found += [n for n in ("trimul", "glue", "disto") if (HO.stats().get("levers") or {}).get(n)]
        if EO is not None and getattr(getattr(EO, "CFG", None), "loop_static", False): found.append("ls")
        if EO is not None and getattr(getattr(EO, "CFG", None), "recycle_graph", False): found.append("rg")
    except Exception as e:  # noqa: BLE001 — a lever module without the record API: named, not fatal
        found.append(f"unreadable({type(e).__name__})")
    out["route_off_installed"] = ",".join(n for n in NOT_FOR_ROUTE if n in found) or "none"
    return out


def _patch(owner, name: str, new, label: str):
    """``PATCHES.replace`` with a NAME for a module-instance owner (the install record prints ``Owner.name``; an ``nn.Module`` instance has no
    ``__name__`` and its repr is the whole module tree)."""
    if not hasattr(owner, "__name__"):
        owner.__dict__["__name__"] = label
    return PATCHES.replace(owner, name, new)


def uninstall() -> List[str]:
    """Restore every original (tests; a rank process simply exits)."""
    try:
        from . import atom_swa as _swa
        _swa.uninstall()                                                          # atom_swa's class-level forward
    except ImportError:
        pass
    names = PATCHES.restore()
    if "sampler_budget_prev" in STATE:                                            # ef2_opt's sampler-site graph budget as it was before install
        EO = sys.modules.get("ef2_opt")
        if EO is not None:
            EO.CFG.graph_budget_tokens_sampler = STATE["sampler_budget_prev"]
        STATE.pop("sampler_budget_prev", None); STATE.pop("sampler_budget", None)
    STATE.pop("sampler", None)
    m = STATE.pop("msa_model", None)
    if m is not None:
        from . import rowpair_msa as _msa
        _msa.uninstall_msa_rows(m)                                                # the MSA install mark and the blocks' PairTransition chunk sizes
    TRIMUL_STATS.clear()
    RF.reset()
    STATE.update(installed=False, layouts={}, P=1, rank=0)
    return names


def report() -> dict:
    """The adapter's state for status(): ``{n_gpu, sharding, installed, patches, backend, trimul, folds: [...]}``."""
    P = int(STATE["P"])
    return {"n_gpu": P, "sharding": EV.sharding_value(P), "installed": bool(STATE["installed"]), "rank": int(STATE["rank"]),
            "patches": PATCHES.names(), "backend": STATE.get("backend"), "trimul": STATE.get("trimul"), "replicated": list(REPLICATED),
            RF.WORD_FORM: RF.STATE["form"], "x_pred_mismatch": int(STATE.get("x_pred_mismatch", 0)), "folds": [_fold_record(f) for f in STATE.get("folds", [])]}


def _fold_record(f: dict) -> dict:
    """A fold's evidence as plain JSON types (the CUDA events stay in the live record until resolved)."""
    return {k: v for k, v in f.items() if k != "events"}


# ================================================================================================================ per-fold layout, timing
class _Ctx:
    """The sharding context of the running fold (module-global: one fold at a time per process)."""
    active: bool = False          # inside a sharded trunk call: the trimul dispatchers take the sharded form
    layout = None                 # opt_core.mem.rowpair.dist.Layout of this fold's N
    mask_loc = None               # [B, R, N] pair mask rows of this rank
    pending_coda: bool = False    # _run_one_loop returned a shard; parcae_coda gathers
    fold: Optional[dict] = None   # the evidence record of this fold (STATE['folds'][-1])
    below_floor: bool = False     # the running fold is below the sharding floor: every rebound callable takes its kept original (the single-GPU statements)


CTX = _Ctx()


def _layout(N: int):
    from opt_core.mem.rowpair import dist as RD
    lay = STATE["layouts"].get(int(N))
    if lay is None:
        lay = RD.require_sharded(RD.Layout.auto(int(N), int(STATE["P"]), int(STATE["rank"])), "esmfold2 rowpair layout")
        STATE["layouts"][int(N)] = lay
    return lay


def _layout_or_none(N: int):
    """The sharded layout of this N over P ranks, or None when N is below the sharding floor — ``Layout.auto`` (the family's grid rule, and
    ONLY it) has no row grid without an empty rank, e.g. a 20-token input on 2 GPUs — that input is folded whole (:func:`_forward_below_floor`).
    Every other refusal of the layout (no process group, P=1) stays a refusal."""
    from opt_core.mem.rowpair import dist as RD
    if int(N) in STATE["layouts"]:
        lay = STATE["layouts"][int(N)]
    else:
        try:
            lay = RD.Layout.auto(int(N), int(STATE["P"]), int(STATE["rank"]))
        except RowpairRefused:                                                            # the grid rule: N below the sharding floor for P
            lay = None
        STATE["layouts"][int(N)] = lay
    return None if lay is None else RD.require_sharded(lay, "esmfold2 rowpair layout")


def _forward_below_floor(self, N: int, call: dict):
    """The fold of an input below the sharding floor under n_gpu>1: ``ESMFold2Model.forward`` as kept (the single-GPU statements this lever
    rebound nothing of — every wrapper takes its original while ``CTX.below_floor``), on THIS rank like on every other (the folds are the
    same computation; the output rank's is written; no collective runs, so no rank waits on another). Named once; the fold's census record
    carries ``sharding=none:below_floor``."""
    P, rank = int(STATE["P"]), int(STATE["rank"])
    print(f"{_report.PREFIX} n_gpu={P}: input below the sharding floor (N={int(N)}: no row grid over {P} ranks) — folded unsharded on rank {rank} "
          f"(every rank folds it whole, rank 0's is written; no collective) sharding={BELOW_FLOOR_WORD}", file=sys.stderr, flush=True)
    f = {"N": int(N), "P": P, "rank": rank, "sharding": BELOW_FLOOR_WORD, "t": {}, "events": [], "cuda": False, **RF.fold_fields()}
    STATE["folds"].append(f)
    orig = PATCHES.original(self, "forward")
    prev = (CTX.active, CTX.layout, CTX.mask_loc, CTX.pending_coda, CTX.fold)
    CTX.below_floor, CTX.active, CTX.layout, CTX.mask_loc, CTX.pending_coda, CTX.fold = True, False, None, None, False, f
    t0 = time.time()
    try:
        out = orig(**call)
    finally:
        CTX.below_floor = False
        CTX.active, CTX.layout, CTX.mask_loc, CTX.pending_coda, CTX.fold = prev
    f["wall_s"] = round(time.time() - t0, 3)
    return out


@contextlib.contextmanager
def _sharded(lay, mask_loc):
    prev = (CTX.active, CTX.layout, CTX.mask_loc)
    CTX.active, CTX.layout, CTX.mask_loc = True, lay, mask_loc
    try:
        yield
    finally:
        CTX.active, CTX.layout, CTX.mask_loc = prev


def pair_block_rows_(block, z_rows, mask_rows, layout=None):
    """The row form of ONE stock pair block (``PairUpdateBlock`` / ``MSAEncoderBlock`` pair sub-modules / a ``FoldingTrunk``): the stock module
    is CALLED on this rank's rows ``[B, R, N, C]`` inside the sharded context, where the class-level trimul dispatchers take the row-sharded
    form (both backends: the stock statements through opt_core ``trimul_update_``; transitions per row block) and LayerNorms are row-local
    as they stand. ``mask_rows``: ``[B, R, N]`` pair-mask rows (float or bool). Returns what the block returns. (``rowpair_msa`` binds by this name.)"""
    lay = layout if layout is not None else CTX.layout
    if lay is None or not _is_shard_of(z_rows, lay):
        raise RowpairRefused(f"pair_block_rows_: z {tuple(z_rows.shape)} is not a row shard of the running fold's layout")
    m = mask_rows.float() if mask_rows is not None else None
    with _sharded(lay, m.contiguous() if m is not None else None):
        return block(z_rows, pair_attention_mask=m)


def _memfields(torch, device, tag: str, f: dict) -> None:
    """Per-phase memory evidence on this card: torch's peak allocated and peak reserved bytes (GiB) since the last reset, and the driver's
    used bytes now (``mem_get_info``: everything resident on the device — this process's pools, the collective library's buffers, contexts)."""
    if device.type != "cuda":
        return
    gib = 2 ** 30
    f[f"max_alloc_gib_{tag}"] = torch.cuda.max_memory_allocated(device) / gib
    f[f"max_reserved_gib_{tag}"] = torch.cuda.max_memory_reserved(device) / gib
    free, total = torch.cuda.mem_get_info(device)
    f[f"driver_used_gib_{tag}"] = (total - free) / gib


def _new_fold(lay, torch, device) -> dict:
    cuda = device.type == "cuda"
    f = {"N": lay.N, **{k: v for k, v in lay.facts().items() if k != "N"}, "t": {}, "events": [], "cuda": cuda,
         "alloc_gib_entry": (torch.cuda.memory_allocated(device) / 2 ** 30) if cuda else None, **RF.fold_fields()}
    _memfields(torch, device, "before_loop", f)
    STATE["folds"].append(f)
    CTX.fold = f
    return f


def _timed(name: str, fn, torch):
    """``fn()`` with its wall recorded under ``name`` in the fold record (CUDA events resolved at the fold's end; perf_counter on CPU)."""
    f = CTX.fold
    if f is None:
        return fn()
    if f["cuda"]:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = fn()
        e1.record()
        f["events"].append((name, e0, e1))
        return out
    t0 = time.perf_counter()
    out = fn()
    f["t"].setdefault(name, []).append(round(time.perf_counter() - t0, 4))
    return out


def _resolve_events(f: dict, torch) -> None:
    if f.get("events"):
        torch.cuda.synchronize()
        for name, e0, e1 in f["events"]:
            f["t"].setdefault(name, []).append(round(e0.elapsed_time(e1) / 1000.0, 4))
        f["events"] = []


def _fold_line(f: dict) -> str:
    from opt_core.report import kv
    t = {k: (round(sum(v) / len(v), 3), len(v)) for k, v in f["t"].items() if v}
    pairs = [("N", f["N"]), ("P", f["P"]), ("B", f["B"]), ("rank", f["rank"]), ("rows", f"{f['r0']}:{f['r1']}"), ("R", f["R"]), ("Rmax", f["Rmax"]),
             ("freed_gib", f.get("freed_gib")), ("alloc_gib_entry", _r2(f.get("alloc_gib_entry"))), ("alloc_gib_sharded", _r2(f.get("alloc_gib_sharded")))]
    for tag in ("before_loop", "after_coda", "after_conf"):
        pairs += [(f"max_alloc_gib_{tag}", _r2(f.get(f"max_alloc_gib_{tag}"))), (f"max_reserved_gib_{tag}", _r2(f.get(f"max_reserved_gib_{tag}"))),
                  (f"driver_used_gib_{tag}", _r2(f.get(f"driver_used_gib_{tag}")))]
    pairs += [(f"{k}_s", f"{m}x{n}") for k, (m, n) in sorted(t.items())]
    pairs += [("empty_cache_phases", f.get("empty_cache_phases") or "none"), ("empty_cache_phase_s", f.get("empty_cache_phase_s", 0.0))]
    pairs += [("diff_noise", f.get("diff_noise", "-")), ("diff_rank_spread_A", f.get("diff_rank_spread_A", "-"))]
    pairs += [("warm_s", _r2(STATE.get("warm_s"))), ("empty_cache_s1_s", _r2(f.get("empty_cache_s1_s"))), ("empty_cache_tail_s", _r2(f.get("empty_cache_tail_s")))]
    for _d, _st in sorted(TRIMUL_STATS.items()):                                # opt_core ContractStats of this fold's triangle multiplications (RB, tiles, deferred bytes, mode)
        pairs += [(f"trimul_{_d}_{k}", ("".join(str(v).split()) if not isinstance(v, (int, float)) else v)) for k, v in sorted(_st.facts().items())]
    pairs += [(k, f.get(k)) for k in ("sampler_route", "sampler_route_why", "sampler_word", "sampler_S", "sampler_need_gib", "sampler_headroom_gib", "sampler_live_gib", "sampler_avail_gib",
                                          "sampler_whole_steps", "sampler_z_whole_gib", "sampler_dit_steps") if k in f]
    if _DM_BOX.get("route") == "rows":                                                    # the rows route ran the token transformer: which attention core served its query blocks
        from . import rowpair_heads as _H
        pairs += list(_H.dit_rows_facts(_DM_BOX).items())
    pairs += [("trimul_provider", ("bound" if TRIMUL_PROVIDER["bound"] else ("stock:" + str(TRIMUL_PROVIDER["why"]) if TRIMUL_PROVIDER["bound"] is False else "unused"))), ("trimul_provider_why", TRIMUL_PROVIDER["why"])]
    pairs += list(EV.schedule_fields()) + [("guards", ",".join(f.get("guards", [])) or "none"), ("features_replicated", f.get("features_replicated")),
                                          ("features_differ", f.get("features_differ", "none")), ("x_pred_replicated", f.get("x_pred_replicated")),
                                          ("rows_tiled", f.get("rows_tiled")), ("row_bounds", f.get("row_bounds"))]
    pairs += [(k, f.get(k)) for k in RF.fold_fields()]                            # data_form= feats_digest= feats_ranks_equal= (rowpair_feats)
    _rci = sys.modules.get("esmfold2_opt.rowchunk.install")                       # the route's row-chunking levers: rowchunk=<members bound|none> rowchunk_floor=<tokens>
    if _rci is not None:                                                          # rowchunk_rows=blocked|whole:below_floor (this fold's N vs the floor) and every member's counters so far (rc_<member>_<counter>=)
        pairs += list(_rci.fold_fields(int(f["N"])))
    return f"{_report.PREFIX} rowpair fold " + kv(*pairs)


def _r2(x):
    return None if x is None else round(float(x), 2)


# ================================================================================================================ seam IN: the loop on rows
# --- rowchunk: row-chunked recycle inject (lever injrows) -----------------------------------------
# The six statements this replaces each materialise a WHOLE row shard [B, R, N, c_z].  At N=6912,
# P=2 (R=3456, c_z=128) one fp32 shard is 11.39 GiB, the allocation an unchunked P=2 fold runs out
# of memory on at that size (the `a * z_loc + F.linear(...)` statement below).  Sharding splits that
# object ACROSS ranks; nothing bounded it WITHIN a rank.
#
# Every op here is row-local -- LayerNorm normalises over the channel axis, Linear is a per-row
# GEMM, and the residual and the lm_z add are elementwise -- so the values are the full-shard
# statement's.  The GEMM is split on M, so this is NOT bitwise against the single GEMM: the same
# caveat this module already makes at the top for the sharded line.
#
# Peak: one row block of transients instead of up to five whole row shards.
INJ_STATS = {"calls": 0, "chunked": 0, "blocks": 0, "rows_per_block": None, "floor_skips": 0}   # lever injrows' counters: calls = every inject; chunked = ran per row block; floor_skips = the whole-shard statements ran (off / below the floor / off the GPU)


def _inj_on_device(t) -> bool:                                        # the row-blocked inject serves CUDA tensors; tests rebind this to exercise it on CPU
    return bool(getattr(t, "is_cuda", False))
INJ = {"on": False, "mb": 512, "min_tokens": 1024}                # lever injrows: set by esmfold2_opt.rowchunk.install from EF2_ROWPAIR_INJECT_MB / EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS;
                                                                   # off until the installer turns it on (the whole-shard statements run); never read from the shell here


def _inject_rows_count(R, N, C, elt):
    per_row = max(1, N * C * elt)
    return max(1, min(R, int(float(INJ["mb"]) * (1 << 20)) // per_row))


def _inject_rows(self, z_loc, z_inject_pair, refined_lm_z, a, b_mat):
    """The recycle inject statements, issued per LOCAL ROW BLOCK.  Row axis is -3 ([B, R, N, c_z])."""
    import torch.nn.functional as F

    R, N, C = int(z_loc.shape[-3]), int(z_loc.shape[-2]), int(z_loc.shape[-1])
    INJ_STATS["calls"] += 1
    if not INJ["on"] or N < int(INJ["min_tokens"]) or not _inj_on_device(z_loc):
        INJ_STATS["floor_skips"] += 1
        if refined_lm_z is not None:
            z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)
        injected_pair = self.parcae_input_norm(z_inject_pair)
        return a * z_loc + F.linear(injected_pair.to(z_loc.dtype), b_mat)
    rows = _inject_rows_count(R, N, C, z_loc.element_size())
    INJ_STATS["chunked"] += 1; INJ_STATS["blocks"] += -(-R // rows); INJ_STATS["rows_per_block"] = rows
    # z_loc is owned by the loop -- the caller rebinds `z` to this function's return value and holds
    # no other reference -- so the residual is written in place, saving one more whole shard.
    out = z_loc
    for s in range(0, R, rows):
        e = min(s + rows, R)
        zi = z_inject_pair[..., s:e, :, :]
        if refined_lm_z is not None:
            zi = zi + refined_lm_z[..., s:e, :, :].to(zi.dtype)
        y = F.linear(self.parcae_input_norm(zi).to(z_loc.dtype), b_mat)
        del zi
        zb = out[..., s:e, :, :]
        zb.mul_(a)
        zb.add_(y)
        del zb
        del y
    return out


def _run_one_loop_sharded(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat, tok_mask, total_steps):
    """``ESMFold2Model._run_one_loop`` (modeling_esmfold2.py:754-848) on this rank's rows — the same statements in the same order. ``z``,
    ``z_init`` and ``lm_z`` ARRIVE as rows ``[B, R, N, C]`` (``_forward_rows``); ``pair_mask`` arrives as rows ``[B, R, N]``; the per-loop
    LM-pair dropout is rows of the stock Philox stream (``rowpair_heads.lm_dropout_rows``); the MSA branch (variant full_msa) runs the stock
    statements with ``model.msa_encoder`` rebound for row inputs by ``rowpair_msa``. Returns the SHARD ``[B, R, N, C]``."""
    if CTX.below_floor:                                                                    # the fold is below the sharding floor: the kept loop, whole
        return PATCHES.original(self, "_run_one_loop")(z=z, z_init=z_init, lm_z=lm_z, _msa_inputs=_msa_inputs, pair_mask=pair_mask, a=a, b_mat=b_mat,
                                                        tok_mask=tok_mask, total_steps=total_steps)
    import torch
    import torch.nn.functional as F
    from . import rowpair_heads as H
    CMN, MOD = _modules()
    lay = CTX.layout
    if lay is None or not _is_shard_of(z, lay):
        raise RowpairRefused(f"_run_one_loop: z {tuple(z.shape)} is not this rank's row shard (the loop is entered through _forward_rows under n_gpu>1)")
    if _msa_inputs is not None and STATE.get("msa_rows") is None:
        raise RowpairRefused("_run_one_loop: MSA inputs under n_gpu>1 without rowpair_msa (variant full_msa needs EF2-MSA's rowpair_msa.py)")
    N = lay.N
    f = CTX.fold
    mask_loc = pair_mask if int(pair_mask.shape[-2]) == lay.R and lay.R != N else None
    if mask_loc is None:
        from opt_core.mem.rowpair import shard as RS
        mask_loc = RS.shard_mask(pair_mask, lay)
    mask_loc = mask_loc.contiguous()
    if STATE.get("xl_loopfree"):                                              # the memory line's x2 loop is the statement re-issued here on rows: the add-on's own
        sys.modules[XL_MODULE]._STATS["loop_calls"] += 1                      # per-fold evidence counter (the kit's xl gate expects one loop per fold) counts this call
    if f is not None and f["cuda"]:
        f["alloc_gib_sharded"] = torch.cuda.memory_allocated(z.device) / 2 ** 30
    z_loc, z_init_loc, lm_loc = z, z_init, lm_z
    del z, z_init, lm_z
    # ---- the loop's own preamble (modeling_esmfold2.py:_run_one_loop)
    lm_cfg = self.config.lm_encoder
    _per_loop_lm_dropout = (lm_loc is not None and getattr(lm_cfg, "per_loop_lm_dropout", False) and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0)
    _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)
    own = bool(STATE.get("xl_own"))
    if (not _per_loop_lm_dropout) and lm_loc is not None and self.lm_encoder is not None and lm_loc.dtype != z_init_loc.dtype:
        lm_loc = lm_loc.to(z_init_loc.dtype)                                            # the XL fast line's LOOPFREE pre-cast (x2): once, not per loop, when the dropout is off
    for _step in range(total_steps):
        if _per_loop_lm_dropout:
            lm_z_i, rng_mode = H.lm_dropout_rows(lay, lm_loc, _lm_dropout_p, N)
            if f is not None:
                f["lm_dropout"] = rng_mode
        else:
            lm_z_i = lm_loc
        refined_lm_z = None
        if lm_z_i is not None and self.lm_encoder is not None:
            xin = [lm_z_i.to(z_init_loc.dtype)]                                      # 1-element holder: under XL OWN the trunk pops it and frees block inputs early
            if lm_z_i is not lm_loc:
                lm_z_i = None
            arg = xin if own else xin[0]
            with _sharded(lay, mask_loc):
                refined_lm_z = _timed("lm_encoder", lambda: self.lm_encoder(arg, pair_attention_mask=mask_loc), torch)
            del xin, arg
        z_inject_pair = z_init_loc
        if lm_z_i is not None and self.lm_encoder is None:
            z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)
        if self.msa_encoder is not None and _msa_inputs is not None:               # variant full_msa (stock statements; msa_encoder rebound for rows by rowpair_msa)
            msa_i, mask_i, hd_i, dv_i = CMN.maybe_subsample_msa(_msa_inputs["msa"], _msa_inputs["msa_attention_mask"], _msa_inputs["has_deletion"],
                                                                _msa_inputs["deletion_value"], max_depth=_msa_inputs["max_depth"], enabled=_msa_inputs["subsample_enabled"])
            B_msa, M, L_msa = msa_i.shape
            msa_oh = F.one_hot(msa_i.permute(0, 2, 1).long(), num_classes=MOD.NUM_RES_TYPES).float()
            msa_attn = (mask_i.permute(0, 2, 1).float() if mask_i is not None else tok_mask[:, :, None].expand(-1, -1, M).float())
            msa_oh = msa_oh * msa_attn.unsqueeze(-1)
            hd = (hd_i.permute(0, 2, 1).float() if hd_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
            dv = (dv_i.permute(0, 2, 1).float() if dv_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
            with _sharded(lay, mask_loc):
                msa_pair = _timed("msa_encoder", lambda: self.msa_encoder(x_pair=z_inject_pair, x_inputs=_msa_inputs["x_inputs"], msa_oh=msa_oh, has_deletion=hd,
                                                                         deletion_value=dv, msa_attention_mask=msa_attn), torch).to(z_inject_pair.dtype)
            z_inject_pair = (msa_pair if self.config.msa_encoder_overwrite else (z_inject_pair + msa_pair))
            del msa_pair, msa_oh, msa_attn, hd, dv
        z_loc = _inject_rows(self, z_loc, z_inject_pair, refined_lm_z, a, b_mat)
        if f is not None:
            f["reached_inject"] = int(f.get("reached_inject", 0)) + 1                # the recycle inject ran on this rank this fold (lever injrows' reachability evidence)
        del lm_z_i, refined_lm_z, z_inject_pair
        zin = [z_loc]
        del z_loc
        arg = zin if own else zin[0]
        _release_cached(torch, mask_loc.device if hasattr(mask_loc, "device") else arg.device, "loop")
        with _sharded(lay, mask_loc):
            z_loc = _timed("folding_trunk", lambda: self.folding_trunk(arg, pair_attention_mask=mask_loc), torch)
        del zin, arg
    CTX.mask_loc, CTX.pending_coda = mask_loc, True
    return z_loc


# ================================================================================================================ seam OUT / head trunks
def _guard_weights_replicated(model) -> None:
    """One float64 checksum of every parameter and buffer of the model, proven identical on every rank (``opt_core trunk.guard_replicated``:
    a mismatch raises RowpairRefused naming ``esmfold2.weights`` and the per-rank sums)."""
    import torch
    from opt_core.mem.rowpair import trunk as RTR
    dev = next(model.parameters()).device
    with torch.no_grad():
        acc = torch.zeros((1,), dtype=torch.float64, device=dev)
        for t in list(model.parameters()) + list(model.buffers()):
            if t.device == dev and (t.is_floating_point() or t.dtype in (torch.int64, torch.int32, torch.bool, torch.uint8)):
                acc += t.detach().sum(dtype=torch.float64)
    RTR.guard_replicated(acc, "esmfold2.weights")


def _empty_cache_wanted(phase: str) -> bool:
    """The ``EF2_ROWPAIR_EMPTY_CACHE`` word: is ``phase`` one of the boundaries that release the allocator's cache (default: all)."""
    word = (os.environ.get(ENV_EMPTY_CACHE, "") or EMPTY_CACHE_DEFAULT).strip().lower()
    if word in ("all", "1", "on"):
        return True
    if word in ("none", "0", "off"):
        return False
    return phase in {w.strip() for w in word.split(",") if w.strip()}


def _release_cached(torch, device, phase: str) -> None:
    """``torch.cuda.empty_cache()`` at a phase boundary of the rows forward (phases born / loop / heads /
    confidence / tail): the phase's freed transients (block-sized, sizes specific to the phase) are returned to the driver so the next
    phase's blocks map fresh instead of growing the reservation beside them. Census: ``empty_cache_phases=`` and the per-phase seconds."""
    if getattr(device, "type", "") != "cuda" or not _empty_cache_wanted(phase):
        return
    t0 = time.time(); torch.cuda.synchronize(device); torch.cuda.empty_cache()
    f = CTX.fold
    if f is not None:
        f["empty_cache_phases"] = f.get("empty_cache_phases", "") + ("," if f.get("empty_cache_phases") else "") + phase
        f["empty_cache_phase_s"] = round(float(f.get("empty_cache_phase_s", 0.0)) + time.time() - t0, 3)


def _is_shard_of(pair, lay) -> bool:
    return pair.dim() == 4 and int(pair.shape[-3]) == lay.R and int(pair.shape[-2]) == lay.N and lay.R != lay.N


def _coda_forward(inner: Callable):
    """``parcae_coda.forward`` under sharding: the loop returned this rank's rows (through the row-local ``parcae_readout``); the coda's blocks
    run on them and the ROWS are returned — nothing is gathered (``_forward_rows`` feeds the distogram, the diffusion conditioning and the
    confidence head rows)."""
    def forward(pair, pair_attention_mask=None):
        import torch
        lay = CTX.layout
        if not (CTX.pending_coda and lay is not None and _is_shard_of(pair, lay)):
            return inner(pair, pair_attention_mask=pair_attention_mask)
        mask_loc = CTX.mask_loc
        CTX.pending_coda = False
        with _sharded(lay, mask_loc):
            out_loc = _timed("parcae_coda", lambda: inner(pair, pair_attention_mask=mask_loc), torch)
        f = CTX.fold
        if f is not None:
            if f["cuda"]:
                _memfields(torch, out_loc.device, "after_coda", f)                   # the peaks so far (process-wide: loop entry .. coda included)
                t_ec = time.time(); torch.cuda.empty_cache(); f["empty_cache_tail_s"] = time.time() - t_ec
                torch.cuda.reset_peak_memory_stats(out_loc.device)
            census = dict(EV.rows_census(lay))                                      # collective: every rank's (r0, r1)
            f["rows_tiled"], f["row_bounds"] = census["rows_tiled"], census["row_bounds"]
            f["coda_gather"] = "none"
        _guard_rng_state(out_loc.device, torch, f)                                  # GUARD (always on): the structure module's noise is drawn next
        return out_loc
    forward.__wrapped_forward__ = inner
    return forward


def _guard_features(feats: dict, torch, f: Optional[dict]) -> None:
    """The replicated-input guard at the seam IN, DET-SCOPED: the loop's inputs (z, z_init, lm_z, mask, parcae a/b) are RECOMPUTED on every rank, not broadcast — under
    the kit's deterministic recipe (``--det >= 1``: torch's deterministic algorithms on) they are bitwise identical across ranks and a mismatch
    is refused by name (``features@S1``); outside the recipe the fast line's kernels may differ in the last bits run to run, so each rank's row
    block is conditioned on its own recomputation (the tier-2 statement vs the single-GPU line covers it) and the comparison is RECORDED
    (``features_replicated=`` in the fold line), never refused."""
    from opt_core.mem.rowpair import bcast as RB
    feats = {k: v for k, v in feats.items() if v is not None}
    if torch.are_deterministic_algorithms_enabled():
        RB.assert_replicated(feats, "features@S1")
        if f is not None:
            f.setdefault("guards", []).append("features@S1")
            f["features_replicated"] = True
        return
    local = RB.tensordict_checksum(feats)
    from opt_core.mem.rowpair import dist as RD
    allv = RD.all_gather_obj(local) if hasattr(RD, "all_gather_obj") else _all_gather_object(local, torch)
    same = all(v == allv[0] for v in allv)
    if f is not None:
        f["features_replicated"] = bool(same)
        if not same:
            f["features_differ"] = ",".join(sorted(k for k in local if any(v.get(k) != local[k] for v in allv)))


def _all_gather_object(obj, torch):
    import torch.distributed as tdist
    if not (tdist.is_available() and tdist.is_initialized()):
        return [obj]
    allv = [None] * tdist.get_world_size()
    tdist.all_gather_object(allv, obj)
    return allv


def _guard_rng_state(device, torch, f: Optional[dict]) -> None:
    """The replicated-state production guard at the seam OUT: the default generators every replicated statement after the trunk draws from (the diffusion
    sampler's initial noise and per-step noise; CPU and this rank's CUDA device) hold bitwise-identical STATES on all ranks — a stronger
    statement than checking one noise tensor, and engine-free. A mismatch raises ``RowpairRefused('replicated tensor ... differs across ranks')``."""
    from opt_core.mem.rowpair import dist as RD
    RD.allreduce_checksum(torch.get_rng_state(), "rng_state@structure_module(cpu)")
    if device.type == "cuda":
        RD.allreduce_checksum(torch.cuda.get_rng_state(device), "rng_state@structure_module(cuda)")
    if f is not None:
        f.setdefault("guards", []).append("rng_state@structure_module")


def _conf_head_rows(head, trunk_eager: Callable):
    """``confidence_head.forward`` under sharding (``rowpair_heads.confidence_forward_rows``): the structure sample ``x_pred`` (the one
    stochastic input of the head) is broadcast from rank 0 first (a checksum records whether the ranks' own samples already agreed:
    ``x_pred_replicated``), the head's pair statements run on this rank's rows with its 4-block trunk through the presharded pair-stack
    dispatchers, and the fold's evidence line is printed at exit."""
    def run_trunk(trunk, pair_rows, mask_rows):
        import torch
        lay = CTX.layout
        with _sharded(lay, mask_rows.contiguous()):
            return _timed("confidence_trunk", lambda: trunk_eager(pair_rows, pair_attention_mask=mask_rows), torch)

    def forward(*args, **kwargs):
        import torch
        from opt_core.mem.rowpair import bcast as RB
        from opt_core.mem.rowpair import dist as RD
        from . import rowpair_heads as H
        if CTX.below_floor:                                                                # the fold is below the sharding floor: the kept head, whole
            return PATCHES.original(head, "forward")(*args, **kwargs)
        if args:
            raise RowpairRefused("confidence_head under n_gpu>1: keyword arguments only (the stock forward calls it by keyword)")
        x_pred = kwargs["x_pred"]
        same = True
        try:
            RD.allreduce_checksum(x_pred.contiguous(), "x_pred", raise_on_mismatch=True)
        except Exception as e:  # noqa: BLE001  (recorded, then repaired by the broadcast; a GPU out-of-memory error propagates)
            from opt_core.oom import is_oom
            if is_oom(e): raise
            same = False
            STATE["x_pred_mismatch"] = int(STATE.get("x_pred_mismatch", 0)) + 1
        x0 = RB.broadcast_tensordict({"x_pred": x_pred.contiguous()}, src=0)["x_pred"]          # SAMPLER EXIT: rank 0's coordinates replace every rank's
        spread = (x_pred.float() - x0.float()).abs().reshape(-1).max().reshape(1) if x_pred.numel() else torch.zeros(1, device=x0.device)
        RD.allreduce_(spread, "max")                                                            # the cross-rank spread max|x_rank - x_rank0| (Angstrom)
        spread_a = float(spread.item())
        if CTX.fold is not None:
            CTX.fold["diff_rank_spread_A"] = round(spread_a, 6)
            CTX.fold["diff_noise"] = "guard" if H.diffusion_sync_mode(torch) == "guard" else "bcast_rank0_state"
        if spread_a > DIFF_RANK_SPREAD_MAX_A:
            raise RowpairRefused(f"sampler exit: cross-rank coordinate spread {spread_a:.3f} A > {DIFF_RANK_SPREAD_MAX_A} A (diff_rank_spread_A) — the ranks' diffusion states diverged")
        kwargs["x_pred"] = x0
        atom = sys.modules.get("ef2_atom")                                                      # the sampler returned: the atom-path static buffers
        if atom is not None and hasattr(atom, "release_statics_if_eager"):                        # (growing with N) are dead in this graph-free route
            released = float(atom.release_statics_if_eager("confidence"))
            if CTX.fold is not None:
                CTX.fold["atom_statics_released_gib"] = round(released, 3)
        _release_cached(torch, x_pred.device, "confidence")
        out = H.confidence_forward_rows(head, CTX.layout, run_trunk, **kwargs)
        f = CTX.fold
        if f is not None:
            f["x_pred_replicated"] = bool(same)
            if f["cuda"]:
                _memfields(torch, x_pred.device, "after_conf", f)
            _resolve_events(f, torch)
            print(_fold_line(f), file=sys.stderr, flush=True)
        return out
    forward.__wrapped_forward__ = head.forward
    return forward


def _dm_forward(dm):
    """``structure_head.diffusion_module.forward`` under sharding = ``rowpair_heads.dm_forward_rows`` (z_cond rows once per roll-out, local-query
    token transformer); the schedule (``opt_core.mem.rowpair.diffusion.DiffusionSchedule``) is decided once per roll-out and printed in the
    fold line. Whatever forward the fast line had installed on the instance (the MK batched sampler's) is the kept original: NAMED
    ``dm_forward_replaced`` in the install line, restored by uninstall."""
    box = _DM_BOX

    def forward(*args, **kwargs):
        from . import rowpair_heads as H
        if CTX.below_floor:                                                                # the fold is below the sharding floor: the kept diffusion forward
            return PATCHES.original(dm, "forward")(*args, **kwargs)
        route = box.get("route")
        if route is None:                                                                  # once per fold, at the sampler's first denoiser call: whole (the fused step over the
            route = _sampler_route(dm, box, args, kwargs)                                 # gathered conditioned pair) when it fits under the trunk's peak on every rank, else rows
        if route == "whole":
            return H.dm_forward_whole(dm, PATCHES.original(dm, "forward"), CTX.layout, box, sys.modules.get("ef2_dit"), *args, **kwargs)
        DIT = sys.modules.get("ef2_dit")
        if DIT is not None and box.get("dit_ready") == "ready":                           # the fused step is installed and this fold runs the rows form: each denoiser
            DIT.STATS["dit_fallback"] += 1; DIT.STATS["dit_fallback_rows_route_n_gpu"] += 1   # step is a step-aside of lever dit BY NAME in its own counters (the run's record:
        return H.dm_forward_rows(dm, CTX.layout, box, *args, **kwargs)                     # dit `gated`, named — not an unreached lever), like ro's count at S > 1
    forward.__wrapped_forward__ = dm.forward
    return forward


def _sampler_route(dm, box, args, kwargs) -> str:
    """Decide the sampler's token path for this fold (:func:`rowpair_heads.sampler_route_decide`; ``EF2_ROWPAIR_SAMPLER``) and write its facts
    into the fold record. The whole route needs the one-GPU line's fused step (``ef2_dit`` with ``dit`` on) as the kept diffusion forward and
    the sampler's inference cache; anything else is the rows route, named."""
    import torch
    from . import rowpair_heads as H
    DIT = sys.modules.get("ef2_dit")
    inner = PATCHES.original(dm, "forward")
    inner_fn = getattr(inner, "__func__", inner)
    if DIT is None:
        ready = "dit_absent"
    elif inner_fn is not getattr(DIT, "_dm_forward_dit", None) or getattr(dm, "_dit", None) is None:
        ready = "dit_not_installed"
    elif not (getattr(DIT, "_CFG", {}) or {}).get("dit"):
        ready = "dit_off"
    else:
        ready = "ready"
    cache = kwargs.get("inference_cache")
    x_noisy = kwargs.get("x_noisy", args[0] if args else None)
    S = int(kwargs.get("num_diffusion_samples", 1) or 1)
    word = (os.environ.get(ENV_SAMPLER, "") or "auto").strip().lower()
    if word not in H.SAMPLER_ROUTE_WORDS:
        raise RowpairRefused(f"{ENV_SAMPLER}={word!r}: one of {'|'.join(H.SAMPLER_ROUTE_WORDS)}")
    route, facts = H.sampler_route_decide(dm, CTX.layout, S=S, word=word, dit_ready=ready, has_cache=cache is not None,
                                          device=(x_noisy.device if x_noisy is not None else None), torch=torch)
    box["route"] = route
    box["dit_ready"] = ready
    box["dm"] = dm
    f = CTX.fold
    if f is not None:
        f.update(facts)
    return route


_DM_BOX: Dict[str, object] = {}                                        # per roll-out diffusion schedule + pair-bias cache (rowpair_heads.dm_forward_rows)


def _release_lm_cache() -> None:
    """Under ``n_gpu > 1`` the exact LM-output cache (the fast line's ``ec``: the LM hidden states of the last inputs, kept on the device
    for a repeated fold of the same input) is emptied once this fold's LM pair rows exist: on the memory line across P ranks every rank would
    otherwise hold the replicated ``[N+2, layers+1, d]`` states of up to two inputs for the whole fold. A repeated fold of the same input
    recomputes its LM pass (the cache's clears counter names it)."""
    if int(STATE.get("P") or 1) <= 1:
        return
    opt = sys.modules.get("ef2_opt")
    cache = getattr(opt, "_ESMC_CACHE", None) if opt is not None else None
    if cache is not None and getattr(cache, "d", None):
        cache.clear()


def _release_sampler_rows() -> None:
    """The diffusion roll-out's per-block pair-bias rows (``opt_core.mem.rowpair.diffusion.PairBiasCache`` in ``_DM_BOX``) are dropped when
    the roll-out has returned; the box itself is re-decided per fold. (ef2_dit's eager roll drops its own references to the fold's rows and
    inference cache when ``sample()`` returns — ``roll_eager_refs_dropped`` — so nothing of the sampler outlives this point on a rank.)"""
    cache = _DM_BOX.get("bias_cache")
    if cache is not None:
        cache.clear()
    if _DM_BOX.get("route") == "whole":                                                   # the whole route: the fused step's per-shape hoist (the 12 [1, H, N, N] pair biases) is
        dm, DIT = _DM_BOX.get("dm"), sys.modules.get("ef2_dit")                           # the one N×N-class state left on the rank — dropped here so the confidence head and the
        st = getattr(dm, "_dit", None)                                                     # next item's trunk meet the same free memory as on the rows route (re-hoisted at the
        if st is not None and getattr(st, "hoist", None) is not None:                      # next fold's step 0; no graph reads it at n_gpu > 1)
            st.hoist.clear(); st.cur = None
        f = CTX.fold
        if f is not None:
            f["sampler_whole_steps"] = int(_DM_BOX.get("whole_steps", 0)); f["sampler_z_whole_gib"] = _DM_BOX.get("z_whole_gib")
            if DIT is not None:
                f["sampler_dit_steps"] = int(DIT.STATS.get("dit_steps", 0))

def _forward_rows(self, token_index, residue_index, asym_id, sym_id, entity_id, mol_type, res_type, token_bonds, token_attention_mask, ref_pos,
                  ref_element, ref_charge, ref_atom_name_chars, ref_space_uid, atom_attention_mask, atom_to_token, distogram_atom_idx,
                  deletion_mean=None, msa=None, has_deletion=None, deletion_value=None, msa_attention_mask=None, input_ids=None,
                  lm_hidden_states=None, num_loops=None, num_diffusion_samples=None, num_sampling_steps=None, lm_mask_pct=None,
                  msa_max_depth: int = 1024, msa_column_mask_rate: float = 0.1, msa_subsample_at_inference: bool = True, **kwargs):
    """``ESMFold2Model.forward`` (modeling_esmfold2.py:851-1082) under n_gpu>1: the SAME statements in the same order with every pair tensor
    BORN as this rank's rows (``rowpair_heads``: z_init = outer-sum rows + relative-position rows + token-bond rows; lm_z rows; the pair
    state's RNG draw as rows of the stock stream; pair mask rows), the loop / readout / coda on rows, the distogram on rows (zᵀ rows by one
    all-to-all), the structure module with the diffusion module rebound for pair rows, and the confidence head on rows. Token-level and
    atom-level statements are verbatim (replicated by design). An input below the sharding floor takes :func:`_forward_below_floor`."""
    _call = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}; _call.update(kwargs)      # the call as made, for the below-floor route
    if RF.STATE["form"] == RF.FORM_PER_RANK:                                               # no builder binding in this process (every rank featurised): the digest of THIS rank's model-input
        RF.census_per_rank(_call)                                                          # tensors gathered with every other rank's — the family's `[feats] … feats_ranks_equal=` line (names a disagreement; refuses nothing).
    #   Under the builder binding (data_form=rank0_bcast) these tensors are rank 0's, digested and agreed on at the hand-over (rowpair_feats.featurise_rank0).
    import torch
    import torch.nn.functional as F
    from . import rowpair_heads as H
    tok_mask = token_attention_mask
    atm_mask = atom_attention_mask
    disto_idx = distogram_atom_idx
    n_loops: int = num_loops if num_loops is not None else self.config.num_loops
    n_samples: int = (num_diffusion_samples if num_diffusion_samples is not None else self.config.num_diffusion_samples)
    total_steps = max(1, n_loops + 1)
    if int(tok_mask.shape[0]) != 1:
        raise RowpairRefused(f"forward under n_gpu>1: batch {int(tok_mask.shape[0])} != 1 — one item per fold call is the row-sharded line's domain at n_gpu={STATE.get('P')}: "
                             "fold the items separately (one input per call, EF2_MS_MAX_BATCH=1 / one seed per fold) or use --n_gpu 1")
    N = int(tok_mask.shape[1])
    lay = _layout_or_none(N)
    if lay is None:                                                                        # below the sharding floor: folded whole (named), not refused
        return _forward_below_floor(self, N, _call)
    CMN, MOD = _modules()
    f = _new_fold(lay, torch, ref_pos.device)
    CTX.layout, CTX.fold = lay, f
    _DM_BOX.clear()
    H.mark("trunk_entry")
    with torch.inference_mode():
        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=MOD.NUM_RES_TYPES).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()
        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=MOD.NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
                del mask_f, valid_seq_count
            else:
                profile = msa_oh_profile.mean(dim=1)
            del msa_oh_profile                                                             # the one-hot MSA [1, M, N, 33] fp32 (132·M bytes per token, replicated on every rank) is read
        else:                                                                              # only here: released now instead of at the end of the fold (upstream keeps the local alive
            profile = res_type_oh                                                          # through trunk, sampler and heads; on rows it sat inside every rank's peak)
        if deletion_mean is None:
            deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)
        ref_element_oh = F.one_hot(ref_element.long(), num_classes=CMN.MAX_ATOMIC_NUMBER).float()
        ref_atom_name_chars_oh = F.one_hot(ref_atom_name_chars.long(), num_classes=CMN.CHAR_VOCAB_SIZE).float()
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()
        use_amp = ref_pos.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            x_inputs = self.inputs_embedder(aatype=res_type_oh, profile=profile.float(), deletion_mean=deletion_mean.float(), ref_pos=ref_pos,
                                            atom_attention_mask=atm_mask, ref_space_uid=ref_space_uid, ref_charge=ref_charge, ref_element=ref_element_oh,
                                            ref_atom_name_chars=ref_atom_name_chars_oh, atom_to_token=atom_to_token)
            _guard_features({"x_inputs": x_inputs}, torch, f)                       # the replicated token single every row statement reads
            z_init = H.z_init_rows(self, lay, x_inputs)                               # rows [B, R, N, c_z]
            relative_position_encoding = H.in_row_blocks(lambda span: H.relpos_rows(self.rel_pos, span, residue_index=residue_index, asym_id=asym_id, sym_id=sym_id,
                                                       entity_id=entity_id, token_index=token_index), lay)
            z_init += H.token_bonds_rows(self, lay, token_bonds)                        # accumulated IN PLACE (the XL init-lean discipline on rows): one
            z_init += relative_position_encoding                                        #   shard-sized sum alive, never four; relpos rows stay for the diffusion conditioning
            if (lm_hidden_states is None and input_ids is not None and self._esmc is not None):
                lm_hidden_states = self._compute_lm_hidden_states(input_ids, asym_id, residue_index, mol_type, tok_mask,
                                                                  lm_mask_pct=(lm_mask_pct if lm_mask_pct is not None else self.config.lm_mask_pct))
            lm_z = None
            if lm_hidden_states is not None:
                lm_z = H.lm_pair_rows(self.language_model, lay, lm_hidden_states.detach())
            del lm_hidden_states
            _release_lm_cache()                                                        # the LM states are pair rows now: the exact cache's device copy goes (n_gpu > 1)
            pair_mask = H.pair_mask_rows(lay, tok_mask)                               # rows [B, R, N]; the coda / conf trunks take rows too
            z, rng_mode = H.init_pair_state_rows(lay, z_init)
            f["pair_rng"] = rng_mode
            a, b = self._discretized_dynamics()
            a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
            b_mat = b.to(device=z.device, dtype=z.dtype)
            _msa_inputs = None
            if self.msa_encoder is not None and msa is not None:
                msa_attention_mask = CMN.maybe_apply_msa_column_masking(msa_attention_mask, rate=msa_column_mask_rate)
                _msa_inputs = dict(msa=msa, msa_attention_mask=msa_attention_mask, has_deletion=has_deletion, deletion_value=deletion_value,
                                   x_inputs=x_inputs, max_depth=msa_max_depth, subsample_enabled=msa_subsample_at_inference)
            H.mark("after_msa" if _msa_inputs is not None else "after_template")
            _release_cached(torch, z.device, "born")
            z = self._run_one_loop(z=z, z_init=z_init, lm_z=lm_z, _msa_inputs=_msa_inputs, pair_mask=pair_mask, a=a, b_mat=b_mat, tok_mask=tok_mask,
                                   total_steps=total_steps)
            del z_init, lm_z, _msa_inputs, a, b_mat
            z = self.parcae_readout(z)
            z = self.parcae_coda(z, pair_attention_mask=pair_mask)
            z = z.float()
        H.mark("after_pairstack"); H.mark("no_gather")
        distogram_logits = _timed("distogram", lambda: H.distogram_rows(self, lay, z), torch)     # rows [B, R, N, bins]
        H.mark("distogram")
        _release_cached(torch, z.device, "heads")
        structure_output = self.structure_head.sample(
            z_trunk=z, s_inputs=x_inputs, s_trunk=None, relative_position_encoding=relative_position_encoding, ref_pos=ref_pos, ref_charge=ref_charge,
            ref_mask=atm_mask, ref_element=ref_element_oh, ref_atom_name_chars=ref_atom_name_chars_oh, ref_space_uid=ref_space_uid, tok_idx=atom_to_token,
            asym_id=asym_id, residue_index=residue_index, entity_id=entity_id, token_index=token_index, sym_id=sym_id, token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples, num_sampling_steps=num_sampling_steps, return_atom_repr=False, denoising_early_exit_rmsd=None)
        f["reached_sample"] = 1                                                       # the sampler returned on this rank this fold (lever biasfree's reachability evidence)
        _release_sampler_rows()                                                       # the roll-out's pair-bias row cache is dead past this point (the confidence head is next)
        sample_coords = structure_output["sample_atom_coords"]
        assert sample_coords is not None
        output = {"distogram_logits": distogram_logits}
        output["sample_atom_coords"] = sample_coords
        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(), z=z.detach().float(), x_pred=sample_coords.detach(), distogram_atom_idx=disto_idx, token_attention_mask=tok_mask,
            atom_to_token=atom_to_token, atom_attention_mask=atm_mask, asym_id=asym_id, mol_type=mol_type, num_diffusion_samples=n_samples,
            relative_position_encoding=relative_position_encoding.detach(), token_bonds_encoding=H.token_bonds_rows(self, lay, token_bonds).detach())   # the
        #   token-bond rows are re-issued here (a Linear over [R, N, 1]) instead of held alive through the trunk and the diffusion
        f["reached_confidence"] = 1                                                   # the confidence head returned on this rank this fold (the confidence-path levers' reachability evidence)
        output.update(confidence_output)
        output["atom_pad_mask"] = (atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask)
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        sched = _DM_BOX.get("sched")
        f["diffusion"] = dict(sched.fields()) if sched is not None else {"diff_schedule": "none"}
        f["distogram_logits"] = "rows"
        output["_rowpair"] = {"rows": (lay.r0, lay.r1), "N": N, "P": lay.P, "distogram_logits": "rows", "pair_rng": f.get("pair_rng")}
    return output


def fold_evidence_flush() -> Optional[str]:
    """Print (and return) the running fold's evidence line if the confidence head did not (a model without one); tests call it."""
    f = CTX.fold
    if f is None:
        return None
    try:
        import torch
        _resolve_events(f, torch)
    except Exception:  # noqa: BLE001
        pass
    line = _fold_line(f)
    print(line, file=sys.stderr, flush=True)
    return line


# ================================================================================================================ triangle multiplication on rows
ENV_TRIMUL_CHUNK = "EF2_ROWPAIR_TRIMUL_CHUNK"                          # rows per a-projection chunk of the triangle multiplication (opt_core trimul_update_ inplace_chunk; default 256)
ENV_TRANSITION_ROWS = "EF2_ROWPAIR_TRANSITION_ROWS"                    # rows per pair-transition block on a shard (default TRANSITION_ROWS_DEFAULT; the 4x hidden is [rows, N, 4C], never [R, N, 4C]); the MSA encoder's blocks follow it (rowpair_msa.pair_transition_rows)
ENV_EMPTY_CACHE = "EF2_ROWPAIR_EMPTY_CACHE"                            # the phase boundaries of the rows forward at which torch.cuda.empty_cache() runs: a comma list over
#   born,loop,heads,confidence,tail | all | none (default EMPTY_CACHE_DEFAULT = all) — `loop` runs once per Parcae loop (a device synchronize + the allocator's
#   release; the next loop re-maps its transients), the others once per fold. n_gpu > 1 only.
EMPTY_CACHE_PHASES = ("born", "loop", "heads", "confidence", "tail")
EMPTY_CACHE_DEFAULT = "all"                                            # every boundary (the memory line's choice: the per-loop release keeps the device reservation lowest);
#   `born,heads,confidence,tail` (no per-loop release) is the speed word: it skips the per-loop synchronize + release for a faster trunk at an equal
#   allocator peak and a higher device reservation, for whoever has the headroom
#   EF2_ROWPAIR_DIT_ROWS (rowpair_heads.ENV_DIT_ROWS): the rows route's query-block attention core — big (default: the shared core's rows face,
#   opt_core.kernels.apb.pair_bias_attention_rows, served row apb_attn) | a row word | engine (the materialised statement); fold line dit_rows=.
ENV_SAMPLER = "EF2_ROWPAIR_SAMPLER"                                    # the diffusion sampler's token path at n_gpu > 1: auto (default) | whole | rows. `whole` = the one-GPU line's
#   fused step (ef2_dit) on every rank over the conditioned pair gathered whole in bf16 from the rows — taken by `auto` when that fits inside memory the
#   trunk already peaked at on every rank and inside free device memory with margin (rowpair_heads.sampler_route_decide: it can neither raise the
#   item's per-rank peak nor run out of memory); `rows` = the row-sharded token transformer (local query rows, bias rows, one all-gather per block).
#   The fold line names which ran (sampler_route=whole_replicated|sharded_rows sampler_route_why=…).
ENV_TRIMUL_MIN_TOKENS = "EF2_ROWPAIR_TRIMUL_MIN_TOKENS"                # pair size N from which a row block's triangle-multiplication statements run as the core's fused
#   row-block kernels (opt_core.mem.rowpair.trimul_fused, lever F2.trimul_rows: fpf_trimul_v4 K1 into the GEMM plane + K3 tile epilogue; the torch
#   statements below stay the named fallback of every unit the provider declines); unset = the core's DEFAULT_MIN_TOKENS; ROWPAIR_TRIMUL_KERNELS=torch
#   (the core's engineering switch) declines every unit by request. n_gpu > 1 only: nothing here is reached at P = 1.
_FUSED_TRIMUL: Dict[object, object] = {}                               # (block key) -> the provider-wrapped TriMulFns (weights packed once per block and direction)
TRIMUL_PROVIDER: Dict[str, object] = {"bound": None, "why": None}      # census: provider module bound (True) / the statements only (False, why)
TRANSITION_ROWS_DEFAULT = 256
TRIMUL_STATS: Dict[str, object] = {}                                     # the fold's opt_core ContractStats per direction (schedule census: trimul_* fields)


def _trimul_fns(*, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, eps: float, key=None):
    """``TriangleMultiplicativeBlock.forward``'s statements (modeling_esmfold2_common) as opt_core ``TriMulFns`` — the SAME statements the
    stock module and the fused line's ``_fused_trimul_with_residual`` compute, issued per row block by the family driver
    (``opt_core.mem.rowpair.trimul.trimul_update_``: a from this rank's rows, b sub-blocks projected just in time and ring-passed, the
    output written per (row block, column block) with the gate read from the original values). ``p_in_weight`` / ``g_in_weight`` =
    ``TriangleMultiplicativeBlock.split_kernel_weights()`` (signal rows a|b, gate-logit rows a|b). Operand dtype = the projection's dtype
    (bf16 under the model's autocast: the fused line's plane dtype; census ``trimul_planes``); LayerNorms run as the stock F.layer_norm."""
    import torch
    import torch.nn.functional as F
    C_h = int(p_in_weight.shape[0]) // 2
    C_z = int(norm_in_weight.shape[0])

    def proj(zb, mb, is_a):                                              # routed = signal * sigmoid(gate_logits) * visibility; a = first C_h, b = second C_h
        o = 0 if is_a else C_h
        n = F.layer_norm(zb, (C_z,), norm_in_weight, norm_in_bias, eps)
        p = F.linear(n, p_in_weight[o:o + C_h]) * torch.sigmoid(F.linear(n, g_in_weight[o:o + C_h]))
        return (p * mb.to(p.dtype)).to(zb.dtype)                          # the pair schedule's dtype (z's) BY CONSTRUCTION: under autocast F.linear yields bf16 —
        #   a ring sub-block whose dtype differed from z's would post unequal byte counts across ranks

    def out(x):                                                          # mixed = proj_emit(norm_mix(contracted)), in z's dtype
        return F.linear(F.layer_norm(x, (C_h,), norm_out_weight, norm_out_bias, eps), p_out_weight).to(x.dtype)

    def gate(zb):                                                        # output_gate = sigmoid(proj_gate(norm_start(pair))), in z's dtype
        return torch.sigmoid(F.linear(F.layer_norm(zb, (C_z,), norm_in_weight, norm_in_bias, eps), g_out_weight)).to(zb.dtype)

    from opt_core.mem.rowpair import trimul as RT
    stock = RT.TriMulFns(proj, out, gate, C_h)
    return _with_trimul_provider(stock, key, eps, C_h, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight, g_in_weight=g_in_weight,
                                 norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight)


def _with_trimul_provider(stock, key, eps: float, C_h: int, *, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias,
                          p_out_weight, g_out_weight):
    """The statements ``stock`` wrapped by the core's fused row-block provider (:mod:`opt_core.mem.rowpair.trimul_fused`,
    ``fused_trimul_fns``: the driver's optional ``proj_into`` / ``tile_epilogue`` / ``operand_dtype`` hooks served by the carried
    fpf_trimul_v4 kernels per row block; every unit it declines — below its size gate, an unsupported width / dtype, the
    ``ROWPAIR_TRIMUL_KERNELS=torch`` request — runs ``stock``, counted by reason on the core's ``LEVER name=F2.trimul_rows`` line). The
    weights are handed over in the core's ``opt_core.trimul_weights`` vocabulary (a = signal/gate rows ``0:C_h``, b = rows ``C_h:2C_h`` of
    ``split_kernel_weights()``; no biases). ``key`` (the caller's stable identity of the block and direction) caches the wrapped callables so
    the provider packs a block's weights once per process; ``None`` wraps per call. A core without the provider module = ``stock`` (census
    ``trimul_provider=absent``)."""
    if TRIMUL_PROVIDER["bound"] is False:
        return stock
    hit = _FUSED_TRIMUL.get(key) if key is not None else None
    if hit is not None:
        return hit
    try:
        from opt_core.mem.rowpair import trimul_fused as RTF
    except ImportError as e:                                             # an older core: the statements, named once in the fold census
        TRIMUL_PROVIDER.update(bound=False, why=f"absent:{type(e).__name__}")
        return stock
    mt = (os.environ.get(ENV_TRIMUL_MIN_TOKENS, "") or "").strip()
    if not mt:                                                           # no request: served from the sharding floor up on cc >= 9.0 (the fused row-block kernels are faster than the
        try:                                                             #   statements at every size there), the core's own gate (DEFAULT_MIN_TOKENS) on other classes
            import torch                                                 #
            if p_in_weight.is_cuda and torch.cuda.get_device_capability(p_in_weight.device)[0] >= 9:
                mt = "0"
        except Exception:
            mt = ""
    weights = dict(ln_in_w=norm_in_weight, ln_in_b=norm_in_bias, w_ap=p_in_weight[:C_h], w_bp=p_in_weight[C_h:2 * C_h], w_ag=g_in_weight[:C_h],
                   w_bg=g_in_weight[C_h:2 * C_h], ln_out_w=norm_out_weight, ln_out_b=norm_out_bias, w_o=p_out_weight, w_og=g_out_weight)
    fns = RTF.fused_trimul_fns(weights, stock, eps=float(eps), min_tokens=(int(mt) if mt else None))
    TRIMUL_PROVIDER.update(bound=True, why=f"{RTF.LEVER}:{RTF.KERNEL} min_tokens={fns.min_tokens}")
    if key is not None:
        if len(_FUSED_TRIMUL) > 512:
            _FUSED_TRIMUL.clear()
        _FUSED_TRIMUL[key] = fns
    return fns


def _mask_rows2d(mask, b: int, R: int, N: int, torch):
    """The pair-mask rows ``[R, N]`` (float) of batch element ``b`` from the block's ``pair_attention_mask`` rows (``[B, R, N]`` / ``[R, N]`` /
    None)."""
    if mask is None:
        return None
    m = mask if mask.dim() == 2 else mask[min(b, int(mask.shape[0]) - 1)]
    if tuple(m.shape) != (R, N):
        raise RowpairRefused(f"pair mask rows {tuple(mask.shape)} do not match the shard rows [{R}, {N}]")
    return m.to(torch.float32).contiguous()


def _trimul_rows_(fns, z_rows, mask, lay, *, outgoing: bool, add: bool, torch):
    """``trimul_update_`` over the batch elements of the shard ``z_rows [B, R, N, C]`` IN PLACE (``add``: residual add; else the update is
    written). ``z_rows`` must be contiguous (it is the block's own pair tensor)."""
    from opt_core.mem.rowpair import trimul as RT
    B, R, N = int(z_rows.shape[0]), int(z_rows.shape[1]), int(z_rows.shape[2])
    st = TRIMUL_STATS.setdefault("outgoing" if outgoing else "incoming", RT.ContractStats())
    chunk = int(os.environ.get(ENV_TRIMUL_CHUNK, "256") or 256)
    for b in range(B):
        RT.trimul_update_(fns, z_rows[b], _mask_rows2d(mask, b, R, N, torch), lay, outgoing=outgoing, add=add, inplace_chunk=chunk, stats=st)
    return z_rows


def _trimul_fused_dispatch(pair, direction, residual, drop_mask, *, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight,
                           norm_out_bias, p_out_weight, g_out_weight, mask=None, eps=1e-5, precision=0):
    """``modeling_esmfold2_common._fused_trimul_with_residual`` under sharding (the name W4's ``_trimul_w4`` serves): outside a sharded trunk
    call the original runs unchanged; inside, ``pair`` is this rank's rows and the residual triangle-multiplication update runs IN PLACE on
    them through the family driver (:func:`_trimul_fns`) — no shard-sized plane, contraction output or prologue/epilogue temporary exists."""
    CMN = sys.modules[COMMON_MODULE]
    orig = PATCHES.original(CMN, "_fused_trimul_with_residual")
    if not CTX.active:
        return orig(pair, direction, residual, drop_mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight,
                    g_in_weight=g_in_weight, norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight,
                    g_out_weight=g_out_weight, mask=mask, eps=eps, precision=precision)
    import torch
    lay = CTX.layout
    if direction not in ("outgoing", "incoming"):
        raise RowpairRefused(f"trimul: direction {direction!r} (outgoing | incoming expected)")
    if not (residual is pair and drop_mask is None and int(precision) == 0):
        raise RowpairRefused(f"trimul {direction}: the row form is the residual update in place (needs residual==pair, no drop mask, precision 0; "
                             f"got residual_is_pair={residual is pair} drop_mask={drop_mask is not None} precision={precision})")
    if not _is_shard_of(pair, lay) or int(pair.shape[-1]) != D_PAIR:
        raise RowpairRefused(f"trimul {direction}: pair {tuple(pair.shape)} is not this rank's shard [B, {lay.R}, {lay.N}, {D_PAIR}]")
    if not pair.is_contiguous():
        raise RowpairRefused(f"trimul {direction}: the shard must be contiguous (it is updated in place)")
    W4 = _w4()
    if W4 is not None:
        W4.STATS["tx_rowpair_calls"] += 1                                # the fused line's counter: a trimul call served on rows
    fns = _trimul_fns(norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight, g_in_weight=g_in_weight,
                      norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight, eps=eps,
                      key=((W4._TX.get("cur_key") if W4 is not None and isinstance(getattr(W4, "_TX", None), dict) else None) or None))   # W4's PairUpdateBlock wrapper names (id(block), direction, weights key) around this call
    return _trimul_rows_(fns, pair, mask, lay, outgoing=(direction == "outgoing"), add=True, torch=torch)


def trimul_residual_rows_(module, pair, visibility=None):
    """``pair += module(pair, visibility)`` for a stock ``TriangleMultiplicativeBlock`` on this rank's rows, IN PLACE — the statement the
    folding trunk's fused path makes (:func:`_trimul_fused_dispatch`, ``add=True``): the update is computed from the original rows and added
    into ``pair`` block by block, so neither the update copy of :func:`_tmb_forward_dispatch` nor the caller's ``pair + update`` result
    exists. For a pair tensor the CALLER owns (the MSA encoder's own row copy); inside the sharded context only."""
    import torch
    lay = CTX.layout
    if not CTX.active or lay is None:
        raise RowpairRefused("trimul_residual_rows_: outside the sharded context")
    module = getattr(module, "_engine", module)                             # the MSA blocks hold the engine behind the thin TriangleMultiplicativeUpdate wrapper
    if getattr(module, "_use_kernels", False):
        raise RowpairRefused("TriangleMultiplicativeBlock: the cuequivariance kernel has no row-sharded form")
    if not _is_shard_of(pair, lay) or not pair.is_contiguous():
        raise RowpairRefused(f"trimul_residual_rows_: pair {tuple(pair.shape)} is not this rank's contiguous shard [B, {lay.R}, {lay.N}, C]")
    p_in, g_in = module.split_kernel_weights()
    fns = _trimul_fns(norm_in_weight=module.norm_start.weight, norm_in_bias=module.norm_start.bias, p_in_weight=p_in, g_in_weight=g_in,
                      norm_out_weight=module.norm_mix.weight, norm_out_bias=module.norm_mix.bias, p_out_weight=module.proj_emit.weight,
                      g_out_weight=module.proj_gate.weight, eps=float(module.norm_start.eps), key=("tmb", id(module), module.flow))
    return _trimul_rows_(fns, pair, visibility, lay, outgoing=(module.flow == "outgoing"), add=True, torch=torch)


def transition_residual_rows_(module, pair, rows: int):
    """``pair += module(pair)`` for a delta-returning row-local module (the MSA blocks' ``PairTransition``) per block of ``rows`` rows IN
    PLACE (``opt_core.mem.rowpair.pairstack.transition_update_``): the ``[rows, N, ·]`` delta is one block's, never the shard's, and no
    ``pair + delta`` result tensor exists. Same values per element as the whole-shard statement."""
    from opt_core.mem.rowpair import pairstack as RPS
    lay = CTX.layout
    if not CTX.active or lay is None or not _is_shard_of(pair, lay) or not pair.is_contiguous():
        raise RowpairRefused(f"transition_residual_rows_: pair {tuple(pair.shape)} is not this rank's contiguous shard inside the sharded context")
    for b in range(int(pair.shape[0])):
        RPS.transition_update_(lambda zb, mb: module(zb.unsqueeze(0))[0], pair[b], None, max(1, int(rows)), add=True)
    return pair


def _tmb_forward_dispatch(self, pair_grid, visibility=None):
    """``TriangleMultiplicativeBlock.forward`` (the REFERENCE backend's triangle multiplication; the MSA encoder's implementation under every backend)
    under sharding: outside a sharded trunk call the original runs; inside, the module's statements run on this rank's rows through the family
    driver (:func:`_trimul_fns`) and the UPDATE (what the stock forward returns; the caller adds it) is produced in a copy of the rows."""
    CMN = sys.modules[COMMON_MODULE]
    orig = PATCHES.original(CMN.TriangleMultiplicativeBlock, "forward")
    if not CTX.active:
        return orig(self, pair_grid, visibility)
    import torch
    lay = CTX.layout
    if getattr(self, "_use_kernels", False):
        raise RowpairRefused("TriangleMultiplicativeBlock: the cuequivariance kernel has no row-sharded form")
    if not _is_shard_of(pair_grid, lay):
        raise RowpairRefused(f"TriangleMultiplicativeBlock: pair {tuple(pair_grid.shape)} is not this rank's shard [B, {lay.R}, {lay.N}, C]")
    p_in, g_in = self.split_kernel_weights()
    fns = _trimul_fns(norm_in_weight=self.norm_start.weight, norm_in_bias=self.norm_start.bias, p_in_weight=p_in, g_in_weight=g_in,
                      norm_out_weight=self.norm_mix.weight, norm_out_bias=self.norm_mix.bias, p_out_weight=self.proj_emit.weight,
                      g_out_weight=self.proj_gate.weight, eps=float(self.norm_start.eps), key=("tmr", id(self), self.flow))
    update = pair_grid.contiguous().clone()                              # the driver reads original rows from, and writes the update into, this copy
    return _trimul_rows_(fns, update, visibility, lay, outgoing=(self.flow == "outgoing"), add=False, torch=torch)


def _transition_dispatch(self, x):
    """``Transition.forward`` under sharding: on this rank's pair shard the stock forward (``x + ffn(norm(x))``, row-local) runs per row block
    of ``EF2_ROWPAIR_TRANSITION_ROWS`` rows IN PLACE (``opt_core.mem.rowpair.pairstack.transition_update_``), so the expanded hidden is
    ``[rows, N, 4C]``; every other tensor takes the original forward."""
    CMN = sys.modules[COMMON_MODULE]
    orig = PATCHES.original(CMN.Transition, "forward")
    lay = CTX.layout
    if not CTX.active or not _is_shard_of(x, lay) or not x.is_contiguous():
        return orig(self, x)
    from opt_core.mem.rowpair import pairstack as RP
    rows = int(os.environ.get(ENV_TRANSITION_ROWS, "") or TRANSITION_ROWS_DEFAULT)
    for b in range(int(x.shape[0])):
        RP.transition_update_(lambda zb, mb: orig(self, zb.unsqueeze(0))[0], x[b], None, rows, add=False)
    return x


