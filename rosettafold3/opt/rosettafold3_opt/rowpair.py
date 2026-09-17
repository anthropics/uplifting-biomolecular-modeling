"""The kit's tensor-parallel adapter: ``--mode big --n_gpu P`` row-shards RF3's pair representation over the P GPUs of one node, from the
statement that creates it to the last statement that reads it.

``opt_core.mem.rowpair`` (the shared core) is the MECHANISM — the P rank processes the kit spawns itself, the process group, the
row partition ``Layout``, every row statement, block schedule and collective; this module is the INSTALL — which RF3 statement runs on which
core statement, with RF3's sub-modules passed as callables. It holds no statement body (``opt_core/mem/rowpair/ADAPTER_GUIDE.md`` is the
recipe this module follows; ``API.md`` there carries the interface words).

RF3's pair tensor is UNBATCHED ``Z_II [I, I, 128]`` in bf16 storage under the engine's autocast; under ``n_gpu = P > 1`` every rank holds
its rows ``z[r0:r1, :, :]`` (``[R, I, 128]``, the balanced contiguous partition of ``opt_core.mem.rowpair.dist.ctx``) and nothing
``I x I x c`` is ever whole on a rank:

    stage (stock name)                                  site rebound here                              core statement
    input featurisation (the engine's Transform         BaseInferenceEngine._construct_pipeline        rankdata.broadcast_features: rank 0 runs the pipeline, ranks > 0 receive its
      pipeline of the featurised example; the input       (the pipeline it builds, rf3.py:545)          featurised example (store rendezvous first; rank 0's host RNG state adopted) +
      parse stays per rank)                                                                              rankdata.assert_ranks_agree (refusing): data_form=rank0_bcast
    pair init: outer sum + relpos + token bonds         FeatureInitializer.forward                     trunk.init_pair_shard (z BORN as rows)
    recycling  Z = Z_init + linear(LN(Z_prev))          Recycler.forward                               trunk.recycle_shard_ (Z_init served by trunk.ShardPark:
      (Z_init parked on pinned host between recycles)                                                   ROWPAIR_PARK_ZINIT, parked at its birth in FeatureInitializer)
    template embedder (RF3's one conditioning slot)     Recycler.forward -> RF3TemplateEmbedder        template.template_embed_rows + pairstack.pair_stack_
    MSA module (OPM rows, pair-weighted averaging,      Recycler.forward -> MSAModule                  msa.msa_module_sharded (opm_rows_budgeted, pwa_rows,
      MSA transition, 4 weight-shared pair blocks)                                                       msa_transition_rows, pairstack.pair_block_)
    48 Pairformer blocks (tri-mult out/in, tri-att      Recycler.forward -> PairformerBlock x 48       pairstack.pair_stack_ (trimul.TriMulFns, triatt.TriAttFns,
      start/end, transition, attention-pair-bias)                                                        transition.apb_local_queries)
    distogram head  predictor(Z + Z^T)                  DistogramHead.forward                          dist.transpose_blocks (rows kept per rank: no consumer)
    confidence head (per diffusion sample + the         ConfidenceHead.forward                         tp_conf.run_confidence_sharded (embedding rows,
      early-stop probe after recycle 0: refused by name)                                                 this module's pair-block driver, logits per row block)
    diffusion conditioning pair (once per roll-out)     DiffusionConditioning.forward_hoisted          diffusion.pair_cond_rows
    diffusion transformer 24 blocks x steps x samples   DiffusionTransformer.forward                   diffusion.diffusion_transformer_sharded (local query rows)
    atom-encoder token-pair window term                 levers.PAIR_WINDOWS (big's window statement) diffusion.band_plan / pair_band_rows / band_lookup
    every random draw of the roll-out                   SampleDiffusion._get_initial_structure/_predraw diffusion.sync_replicated (bcast: rank 0's draw on every rank, then the guard)
    N x N INPUT features (token_bonds, distogram cond.)  model entry                              _shard_pair_features: each rank keeps its rows only (entry_pair_feats census)
    raw MSA stack msa_stack [n_rec, S, I, c] (per-recycle     engine H2D (Fabric.to_device) + MSAModule       msa_host.park_features / host_placeholder / rows_to_device (ROWPAIR_MSA_HOST:
      rows drawn by the data pipeline; RF3.py:264 selects [i])                                           rank 0's pinned host copy; the cycle's rows [S, I, c] reach the device, cast after the move)
    every denoiser call's input state                   DiffusionModule.forward                  diffusion.sync_replicated (bcast: rank 0's X_noisy_L on every rank, every step)
    sampler exit -> confidence entry                    ConfidenceHead.forward                   rank 0's sample on every rank + diff_rank_spread_A census (refused by name > 1 A)
    model entry                                         RF3WithConfidence.forward / RF3.forward        bcast.sync_tensordict_from_rank0 (rank 0's features)

REPLICATED BY DESIGN (named on the LEVER line): the single representation ``S`` and the token features, ``S_inputs`` / ``S_init`` (made
rank 0's by broadcast right after the atom encoder), the MSA representation ``m [S, I, 64]``, the gathered triangle-attention bias
``[I, I, 4]`` per call, the atom tensors and the atom transformer (windowed), the diffusion activations ``a [D, I, 768]`` (query rows local,
gathered per block), the predicted coordinates. Kit levers that cannot coexist with row sharding are OFF BY NAME under ``n_gpu > 1`` (the
ACTIVE line): the FPF whole-tile triangle-multiplication kernel and its fused residual epilogue (``fpf_trimul=off fpf_res=off
reason=conflict:n_gpu`` — the row-sharded contraction runs instead, its projections / epilogue on the core's fused row kernels
(``TP_TRIMUL_KERNELS``; the triangle attention per row block on ``TP_TRIATT_KERNEL``) at ``N >= TP_KERNEL_MIN_TOKENS`` and on the
module's own LayerNorm / Linear statements below, each non-served call a named event on the core's ``F2.trimul_rows`` /
``F1.flash_triattn`` LEVER lines),
the sampler's CUDA graph (``graph=replay reason=conflict:n_gpu``: a captured denoiser step would hold the transformer's all-gathers; the
kit's eager sampler replays the same pre-drawn random tensors), the ``dtk`` token-attention kernel (``dtk=off reason=conflict:n_gpu``: the
diffusion transformer's attention is the row-sharded statement here — this rank's query rows against all keys — which never enters the
module forward where the kernel's square-problem seam lives; each rank's exit tally carries the word and ``pred`` reads every rank's). Levers whose SITE this module takes are off by property
(``big.ROWPAIR_OWNS``). Refused by name under ``n_gpu > 1``: cyclic-chain inputs (``cyclic_asym_ids``), a batched trunk, training mode,
``use_deepspeed_evo`` attention, a Recycler.forward another arm owns.

Layers of this module::

    plan(n_gpu)                 pure facts (no torch): the SITES table, what stays replicated, the numerics class — for ``check``
    launch_argv(cmd, env, P)    what ``fold.launch`` runs at P > 1: ``(run_ranks, kwargs)``; ``run_ranks`` starts the P rank processes through
                                ``opt_core.mem.rowpair.launch.run_rank_processes`` (rank r on visible device r, loopback rendezvous, rank 0's
                                transcript streamed into ``stdout``, every rank's transcript under :func:`ranks_dir`)
    rank_argv(argv, r, dir)     rank r's command line (ranks > 0 write under ``<out_dir>.ranks/rank<r>``; rank 0's directory is the run's output)
    install(rep, n_gpu)         inside a rank process, at the model trigger, AFTER the FPF arm and the big levers: binds the group and rebinds
                                the sites (``opt_core.mem.patchset.PatchSet``, originals kept); P == 1 installs NOTHING
    state() / lever_fields()    the census the EXIT tally and the LEVER line carry (layout, sites reached, sync census, the core's schedule
                                census ``evidence.schedule_fields()``, per-rank peaks)

Numerics: every row statement is the dense statement on fewer rows (GEMM shapes differ from the one-GPU line only in M); the sharded
triangle-multiplication contraction keeps the whole k-sum inside one matmul per tile; softmaxes are whole per row. ``n_gpu > 1`` is
big-only: tier 2 (fast class), never bitwise with the one-GPU line (whose triangle multiplication is a fused kernel).
"""
from __future__ import annotations

import functools
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import _core
from . import templ as _templ

LEVER = "rowpair"
STRATEGY = "F7.tensor_parallel"
OUT_TOKEN = "out_dir="                             # the rf3 CLI's hydra token naming the output directory (fold.py builds it)
RANKS_SUFFIX = ".ranks"                            # <out_dir>.ranks/: rank<r>.log (every rank's transcript) and rank<r>/ (the outputs of ranks > 0)
TP_NCCL_TIMEOUT_S = 1800.0                          # the ×P group's collective timeout (opt_core.mem.rowpair.launch nccl_timeout_s; a rank_timeout is a named event)
ALIGN = 1                                          # RF3 pins no attention chunk: the balanced contiguous row partition (dist.ctx align=1)
TRIATT_ROWS = 256                                  # pair rows per triangle-attention / transition row-batch block (pairstack.bind chunk)
TRIMUL_CHUNK = 256                                 # the b sub-block grid's column period (trimul_update_ inplace_chunk): RF3's stock statement is one
                                                   # dense einsum / one fused kernel, so the grid is this adapter's choice (printed in the schedule census)
GRAPH_MODE_TP = "replay"                           # RF3_CUDAGRAPH under n_gpu>1: the kit's eager sampler replaying the pre-drawn random tensors
TRIATT_KERNELS = ("torch", "flash_triattn", "cueq")  # the triangle-attention core per row block: "flash_triattn" (this line's) | "cueq" = opt_core's row-block attention
TP_TRIATT_KERNEL = "flash_triattn"                 # core (mem.rowpair.triatt.attention_core) over the module's projections, RF3's einsum statement its named fallback;
                                                   # "torch" = the module's own statement (cuEquivariance's kernel when the run engages it, RF3's einsum otherwise).
                                                   # One word per process; the core's ROWPAIR_TRIATT_CORE overrides it inside the core.
TRIMUL_KERNELS = ("torch", "fpf_v4")               # the triangle-multiplication kernels of the row-sharded contraction: "fpf_v4" (this line's) = opt_core's row-sharded
TP_TRIMUL_KERNELS = "fpf_v4"                       # fused TriMul (mem.rowpair.trimul_fused: K1 projections / K3 epilogue per tile on the FPF add-on's fpf_trimul_v4
                                                   # cells) with the module's LayerNorm / Linear statements around the shared torch contraction (trimul.TriMulFns) as
                                                   # its named fallback; "torch" = those statements only (the core's ROWPAIR_TRIMUL_KERNELS overrides in the core)
TP_KERNEL_MIN_TOKENS = 2048                        # pair size N below which both fused row kernels leave every call to the torch statements (the core's below_gate
                                                   # event, counted): below this size the fused TriMul rows are not faster than the statements

# TP-MAP: site -> (module, class, attribute, core statement, communication, numerics class at fixed P); ``plan()`` renders it.
SITES: Tuple[Tuple[str, str, str, str, str, str, str], ...] = (
    ("entry", "rf3.model.RF3", "RF3WithConfidence", "forward", "bcast.sync_tensordict_from_rank0 of the input dict (rank 0's features on every rank)", "broadcast", "bitwise (data movement)"),
    ("entry_rf3", "rf3.model.RF3", "RF3", "forward", "the same entry sync for the confidence-free model class", "broadcast", "bitwise (data movement)"),
    ("pair_init", "rf3.model.layers.pairformer_layers", "FeatureInitializer", "forward", "trunk.init_pair_shard(outer_sum_rows + relpos_onehot_rows + feature_rows): z born as rows; S_inputs/S_init broadcast from rank 0 first", "broadcast of S_inputs/S_init", "row-local"),
    ("recycler", "rf3.model.RF3_structure", "Recycler", "forward", "trunk.recycle_shard_ -> template.template_embed_rows -> msa.msa_module_sharded -> pairstack.pair_stack_ (48 blocks); the shard is never gathered", "ring / all-to-all / bias all-gather / row all-gather (pair stack), all-gather (PWA)", "band (tile schedule printed)"),
    ("distogram", "rf3.model.RF3_structure", "DistogramHead", "forward", "predictor(z_rows + zT_rows) with zT rows from dist.transpose_blocks; logits stay rows (no inference consumer)", "all-to-all", "row-local"),
    ("conf_head", "rf3.model.layers.af3_auxiliary_heads", "ConfidenceHead", "forward", "tp_conf.run_confidence_sharded: embedding rows, the 4 pairformer blocks through pairstack.pair_stack_, pae/pde logits consumed per row block (ContextReducer; expected matrices to rank 0's pinned host)", "pair stack collectives + one T gather + host row gathers", "band"),
    ("denoiser", "rf3.model.RF3_structure", "DiffusionModule", "forward", "every denoiser call's input state X_noisy_L becomes rank 0's (diffusion.sync_replicated mode=bcast) before the stock body: replicated-by-recompute drift never enters the sharded blocks", "one broadcast of [D, L, 3] per step", "rank 0's state (no arithmetic)"),
    ("diff_cond", "rf3.model.RF3_structure", "DiffusionConditioning", "forward_hoisted", "diffusion.pair_cond_rows (once per roll-out; the hoist holds the shard)", "none", "row-local"),
    ("dit", "rf3.model.layers.af3_diffusion_transformer", "DiffusionTransformer", "forward", "diffusion.diffusion_transformer_sharded (query rows local, K/V whole, bias rows cached per roll-out)", "all-gather of a rows per block", "row-local"),
    ("compile", "rf3.utils.predicted_error", "-", "compile_af3_style_confidence_outputs", "tp_conf.compile_on_expected: RF3's compile verbatim on the expected PAE/PDE host matrices (rank 0; scratch on ranks > 0)", "none", "verbatim"),
    ("metric_ptm", "rf3.metrics.predicted_error", "ComputePTM", "compute", "tp_conf.metrics_from_tm: the TM scalars the row-block reducer finished", "none", "verbatim"),
    ("metric_iptm", "rf3.metrics.predicted_error", "ComputeIPTM", "compute", "as metric_ptm (the four interface masks)", "none", "verbatim"),
    ("noise_init", "rf3.diffusion_samplers.inference_sampler", "SampleDiffusion", "_get_initial_structure", "the stock draw, then diffusion.sync_replicated('diffusion_noise_init', mode=bcast): rank 0's draw on every rank + the guard", "one broadcast + all-gather of 3 integers", "rank 0's draw (no arithmetic)"),
    ("featurise", "foundry.inference_engines.base", "BaseInferenceEngine", "_construct_pipeline", "the input featurisation runs on rank 0 alone (data_form=rank0_bcast): a CLASS-level wrap of the engine base's pipeline constructor, so the Transform pipeline it builds (inference_engines/rf3.py:545 `self.pipeline(...)`) runs on rank 0 and ranks > 0 receive its featurised example through opt_core.mem.rowpair.rankdata.broadcast_features — rendezvous on the group's store first (a long featurisation holds no collective), the feature tensors by broadcast, the atom array and the other non-tensor leaves pickled with the tree, rank 0's host RNG state after featurising adopted by the receivers, the raw MSA stack NOT sent while ROWPAIR_MSA_HOST is set (ranks > 0 hold its zero-row placeholder, msa_host.place_skipped; mode all fills it at the model entry, msa_host.sync_host_features_) — then every rank proves it holds rank 0's bytes (rankdata.assert_ranks_agree, refusing: feats_ranks_differ); a featurisation that raised on rank 0 is the same refusal on every rank (feats_rank0_failed); the input parse (prepare_inference_inputs_from_paths) stays per rank", "one store rendezvous, one object broadcast and the feature tree's broadcasts per item + one digest all-reduce", "rank 0's features (data movement)"),
    ("h2d", "lightning.fabric.fabric", "Fabric", "to_device", "ROWPAIR_MSA_HOST: a CLASS-level wrap of lightning's Fabric.to_device, installed under n_gpu>1 only and restored at uninstall (PatchSet) — engaged solely for the engine's H2D of the featurised example (inference_engines/rf3.py:562 trainer.fabric.to_device: a mapping whose feats hold a host msa_stack, the lever set), every other call passes through unchanged; the raw MSA stack is held back — msa_host.park_features keeps rank 0's copy on pinned host (rank0) / every rank's (all); the model's dict entry is a zero-token placeholder; per recycle msa_host.rows_to_device moves that cycle's rows (rank0: broadcast); anything else passes through", "per-cycle chunked broadcast of [S, I, c] (rank0)", "bitwise (data movement; cast after select)"),
    ("noise_predraw", "rf3.diffusion_samplers.inference_sampler", "SampleDiffusion", "_predraw", "the kit sampler's pre-drawn random tensors of the roll-out, each through sync_replicated(mode=bcast): rank 0's on every rank + the guard", "one broadcast + all-gather of 3 integers per tensor", "rank 0's draws (no arithmetic)"),
    ("templ_feats", "rf3.data.ground_truth_template", "-", "featurize_noised_ground_truth_as_template_distogram", "templ.distogram_condition_precursors: the noised-template distogram FEATURE born as per-token precursors (noised centres, fill mask, molecule ids, bin edges) in place of the dense [N, N, 64] one-hot + [N, N] mask; the template embedder builds its pair rows from them (templ.distogram_condition_rows)", "none (the entry sync carries the precursors)", "bitwise (the dense statement's rows)"),
)
SEAMS = (                                          # kit seams this module rebinds under n_gpu>1 (no class attribute: a module-level callable of the kit)
    ("pair_windows", "rosettafold3_opt.levers", "PAIR_WINDOWS", "diffusion.band_plan + pair_band_rows + band_lookup: the atom encoder's token-pair window term from the conditioned shard", "all-gather of the band [I, 2W+1, 16]", "row-local"),
)
REPLICATED: Tuple[str, ...] = (
    "the featurised example (rank 0's Transform pipeline output, received by ranks > 0: data_form=rank0_bcast; the raw MSA stack under ROWPAIR_MSA_HOST excepted: msa_host)",
    "S / S_inputs / S_init and the per-token features (S_inputs, S_init broadcast from rank 0 after the atom encoder)",
    "the MSA representation m [S, I, 64] (msa_m=replicated)",
    "the gathered triangle-attention bias [I, I, 4] per call",
    "atom tensors, the windowed atom pair and the atom transformer",
    "the diffusion activations a [D, I, 768] (query rows local, gathered per block) and the predicted coordinates",
    "the writer tail (every rank; ranks > 0 under the scratch dir)",
)
CONFLICTS = {                                      # kit levers OFF BY NAME under n_gpu>1 (the ACTIVE / LEVER line carries lever=off reason=conflict:n_gpu)
    "fpf_trimul": "the FPF whole-tile triangle-multiplication kernel cannot run on a row shard: the module's LayerNorm/Linear statements run over the core contraction",
    "fpf_res": "the fused residual epilogue belongs to the FPF kernel",
    "graph": f"a captured denoiser step would hold the transformer's collectives: RF3_CUDAGRAPH={GRAPH_MODE_TP} (eager replay of the same pre-drawn tensors)",
    "dtk": "the diffusion transformer's token attention runs as the row-sharded statement (this rank's query rows against all keys, the pair bias as rows: opt_core.mem.rowpair.diffusion.dit_block_sharded), which never enters AttentionPairBiasDiffusion.forward where the dtk kernel's seam lives (its contract is one square [H, I, I] problem per call)",
}


class RowpairUnavailable(RuntimeError):
    """``n_gpu > 1`` asked of a core that carries no ``opt_core.mem.rowpair``."""


# ------------------------------------------------------------------------------------------------------------------ lazy core / torch
_RP: Dict[str, Any] = {}


def _rp(name: str = ""):
    """``opt_core.mem.rowpair[.name]`` through the kit's pinned-core loader; imported on first use (never at P == 1)."""
    key = "mem.rowpair" + ("." + name if name else "")
    mod = _RP.get(key)
    if mod is None:
        try:
            mod = _core.load(key)
        except (ImportError, AttributeError) as e:
            raise RowpairUnavailable(f"refused: core_missing:opt_core.{key} (n_gpu>1 needs opt_core >= 0.4.3; this core: {e})") from None
        _RP[key] = mod
    return mod


def _torch():
    return _rp("_torch").torch


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else int(default)
    except ValueError:
        return int(default)


# ---------------------------------------------------------------------------------------------------------------------------- facts
def plan(n_gpu: int = 1, mode: Optional[str] = None) -> dict:
    """Pure facts of a ``--n_gpu`` value (no torch): what shards, what stays replicated, which kit levers are off by name, the numerics
    class. Refusals (n_gpu>1 outside big, fewer visible GPUs) are the CLI's, through ``opt_core.mem.ngpu``; this only describes."""
    p = int(n_gpu)
    if p < 1:
        raise ValueError(f"n_gpu must be >= 1 (got {n_gpu!r})")
    keys = ("site", "module", "cls", "attr", "statement", "comm", "numerics")
    return {
        "lever": LEVER, "strategy": STRATEGY, "n_gpu": p, "sharding": "rowpair" if p > 1 else "none", "scheme": "rowpair" if p > 1 else "none",
        "reason": None, "mode": mode,
        "installs": [(f"{mod}.{attr}" if cls == "-" else f"{cls}.{attr}") for _, mod, cls, attr, _, _, _ in SITES] + [f"{mod}.{attr}" for _, mod, attr, _, _, _ in SEAMS] if p > 1 else [],
        "sites": [dict(zip(keys, row)) for row in SITES] + [dict(zip(("site", "module", "attr", "statement", "comm", "numerics"), row)) for row in SEAMS],
        "replicated": list(REPLICATED),
        "conflicts": dict(CONFLICTS) if p > 1 else {},
        "numerics": "band (fast class; per-row statements dense, tile schedule printed)" if p > 1 else "the mode's own (nothing installed)",
        "knobs": {"ROWPAIR_TRIATT_QROWS": "0 (one call per row batch)", "align": ALIGN, "triatt_rows": TRIATT_ROWS, "trimul_chunk": TRIMUL_CHUNK},
    }


# --------------------------------------------------------------------------------------------------------------------------- launcher
def rank_argv(argv: Sequence[str], rank: int, scratch_dir: str) -> List[str]:
    """The command line of rank ``rank``: rank 0's is ``argv`` unchanged; rank r > 0 gets its ``out_dir=`` token rewritten to
    ``<scratch_dir>/rank<r>`` (every rank runs the tail and writes; rank 0's directory is the run's output). An argv without an
    ``out_dir=`` token is refused by name (the ranks' writers would collide)."""
    r = int(rank)
    out = list(argv)
    if r == 0:
        return out
    hits = [i for i, tok in enumerate(out) if isinstance(tok, str) and tok.startswith(OUT_TOKEN)]
    if len(hits) != 1:
        raise ValueError(f"rank_argv: expected exactly one {OUT_TOKEN!r} token in the child argv, found {len(hits)}: ranks > 0 need a scratch out_dir")
    out[hits[0]] = OUT_TOKEN + os.path.join(scratch_dir, f"rank{r}")
    return out


def ranks_dir(argv: Sequence[str]) -> str:
    """``<out_dir>.ranks`` for the child command ``argv`` (its one ``out_dir=`` token); refused by name without the token."""
    hits = [tok for tok in argv if isinstance(tok, str) and tok.startswith(OUT_TOKEN)]
    if len(hits) != 1:
        raise ValueError(f"ranks_dir: expected exactly one {OUT_TOKEN!r} token in the child argv, found {len(hits)}")
    return hits[0][len(OUT_TOKEN):].rstrip("/") + RANKS_SUFFIX


def launch_argv(cmd: Sequence[str], env: Dict[str, str], n_gpu: int):
    """What ``fold.launch`` runs at ``n_gpu > 1``: ``(run_ranks, kwargs)`` — ``run_ranks(stdout=None, timeout=None, **kwargs) -> rc``. Rank r's
    command is :func:`rank_argv`; ``env`` is the base every rank's ``ROWPAIR_*`` names are layered on; transcripts
    land in ``<out_dir>.ranks/rank<r>.log``."""
    p = int(n_gpu)
    if p <= 1:
        raise ValueError(f"launch_argv: n_gpu={n_gpu}: P == 1 is the kit's own child process (fold.launch), never the rank launcher")
    rd = ranks_dir(cmd)
    return run_ranks, {"n_gpu": p, "argv_of": functools.partial(_rank_argv_of, list(cmd), rd), "env": dict(env), "log_dir": rd}


def _rank_argv_of(argv: Sequence[str], rd: str, r: int) -> List[str]:
    return rank_argv(argv, r, rd)


def run_ranks(*, n_gpu: int, argv_of: Callable[[int], Sequence[str]], env: Dict[str, str], log_dir: str, stdout=None,
              timeout: Optional[float] = None) -> int:
    """Start the P rank processes (``opt_core.mem.rowpair.launch.run_rank_processes``: one visible device per rank, loopback rendezvous,
    fail-fast teardown) and return 0, or the failing rank's exit code after ONE ``ROWPAIR event=<rank_failed|rank_timeout> rank=<r>
    exitcode=<c> log=<path>`` line (``evidence.rank_failed_line``). Every rank starts under ONE ``PYTHONHASHSEED`` — ``env``'s value when it names
    an integer, else ``0`` (unset / empty / ``random``) — decided by the core launcher, which says so once on stderr: ``[rosettafold3-opt] RANKENV hashseed=<v> source=default|inherited
    ranks=<P>`` (``opt_core.mem.rowpair.rankdata``). Rank 0's transcript streams into ``stdout`` (the pred log) as it arrives."""
    launch = _rp("launch")
    ev = _rp("evidence")
    sink = stdout if stdout is not None else sys.stderr

    def on_line(line: str) -> None:
        sink.write(line if line.endswith("\n") else line + "\n")
        sink.flush()

    os.makedirs(log_dir, exist_ok=True)
    try:
        launch.run_rank_processes(int(n_gpu), argv_of=argv_of, mode=_rp().MEMORY_MODE, env=dict(env, **{launch.ENV_TAG: _report_prefix()}),   # the core's launch line `[<tag>] RANKENV hashseed=<v> source=default|inherited ranks=P` carries the kit's tag
                                  log_dir=log_dir, isolate_devices=True, nccl_timeout_s=TP_NCCL_TIMEOUT_S, run_timeout_s=timeout, on_line=on_line, what="rosettafold3 rank")
        return 0
    except launch.RankFailed as e:
        on_line(ev.rank_failed_line(_report_prefix(), e))
        code = e.exitcode if isinstance(e.exitcode, int) and e.exitcode not in (0, None) else 1
        return int(code) if code > 0 else 128 - int(code)


def _report_prefix() -> str:
    try:
        from . import report as _report
        return str(_report.PREFIX).strip("[]")
    except Exception:  # noqa: BLE001
        return "rosettafold3-opt"


def is_output_rank() -> bool:
    """Rank 0 (or any P == 1 process): the rank whose lines and files are the run's (``ROWPAIR_RANK`` unset or 0)."""
    return os.environ.get("ROWPAIR_RANK", "").strip() in ("", "0")


def env_world() -> int:
    """``ROWPAIR_WORLD`` of this process (1 when unset): the P the kit's launcher started this rank under."""
    return max(1, _env_int("ROWPAIR_WORLD", 1))


# ---------------------------------------------------------------------------------------------------------------------------- state
_LOCK = threading.Lock()
CTX: Dict[str, Any] = {
    "installed": False, "census": None, "n_gpu": 1, "rank": 0, "device": None, "patches": None, "layout": None,
    "calls": {}, "sync": {}, "stats": {}, "layouts": {}, "timing": {"trunk": [], "conf": [], "diffusion": []}, "peaks": {}, "mem_trace": [], "errors": [],
    "absent_sites": [], "conflicts": {}, "cycle": 0, "fns": {}, "band": {}, "f": None, "tm": [], "params_guarded": False, "pair_feats": {}, "nxn_census_done": False,
    "item": 0,                                     # this rank's item ordinal: +1 at every model entry (a per-process census counts an item once, at whichever recycle first reaches it: _templ_census); a dedicated ordinal — "calls" is a site census, reassignable, and the featurise ordinal counts another event
    "ckpt": None, "resume": None, "example_id": None, "trunk_n": None,   # ROWPAIR_CKPT_DIR: the item's core TrunkCheckpointer, its resume state {tag, cycle, s, z_loc[, s_input]}, the pipeline's example_id (H2D seam), n_recycle
    "park_pool": None,                             # the process-lifetime pinned host pool of the parks (core trunk.park_pool: ROWPAIR_PARK_PIN_MAX_GB / ROWPAIR_PARK_STRICT), shared by the z_init park and the confidence plan
    "conf_plan": None, "n_samples": None,         # the confidence stage's heads.ZTrunkPlan over the item's per-sample passes {"plan", "i", "moments", "passes"}; D = input["t"].shape[0] (RF3.py:406)
    "msa_host": None,                              # ROWPAIR_MSA_HOST: {"mode", "feats": {"msa_stack": rank 0's pinned host stack | the zero-row placeholder on ranks > 0}, "shape", "dtype", "site"} of the item in flight
    "zinit": None, "zinit_tensor": None,          # the z_init shard's park (trunk.ShardPark, ROWPAIR_PARK_ZINIT) and the tensor object the stock control flow carries for it
}


def _count(site: str, kind: str = "sharded") -> None:
    c = CTX["calls"].setdefault(site, {"sharded": 0, "replicated": 0})
    c[kind] = int(c.get(kind, 0)) + 1


def _mem_mark(stage: str) -> None:
    """One census point of this rank's device memory at a stage boundary: ``(stage, GiB allocated now, GiB running max)`` — the stage in
    which the running maximum last rose is where the per-rank peak sits (``peak_stage`` on the LEVER line). Nothing is reset."""
    torch = _torch()
    if not torch.cuda.is_available():
        return
    tr = CTX["mem_trace"]
    if len(tr) >= 64:                                                                       # entry, init, ≤ 11 recycles, distogram, cond, heads: bounded
        return
    tr.append((stage, round(torch.cuda.memory_allocated() / 2 ** 30, 2), round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))


def _peak_stage(trace) -> Optional[str]:
    """The stage during which the running device-memory maximum last increased (None without a trace)."""
    best, last = None, -1.0
    for stage, _cur, mx in trace:
        if mx > last:
            best, last = stage, mx
    return best


def state() -> dict:
    """The exit tally's block (JSON-able; this rank's view). ``installed`` False at P == 1."""
    peaks = dict(CTX["peaks"])
    schedule = {}
    try:
        if CTX["installed"]:
            torch = _torch()
            if torch.cuda.is_available():
                peaks["self_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            schedule = dict(_rp("evidence").schedule_fields())
            apb = CTX.get("apb")
            if apb:                                                                          # the attention-pair-bias core on local query rows: word (line|env),
                schedule.update(apb_core=apb["core"], apb_core_src=apb["src"], apb_served=int(apb["served"]),   # calls served by the fused statement, named
                                apb_fallback=int(apb["fallback"]),                           # fallbacks of refused calls by event (none: absent)
                                apb_fallback_by=",".join(f"{k}:{v}" for k, v in sorted(apb["by"].items())) or None)
            if CTX.get("opm_row_bytes"):
                schedule["opm_row_bytes"] = int(CTX["opm_row_bytes"])                       # the OPM statement's per-row transient the row block was sized for
    except Exception as e:  # noqa: BLE001 — evidence must not raise at exit
        CTX["errors"].append(f"state: {type(e).__name__}: {e}")
    lay = CTX["layouts"].get("trunk") or {}
    stats = {}
    for k, st in CTX["stats"].items():
        stats[k] = {a: getattr(st, a, None) for a in ("passes", "slabs", "ring_steps", "tiles", "mode", "RA", "RB")}
    trunk = CTX["timing"]["trunk"]
    return {
        "lever": LEVER, "strategy": STRATEGY, "installed": bool(CTX["installed"]), "n_gpu": int(CTX["n_gpu"]), "rank": int(CTX["rank"]),
        "layout": (f"N={lay.get('N')},P={lay.get('P')},align={lay.get('align')}" if lay else None), "rows": (",".join(f"{int(a)}:{int(b)}" for a, b in lay["bounds"]) if lay and lay.get("bounds") else None),
        "peak_gib": peaks.get("self_gib"), "trunk_s": (round(sum(trunk) / len(trunk), 3) if trunk else None),
        "timing": {k: list(v) for k, v in CTX["timing"].items()}, "absent_sites": list(CTX["absent_sites"]),
        "mem_trace": list(CTX["mem_trace"]), "peak_stage": _peak_stage(CTX["mem_trace"]),
        "sharding": "rowpair" if CTX["installed"] else "none", "device": CTX["device"],
        "patched": list(CTX["patches"].names()) if CTX["patches"] is not None else [],
        "layouts": dict(CTX["layouts"]), "calls": {k: dict(v) for k, v in CTX["calls"].items()},
        "sync": {k: dict(v) for k, v in CTX["sync"].items()}, "conflicts": dict(CTX["conflicts"]), "replicated": list(REPLICATED),
        "trimul": stats, "schedule": schedule, "band": dict(CTX["band"]), "peaks": peaks, "errors": list(CTX["errors"]),
    }


def lever_fields() -> List[Tuple[str, object]]:
    """Evidence pairs for the kit's LEVER line of this lever (``report.lever_lines`` renders them after the registry's own fields):
    ``n_gpu sharding`` first (the core's token order), then the layout, the call census, the sync census, the conflicts, the schedule."""
    st = state()
    if not st["installed"]:
        return [("n_gpu", st["n_gpu"]), ("sharding", "none")]
    ev = _rp("evidence")
    lay = CTX["layouts"].get("trunk") or {}
    pairs = list(ev.fields(st["n_gpu"]))
    pairs += [("N", lay.get("N")), ("P", lay.get("P")), ("R", lay.get("R")), ("row_bounds", st.get("rows"))]
    pairs += [(f"calls_{k}", f"{v['sharded']}s/{v['replicated']}r") for k, v in sorted(st["calls"].items())]
    pairs += [(f"sync_{k}", f"{v.get('synced', 0)}synced/{v.get('skipped', 0)}host") for k, v in sorted(st["sync"].items())]
    pairs += [(k, "off:conflict:n_gpu") for k in sorted(st["conflicts"])]
    pairs += [(f"tp_{k}", w) for k, w in sorted(TP_KERNELS.items()) if w is not None]      # a fused triangle kernel word in force, or torch:<reason> (absent = torch)
    pairs += [("replicated", "s,s_inputs,m,tri_bias,atoms,a,x"), ("conf_finish", "row_blocks"), ("distogram", "rows")]
    pairs += list(st["schedule"].items())
    pairs += [("peak_gib", st["peak_gib"]), ("peak_stage", st.get("peak_stage")),
              ("mem_trace", ";".join(f"{a}:{b}/{c}" for a, b, c in st.get("mem_trace") or []) or None)]
    return pairs


# -------------------------------------------------------------------------------------------------------------------------- install
def install(rep: Optional[dict] = None, n_gpu: Optional[int] = None, classes: Optional[Dict[str, Any]] = None, *,
            init_group: bool = True) -> dict:
    """Inside a rank process, after the FPF arm and the big levers: bind the process group (``opt_core.mem.rowpair.dist.init_from_env``,
    NCCL, this rank's one visible device), switch the sampler to eager replay (:data:`CONFLICTS`) and rebind the SITES. ``n_gpu`` defaults
    to ``ROWPAIR_WORLD``. Returns the census dict (also stored under ``rep["rowpair"]``). ``n_gpu == 1``: returns ``{installed: False,
    sharding: none}`` and touches nothing. Refused by name (``RowpairRefused``): a world that is not ``n_gpu``, a Recycler.forward another arm
    owns, a second install. ``classes`` ({site: class-or-module}) replaces the SITES imports and ``init_group=False`` skips the group (the CPU
    tests' stand-ins under ``opt_core.testing.run_ranks``); the kit never passes them."""
    p = env_world() if n_gpu is None else int(n_gpu)
    census = {"lever": LEVER, "strategy": STRATEGY, "n_gpu": p, "sharding": "none", "scheme": "none", "installed": False, "patched": [], "rank": 0,
              "reason": "n_gpu:1" if p == 1 else None, "sites": [row[0] for row in SITES] + [row[0] for row in SEAMS] if p > 1 else [],
              "conflicts": dict(CONFLICTS) if p > 1 else {}}
    if p == 1:
        if rep is not None:
            rep["rowpair"] = census
        return census
    rp = _rp()
    with _LOCK:
        if CTX["installed"]:                                                                # ONCE per process: a second call is a named no-op when it asks for the
            if int(CTX["n_gpu"]) != p:                                                      # installed P (the census as installed, reinstall word), refused otherwise
                raise rp.RowpairRefused(f"rowpair.install called again with n_gpu={p}; installed under n_gpu={CTX['n_gpu']}")
            census = dict(CTX["census"], reinstall="skipped:already_installed")
            print(f"[{_report_prefix()}] ROWPAIR REINSTALL skipped=already_installed n_gpu={p} (the sites stay as bound)", file=sys.stderr, flush=True)
            if rep is not None:
                rep["rowpair"] = census
            return census
        dist = _rp("dist")
        if init_group:
            world = env_world()
            if world != p:
                raise rp.RowpairRefused(f"n_gpu={p} but this process's ROWPAIR_WORLD={world}: ranks are started by the kit's launcher (opt_core.mem.rowpair.launch), never by hand")
            P, rank, device = dist.init_from_env(device_index=_rp("launch").local_rank())
        else:
            P, rank = dist.world()
            device = "cpu"
        if int(P) != p:
            raise rp.RowpairRefused(f"process group world {P} != n_gpu {p}")
        patches = _core.load("mem.patchset").PatchSet(LEVER)
        try:
            _install_sites(patches, classes)
        except Exception:
            patches.restore()
            if init_group:
                dist.destroy()
            raise
        census.update(sharding="rowpair", scheme="rowpair", installed=True, patched=patches.names(), rank=int(rank), device=str(device),
                      absent_sites=list(CTX["absent_sites"]))
        CTX.update(installed=True, n_gpu=p, rank=int(rank), device=str(device), patches=patches, conflicts=dict(CONFLICTS), census=census)
        if rep is not None:
            rep["rowpair"] = census
        return census


def uninstall(destroy_group: bool = True) -> List[str]:
    """Restore every rebound site (and destroy the group unless ``destroy_group=False``); a kit process installs once and exits — tests use it."""
    with _LOCK:
        names = []
        if CTX["patches"] is not None:
            names = CTX["patches"].restore()
        if CTX["installed"] and destroy_group:
            _rp("dist").destroy()
        _release_zinit()
        CTX["msa_host"] = None
        CTX.pop("featurise_calls", None)
        CTX.update(installed=False, census=None, n_gpu=1, rank=0, device=None, patches=None, layout=None, fns={}, band={}, featurise_site=False)
        _rp("dist").clear_layout_cache()
        return names


def _install_sites(patches, classes: Optional[Dict[str, Any]] = None) -> None:
    import importlib
    rp = _rp()
    objs: Dict[str, Any] = {}
    for site, modname, clsname, attr, _, _, _ in SITES:
        if classes is not None:
            objs[site] = classes.get(site)
            continue
        try:
            m = importlib.import_module(modname)
            objs[site] = m if clsname == "-" else getattr(m, clsname)
        except (ImportError, AttributeError) as e:
            raise rp.RowpairRefused(f"site {site}: {modname}.{clsname} is not importable ({type(e).__name__}: {e}); the pinned rf3 defines it") from None
    for site, modname, attr, _, _, _ in SEAMS:
        if classes is not None:
            objs[site] = classes.get(site)
            continue
        try:
            objs[site] = importlib.import_module(modname)
        except ImportError as e:
            raise rp.RowpairRefused(f"seam {site}: {modname} is not importable ({e})") from None
    required = ("pair_init", "recycler")
    for site in required:
        if objs.get(site) is None:
            raise rp.RowpairRefused(f"site {site}: no class given (classes= must name at least {required})")
    Recycler = objs["recycler"]
    f = Recycler.forward
    owner = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', '?')}"
    if classes is None and owner != "rf3.model.RF3_structure.Recycler.forward":
        raise rp.RowpairRefused(f"Recycler.forward is {owner}: rowpair needs the stock recycler entry (an arm that rebinds it, e.g. the FPF trunk graph `tg`, cannot run under n_gpu>1)")
    # the sampler: eager replay under n_gpu>1 (a captured step would hold collectives) — set on the kit's own flag module, by name
    if classes is None:
        GF = sys.modules.get("rf3.graph_flags")
        if GF is None:
            raise rp.RowpairRefused("rf3.graph_flags is not imported: the kit's patched tree is the n_gpu>1 tree (tree_state=patched)")
        GF.set_mode(GRAPH_MODE_TP, safe_ops=bool(GF.GRAPH_SAFE_OPS), hoist=bool(GF.HOIST))
    for site in ("entry", "entry_rf3"):
        if objs.get(site) is not None:
            _wrap(patches, objs[site], "forward", site, _entry_forward)
    _wrap(patches, objs["pair_init"], "forward", "pair_init", _feature_init_forward)
    _wrap(patches, Recycler, "forward", "recycler", _recycler_forward)
    if objs.get("distogram") is not None:
        _wrap(patches, objs["distogram"], "forward", "distogram", _distogram_forward)
    if objs.get("conf_head") is not None:
        _wrap(patches, objs["conf_head"], "forward", "conf_head", _confidence_forward)
    if objs.get("diff_cond") is not None:
        if hasattr(objs["diff_cond"], "forward_hoisted"):
            _wrap(patches, objs["diff_cond"], "forward_hoisted", "diff_cond", _diff_cond_forward)
        _wrap(patches, objs["diff_cond"], "forward", "diff_cond", _diff_cond_forward)
    if objs.get("dit") is not None:
        _wrap(patches, objs["dit"], "forward", "dit", _dit_forward)
    if objs.get("denoiser") is not None:
        _wrap(patches, objs["denoiser"], "forward", "denoiser", _denoiser_forward)
    if objs.get("compile") is not None:                                                   # a module-level function (+ the engine module's imported alias)
        PE = objs["compile"]
        stock = getattr(PE, "compile_af3_style_confidence_outputs")
        bound = functools.partial(_compile_expected, stock)
        functools.update_wrapper(bound, stock)
        patches.replace(PE, "compile_af3_style_confidence_outputs", bound)
        EN = sys.modules.get("rf3.inference_engines.rf3")
        if EN is not None and getattr(EN, "compile_af3_style_confidence_outputs", None) is stock:
            patches.replace(EN, "compile_af3_style_confidence_outputs", bound)
    if objs.get("templ_feats") is not None:                                               # a module-level function: the Transform of the pipeline calls it by name
        GT = objs["templ_feats"]
        stock_feat = getattr(GT, _templ.FEATURIZER)
        bound_feat = functools.partial(_templ.distogram_condition_precursors, stock_feat)
        functools.update_wrapper(bound_feat, stock_feat)
        patches.replace(GT, _templ.FEATURIZER, bound_feat)
    for site, kind in (("metric_ptm", "ptm"), ("metric_iptm", "iptm")):
        if objs.get(site) is not None:
            _wrap(patches, objs[site], "compute", site, _tm_metrics(kind))
    CTX["featurise_site"] = objs.get("featurise") is not None                           # the engine base's pipeline constructor: rank 0 featurises, ranks > 0 receive (rankdata.broadcast_features);
    if CTX["featurise_site"]:                                                              # data_form=rank0_bcast is recorded when the seam ENGAGES (per item), never on the install alone
        _wrap(patches, objs["featurise"], "_construct_pipeline", "featurise", _construct_pipeline_rank0)
    else:
        CTX["absent_sites"].append("featurise:BaseInferenceEngine._construct_pipeline (every rank featurises: data_form=per_rank; the entry sync carries rank 0's features)")
        _rp("evidence").record_schedule(data_form=_rp("rankdata").check_data_form("per_rank"))
    if objs.get("h2d") is not None:                                                       # the engine's H2D statement (lightning Fabric.to_device): ROWPAIR_MSA_HOST holds the raw MSA stack back
        _wrap(patches, objs["h2d"], "to_device", "h2d", _h2d_to_device)
    else:
        CTX["absent_sites"].append("h2d:Fabric.to_device (no H2D seam: ROWPAIR_MSA_HOST is refused by name at the model entry)")
    if objs.get("noise_init") is not None:
        _wrap(patches, objs["noise_init"], "_get_initial_structure", "noise_init", _noise_init_guard)
        if hasattr(objs["noise_init"], "_predraw"):
            _wrap(patches, objs["noise_init"], "_predraw", "noise_predraw", _predraw_guard)
        else:
            CTX["absent_sites"].append("noise_predraw:_predraw (stock sampler: _get_initial_structure is the guarded draw)")
    lv = objs.get("pair_windows")
    if lv is not None:
        if not hasattr(lv, "PAIR_WINDOWS"):
            raise rp.RowpairRefused("levers.PAIR_WINDOWS is absent: this tree's levers.py predates the token-pair window seam")
        patches.replace(lv, "PAIR_WINDOWS", pair_windows_banded)


def _wrap(patches, cls, attr: str, site: str, sharded: Callable) -> None:
    """``cls.<attr>`` -> ``sharded(self, prev, *a, **k)`` (prev = the callable found now; the sharded body decides by the active layout)."""
    prev = getattr(cls, attr)

    def bound(self, *args, **kwargs):
        return sharded(self, prev, *args, **kwargs)

    bound.__name__ = f"{getattr(cls, '__name__', 'obj')}_{attr}_rowpair"
    bound.__qualname__ = bound.__name__
    bound.__wrapped__ = prev
    patches.replace(cls, attr, bound)


# --------------------------------------------------------------------------------------------------------------------- shared helpers
def _layout_of(N: int):
    """The refusing, aligned layout of pair size N for this rank (dist.ctx: never a bare Layout, never grid-replicated)."""
    dist = _rp("dist")
    lay = dist.ctx(int(N), align=ALIGN)
    dist.require_sharded(lay, "rosettafold3 rowpair")
    if "trunk" not in CTX["layouts"] or CTX["layouts"]["trunk"].get("N") != lay.N:
        CTX["layouts"]["trunk"] = {"N": lay.N, "P": lay.P, "R": lay.R, "r0": lay.r0, "r1": lay.r1, "align": lay.align, "bounds": list(lay.bounds)}
    CTX["layout"] = lay
    return lay


def _is_shard(z, lay) -> bool:
    return lay is not None and z is not None and z.dim() == 3 and int(z.shape[0]) == lay.R and int(z.shape[1]) == lay.N and lay.R != lay.N


def _sync(tag: str, d: dict) -> dict:
    """Make the replicated tensors of ``d`` rank 0's on every rank (``bcast.sync_tensordict_from_rank0``: bit movement) and record its census
    (``synced`` leaves broadcast, ``skipped`` host-resident leaves left alone, per tag) for the LEVER line."""
    census = _rp("bcast").sync_tensordict_from_rank0(d, tag=tag, strict=False)
    s = CTX["sync"].setdefault(tag, {"calls": 0, "synced": 0, "checked": 0, "skipped": 0, "realloc": 0})
    s["calls"] += 1
    for k in ("synced", "checked", "realloc"):
        s[k] += int(census.get(k, 0) or 0)
    s["skipped"] += len(census.get("skipped") or [])
    return d


PAIR_INPUT_KEYS = ("token_bonds", "distogram_condition", "has_distogram_condition",   # RF3's N x N input feature keys: the only dict
                   "template_distogram", "template_unit_vector")                     # entries the reach below rewrites outside ``f``


def _shard_pair_features(f: dict, lay) -> None:
    """Every ``[N, N, ...]`` tensor of the model's feature dict ``f`` (``token_bonds``, ``distogram_condition``, ``has_distogram_condition``
    — the N²-shaped INPUTS the model reads) becomes this rank's rows ``[R, N, ...]`` in place of the dict entry: their only consumers under
    n_gpu>1 are this module's row statements (:func:`_pf_rows` indexes local rows). The same rewrite reaches the ENGINE's copies: on the stock
    route the features arrive already cast and the engine keeps its pre-cast batch dict resident beside ``f`` (an fp32 ``[N, N, 64]``
    distogram condition among them) — every dict outside ``f`` that holds an ``[N, N, ...]`` tensor of this device under a key of
    ``PAIR_INPUT_KEYS`` (found through the collector) gets this rank's rows too, so no whole N×N input stays device-resident on any rank when
    its holders are dicts (a holder that is not a dict is left and shows in ``nxn_whole_resident``). Idempotent (a shard is left alone);
    census: ``entry_pair_feats`` / ``entry_pair_feats_gib`` (``f``), ``engine_pair_feats`` / ``engine_pair_feats_gib`` (outside ``f``)."""
    import gc
    torch = _torch()
    N, r0, r1 = int(lay.N), int(lay.r0), int(lay.r1)

    def is_pair(v):
        return isinstance(v, torch.Tensor) and v.dim() >= 2 and int(v.shape[0]) == N and int(v.shape[1]) == N and (r1 - r0) < N

    keys, freed, dev = [], 0, None
    for k in list(f.keys()):
        v = f[k]
        if isinstance(v, torch.Tensor) and dev is None:
            dev = v.device
        if not is_pair(v):
            continue
        rows = v[r0:r1].contiguous().clone()
        freed += (v.numel() - rows.numel()) * v.element_size()
        f[k] = rows
        keys.append(k)
        del v
    mine = {id(v) for v in f.values() if isinstance(v, torch.Tensor)}
    ekeys, efreed = [], 0
    for obj in gc.get_objects():                                                             # the engine's pre-cast batch: dict entries outside f
        if not is_pair(obj) or id(obj) in mine or (dev is not None and obj.device != dev):
            continue
        for ref in gc.get_referrers(obj):
            if not isinstance(ref, dict) or ref is f:
                continue
            for k in [k for k, v in list(ref.items()) if v is obj and k in PAIR_INPUT_KEYS]:
                rows = obj[r0:r1].contiguous().clone()
                efreed += (obj.numel() - rows.numel()) * obj.element_size()
                ref[k] = rows
                ekeys.append(str(k))
        del obj
    acc = CTX.setdefault("pair_feats", {})                                                  # cumulative over the process's entries (a later entry finds
    acc["keys"] = sorted(set(acc.get("keys", [])) | set(keys))                               # rows already and adds nothing)
    acc["engine_keys"] = sorted(set(acc.get("engine_keys", [])) | set(ekeys))
    acc["gib"] = round(acc.get("gib", 0.0) + freed / 2 ** 30, 3)
    acc["engine_gib"] = round(acc.get("engine_gib", 0.0) + efreed / 2 ** 30, 3)
    _rp("evidence").record_schedule(entry_pair_feats=",".join(acc["keys"]) or "none", entry_pair_feats_gib=acc["gib"],
                                    engine_pair_feats=",".join(acc["engine_keys"]) or "none", engine_pair_feats_gib=acc["engine_gib"])


def _nxn_whole_census(lay, device) -> None:
    """Once per process, at the first pair statement: how many WHOLE ``[N, N, ...]`` tensors (any dtype, any holder) are resident on this
    rank's device and their GiB — ``nxn_whole_resident`` / ``nxn_whole_resident_gib`` in the schedule census (0 is the design)."""
    import gc
    torch = _torch()
    N = int(lay.N)
    n, b, shapes = 0, 0, []
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.dim() >= 2 and int(obj.shape[0]) == N and int(obj.shape[1]) == N and obj.device == device:
            n += 1
            b += obj.numel() * obj.element_size()
            shapes.append(f"{list(obj.shape)}:{str(obj.dtype).replace('torch.', '')}".replace(" ", ""))
    _rp("evidence").record_schedule(nxn_whole_resident=n, nxn_whole_resident_gib=round(b / 2 ** 30, 3), nxn_whole_shapes=";".join(shapes[:6]) or "none")


def _pf_rows(x, g0: int, g1: int, lay, device=None, row_dim: int = 0):
    """GLOBAL pair rows ``[g0, g1)`` of a pair feature that is either whole (``[N, N, ...]``: :func:`trunk.feature_rows`) or this rank's
    shard (``[R, N, ...]`` after :func:`_shard_pair_features`: local rows ``[g0 - r0, g1 - r0)``)."""
    trunk = _rp("trunk")
    n0 = int(x.shape[row_dim])
    if n0 == int(lay.N):
        return trunk.feature_rows(x, g0, g1, device, row_dim=row_dim - x.dim() if row_dim >= 0 else row_dim)
    r0, r1 = int(lay.r0), int(lay.r1)
    if n0 != r1 - r0 or not (r0 <= g0 <= g1 <= r1):
        raise _rp().RowpairRefused(f"refused: pair feature rows [{g0},{g1}) outside this rank's shard [{r0},{r1}) (feature rows {n0}, N {lay.N})")
    slab = x.narrow(row_dim, g0 - r0, g1 - g0)
    torch = _torch()
    return slab if device is None or slab.device == torch.device(device) else slab.to(device)


def _relpos_rows(rpe, f, g0: int, g1: int):
    """RF3's ``RelativePositionEncoding.forward`` for GLOBAL pair rows ``[g0, g1)``: the four one-hot terms of the stock statement on the
    row slab (``trunk.relpos_onehot_rows`` / ``same_rows``), its linear last. Cyclic chains are refused by name."""
    torch = _torch()
    trunk = _rp("trunk")
    if len(f.get("cyclic_asym_ids", [])) > 0:
        raise _rp().RowpairRefused("refused: cyclic chains (cyclic_asym_ids) under n_gpu>1 — the cyclic relative-offset rows are not installed")
    same_chain = trunk.same_rows(f["asym_id"], g0, g1)
    same_res = trunk.same_rows(f["residue_index"], g0, g1)
    same_entity = trunk.same_rows(f["entity_id"], g0, g1)
    a_pos = trunk.relpos_onehot_rows(f["residue_index"], g0, g1, rpe.r_max, condition=same_chain, dtype=torch.int64)
    a_tok = trunk.relpos_onehot_rows(f["token_index"], g0, g1, rpe.r_max, condition=same_chain & same_res, dtype=torch.int64)
    a_chain = trunk.relpos_onehot_rows(f["sym_id"], g0, g1, rpe.s_max, condition=same_entity, dtype=torch.int64)
    feats = torch.cat([a_pos, a_tok, same_entity.unsqueeze(-1).to(torch.int64), a_chain], dim=-1).to(rpe.linear.weight.dtype)
    return rpe.linear(feats)


# ------------------------------------------------------------------------------------------------------------ rank-0 featurisation
FEATS_WHAT = "feats"                               # the word of the digest gate / refusals (rankdata: feats_ranks_equal, feats_ranks_differ, feats_rank0_failed, feats_bcast_malformed)


def _feats_wire_key() -> str:
    """The raw MSA stack's TOP-LEVEL name on the wire (``feats.msa_stack``): ``rankdata.broadcast_features`` skips, places and the digest
    excludes top-level keys, so rank 0 hoists the stack out of ``feats`` for the transfer and every rank puts it back (:func:`_feats_unwire`)."""
    return "feats." + MSA_STACK_KEY


def _feats_wire(example: dict) -> dict:
    """Rank 0's featurised example in wire form: a shallow copy with the raw MSA stack hoisted to :func:`_feats_wire_key`; the example itself
    is left as it was (rank 0 keeps using it)."""
    wire = dict(example)
    feats = wire.get("feats")
    if isinstance(feats, dict) and MSA_STACK_KEY in feats:
        feats = dict(feats)
        wire[_feats_wire_key()] = feats.pop(MSA_STACK_KEY)
        wire["feats"] = feats
    return wire


def _feats_unwire(wire: dict, skipped: dict) -> dict:
    """A received wire in the engine's form again: the raw MSA stack back under ``feats`` — the received tensor, or, when it did not travel
    (``ROWPAIR_MSA_HOST`` set: ``skipped`` names its shape / dtype), the zero-row placeholder that carries its ``n_recycle``
    (``msa_host.place_skipped``), which the H2D seam holds as such (:func:`_h2d_to_device`)."""
    example = dict(wire)
    key = _feats_wire_key()
    if key in skipped:
        _rp("msa_host").place_skipped(example, {key: skipped[key]}, row_dims={key: 0})
    if key in example:
        feats = dict(example.get("feats") or {})
        feats[MSA_STACK_KEY] = example.pop(key)
        example["feats"] = feats
    return example


def _item_id(args) -> Optional[str]:
    """The example id of a featurisation call's input (``InferenceInput.to_pipeline_input()``: a dict carrying ``example_id``), or None."""
    inp = args[0] if args else None
    eid = inp.get("example_id") if isinstance(inp, dict) else None
    return str(eid) if eid is not None else None


def _eq_word(rd, identical) -> Dict[str, str]:
    """``{feats_ranks_equal: yes|no|n/a}`` — the core's word (``rankdata.agree_word``) as a schedule / line field."""
    k, v = rd.agree_word(FEATS_WHAT, identical).split("=", 1)
    return {k: v}


def _feats_line(n: int, rank: int, item: Optional[str], words: Dict[str, object]) -> None:
    """``[rosettafold3-opt] FEATS item=<n> rank=<r> example=<id> data_form=rank0_bcast feats_digest=<16 hex> feats_items=<n> feats_digest_excludes=<keys|none>
    feats_unhashed=<…|none> feats_ranks_equal=yes|no`` — once per featurised item on every rank's stderr (the rank log)."""
    body = " ".join(f"{k}={v}" for k, v in words.items())
    sys.stderr.write(f"[{_report_prefix()}] FEATS item={n} rank={rank} example={item if item is not None else '-'} {body}\n")   # one write: a line stays whole among concurrent writers
    sys.stderr.flush()


def _construct_pipeline_rank0(self, prev, *args, **kwargs):
    """``BaseInferenceEngine._construct_pipeline`` under n_gpu>1: the engine builds its Transform pipeline as stock on every rank (the object;
    ranks > 0 never call it), then ``self.pipeline`` is :func:`_features_from_rank0` around it."""
    out = prev(self, *args, **kwargs)
    if not CTX["installed"]:
        return out
    pipeline = getattr(self, "pipeline", None)
    if not callable(pipeline):                                                              # no silent per-rank featurisation under an n_gpu>1 install: named
        raise _rp().RowpairRefused(f"refused: featurise: BaseInferenceEngine._construct_pipeline left self.pipeline={type(pipeline).__name__} under n_gpu>1 "
                                   "(a callable Transform pipeline is what rank 0 runs and ranks > 0 skip; nothing is improvised here)")
    if not getattr(pipeline, "_rowpair_rank0", False):
        self.pipeline = _features_from_rank0(pipeline)
    return out


def _features_from_rank0(pipeline: Callable) -> Callable:
    """The engine's featurisation call (``pipeline_output = self.pipeline(...)``, inference_engines/rf3.py:545) under n_gpu>1: rank 0 runs
    ``pipeline``; every rank returns rank 0's featurised example — ``rankdata.broadcast_features`` (the store rendezvous first, so ranks > 0
    wait outside any collective however long rank 0 featurises; rank 0's host RNG state after featurising is adopted by the receivers, so the
    replicated draws downstream agree; with ``ROWPAIR_MSA_HOST`` set the raw MSA stack does not travel: ranks > 0 hold its zero-row
    placeholder — mode ``rank0`` for good, mode ``all`` until the model entry streams rank 0's stack into it), then the digest gate on every rank (``rankdata.assert_ranks_agree``, refusing: ``refused: feats_ranks_differ: …``). A
    featurisation that raised on rank 0 is ``refused: feats_rank0_failed: <Type>: <msg>`` on every rank; a receiver whose own parsed item is not
    the one rank 0 featurised (rank 0 names it in the status word) is ``refused: feats_item_mismatch: …``. Census: one ``FEATS item=<n> rank=<r>
    example=<id> data_form=rank0_bcast feats_digest=<16 hex> … feats_ranks_equal=yes|no`` line per item on every rank (:func:`_feats_line`), the
    same words on the rank's ``LEVER name=rowpair`` line and in its exit tally (schedule) + the core's ``feats_*`` / ``msa_host_*`` facts;
    ``calls_featurise`` = ``1s/0r`` on rank 0, ``0s/1r`` on a receiver, per item."""

    def features_from_rank0(*args, **kwargs):
        rd, ev = _rp("rankdata"), _rp("evidence")
        rank = int(_rp("dist").world()[1])
        book = CTX.setdefault("featurise_calls", {})
        book[rank] = n = int(book.get(rank, 0)) + 1                                        # the item's ordinal on this rank: the rendezvous key is unique per call, equal across ranks
        key = f"{FEATS_WHAT}/{n}"
        md = _msa_host_mode()
        skip = (_feats_wire_key(),) if md is not None else ()                                 # ROWPAIR_MSA_HOST set (rank0 | all): the raw MSA stack does not travel here — rank0: ranks > 0 never hold it;
                                                                                            # all: msa_host.sync_host_features_ streams rank 0's into every rank's host copy at the model entry, in pieces
        item = _item_id(args)                                                               # this rank's own example id (the per-rank input parse): rank 0 names its item in the status word, receivers hold theirs against it
        ev.record_schedule(data_form=rd.check_data_form("rank0_bcast"))                     # recorded when the seam engages, per item
        source = rank == 0
        if source:
            try:
                example = pipeline(*args, **kwargs)
            except Exception as exc:                                                        # the receivers wait on this key: they get the failure word and every rank, this one included, raises
                rd.broadcast_features(None, key=key, status=rd.status_word(exc), what=FEATS_WHAT)   # refused: feats_rank0_failed (chained to exc, whose traceback stays in rank 0's log)
                raise                                                                       # not reached: broadcast_features raises on a failed status; kept so a featurisation error can never pass silently
            wire = _feats_wire(example)
            fb = rd.broadcast_features(wire, key=key, status=rd.status_word(extra=item or ""), skip_keys=skip, what=FEATS_WHAT)
        else:
            fb = rd.broadcast_features(None, key=key, skip_keys=skip, what=FEATS_WHAT)
            wire = fb.feats
        _count("featurise", "sharded" if source else "replicated")
        dg = rd.digest_features(wire, exclude=tuple(fb.skipped))                            # what every rank holds, minus the stack that did not travel (by design: msa_host)
        words = {"data_form": "rank0_bcast", f"{FEATS_WHAT}_digest": dg.digest[:rd.DIGEST_HEX], f"{FEATS_WHAT}_items": n,
                 f"{FEATS_WHAT}_digest_excludes": ",".join(sorted(fb.skipped)) or "none", f"{FEATS_WHAT}_unhashed": ",".join(dg.unhashed) or "none"}
        try:
            same = rd.assert_ranks_agree(dg.digest, what=FEATS_WHAT, mode="refuse")         # refused: feats_ranks_differ on every rank when a receiver's bytes are not rank 0's
        except _rp().RowpairRefused:
            _feats_line(n, rank, item, dict(words, **_eq_word(rd, False)))                  # every rank's FEATS line names its digest; the refusal names the pair that differs
            raise
        words.update(_eq_word(rd, same))
        _feats_line(n, rank, item, words)
        ev.record_schedule(**words)
        if source:
            return example
        sent = str(fb.status or "").split(None, 1)
        sent_item = sent[1] if len(sent) > 1 else None
        if item is not None and sent_item is not None and sent_item != item:               # this rank parsed another item than rank 0 featurised (input order / count differ across ranks): named, never folded
            raise _rp().RowpairRefused(f"refused: {FEATS_WHAT}_item_mismatch: rank {rank} holds item {item!r} at featurisation {n}, rank 0 sent {sent_item!r}")
        example = _feats_unwire(wire, fb.skipped)
        if fb.skipped and md is not None:                                                   # the receiver's placement words (place_skipped speaks for mode rank0 and the wire key)
            ev.record_schedule(msa_host_mode=md, msa_host_where=f"{MSA_STACK_KEY}:placeholder")
        return example

    features_from_rank0._rowpair_rank0 = True
    features_from_rank0.__wrapped__ = pipeline
    return features_from_rank0


# --------------------------------------------------------------------------------------------------------------- host-resident MSA
MSA_STACK_KEY = "msa_stack"                        # RF3's raw MSA features [n_recycle, S, I, c] (data pipeline: FeaturizeMSALikeAF3, one i.i.d. subsample per recycle);
MSA_KEY = "msa"                                    # the model's per-cycle selection f["msa"] = f["msa_stack"][i_cycle] (RF3.py:264)
MSA_TOKEN_DIM = 2                                  # the token dim of msa_stack: the placeholder the model carries is zero THERE (dim 0 = n_cycle is read by the trainer, dims 1 / 3 = S / c declare the cycle rows' shape)


def _msa_host_mode():
    """``ROWPAIR_MSA_HOST`` in the core's placement words (``msa_host.host_mode``): None (off) | ``all`` | ``rank0``."""
    return _rp("msa_host").host_mode()


def _is_pipeline_output(obj) -> bool:
    """The engine's featurised example on its way to the device (``RF3InferenceEngine.run``: ``trainer.fabric.to_device(pipeline_output)``):
    a mapping whose ``feats`` mapping holds the raw MSA stack as a HOST tensor."""
    torch = _torch()
    feats = obj.get("feats") if isinstance(obj, dict) else None
    t = feats.get(MSA_STACK_KEY) if isinstance(feats, dict) else None
    return torch.is_tensor(t) and not t.is_cuda and t.dim() == 4


def _h2d_to_device(self, prev, obj, *args, **kwargs):
    """``Fabric.to_device`` under n_gpu>1: with ``ROWPAIR_MSA_HOST`` set and ``obj`` the engine's featurised example, the raw MSA stack
    ``feats["msa_stack"] [n_recycle, S, I, c]`` is held back from the move — ``msa_host.park_features``: rank 0's copy on PINNED host
    (``rank0``; every rank's under ``all``), the zero-row placeholder on ranks > 0 — and kept for the item (:data:`CTX` ``msa_host``); the
    entry the model reads becomes the zero-TOKEN placeholder ``[n_recycle, S, 0, c]`` (a host tensor: the entry sync leaves it alone), so the
    stock statements that touch it (the autocast cast RF3.py:394-404, the per-cycle selection RF3.py:264, the trainer's ``n_cycle =
    msa_stack.shape[0]``) run unchanged and for free; the cycle's rows reach the device in :func:`_msa_rows`. Anything else (a module, a
    mapping without a host ``msa_stack``, the lever off) is the stock move."""
    if CTX["installed"] and isinstance(obj, dict) and isinstance(obj.get("feats"), dict) and obj.get("example_id") is not None:
        CTX["example_id"] = str(obj["example_id"])                                        # the item's query id (ROWPAIR_CKPT_DIR names its checkpoint tree <example_id>_<seed>)
    md = _msa_host_mode() if CTX["installed"] else None
    if md is None or not _is_pipeline_output(obj):
        return prev(self, obj, *args, **kwargs)
    MH = _rp("msa_host")
    feats_in = obj["feats"]
    stack = feats_in.pop(MSA_STACK_KEY)
    try:
        out = prev(self, obj, *args, **kwargs)
    finally:
        feats_in[MSA_STACK_KEY] = stack                                                    # the caller's mapping is left as it was
    feats = out["feats"] if isinstance(out, dict) and isinstance(out.get("feats"), dict) else None
    if feats is None:
        raise _rp().RowpairRefused(f"refused: {MH.ENV_MSA_HOST}={md}: Fabric.to_device returned {type(out).__name__} without a 'feats' mapping "
                                   "(the H2D seam cannot place the MSA placeholder)")
    held = {MSA_STACK_KEY: stack}
    if MH.where(stack) == "placeholder":                                                   # a received example (site featurise): the zero-row placeholder stands in already, placed and recorded at receipt (mode all: filled at the model entry)
        facts = {"parked": [], "placeholders": [MSA_STACK_KEY]}
    else:
        facts = MH.park_features(held, (MSA_STACK_KEY,), mode=md, row_dims={MSA_STACK_KEY: 0}, log=_rp("dist").comm().verbose)   # rank 0 | all: the pinned host copy; ranks > 0 (rank0): [0, S, I, c]
    shape = (MH.parked_rows(stack, 0),) + tuple(int(x) for x in stack.shape[1:])           # [n_recycle, S, I, c]: a placeholder carries the n_recycle it stands in for
    CTX["msa_host"] = {"mode": md, "feats": held, "shape": shape, "dtype": str(stack.dtype).replace("torch.", ""), "site": "h2d",
                       "parked": list(facts.get("parked") or []), "placeholders": list(facts.get("placeholders") or [])}
    feats[MSA_STACK_KEY] = MH.host_placeholder_of(shape, stack.dtype, MSA_TOKEN_DIM)       # [n_recycle, S, 0, c] (host): what the model's own statements read
    del stack
    return out


def _msa_host_entry(f: dict, device, held: Optional[dict] = None) -> None:
    """Model entry under ``ROWPAIR_MSA_HOST``: the H2D seam must have engaged for this item — ``f["msa_stack"]`` is the zero-token placeholder
    and the item's host stack (or its rank-0 placeholder) is held — else the run is refused by name (a whole device-resident stack under the
    lever would be a silent no-op; there is no D2H fallback). Mode ``all``: every rank's host copy becomes rank 0's
    (``msa_host.sync_host_features_``, streamed through the device) — the entry sync's job for a host-resident feature; a receiver's
    zero-row placeholder is filled there. ``held``: the item's host-resident record (default: this process's current item, :data:`CTX`
    ``msa_host``). Census ``msa_host``."""
    torch = _torch()
    ev = _rp("evidence")
    md = _msa_host_mode()
    held = CTX.get("msa_host") if held is None else held                                  # the item's host-resident record (this process's current item unless given)
    if md is None:
        if held is not None:
            raise _rp().RowpairRefused("refused: an MSA stack is held on the host but ROWPAIR_MSA_HOST reads off at the model entry (the lever changed mid-item)")
        ev.record_schedule(msa_host="off")
        return
    MH = _rp("msa_host")
    t = f.get(MSA_STACK_KEY)
    is_placeholder = torch.is_tensor(t) and t.dim() == 4 and int(t.shape[MSA_TOKEN_DIM]) == 0
    if held is None or not is_placeholder:
        where = (f"{tuple(t.shape)} on {t.device}" if torch.is_tensor(t) else type(t).__name__)
        raise _rp().RowpairRefused(f"refused: {MH.ENV_MSA_HOST}={md} but the raw MSA stack reached the model as {where} (held={held is not None}): the engine's H2D "
                                   "seam (Fabric.to_device of the featurised example) did not engage for this item — no host placement happened and none is "
                                   f"improvised here; unset {MH.ENV_MSA_HOST} to run with the device-resident stack")
    if held["mode"] != md:
        raise _rp().RowpairRefused(f"refused: {MH.ENV_MSA_HOST}={md} at the model entry, {held['mode']} at the H2D seam (one placement per item)")
    n_rec, S, _, c = (int(x) for x in t.shape)
    hs = held["feats"][MSA_STACK_KEY]
    if int(hs.shape[1]) != S or int(hs.shape[3]) != c or (int(hs.shape[0]) not in (0, n_rec)):
        raise _rp().RowpairRefused(f"refused: the held MSA stack {tuple(hs.shape)} and the model's placeholder {tuple(t.shape)} disagree (n_recycle / S / c)")
    if md == "all":
        MH.sync_host_features_(held["feats"], (MSA_STACK_KEY,), device, src=0, mode=md)
    ev.record_schedule(msa_host=md, msa_host_site=held["site"], msa_host_stack="x".join(str(x) for x in held["shape"]) + ":" + held["dtype"])


def _msa_rows(f: dict, S_inputs_I, lay):
    """The cycle's raw MSA rows ``[S, I, c]`` on the device, in the working dtype. Lever off: the tensor the model selected (RF3.py:264,
    device-resident). ``ROWPAIR_MSA_HOST``: cycle ``i``'s rows of the held host stack moved and THEN cast to the placeholder's dtype (the
    dtype the stock autocast cast decided) — ``msa_host.rows_to_device``: under ``rank0`` rank 0 moves them and ranks > 0 receive them by
    chunked broadcast; ``i`` = this item's recycle index (:data:`CTX` ``cycle``: RF3's loop runs one ``Recycler.forward`` per ``i_cycle``).
    Elementwise ``select -> cast == cast -> select``: the values are the stock statement's."""
    held = CTX.get("msa_host")
    ph = f[MSA_KEY]
    if held is None:
        return ph
    MH = _rp("msa_host")
    torch = _torch()
    if not (torch.is_tensor(ph) and ph.dim() == 3 and int(ph.shape[1]) == 0):
        raise _rp().RowpairRefused(f"refused: ROWPAIR_MSA_HOST={held['mode']}: f['msa'] is {tuple(ph.shape) if torch.is_tensor(ph) else type(ph).__name__}, not the cycle's "
                                   "zero-token placeholder [S, 0, c] (the model's per-cycle selection did not run on the placeholder)")
    i = int(CTX["cycle"])
    src = held["feats"][MSA_STACK_KEY]                                                     # [n_rec, S, I, c] pinned host (rank 0 | all) or [0, S, I, c] (ranks > 0 under rank0)
    n_rec, S, c = int(held["shape"][0]), int(ph.shape[0]), int(ph.shape[2])
    if i >= n_rec:
        raise _rp().RowpairRefused(f"refused: recycle index {i} >= the MSA stack's n_recycle {n_rec} (one Recycler.forward per cycle expected)")
    dev, dtype = S_inputs_I.device, ph.dtype

    def build():                                                                            # rank 0 (rank0) / every rank (all): select + cast ON THE HOST (a working-dtype temp:
        rows = src[i] if src.dtype == dtype else src[i].to(dtype=dtype)                     # identical values to the stock device cast, half the H2D bytes, no fp32 device
        return rows.to(device=dev, non_blocking=(src.dtype == dtype)).contiguous()           # transient), then move (stream-ordered from the pinned copy; a cast temp is pageable)

    return MH.rows_to_device(build, shape=(S, int(lay.N), c), dtype=dtype, device=dev, mode=held["mode"], name=f"msa_rows.cycle{i}")


# ------------------------------------------------------------------------------------------------------------------------ entry
def _entry_forward(self, prev, input, *args, **kwargs):
    """Model entry (``RF3WithConfidence.forward`` / ``RF3.forward``): every rank's input dict becomes rank 0's (features, coordinates to noise,
    indices), so every replicated statement downstream reads identical operands. Then the kit's forward (the big levers' wrappers included)."""
    if not CTX["installed"]:
        return prev(self, input, *args, **kwargs)
    if self.training:
        raise _rp().RowpairRefused("refused: training mode under n_gpu>1 (inference only)")
    _count("entry")
    CTX["item"] += 1                                                                       # the item's ordinal on this rank (a per-process census counts the item once over its recycles: _templ_census)
    if CTX.get("featurise_site") and not CTX["calls"].get("featurise"):                  # the featurise seam is bound but no featurisation passed through it before the model
        raise _rp().RowpairRefused("refused: feats_site_unengaged: the model entry runs but the rank-0 featurisation seam (BaseInferenceEngine._construct_pipeline "
                                   "-> self.pipeline) never engaged in this process — the featurised example did not come through rankdata.broadcast_features, "
                                   "so data_form=rank0_bcast cannot be claimed; nothing is improvised here")   # entry ran: the engine featurised somewhere the seam does not cover
    if not CTX.get("params_guarded"):                                                      # once per process: the replicated WEIGHTS are identical on
        torch = _torch()                                                                   # every rank (per-parameter sums, one small all-gather), else
        sums = torch.stack([p.detach().float().sum() for p in self.parameters()]).cpu()   # refused by name — a desynced checkpoint never folds
        _rp("trunk").guard_replicated(sums, "entry.parameters")
        CTX["params_guarded"] = True
    _sync("entry", input)
    if callable(kwargs.get("should_early_stop_fn")):                                    # ROWPAIR_CKPT_DIR resume: the recycle-0 probe's decision is REPLAYED (_replayed_early_stop)
        kwargs = dict(kwargs, should_early_stop_fn=_replayed_early_stop(kwargs["should_early_stop_fn"]))
    elif len(args) >= 3 and callable(args[2]):
        args = tuple(args[:2]) + (_replayed_early_stop(args[2]),) + tuple(args[3:])
    f_in = input.get("f") if isinstance(input, dict) else None
    t_in = input.get("t") if isinstance(input, dict) else None
    CTX["n_samples"] = int(t_in.shape[0]) if (_torch().is_tensor(t_in) and t_in.dim() >= 1) else None   # D: the confidence stage's pass count (diffusion_batch_size = input["t"].shape[0], RF3.py:406)
    _conf_plan_close()
    if isinstance(f_in, dict) and MSA_STACK_KEY in f_in:
        _msa_host_entry(f_in, next(self.parameters()).device)                              # ROWPAIR_MSA_HOST: the H2D seam engaged for this item, or refused by name
    if isinstance(f_in, dict) and ("token_bonds" in f_in or "asym_id" in f_in):
        n_tok = int((f_in["token_bonds"] if "token_bonds" in f_in else f_in["asym_id"]).shape[-1])
        _shard_pair_features(f_in, _layout_of(n_tok))                                     # N x N inputs -> this rank's rows, before anything reads them
    CTX["mem_trace"] = []
    CTX["state_syncs"] = 0
    CTX["spread_A"] = 0.0
    _mem_mark("entry")
    CTX["cycle"] = 0
    CTX["f"] = input.get("f") if isinstance(input, dict) else None                        # asym_id / is_ligand for the confidence finish
    CTX["tm"] = []                                                                         # per-sample TM scalars, read by the metrics tail
    try:
        return prev(self, input, *args, **kwargs)
    finally:
        _release_zinit()
        _conf_plan_close()
        _ckpt_finalize()
        CTX["msa_host"] = None                                                             # the item's host MSA stack (rank 0's pinned copy) is dropped with the item
        CTX["layout"] = None
        CTX["band"] = {}
        CTX["f"] = None


# ------------------------------------------------------------------------------------------------------------------------ pair init
def _feature_init_forward(self, prev, f):
    """``FeatureInitializer.forward`` with the pair representation BORN as this rank's rows: ``S_inputs`` (atom encoder) and ``S_init``
    replicated and made rank 0's; ``Z_init[r0:r1] = to_z_init_i(S)[None, :] + to_z_init_j(S)[:, None] + relpos rows + token-bond rows``."""
    if not CTX["installed"]:
        return prev(self, f)
    trunk = _rp("trunk")
    _count("pair_init")
    S_inputs_I = self.input_feature_embedder(f)
    S_init_I = self.to_s_init(S_inputs_I)
    if S_inputs_I.dim() != 2:
        raise _rp().RowpairRefused(f"refused: batched trunk under n_gpu>1 (S_inputs {tuple(S_inputs_I.shape)}; RF3 inference is unbatched)")
    _sync("s_inputs", {"S_inputs_I": S_inputs_I, "S_init_I": S_init_I})
    lay = _layout_of(int(S_inputs_I.shape[0]))
    if not CTX.get("nxn_census_done"):                                                     # once per process: whole N x N tensors resident now
        CTX["nxn_census_done"] = True
        _nxn_whole_census(lay, S_inputs_I.device)
    zi = self.to_z_init_i(S_inputs_I)                           # indexed by the column j (stock: .unsqueeze(-3))
    zj = self.to_z_init_j(S_inputs_I)                           # indexed by the row i (stock: .unsqueeze(-2))
    rpe = self.relative_position_encoding
    bonds = f["token_bonds"]
    dev = zi.device

    def rows_fn(g0: int, g1: int):
        z = trunk.outer_sum_rows(zj, zi, g0, g1)
        z = z + _relpos_rows(rpe, f, g0, g1)
        z = z + self.process_token_bonds(_pf_rows(bonds, g0, g1, lay, dev).unsqueeze(-1).to(zi.dtype))
        return z

    Z_init = trunk.init_pair_shard(lay, rows_fn, zj, channels=3 * int(zj.shape[-1]) + _n_relpos(rpe))
    _park_zinit(Z_init)                                         # ROWPAIR_PARK_ZINIT: host copy + device release; the tensor object keeps shape / dtype / device for the stock zeros_like (RF3.py:295)
    _mem_mark("pair_init")
    return S_inputs_I, S_init_I, Z_init


def _park_zinit(z) -> None:
    """The z_init shard through ``trunk.ShardPark`` (``park=None``: ``ROWPAIR_PARK_ZINIT`` decides; ``ROWPAIR_PARK_STRICT`` / ``ROWPAIR_PARK_PIN_MAX_GB``
    apply): parked = one D2H copy into the core's pinned pool and the shard's DEVICE STORAGE RELEASED at birth, each recycle's ``Z_init + process_zh(Z)``
    reading it back in row blocks (``trunk.recycle_shard_`` takes the park: :func:`_recycler_forward`); resident (``where=device``) when the lever
    is off or ``z`` is a CPU tensor. Released at the end of the trunk (:func:`_distogram_forward`) and at the item's exit. Census ``park_z_init``."""
    _release_zinit()
    park = _rp("trunk").ShardPark(z, park=None, name="z_init", log=_rp("dist").comm().log, pool=_park_pool(z))
    CTX["zinit"], CTX["zinit_tensor"] = park, z
    words = park.census_words("park_z_init")                                                 # where | gib | release, the core's vocabulary
    _rp("evidence").record_schedule(**words)


def _park_pool(t):
    """The ONE pinned host pool of this rank process for its parks (``trunk.park_pool``: budget ``ROWPAIR_PARK_PIN_MAX_GB``,
    ``ROWPAIR_PARK_STRICT``), built at the first park and kept for the process so an item's page-locked buffers are re-used by the next item,
    not re-locked; shared by the z_init park (released at the trunk's end) and the confidence stage's plan (parked after it)."""
    if CTX.get("park_pool") is None:
        CTX["park_pool"] = _rp("trunk").park_pool(pin=bool(t.is_cuda))
    return CTX["park_pool"]


def _zinit_source(Z_init_II):
    """What ``recycle_shard_`` reads ``Z_init`` from: the park of THIS item's shard (the tensor object the stock control flow hands back), else
    the tensor itself. A parked shard whose tensor is not the one handed back is refused by name (its device storage is released: nothing
    else can serve the rows)."""
    park = CTX.get("zinit")
    if park is not None and CTX.get("zinit_tensor") is Z_init_II:
        return park
    if park is not None and park.park:
        raise _rp().RowpairRefused("refused: Recycler.forward received a Z_init_II that is not this item's parked z_init shard (ROWPAIR_PARK_ZINIT=1: "
                                   "the parked shard's device storage is released; only its park serves the rows)")
    return Z_init_II


def _release_zinit() -> None:
    """Drop the z_init park (host buffer back to the pool's budget; idempotent)."""
    park = CTX.get("zinit")
    CTX["zinit"], CTX["zinit_tensor"] = None, None
    if park is not None:
        park.release()


# ------------------------------------------------------------------------------------------------------------------------ pair blocks
def _trimul_fns(m, N: int):
    """``trimul.TriMulFns`` of one RF3 ``TriangleMultiplication``: a|b = sigmoid(g_in) * p_in of LN_in rows (the b operand carries the vanilla
    route's ``/ L``; the cuEquivariance route has none), out = p_out(LN_out(x)), gate = sigmoid(g_out(LN_in(z))) — each returned in the pair
    block's own dtype (the schedule's dtype is z's; the engine's autocast would otherwise answer bf16 for an fp32 z). Under
    ``TP_TRIMUL_KERNELS = "fpf_v4"`` these callables are the named fallback of the core's row-sharded fused TriMul (:func:`_tp_trimul_fused`)."""
    torch = _torch()
    trimul = _rp("trimul")
    d = int(m.d_hidden)
    vanilla_scale = not (m.use_cuequivariance and _cueq_on())

    def proj(zb, mb, is_a):                                     # every callable returns THE PAIR BLOCK'S dtype: under the engine's autocast the
        x = m.norm_in(zb)                                        # Linears answer in bf16 whatever z is stored in (fp32 in the confidence head),
        p = m.p_in(x)                                            # and the ring driver's byte counts are z's — one dtype on every rank, every step
        g = torch.sigmoid(m.g_in(x))
        if is_a:
            out = g[..., :d] * p[..., :d]
        else:
            out = g[..., d:] * p[..., d:]
            if vanilla_scale:
                out = out / float(N)
        if mb is not None:
            out = out * mb
        return out.to(zb.dtype)

    def out_fn(x):
        return m.p_out(m.norm_out(x)).to(x.dtype)

    def gate(zb):
        return torch.sigmoid(m.g_out(m.norm_in(zb))).to(zb.dtype)

    fns = trimul.TriMulFns(proj=proj, out=out_fn, gate=gate, C_h=d)
    fused = _tp_trimul_fused(m, fns, vanilla_scale)
    return fused if fused is not None else fns


def _cueq_on() -> bool:
    fd = sys.modules.get("foundry")
    return bool(getattr(fd, "SHOULD_USE_CUEQUIVARIANCE", False))


# ------------------------------------------------------------------------------------------ P>1 triangle kernels (the core's fused row kernels)
TP_KERNELS: Dict[str, Optional[str]] = {"triatt": None, "trimul": None}   # per process: the word in force when a fused word is selected — the word itself,
                                                                         # or `torch:<reason>` when it could not be honoured (lever_fields prints them)
_TP_STATE: Dict[str, Any] = {"triatt_core": {}, "emitted": set()}


def _tp_core_module(name: str):
    """``opt_core.mem.rowpair.<name>`` when the pinned core carries it (``triatt`` with ``attention_core`` / ``trimul_fused``), else None."""
    try:
        mod = _core.load("mem.rowpair." + name)
    except Exception:  # noqa: BLE001 — a core without them: the torch words serve, named
        return None
    if name == "triatt" and not hasattr(mod, "attention_core"):
        return None
    return mod


def _tp_emit_at_exit(kind: str, mod) -> None:
    """The core's census line of the fused kernel (``LEVER name=F1.flash_triattn`` / ``LEVER name=F2.trimul_rows``, the core's grammar) printed
    once per process at exit with the kit's tag."""
    if kind in _TP_STATE["emitted"]:
        return
    _TP_STATE["emitted"].add(kind)
    import atexit
    tag = _report_prefix()
    atexit.register(lambda: (mod.emit_core_line(tag) if kind == "triatt" else mod.emit_line(tag)))


def _tp_trimul_weights(m) -> Dict[str, Any]:
    """One RF3 ``TriangleMultiplication``'s tensors under ``opt_core.trimul.WEIGHT_KEYS`` — the mapping the FPF add-on serves at P == 1
    (``rf3fpf/fpf_rf3_adapter.py`` ``_weights_fast``; tests lock the two together): LN_in = norm_in, a|b gate / projection = the first / second
    ``d_hidden`` rows of g_in / p_in, LN_out = norm_out, out projection = p_out, out gate = g_out; RF3's Linears carry no biases."""
    D = int(m.d_hidden)
    return dict(ln_in_w=m.norm_in.weight.detach(), ln_in_b=m.norm_in.bias.detach(),
                w_ag=m.g_in.weight.detach()[:D], w_ap=m.p_in.weight.detach()[:D],
                w_bg=m.g_in.weight.detach()[D:], w_bp=m.p_in.weight.detach()[D:],
                ln_out_w=m.norm_out.weight.detach(), ln_out_b=m.norm_out.bias.detach(),
                w_o=m.p_out.weight.detach(), w_og=m.g_out.weight.detach())


def _tp_trimul_fused(m, stock_fns, vanilla_scale: bool):
    """The core's row-sharded fused TriMul fns over ``stock_fns`` when ``TP_TRIMUL_KERNELS = "fpf_v4"`` and the pinned core carries
    ``mem.rowpair.trimul_fused``; None (the torch fns serve) otherwise, the reason recorded by name. The fused kernels compute cuEquivariance's
    statement (no ``1/N`` on the b operand): RF3's vanilla route keeps the torch fns (``torch:vanilla_route``)."""
    if TP_TRIMUL_KERNELS == "torch":
        return None
    if TP_TRIMUL_KERNELS not in TRIMUL_KERNELS:
        raise _rp().RowpairRefused(f"refused: TP_TRIMUL_KERNELS={TP_TRIMUL_KERNELS!r} not in {TRIMUL_KERNELS}")
    RF = _tp_core_module("trimul_fused")
    if RF is None:
        TP_KERNELS["trimul"] = "torch:core_missing(opt_core.mem.rowpair.trimul_fused)"
        return None
    if vanilla_scale:
        TP_KERNELS["trimul"] = "torch:vanilla_route"
        return None
    if float(m.norm_in.eps) != float(m.norm_out.eps):
        TP_KERNELS["trimul"] = "torch:eps_mismatch"
        return None
    TP_KERNELS["trimul"] = TP_TRIMUL_KERNELS
    _tp_emit_at_exit("trimul", RF)
    return RF.fused_trimul_fns(_tp_trimul_weights(m), stock_fns, eps=float(m.norm_in.eps), cells=None, ledger=None, min_tokens=TP_KERNEL_MIN_TOKENS)


def _tp_triatt_register(m, route_cueq: bool) -> Optional[str]:
    """Record what the current row-batch statement of this ``TriangleAttention`` is — the cuEquivariance call or RF3's einsum (one route per
    process) and ``m.scaling`` per head width D — so :func:`_triatt_stock_core` reproduces it from (q, k, v, bias) alone. Returns None, or the
    reason the module keeps its current branch outside the core (``mixed_routes`` / ``mixed_scaling``: never in RF3, refused by name if ever)."""
    st = _TP_STATE.setdefault("triatt_stock", {"cueq": bool(route_cueq), "scaling": {}})
    if st["cueq"] != bool(route_cueq):
        return "mixed_routes"
    D = int(m.to_q.weight.shape[0]) // int(m.h)
    if st["scaling"].setdefault(D, float(m.scaling)) != float(m.scaling):
        return "mixed_scaling"
    return None


def _tp_triatt_core(m=None, route_cueq: bool = False):
    """The core's attention core per row block for ``TP_TRIATT_KERNEL`` in {flash_triattn, cueq} (one per process; ``m`` given = bind this
    module's stock facts first, :func:`_tp_triatt_register`): q / k / v as ``[B, rows, H, N, D]`` views of the module's projections (q
    unscaled, the kernel applies ``D ** -0.5``), the triangle bias the one bias (no pair mask in RF3), and the current statement
    (:func:`_triatt_stock_core`: the cuEquivariance call when the engine engages it, RF3's einsum otherwise) its named fallback — below
    ``TP_KERNEL_MIN_TOKENS`` every call is that fallback (``below_gate``). None when the word is ``torch``, the pinned core lacks
    ``attention_core``, or the module's facts do not match the process's (the reason recorded by name; the current branch serves)."""
    if TP_TRIATT_KERNEL == "torch":
        return None
    if TP_TRIATT_KERNEL not in TRIATT_KERNELS:
        raise _rp().RowpairRefused(f"refused: TP_TRIATT_KERNEL={TP_TRIATT_KERNEL!r} not in {TRIATT_KERNELS}")
    if m is not None:
        why = _tp_triatt_register(m, route_cueq)
        if why:
            TP_KERNELS["triatt"] = "torch:" + why
            return None
    cache = _TP_STATE["triatt_core"]
    if TP_TRIATT_KERNEL not in cache:
        RA = _tp_core_module("triatt")
        if RA is None:
            TP_KERNELS["triatt"] = "torch:core_missing(opt_core.mem.rowpair.triatt.attention_core)"
            cache[TP_TRIATT_KERNEL] = None
        else:
            TP_KERNELS["triatt"] = TP_TRIATT_KERNEL
            cache[TP_TRIATT_KERNEL] = RA.attention_core(_triatt_stock_core, kernel=TP_TRIATT_KERNEL, min_tokens=TP_KERNEL_MIN_TOKENS, ledger=None,
                                                        scale=None, layout="bnhsd", mask_from="none", tri_bias="bias0")
            _tp_emit_at_exit("triatt", RA)
    return cache[TP_TRIATT_KERNEL]


def _triatt_qkv(m, pair):
    """The row batch's q / k / v (``pair [B, r, N, C]`` LayerNorm'd; q unscaled), each as a ``[B, r, H, N, D]`` view of the module's own
    ``[B, r, N, H, D]`` projection — the layout ``_forward_cuequivariance`` hands cuEquivariance and a free view of ``_forward_vanilla``'s."""
    B, r, N = int(pair.shape[0]), int(pair.shape[1]), int(pair.shape[2])
    query = m.to_q(pair).reshape(B, r, N, m.h, -1)
    key = m.to_k(pair).reshape(B, r, N, m.h, -1)
    value = m.to_v(pair).reshape(B, r, N, m.h, -1)
    return query.permute(0, 1, 3, 2, 4), key.permute(0, 1, 3, 2, 4), value.permute(0, 1, 3, 2, 4)


def _triatt_bias_view(bias):
    """to_b's ``[B, N, N, H]`` triangle bias as the ``[B, 1, H, N, N]`` view an attention core (and cuEquivariance) takes."""
    return bias.permute(0, 3, 1, 2).unsqueeze(1)


def _triatt_stock_core(q, k, v, biases):
    """The current row-batch attention statement as an attention core over ``q / k / v [B, r, H, N, D]`` (q unscaled) and ``biases = (the triangle
    bias view [B, 1, H, N, N],)`` -> ``o [B, r, H, N, D]``, from the facts bound by :func:`_tp_triatt_register`: on the cuEquivariance route
    ``cuequivariance_torch.triangle_attention(q, k, v, bias=, scale=m.scaling)`` — the call ``TriangleAttention._forward_cuequivariance``
    makes, on the same views; otherwise ``_forward_vanilla``'s einsum with the views undone first, so it runs on the module's own tensors
    and strides (q * scaling, the bias broadcast over the query rows). Either way byte-identical to the module's route."""
    st = _TP_STATE["triatt_stock"]
    scaling = st["scaling"][int(q.shape[-1])]
    if st["cueq"]:
        bias = biases[0] if biases[0].dtype == q.dtype else biases[0].to(dtype=q.dtype)
        return _cuet().triangle_attention(q, k, v, bias=bias, scale=scaling)
    torch = _torch()
    query, key, value = q.permute(0, 1, 3, 2, 4), k.permute(0, 1, 3, 2, 4), v.permute(0, 1, 3, 2, 4)
    bias = biases[0].squeeze(1).permute(0, 2, 3, 1)
    r = int(query.shape[1])
    query = query * scaling
    attn = torch.einsum("bijhd,bikhd->bijkh", query, key)
    attn = attn + bias.unsqueeze(1).expand(-1, r, -1, -1, -1)
    attn = torch.nn.functional.softmax(attn, dim=-2)
    out = torch.einsum("bijkh,bikhd->bijhd", attn, value)
    return out.permute(0, 1, 3, 2, 4)


def _cuet():
    """``cuequivariance_torch`` — the module RF3's ``attention.py`` calls as ``cuet`` (imported by foundry when the engine engages it)."""
    mod = sys.modules.get("cuequivariance_torch")
    if mod is None:
        import cuequivariance_torch as mod  # noqa: PLC0415 — the engaged route only
    return mod


def _triatt_rows(m, pair, bias, core):
    """The row batch (``pair [1, r, N, C]`` LayerNorm'd, ``bias [1, N, N, H]``) through the attention core ``core`` over the module's projections:
    gate and layout as in both of the module's routes (``sigmoid(to_g) * out``, heads merged last)."""
    torch = _torch()
    B, r, N = int(pair.shape[0]), int(pair.shape[1]), int(pair.shape[2])
    gate = torch.sigmoid(m.to_g(pair))
    q, k, v = _triatt_qkv(m, pair)
    o = _rp("triatt").attend_query_blocks(core, q, k, v, (_triatt_bias_view(bias),), None)
    out = o.permute(0, 1, 3, 2, 4).reshape(B, r, N, -1)
    return gate * out


def _triatt_vanilla_rows(m, pair, bias):
    """RF3's ``TriangleAttention._forward_vanilla`` for ``r`` query rows (``pair [1, r, N, C]`` LayerNorm'd, ``bias [1, N, N, H]``): the einsum
    statement with the bias broadcast over the query rows (the stock reshapes with ``L`` from a square input)."""
    torch = _torch()
    B, r, N = int(pair.shape[0]), int(pair.shape[1]), int(pair.shape[2])
    gate = torch.sigmoid(m.to_g(pair))
    query = m.to_q(pair).reshape(B, r, N, m.h, -1)
    key = m.to_k(pair).reshape(B, r, N, m.h, -1)
    value = m.to_v(pair).reshape(B, r, N, m.h, -1)
    query = query * m.scaling
    attn = torch.einsum("bijhd,bikhd->bijkh", query, key)
    attn = attn + bias.unsqueeze(1).expand(-1, r, -1, -1, -1)
    attn = torch.nn.functional.softmax(attn, dim=-2)
    out = torch.einsum("bijkh,bikhd->bijhd", attn, value).reshape(B, r, N, -1)
    return gate * out


def _triatt_fns(m, starting: bool):
    """``triatt.TriAttFns`` of one RF3 ``TriangleAttention``: ln = norm, bias = to_b, attend = q/k/v/gate of the row batch + the module's kernel
    route + to_out. The ENDING node runs on rows of z^T (the pair-block driver transposes) with the gathered bias turned back to the stock
    orientation: RF3 builds the ending bias on the UNtransposed pair (``attention.py``: bias before the rearrange). Under ``TP_TRIATT_KERNEL``
    in {flash_triattn, cueq} the row batch runs on the core's attention core (:func:`_tp_triatt_core`, bound here with this module's route),
    whose fallback below the size gate is the very call the branches below make."""
    triatt = _rp("triatt")
    route_cueq = bool(m.use_cuequivariance and _cueq_on())
    core = _tp_triatt_core(m, route_cueq)

    def attend(x_rows, mask_rows, tb_full, _blk):
        pair = x_rows.unsqueeze(0)
        bias = (tb_full if starting else tb_full.transpose(0, 1)).unsqueeze(0)
        if core is not None:
            out = _triatt_rows(m, pair, bias, core)
        elif route_cueq:
            out = m._forward_cuequivariance(pair, bias)
        else:
            out = _triatt_vanilla_rows(m, pair, bias)
        return m.to_out(out)[0]

    return triatt.TriAttFns(ln=m.norm, bias=m.to_b, attend=attend)


APB_CORES = ("sdpa", "torch")                       # the attention-pair-bias core on local query rows: "sdpa" (this line's) = the shared core's fused
TP_APB_CORE = "sdpa"                                # SDPA-with-bias statement (opt_core.attn.sdpa_bias — the `sdpa` row the one-GPU big line's apb.big
                                                    # word resolves to on cc 9.0 / 8.0 above the apb_attn envelope), rectangular (R local queries x N keys),
                                                    # no [R, N, H] fp32 logits tensor; "torch" = RF3's einsum / softmax statement (the named fallback of a
                                                    # refused call, counted; ROWPAIR_APB_CORE=torch forces it: census apb_core_src=env)


APB_LN_ROWS = 256                                   # attention-pair-bias: LN_0 + to_b of the local pair rows run this many rows at a time into the heads-first
                                                    # bias buffer (a whole-R LayerNorm fp32 output would be one [R, N, c_z] tensor = 2u live in the pair stack AND in
                                                    # every confidence pass — 3.8 GiB per rank at 4000 tokens x2, the largest block of both stages)

TP_DIT_ROWS = "big"                                # the DiT query rows' attention core at n_gpu>1 (the core's rows face): a tier word (big | fast: the provider's
DIT_ROWS_WORDS = ("big", "fast", "apb_attn", "sdpa", "sba", "naive", "engine")   # rows cell's winner, else apb_attn by name) or a row word; 'engine' = RF3's
                                                    # statement (the [q, N, H] fp32 logits einsum + softmax). ROWPAIR_DIT_ROWS=<word> overrides.


def _dit_rows_word() -> str:
    w = (os.environ.get("ROWPAIR_DIT_ROWS") or TP_DIT_ROWS).strip()
    head = w.split(":", 1)[0]
    if head not in DIT_ROWS_WORDS:
        raise ValueError(f"ROWPAIR_DIT_ROWS={w!r}: one of {DIT_ROWS_WORDS} (row words may carry a :variant)")
    return w


DIT_BIAS_CACHE_POLICIES = ("fit", "core")             # the DiT pair-bias cache (24 blocks x [H, R, N] in z_cond's dtype = 3u: no per-step LN + projection of the
TP_DIT_BIAS_CACHE = "fit"                           # conditioned rows) — "fit" (this line's): ON iff the cache plus the roll-out's transients fit the free bytes
DIT_CACHE_RESERVE_GIB = 1.0                         # AGREED across ranks at roll-out entry (opt_core.mem.rowpair.dist.agreed_free_bytes, one collective every rank
                                                    # reaches): cache_bytes + 2 x the row-block work budget (ROWPAIR_DIFF_WORK_GB) + DIT_CACHE_RESERVE_GIB <= free —
                                                    # fastest where it fits, the recompute (24 x steps x S LN+projection passes) only at the memory edge where the
                                                    # cache would be the OOM; "core" = the shared core's own policy (ROWPAIR_DIFF_BIAS_CACHE_GB constant / auto
                                                    # fraction). ROWPAIR_DIFF_BIAS_CACHE=0|1 (the core's pin) always wins; ROWPAIR_DIT_BIAS_CACHE=fit|core selects
                                                    # the policy (census dit_cache_policy= dit_cache_gib= dit_free_gib= + the core's diff_bias_cache=on|off)


def _dit_bias_cache_fit(*, n_blocks: int, H: int, lay, elt: int) -> Optional[bool]:
    """The kit's pair-bias-cache decision for one roll-out (``DiffusionSchedule.decide(bias_cache=)``): None = the core decides (policy ``core``
    or an explicit ``ROWPAIR_DIFF_BIAS_CACHE`` pin), else cache_bytes + reserve <= agreed free bytes. Called at a point every rank reaches."""
    pol = (os.environ.get("ROWPAIR_DIT_BIAS_CACHE") or TP_DIT_BIAS_CACHE).strip().lower()
    pin = (os.environ.get("ROWPAIR_DIFF_BIAS_CACHE") or "auto").strip().lower()
    rec = _rp("evidence").record_schedule
    if pol not in DIT_BIAS_CACHE_POLICIES or pol == "core" or pin in ("0", "1", "on", "off"):
        rec(dit_cache_policy="core" if pol == "core" else (f"pin:{pin}" if pin != "auto" else f"unknown:{pol}"))
        return None
    free = _rp("dist").agreed_free_bytes()                                                  # min over ranks; None = no CUDA anywhere (CPU tests)
    if free is None:
        rec(dit_cache_policy="fit:nocuda")
        return None
    cache = int(n_blocks) * int(H) * int(lay.Rmax) * int(lay.N) * int(elt)
    try:
        work_gb = float(os.environ.get("ROWPAIR_DIFF_WORK_GB") or 8.0)
    except ValueError:                                                                       # 'auto' or malformed: the core names it; reserve the default
        work_gb = 8.0
    reserve = int(2 * work_gb * 1e9 + DIT_CACHE_RESERVE_GIB * 2 ** 30)
    fit = cache + reserve <= int(free)
    rec(dit_cache_policy="fit", dit_cache_gib=round(cache / 2 ** 30, 2), dit_free_gib=round(int(free) / 2 ** 30, 2), dit_cache_fit=int(fit))
    return bool(fit)


def _apb_core_word() -> str:
    w = (os.environ.get("ROWPAIR_APB_CORE") or "").strip().lower()
    return w if w in APB_CORES else TP_APB_CORE


def _apb_fn(m):
    """The attention-pair-bias statement of ``AttentionPairBiasPairformerDeepspeed`` with query rows = this rank's rows, keys / values whole,
    pair bias = the local pair rows (``transition.apb_local_queries`` gathers the output rows). The softmax(QKᵀ + b)·V core runs on the shared
    core's fused SDPA-with-bias statement (``opt_core.attn.sdpa_bias``, rectangular: R local queries × N keys, bias rows ``[R, N, H]`` read
    heads-last in place — the ``sdpa`` row the one-GPU line's ``apb.big`` word serves; fast-class numerics, the logits never materialised as an
    ``[R, N, H]`` fp32 tensor); a call the core refuses by name runs RF3's einsum / softmax statement (counted: LEVER words ``apb_core=
    apb_served= apb_fallback= apb_fallback_by=``)."""
    torch = _torch()
    if m.use_deepspeed_evo:
        raise _rp().RowpairRefused("refused: use_deepspeed_evo attention under n_gpu>1")
    word = _apb_core_word()
    cen = CTX.setdefault("apb", {"core": word, "src": "env" if os.environ.get("ROWPAIR_APB_CORE") else "line", "served": 0, "fallback": 0, "by": {}})
    sdpa = None
    if word == "sdpa":
        try:
            sdpa = _core.load("attn.sdpa_bias")
        except Exception as e:                                                              # noqa: BLE001 — not importable in this tree: named, the torch statement serves
            cen["by"][f"import:{type(e).__name__}"] = cen["by"].get(f"import:{type(e).__name__}", 0) + 1
            sdpa = None

    def stock_core(Q_IH, K_IH, V_IH, B_IIH):                                                 # RF3's statement (attention.py), Q pre-scaled
        A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
        return torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)

    def attn(q_src, kv_src, z_rows):
        lay = CTX["layout"]
        a_kv = m.ln_1(kv_src)
        if m.force_bfloat16 and a_kv.device.type != "mps":
            a_kv = a_kv.to(torch.bfloat16)
        a_q = a_kv[..., lay.r0:lay.r1, :]
        Q_IH = m.to_q(a_q)
        K_IH = m.to_k(a_kv)
        V_IH = m.to_v(a_kv)
        G_IH = m.to_g(a_q)
        Q_IH = Q_IH / torch.sqrt(torch.tensor(m.c).to(Q_IH.device, Q_IH.dtype))

        def stock_bias():                                                                   # RF3's statement: LN_0 of ALL local pair rows at once (an [R, N, c_z] fp32
            return m.to_b(m.ln_0(z_rows)) + torch.tensor([0.0], device=z_rows.device)[..., None]   # LayerNorm output = 2u live) -> [R, N, H] promoted to fp32

        def bias_heads_first(dtype):                                                        # the same rows for the fused call: [1.., H, R, N] in q's dtype, produced
            lead_z = z_rows.dim() == 3                                                      # APB_LN_ROWS local rows at a time — the LayerNorm's fp32 output and the
            R_, N_, H_ = int(z_rows.shape[-3]), int(z_rows.shape[-2]), int(m.to_b.out_features)   # projection are per-row statements (identical values), their
            shape = (1, H_, R_, N_) if lead_z else tuple(z_rows.shape[:-3]) + (H_, R_, N_)  # transient bounded to APB_LN_ROWS x N x c_z x 4 B instead of 2u
            b = torch.empty(shape, dtype=dtype, device=z_rows.device)
            step = max(1, int(APB_LN_ROWS))
            for c0 in range(0, R_, step):
                c1 = min(R_, c0 + step)
                blk = m.to_b(m.ln_0(z_rows[..., c0:c1, :, :]))                             # [.., rows, N, H]
                if lead_z:
                    b[0, :, c0:c1, :] = blk.movedim(-1, -3)
                else:
                    b[..., :, c0:c1, :] = blk.movedim(-1, -3)
                del blk
            return b

        A = None
        B_IIH = None
        if sdpa is not None:
            try:                                                                            # one fused call: q/k/v as [1.., H, L, D] views, the bias rows heads-first in
                lead = Q_IH.dim() == 3                                                      # q's dtype (16·R·N·2 B; the statement's [R, N, H] fp32 logits + softmax are
                q4, k4, v4 = (t[None] if lead else t for t in (Q_IH, K_IH, V_IH))           # 3x that), scale 1 (Q carries it); unbatched RF3 tensors get the batch axis
                b4 = bias_heads_first(Q_IH.dtype)                                           # the fused kernels take
                A, _ev = sdpa.sdpa_bias(q4.transpose(-3, -2), k4.to(Q_IH.dtype).transpose(-3, -2), v4.to(Q_IH.dtype).transpose(-3, -2), b4, None,
                                      layout="bhld", bias_layout="bhqk", scale=1.0, out_dtype=V_IH.dtype)
                A = A.transpose(-3, -2)                                                     # [1.., R, H, D]
                A = A[0] if lead else A
                cen["served"] += 1
                cen["ev"] = _ev
            except sdpa.Refused as e:                                                       # named refusal: the stock statement below, counted by event
                cen["fallback"] += 1
                cen["by"][e.event] = cen["by"].get(e.event, 0) + 1
                A = None
        if A is None:
            if sdpa is None:
                cen["fallback"] += 1
                cen["by"]["core_torch" if word == "torch" else "unavailable"] = cen["by"].get("core_torch" if word == "torch" else "unavailable", 0) + 1
            B_IIH = stock_bias()
            A = stock_core(Q_IH, K_IH, V_IH, B_IIH)
        A = (G_IH * A).flatten(start_dim=-2)
        return m.to_a(A)

    return attn


def _pair_block_fns(block, lay):
    """``pairstack.PairBlockFns`` of one RF3 ``PairformerBlock`` (cached per module): its five pair updates bound to the core's streamed
    schedules, plus — when the block carries the single track (c_s > 0) — attention-pair-bias on local query rows and the single transition."""
    key = ("pf", id(block), lay.N)
    fns = CTX["fns"].get(key)
    if fns is not None:
        return fns
    pairstack = _rp("pairstack")
    tr = _rp("transition")
    stats = CTX["stats"]
    apb = single = None
    if hasattr(block, "attention_pair_bias"):
        attn = _apb_fn(block.attention_pair_bias)
        apb = lambda s, z, l: s + tr.apb_local_queries(attn, s, z, l, gather=True)          # noqa: E731
        single = lambda s: s + block.s_transition(s)                                      # noqa: E731
    fns = pairstack.bind(trimul_out=_trimul_fns(block.tri_mul_outgoing, lay.N), trimul_in=_trimul_fns(block.tri_mul_incoming, lay.N),
                         triatt_start=_triatt_fns(block.tri_attn_start, True), triatt_end=_triatt_fns(block.tri_attn_end, False),
                         transition=lambda rows, mask_u: block.z_transition(rows), chunk=TRIATT_ROWS, apb=apb, single_transition=single,
                         trimul_kw={"inplace_chunk": TRIMUL_CHUNK}, stats=stats)
    CTX["fns"][key] = fns
    return fns


def _msa_pair_fns(mm, lay):
    """The MSA module's weight-shared pair block (``tri_mult_outgoing/incoming``, ``tri_attn_start/end``, ``pair_transition``) as PairBlockFns."""
    key = ("msa", id(mm), lay.N)
    fns = CTX["fns"].get(key)
    if fns is None:
        fns = _rp("pairstack").bind(trimul_out=_trimul_fns(mm.tri_mult_outgoing, lay.N), trimul_in=_trimul_fns(mm.tri_mult_incoming, lay.N),
                                    triatt_start=_triatt_fns(mm.tri_attn_start, True), triatt_end=_triatt_fns(mm.tri_attn_end, False),
                                    transition=lambda rows, mask_u: mm.pair_transition(rows), chunk=TRIATT_ROWS,
                                    trimul_kw={"inplace_chunk": TRIMUL_CHUNK}, stats=CTX["stats"])
        CTX["fns"][key] = fns
    return fns


def _pair_stack(blocks, z, lay, s=None):
    """``pairstack.pair_stack_`` over RF3 PairformerBlocks on the shard ``z [R, N, C]`` (no pair mask in RF3)."""
    fns = [_pair_block_fns(b, lay) for b in blocks]
    z, s = _rp("pairstack").pair_stack_(fns, z, None, lay, s=s, transition_mask=False)
    return z, s


# ------------------------------------------------------------------------------------------------------------------------ recycler
def _recycler_forward(self, prev, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II):
    """One recycle on this rank's rows: ``Z = Z_init + process_zh(Z)`` in place, ``Z += template(f, Z)``, ``Z = msa_module(f, Z, S_inputs)``,
    ``S = S_init + process_sh(S)``, the 48 pairformer blocks — the shard is never gathered. A dense ``Z_init`` (no active layout) runs the
    kit's forward unchanged."""
    lay = CTX["layout"]
    if not CTX["installed"] or not _is_shard(Z_init_II, lay):
        _count("recycler", "replicated")
        return prev(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II)
    trunk = _rp("trunk")
    cycle = int(CTX["cycle"])
    if cycle == 0:
        _ckpt_begin(f, S_inputs_I, Z_init_II, lay)                                            # ROWPAIR_CKPT_DIR: the item's checkpointer + resume (before any trunk compute)
    res = CTX.get("resume")
    if res is not None and (res["tag"] == _rp("ckpt").TRUNK_FINAL or cycle <= int(res["cycle"])):
        _count("recycler", "resumed")                                                          # FAST-FORWARD: this cycle's state is the checkpoint's — no compute, no collective
        CTX["cycle"] += 1
        return res["s"], res["z_loc"]
    _count("recycler")
    t0 = time.time()
    try:
        trunk.guard_rng_replicated(f"trunk.rng.cycle{CTX['cycle']}", device=Z_init_II.device if Z_init_II.is_cuda else None)
    except _rp().RowpairRefused:
        raise
    except Exception as e:  # noqa: BLE001 — the guard is evidence on backends without generator state access
        CTX["errors"].append(f"rng guard: {type(e).__name__}: {e}")
    z = trunk.recycle_shard_(Z_II, _zinit_source(Z_init_II), self.process_zh, lay)      # the park serves Z_init's row blocks (ROWPAIR_PARK_ZINIT) or the resident shard
    z = _template_rows(self.template_embedder, f, z, lay)
    _mem_mark(f"templ{CTX['cycle']}")
    z = _msa_module_rows(self.msa_module, f, z, S_inputs_I, lay)
    _mem_mark(f"msa{CTX['cycle']}")
    S_I = S_init_I + self.process_sh(S_I)
    z, S_I = _pair_stack(list(self.pairformer_stack), z, lay, s=S_I)
    _ckpt_save(cycle, lay, S_inputs_I, S_I, z)                                                # ROWPAIR_CKPT_DIR: cycle_<kkk> per ROWPAIR_CKPT_EVERY, trunk_final at the last recycle
    CTX["cycle"] += 1
    CTX["timing"]["trunk"].append(round(time.time() - t0, 3))
    _mem_mark(f"recycle{CTX['cycle']}")
    return S_I, z


# ------------------------------------------------------------------------------------------------------------------------ template
def _templ_census(n_fill: int, n_tokens: int) -> dict:
    """The noised-template feature's census under n_gpu>1, CUMULATIVE over this rank process's items, as ``templ_*`` schedule words: the
    ``LEVER name=rowpair`` line carries ``templ_form=row_born`` (the feature's form: per-token precursors, rows built at the embedder),
    ``templ_items`` = items whose feature reached the row-sharded template embedder, ``templ_real=<r>/<items>`` = those with a real template
    (RF3's one conditioning slot: a fill mask with a token set), ``templ_fill_max`` = the most filled tokens of any one item; the exit tally's
    schedule additionally ``templ_tokens`` (the current item's token count) and ``templ_row_blocks`` (template row blocks built, all items).
    Called at every recycle's template call with the item's filled / total token counts; an item (``CTX["item"]``, the model entry's ordinal)
    is counted once, at its first call — recycle 0, or a resumed trunk's first computed one. Returns the census dict (``CTX["templ_feats"]``)."""
    tf = CTX.setdefault("templ_feats", {"row_blocks": 0, "item": None, "items": 0, "real": 0, "fill_max": 0})
    if tf["item"] != CTX["item"]:
        tf.update(item=CTX["item"], items=tf["items"] + 1, real=tf["real"] + int(n_fill > 0), fill_max=max(tf["fill_max"], n_fill))
    _rp("evidence").record_schedule(templ_form="row_born", templ_items=tf["items"], templ_real=f"{tf['real']}/{tf['items']}", templ_fill_max=tf["fill_max"],
                                    templ_tokens=n_tokens)
    return tf


def _template_rows(te, f, z, lay):
    """``Z += RF3TemplateEmbedder(f, Z)`` on the shard: per row block ``v = emb_pair(LN(z rows)) + emb_templ(feature rows)`` (RF3's one
    conditioning slot: ``distogram_condition`` rows, ``has_distogram_condition`` rows — built here from the feature's per-token precursors
    (``templ.distogram_condition_rows``: no rank holds an ``[I, I]`` template tensor) — the joint noise level of the rows), the two template
    pairformer blocks on the ``v`` shard, ``u = 0 + LN_after(v)``, ``agg_emb(relu(u))`` added in place. A dense feature is refused by name."""
    torch = _torch()
    templ = _rp("template")
    PL = sys.modules[type(te).__module__]
    pre = _templ.precursors_of(f)                                # the feature's per-token precursors (templ_feats site): no rank holds [I, I] template tensors
    if pre is None:
        if any(k in f for k in _templ.DENSE):
            raise _rp().RowpairRefused("refused: the template distogram feature arrived dense ([N, N, 64] one-hot) under n_gpu>1 — "
                                       f"{_templ.FEATURIZER_MODULE}.{_templ.FEATURIZER} is not the kit's precursor form (site templ_feats)")
        raise _rp().RowpairRefused(f"refused: the feature dict carries neither the template distogram precursors {list(_templ.PRECURSORS)} nor the dense feature")
    host = {k: v.detach().to("cpu") for k, v in pre.items()}    # [I, 3] coordinates, [I] mask, [I] ids, the edges: the dense statement's operands, on the host as stock has them
    scale = f["distogram_condition_noise_scale"]                # [I]
    dev = z.device
    _count("template")
    tf = _templ_census(int(host["fill"].sum()), int(host["fill"].shape[0]))   # the feature's census, cumulative over this process's items (the LEVER line's templ_* words)

    def unit_rows_fn(z_rows, slot, g):
        g0, g1 = g
        cond_rows, has_rows = _templ.distogram_condition_rows(host["centers"], host["fill"], host["molecule"], host["edges"], g0, g1)
        cond_rows, has_rows = cond_rows.to(dev), has_rows.to(dev)                           # [rows, I, 64] fp32 one-hot, [rows, I] bool: GLOBAL rows [g0, g1)
        tf["row_blocks"] += 1
        _rp("evidence").record_schedule(templ_row_blocks=tf["row_blocks"])
        joint = (scale[None, :] ** 2 + scale[g0:g1, None] ** 2).sqrt()                   # [rows, I]  (stock: scale[None,:]**2 + scale[:,None]**2)
        level = PL.af3_noise_scale_to_noise_level(joint)
        feats = torch.cat([cond_rows, has_rows.unsqueeze(-1), level.unsqueeze(-1)], dim=-1)
        feats = feats * has_rows.unsqueeze(-1)
        v = te.emb_pair(te.norm_pair_before_pairformer(z_rows)) + te.emb_templ(feats)   # [rows, I, c]
        return v.unsqueeze(0)                                                              # [1, rows, I, c]

    def pair_stack_fn(u, mask_loc):
        v, _ = _pair_stack(list(te.pairformer), u[0], lay, s=None)
        return te.norm_after_pairformer(v).unsqueeze(0)

    def finish_fn(t):
        t0 = t.select(-4, 0)                                                               # the one slot: [rows, I, c]
        u = torch.zeros(t0.shape, device=t0.device) + t0                                  # stock: u = zeros(I, I, c) + LN_after(v)
        return te.agg_emb(torch.nn.functional.relu(u))

    return templ.template_embed_rows(z, lay, n_templ=1, c_t=int(te.c), unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn,
                                     finish_fn=finish_fn, mask_loc=None, add=True)


# ------------------------------------------------------------------------------------------------------------------------ MSA module
PWA_CHUNK_GIB = 1.0                                 # pair-weighted averaging: sequences per chunk such that the chunk's values + gates + gathered output rows
PWA_CHUNK_MIN = 16                                  # ([chunk, N, H*c] each, replicated on every rank: values need every token) stay under PWA_CHUNK_GIB — the
                                                    # whole stack at once would be 3 x [S, N, 256] bf16 = 12.6 GB per rank at 8000 tokens whatever P (the MSA stage
                                                    # does not shrink with ranks); ROWPAIR_PWA_S_CHUNK=<int> pins it (0 = one chunk); census pwa_s_chunk (core)


def _pwa_s_chunk(S: int, N: int, F: int) -> Optional[int]:
    """Sequences per pair-weighted-averaging chunk (``msa.pwa_rows(s_chunk=)``): the pin, else the largest multiple of 8 (>= PWA_CHUNK_MIN) whose
    three ``[chunk, N, F]`` transients fit PWA_CHUNK_GIB; None = one chunk (the stack fits, or S is small)."""
    pin = (os.environ.get("ROWPAIR_PWA_S_CHUNK") or "").strip()
    if pin:
        v = int(pin)
        return None if v <= 0 else v
    per_seq = 3 * int(N) * int(F) * 2                                                     # values / gates / gathered rows answer in the autocast dtype (2 B) whatever m is
    fit = int(PWA_CHUNK_GIB * 2 ** 30) // max(1, per_seq)
    if fit >= S:
        return None
    return max(PWA_CHUNK_MIN, (fit // 8) * 8)


def _msa_module_rows(mm, f, z, S_inputs_I, lay):
    """``MSAModule.forward`` on the shard: ``m`` (the embedded subsample, replicated) and, per block, ``Z += OPM(m)`` rows in place,
    ``m += PWA(m, Z rows)`` (local query rows, softmax whole per row, output rows all-gathered), ``m += transition(m)``, the weight-shared
    pair block through the pair-block driver."""
    torch = _torch()
    msa = _rp("msa")
    m = mm.msa_subsampler(_msa_rows(f, S_inputs_I, lay), S_inputs_I)   # [S, I, 64] replicated (the cycle's raw rows: device-resident, or moved from the host under ROWPAIR_MSA_HOST)
    _count("msa_module")
    opm_mod = mm.outer_product
    pwa = mm.msa_pair_weighted_averaging
    pair_fns = _msa_pair_fns(mm, lay)
    pairstack = _rp("pairstack")

    def opm(m_, z_):
        x = opm_mod.norm(m_)
        left = opm_mod.proj_left(x)                             # [S, I, c]
        right = opm_mod.proj_right(x) / float(m_.shape[0])      # stock: right / N with N = the number of sequences

        def outer_fn(a_blk, b):
            rows = int(a_blk.shape[1])
            out = torch.einsum("sli,smj->lmij", a_blk, b).reshape(rows, int(b.shape[1]), -1)
            return opm_mod.proj_out(out)

        # the row block is the core's ROWPAIR_ROWBLK_MB target (512 MiB) over THIS statement's per-row transient: the einsum's [rows, N, c, c]
        # product in the operands' dtype, its head-major copy the reshape takes (the einsum hands back a permuted view), and the projected
        # [rows, N, C_z] rows twice (proj_out's output + the in-place add's operand) — not the core's default N·2·C_z·elt (the OUTPUT rows only,
        # 8.6x under this statement's bytes at c=32: a 4.3 GiB transient at every N >= 1400 instead of the 0.5 GiB the target names)
        c_a, c_b, C_z_ = int(left.shape[-1]), int(right.shape[-1]), int(z_.shape[-1])
        per_row = int(lay.N) * (2 * c_a * c_b * int(left.element_size()) + 2 * C_z_ * int(z_.element_size()))
        CTX["opm_row_bytes"] = per_row
        return msa.opm_rows_budgeted(left, right, lay, outer_fn, C_z=C_z_, out=z_, add=True, row_dim=1, bytes_per_row=per_row)

    def msa_update(m_, z_):
        def prep_fn(z_rows, g0, g1):
            return pwa.to_bias(pwa.norm_pair(z_rows)).permute(2, 0, 1)                    # [H, rows, I]

        bias_shard = msa.pwa_bias_rows(prep_fn, z_, lay)                                  # [H, R, I]
        H, c = int(pwa.n_heads), int(pwa.weighted_average_channels)

        def values_fn(m_chunk):
            mn = pwa.norm_msa(m_chunk)
            v = pwa.to_v(mn).reshape(int(m_chunk.shape[0]), int(m_chunk.shape[1]), H, c)   # [S, I, H, c]
            gate = torch.sigmoid(pwa.to_gate(mn))                                          # [S, I, H*c] (per channel) or [S, I, H]
            return (v, gate)

        def attend_fn(w, state, g0, g1):
            v, gate = state                                                                # w: [q, I, H] (softmax_fn's stock layout)
            o = torch.einsum("ijh,sjhc->sihc", w, v)                                       # [S, q, H, c]
            g_rows = gate[:, g0:g1]
            if pwa.separate_gate_for_every_channel:
                o = o.reshape(int(v.shape[0]), g1 - g0, -1) * g_rows
            else:
                o = (o * g_rows.unsqueeze(-1)).reshape(int(v.shape[0]), g1 - g0, -1)
            return o

        def softmax_fn(x):                                                                 # x: bias rows [H, q, I] -> the stock w_IIH statement
            return torch.nn.functional.softmax(x.permute(1, 2, 0), dim=-2)                 # [q, I(j), H], softmax over j

        s_chunk = _pwa_s_chunk(int(m_.shape[0]), int(lay.N), H * c)                       # sequences per chunk: the values / gates / gathered rows under PWA_CHUNK_GIB
        upd = msa.pwa_rows(m_, bias_shard, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=pwa.to_out, softmax_fn=softmax_fn, s_chunk=s_chunk)
        m_ = m_ + upd                                                                      # dropout: a no-op at inference
        m_ = m_ + msa.msa_transition_rows(lambda mt, t0, t1: mm.msa_transition(mt), m_, lay, shard_tokens=False)
        return m_

    def pair_block(z_):
        z_, _ = pairstack.pair_block_(pair_fns, z_, None, lay, s=None, transition_mask=False)
        return z_

    blocks = [{"opm": opm, "msa_update": msa_update, "pair_block": pair_block, "opm_first": True} for _ in range(int(mm.n_block))]
    return msa.msa_module_sharded(m, z, lay, blocks)


# ------------------------------------------------------------------------------------------------------------------------ distogram
def _distogram_forward(self, prev, Z_II):
    """``predictor(Z + Z^T)`` on this rank's rows: the transposed rows arrive block by block (``dist.transpose_blocks``); the logits stay a
    row shard ``[R, I, bins]`` (RF3's inference tail reads no distogram; a consumer expecting ``[I, I, bins]`` fails loudly, never silently)."""
    lay = CTX["layout"]
    if not CTX["installed"] or not _is_shard(Z_II, lay):
        _count("distogram", "replicated")
        return prev(self, Z_II)
    dist = _rp("dist")
    _count("distogram")
    _release_zinit()                                            # the trunk is over: the z_init park's host buffer returns to the pool
    _ckpt_phase("diffusion")
    out = None
    for i0, i1, zT_blk in dist.transpose_blocks(Z_II, lay):
        y = self.predictor(Z_II[i0:i1] + zT_blk)
        if out is None:
            out = y.new_empty((lay.R,) + tuple(y.shape[1:]))
        out[i0:i1] = y
    _mem_mark("distogram")
    _conf_plan_item(Z_II)["plan"].park_now()                    # ROLL-OUT ENTRY: the trunk shard parked (ROWPAIR_CONF_PARK_ZTRUNK) before the diffusion roll-out — the
    return out                                                  # conditioning reads its rows from the park (_diff_cond_forward), the confidence passes are served from it


# ------------------------------------------------------------------------------------------------------------------------ confidence
def _confidence_forward(self, prev, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs=None):
    """``ConfidenceHead.forward`` (one diffusion sample) on the trunk shard: ``tp_conf.run_confidence_sharded`` — the head's embedding rows,
    its pairformer blocks through this module's pair-block driver (``pair_stack_fn``), the PAE / PDE logits produced and consumed per row
    block (never whole on a rank), pLDDT / resolved on the replicated single representation. Returns what ``RF3WithConfidence.forward``
    concatenates over samples: ``plddt_logits`` / ``exp_resolved_logits`` (device, replicated) and, under the stock keys ``pae_logits`` /
    ``pde_logits``, the EXPECTED PAE / PDE matrices ``[1, I, I]`` fp32 on pinned HOST (rank 0; zero-valued host matrices on ranks > 0, whose
    tail output is scratch) — the tail's consumers are re-pointed to read them (:func:`_compile_expected`, :func:`_tm_metrics`). The recycle-0
    early-stopping probe (``X_pred_L is None``, ``RF3.py:419-431``) is the same call: the finish runs the head without the distance term and
    returns pLDDT / resolved only (``conf_probe=plddt_only`` in its census; the probe's consumer reads ``plddt_logits``)."""
    lay = CTX["layout"]
    if not CTX["installed"] or not _is_shard(Z_trunk_II, lay):
        _count("conf_head", "replicated")
        return prev(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs)
    torch = _torch()
    _count("conf_head")
    t0 = time.time()
    f = CTX.get("f")
    if f is None or "asym_id" not in f:
        raise _rp().RowpairRefused("refused: confidence head under n_gpu>1 without the model entry's feature dict (asym_id / is_ligand for the TM finish)")
    probe = X_pred_L is None
    if probe and _probe_replayed():                                                            # ROWPAIR_CKPT_DIR resume past cycle 0: the probe is not re-judged on later state
        _count("conf_probe", "replayed")                                                       # (its CONTINUE decision is replayed by _replayed_early_stop) — no head pass
        return {"plddt_logits": None, "exp_resolved_logits": None, "pae_logits": None, "pde_logits": None}
    if not probe:
        X_pred_L = _rank_spread_then_sync("x_pred", X_pred_L)                                # rank 0's sample on every rank (+ the spread census)
        if CTX.get("conf_plan") is None:
            _ckpt_phase("confidence")
    from . import tp_conf
    import importlib
    MU = importlib.import_module("rf3.metrics.metric_utils") if "conf_fns" not in CTX["fns"] else None
    fns = CTX["fns"].get("conf_fns")
    if fns is None:
        AH = sys.modules[type(self).__module__]
        fns = tp_conf.RF3Fns(find_bin_midpoints=MU.find_bin_midpoints, unbin_logits=MU.unbin_logits, discretize_distance_matrix=AH.discretize_distance_matrix)
        CTX["fns"]["conf_fns"] = fns
    blocks = list(self.pairformer)

    def pair_stack_fn(z_rows, S):
        if not probe and S.dim() == 2:                                                     # stock: S gains the sample dim inside the head's pairformer by
            S = S.unsqueeze(0)                                                             # broadcasting against the pair operand, which is [1, I, I, c] only when
        return _pair_stack(blocks, z_rows, lay, s=S)                                       # the distance term of X_pred_L [1, L, 3] was added — the probe (no
                                                                                           # coordinates) keeps [I, c_s], so its pLDDT logits are [I, bins] exactly as
                                                                                           # stock's (should_early_stop_by_mean_plddt unsqueezes them itself)

    is_ligand = f["is_ligand"] if "is_ligand" in f else torch.zeros_like(f["asym_id"], dtype=torch.bool)
    n_pae, n_pde = int(self.predict_pae.out_features), int(self.predict_pde.out_features)
    st = _conf_plan(Z_trunk_II, probe)                                                        # the trunk shard's placement over this item's passes
    out = tp_conf.run_confidence_sharded(self, z_trunk_shard=Z_trunk_II, layout=lay, S_inputs_I=S_inputs_I, S_trunk_I=S_trunk_I, X_pred_L=X_pred_L,
                                         rep_atoms=rep_atoms, pair_stack_fn=pair_stack_fn, asym_id=f["asym_id"], is_ligand=is_ligand, rf3=fns,
                                         pae=(PAE_MAX, n_pae), pde=(PDE_MAX, n_pde), tm=(PAE_MAX, n_pae),
                                         plan=st["plan"], pass_index=st["i"], last_use=False if probe else None, moments=st["moments"])
    _conf_plan_end(st, probe, out.pop("moments", None))
    I = int(lay.N)
    pae, pde = out.get("pae"), out.get("pde")
    if probe:
        _count("conf_probe")
    else:
        CTX["tm"].append(out.get("tm"))                                                    # sample order = call order (the tail's metrics read it)
        if pae is None:                                                                    # ranks > 0: the tail runs on zero-valued host matrices (scratch output)
            pae = torch.zeros((1, 1, 1), dtype=torch.float32).expand(1, I, I)
            pde = pae
    CTX["timing"]["conf"].append(round(time.time() - t0, 3))
    _mem_mark("conf_probe" if probe else "conf")
    return {"pae_logits": pae, "pde_logits": pde, "plddt_logits": out["plddt_logits"], "exp_resolved_logits": out["exp_resolved_logits"]}


PAE_MAX = 32.0                                     # the shipped confidence-loss config: pae / pde max_value (64 bins from the head's Linear)
PDE_MAX = 32.0


PRECISION_WORDS = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float64": "fp64"}


def _precision_word(z) -> str:
    """The line's precision word for a checkpoint's tag (``ckpt.precision_tag``): the trunk pair shard's storage dtype (bf16 under the
    engine's autocast, fp32 / fp64 in the CPU tests)."""
    return PRECISION_WORDS.get(str(z.dtype).replace("torch.", ""), str(z.dtype).replace("torch.", ""))


def _ckpt_begin(f: dict, S_inputs_I, z, lay) -> None:
    """``ROWPAIR_CKPT_DIR`` (the core's per-cycle trunk checkpoints, ``opt_core.mem.rowpair.ckpt``): at the item's FIRST ``Recycler.forward`` the
    adapter builds the core ``TrunkCheckpointer`` — query id = the pipeline's ``example_id`` (the H2D seam saw it), seed = the ``seed=<int>`` token
    of this rank's command line (``fold.cmd_seed``; rule R1 of :func:`launch` requires it), N, feature tag = a per-rank digest of the model's feature
    dict (the per-cycle ``msa`` view excluded; rank 0's host MSA stack included under ``ROWPAIR_MSA_HOST``) + the precision word of the trunk shard
    — and asks it to resume (``try_resume``: ``trunk_final`` if complete and compatible else the newest compatible cycle tag; ``ROWPAIR_RESUME`` /
    ``ROWPAIR_RESUME_TAG`` / ``ROWPAIR_CKPT_READ_DIR`` / ``ROWPAIR_RESUME_RNG`` / ``ROWPAIR_RESUME_ALLOW_FEATS`` are the core's; a tag whose feature
    digest or precision differs is REFUSED BY NAME there). A resume is re-entered by FAST-FORWARD inside RF3's own recycle loop
    (:func:`_recycler_forward` hands the checkpoint's ``(S_I, Z shard)`` back for every cycle the tag covers, computing nothing); a ``trunk_final``
    resume also writes the saved ``S_inputs`` INTO the tensor the stock flow carries (``Recycler.forward`` cannot return it). ``S_init`` / ``Z_init``
    are recomputed by ``pre_recycle`` as on the core driver's cycle-k resume. Nothing is built when the variable is unset (census ``ckpt=off``)."""
    torch = _torch()
    ev = _rp("evidence")
    CTX["ckpt"], CTX["resume"] = None, None
    CTX["trunk_n"] = int(f["msa_stack"].shape[0]) if torch.is_tensor(f.get("msa_stack")) else None   # RF3's n_cycle IS msa_stack.shape[0] (trainers/rf3.py:302; the MSA_HOST placeholder keeps dim 0)
    if not (os.environ.get("ROWPAIR_CKPT_DIR") or "").strip():
        ev.record_schedule(ckpt="off")
        return
    CK = _rp("ckpt")
    from . import fold as _fold                                                               # the seed token parser of the kit's own command lines (one definition)
    seed = _fold.cmd_seed([str(a) for a in sys.argv])
    qid = CTX.get("example_id")
    if not qid:                                                                                # a checkpoint tree is named by the query: a nameless item is refused, never filed as "item"
        raise _rp().RowpairRefused(f"refused: ROWPAIR_CKPT_DIR is set but the item carries no example_id (the engine's H2D seam saw none for it) — "
                                   f"the checkpoint tree <dir>/<example_id>_<seed>/ cannot be named")
    feats = {"f": {k: v for k, v in f.items() if k != "msa"}}                                  # "msa" = the cycle's view of msa_stack (RF3.py:264), covered by the stack itself
    held = CTX.get("msa_host")
    if held is not None:
        feats["msa_host"] = held["feats"]                                                     # rank 0's host stack (rank0) / every rank's (all); ranks > 0: their placeholder
    ck = CK.TrunkCheckpointer(str(qid), "noseed" if seed is None else int(seed), int(lay.N), features=feats, precision=_precision_word(z), dtype=z.dtype,
                              log=_rp("dist").comm().log)
    CTX["ckpt"] = ck
    ev.record_schedule(ckpt="on" if ck.enabled else "off", ckpt_query=str(ck.query), ckpt_every=int(ck.every), ckpt_n_recycle=CTX["trunk_n"] if CTX["trunk_n"] is not None else "unknown")
    if not ck.enabled:
        return
    res = ck.try_resume(layout=lay, device=S_inputs_I.device, num_cycles=CTX["trunk_n"])     # refused by name inside on a digest / precision / dtype mismatch
    if res is None:
        ev.record_schedule(ckpt_ff_cycles=0)
        return
    if res["tag"] == CK.TRUNK_FINAL:
        if res.get("s_input") is not None:
            S_inputs_I.copy_(res["s_input"])                                                  # the stock flow's S_inputs tensor now holds the checkpoint's (in place)
        ff = CTX["trunk_n"] if CTX["trunk_n"] is not None else "all"
    else:
        ff = int(res["cycle"]) + 1
    CTX["resume"] = res
    ev.record_schedule(ckpt_ff_cycles=ff, ckpt_resume_probe="not_probed")
    _rp("dist").comm().log(f"[ckpt] resume re-entry: fast-forward {ff} recycle(s) of RF3's loop from {res['tag']}")


def _ckpt_save(cycle: int, lay, S_inputs_I, S_I, z) -> None:
    """After a computed recycle: ``trunk_final`` (shard + ``s`` + ``s_input``) at the last recycle, else ``cycle_<kkk>`` per ``ROWPAIR_CKPT_EVERY``
    (shard + ``s``) — the core's writer, format and census (``ckpt_wrote``)."""
    ck = CTX.get("ckpt")
    if ck is None or not ck.enabled:
        return
    n = CTX.get("trunk_n")
    if n is not None and int(cycle) == int(n) - 1:
        ck.save_trunk_final(layout=lay, s_input=S_inputs_I, s=S_I, z_loc=z, cycle=int(cycle))
    else:
        ck.maybe_save_cycle(int(cycle), layout=lay, s=S_I, z_loc=z)


def _ckpt_phase(name: str) -> None:
    ck = CTX.get("ckpt")
    if ck is not None and ck.enabled:
        ck.phase(name)


def _ckpt_finalize() -> None:
    """Item exit: join a pending asynchronous checkpoint write; drop the item's checkpointer and resume state."""
    ck = CTX.get("ckpt")
    CTX["ckpt"], CTX["resume"], CTX["example_id"], CTX["trunk_n"] = None, None, None, None
    if ck is not None:
        ck.finalize()


def _probe_replayed() -> bool:
    """True when this item resumed from a tag past cycle 0 (``cycle_<k>``, k >= 1, or ``trunk_final``): such a tag exists only because the
    uninterrupted run's recycle-0 probe decided CONTINUE (an early-stopped item writes no later tag), so the probe's decision is REPLAYED from
    the checkpoint's existence and never re-judged on later state. A resume from ``cycle_000`` probes cycle-0 state exactly as uninterrupted."""
    res = CTX.get("resume")
    if res is None:
        return False
    return res["tag"] == _rp("ckpt").TRUNK_FINAL or int(res["cycle"]) >= 1


def _replayed_early_stop(fn):
    """RF3's ``should_early_stop_fn`` (RF3.py:431-436) under a checkpoint resume: ``(False, {})`` — CONTINUE, replayed — when
    :func:`_probe_replayed`; the caller's function otherwise. Census ``ckpt_resume_probe=replayed:continue|cycle0_state``."""
    @functools.wraps(fn)
    def should_early_stop(*args, **kwargs):
        ev = _rp("evidence")
        if _probe_replayed():
            ev.record_schedule(ckpt_resume_probe="replayed:continue")
            return False, {}
        if CTX.get("resume") is not None:
            ev.record_schedule(ckpt_resume_probe="cycle0_state")
        return fn(*args, **kwargs)
    return should_early_stop


def _conf_plan(z, probe: bool):
    """The confidence stage's placement of the trunk shard ``z`` (``heads.ZTrunkPlan``; ``ROWPAIR_FREE_ZTRUNK`` /
    ``ROWPAIR_CONF_PARK_ZTRUNK`` decide, the form rule is the core's): RF3 runs the head ONCE PER DIFFUSION SAMPLE (RF3.py:493-507), so the item's
    plan has ``passes = D`` (``input["t"].shape[0]``, RF3.py:406) of one sample each and PERSISTS across those calls (:func:`_conf_plan_item`:
    pass ``i`` is the i-th call; a parked shard's host copy is re-served, never re-parked; the pass-0 global-LayerNorm moments are re-used). The
    recycle-0 early-stop probe (RF3.py:413-440) is its own one-pass plan with ``last_use=False``: recycles 1.. read AND write the shard after it,
    so its pass ends with the device tensor restored (:func:`_conf_plan_end`). Returns ``{"plan", "i", "moments", "passes", "probe"}``."""
    if probe:
        H = _rp("heads")
        return {"plan": H.ZTrunkPlan(z, passes=1, name="z_trunk.probe", log=_rp("dist").comm().log, pool=_park_pool(z)), "i": 0, "moments": None,
                "passes": 1, "probe": True}
    return _conf_plan_item(z)


def _conf_plan_item(z) -> dict:
    """The item's plan over the trunk shard ``z`` (``CTX["conf_plan"]``), created at its first use — the ROLL-OUT ENTRY (:func:`_distogram_forward`
    parks the shard there, ``plan.park_now()``: the diffusion conditioning then reads the trunk rows from the park, :func:`_ztrunk_source`, and the
    D confidence passes are served from the same host copy) or the first confidence pass. A plan over another tensor is the previous item's: closed."""
    st = CTX.get("conf_plan")
    if st is not None and st["plan"].z is not z:                                              # another trunk shard: the earlier item's plan is over
        _conf_plan_close()
        st = None
    if st is None:
        D = CTX.get("n_samples")
        if D is None or int(D) < 1:
            raise _rp().RowpairRefused("refused: the trunk shard's placement under n_gpu>1 spans the item's per-sample confidence passes and needs "
                                       "their count D = input['t'].shape[0] (RF3.py:406); the model entry recorded none")
        H = _rp("heads")
        st = {"plan": H.ZTrunkPlan(z, passes=int(D), name="z_trunk", log=_rp("dist").comm().log, pool=_park_pool(z)), "i": 0, "moments": None,
              "passes": int(D), "probe": False}
        CTX["conf_plan"] = st
    return st


def _ztrunk_source(z):
    """What a row statement between the trunk and the confidence stage reads the trunk rows from: the item's live park (``plan.source()``: a
    ``template.ParkedStorage`` serving ``.zrows`` while the device storage is released) or the shard tensor itself. A tensor OTHER than the
    plan's shard while that shard is parked (a caller's ``.float()`` / ``.detach()`` copy of a tensor whose storage is released would carry no
    data) is refused by name."""
    st = CTX.get("conf_plan")
    if st is None:
        return z
    if st["plan"].z is z:
        return st["plan"].source()
    src = st["plan"].source()
    if hasattr(src, "zrows"):
        raise _rp().RowpairRefused(f"refused: the diffusion conditioning was handed a pair tensor {tuple(z.shape)} that is not the item's trunk shard while that "
                                   "shard is PARKED (device storage released; ROWPAIR_CONF_PARK_ZTRUNK): a copy made from it holds no data — the hoisted "
                                   "conditioning (DiffusionConditioning.forward_hoisted) receives the shard object itself")
    return z


def _conf_plan_end(st: dict, probe: bool, moments) -> None:
    """Close pass ``st["i"]`` after ``run_confidence_sharded`` returned (its pair input is dead). Probe: ``device_needed_next=True`` — the later
    recycles read and write the trunk shard whole, so a parked shard is restored into its own storage (every view valid). Sample passes:
    ``device_needed_next=False`` — no statement reads ``Z_II`` between two confidence calls or after the last (RF3.py:493-546: the diffusion of
    every sample ran before the loop; the outputs carry no pair tensor), so a park stays live for the next pass and its host copy is dropped at
    the last, the device storage staying released. Census ``conf_ztrunk`` / ``conf_ztrunk_probe``."""
    plan, i = st["plan"], int(st["i"])
    if st["moments"] is None:
        st["moments"] = moments
    plan.end(i, device_needed_next=bool(probe))
    if probe:
        _rp("evidence").record_schedule(conf_ztrunk_probe=str(plan.words[0]))
        return
    st["i"] = i + 1
    if st["i"] >= st["passes"]:
        CTX["conf_plan"] = None                                                                # plan.end(last) closed it: host copy dropped, device storage released (consumed)


def _conf_plan_close() -> None:
    """Drop the item's confidence plan (a live host copy is released; idempotent)."""
    st = CTX.get("conf_plan")
    CTX["conf_plan"] = None
    if st is not None:
        if int(st["i"]) < int(st["passes"]):                                                    # fewer confidence calls than the plan's D passes: named, never silent
            _rp("evidence").record_schedule(conf_ztrunk_incomplete=f"{int(st['i'])}/{int(st['passes'])}")
        st["plan"].close()


def _compile_expected(prev, plddt_logits, pae_logits, pde_logits, *args, **kwargs):
    """``compile_af3_style_confidence_outputs`` under n_gpu>1: RF3's function VERBATIM (``tp_conf.compile_on_expected``) with the pre-computed
    expected PAE / PDE matrices the sharded head returned under the logits' keys (HOST tensors on rank 0: the function builds its pair masks on
    ``pae.device``, so the pair statistics run on the host and no output matrix is ever device-resident); ``plddt_logits`` and the atom
    arguments stay where the engine put them (its per-atom statements index them with device tensors)."""
    if not CTX["installed"]:
        return prev(plddt_logits, pae_logits, pde_logits, *args, **kwargs)
    from . import tp_conf
    PE = sys.modules["rf3.utils.predicted_error"]
    _count("compile")
    return tp_conf.compile_on_expected(PE, plddt_logits, pae_logits, pde_logits, *args, compile_fn=prev, **kwargs)


def _tm_metrics(kind: str):
    """``ComputePTM.compute`` / ``ComputeIPTM.compute`` under n_gpu>1: the per-sample TM scalars the sharded head's reducer finished
    (``tp_conf.metrics_from_tm``; ``{}`` on ranks > 0)."""
    def compute(self, prev, *args, **kwargs):
        if not CTX["installed"]:
            return prev(self, *args, **kwargs)
        from . import tp_conf
        tms = CTX["tm"]
        _count(f"metric_{kind}")
        if _rp("launch").local_rank() != 0 or not tms or any(t is None or any(v is None for v in t.values()) for t in tms):
            return {}                                                                       # ranks > 0 hold no TM statistics (rank 0's table); their tail is scratch
        return tp_conf.metrics_from_tm(tms)[kind]
    compute.__name__ = f"compute_{kind}"
    return compute


# ------------------------------------------------------------------------------------------------------------------------ diffusion
SPREAD_REFUSE_A = 1.0                                                                    # the cross-rank coordinate spread at the sampler exit above which the
                                                                                         # run is refused by name (replicated-by-recompute drift is ~1e-3 A)


def _denoiser_forward(self, prev, X_noisy_L, *args, **kwargs):
    """``DiffusionModule.forward`` (the denoiser, once per step): its input state ``X_noisy_L`` (positions after noise + augmentation)
    becomes rank 0's on every rank BEFORE the stock body — written IN PLACE into the caller's tensor, which is the sampler's loop variable
    (``inference_sampler.py``: ``delta_L = (X_noisy_L - X_denoised_L) / t_hat``; ``X_L = X_noisy_L + step_scale * d_t * delta_L`` read the
    same object), so every rank integrates rank 0's state and replicated-by-recompute drift (atom attention, SDPA under non-deterministic
    kernels) never accumulates across ranks (:func:`_sync_mode`: rank 0's draw replaces every rank's, then the guard). Then the stock body."""
    if not CTX["installed"]:
        return prev(self, X_noisy_L, *args, **kwargs)
    _count("denoiser")
    mode = _sync_mode()
    synced = _rp("diffusion").sync_replicated(X_noisy_L.contiguous(), "diffusion_state", mode=mode)
    if synced is not X_noisy_L:                                                          # the sync LANDS ON THE SAMPLER'S LOOP TENSOR: the caller's
        X_noisy_L.copy_(synced)                                                          # X_noisy_L object is what its update statement reads next
    CTX["state_syncs"] = CTX.get("state_syncs", 0) + 1                                   # (delta_L = (X_noisy_L - X_denoised_L) / t_hat), so rank 0's
    _rp("evidence").record_schedule(diff_noise=("bcast_rank0_state" if mode == "bcast" else f"{mode}_state"),   # state is integrated on every rank
                                    diff_state_syncs=CTX["state_syncs"], diff_state_inplace=1)
    return prev(self, X_noisy_L, *args, **kwargs)


def _rank_spread_then_sync(tag: str, X):
    """Sampler exit → confidence entry: record the cross-rank coordinate spread ``max |x_rank - x_rank0|`` (Å; one all-reduce) as
    ``diff_rank_spread_A`` and make ``X`` rank 0's on every rank (:func:`_sync`); a spread above ``SPREAD_REFUSE_A`` is refused by name."""
    torch = _torch()
    _mem_mark("sampler")
    mine = X.detach().clone()
    _sync(tag, {"X": X})                                                                     # in place: X is rank 0's now
    d = (torch.nan_to_num(mine.float()) - torch.nan_to_num(X.detach().float())).abs().max().reshape(1)
    _rp("dist").allreduce_(d, op="max")
    spread = float(d.item())
    CTX["spread_A"] = max(CTX.get("spread_A", 0.0), spread)
    _rp("evidence").record_schedule(diff_rank_spread_A=round(CTX["spread_A"], 6))
    if spread > SPREAD_REFUSE_A:
        raise _rp().RowpairRefused(f"refused: diffusion samples differ across ranks by {spread:.3f} A at the sampler exit (> {SPREAD_REFUSE_A} A): "
                                   "the per-step state broadcast did not hold")
    return X


def _diff_cond_forward(self, prev, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):
    """``DiffusionConditioning.forward_hoisted`` with the pair conditioning on this rank's rows, once per roll-out: ``to_zii(cat[z rows
    .float(), relpos rows])`` then the two pair transitions added per row block (``diffusion.pair_cond_rows``), stored in the kit's hoist
    cache under the key the kit's statement reads (``(id(self), "Z_II")``) — so the kit's own ``forward_hoisted`` runs next, finds the shard,
    and computes the single conditioning with its own statements (replicated). Refused by name: the hoist off (``RF3_HOIST=0``: the stock
    per-step pair conditioning is a dense statement)."""
    lay = CTX["layout"]
    if not CTX["installed"] or not _is_shard(Z_trunk_II, lay):
        _count("diff_cond", "replicated")
        return prev(self, t, f, S_inputs_I, S_trunk_I, Z_trunk_II)
    torch = _torch()
    diffusion = _rp("diffusion")
    GF = sys.modules.get("rf3.graph_flags")
    if GF is None or not getattr(GF, "HOIST", False) or getattr(GF, "HOIST_CACHE", None) is None or getattr(prev, "__name__", "") != "forward_hoisted":
        raise _rp().RowpairRefused("refused: diffusion conditioning under n_gpu>1 needs the kit's hoist (RF3_HOIST=1, forward_hoisted inside a "
                                   "roll-out): the per-step dense pair conditioning is not a row statement")
    _count("diff_cond")
    rpe = self.relative_position_encoding

    def embed_fn(z_rows, g0, g1):
        zr = z_rows.float() if z_rows.dtype in (torch.float16, torch.bfloat16) else z_rows
        x = torch.cat([zr, _relpos_rows(rpe, f, g0, g1).to(zr.dtype)], dim=-1)
        return self.to_zii(x)

    transitions = [(lambda x_rows, g0, g1, b=b: self.transition_1[b](x_rows)) for b in range(2)]

    def _pair():
        t0 = time.time()
        rows = diffusion.DiffusionSchedule.decide(lay, c_z=int(Z_trunk_II.shape[-1]), c_in=int(Z_trunk_II.shape[-1]) + int(rpe.linear.out_features),
                                                 c_cond=int(self.to_zii[-1].out_features) if hasattr(self.to_zii, "__getitem__") else int(Z_trunk_II.shape[-1]),
                                                 H=16, S=1, n_blocks=24, c_pair=16, elt=4, record=False).cond_rows   # elt 4: embed_fn's cat / LayerNorm /
        # to_zii input run in fp32 (z rows .float() + the one-hot rows) whatever the shard's dtype — sized at the shard's bf16 (2) the block's
        # transient would be twice the work budget the schedule names)
        src = _ztrunk_source(Z_trunk_II)                                                   # the trunk rows: the item's park (roll-out entry) or the device shard
        row0 = src.zrows(0, 1) if hasattr(src, "zrows") else Z_trunk_II[0:1]
        probe = embed_fn(row0, lay.r0, lay.r0 + 1)                                          # the conditioned dtype under the engine's autocast
        out = probe.new_empty((lay.R, lay.N, int(probe.shape[-1])))
        del probe, row0
        z = diffusion.pair_cond_rows(embed_fn, src, lay, c_out=int(out.shape[-1]), transitions=transitions, rows=rows, out=out)
        CTX["timing"]["diffusion"].append(round(time.time() - t0, 3))
        _mem_mark("diff_cond")
        return z

    GF.hoist_get((id(self), "Z_II"), _pair)                                                # the kit's statement reads this key next
    return prev(self, t, f, S_inputs_I, S_trunk_I, Z_trunk_II)


def _n_relpos(rpe) -> int:
    """Feature count of RF3's relative-position one-hots: two ``2 r_max + 2`` terms, the same-entity flag, one ``2 s_max + 2`` term."""
    return 2 * (2 * int(rpe.r_max) + 2) + 1 + (2 * int(rpe.s_max) + 2)


def _dit_fns(block):
    """``diffusion.DiTBlockFns`` of one RF3 ``DiffusionTransformerBlock``: norm = AdaLN(a, s) (+ the bf16 cast), kv = K (LayerNorm'd) / V of all
    rows, attn = the pair-biased attention of the query rows (bias rows from the core, heads last as the stock einsum adds them), update = the
    AdaLN-zero output gate from the s rows, the residual and the conditioned transition of the rows."""
    torch = _torch()
    diffusion = _rp("diffusion")
    apb = block.attention_pair_bias
    ctb = block.conditioned_transition_block
    if apb.use_deepspeed_evo:
        raise _rp().RowpairRefused("refused: use_deepspeed_evo attention under n_gpu>1")
    H, c = int(apb.n_head), int(apb.c)

    def norm(a, s):
        x = apb.ada_ln_1(a, s)
        if apb.force_bfloat16 and x.device.type != "mps":
            x = x.to(torch.bfloat16)
        return x

    def kv(x):
        K = apb.to_k(x)
        if apb.kq_norm:
            K = apb.key_layer_norm(K.reshape(-1, H * c)).reshape(K.shape)
        return (K, apb.to_v(x))

    def attn_engine(x_q, kv_, bias_q, g):                                                # RF3's statement (the query block's [q, N, H] fp32 logits + softmax):
        K, V = kv_                                                                          # the rows face's fallback BY NAME, and the whole path on cores
        Q = apb.to_q(x_q)                                                                   # without the face / under ROWPAIR_DIT_ROWS=engine
        if apb.kq_norm:
            Q = apb.query_layer_norm(Q.reshape(-1, H * c)).reshape(Q.shape)
        G = apb.to_g(x_q)
        Q = Q / math.sqrt(c)
        B = bias_q.movedim(-3, -1)                                                         # [*, H, q, N] -> [*, q, N, H]
        A = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q, K) + B, dim=-2)
        o = torch.einsum("...ijh,...jhc->...ihc", A, V)
        o = (G * o).flatten(start_dim=-2)
        return apb.to_a(o)

    rows_word = _dit_rows_word()                                                            # the query rows' pair-biased attention
    attn = attn_engine                                                                      # on the rectangular rows face (opt_core.kernels.apb.pair_bias_attention_rows:
    kv_engine = kv                                                                          # q = this rank's rows, keys / values = all N rows, the [H, q, N] bias rows read
    if rows_word != "engine" and hasattr(diffusion, "dit_attention_rows"):                 # in place — no [q, N, H] fp32 logits), the engine statement its fallback by
        def kv(x):                                                                          # name (census dit_rows=<row>:<n>[,stock:<event>:<n>], dit_rows_core=)
            K, V = kv_engine(x)
            return diffusion.DitKV(diffusion.cast16(K.movedim(-2, -3)), diffusion.cast16(V.movedim(-2, -3)), None, (K, V))   # [.., S, H, N, c] views; RF3's
                                                                                            # token attention has no key mask; stock = the engine's (K, V)
        def q_fn(x_q):                                                                      # the engine's query projection, pre-scaled -> scale=1.0; [.., S, H, q, c] view
            Q = apb.to_q(x_q)
            if apb.kq_norm:
                Q = apb.query_layer_norm(Q.reshape(-1, H * c)).reshape(Q.shape)
            return (Q / math.sqrt(c)).movedim(-2, -3)

        def out_fn(o, x_q):                                                                 # the engine's epilogue: sigmoid gate rows x heads, output projection
            G = apb.to_g(x_q)
            return apb.to_a((G * o.reshape(*o.shape[:-1], H, c)).flatten(start_dim=-2))

        attn = diffusion.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=attn_engine, num_heads=H, core_word=rows_word, scale=1.0, kind="dit")

    def update(a, o_rows, s, rr):
        r0, r1 = rr
        a_r = a[..., r0:r1, :]
        s_r = s[..., r0:r1, :]
        b = apb.linear_output_project(s_r) * o_rows
        if block.no_residual_connection_between_attention_and_transition:
            return a_r + b + ctb(a_r, s_r)
        a_r = a_r + b
        return a_r + ctb(a_r, s_r)

    def bias(z_rows):
        return apb.to_b(apb.ln_0(z_rows))

    if hasattr(diffusion, "DitBias"):                                                       # the block's pair-bias rows producer as WEIGHTS + the engine statement —
        bias_into = diffusion.DitBias(engine=bias, ln_weight=getattr(apb.ln_0, "weight", None), ln_bias=getattr(apb.ln_0, "bias", None),   # under
                                      weight=apb.to_b.weight, linear_bias=getattr(apb.to_b, "bias", None), eps=float(getattr(apb.ln_0, "eps", 1e-5)))
        return diffusion.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias, bias_into=bias_into)   # ROWPAIR_DIFF_BIAS=ln_proj (TP_ENV)
    return diffusion.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias)   # the core writes LN + projection head-major into the [H, rows, N]
                                                                                            # slice in one pass (no [rows, N, c_z] fp32 LN output); the engine statement is the bitwise form


def _dit_forward(self, prev, A_I, S_I, Z_II, Beta_II):
    """``DiffusionTransformer.forward`` (token level) on the conditioned shard: query rows local per block, K / V from all rows, the block's pair
    bias projected from this rank's rows once per roll-out (``diffusion.PairBiasCache`` held in the kit's hoist store), the updated rows
    all-gathered per block (``diffusion.diffusion_transformer_sharded``). The atom transformer (``Beta_II`` set: windowed atom pair) is the
    kit's forward."""
    lay = CTX["layout"]
    if not CTX["installed"] or Beta_II is not None or not _is_shard(Z_II, lay):
        return prev(self, A_I, S_I, Z_II, Beta_II)
    diffusion = _rp("diffusion")
    GF = sys.modules.get("rf3.graph_flags")
    _count("dit")
    key = ("dit", id(self))
    fns = CTX["fns"].get(key)
    if fns is None:
        fns = [_dit_fns(b) for b in self.blocks]
        CTX["fns"][key] = fns
    apb0 = self.blocks[0].attention_pair_bias

    def _sched():                                                                            # once per roll-out on every rank (hoisted): the pair-bias cache
        fit = _dit_bias_cache_fit(n_blocks=len(fns), H=int(apb0.n_head), lay=lay, elt=int(Z_II.element_size()))   # decision from the AGREED free bytes
        S_ = int(A_I.shape[-3]) if A_I.dim() >= 3 else 1
        extra = {}
        if _dit_rows_word() != "engine" and hasattr(diffusion, "dit_rows_core"):          # the rows face's static admission once per roll-out (census
            attn_core, _sel, _why = diffusion.dit_rows_core(_dit_rows_word(), dtype=_torch().bfloat16, heads=int(apb0.n_head), head_dim=int(apb0.c),
                                                            samples=S_, kind="dit", device=A_I.device)   # dit_rows_core=); admitted -> one call per
            extra["attn_core"] = attn_core                                                   # block over all local query rows (q_rows = R, source q:kernel)
        return diffusion.DiffusionSchedule.decide(lay, c_z=int(Z_II.shape[-1]), c_in=int(Z_II.shape[-1]), c_cond=int(Z_II.shape[-1]),
                                                  H=int(apb0.n_head), S=S_, n_blocks=len(fns),
                                                  c_pair=16, elt=4, bias_cache=fit, **extra)  # elt 4: the blocks' logits / LayerNorm outputs are fp32 whatever z_cond is stored in

    hoisting = GF is not None and getattr(GF, "HOIST_CACHE", None) is not None
    sched = GF.hoist_get((id(self), "rowpair_schedule"), _sched) if hoisting else _sched()
    cache = GF.hoist_get((id(self), "rowpair_bias_cache"), lambda: diffusion.PairBiasCache(enabled=sched.bias_cache)) if hoisting else None
    return diffusion.diffusion_transformer_sharded(fns, A_I, S_I, Z_II, lay, schedule=sched, bias_cache=cache)


def pair_windows_banded(process_z: Callable, Z_II, tq, tk):
    """The atom encoder's token-pair window term ``process_z(Z)[tq, tk]`` (``[nqb, 32, 128, c_atompair]``) from the conditioned SHARD: the
    projected band ``[I, 2W+1, c]`` of every rank's rows all-gathered once per roll-out call (``diffusion.pair_band_rows``), then the window
    lookup (``band_lookup``). A dense pair (no active layout) is the lever's own two lines (``levers.pair_windows_dense``)."""
    lay = CTX["layout"]
    from . import levers as _levers
    if not CTX["installed"] or not _is_shard(Z_II, lay):
        return _levers.pair_windows_dense(process_z, Z_II, tq, tk)
    torch = _torch()
    diffusion = _rp("diffusion")
    _count("pair_windows")
    key = ("plan", tuple(tq.shape), lay.N)
    plan = CTX["band"].get(key)
    if plan is None:
        valid = torch.ones(tuple(tq.shape) + (int(tk.shape[-1]),), dtype=torch.bool, device=tq.device).unsqueeze(0)
        plan = diffusion.band_plan(tq.unsqueeze(0), tk.unsqueeze(0), valid, lay.N, max_w=None)
        CTX["band"][key] = plan
        CTX["band"]["W"] = int(getattr(plan, "W", -1))
    band, extras = diffusion.pair_band_rows(process_z, Z_II, lay, plan)
    return diffusion.band_lookup(band, extras, plan)[0]


# ------------------------------------------------------------------------------------------------------------------------ noise guards
NOISE_SYNC = "bcast"                                                                     # diffusion.sync_replicated mode: rank 0's replicated tensor REPLACES
                                                                                         # every rank's (then the guard: a rank whose copy differs is refused by name)


def _sync_mode() -> str:
    """The replicated-tensor sync policy of this process (``NOISE_SYNC``), recorded as ``noise_sync``."""
    _rp("evidence").record_schedule(noise_sync=NOISE_SYNC)
    return NOISE_SYNC


def _noise_init_guard(self, prev, *args, **kwargs):
    """The stock initial-structure draw; under n_gpu>1 rank 0's draw is every rank's (``diffusion.sync_replicated(mode=_sync_mode())``:
    broadcast, then the cross-rank guard — refused by name on a mismatch)."""
    out = prev(self, *args, **kwargs)
    if CTX["installed"]:
        _count("noise_init")
        out = _rp("diffusion").sync_replicated(out, "diffusion_noise_init", mode=_sync_mode())
    return out


def _predraw_guard(self, prev, *args, **kwargs):
    """The kit sampler's pre-drawn random tensors of the roll-out; under n_gpu>1 every tensor it returns becomes rank 0's (broadcast + guard)."""
    out = prev(self, *args, **kwargs)
    if CTX["installed"]:
        _count("noise_predraw")
        torch = _torch()
        sync = _rp("diffusion").sync_replicated
        if isinstance(out, dict):
            for k in list(out):
                if isinstance(out[k], torch.Tensor):
                    out[k] = sync(out[k], f"diffusion_step_draws.{k}", mode=_sync_mode())
        elif isinstance(out, list):
            for k, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    out[k] = sync(v, f"diffusion_step_draws.{k}", mode=_sync_mode())
        elif isinstance(out, tuple):
            out = tuple(sync(v, f"diffusion_step_draws.{k}", mode=_sync_mode()) if isinstance(v, torch.Tensor) else v for k, v in enumerate(out))
        elif isinstance(out, torch.Tensor):
            out = sync(out, "diffusion_step_draws", mode=_sync_mode())
        else:
            raise _rp().RowpairRefused(f"refused: the sampler's pre-draw returned {type(out).__name__} (dict / list / tuple / Tensor expected) under n_gpu>1")
    return out


