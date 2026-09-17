"""openfold3_opt.tp_rowpair — the big/tp line's ADAPTER onto ``opt_core.mem.rowpair`` (opt_core >= 0.4.3): OpenFold3 0.4.1's module
names, attribute paths, config plumbing and control flow (``run_trunk`` / ``_rollout`` statement order) bound onto the core's row-sharded
pair-representation statements. Every tensor statement, layout, collective and block schedule is the core's; this package re-implements
none of them (one implementation: ``opt_core.mem.rowpair``).

Layout (the core's words): under ``pred --mode big --n_gpu P`` (P in 2|4|8) each of the P rank processes holds the pair representation
as a ROW SHARD ``z_loc = z[:, r0:r1, :, :]`` (``[1, n_loc, N, 128]`` fp32) from the statement that creates it (the input embedder's outer
sum, ``trunk.input_embedder_rows``) through recycling, the template embedder, the MSA module, the Pairformer, the confidence head and
diffusion conditioning; nothing ``N x N x c`` is ever whole on a rank. Row boundaries are multiples of the run's pinned chunk
(``OF3TP_CHUNK`` -> ``ROWPAIR_CHUNK_ALIGN``), the last rank ragged (``opt_core.mem.rowpair.dist.ctx(N, align=chunk)``, the refusing form: a rank that would own no rows is refused by name).
The pair representation is row-sharded at every size (there is no replicated-pair work-split path in this kit).

REPLICATED BY DESIGN on every rank (named here and in the run's schedule census, ``model.census_fields``): the single representation
``s`` / ``s_input`` ``[1, N, 384|449]``, the MSA representation ``m`` ``[1, 1024, N, 64]`` (the MSA embedder's random subsample runs
with the identical RNG stream on every rank and is checksummed), every atom-level tensor (atom attention encoder/decoder windows, diffusion
coordinates and noise), the triangle-attention bias ``[1, N, N, 4]`` per call (all-gathered from row slices), the DiffusionTransformer
K/V projections and token activations, ``plm_z`` (the atom-pair token term), and rank 0's full ``[N, N]`` PAE / PDE / contact matrices at
the writer. Raw MSA features (``msa``, ``has_deletion``, ``deletion_value``) live on rank 0's HOST and ``token_bonds`` on every rank's host
(``msa.MSA_HOST`` / ``msa.BONDS_ON_HOST``); the selected MSA rows are broadcast per recycle.

The structural n_gpu=1 rule (``opt_core/mem/rowpair/API.md``): at ``--n_gpu 1`` this package INSTALLS NOTHING — no layout, no PatchSet,
no rebinding; the kit's one-GPU big line (``resident``) runs byte for byte. ``install()`` refuses by name when the
process is not a rank of a P>1 group; the core's sharded statements refuse a P=1 layout by name themselves.

Modules: ``env`` (the kit-surface ``OF3TP_*`` switch names -> the core's ``ROWPAIR_*`` names), ``core`` (the one import surface of the
core's seams; a missing seam refuses by name with the opt_core version), ``trunk`` (``run_trunk`` on shards), ``template``, ``msa``
(+ the host-resident MSA embedder hook), ``pairstack`` (PairBlock / PairFormerBlock / PairFormerStack bindings onto the core's ONE
pair-block driver), ``confidence`` (AuxiliaryHeadsAllAtom on shards -> the core's row-block reducer), ``diffusion`` (SampleDiffusion on
``z_cond`` rows + the sampler-hook client), ``model`` (the PatchSet: ``OpenFold3.run_trunk`` / ``_rollout``, the Lightning module's
``_compute_confidence_scores`` / ``transfer_batch_to_device``, ``OF3OutputWriter.on_predict_batch_end``, ``MSAModuleEmbedder.forward``,
the batch sync at ``OpenFold3.forward`` entry), ``hook/sitecustomize.py`` (the rank process's import hook: DET recipe, install on import).
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional

LEVER = "tp"                                    # the kit's line name; the core's strategy id rides the LEVER line (rowpair.LEVER)
TAG = "openfold3-opt tp"
FEATS_TAG = "[feats] rank "                      # the cross-rank input digest gate's per-rank transcript line (model.feature_digest), one per forward:
                                                #   `[feats] rank <r> shared <d16> keys <n> … ranks_identical=True|False|n/a; <s>s)` — spelled here once, read back by the launcher (tp.parse_rank_log)
FEATS_IDENTICAL = "ranks_identical"             # that line's verdict word (opt_core rankdata.ranks_agree: one answer on every rank of the group; n/a without a multi-rank group);
                                                #   the launcher's census carries it per rank and over the ranks (tp.census `ranks_identical`, the `census …` line)

__all__ = ["LEVER", "TAG", "FEATS_TAG", "FEATS_IDENTICAL", "world_of", "rank_of", "is_rank_process", "active", "install", "uninstall", "installed_names", "NotARank"]


class NotARank(RuntimeError):
    """``install()`` in a process that is not a rank of a P>1 group: the adapter installs nothing at n_gpu=1 (structural rule)."""


def world_of(environ: Optional[Mapping[str, str]] = None) -> int:
    """The group size this process belongs to: ``ROWPAIR_WORLD`` (the core launcher's name) else ``OF3TP_WORLD`` (the kit surface), 1 when
    neither is set or readable."""
    environ = os.environ if environ is None else environ
    for k in ("ROWPAIR_WORLD", "OF3TP_WORLD"):
        v = (environ.get(k) or "").strip()
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                return 1
    return 1


def rank_of(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    environ = os.environ if environ is None else environ
    for k in ("ROWPAIR_RANK", "OF3TP_RANK"):
        v = (environ.get(k) or "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                return None
    return None


def is_rank_process(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True iff this process is one rank of a P>1 group (world > 1 AND a rank index set)."""
    return world_of(environ) > 1 and rank_of(environ) is not None


def active() -> bool:
    """True iff the PatchSet is installed in this process (see ``model.PATCHES``)."""
    return bool(installed_names())


def install(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Install the tp line in THIS rank process: export the core's env names (``env.export_core_env``), then rebind OpenFold3's seams
    (``model.install``). Returns the ``Owner.name`` list patched. Refuses by name (``NotARank``) outside a P>1 rank process — the
    adapter installs nothing at n_gpu=1."""
    environ = os.environ if environ is None else environ
    if not is_rank_process(environ):
        raise NotARank(f"{TAG}: refused: install() outside a rank process (world={world_of(environ)} rank={rank_of(environ)}): "
                       f"the tp line installs nothing at n_gpu=1 — `pred --mode big --n_gpu P` (P in 2|4|8) spawns the ranks")
    from . import core as _core, env as _env, model
    _env.export_core_env()
    _core.fn("dist", "apply_rank_threads")()                  # ROWPAIR_RANK_THREADS (the launcher lays `auto` = cpus // P: tp.rank_env) -> torch.set_num_threads + OMP/MKL/OPENBLAS caps + census rank_threads; unset/inherit = a no-op
    return model.install()


def uninstall() -> List[str]:
    from . import model
    return model.uninstall()


def schedule_record() -> dict:
    """This rank's schedule evidence for its run record (``tp_schedule``): the core's recorded schedule words
    (``opt_core.mem.rowpair.evidence.schedule_fields``: layout / block sizes with their source, ``diff_noise_sync*``, ``msa_draw*``, ``trunk_guard``,
    ``diff_rank_spread_A``, ``sample_*``), the adapter's census fields (``model.census_fields``) and the sample_loop lever's last fields. Empty
    parts when nothing ran in this process."""
    out = {"schedule": {}, "census": {}, "sample_loop": {}}
    try:
        from . import core as C, model
        out["schedule"] = {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in C.fn("evidence", "schedule_fields")()}
        out["census"] = model.census_fields(model.LAST_LAYOUT)
    except Exception as e:                                   # noqa: BLE001 — evidence is a record, never a gate: the reason is written instead
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        from .. import sample_loop as SLB
        out["sample_loop"] = dict(SLB.LAST)
    except Exception as e:                                   # noqa: BLE001
        out["sample_loop_error"] = f"{type(e).__name__}: {e}"
    try:                                                     # the schedule census words in the rank's transcript (the run record itself is transient: the launcher reads
        import sys as _sys                                   #  rank 0's and removes the records directory) -- conf_reducer / conf_ctx / conf_index_s / conf_gathers / conf_finalize_s /
        words = dict(out.get("schedule") or {})              #  conf_reducer_mismatch, rank_threads, host_slab*, zinit_park / park_z_init, ztrunk_retired, diff_* ... one grep-able line
        _sys.stderr.write("[openfold3-opt tp rank %s] SCHEDULE %s\n" % (rank_of(os.environ), " ".join("%s=%s" % (k, v) for k, v in words.items())))
    except Exception:                                        # noqa: BLE001 - a census print never gates
        pass
    return out


def installed_names() -> List[str]:
    from . import model
    return list(model.PATCHES.names())
