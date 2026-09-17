"""The tp line's install record (``PATCHES``, an ``opt_core.mem.patchset.PatchSet``) and the model-level control flow it rebinds:

  ``OpenFold3.run_trunk``                  -> ``trunk.run_trunk_rows`` (z row-sharded from birth)
  ``OpenFold3._rollout``                   -> ``rollout_rows`` (``model.py:318-420`` statements verbatim: ``sample_diffusion`` on ``z_cond``
                                              rows (``diffusion.sample_diffusion_rows``), the auxiliary heads on the shard
                                              (``confidence.aux_heads_rows``), the confidence dict reduced inside the rollout)
  ``OpenFold3.forward`` (entry)            -> ``forward_synced``: every rank's batch tensors are overwritten with rank 0's
                                              (``opt_core.mem.rowpair.bcast.sync_tensordict_from_rank0``) so a featurizer that is not
                                              rank-deterministic cannot desynchronise the ranks; mismatches are named in the log
  ``OpenFold3AllAtom._compute_confidence_scores`` -> returns the dict the rollout already reduced (``_tp_confidence``)
  ``OpenFold3AllAtom.transfer_batch_to_device``   -> keeps the host-resident feature keys on host (``msa.host_keys``)
  ``MSAModuleEmbedder.forward``            -> ``msa.msa_module_embedder_forward_host``
  ``OF3OutputWriter.on_predict_batch_end`` -> rank 0 writes; ranks > 0 no-op (named once in their log)

``census_fields()`` is the schedule census the rank's ACTIVE record and the launcher's manifest carry: layout (N, P, rows, align),
every block size the run derived (and from what), and the replicated-by-design tensors by name.
"""
from __future__ import annotations

import multiprocessing
import contextlib
import os
import sys
import random
import time
from typing import Dict, List

import torch

from . import LEVER, FEATS_TAG, FEATS_IDENTICAL, core as C

from opt_core.mem.patchset import PatchSet

PATCHES = PatchSet(LEVER)
GROUP: Dict[str, object] = {}                               # the rank's process-group record once joined (world, rank, backend, group_ready): the kit's probe reads it

REPLICATED_BY_DESIGN = ("s", "s_input", "m(msa rep; identical RNG stream, checksummed)", "atom tensors", "triangle_bias[N,N,4] per call (all-gathered)",
                        "DiT K/V + token activations", "plm_z", "rank0 PAE/PDE/contact matrices on HOST at the writer (never on device)")
HOST_RESIDENT = ("msa/has_deletion/deletion_value (rank 0)", "token_bonds (every rank)", "z_init_loc between recycles (ROWPAIR_PARK_ZINIT)",
                 "z_trunk shard from roll-out entry through the last confidence pass (ROWPAIR_CONF_PARK_ZTRUNK; heads.ZTrunkPlan, census conf_ztrunk)")


# ----------------------------------------------------------------------------------------------------------------- sampler exit
SPREAD_REFUSE_A = 1.0
"""Cross-rank coordinate spread above which the rollout is refused by name (Angstrom): under non-deterministic kernels the replicated atom
tensors agree to ~1e-3 A; a larger spread means the ranks sampled different trajectories."""


class RolloutRefused(RuntimeError):
    pass


def adopt_rank0_coordinates(x):
    """Sampler exit: rank 0's predicted coordinates ``[*, S, N_atom, 3]`` replace every rank's own before the confidence head embeds them into
    its pair rows (identical inputs on every rank by construction, deterministic kernels or not). The cross-rank spread
    max|x_rank - x_rank0| is all-reduced, recorded (``diff_rank_spread_A``) and refused by name above ``SPREAD_REFUSE_A``."""
    c = C.comm()
    if not c.active:
        return x
    x0 = C.fn("bcast", "broadcast_tensordict")(x.contiguous(), src=0)
    x0 = x0.to(x.device) if x0.device != x.device else x0
    spread = (x.float() - x0.float()).abs().amax().reshape(1) if x.numel() else torch.zeros(1, device=x.device)
    c.allreduce_(spread, "max")
    val = float(spread.item())
    from . import msa as MSA
    C.fn("evidence", "record_schedule")(diff_noise=("guard" if MSA.sync_policy() == "guard" else "bcast_rank0_state"), sampler_exit="rank0_coordinates", diff_rank_spread_A=f"{val:.3g}")
    c.log(f"[model] sampler exit: rank 0's coordinates adopted on every rank; diff_rank_spread_A={val:.3g} (refused above {SPREAD_REFUSE_A})")
    if val > SPREAD_REFUSE_A:
        raise RolloutRefused(f"refused: diff_rank_spread_A={val:.3g} > {SPREAD_REFUSE_A}: the ranks' diffusion trajectories diverged (replicated state not identical across ranks)")
    return x0


# ----------------------------------------------------------------------------------------------------------------- LayerNorm launch guard
LN_GUARD: Dict[str, object] = {"state": "not_installed", "launches": 0, "split": 0, "unsplittable": 0, "shapes": []}
"""This process's LayerNorm-guard record: ``state`` (``installed:<classes>`` | ``not_installed``), LayerNorm launches routed, launches
the core's statement split into row blocks, and the first operand shapes it split (``census_fields`` prints them)."""
LN_GUARD_CLASSES = ("openfold3.core.model.primitives.normalization.LayerNorm", "torch.nn.LayerNorm")


def _normalized_dims(ln) -> int:
    """How many trailing dims the LayerNorm instance normalizes over (``torch.nn.LayerNorm.normalized_shape``; openfold3's primitive: ``c_in``)."""
    shape = getattr(ln, "normalized_shape", None)
    if shape is None:
        shape = getattr(ln, "c_in", None)
    if shape is None:
        return 1
    return len(tuple(shape)) if isinstance(shape, (tuple, list, torch.Size)) else 1


def ln_forward_guarded(orig):
    """``forward(self, x, ...)`` of a LayerNorm class through ``shard.ln_rows_guarded`` along the longest dim outside the instance's normalized dims
    (a 2-D operand rides as the view ``[1, M, C]``)."""
    guarded = C.fn("shard", "ln_rows_guarded")

    def forward(self, x, *a, **kw):
        if not torch.is_tensor(x) or x.dim() < 2:
            return orig(self, x, *a, **kw)
        LN_GUARD["launches"] += 1
        n_norm = _normalized_dims(self)
        x3 = x if x.dim() >= 3 else x.unsqueeze(0)
        if x3.dim() - n_norm < 1:                                                            # no dim outside the normalized ones: one launch, counted
            LN_GUARD["unsplittable"] += 1
            return orig(self, x, *a, **kw)
        rows_dim = max(range(x3.dim() - n_norm), key=lambda d: int(x3.shape[d])) - x3.dim()   # the longest dim OUTSIDE the normalized dims: the smallest row blocks, per-row statistics untouched
        pieces = [0]

        def ln(t):
            pieces[0] += 1
            return orig(self, t, *a, **kw)

        out = guarded(ln, x3, rows_dim=rows_dim)
        if pieces[0] > 1:
            LN_GUARD["split"] += 1
            shape = tuple(int(v) for v in x.shape)
            if shape not in LN_GUARD["shapes"] and len(LN_GUARD["shapes"]) < 8:
                LN_GUARD["shapes"].append(shape)
                C.comm().log(f"[ln_guard] LayerNorm on {shape} ({x.numel():,} elements) ran as {pieces[0]} row blocks along dim {rows_dim} (ROWPAIR_LN_GUARD_ELEMS)")
        return out if x.dim() >= 3 else out.squeeze(0)

    return forward


def install_ln_guard() -> str:
    """Rebind ``forward`` of ``LN_GUARD_CLASSES`` through ``ln_forward_guarded`` and say so once: every LayerNorm launch of the process is routed
    through the core's ``shard.ln_rows_guarded`` — one stock launch below ``ROWPAIR_LN_GUARD_ELEMS`` elements (default 2^31), row blocks along the
    operand's longest non-channel dim at or above it (``F.layer_norm`` on CUDA fp32 operands of >= 2^32 elements returns wrong rows; LayerNorm rows
    are independent, so the per-row arithmetic is unchanged). Records ``ln_guard`` / ``ln_guard_elems`` in the schedule census; returns the state word."""
    record = C.fn("evidence", "record_schedule")
    elems = int(C.fn("dist", "env_float")("ROWPAIR_LN_GUARD_ELEMS", float(C.seam("shard").LN_GUARD_ELEMS)))
    from openfold3.core.model.primitives.normalization import LayerNorm as OF3LayerNorm
    for cls in (OF3LayerNorm, torch.nn.LayerNorm):
        PATCHES.replace(cls, "forward", ln_forward_guarded(cls.forward))
    LN_GUARD["state"] = "installed:" + "+".join(LN_GUARD_CLASSES)
    record(ln_guard=LN_GUARD["state"], ln_guard_elems=elems)
    C.comm().log(f"[ln_guard] ln_guard installed: every LayerNorm launch ({', '.join(LN_GUARD_CLASSES)}) runs through shard.ln_rows_guarded; "
                 f"row blocks from {elems:,} elements (ROWPAIR_LN_GUARD_ELEMS; 0 = never)")
    return LN_GUARD["state"]


# ----------------------------------------------------------------------------------------------------------------- rebound methods
NCCL_TIMEOUT_S = 7200.0                                      # the process group's collective timeout (seconds): sized for the tp line's longest single-rank phases at the largest inputs (rank 0's confidence heads /
                                                            # output writing while the peers wait in a collective); a peer that died is noticed within it. ROWPAIR_NCCL_TIMEOUT_S is a diagnostic override only


def ensure_group() -> None:
    """Join this rank to the run's process group once (``opt_core.mem.rowpair.dist.init_from_env``: ``ROWPAIR_RANK / WORLD / ADDR / PORT`` as the
    launcher set them; backend NCCL on a host with CUDA devices, gloo on a host without; collective timeout ``NCCL_TIMEOUT_S``; the one visible
    GPU of this rank process is its device). Prints the ``process group ready: ...`` line the launcher's census reads."""
    from . import world_of
    if world_of() <= 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    timeout_s = float(os.environ.get("ROWPAIR_NCCL_TIMEOUT_S", "").strip() or NCCL_TIMEOUT_S)   # the line's constant; the env name is the core's diagnostic override (dist.init_from_env reads it too)
    device_index = 0 if (torch.cuda.is_available() and torch.cuda.device_count() == 1) else None
    P, r, device = C.fn("dist", "init_from_env")(backend=backend, timeout_s=timeout_s, device_index=device_index)
    got = torch.distributed.get_backend() if torch.distributed.is_initialized() else "none"
    nccl = ".".join(map(str, torch.cuda.nccl.version())) if got == "nccl" else "n/a"
    GROUP.update(world=int(P), rank=int(r), backend=got, group_ready=bool(torch.distributed.is_initialized()), device=str(device))
    C.comm().log(f"process group ready: backend={got} world={P} mode=S nccl={nccl} rank={r} device={device} timeout_s={timeout_s:g}")


def run_trunk_patched(self, batch, num_cycles, inplace_safe=False):
    from . import trunk
    ensure_group()
    N = int(batch["token_mask"].shape[-1])
    t0 = time.time()
    s_input, s, z_loc, lay = trunk.run_trunk_rows(self, batch, num_cycles, inplace_safe=inplace_safe)
    self._tp_layout = lay
    C.comm().log(f"[model] trunk done N={N} P={lay.P} rows {lay.r0}:{lay.r1} in {time.time() - t0:.1f}s")
    return s_input, s, z_loc


LAST_LAYOUT = None                                          # the layout of the last trunk/rollout this process ran (schedule_record reads it)


def sample_plan(n_samples: int, batch):
    """The sample loop's plan for this query (``sample_loop.plan_for``: ``sample_loop.CHUNK`` = 1 sample per pass), with the one input class the
    loop cannot split: a POCKET-CONDITIONED query (``pocket_constraints._pocket_sampling_enabled``: ``SampleDiffusion.forward`` restarts the roll-out
    from parents selected ACROSS the S samples, ``diffusion_module.py:442-472``) runs ONE batched pass of all S samples — named in the log and in the
    schedule census (``pocket_sampling=1 sample_loop_pocket=one_pass_forced``; ``one_pass`` when the plan already was one pass), never silent.
    Value-identical to stock either way; only the memory schedule differs."""
    from .. import sample_loop as SLB
    from openfold3.core.model.structure.pocket_constraints import _pocket_sampling_enabled
    plan = SLB.plan_for(int(n_samples))
    if not bool(_pocket_sampling_enabled(batch)):
        return plan
    if plan.passes > 1:
        C.log_fallback("sample_loop_pocket", "one_pass_forced", f"pocket_sampling=1: the sample loop's {plan.passes} passes of {plan.chunk} are replaced by ONE batched pass of "
                       f"{n_samples} samples (SampleDiffusion's pocket restart selects parents across all samples)")
        plan = SLB.core().plan(int(n_samples), int(n_samples), default_chunk=SLB.CHUNK)          # k == S: the one batched pass
        C.fn("evidence", "record_schedule")(pocket_sampling=1, sample_loop_pocket="one_pass_forced")
    else:
        C.fn("evidence", "record_schedule")(pocket_sampling=1, sample_loop_pocket="one_pass")
    return plan


def rollout_rows(self, batch, si_input, si_trunk, zij_trunk, inplace_safe=False):
    """``OpenFold3._rollout`` with ``zij_trunk`` = the shard ``[1, 1, n_loc, N, C]`` (the sample dim added by ``forward``). Statements verbatim; aux
    heads -> ``confidence.aux_heads_rows``.
    The roll-out and the heads run under the ``sample_loop`` lever (``openfold3_ob0_opt.sample_loop``: ``chunk`` of the S samples per pass, default 1,
    ``k == S`` = the one batched pass) — the per-pass body below is the same statements on
    samples ``[chunk.start, chunk.stop)``; at every pass's sampler exit rank 0's coordinates are adopted on every rank (``adopt_rank0_coordinates``).
    Rank 0 prints the lever's census line ``[sample_loop] samples= chunk= passes= draw_order= host= peak_gb=`` (``sample_loop.emit`` through this
    rank's logger) and the fields ride the schedule census (``schedule_record``)."""
    from openfold3.projects.of3_all_atom import model as _M
    from . import confidence as CONF, diffusion as DIFF
    from .. import sample_loop as SLB
    lay = getattr(self, "_tp_layout", None) or C.layout_of(int(batch["token_mask"].shape[-1]))
    global LAST_LAYOUT
    LAST_LAYOUT = lay
    from openfold3.core.utils.device_utils import autocast_device_type
    mode_mem_settings = self._get_mode_mem_settings()
    offload_confidence_heads = self._do_inference_offload(seq_len=batch["token_mask"].shape[-1], module_name=_M.OffloadModules.CONFIDENCE_HEADS.value)   # model.py:357-360; named in confidence.aux_heads_rows
    no_rollout_steps = self.shared.diffusion.no_mini_rollout_steps if self.training else self.shared.diffusion.no_full_rollout_steps
    no_rollout_samples = self.shared.diffusion.no_mini_rollout_samples if self.training else self.shared.diffusion.no_full_rollout_samples
    use_trunk_embedding = random.random() < self.shared.use_confidence_emb_prob if self.training else True
    kernel_kw = dict(use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention, use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                     use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels, use_lma=mode_mem_settings.use_lma)
    plan = sample_plan(int(no_rollout_samples), batch)
    last = plan.chunks[-1]
    C.mark("diffusion_start")
    device_type = autocast_device_type(si_input)            # model.py:371
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.float32):
        noise_schedule = _M.create_noise_schedule(no_rollout_steps=no_rollout_steps, **self.config.architecture.noise_schedule, dtype=si_input.dtype, device=si_input.device)
    from .. import conf_dtype as CD
    cast_dtype, pairformer_dtype = CD.head_dtypes(torch.float32 if self.training else si_trunk.dtype)   # model.py:419 `cast_dtype`; head_modules `pairformer_dtype=float32` — both bf16 under the conf_dtype lever (OPENFOLD3_OB0_OPT_CONF_DTYPE=bf16), upstream's otherwise
    one_pass = plan.passes == 1
    comm = C.comm()
    # The trunk shard's placement over the whole roll-out: ONE core plan spanning the sample loop's passes (heads.ZTrunkPlan; the words ride the
    # schedule census as conf_ztrunk=...). Order at entry: (1) the distogram head's contact rows from the DEVICE shard (column-slab exchange; the
    # trunk rows' only whole-tensor reader besides the embedding), (2) park_now(): under ROWPAIR_CONF_PARK_ZTRUNK=1 the shard is copied to pinned
    # host and its device storage released — from here the diffusion conditioning (diffusion.pair_cond_rows on plan.source()) and every confidence
    # pass (plan.begin) read trunk rows from the host by row block; nothing reads the device tensor whole again, so no pass restores it (zero
    # H2D round trips; the host copy is dropped after the last pass). Device holds ONE shard-equivalent through diffusion and confidence at any S.
    zplan = CONF.ztrunk_plan(zij_trunk, lay, passes=plan.passes, log=comm.log if comm.verbose else None)
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=cast_dtype):
        contact_loc = CONF.trunk_contact_rows(self.aux_heads, zplan.source(), lay, self.config, S=int(plan.chunks[0].stop - plan.chunks[0].start))
    park_word = zplan.park_now()
    comm.log(f"[model] roll-out entry: z_trunk shard {park_word} (ROWPAIR_CONF_PARK_ZTRUNK={int(zplan.park)} ROWPAIR_FREE_ZTRUNK={int(zplan.free)}; "
             f"{plan.passes} pass(es) of {no_rollout_samples} samples; confidence phase cast_dtype={str(cast_dtype).rsplit('.', 1)[-1]} pairformer_dtype={str(pairformer_dtype).rsplit('.', 1)[-1]} ({CD.census_line().split('] ', 1)[1]}))")
    zij_trunk_after = zij_trunk[..., :0, :, :]              # what output["zij_trunk"] becomes once the plan consumed the shard (zero rows; keep write_latent_outputs off)

    def diffuse(chunk, draws):                              # samples [chunk.start, chunk.stop) of S: [1, k, N_atom, 3], replicated (rank 0's on every rank at exit)
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.float32):   # model.py:371-395: sample_diffusion(...) with no kernel switches, use_high_precision_attention=True
            x = DIFF.sample_diffusion_rows(self.sample_diffusion, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=zplan.source(), lay=lay,
                                           noise_schedule=noise_schedule, no_rollout_samples=no_rollout_samples, use_conditioning=True,
                                           chunk_size=mode_mem_settings.chunk_size, use_high_precision_attention=True, _mask_trans=True,
                                           draws=None if one_pass else draws, chunk=None if one_pass else chunk)
            self.clear_autocast_cache()
        C.mark("diffusion_peak")
        x = adopt_rank0_coordinates(x)
        comm.log(f"[model] diffusion done in {time.time() - t0:.1f}s (samples {chunk.start}:{chunk.stop} of {no_rollout_samples})")
        if comm.rank == 0:                                  # STRUCTURE FIRST: the structures of these samples are on disk before their confidence heads start
            try:
                from openfold3.core.runners.writer import OF3OutputWriter
                from . import env as ENV
                C.fn("structure_first", "write_early")(batch, x, int(chunk.start), OF3OutputWriter.write_structure_prediction, log=comm.log,
                                                       enabled=ENV.structure_first_enabled(), switch="OF3TP_STRUCTURE_FIRST")
            except Exception as e:  # noqa: BLE001 - an early-write failure is named and never costs the prediction (upstream's writer still runs after the heads)
                comm.log(f"[structure_first] early write FAILED ({type(e).__name__}: {str(e)[:200]}); the heads run and upstream's writer writes as today")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                        # the rollout's z_cond rows / bias cache are dead: return the blocks before the confidence pair rep
        return x

    def heads(x, chunk):                                    # the auxiliary heads + the finished confidence dict on the k samples of x
        t0 = time.time()
        i = plan.chunks.index(chunk)
        comm.log(f"[conf] {C.fn('structure_first', 'utc')()} confidence heads start (samples {chunk.start}:{chunk.stop})")   # ordering evidence: stamped, after the structure-first write
        output = {"si_trunk": si_trunk, "zij_trunk": zij_trunk, "atom_positions_predicted": x}
        with torch.amp.autocast(device_type=autocast_device_type(si_trunk), dtype=cast_dtype):   # model.py:420-441
            aux = CONF.aux_heads_rows(self.aux_heads, batch=batch, si_input=si_input, output=output, lay=lay, config=self.config,
                                     use_zij_trunk_embedding=use_trunk_embedding, chunk_size=mode_mem_settings.chunk_size, inplace_safe=inplace_safe,
                                     offload_inference=offload_confidence_heads, _mask_trans=True, ztrunk=(zplan, i, None, False), contact_loc=contact_loc,
                                     pairformer_dtype=pairformer_dtype, **kernel_kw)
        C.mark("confidence")
        comm.log(f"[model] confidence done in {time.time() - t0:.1f}s (samples {chunk.start}:{chunk.stop} of {no_rollout_samples}; z_trunk {zplan.words[i]})")
        return aux

    try:
        if one_pass:                                        # k == S: the one batched pass, statements as above with no host round trip
            x = diffuse(last, None)
            output = {"si_trunk": si_trunk, "zij_trunk": zij_trunk, "atom_positions_predicted": x}
            output.update(heads(x, last))
            output["_sample_loop"] = SLB.one_pass_fields(plan)
            SLB.emit(output["_sample_loop"], comm.log, comm.rank, C.fn("evidence", "record_schedule"))
        else:
            output = {"si_trunk": si_trunk}
            output.update(SLB.run_rollout(plan, diffuse, heads, device=si_input.device, log=comm.log, rank=comm.rank,
                                          record_schedule=C.fn("evidence", "record_schedule")))
    except BaseException as e:                              # a failed roll-out releases the host copy on the way out: named, so the log's drop line reads as the error path
        comm.log(f"[model] roll-out aborted ({type(e).__name__}): closing the z_trunk plan on the error path (host copy released, device storage stays released)")
        raise
    finally:
        early = not zplan.closed and plan.passes > 0        # every pass ran: the LAST pass's plan.end closed the plan (host copy dropped there, after its embedding);
        zplan.close()                                       # idempotent; on an error path: the host copy released here
    if early:
        raise CONF.ConfidenceRefused("refused: the z_trunk plan was still open after the last confidence pass (plan.end(last) did not run)")
    output["zij_trunk"] = zij_trunk_after if zplan.consumed else zij_trunk
    self._tp_sample_loop = output.pop("_sample_loop", None)   # the lever's census fields (census_fields reads them); never a model output
    C.mark("done")
    return output


def feature_digest(batch) -> Dict[str, object]:
    """The cross-rank input digest, taken AFTER the batch sync — the core's rank input contract (``opt_core.mem.rowpair.rankdata``):
    ``feats_digest_shared`` = ``rankdata.digest_features`` (sha256, keys sorted: every tensor by dtype, shape and bytes; the query's atom array by
    its canonical annotation-order-independent summary; str / number leaves as canonical JSON; any other object excluded by name and printed,
    ``nontensor_unhashed=``) over the keys every rank holds alike — this rank's host copies of the parked MSA / bond features (``msa.host_keys()``: rank 0 holds the full MSA on host,
    ranks > 0 zero-row placeholders) are excluded by name — compared across ranks by ``rankdata.ranks_agree`` (one broadcast + one all-reduce: the
    same answer on every rank): ranks whose shared digests differ are REFUSED by name on every rank (``rankdata.differ_refusal``:
    ``feats_ranks_differ``). Line (``FEATS_TAG`` / ``FEATS_IDENTICAL``, the one spelling the launcher reads back): ``[feats] rank r shared <sha16>
    keys n … ranks_identical=True|False|n/a``; census ``feats_digest_shared feats_keys feats_host_keys feats_ranks_identical``; the launcher's
    census repeats the verdict per rank and over the ranks (``tp.census`` ``ranks_identical``)."""
    from . import msa as MSA
    comm = C.comm()
    host_keys = [k for k in MSA.host_keys() if k in batch]
    t0 = time.time()
    fd = C.fn("rankdata", "digest_features")(batch, exclude=host_keys)   # tensors by dtype/shape/bytes; the atom array by its canonical summary; other objects excluded BY NAME (fd.unhashed)
    shared = fd.digest
    identical = C.fn("rankdata", "ranks_agree")(shared, comm)               # True / False on every rank alike; None without a multi-rank group
    rec = {"feats_digest_shared": shared[:16], "feats_keys": len(batch), "feats_host_keys": ",".join(host_keys) or "none",
           "feats_tensor_leaves": fd.tensor_leaves, "feats_nontensor_leaves": ",".join(fd.nontensor_leaves) or "none", "feats_nontensor_unhashed": ",".join(fd.unhashed) or "none",
           "feats_ranks_identical": identical, "feats_digest_s": round(time.time() - t0, 2)}
    C.fn("evidence", "record_schedule")(**rec)
    comm.log(f"{FEATS_TAG}{comm.rank} shared {shared[:16]} keys {len(batch)} {fd.words()} (host-parked keys out of the shared digest: {rec['feats_host_keys']}; "
             f"{FEATS_IDENTICAL}={identical if identical is not None else 'n/a'}; {rec['feats_digest_s']}s)")
    if identical is False:
        raise C.fn("rankdata", "differ_refusal")("feats", comm.rank, shared)
    return rec


def forward_synced(self, batch, *a, **kw):
    from . import world_of
    ensure_group()
    if world_of(os.environ) > 1 and data_form() == "rank0_bcast":             # every rank already holds rank 0's batch (transfer_rank0_bcast): no second broadcast; the digest gate below is the check
        C.comm().log("[feats] batch sync: none (data_form=rank0_bcast: this rank holds rank 0's batch from the transfer hook); the cross-rank digest gate follows")
    else:
        C.fn("bcast", "sync_tensordict_from_rank0")(batch, skip_keys=(), strict=False, tag="batch")   # every rank's batch tensors overwritten with rank 0's (one featurization for the group)
    feature_digest(batch)
    from openfold3.projects.of3_all_atom.model import OpenFold3
    return PATCHES.original(OpenFold3, "forward")(self, batch, *a, **kw)


def compute_confidence_scores_patched(self, batch, outputs):
    if "_tp_confidence" in outputs:
        return outputs.pop("_tp_confidence")
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    return PATCHES.original(OpenFold3AllAtom, "_compute_confidence_scores")(self, batch, outputs)


# ----------------------------------------------------------------------------------------------- the data form: who featurises (rank0_bcast | per_rank)
DATA_FORM_ENV = "OF3TP_DATA_FORM"
"""The tp line's data form at P > 1: ``rank0_bcast`` (default) — rank 0 alone runs upstream's input pipeline and its batch reaches every rank
through the core (``opt_core.mem.rowpair.rankdata.broadcast_features``); ``per_rank`` — every rank featurises the query itself (the form the digest
gate compares). Any other value is refused by name. P == 1 has no data form (nothing is read)."""
DATA_FORM_DEFAULT = "rank0_bcast"
_FEATS_CALLS: Dict[int, int] = {}                             # transfer-hook calls per rank of this process (the rendezvous key of a batch: the same sequence on every rank)


def data_form(environ=None) -> str:
    """``rank0_bcast`` | ``per_rank`` from ``OF3TP_DATA_FORM`` (default ``rank0_bcast``); anything else is a RowpairRefused naming the variable."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(DATA_FORM_ENV) or "").strip() or DATA_FORM_DEFAULT
    try:
        return C.fn("rankdata", "check_data_form")(raw)
    except Exception as e:                                    # the core's sentence, with the kit's variable named
        raise C._refused(f"{getattr(e, 'reason', e)} ({DATA_FORM_ENV}={raw!r})") from None


def _record_word(batch) -> str:
    """``q=<query_id> s=<seed>`` of a (collated) predict record — the fields every rank's DataLoader yields alike (upstream's datapoint cache)."""
    q = batch.get("query_id")
    q = q[0] if isinstance(q, (list, tuple)) and q else q
    sd = batch.get("seed")
    try:
        sd = int(sd.reshape(-1)[0]) if torch.is_tensor(sd) else (int(sd[0]) if isinstance(sd, (list, tuple)) else sd)
    except Exception:                                          # noqa: BLE001 — an unreadable seed is named as such, on every rank alike
        sd = "?"
    return f"q={q} s={sd}"


def transfer_rank0_bcast(self, batch, device, dataloader_idx):
    """The transfer hook under the ``rank0_bcast`` data form. Rank 0 holds upstream's featurised batch, ranks > 0 hold the light record
    (``data.RANK_STUB_FLAG``). Rank 0: the sharding-floor verdict on its featurised token count, the host keys held back, upstream's transfer, the
    MSA placement — as under ``per_rank`` — then ``rankdata.broadcast_features`` (status word through the group's store first: ``ok n=<N> q=<query> s=<seed>`` |
    ``below_floor:<N>`` | ``skip:not_featurised``; then meta and tensors; the parked MSA keys travel as meta only, ``token_bonds`` lands on the host).
    Ranks > 0: wait at the rendezvous (no collective pending while rank 0 featurises), receive rank 0's batch, build the MSA zero-row placeholders
    from the meta and park ``token_bonds`` exactly as ``msa.place_batch_features`` leaves them under ``per_rank``. ``below_floor``: every rank prints
    the floor line and exits ``floor.EXIT_BELOW_FLOOR`` (the launcher serves the query unsharded); ``skip``: rank 0 did not featurise the query
    (upstream logged why) — every rank hands upstream an invalid record, which upstream's predict step skips on every rank alike."""
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from . import data as DATA, floor as FLOOR, msa as MSA, world_of
    ensure_group()
    comm = C.comm()
    RD, MH = C.seam("rankdata"), C.seam("msa_host")
    _FEATS_CALLS[comm.rank] = _FEATS_CALLS.get(comm.rank, 0) + 1
    key = f"feats/{_FEATS_CALLS[comm.rank]}"
    world, align = world_of(os.environ), C.fn("dist", "chunk_align")()
    skip_keys = MSA.HOST_KEYS_MSA if MSA.mode() == "rank0" else ()
    host_keys = MSA.HOST_KEYS_BONDS if MSA.bonds_on_host() else ()
    stub = DATA.RANK_STUB_FLAG in batch
    mine = _record_word(batch)                                 # q=<query_id> s=<seed> of the record THIS rank's DataLoader yielded (the lockstep check)
    transfer = PATCHES.original(OpenFold3AllAtom, "transfer_batch_to_device")
    if comm.rank == 0:
        try:                                                   # anything that fails on rank 0 before the broadcast is SIGNALLED (failed:<Type>:<msg>) so no receiver waits on it, then re-raised here
            if stub:
                raise C._refused("refused: rank0_bcast: rank 0 holds the light record — rank 0 must featurise (data.install_rank_stub ran on rank 0)")
            if "token_mask" in batch:
                v = FLOOR.verdict(int(batch["token_mask"].shape[-1]), world, align)
                status = RD.status_word(extra=f"n={v['N']} {mine}") if v["ok"] else f"below_floor:{v['N']}"
            else:                                              # upstream's failure record (valid_sample False): nothing was featurised
                status = "skip:not_featurised"
            if not status.startswith("ok"):                    # a control word travels BEFORE any transfer: receivers act on it at once, nothing else is sent
                fb = RD.broadcast_features(batch, comm=comm, key=key, status=status, what="feats")
                out = transfer(self, batch, device, dataloader_idx) if status.startswith("skip") else None
            else:
                held = {k: batch.pop(k) for k in MSA.host_keys() if k in batch}
                out = transfer(self, batch, device, dataloader_idx)
                out.update(held)
                if held:
                    MSA.place_batch_features(out)
        except BaseException as e:
            try:
                RD.broadcast_features(None, comm=comm, key=key, status=RD.status_word(e), what="feats")
            except Exception:                                  # noqa: BLE001 — the core raises the refusal on the source too; this rank re-raises its own error below
                pass
            raise
        if status.startswith("ok"):
            fb = RD.broadcast_features(out, comm=comm, key=key, status=status, skip_keys=skip_keys, host_keys=host_keys, device=device, pin_host=True)
    else:
        if not stub:
            raise C._refused("refused: rank0_bcast: this rank featurised the query itself — the light-record dataset was not installed (data.install_rank_stub)")
        fb = RD.broadcast_features(None, comm=comm, key=key, skip_keys=skip_keys, host_keys=host_keys, device=device, pin_host=True)
        if fb.feats is not None:
            out = fb.feats
            MH.place_skipped(out, fb.skipped, row_dims=MSA.ROW_DIMS_MSA)   # the zero-row MSA placeholders ranks > 0 hold under per_rank too
            MSA.place_batch_features(out)                    # token_bonds parked on this rank's pinned host, as under per_rank
        else:
            out = None
    comm.log(f"[feats] {RD.data_form_word('rank0_bcast')} rank {comm.rank} {fb.words()} key={key}")
    C.fn("evidence", "record_schedule")(data_form="rank0_bcast")
    tag = fb.status.split(":", 1)[0].split()[0]
    if tag == "ok":
        sent = " ".join(w for w in fb.status.split() if w.startswith(("q=", "s=")))
        if sent != mine:                                       # every rank's DataLoader yields the same (query, seed) sequence; a rank out of step is refused by name, never fed another query's batch
            raise C._refused(f"refused: feats_lockstep: rank 0 sent {sent!r}, rank {comm.rank} expected {mine!r} ({key})")
        return out
    if tag == "below_floor":                                   # every rank: the floor line, then exit 70 — the launcher serves the query unsharded
        n = int(fb.status.split(":")[1])
        vv = FLOOR.verdict(n, world, align)
        comm.log(f"below the sharding floor: the featurised query's {FLOOR.words(vv)}; every rank exits {FLOOR.EXIT_BELOW_FLOOR} — the launcher serves the query unsharded ({FLOOR.REASON})")
        C.fn("dist", "destroy")()
        raise FLOOR.BelowFloor(vv)
    if tag == "skip":                                          # rank 0 did not featurise: upstream's invalid record on every rank (its predict step skips it everywhere)
        comm.log("[feats] rank 0 did not featurise this query (upstream logged the reason on rank 0); this rank skips it too")
        rec = batch if comm.rank != 0 else out
        rec.pop(DATA.RANK_STUB_FLAG, None)
        rec["valid_sample"] = torch.zeros_like(rec["valid_sample"]) if torch.is_tensor(rec.get("valid_sample")) else torch.tensor([False])
        return rec if comm.rank == 0 else transfer(self, rec, device, dataloader_idx)
    raise C._refused(f"refused: feats status word {fb.status!r} is none of ok|below_floor|skip ({key})")


def _is_upstream_writer(cb) -> bool:
    """upstream's OF3OutputWriter callback (the structure-first write goes through its static write_structure_prediction and its output_dir / structure_format)."""
    try:
        from openfold3.core.runners.writer import OF3OutputWriter
    except ImportError:
        return False
    return isinstance(cb, OF3OutputWriter)


def transfer_batch_to_device_patched(self, batch, device, dataloader_idx):
    """``OpenFold3AllAtom.transfer_batch_to_device`` (Lightning's hook) with the host-resident feature keys kept on the host (``msa.host_keys``);
    the template keys move as they are (``[T, 1, 1, F]`` placeholders, token-level features, the real slots' per-token precursors)."""
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from . import floor as FLOOR, msa as MSA, rank_of, world_of
    if world_of(os.environ) > 1 and data_form() == "rank0_bcast":
        out = transfer_rank0_bcast(self, batch, device, dataloader_idx)
        C.fn("structure_first", "capture")(self, out, _is_upstream_writer)   # structure first: upstream's writer instance + the batch the writer will see (query_id / seed / atom_array)
        return out
    FLOOR.check_featurised(batch, world_of(os.environ), C.fn("dist", "chunk_align")(),
                           log=lambda s: (sys.stderr.write(f"[rowpair r{rank_of(os.environ)}] {s}\n"), sys.stderr.flush()))   # the sharding floor on the featurised token count, BEFORE this rank joins the
    ensure_group()                                          # group or issues a collective (identical on every rank: one query, one count); below it the rank exits 70 and the launcher
                                                            # serves the query unsharded. The row layout of the cut below is this rank's in the P-wide group (dist.default_ctx)
    keys = MSA.host_keys()
    held = {k: batch.pop(k) for k in keys if k in batch}
    out = PATCHES.original(OpenFold3AllAtom, "transfer_batch_to_device")(self, batch, device, dataloader_idx)
    out.update(held)
    if held:
        MSA.place_batch_features(out)
    C.fn("structure_first", "capture")(self, out, _is_upstream_writer)   # structure first (as above)
    return out


def timer_on_predict_batch_end_patched(self, *a, **kw):
    """``PredictTimer.on_predict_batch_end`` (the per-query timing.json run log) on rank 0; a no-op on ranks > 0, which write no prediction tree."""
    if C.comm().rank != 0:
        return None
    from openfold3.core.utils.callbacks import PredictTimer
    return PATCHES.original(PredictTimer, "on_predict_batch_end")(self, *a, **kw)


def writer_on_predict_end_patched(self, trainer, pl_module):
    """``OF3OutputWriter.on_predict_end`` (the run summary) after this rank LEAVES the process group: every prediction is written by then, and
    OpenFold3's summary all-gathers over ``trainer.world_size`` (= 1: each rank is a single-device trainer) whenever a group is initialised."""
    C.comm().log("[writer] predict end: leaving the process group; OpenFold3's run summary is this rank's own")
    C.fn("dist", "destroy")()
    from openfold3.core.runners.writer import OF3OutputWriter
    return PATCHES.original(OF3OutputWriter, "on_predict_end")(self, trainer, pl_module)


ENV_WRITER_NPZ = "OF3TP_WRITER_NPZ"                            # stored (default) | deflated — env.KIT_SWITCHES
_WRITER_NPZ_NAMED = [False]


@contextlib.contextmanager
def writer_npz_container(writer, comm):
    """Around upstream's ``OF3OutputWriter.on_predict_batch_end`` on rank 0: with ``OF3TP_WRITER_NPZ=stored`` (the tp line's default) numpy's
    ``savez_compressed`` IS ``savez`` for the duration of the call — the writer's ``np.savez_compressed(out_file_full, **full_confidence_scores)``
    (writer.py) then writes the same ``.npz`` container with the same keys and byte-identical arrays, STORED instead of DEFLATED (``np.load`` reads
    both; no zlib pass over the [N, N] matrices, which dominates the writer at large N when deflated). ``deflated`` = upstream's statement untouched.
    Census ``writer_npz``."""
    word = os.environ.get(ENV_WRITER_NPZ, "stored").strip().lower() or "stored"
    if word not in ("stored", "deflated"):
        raise C._refused(f"refused: {ENV_WRITER_NPZ}={word!r}: stored | deflated")
    fmt = str(getattr(writer, "full_confidence_format", "?"))
    full_on = bool(getattr(writer, "write_full_confidence_scores", False))
    if not _WRITER_NPZ_NAMED[0]:
        _WRITER_NPZ_NAMED[0] = True
        try:
            C.fn("evidence", "record_schedule")(writer_npz=f"{word}:{fmt}:{'on' if full_on else 'off'}")
        except Exception:                                    # noqa: BLE001 - census only
            pass
        comm.log(f"[writer] full confidence {'on' if full_on else 'off'} format={fmt} npz container={word} (OF3TP_WRITER_NPZ)")
    if word != "stored" or fmt != "npz" or not full_on:
        yield
        return
    import numpy as _np
    saved = _np.savez_compressed
    _np.savez_compressed = _np.savez
    try:
        yield
    finally:
        _np.savez_compressed = saved


def writer_on_predict_batch_end_patched(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    from openfold3.core.runners.writer import OF3OutputWriter
    comm = C.comm()
    if comm.rank != 0:
        if not getattr(self, "_tp_named", False):
            comm.log("[writer] rank > 0: OF3OutputWriter is a no-op here (rank 0 writes the prediction)")
            self._tp_named = True
        return None
    with writer_npz_container(self, comm):                 # the full-confidence npz STORED (np.savez) instead of deflated on the tp line (OF3TP_WRITER_NPZ; arrays identical)
        ret = PATCHES.original(OF3OutputWriter, "on_predict_batch_end")(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    try:                                                    # structure first: upstream's writer just rewrote the structure files with the pLDDT column -> sidecars say so
        C.fn("structure_first", "mark_final")(batch, outputs=outputs, log=comm.log)   # outputs is None when predict_step caught an exception: nothing rewritten, sidecars keep 'confidence not written'
    except Exception as e:  # noqa: BLE001
        comm.log(f"[structure_first] sidecar update failed ({type(e).__name__}: {str(e)[:160]})")
    return ret


# ----------------------------------------------------------------------------------------------------------------- install / census
def install() -> List[str]:
    """Rebind the seams above (idempotent per process). Resolves every core seam first so a short core refuses by name before any patch lands."""
    if PATCHES.names():
        return PATCHES.names()
    for name in ("dist", "shard", "ring", "trunk", "template", "msa", "msa_host", "pairstack", "trimul", "triatt", "transition", "heads", "confidence", "frames", "diffusion", "bcast", "rankdata"):
        C.seam(name)
    from openfold3.projects.of3_all_atom import model as _M
    from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
    from . import data as DATA, msa as MSA
    PATCHES.replace(_M.OpenFold3, "run_trunk", run_trunk_patched)
    PATCHES.replace(_M.OpenFold3, "_rollout", rollout_rows)
    PATCHES.replace(_M.OpenFold3, "forward", forward_synced)
    PATCHES.replace(MSAModuleEmbedder, "forward", MSA.msa_module_embedder_forward_host)
    install_ln_guard()
    try:                                                   # runner / writer pull the data-stack deps (biotite, ...); model-only test envs may lack them
        from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
        from openfold3.core.runners.writer import OF3OutputWriter
        PATCHES.replace(OpenFold3AllAtom, "_compute_confidence_scores", compute_confidence_scores_patched)
        PATCHES.replace(OpenFold3AllAtom, "transfer_batch_to_device", transfer_batch_to_device_patched)
        PATCHES.replace(OF3OutputWriter, "on_predict_batch_end", writer_on_predict_batch_end_patched)
        PATCHES.replace(OF3OutputWriter, "on_predict_end", writer_on_predict_end_patched)
        from openfold3.core.utils.callbacks import PredictTimer          # writes <output_dir>/<query>/seed_*/timing.json: rank 0 only (ranks > 0 hold no prediction tree)
        PATCHES.replace(PredictTimer, "on_predict_batch_end", timer_on_predict_batch_end_patched)
    except ImportError as e:
        C.comm().log(f"[model] runner/writer not importable ({type(e).__name__}: {e}) — model-level seams installed only")
    DATA.install()                                          # the featurizer emits [T, 1, 1, F] template placeholders (never [T, N, N, F]); recorded in data.PATCHES —
                                                            # already done by the import hook if this process imported the featurizer module first (the DataLoader workers get it there)
    from . import world_of, rank_of
    if world_of(os.environ) > 1:
        form = data_form()                                  # refused by name here, before any featurisation, when OF3TP_DATA_FORM is not a form
        C.fn("evidence", "record_schedule")(data_form=form)
        if form == "rank0_bcast" and multiprocessing.parent_process() is None:   # the rank MAIN process joins the group NOW (the store carries rank 0's featurisation status; ranks > 0 wait on it outside any collective); a DataLoader worker that imports the runner never joins
            ensure_group()
            if (rank_of(os.environ) or 0) > 0:
                DATA.install_rank_stub()                    # this rank's predict dataset hands the DataLoader the light record; rank 0 featurises
        C.comm().log(f"[model] data_form={form}" + (" (rank 0 featurises; this rank receives the batch — opt_core rankdata.broadcast_features)" if form == "rank0_bcast" and (rank_of(os.environ) or 0) > 0 else ""))
    C.comm().log(f"[model] installed: {', '.join(PATCHES.names())}")
    return PATCHES.names()


def uninstall() -> List[str]:
    from . import data as DATA
    names = PATCHES.names() + DATA.PATCHES.names()
    PATCHES.restore()
    DATA.uninstall()
    DATA.uninstall_rank_stub()
    return names


def census_fields(lay=None) -> Dict[str, object]:
    """The schedule census: layout, every derived block size with its source, the replicated-by-design and host-resident tensors by name, the
    pair-stack driver form words as their statements read them (``pairstack.pair_stream`` / ``reuse_z_storage``), the LayerNorm-guard record."""
    from . import pairstack as PS
    env_int = C.fn("dist", "env_int")
    out: Dict[str, object] = {
        "line": "tp", "sharding": "rowpair",
        "replicated_by_design": list(REPLICATED_BY_DESIGN), "host_resident": list(HOST_RESIDENT),
        "row_block_mb": env_int("ROWPAIR_ROWBLK_MB", 512), "triatt_qblock": env_int("ROWPAIR_TRIATT_QBLOCK", 0), "triatt_rowblock": env_int("ROWPAIR_TRIATT_ROWBLOCK", 0),
        "apb_qblock": env_int("ROWPAIR_APB_QBLOCK", 0), "apb_rowblock": env_int("ROWPAIR_APB_ROWBLOCK", 0), "trimul_sub": env_int("ROWPAIR_TRIMUL_SUB", 0),
        "opm_rows": env_int("ROWPAIR_OPM_ROWS", 0), "pair_stream": int(PS.pair_stream()), "reuse_z_storage": int(PS.reuse_z_storage()), "transpose_budget_gb": env_int("ROWPAIR_TRANSPOSE_BUDGET_GB", 24),
        "diff_cond_rows": env_int("ROWPAIR_DIFF_COND_ROWS", 0), "diff_q_rows": env_int("ROWPAIR_DIFF_Q_ROWS", 0), "conf_rows": env_int("ROWPAIR_CONF_ROWS", 0),
    }
    if lay is not None:
        out.update({"N": int(lay.N), "P": int(lay.P), "rank": int(lay.rank), "rows": f"{lay.r0}:{lay.r1}", "align": int(getattr(lay, "align", 0) or 0),
                    "parts": [[int(v) for v in x] if isinstance(x, (tuple, list)) else int(x) for x in (getattr(lay, "parts", None) or getattr(lay, "bounds", None) or [])]})
    out["sample_chunk_env"] = "default(1)"                                                   # one sample per pass (sample_loop.CHUNK)
    tc, tm = PS.triatt_census(), PS.trimul_census()                                          # the tp_triatt / tp_trimul levers: kernel words + the core dispatch's / provider's served and fallback counts of this rank
    out.update({"triatt_kernel": tc["kernel"] or PS.triatt_kernel(), "triatt_calls": int(tc["calls"]), "triatt_served": int(tc["served"]), "triatt_fallback": int(tc["fallback"]),
                "triatt_fallback_by": dict(tc["fallback_by"]), "triatt_impl": tc.get("impl"),
                "trimul_kernels": tm["kernels"] or PS.trimul_kernels(), "trimul_served": int(tm["served"]), "trimul_fallback": int(tm["fallback"]), "trimul_fallback_by": dict(tm["fallback_by"]),
                "trimul_cells": tm.get("cells"), "trimul_k1_impl": tm.get("k1_impl"), "trimul_impl": tm.get("impl")})
    out.update({"ln_guard": LN_GUARD["state"], "ln_guard_launches": int(LN_GUARD["launches"]), "ln_guard_split": int(LN_GUARD["split"]),
                "ln_guard_unsplittable": int(LN_GUARD.get("unsplittable", 0)), "ln_guard_split_shapes": [list(s) for s in LN_GUARD["shapes"]]})
    return out
