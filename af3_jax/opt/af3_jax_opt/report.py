"""The evidence lines and the activation report's text forms. One prefix for every line the package prints."""
from __future__ import annotations

import json
import sys

from opt_core import modes as _core_modes                                # levers_label: the levers= value of the ACTIVE line
from opt_core import report as _core                                    # the core's line grammar the kit composes from (prefix, kv, lever_line)

TAG = "af3-jax-opt"
PREFIX = _core.prefix(TAG)                                              # "[af3-jax-opt]"
ACTIVE, NOT_ACTIVE = "ACTIVE", "NOT ACTIVE"


def activation_line(rep: dict) -> str:
    """`[af3-jax-opt] ACTIVE mode=exact variant=p2 script=run_alphafold_fast.py launcher=levers_launch.py alphafold3=<tag@commit>
    key=... cache=... executables=N levers=FIX1+GLUT+ATTNCFG+L1 [partial=cold_cache] install=stock`
    (``levers=FIX1+FPF_TRIMUL+FPF_TRIATT+FPF_HOIST+TTR+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+L1+WRITER launcher=fpf_launch.py`` for fast: the mode's row, modes.KIT_MODES)
    or `[af3-jax-opt] NOT ACTIVE mode=... variant=... reason=...` (pred exits 3 on the latter)."""
    if rep.get("active"):
        parts = [PREFIX, ACTIVE, f"mode={rep['mode']}", f"variant={rep['variant']}", f"script={rep.get('script')}",
                 f"launcher={(rep.get('launcher') or ['none'])[0]}", f"alphafold3={rep.get('alphafold3_version')}",
                 f"key={rep.get('key')}", f"cache={rep.get('cache_dir')}", f"autotune={rep.get('autotune') or 'class_cache'}"]   # skipped(cache_dir empty) | fork_default | class_cache (the tree's cache class holds the fork's autotune result)
        if rep["mode"] != "off":
            parts.append(f"executables={rep.get('executables', 0)}")
            parts.append("levers=" + _core_modes.levers_label(rep.get("levers_applied") or []))
            if rep.get("levers_ablated"):                                # MODEL_OPT_LEVERS_OFF removed these from the composition for this run (ablation.py): named once, right after the set that runs
                from .ablation import TOKEN as _ABLATED
                parts.append(f"{_ABLATED}=" + ",".join(rep["levers_ablated"]))
            from .ablation import row_words as _row_words, ROWS_OFF_TOKEN as _ROWS_OFF
            if _row_words():                                               # MODEL_OPT_LEVERS_OFF's library row words (triattn_xla:<row>, ablation.validate_row_words): the rows switched off inside TRIATT_XLA, named once
                parts.append(f"{_ROWS_OFF}=" + ",".join(_row_words()))
            if rep.get("superseded"):
                parts.append("superseded=" + _core_modes.levers_label(rep["superseded"]))   # single-device memory levers whose sites the row-sharded stack owns under n_gpu > 1
            if rep.get("partial_conditions"):
                parts.append("partial=" + "+".join(rep["partial_conditions"]))   # cold_cache (stack.PARTIAL_CONDITIONS): named, the run proceeds
        parts.append(ngpu_fields(rep.get("n_gpu") or 1))                 # `n_gpu=P sharding=rowpair` (P = 1: `n_gpu=1 sharding=none`) — opt_core.mem.ngpu's words
        inst = rep.get("install") or {}
        if inst:
            parts.append(f"install={inst.get('install')}")
        reg = rep.get("region") or {}
        if reg:                                                            # the memory mode's region, decided up front (modes.big_region): whitespace-free tokens at the END of the line
            parts += [f"region={reg['region']}", f"n_est={'unknown' if reg.get('n_est') is None else reg['n_est']}", f"n_star={reg['n_star']}"]
        return " ".join(parts)
    return f"{PREFIX} {NOT_ACTIVE} mode={rep.get('mode')} variant={rep.get('variant')} reason={rep.get('reason')}"


def lever_lines(mode: str, per_lever: dict) -> list:
    """One `[af3-jax-opt] LEVER name=<shared strategy id | lever id> state=on|skipped [reason=...] impl=<mechanism> origin=kit lever=<id> mode=<m>
    site=<file> [served=<n> fallback=<n>] [evidence=...]` line per lever of the composition (modes.per_lever), in composition order — the
    per-lever activation census of this process's model run, in the core's pinned schema (opt_core.report.lever_line). origin is `kit` for
    every lever of this tree: the carried kit directories and the tree's own in-process module alike are this kit's files (site= says which)."""
    from . import registry
    out = []
    for lever, d in per_lever.items():
        reg = registry.LEVERS.get(lever) or {}
        fields = {"lever": lever, "mode": mode, "site": (reg.get("touches") or [reg.get("site")])[0]}
        fields.update(d.get("fields") or {})                            # a kernel lever's served= / fallback= counts
        if d.get("evidence"):
            fields["evidence"] = str(d["evidence"]).replace(" ", "_")
        reason = (d.get("reason") or "unknown").replace(" ", "_") if d["state"] != "on" else None
        out.append(_core.lever_line(TAG, d["name"], d["state"], reason=reason, impl=reg.get("impl"), origin="kit", **fields))
    return out

def xla_pool_note(rec: dict) -> str:
    """ONE line naming the XLA pool fraction the model process runs with at --n_gpu P and who set it (stack.xla_pool_fraction):
    `[af3-jax-opt] NOTE xla_mem_fraction=<f> source=kit|user|image name=<VAR> n_gpu=<P> (the row-sharded stack's precondition: pool fraction <= <c>; NCCL communicators allocate outside XLA's pool)`."""
    return (f"{PREFIX} NOTE xla_mem_fraction={rec.get('value')} source={rec.get('source')} name={rec.get('name')} n_gpu={rec.get('n_gpu')} "
            f"(the row-sharded stack's precondition: pool fraction <= {rec.get('ceiling')}; NCCL communicators allocate outside XLA's pool)")


PHASE_FIELDS = ("lm_s", "trunk_s", "sampler_s", "conf_s")                # the per-item phase fields of the PHASE line, in line order (total_s follows)
NO_LM = "-"                                                              # AlphaFold 3 has no protein-language-model encode: `lm_s=-` on every route
NOT_SEPARABLE = ("trunk", "sampler", "conf")                             # the phases with no Python boundary on ANY route of this model: `NA` on the line, one PHASE-NOTE each
ONE_PROGRAM = ("the fork's ModelRunner._model is ONE hk.transform of model.Model(config)(batch) under ONE jax.jit (stock run_alphafold.py:597-610; "
               "the kit script's _jitted_apply/_serialized_model :635-675, one executable per bucket; the row-sharded stack rebinds the same "
               "transform, inprocess/rowpair.py) - the trunk with its recycles, the diffusion sampler (every step x every sample) and the "
               "confidence head run inside one XLA program with no host round trip between them, so no perf_counter boundary exists short of "
               "re-cutting the program (a different executable than the one timed)")


PHASE_ITEM_UNNAMED = "unnamed"   # the item word when the transcript carried no `Running fold job <name>...` line before the timer


def peak_item_lines(rec: dict, jobs) -> list:
    """The cross-engine JAX peak grammar, per item: `[af3-jax-opt] PEAK item=<fold job> inuse_gib=<f>` (GiB = 2^30; the pass's
    jax peak_bytes_in_use from the peak record). JAX's counter is process-cumulative and the fork folds every input of a pass in ONE process,
    so a per-item value exists only when the pass had ONE input: then one PEAK item line; with several inputs ONE
    `PEAK-NOTE scope=pass items=<n> inuse_gib=<f> per_item=unavailable(jax_peak_bytes_in_use_is_process_cumulative)` and no per-item line
    (never a pass value dressed as an item's). Nothing when the record has no value (the pass PEAK line says why)."""
    if rec.get("status") != "ok" or rec.get("peak_alloc_gb") is None:
        return []
    from .peakmem import gib as _gib                                        # the one GB -> GiB conversion
    gib = _gib(rec["peak_alloc_gb"])
    names = [str(j).replace(" ", "_") for j in jobs if j] or [PHASE_ITEM_UNNAMED]
    if len(names) == 1:
        return [line("PEAK", item=names[0], inuse_gib=f"{gib:.2f}")]
    return [line("PEAK-NOTE", scope="pass", items=len(names), inuse_gib=f"{gib:.2f}", per_item="unavailable(jax_peak_bytes_in_use_is_process_cumulative)")]


def phase_line(seed: int, total_s: float, job=None) -> str:
    """`[af3-jax-opt] PHASE item=<fold job> seed=<N> lm_s=- trunk_s=NA sampler_s=NA conf_s=NA total_s=<s>` — ONE line per item on every
    route (off, exact, fast, big), an item being one ``ModelRunner.run_inference`` call (one seed of one fold input). The boundary is the
    fork's own per-seed timer, `Running model inference with seed N took S seconds.` (stock run_alphafold.py:744-750; the kit script
    :1021-1027 — the same statement in both files): total_s IS that timer's seconds (device_put of the features, the one forward program, the
    results copied to host), re-printed for the cross-check; the three model phases are `NA` (NOT_SEPARABLE, ONE_PROGRAM) and lm_s is `-`
    (NO_LM). item = the fold input's name as the transcript named it (`Running fold job <name>...` — the plan item's name; `unnamed` when no such
    line preceded the timer) and seed = the seed of this item's forward, its own token: the cross-engine PHASE grammar (item=<plan item name>,
    optional seed=<s>). A timing line only: nothing computed reads it."""
    return line("PHASE", item=str(job).replace(" ", "_") if job else PHASE_ITEM_UNNAMED, seed=str(int(seed)), lm_s=NO_LM, trunk_s="NA",
                sampler_s="NA", conf_s="NA", total_s=f"{float(total_s):.2f}")


def phase_notes() -> list:
    """One `[af3-jax-opt] PHASE-NOTE <phase> not separable: <reason>` line per NA phase (NOT_SEPARABLE) — printed once per pass, before the
    model process runs: separability is a property of the program, not of an item."""
    return [f"{PREFIX} PHASE-NOTE {p} not separable: {ONE_PROGRAM}" for p in NOT_SEPARABLE]


def line(tag: str, **kv) -> str:
    """One report line: `[af3-jax-opt] TAG k=v ...`; a None value drops its key."""
    return " ".join([PREFIX, tag] + [f"{k}={v}" for k, v in kv.items() if v is not None])


def emit(text: str, stream=None) -> None:
    (stream or sys.stderr).write(text + "\n")
    (stream or sys.stderr).flush()


def dump(rep: dict) -> str:
    return json.dumps(rep, indent=1, sort_keys=True, default=str)


def ngpu_fields(P: int) -> str:
    """`n_gpu=P sharding=<rowpair|none>` — the resource-axis tokens of the ACTIVE and DONE lines (opt_core.mem.ngpu.active_fields, the one producer)."""
    from opt_core.mem import ngpu
    return ngpu.active_fields(int(P))

