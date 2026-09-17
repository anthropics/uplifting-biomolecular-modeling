"""Ran-or-refuse: every lever a line activates proves it executed on the predictions of this process, or the run is PARTIAL by name.

A lever binds upstream symbols by name; an upstream refactor can leave a bind installed but never reached (a class-level method re-bind the
engine no longer calls), which would be a stock run under an ACTIVE banner. So after the first prediction every planned-and-applied lever is
asked for its OWN counter (the kits count already: ARM-T ``COUNTS``, the DITFAST / XL / OFFLOAD ``STATS``, ``fpf_engines.STATS``,
the tree's adapters' ``STATS``): a counter at zero is the named event ``lever_never_ran:<lever>`` in ``levers_fallback`` (PARTIAL, ``pred``
exit 3, like any fallback; ``stack.refresh``). Levers whose kit bytes expose no per-call counter are listed in ``activation.levers_uncounted``
with the reason they cannot silently no-op (``UNCOUNTED``: process levers, instance-level replacements and import-time patches that raise when
their target moved) — an explicit, named gap, never silence. Inertness is not silence: a lever the call's facts put outside its documented
domain (``registry.ENGAGEMENT``: the deterministic recipe, the items' token counts against the unit's own size gate, one
card for the row-sharded stacks) is ``lever_inert_by_design:<lever>(<reason>)`` in ``activation.levers_inert`` with the run COMPLETE; inside
its domain a zero counter is the defect. ``tests/test_ran.py`` locks: every registry lever is counted, uncounted with a reason, or off every
line on the pin; the predicates read the units' constants.
"""
from __future__ import annotations

import os

import sys


def _mod(name):
    return sys.modules.get(name)


def _dict_count(modname: str, attr: str, *keys: str):
    """Sum of integer counters ``module.attr[key]`` over ``keys`` (a list-valued key counts its length); None when the module is not loaded."""
    m = _mod(modname)
    if m is None:
        return None
    d = getattr(m, attr, None) or {}
    n = 0
    for k in keys:
        v = d.get(k, 0)
        n += len(v) if isinstance(v, (list, tuple, dict)) else int(v or 0)
    return n


def _fpf_calls():
    m = _mod("fpf_engines")
    if m is None:
        return None
    st = getattr(m, "STATS", {}) or {}
    return sum(int((v or {}).get("calls", 0) or 0) for v in st.values() if isinstance(v, dict))


def _truthy(v) -> int:
    """A kit STATS value as evidence: a flag / record (True, a non-empty dict or list, a string) counts 1, a number counts itself."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return 1 if v else 0


def _accel(section: str, *keys: str):
    m = _mod("odde_accel_v2")
    if m is None:
        return None
    d = (getattr(m, "STATS", {}) or {}).get(section) or {}
    return sum(_truthy(d.get(k)) for k in keys)


def _house_p():
    m = sys.modules.get("opendde_opt.prefetch")
    return None if m is None else int(m.PSTATS.get("engaged") or 0)


def _house_j():
    m = sys.modules.get("opendde_opt.writer_overlap")
    return None if m is None else int(m.JSTATS.get("files") or 0)


def _house(modname: str, *keys: str):
    return _dict_count("opendde_opt." + modname, "STATS", *keys)


# lever -> zero-argument callable returning the number of times the lever's bound statement was entered in this process (None = its module
# is not loaded here, which the install accounting already names). One entry per COUNTED lever.
def _house_sum(modname: str, *keys: str):
    """sum over house counters of one module (None when the module is not loaded in this process)."""
    vals = [_dict_count("opendde_opt." + modname, "STATS", k) for k in keys]
    if any(v is None for v in vals):
        return None
    return sum(int(v) for v in vals)


def _house_all(*pairs):
    """min over house counters ``(module, key)``: a conjunction — the lever ran only if EVERY named statement executed (any zero reads zero);
    None when a module is not loaded in this process."""
    vals = []
    for modname, key in pairs:
        v = _house(modname, key)
        if v is None:
            return None
        vals.append(int(v))
    return min(vals) if vals else None


COUNTERS = {
    "dit_hoist":          lambda: _dict_count("odde_addon", "STATS", "records", "hits", "bypass", "sampler_calls"),
    "dit_attn_bf16":      lambda: _accel("dit_attn", "bf16_calls", "fp32_calls"),
    "arm_z":              lambda: _dict_count("odde_arm_t", "COUNTS", "zt_block_reached"),          # the entry counter of the class-level block bind
    "arm_u":              lambda: _dict_count("odde_arm_t", "COUNTS", "bf16_stack_calls", "att_prov_calls", "att_stock_calls", "zt_block_reached"),
    "arm_u23":            lambda: _dict_count("odde_arm_t", "COUNTS", "bf16_stack_calls", "att_prov_calls", "att_stock_calls", "zt_block_reached"),
    "fpf_trimul_exact":   _fpf_calls,
    "xl_tri_ln":          lambda: _dict_count("odde_xl", "_STATS", "tri_ln_calls", "tri_ln_chunked_calls"),
    "drop_bond_mask":     lambda: _house("bondmask", "dropped", "missing"),
    "chunk_lift":         lambda: _house("chunklift", "calls"),                       # the wrapped resolution entered (two per item: residue + structural site)
    "lnstream":           lambda: _house("lnstream", "calls"),                        # fused-LayerNorm calls served by the current-stream build (FusedLayerNorm.forward -> the wrapped loader)
    "tmpl_dedup":         lambda: _house("tmpldedup", "calls", "bypass"),            # TemplateEmbedder.forward entered under the lever (de-duplicated, or stood aside by name)
    "keep_pool":          lambda: _house("keeppool", "skipped_total", "passed_total"), # empty_cache calls that reached the policy (kept in-forward + passed through)
    "structok_sync":      lambda: _house("structoksync", "calls"),                   # role-pair projection calls served with one host read
    "sched_host":         lambda: _house_sum("schedhost", "host_cmp", "rot_copies"),      # per-step schedule comparisons answered on the host + rotations moved without a stream wait
    "trimul_core":        lambda: _dict_count("odde_trimul_bind", "COUNTS", "calls"),       # calls served through a provider row (asides are the unit's named decisions, registry.ENGAGEMENT stepped_aside)
    "trimul_exact":       lambda: _dict_count("odde_trimul_bind", "COUNTS", "calls"),
    "ln_core":            lambda: _house("lncore", "calls"),                          # pair-row LayerNorm calls served through a provider row (stock-row cells / other forms are its named asides, registry.ENGAGEMENT stepped_aside)
    "transition_core":    lambda: _dict_count("odde_transition_bind", "COUNTS", "calls"),   # pair-transition calls decided under the word (a provider row or the kit singleton by name; module-kept calls are registry.ENGAGEMENT stepped_aside)
    "transition_exact":   lambda: _dict_count("odde_transition_bind", "COUNTS", "calls"),
    "prefetch":           lambda: _house_p(),                                          # dataloaders built with the worker
    "json_oneshot":       lambda: _house_j(),                                          # confidence documents encoded once + written once
    "writer_overlap":     lambda: _house("writer_overlap", "submitted"),               # items whose dump was handed to the background writer (0 = every item written by the stock path: never ran)
    "zprep_hoist":        lambda: _house_sum("zprephoist", "calls"),                    # denoiser pair preparations served (hits + records)
    "triattn_core":       lambda: _dict_count("odde_triattn_bind", "COUNTS", "calls"),      # calls served through a provider row (asides to the stock op are the unit's named decisions, registry.ENGAGEMENT stepped_aside)
    "triattn_exact":      lambda: _dict_count("odde_triattn_bind", "COUNTS", "calls"),
    "triattn_conf":       lambda: _dict_count("odde_arm_t", "COUNTS", "att_conf_prov_calls"),   # provider-served calls inside the confidence head's pair stack
    "sampler_amp":        lambda: _house("precision", "calls"),                       # the policy's calls behind the word (engaged / above_gate are its evidence words)
    "stepgraph":          lambda: _house("stepgraph", "replays"),                     # denoiser calls served by a graph replay (0 = every sampler call refused or ran eager: never ran)
    "struct_pair_bf16":   lambda: _house("tp", "struct_bf16_calls"),               # refiner pair stacks that ran on a bf16 structural shard
    "tp_triatt":          lambda: _dict_count("opendde_opt.tp_kernels", "TRIATT", "calls"),   # row batches handed to the core dispatch (served + fallback are the core ledger's)
    "rowpair_tp":         lambda: _house_all(("tp", "presharded_calls"), ("tp_diffusion", "zcond_rows"), ("tp_diffusion", "diff_denoise_calls")),   # ran = the sharded pair stacks AND the sharded roll-out (z_cond rows, denoiser calls) all executed
    "pair_offload_struct": lambda: _dict_count("odde_offload", "STATS", "calls_struct", "calls_diff_cache"),   # the unit's flat per-stage entry counters (levers/OFFLOAD,
    "pair_offload_trunk": lambda: _dict_count("odde_offload", "STATS", "calls_trunk"),              # odde_offload.STATS; its stage timers also wrap stock stages and are not evidence)
    "pair_offload_conf":  lambda: _dict_count("odde_offload", "STATS", "calls_conf", "calls_disto"),
    "diffz":              lambda: _dict_count("odde_offload", "STATS", "calls_diffz"),
    "free_templ":         lambda: _dict_count("odde_offload", "STATS", "calls_free_templ"),
    "bigln_guard":        lambda: _dict_count("odde_offload", "_BIGLN", "calls"),                   # the LayerNorm guard's entries (`hits` = the calls it re-routed)
    # levers/SAMPLER: the packages' own live counters, published as odde_accel_v2.STATS sections by the routing site (odde_sampler.SECTIONS)
    "dit_attn_apb":       lambda: _accel("apb_launch", "n_apb"),                    # token pair-bias kernel launches (the instance forwards' and the fused stack's alike)
    "atom_attn_apb":      lambda: _accel("apb_launch", "n_atom"),                   # local-window kernel launches
    "cond_dedupe":        lambda: _accel("cond_dedupe", "dedupe_calls", "stock_path_calls"),   # conditioning forwards entered (deduped + guarded-stock)
    "dit_fused":          lambda: _accel("dit_fused", "calls"),                     # fused token-stack forwards entered
    "dit_lowp":           lambda: _accel("dit_fused", "lowp_calls"),                # those forwards that ran 16-bit operands
    "atom_fused":         lambda: _accel("atom_fused", "calls"),                    # fused atom-stack forwards entered
    "dit_attn_exact":     lambda: _accel("dit_attn_exact", "calls"),                # the replaced SDPA statement entered (kernel + counted stock reasons)
}

# levers with no per-call counter in their bytes, and why each cannot silently no-op (a named gap, listed in activation.levers_uncounted)
UNCOUNTED = {
    "served_levers_hook": "proven by its install record, which the activation already requires (stack.levers_from_hook_records): no record, no applied lever",
    "cueq_tuned_cache": "not an opendde bind: a cuEquivariance tuning-cache location (a miss runs the default tiles, bitwise-equal)",
    "dit_align":    "the hoist's slot alignment, counted with dit_hoist (aligned_slots is legitimately zero when N never changes)",
    "alloc_auto":   "the allocator setting of the kit process, proven by the allocator read-back (alloc.fallbacks), not by calls",
    "sample_chunk": "a configs value set before the runner exists (configs.infer_setting.sample_diffusion_chunk_size), recorded in activation.big",
    "no_dit_hoist": "the ABSENCE of the DITFAST hoist at >= the size gate (nothing installed to count), recorded in activation.big",
}


def count(lever: str):
    """The lever's entry count in this process (int), None when uncounted or its module is not loaded."""
    f = COUNTERS.get(lever)
    if f is None:
        return None
    try:
        v = f()
    except Exception:  # noqa: BLE001 — a counter that cannot be read is not evidence of a run
        return None
    return None if v is None else int(v)


def census(levers) -> dict:
    """{lever: count | None} for the given levers (the manifest's activation.lever_counts)."""
    return {lv: count(lv) for lv in levers}


def _rank0_featurises() -> bool:
    """The row-sharded line's data form has rank 0 featurise for every rank (tp.DATA_FORM == "rank0_bcast"): the stated condition of the
    ``featurising_rank`` exemption, read from the adapter so another data form cannot inherit it."""
    from . import tp
    return tp.DATA_FORM == "rank0_bcast"


def engagement(lever: str, facts: dict | None) -> tuple[bool, str | None]:
    """(engaged, reason) for a counted lever on a call's facts (registry.ENGAGEMENT, the one place the predicates live). ``facts`` =
    {"det": bool, "n_gpu": int, "rank": this rank process's index | None, "token_floors": {item: residue tokens} | None, "levers": the line's planned
    levers | None, "layernorm": LAYERNORM_TYPE | None (absent = the process environment's)}; a fact the caller
    does not know (None / absent) cannot make a lever inert — unknown is engaged. The thresholds are the units' own constants read in this
    process. The reason names the fact and the threshold."""
    from . import registry
    e = registry.ENGAGEMENT.get(lever)
    facts = facts or {}
    if e is None:
        return True, None
    if e.multi_gpu and int(facts.get("n_gpu") or 1) <= 1:
        return False, "n_gpu=1: the adapter installs nothing on one card by design"
    if e.single_gpu and int(facts.get("n_gpu") or 1) > 1:
        return False, e.single_gpu
    if e.featurising_rank and int(facts.get("n_gpu") or 1) > 1 and int(facts.get("rank") or 0) > 0 and _rank0_featurises():
        return False, e.featurising_rank
    if e.torch_layernorm and (facts.get("layernorm") or os.environ.get("LAYERNORM_TYPE")) == "fast_layernorm":   # the process's stock-side environment (settings.STOCK_ENV, exported on every route)
        return False, e.torch_layernorm
    if e.fused_layernorm and (facts.get("layernorm") or os.environ.get("LAYERNORM_TYPE")) != "fast_layernorm":  # a caller's own LayerNorm backend: the extension loader is never called
        return False, e.fused_layernorm
    kn = facts.get("stock_knobs")
    if kn is None:                                                                                                  # absent = the process environment's fact (a rank process; the kit process's hook)
        from . import stockknob as _stockknob
        kn = _stockknob.in_force()
    if e.stock_knob and (kn or {}).get(e.stock_knob):                                                               # upstream's own kernel flag stated: the lever is aside by name, the flag's kernel serves
        return False, f"aside:stock_knob:{e.stock_knob}={kn[e.stock_knob]} (upstream's --{e.stock_knob} runs as stated; registry.STOCK_KNOB_LEVERS)"
    if e.stepped_aside:                                                                                             # the unit's own census: it reached its site and every call was the stock op's by a named rule
        why = e.stepped_aside[0]()
        if why:
            return False, f"{why} ({e.stepped_aside[1]})"
    floors = [int(v) for v in (facts.get("token_floors") or {}).values()]
    if floors and e.min_tokens:
        thr = e.min_tokens[0]()
        if thr is not None and max(floors) < int(thr):
            return False, f"every item is below the unit's size gate ({max(floors)} < {int(thr)} tokens; {e.min_tokens[1]})"
    return True, None


def inert_by_design(levers, facts: dict | None) -> dict:
    """{lever: "lever_inert_by_design:<lever>(<reason>)"} for the levers among ``levers`` the call's facts put outside their domain."""
    out = {}
    for lv in levers:
        ok, why = engagement(lv, facts)
        if not ok:
            out[lv] = f"lever_inert_by_design:{lv}({why})"
    return out


def never_ran(levers, facts: dict | None = None) -> list[str]:
    """The named events for levers among ``levers`` whose counter reads zero although the call's facts engage them: ``lever_never_ran:<lever>``."""
    inert = inert_by_design(levers, facts)
    return [f"lever_never_ran:{lv}" for lv in levers if lv in COUNTERS and count(lv) == 0 and lv not in inert]


def uncounted(levers) -> dict:
    """{lever: reason} for the given levers that carry no counter (UNCOUNTED)."""
    return {lv: UNCOUNTED[lv] for lv in levers if lv in UNCOUNTED}


BUCKETS = ("ran", "uncounted", "inert", "fallback", "declared_off")


def partition(rep: dict) -> dict:
    """{lever: [bucket, ...]} for every planned lever of an activation report after a prediction: ``ran`` (applied, its counter > 0),
    ``uncounted`` (applied, no per-call counter, the reason in levers_uncounted), ``inert`` (levers_inert), ``fallback`` (named in
    levers_fallback: the lever, ``<lever>:<reason>`` or ``lever_never_ran:<lever>``), ``declared_off`` (levers_declared_off). Total accounting:
    every planned lever sits in exactly ONE bucket — a lever in none or two is the report's defect (``stack.refresh`` closes the partition;
    tests/test_ran.py locks it for every line)."""
    counts = rep.get("lever_counts") or {}
    unc = rep.get("levers_uncounted") or {}
    fb = rep.get("levers_fallback") or []
    out = {}
    for lv in rep.get("levers_planned") or []:
        b = []
        if lv in (rep.get("levers_applied") or []):
            if lv in unc:
                b.append("uncounted")
            elif (counts.get(lv) or 0) > 0:
                b.append("ran")
        if lv in (rep.get("levers_inert") or []):
            b.append("inert")
        if any(f == lv or f.startswith(lv + ":") or f == f"lever_never_ran:{lv}" for f in fb):
            b.append("fallback")
        if lv in (rep.get("levers_declared_off") or []):
            b.append("declared_off")
        out[lv] = b
    return out
