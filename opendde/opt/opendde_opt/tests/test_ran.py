"""Ran-or-refuse (ran.py): every registry lever is counted, uncounted with a stated reason, or off every line on the pin; a lever whose
counter reads zero after a prediction is `lever_never_ran:<lever>` in levers_fallback (PARTIAL);
census the same way."""
import os
import sys
import types

import pytest

from opendde_opt import modes, ran, registry, stack
from opendde_opt.tests.test_activation import fresh  # noqa: F401 — the fresh-activation fixture (stub upstream first on sys.path, no kit module loaded)


def test_every_lever_is_counted_or_uncounted_with_a_reason_or_off_line():
    levers = set(registry.LEVERS)
    counted, uncounted, off = set(ran.COUNTERS), set(ran.UNCOUNTED), set(registry.OFF_LINE)
    assert not (counted & uncounted), counted & uncounted
    assert not ((counted | uncounted) - levers), (counted | uncounted) - levers
    assert levers - counted - uncounted <= off, levers - counted - uncounted - off        # only superseded / retired levers may lack both
    assert all(isinstance(v, str) and len(v) > 20 for v in ran.UNCOUNTED.values())
    for lv in ("arm_z", "arm_u", "rowpair_tp", "xl_tri_ln", "fpf_trimul_exact", "tp_triatt"):
        assert lv in ran.COUNTERS, lv                                                     # every class-level method re-bind carries a counter


def test_counts_read_the_kits_own_counters(monkeypatch):
    arm = types.ModuleType("odde_arm_t"); arm.COUNTS = {"zt_block_reached": 0, "att_prov_calls": 0, "att_stock_calls": 0, "bf16_stack_calls": 0}
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    assert ran.count("arm_z") == 0 and ran.never_ran(["arm_z", "served_levers_hook"]) == ["lever_never_ran:arm_z"]
    arm.COUNTS["zt_block_reached"] = 96
    assert ran.count("arm_z") == 96 and ran.never_ran(["arm_z"]) == []
    assert ran.count("cueq_tuned_cache") is None and ran.uncounted(["cueq_tuned_cache", "arm_z"]) == {"cueq_tuned_cache": ran.UNCOUNTED["cueq_tuned_cache"]}
    monkeypatch.delitem(sys.modules, "odde_arm_t")
    assert ran.count("arm_z") is None and ran.never_ran(["arm_z"]) == []                   # module not loaded: the install accounting names that, not this gate
    fe = types.ModuleType("fpf_engines"); fe.STATS = {"trimul_out": {"calls": 3, "kernel": 3}, "trimul_in": {"calls": 2, "kernel": 2}}
    monkeypatch.setitem(sys.modules, "fpf_engines", fe)
    assert ran.count("fpf_trimul_exact") == 5


def test_refresh_names_a_lever_that_never_ran_after_a_prediction(fresh, monkeypatch):
    """An applied lever whose counter is zero once a prediction ran is lever_never_ran:<lever> (PARTIAL); before any prediction it is not."""
    rep = stack.activate("exact", dry_run=False)
    if not rep.get("active"):
        pytest.skip(f"exact does not activate on this box: {rep.get('reason')}")
    arm = sys.modules.get("odde_arm_t") or types.ModuleType("odde_arm_t")
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    monkeypatch.setattr(arm, "COUNTS", {"zt_block_reached": 0, "installed": True, "arm": "Z"}, raising=False)
    monkeypatch.setitem(stack._REPORT, "levers_applied", sorted(set(stack._REPORT.get("levers_applied") or []) | {"arm_z"}))
    before = stack.refresh(predicted=False)
    assert "lever_never_ran:arm_z" not in (before.get("levers_fallback") or [])
    assert before["lever_counts"]["arm_z"] == 0 and "served_levers_hook" in before["levers_uncounted"]
    after = stack.refresh(predicted=True)
    assert "lever_never_ran:arm_z" in after["levers_fallback"] and after["partial"] and "arm_z" not in after["levers_applied"]


def _fake_units(monkeypatch, *, arm_calls=0, fpf_kernel=0, fpf_calls=4):
    """The kits' modules as a process holds them after one prediction: ARM-T (its own size gate CFG['MIN_TOKENS'] and COUNTS), the fpf engine
    (STATS per op: calls / kernel / fallbacks) and ACCEL's STATS."""
    arm = types.ModuleType("odde_arm_t"); arm.CFG = {"MIN_TOKENS": 300}
    arm.COUNTS = {"installed": True, "arm": "Z", "zt_block_reached": arm_calls, "zt_block_calls": arm_calls}
    fe = types.ModuleType("fpf_engines"); fe.__version__ = "0.1.2"; fe._ORIG = {("opendde", "trimul_out"): None, ("opendde", "trimul_in"): None}
    fe.STATS = {op: {"calls": fpf_calls, "kernel": fpf_kernel, "stock": fpf_calls - fpf_kernel,
                     "fallbacks": {} if fpf_kernel else {"N<=100_cueq_torch_fallback_regime": fpf_calls}} for op in ("trimul_out", "trimul_in")}
    acc = types.ModuleType("odde_accel_v2"); acc.STATS = {"dit_attn": {}}
    for name, m in (("odde_arm_t", arm), ("fpf_engines", fe), ("odde_accel_v2", acc)):
        monkeypatch.setitem(sys.modules, name, m)


def _activate_s1(monkeypatch, extra=("arm_z",)):
    rep = stack.activate("exact", dry_run=False)
    if not rep.get("active"):
        pytest.skip(f"exact does not activate on this box: {rep.get('reason')}")
    monkeypatch.setitem(stack._REPORT, "levers_applied", sorted(set(stack._REPORT.get("levers_applied") or []) | set(extra)))
    return rep


def test_engagement_predicates_read_the_units_own_constants(monkeypatch):
    _fake_units(monkeypatch)
    assert registry.ENGAGEMENT["arm_z"].min_tokens[0]() == 300 and registry.ENGAGEMENT["fpf_trimul_exact"].min_tokens[0]() == registry.FPF_TRIMUL_MIN_N + 1
    src = open(os.path.join(modes.kit_dir(stack.tree_root(), "fast_inference"), "levers", "ARMT", "odde_trimul_bind.py")).read()
    assert f"N_MIN = {registry.FPF_TRIMUL_MIN_N + 1} " in src and 'raise Aside("n_le_100_stock_small_n_path")' in src   # the one literal the registry restates, locked to the binding's bytes
    assert ran.engagement("arm_z", {"det": False, "token_floors": {"x": 9}})[0] is False
    assert ran.engagement("arm_z", {"det": False, "token_floors": {"x": 9, "y": 600}}) == (True, None)      # one item inside the domain engages the lever
    assert ran.engagement("arm_z", {"det": False, "token_floors": None}) == (True, None)                  # an unknown fact never makes a lever inert
    assert ran.engagement("rowpair_tp", {"n_gpu": 1})[0] is False and ran.engagement("rowpair_tp", {"n_gpu": 4}) == (True, None)
    assert set(registry.ENGAGEMENT) <= set(ran.COUNTERS)                                                  # predicates exist for counted levers only


def test_a_tiny_query_makes_the_size_gated_levers_inert_by_design(fresh, monkeypatch):
    """A 9-residue item (the warm-up item's regime): ARM-T below its own MIN_TOKENS and the fpf engine in its N<=100 stock regime are
    inert by name — the fpf op's counted stock calls are the design there, not a fallback."""
    _activate_s1(monkeypatch, extra=("arm_z", "fpf_trimul_exact"))
    _fake_units(monkeypatch, arm_calls=0, fpf_kernel=0)
    rep = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"nine": 9}})
    for lv in ("arm_z", "fpf_trimul_exact"):
        assert lv in rep["levers_inert"] and lv not in rep["levers_applied"], lv
        assert rep["levers_inert_reasons"][lv].startswith(f"lever_inert_by_design:{lv}(every item is below the unit's size gate (9 < "), rep["levers_inert_reasons"][lv]
        assert not any(f == lv or f.startswith(lv + ":") or f == f"lever_never_ran:{lv}" for f in rep["levers_fallback"]), rep["levers_fallback"]
    assert rep["fpf_fallbacks"]                                                                             # the engine's own count stays recorded


def test_a_production_size_query_engages_every_lever_and_a_zero_counter_is_never_ran(fresh, monkeypatch):
    """600 tokens, no recipe: every predicate is TRUE — ARM Z with a zero counter is the defect (PARTIAL), an fpf op that served no kernel
    call is a fallback by name."""
    _activate_s1(monkeypatch, extra=("arm_z", "fpf_trimul_exact"))
    _fake_units(monkeypatch, arm_calls=0, fpf_kernel=0)
    rep = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"7pzb": 600}})
    assert "lever_never_ran:arm_z" in rep["levers_fallback"] and rep["partial"] is True
    assert "fpf_trimul_exact" in rep["levers_fallback"] and "fpf_trimul_exact" not in (rep.get("levers_inert") or [])


def _offload_unit(monkeypatch):
    """The offload unit installed in this process (the fake `_offload` builds): its shim's `import odde_offload` finds it and installs nothing —
    the stub engine of this driver cannot host the real unit's binds (a box can; there ODDE_OFFLOAD_STRICT=1 ends a failed install)."""
    from .test_lever_mechanics import _offload
    off, offt = _offload(installed=True)
    off.install = lambda *a, **kw: True                                       # the shim's call: already installed (idempotent in the unit)
    monkeypatch.setitem(sys.modules, "odde_offload", off)
    monkeypatch.setitem(sys.modules, "odde_offload_trunk", offt)
    return off, offt


ARM_LETTER = {**{lv: lv[len("arm_"):].upper() for lv in registry.LEVERS if lv.startswith("arm_")}, **{v: k.upper() for k, v in stack.ARM_LEVERS.items()}}   # the hook's arm word per lever

def _every_unit(monkeypatch, planned, n):  # noqa: C901
    """Every unit a CLI-route line can plan, as the process holds them after one prediction in which each installed lever was entered ``n``
    times (n=0: installed, never entered). The served hook's install record is shaped like the kit's (stack.HOOK_RECORD_LEVERS keys)."""
    from opendde_opt import alloc, bondmask, chunklift, lnstream, lncore, tmpldedup, keeppool, stepgraph, precision
    rec = {"kit.cueq_cache": "/x"} if "cueq_tuned_cache" in planned else {}
    if "dit_attn_bf16" in planned:
        rec["accel_v2.dit_attn(TIER-2)"] = {"installed": True}
    if "dit_hoist" in planned:
        rec["ditfast"] = {"dit_hoist": "installed", "dit_align": True}
    arm = next((x for x in planned if x in ARM_LETTER), None)
    if arm:
        rec["odde_arm_t"] = {"arm": ARM_LETTER[arm], "version": "0.3"}
    hook = types.ModuleType("odde_served_levers"); hook.STATE = {"installs": [{"levers": rec}], "errors": {}}
    acc = types.ModuleType("odde_accel_v2")
    acc.STATS = {"dit_attn": {"bf16_calls": n}}
    acc.STATS.update({"apb_launch": {"n_apb": n, "n_apb_lp": n, "n_atom": n}, "cond_dedupe": {"dedupe_calls": n, "stock_path_calls": 0},   # the SAMPLER unit's sections (odde_sampler.SECTIONS)
                      "dit_fused": {"calls": n, "lowp_calls": n}, "atom_fused": {"calls": n}, "dit_attn_exact": {"calls": n}})
    addon = types.ModuleType("odde_addon"); addon.STATS = {"records": n, "hits": n, "bypass": 0, "sampler_calls": n}
    armm = types.ModuleType("odde_arm_t"); armm.CFG = {"MIN_TOKENS": 300}
    armm.COUNTS = {"installed": True, "arm": ARM_LETTER.get(arm), "zt_block_reached": n, "zt_block_calls": n, "bf16_stack_calls": n, "att_prov_calls": n, "att_stock_calls": 0,
                   "att_conf_calls": n, "att_conf_prov_calls": n}                                                       # the confidence head's stack reached, its calls provider-served (triattn_conf)
    fe = types.ModuleType("fpf_engines"); fe.__version__ = "0.1.2"; fe._ORIG = {("opendde", "trimul_out"): None}
    fe.STATS = {"trimul_out": {"calls": n, "kernel": n, "stock": 0, "fallbacks": {}}}
    xl = types.ModuleType("odde_xl"); xl._INSTALLED = tuple(x[len("xl_"):] for x in planned if x.startswith("xl_"))
    xl._STATS = {"tri_ln_calls": n, "tri_ln_chunked_calls": 0}
    off = sys.modules.get("odde_offload")
    if off is None or not isinstance(getattr(off, "STATS", None), dict):
        off, _offt = _offload_unit(monkeypatch)
    for k in ("calls_struct", "calls_diff_cache", "calls_trunk", "calls_conf", "calls_disto", "calls_diffz", "calls_free_templ"):
        off.STATS[k] = n
    off.STATS["diffz_images"] = n; off._BIGLN["calls"] = n
    trib = types.ModuleType("odde_triattn_bind"); trib.WORD = "exact" if "triattn_exact" in planned else "fast"      # the two provider bindings (levers/ARMT): their own census,
    trib.COUNTS = {"active": True, "word": trib.WORD, "calls": n, "stock_calls": 0, "asides": {}, "errors": {}}       # the arm's att_mode / trimul_bind words name them bound
    trmb = types.ModuleType("odde_trimul_bind"); trmb.WORD = "exact" if "trimul_exact" in planned else "fast"
    trmb.COUNTS = {"active": True, "word": trmb.WORD, "calls": n, "stock_calls": 0, "asides": {}, "errors": {}}
    armm.COUNTS.update(att_mode=f"core_{trib.WORD}", trimul_bind=trmb.WORD)
    smp = types.ModuleType("odde_sampler"); smp.REPORT = {"installed": [x for x in planned if x in modes.SAMPLER_LEVERS]}   # the SAMPLER unit's install report
    mods = {"odde_served_levers": hook, "odde_accel_v2": acc, "odde_addon": addon, "odde_arm_t": armm, "fpf_engines": fe, "odde_xl": xl, "odde_sampler": smp,
            "odde_triattn_bind": trib, "odde_trimul_bind": trmb}
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(bondmask, "STATS", {**bondmask.STATS, "installed": True, "armed": True, "dropped": n, "missing": 0})
    monkeypatch.setattr(bondmask, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(chunklift, "STATS", {**chunklift.STATS, "installed": True, "armed": False, "calls": 2 * n, "items": n, "admitted": 2 * n})   # two resolutions per item
    monkeypatch.setattr(chunklift, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(lnstream, "STATS", {**lnstream.STATS, "installed": True, "armed": False, "state": "serving", "calls": n, "jit_s": 1.0, "sites_patched": 25})   # the current-stream build bound, n LayerNorm calls served
    monkeypatch.setattr(lncore, "STATS", {**lncore.STATS, "installed": True, "armed": False, "word": "fast", "calls": n})   # the LayerNorm provider wrapper bound under the word, n pair-row calls served
    monkeypatch.setattr(lncore, "COUNTS", {**lncore.COUNTS, "active": True, "word": "fast", "calls": n, "stock_calls": n, "asides": ({"stock_cell:aten": n} if n else {})})
    monkeypatch.setattr(lnstream, "_sync", lambda: None, raising=False)
    from opendde_opt import schedhost
    monkeypatch.setattr(schedhost, "STATS", {**schedhost.STATS, "installed": True, "patches": 2, "sched_calls": 1, "host_cmp": n, "rot_copies": n})   # the scheduler + augmentation names bound, n steps served
    monkeypatch.setattr(schedhost, "_sync", lambda: None, raising=False)
    from opendde_opt import structoksync
    monkeypatch.setattr(structoksync, "STATS", {**structoksync.STATS, "installed": True, "patches": 3, "calls": n, "host_reads": n})   # the expander's methods bound, n calls served
    monkeypatch.setattr(structoksync, "_sync", lambda: None, raising=False)
    from opendde_opt import writer_overlap
    monkeypatch.setattr(writer_overlap, "STATS", {**writer_overlap.STATS, "installed": True, "items": n, "submitted": n, "written": n, "failures": [], "missing": [], "aside": {}, "sync": 0})   # the dump bound, n items handed to the writer
    monkeypatch.setattr(writer_overlap, "JSTATS", {**writer_overlap.JSTATS, "installed": True, "files": 10 * n, "missing": []})   # save_json bound, 2 documents x 5 samples per item
    from opendde_opt import prefetch as itemcensus
    monkeypatch.setattr(itemcensus, "PSTATS", {**itemcensus.PSTATS, "installed": True, "calls": 1 if n else 0, "engaged": 1 if n else 0, "aside": {}, "missing": []})   # one dataloader built with the worker
    from opendde_opt import zprephoist
    monkeypatch.setattr(zprephoist, "STATS", {**zprephoist.STATS, "installed": True, "patches": 1, "calls": n, "hits": n - 1, "records": 1})   # the diffusion module's permute name bound, n steps served
    monkeypatch.setattr(zprephoist, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(tmpldedup, "STATS", {**tmpldedup.STATS, "installed": True, "armed": False, "calls": 10 * n, "slots": 40 * n, "distinct": 10 * n, "saved": 30 * n, "decisions": n})   # ten recycles per item, 4 slots, one distinct
    monkeypatch.setattr(tmpldedup, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(keeppool, "STATS", {**keeppool.STATS, "installed": True, "skipped_total": 4 * n, "passed_total": n})   # four in-forward sites kept per item, the runner's release passed
    monkeypatch.setattr(keeppool, "fallbacks", lambda planned: [])                          # the wrapper identity check needs the real torch binding (test_keeppool covers it)
    monkeypatch.setattr(stepgraph, "STATS", {**stepgraph.STATS, "installed": True, "armed": False, "sampler_calls": n, "captures": n, "replays": 197 * n, "disabled": None, "refused": {}})   # one capture + 197 replays per sampler call
    monkeypatch.setattr(stepgraph, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(precision, "STATS", {**precision.STATS, "installed": True, "armed": False, "calls": n, "engaged": n})   # the policy site decided once per item, every item engaged
    monkeypatch.setattr(precision, "_sync", lambda: None, raising=False)
    monkeypatch.setattr(alloc, "STATE", {"decision": {"decision": "default", "reason": "test: no query item read", "policy": "expandable"}})
    if "ODDE_ADDON_LEVERS" not in os.environ and "dit_hoist" in planned:
        monkeypatch.setenv("ODDE_ADDON_LEVERS", "dit_hoist,dit_align")
    from opendde_opt import big as _big
    monkeypatch.setattr(_big, "refresh", lambda rep: rep)                    # the memory mode's per-prediction census needs a real prediction (test_lever_mechanics covers its fold)


CLI_LINES = sorted(modes.LINES)


@pytest.mark.parametrize("scenario", ["ran@600", "idle@9", "idle@600"])
@pytest.mark.parametrize("line", CLI_LINES)
def test_every_planned_lever_of_every_line_lands_in_exactly_one_bucket(fresh, monkeypatch, line, scenario):
    """Total accounting (ran.partition): after a prediction every planned lever of the line is in exactly ONE of {ran (counter > 0),
    uncounted (named reason), inert (by design / no call reached), fallback (named), declared_off} — none in zero buckets, none in two —
    whether every lever ran (600 tokens), none was entered on a tiny item (9 tokens: size-gated levers inert, the rest never-ran) or none
    was entered at production size (every counted lever never-ran)."""
    _offload_unit(monkeypatch)                                                               # the unit as a box holds it: installed before activation (its shim finds it)
    rep = stack.activate(line, dry_run=False)
    if not rep.get("active"):
        pytest.skip(f"{line} does not activate on this box: {rep.get('reason')}")
    planned = list(rep["levers_planned"])
    n = 5 if scenario.startswith("ran") else 0
    tokens = int(scenario.split("@")[1])
    _every_unit(monkeypatch, planned, n)
    out = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"item": tokens}})
    buckets = ran.partition(out)
    assert set(buckets) == set(planned)
    bad = {lv: b for lv, b in buckets.items() if len(b) != 1}
    assert bad == {}, (line, scenario, bad, {k: out.get(k) for k in ("levers_applied", "levers_inert", "levers_fallback", "levers_declared_off", "levers_uncounted", "lever_counts")})
    assert out["lever_buckets"] == buckets
    if n:                                                                                        # every lever entered: nothing never-ran, nothing inert by design except one-card rowpair
        assert not [f for f in out["levers_fallback"] if f.startswith("lever_never_ran:")], out["levers_fallback"]
        assert set(out.get("levers_inert_reasons") or {}) <= {"rowpair_tp", "struct_pair_bf16", "tp_triatt"}, out.get("levers_inert_reasons")   # one card: the row-sharded line's levers inert by design


def test_rank_0_alone_featurises_so_the_data_path_lever_is_inert_by_design_on_the_other_ranks():
    """Under --n_gpu P>1 rank 0 featurises every item (tp._rank0_item): drop_bond_mask runs on rank 0 and is inert by design — named, the run
    COMPLETE — on ranks > 0; at P=1 and on rank 0 a zero counter stays lever_never_ran."""
    assert ran.engagement("drop_bond_mask", {"n_gpu": 2, "rank": 1}) == (False, ran.engagement("drop_bond_mask", {"n_gpu": 4, "rank": 3})[1])
    why = ran.engagement("drop_bond_mask", {"n_gpu": 2, "rank": 1})[1]
    assert why.startswith("served_on=rank0: ") and "rank0_bcast" in why
    assert ran.engagement("drop_bond_mask", {"n_gpu": 2, "rank": 0}) == (True, None) and ran.engagement("drop_bond_mask", {"n_gpu": 1, "rank": 0}) == (True, None)
    assert ran.engagement("drop_bond_mask", {"n_gpu": 2}) == (True, None)                       # an unknown rank cannot make it inert
    import opendde_opt.bondmask as _bm                                                        # the counted module loaded, its counter at zero: engaged ranks say never_ran, rank>0 does not
    keep = dict(_bm.STATS)
    try:
        _bm.STATS.update(dropped=0, missing=0)
        assert ran.count("drop_bond_mask") == 0
        assert ran.never_ran(["drop_bond_mask"], {"n_gpu": 2, "rank": 1}) == []
        assert ran.never_ran(["drop_bond_mask"], {"n_gpu": 2, "rank": 0}) == ["lever_never_ran:drop_bond_mask"]
        assert ran.never_ran(["drop_bond_mask"], {"n_gpu": 1, "rank": 0}) == ["lever_never_ran:drop_bond_mask"]
    finally:
        _bm.STATS.clear(); _bm.STATS.update(keep)
    assert list(ran.inert_by_design(["drop_bond_mask", "rowpair_tp"], {"n_gpu": 2, "rank": 1})) == ["drop_bond_mask"]


def test_the_featurising_rank_exemption_is_tied_to_the_rank0_bcast_data_form(monkeypatch):
    """The exemption's stated condition is the line's data form: under another form every rank featurises and a zero counter on rank>0 is a defect."""
    from opendde_opt import tp
    monkeypatch.setattr(tp, "DATA_FORM", "per_rank")
    assert ran.engagement("drop_bond_mask", {"n_gpu": 2, "rank": 1}) == (True, None)


def test_a_sampler_lever_that_stepped_aside_by_name_is_inert_not_a_fallback_and_the_run_is_complete(fresh, monkeypatch):
    """levers/SAMPLER 0.2.47: `dit_attn_exact` on a stack the core provider has no prebuilt for (cc 8.0 today) installs nothing and names the
    aside (odde_sampler.REPORT['aside']); the fold files it under levers_inert with reason aside:no_prebuilt:<stack> — LEVER state=off
    reason=aside:… — never under levers_fallback, so the aside alone never makes the run PARTIAL (the stock fp32 SDPA statement is exact by construction)."""
    _offload_unit(monkeypatch)
    rep = stack.activate("exact", dry_run=False)
    if not rep.get("active"):
        pytest.skip(f"exact does not activate on this box: {rep.get('reason')}")
    planned = list(rep["levers_planned"]); assert "dit_attn_exact" in planned
    _every_unit(monkeypatch, planned, 5)
    smp = sys.modules["odde_sampler"]
    smp.REPORT = {"installed": [x for x in planned if x in modes.SAMPLER_LEVERS and x != "dit_attn_exact"], "aside": {"dit_attn_exact": "no_prebuilt:torch2.7.1-cu126-sm80"}}
    dx = types.ModuleType("opendde_fpf_dit_attn_exact"); dx.STATS = {"installed": False, "aside": "no_prebuilt:torch2.7.1-cu126-sm80", "calls": 0}
    monkeypatch.setitem(sys.modules, "opendde_fpf_dit_attn_exact", dx)
    sys.modules["odde_accel_v2"].STATS["dit_attn_exact"] = {"calls": 0}                          # nothing installed: the replaced statement was never entered through the unit
    out = stack.refresh(predicted=True, facts={"det": True, "n_gpu": 1, "token_floors": {"item": 600}})
    assert "dit_attn_exact" in out["levers_inert"] and out["levers_inert_reasons"]["dit_attn_exact"] == "aside:no_prebuilt:torch2.7.1-cu126-sm80"
    assert "dit_attn_exact" not in out["levers_applied"] and not [f for f in out["levers_fallback"] if "dit_attn_exact" in f], out["levers_fallback"]   # the aside contributes no fallback (other units of this synthetic process may)
    assert ran.partition(out)["dit_attn_exact"] == ["inert"]
    from opendde_opt import report
    row = [l for l in report.lever_lines(out) if "LEVER name=dit_attn_exact " in l]
    assert len(row) == 1 and "state=off reason=aside:no_prebuilt:torch2.7.1-cu126-sm80" in row[0], row



def _big_with_units(monkeypatch, n=5):
    """The big line active in this process with every unit it plans faked after one prediction (each installed lever entered ``n`` times)."""
    _offload_unit(monkeypatch)
    rep = stack.activate("big", dry_run=False)
    if not rep.get("active"):
        pytest.skip(f"big does not activate on this box: {rep.get('reason')}")
    planned = list(rep["levers_planned"])
    assert "triattn_conf" in planned and "pair_offload_conf" in planned, planned
    _every_unit(monkeypatch, planned, n)
    _offload_unit(monkeypatch)                                                                # the offload unit's census again (every_unit does not fake it)
    return planned


def test_above_the_offload_gate_the_confidence_sub_lever_is_inert_by_name_when_the_unit_serves_that_stack(fresh, monkeypatch):
    """0.2.59: on the big lines above the offload unit's size gate `off_confidence_head` runs the confidence head's pair stack over host-streamed
    ROW BLOCKS (off_pairformer_stack -> block.tri_att_*.mha per block) and never enters confidence_head.pairformer_stack.forward, the one site where
    triattn_conf marks its calls: att_conf_calls stays 0 by construction. The unit's own census (odde_offload.STATS['calls_conf'] > 0) makes the
    lever INERT BY NAME -- levers_inert, reason conf_path=offload_rows:<n> (replaced_by=pair_offload_conf), LEVER state=off reason=conf_path=... --
    never lever_never_ran, the run COMPLETE (rc 0)."""
    from opendde_opt import report
    _big_with_units(monkeypatch, 5)
    sys.modules["odde_offload"].STATS.update(calls_conf=8, calls_disto=8)                     # the unit served the confidence stage (8 passes)
    sys.modules["odde_arm_t"].COUNTS.update(att_conf_calls=0, att_conf_prov_calls=0)          # the marked module was never entered
    out = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"n1400": 1400}})
    assert "triattn_conf" in out["levers_inert"], (out.get("levers_inert"), out.get("levers_fallback"))
    why = out["levers_inert_reasons"]["triattn_conf"]
    assert why.startswith("lever_inert_by_design:triattn_conf(conf_path=offload_rows:8 (replaced_by=pair_offload_conf"), why
    assert not [f for f in out["levers_fallback"] if "triattn_conf" in f], out["levers_fallback"]
    assert ran.partition(out)["triattn_conf"] == ["inert"]
    assert "pair_offload_conf" in out["levers_applied"]                                        # the stage's own lever accounts for it
    row = [l for l in report.lever_lines(out) if "LEVER name=triattn_conf " in l]
    assert len(row) == 1 and "state=off reason=lever_inert_by_design:triattn_conf(conf_path=offload_rows:8" in row[0], row
    ev = report.lever_evidence("triattn_conf", report.kit_stats())                            # the LEVER line's census token: which path ran the confidence stack
    assert ev.get("conf_path") == "offload_rows:8" and ev.get("conf_calls") == 0, ev


def test_below_the_offload_gate_the_confidence_sub_lever_must_run_and_a_skipped_hook_still_fails_closed(fresh, monkeypatch):
    """The other side of the gate: the unit did NOT serve the confidence stage (below its size gate, or the conf stage off the line: calls_conf == 0),
    so the head runs upstream's module through the marked stack. Provider-served calls there = applied; the marked stack never entered although
    nothing named explains it = lever_never_ran:triattn_conf (PARTIAL) -- the fail-closed expectation stands."""
    _big_with_units(monkeypatch, 5)
    sys.modules["odde_offload"].STATS.update(calls_conf=0, calls_disto=0)
    out = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"n0800": 800}})
    assert "triattn_conf" in out["levers_applied"] and "triattn_conf" not in (out.get("levers_inert") or []), out.get("levers_inert_reasons")
    sys.modules["odde_arm_t"].COUNTS.update(att_conf_calls=0, att_conf_prov_calls=0)          # the hook skipped for another reason: no named path explains it
    out = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"n0800": 800}})
    assert "lever_never_ran:triattn_conf" in out["levers_fallback"] and out["partial"] is True, (out.get("levers_fallback"), out.get("levers_inert_reasons"))
    sys.modules["odde_arm_t"].COUNTS.update(att_conf_calls=280, att_conf_prov_calls=0)        # every confidence-stack call the stock op's by the cell's rule: inert by that name, as before
    out = stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"n0800": 800}})
    assert out["levers_inert_reasons"]["triattn_conf"].startswith("lever_inert_by_design:triattn_conf(conf_cells_stock:280"), out["levers_inert_reasons"].get("triattn_conf")


def test_in_a_rank_process_the_confidence_sub_lever_is_inert_whatever_the_offload_census(fresh, monkeypatch):
    """--n_gpu P>1 (line BIG_TP): the row-sharded stacks serve triangle attention through the core's row-block dispatch (replaced_by=tp_triatt) --
    the single_gpu rule holds before any census is read, on every rank, with or without the offload unit's conf stage."""
    _big_with_units(monkeypatch, 5)
    sys.modules["odde_arm_t"].COUNTS.update(att_conf_calls=0, att_conf_prov_calls=0)
    for calls_conf in (0, 8):
        sys.modules["odde_offload"].STATS.update(calls_conf=calls_conf)
        for rank in (0, 1):
            assert ran.never_ran(["triattn_conf"], {"det": False, "n_gpu": 2, "rank": rank, "token_floors": {"n1400": 1400}}) == [], (calls_conf, rank)
            ok, why = ran.engagement("triattn_conf", {"det": False, "n_gpu": 2, "rank": rank})
            assert ok is False and why.startswith("replaced_by=tp_triatt"), why
    ok, why = ran.engagement("triattn_conf", {"det": False, "n_gpu": 1, "token_floors": {"n1400": 1400}})   # one card, unit served the stage: the conf-path rule
    assert ok is False and why.startswith("conf_path=offload_rows:8"), why


def test_the_exit_tally_lever_rows_replay_the_prediction_buckets(fresh, monkeypatch):
    """The LEVER rows print at interpreter exit from a bare stack.refresh(); once a prediction ran in the process (refresh(predicted=True, facts)
    -- the PRED line) that bare call replays the same ran-or-refuse bucketing, so a lever the PRED line held inert by name reads
    `state=off reason=lever_inert_by_design:<lever>(<reason>)` on its LEVER row too (before: state=on, the activation-time plan).
    Before any prediction a bare refresh() evaluates nothing (unknown is engaged)."""
    from opendde_opt import report
    _big_with_units(monkeypatch, 5)
    sys.modules["odde_offload"].STATS.update(calls_conf=8, calls_disto=8)
    sys.modules["odde_arm_t"].COUNTS.update(att_conf_calls=0, att_conf_prov_calls=0)
    pre = stack.refresh()                                                                      # no prediction yet: the plan, nothing bucketed
    assert "triattn_conf" in pre["levers_applied"] and "triattn_conf" not in (pre.get("levers_inert") or [])
    stack.refresh(predicted=True, facts={"det": False, "n_gpu": 1, "token_floors": {"n1400": 1400}})   # the PRED line's call
    post = stack.refresh()                                                                     # the exit tally's bare call
    assert "triattn_conf" in post["levers_inert"] and post["levers_inert_reasons"]["triattn_conf"].startswith("lever_inert_by_design:triattn_conf(conf_path=offload_rows:8"), post.get("levers_inert_reasons")
    row = [l for l in report.lever_lines(post) if "LEVER name=triattn_conf " in l]
    assert len(row) == 1 and "state=off reason=lever_inert_by_design:triattn_conf(conf_path=offload_rows:8" in row[0], row
    assert not [f for f in post["levers_fallback"] if "triattn_conf" in f], post["levers_fallback"]
    again = stack.refresh()                                                                    # idempotent: a second replay, the same buckets
    assert again["levers_inert_reasons"]["triattn_conf"] == post["levers_inert_reasons"]["triattn_conf"] and "triattn_conf" not in again["levers_applied"]
