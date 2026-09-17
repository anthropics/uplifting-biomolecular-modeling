"""CPU tests of the memory adapter (boltz2_opt.big) on the core's memory-mode library (opt_core.mem): the mode's row -> the line, the
composition on this kit's mode table, apply over stub hook classes (no boltz, no GPU: the levered tensor paths run on the GPU proof legs),
the per-unit census + exit gate, the report block stack.evidence / report.lever_state read, and the refusals by name."""
import importlib.util
import os

import pytest

from .. import big, modes, registry, report as rep, stack
from .test_cli_and_evidence import TEMPL_OK

TORCH = importlib.util.find_spec("torch") is not None       # the core's allocator lever refuses by name without torch (nothing would read the conf)
CORE_MISSING = big.core_missing()                           # the memory-mode library (opt_core.mem registry/record/chunk/allocator) is the core 0.4.0's: on an older pin the modes refuse by name
needs_core = pytest.mark.skipif(CORE_MISSING is not None, reason=str(CORE_MISSING))


def _hooks():
    """Stub hook classes standing in for boltz's (the attributes the levers rebind)."""
    class Transition:
        def forward(self, x, chunk_size=None):
            return ("stock_transition", x)

    class DiffusionConditioning:
        def forward(self, s_trunk, z_trunk, relative_position_encoding, feats):
            return ("stock_cond",)

    class ConfidenceModule:
        def forward(self, *a, **k):
            return {"stock_conf": True}

    class RelativePositionEncoder:
        def forward(self, feats):
            return ("stock_relpos", feats)

    class Boltz2:
        emulate = ()                                              # the lever call counters the stub predict_step bumps (the levered paths' own marks)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            for key in self.emulate:
                big._cnt(key)
            return {"exception": False, "batch": batch}
    return {"xl_trans": {"transition_cls": Transition}, "xl_cond": {"cond_cls": DiffusionConditioning},
            "xl_free": {"confidence_cls": ConfidenceModule, "cond_cls": DiffusionConditioning, "model_cls": Boltz2},
            "relpos_lazy": {"relpos_cls": RelativePositionEncoder, "model_cls": Boltz2, "cond_cls": DiffusionConditioning}}


def _row(mode="big", **env):
    """The adapter's parse of the mode's row (+ overrides) with the unit levers only: the core's process-scope allocator lever
    (expandable_segments) is confirmed by the allocator read-back once CUDA is up — on a CPU box it is `not in force: never exercised` by
    design (fail-closed; test_the_allocator_lever_* below), so the accounting tests compose the line without it."""
    row = big.row_settings({**modes.env_row(mode), **env})
    return dict(row, levers=tuple(lv for lv in row["levers"] if lv != "expandable_segments"))


@pytest.fixture(autouse=True)
def _reset():
    yield
    big.reset_for_tests()


def test_one_lever_set_across_the_adapter_the_registry_and_the_rows():
    assert big.BIG_LEVERS == ("expandable_segments", "xl_trans", "xl_cond", "xl_free", "relpos_lazy") and not hasattr(big, "BIG_EXACT_LEVERS"), "one memory line"
    assert big.LEVERS == big.BIG_LEVERS, "worker_launch's probe: the adapter's lever names"
    assert set(registry.MEMORY_LEVERS) == set(big.BIG_LEVERS) and all(registry.LEVERS[n]["file"] == "boltz2_opt/big.py" for n in big.BIG_LEVERS)
    assert set(big.XL_NAMES.values()) == set(big.BIG_LEVERS) - {"expandable_segments"}, "BOLTZ_XL_LEVERS tokens name the unit levers; the allocator lever rides PYTORCH_CUDA_ALLOC_CONF"
    for m in ("big",):
        assert [n for n in modes.levers(m) if n in registry.MEMORY_LEVERS] == list(registry.MEMORY_LEVERS)
        assert registry.LEVERS["relpos_lazy"]["switch"] == "BOLTZ_XL_LEVERS" and "relpos" in modes.env_row(m)["BOLTZ_XL_LEVERS"].split(",")
    assert "trimul_rowchunk" not in registry.LEVERS and "trimul_rowchunk" not in big.BIG_LEVERS, "not adopted on the kernels-on line: named as a candidate, never wired"
    assert big.DROP == ("graph_sampler", "dit_hoist") and "BOLTZ_GRAPH_DIFFUSION" not in modes.env_row("big") and "BOLTZ_DIT_HOIST" not in modes.env_row("big") \
        and big.drops_for(modes.env_row("big")) == big.DROP == big.drops_for({}) and big.drops_for(modes.env_row("fast")) == ("graph_sampler",), "the hoist leaves the memory row with the roll-out (no CUDA graphs in the memory mode, 0.3.17): the composed line names the per-step graph patch and the hoist as left out; a row that carries the hoist names only the graph patch"
    assert set(stack.XL_ACTIVITY) == set(big.XL_NAMES.values()) and set(stack.XL_ACTIVITY.values()) <= set(big.STATS)


def test_row_settings_reads_the_mode_row():
    assert big.row_settings({}) is None and big.row_settings({"BOLTZ_XL": "0", "BOLTZ_XL_LEVERS": "trans"}) is None, "no attach switch: nothing to apply (worker_launch refuses by name)"
    b = big.row_settings(modes.env_row("big"))
    assert b["base"] == "fast" and b["levers"] == big.BIG_LEVERS and b["tokens"] == ["trans", "cond", "free", "relpos"] and b["min_tokens"] == 0
    assert b["settings"] == {"xl_trans": {"rows": 256}, "xl_cond": {"rows": 256, "fp32": 1}}
    e = big.row_settings(modes.env_row("big"))
    assert e["base"] == "fast" and e["levers"] == big.BIG_LEVERS
    assert e["attn_base"] == "pairtrack" and e["pairfuse"] is True                              # fast's fused pair track is the pair-stack base (modes.BIG_ATTN_BASE, CHANGES 0.3.4)
    with pytest.raises(RuntimeError, match="composes on fast"):
        big.row_settings({k: v for k, v in modes.env_row("big").items() if k not in ("BOLTZ_PAIRBLOCK", "BOLTZ_PAIRFUSE", "BOLTZ_TRIATTN")})   # a row with neither fast's fused pair track nor the flash patch is refused by name (no kernels-off memory line)
    flash = big.row_settings({**{k: v for k, v in modes.env_row("big").items() if k not in ("BOLTZ_PAIRBLOCK", "BOLTZ_PAIRBLOCK_MIN_TOKENS", "BOLTZ_TRANSITION", "BOLTZ_PAIRFUSE")}, **modes._F2_ROW})
    assert flash["attn_base"] == "flash" and flash["pairfuse"] is False and flash["levers"] == big.BIG_LEVERS   # the other base by name: the staged flash patch in the stock Pairformer statements
    sub = big.row_settings({**modes.env_row("big"), "BOLTZ_XL_LEVERS": "trans,cond,free", "BOLTZ_XL_TRANS_ROWS": "128", "BOLTZ_XL_COND_FP32": "0", "BOLTZ_XL_MIN_TOKENS": "300"})
    assert sub["levers"] == ("expandable_segments", "xl_trans", "xl_cond", "xl_free") and sub["settings"]["xl_trans"]["rows"] == 128 and sub["settings"]["xl_cond"]["fp32"] == 0 and sub["min_tokens"] == 300
    with pytest.raises(RuntimeError, match="BOLTZ_XL_LEVERS names \\['triatt'\\]: not derived by this adapter"):
        big.row_settings({**modes.env_row("big"), "BOLTZ_XL_LEVERS": "trans,triatt"})


@needs_core
def test_compose_on_this_kits_mode_table():
    mode, base, line = big.compose("fast")
    assert (mode, base) == ("big", "fast") and line.levers == big.BIG_LEVERS and line.drop == big.DROP and line.name == "big" and line.base_rule == "fast-else-exact"
    assert line.describe() == "big = fast + [expandable_segments,xl_trans,xl_cond,xl_free,relpos_lazy] - [graph_sampler,dit_hoist]"
    mode, base, line = big.compose("fast", ("expandable_segments", "xl_trans"))
    assert (mode, base) == ("big", "fast") and line.levers == ("expandable_segments", "xl_trans")
    for b_ in ("exact", "exact_nk"):
        with pytest.raises(ValueError, match="unknown big base"):
            big.compose(b_)
    with pytest.raises(ValueError, match="not in the big line"):
        big.compose("fast", ("trimul_rowchunk",))


@needs_core
def test_apply_patches_the_named_sites_and_the_census_accounts_every_unit(capsys):
    hooks = _hooks()
    row = _row()
    env = {**modes.env_row("big")}
    rec = big.attach(row, hooks=hooks, environ=env)
    assert [a.lever for a in rec.applied] == list(row["levers"]) and rec.mode == "big" and rec.base == "fast" and not rec.refused
    T, C, F, R, M = hooks["xl_trans"]["transition_cls"], hooks["xl_cond"]["cond_cls"], hooks["xl_free"]["confidence_cls"], hooks["relpos_lazy"]["relpos_cls"], hooks["xl_free"]["model_cls"]
    assert T.forward is big._trans_forward and C.forward is big._cond_forward and F.forward is big._conf_forward and R.forward is big._relpos_forward and M.predict_step is big._predict_step
    err = capsys.readouterr().err
    assert err.count(f"[{big.TAG}] ACTIVE mode=big:") == 1 and " base=fast " in err and "drop=graph_sampler,dit_hoist" in err, err   # the hoist leaves with the roll-out by rule (0.3.17): a dropped base lever of the line again, as in 0.3.8
    lever_lines = [l for l in err.splitlines() if " LEVER " in l]
    parsed = {dict(t.split("=", 1) for t in l.split(" ") if "=" in t)["name"]: dict(t.split("=", 1) for t in l.split(" ") if "=" in t) for l in lever_lines}
    assert set(parsed) == set(row["levers"]) | set(big.drops_for(env)), parsed.keys()      # one line per unit lever + one per base lever the line names as left out (the per-step graph patch; the hoist rides the row)
    for n in ("xl_trans", "xl_cond", "xl_free", "relpos_lazy"):
        assert parsed[n]["state"] == "on" and parsed[n]["strategy"] == "F7.chunked_eval" and parsed[n]["origin"] == "kit" and parsed[n]["impl"] == "boltz2_opt.big" and parsed[n]["exact"] == "bitwise", parsed[n]
    assert parsed["xl_trans"]["rows"] == "256" and parsed["xl_cond"]["fp32"] == "1" and parsed["graph_sampler"]["state"] == "off" and parsed["graph_sampler"]["reason"] == "drop"
    # two units: the levered paths mark themselves (emulated by the stub's counters); the second unit's conditioner "did not run" -> xl_free skipped by name, xl_cond absent -> partial
    M.emulate = ("xl_trans", "xl_cond", "xl_free", "relpos_lazy")
    M().predict_step({"i": 0}, 0)
    M.emulate = ("xl_trans", "relpos_lazy")
    out = M().predict_step({"i": 1}, 1)
    assert out == {"exception": False, "batch": {"i": 1}}, "the stock predict_step's return value passes through"
    xr = big.report()
    assert xr["applied"] == list(row["levers"]) and xr["n_units"] == 2 and xr["mode"] == "big" and xr["settings"]["xl_cond"] == {"rows": 256, "fp32": 1}
    assert xr["stats"]["trans_rowchunked_calls"] == 2 and xr["stats"]["cond_rowchunked_calls"] == 1 and xr["stats"]["free_events"] == 1 and xr["stats"]["relpos_lazy_released"] == 2
    cen = xr["census"]["units"]
    assert cen["predict_step0"]["partial"] == [] and sorted(cen["predict_step0"]["ran"]) == sorted(["xl_trans", "xl_cond", "xl_free", "relpos_lazy"])
    assert cen["predict_step1"]["partial"] == ["xl_cond"] and "xl_free" in cen["predict_step1"]["skipped"], cen["predict_step1"]
    assert xr["exit"]["partial"] == ["xl_cond"] and xr["exit"]["exit_code"] == 3 and not xr["exit"]["allow_partial"], "fail-closed: a lever absent on one unit refuses the gate"
    # the parent's evidence names the same fact (stack.evidence over this xl_report) and report.lever_state renders it
    log = {"env": {"boltz_levers": ["resid", "mask2"]}, "per_item": [{"name": "a", "seed": 0}], "events": [], "xl_report": xr,
           "templ_report": dict(TEMPL_OK)}
    ev, problems = stack.evidence("big", log, "")
    want = "memory levers partial per the adapter's census (exit gate refused): xl_cond: predict_step1: never marked"
    assert any(p.startswith(want) for p in problems) and any(p.startswith("memory levers not reported applied: expandable_segments") for p in problems), problems
    st, reason, pairs = rep.lever_state("xl_cond", ev, {"levers_fallback": [], "gates": {}})
    assert st == "on" and ("calls", 1) in pairs and ("census", "partial") in pairs and ("fp32", 1) in pairs, pairs
    st, reason, pairs = rep.lever_state("relpos_lazy", ev, {"levers_fallback": [], "gates": {}})
    assert st == "on" and ("calls", 2) in pairs and ("recomputed", 0) in pairs
    assert " xl=" in rep._xl(ev) and "xl_exit=partial" in rep._xl(ev) and "xl_units=2" in rep._xl(ev)
    # reset restores every site
    big.reset_for_tests()
    assert T.forward is not big._trans_forward and M.predict_step is not big._predict_step and C.forward is not big._cond_forward


@needs_core
def test_a_complete_run_passes_the_gate_and_the_parents_evidence():
    hooks = _hooks(); row = _row(); M = hooks["xl_free"]["model_cls"]
    big.attach(row, hooks=hooks, environ={**modes.env_row("big")})
    M.emulate = ("xl_trans", "xl_cond", "xl_free", "relpos_lazy")
    M().predict_step({}, 0); M().predict_step({}, 1)
    xr = big.report()
    assert xr["exit"]["partial"] == [] and xr["exit"]["exit_code"] == 0 and all(u["partial"] == [] for u in xr["census"]["units"].values())
    log = {"env": {"boltz_levers": ["resid", "mask2"]}, "per_item": [{"name": "a", "seed": 0}], "events": [], "xl_report": xr,
           "templ_report": dict(TEMPL_OK)}
    problems = [p for p in stack.evidence("big", log, "")[1] if p.startswith("memory levers")]   # the base's own evidence (fast's flash report, the TriMul census) is not this log's concern
    assert problems == [f"memory levers not reported applied: expandable_segments (xl_report.applied={list(row['levers'])})"], (problems, "the one named problem: this CPU line left the allocator lever out")


@needs_core
def test_a_missing_hook_refuses_by_name_and_leaves_nothing_patched(capsys):
    mem = big._mem()
    hooks = _hooks(); T = hooks["xl_trans"]["transition_cls"]; stock = T.forward
    hooks["relpos_lazy"] = {"model_cls": hooks["xl_free"]["model_cls"], "cond_cls": hooks["xl_cond"]["cond_cls"]}      # no relpos_cls
    with pytest.raises(mem.Refused) as ei:
        big.attach(_row(), hooks=hooks, environ={**modes.env_row("big")})
    assert [r.lever for r in ei.value.record.refused] == ["relpos_lazy"] and ei.value.record.refused[0].precondition == "hooks.relpos_cls"
    assert T.forward is stock, "the levers applied before the refusal are undone: no half-levered process"
    assert f"[{big.TAG}] NOT ACTIVE: big refused — relpos_lazy: hooks.relpos_cls" in capsys.readouterr().err


@needs_core
def test_the_selection_is_explicit_and_reads_no_environment_word():
    """The core applies exactly the row: a BOLTZ2_BIG_* word in the worker's environment changes nothing (the core reads no environment for the
    selection; the caller refuses such a word by name before launch, cli.resolve_mode / _autoload.undeclared_names)."""
    hooks = _hooks()
    rec = big.attach(_row(), hooks=hooks, environ={**modes.env_row("big"), "BOLTZ2_BIG_XL_FREE": "0", "BOLTZ2_BIG_ALLOW_PARTIAL": "1", "BOLTZ2_BIG_XL_TRANS_ROWS": "7"})
    assert [a.lever for a in rec.applied] == ["xl_trans", "xl_cond", "xl_free", "relpos_lazy"] and list(rec.off_by_flag) == [] and rec.allow_partial is False
    assert big._S["ctx"].setting("xl_trans", "rows", 0, cast=int) == 256, "the row's value, not the word's"

def test_relpos_lazy_needs_the_conditioner_hook_in_the_line():
    mem = big._mem()
    row = dict(_row(), levers=tuple(lv for lv in _row()["levers"] if lv not in ("xl_cond", "xl_free")))
    with pytest.raises(mem.Refused) as ei:
        big.attach(row, hooks=_hooks(), environ={**modes.env_row("big")})
    assert ei.value.record.refused[0].lever == "relpos_lazy" and ei.value.record.refused[0].precondition == "lever.xl_cond"


def test_apply_returns_nothing_without_the_rows_switch(monkeypatch):
    for k in [k for k in os.environ if k.startswith("BOLTZ_")]:
        monkeypatch.delenv(k)
    assert big.apply() == [], "BOLTZ_XL unset: nothing applied, boltz and the core's library never imported (worker_launch refuses by name: applied nothing)"
    assert big.report()["applied"] == [] and "error" in big.report()


def test_the_memory_modes_refuse_by_name_on_a_core_without_the_library(monkeypatch):
    """The adapter is written on the core's memory-mode library (big.REQUIRED_PRODUCERS, core >= 0.4.0): on an older core the worker's attach
    refuses the memory modes by name (`[boltz2-opt attach] REFUSED xl: ModuleNotFoundError: producer_missing:<modules> …`, exit 3; a core
    absent altogether reads `core_missing:opt_core …`) — never a silent stock run, never a second implementation in this tree."""
    import importlib.util as iu
    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None if name in ("opt_core.mem.registry", "opt_core.mem.chunk") else real(name, *a, **k))   # an older core: two producers absent
    missing = big.missing_producers()
    assert missing[0] == "opt_core.mem.registry" and "opt_core.mem.chunk" in missing and all(m in big.REQUIRED_PRODUCERS for m in missing), missing
    reason = big.core_missing()
    assert reason.startswith("producer_missing:opt_core.mem.registry,") and "boltz2_opt.big imports opt_core >= 0.4.0" in reason and "this package pins opt_core 0.4.1" in reason, reason
    monkeypatch.setenv("BOLTZ_XL", "1"); monkeypatch.setenv("BOLTZ_XL_LEVERS", "trans,cond,free,relpos"); monkeypatch.setenv("BOLTZ_TRIATTN", "flash")
    with pytest.raises(ModuleNotFoundError, match="^producer_missing:opt_core.mem.registry,"):
        big.apply()
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None if name.startswith("opt_core") else real(name, *a, **k))    # no core at all
    assert big.core_missing().startswith("core_missing:opt_core — the shared core is not importable")
    monkeypatch.setattr(iu, "find_spec", real)
    assert (big.missing_producers() == [] and big.core_missing() is None) if CORE_MISSING is None else big.core_missing().startswith("producer_missing:")


@needs_core
@pytest.mark.skipif(TORCH, reason="the torch-less refusal of the allocator lever")
def test_the_allocator_lever_refuses_by_name_without_torch():
    mem = big._mem()
    row = big.row_settings(modes.env_row("big"))                      # the full line, expandable_segments included
    with pytest.raises(mem.Refused) as ei:
        big.attach(row, hooks=_hooks(), environ={**modes.env_row("big")})
    assert ei.value.record.refused[0].lever == "expandable_segments" and ei.value.record.refused[0].precondition == "torch"


@needs_core
@pytest.mark.skipif(not TORCH, reason="the allocator lever's read-back needs torch importable")
def test_the_allocator_lever_is_fail_closed_until_the_allocator_read_back_proves_it(capsys):
    """On a process that never initialises CUDA the core's expandable_segments lever applies (the conf is exported) but its read-back
    cannot pass: the report says alloc_effective=false and the parent names it — never a silent `set, therefore in force`."""
    hooks = _hooks(); M = hooks["xl_free"]["model_cls"]
    env = {k: v for k, v in modes.env_row("big").items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    rec = big.attach(big.row_settings(modes.env_row("big")), hooks=hooks, environ=env)
    assert [a.lever for a in rec.applied] == list(big.BIG_LEVERS) and env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True", "the lever exports the conf into the worker environment"
    line = [l for l in capsys.readouterr().err.splitlines() if " LEVER name=expandable_segments " in l][0]
    assert "strategy=F7.expandable_segments" in line and "origin=core" in line and "impl=opt_core.mem.allocator" in line, line
    M.emulate = ("xl_trans", "xl_cond", "xl_free", "relpos_lazy"); M().predict_step({}, 0)
    xr = big.report()
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        pytest.skip("CUDA initialised in the test process: the GPU proof legs cover the in-force case")
    assert xr["env"] == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "alloc_effective": "false", "BOLTZ2_MEMTRACE": None}
    assert xr["exit"]["partial"] == ["expandable_segments"] and "never exercised" in xr["exit"]["reasons"]["expandable_segments"], xr["exit"]
    problems = stack.evidence("big", {"env": {"boltz_levers": []}, "per_item": [{"name": "a"}], "events": [], "xl_report": xr}, "")[1]
    assert any(p.startswith("memory levers partial per the adapter's census (exit gate refused): expandable_segments") for p in problems) and \
        any(p.startswith("allocator setting carried in the environment but not effective in the worker process") for p in problems), problems


@needs_core
def test_relpos_lazy_recomputes_the_released_encoding_for_the_stock_conditioner_too():
    """A row without xl_cond (the registration hook of xl_free stands on DiffusionConditioning.forward): a rel-pos encoding that
    relpos_lazy released (storage size 0) is recomputed by the trunk's rel_pos module before the STOCK conditioner reads it — the stock
    body never sees a released tensor; a live encoding passes through untouched."""
    pytest.importorskip("torch", reason="relpos lazy re-encode check needs torch; runs on the torch-extra route")
    import types
    hooks = _hooks(); M = hooks["xl_free"]["model_cls"]; C = hooks["xl_cond"]["cond_cls"]
    seen = []
    C.forward = lambda self, s, z, rel, feats: seen.append(rel) or ("stock_cond",)                     # the "stock" conditioner records the encoding it is handed
    released = types.SimpleNamespace(untyped_storage=lambda: types.SimpleNamespace(nbytes=lambda: 0))          # what a released encoding looks like to the adapter
    live = types.SimpleNamespace(untyped_storage=lambda: types.SimpleNamespace(nbytes=lambda: 4096))
    M.rel_pos = staticmethod(lambda feats: ("recomputed", feats))                                      # the trunk's rel_pos module
    M.predict_step = lambda self, batch, batch_idx, dataloader_idx=0: (C().forward(None, None, released, {"f": 1}), C().forward(None, None, live, {"f": 2}))   # the "stock" unit: the conditioner runs twice
    rec = big.attach(_row(BOLTZ_XL_LEVERS="trans,free,relpos"), hooks=hooks, environ=modes.env_row("big"))   # a row without xl_cond
    assert [a.lever for a in rec.applied] == ["xl_trans", "xl_free", "relpos_lazy"] and C.forward is big._cond_forward, "xl_free installs the registration hook over the stock conditioner"
    M().predict_step({}, 0)
    assert seen == [("recomputed", {"f": 1}), live], seen
    st = big.report()["stats"]
    assert st["relpos_lazy_recomputed"] == 1 and st["cond_stock_calls"] == 2 and st["cond_rowchunked_calls"] == 0, st
