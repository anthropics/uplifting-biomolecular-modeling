"""fastkit.py: the design kit installed on the STOCK cookbook module from outside — (1) the featurisation cache around
``prepare_esmfold2_tensors`` (the stock function computes every miss; hits are clones; non-cacheable inputs pass straight through),
(2) ``ESMFold2Design._enable_fast_kit`` with the ONE composition of each EF2_FAST_KIT value (fastkit.COMPOSITION), (3) ``design`` enabling
for the complex size the loop builds before the stock loop + one guard check after it, (4) at enable: the loop-level levers on the module's
functions with the state guard's check at every design fold after the first outermost. No cookbook file is copied or edited: the tree carries
exactly one cookbook, the stock file. The enable ORDER is the contract tested here: every helper rebinding before any CUDA-graph capture."""
import glob
import logging
import os
import sys
import types

import pytest

from .. import fastkit as FK, launch, modes, stock_design as SD
from . import _paths as P
from .test_big import _fake_cookbook, _fake_kit, _Model


class _Guard:
    def __init__(self, **kw):
        self.kw = kw; self.checks = []

    def check(self, label):
        self.checks.append(label)


def _kit_with(calls, guards):
    kit = _fake_kit(calls)
    kit["ef2_state_guard"].StateGuard = lambda **kw: guards.append(_Guard(**kw)) or guards[-1]
    return kit


def test_the_tree_carries_one_cookbook_the_stock_file():
    assert os.path.isfile(P.STOCK_FILE)
    assert not os.path.exists(os.path.join(P.DK, "cookbook")) and not glob.glob(os.path.join(P.FWD, "**", "binder_design*.py"), recursive=True)
    assert "fastkit_cookbook" not in modes.kit_paths(P.ROOT) and all(m.is_kit == (m.kit_switch is not None) for m in modes.MODES.values())
    assert set(FK.COMPOSITION) == set(FK.KIT_SWITCHES) == {m.kit_switch for m in modes.MODES.values() if m.is_kit} == {"exact", "agk3", "big"} and FK.COMPOSITION["agk3"] is not FK.COMPOSITION["big"]   # one composition per mode switch (fast's word is agk3)


def test_install_hooks_three_names_and_leaves_the_rest(monkeypatch):
    for name, mod in _kit_with([], []).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook()
    before = dict(vars(BD)); orig_design = BD.ESMFold2Design.design
    rec = FK.install(BD, "exact")
    assert rec == {"name": "design_kit", "applied": True, "kit": "exact", "hooks": list(FK.HOOKS), "module": "binder_design", "source": FK.SOURCE, "levers_off": [], "user_switches": {}, "function_sha256": rec["function_sha256"]}
    assert BD._USER_SWITCHES == {} and FK.user_switches_of(BD) == {}                              # no user pair-stack switch: the composition's own values everywhere
    assert BD.prepare_esmfold2_tensors is not before["prepare_esmfold2_tensors"] and BD.prepare_esmfold2_tensors.__wrapped__ is before["prepare_esmfold2_tensors"]
    assert BD.fold_and_get_distogram is before["fold_and_get_distogram"] and BD.ESMFold2Design.design.__wrapped__ is orig_design      # the fold is wrapped at enable, not at install
    assert callable(BD.ESMFold2Design._enable_fast_kit) and BD.FAST_KIT == "exact" and BD._D59_GUARD is None and BD._FEATURE_CACHE == {}
    changed = {k for k in vars(BD) if k not in before or vars(BD)[k] is not before.get(k)}
    assert changed == {"prepare_esmfold2_tensors", "FAST_KIT", "_D59_GUARD", "_FEATURE_CACHE", "_design_folds", "_LEVERS_OFF", "_KIT_ASIDE", "_USER_SWITCHES"}, changed
    assert BD._LEVERS_OFF == () and BD._KIT_ASIDE == {}


def test_featurisation_cache_memoises_the_stock_function_per_designed_sequence(monkeypatch):
    for name, mod in _kit_with([], []).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook(); n_calls = []
    stock = BD.prepare_esmfold2_tensors
    BD.prepare_esmfold2_tensors = lambda input, **kw: n_calls.append(1) or stock(input, **kw)
    FK.install(BD, "exact")
    prot = lambda i, seq: types.SimpleNamespace(id=i, sequence=seq, msa=None, modifications=None)
    inp = types.SimpleNamespace(sequences=[prot("A", "ACDE"), prot("B", "XXXX")])
    a = BD.prepare_esmfold2_tensors(inp, max_atoms=7); b = BD.prepare_esmfold2_tensors(inp, max_atoms=7)
    assert a == b == {"n": [4, 4], "max_atoms": 7} and a is not b and len(n_calls) == 1                  # one stock call; hits are fresh dicts
    BD.prepare_esmfold2_tensors(inp, max_atoms=8); assert len(n_calls) == 2                              # max_atoms is part of the key
    BD.prepare_esmfold2_tensors(types.SimpleNamespace(sequences=[prot("A", "ACDF"), prot("B", "XXXX")]), max_atoms=7); assert len(n_calls) == 3   # a new designed sequence: a miss
    with_msa = types.SimpleNamespace(sequences=[types.SimpleNamespace(id="A", sequence="ACDE", msa=object(), modifications=None)])
    BD.prepare_esmfold2_tensors(with_msa, max_atoms=7); BD.prepare_esmfold2_tensors(with_msa, max_atoms=7); assert len(n_calls) == 5        # not cacheable: straight through, never stored
    assert len(BD._FEATURE_CACHE) == 3


def test_design_enables_for_the_complex_size_then_runs_the_stock_loop_with_the_guard_checked_per_step(monkeypatch):
    calls, guards = [], []
    for name, mod in _kit_with(calls, guards).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook(); stock_fold = BD.fold_and_get_distogram; FK.install(BD, "agk3")
    app = BD.ESMFold2Design()
    out = app.design("pd-l1", "minibinder", seed=2)                       # built-in target "ACDE|FGH" = 7 residues + the seed-2 prompt draw (12)
    assert out == (["SEQ"], {}, []) and app._fast_kit_tokens == 19 == FK.complex_tokens(BD, target_name="pd-l1", target_sequence=None, binder_name="minibinder", binder_sequence=None, seed=2)
    assert FK.complex_tokens(BD, target_name="x", target_sequence="AB|CD", binder_name=None, binder_sequence="WWW", seed=0) == 7
    names = [n for n, _ in calls]
    assert [kw for n, kw in calls if n == "agk.enable"] == [{"trimul": None, "transition": "refround_lean", "checkpoint": "block", "transition_fwd": "t16"}] * 2       # K-D3 through agk; the TriMul is ef2_trimul's
    assert [kw for n, kw in calls if n == "trimul"] == [{"variant": "cueq_tiles"}] + [{"variant": "fused"}] * 2 and names.count("fused_ln") == 1 and names.count("stepgraph") == 2      # the tile table through the critic; 19 tokens <= POOL_MAX_TOKENS: the pool, one per inversion model
    assert names.count("esmc_graph") == 1 and names.count("pppl_graph") == 1 and all(m.chunk_set == [FK.BIG_CHUNK_SIZE] for m in app.inversion_models.values())   # fast carries the chunk word (64)
    assert [kw["copies"] for n, kw in calls if n == "bwd_ckpt"] == [2, 2] and all(kw["extra_reserved_bytes"] > 2.4 * 2 ** 30 for n, kw in calls if n == "bwd_ckpt")   # the LM graph pools + the trunk pools' slots budgeted before the plan
    assert BD.fold_and_get_distogram is not stock_fold and getattr(BD.fold_and_get_distogram, "_ef2_fold_guard", False)                 # hook 4: the guard outermost of the kit's fold chain
    assert len(guards) == 1 and guards[0].kw["mode"] == "assert"
    assert guards[0].checks == ["end of design step 0", "end of design step 1", "end of design step 2"]   # 3 design folds + 1 critic fold: a check per step boundary, the last after the loop
    assert len(BD.folds) == 4                                                                              # every fold reached the stock function unchanged
    calls.clear(); app.design("pd-l1", "minibinder", seed=2)                                             # the same size again: nothing re-installed (the named no-op)
    assert calls == []
    app.design("pd-l1", "minibinder", seed=3)                                                            # a new size in this process: the pools and memos are returned, then re-planned
    names = [n for n, _ in calls]
    assert names[:5] == ["stepgraph.disable", "stepgraph.disable", "esmc_graph.release", "esmc_hoist.release", "pppl_graph.release"] and "loop_prep.release" in names and names.count("bwd_ckpt") == 2 and app._fast_kit_tokens == 20


def test_enable_order_every_rebinding_before_any_capture(monkeypatch):
    """The composition's order (fastkit module docstring): loop-level levers innermost-first then the guard; process-wide rebindings; per-model
    patches; sampler levers on every fold model; the memory plan; then the captures (trunk pool, ESMC forward, the target-chain hoist over it, pPPL)."""
    for kit, n in (("agk3", 200), ("exact", 200), ("exact", 300), ("big", 200), ("big", 900)):
        calls, guards = [], []
        for name, mod in _kit_with(calls, guards).items():
            monkeypatch.setitem(sys.modules, name, mod)
        BD = _fake_cookbook(); FK.install(BD, kit); app = BD.ESMFold2Design(); app._enable_fast_kit(n)
        names = [nm for nm, _ in calls]
        first = {nm: names.index(nm) for nm in set(names)}
        captures = [first[nm] for nm in ("stepgraph", "esmc_graph", "esmc_hoist", "pppl_graph") if nm in first]
        rebind = [first[nm] for nm in ("loop_prep", "loop_pppl", "overlap", "fused_ln", "t16.engage", "trimul", "rope", "agk.enable", "skip_conf", "lazy", "pairbias", "sampler_graph", "bwd_ckpt") if nm in first]
        moff = FK.mode_off(kit)                                                                                         # big: the levers switched off by name are never called
        assert names[:3] == (["loop_prep", "loop_pppl", "overlap"] if "ef2_esmc_overlap" not in moff else ["loop_prep", "loop_pppl", "fused_ln"]) and max(rebind) < min(captures, default=len(names)), (kit, n, names)
        assert first["bwd_ckpt"] > max(first[nm] for nm in ("agk.enable", "trimul", "pairbias", "sampler_graph") if nm in first) and (names[-3:] == ["esmc_graph", "esmc_hoist", "pppl_graph"] or {"ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph"} <= set(moff))   # the hoist wraps OVER the ESMC graph's forward: after it
        assert names.count("rope") == 3 and names.count("pairbias") == (0 if "ef2_pairbias_attn" in moff else 3) and names.count("sampler_graph") == (0 if "ef2_sampler_graph" in moff else 3)      # 2 inversion trunks + the LM's; 2 inversion models + 1 critic
        pool = FK.trunk_pool_wanted(kit, n)
        assert names.count("stepgraph") == (2 if pool else 0) and pool == (n <= FK.POOL_MAX_TOKENS and kit != "big")   # one size rule on the compositions that carry the pool (exact, fast); big carries none at any size
        if kit == "exact":
            assert [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "cueq_tiles"}] * 2 and "fused_ln" not in first and "bf16_conf" not in first and all(m.chunk_set == [] for m in app.inversion_models.values())
            assert [kw for nm, kw in calls if nm == "agk.enable"] == [{"trimul": None, "transition": None, "checkpoint": "block", "transition_fwd": None}] * 2
            assert not [kw for nm, kw in calls if nm == "t16.engage"]                                                             # exact never engages the fast-class forward kernel
            assert app.inversion_models["a"].__dict__["_ef2_trimul_handle"].variant == "cueq_tiles" == app.hf_critic_models["c"].__dict__["_ef2_trimul_handle"].variant   # exact: the table through the first inversion model (the pinned backend) and the critic (its own pair width)
        else:
            assert [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "cueq_tiles"}] + [{"variant": "fused"}] * 2 and [kw for nm, kw in calls if nm == "agk.enable"] == [{"trimul": None, "transition": "refround_lean", "checkpoint": "block", "transition_fwd": "t16"}] * 2
            assert names.count("fused_ln") == 1 and names.count("bf16_conf") == 2 and all(m.chunk_set == [FK.BIG_CHUNK_SIZE] for m in app.inversion_models.values())
            assert names.count("t16.engage") == 1 and first["t16.engage"] < first["agk.enable"] < first["bwd_ckpt"]                # the t16 forward engaged once per process, before agk installs K-D3 on it and before the memory plan / any capture
            assert app.hf_critic_models["c"].__dict__["_ef2_trimul_handle"].variant == "cueq_tiles"     # fast / big: through the hero critic, never an inversion model (theirs is the fused kernel's)


def test_pool_size_rule_and_its_word():
    assert FK.POOL_MAX_TOKENS == 256 and FK.TRUNK_POOL_SLOTS == 2 == FK.TRUNK_PASSES
    assert FK.trunk_pool_wanted("exact", 256) and not FK.trunk_pool_wanted("exact", 257) and FK.trunk_pool_wanted("agk3", 195) and not FK.trunk_pool_wanted("agk3", 257) and not FK.trunk_pool_wanted("big", 195) and not FK.trunk_pool_wanted("big", 257)   # one size rule on the compositions that carry the pool (exact, fast); big carries none
    assert FK.pool_word("big", 200) == "none" == FK.pool_word("big", 431) and FK.pool_word("agk3", 200) == "2 slots" and FK.pool_word("agk3", 431) == "none above 256 tokens" and FK.pool_word("exact", 257) == "none above 256 tokens"


def test_pppl_knob_is_read_with_the_kit_default(monkeypatch):
    calls, guards = [], []
    for name, mod in _kit_with(calls, guards).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv(FK.PPPL_VAR, raising=False); monkeypatch.delenv(FK.D59_VAR, raising=False)
    BD = _fake_cookbook(); FK.install(BD, "exact"); BD.ESMFold2Design()._enable_fast_kit(250)
    assert [n for n, _ in calls].count("pppl_graph") == 1
    monkeypatch.setenv(FK.PPPL_VAR, "0"); calls.clear()
    BD2 = _fake_cookbook(); FK.install(BD2, "exact"); BD2.ESMFold2Design()._enable_fast_kit(250)
    assert [n for n, _ in calls].count("pppl_graph") == 0                                                # F6: the knob's effect (the launcher strips it; envproof proves it absent)


def test_stock_design_enables_before_its_fold_observer_on_kit_arms_only():
    """stock_design.main installs the kit exactly when the mode is a kit arm (ONE `PATCH design_kit …` line either way) and enables it for the
    complex BEFORE wrapping its fold observer, so the kit's loop-level levers and per-step guard run inside the observed call."""
    import inspect
    src = inspect.getsource(SD.main)
    assert "kit_rec = FK.install(BD, mode, user_switches=user_switches) if mode.is_kit else FK.record_off()" in src and src.count("import_cookbook(a.cookbook, \"binder_design\")") == 1
    assert 0 < src.index("app._enable_fast_kit(n_tokens)") < src.index("BD.fold_and_get_distogram = fold_observed") and src.count("BD.fold_and_get_distogram = ") == 1
    assert "binder_design_fastkit" not in inspect.getsource(SD) and "apply_big" not in inspect.getsource(SD)


# ---- the ablation word MODEL_OPT_LEVERS_OFF (fastkit.levers_of / resolve_levers_off; read once at install; enable installs what is not named)
def test_levers_of_each_arm_are_the_census_names_in_enable_order():
    ex, fa, bi = FK.levers_of("exact"), FK.levers_of(modes.MODES["fast"]), FK.levers_of(modes.MODES["big"])
    assert ex == ["ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "cueq_tiles", "ef2_esmc_rope", "skip_unused_confidence", "ef2_lazy_structure", "ef2_pairbias_attn", "ef2_sampler_graph",
                  "ef2_bwd_ckpt", "ef2_stepgraph", "ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph", "featurisation_cache"] == FK.levers_of(modes.MODES["exact"])   # exact pins the switches on every model: no critic-scope lever
    assert fa == ["ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "ef2_fused_ln", "cueq_tiles", "ef2_esmc_rope", "chunk", "agk_transition", "ef2_t16_transition", "ef2_kd3_gemmswiglu", "trimul", "ef2_trimul_nosave", "skip_unused_confidence", "ef2_lazy_structure",
                  "ef2_bf16_confidence", "ef2_pairbias_attn", "ef2_sampler_graph", "ef2_bwd_ckpt", "ef2_stepgraph", "ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph", "featurisation_cache", "critic_switches"]
    assert bi == [n for n in fa if n != "ef2_stepgraph" and n not in FK.BIG_MODE_OFF] and FK.levers_of("agk3") == fa[:-1] and FK.levers_of("big") == bi[:-1] and FK.levers_of("off") == [] and FK.levers_of(modes.MODES["off"]) == []   # big's census names = fast's minus the trunk graph pool (the plan's rule is a word of ef2_bwd_ckpt, not a lever name); a bare switch word carries no critic-scope lever   # was: `fast` is an alias of big: ONE lever list (the pool among them); a bare switch word carries no critic-scope lever


def test_resolve_levers_off_names_the_closure_and_refuses_typos_and_foreign_names_by_name():
    assert FK.resolve_levers_off(None, "agk3") == {"asked": [], "off": [], "chained": {}} == FK.resolve_levers_off(" , ", "agk3")
    r = FK.resolve_levers_off("ef2_esmc_hoist", modes.MODES["exact"])
    assert r["off"] == ["ef2_esmc_hoist"] and r["chained"] == {}
    r = FK.resolve_levers_off("ef2_loop_prep, chunk", modes.MODES["fast"])                       # the chain by name: loop_pppl serves loop_prep, the overlap serves loop_pppl
    assert r["asked"] == ["ef2_loop_prep", "chunk"] and r["off"] == ["ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "chunk"] and r["chained"] == {"ef2_loop_pppl": "ef2_loop_prep", "ef2_esmc_overlap": "ef2_loop_pppl"}
    assert FK.resolve_levers_off("agk_transition", "agk3")["off"] == ["ef2_fused_ln", "agk_transition", "ef2_t16_transition", "ef2_kd3_gemmswiglu"]     # the frozen-LayerNorm backward and the t16 forward kernel serve K-D3 only
    assert FK.resolve_levers_off("ef2_t16_transition", "big")["off"] == ["ef2_t16_transition"]              # the forward kernel alone: K-D3 keeps its own forward
    assert FK.resolve_levers_off("critic_switches", modes.MODES["big"])["off"] == ["cueq_tiles", "critic_switches"]   # fast / big: the tile table reaches the process through a critic on cuequivariance
    with pytest.raises(FK.LeversOffError, match="unknown lever name\\(s\\) esmc_hoist .*EF2_FAST_KIT=exact composes ef2_loop_prep,"):
        FK.resolve_levers_off("esmc_hoist", "exact")                                               # a typo (the census name is ef2_esmc_hoist)
    with pytest.raises(FK.LeversOffError, match="EF2_FAST_KIT=exact does not compose chunk,trimul; its levers are "):
        FK.resolve_levers_off("chunk,trimul", modes.MODES["exact"])                                  # exact's chunk is the mode table's pin; its triangle multiplication is stock's
    with pytest.raises(FK.LeversOffError, match="does not compose critic_switches"):
        FK.resolve_levers_off("critic_switches", "agk3")                                             # a bare kit value knows no critic scope (the mode does)
    assert issubclass(FK.LeversOffError, ValueError) and FK.LEVERS_OFF_VAR == "MODEL_OPT_LEVERS_OFF" == launch.LEVERS_OFF_VAR and FK.LEVERS_OFF_VAR == "MODEL_OPT_LEVERS_OFF"   # a stripped prefix: the stock arm never sees it


def test_enable_yields_to_the_users_pair_stack_switches_by_name(monkeypatch):
    """A user ``--chunk-size`` / ``--kernel-backend`` over the mode's value is every model's already (stock's setter at load; install records it as
    BD._USER_SWITCHES): on fast / big the composition's chunk of 64 makes NO setter call and says so (_KIT_ASIDE["chunk"]); under a user backend
    other than cuequivariance the cuEquivariance tile table is not attempted on any model and steps aside by name (_KIT_ASIDE["cueq_tiles"]) — on
    exact too, where its own backend would have raised; every other lever of the composition enables exactly as before, nothing raises."""
    def run(mode, user, n=200):
        calls, guards = [], []
        for name, mod in _kit_with(calls, guards).items():
            monkeypatch.setitem(sys.modules, name, mod)
        BD = _fake_cookbook(); rec = FK.install(BD, modes.MODES[mode], user_switches=user)
        assert rec["user_switches"] == user == BD._USER_SWITCHES == FK.user_switches_of(BD)
        app = BD.ESMFold2Design(); app._enable_fast_kit(n)
        return BD, app, calls, [nm for nm, _ in calls]
    BD, app, calls, names = run("big", {"chunk_size": None})                                       # the kit's 64 yields to the user's none: no call, named
    assert all(m.chunk_set == [] for m in app.inversion_models.values()) and BD._KIT_ASIDE.get("chunk") == "user_chunk_size:none" and "cueq_tiles" not in BD._KIT_ASIDE
    assert [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "cueq_tiles"}] + [{"variant": "fused"}] * 2 and names[-1] == "bwd_ckpt" and "esmc_graph" not in names   # the rest as composed (big: the plan last, no graph captures)
    BD, app, calls, names = run("big", {"chunk_size": 32})
    assert all(m.chunk_set == [] for m in app.inversion_models.values()) and BD._KIT_ASIDE.get("chunk") == "user_chunk_size:32"
    BD, app, calls, names = run("big", {"kernel_backend": None})                                   # no model on cuequivariance: the table is not attempted anywhere, the fused kernel installs as ever
    assert [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "fused"}] * 2 and BD._KIT_ASIDE.get("cueq_tiles") == "user_kernel_backend:None" and "chunk" not in BD._KIT_ASIDE and all(m.chunk_set == [FK.BIG_CHUNK_SIZE] for m in app.inversion_models.values())
    assert "_ef2_trimul_handle" not in app.hf_critic_models["c"].__dict__
    BD, app, calls, names = run("exact", {"kernel_backend": "fused"})                                # exact under the user's backend: the table steps aside by name instead of raising; every other exact lever enabled
    assert "trimul" not in names and BD._KIT_ASIDE == {"cueq_tiles": "user_kernel_backend:fused"} and names[-3:] == ["esmc_graph", "esmc_hoist", "pppl_graph"] and names[:3] == ["loop_prep", "loop_pppl", "overlap"]
    assert [kw for nm, kw in calls if nm == "agk.enable"] == [{"trimul": None, "transition": None, "checkpoint": "block", "transition_fwd": None}] * 2 and all(m.chunk_set == [] for m in app.inversion_models.values())
    BD, app, calls, names = run("exact", {"chunk_size": 64})                                         # exact under the user's chunk: the setter is the design script's (every model at load), the kit calls none; the table installs as ever
    assert [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "cueq_tiles"}] * 2 and "chunk" not in BD._KIT_ASIDE and "cueq_tiles" not in BD._KIT_ASIDE and all(m.chunk_set == [] for m in app.inversion_models.values())
    BD, app, calls, names = run("big", {})                                                          # no user switch: the composition exactly as before
    assert all(m.chunk_set == [FK.BIG_CHUNK_SIZE] for m in app.inversion_models.values()) and "chunk" not in BD._KIT_ASIDE and "cueq_tiles" not in BD._KIT_ASIDE and [kw for nm, kw in calls if nm == "trimul"] == [{"variant": "cueq_tiles"}] + [{"variant": "fused"}] * 2


def test_install_reads_the_word_once_and_enable_installs_what_is_not_named(monkeypatch):
    calls, guards = [], []
    for name, mod in _kit_with(calls, guards).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook(); stock_prep = BD.prepare_esmfold2_tensors
    rec = FK.install(BD, modes.MODES["fast"], environ={FK.LEVERS_OFF_VAR: "ef2_loop_prep,ef2_esmc_hoist,ef2_stepgraph,chunk,featurisation_cache,critic_switches"})
    assert rec["levers_off"] == ["ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "cueq_tiles", "chunk", "ef2_stepgraph", "ef2_esmc_hoist", "featurisation_cache", "critic_switches"] == list(BD._LEVERS_OFF)
    assert BD.prepare_esmfold2_tensors is stock_prep and BD._FEATURE_CACHE is None and FK.ablated(BD, "chunk") and not FK.ablated(BD, "trimul")     # the featurisation cache not installed; the word readable by name
    monkeypatch.setenv(FK.LEVERS_OFF_VAR, "trimul")                                                  # read ONCE at install: a later environment change is not seen
    app = BD.ESMFold2Design(); app._enable_fast_kit(200)
    names = [n for n, _ in calls]
    for absent in ("loop_prep", "loop_pppl", "overlap", "stepgraph", "esmc_hoist"):
        assert absent not in names, (absent, names)
    assert [kw for n, kw in calls if n == "trimul"] == [{"variant": "fused"}] * 2 and names.count("esmc_graph") == 1 and names.count("pppl_graph") == 1 and names.count("rope") == 3   # no tile table (critic_switches chained it off); the rest as composed
    assert all(m.chunk_set == [None] for m in app.inversion_models.values())                        # `chunk` ablated on fast: the kit sets the pair stack unchunked
    assert [kw["copies"] for n, kw in calls if n == "bwd_ckpt"] == [1, 1]                            # no pool: one copy planned
    assert getattr(BD.fold_and_get_distogram, "_ef2_fold_guard", False)                             # the guard still rides the fold
    off = FK.install(_fake_cookbook(), "big", environ={FK.LEVERS_OFF_VAR: "ef2_bwd_ckpt"})
    assert off["levers_off"] == ["ef2_bwd_ckpt"]
    with pytest.raises(FK.LeversOffError):
        FK.install(_fake_cookbook(), "exact", environ={FK.LEVERS_OFF_VAR: "nope"})


def test_the_kit_line_names_the_ablated_levers(monkeypatch, caplog):
    calls, guards = [], []
    for name, mod in _kit_with(calls, guards).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook(); FK.install(BD, "exact", environ={FK.LEVERS_OFF_VAR: "ef2_esmc_hoist"})
    with caplog.at_level(logging.INFO, logger="fake_cookbook"):
        BD.ESMFold2Design()._enable_fast_kit(200)
    assert any(r.getMessage() == "fast kit: exact, checkpoint policy none for 200 tokens, trunk graph pool 2 slots, esmc hoist ablated, ablated ef2_esmc_hoist" for r in caplog.records), [r.getMessage() for r in caplog.records]
    BD2 = _fake_cookbook(); FK.install(BD2, "agk3", environ={FK.LEVERS_OFF_VAR: "ef2_stepgraph"}); caplog.clear()   # fast's composition with its pool ablated
    with caplog.at_level(logging.INFO, logger="fake_cookbook"):
        BD2.ESMFold2Design()._enable_fast_kit(200)
    assert any(r.getMessage() == "fast kit: agk3, checkpoint policy none for 200 tokens, trunk graph pool none (ablated), esmc hoist on, ablated ef2_stepgraph" for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert FK.pool_word("agk3", 200, True) == "none (ablated)" == FK.pool_word("agk3", 431, True) and FK.pool_word("big", 200, True) == "none"   # the ablation word switches the pool off by name on fast; big has none to switch
    with pytest.raises(FK.LeversOffError, match="does not compose"):
        FK.install(_fake_cookbook(), "big", environ={FK.LEVERS_OFF_VAR: "ef2_stepgraph"})              # big carries no trunk graph pool: the name is refused, not ignored
    BD3 = _fake_cookbook(); FK.install(BD3, "big"); caplog.clear()
    with caplog.at_level(logging.INFO, logger="fake_cookbook"):
        BD3.ESMFold2Design()._enable_fast_kit(200)
    assert any(r.getMessage() == "fast kit: big, checkpoint policy block for 200 tokens, trunk graph pool none, esmc hoist off, off by mode " + ",".join(FK.BIG_MODE_OFF) for r in caplog.records), [r.getMessage() for r in caplog.records]   # big's kit line at a small complex: the floor policy, no pool, the levers it switches off by name


def test_the_launcher_carries_the_word_into_kit_arms_only_and_words_it(monkeypatch, capsys):
    fast, off = modes.MODES["fast"], modes.MODES["off"]
    base = {"PATH": "/usr/bin", FK.LEVERS_OFF_VAR: " ef2_esmc_hoist ", "EF2INV_OTHER": "x"}
    env = launch.arm_env(fast, ["EF2_", "EF2INV_"], base=base)
    assert env[FK.LEVERS_OFF_VAR] == "ef2_esmc_hoist" and "EF2INV_OTHER" not in env and env["EF2_FAST_KIT"] == "agk3"   # fast's own switch word
    assert FK.LEVERS_OFF_VAR not in launch.arm_env(off, ["EF2_", "EF2INV_"], base=base) and FK.LEVERS_OFF_VAR not in launch.arm_env(fast, ["EF2_", "EF2INV_"], base={"PATH": "/usr/bin"})
    from .. import report as R
    assert R.active_line(fast, {"ablated": ["ef2_esmc_hoist"], "cache": "per-box"}).split(" line=")[0].endswith(" cache=per-box compile=stock ablated=ef2_esmc_hoist") and " ablated=" not in R.active_line(fast, {"cache": "per-box"})


def test_no_compile_is_accepted_and_worded_never_refused_and_switches_nothing():
    """`design --no-compile` == MODEL_OPT_LEVERS_OFF=compile (fastkit.COMPILE_LEVER): the opt-out of KIT-ADDED torch.compile levers. This kit adds none (stock's
    own compile is inherited untouched in every mode), so the name resolves on every arm — kit or stock, alone or beside real lever names — to nothing
    switched off plus the word compile=stock:not_kit_added; a plain run words compile=stock; the launcher's NOTE names both spellings."""
    import types
    from .. import cli, report as R
    for arm in ("exact", "agk3", "big", modes.MODES["off"]):
        assert FK.resolve_levers_off("compile", arm) == {"asked": [], "off": [], "chained": {}, "compile": "stock:not_kit_added"}
    r = FK.resolve_levers_off("compile,ef2_esmc_hoist", "agk3")
    assert r["off"] == ["ef2_esmc_hoist"] and r["compile"] == "stock:not_kit_added" and "compile" not in r["asked"]
    with pytest.raises(FK.LeversOffError):
        FK.resolve_levers_off("compile,ef2_no_such_lever", "agk3")                      # a typo beside it is still refused by name
    assert FK.compile_word() == "stock" and FK.compile_word(True) == "stock:not_kit_added" and FK.COMPILE_LEVER == "compile" and cli.NO_COMPILE_FLAG == "--no-compile"
    fast = modes.MODES["fast"]
    assert " compile=stock " in R.active_line(fast, {"cache": "per-box"}) and " compile=stock:not_kit_added " in R.active_line(fast, {"cache": "per-box", "compile": FK.compile_word(True)})
    lines = []
    orig = R.log; R.log = lines.append
    try:
        assert cli.compile_optout(types.SimpleNamespace(no_compile=True), "") == (True, "")
        assert cli.compile_optout(types.SimpleNamespace(no_compile=False), " compile , ef2_esmc_hoist ") == (True, "ef2_esmc_hoist")
        assert cli.compile_optout(types.SimpleNamespace(), "ef2_esmc_hoist") == (False, "ef2_esmc_hoist") and len(lines) == 2
    finally:
        R.log = orig
    assert all(l.startswith("NOTE --no-compile (MODEL_OPT_LEVERS_OFF=compile): compile=stock:not_kit_added") and l.endswith("; proceeding") for l in lines), lines
    ap_src = open(cli.__file__).read()
    assert ap_src.count("add_argument(NO_COMPILE_FLAG") == 2                                # design and warm accept it; check's flags are --mode/--json only


def test_the_design_verb_refuses_a_bad_word_before_any_launch_and_words_a_good_one(monkeypatch, tmp_path, capsys):
    """MODEL_OPT_LEVERS_OFF through the design verb (cli.launch_context): a typo or a lever the arm does not compose -> NOT ACTIVE by name, exit 3, no
    launch; a good word -> ONE NOTE naming the levers switched off, the ACTIVE line's ablated= word, the variable on the kit arm's environment; on
    --mode off a NOTE that the stock arm ignores it and the variable stripped."""
    from .. import cli
    from .test_model_switches import _design_rig, _notes
    ns, card, runs, writes = _design_rig(monkeypatch, tmp_path); card("NVIDIA H100 80GB HBM3", 81559)
    monkeypatch.setenv(FK.LEVERS_OFF_VAR, "esmc_hoist")
    assert cli.cmd_design(ns(mode="exact")) == 3 and runs == []
    err = capsys.readouterr().err
    assert "NOT ACTIVE reason=REFUSED MODEL_OPT_LEVERS_OFF: unknown lever name(s) esmc_hoist" in err, err
    monkeypatch.setenv(FK.LEVERS_OFF_VAR, "trimul")
    assert cli.cmd_design(ns(mode="exact")) == 3 and runs == [] and "EF2_FAST_KIT=exact does not compose trimul" in capsys.readouterr().err
    monkeypatch.setenv(FK.LEVERS_OFF_VAR, "ef2_loop_prep,cueq_tiles")
    assert cli.cmd_design(ns(mode="fast")) == 0 and len(runs) == 1
    err = capsys.readouterr().err
    notes = [ln for ln in err.splitlines() if ln.startswith("[ef2inv-opt] NOTE ")]
    assert len(notes) == 1 and notes[0].startswith("[ef2inv-opt] NOTE MODEL_OPT_LEVERS_OFF=ef2_loop_prep,cueq_tiles: levers ef2_loop_prep,ef2_loop_pppl,ef2_esmc_overlap,cueq_tiles switched off by name (ef2_loop_pppl serves ef2_loop_prep, ef2_esmc_overlap serves ef2_loop_pppl) — an ablation of --mode fast, not the mode"), notes   # invoked as `fast`: the mode is big
    active = [ln for ln in err.splitlines() if ln.startswith("[ef2inv-opt] ACTIVE ")]
    assert len(active) == 1 and " ablated=ef2_loop_prep,ef2_loop_pppl,ef2_esmc_overlap,cueq_tiles line=" in active[0] and active[0].startswith("[ef2inv-opt] ACTIVE mode=fast route=subprocess ") and " alias=" not in active[0], active
    assert runs[0][1][FK.LEVERS_OFF_VAR] == "ef2_loop_prep,cueq_tiles" and runs[0][1]["EF2_FAST_KIT"] == "agk3"
    runs.clear()
    assert cli.cmd_design(ns(mode="off")) == 0 and len(runs) == 1 and FK.LEVERS_OFF_VAR not in runs[0][1]
    err = capsys.readouterr().err
    assert any(ln.startswith("[ef2inv-opt] NOTE MODEL_OPT_LEVERS_OFF=ef2_loop_prep,cueq_tiles names kit levers and --mode off installs none") for ln in err.splitlines()) and " ablated=" not in err


def test_nosave_rows_asked_by_name_and_the_tier_word_printed():
    """k/ef2_trimul_nosave.py ships its candidate rows BY NAME (tx_sm90a > esm_v61 > esm_v5_fwd > v4) — not the provider's tier word, which its
    cell table resolves to v4 (N <= 800) on this kit's stack today — and prints what the tier word resolves to on the running stack next to the
    row that serves (`tier_fast=<row>` on the LEVER line, both states); the kit-enable step tells it the design's size."""
    import ast
    src = open(os.path.join(P.KDIR, "ef2_trimul_nosave.py"), encoding="utf-8").read()
    consts = {t.id: ast.literal_eval(node.value) for node in ast.parse(src).body if isinstance(node, ast.Assign) for t in node.targets
              if isinstance(t, ast.Name) and t.id in ("ROWS", "TIER_WORD", "ENGAGE_CC")}
    assert consts == {"ROWS": ("tx_sm90a", "esm_v61", "esm_v5_fwd", "v4"), "TIER_WORD": "fast", "ENGAGE_CC": ((9, 0),)}, consts
    assert 'tier = f"tier_fast={h.tier_fast or \'unasked\'}"' in src and "refused={refused} {tier}" in src and "cc=sm_{h.cc[0]}{h.cc[1]} {tier}" in src   # the token on the stepped_aside and the on line
    assert "def tier_row(" in src and "word=TIER_WORD, stack=PT.stack_word()" in src                                                              # the provider's pure select on the running stack, nothing bound
    assert 'ef2_trimul_nosave.enable(m, word=comp["nosave_fwd"], n_tokens=n_tokens)' in open(os.path.join(os.path.dirname(FK.__file__), "fastkit.py"), encoding="utf-8").read()
