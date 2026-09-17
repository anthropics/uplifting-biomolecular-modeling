"""stepgraph (opendde_opt/stepgraph.py): CPU tests of the control flow, refusals, RNG order and the interplay with the kit's dit_hoist on the
STOCK OpenDDE 1.1.1 sampler at tiny widths (tests/_tiny_sampler.py; skipped where upstream `opendde` is not importable). The CUDA graph itself
cannot run here: the CPU stand-in of the capture cache (tests/_eagercache.py, installed by monkeypatch) keeps the exact control flow (eager head, capture call, copy-in + replay on static buffers,
the poison probe) and re-runs the callable on the static copies, so equality with the stock sampler tests everything but the graph launch
(a CUDA run holds that). Plus the size gate's composition words, which need no upstream."""
import os
import sys

import pytest

from opendde_opt import modes, ran, registry, stepgraph

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------------------------- composition (no upstream needed)

@pytest.fixture(scope="module", autouse=True)
def _upstream_modules_leave_with_this_module():
    """The real upstream modules these tests import (`opendde.*`) are dropped from sys.modules when the module's tests are done: a later test's
    synthetic process must not find an imported-but-unwrapped upstream (the house levers' `fallbacks()` read sys.modules)."""
    import sys
    before = {k for k in sys.modules if k.split(".")[0] == "opendde"}
    yield
    for k in [k for k in sys.modules if k.split(".")[0] == "opendde" and k not in before]:
        del sys.modules[k]
    for k in [k for k in sys.modules if k.rsplit(".", 1)[-1] == "_tiny_sampler"]:
        del sys.modules[k]

@pytest.fixture
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("ODDE_STEPGRAPH_"):
            monkeypatch.delenv(k, raising=False)
    stepgraph._reset()
    yield
    stepgraph._reset()


def _toks(*counts):
    return {f"item{i}": {"residue_tokens": c, "ligands_uncounted": 0} for i, c in enumerate(counts)}


def test_the_lever_rides_exact_fast_and_resident_big_and_is_tabled(clean):
    assert "stepgraph" in modes.LINES["S1"].levers and "stepgraph" in modes.LINES["LSTAR2A"].levers
    assert "stepgraph" not in modes.LINES["BIG_TP"].levers                                 # --n_gpu P>1: never (modes.BIG_TP_DROP)
    assert "stepgraph" not in modes.LINES["BIG_F"].levers                                  # the static big row runs the offload unit: pair_offload turns the row off
    assert "stepgraph" in modes.BIG_KIT_ROWS_OFF["pair_offload"]
    lv = registry.LEVERS["stepgraph"]
    assert lv.tier == "exact" and lv.switch == "-" and lv.file == "opendde_opt/stepgraph.py" and lv.kit == registry.HOUSE
    assert registry.PIN_STATUS["stepgraph"][0] == "tested" and "stepgraph" in ran.COUNTERS and modes.LEVER_SWITCHES["stepgraph"] == ()
    for k in ("ODDE_STEPGRAPH_MAX_TOKENS", "ODDE_STEPGRAPH_MIN_TOKENS", "ODDE_STEPGRAPH_RTOL", "ODDE_STEPGRAPH_MEMFRAC"):
        assert k in registry.KNOBS, k
    assert not [k for k in registry.KNOBS if k in ("ODDE_STEPGRAPH_BACKEND", "ODDE_STEPGRAPH_POISON", "ODDE_STEPGRAPH_CAPTURE_MODE")]   # no test hook in the release tree


def test_gate_policy_and_words(clean, monkeypatch):
    s1 = modes.LINES["S1"]
    assert stepgraph.policy(None)["within"] is True and stepgraph.compose(s1) is s1 and stepgraph.gated_off(s1) == {}   # no query read: composed in
    monkeypatch.setenv("ODDE_STEPGRAPH_MAX_TOKENS", "800"); monkeypatch.setenv("ODDE_STEPGRAPH_MIN_TOKENS", "100")
    assert stepgraph.gate() == (100, 800)
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(400, 640))
    assert stepgraph.word() is None and stepgraph.compose(s1) is s1
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(400, 1200))
    out = stepgraph.compose(s1)
    assert "stepgraph" not in out.levers and stepgraph.word() == "above_gate:1200/800" and stepgraph.gated_off(out) == {"stepgraph": "above_gate:1200/800"}
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(60))
    assert stepgraph.word() == "below_gate:60/100" and "stepgraph" not in stepgraph.compose(s1).levers
    monkeypatch.setenv("ODDE_STEPGRAPH_MAX_TOKENS", "junk")
    assert stepgraph.gate() == (100, stepgraph.GATE_MAX)                                     # a malformed knob: the measured default


def test_an_offload_line_composes_the_lever_out_by_name(clean):
    resident = modes.big_line("BIG_F", ("sample_chunk",))                              # below the offload gate: fast's resident set carries the lever
    assert "stepgraph" in resident.levers and "stepgraph" not in modes.BIG_DROP             # (0.2.65: 0.2.46-0.2.64 dropped it from every big line by name and the resident line ran the eager step)
    assert "stepgraph" not in modes.LINES["BIG_F"].levers and "stepgraph" not in modes.LINES["BIG_TP"].levers   # off the offload line (BIG_KIT_ROWS_OFF) and the row-sharded line (BIG_TP_DROP)
    hacked = modes.Line(resident.name, resident.hook, resident.path_order, {**resident.exports, "ODDE_OFFLOAD": "all"}, resident.unset,
                        resident.levers, resident.tier, resident.note, allocator=resident.allocator)
    assert stepgraph.word(hacked) == "offload_line" and "stepgraph" not in stepgraph.compose(hacked).levers


def test_resolution_words(clean, monkeypatch):
    from opendde_opt import stack
    from opendde_opt.tests.test_activation import fresh  # noqa: F401
    monkeypatch.setenv("ODDE_STEPGRAPH_MAX_TOKENS", "500")
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(612))
    res = modes.resolve("exact", "/no/tree", {})
    assert "stepgraph" not in res.line.levers and "chunk_lift" not in res.line.levers          # (chunk_lift: no ceiling plan in this process)


def test_fallback_words(clean):
    assert stepgraph.fallbacks(["chunk_lift"]) == []
    assert stepgraph.fallbacks(["stepgraph"]) == []                                            # nothing happened: no event (ran-or-refuse names a zero counter)
    stepgraph.STATS["refused"]["inplace_safe_false_mutates_cached_p_lm"] = 2                   # a by-design step-aside: counted, never a fallback
    assert stepgraph.fallbacks(["stepgraph"]) == [] and stepgraph.asides() == {"inplace_safe_false_mutates_cached_p_lm": 2}
    stepgraph.STATS["refused"]["layernorm_not_capture_safe"] = 1                               # a failure word: a fallback by name
    assert stepgraph.fallbacks(["stepgraph"]) == ["stepgraph:refused:layernorm_not_capture_safe(x1)"]
    stepgraph.STATS["disabled"] = "capture_failed:x"
    assert stepgraph.fallbacks(["stepgraph"])[0] == "stepgraph:disabled:capture_failed:x"


# ----------------------------------------------------------------------------------------------- the stock sampler at tiny widths (real upstream)
def _upstream():
    torch = pytest.importorskip("torch")
    pytest.importorskip("opendde.model.modules.diffusion")
    sys.path.insert(0, HERE)
    import _tiny_sampler
    torch.set_num_threads(4)
    return torch, _tiny_sampler


@pytest.fixture
def sg(clean, monkeypatch):
    torch, tiny = _upstream()
    from opendde_opt.tests import _eagercache as EC                                            # the CPU stand-in of the core's GraphCache (test support)
    monkeypatch.setattr(stepgraph, "_make_cache", lambda det: EC.EagerTestCache(stepgraph.LEVER, warmup=1))
    monkeypatch.setattr(stepgraph, "_device_refusal", lambda x: None)
    import opendde.model.opendde as OM
    orig = OM.sample_diffusion
    inner = getattr(orig, "_orig", None)
    if inner is not None:                                                                      # a previous test's wrapper: unwrap first
        orig = inner
    OM.sample_diffusion = stepgraph.make_wrapper(orig)
    yield torch, tiny
    OM.sample_diffusion = orig
    stepgraph.remove()


def test_bitwise_vs_stock_4steps_2samples(sg):
    torch, tiny = sg
    import opendde.model.opendde as OM
    b = tiny.build()
    wrapped = OM.sample_diffusion; OM.sample_diffusion = wrapped._orig
    ref = tiny.run_sampler(b, n_step=4, n_sample=2)
    OM.sample_diffusion = wrapped
    out = tiny.run_sampler(b, n_step=4, n_sample=2)
    c = stepgraph.kit_stats()
    assert torch.equal(out, ref), float((out - ref).abs().max())
    # N_step=4 -> 4 denoiser calls: 1 eager head, 1 capture (eager-first answer), 2 replays (+1 replay-equivalent for the poison probe on the stand-in: not counted)
    assert c["denoiser_calls"] == 4 and c["eager_head"] == 1 and c["captures"] == 1 and c["replays"] == 2 and c["recaptures"] == 0, c
    assert c["tol_checks"] == 1 and c["tol_max_rel"] == 0.0 and not c["refused"] and c["poison_probe"] == "pass", c
    assert ran.count("stepgraph") == 2
    out2 = tiny.run_sampler(b, n_step=4, n_sample=2)                                           # a second sampler call re-captures (per-item graphs)
    c2 = stepgraph.kit_stats()
    assert torch.equal(out2, ref) and c2["captures"] == 2 and c2["recaptures"] == 1 and c2["replays"] == 4, c2


def test_sample_chunks_share_one_capture_and_ragged_tail_gets_its_own(sg):
    torch, tiny = sg
    import opendde.model.opendde as OM
    b = tiny.build(seed=5)
    wrapped = OM.sample_diffusion; OM.sample_diffusion = wrapped._orig
    ref = tiny.run_sampler(b, n_step=5, n_sample=3, chunk=2)                                  # chunks of 2 + 1
    OM.sample_diffusion = wrapped
    out = tiny.run_sampler(b, n_step=5, n_sample=3, chunk=2)
    c = stepgraph.kit_stats()
    assert torch.equal(out, ref)
    assert c["denoiser_calls"] == 10 and c["eager_head"] == 1 and c["captures"] == 2, c     # 2 chunks x 5 steps; the ragged tail is a new key
    assert c["replays"] == 10 - 1 - 2, c


def test_refuses_by_name_and_stays_stock(sg, monkeypatch):
    torch, tiny = sg
    import opendde.model.opendde as OM
    b = tiny.build()
    wrapped = OM.sample_diffusion; OM.sample_diffusion = wrapped._orig
    ref = tiny.run_sampler(b)
    OM.sample_diffusion = wrapped
    for var, val, word in (("ODDE_OFFLOAD", "all", "offload_unit_active"), ("DIT_HOIST_CROSSCHECK", "3", "dit_hoist_crosscheck_set")):
        stepgraph._reset(); monkeypatch.setenv(var, val)
        out = tiny.run_sampler(b)
        c = stepgraph.kit_stats()
        assert torch.equal(out, ref) and c["replays"] == 0 and word in c["aside"] and not c["refused"], c   # a by-design step-aside: counted, not a fallback
        assert stepgraph.fallbacks(["stepgraph"]) == [] and stepgraph.aside_reason() == f"aside:{word}"   # by design: inert (LEVER state=off reason=aside:<word>), not a fallback
        monkeypatch.delenv(var)
    stepgraph._reset()
    tiny.run_sampler(b, inplace_safe=False)                                                   # upstream MUTATES the cached p_lm on this path (b is spent after this): refuse
    c = stepgraph.kit_stats()
    assert "inplace_safe_false_mutates_cached_p_lm" in c["aside"] and c["replays"] == 0, c


def test_grad_mode_stands_aside(sg):
    torch, tiny = sg
    b = tiny.build()
    import opendde.model.opendde as OM
    OM.sample_diffusion(**tiny.sampler_kwargs(b, inplace_safe=True))                          # autograd ON (no no_grad scope)
    c = stepgraph.kit_stats()
    assert c["replays"] == 0 and "autograd_on" in c["aside"], c


def test_with_the_kits_dit_hoist_bitwise(sg, monkeypatch):
    """The shipped configuration: dit_hoist (record at call 1, hit afterwards) + stepgraph on top; equality with the stock sampler."""
    torch, tiny = sg
    import opendde.model.opendde as OM
    p = tiny.kit_paths()
    monkeypatch.syspath_prepend(p["ditfast"])
    try:
        import odde_addon
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"kit DITFAST unit not importable here: {e!r}")
    b = tiny.build(seed=11)
    wrapped = OM.sample_diffusion; OM.sample_diffusion = wrapped._orig
    ref = tiny.run_sampler(b, n_step=6, n_sample=2)
    OM.sample_diffusion = wrapped                                                              # the hoist wraps OUR wrapper (runner creation comes after activation)
    odde_addon.install(b["model"], ["dit_hoist", "dit_align"])
    try:
        stepgraph._reset()
        out = tiny.run_sampler(b, n_step=6, n_sample=2)
        c = stepgraph.kit_stats()
        assert torch.equal(out, ref), float((out - ref).abs().max())
        assert c["eager_head"] == 1 and c["captures"] == 1 and c["replays"] == 4 and c["poison_probe"] == "pass", c
        assert odde_addon.STATS["records"] > 0 and odde_addon.STATS["hits"] > 0
    finally:
        odde_addon.remove(b["model"])


def test_segment_sum_is_bitwise_the_deterministic_scatter_add(sg):
    """segment_sum (the capturable atom->token sum served under deterministic algorithms while capturing) = torch's scatter_add_ result bit for
    bit: fp32 and bf16 sources, several widths / leading dims / token counts (the GPU test holds the same on CUDA against the deterministic kernel)."""
    torch, tiny = sg
    import itertools
    g = torch.Generator().manual_seed(0)
    for dtype, d, lead, n_token, shuffled in itertools.product((torch.float32, torch.bfloat16), (8, 128), ((3,), (1, 2), ()), (12, 40), (False, True)):
        counts = torch.randint(1, 9, (n_token,), generator=g)
        idx = torch.repeat_interleave(torch.arange(n_token), counts)
        if shuffled:                                                                           # atoms of a token need not be contiguous (ligand / modified-residue orders)
            idx = idx[torch.randperm(idx.numel(), generator=g)]
        seg = stepgraph.segment_table(idx, n_token)
        assert seg is not None
        src = (torch.randn(*lead, idx.numel(), d, generator=g) * 3).to(dtype)
        index = idx.view(*([1] * len(lead)), -1, 1).expand_as(src)
        ref = torch.zeros(*lead, n_token, d, dtype=dtype).scatter_add_(-2, index, src)
        out = stepgraph.segment_sum(src, seg[0], seg[1], seg[2])
        assert torch.equal(ref, out), (dtype, d, lead, n_token)
    assert stepgraph.segment_table(torch.tensor([0, 0, 3]), 3) is None                      # an index outside the token range: no table (the call refuses by name)


def test_deterministic_recipe_routes_the_atom_sum_and_stays_bitwise(sg):
    """Under torch's deterministic algorithms the 'captured' step (eagertest backend) serves the atom->token sums by segment_sum: counted
    (segsum_calls > 0), the roll-out bitwise the stock sampler's, the stock scatter_sum back in place after remove()."""
    torch, tiny = sg
    import opendde.model.opendde as OM
    import opendde.utils.scatter_utils as SU
    b = tiny.build()
    wrapped = OM.sample_diffusion; OM.sample_diffusion = wrapped._orig
    ref = tiny.run_sampler(b, n_step=4, n_sample=2)
    OM.sample_diffusion = wrapped
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        out = tiny.run_sampler(b, n_step=4, n_sample=2)
        st = stepgraph.kit_stats()
        assert st["det"] is True and st["segsum_calls"] > 0 and st["captures"] == 1 and st["replays"] == 2 and not st["refused"], st
        assert SU.scatter_sum is stepgraph._scatter_sum_routed                                 # routed (from the first admission under the recipe)
        assert torch.equal(out, ref)
    finally:
        torch.use_deterministic_algorithms(prev)
    stepgraph.remove()
    assert SU.scatter_sum is not stepgraph._scatter_sum_routed                                 # the stock function back


def test_by_design_step_asides_are_inert_not_fallbacks(clean):
    """A sampler call outside the lever's domain (guidance, another rank, the memory admission, ...) runs the STOCK sampler by design: counted by
    word, never a fallback; when no call of the process engaged, the lever is inert by design (LEVER state=off reason=aside:<word>), the run
    complete. A capture that failed is a fallback by name (PARTIAL)."""
    from opendde_opt import report
    stepgraph._refuse("guidance_enabled"); stepgraph._refuse("guidance_enabled"); stepgraph._refuse("memory_est_1MiB_over_0.5_of_free_2MiB")
    assert stepgraph.fallbacks(("stepgraph",)) == []
    assert stepgraph.asides() == {"guidance_enabled": 2, "memory_est_1MiB_over_0.5_of_free_2MiB": 1}
    assert stepgraph.aside_reason() == "aside:guidance_enabled+1_more"
    rep = {"active": True, "levers_planned": ["stepgraph"], "levers": ["stepgraph"], "levers_applied": [], "levers_inert": ["stepgraph"],
           "levers_inert_reasons": {"stepgraph": stepgraph.aside_reason()}}
    assert report.lever_state("stepgraph", rep) == ("off", "aside:guidance_enabled+1_more")
    STATS = stepgraph.STATS
    STATS["replays"] = 5                                                                       # some call engaged: not inert (the census carries aside=)
    assert stepgraph.aside_reason() is None and stepgraph.kit_stats()["aside"]["guidance_enabled"] == 2
    STATS["replays"] = 0
    stepgraph._refuse("capture_failed_or_not_bit_exact")                                     # a failure: a fallback by name, and no longer inert
    assert stepgraph.fallbacks(("stepgraph",)) == ["stepgraph:refused:capture_failed_or_not_bit_exact(x1)"] and stepgraph.aside_reason() is None
    assert stepgraph.kit_stats()["refused"] == {"capture_failed_or_not_bit_exact": 1}


def test_leaving_lnstream_out_sheds_stepgraph_by_name(clean):
    """MODEL_OPT_LEVERS_OFF=lnstream: the step graph goes with it (modes.LEVER_DEPENDENTS) — named as ablated, never a refusal at run time."""
    from opendde_opt import ablate
    s1 = modes.LINES["S1"]
    out = ablate.compose(s1, {ablate.ENV: "lnstream"})
    assert "lnstream" not in out.levers and "stepgraph" not in out.levers
    assert ablate.dropped() == ("lnstream", "stepgraph")
    ablate._reset()


def test_the_sm80_row_caps_the_step_graph_at_1024_tokens_by_name(clean, monkeypatch):
    """CARD_ROWS (data): on compute capability 8.0 the per-step graph is composed OUT of the fast / exact line above 1,024 residue tokens by name --
    LEVER state=off reason=card_gate:sm80_replay_not_bit_exact_above:1024 -- and IN at or under it (the capture holds bit-exact and serves at 800 on that
    card); sm_90 and a card without a row keep the general gate only (no cap)."""
    from opendde_opt import report, stack
    fast = modes.LINES["LSTAR2A"]
    assert "stepgraph" in fast.levers
    monkeypatch.setattr(stepgraph, "_card_sm", lambda: "80")
    assert stepgraph.card_row() == (1024, "sm80_replay_not_bit_exact", "80")
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(800, 780))                                # A100 @800: in
    assert stepgraph._PLAN["plan"]["within"] is True and stepgraph.word() is None and stepgraph.compose(fast) is fast
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(1400))                                    # A100 @1400: out by the card's name
    assert stepgraph._PLAN["plan"]["case"] == "card_above", stepgraph._PLAN["plan"]
    assert stepgraph.word() == "card_gate:sm80_replay_not_bit_exact_above:1024"
    out = stepgraph.compose(fast)
    assert "stepgraph" not in out.levers and stepgraph.gated_off(out) == {"stepgraph": "card_gate:sm80_replay_not_bit_exact_above:1024"}
    rep = {"active": True, "mode": "fast", "levers_planned": list(out.levers), "levers": list(out.levers), "levers_applied": [], "levers_gated_off": stepgraph.gated_off(out)}
    assert report.lever_state("stepgraph", rep) == ("off", "card_gate:sm80_replay_not_bit_exact_above:1024")
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(1024))                                    # the row's edge: in
    assert stepgraph.word() is None
    res = modes.resolve("fast", "/no/tree", {})                                                 # the composed fast line on this card at 1400: without the lever
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(1400))
    res = modes.resolve("fast", "/no/tree", {})
    assert "stepgraph" not in res.line.levers
    monkeypatch.setattr(stepgraph, "_card_sm", lambda: "90")                                    # H100 @1400: in (no row)
    assert stepgraph.card_row() is None
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(1400))
    assert stepgraph.word() is None and stepgraph.compose(fast) is fast and "stepgraph" in modes.resolve("fast", "/no/tree", {}).line.levers
    monkeypatch.setattr(stepgraph, "_card_sm", lambda: None)                                    # no GPU visible (a dry run off the box): in
    stepgraph._PLAN["plan"] = stepgraph.policy(_toks(1400))
    assert stepgraph.word() is None
    monkeypatch.undo(); stepgraph._reset()
    gi = stack.gpu_info()                                                                      # the reader itself: this CPU process has no card -> no row
    assert stepgraph.card_row() is None or gi.get("sm") in stepgraph.CARD_ROWS


STAGE1_MSG = (f"refused ({stepgraph.CAPTURE_CHECK_KIND}): replay differs from eager on the capture inputs ({{'max_abs_diff': 0.0217, 'held_max_abs_diff': None, 'stage': 'verify1'}}) "
              "— a kernel in the callable is not capture-safe or reads memory produced outside the graph")   # the core GraphCache's disable reason at its first consistency stage


def test_a_capture_check_mismatch_is_a_named_step_aside_to_the_eager_sampler(clean):
    """The core cache's consistency replay differed from the eager step (stages verify1 / verify2) BEFORE any graph output was consumed -- the
    cache answered that call eagerly and disabled itself: the lever accounts it as the step-aside verify_mismatch_<stage>_maxabs_<diff> (ASIDE
    word, counted in refused), never a failure: fallbacks() empty, aside_reason() names it, the kit's fold holds the lever INERT (LEVER state=off
    reason=aside:verify_mismatch_verify1_maxabs_2.170e-02), PRED partial=False, rc 0."""
    from opendde_opt import report
    word = stepgraph.note_cache_disabled(STAGE1_MSG)
    assert word == "verify_mismatch_verify1_maxabs_2.170e-02", word
    S = stepgraph.STATS
    assert S["verify_aside"] == word and S["refused"] == {word: 1} and S["failures"] == 0 and S["disabled"] is None
    assert stepgraph.fallbacks(("stepgraph",)) == [] and stepgraph.asides() == {word: 1}
    assert stepgraph.aside_reason() == f"aside:{word}"
    stepgraph._refuse("guidance_enabled"); stepgraph._refuse("guidance_enabled")                # other by-design asides do not displace THE reason
    assert stepgraph.aside_reason() == f"aside:{word}"
    rep = {"active": True, "levers_planned": ["stepgraph"], "levers": ["stepgraph"], "levers_applied": [], "levers_inert": ["stepgraph"],
           "levers_inert_reasons": {"stepgraph": stepgraph.aside_reason()}}
    assert report.lever_state("stepgraph", rep) == ("off", f"aside:{word}")
    ev = report.lever_evidence("stepgraph", {"stepgraph": stepgraph.kit_stats()})
    assert ev.get("disabled") is None and word in (ev.get("aside") or ""), ev
    S["replays"] = 199                                                                         # an earlier item of the process replayed fine, a later one mismatched: the lever
    assert stepgraph.aside_reason() is None and stepgraph.fallbacks(("stepgraph",)) == []     # served (state=on, census aside=…), still no fallback
    # a verify2-stage message and one without a parsable detail keep the class
    stepgraph._reset()
    assert stepgraph.note_cache_disabled(STAGE1_MSG.replace("verify1", "verify2")) == "verify_mismatch_verify2_maxabs_2.170e-02"
    stepgraph._reset()
    assert stepgraph.note_cache_disabled(f"refused ({stepgraph.CAPTURE_CHECK_KIND}): replay differs from eager") == "verify_mismatch_capture_maxabs_unknown"
    assert stepgraph.fallbacks(("stepgraph",)) == []


def test_a_failure_after_a_replayed_graph_was_used_stays_partial(clean):
    """PARTIAL semantics kept for what is not a capture-check step-aside: a capture that raised (the cache's `capture failed: …`), a failed poison
    probe or non-finite coordinates after a replay -- STATS['disabled'] set, fallbacks() names it, the run PARTIAL as before."""
    word = stepgraph.note_cache_disabled("capture failed: CUDA error: an illegal memory access was encountered; recovery={'healthy': True}")
    assert word == "capture_failed_or_not_bit_exact"
    assert stepgraph.STATS["disabled"].startswith("capture_failed:capture failed: CUDA error") and stepgraph.STATS["failures"] == 1
    assert stepgraph.fallbacks(("stepgraph",))[0].startswith("stepgraph:disabled:capture_failed:") and stepgraph.aside_reason() is None
    stepgraph._reset()
    stepgraph.STATS["disabled"] = "poison_probe_failed"; stepgraph._refuse("poison_probe_failed"); stepgraph.STATS["replays"] = 3
    assert stepgraph.fallbacks(("stepgraph",)) == ["stepgraph:disabled:poison_probe_failed", "stepgraph:refused:poison_probe_failed(x1)"] and stepgraph.aside_reason() is None
    stepgraph._reset()
    stepgraph.STATS["disabled"] = "nonfinite_coordinates_after_replay"
    assert stepgraph.fallbacks(("stepgraph",)) == ["stepgraph:disabled:nonfinite_coordinates_after_replay"]
