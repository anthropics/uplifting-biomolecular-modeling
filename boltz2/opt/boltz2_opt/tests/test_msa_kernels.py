"""boltz2_opt.msa_kernels — the fused MSA-module kernels (registry fpf_opm / fpf_pwa): the row's words and refusals, the carried bundle's pins,
the registry / mode-table / attach wiring, the evidence line grammar and the fail-closed gate on a stand-in ledger. CPU only, no torch: the
kernels are Triton GPU kernels — their numerics are the GPU legs' (the seed-spread band of `fast`), never a CPU test's."""
import os

import pytest

from .. import modes, msa_kernels as MK, registry, stack
from ..worker_launch import ATTACH


def test_units_and_levers_follow_the_switch():
    assert MK.units({}) == [] and not MK.requested({}) and MK.problems({}) == []
    assert MK.units({"BOLTZ_FPF_MSA": "opm,pwa"}) == ["opm", "pwa"] and MK.levers_of({"BOLTZ_FPF_MSA": "pwa,opm,opm"}) == ["fpf_pwa", "fpf_opm"]
    with pytest.raises(ValueError, match="'xyz' is not a unit"):
        MK.units({"BOLTZ_FPF_MSA": "opm,xyz"})
    assert MK.problems({"BOLTZ_FPF_MSA": "bogus"}) and "not a unit" in MK.problems({"BOLTZ_FPF_MSA": "bogus"})[0]
    assert set(MK.LEVERS) == {"fpf_opm", "fpf_pwa"} and MK.EXPECTED == ()


def test_conflicting_owners_are_refused_by_name():
    row = {"BOLTZ_FPF_MSA": "opm,pwa"}
    assert MK.conflicting(row) == [] and MK.problems(row) == [], MK.problems(row)
    c = MK.conflicting({**row, "BOLTZ_FPF_STACK": "tier2"})
    assert len(c) == 1 and "BOLTZ_FPF_STACK=tier2" in c[0]
    assert MK.conflicting({**row, "BOLTZ_FPF_STACK": "off"}) == [] and MK.conflicting({**row, "BOLTZ_FPF_STACK": "0"}) == []
    with pytest.raises(RuntimeError, match="conflicting levers"):
        os.environ["BOLTZ_FPF_STACK"] = "tier2"
        try:
            MK.apply("opm")
        finally:
            del os.environ["BOLTZ_FPF_STACK"]
    assert MK.report()["applied"] == [] and MK.apply("") == []


def test_the_carried_bundle_is_present():
    """Every file the Boltz-2 entry points import is a carried file of this tree, present on disk; the adapter imports nothing else of the bundle."""
    sums = stack.read_sums()
    for rel in MK.PACKAGE_FILES:
        assert rel in sums, rel
        assert os.path.isfile(stack.kit_path(rel)), rel
    assert MK.carried_problems() == []
    assert MK.PACKAGE_FILES[0].startswith(MK.BUNDLE + "/fpf_msa/") and all("/fpf_msa/" in r for r in MK.PACKAGE_FILES)
    assert registry.LEVER_IDS["fpf_opm"]["impl"].startswith(MK.CORE_OPM) and registry.LEVER_IDS["fpf_pwa"]["impl"].startswith(MK.CORE_PWA)   # the catalogue names the core cells the entry points bind


def test_registry_modes_and_attach_wiring():
    for name in MK.LEVERS:
        L = registry.LEVERS[name]
        assert L["switch"] == MK.SWITCH == "BOLTZ_FPF_MSA" and L["tier"] == 2 and L["file"] == "boltz2_opt/msa_kernels.py" and L["origin"] == "core"   # the kernels are the shared core's cells; the adapter is the kit's
    assert ATTACH["msa"] == {"module": "boltz2_opt.msa_kernels", "trigger": "boltz.model.models.boltz2", "report_key": "msa_report"}
    for m, units in (("fast", "opm,pwa"), ("big", "opm")):           # big tables both units as fast does; the fused PWA is the memory row's allocator peak and leaves it BY NAME (MODES["big"]["off"], CHANGES 0.3.3)
        assert modes.env_row(m)["BOLTZ_FPF_MSA"] == units and "msa" in modes.resolve(m)["attach"] and {"fpf_opm", "fpf_pwa"} <= set(modes.MODES[m]["levers"]) and "fpf_opm" in modes.levers(m), m
    assert "fpf_pwa" not in modes.levers("big") and modes.resolve("big")["off"]["fpf_pwa"].startswith("measured_memory")
    assert modes.attachments("big").index("msa") < modes.attachments("big").index("xl"), "the memory adapter lands over the MSA kernels (disjoint attribute sets)"
    assert "BOLTZ_FPF_MSA" not in modes.env_row("exact") and "msa" not in modes.resolve("exact")["attach"] and not ({"fpf_opm", "fpf_pwa"} & set(modes.levers("exact"))), "Tier 2: never on exact"
    assert stack.attachment_problems("big") == [], stack.attachment_problems("big")
    assert registry.tier_of(["fpf_opm"]) == 2, "Tier 2 by construction: never on the exact row"
    assert stack.attachment_problems("fast") == [], stack.attachment_problems("fast")


def test_line_grammar_and_gate_on_a_stand_in_ledger(monkeypatch):
    """The LEVER line before apply (state=off, the reason) and the report's fail-closed gate over the core ledger: served-only passes; a
    fallback word (EXPECTED is empty) or a kernel error refuses; the report carries the configuration facts beside the census."""
    from opt_core.counters import Ledger
    assert MK.line("opm").startswith("[boltz2-opt] LEVER name=fpf_opm state=off") and "not_requested" in MK.line("opm")
    led = Ledger("fpf_opm", impl="fpf_msa.boltz2:outer_product_mean", origin="kit", min_tokens=None, expected=MK.EXPECTED)
    led.set("cfg", "f1"); led.set("tma", True)
    monkeypatch.setitem(MK._STATE, "ledgers", {"opm": led}); monkeypatch.setitem(MK._STATE, "applied", ["fpf_opm"]); monkeypatch.setitem(MK._STATE, "units", ["opm"])
    monkeypatch.setitem(MK._STATE, "facts", {"opm": {"cfg": "f1", "tma": True}}); monkeypatch.setitem(MK._STATE, "patched", ["OuterProductMean.forward"])
    for _ in range(48):
        led.serve("1x512x400x64/bfloat16")
    r = MK.report()
    assert r["applied"] == ["fpf_opm"] and r["variant"] == "opm" and r["gate"]["ok"] and r["units"]["opm"]["census"]["served"] == 48 and r["units"]["opm"]["cfg"] == "f1"
    assert "state=on" in r["units"]["opm"]["line"] and "gate=ok" in r["units"]["opm"]["line"]
    led.fallback("dims_not_pinned")
    r = MK.report()
    assert not r["gate"]["ok"] and "unexpected fallback" in r["gate"]["reason"] and "gate=refused" in r["units"]["opm"]["line"]
    led.clear(); led.serve("x"); led.error(RuntimeError("triton compile"))
    assert not MK.report()["gate"]["ok"] and "kernel error" in MK.report()["gate"]["reason"]


def test_n_gpu_above_1_names_the_disposition_instead_of_installing(monkeypatch):
    """At n_gpu > 1 (BOLTZ_TP=<P>) the row-sharded trunk owns both computations (rowpair.REPLACED_LEVERS): apply() installs nothing, the report
    names replaced_by_rowpair per unit with the statement, the gate passes by that name, the LEVER line carries state and execution words, and
    stack.evidence requires exactly that word (an installed unit there is a problem)."""
    monkeypatch.setitem(modes.MODES["big"], "off", {})               # the evidence below reads a two-unit report against the memory row: table both units on it for the test (the shipped row names fpf_pwa off, CHANGES 0.3.3)
    from ..rowpair import ENV_P, REPLACED_LEVERS
    assert {"fpf_opm", "fpf_pwa"} <= set(REPLACED_LEVERS) and "_opm_add_" in REPLACED_LEVERS["fpf_opm"] and "_pwa_update" in REPLACED_LEVERS["fpf_pwa"]
    MK.undo()
    monkeypatch.setenv("BOLTZ_FPF_MSA", "opm,pwa"); monkeypatch.setenv(ENV_P, "2")
    assert MK.worker_n_gpu() == 2 and MK.apply() == []                  # nothing imported, nothing patched
    assert MK.dispositions() == {"fpf_opm": MK.REPLACED, "fpf_pwa": MK.REPLACED}, "the launcher's attach hook accepts the empty apply() against this word (worker_launch._attach)"
    r = MK.report()
    assert r["applied"] == [] and r["patched"] == [] and r["n_gpu"] == 2 and r["variant"] == "opm,pwa" and r["gate"]["ok"] and r["gate"]["reason"] == MK.REPLACED
    assert r["execution"] == {"opm": MK.REPLACED, "pwa": MK.REPLACED} and r["units"]["opm"]["statement"] == REPLACED_LEVERS["fpf_opm"]
    for u, lever in (("opm", "fpf_opm"), ("pwa", "fpf_pwa")):
        ln = r["units"][u]["line"]
        assert ln.startswith(f"[boltz2-opt] LEVER name={lever} state=skipped reason={MK.REPLACED}") and f"execution={MK.REPLACED}" in ln and "n_gpu=2" in ln and "statement=" in ln, ln
    log = {"env": {"boltz_levers": ["resid", "mask2"], "f2_patch_active": True}, "per_item": [{"name": "a", "seed": 1}], "events": [], "msa_report": r}
    ev, problems = stack.evidence("big", log, n_gpu=2)
    assert not any("MSA" in p for p in problems), problems
    assert ev["msa_execution"] == {"opm": MK.REPLACED, "pwa": MK.REPLACED}
    bad = dict(r, execution={"opm": MK.REPLACED, "pwa": "executed:96"}, applied=["fpf_pwa"])
    ev, problems = stack.evidence("big", dict(log, msa_report=bad), n_gpu=2)
    assert any("units pwa not reported replaced_by_rowpair" in p for p in problems) and any("installed at n_gpu=2" in p for p in problems), problems
    from .. import report as REP
    st_, why, pairs = REP.lever_state("fpf_opm", {"msa_report": r}, {"n_gpu": 2})
    assert st_ == "skipped" and why == MK.REPLACED and ("execution", MK.REPLACED) in pairs
    MK.undo()
    assert MK.dispositions() == {}
    monkeypatch.delenv(ENV_P)
    assert MK.worker_n_gpu() == 1


def test_execution_word_at_n_gpu_1_is_the_call_count(monkeypatch):
    from opt_core.counters import Ledger
    MK.undo()
    led = Ledger("fpf_pwa", impl="fpf_msa.boltz2:msa_pair_weighted_avg", origin="kit", min_tokens=None, expected=MK.EXPECTED)
    monkeypatch.setitem(MK._STATE, "ledgers", {"pwa": led}); monkeypatch.setitem(MK._STATE, "applied", ["fpf_pwa"]); monkeypatch.setitem(MK._STATE, "units", ["pwa"])
    monkeypatch.setitem(MK._STATE, "execution", {"pwa": "executed"}); monkeypatch.setitem(MK._STATE, "facts", {"pwa": {"cfg": "default"}})
    for _ in range(96):
        led.serve("s")
    r = MK.report()
    assert r["execution"] == {"pwa": "executed:96"} and "execution=executed:96" in r["units"]["pwa"]["line"] and r["n_gpu"] == 1


class _FakeT:                                                         # a stand-in for the MSA input m [B, S, N, c_m]: the predicate reads shapes only
    def __init__(self, *shape): self.shape = tuple(shape); self.dtype = "float32"
    def dim(self): return len(self.shape)


class _FakePWA:                                                       # boltz PairWeightedAveraging's pinned dims: 8 heads x c_h 32 -> HC = 256
    num_heads, c_h = 8, 32


def _unit_ledger(unit="pwa"):
    from opt_core.counters import Ledger
    U = MK.UNITS[unit]
    return Ledger(U["lever"], impl=f"fpf_msa.boltz2:{U['entry']}", origin="kit", min_tokens=None, expected=MK.EXPECTED)


def test_int32_predicate_words_and_the_reach_shape():
    """unit pwa's launch bound: S * N * HC (the kernels' v [S, N, H*C]) < 2**31 (msa_kernels.INT32_LIMIT); the predicate names a call at/over
    the bound BEFORE launch; opm carries no bound."""
    assert MK.INT32_LIMIT == 35_000_000 * 256 + 1, "the tested launch envelope on S*N*HC (int64 offsets in the kernels): 8.96e9"
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 8192, 3762, 64),), {}) is None, "the reach box's shape (7.9e9) is inside the envelope: launched"
    BIG = 8192 * 4375 * 256
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 8192, 4375, 64),), {}) == f"int32_limit:S·N·HC={BIG}>=2^31"          # past the envelope: skipped by name before launch
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 4104, 400, 64),), {}) is None                                       # the ladder's 400-token shape: launched
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 8192, 1024, 64),), {}) is None and MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 8191, 1024, 64),), {}) is None
    assert MK.precall_disposition("opm", _FakePWA(), (_FakeT(1, 8192, 3762, 64),), {}) is None, "opm.py carries no int32 limit: never skipped by this name"
    assert MK.precall_disposition("pwa", _FakePWA(), (), {"m": _FakeT(1, 8192, 4375, 64)}) is not None, "keyword call form"


def test_over_the_limit_the_kernel_is_not_called_and_the_skip_is_named(capsys):
    led = _unit_ledger(); calls = {"entry": 0, "orig": 0}

    def entry(module, *a, **k): calls["entry"] += 1; return "kernel"

    def orig(): calls["orig"] += 1; return "stock"

    big = (_FakeT(1, 8192, 4375, 64), "z", "mask")
    assert MK.dispatch("pwa", led, entry, RuntimeError, _FakePWA(), big, {}, orig) == "stock"
    assert MK.dispatch("pwa", led, entry, RuntimeError, _FakePWA(), big, {}, orig) == "stock"
    assert calls == {"entry": 0, "orig": 2}, "decided before launch: the kernel entry is never called"
    assert MK.skipped(led) == {"int32_limit": 2} and led.get("skipped_last") == f"int32_limit:S·N·HC={8192 * 4375 * 256}>=2^31"
    f = led.fields()
    assert f["served"] == 0 and f["fallback"] == 0 and not f["errors"], "a named skip is neither a fallback nor an error"
    assert MK.verdict(led)["ok"], "skipped_by_name alone never refuses the gate"
    assert MK.execution_word(led) == "executed:0,skipped_by_name:int32_limit:2"
    err = capsys.readouterr().err
    assert err.count("skipped_by_name") == 2 and f"int32_limit:S·N·HC={8192 * 4375 * 256}" in err and "1x8192x4375x64" in err


def test_under_the_limit_the_kernel_serves_and_nothing_is_skipped():
    led = _unit_ledger(); calls = {"entry": 0, "orig": 0}

    def entry(module, *a, **k): calls["entry"] += 1; return "kernel"

    def orig(): calls["orig"] += 1; return "stock"

    small = (_FakeT(1, 4104, 400, 64), "z", "mask")
    assert MK.dispatch("pwa", led, entry, RuntimeError, _FakePWA(), small, {}, orig) == "kernel"
    assert calls == {"entry": 1, "orig": 0} and led.fields()["served"] == 1 and MK.skipped(led) == {} and MK.execution_word(led) == "executed:1"
    assert MK.verdict(led)["ok"]


def test_report_and_lever_line_carry_the_named_skip(monkeypatch):
    monkeypatch.setitem(modes.MODES["big"], "off", {})               # the evidence below reads a two-unit report against the memory row: table both units on it for the test (the shipped row names fpf_pwa off, CHANGES 0.3.3)
    MK.undo()
    led = _unit_ledger()
    monkeypatch.setitem(MK._STATE, "ledgers", {"pwa": led}); monkeypatch.setitem(MK._STATE, "applied", ["fpf_pwa"]); monkeypatch.setitem(MK._STATE, "units", ["pwa"])
    monkeypatch.setitem(MK._STATE, "execution", {"pwa": "executed"}); monkeypatch.setitem(MK._STATE, "facts", {"pwa": {"cfg": "default"}})
    for _ in range(46):
        led.serve("1x4104x400x64/float32")
    MK.skip_by_name("pwa", led, "int32_limit:S·N·HC=7889485824>=2^31", (_FakeT(1, 8192, 3762, 64),))
    MK.skip_by_name("pwa", led, "int32_limit:S·N·HC=7889485824>=2^31", (_FakeT(1, 8192, 3762, 64),))
    r = MK.report()
    u = r["units"]["pwa"]
    assert r["gate"]["ok"] and u["execution"] == "executed:46,skipped_by_name:int32_limit:2" and u["skipped_by_name"] == {"int32_limit": 2} and u["census"]["served"] == 46
    assert "execution=executed:46,skipped_by_name:int32_limit:2" in u["line"] and "skipped_by_name=" in u["line"] and "gate=ok" in u["line"], u["line"]
    log = {"env": {"boltz_levers": ["resid", "mask2"], "f2_patch_active": True}, "per_item": [{"name": "a", "seed": 1}], "events": [], "msa_report": dict(r, applied=["fpf_opm", "fpf_pwa"], variant="opm,pwa",
           units={"opm": dict(u, lever="fpf_opm", skipped_by_name={}), "pwa": dict(u, census=dict(u["census"], served=0))})}
    ev, problems = stack.evidence("big", log)
    assert not any("unit pwa installed but served no call" in p for p in problems), "every pwa call skipped by name is accounted, not idle"
    assert ev["msa_skipped_by_name"]["pwa"] == {"int32_limit": 2}
    from .. import report as REP
    st_, why, pairs = REP.lever_state("fpf_pwa", {"msa_report": r}, {})
    assert st_ == "on" and ("execution", "executed:46,skipped_by_name:int32_limit:2") in pairs


def test_the_card_token_floor_for_opm_on_cc_8_0_is_a_named_precall_disposition(monkeypatch):
    """CARD_MIN_TOKENS: on compute capability 8.0 the OPM kernel serves from 385 tokens (Boltz-2's hidden-chunked regime, where the sm_80 row
    gains); a call under the floor runs the stock forward BY NAME (below_min_tokens), decided from the shapes before any launch — counted as
    skipped_by_name, never a fallback or an error. No floor on any other card, none for pwa; the int32 disposition is unchanged."""
    assert MK.CARD_MIN_TOKENS == {"8.0": {"opm": 385}}
    monkeypatch.setitem(MK._STATE, "cc", "8.0")
    assert MK.card_min_tokens("opm") == 385 and MK.card_min_tokens("pwa") is None
    assert MK.precall_disposition("opm", object(), (_FakeT(1, 1024, 384, 64),), {}) == "below_min_tokens:N=384<385(cc8.0)"
    assert MK.precall_disposition("opm", object(), (_FakeT(1, 1024, 385, 64),), {}) is None
    assert MK.precall_disposition("opm", object(), (), {"m": _FakeT(1, 1, 20, 64)}) == "below_min_tokens:N=20<385(cc8.0)"
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 1024, 199, 64),), {}) is None, "pwa has no floor on cc 8.0"
    assert MK.precall_disposition("pwa", _FakePWA(), (_FakeT(1, 8192, 4375, 64),), {}).startswith("int32_limit:"), "the int32 disposition still speaks"
    led = _unit_ledger("opm")
    MK.skip_by_name("opm", led, "below_min_tokens:N=20<385(cc8.0)", (_FakeT(1, 1, 20, 64),))
    assert MK.skipped(led) == {"below_min_tokens": 1} and MK.verdict(led)["ok"] and MK.execution_word(led) == "executed:0,skipped_by_name:below_min_tokens:1"
    for cc in ("9.0", "10.0", None):
        monkeypatch.setitem(MK._STATE, "cc", cc)
        assert MK.card_min_tokens("opm") is None and MK.precall_disposition("opm", object(), (_FakeT(1, 1, 20, 64),), {}) is None, cc


def test_the_core_cell_carries_one_opm_row_for_cc_8_0_sized_for_sm_80_shared_memory():
    """opt_core.ops.msa_opm (the shared core's cell; read as text — it imports torch/triton): _CFG_BY_CC has exactly the (8, 0) row for
    ("mask_norm", 128) — Boltz-2's schema — a pointer-load layout (no TMA / fused prologue — sm_90 features) whose per-program shared memory
    3 stages x (BM*BK + BK*BN) x 2 B with BM = 32*BI, BN = 32*BJ is 147,456 B <= sm_80's 166,912 B opt-in limit (the f1 / p8 rows need
    198,656 B); the decision order in cfg_for is override -> card row -> f1 (TMA importable) -> p8, and forward_mask_norm / opm_core take it
    from cfg_for only. The adapter reads the same words (msa_kernels.CORE_OPM)."""
    import ast, re, importlib.util
    spec = importlib.util.find_spec(MK.CORE_OPM)
    assert spec is not None and spec.origin, MK.CORE_OPM
    src = open(spec.origin).read()
    m = re.search(r"^_CFG_BY_CC = (\{.*\})\s+#", src, re.M)
    table = eval(m.group(1), {"dict": dict})                           # a literal of dict(...) calls
    assert list(table) == [(8, 0)] and list(table[(8, 0)]) == [("mask_norm", 128)]
    row = table[(8, 0)][("mask_norm", 128)]
    assert row == dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, AT=True) and not row.get("TMA") and not row.get("FP")
    CH = 32; BM, BN = CH * row["BI"], CH * row["BJ"]
    smem = row["num_stages"] * (BM * row["BK"] + row["BK"] * BN) * 2
    assert smem == 147456 <= 166912, smem
    assert "        (\"mask_norm\", 128): dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True)}" in src, "the H100 row f1, unchanged"
    body = src[src.index("def cfg_for("):src.index("def cfg_name(")].split('"""')[2]      # the code after the docstring
    order = [body.index(k) for k in ("_cfg_override()", "_CFG_BY_CC", "_CFG[(form, CZ)]", "_CFG_NO_TMA")]
    assert order == sorted(order), "override -> card row -> TMA row -> pointer row"
    assert "cfg = cfg_for(\"mask_norm\", cache[\"CZ\"], m.device)[0]" in src and "cfg = cfg or cfg_for(form, CZ, dev)[0]" in src and src.count("_CFG_NO_TMA[") == 1
    tree = ast.parse(src)   # the module still parses
    assert any(isinstance(n, ast.FunctionDef) and n.name == "device_cc" for n in tree.body)
    assert MK.PACKAGE_FILES == ("forward/fpf_msa/__init__.py", "forward/fpf_msa/_compat.py", "forward/fpf_msa/boltz2.py"), "the kit carries the Boltz-2 entry points only; the kernels are the core's"


