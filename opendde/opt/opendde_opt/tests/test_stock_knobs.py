"""Upstream's own triangle-kernel flags (`--triatt_kernel` / `--trimul_kernel`, runner/batch_inference.py:749-760) run as stated on every route:
the KERNELS census expects what the flag says for that site (never a refusal, `off` included), and under a kit mode the kit levers of that
site step aside BY NAME (opendde_opt/stockknob.py, registry.STOCK_KNOB_LEVERS) — LEVER state=off reason=aside:stock_knob:<flag>=<value>, the
rest of the line unchanged, the run complete. Nothing stated (or `auto`): expectations and lines are today's, byte for byte."""
import os
import sys
import types

import pytest

from opendde_opt import ablate, lncensus, modes, ran, registry, report, settings, stack, stockknob
from opendde_opt.tests import _stubs
from opendde_opt.tests.test_kernel_census import LN_OK, stubbed  # noqa: F401  (fixture)

TREE = _stubs.TREE


@pytest.fixture(autouse=True)
def _clean():
    stockknob._reset(); ablate._reset()
    os.environ.pop(stockknob.ENV, None)
    yield
    stockknob._reset(); ablate._reset()
    os.environ.pop(stockknob.ENV, None)


def _runner_module():
    mod = types.ModuleType("runner.inference")

    class InferenceRunner:                                                               # upstream's runner: configs resolved in __init__
        def __init__(self, configs):
            self.configs, self.device = configs, "cuda:0"
    mod.InferenceRunner = InferenceRunner
    return mod


# ---------------------------------------------------------------------------------------------- (i) the census: as the caller's flag says
@pytest.mark.parametrize("flag,site", sorted(settings.KERNEL_KNOBS.items()))
@pytest.mark.parametrize("route,line", [("stock", None), ("exact", "S1"), ("fast", "LSTAR2A"), ("big", "BIG_F")])
def test_a_stated_kernel_flag_is_the_sites_word_on_every_route_never_a_refusal(stubbed, capsys, flag, site, route, line):
    """`--<flag> torch` stated: the site's expectation is kind `user` (nothing required of the cuEquivariance stack, no resolved value expected);
    the runner resolving that site to torch is NOT a refusal (no exit-5 path) and the KERNELS line says `user:<flag>=torch`; the other site keeps
    its table word."""
    ln = modes.LINES[line] if line else None
    exp = modes.kernel_expectations(ln, ln_requested=True, knobs={flag: "torch"})
    assert exp[site] == {"kind": "user", "reason": f"{flag}=torch", "resolved": None, "need_stack": False}
    other = next(s for s in lncensus.KERNEL_SITES if s != site)
    assert exp[other] == modes.kernel_expectations(ln, ln_requested=True)[other]          # the other site: today's word exactly
    lncensus.arm(route, exp, tag="[opendde-opt]", strict=True, stream=sys.stdout, layernorm=LN_OK, do_probe=True)
    mod = _runner_module()
    lncensus._on_runner_module(mod)
    resolved = {"triangle_attention": "cuequivariance", "triangle_multiplicative": "cuequivariance", "dtype": "bf16"}
    resolved[lncensus.RESOLVED_ATTRS[site]] = "torch"                                   # upstream resolved the caller's flag: torch at that site
    mod.InferenceRunner(types.SimpleNamespace(**resolved))                                # no KernelsRefused
    assert lncensus.finish(0) == 0
    out = capsys.readouterr().out
    assert f"{site}=user:{flag}=torch" in out and "REFUSED" not in out and "exit=5" not in out, out
    assert lncensus.problems_of(lncensus.words()) == []


def test_nothing_stated_or_auto_keeps_todays_expectations():
    """No flag / `auto`: the expectations are byte-identical to today's on every route (the guard's table unchanged)."""
    for ln in (None, modes.LINES["S1"], modes.LINES["LSTAR2A"], modes.LINES["BIG_F"], modes.LINES["BIG_TP"]):
        base = modes.kernel_expectations(ln, ln_requested=True, n_gpu=2 if ln is modes.LINES["BIG_TP"] else 1)
        assert modes.kernel_expectations(ln, ln_requested=True, n_gpu=2 if ln is modes.LINES["BIG_TP"] else 1, knobs={}) == base
        assert modes.kernel_expectations(ln, ln_requested=True, n_gpu=2 if ln is modes.LINES["BIG_TP"] else 1, knobs=settings.stock_knobs({"triatt_kernel": "auto"})) == base
    assert settings.stock_knobs({}) == {} and settings.stock_knobs({"triatt_kernel": "auto", "trimul_kernel": "auto"}) == {}
    assert settings.KERNEL_KNOBS == modes._KNOB_SITES and set(settings.KERNEL_KNOBS) <= set(settings.FLAGS)
    assert set(settings.KERNEL_KNOBS.values()) == set(lncensus.KERNEL_SITES)


def test_the_stock_caller_reads_the_flags_from_the_arguments_it_hands_upstream():
    argv = ["pred", "-i", "/q.json", "-o", "/out", "--dtype", "bf16", "--triatt_kernel", "torch", "--trimul_kernel=torch"]
    assert settings.stated_in(argv) == {"triatt_kernel": "torch", "trimul_kernel": "torch"}
    assert settings.stock_knobs(settings.stated_in(argv[:7])) == {}
    assert settings.stock_knobs(settings.stated_in(["--triatt_kernel", "auto"])) == {}


# ---------------------------------------------------------------------------------------------- (ii) the kit levers of that site step aside by name
CASES = [                                                                                  # (mode, knobs) -> the levers that leave the line (registry.STOCK_KNOB_LEVERS + dependents)
    ("exact", {"triatt_kernel": "torch"}, {"triattn_exact", "triattn_conf"}),
    ("exact", {"trimul_kernel": "torch"}, {"trimul_exact", "fpf_trimul_exact"}),
    ("fast", {"triatt_kernel": "torch"}, {"triattn_core", "triattn_conf"}),
    ("fast", {"trimul_kernel": "torch"}, {"trimul_core", "arm_u23"}),
    ("big", {"triatt_kernel": "torch", "trimul_kernel": "cuequivariance"}, {"triattn_core", "triattn_conf", "trimul_core", "arm_u23"}),
]


@pytest.mark.parametrize("mode,knobs,gone", CASES)
def test_a_stated_kernel_flag_composes_the_kits_levers_of_that_site_out_by_name(mode, knobs, gone):
    line0 = modes.resolve(mode, TREE, {}).line                                             # today's line for this call (no query: the size gates' own decisions, alike in both)
    stockknob.plan(knobs)
    res = modes.resolve(mode, TREE, {})
    left = {x for x in line0.levers if x not in res.line.levers}
    assert left == gone, (left, gone)
    g = stockknob.gated_off(res.line)
    assert set(g) == gone and all(w.startswith("aside:stock_knob:") for w in g.values()), g
    for lv in gone:                                                                        # each names the flag that put it aside
        flag = next(f for f, lvs in registry.STOCK_KNOB_LEVERS.items() if lv in lvs or any(lv in modes.LEVER_DEPENDENTS.get(p, ()) for p in lvs))
        assert g[lv] == f"aside:stock_knob:{flag}={knobs[flag]}", (lv, g[lv])
    for lv in gone:                                                                        # its switches left the environment contract (the ablation rule, modes.LEVER_SWITCHES)
        for sw in modes.LEVER_SWITCHES.get(lv, ()):
            assert sw not in res.line.exports or (lv == "triattn_conf"), (lv, sw)
    if "trimul_kernel" in knobs and mode == "exact":
        assert not any(k.startswith("FPF_") for k in res.line.exports) and "ODDE_TRIMUL" not in res.line.exports
    if "triatt_kernel" in knobs:
        assert "ODDE_TRIATTN" not in res.line.exports
    rest = [x for x in line0.levers if x not in gone]                                      # the rest of the line unchanged (LN, sampler, writer, memory levers ...)
    assert [x for x in res.line.levers] == rest
    assert stockknob.word() == ",".join(f"{f}={knobs[f]}" for f in settings.KERNEL_KNOBS if f in knobs)
    env = {}
    assert stockknob.export(env) == stockknob.word() and stockknob.in_force(env) == knobs   # the fact the kit process / rank processes read


def test_lever_rows_read_aside_stock_knob_and_the_run_is_complete(monkeypatch):
    """The activation report of a dry run under `--triatt_kernel torch`: the LEVER rows of the site's levers read state=off reason=aside:stock_knob:…,
    the ACTIVE line carries stock_knobs=, nothing is a fallback and the report is not PARTIAL; ran.engagement names the same reason (inert, never
    lever_never_ran) for a process that reads the fact."""
    stockknob.plan({"triatt_kernel": "torch", "trimul_kernel": "torch"})
    rep = stack.activate("fast", dry_run=True, tree=TREE)
    assert rep.get("stock_knobs") == "triatt_kernel=torch,trimul_kernel=torch" and not rep.get("partial") and not rep.get("levers_fallback")
    for lv, flag in (("triattn_core", "triatt_kernel"), ("triattn_conf", "triatt_kernel"), ("trimul_core", "trimul_kernel"), ("arm_u23", "trimul_kernel")):
        assert lv not in (rep.get("levers_planned") or []), lv
        assert report.lever_state(lv, rep) == ("off", f"aside:stock_knob:{flag}=torch"), (lv, report.lever_state(lv, rep))
    line_text = str(rep.get("line") or "")                                                 # the composed line's environment contract (describe_line)
    assert line_text.startswith("LSTAR2A(") and "ODDE_ARM_U=1" in line_text                # the rest of the line stays (ARM U itself, the sampler words ...)
    for k in ("ODDE_TRIATTN=", "ODDE_TRIMUL=", "ODDE_ARM_U2_TRIMUL="):
        assert k not in line_text, k
    assert "stock_knobs=triatt_kernel=torch,trimul_kernel=torch" in report.activation_line(rep)
    for lv in ("triattn_core", "triattn_exact", "triattn_conf"):
        ok, why = ran.engagement(lv, {"stock_knobs": {"triatt_kernel": "torch"}})
        assert not ok and why.startswith("aside:stock_knob:triatt_kernel=torch"), (lv, why)
    for lv in ("trimul_core", "trimul_exact", "arm_u23", "fpf_trimul_exact"):
        ok, why = ran.engagement(lv, {"stock_knobs": {"trimul_kernel": "torch"}})
        assert not ok and why.startswith("aside:stock_knob:trimul_kernel=torch"), (lv, why)
    monkeypatch.setenv(stockknob.ENV, "triatt_kernel=torch")                              # a rank process / the hook: the fact in the environment
    assert ran.engagement("triattn_core", {})[0] is False and ran.engagement("trimul_core", {})[0] is True


def test_no_knob_changes_nothing():
    stockknob.plan({})
    for mode in ("exact", "fast", "big"):
        a = modes.resolve(mode, TREE, {}).line
        stockknob.plan({"triatt_kernel": "auto"} if False else {})
        b = modes.resolve(mode, TREE, {}).line
        assert a == b and stockknob.gated_off(b) == {} and stockknob.word() is None
    env = {"ODDE_STOCK_KNOBS": "stale"}
    assert stockknob.export(env) is None and stockknob.ENV not in env                     # nothing stated: the fact is removed, never left stale
    assert stockknob.ENV in modes.RUN_FACTS and stockknob.ENV not in modes._ALL_SWITCHES   # the package's own per-run fact: no line exports or unsets it


def test_the_table_names_levers_with_a_composition_rule():
    for flag, lvs in registry.STOCK_KNOB_LEVERS.items():
        assert flag in settings.KERNEL_KNOBS
        for lv in lvs:
            assert lv in registry.LEVERS and lv in modes.LEVER_SWITCHES and lv not in modes.LEVER_NOT_SHEDDABLE, lv
            assert registry.ENGAGEMENT[lv].stock_knob == flag, lv
    owners = {lv for site in lncensus.KERNEL_SITES for lv in registry.KERNEL_SITE_OWNERS[site]}
    named = {lv for lvs in registry.STOCK_KNOB_LEVERS.values() for lv in lvs}
    assert {"fpf_trimul_exact", "arm_u23"} <= named and "arm_u" in owners and "arm_u" not in named   # ARM U itself stays (bf16 stacks, transition): its own sites read the fact


def test_the_arm_reads_the_fact_and_keeps_its_own_sites_on_the_stock_op():
    """levers/ARMT/odde_arm_t: the inseparable attention site / TriMul routes of the arm read ODDE_STOCK_KNOBS (CPU: the reader only)."""
    sys.path.insert(0, os.path.join(os.path.dirname(TREE), "opendde", "opt", "forward", "fast_inference", "levers", "ARMT")) if False else None
    armt = os.path.join(TREE, "opt", "forward", "fast_inference", "levers", "ARMT")
    src = open(os.path.join(armt, "odde_arm_t", "__init__.py")).read()
    assert 'def stock_knobs(environ=None):' in src and 'att_mode = "stock:stock_knob"' in src and 'trimul_mode = "stock:stock_knob"' in src and 'CFG["U2_TRIMUL"] = False' in src
    ns = {}
    import re as _re
    body = src[src.index("def stock_knobs(environ=None):"):src.index("def install(")]
    exec("import os\n" + body, ns)
    assert ns["stock_knobs"]({"ODDE_STOCK_KNOBS": "triatt_kernel=torch,trimul_kernel=cuequivariance"}) == {"triatt_kernel": "torch", "trimul_kernel": "cuequivariance"}
    assert ns["stock_knobs"]({}) == {} and ns["stock_knobs"]({"ODDE_STOCK_KNOBS": ","}) == {}
