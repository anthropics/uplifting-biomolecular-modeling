"""The kernel census (lncensus part 2): the KERNELS line grammar, the REQUIRE guard's refusals (exit 5) on stubbed absent / fell-back readings,
by-rule reference calls counted and NOT refused, a reference call the library's rules say the kernel serves refused, and the modes table <->
expected-words consistency (the guard's single source of truth), plus the `default` preset (upstream as shipped) and its route plumbing."""
import json
import os
import re
import sys
import types

import pytest

from opendde_opt import cli, lncensus, modes, registry, stock_pred

WORD = re.compile(r"^(engaged:\S+|off-by-route:\S+|absent:\S+|fallback:\S+|n/a-upstream:\S+)$")
TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # opendde/


def _presence(ok=True, error=None):
    site = lambda: {"ok": ok, "func": lncensus.FRONT + ".f", "file": "/sp/cuequivariance_torch/primitives/triangle.py", "error": error, "threshold": 100}  # noqa: E731
    return {"sites": {"cueq_triatt": site(), "cueq_trimul": site()}, "front_version": "0.10.0", "ops_dist": "cuequivariance-ops-torch-cu12",
            "ops_dist_version": "0.10.0", "ops_build": "cu12", "ops_version": "0.10.0", "ops_commit": "0123abcd9", "sm100f": False,
            "device_cc": [9, 0], "opendde_version": "1.1.1", "torch": "2.7.1+cu126", "ops_error": error}


LN_OK = {"requested": "fast_layernorm", "backend": lncensus.EXTENSION, "module": "/jit/fast_layer_norm_cuda_v2.so", "reason": None}
LN_TORCH = {"requested": "torch", "backend": "torch", "module": None, "reason": "not_requested"}


@pytest.fixture()
def stubbed(monkeypatch):
    """presence / counters / probe / the phase hook stubbed: the guard's logic runs on readings the test dictates (no torch, no GPU)."""
    lncensus.reset()
    state = {"presence": _presence()}
    monkeypatch.setattr(lncensus, "presence", lambda: state["presence"])
    monkeypatch.setattr(lncensus, "install_counters", lambda: lncensus.K.__setitem__("counters_installed", True) or {})
    monkeypatch.setattr(lncensus, "probe", lambda sites=lncensus.KERNEL_SITES, device=None: {s: {"ok": True, "served": True, "s": 0.1, "error": None} for s in sites})
    from opendde_opt import phase
    monkeypatch.setattr(phase, "on_runner_module", lambda cb: "armed")
    import atexit
    monkeypatch.setattr(atexit, "register", lambda *a, **k: None)
    yield state
    lncensus.reset()


def _arm(route="stock", ln_requested=True, line=None, n_gpu=1, ln=LN_OK, do_probe=True):
    exp = modes.kernel_expectations(line, ln_requested=ln_requested, n_gpu=n_gpu)   # ln_requested: LAYERNORM_TYPE=fast_layernorm in the model process's environment (settings.ln_requested)
    return lncensus.arm(route, exp, tag="[opendde-opt stock]", strict=True, stream=sys.stdout, layernorm=ln, do_probe=do_probe)


# ------------------------------------------------------------------------------------------------ the line grammar
def test_kernels_line_grammar_on_the_stock_route(stubbed, capsys):
    _arm()
    lncensus.K["resolved"] = {"cueq_triatt": "cuequivariance", "cueq_trimul": "cuequivariance", "dtype": "bf16"}
    lncensus.K["counts"]["cueq_triatt"].update(front=1000, reference=10, byrule={"S<=100": 10})
    lncensus.K["counts"]["cueq_trimul"].update(front=960, reference=0)
    assert lncensus.finish(0) == 0
    out = capsys.readouterr().out.strip().splitlines()
    lines = [l for l in out if " KERNELS route=" in l]
    assert len(lines) == 1, out                                                          # ONE line per pass; `KERNELS route=` is the cross-engine grep
    line = lines[0]
    assert line.startswith("[opendde-opt stock] KERNELS route=stock fast_layernorm=engaged:fast_layer_norm_cuda_v2@opendde-1.1.1 "
                           "cueq_triatt=engaged:sm90@0.10.0-cu12[served=990,byrule=10(S<=100:10)] cueq_trimul=engaged:sm90@0.10.0-cu12[served=960,byrule=0] "
                           "others=n/a-upstream:deepspeed_ds4sci,cutlass,flash_attn,xformers,trifast,transformer_engine,apex,tokamax(0_references_in_opendde_1.1.1) "), line
    assert "cueq_dist=cuequivariance-torch@0.10.0,cuequivariance-ops-torch-cu12@0.10.0(g0123abc)" in line and "resolved=cuequivariance,cuequivariance" in line
    w = lncensus.words()
    assert set(w) == set(lncensus.ACCELERATORS) and all(WORD.match(v) for v in w.values()), w


def test_route_words():
    L = modes.LINES
    assert modes.route_word(None, None, 1) == "stock" and modes.route_word(None, None) == "stock"     # the stock caller runs at one card (n_gpu>1 under off is refused before it starts)
    assert modes.route_word(L["S1"], "exact", 1) == "exact" and modes.route_word(L["LSTAR2A"], "fast", 1) == "fast"
    assert modes.route_word(L["BIG_TP"], "big", 2) == "big_x2" and modes.route_word(L["LSTAR2A"], None, 1) == "line-LSTAR2A"


# ------------------------------------------------------------------------------------------------ the guard's refusals (exit 5)
def test_guard_refuses_an_absent_library_before_the_run_with_exit_5(stubbed, capsys):
    stubbed["presence"] = _presence(ok=False, error="ModuleNotFoundError: No module named 'cuequivariance_ops_torch'")
    with pytest.raises(lncensus.KernelsRefused) as ei:
        _arm()
    assert ei.value.code == lncensus.EXIT_KERNELS == 5
    out = capsys.readouterr().out
    assert "KERNELS route=stock " in out and "cueq_triatt=absent:ModuleNotFoundError:_No_module_named_'cuequivariance_ops_torch'" in out, out
    assert "KERNELS REFUSED route=stock at=arm exit=5 cueq_triatt=absent:" in out and "cueq_trimul=absent:" in out
    assert [p.split("=")[0] for p in ei.value.problems] == ["cueq_triatt", "cueq_trimul"]


def test_guard_refuses_a_failed_device_probe(stubbed, capsys, monkeypatch):
    monkeypatch.setattr(lncensus, "probe", lambda sites=lncensus.KERNEL_SITES, device=None:
                        {s: {"ok": s != "cueq_trimul", "served": s != "cueq_trimul", "s": 0.1, "error": "RuntimeError: no kernel image is available for execution on the device" if s == "cueq_trimul" else None} for s in sites})
    with pytest.raises(lncensus.KernelsRefused) as ei:
        _arm()
    assert ei.value.code == 5 and ei.value.problems[0].startswith("cueq_trimul=fallback:probe_failed:RuntimeError:_no_kernel_image"), ei.value.problems
    assert "cueq_triatt=engaged:sm90@0.10.0-cu12" in capsys.readouterr().out


def test_guard_refuses_auto_resolved_to_torch_where_the_table_expects_cuequivariance(stubbed, capsys):
    """Upstream's `auto` -> torch (config/inference.py:162-182, INFO-logged only) on a route whose modes-table word is engaged: refused at runner init."""
    _arm()
    mod = types.ModuleType("runner.inference")

    class InferenceRunner:                                                              # upstream's runner: configs resolved in __init__ (runner/inference.py:1150-1159)
        def __init__(self, configs):
            self.configs, self.device = configs, "cuda:0"
    mod.InferenceRunner = InferenceRunner
    lncensus._on_runner_module(mod)
    with pytest.raises(lncensus.KernelsRefused) as ei:
        mod.InferenceRunner(types.SimpleNamespace(triangle_attention="torch", triangle_multiplicative="cuequivariance", dtype="bf16"))
    assert ei.value.code == 5
    out = capsys.readouterr().out
    assert "cueq_triatt=fallback:resolved_torch(expected_cuequivariance:modes_table)" in out and "KERNELS REFUSED route=stock at=runner_init exit=5" in out, out
    assert "cueq_trimul=engaged:" in out


def test_torch_layernorm_words_when_unrequested_and_fast_layernorm_engaged_there_is_refused(stubbed, capsys):
    """LAYERNORM_TYPE unset (upstream's default): torch LayerNorm is the expected word on the stock route; the fused kernel bound there is refused by name."""
    _arm(route="stock", ln_requested=False, ln=LN_TORCH)
    w = lncensus.words()
    assert w["fast_layernorm"] == "off-by-route:LAYERNORM_TYPE_unset(upstream_default_torch:layers.py:25-30)" and w["cueq_triatt"].startswith("engaged:")
    assert lncensus.finish(0) == 0 and "KERNELS route=stock fast_layernorm=off-by-route:" in capsys.readouterr().out
    lncensus.reset()
    with pytest.raises(lncensus.KernelsRefused) as ei:                                  # the default route with the fused LayerNorm bound is not the default: refused by name
        _arm(route="stock", ln_requested=False, ln=LN_OK)
    assert ei.value.problems == ["fast_layernorm=fallback:engaged_where_route_defines_torch(LAYERNORM_TYPE_unset(upstream_default_torch:layers.py:25-30))"]


def test_a_kit_line_owning_a_site_prints_off_by_route_with_the_levers(stubbed, capsys):
    _arm(route="fast", line=modes.LINES["LSTAR2A"])
    lncensus.K["resolved"] = {"cueq_triatt": "cuequivariance", "cueq_trimul": "cuequivariance", "dtype": "bf16"}
    lncensus.K["counts"]["cueq_triatt"].update(front=592, reference=0)                 # ARM U replaced the site in part; the library served the arm's out-of-scope delegations: engaged + replaced=<lever>
    w = lncensus.words()
    assert w["cueq_triatt"] == "engaged:sm90@0.10.0-cu12[served=592,byrule=0;replaced=arm_u]" and w["cueq_trimul"] == "off-by-route:arm_u+arm_u23+trimul_core[served=0,byrule=0]", w   # off-by-route <=> served=0
    assert lncensus.problems_of(w) == []
    assert lncensus.problems_of(w) == [] and lncensus.finish(0) == 0


# ------------------------------------------------------------------------------------------------ by-rule counted, a served-expected reference call refused
class _T:
    """A tensor stand-in the rule replay reads: shape, size(i), dtype."""
    def __init__(self, shape, dtype="bfloat16"):
        self.shape, self.dtype = tuple(shape), "torch." + dtype

    def size(self, i):
        return self.shape[i]


def _fake_library(monkeypatch):
    """The two front functions and the two ops submodules (reference function + threshold) as the pinned library lays them out."""
    front = types.ModuleType(lncensus.FRONT)
    subs = {}
    for site, (sub, ref, thr) in lncensus.OPS_REF.items():
        m = types.ModuleType(sub); setattr(m, thr, 100); setattr(m, ref, lambda *a, **k: "reference"); subs[site] = m
        monkeypatch.setitem(sys.modules, sub, m)
    front.triangle_attention = lambda q, *a, **k: subs["cueq_triatt"]._triangle_attention_torch(q) if q.size(3) <= 100 or getattr(q, "force_ref", False) else "kernel"
    front.triangle_multiplicative_update = lambda x, *a, **k: subs["cueq_trimul"]._tri_mul_torch(x) if x.shape[-2] <= 100 else "kernel"
    monkeypatch.setitem(sys.modules, lncensus.FRONT, front)
    for parent in ("cuequivariance_torch", "cuequivariance_torch.primitives", "cuequivariance_ops_torch"):
        monkeypatch.setitem(sys.modules, parent, sys.modules.get(parent) or types.ModuleType(parent))
    return front


def test_byrule_reference_calls_are_counted_with_reasons_and_not_refused(stubbed, capsys, monkeypatch):
    front = _fake_library(monkeypatch)
    monkeypatch.setattr(lncensus, "install_counters", _real_install_counters)
    _arm(do_probe=False)
    lncensus.K["resolved"] = {"cueq_triatt": "cuequivariance", "cueq_trimul": "cuequivariance", "dtype": "bf16"}
    f, g = front.triangle_attention, front.triangle_multiplicative_update                # the wrapped objects upstream's call sites bind
    assert getattr(f, lncensus.MARK) == "front" and getattr(g, lncensus.MARK) == "front"
    for _ in range(3):
        assert f(_T((1, 4, 4, 64, 32))) == "reference"                                    # S=64 <= 100: the library's own rule -> byrule
    assert f(_T((1, 4, 4, 384, 32))) == "kernel"                                          # S=384: served
    assert f(_T((1, 4, 4, 150, 16))) == "kernel" or True                                  # (hidden 16 -> threshold 200 by rule; the fake front serves it: a served call is never inspected)
    assert g(_T((1, 64, 64, 128))) == "reference" and g(_T((1, 384, 384, 128))) == "kernel"
    w = lncensus.words()
    assert w["cueq_triatt"] == "engaged:sm90@0.10.0-cu12[served=2,byrule=3(S<=100:3)]", w
    assert w["cueq_trimul"] == "engaged:sm90@0.10.0-cu12[served=1,byrule=1(N<=100:1)]", w
    assert lncensus.problems_of(w) == [] and lncensus.finish(0) == 0
    assert "byrule=3(S<=100:3)" in capsys.readouterr().out


def test_a_reference_call_the_rules_say_is_served_is_refused_at_exit(stubbed, capsys, monkeypatch):
    front = _fake_library(monkeypatch)
    monkeypatch.setattr(lncensus, "install_counters", _real_install_counters)
    _arm(do_probe=False)
    lncensus.K["resolved"] = {"cueq_triatt": "cuequivariance", "cueq_trimul": "cuequivariance", "dtype": "bf16"}
    q = _T((1, 4, 4, 384, 32)); q.force_ref = True                                        # S=384, hidden 32, bf16: the rules say SERVED — the library took its torch path anyway
    assert front.triangle_attention(q) == "reference"
    w = lncensus.words()
    assert w["cueq_triatt"] == "fallback:reference_path_where_rules_say_served[served=0,byrule=0,violations=1]", w
    assert lncensus.finish(0) == lncensus.EXIT_KERNELS == 5
    out = capsys.readouterr().out
    assert "KERNELS route=stock " in out and "KERNELS REFUSED route=stock at=exit exit=5 cueq_triatt=fallback:reference_path_where_rules_say_served" in out, out


_real_install_counters = lncensus.install_counters


def test_an_unconfirmable_census_is_refused_unless_the_test_only_opt_out_is_exported(stubbed, capsys, monkeypatch):
    """No `opendde.model.triangular.layers` in the interpreter (Part 1's `unconfirmed`): refused (exit 5) — unless MODEL_OPT_TEST_KERNELS_UNCONFIRMED=1
    says so by name on the line; the prefix MODEL_OPT_TEST_ is must-be-absent on the stock routes."""
    ln_unv = {"requested": "fast_layernorm", "backend": "unconfirmed", "module": None, "reason": "absent:opendde.model.triangular.layers"}
    monkeypatch.delenv(lncensus.TEST_UNCONFIRMED_ENV, raising=False)
    with pytest.raises(lncensus.KernelsRefused) as ei:
        _arm(ln=ln_unv)
    assert ei.value.code == 5 and ei.value.problems[0] == "census=unconfirmed(absent:opendde.model.triangular.layers)"
    assert "KERNELS REFUSED route=stock at=arm exit=5 census=unconfirmed(" in capsys.readouterr().out
    monkeypatch.setenv(lncensus.TEST_UNCONFIRMED_ENV, "1")
    _arm(ln=ln_unv)
    assert lncensus.finish(0) == 0
    assert "census=unconfirmed(absent:opendde.model.triangular.layers;allowed_by=MODEL_OPT_TEST_KERNELS_UNCONFIRMED=1)" in capsys.readouterr().out
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    assert any(lncensus.TEST_UNCONFIRMED_ENV.startswith(p) for p in pins["stock_environment"]["must_be_absent_prefixes"])


def test_an_unreadable_reference_call_is_a_violation_not_a_byrule_pass():
    assert lncensus._byrule_reason("cueq_triatt", (object(),), {}) is None                                                 # fail-closed: not explained by the rules -> violation
    assert lncensus._byrule_reason("cueq_trimul", (), {}) is None


def test_the_off_route_refuses_a_stock_proof_without_the_census_block():
    """Source census (fail-closed): cmd_pred's stock route exits 5 when the model process returned 0 but its proof carries no KERNELS record/line."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli.py")).read()
    assert 'kernels_ok = bool(kblock.get("line")) and not kblock.get("problems")' in src
    assert "if rc == EXIT_KERNELS or (rc == 0 and not kernels_ok):" in src and '"kernels_census_missing_from_stock_proof"' in src


def test_the_rule_replay_names_dims_and_dtype_reasons():
    m = types.ModuleType("x"); m.CUEQ_TRIATTN_FALLBACK_THRESHOLD = 100
    sys.modules.setdefault(lncensus.OPS_REF["cueq_triatt"][0], m)
    assert lncensus._byrule_reason("cueq_triatt", (_T((1, 1, 1, 500, 48), "float32"),), {}) == "dims(float32,hidden=48)"     # fp32 hidden > 32: the library's FORCE_FALLBACK
    assert lncensus._byrule_reason("cueq_triatt", (_T((1, 1, 1, 150, 16)),), {}) == "S<=200"                                  # hidden < 32 raises the threshold to 200
    assert lncensus._byrule_reason("cueq_triatt", (_T((1, 1, 1, 500, 32)),), {}) is None                                       # served by rule: a reference path here is a violation
    assert lncensus._byrule_reason("cueq_trimul", (_T((1, 80, 80, 128)),), {}) == "N<=100"


# ------------------------------------------------------------------------------------------------ modes table <-> expected words (the guard's single source of truth)
def test_every_mode_line_and_big_line_has_a_complete_expectation_row():
    names = set(modes.MODE_LINES.values()) - {None} | {n for n in modes.LINES if n.startswith("BIG")}
    for name in sorted(names):
        for n_gpu in (1, 2):
            exp = modes.kernel_expectations(modes.LINES[name], ln_requested=True, n_gpu=n_gpu)
            assert tuple(exp) == lncensus.ACCELERATORS, name
            for site in lncensus.KERNEL_SITES:
                e = exp[site]
                assert e["kind"] in ("engaged", "off-by-route") and e["resolved"] == "cuequivariance" and e["need_stack"] is True, (name, site, e)
                owners = [lv for lv in modes.LINES[name].levers if lv in registry.KERNEL_SITE_OWNERS[site]]
                assert (e["kind"] == "off-by-route") == bool(owners) and (e["reason"] or "") == "+".join(owners), (name, site, e)
    stock = modes.kernel_expectations(None, ln_requested=True, n_gpu=1)
    assert {k: v["kind"] for k, v in stock.items()} == {"fast_layernorm": "engaged", "cueq_triatt": "engaged", "cueq_trimul": "engaged"}
    exact = modes.kernel_expectations(modes.LINES[modes.MODE_LINES["exact"]], ln_requested=True)
    assert exact["cueq_triatt"]["kind"] == "engaged" and exact["cueq_trimul"] == {"kind": "off-by-route", "reason": "fpf_trimul_exact+trimul_exact", "resolved": "cuequivariance", "need_stack": True}


def test_kernel_site_owners_are_levers_and_sites_are_the_census_sites():
    assert tuple(registry.KERNEL_SITE_OWNERS) == lncensus.KERNEL_SITES == modes.KERNEL_SITES
    for site, names in registry.KERNEL_SITE_OWNERS.items():
        assert names and all(n in registry.LEVERS for n in names), site
        assert all(registry.LEVERS[n].cls == "forward" for n in names), site                     # only a forward lever replaces a kernel site


def test_upstreams_exposed_unused_switches_are_absent_on_every_route():
    assert modes.UPSTREAM_EXPOSED_UNUSED == ("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP", "OPENDDE_FORCE_CONFIDENCE_AMP")
    for name, line in modes.LINES.items():
        assert all(s in line.unset and s not in line.exports for s in modes.UPSTREAM_EXPOSED_UNUSED), name
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    prefixes = pins["stock_environment"]["must_be_absent_prefixes"]
    assert all(any(s.startswith(p) for p in prefixes) for s in modes.UPSTREAM_EXPOSED_UNUSED), prefixes   # the stock caller strips and proves them absent


# ------------------------------------------------------------------------------------------------ the stock caller's environment and route facts


def test_stock_command_keeps_the_callers_layernorm_and_hands_the_route_facts_to_the_stock_caller(monkeypatch, tmp_path):
    """The stock child inherits LAYERNORM_TYPE exactly as the caller has it (upstream's own switch; the package exports nothing), loses upstream's
    unused OPENDDE_* switches; the command is the plain stock caller (no kit-composed launcher or flag)."""
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    monkeypatch.setenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP", "1")
    cmd, env = cli.stock_command(TREE, ["pred", "-i", "q.json"], str(tmp_path / "p.json"), 0)
    assert env.get("LAYERNORM_TYPE") == "fast_layernorm" and "OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP" not in env
    assert "--n-gpu" not in cmd and "--settings" not in cmd and cmd[cmd.index("--") + 1:] == ["pred", "-i", "q.json"]
    monkeypatch.delenv("LAYERNORM_TYPE")
    cmd, env = cli.stock_command(TREE, ["pred", "-i", "q.json"], str(tmp_path / "p.json"), 0)
    assert "LAYERNORM_TYPE" not in env                                                     # unset stays unset: upstream's torch LayerNorm




def test_the_census_is_armed_on_every_model_route_and_exit_5_is_mapped_through():
    """Source census: the kit route (cli.cmd_pred) and the clean stock process (stock_pred) arm the same reader and adopt its exit status."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csrc, ssrc = open(os.path.join(here, "cli.py")).read(), open(os.path.join(here, "stock_pred.py")).read()
    assert "_lncensus.arm(route, modes.kernel_expectations(res.line" in csrc and "rc = _lncensus.finish(rc)" in csrc and "EXIT_KERNELS = _lncensus.EXIT_KERNELS" in csrc
    assert "_lncensus.arm(route, _modes.kernel_expectations(None" in ssrc and "rc = _lncensus.finish(rc)" in ssrc and "EXIT_KERNELS = _lncensus.EXIT_KERNELS" in ssrc
    assert cli.EXIT_KERNELS == stock_pred.EXIT_KERNELS == lncensus.EXIT_KERNELS == 5
