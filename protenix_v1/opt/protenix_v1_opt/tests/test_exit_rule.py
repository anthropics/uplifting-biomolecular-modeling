"""The exit rule (report.py): the family's partial-exit line byte for byte, the allowed form, the verdict from the kit's own counters,
and every entry point through it — `pred` (rc 3 + the line on a partial run; rc 0 + the recorded allowance with --allow-partial, the one
opt-out: no environment spelling; rc 0 and no line on a full activation), the runner wrap (a lever the kit did not apply), `check`
(the flag recorded, no partial plan), `warm` (the flag passed through), the environment route's record at
exit. Also the per-process item census the runner wrap keeps (report's ITEM / ITEMS lines, the exit-code rule): every `predict`
item is ok or failed-with-a-named-reason, and a failed item is exit 1 even though the stock's per-item handler continues with
exit 0. No torch, no protenix: the stock CLI and the kit's levers module are stubs; the kit's account is synthetic in the kit's own
counter vocabulary (report.kit_evidence's docstring cites the kit lines)."""
import json
import os
import sys
import types

import pytest

import protenix_v1_opt
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK
from protenix_v1_opt import cli, modes, report as R, stack

FAST = modes.resolve("fast")
EXACT = modes.resolve("exact")


def full_account(res):
    """A kit account (levers_ptx1.describe(), JSON-safe) in which every lever of `res` served and nothing fell back."""
    tri = "gflash" if "gflash" in res.levers else "gblock"; tr = "ttr" if "ttr" in res.levers else "xtr"          # the arm's tri-attention / transition lever words
    counts = {"trimul": {res.trimul: 12, "stock:path": 4}, "triattn": {tri: 20, "stock:gate": 2}, "transition": {tr + ":C=128": 40, "stock:C=384": 8},
              "dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12, "stock:bias_per_sample": 1}}
    counts.update({"core:" + w: {r: 20} for w, r in (("triexact", "cueq"), ("tricuda", "cuda_sm90a")) if w in res.levers})     # the provider words' census (levers_ptx1 COUNTS core:<word>)
    return {"cfg": {"trimul": res.trimul, **{lv: True for lv in res.levers}}, "counts": counts,
            "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "fastln": {"how": "prebuilt"}, "errors": [],
                        "sampler": {"captures": 1, "replays": 199, "eager_steps": 0, "bypass": 0}, "hoist": {"hits": 199, "records": 1, "bypass": 0}}}


@pytest.fixture(autouse=True)
def _fresh_stack():
    saved = (stack._REPORT, stack._LEVERS, dict(stack._RUNNERS))
    stack._REPORT, stack._LEVERS = None, None
    stack._RUNNERS.update(built=0, wrapped=False, hook_registered=False)
    yield
    stack._REPORT, stack._LEVERS = saved[0], saved[1]
    stack._RUNNERS.update(saved[2])


# ---------------------------------------------------------------------------------------------------------------- the lines
def test_partial_exit_line_is_the_family_grammar():
    line = R.partial_exit_line(["fast", "sg"], "fast fallback {'error:RuntimeError': 3}")
    assert line == "[protenix-v1-opt] NOT ACTIVE: partial activation — fast,sg: fast fallback {'error:RuntimeError': 3}; exit 3 (--allow-partial records and proceeds)"
    assert line.startswith(R.PREFIX + " NOT ACTIVE: partial activation — ") and line.endswith("; exit 3 (--allow-partial records and proceeds)")
    assert line == R.not_active_line(R.partial_exit_reason(["fast", "sg"], "fast fallback {'error:RuntimeError': 3}"))
    assert R.EXIT_NOT_ACTIVE == 3 and R.ALLOW_PARTIAL_FLAG == "--allow-partial"


def test_partial_allowed_line():
    line = R.partial_allowed_line(["sg"], "sg served 0 calls (gated {'bypass': 200})")
    assert line == "[protenix-v1-opt] PARTIAL allowed: sg: sg served 0 calls (gated {'bypass': 200}) (--allow-partial, recorded)"


def test_activation_line_names_the_partial_levers():
    rep = {"mode": "fast", "arm": FAST.arm, "cfg": {"trimul": "fast", "gflash": True}, "levers": {}, "partial": ["sg", "hoist"]}
    assert " partial=sg,hoist " in R.activation_line(rep)
    assert " partial=- " in R.activation_line(dict(rep, partial=[]))


# ------------------------------------------------------------------------------------------------------------- the verdict
def test_full_account_is_not_partial():
    ev = R.kit_evidence(full_account(FAST), FAST.trimul, FAST.levers)
    assert set(ev) == {"fast", "gflash", "tricuda", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused"}
    assert all(e["served"] and not e["fallback"] for e in ev.values())
    assert R.partial_of(ev) == ([], None)
    v = R.verdict({"active": True, "levers": full_account(FAST)}, FAST.trimul, FAST.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK and v["allow_partial"] is False


@pytest.mark.parametrize("mutate, lever, key", [
    (lambda a: a["counts"]["trimul"].update({"error:RuntimeError": 2}), "fast", "error:RuntimeError"),          # levers_ptx1.py:93 kernel error -> stock
    (lambda a: a["counts"]["trimul"].update({"stock:unsupported:shape": 5}), "fast", "stock:unsupported:shape"),  # levers_ptx1.py:76 the fast cell refused the shape
    (lambda a: a["sampler"]["sampler"].update({"eager_steps": 7}), "sg", "eager_steps"),                      # graphed.py:228/:329
    (lambda a: a["sampler"].update({"graphs": False}), "sg", "graphs"),
    (lambda a: a["sampler"].update({"hoist_installed": False}), "hoist", "hoist_installed"),
])
def test_a_fallback_counter_is_partial(mutate, lever, key):
    acc = full_account(FAST)
    mutate(acc)
    ev = R.kit_evidence(acc, FAST.trimul, FAST.levers)
    assert key in ev[lever]["fallback"]
    partial, reason = R.partial_of(ev)
    assert partial == [lever] and lever in reason and key in reason


def test_the_structural_launch_grid_word_is_a_named_gate_not_partial():
    acc = full_account(EXACT)
    key = R.structural_gates()[0]; assert key == "stock:unsupported:launch_grid_y>65535", key
    acc["counts"]["trimul"][key] = 96                                                # an item above the exact TriMul's launch edge: every call answered by the stock statement, by name
    ev = R.kit_evidence(acc, EXACT.trimul, EXACT.levers)
    assert ev["exact"]["fallback"] == {} and ev["exact"]["gated"].get(key) == 96
    assert R.partial_of(ev) == ([], None)


def test_no_served_call_is_partial_with_the_gate_named():
    acc = full_account(FAST)
    acc["counts"]["triattn"] = {"stock:gate": 30}                                      # every triangle attention call under the kit's own gate (levers_ptx1.py:129)
    partial, reason = R.partial_of(R.kit_evidence(acc, FAST.trimul, FAST.levers))
    assert partial == ["gflash"] and "gflash served 0 calls (gated {'stock:gate': 30})" == reason


def test_contig_retry_is_recorded_not_partial():
    acc = full_account(FAST)
    acc["counts"]["triattn"]["kernel-error->contig:RuntimeError"] = 1                  # levers_ptx1.py:157: the same fused path retried
    ev = R.kit_evidence(acc, FAST.trimul, FAST.levers)
    assert ev["gflash"]["retried"] == {"kernel-error->contig:RuntimeError": 1} and R.partial_of(ev) == ([], None)


def test_missing_account_is_partial_for_every_lever():
    for acc in (None, {"error": "RuntimeError('describe')"}):                              # None: no account at all; the second: the account's own error text
        partial, reason = R.partial_of(R.kit_evidence(acc, EXACT.trimul, EXACT.levers))
        assert partial == [EXACT.trimul, *EXACT.levers] and "account" in reason            # every lever of the arm, in arm order


def test_verdict_joins_activation_partial_and_run_partial():
    acc = full_account(EXACT)
    acc["sampler"]["sampler"]["eager_steps"] = 1
    rep = {"active": True, "levers": acc, "partial": ["hoist"], "partial_reason": "the kit applied {...}: hoist=False expected True"}
    v = R.verdict(rep, EXACT.trimul, EXACT.levers, allow_partial=False)
    assert v["partial"] == ["hoist", "sg"] and v["partial_reason"].startswith("the kit applied") and "eager_steps" in v["partial_reason"]
    assert v["exit_code"] == R.EXIT_NOT_ACTIVE
    assert R.verdict(rep, EXACT.trimul, EXACT.levers, allow_partial=True)["exit_code"] == R.EXIT_OK
    assert R.verdict(rep, EXACT.trimul, EXACT.levers, allow_partial=False, run_ok=False)["exit_code"] is None



def test_a_failed_item_is_the_exit_not_a_partial_verdict():
    """An item that raised inside `predict` (out of memory, …) cuts the run short: the levers past it served no call by consequence. The
    verdict records the partial state but does not judge it (exit_code None → items_exit_code: 1), names the failed items, and the printed
    line is the NOTE form, never `NOT ACTIVE: partial activation` (whose advice, --allow-partial, would be wrong for a run that failed)."""
    acc = full_account(FAST)
    acc["sampler"]["sampler"] = {"captures": 0, "replays": 0, "eager_steps": 0, "bypass": 0}            # the sampler never ran: the item died in the trunk
    census = {"total": 1, "ok": 0, "failed": 1, "open": 0, "failed_items": ["n2000:OutOfMemoryError"]}
    rep = {"active": True, "levers": acc, "items_census": census}
    v = R.verdict(rep, FAST.trimul, FAST.levers, allow_partial=False, run_ok=True)
    assert v["partial"] == ["sg", "sampler_prep"] and v["exit_code"] is None and v["items_failed"] == ["n2000:OutOfMemoryError"]
    assert R.items_exit_code(R.EXIT_OK if v["exit_code"] is None else v["exit_code"], census) == R.EXIT_FAIL
    line = R.log_verdict(v, stream=open(os.devnull, "w"))
    assert line == ("[protenix-v1-opt] NOTE exit rule not judged: 1 item(s) failed or open (n2000:OutOfMemoryError) — sg,sampler_prep served no call past the failure; "
                    "the exit code is the failure's (1)")
    assert R.verdict(rep, FAST.trimul, FAST.levers, allow_partial=True, run_ok=True)["exit_code"] is None   # the allowance changes nothing: the run failed
    open_only = R.verdict(dict(rep, items_census={"total": 1, "ok": 0, "failed": 0, "open": 1, "failed_items": []}), FAST.trimul, FAST.levers, False)
    assert open_only["items_failed"] == ["?:open"] and open_only["exit_code"] is None                  # an item that never returned counts the same
    full = R.verdict({"active": True, "levers": full_account(FAST), "items_census": {"total": 1, "ok": 1, "failed": 0, "open": 0, "failed_items": []}},
                     FAST.trimul, FAST.levers, False)
    assert full["items_failed"] == [] and full["exit_code"] == R.EXIT_OK                              # a complete census leaves the rule as it was

# ---------------------------------------------------------------------------------------------------------- the runner wrap
class _FakeLevers:
    def __init__(self, cfg, account):
        self.cfg, self.account, self.calls = cfg, account, []

    def bind_model(self, model):
        self.calls.append(("bind", model))

    def apply(self, arm):
        self.calls.append(("apply", arm))
        return dict(self.cfg)

    def describe(self):
        return dict(self.account)


def _fake_runner_module(monkeypatch):
    mod = types.ModuleType(stack.RUNNER_MODULE)

    class InferenceRunner:
        def __init__(self, *a, **kw):
            self.model = types.SimpleNamespace(eval=lambda: None)

        def predict(self, data):                                   # the per-item entry the package wraps (runner/inference.py:204)
            return {"predicted": data.get("N_token")}
    mod.InferenceRunner = InferenceRunner
    monkeypatch.setitem(sys.modules, stack.RUNNER_MODULE, mod)
    return mod


def _armed(res, allow_partial, trigger=None):
    stack._REPORT = dict(stack._base(res.mode, trigger, False, allow_partial), armed=True, arm=res.arm, line=res.line, tier=res.tier, levers_requested=list(res.levers))


def test_runner_wrap_refuses_a_lever_the_kit_did_not_apply(monkeypatch, capsys):
    mod = _fake_runner_module(monkeypatch)
    cfg = {"trimul": "exact", **{lv: True for lv in EXACT.levers}, "hoist": False}          # the kit applied the arm without the hoist
    stack._LEVERS = _FakeLevers(cfg, full_account(EXACT))
    _armed(EXACT, allow_partial=False)
    stack._wrap_runner(EXACT)
    with pytest.raises(protenix_v1_opt.ActivationError) as e:
        mod.InferenceRunner()
    reason = str(e.value)
    assert reason.startswith("partial activation — hoist: the kit applied {") and reason.endswith("; exit 3 (--allow-partial records and proceeds)")
    err = capsys.readouterr().err
    assert R.partial_exit_line(["hoist"], stack.status()["partial_reason"]) in err.splitlines()
    assert stack.status()["active"] is False and stack.status()["partial"] == ["hoist"]


def test_runner_wrap_proceeds_with_the_allowance_recorded(monkeypatch, capsys):
    mod = _fake_runner_module(monkeypatch)
    cfg = {"trimul": "exact", **{lv: True for lv in EXACT.levers}, "hoist": False}
    stack._LEVERS = _FakeLevers(cfg, full_account(EXACT))
    _armed(EXACT, allow_partial=True)
    stack._wrap_runner(EXACT)
    mod.InferenceRunner()
    rep = stack.status()
    assert rep["active"] is True and rep["partial"] == ["hoist"] and rep["allow_partial"] is True and "hoist=False expected True" in rep["partial_reason"]
    lines = capsys.readouterr().err.splitlines()
    assert any(l.startswith("[protenix-v1-opt] ACTIVE ") and " partial=hoist " in l for l in lines)
    assert R.partial_allowed_line(["hoist"], rep["partial_reason"]) in lines


def test_runner_wrap_full_activation_has_no_partial(monkeypatch, capsys):
    mod = _fake_runner_module(monkeypatch)
    stack._LEVERS = _FakeLevers({"trimul": "fast", **{lv: True for lv in FAST.levers}}, full_account(FAST))
    _armed(FAST, allow_partial=False)
    stack._wrap_runner(FAST)
    mod.InferenceRunner()
    rep = stack.status()
    assert rep["active"] and rep["partial"] == [] and rep["partial_reason"] is None
    assert "PARTIAL" not in capsys.readouterr().err


def test_environment_route_partial_exits_3(monkeypatch, capsys):
    """The .pth route: the partial activation at the runner stops the process with report.EXIT_NOT_ACTIVE (stack._wrap_runner)."""
    mod = _fake_runner_module(monkeypatch)
    stack._LEVERS = _FakeLevers({"trimul": "exact", **{lv: True for lv in EXACT.levers}, "sg": False}, full_account(EXACT))
    _armed(EXACT, allow_partial=False, trigger="runner")
    stack._wrap_runner(EXACT)
    with pytest.raises(SystemExit) as e:
        mod.InferenceRunner()
    assert e.value.code == R.EXIT_NOT_ACTIVE
    assert "[protenix-v1-opt] NOT ACTIVE: partial activation — sg: " in capsys.readouterr().err


def test_the_flag_is_the_cli_opt_out_the_variable_is_the_environment_routes(monkeypatch, tmp_path):
    """`pred` reads `--allow-partial` only: PROTENIX_V1_OPT_ALLOW_PARTIAL=1 in the caller's environment changes nothing on the CLI route
    (a partial arm refuses, exit 3). The variable is the ENVIRONMENT route's opt-out (`PROTENIX_V1_OPT=<mode> protenix pred`): the .pth
    hook reads it at interpreter start (_autoload.install) and hands it to enable(); it stays a declared package switch, stripped
    from the stock arm."""
    import protenix_v1_opt
    from protenix_v1_opt import _autoload as A
    assert stack.ENV_ALLOW_PARTIAL == A.ENV_ALLOW_PARTIAL == "PROTENIX_V1_OPT_ALLOW_PARTIAL" and stack.ENV_ALLOW_PARTIAL in stack.PACKAGE_ENV
    assert stack.stock_env_violations({"PROTENIX_V1_OPT_ALLOW_PARTIAL": "1"}) == ["PROTENIX_V1_OPT_ALLOW_PARTIAL"]
    acc = full_account(EXACT)
    acc["sampler"]["sampler"]["eager_steps"] = 4
    _stub_pred(monkeypatch, EXACT, acc)
    monkeypatch.setenv("PROTENIX_V1_OPT_ALLOW_PARTIAL", "1")
    assert cli.main(["pred", "--mode", "exact", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 3      # the CLI route ignores it
    seen = {}
    for env, want in (({A.ENV: "exact", A.ENV_ALLOW_PARTIAL: "1"}, True), ({A.ENV: "exact"}, False), ({A.ENV: "exact", A.ENV_ALLOW_PARTIAL: "true"}, False)):
        f = A.install(dict(env), exit=lambda code: None)
        try:
            assert f is not None and f.allow_partial is want, (env, f and f.allow_partial)
            monkeypatch.setattr(protenix_v1_opt, "enable", lambda mode, **kw: seen.update(kw) or {})
            f._fire("runner")                                                                                  # the hook hands the route's opt-out to enable()
            assert seen["allow_partial"] is want and seen["strict"] is True and seen["trigger"] == "runner"
        finally:
            if f in sys.meta_path:
                sys.meta_path.remove(f)


# ------------------------------------------------------------------------------------------------------------------ items
def test_items_line_grammar():
    assert R.items_line({"total": 2, "ok": 1, "failed": 1, "open": 0, "failed_items": ["8G5T:RuntimeError"]}) == \
        f"{R.PREFIX} ITEMS total=2 ok=1 failed=1 open=0 failed_items=8G5T:RuntimeError"
    assert R.items_line({"total": 1, "ok": 1, "failed": 0, "open": 0, "failed_items": []}).endswith("failed_items=-")
    assert R.items_line(None) == f"{R.PREFIX} ITEMS total=0 ok=0 failed=0 open=0 failed_items=-"


def test_items_exit_code_rule():
    assert R.items_exit_code(R.EXIT_OK, {"failed": 1, "open": 0}) == R.EXIT_FAIL == 1
    assert R.items_exit_code(R.EXIT_OK, {"failed": 0, "open": 1}) == R.EXIT_FAIL
    assert R.items_exit_code(R.EXIT_OK, {"failed": 0, "open": 0}) == R.EXIT_OK
    assert R.items_exit_code(R.EXIT_NOT_ACTIVE, {"failed": 1}) == R.EXIT_NOT_ACTIVE          # a non-zero code stands
    assert R.items_exit_code(R.EXIT_OK, None) == R.EXIT_OK


def test_predict_wrap_names_a_failed_item_and_reraises(monkeypatch, capsys):
    mod = types.ModuleType(stack.RUNNER_MODULE)

    class InferenceRunner:
        def __init__(self):
            self.configs = {"use_template": False}

        def predict(self, data):
            if data.get("boom"):
                raise RuntimeError("kernel exploded")
            return "ran"

    mod.InferenceRunner = InferenceRunner
    monkeypatch.setitem(sys.modules, "runner", sys.modules.get("runner") or types.ModuleType("runner"))
    monkeypatch.setitem(sys.modules, stack.RUNNER_MODULE, mod)
    monkeypatch.setattr(stack, "_RUNNERS", {"built": 0, "wrapped": False, "hook_registered": False})
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "mode": "fast"})
    monkeypatch.setattr(stack, "_ITEMS", [])
    orig_init = InferenceRunner.__init__
    stack._wrap_runner(FAST)
    InferenceRunner.__init__ = orig_init
    r = InferenceRunner()
    assert r.predict({"sample_name": "fine", "N_token": 3}) == "ran"
    with pytest.raises(RuntimeError):
        r.predict({"sample_name": "8G5T", "N_token": 1956, "boom": True})
    err = capsys.readouterr().err
    assert f"{R.PREFIX} ITEM event=failed item=8G5T N_token=1956 error=RuntimeError:_kernel_exploded" in err
    c = stack.items_census()
    assert (c["total"], c["ok"], c["failed"], c["open"]) == (2, 1, 1, 0) and c["failed_items"] == ["8G5T:RuntimeError"]


# ------------------------------------------------------------------------------------------------------------------- pred
def _stub_pred(monkeypatch, res, account, activation_partial=(), reason=None):
    """`pred` on a box without the stack: enable() arms a synthetic active report, the stock CLI is a stub that returns."""
    def fake_enable(mode, *, strict=False, trigger=None, det=False, allow_partial=None, n_gpu=None):
        allow = bool(allow_partial)
        stack._LEVERS = _FakeLevers({}, account)
        stack._REPORT = dict(stack._base(mode, trigger, det, allow), armed=True, active=True, arm=res.arm, line=res.line, tier=res.tier,
                             cfg={"trimul": res.trimul}, partial=list(activation_partial), partial_reason=reason, levers={})
        return dict(stack._REPORT)
    monkeypatch.setattr(protenix_v1_opt, "enable", fake_enable)
    monkeypatch.setattr(cli.K, "frozen_weights_check", lambda *a, **k: {})       # the weights boot gate admits (test_frozen_weights.py holds it to its contract): the exit rule runs with or without the pinned stock package on the box
    bi = types.ModuleType("runner.batch_inference")
    bi.protenix_cli = types.SimpleNamespace(main=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "runner", types.ModuleType("runner"))
    monkeypatch.setitem(sys.modules, "runner.batch_inference", bi)


def test_pred_partial_exits_3_with_the_line(monkeypatch, capsys, tmp_path):
    acc = full_account(FAST)
    acc["counts"]["trimul"]["error:RuntimeError"] = 2
    _stub_pred(monkeypatch, FAST, acc)
    rc = cli.main(["pred", "--mode", "fast", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 3
    err = capsys.readouterr().err.splitlines()
    assert "[protenix-v1-opt] NOT ACTIVE: partial activation — fast: fast fallback {'error:RuntimeError': 2}; exit 3 (--allow-partial records and proceeds)" in err
    assert stack.refresh_levers()["levers"]["counts"]["trimul"]["error:RuntimeError"] == 2      # the kit's account after the run
    assert not (tmp_path / "opt_manifest.json").exists()                                        # nothing but the stock CLI's outputs is written


def test_pred_partial_with_allow_partial_exits_0_and_records_it(monkeypatch, capsys, tmp_path):
    acc = full_account(FAST)
    acc["counts"]["trimul"]["error:RuntimeError"] = 2
    _stub_pred(monkeypatch, FAST, acc)
    rc = cli.main(["pred", "--mode", "fast", "--allow-partial", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 0
    err = capsys.readouterr().err.splitlines()
    assert "[protenix-v1-opt] PARTIAL allowed: fast: fast fallback {'error:RuntimeError': 2} (--allow-partial, recorded)" in err
    assert not any("NOT ACTIVE" in l for l in err)


def test_pred_with_a_failed_item_exits_1_and_does_not_print_the_partial_line(monkeypatch, capsys, tmp_path):
    """`pred` whose one item raised inside `predict` (the stock's per-item handler continues and the CLI returns 0): the sampler lever served
    no call past the failure — exit 1 (the failure's), the NOTE line, the ITEMS census; never exit 3 / `NOT ACTIVE: partial activation`."""
    acc = full_account(FAST)
    acc["sampler"]["sampler"] = {"captures": 0, "replays": 0, "eager_steps": 0, "bypass": 0}
    _stub_pred(monkeypatch, FAST, acc)
    monkeypatch.setattr(stack, "_ITEMS", [{"item": "n2000", "N_token": 2000, "failed": "OutOfMemoryError: CUDA out of memory."}])
    rc = cli.main(["pred", "--mode", "fast", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == R.EXIT_FAIL == 1
    err = capsys.readouterr().err.splitlines()
    assert not any("NOT ACTIVE" in l for l in err), err
    assert "[protenix-v1-opt] NOTE exit rule not judged: 1 item(s) failed or open (n2000:OutOfMemoryError) — sg,sampler_prep served no call past the failure; the exit code is the failure's (1)" in err
    assert "[protenix-v1-opt] ITEMS total=1 ok=0 failed=1 open=0 failed_items=n2000:OutOfMemoryError" in err
    assert lever_facts(err, "sg")["state"] == "skipped" and lever_facts(err, "sg")["reason"] == "served0"
    assert err[-1] == "[protenix-v1-opt] EXIT mode=fast rc=1 n_gpu=1 sharding=none"
    rc = cli.main(["pred", "--mode", "fast", "--allow-partial", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 1 and not any("PARTIAL allowed" in l for l in capsys.readouterr().err.splitlines())   # the allowance is moot on a failed run


def lever_facts(err_lines, lever):
    """The k=v facts of the one printed LEVER line whose `lever=` word is `lever` (report.lever_line's grammar: blank-separated k=v after `LEVER`)."""
    rows = [dict(kv.split("=", 1) for kv in l.split(" LEVER ", 1)[1].split()) for l in err_lines if l.startswith(R.PREFIX + " LEVER ")]
    hits = [r for r in rows if r.get("lever") == lever]
    assert len(hits) == 1, (lever, rows)
    return hits[0]


def test_pred_allow_partial_is_recorded_on_the_printed_lines(monkeypatch, capsys, tmp_path):
    acc = full_account(EXACT)
    acc["sampler"]["sampler"]["eager_steps"] = 4
    _stub_pred(monkeypatch, EXACT, acc)
    assert cli.main(["pred", "--mode", "exact", "--allow-partial", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err.splitlines()
    assert "[protenix-v1-opt] PARTIAL allowed: sg: sg fallback {'eager_steps': 4} (--allow-partial, recorded)" in err      # the allowance and the partial lever, printed
    assert not any("NOT ACTIVE" in l for l in err)
    sg = lever_facts(err, "sg")
    assert sg["state"] == "skipped" and sg["reason"] == "fallback"                                                          # the lever's own line says why
    assert "[protenix-v1-opt] EXIT mode=exact rc=0 n_gpu=1 sharding=none" in err
    assert not (tmp_path / "o").exists() or list((tmp_path / "o").iterdir()) == []                                        # nothing is written: the lines are the record


def test_pred_full_activation_exits_0_without_a_partial_line(monkeypatch, capsys, tmp_path):
    _stub_pred(monkeypatch, FAST, full_account(FAST))
    assert cli.main(["pred", "--mode", "fast", "--input", "x.json", "--out_dir", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "PARTIAL" not in err and "NOT ACTIVE" not in err
    rep = stack.refresh_levers(); res = modes.resolve("fast")
    v = R.verdict(rep, res.trimul, res.levers, False, run_ok=True)
    assert v["partial"] == [] and v["partial_reason"] is None and v["allow_partial"] is False and not v["exit_code"]
    assert set(v["evidence"]) == {"fast", "gflash", "tricuda", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused"}


def test_pred_activation_partial_admitted_is_on_the_printed_lines(monkeypatch, capsys, tmp_path):
    reason = "the kit applied {...}: hoist=False expected True"
    _stub_pred(monkeypatch, EXACT, full_account(EXACT), activation_partial=["hoist"], reason=reason)
    assert cli.main(["pred", "--mode", "exact", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 3
    err = capsys.readouterr().err.splitlines()
    assert f"[protenix-v1-opt] NOT ACTIVE: partial activation — hoist: {reason}; exit 3 (--allow-partial records and proceeds)" in err
    assert "[protenix-v1-opt] EXIT mode=exact rc=3 n_gpu=1 sharding=none" in err
    stack._REPORT, stack._LEVERS = None, None                                                                                 # a second process
    assert cli.main(["pred", "--mode", "exact", "--allow-partial", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err.splitlines()
    assert f"[protenix-v1-opt] PARTIAL allowed: hoist: {reason} (--allow-partial, recorded)" in err
    assert not any("NOT ACTIVE" in l for l in err) and "[protenix-v1-opt] EXIT mode=exact rc=0 n_gpu=1 sharding=none" in err


def test_pred_printed_lines_carry_the_partial_lever(monkeypatch, capsys, tmp_path):
    acc = full_account(FAST)
    acc["sampler"]["sampler"]["eager_steps"] = 1
    _stub_pred(monkeypatch, FAST, acc)
    assert cli.main(["pred", "--mode", "fast", "--allow-partial", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err.splitlines()
    assert "[protenix-v1-opt] PARTIAL allowed: sg: sg fallback {'eager_steps': 1} (--allow-partial, recorded)" in err
    sg = lever_facts(err, "sg")
    assert sg["state"] == "skipped" and sg["reason"] == "fallback"
    for lv in ("fast", "gflash", "ttr", "hoist"):                                                                      # every other lever of the mode served: its line says on
        assert lever_facts(err, lv)["state"] == "on", lv
    assert "[protenix-v1-opt] EXIT mode=fast rc=0 n_gpu=1 sharding=none" in err


def test_pred_refused_at_activation_exits_3_and_writes_nothing(monkeypatch, capsys, tmp_path):
    def refuse(mode, **kw):
        stack._REPORT = dict(stack._base(mode, None, False, bool(kw.get("allow_partial"))), reason="gate")
        raise protenix_v1_opt.ActivationError("gate")
    monkeypatch.setattr(protenix_v1_opt, "enable", refuse)
    monkeypatch.setattr(cli.K, "frozen_weights_check", lambda *a, **k: {})       # the weights boot gate admits; the refusal under test is the activation's
    assert cli.main(["pred", "--mode", "fast", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 3
    err = capsys.readouterr().err.splitlines()
    assert "[protenix-v1-opt] EXIT mode=fast rc=3 n_gpu=1 sharding=none" in err
    assert not any(" LEVER " in l or "PARTIAL" in l for l in err) and not (tmp_path / "o").exists()      # a refused activation ran nothing: no lever line, no directory


# ---------------------------------------------------------------------------------------------------------- check / warm
def test_check_records_the_flag_and_has_no_partial_plan(capsys):
    rc = cli.main(["check", "--mode", "fast", "--allow-partial"])
    out = json.loads(capsys.readouterr().out)
    assert out["allow_partial"] is True and out["partial"] == [] and rc in (0, 3)
    cli.main(["check", "--mode", "off"])
    assert json.loads(capsys.readouterr().out)["allow_partial"] is False


def test_warm_always_proceeds_past_a_partial(monkeypatch, tmp_path):
    """warm fills the caches and loads the weights on the kit's small warm input: a lever whose size gate keeps it off every call there is
    RECORDED (the PARTIAL line of pred --allow-partial) and the verb exits 0 — the activation verdict is check's / pred's. The flag is passed
    with or without --allow-partial on warm's own command line (accepted, redundant)."""
    seen = {}
    monkeypatch.setattr(cli, "cmd_pred", lambda argv: seen.update(argv=argv) or 0)
    assert cli.main(["warm", "--mode", "exact", "--allow-partial", "--out_dir", str(tmp_path)]) == 0
    assert seen["argv"].count("--allow-partial") == 1
    assert cli.main(["warm", "--mode", "exact", "--out_dir", str(tmp_path)]) == 0
    assert "--allow-partial" in seen["argv"]


# -------------------------------------------------------------------------------------------------- the environment route's record
def test_hook_exit_record_prints_the_lines(monkeypatch, capsys, tmp_path):
    acc = full_account(FAST)
    acc["sampler"]["sampler"]["eager_steps"] = 3
    stack._LEVERS = _FakeLevers({}, acc)
    stack._REPORT = dict(stack._base("fast", "runner", False, False), armed=True, active=True, arm=FAST.arm, line=FAST.line, tier=FAST.tier, cfg={}, levers={})
    monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", "x.json", "--out_dir", str(tmp_path)])
    v = stack._hook_exit_record()
    assert v["partial"] == ["sg"] and v["exit_code"] is None
    err = capsys.readouterr().err.splitlines()
    assert R.partial_exit_line(["sg"], v["partial_reason"]) in err and "[protenix-v1-opt] EXIT the exit code is the stock CLI's on the environment route (`pred` is the gated form)" in err
    assert v["allow_partial"] is False and list(tmp_path.iterdir()) == []                    # nothing written beside the stock outputs
