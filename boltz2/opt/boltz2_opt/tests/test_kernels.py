"""The KERNELS census (boltz2_opt/kernels.py) and its REQUIRE guard, CPU-only: the line grammar, the expectations as ONE function of the mode
table, the stock caller's and the worker launcher's options, the refusal path (exit 5, accelerator + route named) on a stubbed absent /
fallback reading, the per-call census (byrule counted, not refused; a reference path where the rules say served -> refused; a fallback
signal at S > threshold -> refused), and the parent's exit rules over the record (report.kernels_exit) through the stubbed worker route."""
import json
import logging
import os
import re
import warnings

import pytest

from .. import manifest as mf
from .. import cli, det, kernels, modes, report as rep, rowpair, settings, stack, worker
from . import _stubs


@pytest.fixture(autouse=True)
def fresh():
    kernels.reset()
    yield
    kernels.reset()


class _Model:                       # a stand-in for boltz.model.models.boltz2.Boltz2: the census wraps predict_step and reads use_kernels
    use_kernels = True

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return {"exception": False}


def _present(monkeypatch, probe_ok=True):
    """Stub the library as importable and (optionally) the probe as served: the engaged path without a GPU."""
    monkeypatch.setattr(kernels, "install_counters", lambda: {a: "" for a in kernels.ACCELERATORS})
    monkeypatch.setattr(kernels, "static_facts", lambda: {"dists": {"cuequivariance-torch": "0.10.0", "cuequivariance-ops-torch-cu13": "0.10.0"}, "build": "cu13", "version": "0.10.0",
                                                        "specs": {kernels.OPS[a]["ops_module"]: f"/sp/{kernels.OPS[a]['ops_module']}.py" for a in kernels.ACCELERATORS}, "cubin_lib": None, "cubin_sm_tags": ["sm_90"]})
    monkeypatch.setattr(kernels, "probe", lambda device=None: {a: {"ok": probe_ok, "path": "served" if probe_ok else "byrule", "ms": 1.0, "error": None if probe_ok else "probe call at S=128 took path=byrule", "S": 128} for a in kernels.ACCELERATORS})
    monkeypatch.setattr(kernels, "device_facts", lambda: {"index": 0, "name": "NVIDIA H100 80GB HBM3", "cc": [9, 0], "torch": "2.12.0+cu130", "cuda": "13.0"})


def _install(route="exact", expect="cueq_triatt=engaged,cueq_trimul=engaged", tmp_path=None, **kw):
    kernels.install(route, kernels.parse_expect(expect), settings=kw.pop("settings", "defaults"), mode=kw.pop("mode", None), model_cls=_Model,
                    json_path=str(tmp_path / "census.json") if tmp_path else None, **kw)


# ---------------------------------------------------------------- grammar ----------------------------------------------------------------
def test_expect_spec_round_trips_and_refuses_unknown_names_and_kinds():
    e = kernels.parse_expect("cueq_triatt=engaged,cueq_trimul=off-by-route:replaced_by_rowpair")
    assert e == {"cueq_triatt": "engaged", "cueq_trimul": "off-by-route:replaced_by_rowpair"} and kernels.parse_expect(kernels.format_expect(e)) == e
    for bad in ("cueq_triatt=engaged", "cueq_triatt=engaged,cueq_trimul=absent", "flash=engaged,cueq_triatt=engaged,cueq_trimul=engaged", "cueq_triatt=engaged,cueq_triatt=engaged,cueq_trimul=engaged"):
        with pytest.raises(ValueError):
            kernels.parse_expect(bad)
    assert [kernels.word_kind(w) for w in ("engaged:x@1-cu13[served=1,byrule=0]", "off-by-route:--no_kernels", "absent:m", "fallback:probe:x", "n/a-upstream:no_caller", "weird")] == \
           ["engaged", "off-by-route", "absent", "fallback", "n/a-upstream", "?"]


def test_line_grammar_route_token_and_parse_round_trip():
    words = {"cueq_triatt": "engaged:cuequivariance_ops_torch@0.10.0-cu13[served=96,byrule=0]", "cueq_trimul": "engaged:cuequivariance_ops_torch@0.10.0-cu13[served=8,byrule=4(S<=100=4)]"}
    for route in ("stock", "default", "exact", "fast", "big", "big_x2"):
        line = kernels.format_line("m", route, "defaults", words, "PASS", ["trifast=n/a-upstream:no_caller", "device=sm90"])
        assert re.match(r"^\[boltz2-opt m\] KERNELS route=" + route + r" settings=defaults cueq_triatt=\S+ cueq_trimul=\S+ trifast=n/a-upstream:no_caller device=sm90 verdict=PASS$", line), line
        assert kernels.LINE_MARK in line and "KERNELS route=" == kernels.LINE_MARK, "one cross-engine grep `KERNELS route=` finds every pass"
        p = kernels.parse_line(line)
        assert p["route"] == route and p["words"] == words and p["verdict"] == "PASS" and p["tokens"]["device"] == "sm90"
    assert kernels.ROUTE_RE.match("big_x8") and not kernels.ROUTE_RE.match("off") and not kernels.ROUTE_RE.match("big_x")
    refused = "[boltz2-opt exact] " + kernels.REFUSED_MARK + "exact settings=defaults cueq_triatt=absent:x (expected engaged); exit 5"
    assert kernels.find_lines("noise\n" + refused + "\n" + line + "\n") == [kernels.parse_line(line)], "the point-of-refusal line is not a pass line"
    assert kernels.parse_line("[boltz2-opt x] KERNELS something else") is None


def test_route_words():
    assert kernels.route_word("off") == "stock" == kernels.route_word("off", 1), "the stock CLI is the KERNELS route `stock` (its settings token says defaults | flags)"
    assert [kernels.route_word(m) for m in ("exact", "fast", "big")] == ["exact", "fast", "big"] and kernels.route_word("big", 2) == "big_x2" and kernels.route_word("big", 8) == "big_x8"


# ---------------------------------------------------------------- expectations: ONE function of the mode table ----------------------------------------------------------------
def test_expectations_follow_the_mode_table_and_upstreams_own_switch():
    assert set(kernels.ACCELERATORS) == {"cueq_triatt", "cueq_trimul"}
    for m in modes.MODE_NAMES:
        if m == "off":
            continue
        assert modes.resolve(m)["kernels"] == "on" and modes.worker_args(m)["kernels"] == "on", m
        assert modes.kernels_expected(m) == {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}, f"{m}: kernels on in the row -> both cuEquivariance accelerators expected engaged"
    assert modes.kernels_expected("off", stock_args=["predict", "x.yaml", "--out_dir", "o"]) == {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}, "the stock CLI as shipped: kernels on (main.py:1321)"
    off = {"cueq_triatt": "off-by-route:--no_kernels", "cueq_trimul": "off-by-route:--no_kernels"}
    assert modes.kernels_expected("off", stock_args=["predict", "x.yaml", "--no_kernels"]) == off and modes.KERNELS_OFF_FLAG == "--no_kernels"
    assert modes.kernels_expected("off", stock_args=["predict", "x.yaml"] + settings.stock_argv({"num_workers": 1, "no_kernels": True})) == off, "--no_kernels given: kernels off by route, named"
    assert modes.kernels_expected("off", stock_args=["predict", "x.yaml"] + det.stock_args(1, ["predict", "x.yaml"])) == off, "the det recipe (--det 1) adds --no_kernels: kernels off by route, named"
    assert modes.kernels_expected("off", stock_args=["predict", "x.yaml"] + settings.stock_argv({"recycling_steps": 10, "seed": 3})) == {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}


def test_n_gpu_expectations_name_what_the_row_sharded_stack_replaces():
    assert modes.KERNELS_TP_REPLACED == {"cueq_trimul": "replaced_by_rowpair"} and set(modes.KERNELS_TP_REPLACED) <= set(kernels.ACCELERATORS)
    assert "cuequivariance_trimul" in rowpair.REPLACED_LEVERS, "rowpair replaces the cuEquivariance TriMul (whole-tensor kernel) at P > 1: the expectation's source"
    assert modes.kernels_expected("big", n_gpu=2) == {"cueq_triatt": "engaged", "cueq_trimul": "off-by-route:replaced_by_rowpair"} == modes.kernels_expected("big", n_gpu=8)
    assert modes.kernels_expected("big", n_gpu=1) == {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}


def test_stock_command_and_worker_launch_carry_the_expectations():
    y = "/tmp/x.yaml"
    for flags, level, word, exp in (({"seed": 3}, None, "flags", "cueq_triatt=engaged,cueq_trimul=engaged"), ({}, None, "defaults", "cueq_triatt=engaged,cueq_trimul=engaged"),
                                    ({"recycling_steps": 3, "no_kernels": True, "seed": 3}, None, "flags", "cueq_triatt=off-by-route:--no_kernels,cueq_trimul=off-by-route:--no_kernels"),
                                    ({"seed": 3}, 1, "flags+det1", "cueq_triatt=off-by-route:--no_kernels,cueq_trimul=off-by-route:--no_kernels")):
        argv, _env = cli.stock_command(y, "/tmp/o", flags, [], None, level)
        head = argv[:argv.index("--")]; route = "stock"
        assert head[head.index("--kernels-route") + 1] == route and head[head.index("--kernels-expect") + 1] == exp, (flags, head)
        assert head[head.index("--kernels-settings") + 1] == word and head[head.index("--kernels-json") + 1] == os.path.join("/tmp/o", cli.KERNELS_CENSUS)
        kw, rest = kernels.parse_cli_opts(head)
        assert kw["route"] == route and kw["expected"] == kernels.parse_expect(exp) and "--kernels-route" not in rest and "--proof-json" not in rest
    for m in ("exact", "fast", "big"):
        opts = stack.launch_opts(m); kw, rest = kernels.parse_cli_opts(opts)
        assert kw["route"] == m and kw["expected"] == modes.kernels_expected(m) and kw["settings"] == "defaults" and kw["n_gpu"] == 1 and kw["mode"] == m and "--kernels-expect" not in rest
        assert stack.kernels_opts(m) == opts[len(opts) - len(stack.kernels_opts(m)):], "the KERNELS options close the launcher's option list"
    kw2, _ = kernels.parse_cli_opts(stack.launch_opts("big", n_gpu=2))
    assert kw2["route"] == "big_x2" and kw2["n_gpu"] == 2 and kw2["expected"]["cueq_trimul"] == "off-by-route:replaced_by_rowpair"
    assert kernels.parse_cli_opts(["--route", "fpf_trimul", "--attach", "templ"]) == ({}, ["--route", "fpf_trimul", "--attach", "templ"]), "no KERNELS options: nothing installed, argv untouched"
    with pytest.raises(ValueError):
        kernels.parse_cli_opts(["--kernels-route", "exact"])


# ---------------------------------------------------------------- the REQUIRE guard (in-process, exit 5) ----------------------------------------------------------------
def test_absent_library_refuses_before_the_first_step_with_exit_5(tmp_path, monkeypatch, capsys):
    """A box without cuequivariance (the import verdict stubbed `absent`, as install_counters words it): the guard refuses at the first
    predict_step naming both accelerators and the route, exits 5, and the exit-time line / record carry verdict REFUSED."""
    monkeypatch.setattr(kernels, "install_counters", lambda: {a: "absent:cuequivariance_ops_torch(ModuleNotFoundError)" for a in kernels.ACCELERATORS})
    monkeypatch.setattr(kernels, "device_facts", lambda: {"cc": None})
    _install("exact", tmp_path=tmp_path)
    with pytest.raises(SystemExit) as ex:
        _Model().predict_step({}, 0)
    assert ex.value.code == kernels.EXIT_KERNELS == rep.EXIT_KERNELS == 5
    out = capsys.readouterr().out
    assert "[boltz2-opt exact] KERNELS-REFUSED route=exact settings=defaults cueq_triatt=absent:" in out and "cueq_trimul=absent:" in out and "(expected engaged)" in out and "; exit 5" in out, out
    rec = kernels.emit()
    assert rec["verdict"] == "REFUSED(cueq_triatt,cueq_trimul)" and all(kernels.word_kind(w) == "absent" for w in rec["words"].values())
    line = capsys.readouterr().out.strip()
    assert line.startswith("[boltz2-opt exact] KERNELS route=exact settings=defaults cueq_triatt=absent:") and line.endswith("verdict=REFUSED(cueq_triatt,cueq_trimul)"), line
    assert json.load(open(tmp_path / "census.json"))["verdict"] == rec["verdict"], "the record beside the outputs says what the line says"


def test_use_kernels_false_on_an_engaged_route_is_a_fallback_refusal(tmp_path, monkeypatch, capsys):
    _present(monkeypatch); _install("stock", mode="stock", settings="upstream", tmp_path=tmp_path)
    m = _Model(); m.use_kernels = False                       # boltz2.py:361-366 turned the kernels off (no CUDA / cc < 8): silent upstream, a refusal here
    with pytest.raises(SystemExit) as ex:
        m.predict_step({}, 0)
    assert ex.value.code == 5
    assert "KERNELS-REFUSED route=stock settings=upstream cueq_triatt=fallback:use_kernels=False(boltz2.py:361-366) (expected engaged)" in capsys.readouterr().out


def test_probe_reference_path_is_a_refusal(tmp_path, monkeypatch, capsys):
    _present(monkeypatch, probe_ok=False); _install("fast", tmp_path=tmp_path)
    with pytest.raises(SystemExit) as ex:
        _Model().predict_step({}, 0)
    assert ex.value.code == 5 and "cueq_triatt=fallback:probe:" in capsys.readouterr().out


def test_engaged_pass_prints_one_line_with_the_census(tmp_path, monkeypatch, capsys):
    _present(monkeypatch); _install("fast", tmp_path=tmp_path)
    assert _Model().predict_step({}, 0) == {"exception": False}, "the wrapper delegates after the guard passes"
    kernels._COUNTS["cueq_triatt"].update(calls=96, served=96); kernels._COUNTS["cueq_trimul"].update(calls=12, served=8, byrule={"S<=100": 4})
    rec = kernels.emit(); rec2 = kernels.emit()
    out = capsys.readouterr().out.strip().splitlines()
    assert len([l for l in out if kernels.LINE_MARK in l]) == 1, "ONE line per pass (emit is idempotent)"
    p = kernels.parse_line(out[-1])
    assert p["route"] == "fast" and p["verdict"] == "PASS" and p["words"]["cueq_triatt"] == "engaged:cuequivariance_ops_torch@0.10.0-cu13[served=96,byrule=0]"
    assert p["words"]["cueq_trimul"] == "engaged:cuequivariance_ops_torch@0.10.0-cu13[served=8,byrule=4(S<=100=4)]", "byrule calls are named and counted, not refused"
    assert p["tokens"]["trifast"] == "n/a-upstream:no_caller" and p["tokens"]["use_kernels"] == "True" and p["tokens"]["device"] == "sm90" and p["tokens"]["probe"].startswith("triatt:ok@")
    assert rec["verdict"] == rec2["verdict"] == "PASS" and json.load(open(tmp_path / "census.json"))["words"] == p["words"]


def test_off_by_route_words_stand_when_nothing_calls_the_accelerator(tmp_path, monkeypatch, capsys):
    _present(monkeypatch); _install("stock", expect="cueq_triatt=off-by-route:--no_kernels,cueq_trimul=off-by-route:--no_kernels", mode="stock", settings="upstream+det1", tmp_path=tmp_path)
    m = _Model(); m.use_kernels = False; m.predict_step({}, 0)
    rec = kernels.emit()
    assert rec["verdict"] == "PASS" and rec["words"] == {"cueq_triatt": "off-by-route:--no_kernels", "cueq_trimul": "off-by-route:--no_kernels"} and rec["probe"] == {}, "kernels off by route: no probe, no import, the words stand"
    kernels.reset(); _present(monkeypatch); _install("big_x2", expect="cueq_triatt=engaged,cueq_trimul=off-by-route:replaced_by_rowpair", n_gpu=2, rank=0, tmp_path=tmp_path)
    _Model().predict_step({}, 0)
    kernels._COUNTS["cueq_trimul"].update(calls=3, served=3)      # something called the replaced TriMul anyway: a contradiction only the totals show -> verdict REFUSED (the parent exits 5)
    rec = kernels.emit()
    assert rec["verdict"] == "REFUSED(cueq_trimul)" and rec["words"]["cueq_trimul"].startswith("engaged:") and rec["refused"][0]["expected"] == "off-by-route:replaced_by_rowpair"
    assert " rank=0 " in rec["line"] + " " and "route=big_x2" in rec["line"] and "n_gpu=2" in rec["line"]


# ---------------------------------------------------------------- the per-call census ----------------------------------------------------------------
class _FakeOps:
    """A stand-in for the ops layer: `public` routes to the module-global `reference` when S <= threshold (the library's rule), else 'serves'."""
    CUEQ_TRIATTN_FALLBACK_THRESHOLD = 100

    def __init__(self):
        self.reference_calls = 0

    def reference(self, q, *a, **k):
        self.reference_calls += 1; return "ref"

    def public(self, q, *a, **k):
        return self.reference_bound(q) if q.shape[-2] <= self.CUEQ_TRIATTN_FALLBACK_THRESHOLD or k.get("force_ref") else "kernel"


class _T:                               # a tensor stand-in: shape and dtype are all the census reads
    def __init__(self, S, hd=32, dtype="torch.bfloat16"):
        self.shape = (1, 4, 4, S, hd); self.dtype = dtype


def _wire(monkeypatch, ops):
    import sys, types
    mod = types.ModuleType(kernels.OPS["cueq_triatt"]["ops_module"]); mod.CUEQ_TRIATTN_FALLBACK_THRESHOLD = ops.CUEQ_TRIATTN_FALLBACK_THRESHOLD
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    ops.reference_bound = kernels._wrap_reference("cueq_triatt", ops.reference)
    return kernels._wrap_public("cueq_triatt", ops.public)


def test_byrule_calls_are_counted_not_refused_and_served_calls_are_counted(tmp_path, monkeypatch):
    _present(monkeypatch); _install("exact", tmp_path=tmp_path); _Model().predict_step({}, 0)
    ops = _FakeOps(); public = _wire(monkeypatch, ops)
    assert [public(_T(S)) for S in (64, 100, 101, 400, 1000)] == ["ref", "ref", "kernel", "kernel", "kernel"]
    c = kernels._COUNTS["cueq_triatt"]
    assert (c["calls"], c["served"], c["byrule"], c["violations"]) == (5, 3, {"S<=100": 2}, 0) and ops.reference_calls == 2
    assert kernels.words()["cueq_triatt"] == "engaged:cuequivariance_ops_torch@0.10.0-cu13[served=3,byrule=2(S<=100=2)]" and kernels.emit()["verdict"] == "PASS"
    assert kernels.byrule_reason("cueq_triatt", (_T(150, hd=16),), {}) == ("S<=200_hidden_dim=16", True), "hidden_dim < 32 raises the library threshold to 200: by rule"
    assert kernels.byrule_reason("cueq_triatt", (_T(300, hd=48, dtype="torch.float32"),), {}) == ("hidden_dim=48_float32", True), "fp32 supports hidden_dim <= 32 only: by rule"
    assert kernels.byrule_reason("cueq_triatt", (_T(300),), {}) == ("reference@S=300>100", False), "bf16, hidden 32, S > threshold: the rules say served — a reference call is a violation"


def test_reference_path_where_the_rules_say_served_is_refused(tmp_path, monkeypatch, capsys):
    _present(monkeypatch); _install("exact", tmp_path=tmp_path); _Model().predict_step({}, 0)
    ops = _FakeOps(); public = _wire(monkeypatch, ops)
    with pytest.raises(SystemExit) as ex:
        public(_T(300), force_ref=True)               # S = 300 > 100, bf16, hidden 32: served by the rules — the reference path here is a fallback
    assert ex.value.code == 5
    out = capsys.readouterr().out
    assert "KERNELS-REFUSED route=exact settings=defaults cueq_triatt=fallback:reference@S=300>100 (expected engaged); exit 5" in out, out
    assert kernels.emit()["words"]["cueq_triatt"].startswith("fallback:reference@S=300>100[served=0,byrule=0")


def test_fallback_signal_above_threshold_refuses_below_threshold_is_recorded(tmp_path, monkeypatch, capsys):
    _present(monkeypatch); _install("fast", tmp_path=tmp_path); _Model().predict_step({}, 0)
    ops = _FakeOps()

    def public(q, *a, **k):                           # the library warns FALLING BACK from inside its own file while serving
        warnings.warn_explicit("Non-SM100f device: FALLING BACK to the slow PyTorch reference path", UserWarning, filename="/sp/cuequivariance_ops_torch/triangle_attention.py", lineno=180, module="cuequivariance_ops_torch.triangle_attention")
        return "kernel"
    wrapped = kernels._wrap_public("cueq_triatt", public)
    monkeypatch.setitem(__import__("sys").modules, kernels.OPS["cueq_triatt"]["ops_module"], type("M", (), {"CUEQ_TRIATTN_FALLBACK_THRESHOLD": 100}))
    assert wrapped(_T(50)) == "kernel" and len(kernels._STATE["signals"]) == 1 and kernels._STATE["signals"][0]["S"] == 50, "at S <= threshold the fallback word belongs to byrule: recorded, not refused"
    with pytest.raises(SystemExit) as ex:
        wrapped(_T(400))
    assert ex.value.code == 5 and "cueq_triatt=fallback:signal@S=400>100:" in capsys.readouterr().out
    kernels.reset(); _present(monkeypatch); _install("fast", tmp_path=tmp_path); _Model().predict_step({}, 0)
    wrapped2 = kernels._wrap_public("cueq_trimul", lambda x, **k: (logging.getLogger("cuequivariance_ops_torch.triangle_multiplicative_update").warning("kernel unavailable, falling back to torch"), "kernel")[1])
    monkeypatch.setitem(__import__("sys").modules, kernels.OPS["cueq_trimul"]["ops_module"], type("M", (), {"CUEQ_TRIMUL_FALLBACK_THRESHOLD": 100}))
    with pytest.raises(SystemExit):
        wrapped2(_T(512))                             # a logging record from the library is the same signal as a warning
    assert kernels._STATE["signals"][-1]["source"].startswith("log:cuequivariance_ops_torch")


def test_install_counters_words_a_missing_library_absent(monkeypatch):
    import builtins
    real_import = builtins.__import__
    monkeypatch.setattr(kernels.importlib, "import_module", lambda name, *a, **k: (_ for _ in ()).throw(ModuleNotFoundError(f"No module named {name!r}")) if name.startswith("cuequivariance") else real_import(name))
    assert kernels.install_counters() == {a: "absent:cuequivariance_ops_torch(ModuleNotFoundError)" for a in kernels.ACCELERATORS}


def test_static_facts_on_this_box_read_absent_without_importing_torch():
    import sys
    f = kernels.static_facts()
    assert set(f["dists"]) >= {"cuequivariance-torch", "cuequivariance-ops-torch-cu13", "cuequivariance-ops-cu13", "cuequivariance"}
    assert f["build"] in (None, "cu13", "cu12") and isinstance(f["cubin_sm_tags"], list)
    assert kernels.cubin_sm_tags(None) == [] and kernels.cubin_sm_tags("/nonexistent.so") == []


def test_cubin_tags_are_read_from_the_shared_object_bytes(tmp_path):
    p = tmp_path / "libcue_ops.so"; p.write_bytes(b"\x7fELF....text.sm_90..sm_80 sm_100a ... sm_90a ... sm_121a")
    assert kernels.cubin_sm_tags(str(p)) == ["80", "90", "90a", "100a", "121a"]


# ---------------------------------------------------------------- the parent's exit rules (report.kernels_exit) and the routes end to end (stubbed worker) ----------------------------------------------------------------
def test_kernels_exit_rules(capsys):
    assert rep.kernels_exit("exact", 5, [], "x") == 5 and "KERNELS REQUIRE refused in the worker process" in capsys.readouterr().out
    assert rep.kernels_exit("off", 5, [{"verdict": "PASS"}], "x") == 5 and "in the stock process" in capsys.readouterr().out
    assert rep.kernels_exit("exact", 0, [], "census.json") == rep.EXIT_NOT_ACTIVE and "NOT ACTIVE: no KERNELS census (census.json)" in capsys.readouterr().out
    assert rep.kernels_exit("exact", 1, [], "x") == 1, "a failed process keeps its own code"
    assert rep.kernels_exit("exact", 0, [{"verdict": "PASS", "route": "exact"}], "x") == 0 and capsys.readouterr().out == ""
    assert rep.kernels_exit("big", 0, [{"verdict": "PASS"}, {"verdict": "REFUSED(cueq_trimul)", "route": "big_x2", "refused": [{"accelerator": "cueq_trimul", "word": "engaged:x[served=3,byrule=0]", "expected": "off-by-route:replaced_by_rowpair"}]}], "x") == 5
    assert "KERNELS REQUIRE refused: route=big_x2 verdict=REFUSED(cueq_trimul) cueq_trimul=engaged:x[served=3,byrule=0] (expected off-by-route:replaced_by_rowpair); exit 5" in capsys.readouterr().out
    assert rep.kernels_exit("off", 0, [{"verdict": "NO-STEP", "route": "stock", "words": {"cueq_triatt": "engaged:x[served=0,byrule=0]"}}], "x") == 5, "a pass whose model never stepped proves nothing: refused"


def test_pred_worker_route_echoes_the_line_records_the_census_and_exits_by_the_rules(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); monkeypatch.setenv("BOLTZ2_OPT", "fast"); ys = _stubs.write_yamls(str(tmp_path))
    _stubs.install_fake_stage(monkeypatch, kernels="pass")
    rc = worker.run("fast", ys, str(tmp_path / "o1"), [0], tag="t"); out = capsys.readouterr().out
    assert rc == 0, out
    lines = [l for l in out.splitlines() if kernels.LINE_MARK in l]
    assert len(lines) == 1 and lines[0].startswith("[boltz2-opt fast] KERNELS route=fast settings=defaults cueq_triatt=engaged:") and lines[0].endswith("verdict=PASS"), out
    m = mf.LAST
    assert m["kernels_census"]["verdict"] == "PASS" and m["evidence"]["kernels"]["route"] == "fast" and m["rc"] == 0
    rel = [l for l in out.splitlines() if l.startswith("PHASE item=") or l.startswith("[boltz2-opt] PEAK item=")]
    assert len(rel) == 2 * len(ys) and re.match(r"\[boltz2-opt\] PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$", rel[1]), (rel, out)   # the worker's per-item PHASE + PEAK lines relayed verbatim on pred's stdout (a caller reads one transcript for both routes)
    klines = [l for l in out.splitlines() if l.startswith("[boltz2-opt fast] KERNELS route=fast ")]
    assert len(klines) == 1 and "verdict=PASS" in klines[0], (klines, out)                       # the worker's KERNELS pass line relayed verbatim (worker.RELAY_LINES: KERNELS, KERNELS-REFUSED, PHASE, PEAK — one function)
    wtext = open(tmp_path / "o1" / "t_worker.log").read()
    assert worker.relay_lines(wtext, []) == [l for l in wtext.splitlines() if l.startswith(("PHASE item=", "[boltz2-opt] PEAK item=", "[boltz2-opt fast] KERNELS route="))], "relay = exactly the worker's own lines, in order"
    assert "[boltz2-opt] PEAK-SUMMARY items=%d rank0_max_alloc_gib=5.25 rankmax_alloc_gib=5.25 rankmax_reserved_gib=6.50 ranks=1" % len(ys) in out
    _stubs.install_fake_stage(monkeypatch, kernels="refuse")
    rc = worker.run("fast", ys, str(tmp_path / "o2"), [0], tag="t"); out = capsys.readouterr().out
    assert rc == rep.EXIT_KERNELS == 5, out
    assert "[boltz2-opt fast] KERNELS-REFUSED route=fast settings=defaults cueq_triatt=absent:" in out and "KERNELS REQUIRE refused in the worker process (exit 5" in out, out
    assert sum(l.startswith("[boltz2-opt fast] KERNELS-REFUSED route=fast") for l in out.splitlines()) == 1, "the refusal line relayed once, verbatim"
    assert mf.LAST["report"]["active"] is False
    _stubs.install_fake_stage(monkeypatch, kernels="none")
    rc = worker.run("fast", ys, str(tmp_path / "o3"), [0], tag="t"); out = capsys.readouterr().out
    assert rc == rep.EXIT_NOT_ACTIVE and "NOT ACTIVE: no KERNELS census (t_kernels_census.json / the KERNELS line in t_worker.log)" in out, "fail-closed: a worker pass without its line / record is NOT ACTIVE (report.kernels_exit)"
    _stubs.install_fake_stage(monkeypatch, kernels="contradiction")
    rc = worker.run("fast", ys, str(tmp_path / "o4"), [0], tag="t"); out = capsys.readouterr().out
    assert rc == 5 and "KERNELS REQUIRE refused: route=fast verdict=REFUSED(" in out, "a verdict other than PASS in the record is the parent's refusal (exit 5)"


def test_stock_route_reads_the_census_record_into_the_manifest_and_exits_by_the_rules(tmp_path, monkeypatch, capsys):
    """cmd_pred_off: the stock subprocess is replaced by a stand-in that writes the record; rc 0 + PASS -> 0; rc 0 + no record -> 3; rc 5 -> 5."""
    import types
    y = tmp_path / "x.yaml"; y.write_text("version: 1\nsequences: []\n")
    monkeypatch.setattr(stack, "cache_check", lambda *a, **k: [])
    monkeypatch.setattr(stack, "weights_status", lambda *a, **k: {"status": "pinned"}); monkeypatch.setattr(stack, "weights_line", lambda *a, **k: "[boltz2-opt] WEIGHTS pinned sha256=stub (the pinned checkpoint)")
    monkeypatch.setattr(stack, "cache_dir", lambda: str(tmp_path))
    runs = []

    def fake_run(argv, *a, env=None, timeout=None, **k):
        if not isinstance(argv, list) or "boltz2_opt.stock_pred" not in argv:   # only the stock caller is stood in for (the manifest's nvidia-smi probe etc. find no binary)
            raise FileNotFoundError(str(argv[:1]))
        runs.append(argv); head = argv[:argv.index("--")]; tail = argv[argv.index("--") + 1:]
        pred = os.path.join(tail[tail.index("--out_dir") + 1], "boltz_results_x", "predictions", "x"); os.makedirs(pred, exist_ok=True)   # what a stock pass that ran leaves behind: its one input's prediction (cmd_pred_off accounts the unit by it)
        open(os.path.join(pred, "x_model_0.cif"), "w").write("cif")
        beh = fake_run.behaviour
        if beh != "none":
            rec = {"route": head[head.index("--kernels-route") + 1], "verdict": "PASS" if beh == "pass" else "REFUSED(cueq_triatt)", "words": {}, "refused": [] if beh == "pass" else [{"accelerator": "cueq_triatt", "word": "absent:x", "expected": "engaged"}]}
            cpath = head[head.index("--kernels-json") + 1]; os.makedirs(os.path.dirname(cpath), exist_ok=True); json.dump(rec, open(cpath, "w"))   # kernels.emit's own mkdir
        return types.SimpleNamespace(returncode=5 if beh == "refuse" else 0)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    for beh, want in (("pass", 0), ("none", rep.EXIT_NOT_ACTIVE), ("refuse", 5), ("contradiction", 5)):
        fake_run.behaviour = beh
        rc = cli.main(["pred", "--mode", "off", "--input", str(y), "--out_dir", str(tmp_path / beh)] + ([] if beh == "pass" else ["--seed", "101"]))
        out = capsys.readouterr().out
        assert rc == want, (beh, rc, out)
        m = mf.LAST
        assert m["rc"] == want and (m["kernels_census"] is None) == (beh == "none"), (beh, m["kernels_census"])
    head = runs[0][:runs[0].index("--")]
    assert head[head.index("--kernels-route") + 1] == "stock" and head[head.index("--kernels-settings") + 1] == "defaults" and runs[0][runs[0].index("--") + 1:] == ["predict", str(y), "--out_dir", str(tmp_path / "pass")], "nothing given: the bare command"
    assert "--seed" not in runs[0][runs[0].index("--"):] and runs[1][runs[1].index("--"):][-2:] == ["--seed", "101"], "--seed only when the caller passes one"


def _fake_boltz_tree(root):
    """A stand-in `boltz.model.models.boltz2` package (the census's trigger module) with a Boltz2 whose predict_step returns a marker."""
    d = root / "fakeboltz" / "boltz" / "model" / "models"; d.mkdir(parents=True)
    for p in (root / "fakeboltz" / "boltz", root / "fakeboltz" / "boltz" / "model", d):
        (p / "__init__.py").write_text("")
    (d / "boltz2.py").write_text("class Boltz2:\n    use_kernels = True\n\n    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):\n        return {'exception': False}\n")
    return str(root / "fakeboltz")


def test_worker_launch_installs_the_census_on_the_model_import_with_routes_named_and_refuses_or_passes(tmp_path):
    """End to end through the real launcher (subprocess): `--route` named (the core's kernel router is bound in the same function — the census
    module keeps its own name), the KERNELS options, a script that imports the trigger module and runs one predict_step. Library stubbed absent ->
    `KERNELS-REFUSED route=exact cueq_triatt=absent:…`, the step never runs, exit 5, the record says REFUSED; library stubbed present + probe served ->
    the step runs, exit 0, one pass line with verdict=PASS and the record beside the batch's out_dir."""
    import subprocess, sys
    fake = _fake_boltz_tree(tmp_path)
    batch = tmp_path / "b.json"; batch.write_text(json.dumps({"out_dir": str(tmp_path), "kit_dir": str(tmp_path), "tag": "t", "items": [], "seeds": [0]}))
    script = tmp_path / "w.py"
    script.write_text(
        "import sys, json\n"
        "import boltz2_opt.kernels as k\n"
        "beh = sys.argv[sys.argv.index('--beh') + 1]\n"
        "if beh == 'absent':\n"
        "    k.install_counters = lambda: {a: 'absent:cuequivariance_ops_torch(ModuleNotFoundError)' for a in k.ACCELERATORS}\n"
        "    k.device_facts = lambda: {'cc': None}\n"
        "else:\n"
        "    k.install_counters = lambda: {a: '' for a in k.ACCELERATORS}\n"
        "    k.static_facts = lambda: {'dists': {'cuequivariance-ops-torch-cu13': '0.10.0'}, 'build': 'cu13', 'version': '0.10.0', 'specs': {k.OPS[a]['ops_module']: '/sp/x.py' for a in k.ACCELERATORS}, 'cubin_lib': None, 'cubin_sm_tags': ['90']}\n"
        "    k.device_facts = lambda: {'index': 0, 'name': 'H100', 'cc': [9, 0], 'torch': 'x', 'cuda': '13.0'}\n"
        "    k.probe = lambda device=None: {a: {'ok': True, 'path': 'served', 'ms': 1.0, 'error': None, 'S': 128} for a in k.ACCELERATORS}\n"
        "import boltz.model.models.boltz2 as m\n"
        "print('STEP-RAN', json.dumps(m.Boltz2().predict_step({}, 0)), flush=True)\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([fake] + sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    base = [sys.executable, "-m", "boltz2_opt.worker_launch", "--route", "fpf_trimul"] + stack.kernels_opts("exact") + ["--", str(script), "--batch", str(batch)]
    assert base[base.index("--kernels-route") + 1] == "exact"
    r = subprocess.run(base + ["--beh", "absent"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == kernels.EXIT_KERNELS == 5, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert "[boltz2-opt exact] KERNELS-REFUSED route=exact settings=defaults cueq_triatt=absent:cuequivariance_ops_torch(ModuleNotFoundError) (expected engaged)" in r.stdout, r.stdout
    assert "STEP-RAN" not in r.stdout, "the guard refuses before the first item's step"
    rec = json.load(open(tmp_path / "t_kernels_census.json"))
    assert rec["verdict"].startswith("REFUSED(") and rec["route"] == "exact" and rec["words"]["cueq_trimul"].startswith("absent:")
    lines = kernels.find_lines(r.stdout)
    assert len(lines) == 1 and lines[0]["verdict"].startswith("REFUSED"), "one pass line per process, at exit, with the verdict"
    r = subprocess.run(base + ["--beh", "present"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0 and "STEP-RAN" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    lines = kernels.find_lines(r.stdout)
    assert len(lines) == 1 and lines[0]["route"] == "exact" and lines[0]["verdict"] == "PASS" and lines[0]["words"]["cueq_triatt"].startswith("engaged:cuequivariance_ops_torch@0.10.0-cu13[served="), lines
    assert json.load(open(tmp_path / "t_kernels_census.json"))["verdict"] == "PASS"
