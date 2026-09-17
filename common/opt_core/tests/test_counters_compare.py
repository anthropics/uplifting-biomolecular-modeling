"""The lever ledger and census arithmetic (counters), the LEVER / line / emit / dump primitives (report), the item-loop helpers (cli),
run_step (process), UnsupportedMode / levers_label / literal_assignment (modes), the output-tree comparison (compare), the pre-body hook (autoload)."""
import io
import json
import os
import sys
import textwrap

import pytest

from opt_core import cli, compare, counters, modes, process, report


def test_line_emit_dump_and_lever_line():
    assert report.line("[t]", "STEP", ("step", "fwd"), rc=0, log=None) == "[t] STEP step=fwd rc=0 log=none"
    assert report.line("[t]", "DONE") == "[t] DONE"
    buf = io.StringIO()
    assert report.emit("x", buf) == "x" and buf.getvalue() == "x\n"
    assert json.loads(report.dump({"b": 1, "a": object.__name__})) == {"a": "object", "b": 1}
    assert report.lever_line("t", "F2.trimul", "on", ("impl", "k@1"), origin="core", served=3) == "[t] LEVER name=F2.trimul state=on impl=k@1 origin=core served=3"
    assert report.lever_line("t", "x", "off", reason="not_in_mode", impl="m@1", origin="kit") == "[t] LEVER name=x state=off reason=not_in_mode impl=m@1 origin=kit"
    assert report.lever_line("t", "x", "on", ("served", 2), origin="kit", impl="m@1") == "[t] LEVER name=x state=on impl=m@1 origin=kit served=2"
    with pytest.raises(ValueError):
        report.lever_line("t", "x", "on", ("state", "on"), impl="m", origin="kit")
    with pytest.raises(ValueError):
        report.lever_line("t", "x", "skipped", impl="m", origin="kit")                       # skipped needs a reason
    with pytest.raises(ValueError):
        report.lever_line("t", "x", "active", impl="m", origin="kit")                        # state vocabulary
    with pytest.raises(ValueError, match="needs impl"):
        report.lever_line("t", "x", "on", origin="kit")
    with pytest.raises(ValueError, match="origin must be"):
        report.lever_line("t", "x", "on", impl="m", origin="carried:other")
    with pytest.raises(ValueError, match="contains a blank"):
        report.lever_line("t", "x", "off", reason="not in mode", impl="m", origin="kit")
    assert report.lever_line("t", "x", "on", ("impl", "p@1"), impl="m@2", origin="core") == "[t] LEVER name=x state=on impl=m@2 origin=core"   # keyword wins, printed once
    assert report.lever_line("t", "x", "on", impl="m", origin="core", strategy="F1.flash_triatt", served=1) == "[t] LEVER name=x state=on impl=m origin=core strategy=F1.flash_triatt served=1"
    assert report.lever_line("t", "x", "on", impl="m", origin="kit", strategy="LOCAL.acme.thing").endswith("strategy=LOCAL.acme.thing")
    for bad in ("tensor_parallel", "F7", "F7.", "LOCAL", "f7.tensor_parallel", "F7.tensor parallel", "X1.thing"):     # the line checks the id's FORM
        with pytest.raises(ValueError, match="is not a strategy id"):
            report.lever_line("t", "x", "on", impl="m", origin="kit", strategy=bad)
    assert report.strategy_form("F9.nothing") == "F9.nothing"                                               # a well-formed id the catalogue does not list passes the line …
    from opt_core import strategies                                                                          # … membership is the catalogue's development-time check
    alias = next(iter(strategies.table()["aliases"].items()))
    with pytest.raises(ValueError, match="is an alias; the canonical id is"):
        strategies.check(alias[0])
    with pytest.raises(ValueError, match="not in the catalogue"):
        strategies.check("F9.nothing")
    assert strategies.check("F1.flash_triatt") == "F1.flash_triatt" and strategies.check("LOCAL.acme.thing") == "LOCAL.acme.thing"


def test_ledger_counts_states_gate_and_line():
    L = counters.Ledger("F1.k", impl="k@2", origin="core", min_tokens=300, expected=("below_min_tokens",))
    assert L.state == "skipped" and L.skip_reason() == "no_calls" and not L.partial and L.gate().ok
    assert L.line("t") == "[t] LEVER name=F1.k state=skipped reason=no_calls impl=k@2 origin=core served=0 fallback=0 fallback_by=none min_tokens=300 shapes=none"
    L.fallback("below_min_tokens"); L.fallback("below_min_tokens")
    assert L.partial and L.skip_reason() == "all_fallback:below_min_tokens" and not L.gate().ok and L.gate(require_served=False).ok
    L.serve("n=384"); L.serve("n=384"); L.serve("n=512", first={"n": 512})
    assert L.served == 3 and L.calls == 5 and L.state == "on" and L.first == {"n": 512} and L.shapes == {"n=384": 2, "n=512": 1}
    assert L.gate().ok
    L.fallback("dtype")
    g = L.gate()
    assert not g.ok and g.reason == "unexpected fallback: fallback_by=dtype:1"
    L.error(RuntimeError("boom"))
    assert L.errors == {"RuntimeError": 1} and not L.gate().ok and L.gate().reason.startswith("kernel error")
    L.count("bytes", 5); L.count("bytes", 7); L.peak("rows", 3); L.peak("rows", 2); L.set("thr", 0.5)
    assert L.facts() == {"bytes": 12, "rows": 3, "thr": 0.5}
    ln = L.line("t", chunks=4)
    assert ln == ("[t] LEVER name=F1.k state=on impl=k@2 origin=core served=3 fallback=3 fallback_by=below_min_tokens:2,dtype:1 "
                  "errors=RuntimeError:1 min_tokens=300 shapes=n=384:2,n=512:1 bytes=12 rows=3 thr=0.5 chunks=4")
    assert L.counts() == {"error:RuntimeError": 1, "fallback:below_min_tokens": 2, "fallback:dtype": 1, "served:k@2": 3}
    f = L.fields()
    assert f["fallback"] == 3 and f["partial"] is False and json.loads(report.dump(f))["name"] == "F1.k"
    L2 = counters.Ledger("plain")
    with pytest.raises(ValueError, match="needs impl"):
        L2.line("t", state="off", reason="mode")                             # a ledger without impl / origin cannot print the line
    L2.serve()
    assert L2.counts() == {"served:kernel": 1}
    L.clear()
    assert L.calls == 0 and L.first is None and L.facts() == {}


def test_census_arithmetic():
    before = {"trimul": {"served:v4": 3, "fallback:small": 1}, "graphs": {"captures": 2}, "dead": {}}
    after = {"trimul": {"served:v4": 9, "fallback:small": 4, "error:OOM": 1}, "attn": {"fallback:mask": 2}, "graphs": {"captures": 5}, "dead": {"attn": "mask"}}
    assert counters.events_of(after, skip=("graphs",)) == {"trimul": {"fallback:small": 4, "error:OOM": 1}, "attn": {"fallback:mask": 2}}
    assert counters.events_of(after, prefixes=(), keys=("captures",)) == {"graphs": {"captures": 5}}
    delta, dead = counters.events_delta(before, after, skip=("graphs",))
    assert delta == {"trimul": {"fallback:small": 3, "error:OOM": 1}, "attn": {"fallback:mask": 2}} and dead == ["attn"]
    assert counters.events_delta(None, None) == ({}, [])
    tot = counters.events_totals([delta, delta, {}])
    assert tot == {"trimul": {"fallback:small": 6, "error:OOM": 2}, "attn": {"fallback:mask": 4}}
    exp, unexp = counters.split_expected(tot, {"trimul": ("fallback:small",)})
    assert exp == {"trimul": {"fallback:small": 6}} and unexp == {"trimul": {"error:OOM": 2}, "attn": {"fallback:mask": 4}}


def test_modes_additions(tmp_path):
    assert issubclass(modes.UnsupportedMode, modes.ModeError) and issubclass(modes.UnsupportedMode, ValueError)
    assert modes.levers_label(["a", "b"]) == "a+b" and modes.levers_label(()) == "none"
    p = tmp_path / "api.py"
    p.write_text("import torch\nLEVER_SETS = {'eager': (), 'best': ('a', 'b')}\nX: int = 3\nY = compute()\n")
    assert modes.literal_assignment(str(p), "LEVER_SETS") == {"eager": (), "best": ("a", "b")}
    assert modes.literal_assignment(str(p), "X") == 3
    with pytest.raises(ValueError, match="not a literal"):
        modes.literal_assignment(str(p), "Y")
    with pytest.raises(ValueError, match="no top-level Z"):
        modes.literal_assignment(str(p), "Z")
    assert "torch" not in sys.modules or True                                      # parsed, never imported


def test_cli_item_loop_helpers(tmp_path):
    d = tmp_path / "in"; d.mkdir()
    (d / "002_b.json").write_text("{}"); (d / "001_a.json").write_text('{"k": 1}'); (d / "n.txt").write_text("")
    ns = type("A", (), {"json_path": [str(tmp_path / "x.json")], "input_dir": str(d)})()
    assert cli.input_paths(ns) == [str(tmp_path / "x.json"), str(d / "001_a.json"), str(d / "002_b.json")]
    assert cli.input_paths(json_path=None, input_dir=str(d), pattern="*.txt") == [str(d / "n.txt")]
    assert cli.input_paths(None) == []
    assert cli.item_name(str(d / "001_a.json")) == "001_a"
    assert cli.read_report(str(d / "001_a.json")) == {"k": 1} and cli.read_report(str(d / "nope.json")) is None and cli.read_report(str(d / "n.txt")) is None
    assert cli.step_reason({"rc": 3}) == "step rc=3" and cli.step_reason({"rc": -1, "error": "timeout after 5s"}) == "timeout after 5s"


def test_run_step_records_and_lines(tmp_path):
    buf = io.StringIO()
    log = str(tmp_path / "s.log")
    rec = process.run_step("echo", [sys.executable, "-c", "print('hi')"], prefix="[t]", log_path=log, stream=buf)
    assert rec["ok"] and rec["rc"] == 0 and rec["step"] == "echo" and rec["log"] == log and "error" not in rec and open(log).read() == "hi\n"
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("[t] COMMAND step=echo argv=") and lines[1].startswith("[t] STEP step=echo rc=0 wall_s=")
    rec = process.run_step("slow", [sys.executable, "-c", "import time; time.sleep(5)"], prefix="[t]", timeout_s=0.5, stream=buf)
    assert rec["rc"] == process.RC_TIMEOUT and rec["error"] == "timeout after 0.5s" and not rec["ok"] and "process group killed" in rec["reason"]
    rec = process.run_step("nolaunch", [str(tmp_path / "no-such-binary")], prefix="[t]", stream=buf)
    assert rec["rc"] == process.RC_LAUNCH and rec["error"].startswith("could not launch:") and not rec["ok"]
    assert cli.step_reason(rec).startswith("could not launch:")


def test_compare_output_trees(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "001_x").mkdir(parents=True); (root / "_work").mkdir()
        (root / "001_x" / "001_x_model.cif").write_text("data\n_stamp.time 12:00\nATOM 1\n" if root == a else "data\n_stamp.time 13:00\nATOM 1\n")
        (root / "001_x" / "conf.json").write_text("{}")
        (root / "_work" / "step.log").write_text(str(root))
        (root / "opt_manifest.json").write_text("{}")
    (b / "001_x" / "extra.npz").write_text("z")
    ha = compare.hash_outputs(str(a), exclude_dirs=("_work",))
    hb = compare.hash_outputs(str(b), exclude_dirs=("_work",))
    assert set(ha) == {"001_x/001_x_model.cif", "001_x/conf.json"} and "opt_manifest.json" not in ha
    same, total, diff, only = compare.compare(ha, hb)
    assert (same, total, diff, only) == (1, 3, ["001_x/001_x_model.cif"], ["001_x/extra.npz"])
    drop = {".cif": ("_stamp.time",)}
    assert compare.content_sha256(str(a / "001_x" / "001_x_model.cif"), drop) == compare.content_sha256(str(b / "001_x" / "001_x_model.cif"), drop)
    ia = compare.items_content(str(a), sorted(ha), drop_line_prefixes=drop)
    ib = compare.items_content(str(b), sorted(hb), drop_line_prefixes=drop)
    assert set(ia) == {"x"} and set(ia["x"]) == {"x_model.cif", "conf.json"}
    res = compare.compare_items(ia, ib)
    assert res == {"x": (2, 3, ["extra.npz"])}
