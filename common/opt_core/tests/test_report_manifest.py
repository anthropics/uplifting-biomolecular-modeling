"""The exit table, the verdict rule, the line primitives, the exit tally (in a subprocess), opt_manifest.json round trips, recorded fixture helpers."""
import json
import os
import subprocess
import sys

from opt_core import manifest, report, testing


def test_exit_table():
    assert (report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE) == (0, 1, 2, 3)


def test_line_primitives():
    assert report.prefix("acme-opt") == "[acme-opt]"
    assert report.join([]) == "none" and report.join(["a", "b"]) == "a,b" and report.join("x") == "x"
    assert report.kv(("mode", "fast"), levers=["a", "b"], fallbacks=None, n=3, d={"k": 1}) == "mode=fast levers=a,b fallbacks=none n=3 d=k:1"
    assert report.gpu_label({"name": "NVIDIA H100 80GB HBM3", "sm": "sm90"}) == "NVIDIA H100 80GB HBM3(sm90)"
    assert report.gpu_label({"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"}) == "NVIDIA H100 80GB HBM3(sm90)"
    assert report.gpu_label({"name": "X"}) == "X" and report.gpu_label({}) == "none" and report.gpu_label(None) == "none"
    assert report.not_active_line("acme-opt", "mode off: stock", "mode=off") == "[acme-opt] NOT ACTIVE: mode off: stock (mode=off)"


def test_word_grammar_and_collection():
    assert report.word("card", "uncertified", "NVIDIA H100 NVL", "95830MiB") == "card=uncertified(NVIDIA_H100_NVL,95830MiB)"
    assert report.word("core_pin", "unreadable") == "core_pin=unreadable" and report.word("cache", "miss", 3) == "cache=miss(3)"
    assert report.word("stack", "drift", "torch:2.13.0!=2.12.1", "numpy:None!=1.26.4") == "stack=drift(torch:2.13.0!=2.12.1,numpy:None!=1.26.4)"
    assert report.word("k", None) == "k=none" and report.word("a b", " x\ty ") == "a_b=x_y"

    class WithAttr:
        words = ("a=1", "b=2")

    class WithMethod:
        def words(self):
            return ["b=2", "c=3"]

    assert report.words_of("a=1", None, WithAttr(), [WithMethod(), {"words": ["d=4"]}, "a=1"], {"nowords": 1}) == ["a=1", "b=2", "c=3", "d=4"]
    assert report.words_of() == [] and report.words_of(None, [], {"words": None}) == []
    assert report.with_words("[acme-opt] ACTIVE mode=fast", ["card=uncertified(NVIDIA_L4,22731MiB)", WithAttr()]) == \
        "[acme-opt] ACTIVE mode=fast card=uncertified(NVIDIA_L4,22731MiB) a=1 b=2"
    assert report.with_words("[acme-opt] ACTIVE mode=fast", None) == "[acme-opt] ACTIVE mode=fast" == report.with_words("[acme-opt] ACTIVE mode=fast", [])


def test_verdict_truth_table():
    ok = report.verdict(0, {"partial": []}, False)
    assert ok == {"exit_code": 0, "partial": [], "gated": [], "allow_partial": False, "incomplete": None, "words": []}
    named = report.verdict(0, {"partial": [], "words": ["stack=drift(torch:2.13.0!=2.12.1)", "card=uncertified(NVIDIA_L4,22731MiB)"]}, False)
    assert named["exit_code"] == 0 and named["words"] == ["stack=drift(torch:2.13.0!=2.12.1)", "card=uncertified(NVIDIA_L4,22731MiB)"]   # words are recorded, never priced
    assert report.verdict(0, {"partial": ["lever_a"], "words": ["cache=miss(2)"]}, False) == \
        {"exit_code": 3, "partial": ["lever_a"], "gated": [], "allow_partial": False, "incomplete": None, "words": ["cache=miss(2)"]}
    assert report.verdict(0, {"partial": ["lever_a"]}, False)["exit_code"] == 3
    assert report.verdict(0, {"partial": ["lever_a"]}, True)["exit_code"] == 0
    assert report.verdict(0, {"partial": []}, False, incomplete="2 of 3 items")["exit_code"] == 1
    assert report.verdict(0, {"partial": ["lever_a"]}, True, incomplete="2 of 3 items")["exit_code"] == 1
    assert report.verdict(5, {"partial": ["lever_a"]}, False)["exit_code"] == 5            # a run that failed on its own keeps its code


def test_partial_lines():
    v = report.verdict(0, {"partial": ["lever_a", "lever_b"]}, False)
    assert report.partial_line("acme-opt", v, {"lever_a": "kernel missing"}) == (
        "[acme-opt] NOT ACTIVE: partial activation — levers=lever_a,lever_b (lever_a: kernel missing; lever_b: no reason recorded); "
        "exit 3 (--allow-partial records and proceeds)")
    v = report.verdict(0, {"partial": ["lever_a"]}, True)
    assert report.partial_line("acme-opt", v, {"lever_a": "x"}) == "[acme-opt] PARTIAL allowed: levers=lever_a (lever_a: x) (--allow-partial, recorded)"
    assert report.partial_line("acme-opt", report.verdict(0, {}, False)) is None
    assert report.partial_line("acme-opt", report.verdict(4, {"partial": ["a"]}, False)) is None


def test_allow_partial_reader(monkeypatch):
    monkeypatch.delenv("ACME_ALLOW_PARTIAL", raising=False)
    assert report.allow_partial(False, "ACME_ALLOW_PARTIAL") is False and report.allow_partial(True, "ACME_ALLOW_PARTIAL") is True
    assert report.allow_partial(False, "ACME_ALLOW_PARTIAL", {"ACME_ALLOW_PARTIAL": " 1 "}) is True
    assert report.allow_partial(False, "ACME_ALLOW_PARTIAL", {"ACME_ALLOW_PARTIAL": "yes"}) is False


def test_log_once(capsys):
    rep = {}
    report.log_once(rep, "[acme-opt] ACTIVE mode=fast")
    report.log_once(rep, "[acme-opt] ACTIVE mode=fast")
    assert capsys.readouterr().err == "[acme-opt] ACTIVE mode=fast\n" and rep["logged"] is True


def test_exit_tally_prints_once_at_exit_and_never_silently(tmp_path):
    code = """
import sys
sys.path.insert(0, %r)
from opt_core import report
n = [0]
def line():
    n[0] += 1
    return "[acme-opt] EXIT graph_stats=none calls=%%d" %% n[0]          # the kit's WHOLE line, its own grammar
assert report.register_exit_tally("acme-opt", line) is True
assert report.register_exit_tally("acme-opt", line) is False
def boom():
    raise RuntimeError("counters gone")
assert report.register_exit_tally("other-opt", boom) is True
assert report.register_exit_tally("none-opt", lambda: report.exit_tally_line("none-opt", None)) is True
""" % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("[none-opt] EXIT pid=") and lines[0].endswith("no lever counters: the kit's lever modules were never loaded in this process")
    assert lines[1] == "[other-opt] EXIT tally failed: RuntimeError('counters gone')"
    assert lines[2] == "[acme-opt] EXIT graph_stats=none calls=1"                       # the kit's line, byte for byte


# ------------------------------------------------------------------------------------------------------------ manifest


def test_manifest_build_write_read_round_trip(tmp_path):
    rep = {"active": True, "levers": ["a"], "logged": True}
    doc = manifest.build(package="acme_opt", package_version="1.2.3", schema="acme_opt/3", mode="fast", report=rep, route="in-process",
                         command=["python", "-m", "acme_opt", "pred"], stack={"python": "3.12"}, pass_={"pass_status": "ok", "n_items": 2, "n_items_complete": 2},
                         exit_={"exit_code": 0}, kit={"server_mode": "persistent", "gpu_class": "H100"}, top_level=("gpu_class",))
    assert doc["activation"] == {"active": True, "levers": ["a"]} and doc["gpu_class"] == "H100" and "server_mode" not in doc
    assert doc["items_complete"] is None and doc["n_items"] == 2 and doc["exit"] == {"exit_code": 0}
    path = manifest.write(str(tmp_path / "out" / manifest.FILENAME), doc)
    text = open(path).read()
    assert text.endswith("\n") and json.loads(text) == manifest.read(path) == doc
    assert sorted(os.listdir(tmp_path / "out")) == [manifest.FILENAME]                  # no temp file left behind
    assert manifest.line("acme-opt", path) == f"[acme-opt] manifest: {path}"


def test_stack_block_carries_the_core_block_without_imports(monkeypatch):
    monkeypatch.setenv("PATH", "")
    b = manifest.stack_block(gpu={"name": "X"}, cuda="12.4", stack_key="k")
    assert set(b) == {"python", "torch", "cuda", "triton", "gpu", "stack_key", "core"}
    assert set(b["core"]) == {"version", "package_dir"} and b["gpu"] == {"name": "X"} and b["cuda"] == "12.4"


def test_manifest_cli_prints_the_shared_block(tmp_path, capsys):
    doc = manifest.build(package="acme_opt", package_version="1", schema="s", mode="off", report={}, route="stock", exit_={"exit_code": 3})
    p = manifest.write(str(tmp_path / manifest.FILENAME), doc)
    assert manifest.main([p]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "schema=s" and out[3] == "mode=off" and out[-1] == "exit_code=3"
    assert manifest.main([]) == 2


# ------------------------------------------------------------------------------------------------------------ recorded fixture helpers


def test_golden_lines():
    produced = ["[acme-opt] ACTIVE mode=fast levers=a,b", "[acme-opt] APPLIED a=1", "noise"]
    r = testing.golden_lines(["[acme-opt] ACTIVE mode=fast levers=a,b", r"re:^\[acme-opt\] APPLIED a=\d+$", "[acme-opt] READY"], produced, forbid=["re:NOT ACTIVE"])
    assert r == {"unmatched": ["[acme-opt] READY"], "forbidden": [], "ok": False}
    r = testing.golden_lines([], ["[acme-opt] NOT ACTIVE: x"], forbid=["re:NOT ACTIVE"])
    assert r["forbidden"] == ["[acme-opt] NOT ACTIVE: x"] and not r["ok"]


def test_manifest_diff():
    a = {"schema": "s", "stack": {"python": "3.12", "wall_s": 1.5}, "items": [1, 2], "empty": {}}
    b = {"schema": "s", "stack": {"python": "3.12", "wall_s": 2.5, "core": {"version": "0.1.0"}}, "items": [1, 2], "empty": {}}
    d = testing.manifest_diff(a, b, volatile=["stack.wall_s"])
    assert d == {"added": ["stack.core.version"], "removed": [], "changed": {}, "ok": True}
    d = testing.manifest_diff(a, {"schema": "t", "items": [1]})
    assert d["removed"] == ["empty", "items.1", "stack.python", "stack.wall_s"] and d["changed"] == {"schema": ("s", "t")} and not d["ok"]
