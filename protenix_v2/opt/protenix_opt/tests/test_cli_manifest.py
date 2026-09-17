"""CPU tests for the checkpoint identity helper (manifest.py) and the activation / exit lines (report.py)."""
from __future__ import annotations

import hashlib
import json
import os

from protenix_opt import manifest, report

REPORT = {"active": True, "mode": "exact", "levers_applied": ["T1:fused", "BLK:2", "DEADSKIP"], "levers_fallback": [],
          "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": 90}, "protenix_version": "2.0.0", "package_version": "0.1.0"}


# ----------------------------------------------------------------------------------------------------------------- checkpoint identity
def test_checkpoint_sha(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    (root / "checkpoint").mkdir(parents=True)
    blob = os.urandom(100_000)
    (root / "checkpoint" / "protenix-v2.pt").write_bytes(blob)
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(root))
    ck = manifest.checkpoint_info(memo_dir=str(tmp_path / "memo"))
    assert ck["present"] is True and ck["bytes"] == len(blob) and ck["sha256"] == hashlib.sha256(blob).hexdigest()
    assert manifest.checkpoint_info(hash_it=False).get("sha256") is None
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(tmp_path / "nowhere"))
    assert manifest.checkpoint_info()["present"] is False and manifest.checkpoint_info()["reason"] == "file not found"


# ----------------------------------------------------------------------------------------------------------------- report: activation line


def test_log_activation_respects_logged_flag(capsys):
    report.log_activation(REPORT)
    assert capsys.readouterr().err.strip() == report.activation_line(REPORT)
    report.log_activation(dict(REPORT, logged=True))
    assert capsys.readouterr().err == ""


# ----------------------------------------------------------------------------------------------------------------- report: exit tally
def _write_report(path, recs):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_read_lever_report_pid_filter(tmp_path):
    p = tmp_path / "lr.jsonl"
    _write_report(p, [{"pid": 1, "applied": ["a"]}, {"pid": 2, "applied": ["b"]}, {"pid": 1, "applied": ["c"]}])
    assert report.read_lever_report(str(p))["applied"] == ["c"]
    assert report.read_lever_report(str(p), pid=1)["applied"] == ["c"]
    assert report.read_lever_report(str(p), pid=2)["applied"] == ["b"]
    assert report.read_lever_report(str(p), pid=3) is None
    assert report.read_lever_report(str(tmp_path / "missing.jsonl")) is None and report.read_lever_report(None) is None
    p.write_text("not json\n\n")
    assert report.read_lever_report(str(p)) is None and report.count_lines(str(p)) == 1


def test_read_lever_report_merges_the_pids_records(tmp_path):
    """The kit writes several lines per pid and the package appends its own record last: the tally must carry the trunk record's
    applied/counters, not the last line's absence of them (the failure mode: 'applied=none' after a full prediction)."""
    p = tmp_path / "lr.jsonl"
    trunk = {"pid": 169, "applied": ["T1:fused", "BLK:2(pro+cueq+epi+fusion_transition)", "DEADSKIP:hooked(load_checkpoint)"], "blk_calls": 480,
             "blk_fallback": 0, "t1_fused_calls": 960, "deadskip": {"n": 41, "n_expected": 41}}
    package = {"pid": 169, "protenix_opt": {"mode": "exact", "active": True}}
    _write_report(p, [{"pid": 169, "clisampler": {"graphs": 1}, "applied": []}, trunk, {"pid": 169, "stackgraph": {"captures": 3}}, package, {"pid": 170, "applied": ["other"]}])
    rec = report.read_lever_report(str(p), pid=169)
    assert rec["applied"] == trunk["applied"] and rec["blk_calls"] == 480 and rec["deadskip"] == {"n": 41, "n_expected": 41}
    assert rec["protenix_opt"]["mode"] == "exact" and rec["clisampler"] == {"graphs": 1} and rec["stackgraph"] == {"captures": 3} and rec["records"] == 4
    line = report.exit_tally_line(str(p), pid=169)
    assert "applied=T1:fused,BLK:2(pro+cueq+epi+fusion_transition),DEADSKIP:hooked(load_checkpoint)" in line and "blk_calls=480" in line and "deadskip=41/41" in line
    assert "applied=none" not in line
    # two-record shape exactly as a prediction leaves it: trunk line then the package line
    q = tmp_path / "two.jsonl"
    _write_report(q, [trunk, package])
    line2 = report.exit_tally_line(str(q), pid=169)
    assert "blk_calls=480" in line2 and "applied=none" not in line2 and report.read_lever_report(str(q), pid=169)["records"] == 2
    # a later non-empty value still wins (a second trunk write of the same pid updates the counters)
    _write_report(q, [trunk, dict(trunk, blk_calls=960), package])
    assert report.read_lever_report(str(q), pid=169)["blk_calls"] == 960


def test_exit_tally_sources(tmp_path):
    p = tmp_path / "lr.jsonl"
    rec = {"pid": 42, "applied": ["T1:fused", "BLK:2(pro+cueq+epi+fusion_transition)"], "blk_calls": 480, "blk_fallback": 0, "t1_fused_calls": 960,
           "deadskip": {"n": 41, "n_expected": 41}, "v02": {"applied": ["msablk"]}, "fpf": {"trimul_out": "fpf_trimul_exact"}}
    _write_report(p, [{"pid": 7, "applied": []}, rec])
    line = report.exit_tally_line(str(p), pid=42)
    assert line.startswith("[protenix-opt] EXIT pid=42 source=file report=")
    for frag in ("applied=T1:fused,BLK:2(pro+cueq+epi+fusion_transition)", "deadskip=41/41", "blk_calls=480", "blk_fallback=0",
                 "v02=msablk", "fpf_ops={'trimul_out': 'fpf_trimul_exact'}"):
        assert frag in line, frag
    mem = report.exit_tally_line(str(p), pid=99, memory_stats={"applied": ["X"], "blk_calls": 1})
    assert "source=memory" in mem and "lines_for_this_pid=0 lines_total=2" in mem and "applied=X blk_calls=1" in mem
    none = report.exit_tally_line(str(tmp_path / "absent.jsonl"), pid=99)
    assert "no lever counters" in none and "lines_total=file-missing" in none
    assert "report=<unset>" in report.exit_tally_line(None, pid=1)


def test_register_exit_tally_idempotent(monkeypatch):
    core = report._core_report
    monkeypatch.setattr(core, "_TALLIES", {})
    registered = []
    monkeypatch.setattr(core.atexit, "register", lambda fn, *args: registered.append((fn, *args)))
    assert report.register_exit_tally() is True
    assert report.register_exit_tally() is False
    assert registered == [(core._print_exit_tally, report.TAG, report.exit_line)], "the core prints the kit's whole line, once per tag"


def test_print_exit_tally_no_levers(monkeypatch, capsys):
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "ptx_trunk2_levers", raising=False)
    line = report.exit_line()
    report._core_report._print_exit_tally(report.TAG, report.exit_line)                       # the atexit printer: the line, verbatim, to stderr
    assert "no lever counters" in line and capsys.readouterr().err.strip() == line
