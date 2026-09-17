"""opt_core.kernels.safe_settings — the ONE cc|triton resolution + safe-settings mechanism: resolution order, the safety net's three cases,
the ONE line, the census word (pure Python; no GPU, no triton needed)."""
import io
import contextlib
import pytest

from opt_core.kernels import safe_settings as S

TABLE = {"8.0|*": {"c32": "tuned32", "c16": "tuned16"}, "8.0|2.3": {"c32": "safe32"}}


class Refused(RuntimeError):
    pass


class FakeCompile(RuntimeError):
    def __init__(self, msg="PassManager::run failed"):
        super().__init__(msg)


def _raise(e):
    raise e


def test_resolution_exact_then_default_then_none():
    assert S.cc_word((8, 0)) == "8.0" and S.cc_word("9.0") == "9.0" and S.cc_word(None) is None
    assert S.resolve_key(TABLE, (8, 0), "2.3") == "8.0|2.3" and S.resolve_key(TABLE, (8, 0), "3.6") == "8.0|*" and S.resolve_key(TABLE, "8.0", "") == "8.0|*"
    assert S.resolve_key(TABLE, (9, 0), "3.6") is None and S.resolve_row(TABLE, (9, 0), "3.6", default={}) == {}
    assert S.resolve_cell(TABLE, (8, 0), "2.3", "c32") == ("8.0|2.3", "safe32")
    assert S.resolve_cell(TABLE, (8, 0), "2.3", "c16") == ("8.0|*", "tuned16")          # the exception row need not repeat every cell: the default row's cell
    assert S.resolve_cell(TABLE, (8, 0), "3.7", "c64") == (None, None) and S.resolve_cell(TABLE, None, "3.7", "c32") == (None, None)
    assert S.where_word((8, 0), "3.6") == "cc 8.0, triton 3.6" and S.where_word(None, "") == "cc ?, triton ?"


def test_run_build_failure_serves_safe_with_one_line_then_stays_safe():
    net = S.SafeNet("lever_x", refused=Refused); calls = []
    def build(settings):
        calls.append(settings)
        if settings != "SAFE":
            raise FakeCompile()
        return "ran:" + settings
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert net.run(build, "TUNED", "SAFE", where="cc 8.0, triton 9.9") == "ran:SAFE"
        assert net.run(build, "TUNED", lambda: "SAFE", where="cc 8.0, triton 9.9") == "ran:SAFE"      # already safe: straight to the (provider's) safe settings
    assert calls == ["TUNED", "SAFE", "SAFE"]
    assert [l for l in err.getvalue().splitlines() if l] == ["[opt_core/lever_x] safe settings served (build_failed:FakeCompile, cc 8.0, triton 9.9)"]
    assert net.on and net.wide and net.word() == "safe:build_failed:FakeCompile" and net.snapshot() == {"on": True, "reason": "build_failed:FakeCompile", "where": "cc 8.0, triton 9.9", "note": None,
                                                                                                   "wide": True, "cells": {}}      # a BUILD failure: process-wide


def test_safe_failing_too_is_the_levers_refusal():
    net = S.SafeNet("lever_y", refused=Refused)
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(Refused) as e:
            net.run(lambda s: _raise(FakeCompile("failed to legalize operation")), "TUNED", "SAFE", where="cc 8.0, triton 9.9")
    assert "lever_y: the safe settings cannot build/run either (build_failed:FakeCompile, cc 8.0, triton 9.9)" in str(e.value) and "--mode off" in str(e.value)


def test_no_cell_serves_safe_with_one_line_and_refuses_when_safe_fails():
    net = S.SafeNet("pair_fused:transition", refused=Refused); err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert net.no_cell("N705xC128", lambda s: "ran:" + s, "SAFE", where=S.where_word((8, 0), "3.6")) == "ran:SAFE"
    assert err.getvalue().strip() == "[opt_core/pair_fused:transition] safe settings served (no_cell:N705xC128, cc 8.0, triton 3.6)" and net.word() == "safe:no_cell:N705xC128"
    net2 = S.SafeNet("pair_fused:trimul", refused=Refused)
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(Refused):
            net2.no_cell("N705xC128", lambda s: _raise(FakeCompile()), "SAFE", where="w")
        with pytest.raises(KeyError):                                                          # a non-build exception under the safe settings propagates as raised
            net2.no_cell("N705xC128", lambda s: _raise(KeyError("x")), "SAFE", where="w")


def test_other_exceptions_explicit_settings_and_oom_propagate_untouched():
    net = S.SafeNet("lever_z", refused=Refused)
    with pytest.raises(ValueError):
        net.run(lambda s: _raise(ValueError("x")), "TUNED", "SAFE", where="w")
    with pytest.raises(RuntimeError):                                                        # a RuntimeError without the compiler's words (a CUDA runtime error): not a build failure
        net.run(lambda s: _raise(RuntimeError("CUDA error: an illegal memory access was encountered")), "TUNED", "SAFE", where="w")
    with pytest.raises(FakeCompile):                                                         # caller-pinned settings are never retried
        net.run(lambda s: _raise(FakeCompile()), "PINNED", "SAFE", where="w", explicit=True)
    with pytest.raises(RuntimeError):
        net.run(lambda s: _raise(RuntimeError("CUDA out of memory. Tried to allocate")), "TUNED", "SAFE", where="w")
    assert not net.on and net.word() is None


def test_serve_one_call_form():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        net = S.SafeNet("lever_s", refused=Refused)
        assert S.serve(net, TABLE, (8, 0), "3.6", "c32", lambda s: s, "SAFE") == ("8.0|*", "tuned32")
        assert S.serve(net, TABLE, (8, 0), "2.3", "c32", lambda s: s, "SAFE") == ("8.0|2.3", "safe32")
        assert S.serve(net, {}, (9, 0), "3.6", "c32", lambda s: s, "SAFE", default_settings="H100") == ("default", "H100")      # a lever without a cc table: its own tables, no line
        assert err.getvalue() == "" and not net.on and net.note() is None
        assert S.serve(net, TABLE, (12, 0), "3.6", "c32", lambda s: s, "SAFE", default_settings="DEF") == ("default", "DEF")     # a capability no row names: the default settings + ONE info line
        assert err.getvalue().strip() == "[opt_core/lever_s] default settings (no row for cc 12.0, cc 12.0, triton 3.6); tuned rows exist for cc 8.0" and net.note() == "default:no_row" and not net.on
        assert S.serve(net, TABLE, (8, 0), "3.6", "c64", lambda s: s, "SAFE", shape_word="D64") == ("safe", "SAFE")           # no cell anywhere: safe + ONE line
        assert S.serve(net, TABLE, (8, 0), "3.6", "c32", lambda s: s, "SAFE") == ("8.0|*", "tuned32")                         # SCOPED: the pinned cell keeps its settings in the same process
        assert S.serve(net, TABLE, (8, 0), "3.6", "c64", lambda s: s, "SAFE", shape_word="D64") == ("safe", "SAFE")           # the unknown cell stays on the safe settings, no second line
        assert net.on and not net.wide and net.cells == {"D64": "no_cell:D64"} and net.serves_safe("D64") and not net.serves_safe("c32")
        assert S.serve(net, TABLE, (8, 0), "3.6", "c32", lambda s: s, "SAFE", explicit_settings="PIN") == ("explicit", "PIN")
        net2 = S.SafeNet("lever_t", refused=Refused)
        assert S.serve(net2, TABLE, (8, 0), "3.6", "c32", lambda s: s if s == "SAFE" else _raise(FakeCompile()), "SAFE") == ("safe", "SAFE")  # build failure of the tuned cell
        assert net2.wide and S.serve(net2, TABLE, (8, 0), "3.6", "c32", lambda s: s, "SAFE") == ("safe", "SAFE")               # a BUILD failure holds for the rest of the process
    assert err.getvalue().count("safe settings served") == 2 and err.getvalue().count("default settings (") == 1


def test_classifier():
    assert S.is_build_failure(FakeCompile()) and S.is_build_failure(RuntimeError("ptxas fatal: ..."))
    assert not S.is_build_failure(ValueError("x")) and not S.is_build_failure(RuntimeError("CUDA out of memory"))
    for t in S.build_error_types():
        assert issubclass(t, BaseException)


def test_safe_rows_pair_fused_content_and_provider():
    tr8 = S.safe_row("pair_fused:transition", (8, 0)); tr9 = S.safe_row("pair_fused:transition", "9.0")
    assert tr8["settings"] == {"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0} and tr8["status"] == "SAFE"
    assert tr9["settings"] == tr8["settings"] and tr9["status"] == "SAFE:inferred"                       # one cross-card cell; the H100 validity is inferred, said so
    tm8 = S.safe_row("pair_fused:trimul", "8.0")["settings"]; tm9 = S.safe_row("pair_fused:trimul", (9, 0))["settings"]
    assert tm8 == {"k1": {"BM": 128, "BN": 64, "num_warps": 8, "num_stages": 2}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    assert tm9["k1"] == {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1} and tm9["k3"]["num_stages"] == 1
    assert S.safe_row("pair_fused:transition", (12, 0)) is S.SAFE_ROWS["pair_fused:transition"]["*"] and S.safe_row("no_such_lever", "8.0") is None   # any capability: the '*' row
    assert S.safe_settings_of("pair_fused:trimul", (8, 0))() == tm8
    assert S.safe_settings_of("pair_fused:transition", (12, 0))() == S.SAFE_ROWS["pair_fused:transition"]["*"]["settings"]
    with pytest.raises(LookupError):
        S.safe_settings_of("no_such_lever", (12, 0))()
    for lever, rows in S.SAFE_ROWS.items():
        for cc, row in rows.items():
            assert {"settings", "status", "evidence"} <= set(row) <= {"settings", "status", "evidence", "when", "alternatives", "refusal"}, (lever, cc)   # when/alternatives/refusal: a row's shape bounds (safe_settings_for)
            assert row["status"].startswith("SAFE") and row["evidence"], (lever, cc)


def test_inform_is_one_info_line_and_a_note_not_the_safe_settings():
    net = S.SafeNet("lever_i", refused=Refused); err = io.StringIO()
    with contextlib.redirect_stderr(err):
        net.inform("no row for cc 8.6", S.where_word((8, 6), "3.7"), tuned=["8.0", "9.0"]); net.inform("again", "w")
    assert err.getvalue().strip() == "[opt_core/lever_i] default settings (no row for cc 8.6, cc 8.6, triton 3.7); tuned rows exist for cc 8.0, 9.0"
    assert net.note() == "default:no_row" and net.word() is None and not net.on and net.snapshot()["note"] == "default:no_row"
    net.reset(); assert net.note() is None
    assert S.table_ccs({"8.0|*": 1, "8.0|2.3": 2, "10.0|*": 3, "9.0": 4, "_doc": 5}) == ["8.0", "9.0", "10.0"]


def test_device_rows_and_block_ladders():
    """row_for_device / key_for_device: the row of the call's device (device_cc under this triton), memoised per device slot in the caller's dict; no CUDA
    (or no torch) = capability None = the default; pick_by_block: the first (max block, value) pair covering the block, the last pair beyond it."""
    table = {"8.0|*": {"w": 1}, "8.0|3.0": {"w": 2}}
    assert S.pick_by_block(((64, 1), (128, 2), (512, 4), (1024, 8)), 32) == 1 and S.pick_by_block(((64, 1), (128, 2)), 65) == 2
    assert S.pick_by_block(((64, 1), (128, 2)), 4096) == 2 and S.pick_by_block(((1 << 30, 4),), 1) == 4
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if not cuda:                                                                  # the CPU interpreters this suite runs on
        cache = {}
        assert S.device_cc() is None and S.device_cc("cpu") is None and S.key_for_device(table) is None
        assert S.row_for_device(table, None, "dflt", cache=cache) == "dflt" and cache == {("none",): "dflt"}
        assert S.row_for_device(table, "cpu", "dflt") == "dflt"
    else:                                                                         # a CUDA box: the row of the device's capability, memoised
        cache = {}
        cc = S.device_cc()
        row = S.row_for_device(table, None, "dflt", cache=cache)
        assert row == (S.resolve_row(table, cc, S.triton_mm(), default="dflt")) and list(cache) == [("cuda", torch.cuda.current_device())]
        assert S.row_for_device(table, torch.device("cuda"), "dflt", cache=cache) is row and S.row_for_device(table, 0, "dflt") == row
        assert S.device_cc("cpu") is None and S.row_for_device(table, "cpu", "dflt") == "dflt"
