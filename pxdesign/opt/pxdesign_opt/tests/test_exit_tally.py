"""The exit tally reads the kit module's own counters; when the module never loaded it says so."""
import sys
import types

from pxdesign_opt import report


def test_tally_without_lever_module():
    sys.modules.pop("pxd_xattempt.hoist", None)
    line = report.exit_tally()
    assert line.startswith("[pxdesign-opt] EXIT tally:") and "never loaded" in line


def test_activate_registers_the_tally(fresh_stack, monkeypatch):
    """activate() wires the tally into the process (registered once, at activation, before the kit module loads) — the line real runs print."""
    from . import _stubs
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    calls = []
    monkeypatch.setattr(report, "register_exit_tally", lambda: calls.append(True))
    assert stack.activate("exact")["active"]
    assert calls == [True]
    stack.activate("exact")                                                          # idempotent activation registers nothing more
    assert calls == [True]


def test_tally_reads_kit_stats(monkeypatch):
    m = types.ModuleType("pxd_xattempt.hoist")
    m._STATE = {"installed": True, "stats": {"prepares": 3, "hits": {"f_forward": 1200, "apb_tok": 19200}, "prepare_s": [0.5, 0.25, 0.25]}}
    m.stats = lambda: m._STATE["stats"]
    monkeypatch.setitem(sys.modules, "pxd_xattempt.hoist", m)
    line = report.exit_tally()
    assert line.endswith("installed=True prepares=3 prepare_s_total=1.0 hits=apb_tok=19200,f_forward=1200") or "hits=apb_tok=19200,f_forward=1200 partial=" in line, line


def test_register_once():
    """The core registers one EXIT line per tag per process (opt_core.report.register_exit_tally): a second registration is a no-op."""
    from opt_core import report as core
    core._TALLIES.pop(report.TAG, None) if hasattr(core, "_TALLIES") else None
    first = report.register_exit_tally()
    assert report.register_exit_tally() is False
    assert first in (True, False)


def test_activation_line_forms():
    rep = {"active": True, "mode": "exact", "upstream": {"pxdesign": "0.1.0", "protenix": "0.5.0+pxd", "pxdbench": "0.1.2"}, "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90"},
           "levers_planned": ["h1", "h2"], "env": {"PXD_HOIST": "1"}, "applied": "deferred"}
    assert report.activation_line(rep) == "[pxdesign-opt] ACTIVE mode=exact pxdesign=0.1.0 protenix=0.5.0+pxd pxdbench=0.1.2 gpu=NVIDIA H100 80GB HBM3(sm90) levers=h1,h2 env=PXD_HOIST=1 applied=deferred package_levers=none tier=None"
    rep2 = dict(rep, package_levers=[], tier="2")
    assert report.activation_line(rep2).endswith("applied=deferred package_levers=none tier=2")
    assert report.activation_line(dict(rep, active=False, reason="no visible GPU")).startswith("[pxdesign-opt] NOT ACTIVE: no visible GPU")
    assert report.activation_line(dict(rep, dry_run=True, active=False)).startswith("[pxdesign-opt] DRY-RUN mode=exact")
