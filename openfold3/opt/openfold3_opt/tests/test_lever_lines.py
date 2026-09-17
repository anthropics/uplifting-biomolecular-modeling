"""The per-lever activation-evidence lines (report.lever_lines): one `[openfold3-opt] LEVER name=<id> state=<on|off|skipped> ...` line per
registry lever for a kit-arm process, printed after the EXIT line; nothing for a process where nothing is active."""
from openfold3_opt import modes, report
from openfold3_opt.registry import LEVERS


def test_states_and_reasons():
    rep = {"active": True, "mode": "fast", "line": None, "levers_requested": ["fast_init", "templ_distinct", "paircache", "trimul_cueq"],
           "levers_applied": ["fast_init", "templ_distinct", "trimul_cueq"], "levers_unavailable": [], "size_gate": "graph=eager:n_tok>400", "gate_reason": "n_tok>cap",
           "n_tokens": 900, "graphs_cap": 400, "core_routes": []}
    lines = report.lever_lines(rep)
    assert len(lines) == len(LEVERS) + 1 and all(l.startswith("[openfold3-opt] LEVER name=") for l in lines)   # + the fast line's precision parameter line
    assert lines[-1].startswith("[openfold3-opt] LEVER name=precision_bf16 state=off reason=not_selected impl=pl_trainer_args.precision:bf16-mixed origin=kit mode=fast ")
    assert " state=on impl=" in report.lever_lines(dict(rep, precision="bf16"))[-1]
    by = {l.split("name=")[1].split()[0]: l for l in lines}
    assert by["fast_init"].startswith("[openfold3-opt] LEVER name=fast_init state=on impl=fast_inference origin=kit mode=fast ") and "reason=" not in by["fast_init"]   # the core grammar's order: name, state, [reason], impl, origin, then evidence
    assert "state=skipped reason=pending impl=" in by["paircache"]
    assert "state=off reason=graph_gate impl=" in by["cuda_graphs"] and "state=off reason=graph_gate" in by["graphs_strict"]
    assert " gate=n_tok>cap n_tokens=900 cap=400" in by["cuda_graphs"] and "gate=" not in by["graphs_strict"]                     # the gate's decision printed ONCE, on the cuda_graphs line
    assert "state=off reason=not_in_line" in by["trimul_hostsnap"] and by["trimul_cueq"].endswith("tier=tolerance")
    routed = {l.split("name=")[1].split()[0]: l for l in report.lever_lines(dict(rep, levers_applied=["triatt_block"], core_routes=["triatt_block"]))}
    assert " origin=core " in routed["triatt_block"] and " origin=kit " in routed["trimul_cueq"]
    rep["levers_unavailable"] = ["paircache"]
    assert "state=skipped reason=unavailable" in {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep)}["paircache"]


def test_unavailable_offload_lever_names_its_cause_on_the_line():
    """recycle_rows (an _OFFLOAD_FN_TARGETS lever) unavailable: the printed LEVER line's reason carries the concrete occupant
    (`unavailable:<module>.<qualname>`), not a bare `unavailable` -- a non-offload lever like paircache (test_states_and_reasons,
    above) has no such table entry and keeps the plain `unavailable`, unchanged."""
    import sys
    import types

    def offload_run_trunk(self, *a, **k):
        pass

    def stock_run_trunk(self, *a, **k):
        pass
    stock_run_trunk.__module__, stock_run_trunk.__qualname__ = "openfold3.projects.of3_all_atom.model", "OpenFold3.run_trunk"

    class OpenFold3:
        pass
    OpenFold3.run_trunk = stock_run_trunk
    sys.modules["openfold3.projects.of3_all_atom.model"] = types.SimpleNamespace(OpenFold3=OpenFold3)
    sys.modules["of3_offload"] = types.SimpleNamespace(_APPLIED=True, run_trunk=offload_run_trunk)
    try:
        rep = {"active": True, "mode": "big", "line": "resident", "levers_requested": ["recycle_rows"],
               "levers_applied": [], "levers_unavailable": ["recycle_rows"], "core_routes": []}
        by = {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep)}
        assert "state=skipped reason=unavailable:openfold3.projects.of3_all_atom.model.OpenFold3.run_trunk impl=offload" in by["recycle_rows"]
    finally:
        sys.modules.pop("openfold3.projects.of3_all_atom.model", None)
        sys.modules.pop("of3_offload", None)


def test_nothing_when_not_active():
    assert report.lever_lines(None) == [] and report.lever_lines({"active": False, "mode": "off"}) == []


def test_exit_tally_carries_them_after_the_exit_line(monkeypatch):
    monkeypatch.setattr(report, "kit_counters", lambda: {})
    txt = report.exit_tally_line({"active": True, "mode": "exact", "line": "sampler", "levers_requested": ["fast_init"], "levers_applied": ["fast_init"]}, 1)
    first, *rest = txt.splitlines()
    assert first.startswith("[openfold3-opt] exit mode=exact ") and len(rest) == len(LEVERS) and rest[0].startswith("[openfold3-opt] LEVER ")   # no precision line outside fast
    assert report.exit_tally_line({"active": False, "mode": "off"}, 0).count("\n") == 0
