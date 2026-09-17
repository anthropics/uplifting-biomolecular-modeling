"""LEVER-row bookkeeping for levers whose install record is not the served-levers hook's: ARM U's sub-arms (`arm_u23`, the arm's own
COUNTS) and the exact line's FPF TriMul adapter (`fpf_trimul_exact`, fpf_engines' bound ops). A lever the KERNELS / EXIT lines show
serving prints ``state=on``; one that did not engage is a named fallback — never ``skipped: planned, no install record``."""
import sys
import types

from opendde_opt import modes, stack
from opendde_opt import report as _report


def _hook(arm: str):
    m = types.ModuleType("odde_served_levers")
    m.STATE = {"version": "0.2.0", "active": True, "wrapped": True, "errors": {}, "deterministic_forced": False,
               "installs": [{"event": "install", "requested": {"addon_levers": ["dit_hoist", "dit_align"], "arm": arm.upper()},
                             "levers": {"kit.cueq_cache": "/x/cueq_cache_shipped", "ditfast": {"dit_hoist": "installed", "dit_align": True},
                                        "odde_arm_t": {"arm": arm.upper(), "version": "0.3", "composition": None, "min_tokens": 300}}}]}
    return m


def test_arm_u23_serving_is_on_and_a_dead_sub_arm_is_named(monkeypatch):
    line = modes.LINES["LSTAR2A"]
    monkeypatch.setitem(sys.modules, "odde_served_levers", _hook("u"))
    arm = types.ModuleType("odde_arm_t"); arm.COUNTS = {"arm": "u", "u2_trimul": True, "u3": True, "trimul_bf16_calls": 1168}
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    monkeypatch.setenv("ODDE_ADDON_LEVERS", "dit_hoist,dit_align")
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "mode": "fast", "line": line.name, "levers": list(line.levers), "levers_planned": list(line.levers),
                                           "levers_applied": list(line.levers), "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "arm_u" in rep["levers_applied"] and "arm_u23" in rep["levers_applied"] and not [f for f in rep["levers_fallback"] if f.startswith("arm_u23")]
    assert _report.lever_state("arm_u23", rep) == ("on", None)
    row = [ln for ln in _report.lever_lines(rep, stats={}) if " name=arm_u23 " in ln][0]
    assert row.startswith("[opendde-opt] LEVER name=arm_u23 state=on "), row
    arm.COUNTS = {"arm": "u", "u2_trimul": False, "u3": False}                     # the arm serves, its sub-arms did not engage: named, partial
    rep = stack.refresh()
    assert "arm_u23" not in rep["levers_applied"] and rep["partial"] and "arm_u23:u2_trimul=False u3=False" in rep["levers_fallback"]
    assert _report.lever_state("arm_u23", rep)[0] == "skipped" and "fell back" in _report.lever_state("arm_u23", rep)[1]


def test_fpf_trimul_exact_bound_is_on_and_unbound_is_named(monkeypatch):
    line = modes.LINES["S1"]
    monkeypatch.setitem(sys.modules, "odde_served_levers", _hook("z"))
    eng = types.ModuleType("fpf_engines"); eng.__version__ = "0.1.2"; eng.STATS = {}
    eng._ORIG = {("opendde", "trimul_out"): (object, None), ("opendde", "trimul_in"): (object, None)}
    monkeypatch.setitem(sys.modules, "fpf_engines", eng)
    xl = types.ModuleType("odde_xl"); xl._INSTALLED = ("tri_ln",); monkeypatch.setitem(sys.modules, "odde_xl", xl)   # the XL unit's own record
    monkeypatch.setenv("ODDE_ADDON_LEVERS", "dit_hoist,dit_align")
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "mode": "exact", "line": line.name, "levers": list(line.levers), "levers_planned": list(line.levers),
                                           "levers_applied": list(line.levers), "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "fpf_trimul_exact" in rep["levers_applied"] and not [f for f in rep["levers_fallback"] if f.startswith("fpf_trimul_exact")]
    assert _report.lever_state("fpf_trimul_exact", rep) == ("on", None)
    eng._ORIG = {}                                                                  # imported, nothing bound: a fallback by name
    rep = stack.refresh()
    assert "fpf_trimul_exact" not in rep["levers_applied"] and rep["partial"] and "fpf_trimul_exact:no trimul op bound (ops=none)" in rep["levers_fallback"]
    monkeypatch.delitem(sys.modules, "fpf_engines")
    rep = stack.refresh()
    assert "fpf_trimul_exact:fpf_engines not imported" in rep["levers_fallback"]
