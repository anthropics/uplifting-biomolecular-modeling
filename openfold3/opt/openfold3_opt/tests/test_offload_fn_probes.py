"""The offload port's _is_fn-based levers (trimul_hostsnap, triatt_lean, trans_inplace, cond_once, input_rows, recycle_rows, templ_host):
a class attribute holding the port's own function, DIRECTLY or through a transparent wrapper (a phase-timing caller installed after
the port's own hook, calling through to what it wraps -- a timer of that convention
closes over its `orig` exactly this way, no functools.wraps) -- must read as applied either way; a genuinely different
function occupying the slot must read as unavailable, with the actual occupant named in the evidence (not just "no"). No openfold3, no
GPU: fakes the two class attributes the port ever proble-checks (a plain function) and of3_offload's own module record."""
import sys
import types

from openfold3_opt import stack


def _fake_offload(**extra):
    ns = types.SimpleNamespace(_APPLIED=True, **extra)
    sys.modules["of3_offload"] = ns
    return ns


def _fake_model_module(run_trunk_fn):
    class OpenFold3:
        pass
    OpenFold3.run_trunk = run_trunk_fn
    mod = types.SimpleNamespace(OpenFold3=OpenFold3)
    sys.modules["openfold3.projects.of3_all_atom.model"] = mod
    return OpenFold3


def _clear():
    for k in ("of3_offload", "openfold3.projects.of3_all_atom.model"):
        sys.modules.pop(k, None)


def test_direct_identity_is_applied():
    def offload_run_trunk(self, *a, **k):
        pass
    _fake_offload(run_trunk=offload_run_trunk)
    _fake_model_module(offload_run_trunk)
    try:
        assert stack.lever_applied("recycle_rows") is True
    finally:
        _clear()


def test_transparent_wrapper_reads_as_applied_not_unavailable():
    """An external caller/timer wrapper that CALLS THROUGH to the port's function (the tree's own convention: a plain closure over `orig`, no
    functools.wraps) must not make a live lever read as unavailable -- it is still the port's code running, one frame further in."""
    def offload_run_trunk(self, *a, **k):
        pass

    def install_timer(orig):
        def run_trunk(self, *a, **k):                 # closes over `orig` -- the phase-timer shape
            return orig(self, *a, **k)
        return run_trunk

    _fake_offload(run_trunk=offload_run_trunk)
    cls = _fake_model_module(install_timer(offload_run_trunk))
    try:
        assert stack.lever_applied("recycle_rows") is True
        r = stack.levers_record({"levers_requested": ["recycle_rows"]})
        assert r["levers_applied"] == ["recycle_rows"] and r["levers_unavailable"] == []
    finally:
        _clear()


def test_double_wrapped_still_reachable():
    """Two layers of transparent wrapping (e.g. an external caller wrapper installed on top of the port's own run_trunk_timed) still resolve."""
    def offload_run_trunk(self, *a, **k):
        pass

    def wrap(orig):
        def inner(self, *a, **k):
            return orig(self, *a, **k)
        return inner

    _fake_offload(run_trunk=offload_run_trunk)
    _fake_model_module(wrap(wrap(offload_run_trunk)))
    try:
        assert stack.lever_applied("recycle_rows") is True
    finally:
        _clear()


def test_genuinely_untouched_is_unavailable_with_the_stock_qualname_named():
    """upstream's own run_trunk (never patched -- a real miss, not a wrapper): unavailable, and the evidence names it by
    module.qualname so 'genuinely never installed' is visually distinct from 'installed and re-wrapped'."""
    def offload_run_trunk(self, *a, **k):
        pass

    def stock_run_trunk(self, *a, **k):
        pass
    stock_run_trunk.__module__ = "openfold3.projects.of3_all_atom.model"
    stock_run_trunk.__qualname__ = "OpenFold3.run_trunk"

    _fake_offload(run_trunk=offload_run_trunk)
    _fake_model_module(stock_run_trunk)
    try:
        assert stack.lever_applied("recycle_rows") is False
        r = stack.levers_record({"levers_requested": ["recycle_rows"]})
        assert r["levers_unavailable"] == ["recycle_rows"]
        assert "openfold3.projects.of3_all_atom.model.OpenFold3.run_trunk" in r["lever_evidence"]["recycle_rows"]
    finally:
        _clear()


def test_a_different_functions_wrapper_is_not_a_false_positive():
    """A wrapper that closes over some OTHER function entirely (not the port's) must not read as reachable -- the closure walk finds
    the real callable it wraps, and that callable genuinely is not this lever's target. The evidence names the OUTERMOST occupant
    (what is literally bound to the attribute) -- it does not claim to be the port's own function."""
    def offload_run_trunk(self, *a, **k):
        pass

    def unrelated(self, *a, **k):
        pass

    def wrap(orig):
        def inner(self, *a, **k):
            return orig(self, *a, **k)
        return inner

    _fake_offload(run_trunk=offload_run_trunk)
    _fake_model_module(wrap(unrelated))
    try:
        assert stack.lever_applied("recycle_rows") is False
        r = stack.levers_record({"levers_requested": ["recycle_rows"]})
        evidence = r["lever_evidence"]["recycle_rows"]
        assert "inner" in evidence and "offload_run_trunk" not in evidence
    finally:
        _clear()


def test_offload_fn_detail_reports_absent_when_class_unresolvable():
    _fake_offload(run_trunk=lambda self, *a, **k: None)
    # no openfold3.projects.of3_all_atom.model registered at all
    sys.modules.pop("openfold3.projects.of3_all_atom.model", None)
    try:
        assert stack._offload_fn_detail("recycle_rows") == "absent"
    finally:
        _clear()


def test_offload_fn_detail_is_none_for_a_lever_outside_the_table():
    assert stack._offload_fn_detail("chunk_pin") is None                      # not an _is_fn-based probe -- no target row, no detail to give
    assert stack._offload_fn_detail("not_a_real_lever") is None
