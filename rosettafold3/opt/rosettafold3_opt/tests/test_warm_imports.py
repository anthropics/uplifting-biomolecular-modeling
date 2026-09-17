"""stack.warm_imports: the activation-time stock-library import (opt_core.warm_imports, core >= 0.5.66.0) — a report dict, never an exception, a named step-aside on a core without the face."""
import sys
import types

from rosettafold3_opt import stack


def test_warm_imports_reports_and_never_raises():
    rep = stack.warm_imports()
    assert isinstance(rep, dict) and rep["state"].split(":")[0] in {"on", "absent", "error"}
    if rep["state"] == "on":
        assert isinstance(rep["report"], dict)


def test_a_core_without_the_face_steps_aside_by_name(monkeypatch):
    fake = types.ModuleType("opt_core")
    monkeypatch.setitem(sys.modules, "opt_core", fake)
    assert stack.warm_imports() == {"state": "absent:opt_core.warm_imports"}


def test_a_raising_face_is_reported_not_raised(monkeypatch):
    fake = types.ModuleType("opt_core")
    def boom(**kw):
        raise RuntimeError("x")
    fake.warm_imports = boom
    monkeypatch.setitem(sys.modules, "opt_core", fake)
    assert stack.warm_imports() == {"state": "error:RuntimeError"}


def test_torch_is_asked_before_the_ops_library(monkeypatch):
    """libcue_ops.so resolves libnvrtc/libcudart through torch's loading: the kit asks torch first, the ops library second, cuequivariance_torch last."""
    seen = {}
    fake = types.ModuleType("opt_core")
    def rec(libraries=None, origin=None, **kw):
        seen["libraries"] = tuple(libraries); seen["origin"] = origin
        return {n: "absent" for n in libraries}
    fake.warm_imports = rec
    monkeypatch.setitem(sys.modules, "opt_core", fake)
    rep = stack.warm_imports()
    assert rep["state"] == "on"
    assert seen["origin"] == "kit"
    libs = seen["libraries"]
    assert libs[0] == "torch" and libs.index("cuequivariance_ops_torch") < libs.index("cuequivariance_torch")
    assert stack.WARM_LIBRARIES == libs
