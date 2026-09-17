"""Start-up: the core's import warm-up (`openfold3_opt.warm_core_imports`, called once by `enable()` in every mode except `off`): torch is asked
before the cuEquivariance ops library (the order is load-bearing on cu13 wheels), a core without `warm_imports` (older than opt_core 0.5.66.0)
is tolerated, a `warm_imports` without the `libraries` parameter is called plain, and `off` never warms.  No CUDA, no real core call."""
import sys
import types

import openfold3_opt as pkg


def test_a_core_without_the_attribute_is_tolerated():
    fake = types.ModuleType("opt_core")                       # an older core: no warm_imports attribute
    assert pkg.warm_core_imports(fake) is None


def test_torch_is_asked_before_the_ops_library():
    """libcue_ops.so resolves libnvrtc / libcudart through torch's loading: torch first, the ops library second, cuequivariance_torch last."""
    seen = {}
    fake = types.ModuleType("opt_core")
    def rec(libraries=None, rows=None, *, origin="kit"):
        seen["libraries"] = tuple(libraries); return {n: "absent" for n in libraries}
    fake.warm_imports = rec
    rep = pkg.warm_core_imports(fake)
    libs = seen["libraries"]
    assert libs == pkg.WARM_LIBRARIES and libs[0] == "torch" and libs.index("cuequivariance_ops_torch") < libs.index("cuequivariance_torch")
    assert rep == {n: "absent" for n in libs}


def test_a_warm_imports_without_the_libraries_parameter_is_called_plain():
    calls = []
    fake = types.ModuleType("opt_core")
    fake.warm_imports = lambda: (calls.append(1), "warm")[1]
    assert pkg.warm_core_imports(fake) == "warm" and calls == [1]


def test_enable_off_never_warms(monkeypatch):
    """`off` is the stock route: nothing of the core is warmed on it (enable("off") returns the inactive report before the call site)."""
    called = []
    monkeypatch.setattr(pkg, "warm_core_imports", lambda core=None: called.append(1))
    rep = pkg.enable("off")
    assert called == [] and rep.get("active") is False
