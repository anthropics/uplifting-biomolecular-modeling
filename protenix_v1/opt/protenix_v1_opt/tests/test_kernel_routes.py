"""The three kernels route BY NAME to the core copies (opt_core.kernels): the core copies hold to their sums files, nothing under the
kit's lib/ answers to a routed name, the route resolves to the core copy without importing, the kit's cell table is the export, and
the KERNELS line names every resolved path."""
import os
import sys

import pytest
from opt_core import kernels as KR

from protenix_v1_opt import kit as K
from protenix_v1_opt import report as R
from protenix_v1_opt import stack

from .conftest import KIT


def _core_kernel_drift():
    """None if every routed kernel's live core bytes still match the core's own SUMS record; else the first problem found, e.g.
    'fpf_trimul/trimul.py: differs from SUMS' -- a drift INSIDE common/opt_core (its SUMS file vs its own kernel file), never
    this kit's PINS.json. Not this kit's to fix (opt_core is a shared dependency); the three tests below need it to be clean before
    route_kernels() will even resolve, so they skip by name, live, while the core's record and its kernel files disagree."""
    for name in stack.ROUTED_KERNELS:
        problems = KR.verify_carry(name)
        if problems:
            return problems[0]
    return None


_CORE_DRIFT = _core_kernel_drift()


def _export_vars():
    """Every environment name the routed kernels' SUMS export specs declare (the kit's TriMul cell table + the core's own data files)."""
    out = set()
    for name in stack.ROUTED_KERNELS:
        out.update((KR.sums(name).get("exports") or {}).keys())
    return sorted(out)


_EXPORT_VARS = _export_vars()


@pytest.fixture(autouse=True)
def _no_routes():
    saved = list(sys.meta_path); env = {k: os.environ.get(k) for k in _EXPORT_VARS}
    for name in stack.ROUTED_KERNELS:
        sys.modules.pop(name, None)
    yield
    sys.meta_path[:] = saved
    for k, v in env.items():
        (os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))


@pytest.mark.skipif(_CORE_DRIFT is not None, reason=f"opt_core kernel drift, core-side: {_CORE_DRIFT}")
def test_core_copies_hold_and_the_kit_dirs_answer_no_routed_name():
    for name in stack.ROUTED_KERNELS:
        assert KR.verify_carry(name) == [], name                                            # the core copy is its sums file's bytes
        assert KR.resolve(name, K.kit_sys_paths(KIT)) is None, name                          # unrouted, the kit's own directories do not serve the name: one copy, the core's
    assert "fpf_trimul" in stack.ROUTED_KERNELS and "fpf_trimul_v4" not in stack.ROUTED_KERNELS      # kit 0.2.38: the TriMul provider reaches its v4 row through the core's own path and cell tables (no kit table, no by-name import)


@pytest.mark.skipif(_CORE_DRIFT is not None, reason=f"opt_core kernel drift, core-side: {_CORE_DRIFT}")
def test_routes_resolve_to_the_core_and_check_before_import():
    out = stack.route_kernels(KIT)
    assert set(out["routed"]) == set(stack.ROUTED_KERNELS)
    for name, v in out["routed"].items():
        assert v["resolved"] == v["core_copy"] == KR.carried_path(name), v
        assert name not in sys.modules
    from opt_core.attn import pair_fused as PF
    assert out["exports"] == PF.carried_exports(list(stack.ROUTED_KERNELS)) and "FPF_TRIMUL_V4_CELLS" not in out["exports"]   # the core's own data files; no kit TriMul cell table
    assert set(out["exports"]) == set(_EXPORT_VARS) and all(os.path.isfile(v) for v in out["exports"].values())
    assert all(os.environ[k] == v for k, v in out["exports"].items())
    assert KR.routed() == sorted(stack.ROUTED_KERNELS)
    line = R.kernels_line({"kernels": out})
    assert line.startswith("[protenix-v1-opt] KERNELS flash_triattn=") and all(f"{n}={v['resolved']}" in line for n, v in out["routed"].items()) and line.endswith(")")


@pytest.mark.skipif(_CORE_DRIFT is not None, reason=f"opt_core kernel drift, core-side: {_CORE_DRIFT}")
def test_an_unrouted_kit_copy_is_still_the_kit_path():
    """Only the routed names go to the core: another kit module resolves through sys.path (the kit dirs), never the finder."""
    stack.route_kernels(KIT)
    assert KR.resolve("fastln_prebuilt", [os.path.join(KIT, "lib")]) == os.path.join(KIT, "lib", "fastln_prebuilt.py")


def test_a_missing_export_refuses_by_name(monkeypatch):
    from opt_core.attn import pair_fused as PF
    needed = sorted((n, var) for n in stack.ROUTED_KERNELS for var, spec in (KR.sums(n).get("exports") or {}).items() if spec.get("required"))
    if not needed:
        pytest.skip("no routed kernel has a required export at this core")
    monkeypatch.setattr(PF, "carried_exports", lambda names: {})
    name, var = needed[0]
    with pytest.raises(stack.ActivationError, match=f"{name} exports: {var} not exported"):
        stack.route_kernels(KIT)
