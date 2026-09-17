"""fpf_glue_v2 / fpf_mkpf cell resolution under P12: the exact (cc, triton) entries resolve as recorded; the capability's cells APPLY on an untested triton, named
(the =0 switch opts out by name); a capability certified engines run with the overlay off is named `off: not-measured`; an unknown capability is named `unknown
capability` (the shipped kernels serve).  CPU only: cc and triton are named."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


def _routed(name):
    from opt_core import kernels as CK
    core_copy = os.path.join(os.path.dirname(CK.__file__), name)
    top = sys.modules.get(name)
    if top is not None and not os.path.abspath(getattr(top, "__file__", "") or "").startswith(os.path.abspath(core_copy)):
        for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[m]
    CK.route(name)
    import importlib
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def G():
    return _routed("fpf_glue_v2")


@pytest.fixture(scope="module")
def M():
    return _routed("fpf_mkpf")


@pytest.fixture(autouse=True)
def _env():
    yield
    for v in ("FPF_GLUE_V2_ALLOW_TRITON", "FPF_MKPF_ALLOW_TRITON"):
        os.environ.pop(v, None)


def test_glue_exact_entries_resolve_as_recorded(G, capfd):
    for cc, tt, ent in (("10.0", "3.7", "sm100_t37"), ("9.0", "3.7", "sm90_t37"), ("9.0", "3.3", "sm90_t33"), ("10.0", "3.6", "sm100_t36"), ("8.0", "3.7", "sm80_t37")):
        cells = G.resolve_cells(cc=cc, triton=tt)
        assert set(cells) >= {"prologue", "epilogue"} and G._STATE["why"] == f"cc {cc} triton {tt} -> entry '{ent}'"
    assert capfd.readouterr().err == ""


def test_glue_untested_triton_applies_the_capabilitys_cells_named(G, capfd):
    cells = G.resolve_cells(cc="9.0", triton="3.9")
    assert set(cells) >= {"prologue", "epilogue"} and G._STATE["why"] == "cc 9.0 triton 3.9 -> entry 'sm90_t37' (triton 3.9 unverified)"
    assert "WARNING: cc 9.0 cells were measured on triton" in capfd.readouterr().err
    os.environ["FPF_GLUE_V2_ALLOW_TRITON"] = "0"
    assert G.resolve_cells(cc="9.0", triton="3.9") == {} and G._STATE["why"] == "triton 3.9 unverified, opted out (FPF_GLUE_V2_ALLOW_TRITON=0)"
    assert "cells NOT applied" in capfd.readouterr().err


@pytest.mark.parametrize("cc", ["10.3"])                        # cc 8.0 has measured prologue / epilogue cells since the sm80 entry; 10.3 stays off: not-measured
def test_glue_measured_off_capability_is_named(G, cc):
    assert G.resolve_cells(cc=cc, triton="3.7") == {} and G._STATE["why"].startswith(f"no cells for cc {cc} (off: not-measured")


def test_glue_unknown_capability_is_named(G):
    assert G.resolve_cells(cc="12.0", triton="3.7") == {} and G._STATE["why"].startswith("no cells for cc 12.0 (unknown capability: the shipped kernels serve")


def test_mkpf_resolution(M, capfd):
    assert "f1" in M.resolve_cells(cc="9.0", triton="3.3") and M._STATE["why"] == "9.0|3.3 -> 'sm90_t33'"
    assert "f1" in M.resolve_cells(cc="10.0", triton="3.7") and M._STATE["why"] == "10.0|3.7 -> 'sm100_t37'"
    assert capfd.readouterr().err == ""
    assert "f1" in M.resolve_cells(cc="10.0", triton="3.9") and M._STATE["why"] == "10.0|3.9 -> 'sm100_t37' (triton 3.9 unverified)"
    assert "WARNING: no pinned cells for 10.0|3.9" in capfd.readouterr().err
    os.environ["FPF_MKPF_ALLOW_TRITON"] = "0"
    assert M.resolve_cells(cc="10.0", triton="3.9") == {} and "FPF_MKPF_ALLOW_TRITON=0 refuses" in M._STATE["why"]
    os.environ.pop("FPF_MKPF_ALLOW_TRITON")
    assert M.resolve_cells(cc="10.3", triton="3.7") == {} and M._STATE["why"].startswith("no MKPF cells for cc 10.3 (off: not-measured")
    assert M.resolve_cells(cc="8.6", triton="3.7") == {} and M._STATE["why"].startswith("no MKPF cells for cc 8.6 (unknown capability")
