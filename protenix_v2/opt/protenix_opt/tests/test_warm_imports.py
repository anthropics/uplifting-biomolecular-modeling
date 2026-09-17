"""The activation route's early library import (stack.warm_imports -> the shared core's ``opt_core.warm_imports()``): one
``[protenix-opt] WARM imports: …`` line carrying the core's words, the words kept as the activation report's ``warm_imports``, and the
by-name step-aside on a shared core that predates the function — a line, never a refusal, never an exception. CPU only: the core's
function is replaced by a recorder here (the real one imports the CUDA libraries where they are installed), and an older core is a stub
module under the core's name."""
import io
import sys
import types

from protenix_opt import _core, stack  # noqa: F401  (_core first: opt_core importable when it is not installed)
import opt_core


def test_line_carries_the_cores_words(monkeypatch):
    seen = {}

    def fake(libraries=None, rows=None, *, origin="kit"):
        seen["origin"], seen["libraries"] = origin, libraries
        return {"torch": "present", "cuequivariance_ops_torch": 1.5, "cuequivariance_torch": "present"}

    monkeypatch.setattr(opt_core, "warm_imports", fake, raising=False)      # bound in the package dict: the core's lazy __getattr__ is never reached
    out = io.StringIO()
    rep = stack.warm_imports(stream=out)
    line = out.getvalue()
    assert line.startswith(f"{stack.LOG} {stack.WARM_LINE} "), line
    assert line.count("\n") == 1, line
    assert "torch=present cuequivariance_ops_torch=1.500s cuequivariance_torch=present" in line, line
    assert rep == {"torch": "present", "cuequivariance_ops_torch": 1.5, "cuequivariance_torch": "present"}
    assert seen["origin"] == "protenix_opt"
    assert tuple(seen["libraries"]) == stack.WARM_LIBRARIES and stack.WARM_LIBRARIES[0] == "torch", seen["libraries"]   # torch ahead of the cuEquivariance ops, by name


def test_a_core_function_without_the_libraries_word_steps_aside_by_name(monkeypatch):
    monkeypatch.setattr(opt_core, "warm_imports", lambda *a, **k: (_ for _ in ()).throw(TypeError("unexpected keyword argument 'libraries'")), raising=False)
    out = io.StringIO()
    rep = stack.warm_imports(stream=out)
    line = out.getvalue()
    assert line.startswith(f"{stack.LOG} {stack.WARM_LINE} unavailable(opt_core {opt_core.__version__}: warm_imports without libraries=)"), line
    assert line.count("\n") == 1 and list(rep) == ["unavailable"]


def test_older_core_steps_aside_by_name(monkeypatch):
    old = types.ModuleType("opt_core")
    old.__version__ = "0.5.40.0"                                            # a core below stack.WARM_MIN_CORE: no warm_imports attribute at all
    monkeypatch.setitem(sys.modules, "opt_core", old)
    out = io.StringIO()
    rep = stack.warm_imports(stream=out)                                    # must not raise, must not exit
    line = out.getvalue()
    assert line.startswith(f"{stack.LOG} {stack.WARM_LINE} unavailable(opt_core 0.5.40.0 < {stack.WARM_MIN_CORE}"), line
    assert line.count("\n") == 1
    assert rep == {"unavailable": f"opt_core 0.5.40.0 < {stack.WARM_MIN_CORE}"}


def test_the_pinned_core_carries_it():
    """The kit's pin is at or above the first core with the function, and the importable core has it (the step-aside is for a tree run
    against an older core by hand, never the shipped pairing)."""
    from protenix_opt._core_gate import read_table, version_tuple
    pin = read_table(_core.PYPROJECT, "tool.opt_core")["version"]
    assert version_tuple(pin) >= version_tuple(stack.WARM_MIN_CORE), pin
    assert callable(getattr(opt_core, "warm_imports", None)), opt_core.__version__


def test_the_levers_are_applied_after_it():
    """By source: the one call sits in _apply before the kit's sitecustomize runs (the sitecustomize, the levers and the model import torch
    and the libraries from depths of their own there), after _apply's own pre-activation refusals; the dry run (`check`) never reaches it."""
    import inspect
    src = inspect.getsource(stack._apply)
    i_refuse, i_warm, i_site = src.index("was imported before activation"), src.index("warm_imports()"), src.index("_run_kit_sitecustomize(fpf_home)")
    assert i_refuse < i_warm < i_site
    assert "warm_imports" not in inspect.getsource(stack._dry_run)
