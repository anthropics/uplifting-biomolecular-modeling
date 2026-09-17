"""The core pin: opt/pyproject.toml [tool.opt_core] names a path + a minimum version that the imported opt_core satisfies, the build
backend == the core's template byte for byte, the kit's finder holds the property the core's finder lacks (a find_spec probe must not
disarm it)."""
import importlib
import importlib.util
import os
import sys

from opt_core import autoload as core_autoload
from opt_core import gates, home

from .. import _autoload, _core, _core_gate, stack


def test_pin_matches_the_imported_core():
    g = gates.core_pin_check(_core.PYPROJECT)
    assert g.ok, g.reason
    pinned = gates.core_pin(_core.PYPROJECT)
    assert gates.version_tuple(gates.imported_core()["version"]) >= gates.version_tuple(pinned["version"])   # the pin is a floor (_core_gate: pinned >= v<want>)
    assert _core.origin() in ("installed", "pin_path")
    assert os.path.isdir(os.path.join(_core.pin_path(), "opt_core"))


def test_backend_is_the_core_template_byte_for_byte():
    template = os.path.join(os.path.dirname(os.path.abspath(home.__file__)), os.pardir, "kit_template", "_build_backend.py")
    assert open(os.path.join(stack.opt_root(), "_build_backend.py"), "rb").read() == open(template, "rb").read()


def test_tree_root_ignores_a_box_wide_tree_variable(monkeypatch):
    """The tree is this package's own location — never MODEL_OPT (run.sh's variable names the tree run.sh runs; two trees on one box each
    resolve their own kit, FPF and stock directories)."""
    own = stack.tree_root()
    assert own == os.path.dirname(stack.opt_root()) and stack.kit_home().startswith(stack.opt_root()) and stack.fpf_home().startswith(stack.opt_root())
    monkeypatch.setenv("MODEL_OPT", "/elsewhere/rosettafold3")
    assert stack.tree_root() == own


def test_activation_report_carries_the_core_gate(monkeypatch):
    g = stack.core_gate()
    assert g["ok"] and g["reason"] is None
    assert set(g["pinned"]) == {"path", "version"} and set(g["imported"]) == {"version", "root"}
    assert _core_gate.version_tuple(g["imported"]["version"]) >= _core_gate.version_tuple(g["pinned"]["version"])   # the pin is a floor


def test_both_finders_survive_a_probe(monkeypatch, tmp_path):
    """A probe before the first import (importlib.util.find_spec("rf3")) must leave the hook armed, else the following import runs
    unhooked — stock silently under the variable. The kit's own finder holds it (test_autoload.py); the core's (opt_core.autoload.Finder)
    holds it from 0.2.4 on — checked here against the pinned core. The kit keeps its own hook."""
    pkg = tmp_path / "rf3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    spec = core_autoload.AutoloadSpec(env=_autoload.ENV, package="rosettafold3_opt", tag="rosettafold3-opt", triggers=_autoload.TRIGGERS,
                                      modes=_autoload.MODES)
    f = core_autoload.install(spec, {"ROSETTAFOLD3_OPT": "exact"})
    try:
        assert f is not None and f.armed
        assert importlib.util.find_spec("rf3") is not None
        assert f.armed                                                    # the pinned core's finder survives the probe
    finally:
        f.remove()
    sys.modules.pop("rf3", None)
