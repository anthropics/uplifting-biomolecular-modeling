"""The core adoption locks: the pin is a FLOOR the imported core satisfies (installed >= [tool.opt_core] version), the autoload hook is the kit's own
(no opt_core.autoload), and the stock process holds no core module beyond the proof's."""
import os

from opt_core import gates
from opt_core.stock_proof import CORE_ALLOWED_IN_STOCK

from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def _v(word):
    return tuple(int(x) for x in str(word).split(".") if x.isdigit())


def test_pin_is_a_floor_the_imported_core_satisfies():
    gate = gates.core_pin_check(os.path.join(HOME, "opt", "pyproject.toml"))
    assert gate.ok, gate.reason
    pin = gates.core_pin(os.path.join(HOME, "opt", "pyproject.toml"))
    assert _v(gates.imported_core()["version"]) >= _v(pin["version"]), (gates.imported_core()["version"], pin["version"])   # the kit's [tool.opt_core] pin is a MINIMUM version: a newer core in the tree satisfies it


def test_stock_child_holds_only_the_allowed_core_modules(tmp_path):
    """The stock child, run to its proof with a stub `openfold3` (test_off_env_proof._run_stock): the census of core modules in the
    proof is empty (opt_core.stock_proof.CORE_ALLOWED_IN_STOCK is the allowance), before the stock call and after it, and the proof
    is fail-closed on it."""
    from openfold3_ob0_opt.tests.test_off_env_proof import _run_stock, _stub_openfold3
    stub = _stub_openfold3(str(tmp_path / "site"))
    out = str(tmp_path / "out")
    r, proof = _run_stock({"PYTHONPATH": _stubs.subprocess_pythonpath(HOME, stub), "OPENFOLD3_OB0_CKPT": "/w.pt"},
                          stock_args=("--query-json", "q.json", "--output-dir", out))
    assert r.returncode == 0, r.stderr[-800:]
    assert proof["core_modules_loaded"] == [] and proof["core_modules_loaded_after"] == [] and proof["ok"] and proof["ok_after"]
    assert "core modules" not in r.stderr


def test_stock_child_refuses_a_core_module_beyond_the_proof(tmp_path):
    """A core module loaded before the proof (here: opt_core.home, imported by a sitecustomize on the child's path) is a NOT STOCK
    refusal, exit 3, named in the line and the proof."""
    from openfold3_ob0_opt.tests.test_off_env_proof import _run_stock
    site = str(tmp_path / "sc"); os.makedirs(site)
    with open(os.path.join(site, "sitecustomize.py"), "w") as fh:
        fh.write("import opt_core.home\n")
    r, proof = _run_stock({"PYTHONPATH": os.pathsep.join([site, os.path.join(HOME, "opt"), _stubs.core_dir()])})
    assert r.returncode == 3 and "NOT STOCK" in r.stderr and "['opt_core.home']" in r.stderr, r.stderr[-600:]
    assert proof["core_modules_loaded"] == ["opt_core.home"] and proof["ok"] is False
