"""Model-specific: the three hooks' order on sys.path, OF3T_KIT_LEVERS, the exit tally after the hooks, and det.py's recipe."""
import os
import sys

import pytest

from openfold3_ob0_opt import det, hooks, modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


@pytest.fixture
def pkg():
    d = _stubs.stub_dist()
    pkg = _stubs.reset_package()
    yield pkg
    _stubs.reset_package()
    _stubs.unstub_dist(d)


def test_sys_path_order_matches_the_line(pkg):
    rep = pkg.enable("exact")
    assert rep["active"] and rep["line"] == "cueq"
    want = [os.path.join(HOME, modes.CELLS_HOOK_DIR), os.path.join(HOME, "opt", "forward", "trunk_kernels", "of3t_hook"), os.path.join(HOME, "opt", "forward", "fast_inference", "of3_levers")]
    assert rep["hook_dirs"] == want
    idx = [sys.path.index(d) for d in want]
    assert idx == sorted(idx)                                                     # the line's order on sys.path
    assert rep["entry_hook"] == os.path.join(want[0], "sitecustomize.py")
    assert os.environ["OF3T_KIT_LEVERS"] == want[2]
    inst = {h["kit"]: h for h in hooks.installed()}
    assert inst["trunk_kernels"]["targets"] == "openfold3.projects.of3_all_atom.model"


def test_fast_line_hook_dirs_are_the_kits_own_directories(pkg):
    """The fast line: the package's cells hook first, then the trunk-kernels add-on's hook, then the fast-inference add-on's lever directory (chained
    through OF3T_KIT_LEVERS) — one hook directory per add-on, the add-ons' own."""
    rep = pkg.enable("fast")
    fast = modes.LINES[("fast", None)]
    assert rep["active"] and os.environ["OF3T_KIT_LEVERS"] == modes.hook_dir(HOME, "fast_inference", line=fast) == modes.hook_dir(HOME, "fast_inference")
    assert rep["hook_dirs"] == [os.path.join(HOME, modes.CELLS_HOOK_DIR), modes.hook_dir(HOME, "trunk_kernels"), os.environ["OF3T_KIT_LEVERS"]]


def test_det_recipe():
    assert det.ENV == {"OF3_DETERMINISTIC": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    e = {}
    assert det.apply_env(0, environ=e) == {} and e == {}
    assert det.apply_env(1, graphed=True, environ=e) == {"OF3_DETERMINISTIC": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OF3_GRAPHS_STRICT": "1"}
    e2 = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    assert det.apply_env(1, environ=e2) == {"OF3_DETERMINISTIC": "1"} and e2["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"     # a caller's value is kept
    assert det.level(1) == 1 and det.LEVELS == (0, 1)
    with pytest.raises(ValueError):
        det.level(2)
    assert os.path.isfile(os.path.join(det.det_site(HOME), "sitecustomize.py"))       # the det site: the one directory the stock child's carve-out allows


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_pred_route_runs_the_kits_own_levers(pkg, mode):
    """`enable()` (pred / check / warm: one job per process, the runs as shipped): the exact line puts the shipped-kit `of3_levers` on the path and
    `of3_graphs` resolves there; the fast line adds the package's cells hook ahead of them."""
    import importlib.util
    rep = pkg.enable(mode)
    assert rep["active"]
    spec = importlib.util.find_spec("of3_graphs")
    assert rep["hook_dirs"] == [os.path.join(HOME, modes.CELLS_HOOK_DIR), modes.hook_dir(HOME, "trunk_kernels"), modes.hook_dir(HOME, "fast_inference")]   # both lines enter through the package's cells hook
    assert os.path.abspath(spec.origin).startswith(os.path.abspath(rep["hook_dirs"][-1]))                          # of3_graphs resolves in the fast-inference kit's lever directory
    for kit in modes.KITS:
        assert modes.hook_dir(HOME, kit) == os.path.join(HOME, modes.KITS[kit], modes.HOOK_DIR[kit])
    assert modes.ERRATA == {}


