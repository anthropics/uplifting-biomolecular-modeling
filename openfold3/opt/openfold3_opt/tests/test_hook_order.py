"""Model-specific: the three hooks' order on sys.path, OF3T_KIT_LEVERS, the exit tally after the hooks, and det.py's recipe."""
import os
import sys

import pytest

from openfold3_opt import det, hooks, modes
from openfold3_opt.tests import _stubs

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


def test_fast_kit_levers_points_at_the_kits_levers(pkg):
    """The fast line's chained levers are the fast-inference kit's `of3_levers`, its trunk hook the trunk-kernels add-on's `of3t_hook` (one copy of each add-on)."""
    rep = pkg.enable("fast")
    assert rep["active"] and os.environ["OF3T_KIT_LEVERS"] == modes.hook_dir(HOME, "fast_inference") == os.path.join(HOME, modes.KITS["fast_inference"], "of3_levers")
    assert rep["hook_dirs"] == [os.path.join(HOME, modes.CELLS_HOOK_DIR), os.path.join(HOME, modes.KITS["trunk_kernels"], "of3t_hook"), os.environ["OF3T_KIT_LEVERS"]]


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
    assert os.path.isfile(det.carrier(HOME))


def test_det_carrier_installs_no_finder_without_a_lever_switch(pkg):
    """The stock arm under --det 1 runs the kit hook with no lever switch: the deterministic block (needs torch) is skipped here, and no
    finder is installed — the same code path the kit's own deterministic stock arm takes."""
    info = det.stock_arm_carrier(HOME)
    assert not info["lever_finder_installed"] and hooks.installed() == []
    assert sys.path[0] == os.path.dirname(det.carrier(HOME))


@pytest.mark.parametrize("mode, line", [("exact", "cueq"), ("fast", None)])
def test_pred_route_runs_the_kits_own_levers(pkg, mode, line):
    """`enable()` (pred / check / warm: one job per process, the runs as shipped): both lines put the fast-inference kit's `of3_levers` on the path and
    `of3_graphs` resolves there (one copy per add-on: every hook directory is its kit's own)."""
    import importlib.util
    rep = pkg.enable(mode)
    assert rep["active"] and "served" not in rep
    spec = importlib.util.find_spec("of3_graphs")
    levers = os.path.join(HOME, modes.KITS["fast_inference"], "of3_levers")
    want = [os.path.join(HOME, modes.CELLS_HOOK_DIR), os.path.join(HOME, modes.KITS["trunk_kernels"], "of3t_hook"), levers]   # both lines enter through the package's cells hook
    assert rep["hook_dirs"] == want and os.environ.get(modes.KIT_LEVERS_ENV) == levers
    assert os.path.abspath(spec.origin).startswith(levers)
    for kit in modes.KITS:
        assert modes.hook_dir(HOME, kit) == os.path.join(HOME, modes.KITS[kit], modes.HOOK_DIR[kit]) and modes.hook_dirs_of(HOME, kit) == [os.path.abspath(modes.hook_dir(HOME, kit))]


