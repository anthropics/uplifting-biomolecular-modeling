"""The multi-GPU line's rank-side routing (tp_route): the frozen ROUTES table accounts for every public function/class of the carried
modules it names (each is routed or in CARRIED with a reason); with the shared core's rowpair package importable, every routed name
exists in its core module; install() rebinds the carried names to the core's objects at the carried module's import and prints the
TP-ROUTE evidence line; bridge_env derives the core's rank environment from torchrun's; nothing is installed without the rank env."""
import ast
import importlib
import io
import os
import subprocess
import sys

import pytest

from protenix_opt import _autoload, tp, tp_route

UNIT = tp.unit_dir()


def _public_defs(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")}


def test_routes_account_for_every_public_function_of_the_carried_modules():
    for carried, (core_module, names) in tp_route.ROUTES.items():
        path = os.path.join(UNIT, *carried.split(".")) + ".py"
        assert os.path.isfile(path), path
        defs = _public_defs(path)
        routed = set(names)
        kept = {n for (m, n) in tp_route.CARRIED if m == carried}
        assert routed <= defs, f"{carried}: routed names not defined in the carried module: {sorted(routed - defs)}"
        assert defs == routed | kept, f"{carried}: public functions neither routed nor named CARRIED: {sorted(defs - routed - kept)}"
        assert not (routed & kept)
        assert len(names) == len(set(names))
        assert core_module.startswith(tp_route.CORE_PKG + ".")
    assert tp_route.ROUTED == ("dist", "blockreduce", "bcast", "contract")
    for (m, n), target in tp_route.ELSEWHERE.items():
        assert m in tp_route.ROUTES and n in tp_route.ROUTES[m][1] and target.startswith(tp_route.CORE_PKG + ".")


def _core_rowpair():
    try:
        return importlib.import_module(tp_route.CORE_PKG)
    except ImportError as e:
        pytest.skip(f"the pinned core has no {tp_route.CORE_PKG} ({e}): the routing needs the core that carries mem.rowpair")


def test_every_routed_name_exists_in_its_core_module():
    _core_rowpair()
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("the core's rowpair modules import torch lazily at first attribute use; checked by source instead below")
    for carried, (core_module, names) in tp_route.ROUTES.items():
        for n in names:
            target = tp_route.ELSEWHERE.get((carried, n), core_module)
            mod = importlib.import_module(target)
            assert hasattr(mod, n), f"{target} has no {n} (routed from {carried})"


def test_every_routed_name_exists_in_its_core_module_by_source():
    pkg = _core_rowpair()
    root = os.path.dirname(pkg.__file__)
    for carried, (core_module, names) in tp_route.ROUTES.items():
        for n in names:
            target = tp_route.ELSEWHERE.get((carried, n), core_module)
            defs = _public_defs(os.path.join(root, target.rsplit(".", 1)[1] + ".py"))
            assert n in defs, f"{target} does not define {n} (routed from {carried})"


def test_bridge_env_derives_the_core_rank_environment_from_torchruns():
    env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29642", "TORCHELASTIC_RUN_ID": "run:7/x"}
    set_ = tp_route.bridge_env(env)
    assert env["ROWPAIR_RANK"] == "1" and env["ROWPAIR_WORLD"] == "2" and env["ROWPAIR_LOCAL_RANK"] == "1"
    assert env["ROWPAIR_ADDR"] == "127.0.0.1" and env["ROWPAIR_PORT"] == "29642"
    assert env["ROWPAIR_STORE"].endswith(os.path.join("protenix_opt_rowpair-uid%d" % os.geteuid(), "run_7_x", "c10d_store")) and os.path.isdir(os.path.dirname(env["ROWPAIR_STORE"]))
    assert set(set_) == {"ROWPAIR_RANK", "ROWPAIR_WORLD", "ROWPAIR_LOCAL_RANK", "ROWPAIR_ADDR", "ROWPAIR_PORT", "ROWPAIR_STORE"}
    env2 = dict(env); env2["ROWPAIR_RANK"] = "0"; env2["ROWPAIR_STORE"] = "/x/store"
    assert "ROWPAIR_RANK" not in tp_route.bridge_env(env2) and env2["ROWPAIR_RANK"] == "0" and env2["ROWPAIR_STORE"] == "/x/store"
    assert tp_route.bridge_env({"PATH": "/bin"}) == {}                       # not a rank: untouched


def test_rank_env_selects_the_route_and_a_single_gpu_run_installs_nothing(monkeypatch):
    env = tp.rank_env(2, "/o", exports={}, base={"PATH": "/bin"})
    assert env[tp_route.ENV] == tp_route.WORD and tp_route.ENV in _autoload.DECLARED and tp_route.ENV == _autoload.ENV_TP_ROUTE
    monkeypatch.delenv(tp_route.ENV, raising=False)
    assert not tp_route.selected() and tp_route._patches == []            # this (P=1) process: nothing armed
    _autoload.route_tp_unit({})                                            # no rank env: a no-op
    assert tp_route._patches == []


def test_an_undeclared_route_value_exits_3(monkeypatch, capsys):
    with pytest.raises(SystemExit) as ei:
        _autoload.route_tp_unit({tp_route.ENV: "yes"})
    assert ei.value.code == 3 and "PROTENIX_OPT_TP_ROUTE='yes' is not a declared value" in capsys.readouterr().err


def test_install_rebinds_the_carried_names_to_the_core_in_a_rank_process():
    """A child interpreter with the rank env imports the carried modules: every routed name IS the core's object, the CARRIED name is
    the unit's own, and one TP-ROUTE line per carried module is printed (the events' evidence)."""
    _core_rowpair()
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("the carried unit's modules import torch")
    code = r"""
import importlib, os, sys
from protenix_opt import tp_route
tp_route.install()
out = {}
for carried, (core_module, names) in tp_route.ROUTES.items():
    cm = importlib.import_module(carried)
    for n in names:
        target = tp_route.ELSEWHERE.get((carried, n), core_module)
        core_obj = getattr(importlib.import_module(target), n)
        assert getattr(cm, n) is core_obj, f"{carried}.{n} is not {target}.{n}"
import ptx_tp.contract as c, opt_core.mem.rowpair.trimul as tm
assert c.rowsplit_attention is not getattr(tm, "rowsplit_attention", None) and c.rowsplit_attention.__module__ == "ptx_tp.contract"
assert os.environ["ROWPAIR_RANK"] == "0" and os.environ["ROWPAIR_STORE"]
print("ROUTED-OK", len(tp_route._patches))
"""
    env = dict(os.environ, RANK="0", WORLD_SIZE="2", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29777",
               PYTHONPATH=os.pathsep.join([UNIT, os.path.dirname(os.path.dirname(tp.__file__))] + [p for p in sys.path if p]))
    env.pop(tp_route.ENV, None); env.pop("PROTENIX_OPT", None)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ROUTED-OK" in r.stdout
    lines = tp_route.route_lines(r.stderr)
    assert sorted(set(lines)) == sorted(tp_route.ROUTES), (lines, r.stderr[-1500:])
    n_patches = int(r.stdout.split("ROUTED-OK")[1].split()[0])
    from protenix_opt.tp_bind import census as tp_census
    assert n_patches == len(tp_route.ROUTES) + 1 + len(tp_census.SITES) + 1   # per carried module + the launcher site + the census marks + the layout guard
