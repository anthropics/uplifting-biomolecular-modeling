"""The big adapter: apply() = the line's activation in-process over modes.resolve + hooks (no second table), report() = the
port's census beside the levers record; MODES carries a DETERMINISM row per mode."""
import json
import os
import subprocess
import sys

import pytest

from openfold3_opt import ActivationError, big, hooks, modes
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_every_mode_has_a_determinism_row():
    assert set(modes.DETERMINISM) == set(modes.MODES)
    for m, row in modes.DETERMINISM.items():
        assert set(row) == {"recipe", "graphs", "seed", "equality"}, m


def test_resolution_is_the_one_resolver():
    res = big.resolution(environ={}, home=HOME)
    ln = modes.LINES[("big", "resident")]
    assert (res.mode, res.line, res.hooks) == ("big", "resident", list(ln.hooks))
    assert {k: v for k, v in res.exports.items() if k in ln.env} == dict(ln.env)
    assert big.resolution(environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2"}, home=HOME, n_gpu="2").line == "tp"


def test_apply_refuses_a_contradicting_preset(monkeypatch):
    monkeypatch.setenv("OF3O_LAYER", "9")
    with pytest.raises(ActivationError, match="OF3O_LAYER"):
        big.apply(home=HOME)


def test_apply_installs_the_chain_in_process():
    """In a fresh interpreter (the hook chain mutates the environment, sys.path and sys.meta_path): the entry hook runs and chains — with no
    token count the resident line carries its confidence levers, so the package's confhead hook is the prelude (first directory, executed) and
    chains into the offload port's hook (`attach` = the entry port); the port's finder, the confhead finder and the add-ons' finders sit on
    sys.meta_path, the exports are in the environment, the graph switches are not."""
    # -I (isolated mode) ignores PYTHONPATH by design, so the child finds this package and the core via an explicit sys.path
    # insert instead of the environment -- the one thing -I cannot make disappear.
    code = ("import json, os, sys\n"
            f"sys.path[:0] = [{os.path.join(HOME, 'opt')!r}, {_stubs.core_dir()!r}]\n"
            "for k in [k for k in os.environ if k.startswith(('OF3', 'OPENFOLD3_OPT'))]: del os.environ[k]\n"
            "from openfold3_opt import big, hooks\n"
            f"out = big.apply(home={HOME!r})\n"
            "print(json.dumps({'attach': out['attach'], 'line': out['line'], 'finders': out['finders'], 'layer': os.environ.get('OF3O_LAYER'), 'graphs': os.environ.get('OF3_CUDA_GRAPHS'), 'known': [c for c in out['finders'] if c in hooks.FINDER_CLASSES], "
            "'hook_dirs': out['hook_dirs'], 'entry': out['entry_hook'], 'confhead': os.environ.get('OPENFOLD3_OPT_CONFHEAD'), 'chain': os.environ.get('OPENFOLD3_OPT_CONFHEAD_CHAIN'), 'conf_mode': os.environ.get('OF3O_CONF_MODE')}))\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["attach"] == "offload" and out["line"] == "resident" and out["layer"] == "1" and out["graphs"] is None
    assert out["known"], out["finders"]
    cells_dir, conf_dir = modes.hook_dir(HOME, "cells"), os.path.join(HOME, modes.PACKAGE_HOOKS["confhead"])   # the prelude: the row-block head's hook first on the path, its file executed; its chain variable = the port's hook directory; the port chains the cells' hook
    assert out["hook_dirs"][0] == conf_dir and out["hook_dirs"][2] == cells_dir and out["entry"] == os.path.join(conf_dir, modes.HOOK_FILE)
    assert out["chain"] == out["hook_dirs"][1] == modes.hook_dir(HOME, "offload") and out["confhead"] == "1" and out["conf_mode"] == "chunked"
    assert out["finders"].count("_PostImportFinder") >= 2, out["finders"]                # the confhead hook's finder and the offload port's


def test_report_shape():
    r = big.report()
    assert set(r) == {"census", "tp", "levers"}                      # the offload census, the tp line's process-group state (None outside a rank), the levers


def test_a_chain_without_a_port_is_refused_by_name(monkeypatch):
    """Every big line carries a PORT directory; a resolution whose chain has none is a broken table and `apply()` refuses it by name
    (ActivationError 'no entry port in hook chain: …') — never a quiet attach to the first hook."""
    import pytest
    from openfold3_opt import ActivationError, big, modes
    res = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OPT_N_GPU": "2"})
    import dataclasses
    broken = res._replace(hooks=("fast_inference",), hook_dirs=res.hook_dirs[:1]) if hasattr(res, "_replace") else dataclasses.replace(res, hooks=("fast_inference",), hook_dirs=res.hook_dirs[:1])
    monkeypatch.setattr(big, "resolution", lambda *a, **k: broken)
    with pytest.raises(ActivationError, match="no entry port in hook chain: fast_inference"):
        big.apply(environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OPT_N_GPU": "2"}, home=HOME)
