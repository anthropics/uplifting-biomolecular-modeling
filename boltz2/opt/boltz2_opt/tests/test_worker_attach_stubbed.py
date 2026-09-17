"""boltz2_opt.worker_launch --attach: the memory lines' attach hook driven with stub modules — no boltz, no torch, no GPU.

The hook applies the engine adapter (boltz2_opt.big) right after the trigger module's own body has executed (before the script's next
statement), merges the adapter's report into the worker's own log after the worker's atexit dump, and refuses (exit 3, a `[boltz2-opt attach]
REFUSED` line) when the adapter has no levers installed, applies nothing, cannot import, resolves outside the kit package, or the attachment
name is not one of this tree's. The launcher runs from a COPY of the kit package in tmp_path (its HOME), so a stub adapter can sit under it.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # boltz2_opt/

STUB_BOLTZ2 = "import sys\nsys.modules['__main__'].ORDER.append('boltz2 module body')\nclass Boltz2: pass\n"
STUB_ADAPTER = """import os, sys
LEVERS = ('xl_trans', 'xl_cond', 'xl_free')
STATE = {"applied": None}
def apply(spec=None):
    import boltz.model.models.boltz2 as b
    sys.modules['__main__'].ORDER.append('xl apply (Boltz2 bound: %s)' % hasattr(b, 'Boltz2'))
    STATE['applied'] = ['trans', 'cond', 'free'] if os.environ.get('BOLTZ_XL') == '1' else []
    return list(STATE['applied'])
def report():
    return {"applied": STATE['applied'], "disabled": {}, "stats": {"trans_rowchunked_calls": 3, "cond_rowchunked_calls": 1, "free_events": 1},
            "env": {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}}
"""
SCRIPT = """import atexit, json, os, sys
ORDER = ['script start']
from boltz.model.models.boltz2 import Boltz2
ORDER.append('after import')
b = json.load(open(sys.argv[sys.argv.index('--batch') + 1]))
p = os.path.join(b['kit_dir'], b['tag'] + '_worker_log.json')
def dump():
    json.dump({'events': ORDER, 'per_item': [{'name': 'x'}], 'dumped_by': 'worker'}, open(p, 'w'))
atexit.register(dump)                       # the worker's own dump (bz_worker_lev.py:64): registered AFTER the hook's, so it runs FIRST
json.dump({'order': ORDER}, open(os.path.join(b['out_dir'], 'order.json'), 'w'))
"""


@pytest.fixture
def stubs(tmp_path):
    for p in ("boltz/__init__.py", "boltz/model/__init__.py", "boltz/model/models/__init__.py"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True); (tmp_path / p).write_text("")
    (tmp_path / "boltz/model/models/boltz2.py").write_text(STUB_BOLTZ2)
    shutil.copytree(PKG, tmp_path / "boltz2_opt", ignore=shutil.ignore_patterns("__pycache__", "tests"))   # the kit package copied: the launcher's HOME
    (tmp_path / "boltz2_opt/big.py").write_text(STUB_ADAPTER)                                            # the adapter stubbed under it
    (tmp_path / "script.py").write_text(SCRIPT)
    (tmp_path / "b.json").write_text(json.dumps({"tag": "pred", "out_dir": str(tmp_path), "kit_dir": str(tmp_path)}))
    return tmp_path


def _run(cwd, args, **env):
    e = dict(os.environ); e.update(env)
    e.update({"PYTHONPATH": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"})
    return subprocess.run([sys.executable, "-m", "boltz2_opt.worker_launch"] + args, cwd=str(cwd), env=e, capture_output=True, text=True, timeout=120)


def test_attach_order_and_report_merge(stubs):
    r = _run(stubs, ["--attach", "xl", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="1", BOLTZ_XL_LEVERS="trans,cond,free", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert json.load(open(stubs / "order.json"))["order"] == ["script start", "boltz2 module body", "xl apply (Boltz2 bound: True)", "after import"]
    assert "[boltz2-opt attach] xl: boltz2_opt.big = " in r.stderr and "under the kit package" in r.stderr
    assert "[boltz2-opt attach] xl: boltz2_opt.big applied ['trans', 'cond', 'free'] after boltz.model.models.boltz2" in r.stderr
    log = json.load(open(stubs / "pred_worker_log.json"))
    assert log["dumped_by"] == "worker" and log["per_item"] == [{"name": "x"}], "the worker's own dump ran first and was kept"
    assert log["xl_report"] == {"applied": ["trans", "cond", "free"], "disabled": {}, "stats": {"trans_rowchunked_calls": 3, "cond_rowchunked_calls": 1, "free_events": 1},
                                "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}, "the adapter's report merged after the worker's dump"


def test_attach_refuses_when_nothing_applies(stubs):
    r = _run(stubs, ["--attach", "xl", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="")
    assert r.returncode == 3 and "[boltz2-opt attach] REFUSED xl: boltz2_opt.big.apply() applied nothing" in r.stderr, (r.returncode, r.stderr)
    assert not (stubs / "order.json").exists(), "the script never ran past the import"


def test_attach_refuses_an_unknown_name(stubs):
    r = _run(stubs, ["--attach", "nope", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="1")
    from ..worker_launch import ATTACH
    assert r.returncode == 3 and f"[boltz2-opt attach] REFUSED nope: not an attachment of this tree ({sorted(ATTACH)})" in r.stderr, (r.returncode, r.stderr)


def test_attach_refuses_an_import_failure(stubs):
    (stubs / "boltz2_opt/big.py").write_text("raise ImportError('no torch here')\n")
    r = _run(stubs, ["--attach", "xl", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="1")
    assert r.returncode == 3 and "[boltz2-opt attach] REFUSED xl: ImportError: no torch here" in r.stderr, (r.returncode, r.stderr)


def test_attach_refuses_the_shipped_adapters_outside_their_row_or_engine(stubs):
    """The adapters as shipped refuse by name: without their row's switch they apply nothing; with it, on a box whose boltz lacks the modules
    they patch (here: stubs), the import error is the refusal — never a silent stock run."""
    shutil.copyfile(os.path.join(PKG, "big.py"), stubs / "boltz2_opt/big.py")
    e = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_")}
    r = subprocess.run([sys.executable, "-m", "boltz2_opt.worker_launch", "--attach", "xl", "--", "script.py", "--batch", "b.json"], cwd=str(stubs), env={**e, "PYTHONPATH": str(stubs), "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "[boltz2-opt attach] REFUSED xl: boltz2_opt.big.apply() applied nothing" in r.stderr, (r.returncode, r.stderr)
    r = _run(stubs, ["--attach", "xl", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="1", BOLTZ_XL_LEVERS="trans,cond,free")
    assert r.returncode == 3 and "[boltz2-opt attach] REFUSED xl: " in r.stderr and ("composes on fast" in r.stderr or "ModuleNotFoundError" in r.stderr), (r.returncode, r.stderr)   # a row without fast's flash switch: refused by name (or, without the core, the core's absence first)
    r = _run(stubs, ["--attach", "xl", "--", "script.py", "--batch", "b.json"], BOLTZ_XL="1", BOLTZ_XL_LEVERS="trans,cond,free", BOLTZ_TRIATTN="flash")
    assert r.returncode == 3 and "[boltz2-opt attach] REFUSED xl: ModuleNotFoundError" in r.stderr, (r.returncode, r.stderr)
    assert not (stubs / "order.json").exists()
    r = _run(stubs, ["--attach", "trimul", "--", "script.py", "--batch", "b.json"], BOLTZ_FPF_TRIMUL="1")
    assert r.returncode == 0 and "[boltz2-opt attach] trimul: boltz2_opt.trimul = " in r.stderr and "REFUSED" not in r.stderr, "the TriMul hook waits for boltz.model.layers.triangular_mult; a script that never imports it runs unpatched and writes no trimul_report (stack.evidence then refuses the run)"


def test_attach_origin_refuses_a_module_outside_the_kit_package(tmp_path, monkeypatch):
    """The origin rule: the adapter must resolve to a file under the launcher's own package directory (HOME); any other origin is refused."""
    from boltz2_opt import worker_launch as wlm
    elsewhere = tmp_path / "big.py"; elsewhere.write_text("LEVERS = ()\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: importlib.util.spec_from_file_location(name, str(elsewhere)))
    with pytest.raises(RuntimeError, match="outside the kit package"):
        wlm.attach_origin("xl")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    with pytest.raises(ModuleNotFoundError, match="in the kit package"):
        wlm.attach_origin("xl")


def test_attach_origin_of_the_shipped_adapter():
    from boltz2_opt import worker_launch as wlm
    origin, sha = wlm.attach_origin("xl")
    assert origin == os.path.realpath(os.path.join(PKG, "big.py")) and len(sha) == 64


def test_usage_without_route_or_attach():
    r = subprocess.run([sys.executable, "-m", "boltz2_opt.worker_launch", "--", "x.py"], env=dict(os.environ, PYTHONPATH=os.path.dirname(PKG)), capture_output=True, text=True)
    assert r.returncode == 2 and "usage:" in r.stderr
