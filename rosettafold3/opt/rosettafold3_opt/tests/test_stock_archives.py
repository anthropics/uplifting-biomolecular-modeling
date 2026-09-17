"""stock/: the required src files and the archive are present; check_pins accepts exactly the pin (git commit route, or a live hash of
the tracked archive on the archive route — never a stored manifest)."""
import hashlib
import json
import os
import sys

import pytest

from .. import stack
from . import _stubs


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stock_dir():
    return os.path.join(stack.tree_root(), "stock")


def test_stock_src_carries_the_required_files():
    assert os.path.isfile(os.path.join(stock_dir(), "foundry-4010e3e2e.tar.gz"))
    for f in ("src/models/rf3/src/rf3/diffusion_samplers/inference_sampler.py", "src/models/rf3/src/rf3/model/RF3_structure.py",
              "src/models/rf3/src/rf3/model/layers/af3_diffusion_transformer.py", "src/models/rf3/src/rf3/loss/loss.py", "src/src/foundry/utils/torch.py"):
        assert os.path.isfile(os.path.join(stock_dir(), f)), f


def test_pins_archive_and_commit():
    p = stack.pins()
    assert p["upstream"]["commit"] == _stubs.PIN and p["upstream"]["tag"] is None
    assert os.path.getsize(os.path.join(stock_dir(), p["archive"]["filename"])) == p["archive"]["bytes"]
    assert p["weights"]["rf3"]["sha256"] == "364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0"
    assert p["weights"]["rf3"]["bytes"] == 3038876446
    assert p["package_prefix"] == "ROSETTAFOLD3_OPT"
    se = p["stock_environment"]                                    # the one list the stock caller reads (stock_fold.must_be_absent)
    assert "RF3_" in se["must_be_absent_prefixes"] and p["package_prefix"] in se["must_be_absent_prefixes"]
    assert "kit_prefixes" not in p and "kit_exact_names" not in p
    assert "files_rf3" not in p["tree_states"] and "files_det" not in p["tree_states"]   # tree.py RF3_FILES is the one copy
    assert "tested_on" in p["pinned_stack"] and "image" not in p["pinned_stack"] and p["pinned_stack"]["freeze"] == "environment/requirements.lock"
    lock = [l.strip() for l in open(os.path.join(stack.tree_root(), p["pinned_stack"]["freeze"])) if l.strip() and not l.startswith("#")]
    assert len(lock) == 148 and f"numpy=={p['pinned_stack']['numpy']}" in lock and f"triton=={p['pinned_stack']['triton']}" in lock   # the one list of the stack: the lock environment/Dockerfile installs (torch's line carries PyPI's version, without the +cu130 local tag the pin names)


def test_archive_lists_the_recipe_paths():
    import tarfile
    p = stack.pins()
    names = tarfile.open(os.path.join(stock_dir(), p["archive"]["filename"])).getnames()
    assert len(names) == p["archive"]["entries"]
    for path in p["archive"]["paths"]:
        assert any(n == f"foundry-4010e3e2e/{path}" or n.startswith(f"foundry-4010e3e2e/{path}/") for n in names), path
    assert not any("/tests/" in n or n.endswith(".png") for n in names)


@pytest.mark.parametrize("route,commit,ok", [("vcs", _stubs.PIN, True), ("vcs", "0" * 40, False), ("none", None, False)])
def test_check_pins_routes(tmp_path, monkeypatch, route, commit, ok):
    _stubs.make_dist(str(tmp_path), commit=commit or _stubs.PIN, route=route)
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.invalidate_caches()
    res = stack.pin_check()
    assert res["ok"] is ok, res
    assert res["route"] == route or (route == "none" and res["route"] == "none")


def test_check_pins_archive_route(tmp_path, monkeypatch):
    p = stack.pins()
    archive_sha = sha(os.path.join(stock_dir(), p["archive"]["filename"]))                # live-computed; no stored digest of the archive
    _stubs.make_dist(str(tmp_path), route="archive", archive_sha=archive_sha, version="0.0.0")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.invalidate_caches()
    res = stack.pin_check()
    assert res["ok"] and res["route"] == "archive" and res["version"] == "0.0.0"


def test_check_pins_on_second_interpreter(tmp_path):
    _stubs.make_dist(str(tmp_path / "sp"), route="vcs")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), str(tmp_path / "sp"))
    res = stack.pin_check(py)
    assert res["ok"] and res["route"] == "vcs" and res["commit"] == _stubs.PIN
