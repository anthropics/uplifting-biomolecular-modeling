"""The registry (variants and checkpoints from the kit's pins module by file path; the stock/PINS.json cross-check) and the
deterministic recipe (values from the pins module; the torch switches applied to a stub torch)."""
import json
import os
import sys

import pytest

from e1_opt import det, registry, stack
from e1_opt.tests import _stubs


def test_variants_and_checkpoints_come_from_the_pins_module():
    pins = _stubs.pins_or_skip()
    vs = registry.variants(pins)
    assert vs == list(pins.WEIGHTS) and len(vs) == 3
    before = {m for m in sys.modules if m.startswith(("engines", "compare"))}
    registry._LOADED.clear()
    pins2 = registry.load_pins(stack.forward_root())
    assert pins2.WEIGHTS == pins.WEIGHTS and {m for m in sys.modules if m.startswith(("engines", "compare"))} == before   # no kit package bound by the load
    for v in vs:
        c = registry.checkpoint(v, pins)
        assert c["repo"] == pins.WEIGHTS[v]["repo"] and c["rev"] == pins.WEIGHTS[v]["rev"] and len(c["sha256"]) == 64
        assert registry.snapshot_dir(v, pins, "/hf").startswith("/hf/hub/models--") and registry.snapshot_dir(v, pins, "/hf").endswith(c["rev"])
    assert registry.check_variant(" 300M ", pins) == "300m" and registry.check_variant(None, pins) is None
    with pytest.raises(ValueError):
        registry.check_variant("900m", pins)


def test_pins_json_cross_check(tmp_path):
    pins = _stubs.pins_or_skip()
    v = "300m"
    ok = {"upstream": {"E1": {"commit": pins.STOCK["commit"]}}, "weights": {v: {"repo": pins.WEIGHTS[v]["repo"], "rev": pins.WEIGHTS[v]["rev"]}}}
    p = tmp_path / "PINS.json"
    p.write_text(json.dumps(ok))
    assert registry.cross_check(pins, str(p)) == []
    assert registry.cross_check(pins, str(tmp_path / "absent.json")) == []
    bad = dict(ok, upstream={"E1": {"commit": "0" * 40}})
    p.write_text(json.dumps(bad))
    assert any("stock commit" in b for b in registry.cross_check(pins, str(p)))


def test_det_recipe_values_and_env_from_pins():
    pins = _stubs.pins_or_skip()
    r = det.recipe(pins)
    assert r["seed"] == pins.DET_RECIPE["seed"] and r["env"] == pins.DET_RECIPE["env"] and "CUBLAS_WORKSPACE_CONFIG" in r["env"]
    assert det.env_for(pins, 0) == {} and det.env_for(pins, 1) == {k: str(v) for k, v in pins.DET_RECIPE["env"].items()}
    assert det.check_level("1") == 1 and det.check_level(0) == 0
    with pytest.raises(ValueError):
        det.check_level(2)
    e = {}
    det.apply_env(pins, 1, e)
    assert e == det.env_for(pins, 1)
    assert det.describe(pins, 0)["level"] == 0 and "recipe" in det.describe(pins, 1)


def test_det_apply_torch_on_a_stub(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    pins = box["pins"]
    import torch
    assert torch.__version__ == pins.STACK["torch_version_str"], "the stub torch must shadow any installed torch in this process"
    rec = det.apply_torch(pins, 1)
    a = rec["applied"]
    assert a["torch.manual_seed"] == pins.DET_RECIPE["seed"] and a["torch.use_deterministic_algorithms"] == bool(pins.DET_RECIPE["deterministic_algorithms"])
    assert torch.are_deterministic_algorithms_enabled() == bool(pins.DET_RECIPE["deterministic_algorithms"])
    assert torch.backends.cuda.matmul.allow_tf32 == bool(pins.DET_RECIPE["tf32"]) and torch.backends.cudnn.benchmark == bool(pins.DET_RECIPE["cudnn_benchmark"])
    st = det.torch_state()
    assert st["deterministic_algorithms"] == bool(pins.DET_RECIPE["deterministic_algorithms"])
    assert det.apply_torch(pins, 0) == {"level": 0, "applied": {}}
