"""opt_core.seq.det_torch (a façade over opt_core.precision.recipe) against the stand-in torch of test_seq_numerics: exactly the given
fields, the seeds in scope (named when skipped), the whole recipe held, the record keys the kits' manifests read."""
import importlib.util
import os
import random
import sys

import pytest

from opt_core.det import Recipe
from opt_core.precision import recipe as root_recipe
from opt_core.seq import det_torch, numerics

from .test_seq_numerics import fake_torch

RECIPE = Recipe(level=1, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


@pytest.fixture
def torch_mod():
    saved = sys.modules.get("torch")
    t = fake_torch(cudnn_tf32=True)
    sys.modules["torch"] = t
    yield t
    if saved is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = saved


def test_facade_wiring():
    assert det_torch.DetRecipeError is root_recipe.RecipeOrderError                 # one refusal class
    expected = "opt_core.precision.recipe" if hasattr(root_recipe, "apply_switches") else "opt_core.seq.det_torch"
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch()
    try:
        assert det_torch.apply(None, environ={})["applier"] == expected
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_applies_exactly_the_given_fields(torch_mod):
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = det_torch.apply(RECIPE, tf32=False, cudnn_benchmark=False, environ=env)
    assert tuple(rec) == det_torch.RECORD_KEYS
    assert rec["applied"] == {"torch.backends.cuda.matmul.allow_tf32": False, "torch.backends.cudnn.allow_tf32": False,
                              "torch.backends.cudnn.benchmark": False}
    assert torch_mod.backends.cudnn.allow_tf32 is False and torch_mod.backends.cudnn.deterministic is False     # deterministic untouched
    assert torch_mod._state["det"] is False and torch_mod._state["seed"] is None                                 # no seed, no det algos
    assert rec["env"] == env and rec["env_late"] == [] and rec["readback"] == numerics.readback(env)
    assert numerics.classify(rec["readback"]) == "fp32-conv"


def test_full_recipe_reads_back_det(torch_mod):
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = det_torch.apply(RECIPE, seed=0, deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False, tf32=False,
                          environ=env)
    assert rec["applied"]["torch.use_deterministic_algorithms"] == {"mode": True, "warn_only": False}
    np_seed = 0 if importlib.util.find_spec("numpy") is not None else det_torch.SKIPPED_NUMPY       # numpy absent: the NAMED skip, not 0
    assert rec["applied"]["torch.manual_seed"] == 0 and rec["applied"]["random.seed"] == 0 and rec["applied"]["numpy.random.seed"] == np_seed
    assert rec["applied"]["torch.cuda.manual_seed_all"] == det_torch.SKIPPED_CUDA == "skipped:cuda-unavailable"    # named, not dropped
    assert torch_mod._state["det"] is True and torch_mod._state["seed"] == 0 and torch_mod._state["seed_all"] is None
    assert numerics.classify(rec["readback"]) == "det" and rec["readback"]["cublas_workspace_config"] == ":4096:8"


def test_warn_only_and_seed_scope(torch_mod):
    random.seed(12345)
    expect = random.random()
    random.seed(12345)
    rec = det_torch.apply(None, seed=0, seed_scope=("torch",), deterministic_algorithms=True, warn_only=True, environ={})
    assert rec["applied"] == {"torch.manual_seed": 0, "torch.cuda.manual_seed_all": det_torch.SKIPPED_CUDA,
                              "torch.use_deterministic_algorithms": {"mode": True, "warn_only": True}}
    assert random.random() == expect                                                                             # python's random untouched
    assert torch_mod._state["warn_only"] is True and rec["env"] == {}
    with pytest.raises(ValueError):
        det_torch.apply(None, seed=0, seed_scope=("tensorflow",), environ={})


def test_numpy_unavailable_is_named(torch_mod, monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)                             # `import numpy` raises ImportError now
    rec = det_torch.apply(None, seed=0, seed_scope=("numpy", "torch"), environ={})
    assert rec["applied"] == {"numpy.random.seed": "skipped:numpy-unavailable", "torch.manual_seed": 0,
                              "torch.cuda.manual_seed_all": "skipped:cuda-unavailable"}


def test_env_exported_late_before_cuda_and_refused_after(torch_mod):
    env = {}
    rec = det_torch.apply(RECIPE, tf32=False, environ=env)
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} and rec["env_late"] == ["CUBLAS_WORKSPACE_CONFIG"]
    assert rec["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    sys.modules["torch"] = fake_torch(cuda_available=True, cuda_initialised=True)
    with pytest.raises(det_torch.DetRecipeError) as e:
        det_torch.apply(RECIPE, tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    assert "CUBLAS_WORKSPACE_CONFIG" in str(e.value) and "(':16:8', ':4096:8')" in str(e.value)
    rec = det_torch.apply(RECIPE, tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                          env_accept={"CUBLAS_WORKSPACE_CONFIG": (":4096:8", ":16:8")})
    assert rec["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":16:8"} and rec["env_late"] == []


def test_unset_names_are_enforced(torch_mod):
    r = Recipe(level=1, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, unset=("NVIDIA_TF32_OVERRIDE",))
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"}
    rec = det_torch.apply(r, tf32=False, environ=env)                          # before CUDA: removed now, named
    assert "NVIDIA_TF32_OVERRIDE" not in env and rec["env_late"] == ["NVIDIA_TF32_OVERRIDE"]
    assert rec["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": None}
    sys.modules["torch"] = fake_torch(cuda_available=True, cuda_initialised=True)
    with pytest.raises(det_torch.DetRecipeError) as e:                         # after CUDA: refused, naming it
        det_torch.apply(r, tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"})
    assert "'NVIDIA_TF32_OVERRIDE': ('0', None)" in str(e.value)


def test_pythonpath_entries_are_enforced(torch_mod):
    r = Recipe(level=1, pythonpath=("/kit/det_hooks",))
    with pytest.raises(det_torch.DetRecipeError) as e:
        det_torch.apply(r, tf32=False, environ={"PYTHONPATH": "/elsewhere"})
    assert "['/kit/det_hooks']" in str(e.value) and "interpreter-start" in str(e.value)
    rec = det_torch.apply(r, tf32=False, environ={"PYTHONPATH": "/elsewhere" + os.pathsep + "/kit/det_hooks"})
    assert rec["env_late"] == [] and rec["env"] == {}
    assert det_torch.pythonpath_missing(r, {}) == ["/kit/det_hooks"]


def test_env_missing_shape():
    assert det_torch.env_missing(RECIPE, {}) == {"CUBLAS_WORKSPACE_CONFIG": (None, ":4096:8")}
    assert det_torch.env_missing(RECIPE, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}) == {}
    assert det_torch.env_missing(RECIPE, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, {"CUBLAS_WORKSPACE_CONFIG": (":16:8",)}) == {}
    assert det_torch.env_missing(Recipe(level=1, unset=("X",)), {"X": "1"}) == {"X": ("1", None)}
    assert not hasattr(det_torch, "readback")                                    # one import path: opt_core.seq.numerics.readback
