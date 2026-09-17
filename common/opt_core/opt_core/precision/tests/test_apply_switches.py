"""opt_core.precision.recipe.apply_switches — the one torch applier: exactly the switches named, seeds named when skipped, the whole
recipe held (env / unset / pythonpath), the CUDA-initialisation order rule; apply_torch's level record over it. Stand-in torch, no GPU."""
from __future__ import annotations

import os
import random
import sys
import types

import pytest

from opt_core.det import Recipe
from opt_core.precision import recipe

RECIPE = Recipe(level=1, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


class _Matmul:
    """cuda.matmul stand-in: ``allow_tf32`` coupled to the process matmul precision word, as in torch."""

    def __init__(self, state):
        object.__setattr__(self, "_s", state)

    @property
    def allow_tf32(self):
        if self._s["refuse_old_api"]:
            raise RuntimeError("mixed allow_tf32 and fp32_precision APIs")
        return self._s["precision"] != "highest"

    @allow_tf32.setter
    def allow_tf32(self, v):
        self._s["precision"] = "high" if v else "highest"
        self._s["fp32_precision_mm"] = "tf32" if v else "ieee"

    @property
    def fp32_precision(self):
        return self._s["fp32_precision_mm"]


class _Cudnn:
    def __init__(self, state):
        object.__setattr__(self, "_s", state)
        object.__setattr__(self, "benchmark", False)
        object.__setattr__(self, "deterministic", False)

    @property
    def allow_tf32(self):
        if self._s["refuse_old_api"]:
            raise RuntimeError("mixed allow_tf32 and fp32_precision APIs")
        return self._s["cudnn_tf32"]

    @allow_tf32.setter
    def allow_tf32(self, v):
        self._s["cudnn_tf32"] = bool(v)

    @property
    def conv(self):
        return types.SimpleNamespace(fp32_precision="tf32" if self._s["cudnn_tf32"] else "ieee")


def fake_torch(*, precision="highest", cudnn_tf32=True, cuda_available=False, cuda_initialised=False, refuse_old_api=False):
    """A stand-in exposing exactly the surface policy/recipe read and set (no GPU, no real torch)."""
    s = {"precision": precision, "fp32_precision_mm": "ieee" if precision == "highest" else "tf32", "cudnn_tf32": cudnn_tf32,
         "refuse_old_api": refuse_old_api, "det": False, "warn_only": False, "seed": None, "seed_all": None}
    t = types.ModuleType("torch")
    t._state = s
    t.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=_Matmul(s)), cudnn=_Cudnn(s))

    def get_precision():
        if s["refuse_old_api"]:
            raise RuntimeError("mixed allow_tf32 and fp32_precision APIs")
        return s["precision"]

    def set_precision(p):
        s["precision"] = p
        s["fp32_precision_mm"] = "ieee" if p == "highest" else "tf32"
    t.get_float32_matmul_precision = get_precision
    t.set_float32_matmul_precision = set_precision
    t.are_deterministic_algorithms_enabled = lambda: s["det"]
    t.is_deterministic_algorithms_warn_only_enabled = lambda: s["warn_only"]

    def use_det(mode, warn_only=False):
        s["det"] = bool(mode)
        s["warn_only"] = bool(warn_only)
    t.use_deterministic_algorithms = use_det
    t.is_autocast_enabled = lambda device_type=None: False
    t.get_autocast_dtype = lambda device_type: "torch.float16"

    def manual_seed(v):
        s["seed"] = v
    t.manual_seed = manual_seed

    def seed_all(v):
        s["seed_all"] = v
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda_available, is_initialized=lambda: cuda_initialised, manual_seed_all=seed_all)
    return t


def test_applies_exactly_the_given_fields():
    t = fake_torch()
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = recipe.apply_switches(RECIPE, torch=t, tf32=False, cudnn_benchmark=False, environ=env)
    assert rec["applied"] == {"torch.backends.cuda.matmul.allow_tf32": False, "torch.backends.cudnn.allow_tf32": False,
                              "torch.backends.cudnn.benchmark": False}
    assert t.backends.cudnn.allow_tf32 is False and t.backends.cudnn.deterministic is False and t._state["det"] is False
    assert rec["env"] == env and rec["env_late"] == [] and rec["env_missing"] == []
    assert rec["readback"]["cudnn_tf32"] is False and rec["readback"]["cublas_workspace"] == ":4096:8" and rec["changed"] == ["cudnn_tf32"]


def test_full_recipe_and_named_seed_skips(monkeypatch):
    t = fake_torch()
    rec = recipe.apply_switches(RECIPE, torch=t, seed=0, deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                                tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert rec["applied"]["torch.use_deterministic_algorithms"] == {"mode": True, "warn_only": False}
    assert rec["applied"]["torch.manual_seed"] == 0 and rec["applied"]["random.seed"] == 0
    assert rec["applied"]["torch.cuda.manual_seed_all"] == recipe.SKIPPED_CUDA == "skipped:cuda-unavailable"
    assert t._state["det"] is True and rec["readback"]["deterministic_algorithms"] is True
    monkeypatch.setitem(sys.modules, "numpy", None)                            # numpy unimportable -> named, never silent
    rec = recipe.apply_switches(None, torch=t, seed=1, seed_scope=("numpy", "torch"), environ={})
    assert rec["applied"] == {"numpy.random.seed": "skipped:numpy-unavailable", "torch.manual_seed": 1,
                              "torch.cuda.manual_seed_all": "skipped:cuda-unavailable"}


def test_seed_scope_and_warn_only():
    t = fake_torch()
    random.seed(12345)
    expect = random.random()
    random.seed(12345)
    rec = recipe.apply_switches(None, torch=t, seed=0, seed_scope=("torch",), deterministic_algorithms=True, warn_only=True, environ={})
    assert random.random() == expect and t._state["warn_only"] is True and rec["env"] == {}
    with pytest.raises(recipe.RecipeLevelError):
        recipe.apply_switches(None, torch=t, seed=0, seed_scope=("tensorflow",), environ={})


def test_env_unset_pythonpath_are_enforced():
    env = {}
    rec = recipe.apply_switches(RECIPE, torch=fake_torch(), tf32=False, environ=env)      # before CUDA: exported late, named
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} and rec["env_late"] == ["CUBLAS_WORKSPACE_CONFIG"]
    env = {}
    rec = recipe.apply_switches(RECIPE, torch=fake_torch(), tf32=False, environ=env, export_env=False)
    assert env == {} and rec["env_missing"] == ["CUBLAS_WORKSPACE_CONFIG"] and rec["env_late"] == []
    hot = fake_torch(cuda_available=True, cuda_initialised=True)
    with pytest.raises(recipe.RecipeOrderError) as e:                                     # after CUDA: refused, naming it
        recipe.apply_switches(RECIPE, torch=hot, tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    assert "(':16:8', ':4096:8')" in str(e.value) and e.value.event == "det_recipe_order"
    rec = recipe.apply_switches(RECIPE, torch=hot, tf32=False, environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                                env_accept={"CUBLAS_WORKSPACE_CONFIG": (":16:8",)})
    assert rec["env_late"] == []
    r = Recipe(level=1, unset=("NVIDIA_TF32_OVERRIDE",))
    env = {"NVIDIA_TF32_OVERRIDE": "0"}
    rec = recipe.apply_switches(r, torch=fake_torch(), environ=env)
    assert env == {} and rec["env_late"] == ["NVIDIA_TF32_OVERRIDE"] and rec["env"] == {"NVIDIA_TF32_OVERRIDE": None}
    with pytest.raises(recipe.RecipeOrderError):
        recipe.apply_switches(r, torch=hot, environ={"NVIDIA_TF32_OVERRIDE": "0"})
    rp = Recipe(level=1, pythonpath=("/kit/det_site",))
    with pytest.raises(recipe.RecipeOrderError) as e:
        recipe.apply_switches(rp, torch=fake_torch(), environ={"PYTHONPATH": "/elsewhere"})
    assert "['/kit/det_site']" in str(e.value)
    assert recipe.apply_switches(rp, torch=fake_torch(), environ={"PYTHONPATH": "/x" + os.pathsep + "/kit/det_site"})["env"] == {}
    assert recipe.env_missing(Recipe(level=1, unset=("X",)), {"X": "1"}) == {"X": ("1", None)}
    assert recipe.pythonpath_missing(rp, {}) == ["/kit/det_site"]


def test_apply_torch_is_the_level_over_apply_switches():
    t = fake_torch()
    t.backends.cudnn.benchmark = True
    r0 = recipe.apply_torch(0, torch=t, environ={})
    assert r0["det"] == 0 and r0["changed"] == [] and t._state["det"] is False and r0["cublas_workspace"] == "unset"
    r1 = recipe.apply_torch(1, torch=t, environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert r1 == {"det": 1, "deterministic_algorithms": True, "warn_only": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
                  "cublas_workspace": ":4096:8", "changed": ["cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms"]}
    hot = fake_torch(cuda_available=True, cuda_initialised=True)
    with pytest.raises(recipe.RecipeOrderError):
        recipe.apply_torch(1, torch=hot, environ={})
    assert recipe.apply_torch(1, torch=hot, environ={}, check_order=False)["det"] == 1


def test_apply_torch_records_det_site_pythonpath_instead_of_raising():          # FIX-1: the det site is the stock child's
    rp = Recipe(level=1, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, pythonpath=("/kit/det_site",))
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": "/elsewhere"}
    for t in (fake_torch(), fake_torch(cuda_available=True, cuda_initialised=True)):
        r1 = recipe.apply_torch(rp, torch=t, environ=dict(env))
        assert r1["det"] == 1 and r1["deterministic_algorithms"] is True and t._state["det"] is True
    with pytest.raises(recipe.RecipeOrderError):                                 # the strict default (a stock interpreter) still refuses
        recipe.apply_switches(rp, torch=fake_torch(), environ=dict(env))
    rec = recipe.apply_switches(rp, torch=fake_torch(), environ=dict(env), hold_pythonpath=False)
    assert rec["pythonpath_missing"] == ["/kit/det_site"] and rec["env_late"] == [] and rec["env_missing"] == []
    assert recipe.apply_switches(RECIPE, torch=fake_torch(), environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})["pythonpath_missing"] == []


def test_apply_torch_order_check_is_the_cublas_variable_only():                  # FIX-2: kit switches are recorded, not refused
    rk = recipe.torch_recipe(1, switches={"KIT_DETERMINISTIC": "1"})
    hot = fake_torch(cuda_available=True, cuda_initialised=True)
    r1 = recipe.apply_torch(rk, torch=hot, environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert r1["det"] == 1 and r1["cublas_workspace"] == ":4096:8" and hot._state["det"] is True
    with pytest.raises(recipe.RecipeOrderError) as e:                            # the cuBLAS variable itself is still refused when late
        recipe.apply_torch(rk, torch=hot, environ={"KIT_DETERMINISTIC": "1"})
    assert "CUBLAS_WORKSPACE_CONFIG" in str(e.value) and "KIT_DETERMINISTIC" not in str(e.value)
    with pytest.raises(recipe.RecipeOrderError) as e:                            # strict default: every recipe name is order-sensitive
        recipe.apply_switches(rk, torch=hot, environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert "KIT_DETERMINISTIC" in str(e.value)
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = recipe.apply_switches(rk, torch=hot, environ=env, order_sensitive=("CUBLAS_WORKSPACE_CONFIG",), export_env=False)
    assert rec["env_missing"] == ["KIT_DETERMINISTIC"] and rec["env_late"] == [] and env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = recipe.apply_switches(rk, torch=hot, environ=env, order_sensitive=("CUBLAS_WORKSPACE_CONFIG",))
    assert rec["env_late"] == ["KIT_DETERMINISTIC"] and env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KIT_DETERMINISTIC": "1"}
