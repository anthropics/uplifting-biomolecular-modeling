"""CPU, no torch needed: the import surface is light; recipes render into the core's shape with the words the kits qualify under."""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from opt_core import det
from opt_core.precision import PrecisionError, recipe, xla

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_importing_the_package_imports_no_framework():
    code = ("import sys; import opt_core.precision as p; import opt_core.precision.policy, opt_core.precision.recipe, opt_core.precision.xla, "
            "opt_core.precision.det_scatter, opt_core.precision.cast_cache; "
            "bad = [m for m in ('torch', 'triton', 'jax', 'numpy') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    import opt_core
    root = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))          # the source tree or site-packages holding opt_core
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root] + [e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_sources_parse_as_python38():
    for dirpath, _, files in os.walk(PKG_DIR):
        for fn in files:
            if fn.endswith(".py"):
                src = open(os.path.join(dirpath, fn), encoding="utf-8").read()
                ast.parse(src, filename=fn, feature_version=(3, 8))


def test_torch_recipe_level0_is_production():
    assert recipe.torch_recipe(0) is det.PRODUCTION
    assert det.stock_exception(recipe.torch_recipe(0)) is None


def test_torch_recipe_level1_words():
    r = recipe.torch_recipe(1, switches={"KIT_DETERMINISTIC": "1"}, det_site="/opt/det_site")
    assert r.level == 1
    assert r.env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KIT_DETERMINISTIC": "1"}
    assert r.pythonpath == ("/opt/det_site",)
    assert det.describe(r).startswith("det=1 env=CUBLAS_WORKSPACE_CONFIG,KIT_DETERMINISTIC unset=none pythonpath=1")
    assert det.stock_exception(r) == {"env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KIT_DETERMINISTIC": "1"}, "pythonpath": ["/opt/det_site"]}


def test_torch_recipe_without_cublas_and_with_extra_env():
    r = recipe.torch_recipe(2, switches=None, cublas=False, extra_env={"A": 1}, note="n")
    assert r.env == {"A": "1"} and r.level == 2 and r.note == "n"


def test_apply_env_round_trip():
    env = {"PYTHONPATH": "/x", "KIT_DETERMINISTIC": "0"}
    r = recipe.torch_recipe(1, switches={"KIT_DETERMINISTIC": "1"}, det_site="/site")
    before = recipe.apply_env(r, env)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["KIT_DETERMINISTIC"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep) == ["/site", "/x"]
    assert before == {"CUBLAS_WORKSPACE_CONFIG": None, "KIT_DETERMINISTIC": "0", "PYTHONPATH": "/x"}


def test_xla_recipe_prepends_and_dedups():
    r = xla.xla_recipe(1, flags=("--xla_gpu_autotune_level=0",), environ={"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"})
    assert r.env == {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"}
    r2 = xla.xla_recipe(1, flags=("--a=1", "--b=2"), environ={"XLA_FLAGS": "--b=2 --c=3"}, extra_env={"JAX_CACHE": "/c"})
    assert r2.env == {"XLA_FLAGS": "--a=1 --b=2 --c=3", "JAX_CACHE": "/c"}
    assert xla.xla_recipe(0) is det.PRODUCTION
    assert xla.xla_recipe(1, environ={}).env == {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}


def test_level_check_is_named():
    assert recipe.level("1") == 1
    with pytest.raises(recipe.RecipeLevelError) as ei:
        recipe.level(2, levels=(0, 1))
    assert isinstance(ei.value, PrecisionError) and ei.value.event == "det_level_refused" and ei.value.record()["levels"] == (0, 1)
    with pytest.raises(recipe.RecipeLevelError):
        recipe.level("warn")


def test_require_torch_refusal_is_named(monkeypatch):
    import builtins

    from opt_core.precision import FrameworkMissing, require_torch
    if "torch" in sys.modules:
        pytest.skip("torch already imported in this interpreter")
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch here")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(FrameworkMissing) as ei:
        require_torch()
    assert ei.value.record()["event"] == "framework_missing" and ei.value.record()["framework"] == "torch"

    class FakeTorch:
        pass
    assert require_torch(FakeTorch) is FakeTorch
