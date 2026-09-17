"""The stock arm: a clean environment proved and printed, the pristine tree proved by sha, upstream's own CLI, no kit dir on the path."""
import io
import os
import subprocess
import sys

import pytest

from .. import stack, stock_fold, tree
from . import _stubs


def test_clean_env_strips_every_kit_and_package_name():
    pins = stack.pins()
    env = {"PATH": "/bin", "HOME": "/h", "RF3_HOIST": "1", "RF3_CUDAGRAPH": "1", "RFD3_X": "1", "CUDA_MPS_PIPE_DIRECTORY": "/p", "FPF_RF3_TG_MAX": "8", "FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS": "1",
           "FOUNDRY_DET_SCATTER": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0", "ROSETTAFOLD3_OPT": "exact",
           "ROSETTAFOLD3_OPT_CKPT": "/c", "LOCAL_MSA_DIRS": "/m", "TRITON_CACHE_DIR": "/t"}
    out, stripped = stock_fold.clean_env(pins, env)
    assert stripped == sorted(["RF3_HOIST", "RF3_CUDAGRAPH", "RFD3_X", "CUDA_MPS_PIPE_DIRECTORY", "FPF_RF3_TG_MAX", "FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS",
                               "ROSETTAFOLD3_OPT", "ROSETTAFOLD3_OPT_CKPT"])
    assert set(out) == {"PATH", "HOME", "LOCAL_MSA_DIRS", "TRITON_CACHE_DIR", "PYTHONDONTWRITEBYTECODE", "FOUNDRY_DET_SCATTER", "CUBLAS_WORKSPACE_CONFIG",
                        "NVIDIA_TF32_OVERRIDE"}   # only the kits' names leave; the caller's other variables (a determinism setting included) stay; no PYTHONPATH of its own


def test_command_is_upstreams_cli():
    cmd = stock_fold.command("/v/bin/python", "/in.json", "/out", "/w.ckpt", ["early_stopping_plddt_threshold=0"], 42)
    assert cmd == ["/v/bin/python", "-m", "rf3.cli", "fold", "inputs=/in.json", "out_dir=/out", "ckpt_path=/w.ckpt", "early_stopping_plddt_threshold=0", "seed=42"]


def test_proof_lines_on_a_pristine_interpreter(tmp_path, monkeypatch):
    lists = stack.tree_digests()
    root = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), root)
    monkeypatch.setenv("RF3_HOIST", "1")                        # must be stripped and reported
    real = stock_fold.kit_dirs_on_path
    monkeypatch.setattr(stock_fold, "kit_dirs_on_path", lambda p, r, e: [x for x in real(p, r, e) if not x.startswith("rosettafold3_opt@")])   # this stub is this interpreter, which carries the package
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    env, ts, line = stock_fold.proof(py, stack.pins(), lists, stack.opt_root())
    assert line.startswith("[rosettafold3-opt stock] ENV-CLEAN ok: absent=RF3_,RFD3_,FPF_RF3_,FPF_TRIMUL_,CUDA_MPS_,ROSETTAFOLD3_OPT files=stock(5/5) kit_dirs=none")
    assert "stripped=RF3_HOIST" in line
    assert f"[rosettafold3-opt stock] TREE tree=stock(5/5) site-packages={ts.site_packages}" in buf.getvalue()
    assert "RF3_HOIST" not in env and ts.state == "stock"


def test_proof_refuses_a_patched_interpreter(tmp_path):
    lists = stack.tree_digests()
    root = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), root)
    with pytest.raises(RuntimeError, match="tree state is 'patched'"):
        stock_fold.proof(py, stack.pins(), lists, stack.opt_root())


def test_path_probe_ignores_start_up_output_of_a_site_hook(tmp_path, monkeypatch):
    """A site attached from outside (a ``sitecustomize.py`` first on the caller's PYTHONPATH) may print a banner at every interpreter start;
    the probe's verdict is its own sentinel line, so the banner is not reported as a kit directory (stock_fold.kit_dirs_on_path)."""
    lists = stack.tree_digests()
    root = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    site = tmp_path / "a_site"; site.mkdir()
    (site / "sitecustomize.py").write_text("print('[a_site] hello at start-up', flush=True)\n")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), root + os.pathsep + str(site))
    monkeypatch.setenv("PYTHONPATH", str(site))
    env, _ = stock_fold.clean_env(stack.pins())
    kd = [x for x in stock_fold.kit_dirs_on_path(py, stack.opt_root(), env) if not x.startswith("rosettafold3_opt@")]   # this stub is this interpreter, which carries the package
    assert kd == []
    r = subprocess.run([py, "-c", "print('hello from start-up')"], capture_output=True, text=True, env=env)
    assert "[a_site]" in r.stdout                                    # the banner is printed under the attached site


def test_path_probe_needs_its_verdict_line(tmp_path):
    """An interpreter whose stdout carries no sentinel line (here: a stub that only prints start-up text) gives no verdict — refused, never `none`."""
    mute = tmp_path / "bin" / "python"; mute.parent.mkdir()
    mute.write_text("#!/bin/sh\necho '[some_site] start-up text'\nexit 0\n"); mute.chmod(0o755)
    with pytest.raises(RuntimeError, match="gave no path-probe verdict"):
        stock_fold.kit_dirs_on_path(str(mute), stack.opt_root(), {"PATH": os.environ.get("PATH", "/bin")})


def test_proof_refuses_a_kit_dir_on_the_path(tmp_path, monkeypatch):
    lists = stack.tree_digests()
    root = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), root + os.pathsep + stack.opt_root())
    with pytest.raises(RuntimeError, match="sees this tree's kit directories or package"):
        stock_fold.proof(py, stack.pins(), lists, stack.opt_root())
