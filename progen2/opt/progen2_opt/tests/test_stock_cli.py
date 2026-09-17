"""The stock caller: the env proof passes clean and fails dirty; the child command shape; the block extraction from a stock stdout."""
import os
import subprocess
import sys

import pytest

from progen2_opt import outputs, stock_cli
from .conftest import subprocess_env

STOCK_SAMPLE_STDOUT = "loading parameters\nloading parameters took 3.21s\nloading tokenizer\nloading tokenizer took 0.02s\nsanity\nsanity took 0.5s\nsampling\n1\n\n0\nMKV\nsampling took 1.2s\ndone.\n"
STOCK_LL_STDOUT = "loading parameters\nloading parameters took 3.21s\nloading tokenizer\nloading tokenizer took 0.02s\nlog-likelihood (left-to-right, right-to-left)\nll_sum=-1.55\nll_mean=-0.105\nlog-likelihood (left-to-right, right-to-left) took 0.61s\ndone.\n"   # likelihood.py's stdout: print_time's desc / took lines around sections 3, 4 and 7, the pair, done.


def test_env_proof_clean_and_dirty():
    proof = stock_cli.env_proof(["PROGEN2_"], environ={"PATH": "/bin", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, modules={"os": os}, path=["/usr/lib"])   # the caller's own variables (a cuBLAS setting here) are nothing to the proof
    assert proof["forbidden_present"] == [] and proof["kit_modules"] == [] and proof["kit_dirs"] == []
    with pytest.raises(RuntimeError, match="forbidden variables present"):
        stock_cli.env_proof(["PROGEN2_"], environ={"PROGEN2_KIT": "v0_ew"}, modules={}, path=[])
    with pytest.raises(RuntimeError, match="kit modules loaded"):
        stock_cli.env_proof(["PROGEN2_"], environ={}, modules={"engines.progen2.kits.v0_ew": object()}, path=[])


def test_kit_dir_on_path_is_refused():
    from progen2_opt import stack
    with pytest.raises(RuntimeError, match="kit directories on sys.path"):
        stock_cli.env_proof(["PROGEN2_"], environ={}, modules={}, path=[stack.kit_dir("serving")])


def test_child_command_and_argv(monkeypatch):
    item = {"item_id": "a", "context": "1", "max_length": 256, "num_samples": 1, "t": 0.2, "p": 0.95, "rng_seed": 42}
    monkeypatch.setenv("PROGEN2_PYTHON", "/opt/venv/bin/python")                  # the one interpreter resolution (stack.python)
    cmd = stock_cli.child_command("sample", stock_cli.item_argv("sample", item, "progen2-small", ["--fp16", "true"]), ["PROGEN2_"])
    assert cmd[:4] == ["/opt/venv/bin/python", "-s", "-m", "progen2_opt.stock_cli"] and "--" in cmd and "--det" not in cmd
    tail = cmd[cmd.index("--") + 1:]
    assert tail == ["--model", "progen2-small", "--fp16", "true", "--context", "1", "--max-length", "256", "--num-samples", "1", "--t", "0.2", "--p", "0.95", "--rng-seed", "42"]   # --model, the job-level flags as given, then the item's own
    assert stock_cli.item_argv("score", {"item_id": "b", "context": "1ABC"}, "progen2-base", []) == ["--model", "progen2-base", "--context", "1ABC"]
    assert stock_cli.child_command("score", ["--model", "progen2-base", "--sanity", "TRUE"], ["PROGEN2_"])[-4:] == ["--model", "progen2-base", "--sanity", "TRUE"]   # a single call: the flags exactly as given


def test_block_and_load_extraction():
    assert outputs.block_of_stdout("sample", STOCK_SAMPLE_STDOUT) == "1\n\n0\nMKV\ndone.\n"      # the generation kit's block grammar (progen2_decode.py Handle.unit)
    assert outputs.block_of_stdout("score", STOCK_LL_STDOUT) == "ll_sum=-1.55\nll_mean=-0.105\n"
    assert stock_cli.load_wall_of_stdout(STOCK_SAMPLE_STDOUT) == 3.21
    assert outputs.block_of_stdout("sample", "no marker\n") == ""


def test_child_runs_script_as_main_after_proof(fake_stock):
    (fake_stock / "likelihood.py").write_text('import sys\nprint("ARGV", sys.argv)\n')
    r = subprocess.run([sys.executable, "-s", "-m", "progen2_opt.stock_cli", "--script", "likelihood.py", "--env-absent", "PROGEN2_",
                        "--env-allowed", "PROGEN2_PYTHON", "--", "--model", "progen2-small", "--context", "1ABC"], cwd=str(fake_stock), env=subprocess_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "[progen2-opt stock] ENV-CLEAN ok" in r.stderr and "autoload=" not in r.stderr and "ARGV ['likelihood.py', '--model', 'progen2-small', '--context', '1ABC']" in r.stdout
    r2 = subprocess.run([sys.executable, "-s", "-m", "progen2_opt.stock_cli", "--script", "likelihood.py", "--env-absent", "PROGEN2_", "--env-allowed", "PROGEN2_PYTHON", "--", "--model", "x"],
                        cwd=str(fake_stock), env=subprocess_env({"PROGEN2_KIT": "v0_ew"}), capture_output=True, text=True)
    assert r2.returncode != 0 and "forbidden variables present: ['PROGEN2_KIT']" in r2.stderr
    r3 = subprocess.run([sys.executable, "-s", "-m", "progen2_opt.stock_cli", "--script", "likelihood.py", "--det", "0", "--", "--model", "x"], cwd=str(fake_stock), env=subprocess_env(), capture_output=True, text=True)
    assert r3.returncode == 2 and "unrecognized arguments" in r3.stderr                            # the child has no --det
