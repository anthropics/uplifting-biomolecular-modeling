"""The stock route: the environment strip, the proof, the stock caller's refusal (NOT STOCK, exit 3) when a forbidden name or a kit
directory reaches it or kit code loads during the call, and the call itself — `run_openfold predict` through the console script's entry
point on a stub `openfold3` distribution (the interpreter here has none): the argument list the stock CLI receives, the proof after the
call, the exit code."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap

from openfold3_ob0_opt import env, modes, stock_pred
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_strip_removes_every_forbidden_name_and_kit_dir():
    pre = env.must_be_absent(HOME)
    base = {"PATH": "/usr/bin", "OF3_FAST_INIT": "1", "OF3T_PAIRCACHE": "1", "OF3O_LAYER": "1", "OPENFOLD3_OB0_OPT": "fast", "OPENFOLD3_OB0_OPT_EXACT_LINE": "composed",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_MPS_PIPE_DIRECTORY": "/tmp/x", "CUTLASS_PATH": "x", "OPENFOLD3_OB0_CKPT": "/w.pt",
            "PYTHONPATH": modes.hook_dir(HOME, "fast_inference") + os.pathsep + "/other/dir"}
    out = env.strip(pre, base)
    assert env.forbidden_present(pre, out) == []
    assert out["PYTHONPATH"] == "/other/dir" and out["OPENFOLD3_OB0_CKPT"] == "/w.pt" and out["PATH"] == "/usr/bin"


def test_env_proof_flags_each_violation():
    pre = env.must_be_absent(HOME)
    p = env.env_proof(pre, environ={"OF3_CUDA_GRAPHS": "1"})
    assert p["forbidden_present"] == ["OF3_CUDA_GRAPHS"] and not p["ok"]
    d = modes.hook_dir(HOME, "trunk_kernels")
    sys.path.insert(0, d)
    try:
        p = env.env_proof(pre, environ={})
        assert d in p["kit_dirs_on_path"] and not p["ok"]
    finally:
        sys.path.remove(d)


def _run_stock(extra_env, stock_args=("--query-json", "q.json", "--output-dir", "o")):
    pre = env.must_be_absent(HOME)
    with tempfile.TemporaryDirectory() as td:
        proof = os.path.join(td, "proof.json")
        e = env.strip(pre, dict(os.environ))
        e["PYTHONPATH"] = _stubs.subprocess_pythonpath(HOME)             # the package itself plus the core (not installed in the test interpreter)
        e.update(extra_env)
        cmd = [sys.executable, "-s", "-m", "openfold3_ob0_opt.stock_pred", "--proof-json", proof, "--env-absent", ",".join(pre),
               "--kit-dirs", os.pathsep.join(env.kit_dirs(HOME)), "--home", HOME, "--"] + list(stock_args)
        r = subprocess.run(cmd, env=e, capture_output=True, text=True)
        return r, json.load(open(proof)) if os.path.isfile(proof) else None


def test_stock_caller_refuses_a_forbidden_name():
    r, proof = _run_stock({"OF3_FAST_INIT": "1"})
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "NOT STOCK" in r.stderr and proof["forbidden_present"] == ["OF3_FAST_INIT"]


def test_stock_caller_refuses_a_kit_dir_on_pythonpath():
    r, proof = _run_stock({"PYTHONPATH": _stubs.subprocess_pythonpath(HOME, modes.hook_dir(HOME, "fast_inference"))})
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and proof["kit_dirs_on_path"]
    assert "NOT STOCK" in r.stderr and proof["ok"] is False and "not importable" not in r.stderr        # refused by the proof, not by the missing package


def test_stock_caller_proves_clean_then_needs_openfold3(tmp_path):
    """A clean environment whose `openfold3` has no run_openfold module (a stub shadows any installed one): the proof passes, the entry
    point is refused by name (NOT STOCK, exit 3)."""
    stub = str(tmp_path / "site"); os.makedirs(os.path.join(stub, "openfold3")); open(os.path.join(stub, "openfold3", "__init__.py"), "w").close()
    r, proof = _run_stock({"PYTHONPATH": os.pathsep.join([stub, os.path.join(HOME, "opt"), _stubs.core_dir()]), "OPENFOLD3_OB0_CKPT": "/w.pt"})
    assert proof["ok"] and "ENV-CLEAN ok" in r.stderr
    assert proof["stock_argv"][:2] == ["--query-json", "q.json"]
    assert proof["runner_yaml"] == os.path.join(HOME, modes.STOCK_YAML)                                 # the stock configuration added (the stock caller's default yaml)
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "not importable" in r.stderr, r.stderr[-600:]


STUB_CLI = textwrap.dedent("""
    import json, os, sys
    class _Cli:                                            # the shape of the click group `openfold3.run_openfold:cli` the stock caller calls
        def main(self, args=None, prog_name=None, standalone_mode=True):
            args = list(args or [])
            out = args[args.index("--output-dir") + 1]
            os.makedirs(os.path.join(out, "q", "seed_42"), exist_ok=True)
            with open(os.path.join(out, "q", "seed_42", "q_seed_42_sample_1_model.cif"), "w") as fh:
                fh.write("data_q")
            with open(os.path.join(out, "argv.json"), "w") as fh:
                json.dump({"args": args, "prog_name": prog_name, "standalone_mode": standalone_mode, "ckpt_env": os.environ.get("OPENFOLD3_OB0_CKPT")}, fh)
            d = os.environ.get("STUB_LOAD_KIT_DIR")
            if d:                                          # the failure the after-call gate must catch: kit code loaded by the stock call
                sys.path.insert(0, d)
                import of3_fastinit  # noqa: F401
            raise SystemExit(0)
    cli = _Cli()
""")


def test_stock_caller_refuses_without_a_checkpoint(tmp_path):
    """No --inference-ckpt-path/-name on the command and OPENFOLD3_OB0_CKPT unset: refused by name BEFORE the stock CLI (upstream would resolve
    $OPENFOLD_CACHE/ckpt_root and write it, then reach its download fallback — a frozen-weights run names its checkpoint)."""
    stub = _stub_openfold3(str(tmp_path / "site"))
    e = {"PYTHONPATH": _stubs.subprocess_pythonpath(HOME, stub), "OPENFOLD3_OB0_CKPT": ""}                # the box's own OPENFOLD3_OB0_CKPT, if any, blanked
    r, proof = _run_stock(e)
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "NOT STOCK: no checkpoint" in r.stderr and "ckpt_root" in r.stderr, r.stderr[-600:]   # upstream 0.5.0 resolves <OPENFOLD_CACHE>/ckpt_root and writes it: named
    assert proof["not_stock_after"] == "no checkpoint" and not os.path.exists(os.path.join(str(tmp_path / "out"), "argv.json"))


def _stub_openfold3(root):
    pkg = os.path.join(root, "openfold3")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    with open(os.path.join(pkg, "run_openfold.py"), "w") as fh:
        fh.write(STUB_CLI)
    return root


def test_stock_caller_runs_the_stock_cli_entry_point(tmp_path):
    """On a clean environment the stock caller calls `openfold3.run_openfold:cli` with `predict`, the caller's arguments unchanged plus
    the stock configuration and the checkpoint from OPENFOLD3_OB0_CKPT, returns the CLI's exit code, and the proof after the call is clean."""
    stub = _stub_openfold3(str(tmp_path / "site"))
    out = str(tmp_path / "out")
    r, proof = _run_stock({"PYTHONPATH": _stubs.subprocess_pythonpath(HOME, stub), "OPENFOLD3_OB0_CKPT": "/w.pt"},
                          stock_args=("--query-json", "q.json", "--output-dir", out))
    assert r.returncode == 0, r.stderr[-800:]
    assert proof["ok"] and proof["ok_after"] and proof["exit_code"] == 0 and proof["n_cif"] == 1
    assert proof["kit_modules_loaded_after"] == [] and proof["kit_hooks_installed_after"] == []
    got = json.load(open(os.path.join(out, "argv.json")))
    assert got["prog_name"] == "run_openfold" and got["standalone_mode"] is True and got["ckpt_env"] == "/w.pt"
    args = got["args"]
    assert args[0] == "predict" and args[1:5] == ["--query-json", "q.json", "--output-dir", out]
    assert args[args.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_YAML)                  # the stock configuration added
    assert args[args.index("--inference-ckpt-path") + 1] == "/w.pt"
    assert proof["checkpoint"]["path"] == "/w.pt" and proof["checkpoint"]["hashed"] is False               # the parent hashes, not the subprocess
    assert "stock: run_openfold predict" in r.stderr and "NOT STOCK" not in r.stderr


def test_stock_caller_refuses_when_kit_code_loads_during_the_call(tmp_path):
    """The after-call scan is a gate: a kit module imported while the stock CLI ran sets ok false, prints NOT STOCK and exits 3 — the
    CLI's own exit code (0 here) does not stand."""
    stub = _stub_openfold3(str(tmp_path / "site"))
    out = str(tmp_path / "out")
    r, proof = _run_stock({"PYTHONPATH": _stubs.subprocess_pythonpath(HOME, stub), "STUB_LOAD_KIT_DIR": modes.hook_dir(HOME, "fast_inference"), "OPENFOLD3_OB0_CKPT": "/w.pt"},
                          stock_args=("--query-json", "q.json", "--output-dir", out))
    assert r.returncode == stock_pred.EXIT_NOT_STOCK
    assert proof["ok"] is False and proof["ok_after"] is False and proof["exit_code"] == 0 and proof["n_cif"] == 1
    assert proof["kit_modules_loaded_after"] == ["of3_fastinit"] and "kit code loaded during the stock call" in proof["not_stock_after"]
    assert "NOT STOCK: kit code loaded during the stock call" in r.stderr and "of3_fastinit" in r.stderr

