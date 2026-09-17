"""``pred --mode off`` is provably stock: the stock CLI runs in a fresh subprocess whose environment has no name under the must-be-absent
prefixes (stock/PINS.json) and no kit directory on PYTHONPATH, which proves itself before importing anything of protenix (no forbidden
name, no kit module, no kit directory on sys.path, no armed autoload finder, not the kit's sitecustomize, torch not loaded) and writes
the proof's JSON goes to a temporary directory of the call (never the output directory). A process under the env.sh activation route (the kit's
sitecustomize active) refuses mode off."""
import json
import os
import sys
import types

import pytest

import protenix_opt
from protenix_opt import _autoload, cli, modes, stack, stock_pred
from protenix_opt.tests import _stock_stub
from protenix_opt.tests.conftest import needs_durable_install

POLLUTION = {"PTX_BLK": "2", "PTX_T1_TRANS": "fused", "FPF_OPS": "trimul_out=fpf_smalln:trimul_c256", "INFOPT_GRAPHS_GC": "0", "PF_TRIATTN": "1",
             "PROTENIX_OPT": "exact", "PROTENIX_OPT_FORCE": "1", "CUEQ_TRITON_CACHE_DIR": "/nonexistent/cueq"}


@pytest.fixture
def stub_runner(tmp_path, monkeypatch):
    """The on-disk stock stand-in, importable by this process (the stock parser) and by the subprocess (PYTHONPATH)."""
    root = _stock_stub.write_stub_runner(str(tmp_path / "stock_stub"))
    for m in [m for m in sys.modules if m == "runner" or m.startswith("runner.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.syspath_prepend(root)
    return root


@pytest.fixture
def off_env(monkeypatch):
    monkeypatch.setattr(stack, "_REPORT", None)
    for k in list(os.environ):
        if k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT):
            monkeypatch.delenv(k)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(stack, "_kit_sitecustomize_present", lambda: None)    # this test process is not under the env.sh route
    yield
    stack._REPORT = None


def _kit_dirs():
    return [os.path.realpath(d) for d in stack.kit_class_dirs()]


def _under_kit(p):
    rp = os.path.realpath(p)
    return any(rp == r or rp.startswith(r + os.sep) for r in _kit_dirs())


def test_must_be_absent_prefixes_come_from_pins():
    p = stack.pins()
    assert p["stock_environment"]["must_be_absent_prefixes"] == list(stack.DEFAULT_STOCK_ENV_ABSENT)
    assert stack.stock_env_absent() == stack.DEFAULT_STOCK_ENV_ABSENT
    assert stack.stock_env_absent({"stock_environment": {"must_be_absent_prefixes": ["FPF_"]}}) == ("FPF_", "PROTENIX_OPT", "PTX_")


def test_stock_command_strips_the_kit_names_and_kit_pythonpath(tmp_path):
    kit = stack.kit_home()
    environ = dict(POLLUTION, PATH="/usr/bin", HOME="/h", PROTENIX_ROOT_DIR="/zoo", LAYERNORM_TYPE="fast_layernorm",
                   PYTHONPATH=os.pathsep.join([os.path.join(kit, "src"), "/opt/mine", os.path.join(kit, "third_party", "kit112_src"), str(tmp_path)]))
    cmd, env, stripped = cli.stock_command(["--input", "x"], str(tmp_path / "proof.json"), environ=environ)
    assert stripped["env"] == sorted(POLLUTION) and not any(k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT) for k in env)
    assert {k: env[k] for k in ("PATH", "HOME", "PROTENIX_ROOT_DIR", "LAYERNORM_TYPE")} == {"PATH": "/usr/bin", "HOME": "/h", "PROTENIX_ROOT_DIR": "/zoo", "LAYERNORM_TYPE": "fast_layernorm"}
    assert env["PYTHONPATH"] == os.pathsep.join(["/opt/mine", str(tmp_path)]) and stripped["pythonpath"] == [os.path.join(kit, "src"), os.path.join(kit, "third_party", "kit112_src")]
    assert cmd[:4] == [sys.executable, "-s", "-m", "protenix_opt.stock_pred"] and cmd[-4:] == ["--", "pred", "--input", "x"]
    assert cmd[cmd.index("--env-absent") + 1] == ",".join(stack.DEFAULT_STOCK_ENV_ABSENT)
    assert cmd[cmd.index("--kit-dirs") + 1].split(os.pathsep) == stack.kit_class_dirs()
    _, env2, stripped2 = cli.stock_command([], "p.json", environ={"PYTHONPATH": os.path.join(kit, "src")})
    assert "PYTHONPATH" not in env2 and stripped2["pythonpath"] == [os.path.join(kit, "src")]


def test_env_proof_names_every_violation(tmp_path):
    kit = stack.kit_class_dirs()
    forward = os.path.join(kit[0], "flashpairformer", "src")
    kit_file_mod = types.ModuleType("some_helper"); kit_file_mod.__file__ = os.path.join(forward, "helper.py")
    sc = types.ModuleType("sitecustomize"); sc.__file__ = os.path.join(forward, "sitecustomize.py")
    armed = type("Finder", (), {"armed": True}); armed.__module__ = "protenix_opt._autoload"
    disarmed = type("Finder", (), {"armed": False}); disarmed.__module__ = "protenix_opt._autoload"
    proof = stock_pred.env_proof(stack.DEFAULT_STOCK_ENV_ABSENT, kit, environ={"PTX_BLK": "2", "HOME": "/h", "PROTENIX_OPT_FORCE": "1"},
                                 modules={"ptx_trunk2_levers": types.ModuleType("ptx_trunk2_levers"), "some_helper": kit_file_mod, "sitecustomize": sc,
                                          "torch": types.ModuleType("torch"), "os": os},
                                 path=[forward, "/usr/lib/python3", ""], meta_path=[armed(), disarmed()])
    assert proof["ok"] is False
    assert proof["forbidden_present"] == ["PROTENIX_OPT_FORCE", "PTX_BLK"] and proof["kit_modules_loaded"] == ["ptx_trunk2_levers", "sitecustomize", "some_helper"]
    assert proof["kit_dirs_on_path"] == [forward] and proof["autoload_armed"] == ["Finder"] and proof["kit_sitecustomize"] == sc.__file__
    assert proof["torch_loaded_before_proof"] is True
    clean = stock_pred.env_proof(stack.DEFAULT_STOCK_ENV_ABSENT, kit, environ={"HOME": "/h", "PROTENIX_ROOT_DIR": "/zoo"}, modules={"os": os, "json": json},
                                 path=["/usr/lib/python3", str(tmp_path)], meta_path=[disarmed()])
    assert clean["ok"] is True and clean["forbidden_present"] == [] and clean["kit_modules_loaded"] == [] and clean["kit_dirs_on_path"] == []
    assert clean["autoload_armed"] == [] and clean["kit_sitecustomize"] is None and clean["torch_loaded_before_proof"] is False



def _proof(tmpdir) -> dict:
    """The stock child's environment proof: the newest ``protenix_opt_stock_*/stock_env_proof.json`` under ``tmpdir`` (tempfile.tempdir points there)."""
    import glob
    return json.load(open(sorted(glob.glob(os.path.join(str(tmpdir), "protenix_opt_stock_*", cli.STOCK_PROOF_NAME)), key=os.path.getmtime)[-1], encoding="utf-8"))

@needs_durable_install
def test_off_runs_the_stock_cli_in_a_clean_subprocess(stub_runner, off_env, tmp_path, monkeypatch, capsys):
    """A polluted environment (kit switches, the package's own switches, kit directories on PYTHONPATH): the subprocess gets none of it,
    proves so, and the stock stub sees zero kit names, no kit module and no kit directory."""
    kit = stack.kit_home()
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    for k, v in POLLUTION.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([os.path.join(kit, "src"), stub_runner, os.path.join(kit, "third_party", "kit112_src")]))
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "off", "--input", "x", "--out_dir", str(out), "--seeds", "7"])
    err = capsys.readouterr().err
    assert rc == 0
    proof = _proof(tmp_path)
    assert proof["ok"] is True and proof["forbidden_present"] == [] and proof["kit_modules_loaded"] == [] and proof["kit_dirs_on_path"] == []
    assert proof["autoload_armed"] == [] and proof["kit_sitecustomize"] is None and proof["torch_loaded_before_proof"] is False and proof["no_user_site"] is True
    assert proof["stock_argv"] == ["pred", "--input", "x", "--out_dir", str(out), "--seeds", "7"] and proof["env_absent"] == list(stack.DEFAULT_STOCK_ENV_ABSENT)
    sub = [l for l in err.splitlines() if l.startswith("[protenix-opt] stock subprocess:")][-1]
    assert f"stock subprocess: {sys.executable} -s -m protenix_opt.stock_pred ... (env stripped of {','.join(sorted(POLLUTION))}; PYTHONPATH stripped of {os.path.join(kit, 'src')},{os.path.join(kit, 'third_party', 'kit112_src')})" in sub, sub
    assert not (out / cli.STOCK_PROOF_NAME).exists(), "the proof never lands in the output directory"
    rec = json.load(open(out / _stock_stub.RUN_RECORD, encoding="utf-8"))
    assert rec["pid"] != os.getpid(), "the stock CLI ran in another process"
    assert rec["env_kit_names"] == [] and rec["modules_kit"] == [] and rec["torch_loaded"] is False and rec["no_user_site"] is True
    assert not any(_under_kit(p) for p in rec["sys_path"] if p) and rec["pythonpath"] == stub_runner
    assert rec["sitecustomize"] is None or not _under_kit(rec["sitecustomize"])
    assert rec["params"] == {**_stock_stub.STOCK_PRED_DEFAULTS, "input": "x", "out_dir": str(out), "seeds": "7"}
    assert os.environ["PROTENIX_OPT"] == "exact", "the pred process's own environment is left alone"


@needs_durable_install
def test_off_passes_the_stock_exit_code_through(stub_runner, off_env, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", stub_runner)
    out = tmp_path / "f"
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    assert cli.main(["pred", "--mode", "off", "--input", "FAIL", "--out_dir", str(out)]) == 5
    assert _proof(tmp_path)["ok"] is True


@pytest.mark.skipif(not os.path.isfile(os.path.join(stack.kit_home(), "env.sh")), reason="kit env.sh not in the tree")
@needs_durable_install
def test_off_after_env_sh_is_sourced(stub_runner, off_env, tmp_path):
    """The environment env.sh leaves behind (ARM=E): every variable it exports under the must-be-absent prefixes and every PYTHONPATH entry
    it adds are stripped for the subprocess, which proves itself clean and runs the stock stub."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT)}
    base["PYTHONPATH"] = stub_runner
    before, after, _ = modes.source_env_sh(stack.kit_home(), "E", base)
    exported = sorted(k for k in after if k not in before or after[k] != before[k])
    assert any(k.startswith("PTX_") for k in exported) and "PYTHONPATH" in exported and "CUEQ_TRITON_CACHE_DIR" in exported
    proof_path = str(tmp_path / "proof.json")
    cmd, env, stripped = cli.stock_command(["--input", "x", "--out_dir", str(tmp_path / "o")], proof_path, environ=after)
    assert not any(k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT) for k in env), "zero kit keys"
    assert set(k for k in exported if k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT)) <= set(stripped["env"])
    assert env["PYTHONPATH"] == stub_runner and all(_under_kit(p) for p in stripped["pythonpath"]) and len(stripped["pythonpath"]) >= 3
    assert env.get("LAYERNORM_TYPE") == after["LAYERNORM_TYPE"], "stock's own variable stays"
    import subprocess
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[protenix-opt stock] ENV-CLEAN ok:" in r.stderr
    proof = json.load(open(proof_path, encoding="utf-8"))
    assert proof["ok"] is True and proof["forbidden_present"] == [] and proof["kit_modules_loaded"] == [] and proof["kit_dirs_on_path"] == []
    rec = json.load(open(tmp_path / "o" / _stock_stub.RUN_RECORD, encoding="utf-8"))
    assert rec["env_kit_names"] == [] and rec["modules_kit"] == [] and not any(_under_kit(p) for p in rec["sys_path"] if p)


@needs_durable_install
def test_subprocess_refuses_a_polluted_environment(stub_runner, tmp_path):
    """The stock caller run by hand in a dirty environment: the proof fails, nothing runs, exit 3."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT)}
    env.update(PTX_BLK="2", PYTHONPATH=stub_runner)
    proof_path = str(tmp_path / "proof.json")
    r = subprocess.run([sys.executable, "-s", "-m", "protenix_opt.stock_pred", "--proof-json", proof_path, "--env-absent", "PTX_,FPF_",
                        "--kit-dirs", os.pathsep.join(stack.kit_class_dirs()), "--", "pred", "--input", "x", "--out_dir", str(tmp_path / "o")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "[protenix-opt stock] NOT STOCK: forbidden env ['PTX_BLK']" in r.stderr
    proof = json.load(open(proof_path, encoding="utf-8"))
    assert proof["ok"] is False and proof["forbidden_present"] == ["PTX_BLK"]
    assert not (tmp_path / "o").exists(), "the stock CLI did not run"


def test_off_is_refused_under_the_env_sh_route(stub_runner, tmp_path, monkeypatch):
    """The kit's sitecustomize active in the pred process (env.sh sourced before python started): off is refused by name, no subprocess."""
    monkeypatch.setattr(stack, "_REPORT", None)
    monkeypatch.delenv("PROTENIX_OPT", raising=False)
    monkeypatch.setattr(stack, "_kit_sitecustomize_present", lambda: os.path.join(stack.kit_home(), stack.SITECUSTOMIZE_RELPATH))
    out = tmp_path / "o"
    assert cli.main(["pred", "--mode", "off", "--input", "x", "--out_dir", str(out)]) == cli.EXIT_NOT_ACTIVE
    rep = protenix_opt.status()
    assert rep["refused"] is True and rep["activated_by"] == "sitecustomize" and "mode off refused: the kit's sitecustomize is active" in rep["reason"]
    assert not out.exists(), "nothing ran"
    stack._REPORT = None
    assert cli.main(["check", "--mode", "off"]) == cli.EXIT_NOT_ACTIVE
    stack._REPORT = None
    with pytest.raises(protenix_opt.ActivationError):
        protenix_opt.enable("off", strict=True)
    stack._REPORT = None


def test_help_disarms_the_autoload_finder(stub_runner, monkeypatch, capsys):
    f = _autoload.install({"PROTENIX_OPT": "exact"})
    assert f is not None and f in sys.meta_path
    try:
        assert cli.main(["pred", "--help"]) == 0
        assert f not in sys.meta_path and f.armed is False
    finally:
        if f in sys.meta_path:
            sys.meta_path.remove(f)


@needs_durable_install
def test_proof_is_written_before_and_after_the_stock_call(stub_runner, off_env, tmp_path, monkeypatch):
    """The proof file exists (without the after-scan) when the stock CLI runs and is rewritten once the call returns, with
    kit_modules_loaded_after."""
    monkeypatch.setenv("PYTHONPATH", stub_runner)
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "o"
    assert cli.main(["pred", "--mode", "off", "--input", "x", "--out_dir", str(out)]) == 0
    rec = json.load(open(out / _stock_stub.RUN_RECORD, encoding="utf-8"))
    assert rec["proof_at_call"] is not None and "kit_modules_loaded" in rec["proof_at_call"]["keys"]
    assert "kit_modules_loaded_after" not in rec["proof_at_call"]["keys"], "the after-scan is added once the stock call has returned"
    proof = _proof(tmp_path)
    assert proof["ok"] is True and proof["kit_modules_loaded"] == [] and proof["kit_modules_loaded_after"] == []


@needs_durable_install
def test_kit_module_imported_by_the_stock_call_is_recorded(stub_runner, tmp_path):
    """A kit-named module imported during the stock call shows in kit_modules_loaded_after (the before-scan stays clean), and is named
    on stderr; the stock exit code passes through."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT)}
    env["PYTHONPATH"] = stub_runner
    proof_path = str(tmp_path / "proof.json")
    r = subprocess.run([sys.executable, "-s", "-m", "protenix_opt.stock_pred", "--proof-json", proof_path, "--env-absent", "PTX_,FPF_",
                        "--kit-dirs", os.pathsep.join(stack.kit_class_dirs()), "--", "pred", "--input", "IMPORT_KIT", "--out_dir", str(tmp_path / "o")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    proof = json.load(open(proof_path, encoding="utf-8"))
    assert proof["kit_modules_loaded"] == [] and proof["kit_modules_loaded_after"] == [_stock_stub.KIT_MARKER_MODULE]
    assert f"[protenix-opt stock] KIT MODULES LOADED during the stock call: ['{_stock_stub.KIT_MARKER_MODULE}']" in r.stderr
    r = subprocess.run([sys.executable, "-s", "-m", "protenix_opt.stock_pred", "--proof-json", proof_path, "--env-absent", "PTX_,FPF_",
                        "--kit-dirs", os.pathsep.join(stack.kit_class_dirs()), "--", "pred", "--input", "FAIL", "--out_dir", str(tmp_path / "f")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 5, "the stock exit code passes through the finally"
    assert json.load(open(proof_path, encoding="utf-8"))["kit_modules_loaded_after"] == []


def test_stock_env_names_and_strip():
    environ = dict(POLLUTION, PATH="/usr/bin", PROTENIX_ROOT_DIR="/zoo")
    assert cli.stock_env_names(environ) == sorted(POLLUTION)
    assert cli.stock_env_names(environ, keep=("PROTENIX_OPT",)) == sorted(set(POLLUTION) - {"PROTENIX_OPT"})
    popped = cli.strip_stock_env(environ, keep=("PROTENIX_OPT",))
    assert popped == sorted(set(POLLUTION) - {"PROTENIX_OPT"}) and set(environ) == {"PATH", "PROTENIX_ROOT_DIR", "PROTENIX_OPT"}

