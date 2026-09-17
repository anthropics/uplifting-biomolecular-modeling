"""run.sh's contract (config sourcing, the mode/env and variant/env refusals, the install line), the warm-up fixture and the warm verdict
on a refused box, and the build backend's .pth at the wheel root (when setuptools is available)."""
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from e1_opt import report, stack, warm
from e1_opt.tests import _stubs

RUN_SH = os.path.join(os.path.dirname(_stubs.OPT_DIR), "run.sh")
H100_ENV = os.path.join(os.path.dirname(_stubs.OPT_DIR), "configs", "h100.env")


def _bash(args, env):
    if not shutil.which("bash") or not os.path.isfile(RUN_SH):
        pytest.skip("bash or e1/run.sh not available")
    return subprocess.run(["bash", RUN_SH, *args], env=env, capture_output=True, text=True, timeout=120)


def test_run_sh_refuses_mode_env_disagreement_and_bad_usage(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    env = _stubs.child_env(box, E1_OPT="off")
    p = _bash(["check", "--variant", box["variant"], "--mode", "exact"], env)
    assert p.returncode == 2 and p.stdout.startswith("[e1-opt] NOT ACTIVE: mode exact disagrees with E1_OPT=off from the environment"), p.stdout + p.stderr
    # with a config named, a set E1_OPT still comes from the environment (a config carries no mode): the line says so
    p = subprocess.run(["bash", RUN_SH, "check", "--config", "h100", "--variant", box["variant"], "--mode", "exact"], env=dict(env, E1_OPT="off"), capture_output=True, text=True)
    assert p.returncode == 2 and "disagrees with E1_OPT=off from the environment" in p.stdout, p.stdout + p.stderr
    env = _stubs.child_env(box, E1_VARIANT="600m")
    p = _bash(["check", "--variant", "300m"], env)
    assert p.returncode == 2 and "variant 300m disagrees with E1_VARIANT=600m" in p.stdout
    p = _bash(["frobnicate"], _stubs.child_env(box))
    assert p.returncode == 2
    p = _bash(["check", "--variant", "900m"], _stubs.child_env(box))
    assert p.returncode == 2 and "not a variant" in p.stderr
    p = _bash(["check", "--config", "nope", "--variant", "300m"], _stubs.child_env(box))
    assert p.returncode == 2 and "no such config" in p.stderr


def test_run_sh_reaches_the_package_and_sources_the_config(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    env = _stubs.child_env(box)
    env["TRITON_CACHE_DIR"] = str(tmp_path / "jit" / "triton")          # a pre-set cache directory passes through the config untouched
    (tmp_path / "jit").mkdir()
    p = _bash(["check", "--config", "h100", "--variant", box["variant"]], env)
    line = p.stdout.splitlines()[0]
    assert report.RE_DRY_RUN.match(line) and f"gpu={box['gpu'][0]} " in line and "card=h100" in line, p.stdout + p.stderr
    assert p.returncode == 0 and line.endswith("would_refuse=none") and ' notes="hub kernel snapshot ' in line, line   # the placeholder kernel file: named, never a refusal


def test_config_keys_the_jit_caches_under_model_opt_jit_root(monkeypatch, tmp_path):
    """MODEL_OPT_JIT_ROOT set: TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR = <root>/<e1_opt.stack.jit_cache_key()>/{triton,inductor}, each unless
    already set; unset: the config exports neither (the libraries' own defaults apply)."""
    box = _stubs.setup_box(monkeypatch, tmp_path)
    if not shutil.which("bash") or not os.path.isfile(H100_ENV):
        pytest.skip("bash or e1/configs/h100.env not available")
    show = f'source "{H100_ENV}" && printf "%s|%s|%s" "${{TRITON_CACHE_DIR:-}}" "${{TORCHINDUCTOR_CACHE_DIR:-}}" "${{MODEL_OPT_STACK_KEY:-}}"'
    env = {k: v for k, v in _stubs.child_env(box).items() if k not in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "MODEL_OPT_STACK_KEY", "MODEL_OPT_JIT_ROOT")}
    key = subprocess.run([sys.executable, "-c", "from e1_opt.stack import jit_cache_key; print(jit_cache_key())"], env=env, capture_output=True, text=True, timeout=60).stdout.strip()
    assert key.startswith("torch"), key
    p = subprocess.run(["bash", "-c", show], env=dict(env, MODEL_OPT_JIT_ROOT=str(tmp_path / "jit")), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout == f"{tmp_path}/jit/{key}/triton|{tmp_path}/jit/{key}/inductor|{key}", p.stdout + p.stderr
    p = subprocess.run(["bash", "-c", show], env=dict(env, MODEL_OPT_JIT_ROOT=str(tmp_path / "jit"), TRITON_CACHE_DIR="/pre/set"), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout == f"/pre/set|{tmp_path}/jit/{key}/inductor|{key}", p.stdout + p.stderr
    p = subprocess.run(["bash", "-c", show], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout == "||", p.stdout + p.stderr


def test_run_sh_refuses_when_the_package_is_not_installed(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    env = _stubs.child_env(box, PYTHONPATH=box["site"])
    env["PATH"] = box["bin"] + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + "/usr/bin:/bin"
    py = tmp_path / "bin" / "python"                                    # `python` on PATH = this interpreter WITHOUT its site-packages (-S): an installed e1_opt is
    py.write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n")        # invisible to it, so the refusal is exercised on an interpreter that has the package too
    py.chmod(0o755)
    p = _bash(["check", "--variant", box["variant"]], env)
    if "import e1_opt" not in open(RUN_SH).read():
        pytest.skip("no install line")
    assert p.returncode == 3 and "NOT ACTIVE: e1_opt is not installed" in p.stdout, p.stdout + p.stderr


def test_warm_fixture_is_a_valid_assay():
    def records(path):
        out, head = {}, None
        for line in open(path):
            line = line.strip()
            if line.startswith(">"):
                head = line[1:]; out[head] = ""
            elif line:
                out[head] += line
        return out
    parent = records(os.path.join(warm.FIXTURE_DIR, "parent.fasta")); muts = records(os.path.join(warm.FIXTURE_DIR, "mutants.fasta"))
    assert len(parent) == 1 and next(iter(parent)).startswith("synthetic") and len(muts) >= 3
    plen = len(next(iter(parent.values())))
    assert all(len(s) == plen for s in muts.values())                     # single substitutions of the parent: the tool's own input contract


def test_warm_reports_a_refused_scoring(monkeypatch, tmp_path, capsys):
    box = _stubs.setup_box(monkeypatch, tmp_path)                       # no weights file where the activation looks: the child `score` refuses by name, warm says so
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no_such_cache"))
    (tmp_path / "triton").mkdir()
    res = warm.run("exact", box["variant"], det=0)
    out = capsys.readouterr().out.splitlines()
    assert res["status"] == "FAIL" and res["exit_code"] == 3 and res["reason"].startswith("the scoring was refused"), (res, out)
    assert any(s.startswith("[e1-opt] NOT ACTIVE: weights:") for s in out), out
    last = out[-1]
    assert report.RE_WARM.match(last) and last.startswith(f"[e1-opt] WARM FAIL mode=exact variant={box['variant']} rows=0 triton_cache=0->0")


def test_wheel_carries_the_autoload_pth(tmp_path):
    try:
        import setuptools  # noqa: F401
    except ImportError:
        pytest.skip("setuptools not installed")
    src = tmp_path / "opt"                                             # build from a copy: setuptools writes build/ and *.egg-info beside the sources
    src.mkdir()
    for name in ("pyproject.toml", "_build_backend.py", "e1_opt_autoload.pth"):
        shutil.copy(os.path.join(_stubs.OPT_DIR, name), src / name)
    shutil.copytree(os.path.join(_stubs.OPT_DIR, "e1_opt"), src / "e1_opt", ignore=shutil.ignore_patterns("__pycache__"))
    p = subprocess.run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "--no-build-isolation", "-w", str(tmp_path / "w"), "-q"],
                       capture_output=True, text=True, timeout=600)
    tmp_path = tmp_path / "w"
    if p.returncode != 0:
        if "No module named pip" in (p.stderr or p.stdout):
            pytest.skip("pip is not available on this interpreter")
        pytest.fail(f"pip wheel failed: {(p.stderr or p.stdout)[-600:]}")             # a build the backend refuses (e.g. a hand-written .pth) is a failure, never a skip
    whl = [f for f in os.listdir(tmp_path) if f.endswith(".whl")]
    from e1_opt import __version__, _names
    assert len(whl) == 1 and whl[0].startswith(f"e1_opt-{__version__}-")
    spec = importlib.util.spec_from_file_location("_e1_build_backend", str(src / "_build_backend.py"))   # the kit's own generator: the shipped .pth IS its output
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    want = backend.pth_text("e1_opt", _names.ENV, report.PREFIX.strip("[]"), _names.EXIT_NOT_ACTIVE).encode()
    assert open(os.path.join(_stubs.OPT_DIR, "e1_opt_autoload.pth"), "rb").read() == want
    with zipfile.ZipFile(os.path.join(tmp_path, whl[0])) as z:
        names = z.namelist()
        assert "e1_opt_autoload.pth" in names and z.read("e1_opt_autoload.pth") == want
        record = [n for n in names if n.endswith(".dist-info/RECORD")][0]
        assert "e1_opt_autoload.pth,sha256=" in z.read(record).decode()
        assert any(n.endswith("fixtures/warm/parent.fasta") for n in names) and "e1_opt/stock_score.py" in names
        meta = z.read([n for n in names if n.endswith(".dist-info/METADATA")][0]).decode()
        assert "Private :: Do Not Upload" in meta
        entry = z.read([n for n in names if n.endswith("entry_points.txt")][0]).decode()
        assert "e1-opt = e1_opt.cli:main" in entry
