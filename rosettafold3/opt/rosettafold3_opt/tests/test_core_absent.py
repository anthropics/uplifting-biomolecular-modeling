"""An ABSENT core, or a core older than the pin (the pin is a FLOOR: a newer core passes, an older one is refused), is refused BY NAME
with exit 3 BEFORE any ``opt_core`` import — through every entry route: ``python -m rosettafold3_opt <verb>``, ``bash run.sh <verb>``, the
``.pth`` hook (``ROSETTAFOLD3_OPT=<mode>`` + the upstream import), ``configs/h100.env`` when sourced, and ``rosettafold3_opt.enable``
in-process. Statement one of every route is the pin gate ``_core_gate.gate`` (the house template: ``reason=core_missing:opt_core (…)`` /
``reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed v<have> at <dir>``, ``<have>`` being ``?`` when the installed
``opt_core`` carries no ``__version__`` literal at all); the producer check ``_core.require`` comes second (``reason=producer_missing:…``).
Cores are shadowed on PYTHONPATH: ``stale`` = an ``opt_core`` whose ``__version__`` is older than the pin, ``unversioned`` = an ``opt_core``
with no ``__version__`` literal, ``absent`` = a relocated copy of this package whose pin path carries no checkout and no ``opt_core``
anywhere on the path (in a tree checkout the pinned core is never absent: the tree carries it and ``_core.expose_pin_path`` puts it on
sys.path), ``lying`` = an ``opt_core`` whose ``__version__`` claims the pin while its tree lacks the modules this package loads (the pin
gate passes; the producer check catches it)."""
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from .. import _core, _core_gate, stack

LINE = "[rosettafold3-opt] NOT ACTIVE: reason="
REASON = {"stale": "core_mismatch: opt_core pinned >= v", "unversioned": "core_mismatch: opt_core pinned >= v", "absent": "core_missing:opt_core (pinned >= v",
          "lying": "producer_missing:opt_core."}                            # lying: __version__ claims the pin, the tree beneath it lacks the modules → the PRODUCER check refuses
INSTALLED = {"stale": "installed v0.2.5 at ", "unversioned": "installed v? at ", "absent": "nothing importable as opt_core on sys.path",
             "lying": "imports opt_core >= 0.5"}


def _shadow(tmp_path, kind):
    """A directory to put FIRST on PYTHONPATH: an opt_core whose ``__version__`` is older than the pin (stale) or absent (unversioned),
    one whose ``__version__`` claims the pin while its tree lacks the modules this package loads (lying), or nothing at all (absent)."""
    root = tmp_path / f"shadow_{kind}"
    root.mkdir(parents=True, exist_ok=True)
    if kind == "absent":
        return str(root)
    pkg = root / "opt_core"
    if pkg.exists():                                                    # one shadow per test directory, reused across the verbs of a test
        return str(root)
    pkg.mkdir()
    if kind == "unversioned":
        (pkg / "__init__.py").write_text("")                            # no __version__ literal at all: the gate's documented '?' fallback
    elif kind == "lying":                                                # __version__ claims the pin; the tree below still lacks the producer modules (the stub loop, below)
        (pkg / "__init__.py").write_text(f'__version__ = "{_core.pin_version()}"\n')
    else:                                                                # stale: older than the pin — FLOOR semantics refuse it (never "same version, different bytes")
        (pkg / "__init__.py").write_text('__version__ = "0.2.5"\n')
    for m in ("gates", "report", "home", "kernels", "process", "jit_cache", "det", "modes", "stock_proof"):
        (pkg / f"{m}.py").write_text("")
    return str(root)


def _opt_root(tmp_path, kind):
    """The package the interpreter runs: the tree's own ``opt/`` (its pinned checkout beside it), or — for ``absent`` — a relocated copy of
    ``opt/pyproject.toml`` + ``rosettafold3_opt/`` whose ``[tool.opt_core] path`` carries no checkout."""
    if kind != "absent":
        return stack.opt_root()
    dst = tmp_path / "relocated_opt"
    if not dst.exists():
        shutil.copytree(os.path.join(stack.opt_root(), "rosettafold3_opt"), str(dst / "rosettafold3_opt"), ignore=shutil.ignore_patterns("__pycache__", "tests"))
        shutil.copyfile(_core.PYPROJECT, str(dst / "pyproject.toml"))
    return str(dst)


def _py(tmp_path):
    """The interpreter under test without its site dirs (``-S``): PYTHONPATH is then all it sees (an editable install's meta-path finder would
    otherwise serve the real core under a shadow package — a chimera no user has)."""
    w = tmp_path / "py_no_site"
    w.write_text(f"#!/bin/bash\nexec {sys.executable} -S \"$@\"\n")
    w.chmod(0o755)
    return str(w)


def _env(tmp_path, kind, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and k not in ("PYTHONPATH", "MODEL_OPT_STACK_KEY", "TRITON_CACHE_DIR")}
    env.update({"PYTHONPATH": os.pathsep.join([_shadow(tmp_path, kind), _opt_root(tmp_path, kind)]), "PYTHONDONTWRITEBYTECODE": "1", "ROSETTAFOLD3_OPT_PYTHON": _py(tmp_path)})
    env.update(extra)
    return env


def _one_line(r, kind):
    lines = [l for l in r.stderr.splitlines() if l.strip()]
    assert r.returncode == 3 and "Traceback" not in r.stderr, (kind, r.returncode, r.stdout[-300:], r.stderr[-600:])
    assert lines and lines[0].startswith(LINE + REASON[kind]) and INSTALLED[kind] in lines[0], (kind, lines)
    return lines


def test_the_gate_is_the_house_template_byte_for_byte():
    tpl = os.path.join(_core.pin_path(), "kit_template", "_core_gate.py")
    assert open(tpl, "rb").read() == open(os.path.join(os.path.dirname(_core.__file__), "_core_gate.py"), "rb").read()


def test_required_lists_every_core_module_the_package_loads():
    import glob, re
    loaded = set()
    for f in glob.glob(os.path.join(stack.opt_root(), "rosettafold3_opt", "*.py")):
        loaded |= {f"opt_core.{m}" for m in re.findall(r'_core\.load\("([a-z_.]+)"\)', open(f).read())}
    assert loaded and loaded <= set(_core.REQUIRED), sorted(loaded - set(_core.REQUIRED))
    assert _core.missing_producers() == []                              # the interpreter under test carries the pinned core whole
    g = _core_gate.gate(_core.__file__)
    assert _core_gate.version_tuple(g["installed"]["version"]) >= _core_gate.version_tuple(g["pinned"]["version"])   # the pin is a floor


@pytest.mark.parametrize("kind", ["stale", "unversioned", "absent", "lying"])
def test_python_m_refuses_by_name_before_any_command(tmp_path, kind):
    for verb in (["check", "--mode", "fast"], ["check", "--mode", "big", "--n_gpu", "2"], ["pred", "--mode", "exact", "--input", "x.json", "--out_dir", str(tmp_path / "o")], ["install", "--help"]):
        r = subprocess.run([_py(tmp_path), "-m", "rosettafold3_opt"] + verb, capture_output=True, text=True, env=_env(tmp_path, kind), cwd=str(tmp_path))   # cwd: `-m` puts it first on sys.path — never a directory holding the tree's own package
        assert len(_one_line(r, kind)) == 1, (verb, r.stderr)


@pytest.mark.parametrize("kind", ["stale", "unversioned", "lying"])       # run.sh runs the tree's own opt/: its pinned checkout is beside it, so `absent` is not a state of a tree
def test_run_sh_refuses_by_name_before_any_command(tmp_path, kind):
    for verb in (["check", "--mode", "fast"], ["pred", "--mode", "big", "--input", "x.json", "--out_dir", str(tmp_path / "o")], ["check", "--config", "h100", "--mode", "exact"]):
        r = subprocess.run(["bash", os.path.join(stack.tree_root(), "run.sh")] + verb, capture_output=True, text=True, env=_env(tmp_path, kind))
        _one_line(r, kind)


@pytest.mark.parametrize("kind", ["stale", "unversioned", "absent", "lying"])
def test_the_pth_hook_refuses_by_name_at_the_upstream_import(tmp_path, kind):
    """The env route: the finder installed (as the .pth line does), ROSETTAFOLD3_OPT=fast, then `import rf3` (a stub package): one line, exit 3."""
    stub = tmp_path / "stub"; (stub / "rf3").mkdir(parents=True)
    (stub / "rf3" / "__init__.py").write_text("")
    (stub / "rf3" / "graph_flags.py").write_text("RAN = True\n")
    code = textwrap.dedent("""
        import rosettafold3_opt._autoload            # what the site-processed .pth line does at interpreter start
        import rf3.graph_flags                        # the trigger: the hook activates BEFORE this body runs — or refuses and exits 3
        print("STOCK RAN")                            # never reached under a refused activation
    """)
    env = _env(tmp_path, kind, ROSETTAFOLD3_OPT="fast")
    env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"].split(os.pathsep)[0], str(stub)] + env["PYTHONPATH"].split(os.pathsep)[1:])
    r = subprocess.run([_py(tmp_path), "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))   # `-c` puts the cwd first on sys.path: not the tree's opt/
    assert "STOCK RAN" not in r.stdout
    assert len(_one_line(r, kind)) == 1, r.stderr


@pytest.mark.parametrize("kind", ["stale", "unversioned", "lying"])
def test_the_config_env_refuses_by_name_when_sourced(tmp_path, kind):
    """configs/h100.env derives MODEL_OPT_STACK_KEY through the package (`python -m rosettafold3_opt._require`: the pin gate, then the producer
    check), so a stale core is the NOT ACTIVE line and rc 3 from the sourced file — never an empty or defaulted key exported silently."""
    script = f"source {os.path.join(stack.tree_root(), 'configs', 'h100.env')}; rc=$?; echo rc=$rc key=${{MODEL_OPT_STACK_KEY:-unset}} triton=${{TRITON_CACHE_DIR:-unset}}; exit $rc"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_env(tmp_path, kind))
    lines = _one_line(r, kind)
    assert r.stdout.strip() == "rc=3 key=unset triton=unset" and "configs/h100.env" in lines[-1], (r.stdout, lines)   # the sourced file returns 3 (run.sh: `source … || exit $?`) and exports nothing past the probe


def test_enable_refuses_in_process_by_name(monkeypatch, capsys):
    """enable()'s statement one is the gate: a core that is not the pinned one → the line + SystemExit(3) (CoreGateRefused); a core lacking a
    module this package loads → the producer_missing line + SystemExit(3) (_core.require_or_exit)."""
    import rosettafold3_opt
    monkeypatch.setattr(_core_gate, "installed_core", lambda: {"package_dir": "/x/opt_core", "root": "/x", "version": "0.2.5"})
    with pytest.raises(SystemExit) as e:
        rosettafold3_opt.enable("fast")
    assert e.value.code == 3 and isinstance(e.value, _core_gate.CoreGateRefused) and e.value.reason == "core_mismatch"
    assert capsys.readouterr().err.strip().startswith(LINE + "core_mismatch: opt_core pinned ")
    monkeypatch.undo()
    monkeypatch.setattr(_core, "missing_producers", lambda: ["opt_core.mem.ngpu", "opt_core.mem.rowpair"])
    with pytest.raises(SystemExit) as e:                              # the producer check through _core.require_or_exit: the line + exit 3, never a raw ProducersMissing
        rosettafold3_opt.enable("fast")
    assert e.value.code == 3
    err = capsys.readouterr().err.strip()
    assert err.startswith(LINE + "producer_missing:opt_core.mem.ngpu,opt_core.mem.rowpair — ") and "imports opt_core >= 0.5" in err and "Traceback" not in err, err
    with pytest.raises(_core.ProducersMissing, match=r"^producer_missing:opt_core.mem.ngpu,opt_core.mem.rowpair — "):   # require() itself still raises the typed error (callers that want it)
        _core.require()
    assert _core.not_active_reason(_core.ProducersMissing("producer_missing:x — y")) == "reason=producer_missing:x — y"


def test_enable_under_a_lying_version_exits_by_name_in_a_subprocess(tmp_path):
    """The in-process route end to end: `import rosettafold3_opt; rosettafold3_opt.enable("fast")` on an interpreter whose first opt_core's
    __version__ claims the pin over a tree without the modules → one NOT ACTIVE line (producer_missing), rc 3, no traceback."""
    r = subprocess.run([_py(tmp_path), "-c", "import rosettafold3_opt; rosettafold3_opt.enable('fast'); print('ENABLED')"], capture_output=True, text=True, env=_env(tmp_path, "lying"))
    assert "ENABLED" not in r.stdout
    assert len(_one_line(r, "lying")) == 1, r.stderr


def test_every_entry_route_runs_the_one_producer_check():
    """DRY: the five gate sites call _core.require_or_exit (no per-site wrapper words): enable(), python -m, _require, cli.main; the .pth hook goes through enable()."""
    import inspect, rosettafold3_opt
    from .. import cli
    for src in (inspect.getsource(rosettafold3_opt.enable), open(os.path.join(os.path.dirname(_core.__file__), "__main__.py")).read(),
                open(os.path.join(os.path.dirname(_core.__file__), "_require.py")).read(), inspect.getsource(cli.main)):
        assert "_core.require_or_exit()" in src and "_core.require()" not in src
