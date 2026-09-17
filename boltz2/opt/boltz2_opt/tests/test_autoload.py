"""The env route: BOLTZ2_OPT installs the lazy finder (the core's, opt_core.autoload, on this kit's AutoloadSpec), fires on the first import
of boltz.model and refuses (exit 3); off/unset install nothing and import nothing of the core."""
import os
import subprocess
import sys

import pytest

from .. import _autoload, modes, stack
from . import _stubs


def test_autoload_mode_list_is_the_mode_table():
    assert set(_autoload.MODES) == set(modes.MODE_NAMES), "the finder's restated mode list (it imports nothing at interpreter start) is the mode table"

CODE_IMPORT = "import sys; import boltz.model; print('IMPORTED', [type(f).__name__ for f in sys.meta_path][:2])"


def _run(env_extra, code, tmp_path):
    site = _stubs.unpack_wheel(str(tmp_path / "site"))
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    env["PYTHONPATH"] = site; env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_finder_installed_only_for_a_mode(tmp_path):
    r = _run({}, "import sys; print([type(f).__name__ for f in sys.meta_path])", tmp_path)
    assert "'Finder'" not in r.stdout
    r = _run({"BOLTZ2_OPT": "off"}, "import sys; print([type(f).__name__ for f in sys.meta_path])", tmp_path)
    assert "'Finder'" not in r.stdout and r.returncode == 0
    r = _run({"BOLTZ2_OPT": "exact"}, "import sys; print([type(f).__name__ for f in sys.meta_path])", tmp_path)
    assert "'Finder'" in r.stdout and r.returncode == 0, r.stderr
    assert "torch" not in r.stdout


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_nothing_imported_at_start(tmp_path):
    r = _run({"BOLTZ2_OPT": "exact"}, "import sys; print(sorted(m for m in sys.modules if m.startswith('boltz')))", tmp_path)
    assert r.stdout.strip() == "['boltz2_opt', 'boltz2_opt._autoload', 'boltz2_opt._core_gate']", r.stdout   # a named mode runs the kit's core pin gate (stdlib only) before the core


CODE_CENSUS = ("import sys; F = [f for f in sys.meta_path if type(f).__name__ == 'Finder']; "
               "print(sorted(m for m in sys.modules if m.startswith(('boltz', 'opt_core'))), [type(f).__module__ for f in F], [f.mode for f in F])")


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_nothing_of_the_core_unless_a_mode_is_named(tmp_path):
    """The .pth runs in every process: with BOLTZ2_OPT unset or off nothing of the core is imported; a named mode arms the core's finder
    (opt_core.autoload) with the selection stripped and case-folded; `python -S` processes no .pth and is untouched."""
    for extra in ({}, {"BOLTZ2_OPT": "off"}, {"BOLTZ2_OPT": " OFF "}):
        r = _run(extra, CODE_CENSUS, tmp_path)
        assert r.returncode == 0 and r.stdout.strip() == "['boltz2_opt', 'boltz2_opt._autoload'] [] []", (extra, r.stdout, r.stderr)
    for raw, mode in (("exact", "exact"), ("FAST", "fast"), (" Big ", "big")):
        r = _run({"BOLTZ2_OPT": raw}, CODE_CENSUS, tmp_path)
        assert r.returncode == 0 and r.stdout.strip() == f"['boltz2_opt', 'boltz2_opt._autoload', 'boltz2_opt._core_gate', 'opt_core', 'opt_core.autoload'] ['opt_core.autoload'] ['{mode}']", (raw, r.stdout, r.stderr)
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}; env["BOLTZ2_OPT"] = "turbo"
    r = subprocess.run([sys.executable, "-S", "-c", "import sys; print('ran', [type(f).__name__ for f in sys.meta_path if type(f).__name__ == 'Finder'], 'boltz2_opt' in sys.modules)"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "ran [] False" and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stdout, r.stderr)


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_env_route_refuses_on_the_trigger_and_exits_3(tmp_path):
    r = _run({"BOLTZ2_OPT": "exact", "MODEL_OPT": stack.tree_dir()}, CODE_IMPORT, tmp_path)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert "[boltz2-opt] NOT ACTIVE:" in r.stdout and "IMPORTED" not in r.stdout


def test_env_route_off_runs_the_import_untouched(tmp_path):
    r = _run({"BOLTZ2_OPT": "off"}, CODE_IMPORT, tmp_path)
    assert r.returncode == 0 and "IMPORTED" in r.stdout and "'Finder'" not in r.stdout


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_unknown_env_mode_is_named_and_exits_3(tmp_path):
    """An unknown selection under the variable never runs stock silently: the NOT ACTIVE line, then exit EXIT_NOT_ACTIVE (D71)."""
    r = _run({"BOLTZ2_OPT": "turbo"}, "print('ran')", tmp_path)
    assert r.returncode == 3 and "ran" not in r.stdout and "[boltz2-opt] NOT ACTIVE: unknown BOLTZ2_OPT='turbo'" in r.stderr, (r.returncode, r.stdout, r.stderr)
    assert "Traceback" not in r.stderr and len([l for l in r.stderr.splitlines() if l.strip()]) == 1, r.stderr
    with pytest.raises(SystemExit) as e:
        _autoload.install({"BOLTZ2_OPT": "turbo"})
    assert e.value.code == _autoload.EXIT_NOT_ACTIVE == 3


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_find_spec_probe_does_not_disarm_the_hook(tmp_path):
    """importlib.util.find_spec('boltz.model') before the real import (a probe) leaves the hook armed: the import that follows fires it."""
    code = ("import importlib.util, sys; s = importlib.util.find_spec('boltz.model'); assert s is not None; "
            "f = [x for x in sys.meta_path if type(x).__name__ == 'Finder'][0]; assert f.armed, 'probe disarmed the hook'; "
            "import boltz.model; print('IMPORTED')")
    r = _run({"BOLTZ2_OPT": "exact", "MODEL_OPT": stack.tree_dir()}, code, tmp_path)
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE:" in r.stdout and "IMPORTED" not in r.stdout, (r.returncode, r.stdout, r.stderr)


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_undeclared_opt_names_are_refused_by_name(tmp_path, monkeypatch):
    """BOLTZ2_OPT_MODE=exact (a mistyped name under the prefix) is never ignored: the env route exits 3 with its line, the CLI exits 2."""
    r = _run({"BOLTZ2_OPT_MODE": "exact"}, "print('ran')", tmp_path)
    assert r.returncode == 3 and "ran" not in r.stdout and "NOT ACTIVE: undeclared BOLTZ2_OPT_MODE" in r.stderr, (r.returncode, r.stdout, r.stderr)
    assert _autoload.undeclared_names({"BOLTZ2_OPT": "exact", "BOLTZ2_OPT_MODE": "x", "BOLTZ2_OPTS": "y", "BOLTZ_LEVERS": "z", "BOLTZ2_BIG_XL_FREE": "0", "BOLTZ2_BIG_ALLOW_PARTIAL": "1"}) == ["BOLTZ2_BIG_ALLOW_PARTIAL", "BOLTZ2_BIG_XL_FREE", "BOLTZ2_OPTS", "BOLTZ2_OPT_MODE"], "a memory-line word in the caller's environment is refused by name like a mistyped selection"
    with pytest.raises(SystemExit) as e:
        _autoload.install({"BOLTZ2_OPT_MODE": "exact"})
    assert e.value.code == 3
    from .. import cli
    monkeypatch.setenv("BOLTZ2_OPT_MODE", "exact")
    assert cli.main(["check", "--mode", "exact"]) == 2


def test_core_missing_is_not_active_exit_3():
    """BOLTZ2_OPT=<mode> with the shared core absent from sys.path: the hook prints the NOT ACTIVE line with reason=core_missing:<module> and
    the process exits 3 — never a traceback, never a silent stock run (the .pth form: os._exit after the line). `-S`: no site, so neither the
    kit's .pth nor an installed core is on the path — only this package, by PYTHONPATH."""
    import subprocess, sys
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # opt/ (holds boltz2_opt/)
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ2_OPT") and k != "PYTHONPATH"}
    env.update({"PYTHONPATH": pkg_parent, "BOLTZ2_OPT": "exact"})
    probe = subprocess.run([sys.executable, "-S", "-c", "import importlib.util as u; print(u.find_spec('opt_core') is None)"], env=env, capture_output=True, text=True, timeout=60)
    assert probe.stdout.strip() == "True", "the premise: opt_core is not importable under -S with only opt/ on the path"
    r = subprocess.run([sys.executable, "-S", "-c", "import boltz2_opt._autoload; print('reached')"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr and "reached" not in r.stdout, (r.returncode, r.stdout, r.stderr[-600:])
    env["BOLTZ2_OPT"] = "off"                                                                       # the stock arm never touches the core
    r = subprocess.run([sys.executable, "-S", "-c", "import boltz2_opt._autoload; print('reached')"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "reached" in r.stdout and r.stderr == "", (r.returncode, r.stdout, r.stderr[-600:])
