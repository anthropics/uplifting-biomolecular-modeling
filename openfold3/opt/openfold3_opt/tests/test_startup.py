"""Interpreter start-up: `import openfold3_opt` imports nothing else; the autoload finder is installed only for OPENFOLD3_OPT=exact|fast|big,
fires after the trigger package's body, and exits 3 when the mode cannot be activated."""
import os
import subprocess

import pytest
import sys

from openfold3_opt import modes, _autoload
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_import_is_light():
    code = "import sys; import openfold3_opt; print(sorted(m for m in sys.modules if m.startswith('openfold3_opt') or m in ('torch','openfold3')))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": _stubs.subprocess_pythonpath(HOME)})
    assert r.returncode == 0, r.stderr
    loaded = eval(r.stdout.strip())
    assert set(loaded) <= {"openfold3_opt", "openfold3_opt._autoload"}, r.stdout     # _autoload is the installed .pth's own import (nothing else)


def test_install_only_for_a_mode():
    from opt_core.autoload import Finder
    for f in list(sys.meta_path):
        if isinstance(f, Finder):
            sys.meta_path.remove(f)
    assert _autoload.install({}) is None and _autoload.install({"OPENFOLD3_OPT": "off"}) is None
    with pytest.raises(SystemExit) as e:                                                # an unknown selection: the NOT ACTIVE line, exit 3 — never stock under the variable
        _autoload.install({"OPENFOLD3_OPT": "nope"})
    assert e.value.code == 3
    with pytest.raises(SystemExit) as e:                                                # a mistyped switch under the prefix is refused by name, not ignored
        _autoload.install({"OPENFOLD3_OPT": "fast", "OPENFOLD3_OPT_LINE": "cueq"})
    assert e.value.code == 3
    f = _autoload.install({"OPENFOLD3_OPT": "fast"})
    try:
        assert isinstance(f, Finder) and f.mode == "fast" and f.armed and sys.meta_path[0] is f          # the core's finder on this kit's spec
        assert _autoload.install({"OPENFOLD3_OPT": "fast"}) is f
        assert _autoload.install({"OPENFOLD3_OPT": " Fast "}) is f                                       # the selection stripped and case-folded
    finally:
        sys.meta_path.remove(f)


def test_nothing_of_the_core_unless_a_mode_is_named():
    """The .pth route in fresh interpreters: unset / off / " OFF " import nothing of the core; a named mode loads the finder's module alone;
    `python -S` (no site, no .pth) is untouched."""
    # .pth files are only processed for directories registered via site.addsitedir() (what `pip install -e` sets up) -- plain
    # PYTHONPATH membership never triggers it. This CPU run reaches `opt` via PYTHONPATH (no install), so the subprocess
    # registers it itself, reproducing exactly what an installed environment's site initialisation would do.
    code = ("import site; site.addsitedir(%r)\n"
            "import sys; print(sorted(m for m in sys.modules if m == 'opt_core' or m.startswith('opt_core.')))") % os.path.join(HOME, "opt")
    base = {k: v for k, v in os.environ.items() if not k.startswith("OPENFOLD3_OPT")}
    for env in ({}, {"OPENFOLD3_OPT": "off"}, {"OPENFOLD3_OPT": " OFF "}):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**base, **env, "PYTHONPATH": _stubs.core_dir()})
        assert r.returncode == 0 and r.stdout.strip() == "[]", (env, r)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**base, "OPENFOLD3_OPT": "exact", "PYTHONPATH": _stubs.core_dir()})
    assert r.returncode == 0 and r.stdout.strip() == "['opt_core', 'opt_core.autoload']", r
    r = subprocess.run([sys.executable, "-S", "-c", "import sys; print('reached', 'openfold3_opt' in sys.modules)"], capture_output=True, text=True,
                       env={**base, "OPENFOLD3_OPT": "bogus"})
    assert r.returncode == 0 and "reached False" in r.stdout and "NOT ACTIVE" not in r.stderr, r


def test_finder_fires_after_the_trigger_and_exits_3_when_refused(tmp_path):
    """A stub `openfold3` package + a 0.5.0 stub distribution: the finder fires on `import openfold3`, enable() refuses (pin), exit 3."""
    site = tmp_path / "site"
    (site / "openfold3").mkdir(parents=True)
    (site / "openfold3" / "__init__.py").write_text("BODY_RAN = True\n")
    di = site / "openfold3-0.5.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: openfold3\nVersion: 0.5.0\n")
    code = ("import sys, openfold3_opt._autoload as a; assert a.FINDER is not None and a.FINDER.armed\n"
            "import openfold3\nprint('UNREACHABLE')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": _stubs.subprocess_pythonpath(HOME, str(site)), "OPENFOLD3_OPT": "exact"})
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "[openfold3-opt] NOT ACTIVE:" in r.stderr and "0.5.0" in r.stderr and "UNREACHABLE" not in r.stdout


def test_finder_fires_and_activates_with_the_pinned_stub(tmp_path):
    site = tmp_path / "site"
    (site / "openfold3").mkdir(parents=True)
    (site / "openfold3" / "__init__.py").write_text("BODY_RAN = True\n")
    di = site / "openfold3-0.4.1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: openfold3\nVersion: 0.4.1\n")
    code = ("import openfold3_opt._autoload\nimport openfold3\nimport openfold3_opt, os\nrep = openfold3_opt.status()\n"     # the .pth line, then the trigger
            "print(rep['active'], rep['mode'], rep['trigger'], os.environ.get('OF3_DETERMINISTIC'), openfold3.BODY_RAN)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": _stubs.subprocess_pythonpath(HOME, str(site)), "OPENFOLD3_OPT": "exact"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"True exact openfold3 {modes.LINES[('exact', modes.line_arg('exact'))].env.get('OF3_DETERMINISTIC')} True", r.stdout
    assert "[openfold3-opt] ACTIVE mode=exact" in r.stderr
    assert "[openfold3-opt] exit mode=exact" in r.stderr                   # the exit tally ran at interpreter exit


def _stub_site(tmp_path, version="0.5.0"):
    site = tmp_path / "site"
    (site / "openfold3").mkdir(parents=True)
    (site / "openfold3" / "__init__.py").write_text("BODY_RAN = True\n")
    di = site / f"openfold3-{version}.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: openfold3\nVersion: {version}\n")
    return site


def test_finder_survives_a_find_spec_probe(tmp_path):
    """importlib.util.find_spec('openfold3') before the real import must not disarm the hook: the import that follows still fires (exit 3 here: the stub pin refuses)."""
    site = _stub_site(tmp_path)
    code = ("import sys, importlib.util, openfold3_opt._autoload as a; assert a.FINDER is not None and a.FINDER.armed\n"
            "s = importlib.util.find_spec('openfold3'); assert s is not None and a.FINDER.armed, 'a probe disarmed the hook'\n"
            "import openfold3\nprint('UNREACHABLE')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": _stubs.subprocess_pythonpath(HOME, str(site)), "OPENFOLD3_OPT": "exact"})
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "[openfold3-opt] NOT ACTIVE:" in r.stderr and "UNREACHABLE" not in r.stdout


def test_unknown_selection_exits_3_at_interpreter_start(tmp_path):
    """OPENFOLD3_OPT=bogus: the .pth hook prints the NOT ACTIVE line and the interpreter exits 3 before any import — stock never runs under the variable."""
    site = _stub_site(tmp_path)
    for env_extra in ({"OPENFOLD3_OPT": "bogus"}, {"OPENFOLD3_OPT": "fast", "OPENFOLD3_OPT_MODE": "exact"}):
        r = subprocess.run([sys.executable, "-c", "import openfold3_opt._autoload\nimport openfold3\nprint('UNREACHABLE')\n"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": _stubs.subprocess_pythonpath(HOME, str(site)), **env_extra})
        assert r.returncode == 3, (env_extra, r.returncode, r.stderr)
        assert "[openfold3-opt] NOT ACTIVE:" in r.stderr and "UNREACHABLE" not in r.stdout


def test_every_prefixed_name_the_package_spells_is_declared_to_the_gate():
    """The .pth gate (`_autoload.install`) runs first in EVERY interpreter the kit's processes start — the tp launcher's ranks, their
    workers — and exits 3 on an OPENFOLD3_OPT_* name outside `_autoload.DECLARED`. So every name under the prefix the package's source
    spells (a switch it reads, a variable it hands a child: manifest.RECORDS_ENV, the tp launcher's records directory) must be declared: a
    name added anywhere in the package without extending DECLARED fails here, on CPU, not as dead ranks on a GPU box. `_autoload.py` itself
    is not scanned (its words quote a mistyped switch as the example of what the gate refuses); tests are not the package."""
    import re
    pkg = os.path.dirname(os.path.abspath(_autoload.__file__))
    spelled = {}
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
        for f in sorted(files):
            if f.endswith((".py", ".sh")) and f != "_autoload.py":
                with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                    for m in re.finditer(_autoload.ENV + r"_[A-Z0-9_]+", fh.read()):
                        spelled.setdefault(m.group(0), os.path.relpath(os.path.join(root, f), pkg))
    assert "OPENFOLD3_OPT_N_GPU" in spelled and "OPENFOLD3_OPT_RECORDS" in spelled            # the scan sees the package (modes.py's line parameter, manifest.RECORDS_ENV)
    undeclared = {k: v for k, v in spelled.items() if k not in _autoload.DECLARED}
    assert not undeclared, f"OPENFOLD3_OPT_* names the package spells but _autoload.DECLARED lacks (the .pth gate exits 3 on them in any child that inherits one): {undeclared}"


def test_autoload_mode_names_are_the_mode_table():
    """_autoload spells the mode names itself so the .pth imports nothing; the spelling is the mode table's."""
    from openfold3_opt import report
    assert set(_autoload.MODES) == set(modes.MODES) and len(_autoload.MODES) == len(modes.MODES)
    assert _autoload.ENV == "OPENFOLD3_OPT" and _autoload.EXIT_NOT_ACTIVE == 3 and _autoload.TAG == report.TAG
