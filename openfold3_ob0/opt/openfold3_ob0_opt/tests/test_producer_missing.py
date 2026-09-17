"""An older core beside this package (opt_core 0.3.x: no opt_core.mem.ngpu, no opt_core.arch) is refused BY NAME on every route —
`NOT ACTIVE: reason=producer_missing:<module>` exit 3 — never a traceback at the first ACTIVE line and never a one-GPU run that drops the
`--n_gpu` axis silently (_autoload.core_refusal = the one probe; modes.REQUIRED_PRODUCERS; stack.activate, __main__.main, _autoload.install, openfold3_ob0_opt.enable)."""
import os
import subprocess
import sys

import pytest

from openfold3_ob0_opt import modes, stack
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_the_table_names_the_core_modules_beyond_0_3():
    assert modes.REQUIRED_PRODUCERS == ("opt_core.mem.ngpu", "opt_core.arch") and modes.core_refusal() is None
    from openfold3_ob0_opt import _autoload
    assert tuple(m for m, _ in _autoload.REQUIRED_PRODUCER_FILES) == modes.REQUIRED_PRODUCERS          # the hook's file table is the modes table


def _fake_older_core(monkeypatch, tmp_path):
    """Point opt_core.__file__ at a directory without the producer files: what _autoload.core_refusal probes."""
    import opt_core
    d = tmp_path / "fakecore" / "opt_core"; d.mkdir(parents=True); (d / "__init__.py").write_text("")
    monkeypatch.setattr(opt_core, "__file__", str(d / "__init__.py"))


def test_activate_refuses_by_name_when_a_producer_is_missing(monkeypatch, tmp_path):
    _fake_older_core(monkeypatch, tmp_path)
    r = modes.core_refusal()
    assert r.startswith("producer_missing:opt_core.mem.ngpu,opt_core.arch — ") and "opt_core >= 0.4.0" in r and "[tool.opt_core]" in r, r
    for mode in ("fast", "exact", "big"):
        rep = stack.activate(mode, dry_run=True, home=HOME, environ={"OPENFOLD3_OB0_OPT_HOME": HOME}, log=False)
        assert rep["active"] is False and rep["reason"].startswith("producer_missing:opt_core.mem.ngpu"), rep
    with pytest.raises(stack.ActivationError, match="producer_missing:opt_core.mem.ngpu"):
        stack.activate("fast", dry_run=True, strict=True, home=HOME, environ={"OPENFOLD3_OB0_OPT_HOME": HOME}, log=False)


def _older_core(tmp_path):
    """A copy of the installed core WITHOUT the 0.4.0 producers (mem/ngpu.py, arch.py): the shape of an older core beside this package.
    __init__.py (hence __version__) is carried unchanged, so the pin gate passes on version alone and the PRODUCER probe is what refuses."""
    import shutil
    import opt_core
    src = os.path.dirname(os.path.abspath(opt_core.__file__))
    dst = tmp_path / "oldcore" / "opt_core"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "ngpu.py", "arch.py"))
    assert not (dst / "mem" / "ngpu.py").exists() and not (dst / "arch.py").exists() and (dst / "autoload.py").exists()
    return str(tmp_path / "oldcore")


def _env_with(core_dir):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OB0_OPT", "OF3", "PYTHONPATH"))}
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            # …/opt (openfold3_ob0_opt's parent)
    env["PYTHONPATH"] = core_dir + os.pathsep + pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OPENFOLD3_OB0_OPT_HOME"] = HOME
    return env


def test_cli_and_pth_routes_exit_3_with_the_line(tmp_path):
    env = _env_with(_older_core(tmp_path))
    r = subprocess.run([sys.executable, "-S", "-m", "openfold3_ob0_opt", "check", "--mode", "fast"], capture_output=True, text=True, env=env)   # -S: no site dir, so an editable
    # install's finder of the real core cannot serve opt_core.mem.ngpu behind the shadow (the shadow on PYTHONPATH is the only core)
    assert r.returncode == 3 and "[openfold3_ob0-opt] NOT ACTIVE: reason=producer_missing:opt_core.mem.ngpu,opt_core.arch" in r.stderr, (r.returncode, r.stderr[-600:])
    r = subprocess.run([sys.executable, "-S", "-c", "from openfold3_ob0_opt import _autoload; _autoload.install(environ={'OPENFOLD3_OB0_OPT': 'fast'})"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "NOT ACTIVE: reason=producer_missing:opt_core.mem.ngpu,opt_core.arch" in r.stderr and "OPENFOLD3_OB0_OPT='fast'" in r.stderr, (r.returncode, r.stderr[-600:])


def test_enable_and_status_refuse_by_name_under_an_older_core(tmp_path):
    """`import openfold3_ob0_opt; openfold3_ob0_opt.enable('fast')` beside an older core: the NOT ACTIVE report (active False, reason
    producer_missing:…) and the line on stderr — with strict=True an ActivationError by name — never a raw ModuleNotFoundError."""
    env = _env_with(_older_core(tmp_path))
    code = ("import openfold3_ob0_opt, sys\n"
            "rep = openfold3_ob0_opt.enable('fast')\n"
            "assert rep['active'] is False and rep['reason'].startswith('producer_missing:opt_core.mem.ngpu'), rep\n"
            "assert openfold3_ob0_opt.status()['reason'].startswith('producer_missing:')\n"
            "try:\n    openfold3_ob0_opt.enable('big', strict=True)\nexcept openfold3_ob0_opt.ActivationError as e:\n    print('refused:', e); sys.exit(3)\n"
            "sys.exit(0)\n")
    r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "[openfold3_ob0-opt] NOT ACTIVE: reason=producer_missing:opt_core.mem.ngpu,opt_core.arch" in r.stderr and "refused: producer_missing" in r.stdout, (r.returncode, r.stderr[-700:], r.stdout[-300:])


def test_enable_refuses_by_name_when_the_core_is_absent(tmp_path):
    env = _env_with(str(tmp_path / "no_core_here"))                                 # nothing importable as opt_core (-S: no site dir): the pin gate's core_missing
    r = subprocess.run([sys.executable, "-S", "-c", "import openfold3_ob0_opt, sys; rep = openfold3_ob0_opt.enable('fast'); sys.exit(3 if rep['reason'].startswith('core_missing:') else 0)"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "[openfold3_ob0-opt] NOT ACTIVE: reason=core_missing:" in r.stderr, (r.returncode, r.stderr[-600:])
