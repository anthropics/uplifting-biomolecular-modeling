"""The .pth route: nothing installed for off/unset, an unknown mode refused with a line, the finder fires BEFORE the trigger's body."""
import io
import os
import sys

import pytest

from .. import _autoload


def _drop_finders():
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            sys.meta_path.remove(f)


def test_off_or_unset_installs_nothing():
    _drop_finders()
    assert _autoload.install({}) is None
    assert _autoload.install({"ROSETTAFOLD3_OPT": "off"}) is None
    assert _autoload.install({"ROSETTAFOLD3_OPT": "  "}) is None
    assert not any(isinstance(f, _autoload.Finder) for f in sys.meta_path)


def test_unknown_mode_prints_its_line_and_exits_3(monkeypatch):
    """A mistyped selection under the variable never runs stock silently: the NOT ACTIVE line, then exit 3 at interpreter start."""
    _drop_finders()
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    stops = []
    monkeypatch.setattr(_autoload, "_stop", lambda: stops.append(3) or (_ for _ in ()).throw(SystemExit(3)))   # os._exit stubbed: the test process must survive
    with pytest.raises(SystemExit):
        _autoload.install({"ROSETTAFOLD3_OPT": "faster"})
    assert stops == [3]
    assert "[rosettafold3-opt] NOT ACTIVE: unknown ROSETTAFOLD3_OPT='faster' (expected off|exact|fast|big)" in buf.getvalue()
    assert not any(isinstance(f, _autoload.Finder) for f in sys.meta_path)


def test_undeclared_name_refused_by_the_hook(monkeypatch):
    """A mistyped ROSETTAFOLD3_OPT_* name is refused by the hook with or without a mode: the line, then exit 3."""
    _drop_finders()
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    stops = []
    monkeypatch.setattr(_autoload, "_stop", lambda: stops.append(3) or (_ for _ in ()).throw(SystemExit(3)))
    with pytest.raises(SystemExit):
        _autoload.install({"ROSETTAFOLD3_OPT_MODE": "exact"})
    assert stops == [3] and "[rosettafold3-opt] NOT ACTIVE: undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (mistyped? this tree reads ROSETTAFOLD3_OPT_STOCK_PYTHON, " in buf.getvalue()
    assert "ROSETTAFOLD3_OPT_MODE='exact'" in buf.getvalue()
    assert _autoload.install({"ROSETTAFOLD3_OPT_CKPT": "/w", "ROSETTAFOLD3_OPT": "off"}) is None and stops == [3]   # declared names pass
    assert _autoload.undeclared({"ROSETTAFOLD3_OPTS": "x"}) == {}                                                   # not under the prefix: not this rule


def _pth_run(tmp_path, env, form, trigger=False, extra_path=None, preamble=""):
    """Run `python -c pass` with the kit's .pth line processed the way site does — `startup`: the real interpreter start (PYTHONUSERBASE
    user site dir; needs a user-site-enabled interpreter), `addsitedir`: site.addsitedir on the same .pth file (the same addpackage code
    path, driven explicitly). Returns (rc, stderr)."""
    import subprocess
    from .. import stack
    base = tmp_path / "userbase"
    probe = subprocess.run([sys.executable, "-c", "import site; print(site.getusersitepackages()); print(site.ENABLE_USER_SITE)"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONUSERBASE": str(base)})
    site_dir, enabled = probe.stdout.split()[0], probe.stdout.split()[1] == "True"
    os.makedirs(site_dir, exist_ok=True)
    import shutil
    shutil.copyfile(os.path.join(stack.opt_root(), "rosettafold3_opt_autoload.pth"), os.path.join(site_dir, "rosettafold3_opt_autoload.pth"))   # the kit's own .pth bytes
    clean = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT")}
    env_ = {**clean, **env, "PYTHONUSERBASE": str(base), "PYTHONPATH": stack.opt_root() + (os.pathsep + extra_path if extra_path else ""), "PYTHONDONTWRITEBYTECODE": "1"}
    body = "import rf3.graph_flags" if trigger else "pass"                                 # trigger: the hook fires on the real import
    if form == "startup":
        if not enabled:
            pytest.skip("this interpreter disables the user site dir (a venv without --system-site-packages): the startup form runs on the box's interpreter")
        cmd = [sys.executable, "-c", body]
    else:
        cmd = [sys.executable, "-S", "-c", f"{preamble}import site; site.addsitedir({site_dir!r}); {body}"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env_, timeout=120)
    return r.returncode, r.stderr


@pytest.mark.parametrize("form", ["addsitedir", "startup"])
def test_pth_refusals_exit_3_with_one_line(tmp_path, form):
    """Through the .pth: a mistyped selection and an undeclared name each exit 3 with exactly the NOT ACTIVE line (no traceback, no fatal
    init error); a mode installs the hook and the process runs; unset/off run untouched."""
    rc, err = _pth_run(tmp_path, {"ROSETTAFOLD3_OPT": "bogus"}, form)
    assert rc == 3 and err == "[rosettafold3-opt] NOT ACTIVE: unknown ROSETTAFOLD3_OPT='bogus' (expected off|exact|fast|big)\n", (rc, err)
    rc, err = _pth_run(tmp_path, {"ROSETTAFOLD3_OPT_MODE": "exact"}, form)
    assert rc == 3 and err.startswith("[rosettafold3-opt] NOT ACTIVE: undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment") and err.count("\n") == 1 and "Traceback" not in err, (rc, err)
    for env in ({"ROSETTAFOLD3_OPT": "exact"}, {"ROSETTAFOLD3_OPT": "off"}, {}):
        rc, err = _pth_run(tmp_path, env, form)
        assert (rc, err) == (0, ""), (env, rc, err)


@pytest.mark.parametrize("form", ["addsitedir", "startup"])
def test_pth_trigger_time_refusal_exits_3_with_one_line(tmp_path, form):
    """A gate refusal at the TRIGGER import (here: a switch already in the environment contradicting the row, RF3_HOIST=0 under exact,
    on a stub patched tree) is the NOT ACTIVE line and exit 3 — never a traceback out of the import, never an exported row."""
    from . import _stubs
    root = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    _stubs.make_dist(root, route="vcs")
    rc, err = _pth_run(tmp_path, {"ROSETTAFOLD3_OPT": "exact", "RF3_HOIST": "0"}, form, trigger=True, extra_path=root)
    assert rc == 3 and err == "[rosettafold3-opt] NOT ACTIVE: environment contradicts row exact: RF3_HOIST='0' (row says '1')\n", (rc, err)


def test_finder_idempotent_and_fires_before_the_body(monkeypatch, tmp_path):
    _drop_finders()
    f = _autoload.install({"ROSETTAFOLD3_OPT": "exact"})
    assert f is not None and _autoload.install({"ROSETTAFOLD3_OPT": "exact"}) is f
    events = []
    import rosettafold3_opt

    def fake_enable(mode, strict=False, trigger=None):
        events.append(("enable", mode, trigger))
        return {"active": True}
    monkeypatch.setattr(rosettafold3_opt, "enable", fake_enable)
    pkg = tmp_path / "rf3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import sys\nsys.modules['_rf3_events'].append('body')\n")
    sys.modules["_rf3_events"] = events
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.invalidate_caches()
    importlib.import_module("rf3")
    assert events[0] == ("enable", "exact", "rf3") and events[1] == "body"
    assert f.fired == "rf3" and f not in sys.meta_path
    sys.modules.pop("_rf3_events", None)
    sys.modules.pop("rf3", None)


def test_a_find_spec_probe_leaves_the_finder_armed(monkeypatch, tmp_path):
    """importlib.util.find_spec('rf3') before the first import (the usual "is rf3 installed?" probe) must not disarm the hook."""
    _drop_finders()
    f = _autoload.install({"ROSETTAFOLD3_OPT": "exact"})
    events = []
    import rosettafold3_opt

    def fake_enable(mode, strict=False, trigger=None):
        events.append(("enable", mode, trigger))
        return {"active": True}
    monkeypatch.setattr(rosettafold3_opt, "enable", fake_enable)
    pkg = tmp_path / "rf3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import sys\nsys.modules['_rf3_events'].append('body')\n")
    sys.modules["_rf3_events"] = events
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    import importlib.util
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("rf3")
    assert spec is not None and f.armed and f.fired is None and f in sys.meta_path and events == []
    importlib.import_module("rf3")
    assert events == [("enable", "exact", "rf3"), "body"]
    assert f.fired == "rf3" and not f.armed and f not in sys.meta_path
    importlib.reload(sys.modules["rf3"])                         # the wrapped loader runs the body again, the hook never twice
    assert events == [("enable", "exact", "rf3"), "body", "body"]
    sys.modules.pop("_rf3_events", None)
    sys.modules.pop("rf3", None)


def test_autoload_names_are_the_mode_table(monkeypatch):
    """_autoload.py repeats modes.py's names (nothing may be imported at interpreter start): locked here, one table."""
    from .. import modes, stack
    assert _autoload.MODES == modes.MODES
    assert _autoload.ENV == modes.ENV
    assert _autoload.ENV_NAMES == stack.ENV_NAMES                    # the declared ROSETTAFOLD3_OPT_* names: one table, restated here
    assert open(_autoload.__file__).read().count("os._exit(3)") == 1 and "sys.exit(" not in open(_autoload.__file__).read()   # the one exit, the .pth-safe form (checked by reading the source itself)


def test_finder_exits_3_when_refused(monkeypatch, tmp_path):
    _drop_finders()
    f = _autoload.install({"ROSETTAFOLD3_OPT": "exact"})
    import rosettafold3_opt

    def refuse(mode, strict=False, trigger=None):
        assert mode == "exact"
        raise rosettafold3_opt.ActivationError("no")
    monkeypatch.setattr(rosettafold3_opt, "enable", refuse)
    stops = []
    monkeypatch.setattr(_autoload, "_stop", lambda: stops.append(3) or (_ for _ in ()).throw(SystemExit(3)))   # os._exit stubbed: the test process must survive
    pkg = tmp_path / "rf3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.invalidate_caches()
    with pytest.raises(SystemExit):
        importlib.import_module("rf3")
    assert stops == [3]                                              # the real _stop: the one os._exit(3) (test_pth_refusals_exit_3_with_one_line)
    _drop_finders()


def test_pth_text_is_the_backends_and_imports_the_hook():
    """The shipped .pth is the build backend's generated text for this package (opt/_build_backend.py, the core's template: its
    pth_fields parse it back to this package / variable / tag) and its one import is the hook module."""
    import importlib.util
    from .. import stack
    text = open(os.path.join(stack.opt_root(), "rosettafold3_opt_autoload.pth")).read()
    assert "import rosettafold3_opt._autoload" in text and text.endswith("\n")
    spec = importlib.util.spec_from_file_location("_kit_build_backend", os.path.join(stack.opt_root(), "_build_backend.py"))
    bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
    assert bb.PTH == "rosettafold3_opt_autoload.pth"
    assert bb.pth_fields(text) == ("rosettafold3_opt", "ROSETTAFOLD3_OPT", "rosettafold3-opt", 3)   # the guarded form: header + one guarded import line
    assert text == bb.pth_text("rosettafold3_opt", "ROSETTAFOLD3_OPT", "rosettafold3-opt")            # byte for byte the backend's output


_BLOCK_CORE = ("import sys\n"
               "class _NoCore:\n"
               "    def find_spec(self, name, path=None, target=None):\n"
               "        if name == 'opt_core' or name.startswith('opt_core.'):\n"
               "            raise ModuleNotFoundError(f'No module named {name!r} (blocked for the test)', name=name)\n"
               "        return None\n"
               "sys.meta_path.insert(0, _NoCore()); [sys.modules.pop(m) for m in list(sys.modules) if m == 'opt_core' or m.startswith('opt_core.')]\n")


def test_pth_core_missing_at_the_trigger_exits_3_with_one_line(tmp_path):
    """The pinned core not importable at activation (neither installed nor at the pin's path — simulated by an import blocker) is ONE
    NOT ACTIVE line naming it (reason=core_missing:<module>) and exit 3: never a traceback out of the trigger import, never stock silently."""
    from . import _stubs
    root = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    _stubs.make_dist(root, route="vcs")
    rc, err = _pth_run(tmp_path, {"ROSETTAFOLD3_OPT": "exact"}, "addsitedir", trigger=True, extra_path=root, preamble=_BLOCK_CORE)
    assert rc == 3 and err.count("\n") == 1 and "Traceback" not in err, (rc, err)
    assert err.startswith("[rosettafold3-opt] NOT ACTIVE: reason=core_missing:opt_core"), err


def test_cli_core_missing_exits_3_with_one_line():
    """``python -m rosettafold3_opt <command>`` with the core not importable: the same line, exit 3 (cli.EXIT_NOT_ACTIVE), no traceback."""
    import subprocess
    from .. import stack
    clean = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT")}
    code = _BLOCK_CORE + "import runpy, sys\nsys.argv = ['rosettafold3_opt', 'check', '--mode', 'exact']\nrunpy.run_module('rosettafold3_opt', run_name='__main__')\n"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**clean, "PYTHONPATH": stack.opt_root(), "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 3 and r.stderr.count("\n") == 1 and r.stderr.startswith("[rosettafold3-opt] NOT ACTIVE: reason=core_missing:opt_core"), (r.returncode, r.stderr[-600:])


def test_not_active_reason_words():
    from .. import _core
    assert _core.not_active_reason(ModuleNotFoundError("No module named 'opt_core.kernels'", name="opt_core.kernels")).startswith("reason=core_missing:opt_core.kernels (")
    assert _core.not_active_reason(_core.CoreError("opt_core is not installed on X")).startswith("reason=core_missing:opt_core (")
    assert _core.not_active_reason(ModuleNotFoundError("No module named 'einops'", name="einops")).startswith("reason=import_error:einops (")
    assert _core.not_active_reason(RuntimeError("boom")) == "reason=activation_error:RuntimeError (boom)"
