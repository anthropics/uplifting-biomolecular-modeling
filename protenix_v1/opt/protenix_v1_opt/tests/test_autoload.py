"""The .pth finder: installed only for PROTENIX_V1_OPT=exact|fast|big, fires once after the trigger package's body, calls enable(mode, strict=True,
trigger=..., det=..., allow_partial=...), removes itself; a refusal exits 3."""
import importlib
import os
import subprocess
import site
import shutil
import sys

import pytest

from protenix_v1_opt import _autoload as A

OPT = os.path.dirname(os.path.dirname(os.path.abspath(A.__file__)))   # opt/: the package's parent, on the subprocess's path


def _fake_family(tmp_path):
    (tmp_path / "runner").mkdir()
    (tmp_path / "runner" / "__init__.py").write_text("BODY_RAN = True\n")
    (tmp_path / "protenix").mkdir()
    (tmp_path / "protenix" / "__init__.py").write_text("")
    return str(tmp_path)


@pytest.fixture
def family(tmp_path, monkeypatch):
    root = _fake_family(tmp_path)
    monkeypatch.syspath_prepend(root)
    for m in ("runner", "protenix"):
        sys.modules.pop(m, None)
    yield root
    for m in ("runner", "protenix"):
        sys.modules.pop(m, None)
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, A.Finder)]


def test_install_only_for_a_kit_mode():
    assert A.install({}) is None
    assert A.install({"PROTENIX_V1_OPT": "off"}) is None
    seen = []
    assert A.install({"PROTENIX_V1_OPT": "turbo"}, exit=seen.append) is None and seen == [3]      # refused by name, exit 3 (below: the line)
    f = A.install({"PROTENIX_V1_OPT": "exact"})
    try:
        assert isinstance(f, A.Finder) and f.mode == "exact" and f.det is False and f in sys.meta_path
        assert A.install({"PROTENIX_V1_OPT": "exact"}) is f
    finally:
        sys.meta_path.remove(f)
    f = A.install({"PROTENIX_V1_OPT": "fast", "PROTENIX_V1_OPT_DET": "1"})
    try:
        assert f.det is True
    finally:
        sys.meta_path.remove(f)


def test_fires_after_the_trigger_body_then_removes_itself(family, monkeypatch):
    calls = []

    def fake_enable(mode, strict=False, trigger=None, det=False, allow_partial=False):
        calls.append((mode, strict, trigger, det, allow_partial, sys.modules["runner"].BODY_RAN))
        return {"active": True}
    import protenix_v1_opt
    monkeypatch.setattr(protenix_v1_opt, "enable", fake_enable)
    f = A.install({"PROTENIX_V1_OPT": "exact", "PROTENIX_V1_OPT_DET": "1", "PROTENIX_V1_OPT_ALLOW_PARTIAL": "1"})       # the route's two stand-ins, read at interpreter start
    importlib.import_module("runner")
    assert calls == [("exact", True, "runner", True, True, True)]
    assert f not in sys.meta_path and f.fired == "runner"


def test_a_runner_without_protenix_beside_it_is_not_a_trigger(tmp_path, monkeypatch):
    (tmp_path / "runner").mkdir()
    (tmp_path / "runner" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("runner", None)
    calls = []
    import protenix_v1_opt
    monkeypatch.setattr(protenix_v1_opt, "enable", lambda *a, **k: calls.append(a))
    f = A.install({"PROTENIX_V1_OPT": "exact"})
    try:
        importlib.import_module("runner")
        assert calls == [] and f.armed
    finally:
        sys.meta_path.remove(f)
        sys.modules.pop("runner", None)


def test_find_spec_probe_then_import_still_fires(family, monkeypatch):
    """A bare importlib.util.find_spec('<trigger>') probe (what a tool does before importing) must not disarm the hook: the real import
    that follows still fires enable() (the core's 0.2.1 finder disarmed on the probe — this kit keeps its own hook)."""
    calls = []
    import protenix_v1_opt
    monkeypatch.setattr(protenix_v1_opt, "enable", lambda mode, **kw: calls.append((mode, kw.get("trigger"))) or {"active": True})
    f = A.install({"PROTENIX_V1_OPT": "exact"})
    try:
        import importlib.util
        assert importlib.util.find_spec("runner") is not None and calls == [] and f in sys.meta_path and f.fired is None
        importlib.import_module("runner")
        assert calls == [("exact", "runner")] and f not in sys.meta_path and f.fired == "runner"
    finally:
        if f in sys.meta_path:
            sys.meta_path.remove(f)


def test_unknown_selection_exits_not_active_with_the_kits_line():
    """A mistyped selection never lets stock run: the kit's own NOT ACTIVE line, exit 3 at interpreter start (the .pth path)."""
    for value in ("turbo", "Exact ", "fast,exact", "1"):
        r = subprocess.run([sys.executable, "-c", "import protenix_v1_opt._autoload; print('STOCK WOULD RUN')"], capture_output=True, text=True,
                           env={**os.environ, "PROTENIX_V1_OPT": value, "PYTHONPATH": os.pathsep.join([OPT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])})
        if value.strip().lower() in ("exact", "fast", "big"):
            continue
        assert r.returncode == A.EXIT_NOT_ACTIVE == 3 and "STOCK WOULD RUN" not in r.stdout, (value, r.returncode, r.stdout, r.stderr[-300:])
        assert f"[protenix-v1-opt] NOT ACTIVE: unknown PROTENIX_V1_OPT={value.strip().lower()!r} (expected exact|fast|big|off); exit 3: stock never runs under a set PROTENIX_V1_OPT" in r.stderr


def test_undeclared_package_names_refuse_the_process():
    """A mistyped switch under the package prefix (PROTENIX_V1_OPT_MODE=exact, PROTENIX_V1_OPTS=fast) is refused by name at interpreter
    start, exit 3, whether or not PROTENIX_V1_OPT itself is set — never silently ignored."""
    seen = []
    assert A.install({"PROTENIX_V1_OPT_MODE": "exact"}, exit=seen.append) is None and seen == [3]
    seen.clear()
    assert A.install({"PROTENIX_V1_OPT": "exact", "PROTENIX_V1_OPTS": "fast", "PROTENIX_V1_OPT_DETT": "1"}, exit=seen.append) is None and seen == [3]
    for name in A.DECLARED:
        seen.clear()
        f = A.install({name: "off" if name == "PROTENIX_V1_OPT" else "x"}, exit=seen.append)
        assert f is None and seen == [], name                          # every declared name alone is not an undeclared one
    r = subprocess.run([sys.executable, "-c", "import protenix_v1_opt._autoload; print('STOCK WOULD RUN')"], capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT")}, "PROTENIX_V1_OPT_MODE": "exact", "PYTHONPATH": os.pathsep.join([OPT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])})
    assert r.returncode == 3 and "STOCK WOULD RUN" not in r.stdout
    assert "[protenix-v1-opt] NOT ACTIVE: undeclared PROTENIX_V1_OPT_MODE (the package reads " + ", ".join(A.DECLARED) + "); exit 3: a mistyped switch never runs stock" in r.stderr


def _user_site_enabled() -> bool:
    """Whether a child interpreter processes a user-site .pth (PYTHONNOUSERSITE unset, no -s): the real startup route exists."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONNOUSERSITE"}
    r = subprocess.run([sys.executable, "-c", "import site, sys; print(int(bool(site.ENABLE_USER_SITE) and not sys.flags.no_user_site))"], capture_output=True, text=True, env=env)
    return r.returncode == 0 and r.stdout.strip() == "1"


@pytest.mark.skipif(not _user_site_enabled(), reason="the interpreter disables the user site: no real .pth startup route to test")
def test_the_pth_refusal_on_the_real_startup_route(tmp_path):
    """The .pth line at REAL interpreter start (site.addpackage): a refusal must end the process with EXIT_NOT_ACTIVE and exactly the one
    line — never a Traceback and rc 1 (a SystemExit inside site's try/except is reported as a site error and stock would run); the form
    is os._exit at .pth time (_refuse_process). Installed here as a user-site .pth (PYTHONUSERBASE), the package findable from a path
    .pth sorted before it."""
    ub = tmp_path / "userbase"
    usersite = subprocess.run([sys.executable, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True, env={**os.environ, "PYTHONUSERBASE": str(ub)}).stdout.strip()
    os.makedirs(usersite, exist_ok=True)
    shutil.copy(os.path.join(OPT, "protenix_v1_opt_autoload.pth"), usersite)
    import opt_core
    with open(os.path.join(usersite, "0_protenix_v1_opt_path.pth"), "w") as fh:
        fh.write(OPT + "\n" + os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__))) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT") and k not in ("PYTHONPATH", "PYTHONNOUSERSITE")}
    env["PYTHONUSERBASE"] = str(ub)
    for extra, line in (({"PROTENIX_V1_OPT": "bogus"}, "[protenix-v1-opt] NOT ACTIVE: unknown PROTENIX_V1_OPT='bogus' (expected exact|fast|big|off); exit 3: stock never runs under a set PROTENIX_V1_OPT"),
                        ({"PROTENIX_V1_OPT_MODE": "exact"}, "[protenix-v1-opt] NOT ACTIVE: undeclared PROTENIX_V1_OPT_MODE (the package reads " + ", ".join(A.DECLARED) + "); exit 3: a mistyped switch never runs stock")):
        r = subprocess.run([sys.executable, "-c", "print('STOCK WOULD RUN')"], capture_output=True, text=True, env={**env, **extra}, timeout=60)
        assert r.returncode == A.EXIT_NOT_ACTIVE == 3 and r.stdout == "" and r.stderr.strip().splitlines() == [line], (extra, r.returncode, r.stdout, r.stderr[-400:])
    r = subprocess.run([sys.executable, "-c", "import sys; print('finder' if any(type(f).__name__ == 'Finder' for f in sys.meta_path) else 'none')"], capture_output=True, text=True, env={**env, "PROTENIX_V1_OPT": "exact"}, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "finder" and "Traceback" not in r.stderr, (r.returncode, r.stdout, r.stderr[-300:])


def test_refusal_exits_3(family, monkeypatch):
    import protenix_v1_opt

    def refuse(mode, **kw):
        raise protenix_v1_opt.ActivationError("gate")
    monkeypatch.setattr(protenix_v1_opt, "enable", refuse)
    A.install({"PROTENIX_V1_OPT": "fast"})
    with pytest.raises(SystemExit) as e:
        importlib.import_module("runner")
    assert e.value.code == 3


def test_unknown_env_value_refuses_the_process_by_name(capsys):
    """A PROTENIX_V1_OPT value that is not a mode never runs stock: the NOT ACTIVE line names the value and the process exits 3 at
    interpreter start (os._exit: a .pth line runs inside site's try/except, where a SystemExit would be swallowed and stock would run)."""
    seen = []
    A.install({"PROTENIX_V1_OPT": "turbo"}, exit=seen.append)
    err = capsys.readouterr().err
    assert seen == [A.EXIT_NOT_ACTIVE] == [3]
    assert "[protenix-v1-opt] NOT ACTIVE: unknown PROTENIX_V1_OPT='turbo' (expected exact|fast|big|off); exit 3: stock never runs under a set PROTENIX_V1_OPT" in err
    r = subprocess.run([sys.executable, "-c", "import protenix_v1_opt._autoload; print('STOCK WOULD RUN')"], capture_output=True, text=True,
                       env={**os.environ, "PROTENIX_V1_OPT": "turbo", "PYTHONPATH": os.pathsep.join([OPT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])})
    assert r.returncode == 3 and "STOCK WOULD RUN" not in r.stdout and "NOT ACTIVE: unknown PROTENIX_V1_OPT='turbo'" in r.stderr
