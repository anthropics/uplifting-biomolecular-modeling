"""The pin gate + producers gate on EVERY entry route. The shared core ABSENT (nothing importable as opt_core) and a STALE core PRESENT
(a shadow `opt_core` 0.2.5 whose MANIFEST is not the pinned tree) both end in ONE NOT ACTIVE line and exit 3 — the pin gate's words
(`_core_gate.gate`, the core's kit template carried byte-for-byte: `reason=core_missing:opt_core (…)` / `reason=core_mismatch: …`) — through
`python -m af3_torch_opt <verb>`, the console-script target (`__main__.main`), `bash run.sh <verb>` (+ `--config h100`: the exports probe),
the `.pth` hook (`_autoload` under AF3_TORCH_OPT) and the in-process interface (`af3_torch_opt.enable(...)`); never a raw
ModuleNotFoundError, never a traceback. The producers gate (`_producers`, statement two) names the modules an importable-but-incomplete
core lacks (`reason=producer_missing:<modules>`); REQUIRED_PRODUCERS lists every opt_core module the package sources import."""
import hashlib
import os
import re
import stat
import subprocess
import sys

from af3_torch_opt import _producers, report

from .conftest import HOME

OPT = os.path.join(HOME, "opt")
PKG = os.path.join(OPT, "af3_torch_opt")
GATE_HEAD = f"{report.PREFIX} NOT ACTIVE: reason="


def _stale_core(tmp_path, version="0.2.5"):
    """A directory holding an importable `opt_core` that is NOT the pinned floor: __init__.py names a version older than the pin, `report` /
    `gates` present (empty), no `opt_core.mem`, no `opt_core.arch` — first on the path it shadows any installed core."""
    d = tmp_path / "stale"; core = d / "opt_core"; core.mkdir(parents=True)
    (core / "__init__.py").write_text(f'__version__ = "{version}"\n')
    for m in ("report", "gates", "process", "home"):
        (core / f"{m}.py").write_text("# shadow\n")
    return str(d)


def _env(first_on_path=None, mode=None):
    """-S interpreters see only PYTHONPATH: the package (opt/) plus, when given, a shadow core FIRST; the installed core (a site .pth) is invisible."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_TORCH", "PYTHON"))}
    env["PYTHONPATH"] = os.pathsep.join([p for p in (first_on_path, OPT) if p])
    if mode:
        env["AF3_TORCH_OPT"] = mode
    return env


def _run(argv, env, cwd):
    return subprocess.run(argv, env=env, capture_output=True, text=True, cwd=str(cwd), timeout=120)


def _assert_refused(r, reason_prefix):
    assert r.returncode == 3, (r.returncode, r.stdout[-500:], r.stderr[-800:])
    assert "Traceback" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr[-800:]
    lines = [l for l in r.stderr.splitlines() if "NOT ACTIVE" in l]
    assert len(lines) == 1, r.stderr[-800:]
    assert lines[0].startswith(GATE_HEAD + reason_prefix), lines[0]
    return lines[0]


def test_gate_copy_is_the_cores_template():
    """opt/af3_torch_opt/_core_gate.py is byte-identical to common/opt_core/kit_template/_core_gate.py when the tree carries the core."""
    tpl = os.path.join(HOME, "..", "common", "opt_core", "kit_template", "_core_gate.py")
    mine = os.path.join(PKG, "_core_gate.py")
    assert os.path.isfile(mine)
    if os.path.isfile(tpl):
        assert hashlib.sha256(open(mine, "rb").read()).hexdigest() == hashlib.sha256(open(tpl, "rb").read()).hexdigest()


def test_every_entry_runs_the_gate_first():
    """`from ._core_gate import gate` + `gate(__file__ …)` precede any opt_core import in __main__.main, _autoload (armed) and the in-process
    interface (__init__.__getattr__); the producers gate follows it."""
    for name, must in (("__main__.py", "def main("), ("_autoload.py", "if _mode and _mode != \"off\":"), ("__init__.py", "def __getattr__(")):
        src = open(os.path.join(PKG, name), encoding="utf-8").read()
        body = src[src.index(must):]
        g = body.index("from ._core_gate import gate"); p = body.index("refuse_if_missing(")
        first_core = min([i for i in (body.find("import opt_core"), body.find("from opt_core"), body.find("from .modes import"), body.find("from .cli import"), body.find("importlib.import_module(")) if i >= 0] or [len(body)])
        assert g < p < first_core, (name, g, p, first_core)


def test_prefix_and_exit_code_are_the_package_pair():
    from opt_core.report import EXIT_NOT_ACTIVE
    from af3_torch_opt import _core_gate
    assert _producers.PREFIX == report.PREFIX and _producers.EXIT_NOT_ACTIVE == EXIT_NOT_ACTIVE == _core_gate.EXIT_NOT_ACTIVE == 3
    assert _producers.MIN_CORE == _core_gate.read_table(os.path.join(OPT, "pyproject.toml"), "tool.opt_core")["version"]   # one place the pinned version is written


def test_required_producers_cover_every_core_import_in_the_sources():
    """Every `opt_core[.x[.y]]` module a package source imports (import / from-import, at any indentation; the model-process scripts
    included: they import the same core from --opt-core) is in REQUIRED_PRODUCERS, and every listed name is a real module of the installed core."""
    pat = re.compile(r"^\s*(?:from\s+(opt_core(?:\.\w+)*)\s+import\s+([\w, ]+)|import\s+(opt_core(?:\.\w+)*))", re.M)
    found = set()
    for dirpath, _, files in os.walk(PKG):
        if os.sep + "tests" in dirpath[len(PKG):]:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            src = open(os.path.join(dirpath, f), encoding="utf-8").read()
            for m in pat.finditer(src):
                base = m.group(1) or m.group(3)
                found.add(base)
                if m.group(1):                                              # `from opt_core.mem import ngpu as X` names a submodule when it is one
                    for name in (n.strip().split(" as ")[0].strip() for n in m.group(2).split(",")):
                        cand = f"{base}.{name}"
                        if _importable(cand):
                            found.add(cand)
    missing = sorted(n for n in found if n not in _producers.REQUIRED_PRODUCERS)
    assert not missing, f"opt_core modules imported by the package but absent from _producers.REQUIRED_PRODUCERS: {missing}"
    unreal = [n for n in _producers.REQUIRED_PRODUCERS if not _importable(n)]
    assert not unreal, f"REQUIRED_PRODUCERS names that are not modules of the installed core: {unreal}"


def _importable(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def test_missing_producers_words():
    assert _producers.missing_producers(("opt_core", "opt_core.report")) == []
    assert _producers.missing_producers(("opt_core", "opt_core.no_such_module_xyz")) == ["opt_core.no_such_module_xyz"]
    assert _producers.refusal_line(["opt_core"], "big").startswith(f"{report.PREFIX} NOT ACTIVE mode=big reason=core_missing:opt_core — ")
    assert _producers.refusal_line(["opt_core.mem", "opt_core.mem.ngpu"]).startswith(f"{report.PREFIX} NOT ACTIVE mode=none reason=producer_missing:opt_core.mem,opt_core.mem.ngpu — ")
    assert _producers.mode_word(["x", "--mode", "big"], {}) == "big" and _producers.mode_word(["x", "--mode=off"], {}) == "off"
    assert _producers.mode_word(["x"], {"AF3_TORCH_OPT": "fast"}) == "fast" and _producers.mode_word(["x"], {}) == "none"


ROUTES = lambda tmp: ((  # noqa: E731 — (argv, needs the mode variable)
    ([sys.executable, "-S", "-m", "af3_torch_opt", "check", "--mode", "fast"], None),
    ([sys.executable, "-S", "-m", "af3_torch_opt", "pred", "--mode", "big", "--n_gpu", "2", "--json_path", "x.json", "--output_dir", str(tmp)], None),
    ([sys.executable, "-S", "-m", "af3_torch_opt", "exports"], None),                                                                   # configs/<gpu>.env's probe
    ([sys.executable, "-S", "-c", "import sys; from af3_torch_opt.__main__ import main; sys.exit(main(['check', '--mode', 'fast']))"], None),   # the console script's target
    ([sys.executable, "-S", "-c", "import af3_torch_opt._autoload; print('reached')"], "fast"),                                          # the .pth hook, armed
    ([sys.executable, "-S", "-c", "import af3_torch_opt as p; p.enable('fast'); print('reached')"], None),                               # the in-process interface
    ([sys.executable, "-S", "-c", "import af3_torch_opt as p; p.check('off')"], None),
))


def test_stale_core_is_refused_by_name_on_every_route(tmp_path):
    stale = _stale_core(tmp_path)
    for argv, mode in ROUTES(tmp_path):
        r = _run(argv, _env(stale, mode), tmp_path)
        line = _assert_refused(r, "core_mismatch: opt_core pinned ")
        assert "installed v0.2.5 at " in line and str(tmp_path) in line, line
        assert "reached" not in r.stdout
    r = _run([sys.executable, "-S", "-c", "import af3_torch_opt as p; print(p.__version__, p.VARIANTS)"], _env(stale), tmp_path)      # facts that need no core stay readable
    assert r.returncode == 0 and r.stdout.strip(), r


def test_absent_core_is_refused_by_name_on_every_route(tmp_path):
    for argv, mode in ROUTES(tmp_path):
        r = _run(argv, _env(None, mode), tmp_path)
        line = _assert_refused(r, "core_missing:opt_core (pinned >= v")
        assert "nothing importable as opt_core on sys.path" in line, line
        assert "reached" not in r.stdout


RUN = os.path.join(HOME, "run.sh")


def _real_python_on_path(tmp_path, first_on_path):
    """`python` on PATH = this interpreter under -S with PYTHONPATH = [shadow,] opt/ (run.sh runs `python`)."""
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    py = b / "python"
    py.write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n"); py.chmod(py.stat().st_mode | stat.S_IXUSR)
    env = _env(first_on_path)
    env["PATH"] = f"{b}:{os.environ['PATH']}"
    return env


def test_run_sh_routes_refuse_by_name(tmp_path):
    """`bash run.sh check --mode fast` with a stale / absent core: the install probe passes (the package imports nothing of the core at
    import), the pins probe is the frozen stdlib script, the exec'd `python -m af3_torch_opt check` prints the gate's line, rc 3. With
    `--config h100` the config's exports probe (`python -m af3_torch_opt exports`) is the first route to refuse: the same line, rc 3, and
    run.sh stops there (the config line names it: `configs/h100.env: refused (rc 3)`)."""
    for label, reason in (("stale", "core_mismatch: opt_core pinned "), ("absent", "core_missing:opt_core (pinned >= v")):
        sub = tmp_path / label; sub.mkdir()
        env = _real_python_on_path(sub, _stale_core(sub) if label == "stale" else None)
        r = subprocess.run(["bash", RUN, "check", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=str(sub), timeout=120)
        if "pins not met" in r.stderr:                                       # a box without the image's interpreters: the frozen pins probe refuses first (rc 3) — the route
            assert r.returncode == 3                                         # under test is then covered by the --config form below, whose probe runs before the pins
        else:
            _assert_refused(r, reason)
        r = subprocess.run(["bash", RUN, "check", "--config", "h100", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=str(sub), timeout=120)
        assert r.returncode == 3 and "configs/h100.env: refused (rc 3)" in r.stderr, (r.returncode, r.stderr[-600:])
        _assert_refused(r, reason)
