"""An absent, stale or older shared core (opt_core) is refused BY NAME with rc 3 on every entry route before anything of the core is
imported: statement one is the core's template gate (`_core_gate.gate`: the kit's [tool.opt_core] pin vs the opt_core the interpreter
would import, located not imported — `core_missing:opt_core` / `core_mismatch: …`), statement two the package's producer list
(`_producers`: `producer_missing:<modules>` for a core whose MANIFEST matches the pin but lacks a module this package imports). Routes:
`python -m opendde_opt <verb>` (= the console script `opendde-opt`), `bash run.sh check`, the `.pth` hook under a kit selection, the
in-process `opendde_opt.enable()`. The child interpreter sees an absent core through a meta-path blocker (sitecustomize first on
PYTHONPATH), a stale / older core through a SHADOW copy of the installed core.

The core pin and the carried template: opt/pyproject.toml [tool.opt_core] names the opt_core this kit runs on, by path and by a
minimum version; opt/_build_backend.py is the core's template byte-for-byte.

The entry scripts' package probe passes the package's own refusal through: `run.sh` and `configs/h100.env` run `python -m opendde_opt help`
first; whatever that interpreter writes to stderr is shown verbatim and its exit code is the script's; the scripts name 'not installed'
themselves only for a genuine ModuleNotFoundError of the package.

Fail-loud refusals: every `NOT ACTIVE` sentence the package prints carries `reason=<name>` — the activation line (report.activation_line)
right after its mode / line fields, the autoload gate's own sentences (_autoload._refuse), the CLI's refusals; a report that reaches the
printer without a reason prints `reason=unspecified (defect: …)`, never nothing."""
import io
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tokenize

import pytest

from opt_core import gates

from opendde_opt import _autoload, _core_gate, _producers, modes, registry, report, stack
from opendde_opt.tests import _stubs


KIT = os.path.abspath(os.path.join(_stubs.TREE))
STALE_VERSION = "0.2.5"                                     # older than any real pin: the floor always refuses it
GATE_LINE_ABSENT = "[opendde-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v"
GATE_LINE_MISMATCH = "[opendde-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned "


def _env(first=()):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENDDE_OPT", "ODDE_", "MODEL_OPT_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in first] + [os.path.join(KIT, "opt")] + [p for p in sys.path if p])
    env["MODEL_OPT"] = KIT
    return env


def _absent_core_env(tmp_path, site=None):
    """The child cannot resolve `opt_core` at all: a meta-path blocker installed by a sitecustomize first on PYTHONPATH."""
    d = tmp_path / "blk"; d.mkdir(exist_ok=True)
    (d / "sitecustomize.py").write_text(textwrap.dedent("""
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "opt_core" or name.startswith("opt_core."):
                    raise ModuleNotFoundError(f"blocked for the test: {name}", name=name)
                return None
        sys.meta_path.insert(0, _Block())
    """))
    return _env([d] + ([site] if site else []))


def _shadow_core_env(tmp_path, without=(), version=None):
    """The child resolves `opt_core` to a SHADOW copy of the installed core: `without` modules removed, `version` overridden in the
    copy's `__init__.py` (an older string simulates a stale core — the gate's core_mismatch) or left as installed (the gate passes,
    and the producer list judges which modules are present)."""
    import opt_core
    src = os.path.dirname(opt_core.__file__)
    root = tmp_path / "shadow"; root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, root / "opt_core", ignore=shutil.ignore_patterns("__pycache__"))
    if version is not None:
        init = root / "opt_core" / "__init__.py"
        txt = init.read_text()
        patched = re.sub(r'__version__\s*=\s*"[^"]*"', '__version__ = "%s"' % version, txt, count=1)
        assert patched != txt, "opt_core/__init__.py carries no __version__ literal to override"
        init.write_text(patched)
    for name in without:
        p = (root / "opt_core").joinpath(*name.split(".")[1:])
        if p.with_suffix(".py").is_file():
            p.with_suffix(".py").unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    return _env([root])


def _gate_passes_here():
    import io
    try:
        _core_gate.gate(os.path.join(KIT, "opt", "opendde_opt", "__init__.py"), "opendde-opt", stream=io.StringIO())
    except _core_gate.CoreGateRefused as e:
        return e.line
    return None


needs_matching_core = pytest.mark.skipif(_gate_passes_here() is not None, reason=f"the kit's pin does not match the opt_core installed here ({_gate_passes_here()}): "
                                                                              "the producer-list words sit behind the gate")


def test_declared_producers_are_importable_here():
    if _gate_passes_here() or _producers.missing_producers():
        pytest.skip(f"the installed core is not the pinned one here ({_gate_passes_here() or _producers.refusal()})")
    assert _producers.refusal() is None
    assert "opt_core.mem.ngpu" in _producers.REQUIRED_PRODUCERS and "opt_core.mem.rowpair" in _producers.REQUIRED_PRODUCERS


def test_python_m_route_refuses_an_absent_core_by_name(tmp_path):
    r = subprocess.run([sys.executable, "-m", "opendde_opt", "check", "--mode", "fast"], env=_absent_core_env(tmp_path), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_ABSENT in r.stderr and "Traceback" not in r.stderr, r


def test_python_m_route_refuses_a_stale_core_by_name(tmp_path):
    """A core older than the kit's pinned floor is not the pinned one: core_mismatch, exit 3."""
    r = subprocess.run([sys.executable, "-m", "opendde_opt", "pred", "--mode", "big", "--n_gpu", "2", "-i", "q.json", "-o", "o"],
                       env=_shadow_core_env(tmp_path, version=STALE_VERSION), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_MISMATCH in r.stderr and "Traceback" not in r.stderr, r


@needs_matching_core
def test_python_m_route_refuses_an_older_core_by_name(tmp_path):
    """A core whose version is the pinned one but which lacks modules this package imports: producer_missing, exit 3."""
    r = subprocess.run([sys.executable, "-m", "opendde_opt", "pred", "--mode", "big", "--n_gpu", "2", "-i", "q.json", "-o", "o"],
                       env=_shadow_core_env(tmp_path, ("opt_core.mem.ngpu", "opt_core.mem.rowpair")), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3, r
    assert "NOT ACTIVE reason=producer_missing:opt_core.mem.ngpu,opt_core.mem.rowpair," in r.stderr and f"opt_core >= {_producers.MIN_CORE}" in r.stderr and "Traceback" not in r.stderr


def test_run_sh_check_refuses_an_absent_and_a_stale_core_by_name(tmp_path):
    for env, line in ((_absent_core_env(tmp_path), GATE_LINE_ABSENT), (_shadow_core_env(tmp_path / "s", version=STALE_VERSION), GATE_LINE_MISMATCH)):
        r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 3 and line in r.stderr and "Traceback" not in r.stderr, r


def test_config_env_refuses_an_absent_core_by_name(tmp_path):
    """`source configs/h100.env` runs the real entry before any probe: the gate's line, return status 3, MODEL_OPT_STACK_KEY never `unknown`."""
    r = subprocess.run(["bash", "-c", f"source {os.path.join(KIT, 'configs', 'h100.env')}; echo AFTER rc=$?"], env=_absent_core_env(tmp_path),
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert "AFTER rc=3" in r.stdout and GATE_LINE_ABSENT in r.stderr and "Traceback" not in r.stderr, r


def test_pth_route_refuses_an_absent_core_by_name_under_a_selection(tmp_path):
    """The `.pth` imports `opendde_opt._autoload` at interpreter start; under `OPENDDE_OPT=<mode>` its finder activates when upstream's
    `runner` is imported — with the core absent that activation is the gate's NOT ACTIVE line and exit 3, never a stock run under the
    variable. `-S`: no site processing, so no installed `.pth` ran before the blocker; then the hook's import exactly as the `.pth` line does it."""
    env = _absent_core_env(tmp_path, site=_stubs.make_site(str(tmp_path))); env["OPENDDE_OPT"] = "fast"
    r = subprocess.run([sys.executable, "-S", "-c", "import sitecustomize, opendde_opt._autoload, runner"], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_ABSENT in r.stderr and "Traceback" not in r.stderr, r


def test_in_process_enable_refuses_an_absent_a_stale_and_an_older_core_by_name(tmp_path):
    """`opendde_opt.enable(mode)`: the gate is statement one (SystemExit 3 with its line on an absent / stale core); a core of the pinned
    version that lacks specific modules returns an inactive report naming producer_missing (strict=True raises ActivationError)."""
    code = "import opendde_opt; rep = opendde_opt.enable('fast'); print('ACTIVE' if rep.get('active') else 'INACTIVE:' + str(rep.get('reason')))"
    r = subprocess.run([sys.executable, "-S", "-c", "import sitecustomize; " + code], env=_absent_core_env(tmp_path), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_ABSENT in r.stderr and "Traceback" not in r.stderr, r
    r = subprocess.run([sys.executable, "-c", code], env=_shadow_core_env(tmp_path / "st", version=STALE_VERSION), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_MISMATCH in r.stderr and "Traceback" not in r.stderr, r
    if _gate_passes_here() is None:
        r = subprocess.run([sys.executable, "-c", code], env=_shadow_core_env(tmp_path / "old", ("opt_core.mem.ngpu", "opt_core.mem.rowpair")),
                           capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 0 and "INACTIVE:reason=producer_missing:opt_core.mem.ngpu,opt_core.mem.rowpair" in r.stdout and "Traceback" not in r.stderr, r


def test_console_script_is_the_module_entry():
    """`opendde-opt` = `opendde_opt.__main__:main` (pyproject [project.scripts]): the gate above covers it by construction."""
    txt = open(os.path.join(KIT, "opt", "pyproject.toml")).read()
    assert 'opendde-opt = "opendde_opt.__main__:main"' in txt


def _imported_opt_core_modules() -> set:
    """Every ``opt_core`` module the package's sources import: ``import opt_core.x.y``, ``from opt_core.x import y`` (``y`` counted as a module
    when ``opt_core/x/y.py`` or ``opt_core/x/y/`` exists in the core of this tree, else it is a name inside ``opt_core.x``)."""
    import ast
    import opt_core
    core_root = os.path.dirname(os.path.dirname(opt_core.__file__))
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def is_module(dotted: str) -> bool:
        rel = os.path.join(core_root, *dotted.split("."))
        return os.path.isfile(rel + ".py") or os.path.isfile(os.path.join(rel, "__init__.py"))
    found = set()
    for dp, _dn, fn in os.walk(pkg_dir):
        if os.sep + "tests" in dp[len(pkg_dir):]:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(dp, f), encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name == "opt_core" or a.name.startswith("opt_core."):
                            found.add(a.name)
                elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "opt_core" or node.module.startswith("opt_core.")):
                    found.add(node.module)
                    for a in node.names:
                        if is_module(node.module + "." + a.name):
                            found.add(node.module + "." + a.name)
    return found


def test_required_producers_cover_every_opt_core_module_the_sources_import():
    """REQUIRED_PRODUCERS ⊇ the opt_core modules the package imports (derived from the sources, so the list cannot go stale): a core with the
    right version string but a missing module is refused BY NAME (producer_missing:…), never a raw ImportError inside a rank."""
    import opt_core
    have = tuple(int(x) for x in opt_core.__version__.split(".")[:3])
    need = tuple(int(x) for x in _producers.MIN_CORE.split(".")[:3])
    if have < need:
        pytest.skip(f"the importable core is {opt_core.__version__} < MIN_CORE {_producers.MIN_CORE}: submodule names are classified against the pinned core's files")
    imported = _imported_opt_core_modules()
    assert "opt_core.mem.rowpair.pairstack" in imported and "opt_core.mem.rowpair.diffusion" in imported, sorted(imported)
    missing = sorted(imported - set(_producers.REQUIRED_PRODUCERS))
    assert missing == [], missing


OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_core_pin_is_a_floor_the_imported_core_meets():
    g = gates.core_pin_check(os.path.join(OPT, "pyproject.toml"))
    assert g.ok, g.reason
    pin = gates.core_pin(os.path.join(OPT, "pyproject.toml"))
    from opendde_opt._core_gate import version_tuple
    have = gates.imported_core()["version"]
    assert version_tuple(have) >= version_tuple(pin["version"]), (have, pin["version"])   # the pin is a floor (_core_gate refuses only an OLDER core), never an equality
    assert pin["path"] == "../../common/opt_core"


def _run(args, env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "OPENDDE_OPT")}
    env.update(env_extra, PYTHONPATH=OPT)                                                   # the package importable, the core NOT (no common/opt_core entry)
    return subprocess.run([sys.executable, "-I"] + args, env=env, capture_output=True, text=True, cwd=OPT)


def _core_hidden():
    r = _run(["-c", "import sys; sys.path.insert(0, %r); import opt_core" % OPT], {})
    return r.returncode != 0                                                                  # True when opt_core is NOT importable from opt/ alone


def test_cli_route_core_missing_is_exit_3_named():
    if not _core_hidden():
        import pytest; pytest.skip("opt_core is importable without common/opt_core on the path (installed in this environment)")
    r = _run(["-c", "import sys; sys.path.insert(0, %r); from opendde_opt.__main__ import main; sys.exit(main(['pred', '--mode', 'fast', '-i', 'x.json', '-o', 'o']))" % OPT], {})
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])
    assert "[opendde-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr


def test_env_route_core_missing_is_exit_3_named():
    if not _core_hidden():
        import pytest; pytest.skip("opt_core is importable without common/opt_core on the path (installed in this environment)")
    r = _run(["-c", "import sys; sys.path.insert(0, %r); import opendde_opt._autoload" % OPT], {"OPENDDE_OPT": "fast"})
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])
    assert "NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr


NOT_INSTALLED = "is not installed on"
OLD_WORDS = ("not usable on", "pinned upstream wheel is not installed")


def _absent_package_env(tmp_path):
    """The child cannot resolve `opendde_opt` at all (as on an interpreter where the package was never installed)."""
    d = tmp_path / "nopkg"; d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(textwrap.dedent("""
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "opendde_opt" or name.startswith("opendde_opt."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None
        sys.meta_path.insert(0, _Block())
    """))
    return _env([d])


def test_run_sh_passes_the_packages_refusal_through_with_its_exit_code(tmp_path):
    r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--mode", "fast"], env=_absent_core_env(tmp_path), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and GATE_LINE_ABSENT in r.stderr, r                       # the package's own line and rc (the core gate: 3)
    assert NOT_INSTALLED not in r.stderr and not any(w in r.stderr for w in OLD_WORDS), r.stderr[-600:]   # no word of the script's own over it


def test_run_sh_names_not_installed_only_for_an_absent_package(tmp_path):
    r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--mode", "fast"], env=_absent_package_env(tmp_path), capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and "run.sh: opendde_opt " + NOT_INSTALLED in r.stderr, r


def test_config_env_passes_the_refusal_through_and_names_not_installed_only_for_an_absent_package(tmp_path):
    src = f"source {os.path.join(KIT, 'configs', 'h100.env')}; echo AFTER rc=$?"
    r = subprocess.run(["bash", "-c", src], env=_absent_core_env(tmp_path), capture_output=True, text=True, cwd=str(tmp_path))
    assert "AFTER rc=3" in r.stdout and GATE_LINE_ABSENT in r.stderr and NOT_INSTALLED not in r.stderr and "not usable on" not in r.stderr, r
    r = subprocess.run(["bash", "-c", src], env=_absent_package_env(tmp_path / "b"), capture_output=True, text=True, cwd=str(tmp_path))
    assert "AFTER rc=3" in r.stdout and "reason=opendde_opt " + NOT_INSTALLED in r.stderr, r


def test_run_sh_passes_an_undeclared_variable_refusal_through(tmp_path):
    """With the autoload hook live in this interpreter's site, an undeclared OPENDDE_OPT_* name ends every interpreter at start with the hook's
    line and exit 3: run.sh shows that line and exits 3 — never 'not installed', never the wheel probe's word."""
    live = subprocess.run([sys.executable, "-c", "import sys; print('opendde_opt._autoload' in sys.modules)"], capture_output=True, text=True, env=_env()).stdout.strip()
    if live != "True":
        pytest.skip("opendde_opt_autoload.pth is not live in this interpreter's site (the package is importable through PYTHONPATH only here)")
    env = _env(); env["OPENDDE_OPT_BOGUS"] = "1"
    r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and "undeclared variable(s) OPENDDE_OPT_BOGUS" in r.stderr, r
    assert NOT_INSTALLED not in r.stderr and not any(w in r.stderr for w in OLD_WORDS), r.stderr[-600:]


PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_activation_line_always_names_the_reason_of_a_refusal():
    line = report.activation_line({"active": False, "mode": "exact", "line": "S1(hook=ACCEL; …)", "reason": "pending_rebase mode=exact line=S1"})
    assert line.startswith("[opendde-opt] NOT ACTIVE mode=exact reason=pending_rebase mode=exact line=S1 line=S1(hook=ACCEL; …)"), line
    bare = report.activation_line({"active": False, "mode": "exact", "gpu": {"name": "H100", "sm": 90}})      # a report without a reason: named as a defect, never blank
    assert "reason=" + report.MISSING_REASON in bare and bare.index("reason=") < bare.index("gpu="), bare
    assert "reason=" not in report.activation_line({"active": True, "mode": "fast", "levers_applied": ["arm_u"]})
    assert "reason=" not in report.activation_line({"active": False, "dry_run": True, "mode": "fast", "reason": "dry run: resolved and gated; nothing applied"}).split(" DRY RUN ")[0]


def test_the_autoload_gates_sentences_carry_reason(tmp_path):
    """_autoload._refuse ends the interpreter (os._exit 3): run it in a child and read its one line."""
    for text in ("undeclared variable(s) OPENDDE_OPT_MDOE (the package reads OPENDDE_OPT)", "reason=core_missing:opt_core (…)"):
        r = subprocess.run([sys.executable, "-c", f"from opendde_opt import _autoload; _autoload._refuse({text!r})"], capture_output=True, text=True,
                           env={**os.environ, "OPENDDE_OPT": "", "PYTHONPATH": os.pathsep.join(sys.path)})
        assert r.returncode == 3 and r.stderr.count("\n") == 1, r
        assert r.stderr.startswith("[opendde-opt] NOT ACTIVE: reason=") and r.stderr.count("reason=") == 1, r.stderr


_COMPOSERS = (                      # string tokens that compose a NOT ACTIVE sentence from parts checked above (activation_line, _refuse), parse one, or document one
    '"NOT ACTIVE"',                                             # report.activation_line's tag word
    'f"[{TAG}] NOT ACTIVE: {reason}\\n"',                       # _autoload._refuse (reason= enforced inside)
    '" NOT ACTIVE "', '"NOT ACTIVE: "',                          # parsers of a printed sentence
    'ends the interpreter with NOT ACTIVE, exit 3',             # registry KNOBS prose (the offload shim's own sentence, not this package's)
)


def test_every_other_not_active_literal_in_the_package_names_a_reason():
    """Static census of the package source (tokenize: string tokens only; comments and triple-quoted prose are not sentences the package
    prints): a single-line string literal that spells `NOT ACTIVE` carries `reason=` in the same logical line, or is a composer / parser /
    documentation token listed above."""
    offenders = []
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        src = open(os.path.join(PKG, fn)).read()
        lines = src.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.STRING or "NOT ACTIVE" not in tok.string or tok.string.lstrip("rbfuRBFU").startswith(('"""', "'''")):
                continue
            logical = " ".join(x.strip() for x in lines[tok.start[0] - 1: tok.end[0] + 1])
            if "reason=" in logical or any(c in logical for c in _COMPOSERS):
                continue
            offenders.append(f"{fn}:{tok.start[0]}: {tok.string[:160]}")
    assert offenders == [], offenders
