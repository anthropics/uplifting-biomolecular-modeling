"""The core pin gate on every documented route (`_core_gate.gate` through `_autoload.core_gate()`, statement one of each entry), against
an ABSENT core (the kit's `opt/` + `run.sh` + `configs/` copied into a tree with no `common/opt_core`, children started with `python -S`
so an installed core is out of reach) and — through the one route that exercises a real pip-style install (a stdlib venv whose
site-packages holds the kit's generated `.pth`) — a STALE core (a shadow `opt_core 0.2.5` first on PYTHONPATH): each route ends with
exactly one `[rfdiffusion3-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: …` line, exit 3, no traceback, and nothing
after the gate runs; with RFDIFFUSION3_OPT unset or `off` the package import and the real `.pth` line stay inert and core-free (rc 0, no
line), and the core-free modules (`install`, `registry`, `settings`, `inputs`, `stock_design` at import) load nothing of the
core. One representative per documented route family (README.md "The command", run.sh header, configs/h100.env header, pyproject
[project.scripts], `__init__` docstring: `python -m rfdiffusion3_opt`, `enable()`, the `.pth` under RFDIFFUSION3_OPT=<mode>, a lazy
public name, a child module, `run.sh <verb>`, `source configs/h100.env`) proves that route calls the gate; the gate's own refusal logic
(absent vs. stale vs. unversioned, the exact line text) is exercised exhaustively in test_core_adoption.py without subprocess cost —
this file is about coverage of ENTRY POINTS, not of the gate's decision table."""
import ast
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import venv

import rfdiffusion3_opt as pkg

from .. import _autoload, _core_gate, registry

TREE = registry.tree_home()
OPT_HOME = os.path.join(TREE, "opt")
PKG = "rfdiffusion3_opt"
PTH = "rfdiffusion3_opt_autoload.pth"
MARKER = "stock would run here"
STALE_VERSION = "0.2.5"


class _Tree:
    """The fixture tree: <root>/rfdiffusion3/{opt/{pyproject.toml,_build_backend.py,<pkg>/,<pkg>_autoload.pth},run.sh,configs/}, no <root>/common;
    a shadow stale core, a `python` wrapper (this interpreter with -S) first on PATH, a site dir holding the kit's real .pth, and a fake
    upstream `rfd3` whose body prints MARKER."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="rfd3_core_gate_routes_")
        kit = os.path.join(self.root, "rfdiffusion3")
        self.opt = os.path.join(kit, "opt")
        os.makedirs(self.opt)
        for name in ("pyproject.toml", "_build_backend.py", PTH):
            shutil.copy(os.path.join(OPT_HOME, name), os.path.join(self.opt, name))
        shutil.copytree(os.path.join(OPT_HOME, PKG), os.path.join(self.opt, PKG), ignore=shutil.ignore_patterns("tests", "__pycache__", "*.egg-info"))
        shutil.copy(os.path.join(TREE, "run.sh"), os.path.join(kit, "run.sh"))
        shutil.copytree(os.path.join(TREE, "configs"), os.path.join(kit, "configs"))
        self.kit = kit
        assert not os.path.exists(os.path.join(self.root, "common"))
        self.stale = self._core(STALE_VERSION)
        self.bin = os.path.join(self.root, "bin")
        os.makedirs(self.bin)
        wrapper = os.path.join(self.bin, "python")
        with open(wrapper, "w") as fh:
            fh.write("#!/bin/sh\nexec '%s' -S \"$@\"\n" % sys.executable)          # -S: no site-packages, so an installed opt_core is out of reach; PYTHONPATH is honoured
        os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.site = os.path.join(self.root, "site")
        os.makedirs(self.site)
        shutil.copy(os.path.join(self.opt, PTH), os.path.join(self.site, PTH))       # the kit's real, generated .pth
        self.upstream = os.path.join(self.root, "upstream")
        os.makedirs(os.path.join(self.upstream, "rfd3"))
        with open(os.path.join(self.upstream, "rfd3", "__init__.py"), "w") as fh:
            fh.write("print(%r)\n" % MARKER)
        pin = _core_gate.read_table(os.path.join(self.opt, "pyproject.toml"), "tool.opt_core")   # the pin as the gate reads it (one reader)
        self.want_version = pin["version"]
        # a real interpreter whose site-packages carries the kit's generated .pth (as `pip install -e opt` places it) and a path .pth for the
        # package (+ the shadow stale core): every `python` the bash routes start processes both, as on an installed box
        self.venv = os.path.join(self.root, "venv")
        venv.create(self.venv, with_pip=False, symlinks=os.name != "nt")
        self.venv_python = os.path.join(self.venv, "bin", "python")
        self.venv_site = subprocess.run([self.venv_python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                        capture_output=True, text=True, check=True, timeout=120).stdout.strip()
        shutil.copy(os.path.join(self.opt, PTH), os.path.join(self.venv_site, PTH))
        self.venv_paths_pth = os.path.join(self.venv_site, "aaa_rfd3_core_gate_paths.pth")   # sorts before the kit's .pth: the package is importable when its line runs

    def _core(self, version):
        root = os.path.join(self.root, "stale_core")
        os.makedirs(os.path.join(root, "opt_core"))
        with open(os.path.join(root, "opt_core", "__init__.py"), "w") as fh:
            fh.write('__version__ = "%s"\n' % version)
        return root

    def env(self, fixture, mode=None, upstream=False):
        paths = {"absent": [], "stale": [self.stale]}[fixture] + [self.opt] + ([self.upstream] if upstream else [])
        env = {"PATH": self.bin + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(paths), "HOME": self.root,
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": os.environ.get("LANG", "C.UTF-8")}
        if mode is not None:
            env[_autoload.ENV] = mode
        return env

    def python(self, args, fixture, mode=None, upstream=False):
        return subprocess.run([sys.executable, "-S"] + list(args), env=self.env(fixture, mode, upstream), capture_output=True, text=True, cwd=self.root, timeout=120)

    def bash(self, script, fixture, mode=None):
        return subprocess.run(["bash", "-c", script], env=self.env(fixture, mode), capture_output=True, text=True, cwd=self.kit, timeout=120)

    def venv_bash(self, script, fixture, extra_env):
        """`script` in the fixture's rfdiffusion3/ with the venv's python first on PATH, no PYTHONPATH, the fixture's paths in the path .pth."""
        paths = {"absent": [], "stale": [self.stale]}[fixture] + [self.opt]
        with open(self.venv_paths_pth, "w") as fh:
            fh.write("\n".join(paths) + "\n")
        env = {"PATH": os.path.join(self.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": self.root, "PYTHONDONTWRITEBYTECODE": "1",
               "LANG": os.environ.get("LANG", "C.UTF-8")}
        env.update(extra_env)
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=self.kit, timeout=120)


T = None


def setUpModule():
    global T
    T = _Tree()


def tearDownModule():
    shutil.rmtree(T.root, ignore_errors=True)


CONSOLE = "import sys; from rfdiffusion3_opt.__main__ import main; sys.exit(main())"          # the console script `rfdiffusion3-opt`, as setuptools writes it
PTH_CHILD = "import site; site.addsitedir(%r); import rfd3; print('after the trigger')"      # upstream's own command line under the .pth: the trigger is `import rfd3`
STOCK_DESIGN = "import sys; from rfdiffusion3_opt.stock_design import main; sys.exit(main(['--', 'design', 'x=1']))"   # `python -I -m rfdiffusion3_opt.stock_design -- design …` (-I drops PYTHONPATH, so the fixture reaches main() by import)

# One representative per documented route family — not an exhaustive cross-product (the gate's own decision table is test_core_adoption.py's job).
PY_ROUTES = {                                   # id -> (argv after `python -S`, RFDIFFUSION3_OPT, needs the fake upstream)
    "python_m_check_mode_exact": (["-m", PKG, "check", "--mode", "exact"], None, False),
    "console_script_check_mode_fast": (["-c", CONSOLE, "check", "--mode", "fast"], None, False),
    "enable_exact_in_process": (["-c", "import rfdiffusion3_opt as k; k.enable('exact'); print('REACHED')"], None, False),
    "pth_import_rfd3_under_RFDIFFUSION3_OPT_exact": (["-c", PTH_CHILD], "exact", True),
    "child_stock_design_main": (["-c", STOCK_DESIGN], None, False),
    "lazy_attribute_MODES": (["-c", "import rfdiffusion3_opt as k; k.MODES; print('REACHED')"], None, False),   # representative of the __getattr__ gate shared by every name in pkg._CORE_NAMES
}
BASH_ROUTES = {                                 # id -> (script run in the fixture's rfdiffusion3/ directory, RFDIFFUSION3_OPT)
    "run_sh_check_mode_exact": ("bash run.sh check --mode exact", None),
    "source_configs_h100_env": ("source configs/h100.env || exit $?; echo REACHED", None),
}
VENV_ROUTES = {                                 # id -> (script, environment on top of the venv's): the variable form through the INSTALLED .pth
    "VAR_exact_run_sh_check": ("bash run.sh check", {"RFDIFFUSION3_OPT": "exact"}),
    "VAR_exact_source_configs_h100_env": ("source configs/h100.env || exit $?; echo REACHED", {"RFDIFFUSION3_OPT": " Exact "}),
}
REASON = {"absent": "NOT ACTIVE: reason=core_missing:opt_core", "stale": "NOT ACTIVE: reason=core_mismatch:"}
CORE_FREE_NAMES = ("registry", "settings", "manifest", "install", "det", "stock_design", "_autoload", "ActivationError", "__version__")   # __init__._LAZY minus _CORE_NAMES, plus the two plain attributes (det: the recipe's table and readers; its torch-side det_scatter is not a lazy name)


def module_level_core_closure():
    """{lazy public name: True when importing it reaches `opt_core` at module level} from the package sources: module-level import statements
    (function and class bodies excluded), closed over sibling modules. `_autoload`'s one module-level core import sits under a set
    RFDIFFUSION3_OPT after its own core_gate() (the inert tests below hold it import-free otherwise), so the closure stops there."""
    pkg_dir = os.path.dirname(os.path.abspath(pkg.__file__))
    mods = {fn[:-3] for fn in os.listdir(pkg_dir) if fn.endswith(".py")}

    def imports(mod):
        found = set()

        def walk(nodes):
            for n in nodes:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(n, ast.Import):
                    found.update(a.name.split(".")[0] for a in n.names)
                elif isinstance(n, ast.ImportFrom):
                    if n.level:
                        found.update(["." + n.module.split(".")[0]] if n.module else ["." + a.name for a in n.names])
                    else:
                        found.add(n.module.split(".")[0])
                walk(ast.iter_child_nodes(n))
        walk(ast.parse(open(os.path.join(pkg_dir, mod + ".py"), encoding="utf-8").read()).body)
        return found

    table = {m: imports(m) for m in mods}

    def reaches(mod, seen):
        if mod in seen or mod == "_autoload" or mod not in mods:
            return False
        seen.add(mod)
        return "opt_core" in table[mod] or any(reaches(i[1:], seen) for i in table[mod] if i.startswith("."))
    names = {n: ("modes" if n in ("MODES", "VARIANTS") else n) for n in tuple(pkg._LAZY) + ("MODES", "VARIANTS")}
    return {n: reaches(m, set()) for n, m in names.items()}


class TestCoreGateRoutes(unittest.TestCase):
    def refused(self, r, fixture):
        """rc 3, exactly one NOT ACTIVE line naming the reason, no traceback, nothing after the gate."""
        msg = "rc=%s\nstdout=%r\nstderr=%r" % (r.returncode, r.stdout[-600:], r.stderr[-900:])
        self.assertEqual(r.returncode, _autoload.EXIT_NOT_ACTIVE, msg)
        self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, msg)
        line = next(l for l in r.stderr.splitlines() if "NOT ACTIVE" in l)
        self.assertTrue(line.startswith("[%s] %s" % (_autoload.TAG, REASON[fixture])), msg)
        self.assertIn("v%s" % T.want_version, line, msg)
        if fixture == "stale":
            self.assertIn("v%s" % STALE_VERSION, line, msg)
            self.assertIn(T.stale, line, msg)
        for word in ("Traceback", "Error processing line", "ModuleNotFoundError", "not importable"):
            self.assertNotIn(word, r.stderr, msg)
        for marker in ("REACHED", MARKER, "after the trigger", "DRY-RUN", "ACTIVE mode="):
            self.assertNotIn(marker, r.stdout, msg)
            self.assertNotIn(marker, r.stderr.replace("NOT ACTIVE", ""), msg)

    def inert(self, r, marker):
        msg = "rc=%s\nstdout=%r\nstderr=%r" % (r.returncode, r.stdout[-600:], r.stderr[-900:])
        self.assertEqual(r.returncode, 0, msg)
        self.assertNotIn("NOT ACTIVE", r.stderr, msg)
        self.assertIn(marker, r.stdout, msg)

    # --- inert and core-free under an ABSENT core --------------------------------------------------------------------------------
    def test_absent__package_import_is_inert_and_core_free_unset_and_off(self):
        code = ("import sys, rfdiffusion3_opt, rfdiffusion3_opt._autoload as a; assert a.FINDER is None; "
                "loaded = sorted(m for m in sys.modules if m.startswith(('opt_core', 'rfdiffusion3_opt'))); "
                "assert loaded == ['rfdiffusion3_opt', 'rfdiffusion3_opt._autoload'], loaded; print('inert')")
        self.inert(T.python(["-c", code], "absent"), "inert")
        self.inert(T.python(["-c", code], "absent", mode="off"), "inert")
        self.inert(T.python(["-c", code], "absent", mode=""), "inert")

    def test_absent__pth_line_is_inert_unset_and_off(self):
        for mode in (None, "off", " OFF "):
            r = T.python(["-c", PTH_CHILD % T.site + "; import sys; assert not [m for m in sys.modules if m.startswith('opt_core')]"], "absent", mode=mode, upstream=True)
            self.inert(r, MARKER)
            self.assertIn("after the trigger", r.stdout)

    def test_absent__core_free_modules_load_nothing_of_the_core(self):
        code = ("import sys, rfdiffusion3_opt.install, rfdiffusion3_opt.registry, rfdiffusion3_opt.settings, "
                "rfdiffusion3_opt.stock_design, rfdiffusion3_opt.det, rfdiffusion3_opt.routes, rfdiffusion3_opt.census, rfdiffusion3_opt._core_gate; "
                "assert not [m for m in sys.modules if m.startswith('opt_core')]; print('core-free')")
        self.inert(T.python(["-c", code], "absent"), "core-free")

    def test_absent__run_sh_usage_and_install_are_not_gated(self):
        r = T.bash("bash run.sh", "absent")                                                     # usage: rc 2 before any python
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("NOT ACTIVE", "\n".join(l for l in r.stderr.splitlines() if not l.startswith("#")))   # no gate line (the usage text itself, the header comment, documents the NOT ACTIVE exit code)
        r = T.bash("bash run.sh install", "absent")                                             # the install verb IS the core's install line: never gated (here pip is out of reach under -S, or the path is absent)
        self.assertNotEqual(r.returncode, _autoload.EXIT_NOT_ACTIVE, r.stderr)
        self.assertNotIn("NOT ACTIVE", r.stderr)

    def test_core_names_are_exactly_the_lazy_names_that_reach_the_core(self):
        """`__init__._CORE_NAMES` (gated in `__getattr__`) == the lazy public names whose import reaches opt_core at module level — computed
        from the sources, so a module that starts importing the core joins the gated set or this fails."""
        closure = module_level_core_closure()
        self.assertEqual(sorted(n for n, core in closure.items() if core), sorted(pkg._CORE_NAMES))
        self.assertTrue(set(pkg.__all__) <= set(closure) | {"enable", "status", "ActivationError"}, pkg.__all__)
        for n in CORE_FREE_NAMES:
            self.assertFalse(closure.get(n, False), n)

    def test_absent__core_free_lazy_names_are_inert(self):
        code = ("import sys, rfdiffusion3_opt as k; [getattr(k, n) for n in %r]; "
                "assert not [m for m in sys.modules if m.startswith('opt_core')], sorted(sys.modules); print('inert')" % (CORE_FREE_NAMES,))
        self.inert(T.python(["-c", code], "absent"), "inert")

    def test_absent__venv_typo_name_is_refused_by_name_not_as_not_importable(self):
        """An undeclared RFDIFFUSION3_OPT_<x> is refused at interpreter start in every python the script starts: one named line, rc 3 — the
        import probe (run with a clean environment) does not mistake that refusal for a missing package."""
        r = T.venv_bash("bash run.sh check --mode exact", "absent", {"RFDIFFUSION3_OPT_TYPO": "1"})
        msg = "rc=%s stderr=%r" % (r.returncode, r.stderr[-600:])
        self.assertEqual(r.returncode, _autoload.EXIT_NOT_ACTIVE, msg)
        self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, msg)
        self.assertIn("NOT ACTIVE: undeclared RFDIFFUSION3_OPT_TYPO", r.stderr, msg)
        self.assertNotIn("not importable", r.stderr, msg)

    def test_the_fixture_gate_passes_on_a_matching_core(self):
        """Control: the same child with THIS process's core root on the path passes the gate (so the refusals above are the gate's, not the fixture's)."""
        root = _autoload.core_gate()["installed"]["root"]
        r = subprocess.run([sys.executable, "-S", "-c", "from rfdiffusion3_opt._autoload import core_gate; f = core_gate(); print('OK', f['installed']['version'], f['tag'])"],
                           env=dict(T.env("absent"), PYTHONPATH=root + os.pathsep + T.opt), capture_output=True, text=True, timeout=120)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (0, "OK %s %s" % (T.want_version, _autoload.TAG), ""))


def _make_py(route_id, fixture):
    def test(self):
        args, mode, upstream = PY_ROUTES[route_id]
        args = [a % T.site if (a.startswith("import site") and "%r" in a) else a for a in args]
        self.refused(T.python(args, fixture, mode=mode, upstream=upstream), fixture)
    test.__name__ = "test_%s__%s" % (fixture, route_id)
    return test


def _make_bash(route_id, fixture):
    def test(self):
        script, mode = BASH_ROUTES[route_id]
        self.refused(T.bash(script, fixture, mode=mode), fixture)
    test.__name__ = "test_%s__%s" % (fixture, route_id)
    return test


def _make_venv(route_id, fixture):
    def test(self):
        script, extra = VENV_ROUTES[route_id]
        self.refused(T.venv_bash(script, fixture, extra), fixture)
    test.__name__ = "test_%s__venv_%s" % (fixture, route_id)
    return test


for _rid in PY_ROUTES:                                             # one entry-point-family sample, "absent" fixture only
    setattr(TestCoreGateRoutes, "test_absent__%s" % _rid, _make_py(_rid, "absent"))
for _rid in BASH_ROUTES:
    setattr(TestCoreGateRoutes, "test_absent__%s" % _rid, _make_bash(_rid, "absent"))
for _fixture in ("absent", "stale"):                                # the venv route is the one place a real pip-style install is exercised: both fixtures
    for _rid in VENV_ROUTES:
        setattr(TestCoreGateRoutes, "test_%s__venv_%s" % (_fixture, _rid), _make_venv(_rid, _fixture))
del _fixture, _rid


if __name__ == "__main__":
    unittest.main()
