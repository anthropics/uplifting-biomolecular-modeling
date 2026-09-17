"""run.sh's RF6 gate (env route only, hook-live): a run under CHAI1_OPT with NO --mode is the env-route form, and that route needs the
autoload hook LIVE — `chai1_opt._autoload` in sys.modules of a fresh `env -u CHAI1_OPT python` (the effect of chai1_opt_autoload.pth processed by
site.py); a package that is merely importable (PYTHONPATH) would run stock silently under CHAI1_OPT in the environment's other
processes. The refusal line names the diagnostic. The --mode route is not gated: it activates by construction (the package's own line is
the evidence). Two venvs, built here from the interpreter that runs the tests (``--without-pip``, no network), each given that
interpreter's own site directories EXPLICITLY — one ``.pth`` of plain path lines in the venv's site (``_venv``): the packages the parent
carries (setuptools for the build backend, the pinned upstream for run.sh's pin check) are importable in both venvs, while the parent's own
``.pth`` files are NOT processed there (site.py processes ``.pth`` files of site directories only, never of paths they add) — so a
``chai1_opt`` installed in the parent does not make the hook live in venv B, whatever interpreter runs the suite (a bare image python, a
venv, an interpreter with this package installed):

* venv A — the template install ``pip install -e <core> -e <opt>`` (the parent's pip targeting the venv, build isolation off: the
  parent's setuptools builds the editable wheels, nothing is fetched): the .pth lands in venv A's site;
  ``CHAI1_OPT=exact run.sh check --mode exact`` passes the gate (the refusal line is absent; what follows is the package's own gates).
* venv B — no install; the package importable through PYTHONPATH only: ``CHAI1_OPT=exact run.sh check`` (no --mode) is refused by name,
  exit 3, the one line — with the .pth beside the PYTHONPATH entry (the tree's opt/) the diagnostic reads 'present … but not in a site
  directory'; with a PYTHONPATH copy of the package that carries no .pth it reads 'absent from the sites searched'. ``run.sh check --mode
  exact`` in the same venv (CHAI1_OPT set or not) is NOT refused: the package's own line answers; ``CHAI1_OPT=off run.sh check`` is exempt.
* the stale copy — venv A's hook is live but run.sh is invoked from a COPY of the tree: chai1_opt answers from the installed tree, not
  the copy's opt/ → refused as a stale copy.
* the user site — the same template install with ``pip install --user`` under a PYTHONUSERBASE of the test's own (the parent
  interpreter itself): the .pth lands in the user site, which the route's plain ``python`` processes → the gate passes. Inside a venv the
  user site is disabled by design (pip refuses ``--user`` there): skipped by name (``VENV_NO_USER_SITE``).
* the mutation check — a copy of run.sh with the gate block removed (the ``# RF6 gate`` … ``# /RF6 gate`` lines) lets venv B through:
  the assertion above depends on the gate.

The cases whose run passes the gate reach run.sh's pin check next (``python -I stock/check_pins.py``): they need the pinned upstream
importable by the parent interpreter and are skipped BY NAME otherwise, the reason being the pin check's own words
(``upstream_absent_reason``; ``TestSkipDiscipline`` asserts the reason); the refusal cases exit at the gate, before the pin check, and
run either way.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.abspath(os.path.join(HERE, "..", ".."))                        # chai1/opt
TREE = os.path.abspath(os.path.join(OPT, ".."))                              # chai1/
CORE = os.path.abspath(os.path.join(TREE, "..", "common", "opt_core"))
PTH = "chai1_opt_autoload.pth"
REFUSAL = "is set but the autoload hook is not live in a fresh"
STALE = "a STALE COPY answers"
NOT_IN_SITE = "but not in a site directory of this interpreter"
ABSENT = "the .pth is absent from the sites searched"
PARENT_SITES_PTH = "_parent_sites.pth"                                        # the parent interpreter's site directories, as plain path lines
VENV_NO_USER_SITE = "the user site is disabled inside a venv (sys.prefix != sys.base_prefix; pip refuses --user there): the user-site route is exercised on a plain interpreter"


def parent_sites():
    """The site directories of the interpreter running the tests (system / venv site-packages, the user site when enabled), existing ones."""
    import site
    dirs = [sysconfig_purelib()] + list(site.getsitepackages()) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else [])
    dirs += [d for d in sys.path if d.endswith(("site-packages", "dist-packages"))]
    out = []
    for d in dirs:
        d = os.path.realpath(d)
        if os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def sysconfig_purelib():
    import sysconfig
    return sysconfig.get_paths()["purelib"]


def upstream_absent_reason():
    """None when run.sh's pin check passes on the parent interpreter (the pinned upstream importable as pinned); else the skip reason BY
    NAME — the pin check's own line (stock/check_pins.py, the checker run.sh runs)."""
    r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--quiet"], capture_output=True, text=True)
    if r.returncode == 0:
        return None
    words = " ".join((r.stdout + r.stderr).split()) or f"exit {r.returncode}"
    return f"run.sh's pin check refuses on this interpreter ({sys.executable}), so a run that passes the RF6 gate stops there: {words}"


def _venv(path):
    """A venv of the parent interpreter (no pip, no system site) whose site carries the parent's site directories as plain path lines:
    the parent's packages importable, the parent's .pth files not processed."""
    r = subprocess.run([sys.executable, "-m", "venv", "--without-pip", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
    py = os.path.join(path, "bin", "python")
    site_dir = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True).stdout.strip()
    if not os.path.isdir(site_dir):
        raise AssertionError(f"venv site directory not found (a precondition of this test): {site_dir!r}")
    with open(os.path.join(site_dir, PARENT_SITES_PTH), "w", encoding="utf-8") as f:
        f.write("\n".join(parent_sites()) + "\n")
    return py


class TestRf6Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.py_a = _venv(os.path.join(cls.tmp, "venvA")); cls.py_b = _venv(os.path.join(cls.tmp, "venvB"))
        r = subprocess.run([sys.executable, "-m", "pip", "--python", cls.py_a, "install", "-q", "--no-build-isolation", "--no-deps", "-e", CORE, "-e", OPT],
                           capture_output=True, text=True)                    # the parent's pip targeting venv A (pip >= 22.3 --python); the parent's setuptools builds
        if r.returncode != 0:
            raise AssertionError("the template install into venv A failed (a precondition of this test — the parent interpreter needs pip >= 22.3 and "
                                 "setuptools >= 64, the [test] extra): " + (r.stdout + r.stderr)[-800:])
        cls.upstream_absent = upstream_absent_reason()                         # the cases that pass the gate reach the pin check: skipped by name without the upstream
        cls.site_a = subprocess.run([cls.py_a, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True).stdout.strip()
        cls.mutant = os.path.join(cls.tmp, "run_nogate.sh")                  # the gate block cut out; everything else the same bytes
        with open(os.path.join(TREE, "run.sh"), encoding="utf-8") as f: src = f.read()
        a, b = src.index("# RF6 gate"), src.index("# /RF6 gate\n") + len("# /RF6 gate\n")
        with open(cls.mutant, "w", encoding="utf-8") as f: f.write(src[:a] + src[b:])
        os.chmod(cls.mutant, 0o755)
        cls.nopth = os.path.join(cls.tmp, "nopth")                          # the package importable with NO .pth anywhere on the path
        shutil.copytree(os.path.join(OPT, "chai1_opt"), os.path.join(cls.nopth, "chai1_opt"), ignore=shutil.ignore_patterns("__pycache__"))
        cls.tree2 = os.path.join(cls.tmp, "tree2")                          # a copy of the tree: run.sh here, the install answers from TREE
        shutil.copytree(TREE, cls.tree2, ignore=shutil.ignore_patterns("__pycache__", "forward", "src", "*.whl"), symlinks=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def env(python, pythonpath=None, chai1_opt=None, extra=None):
        e = {"PATH": os.path.dirname(python) + os.pathsep + os.environ.get("PATH", ""), "HOME": os.path.dirname(os.path.dirname(python)),
             "PYTHONDONTWRITEBYTECODE": "1", "LANG": os.environ.get("LANG", "C.UTF-8")}
        if pythonpath: e["PYTHONPATH"] = pythonpath
        if chai1_opt: e["CHAI1_OPT"] = chai1_opt
        e.update(extra or {})
        return e

    def run_sh(self, python, args, pythonpath=None, chai1_opt=None, script=None, tree=None, extra=None):
        tree = tree or TREE; script = script or os.path.join(tree, "run.sh")
        r = subprocess.run(["bash", script] + args, capture_output=True, text=True, env=self.env(python, pythonpath, chai1_opt, extra), cwd=tree)
        return r.returncode, r.stdout + r.stderr

    def test_a_startup_refusal_is_relayed_by_name_not_misread_as_absent(self):
        """The hook is LIVE (venv A, the .pth processed at interpreter start): a refusal the package prints at start-up — an unknown
        CHAI1_OPT, an undeclared CHAI1_OPT_* variable — reaches the user BY NAME with rc 3 through run.sh's and configs/h100.env's
        import probes; never the false 'chai1_opt is not installed'. A valid CHAI1_OPT passes the probes (the .pth arms the trigger's
        finder only; the core gate runs at the trigger, not at start-up)."""
        for args in (["check"], ["check", "--config", "h100"]):
            rc, out = self.run_sh(self.py_a, args, chai1_opt="bogus")
            self.assertEqual(rc, 3, (args, out[-600:]))
            self.assertIn("[chai1-opt] NOT ACTIVE: unknown CHAI1_OPT='bogus' (expected exact|fast|big|off)", out, (args, out[-600:]))
            self.assertNotIn("is not installed", out, (args, out[-600:]))
            rc, out = self.run_sh(self.py_a, args, chai1_opt="exact", extra={"CHAI1_OPT_BOGUS": "1"})
            self.assertEqual(rc, 3, (args, out[-600:]))
            self.assertIn("[chai1-opt] NOT ACTIVE: undeclared variable(s) CHAI1_OPT_BOGUS", out, (args, out[-600:]))
            self.assertNotIn("is not installed", out, (args, out[-600:]))
        probe = subprocess.run([self.py_a, "-c", "import chai1_opt; print('importable')"], capture_output=True, text=True, cwd=os.path.dirname(self.py_a), env=self.env(self.py_a, chai1_opt="fast"))
        self.assertEqual((probe.returncode, probe.stdout.strip()), (0, "importable"), probe.stderr[-300:])   # a valid variable: the probe lives (no gate at start-up)

    def needs_upstream(self):
        """The case's run passes the RF6 gate and reaches run.sh's pin check: skipped BY NAME when the pinned upstream is absent."""
        if self.upstream_absent:
            self.skipTest(self.upstream_absent)

    def test_a_template_install_hook_live_passes_the_gate(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site_a, PTH)), f"{PTH} not in venv A's site {self.site_a}")
        probe = subprocess.run([self.py_b, "-c", "import sys, importlib.util as u; print('chai1_opt._autoload' in sys.modules, u.find_spec('chai1_opt') is not None)"],
                               capture_output=True, text=True, cwd=os.path.dirname(self.py_b), env=self.env(self.py_b))
        self.assertEqual(probe.stdout.strip(), "False False", probe.stderr[-300:])   # venv B: no hook, no package — whatever the parent interpreter carries
        self.needs_upstream()
        probe = subprocess.run([self.py_a, "-c", "import sys; print('chai1_opt._autoload' in sys.modules)"], capture_output=True, text=True, cwd=os.path.dirname(self.py_a),
                               env={k: v for k, v in os.environ.items() if k != "CHAI1_OPT"})   # the variable unset, as run.sh probes
        self.assertEqual(probe.stdout.strip(), "True", probe.stderr[-300:])   # the hook is live in a fresh interpreter: the .pth's effect
        rc, out = self.run_sh(self.py_a, ["check"], chai1_opt="exact")        # the env-route form: no --mode
        self.assertNotIn(REFUSAL, out, out[-600:]); self.assertNotIn(STALE, out, out[-600:])   # the gate lets it through; the package's own gates follow
        self.assertIn("[chai1-opt]", out, out[-600:])

    def test_b_pth_beside_pythonpath_is_refused_present_but_not_processed(self):
        rc, out = self.run_sh(self.py_b, ["check"], pythonpath=OPT + os.pathsep + CORE, chai1_opt="exact")
        self.assertEqual(rc, 3, out[-600:])
        lines = [l for l in out.splitlines() if REFUSAL in l]
        self.assertEqual(len(lines), 1, out[-600:]); self.assertIn(NOT_IN_SITE, lines[0]); self.assertIn("pip install -e", lines[0])

    def test_b_pythonpath_copy_without_pth_is_refused_absent(self):
        rc, out = self.run_sh(self.py_b, ["check"], pythonpath=self.nopth + os.pathsep + CORE, chai1_opt="exact")
        self.assertEqual(rc, 3, out[-600:])
        lines = [l for l in out.splitlines() if REFUSAL in l]
        self.assertEqual(len(lines), 1, out[-600:]); self.assertIn(ABSENT, lines[0])

    def test_b_mode_route_is_not_gated(self):
        self.needs_upstream()
        for chai1_opt in ("exact", None):                                     # --mode given: activation by construction, the package's own line answers
            rc, out = self.run_sh(self.py_b, ["check", "--mode", "exact"], pythonpath=OPT + os.pathsep + CORE, chai1_opt=chai1_opt)
            self.assertNotIn(REFUSAL, out, out[-600:]); self.assertIn("[chai1-opt]", out, out[-600:])

    def test_usersite_install_passes_the_gate(self):
        if sys.prefix != sys.base_prefix:
            self.skipTest(VENV_NO_USER_SITE)
        self.needs_upstream()
        ub = os.path.join(self.tmp, "userbase")
        env = {k: v for k, v in os.environ.items() if k not in ("CHAI1_OPT", "PYTHONNOUSERSITE")}; env["PYTHONUSERBASE"] = ub
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "--no-build-isolation", "--no-deps", "-e", CORE, "-e", OPT], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, "the user-site install (a precondition of this case) failed: " + (r.stdout + r.stderr)[-600:])
        usite = subprocess.run([sys.executable, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True, env=env).stdout.strip()
        self.assertTrue(usite.startswith(ub) and os.path.isfile(os.path.join(usite, PTH)), f"{PTH} not in the user site {usite}")
        e = self.env(sys.executable, chai1_opt="exact"); e["PYTHONUSERBASE"] = ub
        r = subprocess.run(["bash", os.path.join(TREE, "run.sh"), "check"], capture_output=True, text=True, env=e, cwd=TREE); out = r.stdout + r.stderr
        self.assertNotIn(REFUSAL, out, out[-600:]); self.assertNotIn(STALE, out, out[-600:]); self.assertIn("[chai1-opt]", out, out[-600:])

    def test_stale_copy_is_refused(self):
        rc, out = self.run_sh(self.py_a, ["check"], chai1_opt="exact", tree=self.tree2)
        self.assertEqual(rc, 3, out[-600:])
        self.assertEqual(sum(1 for l in out.splitlines() if STALE in l), 1, out[-600:])

    def test_b_off_is_exempt(self):
        rc, out = self.run_sh(self.py_b, ["check"], pythonpath=OPT + os.pathsep + CORE, chai1_opt="off")           # the stock route needs no hook
        self.assertNotIn(REFUSAL, out, out[-600:])

    def test_mutation_gate_removed_lets_venv_b_through(self):
        for pp in (OPT, self.nopth):
            rc, out = self.run_sh(self.py_b, ["check"], pythonpath=pp + os.pathsep + CORE, chai1_opt="exact", script=self.mutant)
            self.assertNotIn(REFUSAL, out, out[-600:])                      # without the gate the refusal never appears: the assertions above are the gate's


class TestSkipDiscipline(unittest.TestCase):
    """The skips above are by name: the reason is the pin check's own refusal (or the venv's user-site rule), never a silent pass."""

    def test_upstream_skip_reason_is_the_pin_checks_words(self):
        reason = upstream_absent_reason()
        rc = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--quiet"], capture_output=True, text=True).returncode
        if rc == 0:
            self.assertIsNone(reason)                                          # the upstream as pinned: nothing is skipped for it
        else:
            self.assertTrue(reason.startswith("run.sh's pin check refuses on this interpreter ("), reason)
            self.assertIn("chai_lab", reason); self.assertIn("check_pins", reason)   # by name: the package and the checker

    def test_parent_sites_are_directories_of_this_interpreter(self):
        dirs = parent_sites()
        self.assertTrue(dirs and all(os.path.isabs(d) and os.path.isdir(d) for d in dirs), dirs)
        self.assertIn(os.path.realpath(sysconfig_purelib()), dirs)


class TestStartup(unittest.TestCase):
    """Packaging, interpreter start and the autoload trigger: `pip install -e chai1/opt` works in a fresh venv WITHOUT torch or chai_lab,
    ships chai1_opt_autoload.pth into site-packages, that .pth leaves `python -c pass` free of torch / chai_lab with and without CHAI1_OPT
    set, the console script answers, and under CHAI1_OPT=exact the finder fires on the first import of `chai_lab.chai1` (a stub package on
    PYTHONPATH) — the activation then fails on the pin gate and the process exits 3, never running stock silently. Also builds the wheel
    and checks the .pth sits at its root with a RECORD entry. Runs in a throwaway venv (~20-60 s); venv/pip failures are test failures."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.venv = os.path.join(cls.tmp, "venv")
        r = subprocess.run([sys.executable, "-m", "venv", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp,
                   "PYTHONDONTWRITEBYTECODE": "1"}
        for k, v in os.environ.items():                       # pip's build isolation fetches setuptools: keep the box's proxy / CA settings
            if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE") or k.startswith("PIP_"):
                cls.env[k] = v
        r = subprocess.run([cls.python, "-m", "pip", "install", "-q", "-e", CORE, "-e", OPT], capture_output=True, text=True, env=cls.env)   # the core beside the kit
        if r.returncode != 0:
            raise AssertionError("pip install -e failed: " + (r.stdout + r.stderr)[-800:])
        cls.site = subprocess.run([cls.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True,
                                  text=True, env=cls.env).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_py(self, code, extra_env=None, cwd=None):
        env = dict(self.env); env.update(extra_env or {})
        return subprocess.run([self.python, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env, cwd=cwd or self.tmp)

    def test_pth_installed_and_start_is_clean(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site, PTH)), os.listdir(self.site))
        for extra in ({}, {"CHAI1_OPT": "exact"}, {"CHAI1_OPT": "off"}):
            r = self.run_py("import sys; print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch','chai_lab','numpy')))", extra)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "[]", (extra, r.stdout))
        r = self.run_py("import sys; f=[x for x in sys.meta_path if type(x).__name__=='Finder']; print(len(f), f[0].mode if f else None)", {"CHAI1_OPT": "exact"})
        self.assertEqual(r.stdout.strip(), "1 exact")
        r = self.run_py("import sys; print(len([x for x in sys.meta_path if type(x).__name__=='Finder']))")
        self.assertEqual(r.stdout.strip(), "0")
        r = self.run_py("pass", {"CHAI1_OPT": "turbo"})
        self.assertIn("NOT ACTIVE: unknown CHAI1_OPT='turbo'", r.stderr)
        self.assertEqual(r.returncode, 3, r.stderr)                    # an unknown selection never lets stock run silently under the variable

    def test_console_script(self):
        r = subprocess.run([os.path.join(self.venv, "bin", "chai1-opt"), "--help"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 0); self.assertIn("usage: chai1-opt <command>", r.stdout)
        env = {k: v for k, v in self.env.items() if k != "CHAI1_OPT"}
        r = subprocess.run([os.path.join(self.venv, "bin", "chai1-opt"), "check"], capture_output=True, text=True, env=env)
        # no --mode, no CHAI1_OPT: the package default (exact) resolves and the gates run — this venv has no chai_lab, so the pin refuses
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[chai1-opt] NOT ACTIVE: stock pin: chai_lab is not installed", r.stderr)
        r = subprocess.run([os.path.join(self.venv, "bin", "chai1-opt"), "check", "--mode", "turbo"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2); self.assertIn("not a mode", r.stderr)

    def test_trigger_fires_and_refuses_to_run_stock_silently(self):
        stub = os.path.join(self.tmp, "stubpath")
        os.makedirs(os.path.join(stub, "chai_lab"), exist_ok=True)
        open(os.path.join(stub, "chai_lab", "__init__.py"), "w").write("")
        open(os.path.join(stub, "chai_lab", "chai1.py"), "w").write("BODY_RAN = True\n")
        code = "import chai_lab.chai1 as c; print('imported', c.BODY_RAN)"
        r = self.run_py(code, {"CHAI1_OPT": "exact", "PYTHONPATH": stub})
        self.assertEqual(r.returncode, 3, (r.stdout, r.stderr))
        self.assertIn("[chai1-opt] NOT ACTIVE:", r.stderr)
        self.assertNotIn("imported True", r.stdout)                       # the program never proceeds past the failed activation
        r = self.run_py(code, {"CHAI1_OPT": "exact", "PYTHONPATH": stub, "CHAI_DETERMINISTIC": "warn"})   # the kit's switch, not a level here
        self.assertEqual(r.returncode, 3, (r.stdout, r.stderr))
        self.assertIn("NOT ACTIVE: CHAI_DETERMINISTIC='warn': the package's levels are 0|1", r.stderr); self.assertNotIn("imported True", r.stdout)
        r = self.run_py(code, {"PYTHONPATH": stub})                        # no CHAI1_OPT: nothing fires
        self.assertEqual(r.returncode, 0); self.assertIn("imported True", r.stdout); self.assertNotIn("[chai1-opt]", r.stderr)
        r = self.run_py("import chai1_opt._autoload as a; print(a.TRIGGERS, a.MODES)")
        self.assertEqual(r.stdout.strip(), "('chai_lab.chai1',) ('exact', 'fast', 'big')")

    def test_wheel_has_the_pth_at_its_root(self):
        wd = os.path.join(self.tmp, "wheel")
        src = os.path.join(self.tmp, "opt_copy")                                   # build from a copy: setuptools writes build/ beside the source
        shutil.copytree(OPT, src, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "forward"))
        r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", "--no-deps", "-w", wd, src], capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-500:])
        whl = glob.glob(os.path.join(wd, "chai1_opt-*.whl"))[0]
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            self.assertIn(PTH, names)
            rec = [n for n in names if n.endswith("RECORD")][0]
            self.assertIn(PTH + ",sha256=", z.read(rec).decode())
            self.assertTrue(any(n.startswith("chai1_opt/tests/") for n in names))


if __name__ == "__main__":
    unittest.main()
