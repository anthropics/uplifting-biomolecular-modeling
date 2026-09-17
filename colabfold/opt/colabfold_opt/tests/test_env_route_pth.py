"""The environment route — `COLABFOLD_OPT=<kit mode>` and no `--mode` on the command line —
fires through the hook `colabfold_opt._autoload`, which `colabfold_opt_autoload.pth` imports at interpreter start only when site.py
processes the file; a package merely importable proves nothing. `run.sh` gates that route with `stack.autoload_pth_check` (a FRESH
`python`, started as the route's own — the user site and PYTHONPATH as the route sees them — must carry the hook in sys.modules; the
line names the diagnostic). Two venvs: A = the kit installed (`pip install -e <opt
copy>`: the .pth in site, the hook live) → `COLABFOLD_OPT=fast run.sh check` passes the gate (its refusal, when any, is another gate's — the
pinned upstream is not in a test venv — never the hook's); B = the same package importable through PYTHONPATH, NO .pth in site →
`COLABFOLD_OPT=fast run.sh check|pred` (no --mode) refuse with the one NOT ACTIVE line naming the hook and the diagnostic (absent
from the searched sites), exit 3; the .pth COPIED beside the PYTHONPATH entry → still refused, `present beside … but not processed`;
`run.sh check --mode fast` in venv B is NOT refused by this gate (the --mode route is un-gated: whatever refuses is another gate's);
`COLABFOLD_OPT=off run.sh check` is exempt; C = a `--system-site-packages` venv with the kit installed `--user` into a PYTHONUSERBASE
user site (the .pth in the USER site, which the route's `python` processes) → passes the gate. Mutation check: without the gate venv B's
verbs refuse for another reason and these assertions fail."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from colabfold_opt import report, stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <kit>/opt
RUN_SH = os.path.join(os.path.dirname(OPT), "run.sh")
A3M = os.path.join(os.path.dirname(OPT), "tests", "inputs", "1BRS_AD.a3m")                   # the kit's own test input (test_cli_manifest.py)


class TestAutoloadHookGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        tree_copy = os.path.join(cls.tmp, "tree")                        # the installs write beside their source: never the tree
        shutil.copytree(os.path.dirname(OPT), tree_copy, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", ".pytest_cache", "*.whl", "src"))
        cls.opt_copy = os.path.join(tree_copy, "opt")
        cls.pkg_only = os.path.join(cls.tmp, "pkgonly")                  # venv B's PYTHONPATH entry: the package alone, no .pth beside it
        shutil.copytree(os.path.join(cls.opt_copy, "colabfold_opt"), os.path.join(cls.pkg_only, "colabfold_opt"), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(os.path.join(cls.opt_copy, "pyproject.toml"), os.path.join(cls.pkg_only, "pyproject.toml"))   # the package's pin travels with it: the core pin gate reads [tool.opt_core] above the package
        import opt_core                                                  # the core this process imports (a kit = the core + itself)
        cls.core_dir = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
        cls.py = {}
        for name in ("A", "B"):
            venv = os.path.join(cls.tmp, "venv" + name)
            subprocess.run([sys.executable, "-m", "venv", venv], check=True, capture_output=True)
            cls.py[name] = os.path.join(venv, "bin", "python")
        r = subprocess.run([cls.py["A"], "-m", "pip", "install", "-q", "-e", cls.core_dir, "-e", cls.opt_copy], capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"editable install failed (pip needs setuptools from the index): {r.stderr[-500:]}")
        r = subprocess.run([cls.py["B"], "-m", "pip", "install", "-q", "-e", cls.core_dir], capture_output=True, text=True)   # B: the core only; the kit by PYTHONPATH
        if r.returncode != 0:
            raise unittest.SkipTest(f"core install failed: {r.stderr[-500:]}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def env(self, **kw):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.pop("PYTHONNOUSERSITE", None)          # some environments force this ambiently; strip it so site.ENABLE_USER_SITE reflects CPython's real default, not the host's
        e["COLABFOLD_OPT_HOME"] = os.path.dirname(self.opt_copy)
        e.update(kw)
        return e

    def run_sh(self, venv, *args, **env):
        """`run.sh <args>` with the venv's python first on PATH (run.sh calls `python`); venv B carries the package by PYTHONPATH only."""
        e = self.env(**env)
        e["PATH"] = os.path.dirname(self.py[venv]) + os.pathsep + e.get("PATH", "")
        if venv == "B":
            e["PYTHONPATH"] = self.pkg_only                               # importable, not installed: no .pth in site, no hook
        return subprocess.run(["bash", RUN_SH, *args], capture_output=True, text=True, env=e, cwd=self.tmp)   # a neutral cwd: pytest's cwd (<kit>/opt) would put the package on sys.path by itself

    def probe(self, venv, **env):
        code = "from colabfold_opt import stack; import json; r, d = stack.autoload_pth_check(); print(json.dumps([r, d]))"
        r = subprocess.run([self.py[venv], "-c", code], capture_output=True, text=True, env=self.env(**env), cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_e_user_site_install_passes_the_gate(self):
        """`pip install --user` of the kit into a PYTHONUSERBASE user site of a --system-site-packages venv (the user site enabled): the route's
        `python` processes that site's .pth, so the hook is live and the gate passes — a probe under -I would refuse it ('present … but not
        processed'), an over-refusal."""
        venv = os.path.join(self.tmp, "venvC"); ub = os.path.join(self.tmp, "userbase")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=True, capture_output=True)
        py = os.path.join(venv, "bin", "python"); self.py["C"] = py
        r = subprocess.run([py, "-c", "import site; print(site.ENABLE_USER_SITE, site.getusersitepackages())"], capture_output=True, text=True, env=self.env(PYTHONUSERBASE=ub))
        enabled, usersite = r.stdout.split()
        self.assertEqual(enabled, "True", r.stdout)                         # a --system-site-packages venv keeps the user site enabled
        r = subprocess.run([py, "-m", "pip", "install", "-q", "--user", "-e", self.core_dir, "-e", self.opt_copy], capture_output=True, text=True, env=self.env(PYTHONUSERBASE=ub))
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        self.assertTrue(os.path.isfile(os.path.join(usersite, stack.PTH_NAME)), os.listdir(usersite))   # the .pth in the USER site
        reasons, det = self.probe("C", PYTHONUSERBASE=ub)
        self.assertEqual(reasons, [], det); self.assertTrue(det["probe"]["live"], det)
        c = self.run_sh("C", "check", COLABFOLD_OPT="fast", PYTHONUSERBASE=ub)
        self.assertNotIn(stack.HOOK_MODULE + " is not live", c.stdout + c.stderr)   # the gate passes; whatever refuses next is another gate's

    def refused(self, verbs):
        """Every verb on the environment route (COLABFOLD_OPT=fast, no --mode) refuses with the ONE NOT ACTIVE line naming the hook; the lines."""
        want = report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason="").rstrip()
        out = []
        for verb in verbs:
            c = self.run_sh("B", *verb, COLABFOLD_OPT="fast")
            with self.subTest(verb=verb[0]):
                self.assertEqual(c.returncode, report.EXIT_NOT_ACTIVE, (c.stdout[:300], c.stderr[:400]))
                lines = [l for l in c.stderr.splitlines() if l.startswith(report.PREFIX)]
                self.assertEqual(len(lines), 1, c.stderr[-600:])          # … the ONE line
                self.assertTrue(lines[0].startswith(want), lines[0])
                self.assertIn(stack.HOOK_MODULE + " is not live at interpreter start", lines[0]); self.assertIn("run stock silently", lines[0])
                out.append(lines[0])
        return out

    VERBS = ([["check"], ["pred", A3M, "out", "--data", "."]])

    def test_a_hook_live_passes_the_gate(self):
        reasons, det = self.probe("A")
        self.assertEqual(reasons, [], det)
        self.assertTrue(det["probe"]["live"], det)
        self.assertTrue(det["probe"]["file"].startswith(os.path.realpath(self.opt_copy)) or det["probe"]["file"].startswith(self.opt_copy), det)
        c = self.run_sh("A", "check", COLABFOLD_OPT="fast")
        self.assertNotIn(stack.HOOK_MODULE + " is not live", c.stdout + c.stderr)   # whatever else refuses in a test venv (the pinned upstream absent), never the hook

    def test_b_no_pth_in_site_refuses_by_name_before_anything_runs(self):
        r = subprocess.run([self.py["B"], "-c", "import colabfold_opt; print(colabfold_opt.__file__)"], capture_output=True, text=True,
                           env=self.env(PYTHONPATH=self.pkg_only), cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])                  # importable …
        self.assertIn("pkgonly/colabfold_opt", r.stdout)
        reasons, det = self.probe("B", PYTHONPATH=self.pkg_only)
        self.assertEqual(len(reasons), 1, det); self.assertFalse(det["probe"]["live"], det)
        self.assertIn("absent from the searched sites", reasons[0])          # … but the hook is not live: the diagnostic
        lines = self.refused(self.VERBS)
        self.assertTrue(all("absent from the searched sites" in l for l in lines), lines)

    def test_c_pth_copied_beside_pythonpath_is_present_but_not_processed(self):
        shutil.copy(os.path.join(OPT, stack.PTH_NAME), os.path.join(self.pkg_only, stack.PTH_NAME))   # beside the PYTHONPATH entry: site.py never reads it there
        try:
            reasons, det = self.probe("B", PYTHONPATH=self.pkg_only)
            self.assertEqual(len(reasons), 1, det); self.assertFalse(det["probe"]["live"], det)
            self.assertIn("present beside", reasons[0]); self.assertIn("but not processed", reasons[0])
            lines = self.refused(self.VERBS[:1])
            self.assertIn("present beside", lines[0])
        finally:
            os.remove(os.path.join(self.pkg_only, stack.PTH_NAME))

    def test_d_mode_route_is_ungated_and_off_is_exempt(self):
        c = self.run_sh("B", "check", "--mode", "fast")
        self.assertNotIn(stack.HOOK_MODULE, c.stdout + c.stderr)           # --mode <kit mode>: un-gated (the launcher sets the mode for the model process; the manifest verdict judges the launch)
        for e in ({"COLABFOLD_OPT": "off"}, {}):
            c = self.run_sh("B", "check", **e)
            self.assertNotIn(stack.HOOK_MODULE, c.stdout + c.stderr)       # COLABFOLD_OPT=off / nothing set: no hook needed, never the hook


if __name__ == "__main__":
    unittest.main()
