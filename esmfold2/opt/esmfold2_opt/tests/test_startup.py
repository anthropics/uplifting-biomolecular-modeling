"""Packaging and interpreter start: `pip install -e esmfold2/opt` works in a fresh venv WITHOUT torch or the upstream packages, ships
esmfold2_opt_autoload.pth into site-packages, and that .pth leaves `python -c pass` unchanged — no torch, no upstream, nothing of the
core with ESMFOLD2_OPT unset or off, a sub-millisecond autoload import — with and without ESMFOLD2_OPT set. The startup matrix on the
real route (the installed .pth, a fresh interpreter each): an unknown mode or an undeclared ESMFOLD2_OPT_* name exits 3 with the
kit's NOT ACTIVE line and no traceback; a known mode arms the core's finder and folds its case; a bare find_spec probe leaves it
armed and the import that follows fires it; off / unset install nothing; `python -S` is untouched. Also builds the wheel and checks
the .pth sits at its root with a RECORD entry, and that the console script answers.

Runs in a throwaway venv created from this interpreter (~15-40 s; the build backend's setuptools comes through pip's build
isolation); venv/pip failures are test failures, not skips."""
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from esmfold2_opt import stack

N_RUNS = 15
WALL_BUDGET_FACTOR = 3.0              # a bare `python -c pass` after the editable install vs before: pip's editable finders for the core and the kit are part of every start
                                     # (0.013 s -> 0.029 s on the CPU test box); the kit's own .pth imports nothing unless ESMFOLD2_OPT names a mode
IMPORTTIME_BUDGET_US = 30000         # cumulative -X importtime for esmfold2_opt + esmfold2_opt._autoload under a set variable: the core gate + producer census + the core's
                                     # finder, resolved through pip's editable finder (the install form pinned); 20.0 ms measured on the CPU test box, ceiling 1.5x that
PTH = "esmfold2_opt_autoload.pth"
FINDER_MODULE = "opt_core.autoload"  # the finder's class comes from the core; the kit's _autoload.py holds the spec
TRIGGER = "transformers.models.esmfold2"
UNKNOWN_MODE_LINE = "[esmfold2-opt] NOT ACTIVE: unknown ESMFOLD2_OPT='turbo' (expected exact|fast|off|big)"
FINDERS = f"[type(f).__name__ for f in sys.meta_path if type(f).__module__ == {FINDER_MODULE!r}]"


def _wall(python, env, n=N_RUNS):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        subprocess.run([python, "-c", "pass"], env=env, check=True)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


class TestStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.venv = os.path.join(cls.tmp, "venv")
        r = subprocess.run([sys.executable, "-m", "venv", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp,
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            if k in os.environ:
                cls.env[k] = os.environ[k]
        cls.wall_before = _wall(cls.python, cls.env)
        # an EDITABLE install of the core and the kit — the deployment form pinned (README §install, run.sh): the kit's core gate reads
        # opt/pyproject.toml above the package and the core's __version__ literal beside its package, both of which only the source trees carry
        # (a wheel install refuses by name at interpreter start: reason=core_pin_unreadable); pip's editable finder is part of every start
        from opt_core.gates import core_pin                                    # the kit is installed as the core plus itself: the core at the pinned path
        core_dir = core_pin(os.path.join(stack.opt_home(), "pyproject.toml"))["abs_path"]
        r = subprocess.run([cls.python, "-m", "pip", "install", "-q", "-e", core_dir, "-e", stack.opt_home()], env=cls.env, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("pip install failed: " + r.stderr[-1500:])
        cls.site = subprocess.run([cls.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=cls.env,
                                  capture_output=True, text=True, check=True).stdout.strip()
        cls.stub = os.path.join(cls.tmp, "stub")                                  # an empty trigger package: the finder fires on its import, nothing else is needed
        os.makedirs(os.path.join(cls.stub, *TRIGGER.split(".")))
        for i in range(1, len(TRIGGER.split(".")) + 1):
            open(os.path.join(cls.stub, *TRIGGER.split(".")[:i], "__init__.py"), "w").close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_no_torch_no_upstream_in_the_venv(self):
        r = subprocess.run([self.python, "-c", "import importlib.util as u; print([u.find_spec(m) is None for m in ('torch','esm','transformers')])"],
                           env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "[True, True, True]")

    def test_run_sh_hook_live_gate_env_route_two_venvs(self):
        """RF-6 (D77 / D77.2): the ENV route — ESMFOLD2_OPT=<kit mode>, no --mode on the command line — is gated on the hook being LIVE in a
        fresh interpreter (esmfold2_opt._autoload in sys.modules at start: the site-processed .pth's effect). venv A (this class's editable
        install, the .pth in site) passes; venv B (the package importable via PYTHONPATH, no install) is refused by name — exit 3, one line
        naming the hook, the interpreter and the DIAGNOSTIC: B1 = the .pth beside the PYTHONPATH entry (present but not processed), B2 = a
        package copy without the .pth (absent from the searched sites). The --mode route is not gated (in B `--mode exact` reaches the
        next gate, the pinned forks, with no gate line); --mode off / ESMFOLD2_OPT=off exempt. Mutation-checked: remove the gate from run.sh
        and B1 runs on to the pinned-forks refusal (a different line, a different code)."""
        from esmfold2_opt import pth_gate
        self.assertEqual(pth_gate.PTH_NAME, PTH)
        run_sh = os.path.join(os.path.dirname(stack.opt_home()), "run.sh")
        venv_b = os.path.join(self.tmp, "venv_b")
        subprocess.run([sys.executable, "-m", "venv", venv_b], capture_output=True, text=True, check=True)
        from opt_core.gates import core_pin
        core_src = core_pin(os.path.join(stack.opt_home(), "pyproject.toml"))["abs_path"]
        pkg_copy = os.path.join(self.tmp, "pkg_copy"); os.makedirs(pkg_copy)
        shutil.copytree(os.path.join(stack.opt_home(), "esmfold2_opt"), os.path.join(pkg_copy, "esmfold2_opt"), ignore=shutil.ignore_patterns("__pycache__", "tests"))
        base_b = dict(self.env, PATH=os.path.join(venv_b, "bin") + os.pathsep + self.env["PATH"], HF_HOME=self.tmp)   # a weights root: the weights gate passes, this gate is what decides
        for label, pythonpath, diag in (("B1", stack.opt_home(), "present at"), ("B2", pkg_copy, "absent from the searched sites")):
            env_b = dict(base_b, PYTHONPATH=pythonpath + os.pathsep + core_src)
            for verb in ("check", "pred"):
                rb = subprocess.run(["bash", run_sh, verb, "--variant", "fast"], env=dict(env_b, ESMFOLD2_OPT="exact"), capture_output=True, text=True, timeout=120, cwd=self.tmp)
                self.assertEqual(rb.returncode, 3, (label, verb, rb.stderr[-600:]))
                lines = [l for l in rb.stderr.splitlines() if l.strip()]
                self.assertEqual(len(lines), 1, rb.stderr[-600:])
                self.assertTrue(lines[0].startswith("[esmfold2-opt] NOT ACTIVE: the autoload hook esmfold2_opt._autoload is not live"), lines[0])
                self.assertIn("diagnostic: " + (PTH + " is " + diag if label == "B2" else PTH + " is " + diag), lines[0]); self.assertIn("pip install -e", lines[0])
            rb = subprocess.run(["bash", run_sh, "pred", "--mode", "exact", "--variant", "fast"], env=env_b, capture_output=True, text=True, timeout=120, cwd=self.tmp)
            self.assertNotIn("autoload hook", rb.stderr, (label, rb.stderr[-400:]))          # the --mode route: not this gate (the next refusal here is the pinned forks')
            self.assertIn("pinned upstream forks", rb.stderr, (label, rb.stderr[-400:]))
            for env_off in (dict(env_b, ESMFOLD2_OPT="off"), env_b):
                rb = subprocess.run(["bash", run_sh, "pred", "--variant", "fast", *(["--mode", "off"] if "ESMFOLD2_OPT" not in env_off else [])], env=env_off, capture_output=True, text=True, timeout=120, cwd=self.tmp)
                self.assertNotIn("autoload hook", rb.stderr, (label, rb.stderr[-400:]))      # the stock route: exempt
        ra = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=dict(self.env, HF_HOME=self.tmp, ESMFOLD2_OPT="exact"), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertNotIn("autoload hook", ra.stderr, ra.stderr[-600:])                      # venv A: the hook is live, the gate passes (the next refusal, if any, is another gate's)
        # C: a USER-SITE editable install (pip install --user -e under PYTHONUSERBASE, the user site enabled: a venv with system site-packages) — the .pth is
        # processed from the user site at the route's interpreter start, so the gate PASSES (the probe runs the route's own `python`, not -I)
        venv_c = os.path.join(self.tmp, "venv_c"); userbase = os.path.join(self.tmp, "userbase")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv_c], capture_output=True, text=True, check=True)
        env_c = dict(self.env, PATH=os.path.join(venv_c, "bin") + os.pathsep + self.env["PATH"], PYTHONUSERBASE=userbase, HF_HOME=self.tmp)
        rc_ = subprocess.run([os.path.join(venv_c, "bin", "python"), "-m", "pip", "install", "-q", "--user", "-e", core_src, "-e", stack.opt_home()], env=env_c, capture_output=True, text=True)
        self.assertEqual(rc_.returncode, 0, "user-site pip install failed: " + rc_.stderr[-800:])
        user_site = subprocess.run([os.path.join(venv_c, "bin", "python"), "-c", "import site; print(site.getusersitepackages())"], env=env_c, capture_output=True, text=True).stdout.strip()
        self.assertTrue(os.path.isfile(os.path.join(user_site, PTH)), (user_site, sorted(os.listdir(user_site)) if os.path.isdir(user_site) else None))
        rg = subprocess.run([os.path.join(venv_c, "bin", "python"), "-m", "esmfold2_opt.pth_gate"], env=env_c, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual((rg.returncode, rg.stderr), (0, ""), "the user-site install must pass the gate (the probe is the route's own python, the user site enabled)")
        rcheck = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=dict(env_c, ESMFOLD2_OPT="exact"), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertNotIn("autoload hook", rcheck.stderr, rcheck.stderr[-600:])
        rno = subprocess.run([os.path.join(venv_c, "bin", "python"), "-m", "esmfold2_opt.pth_gate"], env=dict(env_c, PYTHONNOUSERSITE="1", PYTHONPATH=stack.opt_home() + os.pathsep + core_src), capture_output=True, text=True, cwd=self.tmp)
        sys_sites = subprocess.run([os.path.join(venv_c, "bin", "python"), "-c", "import site; print(chr(10).join(site.getsitepackages()))"], env=dict(env_c, PYTHONNOUSERSITE="1"), capture_output=True, text=True).stdout.split()
        if any(os.path.isfile(os.path.join(d, PTH)) for d in sys_sites):   # venv C inherits the system site: when THIS interpreter's system site already carries the hook
            self.skipTest(f"the interpreter's system site already carries {PTH} ({sys_sites}): with the user site disabled the hook is still live from the system site, "
                          "so the user-site-disabled refusal has no premise here (run from an interpreter without the kit installed system-wide to exercise it)")
        self.assertEqual(rno.returncode, 3, rno.stderr[-400:]); self.assertIn("user site", rno.stderr)   # the same install with the user site disabled: refused, the diagnostic names the user site
        rg = subprocess.run([self.python, "-m", "esmfold2_opt.pth_gate"], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual((rg.returncode, rg.stderr), (0, ""))
        self.assertTrue(pth_gate.hook_live(self.python))

    def _mutated_tree(self, drop_marker: str, label: str) -> str:
        """A tree beside this one whose run.sh lacks the block starting at the line holding `drop_marker` up to its closing `fi` (stock/,
        opt/, configs/ symlinked: run.sh resolves $HERE to the copy)."""
        tree = os.path.dirname(stack.opt_home()); mut = os.path.join(self.tmp, label); os.makedirs(mut, exist_ok=True)
        for d in ("stock", "opt", "configs"):
            if os.path.exists(os.path.join(tree, d)) and not os.path.exists(os.path.join(mut, d)):
                os.symlink(os.path.join(tree, d), os.path.join(mut, d))
        lines = open(os.path.join(tree, "run.sh"), encoding="utf-8").read().split("\n")
        i = next(n for n, l in enumerate(lines) if drop_marker in l); j = next(n for n in range(i, len(lines)) if lines[n].strip() == "fi")
        open(os.path.join(mut, "run.sh"), "w", encoding="utf-8").write("\n".join(lines[:i] + lines[j + 1:]))
        return os.path.join(mut, "run.sh")

    def test_env_route_probes_survive_the_core_gate_at_interpreter_start(self):
        """A13: with ESMFOLD2_OPT exported (the env route) the INSTALLED .pth runs the kit's core gate at every interpreter start, so a core the
        gate refuses kills each `python` run.sh starts. run.sh's silenced importability probes run with the variables removed (`env -u`) and its
        liveness probe is not silenced: the outcome is the gate's own line — `NOT ACTIVE: reason=core_mismatch …`, exit 3 — never the probes'
        'not importable' / 'predates the autoload gate' / 'hook is not live' diagnoses and never silence. Same for `source configs/h100.env`."""
        from esmfold2_opt.tests.test_core_missing import shadow_core
        run_sh = os.path.join(os.path.dirname(stack.opt_home()), "run.sh"); env_file = os.path.join(os.path.dirname(stack.opt_home()), "configs", "h100.env")
        stale = shadow_core(os.path.join(self.tmp, "stale_core_a13"), version="0.0.1")
        env = dict(self.env, PYTHONPATH=stale, ESMFOLD2_OPT="exact", HF_HOME=self.tmp)                # venv A: kit + core installed editable, the .pth live; the stale core first
        probes = ((["bash", run_sh, "check", "--variant", "fast"], "run.sh env route"),
                  (["bash", "-c", f'source "{env_file}" || exit $?; echo REACHED'], "configs/h100.env"))
        for cmd, what in probes:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180, cwd=self.tmp)
            lines = [l for l in r.stderr.splitlines() if l.strip()]
            self.assertEqual(r.returncode, 3, (what, r.stderr[-600:]))
            self.assertTrue(lines and all(l.startswith("[esmfold2-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned ") for l in lines), (what, lines))
            for false_diag in ("predates the autoload gate", "is not installed", "is not live", "not importable"):
                self.assertNotIn(false_diag, r.stderr, what)
            self.assertNotIn("REACHED", r.stdout, what)
        ok = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=dict(self.env, ESMFOLD2_OPT="exact", HF_HOME=self.tmp), capture_output=True, text=True, timeout=180, cwd=self.tmp)
        self.assertNotIn("reason=core_", ok.stderr, ok.stderr[-400:])                                        # the pinned core: the gate passes at every start on the same route

    def test_run_sh_hook_gate_mutation_stale_copy_and_older_package(self):
        """The RF-6 gate's negative space. MUTATION: with the gate block cut out of run.sh, venv B's env-route call runs on to the NEXT
        refusal (the pinned forks) instead of the hook line — the block is what refuses. STALE COPY: the package importable from PYTHONPATH
        beside a stale .pth in the venv's site (content not the hook line: the import check passes, the hook is not live) -> refused, the
        diagnostic names the stale copy. OLDER PACKAGE: a package copy without pth_gate.py (an install that predates the gate) -> refused
        by name (exit 3), never `No module named` exit 1."""
        run_sh = os.path.join(os.path.dirname(stack.opt_home()), "run.sh")
        from opt_core.gates import core_pin
        core_src = core_pin(os.path.join(stack.opt_home(), "pyproject.toml"))["abs_path"]
        venv_b = os.path.join(self.tmp, "venv_mut"); subprocess.run([sys.executable, "-m", "venv", venv_b], capture_output=True, text=True, check=True)
        env_b = dict(self.env, PATH=os.path.join(venv_b, "bin") + os.pathsep + self.env["PATH"], HF_HOME=self.tmp, PYTHONPATH=stack.opt_home() + os.pathsep + core_src, ESMFOLD2_OPT="exact")
        rb = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=env_b, capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(rb.returncode, 3, rb.stderr[-600:]); self.assertIn("autoload hook", rb.stderr)                    # the gate, intact
        mutated = self._mutated_tree("RF-6 (env route only", "tree_no_gate")
        rm = subprocess.run(["bash", mutated, "check", "--variant", "fast"], env=env_b, capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertNotIn("autoload hook", rm.stderr, rm.stderr[-600:])                                                       # gate cut: the next refusal (the pinned forks), not this one
        self.assertIn("pinned upstream forks", rm.stderr, rm.stderr[-600:])
        site_b = subprocess.run([os.path.join(venv_b, "bin", "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=env_b, capture_output=True, text=True).stdout.strip()
        open(os.path.join(site_b, PTH), "w").write("# a stale copy: not the hook line\n")                                   # a .pth of the right name, the wrong content
        rs_ = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=env_b, capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(rs_.returncode, 3, rs_.stderr[-600:])
        line = [l for l in rs_.stderr.splitlines() if "NOT ACTIVE" in l][0]
        self.assertIn("diagnostic: a stale copy at " + os.path.join(site_b, PTH), line); self.assertIn("expected the text of ", line); self.assertIn("esmfold2_opt_autoload.pth", line)
        os.remove(os.path.join(site_b, PTH))
        old_pkg = os.path.join(self.tmp, "pkg_old"); os.makedirs(old_pkg)
        shutil.copytree(os.path.join(stack.opt_home(), "esmfold2_opt"), os.path.join(old_pkg, "esmfold2_opt"), ignore=shutil.ignore_patterns("__pycache__", "tests", "pth_gate.py"))
        ro = subprocess.run(["bash", run_sh, "check", "--variant", "fast"], env=dict(env_b, PYTHONPATH=old_pkg + os.pathsep + core_src), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(ro.returncode, 3, ro.stderr[-600:]); self.assertIn("[esmfold2-opt] NOT ACTIVE: the installed esmfold2_opt (" + os.path.join(old_pkg, "esmfold2_opt") + ") predates the autoload gate", ro.stderr)
        self.assertNotIn("No module named", ro.stderr)

    def test_run_sh_weights_gate_refuses_by_name_and_defaults_to_offline(self):
        """4e-bp frozen weights: pred/warm refuse by name (the kit's NOT ACTIVE line, exit 3) when HF_HOME is unset or names no
        directory, on the stock route too; run.sh exports HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 unless the caller set them (xtrace). The
        package's data_path_gate then checks every pinned file under the root (test_registry_settings_det)."""
        run_sh = os.path.join(os.path.dirname(stack.opt_home()), "run.sh")
        base = {k: v for k, v in self.env.items() if k not in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        for verb in ("pred", "warm"):
            r = subprocess.run(["bash", run_sh, verb, "--mode", "off" if verb == "pred" else "fast", "--variant", "fast"], env=base, capture_output=True, text=True, timeout=120, cwd=self.tmp)
            self.assertEqual(r.returncode, 3, (verb, r.stderr[-400:]))
            self.assertTrue(r.stderr.strip().startswith("[esmfold2-opt] NOT ACTIVE: HF_HOME (the weights root"), (verb, r.stderr[-400:]))
        r = subprocess.run(["bash", run_sh, "pred", "--mode", "off", "--variant", "fast"], env=dict(base, HF_HOME=os.path.join(self.tmp, "nope")), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr[-400:]); self.assertTrue(r.stderr.strip().startswith("[esmfold2-opt] NOT ACTIVE: HF_HOME=" + os.path.join(self.tmp, "nope") + " does not exist"), r.stderr[-400:])
        tr_ = subprocess.run(["bash", "-x", run_sh, "pred", "--mode", "off", "--variant", "fast"], env=dict(base, HF_HOME=self.tmp), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertIn("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1", tr_.stderr, tr_.stderr[-800:])                    # the default: both set to 1
        self.assertNotIn("NOT ACTIVE: HF_HOME", tr_.stderr)                                                                  # an existing root passes run.sh's check (the next gate is the pinned forks)
        tr2 = subprocess.run(["bash", "-x", run_sh, "pred", "--mode", "off", "--variant", "fast"], env=dict(base, HF_HOME=self.tmp, HF_HUB_OFFLINE="0"), capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertIn("export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=1", tr2.stderr, tr2.stderr[-800:])                    # unless set: the caller's value stands

    def test_pth_installed(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site, PTH)), sorted(os.listdir(self.site)))
        from esmfold2_opt import pth_gate
        self.assertEqual(open(os.path.join(self.site, PTH)).read().strip(), pth_gate.pth_text())          # the site copy is the tree's generated text (opt/_build_backend.py pth_text)
        self.assertIn("import esmfold2_opt._autoload", pth_gate.pth_text()); self.assertTrue(pth_gate.pth_text().startswith("# opt_core autoload guard: package=esmfold2_opt env=ESMFOLD2_OPT tag=esmfold2-opt exit=3"))

    def _py(self, code, **env):
        return subprocess.run([self.python, "-c", code], env=dict(self.env, **env), capture_output=True, text=True)

    def test_interpreter_start_unchanged(self):
        after = _wall(self.python, self.env)
        self.assertLessEqual(after, WALL_BUDGET_FACTOR * self.wall_before, f"start-up {self.wall_before:.4f}s -> {after:.4f}s (budget {WALL_BUDGET_FACTOR}x)")
        loaded = "import sys; print(sorted(m for m in sys.modules if m.startswith(('esmfold2_opt','opt_core','torch','esm','transformers'))))"
        for env in ({}, {"ESMFOLD2_OPT": "off"}, {"ESMFOLD2_OPT": " OFF "}, {"ESMFOLD2_OPT_FORCE": "1"}):
            r = self._py(loaded, **env)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, "['esmfold2_opt', 'esmfold2_opt._autoload']"), (env, r.stderr[-500:]))   # nothing of the core
        r = self._py("import sys; print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch','esm','transformers')), " + FINDERS + ")",
                     ESMFOLD2_OPT="fast", ESMFOLD2_VARIANT="fast")
        self.assertEqual(r.stdout.strip(), "[] ['Finder']", r.stderr[-500:])   # armed, nothing heavy imported
        self.assertNotIn("opt_core.gates", self._py("import sys; print(sorted(sys.modules))", ESMFOLD2_OPT="fast").stdout)   # the finder module alone

    def test_known_mode_arms_the_finder_case_folded(self):
        """A declared mode arms the core's finder with the kit's selection; the mode is case-folded (ESMFOLD2_OPT=EXACT selects exact)."""
        for raw, mode in (("exact", "exact"), ("FAST", "fast"), (" Exact ", "exact")):
            r = self._py("import sys; f = [f for f in sys.meta_path if type(f).__module__ == %r]; print(len(f), f[0].mode, f[0].armed, f[0].spec.package, f[0].spec.triggers)" % FINDER_MODULE, ESMFOLD2_OPT=raw)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertEqual(r.stdout.strip(), f"1 {mode} True esmfold2_opt ('transformers.models.esmfold2', 'esm.models.esmfold2')", raw)

    def test_probe_survives_and_the_trigger_import_fires(self):
        """On the real route (the installed .pth, a trigger package on the path) a bare find_spec probe leaves the finder armed and the import
        that follows fires it: the activation runs, refuses by name in this venv (no kit, no GPU, no variant), and the process exits 3 with
        the kit's NOT ACTIVE line and no traceback — the program never continues under ESMFOLD2_OPT on a refused activation."""
        env = {"PYTHONPATH": self.stub}
        fire = f"import {TRIGGER}; print('REACHED')"
        probe = f"import importlib.util as u, sys; assert u.find_spec({TRIGGER!r}) is not None; " + FINDERS + "[0]; " + fire
        for code in (fire, probe):
            r = self._py(code, ESMFOLD2_OPT="exact", **env)
            self.assertEqual(r.returncode, 3, (code, r.stderr[-1200:]))
            self.assertIn("[esmfold2-opt] NOT ACTIVE: ", r.stderr); self.assertNotIn("Traceback", r.stderr); self.assertNotIn("REACHED", r.stdout)
        r = self._py(fire + "; import sys; print(" + FINDERS + ")", **env)                  # unset: the trigger imports untouched, no finder
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "REACHED\n[]"), r.stderr[-500:])
        r = self._py(fire + "; import sys; print(" + FINDERS + ")", ESMFOLD2_OPT="off", **env)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "REACHED\n[]"), r.stderr[-500:]); self.assertNotIn("[esmfold2-opt]", r.stderr)

    def test_pending_mode_refuses_at_the_trigger(self):
        """A mode the hook declares ahead of modes.MODES (test_merge_locks.TABLE_PENDING): the finder arms and the trigger import fires
        enable(<mode>, strict=True), which refuses by name with the kit's NOT ACTIVE line and exit 3 — never a traceback (a ValueError
        escaping the import would be swallowed by a host program's `except Exception` where SystemExit is not)."""
        from esmfold2_opt.tests.test_merge_locks import TABLE_PENDING
        fire = f"import {TRIGGER}; print('REACHED')"
        for mode in TABLE_PENDING + tuple(" " + m.capitalize() + " " for m in TABLE_PENDING):
            r = self._py(fire, ESMFOLD2_OPT=mode, PYTHONPATH=self.stub)
            self.assertEqual(r.returncode, 3, (mode, r.stderr[-1200:]))
            self.assertIn(f"[esmfold2-opt] NOT ACTIVE: unknown mode {mode.strip().lower()!r}; expected one of ", r.stderr, (mode, r.stderr[-800:]))
            self.assertNotIn("Traceback", r.stderr); self.assertNotIn("REACHED", r.stdout)

    def test_python_S_is_untouched(self):
        """`python -S` processes no .pth: nothing is installed and nothing refuses, whatever ESMFOLD2_OPT says."""
        r = subprocess.run([self.python, "-S", "-c", "import sys; print('REACHED', " + FINDERS + ", 'esmfold2_opt' in sys.modules)"],
                           env=dict(self.env, ESMFOLD2_OPT="turbo"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "REACHED [] False"), r.stderr[-500:]); self.assertNotIn("[esmfold2-opt]", r.stderr)

    def test_importtime_budget(self):
        r = subprocess.run([self.python, "-X", "importtime", "-c", "pass"], env=dict(self.env, ESMFOLD2_OPT="fast"), capture_output=True, text=True, check=True)
        us = {m.group(2).strip(): int(m.group(1)) for m in re.finditer(r"import time:\s+\d+ \|\s+(\d+) \|(.*)", r.stderr)}
        total = us.get("esmfold2_opt", 0) + us.get("esmfold2_opt._autoload", 0)
        self.assertGreater(total, 0, r.stderr[-500:])
        print(f"importtime esmfold2_opt+_autoload = {total} us (budget {IMPORTTIME_BUDGET_US})")
        self.assertLess(total, IMPORTTIME_BUDGET_US)

    def test_unknown_mode_exits_not_active(self):
        """An unknown selection under the variable never runs stock silently: the kit's NOT ACTIVE line, exit 3 at interpreter start (the
        kit's own code, not the interpreter's init status 1), no traceback."""
        r = self._py("print('REACHED')", ESMFOLD2_OPT="turbo")
        self.assertEqual(r.returncode, 3, r.stderr[-800:])
        self.assertEqual(r.stderr.strip().splitlines()[-1], UNKNOWN_MODE_LINE); self.assertNotIn("Traceback", r.stderr); self.assertNotIn("REACHED", r.stdout)
        r = self._py("print('REACHED')", ESMFOLD2_OPT="turbo", PYTHONPATH=self.stub)   # the same with a trigger on the path: refused before any import
        self.assertEqual((r.returncode, r.stdout), (3, ""), r.stderr[-800:])


    def test_undeclared_switch_name_exits_not_active(self):
        """A mistyped name under the package prefix (ESMFOLD2_OPT_MODE=exact) is refused by name, never silently ignored."""
        r = subprocess.run([self.python, "-c", "print('REACHED')"], env=dict(self.env, ESMFOLD2_OPT_MODE="exact"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-800:])
        self.assertIn("[esmfold2-opt] NOT ACTIVE: undeclared ESMFOLD2_OPT_MODE", r.stderr); self.assertNotIn("REACHED", r.stdout)
        r = subprocess.run([self.python, "-c", "print('REACHED')"], env=dict(self.env, ESMFOLD2_OPT_FORCE="1"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])                       # a declared switch alone installs nothing and refuses nothing

    def test_console_script_and_module(self):
        exe = os.path.join(self.venv, "bin", "esmfold2-opt")
        self.assertTrue(os.path.isfile(exe))
        r = subprocess.run([exe, "--help"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("commands", r.stdout); self.assertIn("pred", r.stdout)
        r = subprocess.run([self.python, "-m", "esmfold2_opt"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        r = subprocess.run([self.python, "-m", "esmfold2_opt", "check", "--mode", "off", "--variant", "fast"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("NOT ACTIVE: mode off", r.stderr)

    def test_wheel_ships_pth_at_root(self):
        wd = os.path.join(self.tmp, "wheels")
        r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", "--no-deps", "-w", wd, stack.opt_home()], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        whl = glob.glob(os.path.join(wd, "esmfold2_opt-*.whl"))
        self.assertEqual(len(whl), 1, whl)
        with zipfile.ZipFile(whl[0]) as zf:
            names = zf.namelist()
            self.assertIn(PTH, names)
            record = next(n for n in names if n.endswith(".dist-info/RECORD"))
            self.assertIn(PTH + ",sha256=", zf.read(record).decode())
            self.assertEqual(len([n for n in names if n == record]), 1)


if __name__ == "__main__":
    unittest.main()
