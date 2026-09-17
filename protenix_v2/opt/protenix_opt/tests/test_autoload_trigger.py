"""The lazy autoload (the core's finder on this kit's AutoloadSpec, `_autoload.install`) fires once, after the trigger package's body,
and before `runner.inference` starts executing.

Stub packages mirror the stock import chain (`runner.batch_inference` -> protenix.config -> `runner.inference` ->
protenix.model.protenix -> protenix.model.modules.pairformer, with InferenceRunner defined after those imports). The
activation callback does what ptx_lazy_init.install() and the DEADSKIP hook do — `import runner.inference; RI.InferenceRunner`
— so a trigger that fires from inside runner.inference's own import fails here exactly as it would in the kit."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from protenix_opt import _autoload, stack, tp

STUBS = {
    "protenix/__init__.py": "from .version import __version__\n",
    "protenix/version.py": "__version__ = '2.0.0'\n",
    "protenix/config/__init__.py": "",
    "protenix/config/config.py": "def parse_configs():\n    pass\n",
    "protenix/data/__init__.py": "",
    "protenix/data/tokenizer.py": "TOKENS = 1\n",
    "protenix/model/__init__.py": "",
    "protenix/model/protenix.py": "from protenix.model.modules.pairformer import PairformerBlock\n",
    "protenix/model/modules/__init__.py": "",
    "protenix/model/modules/pairformer.py": "class PairformerBlock:\n    pass\n",
    "runner/__init__.py": "",
    "runner/batch_inference.py": "from protenix.config.config import parse_configs\nfrom runner.inference import InferenceRunner\n",
    "runner/inference.py": "from protenix.model.protenix import PairformerBlock\n\nclass InferenceRunner:\n    pass\n",
}
FOREIGN = {"runner/__init__.py": "", "runner/thing.py": "X = 1\n"}

SCRIPT = textwrap.dedent("""
    import json, os, sys
    sys.path.insert(0, os.environ["STUBS"]); sys.path.insert(0, os.environ["OPT"])
    import protenix_opt, protenix_opt._autoload as A
    if os.environ.get("TRIGGERS"):
        A.TRIGGERS = tuple(os.environ["TRIGGERS"].split(","))
    log = []
    def fake_enable(mode, strict=False, trigger=None):
        log.append({"mode": mode, "trigger": trigger, "runner_inference_half_initialised": "runner.inference" in sys.modules})
        import runner.inference as RI            # ptx_lazy_init.install() L68 / ptx_trunk2_levers._apply_deadskip L1512
        RI.InferenceRunner
        log[-1]["ok"] = True
    protenix_opt.enable = fake_enable
    f = A.install({"PROTENIX_OPT": os.environ.get("MODE", "exact")})
    err = None
    try:
        exec(os.environ["CHAIN"])
    except Exception as e:
        err = repr(e)
    print(json.dumps({"log": log, "err": err, "installed": f is not None, "armed": bool(f and f.armed), "fired": f.fired if f else None,
                      "in_meta_path": bool(f and f in sys.meta_path), "loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("runner", "protenix"))}))
""")


def _write(tree, files):
    for rel, body in files.items():
        p = os.path.join(tree, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(body)


class TestAutoloadTrigger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.stubs = os.path.join(cls.tmp, "stubs"); _write(cls.stubs, STUBS)
        cls.foreign = os.path.join(cls.tmp, "foreign"); _write(cls.foreign, FOREIGN)
        cls.script = os.path.join(cls.tmp, "scenario.py")
        with open(cls.script, "w") as fh:
            fh.write(SCRIPT)

    def run_chain(self, chain, triggers=None, stubs=None, mode="exact"):
        env = {"PATH": os.environ.get("PATH", ""), "STUBS": stubs or self.stubs, "OPT": stack.opt_home(), "CHAIN": chain, "MODE": mode}
        if triggers:
            env["TRIGGERS"] = ",".join(triggers)
        r = subprocess.run([sys.executable, "-S", self.script], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_cli_chain_fires_on_runner_before_runner_inference(self):
        out = self.run_chain("import runner.batch_inference")
        self.assertIsNone(out["err"])
        self.assertEqual(len(out["log"]), 1, "fires exactly once")
        self.assertEqual(out["log"][0]["trigger"], "runner")
        self.assertFalse(out["log"][0]["runner_inference_half_initialised"])
        self.assertTrue(out["log"][0]["ok"])
        self.assertFalse(out["in_meta_path"], "the finder removes itself after firing")
        self.assertIn("protenix.model.modules.pairformer", out["loaded"])

    def test_library_chain_fires_on_protenix_model(self):
        out = self.run_chain("import protenix.model.protenix")
        self.assertIsNone(out["err"])
        self.assertEqual([e["trigger"] for e in out["log"]], ["protenix.model"])
        self.assertTrue(out["log"][0]["ok"])

    def test_mode_is_passed_through(self):
        out = self.run_chain("import runner", mode="fast")
        self.assertEqual(out["log"][0]["mode"], "fast")

    def test_data_only_import_does_not_fire(self):
        out = self.run_chain("import protenix.data.tokenizer")
        self.assertEqual(out["log"], []); self.assertTrue(out["armed"]); self.assertTrue(out["in_meta_path"])
        self.assertNotIn("protenix.model", out["loaded"])

    def test_foreign_runner_package_does_not_fire(self):
        out = self.run_chain("import runner.thing", stubs=self.foreign)
        self.assertEqual(out["log"], []); self.assertTrue(out["armed"])

    def test_pairformer_trigger_breaks_the_runner_inference_levers(self):
        """Why the trigger is not protenix.model.modules.pairformer: it is first imported from inside runner.inference's body, so an
        activation there meets a half-initialised runner.inference (no InferenceRunner yet) — and the core's finder turns the activation's
        exception into the kit's NOT ACTIVE line and exit 3 (fail-closed), never a stock run."""
        env = {"PATH": os.environ.get("PATH", ""), "STUBS": self.stubs, "OPT": stack.opt_home(), "CHAIN": "import runner.batch_inference", "MODE": "exact",
               "TRIGGERS": "protenix.model.modules.pairformer"}
        r = subprocess.run([sys.executable, "-S", self.script], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-600:])
        self.assertIn("[protenix-opt] NOT ACTIVE: enable() raised at protenix.model.modules.pairformer: AttributeError", r.stderr)
        self.assertIn("InferenceRunner", r.stderr)                       # runner.inference half-initialised at that trigger
        self.assertNotIn("Traceback", r.stderr)

    def test_activation_failure_exits_3_never_runs_stock(self):
        """PROTENIX_OPT=exact with a mode that cannot be activated: the core's NOT ACTIVE line, then exit 3 — the trigger import never completes."""
        script = os.path.join(self.tmp, "fail.py")
        with open(script, "w") as fh:
            fh.write(textwrap.dedent("""
                import os, sys
                sys.path.insert(0, os.environ["STUBS"]); sys.path.insert(0, os.environ["OPT"])
                import protenix_opt, protenix_opt._autoload as A
                def fake_enable(mode, strict=False, trigger=None):
                    sys.stderr.write("[protenix-opt] NOT ACTIVE: stub reason\\n")
                    raise protenix_opt.ActivationError("stub reason")
                protenix_opt.enable = fake_enable
                A.install({"PROTENIX_OPT": "exact"})
                import runner.batch_inference
                print("STOCK RAN")
            """))
        env = dict(os.environ, STUBS=self.stubs, OPT=os.path.dirname(os.path.dirname(_autoload.__file__)))   # opt/ (the package's parent): `python -I` ignores PYTHONPATH, the script puts the tree's package on sys.path itself
        env.pop("PROTENIX_OPT", None)
        r = subprocess.run([sys.executable, "-I", script], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-500:])
        self.assertNotIn("STOCK RAN", r.stdout)
        self.assertEqual(r.stderr.count("[protenix-opt] NOT ACTIVE"), 1, r.stderr[-500:])
        self.assertNotIn("Traceback", r.stderr)

    def test_install_semantics(self):
        self.assertIsNone(_autoload.install({}))
        self.assertIsNone(_autoload.install({"PROTENIX_OPT": "off"}))
        self.assertIsNone(_autoload.install({"PROTENIX_OPT": " OFF "}))
        f = _autoload.install({"PROTENIX_OPT": " Exact "})                     # the core's finder, the selection stripped and case-folded
        try:
            self.assertEqual((type(f).__module__, f.mode, f.armed, f.spec.package, f.spec.triggers), ("opt_core.autoload", "exact", True, "protenix_opt", _autoload.TRIGGERS))
            self.assertIs(_autoload.install({"PROTENIX_OPT": "exact"}), f, "idempotent")
            self.assertIsNone(f.find_spec("os"), "non-trigger names are ignored")
            self.assertIsNone(f.find_spec("protenix.model.modules.pairformer"))
            self.assertTrue(f.armed)
            self.assertTrue(_autoload.disarm()); self.assertFalse(f.armed); self.assertNotIn(f, sys.meta_path)
            self.assertFalse(_autoload.disarm())
        finally:
            _autoload.disarm()

    def test_find_spec_probe_does_not_disarm(self):
        """An importlib.util.find_spec probe of a trigger name returns a spec the import machinery discards: the finder stays armed and the
        real import that follows still fires (a probe must never let stock run unhooked under PROTENIX_OPT)."""
        out = self.run_chain("import importlib.util; importlib.util.find_spec('runner'); importlib.util.find_spec('protenix.model'); import runner.batch_inference")
        self.assertIsNone(out["err"])
        self.assertEqual([e["trigger"] for e in out["log"]], ["runner"], "fires exactly once, on the import after the probes")
        self.assertTrue(out["log"][0]["ok"]); self.assertFalse(out["in_meta_path"])

    def test_unknown_mode_exits_3(self):
        """A selection under PROTENIX_OPT that names no mode: the NOT ACTIVE line and exit 3 — stock never runs under the variable."""
        r = subprocess.run([sys.executable, "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import protenix_opt._autoload as A; "
                            "A.install({'PROTENIX_OPT': 'turbo'}); print('STOCK RAN')", stack.opt_home()], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-300:]); self.assertNotIn("STOCK RAN", r.stdout)
        self.assertIn("NOT ACTIVE: unknown PROTENIX_OPT='turbo'", r.stderr); self.assertNotIn("Traceback", r.stderr)

    def test_undeclared_name_exits_3(self):
        """A mistyped name under the kit prefix (PROTENIX_OPT_MODE=exact, PROTENIX_OPT unset): refused by name with exit 3 — never ignored or
        stripped silently; the declared names pass. The declared tuple is locked to the names the package reads."""
        from protenix_opt import modes
        self.assertEqual(set(_autoload.DECLARED), {stack.ENV_MODE, stack.ENV_FORCE, stack.ENV_HOME, tp.ENV_NGPU, _autoload.ENV_TP_ROUTE, _autoload.ENV_CACHE_DIR, *modes._OWN})
        self.assertEqual(_autoload.undeclared({"PROTENIX_OPT": "exact", _autoload.ENV_CACHE_DIR: "/jitcache/key"}), [])   # the cache root configs/h100.env exports is a declared name
        self.assertEqual(_autoload.undeclared({"PROTENIX_OPT": "exact", "PROTENIX_OPT_FORCE": "1", "PROTENIX_OPT_HOME": "/x", "PATH": "/bin"}), [])
        self.assertEqual(_autoload.undeclared({"PROTENIX_V2_BIG_COND_CHUNK": "0", "PROTENIX_V2_BIG_ALLOW_PARTIAL": "1"}), ["PROTENIX_V2_BIG_ALLOW_PARTIAL", "PROTENIX_V2_BIG_COND_CHUNK"])   # the memory line has no switches
        self.assertEqual(_autoload.undeclared({"PROTENIX_OPT_MODE": "exact", "PROTENIX_OPTIMIZE": "1"}), ["PROTENIX_OPTIMIZE", "PROTENIX_OPT_MODE"])
        pth = os.path.join(stack.opt_home(), "protenix_opt_autoload.pth")
        line = next(l.strip() for l in open(pth, encoding="utf-8") if l.startswith("import "))
        env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT")}; env.update(PROTENIX_OPT_MODE="exact", PYTHONPATH=stack.opt_home())
        r = subprocess.run([sys.executable, "-S", "-c", line + "; print('STOCK RAN')"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-300:]); self.assertNotIn("STOCK RAN", r.stdout)
        self.assertIn("NOT ACTIVE: undeclared PROTENIX_OPT_MODE", r.stderr); self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run([sys.executable, "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from protenix_opt import cli; sys.exit(cli.main(['check']))", stack.opt_home()],
                           env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr[-300:]); self.assertIn("NOT ACTIVE: undeclared PROTENIX_OPT_MODE", r.stderr)

    def _startup_env(self):
        """An environment whose interpreter start processes the kit's .pth: the installed one (site-packages, after `pip install -e opt` — the
        box condition), else a copy in a user site (PYTHONUSERBASE) where the interpreter enables the user site; else None (skip by name)."""
        env0 = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT")}
        env0.update(PYTHONPATH=stack.opt_home(), PYTHONDONTWRITEBYTECODE="1")
        probe = "import site, os, json; print(json.dumps({'sp': site.getsitepackages(), 'user': site.ENABLE_USER_SITE, 'usp': site.getusersitepackages()}))"
        info = json.loads(subprocess.run([sys.executable, "-c", probe], env=env0, capture_output=True, text=True).stdout)
        if any(os.path.exists(os.path.join(d, "protenix_opt_autoload.pth")) for d in info["sp"]):
            return env0, None
        if not info["user"]:
            return None, None
        ub = tempfile.mkdtemp(prefix="ptx_userbase_")
        env0["PYTHONUSERBASE"] = ub
        usp = subprocess.run([sys.executable, "-c", "import site; print(site.getusersitepackages())"], env=env0, capture_output=True, text=True).stdout.strip()
        os.makedirs(usp, exist_ok=True)
        shutil.copyfile(os.path.join(stack.opt_home(), "protenix_opt_autoload.pth"), os.path.join(usp, "protenix_opt_autoload.pth"))
        return env0, ub


    def test_pth_line_under_site_addpackage_exits_3_without_a_traceback(self):
        """The .pth line exec'ed the way `site` does it (site.addpackage) — always runnable: the refusal ends the process by os._exit(3) with
        the one line out and no SystemExit traceback; the CLI route keeps SystemExit(3)."""
        code = "import site, sys; site.addpackage(sys.argv[1], 'protenix_opt_autoload.pth', None); print('STOCK RAN')"
        env0 = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT")}
        env0.update(PYTHONPATH=stack.opt_home(), PYTHONDONTWRITEBYTECODE="1")
        for extra, needle in (({"PROTENIX_OPT": "bogus"}, "NOT ACTIVE: unknown PROTENIX_OPT='bogus'"), ({"PROTENIX_OPT_MODE": "exact"}, "NOT ACTIVE: undeclared PROTENIX_OPT_MODE")):
            r = subprocess.run([sys.executable, "-S", "-c", code, stack.opt_home()], env=dict(env0, **extra), capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, (extra, r.stderr[-400:])); self.assertNotIn("STOCK RAN", r.stdout)
            lines = [l for l in r.stderr.splitlines() if l.strip()]
            self.assertEqual(len(lines), 1, r.stderr[-400:]); self.assertIn(needle, lines[0]); self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run([sys.executable, "-S", "-c", code, stack.opt_home()], env=dict(env0, PROTENIX_OPT="off"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "STOCK RAN"), r.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
