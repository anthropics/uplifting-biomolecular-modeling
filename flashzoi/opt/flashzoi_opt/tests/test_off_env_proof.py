"""The stock route: the tree's standalone stock caller (flashzoi/opt/flashzoi_opt/stock_pred.py — nothing of this package imported by it) and its
environment proof in fresh interpreters: PASS in a clean environment; FAIL by name with FLASHZOI_OPT set (the autoload finder), with a
kit switch set, with a kit root on sys.path; the CLI's stock environment strips the package's and the kit's variables and keeps the
data paths; the stock command line; `pred --mode off` forwards the script's exit code. The script is located through the package's
tree lookup (stack.tree_home()/opt/flashzoi_opt/stock_pred.py); the tests copy the tree's script into a temporary tree."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from flashzoi_opt import cli, stack
from flashzoi_opt.tests import _stubs

STOCK_SCRIPT = os.path.join(stack.opt_home(), "flashzoi_opt", "stock_pred.py")          # flashzoi/opt/flashzoi_opt/stock_pred.py in the tree
SCRIPT_REL = os.path.join("opt", "flashzoi_opt", "stock_pred.py")
PROBE = "import json, sys, importlib.util; spec = importlib.util.spec_from_file_location('stock_pred_script', %r); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(json.dumps(m.env_proof(det=%s)))"


def _pth_installed() -> bool:
    """True when this interpreter has the package installed (its generated guard .pth in site-packages) — the state in which FLASHZOI_OPT
    arms the autoload finder at interpreter start; a tree imported through PYTHONPATH alone has no .pth."""
    import glob, sysconfig
    return bool(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "flashzoi_opt_autoload.pth")))


def require_script():
    if not os.path.isfile(STOCK_SCRIPT):
        raise unittest.SkipTest(f"the tree's stock caller is not beside the package: {STOCK_SCRIPT}")
    return STOCK_SCRIPT


def _proof(env: dict, det: bool = False, extra_path=None) -> dict:
    """The proof in a fresh interpreter whose environment is the CLEAN one under test plus `env`: the caller's own PYTHONPATH is not part
    of it (a test runner started under a kit PYTHONPATH would otherwise hand the child a kit root on sys.path,
    which the proof rightly refuses: test_kit_root_on_path_fails is that case, given explicitly through `extra_path`)."""
    e = {k: v for k, v in os.environ.items() if k not in stack.PACKAGE_ENV and k not in stack.KIT_ENV_SWITCHES and k != "PYTHONPATH"}
    e.pop(cli.DET_ENV, None)
    e.update(env)
    if extra_path:
        e["PYTHONPATH"] = extra_path
    r = subprocess.run([sys.executable, "-s", "-c", PROBE % (require_script(), det)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


class TestOffEnvProof(unittest.TestCase):
    def test_clean_passes(self):
        p = _proof({})
        self.assertTrue(p["pass"], p["reasons"])
        self.assertEqual(p["finder_installed"], []); self.assertEqual(p["kit_modules_loaded"], [])
        self.assertEqual(p["kit_roots_on_sys_path"], []); self.assertEqual(p["env_hits"], [])
        self.assertIn("borzoi-pytorch", p["versions"]); self.assertIn("torch", p["versions"])
        self.assertTrue(set(p["flashzoi_opt_modules_loaded"]) <= set(p["pth_shim_modules"]))      # the installed package's .pth shim at most, never the core

    def test_script_imports_nothing_of_the_package(self):
        """Every import statement of the script (module level or inside a function) names stdlib / numpy / torch / borzoi_pytorch / huggingface_hub only."""
        import ast
        tree = ast.parse(open(require_script(), encoding="utf-8").read())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertFalse(roots & {"flashzoi_opt", "engines", "compare"}, roots)
        self.assertTrue(roots <= {"argparse", "hashlib", "importlib", "json", "os", "sys", "time", "numpy", "torch", "borzoi_pytorch", "huggingface_hub"}, roots)

    def test_mode_env_fails_by_name(self):
        p = _proof({"FLASHZOI_OPT": "exact"})
        self.assertFalse(p["pass"]); self.assertIn("FLASHZOI_OPT", p["env_hits"])
        self.assertTrue(any("forbidden environment variables set" in r and "FLASHZOI_OPT" in r for r in p["reasons"]), p["reasons"])
        if _pth_installed():   # the autoload finder is the INSTALLED package's (its site .pth reads FLASHZOI_OPT at interpreter start); a bare tree on PYTHONPATH has no .pth, so no finder to name
            self.assertTrue(any("finder on sys.meta_path" in r for r in p["reasons"]), p["reasons"])
        else:
            self.assertEqual(p["finder_installed"], [], p)

    def test_det_env_without_det_fails(self):
        p = _proof({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        self.assertFalse(p["pass"]); self.assertTrue(any("without --det" in r for r in p["reasons"]))
        self.assertTrue(_proof({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, det=True)["pass"])

    def test_kit_root_on_path_fails(self):
        kit = _stubs.require_kit()
        p = _proof({}, extra_path=kit)
        self.assertFalse(p["pass"]); self.assertIn(kit, p["kit_roots_on_sys_path"])

    def test_stock_env_strips_package_and_kit_variables(self):
        saved = dict(os.environ)
        try:
            os.environ.update({"FLASHZOI_OPT": "exact", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HOME": "/weights", "TRITON_CACHE_DIR": "/jc"})
            e = cli.stock_env(det=False)
            for k in ("FLASHZOI_OPT", "CUBLAS_WORKSPACE_CONFIG"):
                self.assertNotIn(k, e)
            self.assertEqual(e["HF_HOME"], "/weights"); self.assertEqual(e["TRITON_CACHE_DIR"], "/jc")
            self.assertEqual(cli.stock_env(det=True)["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        finally:
            os.environ.clear(); os.environ.update(saved)

    def _tree_with_script(self):
        t = _stubs.Tree().enter()
        os.makedirs(os.path.join(t.root, "opt", "flashzoi_opt")); shutil.copy(require_script(), os.path.join(t.root, SCRIPT_REL))
        return t

    def test_stock_command(self):
        t = self._tree_with_script()
        try:
            a = type("A", (), {"input": "items", "out": "outdir", "det": True, "device": "cuda", "items": "a,b", "tracks": "0-9,89"})()
            cmd = cli.stock_command(a)
            self.assertEqual(cmd[:3], [sys.executable, "-s", os.path.join(t.root, SCRIPT_REL)])
            self.assertIn("--det", cmd); self.assertEqual(cmd[cmd.index("--items") + 1], "a,b"); self.assertEqual(cmd[cmd.index("--tracks") + 1], "0-9,89")
            a.tracks = "all"; self.assertNotIn("--tracks", cli.stock_command(a))                 # the default is the script's own default: not passed
        finally:
            t.exit()

    def test_missing_script_is_a_usage_error(self):
        t = _stubs.Tree().enter()
        try:
            with self.assertRaises(cli.CliError):
                cli.stock_script()
        finally:
            t.exit()

    def test_refused_proof_exits_3_through_the_cli(self):
        """`pred --mode off` runs the script in the clean environment; a forbidden variable the CLI cannot strip (here: injected after the strip) makes
        the script refuse (exit 3, its REFUSED line, the proof written) before importing torch; the CLI forwards the exit code."""
        out = tempfile.mkdtemp(); inp = tempfile.mkdtemp()
        t = self._tree_with_script()
        try:
            e = cli.stock_env(det=False); e.pop("PYTHONPATH", None); e["FLASHZOI_OPT"] = "exact"   # the runner's own PYTHONPATH is not the subject (a kit root on it is test_kit_root_on_path_fails' case); the injected switch is
            a = type("A", (), {"input": inp, "out": out, "det": False, "device": "cpu", "items": None})()
            r = subprocess.run(cli.stock_command(a), env=e, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr)
            self.assertIn("[flashzoi-stock] REFUSED: forbidden environment variables set: ['FLASHZOI_OPT']", r.stderr)
            self.assertIn("[flashzoi-stock] stock environment proof FAIL", r.stderr)
            self.assertFalse(json.load(open(os.path.join(out, "opt_manifest.json")))["stock_env_proof"]["pass"])
            # the CLI's clean environment: the proof passes, then no items -> the script's usage exit (2), forwarded
            os.environ["FLASHZOI_OPT"] = "off"; ppath = os.environ.pop("PYTHONPATH", None)   # the runner's PYTHONPATH is not the CLI's subject here (a kit root on it is the script's own, separately tested refusal)
            try:
                rc = cli.main(["pred", "--mode", "off", "--input", inp, "--out", out])
            finally:
                os.environ.pop("FLASHZOI_OPT", None)
                if ppath is not None: os.environ["PYTHONPATH"] = ppath
            self.assertEqual(rc, 2)
            self.assertTrue(json.load(open(os.path.join(out, "opt_manifest.json")))["stock_env_proof"]["pass"])
        finally:
            t.exit()


if __name__ == "__main__":
    unittest.main()
