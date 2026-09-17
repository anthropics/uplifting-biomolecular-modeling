"""The core pin (path + minimum version, read through both the gate's and the core's own TOML fallback readers, and through the
package's launcher scripts before any other probe) and the stock pins: check_pins.py refuses on this box (no upstream installed)
with one line per package, accepts fake installed-distribution metadata at the pinned commits and refuses one at another commit or
a non-git/non-archive source; environment/requirements.lock is the 201-line list of the pinned stack (the tested stack's `pip freeze --all`)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from .. import stack

TREE = stack.tree_home()



class TestCorePin(unittest.TestCase):
    """The shared core as this kit stands on it. ``opt/pyproject.toml`` ``[tool.opt_core]`` is readable with all three keys through BOTH
    readers of the gate (``_core_gate.read_table``) and of the core (``opt_core.gates.read_pin_table``) — ``tomllib`` and the line readers an
    interpreter without ``tomllib`` takes — and they agree; the header line is exactly ``[tool.opt_core]``. The pin is a floor the installed
    core meets or exceeds (the core pin gate's facts: version, located without importing), under the kit's one tag. ``opt/_build_backend.py`` and
    ``caliby_opt/_core_gate.py`` are the installed core's ``kit_template/`` files byte for byte, and ``opt/caliby_opt_autoload.pth`` is the
    backend's generated text for (caliby_opt, CALIBY_OPT, the tag). In ``run.sh`` and ``configs/h100.env`` the first two ``python`` lines are
    the importability probe and the gate, before any other probe or verb."""

    def test_pin_table_reads_through_the_fallback_readers_and_tomllib_alike(self):
        import builtins
        from unittest import mock
        from opt_core import gates
        from .. import _core_gate
        path = stack.pyproject_path()
        with open(path, encoding="utf-8") as fh:
            self.assertIn("[tool.opt_core]\n", fh.read())                                   # the header alone on its line
        real_import = builtins.__import__

        def no_tomllib(name, *a, **k):
            if name == "tomllib":
                raise ModuleNotFoundError("tomllib masked by the test")
            return real_import(name, *a, **k)
        with mock.patch.dict(sys.modules):
            sys.modules.pop("tomllib", None)
            with mock.patch.object(builtins, "__import__", no_tomllib):
                fallback = gates.read_pin_table(path)
                gate_fallback = _core_gate.read_table(path, "tool.opt_core")
        for key in gates.PIN_KEYS:
            self.assertTrue(fallback.get(key), (key, fallback))                              # path, version both come back
        self.assertEqual({k: gate_fallback[k] for k in _core_gate.PIN_KEYS}, {k: fallback[k] for k in gates.PIN_KEYS})
        try:
            import tomllib  # noqa: F401
        except ModuleNotFoundError:
            return                                                                            # an interpreter without tomllib: the fallback IS the reader
        self.assertEqual(gates.read_pin_table(path), fallback)
        self.assertEqual(_core_gate.read_table(path, "tool.opt_core"), gate_fallback)
        pin = gates.core_pin(path)                                                            # the composed pin (abs_path resolved) is the same table
        self.assertEqual({k: pin[k] for k in gates.PIN_KEYS}, {k: fallback[k] for k in gates.PIN_KEYS})

    def test_pin_is_the_installed_core(self):
        import opt_core                                                                       # the test may import it; the gate did not
        from .. import _core_gate, cli
        f = stack.core_gate()
        self.assertEqual((f["tag"], stack.TAG, cli.PROG), ("caliby-opt", "caliby-opt", "caliby-opt"))   # the gate's line, every kit line, the console script: one spelling
        self.assertGreaterEqual(_core_gate.version_tuple(f["installed"]["version"]), _core_gate.version_tuple(f["pinned"]["version"]))  # the floor: installed >= pinned
        self.assertTrue(os.path.samefile(f["pinned"]["pyproject"], stack.pyproject_path()))  # the pin the gate reads beside the package is the tree's opt/pyproject.toml
        self.assertTrue(os.path.samefile(f["installed"]["package_dir"], os.path.dirname(opt_core.__file__)))   # the core it located is the one an import binds
        self.assertEqual(opt_core.__version__, f["installed"]["version"])                     # the gate's located version is what an import binds

    def test_the_gate_is_the_first_probe_of_run_sh_and_the_config(self):
        gate_probe = 'python -c "from caliby_opt.stack import core_gate; core_gate()" >/dev/null || '
        for rel in ("run.sh", os.path.join("configs", "h100.env")):
            with open(os.path.join(stack.tree_home(), rel), encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if not ln.lstrip().startswith("#")]
            py = [ln.strip() for ln in lines if "python " in ln]                              # every line that runs python, in order
            if rel == "run.sh":                                                                # `run.sh install` is the one verb before every probe: it installs the package the probes import
                first = next(i for i, ln in enumerate(py) if ln.startswith('python -c "import caliby_opt"'))
                install, py = py[:first], py[first:]                                            # the install block's own lines: python on PATH, the installed-from-this-tree probe, pip, the pin check, the weights step
                self.assertTrue(any("python -m pip install -e" in ln for ln in install), (rel, install))
                self.assertTrue(any(ln.startswith('python -I "$HERE/stock/check_pins.py"') for ln in install), (rel, install))
                self.assertTrue(any("python -m caliby_opt.weights" in ln for ln in install), (rel, install))
            self.assertTrue(py[0].startswith('python -c "import caliby_opt" 2>/dev/null || {'), (rel, py[0]))   # 1: the package is importable (its own message, rc 3)
            self.assertTrue(py[1].startswith(gate_probe), (rel, py[1]))                       # 2: the core pin gate, its line never swallowed, rc 3
            self.assertIn("exit 3", py[0]); self.assertIn("exit 3", py[1])
            self.assertNotIn("2>", py[1].split("||")[0], rel)                                 # the gate's stderr reaches the caller
        with open(os.path.join(stack.tree_home(), "configs", "h100.env"), encoding="utf-8") as fh:
            self.assertIn(f'echo "[{stack.TAG}] NOT ACTIVE: caliby_opt is not importable on ', fh.read())   # the config's own refusal carries the kit's tag


class TestStockPins(unittest.TestCase):
    def test_pinned_freeze(self):
        lines = [ln for ln in open(os.path.join(TREE, "environment", "requirements.lock")).read().splitlines() if ln and not ln.startswith("#")]
        self.assertEqual(len(lines), 201)
        self.assertIn("torch==2.6.0", lines)
        self.assertIn("triton==3.2.0", lines)
        self.assertIn("biotite==1.6.0", lines)

    def test_check_pins_refuses_here(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertIn("check_pins: caliby: not installed", r.stderr)
        self.assertIn("check_pins: atomworks-caliby: not installed", r.stderr)
        self.assertIn("protpardelle: not installed (needed by the ensemble32 variant only)", r.stdout)

    def test_check_pins_reads_the_installed_commit(self):
        """A fake site-packages with the three distributions' metadata: at the pinned commits check_pins exits 0; one at another
        commit, or from a non-git source, is refused by name with the commit it found."""
        pins = stack.pins()["upstream"]

        def site(commits):
            root = tempfile.mkdtemp()
            for name, pin in pins.items():
                dist = os.path.join(root, f"{name.replace('-', '_')}-{pin['version']}.dist-info")
                os.makedirs(dist)
                open(os.path.join(dist, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {pin['version']}\n")
                c = commits.get(name, pin["commit"])
                url = {"url": pin["repo"] + ".git", "vcs_info": {"vcs": "git", "commit_id": c, "requested_revision": c}} if c else {"url": "https://pypi.org/x", "archive_info": {}}
                json.dump(url, open(os.path.join(dist, "direct_url.json"), "w"))
            return root

        def run(root):                                                 # -I ignores PYTHONPATH: the fake site goes on sys.path in-process
            code = f"import sys, runpy; sys.path.insert(0, {root!r}); sys.argv = ['check_pins.py']; runpy.run_path({os.path.join(TREE, 'stock', 'check_pins.py')!r}, run_name='__main__')"
            return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        r = run(site({}))
        self.assertEqual((r.returncode, r.stderr), (0, ""))
        r = run(site({"caliby": "deadbeef" * 5}))
        self.assertEqual(r.returncode, 3)
        self.assertIn("check_pins: caliby 0.1: installed from commit deadbeefdeadbeef", r.stderr)
        self.assertIn("want https://github.com/ProteinDesignLab/caliby @ 41d31560c3c73d7980d94f40f3c852b90bfab5c0", r.stderr)
        self.assertNotIn("atomworks-caliby", r.stderr)
        r = run(site({"atomworks-caliby": None}))
        self.assertEqual(r.returncode, 3)
        self.assertIn("atomworks-caliby 1.0.0: installed from a non-git, non-archive source", r.stderr)

    def test_check_pins_accepts_an_archive_install_by_filename(self):
        """A distribution's direct_url.json with non-empty archive_info and a URL whose filename equals the pin's archive
        basename is `pinned` (an archive install, not a git install) -- the branch check_pins.py takes for a user who ran
        ``pip install stock/<archive>`` instead of the git line."""
        pins = stack.pins()["upstream"]
        root = tempfile.mkdtemp()
        for name, pin in pins.items():
            dist = os.path.join(root, f"{name.replace('-', '_')}-{pin['version']}.dist-info")
            os.makedirs(dist)
            open(os.path.join(dist, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {pin['version']}\n")
            if name == "caliby":                                      # the one package installed from its pinned archive
                url = {"url": f"file:///tmp/{os.path.basename(pin['archive'])}", "archive_info": {"hash": "sha256=00"}}
            else:
                url = {"url": pin["repo"] + ".git", "vcs_info": {"vcs": "git", "commit_id": pin["commit"], "requested_revision": pin["commit"]}}
            json.dump(url, open(os.path.join(dist, "direct_url.json"), "w"))
        code = (f"import sys, runpy; sys.path.insert(0, {root!r}); sys.argv = ['check_pins.py']; "
                f"runpy.run_path({os.path.join(TREE, 'stock', 'check_pins.py')!r}, run_name='__main__')")
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stderr), (0, ""))
        self.assertIn(f"caliby 0.1: pinned (archive {os.path.basename(pins['caliby']['archive'])})", r.stdout)


if __name__ == "__main__":
    unittest.main()
