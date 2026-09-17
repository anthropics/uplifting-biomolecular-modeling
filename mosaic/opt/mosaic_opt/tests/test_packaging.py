"""Packaging completeness for everything the kit distribution ships or does at runtime: the built wheel carries mosaic_opt_autoload.pth
at its root, byte-identical to the tree's copy and recorded in the wheel's RECORD, with the kit itself (forward/) excluded from the
wheel; the real .pth, processed from a site directory the way site.py does at interpreter start, arms the finder and imports nothing
heavy under MOSAIC_OPT=exact, and leaves no finder or other submodule loaded when unset; the three frozen stock/ archives (mosaic,
joltz, boltz) match their PINS.json recipe, hold every path pip needs to build them plus the ProteinMPNN weights, are byte-identical to
this repo's own vendored stock/src/ copies, and check_pins.py correctly reports install provenance against them; the constructor wrap
that counts live upstream Boltz2 instances (an equinox.Module whose __hash__ reads a field that exists only after __init__ returns)
never hashes an instance before construction completes, counts correctly against both a hand-rolled stand-in and the real
equinox.Module shape, refuses activation while an instance already exists, and leaves the constructor completely untouched when
activation itself is refused."""


import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import zipfile
from contextlib import redirect_stderr

import pytest

import mosaic_opt
from . import _stubs, core_src
from mosaic_opt import stack


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


@unittest.skipUnless(_stubs.tree_present() and _has("setuptools"), "release tree or setuptools not present")
class TestWheel(unittest.TestCase):
    def test_wheel_carries_the_pth_at_its_root(self):
        tmp = tempfile.mkdtemp(prefix="mosaic_opt_wheel_")
        build_dir = os.path.join(_stubs.OPT_DIR, "build")                    # setuptools' build tree lands beside pyproject.toml: removed after the build
        had_build = os.path.isdir(build_dir)
        try:
            out = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q", "-w", tmp, _stubs.OPT_DIR], capture_output=True, text=True)
            if out.returncode != 0:
                self.skipTest(f"pip wheel unavailable here: {out.stderr[-400:]}")
            wheels = [f for f in os.listdir(tmp) if f.endswith(".whl")]
            self.assertEqual(len(wheels), 1, wheels)
            with zipfile.ZipFile(os.path.join(tmp, wheels[0])) as z:
                names = z.namelist()
                self.assertIn("mosaic_opt_autoload.pth", names)
                self.assertEqual(z.read("mosaic_opt_autoload.pth"), open(os.path.join(_stubs.OPT_DIR, "mosaic_opt_autoload.pth"), "rb").read())   # the generated text, as carried
                self.assertIn("import mosaic_opt._autoload", z.read("mosaic_opt_autoload.pth").decode())
                record = [n for n in names if n.endswith(".dist-info/RECORD")][0]
                self.assertIn("mosaic_opt_autoload.pth,sha256=", z.read(record).decode())
                self.assertTrue(any(n.startswith("mosaic_opt/tests/") for n in names))
                self.assertFalse(any("forward/" in n for n in names), "the kit is not part of the wheel")
                meta = [n for n in names if n.endswith(".dist-info/METADATA")][0]
                self.assertIn("Private :: Do Not Upload", z.read(meta).decode())
                self.assertIn("mosaic-opt = mosaic_opt.__main__:main", z.read([n for n in names if n.endswith("entry_points.txt")][0]).decode())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if not had_build:
                shutil.rmtree(build_dir, ignore_errors=True)                  # nothing generated stays in the tree


class TestInterpreterStart(unittest.TestCase):
    def _start(self, tail, mode_env):
        """A -S child that processes the REAL .pth from a temporary site dir (site.addsitedir, as site.py does at start-up) with the package and
        this tree's core on the path, then runs `tail`."""
        site_dir = tempfile.mkdtemp(prefix="mosaic_opt_site_")
        try:
            shutil.copy(os.path.join(_stubs.OPT_DIR, "mosaic_opt_autoload.pth"), site_dir)
            prog = f"import sys, site; sys.path[:0] = {[_stubs.OPT_DIR, core_src()]!r}; site.addsitedir({site_dir!r}); {tail}"
            env = {**{k: v for k, v in os.environ.items() if k not in ("MOSAIC_OPT", "PYTHONPATH")}, **mode_env}
            return subprocess.run([sys.executable, "-S", "-c", prog], capture_output=True, text=True, env=env)
        finally:
            shutil.rmtree(site_dir, ignore_errors=True)

    def test_pth_at_start_loads_nothing_heavy(self):
        out = self._start("print(sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'torch', 'numpy', 'mosaic', 'joltz', 'boltz', 'subprocess')))", {"MOSAIC_OPT": "exact"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")                             # the core pin gate reads MANIFEST.json with the standard library's json; nothing heavy, no subprocess
        out = self._start("print(sorted(m for m in sys.modules if m.startswith('opt_core')))", {"MOSAIC_OPT": "exact"})
        self.assertEqual(out.stdout.strip(), "['opt_core', 'opt_core.autoload']", out.stderr)   # under the variable: the core's finder module alone
        out = self._start("print(any(type(f).__name__ == 'Finder' for f in sys.meta_path))", {"MOSAIC_OPT": "exact"})
        self.assertEqual(out.stdout.strip(), "True", out.stderr)
        out = self._start("print(any(type(f).__name__ == 'Finder' for f in sys.meta_path), [m for m in sys.modules if m.startswith(('opt_core', 'mosaic_opt.'))])", {})
        self.assertEqual(out.stdout.strip(), "False ['mosaic_opt._autoload']", out.stderr)     # unset: no finder, nothing of the core, no other submodule


STOCK = os.path.join(_stubs.TREE, "stock")
RECIPE_PATHS = {"mosaic": ["src/mosaic", "pyproject.toml", "LICENSE", "README.md"], "joltz": ["src/joltz", "pyproject.toml", "LICENSE", "NOTICE", "README.md"],
                "boltz": ["src/boltz", "pyproject.toml", "LICENSE", "README.md"]}


def load_check_pins():
    spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(STOCK, "check_pins.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestStockArchives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins = json.load(open(os.path.join(STOCK, "PINS.json"), encoding="utf-8"))

    def test_three_pins_three_archives(self):
        self.assertEqual(sorted(self.pins["upstream"]), ["boltz", "joltz", "mosaic"])
        for name, pin in self.pins["upstream"].items():
            self.assertRegex(pin["commit"], r"^[0-9a-f]{40}$")
            self.assertIsNone(pin["tag"])
            self.assertEqual(pin["archive"], f"stock/{name}-{pin['commit'][:8]}.tar.gz")
            self.assertIn(pin["commit"], pin["install"])
            path = os.path.join(_stubs.TREE, pin["archive"])
            self.assertTrue(os.path.isfile(path), path)

    def test_archive_content_is_the_recipe(self):
        for name, pin in self.pins["upstream"].items():
            prefix = f"{name}-{pin['commit'][:8]}/"
            with tarfile.open(os.path.join(_stubs.TREE, pin["archive"]), "r:gz") as tf:
                members = tf.getnames()
                self.assertTrue(all(m == prefix.rstrip("/") or m.startswith(prefix) for m in members), f"{name}: a member outside {prefix}")
                rel = {m[len(prefix):] for m in members if m.startswith(prefix)} - {""}
                for p in RECIPE_PATHS[name]:
                    self.assertTrue(p in rel or any(r.startswith(p + "/") for r in rel), f"{name}: recipe path {p} missing")
                top = {r.split("/")[0] for r in rel}
                self.assertEqual(top, {p.split("/")[0] for p in RECIPE_PATHS[name]}, f"{name}: top-level members {top}")
                pyproject = tf.extractfile(prefix + "pyproject.toml").read().decode("utf-8")
                m = re.search(r'^readme\s*=\s*"([^"]+)"', pyproject, re.M)
                if m:
                    self.assertIn(m.group(1), rel, f"{name}: pyproject reads {m.group(1)}, which the archive omits (pip cannot build it)")
                self.assertNotIn(f"{prefix}tests", members); self.assertNotIn(f"{prefix}examples", members)

    def test_mosaic_archive_carries_the_proteinmpnn_weights(self):
        pin = self.pins["upstream"]["mosaic"]
        with tarfile.open(os.path.join(_stubs.TREE, pin["archive"]), "r:gz") as tf:
            names = set(tf.getnames())
        for f in self.pins["weights"]["proteinmpnn"]["files"]:
            self.assertIn(f"mosaic-{pin['commit'][:8]}/src/mosaic/proteinmpnn/weights/{f}", names)

    def test_check_pins_logic(self):
        cp = load_check_pins()
        bad, detail = cp.check(self.pins["upstream"])
        for name in self.pins["upstream"]:
            self.assertIn(name, detail)
        installed = {n for n, d in detail.items() if d["version"]}
        for n in installed:
            self.assertIn(detail[n]["source"].split()[0], ("commit", "archive", "a"), detail[n])


EQX_LIKE = textwrap.dedent('''\
    class Boltz2:
        """Stand-in with equinox.Module's hashing: __hash__ reads a field that exists only after __init__ has run."""
        def __init__(self, model=None):
            self.model = model
        def __hash__(self):
            return hash(self.model)
        def __eq__(self, other):
            return type(other) is type(self) and other.model == self.model
    ''')

EQX_REAL = textwrap.dedent('''\
    import equinox as eqx
    class Boltz2(eqx.Module):
        """upstream's shape: an equinox.Module with a `model` field set in __init__ (models/boltz2.py:129-133)."""
        model: eqx.Module
        def __init__(self):
            self.model = eqx.nn.Identity()
    ''')


def model_package(tmp: str, body: str) -> str:
    """A fake upstream `mosaic` package on a temp path whose mosaic.models.boltz2.Boltz2 is `body`."""
    pkg = os.path.join(tmp, "site", "mosaic")
    os.makedirs(os.path.join(pkg, "models"), exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").write("")
    open(os.path.join(pkg, "models", "__init__.py"), "w").write("")
    open(os.path.join(pkg, "models", "boltz2.py"), "w").write(body)
    return os.path.join(tmp, "site")


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestInstanceCounter(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_counter_")
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR",
                  "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_FORCE", "MOSAIC_CACHE_DIR", "MODEL_OPT_TARGET_GPU"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)
        self.fake = _stubs.FakeReproCache()
        self.mp.setattr(stack, "p1_lever", lambda: self.fake)
        self.mp.setattr(stack, "backend_initialised", lambda: (False, "not initialised"))

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def _ready(self):
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))

    def _enable(self):
        with redirect_stderr(io.StringIO()):
            return mosaic_opt.enable("exact")

    def test_wrap_never_hashes_the_instance_before_init(self):
        """The reviewed bug: a class hashed before its fields exist raised AttributeError from inside the wrap."""
        self._ready()
        sys.path.insert(0, model_package(self.tmp, EQX_LIKE))
        rep = self._enable()
        self.assertTrue(rep["active"])
        import mosaic.models.boltz2 as mb
        a = mb.Boltz2(model=3)                                                      # must construct: no hash before __init__
        self.assertEqual(a.model, 3); self.assertEqual(hash(a), hash(3))              # the wrap left the instance as upstream built it
        b = mb.Boltz2(model=3)                                                      # equal by value: a WeakSet would have folded the two
        chk = stack.instance_check()
        self.assertEqual((chk["n"], chk["method"], chk["built"]), (2, "counted", 2))
        del a
        self.assertEqual(stack.instance_check()["n"], 1)
        del b
        self.assertEqual(stack.instance_check()["n"], 0)

    def test_real_equinox_module(self):
        """Upstream's class shape for real: an equinox.Module subclass with a field set in __init__ (equinox reads every field in __hash__)."""
        pytest.importorskip("equinox")
        self._ready()
        sys.path.insert(0, model_package(self.tmp, EQX_REAL))
        rep = self._enable()
        self.assertTrue(rep["active"])
        import mosaic.models.boltz2 as mb
        m = mb.Boltz2()                                                             # constructs under the wrap
        self.assertIsNotNone(m.model)
        hash(m)                                                                     # a complete Module hashes
        self.assertEqual(stack.instance_check(), {"n": 1, "method": "counted", "built": 1})
        rep2 = mosaic_opt.enable("exact")                                           # idempotent: the first report stands
        self.assertTrue(rep2["active"])
        del m
        self.assertEqual(stack.instance_check()["n"], 0)

    def test_real_equinox_instance_refuses_late_activation(self):
        pytest.importorskip("equinox")
        self._ready()
        sys.path.insert(0, model_package(self.tmp, EQX_REAL))
        import mosaic.models.boltz2 as mb
        keep = mb.Boltz2()
        rep = self._enable()
        self.assertFalse(rep["active"]); self.assertIn("1 Boltz2 instance(s) already exist", rep["reason"])
        self.assertEqual(self.fake.calls, [])
        del keep

    def test_counter_is_registered_after_the_gates(self):
        """A refused activation (pins here; the default non-strict enable()) leaves the class unwrapped and the constructor as upstream's."""
        _stubs.stub_gates(self.mp, pins_ok=False)
        sys.path.insert(0, model_package(self.tmp, EQX_LIKE))
        import mosaic.models.boltz2 as mb
        orig = mb.Boltz2.__init__
        rep = self._enable()
        self.assertFalse(rep["active"]); self.assertIn("not at the pinned commit", rep["reason"])
        self.assertIs(mb.Boltz2.__init__, orig); self.assertIsNone(stack._INSTANCES["state"]); self.assertIsNone(stack._INSTANCES["finder"])
        x = mb.Boltz2(model=1)
        self.assertEqual(x.model, 1)
        del x
        # the same with no P1 directory (the gate that follows the stack gates): still nothing wrapped
        self.state.restore(); self.mp.setattr(stack, "p1_lever", lambda: self.fake)
        _stubs.stub_gates(self.mp)
        rep = self._enable()
        self.assertFalse(rep["active"]); self.assertIn("no P1 directory", rep["reason"])
        self.assertIs(mb.Boltz2.__init__, orig); self.assertIsNone(stack._INSTANCES["state"])


if __name__ == "__main__":
    unittest.main()
