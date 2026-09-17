import io
import os
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

from enformer_opt import _runtime as R

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeKit(types.SimpleNamespace):
    """Stands in for engines.enformer.kits.v0_2: KIT_PINS, BINARIES, check_files, attach."""

    def __init__(self, class_pins=None, missing=None):
        super().__init__()
        self.KIT_PINS = {"version": "v0.2", "sm": (9, 0), "ptx": (9, 0), "levers": ("poscache", "fused", "graph", "xattn"),
                         "fused_so": "enformer_fastkit_fused.so", "xattn_so": "enformer_xattn_ext.so"}
        self.BINARIES = ("fused_so", "xattn_so")
        self._class_pins, self._missing = class_pins, missing

    def check_files(self, so_paths):
        if self._missing:
            raise FileNotFoundError(f"kit v0.2: {self._missing} missing")
        out = {"class_pins": self._class_pins}
        for k, p in so_paths.items():
            out[f"{k}_path"] = (self._class_pins or {}).get("dir", "") + "/" + os.path.basename(p) if self._class_pins else p
            out[f"{k}_sha256"] = "ab" * 32
        return out


def device(sm=(9, 0), name="NVIDIA H100 80GB HBM3", torch="2.13.0+cu130"):
    return {"name": name, "sm": sm, "memory_mib": 81559, "torch": torch, "cuda": "13.0", "cudnn": 92000}


def resolve_with(dev, kit=None, upstream="0.8.12", dry_run=True):
    kit = kit or FakeKit()
    with mock.patch.object(R, "_device", lambda: dev), mock.patch.object(R, "_upstream_version", lambda: upstream), mock.patch.object(R, "_kit", lambda: kit):
        return R.resolve(dry_run=dry_run)


class Resolution(unittest.TestCase):
    def test_h100_engages_whole_lever_set(self):
        rep = resolve_with(device())
        self.assertTrue(rep["active"]); self.assertEqual(rep["build"], "sm_90"); self.assertEqual(rep["levers"], ["poscache", "fused", "graph", "xattn"]); self.assertEqual(rep["notes"], [])
        self.assertEqual(R.line_of(rep), "[enformer-opt] DRY mode=exact gpu=NVIDIA H100 80GB HBM3(sm90) kit=v0.2 levers=poscache,fused,graph,xattn build=sm_90")
        rep["dry_run"] = False
        self.assertEqual(R.line_of(rep), "[enformer-opt] ACTIVE mode=exact gpu=NVIDIA H100 80GB HBM3(sm90) kit=v0.2 levers=poscache,fused,graph,xattn build=sm_90 applied=deferred")

    def test_a100_takes_the_class_build(self):
        cp = {"slug": "a100_80gb", "sm": (8, 0), "dir": "/t/classes/a100_80gb"}
        rep = resolve_with(device((8, 0), "NVIDIA A100-SXM4-40GB"), FakeKit(class_pins=cp))
        self.assertTrue(rep["active"]); self.assertEqual(rep["build"], "class:a100_80gb"); self.assertEqual(rep["notes"], [])
        self.assertTrue(rep["binaries"]["fused_so"]["path"].startswith("/t/classes/a100_80gb/"))

    def test_uncertified_devices_are_served_and_named(self):
        cp = {"slug": "a100_80gb", "sm": (8, 0), "dir": "/t/c", "serves": "sm_89 served by the sm_80 class build (same-major binary compatibility)"}
        rep = resolve_with(device((8, 9), "NVIDIA L40S"), FakeKit(class_pins=cp))
        self.assertTrue(rep["active"]); self.assertEqual(rep["build"], "class:a100_80gb")
        self.assertRegex(R.line_of(rep), r" notes=uncertified architecture — sm_89 served by the sm_80 class build")
        rep = resolve_with(device((10, 0), "NVIDIA B200"))
        self.assertTrue(rep["active"]); self.assertEqual(rep["build"], "sm_90-ptx")
        self.assertRegex(R.line_of(rep), r" notes=uncertified architecture — sm_100 runs the kit's sm_90 objects through their compute_90 PTX")

    def test_torch_drift_is_a_note_not_a_refusal(self):
        rep = resolve_with(device(torch="2.14.0+cu130"))
        self.assertTrue(rep["active"]); self.assertRegex(R.line_of(rep), r"notes=torch 2\.14\.0\+cu130 differs from the tested 2\.13\.0\+cu130")

    def test_refusals_by_name(self):
        for dev, kit, up, rx in (
                ({"name": None, "reason": "no CUDA device visible (torch.cuda.is_available() is False)"}, None, "0.8.12", r"reason=no CUDA device visible"),
                (device(), None, "0.8.11", r"reason=enformer-pytorch 0\.8\.11 is installed, the kit reproduces 0\.8\.12"),
                (device(), None, None, r"reason=enformer-pytorch is not installed"),
                (device(), FakeKit(missing="/x/enformer_xattn_ext.so"), "0.8.12", r"reason=FileNotFoundError: kit v0\.2: /x/enformer_xattn_ext\.so missing"),
        ):
            rep = resolve_with(dev, kit, up)
            self.assertFalse(rep["active"]); self.assertRegex(R.line_of(rep), r"^\[enformer-opt\] NOT ACTIVE mode=exact " + rx)

    def test_v100_refused_by_the_kit_reader(self):
        class K(FakeKit):
            def check_files(self, so_paths):
                raise RuntimeError("kit v0_2: no build in this tree runs on a device of capability sm_70 — cannot run here")
        rep = resolve_with(device((7, 0), "Tesla V100"), K())
        self.assertFalse(rep["active"]); self.assertRegex(R.line_of(rep), r"NOT ACTIVE mode=exact reason=RuntimeError: kit v0_2: no build in this tree runs on a device of capability sm_70")

    def test_check_prints_one_line_and_applies_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err), mock.patch.object(R, "_device", lambda: device()), mock.patch.object(R, "_upstream_version", lambda: "0.8.12"), mock.patch.object(R, "_kit", lambda: FakeKit()):
            rep = R.check()
        self.assertTrue(rep["active"]); self.assertEqual(err.getvalue().count("\n"), 1); self.assertIn("] DRY mode=exact", err.getvalue()); self.assertFalse(R._HOOK["installed"])


class Enable(unittest.TestCase):
    def setUp(self):
        R.disable(); R._REPORT = None

    def tearDown(self):
        R.disable(); R._REPORT = None
        sys.modules.pop("fake_upstream_mod", None)

    def _upstream(self):
        mod = types.ModuleType("fake_upstream_mod")

        class Enformer:
            def forward(self, x, *a, **kw):
                return ("stock", x, a, kw)

            def __call__(self, x, *a, **kw):
                return self.forward(x, *a, **kw)
        mod.Enformer = Enformer
        sys.modules["fake_upstream_mod"] = mod
        return mod

    def test_not_active_strict_raises_after_the_line(self):
        err = io.StringIO()
        with redirect_stderr(err), mock.patch.object(R, "_device", lambda: {"name": None, "reason": "no CUDA device visible"}):
            with self.assertRaises(R.ActivationError):
                R.enable(strict=True)
            rep = R.enable()
        self.assertFalse(rep["active"]); self.assertEqual(err.getvalue().count("NOT ACTIVE"), 2); self.assertFalse(R._HOOK["installed"])

    def test_enable_hooks_the_class_and_disable_restores_it(self):
        mod = self._upstream(); orig = mod.Enformer.forward
        err = io.StringIO()
        with redirect_stderr(err), mock.patch.object(R, "MODEL_MODULE", "fake_upstream_mod"), mock.patch.object(R, "_device", lambda: device()), \
                mock.patch.object(R, "_upstream_version", lambda: "0.8.12"), mock.patch.object(R, "_kit", lambda: FakeKit()):
            rep = R.enable(); rep2 = R.enable()
        self.assertTrue(rep["active"]); self.assertIs(rep, rep2); self.assertEqual(err.getvalue().count("ACTIVE"), 1)      # idempotent, one line
        self.assertTrue(getattr(mod.Enformer.forward, "__enformer_opt__", False)); self.assertIs(mod.Enformer.forward.__wrapped__, orig)
        m = mod.Enformer(); m.training = True                                     # a training-mode instance: stock, said once
        m.parameters = lambda: iter(())
        with redirect_stderr(err):
            out1 = m("x"); out2 = m("y")
        self.assertEqual(out1[0], "stock"); self.assertEqual(err.getvalue().count("STOCK model#1 is in training mode"), 1)
        with redirect_stderr(err):
            R.disable()
        self.assertIs(mod.Enformer.forward, orig); self.assertIsNone(R._REPORT)

    def test_apply_requires_enable(self):
        with self.assertRaises(R.ActivationError):
            R.apply(object())


class Autoload(unittest.TestCase):
    """ENFORMER_OPT semantics at interpreter start, in a child python with only the package on the path."""

    def run_child(self, value, code="import sys; import enformer_opt._autoload as A; print('FINDER', A.FINDER is not None, any(type(f).__name__ == 'Finder' for f in sys.meta_path))"):
        env = {k: v for k, v in os.environ.items() if k != "ENFORMER_OPT"}
        env["PYTHONPATH"] = OPT
        if value is not None:
            env["ENFORMER_OPT"] = value
        return subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True)

    def test_unset_and_off_install_nothing(self):
        for v in (None, "", "off", "OFF"):
            r = self.run_child(v)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "FINDER False False"); self.assertEqual(r.stderr, "")

    def test_exact_arms_the_finder(self):
        r = self.run_child("exact")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "FINDER True True"); self.assertEqual(r.stderr, "")

    def test_unknown_value_exits_3_by_name(self):
        for v in ("fast", "1", "exact-b8"):
            r = self.run_child(v)
            self.assertEqual(r.returncode, 3, (v, r.stderr)); self.assertRegex(r.stderr, r"^\[enformer-opt\] NOT ACTIVE mode=\S+ reason=unknown ENFORMER_OPT value"); self.assertEqual(r.stdout, "")

    def test_exact_without_a_device_exits_3_after_the_line_when_upstream_is_imported(self):
        # a stub `enformer_pytorch` package on the path: importing it fires the finder -> enable(strict) -> NOT ACTIVE (no torch / no device here) -> exit 3
        import tempfile
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "enformer_pytorch"))
        open(os.path.join(d, "enformer_pytorch", "__init__.py"), "w").write("LOADED = True\n")
        code = "import enformer_opt._autoload, enformer_pytorch; print('RAN PAST THE IMPORT')"
        env = {k: v for k, v in os.environ.items() if k != "ENFORMER_OPT"}; env.update(PYTHONPATH=OPT + os.pathsep + d, ENFORMER_OPT="exact")
        r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertRegex(r.stderr, r"\[enformer-opt\] NOT ACTIVE mode=exact reason="); self.assertNotIn("RAN PAST", r.stdout)


class Tree(unittest.TestCase):
    def test_the_vendored_closure_is_where_the_package_expects_it(self):
        self.assertTrue(os.path.isdir(os.path.join(R.KIT_HOME, "engines", "enformer", "kits", "v0_2")), R.KIT_HOME)
        for so in ("enformer_fastkit_fused.so", "enformer_xattn_ext.so"):
            self.assertTrue(os.path.isfile(os.path.join(R.BIN_DIR, so)), so)
            self.assertTrue(os.path.isfile(os.path.join(R.OPT_HOME, "forward", "classes", "a100_80gb", so)), so)

    def test_no_environment_switch_but_the_one(self):
        import glob, re
        names = set()
        for p in glob.glob(os.path.join(R.OPT_HOME, "**", "*.py"), recursive=True):
            if os.sep + "tests" + os.sep in p:
                continue
            names |= set(re.findall(r"""environ(?:\.get|\.pop|\.setdefault)?[\(\[]\s*['"]([A-Z][A-Z0-9_]+)['"]""", open(p, errors="replace").read()))
            names |= set(re.findall(r"""^ENV = ['"]([A-Z][A-Z0-9_]+)['"]""", open(p, errors="replace").read(), re.M))
        self.assertEqual(names - {"TORCH_EXTENSIONS_DIR"}, {"ENFORMER_OPT"}, names)     # TORCH_EXTENSIONS_DIR: torch's own build-dir variable, read by the nvcc fallback

    def test_main_check_usage(self):
        r = subprocess.run([sys.executable, "-m", "enformer_opt", "bogus"], env=dict(os.environ, PYTHONPATH=OPT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 2); self.assertIn("usage: python -m enformer_opt check", r.stderr)


if __name__ == "__main__":
    unittest.main()
