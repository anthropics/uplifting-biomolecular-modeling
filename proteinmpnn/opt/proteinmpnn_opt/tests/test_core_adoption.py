"""The core this kit stands on (opt_core, pinned in opt/pyproject.toml [tool.opt_core]): the core pin gate's facts hold the installed core (the pin is a
floor — installed >= pinned, the tree's core may run ahead of it — and the core's own live comparison agrees), the kit's own required files are present, the environment variable precedence for the tree's
home, the manifest carries the core's block and the recipe line, the activation report's core facts and the fallback pin reader agree with the gate,
and a wrong core (absent, older, unversioned) is the gate's one NOT ACTIVE line and SystemExit(3) in process — through `core_gate()` and through
`stack.activate`, whose core facts come from the same producer. Standard library + opt_core only; no torch, no GPU."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import opt_core
from opt_core import gates as core_gates, manifest as core_manifest

import proteinmpnn_opt
from proteinmpnn_opt import _core_gate, det, manifest, report, stack

OPT = os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__)))
CORE = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


class TestCoreAdoption(unittest.TestCase):
    def test_core_pin_gate_holds_the_installed_core(self):
        facts = proteinmpnn_opt.core_gate()                                       # the pin gate: returns its facts on a match, SystemExit(3) otherwise
        self.assertEqual(facts["tag"], proteinmpnn_opt.TAG)
        self.assertEqual(facts["pinned"]["pyproject"], os.path.join(OPT, "pyproject.toml"))
        self.assertLessEqual(_core_gate.version_tuple(facts["pinned"]["version"]), _core_gate.version_tuple(opt_core.__version__))   # the pin is a minimum version (_core_gate / opt_core.gates.core_pin_check): the installed core meets or exceeds it
        self.assertEqual(facts["installed"]["version"], opt_core.__version__)     # the gate's own located core is the one this interpreter imports
        self.assertEqual(facts["installed"]["package_dir"], os.path.dirname(os.path.abspath(opt_core.__file__)))   # the located core is the one this interpreter imports
        self.assertEqual(os.path.normpath(facts["installed"]["root"]), os.path.normpath(CORE))
        self.assertEqual(os.path.normpath(os.path.join(OPT, facts["pinned"]["path"])), os.path.normpath(CORE))   # the pin's path resolves to the installed core (editable, beside the kit)
        g = core_gates.core_pin_check(facts["pinned"]["pyproject"])               # the core's live-tree comparison (tests and tools) agrees with the gate's read
        self.assertTrue(g.ok, g.reason)
        self.assertEqual(g.details["imported"]["version"], facts["installed"]["version"])

    def test_carried_kit_files_are_present(self):
        """check_kits is a presence gate (KIT_REQUIRED_FILES), not a hash comparison through the core — the tree's git commit names the bytes."""
        n_ok, bad = stack.check_kits()
        self.assertEqual(bad, [])
        self.assertEqual(n_ok, sum(len(v) for v in stack.KIT_REQUIRED_FILES.values()))

    def test_home_honours_the_package_variable_then_the_trees(self):
        here = stack.tree_home()
        with mock.patch.dict(os.environ, {"MODEL_OPT": "/nonexistent/tree", stack.ENV_HOME: here}):
            self.assertEqual(stack.tree_home(), here)                            # the package's own variable first
        with mock.patch.dict(os.environ, {"MODEL_OPT": here}, clear=False):
            os.environ.pop(stack.ENV_HOME, None)
            self.assertEqual(stack.tree_home(), here)

    def test_manifest_carries_the_core_block_and_the_recipe_line(self):
        m = manifest.build({"mode": "exact", "variant": "soluble", "route": "worker", "active": True}, command="check")
        self.assertEqual(m["core"], core_manifest.core_block())
        self.assertEqual(m["core"]["version"], proteinmpnn_opt.core_gate()["installed"]["version"])
        self.assertEqual(m["det"], det.describe())

    def test_activation_report_carries_the_gates_core_facts(self):
        """`stack.activate` fills `opt_core` from the gate's facts (the one producer of 'pinned X, installed Y'): the stock route included."""
        facts = proteinmpnn_opt.core_gate()
        with mock.patch.object(stack, "gpu_info", return_value=None), mock.patch.object(stack, "check_pins", return_value={"pinned": True, "detail": {}, "findings": []}):
            for mode in ("exact", "off"):
                rep = stack.activate(mode, "soluble", dry_run=True)
                self.assertEqual(rep["opt_core"], dict(facts["installed"], ok=True, pinned=facts["pinned"]), mode)

    def test_pin_table_reads_through_the_gates_fallback_reader(self):
        """Whatever python the box runs: the gate's no-tomllib reader (a python < 3.11 stack) must read this kit's [tool.opt_core] table — the
        header line is exactly `[tool.opt_core]` (a trailing comment on it hides the whole table from that reader) — and agree with the tomllib
        read and with the core's own reader."""
        import builtins
        real_import = builtins.__import__
        pyproject = proteinmpnn_opt.core_gate()["pinned"]["pyproject"]

        def no_tomllib(name, *a, **k):
            if name == "tomllib":
                raise ModuleNotFoundError("No module named 'tomllib'")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules, {"tomllib": None}):
            with mock.patch.object(builtins, "__import__", no_tomllib):
                fallback = _core_gate.read_table(pyproject, "tool." + _core_gate.PIN_TABLE)
                facts = _core_gate.gate(pyproject, tag=proteinmpnn_opt.TAG)          # the whole gate passes on the fallback reader too
        self.assertEqual(set(fallback), set(_core_gate.PIN_KEYS), fallback)
        self.assertEqual(fallback, _core_gate.read_table(pyproject, "tool." + _core_gate.PIN_TABLE))   # == the tomllib read
        self.assertEqual(fallback, {k: v for k, v in core_gates.read_pin_table(pyproject).items() if k in fallback})   # == the core's reader
        self.assertEqual((fallback["path"], fallback["version"]), (facts["pinned"]["path"], facts["pinned"]["version"]))
        header = [l for l in open(pyproject, encoding="utf-8").read().split("\n") if l.lstrip().startswith("[tool.opt_core]")]
        self.assertEqual(header, ["[tool.opt_core]"])

    def _refused(self, fn):
        """fn() must be the gate's refusal: (SystemExit code, the one stderr line)."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            fn()
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        return cm.exception.code, lines[0]

    def test_a_wrong_core_is_the_gates_one_line_and_exit_3_in_process(self):
        """An older core, a core with no `__version__` literal at all (unversioned) and no core at all, as the gate locates them
        (`_core_gate.installed_core`): `core_gate()`, `enable()` and `stack.activate` (whose core facts come from the same producer) are
        SystemExit(3) with the one NOT ACTIVE line naming both sides — the exact format `_core_gate._refuse` writes."""
        real = _core_gate.installed_core()
        pin = proteinmpnn_opt.core_gate()["pinned"]
        want_v, pin_path = pin["version"], pin["path"]
        cases = {"stale": (lambda: dict(real, version="0.2.5"),
                            "reason=core_mismatch: opt_core pinned >= v%s at %s" % (want_v, pin_path),
                            "installed v0.2.5 at %s" % real["root"]),
                 "unversioned": (lambda: dict(real, version=None),
                            "reason=core_mismatch: opt_core pinned >= v%s at %s" % (want_v, pin_path),
                            "installed v? at %s" % real["root"]),
                 "absent": (lambda: None,
                            "reason=core_missing:opt_core (pinned >= v%s at %s;" % (want_v, pin_path),
                            "nothing importable as opt_core on sys.path")}
        for name, (located, reason, detail) in cases.items():
            with mock.patch.object(_core_gate, "installed_core", located), \
                 mock.patch.object(stack, "gpu_info", return_value=None), mock.patch.object(stack, "check_pins", return_value={"pinned": True, "detail": {}, "findings": []}):
                for fn in (proteinmpnn_opt.core_gate, lambda: proteinmpnn_opt.enable("exact", "soluble"), lambda: stack.activate("exact", "soluble", dry_run=True),
                           lambda: stack.activate("off", "soluble", dry_run=True)):
                    code, line = self._refused(fn)
                    self.assertEqual(code, report.EXIT_NOT_ACTIVE, name)
                    self.assertTrue(line.startswith("[proteinmpnn-opt] NOT ACTIVE: " + reason), (name, line))
                    self.assertIn(detail, line, name)
        self.assertEqual(proteinmpnn_opt.core_gate()["installed"], real)          # unpatched: the facts again, nothing raised


if __name__ == "__main__":
    unittest.main()
