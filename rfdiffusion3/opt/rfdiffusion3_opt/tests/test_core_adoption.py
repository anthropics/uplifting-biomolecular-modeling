"""The core this kit stands on (opt_core, pinned in opt/pyproject.toml [tool.opt_core] as a path + minimum version): the core pin gate's
facts hold the installed core and fill the activation report, a core older than the pin is refused by name with exit 3, the pin table
reads the same with and without tomllib, the build backend and the gate are the installed core's templates byte for byte, the .pth is the
backend's generated text for this package / switch / tag, the autoload module's restated constants equal the core's, the exit table and
the line grammar are the core's, the manifest carries exactly the core block, the JIT cache key is the core's spelling."""
import glob
import importlib.util
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

import opt_core
from opt_core import gates as core_gates, jit_cache as core_jit_cache, manifest as core_manifest, report as core_report

from .. import _autoload, _core_gate, manifest as mf, modes, report, stack

OPT = os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__)))          # opt/: pyproject.toml, _build_backend.py, the .pth
PYPROJECT = os.path.join(OPT, "pyproject.toml")
PTH = os.path.join(OPT, "rfdiffusion3_opt_autoload.pth")


def _backend():
    """opt/_build_backend.py loaded as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location("_kit_build_backend", os.path.join(OPT, "_build_backend.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCoreAdoption(unittest.TestCase):
    def test_core_pin_gate_holds_the_installed_core_and_fills_the_report(self):
        facts = _autoload.core_gate()                                                # returns only on a match (else SystemExit 3)
        self.assertEqual(facts["tag"], _autoload.TAG)
        self.assertEqual(facts["pinned"]["pyproject"], PYPROJECT)
        self.assertEqual(facts["pinned"]["version"], opt_core.__version__)
        self.assertEqual(facts["pinned"]["version"], facts["installed"]["version"])
        self.assertEqual(os.path.realpath(facts["installed"]["package_dir"]), os.path.realpath(os.path.dirname(os.path.abspath(opt_core.__file__))))   # located without importing: the core this process imports
        self.assertEqual(facts["installed"]["version"], core_manifest.core_block()["version"])                            # the gate's own regex-read version agrees with the core's __version__ attribute
        rep = stack.activate("exact", dry_run=True)                                  # the gate's facts are the report's core block (one producer)
        self.assertEqual(rep["core"]["ok"], True)
        self.assertEqual(rep["core"]["pinned"], facts["pinned"])
        self.assertEqual({k: rep["core"][k] for k in facts["installed"]}, facts["installed"])

    def test_a_wrong_pin_is_refused_by_name_exit_3(self):
        """An absent core and a core older than the pin (or unversioned) against this kit's real pin: one NOT ACTIVE line, SystemExit 3
        (the gate anchors on the package's own pyproject.toml; the installed side is what `installed_core()` reports). Floor semantics:
        a newer core always passes, so a refused fixture must be strictly OLDER than the pin — never same-version-different-sha, which
        no longer means anything (the git commit identifies the core now, not a stored digest)."""
        real = _core_gate.installed_core()
        want = _core_gate.read_table(PYPROJECT, "tool.opt_core")
        cases = {
            "core_missing:opt_core (pinned >= v%s at %s;" % (want["version"], want["path"]): lambda: None,
            "core_mismatch: opt_core pinned >= v%s at %s, installed v0.2.5 at %s" % (want["version"], want["path"], real["root"]): lambda: dict(real, version="0.2.5"),
            "core_mismatch: opt_core pinned >= v%s at %s, installed v? at %s" % (want["version"], want["path"], real["root"]): lambda: dict(real, version=None),
        }
        for reason_prefix, fake in cases.items():
            err = io.StringIO()
            with mock.patch.object(_core_gate, "installed_core", fake), redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                _autoload.core_gate()
            self.assertEqual(cm.exception.code, 3)
            lines = err.getvalue().splitlines()
            self.assertEqual(len(lines), 1, lines)
            self.assertTrue(lines[0].startswith("[%s] NOT ACTIVE: reason=%s" % (_autoload.TAG, reason_prefix)), lines[0])

    def test_pin_table_reads_the_same_without_tomllib_and_its_header_is_bare(self):
        """opt/pyproject.toml [tool.opt_core] through the two readers that work without tomllib (the kit's gate, the core's gates.read_pin_table on a
        Python < 3.11 stack): path and version equal the tomllib read; the table's header line is exactly `[tool.opt_core]`."""
        import sys
        with_toml = _core_gate.read_table(PYPROJECT, "tool.opt_core")
        self.assertEqual({k: with_toml[k] for k in ("path", "version")}, {k: core_gates.read_pin_table(PYPROJECT)[k] for k in ("path", "version")})
        saved = sys.modules.get("tomllib", KeyError)
        sys.modules["tomllib"] = None                                                # `import tomllib` raises ModuleNotFoundError -> the fallback readers
        try:
            gate_fallback = _core_gate.read_table(PYPROJECT, "tool.opt_core")
            core_fallback = core_gates.read_pin_table(PYPROJECT)
        finally:
            if saved is KeyError:
                del sys.modules["tomllib"]
            else:
                sys.modules["tomllib"] = saved
        for fallback in (gate_fallback, core_fallback):
            self.assertEqual({k: fallback[k] for k in ("path", "version")}, {k: with_toml[k] for k in ("path", "version")}, fallback)
        self.assertEqual(gate_fallback["version"], opt_core.__version__)
        self.assertEqual([ln.rstrip("\n") for ln in open(PYPROJECT, encoding="utf-8") if ln.lstrip().startswith("[tool.opt_core]")], ["[tool.opt_core]"])

    def test_test_extra_carries_the_build_backends_setuptools_floor(self):
        """`pip install -e opt[test]` (tests/README) brings what the suite imports: pytest, and setuptools at the [build-system] floor — the
        adoption test above loads opt/_build_backend.py (setuptools.build_meta), and a Python >= 3.12 venv carries no setuptools of its own."""
        import ast
        import re
        text = open(PYPROJECT, encoding="utf-8").read()

        def array(table, key):                                                   # `key = [ ... ]` inside `[table]` (a TOML array of strings reads as a Python list literal)
            body = re.search(r"^\[%s\]\s*\n(.*?)(?=^\[|\Z)" % re.escape(table), text, re.M | re.S).group(1)
            return ast.literal_eval(re.search(r"^%s\s*=\s*(\[.*?\])" % re.escape(key), body, re.M | re.S).group(1))
        build = array("build-system", "requires")
        test_extra = array("project.optional-dependencies", "test")
        floor = [r for r in build if r.replace(" ", "").startswith("setuptools")]
        self.assertEqual(len(floor), 1, build)
        self.assertIn(floor[0], test_extra)
        self.assertIn("pytest", test_extra)

    def test_build_backend_and_core_gate_are_the_installed_cores_templates_byte_for_byte(self):
        template_dir = os.path.join(_autoload.core_gate()["installed"]["root"], "kit_template")
        for kit_copy, name in ((os.path.join(OPT, "_build_backend.py"), "_build_backend.py"), (_core_gate.__file__, "_core_gate.py")):
            self.assertEqual(open(kit_copy, "rb").read(), open(os.path.join(template_dir, name), "rb").read(), kit_copy)
        self.assertEqual([os.path.relpath(p, OPT) for p in glob.glob(os.path.join(OPT, "**", "_core_gate.py"), recursive=True)], [os.path.join("rfdiffusion3_opt", "_core_gate.py")])   # one copy, inside the package

    def test_pth_is_the_backends_generated_text_for_this_package_switch_and_tag(self):
        backend = _backend()
        text = open(PTH, encoding="utf-8").read()
        self.assertEqual(text, backend.pth_text("rfdiffusion3_opt", _autoload.ENV, _autoload.TAG, _autoload.EXIT_NOT_ACTIVE))
        self.assertEqual(backend.pth_fields(text), ("rfdiffusion3_opt", "RFDIFFUSION3_OPT", "rfdiffusion3-opt", 3))
        self.assertEqual(backend.PTH, os.path.basename(PTH))                        # the one *_autoload.pth beside the backend: what the wheel ships

    def test_restated_constants_and_the_exit_table_are_the_cores(self):
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, core_report.EXIT_NOT_ACTIVE)
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, _core_gate.EXIT_NOT_ACTIVE)
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE),
                         (core_report.EXIT_OK, core_report.EXIT_FAIL, core_report.EXIT_USAGE, core_report.EXIT_NOT_ACTIVE))
        self.assertEqual(report.PREFIX, core_report.prefix(report.TAG))
        self.assertEqual(report.not_active_line("x"), core_report.not_active_line(report.TAG, "x"))
        self.assertIs(modes.TABLE.__class__, __import__("opt_core.modes", fromlist=["ModeTable"]).ModeTable)

    def test_manifest_carries_the_core_block(self):
        """The manifest's "core" key is exactly core_manifest.core_block() — one producer; whatever fields the installed core's
        introspection carries this month ride along unexamined here (locking their shape is opt_core's own test, not the kit's)."""
        m = mf.build({"mode": "exact", "active": True, "env": {"RFD3_HOIST": "1"}}, hash_checkpoint=False)
        self.assertEqual(m["core"], core_manifest.core_block())
        self.assertEqual(m["core"]["version"], _autoload.core_gate()["installed"]["version"])
        self.assertEqual(m["schema"], "rfdiffusion3_opt.manifest/3")

    def test_jit_cache_key_is_the_cores_spelling(self):
        self.assertEqual(core_jit_cache.key(version="2.13.0", cuda="13.0", cc="9.0"), "torch2.13.0-cu130-sm90")   # the pinned stack's key (INDEX.md)


if __name__ == "__main__":
    unittest.main()
