"""The core this kit stands on (opt_core, pinned in opt/pyproject.toml [tool.opt_core] as path + minimum version): the pin is satisfied
by the installed core's own `__version__` through the package's core pin gate (`boltzgen_opt.core_gate` -> `_core_gate.gate`), which
agrees with the core's own live-tree comparison; an older or absent core is `SystemExit(3)` with the gate's one NOT ACTIVE line; the
gate's tag is the package's one tag; `opt/_build_backend.py` and `opt/boltzgen_opt/_core_gate.py` are the core's kit templates byte for
byte and the `.pth` is the backend's generated text; the exit table and the family's partial lines are the core's, the autoload module's
restated constant equals the core's, the manifest carries the core block, the tree home follows the core's rule, the late-activation
counter is the core's."""
import contextlib
import importlib.util
import io
import os
import unittest
from unittest import mock

import boltzgen_opt
import opt_core
from opt_core import gates as core_gates, home as core_home, instances as core_instances, manifest as core_manifest, report as core_report

from .. import _autoload, _core_gate, codes, manifest as mf, modes, report, stack

OPT = os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__)))
CORE = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
PYPROJECT = os.path.join(OPT, "pyproject.toml")
PTH = "boltzgen_opt_autoload.pth"


def _backend():
    """The kit's own opt/_build_backend.py, loaded by path (it lives beside pyproject.toml, outside the package)."""
    spec = importlib.util.spec_from_file_location("_boltzgen_opt_build_backend", os.path.join(OPT, "_build_backend.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCoreAdoption(unittest.TestCase):
    def test_core_pin_holds_the_installed_core(self):
        facts = boltzgen_opt.core_gate()                                       # the package's pin gate: passes here, and returns the facts
        self.assertEqual(facts["pinned"]["pyproject"], PYPROJECT)
        self.assertEqual(facts["pinned"]["path"], "../../common/opt_core")
        self.assertEqual(facts["pinned"]["version"], opt_core.__version__)
        self.assertEqual(facts["installed"]["version"], opt_core.__version__)
        self.assertEqual(os.path.normpath(facts["installed"]["root"]), os.path.normpath(CORE))
        live = core_gates.core_pin_check(PYPROJECT)                            # the core's live-tree comparison agrees with the gate it trusts
        self.assertTrue(live.ok, live.reason)
        self.assertEqual(live.details["imported"]["version"], facts["installed"]["version"])

    def test_the_gates_tag_is_the_packages_one_tag(self):
        self.assertEqual(boltzgen_opt.TAG, "boltzgen-opt")
        self.assertIs(report.TAG, boltzgen_opt.TAG); self.assertIs(_autoload.TAG, boltzgen_opt.TAG)     # one object: imported, never restated
        self.assertEqual(boltzgen_opt.core_gate()["tag"], boltzgen_opt.TAG)    # `[boltzgen-opt]` on the gate's line as on every other line
        self.assertEqual(report.PREFIX, f"[{boltzgen_opt.TAG}]")

    def _refusal(self, installed):
        err = io.StringIO()
        with mock.patch.object(_core_gate, "installed_core", installed), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            boltzgen_opt.enable("exact")
        self.assertEqual(cm.exception.code, codes.EXIT_NOT_ACTIVE)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_a_wrong_pin_is_exit_3_with_the_mismatch_line_naming_both_sides(self):
        """FLOOR semantics: an OLDER installed version refuses (never a same-version/different-identity fixture — identity is the git commit's)."""
        real = _core_gate.installed_core()
        line = self._refusal(lambda: dict(real, version="0.2.5"))
        self.assertTrue(line.startswith(f"{report.PREFIX} {report.NOT_ACTIVE}: reason=core_mismatch: opt_core pinned "), line)
        self.assertIn(f"opt_core pinned >= v{opt_core.__version__} at ../../common/opt_core, installed v0.2.5 at {real['root']}", line)

    def test_an_absent_core_is_exit_3_with_the_missing_line(self):
        line = self._refusal(lambda: None)
        self.assertEqual(line, f"{report.PREFIX} {report.NOT_ACTIVE}: reason=core_missing:opt_core (pinned >= v{opt_core.__version__} "
                                f"at ../../common/opt_core; nothing importable as opt_core on sys.path)")

    def test_build_backend_and_core_gate_are_the_core_templates_byte_for_byte(self):
        root = boltzgen_opt.core_gate()["installed"]["root"]                    # the installed core's directory: kit_template/ beside its package
        for copy, template in ((os.path.join(OPT, "_build_backend.py"), os.path.join(root, "kit_template", "_build_backend.py")),
                               (os.path.abspath(_core_gate.__file__), os.path.join(root, "kit_template", "_core_gate.py"))):
            with open(copy, "rb") as a, open(template, "rb") as b:
                self.assertEqual(a.read(), b.read(), copy)
        self.assertEqual(os.path.dirname(os.path.abspath(_core_gate.__file__)), os.path.dirname(os.path.abspath(boltzgen_opt.__file__)))   # inside the package

    def test_pth_is_the_backends_generated_text(self):
        backend = _backend()
        text = open(os.path.join(OPT, PTH), encoding="utf-8").read()
        self.assertEqual(text, backend.pth_text("boltzgen_opt", modes.ENV, report.TAG, codes.EXIT_NOT_ACTIVE))
        self.assertEqual(backend.pth_fields(text), ("boltzgen_opt", _autoload.ENV, report.TAG, _autoload.EXIT_NOT_ACTIVE))
        self.assertEqual(backend.PTH, PTH)

    def test_exit_table_and_partial_lines_are_the_cores(self):
        self.assertEqual((codes.EXIT_OK, codes.EXIT_FAIL, codes.EXIT_USAGE, codes.EXIT_NOT_ACTIVE),
                         (core_report.EXIT_OK, core_report.EXIT_FAIL, core_report.EXIT_USAGE, core_report.EXIT_NOT_ACTIVE))
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, core_report.EXIT_NOT_ACTIVE)
        self.assertEqual(report.PREFIX, core_report.prefix(report.TAG))
        d = report.partial_detail(["hoist"], {"hoist": "why"})
        self.assertEqual(report.partial_exit_line(["hoist"], {"hoist": "why"}), f"{report.PREFIX} NOT ACTIVE: partial activation — {d}; a mode is all of its levers: exit {core_report.EXIT_NOT_ACTIVE}")   # the refusal; the core's opt-out template (PARTIAL_ALLOWED) is not this kit's
        self.assertEqual(report.not_active_line(report.partial_reason(["hoist"], {"hoist": "why"})), report.partial_exit_line(["hoist"], {"hoist": "why"}))

    def test_manifest_carries_the_core_block(self):
        man = mf.build({"active": False, "mode": "off"}, command="check")
        self.assertEqual(man["core"], core_manifest.core_block())
        self.assertEqual(list(man)[:4], ["schema", "package", "package_version", "core"])
        self.assertEqual(man["core"]["version"], boltzgen_opt.core_gate()["installed"]["version"])

    def test_tree_home_is_the_cores_rule(self):
        saved = {k: os.environ.pop(k, None) for k in (stack.ENV_HOME, core_home.ENV_TREE)}
        try:
            self.assertEqual(stack.opt_home(), os.path.join(core_home.tree_home(stack.__file__), "opt"))
            self.assertEqual(stack.tree_home(), core_home.tree_home(stack.__file__))
            os.environ[core_home.ENV_TREE] = "/x/boltzgen"
            self.assertEqual(stack.opt_home(), "/x/boltzgen/opt")
            os.environ[stack.ENV_HOME] = "/y/opt"                              # the kit's own variable names opt/ and wins
            self.assertEqual((stack.opt_home(), stack.tree_home()), ("/y/opt", "/y"))
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_stack_key_string_is_the_cores_rule_and_unchanged_on_the_stack_of_record(self):
        from opt_core import jit_cache as core_jit_cache
        rec = stack.pins()["stack_of_record"]                                  # torch 2.13.0+cu130 on the pinned H100 (stock/PINS.json)
        tv, _, local = rec["torch"].partition("+")
        key = core_jit_cache.key(version=tv, cuda=local[2:], cc=str(stack.pins()["gpu_of_record"]["compute_capability"]))
        self.assertEqual(key, f"torch{tv}-cu{local[2:]}-sm90")
        self.assertEqual(key, "torch2.13.0-cu130-sm90")                       # the key configs/h100.env documents; a changed key would orphan the mounted warm caches
        self.assertEqual(stack.stack_key({"compute_cap": "9.0"}).rsplit("-sm", 1)[1], "90")

    def test_instance_counter_is_the_cores(self):
        self.assertEqual(stack.arm_instance_counter(), stack.INSTANCE_COUNTER_WORDS[core_instances.register_instance_counter(stack.BOLTZ_MODULE, stack.BOLTZ_CLASS)["method"]])
        self.assertEqual(stack.instance_count(), core_instances.instance_check(stack.BOLTZ_MODULE, stack.BOLTZ_CLASS)["n"])


if __name__ == "__main__":
    unittest.main()


class TestPinReadableWithoutTomllib(unittest.TestCase):
    """The `[tool.opt_core]` table reads back through the core's no-tomllib fallback reader (a python < 3.11 stack) exactly as through tomllib:
    a header line the fallback cannot enter (trailing text after `]`) would leave the pin unreadable there and the first gate refusing."""

    def test_fallback_reader_returns_the_pin(self):
        import builtins
        real_import = builtins.__import__

        def no_tomllib(name, *a, **k):
            if name == "tomllib":
                raise ModuleNotFoundError("tomllib masked: the python < 3.11 path")
            return real_import(name, *a, **k)

        builtins.__import__ = no_tomllib
        try:
            pin = core_gates.read_pin_table(PYPROJECT)
            gate_pin = _core_gate.read_table(PYPROJECT, "tool.opt_core")           # the gate's own reader on the same no-tomllib path
        finally:
            builtins.__import__ = real_import
        self.assertTrue(all(pin.get(k) for k in core_gates.PIN_KEYS), pin)
        with_tomllib = {k: v for k, v in core_gates.read_pin_table(PYPROJECT).items() if k in core_gates.PIN_KEYS}
        self.assertEqual({k: pin[k] for k in core_gates.PIN_KEYS}, with_tomllib)
        self.assertEqual({k: gate_pin[k] for k in _core_gate.PIN_KEYS}, with_tomllib)

    def test_header_line_stands_alone(self):
        with open(PYPROJECT, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
        self.assertIn("[tool.opt_core]", lines, "the [tool.opt_core] header stands alone on its line (the core's fallback reader matches the whole stripped line)")
