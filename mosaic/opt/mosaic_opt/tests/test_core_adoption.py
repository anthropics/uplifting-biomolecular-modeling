"""Core locks: the kit runs on the shared core (`common/opt_core`, pinned in `opt/pyproject.toml` `[tool.opt_core]`). The pin table names
the core in this tree and is at or below the installed version; `core_gate()` returns that pin/installed pair for a passing core and refuses
by name (SystemExit 3, one `[mosaic-opt] NOT ACTIVE: reason=...` line) for a core below the pin or missing entirely; the generated .pth is
exactly the build backend's own template output; and the P1 row (`opt_core.jax_design.pcc`) is what both the driver route
(rows B/C) and the in-process route apply. No jax, no GPU. The gate through every entry route on an absent / older
core: `test_core_gate_routes.py`."""
import importlib.util
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

from . import core_src
import mosaic_opt
from mosaic_opt import _autoload, _core_gate, det, modes, report, stack


PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/mosaic_opt
OPT = os.path.dirname(PKG)                                                    # opt/
TREE = os.path.dirname(OPT)                                                   # mosaic/
PTH = os.path.join(OPT, "mosaic_opt_autoload.pth")


def kit_backend():
    """opt/_build_backend.py imported by path (the kit's own copy of the template: pth_text / pth_fields)."""
    spec = importlib.util.spec_from_file_location("_mosaic_opt_build_backend", os.path.join(OPT, "_build_backend.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCorePin(unittest.TestCase):
    def test_pin_table_names_the_core_in_this_tree(self):
        """[tool.opt_core] pins the core by path and version (a floor: the minimum core version the kit needs); the path is the tree's
        common/opt_core, and the installed core there is at least that version."""
        from opt_core.gates import core_pin, version_tuple
        pin = core_pin(os.path.join(OPT, "pyproject.toml"))
        self.assertEqual(set(pin), {"path", "version", "abs_path"})
        self.assertEqual(pin["abs_path"], core_src())
        self.assertTrue(os.path.isfile(os.path.join(pin["abs_path"], "opt_core", "__init__.py")), pin["abs_path"])
        import opt_core
        self.assertGreaterEqual(version_tuple(opt_core.__version__), version_tuple(pin["version"]),
                                 "the tree's installed core is older than opt/pyproject.toml [tool.opt_core]'s floor")

    def test_the_pin_is_the_installed_core(self):
        """`core_gate()` passes in this tree and returns the facts: the pin (read from the pyproject `stack.core_pin_path()` names) is at or
        below the installed core's version, and that core is the `opt_core` this process imports."""
        import opt_core
        from opt_core.gates import version_tuple
        facts = mosaic_opt.core_gate()
        self.assertEqual(set(facts), {"pinned", "installed", "tag"})
        self.assertEqual(set(facts["pinned"]), {"path", "version", "pyproject"})
        self.assertEqual(set(facts["installed"]), {"package_dir", "root", "version"})
        self.assertEqual(os.path.abspath(facts["pinned"]["pyproject"]), os.path.abspath(stack.core_pin_path()))
        self.assertEqual(stack.core_pin_path(), os.path.join(stack.opt_home(), "pyproject.toml"))
        self.assertGreaterEqual(version_tuple(facts["installed"]["version"]), version_tuple(facts["pinned"]["version"]))
        self.assertEqual(facts["installed"]["version"], opt_core.__version__)
        self.assertEqual(os.path.abspath(facts["installed"]["package_dir"]), os.path.dirname(os.path.abspath(opt_core.__file__)))

    def test_a_core_below_the_floor_is_refused_by_name(self):
        """The pin is a FLOOR (`>=`): a core OLDER than `[tool.opt_core] version` and no core at all are refused — `core_gate()` writes ONE
        `[mosaic-opt] NOT ACTIVE: reason=...` line naming both sides (the floor as `pinned >= v<floor>`, the installed version) and raises
        SystemExit(3) (`CoreGateRefused`); the installed core (at or above the floor) and any NEWER core pass. The gate anchors on this
        package's real pyproject, so the installed side is what varies here."""
        from opt_core.gates import core_pin, version_tuple
        real = _core_gate.installed_core()
        self.assertIsNotNone(real)
        floor = core_pin(os.path.join(OPT, "pyproject.toml"))["version"]                     # the floor every refusal names — not the installed version (a newer core is normal)
        self.assertGreaterEqual(version_tuple(real["version"]), version_tuple(floor))
        older = "0.2.5"
        self.assertLess(version_tuple(older), version_tuple(floor))
        cases = [(lambda: dict(real, version=older), "core_mismatch", [f"pinned >= v{floor}", f"installed v{older}"]),
                 (lambda: None, "core_missing:opt_core", ["nothing importable as opt_core", f"pinned >= v{floor}"])]
        for fake, reason, words in cases:
            err = io.StringIO()
            with mock.patch.object(_core_gate, "installed_core", fake), redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                mosaic_opt.core_gate()
            self.assertEqual(cm.exception.code, 3); self.assertIsInstance(cm.exception, _core_gate.CoreGateRefused)
            self.assertEqual(cm.exception.reason, reason.split(":")[0])
            lines = err.getvalue().splitlines()
            self.assertEqual(len(lines), 1, lines); self.assertTrue(lines[0].startswith(f"[mosaic-opt] NOT ACTIVE: reason={reason}"), lines[0])
            for w in words:
                self.assertIn(w, lines[0])
        head = version_tuple(floor)
        newer = ".".join(str(x) for x in (head[0], head[1] + 1, 0, 0))                         # a core above the floor (the next minor): passes, its own version reported
        at_floor = dict(real, version=floor)
        for ok in (real, at_floor, dict(real, version=newer)):
            with mock.patch.object(_core_gate, "installed_core", lambda ok=ok: ok), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mosaic_opt.core_gate()["installed"]["version"], ok["version"])
            self.assertEqual(err.getvalue(), "")                                                 # a pass prints nothing

    def test_pth_is_the_backends_generated_text(self):
        """opt/mosaic_opt_autoload.pth is exactly `python opt/_build_backend.py mosaic_opt MOSAIC_OPT mosaic-opt`: the header naming the three and
        the one guarded import of mosaic_opt._autoload (the backend refuses to build a wheel with any other text)."""
        b = kit_backend()
        text = open(PTH, encoding="utf-8").read()
        self.assertEqual(text, b.pth_text("mosaic_opt", _autoload.ENV, mosaic_opt.TAG))
        self.assertEqual(b.pth_fields(text), ("mosaic_opt", "MOSAIC_OPT", "mosaic-opt", report.EXIT_NOT_ACTIVE))
        self.assertEqual(b.PTH, "mosaic_opt_autoload.pth")
        self.assertIn("import mosaic_opt._autoload", text)


class TestP1IsTheSharedPrimitive(unittest.TestCase):
    """P1 = opt_core.jax_design.pcc: the stack key, the autotune file name, and the row's variables."""

    def test_stack_key(self):
        from opt_core.jax_design import pcc
        self.assertEqual(modes.jit_cache_key("0.10.2", "0.10.2", "0.10.2", "NVIDIA H100 80GB HBM3"), "jax0.10.2-jaxlib0.10.2-cuda12plugin0.10.2-nvidia-h100-80gb-hbm3")   # the pinned stack (stock/PINS.json pins)
        self.assertEqual(modes.jit_cache_key("0.10.2", "0.10.2", "0.10.2", "NVIDIA H100 80GB HBM3"), pcc.key("0.10.2", "0.10.2", "0.10.2", "NVIDIA H100 80GB HBM3"))
        self.assertTrue(modes.jit_cache_key("0.10.2", "0.10.2", "0.10.2", "NVIDIA H100 NVL").endswith("-nvidia-h100-nvl"))   # the product name, not the compute capability: P1 files are per GPU type
        for missing in (dict(gpu_name=""), dict(plugin_version=None)):                       # a missing part is refused by name, never an 'unknown'/'none' key
            kw = dict(jax_version="0.10.2", jaxlib_version="0.10.2", plugin_version="0.10.2", gpu_name="NVIDIA H100 80GB HBM3"); kw.update(missing)
            if missing.get("plugin_version", "") is None and __import__("opt_core.gates", fromlist=["dist_version"]).dist_version("jax-cuda12-plugin"):
                continue                                                                       # a box with the plugin installed reads it instead
            with mock.patch("opt_core.gates.nvidia_smi_probe", lambda *a, **k: {}), self.assertRaises(RuntimeError):   # no GPU answers here whatever the box: the missing part is refused by name
                modes.jit_cache_key(**kw)
        self.assertEqual(modes.AUTOTUNE_FILENAME, pcc.AUTOTUNE_FILENAME)
        self.assertIs(stack.p1_lever(), pcc)

    def test_rows_export_what_pcc_composes(self):
        """The exact row (C, load) and the populate row (B, dump) export exactly pcc.env's variables for the same directory — the driver route
        (the row, modes.ROWS) and the in-process route (pcc.enable) put the same P1 in force."""
        import tempfile
        from opt_core.jax_design import pcc
        p1_vars = {pcc.CACHE_DIR_VAR, pcc.XLA_FLAGS_VAR, *pcc.STORE_ALL}
        root = tempfile.mkdtemp(prefix="mosaic_opt_pcc_")
        for phase, autotune in (("warm", "load"), ("populate", "dump")):
            d = os.path.join(root, phase, "xla_cache"); os.makedirs(d)
            if autotune == "load":                                                            # a warm shape: the autotune file exists (pcc loads or refuses)
                open(os.path.join(d, modes.AUTOTUNE_FILENAME), "wb").write(b"x")
            res = modes.resolve("exact", cache_dir=d, features="$F1", features_sha="$FSHA", phase=phase)
            row_env = {k: v for k, v in res.env.items() if k in p1_vars}
            self.assertEqual(row_env, pcc.env(d, autotune=autotune, environ={}), (phase, res.row))
            self.assertEqual(set(res.env) - p1_vars, set(), f"the row exports a variable pcc does not know: {set(res.env) - p1_vars}")
        for word in ("fast", "big"):                                                        # the tier rows: P1 transparent = pcc's tolerance-class exports (the cache, store-all, NO autotune flag)
            d = os.path.join(root, f"xla_cache_{word}")
            res = modes.resolve(word, cache_dir=d)
            self.assertEqual(res.env, pcc.env(d, autotune="off", environ={}), (word, res.row))
            self.assertNotIn(pcc.XLA_FLAGS_VAR, res.env)

    def test_in_process_route_puts_the_rows_settings_in_force(self):
        """pcc.enable (the in-process route) writes the row's variables AND mirrors them into a live jax.config: the config names it updates are
        the variables' own (JAX reads `jax_x` from `JAX_X`), with the same values — one P1, two routes."""
        import sys, tempfile, types
        from opt_core.jax_design import pcc
        d = os.path.join(tempfile.mkdtemp(prefix="mosaic_opt_pcc_"), "xla_cache")
        updates = {}
        fake_jax = types.SimpleNamespace(config=types.SimpleNamespace(update=lambda k, v: updates.__setitem__(k, v), jax_compilation_cache_dir=None))
        bridge = types.ModuleType("jax._src.xla_bridge"); bridge.backends_are_initialized = lambda: False
        saved = sys.modules.get("jax._src.xla_bridge"); sys.modules["jax._src.xla_bridge"] = bridge
        try:
            environ = {}
            rec = pcc.enable(d, autotune="dump", environ=environ, jax_module=fake_jax)
        finally:
            if saved is None:
                del sys.modules["jax._src.xla_bridge"]
            else:
                sys.modules["jax._src.xla_bridge"] = saved
        res = modes.resolve("exact", cache_dir=d, features="$F1", features_sha="$FSHA", phase="populate")
        row_env = {k: v for k, v in res.env.items() if k != pcc.XLA_FLAGS_VAR}
        self.assertEqual(rec["applied_via"], "environ+jax.config")
        self.assertEqual({k: v for k, v in environ.items() if k != pcc.XLA_FLAGS_VAR}, row_env)              # the variables: the row's
        self.assertEqual(environ[pcc.XLA_FLAGS_VAR], res.env[pcc.XLA_FLAGS_VAR])                              # the pin: the row's flag
        self.assertEqual({k.upper() for k in updates}, set(row_env))                                          # jax.config names == the variables' names
        for k, v in updates.items():
            self.assertEqual(float(v) if isinstance(v, (int, float)) else v, float(row_env[k.upper()]) if isinstance(v, (int, float)) else row_env[k.upper()], k)


class TestDetRecipeShape(unittest.TestCase):
    def test_levels_are_recipes(self):
        from opt_core.det import Recipe
        self.assertEqual(set(det.RECIPES), set(det.LEVELS))
        self.assertTrue(all(isinstance(r, Recipe) for r in det.RECIPES.values()))
        self.assertEqual(dict(det.RECIPES[1].env), det.ENV); self.assertEqual(dict(det.RECIPES[0].env), {})
        self.assertEqual([r.unset for r in det.RECIPES.values()], [(), ()])          # refusals (MUST_BE_UNSET) are never silent unsets
        self.assertEqual(det.apply_env({"PYTHONUNBUFFERED": "0"}, 1), {"PYTHONUNBUFFERED": "0"})              # a set value stands; the recipe adds no allocator variable
        self.assertEqual(det.apply_env({}, 1), {"PYTHONUNBUFFERED": "1"}); self.assertFalse(any(k.startswith("XLA_PYTHON_CLIENT") for k in det.ENV))
        self.assertEqual(det.apply_env({"A": "1"}, 0), {"A": "1"})
        self.assertIn("PYTHONUNBUFFERED", det.describe(1)); self.assertNotIn("XLA_PYTHON_CLIENT", det.describe(1)); self.assertTrue(det.describe(1).startswith("det=1 "))


if __name__ == "__main__":
    unittest.main()
