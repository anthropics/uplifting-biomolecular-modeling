"""The mode table: off | exact | fast | big (modes.TABLE: exact = DEVICE_RESIDENT, fast = DEVICE_RESIDENT + SUBBATCH + TRIMUL_PALLAS + AF_PALLAS_ATTN + PALLAS_MSA + TRIATTN_XLA, big = fast + ROWPAIR), the default mode `fast` (the house default wherever a fast composition ships), the kit's own switch as the mode's
environment (read from the carried kit file, never transcribed), the documented pairs with _autoload. No jax, no GPU."""
import ast
import os
import sys
import unittest

from colabfold_opt import _autoload, cli, modes, registry, report, stack

KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "af2_pallas_flash")
KIT_MODULE_PATH = os.path.join(KIT, "af2_pallas_flash", "af2_pallas_attn.py")
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                  # opt/: registry kit_file of the package lever is relative to it
import opt_core as _opt_core                                                                                 # noqa: E402
CORE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(_opt_core.__file__)))                         # common/opt_core/: registry kit_file of the kernel's switches is relative to it


class TestModeTable(unittest.TestCase):
    def test_table(self):
        self.assertEqual(tuple(modes.MODES), ("off", "exact", "fast", "big"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")                   # the house default: fast wherever a fast composition ships (exact only where no fast mode exists)
        self.assertIn(modes.DEFAULT_MODE, modes.MODES)
        self.assertEqual(tuple(modes.KIT_MODE_NAMES), ("exact", "fast", "big"))
        self.assertEqual(set(modes.TABLE), set(modes.MODES))

    def test_unknown_refused_by_name(self):
        with self.assertRaises(modes.UnsupportedMode) as cm:
            modes.resolve("turbo")
        self.assertEqual(str(cm.exception), "'turbo' is not a mode (off|exact|fast|big; opt/colabfold_opt/modes.py)")
        self.assertTrue(issubclass(modes.UnsupportedMode, ValueError))

    def test_rows(self):
        off, exact, fast = modes.resolve("off"), modes.resolve("exact"), modes.resolve("fast")
        self.assertEqual((off["levers"], off["kit_env"], off["tier"]), ((), {}, "stock"))
        self.assertEqual((exact["levers"], exact["kit_env"], exact["tier"]), (("DEVICE_RESIDENT",), {}, 1))               # the transfer lever alone: no carried-kit switch
        self.assertEqual((fast["levers"], fast["kit_env"], fast["tier"]), (("DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"), {"AF_PALLAS_ATTN": "1"}, 2))
        self.assertEqual((modes.MSA_LEVER, modes.MSA_LEVER_MODULE), ("PALLAS_MSA", "colabfold_opt.msa_attn"))
        self.assertEqual((modes.TRIATTN_LEVER, modes.TRIATTN_LEVER_MODULE), ("TRIATTN_XLA", "colabfold_opt.triattn_xla"))
        self.assertEqual((modes.COL_LEVER, modes.COL_LEVER_MODULE), ("MSA_COL_CUDNN", "colabfold_opt.msa_col_cudnn"))
        self.assertEqual(modes.SUPERSEDES, {"TRIATTN_XLA": ("AF_PALLAS_ATTN",), "MSA_COL_CUDNN": ("PALLAS_MSA",)})
        self.assertEqual(modes.ONE_DEVICE_PAIR_LEVERS, ("TRIMUL_PALLAS", "TEMPL_DEDUP", "TRANSITION"))   # 0.2.23: TRIATTN_XLA and MSA_COL_CUDNN stay on at --n_gpu P > 1; self.assertEqual(tuple(l for l, _ in modes.ONE_DEVICE_LEVERS), modes.ONE_DEVICE_PAIR_LEVERS)                       # the pair-stack levers that step aside by name under big at --n_gpu P>1
        self.assertEqual((modes.HOST_LEVER, modes.LEVER, modes.HOST_LEVER_MODULE), ("DEVICE_RESIDENT", "AF_PALLAS_ATTN", "colabfold_opt.device_resident"))
        self.assertEqual((modes.SUBBATCH_LEVER, modes.SUBBATCH_LEVER_MODULE, modes.TRIMUL_LEVER, modes.TRIMUL_LEVER_MODULE, modes.PAIR_MUL_LEVERS),
                         ("SUBBATCH", "colabfold_opt.subbatch", "TRIMUL_PALLAS", "colabfold_opt.trimul_pallas", ("TRIMUL_PALLAS",)))
        big = modes.resolve("big")                                                                                  # fast's lever set + the row-sharded pair stack, fast-class numerics
        self.assertEqual((big["levers"], big["kit_env"], big["tier"]), (("DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION", "ROWPAIR"), {"AF_PALLAS_ATTN": "1"}, 2))
        self.assertEqual(big["levers"][:9], fast["levers"]); self.assertEqual(big["tier"], fast["tier"])
        self.assertEqual((modes.TP_LEVER, modes.TP_LEVER_MODULE, modes.ENV_N_GPU, modes.N_GPU_SUPPORTED, modes.TP_MODES),
                         ("ROWPAIR", "colabfold_opt.big", "COLABFOLD_OPT_N_GPU", (1, 2, 4, 8), ("big",)))

    def test_kit_switch_is_the_kits_own(self):
        """The mode's environment is the kit's import-time switch: af2_pallas_attn.py:124 `os.environ.get("AF_PALLAS_ATTN", "0") == "1"`."""
        lines = open(KIT_MODULE_PATH, encoding="utf-8").read().splitlines()
        self.assertEqual(lines[123].strip(), 'if os.environ.get("AF_PALLAS_ATTN", "0") == "1":')
        self.assertEqual(lines[124].strip(), "enable()")
        self.assertEqual(modes.KIT_SWITCH, "AF_PALLAS_ATTN"); self.assertEqual(modes.KIT_SWITCH_ON, "1")
        self.assertEqual(modes.KIT_MODULE, "af2_pallas_attn")
        tree = ast.parse(open(KIT_MODULE_PATH, encoding="utf-8").read())
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "enable"][0]
        self.assertEqual([a.arg for a in fn.args.args], ["modules_list", "all_calls"])           # enable() takes no required argument
        self.assertEqual(fn.lineno, 69)

    def test_from_env(self):
        self.assertEqual(modes.from_env({}), "off")
        self.assertEqual(modes.from_env({"COLABFOLD_OPT": "fast"}), "fast")
        self.assertEqual(modes.from_env({"COLABFOLD_OPT": " Fast "}), "fast")

    def test_n_gpu_from_env(self):
        self.assertEqual(modes.n_gpu_from_env({}), 1)                                            # absent == --n_gpu 1
        self.assertEqual(modes.n_gpu_from_env({"COLABFOLD_OPT_N_GPU": "4"}), 4)
        for bad in ("0", "-2", "two", "2.0", " 02"):
            with self.assertRaises(ValueError):
                modes.n_gpu_from_env({"COLABFOLD_OPT_N_GPU": bad})


class TestRegistry(unittest.TestCase):
    def test_registry_and_modes_agree(self):
        self.assertEqual(registry.IN_MODE, (modes.HOST_LEVER, modes.SUBBATCH_LEVER, modes.TRIMUL_LEVER, modes.LEVER, modes.MSA_LEVER, modes.TRIATTN_LEVER, modes.COL_LEVER, modes.TEMPL_LEVER, modes.TRANSITION_LEVER, modes.TP_LEVER))   # the levers a mode applies, in application order
        self.assertEqual(tuple(modes.resolve("big")["levers"]), registry.IN_MODE); self.assertEqual(tuple(modes.resolve("fast")["levers"]), registry.IN_MODE[:9])
        self.assertEqual(modes.resolve("off")["levers"], ())
        self.assertEqual(registry.MARKERS, {"AF_PALLAS_ATTN": registry.PATCH_MARKER, "DEVICE_RESIDENT": registry.HOST_MARKER, "SUBBATCH": registry.SUBBATCH_MARKER,
                                            "TRIMUL_PALLAS": registry.TRIMUL_MARKER, "PALLAS_MSA": registry.MSA_MARKER, "TRIATTN_XLA": registry.TRIATTN_MARKER,
                                            "MSA_COL_CUDNN": registry.COL_MARKER, "TEMPL_DEDUP": registry.TEMPL_MARKER, "TRANSITION": registry.TRANSITION_MARKER, "ROWPAIR": registry.TP_MARKER})
        self.assertEqual(registry.NOT_WIRED, ("AF_PALLAS_ATTN_ALL", "AF_PALLAS_ATTN_PRECISE_BWD", "AF_PALLAS_ATTN_DBIAS_F32", "AF_PALLAS_ATTN_BWD_BATCH_CHUNK"))
        self.assertTrue(registry.SWITCH.startswith(f"{modes.KIT_SWITCH}={modes.KIT_SWITCH_ON} "))
        for n, lv in registry.LEVERS.items():
            self.assertEqual(n, lv.name); self.assertIn(lv.cls, ("forward", "datapath", "serving", "orchestration"))
            home = PKG_PARENT if lv.kit_file.startswith("colabfold_opt/") else (CORE_PARENT if lv.kit_file.startswith("opt_core/") else KIT)   # the package's own levers live in the package, the kernel's switches in the shared core's carried kernel, the rest in the carried kit
            self.assertTrue(os.path.isfile(os.path.join(home, lv.kit_file)), lv.kit_file)
            self.assertTrue(lv.lines and lv.tier and lv.what)

    def test_registry_lines_are_the_kit_files(self):
        """Every switch the carried adapter reads from the environment (grep of os.environ.get over af2_pallas_flash/af2_pallas_attn.py) is a
        registry entry, and every registry entry citing the shared core's carried kernel (opt_core/kernels/pallas_attn/af2_flash_pallas.py —
        the ONE kernel module of a process, stack.route_kernel_to_core) sits on the cited line; the kernel's further switches (CORE_ONLY: its
        f32-precision and gradient-mode knobs) are the core's own surface, set by no mode and absent from this kit's registry by name."""
        import re
        rx = re.compile(r'os\.environ\.get\("(AF_PALLAS_ATTN[A-Z0-9_]*)"')
        def scan(path, label):
            out = {}
            for i, ln in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
                for m in rx.finditer(ln):
                    out.setdefault(m.group(1), []).append(f"{label}:{i}")
            return out
        kit_seen = scan(os.path.join(KIT, "af2_pallas_flash", "af2_pallas_attn.py"), "af2_pallas_attn.py")
        core_seen = scan(os.path.join(CORE_PARENT, "opt_core", "kernels", "pallas_attn", "af2_flash_pallas.py"), "af2_flash_pallas.py")
        self.assertEqual(sorted(os.listdir(os.path.join(KIT, "af2_pallas_flash"))), ["af2_pallas_attn.py"])              # the carried kit is the adapter alone: the kernel is the core's copy, no second copy here
        self.assertEqual(sorted(kit_seen), sorted(n for n, lv in registry.LEVERS.items() if lv.kit_file.startswith("af2_pallas_flash/")))
        self.assertEqual(kit_seen["AF_PALLAS_ATTN"], ["af2_pallas_attn.py:124"]); self.assertEqual(kit_seen["AF_PALLAS_ATTN_ALL"], ["af2_pallas_attn.py:77"])
        core_rows = {n: lv for n, lv in registry.LEVERS.items() if lv.kit_file.startswith("opt_core/")}
        self.assertEqual(sorted(core_rows), ["AF_PALLAS_ATTN_BWD_BATCH_CHUNK", "AF_PALLAS_ATTN_DBIAS_F32", "AF_PALLAS_ATTN_PRECISE_BWD"])
        for n, lv in core_rows.items():
            self.assertEqual(core_seen[n], [lv.lines], n)                                                                 # the cited line is the line that reads it
        CORE_ONLY = {"AF_PALLAS_ATTN_BWD_F32_PRECISION", "AF_PALLAS_ATTN_DBIAS", "AF_PALLAS_ATTN_DQ", "AF_PALLAS_ATTN_F32_PRECISION"}
        self.assertEqual(set(core_seen) - set(core_rows), CORE_ONLY)                                                       # a new kernel switch lands here by name, never silently
        seen = {**kit_seen, **{n: core_seen[n] for n in core_rows}}
        for n, locs in seen.items():
            for loc in locs:
                self.assertIn(loc, registry.LEVERS[n].lines if n != modes.LEVER else registry.SWITCH.replace("124-125", "124"), n)

    def test_probes_on_the_stub_kit(self):
        import importlib, tempfile, shutil
        from colabfold_opt.tests import _stubs
        tmp = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, tmp, True)
        kit = _stubs.make_kit_dir(tmp)
        mods, saved = _stubs.install()
        self.addCleanup(_stubs.remove, saved)
        sys.path.insert(0, os.path.join(kit, "af2_pallas_flash")); self.addCleanup(sys.path.remove, os.path.join(kit, "af2_pallas_flash"))
        os.environ.pop("AF_PALLAS_ATTN", None)
        km = importlib.import_module(modes.KIT_MODULE); self.addCleanup(sys.modules.pop, modes.KIT_MODULE, None)
        self.assertEqual(registry.applied_in_mode(km), []); self.assertFalse(registry.patch_marker_present()); self.assertFalse(registry.marker_present(modes.HOST_LEVER))
        km.enable()
        self.assertEqual(registry.applied_in_mode(km), [modes.LEVER]); self.assertTrue(registry.patch_marker_present())
        from colabfold_opt import device_resident
        self.addCleanup(device_resident.reset_for_tests)
        device_resident.enable()
        self.assertEqual(registry.applied_in_mode(km), [modes.HOST_LEVER, modes.LEVER]); self.assertTrue(registry.marker_present(modes.HOST_LEVER))
        self.assertEqual({n: v for n, v in registry.applied(km).items() if n in registry.NOT_WIRED}, {n: False for n in registry.NOT_WIRED})


class TestDocumentedPairs(unittest.TestCase):
    def test_autoload_pairs(self):
        self.assertFalse(hasattr(_autoload, "MODES"))                                   # the finder is mode-agnostic: modes.resolve decides at the trigger import
        self.assertEqual(_autoload.ENV, modes.ENV)
        self.assertEqual(_autoload.TRIGGERS, (stack.TRIGGER_MODULE,))
        self.assertEqual((_autoload.PREFIX, _autoload.EXIT_NOT_ACTIVE), (report.PREFIX, report.EXIT_NOT_ACTIVE))
        from colabfold_opt import xla_cache
        self.assertIn(xla_cache.ROOT_ENV, _autoload.ENV_NAMES)                             # the config's cache root is a declared name (an undeclared one is refused at the trigger import)
        for name in (modes.ENV + "_DATA_DIR", modes.ENV + "_WORK_DIR", modes.ENV_N_GPU):
            self.assertIn(name, _autoload.ENV_NAMES)

    def test_stock_proof_prefixes_are_derived(self):
        self.assertEqual(stack.PACKAGE_ENV_PREFIXES, (modes.ENV, modes.KIT_SWITCH))
        pins = stack.pins()
        self.assertEqual(pins["stock_proof"]["must_be_absent_prefixes"], list(stack.PACKAGE_ENV_PREFIXES))


class TestPartialExitLine(unittest.TestCase):
    """The house line of a partial activation: the fixed parts byte-literal, one formatter (report.py) every entry point calls, the
    exit code in its one home (report.EXIT_NOT_ACTIVE, the value cli / stack / manifest / _autoload exit with)."""

    def test_literal_parts_and_one_formatter(self):
        self.assertEqual(report.partial_exit_line("AF_PALLAS_ATTN: why"), "[colabfold-opt] NOT ACTIVE: partial activation — AF_PALLAS_ATTN: why; exit 3")
        self.assertEqual(report.lever_refused_line({"lever_fallbacks": {"FPF_TRIMUL": "w"}}),
                         "[colabfold-opt] NOT ACTIVE: FPF_TRIMUL=w: the lever cannot run on this GPU (no tile table for its compute capability) — a mode is all of its levers; exit 3 (--mode off runs stock)")
        self.assertEqual((report.EXIT_NOT_ACTIVE, cli.EXIT_NOT_ACTIVE), (3, 3))
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in sorted(os.listdir(pkg)):
            if not name.endswith(".py") or name == "report.py":
                continue
            src = open(os.path.join(pkg, name), encoding="utf-8").read()
            tree = ast.parse(src)
            docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                          and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for node in ast.walk(tree):                                    # no second spelling: the literal parts live in report.py only (docstrings describe them)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    self.assertNotIn("partial activation —", node.value, name)
                    self.assertNotIn("(--mode off runs stock)", node.value, name)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.exit":
                    self.assertFalse(isinstance(node.args[0], ast.Constant) if node.args else False, f"{name}: sys.exit with a typed code")

