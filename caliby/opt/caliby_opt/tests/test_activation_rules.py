"""Activation rules and the late-activation rule, on fake site-packages trees built from the kits' own copies:
tree-state classification; both modes refused on a patched tree (NOT STOCK, naming the reinstall of the pinned upstream); exact on the pinned tree
plans the overlay (the kits' files under the upstream module names, digests proven) and enable() installs it; switch conflicts and
kit switches outside the row; the target-GPU comparison reported and never a refusal; late activation refused once a lever module is
imported; a second mode in the same process refused; the activation line and the exit tally are never silent (the tally is registered
by enable; its tree= is what the loaded lever modules actually are); the lever fields of the report. The pin and weights gates and the
tree digest (a subprocess over the real site-packages) are stubbed here (this box has no upstream)."""
import contextlib
import importlib.util
import os
import tempfile
import sys
import types
import unittest
from unittest import mock

from .. import ActivationError, activate, modes, overlay, report, stack, tree
from ._fixtures import FakeSitePackages, kit_sources

GPU_STUB = {"available": True, "name": "stub", "sm": "9.0", "memory_gb": 80, "torch": "2.6.0+cu124", "cuda": "12.4", "reason": None}
NO_GPU = {"available": False, "name": None, "sm": None, "reason": "no CUDA device", "torch": "0", "cuda": None}


def _stub_gates(monkey_stack):
    monkey_stack.enter_context(mock.patch.object(stack, "check_pins_detail", lambda: ([], {"caliby": {"pinned": True, "source": "stub"}, "protpardelle": {"pinned": True, "source": "stub"}})))
    monkey_stack.enter_context(mock.patch.object(stack, "weights_gate", lambda variant, p=None, env=None, ckpt=None: (None, {"stub": True, "ckpt": ckpt})))
    monkey_stack.enter_context(mock.patch.object(stack, "compiler", lambda: "/usr/bin/cc"))
    monkey_stack.enter_context(mock.patch.object(tree, "digest", lambda: (stack.pins()["tree_digest_upstream"]["sha256"], {})))   # the installed tree at the pin


class TestTreeState(unittest.TestCase):
    def setUp(self):
        self.sp = FakeSitePackages()

    def tearDown(self):
        self.sp.close()

    def test_states(self):
        self.assertEqual(stack.tree_state(), "stock")
        self.sp.set_state("exact")
        self.assertEqual(stack.tree_state(), "exact")
        f = stack.installed_files()
        self.assertEqual(f["api.py"]["state"], "addon")
        self.assertEqual(f["protpardelle_core_models.py"]["state"], "addon")
        self.sp.corrupt("api.py")
        self.assertEqual(stack.tree_state(), "unknown")
        self.sp.set_state("stock")
        self.sp.set_state("exact")
        # one upstream file among the add-on's is 'mixed'
        stock, addon = kit_sources()
        import shutil
        shutil.copyfile(stock["potts.py"], self.sp.path_of("potts.py"))
        self.assertEqual(stack.tree_state(), "mixed")

    def test_exact_without_protpardelle_package(self):
        self.sp.close()
        sp2 = FakeSitePackages(with_protpardelle=False)
        try:
            sp2.set_state("exact")
            self.assertEqual(stack.installed_files()["protpardelle_core_models.py"]["state"], "package-absent")
            self.assertEqual(stack.tree_state(), "exact")
            sp2.set_state("stock")
            self.assertEqual(stack.tree_state(), "stock")
        finally:
            sp2.close()
            self.sp = FakeSitePackages()

    def test_touched_set_and_digest_trees(self):
        """stack.TOUCHED covers the never-replaced clean_pdbs.py; the tree digest walks the three installed trees, imports nothing upstream and
        moves when one byte of a tree moves."""
        ours = set(stack.TOUCHED.values())
        self.assertIn(("caliby", "data/preprocessing/atomworks/clean_pdbs.py"), ours)
        self.assertEqual(stack.installed_files()["clean_pdbs.py"]["state"], "upstream")
        self.assertEqual(stack.DIGEST_TREES, ("caliby", "chroma", "protpardelle"))
        d1, per = stack.tree_digest()
        self.assertEqual(sorted(per), sorted(stack.DIGEST_TREES))
        self.assertTrue(all(per[t]["n_files"] > 0 for t in ("caliby", "chroma")), per)
        self.assertEqual(tree.digest()[0], d1)                        # one producer
        self.sp.set_state("exact")
        self.assertNotEqual(stack.tree_digest()[0], d1)
        self.sp.set_state("stock")
        self.assertEqual(stack.tree_digest()[0], d1)
        for m in ("caliby", "chroma", "protpardelle"):
            self.assertNotIn(m, sys.modules)

    def test_never_replaced_file_changed_is_neither_stock_nor_exact(self):
        self.sp.corrupt("clean_pdbs.py")
        self.assertEqual(stack.tree_state(), "unknown")
        self.sp.set_state("exact")
        self.assertEqual(stack.tree_state(), "exact")
        self.sp.corrupt("clean_pdbs.py")
        self.assertEqual(stack.tree_state(), "unknown")
        self.sp.set_state("stock")
        self.assertEqual(stack.tree_state(), "stock")


class TestOverlay(unittest.TestCase):
    """The import hook itself: the plan over the kits' files, their digests, the finder's answers, and the exit-state rule."""

    def setUp(self):
        self.sp = FakeSitePackages()
        overlay.uninstall()

    def tearDown(self):
        overlay.uninstall()
        self.sp.close()

    def test_plan_names_the_row_files_and_their_digests_prove(self):
        plan = overlay.plan()
        stock_src, addon = kit_sources()
        self.assertEqual(sorted(e["file"] for e in plan.values()), sorted(addon))                 # the add-on's eight files: the 8 replaced modules, nothing else
        self.assertNotIn("protpardelle.sample", plan)                                             # no upstream module is served stamped or copied: only lever files
        self.assertEqual(overlay.plan(mode="off"), {})                                            # the stock route installs no finder and serves nothing
        self.assertNotIn("caliby.data.preprocessing.atomworks.clean_pdbs", plan)                 # never replaced by a kit: passes through to the installed file
        self.assertEqual(plan["caliby.api"]["path"], addon["api.py"])
        self.assertEqual(plan["chroma.layers.complexity"]["file"], "complexity.py")
        self.assertEqual(plan["protpardelle.core.models"]["file"], "protpardelle_core_models.py")
        self.assertEqual(overlay.missing(plan), [])                                                # every row file is present at its planned path
        bad = overlay.missing(dict(plan, **{"caliby.api": dict(plan["caliby.api"], path=plan["caliby.api"]["path"] + ".missing")}))
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("api.py: missing at "), bad)

    def test_protpardelle_pair_planned_only_when_the_package_is_installed(self):
        self.sp.close()
        sp2 = FakeSitePackages(with_protpardelle=False)
        try:
            plan = overlay.plan()
            self.assertNotIn("protpardelle.core.models", plan)
            self.assertIn("caliby.api", plan)
            self.assertEqual(len(plan), 6)
        finally:
            sp2.close()
            self.sp = FakeSitePackages()

    def test_finder_serves_the_kit_files_and_passes_everything_else(self):
        fnd = overlay.install(overlay.plan())
        self.assertIs(sys.meta_path[0], fnd)
        self.assertIs(overlay.install(overlay.plan()), fnd)                                       # idempotent for the same plan
        stock_src, addon = kit_sources()
        spec = importlib.util.find_spec("caliby.api")
        self.assertEqual((spec.origin, spec.loader.path), (self.sp.path_of("api.py"), addon["api.py"]))   # __file__ = the installed module's path (api.py reads configs/ beside it); the code = the kit file
        self.assertEqual(importlib.util.find_spec("caliby.model.seq_denoiser.denoisers.seq_design.potts").loader.path, addon["potts.py"])
        spec = importlib.util.find_spec("caliby.data.preprocessing.atomworks.clean_pdbs")           # not in the plan: the installed file by the ordinary loader
        self.assertEqual(os.path.realpath(spec.origin), os.path.realpath(self.sp.path_of("clean_pdbs.py")))
        self.assertNotIsInstance(spec.loader, overlay._Loader)
        self.assertEqual(stack.tree_state(), "stock")                                              # nothing written into the tree
        self.assertTrue(overlay.uninstall())
        self.assertNotIn(fnd, sys.meta_path)

    def test_loader_executes_the_file_under_the_planned_name_without_bytecode(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "impl.py")
        with open(src, "w") as fh:
            fh.write("VALUE = 41 + 1\n")
        with open(src, "a") as fh:
            fh.write("import os\nHERE = os.path.dirname(os.path.abspath(__file__))\n")            # what caliby/api.py does with __file__
        installed = os.path.join(d, "site", "pkg", "impl.py")                                     # where the module's file would be installed (need not exist)
        name = "caliby_opt_overlay_probe.mod"
        sys.modules.setdefault("caliby_opt_overlay_probe", types.ModuleType("caliby_opt_overlay_probe")).__path__ = []
        fnd = overlay.OverlayFinder({name: {"file": "impl.py", "path": src, "installed": installed, "sha256": stack.sha256_file(src)}})
        sys.meta_path.insert(0, fnd)
        try:
            mod = importlib.import_module(name)
            self.assertEqual((mod.VALUE, mod.__file__, mod.__spec__.origin, mod.HERE), (42, installed, installed, os.path.dirname(installed)))
            self.assertEqual(mod.__spec__.loader.get_filename(name), src)                          # the code (and tracebacks) name the kit file
            self.assertEqual(overlay.source_of(mod), src)
            self.assertEqual(fnd.served, {name: src})
            self.assertFalse(os.path.isdir(os.path.join(d, "__pycache__")))                       # the loader writes no bytecode
        finally:
            sys.meta_path.remove(fnd)
            sys.modules.pop(name, None)
            sys.modules.pop("caliby_opt_overlay_probe", None)

    def test_install_refused_after_a_lever_module_is_imported(self):
        sys.modules["chroma.layers.complexity"] = types.ModuleType("chroma.layers.complexity")
        try:
            with self.assertRaises(overlay.OverlayError) as cm:
                overlay.install(overlay.plan())
            self.assertIn("late activation: chroma.layers.complexity", str(cm.exception))
        finally:
            del sys.modules["chroma.layers.complexity"]

    def test_exit_state_is_what_the_loaded_modules_are(self):
        self.assertEqual(overlay.exit_state("off"), "stock")                                       # nothing loaded, no hook
        plan = overlay.plan()
        overlay.install(plan)
        self.assertEqual(overlay.exit_state("exact"), "exact")                                     # hook installed, nothing loaded yet
        import importlib.machinery
        m = types.ModuleType("caliby.api")                                                        # an overlaid module: __file__ the installed path, code from the kit file
        m.__file__ = self.sp.path_of("api.py")
        m.__spec__ = importlib.machinery.ModuleSpec("caliby.api", overlay._Loader("caliby.api", plan["caliby.api"]["path"]), origin=m.__file__)
        sys.modules["caliby.api"] = m
        try:
            self.assertEqual(overlay.exit_state("exact"), "exact")
            self.assertTrue(overlay.loaded()["caliby.api"]["ok"])
            m.__spec__ = importlib.machinery.ModuleSpec("caliby.api", importlib.machinery.SourceFileLoader("caliby.api", m.__file__), origin=m.__file__)
            self.assertEqual(overlay.exit_state("exact"), "mixed")                                 # the installed upstream file's code under an exact activation
            overlay.uninstall()
            self.assertEqual(overlay.exit_state("off"), "stock")                                   # off: the installed file is right
            m.__spec__ = importlib.machinery.ModuleSpec("caliby.api", overlay._Loader("caliby.api", plan["caliby.api"]["path"]), origin=m.__file__)
            self.assertEqual(overlay.exit_state("off"), "mixed")                                   # off: kit code loaded is wrong
        finally:
            del sys.modules["caliby.api"]


class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.sp = FakeSitePackages()
        self.ctx = contextlib.ExitStack()
        _stub_gates(self.ctx)
        activate._REPORT = None
        overlay.uninstall()
        self.env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(stack.STOCK_ABSENT_PREFIXES):
                del os.environ[k]

    def tearDown(self):
        self.ctx.close()
        overlay.uninstall()
        self.sp.close()
        activate._REPORT = None
        os.environ.clear()
        os.environ.update(self.env)

    def test_exact_on_the_pinned_tree_would_activate(self):
        rep = activate.check("fast", "single", need_gpu=False)
        self.assertFalse(rep["would_refuse"], rep["reason"])
        self.assertEqual((rep["tree_state"], rep["site_packages_state"], rep["row_label"]), ("exact", "stock", "X"))
        self.assertEqual(rep["switches"]["CALIBY_X_CLEAN"], "0")
        self.assertEqual(rep["overlay"]["caliby.api"]["file"], "api.py")
        self.assertEqual(rep["overlay"]["caliby.api"]["path"], os.path.join("opt", "forward", stack.KIT_ADDON, "fast", "api.py"))   # relative to the tree, in the report
        self.assertEqual(len(rep["overlay"]), 8)                                                  # the row's eight kit files, nothing else
        self.assertNotIn("staged", rep)
        self.assertIsNone(overlay.installed())                                                    # check installs nothing
        core = rep["opt_core"]                                                                    # gate 0's facts: the installed core meets the pin, both sides recorded
        self.assertTrue(core["ok"], core)
        self.assertEqual(core["version"], core["pinned"]["version"])
        rep = activate.check("fast", "single", clean_workers=4, need_gpu=False)
        self.assertEqual(rep["row_label"], "XL")
        self.assertEqual(rep["switches"]["CALIBY_X_CLEAN"], "loader")

    def test_both_modes_refused_on_a_patched_tree(self):
        for state in ("exact",):
            self.sp.set_state(state)
            for mode in ("off", "fast"):
                rep = activate.check(mode, "single", need_gpu=False)
                self.assertTrue(rep["would_refuse"], (state, mode))
                self.assertTrue(rep["reason"].startswith("NOT STOCK"), rep["reason"])
                self.assertIn(tree.REINSTALL, rep["reason"])
                self.assertIn(f"tree state {state!r}", rep["reason"])
        self.sp.corrupt("clean_pdbs.py")
        self.assertIn("tree state 'unknown'", activate.check("fast", "single", need_gpu=False)["reason"])

    def test_kit_file_missing_refused(self):
        import shutil
        fake_fast = tempfile.mkdtemp()
        for name, src in kit_sources()[1].items():
            if name == "potts.py":
                continue                                                                          # left absent (presence, not digest -- a kit file is tracked in this tree)
            shutil.copyfile(src, os.path.join(fake_fast, name))
        with mock.patch.object(overlay, "overlay_dir", lambda: fake_fast):
            rep = activate.check("fast", "single", need_gpu=False)
        self.assertTrue(rep["would_refuse"])
        self.assertIn("overlay files not proven", rep["reason"])
        self.assertIn("potts.py: missing at ", rep["reason"])
        with mock.patch.object(overlay, "overlay_dir", lambda: fake_fast):
            off = activate.check("off", "single", need_gpu=False)
        self.assertFalse(off["would_refuse"])                                                     # off reads no kit file at all: a missing lever file is not its concern
        self.assertEqual(off["overlay"], {})                                                      # and serves nothing over the installed upstream files

    def test_core_pin_is_gate_zero(self):
        """An OLDER core or NO importable core: the core pin gate's one NOT ACTIVE line and SystemExit(3) — from check(), enable() and
        the strict (.pth) form alike, both modes — never an inactive report, never a resolved row, never a stock run. The gate located
        the core without importing it; here its locator is what the test bends. A newer core is not tested here (it passes — the pin
        is a floor, not policed by this gate zero at all)."""
        import contextlib
        import io
        from .. import _core_gate
        real = _core_gate.installed_core()
        pin = stack.core_gate()["pinned"]
        self.assertEqual(real["version"], pin["version"])                                        # on this box the installed core is at the pin
        stale = dict(real, version="0.2.5")
        for core, want in ((lambda: stale, f"[{stack.TAG}] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v{pin['version']} at {pin['path']}, installed v0.2.5 at "),
                           (lambda: None, f"[{stack.TAG}] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v{pin['version']} at {pin['path']}; nothing importable as opt_core on sys.path)")):
            with mock.patch.object(_core_gate, "installed_core", core), mock.patch.object(modes, "resolve") as resolve:
                for call in (lambda m: activate.check(m, "single", need_gpu=False), lambda m: activate.enable(m, "single"), lambda m: activate.enable(m, "single", strict=True)):
                    for mode in ("off", "fast"):
                        err = io.StringIO()
                        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                            call(mode)
                        self.assertEqual(cm.exception.code, 3)
                        self.assertEqual(err.getvalue().count("NOT ACTIVE"), 1, err.getvalue())
                        self.assertTrue(err.getvalue().startswith(want), (err.getvalue(), want))
                        self.assertIsNone(activate._REPORT)                                        # no report kept: nothing of the activation ran
                resolve.assert_not_called()                                                       # refused before the row is even resolved

    def test_exact_on_single_refused_by_name_and_ensemble32_resolves_under_exact_and_fast(self):
        """The mode table's one refusal through check() and enable() (the .pth route surfaces enable()'s ActivationError as its NOT ACTIVE
        line and exit 3: test_startup_autoload): exact on single is the words, would_refuse, no row, one NOT ACTIVE line, nothing installed;
        exact on ensemble32 resolves its row (this box then refuses on pins, not on the mode)."""
        rep = activate.check("exact", "single", need_gpu=False)
        self.assertEqual((rep["would_refuse"], rep["reason"], rep["row_label"], rep["switches"]), (True, modes.REFUSED[("exact", "single")], None, {}))
        with mock.patch.object(report, "say") as say, self.assertRaises(ActivationError) as cm:
            activate.enable("exact", "single")
        self.assertIn("exact refused: the single-sequence route differs from stock deterministically", str(cm.exception))
        say.assert_called_once_with("[caliby-opt] NOT ACTIVE: " + modes.REFUSED[("exact", "single")])
        self.assertIsNone(overlay.installed())
        activate._REPORT = None
        rep = activate.check("exact", "ensemble32", need_gpu=False)
        self.assertEqual((rep["row_label"], rep["tier"]), ("ensemble command", 1))
        activate._REPORT = None
        rep = activate.check("fast", "ensemble32", need_gpu=False)      # fast on ensemble32: the same row exact names (bit-identical satisfies fast)
        self.assertEqual((rep["would_refuse"], rep["row_label"], rep["tier"], rep["row_source"]), (False, "ensemble command", 2, "modes.py:ensemble32"))

    def test_off_with_a_switch_in_env_refused(self):
        os.environ["CALIBY_X_LCP"] = "1"
        rep = activate.check("off", "single", need_gpu=False)
        self.assertTrue(rep["would_refuse"])
        self.assertIn("CALIBY_X_LCP", rep["reason"])
        del os.environ["CALIBY_X_LCP"]
        os.environ["CALIBY_OPT"] = "exact"
        rep = activate.check("off", "single", need_gpu=False)
        self.assertIn("CALIBY_OPT", rep["reason"])


    def test_exact_switch_conflict_refused(self):
        os.environ["CALIBY_X_LCP"] = "0"
        rep = activate.check("fast", "single", need_gpu=False)
        self.assertTrue(rep["would_refuse"])
        self.assertIn("CALIBY_X_LCP", rep["reason"])
        os.environ["CALIBY_X_LCP"] = "1"                                             # the same value as the row's: no conflict (the row is exported whole either way)
        rep = activate.check("fast", "single", need_gpu=False)
        self.assertFalse(rep["would_refuse"], rep["reason"])
        self.assertEqual((rep["row_label"], rep["switches"]["CALIBY_X_LCP"]), ("X", "1"))
        self.assertNotIn("lcp_kernel", rep)                                          # no sub-selector of a row: a mode is all of its levers
        self.assertIn("CALIBY_X_LCP", rep["levers_applied"])

    def test_exact_kit_switch_outside_the_row_refused(self):
        os.environ["CALIBY_X_TIED_DET"] = "1"                          # the kit README's ensemble export, in a single-variant shell
        rep = activate.check("fast", "single", need_gpu=False)
        self.assertTrue(rep["would_refuse"])
        self.assertIn("outside the row", rep["reason"])
        self.assertEqual(rep["switches_extra_in_env"], {"CALIBY_X_TIED_DET": "1"})
        rep = activate.check("fast", "single", need_gpu=False, env={})                  # gating another environment: clean
        self.assertFalse(rep["would_refuse"], rep["reason"])
        rep = activate.check("exact", "ensemble32", need_gpu=False)                     # on the ensemble line it is the row's own switch
        self.assertFalse(rep["would_refuse"], rep["reason"])

    def test_target_gpu_is_reported_never_refused(self):
        os.environ[stack.ENV_TARGET_GPU] = "H100"
        other = {"available": True, "name": "NVIDIA A100-SXM4-80GB", "sm": "8.0", "memory_gb": 79.2, "torch": "2.6.0+cu124", "cuda": "12.4", "reason": None}
        pinned_triton = mock.patch.object(stack, "dist_version", lambda name: {"triton": stack.pins()["pinned_stack"]["triton"]}.get(name))   # this box's own triton is not the subject
        with mock.patch.object(stack, "gpu_info", lambda: other), pinned_triton:
            rep = activate.check("fast", "single")
        self.assertFalse(rep["would_refuse"], rep["reason"])
        self.assertEqual(rep["target_gpu"], "H100")
        self.assertIn("MODEL_OPT_TARGET_GPU=H100 but the GPU is NVIDIA A100-SXM4-80GB (sm 8.0, 79.2 GB)", rep["target_gpu_note"])
        self.assertIn("the levers engage as they are", rep["target_gpu_note"])                    # report only: the levers run or step aside by name
        self.assertIn(" target_gpu=H100 MISMATCH: MODEL_OPT_TARGET_GPU=H100 but the GPU is NVIDIA A100-SXM4-80GB", report.check_line(rep))
        h100 = dict(other, name="NVIDIA H100 80GB HBM3", sm="9.0")
        with mock.patch.object(stack, "gpu_info", lambda: h100), pinned_triton:
            rep = activate.check("fast", "single")
        self.assertIsNone(rep["target_gpu_note"])
        self.assertIn(" target_gpu=H100 -> would activate", report.check_line(rep))
        self.assertIsNone(activate.target_gpu_note(None, h100))
        self.assertIsNone(activate.target_gpu_note("H100", {"name": None}))

    def test_a_kit_mode_needs_a_gpu_and_names_a_missing_compiler(self):
        """No CUDA device: a kit mode cannot run (refused by name). No C compiler: named on the activation line (`notes=`), never a refusal at
        activation — CALIBY_X_LCP builds from the Triton cache, or the mode refuses by name at the kernel's first call; a torch / triton other
        than the tested stack is named the same way."""
        with mock.patch.object(stack, "gpu_info", lambda: NO_GPU):
            rep = activate.resolve_report("fast", "single", dry_run=False)
            self.assertIn("needs a CUDA GPU", rep["reason"])
        with mock.patch.object(stack, "compiler", lambda: None):
            rep = activate.check("fast", "single", need_gpu=False)
            self.assertFalse(rep["would_refuse"], rep["reason"])
            self.assertEqual(len(rep["notes"]), 1)
            self.assertTrue(rep["notes"][0].startswith("no C compiler on PATH: CALIBY_X_LCP builds from the Triton cache or the mode refuses by name"), rep["notes"])
            self.assertTrue(report.check_line(rep).split(" -> ")[0].endswith("notes=" + rep["notes"][0]))
        rep = activate.check("fast", "single", need_gpu=False)
        self.assertEqual((rep["notes"], report.notes_part(rep)), ([], ""))                          # the tested box: no notes, no suffix
        gpu = {"available": True, "name": "NVIDIA L40S", "sm": "8.9", "memory_gb": 44.4, "torch": "2.7.1+cu126", "cuda": "12.6", "reason": None}
        with mock.patch.object(stack, "dist_version", lambda name: {"triton": "3.3.1"}.get(name)):
            notes = activate.environment_notes({"levers_applied": ["CALIBY_X_LCP"], "compiler": "/usr/bin/cc", "gpu": gpu}, stack.pins())
        self.assertEqual(notes, ["torch 2.7.1+cu126 is not the kit's tested 2.6.0+cu124 (stock/PINS.json pinned_stack): levers engage as they are",
                                 "triton 3.3.1 is not the kit's tested 3.2.0: CALIBY_X_LCP builds its kernel with it or the mode refuses by name"])
        with mock.patch.object(stack, "dist_version", lambda name: {"triton": "3.2.0"}.get(name)):
            self.assertEqual(activate.environment_notes({"levers_applied": ["CALIBY_X_LCP"], "compiler": "/usr/bin/cc", "gpu": dict(gpu, torch="2.6.0+cu124")}, stack.pins()), [])

    def test_unknown_mode_variant(self):
        self.assertIn("unknown mode 'turbo' (expected off|exact|fast)", activate.check("turbo", "single", need_gpu=False)["reason"])
        self.assertIn("unknown variant", activate.check("exact", "e32", need_gpu=False)["reason"])
        with self.assertRaises(ValueError):
            os.environ["CALIBY_VARIANT"] = "ensemble32"
            activate.check("fast", "single", need_gpu=False)


    def test_late_activation_refused_by_name(self):
        sys.modules["caliby.api"] = types.ModuleType("caliby.api")
        try:
            rep = activate.resolve_report("fast", "single", dry_run=False, need_gpu=False)
            self.assertTrue(rep["would_refuse"])
            self.assertIn("late activation: caliby.api is already imported", rep["reason"])
            with self.assertRaises(activate.ActivationError):
                activate.enable("fast", "single")
        finally:
            del sys.modules["caliby.api"]


    def test_enable_off_then_exact_refused_and_lines(self):
        with mock.patch.object(stack, "gpu_info", lambda: NO_GPU):
            digest = stack.pins()["tree_digest_upstream"]["sha256"]
            with mock.patch.object(tree, "digest", lambda: (digest, {})):
                with mock.patch.object(report, "say") as say:
                    rep = activate.enable("off", "single")
                    self.assertFalse(rep["active"])
                    self.assertEqual(rep["mode"], "off")
                    line = say.call_args[0][0]
                    self.assertTrue(line.startswith("[caliby-opt] STOCK mode=off"), line)
                    self.assertIn("switches=none", line)
                    self.assertEqual(activate.enable("off", "single"), rep)          # idempotent
                    with self.assertRaises(activate.ActivationError):
                        activate.enable("fast", "single")
                    self.assertIn("already activated", say.call_args[0][0])
                    self.assertEqual(activate.status()["mode"], "off")


    def test_modules_exit_is_fail_closed(self):
        """design_run / stock_design exit 3 by name when a lever module loaded in the process was not the arm's file — whatever the
        writer's rc says (activate.modules_exit over completion()'s modules_state)."""
        import importlib.machinery
        import io
        with mock.patch.object(stack, "gpu_info", lambda: GPU_STUB), mock.patch.object(report, "say"):
            rep = activate.enable("fast", "single")
        plan = overlay.plan()
        m = types.ModuleType("caliby.api")
        m.__file__ = self.sp.path_of("api.py")
        m.__spec__ = importlib.machinery.ModuleSpec("caliby.api", importlib.machinery.SourceFileLoader("caliby.api", m.__file__), origin=m.__file__)
        sys.modules["caliby.api"] = m                                                             # the INSTALLED file's code under an exact activation
        try:
            done = activate.completion(rep)
            self.assertEqual((done["modules_state"], done["modules_expected"], done["modules_wrong"]), ("mixed", "exact", ["caliby.api"]))
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                self.assertEqual(activate.modules_exit(0, done, 3), 3)                             # rc 0 from the writer does not stand
                self.assertEqual(activate.modules_exit(1, done, 3), 3)                             # nor does a writer failure hide it
            self.assertIn("[caliby-opt] NOT ACTIVE: the lever modules loaded in this process were not the exact arm's files (tree=mixed: caliby.api)", buf.getvalue())
            m.__spec__ = importlib.machinery.ModuleSpec("caliby.api", overlay._Loader("caliby.api", plan["caliby.api"]["path"]), origin=m.__file__)
            done = activate.completion(rep)
            self.assertEqual((done["modules_state"], done["modules_wrong"]), ("exact", []))
            self.assertEqual(activate.modules_exit(0, done, 3), 0)                                 # the arm's files: the writer's rc stands
        finally:
            del sys.modules["caliby.api"]

    def test_enable_exact_installs_the_overlay_exports_row_and_tally(self):
        with mock.patch.object(stack, "gpu_info", lambda: GPU_STUB):
            with mock.patch.object(report, "say") as say, mock.patch.object(report, "register_exit_tally", wraps=report.register_exit_tally) as reg:
                rep = activate.enable("fast", "single")
        report.exit_owned()                                              # this test process decides its own exit code: the environment route's exit gate enable() registered stands down
        self.assertTrue(rep["active"])
        self.assertIs(sys.meta_path[0], overlay.installed())                                      # the hook is in, first
        self.assertEqual(importlib.util.find_spec("caliby.api").loader.path, kit_sources()[1]["api.py"])
        self.assertEqual(stack.tree_state(), "stock")                                             # and the installed tree untouched
        self.assertEqual(rep["tree_digest"], stack.pins()["tree_digest_upstream"]["sha256"])
        self.assertEqual(os.environ["CALIBY_X_LCP"], "1")
        self.assertEqual(os.environ["CALIBY_X_BG_CIF"], "fork")
        reg.assert_called_once_with("fast", "exact")                   # the exit tally is registered by enable, for the mode and the lever-module (tree) state
        line = say.call_args[0][0]
        self.assertTrue(line.startswith("[caliby-opt] ACTIVE mode=fast variant=single tier=2 row=X source=modes.py:single/serial tree=exact gpu=stub sm=9.0 stack=torch2.6.0-cu124-sm90 switches="), line)
        self.assertIn("switches=CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2", line)
        self.assertNotIn("target_gpu=", line)                          # MODEL_OPT_TARGET_GPU unset: nothing to compare
        tally = report.exit_tally_line(1)
        self.assertTrue(tally.startswith("[caliby-opt] EXIT pid=1 mode=fast tree=exact touched_modules_loaded=0 "), tally)
        self.assertIn("CALIBY_X_LCP=1", tally)
        # the report's lever fields (PORTING §8) and their completion at exit from the tally
        self.assertEqual(rep["levers_applied"], ["CALIBY_FAST_SAMPLER", "CALIBY_FAST_POTTS_PARAMS", "CALIBY_X_SPARSE_EXACT", "CALIBY_X_LCP", "CALIBY_X_BG_CIF", "CALIBY_X_MULTISEQ", "CALIBY_X_CIF_WORKERS"])
        self.assertEqual((rep["levers_fallback"], rep["levers_unavailable"], rep["partial"]), ([], [], False))
        self.assertNotIn("row_gate", rep)
        done = activate.completion(rep)
        self.assertEqual((done["levers_fallback"], done["fallback_reasons"], done["partial"]), ([], {}, False))
        self.assertEqual((done["exit_tally"]["mode"], done["exit_tally"]["tree_state"]), ("fast", "exact"))
        with mock.patch.object(report, "tally", lambda: {"mode": "fast", "lcp_kernel": "disabled('no triton')", "fallback_lines": {"CALIBY_X_CLEAN": 2}, "switches": {}}):
            done = activate.completion(rep)
        self.assertEqual((done["levers_fallback"], done["partial"]), (["CALIBY_X_LCP"], True))    # CALIBY_X_CLEAN=0 was never applied: its lines do not count
        self.assertEqual(done["fallback_reasons"], {"CALIBY_X_LCP": "disabled('no triton')"})
        with mock.patch.object(report, "tally", lambda: {"mode": "fast", "lcp_kernel": "available", "fallback_lines": {"CALIBY_X_CLEAN": 1, "CALIBY_X_SPARSE_EXACT": 3}, "switches": {}}):
            done = activate.completion(dict(rep, switches=modes.resolve("fast", "single", 4).env))
        self.assertEqual(done["levers_fallback"], ["CALIBY_X_SPARSE_EXACT", "CALIBY_X_CLEAN"])       # the row's order
        self.assertEqual(done["fallback_reasons"]["CALIBY_X_CLEAN"], "1 kit fallback line(s): [CALIBY_X_CLEAN=")
        self.assertTrue(done["partial"])

    def test_fallback_markers_are_the_kits_lines(self):
        """FALLBACK_MARKERS (the registry's stderr probes) are pinned to the kit sources of the exact tree (plus `caliby_opt.multiseq` and
        `caliby_opt.fast_sampler`, which print X010's and the fast sampler's lines for `potts.py`): every stderr line a kit file prints that
        reports a lever falling back (`-> serial fallback`, `falling back`, `-> kit path`, `-> stock sampler`) is covered by its lever's `stderr`
        probe — the prefix contained in the line — and every marker matches at least one kit line; no marker carries the kit's own tag, so a
        `[caliby-opt]` line (LEVER, EXIT, …) is never counted as a fallback. CALIBY_X_LCP prints no such line: its kernel failing to build or
        launch raises (`XLcpKernelError`) and the mode refuses by name; its record is the `module_attr` probe."""
        import re
        from .. import registry
        self.assertEqual(registry.LEVERS["CALIBY_X_LCP"].probe, ("module_attr", "chroma.layers.complexity", "_X_LCP_DISABLED_REASON"))
        kit_files = []
        d = os.path.join(stack.kit_dir(stack.KIT_ADDON), "fast")
        kit_files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
        self.assertTrue(kit_files)
        kit_files.append(os.path.join(os.path.dirname(report.__file__), "multiseq.py"))     # X010's serial-fallback line is printed for potts.py by caliby_opt.multiseq
        kit_files.append(os.path.join(os.path.dirname(report.__file__), "fast_sampler.py")) # the fast sampler's stock-sampler line is printed for potts.py by caliby_opt.fast_sampler
        rx = re.compile(r"""print\((f?)(["'])(\[CALIBY_[A-Z_]+[=\]][^"']*)\2.*file=_?sys\.stderr""")
        kit_lines = {}
        for path in kit_files:
            with open(path, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    m = rx.search(line)
                    if m and re.search(r"-> serial fallback|falling back|-> kit path|-> stock sampler", m.group(3)):
                        kit_lines[f"{os.path.relpath(path, stack.tree_home())}:{n}"] = m.group(3)
        self.assertEqual(len(kit_lines), 5, kit_lines)                                       # api.py x2, potts.py, caliby_opt/multiseq.py, caliby_opt/fast_sampler.py
        for where, text in kit_lines.items():
            lever = re.match(r"\[(CALIBY_[A-Z_]+)", text).group(1)
            probe = registry.LEVERS[lever].probe
            self.assertIsNotNone(probe, f"{where}: {text!r} names {lever}, which has no probe")
            self.assertEqual(probe[0], "stderr", where)                                          # every printed fallback line is a stderr-probed lever's
            self.assertIn(probe[1], text, where)
            self.assertEqual(report.FALLBACK_MARKERS[lever], probe[1])
        self.assertEqual(registry.LEVERS["CALIBY_X_LCP"].probe, ("module_attr", "chroma.layers.complexity", "_X_LCP_DISABLED_REASON"))   # the import record the EXIT line reads; no printed line
        for lever, marker in report.FALLBACK_MARKERS.items():
            self.assertTrue(any(marker in t and t.startswith(f"[{lever}") for t in kit_lines.values()), f"{lever}: {marker!r} matches no kit line")
        self.assertEqual(sorted(report.FALLBACK_MARKERS), ["CALIBY_FAST_SAMPLER", "CALIBY_X_CLEAN", "CALIBY_X_MULTISEQ", "CALIBY_X_SPARSE_EXACT"])
        for marker in report.FALLBACK_MARKERS.values():
            self.assertNotIn(report.PREFIX, marker)

    def test_counting_stderr_counts_per_lever(self):
        import io
        report._TALLY["fallback_lines"] = {}
        buf = io.StringIO()
        w = report._CountingStderr(buf)
        w.write("[CALIBY_X_CLEAN=loader] failed (RuntimeError('x')) -> serial fallback\n")
        w.write("[CALIBY_X_SPARSE_EXACT] edge_idx rows not distinct/in-range -> kit path for this call\n[CALIBY_X_SPARSE_EXACT] edge_idx rows not distinct/in-range -> kit path for this call\n")
        w.write("some library: falling back to a slower path\n")                              # not a kit line: never counted
        w.write("[caliby-opt] NOT ACTIVE: [CALIBY_X_CLEAN= quoted in the package's own line\n")
        self.assertEqual(report._TALLY["fallback_lines"], {"CALIBY_X_CLEAN": 1, "CALIBY_X_SPARSE_EXACT": 2})
        self.assertEqual(report.fallback_lines_part(), "CALIBY_X_CLEAN:1 CALIBY_X_SPARSE_EXACT:2")
        self.assertEqual(buf.getvalue().count("\n"), 5)
        report._TALLY["fallback_lines"] = {}


    def test_enable_exact_passes_the_row_flags(self):
        with mock.patch.object(stack, "gpu_info", lambda: GPU_STUB):
            with mock.patch.object(report, "say") as say:
                rep = activate.enable("fast", "single", clean_workers=8)
        self.assertEqual((os.environ["CALIBY_X_CLEAN"], os.environ["CALIBY_X_LCP"]), ("loader", "1"))
        self.assertEqual((rep["row_label"], rep["row_source"]), ("XL", "modes.py:single/loader"))
        self.assertIn("row=XL source=modes.py:single/loader", say.call_args[0][0])

    def test_digest_mismatch_is_not_stock_in_both_modes(self):
        for mode, gpu in (("off", NO_GPU), ("fast", GPU_STUB)):
            activate._REPORT = None
            with mock.patch.object(stack, "gpu_info", lambda: gpu), mock.patch.object(tree, "digest", lambda: ("0" * 64, {})), mock.patch.object(report, "say") as say:
                with self.assertRaises(activate.ActivationError) as cm:
                    activate.enable(mode, "single")
            self.assertIn("NOT STOCK: tree digest 0000000000000000 != pinned", str(cm.exception), mode)
            self.assertIn("NOT ACTIVE: NOT STOCK", say.call_args[0][0], mode)
            self.assertIsNone(overlay.installed())                                                # refused before the hook

    def test_stack_key_is_the_shared_rule_and_the_pinned_key(self):
        """stack.stack_key is opt_core.jit_cache.key over torch's values; on the pinned stack it is the string configs/h100.env keys the
        Triton cache by (torch2.6.0-cu124-sm90), and without torch it is the unknown:<reason> refusal form, never a bare unknown."""
        from opt_core import jit_cache
        self.assertEqual(stack.stack_key(GPU_STUB), "torch2.6.0-cu124-sm90")
        self.assertEqual(stack.stack_key(GPU_STUB), jit_cache.key(version="2.6.0+cu124", cuda="12.4", cc="9.0"))
        self.assertTrue(stack.stack_key({"torch": None, "reason": "torch not importable: No module named 'torch'"}).startswith("unknown:torch_not_importable__No_module_named"))
        self.assertTrue(stack.stack_key({"torch": None}).startswith("unknown:"))


if __name__ == "__main__":
    unittest.main()
