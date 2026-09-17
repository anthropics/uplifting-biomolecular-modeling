
"""The design loop's construction and the inputs it is built from. The recipe lives ONCE, in the kit's tools/recipe.py (the
driver holds no recipe literal of its own, checked against the driver's source by ast; the package accessor imports the same file; the
stage words / step kind / loss literals are the notebook's; recipe.design chains the notebook's stages once, next-stage-from-previous-BEST).
A target -- the public default, a FASTA record, or one with a staged MSA -- resolves by name and is refused by name (no record, not
protein, an MSA without a file, an MSA on the public target), and its shape composes the driver's token/flag budget the same way whether
read through recipe.target() or the lower-level inputs.shape_from_args(); the chains_yaml featurizer input names an `msa:` line for every
chain so boltz's server route cannot run. The registry/settings/det facts the recipe and its inputs are built on -- which levers are wired,
det's environment levels, the driver's argparse defaults, the public target's own hash, the vendored ProteinMPNN weight sizes -- are read
from the kit's files directly, not restated as a second copy."""
import ast
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

from . import _stubs
from .. import recipe as accessor
from mosaic_opt import det, inputs, registry, settings


KIT = _stubs.KIT


DRIVER = os.path.join(KIT, "tools", "public_design_run.py")


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestRecipeModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = accessor.module(KIT)

    def test_the_accessor_imports_the_kits_file(self):
        self.assertEqual(os.path.realpath(self.M.__file__), os.path.realpath(os.path.join(KIT, "tools", "recipe.py")))
        self.assertIs(accessor.module(KIT), self.M)                                   # cached: one module object per kit home

    def test_stage_words_and_literals(self):
        self.assertEqual(self.M.PHASES, ("stage1", "stage2", "refold")); self.assertEqual([s["phase"] for s in self.M.STAGES], ["stage1", "stage2"])
        self.assertEqual(self.M.KIND_OF_PHASE, {"stage1": "grad", "stage2": "grad", "refold": "forward"}); self.assertEqual(self.M.REFOLD, {"phase": "refold", "kind": "forward", "key": 0, "call": "model_output"})
        self.assertEqual(self.M.STAGE_CALL, "simplex_APGM"); self.assertEqual(self.M.STEP_KIND, "grad")
        s1, s2 = self.M.STAGES
        self.assertEqual((s1["n_steps"], s1["stepsize"], s1["scale"], s1["momentum"], s1["key_index"]), (75, 0.1, 1.0, 0.0, 1))
        self.assertEqual((s2["n_steps"], s2["stepsize"], s2["scale"], s2["momentum"], s2["key_index"], s2["x"]), (50, 0.5, 1.5, 0.0, 2, "best1"))
        self.assertEqual(self.M.LOSS_TERMS, (("BinderTargetContact", 2), ("WithinBinderContact", None), ("InverseFoldingSequenceRecovery", 5.0)))
        self.assertIs(type(self.M.LOSS_TERMS[0][1]), int); self.assertIs(type(self.M.LOSS_TERMS[2][1]), float)   # the notebook's literals as written (2 int, 5.0 float)
        self.assertEqual(self.M.BOLTZ2_LOSS, {"recycling_steps": 1, "sampling_steps": 25, "deterministic": True})
        self.assertEqual((self.M.X0_GUMBEL_SCALE, self.M.MPNN_TEMP, self.M.REFOLD_KEY), (0.5, 0.01, 0))

    def test_stages_override_step_counts_only(self):
        st = self.M.stages(3, None)
        self.assertEqual([s["n_steps"] for s in st], [3, 50]); self.assertEqual(st[0]["stepsize"], 0.1)
        self.assertEqual([s["n_steps"] for s in self.M.STAGES], [75, 50])                      # the constants are not mutated

    def test_the_driver_holds_no_recipe_literal(self):
        src = open(DRIVER, encoding="utf-8").read(); tree = ast.parse(src)
        via_recipe = lambda n: isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "recipe"
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and not via_recipe(n)}
        for forbidden in ("simplex_APGM", "BinderTargetContact", "WithinBinderContact", "InverseFoldingSequenceRecovery", "TargetChain", "Boltz2", "binder_features", "build_loss", "gumbel", "BARSTAR_1BRS_D"):
            self.assertNotIn(forbidden, names, f"the driver names {forbidden}: the recipe's calls live in tools/recipe.py")
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "recipe"}
        for needed in ("target", "stages", "import_stack", "kit_modules", "install_levers", "levers_record", "levers_finalize", "load_model", "load_mpnn", "featurize", "load_frozen_features", "build_loss", "x0", "design", "refold", "peak_memory", "peak_line"):
            self.assertIn(needed, called, f"the driver does not call recipe.{needed}")
        self.assertLess(src.index("recipe.import_stack()"), src.index("numstate.snapshot()"), "the stack is imported BEFORE the numeric-state snapshot (its observation window opens on the loaded stack)")

    def test_design_is_the_one_chaining(self):
        """recipe.design chains the notebook's stages once: run_stage at stage_key per stage, the next stage from the previous stage's BEST —
        the driver (and every whole-trajectory reader) calls it instead of chaining run_stage itself."""
        src = open(self.M.__file__, encoding="utf-8").read(); tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "design")
        body = ast.get_source_segment(src, fn)
        self.assertIn("run_stage(loss, x_in, stage, stage_key(seed, stage), trajectory_fn=trajectory_fn)", body)
        self.assertIn("x_in = best", body); self.assertIn("stage_list=STAGES", body)
        drv = open(DRIVER, encoding="utf-8").read()
        self.assertNotIn("recipe.run_stage(", drv); self.assertIn("recipe.design(loss, x0, seed, STAGES", drv)
        calls = []
        with mock.patch.object(self.M, "run_stage", side_effect=lambda loss, x, stage, key, trajectory_fn=None, n_steps=None: (calls.append((x, stage["phase"], key)) or (x + "_x", x + "_best", [{"t": stage["phase"]}]))), \
             mock.patch.object(self.M, "stage_key", side_effect=lambda seed, stage: (seed, stage["key_index"])):
            out = self.M.design("LOSS", "x0", 7)
        self.assertEqual([c[0] for c in calls], ["x0", "x0_best"]); self.assertEqual([c[2] for c in calls], [(7, 1), (7, 2)])
        self.assertEqual([(r["x"], r["best"], r["stage"]["phase"]) for r in out], [("x0_x", "x0_best", "stage1"), ("x0_best_x", "x0_best_best", "stage2")])

    def test_refold_key(self):
        """refold(model, x, features, key=None): None = REFOLD_KEY (the design route); an explicit key reaches model_output — a reader refolding
        at other keys never edits REFOLD_KEY."""
        import types
        seen = []
        fake_jax = types.SimpleNamespace(random=types.SimpleNamespace(key=lambda k: ("key", k)), numpy=types.SimpleNamespace(asarray=lambda a: a))
        model = types.SimpleNamespace(model_output=lambda PSSM, features, key: seen.append(key) or "out")
        with mock.patch.dict(sys.modules, {"jax": fake_jax, "jax.numpy": fake_jax.numpy}):
            self.M.refold(model, "x", {"f": 1}); self.M.refold(model, "x", {"f": 1}, key=5)
        self.assertEqual(seen, [("key", self.M.REFOLD_KEY), ("key", 5)]); self.assertEqual(self.M.REFOLD_KEY, 0)

    def test_importing_the_recipe_loads_no_stack(self):
        import sys
        for heavy in ("jax", "torch", "mosaic", "equinox"):
            if heavy in sys.modules and not getattr(sys.modules[heavy], "__stub__", False):
                continue                                                                  # already imported by another test's stack; nothing to assert
            self.assertNotIn(heavy, sys.modules, f"tools/recipe.py imported {heavy} at module import")


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestKitModules(unittest.TestCase):
    """recipe.kit_modules: the helpers load from the installed package when it carries them, else from tools/; the home is named; whichever
    home supplies the pair is trusted as-is (no live byte check, even when its content differs from tools/); the two helpers must load from
    the SAME home (kit_modules_split otherwise)."""

    def _package(self, same):
        import shutil, sys
        root = tempfile.mkdtemp(); pkg = os.path.join(root, "standin_fast"); os.makedirs(pkg); open(os.path.join(pkg, "__init__.py"), "w").close()
        for n in ("fastload", "numstate"):
            src = os.path.join(KIT, "tools", n + ".py")
            if same: shutil.copyfile(src, os.path.join(pkg, n + ".py"))
            else: open(os.path.join(pkg, n + ".py"), "w").write("# a second version\n")
        sys.path.insert(0, root); self.addCleanup(sys.path.remove, root); self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k.startswith("standin_fast")])
        return "standin_fast"

    def test_tools_home_when_package_absent(self):
        M = accessor.module(KIT)
        try: fl, ns, rec = M.kit_modules("no_such_package_anywhere.fast")
        except ImportError as e: self.skipTest(f"the helpers import the stack: {e}")
        from mosaic_opt.registry import KIT_FAST_DIR
        self.assertEqual(rec["source"], "kit_src")                                                  # the recipe's word for the kit's own lever files (tools/recipe.py kit_module)
        self.assertTrue(rec["files"]["numstate"].startswith(os.path.join(os.path.dirname(os.path.abspath(registry.__file__)), KIT_FAST_DIR)), rec["files"])   # served from the kit's lever-file directory (mosaic_opt/<KIT_FAST_DIR>), not a package copy

    def test_package_copy_is_trusted_without_a_byte_check(self):
        """A stand-in package supplies the pair with content that differs from tools/ (`_package(False)`): no live byte check runs any
        more (the git commit names the bytes, not a runtime hash) — kit_modules() trusts whichever home it resolved to and returns it,
        unrefused."""
        M = accessor.module(KIT)
        fl, ns, rec = M.kit_modules(self._package(False))
        self.assertEqual(rec["source"], "standin_fast")
        self.assertTrue(rec["files"]["fastload"].endswith(os.path.join("standin_fast", "fastload.py")))
        self.assertTrue(rec["files"]["numstate"].endswith(os.path.join("standin_fast", "numstate.py")))
        self.assertIsNotNone(fl); self.assertIsNotNone(ns)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestTarget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = accessor.module(KIT)
        cls.tmp = tempfile.mkdtemp()
        cls.fasta = os.path.join(cls.tmp, "t.fasta"); open(cls.fasta, "w").write(">pdl1_N200 some words\nMKTAYIAKQR\nQISFVKSHFS\n")          # ONE record: the target
        cls.multi = os.path.join(cls.tmp, "m.fasta"); open(cls.multi, "w").write(">pdl1_N200 some words\nMKTAYIAKQR\nQISFVKSHFS\n>second\nAAAA\n>third\nGGG\n")   # three records
        cls.a3m = os.path.join(cls.tmp, "t.a3m"); open(cls.a3m, "w").write(">query\nMKTAYIAKQRQISFVKSHFS\n>hit1\nMKTAYIAKQRQISFVK-HFS\n")

    def test_public_target_by_default(self):
        t = self.M.target()
        self.assertEqual((t["name"], t["id"], t["length"], t["copies"], t["use_msa"], t["source"]), ("barstar", "barstar", 89, 1, False, "public"))
        self.assertEqual(self.M.tokens(t, 80), 169); self.assertEqual(self.M.tokens(self.M.target(copies=2), 80), 258)

    def test_fasta_one_record_and_id(self):
        t = self.M.target(fasta=self.fasta, copies=1)
        self.assertEqual((t["name"], t["sequence"], t["length"], t["use_msa"], t["msa"]), ("pdl1_N200", "MKTAYIAKQRQISFVKSHFS", 20, False, None))
        self.assertEqual((t["fasta_records"], t["fasta_record_used"], t["first_record"]), (1, 1, False))
        self.assertRegex(t["id"], r"^pdl1_N200-[0-9a-f]{8}$"); self.assertEqual(self.M.tokens(t, 180), 200)
        self.assertNotEqual(self.M.target_id("x", "AAAA"), self.M.target_id("x", "AAAG"))    # same record id, different sequence: different id
        self.assertEqual(self.M.input_lines(t), [])                                            # nothing narrowed: no INPUT line
        pub = self.M.target()
        self.assertEqual((pub["fasta_records"], pub["fasta_record_used"], pub["first_record"]), (None, None, False))

    def test_multi_record_fasta_refused_unless_first_record(self):
        """A FASTA with more than one record is refused by name (`fasta_multi_record`, naming the count and both remedies) — the target is ONE chain;
        `first_record=True` (the driver's --first-record) takes record 1 and the record SAYS so: fasta_records=<n>, fasta_record_used=1,
        first_record=True, and the driver's INPUT line names the count. --first-record without a FASTA is refused by name too."""
        R = self.M.RecipeError
        with self.assertRaisesRegex(R, r"fasta_multi_record: .*holds 3 FASTA records.*--first-record") as cm:
            self.M.target(fasta=self.multi)
        self.assertIn("INPUT target_fasta records=3 used=1", str(cm.exception))
        t = self.M.target(fasta=self.multi, first_record=True, copies=2)
        self.assertEqual((t["name"], t["sequence"], t["fasta_records"], t["fasta_record_used"], t["first_record"], t["copies"]), ("pdl1_N200", "MKTAYIAKQRQISFVKSHFS", 3, 1, True, 2))
        self.assertEqual(self.M.input_lines(t), ["INPUT target_fasta records=3 used=1 (--first-record)"])
        one = self.M.target(fasta=self.fasta, first_record=True)                              # harmless on a one-record file, and still said
        self.assertEqual(self.M.input_lines(one), ["INPUT target_fasta records=1 used=1 (--first-record)"])
        with self.assertRaisesRegex(R, "first_record_without_fasta"): self.M.target(first_record=True)
        with self.assertRaisesRegex(R, "fasta_multi_record"): accessor.recipe(KIT, binder_length=80, target_fasta=self.multi)
        with self.assertRaisesRegex(ValueError, "fasta_multi_record"): inputs.shape_from_args(80, 1, settings.driver_defaults(KIT), target_fasta=self.multi, kit_home=KIT)   # the CLI's parser is the same module (its refusal = a usage ValueError, same words)
        indented = os.path.join(self.tmp, "indented.fasta"); open(indented, "w").write(">first\nMKTAYIAKQR\n   >second header with leading blanks\nGGGG\n")
        self.assertEqual(self.M.fasta_records(indented), 2)                                  # ONE header predicate: the census counts what the reader stops at
        with self.assertRaisesRegex(R, r"fasta_multi_record: .*holds 2 FASTA records"): self.M.target(fasta=indented)
        ti = self.M.target(fasta=indented, first_record=True)
        self.assertEqual((ti["sequence"], ti["fasta_records"], self.M.input_lines(ti)), ("MKTAYIAKQR", 2, ["INPUT target_fasta records=2 used=1 (--first-record)"]))
        self.assertTrue(self.M.is_fasta_header("  >x")); self.assertTrue(self.M.is_fasta_header(">x")); self.assertFalse(self.M.is_fasta_header("MK>TA"))

    def test_real_driver_refuses_or_names_its_inputs_before_the_stack(self):
        """The kit driver itself (tools/public_design_run.py, run as a process on this CPU box: it resolves its inputs before importing jax): a
        multi-record --target-fasta ends `[run] <tag> inputs_refused: fasta_multi_record: …` rc 1 with nothing designed; with --first-record and
        --epitope it prints both INPUT lines FIRST and records the facts in results.json's manifest (the design then stops at the missing GPU stack
        here — status=error — which is this box, not the inputs)."""
        import subprocess
        out = os.path.join(self.tmp, "drv")
        r = subprocess.run([sys.executable, DRIVER, "--tag", "r", "--out", out, "--target-fasta", self.multi], capture_output=True, text=True, cwd=os.path.dirname(DRIVER), timeout=120)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("[run] r inputs_refused: fasta_multi_record: ", r.stdout); self.assertIn("--first-record", r.stdout); self.assertIn("[run] r done in", r.stdout)
        self.assertEqual(json.load(open(os.path.join(out, "r", "results.json")))["status"], "error")
        r2 = subprocess.run([sys.executable, DRIVER, "--tag", "r2", "--out", out, "--target-fasta", self.multi, "--first-record", "--target-copies", "2", "--epitope", "5, 2,4-5"],
                            capture_output=True, text=True, cwd=os.path.dirname(DRIVER), timeout=120)
        lines = r2.stdout.splitlines()
        self.assertEqual(lines[:2], ["INPUT target_fasta records=3 used=1 (--first-record)",
                                     "INPUT epitope residues=2,4-5 positions=3 target_length=20 copies=2 epitope_idx=6 (--epitope: 1-based target residue positions, applied on every target copy)"], r2.stdout + r2.stderr)
        man = json.load(open(os.path.join(out, "r2", "results.json")))["manifest"]
        self.assertEqual((man["target"]["fasta_records"], man["target"]["fasta_record_used"], man["target"]["first_record"], man["target"]["copies"]), (3, 1, True, 2))
        self.assertEqual((man["epitope"]["spec"], man["epitope"]["residues"], man["epitope"]["idx"], man["args"]["epitope"], man["args"]["first_record"]), ("2,4-5", [2, 4, 5], [1, 3, 4, 21, 23, 24], "5, 2,4-5", True))
        r3 = subprocess.run([sys.executable, DRIVER, "--tag", "r3", "--out", out, "--epitope", "0-3"], capture_output=True, text=True, cwd=os.path.dirname(DRIVER), timeout=120)
        self.assertEqual(r3.returncode, 1); self.assertIn("[run] r3 inputs_refused: epitope_format: ", r3.stdout)

    def test_epitope_grammar_and_index_space(self):
        """--epitope: comma-separated 1-based positions / inclusive ranges along the target sequence → sorted unique residues, a canonical spelling,
        and upstream's `BinderTargetContact(epitope_idx=…)` list: 0-based over the target tokens after the binder, the same residues on every copy
        ((p-1) + c·length). Misspellings and out-of-range positions are refused by name."""
        M, R = self.M, self.M.RecipeError
        self.assertEqual(M.parse_epitope("12,15,40-42"), [12, 15, 40, 41, 42]); self.assertEqual(M.parse_epitope(" 42-40, 3 ".replace("42-40", "40-42")), [3, 40, 41, 42])
        self.assertEqual(M.parse_epitope("5,5,4-6,1"), [1, 4, 5, 6]); self.assertEqual(M.epitope_spec([1, 4, 5, 6]), "1,4-6"); self.assertEqual(M.epitope_spec(M.parse_epitope("7,3,4,5,9-10")), "3-5,7,9-10")
        for bad in ("", " ", "0", "3-1", "a", "1,,2", "1-", "-3", "1.5", "2;3"):
            with self.assertRaisesRegex(R, "epitope_format", msg=bad): M.parse_epitope(bad)
        t = M.target(fasta=self.fasta, copies=1)                                                # 20 residues
        e = M.epitope("2,4-5", t)
        self.assertEqual((e["spec"], e["residues"], e["idx"], e["target_length"], e["copies"], e["n_idx"]), ("2,4-5", [2, 4, 5], [1, 3, 4], 20, 1, 3))
        e2 = M.epitope("20,4-5,2", M.target(fasta=self.fasta, copies=2))                       # two copies: the same residues on both chains
        self.assertEqual((e2["spec"], e2["idx"]), ("2,4-5,20", [1, 3, 4, 19, 21, 23, 24, 39]))
        with self.assertRaisesRegex(R, r"epitope_out_of_range: .*21.* 1\.\.20"): M.epitope("5,21", t)
        with self.assertRaisesRegex(R, r"epitope_out_of_range"): M.epitope("90", M.target())     # the public target has 89 residues
        self.assertEqual(M.epitope("89,1", M.target())["idx"], [0, 88])
        self.assertEqual(M.input_lines(t, e), ["INPUT epitope residues=2,4-5 positions=3 target_length=20 copies=1 epitope_idx=3 (--epitope: 1-based target residue positions, applied on every target copy)"])
        ft = M.target(fasta=self.multi, first_record=True)
        self.assertEqual(len(M.input_lines(ft, M.epitope("1", ft))), 2)                          # both input options named

    def test_loss_expression_hands_epitope_idx_to_upstream(self):
        """The loss is constructed through upstream's own classes: with an epitope, `BinderTargetContact(epitope_idx=[…])` at the notebook's weight
        2; without one, `BinderTargetContact()` BARE — the notebook's construction byte for byte (no epitope_idx=None passed). Stand-in `mosaic` /
        `jax` modules capture the constructor calls; no model runs."""
        import sys, types
        calls = []

        class Term:
            def __init__(self, name, *a, **kw):
                self.name, self.a, self.kw = name, a, kw; calls.append((name, a, kw))
            def __rmul__(self, w):
                return Term("mul", w, self)
            def __add__(self, other):
                return Term("add", self, other)

        sp = types.ModuleType("mosaic.losses.structure_prediction")
        sp.BinderTargetContact = lambda *a, **kw: Term("BinderTargetContact", *a, **kw)
        sp.WithinBinderContact = lambda *a, **kw: Term("WithinBinderContact", *a, **kw)
        pm = types.ModuleType("mosaic.losses.protein_mpnn")
        pm.InverseFoldingSequenceRecovery = lambda mpnn, temp=None: Term("InverseFoldingSequenceRecovery", mpnn, temp=temp)
        mosaic = types.ModuleType("mosaic"); losses = types.ModuleType("mosaic.losses"); mosaic.losses = losses; losses.structure_prediction = sp; losses.protein_mpnn = pm
        jax = types.ModuleType("jax"); jnp = types.ModuleType("jax.numpy"); jnp.array = lambda v: ("array", v); jax.numpy = jnp
        names = {"mosaic": mosaic, "mosaic.losses": losses, "mosaic.losses.structure_prediction": sp, "mosaic.losses.protein_mpnn": pm, "jax": jax, "jax.numpy": jnp}
        saved = {k: sys.modules.get(k) for k in names}
        sys.modules.update(names)
        try:
            def contact_kwargs(expr_calls):
                return [kw for name, a, kw in expr_calls if name == "BinderTargetContact"]
            calls.clear(); bare = self.M.loss_expression("MPNN")
            self.assertEqual(contact_kwargs(calls), [{}])                                        # unset: the notebook's bare term — no epitope_idx keyword at all
            self.assertEqual([n for n, _, _ in calls if n in ("BinderTargetContact", "WithinBinderContact", "InverseFoldingSequenceRecovery")],
                             ["BinderTargetContact", "WithinBinderContact", "InverseFoldingSequenceRecovery"])       # LOSS_TERMS' order
            self.assertEqual((bare.name, bare.a[0].name, bare.a[0].a[0].name, bare.a[0].a[0].a[0]), ("add", "add", "mul", 2))   # 2·contact + within + 5·recovery
            ifr = [c for c in calls if c[0] == "InverseFoldingSequenceRecovery"][0]
            self.assertEqual((ifr[1][0], ifr[2]["temp"]), ("MPNN", ("array", self.M.MPNN_TEMP)))
            calls.clear(); self.M.loss_expression("MPNN", epitope_idx=[1, 3, 4])
            self.assertEqual(contact_kwargs(calls), [{"epitope_idx": [1, 3, 4]}])                   # given: upstream's own field, the same weight and term order
            self.assertEqual([c for c in calls if c[0] == "mul"][0][1][0], 2)
            calls.clear()
            class Model:                                                                         # build_loss threads it through and asserts the Boltz2Loss settings on the result
                def build_loss(self, loss, features):
                    return types.SimpleNamespace(loss=loss, features=features, **self_M.BOLTZ2_LOSS)
            self_M = self.M
            built = self.M.build_loss(Model(), "MPNN", {"f": 1}, epitope_idx=[7])
            self.assertEqual((contact_kwargs(calls), built.features), ([{"epitope_idx": [7]}], {"f": 1}))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_msa_staged(self):
        t = self.M.target(fasta=self.fasta, msa=self.a3m)
        self.assertTrue(t["use_msa"]); self.assertEqual(t["msa"], os.path.abspath(self.a3m)); self.assertEqual(len(t["msa_sha256"]), 64)
        self.assertEqual(self.M.msa_words(t)[:15], "msa: staged a3m"); self.assertEqual(self.M.msa_words(self.M.target()), "msa: empty -> single-sequence")
        self.assertEqual(self.M.MSA_SUFFIXES, (".csv", ".a3m"))

    def test_loss_terms_of_flattens_the_aux(self):
        import numpy as np
        got = self.M.loss_terms_of([{"contact": 1.5, "nested": {"a": np.ones(3)}}, {"seq_recovery": [2.0, np.zeros((2, 2))]}, "words"])
        self.assertEqual(got, {"[0].contact": 1.5, "[0].nested.a": 1.0, "[1].seq_recovery[0]": 2.0, "[1].seq_recovery[1]": 0.0})

    def test_refusals_by_name(self):
        R = self.M.RecipeError
        with self.assertRaisesRegex(R, "msa_not_found"): self.M.target(fasta=self.fasta, msa=os.path.join(self.tmp, "absent.a3m"))
        bad = os.path.join(self.tmp, "t.sto"); open(bad, "w").write("# STOCKHOLM\n")
        with self.assertRaisesRegex(R, "msa_format"): self.M.target(fasta=self.fasta, msa=bad)
        csv = os.path.join(self.tmp, "t.csv"); open(csv, "w").write("key,sequence\n-1,MKTAYIAKQRQISFVKSHFS\n-1,MKTAY-AKQRqqQISFVKSHFT\n")
        t2 = self.M.target(fasta=self.fasta, msa=csv)
        self.assertEqual((t2["msa"], t2["msa_rows"]), (os.path.abspath(csv), 2))                          # boltz's processed csv; its rows counted
        other = os.path.join(self.tmp, "o.csv"); open(other, "w").write("key,sequence\n-1,MKTAYIAKQRQISFVKSHFT\n")
        with self.assertRaisesRegex(R, "msa_query_mismatch"): self.M.target(fasta=self.fasta, msa=other)   # another target's alignment: refused by name
        with self.assertRaisesRegex(R, "msa_chains_outside_yaml"): self.M.chains(t2)                      # an MSA target's chains only through featurize / chains_yaml
        self.assertEqual(self.M.msa_query(csv), ("MKTAYIAKQRQISFVKSHFS", 2))
        self.assertEqual(self.M.msa_words(self.M.target(fasta=self.fasta, msa=csv))[:15], "msa: staged csv")
        with self.assertRaisesRegex(R, "msa_public_target"): self.M.target(msa=self.a3m)
        empty = os.path.join(self.tmp, "e.fasta"); open(empty, "w").write("\n")
        with self.assertRaisesRegex(R, "fasta_no_record"): self.M.target(fasta=empty)
        dna = os.path.join(self.tmp, "d.fasta"); open(dna, "w").write(">x\nMKTA1234\n")
        with self.assertRaisesRegex(R, "fasta_not_protein"): self.M.target(fasta=dna)
        with self.assertRaisesRegex(R, "not both"): self.M.target(fasta=self.fasta, sequence="MKT")

    def test_accessor_record(self):
        rec = accessor.recipe(KIT, binder_length=180, target_fasta=self.fasta, msa=self.a3m)
        self.assertEqual(rec["tokens"], 200); self.assertEqual(rec["phases"], ["stage1", "stage2", "refold"]); self.assertEqual(rec["kind_of_phase"]["refold"], "forward"); self.assertEqual(rec["stage_call"], "simplex_APGM")
        self.assertEqual(rec["flags"], ["--binder-length", "180", "--target-copies", "1", "--target-fasta", os.path.abspath(self.fasta), "--msa", os.path.abspath(self.a3m)])
        self.assertIs(rec["module"], self.M); self.assertIsNone(rec["epitope"])
        self.assertEqual(accessor.recipe(KIT, binder_length=80)["flags"], ["--binder-length", "80", "--target-copies", "1"])   # the public target: no target flag
        both = accessor.recipe(KIT, binder_length=180, target_fasta=self.multi, first_record=True, epitope="5-4,2".replace("5-4", "4-5"))
        self.assertEqual(both["flags"], ["--binder-length", "180", "--target-copies", "1", "--target-fasta", os.path.abspath(self.multi), "--first-record", "--epitope", "2,4-5"])   # the canonical spelling rides
        self.assertEqual((both["epitope"]["idx"], both["target"]["fasta_records"]), ([1, 3, 4], 3))
        self.assertEqual((both["shape"]["first_record"], both["shape"]["epitope"], both["shape"]["fasta_records"]), (True, "2,4-5", 3))   # the Shape the flags come from (inputs.shape_of: the command line's composer too)
        self.assertEqual(both["flags"], inputs.shape_of(both["target"], 180, 1, both["epitope"]).flags())
        self.assertEqual(accessor.recipe(KIT, binder_length=80, epitope="1,89")["flags"], ["--binder-length", "80", "--target-copies", "1", "--epitope", "1,89"])   # the public target takes an epitope too


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestChainsYaml(unittest.TestCase):
    """chains_yaml composes upstream's OWN _prefix / chain_yaml — executed here from the vendored stock/src/mosaic models/boltz2.py source (ast:
    the two functions' literal text, no stand-in) — adds the staged file's `msa:` line in chain_yaml's continuation form, joins upstream's way,
    and the result PARSES (yaml) into one protein entry per chain, each with an msa key."""

    @staticmethod
    def _upstream_funcs():
        import ast as _ast
        src_path = os.path.join(_stubs.TREE, "stock", "src", "mosaic", "src", "mosaic", "models", "boltz2.py")
        src = open(src_path).read(); tree = _ast.parse(src)
        fns = {n.name: _ast.get_source_segment(src, n) for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in ("_prefix", "chain_yaml")}
        ns = {}
        exec(fns["_prefix"], ns); exec(fns["chain_yaml"].replace("chain: TargetChain", "chain"), ns)   # noqa: S102 — upstream's literal function bodies
        return ns["_prefix"], ns["chain_yaml"]

    def test_every_chain_names_an_msa_line_and_the_text_parses(self):
        import sys, types, yaml
        M = accessor.module(KIT)
        _prefix, chain_yaml = self._upstream_funcs()
        self.assertEqual(_prefix(), "version: 1\nsequences:")                                   # upstream's literal: no trailing newline (the join supplies it)
        saved = {k: sys.modules.get(k) for k in ("mosaic", "mosaic.models", "mosaic.models.boltz2", "mosaic.structure_prediction")}
        try:
            sp = types.ModuleType("mosaic.structure_prediction")
            class TargetChain:
                def __init__(self, sequence, polymer_type="PROTEIN", use_msa=True, template_chain=None):
                    self.sequence, self.polymer_type, self.use_msa, self.template_chain = sequence, polymer_type, use_msa, template_chain
            sp.TargetChain = TargetChain
            mb = types.ModuleType("mosaic.models.boltz2"); mb._prefix = _prefix; mb.chain_yaml = chain_yaml
            pkg = types.ModuleType("mosaic"); pkg.__stub__ = True; models = types.ModuleType("mosaic.models")
            sys.modules.update({"mosaic": pkg, "mosaic.models": models, "mosaic.models.boltz2": mb, "mosaic.structure_prediction": sp})
            tmp = tempfile.mkdtemp(); fa = os.path.join(tmp, "t.fasta"); open(fa, "w").write(">t\nMKTAYIAK\n"); csv = os.path.join(tmp, "t.csv"); open(csv, "w").write("key,sequence\n-1,MKTAYIAK\n")
            text, objs = M.chains_yaml(10, M.target(fasta=fa, copies=2, msa=csv))
            doc = yaml.safe_load(text)
            ents = [s["protein"] for s in doc["sequences"]]
            self.assertEqual([e["id"] for e in ents], [["A"], ["B"], ["C"]]); self.assertEqual(ents[0]["sequence"], "X" * 10)
            self.assertEqual([e["msa"] for e in ents], ["empty", os.path.abspath(csv), os.path.abspath(csv)])   # the binder single-sequence, both target copies on the staged file
            self.assertTrue(text.startswith("version: 1\nsequences:\n  - protein:"))
            text1, _ = M.chains_yaml(10, M.target(fasta=fa))
            self.assertEqual([s["protein"]["msa"] for s in yaml.safe_load(text1)["sequences"]], ["empty", "empty"])   # no alignment staged: every chain single-sequence
            t = M.target(fasta=fa); t["use_msa"] = True                                                      # a chain asking for an MSA with no file: refused by name
            with self.assertRaisesRegex(M.RecipeError, "msa_required_no_path"): M.chains_yaml(10, t)
            with self.assertRaisesRegex(M.RecipeError, "yaml_msa_lines"): M.check_chains_yaml("version: 1\nsequences:  - protein:\n    id: [A]", 1)   # a text that does not parse: refused by name
            with self.assertRaisesRegex(M.RecipeError, "yaml_msa_lines"): M.check_chains_yaml("version: 1\nsequences:\n  - protein:\n        id: [A]\n        sequence: XX", 1)   # an entry without an msa key (boltz would fetch): refused
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestShapeWithTarget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.fasta = os.path.join(cls.tmp, "cd45_N400.fasta"); open(cls.fasta, "w").write(">cd45_N400\n" + "MKTAYIAKQR" * 32 + "\n")
        cls.a3m = os.path.join(cls.tmp, "cd45_N400.a3m"); open(cls.a3m, "w").write(">q\n" + "MKTAYIAKQR" * 32 + "\n")
        cls.D = settings.driver_defaults(_stubs.KIT)

    def test_public_shape_unchanged(self):
        s = inputs.shape_from_args(None, None, self.D)
        self.assertEqual((s.key, s.flags(), s.tokens(89), s.target), ("L80_c1", ["--binder-length", "80", "--target-copies", "1"], 169, None))

    def test_fasta_shape(self):
        s = inputs.shape_from_args(80, 1, self.D, target_fasta=self.fasta, kit_home=_stubs.KIT)
        self.assertRegex(s.key, r"^cd45_N400-[0-9a-f]{8}_L80_c1$"); self.assertEqual(s.target_length, 320); self.assertEqual(s.tokens(89), 400)
        self.assertEqual(s.flags(), ["--binder-length", "80", "--target-copies", "1", "--target-fasta", os.path.abspath(self.fasta)])
        m = inputs.shape_from_args(80, 1, self.D, target_fasta=self.fasta, msa=self.a3m, kit_home=_stubs.KIT)
        self.assertRegex(m.key, r"^cd45_N400-[0-9a-f]{8}-msa[0-9a-f]{8}_L80_c1$"); self.assertNotEqual(m.key, s.key)      # the staged MSA is part of the key: different features, different executable
        self.assertEqual(m.flags()[-2:], ["--msa", os.path.abspath(self.a3m)])
        self.assertEqual(m.record()["target"], s.target); self.assertEqual(m.record()["msa"], os.path.abspath(self.a3m))

    def test_msa_without_fasta_refused(self):
        with self.assertRaisesRegex(ValueError, "msa_public_target"):                        # the recipe module's refusal by name (the one parser) as the command line's usage ValueError
            inputs.shape_from_args(80, 1, self.D, msa=self.a3m, kit_home=_stubs.KIT)
        with self.assertRaisesRegex(ValueError, "first_record_without_fasta"):
            inputs.shape_from_args(80, 1, self.D, first_record=True, kit_home=_stubs.KIT)

    def test_first_record_and_epitope_ride_the_shape(self):
        """--first-record and --epitope are input options the Shape carries to every driver launch (flags(), record()); the epitope keys the shape
        (`ep<sha8>` of its canonical spelling: another loss, another executable) on the public target and on a FASTA target alike; a position
        outside the target or a misspelled value is refused by name before anything runs."""
        multi = os.path.join(self.tmp, "two.fasta"); open(multi, "w").write(">cd45_N400\n" + "MKTAYIAKQR" * 32 + "\n>other\nAAAA\n")
        with self.assertRaisesRegex(ValueError, "fasta_multi_record"):
            inputs.shape_from_args(80, 1, self.D, target_fasta=multi, kit_home=_stubs.KIT)
        f = inputs.shape_from_args(80, 1, self.D, target_fasta=multi, first_record=True, kit_home=_stubs.KIT)
        one = inputs.shape_from_args(80, 1, self.D, target_fasta=self.fasta, kit_home=_stubs.KIT)
        self.assertEqual((f.key, f.first_record, f.fasta_records, f.record()["first_record"], f.record()["fasta_records"]), (one.key, True, 2, True, 2))   # same record 1: same target id and key
        self.assertEqual(f.flags(), ["--binder-length", "80", "--target-copies", "1", "--target-fasta", os.path.abspath(multi), "--first-record"])
        e = inputs.shape_from_args(80, 2, self.D, target_fasta=self.fasta, epitope="12, 3-5,4", kit_home=_stubs.KIT)
        self.assertEqual((e.epitope, e.record()["epitope"], e.flags()[-2:]), ("3-5,12", "3-5,12", ["--epitope", "3-5,12"]))
        self.assertRegex(e.key, r"^cd45_N400-[0-9a-f]{8}_ep" + hashlib.sha256(b"3-5,12").hexdigest()[:8] + r"_L80_c2$")
        self.assertNotEqual(e.key, inputs.shape_from_args(80, 2, self.D, target_fasta=self.fasta, kit_home=_stubs.KIT).key)
        self.assertEqual(inputs.shape_from_args(80, 2, self.D, target_fasta=self.fasta, epitope="4,12,5,3", kit_home=_stubs.KIT).key, e.key)   # any spelling of one residue set: one key
        p = inputs.shape_from_args(None, None, self.D, epitope="89,1-2", kit_home=_stubs.KIT)   # the public target (89 residues)
        self.assertEqual((p.key, p.target, p.flags()), ("ep" + hashlib.sha256(b"1-2,89").hexdigest()[:8] + "_L80_c1", None, ["--binder-length", "80", "--target-copies", "1", "--epitope", "1-2,89"]))
        with self.assertRaisesRegex(ValueError, "epitope_out_of_range"):
            inputs.shape_from_args(None, None, self.D, epitope="90", kit_home=_stubs.KIT)
        with self.assertRaisesRegex(ValueError, "epitope_out_of_range"):
            inputs.shape_from_args(80, 1, self.D, target_fasta=self.fasta, epitope="321", kit_home=_stubs.KIT)
        with self.assertRaisesRegex(ValueError, "epitope_format"):
            inputs.shape_from_args(80, 1, self.D, target_fasta=self.fasta, epitope="12-", kit_home=_stubs.KIT)

    def test_driver_flags(self):
        for f in ("--target-fasta", "--msa", "--epitope"):
            self.assertIn(f, settings.DRIVER_FLAGS); self.assertIsNone(self.D[f])
        self.assertIn("--first-record", settings.DRIVER_FLAGS); self.assertIn(self.D["--first-record"], (None, False))   # a store_true flag: no default literal


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestRegistrySettingsDet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pins = json.load(open(os.path.join(_stubs.TREE, "stock", "PINS.json"), encoding="utf-8"))

    def test_wired_levers_are_the_three_bitwise_ones(self):
        self.assertEqual(registry.WIRED, ("P1", "P2", "P3", "P5", "E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"))   # the qol levers of the exact row + the tiers' per-step levers
        self.assertEqual(registry.IN_PROCESS, ("P1", "P5", "E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"))                          # what enable() / the probe apply without the caller's cooperation: the environment lever and the per-step levers
        self.assertEqual(registry.CALL_SITE, ("P2", "P3"))

    def test_det_levels_and_refusal(self):
        self.assertEqual(det.apply_env({}, 0), {})
        self.assertEqual(det.apply_env({}, 1), det.ENV)
        self.assertEqual(det.apply_env({"PYTHONUNBUFFERED": "0"}, 1)["PYTHONUNBUFFERED"], "0")       # a set value stands
        self.assertIsNone(det.precision_refusal({}))
        self.assertIn("JAX_ENABLE_X64", det.precision_refusal({"JAX_ENABLE_X64": "1"}))
        with self.assertRaises(ValueError):
            det.level(2)

    def test_driver_defaults_read_from_the_kit_file(self):
        d = settings.driver_defaults(_stubs.KIT)
        self.assertEqual((d["--binder-length"], d["--target-copies"], d["--steps1"], d["--steps2"], d["--weights"]), (80, 1, 75, 50, "torch"))
        eff = settings.effective(_stubs.KIT)
        self.assertEqual(settings.flags(eff), ["--steps1", "75", "--steps2", "50"])
        self.assertEqual(settings.flags(settings.effective(_stubs.KIT, steps1=10)), ["--steps1", "10", "--steps2", "50"])
        self.assertFalse(hasattr(settings, "PRESETS"))                                    # no named presets: the driver's knobs pass through under their own names

    def test_public_target_from_the_kit_file(self):
        t = inputs.public_target(_stubs.KIT)
        self.assertEqual((t["pdb"], t["length"]), ("1BRS", 89))
        self.assertEqual(t["sha256"], hashlib.sha256(t["sequence"].encode()).hexdigest())
        default = inputs.shape_from_args(None, None, settings.driver_defaults(_stubs.KIT))       # the driver's defaults: L80, one copy
        self.assertEqual(default.tokens(t["length"]), 169)
        self.assertEqual(inputs.Shape(80, 2).tokens(t["length"]), 258)

    def test_proteinmpnn_size_equals_the_stock_archive_file(self):
        """The ProteinMPNN weight files: size in stock/PINS.json equal to the archive member's size. The archive's own bytes are
        this repo's commit (test_stock_archives.py checks the tarball byte-for-byte against stock/src); no separate digest of
        these files is kept."""
        pin = self.pins["upstream"]["mosaic"]
        path = os.path.join(_stubs.TREE, pin["archive"])
        want = self.pins["weights"]["proteinmpnn"]["files"]
        with tarfile.open(path, "r:gz") as tf:
            for name, rec in want.items():
                member = f"mosaic-{pin['commit'][:8]}/src/mosaic/proteinmpnn/weights/{name}"
                data = tf.extractfile(member).read()
                self.assertEqual(len(data), rec["size_bytes"], name)


class TestRegistryVocabulary(unittest.TestCase):
    """The class / route vocabulary of the 0.3 kit line: every entry's words are the table's, bitwise <=> exact|qol, the per-step levers are the
    install-route ones (levers.py's domain), the qol levers keep their environment / call-site routes, a wired per-step lever names its module."""

    def test_words(self):
        self.assertEqual(registry.KLASSES, ("exact", "fast", "big", "qol")); self.assertEqual(registry.ROUTES, ("env", "flag", "install"))
        for k, v in registry.LEVERS.items():
            self.assertIn(v.klass, registry.KLASSES, k); self.assertIn(v.route, registry.ROUTES, k); self.assertIn(v.origin, ("kit", "core"), k)
            self.assertEqual(v.tier == "bitwise", v.klass in ("exact", "qol"), k)

    def test_routes_of_the_kit_levers(self):
        self.assertEqual([registry.LEVERS[k].route for k in ("P1", "P2", "P3", "P5")], ["env", "flag", "flag", "install"])
        self.assertEqual([registry.LEVERS[k].klass for k in ("P1", "P2", "P3", "P5")], ["qol", "qol", "qol", "big"])
        self.assertEqual(registry.INSTALL, ("P5", "E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"))
        self.assertEqual(registry.WIRED, tuple(registry.LEVERS))                          # every lever of the registry is composed by a mode (no lever ships outside a row)
        self.assertEqual(registry.LEVERS["P2"].flag, "--weights fastinit"); self.assertEqual(registry.LEVERS["P3"].flag, '--features-in "$F1" --features-sha "$FSHA"')
        self.assertIsNone(registry.LEVERS["P1"].flag)                                        # environment-only
        self.assertEqual(registry.LEVERS_FLAG, "--levers"); self.assertEqual(registry.probe_key("F2"), ("manifest", "levers.F2.state", "equals:on"))
        for k in registry.WIRED:
            if registry.LEVERS[k].route == "install":
                self.assertTrue(registry.LEVERS[k].module, k)


if __name__ == "__main__":
    unittest.main()

