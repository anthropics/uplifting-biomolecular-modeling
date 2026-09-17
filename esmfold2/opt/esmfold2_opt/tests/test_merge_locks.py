"""Single-place locks: each concern below lives in exactly one place. Each test names the one place and fails when a
second copy appears (a mode table, a settings table, a stock caller, a deterministic recipe, an add-on directory, the
command list). Source-level where the concern is a table; filesystem-level where it is a file. No torch, no GPU."""
import os
import re
import unittest

from esmfold2_opt import _autoload, cli, det, modes, report, settings, stack

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/esmfold2_opt
OPT = os.path.dirname(PKG)                                                    # opt/
TREE = os.path.dirname(OPT)                                                   # esmfold2/


def _modules():
    return {f: open(os.path.join(PKG, f), encoding="utf-8").read() for f in sorted(os.listdir(PKG)) if f.endswith(".py")}


def _tree_present():
    return os.path.isfile(os.path.join(TREE, "run.sh")) and os.path.isdir(os.path.join(TREE, "stock"))

TABLE_PENDING: tuple = ()               # modes the hook declares ahead of modes.MODES (none: the table carries every mode the hook declares)

class TestLeverfoldKeysAreRealCounters(unittest.TestCase):
    def test_every_leverfold_key_is_a_counter_its_module_writes(self):
        """report.LEVERFOLD_KEYS names, per lever module, counters that module actually writes (STATS["<key>"] or its COUNTERS words in the
        driver source; ef2_opt's graph_budget_eager_<site> is composed from the site name) — a misspelt key would print a constant 0 per fold."""
        here = os.path.dirname(os.path.abspath(__file__))                       # .../opt/esmfold2_opt/tests
        drv = os.path.join(os.path.dirname(os.path.dirname(here)), "forward", "fast_inference", "driver")
        if not os.path.isdir(drv):
            self.skipTest("kit tree not present")
        for mod, keys in report.LEVERFOLD_KEYS.items():
            src = open(os.path.join(drv, mod + ".py")).read()
            for k in keys:
                if mod == "ef2_opt" and k.startswith("graph_budget_eager_"):
                    self.assertIn('STATS["graph_budget_eager_" + where]', src); continue
                if mod == "ef2_opt" and k.startswith("sampler_graph_clears_"):                          # counted by reason: STATS["sampler_graph_clears_" + reason] (clear_sampler_graphs)
                    self.assertIn('STATS["sampler_graph_clears_" + reason]', src); continue
                self.assertTrue(f'STATS["{k}"]' in src or f'"{k}"' in src, f"{mod}.{k} is not a counter of {mod}.py")


class TestOneTableEach(unittest.TestCase):
    def test_one_mode_table(self):
        """modes.KIT_MODES is the only place a kit server mode is named in package code."""
        src = _modules()
        for f, s in src.items():
            if f == "modes.py":
                continue
            for name in ("opt14_msa", "opt7x", "opt7x_msa", "opt12x"):
                self.assertNotRegex(s, rf'["\']{name}["\']', f"{f} names the server mode {name}: modes.KIT_MODES is the one table")
        self.assertEqual(tuple(modes.MODES), ("exact", "fast", "big", "off"))                       # the user-facing tuple: off / exact / fast / big, named by guarantee
        self.assertEqual(modes.DEFAULT_MODE, "fast")                     # the package default: fast wherever a fast mode ships
        run_sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        self.assertIn("opt/esmfold2_opt/modes.py DEFAULT_MODE", run_sh)      # run.sh names the one constant …
        self.assertNotRegex(run_sh, r"package default \((exact|fast|off)\)", "run.sh carries a second default literal")   # … never a literal default
        self.assertEqual(set(modes.KIT_MODES), {"exact", "fast", "big"})
        self.assertEqual(modes.KIT_MODES["big"].server_mode, "opt14_msa")                              # big's default line = the fast composition + the XL levers (test_big.py)
        self.assertTrue(modes.KIT_MODES["big"].xl)
        self.assertEqual({n for n, km in modes.KIT_MODES.items() if km.xl}, {"fast", "big"})           # the XL storage levers ride the fast line and therefore big (fast is a subset of big by construction: test_fast_subset_big.py)
        self.assertEqual([n for n, km in modes.KIT_MODES.items() if km.alloc_strict], ["big"])         # one memory mode: the only line that refuses a process by its allocator
        self.assertFalse(any(hasattr(km, "ol") for km in modes.KIT_MODES.values()))                     # one memory composition: the XL levers; no offload option
        self.assertEqual(set(modes.KIT_LINES["big"]), {"fast"}); self.assertFalse(hasattr(modes, "BIG_HOST")); self.assertFalse(hasattr(modes, "OFFLOADS"))
        # the hook's import-free copy of the mode list (the AUTOLOAD track's region) and the table agree exactly: TABLE_PENDING names a mode the hook
        # declares ahead of the table (none today) and must retire when the table gains it; the lock fails loud both ways
        self.assertEqual(set(_autoload.MODES), set(modes.MODES) | set(TABLE_PENDING), "the import-free copy in _autoload.py drifted from modes.MODES")
        self.assertFalse(set(TABLE_PENDING) & set(modes.MODES), "a pending mode reached modes.MODES: retire it from TABLE_PENDING")
        self.assertFalse(hasattr(modes, "RETIRED_MODES")); self.assertFalse(hasattr(_autoload, "RETIRED_MODES"))   # no alias table: the modes are exactly MODES
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE, "the import-free copy in _autoload.py drifted from report.EXIT_NOT_ACTIVE")
        self.assertEqual(_autoload.TAG, report.TAG, "the import-free copy in _autoload.py drifted from report.TAG")

    def test_autoload_declares_every_switch_name(self):
        """_autoload.ENV_NAMES (import-free) is exactly the package's declared ESMFOLD2_OPT* names (stack.py, report.py, attn.py): a name added to
        the package without being declared there would be refused at interpreter start."""
        from esmfold2_opt import attn, ablation
        declared = {stack.ENV_MODE, stack.ENV_FORCE, stack.ENV_HOME, stack.ENV_KIT, stack.ENV_WEIGHTS_MEMO_DIR, report.ENV_CACHE_SCOPE, attn.ENV_REQUIRE, ablation.ENV_ABLATE}
        self.assertEqual(set(_autoload.ENV_NAMES), declared)
        src = _modules()
        used = {m for s in src.values() for m in re.findall(r"ESMFOLD2_OPT(?:_[A-Z][A-Z_]*[A-Z])?(?![A-Z_])", s)}
        self.assertEqual(used - set(_autoload.ENV_NAMES), set(), "an ESMFOLD2_OPT* name in package code that _autoload does not declare")

    def test_the_pin_names_the_imported_core_and_the_gate_is_statement_one_of_every_entry(self):
        """[tool.opt_core] pins the core this process imports (path, a minimum version) and the gate passes on it; the gate is statement
        one of every entry (cli.main, the .pth hook under a named mode, stack's activation gates) and the kit makes no second pin call of
        its own. (Byte identity of opt/_build_backend.py / esmfold2_opt/_core_gate.py against the core's kit_template copies is not
        re-checked here: they are carried byte-identical by the commit that added them, per the kit -- not a runtime re-hash.)"""
        from opt_core.gates import core_pin, imported_core
        from esmfold2_opt._core_gate import gate, version_tuple
        pin = core_pin(os.path.join(OPT, "pyproject.toml")); core = imported_core()
        self.assertLessEqual(version_tuple(pin["version"]), version_tuple(core["version"]))                 # the pin is a FLOOR (>=): the imported core is at or above it (never package_tree_sha256/files: mid-transition on the core branch)
        facts = gate(os.path.join(PKG, "cli.py"))                                                          # passes (returns the facts) on the pinned core
        self.assertEqual(facts["pinned"]["version"], pin["version"])                                       # the gate reports the pin it read; passing on a newer core is the floor semantics
        self.assertEqual(facts["installed"]["package_dir"], core["package_dir"])                            # the gate's notion of "installed" is the same core imported_core() sees
        cli_src = open(os.path.join(PKG, "cli.py"), encoding="utf-8").read()
        body = cli_src.split("def main(", 1)[1].split("\n", 1)[1]
        self.assertTrue(body.lstrip().startswith("core_gate(kit_anchor(__file__))"), "cli.main's first statement is the core gate")
        auto = open(os.path.join(PKG, "_autoload.py"), encoding="utf-8").read()
        self.assertLess(auto.find("\n    _core_gate(_kit_anchor(__file__))"), auto.find("\n        from opt_core.autoload import install"), "the .pth hook gates before its module-level opt_core import")
        self.assertGreater(auto.find("\n    _core_gate(_kit_anchor(__file__))"), 0)
        stack_src = open(os.path.join(PKG, "stack.py"), encoding="utf-8").read()
        self.assertIn("core_gate(os.path.join(opt_home(), \"pyproject.toml\")", stack_src); self.assertNotRegex(stack_src, r"core_pin_check\(")   # one pin producer: the gate; no second pin call

    def test_stock_child_modules_hold_no_core_beyond_the_proof_machinery(self):
        """The stock caller's import closure (stock_fold -> modes, report, settings, det, inputs, outputs) imports no opt_core module at
        module level: every core consumer in those modules is a call-time import (the stock process holds nothing of the core)."""
        src = _modules()
        for f in ("stock_fold.py", "modes.py", "report.py", "settings.py", "det.py", "inputs.py", "outputs.py", "_autoload.py"):
            for line in src[f].splitlines():
                if re.match(r"(from opt_core|import opt_core)", line):
                    self.fail(f"{f}: module-level core import: {line.strip()}")

    def test_one_settings_resolver_and_no_presets(self):
        src = _modules()
        for f, s in src.items():
            self.assertNotRegex(s, r"(?m)^(SETTINGS_PRESETS|DEFAULT_PRESET|PRESETS)\s*=", msg=f"{f} carries a settings-presets table: the kit exposes the library's own knobs only")
            self.assertNotIn('"--settings"', s, f"{f}: a --settings preset flag"); self.assertNotIn('"--preset"', s, f"{f}: a --preset flag")
        self.assertIn("settings.resolve(", src["cli.py"])
        self.assertNotIn("seeds", settings.STOCK_FLAGS)                                      # no seed knob of its own: --seeds / the inputs' keys (the library's seed=None is unseeded)

    def test_stock_route_adds_nothing_beyond_the_settings(self):
        """The stock caller is a pass-through: outside det.py (the named --det recipe) no package module writes a torch backend flag or a
        library switch, and the stock caller's only model calls are the settings' two (settings.py: the library defaults) or --backend's."""
        src = _modules()
        for f, s in src.items():
            if f == "det.py":
                continue
            self.assertNotRegex(s, r"torch\.backends\.\w+(\.\w+)*\s*=", msg=f"{f} writes a torch backend flag outside det.py")
            self.assertNotRegex(s, r"use_deterministic_algorithms\(|set_float32_matmul_precision\(", msg=f"{f} sets a torch switch outside det.py")
        calls = re.findall(r"model\.set_\w+\(", src["stock_fold.py"])
        self.assertEqual(sorted(set(calls)), [], msg="stock_fold.py names a model setter directly; model_calls() is the only path")

    def test_one_variant_table(self):
        src = _modules()
        self.assertEqual(tuple(modes.VARIANTS), ("fast", "full_msa", "full_nomsa"))
        for table in (modes.SERVER_VARIANT, modes.VARIANT_USES_MSA):                     # the per-variant tables name exactly the variants
            self.assertEqual(set(table), set(modes.VARIANTS))

    def test_one_deterministic_recipe(self):
        """det.py is the one place the recipe's values live: the workspace config literal appears nowhere else in package code."""
        src = _modules()
        for f, s in src.items():
            if f == "det.py":
                continue
            self.assertNotIn(":4096:8", s, f"{f} spells the CUBLAS workspace literal: det.py is the one place")
            self.assertNotRegex(s, r"^(DET_ENV|DET_FOLD_KWARGS)\s*=", msg=f"{f} carries a second deterministic recipe")
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", det.ENV)

    def test_verbs(self):
        """The package's verbs: pred (upstream's one task), check, warm — no server, no task verb of its own."""
        self.assertEqual(set(cli.COMMANDS), {"pred", "check", "warm"})
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(cli.__file__), "serve.py")))

    def test_no_measurement_side_file(self):
        """pred writes upstream's outputs and the rows file only: no manifest / env-proof / lever-report writer in any module (the printed lines are the record)."""
        src = _modules()
        self.assertNotIn("manifest.py", src)
        for f, s in src.items():
            for word in ("opt_manifest.json", "stock_env_proof.json", "lever_report", "MANIFEST_SCHEMA"):
                self.assertNotIn(word, s, msg=f"{f} names {word}")


@unittest.skipUnless(_tree_present(), "release tree not present around the package (installed copy): tree locks skipped")
class TestOneFileEach(unittest.TestCase):
    def test_one_stock_caller(self):
        self.assertTrue(os.path.isfile(os.path.join(PKG, "stock_fold.py")))
        self.assertFalse(os.path.exists(os.path.join(TREE, "stock", "stock_fold.py")), "a second stock caller exists under stock/")
        run_sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        self.assertNotIn("stock/stock_fold.py", run_sh)
        self.assertIn("opt/esmfold2_opt/stock_fold.py", run_sh)

    def test_one_kit_class_one_kit(self):
        self.assertTrue(os.path.isdir(os.path.join(TREE, "opt/forward/fast_inference")))
        self.assertEqual(sorted(os.listdir(os.path.join(TREE, "opt/forward"))), ["EF2_XL_ADDON_v1", "fast_inference"])   # the kit + the memory add-on, each carried whole
        for rel in ("opt/serving", "opt/datapath", "opt/forward/fpft"):
            self.assertFalse(os.path.exists(os.path.join(TREE, rel)), f"{rel}: the tree carries the kit only (forward class); the opt-in add-ons are not carried")

    def test_run_sh_commands_match_the_cli(self):
        run_sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        m = re.search(r'case "\$CMD" in ([a-z|]+)\)', run_sh)
        self.assertIsNotNone(m, "run.sh's command case line not found")
        self.assertEqual(set(m.group(1).split("|")), set(cli.COMMANDS))
        self.assertNotRegex(run_sh, r"run\.sh fold\b", msg="the command is pred, not fold")
        self.assertIn("python -m esmfold2_opt", run_sh)

    def test_one_variants_literal_and_run_sh_in_step(self):
        """ONE literal for the variant names (registry.ALL_VARIANTS); modes.VARIANTS re-exports it; run.sh's usage names the same set;
        no second tuple of the three names in package code."""
        from esmfold2_opt import modes, registry
        self.assertIs(modes.VARIANTS, registry.ALL_VARIANTS)
        run_sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        m = re.search(r"--variant ([a-z_|]+)", run_sh)
        self.assertIsNotNone(m); self.assertEqual(m.group(1).split("|"), list(registry.ALL_VARIANTS))
        literal = re.compile(r'\(\s*"fast"\s*,\s*"full_msa"\s*,\s*"full_nomsa"\s*\)')
        hits = [f for f in os.listdir(PKG) if f.endswith(".py") and literal.search(open(os.path.join(PKG, f), encoding="utf-8").read())]
        self.assertEqual(hits, ["registry.py"], f"the variants tuple is written in more than one place: {hits}")

    def test_packed_serving_is_not_wired(self):
        """No --pack / PACK_K anywhere in the house files: packed serving is the add-ons' own launchers (serve.py's own docstring)."""
        files = [os.path.join(TREE, f) for f in ("run.sh", "configs/h100.env")]
        files += [os.path.join(PKG, f) for f in os.listdir(PKG) if f.endswith(".py")]
        self._assert_not_wired(files)

    def test_packed_serving_is_not_wired_in_the_root_readme(self):
        """The same lock over the release tree's root README (its ESMFold2 section) — beside the kit in the tree; a kit copy (the frozen
        engine subtree) has no root README: skipped by name, never a silent pass."""
        root_readme = os.path.join(os.path.dirname(TREE), "README.md")
        if not os.path.isfile(root_readme):
            raise unittest.SkipTest(f"the release tree's root README is not beside the kit ({root_readme}): the kit runs from its engine subtree alone; the root-README lock is skipped")
        self._assert_not_wired([root_readme])

    def _assert_not_wired(self, files):
        for p in files:
            src = open(p, encoding="utf-8").read()
            if p.endswith("README.md") and "## ESMFold2" in src:
                src = src[src.index("## ESMFold2"):]                                       # the root README's ESMFold2 section
                src = src[:src.index("\n## ", 1)] if "\n## " in src[1:] else src
            self.assertNotIn("PACK_K", src, p)
            self.assertNotRegex(src, r"--pack\b(?!`\.)", msg=p)                             # the one mention is the 'has no --pack' sentence

    def test_no_retired_names_in_release_files(self):
        """House-written release files name no mode outside the kit's set (one such word is checked), and
        stock/PINS.json carries no settings block. README.md's own word list is
        test_readme_carries_no_forbidden_vocabulary, below."""
        files = [os.path.join(TREE, f) for f in ("run.sh", "README.md", "configs/h100.env", "stock/PINS.json",
                                                 "opt/pyproject.toml", "opt/_build_backend.py", "opt/esmfold2_opt_autoload.pth")]
        files += [os.path.join(PKG, f) for f in os.listdir(PKG) if f.endswith(".py")]
        files += [os.path.join(os.path.dirname(TREE), "README.md")]
        for p in files:
            if not os.path.isfile(p):
                continue
            s = open(p, encoding="utf-8").read()
            self.assertNotIn("conservative", s, p)
            self.assertNotRegex(s, r"--mode conservative|run\.sh fold ", p)
        pins = stack.pins()
        self.assertNotIn("settings", pins, "stock/PINS.json carries a settings block: the library's own defaults (library_defaults) are the only settings record")

    def test_readme_carries_no_forbidden_vocabulary(self):
        """README.md stays free of the words itemized below (package-internal terms, time-relative wording, a
        bare pointer line, a mode name outside the kit's set), checked directly and word-boundary matched
        where a plain substring check would misfire."""
        readme = os.path.join(TREE, "README.md")
        s = open(readme, encoding="utf-8").read().lower()
        # ALLOW (scoped): the release tree's shared pointer line and the four code spans in which this README quotes the kit's own
        # printed tokens / variable name literally — the program's spelling, not prose; any other span and all prose stay held to
        # the list below.
        s = s.replace("lever detail: changes.md · pins and variables: stock.md.", "")
        for quoted in ("`[esmfold2-opt] active mode=fast variant=fast server_mode=opt14_msa … n_gpu=1 sharding=none levers=…`",
                       "`lever name=… state=…`", "`lever`", "`model_opt_levers_off=<lever>[,<lever>…]`"):
            s = s.replace(quoted, "")
        # Word-boundary matched, not plain substring: a bare `in` check on "gate"/"arms"/"seal" would false-positive on
        # ordinary English ("aggregate", "warms", "resealed"). "class-2" and "named refusal" are distinctive phrases
        # where that risk doesn't apply, so they stay as direct substring checks.
        forbidden_word_boundary = ("arms", "lever", "levers", "gate", "holder", "seal")
        for word in forbidden_word_boundary:
            self.assertNotRegex(s, rf"\b{re.escape(word)}\b", f"README.md uses forbidden project vocabulary: {word!r}")
        for phrase in ("class-2", "named refusal"):
            self.assertNotIn(phrase, s, f"README.md uses forbidden project vocabulary: {phrase!r}")
        provenance_words = ("previously", "superseded", "pending")
        for word in provenance_words:
            self.assertNotRegex(s, rf"\b{re.escape(word)}\b", f"README.md uses provenance language: {word!r}")
        self.assertNotRegex(s, r"\bwave \d", "README.md uses provenance language: a 'wave N' reference")
        self.assertNotRegex(s, r"(?m)^\s*results:\s*$", "README.md has a bare 'Results:' pointer line")

if __name__ == "__main__":
    unittest.main()
