"""The mode table: modes.KIT_MODES is the one table; exact resolves to the kit README rows (read, not transcribed), the base variants' row with the
lowmem transform; --x_all expands to what the worker itself applies; fast is not a mode of this engine and is refused by name before anything
resolves; the documented flags exist in the kit executables' argparse. No torch, no GPU."""
import os
import re
import unittest

from proteinmpnn_opt import modes, registry, stack

KIT = stack.kit_home()


class TestTable(unittest.TestCase):
    def test_modes_and_variants(self):
        self.assertEqual(tuple(modes.MODES), ("exact", "off"))
        self.assertEqual((modes.DEFAULT_MODE, modes.KIT_MODE), (None, "exact")); self.assertIn(modes.KIT_MODE, modes.MODES)
        self.assertEqual(tuple(modes.VARIANTS), ("soluble", "vanilla"))
        self.assertEqual(modes.DEFAULT_VARIANT, "vanilla")                    # upstream's own default weight set (protein_mpnn_run.py without --use_soluble_model)
        self.assertEqual(set(modes.KIT_MODES), set(modes.MODES))
        for mode in modes.MODES:
            self.assertIn("route", modes.KIT_MODES[mode], mode)                   # one row per mode: the variants are weight sets of one program

    def test_off_is_stock_for_every_variant(self):
        for v in modes.VARIANTS:
            r = modes.resolve("off", v, KIT)
            self.assertEqual(r.route, "stock")
            self.assertEqual(r.flags, [])
            self.assertEqual(r.levers, [])

    def test_exact_is_the_readme_row(self):
        for v in modes.VARIANTS:
            r = modes.resolve("exact", v, KIT)
            self.assertEqual(r.route, "worker")
            self.assertEqual(r.executable, "kit/mpnn_worker2.py")
            self.assertEqual(r.flags, ["--mode", "stream", "--bb_batch", "16", "--sort_by_length", "--x_all", "--hybrid_gemm"])
            self.assertEqual(r.bb_batch, 16)
            self.assertTrue(r.hybrid_gemm)                                       # the probe-gated lever is requested (KIT_MODES probe_gated)
            self.assertEqual(r.probe_gated, ["--hybrid_gemm"])
            self.assertEqual(r.row, modes.EXACT_ROW)
            self.assertIn("--x_all", r.line)
            self.assertTrue(r.line.startswith("python kit/mpnn_worker2.py"), r.line)
            self.assertEqual(r.transform, modes.LOWMEM)                          # executed as kit/mpnn_worker2_lowmem.py (stage.stage_lowmem)
            self.assertEqual(r.levers[-1], "lowmem")                             # reported and evidenced like a flag lever
            self.assertIn("mpnn_worker2_lowmem.py", modes.describe_line(r))
            self.assertEqual(modes.describe_line(r), "kit/mpnn_worker2.py <stock args> --mode stream --bb_batch 16 --sort_by_length --x_all --hybrid_gemm "
                                                    "(run as kit/mpnn_worker2_lowmem.py: + the low-memory featuriser and decoding-order mask, lowmem.py)")



    def test_resolve_takes_the_callers_backbone_batch(self):
        r = modes.resolve("exact", "soluble", KIT, bb_batch=3)
        self.assertEqual(r.flags, ["--mode", "stream", "--bb_batch", "3", "--sort_by_length", "--x_all", "--hybrid_gemm"]); self.assertEqual(r.bb_batch, 3)
        self.assertEqual(r.levers, modes.resolve("exact", "soluble", KIT).levers)          # the batch is no lever: the set is the row's
        self.assertEqual(modes.resolve("exact", "vanilla", KIT, bb_batch=None).flags[3], "16")
        for bad in (0, -1, True, 2.5, "4"):
            with self.assertRaises(modes.ModeError):
                modes.resolve("exact", "soluble", KIT, bb_batch=bad)
        with self.assertRaises(modes.ModeError) as cm:
            modes.resolve("off", "soluble", KIT, bb_batch=8)
        self.assertIn("mode off runs the stock command line, one backbone per forward pass; the backbone batch is the kit line's (--mode exact)", str(cm.exception))
    def test_readme_row_is_a_command_of_the_named_executable(self):
        base = modes.readme_row(KIT, modes.EXACT_ROW)
        self.assertTrue(base.startswith("python kit/mpnn_worker2.py "), base)
        # the settings the row carries are the kit's own (stripped by resolve): all present in the row, none in the flags
        for flag in ("--seed", "--num_seq_per_target", "--batch_size", "--sampling_temp", "--out_folder", "--jsonl_path"):
            self.assertIn(flag, base)

    def test_x_all_expansion_matches_the_workers_own_assignment(self):
        exp, asg = modes.x_all_expansion(KIT), modes.x_all_assignment(KIT)
        self.assertEqual(sorted(exp), sorted(asg))
        self.assertEqual(len(exp), 7)
        self.assertNotIn("--hybrid_gemm", exp)                      # the probe-gated lever is not in the worker's --x_all set: the table requests it beside
        r = modes.resolve("exact", "soluble", KIT)
        self.assertEqual(set(r.levers), {"stream", "sort_by_length", "hybrid_gemm"} | {f.lstrip("-") for f in exp} | {modes.LOWMEM})   # + the package transform, reported like a lever
        self.assertEqual(modes.worker_flag(KIT, "--hybrid_gemm"), "--hybrid_gemm")
        with self.assertRaises(modes.ModeError):
            modes.worker_flag(KIT, "--no_such_switch")

    def test_cpu_rule_is_the_workers_own(self):
        off = modes.cpu_disabled_levers(KIT)
        self.assertEqual(set(off), {"--graph_rng", "--single_graph", "--fused_draw"})

    def test_probe_statement_is_read_from_the_readme(self):
        p = modes.probe_kit_observed(KIT)
        self.assertTrue(any("sm_90" in n for n in p["PASS"]))
        self.assertTrue(any("sm_89" in n for n in p["FAIL"]))
        self.assertEqual(modes.probe_verdict_kit_observed(KIT, "sm_90"), "PASS")
        self.assertEqual(modes.probe_verdict_kit_observed(KIT, "sm_89"), "FAIL")
        self.assertEqual(modes.probe_verdict_kit_observed(KIT, "sm_80"), "PASS")     # A100: grouped under the worker's row ceiling (HYBRID_GROUP_ROWS)
        self.assertIsNone(modes.probe_verdict_kit_observed(KIT, "sm_86"))
        self.assertIsNone(modes.probe_verdict_kit_observed(KIT, None))

    def test_fast_is_refused_by_name(self):
        """proteinmpnn ships no fast tier: the name is refused before anything resolves (check_mode -> UnsupportedMode, a ModeError) with the pointer,
        for every variant and spelling; the table has no fast row; an unknown name stays a usage error with the kit's usage text."""
        pointer = "proteinmpnn ships no fast tier: select --mode exact"
        self.assertEqual(modes.unsupported_message("fast"), pointer)
        for v in modes.VARIANTS:
            with self.assertRaises(modes.UnsupportedMode) as cm:
                modes.resolve("fast", v, KIT)
            self.assertEqual(str(cm.exception), pointer, v)
        with self.assertRaises(modes.UnsupportedMode):
            modes.check_mode(" FAST ")
        self.assertNotIn("fast", modes.MODES); self.assertNotIn("fast", modes.KIT_MODES); self.assertEqual(set(modes.KIT_MODES), set(modes.MODES))
        with self.assertRaises(modes.ModeError) as cm:
            modes.check_mode("turbo")
        self.assertNotIsInstance(cm.exception, modes.UnsupportedMode); self.assertEqual(str(cm.exception), "unknown mode 'turbo': choose one of exact|off")

    def test_no_default_mode(self):
        """DEFAULT_MODE is a value written in modes.py, never computed from the table: None — the family default (fast) does not ship here, so no mode
        named is a usage error whose one line lists the modes served, and fast by name is the refusal pointing at the kit mode; for every variant."""
        self.assertIsNone(modes.DEFAULT_MODE)
        for v in modes.VARIANTS:
            for given in (None, "", "  "):
                with self.assertRaises(modes.ModeError) as cm:
                    modes.resolve(given, v, KIT)
                self.assertNotIsInstance(cm.exception, modes.UnsupportedMode)
                self.assertEqual(str(cm.exception), "no mode named: --mode off|exact (or PROTEINMPNN_OPT) — proteinmpnn ships no fast tier, so it has no default mode")
            for given in ("fast", " FAST "):
                with self.assertRaises(modes.UnsupportedMode) as cm:
                    modes.resolve(given, v, KIT)
                self.assertEqual(str(cm.exception), "proteinmpnn ships no fast tier: select --mode exact")
            self.assertEqual(modes.resolve("exact", v, KIT).mode, "exact"); self.assertEqual(modes.resolve("off", v, KIT).mode, "off")

    def test_unknown_names(self):
        with self.assertRaises(modes.ModeError):
            modes.resolve("turbo", "soluble", KIT)
        with self.assertRaises(modes.ModeError):
            modes.resolve("exact", "membrane", KIT)
        with self.assertRaises(modes.ModeError):
            modes.check_mode(None)                                                # no default mode: a usage error
        self.assertEqual(modes.check_variant(""), modes.DEFAULT_VARIANT)

    def test_documented_flags_exist_in_the_kit_executable(self):
        with open(os.path.join(KIT, modes.WORKER_DIR, modes.WORKER), encoding="utf-8") as fh:
            worker_flags = set(re.findall(r'add_argument\("(--\w+)"', fh.read()))
        r = modes.resolve("exact", "soluble", KIT)
        for f in r.flags:
            if f.startswith("--"):
                self.assertIn(f, worker_flags, f)
        for f in modes.x_all_expansion(KIT):
            self.assertIn(f, worker_flags, f)

    def test_every_lever_flag_is_in_the_registry(self):
        for v in modes.VARIANTS:
            r = modes.resolve("exact", v, KIT)
            for f in r.flags:
                if f.startswith("--"):
                    self.assertIn(f, registry.LEVERS, f)
            self.assertEqual(registry.tier_of(modes.expanded_flags(r.flags, KIT)), 1)      # the exact rows are tier 1 as sets
        for f in modes.x_all_expansion(KIT):
            self.assertIn(f, registry.LEVERS, f)
            self.assertEqual(registry.LEVERS[f]["tier"], 1)
        self.assertEqual((registry.WORKER_DIR, registry.PARSER_DIR), (modes.WORKER_DIR, modes.PARSER_DIR))   # the registry's file paths use the mode table's directory names

    def test_tier_is_a_property_of_the_set(self):
        exact = modes.expanded_flags(modes.resolve("exact", "soluble", KIT).flags, KIT)
        self.assertIn("--chunk_gemm", exact); self.assertIn("--hybrid_gemm", exact); self.assertEqual(registry.tier_of(exact), 1)
        candidate = [f for f in exact if f not in ("--chunk_gemm", "--hybrid_gemm")]   # without --chunk_gemm --hybrid_gemm: batched-GEMM accumulation, tier 2
        self.assertEqual(registry.tier_of(candidate), 2)
        self.assertEqual(registry.tier_of(candidate + ["--chunk_gemm"]), 1)
        one_bb = list(candidate)
        one_bb[one_bb.index("--bb_batch") + 1] = "1"                        # K = 1: the stock shape whatever the GEMM path
        self.assertEqual(registry.tier_of(one_bb), 1)
        self.assertEqual(registry.tier_of(exact + ["--hybrid_gemm"]), 1)     # probe-gated, bit-identical either way (with --chunk_gemm)
        self.assertEqual(registry.tier_of([]), 1)
        for f, v in registry.LEVERS.items():                                 # every worker flag the registry names is one the worker declares
            if f.startswith("--") and v["file"].endswith("mpnn_worker2.py"):
                self.assertEqual(modes.worker_flag(KIT, f), f)
