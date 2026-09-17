"""The `big` mode: the kit's fast composition (server mode `opt14_msa`) with no CUDA graph captured and the XL add-on's levers
(modes.XL_BIG_SET) installed before configure().
Source-level and pure-python: the knob set, the lever accounting. The engagement gate (every lever's counter,
no TriMul passthrough) is exercised by a GPU run on a box."""
import os
import re
import subprocess
import sys
import unittest
from unittest import mock
import unittest.mock

from esmfold2_opt import big, cli, modes, registry, stack

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBigMode(unittest.TestCase):
    def test_definition(self):
        self.assertIs(modes.KIT_MODES["big"], modes.BIG_FAST)       # the default line: the fast composition (PROTOCOL: big composes on top of fast)
        self.assertEqual(set(modes.XL_BIG_SET), set(registry.XL_LEVERS))               # every registered XL lever is in the mode's set: no lever without a mode
        self.assertEqual(modes.XL_BIG_SET, modes.XL_FAST_SET + ("x4",))                # the default line: the fused-compatible frees + the LM host offload
        self.assertEqual(modes.BIG_FAST.xl_set, modes.XL_BIG_SET); self.assertEqual(modes.BIG_FAST.xl_knobs["esmc_min_tok"], modes.X4_MIN_TOKENS)
        for n in modes.XL_BIG_SET:
            self.assertEqual(registry.LEVERS[n].field, registry.FIELD_XL)
            self.assertEqual(registry.LEVERS[n].probe[0], "xl")

    def test_knobs_follow_the_set(self):
        """`cond` is 'lean'; x2b only at one diffusion sample; x4 by the set."""
        kb = stack.xl_knobs(samples=1, has_msa_encoder=False, xl_set=modes.XL_BIG_SET)
        self.assertEqual(kb["cond"], "lean")
        self.assertTrue(kb["esmc_offload"]); self.assertEqual(stack.xl_levers_on(kb), ["x2b", "x3", "x6", "x7", "x8", "x10", "x4"])
        self.assertEqual(dict(kb, **modes.XL_BIG_KNOBS)["esmc_min_tok"], modes.X4_MIN_TOKENS)   # the line's fixed keywords carry the threshold (big.xl_install merges them)
        k5 = stack.xl_knobs(samples=5, has_msa_encoder=True, xl_set=modes.XL_BIG_SET)
        self.assertEqual(k5["free"], "")                                                  # x2b (FREE) only at one diffusion sample

    def test_resolution_names_the_xl_levers(self):
        """resolve() lists the XL levers for the mode."""
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        res = modes.resolve("big", "fast", kit)
        self.assertTrue(res.composition["xl"])
        self.assertEqual([n for n in res.levers if registry.LEVERS[n].field == registry.FIELD_XL], list(modes.XL_BIG_SET))   # the XL levers by their registry field (xln is the LayerNorm lever, not an XL one)
        self.assertFalse(modes.resolve("exact", "fast", kit).composition["xl"])

    def test_xl_addon_is_in_the_tree_and_the_gate_is_idle_without_an_install(self):
        self.assertTrue(os.path.isfile(os.path.join(stack.xl_home(), "ef2_xl.py")))
        self.assertEqual(stack.xl_gate(), {"checked": False})


class TestBigFastLine(unittest.TestCase):
    """`big --line fast`: the kit's fast composition (opt14_msa) with no CUDA graph captured and the XL levers that compose on the fused
    kernels. The default line is `fast`."""

    def test_compositions_and_the_mode(self):
        """big = ONE lever set (its `fast` composition); no line selector."""
        self.assertEqual(set(modes.KIT_LINES), {"big"})
        self.assertEqual(modes.DEFAULT_LINE, {"big": "fast"})
        self.assertIs(modes.KIT_LINES["big"]["fast"], modes.KIT_MODES["big"])
        self.assertEqual(set(modes.KIT_LINES["big"]), {"fast"})
        self.assertFalse(hasattr(modes, "BIG_REFERENCE")); self.assertNotIn("--line", cli.pred_parser().format_help())
        fast = modes.KIT_LINES["big"]["fast"]
        self.assertEqual((fast.server_mode, fast.overrides, fast.xl, fast.backend, fast.line), ("opt14_msa", {modes.ENV_GRAPH_CAPTURE: modes.BIG_GRAPH_CAPTURE, modes.ENV_W4_IDPROBE: "0"}, True, None, "fast"))
        self.assertEqual(fast.xl_set, modes.XL_BIG_SET)
        self.assertEqual(fast.xl_knobs, {"own": False, "esmc_min_tok": modes.X4_MIN_TOKENS})
        for name, km in modes.KIT_MODES.items():
            self.assertEqual(km.line, "fast" if name == "big" else None, name)

    def test_apply_follows_the_resolved_line_not_the_mode_default(self):
        """The model calls and the knobs come from the RESOLVED composition's entry: the entry looked up by the resolved key,
        never a second table (the final20 finding: the OPM lever silently passed through, caught by the census gate)."""
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        fast = modes.resolve("big", "full_msa", kit)
        self.assertIsNone(modes.kit_mode(fast.mode, fast.line).backend)
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stack.py"), encoding="utf-8").read()
        self.assertIn("_kit_mode(res.mode, res.line)", src)                                   # apply_to looks the entry up by the resolved line
        self.assertNotIn("_KIT_MODES.get(res.mode)", src)

    def test_kit_mode_refuses_a_line_by_name(self):
        with self.assertRaises(ValueError):
            modes.kit_mode("exact", "fast")
        with self.assertRaises(ValueError):
            modes.kit_mode("big", "nope")
        self.assertIs(modes.kit_mode("big", None), modes.KIT_MODES["big"])

    def test_the_row_sharded_route_subtracts_its_own_table_by_name(self):
        """resolve(n_gpu > 1) removes the levers the multi-GPU line's table names (tp.not_for_route -> modes.route_drop) from configure()'s set with
        their reason, exactly like the line's drops; the LEVER line says ``state=skipped reason=not_for_route:n_gpu=<P>:<reason>``, the APPLIED /
        DRY-RUN lines carry ``not_for_route=<...>``; nothing of it at n_gpu 1."""
        from unittest import mock
        from esmfold2_opt import report, tp
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        table = {"ls": "rows_reissue_the_loop", "m17": "rows_own_opm", "nope": "not_in_the_set"}
        with mock.patch.object(tp, "not_for_route", create=True, side_effect=lambda P: dict(table) if P > 1 else {}):
            one = modes.resolve("fast", "full_msa", kit)
            self.assertEqual((one.not_for_route, one.n_gpu), ({}, 1)); self.assertIn("ls", one.levers); self.assertIn("m17", one.levers)
            two = modes.resolve("fast", "full_msa", kit, n_gpu=2)
            self.assertEqual(two.not_for_route, {"ls": "rows_reissue_the_loop", "m17": "rows_own_opm"})   # only names of the set; an outsider of the table is ignored
            self.assertNotIn("ls", two.levers); self.assertNotIn("m17", two.levers); self.assertIn("t15", two.levers)
            self.assertIn("ls", two.levers_off); self.assertIn("m17", two.levers_off)
        rep = {"mode": "fast", "variant": "full_msa", "n_gpu": 2, "levers_not_for_route": dict(two.not_for_route), "levers_planned": two.levers_for_variant}
        lines = report.lever_lines(rep, {"levers_applied": [], "model_index": 0})
        ls_line = next(l for l in lines if " name=ls " in l)
        self.assertIn("state=skipped reason=not_for_route:n_gpu=2:rows_reissue_the_loop", ls_line)
        self.assertEqual(report.route_word(rep), " not_for_route=ls,m17"); self.assertEqual(report.route_word(dict(rep, n_gpu=1)), "")

    def test_resolution_of_the_fast_line(self):
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        res = modes.resolve("big", "full_msa", kit, line="fast")
        self.assertEqual((res.line, res.server_mode, res.overrides, res.xl_set), ("fast", "opt14_msa", {modes.ENV_GRAPH_CAPTURE: modes.BIG_GRAPH_CAPTURE, modes.ENV_W4_IDPROBE: "0"}, modes.XL_BIG_SET))
        self.assertTrue(res.composition["xl"])
        self.assertIn("x2b", res.levers); self.assertNotIn("x2", res.levers); self.assertIn("m17", res.levers); self.assertIn("t15", res.levers); self.assertIn("af", res.levers)
        self.assertEqual(res.drop, modes.BIG_DROP)
        for dropped in ("disto", "mh"):                                                   # the line's own drops: x10 owns the distogram move, mh pins memory; ls / rg / ro / kd / dit ride
            self.assertNotIn(dropped, res.levers); self.assertIn(dropped, modes.BIG_DROP)   # the line — G4 / G5 own x2's release in their frames, the fused step calls x3's hook
        for carried in ("ls", "rg", "ro", "kd", "dit"):
            self.assertIn(carried, res.levers); self.assertNotIn(carried, modes.BIG_DROP)
        bare = modes.resolve("big", "full_msa", kit)
        self.assertEqual((bare.line, bare.server_mode, bare.levers), ("fast", "opt14_msa", res.levers))   # the bare mode = the default line

    def test_census_gate_names_every_passthrough(self):
        """The carry's fallback census: a stock probe with calls beyond the scope rule, a row-wise lever whose xl count trails its call count,
        a per-fold counter off the fold count — each fails the gate BY NAME; the counters of a
        clean run pass."""
        saved = dict(stack.XL_STATE); saved_calls = dict(stack.XL_STOCK_CALLS)
        clean = {"cond_calls": 4, "cond_xl_calls": 4, "s2p_calls": 4, "s2p_xl_calls": 4, "relpos_calls": 4, "relpos_xl_calls": 4, "loop_calls": 4,
                 "init_xl_calls": 4, "disto_cpu_events": 4, "free_events": 0, "owned_blocks": 1256}
        levers = ["x3", "x6", "x7", "x8", "x10"]
        try:
            stack.XL_STATE.clear(); stack.XL_STATE.update({"installed": True, "levers": levers, "knobs": {"own": False}, "samples": 5})
            stack.XL_STOCK_CALLS.clear(); stack.XL_STOCK_CALLS.update({"relpos": 0, "init": 0})
            with unittest.mock.patch.object(big, "xl_stats", return_value=dict(clean)):
                rec = stack.xl_gate(expected_folds=4)
            self.assertTrue(rec["ok"]); self.assertEqual(rec["stock_probes"]["relpos"], 0); self.assertEqual(rec["expected_folds"], 4)
            cases = [({"relpos": 1}, {}, "relpos stock calls=1 expected=0"),
                     ({}, {"cond_xl_calls": 3}, "x3 passthrough (1 of 4"), ({}, {"init_xl_calls": 3}, "x8 init_xl_calls=3 expected=4")]
            for probe_over, stat_over, needle in cases:
                stack.XL_STOCK_CALLS.update({"relpos": 0, "init": 0}); stack.XL_STOCK_CALLS.update(probe_over)
                with unittest.mock.patch.object(big, "xl_stats", return_value=dict(clean, **stat_over)):
                    with self.assertRaises(stack.ActivationError) as cm:
                        stack.xl_gate(expected_folds=4)
                self.assertIn(needle, str(cm.exception), needle)
        finally:
            stack.XL_STATE.clear(); stack.XL_STATE.update(saved); stack.XL_STOCK_CALLS.clear(); stack.XL_STOCK_CALLS.update(saved_calls)

    def test_gate_accounts_x4_by_size(self):
        """x4 engages by size: below its threshold on every LM pass it is `off-by-size` (named, gate ok); at or past the threshold every pass
        must have offloaded (events == expected) or the gate refuses BY NAME; a clean engaged run passes with the census in the record."""
        saved = dict(stack.XL_STATE); saved_tok = list(stack.XL_LM_TOKENS)
        try:
            stack.XL_STATE.clear(); stack.XL_STATE.update({"installed": True, "levers": ["x4"], "knobs": {"own": False, "esmc_offload": True, "esmc_min_tok": 1500}})
            stack.XL_LM_TOKENS[:] = [1002, 1002]
            with unittest.mock.patch.object(big, "xl_stats", return_value={"esmc_offload_events": 0}):
                rec = stack.xl_gate(expected_folds=2)
            self.assertTrue(rec["ok"]); self.assertEqual((rec["x4"]["state"], rec["x4"]["expected"], rec["x4"]["max_tokens"], rec["x4"]["threshold"]), ("off-by-size", 0, 1002, 1500))
            stack.XL_LM_TOKENS[:] = [1559, 1002]
            with unittest.mock.patch.object(big, "xl_stats", return_value={"esmc_offload_events": 1}):
                rec = stack.xl_gate(expected_folds=2)
            self.assertTrue(rec["ok"]); self.assertEqual((rec["x4"]["state"], rec["x4"]["events"], rec["x4"]["expected"]), ("engaged", 1, 1))
            with unittest.mock.patch.object(big, "xl_stats", return_value={"esmc_offload_events": 0}):
                with self.assertRaises(stack.ActivationError) as cm:
                    stack.xl_gate(expected_folds=2)
            self.assertIn("x4 esmc_offload_events=0 expected=1 (1 of 2 LM passes at >= 1500 tokens; max 1559)", str(cm.exception))
            stack.XL_LM_TOKENS[:] = [1002]
            with unittest.mock.patch.object(big, "xl_stats", return_value={"esmc_offload_events": 1}):      # an offload below the threshold is a mismatch too
                with self.assertRaises(stack.ActivationError):
                    stack.xl_gate(expected_folds=1)
        finally:
            stack.XL_STATE.clear(); stack.XL_STATE.update(saved); stack.XL_LM_TOKENS[:] = saved_tok

    def test_lm_probe_records_token_counts_and_calls_through(self):
        class M:
            def _compute_lm_hidden_states(self, input_ids, *a, **k):
                return ("lm", input_ids.shape[-1])
        class Ids:
            def __init__(self, n): self.shape = (1, n)
        saved_tok = list(stack.XL_LM_TOKENS)
        try:
            stack.XL_LM_TOKENS[:] = []
            self.assertTrue(stack.xl_lm_probe(M())); self.assertTrue(stack.xl_lm_probe(M()))                 # idempotent
            self.assertEqual(M()._compute_lm_hidden_states(Ids(1557)), ("lm", 1557)); M()._compute_lm_hidden_states(Ids(998))
            self.assertEqual(stack.XL_LM_TOKENS, [1557, 998]); self.assertEqual(M._compute_lm_hidden_states._ef2_opt_probe, "lm_tokens")
        finally:
            stack.XL_LM_TOKENS[:] = saved_tok

    def test_knobs_of_the_fast_set(self):
        k = stack.xl_knobs(samples=5, has_msa_encoder=True, xl_set=modes.XL_FAST_SET)
        self.assertEqual(k["free"], "")
        self.assertTrue(k["lmpair"] and k["relpos"] and k["initlean"] and k["distocpu"]); self.assertEqual(k["cond"], "lean")
        self.assertFalse(k["own"]); self.assertNotIn("loopfree", k)                                   # the add-on's loop re-issue is never applied by the kit (ef2_opt's static loop releases the recycle's temporaries)
        self.assertEqual(stack.xl_levers_on(k), ["x3", "x6", "x7", "x8", "x10"])
        k1 = dict(stack.xl_knobs(samples=1, has_msa_encoder=True, xl_set=modes.XL_FAST_SET), **modes.XL_FAST_KNOBS)
        self.assertEqual(stack.xl_levers_on(k1), ["x2b", "x3", "x6", "x7", "x8", "x10"]); self.assertFalse(k1["own"]); self.assertEqual(k1["free"], "z,relpos")


class TestHostResidencyOption(unittest.TestCase):
    """big ships ONE composition: no `--offload host` option, no `--line` selector."""
    def test_no_host_option_and_no_selector(self):
        self.assertNotIn("big_host", modes.MODES); self.assertEqual(set(modes.KIT_LINES["big"]), {"fast"})
        self.assertFalse(hasattr(modes, "BIG_HOST")); self.assertFalse(hasattr(modes, "OFFLOADS"))
        help_text = cli.pred_parser().format_help()
        self.assertNotIn("--offload", help_text); self.assertNotIn("--line", help_text)

    def test_cli_refuses_the_removed_selectors_by_name(self):
        env = dict(os.environ); env.pop("ESMFOLD2_OPT", None); env["MODEL_OPT"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for argv, word in ((["--mode", "big", "--offload", "host"], "unrecognized arguments: --offload"), (["--mode", "big", "--line", "reference"], "unrecognized arguments: --line")):
            r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", *argv, "--variant", "fast", "--input", "x.json", "--out_dir", "/nonexistent", "--seeds", "42"],
                               env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, cli.EXIT_USAGE, (argv, r.stderr[-500:])); self.assertIn(word, r.stderr)


class TestProbeOffUnderTheMemoryMode(unittest.TestCase):
    """ef2_w4's first-call probe (ENV_W4_IDPROBE) allocates the stock op's O(N²·C) buffers beside T9/T10 once per name: OFF ("0") on every
    big line — resident fast line, reference line, the host-residency option — and therefore at every --n_gpu>1 (P>1 is big-only); exact / fast at P=1
    leave the add-on's default. A diagnostic never costs reach."""
    def test_probe_env_by_line(self):
        for name, km in (("big", modes.KIT_MODES["big"]),):
            self.assertEqual(km.overrides.get(modes.ENV_W4_IDPROBE), "0", name)
        for name in ("exact", "fast"):
            self.assertNotIn(modes.ENV_W4_IDPROBE, modes.KIT_MODES[name].overrides, name)
        src = open(os.path.join(stack.kit_home(), "driver", "ef2_w4.py"), encoding="utf-8").read() if os.path.isfile(os.path.join(stack.kit_home(), "driver", "ef2_w4.py")) else ""
        if src:
            self.assertIn(f'os.environ.get("{modes.ENV_W4_IDPROBE}", "1") != "0"', src)        # the word the driver reads: "0" disables


class TestEngineAdapter(unittest.TestCase):
    """D40: every memory INSTALL lives in the engine adapter esmfold2_opt.big (apply() / report()); stack calls it and re-exports its names
    (one spelling for the readers); the TP hook point refuses more than one GPU by name."""
    def test_apply_and_report_are_the_adapter_surface(self):
        self.assertTrue(callable(big.apply) and callable(big.report))
        for name in stack._BIG_NAMES:
            self.assertIs(getattr(stack, name), getattr(big, name), name)
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stack.py"), encoding="utf-8").read()
        for fn in ("def xl_install(", "def xl_gate(", "def xl_stock_probes("):
            self.assertNotIn(fn, src, fn)                                       # no second copy of an install in stack
        self.assertIn("_big.apply(model, res, samples, out_dir)", src)
        self.assertIn("out.update(_big.report())", src)

    def test_n_gpu_above_one_refuses_by_name_at_the_install(self):
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        res = modes.resolve("big", "fast", kit)
        from esmfold2_opt import tp
        with self.assertRaises(stack.ActivationError) as cm:
            big.apply(object(), res, 1, None, n_gpu=3)
        self.assertIn(tp.REFUSE_NOT_SHIPPED.format(P=3), str(cm.exception))          # the axis's one sentence (tp.py), not a second spelling
        self.assertNotIn("tp", big.report())

    def test_report_is_empty_before_an_install(self):
        saved = dict(big.XL_STATE)
        try:
            big.XL_STATE.clear(); big.XL_STATE["installed"] = False
            self.assertEqual(big.report(), {})
        finally:
            big.XL_STATE.clear(); big.XL_STATE.update(saved)


class TestExitTallyCarriesTheBudgetCounters(unittest.TestCase):
    def test_budget_eager_counters_pass_the_key_word_filter(self):
        """A budget-eager fold is a named event in the EXIT tally (ef2_opt STATS graph_budget_eager_<site>), never folded into an ambiguous
        sampler_fallback count: the key-word filter keeps every counter of the lever."""
        from esmfold2_opt import report
        for k in ("graph_budget_eager_trunk", "graph_budget_eager_sampler", "graph_budget_eager_generic"):
            self.assertTrue(any(w in k for w in report.OPT_KEY_WORDS), k)
        opt = {"graph_budget_eager_trunk": 1, "sampler_fallback": 0, "unrelated_field": 3}
        kept = {k: v for k, v in opt.items() if any(w in str(k) for w in report.OPT_KEY_WORDS)}
        self.assertEqual(set(kept), {"graph_budget_eager_trunk", "sampler_fallback"})

    def test_pair_bias_row_launches_pass_the_key_word_filter(self):
        """A pair plane launched in row blocks past the fused pair-bias kernel's 32-bit offset bound (ef2_opt STATS pair_bias_row_launches)
        is a named count in the EXIT tally, never dropped by the key-word filter."""
        from esmfold2_opt import report
        opt = {"pair_bias_row_launches": 24, "unrelated_field": 3}
        kept = {k: v for k, v in opt.items() if any(w in str(k) for w in report.OPT_KEY_WORDS)}
        self.assertEqual(kept, {"pair_bias_row_launches": 24})


class TestMemoryModesBase(unittest.TestCase):
    def test_every_mode_runs_on_the_fused_base(self):
        """Every mode and line runs on the kit server's fused base: the preset's own model calls, no stock-side backend of its own."""
        every = list(modes.KIT_MODES.items()) + [(f"big:{ln}", km) for ln, km in modes.KIT_LINES["big"].items()]
        for name, km in every:
            self.assertIsNone(km.backend, name)


if __name__ == "__main__":
    unittest.main()
