"""The stock options passed through: the option table read from the pinned sources (names, defaults, definition lines), ``parse``
(stock options only: a lever, a non-stock option or a package-supplied option refused by name), what mode exact cannot serve
(``worker_refuses``: refused by name with the mechanism, exit 3), the two argv maps (stock: verbatim; kit worker: implied / inert / kit-applied options dropped, upstream's defaults handed for the knobs the
worker defaults differently), warm's constant against the worker, and the recipe lines."""
import os
import re
import unittest

from proteinmpnn_opt import det, modes, registry, settings, stack, warm

STOCK = stack.stock_dir()


def _defaults(path):
    """upstream's argparse defaults, read independently of settings._argparse_table (a second reading of the same source)."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    out = {}
    for m in re.finditer(r'add_argument\(\s*"--(\w+)"(.*?)\)\s*$', src, re.M | re.S):
        d = re.search(r"default=([^,\)]+)", m.group(2))
        if d:
            out[m.group(1)] = d.group(1).strip().strip('"').strip("'")
    return out


class TestStockTable(unittest.TestCase):
    def test_defaults_are_upstreams(self):
        d = _defaults(os.path.join(STOCK, "src", "protein_mpnn_run.py"))
        t = settings.stock_table("soluble")
        for k in ("seed", "num_seq_per_target", "batch_size", "sampling_temp", "backbone_noise", "save_score", "save_probs", "omit_AAs", "model_name", "max_length"):
            self.assertEqual(t["--" + k][0], d[k], k)
        self.assertEqual((t["--seed"][0], t["--save_score"][0], t["--save_probs"][0], t["--num_seq_per_target"][0], t["--batch_size"][0]), ("0", "0", "0", "1", "1"))
        src = open(os.path.join(STOCK, "src", "protein_mpnn_run.py"), encoding="utf-8").read().split("\n")
        for f in ("--seed", "--sampling_temp", "--omit_AAs"):
            self.assertIn(f'add_argument("{f}"', src[t[f][1] - 1], f)                 # the definition line the table cites
        self.assertEqual(settings.stock_default("vanilla", "--model_name"), "v_48_020")


    def test_options_are_read_from_the_carried_sources(self):
        so = settings.stock_options("soluble")
        for f in ("--seed", "--jsonl_path", "--max_length", "--ca_only", "--suppress_print", "--tied_positions_jsonl"):
            self.assertIn(f, so, f)
        for f in ("--x_all", "--hybrid_gemm", "--bb_batch", "--mode", "--dtype"):
            self.assertNotIn(f, so, f)
        wo = settings.worker_options()
        for f in ("--x_all", "--max_length", "--omit_AAs", "--fixed_positions_jsonl", "--hybrid_gemm", "--seed", "--num_seq_per_target", "--batch_size", "--sampling_temp", "--model_name"):
            self.assertIn(f, wo, f)
        self.assertNotIn("--ca_only", wo); self.assertNotIn("--suppress_print", wo)
        for f in settings.WORKER_KNOBS:
            self.assertIn(f, wo, f); self.assertIn(f, so, f)


class TestParse(unittest.TestCase):
    def _refused(self, toks, variant, *words):
        with self.assertRaises(settings.SettingsError) as cm:
            settings.parse(toks, variant)
        for w in words:
            self.assertIn(w, str(cm.exception), str(cm.exception))

    def test_tokens_are_kept_verbatim_in_order(self):
        self.assertEqual(settings.parse(["--seed", "37", "--num_seq_per_target=8", "--ca_only", "--sampling_temp", "0.1 0.2"], "soluble"),
                         [("--seed", "37"), ("--num_seq_per_target", "8"), ("--ca_only", None), ("--sampling_temp", "0.1 0.2")])
        self.assertEqual(settings.parse([], "soluble"), []); self.assertEqual(settings.parse(None, "vanilla"), [])
        self.assertEqual(settings.stock_argv(settings.parse(["--sampling_temp", "0.2 0.3", "--ca_only", "--path_to_fasta", "x.fa"], "vanilla")),
                         ["--sampling_temp", "0.2 0.3", "--ca_only", "--path_to_fasta", "x.fa"])

    def test_levers_are_refused(self):
        self._refused(["--hybrid_gemm"], "soluble", "--hybrid_gemm", "kit lever")
        self._refused(["--chunk_gemm"], "vanilla", "--chunk_gemm", "kit lever")
        self._refused(["--max_length", "500", "--bb_batch", "32"], "soluble", "--bb_batch", "kit lever")
        for f in registry.LEVERS:                                          # every registered switch, both weight sets
            if f.startswith("--"):
                for v in ("soluble", "vanilla"):
                    with self.assertRaises(settings.SettingsError):
                        settings.parse([f], v)

    def test_non_stock_and_package_supplied_options_are_refused(self):
        self._refused(["--frobnicate", "1"], "soluble", "--frobnicate", "not an option of the stock command line")
        self._refused(["--dtype", "bf16"], "soluble", "--dtype", "not an option")                      # neither the stock CLI's nor a lever
        self._refused(["--num_seq", "8"], "soluble", "--num_seq", "not an option")                      # upstream's spelling is --num_seq_per_target
        for f in ("--jsonl_path", "--chain_id_jsonl", "--out_folder", "--fixed_positions_jsonl", "--pdb_path"):   # design's own arguments (cli.cmd_design reads them once): met among the stock options = given twice
            self._refused([f, "1"], "soluble", f, "given once")
        # upstream's own weights selectors pass through on every route (the kit renders its variant's weights only for the role a pass leaves unset)
        self.assertEqual(settings.parse(["--path_to_model_weights", "/w", "--use_soluble_model"], "vanilla"), [("--path_to_model_weights", "/w"), ("--use_soluble_model", None)])
        self.assertTrue(settings.weights_given(settings.parse(["--use_soluble_model"], "vanilla")))
        self.assertFalse(settings.weights_given(settings.parse(["--model_name", "v_48_030"], "vanilla")))
        self.assertEqual(settings.base_weights_dir(settings.parse([], "vanilla"), "vanilla", "/ck"), "/ck/vanilla_model_weights")
        self.assertEqual(settings.base_weights_dir(settings.parse(["--use_soluble_model"], "vanilla"), "vanilla", "/ck"), "/ck/soluble_model_weights")   # protein_mpnn_run.py:41-47
        self.assertEqual(settings.base_weights_dir(settings.parse(["--path_to_model_weights", "/w", "--use_soluble_model"], "soluble"), "soluble", "/ck"), "/w")   # protein_mpnn_run.py:35-38
        self._refused(["500"], "soluble", "is not an option")
        for f in ("--seed", "--num_seq_per_target", "--batch_size", "--sampling_temp", "--backbone_noise", "--save_score", "--save_probs", "--omit_AAs", "--model_name"):
            self.assertEqual(settings.parse([f, "1"], "soluble"), [(f, "1")])                          # the stock command line's own options pass


SINGLE_SEQ = ["--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--seed", "37", "--backbone_noise", "0.0",
              "--save_score", "1", "--save_probs", "1", "--omit_AAs", "X", "--model_name", "v_48_020"]   # one seeded sequence per backbone, every output written


class TestWorkerRefuses(unittest.TestCase):
    """Kit route: what mode exact cannot serve is refused by name with its mechanism (cli: NOT ACTIVE, exit 3); everything else of the design pass is served."""

    def _refused(self, toks, variant="soluble"):
        return settings.worker_refuses(settings.parse(toks, variant), variant)

    def _tokens(self, toks, variant="soluble"):
        return [t for t, _ in self._refused(toks, variant)]

    def test_the_design_pass_is_served(self):
        ok = SINGLE_SEQ
        self.assertEqual(self._refused(ok), [])
        self.assertEqual(self._refused([]), [])                                          # no option given: upstream's defaults, --seed 0 included (the worker draws the seed as upstream draws it)
        for more in (["--seed", "0"], ["--num_seq_per_target", "8"], ["--num_seq_per_target", "8", "--batch_size", "4"], ["--sampling_temp", "0.1 0.2 0.3"],
                     ["--suppress_print", "1"], ["--max_length", "500"], ["--omit_AA_jsonl", "o.jsonl", "--bias_AA_jsonl", "b.jsonl", "--bias_by_res_jsonl", "r.jsonl"],
                     ["--pssm_jsonl", "p.jsonl", "--pssm_multi", "0.3", "--pssm_threshold", "0.1", "--pssm_log_odds_flag", "1", "--pssm_bias_flag", "1"],
                     ["--pdb_path_chains", "A B"], ["--path_to_model_weights", ""], ["--use_soluble_model", "--path_to_model_weights", "/w"], ["--save_score", "0", "--save_probs", "0"],
                     ["--score_only", "0", "--conditional_probs_only", "0", "--unconditional_probs_only", "0", "--tied_positions_jsonl", ""],   # the pass selectors at upstream's defaults: the plain design pass
                     ["--path_to_fasta", "x.fa"], ["--conditional_probs_only_backbone", "1"], ["--backbone_noise", "-0.5"]):                 # read only by passes exact refuses: inert alone; noise below 0: none added upstream
            self.assertEqual(self._refused(["--seed", "5"] + more), [], more)
            self.assertEqual(self._refused(["--seed", "5"] + more, "vanilla"), [], more)

    def test_what_the_worker_cannot_serve_is_refused_with_the_mechanism(self):
        ok = SINGLE_SEQ
        for more, tokens, word in ((["--ca_only"], ["--ca_only"], "CA-only"), (["--score_only", "1", "--path_to_fasta", "x.fa"], ["--score_only=1"], "scoring-only"),
                                   (["--conditional_probs_only", "1", "--conditional_probs_only_backbone", "1"], ["--conditional_probs_only=1"], "conditional-probabilities"),
                                   (["--unconditional_probs_only", "1"], ["--unconditional_probs_only=1"], "unconditional-probabilities"),
                                   (["--tied_positions_jsonl", "t.jsonl"], ["--tied_positions_jsonl=t.jsonl"], "tied decoding"),
                                   (["--backbone_noise", "0.1"], ["--backbone_noise=0.1"], "randn_like"),
                                   (["--num_seq_per_target", "2", "--batch_size", "4"], ["--num_seq_per_target=2<--batch_size=4"], "0 batches")):
            got = self._refused([t for t in ok if t not in ("--backbone_noise", "0.0", "--num_seq_per_target", "--batch_size", "1")] + more)
            self.assertEqual([t for t, _ in got], tokens, more)
            self.assertTrue(all(word in why for _, why in got if _ == tokens[0]), got)          # the mechanism sentence names what changes inside the re-stated region
        for flag in settings.REFUSED_BY_WORKER:
            self.assertIn(flag, settings.stock_options("soluble"))                            # every refused name is a stock option upstream defines
            self.assertNotIn(flag, settings.worker_options())                                 # and not one the worker takes
        line = settings.refusal_reason("exact", [("--ca_only", "why A"), ("--x=1", "why B")])
        self.assertEqual(line, "cannot serve --ca_only (why A); --x=1 (why B) under mode exact — refused by name, nothing launched (exit 3): --mode off runs the stock command line with these options")

    def test_every_stock_option_has_a_place(self):
        """Each option protein_mpnn_run.py defines is the worker's own, implied, inert, applied by the kit, a weights selector, managed by design, or refused by
        name with a mechanism — none is silently dropped."""
        placed = set(settings.worker_options()) | set(settings.WORKER_IMPLIED) | set(settings.WORKER_INERT) | set(settings.KIT_HANDLED) | set(settings.BASE_WEIGHTS_FLAGS) | set(settings.MANAGED_FLAGS) | set(settings.REFUSED_BY_WORKER)
        self.assertEqual(sorted(set(settings.stock_options("vanilla")) - placed), [])

    def test_kit_argv_carries_only_the_workers_own_options(self):
        argv = settings.kit_argv(settings.parse(["--seed", "5", "--score_only", "0", "--tied_positions_jsonl", "", "--path_to_fasta", "x.fa", "--suppress_print", "1", "--pdb_path_chains", "A",
                                                 "--backbone_noise", "0.0", "--omit_AA_jsonl", "o.jsonl"], "vanilla"), "vanilla")
        self.assertEqual(argv[:4], ["--seed", "5", "--omit_AA_jsonl", "o.jsonl"])
        for dropped in ("--score_only", "--tied_positions_jsonl", "--path_to_fasta", "--suppress_print", "--pdb_path_chains", "--backbone_noise"):
            self.assertNotIn(dropped, argv)

    def test_designs_per_target_is_upstreams_whole_batches_rule(self):
        self.assertEqual(settings.designs_per_target(settings.parse(["--num_seq_per_target", "6", "--batch_size", "4"], "vanilla"), "vanilla"), {"requested": 6, "batch_size": 4, "produced": 4})
        self.assertEqual(settings.designs_per_target(settings.parse([], "vanilla"), "vanilla"), {"requested": 1, "batch_size": 1, "produced": 1})   # upstream's defaults
        self.assertEqual(settings.designs_per_target(settings.parse(["--num_seq_per_target", "16", "--batch_size", "8"], "soluble"), "soluble")["produced"], 16)

    def test_pdb_path_chains_as_upstream_reads_it(self):
        self.assertEqual(settings.pdb_path_chains(settings.parse(["--pdb_path_chains", "B A"], "vanilla")), ["B", "A"])
        self.assertIsNone(settings.pdb_path_chains(settings.parse(["--pdb_path_chains", ""], "vanilla"))); self.assertIsNone(settings.pdb_path_chains(settings.parse([], "vanilla")))

    def test_a_malformed_value_is_usage(self):
        for bad in (["--seed", "x"], ["--sampling_temp", "0.1,0.2"], ["--batch_size", "1.5"]):
            with self.assertRaises(settings.SettingsError):
                self._refused(bad)


class TestArgumentMap(unittest.TestCase):
    def test_stock_argv_is_verbatim(self):
        for v, toks in (("soluble", SINGLE_SEQ), ("soluble", ["--ca_only", "--seed=3"]), ("vanilla", ["--seed", "111", "--save_score", "1"]), ("vanilla", [])):
            given = settings.parse(toks, v)
            self.assertEqual(settings.stock_argv(given), [t for tok in toks for t in tok.split("=", 1)] if any("=" in t for t in toks) else toks)

    def test_kit_argv_drops_the_implied_options_and_hands_upstreams_defaults(self):
        self.assertEqual(settings.kit_argv(settings.parse(SINGLE_SEQ, "soluble"), "soluble"),
                         ["--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--seed", "37", "--save_score", "1", "--save_probs", "1", "--omit_AAs", "X", "--model_name", "v_48_020"])
        # the least a kit pass names: the knobs not given follow, with upstream's defaults read from the carried source
        self.assertEqual(settings.kit_argv(settings.parse(["--seed", "5", "--save_score", "1", "--save_probs", "1"], "vanilla"), "vanilla"),
                         ["--seed", "5", "--save_score", "1", "--save_probs", "1", "--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--omit_AAs", "X", "--model_name", "v_48_020"])   # the write gates pass verbatim
        self.assertEqual(settings.kit_argv(settings.parse(["--save_probs", "1", "--seed", "5", "--max_length", "500", "--save_score", "0", "--model_name", "m1"], "soluble"), "soluble"),
                         ["--save_probs", "1", "--seed", "5", "--max_length", "500", "--save_score", "0", "--model_name", "m1", "--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--omit_AAs", "X"])
        self.assertEqual(settings.kit_argv(settings.parse(["--seed", "5", "--backbone_noise", "0.0", "--use_soluble_model", "--path_to_model_weights", "/w"], "vanilla"), "vanilla"),
                         ["--seed", "5", "--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--omit_AAs", "X", "--model_name", "v_48_020"])   # the implied noise and the weights selectors (rendered by kit_run, base_weights_dir) are not copied
        self.assertEqual(settings.model_name(settings.parse(["--seed", "5"], "soluble"), "soluble"), "v_48_020")
        self.assertEqual(settings.model_name(settings.parse(["--model_name", "pmft_v1"], "vanilla"), "vanilla"), "pmft_v1")

    def test_warms_constant_is_served_by_the_worker_and_names_every_knob(self):
        base = settings.parse(warm.WARM_STOCK_ARGS, "soluble")
        self.assertEqual(settings.worker_refuses(base, "soluble"), [])
        self.assertEqual(warm.WARM_STOCK_ARGS, SINGLE_SEQ)
        self.assertEqual({f for f, _ in base} - set(settings.WORKER_IMPLIED) - {"--save_score", "--save_probs"}, set(settings.WORKER_KNOBS))   # the write gates: the worker's own options, passed verbatim
        self.assertNotEqual(dict(base)["--seed"], "0")


class TestReadmeRowsAreStockOptionsPlusLevers(unittest.TestCase):
    """The kit README row's non-lever arguments are stock options (SETTINGS_FLAGS) upstream defines, so a pass can spell the row."""

    def test_rows(self):
        for kind, variant in (("exact", "soluble"), ("exact", "vanilla")):
            line = modes.readme_row(stack.kit_home(), modes.EXACT_ROW)
            opts = settings.stock_options(variant)
            named = [tok for tok in line.split() if tok.startswith("--") and tok in modes.SETTINGS_FLAGS]
            self.assertTrue(named, line)
            for tok in named:
                self.assertIn(tok, opts, (kind, tok))


class TestRecipe(unittest.TestCase):
    def test_lines(self):
        self.assertIn("explicit non-zero --seed", det.recipe()); self.assertIn("--backbone_noise 0.0", det.recipe())
        self.assertTrue(det.describe().startswith("det=0 "))


if __name__ == "__main__":
    unittest.main()
