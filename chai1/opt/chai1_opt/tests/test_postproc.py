"""The fold's host-side levers: rankcc (postproc.clash_scores_cc == chai_lab's clash census, integers and dtypes, when torch +
chai_lab are importable — skipped otherwise), tailasync (writers on one thread, call order kept, joined before StructureCandidates exists, a
writer's exception re-raised at the join), prefetch (the next uid's feature build served to the worker's later call by FASTA content + bound
arguments; misses / server modes / ESM / no worker stand aside by name). No GPU."""
import os
import sys
import tempfile
import threading
import time
import types
import unittest

from chai1_opt import featfast, modes, postproc, registry


class TestRegistration(unittest.TestCase):
    def test_names_probes_rows(self):
        self.assertEqual(postproc.LEVERS + featfast.LEVERS, modes.POSTPROC_LEVERS)
        self.assertEqual(registry.ATTR_PROBES["rankcc"], ("attr", "chai_lab.chai1", "rank", postproc.RANK_NAME))
        self.assertEqual(registry.ATTR_PROBES["tailasync"], ("attr", "chai_lab.chai1", "save_to_cif", postproc.CIF_NAME))
        self.assertEqual(registry.ATTR_PROBES["prefetch"], ("attr", "chai_lab.chai1", "make_all_atom_feature_context", featfast.FEAT_NAME))
        for row in ("exact", "fast", "big"):
            self.assertEqual(modes._ROWS[row].postproc, modes.POSTPROC_LEVERS)                 # exact-class: every kit row (R3 tiers)
            for n in modes.POSTPROC_LEVERS:                                                    # each switchable on its own (MODEL_OPT_LEVERS_OFF)
                self.assertNotIn(n, modes.without(modes._ROWS[row], (n,)).lever_names)
        self.assertEqual(postproc.EXPECTED_FALLBACKS, {"rankcc": ("batch", "cuda"), "tailasync": ()})
        self.assertEqual(registry.ATTR_PROBES["confmemo"], ("attr", "chai_lab.chai1", "load_chains_from_raw", featfast.CHAINS_NAME))
        self.assertEqual(featfast.EXPECTED_FALLBACKS, ("first_item", "miss", "server_mode", "esm_embeddings", "no_worker", "last_item"))


class TestTailAsync(unittest.TestCase):
    def setUp(self):
        self.C1 = types.ModuleType("chai_lab.chai1")
        self.log = []

        def save_to_cif(coords, output_batch, write_path, asym_entity_names, bfactors=None):
            time.sleep(0.02); self.log.append(("cif", write_path, threading.current_thread().name))
            if output_batch.get("boom"):
                raise RuntimeError("writer failed")

        def plot_msa(input_tokens, msa_tokens, out_fname):
            time.sleep(0.01); self.log.append(("plot", out_fname, threading.current_thread().name)); return out_fname

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class StructureCandidates:
            cif_paths: list

            def __post_init__(self):
                assert isinstance(self.cif_paths, list)
        self.C1.save_to_cif, self.C1.plot_msa, self.C1.StructureCandidates = save_to_cif, plot_msa, StructureCandidates
        self.addCleanup(postproc.reset_for_tests, self.C1)

    def test_writers_run_in_order_on_one_thread_and_are_joined_before_the_candidates_exist(self):
        postproc.install(("tailasync",), chai1_mod=self.C1)
        self.assertEqual(self.C1.save_to_cif.__name__, postproc.CIF_NAME); self.assertEqual(postproc.applied(), ("tailasync",))
        self.assertEqual(self.C1.plot_msa(1, 2, "msa.pdf"), "msa.pdf")                        # upstream's return value, at once
        for i in range(4):
            self.assertIsNone(self.C1.save_to_cif(coords=i, output_batch={}, write_path=f"p{i}.cif", asym_entity_names={}))
        self.assertLess(len(self.log), 5)                                                     # still being written …
        cand = self.C1.StructureCandidates(cif_paths=["p0.cif"])                              # … and all landed before the fold's value exists
        self.assertEqual([e[1] for e in self.log], ["msa.pdf", "p0.cif", "p1.cif", "p2.cif", "p3.cif"])
        self.assertTrue(all(e[2].startswith("chai1_tailasync") for e in self.log)); self.assertEqual(cand.cif_paths, ["p0.cif"])
        ev = postproc.evidence("tailasync"); self.assertEqual((ev["cif"], ev["plot"], ev["joins"], ev["errors"]), (4, 1, 1, 0))
        self.assertRegex(" ".join(postproc.tally_fields()), r"tailasync_cif=4 tailasync_plot=1 tailasync_join_s=\d")

    def test_a_writer_error_is_raised_at_the_join(self):
        postproc.install(("tailasync",), chai1_mod=self.C1)
        self.C1.save_to_cif(coords=0, output_batch={"boom": 1}, write_path="bad.cif", asym_entity_names={})
        self.C1.save_to_cif(coords=1, output_batch={}, write_path="good.cif", asym_entity_names={})
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            self.C1.StructureCandidates(cif_paths=["bad.cif"])
        self.assertEqual([e[1] for e in self.log], ["bad.cif", "good.cif"])                   # the later write still ran (as upstream's loop order would have)
        self.assertEqual(postproc.evidence("tailasync")["errors"], 1)


class TestConfMemo(unittest.TestCase):
    def test_one_tokenizer_per_process_when_the_caller_built_none(self):
        C1 = types.ModuleType("chai_lab.chai1"); seen = []
        C1.load_chains_from_raw = lambda inputs, identifier="test", tokenizer=None: seen.append(tokenizer) or ["chains", inputs]
        self.addCleanup(featfast.reset_for_tests, C1)
        featfast.install(("confmemo",), chai1_mod=C1)
        self.assertEqual(C1.load_chains_from_raw.__name__, featfast.CHAINS_NAME); self.assertEqual(featfast.applied(), ("confmemo",))
        cm = featfast._STATE["cm"]; cm.tokenizer = "TOK"                                     # stands in for AllAtomResidueTokenizer(RefConformerGenerator()) (chai_lab absent here)
        self.assertEqual(C1.load_chains_from_raw("in1"), ["chains", "in1"]); C1.load_chains_from_raw("in2", identifier="x"); C1.load_chains_from_raw("in3", tokenizer="OWN")
        self.assertEqual(seen, ["TOK", "TOK", "OWN"])                                          # the caller's own tokenizer passes through untouched
        self.assertEqual(featfast.census()["confmemo"], {"served": 2, "builds": 0, "build_s": 0.0, "passed_through": 1})
        self.assertIn("confmemo_served=2 confmemo_builds=0", " ".join(featfast.tally_fields()))


class TestPrefetch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.srcs = []
        for i, body in enumerate((">protein|A\nMKV\n", ">protein|A\nMKVL\n", ">protein|A\nMKVL\n")):   # items 1 and 2 have the same bytes (the ladder's timed copies do)
            p = os.path.join(self.d, f"in{i}.fasta"); open(p, "w").write(body); self.srcs.append(p)
        self.C1 = types.ModuleType("chai_lab.chai1"); self.calls = []

        def make_all_atom_feature_context(fasta_file, *, output_dir, use_esm_embeddings=True, use_msa_server=False, msa_server_url="x",
                                          msa_directory=None, constraint_path=None, use_templates_server=False, templates_path=None, esm_device=None):
            self.calls.append((os.fspath(fasta_file), threading.current_thread().name)); time.sleep(0.05)
            return ("ctx", open(fasta_file).read(), os.fspath(msa_directory))
        self.C1.make_all_atom_feature_context = make_all_atom_feature_context
        self.cp = types.ModuleType("chai_proto")
        self.cp.input_spec = lambda uid: {"key": os.path.basename(uid)[:-6], "fasta_src": uid[len("fasta:"):] if uid.startswith("fasta:") else uid}
        self.saved = (sys.modules.get("chai_proto"), list(sys.argv))
        sys.modules["chai_proto"] = self.cp
        sys.argv = ["chai_worker.py", ",".join("fasta:" + s for s in self.srcs), "42", "/out", "--msa_dir", "/msas"]
        self.addCleanup(self._restore)

    def _restore(self):
        featfast.reset_for_tests(self.C1)
        if self.saved[0] is None: sys.modules.pop("chai_proto", None)
        else: sys.modules["chai_proto"] = self.saved[0]
        sys.argv = self.saved[1]

    def _worker_call(self, i, **kw):
        w = os.path.join(self.d, f"work{i}"); os.makedirs(w, exist_ok=True)
        f = os.path.join(w, f"in{i}.fasta"); open(f, "w").write(open(self.srcs[i]).read())       # the worker's own copy (cp.write_fasta)
        base = dict(use_esm_embeddings=False, msa_directory="/msas")
        base.update(kw)
        return self.C1.make_all_atom_feature_context(fasta_file=f, output_dir=os.path.join(w, "out"), **base)

    def test_the_next_items_build_is_served_from_the_helper_thread(self):
        featfast.install(("prefetch",), chai1_mod=self.C1)
        self.assertEqual(self.C1.make_all_atom_feature_context.__name__, featfast.FEAT_NAME)
        r0 = self._worker_call(0); time.sleep(0.2)                                            # item 0 inline (first_item); item 1 kicked
        r1 = self._worker_call(1); time.sleep(0.2)                                            # served; item 2 kicked (same bytes as item 1: keyed by content, positioned by the uid cursor)
        r2 = self._worker_call(2)                                                             # served; last item: nothing kicked
        self.assertEqual([r0[1], r1[1], r2[1]], [">protein|A\nMKV\n", ">protein|A\nMKVL\n", ">protein|A\nMKVL\n"])
        threads = [t for _, t in self.calls]
        self.assertEqual(threads[0], threading.current_thread().name); self.assertTrue(all(t.startswith("chai1_prefetch") for t in threads[1:]), self.calls)
        self.assertEqual(len(self.calls), 3)                                                  # three builds in all: no duplicate work
        ev = featfast.evidence("prefetch")
        self.assertEqual((ev["served"], ev["kicked"], ev["fallback_by"], ev["aside_by"]), (2, 2, {"first_item": 1}, {"last_item": 1}))
        self.assertIsNone(featfast.verdict())

    def test_a_call_that_matches_nothing_is_computed_inline_and_counted(self):
        featfast.install(("prefetch",), chai1_mod=self.C1)
        self._worker_call(0); time.sleep(0.2)
        r1 = self._worker_call(1, constraint_path="/x.csv")                                   # other arguments than the prefetched build: a miss, inline
        self.assertEqual(r1[1], ">protein|A\nMKVL\n"); self.assertEqual(featfast.evidence("prefetch")["fallback_by"], {"first_item": 1, "miss": 1})

    def test_stands_aside_by_name(self):
        featfast.install(("prefetch",), chai1_mod=self.C1)
        self._worker_call(0, use_esm_embeddings=True); self._worker_call(1, use_msa_server=True)
        self.assertEqual(featfast.evidence("prefetch")["aside_by"], {"esm_embeddings": 1, "server_mode": 1}); self.assertEqual(featfast.evidence("prefetch")["kicked"], 0)
        sys.argv = ["python"]                                                                 # not the kit worker's command line
        self._worker_call(2)
        self.assertEqual(featfast.evidence("prefetch")["aside_by"].get("no_worker"), 1)


class TestRankCC(unittest.TestCase):
    def test_clash_census_equals_upstreams_integers_and_dtypes(self):
        try:                                                                                  # torch + the REAL chai_lab (other test modules install stub chai_lab modules)
            import torch
            import chai_lab.ranking.clashes as clashes
            from chai_lab.utils.tensor_utils import cdist
            assert callable(getattr(clashes, "get_scores", None)) and callable(getattr(torch, "randn", None)) and getattr(clashes, "__file__", None)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"torch + the real chai_lab are needed for the clash-census equivalence ({type(e).__name__})")
        g = torch.Generator().manual_seed(0)
        for trial in range(6):
            a = 700 + 37 * trial; n_tok = 90
            exists = torch.rand(1, a, generator=g) < 0.55
            coords = torch.randn(1, a, 3, generator=g) * (2.0 + trial)                       # dense enough for many sub-1.1 A pairs
            tok_asym = torch.randint(1, 4, (1, n_tok), generator=g).sort(dim=-1).values
            atom_tok = torch.randint(0, n_tok, (1, a), generator=g).sort(dim=-1).values
            atom_asym = torch.gather(tok_asym, -1, atom_tok); atom_ent = torch.zeros(1, a, dtype=torch.int64)
            kw = dict(atom_coords=coords, atom_mask=exists, atom_asym_id=atom_asym, atom_entity_type=atom_ent, clash_threshold=1.1, max_clashes=100, max_clash_ratio=0.5)
            ref = clashes.get_scores(**kw); got = postproc.clash_scores_cc(**kw)
            self.assertTrue(postproc._clash_equal(got, ref), (trial, ref.chain_chain_clashes, got.chain_chain_clashes))
            self.assertGreater(int(ref.total_clashes), 0)
        # the premise: per-pair arithmetic — the existing rows' bits do not depend on the padded rows
        x = torch.randn(1, 500, 3, generator=g); idx = torch.randperm(500, generator=g)[:200].sort().values
        self.assertTrue(torch.equal(cdist(x)[0][idx][:, idx], cdist(x[:, idx])[0]))


if __name__ == "__main__":
    unittest.main()
