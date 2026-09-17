"""The prediction loop (stock_fold.fold_items) under a stub upstream: after every fold is written or staged its result and features are
released and the CUDA cache is emptied — between folds, never inside the stock call — and a pass that fails on a later complex still
delivers the completed ones as outputs, with the INCOMPLETE line naming the failure (stock route and kit route)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from esmfold2_opt import outputs, stock_fold
from esmfold2_opt.tests import _stubs

ITEMS = [{"id": cid, "sequences": [{"type": "protein", "id": "A", "sequence": seq, "msa": None}, {"type": "protein", "id": "B", "sequence": "MKV", "msa": None}]}
         for cid, seq in (("p1", "MKVLAG"), ("p2", "MKVLAGW"), ("p3", "MKVLAGWA"))]

RELEASE_SCRIPT = textwrap.dedent('''
    import gc, json, os, sys, weakref
    import torch
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esmfold2_opt import outputs, settings, stack, stock_fold
    events = []
    state = {"in_fold": False, "prev": None, "prev_alive_at_next_fold": []}
    torch.cuda.is_available = lambda: True
    def empty_cache():
        events.append("empty_cache" + (":INSIDE_FOLD" if state["in_fold"] else ""))
    torch.cuda.empty_cache = empty_cache
    orig_fold = ESMFold2InputBuilder.fold
    def fold(self, model, spi, **kw):
        state["in_fold"] = True
        if state["prev"] is not None:
            gc.collect()
            state["prev_alive_at_next_fold"].append(state["prev"]() is not None)
        events.append("fold:" + kw["complex_id"])
        res = orig_fold(self, model, spi, **kw)
        state["prev"] = weakref.ref(res[0] if isinstance(res, list) else res)
        state["in_fold"] = False
        return res
    ESMFold2InputBuilder.fold = fold
    items = json.load(open(sys.argv[1])); out = sys.argv[2]
    st = settings.resolve(pins=stack.pins(), num_loops=10, num_sampling_steps=68, num_diffusion_samples=1, msa_max_depth=2048, remove_insertions=True, max_sequences=2048)
    def writer(item, seed, res, depths, feats, wall):
        events.append("write:" + item["id"])
        return [{"complex_id": item["id"]}]
    n_kit = stock_fold.fold_items(object(), ESMFold2InputBuilder(), items, st, "fast", out, seeds=[0], writer=writer, device="cpu", log=lambda s: None)
    events.append("--")
    stage = os.path.join(out, "staged")
    orig_stage = outputs.stage_result
    def stage_result(*a, **k):
        events.append("stage:" + a[1]["id"]); return orig_stage(*a, **k)
    outputs.stage_result = stage_result
    n_stock = stock_fold.fold_items(object(), ESMFold2InputBuilder(), items, st, "fast", out, seeds=[0], stage_dir=stage, device="cpu", log=lambda s: None)
    print(json.dumps({"events": events, "n": [n_kit, n_stock], "prev_alive": state["prev_alive_at_next_fold"]}))
''')

FINALISE_SCRIPT = textwrap.dedent('''
    import json, os, sys
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esmfold2_opt import outputs, settings, stack, stock_fold
    items = json.load(open(sys.argv[1])); out = sys.argv[2]; stage = os.path.join(out, "staged")
    items[1]["num_diffusion_samples"] = 2
    outcome = stock_fold.Outcome(items[:2], path=os.path.join(stage, outputs.PROGRESS_NAME))
    stock_fold.fold_items(object(), ESMFold2InputBuilder(), items[:2], settings.resolve(pins=stack.pins(), num_loops=10, num_sampling_steps=68, num_diffusion_samples=1, msa_max_depth=2048, remove_insertions=True, max_sequences=2048), "fast", out, seeds=[0], stage_dir=stage, device="cpu",
                          log=lambda s: None, outcome=outcome)
    staged = sorted(os.listdir(stage))
    for fn in os.listdir(stage):                                   # the second sample of p2 never finished staging
        if fn.startswith("p2__fast__s0_x1"):
            os.remove(os.path.join(stage, fn))
    rows = outputs.finalise_staged(outputs.kit_server(), stage, out, "pred")
    print(json.dumps({"staged": staged, "rows": [(r["complex_id"], r["seed"], r["sample_index"]) for r in rows], "stage_gone": not os.path.exists(stage),
                      "record": stock_fold.read_outcome(os.path.join(stage, outputs.PROGRESS_NAME)), "cif": sorted(os.listdir(os.path.join(out, "cif_all")))}))
''')


class TestFoldLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_kit = _stubs.require_kit()
        cls.pins = _stubs.require_pins()
        cls.tmp = tempfile.mkdtemp(prefix="ef2_fold_loop_")
        cls.site = _stubs.write_stub_upstream(cls.tmp, cls.pins)
        cls.kit = _stubs.write_stub_kit(cls.tmp, cls.real_kit)
        cls.tree = _stubs.write_stub_tree(cls.tmp, cls.pins)
        cls.items = os.path.join(cls.tmp, "items.json")
        with open(cls.items, "w") as fh:
            json.dump(ITEMS, fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, args, **extra):
        env = _stubs.env_for_stub(self.site, self.kit, self.tree, **extra)
        return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)

    def _json(self, r):
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def _pred(self, mode, out, *argv, **extra):
        return self._run(["-m", "esmfold2_opt", "pred", "--mode", mode, "--variant", "fast", "--input", self.items, "--out_dir", out, "--device", "cpu", "--seeds", "0",
                          "--num_loops", "10", "--num_sampling_steps", "68", "--num_diffusion_samples", "1", "--msa_max_depth", "2048", "--remove_insertions", "true", "--max_sequences", "2048", *argv], **extra)


    def test_release_runs_between_folds_and_never_inside_the_stock_call(self):
        out = self._json(self._run(["-c", RELEASE_SCRIPT, self.items, os.path.join(self.tmp, "release")]))
        kit_route, stock_route = "|".join(out["events"]).split("|--|")
        self.assertEqual(kit_route.split("|"), ["fold:p1", "write:p1", "empty_cache", "fold:p2", "write:p2", "empty_cache", "fold:p3", "write:p3", "empty_cache"])
        self.assertEqual(stock_route.split("|"), ["fold:p1", "stage:p1", "empty_cache", "fold:p2", "stage:p2", "empty_cache", "fold:p3", "stage:p3", "empty_cache"])
        self.assertNotIn("INSIDE_FOLD", "|".join(out["events"]))
        self.assertEqual(out["prev_alive"], [False] * 5, "the previous fold's result must be released before the next stock call")
        self.assertEqual(out["n"], [3, 3])

    def test_empty_cuda_cache_is_a_no_op_without_cuda(self):
        torch = sys.modules.get("torch")
        try:
            sys.modules.pop("torch", None)
            self.assertFalse(stock_fold.empty_cuda_cache())
        finally:
            if torch is not None:
                sys.modules["torch"] = torch

    def test_stock_pass_partial_after_a_failed_fold(self):
        out = os.path.join(self.tmp, "stock_partial")
        r = self._pred("off", out, STUB_FOLD_FAIL_ON="p3")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("[esmfold2-opt] pred INCOMPLETE: 2/3 complexes complete; failed on p3 s0: RuntimeError: CUDA out of memory (stub) on p3", r.stderr)
        cifs = sorted(f for f in os.listdir(os.path.join(out, "cif_all")) if f.endswith(".cif"))
        self.assertEqual(cifs, ["p1__fast__s0_x0.cif", "p2__fast__s0_x0.cif"], "the completed complexes are outputs")
        rows = [json.loads(l) for l in open(os.path.join(out, "pred_rows.jsonl"))]
        self.assertEqual([(x["complex_id"], x["seed"]) for x in rows], [("p1", 0), ("p2", 0)])
        self.assertFalse(os.path.exists(os.path.join(out, outputs.STAGE_DIRNAME)))
        self.assertIn("[esmfold2-opt stock] pred INCOMPLETE: 2/3 complexes complete; failed on p3 s0: RuntimeError: CUDA out of memory (stub) on p3", r.stderr)   # the subprocess's own line too
        self.assertIn("[esmfold2-opt stock] ENV-CLEAN ok", r.stderr, "the environment proof holds on a failed pass too")
        self.assertEqual(sorted(os.listdir(out)), ["cif_all", "pred_rows.jsonl"]); self.assertEqual(len(os.listdir(os.path.join(out, "cif_all"))), 4)   # 2 cif + 2 npz, no side file
        # the bytes of the completed complexes are those of a complete pass
        full = os.path.join(self.tmp, "stock_complete")
        r = self._pred("off", full)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("[esmfold2-opt stock] DONE predictions=3 items=3", r.stderr); self.assertNotIn("INCOMPLETE", r.stderr)
        partial_listing = outputs.listing(out); full_listing = outputs.listing(full)
        self.assertEqual(set(partial_listing), {k for k in full_listing if "p3" not in k})
        self.assertEqual(partial_listing, {k: v for k, v in full_listing.items() if k in partial_listing})

    def test_kit_pass_partial_after_a_failed_fold(self):
        out = os.path.join(self.tmp, "kit_partial")
        r = self._pred("fast", out, STUB_FOLD_FAIL_ON="p2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("[esmfold2-opt] pred FAILED: RuntimeError: CUDA out of memory (stub) on p2", r.stderr)
        self.assertIn("[esmfold2-opt] pred INCOMPLETE: 1/3 complexes complete; failed on p2 s0: RuntimeError: CUDA out of memory (stub) on p2", r.stderr)
        self.assertEqual(r.returncode, 1)                                                      # incomplete: EXIT_FAIL (1/3 on the line) — a different condition from partial (3)
        self.assertEqual(sorted(f for f in os.listdir(os.path.join(out, "cif_all")) if f.endswith(".cif")), ["p1__fast__s0_x0.cif"])
        rows = [json.loads(l) for l in open(os.path.join(out, "pred_rows.jsonl"))]
        self.assertEqual([(x["complex_id"], x["seed"]) for x in rows], [("p1", 0)]); self.assertEqual(sorted(os.listdir(out)), ["cif_all", "pred_rows.jsonl"])

    def test_kit_mode_refuses_by_name_before_folding_when_a_lever_cannot_run(self):
        """A mode is all of its levers on a GPU class: when the kit's records show a lever of the set not applied (here af — a word of the fast set on EVERY GPU class and never of exact's; the stub's
        atom record reports af not set) pred prints the APPLIED line with the fallback, one `NOT ACTIVE: partial activation — levers=t9 (…)` line
        and exits 3 BEFORE the first fold — no ready line, no fold, no output written; never a run under the mode's name with a subset."""
        root = tempfile.mkdtemp(prefix="ef2_refuse_", dir=self.tmp)
        kit = _stubs.write_stub_kit(root, self.real_kit, unapplied=("af",))
        out = os.path.join(self.tmp, "kit_refused")
        r = self._pred("fast", out, ESMFOLD2_OPT_KIT=kit)
        self.assertEqual(r.returncode, 3, r.stderr[-3000:])
        self.assertIn(" fallbacks=af ", r.stderr)
        self.assertIn("[esmfold2-opt] NOT ACTIVE: partial activation — levers=af (af: kit record atom:af is not set after configure()): a mode is all of its levers on a GPU class, refused by name; exit 3", r.stderr)
        self.assertIn("[esmfold2-opt] LEVER name=af state=skipped reason=fallback:kit_record_atom:af_is_not_set_after_configure() ", r.stderr)
        self.assertNotIn("[esmfold2-opt] ready variant=", r.stderr); self.assertNotIn("[esmfold2-opt] fold ", r.stderr)
        self.assertFalse(os.path.isdir(os.path.join(out, "cif_all")), os.listdir(out) if os.path.isdir(out) else None)
        r = self._pred("exact", os.path.join(self.tmp, "kit_exact_ok"), ESMFOLD2_OPT_KIT=kit)      # exact's set has no af: the same box runs exact complete
        self.assertEqual(r.returncode, 0, r.stderr[-3000:]); self.assertIn(" fallbacks=none ", r.stderr)

    def test_complete_kit_pass_is_marked_complete(self):
        out = os.path.join(self.tmp, "kit_complete")
        r = self._pred("fast", out)                                                      # the stub records a complete application of the mode's set
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn(" fallbacks=none ", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertNotIn("pred INCOMPLETE", r.stderr)
        self.assertEqual(sorted(f for f in os.listdir(os.path.join(out, "cif_all")) if f.endswith(".cif")), ["p1__fast__s0_x0.cif", "p2__fast__s0_x0.cif", "p3__fast__s0_x0.cif"])
        self.assertEqual(sorted(os.listdir(out)), ["cif_all", "pred_rows.jsonl"])

    def test_finalise_skips_an_unfinished_group_and_the_pass_record(self):
        out = self._json(self._run(["-c", FINALISE_SCRIPT, self.items, os.path.join(self.tmp, "finalise")]))
        self.assertIn(outputs.PROGRESS_NAME, out["staged"])
        self.assertEqual(out["rows"], [["p1", 0, 0]], "p2's fold has one of its two samples staged: left out")
        self.assertEqual(out["cif"], ["p1__fast__s0_x0.cif", "p1__fast__s0_x0_pae.npz"])
        self.assertTrue(out["stage_gone"]); self.assertIsNone(out["record"])


class TestPassSummary(unittest.TestCase):
    def test_outcome_record_round_trip(self):
        tmp = tempfile.mkdtemp(prefix="ef2_outcome_")
        try:
            path = os.path.join(tmp, "staged", outputs.PROGRESS_NAME)
            o = stock_fold.Outcome(ITEMS, path=path)
            self.assertEqual(stock_fold.read_outcome(path), {"n_items": 3, "items_complete": [], "current": None, "item_failed": None, "n_predictions": 0})
            o.start("p1"); o.start("p1", 0); o.fold_done(1); o.item_done("p1"); o.start("p2"); o.start("p2", 0)
            self.assertEqual(stock_fold.read_outcome(path)["current"], {"id": "p2", "seed": 0})
            o.fail(RuntimeError("boom"))
            rec = stock_fold.read_outcome(path)
            self.assertEqual(rec["item_failed"], {"id": "p2", "seed": 0, "error": "RuntimeError: boom"})
            self.assertEqual((rec["items_complete"], rec["n_predictions"]), (["p1"], 1))
            self.assertIsNone(stock_fold.read_outcome(os.path.join(tmp, "none.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pass_summary(self):
        self.assertEqual(stock_fold.pass_summary(None, 0), {"pass_status": "complete", "n_items": None, "n_items_complete": 0, "items_complete": [], "item_failed": None})
        self.assertEqual(stock_fold.pass_summary(None, 1)["pass_status"], "failed")
        done = {"n_items": 2, "items_complete": ["p1", "p2"], "current": None, "item_failed": None, "n_predictions": 2}
        self.assertEqual(stock_fold.pass_summary(done, 0), {"pass_status": "complete", "n_items": 2, "n_items_complete": 2, "items_complete": ["p1", "p2"], "item_failed": None})
        failed = {"n_items": 3, "items_complete": ["p1"], "current": {"id": "p2", "seed": 0}, "item_failed": {"id": "p2", "seed": 0, "error": "RuntimeError: x"}, "n_predictions": 1}
        s = stock_fold.pass_summary(failed, 1)
        self.assertEqual((s["pass_status"], s["n_items_complete"], s["item_failed"]["id"]), ("incomplete", 1, "p2"))
        self.assertEqual(stock_fold.pass_line(s), "pred INCOMPLETE: 1/3 complexes complete; failed on p2 s0: RuntimeError: x")
        killed = {"n_items": 3, "items_complete": ["p1", "p2"], "current": {"id": "p3", "seed": 0}, "item_failed": None, "n_predictions": 2}
        s = stock_fold.pass_summary(killed, 137)
        self.assertEqual(s["pass_status"], "incomplete")
        self.assertEqual(s["item_failed"], {"id": "p3", "seed": 0, "error": "the process ended (exit code 137) during this fold"})
        nothing = {"n_items": 3, "items_complete": [], "current": None, "item_failed": None, "n_predictions": 0}
        s = stock_fold.pass_summary(nothing, 1)
        self.assertEqual((s["pass_status"], s["item_failed"]["id"]), ("failed", None))
        self.assertEqual(stock_fold.pass_line(s), "pred FAILED: 0/3 complexes complete; failed on before the first fold: the process ended (exit code 1) before its next fold")


if __name__ == "__main__":
    unittest.main()
