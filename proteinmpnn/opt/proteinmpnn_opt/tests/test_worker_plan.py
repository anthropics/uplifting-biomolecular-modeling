"""The exact worker's batch planner (opt/forward/mpnn_exact_worker/addon/mpnn_worker2.py ``plan_batches``, read out of the carried source: the
worker imports torch, the tests do not): with no backbone shorter than k_neighbors the batches are exactly the ``--bb_batch`` slices of the
processing order (the batching the exact line has always run); a shorter backbone is a batch of its own (the batched featuriser's pad-free neighbour sets need
L >= k; alone it runs the stock shapes) and the others keep filling batches of ``--bb_batch`` — the lever steps aside for that backbone by
name (the worker prints one ``bb_batch: ...`` line), never for the job, and nothing asserts."""
import ast
import os
import unittest

from proteinmpnn_opt import modes, stack

WORKER = os.path.join(stack.kit_home(), modes.WORKER_DIR, "addon", "mpnn_worker2.py")


def _plan_batches():
    src = open(WORKER, encoding="utf-8").read()
    tree = ast.parse(src)
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan_batches"]
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), WORKER, "exec"), ns)
    return ns["plan_batches"], src


class TestPlanBatches(unittest.TestCase):
    def test_no_short_backbone_is_the_plain_slicing(self):
        plan, _ = _plan_batches()
        for n in (1, 2, 15, 16, 17, 33, 100):
            order = list(range(n))[::-1]                                          # any processing order: the slices follow it
            lengths = [100 + i for i in range(n)]
            for bb in (1, 4, 16, 32):
                self.assertEqual(plan(order, lengths, bb, 48), [order[i:i + bb] for i in range(0, n, bb)], (n, bb))

    def test_a_short_backbone_is_a_batch_of_its_own(self):
        plan, _ = _plan_batches()
        lengths = [30, 47, 48, 100, 120, 20, 300, 301, 302]                           # indices 0, 1, 5 are shorter than k = 48
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])                 # --sort_by_length: 5, 0, 1, 2, 3, 4, 6, 7, 8
        self.assertEqual(plan(order, lengths, 4, 48), [[5], [0], [1], [2, 3, 4, 6], [7, 8]])
        self.assertEqual(plan(list(range(9)), lengths, 4, 48), [[0], [1], [2, 3, 4], [5], [6, 7, 8]])   # unsorted: a short one closes the batch before it
        self.assertEqual(plan(order, lengths, 1, 48), [[i] for i in order])          # --bb_batch 1: one per batch whatever the lengths
        flat = [i for b in plan(order, lengths, 4, 48) for i in b]
        self.assertEqual(flat, order)                                                 # every backbone once, processing order kept (each backbone's RNG stream offset is its own)

    def test_main_runs_the_plan_and_names_the_step_aside(self):
        _, src = _plan_batches()
        self.assertEqual(src.count('batches = plan_batches(order, [len(p["seq"]) for p in proteins], args.bb_batch, int(ck["num_edges"]))'), 1)
        self.assertEqual(src.count("Ks = [len(b) for b in batches]"), 1)             # the probe judges the cells of the batches that run
        self.assertIn('for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):', src); self.assertNotIn("assert min(len(p[\"seq\"]) for p in batch)", src)
        self.assertIn('print("bb_batch: %d backbone(s) shorter than k_neighbors=%d residues run one per batch', src)

    def test_a_probe_fail_refuses_the_job_before_any_output(self):
        """The line is all of its levers: when the probe-gated --hybrid_gemm is requested and the worker's on-device probe fails, main() refuses the job by
        name (its REFUSED line, `refused` in its end-of-run record, exit 3) before the batch loop — guarded on the request, so --hybrid_gemm 0 and a
        probe PASS run exactly as before."""
        _, src = _plan_batches()
        guard = '    if bool(args.hybrid_gemm and args.chunk_gemm) and not w.hybrid_active:\n'
        self.assertEqual(src.count(guard), 1)
        i = src.index(guard); j = src.index("    t_run0 = time.time(); done = 0\n")
        block = src[i:j]
        self.assertLess(src.index('T["hybrid_gemm_probe"] = w.decide_hybrid('), i); self.assertLess(j, src.index("    " + 'for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):'))
        self.assertIn('T["refused"] = "hybrid_gemm: PROBE FAIL"', block); self.assertIn('print("hybrid_gemm: REFUSED ->', block)
        self.assertIn("print(json.dumps(T)); sys.exit(3)", block)


if __name__ == "__main__":
    unittest.main()
