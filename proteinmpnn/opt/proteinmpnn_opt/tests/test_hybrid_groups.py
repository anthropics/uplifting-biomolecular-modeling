"""The worker's --hybrid_gemm GEMM grouping (addon/mpnn_worker2.py: HYBRID_GROUP_ROWS, hybrid_groups): a decode batch of K backbones is served by
one batched message GEMM ([K]) unless the card's row ceiling splits it into the fewest near-equal groups of at most ceiling // B backbones; the
probe (Worker.decide_hybrid) and the decoder layer (Worker.dec_layer) group through the same function. The two definitions are read out of the
worker's source with ast and executed alone (the worker imports torch at module level; the package tests carry no torch). No GPU."""
import ast
import os
import unittest

from proteinmpnn_opt import modes, stack

WORKER = os.path.join(modes.worker_home(stack.kit_home()), modes.WORKER)   # the file modes reads the worker facts from (modes._worker_source)


def worker_grouping():
    """{'HYBRID_GROUP_ROWS': ..., 'hybrid_groups': ...} compiled from the worker's own top-level definitions."""
    with open(WORKER, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "HYBRID_GROUP_ROWS" for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "hybrid_groups")]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), WORKER, "exec"), ns)
    return ns, src


class TestHybridGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.src = worker_grouping()

    def test_definitions_present(self):
        self.assertIn("HYBRID_GROUP_ROWS", self.ns); self.assertIn("hybrid_groups", self.ns)

    def test_no_ceiling_is_one_group(self):
        f = self.ns["hybrid_groups"]
        for K in range(1, 33):
            for B in (1, 2, 8):
                self.assertEqual(f(K, B, None), [K])

    def test_sm80_ceiling(self):
        rows, f = self.ns["HYBRID_GROUP_ROWS"], self.ns["hybrid_groups"]
        self.assertEqual(rows[(8, 0)], 112)                                   # 14 backbones x 8 rows: the largest bit-identical group measured on A100-SXM4-80GB
        c = rows[(8, 0)]
        for K in range(1, 15):
            self.assertEqual(f(K, 8, c), [K], K)                              # fits: one group, the sm_90 shape
        self.assertEqual(f(15, 8, c), [8, 7])
        self.assertEqual(f(16, 8, c), [8, 8])                                 # the exact line's --bb_batch 16 at B = 8: two batched GEMMs of 64 rows x 48 neighbours
        self.assertEqual(f(32, 8, c), [11, 11, 10])
        self.assertEqual(f(16, 7, c), [16])                                   # B = 7: 112 rows exactly
        self.assertEqual(f(16, 1, c), [16])
        for K in range(1, 65):
            for B in range(1, 9):
                gs = f(K, B, c)
                self.assertEqual(sum(gs), K, (K, B)); self.assertTrue(all(g >= 1 for g in gs), (K, B))
                self.assertTrue(all(g * B <= c or g == 1 for g in gs), (K, B, gs))   # every group under the ceiling (a single backbone is the floor)
                self.assertLessEqual(max(gs) - min(gs), 1, (K, B, gs))        # near-equal

    def test_sm90_has_no_ceiling(self):
        self.assertNotIn((9, 0), self.ns["HYBRID_GROUP_ROWS"])              # H100 / H200: the whole decode batch in one GEMM, as probed there

    def test_probe_and_layer_group_alike(self):
        self.assertEqual(self.src.count("hybrid_groups(K, B, self.hybrid_group_rows)"), 2)   # Worker.dec_layer and Worker.decide_hybrid: one grouping rule


if __name__ == "__main__":
    unittest.main()
