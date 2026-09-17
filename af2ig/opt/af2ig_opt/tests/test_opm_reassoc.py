"""L12 -opm_reassoc: the re-associated OuterProductMean (af2ig_opt.opm) — the two float32 orderings against a float64 reference at small shapes
(numpy on CPU: the package tests need no jax), the refusal rule, and the lever's place in the kit's statement (MANIFEST tier2_opt_in, the mode
table, the registry, the driver's flag, the stock caller's forbidden list, the patch series)."""
import json
import os
import unittest

import numpy as np

from af2ig_opt import modes, opm, registry, report, stack, stock_cli
from . import _stubs


class Shape:
    def __init__(self, shape): self.shape = shape


class TestOpmNumerics(unittest.TestCase):
    def _case(self, S, N, C, F, seed, masked_rows=0):
        rng = np.random.default_rng(seed)
        mask = np.ones((S, N, 1), np.float32)
        if masked_rows: mask[S - masked_rows:] = 0.0                     # padded MSA rows (af2ig: 1 real cluster row of 5; 0 real extra rows of 5)
        left = (mask * rng.standard_normal((S, N, C))).astype(np.float32)
        right = (mask * rng.standard_normal((S, N, C))).astype(np.float32)
        w = (rng.standard_normal((C, C, F)) / np.sqrt(C * C)).astype(np.float32)
        b = rng.standard_normal((F,)).astype(np.float32)
        return left, right, w, b

    def test_orderings_agree_and_have_the_same_fp64_error(self):
        for (S, N, C, F, masked) in ((5, 24, 32, 128, 0), (5, 24, 32, 128, 4), (5, 17, 32, 128, 5), (1, 9, 8, 16, 0), (7, 12, 32, 64, 2)):
            left, right, w, b = self._case(S, N, C, F, seed=S * 100 + N, masked_rows=masked)
            ref = opm.stock_order(*(x.astype(np.float64) for x in (left, right, w, b)), np)        # float64 reference in stock's order
            ref2 = opm.reassociated(*(x.astype(np.float64) for x in (left, right, w, b)), np)     # the algebra: identical in exact arithmetic
            self.assertLess(np.abs(ref - ref2).max(), 1e-11 * max(1.0, np.abs(ref).max()))
            st = opm.stock_order(left, right, w, b, np); ra = opm.reassociated(left, right, w, b, np)
            self.assertEqual((st.shape, ra.shape, st.dtype, ra.dtype), ((N, N, F), (N, N, F), np.float32, np.float32))
            scale = max(1.0, float(np.abs(ref).max()))
            e_st, e_ra, d = float(np.abs(st - ref).max()) / scale, float(np.abs(ra - ref).max()) / scale, float(np.abs(st - ra).max()) / scale
            self.assertLess(e_st, 2e-6, (S, N, e_st)); self.assertLess(e_ra, 2e-6, (S, N, e_ra))   # both float32 orderings sit at float32 rounding of the fp64 value
            self.assertLess(d, 4e-6, (S, N, d))                                                     # and differ from each other by the same order (re-association only)
            if masked == S:                                                                          # every row masked: the update is the bias alone in both bodies
                self.assertTrue(np.array_equal(ra, np.broadcast_to(b, (N, N, F)))); self.assertTrue(np.array_equal(st, ra))

    def test_refusal_rule(self):
        self.assertIsNone(opm.reason_for(Shape((5, 40, 256)), Shape((5, 40))))
        self.assertIsNone(opm.reason_for(Shape((5, 40, 64)), Shape((5, 40))))                       # the extra-MSA stack (c_m 64) is served like the Evoformer's (256)
        self.assertEqual(opm.reason_for(Shape((5, 40, 256)), Shape((5, 41))), opm.BAD_INPUT)
        self.assertEqual(opm.reason_for(Shape((40, 256)), Shape((5, 40))), opm.BAD_INPUT)
        self.assertEqual(opm.EXPECTED_FALLBACKS, ())                                                 # declared empty: a refused call makes the run partial (stack._undeclared)
        self.assertEqual(opm._shape_key(Shape((5, 400, 256))), "S5xN400xM256")

    def test_lever_wiring(self):
        L = registry.OPM
        self.assertEqual((opm.LEVER, opm.FLAG, opm.ORIGIN), (L, "-opm_reassoc", "kit"))
        lv = registry.LEVERS[L]
        self.assertEqual((lv.switch, lv.class_4, lv.strategy), ("-opm_reassoc", "forward", "LOCAL.af2ig.opm_reassoc")); self.assertFalse(lv.in_preset)
        self.assertEqual(report.IMPL[L], ("opm_reassoc_jax", "kit"))
        self.assertEqual(stack.FUSED_CENSUS[L][:2], ("opm_reassoc", "OuterProductMean"))
        self.assertIn(opm.FLAG, stock_cli.FORBIDDEN_FLAGS)                                          # never on the stock line
        self.assertIn(opm.FLAG, modes.kit_levers(_stubs.KIT)["tier2_opt_in"])
        for mode in ("fast", "big"):                                                               # fast AND big (it lowers the per-call working set); exact and off never
            r = modes.resolve(mode, _stubs.KIT)
            self.assertIn(opm.FLAG, r.flags, mode); self.assertIn(L, r.levers, mode)
            self.assertEqual(r.flags.index(opm.FLAG), r.flags.index(modes.FLAG_FTRIMUL) + 1)         # after -fused_trimul, before the memory line
        self.assertNotIn(opm.FLAG, modes.resolve("exact", _stubs.KIT).flags); self.assertEqual(modes.resolve("off", _stubs.KIT).flags, [])
        pins = json.load(open(os.path.join(_stubs.TREE, "stock", "PINS.json"), encoding="utf-8"))
        self.assertIn("patches/12_opm_reassoc.diff", pins["checkout"]["patches"])
        drv = open(os.path.join(_stubs.KIT, registry.DRIVER), encoding="utf-8").read()
        self.assertIn('parser.add_argument( "-opm_reassoc", action="store_true"', drv); self.assertIn("from af2ig_opt import opm as OPM", drv)
        self.assertIn("timers.emit('opm_reassoc', L_compiled=None, final=True, **OPM.census())", drv)

    def test_applied_reads_the_census(self):
        """served >= 1 and no fallback = on (evidence names the served OuterProductMean calls); 0 served = partial; an undeclared fallback = partial."""
        import tempfile
        L = registry.OPM
        def run(cens):
            d = tempfile.mkdtemp(); p = os.path.join(d, "timers.jsonl")
            recs = [{"kind": "proc_start", "argv": ["predict_pdb.py", "-opm_reassoc"]}] + ([dict(kind="opm_reassoc", **cens)] if cens is not None else [])
            with open(p, "w") as fh:
                for r in recs: fh.write(json.dumps(r) + "\n")
            return stack.applied({"levers_planned": [L], "mode": "fast"}, p)
        on = run({"served": 2, "fallback": 0, "fallback_by": {}, "impl": "opm_reassoc_jax", "origin": "kit", "precision": "default", "shapes": {"S5xN400xM256": 1, "S5xN400xM64": 1}})
        self.assertEqual((on["levers_applied"], on["partial"]), ([L], [])); self.assertIn("2 OuterProductMean call(s) traced onto the block opm_reassoc_jax (origin kit)", on["evidence"][L])
        self.assertNotIn("tiles=", on["evidence"][L])
        self.assertEqual(run({"served": 0, "fallback": 2, "fallback_by": {"unexpected_input_rank": 2}})["partial"], [L])
        self.assertEqual(run({"served": 2, "fallback": 1, "fallback_by": {"unexpected_input_rank": 1}})["partial"], [L])
        self.assertEqual(run(None)["partial"], [L])


if __name__ == "__main__":
    unittest.main()
