"""The per-lever evidence line: one `[esmfold2-opt] LEVER ...` line per registry lever for the process's arm, composed from the shared
grammar (opt_core.report.prefix / kv), every lever filed under exactly one cross-engine strategy id (registry.STRATEGY), and the state
word derived from the kit's own records the way the APPLIED line is (on / off / skipped + reason)."""
import re
import unittest

from esmfold2_opt import registry, report


class TestLeverLines(unittest.TestCase):
    def test_every_lever_is_filed_under_one_strategy(self):
        self.assertEqual(set(registry.STRATEGY), set(registry.LEVERS), "registry.STRATEGY files exactly the registry's levers")
        from opt_core.strategies import check_strategy
        for name, sid in registry.STRATEGY.items():
            self.assertRegex(sid, r"^(F[2-7]\.[a-z0-9_]+|LOCAL\.[a-z0-9_]+|LOCAL\.esmfold2\.[a-z0-9_]+)$", name)      # ESMFold2 has no triangle attention: no F1 lever
            self.assertEqual(check_strategy(sid), sid, f"{name}: {sid} is not canonical (opt_core STRATEGIES.json) nor LOCAL.esmfold2.<name>")
        self.assertEqual(registry.STRATEGY["tx"], "F2.fpf_trimul_fast")

    def test_one_line_per_lever_with_state_and_reason(self):
        rep = {"mode": "fast", "variant": "fast", "server_line": "opt14_msa"}
        rec = {"model_index": 0, "levers_applied": ["fused", "tg", "tx"], "levers_fallback": ["t10", "t3"], "levers_substituted": {"t10": "t1"},
               "fallback_reasons": {"t3": "kit record w4:tiles is not set after configure()"}, "levers_not_for_variant": ["t11", "msa"]}
        lines = report.lever_lines(rep, rec)
        census = {n for n, lv in registry.LEVERS.items() if lv.field != registry.FIELD_RC}   # the multi-GPU route's row-chunking levers are in no n_gpu 1 set: no line for them here
        self.assertEqual(len(lines), len(census))                          # (test_rowchunk: at n_gpu 2 every registry lever has its line, those seven included)
        by = {re.search(r" name=(\S+)", l).group(1): l for l in lines}
        self.assertEqual(set(by), census)
        self.assertFalse({"injrows", "confrows", "zbf16"} & set(by))
        for l in lines:
            self.assertTrue(l.startswith("[esmfold2-opt] LEVER name="), l)
            self.assertRegex(l, r"^\[esmfold2-opt\] LEVER name=\S+ state=(on|off|skipped) (reason=\S+ )?impl=\S+ origin=kit strategy=(F[2-7]|LOCAL)\.[a-z0-9_.]+ mode=fast variant=fast model=0 tier_vs_stock=")
        self.assertIn("name=tx state=on impl=ef2_w4.py origin=kit strategy=F2.fpf_trimul_fast ", by["tx"])
        self.assertIn(" state=skipped reason=substituted_by:t1 ", by["t10"])
        self.assertIn(" state=skipped reason=fallback:kit_record_w4:tiles_is_not_set_after_configure() impl=ef2_w4.py origin=kit strategy=LOCAL.esmfold2.tile_table ", by["t3"])
        self.assertIn(" state=skipped reason=not_for_variant:fast ", by["t11"])
        self.assertIn(" state=off impl=", by["x4"])                            # a big lever in a fast process: not in this mode's set (no reason key)
        skipped = [l for l in lines if " state=skipped " in l]
        self.assertTrue(all(" reason=" in l for l in skipped), "a skipped lever always carries its reason")
        self.assertTrue(all(" reason=" not in l for l in lines if " state=on " in l or " state=off " in l))
        for l in lines:                                                      # every value is ONE blank-free token: the line splits into k=v pairs
            body = l[len("[esmfold2-opt] LEVER "):]
            self.assertTrue(all("=" in tok for tok in body.split(" ")), l)


if __name__ == "__main__":
    unittest.main()
