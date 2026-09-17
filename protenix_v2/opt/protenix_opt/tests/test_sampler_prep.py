"""sampler_prep — the graphed sampler's host path: registry row (EXACT, env probe, one switch), membership (exact + fast; BIG_DROPPED with the
sampler graph), env.sh's switch line, the reconciliation rule (the record's prep= word must name all six parts, else a fallback by name), the LEVER
evidence off the clisampler record, and the shipped module's surface (reads PTX_SAMPLER_PREP only)."""
import os
import re
import unittest

from protenix_opt import modes, registry, report, stack

HERE = os.path.dirname(os.path.abspath(__file__))
FPF = os.path.abspath(os.path.join(HERE, "..", "..", "forward", "flashpairformer"))
SP = os.path.join(FPF, "src", "infopt_graphs", "protenix", "sampler_prep.py")
PARTS = "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec"


class TestRow(unittest.TestCase):
    def test_registry_modes_env_sh(self):
        lv = registry.LEVERS["sampler_prep"]
        self.assertEqual((lv.tier, lv.probe, lv.env_keys), (registry.EXACT, "env", ("PTX_SAMPLER_PREP",)))
        for mode in ("exact", "fast"):
            m = modes.MODES[mode]; self.assertEqual(m.index("sampler_prep"), m.index("sampler_graph_cache_policy") + 1, mode)
        self.assertNotIn("sampler_prep", modes.BIG_DROPPED); self.assertIn("sampler_prep", modes.MODES["big"])      # the graphed sampler rides the memory row under sampler_admit (kit 0.3.29): its host path with it
        env_sh = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
        self.assertIn("PTX_SAMPLER_PREP=${PTX_SAMPLER_PREP:-1}", env_sh); self.assertIn('[ "$PTX_SAMPLER_GRAPH" = "0" ] && unset PTX_SAMPLER_GRAPH PTX_SAMPLER_HOIST PTX_SAMPLER_PREP', env_sh)
        self.assertEqual(stack.LATE_RECORDS["sampler_prep"][0], "clisampler")
        self.assertEqual(report.STRATEGY_IDS["sampler_prep"], "LOCAL.protenix_v2.sampler_prep")

    def test_the_module_parts_are_the_kits_word_and_it_reads_one_switch(self):
        src = open(SP, encoding="utf-8").read()
        parts = re.search(r'^PARTS\s*=\s*\(([^)]*)\)', src, re.M).group(1)
        self.assertEqual("+".join(re.findall(r'"([a-z0-9_]+)"', parts)), PARTS); self.assertEqual(stack.SAMPLER_PREP_PARTS, PARTS)
        self.assertEqual(sorted(set(re.findall(r"""(?:environ|env)(?:\.get)?\s*[\[(]\s*["']([A-Z][A-Z0-9_]+)["']""", src))), ["PTX_SAMPLER_PREP", "PTX_SAMPLER_REACH"],
                         "v1.0.2 reads its own switch and the (not yet registered, never exported) sampler_reach switch — nothing else")


class TestReconcile(unittest.TestCase):
    def _rep(self):
        return {"active": True, "mode": "exact", "levers_applied": ["sampler_graph", "sampler_graph_cache_policy", "sampler_prep"], "levers_fallback": [], "fallback_reasons": {}}

    def _reconcile(self, rec):
        return stack.reconcile(self._rep(), {"clisampler": rec})

    def test_full_parts_word_keeps_the_lever_on_and_feeds_the_evidence(self):
        r = self._reconcile({"installed": True, "prep": PARTS, "prep_poison": "ok", "prep_stats": {"key_shape_miss": 2, "pool_chained": 1, "pool_renewed": 0, "poison_skipped": 1, "rot_ring_waits": 0, "key_value_miss": 0}})
        self.assertIn("sampler_prep", r["levers_applied"]); self.assertNotIn("sampler_prep", r["levers_fallback"])
        self.assertEqual(report.sampler_prep_evidence(r), [("parts", PARTS), ("poison", "ok"), ("key_shape_miss", 2), ("key_value_miss", 0), ("pool_chained", 1), ("pool_renewed", 0), ("poison_skipped", 1), ("rot_ring_waits", 0)])

    def test_a_subset_or_off_is_a_fallback_by_name(self):
        for word in ("rot_async+keycheck+warmup1+poison_once+pool_chain", "off", "none"):
            r = self._reconcile({"installed": True, "prep": word, "prep_poison": "ok"})
            self.assertIn("sampler_prep", r["levers_fallback"], word); self.assertNotIn("sampler_prep", r["levers_applied"])
            self.assertIn(f"prep={word} (expected {PARTS})", r["fallback_reasons"]["sampler_prep"])

    def test_a_failed_poison_test_is_a_fallback_by_name(self):
        for word in ("failed", "failed;squash:hoist.atomenc#.layernorm_kv.sig"):
            r = self._reconcile({"installed": True, "prep": PARTS, "prep_poison": word})
            self.assertIn("sampler_prep", r["levers_fallback"], word); self.assertIn("prep_poison=failed", r["fallback_reasons"]["sampler_prep"])

    def test_ok_with_squash_or_subsumed_suffixes_stays_on(self):
        for word in ("ok", "ok;squash:hoist.atomenc#.layernorm_kv.sig,hoist.atomdec#.layernorm_kv.lin", "ok;subsumed:hoist.x;unlisted:hoist.y", "not-run"):
            r = self._reconcile({"installed": True, "prep": PARTS, "prep_poison": word})
            self.assertIn("sampler_prep", r["levers_applied"], word)

    def test_no_record_yet_reads_record_none(self):
        self.assertEqual(report.sampler_prep_evidence({"gates": {}}), [("record", "none")])


if __name__ == "__main__":
    unittest.main()
