"""af2ig_opt.stack.applied — L11 (-fused_trimul) above the memory line's -trimul_chunk floor: every TriangleMultiplication trace runs the
row-chunked body (af2ig_opt.pairstack) and none reaches the fused block, by the mode's definition. The pairstack census counts those traces, so the
lever is SUPERSEDED — its own bucket (`superseded`, reason `superseded_by:trimul_chunk …`), printed `state=off` with that reason, neither partial nor
`on` nor in levers_applied; 0 served WITHOUT an engaged trimul_chunk stays partial."""
import json, os, tempfile, unittest

from af2ig_opt import registry, report, stack


def timers(records):
    d = tempfile.mkdtemp(); p = os.path.join(d, "timers.jsonl")
    with open(p, "w") as fh:
        for r in records: fh.write(json.dumps(r) + "\n")
    return p


def base_records(fused_trimul, pairstack):
    lv = registry.LEVERS[registry.L11]
    recs = [{"kind": "proc_start", "argv": ["predict_pdb.py", lv.switch, registry.LEVERS[registry.TRIMUL].switch, "256:1473"]}]
    recs.append(dict(kind="fused_trimul", **fused_trimul))
    if pairstack is not None: recs.append(dict(kind="pairstack", **pairstack))
    return recs


class TrimulSuperseded(unittest.TestCase):
    def test_zero_fused_calls_above_the_chunk_floor_is_superseded_not_partial_not_on(self):
        recs = base_records({"served": 0, "fallback": 0, "fallback_by": {}}, {"installed": True, "trimul_chunk": {"rows": 256, "min_residues": 1473, "engaged_traces": 4, "disengaged_traces": 0, "producer": "x"}})
        rep = {"levers_planned": [registry.L11, registry.TRIMUL], "mode": "big"}
        got = stack.applied(rep, timers(recs))
        self.assertEqual(got["partial"], [], got.get("partial_reasons"))
        self.assertNotIn(registry.L11, got["levers_applied"]); self.assertNotIn(registry.L11, got["evidence"])       # not applied: no call reached the block
        self.assertEqual(list(got["superseded"]), [registry.L11]); self.assertTrue(got["superseded"][registry.L11].startswith("superseded_by:trimul_chunk — 0 TriangleMultiplication calls reached the fused block, 4 trace(s) ran the row-chunked body (rows=256 min_residues=1473)"), got["superseded"])
        self.assertEqual(got["levers_applied"], [registry.TRIMUL])                                                      # planned = applied + partial + superseded, disjoint
        lines = report.lever_lines(dict(rep, **got))
        l11 = [ln for ln in lines if ln.startswith("[af2ig-opt] LEVER name=L11 ")]
        self.assertEqual(len(l11), 1); self.assertIn(" state=off reason=superseded_by:trimul_chunk_—_0_TriangleMultiplication_calls_reached_the_fused_block,_4_trace(s)_ran_the_row-chunked_body_", l11[0])
        self.assertIn(" flag=-fused_trimul arm=big", l11[0]); self.assertNotIn("evidence=", l11[0]); self.assertNotIn("name=L11 state=on", "\n".join(lines))
        self.assertTrue(any(ln.startswith("[af2ig-opt] LEVER name=trimul_chunk state=on ") for ln in lines), lines)

    def test_zero_fused_calls_without_an_engaged_chunk_stays_partial(self):
        for ps in (None, {"installed": True, "trimul_chunk": {"rows": 256, "min_residues": 1473, "engaged_traces": 0, "disengaged_traces": 3}}):
            recs = base_records({"served": 0, "fallback": 0, "fallback_by": {}}, ps)
            got = stack.applied({"levers_planned": [registry.L11], "mode": "big"}, timers(recs))
            self.assertEqual(got["partial"], [registry.L11])
            self.assertRegex(got["partial_reasons"][registry.L11], "0 TriangleMultiplication calls served")

    def test_fallbacks_with_zero_served_stay_partial_even_when_chunked(self):
        recs = base_records({"served": 0, "fallback": 2, "fallback_by": {"dtype": 2}}, {"installed": True, "trimul_chunk": {"rows": 256, "min_residues": 1473, "engaged_traces": 4, "disengaged_traces": 0}})
        got = stack.applied({"levers_planned": [registry.L11], "mode": "big"}, timers(recs))
        self.assertEqual(got["partial"], [registry.L11])


if __name__ == "__main__":
    unittest.main()
