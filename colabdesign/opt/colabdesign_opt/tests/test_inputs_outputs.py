"""Inputs (chains, CA residues, the case), outputs read back (files, completeness), the
report lines."""
import os
import tempfile
import unittest

from . import _stubs
from colabdesign_opt import inputs, modes, names, outputs, report


class TestInputs(unittest.TestCase):
    def test_chains_and_target(self):
        self.assertEqual(inputs.parse_chains("A,B, C"), ["A", "B", "C"]); self.assertEqual(inputs.parse_chains("A+B"), ["A", "B"])
        with self.assertRaises(ValueError):
            inputs.parse_chains("AB")
        tmp = tempfile.mkdtemp()
        p = _stubs.write_target(os.path.join(tmp, "t.pdb"), ("A", "B"), 12)
        t = inputs.target(p, "A,B")
        self.assertEqual((t.residues, t.target_res, t.chains), ({"A": 12, "B": 12}, 24, ["A", "B"]))
        self.assertEqual(len(t.sha256), 64)
        with self.assertRaises(ValueError):
            inputs.target(p, "C")
        with self.assertRaises(FileNotFoundError):
            inputs.target(os.path.join(tmp, "nope.pdb"), "A")
        c = inputs.case(t, 100, 7, "fast", {"soft": 75, "temp": 45, "hard": 5, "greedy": 15}, hotspot="A5,A7")
        self.assertEqual((c["tokens"], c["seed"], c["chains"], c["steps"], c["greedy_rounds"], c["hotspot"]), (124, 7, "A,B", 125, 15, "A5,A7"))
        with self.assertRaises(ValueError):
            inputs.case(t, 0, 0, "off", {"soft": 1, "temp": 1, "hard": 1, "greedy": 1})

    def test_altloc_and_hetero_ignored(self):
        tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "t.pdb")
        open(p, "w").write("ATOM      1  CA AALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
                           "ATOM      2  CA BALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
                           "HETATM    3  CA  HOH A   2       0.000   0.000   0.000  1.00 20.00           C\n"
                           "ATOM      4  CA  GLY A   3       0.000   0.000   0.000  1.00 20.00           C\n")
        self.assertEqual(inputs.ca_residues(p, ["A"]), {"A": 2})


class TestOutputs(unittest.TestCase):
    def test_files_and_completeness(self):
        tmp = tempfile.mkdtemp()
        for n in ("design.pdb", "design.fasta"):
            open(os.path.join(tmp, n), "w").write(n)
        fs = outputs.files(tmp)
        self.assertTrue(fs["design.pdb"]["present"]); self.assertFalse(fs["trajectory.jsonl"]["present"]); self.assertEqual(set(fs), set(outputs.OUTPUTS))
        self.assertEqual(outputs.complete(fs), ["trajectory.jsonl"])
        self.assertEqual(fs["design.pdb"]["sha256"], outputs.sha256_file(os.path.join(tmp, "design.pdb"))); self.assertEqual(fs["design.pdb"]["bytes"], len("design.pdb"))
        res = modes.resolve("fast")
        self.assertIn("EXIT rc=0 mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla evidence=applied=none;fallback=none;missing=none", report.exit_line(0, res, {"applied": []}, tmp))


class TestReport(unittest.TestCase):
    def test_lines(self):
        rep = {"active": True, "mode": "fast", "route": "check", "tier": 2, "numerics_class": 2, "levers": ["nosub"], "arm": "kit", "line": "nosub: guarantee", "settings": "default_4stage_multimer",
               "upstream": {"colabdesign": {"version": "1.1.3", "commit": "e31a56fe1d9b4de2"}, "jax": "0.6.0", "dm-haiku": "0.0.17"}, "gpu": _stubs.GPU}
        line = report.active_line(rep)
        self.assertTrue(line.startswith("ACTIVE mode=fast route=check tier=2 class=2 colabdesign=1.1.3@e31a56fe jax=0.6.0 haiku=0.0.17 gpu=NVIDIA-H100-80GB-HBM3(sm90,81559MiB) "
                                        "levers=nosub skipped=none arm=kit settings=default_4stage_multimer line=nosub: guarantee"), line)
        self.assertEqual(report.active_line({"active": False, "reason": "x"}), "NOT ACTIVE reason=x")
        self.assertEqual(report.gpu_label({}), "none")
        self.assertEqual(report.run_line({"mode": "off", "target": "t.pdb", "chains": "A", "binder_len": 100, "tokens": 208, "seed": 0, "steps": 125, "greedy_rounds": 15}),
                         "RUN mode=off target=t.pdb chains=A binder_len=100 tokens=208 hotspot=none seed=0 steps<=125 greedy_rounds=15")
        ev = {"applied": ["nosub"], "gated": [], "fallback": [], "missing": ["pallas"], "grad_subbatch": "none", "fn_subbatch": "4", "pallas_served": None, "pallas_fallback": None, "pallas_fallback_by": None}
        self.assertEqual(report.evidence_line(ev), "EVIDENCE levers=nosub gated=none fallback=none missing=pallas skipped=none grad_subbatch=none fn_subbatch=4 compile_cache=n/a compile_threads=n/a prev_on_device=n/a trimul_served=n/a trimul_fallback_by=n/a triatt_served=n/a triatt_fallback_by=n/a opm_fold_served=n/a opm_fold_fallback_by=n/a ln_served=n/a ln_fallback_by=n/a proj_served=n/a proj_fallback_by=n/a transition_served=n/a transition_fallback_by=n/a txla_served=n/a txla_fallback_by=n/a pallas_served=n/a pallas_fallback=n/a pallas_fallback_by=none")
        self.assertEqual(report.settings_line({"name": "default_4stage_multimer", "file": "stock/x.json", "sha256": "ab" * 32}, {"design_models": [0, 1], "helicity_value": -0.3, "design_name": "d_l100_s0"}, None),
                         "SETTINGS default_4stage_multimer file=stock/x.json sha256=abababababababab design_models=[0, 1] helicity=-0.3 hotspot=none trajectory=d_l100_s0")
        run = {"tokens": 208, "steps": 125, "n_steps": {"soft": 75, "temp": 45, "hard": 5, "greedy_rounds": 15, "greedy_forward": 16}, "terminate": {"verdict": "", "gate": None},
               "timing": {"steady_s": 0.5, "phase_steady_s": {"soft": 0.5, "temp": 0.5, "hard": 0.6, "greedy_forward": 0.25}, "first_calls_s": [40.0, 1.0], "total_s": 150.0},
               "final": {"loss": 1.2345, "plddt": 0.8, "i_ptm": 0.7}, "files": {"design.pdb": {"sha256": "ab" * 32}}}
        s = report.run_summary_line("t", run)
        self.assertTrue(s.startswith("[run] t: tokens 208 steps 125 (soft 75 temp 45 hard 5 greedy_rounds 15 greedy_forward 16) terminate=complete steady 0.500 s/step soft 0.500 temp 0.500 hard 0.600 greedy_forward 0.250 first_calls 40.0,1.0 total 150.0 s loss[-1]=1.2345"), s)
        gated = report.run_summary_line("t", dict(run, terminate={"verdict": "LowConfidence", "gate": "Trajectory_logits_pLDDT"}))
        self.assertIn("terminate=terminated:Trajectory_logits_pLDDT ", gated)
        self.assertIn("design.pdb sha256=abababababababab", s)
        # the line read back (the calling process's only channel): every value the renderer wrote, at its printed precision
        p = report.parse_run_line(f"{names.PREFIX} {s}")
        self.assertEqual(p, {"tag": "t", "tokens": 208, "steps": 125, "n_steps": run["n_steps"], "terminate": "complete",
                             "timing": {"steady_s": 0.5, "phase_steady_s": {"soft": 0.5, "temp": 0.5, "hard": 0.6, "greedy_forward": 0.25}, "first_calls_s": [40.0, 1.0], "total_s": 150.0, "recycles": [], "segments": []},
                             "final": {"loss": 1.2345, "plddt": 0.8, "i_ptm": 0.7}, "design_pdb_sha256_16": "abababababababab"})
        self.assertTrue(s.endswith(" design.pdb sha256=abababababababab recycles=none segments=none"), s)     # no per-step rows in this record: the equal-work words print none
        # the equal-work cells: steady s/step per (stage, num_recycles) and the runs of constant num_recycles (BindCraft's optimise_beta branch raises it mid-run)
        grads = [{"k": k, "phase": ("soft" if k < 75 else "temp" if k < 120 else "hard"), "recycles": (1 if k < 50 else 3), "wall_s": (40.0 if k == 0 else (1.47 if k < 50 else 2.145)), "first_call": k in (0, 75, 120)} for k in range(125)]
        segs, rec = report.work_segments(grads)
        self.assertEqual(rec, [{"recycles": 1, "k0": 0, "k1": 49}, {"recycles": 3, "k0": 50, "k1": 124}])
        self.assertEqual(segs, [{"phase": "soft", "recycles": 1, "n": 50, "steady_s": 1.47}, {"phase": "soft", "recycles": 3, "n": 25, "steady_s": 2.145}, {"phase": "temp", "recycles": 3, "n": 45, "steady_s": 2.145}, {"phase": "hard", "recycles": 3, "n": 5, "steady_s": 2.145}])
        s2 = report.run_summary_line("t", dict(run, timing=dict(run["timing"], segments=segs, recycles=rec)))
        self.assertTrue(s2.endswith(" recycles=1:0-49,3:50-124 segments=soft/r1:50@1.470,soft/r3:25@2.145,temp/r3:45@2.145,hard/r3:5@2.145"), s2)
        p2 = report.parse_run_line(f"{names.PREFIX} {s2}")
        self.assertEqual((p2["timing"]["recycles"], p2["timing"]["segments"]), (rec, segs))
        self.assertEqual(report.parse_run_line(f"{names.PREFIX} {s2.split(' recycles=')[0]}")["timing"]["segments"], [])   # a 0.5.x line (no such words) still parses
        self.assertEqual(report.parse_run_line(gated)["terminate"], "terminated:Trajectory_logits_pLDDT")
        bare = report.parse_run_line(report.run_summary_line("u", {"tokens": 80, "steps": 0}))          # absent values print nan / None / none / ? and read back as None / []
        self.assertEqual((bare["tokens"], bare["steps"], bare["n_steps"]["soft"], bare["timing"]["steady_s"], bare["timing"]["first_calls_s"], bare["final"]["loss"], bare["design_pdb_sha256_16"]), (80, 0, None, None, [], None, "?"))
        self.assertIsNone(report.parse_run_line(f"{names.PREFIX} RUN mode=off target=t.pdb")); self.assertIsNone(report.parse_run_line("Stage 1: Test Logits"))
        # run_lines: every [run] line in order; ready_s = the arrival of the SETTINGS line last printed before it (driver.launch stamps), None without one
        st = f"{names.PREFIX} {report.settings_line({'name': 'default_4stage_multimer', 'file': 'f', 'sha256': '0' * 64}, {'design_models': [0], 'helicity_value': 0, 'design_name': 't'}, None)}"
        tr = [f"{names.PREFIX} RUN mode=off", st, "Stage 1: Test Logits", f"{names.PREFIX} {s}", f"{names.PREFIX} {gated}", st, st.replace("=t", "=v"), f"{names.PREFIX} {report.run_summary_line('v', run)}"]
        got = report.run_lines(tr, [0.1, 7.5, 8.0, 160.0, 161.0, 170.0, 171.0, 330.0])
        self.assertEqual([(r["tag"], r["timing"]["ready_s"]) for r in got], [("t", 7.5), ("t", None), ("v", 171.0)])   # a second [run] with no SETTINGS of its own carries none; a SETTINGS with no [run] after it (a failed design) yields to the next SETTINGS
        self.assertEqual([("ready_s" in r["timing"]) for r in report.run_lines(tr)], [False, False, False])           # no stamps: no ready_s at all
        self.assertTrue(report.is_settings_line(st)); self.assertTrue(report.is_settings_line(st[len(names.PREFIX) + 1:])); self.assertFalse(report.is_settings_line(f"{names.PREFIX} {s}"))   # prefixed as the arm prints it, or bare
        with self.assertRaises(ValueError):
            report.run_lines(tr, [0.0])                                                                             # one stamp per line


if __name__ == "__main__":
    unittest.main()
