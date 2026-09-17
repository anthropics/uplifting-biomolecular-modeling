"""Inputs (shapes, the shape's files under the cache root), outputs (the driver's results.json read back: run fields, lever
classification, files) and the manifest round trip."""
import json
import os
import tempfile
import unittest

from mosaic_opt import inputs, outputs, report



def shown(*on):
    """The driver's per-lever booleans over EVERY registry lever: True for the ids named."""
    from mosaic_opt import registry
    return {k: k in on for k in registry.LEVERS}

class TestInputs(unittest.TestCase):
    def test_shape(self):
        d = {"--binder-length": 80, "--target-copies": 1}                     # the driver's own defaults (settings.driver_defaults reads them)
        s = inputs.shape_from_args(None, None, d)
        self.assertEqual((s.binder_length, s.target_copies, s.key, s.features_name), (80, 1, "L80_c1", "features_L80_c1.npz"))
        self.assertEqual(inputs.Shape(60, 2).flags(), ["--binder-length", "60", "--target-copies", "2"])
        self.assertEqual(inputs.shape_from_args(100, 2, d), inputs.Shape(100, 2))
        with self.assertRaises(TypeError):                                     # no defaults of its own: the driver's are the values used
            inputs.Shape()
        with self.assertRaises(ValueError):
            inputs.shape_from_args(0, 1, d)

    def test_shape_files(self):
        s = inputs.Shape(80, 2)
        self.assertEqual(inputs.shape_dir("/r", s), "/r/L80_c2")
        self.assertEqual(inputs.features_path("/r", s), "/r/L80_c2/features_L80_c2.npz")
        self.assertEqual(inputs.p1_dir("/r", s), "/r/L80_c2/xla_cache")
        tmp = tempfile.mkdtemp(); f = os.path.join(tmp, "f.npz"); open(f, "wb").write(b"abc")
        st = inputs.features_state(f)
        self.assertTrue(st["present"]); self.assertEqual(st["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(inputs.features_state(None), {"path": None, "present": False, "sha256": None})
        self.assertFalse(inputs.features_state(os.path.join(tmp, "nope"))["present"])


class TestOutputs(unittest.TestCase):
    def results(self, load="P2 load_stock_fast_init (torch ckpt -> joltz)", feats="frozen npz /f", cache="/c"):
        return {"manifest": {"tag": "t", "n_tokens": 169, "load_path": load, "features": {"source": feats, "sha256": "s"}, "identity_key_pre": {"jax_compilation_cache_dir": cache},
                             "identity_key_post": {"jax_compilation_cache_dir": cache, "jax_cache_n_entries": 3}, "host_class": {"gpu_nvidia_smi": "H100"}, "versions": {"jax": "0.10.2"}},
                "run": {"loss_traj": [1.0, 0.5], "loss_traj_sha256": "L" * 64, "x0_sha16": "x0", "x1_sha16": "x1", "best1_sha16": "b1", "x2_sha16": "x2", "best2_sha16": "b2",
                        "refold_coords_sha16": "rc", "refold_pae_sha16": "rp", "refold_plddt_sha16": "rl", "seq_stage2_x": "AA", "seq_stage2_best": "AB", "refold_iptm": 0.7,
                        "numeric_state_unchanged": True, "t_first_iter_s": 1.0}, "status": "ok", "t_process_total_s": 2.0}

    def test_run_fields_and_classify(self):
        r = self.results()
        idf = outputs.run_fields(r)
        self.assertEqual(set(idf), set(outputs.RUN_KEYS)); self.assertEqual(idf["best2_sha16"], "b2"); self.assertEqual(idf["refold_iptm"], 0.7)
        self.assertEqual(outputs.classify(r), shown("P1", "P2", "P3"))
        self.assertEqual(outputs.classify(self.results(load="stock Boltz2() (torch ckpt -> joltz.from_torch)", feats="in-process boltz featurization", cache="")),
                         shown())
        self.assertEqual(outputs.classify(None), shown())
        s = outputs.run_summary(r)
        self.assertEqual((s["n_tokens"], s["loss_0"], s["loss_last"], s["n_steps"], s["status"]), (169, 1.0, 0.5, 2, "ok"))
        self.assertEqual(outputs.run_summary(None)["status"], "no results.json")

    def test_read_results_and_files(self):
        tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "t"))
        self.assertIsNone(outputs.read_results(tmp, "t"))
        json.dump(self.results(), open(os.path.join(tmp, "t", "results.json"), "w"))
        open(os.path.join(tmp, "t", "pssm_seed0.npz"), "wb").write(b"x")
        self.assertEqual(outputs.read_results(tmp, "t")["status"], "ok")
        self.assertEqual(outputs.written_files(tmp, "t"), ["t/pssm_seed0.npz", "t/results.json"])


class TestReportLines(unittest.TestCase):
    def test_lines(self):
        rep = {"active": True, "mode": "exact", "route": "in-process", "row_line": "C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only]", "levers_applied": ["P1"], "levers_unavailable": [],
               "partial": False, "levers_driver_only": ["P2", "P3"], "p1": {"cache_dir": "/c", "autotune": "dump"}, "upstream": {"jax": "0.10.2", "jax-cuda12-plugin": "0.10.2", "mosaic": "0.1.0"},
               "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90"}}
        line = report.activation_line(rep)                                                  # the in-process route: the driver's call-site levers ride the row token, never unavailable= / partial=
        self.assertTrue(line.startswith("[mosaic-opt] ACTIVE mode=exact route=in-process row=C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] levers=P1 unavailable=none p1=dump:/c jax=0.10.2"))
        self.assertTrue(line.endswith("gpu=NVIDIA H100 80GB HBM3(sm90)"), line); self.assertNotIn("partial", line)
        genuine = dict(rep, levers_unavailable=["P1"], partial=True, unavailable_reason="pcc refused")   # the partial field's one meaning: a lever the route should apply and did not, with its reason
        self.assertIn("gpu=NVIDIA H100 80GB HBM3(sm90) partial=P1 (not applied on this route: pcc refused)", report.activation_line(genuine))
        self.assertEqual(report.activation_line({"active": False, "mode": "exact", "reason": "why"}), "[mosaic-opt] NOT ACTIVE: why (mode=exact route=none)")
        self.assertEqual(report.tally_line("off", "driver", None), f"[mosaic-opt] EXIT pid={os.getpid()} no lever applied in this process jax_cache=off autotune=off via=none autotune_sha256=none cache_entries=0")
        from opt_core.jax_design import pcc; from opt_core.report import kv
        t = report.tally_line("exact", "driver", "/c", autotune="load")
        self.assertEqual(t, f"[mosaic-opt] EXIT pid={os.getpid()} mode=exact route=driver P1 jax_cache=/c autotune=load via=exports autotune_sha256=none cache_entries=0")   # the fields read the directory as it stands at exit (/c: absent)
        self.assertTrue(report.tally_line("exact", "in-process", "/nonexistent-c").endswith(" route=in-process P1 jax_cache=/nonexistent-c autotune=dump via=none autotune_sha256=none cache_entries=0"))
        import tempfile; d = tempfile.mkdtemp(prefix="mosaic_opt_tally_"); open(os.path.join(d, "xla_autotune_results.pb"), "wb").write(b"abc"); open(os.path.join(d, "entry1"), "wb").write(b"e")
        rec = {"cache_dir": d, "autotune": "load", "applied_via": "environ+jax.config"}
        self.assertEqual(report.tally_line("exact", "in-process", d, via="environ+jax.config"), f"[mosaic-opt] EXIT pid={os.getpid()} mode=exact route=in-process P1 " + kv(**pcc.evidence_fields(rec)))   # one source: pcc
        self.assertTrue(report.tally_line("off", "driver", None).endswith(kv(**pcc.evidence_fields(None))))   # the vocabulary is pcc's, never retyped here
        self.assertEqual(report.gpu_label({"name": "X", "sm": "sm90"}), "X(sm90)"); self.assertEqual(report.gpu_label(None), "none")
        for m, r in (("fast", "driver"), ("big", "driver"), ("fast", "in-process")):             # a kit mode whose P1 was not applied in the process: the mode word, p1=none and where the levers report — never the no-lever wording
            t2 = report.tally_line(m, r, None)
            self.assertTrue(t2.startswith(f"[mosaic-opt] EXIT pid={os.getpid()} mode={m} route={r} p1=none (P1 not applied in this process; "), t2)
            t3 = report.tally_line(m, r, None, why_none="MOSAIC_OPT_CACHE_ROOT unset")               # the transparent form stepping aside: the reason by name
            self.assertTrue(t3.startswith(f"[mosaic-opt] EXIT pid={os.getpid()} mode={m} route={r} p1=none (P1 steps aside: MOSAIC_OPT_CACHE_ROOT unset; "), t3)
            self.assertTrue(t2.endswith(" LEVER lines) " + kv(**pcc.evidence_fields(None))), t2); self.assertNotIn("no lever applied", t2)
            self.assertNotRegex(t2, r"route=\S+ P1 jax_cache=")                                     # cannot read as the P1 tally of a row that has P1
        self.assertIn("no lever applied in this process", report.tally_line(None, None, None))

    def test_forced_word_on_the_lines_that_carry_the_mode(self):
        """MOSAIC_OPT_FORCE=1 with upstream off its pinned commit is a recorded override: stack.forced_word names the packages from the pins detail
        (stock/check_pins.py check()'s rows), the activation note and DONE's `forced=` field carry it; nothing when every package is pinned."""
        from mosaic_opt import stack
        detail = {"mosaic": {"pinned": True, "commit": "70fec525"}, "joltz": {"pinned": False, "commit": "abcd"}, "boltz": {"pinned": False, "commit": None}}
        self.assertEqual(stack.forced_word(detail), "upstream_pins(boltz,joltz)")
        self.assertIsNone(stack.forced_word({"mosaic": {"pinned": True}})); self.assertIsNone(stack.forced_word({"stub": True})); self.assertIsNone(stack.forced_word(None))
        self.assertTrue(stack.forced_note("upstream_pins(joltz)").startswith("forced=upstream_pins(joltz): MOSAIC_OPT_FORCE=1 overrides the upstream-commit pins"))
        run = {"loss_traj_sha256": "ab" * 32, "best2_sha16": "b2", "seq_stage2_best": "AAA", "refold_iptm": 0.5}
        self.assertIn(" iptm=0.5 forced=upstream_pins(joltz) out=/o", report.done_line("fast", run, "/o", 0, {"forced": "upstream_pins(joltz)"}))
        self.assertNotIn("forced=", report.done_line("fast", run, "/o", 0, {"forced": None})); self.assertNotIn("forced=", report.done_line("off", run, "/o", 0, None))
        self.assertEqual(report.forced_field({"forced": "upstream_pins(x)"}), " forced=upstream_pins(x)"); self.assertEqual(report.forced_field({}), "")


if __name__ == "__main__":
    unittest.main()

