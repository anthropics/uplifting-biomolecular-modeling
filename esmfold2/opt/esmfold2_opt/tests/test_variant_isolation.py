"""One variant per process, one mode per process; the levers go on through the kit server's own configure() once per model, in a
subprocess with a stub upstream and a stub kit whose server carries the real MODES table. Also: flag vs ESMFOLD2_VARIANT
disagreement is refused, caller-set EF2_* switches are dropped before the server reads them, the lazy hook applies at the first
fold, and the kit-mode `pred` path writes the same schema, the application on its printed lines."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from esmfold2_opt.tests import _stubs

SCRIPT = textwrap.dedent('''
    import json, os, sys
    import esmfold2_opt
    from esmfold2_opt import stack
    out = {}
    r1 = esmfold2_opt.enable("fast", "full_msa")
    out["r1"] = {k: r1.get(k) for k in ("active", "mode", "variant", "server_mode", "applied", "unset", "env")}
    out["env_after"] = {k: os.environ.get(k) for k in ("EF2_W4", "EF2_MK", "EF2_MSA", "ESMFOLD2_OPT", "ESMFOLD2_VARIANT")}
    r2 = esmfold2_opt.enable("fast", "fast")                      # a second variant in the same process
    out["r2"] = {k: r2.get(k) for k in ("active", "variant", "reason")}
    r3 = esmfold2_opt.enable("exact", "full_msa")          # a second mode in the same process
    out["r3"] = {k: r3.get(k) for k in ("active", "mode", "reason")}
    try:
        esmfold2_opt.enable("exact", "full_msa", strict=True); out["strict_raised"] = False
    except esmfold2_opt.ActivationError as e:
        out["strict_raised"] = True
    r4 = esmfold2_opt.enable("fast", "full_msa")                  # idempotent
    out["r4_same"] = (r4.get("active"), r4.get("server_mode")) == (True, r1["server_mode"])
    # the lazy route: a model folded through the (wrapped) builder gets configured once
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import deserialize_structure_prediction_input
    import ef2_server
    m = ESMFold2Model.from_pretrained("biohub/ESMFold2")
    b = ESMFold2InputBuilder()
    spi = deserialize_structure_prediction_input({"sequences": [{"type": "protein", "id": "A", "sequence": "MKVL", "msa": None}]})
    b.fold(m, spi, seed=0); b.fold(m, spi, seed=1)
    out["configure_calls"] = list(ef2_server.CALLS)
    st = stack.status()
    out["status"] = {k: st.get(k) for k in ("applied", "levers_applied", "levers_fallback", "partial", "models")}
    m2 = ESMFold2Model.from_pretrained("biohub/ESMFold2")
    b.fold(m2, spi, seed=0)
    out["n_configured_after_second_model"] = len(ef2_server.CALLS)
    print(json.dumps(out, default=str))
''')


class TestVariantIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_kit = _stubs.require_kit()
        cls.pins = _stubs.require_pins()
        cls.tmp = tempfile.mkdtemp()
        cls.site = _stubs.write_stub_upstream(cls.tmp, cls.pins)
        cls.kit = _stubs.write_stub_kit(cls.tmp, cls.real_kit)
        cls.tree = _stubs.write_stub_tree(cls.tmp, cls.pins)
        cls.items = os.path.join(cls.tmp, "items.json")
        with open(cls.items, "w") as fh:
            json.dump({"id": "p1", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVLAG", "msa": None},
                                                 {"type": "protein", "id": "B", "sequence": "MKV", "msa": None}],
                       "binder_chains": ["A"], "target_chains": ["B"]}, fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, args, **extra):
        env = _stubs.env_for_stub(self.site, self.kit, self.tree, **extra)
        return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)

    def test_one_variant_and_one_mode_per_process(self):
        r = self._run(["-c", SCRIPT], EF2_W4="t6", EF2_MSA="t13")          # a caller-set server switch (EF2_MSA) must be dropped; EF2_W4 is no switch of the package (named by the startup NOTE, left alone)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["r1"]["active"]); self.assertEqual(out["r1"]["server_mode"], "opt14_msa"); self.assertEqual(out["r1"]["applied"], "deferred")
        self.assertEqual(sorted(out["r1"]["unset"]), ["EF2_MSA"])
        self.assertEqual(out["env_after"], {"EF2_W4": "t6", "EF2_MK": None, "EF2_MSA": None, "ESMFOLD2_OPT": "fast", "ESMFOLD2_VARIANT": "full_msa"})
        self.assertFalse(out["r2"]["active"]); self.assertIn("one mode and one variant per process", out["r2"]["reason"])
        self.assertFalse(out["r3"]["active"]); self.assertIn("already active", out["r3"]["reason"])
        self.assertTrue(out["strict_raised"]); self.assertTrue(out["r4_same"])
        calls = out["configure_calls"]
        self.assertEqual(len(calls), 1, "configure() must run once per model, at the first fold")
        self.assertEqual(calls[0]["mode"], "opt14_msa")
        self.assertEqual((calls[0]["EF2_MK"], calls[0]["EF2_MSA"]), (None, None))                  # the server's two override switches unset at configure()
        self.assertEqual(out["status"]["applied"], "configured")
        self.assertIn("fused", out["status"]["levers_applied"])
        self.assertEqual(out["n_configured_after_second_model"], 2)
        self.assertIn("[esmfold2-opt] ACTIVE mode=fast variant=full_msa server_mode=opt14_msa", r.stderr)
        self.assertIn("[esmfold2-opt] APPLIED model#0", r.stderr)
        self.assertIn("[esmfold2-opt] EXIT pid=", r.stderr)

    def test_exact_exports_no_override_switch(self):
        """exact activates as the bare server mode opt7x on every variant: no EF2_MK / EF2_MSA / EF2_W4 in the process, the same
        levers on the Fast and the Full model; the ACTIVE line carries the bare name."""
        script = textwrap.dedent('''
            import json, os, sys, esmfold2_opt
            r = esmfold2_opt.enable("exact", sys.argv[1])
            print(json.dumps({"active": r["active"], "server_line": r["server_line"], "env": {k: os.environ.get(k) for k in ("EF2_MK", "EF2_MSA", "EF2_W4")},
                              "levers": r["levers_planned"], "not_for_variant": r["levers_not_for_variant"]}))
        ''')
        seen = []
        for v in ("fast", "full_nomsa", "full_msa"):
            r = self._run(["-c", script, v], EF2_MK="1", EF2_MSA="t12")        # a caller's switches are dropped, never re-exported
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            out = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(out["active"])
            self.assertEqual(out["server_line"], "opt7x")
            self.assertEqual(out["env"], {"EF2_MK": None, "EF2_MSA": None, "EF2_W4": None})
            from esmfold2_opt import registry
            self.assertEqual(out["not_for_variant"], [n for n in out["levers"] + out["not_for_variant"] if not registry.acts_on(registry.LEVERS[n], v)][:len(out["not_for_variant"])])
            self.assertEqual(set(out["not_for_variant"]), {n for n in ("mh", "trimul", "glue", "rg", "t15msa", "xtr") if n in out["levers"] + out["not_for_variant"] and not registry.acts_on(registry.LEVERS[n], v)})   # the variant-scoped levers: the MSA-module ones (mh, trimul, glue, t15msa, xtr) on the Full model, rg on the Fast model
            self.assertIn(f"[esmfold2-opt] ACTIVE mode=exact variant={v} server_mode=opt7x ", r.stderr)
            seen.append(sorted(out["levers"] + out["not_for_variant"]))
        self.assertEqual(seen[0], seen[1]); self.assertEqual(seen[1], seen[2])                 # one lever SET on every variant; the registry names which of them act per variant
        self.assertNotIn("mk", seen[0]); self.assertNotIn("msa", seen[0]); self.assertIn("fused", seen[0])

    def test_check_fails_loudly_on_a_data_path_that_does_not_exist(self):
        """An image may bake another launcher's data layout (HF_HOME / ESMCFOLD_CCD_PATH) that a plain sandbox lacks: `check` must fail,
        naming the variable and the path, instead of resolving quietly and failing at load; pred --mode off refuses the same way."""
        r = self._run(["-m", "esmfold2_opt", "check", "--mode", "fast", "--variant", "fast"], HF_HOME="/state/weights/esmfold2/hf")
        self.assertEqual(r.returncode, 3, r.stderr[-1500:])                                   # check: activation would refuse -> NOT ACTIVE, the running verb's code
        self.assertIn("would_refuse='HF_HOME=/state/weights/esmfold2/hf does not exist", r.stderr)
        r = self._run(["-m", "esmfold2_opt", "check", "--mode", "exact", "--variant", "full_msa"], ESMCFOLD_CCD_PATH="/state/weights/esmfold2/ccd.pkl")
        self.assertEqual(r.returncode, 3, r.stderr[-1500:])
        self.assertIn("ESMCFOLD_CCD_PATH=/state/weights/esmfold2/ccd.pkl does not exist", r.stderr)
        r = self._run(["-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items, "--out_dir", os.path.join(self.tmp, "dp")],
                      HF_HOME="/state/weights/esmfold2/hf")
        self.assertEqual(r.returncode, 3, r.stderr[-1500:])                                   # the stock route refuses by name too: the kit's NOT ACTIVE line, exit 3
        self.assertIn("[esmfold2-opt] NOT ACTIVE: HF_HOME=/state/weights/esmfold2/hf does not exist", r.stderr)
        r = self._run(["-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items, "--out_dir", os.path.join(self.tmp, "dp2")], HF_HOME="")
        self.assertEqual(r.returncode, 3, r.stderr[-1500:])                                   # unset (empty): refused by name — never the library's default cache
        self.assertIn("[esmfold2-opt] NOT ACTIVE: HF_HOME is not set", r.stderr)

    def test_flag_env_disagreement_is_refused(self):
        r = self._run(["-m", "esmfold2_opt", "check", "--mode", "fast", "--variant", "fast"], ESMFOLD2_VARIANT="full_msa")
        self.assertEqual(r.returncode, 3)
        self.assertIn("disagrees with ESMFOLD2_VARIANT", r.stderr)
        r = self._run(["-m", "esmfold2_opt", "pred", "--mode", "fast", "--variant", "fast", "--input", self.items, "--out_dir", os.path.join(self.tmp, "x")],
                      ESMFOLD2_VARIANT="full_msa")
        self.assertEqual(r.returncode, 2)
        self.assertIn("disagrees", r.stderr)

    def test_pred_kit_mode_applies_eagerly_and_writes_the_schema(self):
        out = os.path.join(self.tmp, "out_fast")
        base = ["-m", "esmfold2_opt", "pred", "--mode", "fast", "--variant", "full_msa", "--input", self.items, "--device", "cpu", "--seeds", "0,1", "--num_loops", "10", "--num_sampling_steps", "68", "--num_diffusion_samples", "1", "--msa_max_depth", "2048", "--remove_insertions", "true", "--max_sequences", "2048"]
        r = self._run(base + ["--out_dir", out])                                              # the stub kit records a complete application of the mode's set: exit 0, fallbacks=none
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn(" fallbacks=none ", r.stderr); self.assertNotIn("NOT ACTIVE: partial", r.stderr); self.assertNotIn("pred INCOMPLETE", r.stderr)
        self.assertIn("[esmfold2-opt] APPLIED model#0 msa_encoder mode=fast variant=full_msa server_mode=opt14_msa", r.stderr)
        files = sorted(os.listdir(os.path.join(out, "cif_all")))
        self.assertEqual([f for f in files if f.endswith(".cif")], ["p1__full__s0_x0.cif", "p1__full__s1_x0.cif"])   # the driver's file set
        self.assertIn("[esmfold2-opt] ready variant=full_msa t=", r.stderr)
        self.assertIn("[esmfold2-opt] ACTIVE mode=fast variant=full_msa server_mode=opt14_msa ", r.stderr)
        active = [l for l in r.stderr.splitlines() if l.startswith("[esmfold2-opt] ACTIVE ")][0]
        self.assertRegex(active, r" levers=[a-z0-9,]*fused"); self.assertIn(" applied=", active)
        self.assertEqual(sorted(os.listdir(out)), ["cif_all", "pred_rows.jsonl"])                     # the outputs and nothing else
        self.assertIn("[esmfold2-opt] APPLIED model#0 msa_encoder mode=fast variant=full_msa server_mode=opt14_msa levers=", r.stderr)   # the application record, printed once per model
        self.assertEqual(r.returncode, 0); self.assertIn(" fallbacks=none ", r.stderr); self.assertNotIn("[esmfold2-opt] gated:", r.stderr)
        rows = [json.loads(l) for l in open(os.path.join(out, "pred_rows.jsonl"))]
        self.assertEqual([(r["complex_id"], r["seed"], r["variant"], r["mode"]) for r in rows], [("p1", 0, "full", "opt14_msa"), ("p1", 1, "full", "opt14_msa")])
        self.assertIn("iptm", rows[0]); self.assertNotIn("binder_chain_worst", rows[0]); self.assertEqual(len(rows), 2); self.assertEqual(len(files), 4)

    def test_pred_off_and_kit_mode_write_the_same_file_set(self):
        from esmfold2_opt import outputs
        o1, o2 = os.path.join(self.tmp, "same_off"), os.path.join(self.tmp, "same_fast")
        for mode, out in (("off", o1), ("fast", o2)):
            r = self._run(["-m", "esmfold2_opt", "pred", "--mode", mode, "--variant", "fast", "--input", self.items, "--out_dir", out, "--device", "cpu", "--seeds", "0"])   # the stub records a complete application under a kit mode
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(sorted(outputs.listing(o1)), sorted(outputs.listing(o2)))
        self.assertEqual(outputs.listing(o1), outputs.listing(o2))   # the stub model is the same in both arms: identical bytes

    def test_autoload_requires_a_variant_and_fires_on_the_trigger(self):
        script = "import transformers.models.esmfold2; print('REACHED')"
        r = self._run(["-c", script], ESMFOLD2_OPT="fast")
        self.assertEqual(r.returncode, 3, r.stderr[-2000:])
        self.assertIn("NOT ACTIVE: a variant is required", r.stderr); self.assertNotIn("REACHED", r.stdout)
        probe = "import importlib.util as u; assert u.find_spec('transformers.models.esmfold2') is not None; " + script
        r = self._run(["-c", probe], ESMFOLD2_OPT="fast")                          # a bare find_spec probe leaves the finder armed
        self.assertEqual(r.returncode, 3, r.stderr[-2000:])
        self.assertIn("NOT ACTIVE: a variant is required", r.stderr); self.assertNotIn("REACHED", r.stdout)
        r = self._run(["-c", script + "; import esmfold2_opt, json; print(json.dumps(esmfold2_opt.status()['active']))"], ESMFOLD2_OPT="exact", ESMFOLD2_VARIANT="fast")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("REACHED", r.stdout); self.assertIn("true", r.stdout.lower())
        self.assertIn("ACTIVE mode=exact variant=fast", r.stderr)
        r = self._run(["-c", script], ESMFOLD2_OPT="off")
        self.assertEqual(r.returncode, 0); self.assertNotIn("[esmfold2-opt]", r.stderr)

    def test_late_activation_rule(self):
        """enable() is allowed any time after the upstream packages are imported; refused by name once a model instance exists, or once
        the kit's own lever modules report a lever applied; status() says so before any activation."""
        script = textwrap.dedent('''
            import json, sys
            import esmfold2_opt
            from esmfold2_opt import stack
            out = {"status_before": esmfold2_opt.status()}
            import transformers.models.esmfold2.modeling_esmfold2 as M          # importing upstream first is allowed
            out["instances_after_import"] = stack.model_instances()
            m = M.ESMFold2Model.from_pretrained("biohub/ESMFold2")              # a model before activation -> refused
            out["instances"] = stack.model_instances()
            r = esmfold2_opt.enable("fast", "fast")
            out["r_model"] = {k: r.get(k) for k in ("active", "reason")}
            del m
            import gc; gc.collect()
            out["instances_after_del"] = stack.model_instances()
            import types
            w4 = types.ModuleType("ef2_w4"); w4._STATE = dict(enabled=True); sys.modules["ef2_w4"] = w4   # the kit says a lever is on
            r = esmfold2_opt.enable("fast", "fast")
            out["r_kit"] = {k: r.get(k) for k in ("active", "reason")}
            out["kit_applied"] = stack.kit_levers_applied()
            del sys.modules["ef2_w4"]
            r = esmfold2_opt.enable("fast", "fast")                             # nothing in the way now: active
            out["r_ok"] = {k: r.get(k) for k in ("active", "mode", "variant", "server_mode")}
            m2 = M.ESMFold2Model.from_pretrained("biohub/ESMFold2")             # after activation: counted by the constructor wrap
            out["check_after"] = stack.instance_check()
            print(json.dumps(out, default=str))
        ''')
        r = self._run(["-c", script])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["status_before"]["active"], False); self.assertIn("reason", out["status_before"])
        self.assertEqual(out["instances_after_import"], 0); self.assertEqual(out["instances"], 1)
        self.assertFalse(out["r_model"]["active"]); self.assertIn("1 ESMFold2Model instance(s) already exist in this process (counted)", out["r_model"]["reason"])
        self.assertEqual(out["instances_after_del"], 0)
        self.assertFalse(out["r_kit"]["active"]); self.assertIn("kit levers already applied", out["r_kit"]["reason"]); self.assertEqual(out["kit_applied"], ["ef2_w4"])
        self.assertEqual(out["r_ok"], {"active": True, "mode": "fast", "variant": "fast", "server_mode": "opt14_msa"})
        self.assertEqual(out["check_after"], {"n": 1, "method": "counted", "built": 1})


if __name__ == "__main__":
    unittest.main()
