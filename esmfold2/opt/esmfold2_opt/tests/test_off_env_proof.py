"""Mode off is stock: the subprocess the package launches carries no kit variable (every stock/PINS.json must-be-absent prefix and the
package's own switches are stripped even when the caller has them set), runs the stock caller under ``python -s``, and the caller
proves its own environment before importing torch — with a stub upstream the whole off-mode path runs on CPU end to end, writes the
one output schema and the printed ENV-CLEAN / settings / SETTINGS lines (no side file), and an injected kit variable makes it refuse."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from esmfold2_opt import cli, stack, stock_fold
from esmfold2_opt.tests import _stubs
from esmfold2_opt.tests._stubs import read_stub_record


class TestOffEnvProof(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        cls.pins = _stubs.require_pins()
        cls.tmp = tempfile.mkdtemp()
        cls.site = _stubs.write_stub_upstream(cls.tmp, cls.pins)
        cls.tree = _stubs.write_stub_tree(cls.tmp, cls.pins)
        cls.items = os.path.join(cls.tmp, "items.json")
        with open(cls.items, "w") as fh:
            json.dump([{"id": "k1", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVL", "msa": None}, {"type": "protein", "id": "B", "sequence": "MKVLAG", "msa": None}], "seeds": [0, 1]}], fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _pred_env(self, **extra):
        self.record = os.path.join(self.tmp, f"stub_record_{self.id().rsplit('.', 1)[-1]}_{len(os.listdir(self.tmp))}.jsonl")   # what reaches the stub upstream (tests/_stubs STUB_RECORD)
        env = _stubs.env_for_stub(self.site, self.kit, self.tree, STUB_RECORD=self.record, **extra)
        return env

    def test_stock_command_strips_every_kit_variable(self):
        from esmfold2_opt import settings
        st = settings.resolve(pins=self.pins)
        polluted = dict(os.environ, EF2_W4="t6", EF2_MK="1", ESMFOLD2_OPT="fast", ESMFOLD2_VARIANT="fast", ESMFOLD2_OPT_FORCE="1",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8", ESMFOLD2_DETERMINISTIC_SCATTER="1", HF_HOME="/w", KEEP_ME="1")
        old = os.environ.copy()
        os.environ.clear(); os.environ.update(polluted)
        try:
            a = cli.pred_parser().parse_args(["--mode", "off", "--variant", "fast", "--input", self.items, "--out_dir", "o"])
            cmd, env = cli.stock_command(a, st, self.pins)
        finally:
            os.environ.clear(); os.environ.update(old)
        absent = stack.stock_env_absent(self.pins)
        self.assertIn("EF2_", absent)
        for k in env:
            self.assertFalse(any((k.startswith(s) if s.endswith("_") else k == s) for s in absent), f"{k} leaked into the stock environment")
        self.assertEqual(env.get("HF_HOME"), "/w"); self.assertEqual(env.get("KEEP_ME"), "1")   # upstream's own variables pass through
        self.assertEqual(cmd[:4], [sys.executable, "-s", "-m", "esmfold2_opt.stock_fold"])
        self.assertIn("--env-absent", cmd)
        self.assertEqual(cmd[cmd.index("--env-absent") + 1], ",".join(absent))
        self.assertEqual(cmd[cmd.index("--hf-repo") + 1], self.pins["variants"]["fast"]["hf_repo"])

    def test_kit_era_data_path_names_are_stripped_not_mapped(self):
        """EF2_HF_HOME / EF2_CCD_PATH are kit-era spellings: stripped from the stock process like every EF2_* name, never mapped; the
        library's own HF_HOME / ESMCFOLD_CCD_PATH pass through as set."""
        from esmfold2_opt import settings
        st = settings.resolve(pins=self.pins)
        old = os.environ.copy()
        try:
            os.environ.clear(); os.environ.update(dict(old, EF2_HF_HOME="/kit/hf", EF2_CCD_PATH="/kit/ccd.pkl", HF_HOME="/mine")); os.environ.pop("ESMCFOLD_CCD_PATH", None)
            a = cli.pred_parser().parse_args(["--mode", "off", "--variant", "fast", "--input", self.items, "--out_dir", "o"])
            _, env = cli.stock_command(a, st, self.pins)
            self.assertEqual((env.get("HF_HOME"), env.get("ESMCFOLD_CCD_PATH")), ("/mine", None))
            self.assertFalse([k for k in env if k.startswith("EF2_")], "EF2_* must not reach the stock process")
        finally:
            os.environ.clear(); os.environ.update(old)
    def test_env_proof_refuses_pollution(self):
        stock_fold.env_proof(("EF2_", "FOO"), environ={"HF_HOME": "/x"}, modules={}, path=[])
        with self.assertRaises(RuntimeError):
            stock_fold.env_proof(("EF2_",), environ={"EF2_W4": "t6"}, modules={}, path=[])
        with self.assertRaises(RuntimeError):
            stock_fold.env_proof(("EF2_",), environ={}, modules={"ef2_opt": object()}, path=[])
        with self.assertRaises(RuntimeError):
            stock_fold.env_proof(("EF2_",), environ={}, modules={}, path=[os.path.join(self.kit, "driver")])

    def test_off_mode_end_to_end_with_stub_upstream(self):
        out = os.path.join(self.tmp, "out_off")
        env = self._pred_env(EF2_W4="t6,t3,t5", ESMFOLD2_OPT="fast", ESMFOLD2_VARIANT="fast", CUBLAS_WORKSPACE_CONFIG=":4096:8")   # a polluted caller
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items,
                            "--out_dir", out, "--device", "cpu", "--num_loops", "10", "--num_sampling_steps", "68", "--num_diffusion_samples", "1", "--msa_max_depth", "2048", "--remove_insertions", "true", "--max_sequences", "2048"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("[esmfold2-opt stock] ENV-CLEAN ok", r.stderr)
        self.assertIn("NOT ACTIVE: mode off", r.stderr)
        env_clean = [l for l in r.stderr.splitlines() if l.startswith("[esmfold2-opt stock] ENV-CLEAN ok: absent=EF2_,")]      # python -s, nothing of the kit: the printed proof
        self.assertEqual(len(env_clean), 1, r.stderr[-2000:]); self.assertTrue(env_clean[0].endswith(" kit_modules=none kit_dirs=none no_user_site=True"), env_clean[0])
        rec = read_stub_record(self.record)
        self.assertEqual([x["repo"] for x in rec if x["event"] == "load"], [self.pins["variants"]["fast"]["hf_repo"]])
        self.assertIn("[esmfold2-opt] settings num_loops=10 num_sampling_steps=68 num_diffusion_samples=1 msa_max_depth=2048 ", r.stderr)   # the resolved dims, printed
        self.assertIn(" msa_remove_insertions=True msa_read_depth=2048 ", r.stderr); self.assertNotIn("preset", r.stderr.split("[esmfold2-opt] settings ", 1)[1].split("\n", 1)[0])
        folds = [x for x in rec if x["event"] == "fold"]
        self.assertEqual([(x["num_loops"], x["num_sampling_steps"], x["msa_max_depth"]) for x in folds], [(10, 68, 2048)] * 2)   # what reached fold()
        files = sorted(os.listdir(os.path.join(out, "cif_all")))                 # the driver's file set, through the kit's own writer
        for s in (0, 1):
            for suffix in (".cif", "_pae.npz"):
                self.assertIn(f"k1__fast__s{s}_x0{suffix}", files)
        rows = [json.loads(l) for l in open(os.path.join(out, "pred_rows.jsonl"))]
        self.assertEqual([(r["complex_id"], r["seed"], r["variant"], r["mode"], r["tag"]) for r in rows], [("k1", 0, "fast", "off", "pred"), ("k1", 1, "fast", "off", "pred")])
        self.assertEqual((rows[0]["pred_file"], rows[0]["status"]), ("k1__fast__s0_x0.cif", "ok")); self.assertFalse([k for k in rows[0] if k.startswith(("ipsae", "binder", "plddt_"))])
        self.assertFalse(os.path.exists(os.path.join(out, "staged")))              # the staging directory is gone after finalisation
        self.assertIn("[esmfold2-opt] ready variant=fast t=", r.stderr)             # the one 'model loaded' line, from the stock caller
        self.assertEqual(sorted(os.listdir(out)), ["cif_all", "pred_rows.jsonl"])              # the outputs and nothing else: no manifest, no proof file
        self.assertIn("[esmfold2-opt stock] DONE predictions=2 items=1", r.stderr)

    def test_off_mode_identical_outputs_across_runs(self):
        env = self._pred_env()
        outs = []
        from esmfold2_opt import outputs
        for i in range(2):
            out = os.path.join(self.tmp, f"out_rep{i}")
            r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "full_nomsa", "--input", self.items,
                                "--out_dir", out, "--device", "cpu", "--seeds", "0"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            outs.append(outputs.listing(out))
        self.assertEqual(outs[0], outs[1])
        from esmfold2_opt import settings
        n = settings.resolve(pins=self.pins).num_diffusion_samples                          # the library default's samples per seed
        self.assertEqual(sorted(k for k in outs[0] if k.startswith("cif_all")), sorted(f"cif_all/k1__full__s0_x{i}{s}" for i in range(n) for s in (".cif", "_pae.npz")))
        self.assertEqual(len([k for k in outs[0] if k.startswith("row:")]), n)

    def test_stock_caller_refuses_an_injected_kit_variable(self):
        out = os.path.join(self.tmp, "out_refuse")
        env = self._pred_env(EF2_MK="1")
        from esmfold2_opt import settings
        st = settings.resolve(pins=self.pins)
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt.stock_fold", "--variant", "fast", "--hf-repo", "x/y", "--input", self.items,
                            "--out_dir", out, "--settings-json", json.dumps(st.as_dict()), "--env-absent", "EF2_", "--device", "cpu"],
                           env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("NOT STOCK", r.stderr)
        self.assertIn("EF2_MK", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_det_flag_reaches_the_stock_arm(self):
        out = os.path.join(self.tmp, "out_det")
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items,
                            "--out_dir", out, "--device", "cpu", "--det", "1", "--seeds", "0"], env=self._pred_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        fold = [x for x in read_stub_record(self.record) if x["event"] == "fold"][-1]      # the state AT the stock fold() call
        self.assertEqual((fold["cublas_workspace"], fold["scatter_switch"], fold["deterministic_algorithms"]), (":4096:8", "1", [True, True]))
        self.assertEqual((fold["lm_dropout"], fold["msa_column_mask_rate"], fold["seed"]), (0.0, 0.0, 0))   # the recipe's fold keywords + the seed
        load = [x for x in read_stub_record(self.record) if x["event"] == "load"][-1]
        self.assertIsNone(load["config"])                                                     # level 1 passes no config: the shipped one loads untouched
        lib = self.pins["library_defaults"]["model"]                                       # no --backend: the library's own two values
        self.assertIn("SETTINGS backend=shipped model_calls=none ", r.stderr)                 # bare --mode off: no model call, the model as loaded

    def test_det_level_2_sets_the_config_switch_before_load(self):
        """--det 2 = level 1 + lm_encoder.lm_dropout=0 on the model config before the weights load (the stub records the config it was given)."""
        out = os.path.join(self.tmp, "out_det2")
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items,
                            "--out_dir", out, "--device", "cpu", "--det", "2", "--seeds", "0"], env=self._pred_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        rec = read_stub_record(self.record)
        load = [x for x in rec if x["event"] == "load"][-1]; fold = [x for x in rec if x["event"] == "fold"][-1]
        self.assertEqual(load["config"], {"repo": self.pins["variants"]["fast"]["hf_repo"], "lm_encoder.lm_dropout": 0.0, "lm_encoder.per_loop_lm_dropout": True})   # set on the config BEFORE the weights load
        self.assertEqual((fold["cublas_workspace"], fold["deterministic_algorithms"], fold["lm_dropout"]), (":4096:8", [True, True], 0.0))
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items,
                            "--out_dir", out + "b", "--device", "cpu", "--det", "3", "--seeds", "0"], env=self._pred_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_backend_shipped_makes_no_model_call(self):
        """--backend shipped = NO model call: the model exactly as from_pretrained loads it; the propinneds model_calls [] and the stub's
        as-constructed switches untouched; the SETTINGS line names backend=shipped and model_calls=none."""
        out = os.path.join(self.tmp, "out_shipped")
        r = subprocess.run([sys.executable, "-m", "esmfold2_opt", "pred", "--mode", "off", "--variant", "fast", "--input", self.items,
                            "--out_dir", out, "--device", "cpu", "--backend", "shipped", "--seeds", "0"], env=self._pred_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("SETTINGS backend=shipped model_calls=none kernel_backend=None chunk=64 opm_chunk=unread ", r.stderr); self.assertNotIn("preset=", r.stderr)   # no setter ran: as loaded
        fold = [x for x in read_stub_record(self.record) if x["event"] == "fold"][-1]
        self.assertEqual((fold["kernel_backend"], fold["chunk"]), (None, 64))


if __name__ == "__main__":
    unittest.main()
