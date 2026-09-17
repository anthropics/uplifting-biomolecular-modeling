"""The driver over BindCraft's own design step (bindcraft.py), on the stand-in stack: the stock arm end to end (settings line, the four
phases in the observer's rows, BindCraft's terminate verdict as data, the outputs beside BindCraft's own tree); PARITY — the transcript of
ColabDesign calls our arm makes == the transcript BindCraft's `binder_hallucination` makes when called directly with bindcraft.py's own
derivation, and both equal the recipe the settings file states; LEVER TRANSPARENCY — the kit arm (fast) drives the identical calls and
writes byte-identical records; a gate-terminated trajectory is a complete run recorded by name; the named refusals; the import boundary
(no top-level `functions`, PyRosetta names refuse by name)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from . import _stubs
from colabdesign_opt import bindcraft, names, report, settings


def _probe(out):
    """The arm probe's record for a run under `out` (tests/_stubs.ARM_PROBE): {"rows": the observer's rows, "xla_flags"}."""
    return _stubs.read_probe(out + "_probe.json")


def _steps(out):
    return _probe(out)["rows"]


def _runs(out):
    """The run records the design script's bindcraft.run_design calls returned under `out`, by seed in run order (captured test-side by tests/_stubs.ARM_PROBE)."""
    return _probe(out)["runs"]


def _numeric_rows(rows):
    """The observer's rows without the clock (wall_s): what must be identical between arms on the stand-in."""
    return [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf(_stubs.bindcraft_stack_missing(), _stubs.SKIP_BINDCRAFT)
class TestBindcraftDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_bc_")
        cls.site = _stubs.make_stub_stack(cls.tmp)
        cls.target = _stubs.write_target(os.path.join(cls.tmp, "t.pdb"), ("A",), 40)
        cls.params = os.path.join(cls.tmp, "params"); os.makedirs(os.path.join(cls.params, "params"))
        for i in range(1, 6):
            open(os.path.join(cls.params, "params", f"params_model_{i}_multimer_v3.npz"), "w").close()
        cls.env = _stubs.child_env(cls.site)
        cls.S = settings.load(_stubs.TREE)
        cls.adv = cls.S["advanced"]

    def design_args(self, out, binder_len=40, seed=0, hotspot="5,7"):
        seed_args = ["--seed", str(seed)]
        return ["--", "--starting-pdb", self.target, "--chains", "A", "--binder-len", str(binder_len), *(["--target-hotspot-residues", hotspot] if hotspot else []), *seed_args,
                "--params-dir", self.params, "--out", out]

    def stock(self, out, **kw):
        env = dict(self.env, STUB_CALLS_JSON=os.path.join(out + "_calls.json"), **kw.pop("env", {}))
        cmd = _stubs.arm_cmd("stock", out + "_probe.json", ["--pins", _stubs.PINS, *self.design_args(out, **kw)])
        return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=self.tmp)

    def kit(self, out, mode, extra=(), **kw):
        env = dict(self.env, STUB_CALLS_JSON=os.path.join(out + "_calls.json"), **kw.pop("env", {}))
        cmd = _stubs.arm_cmd("kit", out + "_probe.json", ["--pins", _stubs.PINS, "--mode", mode, *extra, *self.design_args(out, **kw)])
        return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=self.tmp)

    def calls(self, out):
        with open(out + "_calls.json") as fh:
            return json.load(fh)

    # ---- the recipe the settings file states, as ColabDesign calls (BindCraft colabdesign_utils.py binder_hallucination, 4stage)
    def expected_calls(self, binder_len=40, seed=0, hotspot="5,7"):
        a = self.adv
        models = [0, 1, 2, 3, 4] if a["use_multimer_design"] else [0, 1]
        common = {"models": models, "num_models": 1, "sample_models": a["sample_models"], "save_best": True}
        import math
        return [("mk_afdesign_model", {"protocol": "binder", "use_multimer": a["use_multimer_design"], "num_recycles": a["num_recycles_design"], "best_metric": "loss", "debug": False, "data_dir": self.params}),
                ("prep_inputs", {"chain": "A", "binder_len": binder_len, "pdb_filename": self.target, "hotspot": hotspot, "seed": seed, "rm_aa": a["omit_AAs"],
                                 "rm_target_seq": a["rm_template_seq_design"], "rm_target_sc": a["rm_template_sc_design"]}),
                ("restart", {"seed": seed, "opt": None, "weights": None, "rm_aa": a["omit_AAs"]}),
                ("design_logits", {"iters": 50, "e_soft": 0.9, **common}),
                ("design_logits", {"iters": a["soft_iterations"] - 50, "e_soft": 1, **common, "ramp_recycles": False}),
                ("design_soft", {"iters": a["temporary_iterations"], "temp": 1, "e_temp": 1e-2, **common, "ramp_recycles": False}),
                ("design_hard", {"iters": a["hard_iterations"], "temp": 1e-2, **common, "dropout": False, "ramp_recycles": False}),
                ("design_pssm_semigreedy", {"soft_iters": 0, "hard_iters": a["greedy_iterations"], "tries": math.ceil(binder_len * (a["greedy_percentage"] / 100)), "e_tries": None,
                                            "ramp_recycles": True, "ramp_models": False, **common})]

    def assertCalls(self, got, want):
        got = [(n, {k: v for k, v in kw.items()}) for n, kw in got]
        self.assertEqual([n for n, _ in got], [n for n, _ in want])
        for (n, g), (_, w) in zip(got, want):
            for k, v in w.items():
                self.assertIn(k, g, (n, k)); self.assertEqual(g[k], v, (n, k, g))

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_stock_arm_end_to_end(self):
        out = os.path.join(self.tmp, "off")
        r = self.stock(out)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn(f"[colabdesign-opt] SETTINGS {settings.NAME} file={settings.SETTINGS_RELPATH} sha256={self.S['sha256'][:16]} design_models=[0, 1, 2, 3, 4] helicity={self.adv['weights_helicity']} hotspot=5,7 trajectory=design_l40_s0", r.stderr)
        for stage in ("Stage 1: Test Logits", "Stage 2: Softmax Optimisation", "Stage 3: One-hot Optimisation", "Stage 4: PSSM Semigreedy Optimisation", "Trajectory successful"):
            self.assertIn(stage, r.stdout)                                            # BindCraft's own words: its function ran
        rows = _steps(out); end = rows[-1]
        self.assertEqual(_probe(out)["callback"], [[r["i"], r] for r in rows[:-1]])                  # the one optional step callback saw every row as it was recorded, in order, and nothing else
        it = settings.iteration_counts(self.adv)
        self.assertEqual(end["kind"], "end"); self.assertEqual(end["phase"], "complete"); self.assertEqual(end["terminate"], {"verdict": "", "gate": None})
        self.assertEqual(end["n_steps"], {"soft": it["soft"], "temp": it["temp"], "hard": it["hard"], "greedy_rounds": it["greedy"], "greedy_forward": it["greedy"] + 1})
        self.assertEqual(end["phases_run"], ["design_logits:1", "design_logits:2", "design_soft:1", "design_hard:1", "design_pssm_semigreedy:1"])
        grads = [x for x in rows if x["kind"] == "grad"]
        self.assertEqual([x["phase"] for x in grads], ["soft"] * it["soft"] + ["temp"] * it["temp"] + ["hard"] * it["hard"])
        self.assertEqual([x["i"] for x in rows], list(range(len(rows)))); self.assertEqual([x["k"] for x in grads], list(range(len(grads))))
        self.assertEqual([x["call"] for x in rows if x.get("first_call")], end["phases_run"])
        for key in ("loss", "plddt", "i_ptm", "rg", "helix", "soft", "temp", "hard", "dropout", "recycles", "models", "wall_s"):
            self.assertIn(key, grads[0])
        self.assertEqual({x["dropout"] for x in grads if x["phase"] == "hard"}, {False}); self.assertEqual(grads[-1]["hard"], 1.0)
        fw = [x for x in rows if x["kind"] == "forward"]; self.assertTrue(all(x["phase"] == "greedy" for x in fw))
        for n in names.OUTPUTS:
            self.assertTrue(os.path.isfile(os.path.join(out, n)), n)
        self.assertEqual(sorted(f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))), sorted([*names.OUTPUTS, bindcraft.FAILURE_CSV]))   # the outputs and BindCraft's own csv: no other file
        run = _runs(out)["0"]
        self.assertEqual(run["settings"], settings.record(self.S)); self.assertEqual(run["terminate"], end["terminate"]); self.assertEqual(run["n_steps"], end["n_steps"])
        printed = report.run_lines(r.stderr.splitlines())                              # the arm's [run] line read back == the run record it renders (the caller's only channel)
        self.assertEqual(len(printed), 1); self.assertEqual((printed[0]["tag"], printed[0]["tokens"], printed[0]["steps"], printed[0]["n_steps"], printed[0]["terminate"]), ("design_l40_s0", run["tokens"], run["steps"], run["n_steps"], "complete"))
        self.assertEqual(printed[0]["design_pdb_sha256_16"], names.sha256_file(os.path.join(out, names.DESIGN_PDB))[:16])
        self.assertEqual(run["hotspot"], "5,7"); self.assertEqual(run["tokens"], 80); self.assertEqual(run["steps"], it["soft"] + it["temp"] + it["hard"])
        a = self.adv                                                                    # BindCraft's weights update, from the file's keys, applied by ITS code
        self.assertEqual({k: run["weights"][k] for k in ("plddt", "pae", "i_pae", "con", "i_con", "i_ptm", "rg", "helix")},
                         {"plddt": a["weights_plddt"], "pae": a["weights_pae_intra"], "i_pae": a["weights_pae_inter"], "con": a["weights_con_intra"], "i_con": a["weights_con_inter"],
                          "i_ptm": a["weights_iptm"], "rg": a["weights_rg"], "helix": a["weights_helicity"]})
        self.assertEqual(run["con"], {"num": a["intra_contact_number"], "cutoff": a["intra_contact_distance"], "binary": False, "seqsep": 9})
        self.assertEqual(run["i_con"], {"num": a["inter_contact_number"], "cutoff": a["inter_contact_distance"], "binary": False})
        self.assertEqual(sorted(run["bindcraft"]), ["commit_dir", "design_paths", "failure_csv"])                       # no file inventory in the record
        self.assertTrue(os.path.isfile(os.path.join(out, "Trajectory", "design_l40_s0.pdb"))); self.assertTrue(os.path.isfile(os.path.join(out, bindcraft.FAILURE_CSV)))
        self.assertCalls(self.calls(out)[0], self.expected_calls())                   # the ColabDesign calls our arm made == the 4stage recipe of the file

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_parity_with_bindcraft_called_directly(self):
        """binder_hallucination called DIRECTLY (bindcraft.py's derivation, no driver) makes the same ColabDesign calls and logs the same trajectory
        as our stock arm: the driver adds records, not behaviour."""
        out_arm = os.path.join(self.tmp, "parity_arm"); r = self.stock(out_arm)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out_dir = os.path.join(self.tmp, "parity_direct"); os.makedirs(out_dir)
        code = f"""
import json, math, os, sys
from colabdesign_opt import bindcraft, settings
tree = {_stubs.TREE!r}
S = settings.load(tree); F = bindcraft.functions(tree); G = F.generic_utils
adv = G.perform_advanced_settings_check(dict(S["advanced"]), bindcraft.bindcraft_dir(tree)); adv["af_params_dir"] = {self.params!r}
design_models = G.load_af2_models(adv["use_multimer_design"])[0]
paths = G.generate_directories({out_dir!r}); fcsv = os.path.join({out_dir!r}, "failure_csv.csv"); G.generate_filter_pass_csv(fcsv, settings.filters_path(tree))
af = F.colabdesign_utils.binder_hallucination("design_l40_s0", {self.target!r}, "A", "5,7", 40, 0, G.load_helicity(adv), design_models, adv, paths, fcsv)
json.dump({{"log": af._tmp["log"], "terminate": af.aux["log"].get("terminate", "")}}, open(os.path.join({out_dir!r}, "direct.json"), "w"), default=str)
"""
        r2 = subprocess.run([sys.executable, "-s", "-c", code], env=dict(self.env, STUB_CALLS_JSON=out_dir + "_calls.json"), capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
        self.assertEqual(self.calls(out_dir)[0], self.calls(out_arm)[0])                # identical transcripts, kwargs and all
        direct = json.load(open(os.path.join(out_dir, "direct.json")))
        arm_log = [json.loads(l) for l in open(os.path.join(out_arm, names.TRAJECTORY))]
        self.assertEqual(len(direct["log"]), len(arm_log))
        self.assertEqual([round(x["loss"], 6) for x in direct["log"]], [round(x["loss"], 6) for x in arm_log])
        self.assertEqual(direct["terminate"], "")

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_kit_arm_levers_step_aside_by_name_on_the_stand_in(self):
        """The kit arm RUNS on the stand-in stack — the kit rule: a lever that cannot engage here steps aside BY NAME and the design runs with the
        rest: `fast`'s Pallas kernels have no GPU lowering on the stub jax (`state=skipped reason=cannot_run`), `hoist_prev` finds the stand-in
        design surface instead of ColabDesign's pinned loop (the same, `exact` and `fast`), proj / txla compose on a kernel that is not
        there (`reason=no_attention_kernel`); nothing is refused and nothing disappears unnamed."""
        out_off = os.path.join(self.tmp, "t_off"); r = self.stock(out_off); self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        r = self.kit(os.path.join(self.tmp, "t_fast"), "fast")
        self.assertEqual(r.returncode, 0, r.stderr[-2500:]); self.assertNotIn("REFUSED", r.stderr)
        self.assertRegex(r.stderr, r"LEVER name=hoist_prev state=skipped reason=cannot_run (\S+ )*detail=HoistPrevError:\S+ source=install"); self.assertRegex(r.stderr, r"LEVER name=proj state=skipped reason=no_attention_kernel ")
        r = self.kit(os.path.join(self.tmp, "t_exact"), "exact")
        self.assertEqual(r.returncode, 0, r.stderr[-2500:]); self.assertNotIn("REFUSED", r.stderr)
        self.assertRegex(r.stderr, r"LEVER name=hoist_prev state=skipped reason=cannot_run (\S+ )*detail=HoistPrevError:\S+ source=install"); self.assertNotIn(" LEVER name=hoist_prev state=on", r.stderr)

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_gate_terminated_trajectory_is_a_complete_run(self):
        """BindCraft's first pLDDT gate stops the trajectory (the stand-in reports 0.5): exit 0, every record present, the end row names the gate."""
        out = os.path.join(self.tmp, "gated")
        r = self.stock(out, env={"STUB_PLDDT": "0.5"})
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("Initial trajectory pLDDT too low", r.stdout)
        end = _steps(out)[-1]
        self.assertEqual(end["kind"], "end"); self.assertTrue(end["phase"].startswith("terminated:"), end)
        self.assertIn("Trajectory_logits_pLDDT", end["terminate"]["gate"]); self.assertEqual(end["terminate"]["verdict"], "LowConfidence")
        self.assertEqual(end["n_steps"], {"soft": 50, "temp": 0, "hard": 0, "greedy_rounds": 0, "greedy_forward": 0}); self.assertEqual(end["phases_run"], ["design_logits:1"])
        for n in names.OUTPUTS:
            self.assertTrue(os.path.isfile(os.path.join(out, n)), n)
        run = _runs(out)["0"]
        self.assertEqual(run["terminate"], end["terminate"]); self.assertEqual(run["steps"], 50)
        self.assertEqual(report.run_lines(r.stderr.splitlines())[0]["terminate"], "terminated:" + end["terminate"]["gate"])   # the [run] line names the gate
        self.assertTrue(os.path.isfile(os.path.join(out, "Trajectory", "LowConfidence", "design_l40_s0.pdb")))   # BindCraft moved its own file; ours stays at design.pdb

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_refusals_by_name(self):
        adv = dict(self.adv, dssp_path="/nonexistent/dssp", design_algorithm="5stage")
        derived = {"design_models": [0, 1, 2, 3, 4]}
        rs = bindcraft.refusals(adv, derived, os.path.join(self.tmp, "no_params"))
        kinds = [x.kind for x in rs]
        self.assertEqual(kinds[:2], ["design_algorithm", "dssp"]); self.assertEqual(kinds.count("params"), 5)
        self.assertIn("knows 2stage|3stage|greedy|mcmc|4stage", str(rs[0])); self.assertIn("optimise_beta=true runs DSSP", str(rs[1]))
        self.assertEqual(bindcraft.refusals(dict(self.adv, dssp_path=_stubs.DSSP), derived, self.params), [])
        # the settings pin re-pointed to another path: refused by name, never read (a copy of the tree's stock/ whose PINS.json names a
        # settings file that is not the one on disk)
        fake = os.path.join(self.tmp, "fake_tree"); os.makedirs(os.path.join(fake, "stock"))
        pins = json.load(open(_stubs.PINS, encoding="utf-8"))
        pins[settings.PIN_KEY]["file"] = pins[settings.PIN_KEY]["file"].replace(settings.NAME, settings.NAME + "_flexible")
        json.dump(pins, open(os.path.join(fake, "stock", "PINS.json"), "w", encoding="utf-8"))
        dst = os.path.join(fake, settings.SETTINGS_RELPATH); os.makedirs(os.path.dirname(dst))
        shutil.copy(settings.path(_stubs.TREE), dst)
        with self.assertRaises(settings.SettingsError) as cm:
            settings.load(fake)
        self.assertIn("expected", str(cm.exception))
        self.assertEqual(settings.load(_stubs.TREE)["sha256"], names.sha256_file(settings.path(_stubs.TREE)))
        # the arm refuses by name (exit 3) when a params file is absent — before any model
        out = os.path.join(self.tmp, "noparams")
        cmd = [sys.executable, "-s", "-m", "colabdesign_opt.stock_launch", "--pins", _stubs.PINS, "--",
               "--starting-pdb", self.target, "--chains", "A", "--binder-len", "40", "--seed", "0", "--params-dir", os.path.join(self.tmp, "nowhere"), "--out", out]
        r = subprocess.run(cmd, env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr[-2000:]); self.assertIn("[colabdesign-opt] REFUSED: params: design model 0", r.stderr); self.assertNotIn("Stage 1", r.stdout)

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_recipe_accessor(self):
        """recipe(): BindCraft's function run on the Recorder in THIS process — the stages / kwargs / weights it returns equal the transcript of the arm."""
        from colabdesign_opt import units
        code = f"""
import json, sys
from colabdesign_opt.tests import recipe
R = recipe.recipe({_stubs.TREE!r}, target={self.target!r}, chain="A", binder_len=40, hotspot="5,7", seed=0)
json.dump({{k: R[k] for k in ("stages", "model", "prep", "weights", "opt", "losses_added", "derived", "conditional")}}, sys.stdout, default=str)
"""
        r = subprocess.run([sys.executable, "-s", "-c", code], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        R = json.loads(r.stdout[r.stdout.index("{"):])
        self.assertEqual([(st["phase"], st["call"]) for st in R["stages"]], [("soft", "design_logits:1"), ("soft", "design_logits:2"), ("temp", "design_soft:1"), ("hard", "design_hard:1"), ("greedy", "design_pssm_semigreedy:1")])
        it = settings.iteration_counts(self.adv)
        self.assertEqual([st["iters"] for st in R["stages"]], [50, it["soft"] - 50, it["temp"], it["hard"], it["greedy"]])
        self.assertTrue(set(st["phase"] for st in R["stages"]) <= set(units.PHASES))
        exp = self.expected_calls()
        self.assertEqual({k: v for k, v in R["model"].items() if k != "data_dir"}, {k: v for k, v in exp[0][1].items() if k != "data_dir"})   # data_dir = recipe()'s own temp params root
        self.assertEqual({k: R["prep"][k] for k in exp[1][1]}, exp[1][1])
        self.assertEqual(R["losses_added"], ["loss_fn", "loss_iptm", "binder_helicity"])                 # rg, i_ptm, helicity — BindCraft's add_* helpers (termini loss off in the file)
        self.assertEqual(R["weights"]["helix"], self.adv["weights_helicity"]); self.assertEqual(R["opt"]["i_con"]["cutoff"], self.adv["inter_contact_distance"])
        out = os.path.join(self.tmp, "recipe_arm"); rr = self.stock(out); self.assertEqual(rr.returncode, 0, rr.stderr[-2000:])
        arm_calls = [c for c in self.calls(out)[0] if c[0] in ("design_logits", "design_soft", "design_hard", "design_pssm_semigreedy")]
        self.assertEqual([json.loads(json.dumps(st["kwargs"], default=str)) for st in R["stages"]],
                         [json.loads(json.dumps({k: v for k, v in kw.items() if k not in ("iters", "hard_iters")}, default=str)) for _, kw in arm_calls])

    def test_import_boundary(self):
        """No top-level `functions` anywhere; BindCraft's modules come through colabdesign_opt.bindcraft_functions from the vendored path, unmodified;
        functions/__init__.py is never executed; a PyRosetta name refuses by name when USED (importing it succeeds, as colabdesign_utils does)."""
        code = f"""
import importlib.util, sys
assert importlib.util.find_spec("functions") is None, "a top-level `functions` module is importable"
from colabdesign_opt import bindcraft
F = bindcraft.functions({_stubs.TREE!r})
import colabdesign_opt.bindcraft_functions as P
assert list(P.__path__) == [F.dir], P.__path__
assert F.colabdesign_utils.__file__ == F.dir + "/colabdesign_utils.py"
assert "pyrosetta" not in sys.modules and importlib.util.find_spec("functions") is None
from colabdesign_opt.bindcraft_functions.pyrosetta_utils import pr_relax      # importing the name succeeds
try:
    pr_relax("x")
except bindcraft.Refusal as e:
    print("REFUSED", e.kind, e)
else:
    raise SystemExit("pr_relax did not refuse")
try:
    F.colabdesign_utils.pr_relax("x")                                          # the name colabdesign_utils itself imported
except bindcraft.Refusal as e:
    print("REFUSED2", e.kind)
"""
        r = subprocess.run([sys.executable, "-s", "-c", code], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("REFUSED pyrosetta BindCraft functions/pyrosetta_utils.py `pr_relax` needs PyRosetta", r.stdout); self.assertIn("REFUSED2 pyrosetta", r.stdout)


class TestDsspExecutableBit(unittest.TestCase):
    def test_a_dssp_that_lost_its_executable_bit_gets_it_back(self):
        """A tree fetched over HTTP drops the exec bit of functions/dssp: the DSSP check sets it back instead of refusing (bindcraft.executable);
        a file that is not there stays refused by name, and the refusal says how to fix it."""
        with tempfile.TemporaryDirectory(prefix="cd_opt_dssp_") as tmp:
            dssp = os.path.join(tmp, "dssp")
            with open(dssp, "w") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(dssp, 0o644)
            self.assertFalse(os.access(dssp, os.X_OK))
            self.assertTrue(bindcraft.executable(dssp))
            self.assertTrue(os.access(dssp, os.X_OK)); self.assertEqual(os.stat(dssp).st_mode & 0o111, 0o111)
            self.assertTrue(bindcraft.executable(dssp))                                  # already executable: unchanged, still True
            missing = os.path.join(tmp, "nowhere", "dssp")
            self.assertFalse(bindcraft.executable(missing))
            rs = bindcraft.refusals({"design_algorithm": "4stage", "optimise_beta": True, "dssp_path": missing}, {"design_models": []}, tmp)
            self.assertEqual([r.kind for r in rs], ["dssp"]); self.assertIn(f"chmod +x {missing}", str(rs[0]))


if __name__ == "__main__":
    unittest.main()
