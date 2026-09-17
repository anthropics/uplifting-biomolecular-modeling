"""The stock arm's environment proof and the CLI contract: both children start from the caller's environment with every kit and
package switch stripped — the stock child (`python -s -m caliby_opt.stock_design`) gets nothing else, the kit child gets the row
and CALIBY_OPT/CALIBY_VARIANT; the default mode (fast on both variants, never exact) when --mode and
CALIBY_OPT are absent; exact on single refused by name with exit 2 before any work; `check` gates the child's environment;
the design knobs map to the writer's own arguments at upstream's defaults; the exit codes (2 usage, 3 not active; the shared core's table); nothing is installed
around the kit child; the stock caller's refusal as a real subprocess and its STOCK path in-process; the exit tally printed at
interpreter exit; run.sh's own exits."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .. import cli, modes, settings, stack, stock_design
from ._fixtures import pythonpath


class TestStockEnv(unittest.TestCase):
    def test_strip_env_removes_every_switch_and_keeps_data_paths(self):
        env = {"CALIBY_FAST_SAMPLER": "2", "CALIBY_X_LCP": "1", "CALIBY_OPT": "exact", "CALIBY_VARIANT": "single", "CALIBY_OPT_HOME": "/x",
               "MODEL_PARAMS_DIR": "/w", "HF_HUB_OFFLINE": "1", "PATH": "/bin", "MODEL_OPT": "/t"}
        out, removed = stack.strip_env(env)
        self.assertEqual(sorted(out), ["HF_HUB_OFFLINE", "MODEL_OPT", "MODEL_PARAMS_DIR", "PATH"])
        self.assertEqual(removed, ["CALIBY_FAST_SAMPLER", "CALIBY_OPT", "CALIBY_OPT_HOME", "CALIBY_VARIANT", "CALIBY_X_LCP"])
        proof = stack.stock_env_proof(out)
        self.assertTrue(proof["clean"])
        self.assertEqual(proof["data_env"], {"MODEL_PARAMS_DIR": "/w", "HF_HUB_OFFLINE": "1"})
        self.assertFalse(stack.stock_env_proof(env)["clean"])
        self.assertEqual(stack.stock_env_proof(env)["present"], removed)

    def test_pins_stock_environment_matches_the_code(self):
        p = stack.pins()["stock_environment"]
        self.assertEqual(tuple(p["must_be_absent_prefixes"]), stack.STOCK_ABSENT_PREFIXES)
        self.assertEqual(tuple(p["reads"]), stack.DATA_ENV)


class TestCliContract(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(stack.STOCK_ABSENT_PREFIXES):
                del os.environ[k]
        self.tmp = tempfile.mkdtemp()
        self.calls = []

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def _fake_child(self, argv, env, n_designs=None):
        """A child that returns 0 and writes the writer's timing.json for the request (inputs x num_seqs_per_pdb designs, or ``n_designs``)."""
        self.calls.append((argv, env))
        w = argv[argv.index("--") + 1:]
        inputs = w[w.index("--inputs") + 1:w.index("--out_dir")]
        want = len(inputs) * int(w[w.index("--num_seqs_per_pdb") + 1] if "--num_seqs_per_pdb" in w else 1)   # not given = upstream's 1
        out = w[w.index("--out_dir") + 1]
        os.makedirs(out, exist_ok=True)
        json.dump({"n_inputs": len(inputs), "n_designs": want if n_designs is None else n_designs}, open(os.path.join(out, "timing.json"), "w"))
        return 0

    def test_stock_child_is_clean_and_uses_the_writer_args(self):
        os.environ["CALIBY_X_LCP"] = "1"                               # a stray switch in the parent must not reach the stock child
        os.environ["CALIBY_OPT"] = "off"
        inp = os.path.join(self.tmp, "a.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", self._fake_child):
            rc = cli.run_design("off", "single", [inp], os.path.join(self.tmp, "out"), seed=11)
        self.assertEqual(rc, 0)
        argv, env = self.calls[0]
        self.assertEqual(argv[:4], [sys.executable, "-s", "-m", "caliby_opt.stock_design"])
        self.assertFalse([k for k in env if k.startswith(stack.STOCK_ABSENT_PREFIXES)])
        self.assertEqual(env["MODEL_OPT"], stack.tree_home())
        w = argv[argv.index("--") + 1:]
        self.assertEqual(w, ["--inputs", inp, "--out_dir", os.path.join(self.tmp, "out"), "--det", "0", "--seed", "11"])   # no knob given: none passed, upstream's defaults apply
        man = json.load(open(os.path.join(self.tmp, "out", "opt_manifest.json")))
        self.assertEqual((man["settings"]["num_seqs_per_pdb"], man["settings"]["model_name"], man["settings"]["given"]), (1, "caliby", {}))
        self.assertNotIn("design_chain", man["settings"])                                     # every chain is designed (upstream's default); no chain knob
        self.assertNotIn("preset", man["settings"])
        self.assertEqual(man["exit_code"], 0)

    def test_exact_child_gets_the_row_and_nothing_is_installed(self):
        """The kit child is one process like the stock child: it activates (proofs, import hook), designs, exits; the parent writes
        nothing into site-packages and drives no install or restore around it."""
        inp = os.path.join(self.tmp, "b.pdb"); open(inp, "w").close()
        os.environ["CALIBY_X_CLEAN"] = "proc"                          # the caller's shell never widens the composition: stripped
        os.environ["CALIBY_FAST_SAMPLER"] = "1"                        # (a conflicting value, likewise stripped, never a refusal here)
        with mock.patch.object(cli, "_run_child", self._fake_child):
            rc = cli.run_design("exact", "ensemble32", [inp], os.path.join(self.tmp, "out2"), seed=12, num_seqs_per_pdb=4, omit_aas="C", pp_batch_size=16)
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 1)                           # one child, no tree subprocesses
        man = json.load(open(os.path.join(self.tmp, "out2", "opt_manifest.json")))
        self.assertNotIn("tree", man)                                  # no install/restore record: there is none
        self.assertEqual(man["schema"], "caliby_opt/opt_manifest/2")
        argv, env = self.calls[0]
        self.assertEqual(argv[:3], [sys.executable, "-m", "caliby_opt.design_run"])
        self.assertEqual((env["CALIBY_OPT"], env["CALIBY_VARIANT"]), ("exact", "ensemble32"))
        self.assertEqual(env["CALIBY_X_ENS_WORKERS"], "8")
        self.assertEqual(env["CALIBY_X_PP_FASTPDB"], "1")
        self.assertEqual(env["CALIBY_FAST_SAMPLER"], "2")
        self.assertNotIn("CALIBY_X_CLEAN", env)
        self.assertEqual(stack.kit_switches_in(env), modes.resolve("exact", "ensemble32").env)   # the row, whole, and nothing else
        self.assertNotIn("--lcp_kernel", argv)                                                  # no sub-selector of a row exists
        w = argv[argv.index("--") + 1:]
        self.assertIn("--ensemble", w)
        self.assertEqual(w[w.index("--pp_batch_size") + 1], "16")
        self.assertEqual(w[w.index("--omit_aas") + 1], "C")
        self.assertNotIn("--num_samples_per_pdb", w)                    # not given: generate_ensembles' own default (32)
        self.assertEqual(w[w.index("--num_seqs_per_pdb") + 1], "4")
        self.assertEqual(w[w.index("--seed") + 1], "12")

    def test_clean_workers_selects_the_loader_row(self):
        inp = os.path.join(self.tmp, "c.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", self._fake_child):
            cli.run_design("fast", "single", [inp], os.path.join(self.tmp, "out3"), seed=11, clean_workers=4)
        argv, env = self.calls[0]
        self.assertEqual(env["CALIBY_X_CLEAN"], "loader")
        self.assertEqual(argv[argv.index("--clean_workers") + 1], "4")
        self.assertNotIn("--clean_lever", argv)

    def test_clean_workers_select_the_row_and_pass_through(self):
        inp = os.path.join(self.tmp, "c2.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", self._fake_child):
            rc = cli.main(["design", "--input", inp, "--out_dir", os.path.join(self.tmp, "out3b"), "--clean_workers", "8"])
        self.assertEqual(rc, 0)
        argv, env = self.calls[0]
        self.assertEqual((env["CALIBY_OPT"], env["CALIBY_X_CLEAN"], env["CALIBY_X_LCP"]), ("fast", "loader", "1"))  # single: fast by default; N>1 clean workers = the loader row, whole
        with mock.patch.object(cli, "_run_child", self._fake_child):                              # ensemble32: N clean workers are upstream's own parallel clean — passed to the writer, the row unchanged
            rc = cli.main(["design", "--variant", "ensemble32", "--clean_workers", "4", "--input", inp, "--out_dir", os.path.join(self.tmp, "out3c"), "--seed", "1"])
        self.assertEqual(rc, 0)
        argv, env = self.calls[1]
        w = argv[argv.index("--") + 1:]
        self.assertEqual((w[w.index("--clean_workers") + 1], env.get("CALIBY_X_CLEAN")), ("4", None))
        self.assertEqual(stack.kit_switches_in(env), modes.resolve("exact", "ensemble32").env)
        for gone in ("--clean_lever", "--lcp_kernel", "--timeout"):                               # no lever selector outside the mode words, no deadline flag
            with self.assertRaises(SystemExit):
                cli.main(["design", "--input", inp, "--out_dir", self.tmp, gone, "1"])

    def test_seed_det_and_model_name_pass_through(self):
        """No --seed = upstream's unseeded call (no --seed reaches the writer); --det 1 needs --seed; --model_name is any name or path, both
        routes, and reaches the kit child (the weights step) as well as the writer; not given = load_model's default, not passed."""
        inp = os.path.join(self.tmp, "c4.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", self._fake_child):
            self.assertEqual(cli.main(["design", "--mode", "off", "--input", inp, "--out_dir", os.path.join(self.tmp, "o1")]), 0)
            self.assertEqual(cli.main(["design", "--mode", "off", "--input", inp, "--out_dir", os.path.join(self.tmp, "o2"), "--seed", "7", "--det", "1", "--model_name", "/w/x.ckpt"]), 0)
            self.assertEqual(cli.main(["design", "--input", inp, "--out_dir", os.path.join(self.tmp, "o3"), "--model_name", "soluble_caliby", "--verbose", "false",
                                       "--sampling_overrides", "potts_sampling.n_sweeps=200", "batch_size=2", "--temperature", "0.1", "--pos_constraint_csv", "/w/c.csv"]), 0)
        w1, w2, w3 = (a[a.index("--") + 1:] for a, _ in self.calls)
        self.assertNotIn("--seed", w1)
        self.assertEqual((w1[w1.index("--det") + 1], "--model_name" in w1), ("0", False))
        self.assertEqual((w2[w2.index("--seed") + 1], w2[w2.index("--det") + 1], w2[w2.index("--model_name") + 1]), ("7", "1", "/w/x.ckpt"))
        self.assertEqual(w3[w3.index("--model_name") + 1], "soluble_caliby")
        self.assertEqual(w3[w3.index("--verbose") + 1], "false")
        self.assertEqual(w3[w3.index("--sampling_overrides") + 1:w3.index("--sampling_overrides") + 3], ["potts_sampling.n_sweeps=200", "batch_size=2"])
        self.assertEqual((w3[w3.index("--temperature") + 1], w3[w3.index("--pos_constraint_csv") + 1]), ("0.1", "/w/c.csv"))
        k3 = self.calls[2][0]
        self.assertEqual(k3[k3.index("--model_name") + 1], "soluble_caliby")           # the kit child's own --model_name (before --): the weights step checks that checkpoint
        with mock.patch.object(cli, "_run_child", side_effect=AssertionError("child started")):
            self.assertEqual(cli.main(["design", "--mode", "off", "--input", inp, "--out_dir", os.path.join(self.tmp, "o4"), "--det", "1"]), 2)   # --det 1 without --seed: usage, by name

    def test_default_mode_is_the_variants_kit_mode(self):
        inp = os.path.join(self.tmp, "e.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "run_design", return_value=0) as rd:
            self.assertEqual(cli.main(["design", "--input", inp, "--out_dir", self.tmp]), 0)
        self.assertEqual(rd.call_args[0][:2], (modes.default_mode("single"), "single"))
        self.assertEqual((modes.default_mode("single"), modes.default_mode("ensemble32")), ("fast", "fast"))   # never exact by default
        with mock.patch.object(cli, "run_design", return_value=0) as rd:
            self.assertEqual(cli.main(["design", "--variant", "ensemble32", "--input", inp, "--out_dir", self.tmp]), 0)
        self.assertEqual(rd.call_args[0][:2], ("fast", "ensemble32"))
        self.assertEqual(modes.resolve("fast", "ensemble32").env, modes.resolve("exact", "ensemble32").env)   # the default on ensemble32 runs the ensemble row
        os.environ["CALIBY_OPT"] = "off"                               # the environment route wins over the default, --mode over both
        with mock.patch.object(cli, "run_design", return_value=0) as rd:
            cli.main(["design", "--input", inp, "--out_dir", self.tmp])
        self.assertEqual(rd.call_args[0][0], "off")
        with mock.patch.object(cli, "run_design", return_value=0) as rd:
            cli.main(["design", "--mode", "fast", "--input", inp, "--out_dir", self.tmp])
        self.assertEqual(rd.call_args[0][0], "fast")
        del os.environ["CALIBY_OPT"]
        with mock.patch.object(cli, "run_design", return_value=0) as rd:
            cli.main(["warm"])
        self.assertEqual(rd.call_args[0][:2], ("fast", "single"))

    def test_exact_on_single_is_refused_by_name_before_any_work(self):
        """exact on single (deterministic, the forward-pass band) exits 2 with the mode table's words on every verb, from --mode or from
        CALIBY_OPT, and nothing runs: no child, no resolution report."""
        import io, contextlib
        inp = os.path.join(self.tmp, "r.cif"); open(inp, "w").close()
        cases = [["design", "--mode", "exact", "--input", inp, "--out_dir", os.path.join(self.tmp, "refused")],
                 ["design", "--mode", "exact", "--variant", "single", "--input", inp, "--out_dir", os.path.join(self.tmp, "refused")],
                 ["warm", "--mode", "exact"], ["check", "--mode", "exact", "--no-gpu"]]
        with mock.patch.object(cli, "run_design", side_effect=AssertionError("work started")), \
                mock.patch.object(cli, "_run_child", side_effect=AssertionError("child started")):
            for argv in cases:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertEqual(cli.main(argv), 2, argv)
                self.assertIn(f"[caliby-opt] {modes.REFUSED[('exact', 'single')]}", err.getvalue(), argv)
            os.environ["CALIBY_OPT"] = "exact"                         # the environment's mode on the default variant (single): the same refusal
            try:
                for argv in (["check", "--no-gpu"],):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err):
                        self.assertEqual(cli.main(argv), 2, argv)
                    self.assertIn("exact refused: the single-sequence route differs from stock deterministically", err.getvalue(), argv)
            finally:
                del os.environ["CALIBY_OPT"]
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "refused")))

    def test_check_gates_the_child_environment(self):
        # check --mode off with CALIBY_OPT=off (run.sh's mode source) and a kit switch in the caller's shell: the stock child would see
        # neither, so neither is the refusal; both are reported as stripped. On this box the verdict is the pins' (upstream absent).
        os.environ["CALIBY_OPT"], os.environ["CALIBY_X_LCP"] = "off", "1"
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(["check", "--mode", "off", "--no-gpu"])
        self.assertEqual(rc, 3)                                        # would refuse: the code design gives on this box
        self.assertIn("upstream not at its pin", out.getvalue())
        self.assertNotIn("switches in the environment", out.getvalue())
        self.assertIn("stripped from the caller's environment: CALIBY_OPT CALIBY_X_LCP", out.getvalue())
        out = io.StringIO()
        del os.environ["CALIBY_OPT"]
        with contextlib.redirect_stdout(out):
            rc = cli.main(["check", "--no-gpu", "--clean_workers", "8", "--json"])
        rep = json.loads(out.getvalue())
        self.assertEqual((rep["mode"], rep["tier"], rep["row_label"], rep["switches"]["CALIBY_X_CLEAN"], rep["switches"]["CALIBY_X_LCP"]), ("fast", 2, "XL", "loader", "1"))
        self.assertEqual(rep["row_source"], "modes.py:single/loader")
        self.assertNotIn("lcp_kernel", rep)                                                    # no sub-selector of a row: the EXIT line's lcp_kernel= word is the kernel's state, not a choice
        self.assertEqual(rep["stripped_from_caller_env"], ["CALIBY_X_LCP"])
        self.assertEqual(rep["levers_applied"], ["CALIBY_FAST_SAMPLER", "CALIBY_FAST_POTTS_PARAMS", "CALIBY_X_SPARSE_EXACT", "CALIBY_X_LCP", "CALIBY_X_CLEAN", "CALIBY_X_BG_CIF", "CALIBY_X_MULTISEQ", "CALIBY_X_CIF_WORKERS"])
        self.assertEqual([a.split(":")[0] for a in rep["row_alternatives"]], ["clean lever"])
        self.assertNotIn("open_slots", rep)

    def test_exit_codes(self):
        self.assertEqual(cli.main(["design", "--mode", "turbo", "--input", "x.cif", "--out_dir", self.tmp]), 2)   # an unknown mode
        self.assertEqual(cli.main(["design", "--mode", "exact", "--input", "x.cif", "--out_dir", self.tmp]), 2)   # exact on single: refused by name
        self.assertEqual(cli.main(["design", "--mode", "off", "--input", os.path.join(self.tmp, "missing.cif"), "--out_dir", self.tmp]), 2)
        self.assertEqual(cli.main([]), 2)
        rc = cli.main(["check", "--mode", "fast", "--no-gpu"])
        self.assertEqual(rc, 3)                                        # this box: upstream not installed -> would refuse (EXIT_NOT_ACTIVE, as design)
        self.assertEqual(cli.main(["check", "--mode", "exact", "--variant", "ensemble32", "--no-gpu"]), 3)   # the ensemble route's kit mode: resolves; this box: pins
        inp = os.path.join(self.tmp, "d.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", lambda argv, env: 3):   # the design child refused (NOT ACTIVE): its code is the command's
            rc = cli.run_design("fast", "single", [inp], os.path.join(self.tmp, "out4"), seed=11)
        self.assertEqual(rc, 3)
        man = json.load(open(os.path.join(self.tmp, "out4", "opt_manifest.json")))
        self.assertEqual(man["exit_code"], 3)
        self.assertEqual((cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE), (0, 1, 2, 3))   # the shared core's exit table, re-exported
        self.assertNotIn("--keep-installed", subprocess.run([sys.executable, "-m", "caliby_opt", "design", "--help"], capture_output=True, text=True,
                                                            env=dict(os.environ, PYTHONPATH=pythonpath())).stdout)

    def test_the_design_child_gets_the_mode_switch_only(self):
        """The kit child's package switches are the mode (and CALIBY_VARIANT); there is no allow-partial flag or switch any more — a lever that
        falls back at call time is named and the run's code stands."""
        inp = os.path.join(self.tmp, "p.cif"); open(inp, "w").close()
        with mock.patch.object(cli, "_run_child", self._fake_child):
            self.assertEqual(cli.main(["design", "--input", inp, "--out_dir", os.path.join(self.tmp, "ap1")]), 0)
            with self.assertRaises(SystemExit):
                cli.main(["design", "--input", inp, "--out_dir", os.path.join(self.tmp, "ap2"), "--allow-partial"])   # no such flag
        (a1, e1), = self.calls
        self.assertNotIn("--allow-partial", a1)
        self.assertEqual(sorted(k for k in e1 if k.startswith("CALIBY_OPT")), ["CALIBY_OPT"])       # the child's package switches: the mode (and CALIBY_VARIANT)

    def test_incomplete_outputs_exit_1_by_name(self):
        """Fewer designs than the request (timing.json n_designs < inputs x n_seqs, or no timing.json at all) is the named state
        `incomplete: <n>/<m>`: EXIT_FAIL with the record in the manifest — never `partial`."""
        inps = []
        for n in ("i1.cif", "i2.cif"):
            p = os.path.join(self.tmp, n); open(p, "w").close(); inps.append(p)
        import io, contextlib
        err = io.StringIO()
        with mock.patch.object(cli, "_run_child", lambda argv, env: self._fake_child(argv, env, n_designs=3)), contextlib.redirect_stderr(err):
            rc = cli.main(["design", "--input", *inps, "--out_dir", os.path.join(self.tmp, "inc1"), "--num_seqs_per_pdb", "2"])
        self.assertEqual(rc, 1)
        man = json.load(open(os.path.join(self.tmp, "inc1", "opt_manifest.json")))
        self.assertEqual((man["exit_code"], man["incomplete"], man["partial"], man["levers_fallback"]), (1, "3/4", False, []))
        self.assertIn("[caliby-opt] incomplete: 3/4 designs written", err.getvalue())
        with mock.patch.object(cli, "_run_child", lambda argv, env: 0):                # rc 0 without timing.json: 0 designs
            rc = cli.main(["design", "--mode", "off", "--input", inps[0], "--out_dir", os.path.join(self.tmp, "inc2")])
        self.assertEqual(rc, 1)
        self.assertEqual(json.load(open(os.path.join(self.tmp, "inc2", "opt_manifest.json")))["incomplete"], "0/1")
        with mock.patch.object(cli, "_run_child", lambda argv, env: 1):                # the writer's own failure keeps its code
            rc = cli.main(["design", "--mode", "off", "--input", inps[0], "--out_dir", os.path.join(self.tmp, "inc3")])
        self.assertEqual(rc, 1)
        self.assertIsNone(json.load(open(os.path.join(self.tmp, "inc3", "opt_manifest.json")))["incomplete"])
        self.assertEqual(cli.incomplete(inps, 2, {"n_designs": 4}), None)

    def test_warm_keeps_a_partial_run(self):
        """warm removes its scratch directory on rc 0; a run in which a lever could not run (rc 3, the manifest's record) keeps it."""
        from .. import warm, manifest

        def design(mode, variant, files, out, *a, partial_run=False, **kw):
            self.assertNotIn("allow_partial", kw)
            os.makedirs(out, exist_ok=True)
            manifest.write(out, {"mode": mode, "variant": variant, "active": True, "partial": partial_run, "levers_fallback": ["CALIBY_X_LCP"] if partial_run else []},
                           command="warm", exit_code=3 if partial_run else 0)
            return 3 if partial_run else 0
        import io, contextlib
        err = io.StringIO()
        with mock.patch.dict(os.environ, {stack.ENV_STATE: self.tmp}), contextlib.redirect_stderr(err):
            rc = warm.run("fast", "single", lambda *a, **kw: design(*a, **dict(kw, partial_run=False)))
            gone = os.path.join(self.tmp, "warm")
            self.assertEqual((rc, os.listdir(gone)), (0, []))                                     # removed on success
            rc = warm.run("fast", "single", lambda *a, **kw: design(*a, **dict(kw, partial_run=True)))
        self.assertEqual(rc, 3)                                                                  # the design process's NOT ACTIVE code, passed through
        kept = os.listdir(gone)
        self.assertEqual(len(kept), 1)
        self.assertTrue(manifest.read(os.path.join(gone, kept[0]))["partial"])
        self.assertIn("[caliby-opt] warm: a lever of the row could not run (levers_fallback=CALIBY_X_LCP) rc=3; outputs kept at", err.getvalue())

    def test_writer_knobs_default_to_upstream(self):
        # no flag given: upstream's own defaults (caliby/api.py:100-106, inference.yaml:4-10, api.py:405), no seed (upstream's API sets none), det 0
        self.assertEqual((settings.DEFAULT_NUM_SEQS_PER_PDB, settings.DET_LEVELS, settings.DEFAULT_MODEL_NAME), (1, (0, 1), "caliby"))
        self.assertEqual(stack.pins()["weights"]["weight_set"], settings.DEFAULT_MODEL_NAME)      # the weights step's default checkpoint = load_model's
        self.assertFalse(hasattr(settings, "DESIGN_CHAIN"))                                     # every chain designed, upstream's pos_constraint_df=None: no chain knob
        self.assertFalse(hasattr(settings, "DEFAULT_SEED"))                                      # no house seed: --seed given is given, absent is upstream's unseeded call
        self.assertEqual(settings.writer_args("single", ["a.cif"], "o"), ["--inputs", "a.cif", "--out_dir", "o", "--det", "0"])   # nothing given, nothing passed
        self.assertEqual(settings.writer_args("ensemble32", ["a.cif"], "o"), ["--inputs", "a.cif", "--out_dir", "o", "--det", "0", "--ensemble"])
        # every knob given is passed verbatim under upstream's name, in the table's order
        self.assertEqual(settings.writer_args("ensemble32", ["a.cif"], "o", seed=3, det=1, clean_workers=4, num_seqs_per_pdb=8, omit_aas="C,G", num_workers=6, batch_size=2,
                                              pp_batch_size=32, model_name="/w/x.ckpt", verbose=False, sampling_overrides=["a.b=1", "c=2"], temperature=0.1,
                                              num_samples_per_pdb=31, max_num_conformers=16, include_primary_conformer=False, use_primary_res_type=True),
                         ["--inputs", "a.cif", "--out_dir", "o", "--det", "1", "--seed", "3", "--ensemble", "--model_name", "/w/x.ckpt", "--num_seqs_per_pdb", "8",
                          "--batch_size", "2", "--omit_aas", "C,G", "--temperature", "0.1", "--num_workers", "6", "--verbose", "false", "--sampling_overrides", "a.b=1", "c=2",
                          "--clean_workers", "4", "--num_samples_per_pdb", "31", "--pp_batch_size", "32", "--max_num_conformers", "16", "--include_primary_conformer", "false",
                          "--use_primary_res_type", "true"])
        self.assertEqual(settings.writer_args("single", ["a.cif"], "o", pp_batch_size=None, num_seqs_per_pdb=2), ["--inputs", "a.cif", "--out_dir", "o", "--det", "0", "--num_seqs_per_pdb", "2"])
        with self.assertRaises(ValueError):
            settings.writer_args("single", ["a.cif"], "o", n_seqs=2)                            # not an upstream keyword
        d = settings.describe("ensemble32", seed=11)
        self.assertEqual((d["ensemble"], d["model_name"], d["num_seqs_per_pdb"], d["given"]), (True, "caliby", 1, {}))
        self.assertEqual(settings.describe("single", omit_aas="C", num_seqs_per_pdb=3)["given"], {"num_seqs_per_pdb": 3, "omit_aas": "C"})
        self.assertEqual(settings.ensemble_only_given({"pp_batch_size": 4, "num_seqs_per_pdb": 2}), ["--pp_batch_size"])

    def test_stock_caller_refuses_in_a_real_subprocess(self):
        # the stock caller on this box: no upstream at the pin -> exit 3, stock_env_proof.json with the verdict, opt_manifest.json exit_code 3
        out = os.path.join(self.tmp, "stock_out")
        env = {k: v for k, v in os.environ.items() if not k.startswith(stack.STOCK_ABSENT_PREFIXES)}
        env.update({"PYTHONPATH": pythonpath(), "MODEL_OPT": stack.tree_home()})
        r = subprocess.run([sys.executable, "-s", "-m", "caliby_opt.stock_design", "--variant", "single", "--out_dir", out, "--", "--inputs", "x.cif"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, r.stderr[-800:])
        self.assertIn("[caliby-opt] NOT ACTIVE:", r.stderr)
        proof = json.load(open(os.path.join(out, "stock_env_proof.json")))
        self.assertTrue(proof["clean"])
        self.assertTrue(proof["verdict"].startswith("REFUSED: "))
        self.assertTrue(proof["sys_flags"]["no_user_site"])
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertEqual((man["exit_code"], man["command"]), (3, "design"))
        self.assertIn("upstream not at its pin", man["reason"])

    def test_stock_caller_stock_path_writes_the_proofs_then_the_writer(self):
        # the STOCK path in-process: enable("off") stubbed to the pinned digest, the writer replaced by a stub that records its arguments
        out = os.path.join(self.tmp, "stock_ok")
        seen = {}

        def fake_writer(args):
            seen["args"] = list(args)
            return 0
        rep = {"active": False, "mode": "off", "variant": "single", "tree_state": "stock", "tree_digest": stack.pins()["tree_digest_upstream"]["sha256"],
               "switches": {}, "stock_env_proof": stack.stock_env_proof({})}
        with mock.patch.object(stock_design, "enable", return_value=rep), mock.patch.object(stock_design, "run_writer", fake_writer):
            rc = stock_design.main(["--variant", "single", "--out_dir", out, "--", "--inputs", "x.cif", "--seed", "11"])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["args"], ["--inputs", "x.cif", "--seed", "11"])
        proof = json.load(open(os.path.join(out, "stock_env_proof.json")))
        self.assertEqual((proof["verdict"], proof["tree_state"], proof["tree_digest"][:16]), ("STOCK", "stock", rep["tree_digest"][:16]))
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertEqual((man["mode"], man["active"], man["exit_code"], man["settings"]["writer_args"][:2], man["levers_applied"]), ("off", False, 0, ["--inputs", "x.cif"], []))

    def test_run_sh_exits(self):
        run_sh = os.path.join(stack.tree_home(), "run.sh")
        env = {k: v for k, v in os.environ.items() if not k.startswith(stack.STOCK_ABSENT_PREFIXES)}

        def run(*args, **extra):
            return subprocess.run(["bash", run_sh, *args], capture_output=True, text=True, env={**env, **extra})
        self.assertEqual(run().returncode, 2)                                                    # usage
        self.assertEqual(run("bogus").returncode, 2)
        self.assertEqual(run("design", "--config", "nosuch").returncode, 2)
        r = run("design", "--variant", "bogus", "--input", "x", "--out_dir", "y")
        self.assertEqual((r.returncode, "not a variant" in r.stderr), (2, True))
        r = run("check", "--mode", "exact", CALIBY_OPT="off")
        self.assertEqual((r.returncode, "disagrees with CALIBY_OPT=off" in r.stderr), (2, True))
        r = run("check")                                                                        # no --mode: the package's default; the core pin gate passes on this box, then: pins not met
        self.assertEqual((r.returncode, "check_pins" in r.stderr, "NOT ACTIVE" in r.stderr), (3, True, False))
        r = run("warm", "--mode", "off")
        self.assertEqual(r.returncode, 3)                                                        # the gate, then pins, before anything of the verb

    def test_exit_tally_printed_at_interpreter_exit(self):
        code = ("import caliby_opt.report as r; r.register_exit_tally('exact'); import sys; "
                "print('[CALIBY_X_CLEAN=loader] failed (x) -> serial fallback', file=sys.stderr)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CALIBY_X_LCP": "1", "PYTHONPATH": pythonpath()})
        self.assertIn("[caliby-opt] EXIT pid=", r.stderr)
        self.assertIn("touched_modules_loaded=0", r.stderr)
        self.assertIn("fallback_lines=CALIBY_X_CLEAN:1 partial=none", r.stderr)                # no activation report in that process: nothing of a row fell back
        self.assertIn("CALIBY_X_LCP=1", r.stderr)


class TestDesignProcessExit(unittest.TestCase):
    """The exit rule of the design process (design_run.py) when a lever of the row could not run at call time, with activation, the writer
    and the exit tally stubbed: a mode is all of its levers, so the process ends NOT ACTIVE (3) by name whatever the writer returned, the
    record in the manifest (`partial`, `levers_fallback`, `fallback_reasons`); no fallback -> the writer's code, nothing said; lever modules
    that are not the arm's files -> NOT ACTIVE (3) too. No flag lets a subset pass under the mode's name."""

    def setUp(self):
        from .. import design_run, report
        self.design_run, self.report = design_run, report
        self.tmp = tempfile.mkdtemp()
        self.env = dict(os.environ)
        self.rep = {"active": True, "mode": "fast", "variant": "single", "switches": modes.resolve("fast", "single").env,
                    "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False}

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def _run(self, args, writer_rc=0, lcp="available", lines=None, env=None, modules_state="exact", modules_wrong=()):
        import io, contextlib
        from .. import overlay
        tally = {"mode": "exact", "tree_state": "exact", "touched_modules_loaded": [], "lcp_kernel": lcp, "fallback_lines": dict(lines or {}), "switches": {}}
        loaded = {n: {"expected": "kit", "source": "installed", "file": "installed", "ok": False} for n in modules_wrong}
        out = os.path.join(self.tmp, f"run{len(os.listdir(self.tmp))}")
        err = io.StringIO()
        with mock.patch.object(self.design_run, "enable", return_value=self.rep), mock.patch.object(self.design_run, "run_writer", return_value=writer_rc), \
                mock.patch.object(self.report, "tally", lambda: tally), mock.patch.object(overlay, "exit_state", lambda mode: modules_state), \
                mock.patch.object(overlay, "loaded", lambda modules=None: loaded), mock.patch.dict(os.environ, env or {}), contextlib.redirect_stderr(err):
            rc = self.design_run.main(["--mode", "fast", "--variant", "single", "--out_dir", out, *args, "--", "--inputs", "x.cif"])
        return rc, json.load(open(os.path.join(self.tmp, os.path.basename(out), "opt_manifest.json"))), err.getvalue()

    def test_exit_table_is_the_cores_and_the_autoload_restatement_agrees(self):
        from opt_core import report as core_report
        from .. import _autoload, _core_gate, cli, report, tree
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE),
                         (core_report.EXIT_OK, core_report.EXIT_FAIL, core_report.EXIT_USAGE, core_report.EXIT_NOT_ACTIVE))
        self.assertEqual((cli.EXIT_NOT_ACTIVE, _autoload.EXIT_NOT_ACTIVE, _core_gate.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE), (3, 3, 3, 3))   # one NOT ACTIVE code everywhere

    def test_lever_modules_not_the_arms_files_exit_not_active_whatever_else_says(self):
        for args, writer_rc, lcp in (([], 0, "available"), ([], 0, "disabled('x')"), ([], 1, "available")):
            rc, man, err = self._run(args, writer_rc=writer_rc, lcp=lcp, modules_state="mixed", modules_wrong=("caliby.api",))
            self.assertEqual((rc, man["exit_code"], man["modules_state"], man["modules_wrong"]), (3, 3, "mixed", ["caliby.api"]), (args, writer_rc))
            self.assertIn("[caliby-opt] NOT ACTIVE: the lever modules loaded in this process were not the exact arm's files (tree=mixed: caliby.api)", err)

    def test_a_lever_that_could_not_run_ends_not_active_by_name(self):
        rc, man, err = self._run([], lcp="disabled('no triton')")
        self.assertEqual(rc, 3)                                                                  # the mode refuses by name: never a subset under its name
        self.assertEqual((man["exit_code"], man["partial"], man["levers_fallback"]), (3, True, ["CALIBY_X_LCP"]))
        self.assertNotIn("allow_partial", man)
        self.assertEqual(man["fallback_reasons"], {"CALIBY_X_LCP": "disabled('no triton')"})
        self.assertIn("[caliby-opt] NOT ACTIVE: fast ran short of its lever set — CALIBY_X_LCP: disabled('no triton'); the outputs on disk are not a fast run; exit 3\n", err)
        rc, man, err = self._run([], lines={"CALIBY_X_SPARSE_EXACT": 2})                        # a stderr-counted lever
        self.assertEqual((rc, man["partial"], man["levers_fallback"]), (3, True, ["CALIBY_X_SPARSE_EXACT"]))
        self.assertIn("[caliby-opt] NOT ACTIVE: fast ran short of its lever set — CALIBY_X_SPARSE_EXACT: 2 kit fallback line(s): [CALIBY_X_SPARSE_EXACT]; the outputs", err)
        with self.assertRaises(SystemExit):                                                     # no --allow-partial flag exists
            self._run(["--allow-partial"])

    def test_run_main_exit_codes(self):
        """The one script runner: a plain return -> 0; SystemExit(None|int|text) -> 0|int|1; an exception the script raised -> its traceback on
        stderr and 1 (never re-raised past the runner, so the design process's exit rule — completion, NOT ACTIVE by name, the manifest — runs)."""
        import contextlib, io
        from .. import design_run
        cases = {"pass\n": 0, "raise SystemExit()\n": 0, "raise SystemExit(5)\n": 5, "raise SystemExit('words')\n": 1, "raise RuntimeError('[CALIBY_X_LCP] stop')\n": 1}
        for i, (body, want) in enumerate(cases.items()):
            p = os.path.join(self.tmp, f"s{i}.py")
            with open(p, "w") as fh:
                fh.write(body)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(design_run.run_main(p, ["--x", "1"]), want, body)
            if "RuntimeError" in body:
                self.assertIn("RuntimeError: [CALIBY_X_LCP] stop", err.getvalue())
                self.assertIn("Traceback", err.getvalue())
            if "words" in body:
                self.assertEqual(err.getvalue(), "words\n")

    def test_partial_line_grammar(self):
        """The NOT ACTIVE line of a run short of its lever set (one formatter in report.py, every entry point through it); the detail is the
        lever names first, each with the kit's own record."""
        detail = self.report.partial_detail(["CALIBY_X_LCP", "CALIBY_X_CLEAN"], {"CALIBY_X_LCP": "disabled('x')", "CALIBY_X_CLEAN": "2 kit fallback line(s): [CALIBY_X_CLEAN="})
        self.assertEqual(detail, "CALIBY_X_LCP: disabled('x'), CALIBY_X_CLEAN: 2 kit fallback line(s): [CALIBY_X_CLEAN=")
        self.assertEqual(self.report.partial_detail(["CALIBY_X_LCP"], {}), "CALIBY_X_LCP")
        self.assertEqual(self.report.partial_not_active_line("fast", detail, 3), f"[caliby-opt] NOT ACTIVE: fast ran short of its lever set — {detail}; the outputs on disk are not a fast run; exit 3")
        self.assertTrue(self.report.partial_not_active_line("exact", "d", 3).startswith(self.report.PREFIX + " NOT ACTIVE: exact ran short of its lever set — d; "))
        self.assertFalse(hasattr(self.report, "partial_line") or hasattr(self.report, "partial_allowed_line"))

    def test_writer_failure_with_a_lever_short_is_not_active_too(self):
        rc, man, err = self._run([], writer_rc=1, lcp="disabled('x')")
        self.assertEqual((rc, man["exit_code"], man["partial"], man["levers_fallback"]), (3, 3, True, ["CALIBY_X_LCP"]))
        self.assertIn("[caliby-opt] NOT ACTIVE: fast ran short of its lever set — CALIBY_X_LCP: disabled('x'); the outputs on disk are not a fast run; exit 3\n", err)

    def test_no_fallback_is_the_writers_code(self):
        rc, man, err = self._run([])
        self.assertEqual((rc, man["exit_code"], man["partial"], man["levers_fallback"]), (0, 0, False, []))
        self.assertNotIn("allow_partial", man)
        self.assertNotIn("PARTIAL", err)
        rc, man, _ = self._run([], lines={"CALIBY_X_CLEAN": 3})                                   # CALIBY_X_CLEAN=0 on the serial row: not a lever applied
        self.assertEqual((rc, man["partial"]), (0, False))


if __name__ == "__main__":
    unittest.main()
