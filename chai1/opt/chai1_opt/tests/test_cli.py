"""The command layer on a CPU box: `check` prints the DRY-RUN / NOT ACTIVE line and exits 0/3 by the rules (a command without a mode runs
the package default), the driver launcher composes the argv the kit driver parses and writes the fold settings into the kit's RUN_KW,
the pin check refuses an absent or foreign upstream. No torch, no GPU."""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout

from chai1_opt import report
from chai1_opt import cli, driver, modes, settings, stack
from chai1_opt.tests import _stubs

TREE = stack.tree_home()


class TestCheck(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests(); self.err = io.StringIO(); self.out = io.StringIO()
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)

    def tearDown(self):
        _stubs.gates_restore(stack, self.gates); _stubs.remove(self.saved); stack.reset_for_tests()

    def test_exit_codes(self):
        with redirect_stderr(self.err), redirect_stdout(self.out):
            self.assertEqual(cli.main(["check", "--mode", "exact", "--json"]), 0)
        rep = json.loads(self.out.getvalue())
        with redirect_stderr(self.err), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["check", "--mode", "off"]), 0)
            self.assertEqual(cli.main(["check", "--mode", "fast", "--no-use-esm-embeddings", "--no-low-memory"]), 0)   # fast at any fold settings (modes.py)
            self.assertEqual(cli.main(["check", "--mode", "fast"]), 0)
            self.assertEqual(cli.main(["check", "--mode", "fast", "--num-diffn-samples", "1", "--no-low-memory"]), 0)
            self.assertEqual(cli.main(["check", "--mode", "fast", "--num-trunk-samples", "2"]), 0)         # run_inference's trunk loop runs in the kit driver too: passes through
            self.assertEqual(cli.main(["check", "--mode", "off", "--num-trunk-samples", "2", "--device", "cuda:1"]), 0)   # off serves any value of stock's own flags
            self.assertEqual(cli.main(["check", "--mode", "fast", "--device", "cuda:1", "--recycle-msa-subsample", "2", "--constraint-path", "c.csv"]), 0)   # every other keyword and the device pass through to the kit driver
            os.environ.pop(modes.ENV, None)
            self.assertEqual(cli.main(["check"]), 0)                                    # no mode: the package default (fast)
            self.assertEqual(cli.main(["bogus"]), 2)
            self.assertEqual(cli.main([]), 2)
            self.assertEqual(cli.main(["--help"]), 0)
        err = self.err.getvalue()
        self.assertIn("DRY-RUN mode=exact levels=W1,W2,W5 eager=tier1 dstep=hoist2 optin=alloc levers_applied=W1,W2,W5,tier1,hoist2,templ_empty,exactln,msa_pad,transition,rankcc,tailasync,confmemo,prefetch,alloc det=0 route=driver", err)
        self.assertIn("DRY-RUN mode=fast levels=W1,W2,W5 eager=tier1 dstep=hoist2,compiled,dit_attn optin=tf32,alloc levers_applied=W1,W2,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,prefetch,tf32,alloc compile=on:dynamo det=0 route=driver", err)
        self.assertIn("NOT ACTIVE mode=off", err)
        self.assertEqual(err.count("DRY-RUN mode=exact"), 1); self.assertEqual(err.count("DRY-RUN mode=fast"), 6); self.assertNotIn("cannot serve", err)                # --mode exact once; --mode fast five times and the default
        self.assertEqual(rep["mode"], "exact"); self.assertFalse(rep["active"]); self.assertTrue(rep["dry_run"])
        self.assertEqual((rep["settings"]["fold"], rep["settings"]["non_default"], rep["settings"]["flags"]), (settings.library_defaults(), {}, [])); self.assertIsNone(rep["settings"]["library_defaults_installed"])
        self.assertNotIn("preset", rep); self.assertEqual(rep["notes"], [])
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported")          # check applied nothing

    def test_check_notes_a_target_gpu_mismatch(self):
        os.environ["MODEL_OPT_TARGET_GPU"] = "A100"
        try:
            with redirect_stderr(self.err), redirect_stdout(self.out):
                self.assertEqual(cli.main(["check", "--mode", "exact", "--json"]), 0)     # a note, never a refusal
            rep = json.loads(self.out.getvalue())
            self.assertEqual(rep["notes"], ["MODEL_OPT_TARGET_GPU=A100 but the GPU is STUB H100 80GB HBM3: not the GPU this configuration targets"])
            self.assertIn(f"route=driver chai_lab={stack.dist_version('chai_lab')} torch=2.5.1+cu124 gpu=STUB H100 80GB HBM3(sm90) big=gate2048:msa_chunk,nograph n_gpu=1 sharding=none notes=MODEL_OPT_TARGET_GPU=A100 but the GPU is", self.err.getvalue())   # chai_lab as the box has it (None on a CPU box, 0.6.1 on the stack image)
            os.environ["MODEL_OPT_TARGET_GPU"] = "h100"
            with redirect_stderr(io.StringIO()), redirect_stdout(self.out):
                self.out.truncate(0); self.out.seek(0)
                self.assertEqual(cli.main(["check", "--mode", "exact", "--json"]), 0)
            self.assertEqual(json.loads(self.out.getvalue())["notes"], [])
        finally:
            os.environ.pop("MODEL_OPT_TARGET_GPU", None)

    def test_pred_refuses_a_kit_mode_without_its_kits_before_anything_runs(self):
        fasta = os.path.join(stack.kit_home(), "tests", "public_inputs", "1BRS_1to1.fasta")
        os.environ[stack.ENV_EAGER] = "/nonexistent/eager_trunk"
        try:
            with tempfile.TemporaryDirectory() as td, redirect_stderr(self.err):
                rc = cli.main(["pred", "--mode", "fast", "--input", fasta, "--out_dir", td])
                rc2 = cli.main(["pred", "--mode", "exact", "--no-use-esm-embeddings", "--no-low-memory", "--input", fasta, "--out_dir", td])
        finally:
            os.environ.pop(stack.ENV_EAGER, None)
        self.assertEqual((rc, rc2), (3, 3)); self.assertIn("eager stack file missing: /nonexistent/eager_trunk/chai1_eager/stack.py", self.err.getvalue())

    def test_a_role_less_fasta_reaches_the_driver_as_its_path(self):
        """A monomer FASTA with no role field: pred hands the driver `--uids fasta:<path>` (no count suffix) and records id / key / fasta /
        seeds for the item — nothing else."""
        import subprocess
        calls = []; real = subprocess.call
        subprocess.call = lambda cmd, **kw: (calls.append(list(cmd)), 0)[1]
        self.addCleanup(setattr, subprocess, "call", real)
        with tempfile.TemporaryDirectory() as td, redirect_stderr(self.err):
            fa = os.path.join(td, "mono.fasta"); open(fa, "w").write(">protein|name=A\nMKVL\n")
            rc = cli.main(["pred", "--mode", "exact", "--seed", "5", "--input", fa, "--out_dir", td])
            tagdir = os.path.join(td, "exact")
            self.assertNotIn("--manifest-extra", calls[0]); self.assertFalse(os.path.isdir(tagdir) and os.listdir(tagdir))   # no handoff file, nothing written by the launcher itself
        self.assertEqual(rc, cli.EXIT_OK, self.err.getvalue()[-600:])
        self.assertEqual(calls[0][calls[0].index("--uids") + 1], f"fasta:{fa}"); self.assertEqual(calls[0][calls[0].index("--seeds") + 1], "5")

    def test_pred_exit_code_is_the_drivers_condition_and_allow_partial_passes_through(self):
        """One code per condition: a driver that refused / was partial (exit 3) -> pred exits 3 (never 0 with the degradation in a status
        field); a failed fold (2) -> 1; every driver 0 -> 0; --allow-partial (the memory mode's per-item gate opt-out) reaches the driver's argv only when asked."""
        import subprocess
        fasta = os.path.join(stack.kit_home(), "tests", "public_inputs", "1BRS_1to1.fasta")
        calls = []; rcs = iter([3, 2, 0])
        real = subprocess.call
        subprocess.call = lambda cmd, **kw: (calls.append(list(cmd)), next(rcs))[1]
        self.addCleanup(setattr, subprocess, "call", real)
        with tempfile.TemporaryDirectory() as td, redirect_stderr(self.err):
            r3 = cli.main(["pred", "--mode", "exact", "--seed", "42", "--input", fasta, "--out_dir", td])
            r1 = cli.main(["pred", "--mode", "exact", "--seed", "42", "--input", fasta, "--out_dir", td, "--allow-partial"])
            r0 = cli.main(["pred", "--mode", "exact", "--seeds", "42,43", "--input", fasta, "--out_dir", td])
            rn = cli.main(["pred", "--mode", "exact", "--input", fasta, "--out_dir", td])                 # a kit mode with no seed named: refused by name, no driver spawned
        self.assertEqual((r3, r1, r0, rn), (cli.EXIT_NOT_ACTIVE, cli.EXIT_FAIL, cli.EXIT_OK, cli.EXIT_NOT_ACTIVE))
        self.assertEqual(len(calls), 3)
        self.assertEqual((calls[0][calls[0].index("--seeds") + 1], calls[2][calls[2].index("--seeds") + 1]), ("42", "42,43"))   # --seed S is the list of one
        self.assertIn("NOT ACTIVE: kit modes need --seed <int> (or --seeds 0,1 / the items' own seeds): outputs are written per seed, <key>/seed_<s>/ — no seed named for 1BRS_1to1 (--mode off without a seed is stock's unseeded call)", self.err.getvalue())
        self.assertNotIn("--allow-partial", calls[0]); self.assertIn("--allow-partial", calls[1]); self.assertNotIn("--allow-partial", calls[2])
        self.assertEqual(calls[1][1:4], ["-m", "chai1_opt.driver", "--mode"])
        a = driver.parse(["--mode", "exact", "--uids", "x", "--seeds", "0", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m", "--allow-partial"])
        self.assertTrue(a.allow_partial); self.assertFalse(driver.parse(["--mode", "exact", "--uids", "x", "--seeds", "0", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m"]).allow_partial)
        self.assertNotIn("--allow-partial", driver.driver_argv(a, "/kit/kit/chai_worker.py"))            # the kit's own argv knows no such flag


class TestWarmVerdict(unittest.TestCase):
    """warm's verdict on the lines pred prints: a partial activation (PARTIAL field; the run itself proceeds) or a refusal (exit 3) is a FAIL
    verdict by name — warm proves the whole mode; an ACTIVE line with predictions and exit 0 is PASS."""
    def _run(self, pred_lines, rc, n_cif=1):
        import subprocess
        from chai1_opt import warm
        class P:
            def __init__(self, *a, **k):
                self.stdout = iter(pred_lines)
            def wait(self):
                return rc
            def kill(self):
                pass
        real_popen, real_glob = subprocess.Popen, warm.glob.glob
        subprocess.Popen = P; warm.glob.glob = lambda pat: ["x.cif"] * n_cif
        try:
            with redirect_stderr(io.StringIO()):
                return warm.run("exact", echo=False)
        finally:
            subprocess.Popen, warm.glob.glob = real_popen, real_glob

    def test_partial_and_refusal_fail_by_name(self):
        active = "[chai1-opt] ACTIVE mode=exact levels=W1,W2,W5 eager=tier1 dstep=hoist2 levers_applied=W1,W2,W5,tier1,hoist2,templ_empty,rankcc,tailasync,confmemo,prefetch det=0 route=driver chai_lab=0.6.1 torch=2.5.1+cu124 gpu=H100\n"
        partial = active.rstrip("\n") + " PARTIAL off=W5\n"
        refused = "[chai1-opt] NOT ACTIVE: STACK not pinned: torch 2.5.1+cu124 (pinned: torch 2.13.0+cu130) — refused under CHAI1_OPT_STRICT_STACK=1 (refused by name)\n"
        ok = self._run([active, "[chai1-opt] ready route=driver t=1.0\n"], 0)
        self.assertEqual(ok["status"], "PASS"); self.assertNotIn("reason", ok)
        r3 = self._run([refused], 3, n_cif=0)
        self.assertEqual(r3["status"], "FAIL"); self.assertIn("pred refused the activation (exit 3)", r3["reason"]); self.assertIn("refused by name", r3["reason"])
        rp = self._run([partial, "[chai1-opt] ready route=driver t=1.0\n"], 0)
        self.assertEqual(rp["status"], "FAIL"); self.assertIn("partial activation", rp["reason"]); self.assertIn("PARTIAL off=W5", rp["reason"])


class TestDriverComposition(unittest.TestCase):
    def test_one_worker(self):
        self.assertEqual(stack.driver_path(), os.path.join(stack.errata02_home(), "chai_worker.py"))                # one worker for pred and warm

    def test_driver_argv_is_what_the_kit_parses(self):
        a = driver.parse(["--mode", "exact", "--no-use-esm-embeddings", "--no-low-memory", "--uids", "fasta:/x/a.fasta:1,fasta:/x/b.fasta:2", "--seeds", "0,1",
                          "--out_dir", "/o", "--tag", "exact", "--msa_dir", "/m", "--det", "1"])
        argv = driver.driver_argv(a, "/kit/kit/chai_worker.py")
        self.assertEqual(argv, ["/kit/kit/chai_worker.py", "fasta:/x/a.fasta:1,fasta:/x/b.fasta:2", "0,1", "/o", "--tag", "exact",
                                "--levels", "W1,W2,W5", "--msa_dir", "/m"])
        self.assertEqual(settings.from_args(a)["non_default"], {"use_esm_embeddings": False, "low_memory": False})   # stock's flags, parsed as pred parses them
        src = open(stack.driver_path(), encoding="utf-8").read()
        for flag in ("--tag", "--levels", "--msa_dir"):
            self.assertIn(f'add_argument("{flag}"', src)
        self.assertIn('ap.add_argument("uids")', src); self.assertIn('ap.add_argument("seeds")', src); self.assertIn('ap.add_argument("out_root")', src)
        a2 = driver.parse(["--mode", "fast", "--no-use-esm-embeddings", "--uids", "x", "--seeds", "0", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m"])
        self.assertEqual(driver.driver_argv(a2, "/kit/kit/chai_worker.py")[-4:], ["--levels", "W1,W2,W5", "--msa_dir", "/m"])   # fast: the same driver line, the eager lever is the package's
        a3 = driver.parse(["--mode", "fast", "--uids", "x", "--seed", "0", "--out_dir", "/o", "--tag", "t", "--msa-directory", "/m", "--device", "cuda:0"])   # stock's spellings: --seed, --msa-directory, --device
        self.assertEqual((a3.seeds, a3.msa_dir, a3.device), ("0", "/m", "cuda:0")); self.assertEqual(settings.to_driver_run_kw(settings.from_args(a3))["device"], "cuda:0")
        self.assertEqual(settings.from_args(a3)["fold"], settings.library_defaults()); self.assertIsNone(modes.refusal("fast"))   # no fold flag: stock's defaults
        pa = cli._argparser("pred").parse_args(["--input", "x.fasta", "--out_dir", "o", "--seed", "3", "--msa-directory", "/m", "--device", "cuda:1"])
        self.assertEqual((pa.seeds, pa.msa_dir, pa.device, settings.from_args(pa)["device"]), ("3", "/m", "cuda:1", "cuda:1"))
        noseed = driver.parse(["--mode", "exact", "--uids", "x", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m"])
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(driver.run(noseed), 3)                                 # the driver itself refuses a call with no seed, by name
        self.assertIn("NOT ACTIVE: kit modes need --seed <int>", err.getvalue())
        dev = driver.parse(["--mode", "exact", "--device", "cuda:1", "--recycle-msa-subsample", "2", "--template-hits-path", "/t/hits.m8", "--uids", "x", "--seeds", "0", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m"])
        rk = settings.to_driver_run_kw(settings.from_args(dev))                     # the device and every fold keyword ride the kit's RUN_KW (its context / fold calls read them)
        self.assertEqual((rk["device"], rk["recycle_msa_subsample"], rk["template_hits_path"], rk["constraint_path"], rk["msa_server_url"]), ("cuda:1", 2, "/t/hits.m8", None, "https://api.colabfold.com"))
        two = driver.parse(["--mode", "exact", "--num-trunk-samples", "2", "--uids", "x", "--seeds", "0", "--out_dir", "/o", "--tag", "t", "--msa_dir", "/m"])
        self.assertEqual(settings.to_driver_run_kw(settings.from_args(two))["num_trunk_samples"], 2)   # run_inference's trunk loop runs in the kit driver (chai_worker.py): the count rides RUN_KW

    def test_settings_are_written_into_the_kits_run_kw(self):
        """(2) the one documented deviation: the launcher writes the fold settings into the kit's RUN_KW dict before the driver's bytes run."""
        for given in ({}, {"low_memory": False}, {"use_esm_embeddings": False, "low_memory": False}, {"use_esm_embeddings": False}, {"num_diffn_samples": 1, "low_memory": False}):
            cp = types.SimpleNamespace(RUN_KW={"stale": True})
            st = settings.from_values(given)
            with redirect_stderr(io.StringIO()):
                written = driver.apply_settings(cp, st)
            self.assertEqual(cp.RUN_KW, settings.to_driver_run_kw(st)); self.assertEqual(written, cp.RUN_KW)
            self.assertEqual(sorted(cp.RUN_KW), sorted(settings.DRIVER_RUN_KW_KEYS)); self.assertEqual(cp.RUN_KW["device"], "cuda:0")
            for k, v in st["fold"].items():
                if k in cp.RUN_KW: self.assertEqual(cp.RUN_KW[k], v, k)
        cp = types.SimpleNamespace(RUN_KW={"stale": True})                                                # ESM off on the kit route: MSA keyword absent (the run's --msa_dir governs)
        err = io.StringIO()
        with redirect_stderr(err):
            driver.apply_settings(cp, settings.from_values({"use_esm_embeddings": False, "low_memory": False}))
        self.assertEqual((cp.RUN_KW["use_esm_embeddings"], cp.RUN_KW["low_memory"]), (False, False)); self.assertNotIn("msa_directory", cp.RUN_KW)
        kw = [l for l in err.getvalue().splitlines() if " SETTINGS " in l]                                   # the driver route prints the dict it wrote, one grammar with the stock route
        self.assertEqual(len(kw), 1); self.assertRegex(kw[0], report.SETTINGS_GRAMMAR); self.assertTrue(kw[0].startswith("[chai1-opt] SETTINGS constraint_path=None device=cuda:0 low_memory=False msa_server_url=https://api.colabfold.com num_diffn_samples=5 "), kw[0])
        self.assertIn(" use_esm_embeddings=False", kw[0]); self.assertIn(" low_memory=False", kw[0]); self.assertIn(" num_trunk_recycles=3", kw[0]); self.assertNotIn("preset=", kw[0])
        cp = types.SimpleNamespace(RUN_KW={"stale": True})                                                # ESM off with stock's own low_memory=True, nothing else
        err = io.StringIO()
        with redirect_stderr(err):
            driver.apply_settings(cp, settings.from_values({"use_esm_embeddings": False}))
        self.assertEqual((cp.RUN_KW["use_esm_embeddings"], cp.RUN_KW["low_memory"]), (False, True))
        kw = [l for l in err.getvalue().splitlines() if " SETTINGS " in l]
        self.assertEqual(len(kw), 1); self.assertRegex(kw[0], report.SETTINGS_GRAMMAR); self.assertIn(" low_memory=True", kw[0])
        err = io.StringIO()
        with redirect_stderr(err):
            driver.apply_settings(types.SimpleNamespace(RUN_KW={}), settings.from_values({}))
        self.assertIn(" use_esm_embeddings=True", err.getvalue()); self.assertIn(" low_memory=True", err.getvalue()); self.assertIn("[chai1-opt] SETTINGS ", err.getvalue())   # no flag: stock's defaults
        cp = types.SimpleNamespace(RUN_KW={"stale": True})
        with redirect_stderr(io.StringIO()):
            driver.apply_settings(cp, settings.from_values({"num_trunk_samples": 2}))
        self.assertEqual((cp.RUN_KW["num_trunk_samples"], "stale" in cp.RUN_KW), (2, False))   # written like every other keyword; the stale dict replaced


class TestCheckPins(unittest.TestCase):
    def test_refuses_absent_upstream_and_reads_wheel_record(self):
        mod = stack._check_pins_module()
        bad, detail = mod.check()
        if stack.dist_version("chai_lab") is None:
            self.assertTrue(any("chai_lab" in b for b in bad), bad)                 # the upstream absent (a CPU box): refused by name
        else:
            self.assertFalse(any("chai_lab" in b for b in bad), bad)                # the upstream present at the pinned wheel bytes (the stack image): pinned
        rec = mod.wheel_record(os.path.join(TREE, "stock", "chai_lab-0.6.1-py3-none-any.whl"))
        self.assertGreaterEqual(len(rec), 60); self.assertIn("chai_lab/chai1.py", rec)
        self.assertTrue(mod.torch_in_range("2.5.1+cu124", "torch<2.7,>=2.3.1")); self.assertFalse(mod.torch_in_range("2.7.0", "torch<2.7,>=2.3.1"))
        self.assertFalse(mod.torch_in_range("2.2.0", "torch<2.7,>=2.3.1")); self.assertTrue(mod.torch_in_range("2.6.0+cu124", "torch<2.7,>=2.3.1"))

    def test_stack_is_a_record_permissive_and_a_refusal_strict(self):
        """The torch / CUDA stack verdict of stock/check_pins.py with the metadata read faked (no torch on the box needed): the pinned build —
        same torch version AND same CUDA build (PINS stacks) — is silent; any other build (another CUDA build of the same version included, and
        a build whose CUDA cannot be read) is ONE `STACK not pinned: ...` line — never among the refusals unless strict; torch absent is
        not pinned, by name; the strict switch's words."""
        mod = stack._check_pins_module()
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        rec = "torch " + pins["stack"]["torch"]                                                                   # torch 2.13.0+cu130
        def torch_lines(bad):
            return [b for b in bad if b.startswith("STACK ")]
        mod.torch_build = lambda: ("2.5.1+cu124", "12.4")                                                         # in upstream's declared range, not the pinned stack
        t = mod.stack_verdict(pins)
        self.assertEqual((t["pinned"], t["in_range"], t["pinned_stack"]), (False, True, None))
        self.assertEqual(t["line"], f"STACK not pinned: torch 2.5.1+cu124 (pinned: {rec})")
        bad, detail = mod.check(pins, strict=False)
        self.assertEqual(torch_lines(bad), []); self.assertEqual(detail["torch"]["line"], t["line"]); self.assertFalse(detail["torch"]["strict"])   # permissive: recorded, not refused
        bad, detail = mod.check(pins, strict=True)
        self.assertEqual(len(torch_lines(bad)), 1); self.assertTrue(torch_lines(bad)[0].startswith(t["line"] + " — refused under --strict-stack / CHAI1_OPT_STRICT_STACK=1"), bad)
        mod.torch_build = lambda: (pins["stack"]["torch"].split("+")[0], "13.0")                                  # the pinned build, its metadata version without the local tag
        t = mod.stack_verdict(pins)
        self.assertEqual((t["pinned"], t["line"], t["in_range"]), (True, None, False)); self.assertEqual(mod.check(pins, strict=True)[1]["torch"]["pinned"], True)
        self.assertEqual(torch_lines(mod.check(pins, strict=True)[0]), [])                                        # strict on the pinned stack: nothing
        mod.torch_build = lambda: (pins["stack"]["torch"], "13.0")                                                # the pinned build with its local tag (torch/version.py)
        self.assertEqual((mod.stack_verdict(pins)["pinned"], mod.stack_verdict(pins)["line"]), (True, None))
        base = pins["stack"]["torch"].split("+")[0]                                                               # 2.13.0
        for build, want in (((base + "+cu128", "12.8"), f"torch {base}+cu128"), ((base + "+cpu", None), f"torch {base}+cpu"),
                            ((base, "12.6"), f"torch {base}+cu126"), ((base, None), f"torch {base} (CUDA build unknown)"),
                            ((base + "+cu130", "12.8"), f"torch {base}+cu130")):                                  # same version, another CUDA build (or an unreadable / inconsistent one): NOT pinnied, by name
            mod.torch_build = lambda build=build: build
            t = mod.stack_verdict(pins)
            self.assertEqual((t["pinned"], t["pinned_stack"], t["line"]), (False, None, f"STACK not pinned: {want} (pinned: {rec})"), build)
            self.assertEqual(len(torch_lines(mod.check(pins, strict=True)[0])), 1, build); self.assertEqual(torch_lines(mod.check(pins, strict=False)[0]), [], build)
        self.assertTrue(mod.build_matches({"torch": "2.13.0+cu130", "cuda": "13.0"}, "2.13.0", "13.0")); self.assertTrue(mod.build_matches({"torch": "2.13.0+cu130"}, "2.13.0+cu130", None))   # the entry's tag names its CUDA; the installed tag names its own
        self.assertFalse(mod.build_matches({"torch": "2.13.0+cu130", "cuda": "13.0"}, "2.13.0", None)); self.assertFalse(mod.build_matches({"torch": "2.13.0"}, "2.13.0", "13.0"))   # an unreadable CUDA build on either side matches nothing
        self.assertEqual((mod.cuda_of_tag("cu130"), mod.cuda_of_tag("cu128"), mod.cuda_of_tag("cu92"), mod.cuda_of_tag("cpu"), mod.cuda_of_tag(None)), ("13.0", "12.8", "9.2", None, None))
        self.assertEqual((mod.cuda_build("2.13.0+cu130", None), mod.cuda_build("2.13.0", "13.0"), mod.cuda_build("2.13.0+cpu", None), mod.cuda_build("2.13.0", None)), ("13.0", "13.0", None, None))
        self.assertEqual(mod.torch_label("2.13.0", "13.0"), "torch 2.13.0+cu130"); self.assertEqual(mod.torch_label("2.13.0+cu130", "13.0"), "torch 2.13.0+cu130")
        self.assertEqual(mod.torch_label("2.4.0", None), "torch 2.4.0 (CUDA build unknown)"); self.assertEqual(mod.torch_label(None, None), "torch not installed")
        mod.torch_build = lambda: (None, None)                                                                    # torch absent: not pinned, by name (the fold would fail at import anyway)
        t = mod.stack_verdict(pins)
        self.assertEqual((t["pinned"], t["line"]), (False, f"STACK not pinned: torch not installed (pinned: {rec})"))
        self.assertTrue(mod.strict_stack(["--quiet", "--strict-stack"], {})); self.assertFalse(mod.strict_stack(["--quiet"], {}))
        self.assertTrue(mod.strict_stack((), {"CHAI1_OPT_STRICT_STACK": "1"})); self.assertFalse(mod.strict_stack((), {"CHAI1_OPT_STRICT_STACK": "0"}))
        self.assertFalse(mod.strict_stack((), {"CHAI1_OPT_STRICT_STACK": ""}))
        with self.assertRaises(ValueError):
            mod.strict_stack((), {"CHAI1_OPT_STRICT_STACK": "true"})
        self.assertEqual((mod.ENV_STRICT_STACK, mod.STRICT_FLAG), (stack.ENV_STRICT_STACK, "--strict-stack"))     # the package restates the variable: locked here
        from chai1_opt import _autoload
        self.assertIn(stack.ENV_STRICT_STACK, _autoload.DECLARED_ENV); self.assertEqual(_autoload.undeclared({stack.ENV_STRICT_STACK: "1"}), [])

    def test_real_torch_build_read(self):
        """``torch_build`` on this box: (None, None) without torch, else the distribution's version (local tag completed from torch/version.py)."""
        mod = stack._check_pins_module()
        v, cuda = mod.torch_build()
        if stack.dist_version("torch") is None:
            self.assertEqual((v, cuda), (None, None))
        else:
            self.assertTrue(v.startswith(stack.dist_version("torch").split("+")[0]), v)
            self.assertEqual(stack.torch_version() if "torch" not in sys.modules else v, v)                       # the package's torch_version reads the same build without importing torch

    def test_run_sh_strict_flag(self):
        src = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        self.assertIn("--strict-stack) export CHAI1_OPT_STRICT_STACK=1; shift ;;", src)                            # one switch underneath: the variable
        self.assertIn('python -I "$HERE/stock/check_pins.py" --quiet ||', src)

    def test_wheel_equals_tag(self):
        import zipfile, hashlib
        z = zipfile.ZipFile(os.path.join(TREE, "stock", "chai_lab-0.6.1-py3-none-any.whl"))
        n = 0
        for name in z.namelist():
            if name.startswith("chai_lab/"):
                p = os.path.join(TREE, "stock", "src", name)
                self.assertTrue(os.path.isfile(p), name)
                self.assertEqual(hashlib.sha256(open(p, "rb").read()).hexdigest(), hashlib.sha256(z.read(name)).hexdigest(), name); n += 1
        self.assertEqual(n, 102)


class TestStockMsaForm(unittest.TestCase):
    """The stock route folds in chai-lab's own default form: msa_directory=None unless the item has an alignment in the run's directory
    (stock_fold.msa_directory_for, chai-lab's own naming); the kit driver's route keeps the kit's contract — one --msa_dir always, an empty
    one when the run has none (cli.driver_msa_dir). pred --mode off never creates msa_empty."""
    def setUp(self):
        stack.reset_for_tests(); self.err = io.StringIO()
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)

    def tearDown(self):
        _stubs.gates_restore(stack, self.gates); _stubs.remove(self.saved); stack.reset_for_tests()

    def test_driver_route_gets_a_directory_always(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(cli.driver_msa_dir(None, td), os.path.join(td, "msa_empty")); self.assertTrue(os.path.isdir(os.path.join(td, "msa_empty")))
            d = os.path.join(td, "aln"); os.makedirs(d)
            self.assertEqual(cli.driver_msa_dir(d, td), d)

    def test_stock_route_plans_none_without_alignments(self):
        import subprocess
        fasta = os.path.join(stack.kit_home(), "tests", "public_inputs", "1BRS_1to1.fasta")
        calls = []; real = subprocess.call; subprocess.call = lambda cmd, **kw: (calls.append((list(cmd), kw)), 0)[1]
        self.addCleanup(setattr, subprocess, "call", real)
        with tempfile.TemporaryDirectory() as td, redirect_stderr(self.err):
            rc = cli.main(["pred", "--mode", "off", "--input", fasta, "--out_dir", td, "--tag", "t"])
            self.assertEqual(rc, cli.EXIT_OK)
            self.assertFalse(os.path.exists(os.path.join(td, "msa_empty")))                    # chai-lab's default: no directory at all
            plans = [c[c.index("--plan") + 1] for c, _ in calls]; self.assertEqual(len(plans), 1)                    # the plan handed to the one stock process (a temp file, not in the output directory)
            self.assertIsNone(json.load(open(plans[0]))["msa_dir"]); self.assertFalse(plans[0].startswith(td))
            aln = os.path.join(td, "aln"); os.makedirs(aln)
            rc = cli.main(["pred", "--mode", "off", "--input", fasta, "--out_dir", td, "--tag", "t2", "--msa_dir", aln])
            self.assertEqual(rc, cli.EXIT_OK)
            plans = [c[c.index("--plan") + 1] for c, _ in calls]
            self.assertEqual(json.load(open(plans[-1]))["msa_dir"], os.path.abspath(aln))   # the directory: the fold decides per item

    def test_stock_route_is_one_process_per_invocation(self):
        """`pred --mode off` with k items spawns exactly ONE clean stock process (one plan naming every item, in order); the launcher writes
        nothing under the output directory itself."""
        import subprocess
        fasta = os.path.join(stack.kit_home(), "tests", "public_inputs", "1BRS_1to1.fasta")
        calls = []; real = subprocess.call; subprocess.call = lambda cmd, **kw: (calls.append((list(cmd), dict(kw))), 0)[1]
        self.addCleanup(setattr, subprocess, "call", real)
        with tempfile.TemporaryDirectory() as td, redirect_stderr(self.err):
            import shutil
            fas = [shutil.copyfile(fasta, os.path.join(td, f"item{c}_1to1.fasta")) for c in "ABC"]                        # three items (distinct stems = distinct kit keys)
            spec = os.path.join(td, "cli_input.json")
            json.dump([{"id": f"item{c}", "fasta": f} for c, f in zip("ABC", fas)], open(spec, "w"))
            rc = cli.main(["pred", "--mode", "off", "--input", spec, "--out_dir", td, "--tag", "t", "--seeds", "0,1"])
            self.assertEqual(rc, cli.EXIT_OK); self.assertEqual(len(calls), 1, calls)                                  # ONE process for three items
            cmd, kw = calls[0]
            self.assertEqual(cmd[1:4], ["-s", "-m", "chai1_opt.stock_fold"]); self.assertEqual(os.path.basename(cmd[cmd.index("--plan") + 1]), "stock.plan.json")
            self.assertNotIn("CHAI1_OPT", " ".join(kw["env"]))                                                          # the clean environment
            plan = json.load(open(cmd[cmd.index("--plan") + 1]))
            self.assertEqual([it["id"] for it in plan["items"]], ["itemA", "itemB", "itemC"]); self.assertEqual([it["seeds"] for it in plan["items"]], [[0, 1]] * 3)
            self.assertFalse(os.path.isdir(os.path.join(td, "t")) and os.listdir(os.path.join(td, "t")))                   # the plan is a handoff in a temp dir; the launcher writes no file under the output directory (the fake fold wrote nothing)
            self.assertIn("(items=3, one process)", self.err.getvalue())

    def test_msa_directory_for_uses_chai_labs_own_naming(self):
        import types
        from chai1_opt import stock_fold
        ds = types.ModuleType("chai_lab.data.dataset.inference_dataset")
        ds.read_inputs = lambda p: [types.SimpleNamespace(sequence="ACDE"), types.SimpleNamespace(sequence="FGHI")]
        ap = types.ModuleType("chai_lab.data.parsing.msas.aligned_pqt")
        ap.expected_basename = lambda q: q.lower() + ".aligned.pqt"
        for name in ("chai_lab.data", "chai_lab.data.dataset", "chai_lab.data.parsing", "chai_lab.data.parsing.msas"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["chai_lab.data.dataset.inference_dataset"] = ds; sys.modules["chai_lab.data.parsing.msas.aligned_pqt"] = ap
        self.addCleanup(sys.modules.pop, "chai_lab.data.dataset.inference_dataset", None)
        self.addCleanup(sys.modules.pop, "chai_lab.data.parsing.msas.aligned_pqt", None)
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(stock_fold.msa_directory_for("x.fasta", None), (None, [], 2))
            self.assertEqual(stock_fold.msa_directory_for("x.fasta", td), (None, [], 2))                        # a directory with no alignment for the item: None
            open(os.path.join(td, "fghi.aligned.pqt"), "w").close()
            d, present, n = stock_fold.msa_directory_for("x.fasta", td)
            self.assertEqual((str(d), present, n), (td, ["fghi.aligned.pqt"], 2))                              # one chain aligned: the directory (the other chain single-sequence, upstream's rule)

class TestWarmCrops(unittest.TestCase):
    """`warm --crops`: the cache filler's crop parsing, synthetic inputs and pred command (no GPU)."""
    def test_parse_and_inputs_and_command(self):
        import json, tempfile
        from chai1_opt import warm
        self.assertEqual(warm.parse_crops(None), ()); self.assertEqual(warm.parse_crops("all"), warm.CROPS); self.assertEqual(warm.parse_crops("1024,512,512"), (1024, 512))
        with self.assertRaises(ValueError):
            warm.parse_crops("1000")
        d = tempfile.mkdtemp()
        path = warm.crop_inputs(d, (256, 2048))
        doc = json.load(open(path)); self.assertEqual(len(doc["items"]), 2); self.assertNotIn("msa_dir", doc)
        seq = open(doc["items"][1]["fasta"]).read().splitlines()
        self.assertEqual(seq[0], ">protein|name=warm-crop2048"); self.assertEqual(len(seq[1]), 2048 - warm.CROP_MARGIN)
        cmd = warm.pred_command("fast", input_path=path, out_dir=d, tag="warm", det_level=0, fold=warm.CACHE_FOLD)
        j = " ".join(cmd)
        self.assertIn("--num-diffn-timesteps 2", j); self.assertIn("--num-trunk-recycles 1", j); self.assertIn("--no-use-esm-embeddings", j); self.assertNotIn("--num-diffn-samples", j)   # five samples: the real run's shapes
        ln = "[chai1-opt] FORWARD item=warm__warm_crop1024 out=seed_0 tokens=1008 crop=1024 forward_s=95.512 (run_folding_on_context, cuda-synced)"
        import re as _re
        m = _re.search(r"warm_crop(\d+)\b.*?forward_s=([\d.]+)", ln); self.assertEqual((int(m.group(1)), float(m.group(2))), (1024, 95.512))

