"""The exit rule, end to end on fake interpreters with the REAL kit module (`rfd3.model.hoist` under a stub torch): a kit-mode run
whose lever has no evidence of application is PARTIAL — the module read `RFD3_HOIST` off at its import, or a completed run never
imported it — and exits 3 with the family line, unless `--allow-partial` / RFDIFFUSION3_OPT_ALLOW_PARTIAL=1 records the allowance
(`allow_partial` in the manifest) and the run proceeds with its own code; full activation exits 0 with the evidence recorded. On
`design` the refusal happens AT the import (the finder hook), before any roll-out; on the `.pth` route likewise; `warm` decides from
its child's lines. A line the levers cannot serve (classifier-free guidance, `low_memory_mode`, the symmetry sampler) under `exact` is REFUSED
BY NAME on both routes — nothing exported, nothing of the run starts, exit 3 —; met at the engine's initialize instead (a spelling the line
reader cannot see), the pass ends 3 there with the one NOT ACTIVE line and no lever evidence after it; `off` passes every line through. The
family grammar of the lines is byte-literal."""
import json
import os
import tempfile
import unittest

from .. import cli, manifest, modes, report, stack
from . import _fixtures as fx

PARTIAL_HEAD = "[rfdiffusion3-opt] NOT ACTIVE: partial activation — "
ALLOWED_HEAD = "[rfdiffusion3-opt] PARTIAL allowed: "


class TestReportLines(unittest.TestCase):
    def test_family_grammar_is_literal(self):
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE), (0, 1, 2, 3))
        self.assertEqual(report.PARTIAL_REFUSED, "{prefix} NOT ACTIVE: partial activation — {detail}; exit 3 (--allow-partial records and proceeds)")   # the core's literals (opt_core.report), byte-identical across kits
        self.assertEqual(report.PARTIAL_ALLOWED, "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)")
        self.assertEqual(report.ENV_ALLOW_PARTIAL, "RFDIFFUSION3_OPT_ALLOW_PARTIAL")
        self.assertIn(report.ENV_ALLOW_PARTIAL, stack.PACKAGE_ENV)                          # stripped from the stock arm's child
        self.assertEqual((cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE), (0, 1, 2, 3))   # one home: report.py

    def test_partial_line(self):
        line = report.partial_line(["RFD3_HOIST"], "read off", False)
        self.assertEqual(line, PARTIAL_HEAD + "RFD3_HOIST (read off); exit 3 (--allow-partial records and proceeds)")
        self.assertEqual(report.partial_line(["RFD3_HOIST"], "read off", True), ALLOWED_HEAD + "RFD3_HOIST (read off) (--allow-partial, recorded)")
        self.assertEqual(report.partial_detail([], None), "-")
        self.assertEqual(report.partial_detail(["A", "B"], None), "A (no reason recorded), B (no reason recorded)")
        self.assertEqual(report.not_active_line("why"), "[rfdiffusion3-opt] NOT ACTIVE: why")

    def test_allow_partial_env(self):
        self.assertFalse(report.allow_partial_env({}))
        self.assertFalse(report.allow_partial_env({"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "0"}))
        self.assertTrue(report.allow_partial_env({"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1"}))
        self.assertTrue(report.allow_partial_env({"RFDIFFUSION3_OPT_ALLOW_PARTIAL": " 1 "}))

    def test_verdict_precedence(self):
        partial = {"partial": True, "levers_fallback": ["RFD3_HOIST"], "partial_reason": "read off"}
        full = {"partial": False, "levers_fallback": [], "partial_reason": None}
        v = report.verdict(0, partial, False)
        self.assertEqual((v["status"], v["exit_code"], v["partial"], v["partial_reason"], v["allow_partial"]), ("partial", 3, ["RFD3_HOIST"], "read off", False))
        v = report.verdict(0, partial, True)
        self.assertEqual((v["status"], v["exit_code"], v["allow_partial"]), ("partial_allowed", 0, True))
        v = report.verdict(1, partial, True)                                               # a failed run is never conflated with partial
        self.assertEqual((v["status"], v["exit_code"], v["partial"]), ("failed", 1, ["RFD3_HOIST"]))
        v = report.verdict(3, partial, True)                                               # refused at the import: the allowance is not re-read
        self.assertEqual((v["status"], v["exit_code"]), ("not_active", 3))
        v = report.verdict(0, full, False)
        self.assertEqual((v["status"], v["exit_code"], v["partial"]), ("ok", 0, []))
        v = report.verdict(0, None, False)
        self.assertEqual((v["status"], v["exit_code"]), ("ok", 0))

    def test_unserved_line_table(self):
        """modes.UNSERVED: the three upstream computations a kit mode's levers do not drive — classifier-free guidance on
        (`inference_sampler.use_classifier_free_guidance` true, by upstream's own rule: not with `cfg_scale` 1.0), the low-memory tokenization
        (`low_memory_mode` true) and the symmetry sampler (`inference_sampler.kind=symmetry`) — read off the line by settings' truth spellings
        (hydra's `+`/`++`/`~` prefixes stripped) and refused BY NAME under exact | fast through the one gate both routes use (stack.line_refusal),
        never under off; the CFG parameters alone (cfg_scale / cfg_t_max / cfg_features), a switch spelled false, the default sampler kind, a
        key's NAME inside another token's value, and hydra's dict / group syntax (read from the built model instead: census.unserved_by_engine)
        are not line refusals."""
        self.assertEqual(sorted(modes.UNSERVED), ["classifier-free guidance", "low_memory_mode", "symmetry sampler"])
        self.assertEqual((modes.CFG_KEY, modes.LOWMEM_KEY, modes.SAMPLER_KIND_KEY), ("inference_sampler.use_classifier_free_guidance", "low_memory_mode", "inference_sampler.kind"))
        line = ["out_dir=/o", "inputs=/s.json", "ckpt_path=/c", "diffusion_batch_size=8", "inference_sampler.step_scale=1.5", "seed=101", "dump_trajectories=True"]
        self.assertFalse(modes.cfg_requested(line)); self.assertFalse(modes.lowmem_requested(line)); self.assertFalse(modes.symmetry_requested(line))
        for tok in ("inference_sampler.kind=symmetry", "++inference_sampler.kind=Symmetry", "inference_sampler.kind='symmetry'"):
            self.assertTrue(modes.symmetry_requested(line + [tok]), tok)
            self.assertEqual(modes.line_unserved(line + [tok], "fast"), modes.unserved_reason("fast", "symmetry sampler", "inference_sampler.kind=symmetry"))
            self.assertEqual(stack.line_refusal(line + [tok], "exact"), modes.unserved_reason("exact", "symmetry sampler", "inference_sampler.kind=symmetry"))
            self.assertIsNone(stack.line_refusal(line + [tok], "off"))
        self.assertIsNone(modes.line_unserved(line + ["inference_sampler.kind=default"], "fast"))
        self.assertFalse(modes.cfg_requested(line + ["inference_sampler.use_classifier_free_guidance=true", "inference_sampler.cfg_scale=1.0"]))   # upstream's own rule (modes.cfg_active)
        self.assertTrue(modes.cfg_requested(line + ["inference_sampler.use_classifier_free_guidance=true", "inference_sampler.cfg_scale=2"]))
        for tok in ("inference_sampler.use_classifier_free_guidance=True", "inference_sampler.use_classifier_free_guidance=true", "+inference_sampler.use_classifier_free_guidance=1",
                    "++inference_sampler.use_classifier_free_guidance=yes"):
            self.assertTrue(modes.cfg_requested(line + [tok]), tok)
            for mode in ("exact", "fast"):
                why = modes.line_unserved(line + [tok], mode)
                self.assertEqual(why, modes.unserved_reason(mode, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true"), tok)
                self.assertTrue(why.startswith(f"mode={mode} cannot serve classifier-free guidance (inference_sampler.use_classifier_free_guidance=true): upstream runs a second, unconditioned denoiser pass per step"), why)
                self.assertTrue(why.endswith("— refused (exit 3); run --mode off (RFDIFFUSION3_OPT=off on upstream's own command line) for the stock path"), why)
                self.assertEqual(stack.line_refusal(line + [tok], mode), why)                # the one gate both routes use
            self.assertIsNone(modes.line_unserved(line + [tok], "off"))                       # off serves every line
            self.assertIsNone(stack.line_refusal(line + [tok], "off"))
        self.assertTrue(modes.cfg_requested(line + ["inference_sampler.use_classifier_free_guidance=true", "inference_sampler.cfg_scale=2.0"]))
        self.assertFalse(modes.cfg_requested(line + ["inference_sampler.use_classifier_free_guidance=true", "inference_sampler.cfg_scale=1.0"]))   # upstream's own rule: cfg_scale 1.0 leaves CFG off (RFD3.py:65-67)
        for tok in ("inference_sampler.use_classifier_free_guidance=False", "+inference_sampler.cfg_scale=2.0", "++inference_sampler.cfg_t_max=0.5", "~inference_sampler.cfg_features",
                    "inference_sampler.cfg_features=[active_donor]", "out_dir=/runs/cfg_scale_sweep/use_classifier_free_guidance", "inputs=/specs/low_memory_mode.json",
                    "inference_sampler.kind=default", "inference_sampler.allow_realignment=true", "inference_sampler.s_jitter_origin=0.5", "inference_sampler.fraction_of_steps_to_fix_motif=0.3",
                    "inference_sampler={use_classifier_free_guidance: true}", "low_memory_mode=False"):   # served tokens, or spellings the built model answers for (dict syntax)
            self.assertFalse(modes.cfg_requested(line + [tok]), tok)
            for mode in ("off", "exact", "fast"):
                self.assertIsNone(modes.line_unserved(line + [tok], mode), (tok, mode))
                self.assertIsNone(stack.line_refusal(line + [tok], mode), (tok, mode))
        for tok in ("low_memory_mode=True", "+low_memory_mode=true", "low_memory_mode=1"):
            self.assertTrue(modes.lowmem_requested(line + [tok]), tok)
            for mode in ("exact", "fast"):
                why = stack.line_refusal(line + [tok], mode)
                self.assertEqual(why, modes.unserved_reason(mode, "low_memory_mode", "low_memory_mode=true"))
                self.assertIn("cannot serve low_memory_mode (low_memory_mode=true): upstream's memory-efficient tokenization drops the resident all-atom pair track (P_LL=None)", why)
            self.assertIsNone(stack.line_refusal(line + [tok], "off"))
        for mode in ("off", "exact", "fast"):
            self.assertIsNone(stack.line_refusal(line, mode), mode)
        for mode, text in (("exact", modes.COMPILE_UNDER_EXACT), ("fast", modes.COMPILE_UNDER_FAST)):   # what line_refusal still refuses: upstream's compile_model=true under a kit mode (no such route)
            reason = stack.line_refusal(line + ["compile_model=true"], mode)
            self.assertIsNotNone(reason, mode); self.assertEqual(reason, modes.route_refusal(line + ["compile_model=true"], mode))
            self.assertIn(text, reason); self.assertIn("`--mode off` runs", text)
        self.assertIsNone(stack.line_refusal(line + ["compile_model=true"], "off"))
        for gone in ("REFUSED_OVERRIDES", "refused_overrides"):
            self.assertFalse(hasattr(modes, gone), gone)


class _Routes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.fake_torch(fx.make_site(os.path.join(self.td.name, "p"), "patched"))
        self.pristine = fx.make_site(os.path.join(self.td.name, "s"), "pristine")
        fx.stub_cli(self.patched)
        fx.stub_cli(self.pristine)
        self.spec = fx.fake_spec(os.path.join(self.td.name, "public.json"))
        self.ckpt = os.path.join(self.td.name, "fake.ckpt")
        open(self.ckpt, "wb").write(b"not a checkpoint")

    def tearDown(self):
        self.td.cleanup()

    def _design(self, site, mode, extra=(), env=None, out="out"):
        out_dir = os.path.join(self.td.name, out)
        extra = list(extra); tail = extra[extra.index("--") + 1:] if "--" in extra else []; head = extra[:extra.index("--")] if "--" in extra else extra
        args = ["design", "--mode", mode, "--ckpt", self.ckpt] + head + [f"inputs={self.spec}", f"out_dir={out_dir}", "diffusion_batch_size=2"] + tail   # the line: upstream's own tokens, verbatim
        r = fx.run_cli(args, site, env=env, bin_dir=self.bin)
        return r, out_dir


class TestDesignExitRule(_Routes):
    def test_full_activation_with_evidence_exits_0(self):
        r, out = self._design(self.patched, "exact", env={"RFD3_FAKE_IMPORT_HOIST": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=True parts=dedup,downcast,pll,valid,where RFD3_COMPILE=False RFD3_TOKEN_SDPA=False", r.stderr)   # report.applied_line from the real kit module's describe()
        self.assertNotIn("partial activation", r.stderr)
        self.assertNotIn("PARTIAL allowed", r.stderr)
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertIn("rfd3.model.hoist", seen["modules"])
        man = manifest.read(out)
        self.assertEqual({k: man[k] for k in manifest.LEVER_KEYS}, {"levers_applied": ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"], "levers_fallback": [], "levers_unavailable": [], "partial": False,
                                                                    "partial_reason": None, "allow_partial": False})
        self.assertEqual(man["exit_code"], 0)
        self.assertEqual(man["activation_report"]["applied"], "read-at-import")
        self.assertEqual(man["activation_report"]["applied_describe"]["RFD3_HOIST"], True)
        self.assertIn("EXIT pid=", r.stderr)
        self.assertIn("source=memory RFD3_HOIST=True", r.stderr)                       # the tally reads the kit module's counters
        self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.layers.layer_utils RFD3_FZT=True installs=1", r.stderr)   # the fused transition's own evidence line
        self.assertIn("fzt.installs=1", r.stderr)

    def test_warnings_channel_reopens_at_the_import(self):
        """Upstream's construction-time blanket `ignore` is removed when the kit's module is imported (the model build): a det-recipe
        warning raised after it reaches stderr; the WARNINGS line at exit shows the channel's state; the report counts the removal."""
        r, out = self._design(self.patched, "exact", env={"RFD3_FAKE_IMPORT_HOIST": "1", "RFD3_FAKE_BLANKET_IGNORE": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("UserWarning: fake_op does not have a deterministic implementation", r.stderr)
        self.assertRegex(r.stderr, r"\[rfdiffusion3-opt\] WARNINGS pid=\d+ filters=\d+ blanket_ignore=0 \[")
        self.assertEqual(manifest.read(out)["activation_report"]["warnings_reopened"], 1)
        r, out = self._design(self.patched, "exact", env={"RFD3_FAKE_IMPORT_HOIST": "1"}, out="out2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(manifest.read(out)["activation_report"]["warnings_reopened"], 0)         # the channel was open: nothing removed
        self.assertIn("blanket_ignore=0", r.stderr)

    def test_reopen_warnings_keeps_every_other_filter(self):
        import warnings
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.filterwarnings("ignore", message=".*noise.*", category=DeprecationWarning)
            warnings.filterwarnings("ignore")                                                    # upstream's blanket, in front
            self.assertEqual(report.reopen_warnings(), 1)
            self.assertEqual([f[0] for f in warnings.filters], ["ignore"])
            self.assertEqual(warnings.filters[0][1].pattern, ".*noise.*")
            self.assertEqual(report.reopen_warnings(), 0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("default")
                warnings.warn("x does not have a deterministic implementation", UserWarning)
            self.assertEqual(len(w), 1)
        self.assertIn("WARNINGS pid=", report.warnings_line())
        self.assertEqual(report.filter_str(("ignore", None, Warning, None, 0)), "ignore:Warning:*:*")

    def test_partial_at_the_import_exits_3_before_any_output(self):
        r, out = self._design(self.patched, "exact", env={"RFD3_FAKE_IMPORT_HOIST": "1", "RFD3_FAKE_DROP_SWITCH": "1"})
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("ACTIVE mode=exact env=RFD3_HOIST=1", r.stderr)                       # the gate passed and exported
        self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=False", r.stderr)
        self.assertIn(PARTIAL_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import (hoist.py:18), mode exact exported it on: "
                      "the environment changed between the export and the import); exit 3 (--allow-partial records and proceeds)", r.stderr)
        self.assertEqual(r.stderr.count("partial activation"), 1)                         # the one line, once
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))              # refused at the import: no roll-out, no output
        man = manifest.read(out)
        self.assertEqual(man["exit_code"], 3)
        # the layers import the kit's hoist module themselves (layer_utils -> cudagraph_sampler -> hoist: the dynamo-boundary marker lives in hoist.py), so the
        # refusal at hoist's import unwinds those imports too: the fused transition and the graph sampler never installed and their names fall back beside RFD3_HOIST
        self.assertEqual((man["levers_applied"], sorted(man["levers_fallback"]), sorted(man["levers_unavailable"]), man["partial"], man["allow_partial"]),
                         ([], sorted(["RFD3_HOIST", "RFD3_FZT", "RFD3_CUDAGRAPH", "RFD3_INIT_CHUNK"]), sorted(["RFD3_HOIST", "RFD3_FZT", "RFD3_CUDAGRAPH", "RFD3_INIT_CHUNK"]), True, False))
        self.assertIn("rfd3.model.layers.layer_utils was never imported in this process: nothing read RFD3_FZT", man["partial_reason"])
        self.assertIn("read RFD3_HOIST off at its import", man["partial_reason"])
        self.assertEqual(man["activation_report"]["applied"], "MISMATCH")
        self.assertEqual(man["outputs"], [])
        self.assertIn("source=memory RFD3_HOIST=False", r.stderr)                      # the tally still reads the module the rule refused

    def test_allow_partial_records_and_proceeds(self):
        for how, extra, env in (("flag", ["--allow-partial"], {}), ("env", [], {"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1"})):
            r, out = self._design(self.patched, "exact", extra=extra, env=dict(env, RFD3_FAKE_IMPORT_HOIST="1", RFD3_FAKE_DROP_SWITCH="1"), out=f"out_{how}")
            self.assertEqual(r.returncode, 0, (how, r.stderr))
            self.assertIn(ALLOWED_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import", r.stderr)
            self.assertIn("(--allow-partial, recorded)", r.stderr)
            self.assertNotIn("exit 3", r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")), how)     # the run proceeded
            man = manifest.read(out)
            self.assertEqual(man["exit_code"], 0)
            self.assertTrue(man["partial"])
            self.assertTrue(man["allow_partial"])
            self.assertEqual(man["levers_fallback"], ["RFD3_HOIST"])
            self.assertEqual(len([o for o in man["outputs"] if o["file"].endswith(".cif.gz")]), 2)

    def test_never_imported_is_partial(self):
        """A completed run that never imported the kit's module has no evidence of application: `APPLIED none`, partial, exit 3."""
        r, out = self._design(self.patched, "exact")                                      # the stub does not import the module
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("[rfdiffusion3-opt] APPLIED none: rfd3.model.hoist is not imported", r.stderr)
        self.assertIn("[rfdiffusion3-opt] APPLIED none: rfd3.model.layers.layer_utils is not imported", r.stderr)
        self.assertIn("[rfdiffusion3-opt] APPLIED none: rfd3.model.cudagraph_sampler is not imported", r.stderr)
        self.assertIn(PARTIAL_HEAD + "RFD3_HOIST (rfd3.model.hoist was never imported in this process: no evidence that RFD3_HOIST was read (hoist.py:18); ", r.stderr)
        for reason in ("rfd3.model.layers.layer_utils was never imported in this process: nothing read RFD3_FZT",
                       "rfd3.model.cudagraph_sampler was never imported in this process: no sampler read RFD3_CUDAGRAPH (inference_sampler.py:610-611)"):
            self.assertIn(reason, r.stderr)                                                     # each lever without evidence, named with the reasons
        self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")))               # the run completed; the rule refuses its exit
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["partial"], man["allow_partial"], sorted(man["levers_fallback"])), (3, True, False, sorted(["RFD3_HOIST", "RFD3_FZT", "RFD3_CUDAGRAPH", "RFD3_INIT_CHUNK"])))
        self.assertEqual(man["activation_report"]["applied"], "none")
        r, out = self._design(self.patched, "exact", extra=["--allow-partial"], out="out2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(ALLOWED_HEAD + "RFD3_HOIST (rfd3.model.hoist was never imported", r.stderr)
        self.assertEqual((manifest.read(out)["exit_code"], manifest.read(out)["allow_partial"]), (0, True))

    def test_a_failed_run_keeps_its_own_code(self):
        """A run that failed on its own (rc 1, the kit never imported): failed ranks before partial — its own code, the partial a RECORD
        (manifest partial=true beside exit_code 1; the line names no exit 3)."""
        r, out = self._design(self.patched, "exact", env={"RFD3_FAKE_EXIT": "1"})
        self.assertEqual(r.returncode, 1, r.stderr)
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["partial"], man["allow_partial"]), (1, True, False))
        self.assertIn("[rfdiffusion3-opt] PARTIAL recorded: RFD3_HOIST (rfd3.model.hoist was never imported in this process", r.stderr)
        self.assertIn("— the run failed on its own, exit 1 (opt_manifest.json: partial, partial_reason)", r.stderr)
        self.assertNotIn(PARTIAL_HEAD, r.stderr)                                           # no NOT ACTIVE line on a process exiting 1
        self.assertNotIn("exit 3", r.stderr)
        self.assertEqual(report.partial_line(["RFD3_HOIST"], "r", False, exit_code=1), "[rfdiffusion3-opt] PARTIAL recorded: RFD3_HOIST (r) — the run failed on its own, exit 1 (opt_manifest.json: partial, partial_reason)")
        self.assertEqual(report.partial_line(["RFD3_HOIST"], "r", False, exit_code=3), report.PARTIAL_REFUSED.format(prefix="[rfdiffusion3-opt]", detail="RFD3_HOIST (r)"))
        self.assertEqual(report.partial_line(["RFD3_HOIST"], "r", True, exit_code=1), report.PARTIAL_ALLOWED.format(prefix="[rfdiffusion3-opt]", detail="RFD3_HOIST (r)"))

    def test_a_full_run_with_the_flag_records_no_allowance_effect(self):
        r, out = self._design(self.patched, "exact", extra=["--allow-partial"], env={"RFD3_FAKE_IMPORT_HOIST": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        man = manifest.read(out)
        self.assertEqual((man["partial"], man["allow_partial"]), (False, True))            # recorded as given; nothing to allow
        self.assertNotIn("PARTIAL", r.stderr)

    def test_off_route_records_the_flag_and_strips_the_switch(self):
        """The stock arm plans no lever: the allowance is recorded as given and can allow nothing; the environment switch is stripped
        from the stock child with the package's other switch (the proof sees neither)."""
        stockpy = fx.make_venv(self.td.name, self.pristine)
        r, out = self._design(self.pristine, "off", extra=["--allow-partial"], env={"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1", "RFDIFFUSION3_STOCK_PYTHON": stockpy})
        self.assertEqual(r.returncode, 0, r.stderr)
        man = manifest.read(out)
        self.assertEqual({k: man[k] for k in manifest.LEVER_KEYS}, {"levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False,
                                                                    "partial_reason": None, "allow_partial": True})
        proof = man["stock_env_proof"]
        self.assertTrue(proof["ok"], proof)
        self.assertIsNone(proof["rfdiffusion3_opt_env"])
        self.assertTrue(all(n not in stack.strip_env({"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1", "RFDIFFUSION3_OPT": "exact", "X": "1"}) for n in stack.PACKAGE_ENV))
        self.assertNotIn("PARTIAL", r.stderr)

    def test_a_computation_met_at_initialize_is_refused_once_and_nothing_more_is_claimed(self):
        """A computation the mode does not drive that the line reader could not see (hydra's dict / group syntax; here the engine's composed
        `inference_sampler` group, RFD3_FAKE_SAMPLER) is met at the engine's initialize: the mode activated (ACTIVE line, the delta exported),
        then ONE `NOT ACTIVE: mode=<m> cannot serve …` line, the process ends 3 before any roll-out — and nothing further is claimed about
        levers: no `APPLIED none`, no second refusal (`partial activation`), no LEVER / KERNELS / PARTIAL line; the manifest records exit_code 3
        and the refusal (`kernels.not_served`), no outputs."""
        cases = (("fast", {"use_classifier_free_guidance": True, "cfg_scale": 1.5}, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance on in the engine's composed config"),
                 ("exact", {"kind": "symmetry"}, "symmetry sampler", "inference_sampler.kind=symmetry in the engine's composed config"),
                 ("exact", {"use_classifier_free_guidance": "true", "cfg_scale": 2.0}, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance on in the engine's composed config"))
        for mode, sampler, feature, asked in cases:
            with self.subTest(mode=mode, feature=feature):
                r, out = self._design(self.patched, mode, extra=[], env={"RFD3_FAKE_IMPORT_HOIST": "1", "RFD3_FAKE_SAMPLER": json.dumps(sampler)}, out=f"initref_{mode}_{feature.split()[0]}")
                self.assertEqual(r.returncode, 3, r.stderr)
                self.assertIn(f"] ACTIVE mode={mode} ", r.stderr)                                   # the mode activated: the line reader saw nothing to refuse
                na = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] NOT ACTIVE: ")]
                self.assertEqual(na, [f"[rfdiffusion3-opt] NOT ACTIVE: {modes.unserved_reason(mode, feature, asked)}"], r.stderr)   # ONE refusal, by name, with the mechanism and the stock path
                for absent in ("] APPLIED none", "partial activation", "] LEVER ", "] KERNELS", "] PARTIAL", "] FALLBACK", "Traceback"):
                    self.assertNotIn(absent, r.stderr)
                man = json.load(open(os.path.join(out, "opt_manifest.json")))
                self.assertEqual((man["exit_code"], man["kernels"]["not_served"]["reason"], man["kernels"]["line"], man["outputs"]), (3, modes.unserved_reason(mode, feature, asked), None, []))
                self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))               # upstream's run never got past initialize
        r, out = self._design(self.patched, "exact", extra=[], env={"RFD3_FAKE_IMPORT_HOIST": "1", "RFD3_FAKE_SAMPLER": json.dumps({"use_classifier_free_guidance": True, "cfg_scale": 1.0})}, out="initref_scale1")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)     # upstream's own rule: the switch with cfg_scale 1.0 is CFG off — served

    def test_an_unserved_line_is_refused_by_name_under_exact(self):
        """A design line that turns classifier-free guidance or the low-memory tokenization on, under exact: refused BY NAME before anything
        of the run starts — one NOT ACTIVE line naming the mode, the feature, the token, the mechanism and the stock path, exit 3; no ACTIVE
        line, nothing exported, upstream's entry point never runs (no cli_seen.json), no output directory, no manifest — with or without
        --allow-partial (a refusal by name has no allowance). The same tokens spelled off, and the CFG parameters alone, are served."""
        for tok, feature, asked in (("inference_sampler.use_classifier_free_guidance=True", "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true"),
                                    ("+inference_sampler.use_classifier_free_guidance=true", "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true"),
                                    ("low_memory_mode=true", "low_memory_mode", "low_memory_mode=true")):
            for extra in ([], ["--allow-partial"]):
                r, out = self._design(self.patched, "exact", extra=extra + ["--", tok], env={"RFD3_FAKE_IMPORT_HOIST": "1"}, out="unserved_" + tok.split("=")[0].split(".")[-1].lstrip("+") + str(len(extra)))
                self.assertEqual(r.returncode, 3, r.stderr)
                na = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] NOT ACTIVE: ")]
                self.assertEqual(na, [f"[rfdiffusion3-opt] NOT ACTIVE: {modes.unserved_reason('exact', feature, asked)}"], r.stderr)
                self.assertIn(f"mode=exact cannot serve {feature} ({asked}): ", na[0]); self.assertIn("— refused (exit 3); run --mode off (RFDIFFUSION3_OPT=off on upstream's own command line) for the stock path", na[0])
                for absent in ("] ACTIVE mode=", "DECLINED", "] APPLIED ", "PARTIAL", "KERNELS", "Traceback"):
                    self.assertNotIn(absent, r.stderr)
                self.assertFalse(os.path.exists(out), out)                                    # nothing ran: no output directory, no cli_seen.json, no manifest
        for tok in ("inference_sampler.cfg_scale=2.0", "inference_sampler.use_classifier_free_guidance=false", "low_memory_mode=False", "dump_trajectories=True"):
            r, out = self._design(self.patched, "exact", extra=["--", tok], env={"RFD3_FAKE_IMPORT_HOIST": "1"}, out="served_" + tok.split("=")[0].split(".")[-1])
            self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("] ACTIVE mode=exact ", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
            self.assertIn(tok, json.load(open(os.path.join(out, "cli_seen.json")))["argv"])   # upstream's entry point ran with its own token, under the mode
        r, out = self._design(self.patched, "exact", extra=["--", "compile_model=true"], env={"RFD3_FAKE_IMPORT_HOIST": "1"}, out="norte")   # upstream's compile under a kit mode: refused likewise (no route)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("NOT ACTIVE", r.stderr); self.assertFalse(os.path.exists(out))

    def test_cfg_override_passes_through_off(self):
        stockpy = fx.make_venv(self.td.name, self.pristine)
        r, out = self._design(self.pristine, "off", extra=["--", "inference_sampler.cfg_scale=2.0"], env={"RFDIFFUSION3_STOCK_PYTHON": stockpy})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("inference_sampler.cfg_scale=2.0", json.load(open(os.path.join(out, "cli_seen.json")))["argv"])

    def test_check_records_the_flag(self):
        r = fx.run_cli(["check", "--mode", "exact", "--json", "--allow-partial"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.loads(r.stdout)
        self.assertTrue(rep["allow_partial"])
        self.assertFalse(rep["partial"])
        r = fx.run_cli(["check", "--mode", "exact", "--json"], self.patched, bin_dir=self.bin)
        self.assertFalse(json.loads(r.stdout)["allow_partial"])
        r = fx.run_cli(["check", "--mode", "exact", "--allow-partial"], self.pristine, bin_dir=self.bin)
        self.assertEqual(r.returncode, 3)                                                  # a refused gate has no allowance


class TestAutoloadExitRule(unittest.TestCase):
    """The .pth route: RFDIFFUSION3_OPT=exact + `import rfd3`, then the kit's module reads the switch off."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.fake_torch(fx.make_site(os.path.join(self.td.name, "p"), "patched"))
        self.seen = os.path.join(self.td.name, "seen.json")

    def tearDown(self):
        self.td.cleanup()

    CODE = '''
import json, os, sys
import rfdiffusion3_opt._autoload as a
import rfd3
if os.environ.get("RFD3_FAKE_DROP_SWITCH") == "1":
    os.environ.pop("RFD3_HOIST", None)
import rfd3.model.hoist as h
from rfdiffusion3_opt import stack
print(json.dumps({"HOIST": h.HOIST, "partial": stack.applied_hook().partial, "status": {k: stack.status().get(k) for k in ("partial", "allow_partial", "levers_fallback", "applied")}}))
'''

    def test_partial_at_the_import_exits_3(self):
        r = fx.run_py(self.CODE, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen, "RFD3_FAKE_DROP_SWITCH": "1"}, bin_dir=self.bin)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn(PARTIAL_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import", r.stderr)
        self.assertEqual(r.stdout.strip(), "")                                             # the process ended at the import

    def test_allow_partial_env_proceeds(self):
        r = fx.run_py(self.CODE, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1", "RFD3_FAKE_SEEN": self.seen, "RFD3_FAKE_DROP_SWITCH": "1"}, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(ALLOWED_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import", r.stderr)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertFalse(d["HOIST"])
        self.assertEqual(d["partial"]["levers"], ["RFD3_HOIST"])
        self.assertTrue(d["partial"]["allowed"])
        self.assertEqual(d["status"], {"partial": True, "allow_partial": True, "levers_fallback": ["RFD3_HOIST"], "applied": "MISMATCH"})

    def test_full_activation_proceeds(self):
        r = fx.run_py(self.CODE, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(d["HOIST"])
        self.assertEqual(d["partial"], {})
        self.assertEqual(d["status"], {"partial": False, "allow_partial": False, "levers_fallback": [], "applied": "read-at-import"})
        self.assertNotIn("PARTIAL", r.stderr)

    def test_a_cfg_line_at_the_trigger_is_refused_by_name(self):
        """RFDIFFUSION3_OPT=exact on upstream's own command line carrying the CFG switch: at the trigger (`import rfd3`) the mode refuses the
        line by name — one NOT ACTIVE line (the mode, the feature, the mechanism, RFDIFFUSION3_OPT=off as the stock path), the process ENDS 3
        right there (nothing after the trigger runs: no output on stdout, the kit's module never imported, nothing exported), with or without
        the partial allowance in the environment; the same line without the switch activates."""
        code = ("import os, sys, json, rfdiffusion3_opt._autoload as a\nimport rfd3\nimport rfd3.model.hoist as h\nimport rfdiffusion3_opt\nst = rfdiffusion3_opt.status() or {}\n"
                "print(json.dumps({'seen': rfd3.SEEN, 'HOIST': h.HOIST, 'env': os.environ.get('RFD3_HOIST'), 'status': {k: st.get(k) for k in ('applied', 'partial', 'active')}}))")
        argv = ["design", "out_dir=/o", "inputs=/s.json", "ckpt_path=/c", "inference_sampler.use_classifier_free_guidance=True"]
        for allow in ({}, {"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1"}):
            r = fx.run_py(code, self.patched, env=dict({"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, **allow), bin_dir=self.bin, argv=argv)
            self.assertEqual(r.returncode, 3, r.stderr)                                         # the process ends at the trigger
            self.assertEqual(r.stdout.strip(), "", r.stdout)                                     # nothing after `import rfd3` ran
            na = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] NOT ACTIVE: ")]
            self.assertEqual(len(na), 1, r.stderr)
            self.assertIn(modes.unserved_reason("exact", "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true"), na[0])
            self.assertIn("RFDIFFUSION3_OPT=off", na[0])
            for absent in ("DECLINED", "] ACTIVE ", "] APPLIED ", "PARTIAL"):
                self.assertNotIn(absent, r.stderr)
        r = fx.run_py(code, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, bin_dir=self.bin, argv=argv[:-1])
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("] ACTIVE mode=exact ", r.stderr)
        r = fx.run_py(code, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, bin_dir=self.bin, argv=argv[:-1] + ["inference_sampler.cfg_scale=1.5", "inference_sampler.allow_realignment=true"])
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("] ACTIVE mode=exact ", r.stderr)   # the CFG parameters alone and the sampler options the graph roll-out drives are served


class TestWarmExitRule(_Routes):
    def _warm(self, extra=(), env=None):
        r = fx.run_cli(["warm", "--mode", "exact", "--ckpt", self.ckpt, "--json"] + list(extra), self.patched, env=env, bin_dir=self.bin)
        res = json.loads(r.stdout) if r.stdout.strip() else None
        return r, res, (res or {}).get("log")                                             # warm keeps its own log; the path rides in the result

    def test_full_activation(self):
        fx.stub_engine(self.patched)
        r, res, log = self._warm()
        self.assertEqual(r.returncode, 0, r.stderr + (open(log).read() if log else ""))
        self.assertIn("WARM PASS mode=exact", r.stderr)
        self.assertNotIn("killed", res)
        self.assertIn("RFD3_HOIST=True", res["applied"])
        self.assertEqual((res["partial"], res["partial_reason"], res["allow_partial"]), ([], None, False))
        self.assertNotIn("PARTIAL", r.stderr)

    def test_partial_exits_3(self):
        fx.stub_engine(self.patched)
        r, res, log = self._warm(env={"RFD3_FAKE_DROP_SWITCH": "1"})
        self.assertEqual(r.returncode, 3, r.stderr + (open(log).read() if log else ""))
        self.assertIn("WARM FAIL mode=exact", r.stderr)
        self.assertIn("error=Partial", r.stderr)
        self.assertIn(PARTIAL_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import (the child's line: [rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=False", r.stderr)
        self.assertIn("; exit 3 (--allow-partial records and proceeds)", r.stderr)
        self.assertEqual((res["status"], res["partial"], res["allow_partial"], res["exit_code"]), ("FAIL", ["RFD3_HOIST"], False, 0))   # the child ran; the verb refuses

    def test_allow_partial(self):
        fx.stub_engine(self.patched)
        for how, extra, env in (("flag", ["--allow-partial"], {}), ("env", [], {"RFDIFFUSION3_OPT_ALLOW_PARTIAL": "1"})):
            r, res, log = self._warm(extra=extra, env=dict(env, RFD3_FAKE_DROP_SWITCH="1"))
            self.assertEqual(r.returncode, 0, (how, r.stderr))
            self.assertIn("WARM PASS mode=exact", r.stderr)
            self.assertIn(ALLOWED_HEAD + "RFD3_HOIST (rfd3.model.hoist read RFD3_HOIST off at its import", r.stderr)
            self.assertEqual((res["status"], res["partial"], res["allow_partial"]), ("PASS", ["RFD3_HOIST"], True))
            self.assertIn("partial activation allowed", res["reason"])


if __name__ == "__main__":
    unittest.main()
