"""The command layer's argument handling, exit codes and the exit rule (report.verdict / the PARTIAL line family grammar); no kit or GPU needed."""
import os
import shutil
import tempfile
import unittest

from esmfold2_opt import cli, report


class TestCli(unittest.TestCase):
    def test_one_exit_code_table_and_the_verdict(self):
        """One table (report.EXIT_*) for every verb; report.verdict: incomplete -> 1, the run's own non-zero code kept, a partial
        application (levers the kit's records show not applied on this device) NAMED and never an exit code."""
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE), (0, 1, 2, 3))
        self.assertEqual(cli.EXIT_NOT_ACTIVE, 3)
        rep = {"partial": ["t9"], "gated": ["mk: every step fell back"]}
        self.assertEqual(report.verdict(0, rep), {"exit_code": 3, "partial": ["t9"], "gated": ["mk: every step fell back"], "incomplete": None, "refused_partial": True})   # a mode is all of its levers: refused
        self.assertEqual(report.verdict(0, rep, "1/3")["exit_code"], 1)                        # incomplete: the run's own outcome first
        self.assertEqual(report.verdict(0, {"partial": []}, "2/3"), {"exit_code": 1, "partial": [], "gated": [], "incomplete": "2/3", "refused_partial": False})
        self.assertEqual(report.verdict(0, {"gated": ["mk: x"]})["exit_code"], 0)             # a documented gate: recorded, exit 0
        self.assertEqual(report.verdict(1, rep)["exit_code"], 1)                              # the run failed on its own: its own code
        self.assertEqual(report.verdict(3, {})["exit_code"], 3)
        line = report.partial_line(report.verdict(0, rep), {"t9": "disabled on this box"})
        self.assertEqual(line, "[esmfold2-opt] NOT ACTIVE: partial activation — levers=t9 (t9: disabled on this box): a mode is all of its levers on a GPU class, refused by name; exit 3")
        self.assertEqual(report.partial_line(report.verdict(0, rep), {}), "[esmfold2-opt] NOT ACTIVE: partial activation — levers=t9 (t9: no reason recorded): a mode is all of its levers on a GPU class, refused by name; exit 3")
        self.assertFalse(any(a.startswith("--allow-partial") for p in (cli.pred_parser(),) for a in p._option_string_actions))   # no opt-out flag: a mode never runs with a subset

    def test_partial_line_family_grammar(self):
        """The family grammar of the partial refusal, byte-literal fixed parts around the engine's own <detail> (the lever names first,
        then the kit's reason): `<PREFIX> NOT ACTIVE: partial activation — <detail>: a mode is all of its levers on a GPU class, refused by
        name; exit 3`. One formatter (report.partial_line)."""
        rep = {"partial": ["t9", "mk"]}
        reasons = {"t9": "OutOfResources: shared memory", "mk": "every step fell back"}
        line = report.partial_line(report.verdict(0, rep), reasons)
        head, tail = report.PREFIX + " NOT ACTIVE: partial activation — ", ": a mode is all of its levers on a GPU class, refused by name; exit 3"
        self.assertTrue(line.startswith(head), line); self.assertTrue(line.endswith(tail), line)
        detail = line[len(head):-len(tail)]
        self.assertEqual(detail, "levers=t9,mk (t9: OutOfResources: shared memory; mk: every step fell back)")
        self.assertEqual(detail, report.partial_detail(report.verdict(0, rep), reasons))
        self.assertEqual(report.verdict(0, rep)["exit_code"], 3)
        self.assertEqual(report.PARTIAL_REFUSED, "{prefix} NOT ACTIVE: partial activation — {detail}: a mode is all of its levers on a GPU class, refused by name; exit 3")
        self.assertIsNone(report.partial_line(report.verdict(1, rep), reasons))                 # the run failed on its own: its own line and code
        self.assertEqual(report.verdict(0, rep, "2/3")["exit_code"], 1); self.assertIsNone(report.partial_line(report.verdict(0, rep, "2/3"), reasons))
        self.assertIsNone(report.partial_line(report.verdict(0, {"partial": []}), {}))
        self.assertFalse(hasattr(report, "PARTIAL_ALLOWED") or hasattr(report, "PARTIAL_LINE") or hasattr(report, "allow_partial"))

    def test_check_exit_codes(self):
        """check: a resolved plan is 0 — a lever whose measured tables do not cover this box is a NOTE of the dry run, never an exit code;
        a refusal (the pinned installation not met, no GPU) is 3; --mode off is 0; --allow-partial is not a flag (usage error)."""
        import contextlib
        import io
        from unittest import mock
        plan = {"active": False, "dry_run": True, "server_mode": "opt14_msa", "server_line": "opt14_msa", "logged": True, "mode": "fast", "variant": "fast", "partial": [],
                "notes": ["t9 cell: no measured row for 8.9|3.7 in the kit's table (8.0|3.7, 9.0|3.7): the table's default cell *|* serves it (ef2_w4's probe launch keeps t9 on where the cell compiles and launches)"]}
        with mock.patch.object(cli, "activate", lambda mode, variant, dry_run=False, line=None: dict(plan)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(["check", "--mode", "fast", "--variant", "fast"]), cli.EXIT_OK)
            self.assertNotIn("PARTIAL", err.getvalue()); self.assertNotIn("NOT ACTIVE", err.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:   # not a flag of the verb: argparse's own usage exit
                cli.main(["check", "--mode", "fast", "--variant", "fast", "--allow-partial"])
            self.assertEqual(cm.exception.code, cli.EXIT_USAGE)
        with mock.patch.object(cli, "activate", lambda mode, variant, dry_run=False, line=None: dict(plan, would_refuse="no CUDA device visible")):
            self.assertEqual(cli.main(["check", "--mode", "fast", "--variant", "fast"]), cli.EXIT_NOT_ACTIVE)
        self.assertEqual(cli.main(["check", "--mode", "off", "--variant", "fast"]), cli.EXIT_OK)

    def test_split_flags(self):
        found, rest = cli.split_flags(["--items", "x.json", "--mode", "fast", "--variant=full_msa", "--seeds", "0"])
        self.assertEqual(found, {"--mode": "fast", "--variant": "full_msa"})
        self.assertEqual(rest, ["--items", "x.json", "--seeds", "0"])
        with self.assertRaises(cli.CliError):
            cli.split_flags(["--mode"])

    def test_usage_and_unknown(self):
        self.assertEqual(cli.main([]), cli.EXIT_USAGE)
        self.assertEqual(cli.main(["--help"]), cli.EXIT_OK)
        self.assertEqual(cli.main(["frobnicate"]), cli.EXIT_USAGE)
        self.assertEqual(cli.main(["check", "--mode", "turbo", "--variant", "fast"]), cli.EXIT_USAGE)
        self.assertEqual(cli.main(["check", "--mode", "fast", "--variant", "huge"]), cli.EXIT_USAGE)

    def test_pred_requires_input_and_variant(self):
        d = tempfile.mkdtemp()
        try:
            self.assertEqual(cli.main(["pred", "--mode", "off", "--variant", "fast", "--input", os.path.join(d, "none.json"), "--out_dir", d]), cli.EXIT_USAGE)
            items = os.path.join(d, "i.json"); open(items, "w").write('{"sequences": []}')
            old = os.environ.pop("ESMFOLD2_VARIANT", None)
            try:
                self.assertEqual(cli.main(["pred", "--mode", "off", "--input", items, "--out_dir", d]), cli.EXIT_USAGE)
            finally:
                if old is not None:
                    os.environ["ESMFOLD2_VARIANT"] = old
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestReport(unittest.TestCase):
    def test_lines(self):
        rep = {"active": True, "mode": "exact", "variant": "fast", "server_line": "opt7x", "upstream": {"esm": "1", "transformers": "2"},
               "gpu": {"name": "H100", "sm": "sm90"}, "levers_applied": ["fused", "t6"], "levers_fallback": ["t3"], "applied": "configured", "partial": True,
               "levers_unavailable": ["t3"]}
        line = report.activation_line(rep)
        self.assertTrue(line.startswith("[esmfold2-opt] ACTIVE mode=exact variant=fast server_mode=opt7x esm=1 transformers=2 gpu=H100(sm90) n_gpu=1 sharding=none levers=fused,t6 fallbacks=t3 applied=configured (partial on H100"))
        self.assertEqual(report.activation_line({"active": False, "mode": "fast", "variant": None, "reason": "r"}), "[esmfold2-opt] NOT ACTIVE: r (mode=fast variant=None)")
        dry = report.activation_line({"dry_run": True, "mode": "fast", "variant": "fast", "server_mode": "opt14_msa", "server_line": "opt14_msa", "upstream": {},
                                      "gpu": {}, "levers_planned": ["fused"], "levers_not_for_variant": ["t12"], "kernel_key": "9.0|3.7", "env": {}})
        self.assertIn("DRY-RUN mode=fast variant=fast server_mode=opt14_msa", dry); self.assertIn("not_for_variant=t12", dry); self.assertIn("kernel_key=9.0|3.7", dry); self.assertNotIn("t9_cell=", dry)
        self.assertIn("no lever counters", report.exit_tally_line(pid=1))


if __name__ == "__main__":
    unittest.main()
