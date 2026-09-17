"""The `--n_gpu P` axis (tp.py) without a GPU: the token text of the ACTIVE / DRY-RUN / EXIT lines, the refusals by name and their exit
codes, the cli routes (`pred` / `check`), the axis's LEVER line, and the interpreter-start refusal of an undeclared ESMFOLD2_OPT_* name."""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

from opt_core.mem import ngpu

from esmfold2_opt import _autoload, big, cli, report, stack, tp
from esmfold2_opt.report import EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE


class Axis(unittest.TestCase):
    def test_request_parsing(self):
        self.assertEqual(tp.n_gpu_requested(None), 1)                                        # absent --n_gpu is 1 (a caller passing no flag)
        self.assertEqual(tp.n_gpu_requested(""), 1)
        self.assertEqual(tp.n_gpu_requested("1"), 1); self.assertEqual(tp.n_gpu_requested(1), 1)   # an explicit 1 is 1
        self.assertEqual(tp.n_gpu_requested(" 4 "), 4)
        for bad in ("x", "0", "-2", "2.5", 0):
            with self.assertRaises(tp.TpError) as cm:
                tp.n_gpu_requested(bad)
            self.assertEqual(cm.exception.code, EXIT_USAGE)

    def test_token_text_is_the_release_trees(self):
        """The exact n_gpu tokens of the ACTIVE line (opt_core.mem.ngpu is the one producer; nothing re-typed here)."""
        self.assertEqual(tp.active_fields(1), "n_gpu=1 sharding=none")
        self.assertEqual(tp.active_fields(2), "n_gpu=2 sharding=rowpair")
        self.assertEqual(tp.active_fields(8), ngpu.active_fields(8, "rowpair"))
        self.assertEqual(tp.MODE, "big"); self.assertIn(tp.SHARDING, ngpu.SCHEMES)
        self.assertEqual(tp.N_GPU_SHIPPED, (1, 2, 4, 8))                                    # README §Applicability: the P set this kit ships

    def test_refusals_by_name_and_code(self):
        self.assertEqual(tp.refusal("big", 1), (None, EXIT_OK))
        self.assertEqual(tp.refusal("fast", 1), (None, EXIT_OK)); self.assertEqual(tp.refusal("off", 1), (None, EXIT_OK))
        for mode in ("fast", "exact", "off"):                                           # n_gpu > 1 outside the memory mode: the mode rule, usage error
            why, code = tp.refusal(mode, 2)
            self.assertEqual((why, code), (ngpu.REFUSE_MODE, EXIT_USAGE))
            self.assertEqual(why, "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)")
        for p in (2, 4, 8):                                                                    # n_gpu > 1 under big inside the shipped set: accepted
            self.assertEqual(tp.refusal("big", p), (None, EXIT_OK))
        for p in (3, 5, 16):                                                                   # outside the shipped set: not shipped, NOT ACTIVE
            why, code = tp.refusal("big", p)
            self.assertEqual(code, EXIT_NOT_ACTIVE)
            self.assertTrue(why.startswith(f"refused: n_gpu={p} not shipped: "), why); self.assertIn("rowpair.py over opt_core.mem.rowpair", why)
        self.assertEqual(tp.precheck("big", None), 1); self.assertEqual(tp.precheck("fast", "1"), 1)
        with self.assertRaises(tp.TpError) as cm:
            tp.precheck("fast", "2")
        self.assertEqual(cm.exception.code, EXIT_USAGE)
        with self.assertRaises(tp.TpError) as cm:
            tp.precheck("big", "3")
        self.assertEqual(cm.exception.code, EXIT_NOT_ACTIVE)
        self.assertEqual(tp.precheck("big", "2"), 2); self.assertEqual(tp.refusal("big", 8), (None, EXIT_OK))

    def test_lever_line_grammar(self):
        line = tp.lever_line(1)
        self.assertTrue(line.startswith("[esmfold2-opt] LEVER name=n_gpu state=off reason=n_gpu=1:single_gpu_line "), line)
        self.assertIn(f"strategy={ngpu.TP_LEVER}", line); self.assertIn(" n_gpu=1 sharding=none", line); self.assertIn(" shipped=1,2,4,8 ", line)
        line2 = tp.lever_line(2)
        self.assertTrue(line2.startswith("[esmfold2-opt] LEVER name=n_gpu state=on "), line2)
        self.assertIn(" impl=esmfold2_opt.rowpair ", line2); self.assertIn(" n_gpu=2 sharding=rowpair", line2); self.assertIn(" tier=2", line2)
        from opt_core.strategies import check_strategy
        check_strategy(ngpu.TP_LEVER)                                                          # canonical in STRATEGIES.json (raises otherwise)

    def test_big_apply_refuses_n_gpu_above_one(self):
        kit = stack.kit_home()
        from esmfold2_opt import modes
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        res = modes.resolve("big", "fast", kit)
        with self.assertRaises(stack.ActivationError) as cm:
            big.apply(object(), res, 1, None, n_gpu=3)
        self.assertIn(tp.REFUSE_NOT_SHIPPED.format(P=3), str(cm.exception))
        with self.assertRaises(stack.ActivationError) as cm:
            big.apply(object(), res, 1, None, n_gpu=2)                                    # shipped P: the memory line still installs nothing sharded itself
        self.assertIn("rowpair.install_rank", str(cm.exception))


class Lines(unittest.TestCase):
    def test_active_dry_run_and_exit_lines_carry_the_token(self):
        rep = {"active": True, "mode": "fast", "variant": "fast", "server_line": "opt14_msa", "server_mode": "opt14_msa", "gpu": {"name": "H100", "sm": "sm90"},
               "levers_applied": ["fused"], "levers_fallback": [], "applied": "configured"}
        line = report.activation_line(rep)
        self.assertIn(" gpu=H100(sm90) n_gpu=1 sharding=none levers=fused ", line)                     # after gpu=, before levers= (readers of `ACTIVE mode=<m> variant=<v> server_mode=<s> .+? gpu=.+? `)
        self.assertTrue(line.startswith("[esmfold2-opt] ACTIVE mode=fast variant=fast server_mode=opt14_msa "))
        self.assertIn(" n_gpu=1 sharding=none", report.activation_line(dict(rep, active=False, dry_run=True)))
        self.assertEqual(report.activation_line({"active": False, "mode": "off", "variant": "full_msa", "reason": "r"}), "[esmfold2-opt] NOT ACTIVE: r (mode=off variant=full_msa)")
        self.assertIn(" n_gpu=1 sharding=none", report.exit_tally_line({}))
        self.assertIn(" n_gpu=1 sharding=none ", report.exit_tally_line({"ef2_opt": {"trunk_calls": 3}}))
        with mock.patch.dict(report.N_GPU, {"P": 2}):
            self.assertIn(" n_gpu=2 sharding=rowpair ", report.activation_line(rep))       # the token follows the process's P (cli sets it)


class Routes(unittest.TestCase):
    def _inp(self, d):
        inp = os.path.join(d, "in.json"); json.dump({"id": "x", "sequences": [{"type": "protein", "id": "A", "sequence": "M"}]}, open(inp, "w")); return inp

    def test_pred_n_gpu_above_one_under_fast_is_a_usage_error_before_any_gate(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"ESMFOLD2_VARIANT": "fast"}, clear=False), redirect_stderr(err):
                rc = cli.main(["pred", "--mode", "fast", "--n_gpu", "2", "--input", self._inp(d), "--out_dir", d])
            self.assertEqual(rc, EXIT_USAGE); self.assertIn(ngpu.REFUSE_MODE, err.getvalue())

    def test_pred_n_gpu_above_one_under_big_is_not_active_before_any_gate(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"ESMFOLD2_VARIANT": "fast"}, clear=False), redirect_stderr(err):
                rc = cli.main(["pred", "--mode", "big", "--n_gpu", "3", "--input", self._inp(d), "--out_dir", d])
            self.assertEqual(rc, EXIT_NOT_ACTIVE); self.assertIn(tp.REFUSE_NOT_SHIPPED.format(P=3), err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(d, "pred_rows.jsonl"))); self.assertFalse(os.path.exists(os.path.join(d, "cif_all")))   # refused before anything loads or writes

    def test_pred_n_gpu_above_the_visible_count_is_not_active_by_name_not_a_traceback(self):
        """`pred --mode big --n_gpu 2` with fewer than 2 GPUs visible: the family's refusal (`refused: n_gpu=2 visible=K`, NGpuRefused) is the
        kit's NOT ACTIVE line and exit 3 — before the weights gate, nothing launched or written, no traceback."""
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"ESMFOLD2_VARIANT": "fast"}, clear=False), \
                    mock.patch("opt_core.mem.rowpair.visible_gpus", return_value=0), redirect_stderr(err):
                rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", self._inp(d), "--out_dir", d])
            self.assertEqual(rc, EXIT_NOT_ACTIVE, err.getvalue()[-600:])
            line = [l for l in err.getvalue().splitlines() if l.startswith("[esmfold2-opt] NOT ACTIVE: ")]
            self.assertEqual(len(line), 1, err.getvalue()[-600:]); self.assertIn("NGpuRefused", line[0]); self.assertIn(ngpu.visible_refusal(2, 0), line[0])
            self.assertEqual(ngpu.visible_refusal(2, 0), "refused: n_gpu=2 visible=0")               # the sentence README's exit table names
            self.assertNotIn("Traceback", err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(d, "pred_rows.jsonl"))); self.assertFalse(os.path.exists(os.path.join(d, "cif_all")))

    def test_pred_an_input_sample_count_against_the_flag_is_a_usage_error_before_anything_loads(self):
        """One sample count per run: an input whose own num_diffusion_samples disagrees with --num_diffusion_samples is refused by name
        (exit 2, the input id, its count and the flag's on the ERROR line) before activation — no fold, no traceback, nothing written;
        the same input WITHOUT the flag passes the gate (its count is the run's)."""
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.json")
            json.dump({"id": "x1", "num_diffusion_samples": 1, "sequences": [{"type": "protein", "id": "A", "sequence": "M"}]}, open(inp, "w"))
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"ESMFOLD2_VARIANT": "fast"}, clear=False), redirect_stderr(err):
                rc = cli.main(["pred", "--mode", "exact", "--input", inp, "--out_dir", d, "--num_diffusion_samples", "5"])
            self.assertEqual(rc, EXIT_USAGE, err.getvalue()[-600:])
            line = [l for l in err.getvalue().splitlines() if l.startswith("[esmfold2-opt] ERROR: ")]
            self.assertEqual(len(line), 1, err.getvalue()[-600:])
            self.assertIn("'x1'", line[0]); self.assertIn("num_diffusion_samples=1", line[0]); self.assertIn("--num_diffusion_samples 5", line[0])
            self.assertNotIn("Traceback", err.getvalue()); self.assertNotIn("ACTIVE", err.getvalue())          # before activation: no ACTIVE / NOT ACTIVE line
            self.assertFalse(os.path.exists(os.path.join(d, "pred_rows.jsonl"))); self.assertFalse(os.path.exists(os.path.join(d, "cif_all")))
            for mode in ("off", "exact"):                                                                  # both routes read the gate the same way
                a = cli.pred_parser().parse_args(["--mode", mode, "--variant", "fast", "--input", inp, "--out_dir", d])
                self.assertEqual(cli._samples_gate(a), (1, "inputs"))
                a.run_samples = cli._samples_gate(a)
                st = cli._run_settings(a)
                self.assertEqual(st.num_diffusion_samples, 1); self.assertIn("inputs: num_diffusion_samples 1", st.source)
                a5 = cli.pred_parser().parse_args(["--mode", mode, "--variant", "fast", "--input", inp, "--out_dir", d, "--num_diffusion_samples", "1"])
                self.assertEqual(cli._samples_gate(a5), (1, "flag"))

    def test_pred_withdraws_the_env_import_hook_before_the_sample_gate_and_the_gate_imports_nothing_of_the_library(self):
        """The command line's mode wins over ESMFOLD2_OPT: cmd_pred disarms the import hook FIRST, then the one-sample-count gate — which
        resolves no fold settings and imports nothing of upstream (an import there would fire an armed hook ahead of _pred_run's activation)."""
        with tempfile.TemporaryDirectory() as d:
            order = []
            def gate(a):
                order.append("gate"); raise cli.CliError("stop here", EXIT_USAGE)
            with mock.patch.dict(os.environ, {"ESMFOLD2_VARIANT": "fast"}, clear=False), \
                 mock.patch.object(cli.stack if hasattr(cli, "stack") else __import__("esmfold2_opt.stack", fromlist=["x"]), "_disarm_autoload", lambda: order.append("disarm")), \
                 mock.patch.object(cli, "_samples_gate", gate), redirect_stderr(io.StringIO()):
                rc = cli.main(["pred", "--mode", "off", "--input", self._inp(d), "--out_dir", d])
            self.assertEqual(rc, EXIT_USAGE); self.assertEqual(order, ["disarm", "gate"])
            from esmfold2_opt import settings
            boom = mock.Mock(side_effect=AssertionError("the sample gate must not resolve fold settings / import the library"))
            with mock.patch.object(cli, "_resolve_settings", boom), mock.patch.object(settings, "upstream", boom):
                a = cli.pred_parser().parse_args(["--mode", "off", "--variant", "fast", "--input", self._inp(d), "--out_dir", d])
                self.assertEqual(cli._samples_gate(a), (None, "default"))
                a5 = cli.pred_parser().parse_args(["--mode", "exact", "--variant", "fast", "--input", self._inp(d), "--out_dir", d, "--num_diffusion_samples", "5"])
                self.assertEqual(cli._samples_gate(a5), (5, "flag"))
            boom.assert_not_called()

    def test_the_old_spelling_is_not_an_argument(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                cli.main(["pred", "--mode", "big", "--gpus", "2", "--input", self._inp(d), "--out_dir", d])
            self.assertEqual(cm.exception.code, 2); self.assertIn("unrecognized arguments: --gpus", err.getvalue())

    def test_check_routes(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(cli.main(["check", "--mode", "big", "--n_gpu", "3", "--variant", "fast"]), EXIT_NOT_ACTIVE)
        self.assertIn(tp.REFUSE_NOT_SHIPPED.format(P=3), err.getvalue())
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(cli.main(["check", "--mode", "fast", "--n_gpu", "4", "--variant", "fast"]), EXIT_USAGE)
        self.assertIn(ngpu.REFUSE_MODE, err.getvalue())
        with redirect_stderr(io.StringIO()):
            try:
                cli.main(["check", "--mode", "big", "--n_gpu", "1", "--variant", "fast"])    # an explicit 1 is the in-process line's own check; its verdict is not this test's
            except SystemExit:
                pass

    def test_no_environment_spelling(self):
        """The axis has no ESMFOLD2_OPT_* name: the package declares none for it, so a set one is refused at interpreter start as undeclared."""
        self.assertFalse([n for n in _autoload.ENV_NAMES if "GPU" in n])
        src = open(tp.__file__, encoding="utf-8").read()
        self.assertFalse(re.findall(r"ESMFOLD2_OPT_[A-Z]+", src))


class Producer(unittest.TestCase):
    def test_the_axis_words_come_from_the_core_producer(self):
        from opt_core.mem import ngpu
        self.assertEqual(tp.MODE, ngpu.MEMORY_MODE); self.assertIs(tp.ngpu, ngpu)
        self.assertEqual(tp.active_fields(1), ngpu.active_fields(1, tp.SHARDING)); self.assertIn(tp.SHARDING, ngpu.SCHEMES)


class Cards(unittest.TestCase):
    """The kit's card rows in the release tree's ONE arch registry (cards.py over opt_core.arch): measured rows on sm90 for every lever and on
    sm80 for the exact / fast sets, the known exclusions by reason word, and the LEVER line's card fields per class."""
    def test_table_and_words(self):
        from esmfold2_opt import cards, registry
        t = cards.table()
        self.assertEqual(set(t), set(registry.LEVERS))
        for flag, row in t.items():
            self.assertEqual(row["sm90"], "supported", flag)                                  # H100 / H200 class: every lever holds its measured rows there
        self.assertEqual(t["t10"]["sm100"], "unsupported:tmem_528_gt_512")                    # the fused pair transition on B200
        self.assertEqual((t["tx"]["sm80"], t["tx"]["sm103"], t["tx"]["sm100"]), ("supported", "uncertified", "uncertified"))   # no measured cell on a class is a word, never an exclusion
        for flag in cards.SM80_TESTED:
            self.assertEqual(t[flag]["sm80"], "supported", flag)                                   # A100 80GB: the exact / fast sets hold measured rows there
        for flag in set(registry.LEVERS) - set(cards.SM80_TESTED):
            self.assertNotEqual(t[flag]["sm80"], "supported", flag)                                # the memory line's levers and the unlisted ones: no A100 row claimed
        self.assertEqual(cards.words_for("t10", "sm90"), {"sm": "sm90"})
        self.assertEqual(cards.words_for("t10", "sm80"), {"sm": "sm80"})
        self.assertEqual(cards.words_for("t10", "sm100"), {"card_reason": "tmem_528_gt_512", "card": "unsupported_card:sm100"})
        self.assertEqual(cards.words_for("fused", "sm80"), {"sm": "sm80"})
        self.assertEqual(cards.words_for("t1", "sm80")["card_support"].split(":")[-1], "sm80")   # a lever with no A100 row (t1, in no mode's set): on, with the registry's no-row word
        self.assertEqual(cards.words_for("x4", "sm80"), {"sm": "sm80"})                              # x4 holds its A100 row (bit-identical with / without at 800 tokens there)
        self.assertEqual(cards.words_for("fused", None)["card_reason"], "no_gpu")
        rep = {"mode": "fast", "variant": "fast", "gpu": {"name": "B200", "cc": "10.0"}}
        line = [l for l in report.lever_lines(rep, {"levers_applied": ["t10"], "model_index": 0}) if " name=t10 " in l][0]
        self.assertIn(" state=on ", line); self.assertIn(" card=unsupported_card:sm100 card_reason=tmem_528_gt_512", line.replace("card_reason=tmem_528_gt_512 card=unsupported_card:sm100", "card=unsupported_card:sm100 card_reason=tmem_528_gt_512"))


class NGpuReachesTheLauncher(unittest.TestCase):
    """The requested --n_gpu reaches the launcher unchanged, and a run that does not honour it is refused (n_gpu_mismatch), never folded on
    fewer cards: `pred --mode big --n_gpu 2` in a process that is not a rank calls rowpair.launch(argv, 'big', 2) and returns ITS code."""
    def test_pred_hands_P_to_the_launcher(self):
        import sys, tempfile, types
        from unittest import mock
        from esmfold2_opt import cli
        calls = []
        fake = types.ModuleType("esmfold2_opt.rowpair")
        fake.is_rank_process = lambda: False
        fake.launch = lambda argv, mode, P, **kw: (calls.append((list(argv), mode, P)), 7)[1]
        fake.refuse_unless_visible = lambda P, visible=None: P                          # the adapter's visible-count refusal (opt_core.mem.rowpair's): P cards visible here
        fake.RL = types.SimpleNamespace(world_size=lambda: 1)
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.json")
            with open(inp, "w") as fh:
                fh.write('[{"id": "t", "sequences": [{"type": "protein", "id": "A", "sequence": "MKV", "msa": null}]}]')
            argv = ["pred", "--mode", "big", "--variant", "fast", "--n_gpu", "2", "--input", inp, "--out_dir", os.path.join(d, "o")]
            from esmfold2_opt import stack as _stack
            with mock.patch.object(tp, "N_GPU_SHIPPED", (1, 2)), mock.patch.dict(sys.modules, {"esmfold2_opt.rowpair": fake}), \
                 mock.patch.object(__import__("esmfold2_opt"), "rowpair", fake, create=True), \
                 mock.patch.object(_stack, "data_path_gate", lambda env=None, variant=None, pins=None: (None, [])):   # the launching process digests the weights before the ranks start (test_weights_digest_memo); no weights root here
                rc = cli.main(argv)
        self.assertEqual(rc, 7)                                                        # the launcher's own exit code comes back unchanged
        self.assertEqual(len(calls), 1); self.assertEqual(calls[0][1:], ("big", 2)); self.assertIn("--n_gpu", calls[0][0])
        self.assertEqual(report.N_GPU["P"], 2)                                         # this process reports the requested P (the ACTIVE / EXIT token's source)
        report.set_n_gpu(1)

    def test_census_refuses_any_disagreement(self):
        from esmfold2_opt.cli import n_gpu_census
        report.set_n_gpu(2)
        try:
            self.assertIsNone(n_gpu_census(2, adapter={"n_gpu": 2, "installed": True}, world=2))
            self.assertEqual(n_gpu_census(2, adapter={"n_gpu": 1, "installed": True}, world=2), "n_gpu_mismatch requested=2 active=2 world=2 adapter=1")
            self.assertEqual(n_gpu_census(2, adapter={"n_gpu": 2, "installed": True}, world=1), "n_gpu_mismatch requested=2 active=2 world=1 adapter=2")
            self.assertEqual(n_gpu_census(2, adapter={"n_gpu": 2, "installed": False}, world=2), "n_gpu_mismatch requested=2 active=2 world=2 adapter=2 adapter_installed=False")
            report.set_n_gpu(1)
            self.assertEqual(n_gpu_census(2, adapter={"n_gpu": 2, "installed": True}, world=2), "n_gpu_mismatch requested=2 active=1 world=2 adapter=2")
            self.assertIsNone(n_gpu_census(1, adapter=None, world=1))
        finally:
            report.set_n_gpu(1)


def _pth_installed() -> bool:
    """The kit's autoload .pth is processed only from a site directory (an editable install); PYTHONPATH does not process .pth files."""
    import site
    dirs = list(site.getsitepackages()) + ([site.getusersitepackages()] if hasattr(site, "getusersitepackages") else [])
    return any(os.path.isfile(os.path.join(d, "esmfold2_opt_autoload.pth")) for d in dirs if d)


class UndeclaredNameAtInterpreterStart(unittest.TestCase):
    """An ESMFOLD2_OPT_* name the package does not declare (ESMFOLD2_OPT_GPUS, ESMFOLD2_OPT_N_GPU, …) refuses at interpreter start by name,
    whatever ESMFOLD2_OPT says — stock never runs under a switch nobody reads (the .pth route, a subprocess with the stub upstream)."""

    @classmethod
    def setUpClass(cls):
        if not _pth_installed():
            raise unittest.SkipTest("the .pth route needs an editable install of the kit (pip install -e opt)")
        from esmfold2_opt.tests import _stubs
        cls.real_kit = _stubs.require_kit(); cls.pins = _stubs.require_pins()
        cls.tmp = tempfile.mkdtemp()
        cls.site = _stubs.write_stub_upstream(cls.tmp, cls.pins)
        cls.kit = _stubs.write_stub_kit(cls.tmp, cls.real_kit)
        cls.tree = _stubs.write_stub_tree(cls.tmp, cls.pins)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, **extra):
        from esmfold2_opt.tests import _stubs
        env = _stubs.env_for_stub(self.site, self.kit, self.tree, **extra)
        return subprocess.run([sys.executable, "-c", "import transformers.models.esmfold2; print('REACHED')"], env=env, capture_output=True, text=True)

    def test_gpus_name_refuses_as_undeclared(self):
        for extra in ({"ESMFOLD2_OPT_GPUS": "2"}, {"ESMFOLD2_OPT_GPUS": "2", "ESMFOLD2_OPT": "off"}, {"ESMFOLD2_OPT_N_GPU": "2", "ESMFOLD2_OPT": "fast", "ESMFOLD2_VARIANT": "fast"}):
            r = self._run(**extra)
            self.assertEqual(r.returncode, EXIT_NOT_ACTIVE, r.stderr[-1500:]); self.assertNotIn("REACHED", r.stdout)
            self.assertIn("NOT ACTIVE", r.stderr); self.assertIn("undeclared", r.stderr); self.assertIn(sorted(k for k in extra if k != "ESMFOLD2_OPT" and k != "ESMFOLD2_VARIANT")[0], r.stderr)

    def test_no_switch_leaves_stock_alone(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr[-1500:]); self.assertIn("REACHED", r.stdout)


if __name__ == "__main__":
    unittest.main()
