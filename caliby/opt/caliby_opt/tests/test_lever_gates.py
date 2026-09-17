"""A lever's declared input gate, the fast sampler's own fallback words, and the environment route's exit gate (CPU; no GPU, no upstream).

* gate — `CALIBY_FAST_POTTS_PARAMS` level 2 keeps the decoder's sparse couplings only for untied positions: a batch holding a structure with
  symmetry-tied positions takes upstream's dense fold (stock arithmetic, the same outputs) and says so ONCE per call as the core's per-lever
  line `[caliby-opt] LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped reason=symmetry_dense_J …`, recorded (`report.tally()["gates"]`, the
  manifest's `lever_gates`) and never counted as a fallback (`partial` stays False, the exit code is unchanged). The predicate and the line
  live in `caliby_opt.potts_params`; the lever file calls it with plain values. A gate reason the registry does not declare is refused.
* fast sampler — `caliby_opt.fast_sampler` holds the ONE served-call predicate (`caliby_opt.multiseq` composes it) and the
  `[CALIBY_FAST_SAMPLER] … -> stock sampler for this call` line, a stderr-probed fallback of its own lever.
* environment route — a process activated by `CALIBY_OPT=<mode>` (nobody owns its exit code) that was not the mode's run — a lever of the
  row could not run, or the lever modules loaded were not the mode's files (tree=mixed; the predicate the design children apply,
  `activate.modules_wrong`) — ends NOT ACTIVE by name with exit 3 after its EXIT line, whatever the host was about to exit with; the design
  children own their exit (`report.exit_owned()`) and decide it themselves.
"""
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from .. import activate, fast_sampler, manifest, modes, multiseq, potts_params, registry, report, stack
from . import _fixtures as fx

DENOISER = os.path.join(stack.kit_dir(stack.KIT_ADDON), "fast", "atom_mpnn_denoiser.py")
POTTS = os.path.join(stack.kit_dir(stack.KIT_ADDON), "fast", "potts.py")


def _reset():
    report._TALLY["gates"] = {}
    report._TALLY["fallback_lines"] = {}


class TestDeclaredGates(unittest.TestCase):
    def setUp(self):
        _reset()

    def test_the_registry_declares_every_gate_and_the_line_is_the_cores_lever_grammar(self):
        self.assertEqual(registry.declared_gates(), {"CALIBY_FAST_POTTS_PARAMS": (registry.GATE_SYMMETRY_DENSE_J,)})
        self.assertEqual(registry.GATE_SYMMETRY_DENSE_J, "symmetry_dense_J")
        self.assertEqual(report.DECLARED_GATES, registry.declared_gates())
        line = report.lever_gate_line("CALIBY_FAST_POTTS_PARAMS", "symmetry_dense_J", structures="1/4", groups=2, J="dense", arithmetic="stock")
        self.assertEqual(line, "[caliby-opt] LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped reason=symmetry_dense_J impl=atom_mpnn_denoiser origin=kit "
                               "structures=1/4 groups=2 J=dense arithmetic=stock")
        from opt_core import report as core
        self.assertEqual(line, core.lever_line(report.TAG, "CALIBY_FAST_POTTS_PARAMS", "skipped", reason="symmetry_dense_J", impl="atom_mpnn_denoiser",
                                              origin="kit", structures="1/4", groups=2, J="dense", arithmetic="stock"))      # one grammar: the core's

    def test_an_undeclared_gate_is_refused_never_printed(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            with self.assertRaises(ValueError):
                report.note_gate("CALIBY_FAST_POTTS_PARAMS", "some_other_reason")
            with self.assertRaises(ValueError):
                report.note_gate("CALIBY_X_LCP", "symmetry_dense_J")                 # a lever with no declared gate
            with self.assertRaises((ValueError, KeyError)):
                report.note_gate("CALIBY_NOT_A_LEVER", "symmetry_dense_J")
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(report.tally()["gates"], {})

    def test_changes_md_states_the_gate_by_name(self):
        with open(os.path.join(stack.tree_home(), "CHANGES.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("reason=symmetry_dense_J", text)
        self.assertIn("LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped", text)


class TestPottsParamsGate(unittest.TestCase):
    """`potts_params.keep_sparse(level, symmetry_pos)` with fake symmetry groups (upstream's shape: per structure, a list of groups of positions)."""

    def setUp(self):
        _reset()
        self.env = dict(os.environ)
        os.environ.pop("CALIBY_X_SPARSE_EXACT", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        _reset()

    def _call(self, level, symmetry_pos):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", report._CountingStderr(err)):                   # the design process's stderr: the fallback-counting tee
            keep = potts_params.keep_sparse(level, symmetry_pos)
        lines = [ln for ln in err.getvalue().splitlines() if ln]
        return keep, lines

    def test_untied_batches_keep_the_sparse_couplings_silently(self):
        self.assertEqual(self._call(2, [[], [], []]), (True, []))
        self.assertEqual(self._call(2, []), (True, []))
        self.assertEqual(report.tally()["gates"], {})

    def test_levels_below_two_densify_as_stock_without_a_line(self):
        for level in (0, 1):
            self.assertEqual(self._call(level, [[], []]), (False, []))
            self.assertEqual(self._call(level, [[[1, 2, 3]], []]), (False, []))            # the gate is level 2's: lower levels always fold, nothing to say
        self.assertEqual(report.tally()["gates"], {})

    def test_a_symmetry_tied_batch_takes_the_dense_fold_and_says_so_once_per_call(self):
        keep, lines = self._call(2, [[[1, 2], [7, 8, 9]], [], [[4, 5]], []])                # 2 of 4 structures tied, 3 groups
        self.assertFalse(keep)
        self.assertEqual(lines, ["[caliby-opt] LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped reason=symmetry_dense_J impl=atom_mpnn_denoiser origin=kit "
                                 "structures=2/4 groups=3 J=dense arithmetic=stock"])
        keep, lines = self._call(2, [[[1, 2]]])
        self.assertEqual((keep, len(lines)), (False, 1))
        t = report.tally()
        self.assertEqual(t["gates"], {"CALIBY_FAST_POTTS_PARAMS": {"symmetry_dense_J": 2}})   # counted per call
        self.assertEqual(t["fallback_lines"], {})                                             # a [caliby-opt] line is never a counted fallback

    def test_the_idle_sparse_exact_lever_is_named_when_its_switch_is_set(self):
        os.environ["CALIBY_X_SPARSE_EXACT"] = "2"
        keep, lines = self._call(2, [[[1, 2]], []])
        self.assertFalse(keep)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith(" J=dense arithmetic=stock idle=CALIBY_X_SPARSE_EXACT"), lines[0])
        os.environ["CALIBY_X_SPARSE_EXACT"] = "0"
        _, lines = self._call(2, [[[1, 2]], []])
        self.assertNotIn("idle=", lines[0])

    def test_completion_records_the_gates_and_stays_whole(self):
        rep = {"active": True, "mode": "fast", "variant": "single", "switches": modes.resolve("fast", "single").env}
        self._call(2, [[[1, 2]], [[3, 4]]])
        done = activate.completion(rep)
        self.assertEqual(done["lever_gates"], {"CALIBY_FAST_POTTS_PARAMS": {"symmetry_dense_J": 1}})
        self.assertEqual((done["levers_fallback"], done["partial"]), ([], False))            # a declared gate is not a fallback: the run stays the mode's run
        self.assertEqual(done["exit_tally"]["gates"], done["lever_gates"])
        with mock.patch.object(report, "tally", lambda: {"mode": "fast", "lcp_kernel": "available", "fallback_lines": {}, "switches": {}}):
            self.assertEqual(activate.completion(rep)["lever_gates"], {})                    # a tally without the key reads as no gates
        with tempfile.TemporaryDirectory() as d:
            manifest.write(d, dict(rep, **done), command="design", argv=["x"])
            man = manifest.read(d)
        self.assertEqual(man["lever_gates"], {"CALIBY_FAST_POTTS_PARAMS": {"symmetry_dense_J": 1}})
        self.assertFalse(man["partial"])
        self.assertEqual(manifest.build(None)["lever_gates"], {})

    def test_the_lever_file_asks_the_predicate_at_the_symmetry_fold(self):
        """Glue in fast/atom_mpnn_denoiser.py: `compute_potts_params` decides keep-sparse vs upstream's `_fold_symmetry_pos` by
        `caliby_opt.potts_params.keep_sparse(level, symmetry_pos)` — one predicate, no second reading of the symmetry groups there."""
        with open(DENOISER, encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(len(re.findall(r"^import caliby_opt\.potts_params as _x_pp\b", src, re.M)), 1)
        calls = re.findall(r"_x_pp\.keep_sparse\((.*?)\):", src)
        self.assertEqual(calls, ['_fast_potts_params_level(), sampling_inputs["symmetry_pos"]'])
        block = src[src.index("# Handle symmetric decoding by folding the Potts parameters."):]
        block = block[:block.index("# Compute pseudolikelihood.") if "# Compute pseudolikelihood." in block else 900]
        self.assertRegex(block, r"if _x_pp\.keep_sparse\(.*\):\n\s+pass .*\n\s+else:\n\s+potts_decoder_aux = _fold_symmetry_pos\(potts_decoder_aux, sampling_inputs\[\"symmetry_pos\"\]\)")
        self.assertNotIn("_no_groups", src)


class TestFastSamplerWords(unittest.TestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def test_one_served_call_predicate_for_the_sampler_and_the_concurrent_path(self):
        served = ("dlmc", False, True, True)
        self.assertIsNone(fast_sampler.refusal(*served))
        self.assertIsNone(fast_sampler.refusal(*served, return_trajectory=False))
        self.assertEqual(fast_sampler.refusal("dlmc", False, True, False), "CPU tensors")
        cfg = "sampler configuration outside the fast path (proposal/rejection_step/differentiable_penalty/return_trajectory)"
        self.assertEqual(fast_sampler.refusal("chromatic", False, True, True), cfg)
        self.assertEqual(fast_sampler.refusal("dlmc", True, True, True), cfg)
        self.assertEqual(fast_sampler.refusal("dlmc", False, False, True), cfg)
        self.assertEqual(fast_sampler.refusal("dlmc", False, True, True, return_trajectory=True), cfg)
        for args in (("dlmc", False, True, True), ("dlmc", False, True, False), ("chromatic", False, True, True), ("dlmc", True, False, False)):
            self.assertEqual(multiseq.refusal(2, *args), fast_sampler.refusal(*args))         # X010 composes the fast sampler's predicate
        self.assertTrue(multiseq.refusal(1, "dlmc", False, True, True).startswith("CALIBY_FAST_SAMPLER=1 "))

    def test_the_stock_sampler_line_is_a_counted_fallback_of_its_own_lever(self):
        self.assertEqual(registry.LEVERS["CALIBY_FAST_SAMPLER"].probe, ("stderr", "[CALIBY_FAST_SAMPLER]"))
        self.assertEqual(report.FALLBACK_MARKERS["CALIBY_FAST_SAMPLER"], fast_sampler.FALLBACK_MARKER)
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", report._CountingStderr(err)):
            fast_sampler.note_stock("CPU tensors")
            fast_sampler.note_stock("CPU tensors")
        self.assertEqual(err.getvalue().splitlines(), ["[CALIBY_FAST_SAMPLER] CPU tensors -> stock sampler for this call (the upstream sweep loop)"] * 2)
        self.assertEqual(report.tally()["fallback_lines"], {"CALIBY_FAST_SAMPLER": 2})     # the one census of the event: the report's stderr probe
        self.assertFalse(hasattr(fast_sampler, "STATE"))                                   # no second count beside it
        rep = {"active": True, "mode": "fast", "variant": "single", "switches": modes.resolve("fast", "single").env}
        done = activate.completion(rep)
        self.assertEqual((done["levers_fallback"], done["partial"]), (["CALIBY_FAST_SAMPLER"], True))   # its own record, not CALIBY_X_MULTISEQ's

    def test_the_lever_file_uses_the_predicate_and_says_the_line(self):
        """Glue in fast/potts.py `sample_potts`: under CALIBY_FAST_SAMPLER >= 1 the fast sampler runs iff `fast_sampler.refusal(...)` is None;
        otherwise the stock sweep loop runs after `fast_sampler.note_stock(reason)` — no inline copy of the predicate."""
        with open(POTTS, encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(len(re.findall(r"^import caliby_opt\.fast_sampler as _x_fs\b", src, re.M)), 1)
        body = src[src.index("    _lvl = _fast_sampler_level()\n"):]
        body = body[:body.index("    S_reduced, h, J, mask_sample_reduced = normalize_potts_inputs(")]
        self.assertRegex(body, r"if _lvl > 0:\n\s+_why = _x_fs\.refusal\(proposal, rejection_step, differentiable_penalty, bool\(h\.is_cuda\), return_trajectory\)")
        self.assertRegex(body, r"if _why is None:\n\s+return _sample_potts_fast\(")
        self.assertRegex(body, r"\n\s+_x_fs\.note_stock\(_why\)")
        self.assertNotIn('proposal == "dlmc"', body)                                           # the predicate is fast_sampler's, not restated inline
        self.assertEqual(src.count("_x_fs.refusal("), 1)
        self.assertEqual(src.count("_x_fs.note_stock("), 1)


class TestEnvRouteExitGate(unittest.TestCase):
    """`CALIBY_OPT=<mode>` in a host script: the exit tally is registered at activation with the route's exit gate behind it. A short
    `python -S` child stands in for the host: activation is stubbed to the point after `enable()` registered the tally (the row exported,
    the report in place, the overlay finder installed — an empty plan, so every planned module is its file: tree=exact; `mixed` leaves the
    finder out, which is what a lever module loaded from another file amounts to), the kit's own fallback line is written to stderr as the
    lever code would, and the host exits with its own code."""

    HOST = textwrap.dedent('''
        import os, sys
        from caliby_opt import activate, modes, overlay, report
        owned, fallback, host_rc, tree = sys.argv[1] == "owned", sys.argv[2] == "fallback", int(sys.argv[3]), sys.argv[4]
        row = modes.resolve("fast", "single").env
        os.environ.update(row)
        activate._REPORT = {"active": True, "mode": "fast", "variant": "single", "switches": dict(row)}
        if tree == "exact":
            overlay.install({})
        report.register_exit_tally("fast", "exact")
        if owned:
            report.exit_owned()
        if fallback:
            print("[CALIBY_X_MULTISEQ] CPU tensors -> serial fallback (the sequences of this call sampled one call at a time)", file=sys.stderr)
        print("host done")
        sys.exit(host_rc)
    ''')

    def _host(self, owner, lever, host_rc=0, tree="exact"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CALIBY_", "MODEL_OPT"))}
        env["PYTHONPATH"] = fx.pythonpath()
        r = subprocess.run([sys.executable, "-S", "-c", self.HOST, owner, lever, str(host_rc), tree], env=env, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout, [ln for ln in r.stderr.splitlines() if ln]

    def test_a_partial_run_on_the_environment_route_ends_not_active_exit_3(self):
        rc, out, err = self._host("free", "fallback")
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, err)
        self.assertIn("host done", out)                                                       # the host ran to its end; the gate acted at interpreter exit
        exit_lines = [i for i, ln in enumerate(err) if ln.startswith("[caliby-opt] EXIT ")]
        gate_lines = [i for i, ln in enumerate(err) if ln.startswith("[caliby-opt] NOT ACTIVE: fast ran short of its lever set")]
        self.assertEqual((len(exit_lines), len(gate_lines)), (1, 1), err)
        self.assertLess(exit_lines[0], gate_lines[0])                                         # the EXIT census first, then the verdict line
        self.assertIn("partial=CALIBY_X_MULTISEQ", err[exit_lines[0]])
        self.assertIn(" tree=exact ", err[exit_lines[0]])
        self.assertTrue(err[gate_lines[0]].endswith("; the outputs on disk are not a fast run; exit 3; CALIBY_OPT=off runs stock"), err[gate_lines[0]])
        self.assertIn("CALIBY_X_MULTISEQ: 1 kit fallback line(s): [CALIBY_X_MULTISEQ]", err[gate_lines[0]])
        self.assertEqual(sum("NOT ACTIVE" in ln for ln in err), 1, err)                     # the modules were the mode's files: the partial line only

    def test_a_mixed_module_set_on_the_environment_route_ends_not_active_exit_3(self):
        """The design children's fail-closed module rule (`activate.modules_wrong` -> `modules_exit`) holds on this route too: lever modules
        that are not the mode's files (tree=mixed on the EXIT line) end the process NOT ACTIVE, exit 3, with no fallback involved."""
        rc, out, err = self._host("free", "clean", host_rc=0, tree="mixed")
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, err)
        self.assertIn("host done", out)
        exit_i = [i for i, ln in enumerate(err) if ln.startswith("[caliby-opt] EXIT ")]
        gate_i = [i for i, ln in enumerate(err) if ln.startswith("[caliby-opt] NOT ACTIVE: the lever modules loaded in this process were not the exact arm's files (tree=mixed: ")]
        self.assertEqual((len(exit_i), len(gate_i)), (1, 1), err)
        self.assertLess(exit_i[0], gate_i[0])
        self.assertIn(" tree=mixed ", err[exit_i[0]])
        self.assertIn("partial=none", err[exit_i[0]])
        self.assertTrue(err[gate_i[0]].endswith("; outputs are not this arm's; exit 3; CALIBY_OPT=off runs stock"), err[gate_i[0]])
        rc, _, err = self._host("free", "fallback", host_rc=0, tree="mixed")                 # both rules at once: both lines, one exit
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, err)
        self.assertEqual(sum("NOT ACTIVE" in ln for ln in err), 2, err)
        rc, _, err = self._host("owned", "clean", host_rc=0, tree="mixed")                   # the design children decide (modules_exit)
        self.assertEqual(rc, 0, err)
        self.assertFalse(any("NOT ACTIVE" in ln for ln in err), err)

    def test_the_gate_overrides_whatever_the_host_was_exiting_with(self):
        rc, _, err = self._host("free", "fallback", host_rc=1)
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, err)

    def test_a_whole_run_keeps_the_hosts_exit_code_and_says_nothing_more(self):
        for host_rc in (0, 5):
            rc, _, err = self._host("free", "clean", host_rc=host_rc)
            self.assertEqual(rc, host_rc, err)
            self.assertTrue(any(ln.startswith("[caliby-opt] EXIT ") and "partial=none" in ln for ln in err), err)
            self.assertFalse(any("NOT ACTIVE" in ln for ln in err), err)

    def test_an_owned_exit_is_the_callers_to_decide(self):
        rc, _, err = self._host("owned", "fallback")                                          # design_run / stock_design: partial_exit / modules_exit decide
        self.assertEqual(rc, 0, err)
        self.assertTrue(any("partial=CALIBY_X_MULTISEQ" in ln for ln in err), err)           # still recorded on the EXIT line
        self.assertFalse(any("NOT ACTIVE" in ln for ln in err), err)

    def test_the_design_children_own_their_exit_and_the_gate_is_inert_before_activation(self):
        for mod in ("design_run.py", "stock_design.py"):
            with open(os.path.join(os.path.dirname(report.__file__), mod), encoding="utf-8") as fh:
                self.assertEqual(fh.read().count("_report.exit_owned()"), 1, mod)
        saved = dict(report._TALLY)
        try:
            report._TALLY.update(registered=False, exit_owned=False)
            with mock.patch.object(activate, "completion", side_effect=AssertionError("not consulted")):
                self.assertIsNone(report._exit_gate())                                        # nothing registered: nothing to judge
            report._TALLY.update(registered=True, exit_owned=True)
            with mock.patch.object(activate, "completion", side_effect=AssertionError("not consulted")):
                self.assertIsNone(report._exit_gate())                                        # owned: the caller decides
        finally:
            report._TALLY.clear()
            report._TALLY.update(saved)

    def test_the_route_lines_are_the_childrens_lines_plus_the_routes_remedy(self):
        """One formatter and one predicate per rule, shared with the design children: the partial line (`partial_not_active_line`) and the
        module rule's line (`modules_not_active_line` under `activate.modules_wrong`), each with the route's remedy appended."""
        reasons = {"CALIBY_X_CLEAN": "2 kit fallback line(s): [CALIBY_X_CLEAN="}
        whole = {"partial": False, "levers_fallback": [], "modules_state": "exact", "modules_expected": "exact", "modules_wrong": []}
        self.assertEqual(report.env_route_not_active_lines("fast", whole), [])
        self.assertEqual(report.env_route_not_active_lines("off", dict(whole, modules_state="stock", modules_expected="stock")), [])
        partial = dict(whole, partial=True, levers_fallback=["CALIBY_X_CLEAN"], fallback_reasons=reasons)
        self.assertEqual(report.env_route_not_active_lines("fast", partial),
                         [report.partial_not_active_line("fast", report.partial_detail(["CALIBY_X_CLEAN"], reasons), 3) + "; CALIBY_OPT=off runs stock"])
        mixed = dict(whole, modules_state="mixed", modules_wrong=["caliby.api"])
        self.assertTrue(activate.modules_wrong(mixed))
        self.assertFalse(activate.modules_wrong(whole))
        self.assertFalse(activate.modules_wrong({}))                                          # no record (before activation): nothing to judge
        self.assertEqual(report.env_route_not_active_lines("fast", mixed),
                         ["[caliby-opt] NOT ACTIVE: the lever modules loaded in this process were not the exact arm's files (tree=mixed: caliby.api); "
                          "outputs are not this arm's; exit 3; CALIBY_OPT=off runs stock"])
        self.assertEqual(len(report.env_route_not_active_lines("fast", dict(partial, modules_state="mixed", modules_wrong=["caliby.api"]))), 2)
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            self.assertEqual(activate.modules_exit(0, mixed, 3), 3)                          # the design child: the same line with the writer's rc
            self.assertEqual(activate.modules_exit(7, whole, 3), 7)
        self.assertEqual([ln for ln in err.getvalue().splitlines() if ln],
                         [report.modules_not_active_line(mixed, 3) + " (writer rc=0)"])


if __name__ == "__main__":
    unittest.main()
