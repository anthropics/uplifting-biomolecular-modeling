"""Activation rules, on stubs of torch, chai_lab, the eager stack and the DSTEP add-on (no GPU): enable() runs the kit's own W1/W5
statements, applies the recipe, then (fast) installs the mode's eager line through the kits' own builders, and reports what the kits'
state shows; idempotent; a second mode in one process is refused; late activation is refused once the traced ESM is loaded or
chai1.load_exported is no longer upstream's; `exact` installs nothing beyond the kit's statements; `fast` installs tier1 + hoist2 (the value-taint hoister lever)
in-process; `off` applies nothing; unknown modes are refused; the gates refuse by name (kit missing, pins, weights, GPU); a kit install
that fails refuses the mode by name; strict=True raises; the exit tally is
registered before the kit code runs; det=1 applies the recipe's four statements after the kit's statements and before the eager
install; a target-GPU mismatch is a note, never a refusal; a mode whose Triton kernels cannot build on the box (no C compiler) is refused by
name before anything loads."""
import io
import json
import os
import types
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

import pytest

import chai1_opt
from chai1_opt import digest_memo as dm
from chai1_opt import modes, pairtrack, report, stack
from chai1_opt.tests import _stubs


class TestActivation(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests()
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)
        self.eager_saved, self.eager = _stubs.eager_stub(stack)
        self.err = io.StringIO()

    def tearDown(self):
        _stubs.eager_restore(stack, self.eager_saved)
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)
        stack.reset_for_tests()

    def test_exact_applies_the_kits_w1_w5_statements_the_tier1_line_and_templ_empty(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertTrue(rep["active"], rep)
        self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "alloc"])   # + the allocator policy every kit row carries (alloc.py, exported in-process)
        self.assertEqual(rep["levers_not_applicable"], ["W2", "prefetch"])
        self.assertEqual(rep["levels"], "W1,W2,W5"); self.assertEqual(rep["eager"], "tier1"); self.assertEqual(rep["dstep"], "hoist2")
        self.assertEqual(rep["route"], "in-process")
        self.assertFalse(rep["partial"])
        # the kit's own functions sit on the upstream attributes under the eager stack's loader (exact = the driver kit + the tier1 line + the hoist2 DSTEP lever: a private denoiser built by the add-on's builder)
        self.assertEqual(self.eager.dstep.builds, [("tier1", ("hoist2",))]); self.assertIsNotNone(stack._STATE["eager"])
        self.assertEqual(stack._STATE["eager"].parts["diffusion"].hoister, "hoist2"); self.assertIsNone(stack._STATE["eager"].parts["diffusion"].ln_policy)
        from chai1_opt import pairtrack
        self.assertEqual(pairtrack.applied(), ("templ_empty", "exactln", "msa_pad", "transition"))
        h = stack._STATE["eager"]
        self.assertIs(self.chai1.load_exported, h.loader); self.assertEqual(h.orig.__code__.co_filename, stack.driver_path())   # the driver's W1 memo underneath, compiled with the kit file's own name
        self.assertEqual(self.esm.esm_model.__name__, "esm_model_resident")
        self.assertEqual(self.esm._get_esm_contexts_for_sequences.__name__, "_get_ctx_cached")
        ns = stack.kit_namespace()
        self.assertEqual(ns["levels"], {"W1", "W5"}); self.assertIn("_MODULE_CACHE", ns); self.assertIn("ESM_STATS", ns)
        line = self.err.getvalue()
        self.assertIn("[chai1-opt] ACTIVE mode=exact levels=W1,W2,W5 eager=tier1 dstep=hoist2 optin=alloc levers_applied=W1,W5,tier1,hoist2,templ_empty,exactln,msa_pad,transition,rankcc,tailasync,confmemo,alloc not_applicable=W2,prefetch", line)
        self.assertIn("gpu=STUB H100 80GB HBM3(sm90)", line)
        self.assertTrue(report._TALLY["registered"])
        self.assertIn("modules_resident=0", report.exit_tally_line())
        self.assertIn("esm_hits=0 esm_misses=0", report.exit_tally_line())
        self.assertIn(" eager=tier1 ", report.exit_tally_line())                                # the tier1 line's counters ride exact's tally
        # the kit's cached loader (underneath the eager loader) delegates to upstream's function and memoises
        m1 = h.orig("trunk.pt", "cuda:0"); m2 = h.orig("trunk.pt", "cuda:0")
        self.assertIs(m1, m2); self.assertIn("modules_resident=1", report.exit_tally_line())

    def test_fast_tally_names_the_stacks_counted_fallback(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep)
        h = stack._STATE["eager"]
        self.assertIn(" eager=tier1 eager_graph_fallbacks=0 eager_events=0 hoist", report.exit_tally_line()); self.assertNotIn("dstep_ln_calls", report.exit_tally_line())   # no LN lever in the row: no LN counters
        h.events.append((256, "graph_capture_failed:RuntimeError", 0.0))
        self.assertIn(" eager=tier1 eager_graph_fallbacks=1 eager_events=1", report.exit_tally_line())   # the stack's counted fallback, named
        h.parts["diffusion"].ln_policy = _stubs.PolicyStub("big"); h.parts["diffusion"].ln_policy.n_fast = 400; h.parts["diffusion"].ln_policy.n_slow = 40   # an add-on LN policy's counters (the dstep_ln_calls= field the activation tables forbid on the rows) still report
        self.assertIn(" dstep_ln_calls=400/440", report.exit_tally_line())                            # the add-on's own policy counters

    def test_fast_installs_tier1_with_the_dstep_lever_in_process(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep); self.assertEqual(rep["eager"], "tier1"); self.assertEqual(rep["dstep"], "hoist2,compiled,dit_attn")
        self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "tf32", "alloc"])
        self.assertEqual(self.eager.installs, [])                                          # not the stack's install: the add-on's builder over build_parts
        self.assertEqual(self.eager.dstep.builds, [("tier1", ("hoist2", "compiled", "dit_attn"))])
        h = stack._STATE["eager"]
        self.assertIs(self.chai1.load_exported, h.loader)
        self.assertEqual(h.orig.__name__, "load_exported_cached")                          # the driver's W1 memo underneath
        self.assertEqual(stack.eager_installed_levers(h, self.chai1.load_exported), "tier1")
        self.assertEqual(stack.dstep_installed_levers(h, self.chai1.load_exported), ("hoist2", "compiled", "dit_attn"))
        self.assertEqual(h.parts["diffusion"].hoister, "hoist2"); self.assertIsNone(h.parts["diffusion"].ln_policy)   # fast's denoiser: the value-taint hoister, no LN lever
        self.assertIn("[chai1-opt] ACTIVE mode=fast levels=W1,W2,W5 eager=tier1 dstep=hoist2,compiled,dit_attn optin=tf32,alloc levers_applied=W1,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,tf32,alloc not_applicable=W2,prefetch", self.err.getvalue())
        self.assertIn(" eager=tier1 eager_graph_fallbacks=0 eager_events=0 hoist", report.exit_tally_line()); self.assertNotIn("dstep_ln_calls", report.exit_tally_line())

    def test_eager_install_failure_refuses_by_name(self):

        def broken(comps, base, line, lv, **kw):
            raise RuntimeError("no such device")
        self.eager.dstep.build_lever_parts = broken
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("the eager stack (tier1+hoist2,compiled,dit_attn+templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n) failed to install: RuntimeError('no such device')", rep["reason"])
        self.assertIn("NOT ACTIVE", self.err.getvalue())
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported_cached")           # the kit's levers stay; the mode is not active
        with self.assertRaises(chai1_opt.ActivationError):
            with redirect_stderr(self.err):
                stack.reset_for_tests(); chai1_opt.enable("fast", strict=True)

    def _classify_w5_off(self):
        real = stack.classify
        def classify_w5_off(*a, **k):                                   # the kits' state as read: W5 not applied
            c = real(*a, **k)
            c["applied"] = [n for n in c["applied"] if n != "W5"]; c["off"] = sorted(set(c["off"]) | {"W5"})
            return c
        stack.classify = classify_w5_off
        self.addCleanup(setattr, stack, "classify", real)

    def test_dstep_stand_down_hook_moves_the_item_to_the_base_hoister_by_name(self):
        """stack.dstep_stand_down('hoist2', on): the per-item hook a memory line calls (big's 40 GB gate): the adopted denoiser runs the item on
        the base hoister (bitwise identical, smaller hoist cache) and says so; off restores the row's hoister; no denoiser / another hoister -> None."""
        self.assertIsNone(stack.dstep_stand_down("hoist2", True))                            # nothing installed yet
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep)
        dw = stack._STATE["eager"].parts["diffusion"]; self.assertEqual(dw.hoister, "hoist2")
        self.assertEqual(stack.dstep_stand_down("hoist2", True), "base"); self.assertEqual(dw.stand_down, "base"); self.assertEqual(dw.hoister_now(), "base")
        self.assertEqual(stack.dstep_stand_down("hoist2", False), "hoist2"); self.assertIsNone(dw.stand_down)
        self.assertIsNone(stack.dstep_stand_down("compiled", True)); self.assertIsNone(dw.stand_down)   # not the denoiser's hoister: nothing to stand down

    # ---------------------------------------------------------------- MODEL_OPT_LEVERS_OFF: the ablation switch, resolved once in modes.py for every route
    def _env_off(self, value):
        os.environ[stack.ENV_LEVERS_OFF] = value
        self.addCleanup(stack.reset_for_tests)                                                # the table back to the rows as composed
        self.addCleanup(os.environ.pop, stack.ENV_LEVERS_OFF, None)
        modes.apply_levers_off()

    def test_levers_off_env_removes_the_named_levers_from_the_row_before_activation(self):
        """MODEL_OPT_LEVERS_OFF=<a,b>: every named lever the mode composes leaves the row's composition (the one table every consumer reads) and the
        run proceeds as a NAMED partial activation — no opt-out flag needed (the variable is the explicit request)."""
        self._env_off("hoist2, tf32 ,W2,templ_empty,nograph")
        km = modes.kit_mode("fast")
        self.assertEqual((km.levels, km.dstep, km.pairtrack, km.memory_gated, km.implied_optin), (("W1", "W5"), ("compiled", "dit_attn"), ("v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n"), ("msa_chunk",), ("alloc",)))
        self.assertEqual(modes.driver_levels("fast"), "W1,W5"); self.assertEqual(modes.with_implied("fast", ()), ("alloc",))
        self.assertEqual(modes.levers_off_for("fast"), ("W2", "hoist2", "templ_empty", "nograph", "tf32"))          # the row's order
        self.assertEqual(modes.levers_off_for("exact"), ("W2", "hoist2", "templ_empty", "nograph"))                 # exact composes no tf32: not its lever (refused for exact, below)
        self.assertEqual(modes._ROWS["fast"].dstep, ("hoist2", "compiled", "dit_attn"))                                                   # the pristine rows stay
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep); self.assertTrue(rep["partial"]); self.assertNotIn("allow_partial", rep)
        self.assertEqual(rep["levers_off_env"], ["W2", "hoist2", "templ_empty", "nograph", "tf32"]); self.assertEqual(rep["levers_off"], rep["levers_off_env"])
        self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "compiled", "dit_attn", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "alloc"])
        self.assertEqual(self.eager.dstep.builds, [("tier1", ("compiled", "dit_attn"))])                # hoist2 off: the add-on builds the compiled step on the base hoister
        e = self.err.getvalue()
        self.assertIn("[chai1-opt] ACTIVE mode=fast levels=W1,W5 eager=tier1 dstep=compiled,dit_attn optin=alloc levers_applied=W1,W5,tier1,compiled,dit_attn,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,alloc not_applicable=prefetch compile=on:dynamo det=0 route=in-process ", e)   # W2 left the row: no longer `not_applicable` (prefetch, the driver loop's other lever, stays)
        self.assertRegex(e, r"\] ACTIVE .* n_gpu=1 sharding=none PARTIAL off=W2,hoist2,templ_empty,nograph,tf32 \(MODEL_OPT_LEVERS_OFF\)\n")
        self.assertIn("[chai1-opt] NOTE partial activation requested: MODEL_OPT_LEVERS_OFF='hoist2, tf32 ,W2,templ_empty,nograph' leaves lever(s) "
                      "W2,hoist2,templ_empty,nograph,tf32 of mode fast off for this run (an ablation, not a benchmark configuration)", e)
        stack.reset_for_tests(); os.environ.pop(stack.ENV_LEVERS_OFF); modes.apply_levers_off()
        self.assertEqual(modes.kit_mode("fast").dstep, ("hoist2", "compiled", "dit_attn")); self.assertEqual(modes.levers_off_for("fast"), ())   # unset: the rows as composed

    def test_levers_off_env_each_lever_of_each_row_is_switchable_or_refused_by_name(self):
        """Every lever a row composes is individually switchable through the one variable — except the ones the kit cannot remove on its own,
        refused BY NAME with the reason (tier1: the eager line everything rides on; W1: the loader the eager line sits on; big's memory line:
        applied as composed)."""
        for mode in modes.KIT_MODE_NAMES:
            row = modes._ROWS[mode]
            self.assertTrue(modes.row_levers(row))
            for name in modes.row_levers(row):
                env = {stack.ENV_LEVERS_OFF: name}
                why = modes.levers_off_refusal(mode, env)
                if name in ("tier1", "W1") or name in row.memory:
                    self.assertIsNotNone(why, (mode, name)); self.assertIn(f"lever {name} of mode {mode} cannot be switched off on its own", why)
                    self.assertEqual(modes.levers_off_for(mode, env), (), (mode, name))
                else:
                    self.assertIsNone(why, (mode, name)); self.assertEqual(modes.levers_off_for(mode, env), (name,), (mode, name))
                    self.assertNotIn(name, modes.row_levers(modes.without(row, (name,))), (mode, name))
        self.assertEqual(set(modes.row_levers(modes._ROWS["exact"])), {"W1", "W2", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "prefetch", "msa_chunk", "nograph", "alloc"})
        self.assertEqual(set(modes.row_levers(modes._ROWS["fast"])) - set(modes.row_levers(modes._ROWS["exact"])), {"compiled", "dit_attn", "v4trimul", "triattn", "trunk_n", "msa_rows", "tf32"})
        self.assertEqual(set(modes.row_levers(modes._ROWS["big"])) - set(modes.row_levers(modes._ROWS["fast"])), {"trunk_chunk", "opm_chunk"})
        self.assertIn("big's memory line is applied as composed", modes.levers_off_refusal("big", {stack.ENV_LEVERS_OFF: "nograph"}))
        self.assertIsNone(modes.levers_off_refusal("fast", {stack.ENV_LEVERS_OFF: "nograph"}))                    # the gated rows switch it
        self.assertIsNone(modes.levers_off_refusal("off", {stack.ENV_LEVERS_OFF: "anything"}))                     # off ignores the variable (the CLI prints one NOTE)
        self.assertIsNone(modes.refusal("off")); self.assertEqual(modes.levers_off_names({stack.ENV_LEVERS_OFF: " a, ,b,a "}), ("a", "b"))

    def test_levers_off_env_refuses_by_name_a_lever_the_mode_does_not_compose(self):
        self._env_off("hoist2,trunk_chunk")
        want = ("MODEL_OPT_LEVERS_OFF='hoist2,trunk_chunk' names lever(s) trunk_chunk that mode fast does not compose "
                "(its levers: W1,W2,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,prefetch,msa_rows,msa_chunk,nograph,tf32,alloc)")
        self.assertEqual(modes.levers_off_refusal("fast"), want); self.assertEqual(modes.refusal("fast"), want)
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertFalse(rep["active"], rep); self.assertEqual(rep["reason"], want)
        self.assertIn("[chai1-opt] NOT ACTIVE: " + want, self.err.getvalue()); self.assertEqual(self.eager.dstep.builds, []); self.assertIsNone(stack._STATE["eager"])
        with self.assertRaises(chai1_opt.ActivationError):
            stack.reset_for_tests(); _stubs.gates_pass(stack)
            with redirect_stderr(io.StringIO()):
                chai1_opt.enable("fast", strict=True)

    def test_levers_off_env_refuses_by_name_a_lever_the_kit_cannot_switch(self):
        self._env_off("tier1")
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"], rep)
        self.assertTrue(rep["reason"].startswith("MODEL_OPT_LEVERS_OFF='tier1': lever tier1 of mode exact cannot be switched off on its own — the eager line"), rep["reason"])

    def test_levers_off_env_dry_run_names_the_partial(self):
        self._env_off("W5,hoist2")
        with redirect_stderr(self.err):
            rep = stack.activate("exact", dry_run=True)
        self.assertIsNone(rep.get("reason")); self.assertEqual(rep["levers_off"], ["W5", "hoist2"]); self.assertTrue(rep["partial"])
        e = self.err.getvalue()
        self.assertIn("[chai1-opt] DRY-RUN mode=exact levels=W1,W2 eager=tier1 dstep=none optin=alloc levers_applied=W1,W2,tier1,templ_empty,exactln,msa_pad,transition,rankcc,tailasync,confmemo,prefetch,alloc det=0 route=driver", e)
        self.assertRegex(e, r"DRY-RUN .* PARTIAL off=W5,hoist2 \(MODEL_OPT_LEVERS_OFF\)\n"); self.assertIn("NOTE partial activation requested: MODEL_OPT_LEVERS_OFF='W5,hoist2'", e)
        self.assertEqual(stack.ENV_LEVERS_OFF, "MODEL_OPT_LEVERS_OFF"); self.assertNotIn(stack.ENV_LEVERS_OFF, stack.DECLARED_ENV)   # one word across the model-opt kits; not a CHAI1_OPT* name

    def test_partial_activation_is_refused_by_name(self):
        """A lever of the mode the kits' state does not show applied (the run's own switches turned it off): a mode runs with every one of
        its levers or not at all — the activation is REFUSED by name (NOT ACTIVE, the levers named, the opt-out named), on both routes (the
        one rule stack.partial_reason), unless the caller opted out (test_allow_partial_is_recorded_beside_the_note)."""
        self._classify_w5_off()
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"], rep); self.assertTrue(rep["partial"]); self.assertEqual(rep["levers_off"], ["W5"]); self.assertNotIn("allow_partial", rep)
        want = ("partial activation: lever(s) W5 of mode exact not applied (the run's own switches turned them off) — a mode runs with every one of its levers "
                "or not at all; --allow-partial (pred) or CHAI1_OPT_ALLOW_PARTIAL=1 (the CHAI1_OPT route) runs on the levers that did apply")
        self.assertEqual(rep["reason"], want)
        e = self.err.getvalue()
        self.assertIn("[chai1-opt] NOT ACTIVE: " + want, e); self.assertNotIn("] ACTIVE mode=", e); self.assertNotIn("NOTE partial activation", e)
        with self.assertRaises(chai1_opt.ActivationError):
            stack.reset_for_tests(); _stubs.gates_pass(stack); self._classify_w5_off()
            with redirect_stderr(io.StringIO()):
                chai1_opt.enable("exact", strict=True)
        self.assertIsNone(stack.partial_reason({"partial": False})); self.assertIsNone(stack.partial_reason({"partial": False}, allow_partial=True))
        r2 = {"partial": True, "levers_off": ["msa_rows", "nograph"], "mode": "big"}
        self.assertEqual(stack.partial_reason(r2), want.replace("W5 of mode exact", "msa_rows,nograph of mode big")); self.assertNotIn("partial_note", r2)
        self.assertIsNone(stack.partial_reason(r2, allow_partial=True)); self.assertIn("lever(s) msa_rows,nograph not applied", r2["partial_note"]); self.assertTrue(r2["allow_partial"])
        self.assertEqual([stack.allow_partial_from_env({k: v} if v is not None else {}) for k, v in (("CHAI1_OPT_ALLOW_PARTIAL", None), ("CHAI1_OPT_ALLOW_PARTIAL", ""), ("CHAI1_OPT_ALLOW_PARTIAL", "0"), ("CHAI1_OPT_ALLOW_PARTIAL", "1"))],
                         [False, False, False, True])                                       # the CHAI1_OPT route's word (read by _autoload only)
        with self.assertRaises(ValueError) as cm:
            stack.allow_partial_from_env({"CHAI1_OPT_ALLOW_PARTIAL": "yes"})
        self.assertIn("CHAI1_OPT_ALLOW_PARTIAL='yes'", str(cm.exception)); self.assertIn("CHAI1_OPT_ALLOW_PARTIAL", stack.DECLARED_ENV)
        for mod in ("cli.py", "driver.py"):                                                  # the command-line route takes --allow-partial alone: it never reads the env word
            src = open(os.path.join(os.path.dirname(stack.__file__), mod), encoding="utf-8").read()
            self.assertNotIn("allow_partial_from_env", src, mod); self.assertNotIn("ALLOW_PARTIAL", src, mod)
        os.environ["CHAI1_OPT_ALLOW_PARTIAL"] = "1"                                           # set in the environment, the rule itself still refuses: only the caller's explicit argument opts out
        try:
            r3 = {"partial": True, "levers_off": ["W5"], "mode": "exact"}
            self.assertEqual(stack.partial_reason(r3), want); self.assertIsNone(stack.partial_reason(dict(r3), allow_partial=True))
        finally:
            del os.environ["CHAI1_OPT_ALLOW_PARTIAL"]

    def test_allow_partial_is_recorded_beside_the_note(self):
        """--allow-partial (the memory mode's per-item gate opt-out) is recorded as given; the activation rule itself needs no opt-out."""
        self._classify_w5_off()
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact", allow_partial=True)
        self.assertTrue(rep["active"], rep); self.assertTrue(rep["partial"]); self.assertTrue(rep["allow_partial"])
        self.assertEqual(rep["levers_applied"], ["W1", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "alloc"]); self.assertEqual(rep["levers_off"], ["W5"])
        self.assertIn("PARTIAL off=W5", self.err.getvalue()); self.assertIn("[chai1-opt] NOTE partial activation", self.err.getvalue())

    def test_eager_install_rejects_a_lever_not_the_kits_own(self):
        _stubs.eager_restore(stack, self.eager_saved)
        self.eager_saved, self.eager = _stubs.eager_stub(stack, levers=("stock", "tier9"))       # a stack without `tier1`
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("eager lever 'tier1' is not one of the stack's own (stock, tier9)", rep["reason"])

    def test_dstep_install_rejects_a_lever_not_the_addons_own(self):
        _stubs.eager_restore(stack, self.eager_saved)
        self.eager_saved, self.eager = _stubs.eager_stub(stack, dstep_levers=("flnhuge",))          # an add-on without hoist2
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("DSTEP lever(s) hoist2,compiled,dit_attn not among the add-on's own (flnhuge)", rep["reason"])

    def test_det_applies_the_recipe_in_process(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            with redirect_stderr(self.err):
                rep = chai1_opt.enable("fast", det=1)
            self.assertTrue(rep["active"], rep); self.assertEqual(rep["det"], 1)
            self.assertEqual(rep["det_applied"]["level"], 1)
            self.assertEqual(self.torch._C.calls, [("profiling", False)])                     # (1)
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")            # (2)
            self.assertTrue(self.torch.backends.cudnn.deterministic); self.assertFalse(self.torch.backends.cudnn.benchmark)   # (3)
            self.assertEqual(self.torch._det, (True, False))                                  # (4)
            h = stack._STATE["eager"]
            self.assertEqual(h.orig.__name__, "load_exported_cached")                          # the levers first, then the recipe, then the stack
            self.assertIs(self.chai1.load_exported, h.loader)
            self.assertFalse(h.parts["diffusion"].graphed)                                     # the stack's own rule: no graph under the recipe
            self.assertIn(" det=1 route=in-process ", self.err.getvalue())
        finally:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    def test_det_zero_leaves_the_flags_alone(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertTrue(rep["active"]); self.assertEqual(rep["det_applied"], {"level": 0})
        self.assertEqual(self.torch._C.calls, []); self.assertFalse(self.torch.backends.cudnn.deterministic); self.assertFalse(hasattr(self.torch, "_det"))
        self.assertIn(" det=0 route=in-process ", self.err.getvalue())

    def test_target_gpu_mismatch_is_a_note(self):
        os.environ["MODEL_OPT_TARGET_GPU"] = "A100"
        try:
            with redirect_stderr(self.err):
                rep = chai1_opt.enable("exact")
            self.assertTrue(rep["active"]); self.assertEqual(rep["target_gpu"], "A100")
            self.assertEqual(rep["notes"], ["MODEL_OPT_TARGET_GPU=A100 but the GPU is STUB H100 80GB HBM3: not the GPU this configuration targets"])
            self.assertIn("gpu=STUB H100 80GB HBM3(sm90) n_gpu=1 sharding=none notes=MODEL_OPT_TARGET_GPU=A100 but the GPU is STUB H100 80GB HBM3: not the GPU this configuration targets", self.err.getvalue())
        finally:
            os.environ.pop("MODEL_OPT_TARGET_GPU", None)
        self.assertIsNone(stack.target_gpu_note("H100", {"name": "NVIDIA H100 80GB HBM3"}))
        self.assertIsNone(stack.target_gpu_note(None, {"name": "NVIDIA A100"})); self.assertIsNone(stack.target_gpu_note("H100", {"name": None}))
        self.assertIn("but the GPU is NVIDIA A100", stack.target_gpu_note("H100", {"name": "NVIDIA A100"}))

    def test_idempotent_and_one_mode_per_process(self):
        with redirect_stderr(self.err):
            r1 = chai1_opt.enable("exact"); r2 = chai1_opt.enable("exact")
            self.assertIs(r1, r2)
            r3 = chai1_opt.enable("off")
        self.assertFalse(r3["active"]); self.assertIn("already active", r3["reason"])
        self.assertIs(chai1_opt.status(), r1)

    def test_off_applies_nothing(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("off")
        self.assertFalse(rep["active"]); self.assertIn("stock", rep["reason"])
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported")
        self.assertIn("NOT ACTIVE mode=off", self.err.getvalue())

    def test_unknown_mode(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("turbo")
        self.assertFalse(rep["active"]); self.assertIn("unknown mode", rep["reason"])

    def test_late_activation_refused_once_esm_is_loaded(self):
        self.esm._esm_model.append(object())                                      # a fold has started
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("late activation refused", rep["reason"]); self.assertIn("ESM", rep["reason"])
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported")

    def test_late_activation_refused_once_load_exported_is_patched(self):
        def shim(comp_key, device):
            return None
        self.chai1.load_exported = shim                                             # another shim (or the kit driver) got there first
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("not upstream's own function", rep["reason"])

    def test_gates_refuse_by_name(self):
        stack.weights_check = lambda downloads_dir=None: (False, "CHAI_DOWNLOADS_DIR=/nope does not exist", [])
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("CHAI_DOWNLOADS_DIR=/nope", rep["reason"])
        stack.weights_check = lambda downloads_dir=None: (True, "/stub", [])
        stack.pins_check = lambda: (["chai_lab 0.5.0 installed; want 0.6.1"], {})
        stack.reset_for_tests()
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertIn("stock pin: chai_lab 0.5.0", rep["reason"])

    def test_no_gpu_refused(self):
        _stubs.remove(self.saved)
        self.torch, self.chai1, self.esm, self.saved = _stubs.install(cuda_available=False)
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("no CUDA device", rep["reason"])

    def test_dry_run_touches_nothing(self):
        with redirect_stderr(self.err):
            rep = stack.activate("exact", dry_run=True)
        self.assertFalse(rep["active"]); self.assertIsNone(rep["reason"]); self.assertEqual(rep["levers_applied"], ["W1", "W2", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "prefetch", "alloc"])
        self.assertEqual(rep["route"], "driver"); self.assertEqual(rep["levers_not_applicable"], [])
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported"); self.assertEqual(self.eager.installs, [])
        self.assertIn("[chai1-opt] DRY-RUN mode=exact levels=W1,W2,W5 eager=tier1 dstep=hoist2 optin=alloc levers_applied=W1,W2,W5,tier1,hoist2,templ_empty,exactln,msa_pad,transition,rankcc,tailasync,confmemo,prefetch,alloc det=0 route=driver", self.err.getvalue())
        self.assertNotIn("det_applied", rep)                                                # a dry run applies nothing, the recipe included
        with redirect_stderr(self.err):
            rep = stack.activate("exact", dry_run=True, route="in-process")
        self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "alloc"]); self.assertEqual(rep["levers_not_applicable"], ["W2", "prefetch"])
        with redirect_stderr(self.err):
            rep = stack.activate("fast", dry_run=True, route="in-process")
        self.assertIsNone(rep["reason"]); self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "tf32", "alloc"])
        self.assertFalse(chai1_opt.status()["active"]); self.assertIn("has not run", chai1_opt.status()["reason"])
        with redirect_stderr(self.err):
            rep = stack.activate("fast", dry_run=True)
        self.assertIsNone(rep["reason"]); self.assertEqual(rep["levers_applied"], ["W1", "W2", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "prefetch", "tf32", "alloc"])
        self.assertIn("[chai1-opt] DRY-RUN mode=fast levels=W1,W2,W5 eager=tier1 dstep=hoist2,compiled,dit_attn optin=tf32,alloc levers_applied=W1,W2,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,prefetch,tf32,alloc compile=on:dynamo det=0 route=driver", self.err.getvalue())


class TestKitMissing(unittest.TestCase):
    def test_kit_missing_named(self):
        stack.reset_for_tests()
        os.environ[stack.ENV_KIT] = "/nonexistent/kit"
        try:
            with redirect_stderr(io.StringIO()):
                rep = stack.activate("exact", dry_run=True)
            self.assertIn("kit file missing: /nonexistent/kit", rep["reason"])
        finally:
            os.environ.pop(stack.ENV_KIT)


REAL_MODE_STACK_CHECK = stack.mode_stack_check                                # gates_pass stubs it (the box builds every row); these tests read the real one



class TestOOMPropagates(unittest.TestCase):
    """An out-of-memory error inside a lever install / the allocator export propagates out of the served entry points (chai1_opt.enable,
    alloc.export) — never converted to a NOT ACTIVE refusal or a lever's refusal by name; any other exception keeps its refusal."""
    setUp = TestActivation.setUp
    tearDown = TestActivation.tearDown

    def _oom_class(self):
        class OutOfMemoryError(RuntimeError):           # torch.cuda.OutOfMemoryError under the stub torch: opt_core.oom.is_oom matches torch's class (by name here)
            pass
        self.torch.cuda.OutOfMemoryError = OutOfMemoryError; self.torch.OutOfMemoryError = OutOfMemoryError
        return OutOfMemoryError

    def test_oom_in_the_eager_install_propagates_out_of_enable(self):
        OOM = self._oom_class()

        def oom(comps, base, line, lv, **kw):
            raise OOM("CUDA out of memory. Tried to allocate 2.00 GiB (mocked)")
        self.eager.dstep.build_lever_parts = oom
        with redirect_stderr(self.err):
            with self.assertRaises(OOM):
                chai1_opt.enable("fast")
        self.assertNotIn("failed to install", self.err.getvalue())                      # no refusal was composed around it
        self.assertFalse((stack._STATE["report"] or {}).get("active"))

    def test_any_other_install_error_keeps_its_refusal_by_name(self):
        self._oom_class()

        def broken(comps, base, line, lv, **kw):
            raise RuntimeError("no such device")
        self.eager.dstep.build_lever_parts = broken
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("failed to install: RuntimeError('no such device')", rep["reason"])

    def test_oom_in_the_allocator_export_propagates_out_of_the_lever(self):
        from unittest import mock
        from chai1_opt import alloc
        OOM = self._oom_class()
        with mock.patch.object(alloc.core(), "export", side_effect=OOM("CUDA out of memory (mocked)")):
            with self.assertRaises(OOM):
                alloc.export({})
        with mock.patch.object(alloc.core(), "export", side_effect=RuntimeError("graphs on")):
            with self.assertRaises(alloc.LeverUnavailable):
                alloc.export({})


class TestWeightsDigest(unittest.TestCase):
    """The checkpoint's sha pin is a note, never a gate: an unknown checkpoint prints a WARNING by name and every entry point
    proceeds (exit 0); the pinned checkpoint reads weights=pinned."""
    setUp = TestActivation.setUp
    tearDown = TestActivation.tearDown
    UNKNOWN = {"status": "unknown", "files": 8, "unknown": [{"local": "models_v2/trunk.pt", "sha256": "ab" * 32}], "seconds": 0.0, "hashed": 8, "cached": 0, "cached_utc": None, "words": "unknown"}
    PINNED = {"status": "pinned", "files": 8, "unknown": [], "seconds": 0.0, "hashed": 8, "cached": 0, "cached_utc": None, "words": "pinned"}
    CACHED = {"status": "pinned", "files": 8, "unknown": [], "seconds": 0.0, "hashed": 0, "cached": 8, "cached_utc": "2026-09-03T09:00:00Z", "words": "pinned (cached digest 2026-09-03T09:00:00Z)"}

    def test_hashing_names_pinned_and_unknown_files(self):
        import hashlib, tempfile
        with tempfile.TemporaryDirectory() as d, _memo_home():
            files = []
            for name, data in (("a.pt", b"pinned bytes"), ("sub/b.pt", b"also pinned bytes")):
                os.makedirs(os.path.dirname(os.path.join(d, name)) or d, exist_ok=True); open(os.path.join(d, name), "wb").write(data)
                files.append({"local": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            saved = stack.pins; stack.pins = lambda: {"weights": {"files": files}}
            try:
                real = _stubs_saved_weights_match(self)
                self.assertEqual(real(d)["status"], "pinned"); self.assertEqual(real(d)["files"], 2)
                open(os.path.join(d, "sub/b.pt"), "wb").write(b"another checkpoint")
                wid = real(d)
                self.assertEqual(wid["status"], "unknown"); self.assertEqual([u["local"] for u in wid["unknown"]], ["sub/b.pt"])
                self.assertEqual((wid["hashed"], wid["cached"]), (1, 1))                       # the unchanged file's digest came from the memo
                self.assertEqual(stack.weights_note(wid), f"WEIGHTS unknown sha=sub/b.pt:{wid['unknown'][0]['sha256'][:12]} (1/2 files not the pinned checkpoint, stock/PINS.json); proceeding")
                self.assertEqual(wid["words"], f"unknown (cached digest {wid['cached_utc']})")          # the manifest's words (digest_memo.word)
                self.assertIsNone(stack.weights_note({"status": "pinned", "files": 2, "unknown": []}))
            finally:
                stack.pins = saved

    def test_unknown_checkpoint_warns_by_name_and_proceeds(self):
        from chai1_opt import cli
        from contextlib import redirect_stdout
        stack.weights_match = lambda d, afresh=False: dict(self.UNKNOWN)
        out = io.StringIO()
        with redirect_stderr(self.err), redirect_stdout(out):
            self.assertEqual(cli.main(["check", "--mode", "fast"]), cli.EXIT_OK)
            self.assertEqual(cli.main(["check", "--mode", "exact"]), cli.EXIT_OK)
            self.assertEqual(cli.main(["check", "--mode", "big"]) in (cli.EXIT_OK, cli.EXIT_NOT_ACTIVE), True)   # big's memory line is not the point here
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["weights"]["status"], "unknown")
        e = self.err.getvalue()
        self.assertIn("weights=unknown(1/8)", e)
        self.assertIn("WEIGHTS unknown sha=models_v2/trunk.pt:abababababab (1/8 files not the pinned checkpoint, stock/PINS.json); proceeding", e)
        self.assertNotIn("NOT ACTIVE: WEIGHTS", e)

    def test_pinned_checkpoint_reads_pinned(self):
        from chai1_opt import cli
        from contextlib import redirect_stdout
        stack.weights_match = lambda d, afresh=False: dict(self.PINNED)
        with redirect_stderr(self.err), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["check", "--mode", "fast"]), cli.EXIT_OK)
        e = self.err.getvalue(); self.assertIn("weights=pinned", e); self.assertIn(" weights_digest=fresh", e); self.assertNotIn("WEIGHTS unknown", e); self.assertNotIn("cached", e)
        self.err.truncate(0); self.err.seek(0); stack.reset_for_tests()
        stack.weights_match = lambda d, afresh=False: dict(self.CACHED)                         # a memo hit: the words name the cached digest and its utc
        with redirect_stderr(self.err), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["check", "--mode", "fast"]), cli.EXIT_OK)
        line = [l for l in self.err.getvalue().splitlines() if " DRY-RUN " in l or " ACTIVE " in l][-1]
        self.assertIn(" weights=pinned ", line); self.assertTrue(line.endswith(" weights_digest=cached@2026-09-03T09:00:00Z"), line)   # the memo state: ONE whitespace-free token, last
        self.assertNotIn("cached digest", line)                                                     # the human words live in the manifest, never inside an ACTIVE-line token
        self.assertRegex(line.rsplit(" ", 1)[1], r"^weights_digest=\S+$")


    def test_pred_gate_reads_the_digest_memo_check_hashes_afresh(self):      # QoL: pred's pre-flight dry run does not re-hash the 8 pinned files (6-7 s) every run
        seen = []
        stack.weights_match = lambda d, afresh=False: (seen.append(bool(afresh)), dict(self.PINNED))[1]
        with redirect_stderr(io.StringIO()):
            stack.activate("exact", dry_run=True, trigger="pred"); stack.activate("exact", dry_run=True, trigger="check"); stack.activate("exact", dry_run=True)
        self.assertEqual(seen, [False, True, True])
    def test_active_line_grammar_with_a_warm_memo(self):
        """The ACTIVE line is whitespace-free key=value tokens up to `gpu=` (the pinned ACTIVE grammar) and keeps its shape with a warm
        memo: `weights=pinned` as before, the memo state as ONE trailing whitespace-free token `weights_digest=cached@<utc>`, the
        words `(cached digest <utc>)` nowhere on the line."""
        import re as _re
        stack.weights_match = lambda d, afresh=False: dict(self.CACHED)
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep.get("reason"))
        line = [l for l in self.err.getvalue().splitlines() if l.startswith("[chai1-opt] ACTIVE ")][-1]
        self.assertRegex(line, r"^\[chai1-opt\] ACTIVE mode=fast levels=\S+ eager=\S+ dstep=\S+ optin=\S+ levers_applied=\S+(?: (?:dstep_table|not_applicable)=\S+)* compile=on:(?:aoti|dynamo) det=\d route=in-process chai_lab=\S+ torch=\S+ gpu=.+ weights=pinned(?: .*)? weights_digest=cached@2026-09-03T09:00:00Z$")
        self.assertNotIn("cached digest", line); self.assertNotRegex(line.rsplit(" ", 1)[1], r"\s")
        self.assertEqual(rep["weights"]["words"], "pinned (cached digest 2026-09-03T09:00:00Z)")            # the manifest carries the words


def _stubs_saved_weights_match(case):
    """The real stack.weights_match (gates_pass replaced it with a stub; the saved tuple holds the original)."""
    return case.gates[4]


from contextlib import contextmanager


@contextmanager
def _memo_home():
    """A temporary XDG_CACHE_HOME so the digest memo (stack.weights_memo_path) lives in the test's own directory."""
    import tempfile
    saved = os.environ.get("XDG_CACHE_HOME")
    with tempfile.TemporaryDirectory() as h:
        os.environ["XDG_CACHE_HOME"] = h
        try:
            yield h
        finally:
            if saved is None: os.environ.pop("XDG_CACHE_HOME", None)
            else: os.environ["XDG_CACHE_HOME"] = saved


class TestWeightsDigestMemo(unittest.TestCase):
    """stack.weights_match goes through the kit-local digest memo (chai1_opt/digest_memo.py; memo file
    <XDG_CACHE_HOME or ~/.cache>/chai1_opt/weights_digests.json): a hit hashes nothing and the words carry `(cached digest <utc>)`; `check`
    (every dry run) passes refresh=True, `pred` / `enable` refresh=False; a file matches by digest only — size/mtime/inode select the memo entry,
    never decide the match."""
    setUp = TestActivation.setUp
    tearDown = TestActivation.tearDown

    def _tree(self, d):
        import hashlib
        files = []
        for name, data in (("a.pt", b"pinned weights A"), ("sub/b.pt", b"pinned weights B")):
            p = os.path.join(d, name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
            files.append({"local": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return files

    def _run(self, files, fn):
        saved = stack.pins; stack.pins = lambda: {"weights": {"files": files}}
        try:
            return fn()
        finally:
            stack.pins = saved

    def test_memo_name_and_dir_are_the_kits_own(self):
        from chai1_opt import digest_memo
        self.assertEqual(digest_memo.MEMO_NAME, "weights_digests.json")
        self.assertEqual(stack.weights_memo_dir(), os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "chai1_opt"))

    def test_match_through_the_memo_hit_words_change_and_afresh(self):
        import tempfile
        from chai1_opt import digest_memo
        real = _stubs_saved_weights_match(self)
        with tempfile.TemporaryDirectory() as d, _memo_home() as home:
            files = self._tree(d); memo = os.path.join(home, "chai1_opt", digest_memo.MEMO_NAME)
            first = self._run(files, lambda: real(d))                                     # every file hashed in full, entries written after
            self.assertEqual((first["status"], first["hashed"], first["cached"], first["cached_utc"], first["memo"]), ("pinned", 2, 0, None, memo))
            self.assertEqual((stack.weights_digest_token(first), first["words"]), ("fresh", "pinned")); self.assertTrue(os.path.isfile(memo))
            self.assertEqual(sorted(e["sha256"] for e in json.load(open(memo)).values()), sorted(f["sha256"] for f in files))
            hit = self._run(files, lambda: real(d))                                       # the entries select: no hashing, the cached words
            self.assertEqual((hit["status"], hit["hashed"], hit["cached"]), ("pinned", 0, 2)); self.assertRegex(hit["cached_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            self.assertEqual((stack.weights_digest_token(hit), hit["words"]), (f"cached@{hit['cached_utc']}", f"pinned (cached digest {hit['cached_utc']})"))
            open(os.path.join(d, "sub/b.pt"), "wb").write(b"another checkpoint!!")        # size and mtime move: that file is hashed again; matched by the NEW digest
            chg = self._run(files, lambda: real(d))
            self.assertEqual((chg["status"], chg["hashed"], chg["cached"]), ("unknown", 1, 1)); self.assertEqual([u["local"] for u in chg["unknown"]], ["sub/b.pt"])
            self.assertEqual(stack.weights_note(chg), f"WEIGHTS unknown sha=sub/b.pt:{chg['unknown'][0]['sha256'][:12]} (1/2 files not the pinned checkpoint, stock/PINS.json); proceeding")
            self.assertEqual(chg["words"], f"unknown (cached digest {chg['cached_utc']})")
            fresh = self._run(files, lambda: real(d, afresh=True))                        # check: every file hashed afresh on a hit, entries rewritten, no cached words
            self.assertEqual((fresh["hashed"], fresh["cached"], fresh["afresh"]), (2, 0, True)); self.assertEqual((stack.weights_digest_token(fresh), fresh["words"]), ("fresh", "unknown"))

    def test_check_passes_refresh_true_and_pred_enable_refresh_false(self):
        import tempfile
        from chai1_opt import cli, digest_memo
        from contextlib import redirect_stdout
        real = _stubs_saved_weights_match(self); stack.weights_match = real                # the real digest function over a mocked helper
        with tempfile.TemporaryDirectory() as d, _memo_home():
            files = self._tree(d); by_path = {os.path.join(d, f["local"]): f["sha256"] for f in files}
            seen = []; saved = (digest_memo.digest, stack.pins, stack.weights_check)
            digest_memo.digest = lambda path, memo_dir, refresh=False, hasher=None: (seen.append((os.path.basename(path), memo_dir, refresh)), (by_path[path], None))[1]
            stack.pins = (lambda orig: (lambda: dict(orig(), weights={"files": files})))(saved[1]); stack.weights_check = lambda: (True, d, [])
            try:
                with redirect_stderr(self.err), redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["check", "--mode", "fast"]), cli.EXIT_OK)
                    n_check = len(seen)
                    rep = chai1_opt.enable("fast")
            finally:
                digest_memo.digest, stack.pins, stack.weights_check = saved
            self.assertTrue(rep["active"], rep.get("reason"))
            self.assertEqual([r for _, _, r in seen[:n_check]], [True, True]); self.assertEqual([r for _, _, r in seen[n_check:]], [False, False])
            self.assertEqual({m for _, m, _ in seen}, {stack.weights_memo_dir()})
            self.assertIn("weights=pinned", self.err.getvalue()); self.assertIn(" weights_digest=fresh", self.err.getvalue())

    def test_an_unwritable_memo_dir_is_named_and_digests_are_computed_afresh(self):
        """A read-only cache root (0o555, a read-only mount) never refuses: every file is hashed afresh, nothing is memoised, the activation
        line carries `WEIGHTS memo unwritable (<dir>: <error>): digests computed afresh, not memoised` and `weights_digest=fresh`."""
        import stat, tempfile
        from chai1_opt import digest_memo
        real = _stubs_saved_weights_match(self)
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as home:
            files = self._tree(d)
            ro = os.path.join(home, "ro"); os.mkdir(ro); os.chmod(ro, 0o555)                       # a read-only cache root
            homes = [ro]
            blocker = os.path.join(home, "afile"); open(blocker, "w").write("x"); homes.append(blocker)    # a cache root under a plain file: unwritable for root too
            saved_env = os.environ.get("XDG_CACHE_HOME")
            try:
                for h in homes:
                    os.environ["XDG_CACHE_HOME"] = h
                    memo_dir = stack.weights_memo_dir()
                    if h == ro and os.geteuid() == 0:                                                  # root writes through 0o555: the file-parent case below carries the branch
                        continue
                    why = stack.memo_dir_unwritable(memo_dir)
                    self.assertIsNotNone(why); self.assertTrue(why.startswith(memo_dir + ": "), why)
                    for afresh in (True, False):                                                       # check and pred alike
                        wid = self._run(files, lambda: real(d, afresh=afresh))
                        self.assertEqual((wid["status"], wid["hashed"], wid["cached"], wid["memo_unwritable"]), ("pinned", 2, 0, why))
                        self.assertEqual(stack.weights_digest_token(wid), "fresh")
                        self.assertEqual(stack.weights_memo_note(wid), f"WEIGHTS memo unwritable ({why}): digests computed afresh, not memoised")
                    self.assertFalse(os.path.exists(os.path.join(memo_dir, digest_memo.MEMO_NAME)))
                os.environ["XDG_CACHE_HOME"] = home                                                    # a writable root again: memoised, no note
                wid = self._run(files, lambda: real(d))
                self.assertIsNone(wid["memo_unwritable"]); self.assertIsNone(stack.weights_memo_note(wid))
            finally:
                os.chmod(ro, stat.S_IRWXU)
                if saved_env is None: os.environ.pop("XDG_CACHE_HOME", None)
                else: os.environ["XDG_CACHE_HOME"] = saved_env

    def test_check_on_a_read_only_memo_dir_exits_0_with_the_note(self):
        from chai1_opt import cli
        from contextlib import redirect_stdout
        wid = dict(TestWeightsDigest.PINNED, memo_unwritable="/ro/chai1_opt: Read-only file system")
        stack.weights_match = lambda d, afresh=False: dict(wid)
        with redirect_stderr(self.err), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["check", "--mode", "fast"]), cli.EXIT_OK)
        line = [l for l in self.err.getvalue().splitlines() if " DRY-RUN " in l][-1]
        self.assertIn("WEIGHTS memo unwritable (/ro/chai1_opt: Read-only file system): digests computed afresh, not memoised", line)
        self.assertIn(" weights=pinned ", line); self.assertTrue(line.endswith(" weights_digest=fresh"), line)


class TestMsaForm(unittest.TestCase):
    """The driver's MSA-form statement (stack.install_msa_form): every feature context the worker builds gets chai-lab's own MSA form for
    the item — the off route's rule (stock_fold.msa_directory_for): the directory when a chain's .aligned.pqt is there, else None."""

    def test_the_wrapper_applies_the_off_routes_rule_and_records_it(self):
        import types
        from chai1_opt import stock_fold
        calls = []
        fake = types.SimpleNamespace(make_all_atom_feature_context=lambda **kw: calls.append(kw) or "ctx")
        saved = stock_fold.msa_directory_for; err = io.StringIO(); stack.MSA_FORMS.clear()
        verdicts = {"/inp/a/a.chai.fasta": ("/inp/msas", ["x.aligned.pqt"], 2), "/inp/b/b.chai.fasta": (None, [], 1)}
        depths = {"/inp/a/a.chai.fasta": (2, [3, 0]), "/inp/b/b.chai.fasta": (1, [0])}
        saved_depths = stock_fold.msa_depths
        stock_fold.msa_directory_for = lambda fasta, msa_dir: verdicts[fasta]; stock_fold.msa_depths = lambda fasta, msa_dir: depths[fasta]
        try:
            stack.install_msa_form(fake); stack.install_msa_form(fake)                       # idempotent
            self.assertEqual(fake.make_all_atom_feature_context.chai1_opt_lever, "msa_form")
            with redirect_stderr(err):
                r1 = fake.make_all_atom_feature_context(fasta_file="/inp/a/a.chai.fasta", msa_directory="/inp/msas", esm_device="cuda:0")
                r2 = fake.make_all_atom_feature_context(fasta_file="/inp/b/b.chai.fasta", msa_directory="/inp/msas", esm_device="cuda:0")
        finally:
            stock_fold.msa_directory_for = saved; stock_fold.msa_depths = saved_depths
        self.assertEqual((r1, r2), ("ctx", "ctx"))
        self.assertIn("[chai1-opt] MSA item=a.chai.fasta chains=2 depth=3,0 dir=/inp/msas", err.getvalue())   # the per-item depth line (kit prefix), beside the unchanged form line
        self.assertIn("[chai1-opt] MSA item=b.chai.fasta chains=1 depth=0 dir=/inp/msas", err.getvalue())
        self.assertEqual([c["msa_directory"] for c in calls], ["/inp/msas", None])                 # b has no alignment there: upstream's empty MSA context, as the off route folds it
        self.assertEqual([c["esm_device"] for c in calls], ["cuda:0", "cuda:0"])                  # every other keyword passes through
        self.assertEqual([(r["item"], r["msa_form"], r["aligned_pqt_present"], r["n_chains"]) for r in stack.MSA_FORMS],
                         [("a.chai.fasta", "directory", 1, 2), ("b.chai.fasta", "none", 0, 1)])
        self.assertEqual(stack.msa_form_fields(), "msa_form_directory=1 msa_form_none=1")
        self.assertIn("[chai1-opt] MSA form=none item=b.chai.fasta aligned_pqt=0/1 (chai-lab's own form, the off route's rule)", err.getvalue())
        stack.MSA_FORMS.clear()

class TestModeStack(unittest.TestCase):
    """``stack.mode_stack_check`` is a capability check of the box, nothing else: fast / big (``stack_rule`` triton) need a C compiler for
    their Triton kernels' launcher build — refused by name without one (the NOT ACTIVE line, exit 3, before weights / GPU are read and before
    anything loads), nothing noted with one; exact (``stack_rule`` bitwise) needs nothing on the box and is never refused or noted here. No
    container tag is read anywhere: the torch / CUDA stack is a RECORD (``stack.stack_pinning`` -> ``stack_pinned`` / ``stack_line`` on
    every report, the NOTE line) and a refusal by name only under ``CHAI1_OPT_STRICT_STACK=1``."""

    UNCERT = {"stack_pinned": False, "stack_line": "STACK not pinned: torch 2.5.1+cu124 (pinned: torch 2.13.0+cu130)", "stack_strict": False}

    def setUp(self):
        stack.reset_for_tests()
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)
        stack.mode_stack_check = REAL_MODE_STACK_CHECK
        self.eager_saved, self.eager = _stubs.eager_stub(stack)
        self.env_saved = {k: os.environ.get(k) for k in (stack.ENV_STRICT_STACK, "CC")}
        self.cc_saved = stack.c_compiler
        self.err = io.StringIO()

    def tearDown(self):
        stack.c_compiler = self.cc_saved
        for k, v in self.env_saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        _stubs.eager_restore(stack, self.eager_saved)
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)
        stack.reset_for_tests()

    def test_the_function(self):
        fast, big, exact = modes.kit_mode("fast"), modes.kit_mode("big"), modes.kit_mode("exact")
        self.assertEqual((exact.stack_rule, fast.stack_rule, big.stack_rule), ("bitwise", "triton", "triton"))
        self.assertEqual((exact.stack, fast.stack, big.stack), (modes.PINNED_STACK,) * 3)                          # documentation: where the tree tested the rows
        self.assertEqual(REAL_MODE_STACK_CHECK(exact, compiler=""), (None, None))                                        # exact: nothing on the box, compiler or not
        self.assertEqual(REAL_MODE_STACK_CHECK(exact, compiler="/usr/bin/gcc"), (None, None))
        self.assertEqual(REAL_MODE_STACK_CHECK(fast, compiler="/usr/bin/gcc"), (None, None))                            # a compiler: fast / big build, nothing noted
        self.assertEqual(REAL_MODE_STACK_CHECK(big, compiler="/usr/bin/gcc"), (None, None))
        refusal, note = REAL_MODE_STACK_CHECK(fast, compiler="")
        self.assertIsNone(note)
        self.assertEqual(refusal, "mode fast needs a C compiler for its Triton kernels ($CC / cc / gcc / clang: none on PATH) — --mode exact and --mode off need none")
        refusal, note = REAL_MODE_STACK_CHECK(big, compiler="")
        self.assertTrue(refusal.startswith("mode big needs a C compiler for its Triton kernels"), refusal)

    def test_c_compiler_reads_cc_then_path(self):
        os.environ["CC"] = sys.executable                                                                               # any resolvable program passes for $CC
        self.assertEqual(stack.c_compiler(), sys.executable)
        os.environ["CC"] = "/nonexistent/cc"; os.environ["PATH"]                                                          # unresolvable $CC falls through to PATH
        self.assertEqual(stack.c_compiler(), next((p for p in map(__import__("shutil").which, stack.C_COMPILERS) if p), None))

    def test_default_mode_without_a_compiler_is_not_active_by_name_before_weights_and_gpu(self):
        stack.c_compiler = lambda: None
        stack.weights_check = lambda downloads_dir=None: (False, "CHAI_DOWNLOADS_DIR=/nope does not exist", [])          # a later gate: never reached
        with redirect_stderr(self.err):
            rep = chai1_opt.enable(modes.DEFAULT_MODE)
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertFalse(rep["active"]); self.assertTrue(rep["reason"].startswith("mode fast needs a C compiler for its Triton kernels"), rep["reason"])
        self.assertIn("— --mode exact and --mode off need none", rep["reason"])
        self.assertIn("[chai1-opt] NOT ACTIVE: mode fast needs a C compiler for its Triton kernels", self.err.getvalue())
        self.assertEqual(self.eager.installs, [])                                                                         # nothing loaded
        stack.reset_for_tests()
        _stubs.gates_restore(stack, self.gates); self.gates = _stubs.gates_pass(stack); stack.mode_stack_check = REAL_MODE_STACK_CHECK; stack.c_compiler = lambda: None
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("exact")                                                                               # exact on that box: runs (it compiles nothing)
        self.assertTrue(rep["active"], rep.get("reason"))

    def test_with_a_compiler_fast_runs_and_nothing_is_noted(self):
        stack.c_compiler = lambda: "/usr/bin/gcc"
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["notes"], [])
        self.assertEqual((rep["stack_pinned"], rep["stack_line"]), (True, None))                                      # gates_pass: the pinned stack — no NOTE line
        self.assertNotIn(" NOTE STACK", self.err.getvalue())

    def test_a_stack_not_pinned_is_a_record_on_every_route_not_a_refusal(self):
        stack.c_compiler = lambda: "/usr/bin/gcc"; stack.stack_pinning = lambda environ=None: dict(self.UNCERT)
        for mode in ("exact", "fast"):
            stack.reset_for_tests(); self.err = io.StringIO()
            _stubs.remove(self.saved); self.torch, self.chai1, self.esm, self.saved = _stubs.install()                    # a fresh upstream per mode (a process has one mode)
            with redirect_stderr(self.err):
                rep = chai1_opt.enable(mode)
            self.assertTrue(rep["active"], rep.get("reason"))                                                            # runs: the stack is recorded, not gated
            self.assertEqual((rep["stack_pinned"], rep["stack_line"]), (False, self.UNCERT["stack_line"]))
            self.assertEqual(rep["notes"], [])                                                                            # the ACTIVE line's notes= field is untouched; the record has its own NOTE line
            self.assertEqual(self.err.getvalue().count(f"[chai1-opt] NOTE {self.UNCERT['stack_line']}\n"), 1, self.err.getvalue())
        from chai1_opt import cli
        out = io.StringIO(); stack.reset_for_tests(); self.err = io.StringIO()
        with redirect_stderr(self.err), __import__("contextlib").redirect_stdout(out):
            rc = cli.main(["check", "--mode", "off", "--json"])                                                             # off carries the record too (every route starts at base_report)
        self.assertEqual(rc, 0, self.err.getvalue()); j = json.loads(out.getvalue())
        self.assertEqual((j["stack_pinned"], j["stack_line"]), (False, self.UNCERT["stack_line"]))
        self.assertIn(f"[chai1-opt] NOTE {self.UNCERT['stack_line']}", self.err.getvalue())

    def test_strict_switch_refuses_a_stack_not_pinned_by_name_exit_3(self):
        from chai1_opt import cli
        stack.c_compiler = lambda: "/usr/bin/gcc"; stack.stack_pinning = lambda environ=None: dict(self.UNCERT, stack_strict=True)
        out = io.StringIO()
        for argv in (["check", "--mode", "exact"], ["check"]):                                                          # every kit mode (the default, fast, included): the strict form
            stack.reset_for_tests(); self.err = io.StringIO()
            with redirect_stderr(self.err), __import__("contextlib").redirect_stdout(out):
                rc = cli.main(argv)
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, (argv, self.err.getvalue()))
            self.assertIn(f"NOT ACTIVE: {self.UNCERT['stack_line']} — refused under {stack.ENV_STRICT_STACK}=1", self.err.getvalue())
        rep = stack.base_report("off", "pred", False)                                                              # the stock route gates too (cli._pred_stock -> stack.gates): off included
        self.assertTrue((stack.gates(rep, need_gpu=False) or "").startswith(f"{self.UNCERT['stack_line']} — refused under {stack.ENV_STRICT_STACK}=1"))
        stack.stack_pinning = lambda environ=None: {"stack_pinned": True, "stack_line": None, "stack_strict": True}   # strict on the pinned stack: runs
        stack.reset_for_tests(); self.err = io.StringIO()
        with redirect_stderr(self.err), __import__("contextlib").redirect_stdout(out):
            rc = cli.main(["check", "--mode", "exact"])
        self.assertEqual(rc, 0, self.err.getvalue())
        self.assertEqual(stack.strict_stack_refusal({"stack_pinned": None, "stack_line": "STACK unknown: x", "stack_strict": False})[:15], "STACK unknown: ")   # a record that could not be computed is refused, never assumed pinned

    def test_strict_variable_in_the_real_environment(self):
        """The real ``stack_pinning`` (no stub): the variable's words — 1 strict, 0 / unset permissive, anything else named."""
        _stubs.gates_restore(stack, self.gates); self.gates = _stubs.gates_pass(stack, pinned=False)
        os.environ[stack.ENV_STRICT_STACK] = "1"
        self.assertTrue(stack.stack_pinning()["stack_strict"])
        os.environ[stack.ENV_STRICT_STACK] = "0"
        self.assertFalse(stack.stack_pinning()["stack_strict"])
        os.environ[stack.ENV_STRICT_STACK] = "please"
        with self.assertRaises(ValueError):
            stack.stack_pinning()
        rep = stack.base_report("exact", None, True)                                                               # base_report names it on the record; gates refuses on it
        self.assertIsNone(rep["stack_pinned"]); self.assertIn("CHAI1_OPT_STRICT_STACK='please'", rep["stack_line"])
        self.assertTrue(stack.strict_stack_refusal(rep).startswith("STACK unknown: "))
        os.environ.pop(stack.ENV_STRICT_STACK)
        self.assertIn(stack.ENV_STRICT_STACK, stack.DECLARED_ENV)                                                        # declared: not an undeclared-variable refusal
        cert = stack.stack_pinning()                                                                                # this test box: whatever torch is here, the record is computed and worded
        self.assertIn(cert["stack_pinned"], (True, False))
        self.assertTrue(cert["stack_line"] is None if cert["stack_pinned"] else cert["stack_line"].startswith("STACK not pinned: torch "))


# --- chai1_opt.digest_memo: the kit-local weights-digest cache, in isolation (pytest-style; no activation stubs needed) ---

def _write(p, data):
    p.write_bytes(data)
    return str(p)


def test_memo_is_written_only_after_a_full_hash(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)

    def interrupted(_):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        dm.digest(f, str(tmp_path / "memo"), hasher=interrupted)
    assert not (tmp_path / "memo" / dm.MEMO_NAME).exists()


def test_a_hit_skips_hashing_and_names_the_cached_time(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    h1, c1 = dm.digest(f, memo)
    assert c1 is None and h1 == dm.sha256_file(f)
    calls = []
    h2, c2 = dm.digest(f, memo, hasher=lambda p: calls.append(p) or "0" * 64)
    assert calls == [] and h2 == h1 and c2 is not None
    assert dm.word("pinned", c2) == "pinned (cached digest %s)" % c2
    assert dm.word("unknown", None) == "unknown"


def test_a_changed_stat_key_is_hashed_again(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    dm.digest(f, memo)
    _write(tmp_path / "w.bin", b"y" * 101)
    h, c = dm.digest(f, memo)
    assert c is None and h == dm.sha256_file(f)


def test_refresh_hashes_afresh_and_rewrites_the_entry(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    dm.digest(f, memo, hasher=lambda p: "1" * 64)  # a stale entry
    calls = []
    h, c = dm.digest(f, memo, refresh=True, hasher=lambda p: calls.append(p) or dm.sha256_file(p))
    assert calls == [os.path.realpath(f)] and c is None and h == dm.sha256_file(f)
    table = json.load(open(os.path.join(memo, dm.MEMO_NAME)))
    assert table[dm.stat_key(f)]["sha256"] == h


def test_match_is_by_digest_not_by_stat(tmp_path):
    a = _write(tmp_path / "a.bin", b"x" * 100)
    b = _write(tmp_path / "b.bin", b"z" * 100)  # same size, other bytes
    memo = str(tmp_path / "memo")
    ha, _ = dm.digest(a, memo)
    hb, cb = dm.digest(b, memo)
    assert ha != hb and cb is None


def test_a_corrupt_memo_file_is_ignored_not_trusted(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = tmp_path / "memo"
    memo.mkdir()
    (memo / dm.MEMO_NAME).write_text("{not json")
    h, c = dm.digest(f, str(memo))
    assert c is None and h == dm.sha256_file(f)


class TestTrunkLeverTallyFields(unittest.TestCase):
    """pairtrack.tally_fields spells the trunk levers' EXIT fields from their census: the core's TriMul census books fallbacks per reason
    ({reason: n}); the line carries the count and, when non-empty, ``v4trimul_<dir>_fallback_by=<reason:n,…>``."""

    def test_v4trimul_fields_carry_counts_not_the_census_dict(self):
        from unittest import mock
        census = {"v4trimul": {"out": {"served": 624, "fallback": {}}, "in": {"served": 600, "fallback": {"small_n": 20, "dtype": 4}}},
                  "templ_empty": {"served": 12, "fallback": 0, "fallback_by": {}}}
        with mock.patch.object(pairtrack, "census", return_value=census):
            f = pairtrack.tally_fields()
        self.assertEqual(f, ["v4trimul_out_served=624 v4trimul_out_fallback=0", "v4trimul_in_served=600 v4trimul_in_fallback=24 v4trimul_in_fallback_by=dtype:4,small_n:20",
                             "templ_empty_served=12 templ_empty_fallback=0"])

class TestNoCompileOptOut(unittest.TestCase):
    """`--no-compile` is MODEL_OPT_LEVERS_OFF=compiled (one mechanism) and the ACTIVE line says compile=on:aoti|on:dynamo|off:user (chai1_opt 0.4.x, QoL rule; 0.4.21: the on word names the route)."""
    def setUp(self):
        self._old = os.environ.pop(modes.ENV_LEVERS_OFF, None); modes.apply_levers_off()
    def tearDown(self):
        os.environ.pop(modes.ENV_LEVERS_OFF, None)
        if self._old is not None: os.environ[modes.ENV_LEVERS_OFF] = self._old
        modes.apply_levers_off()
    def test_flag_is_the_env_word(self):
        env = {}
        self.assertEqual(modes.no_compile_to_env(env), "compiled"); self.assertEqual(modes.no_compile_to_env(env), "compiled")      # idempotent
        env = {modes.ENV_LEVERS_OFF: "dit_attn"}; self.assertEqual(modes.no_compile_to_env(env), "dit_attn,compiled")
        from chai1_opt import cli as _cli
        a = _cli._argparser("pred").parse_args(["--mode", "fast", "--no-compile", "--input", "x.fasta", "--out_dir", "o"]); self.assertTrue(a.no_compile)
    def test_compile_word_probes_the_packages_layout_meta(self):
        """ACTIVE compile=: packages present WITH recorded input layouts -> on:aoti; present but built before layouts were recorded -> the loader
        refuses each by name at its crop, so the word says on:dynamo:no_layout_meta up front (cc 8.0: the card's aside word, unchanged); none -> on:dynamo."""
        import json as _json, os as _os, tempfile as _tempfile
        from chai1_opt import jit
        rep = {"mode": "fast", "levers_applied": ["tier1", "hoist2", "compiled"]}
        with _tempfile.TemporaryDirectory() as root:
            env = {"MODEL_OPT_JIT_ROOT": root, jit.ENV_KEY: "torchX-cuY-h100"}
            d = _os.path.join(root, "torchX-cuY-h100", "aoti"); _os.makedirs(d)
            self.assertEqual(modes.compile_word(rep, environ=env), "on:dynamo")                                   # no package
            name = "dstep_c512_s5_hoist2+dit_attn_0123456789ab"
            open(_os.path.join(d, name + ".pt2"), "w").close()
            open(_os.path.join(d, name + ".meta.json"), "w").write(_json.dumps({"name": name, "kinds": "iic"}))    # built before layouts were recorded
            self.assertEqual(modes.aoti_packages_layout_state(env), "no_layout_meta")
            self.assertEqual(modes.compile_word(rep, environ=env), "off:no_layout_meta")                          # the eager hoisted step serves (no per-process build)
            a100 = dict(env, MODEL_OPT_TARGET_GPU="A100-SXM4-80GB")
            self.assertEqual(modes.compile_word(rep, environ=a100), "off:no_layout_meta")                         # both cards
            name2 = "dstep_c768_s5_hoist2+dit_attn_0123456789ab"
            open(_os.path.join(d, name2 + ".pt2"), "w").close()
            open(_os.path.join(d, name2 + ".meta.json"), "w").write(_json.dumps({"name": name2, "kinds": "iic", "inputs": ["arg:noise_sigma"], "input_strides": [[5, 1]], "input_dtypes": ["torch.float32"]}))
            self.assertEqual(modes.aoti_packages_layout_state(env), "meta")
            self.assertEqual(modes.compile_word(rep, environ=env), "on:aoti")

    def test_compile_token(self):
        self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["tier1", "hoist2", "compiled"]}, environ={}), "on:dynamo")   # no JIT root: nothing built ahead
        with tempfile.TemporaryDirectory() as root:                                                                                             # a package for this stack key under the root → on:aoti
            env = {"MODEL_OPT_JIT_ROOT": root, "MODEL_OPT_STACK_KEY": "torchX-cuY-gpu"}
            self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["compiled"]}, environ=env), "on:dynamo")
            os.makedirs(os.path.join(root, "torchX-cuY-gpu", "aoti")); open(os.path.join(root, "torchX-cuY-gpu", "aoti", "dstep_c512_s5_hoist2+dit_attn_0123456789ab.pt2"), "w").close()
            with open(os.path.join(root, "torchX-cuY-gpu", "aoti", "dstep_c512_s5_hoist2+dit_attn_0123456789ab.meta.json"), "w") as fh:     # layouts recorded (chai1_fastln >= 0.3.8 packages): served -> on:aoti
                fh.write('{"name": "dstep_c512_s5_hoist2+dit_attn_0123456789ab", "kinds": "iic", "inputs": ["arg:noise_sigma"], "input_strides": [[5, 1]], "input_dtypes": ["torch.float32"]}')
            self.assertEqual(modes.aoti_packages_present(env), 1)
            self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["compiled"]}, environ=env), "on:aoti")
            # cc 8.0 (configs/a100.env's MODEL_OPT_TARGET_GPU word): the compile steps aside by card UNLESS packages serve (chai1_opt 0.4.22, A100 QoL rule)
            a100 = {"MODEL_OPT_TARGET_GPU": "A100"}
            self.assertEqual(modes.compile_card_aside_word(a100), "no_aoti_packages_cc80"); self.assertIsNone(modes.compile_card_aside_word({"MODEL_OPT_TARGET_GPU": "H100"}))
            self.assertIsNone(modes.compile_card_aside_word({}))
            self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["compiled"]}, environ=a100), "off:card:no_aoti_packages_cc80")
            self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["compiled"]}, environ=dict(env, MODEL_OPT_TARGET_GPU="A100", MODEL_OPT_STACK_KEY="torchX-cuY-gpu")), "on:aoti")
            self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["hoist2"], "levers_off": ["compiled"]}, environ=a100), "off:user")   # the user's switch keeps its word
        self.assertEqual(modes.compile_word({"mode": "fast", "levers_applied": ["tier1", "hoist2"], "levers_off": ["compiled"]}), "off:user")
        self.assertIsNone(modes.compile_word({"mode": "exact", "levers_applied": ["tier1", "hoist2"]}))                           # exact never compiles: no token
        line = report.activation_line({"active": True, "mode": "fast", "levels": "W1,W2,W5", "eager": "tier1", "dstep": "hoist2,dit_attn", "levers_applied": ["W1", "tier1", "hoist2", "dit_attn"], "levers_off": ["compiled"], "partial": True, "levers_off_env": True, "det": 0, "route": "driver"})
        self.assertIn(" compile=off:user det=0 ", line); self.assertIn(" PARTIAL off=compiled (MODEL_OPT_LEVERS_OFF)", line)
        os.environ[modes.ENV_LEVERS_OFF] = "compiled"; modes.apply_levers_off()
        self.assertNotIn("compiled", modes.kit_mode("fast").lever_names); self.assertIn("compiled", modes.levers_off_for("fast"))

