"""TRANSITION (transition.py: AlphaFold's Transition module on the shared core's provider FACE opt_core.kernels.pallas.serve.transition by the mode's
TIER WORD): the words, the class rebinding and marker, the served call (stock parameter scopes -> the face's parameter tree, the tier word per mode,
n_tokens, strict), the stock statement by the cell's word (cell_stock / no_cell: stock's body, by design), the named fallbacks, a row refusing at
the call, the provider's `pallas:<row>` word, the big step-aside at --n_gpu > 1, the registry row / LEVER grammar, the ablation names, the mode
tuples, and the wrapper's parameter mapping against the face's reference statement on CPU (numpy stand-ins; jax when importable). CPU only: the
faces are the stub of tests/_stubs.py (STUB_PSERVE)."""
from __future__ import annotations

import os
import types
import unittest

from colabfold_opt import ablation, modes, registry, report, stack, transition
from colabfold_opt.tests import _stubs


class TestWords(unittest.TestCase):
    def test_the_lever_names_the_provider_face_and_the_tier_words(self):
        self.assertEqual((transition.NAME, transition.CLASS, transition.MODULES, transition.MARKER), ("TRANSITION", "Transition", "alphafold.model.modules", "_transition_served"))
        self.assertEqual(transition.IMPL, "opt_core.kernels.pallas.serve:transition")
        self.assertEqual((transition.OP, transition.FORM, transition.ACTIVATION, transition.DIRECTION, transition.STOCK_ROW), ("transition", "af2", "relu", "fwd", "xla"))
        self.assertEqual(transition.WORDS, {"fast": "fast", "big": "big"})                 # big asks the provider's big word, not fast's
        self.assertEqual(transition.word_for("fast"), "fast"); self.assertEqual(transition.word_for("big"), "big")
        for m in ("exact", "off", None, "nosuch"):                                            # exact keeps the stock module by name: no word
            with self.assertRaises(transition.Refusal) as cm:
                transition.word_for(m)
            self.assertEqual(cm.exception.kind, "no_word")
        self.assertEqual(transition.PARAM_SCOPES, ("input_layer_norm", "transition1", "transition2"))
        self.assertIn(transition.CELL_STOCK, registry.STEP_ASIDE_RULES); self.assertIn(transition.NO_CELL, registry.STEP_ASIDE_RULES)   # the stock statement by the cell's word is by design, never a partial activation
        self.assertEqual(transition.site_word((400, 400, 128), 4), "pair_c128x4"); self.assertEqual(transition.site_word((508, 400, 256), 4), "msa_c256x4")
        self.assertEqual(transition.site_word((1152, 400, 64), 4), "msa_c64x4"); self.assertEqual(transition.site_word((400, 400, 64), 2), "pair_c64x2")

    def test_real_require_names_its_refusal_kinds(self):
        src = open(transition.__file__, encoding="utf-8").read()
        for kind in ("no_jax", "not_gpu_backend", "core_face_missing"):
            self.assertIn(f'Refusal("{kind}"', src)
        self.assertNotIn("cc_below", src)                                                    # a part the rows do not serve is the provider's word (the stock statement), never this lever's refusal


class TestLever(unittest.TestCase):
    def setUp(self):
        self.mods, self.saved = _stubs.install([])
        self.modules = self.mods["alphafold.model.modules"]
        self.serve = self.mods[_stubs.STUB_PSERVE]
        self.serve.CALLS.clear(); self.serve.RESOLVE_CALLS.clear(); self.serve.REQUIRE_CALLS.clear(); self.serve.REFUSE_AT_CALL.clear(); self.serve.CC = "9.0"
        self._env = os.environ.pop("MODEL_OPT_LEVERS_OFF", None)

    def tearDown(self):
        os.environ.pop("MODEL_OPT_LEVERS_OFF", None)
        if self._env is not None:
            os.environ["MODEL_OPT_LEVERS_OFF"] = self._env
        _stubs.remove(self.saved)

    def _act(self, lead, n, c, dtype="bfloat16"):
        return types.SimpleNamespace(shape=(lead, n, c), ndim=3, dtype=types.SimpleNamespace(name=dtype), astype=lambda d: ("cast", d))

    def _mask(self, lead, n):
        return types.SimpleNamespace(shape=(lead, n), ndim=2)

    def _cfg(self, factor=4):
        return types.SimpleNamespace(num_intermediate_factor=factor)

    def test_enable_rebinds_the_class_with_the_marker_and_takes_the_mode_word(self):
        stock_cls = self.modules.Transition
        st = transition.enable("fast")
        self.assertTrue(st["enabled"]); self.assertEqual(self.serve.REQUIRE_CALLS, ["9.0"])   # the floor is asked first
        self.assertEqual((st["word"], st["cc"], st["core"], st["jax"]), ("fast", "9.0", "stub", "0.5.3"))
        cls = self.modules.Transition
        self.assertIsNot(cls, stock_cls); self.assertTrue(issubclass(cls, stock_cls)); self.assertTrue(getattr(cls, transition.MARKER))
        self.assertEqual(cls.__name__, "Transition"); self.assertTrue(transition.marker_present(self.modules))
        self.assertTrue(registry.marker_present(modes.TRANSITION_LEVER))
        transition.enable("fast")                                                # idempotent: no double subclassing
        self.assertIs(self.modules.Transition, cls)
        transition.disable()
        self.assertIs(self.modules.Transition, stock_cls); self.assertFalse(transition._STATE["enabled"])
        st = transition.enable("big")                                          # big binds the provider's big word literally
        self.assertEqual(st["word"], "big")
        with self.assertRaises(transition.Refusal):
            transition.reset_for_tests(); transition.enable("exact")            # exact carries no word: refused by name if ever asked

    @unittest.skipIf(_stubs._real_haiku(), "the served path reads parameters through hk.get_parameter: outside hk.transform only the stand-in haiku answers (the GPU box covers the real one)")
    def test_served_call_reads_the_stock_scopes_and_calls_the_face_by_tier_word(self):
        transition.enable("fast")
        cls = self.modules.Transition
        act = self._act(400, 400, 128)
        out = cls(self._cfg(4), None, name="pair_transition")(act, self._mask(400, 400), is_training=False)
        self.assertEqual(out, ("served", "mlp_transition", act, act.dtype))     # the face's result, cast back to the activation dtype; the row the provider served
        q = self.serve.RESOLVE_CALLS[-1]
        self.assertEqual((q["op"], q["fam"], q["dtype"], q["n"], q["word"], q["direction"], q["cc"], q["jax"]), ("transition", "af2_relu_c128_x4", "bf16", 400, "fast", "fwd", "9.0", "0.5.3"))
        call = self.serve.CALLS[-1]
        self.assertEqual((call["row"], call["word"], call["shape"], call["n"], call["activation"], call["form"], call["strict"]), ("mlp_transition", "fast", (400, 400, 128), 400, "relu", "af2", True))
        self.assertEqual(call["params"], {"ln_scale": (128,), "ln_offset": (128,), "w1": (128, 512), "b1": (512,), "w2": (512, 128), "b2": (128,)})   # the face's parameter tree from the three stock scopes
        self.assertEqual(str(getattr(call["dtypes"]["ln_scale"], "__name__", call["dtypes"]["ln_scale"])), "float32")   # LayerNorm parameters in f32
        self.assertIs(call["dtypes"]["w1"], act.dtype); self.assertIs(call["dtypes"]["b2"], act.dtype)                 # Linear parameters at the activation dtype (stock's Linear)
        tm = self._act(400, 400, 64)
        self.assertEqual(cls(self._cfg(2), None, name="pair_transition")(tm, self._mask(400, 400), is_training=False)[1], "cd_transition")   # the template pair stack's c 64, factor 2 class: another row, same word
        self.assertEqual(self.serve.RESOLVE_CALLS[-1]["fam"], "af2_relu_c64_x2")
        cls(self._cfg(4), None, name="pair_transition")(self._act(400, 400, 128), self._mask(400, 400), is_training=False)   # the same class again: the provider is asked once per class
        self.assertEqual(len(self.serve.RESOLVE_CALLS), 2)
        st = transition._STATE
        self.assertEqual((st["calls"], st["served"], st["fallbacks"], st["fallback_by"], st["precision"], st["word"]), (3, 3, 0, "none", "bf16", "fast"))
        self.assertEqual(st["rows"], "pair_c128x4:mlp_transition,pair_c64x2:cd_transition")   # the served row per call class, blank-free and sorted
        self.assertEqual(st["shapes"], "pair_c128x4@N400:2,pair_c64x2@N400:1")
        self.assertEqual(st["cells"], "pair_c128x4:0.5|9.0|bf16|transition|af2_relu_c128_x4|N<=800|fwd,pair_c64x2:0.5|9.0|bf16|transition|af2_relu_c64_x2|N<=800|fwd")

    def test_the_stock_statement_by_the_cells_word_runs_stocks_body_by_design(self):
        transition.enable("fast")
        cls = self.modules.Transition
        msa = self._act(508, 400, 256)                                           # the MSA transition class: the cell names the stock statement fastest -> stock's own (chunked) body
        out = cls(self._cfg(4), None, name="msa_transition")(msa, self._mask(508, 400), is_training=False)
        self.assertEqual(out[0], "stock")
        xmsa = self._act(1152, 400, 64)                                          # the extra-MSA transition class (c 64, factor 4): no measured cell -> the stock statement by name
        self.assertEqual(cls(self._cfg(4), None, name="msa_transition")(xmsa, self._mask(1152, 400), is_training=False)[0], "stock")
        f32 = self._act(400, 400, 128, "float32")                                # f32 activations: the cells' word is the stock statement
        self.assertEqual(cls(self._cfg(4), None, name="pair_transition")(f32, self._mask(400, 400), is_training=False)[0], "stock")
        st = transition._STATE
        self.assertEqual((st["calls"], st["fallbacks"]), (0, 3)); self.assertEqual(self.serve.CALLS, [])
        self.assertEqual(st["fallback_by"], "cell_stock:2,no_cell:1")
        self.assertEqual(st["rows"], "msa_c256x4:xla,msa_c64x4:xla,pair_c128x4:xla"); self.assertEqual(st["cells"].split(",")[1], "msa_c64x4:none")
        from colabfold_opt import manifest
        self.assertEqual(manifest.stepped_aside_rule(dict(st)), "cell_stock,no_cell")   # every call stepped aside by the cells' word: by design, not a partial activation

    def test_fallbacks_run_the_stock_body_and_are_named(self):
        transition.enable("fast")
        cls = self.modules.Transition
        cases = [self._act(400, 400, 128, "float16"),                            # dtype outside bf16 / f32
                 types.SimpleNamespace(shape=(2, 400, 400, 128), ndim=4, dtype=types.SimpleNamespace(name="bfloat16"))]   # rank: shape
        for act in cases:
            self.assertEqual(cls(self._cfg(4), None, name="t")(act, self._mask(400, 400), is_training=False)[0], "stock")
        self.serve.REFUSE_AT_CALL["af2_relu_c128_x4"] = "not_served"             # the row refuses THIS call by kind: stock's body serves it, counted under the kind
        self.assertEqual(cls(self._cfg(4), None, name="t")(self._act(400, 400, 128), self._mask(400, 400), is_training=False)[0], "stock")
        self.assertEqual(transition._STATE["fallback_by"], "dtype:1,not_served:1,shape:1")   # blank-free, sorted
        self.assertEqual((transition._STATE["calls"], transition._STATE["fallbacks"]), (0, 3)); self.assertEqual(self.serve.CALLS, [])
        from colabfold_opt import manifest
        self.assertIsNone(manifest.stepped_aside_rule(dict(transition._STATE)))   # these are NOT by design: a lever with only such fallbacks is a partial activation

    @unittest.skipIf(_stubs._real_haiku(), "see test_served_call_reads_the_stock_scopes_and_calls_the_face_by_tier_word")
    def test_the_providers_row_word_steps_one_row_aside_inside_the_tier_word(self):
        os.environ["MODEL_OPT_LEVERS_OFF"] = "pallas:mlp_transition"             # the provider's own word (left in the variable for the core; the stub honours it the way the core does)
        transition.enable("fast")
        cls = self.modules.Transition
        self.assertEqual(cls(self._cfg(4), None, name="t")(self._act(400, 400, 128), self._mask(400, 400), is_training=False)[0], "stock")        # pair c128: mlp_transition aside -> the stock statement
        self.assertEqual(cls(self._cfg(2), None, name="t")(self._act(400, 400, 64), self._mask(400, 400), is_training=False)[1], "cd_transition")  # the template class's row is another one: still served
        self.assertEqual(transition._STATE["rows"], "pair_c128x4:xla,pair_c64x2:cd_transition")

    def test_registry_row_and_lever_line_grammar(self):
        lv = registry.LEVERS[modes.TRANSITION_LEVER]
        self.assertEqual((lv.impl, lv.origin, lv.strategy, lv.kit_file, lv.cls), ("opt_core.kernels.pallas.serve:transition", "core", "LOCAL.fused_transition", "colabfold_opt/transition.py", "forward"))
        self.assertTrue(lv.in_mode); self.assertFalse(lv.deployment); self.assertEqual(tuple(lv.probe), ("module_state", modes.TRANSITION_LEVER_MODULE, "enabled"))
        self.assertEqual(registry.MARKERS[modes.TRANSITION_LEVER], ("class_attr", "alphafold.model.modules", "Transition", "_transition_served"))
        self.assertEqual(transition.STRATEGY, lv.strategy)
        transition.enable("fast")
        ev = {k: v for k, v in transition._STATE.items() if k != "enabled"}
        line = report.lever_line(modes.TRANSITION_LEVER, "on", None, **ev)
        self.assertRegex(line, r"^\[colabfold-opt\] LEVER name=TRANSITION state=on impl=opt_core\.kernels\.pallas\.serve:transition origin=core strategy=LOCAL\.fused_transition "
                               r"calls=0 fallbacks=0 fallback_by=none word=fast rows=none served=0 cells=none shapes=none precision=none cc=9\.0 core=\S+ jax=\S+$")
        self.assertNotIn("  ", line); self.assertEqual(len(line.split(" impl=")), 2)
        self.assertEqual(stack.lever_states()[modes.TRANSITION_LEVER], transition._STATE)   # the exit census reads this module's state

    def test_big_p_gt_1_drops_the_lever_by_name_and_p_1_keeps_it(self):
        gates = _stubs.gates_pass(stack)
        try:
            stack.reset_for_tests()
            rep = stack.check("big", print_line=False, n_gpu=2)
            self.assertNotIn(modes.TRANSITION_LEVER, rep["levers"]); self.assertEqual(rep["levers_dropped"][modes.TRANSITION_LEVER], transition.N_GPU_REASON)
            self.assertEqual(transition.N_GPU_REASON, "n_gpu>1"); self.assertIn(modes.TRANSITION_LEVER, modes.ONE_DEVICE_PAIR_LEVERS)
            for mode in ("fast", "big"):
                rep1 = stack.check(mode, print_line=False)
                self.assertIn(modes.TRANSITION_LEVER, rep1["levers"]); self.assertNotIn(modes.TRANSITION_LEVER, rep1["levers_dropped"])
        finally:
            _stubs.gates_restore(stack, gates)
            stack.reset_for_tests()

    def test_activation_enables_the_lever_with_the_modes_word(self):
        import contextlib, io

        def _activate(mode):
            with contextlib.redirect_stderr(io.StringIO()):
                return stack.activate(mode, queries=_stubs.queries(300))
        def _fresh():                                                            # one activation per process image: re-install the stand-ins between modes
            stack.reset_for_tests(); _stubs.remove(self.saved)
            self.mods, self.saved = _stubs.install([]); self.modules = self.mods["alphafold.model.modules"]; self.serve = self.mods[_stubs.STUB_PSERVE]
        gates = _stubs.gates_pass(stack)
        try:
            for mode, word in (("fast", "fast"), ("big", "big")):
                _fresh()
                rep = _activate(mode)
                self.assertTrue(rep["active"], rep.get("reason")); self.assertIn(modes.TRANSITION_LEVER, rep["levers_applied"])
                self.assertEqual(transition._STATE["word"], word); self.assertTrue(transition.marker_present(self.modules))
            _fresh()
            rep = _activate("exact")
            self.assertTrue(rep["active"]); self.assertNotIn(modes.TRANSITION_LEVER, rep["levers_applied"]); self.assertFalse(transition.marker_present(self.modules))   # exact: the stock module by name
        finally:
            _stubs.gates_restore(stack, gates)
            stack.reset_for_tests(); transition.reset_for_tests()

    def test_modes_and_the_ablation_switch_know_the_lever(self):
        for mode in ("fast", "big"):
            self.assertIn(modes.TRANSITION_LEVER, modes.TABLE[mode][0], mode)
        self.assertNotIn(modes.TRANSITION_LEVER, modes.TABLE["exact"][0]); self.assertEqual(modes.TABLE["off"][0], ())
        fast = modes.resolve("fast")["levers"]
        self.assertEqual(ablation.validate("fast", [modes.TRANSITION_LEVER], fast, 1), [modes.TRANSITION_LEVER])   # a lever of the mode: accepted …
        res = ablation.apply(modes.resolve("fast"), [modes.TRANSITION_LEVER])
        self.assertNotIn(modes.TRANSITION_LEVER, res["levers"])
        with self.assertRaises(ablation.AblationError):                                                             # … a name outside the mode is refused by name
            ablation.validate("exact", [modes.TRANSITION_LEVER], modes.resolve("exact")["levers"], 1)
        self.assertIn(modes.TRANSITION_LEVER, ablation.PROVIDER_LEVERS); self.assertEqual(ablation.row_word("pallas:mlp_transition"), "|".join(ablation.PROVIDER_LEVERS))
        if ablation.provider_rows():                                                                                # the provider's own words need the shared core importable (its row table)
            self.assertIn("mlp_transition", ablation.provider_rows())
            self.assertEqual(ablation.validate("fast", ["pallas:mlp_transition"], fast, 1), ["pallas:mlp_transition"])
            self.assertEqual(ablation.validate("fast", ["pallas:cd_transition", modes.TRIMUL_LEVER], fast, 1), ["pallas:cd_transition", modes.TRIMUL_LEVER])   # one provider lever kept is enough
            for bad in (["pallas:nosuch_row"], ["pallas:"], ["pallas:mlp_transition", *[l for l in ablation.PROVIDER_LEVERS if l in fast]]):   # every provider-bound lever of the mode ablated: nothing left to switch a row of
                with self.assertRaises(ablation.AblationError, msg=bad):
                    ablation.validate("fast", bad, fast, 1)


class TestParameterMappingAgainstTheReference(unittest.TestCase):
    """The wrapper's parameter tree (ln_scale / ln_offset / w1 / b1 / w2 / b2 read from the stock scopes input_layer_norm / transition1 / transition2)
    fed to the face's reference statement equals AlphaFold's own Transition arithmetic (LayerNorm -> Linear -> ReLU -> Linear) on the same numbers."""

    def test_reference_transition_equals_the_stock_arithmetic(self):
        try:
            import numpy as np
            import jax  # noqa: F401
            import jax.numpy as jnp
            from opt_core.kernels.pallas import serve as S
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"jax + the shared core's face are needed for the numerics check ({type(e).__name__}: {e})")
        rng = np.random.default_rng(0)
        S_, N, C, f = 6, 10, 16, 4
        act = rng.standard_normal((S_, N, C)).astype(np.float32)
        p = {"input_layer_norm": {"scale": 1.0 + 0.1 * rng.standard_normal(C), "offset": 0.1 * rng.standard_normal(C)},
             "transition1": {"weights": rng.standard_normal((C, C * f)) / np.sqrt(C), "bias": 0.1 * rng.standard_normal(C * f)},
             "transition2": {"weights": rng.standard_normal((C * f, C)) / np.sqrt(C * f), "bias": 0.1 * rng.standard_normal(C)}}
        # AlphaFold's statement (modules.Transition with common_modules.LayerNorm eps 1e-5, in f32)
        x = act
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        h = (x - mu) / np.sqrt(var + 1e-5) * p["input_layer_norm"]["scale"] + p["input_layer_norm"]["offset"]
        h = np.maximum(h @ p["transition1"]["weights"] + p["transition1"]["bias"], 0.0)
        want = h @ p["transition2"]["weights"] + p["transition2"]["bias"]
        # the wrapper's mapping (transition._build: scope -> face key)
        params = {"ln_scale": jnp.asarray(p["input_layer_norm"]["scale"], jnp.float32), "ln_offset": jnp.asarray(p["input_layer_norm"]["offset"], jnp.float32),
                  "w1": jnp.asarray(p["transition1"]["weights"], jnp.float32), "b1": jnp.asarray(p["transition1"]["bias"], jnp.float32),
                  "w2": jnp.asarray(p["transition2"]["weights"], jnp.float32), "b2": jnp.asarray(p["transition2"]["bias"], jnp.float32)}
        got = np.asarray(S.reference_transition(jnp.asarray(act), params, activation="relu", precision="highest"))
        self.assertEqual(got.shape, want.shape)
        self.assertLess(float(np.abs(got - want).max()), 1e-3, float(np.abs(got - want).max()))


if __name__ == "__main__":
    unittest.main()
