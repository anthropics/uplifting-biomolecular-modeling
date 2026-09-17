"""TRIMUL_PALLAS (trimul_pallas.py over the shared core's provider face opt_core.kernels.pallas.serve, asked by TIER word): the adapter's words, the
class rebinding and marker, the served call (stock parameter scopes → the face's math layout, the equation, the unit, the tier word per mode, the
resolved selection handed back to the face), the named fallbacks incl. the stock-statement-by-name class, the floor's refusal by name, the
provider's row switch, the ROWPAIR step-aside at --n_gpu > 1, the registry row / LEVER grammar, the ablation name, the mode tuples. Class
contracts only — which row a tier word names is the provider's measured table, never asserted here. CPU only: the face is the stub of
tests/_stubs.py (STUB_TRIMUL); the last class checks the tier words against the REAL provider table when opt_core is importable."""
from __future__ import annotations

import importlib.util
import os
import types
import unittest
from unittest import mock

from colabfold_opt import ablation, modes, registry, report, stack, trimul_pallas
from colabfold_opt.tests import _stubs


class TestProviderFace(unittest.TestCase):
    def test_the_adapter_names_the_provider_face_and_no_row(self):
        """the lever names the shared core's call FACE and asks it a tier word; it carries no row name, no cell table and no launch table."""
        self.assertEqual(trimul_pallas.IMPL, "opt_core.kernels.pallas.serve:triangle_multiplication")
        self.assertEqual((trimul_pallas.PROVIDER, trimul_pallas.OP, trimul_pallas.FORM, trimul_pallas.FACE, trimul_pallas.DIRECTION), ("opt_core.kernels.pallas", "trimul", "af2", "triangle_multiplication", "fwd"))
        self.assertIn(trimul_pallas.SERVE, ("opt_core.kernels.pallas.serve", _stubs.STUB_TRIMUL))
        self.assertEqual((trimul_pallas.EQ_OUT, trimul_pallas.EQ_IN), ("ikc,jkc->ijc", "kjc,kic->ijc"))
        for gone in ("ROW", "KERNEL", "KERNEL_NAME", "CFG_BY_CC", "cfg_for", "cfg_label", "kernel_word"):        # 0.2.13-0.2.18's row word and kit-side launch words left with the row binding
            self.assertFalse(hasattr(trimul_pallas, gone), gone)
        self.assertEqual([trimul_pallas.word_for(m) for m in ("fast", "big", "exact", None, "nonsense")], ["fast", "big", "exact", "fast", "fast"])
        self.assertEqual((trimul_pallas.class_word(128, trimul_pallas.EQ_OUT), trimul_pallas.class_word(64, trimul_pallas.EQ_IN)), ("C128E0", "C64E1"))
        self.assertIn(trimul_pallas.STOCK_BY_NAME, registry.STEP_ASIDE_RULES)                  # a class kept on the stock body by the provider's word is by design, not a partial activation


class TestLever(unittest.TestCase):
    def setUp(self):
        self.mods, self.saved = _stubs.install([])
        self.modules = self.mods["alphafold.model.modules"]
        self.face = self.mods[_stubs.STUB_TRIMUL]
        self.face.CALLS.clear(); self.face.RESOLVE_CALLS.clear(); self.face.REQUIRE_CALLS.clear(); self.face.COUNTS.clear()
        self.face.CC = "9.0"; self.face.ROW = "cd_trimul"; self.face.ROW_BY_CLASS.clear(); self.face.REFUSE.clear()

    def tearDown(self):
        _stubs.remove(self.saved)

    def _act(self, n, c, dtype="bfloat16"):
        return types.SimpleNamespace(shape=(n, n, c), ndim=3, dtype=types.SimpleNamespace(name=dtype), astype=lambda d: ("cast", d))

    def _mask(self, n, m=None):
        return types.SimpleNamespace(shape=(n, n if m is None else m), ndim=2, astype=lambda d: ("mask", d))

    def _cfg(self, **kw):
        d = dict(fuse_projection_weights=True, num_intermediate_channel=128, equation="ikc,jkc->ijc"); d.update(kw)
        return types.SimpleNamespace(**d)

    def test_enable_rebinds_the_class_with_the_marker_and_the_modes_word(self):
        stock_cls = self.modules.TriangleMultiplication
        st = trimul_pallas.enable(mode="fast")
        self.assertTrue(st["enabled"]); self.assertEqual(self.face.REQUIRE_CALLS, ["9.0"])       # the floor is asked first
        self.assertEqual((st["cc"], st["word"], st["rows"], st["cells"]), ("9.0", "fast", "none", "none"))
        cls = self.modules.TriangleMultiplication
        self.assertIsNot(cls, stock_cls); self.assertTrue(issubclass(cls, stock_cls)); self.assertTrue(getattr(cls, trimul_pallas.MARKER))
        self.assertEqual(cls.__name__, "TriangleMultiplication"); self.assertTrue(trimul_pallas.marker_present(self.modules))
        self.assertTrue(registry.marker_present(modes.TRIMUL_LEVER))
        trimul_pallas.enable(mode="fast")                                    # idempotent: no double subclassing
        self.assertIs(self.modules.TriangleMultiplication, cls)
        trimul_pallas.disable()
        self.assertIs(self.modules.TriangleMultiplication, stock_cls); self.assertFalse(trimul_pallas._STATE["enabled"])
        for mode_, word in (("big", "big"), ("exact", "exact"), (None, "fast")):   # the tier word is the mode's (big asks big, literally); an explicit word= overrides
            trimul_pallas.reset_for_tests()
            self.assertEqual(trimul_pallas.enable(mode=mode_)["word"], word)
        trimul_pallas.reset_for_tests()
        self.assertEqual(trimul_pallas.enable(mode="fast", word="big")["word"], "big")

    @unittest.skipIf(_stubs._real_haiku(), "the served path reads parameters through hk.get_parameter: outside hk.transform only the stand-in haiku answers (the GPU box covers the real one)")
    def test_served_call_hands_the_face_the_math_layout_the_word_and_the_selection(self):
        trimul_pallas.enable(mode="fast")
        cls = self.modules.TriangleMultiplication
        out_mod = cls(self._cfg(), None, name="triangle_multiplication_outgoing")
        act = self._act(400, 128)
        out = out_mod(act, self._mask(400), is_training=False)
        self.assertEqual(out, ("pallas", "ikc,jkc->ijc", act.dtype))         # the face's result, cast back to the activation dtype
        call = self.face.CALLS[-1]
        self.assertEqual((call["shape"], call["equation"], call["form"], call["unit"], call["word"], call["direction"], call["mask_shape"]),
                         ((400, 400, 128), "ikc,jkc->ijc", "af2", "pair", "fast", "fwd", (400, 400)))   # the model's own N (no pad), the pair unit at c_z 128, the mode's word
        self.assertEqual(sorted(call["params"]), sorted(("ln_in_scale", "ln_in_offset", "left_w", "right_w", "left_gate_w", "right_gate_w", "ln_c_scale", "ln_c_offset", "out_w", "gate_w",
                                                         "left_b", "right_b", "left_gate_b", "right_gate_b", "out_b", "gate_b")))   # serve.TRIMUL_KEYS + TRIMUL_BIAS_KEYS
        res = self.face.RESOLVE_CALLS[-1]
        self.assertEqual((res["op"], res["fam"], res["dtype"], res["n"], res["word"], res["cc"], res["direction"]), ("trimul", "af2_pair_c128_ch128_outgoing", "bf16", 400, "fast", "9.0", "fwd"))
        in_mod = cls(self._cfg(equation="kjc,kic->ijc", num_intermediate_channel=64), None, name="t")
        in_mod(self._act(400, 64), self._mask(400), is_training=False)      # the template stack's c=64 incoming call: unit tmpl, its own family word
        self.assertEqual((self.face.CALLS[-1]["unit"], self.face.RESOLVE_CALLS[-1]["fam"]), ("tmpl", "af2_tmpl_c64_ch64_incoming"))
        out_mod(act, self._mask(400), is_training=False)                    # the same class again: the provider is asked ONCE per (cc, dtype, C_z, C, equation, N)
        self.assertEqual(len(self.face.RESOLVE_CALLS), 2)
        st = trimul_pallas._STATE
        self.assertEqual(st["shapes"], "N400xC128xE0:2,N400xC64xE1:1")       # blank-free and sorted
        self.assertRegex(st["rows"], r"^C128E0:\S+:2,C64E1:\S+:1$")          # <class>:<arm the face served>:<count> — the arm is the provider's answer, not asserted
        self.assertEqual(st["cells"], "C128E0:N<=400,C64E1:N<=400")          # the deciding cell's N bucket per class
        self.assertEqual((st["calls"], st["fallbacks"], st["precision"], st["word"]), (3, 0, "bf16", "fast"))

    @unittest.skipIf(_stubs._real_haiku(), "see test_served_call_hands_the_face_the_math_layout_the_word_and_the_selection")
    def test_a_class_whose_word_names_the_stock_statement_keeps_the_stock_body_by_name(self):
        """no measured cell for the class on this card / jax line (or the stock op won the cell): the provider's tier word names its stock statement — the STOCK body runs
        here and the class is counted `stock_by_name` (a registry step-aside rule), its rows= token says `stock`, its cells= token `none`."""
        self.face.ROW_BY_CLASS["C64E1"] = "xla"
        trimul_pallas.enable(mode="fast")
        cls = self.modules.TriangleMultiplication
        self.assertEqual(cls(self._cfg(equation="kjc,kic->ijc", num_intermediate_channel=64), None, name="t")(self._act(256, 64), self._mask(256), is_training=False)[0], "stock")
        self.assertEqual(cls(self._cfg(), None, name="t")(self._act(256, 128), self._mask(256), is_training=False)[0], "pallas")
        st = trimul_pallas._STATE
        self.assertEqual((st["calls"], st["fallbacks"], st["fallback_by"]), (1, 1, "stock_by_name:1"))
        self.assertRegex(st["rows"], r"^C128E0:\S+:1,C64E1:stock:1$"); self.assertEqual(st["cells"], "C128E0:N<=400,C64E1:none")
        self.assertEqual([c["equation"] for c in self.face.CALLS], ["ikc,jkc->ijc"])       # the face was called for the served class only

    @unittest.skipIf(_stubs._real_haiku(), "see test_served_call_hands_the_face_the_math_layout_the_word_and_the_selection")
    def test_a_provider_refusal_is_a_fallback_by_its_name(self):
        self.face.REFUSE["C128E1"] = "levers_off"                            # e.g. MODEL_OPT_LEVERS_OFF=pallas:<row> with no other arm in the cell: refused by name
        trimul_pallas.enable(mode="fast")
        cls = self.modules.TriangleMultiplication
        self.assertEqual(cls(self._cfg(equation="kjc,kic->ijc"), None, name="t")(self._act(256, 128), self._mask(256), is_training=False)[0], "stock")
        self.assertEqual((trimul_pallas._STATE["fallbacks"], trimul_pallas._STATE["fallback_by"], self.face.CALLS), (1, "levers_off:1", []))

    def test_cc_8_0_and_an_untested_8x_part_engage(self):
        for cc in ("8.0", "8.6"):
            trimul_pallas.reset_for_tests(); self.face.CC = cc
            st = trimul_pallas.enable(mode="fast")
            self.assertTrue(st["enabled"]); self.assertEqual(st["cc"], cc)

    def test_fallbacks_run_the_stock_body_and_are_named(self):
        trimul_pallas.enable(mode="fast")
        cls = self.modules.TriangleMultiplication
        cases = [(self._cfg(fuse_projection_weights=False), self._act(256, 128), self._mask(256)),          # monomer-layout config: unfused_layout
                 (self._cfg(num_intermediate_channel=48), self._act(256, 128), self._mask(256)),             # channels not a power of two
                 (self._cfg(), self._act(256, 128, "float16"), self._mask(256)),                             # dtype outside bf16 / f32
                 (self._cfg(equation="ikc,kjc->ijc"), self._act(256, 128), self._mask(256)),                 # an equation the face does not serve
                 (self._cfg(), self._act(256, 128), self._mask(256, 255)),                                   # mask shape != [N, N]
                 (self._cfg(), types.SimpleNamespace(shape=(2, 256, 256, 128), ndim=4, dtype=types.SimpleNamespace(name="bfloat16")), self._mask(256))]   # rank / square: shape
        for cfg, act, mask in cases:
            self.assertEqual(cls(cfg, None, name="t")(act, mask, is_training=False)[0], "stock")
        self.assertEqual(trimul_pallas._STATE["fallback_by"], "channels:1,dtype:1,equation:1,mask_shape:1,shape:1,unfused_layout:1")   # blank-free, sorted
        self.assertEqual((trimul_pallas._STATE["calls"], trimul_pallas._STATE["fallbacks"]), (0, 6)); self.assertEqual(self.face.CALLS, []); self.assertEqual(self.face.RESOLVE_CALLS, [])

    def test_a_part_below_cc_8_0_refuses_by_name(self):
        self.face.CC = "7.5"
        with self.assertRaises(trimul_pallas.Refusal) as cm:
            trimul_pallas.enable(mode="fast")
        self.assertEqual(cm.exception.kind, "cc_below_8_0"); self.assertIn("TRIMUL_PALLAS: cc_below_8_0", str(cm.exception))
        self.assertFalse(trimul_pallas._STATE["enabled"]); self.assertFalse(trimul_pallas.marker_present(self.modules))   # nothing rebound: the mode refuses, never a stock-bodied fast

    def test_real_require_names_its_refusal_kinds(self):
        src = open(trimul_pallas.__file__, encoding="utf-8").read()
        for kind in ("no_pallas_triton", "no_compiler_params", "not_gpu_backend", "cc_below_8_0", "no_provider_face", "unknown_word"):
            self.assertIn(f'Refusal("{kind}"', src)

    def test_registry_row_and_lever_line_grammar(self):
        lv = registry.LEVERS[modes.TRIMUL_LEVER]
        self.assertEqual((lv.impl, lv.origin, lv.strategy, lv.kit_file, lv.cls), ("opt_core.kernels.pallas.serve:triangle_multiplication", "core", "F2.trimul", "colabfold_opt/trimul_pallas.py", "forward"))
        self.assertTrue(lv.in_mode); self.assertFalse(lv.deployment); self.assertEqual(tuple(lv.probe), ("module_state", modes.TRIMUL_LEVER_MODULE, "enabled"))
        self.assertEqual(registry.MARKERS[modes.TRIMUL_LEVER], ("class_attr", "alphafold.model.modules", "TriangleMultiplication", "_trimul_pallas"))
        self.assertEqual(trimul_pallas.STRATEGY, lv.strategy); self.assertEqual(trimul_pallas.IMPL, lv.impl)
        trimul_pallas.enable(mode="fast")
        ev = {k: v for k, v in trimul_pallas._STATE.items() if k != "enabled"}
        line = report.lever_line(modes.TRIMUL_LEVER, "on", None, **ev)
        self.assertRegex(line, r"^\[colabfold-opt\] LEVER name=TRIMUL_PALLAS state=on impl=opt_core\.kernels\.pallas\.serve:triangle_multiplication origin=core strategy=F2\.trimul "
                               r"calls=0 fallbacks=0 fallback_by=none word=fast rows=none cells=none shapes=none precision=none cc=9\.0 core=\S+ jax=\S+$")
        self.assertNotIn("  ", line); self.assertEqual(len(line.split(" impl=")), 2)

    def test_big_p_gt_1_drops_the_pair_levers_by_name(self):
        gates = _stubs.gates_pass(stack)
        try:
            stack.reset_for_tests()
            rep = stack.check("big", print_line=False, n_gpu=2)
            named = [l for l in modes.PAIR_MUL_LEVERS if l in modes.TABLE["big"][0]]
            self.assertTrue(named)                                           # big names a one-device pair lever …
            for lever in named:                                              # … and the row-sharded pair stack owns the class at P > 1: dropped by name
                self.assertNotIn(lever, rep["levers"]); self.assertEqual(rep["levers_dropped"][lever], trimul_pallas.N_GPU_REASON)
            self.assertEqual(trimul_pallas.N_GPU_REASON, "n_gpu>1")
            rep1 = stack.check("fast", print_line=False)
            self.assertEqual(rep1["levers_dropped"], {})
            for lever in named:
                self.assertIn(lever, rep1["levers"])
        finally:
            _stubs.gates_restore(stack, gates)
            stack.reset_for_tests()

    def test_modes_name_one_pair_lever_and_both_switches_reach_it(self):
        for mode in ("fast", "big"):
            named = [l for l in modes.PAIR_MUL_LEVERS if l in modes.TABLE[mode][0]]
            self.assertEqual(named, [modes.TRIMUL_LEVER], mode)              # the shipped tables: TRIMUL_PALLAS is THE triangle-multiplication lever of fast / big
        self.assertNotIn(modes.TRIMUL_LEVER, modes.TABLE["exact"][0]); self.assertEqual(modes.TABLE["off"][0], ())   # exact: the provider's exact tier names the stock op for every triangle-multiplication cell — the lever is not in the table
        fast = modes.resolve("fast")["levers"]
        ablation.validate("fast", [modes.TRIMUL_LEVER], fast, 1)            # MODEL_OPT_LEVERS_OFF=TRIMUL_PALLAS: a lever of the mode, accepted …
        res = ablation.apply(modes.resolve("fast"), [modes.TRIMUL_LEVER])
        self.assertNotIn(modes.TRIMUL_LEVER, res["levers"])
        with self.assertRaises(ablation.AblationError):                    # … a name outside the mode is refused by name
            ablation.validate("exact", [modes.TRIMUL_LEVER], modes.resolve("exact")["levers"], 1)
        self.assertEqual(ablation.requested({ablation.ENV: "TRIMUL_PALLAS,pallas:cd_trimul"})[0], "TRIMUL_PALLAS")   # the provider's row switch rides the same variable; the kit reads its own names


@unittest.skipUnless(importlib.util.find_spec("opt_core") is not None and importlib.util.find_spec("opt_core.kernels.pallas") is not None, "needs the shared core on the path")
class TestTierWordsAgainstTheRealProvider(unittest.TestCase):
    """the class contract with the REAL table (pure Python, no GPU): for the kit's four call classes on the jax 0.5 line and both measured cards, every tier word resolves
    to SOME arm of the triangle-multiplication family ending in the stock statement — which arm is the provider's measurement, never asserted; the provider's row switch
    removes a row by name."""

    CLASSES = ((128, 128, "ikc,jkc->ijc"), (128, 128, "kjc,kic->ijc"), (64, 64, "ikc,jkc->ijc"), (64, 64, "kjc,kic->ijc"))

    def test_every_tier_word_resolves_for_the_kits_classes_on_both_cards(self):
        P = importlib.import_module("opt_core.kernels.pallas")
        rows = set(P.ROW_NAMES) | set(P.STOCK_ROWS)
        for cc in ("9.0", "8.0"):
            for cz, c, eq in self.CLASSES:
                fam = trimul_pallas._family(P, cz, c, eq)
                self.assertIn(fam.rsplit("_", 1)[1], ("outgoing", "incoming")); self.assertTrue(fam.startswith(f"af2_{trimul_pallas._unit(cz)}_c{cz}_ch{c}_"), fam)
                for word in ("fast", "big", "exact"):
                    for n in (200, 512, 1200, 3000):
                        sel = P.select("0.5.3", cc, "bf16", "trimul", fam, n, "fwd", word=word)
                        self.assertIn(P.parse_arm(sel.arm)[0], rows, (cc, fam, word, n, sel)); self.assertEqual(sel.candidates[-1], "xla")
                        if word == "exact" and sel.cell_key:                 # the exact rule: an arm vouched bitwise, else the stock statement by name — what the exact table relies on
                            self.assertTrue(sel.cls in ("exact", "stock") or sel.arm == "xla", sel)

    def test_the_providers_row_switch_removes_a_row_by_name(self):
        P = importlib.import_module("opt_core.kernels.pallas")
        fam = trimul_pallas._family(P, 128, 128, trimul_pallas.EQ_OUT)
        base = P.select("0.5.3", "9.0", "bf16", "trimul", fam, 512, "fwd", word="fast")
        if base.arm == "xla":
            self.skipTest("no measured row ahead of the stock statement in this table")
        with mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "pallas:" + P.parse_arm(base.arm)[0]}):
            off = P.select("0.5.3", "9.0", "bf16", "trimul", fam, 512, "fwd", word="fast")
        self.assertNotEqual(P.parse_arm(off.arm)[0], P.parse_arm(base.arm)[0]); self.assertNotIn(base.arm, off.candidates)
        with mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "pallas"}):   # every non-stock row off: the stock statement by name
            self.assertEqual(P.select("0.5.3", "9.0", "bf16", "trimul", fam, 512, "fwd", word="fast").arm, "xla")


if __name__ == "__main__":
    unittest.main()
