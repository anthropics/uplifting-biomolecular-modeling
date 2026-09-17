"""0.8.0: the three kernel adapters bind opt_core's provider BY TIER WORD — class contracts (never a row winner by name)."""
import os
import unittest
from unittest import mock

from af2ig_opt import flash_attn, fused_trimul, modes, programs, stack, triattn


class TestTierWord(unittest.TestCase):
    def test_tier_word_and_class(self):
        self.assertEqual(modes.tier_word({}), "fast"); self.assertEqual(modes.tier_word({"AF2IG_OPT_TIER": "big"}), "big")
        self.assertEqual(modes.tier_word({"AF2IG_OPT_TIER": "exact"}), "fast")              # exact binds no kernel: a hand-run driver with a kernel flag asks the fast tier
        self.assertEqual(modes.F32_PRODUCTS, "tf32")                                        # the tolerance tiers' float32 product class, one word for every adapter
        self.assertIn(modes.ENV_TIER, stack.DECLARED_ENV)
        self.assertEqual(stack.LEVER_WORDS, ("AF2IG_OPT_TRIATTN_CORE_DTYPE",))              # the only kernel word a caller hands the child (L19); no length / row words
        for gone in ("BRIDGE_MIN_LEN", "ENV_BRIDGE_MIN_LEN", "ENV_ROW", "provider_for", "bridge_call", "PRECISION"):
            self.assertFalse(hasattr(triattn, gone), gone)
        self.assertFalse(hasattr(fused_trimul, "PRECISION")); self.assertFalse(hasattr(flash_attn, "MIN_TOKENS"))

    def test_faces(self):
        self.assertEqual((triattn.FACE, fused_trimul.FACE, flash_attn.FACE), ("triangle_attention_block", "triangle_multiplication", "attention"))
        self.assertEqual(flash_attn.kind_for(8, 32), "msarow"); self.assertEqual(flash_attn.kind_for(8, 8), "extramsa")
        self.assertEqual(flash_attn.kind_for(4, 16), "tmpl"); self.assertEqual(flash_attn.kind_for(4, 32), "tri")


class TestProviderResolvesTheTierWord(unittest.TestCase):
    """The provider resolves this engine's families by tier word on both cards to a row of the tolerance class or the stock statement — a CLASS contract."""
    def test_af2_families_resolve_on_both_cards(self):
        from opt_core.kernels import pallas as P
        from opt_core.kernels.pallas import serve as PS
        fams = {"triattn": P.family("triattn", form="af2", unit="pair", c=128, heads=4, head_dim=32, orientation="starting"),
                "trimul": P.family("trimul", form="af2", unit="pair", c=128, c_hidden=128, equation="outgoing"),
                "attn": P.family("attn", kind="msarow", heads=8, head_dim=32, n_seq=512)}
        for cc in ("9.0", "8.0"):
            for op, fam in fams.items():
                for word in ("fast", "big"):
                    sel = PS.resolve(op, fam, "float32", 400, word=word, jax_line="0.5", cc=cc)
                    self.assertTrue(sel.candidates, (cc, op, word)); self.assertIn(sel.candidates[-1].split(":")[0].split("@")[0], ("xla", "xla_subbatch4", "xla_sdpa"), sel.candidates)   # the stock statement closes every walk
                    self.assertIn(getattr(sel, "cls", None), ("tol", "exact", "stock", None))

    def test_levers_off_word_is_the_providers(self):
        from opt_core.kernels import pallas as P
        from opt_core.kernels.pallas import serve as PS
        fam = P.family("triattn", form="af2", unit="pair", c=128, heads=4, head_dim=32, orientation="starting")
        with mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "pallas"}):
            try:
                sel = PS.resolve("triattn", fam, "float32", 400, word="fast", jax_line="0.5", cc="9.0")
            except P.Refusal:
                return                                                                          # refused by name: the kit's call site keeps the stock body
            self.assertEqual([c.split(":")[0] for c in sel.candidates][-1], "xla")


class TestL19AndKeys(unittest.TestCase):
    def test_core_dtype_words(self):
        self.assertEqual(triattn.core_dtype({}), "float32"); self.assertEqual(triattn.core_dtype({"AF2IG_OPT_TRIATTN_CORE_DTYPE": "bf16"}), "bfloat16")
        self.assertEqual(triattn.core_dtype({"AF2IG_OPT_TRIATTN_CORE_DTYPE": "fp32"}), "float32")
        self.assertIsNone(triattn.core_dtype_note({"AF2IG_OPT_TRIATTN_CORE_DTYPE": "bf16"})); self.assertIn("unknown_word", triattn.core_dtype_note({"AF2IG_OPT_TRIATTN_CORE_DTYPE": "half"}))

    def test_program_key_carries_tier_and_dtype(self):
        with mock.patch.dict(os.environ, {"AF2IG_OPT_TIER": "fast"}, clear=False):
            f1 = programs.provider_facts("cc9.0")
        with mock.patch.dict(os.environ, {"AF2IG_OPT_TIER": "big"}, clear=False):
            f2 = programs.provider_facts("cc9.0")
        self.assertEqual((f1["tier"], f2["tier"]), ("fast", "big")); self.assertEqual(f1["binding"], "opt_core.kernels.pallas.serve")
        with mock.patch.dict(os.environ, {"AF2IG_OPT_TRIATTN_CORE_DTYPE": "bf16"}, clear=False):
            self.assertEqual(programs.provider_facts()["core_dtype"], "bfloat16")

    def test_bridge_words(self):
        with mock.patch.object(triattn._fused, "served_arms", return_value={"proj+triattn_xla:k": 40}), mock.patch.dict(triattn._STATE, {"core_dtype_applied": "bfloat16"}):
            self.assertTrue(triattn.bridge_word().startswith("on:proj+triattn_xla")); self.assertTrue(triattn.bridge_word().endswith("+core_bfloat16"))
            self.assertEqual(triattn.providers_word(), "proj+triattn_xla:k=40"); self.assertTrue(triattn.bridge_on())
        with mock.patch.object(triattn._fused, "served_arms", return_value={"fpf_block:tf32": 40}), mock.patch.dict(triattn._STATE, {"core_dtype_applied": "float32"}):
            self.assertEqual(triattn.bridge_word(), "none"); self.assertFalse(triattn.bridge_on())

    def test_forwarded_words(self):
        env = {"AF2IG_OPT_TRIATTN_CORE_DTYPE": "bf16", "AF2IG_OPT_TRIATTN_XLA_ROW": "triattn_native", "AF2IG_OPT_TRIATTN_XLA_MIN_LEN": "384"}
        self.assertEqual(stack.lever_words_env(env), {"AF2IG_OPT_TRIATTN_CORE_DTYPE": "bf16"})   # the 0.7.x row / length words are no longer forwarded (nor declared)


if __name__ == "__main__":
    unittest.main()
