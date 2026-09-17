"""transition_core_exact | transition_core: every trunk Transition call binds the shared core's transition provider (opt_core.kernels.transition) by the
mode's TIER WORD — never a row name, never a kit cell table. CPU class contracts: the switch and the words per mode, the marker, the provider's tier words
and the rows its select() names on the kit's two cards for the kit's transition shapes (the kit pins none of them), the deleted kit copies."""
import json
import os
import unittest

from protenix_opt import modes, registry, stack
from protenix_opt.registry import LEVERS

HERE = os.path.dirname(os.path.abspath(__file__))
FPF = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "flashpairformer"))
STACKS = {"9.0": "H100:torch2.13.0+cu130/3.7.1", "8.0": "A100:torch2.13.0+cu130/3.7.1"}


class TransitionCoreContract(unittest.TestCase):
    def test_levers_words_and_modes(self):
        self.assertEqual(LEVERS["transition_core_exact"].tier, registry.EXACT); self.assertEqual(LEVERS["transition_core"].tier, registry.TOLERANCE)
        for n in ("transition_core_exact", "transition_core"):
            self.assertEqual(LEVERS[n].env_keys, ("PTX_TRANSITION",)); self.assertEqual(LEVERS[n].probe, "marker"); self.assertIn(n, stack.MARKERS)
            self.assertIn("TIER WORD", LEVERS[n].description.upper())
        self.assertIn("opt_core.kernels.transition", LEVERS["transition_core_exact"].description)
        self.assertIn("transition_core_exact", modes.MODES["exact"]); self.assertNotIn("transition_core", modes.MODES["exact"])
        for m in ("fast", "big"):
            self.assertIn("transition_core", modes.MODES[m]); self.assertNotIn("transition_core_exact", modes.MODES[m])
        self.assertEqual(modes.BIG_PRE.get("PTX_TRANSITION"), "big", "a big mode binds the word big literally")
        for gone in ("flash_transition", "flash_transition_ln"):
            self.assertNotIn(gone, LEVERS); self.assertNotIn(gone, stack.MARKERS)
        self.assertEqual(stack.MARKERS["transition_core_exact"], ("TRCORE:", ("TRCORE:on(word=exact",)))
        self.assertEqual(stack.MARKERS["transition_core"], ("TRCORE:", ("TRCORE:on(word=fast", "TRCORE:on(word=big")))

    def test_env_sh_exports_the_word_per_arm(self):
        env = open(os.path.join(FPF, "env.sh")).read()
        self.assertIn('export PTX_TRANSITION="${PTX_TRANSITION:-$([ "$_fpf_arm" = "T" ] && echo fast || echo exact)}"', env)
        self.assertIn("transition=$PTX_TRANSITION", env)
        import tempfile
        from protenix_opt.tests.test_modes_match_env_sh import _sandbox_bin
        with tempfile.TemporaryDirectory() as tmp:
            sbin = _sandbox_bin(tmp)
            for mode, word in (("exact", "exact"), ("fast", "fast"), ("big", "big")):
                r = modes.resolve(mode, {"PATH": sbin}, FPF, triton="3.7.1", compute_cap=None)
                got = r.exports.get("PTX_TRANSITION") or r.pre_exports.get("PTX_TRANSITION")
                self.assertEqual(got, word, mode)

    def test_kit_copies_and_cells_deleted(self):
        self.assertFalse(os.path.exists(os.path.join(FPF, "third_party", "protenix_fpf_flash_transition")), "the sealed flash package is the core's flash_sm90a row now")
        cells = json.load(open(os.path.join(FPF, "CELLS.json")))
        self.assertNotIn("transition", cells); self.assertFalse([tv for tv, ov in (cells.get("by_triton") or {}).items() if "transition" in ov])
        src = open(os.path.join(FPF, "src", "ptx_trunk2_levers.py")).read()
        for gone in ("_blk_tr_guarded", "fn_residual as _tr_res", "protenix_fpf_flash_transition", "PTX_FLASH_TRANSITION"):
            self.assertNotIn(gone, src, gone)
        self.assertIn("_TRC.block_pair_transition(self.pair_transition, z)", src); self.assertIn("_TRC.apply(trw)", src)
        b = open(os.path.join(FPF, "src", "ptx_transition_core.py")).read()
        self.assertNotIn("prefer=", b); self.assertIn("T.transition(x, W, word=word", b); self.assertIn("T.select(word,", b)   # the tier word, no row preference

    def test_provider_words_and_cells_on_the_kits_cards(self):
        """The provider carries the three words and names a row (or the stock row / a refusal by name) for the kit's shapes on both cards' stack keys; the exact
        word never names a tolerance row; cc 8.0's exact word refuses the pair transition below its vouch floor (the old kit floor, 2048 GEMM rows ~ 46 tokens, is inside it)."""
        from opt_core.kernels import transition as T
        for w in ("exact", "fast", "big"):
            self.assertIn(w, T.TIER_WORDS, w)
        self.assertEqual(T.cell_word(256, 1024, "pair"), "pair_c256_n4"); self.assertEqual(T.cell_word(384, 1536, "single"), "single_c384_n4")
        self.assertEqual(T.cell_word(64, 256, "rows"), "rows_c64_n4"); self.assertEqual(T.cell_word(64, 128, "rows"), "rows_c64_n2")
        self.assertEqual(T.cell_word(256, 512, "pair"), "pair_c256_n2"); self.assertEqual(T.cell_word(384, 768, "single"), "single_c384_n2")   # the diffusion conditioning's n=2 transitions: cells since opt_core 0.5.117.0 (before: no cell word -> the statement by name)
        for cc, st in STACKS.items():
            for n in (400, 800, 1200):
                for w in ("exact", "fast", "big"):
                    sel = T.select(w, c=256, hidden=1024, n_tokens=n, dtype="bf16", cc=cc, stack=st, family="pair", residual=True, rows_count=n * n, ln_given=(w == "exact"))
                    self.assertNotIn(sel.row, T.STOCK_ROWS, (cc, n, w)); self.assertEqual(sel.tier, w)
                    if w == "exact":
                        self.assertEqual(sel.cls, "bitwise", (cc, n)); self.assertTrue(sel.stack_measured, (cc, n, "the kit's own stack key is a measured column"))
            sel = T.select("exact", c=256, hidden=1024, n_tokens=400, dtype="bf16", cc=cc, stack=st, family="pair", residual=True, rows_count=160000, ln_given=True)
            self.assertEqual(sel.row, {"9.0": "flash_sm90a", "8.0": "v1"}[cc], cc)
        with self.assertRaises(T.Refusal) as cm:                       # cc 8.0: the exact word below the stack's vouch floor names the stock row (the module statement runs, by name)
            T.select("exact", c=256, hidden=1024, n_tokens=20, dtype="bf16", cc="8.0", stack=STACKS["8.0"], family="pair", residual=True, rows_count=400, ln_given=True)
        self.assertIn("exact_vouch", cm.exception.kind); self.assertIn(cm.exception.fallback, T.STOCK_ROWS)
        for n in (20, 33, 45):                                          # the old kit floor (2048 rows) is inside the provider's: nothing fused serves exact below it on 8.0
            with self.assertRaises(T.Refusal):
                T.select("exact", c=256, hidden=1024, n_tokens=n, dtype="bf16", cc="8.0", stack=STACKS["8.0"], family="pair", residual=True, rows_count=n * n, ln_given=True)
        s9 = T.select("exact", c=256, hidden=1024, n_tokens=20, dtype="bf16", cc="9.0", stack=STACKS["9.0"], family="pair", residual=True, rows_count=400, ln_given=True)
        self.assertIn(s9.row, T.STOCK_ROWS, "cc 9.0 exact at 20 tokens: the statement is the cell's row (the module runs it by name)")


if __name__ == "__main__":
    unittest.main()
