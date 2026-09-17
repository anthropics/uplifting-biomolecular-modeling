"""fast is a subset of big BY CONSTRUCTION (R3 tiers): `big` = fast MINUS CUDA-graph capture (modes.ENV_GRAPH_CAPTURE=0: every graph
optimization of the fast line stays in the set and runs its eager forward — the memory line's rule: no CUDA graphs there) MINUS only the levers
with a measured memory cost (modes.BIG_DROP_MEMORY: mh) PLUS the memory levers (modes.BIG_MEMORY_XL: x4; the identity probe off). It is
derived from `modes.FAST_LINE` — the same server mode, fast's XL storage levers and fast's drops — plus ONLY those numbered memory words. A lever present in one tier and absent from its neighbour is in one of those two numbered tables, nowhere
else. The ablation variable (and its release-tree alias MODEL_OPT_LEVERS_OFF) switches every XL lever individually under both modes: an
ablated XL lever leaves the add-on's install set too (Resolution.xl_set), not only its LEVER line."""
import os
import unittest

from esmfold2_opt import ablation, modes, registry, stack


def _kit():
    kit = stack.kit_home()
    return kit if os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)) else None


class FastSubsetBig(unittest.TestCase):
    def test_big_is_derived_from_the_fast_line(self):
        fast, big = modes.KIT_MODES["fast"], modes.KIT_MODES["big"]
        self.assertIs(fast, modes.FAST_LINE); self.assertIs(big, modes.BIG_FAST)
        self.assertEqual(big.server_mode, fast.server_mode)                                   # one lever set in the server's table under both names
        self.assertTrue(fast.xl); self.assertTrue(big.xl)
        self.assertEqual(fast.xl_set, modes.XL_FAST_SET); self.assertEqual(fast.drop, modes.FAST_DROP)
        self.assertEqual(big.xl_set, fast.xl_set + modes.BIG_MEMORY_XL)                     # big's XL set = fast's + the memory-only levers
        self.assertEqual(big.drop, fast.drop + modes.BIG_DROP_MEMORY)                       # big's drops = fast's + the memory drops
        self.assertEqual(modes.BIG_DROP, modes.FAST_DROP + modes.BIG_DROP_MEMORY)
        self.assertEqual({k: v for k, v in big.overrides.items() if k not in (modes.ENV_GRAPH_CAPTURE, modes.ENV_W4_IDPROBE)}, dict(fast.overrides))   # the only extra words: no graph capture, the probe
        self.assertEqual(big.overrides[modes.ENV_GRAPH_CAPTURE], "0"); self.assertEqual(modes.BIG_GRAPH_CAPTURE, "0")   # the memory line captures no CUDA graph at any site
        self.assertNotIn(modes.ENV_GRAPH_BUDGET, big.overrides)                             # no budget word: nothing is captured, so no budget governs anything
        for g in ("tg", "sg", "eg", "ls", "rg", "ro"):                                          # the graph optimizations stay IN the set (they run their eager forwards): fast is a subset of big
            self.assertIn(g, registry.LEVERS)
        self.assertEqual(big.xl_knobs, dict(fast.xl_knobs, esmc_min_tok=modes.X4_MIN_TOKENS))
        self.assertEqual((fast.line, big.line), (None, "fast"))
        self.assertEqual((fast.alloc_strict, big.alloc_strict), (False, True))                # the memory mode refuses a process that cannot take the allocator policy; fast notes it
        for n in modes.XL_FAST_SET + modes.BIG_MEMORY_XL:
            self.assertEqual(registry.LEVERS[n].tier_vs_kit_line, "bitwise", n)                # every XL lever on a line is bitwise on the fused kernels
            self.assertEqual(registry.LEVERS[n].field, registry.FIELD_XL, n)
        self.assertNotIn("x4", modes.XL_FAST_SET)                                               # x4 costs LM-pass time where it engages: memory line only
        self.assertNotIn("x2", modes.XL_FAST_SET + modes.BIG_MEMORY_XL)                       # LOOPFREE is no lever of a line (ls releases in its own frame)

    def test_resolved_sets_differ_only_by_the_numbered_memory_words(self):
        kit = _kit()
        if kit is None:
            self.skipTest("kit tree not present")
        for variant in modes.VARIANTS:
            f = modes.resolve("fast", variant, kit); b = modes.resolve("big", variant, kit)
            only_fast, only_big = set(f.levers) - set(b.levers), set(b.levers) - set(f.levers)
            self.assertLessEqual(only_fast, set(modes.BIG_DROP_MEMORY), (variant, only_fast))   # fast minus ONLY levers with a measured memory cost ...
            self.assertEqual(only_big, set(modes.BIG_MEMORY_XL), (variant, only_big))      # ... plus memory levers
            self.assertEqual(f.xl_set, modes.XL_FAST_SET); self.assertEqual(b.xl_set, modes.XL_BIG_SET)
            self.assertTrue(f.composition["xl"]); self.assertFalse(f.composition["alloc_strict"]); self.assertTrue(b.composition["alloc_strict"])
            for n in modes.FAST_DROP:
                self.assertNotIn(n, f.levers); self.assertNotIn(n, b.levers); self.assertIn(n, f.levers_off)   # x10 owns the distogram move on both lines
            self.assertEqual(f.overrides, {}); self.assertEqual(modes.describe_line(f), "opt14_msa")           # fast's activation spelling is the bare server mode
            self.assertEqual(modes.describe_line(b), "opt14_msa+EF2_GRAPH_CAPTURE=0,EF2_W4_IDENTITY_PROBE=0")   # big's activation spelling: the server mode plus its two words
            self.assertFalse(b.composition["graph_capture"]); self.assertTrue(f.composition["graph_capture"])  # the memory line captures no CUDA graph; fast's budgets decide
            for g in ("tg", "sg", "eg", "ls", "rg", "ro"):
                self.assertIn(g, b.levers); self.assertIn(g, f.levers)                            # the graph optimizations are in BOTH sets (eager under big)
        self.assertFalse(modes.resolve("exact", "fast", kit).composition["xl"])                  # exact carries no XL lever (disto stays there)
        self.assertIn("disto", modes.resolve("exact", "full_msa", kit).levers)

    def test_every_xl_lever_is_individually_switchable_under_both_modes(self):
        kit = _kit()
        if kit is None:
            self.skipTest("kit tree not present")
        for mode, full in (("fast", modes.XL_FAST_SET), ("big", modes.XL_BIG_SET)):
            for n in full:
                res = modes.resolve(mode, "full_msa", kit, ablate=n)
                self.assertNotIn(n, res.levers); self.assertIn(n, res.ablate); self.assertIn(n, res.levers_off)
                self.assertEqual(res.xl_set, tuple(x for x in full if x != n), (mode, n))          # the add-on's install set loses it too (big.xl_install reads res.xl_set)
            everything = modes.resolve(mode, "full_msa", kit, ablate=",".join(full))
            self.assertEqual(everything.xl_set, ()); self.assertTrue(everything.composition["xl"])
            self.assertEqual(stack.xl_levers_on(dict(stack.xl_knobs(1, True, xl_set=everything.xl_set), **everything.xl_knobs)), [])


class LeversOffAlias(unittest.TestCase):
    """MODEL_OPT_LEVERS_OFF is the release tree's uniform spelling: its words naming this kit's levers are ESMFOLD2_OPT_ABLATE tokens; its other
    words address the shared core's kernel packages (they read the same variable) and are left to them, named — never an error, never guessed."""

    def test_alias_words_join_the_kit_variable(self):
        A, U = ablation.ENV_ABLATE, ablation.ENV_LEVERS_OFF
        self.assertEqual((A, U), ("ESMFOLD2_OPT_ABLATE", "MODEL_OPT_LEVERS_OFF")); self.assertEqual(ablation.ENV_WORDS, (A, U))
        self.assertIsNone(ablation.env_text({})); self.assertIsNone(ablation.env_text({U: "trimul_tx pallas:foo"}))   # nothing of this kit: nothing to resolve
        self.assertEqual(ablation.env_text({A: ""}), "")                                                          # set-but-empty stays set (the ACTIVE line says ablate=)
        self.assertEqual(ablation.env_text({U: "x3"}), "x3")
        self.assertEqual(ablation.env_text({U: "x3,trimul_tx dit.gemm=fp32"}), "x3,dit.gemm=fp32")
        self.assertEqual(ablation.env_text({A: "x6, x3", U: "x3 x7 trimul_tx"}), "x6,x3,x7")                       # a word in both variables counts once, order kept
        self.assertEqual(ablation.foreign_words({A: "x6", U: "x3 trimul_tx pallas:t15"}), ["trimul_tx", "pallas:t15"])
        self.assertEqual(ablation.foreign_words({U: "x3"}), []); self.assertIsNone(ablation.foreign_note({U: "x3"}))
        self.assertIn("trimul_tx", ablation.foreign_note({U: "x3 trimul_tx"}))
        self.assertTrue(ablation.kit_word("af.gemm=fp32")); self.assertFalse(ablation.kit_word("nope")); self.assertFalse(ablation.kit_word("triattn_xla:cuda_sm90a"))
        # MODEL_OPT_LEVERS_OFF=compile (the release tree's --no-compile word) is accepted BY NAME as n/a — stock ESMFold2 never compiles and the
        # kit adds no compile step: no lever named compile, nothing turned off, never an error; the ACTIVE line carries compile=n/a (report.COMPILE_WORD)
        self.assertNotIn("compile", registry.LEVERS); self.assertFalse(ablation.kit_word("compile")); self.assertEqual(ablation.COMPILE_WORD, "compile")
        self.assertIsNone(ablation.env_text({U: "compile"}))                                     # nothing of this kit to resolve: the mode's plain set runs
        self.assertEqual(ablation.foreign_words({U: "compile"}), ["compile"])
        self.assertIn("compile: n/a", ablation.foreign_note({U: "compile"})); self.assertNotIn("kernel packages", ablation.foreign_note({U: "compile"}))
        both = ablation.foreign_note({U: "compile trimul_tx"})
        self.assertIn("compile: n/a", both); self.assertIn("kernel packages: trimul_tx", both)
        from esmfold2_opt import report
        self.assertEqual(report.COMPILE_WORD, "compile=n/a")

    def test_alias_tokens_get_the_kit_checks(self):
        kit = _kit()
        if kit is None:
            self.skipTest("kit tree not present")
        text = ablation.env_text({ablation.ENV_LEVERS_OFF: "x3 trimul_tx"})
        res = modes.resolve("fast", "full_msa", kit, ablate=text)
        self.assertNotIn("x3", res.levers); self.assertEqual(res.ablate, ("x3",)); self.assertNotIn("x3", res.xl_set)
        with self.assertRaises(ablation.AblationError):                                          # a kit lever NOT in the mode's set is still refused by name through the alias
            modes.resolve("exact", "full_msa", kit, ablate=ablation.env_text({ablation.ENV_LEVERS_OFF: "x3"}))


if __name__ == "__main__":
    unittest.main()
