"""The cc 8.0 (A100) rows are additive data: kit cells for sm80 within A100's opt-in shared memory, the v4 TriMul 8.0 cells, the
8.0|3.7 compositions row, the exact-TriMul device caps, configs/a100.env as a mirror of configs/h100.env (card lines only), and the
block core's fused tri-attention statement on sm80 cells from a token floor up (CELLS.json blk2_triatt_min_tokens; protenix_opt 0.3.43): the
stock statement below it by name, the levers inside the statement in the arm on that card like H100's."""
import ast
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
FPF = os.path.join(KIT, "opt", "forward", "flashpairformer")
A100_SMEM_OPTIN = 166912          # bytes per block (opt-in) on A100; torch.cuda.get_device_properties(...).shared_memory_per_block_optin


def t1_smem_bytes(C, NH, BM, BH, S):
    """Shared memory the T1 fused-transition kernel stages per CTA: the [BM, C] activation tile + S stages of the [C, 2*BH] gate/up
    weight tiles and the [BH, C] down tile, bf16 (an upper bound of triton's allocation for these cells)."""
    return 2 * (BM * C + S * (C * 2 * BH + BH * C))


def _resolves_by_name(tc, d, cc):
    """The pair_fused CONTRACT for one (cc, triton) key (protenix_opt 0.3.54; replaces served-cell pins): the tier word resolves to a NAMED row
    (``row`` a settings dict, ``served_by`` names the table key / 'safe' / 'default') OR the block kernel steps aside BY NAME to the statement path
    (``row`` None, ``reason`` the provider's ``no-cell:<impl>:<piece>:<key>:<cc>|<triton>…`` word) — never an unnamed fallback.  Which of the two
    holds for a key is decided by the shared core's cell tables, not kit code: the test passes in both states."""
    tc.assertEqual(d.cc, cc, d)
    if d.row is not None:
        tc.assertTrue(d.served_by, f"a served row must name the table key that served it: {d}")
        tc.assertIsInstance(d.row.get("cfg"), dict, d)
    else:
        tc.assertEqual(d.served_by, "", d)
        tc.assertTrue(d.reason.startswith(f"no-cell:{d.impl}:{d.piece}:") and f":{cc}|" in d.reason, f"an unserved key must step aside BY NAME: {d}")


class A100Rows(unittest.TestCase):
    def setUp(self):
        self.cells = json.load(open(os.path.join(FPF, "CELLS.json"), encoding="utf-8"))

    def test_sm80_block_cells_and_no_kit_transition_table(self):
        """cc 8.0: the block path engages (blk2_arch) with the unit's epilogue sm80 cell; the transitions are the provider's (opt_core.kernels.transition by tier
        word) — this kit carries no transition kernel cell, tile table, shared-memory gate, per-M guard or small-M floor (the provider's exact word refuses the
        pair transition below its own vouch floor on this card, by name)."""
        for gone in ("t1_fused", "t1_fused_min_smem_kb", "t1_guard_by_m_arch", "small_m_stock_rows", "transition"):
            self.assertNotIn(gone, self.cells, gone)
        self.assertIn("sm80", self.cells["blk2_arch"])
        self.assertNotIn("prologue", self.cells); self.assertNotIn("epilogue", self.cells)          # protenix_opt 0.3.51: the v3 / v2 cells are opt_core.attn.pair_fused rows (read by the trunk levers through the provider)
        from opt_core.attn import pair_fused as PFC                                            # protenix_opt 0.3.54: the CONTRACT, not a served-cell pin — for the (8.0, 3.7) key each block kernel
        for piece, variant in (("prologue", "v3"), ("epilogue", "v2")):                        # resolves to a named row or steps aside by name to the statement path (the core's cell tables decide)
            _resolves_by_name(self, PFC.lookup_cell("fpf", piece, (256, 8, 32), variant=variant, stack=("8.0", "3.7")), "8.0")
        self.assertEqual(self.cells["blk2_triatt_stock_arch"], [])                      # no configured arch runs the stock tri-attention statement for want of cells …
        self.assertEqual(self.cells["blk2_triatt_min_tokens"], {"sm80": 400})           # … sm80 runs it below a token floor, by name (the fused statement's bitwise measurement on sm80 starts at the floor)

    def test_readme_row_8_0(self):
        from protenix_opt import modes
        row = modes.README_ROWS["8.0|3.7"]
        self.assertEqual(row["exact"], {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_TRIATT_EXACT": "1"}})   # the statement's GLUE_V2 / MK-PF F1 8.0 cells (protenix_opt 0.3.43); PAD8 not set on this card
        self.assertEqual(row["fast"], {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1"}})   # T: v4 TriMul + the same cells + the sampler attention levers through the kit's cc-8.0 binding apb_sm80 (protenix_opt 0.3.37)
        for other in ("9.0|3.3", "9.0|3.7", "10.0|3.7"):
            self.assertIn(other, modes.README_ROWS)

    def test_a100_env_mirrors_h100_env_except_the_card_lines(self):
        """configs/a100.env = configs/h100.env line for line (card label lines reworded) plus ONE declared block: the per-card differences
        list (comment lines only): the modes compose the same lever set on both cards, so the file exports nothing h100.env does not."""
        a = open(os.path.join(KIT, "configs", "a100.env"), encoding="utf-8").read().split("\n")
        h = open(os.path.join(KIT, "configs", "h100.env"), encoding="utf-8").read().split("\n")
        i = next(i for i, l in enumerate(a) if l.startswith("# Per-card differences from configs/h100.env"))
        j = next(j for j in range(i + 1, len(a)) if not a[j].startswith("#"))
        block, rest = a[i:j], a[:i] + a[j:]
        for name in ("blk2_block_path", "STOCK tri-attention statement", "blk2_triatt_stock_arch", "CARD LEVER SET", "nomask", "blk2_chunked_exact", "blk2_chunked_k2b",
                     "k2b_flash_triattention", "pad8", "glue_v2", "mk_pf", "t1_guard_by_m_arch", "trimul_core", "blockfuse_xl", "lean_triatt_stmts",
                     "sampler_graph_maxtok", "transition_core_exact"):
            self.assertIn(name, " ".join(block), name)                 # every per-card difference named in the one place
        self.assertEqual(len(rest), len(h))
        diff = [(x, y) for x, y in zip(rest, h) if x != y]
        self.assertLessEqual(len(diff), 4, diff)
        for x, y in diff:
            self.assertEqual(re.sub(r"A100|a100|8\.0|\(80 GB, compute capability 8\.0\) deployment parameters", "#", x),
                             re.sub(r"H100|h100|9\.0|deployment parameters \(the only configuration in this tree\)", "#", y))
        self.assertIn("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}", " ".join(rest))

    def test_broad_rows_make_an_exact_key_an_override(self):
        from protenix_opt import modes
        self.assertEqual(modes.row_key("9.0|3.7"), "9.0|3.7"); self.assertEqual(modes.row_key("9.0|3.9"), "9.0|*")
        self.assertEqual(modes.row_key("8.0|3.5"), "8.0|*"); self.assertEqual(modes.row_key("10.0|4.0"), "10.0|*"); self.assertEqual(modes.row_key("8.9|3.7"), "other")
        self.assertEqual(modes.row_key(None), "other")
        self.assertEqual(modes.README_ROWS["9.0|*"], modes.README_ROWS["9.0|3.7"]); self.assertEqual(modes.README_ROWS["8.0|*"], modes.README_ROWS["8.0|3.7"])
        self.assertEqual(modes.README_ROWS["10.0|*"], modes.README_ROWS["10.0|3.7"])
        self.assertEqual(modes.readme_row("8.0|3.5", "fast"), {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1"}})   # fast on an unlisted triton keeps the v4 TriMul, the statement cells and the sampler attention levers

    def test_card_lever_set_line(self):
        from protenix_opt import report
        rep = {"active": True, "mode": "exact", "kernel_key": "8.0|3.7", "levers_not_in_arm": ["nomask", "glue_v2", "mk_pf", "diffusion_chunk"],
               "fallback_reasons": {"nomask": "the block core runs the stock tri-attention statement on gpu_arch=sm80 (…)",
                                    "glue_v2": "not in the kit README row for kernel key 8.0|3.7 (…)", "mk_pf": "not in the kit README row for kernel key 8.0|3.7 (…)",
                                    "diffusion_chunk": "opt-in switch unset: PTX_DIFF_CHUNK"}}
        line = report.card_lever_set_line(rep)
        self.assertTrue(line.startswith("[protenix-opt] CARD LEVER SET kernel_key=8.0|3.7: not in the arm on this card — nomask (the block core runs the stock tri-attention statement on gpu_arch=sm80"), line)
        self.assertIn("; glue_v2 (not in the kit README row", line); self.assertIn("; mk_pf (not in the kit README row", line); self.assertNotIn("diffusion_chunk", line)      # an opt-in left unset is not a card difference
        self.assertIsNone(report.card_lever_set_line({"active": True, "levers_not_in_arm": [], "fallback_reasons": {}}))
        self.assertIsNone(report.card_lever_set_line({"active": True, "levers_not_in_arm": ["diffusion_chunk"], "fallback_reasons": {"diffusion_chunk": "opt-in switch unset: X"}}))


class CardLeverSet(unittest.TestCase):
    """A mode is one lever set on every card; a BLK2 lever on an arch without BLK2 cells (here sm86, an unconfigured card) is unavailable
    (the run refuses by name: a mode is all of its levers on the card). cc 8.0 itself carries BLK2 cells (blk2_arch sm80) and runs the block core's
    fused tri-attention statement on sm80 cells (blk2_triatt_stock_arch empty since protenix_opt 0.3.43): its classification is H100's, and the block
    core switch stays as env.sh sets it."""

    def test_resolve_on_cc_8_0_keeps_the_block_core_switch(self):
        from protenix_opt import modes, stack
        base = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")}
        r9 = modes.resolve("exact", dict(base), stack.kit_home(), compute_cap="9.0", triton="3.7.1", probe_gpu=False)
        self.assertEqual(r9.final(base).get("PTX_BLK"), "2")
        for mode in ("exact", "fast", "big"):
            r8 = modes.resolve(mode, dict(base), stack.kit_home(), compute_cap="8.0", triton="3.7.1", probe_gpu=False)
            self.assertEqual(r8.final(base).get("PTX_BLK"), "2", mode)

    def test_classify_blk2_without_cells_is_unavailable_and_cc_8_0_classifies_as_h100(self):
        from protenix_opt import stack, modes
        base = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")}
        res = modes.resolve("exact", dict(base), stack.kit_home(), compute_cap="8.0", triton="3.7.1", probe_gpu=False)
        env = res.final(base)
        on, fb, skipped, why = stack._classify("exact", [], env, row=res.row, row_key=res.row_key, sm="sm86")
        for n in ("blk2_block_path", "blk2_chunked_exact"):
            self.assertIn(n, fb); self.assertNotIn(n, skipped); self.assertIn("no BLK2 cells for gpu_arch=sm86", why[n]); self.assertNotIn("--allow-partial", why[n]); self.assertIn("the mode refuses by name", why[n])
        on9, fb9, sk9, why9 = stack._classify("exact", [], env, row=res.row, row_key=res.row_key, sm="sm90")
        onN, fbN, skN, whyN = stack._classify("exact", [], env, row=res.row, row_key=res.row_key)     # no device probed
        self.assertEqual((on9, fb9, sk9, why9), (onN, fbN, skN, whyN))                                # H100: the classification is unchanged
        self.assertFalse(any(str(v).startswith("no BLK2 cells") for v in why9.values()))
        on80, fb80, sk80, why80 = stack._classify("exact", [], env, row=res.row, row_key=res.row_key, sm="sm80")   # A100: BLK2 cells and the tri-attention statement's sm80 cells exist -> as H100
        inside = [n for n in stack.TRIATT_STATEMENT_LEVERS if n in modes.MODES["exact"]]
        self.assertEqual(inside, ["nomask", "blk2_chunked_exact"])
        self.assertEqual((on80, fb80, sk80, why80), (on9, fb9, sk9, why9)); self.assertNotIn("blk2_block_path", sk80)
        self.assertFalse(any(str(v).startswith(stack.STOCK_TRIATT) for v in why80.values()))                # no lever steps aside for the stock statement on sm80 (the token floor is a runtime row, not a classification)
        self.assertFalse(stack.cells("sm80")["triatt_stock"]); self.assertFalse(stack.cells("sm90")["triatt_stock"])
        _, fbt, skt, whyt = stack._classify("fast", [], modes.resolve("fast", dict(base), stack.kit_home(), compute_cap="8.0", triton="3.7.1", probe_gpu=False).final(base),
                                            row=modes.readme_row("8.0|3.7", "fast"), row_key="8.0|3.7", sm="sm80")
        for n in ("nomask", "blk2_chunked_k2b", "k2b_flash_triattention"):
            self.assertFalse(str(whyt.get(n, "")).startswith(stack.STOCK_TRIATT), n)                      # fast on cc 8.0: K2B and the chunked core classify as on H100
        onf, fbf, skf, whyf = stack._classify("fast", [], modes.resolve("fast", dict(base), stack.kit_home(), compute_cap="8.0", triton="3.7.1", probe_gpu=False).final(base),
                                              row=modes.readme_row("8.0|3.7", "fast"), row_key="8.0|3.7", sm="sm86")
        self.assertIn("blk2_chunked_k2b", fbf); self.assertTrue(whyf["blk2_chunked_k2b"].startswith("no BLK2 cells"))
        self.assertFalse(str(whyf.get("k2b_flash_triattention", "")).startswith("no BLK2 cells"))          # K2B tri-attention is not a BLK2 lever: never refused for the arch's block cells


class Sm80StatementCells(unittest.TestCase):
    """protenix_opt 0.3.43 -> 0.3.51: the block statement's GLUE_V2 / MK-PF kernels take their compute-capability-8.0 cells from the SHARED CORE
    packages' own tables (entry sm80_t37 of opt_core kernels/fpf_glue_v2/GLUE_CELLS.json and kernels/fpf_mkpf/MKPF_CELLS.json, lifted verbatim with provenance
    from this unit's former src/ptx_cells_sm80 tables) — the kit ships no cell table of its own and env.sh exports no cells hook; the unit's CELLS.json carries
    the sm80 token floor of the fused tri-attention statement."""

    def test_no_kit_side_cell_tables(self):
        self.assertFalse(os.path.exists(os.path.join(FPF, "src", "ptx_cells_sm80")))

    def test_glue_8_0_cells_are_the_core_packages_own_entry(self):
        from opt_core.kernels import fpf_glue_v2 as G
        d = json.load(open(os.path.join(os.path.dirname(G.__file__), "GLUE_CELLS.json"), encoding="utf-8"))
        self.assertEqual(d["by_cc_triton"]["8.0|3.7"], "sm80_t37"); self.assertEqual(d["by_cc"]["8.0"], "sm80_t37")
        ent = d["entries"]["sm80_t37"]
        self.assertEqual(tuple(ent["prologue"]["shape"]), (256, 256)); self.assertEqual(tuple(ent["epilogue"]["shape"]), (256, 8, 32))
        self.assertEqual(ent["prologue"]["cfg"], {"BI": 8, "BJ": 8, "num_warps": 8, "num_stages": 1, "BN": 64, "NUM_STAGES_W": 2, "WS": 0, "GLOOP": 1})
        self.assertEqual(ent["epilogue"]["cfg"], {"BI": 16, "BJ": 8, "num_warps": 8, "num_stages": 1, "KC": 32, "NUM_STAGES_K": 2, "WS": 0})          # no warp_specialize on sm_80
        self.assertNotIn("transition", ent); self.assertTrue(ent.get("provenance"))                                                                # the in-block transition keeps the unit's CELLS.json (256, 1024, 'sm80') cell

    def test_statement_cells_resolve_by_name_on_8_0(self):
        """protenix_opt 0.3.54: the block statement's MK-PF F1 / GLUE_V2 prologue-epilogue kernels take the (8.0, 3.7) key through the shared core's ONE
        resolution statement (opt_core.attn.pair_fused.lookup_cell): a NAMED row or a step-aside BY NAME — the kit pins no served cell, no provenance and
        no settings of the core's tables (their rows are the core's to measure and promote)."""
        from opt_core.attn import pair_fused as PFC
        for piece, variant in (("prologue", "mkpf_f1"), ("prologue", "prologue_v4"), ("epilogue", "epilogue_v3"), ("prologue", "v3"), ("epilogue", "v2")):
            with self.subTest(piece=piece, variant=variant):
                _resolves_by_name(self, PFC.lookup_cell("fpf", piece, (256, 8, 32), variant=variant, stack=("8.0", "3.7")), "8.0")

    def test_env_sh_exports_no_cells_hook(self):
        src = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
        self.assertNotIn("export FPF_GLUE_CELLS_JSON", src); self.assertNotIn("export FPF_MKPF_CELLS_JSON", src); self.assertNotIn("ptx_cells_sm80/", src.replace("former src/ptx_cells_sm80 tables", ""))

    def test_the_token_floor_is_read_by_the_trunk_levers_as_data(self):
        src = open(os.path.join(FPF, "src", "ptx_trunk2_levers.py"), encoding="utf-8").read()
        self.assertIn('_CELLS.get("blk2_triatt_min_tokens")', src); self.assertIn("def _triatt_below_floor(z):", src)
        self.assertEqual(src.count("and not _triatt_below_floor(z)"), 2)                                                # both tri-attention nodes of the block statement
        self.assertIn("blk_triatt_below_tokens", src)


if __name__ == "__main__":
    unittest.main()
