"""The cc 10.3 (B300) rows are additive data: BLK2 / T1 cells for sm103, the v4 TriMul 10.3 cells, the 10.3|3.7 / 10.3|* compositions rows,
SUPPORTED_CC, and configs/b300.env + configs/b200.env as mirrors of configs/h100.env (card lines only); every sm90 / sm100 / sm120 / sm80 row is
the one the other cards' tests and runs read (nothing renamed, nothing shadowed)."""
import ast
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
FPF = os.path.join(KIT, "opt", "forward", "flashpairformer")
B300_SMEM_OPTIN = 232448          # bytes per block (opt-in) on B300 SXM6; torch.cuda.get_device_properties(...).shared_memory_per_block_optin
ARCHES = ("sm90", "sm100", "sm120", "sm80", "sm103")   # CELLS.json blk2_arch: every configured card's arch word plus sm120


from protenix_opt.tests.test_a100_rows import t1_smem_bytes   # the one shared-memory bound of the T1 fused-transition kernel's tiles


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


class B300Rows(unittest.TestCase):
    def setUp(self):
        self.cells = json.load(open(os.path.join(FPF, "CELLS.json"), encoding="utf-8"))

    def test_blk2_arch_lists_every_configured_card(self):
        self.assertEqual(self.cells["blk2_arch"], list(ARCHES), "additive: the existing words first, in their order, sm103 appended")

    def test_sm103_blk2_cells_for_the_block_kernels(self):
        """protenix_opt 0.3.54: the v3 prologue / v2 epilogue of the block path take every (cc, triton) key through opt_core.attn.pair_fused's ONE resolution
        statement — a NAMED row or a step-aside BY NAME to the statement path (never an unnamed fallback); the kit carries no cell of its own for them
        (no transition / prologue / epilogue / by_triton table in CELLS.json). Which keys are served is decided by the shared core's cell tables: the test holds in both states."""
        from opt_core.attn import pair_fused as PFC
        self.assertNotIn("transition", self.cells)                                          # the in-block pair transition is the provider's (opt_core.kernels.transition by tier word): no kit cell
        self.assertNotIn("prologue", self.cells); self.assertNotIn("epilogue", self.cells); self.assertNotIn("by_triton", self.cells)
        for cc, tt in (("10.3", "*"), ("10.3", "3.7"), ("10.0", "*"), ("12.0", "*"), ("9.0", "3.7"), ("8.0", "3.7")):
            for piece, variant in (("prologue", "v3"), ("epilogue", "v2")):
                with self.subTest(cc=cc, triton=tt, piece=piece):
                    _resolves_by_name(self, PFC.lookup_cell("fpf", piece, (256, 8, 32), variant=variant, stack=(cc, tt)), cc)

    def test_readme_rows_10_3(self):
        from protenix_opt import modes, stack
        row = modes.README_ROWS["10.3|3.7"]
        self.assertEqual(row["exact"], {"pre": {}, "post": {}})                                     # PAD8 off on cc >= 10; GLUE_V2 / MK-PF have no 10.3 cells
        self.assertEqual(row["fast"], {"pre": {}, "post": {}})
        self.assertEqual(modes.README_ROWS["10.3|*"], row)
        self.assertEqual(modes.row_key("10.3|3.7"), "10.3|3.7"); self.assertEqual(modes.row_key("10.3|3.9"), "10.3|*"); self.assertEqual(modes.row_key("10.0|3.7"), "10.0|3.7")
        self.assertIn("10.3", stack.SUPPORTED_CC); self.assertTrue({"9.0", "10.0", "8.0"} <= stack.SUPPORTED_CC)

    def _mirror(self, name, card, cc, gb, names):
        c = open(os.path.join(KIT, "configs", name), encoding="utf-8").read().split("\n")
        h = open(os.path.join(KIT, "configs", "h100.env"), encoding="utf-8").read().split("\n")
        i = next(i for i, l in enumerate(c) if l.startswith("# Per-card differences from configs/h100.env"))
        j = next(j for j in range(i + 1, len(c)) if not c[j].startswith("#"))
        block, rest = c[i:j], c[:i] + c[j:]
        for n in names: self.assertIn(n, " ".join(block), n)                                       # every per-card difference named in the one place
        self.assertEqual(len(rest), len(h))
        diff = [(x, y) for x, y in zip(rest, h) if x != y]
        self.assertLessEqual(len(diff), 4, diff)
        for x, y in diff:
            self.assertEqual(re.sub(rf"{card}|{card.lower()}|{re.escape(cc)}|\({gb} GB, compute capability {re.escape(cc)}\) deployment parameters", "#", x),
                             re.sub(r"H100|h100|9\.0|deployment parameters \(the only configuration in this tree\)", "#", y))
        self.assertIn("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-%s}" % card, " ".join(rest))

    def test_b300_env_mirrors_h100_env_except_the_card_lines(self):
        self._mirror("b300.env", "B300", "10.3", 288, ("blk2_block_path", "sm103", "t1_fused_transition", "transition_core", "pad8", "glue_v2", "mk_pf", "trimul_core", "10.3|3.7", "k2b_flash_triattention", "blockfuse_xl"))

    def test_b200_env_mirrors_h100_env_except_the_card_lines(self):
        self._mirror("b200.env", "B200", "10.0", 192, ("blk2_block_path", "sm100", "t1_fused_transition", "pad8", "glue_v2", "mk_pf", "trimul_core", "10.0|3.7", "k2b_flash_triattention"))


if __name__ == "__main__":
    unittest.main()
