"""The W4 driver's per-compute-capability rows (A100 = sm_80): T10's launch config and T3's built-in tile table are chosen by the device's cc; every other
class keeps the H100 rows byte-for-byte; the sm_80 T10 row is the smaller pipeline that launched where the H100 row was refused. Pure bookkeeping —
the device record is stubbed; needs the carried driver importable (torch + the pinned transformers fork), skipped by name otherwise."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward", "fast_inference", "driver")


def _w4():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch not importable: {e!r}")
    sys.path.insert(0, DRIVER_DIR)
    try:
        import ef2_w4
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"ef2_w4 not importable here: {e!r}")
    return ef2_w4


class TestCcKeyedRows(unittest.TestCase):
    def setUp(self):
        self.w4 = _w4(); self.saved = dict(self.w4._DEV)
        import torch; self._avail = torch.cuda.is_available; torch.cuda.is_available = lambda: True     # the selection reads the stubbed device record only

    def tearDown(self):
        import torch; torch.cuda.is_available = self._avail
        self.w4._DEV.clear(); self.w4._DEV.update(self.saved)

    def _as(self, cc, smem):
        self.w4._DEV.update(checked=True, name=f"stub-{cc}", cc=cc, smem_optin=smem, small=(smem < self.w4.SMEM_LARGE_MIN), disabled={})

    def test_h100_rows_unchanged(self):
        self._as("sm_90", 232448)
        self.assertEqual(self.w4._t10_cfg(), dict(BLOCK_SIZE_M=128, BLOCK_SIZE_H=32, num_warps=8, num_stages=3))
        self.assertIs(self.w4.default_tiles(), self.w4.H100_DEFAULT_TILES)
        self.assertEqual(self.w4.H100_DEFAULT_TILES["transition"], {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8})

    def test_a100_rows(self):
        self._as("sm_80", 166912)
        self.assertFalse(self.w4._DEV["small"])                                              # 163 KB >= 160 KB: the large-smem class (T6 / T3 stage-5 tiles run as on H100)
        self.assertEqual(self.w4._t10_cfg(), dict(BLOCK_SIZE_M=64, BLOCK_SIZE_H=32, num_warps=4, num_stages=3))
        t = self.w4.default_tiles(); self.assertEqual(t["transition"]["BLOCK_SIZE_K"], 64); self.assertEqual(t["s5_outgemm"]["TILE_K"], 64)   # the stock K tiles (accumulation sequence unchanged)
        self.assertEqual((t["transition"]["BLOCK_SIZE_M"], t["transition"]["BLOCK_SIZE_N"], t["transition"]["num_stages"], t["transition"]["num_warps"]), (128, 64, 3, 4))

    def test_sm80_t10_row_is_smaller_than_the_refused_h100_row(self):
        """On sm_80 Triton refused the H100 row (out of resource: shared memory, Required 172032 > Hardware limit 166912, A100-80GB, triton 3.7.1); the
        sm_80 row halves the row block at the same hidden chunk and stage count, and launched in the same sweep — it must stay no larger on either axis."""
        h, a = self.w4.T10_CFG, self.w4.T10_CFG_BY_CC["sm_80"]
        self.assertLessEqual(a["BLOCK_SIZE_M"], h["BLOCK_SIZE_M"]); self.assertLessEqual(a["BLOCK_SIZE_H"] * a["num_stages"], h["BLOCK_SIZE_H"] * h["num_stages"])
        self.assertLess(a["BLOCK_SIZE_M"] * a["BLOCK_SIZE_H"] * a["num_stages"], h["BLOCK_SIZE_M"] * h["BLOCK_SIZE_H"] * h["num_stages"])

    def test_small_smem_class_untouched(self):
        self._as("sm_89", 101376)
        self.assertTrue(self.w4._DEV["small"]); self.assertIs(self.w4.default_tiles(), self.w4.SMALL_SMEM_TILES); self.assertEqual(self.w4._t10_cfg(), self.w4.T10_CFG)


if __name__ == "__main__":
    unittest.main()
