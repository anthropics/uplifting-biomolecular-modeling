"""kit/rmsnorm_rule.py: ATen's partial-thread count for the bf16 row norm (the fused RMSNorm kernel's order parameter)."""
import unittest

from evo2_opt.kit import rmsnorm_rule as RR


class TestAtenPartials(unittest.TestCase):
    def test_evo2_7b_width(self):
        for rows, T in ((1, 512), (2, 256), (3, 256), (4, 128), (5, 128), (7, 128), (8, 64), (9, 64), (15, 64), (16, 32), (17, 32), (300, 32), (8192, 32), (3 * 32768, 32)):
            self.assertEqual(RR.aten_partials(rows, 4096), T, rows)
            self.assertEqual(RR.row_rule(rows, 4096), T, rows)

    def test_evo2_40b_width(self):
        for rows in (1, 2, 7, 16, 300, 8192, 2 * 8192):
            self.assertEqual(RR.aten_partials(rows, 8192), 512, rows)

    def test_not_served(self):
        self.assertIsNone(RR.aten_partials(64, 512))          # H/4 <= 128: another ATen scheme
        self.assertIsNone(RR.row_rule(64, 4096 + 2048))       # not a multiple of the tail chunk
        self.assertIsNone(RR.aten_partials(1, 131072))        # a grid-wide second pass may be added
        self.assertIsNone(RR.row_rule(0, 4096))


if __name__ == "__main__":
    unittest.main()
