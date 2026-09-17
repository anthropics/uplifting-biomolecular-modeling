"""The torch-conv chains' shape rules (kit/kshapes.py): the long-filter build's channel tiles, the inverse transform's view over the full-row
spectrum storage, the medium filter's rows per group."""
import unittest

from evo2_opt.kit import kshapes as KS


class TestTorchConvShapes(unittest.TestCase):
    def test_filter_tiles_cover_every_channel_once_in_order(self):
        for D, tile in ((4096, 512), (8192, 512), (4096, 4096), (100, 512), (1000, 512), (513, 512), (1, 512)):
            tiles = KS.filter_tiles(D, tile)
            self.assertEqual(tiles[0][0], 0)
            self.assertEqual(tiles[-1][1], D)
            for (a, b), (c, _) in zip(tiles, tiles[1:]):
                self.assertEqual(b, c)                                   # consecutive: no gap, no overlap
            for a, b in tiles:
                self.assertTrue(0 < b - a <= tile)
            self.assertEqual(sum(b - a for a, b in tiles), D)
            self.assertEqual(len(tiles), -(-D // tile))
        self.assertEqual(KS.filter_tiles(4096, 512), [(i * 512, (i + 1) * 512) for i in range(8)])
        with self.assertRaises(AssertionError):
            KS.filter_tiles(0, 512)

    def test_out_view_fits_in_the_full_row_spectrum_storage(self):
        for B, D, L in ((1, 4096, 8192), (4, 8192, 8192), (3, 4096, 3001), (1, 8192, 32768)):
            need, have = KS.out_view_floats(B, D, L)
            self.assertEqual(need, B * D * 2 * L)                         # (B, D, n) fp32, n = 2L
            self.assertEqual(have, 2 * need)                              # (B, D, n) complex64 = twice the floats
            self.assertLessEqual(need, have)

    def test_group_rows_of_the_medium_filter(self):
        self.assertEqual(KS.group_rows(4096, 256), 16)                    # evo2_7b's HCM blocks: 256 filter groups over 4096 channels
        self.assertEqual(KS.group_rows(8192, 512), 16)
        self.assertEqual(KS.group_rows(4096, 1), 4096)                    # one group: every row the same filter
        self.assertEqual(KS.group_rows(4096, None), 1)                    # no groups argument: full rows
        self.assertEqual(KS.group_rows(4096, 4096), 1)                    # one row per group already
        self.assertEqual(KS.group_rows(100, 7), 1)                        # not a divisor: full rows


if __name__ == "__main__":
    unittest.main()
