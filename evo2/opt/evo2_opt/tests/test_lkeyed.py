"""kit/lkeyed.py: the length-keyed caches hold one length at a time across every registered store; batch changes keep them."""
import unittest

from evo2_opt.kit import lkeyed as LK


class TestOneLength(unittest.TestCase):
    def setUp(self):
        LK._REGISTRY.clear(); LK.clear_all(); LK.OneLength.evictions = 0
        self.filters = LK.OneLength("filters")            # keys (owner, L, device)
        self.spectra = LK.OneLength("spectra")            # keys (layer, L, device)

    def test_one_length_resident_across_stores(self):
        for layer in range(3):
            self.filters[(100 + layer, 8192, "cuda:0")] = f"h{layer}"
            self.spectra[(layer, 8192, "cuda:0")] = f"H{layer}"
        self.assertEqual((len(self.filters), len(self.spectra), LK.resident_length()), (3, 3, 8192))
        self.spectra[(0, 8192, "cuda:1")] = "H0'"                             # another device, same length: kept
        self.assertEqual(len(self.spectra), 4)
        self.spectra[(0, 1024, "cuda:0")] = "H0@1024"                          # a new length: every store's 8192 entries go first
        self.assertEqual((len(self.filters), len(self.spectra), LK.resident_length(), LK.OneLength.evictions), (0, 1, 1024, 1))
        self.assertEqual(list(self.spectra), [(0, 1024, "cuda:0")])

    def test_batch_is_not_in_the_key(self):
        self.spectra[(0, 8192, "cuda:0")] = "H0"
        for B in (1, 4, 2):                                                    # the ragged last batch of a score_sequences call and back
            self.assertFalse(LK.hold(8192))
        self.assertEqual((len(self.spectra), LK.OneLength.evictions), (1, 0))

    def test_hold_before_the_first_write_and_clear(self):
        self.spectra[(0, 8192, "cuda:0")] = "H0"
        self.assertTrue(LK.hold(4999))                                         # released before anything of the new length allocates
        self.assertEqual((len(self.spectra), LK.resident_length()), (0, 4999))
        self.assertFalse(LK.hold(4999))
        LK.clear_all()
        self.assertIsNone(LK.resident_length())
        self.assertFalse(LK.hold(8192))                                        # nothing resident: nothing released


if __name__ == "__main__":
    unittest.main()
