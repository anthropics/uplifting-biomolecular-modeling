"""The vortex-kernels FFT chains' shape rules (kit/kshapes.py): sizes, the shared spectrum/output storage, the hcm_fft_conv call predicate."""
import unittest

from evo2_opt.kit import kshapes as KS


class TestKShapes(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(KS.fft_sizes(8192), (16384, 8193))
        self.assertEqual(KS.fft_sizes(4999), (9998, 5000))

    def test_shared_storage_covers_both_tenants(self):
        for B, D, L in ((1, 4096, 8192), (3, 4096, 1024), (8, 4096, 32768), (1, 8192, 5)):
            n, F = KS.fft_sizes(L)
            floats = KS.shared_floats(B, D, L)
            self.assertEqual(floats, B * D * F * 2)                  # the complex spectrum, as fp32 pairs
            self.assertGreaterEqual(floats, B * D * n)               # the fp32 inverse transform fits in the same storage
            self.assertEqual(floats - B * D * n, 2 * B * D)          # larger by one complex bin per row

    def test_resident_bytes_at_the_headline_shape(self):
        r = KS.resident_shape_bytes(1, 4096, 8192)
        self.assertEqual(r["pad_fp32"], 4096 * 16384 * 4)
        self.assertEqual(r["product_complex64"], 4096 * 8193 * 8)
        self.assertEqual(sum(r.values()), 4096 * 16384 * 4 + 2 * 4096 * 8193 * 8)
        self.assertEqual(KS.spectrum_bytes(4096, 8192), 4096 * 8193 * 8)

    def test_hcm_call_predicate(self):
        base = dict(fir_length=128, gate=True, dim_last=False, has_inference_params=False, has_padding_mask=False, has_bias=True,
                    fir_is_conv1d=True, column_split_hyena=False, use_hcm_kernel=True, hcm_bound=True)
        self.assertTrue(KS.is_hcm_fft_call(**base))
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "fir_length": 7}))              # the short gated FIR: hcs_conv's call
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "gate": False, "dim_last": True})) # the featurizer
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "has_inference_params": True}))   # generation prefill: vortex's own path
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "has_padding_mask": True}))
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "use_hcm_kernel": False}))         # the torch-conv route's fftconv_func call
        self.assertFalse(KS.is_hcm_fft_call(**{**base, "hcm_bound": False}))

    def test_spectra_budget(self):
        total = 85_045_542_912                                               # an 80 GB card's total_memory (79.2 GiB)
        self.assertEqual(KS.spectra_budget_bytes(total), total // 7)
        hcl = lambda L: KS.spectrum_bytes(4096, L); hcm = lambda L: KS.spectrum_bytes(256, L)
        self.assertTrue(KS.fits_budget(8 * hcl(32768) + 9 * hcm(32768), hcl(32768), total))      # 32,768 bp: every spectrum cached
        def cached_hcl(L):                                                   # block order interleaves hcm, hcl: hcm 1, hcl 2, hcm 5, hcl 6, ...
            res = 0; n = 0
            for i in range(9):
                if KS.fits_budget(res, hcm(L), total): res += hcm(L)
                if KS.fits_budget(res, hcl(L), total): res += hcl(L); n += 1
            return n, res
        self.assertEqual(cached_hcl(32768)[0], 9)
        self.assertEqual(cached_hcl(65536)[0], 5)
        n, res = cached_hcl(131072)
        self.assertEqual(n, 2)
        self.assertLessEqual(res, total // 7)

    def test_group_rows(self):
        self.assertEqual(KS.group_rows(4096, 256), 16)
        self.assertEqual(KS.group_rows(4096, 4096), 1)
        self.assertEqual(KS.group_rows(4096, None), 1)
        self.assertEqual(KS.group_rows(4096, 1), 4096)
        self.assertEqual(KS.group_rows(4096, 3), 1)


if __name__ == "__main__":
    unittest.main()
