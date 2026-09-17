"""ef2_dit dit0.4 — the fused diffusion-transformer step at num_diffusion_samples > 1 (the S samples of ONE input stacked as [S*L, D] rows).

Every GEMM / LayerNorm / gate of the DiT blocks is row-wise; the pair-bias attention runs PER SAMPLE — the B == 1 attention on rows
[i*L, (i+1)*L) against the fold's one [H, L, L] bias (the pair conditioning is the input's and shared by its samples, exactly as stock's
repeat_interleave shares it). So sample i's rows at S > 1 must equal what the S == 1 path computes on that sample alone. These CPU tests drive
`_token_transformer_dit` on the torch paths (gemm / cond fp32, attention `sdpa`, fused_ew off) with random packed weights of the real layout
(D = H * HD, nb blocks) — the slicing / sharing logic, which is all dit0.4 adds over dit0.3 (S == 1 issues exactly what dit0.3 issued)."""
import os, sys, types, unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
M = None


def setUpModule():
    global M
    import importlib.util
    spec = importlib.util.spec_from_file_location("ef2_dit", os.path.join(DRIVER, "ef2_dit.py"))
    M = importlib.util.module_from_spec(spec)
    sys.modules["ef2_dit"] = M
    spec.loader.exec_module(M)


def tearDownModule():
    sys.modules.pop("ef2_dit", None)


def _w(k, n, g):
    """a packed weight as _mm reads it under fp32: .t32 = W^T [K, N]."""
    return types.SimpleNamespace(t32=torch.randn(k, n, generator=g) * (k ** -0.5), N=n)


def fake_state(nb=2, H=2, HD=8, hid=24, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = H * HD
    st = types.SimpleNamespace()
    st.cfg = dict(gemm="fp32", cond="fp32", attn="sdpa", attn_precision="ieee", fused_ew=False)
    st.nb, st.H, st.HD, st.D, st.hid, st.eps, st.scale = nb, H, HD, D, hid, 1e-5, HD ** -0.5
    st.w_cs_all = _w(D, nb * 4 * D, g); st.w_og_all = _w(D, nb * 2 * D, g)
    st.w_qkvg = [_w(D, 4 * D, g) for _ in range(nb)]; st.w_o = [_w(D, D, g) for _ in range(nb)]
    st.w_sw = [_w(D, 2 * hid, g) for _ in range(nb)]; st.w_lo = [_w(hid, D, g) for _ in range(nb)]
    st.bq = [torch.randn(D, generator=g) for _ in range(nb)]
    st.bcs_a = [torch.randn(D, generator=g) for _ in range(nb)]; st.bcs_t = [torch.randn(D, generator=g) for _ in range(nb)]
    st.bog_a = [torch.randn(D, generator=g) for _ in range(nb)]; st.bog_t = [torch.randn(D, generator=g) for _ in range(nb)]
    st.pb = [[torch.randn(H, L, L, generator=g)] for _ in range(nb)]          # the fold's ONE [H, L, L] bias per block (pair conditioning of the input: batch 1)
    return st


class TestDitMultiSample(unittest.TestCase):
    def test_each_sample_at_S_gt_1_is_the_S1_arithmetic_on_its_rows(self):
        L, S = 6, 3
        st = fake_state(L=L)
        D = st.D
        g = torch.Generator().manual_seed(1)
        a = torch.randn(S * L, D, generator=g); s = torch.randn(S * L, D, generator=g)     # token rows of S samples stacked; per-row conditioning s
        xs = M._token_transformer_dit(st, None, a, s, False, S)
        self.assertEqual(tuple(xs.shape), (S * L, D))
        for i in range(S):
            xi = M._token_transformer_dit(st, None, a[i * L:(i + 1) * L].clone(), s[i * L:(i + 1) * L].clone(), False, 1)
            self.assertTrue(torch.allclose(xs[i * L:(i + 1) * L], xi, rtol=1e-4, atol=1e-4),
                            f"sample {i}: max |d| = {(xs[i * L:(i + 1) * L] - xi).abs().max().item():.3e}")
        self.assertGreaterEqual(M.STATS["attn_multi_sample_blocks"], st.nb)

    def test_S1_issues_one_attention_per_block_and_no_multi_sample_path(self):
        L = 6
        st = fake_state(L=L)
        g = torch.Generator().manual_seed(2)
        a = torch.randn(L, st.D, generator=g); s = torch.randn(L, st.D, generator=g)
        n0 = M.STATS["attn_multi_sample_blocks"]; c0 = M.STATS["attn_sdpa_ieee"]
        x = M._token_transformer_dit(st, None, a, s, False, 1)
        self.assertEqual(tuple(x.shape), (L, st.D))
        self.assertEqual(M.STATS["attn_multi_sample_blocks"] - n0, 0)
        self.assertEqual(M.STATS["attn_sdpa_ieee"] - c0, st.nb)                  # one attention per block
        x5 = M._token_transformer_dit(st, None, a.repeat(5, 1), s.repeat(5, 1), False, 5)
        self.assertEqual(M.STATS["attn_sdpa_ieee"] - c0, st.nb + 5 * st.nb)     # S launches per block at S = 5
        self.assertTrue(torch.allclose(x5[4 * L:], x, rtol=1e-4, atol=1e-4))     # identical samples -> identical rows

    def test_samples_do_not_attend_across_each_other(self):
        """perturbing sample 1's rows must leave sample 0's output unchanged (attention is per sample; everything else is row-wise)."""
        L, S = 5, 2
        st = fake_state(L=L, seed=3)
        g = torch.Generator().manual_seed(4)
        a = torch.randn(S * L, st.D, generator=g); s = torch.randn(S * L, st.D, generator=g)
        x = M._token_transformer_dit(st, None, a, s, False, S)
        a2 = a.clone(); a2[L:] += 10.0
        x2 = M._token_transformer_dit(st, None, a2, s, False, S)
        self.assertTrue(torch.equal(x[:L], x2[:L]))
        self.assertFalse(torch.allclose(x[L:], x2[L:]))

    def test_rows_must_be_a_whole_number_of_samples(self):
        st = fake_state(L=4)
        with self.assertRaises(AssertionError):
            M._token_transformer_dit(st, None, torch.zeros(7, st.D), torch.zeros(7, st.D), False, 2)

    def test_scope_words(self):
        self.assertEqual((M.lever_scope("ro")["num_diffusion_samples"], M.lever_scope("ro")["batch"]), (1, 1))
        for name in ("kd", "dit"):
            self.assertEqual((M.lever_scope(name)["num_diffusion_samples"], M.lever_scope(name)["batch"]), ("N", "N"))

    def test_device_kabsch_head_steps_aside_on_cpu(self):
        """the structure head's _weighted_rigid_align replacement (kd outside the roll-out) runs the stock statements for CPU tensors."""
        calls = []
        M._STOCK["weighted_rigid_align"] = lambda x, x_gt, w, mask: (calls.append(1), x)[1]
        try:
            x = torch.randn(2, 7, 3); n0 = M.STATS["kabsch_head_stock"]
            out = M._weighted_rigid_align_device(x, x + 1.0, torch.ones(2, 7), torch.ones(2, 7))
            self.assertTrue(torch.equal(out, x)); self.assertEqual(len(calls), 1); self.assertEqual(M.STATS["kabsch_head_stock"] - n0, 1)
        finally:
            M._STOCK.pop("weighted_rigid_align", None)


if __name__ == "__main__":
    unittest.main()
