"""``fast``'s batched multichain route is upstream's function statement for statement: on a small randomly initialised
``GVPTransformerModel`` (CPU, no weights file), ``batched.sample_complex_batch`` at B = 1 under one ``torch.manual_seed`` returns the string
``multichain_util.sample_sequence_in_complex`` returns under the same seed, for every chain of a three-chain complex; ``batched.sample_batch``
at B = 1 equals ``model.sample``; and at a vanishing temperature (the draw no longer depends on the random stream) every row of a B = 3 batch
equals upstream's own call. Skips by name where the model stack (torch, fair-esm and its graph packages) is not importable.
Runs under pytest and under ``python -m unittest esm_if1_opt.tests.test_multichain_route`` alike."""
import argparse
import unittest


def tiny_model():
    """upstream's model class with small dimensions: the argument names ``esm.pretrained`` reads off the checkpoint, nothing else."""
    import torch
    from esm.data import Alphabet
    from esm.inverse_folding.gvp_transformer import GVPTransformerModel
    args = argparse.Namespace(encoder_embed_dim=32, decoder_embed_dim=32, encoder_ffn_embed_dim=64, decoder_ffn_embed_dim=64, encoder_attention_heads=4,
                              decoder_attention_heads=4, encoder_layers=1, decoder_layers=1, dropout=0.1, attention_dropout=0.1,
                              gvp_node_hidden_dim_scalar=16, gvp_node_hidden_dim_vector=4, gvp_edge_hidden_dim_scalar=8, gvp_edge_hidden_dim_vector=2,
                              gvp_dropout=0.1, gvp_num_encoder_layers=1, gvp_top_k_neighbors=4)
    torch.manual_seed(0)
    return GVPTransformerModel(args, Alphabet.from_architecture("invariant_gvp")).eval()


class MultichainRouteIsUpstreams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
            import esm.inverse_folding                                    # noqa: F401 — pulls the graph packages; absent -> skip by name
            import esm.inverse_folding.multichain_util as MU
        except ImportError as e:                                         # ModuleNotFoundError included
            raise unittest.SkipTest("the model stack is not importable here (%s): run where fair-esm and torch are installed" % e)
        cls.torch, cls.MU = torch, MU
        cls.model = tiny_model()
        rng = np.random.default_rng(0)
        cls.coords = {c: (3.0 * rng.standard_normal((n, 3, 3))).astype(np.float32) for c, n in (("A", 7), ("B", 5), ("C", 6))}   # a three-chain complex, backbone atoms N, CA, C

    def test_b1_is_sample_sequence_in_complex_under_one_seed(self):
        from esm_if1_opt import batched
        for chain in ("A", "B", "C"):
            for seed in (3, 11):
                self.torch.manual_seed(seed)
                up = self.MU.sample_sequence_in_complex(self.model, self.coords, chain, temperature=1.0)
                self.torch.manual_seed(seed)
                with self.torch.inference_mode():
                    kit = batched.sample_complex_batch(self.model, [self.coords], chain, 1.0, None)
                self.assertEqual(kit, [up], (chain, seed))
                self.assertEqual(len(up), len(self.coords[chain]))

    def test_single_chain_b1_is_model_sample_under_one_seed(self):
        from esm_if1_opt import batched
        self.torch.manual_seed(5)
        up = self.model.sample(self.coords["A"], temperature=1.0)
        self.torch.manual_seed(5)
        with self.torch.inference_mode():
            kit = batched.sample_batch(self.model, [self.coords["A"]], 1.0, None)
        self.assertEqual(kit, [up])

    def test_rows_at_vanishing_temperature_are_upstreams_call(self):
        from esm_if1_opt import batched
        up = self.MU.sample_sequence_in_complex(self.model, self.coords, "B", temperature=1e-6)
        with self.torch.inference_mode():
            rows = batched.sample_complex_batch(self.model, [self.coords] * 3, "B", 1e-6, None)
        self.assertEqual(rows, [up] * 3)

    def test_rows_of_one_batch_share_the_pattern(self):
        from esm_if1_opt import batched
        other = dict(self.coords, B=self.coords["B"][:4])                # a second complex whose designed chain is shorter: rows of unequal length are refused by name, never mis-batched
        with self.assertRaises(ValueError):
            batched.sample_complex_batch(self.model, [self.coords, other], "B", 1.0, None)


if __name__ == "__main__":
    unittest.main()
