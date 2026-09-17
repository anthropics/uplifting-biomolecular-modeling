"""The low-memory transform of the base variants' exact executable (lowmem.py, stage.lowmem_worker_source): the carried worker transforms —
every anchor exactly once, the two dense decoding-order masks gone from its code, lowmem.py inlined whole after the utils import, the lever
evidence joined to the end-of-run record, the result compiles; upstream's ProteinMPNN.forward / .sample at the stock pin transform (the
statements lowmem.install recompiles); a changed worker is refused by name. The mask arithmetic against the dense form when torch is importable
(skipped otherwise: the package tests carry no torch), and the whole install — several distance slabs — against upstream's dense module on CPU:
E_idx, E, log-probs and a sampled batch equal (torch.equal). No GPU."""
import importlib.util
import os
import unittest

from proteinmpnn_opt import lowmem, modes, stack, stage

KIT = stack.kit_home()
WORKER = os.path.join(KIT, modes.WORKER_DIR, modes.WORKER)
STOCK_UTILS = os.path.join(stack.tree_home(), "stock", "src", "protein_mpnn_utils.py")


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


class TestWorkerTransform(unittest.TestCase):
    def setUp(self):
        self.src = _read(WORKER)
        self.out = stage.lowmem_worker_source(self.src)

    def test_anchors_once_in_the_carried_worker(self):
        for anchor in (stage.WORKER_UTILS_IMPORT, stage.WORKER_RECORD_TAIL):
            self.assertEqual(self.src.count(anchor), 1, anchor[:50])
        for stmts in (lowmem.DENSE_MASK_UPSTREAM, lowmem.DENSE_MASK_WORKER):
            lowmem.replace_dense_mask(self.src, stmts, expect=1)                 # raises unless the block occurs exactly once

    def test_dense_masks_replaced_and_lowmem_inlined(self):
        code = self.out.split("# ---- end of the inlined lowmem.py ----", 1)[1]  # the worker's own code after the inlined module
        for stmts in (lowmem.DENSE_MASK_UPSTREAM, lowmem.DENSE_MASK_WORKER):
            self.assertNotIn(stmts[1], code)                                     # no dense einsum left in the executable's code
        self.assertEqual(code.count(lowmem.MASK_CALL), 2)                        # forward_scores and sample
        self.assertIn(_read(lowmem.__file__).rstrip("\n"), self.out)             # lowmem.py inlined whole, byte for byte
        self.assertIn("LOWMEM_SITES = install(_lm_utils)", self.out)
        self.assertIn('"lowmem": True, "lowmem_sites": LOWMEM_SITES', self.out)  # the evidence kit_run.lever_evidence reads
        self.assertNotIn("from __future__", _read(lowmem.__file__))              # inlinable mid-file
        compile(self.out, stage.LOWMEM_WORKER, "exec")

    def test_a_changed_worker_is_refused(self):
        with self.assertRaises(ValueError):
            stage.lowmem_worker_source(self.src.replace(lowmem.DENSE_MASK_WORKER[1], lowmem.DENSE_MASK_WORKER[1] + " "*0 + "#"))
        with self.assertRaises(ValueError):
            stage.lowmem_worker_source(self.src.replace(stage.WORKER_RECORD_TAIL, ""))
        with self.assertRaises(ValueError):
            lowmem.replace_dense_mask(self.src + self.src, lowmem.DENSE_MASK_WORKER, expect=1)

    def test_upstream_forward_and_sample_transform(self):
        """The statements lowmem.install recompiles exist exactly once each in upstream's ProteinMPNN.forward and .sample at the stock pin
        (stock/src/protein_mpnn_utils.py is the tree's copy of the pinned file)."""
        utils = _read(STOCK_UTILS)
        body = utils.split("class ProteinMPNN", 1)[1]
        fwd = body.split("    def forward", 1)[1].split("\n    def ", 1)[0]
        smp = body.split("    def sample", 1)[1].split("\n    def ", 1)[0]
        for part in (fwd, smp):
            out = lowmem.replace_dense_mask(part, lowmem.DENSE_MASK_UPSTREAM, expect=1)
            self.assertIn(lowmem.MASK_CALL, out)
            self.assertNotIn(lowmem.DENSE_MASK_UPSTREAM[1], out)

    def test_sites_named(self):
        self.assertEqual(tuple(lowmem.SITES), ("ProteinFeatures._dist", "ProteinFeatures._get_rbf", "ProteinFeatures.forward", "ProteinMPNN.forward", "ProteinMPNN.sample"))
        self.assertEqual(modes.KIT_MODES["exact"].get("transform"), modes.LOWMEM)


class TestMaskArithmetic(unittest.TestCase):
    def test_mask_attend_equals_the_dense_form(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not importable: the mask arithmetic is exercised on the GPU")
        g = torch.Generator().manual_seed(3)
        for B, L, K in ((1, 7, 3), (3, 50, 48), (2, 64, 17)):
            decoding_order = torch.stack([torch.randperm(L, generator=g) for _ in range(B)])
            E_idx = torch.randint(0, L, (B, L, min(K, L)), generator=g)
            mask_size = L
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size, mask_size))), permutation_matrix_reverse, permutation_matrix_reverse)
            dense = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            self.assertTrue(torch.equal(lowmem.mask_attend(decoding_order, E_idx), dense), (B, L, K))


class TestInstallAgainstDense(unittest.TestCase):
    """lowmem.install on a second import of upstream's protein_mpnn_utils (stock/src, the pinned file) with a slab size that forces several
    slabs; the features (E_idx, E), ProteinMPNN.forward log-probs and ProteinMPNN.sample (S, probs, decoding order) equal the dense module's,
    value for value, on the same weights, inputs and seed."""

    @staticmethod
    def _utils(tag):
        spec = importlib.util.spec_from_file_location(f"protein_mpnn_utils_{tag}", STOCK_UTILS)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    def test_multi_slab_install_equals_dense(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not importable: the install is exercised on the GPU")
        dense_u, lm_u = self._utils("dense"), self._utils("lowmem")
        B, L, k = 2, 37, 8
        chunk = 5 * B * L                                                        # 5 rows per slab -> 8 slabs, the last one partial
        self.assertEqual(lowmem.install(lm_u, chunk_elems=chunk), list(lowmem.SITES))
        self.assertLess(max(1, int(chunk // (B * L))), L)                        # several slabs by construction
        g = torch.Generator().manual_seed(11)
        kw = dict(num_letters=21, node_features=32, edge_features=32, hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1,
                  vocab=21, k_neighbors=k, augment_eps=0.0, dropout=0.0)
        torch.manual_seed(1); dense = dense_u.ProteinMPNN(**kw).eval()
        lm = lm_u.ProteinMPNN(**kw).eval(); lm.load_state_dict(dense.state_dict())
        X = torch.randn(B, L, 4, 3, generator=g) * 1.5 + torch.arange(L, dtype=torch.float32)[None, :, None, None] * torch.tensor([3.8, 0., 0.])
        mask = torch.ones(B, L); mask[1, -3:] = 0.                               # a padded tail: the masked-distance adjustment runs
        S = torch.randint(0, 21, (B, L), generator=g)
        chain_M = torch.ones(B, L); chain_M_pos = torch.ones(B, L)
        residue_idx = torch.arange(L)[None, :].repeat(B, 1); residue_idx[:, 20:] += 100
        chain_encoding_all = torch.ones(B, L, dtype=torch.long); chain_encoding_all[:, 20:] = 2
        randn = torch.randn(B, L, generator=g)
        with torch.no_grad():
            E_d, Ei_d = dense.features(X, mask, residue_idx, chain_encoding_all)
            E_l, Ei_l = lm.features(X, mask, residue_idx, chain_encoding_all)
            self.assertTrue(torch.equal(Ei_d, Ei_l)); self.assertTrue(torch.equal(E_d, E_l))
            lp_d = dense(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn)
            lp_l = lm(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn)
            self.assertTrue(torch.equal(lp_d, lp_l))
            zeros21 = torch.zeros(B, L, 21)
            skw = dict(mask=mask, temperature=0.1, omit_AAs_np=(torch.arange(21) == 20).numpy().astype("float32"), bias_AAs_np=torch.zeros(21).numpy(),
                       chain_M_pos=chain_M_pos, omit_AA_mask=zeros21, pssm_coef=torch.zeros(B, L), pssm_bias=zeros21, pssm_multi=0.0,
                       pssm_log_odds_flag=False, pssm_log_odds_mask=torch.ones(B, L, 21), pssm_bias_flag=False, bias_by_res=zeros21)
            torch.manual_seed(5); sd = dense.sample(X, randn, S, chain_M, chain_encoding_all, residue_idx, **skw)
            torch.manual_seed(5); sl = lm.sample(X, randn, S, chain_M, chain_encoding_all, residue_idx, **skw)
            for key in ("S", "probs", "decoding_order"):
                self.assertTrue(torch.equal(sd[key], sl[key]), key)


if __name__ == "__main__":
    unittest.main()
