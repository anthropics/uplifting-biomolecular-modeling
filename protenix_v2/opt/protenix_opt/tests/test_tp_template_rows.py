"""The multi-GPU line's template pair features are born per row block from per-token precursors, never as dense ``[T, N, N, F]``
tensors: ``ptx_tp.template_real`` (the carried unit) turns the stock featuriser's ``Templates`` object into per-token keys
(``featurizer_emit_real_rows``: pseudo-beta position and mask, CA position, backbone frame and mask — O(T·N)) and rebuilds any
row block ``[g0, g1) x [0, N)`` of the four pair features with the featuriser's own arithmetic (``rows_numpy`` / ``rows_torch``).

Conformance: for every slot and several row blocks (aligned, ragged last block, a single row, the whole range) the rows are
BIT-IDENTICAL to stock's dense construction ``Templates.as_protenix_dict()`` sliced to the same rows — distogram, pseudo-beta 2-D
mask, unit vector, backbone 2-D mask — on synthetic templates with partly missing atoms, fully masked residues, gap residue types
and one all-dummy slot. Needs the stock ``protenix`` package (its featuriser imports rdkit / biotite / Bio); skipped by name where
it is not installed. Runs under pytest and under ``python -m unittest protenix_opt.tests.test_tp_template_rows``.
"""
import os
import sys
import unittest

import numpy as np

from protenix_opt import tp

UNIT_DIR = tp.unit_dir()                                  # opt/forward/PTX_TP/PTX_TP_ADDON: holds the ptx_tp package
DENSE = {"dgram": "template_distogram", "pb2d": "template_pseudo_beta_mask", "uv": "template_unit_vector", "bb2d": "template_backbone_frame_mask"}
N_TOKENS, N_SLOTS, N_ATOMS = 300, 3, 24                   # 300 tokens: two full 128-row blocks and a ragged 44-row block
BLOCKS = ((0, 128), (128, 256), (256, 300), (0, 300), (37, 38), (299, 300))


def _bits(a: np.ndarray) -> np.ndarray:
    """The array's bytes as unsigned integers of the same width (bit-for-bit comparison: -0.0, NaN payloads included)."""
    a = np.ascontiguousarray(a)
    return a.view({4: np.uint32, 8: np.uint64, 1: np.uint8, 2: np.uint16}[a.dtype.itemsize])


def _synthetic_templates(rng: np.random.Generator):
    """[T, N] aatype (incl. the gap type 31), [T, N, 24, 3] positions, [T, N, 24] mask: slot 0 and 1 real with holes, slot 2 all-dummy."""
    T, N, A = N_SLOTS, N_TOKENS, N_ATOMS
    aatype = rng.integers(0, 32, size=(T, N)).astype(np.int64)
    pos = (rng.standard_normal((T, N, A, 3)) * 15.0).astype(np.float32)
    mask = rng.random((T, N, A)) > 0.15                     # most atoms present, some missing (pseudo-beta / backbone atoms among them)
    mask[:, rng.random(N) < 0.1, :] = False                 # fully unresolved template residues
    mask[1, : N // 3, :] = False                            # slot 1: a template covering only part of the query
    mask[2] = False; pos[2] = 0.0; aatype[2] = 31           # slot 2: the featuriser's dummy slot (gap type, no atoms)
    near = rng.random((T, N)) < 0.05                        # a few coincident pseudo-beta sites (zero distance: the first bin's open lower edge)
    pos[:, 1:][near[:, 1:]] = pos[:, :-1][near[:, 1:]]
    return aatype, pos, mask


class TestTemplateRowsEqualDenseSliced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from protenix.data.template.template_featurizer import Templates      # noqa: F401  the stock featuriser (rdkit / biotite / Bio behind it)
        except Exception as e:                                                       # ImportError or a missing native dependency of the stock stack
            raise unittest.SkipTest(f"the stock protenix featuriser is not importable here ({type(e).__name__}: {e}); this test runs on the kit's stack")
        if getattr(Templates.as_protenix_dict, "_ptx_tp_rows", False):
            raise unittest.SkipTest("Templates.as_protenix_dict is the multi-GPU line's rows-mode replacement in this process, not stock's dense build")
        if not os.path.isfile(os.path.join(UNIT_DIR, "ptx_tp", "template_real.py")):
            raise unittest.SkipTest(f"carried unit absent: {UNIT_DIR}")
        cls._path_added = UNIT_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, UNIT_DIR)
        from ptx_tp import template_real
        cls.real = template_real
        cls.Templates = Templates
        cls.aatype, cls.pos, cls.mask = _synthetic_templates(np.random.default_rng(20260907))
        cls.dense = Templates(aatype=cls.aatype, atom_positions=cls.pos, atom_mask=cls.mask).as_protenix_dict()
        cls.rows_dict = template_real.featurizer_emit_real_rows(Templates(aatype=cls.aatype, atom_positions=cls.pos, atom_mask=cls.mask))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and UNIT_DIR in sys.path:
            sys.path.remove(UNIT_DIR)
        getattr(cls, "real", None) and cls.real._NP_CACHE.clear()

    def test_dense_reference_has_the_stock_shapes(self):
        d, T, N = self.dense, N_SLOTS, N_TOKENS
        self.assertEqual(d["template_distogram"].shape, (T, N, N, 39)); self.assertEqual(d["template_unit_vector"].shape, (T, N, N, 3))
        self.assertEqual(d["template_pseudo_beta_mask"].shape, (T, N, N)); self.assertEqual(d["template_backbone_frame_mask"].shape, (T, N, N))
        self.assertGreater(float(d["template_distogram"][0].sum()), 0.0, "slot 0 is a real template")
        self.assertEqual(float(np.abs(d["template_distogram"][2]).sum() + np.abs(d["template_unit_vector"][2]).sum()), 0.0, "slot 2 is all-dummy")

    def test_precursors_are_per_token_only(self):
        """What travels from the featuriser to the ranks: per-token keys, no [*, N, N, *] tensor, no dense template key, both markers set."""
        d, N = self.rows_dict, N_TOKENS
        for k in DENSE.values():
            self.assertNotIn(k, d, f"{k} is a dense [T, N, N, *] key")
        for k, v in d.items():
            shape = tuple(np.shape(v))
            self.assertLessEqual(shape.count(N), 1, f"{k} has shape {shape}: more than one token axis")
        self.assertEqual(int(d[self.real.REAL_MARKER]), 1); self.assertEqual(int(d["template_pair_dense_free"]), 1)
        for k in self.real.KEYS:
            self.assertIn(k, d)
        self.assertTrue(np.array_equal(d["template_pseudo_beta_mask_1d"], d["template_pb_mask_1d"]))
        self.assertTrue(np.array_equal(d["template_backbone_frame_mask_1d"], d["template_bb_mask_1d"]))

    def test_rows_numpy_equal_dense_sliced_bitwise(self):
        for t in range(N_SLOTS):
            for g0, g1 in BLOCKS:
                got = dict(zip(("dgram", "pb2d", "uv", "bb2d"), self.real.rows_numpy(self.rows_dict, t, g0, g1)))
                for key, dense_key in DENSE.items():
                    want = self.dense[dense_key][t, g0:g1]
                    with self.subTest(slot=t, rows=(g0, g1), feature=dense_key):
                        self.assertEqual(got[key].shape, want.shape)
                        self.assertEqual(got[key].dtype, want.dtype)
                        self.assertTrue(np.array_equal(_bits(got[key]), _bits(want)), f"{dense_key} slot {t} rows [{g0},{g1}) differ from the dense features sliced")

    def test_rows_torch_equal_dense_sliced_bitwise(self):
        import torch
        d = {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) and v.dtype != object else v) for k, v in self.rows_dict.items()}  # the pipeline's tensors
        self.real._NP_CACHE.clear()
        for t in range(N_SLOTS):
            for g0, g1 in BLOCKS[:3]:
                got = dict(zip(("dgram", "pb2d", "uv", "bb2d"), self.real.rows_torch(d, t, g0, g1, torch.device("cpu"))))
                for key, dense_key in DENSE.items():
                    want = torch.from_numpy(np.ascontiguousarray(self.dense[dense_key][t, g0:g1]))
                    with self.subTest(slot=t, rows=(g0, g1), feature=dense_key):
                        self.assertEqual(got[key].dtype, torch.float32)
                        self.assertTrue(torch.equal(got[key].view(torch.int32), want.view(torch.int32)), f"{dense_key} slot {t} rows [{g0},{g1}) differ (torch)")

    def test_blocks_tile_the_dense_tensor_exactly(self):
        """Consecutive 128-row blocks concatenated == the whole dense slot (the schedule the embedder runs: BLOCK_ROWS = 128)."""
        from protenix_opt.tp_bind.template import BLOCK_ROWS
        for t in range(N_SLOTS):
            parts = [self.real.rows_numpy(self.rows_dict, t, g0, min(g0 + BLOCK_ROWS, N_TOKENS)) for g0 in range(0, N_TOKENS, BLOCK_ROWS)]
            for i, (key, dense_key) in enumerate(DENSE.items()):
                whole = np.concatenate([p[i] for p in parts], axis=0)
                with self.subTest(slot=t, feature=dense_key):
                    self.assertTrue(np.array_equal(_bits(whole), _bits(self.dense[dense_key][t])))


if __name__ == "__main__":
    unittest.main()
