"""Templated inputs under ``big --n_gpu P``: the template embedder's nine pair INPUTS are born per row block on each rank from per-token
precursors (rowpair_xfold ``_template_precursors`` + ``_construct_input_rows``) — no rank forms an ``[N, N, F]`` template feature. This test
holds every one of the nine inputs of the row statement BITWISE EQUAL (``torch.equal``) to xfold's own dense statement
(``SingleTemplateEmbedding.construct_input``) sliced to the same rows, for a REAL template slot (atoms present, two chains: the same-chain
restriction bites) and for an EMPTY slot (the OpenFold3-weights GAP one-hots), over every rank's rows of the kit's P = 2 and P = 4 layouts of a ragged N
and arbitrary blocks, in fp32 and under a bf16 pair dtype. The nine ``template_pair_embedding_i`` Linears are replaced by recorders on
a stand-in embedder object, so the operands are compared exactly as each Linear receives them (after every mask product and dtype cast);
the kit's statements run unmodified. Needs torch + einops + triton importable (xfold's package imports them; everything runs on CPU) and
opt_core — skipped otherwise, like tests/test_tp_xfold_cpu.py. Runs under pytest or plain ``python -m unittest``."""
import importlib.util
import os
import sys
import unittest

try:
    import torch
    import einops  # noqa: F401
    import triton
    _MISSING = None
except ImportError as _e:                                 # the package's other tests import none of these
    torch = None; _MISSING = str(_e)

if torch is not None and not torch.cuda.is_available():   # xfold.fastnn's module-level @triton.autotune probes a driver at import; its torch routes are all this test touches
    triton.autotune = lambda *a, **kw: (lambda fn: fn)
    triton.heuristics = lambda *a, **kw: (lambda fn: fn)

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.join(os.path.dirname(OPT), "opt", "forward", "af3t", "af3_torch")

N_TOK = 70                                                # ragged on purpose: 70 rows over 2 / 4 ranks
FEATURES = ("dgram", "pseudo_beta_mask_2d", "aatype_col", "aatype_row", "unit_x", "unit_y", "unit_z", "backbone_mask_2d", "query_embedding")


def _rpx():
    if "rowpair_xfold" in sys.modules:
        return sys.modules["rowpair_xfold"]
    spec = importlib.util.spec_from_file_location("rowpair_xfold", os.path.join(OPT, "af3_torch_opt", "rowpair_xfold.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["rowpair_xfold"] = m; spec.loader.exec_module(m)
    return m


class _Recorder:
    """Stands in for one ``template_pair_embedding_i`` Linear: records its operand, contributes zeros."""
    def __init__(self, log, i):
        self.log, self.i = log, i

    def __call__(self, x):
        self.log[self.i] = x.detach().clone()
        return x.new_zeros(x.shape[:-1] + (1,))


class _Embedder:
    """The attributes ``construct_input`` / ``_construct_input_rows`` read of a SingleTemplateEmbedding: the distogram config, the query
    LayerNorm (identity here: the operand under test is the pair rows themselves) and the nine Linears (recorders)."""
    def __init__(self):
        from xfold.nn.template import DistogramFeaturesConfig
        self.dgram_features_config = DistogramFeaturesConfig()
        self.query_embedding_norm = lambda x: x
        self.log = {}
        for i in range(9):
            self.__dict__[f"template_pair_embedding_{i}"] = _Recorder(self.log, i)

    def __getattr__(self, name):                          # construct_input spells `self.__getattr__(f'template_pair_embedding_{i}')` (nn.Module's accessor)
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def take(self):
        out = [self.log[i] for i in range(9)]
        self.log.clear()
        return out


def _slot(real, gen):
    """One template slot's per-token features (xfold.features.Templates fields for ONE slot): a real template (atoms present at random,
    backbone mostly resolved, coordinates ~N(0, 8 Å)) or an empty one (the featuriser's zero padding)."""
    from xfold import features
    if real:
        aatype = torch.randint(0, 21, (N_TOK,), generator=gen, dtype=torch.int32)
        mask = (torch.rand(N_TOK, 24, generator=gen) < 0.7).to(torch.float32)
        mask[:, :3] = (torch.rand(N_TOK, 3, generator=gen) < 0.95).to(torch.float32)      # N, CA, C: the backbone frame atoms
        mask[5:9] = 0.0                                                                   # a gap: unresolved template residues
        pos = torch.randn(N_TOK, 24, 3, generator=gen) * 8.0
    else:
        aatype = torch.zeros(N_TOK, dtype=torch.int32); mask = torch.zeros(N_TOK, 24); pos = torch.zeros(N_TOK, 24, 3)
    return features.Templates(aatype=aatype, atom_positions=pos, atom_mask=mask)


def _clone(t):
    from xfold import features
    return features.Templates(aatype=t.aatype.clone(), atom_positions=t.atom_positions.clone(), atom_mask=t.atom_mask.clone())


def _row_ranges(rpx):
    """[(g0, g1)]: every rank's rows under the kit's own layouts of this N (``Layout(N, P, rank, align=rowpair_xfold.ALIGN)`` for P = 2 and
    P = 4; a rank that owns no rows at this small N computes nothing and is skipped), plus arbitrary blocks the layouts do not produce — the
    row statement's equality to the dense statement sliced is range-agnostic, so both kinds must hold."""
    from opt_core.mem.rowpair.dist import Layout
    out = []
    for P in (2, 4):
        for rank in range(P):
            lay = Layout(N_TOK, P, rank, align=rpx.ALIGN)
            if int(lay.R) > 0:
                out.append((int(lay.r0), int(lay.r1)))
    assert sorted(out)[0][0] == 0 and max(g1 for _, g1 in out) == N_TOK, out
    return out + [(17, 40), (0, 1), (35, N_TOK), (N_TOK - 1, N_TOK)]


def _dense_inputs(ste, z, templates_t, multichain):
    from xfold.nn.template import SingleTemplateEmbedding
    SingleTemplateEmbedding.construct_input(ste, z, _clone(templates_t), multichain)
    return ste.take()


def _row_inputs(rpx, ste, z, templates_t, asym_id, g0, g1):
    pre = rpx._template_precursors(ste, _clone(templates_t), z.dtype)
    rpx._construct_input_rows(ste, z[g0:g1], pre, asym_id, g0, g1, z.dtype)
    return ste.take()


class TemplateInputRowsEqualDenseSliced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest(f"torch / einops / triton not importable here ({_MISSING})")
        try:
            import opt_core.mem.rowpair  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"opt_core.mem.rowpair not importable here ({e})")
        if KIT not in sys.path:
            sys.path.insert(0, KIT)
        torch.set_num_threads(1)
        import af3_torch_api  # noqa: F401  (xfold.of3.OF3 = True before anything is built, as the model process does: the GAP one-hots of an empty slot)
        from xfold import of3
        assert of3.OF3
        cls.rpx = _rpx()
        if "C" not in cls.rpx.STATE:
            cls.rpx.STATE["C"] = cls.rpx._core()
        gen = torch.Generator().manual_seed(20260907)
        cls.asym_id = torch.cat([torch.ones(30, dtype=torch.int32), 2 * torch.ones(N_TOK - 30, dtype=torch.int32)])
        cls.multichain = (cls.asym_id[:, None] == cls.asym_id[None, :]).to(torch.float32)
        cls.z32 = torch.randn(N_TOK, N_TOK, 128, generator=gen)
        cls.slots = {"real": _slot(True, gen), "empty": _slot(False, gen)}

    def test_rows_equal_dense_sliced(self):
        for slot in ("real", "empty"):
            for pair_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(slot=slot, pair_dtype=str(pair_dtype)):
                    ste = _Embedder()
                    z = self.z32.to(pair_dtype)
                    t = self.slots[slot]
                    dense = _dense_inputs(ste, z, t, self.multichain.to(pair_dtype))
                    self.assertEqual(tuple(dense[0].shape[:2]), (N_TOK, N_TOK))
                    for g0, g1 in _row_ranges(self.rpx):
                        rows = _row_inputs(self.rpx, ste, z, t, self.asym_id, g0, g1)
                        for i, name in enumerate(FEATURES):
                            d, r = dense[i], rows[i]
                            d_rows = torch.broadcast_to(d, (N_TOK, N_TOK, d.shape[-1]))[g0:g1]       # aatype_col is [1, N, 31] on both sides: broadcast, then slice
                            r_rows = torch.broadcast_to(r, (g1 - g0, N_TOK, r.shape[-1]))
                            self.assertEqual(r.dtype, d.dtype, name)
                            self.assertTrue(torch.equal(r_rows, d_rows),
                                            f"{name}: rows [{g0},{g1}) of the {slot} slot differ from the dense statement sliced ({pair_dtype})")
                    if slot == "real":                                                            # the test bites: the real slot's features are not trivially zero
                        self.assertGreater(float(dense[0].abs().sum()), 0.0)
                        self.assertGreater(float(dense[4].abs().sum()), 0.0)
                        self.assertTrue(0 < float(dense[7].sum()) < N_TOK * N_TOK)
                    else:                                                                         # an empty slot under the OpenFold3 weights: GAP one-hots (index 21), no geometry
                        self.assertEqual(float(dense[0].abs().sum()), 0.0)
                        self.assertEqual(sorted(int(v) for v in dense[2].float().argmax(-1).unique()), [21])

    def test_slot_groups_evaluate_identical_slots_once(self):
        """The core's slot grouping the row driver uses (template_slot_groups on the three per-token key tensors): identical empty slots form
        ONE group (evaluated once, added once per member in stock order), a real slot is its own group."""
        from xfold import features
        r, e = self.slots["real"], self.slots["empty"]
        T = features.Templates(aatype=torch.stack([e.aatype, r.aatype, e.aatype, e.aatype]),
                               atom_positions=torch.stack([e.atom_positions, r.atom_positions, e.atom_positions, e.atom_positions]),
                               atom_mask=torch.stack([e.atom_mask, r.atom_mask, e.atom_mask, e.atom_mask]))
        groups = self.rpx.STATE["C"]["TP"].template_slot_groups([T.aatype, T.atom_positions, T.atom_mask], 4, None)
        self.assertEqual([(int(rep), [int(m) for m in members]) for rep, members in groups], [(0, [0, 2, 3]), (1, [1])])


if __name__ == "__main__":
    unittest.main()
