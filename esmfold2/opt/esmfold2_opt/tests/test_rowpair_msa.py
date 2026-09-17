"""The full_msa MSA encoder on pair ROWS (esmfold2_opt.rowpair_msa over opt_core.mem.rowpair.msa): the install contract on a stub model (one
PATCHES record, the census facts — msa=rows, m replicated NAMED, zero pair gathers — refusals by name), and — with torch + the transformers
fork of stock/PINS.json importable — the bound OuterProductMean / MSAPairWeightedAveraging / MSAEncoder statements on row shards against the
DENSE stock modules under P in {2, 3} rank-threads on CPU (opt_core.testing.run_ranks; fp32: max|diff| <= 1e-5 asserted, torch.equal
reported). The pair block of each MSA block is the kit's binding in production (esmfold2_opt.rowpair.pair_block_rows_); here a TEST-ONLY
oracle stands in for it (gather rows -> the block's three stock pair statements on the whole pair -> this rank's rows), so these numbers
isolate this module's statements from the triangle-op schedules (the core's own tests cover those)."""
import os
import threading
import unittest
from unittest import mock

try:
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import msa as _core_msa  # noqa: F401 — 0.4.3: the MSA row statements this module binds
    from opt_core.mem.rowpair.dist import Layout
    from opt_core.testing import run_ranks
except ImportError as _e:                                                   # an older / absent core: named skip
    raise unittest.SkipTest(f"producer_missing:opt_core.mem.rowpair.msa ({_e}): the MSA-rows tests need opt_core >= 0.4.3")
from esmfold2_opt import rowpair
from esmfold2_opt import rowpair_msa as RM

SMALL = dict(d_msa=8, d_pair=16, d_inputs=12, d_hidden=4, n_layers=3, n_heads_msa=2, msa_head_width=4)   # MSAEncoder(**SMALL): 3 blocks, last final
N, M, ZB = 48, 6, 16                                                        # tokens, MSA depth, layout grid block (48 = 3 blocks: P=2 -> 32+16 rows, P=3 -> 16 each)
TOL = 1e-5


def _torch_or_skip():
    try:
        import torch
    except (ImportError, RuntimeError) as e:                                  # RuntimeError: torch's double-import condition on some CPU images
    #   ('_has_torch_function already has a docstring') = torch unusable in this interpreter: named skip, as the kit's kernel tests do
        raise unittest.SkipTest(f"torch missing ({e})")
    torch.set_num_threads(1)
    return torch


def _stock_or_skip():
    try:
        import transformers.models.esmfold2.modeling_esmfold2 as MOD
    except Exception as e:  # noqa: BLE001 — the fork or one of its imports is absent: named skip
        raise unittest.SkipTest(f"transformers fork (stock/PINS.json) not importable ({type(e).__name__}: {e})")
    return MOD


class _PT(object):
    """A stub PairTransition: the two members install_msa_rows / pair_ops read (modeling_esmfold2.PairTransition: set_chunk_size, _chunk_size)."""

    def __init__(self, chunk=None):
        self._chunk_size = chunk

    def set_chunk_size(self, chunk_size):
        self._chunk_size = chunk_size

    def __call__(self, pair):
        return 0.0 * pair


class _Blk(object):
    """A stub MSAEncoderBlock: the pair statements pair_ops calls (zero deltas) and the transition module the install chunks."""

    def __init__(self, chunk=None):
        self.pair_transition = _PT(chunk)

    def tri_mul_out(self, pair, mask=None):
        return 0.0 * pair

    def tri_mul_in(self, pair, mask=None):
        return 0.0 * pair


class _Enc(object):
    """A stub msa_encoder: what install_msa_rows reads (blocks, forward, training)."""

    def __init__(self, n=4, blocks=None):
        self.blocks = blocks if blocks is not None else [_Blk() for _ in range(n)]
        self.calls = []

    def forward(self, *a, **k):                                              # the 'stock' forward the install supersedes
        self.calls.append((a, k))
        return "stock"


class _Model(object):
    def __init__(self, enc=True):
        self.msa_encoder = _Enc() if enc else None
        self.training = False


def _stub_pair_block(ops, z_rows, mask_rows, layout):
    return z_rows


class InstallContractTests(unittest.TestCase):
    def tearDown(self):
        rowpair.uninstall()

    def test_census_facts_name_the_replicated_tensors_and_zero_gathers(self):
        f = RM.census_facts()
        self.assertEqual(f["msa"], "rows")
        self.assertEqual(f["msa_m"], "replicated")
        self.assertEqual(f["msa_transition"], "replicated")
        self.assertEqual(f["msa_z_gathers"], 0)
        self.assertEqual(f["msa_p1_levers_replaced"], "t12,t13,m1")
        self.assertEqual(f["msa_p1_levers_kept"], "t11")
        self.assertEqual(set(RM.REPLICATED), {"msa_m", "msa_features", "msa_transition"})
        self.assertEqual(RM.PAIR_BLOCK_NAME, "pair_block_rows_")
        self.assertEqual(RM.TRIMUL_DISPATCH, "TriangleMultiplicativeBlock.forward")
        self.assertEqual(f["msa_trimul"], "reference_rows")

    def test_install_writes_census_and_one_patch_record(self):
        model, census = _Model(), {}
        stock_forward = model.msa_encoder.forward
        facts = RM.install_msa_rows(model, None, census, pair_block=_stub_pair_block)
        for k in ("msa", "msa_m", "msa_features", "msa_transition", "msa_z_gathers", "msa_blocks", "msa_pair_block", "msa_graphs_off", "msa_replicated"):
            self.assertIn(k, census)
            self.assertEqual(census[k], facts[k])
        self.assertEqual((census["msa"], census["msa_m"], census["msa_z_gathers"], census["msa_blocks"]), ("rows", "replicated", 0, 4))
        self.assertEqual(census["msa_pair_block"], "_stub_pair_block")
        self.assertEqual(census["msa_pair_transition"], f"rows:{RM.MSA_PAIR_TRANSITION_ROWS_DEFAULT}x4")   # every block's PairTransition chunked by rows (default 256)
        self.assertEqual([b.pair_transition._chunk_size for b in model.msa_encoder.blocks], [RM.MSA_PAIR_TRANSITION_ROWS_DEFAULT] * 4)
        self.assertEqual(model.msa_encoder._ef2_rowpair_msa_pt_rows, RM.MSA_PAIR_TRANSITION_ROWS_DEFAULT)
        self.assertIn("model.msa_encoder.forward", rowpair.PATCHES.names())
        self.assertIsNot(model.msa_encoder.forward.__func__, _Enc.forward)          # the rows forward shadows the stock one
        self.assertTrue(model.msa_encoder._ef2_rowpair_msa)
        with self.assertRaises(RowpairRefused):                            # one install per rank process
            RM.install_msa_rows(model, None, census, pair_block=_stub_pair_block)
        from opt_core.mem.rowpair import evidence as EV
        self.assertEqual(EV.schedule().get("msa_m"), "replicated")        # the schedule census carries the same facts
        names = rowpair.uninstall()                                        # the lever's uninstall restores the stock forward
        self.assertIn("model.msa_encoder.forward", names)
        self.assertEqual(model.msa_encoder.forward(1), "stock")
        self.assertTrue(RM.uninstall_msa_rows(model))
        self.assertEqual([b.pair_transition._chunk_size for b in model.msa_encoder.blocks], [None] * 4)   # the blocks' chunk sizes as they were before the install
        self.assertFalse(hasattr(model.msa_encoder, "_ef2_rowpair_msa_pt_rows"))
        self.assertFalse(RM.uninstall_msa_rows(model))

    def test_pair_transition_rows_env_and_refusals(self):
        """EF2_ROWPAIR_MSA_TRANSITION_ROWS wins over EF2_ROWPAIR_TRANSITION_ROWS wins over the MSA blocks' default (64); 0 = the model's chunk as loaded (install
        touches no block, census says `whole`); negative / non-integer values, and a block without a PairTransition, are refused BY NAME
        before anything is bound."""
        E, F = RM.ENV_PAIR_TRANSITION_ROWS, rowpair.ENV_TRANSITION_ROWS
        self.assertEqual(RM.pair_transition_rows({}), RM.MSA_PAIR_TRANSITION_ROWS_DEFAULT)
        self.assertEqual(RM.pair_transition_rows({F: "128"}), 128)
        self.assertEqual(RM.pair_transition_rows({E: "64", F: "128"}), 64)
        self.assertEqual(RM.pair_transition_rows({E: "0", F: "128"}), 0)
        self.assertEqual(RM.pair_transition_rows({E: " ", F: "32"}), 32)                  # blank = unset
        for bad in ("-1", "abc", "1.5"):
            with self.assertRaisesRegex(RowpairRefused, "non-negative integer"):
                RM.pair_transition_rows({E: bad})
        with mock.patch.dict(os.environ, {E: "0"}):                                       # opt-out by name: nothing set, fact `whole`
            model, census = _Model(), {}
            model.msa_encoder.blocks[1].pair_transition.set_chunk_size(64)
            RM.install_msa_rows(model, None, census, pair_block=_stub_pair_block)
            self.assertEqual(census["msa_pair_transition"], "whole")
            self.assertEqual([b.pair_transition._chunk_size for b in model.msa_encoder.blocks], [None, 64, None, None])
            self.assertEqual(model.msa_encoder._ef2_rowpair_msa_pt_rows, 0)
            rowpair.uninstall(); RM.uninstall_msa_rows(model)
            self.assertEqual([b.pair_transition._chunk_size for b in model.msa_encoder.blocks], [None, 64, None, None])
        with mock.patch.dict(os.environ, {E: "32"}):
            model, census = _Model(), {}
            RM.install_msa_rows(model, None, census, pair_block=_stub_pair_block)
            self.assertEqual(census["msa_pair_transition"], "rows:32x4")
            rowpair.uninstall(); RM.uninstall_msa_rows(model)
            bare = _Model()                                                                # a block whose pair_transition lacks set_chunk_size -> refused, nothing bound
            bare.msa_encoder = _Enc(blocks=[_Blk(), object(), _Blk()])
            with self.assertRaisesRegex(RowpairRefused, r"blocks\[1\] has no pair_transition"):
                RM.install_msa_rows(bare, None, {}, pair_block=_stub_pair_block)
            self.assertNotIn("model.msa_encoder.forward", rowpair.PATCHES.names())
            self.assertFalse(getattr(bare.msa_encoder, "_ef2_rowpair_msa", False))
        with mock.patch.dict(os.environ, {E: "-2"}):
            with self.assertRaisesRegex(RowpairRefused, "non-negative integer"):
                RM.install_msa_rows(_Model(), None, {}, pair_block=_stub_pair_block)
            self.assertNotIn("model.msa_encoder.forward", rowpair.PATCHES.names())

    def test_pair_ops_refuse_a_chunk_changed_after_install(self):
        """pair_ops(block, pt_rows): the transition statement runs only while the installed row chunk is still on the module; a later
        set_chunk_size (the model-level call cascades into the MSA blocks) is refused BY NAME. pt_rows=0: no check (not installed)."""
        blk = _Blk(chunk=256)
        self.assertEqual(RM.pair_ops(blk, 256)(1.0), 1.0)                                   # zero-delta stubs: pair + 0 + 0 + 0
        self.assertEqual(RM.pair_ops(blk)(1.0), 1.0)
        blk.pair_transition.set_chunk_size(None)                                            # e.g. a model-level set_chunk_size_chunk_size(None) after the install
        with self.assertRaisesRegex(RowpairRefused, "set_chunk_size call after the install"):
            RM.pair_ops(blk, 256)(1.0)
        self.assertEqual(RM.pair_ops(blk, 0)(1.0), 1.0)
        blk.pair_transition.set_chunk_size(64)
        with self.assertRaisesRegex(RowpairRefused, "rows:256"):
            RM.pair_ops(blk, 256)(1.0)

    def test_install_refusals_by_name(self):
        with self.assertRaisesRegex(RowpairRefused, "no msa_encoder"):
            RM.install_msa_rows(_Model(enc=False), None, {}, pair_block=_stub_pair_block)
        m = _Model()
        m.training = True
        with self.assertRaisesRegex(RowpairRefused, "train"):
            RM.install_msa_rows(m, None, {}, pair_block=_stub_pair_block)
        with mock.patch.object(rowpair, RM.PAIR_BLOCK_NAME, None, create=True):   # the kit's binding absent -> refused naming it (never a dense trimul on a shard)
            with self.assertRaisesRegex(RowpairRefused, RM.PAIR_BLOCK_NAME):
                RM.install_msa_rows(_Model(), None, {})
        with mock.patch.object(rowpair, RM.PAIR_BLOCK_NAME, _stub_pair_block, create=True):   # present -> resolved by name
            self.assertIs(RM.resolve_pair_block(), _stub_pair_block)

    def test_forward_refuses_whole_pair_b2_layout_mismatch_and_p1(self):
        torch = _torch_or_skip()
        feats = dict(x_inputs=torch.zeros(1, 32, 12), msa_oh=torch.zeros(1, 32, M, 33), has_deletion=torch.zeros(1, 32, M), deletion_value=torch.zeros(1, 32, M),
                     msa_attention_mask=torch.ones(1, 32, M))
        model = _Model()
        RM.install_msa_rows(model, lambda: Layout.checked(32, 2, 0, 16), {}, pair_block=_stub_pair_block)
        fwd = model.msa_encoder.forward
        m2 = _Model()
        RM.install_msa_rows(m2, None, {}, pair_block=_stub_pair_block)              # layout=None: read from rowpair.CTX per call

        def refusals(r, P):                                                          # inside a rank thread (a comm exists); every call refuses before any collective
            out = {}
            for key, call in (("whole", lambda: fwd(torch.zeros(1, 32, 32, 16), **feats)),
                              ("b2", lambda: fwd(torch.zeros(2, 16, 32, 16), **feats)),
                              ("nmis", lambda: fwd(torch.zeros(1, 16, 40, 16), **feats)),
                              ("p1", lambda: RM.msa_encoder_rows(model.msa_encoder, torch.zeros(1, 32, 32, 16), layout=Layout(32, 1, 0), pair_block=_stub_pair_block, **feats)),
                              ("nolayout", lambda: m2.msa_encoder.forward(torch.zeros(1, 16, 32, 16), **feats))):
                try:
                    call()
                    out[key] = None
                except RowpairRefused as e:
                    out[key] = str(e)
            return out
        ctx = getattr(rowpair, "CTX", None)
        with (mock.patch.object(ctx, "layout", None) if ctx is not None and hasattr(ctx, "layout") else mock.MagicMock()):
            res = run_ranks(2, refusals)
        for out in res:
            self.assertRegex(out["whole"] or "", "WHOLE pair")
            self.assertRegex(out["b2"] or "", r"B == 1")
            self.assertRegex(out["nmis"] or "", "N=32 but the pair rows have N=40")
            self.assertIsNotNone(out["p1"])                                          # a P == 1 layout never reaches the row statements
            self.assertRegex(out["nolayout"] or "", "no Layout")


# ================================================================================================================ numerics vs the dense stock modules
def _fixture(torch, MOD, chunk=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    enc = MOD.MSAEncoder(**SMALL).eval()
    with torch.no_grad():
        for name, p in enc.named_parameters():                              # every Linear / LayerNorm affine randomised, so no statement is inert
            if p.dim() > 1:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith("weight"):                                   # LayerNorm scale near 1
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    enc.set_chunk_size(chunk)
    x_pair = torch.randn(1, N, N, SMALL["d_pair"], generator=g)
    x_inputs = torch.randn(1, N, SMALL["d_inputs"], generator=g)
    msa_oh = torch.nn.functional.one_hot(torch.randint(0, 33, (1, N, M), generator=g), 33).float()
    has_deletion = (torch.rand(1, N, M, generator=g) < 0.2).float()
    deletion_value = torch.rand(1, N, M, generator=g) * has_deletion
    msa_mask = torch.ones(1, N, M)
    msa_mask[:, N - 3:, :] = 0.0                                            # a padded token tail (pair_attention_mask rows/cols off there)
    msa_mask[:, 10:20, M - 1] = 0.0                                         # a partially absent MSA row
    feats = dict(x_inputs=x_inputs, msa_oh=msa_oh, has_deletion=has_deletion, deletion_value=deletion_value, msa_attention_mask=msa_mask)
    return enc, x_pair, feats


def _oracle_pair_block(torch):
    """TEST-ONLY stand-in for the kit's row form of a pair block (esmfold2_opt.rowpair.pair_block_rows_): the whole pair is re-assembled from the
    ranks' rows (all_gather), the block's pair statements (rowpair_msa.pair_ops: the stock sub-modules, dense here) run on it, this rank's rows
    are returned. Exact by construction; never a kit code path."""
    from opt_core.mem.rowpair.shard import unshard_rows

    def pair_block(ops, z_rows, mask_rows, lay):
        assert z_rows.dim() == 4 and mask_rows.dim() == 3, (z_rows.shape, mask_rows.shape)   # [1, R, N, C] / [1, R, N]: the kit's row-form contract
        pair = unshard_rows(z_rows[0].contiguous(), lay, dim=0).unsqueeze(0)
        vis = unshard_rows(mask_rows[0].to(torch.uint8).contiguous(), lay, dim=0).bool().unsqueeze(0)
        return ops(pair, pair_attention_mask=vis)[:, lay.r0:lay.r1].contiguous()
    return pair_block


def _cmp(torch, got, ref):
    return float((got - ref).abs().max()), bool(torch.equal(got, ref))


class NumericTests(unittest.TestCase):
    """Row statements vs the dense stock modules (fp32 CPU, P rank-threads)."""

    @classmethod
    def setUpClass(cls):
        cls.torch = _torch_or_skip()
        cls.MOD = _stock_or_skip()
        cls.report = {}

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "report", None):
            print("\nROWPAIR_MSA_NUMERICS " + " ".join(f"{k}=maxdiff:{v[0]:.3e},equal:{v[1]}" for k, v in sorted(cls.report.items())))

    def tearDown(self):
        rowpair.uninstall()

    def _record(self, key, got, ref):
        d, eq = _cmp(self.torch, got, ref)
        self.report[key] = (d, eq)
        self.assertLessEqual(d, TOL, key)

    def test_opm_rows_vs_stock_module(self):
        torch, MOD = self.torch, self.MOD
        for chunk in (None, 16):
            enc, x_pair, feats = _fixture(torch, MOD, chunk=chunk)
            opm = enc.blocks[0].outer_product_mean
            m = torch.randn(1, N, M, SMALL["d_msa"], generator=torch.Generator().manual_seed(3))
            with torch.no_grad():
                ref = x_pair + opm(m, feats["msa_attention_mask"])          # the dense statement `pair = pair + outer_product_mean(m, msa_attention_mask)`
            for P in (2, 3):
                for rows in (None, 8):                                      # budgeted row block (agreed free bytes / rowblk) and a given one
                    def rank(r, P=P, rows=rows):
                        lay = Layout.checked(N, P, r, ZB)
                        z = x_pair[0, lay.r0:lay.r1].clone()
                        with torch.no_grad():
                            z = RM.opm_add_rows_(opm, m, feats["msa_attention_mask"], z, lay, rows=rows)
                        return lay.r0, lay.r1, z
                    for r0, r1, z in run_ranks(P, rank):
                        self._record(f"opm[chunk={chunk},P={P},rows={rows},r0={r0}]", z, ref[0, r0:r1])
            x16 = x_pair.to(torch.bfloat16)                                 # a narrower shard: the dense `pair + delta` promotes -> so does the returned shard
            with torch.no_grad():
                ref16 = x16 + opm(m, feats["msa_attention_mask"])

            def rank16(r, _P=None):
                lay = Layout.checked(N, 2, r, ZB)
                with torch.no_grad():
                    z = RM.opm_add_rows_(opm, m, feats["msa_attention_mask"], x16[0, lay.r0:lay.r1].clone(), lay, rows=8)
                return lay.r0, lay.r1, z
            for r0, r1, z in run_ranks(2, rank16):
                self.assertEqual(z.dtype, ref16.dtype)
                self._record(f"opm[bf16_shard,chunk={chunk},r0={r0}]", z, ref16[0, r0:r1])

    def test_pwa_rows_vs_stock_module(self):
        torch, MOD = self.torch, self.MOD
        enc, x_pair, feats = _fixture(torch, MOD)
        pwa = enc.blocks[0].msa_pair_weighted_averaging
        m = torch.randn(1, N, M, SMALL["d_msa"], generator=torch.Generator().manual_seed(4))
        tok = feats["msa_attention_mask"][:, :, 0].bool()
        pair_mask = tok.unsqueeze(2) & tok.unsqueeze(1)
        with torch.no_grad():
            ref = pwa(m, x_pair, pair_mask)                                 # [1, N, M, d_msa]
        for P in (2, 3):
            for qb in (None, 4):                                            # whole local rows ('given' R) and a pinned query block
                def rank(r, P=P, qb=qb):
                    lay = Layout.checked(N, P, r, ZB)
                    z = x_pair[0, lay.r0:lay.r1].contiguous()
                    vis = RM.pair_mask_rows(tok, lay.r0, lay.r1)
                    with torch.no_grad():
                        return RM.pwa_delta_rows(pwa, m, z, vis, lay, q_block=qb)
                outs = run_ranks(P, rank)
                for r, delta in enumerate(outs):
                    self._record(f"pwa[P={P},qblock={qb},rank={r}]", delta, ref)
                for r in range(1, P):                                       # the m-update is REPLICATED: identical on every rank
                    self.assertTrue(torch.equal(outs[0], outs[r]), f"pwa delta differs between rank 0 and rank {r}")

    def test_msa_encoder_rows_vs_stock_forward(self):
        torch, MOD = self.torch, self.MOD
        oracle = _oracle_pair_block(torch)
        for chunk in (None, 16):
            enc, x_pair, feats = _fixture(torch, MOD, chunk=chunk, seed=1)
            with torch.no_grad():
                ref = enc(x_pair, **feats)                                  # the dense stock MSAEncoder.forward: [1, N, N, d_pair]
            self.assertFalse(torch.equal(ref, x_pair))
            for P in (2, 3):
                probes = {}

                def probe(r):
                    def f(k, tag, obj):
                        if tag in ("m0", "pwa", "mtr"):                      # replicated tensors: fingerprint per rank
                            probes.setdefault((k, tag), {})[r] = obj.detach().clone()
                    return f

                def rank(r, P=P):
                    lay = Layout.checked(N, P, r, ZB)
                    x_loc = x_pair[:, lay.r0:lay.r1].contiguous()
                    census = {}
                    with torch.no_grad():
                        out = RM.msa_encoder_rows(enc, x_loc, layout=lay, pair_block=oracle, census=census, probe=probe(r), **feats)
                    return lay, x_loc, out, census
                res = run_ranks(P, rank)
                for lay, x_loc, out, census in res:
                    self.assertEqual(tuple(out.shape), (1, lay.R, N, SMALL["d_pair"]))
                    self._record(f"encoder[chunk={chunk},P={P},rank={lay.rank}]", out, ref[:, lay.r0:lay.r1])
                    self.assertTrue(torch.equal(x_loc, x_pair[:, lay.r0:lay.r1]), "x_pair rows were modified (the loop reuses z_init rows)")
                    self.assertEqual((census["msa"], census["msa_m"], census["msa_features"], census["msa_z_gathers"], census["msa_blocks"], census["msa_depth"]),
                                     ("rows", "replicated", "guarded", 0, SMALL["n_layers"], M))
                for (k, tag), per_rank in probes.items():                    # m stays bitwise identical across ranks after every m-update
                    for r in range(1, P):
                        self.assertTrue(torch.equal(per_rank[0], per_rank[r]), f"m differs across ranks after block {k} {tag} (P={P})")

    def test_installed_forward_end_to_end(self):
        """install_msa_rows on a model holder -> the loop's own call `self.msa_encoder(x_pair=<rows>, ...)` returns msa_pair rows (P=2; the
        layout comes from a per-thread callable, as rowpair.CTX.layout does per fold in a rank process)."""
        torch, MOD = self.torch, self.MOD
        enc, x_pair, feats = _fixture(torch, MOD, chunk=16, seed=2)
        with torch.no_grad():
            ref = enc(x_pair, **feats)
        model = torch.nn.Module()
        model.msa_encoder = enc
        model.eval()
        tl = threading.local()
        census = {}
        facts = RM.install_msa_rows(model, lambda: getattr(tl, "lay", None), census, pair_block=_oracle_pair_block(torch))
        self.assertEqual(facts["msa_blocks"], SMALL["n_layers"])

        def rank(r, _P=None):
            tl.lay = Layout.checked(N, 2, r, ZB)
            with torch.no_grad():
                out = model.msa_encoder(x_pair=x_pair[:, tl.lay.r0:tl.lay.r1].contiguous(), x_inputs=feats["x_inputs"], msa_oh=feats["msa_oh"],
                                        has_deletion=feats["has_deletion"], deletion_value=feats["deletion_value"], msa_attention_mask=feats["msa_attention_mask"])
            return tl.lay.r0, tl.lay.r1, out
        for r0, r1, out in run_ranks(2, rank):
            self._record(f"installed[P=2,r0={r0}]", out, ref[:, r0:r1])
        self.assertEqual(census["msa_features"], "guarded")
        rowpair.uninstall()
        RM.uninstall_msa_rows(model)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.msa_encoder(x_pair, **feats), ref))   # the stock forward is back

    def test_installed_forward_row_chunked_pair_transition(self):
        """EF2_ROWPAIR_MSA_TRANSITION_ROWS below a rank's R: every MSA block's PairTransition walks the rows in chunks (LayerNorm + SwiGLU per
        row block, cat along dim 1) — the same numbers as the dense stock forward within TOL on every rank (P in {2, 3}), the census names
        rows:<n>x<blocks> at install AND per call, and uninstall puts the fixture's chunk (16) back so the stock forward is bitwise itself again."""
        torch, MOD = self.torch, self.MOD
        for rows in (8, 5):                                                                  # 8 | R (32+16 / 16x3); 5 leaves ragged last chunks
            enc, x_pair, feats = _fixture(torch, MOD, chunk=16, seed=8)
            with torch.no_grad():
                ref = enc(x_pair, **feats)
            for P in (2, 3):
                model = torch.nn.Module()
                model.msa_encoder = enc
                model.eval()
                tl = threading.local()
                census = {}
                with mock.patch.dict(os.environ, {RM.ENV_PAIR_TRANSITION_ROWS: str(rows)}):
                    facts = RM.install_msa_rows(model, lambda: getattr(tl, "lay", None), census, pair_block=_oracle_pair_block(torch))
                self.assertEqual(facts["msa_pair_transition"], f"rows:{rows}x{SMALL['n_layers']}")
                self.assertEqual([b.pair_transition._chunk_size for b in enc.blocks], [rows] * SMALL["n_layers"])

                def rank(r, _P=P):
                    tl.lay = Layout.checked(N, _P, r, ZB)
                    with torch.no_grad():
                        out = model.msa_encoder(x_pair=x_pair[:, tl.lay.r0:tl.lay.r1].contiguous(), **feats)
                    return tl.lay.r0, tl.lay.r1, out
                for r0, r1, out in run_ranks(P, rank):
                    self.assertLess(r1 - r0, N)
                    self._record(f"pt_rows[rows={rows},P={P},r0={r0}]", out, ref[:, r0:r1])
                self.assertEqual(census["msa_pair_transition"], f"rows:{rows}x{SMALL['n_layers']}")   # the per-call fact (written by msa_encoder_rows)
                enc.blocks[0].pair_transition.set_chunk_size(None)                          # a model-level set_chunk_size after the install -> refused per call, by name

                def rank_refused(r, _P=P):
                    tl.lay = Layout.checked(N, _P, r, ZB)
                    try:
                        with torch.no_grad():
                            model.msa_encoder(x_pair=x_pair[:, tl.lay.r0:tl.lay.r1].contiguous(), **feats)
                    except RowpairRefused as e:
                        return str(e)
                    return None
                for msg in run_ranks(P, rank_refused):
                    self.assertRegex(msg or "", "set_chunk_size call after the install")
                enc.blocks[0].pair_transition.set_chunk_size(rows)
                rowpair.uninstall()
                self.assertTrue(RM.uninstall_msa_rows(model))
                self.assertEqual([b.pair_transition._chunk_size for b in enc.blocks], [16] * SMALL["n_layers"])
                with torch.no_grad():
                    self.assertTrue(torch.equal(model.msa_encoder(x_pair, **feats), ref))   # the stock forward at the fixture's chunk: bitwise itself

    def test_dispatch_record_required_for_the_kit_row_form(self):
        """With the kit's row form resolved by name (no pair_block injected), the rows reach the MSA blocks' trimul only when the lever's install
        record carries the TriangleMultiplicativeBlock.forward row dispatcher — else refused BY NAME (never the dense contraction on a shard)."""
        torch, MOD = self.torch, self.MOD
        import transformers.models.esmfold2.modeling_esmfold2_common as CMN
        enc, x_pair, feats = _fixture(torch, MOD, chunk=16, seed=6)
        with torch.no_grad():
            ref = enc(x_pair, **feats)
        oracle = _oracle_pair_block(torch)
        tl = threading.local()
        with mock.patch.object(rowpair, RM.PAIR_BLOCK_NAME, oracle, create=True):
            model = torch.nn.Module()
            model.msa_encoder = enc
            model.eval()
            RM.install_msa_rows(model, lambda: getattr(tl, "lay", None), {})           # pair_block resolved by name -> dispatch-checked

            def rank(r, _P=None):
                tl.lay = Layout.checked(N, 2, r, ZB)
                try:
                    with torch.no_grad():
                        return model.msa_encoder(x_pair=x_pair[:, tl.lay.r0:tl.lay.r1].contiguous(), **feats)
                except RowpairRefused as e:
                    return str(e)
            outs = run_ranks(2, rank)
            self.assertTrue(all(isinstance(o, str) and RM.TRIMUL_DISPATCH in o for o in outs), outs)   # no dispatcher in PATCHES -> refused by name
            rowpair.PATCHES.replace(CMN.TriangleMultiplicativeBlock, "forward", CMN.TriangleMultiplicativeBlock.forward)   # the record now names it (no-op patch: dense oracle)
            outs = run_ranks(2, rank)
            for r, o in enumerate(outs):
                self.assertFalse(isinstance(o, str), o)
                lay = Layout.checked(N, 2, r, ZB)
                self._record(f"by_name[P=2,rank={r}]", o, ref[:, lay.r0:lay.r1])

    def test_pair_ops_are_the_block_statements(self):
        torch, MOD = self.torch, self.MOD
        enc, x_pair, feats = _fixture(torch, MOD, chunk=None, seed=7)
        blk = enc.blocks[0]
        tok = feats["msa_attention_mask"][:, :, 0].bool()
        vis = tok.unsqueeze(2) & tok.unsqueeze(1)
        with torch.no_grad():
            pair = x_pair + blk.tri_mul_out(x_pair, mask=vis)
            pair = pair + blk.tri_mul_in(pair, mask=vis)
            ref = pair + blk.pair_transition(pair)
            got = RM.pair_ops(blk)(x_pair, pair_attention_mask=vis)
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(RM.pair_ops(blk).__name__, "msa_pair_ops")

    def test_feature_guard_refuses_diverged_ranks(self):
        """MSA features that differ across ranks (RNG out of lockstep) are refused BY NAME, never folded."""
        torch, MOD = self.torch, self.MOD
        enc, x_pair, feats = _fixture(torch, MOD, seed=5)

        def rank(r, _P=None):
            lay = Layout.checked(N, 2, r, ZB)
            f = dict(feats)
            if r == 1:                                                       # rank 1 drew a different subsample (other residues in the first rows)
                oh = feats["msa_oh"].clone()
                oh[:, :7] = torch.nn.functional.one_hot((oh[:, :7].argmax(-1) + 1) % 33, 33).float()
                f["msa_oh"] = oh
            try:
                with torch.no_grad():
                    RM.msa_encoder_rows(enc, x_pair[:, lay.r0:lay.r1].contiguous(), layout=lay, pair_block=_oracle_pair_block(torch), **f)
            except RowpairRefused as e:
                return str(e)
            return None
        msgs = run_ranks(2, rank)
        self.assertTrue(all(m is not None and "msa_features.msa_oh" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
