"""The n_gpu>1 adapter (esmfold2_opt.rowpair over opt_core.mem.rowpair): the structural rule (NOTHING installed at n_gpu=1), the launcher's
refusals, the rank-role words, the stock statements the adapter re-issues on rows (pinned by source sha: a fork that moves them is refused by
name here), and — with torch (+ the transformers fork) present — the sharded forms against the dense statements under a 2-rank gloo group
on CPU: t9's channel-major plane contraction (both directions) and the whole sharded loop -> readout -> coda gather -> confidence trunk on a
small host module that borrows the REAL ``ESMFold2Model._run_one_loop`` and the real ``FoldingTrunk`` (reference kernel backend). The t9
row-launch kernels (K1/K3 over R rows == t9's own launches at R == N, and rows of them) run only on a CUDA box with the kit's W4 module
importable (skipped by name elsewhere)."""
import inspect
import json
import os
import sys
import random
import unittest
from unittest import mock

try:
    from opt_core.mem import ngpu
    from opt_core.mem.rowpair import RowpairRefused
except ImportError as _e:                                                   # an older / absent core: named skip (the kit's entry routes refuse producer_missing by name)
    raise unittest.SkipTest(f"producer_missing:opt_core.mem.rowpair ({_e}): the n_gpu adapter tests need opt_core >= 0.4.0")
from esmfold2_opt import rowpair

# opt_core.diffusion_loop.source_guard.source_sha256(<fn>) at the transformers pin of stock/PINS.json (ef32577f): rowpair._run_one_loop_sharded mirrors
# ESMFold2Model._run_one_loop statement for statement, rowpair._tmb_forward_dispatch mirrors TriangleMultiplicativeBlock.forward, the coda /
# confidence wrappers assume FoldingTrunk.forward's (pair, pair_attention_mask) contract and ConfidenceHead.forward's x_pred parameter.
STOCK_SOURCE_PINS = {
    ("modeling_esmfold2", "ESMFold2Model._run_one_loop"): "0587c24ac4f74d07159e60593e8a599983cb443e4da8fd73dc78f5a95472f19a",
    ("modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"): "431a1d1cfca19d7b8bc2de56dc4c279cea3b521c5f5585c0f95875c53d36ebaa",
    ("modeling_esmfold2_common", "PairUpdateBlock.forward"): "fff506a8250486ef24c4254ded1e4deebc7cca36fdd1ae6fadb83a656f394fbc",
    ("modeling_esmfold2_common", "FoldingTrunk.forward"): "34ae63506205dfb108a1271297ae1f6c4831020284a1ee6faf04c26a2ffafd4d",
    ("modeling_esmfold2", "ConfidenceHead.forward"): "da666d31ad1e42df47c1f5ac3325c3e00bc25cbd1d0215d15a98e7d821bd7efe",
}


def np_random_seed(seed):
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass


def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch is not importable here ({type(e).__name__}): the sharded-form tests run where torch is installed (a CPU or GPU box with torch)")
    return sys.modules["torch"]


def _stock_or_skip():
    _torch_or_skip()
    try:
        from transformers.models.esmfold2 import modeling_esmfold2 as MOD, modeling_esmfold2_common as CMN  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"the transformers fork (stock/PINS.json) is not importable here ({type(e).__name__}: {e}): test skipped by name")
    return CMN, MOD


class StructuralRule(unittest.TestCase):
    """n_gpu = 1: the adapter installs nothing, reads nothing off the model, joins nothing."""

    def test_p1_installs_nothing(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"install(P=1) read model.{name}")
        m = Untouchable()
        rep = rowpair.install(m, 1)
        self.assertEqual(rep["n_gpu"], 1); self.assertEqual(rep["sharding"], "none"); self.assertFalse(rep["installed"])
        self.assertEqual(rowpair.PATCHES.names(), []); self.assertEqual(rep["patches"], [])
        self.assertEqual(rowpair.install_rank(m, 1)["sharding"], "none")
        rowpair.finish()                                                    # no group joined: a no-op
        json.dumps(rowpair.report())                                        # the manifest block is plain JSON

    def test_p1_words_come_from_the_family(self):
        self.assertEqual(rowpair.report()["sharding"], "none")
        self.assertEqual(rowpair.LEVER, "rowpair")

    def test_install_p2_without_a_group_is_refused_by_name(self):
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        with self.assertRaises(Exception) as cm:                           # RowpairRefused (torch present, no group) or the family's torch-absent refusal
            rowpair.install(object(), 2)
        self.assertIn("Refused", type(cm.exception).__name__ + str(type(cm.exception).__mro__))
        self.assertEqual(rowpair.PATCHES.names(), [])

    def test_launch_refuses_by_name(self):
        with self.assertRaises(ngpu.NGpuRefused):
            rowpair.launch([], "fast", 2)                                   # n_gpu>1 outside the memory mode
        with self.assertRaises(ngpu.NGpuRefused):
            rowpair.launch([], "exact", 4)
        with self.assertRaises(RowpairRefused):
            rowpair.launch([], "big", 1)                                  # the single-GPU line has no launcher

    def test_requested_P_reaches_the_family_launcher(self):
        """`pred --n_gpu P`: the launcher side hands EXACTLY P and this same `pred` command line to run_rank_processes (no inner call may
        drop the axis: a P>1 request that folds on one GPU is a silent degraded-protocol fallback)."""
        from opt_core.mem.rowpair import launch as RL
        seen = {}

        def fake_run(n_gpu, argv, **kw):
            seen.update(n_gpu=n_gpu, argv=list(argv), kw=kw)
            return [{"rank": r, "ok": True, "rc": 0, "exitcode": 0, "wall_s": 1.0, "log": f"/tmp/x/rank{r}.log", "error": None} for r in range(n_gpu)]
        argv = ["--variant", "fast", "--mode", "big", "--n_gpu", "4", "--input", "x.json", "--out_dir", "o", "--seeds", "1"]
        with mock.patch.object(RL, "run_rank_processes", fake_run), mock.patch.object(rowpair, "refuse_unless_visible", lambda P, visible=None: P):
            rc = rowpair.launch(argv, "big", 4, stream=open(os.devnull, "w"))
        self.assertEqual(rc, 0); self.assertIsInstance(rc, int)
        self.assertEqual(seen["n_gpu"], 4)
        self.assertEqual(seen["argv"][:4], [sys.executable, "-m", "esmfold2_opt", "pred"]); self.assertEqual(seen["argv"][4:], argv)
        self.assertTrue(seen["kw"].get("isolate_devices")); self.assertEqual(seen["kw"].get("mode"), "big")

        def failing_run(n_gpu, argv, **kw):
            raise RL.RankFailed("rank_failed", 1, 3, detail="tail", log="/tmp/x/rank1.log")
        with mock.patch.object(RL, "run_rank_processes", failing_run), mock.patch.object(rowpair, "refuse_unless_visible", lambda P, visible=None: P):
            rc = rowpair.launch(argv, "big", 4, stream=open(os.devnull, "w"))
        self.assertEqual(rc, 3)                                            # a rank's NOT ACTIVE (3) is the run's exit code, never 0

    def test_rank_roles_follow_the_family_environment(self):
        from opt_core.mem.rowpair import launch as RL
        with mock.patch.dict(os.environ, {RL.ENV_WORLD: "2", RL.ENV_RANK: "1"}):
            self.assertTrue(rowpair.is_rank_process()); self.assertFalse(rowpair.is_output_rank())
        with mock.patch.dict(os.environ, {RL.ENV_WORLD: "2", RL.ENV_RANK: "0"}):
            self.assertTrue(rowpair.is_rank_process()); self.assertTrue(rowpair.is_output_rank())
        env = {k: v for k, v in os.environ.items() if not k.startswith("ROWPAIR_") and k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(rowpair.is_rank_process()); self.assertTrue(rowpair.is_output_rank())

    def test_named_constants(self):
        self.assertEqual(rowpair.D_PAIR, 256)
        self.assertIn("confidence_head.folding_trunk", rowpair.SHARDED_TRUNKS)
        self.assertTrue({"x_inputs", "lm_activations", "atoms", "diffusion_a", "x_pred"} <= set(rowpair.REPLICATED), rowpair.REPLICATED)


class StockSourcePins(unittest.TestCase):
    def test_reissued_statements_match_the_pinned_stock_source(self):
        CMN, MOD = _stock_or_skip()
        from opt_core.diffusion_loop.source_guard import source_sha256          # the release tree's one digest of a stock function's text
        mods = {"modeling_esmfold2": MOD, "modeling_esmfold2_common": CMN}
        got = {}
        for (modname, qual), want in STOCK_SOURCE_PINS.items():
            obj = mods[modname]
            for part in qual.split("."):
                obj = getattr(obj, part)
            got[(modname, qual)] = source_sha256(obj)
        bad = {k: (v, STOCK_SOURCE_PINS[k]) for k, v in got.items() if v != STOCK_SOURCE_PINS[k]}
        self.assertEqual(bad, {}, "stock statements rowpair.py re-issues on rows have moved (got, pinned): re-read them against "
                                  "rowpair._run_one_loop_sharded / _tmb_forward_dispatch / the trunk wrappers, then re-pin")

    def test_confidence_head_takes_x_pred(self):
        CMN, MOD = _stock_or_skip()
        self.assertIn("x_pred", inspect.signature(MOD.ConfidenceHead.forward).parameters)
        self.assertEqual(list(inspect.signature(CMN.FoldingTrunk.forward).parameters)[:3], ["self", "pair", "pair_attention_mask"])


# ============================================================================================ 2-rank gloo group on CPU (spawned workers)
def _mp(P, entry, *args):
    from opt_core.mem.rowpair import launch as RL
    return RL.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=900)


def _entry_tmb(N: int, C: int, seed: int):
    """Every rank builds the same stock TriangleMultiplicativeBlock (both flows) and PairUpdateBlock-style residual on the whole pair; the
    dense stock forward runs whole, rowpair._tmb_forward_dispatch runs on this rank's rows inside the sharded context (the family driver
    trimul_update_), rows are gathered and rank 0 compares (fp32 CPU: the 1e-4-of-scale class; torch.equal reported)."""
    import torch
    from opt_core.mem.rowpair import dist as RD, shard as RS
    CMN, _MOD = _stock_or_skip()
    P, rank = RD.world()
    lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
    torch.manual_seed(seed); random.seed(seed); np_random_seed(seed)              # every RNG a constructor could touch, identical per rank
    res = {}
    z = torch.randn((1, N, N, C))
    mask = (torch.rand((1, N, N)) > 0.1).float()
    for flow in ("outgoing", "incoming"):
        tmb = CMN.TriangleMultiplicativeBlock(C, C, flow).eval()
        with torch.no_grad():
            for p in tmb.parameters():
                p.copy_(torch.randn(p.shape) * 0.3)
            tmb.set_chunk_size(None)
            dense = tmb(z, visibility=mask)
            rows = z[:, lay.r0:lay.r1].contiguous()
            rowpair.PATCHES.replace(CMN.TriangleMultiplicativeBlock, "forward", rowpair._tmb_forward_dispatch)
            try:
                with rowpair._sharded(lay, mask[:, lay.r0:lay.r1].contiguous()):
                    got_rows = tmb(rows, visibility=mask[:, lay.r0:lay.r1].contiguous())
            finally:
                rowpair.PATCHES.restore()
            full = RS.unshard_rows(got_rows.contiguous(), lay, dim=1)
        d = (full - dense).abs().max().item()
        res[flow] = {"max_abs_diff": d, "bitwise": bool(torch.equal(full, dense)), "scale": dense.abs().max().item(),
                     "rows_unchanged": bool(torch.equal(rows, z[:, lay.r0:lay.r1]))}
    res["layout"] = lay.facts()
    res["stats"] = {k: str(v) for k, v in rowpair.TRIMUL_STATS.items()}
    return res


class _ConfHost(object):
    pass


def _build_host(C: int, seed: int):
    """A minimal module with exactly the attributes ESMFold2Model._run_one_loop / the coda / the confidence trunk seams touch; the loop
    function IS the stock one (borrowed unbound), the trunks are the stock FoldingTrunk (reference backend on CPU)."""
    import types as _types
    import torch
    import torch.nn as nn
    from transformers.models.esmfold2 import modeling_esmfold2 as MOD, modeling_esmfold2_common as CMN
    torch.manual_seed(seed); random.seed(seed); np_random_seed(seed)              # identical module weights on every rank (test rule)

    class ConfHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.folding_trunk = CMN.FoldingTrunk(n_layers=1, d_pair=C, expansion_ratio=2)

        def forward(self, s_inputs, z, x_pred, pair_mask=None):
            return {"pair": z + self.folding_trunk(z, pair_attention_mask=pair_mask), "x_pred": x_pred}

    class Host(nn.Module):
        _run_one_loop = MOD.ESMFold2Model._run_one_loop

        def __init__(self):
            super().__init__()
            self.config = _types.SimpleNamespace(lm_encoder=_types.SimpleNamespace(per_loop_lm_dropout=False, lm_dropout=0.0), msa_encoder_overwrite=False)
            self.lm_encoder = CMN.FoldingTrunk(n_layers=1, d_pair=C, expansion_ratio=2)
            self.folding_trunk = CMN.FoldingTrunk(n_layers=2, d_pair=C, expansion_ratio=2)
            self.parcae_input_norm = nn.LayerNorm(C)
            self.parcae_readout = nn.Linear(C, C, bias=False)
            self.parcae_coda = CMN.FoldingTrunk(n_layers=1, d_pair=C, expansion_ratio=2)
            self.msa_encoder = None
            self.confidence_head = None                                    # the real head on rows: test_rowpair_heads

    h = Host().eval()
    with torch.no_grad():
        for p in h.parameters():                                            # away from the near-diagonal init so every statement matters
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(seed + 1 + p.numel() % 97)) * 0.2)
    for t in (h.lm_encoder, h.folding_trunk, h.parcae_coda):
        t.set_chunk_size(16)
    return h


def _host_pass(h, z, z_init, lm_z, pair_mask, a, b_mat, tok_mask, steps):
    import torch
    with torch.no_grad():
        z1 = h._run_one_loop(z, z_init, lm_z, None, pair_mask, a, b_mat, tok_mask, steps)
        z1 = h.parcae_readout(z1)
        z1 = h.parcae_coda(z1, pair_attention_mask=pair_mask)
    return z1


def _host_pass_rows(h, lay, z, z_init, lm_z, pair_mask, a, b_mat, tok_mask, steps):
    """The Mode-S loop entry as ``_forward_rows`` makes it: the layout and fold record set, every pair operand handed over as ROWS."""
    import torch
    from opt_core.mem.rowpair import shard as RS
    rows = slice(lay.r0, lay.r1)
    rowpair.CTX.layout = lay
    rowpair._new_fold(lay, torch, z.device)
    with torch.no_grad():
        z1 = h._run_one_loop(z[:, rows].contiguous(), z_init[:, rows].contiguous(), lm_z[:, rows].contiguous(), None, pair_mask[:, rows].contiguous(),
                             a, b_mat, tok_mask, steps)
        z1 = h.parcae_readout(z1)
        z1 = h.parcae_coda(z1, pair_attention_mask=pair_mask[:, rows].contiguous())
        assert z1.shape[1] == lay.R, (tuple(z1.shape), lay.R)               # rows out: nothing gathered
        return RS.unshard_rows(z1.contiguous(), lay, dim=1)


def _entry_loop(N: int, C: int, steps: int, seed: int, dropout: float):
    import torch
    from opt_core.mem.rowpair import dist as RD
    P, rank = RD.world()
    h = _build_host(C, seed)
    if dropout:
        h.config.lm_encoder.per_loop_lm_dropout, h.config.lm_encoder.lm_dropout = True, float(dropout)
    g = torch.Generator().manual_seed(seed + 7)
    z0 = torch.randn((1, N, N, C), generator=g) * 0.5
    z_init = torch.randn((1, N, N, C), generator=g) * 0.5
    lm_z = torch.randn((1, N, N, C), generator=g) * 0.5
    pair_mask = torch.ones((1, N, N)); pair_mask[:, :, N - 3:] = 0.0; pair_mask[:, N - 3:, :] = 0.0
    tok_mask = torch.ones((1, N), dtype=torch.bool)
    a = torch.rand((C,), generator=g) * 0.5 + 0.25
    b_mat = torch.eye(C) * 0.3 + torch.randn((C, C), generator=g) * 0.02
    out = {"rank": rank}
    torch.manual_seed(seed + 99)
    ref_z = _host_pass(h, z0.clone(), z_init.clone(), lm_z.clone(), pair_mask, a, b_mat, tok_mask, steps)
    assert rowpair.PATCHES.names() == []
    rowpair.STATE.update(P=P, rank=rank)
    from esmfold2_opt import atom_swa                                                  # the stub host has no atom attention: the sub-lever's install is not under test here
    atom_swa.install = lambda model, n_gpu, **kw: {"path": "absent:test_host"}; atom_swa.uninstall = lambda: []
    rep = rowpair.install(h, P)
    out["patches"] = list(rep["patches"])
    try:
        torch.manual_seed(seed + 99)
        lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
        sh_z = _host_pass_rows(h, lay, z0.clone(), z_init.clone(), lm_z.clone(), pair_mask, a, b_mat, tok_mask, steps)
        f = rowpair.CTX.fold
        line = rowpair._fold_line(f) if f is not None else ""
        nxt = torch.rand(1).item()                                           # the default generator advanced identically on every rank
        allv = [None] * P
        torch.distributed.all_gather_object(allv, nxt)
        out.update(z_max_abs_diff=(sh_z - ref_z).abs().max().item(), z_scale=ref_z.abs().max().item(), z_shape=list(sh_z.shape),
                   gen_next=allv, fold_line=line, report=rowpair.report(), finite=bool(torch.isfinite(sh_z).all().item()),
                   lm_dropout=(f or {}).get("lm_dropout"), coda_gather=(f or {}).get("coda_gather"))
    finally:
        out["restored"] = rowpair.uninstall()
        out["names_after"] = rowpair.PATCHES.names()
        out["loop_is_stock_again"] = h._run_one_loop.__func__ is h.__class__._run_one_loop
        rowpair.CTX.layout = None
    return out


class ShardedFormsUnderGloo(unittest.TestCase):
    """P = 2 CPU ranks (gloo); every number compared against the dense statement in the same processes."""

    @classmethod
    def setUpClass(cls):
        _stock_or_skip()

    def test_triangle_multiplication_rows_match_the_dense_module_both_flows(self):
        for N, C in ((80, 4), (203, 6)):                                     # 203: N not a multiple of the layout block
            res = _mp(2, _entry_tmb, N, C, 11)
            for key in ("outgoing", "incoming"):
                r = res[key]
                self.assertLessEqual(r["max_abs_diff"], 1e-4 * max(1.0, r["scale"]), (N, C, key, r))
                self.assertTrue(r["rows_unchanged"], (N, C, key))              # the module's forward returns the update; its input rows are untouched
            print(f"[test] TriangleMultiplicativeBlock rows N={N} C={C} P=2: out bitwise={res['outgoing']['bitwise']} diff={res['outgoing']['max_abs_diff']:.2e} "
                  f"in bitwise={res['incoming']['bitwise']} diff={res['incoming']['max_abs_diff']:.2e} stats={res['stats']} layout={res['layout']}")

    def test_sharded_loop_and_coda_on_rows_match_the_stock_loop(self):
        res = _mp(2, _entry_loop, 80, 8, 2, 5, 0.0)
        print(f"[test] Mode-S loop N=80 C=8 P=2: z diff={res['z_max_abs_diff']:.2e} (scale {res['z_scale']:.2f}) coda_gather={res['coda_gather']} line: {res['fold_line']}")
        self.assertTrue(res["finite"], res)
        self.assertEqual(res["z_shape"], [1, 80, 80, 8])
        self.assertLessEqual(res["z_max_abs_diff"], 2e-4 * max(1.0, res["z_scale"]), res)
        self.assertEqual(res["coda_gather"], "none")
        self.assertEqual(len(set(res["gen_next"])), 1, f"default generators diverged across ranks: {res['gen_next']}")
        self.assertEqual(len(res["patches"]), 5, res["patches"])             # forward, loop, coda, TriangleMultiplicativeBlock.forward, Transition.forward
        self.assertEqual(res["names_after"], []); self.assertTrue(res["loop_is_stock_again"])
        f = res["report"]["folds"][0]
        self.assertTrue(f["rows_tiled"]); self.assertEqual(f["P"], 2); self.assertEqual(f["N"], 80)
        self.assertIn("folding_trunk", f["t"]); self.assertEqual(len(f["t"]["folding_trunk"]), 2)   # one timing per loop
        self.assertIn("rowpair fold", res["fold_line"])
        for key in ("rows_tiled=", "row_bounds=0:", "folding_trunk_s=", "trimul_RA="):
            self.assertIn(key, res["fold_line"])

    def test_sharded_loop_with_lm_dropout_runs_and_keeps_generators_in_step(self):
        res = _mp(2, _entry_loop, 64, 8, 2, 6, 0.25)
        self.assertTrue(res["finite"]); self.assertEqual(res["z_shape"], [1, 64, 64, 8])
        self.assertEqual(len(set(res["gen_next"])), 1, f"default generators diverged across ranks: {res['gen_next']}")
        self.assertEqual(res["lm_dropout"], "rank_rows")                       # CPU ranks: the NAMED private draw (CUDA ranks: philox_rows = the stock stream's rows)
        self.assertGreater(res["z_max_abs_diff"], 0.0)                        # a different dropout realisation than the single-process draw on CPU


class BelowTheShardingFloor(unittest.TestCase):
    """n_gpu>1 with an input the row grid cannot shard (Layout.auto: no grid without an empty rank — tiny inputs): NOT refused. The fold takes
    ESMFold2Model.forward as kept (every callable the lever rebound serves its original while CTX.below_floor), on every rank, no collective;
    one named line; the fold's census record says sharding=none:below_floor; the context is restored afterwards."""

    def setUp(self):
        self._state = {k: rowpair.STATE.get(k) for k in ("P", "rank", "layouts", "folds", "installed")}
        rowpair.STATE.update(P=2, rank=1, layouts={}, folds=[], installed=True)

    def tearDown(self):
        rowpair.PATCHES.restore()
        rowpair.STATE.update(**self._state)
        rowpair.CTX.below_floor = False

    def test_the_floor_is_the_family_grid_rule(self):
        from opt_core.mem.rowpair import dist as RD
        with self.assertRaises(RowpairRefused):                                   # the 20-token monomer of the A100x2 / H100x2 runs: no grid over 2 ranks
            RD.Layout.auto(20, 2, 0)
        self.assertIsNone(rowpair._layout_or_none(20)); self.assertIsNone(rowpair.STATE["layouts"][20])       # cached as "below the floor"
        from opt_core.mem.rowpair import dist as RD2
        self.assertEqual(RD2.Layout.auto(1000, 2, 1).facts()["policy"], "grid")     # a normal item has a grid (every memory-ladder bin x P does)
        with self.assertRaises(RowpairRefused) as cm:                             # ... and outside a process group its layout is still REFUSED (only the grid rule routes below the floor)
            rowpair._layout_or_none(1000)
        self.assertIn("group=no", str(cm.exception))

    def test_a_tiny_input_is_folded_whole_by_the_kept_forward_and_named(self):
        import io, contextlib, types
        torch = _torch_or_skip()
        seen = {}

        class Host(object):
            config = types.SimpleNamespace(num_loops=2, num_diffusion_samples=1)

            def forward(self, **kw):                                               # the KEPT forward (what P=1 runs): records the context it ran under
                seen.update(n=len(kw), below=rowpair.CTX.below_floor, active=rowpair.CTX.active, layout=rowpair.CTX.layout, N=int(kw["token_attention_mask"].shape[1]))
                return {"ok": True, "num_loops": kw.get("num_loops")}
        host = Host()
        rowpair.PATCHES.replace(host, "forward", types.MethodType(rowpair._forward_rows, host))
        names = [p for p in inspect.signature(rowpair._forward_rows).parameters if p not in ("self", "kwargs")]
        call = {n: None for n in names}
        call.update(token_attention_mask=torch.ones(1, 20, dtype=torch.bool), ref_pos=torch.zeros(1, 3), num_loops=3, extra_key="kept")   # an unknown key rides **kwargs
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = host.forward(**call)
        self.assertEqual(out, {"ok": True, "num_loops": 3})                        # the kept forward ran with the call as made
        self.assertEqual(seen, dict(n=len(names) + 1, below=True, active=False, layout=None, N=20))
        line = err.getvalue().strip()
        self.assertEqual(line.count("\n"), 0, line)                                # ONE line
        self.assertIn("n_gpu=2: input below the sharding floor (N=20: no row grid over 2 ranks) — folded unsharded on rank 1", line)
        self.assertIn("sharding=none:below_floor", line); self.assertEqual(rowpair.BELOW_FLOOR_WORD, "none:below_floor")
        f = rowpair.STATE["folds"][-1]
        self.assertEqual((f["N"], f["P"], f["rank"], f["sharding"]), (20, 2, 1, "none:below_floor")); self.assertIn("wall_s", f)
        self.assertEqual(rowpair.report()["folds"][-1]["sharding"], "none:below_floor")
        self.assertFalse(rowpair.CTX.below_floor); self.assertIsNone(rowpair.CTX.fold)                     # context restored
        with self.assertRaises(RowpairRefused):                                   # batch != 1 stays refused ([10] is a refusal by design), floor or not
            host.forward(**dict(call, token_attention_mask=torch.ones(2, 20, dtype=torch.bool)))

    def test_every_rebound_callable_serves_its_original_below_the_floor(self):
        import types
        marks = []

        class DM(object):
            def forward(self, *a, **kw): marks.append(("dm", a, kw)); return "dm_orig"
        class Head(object):
            folding_trunk = None
            def forward(self, *a, **kw): marks.append(("head", a, kw)); return "head_orig"
        class Loop(object):
            def _run_one_loop(self, **kw): marks.append(("loop", tuple(sorted(kw)))); return "loop_orig"
        dm, head, loop = DM(), Head(), Loop()
        rowpair.PATCHES.replace(dm, "forward", rowpair._dm_forward(dm))
        rowpair.PATCHES.replace(head, "forward", rowpair._conf_head_rows(head, lambda *a, **k: None))
        rowpair.PATCHES.replace(loop, "_run_one_loop", types.MethodType(rowpair._run_one_loop_sharded, loop))
        rowpair.CTX.below_floor = True
        try:
            self.assertEqual(dm.forward(1, x=2), "dm_orig")
            self.assertEqual(head.forward(x_pred=None), "head_orig")
            kw = dict(z=None, z_init=None, lm_z=None, _msa_inputs=None, pair_mask=None, a=None, b_mat=None, tok_mask=None, total_steps=2)
            self.assertEqual(loop._run_one_loop(**kw), "loop_orig")
        finally:
            rowpair.CTX.below_floor = False
        self.assertEqual([m[0] for m in marks], ["dm", "head", "loop"]); self.assertEqual(marks[2][1], tuple(sorted(kw)))


# ============================================================================================ t9 row launches (CUDA + the kit's W4 module)
if __name__ == "__main__":
    unittest.main()
