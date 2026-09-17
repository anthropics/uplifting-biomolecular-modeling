"""The multi-GPU route's row-chunking levers (esmfold2_opt.rowchunk) below and above their token floor, on CPU: below
EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS every member runs the unchunked route's statements exactly (the confidence statement dispatches whole to
the kept original function object; zbf16 hands the confidence head the call site's own ``z.detach().float()``; the recycle inject issues
the whole-shard statements), at or above it the row-blocked ones; the engagement guard judges a bound member only when its phase ran at
or above the floor (rowchunk.install.PHASE). Mock tensors / stub callables only: no GPU, no kit server."""
import unittest

import torch

from esmfold2_opt import rowpair
from esmfold2_opt.rowchunk import confrows, zbf16rows, install as rci


class _Lay:
    def __init__(self, N):
        self.N = N


class _Z:
    """A stand-in for the pair rows: only .shape and .is_cuda are read by the dispatch."""
    def __init__(self, N, cuda):
        self.shape = (1, 4, N, 8); self.is_cuda = cuda


class ConfidenceStatementDispatchesOnTheFloor(unittest.TestCase):
    def setUp(self):
        import esmfold2_opt.rowpair_heads as H
        self.H = H
        self.orig = H.confidence_forward_rows_batched
        confrows.unapply()
        confrows.apply(mb=512.0, min_N=1024, bf16_pair=True, pde_skip=True, confmem=True)
        self.calls = []
        self.kept = dict(confrows._ORIG)
        confrows._ORIG["fn"] = lambda *a, **k: self.calls.append("whole") or "whole"
        confrows._ORIG["patched"] = lambda *a, **k: self.calls.append("blocked") or "blocked"
        self.skips0, self.blocked0 = confrows.STATS["floor_skips"], confrows.STATS["dispatched_blocked"]

    def tearDown(self):
        confrows._ORIG.clear(); confrows._ORIG.update(self.kept)
        confrows.unapply()
        self.assertIs(self.H.confidence_forward_rows_batched, self.orig)               # unapply restores the module's own statement

    def _call(self, N, cuda=True, lay=True):
        return self.H.confidence_forward_rows_batched(None, _Lay(N) if lay else None, None, None, _Z(N, cuda), None, num_diffusion_samples=1)

    def test_apply_installs_the_dispatch_and_keeps_the_original(self):
        self.assertTrue(getattr(self.H.confidence_forward_rows_batched, "__confrows__", False))
        self.assertIs(self.kept["fn"], self.orig)                                        # what runs below the floor IS the unpatched function object
        self.assertTrue(getattr(self.kept["patched"], "__confrows__", False))
        # the bf16 prologue / distance-bin embed / head / TM / weight-gather entry points live ONLY in the row-blocked function: below the
        # floor (the original runs) none of them can execute — the below-floor confidence delta measured on the import came from the first two
        blocked, whole = set(self.kept["patched"].__code__.co_names), set(self.orig.__code__.co_names)
        for entry in ("_RC_CONF_PROLOGUE", "_RC_CONF_EMBED_ROWS", "_RC_CONF_HEAD_ROWS", "_RC_CONF_TM_ROWS", "_RC_CONF_WEIN", "_RC_CONF_TRUNK", "_RC_CONF_RAP_ROWS", "_RC_CONF_ADD_ROWS"):
            self.assertIn(entry, blocked); self.assertNotIn(entry, whole)

    def test_below_the_floor_the_whole_statement_runs(self):
        for N in (16, 949, 1012, 1023):
            self.assertEqual(self._call(N), "whole")
        self.assertEqual(self.calls, ["whole"] * 4)
        self.assertEqual(confrows.STATS["floor_skips"] - self.skips0, 4)
        self.assertEqual(confrows.STATS["dispatched_blocked"] - self.blocked0, 0)

    def test_at_or_above_the_floor_the_row_blocked_statement_runs(self):
        for N in (1024, 1025, 6656):
            self.assertEqual(self._call(N), "blocked")
        self.assertEqual(confrows.STATS["dispatched_blocked"] - self.blocked0, 3)
        self.assertEqual(self._call(4096, lay=False), "blocked")                         # N from z when no layout is given

    def test_off_the_gpu_the_whole_statement_runs_at_any_size(self):
        self.assertEqual(self._call(8192, cuda=False), "whole")


class Zbf16CopyOnTheFloor(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(zbf16rows._CFG); zbf16rows._CFG.update(min_N=64, mode="conf", mb=0.01)
        self.dev = zbf16rows._on_device

    def tearDown(self):
        zbf16rows._CFG.clear(); zbf16rows._CFG.update(self.cfg); zbf16rows._on_device = self.dev

    def test_below_the_floor_the_call_sites_own_expression(self):
        z = torch.randn(1, 6, 48, 8)                                                     # N=48 < 64
        s0, c0 = zbf16rows.STATS["floor_skips"], zbf16rows.STATS["conf_calls"]
        out = zbf16rows.z_conf(z)
        self.assertEqual(out.dtype, torch.float32); self.assertEqual(out.data_ptr(), z.data_ptr())   # z.detach().float(): no copy, no cast
        self.assertEqual((zbf16rows.STATS["floor_skips"] - s0, zbf16rows.STATS["conf_calls"] - c0), (1, 0))

    def test_off_the_gpu_is_below_the_floor(self):
        z = torch.randn(1, 6, 128, 8)                                                    # N >= floor but a CPU tensor
        out = zbf16rows.z_conf(z)
        self.assertEqual((out.dtype, out.data_ptr()), (torch.float32, z.data_ptr()))

    def test_at_or_above_the_floor_a_banded_bf16_copy(self):
        zbf16rows._on_device = lambda t: True
        z = torch.randn(1, 6, 128, 8)
        c0, b0 = zbf16rows.STATS["conf_calls"], zbf16rows.STATS["conf_blocks"]
        out = zbf16rows.z_conf(z)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, z.to(torch.bfloat16)))                            # a copy in bands is the whole cast, exactly
        self.assertEqual(zbf16rows.STATS["conf_calls"] - c0, 1)
        self.assertGreater(zbf16rows.STATS["conf_blocks"] - b0, 1)                     # mb=0.01 -> several bands at this size


class _Trunk:
    def __init__(self, C):
        self.parcae_input_norm = torch.nn.LayerNorm(C).double()


class RecycleInjectOnTheFloor(unittest.TestCase):
    def setUp(self):
        self.inj = dict(rowpair.INJ); self.dev = rowpair._inj_on_device
        torch.manual_seed(0)
        self.C = 8
        self.trunk = _Trunk(self.C)
        self.b = torch.randn(self.C, self.C, dtype=torch.float64)
        self.a = torch.tensor(0.7, dtype=torch.float64)

    def tearDown(self):
        rowpair.INJ.clear(); rowpair.INJ.update(self.inj); rowpair._inj_on_device = self.dev

    def _whole(self, z, zi, lm):
        import torch.nn.functional as F
        x = zi + lm.to(zi.dtype) if lm is not None else zi
        return self.a * z + F.linear(self.trunk.parcae_input_norm(x).to(z.dtype), self.b)

    def _tensors(self, R, N):
        return (torch.randn(1, R, N, self.C, dtype=torch.float64), torch.randn(1, R, N, self.C, dtype=torch.float64), torch.randn(1, R, N, self.C, dtype=torch.float64))

    def test_below_the_floor_the_whole_shard_statements(self):
        rowpair.INJ.update(on=True, mb=512, min_tokens=64); rowpair._inj_on_device = lambda t: True
        z, zi, lm = self._tensors(6, 48)
        s0, c0 = rowpair.INJ_STATS["floor_skips"], rowpair.INJ_STATS["chunked"]
        out = rowpair._inject_rows(self.trunk, z.clone(), zi, lm, self.a, self.b)
        self.assertTrue(torch.equal(out, self._whole(z, zi, lm)))
        self.assertEqual((rowpair.INJ_STATS["floor_skips"] - s0, rowpair.INJ_STATS["chunked"] - c0), (1, 0))

    def test_off_or_uninstalled_the_whole_shard_statements(self):
        rowpair.INJ.update(on=False); rowpair._inj_on_device = lambda t: True
        z, zi, lm = self._tensors(6, 128)
        c0 = rowpair.INJ_STATS["chunked"]
        out = rowpair._inject_rows(self.trunk, z.clone(), zi, lm, self.a, self.b)
        self.assertTrue(torch.equal(out, self._whole(z, zi, lm))); self.assertEqual(rowpair.INJ_STATS["chunked"], c0)

    def test_at_or_above_the_floor_per_row_block_same_values(self):
        rowpair.INJ.update(on=True, mb=128 * 8 * 8 * 2 / 2 ** 20, min_tokens=64); rowpair._inj_on_device = lambda t: True   # 2 rows per block at N=128, C=8, fp64
        z, zi, lm = self._tensors(7, 128)
        c0, b0 = rowpair.INJ_STATS["chunked"], rowpair.INJ_STATS["blocks"]
        ref = self._whole(z, zi, lm)
        out = rowpair._inject_rows(self.trunk, z.clone(), zi, lm, self.a, self.b)
        self.assertEqual(rowpair.INJ_STATS["chunked"] - c0, 1); self.assertEqual(rowpair.INJ_STATS["blocks"] - b0, 4); self.assertEqual(rowpair.INJ_STATS["rows_per_block"], 2)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-12, atol=1e-12))              # row-local ops: the whole statement's values (float64: to the last bits of the GEMM order)
        out2 = rowpair._inject_rows(self.trunk, z.clone(), zi, None, self.a, self.b)      # without the LM term
        self.assertTrue(torch.allclose(out2, self._whole(z, zi, None), rtol=1e-12, atol=1e-12))


class EngagementIsJudgedOnlyWhereThePhaseRan(unittest.TestCase):
    """rowchunk.install.engagement: a BOUND member with a zero engagement counter is 'unreached' only if its phase ran at/above the floor."""
    def setUp(self):
        self.state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rci.STATE.items()}
        self.saved = (dict(rowpair.INJ_STATS), dict(confrows.STATS), dict(zbf16rows.STATS))
        from esmfold2_opt.rowchunk import confmem
        self.cm = confmem; self.cm_saved = dict(confmem.STATS)
        rci.STATE.update(installed=True, levers={n: True for n in rci.MEMBERS}, why={}, knobs={"min_tokens": 1024, "inject_mb": 512.0, "conf_mb": 512.0, "zbf16_mb": 1024.0}, P=2)
        for d in (rowpair.INJ_STATS, confrows.STATS, zbf16rows.STATS, confmem.STATS):
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    d[k] = 0

    def tearDown(self):
        rci.STATE.clear(); rci.STATE.update(self.state)
        rowpair.INJ_STATS.clear(); rowpair.INJ_STATS.update(self.saved[0]); confrows.STATS.clear(); confrows.STATS.update(self.saved[1])
        zbf16rows.STATS.clear(); zbf16rows.STATS.update(self.saved[2]); self.cm.STATS.clear(); self.cm.STATS.update(self.cm_saved)

    def test_every_phase_ran_and_nothing_engaged_is_unreached_for_all(self):
        folds = [{"N": 4096, "reached_inject": 21, "reached_sample": 1, "reached_confidence": 1}]
        self.assertEqual(set(rci.engagement(folds)), set(rci.MEMBERS))
        self.assertEqual(rci.expected_calls(folds, 1024)["injrows"], 21)

    def test_counters_present_is_engaged(self):
        rowpair.INJ_STATS["chunked"] = 21; confrows.STATS.update(head_chunked=2, prologue_calls=2, pde_skipped=2, wein_calls=4)
        self.cm.STATS["sample_calls"] = 1; zbf16rows.STATS["conf_calls"] = 1
        self.assertEqual(rci.engagement([{"N": 4096, "reached_inject": 21, "reached_sample": 1, "reached_confidence": 1}]), {})

    def test_folds_below_the_floor_prove_nothing(self):
        self.assertEqual(rci.engagement([{"N": 949, "reached_inject": 21, "reached_sample": 1, "reached_confidence": 1}, {"N": 1012}]), {})

    def test_a_model_without_a_confidence_head_judges_only_inject_and_sampler(self):
        got = rci.engagement([{"N": 4096, "reached_inject": 3, "reached_sample": 1}])        # no reached_confidence mark
        self.assertEqual(set(got), {"injrows", "biasfree"})

    def test_a_fold_that_died_in_the_trunk_judges_nothing_downstream(self):
        rowpair.INJ_STATS["chunked"] = 2
        self.assertEqual(rci.engagement([{"N": 4096, "reached_inject": 2}]), {})               # sampler / confidence never reached: not judged (the run keeps its own failure code)

    def test_unbound_members_are_never_judged(self):
        rci.STATE["levers"] = dict(rci.STATE["levers"], zbf16=False, pdeskip=False)
        got = rci.engagement([{"N": 4096, "reached_inject": 1, "reached_sample": 1, "reached_confidence": 1}])
        self.assertNotIn("zbf16", got); self.assertNotIn("pdeskip", got); self.assertIn("confrows", got)

    def test_not_installed_judges_nothing(self):
        rci.STATE["installed"] = False
        self.assertEqual(rci.engagement([{"N": 4096, "reached_inject": 1, "reached_sample": 1, "reached_confidence": 1}]), {})


if __name__ == "__main__":
    unittest.main()
