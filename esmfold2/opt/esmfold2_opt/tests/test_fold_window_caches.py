"""The timed window and the kit's cross-fold caches: on EVERY route the rows' feature call comes AFTER the timed builder.fold() (inside the
window each arm pays its own featurisation exactly as upstream's fold() does), and one `CACHE` line per fold states the kit caches' traffic
inside the window (counter deltas) so a timing pass can assert 0 cross-input hits. CPU only; the fold loop runs with stub builder /
settings / writer."""
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

from esmfold2_opt import _autoload, report, stock_fold


class _Settings:
    preset = "stub"; msa_read_depth = None; msa_remove_insertions = False; seeds = None; n_seeds = 0
    def fold_kwargs(self, det=0):
        return {"num_diffusion_samples": 1}


class _Builder:
    """prepare_input / fold record their call order; fold() prepares its own input (as upstream's does) and, on the 'kit' arm, bumps the
    kit cache counters the way a first-seen input does (a store inside the window)."""
    def __init__(self, calls, kit_stats=None):
        self.calls = calls; self.kit_stats = kit_stats
    def prepare_input(self, spi, seed=None, device=None):
        self.calls.append("prepare_input"); return {"tok": 1}, None
    def fold(self, model, spi, seed=None, complex_id=None, **kw):
        self.calls.append("fold")
        if self.kit_stats is not None:
            self.kit_stats["feature_cache_stores"] += 1; self.kit_stats["esmc_cache_stores"] += 1
        return types.SimpleNamespace(id=complex_id)


def _item(name):
    return {"id": name, "sequences": [{"type": "protein", "id": "A", "sequence": "MKV"}]}


def _run(builder, seeds=(7,), environ=None, items=None):
    rows = []
    def writer(item, seed, res, depths, feats, wall):
        rows.append((item["id"], seed)); return [{"id": item["id"], "seed": seed}]
    items = [_item("A1")] if items is None else items
    err = io.StringIO()
    with mock.patch("esmfold2_opt.inputs.build_spi", return_value=object()), mock.patch("esmfold2_opt.inputs.msa_depths", return_value=[0]), \
         mock.patch.dict(os.environ, environ or {}), redirect_stderr(err):
        n = stock_fold.fold_items(model=types.SimpleNamespace(device="cpu"), builder=builder, items=items, settings=_Settings(), variant="fast",
                                  out_dir=os.devnull, seeds=list(seeds), det=0, log=lambda *a, **k: None, writer=writer, device="cpu")
    return n, rows, err.getvalue()


class TestRowsFeatureCallFollowsTheTimedFold(unittest.TestCase):
    def test_stock_arm_order_and_absent_caches(self):
        calls = []
        prev = sys.modules.pop("ef2_opt", None)
        self.addCleanup(lambda: sys.modules.__setitem__("ef2_opt", prev) if prev is not None else None)
        n, rows, err = _run(_Builder(calls))
        self.assertEqual(n, 1); self.assertEqual(calls, ["fold", "prepare_input"])                       # the window (fold) first, the rows' features after it
        self.assertIn("[esmfold2-opt] CACHE item=A1 seed=7 window=fold caches=absent", err)

    def test_kit_arm_order_and_window_deltas(self):
        import collections
        fake = types.ModuleType("ef2_opt"); fake.STATS = collections.Counter()
        class _C:
            def __init__(self): self.cleared = 0
            def clear(self): self.cleared += 1
        fake._FEATURE_CACHE, fake._ESMC_CACHE = _C(), _C()
        calls = []
        prev = sys.modules.get("ef2_opt"); sys.modules["ef2_opt"] = fake                                # set / restored by hand (never mock.patch.dict on sys.modules)
        try:
            n, rows, err = _run(_Builder(calls, kit_stats=fake.STATS), seeds=(7, 8), environ={report.ENV_CACHE_SCOPE: "global"})
            self.assertEqual(calls, ["fold", "prepare_input", "fold", "prepare_input"])
            self.assertIn("[esmfold2-opt] CACHE item=A1 seed=7 window=fold feature_cache_hits=0 feature_cache_stores=1 esmc_cache_hits=0 esmc_cache_stores=1 scope=global", err)
            self.assertIn("CACHE item=A1 seed=8 window=fold", err); self.assertEqual(fake._FEATURE_CACHE.cleared, 0)      # scope global: nothing emptied
            calls.clear()
            n, rows, err = _run(_Builder(calls, kit_stats=fake.STATS), seeds=(7, 8))                            # the default scope: input — emptied once per new input, not per seed
            self.assertEqual((fake._FEATURE_CACHE.cleared, fake._ESMC_CACHE.cleared), (1, 1)); self.assertIn("scope=input", err)
            self.assertEqual(calls, ["fold", "prepare_input", "fold", "prepare_input"])
        finally:
            if prev is None:
                sys.modules.pop("ef2_opt", None)
            else:
                sys.modules["ef2_opt"] = prev


class TestCachesHoldTheCurrentInputOnly(unittest.TestCase):
    """Default scope `input`: folding input A then input B in one process leaves no entry of A resident in the kit's feature / ESM-C caches
    (their CUDA tensors are released when the loop moves to B); `global` keeps both. The fake caches hold real entries keyed by input."""
    def _fake(self):
        import collections
        fake = types.ModuleType("ef2_opt"); fake.STATS = collections.Counter()
        class _Cache:
            def __init__(self): self.d = collections.OrderedDict()
            def clear(self): self.d.clear()
        fake._FEATURE_CACHE, fake._ESMC_CACHE = _Cache(), _Cache()
        class _StoringBuilder(_Builder):
            def fold(self, model, spi, seed=None, complex_id=None, **kw):          # a first-seen input stores one entry per cache inside the window
                for c, k in ((fake._FEATURE_CACHE, "feature_cache"), (fake._ESMC_CACHE, "esmc_cache")):
                    if complex_id in c.d:
                        fake.STATS[k + "_hits"] += 1
                    else:
                        c.d[complex_id] = object(); fake.STATS[k + "_stores"] += 1
                self.calls.append("fold"); return types.SimpleNamespace(id=complex_id)
        return fake, _StoringBuilder

    def _with_fake(self, fake):
        prev = sys.modules.get("ef2_opt"); sys.modules["ef2_opt"] = fake
        def restore():
            if prev is None:
                sys.modules.pop("ef2_opt", None)
            else:
                sys.modules["ef2_opt"] = prev
        self.addCleanup(restore)

    def test_default_scope_input_a_then_b_leaves_only_b_resident(self):
        fake, B = self._fake(); self._with_fake(fake)
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(report.ENV_CACHE_SCOPE, None)
            n, rows, err = _run(B([]), seeds=(7, 8), items=[_item("A1"), _item("B2")])
        self.assertEqual(rows, [("A1", 7), ("A1", 8), ("B2", 7), ("B2", 8)])
        self.assertEqual(list(fake._FEATURE_CACHE.d), ["B2"]); self.assertEqual(list(fake._ESMC_CACHE.d), ["B2"])      # no A1 entry resident
        self.assertIn("CACHE item=A1 seed=8 window=fold feature_cache_hits=1 feature_cache_stores=0", err)               # the seeds of one input still hit
        self.assertIn("CACHE item=B2 seed=7 window=fold feature_cache_hits=0 feature_cache_stores=1 esmc_cache_hits=0 esmc_cache_stores=1 scope=input", err)

    def test_scope_global_keeps_both(self):
        fake, B = self._fake(); self._with_fake(fake)
        n, rows, err = _run(B([]), seeds=(7,), items=[_item("A1"), _item("B2")], environ={report.ENV_CACHE_SCOPE: "global"})
        self.assertEqual(list(fake._FEATURE_CACHE.d), ["A1", "B2"]); self.assertEqual(list(fake._ESMC_CACHE.d), ["A1", "B2"])
        self.assertIn("scope=global", err)


class TestCacheLineGrammarAndScope(unittest.TestCase):
    def test_grammar(self):
        b = {k: 3 for k in report.CACHE_KEYS}; a = dict(b, feature_cache_hits=4)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(report.ENV_CACHE_SCOPE, None)
            self.assertEqual(report.cache_line("x", 1, b, a), "[esmfold2-opt] CACHE item=x seed=1 window=fold feature_cache_hits=1 feature_cache_stores=0 esmc_cache_hits=0 esmc_cache_stores=0 scope=input")
        self.assertEqual(report.cache_line("x", 1, None, a), "[esmfold2-opt] CACHE item=x seed=1 window=fold caches=absent")
        self.assertEqual(report.CACHE_KEYS, ("feature_cache_hits", "feature_cache_stores", "esmc_cache_hits", "esmc_cache_stores"))

    def test_scope_words(self):
        self.assertEqual(report.cache_scope({}), "input"); self.assertEqual(report.cache_scope({report.ENV_CACHE_SCOPE: "input"}), "input")
        self.assertEqual(report.cache_scope({report.ENV_CACHE_SCOPE: "GLOBAL"}), "global")
        with self.assertRaises(ValueError):
            report.cache_scope({report.ENV_CACHE_SCOPE: "run"})
        self.assertIn(report.ENV_CACHE_SCOPE, _autoload.ENV_NAMES)                                        # declared: the start-up hook accepts it

    def test_counts_absent_without_the_kit_module(self):
        prev = sys.modules.pop("ef2_opt", None)
        try:
            self.assertIsNone(report.cache_counts()); self.assertFalse(report.cache_clear())
        finally:
            if prev is not None:
                sys.modules["ef2_opt"] = prev


if __name__ == "__main__":
    unittest.main()
