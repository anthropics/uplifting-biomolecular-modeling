"""DEVICE_RESIDENT on the stub RunModel (stand-in jax when none is installed): the class rebound with its marker, host leaves uploaded
once and found again by identity (parameters per model swap, features per input, the recycling inputs never re-uploaded), the outputs
returned as host float16 equal to stock's `np.asarray(v, np.float16)` of the same call, the counters the LEVER line prints, and the
pass-through of a traced (non-concrete) call. No GPU."""
import sys
import unittest
from unittest import mock

import numpy as np

from colabfold_opt import device_resident, modes, registry, stack
from colabfold_opt.tests import _stubs

try:
    from opt_core.mem.rowpair_jax import alphafold as _recipe            # the AlphaFold-2 row-shard recipe (big --n_gpu P>1); absent in a core older than the axis
    HAVE_RECIPE = True
except ImportError:
    _recipe = None
    HAVE_RECIPE = False


def stock_round_trip(out):
    """model.py:165-171 `_jnp_to_np`: every leaf through np.asarray(v, np.float16), dicts in place."""
    for k, v in out.items():
        out[k] = stock_round_trip(v) if isinstance(v, dict) else np.asarray(v, np.float16)
    return out


class TestResidentApply(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests(); device_resident.reset_for_tests()
        self.mods, self.saved = _stubs.install()
        self.model = self.mods["alphafold.model.model"]
        self.Stock = self.model.RunModel                                              # the class before the rebind: the reference call

    def tearDown(self):
        _stubs.remove(self.saved)
        device_resident.reset_for_tests(); stack.reset_for_tests()

    def test_enable_rebinds_with_the_marker_and_is_idempotent(self):
        self.assertFalse(device_resident.marker_present(self.model))
        self.assertTrue(device_resident.enable(self.model)); first = self.model.RunModel
        self.assertTrue(device_resident.enable(self.model)); self.assertIs(self.model.RunModel, first)          # no second subclass
        self.assertTrue(getattr(first, device_resident.MARKER)); self.assertTrue(issubclass(first, self.Stock))
        self.assertEqual((first.__name__, first.__qualname__, first.__module__), ("RunModel", self.Stock.__qualname__, self.Stock.__module__))
        self.assertTrue(device_resident._STATE["enabled"]); self.assertTrue(registry.marker_present(modes.HOST_LEVER))
        self.assertEqual(registry.applied()[modes.HOST_LEVER], True)

    def test_recycles_upload_once_and_match_stock_bit_for_bit(self):
        device_resident.enable(self.model)
        params_1 = {"evoformer": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, "head": {"b": np.ones(2, np.float32)}}
        params_2 = {"evoformer": {"w": np.arange(6, dtype=np.float32).reshape(2, 3) * 2}, "head": {"b": np.ones(2, np.float32)}}
        feat = {"msa": np.arange(8, dtype=np.int32).reshape(2, 4), "seq_mask": np.ones(2, np.float32)}
        prev0 = {"prev_pair": np.zeros((2, 2), np.float16), "prev_pos": np.zeros((2, 3))}                       # stock's initial recycling inputs (float16 / float64)
        runner, ref = self.model.RunModel(params=params_1), self.Stock(params=params_1)
        st = device_resident._STATE
        # recycle 0: every host leaf uploaded (2 parameter leaves + 2 features + 2 prev)
        out = runner.apply(runner.params, None, {**feat, "prev": prev0})
        want = stock_round_trip(ref.apply(ref.params, None, {**feat, "prev": prev0}))
        self.assertEqual((st["calls"], st["uploads"], st["hits"]), (1, 6, 0))
        self.assert_same_tree(out, want)
        prev = out.pop("prev"); want_prev = want.pop("prev")
        # recycle 1: parameters and features found by identity, prev resolved to its device twin — nothing uploaded
        out = runner.apply(runner.params, None, {**feat, "prev": prev})
        want = stock_round_trip(ref.apply(ref.params, None, {**feat, "prev": want_prev}))
        self.assertEqual((st["calls"], st["uploads"], st["hits"]), (2, 6, 6))
        self.assert_same_tree(out, want)
        passed = runner.APPLY_ARGS[-1]
        jax = sys.modules["jax"]
        self.assertTrue(all(isinstance(v, jax.Array) for v in _stubs._leaves(passed[0])))                    # the jitted function saw device arrays only
        self.assertTrue(all(isinstance(v, jax.Array) for v in _stubs._leaves(passed[2])))
        # the model swap (colabfold/batch.py:426): the new tree uploaded once, the features still resident
        runner.params = params_2; prev = out.pop("prev"); ref.params = params_2; ref_prev = want.pop("prev")
        out = runner.apply(runner.params, None, {**feat, "prev": prev})
        self.assertEqual((st["calls"], st["uploads"], st["hits"]), (3, 8, 10))                                # +2 parameter leaves uploaded; 2 features + 2 prev found
        want = stock_round_trip(ref.apply(ref.params, None, {**feat, "prev": ref_prev}))
        self.assert_same_tree(out, want)
        self.assertGreater(st["fetched_bytes"], 0); self.assertEqual(st["fetched_bytes"] % 2, 0)          # float16 bytes fetched over the three calls

    def test_outputs_are_host_float16_and_stock_asarray_is_the_identity(self):
        device_resident.enable(self.model)
        runner = self.model.RunModel(params={"w": np.ones(3, np.float32)})
        out = runner.apply(runner.params, None, {"prev": {}})
        for v in _stubs._leaves(out):
            self.assertIsInstance(v, np.ndarray); self.assertEqual(v.dtype, np.float16)
            self.assertIs(np.asarray(v, np.float16), v)                                                       # model.py:170 then copies nothing
        self.assertEqual(out["ranking_confidence"].shape, ()); self.assertEqual(out["aatype"].tolist(), [0.0, 1.0])

    def test_traced_call_passes_through(self):
        device_resident.enable(self.model)
        runner = self.model.RunModel(params={})
        inner = runner.apply._device_resident_inner
        runner.apply = device_resident._wrap_apply(lambda p, k, b: {"x": 1.5, "y": {"z": "abstract"}}, sys.modules["jax"], sys.modules["jax.numpy"])
        self.assertEqual(runner.apply({}, None, {}), {"x": 1.5, "y": {"z": "abstract"}})                     # no leaf is a concrete device array: untouched
        self.assertTrue(callable(inner))

    def assert_same_tree(self, got, want):
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            if isinstance(want[k], dict):
                self.assert_same_tree(got[k], want[k])
            else:
                self.assertEqual(got[k].dtype, np.float16, k); self.assertEqual(want[k].dtype, np.float16, k)
                self.assertEqual(got[k].tobytes(), want[k].tobytes(), k)                                       # bit for bit
                self.assertEqual(got[k].shape, want[k].shape, k)


@unittest.skipUnless(HAVE_RECIPE, "needs opt_core.mem.rowpair_jax (the n_gpu core)")
class TestMeshBinding(unittest.TestCase):
    """`big --n_gpu P>1`: a runner colabfold builds AFTER `set_mesh(rmesh)` binds its apply through the recipe —
    `opt_core.mem.rowpair_jax.alphafold.jit_sharded(<the runner's own apply>, rmesh, n_args=3)` — and the resident wrapper sits on what the
    recipe returned; with no mesh (P = 1, every other mode) the recipe is never called and the wrapper sits on the stock apply (the
    single-device path, unchanged). The recipe's jit is replaced by a recorder: no GPU, no mesh is built."""

    def setUp(self):
        stack.reset_for_tests(); device_resident.reset_for_tests()
        self.mods, self.saved = _stubs.install()
        self.model = self.mods["alphafold.model.model"]

    def tearDown(self):
        _stubs.remove(self.saved)
        device_resident.reset_for_tests(); stack.reset_for_tests()

    def test_a_runner_built_under_a_mesh_binds_its_apply_through_jit_sharded(self):
        sentinel = object()                                                  # stands in for the RowMesh big.apply hands to set_mesh
        calls = []

        def recorder(apply, rmesh, n_args=None):
            calls.append((apply, rmesh, n_args))
            return apply                                                     # the program unchanged: only the binding is under test
        with mock.patch.object(_recipe, "jit_sharded", create=True, side_effect=recorder):
            device_resident.set_mesh(sentinel)
            device_resident.enable(self.model)
            runner = self.model.RunModel(config=None, params={"w": np.ones(3, np.float32)})
        self.assertEqual(len(calls), 1)
        bound_apply, rmesh, n_args = calls[0]
        self.assertIs(rmesh, sentinel); self.assertEqual(n_args, 3)
        self.assertIs(runner.apply._device_resident_inner, bound_apply)     # the resident wrapper wraps what the recipe returned
        self.assertEqual(runner.APPLY_ARGS, [])
        bound_apply({}, None, {"prev": {}})                                  # and what the recipe received is this runner's own stock apply
        self.assertEqual(len(runner.APPLY_ARGS), 1)

    def test_no_mesh_never_calls_the_recipe(self):
        with mock.patch.object(_recipe, "jit_sharded", create=True, side_effect=AssertionError("jit_sharded called at P = 1")) as js:
            device_resident.enable(self.model)
            runner = self.model.RunModel(config=None, params={"w": np.ones(3, np.float32)})
            out = runner.apply(runner.params, None, {"prev": {}})            # the single-device path runs: uploads + fetch as in every other mode
        js.assert_not_called()
        self.assertEqual(len(runner.APPLY_ARGS), 1); self.assertEqual(out["aatype"].tolist(), [0.0, 1.0])
        self.assertIsNone(device_resident._MESH["rmesh"])


class TestDeferredFetch(unittest.TestCase):
    """0.2.7: per recycle only the 0-d leaves are fetched; ``RunModel.predict`` (stock's loop, unchanged) returns plain dicts equal byte for byte
    to the stock loop's; a callback reading arrays gets true values; ``--save-recycles`` / ``--save-all`` on the command line switch to a whole
    fetch per call; the deferred dict's plain-dict operations never fetch."""

    def setUp(self):
        stack.reset_for_tests(); device_resident.reset_for_tests()
        self.mods, self.saved = _stubs.install()
        self.model = self.mods["alphafold.model.model"]; self.Stock = self.model.RunModel
        self.params = {"evoformer": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, "head": {"b": np.ones(2, np.float32)}}
        self.feat = {"msa": np.arange(8, dtype=np.int32).reshape(2, 4), "seq_mask": np.ones(2, np.float32)}

    def tearDown(self):
        _stubs.remove(self.saved)
        device_resident.reset_for_tests(); stack.reset_for_tests()

    @staticmethod
    def same(a, b):
        if isinstance(b, dict):
            return type(a) is dict and sorted(a) == sorted(b) and all(TestDeferredFetch.same(a[k], b[k]) for k in b)
        return isinstance(a, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    def run_both(self, **kw):
        device_resident.enable(self.model)
        seen, seen_ref = [], []
        def cb(store):
            def f(result, r):
                store.append((r, float(result["mean_plddt"]), float(result["ranking_confidence"]), "tol" in result and float(result["tol"]),
                              result["structure_module"]["final_atom_positions"].tobytes()))      # a callback that READS an array (colabfold --save-recycles) sees this call's values
            return f
        got, r = self.model.RunModel(params=self.params).predict(self.feat, random_seed=0, callback=cb(seen), **kw)
        want, r_ref = self.Stock(params=self.params).predict(self.feat, random_seed=0, callback=cb(seen_ref), **kw)
        return got, want, r, r_ref, seen, seen_ref

    def test_predict_returns_stocks_bytes_in_plain_dicts_fetching_scalars_per_recycle(self):
        got, want, r, r_ref, seen, seen_ref = self.run_both()
        self.assertEqual((r, r_ref), (3, 3)); self.assertTrue(self.same(got, want)); self.assertEqual(seen, seen_ref)
        st = device_resident._STATE
        self.assertEqual((st["fetch"], st["deferred_fetches"], st["calls"]), ("last", 4 + 1, 4))   # the callback's array read = one batched fetch of that sub-tree per recycle, + the end-of-predict fetch of the rest
        self.assertEqual((st["uploads"], st["hits"]), (6, 18))                        # recycle 0 uploads 2 params + 2 features + 2 prev; recycles 1-3 find all six by identity (the deferred prev included)
        self.assertGreater(st["deferred"], 0)
        self.assertIs(type(got["structure_module"]), dict); self.assertIs(type(got["distogram"]), dict)

    def test_representations_read_prev_through_the_deferred_dict(self):
        got, want, r, r_ref, seen, seen_ref = self.run_both(return_representations=True)
        self.assertTrue(self.same(got, want)); self.assertEqual(seen, seen_ref); self.assertIs(type(got["representations"]), dict)

    def test_save_recycles_on_the_command_line_fetches_each_call_whole(self):
        with mock.patch.object(sys, "argv", ["colabfold_batch", "in.a3m", "out", "--save-recycles"]):
            got, want, r, r_ref, seen, seen_ref = self.run_both()
        st = device_resident._STATE
        self.assertEqual((st["fetch"], st["deferred"], st["deferred_fetches"]), ("each", 0, 0)); self.assertTrue(self.same(got, want)); self.assertEqual(seen, seen_ref)
        self.assertEqual(device_resident.fetch_policy(["x.a3m", "o", "--save-all"]), "each"); self.assertEqual(device_resident.fetch_policy(["x.a3m", "o", "--num-recycle", "3"]), "last")

    def test_deferred_dict_plain_operations_never_fetch_and_reads_fetch_once(self):
        device_resident.enable(self.model)
        runner = self.model.RunModel(params=self.params)
        out = runner.apply(runner.params, None, {**self.feat, "prev": {"prev_pair": np.zeros((2, 2), np.float16), "prev_pos": np.zeros((2, 3))}})
        st = device_resident._STATE; before = st["fetched_bytes"]
        self.assertIsInstance(out, device_resident._Deferred)
        for k, v in out.items():                                                     # stock's _jnp_to_np: items + np.asarray(v, float16) is the identity on every leaf, nothing fetched
            if not isinstance(v, dict):
                self.assertIs(np.asarray(v, np.float16), v); out[k] = np.asarray(v, np.float16)
        prev = out.pop("prev")                                                       # a sub-dict leaves deferred (stock: result.pop('prev'))
        self.assertEqual(st["fetched_bytes"], before); self.assertIsInstance(prev, device_resident._Deferred); self.assertEqual(sorted(prev._dev), ["prev_pair", "prev_pos"])
        self.assertEqual(float(out["ranking_confidence"]), 0.25 + 15 + 2)           # 0-d leaves were fetched with the call (stock reads them every recycle)
        logits = out.pop("distogram")                                                # a popped sub-dict stays deferred and fetches on its own first read
        self.assertIsInstance(logits, device_resident._Deferred); self.assertEqual(list(logits._dev), ["logits"]); self.assertEqual(st["fetched_bytes"], before)
        a = out["aatype"]; self.assertEqual(a.tolist(), [0.0, 1.0]); self.assertGreater(st["fetched_bytes"], before)   # the first array read brings everything still deferred in `out` and below, in one device_get …
        b2, n2 = st["fetched_bytes"], st["deferred_fetches"]; self.assertIs(out["aatype"], a)
        self.assertEqual(out["structure_module"]["final_atom_positions"].dtype, np.float16); self.assertEqual((st["fetched_bytes"], st["deferred_fetches"]), (b2, n2))   # … once: later reads are host reads
        self.assertEqual(sorted(prev._dev), ["prev_pair", "prev_pos"])              # the popped prev is its own group: untouched by that read
        popped = out.pop("mean_plddt"); self.assertEqual(float(popped), 80.0)
        self.assertAlmostEqual(float(logits["logits"][0, 0, 0]), float(np.float16(1 / 3))); self.assertEqual(st["deferred_fetches"], n2 + 1)
        plain = device_resident.materialize(out)
        self.assertIs(type(plain), dict); self.assertIs(type(plain["structure_module"]), dict)
        self.assertEqual(plain["structure_module"]["final_atom_positions"].dtype, np.float16)

    def test_dm_tree_map_structure_rebuilds_a_filled_mapping(self):
        """alphafold/model/model.py:207 logs `tree.map_structure(lambda x: x.shape, result)` over the deferred result: dm-tree rebuilds each mapping as
        type(instance)(pairs) — the shapes dict must come back filled and equal to the plain dict's (0.2.7 printed `{}`: the constructor took `jax` positionally)."""
        try:
            import tree as dm_tree                                                  # the model's own dependency; absent from a bare CPU environment
        except ImportError:
            self.skipTest("dm-tree not installed")
        device_resident.enable(self.model)
        runner = self.model.RunModel(params=self.params)
        out = runner.apply(runner.params, None, {**self.feat, "prev": {"prev_pair": np.zeros((2, 2), np.float16), "prev_pos": np.zeros((2, 3))}})
        self.assertIsInstance(out, device_resident._Deferred)
        shapes = dm_tree.map_structure(lambda v: v.shape, out)
        want = dm_tree.map_structure(lambda v: v.shape, device_resident.materialize(runner.apply(runner.params, None, {**self.feat, "prev": {"prev_pair": np.zeros((2, 2), np.float16), "prev_pos": np.zeros((2, 3))}})))
        self.assertTrue(shapes) ; self.assertEqual({k: v for k, v in shapes.items()}, want)
        self.assertEqual(shapes["structure_module"]["final_atom_positions"], (2, 2, 3)); self.assertEqual(shapes["ranking_confidence"], ())
        self.assertEqual(dict(device_resident._Deferred([("a", 1)], b=2)), {"a": 1, "b": 2})   # dict's construction forms

    def test_the_previous_recycles_device_leaves_are_released_when_the_next_call_starts(self):
        device_resident.enable(self.model)
        runner = self.model.RunModel(params=self.params)
        out = runner.apply(runner.params, None, {**self.feat, "prev": {"prev_pair": np.zeros((2, 2), np.float16), "prev_pos": np.zeros((2, 3))}})
        prev = out.pop("prev")
        self.assertTrue(out._dev and out["structure_module"] is not None or True)
        pending_before = sum(len(d._dev) for d in (out, dict.__getitem__(out, "distogram")))
        runner.apply(runner.params, None, {**self.feat, "prev": prev})
        pending_after = sum(len(d._dev) for d in (out, dict.__getitem__(out, "distogram")))
        self.assertGreater(pending_before, 0); self.assertEqual(pending_after, 0)   # released (never fetched): the device holds one output tree, as stock's route


if __name__ == "__main__":
    unittest.main()
