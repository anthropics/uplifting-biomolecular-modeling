"""Lever `hoist_prev` (hoist_prev.py): the recycle features stay on the device between design steps; exact.

CPU tests (no jax): the stock bodies the lever re-states are the pinned ones (a tripwire on `_af_design._recycle` / `run` in the vendored
tree — a changed body upstream must be re-read into hoist_prev.py); the `aux["all"]["prev"]` view gives stock's stacked values without a
copy; the LEVER line parses under the kit's evidence grammar. GPU test (real jax + colabdesign at the pin + AF2 params under
COLABDESIGN_PARAMS_DIR; skipped by name otherwise): two design steps of BindCraft's binder model with and without the lever from one seed —
every loss, the gradient, the sequence parameters, and `np.asarray` of every `prev` leaf are byte-identical; the leaves are device arrays.
"""
import ast
import hashlib
import importlib.util
import os
import pickle
import unittest

import numpy as np

from colabdesign_opt import hoist_prev
from colabdesign_opt.tests import _stubs

STOCK_DESIGN_PY = os.path.join(_stubs.TREE, "stock", "src", "colabdesign", "af", "design.py")     # the vendored ColabDesign design loop (stock/PINS.json extracted list)
PINNED = hoist_prev.PINNED                                   # sha256 of the pinned bodies (colabdesign 1.1.3 @ e31a56fe) the module re-states / relies on
_body = hoist_prev.body_source


@unittest.skipUnless(os.path.isfile(STOCK_DESIGN_PY), "vendored colabdesign/af/design.py not present around the package")
class TestStockBodiesArePinned(unittest.TestCase):
    def test_the_restated_bodies_are_the_pinned_ones(self):
        """hoist_prev.py re-states `_af_design._recycle` and `run` with two changes (device zeros; prev off the host path for one model).
        If upstream's bodies change, this fails first: re-read them into hoist_prev.py, then update PINNED."""
        with open(STOCK_DESIGN_PY, "r", encoding="utf-8") as fh:
            src = fh.read()
        for name, digest in PINNED.items():
            self.assertEqual(hashlib.sha256(_body(src, name).encode()).hexdigest(), digest, f"stock `{name}` changed under the lever (colabdesign/af/design.py)")

    def test_install_refuses_by_name_on_another_design_loop(self):
        """require_pinned: a module whose file lacks a body, or carries another body, is a named refusal (cannot_run) — the stand-in stack's design surface included."""
        import tempfile, types
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "design.py")
            open(p, "w").write("class _af_design:\n    def run(self):\n        return 1\n")
            m = types.ModuleType("fake_design"); m.__file__ = p
            with self.assertRaises(hoist_prev.HoistPrevError) as cm:
                hoist_prev.require_pinned(m)
            self.assertTrue(getattr(cm.exception, "cannot_run", False)); self.assertIn("defines no `_recycle`", str(cm.exception))
            src = open(STOCK_DESIGN_PY).read().replace("aux = self._single(model_params, backprop)", "aux = self._single(model_params,  backprop)", 1)
            open(p, "w").write(src)
            with self.assertRaises(hoist_prev.HoistPrevError) as cm:
                hoist_prev.require_pinned(m)
            self.assertIn("is not the pinned body", str(cm.exception))
            m.__file__ = STOCK_DESIGN_PY
            self.assertEqual(hoist_prev.require_pinned(m), PINNED)
        self.assertEqual(hoist_prev.REFUSALS, (hoist_prev.HoistPrevError,))

    def test_the_module_names_what_it_patches(self):
        self.assertEqual(hoist_prev.PATCHED, ("_recycle", "run"))
        self.assertEqual(set(hoist_prev.STOCK_SOURCE), set(hoist_prev.PATCHED))
        self.assertEqual(hoist_prev.LEVER, "hoist_prev"); self.assertTrue(hoist_prev.IMPL.startswith("hoist_prev@"))


class TestCanonicalZeros(unittest.TestCase):
    """Stock's host mean over one model turns −0.0 into +0.0 and keeps every other float32 bit pattern; canonical_zeros does exactly that on the
    device (real jax probed here, not at import — the stand-in suite shares the process)."""
    def test_negative_zero_only(self):
        try:
            import jax.numpy as jnp
            jnp.zeros(1)
        except Exception as e:  # noqa: BLE001 — the stand-in jax of the CPU suite
            raise unittest.SkipTest(f"canonical_zeros needs real jax: {e!r}")
        x = np.array([[-0.0, 0.0, 1.5, -2.25, np.float32(1e-45), -np.float32(1e-45), np.inf, -np.inf]], np.float32)
        stock = np.stack([x]).mean(0)                                                       # what design.py's run() does to a prev leaf of one model
        ours = np.asarray(hoist_prev.canonical_zeros({"p": jnp.asarray(x)})["p"])
        self.assertEqual(stock.tobytes(), ours.tobytes()); self.assertEqual(ours.tobytes()[:4], np.float32(0.0).tobytes())
        self.assertNotEqual(x.tobytes(), stock.tobytes())                                  # the raw array differs from stock's mean exactly in the −0.0
        i = np.array([[3, 0]], np.int32)
        self.assertEqual(np.asarray(hoist_prev.canonical_zeros({"i": jnp.asarray(i)})["i"]).tobytes(), i.tobytes())


class TestAllPrevView(unittest.TestCase):
    """`aux["all"]["prev"]`: stock holds np.stack over ONE model = the arrays with a leading axis of one; the view gives exactly that on access, no copy held."""
    def test_access_gives_the_model_axis(self):
        pair = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        v = hoist_prev._AllPrev({"prev_pair": pair, "prev_pos": np.zeros((2, 37, 3), np.float32)})
        self.assertEqual(v["prev_pair"].shape, (1, 2, 2, 3))
        self.assertTrue(np.array_equal(v["prev_pair"], np.stack([pair])))
        self.assertTrue(np.array_equal(v["prev_pair"].mean(0), pair))                      # stock's aux["prev"] = the mean over the model axis
        self.assertEqual(sorted(v.keys()), ["prev_pair", "prev_pos"]); self.assertEqual([k for k, _ in v.items()], list(v.keys()))
        self.assertEqual(v.values()[0].shape, (1, 2, 2, 3)); self.assertIsNone(v.get("absent")); self.assertEqual(v.get("prev_pos").shape, (1, 2, 37, 3))
        self.assertIsInstance(v, dict)
        w = pickle.loads(pickle.dumps(v))                                                     # BindCraft's save_trajectory_pickle path: round-trips, same view
        self.assertTrue(np.array_equal(w["prev_pair"], v["prev_pair"]))


class TestLeverLine(unittest.TestCase):
    def test_the_line_parses_under_the_evidence_grammar(self):
        from colabdesign_opt import evidence
        line = hoist_prev.line("install")
        (d,) = evidence.lever_lines([line])
        self.assertEqual((d["name"], d["state"], d["impl"], d["origin"], d["patched"], d["source"]), ("hoist_prev", "on", hoist_prev.IMPL, "kit", "_recycle,run", "install"))
        for k in ("zero_builds", "zero_inits", "device_prev_steps", "multi_model_stock_path", "run_calls", "pid"):
            self.assertIn(k, d); int(d[k])
        self.assertNotIn(" reason=", line)


def _real_stack():
    """(ok, why): real jax with a GPU, colabdesign importable (not the tests' stand-in), AF2 params under COLABDESIGN_PARAMS_DIR."""
    try:
        import jax
        if not hasattr(jax, "jit"):
            return False, "stand-in jax"
        if jax.default_backend() != "gpu":
            return False, "no GPU backend"
        spec = importlib.util.find_spec("colabdesign.af.design")
        if spec is None:
            return False, "colabdesign not importable"
        from colabdesign.af import design as D
        if not hasattr(D._af_design, "_recycle"):
            return False, "stand-in colabdesign"
    except Exception as e:  # noqa: BLE001
        return False, repr(e)
    root = os.environ.get("COLABDESIGN_PARAMS_DIR", "")
    if not os.path.isfile(os.path.join(root, "params", "params_model_1_multimer_v3.npz")):
        return False, "COLABDESIGN_PARAMS_DIR holds no AF2 multimer params"
    return True, ""


class TestExactOnDevice(unittest.TestCase):
    """Stock vs lever from one seed on BindCraft's example target: byte-identical steps; prev leaves are device arrays with stock's bytes.
    The real-stack probe runs here, not at import: importing this module must not import jax or create a backend (the suite's stand-in tests
    and lever parcompile share the process under `unittest discover`)."""
    STEPS = 2

    @classmethod
    def setUpClass(cls):
        ok, why = _real_stack()
        if not ok:
            raise unittest.SkipTest(f"GPU exactness test needs the real stack: {why}")

    def _build(self):
        from colabdesign import mk_afdesign_model, clear_mem
        clear_mem()
        af = mk_afdesign_model(protocol="binder", debug=False, data_dir=os.environ["COLABDESIGN_PARAMS_DIR"], use_multimer=True, num_recycles=1, best_metric="loss")
        af.prep_inputs(pdb_filename=_stubs.EXAMPLE_PDB, chain="A", binder_len=24, hotspot=None, seed=0, rm_aa="C", rm_target_seq=False, rm_target_sc=False)
        return af

    def _steps(self, af):
        rec = []
        for _ in range(self.STEPS):
            af.step(lr_scale=1.0, num_recycles=1, num_models=1, sample_models=True, models=[0, 1], backprop=True, save_best=True, verbose=0)
            rec.append({"log": dict(af.aux["log"]), "grad": np.array(af.aux["grad"]["seq"]), "seq": np.array(af._params["seq"]),
                        "prev": {k: np.asarray(v) for k, v in af.aux["prev"].items()}, "all_prev": {k: np.asarray(af.aux["all"]["prev"][k]) for k in af.aux["prev"]},
                        "types": {k: type(v).__name__ for k, v in af.aux["prev"].items()},
                        "inputs": {k: np.asarray(v) for k, v in af._inputs.items() if hasattr(v, "shape")}})
        return rec

    def test_two_steps_are_byte_identical_and_prev_lives_on_the_device(self):
        import jax
        from colabdesign.af.design import _af_design
        self.assertFalse(hoist_prev.installed())
        stock_recycle, stock_run = _af_design._recycle, _af_design.run
        stock = self._steps(self._build())
        try:
            hoist_prev.install(); self.assertTrue(hoist_prev.installed())
            lever = self._steps(self._build())
            census = hoist_prev.census()
        finally:
            _af_design._recycle, _af_design.run = stock_recycle, stock_run           # leave the class as found for the rest of the session
        self.assertEqual((census["zero_builds"], census["zero_inits"], census["device_prev_steps"], census["multi_model_stock_path"]), (1, self.STEPS, self.STEPS, 0))
        for i, (s, l) in enumerate(zip(stock, lever)):
            self.assertEqual(s["log"], l["log"], f"step {i} log")
            for k in ("grad", "seq"):
                self.assertEqual(s[k].dtype, l[k].dtype); self.assertEqual(s[k].tobytes(), l[k].tobytes(), f"step {i} {k}")
            self.assertEqual(sorted(s["prev"]), sorted(l["prev"]))
            for k in s["prev"]:
                self.assertEqual(s["types"][k], "ndarray"); self.assertIn("Array", l["types"][k])            # the one observable: device arrays, not numpy
                self.assertEqual((s["prev"][k].dtype, s["prev"][k].shape), (l["prev"][k].dtype, l["prev"][k].shape), k)
                self.assertEqual(s["prev"][k].tobytes(), l["prev"][k].tobytes(), f"step {i} prev {k}")
                self.assertEqual(s["all_prev"][k].shape, l["all_prev"][k].shape); self.assertEqual(s["all_prev"][k].tobytes(), l["all_prev"][k].tobytes(), f"step {i} all.prev {k}")
            self.assertEqual(sorted(s["inputs"]), sorted(l["inputs"]))
            for k in s["inputs"]:
                self.assertEqual(s["inputs"][k].tobytes(), l["inputs"][k].tobytes(), f"step {i} _inputs {k}")
        z = hoist_prev.zero_prev(int(stock[0]["prev"]["prev_pos"].shape[0]), False)
        self.assertIs(hoist_prev.zero_prev(int(stock[0]["prev"]["prev_pos"].shape[0]), False), z)   # made once, reused
        self.assertEqual({k: (str(v.dtype), bool((np.asarray(v) == 0).all())) for k, v in z.items()},
                         {"prev_msa_first_row": ("float32", True), "prev_pair": ("float32", True), "prev_pos": ("float32", True)})
        self.assertTrue(all(isinstance(v, jax.Array) for v in z.values()))


if __name__ == "__main__":
    unittest.main()
