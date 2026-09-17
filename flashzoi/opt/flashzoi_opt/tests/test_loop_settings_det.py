"""The settings (torch's TF32 defaults untouched, the other stock defaults explicit), the deterministic recipe's environment rule, and the prediction loop with a stub helper:
the documented call bound by name, the output asserted, the item written verbatim, the row, the per-item line, the counters; a failed
item is a named row."""
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from flashzoi_opt import det, loop, outputs, report
from flashzoi_opt import settings as S
from flashzoi_opt.tests import _stubs


class TestSettingsDet(unittest.TestCase):
    def test_settings_defaults(self):
        st = S.DEFAULT
        self.assertEqual(st.replicates, (0, 1, 2, 3)); self.assertEqual(st.repos(), [f"johahi/flashzoi-replicate-{k}" for k in range(4)])
        self.assertEqual((st.batch, st.slices, st.autocast, st.device), (1, "all", "cuda", "cuda"))
        self.assertIsNone(st.matmul_allow_tf32); self.assertIsNone(st.cudnn_allow_tf32); self.assertFalse(st.cudnn_benchmark); self.assertFalse(st.cudnn_deterministic)   # the TF32 pair: torch's own defaults, never set
        self.assertEqual(st.track_slice(), slice(None))
        self.assertEqual(list(S.Settings(slices="89").track_slice()), [89]); self.assertEqual(S.Settings(slices="89").n_tracks(), 1)
        self.assertEqual(list(S.Settings(slices="5-7,2,6").track_slice()), [2, 5, 6, 7]); self.assertEqual(S.Settings(slices="5-7,2,6").output_shape(4), (1, 4, S.OUTPUT_SHAPE[2], 4))
        for bad in ("7611", "3-2", "-1", "x", ","):
            with self.assertRaises(ValueError):
                S.Settings(slices=bad).track_slice()
        d = st.as_dict()
        self.assertEqual(d["output_shape"], [1, 4, 6144, 7611]); self.assertEqual(d["output_dtype"], "float32"); self.assertEqual(d["seed"], 0)

    def test_apply_numerics_reads_back(self):
        _stubs.require_upstream()
        import torch
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        rb = S.apply_numerics(S.DEFAULT)
        self.assertEqual((rb["matmul_allow_tf32"], rb["cudnn_allow_tf32"]), before)                    # the stock's numerics: the TF32 pair untouched
        self.assertFalse(rb["deterministic_algorithms"]); self.assertIn("autocast_gpu_dtype", rb)
        torch.backends.cudnn.allow_tf32 = not before[1]
        try:
            self.assertEqual(S.apply_numerics(S.DEFAULT)["cudnn_allow_tf32"], not before[1])           # whatever the process has stays
        finally:
            torch.backends.cudnn.allow_tf32 = before[1]

    def test_det_env_rule(self):
        env = {}
        self.assertEqual(det.export_env(env), {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        det.export_env(env); self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
        self.assertEqual(det.RECIPE["seed"], 0); self.assertTrue(det.RECIPE["deterministic_algorithms"]); self.assertNotIn("matmul_allow_tf32", det.RECIPE); self.assertEqual(det.RECIPE["tf32"], "untouched (torch's defaults)")

    def test_det_apply_on_cpu(self):
        _stubs.require_upstream()
        import torch
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        saved = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            rb = det.apply()
            self.assertTrue(rb["deterministic_algorithms"]); self.assertTrue(rb["cudnn_deterministic"]); self.assertEqual(rb["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual((rb["matmul_allow_tf32"], rb["cudnn_allow_tf32"]), before)          # the TF32 pair: untouched even under the deterministic recipe
        finally:
            torch.use_deterministic_algorithms(False); torch.backends.cudnn.deterministic = False
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            if saved is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved


class TestLoopStubbed(unittest.TestCase):
    """predict_tracks replaced by a stub on the upstream helper module (the loop binds the name at call time); a small output shape."""

    def setUp(self):
        _stubs.require_upstream()
        import borzoi_pytorch.pytorch_borzoi_helpers as H
        self.H = H; self._pt = H.predict_tracks
        self._shape = S.OUTPUT_SHAPE
        S.OUTPUT_SHAPE = (1, 2, 3, 5)
        self.calls = []

        def stub(models, x, slices):
            self.calls.append((len(models), tuple(x.shape), str(x.dtype), slices))
            if getattr(models[0], "fail", False):
                raise RuntimeError("boom")
            return np.full((1, len(models), 3, 5), float(x.sum().item()), dtype=np.float32)[..., slices]
        H.predict_tracks = stub
        self.d = tempfile.mkdtemp(); self.out = tempfile.mkdtemp()
        report.ITEMS.update(items=0, ok=0, failed=0)

    def tearDown(self):
        self.H.predict_tracks = self._pt; S.OUTPUT_SHAPE = self._shape
        shutil.rmtree(self.d, ignore_errors=True); shutil.rmtree(self.out, ignore_errors=True)

    def _items(self, n=2):
        items = []
        for i in range(n):
            x = np.zeros((4, S.SEQ_LEN), dtype=np.uint8); x[i % 4, :] = 1
            p = os.path.join(self.d, f"u{i}.npy"); np.save(p, x); items.append((f"u{i}", p))
        return items

    def test_loop_writes_rows_lines_counts(self):
        import types
        models = [types.SimpleNamespace(), types.SimpleNamespace()]
        err = io.StringIO(); se = sys.stderr; sys.stderr = err
        try:
            counts = loop.run_items(models, self._items(2), self.out, S.DEFAULT, device="cpu")
        finally:
            sys.stderr = se
        self.assertEqual(counts, {"items": 2, "ok": 2, "failed": 0}); self.assertEqual(report.ITEMS, {"items": 2, "ok": 2, "failed": 0})
        self.assertEqual(self.calls[0][:3], (2, (4, S.SEQ_LEN), "torch.float32")); self.assertEqual(self.calls[0][3], slice(None))
        rows = outputs.read_rows(self.out)
        self.assertEqual([r["item"] for r in rows], ["u0", "u1"]); self.assertTrue(all(r["ok"] for r in rows))
        y0 = np.load(os.path.join(self.out, "u0.npy"))
        self.assertEqual(y0.shape, (1, 2, 3, 5)); self.assertEqual(rows[0]["sha256"], outputs.sha256_array(y0)); self.assertIn("wall_s", rows[0])
        lines = [ln for ln in err.getvalue().splitlines() if " pred " in ln]
        self.assertEqual(len(lines), 2); self.assertRegex(lines[0], r"^\[flashzoi-opt\] pred u0 s0 samples=1 \d+\.\d{3}s$")

    def test_tracks_subset_reaches_the_documented_call(self):
        """Settings(slices=...) — pred --tracks — hands the int64 index array to predict_tracks as `slices` and the item is the subset."""
        import types
        models = [types.SimpleNamespace()]
        err = io.StringIO(); se = sys.stderr; sys.stderr = err
        try:
            counts = loop.run_items(models, self._items(1), self.out, S.Settings(slices="1,3-4"), device="cpu")
        finally:
            sys.stderr = se
        self.assertEqual(counts, {"items": 1, "ok": 1, "failed": 0}, outputs.read_rows(self.out))
        self.assertEqual([list(c[3]) for c in self.calls], [[1, 3, 4]]); self.assertEqual(str(np.asarray(self.calls[0][3]).dtype), "int64")
        self.assertEqual(np.load(outputs.item_path(self.out, "u0")).shape, (1, 1, 3, 3)); self.assertEqual(outputs.read_rows(self.out)[0]["shape"], [1, 1, 3, 3])

    def test_failed_item_is_a_named_row(self):
        import types
        models = [types.SimpleNamespace(fail=True)]
        err = io.StringIO(); se = sys.stderr; sys.stderr = err
        try:
            counts = loop.run_items(models, self._items(1), self.out, S.DEFAULT, device="cpu")
        finally:
            sys.stderr = se
        self.assertEqual(counts, {"items": 1, "ok": 0, "failed": 1})
        rows = outputs.read_rows(self.out)
        self.assertEqual(rows[0]["ok"], False); self.assertIn("RuntimeError: boom", rows[0]["error"]); self.assertFalse(os.path.exists(os.path.join(self.out, "u0.npy")))

    def test_output_drift_refuses(self):
        import types
        S.OUTPUT_SHAPE = (1, 1, 9, 9)
        with self.assertRaises(loop.OutputDrift):
            loop.predict_item([types.SimpleNamespace()], np.zeros((4, S.SEQ_LEN), np.uint8), S.DEFAULT, device="cpu")


if __name__ == "__main__":
    unittest.main()
