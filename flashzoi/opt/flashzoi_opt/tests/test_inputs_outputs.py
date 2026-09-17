"""The one item loader's rules (inputs.py) and the one output schema (outputs.py)."""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from flashzoi_opt import inputs, outputs


def _onehot(seed=0, n=inputs.SEQ_LEN, dtype=np.uint8, with_n=True):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, n)
    x = np.zeros((4, n), dtype=dtype)
    x[idx, np.arange(n)] = 1
    if with_n:
        x[:, :10] = 0
    return x


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_directory_sorted_and_stems(self):
        for name in ("b", "a", "c"):
            np.save(os.path.join(self.d, f"{name}.npy"), _onehot())
        open(os.path.join(self.d, "notes.txt"), "w").write("x")
        items = inputs.list_items(self.d)
        self.assertEqual([i for i, _ in items], ["a", "b", "c"])

    def test_json_item_list_refused_and_missing_dir(self):
        np.save(os.path.join(self.d, "w.npy"), _onehot())
        spec = os.path.join(self.d, "items.json")
        json.dump([{"id": "u1", "path": "w.npy"}], open(spec, "w"))
        with self.assertRaises(inputs.InputError):                    # --input is a directory of <item>.npy (the stock caller's form); no item-list alternate
            inputs.list_items(spec)
        with self.assertRaises(inputs.InputError):
            inputs.list_items(os.path.join(self.d, "nothing"))
        os.makedirs(os.path.join(self.d, "empty"))
        with self.assertRaises(inputs.InputError):
            inputs.list_items(os.path.join(self.d, "empty"))

    def test_dtypes_values_shape(self):
        p = os.path.join(self.d, "x.npy")
        for dt in (np.uint8, np.bool_, np.float32, np.int64):
            np.save(p, _onehot(dtype=dt))
            u = inputs.load_onehot(p)
            self.assertEqual((u.dtype, u.shape), (np.uint8, (4, inputs.SEQ_LEN))); self.assertTrue(u.flags["C_CONTIGUOUS"])
            self.assertEqual(int(u.sum()), inputs.SEQ_LEN - 10)
        np.save(p, _onehot()[:, :100])
        with self.assertRaises(inputs.InputError):
            inputs.load_onehot(p)
        bad = _onehot(dtype=np.float32); bad[0, 0] = 0.5; np.save(p, bad)
        with self.assertRaises(inputs.InputError):
            inputs.load_onehot(p)
        two = _onehot(); two[:, 20] = 1; np.save(p, two)
        with self.assertRaises(inputs.InputError):
            inputs.load_onehot(p)

    def test_npz_key_x(self):
        p = os.path.join(self.d, "c.npz")
        np.savez(p, x=_onehot())
        self.assertEqual(inputs.load_onehot(p).shape, (4, inputs.SEQ_LEN))
        np.savez(p, y=_onehot())
        with self.assertRaises(inputs.InputError):
            inputs.load_onehot(p)


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_item_rows_and_file(self):
        y = np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3).swapaxes(3, 2)                 # the stock's strided view: written in C order all the same
        row = outputs.write_item(self.d, "it", y)
        self.assertEqual(row["shape"], [1, 2, 3, 4]); self.assertEqual(row["dtype"], "float32"); self.assertNotIn("sha256", row)   # no per-item digest in the loop: the file carries the bytes
        self.assertEqual(sorted(dict(row, wall_s=0.1, ok=True)), ["dtype", "item", "ok", "shape", "wall_s"])   # the stock caller's row schema exactly (stock/pred.py:308) — no key of the package's own
        back = np.load(os.path.join(self.d, "it.npy"))
        self.assertTrue(np.array_equal(back, y)) and self.assertTrue(back.flags["C_CONTIGUOUS"])
        self.assertEqual(outputs.sha256_array(np.asfortranarray(y)), outputs.sha256_array(back))          # C-contiguous bytes, whatever the layout
        outputs.append_row(self.d, dict(row, wall_s=0.1, ok=True)); outputs.append_row(self.d, {"item": "x", "ok": False, "error": "e"})
        rows = outputs.read_rows(self.d)
        self.assertEqual([r["item"] for r in rows], ["it", "x"]); self.assertEqual(rows[0]["shape"], [1, 2, 3, 4])
        p = outputs.write_json(self.d, "j.json", {"a": 1})
        self.assertEqual(json.load(open(p)), {"a": 1})


if __name__ == "__main__":
    unittest.main()
