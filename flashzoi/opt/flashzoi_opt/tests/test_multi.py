"""`pred --jobs <file>`: the jobs reader (the line grammar, relative paths, the refusals by name), the CLI refusals (exit 3 with the
NOT ACTIVE line: --input / --out / --items beside --jobs, --mode off with --jobs, an unreadable jobs file — before any load), the
plain call unchanged without --jobs (--input / --out still required, exit 2), the ` multi=<n>` suffix of the ACTIVE line, and the job
loop with the stub helper: every job into its own directory with the plain call's files, one item line per job, the manifest's block."""
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

from flashzoi_opt import cli, modes, report, stack
from flashzoi_opt.tests import _stubs


def _run(argv):
    err, out = io.StringIO(), io.StringIO()
    se, so = sys.stderr, sys.stdout
    sys.stderr, sys.stdout = err, out
    try:
        rc = cli.main(argv)
    finally:
        sys.stderr, sys.stdout = se, so
    return rc, err.getvalue(), out.getvalue()


class _Env:
    """Set / restore the mode variable around a test."""

    def __init__(self, **kv):
        self.kv = kv; self.saved = {}

    def __enter__(self):
        for k in (modes.ENV_MODE,):
            self.saved[k] = os.environ.pop(k, None)
        for k, v in self.kv.items():
            os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _onehot(i: int) -> np.ndarray:
    x = np.zeros((4, 524288), dtype=np.uint8); x[i % 4, :] = 1
    return x


def _jobs_file(d: str, n_jobs: int, items_per_job: int, name: str = "jobs.tsv") -> str:
    """A jobs file under d: job k = an input directory in_k/ of items_per_job windows, an output directory out_k/ (relative paths)."""
    lines = ["# jobs", ""]
    for k in range(n_jobs):
        inp = os.path.join(d, f"in_{k}"); os.makedirs(inp, exist_ok=True)
        for j in range(items_per_job):
            np.save(os.path.join(inp, f"w{k}_{j}.npy"), _onehot(k + j))
        lines.append(f"in_{k}\tout_{k}")
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


class TestReader(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_reads_jobs_in_order_relative_to_the_file(self):
        p = _jobs_file(self.d, 3, 1)
        jobs = stack.read_multi_jobs(p)
        self.assertEqual([os.path.basename(i) for i, _ in jobs], ["in_0", "in_1", "in_2"])
        self.assertEqual([os.path.basename(o) for _, o in jobs], ["out_0", "out_1", "out_2"])
        self.assertTrue(all(os.path.isabs(i) and os.path.isabs(o) for i, o in jobs))

    def test_refusals_by_name(self):
        for text, word in (("in_0 out_0\n", "input<TAB>out"), ("in_0\tout_0\tx\n", "input<TAB>out"), ("\tout_0\n", "input<TAB>out"),
                           ("missing\tout_0\n", "does not exist"), ("in_0\tout_0\nin_0\tout_0\n", "named twice"), ("# nothing\n\n", "no jobs")):
            os.makedirs(os.path.join(self.d, "in_0"), exist_ok=True)
            p = os.path.join(self.d, "j.tsv")
            with open(p, "w") as fh:
                fh.write(text)
            with self.assertRaises(stack.ActivationError) as cm:
                stack.read_multi_jobs(p)
            self.assertIn(word, str(cm.exception), text)
        with self.assertRaises(stack.ActivationError) as cm:
            stack.read_multi_jobs(os.path.join(self.d, "absent.tsv"))
        self.assertIn("not a file", str(cm.exception))


class TestPackageEnv(unittest.TestCase):
    def test_package_env_is_the_mode_switch_only(self):
        self.assertEqual(stack.PACKAGE_ENV, (modes.ENV_MODE,))
        with _Env(**{modes.ENV_MODE: "exact"}):
            e = cli.stock_env(det=False)
        self.assertNotIn(modes.ENV_MODE, e)


class TestCliRefusals(unittest.TestCase):
    def setUp(self):
        _stubs.reset_activation()
        self.d = tempfile.mkdtemp()
        self.jobs = _jobs_file(self.d, 2, 1)

    def tearDown(self):
        _stubs.reset_activation()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_plain_call_unchanged_without_jobs(self):
        rc, err, _ = _run(["pred", "--mode", "exact"])
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("pred needs --input and --out", err)
        rc, err, _ = _run(["pred", "--mode", "off"])
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("pred needs --input and --out", err)

    def test_line_flags_and_off_exit_3(self):
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", self.jobs, "--input", self.d, "--out", self.d])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("--jobs with --input, --out on the line", err)
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", self.jobs, "--items", "a"])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("--jobs with --items on the line", err)
        import types
        ran = []; real_run, real_script = cli.subprocess.run, cli.stock_script
        cli.subprocess.run = lambda cmd, env=None: (ran.append(cmd), types.SimpleNamespace(returncode=(1 if len(ran) == 2 else 0)))[1]
        cli.stock_script = lambda: "/tree/opt/flashzoi_opt/stock_pred.py"
        try:                                                                               # --jobs under off: one clean stock process per job, in file order; rc = the worst job's
            rc, err, _ = _run(["pred", "--mode", "off", "--jobs", self.jobs, "--tracks", "0-9"])
        finally:
            cli.subprocess.run, cli.stock_script = real_run, real_script
        self.assertEqual(rc, 1); self.assertEqual(len(ran), 2)
        self.assertEqual([c[c.index("--out") + 1] for c in ran], [os.path.join(self.d, f"out_{i}") for i in range(2)])
        self.assertTrue(all(c[c.index("--tracks") + 1] == "0-9" and "--items" not in c for c in ran))

    def test_bad_jobs_file_refuses_before_any_load(self):
        bad = os.path.join(self.d, "bad.tsv")
        with open(bad, "w") as fh:
            fh.write("in_0\tout_0\nnowhere\tout_1\n")
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", bad])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("does not exist", err)
        empty = os.path.join(self.d, "in_empty"); os.makedirs(empty)
        with open(bad, "w") as fh:
            fh.write("in_empty\tout_9\n")
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", bad])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("no items", err)
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", os.path.join(self.d, "absent.tsv")])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("not a file", err)
        self.assertFalse(stack.status().get("active"))                       # refused before activation: nothing armed

    def test_check_has_no_suffix(self):
        kit = _stubs.require_kit()
        t = _stubs.Tree(kit=kit).enter()
        undo = _stubs.stub_box()
        try:
            rc, err, _ = _run(["check", "--mode", "exact"])                    # the dry run: the DRY-RUN line, no job count
            self.assertEqual(rc, cli.EXIT_OK, err); self.assertNotIn("multi=", err)
        finally:
            undo(); t.exit()

    def test_lines(self):
        rep = {"mode": "exact", "components_planned": ["a"], "components_unavailable": [], "gpu": {"name": "g", "cc": "9.0"}, "multi": 4}
        self.assertTrue(report.activation_line(rep).endswith(" (partial: none) multi=4"))
        self.assertFalse(report.activation_line(dict(rep, multi=None)).endswith("multi=4"))
        self.assertEqual(report.multi_item_line(2, 4, "/o/j2", {"items": 9, "ok": 9, "failed": 0}, 1.5), "[flashzoi-opt multi] item 2/4 /o/j2 items=9 ok=9 failed=0 wall=1.500s rc=0")
        self.assertTrue(report.multi_item_line(1, 1, "x", {"items": 1, "ok": 0, "failed": 1}, 0.1).endswith(" rc=1"))
        self.assertEqual(report.multi_ready_line("exact", 19.5, 4), "[flashzoi-opt multi] ready mode=exact load=19.500s jobs=4")
        self.assertEqual(report.multi_done_line(4, 4, 0, 19.5, 3.25), "[flashzoi-opt multi] DONE jobs=4 ok=4 failed=0 load=19.500s wall=3.250s")


class TestJobLoop(unittest.TestCase):
    """The loop over jobs with the stub helper (torch + borzoi-pytorch importable; the models are stand-ins): every job into its own
    directory with the one-process route's files, the per-item rows, one item line per job, the after_job hook, the totals."""

    def setUp(self):
        _stubs.require_upstream()
        import borzoi_pytorch.pytorch_borzoi_helpers as H
        from flashzoi_opt import settings as S
        self.H, self._pt, self.S, self._shape = H, H.predict_tracks, S, S.OUTPUT_SHAPE
        S.OUTPUT_SHAPE = (1, 2, 3, 5)
        H.predict_tracks = lambda models, x, slices: np.full((1, len(models), 3, 5), float(x.sum().item()), dtype=np.float32)
        self.d = tempfile.mkdtemp()
        report.ITEMS.update(items=0, ok=0, failed=0)

    def tearDown(self):
        self.H.predict_tracks = self._pt; self.S.OUTPUT_SHAPE = self._shape
        shutil.rmtree(self.d, ignore_errors=True)

    def test_jobs_in_order_own_directories_same_files(self):
        from flashzoi_opt import multi, outputs
        p = _jobs_file(self.d, 3, 2)
        planned = multi.plan(stack.read_multi_jobs(p))
        self.assertEqual([len(items) for _, _, items in planned], [2, 2, 2])
        models = [types.SimpleNamespace(), types.SimpleNamespace()]
        seen = []
        err = io.StringIO(); se = sys.stderr; sys.stderr = err
        try:
            res = multi.run_jobs(models, planned, self.S.DEFAULT, device="cpu", after_job=lambda i, inp, out, counts, wall: seen.append((i, os.path.basename(out), counts["ok"])))
        finally:
            sys.stderr = se
        self.assertEqual((res["jobs"], res["ok"], res["failed"], res["items"]), (3, 3, 0, 6))
        self.assertEqual(seen, [(1, "out_0", 2), (2, "out_1", 2), (3, "out_2", 2)])
        self.assertEqual(report.ITEMS, {"items": 6, "ok": 6, "failed": 0})
        for k in range(3):
            out = os.path.join(self.d, f"out_{k}")
            rows = outputs.read_rows(out)
            self.assertEqual([r["item"] for r in rows], [f"w{k}_0", f"w{k}_1"])
            for r in rows:
                y = np.load(os.path.join(out, r["item"] + ".npy"))
                self.assertEqual(r["sha256"], outputs.sha256_array(y)); self.assertEqual(y.shape, (1, 2, 3, 5))
        lines = [ln for ln in err.getvalue().splitlines() if ln.startswith("[flashzoi-opt multi] item ")]
        self.assertEqual(len(lines), 3)
        self.assertRegex(lines[1], r"^\[flashzoi-opt multi\] item 2/3 \S+out_1 items=2 ok=2 failed=0 wall=\d+\.\d{3}s rc=0$")
        self.assertEqual(len([ln for ln in err.getvalue().splitlines() if " pred " in ln]), 6)


if __name__ == "__main__":
    unittest.main()
