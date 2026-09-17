"""The weights digest memo as wired in this kit: the `check` verb hashes afresh (refresh=True) while `pred`'s gate and the activation
read the memo (refresh=False); a memo hit is named on the WEIGHTS line and the state stays decided by digest; a memo file that cannot be
written is named on the line, never a refusal."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import colabfold_opt
from colabfold_opt import cli, digest_memo, stack
from colabfold_opt.stack import pins
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base as _ActivationBase


class TestOneDigestMemoModule(unittest.TestCase):
    def test_stack_uses_the_one_digest_memo_module(self):
        self.assertIs(stack.digest_memo, digest_memo)


class _Staged(unittest.TestCase):
    """Small files stand in for the five parameter files (as test_stock_pins.TestWeightsGate); the memo lands in the test's cache root."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.data = os.path.join(self.tmp, "af2"); os.makedirs(os.path.join(self.data, "params"))
        real = pins(); self.saved = stack._CACHE.get("pins"); fake = copy.deepcopy(real)
        open(os.path.join(self.data, real["weights"]["marker"]["file"]), "w").close()
        for i, rel in enumerate(sorted(real["weights"]["files"])):
            body = (f"pinned parameters {i}\n" * 64).encode()
            with open(os.path.join(self.data, rel), "wb") as f:
                f.write(body)
            fake["weights"]["files"][rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        stack._CACHE["pins"] = fake; stack._CACHE.pop("weights_sha256", None)
        self.rels = sorted(real["weights"]["files"])
        self.saved_root = os.environ.get("COLABFOLD_OPT_JIT_ROOT"); self.root = os.path.join(self.tmp, "jit"); os.environ["COLABFOLD_OPT_JIT_ROOT"] = self.root
        self.memo = os.path.join(self.root, digest_memo.MEMO_NAME)

    def tearDown(self):
        if self.saved is None:
            stack._CACHE.pop("pins", None)
        else:
            stack._CACHE["pins"] = self.saved
        stack._CACHE.pop("weights_sha256", None); shutil.rmtree(self.tmp, ignore_errors=True)
        if self.saved_root is None:
            os.environ.pop("COLABFOLD_OPT_JIT_ROOT", None)
        else:
            os.environ["COLABFOLD_OPT_JIT_ROOT"] = self.saved_root

    def new_process(self):
        stack._CACHE.pop("weights_sha256", None)                          # a later process: no in-process digests, the on-disk memo remains


class TestMemoOnTheServedPath(_Staged):
    def test_memo_dir_is_the_kits_cache_root(self):
        self.assertEqual(stack.digest_memo_dir(), self.root)
        self.assertEqual(stack.digest_memo_dir({}), os.path.join(os.path.expanduser("~"), ".cache", "colabfold_opt", "jit"))

    def test_first_process_hashes_and_writes_the_memo_a_later_process_reads_it(self):
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, []); self.assertIs(det["pinned"], True)
        lines = stack.weights_lines(det)
        self.assertTrue(all(line.endswith("(pinned)") for line in lines), lines)          # hashed by this process: the plain word
        table = json.load(open(self.memo))
        self.assertEqual(len(table), 5)
        self.assertEqual(sorted(e["sha256"] for e in table.values()), sorted(d["sha256"] for d in det["files"].values()))
        self.new_process()
        with mock.patch.object(stack._core_gates, "sha256_file", side_effect=AssertionError("a memo hit does not hash")):
            reasons2, det2 = stack.weights_check(self.data)
        self.assertEqual(reasons2, []); self.assertIs(det2["pinned"], True)                # decided by digest, served from the memo
        for rel, line in zip(self.rels, stack.weights_lines(det2)):
            d = det2["files"][rel]
            self.assertIsNotNone(d["digest_cached_utc"]); self.assertIsNone(d["digest_memo_note"])
            self.assertEqual(line, f"[colabfold-opt] weights={os.path.basename(rel)} sha256={d['sha256'][:12]} (pinned) (cached digest {d['digest_cached_utc']})")

    def test_refresh_hashes_afresh_and_rewrites_the_entries(self):
        stack.weights_check(self.data); self.new_process()
        with mock.patch.object(stack._core_gates, "sha256_file", wraps=stack._core_gates.sha256_file) as hashed:
            reasons, det = stack.weights_check(self.data, refresh=True)
        self.assertEqual(hashed.call_count, 5); self.assertEqual(reasons, [])
        self.assertTrue(all(d["digest_cached_utc"] is None for d in det["files"].values()))
        self.assertTrue(all(line.endswith("(pinned)") for line in stack.weights_lines(det)))

    def test_another_checkpoint_is_not_pinned_through_the_memo_too(self):
        stack.weights_check(self.data); self.new_process()
        p = os.path.join(self.data, self.rels[0]); body = open(p, "rb").read()
        with open(p, "wb") as f:
            f.write(body[::-1])                                            # same size, other bytes: a new stat key, hashed again, NOT PINNED by digest
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, []); self.assertIs(det["pinned"], False)
        self.assertEqual(det["files"][self.rels[0]]["state"], "not_pinned"); self.assertIsNone(det["files"][self.rels[0]]["digest_cached_utc"])
        self.assertIn("NOT PINNED", stack.weights_lines(det)[0]); self.assertNotIn("cached digest", stack.weights_lines(det)[0])

    def test_an_unwritable_memo_is_named_on_the_line_never_refused(self):
        open(self.root, "w").close()                                       # the cache root is a plain file: the memo cannot be written under it
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, []); self.assertIs(det["pinned"], True)
        for line in stack.weights_lines(det):
            self.assertRegex(line, r"\(pinned\) \(digest memo unwritable: \w+Error\)$")


class TestWhichVerbRefreshes(_Staged):
    """`check` passes refresh=True; `pred`'s gate (every mode) and the activation pass refresh=False."""

    def test_check_verb_refreshes(self):
        with mock.patch.object(stack, "check", return_value={"would_refuse": None}) as chk:
            rc = cli.cmd_check(argparse.Namespace(n_gpu=1, json=False), "fast")
        self.assertEqual(rc, cli.EXIT_OK)
        self.assertIs(chk.call_args.kwargs["refresh_digests"], True)

    def test_pred_gate_reads_the_memo(self):
        with mock.patch.object(stack, "check", return_value={"would_refuse": None, "weights": None}) as chk:
            cli.pred_gates("fast", self.data, 1)
        self.assertFalse(chk.call_args.kwargs.get("refresh_digests", False))
        with mock.patch.object(stack.digest_memo, "digest", wraps=stack.digest_memo.digest) as dg:
            rc, rep = cli.pred_gates("off", self.data, 1)
        self.assertIsNone(rc); self.assertEqual(dg.call_count, 5)
        self.assertTrue(all(call.kwargs.get("refresh") is False for call in dg.call_args_list))



class TestActivationReadsTheMemo(_ActivationBase):
    """Through the activation's own dry run (the stub environment of test_activation): refresh reaches the parameters gate as given."""

    def test_activation_reads_the_memo_and_check_refreshes(self):
        rep = stack.check("fast", print_line=False)
        self.assertIs(rep["weights"]["refresh"], False)                   # pred's gate, the hook, enable(): the memo is read
        rep = stack.check("fast", print_line=False, refresh_digests=True)
        self.assertIs(rep["weights"]["refresh"], True)                    # the `check` verb: afresh
        rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"]); self.assertIs(rep["weights"]["refresh"], False)

if __name__ == "__main__":
    unittest.main()
