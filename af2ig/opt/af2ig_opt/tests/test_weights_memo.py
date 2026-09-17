"""The weights digest memo as wired into this kit: `check` digests afresh (refresh=True) and `pred` reuses the memo entry (refresh=False) and
says so on the weights line; pinned-or-not stays by digest."""
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

from af2ig_opt import digest_memo, stack
from af2ig_opt.tests import _stubs


class TestRefreshWiring(unittest.TestCase):
    """stack.activate: dry_run (the `check` verb) passes refresh=True, a run (`pred`) passes refresh=False; the memo lives under the cache root."""

    def test_check_refreshes_pred_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            calls = []

            def fake_digest(path, memo_dir, refresh=False, hasher=None):
                calls.append((os.path.basename(path), memo_dir, refresh)); return "0" * 64, None
            with mock.patch.object(digest_memo, "digest", side_effect=fake_digest):
                stack.activate("exact", dry_run=True, gpu=False, environ=env)
                stack.activate("exact", dry_run=False, gpu=False, environ=env)
            self.assertEqual([c[2] for c in calls], [True, False], calls)
            self.assertEqual({c[1] for c in calls}, {os.path.join(tmp, "cache")})                  # AF2IG_OPT_CACHE_DIR of the stub environment
            self.assertEqual(stack.weights_gate(env)["memo"], os.path.join(tmp, "cache", digest_memo.MEMO_NAME))


class TestMemoOnTheCommandLine(unittest.TestCase):
    """`pred` twice on the same parameter file: the first digests it (no cached word, entry written), the second reuses the entry and the weights line ends
    `(cached digest <utc>)`; `check` in between hashes afresh (no cached word) and the verdict is the digest's either way."""

    def test_pred_pred_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); env.pop("AF2IG_OPT_FORCE", None); in_dir, _ = _stubs.inputs(tmp, 1)
            wp = os.path.join(env["AF2_PARAMS"], "params", "params_model_1_ptm.npz")
            pinned12 = hashlib.sha256(open(wp, "rb").read()).hexdigest()[:12]
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "a")], env)
            self.assertEqual(rc, 0, err); self.assertIn(f"sha256={pinned12} (pinned)", err); self.assertNotIn("(cached digest", err)
            memo = json.load(open(os.path.join(tmp, "cache", digest_memo.MEMO_NAME)))
            self.assertEqual([e["sha256"][:12] for e in memo.values()], [pinned12])
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "b")], env)
            self.assertEqual(rc, 0, err); self.assertRegex(err, r"sha256=%s \(pinned\) \(cached digest \d{4}-\d\d-\d\dT[\d:]+Z\)" % pinned12)
            rc, _, err = _stubs.run_cli(["check", "--mode", "exact"], env)                                     # check: afresh, entry rewritten, no cached word
            self.assertEqual(rc, 0, err); self.assertIn(f"sha256={pinned12} (pinned)", err); self.assertNotIn("(cached digest", err)
            with open(wp, "wb") as fh:                                                                            # other bytes at the same path: a new stat key, digested, not pinned
                fh.write(b"a user checkpoint of other bytes")
            other12 = hashlib.sha256(b"a user checkpoint of other bytes").hexdigest()[:12]
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "c")], env)
            self.assertEqual(rc, 0, err); self.assertIn(f"weights sha256={other12} not pinned", err); self.assertNotIn("(cached digest", err)


if __name__ == "__main__":
    unittest.main()
