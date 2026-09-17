"""The kit's wiring of the weights digest memo (`digest_memo.py`, the release tree's one helper — its own seven tests are test_digest_memo.py):
`check` digests afresh (refresh=True) and pred / warm / the hook take memo hits (refresh=False); a rank process of the row-sharded launcher reads the
launching process's entries (rank_digest) and never hashes — a miss refuses by name; the WEIGHTS line says `(cached digest <utc>)` on a hit. CPU only, small
synthetic pins."""
import os
import tempfile
import unittest
from unittest import mock

from esmfold2_opt import cli, digest_memo, stack
from .test_weights_unknown_warns import _small_pins, _lay_out


class _Env:
    def __init__(self, **kv):
        self.kv = kv
    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        return self
    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


class TestKitWiring(unittest.TestCase):
    def setUp(self):
        stack.set_weights_afresh(False); stack._WEIGHTS_SEEN.clear()
        self.pins, self.content = _small_pins()
        self.tmp = tempfile.TemporaryDirectory(); self.hf = os.path.join(self.tmp.name, "hf"); self.memo_dir = os.path.join(self.tmp.name, "cache")
        _lay_out(self.hf, self.pins, self.content)
        self.n = len(stack.pinned_weight_files(self.pins, "fast"))

    def tearDown(self):
        stack.set_weights_afresh(False); self.tmp.cleanup()

    def _gate(self, rank=None):
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir, TRITON_CACHE_DIR=None, ROWPAIR_RANK=rank, ROWPAIR_WORLD=None if rank is None else "2"):
            return stack.data_path_gate(None, variant="fast", pins=self.pins)

    def test_memo_dir_is_the_env_then_the_cache_root_then_a_fixed_temp_dir(self):
        self.assertEqual(stack.weights_memo_dir({"ESMFOLD2_OPT_WEIGHTS_MEMO_DIR": "/x"}), "/x")
        self.assertEqual(stack.weights_memo_dir({"TRITON_CACHE_DIR": "/jitcache/torch2.13.0-cu130-sm90/triton/"}), "/jitcache/torch2.13.0-cu130-sm90")
        self.assertEqual(stack.weights_memo_dir({}), os.path.join(os.path.expanduser("~"), ".cache", "esmfold2_opt"))
        self.assertFalse(stack.weights_memo_dir({}).startswith(tempfile.gettempdir().rstrip("/") + "/"))   # never a fixed name in the shared temporary directory

    def test_pred_serve_take_memo_hits_refresh_false_and_say_cached(self):
        seen = []
        real = digest_memo.digest
        def spy(path, memo_dir, refresh=False, hasher=digest_memo.sha256_file):
            seen.append(refresh); return real(path, memo_dir, refresh=refresh, hasher=hasher)
        with mock.patch.object(digest_memo, "digest", spy):
            why, notes = self._gate()
            self.assertIsNone(why); self.assertEqual(seen, [False] * self.n)
            self.assertTrue(os.path.isfile(os.path.join(self.memo_dir, digest_memo.MEMO_NAME)))
            stack._WEIGHTS_SEEN.clear(); seen.clear()
            calls = []
            with mock.patch.object(digest_memo, "sha256_file", lambda p, chunk=digest_memo.CHUNK: calls.append(p) or "0" * 64):
                why, notes = self._gate()                                            # the default hasher is bound at def time: a hit never reaches any hasher
        self.assertIsNone(why); self.assertEqual(calls, [])
        ok = [x for x in notes if x.startswith("WEIGHTS pinned sha256")]; self.assertEqual(len(ok), 1, notes)
        self.assertRegex(ok[0], r"\(cached digest \d{4}-\d{2}-\d{2}T"); self.assertNotIn("digested", ok[0])
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir):
            st = stack.weights_status("fast", self.pins)
        self.assertEqual(st["word"], stack.WEIGHTS_PINNED); self.assertTrue(all(v.startswith("cached digest") for v in st["digest_source"].values()), st)
        self.assertTrue(st["memo"].endswith("/cache/" + digest_memo.MEMO_NAME), st["memo"])

    def test_check_passes_refresh_true_and_says_afresh(self):
        self._gate()                                                                  # a populated memo
        seen = []; real = digest_memo.digest
        def spy(path, memo_dir, refresh=False, hasher=digest_memo.sha256_file):
            seen.append(refresh); return real(path, memo_dir, refresh=refresh, hasher=hasher)
        stack._WEIGHTS_SEEN.clear()
        with mock.patch.object(stack, "activate", lambda *a, **k: {"active": False, "reason": "stub"}):
            cli.activate("fast", "fast", dry_run=True)                                # `check`'s path arms the afresh digest
        try:
            self.assertTrue(stack._WEIGHTS_AFRESH["on"])
            with mock.patch.object(digest_memo, "digest", spy):
                why, notes = self._gate()
        finally:
            stack.set_weights_afresh(False)
        self.assertIsNone(why); self.assertEqual(seen, [True] * self.n)
        ok = [x for x in notes if x.startswith("WEIGHTS pinned sha256")][0]; self.assertIn("afresh (check)", ok); self.assertNotIn("cached digest", ok)

    def test_a_cached_digest_not_pinned_is_unknown_by_name_with_the_cached_word(self):
        rel = next(r for _x, r, _s in stack.pinned_weight_files(self.pins, "fast") if r.endswith("model.safetensors"))
        digest_memo.digest(os.path.join(self.hf, rel), self.memo_dir, hasher=lambda p: "0" * 64)      # a memo entry whose digest is not pinned
        why, notes = self._gate()
        self.assertIsNone(why); warn = [x for x in notes if x.startswith("WEIGHTS unknown")]
        self.assertEqual(len(warn), 1, notes); self.assertIn(f"{rel} sha256=000000000000", warn[0]); self.assertIn("(cached digest ", warn[0]); self.assertIn("proceeding", warn[0])

    def test_rank_process_reads_the_launchers_entries_and_never_hashes(self):
        why, _ = self._gate()                                                         # the launching process digests (cmd_pred does this before rowpair.launch)
        self.assertIsNone(why)
        stack._WEIGHTS_SEEN.clear()
        def must_not(*a, **k):
            raise AssertionError("a rank process hashed / called digest()")
        with mock.patch.object(digest_memo, "digest", must_not), mock.patch.object(digest_memo, "sha256_file", must_not):
            why, notes = self._gate(rank="1")
        self.assertIsNone(why)
        ok = [x for x in notes if x.startswith("WEIGHTS pinned sha256")]; self.assertEqual(len(ok), 1, notes)
        self.assertIn("(cached digest ", ok[0]); self.assertTrue(ok[0].endswith(" via=launcher"), ok[0])
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir):
            self.assertEqual(stack.weights_status("fast", self.pins)["via"], "launcher memo")

    def test_rank_process_memo_miss_refuses_by_name(self):
        with mock.patch.object(digest_memo, "digest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hashed"))):
            why, _ = self._gate(rank="0")                                             # rank 0 is a rank process too: the LAUNCHING process digests
        self.assertIsNotNone(why); self.assertIn("memo miss in a rank process", why); self.assertIn("refused", why)

    def test_launcher_predigests_before_the_ranks_start(self):
        """cmd_pred at --n_gpu 2 in the launching process: the data-path gate (digest) runs and the memo dir is exported before rowpair.launch."""
        from esmfold2_opt import rowpair
        order = []
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None, ESMFOLD2_VARIANT=None), \
             mock.patch.object(stack, "data_path_gate", lambda env=None, variant=None, pins=None: (order.append(("gate", variant, os.environ.get(stack.ENV_WEIGHTS_MEMO_DIR))), (None, []))[1]), \
             mock.patch.object(rowpair, "launch", lambda argv, mode, n: (order.append(("launch", n)), 0)[1]), \
             mock.patch.object(rowpair, "is_rank_process", lambda: False), \
             mock.patch.object(rowpair, "refuse_unless_visible", lambda P, visible=None: P), \
             mock.patch.object(cli.tp, "precheck", lambda mode, n: 2), mock.patch.object(cli.report, "set_n_gpu", lambda p: p):
            inp = os.path.join(self.tmp.name, "in.json"); open(inp, "w").write('{"sequences": [{"type": "protein", "id": "A", "sequence": "M"}]}')   # a real input: the launching process reads it (one sample count per run) before any rank starts
            rc = cli.cmd_pred(["--variant", "fast", "--mode", "big", "--n_gpu", "2", "--input", inp, "--out_dir", self.tmp.name])
        self.assertEqual(rc, 0); self.assertEqual([o[0] for o in order], ["gate", "launch"], order)
        self.assertEqual(order[0][1], "fast"); self.assertEqual(order[0][2], self.memo_dir)


class TestReadOnlyMemoDirIsNamedNeverARefusal(unittest.TestCase):
    """The memo's usual home is a cache volume a caller may mount read-only: `check` (afresh) and `pred` (memo miss) then digest afresh in the
    process, print ONE named line, memoise nothing, and proceed — never a refusal, never a traceback."""
    def setUp(self):
        stack.set_weights_afresh(False); stack._WEIGHTS_SEEN.clear()
        self.pins, self.content = _small_pins()
        self.tmp = tempfile.TemporaryDirectory(); self.hf = os.path.join(self.tmp.name, "hf"); _lay_out(self.hf, self.pins, self.content)
        self.n = len(stack.pinned_weight_files(self.pins, "fast"))
        ro = os.path.join(self.tmp.name, "ro"); os.makedirs(ro); os.chmod(ro, 0o555)
        if os.access(ro, os.W_OK):                                                    # root ignores 0o555: an uncreatable directory (its parent is a regular file) stands in for the read-only mount
            blocker = os.path.join(self.tmp.name, "afile"); open(blocker, "w").write("x"); ro = os.path.join(blocker, "memo")
        self.memo_dir = ro

    def tearDown(self):
        stack.set_weights_afresh(False)
        try: os.chmod(os.path.join(self.tmp.name, "ro"), 0o755)
        except OSError: pass
        self.tmp.cleanup()

    def _gate(self):
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None):
            return stack.data_path_gate(None, variant="fast", pins=self.pins)

    def test_probe_names_the_unwritable_dir(self):
        ok, why = stack.weights_memo_writable(self.memo_dir)
        self.assertFalse(ok); self.assertTrue(why)
        ok2, _ = stack.weights_memo_writable(os.path.join(self.tmp.name, "rw", "memo")); self.assertTrue(ok2)

    def test_check_on_a_readonly_memo_dir_digests_afresh_names_it_and_proceeds(self):
        stack.set_weights_afresh(True)
        calls = []
        try:
            with mock.patch.object(digest_memo, "sha256_file", lambda p, chunk=digest_memo.CHUNK, _r=digest_memo.sha256_file: (calls.append(p), _r(p, chunk))[1]):
                why, notes = self._gate()
        finally:
            stack.set_weights_afresh(False)
        self.assertIsNone(why, "a read-only memo dir is never a refusal")
        self.assertEqual(len(calls), self.n, "every file digested afresh")
        self.assertEqual(len([x for x in notes if x.startswith("WEIGHTS pinned sha256")]), 1, notes)
        ro = [x for x in notes if x.startswith("WEIGHTS digest memo skipped")]; self.assertEqual(len(ro), 1, notes)
        self.assertIn("read-only", ro[0]); self.assertIn(stack.ENV_WEIGHTS_MEMO_DIR, ro[0])
        self.assertNotIn("OSError", ro[0]); self.assertNotIn("Errno", ro[0])  # K14: must not embed raw exception text a log reader could mistake for a real error
        self.assertFalse(os.path.exists(os.path.join(self.memo_dir, digest_memo.MEMO_NAME)))

    def test_pred_on_a_readonly_memo_dir_with_no_entries_digests_afresh_and_proceeds(self):
        why, notes = self._gate()
        self.assertIsNone(why); self.assertEqual(len([x for x in notes if x.startswith("WEIGHTS digest memo skipped")]), 1, notes)
        ok = [x for x in notes if x.startswith("WEIGHTS pinned sha256")][0]; self.assertIn("digested", ok); self.assertNotIn("cached digest", ok)

    def test_launcher_with_a_readonly_memo_dir_relays_its_digests_to_the_ranks(self):
        """`pred --n_gpu P>1`: the launching process digests afresh (configured dir unwritable), then writes the entries to a private writable dir and
        exports it — a rank process reads them with rank_digest and never hashes; a genuine miss still refuses by name."""
        why, notes = self._gate()                                                     # the launcher's gate: afresh + the named line
        self.assertIsNone(why); self.assertTrue([x for x in notes if x.startswith("WEIGHTS digest memo skipped")])
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=self.memo_dir, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None):
            relay = stack.weights_memo_relay_for_ranks(self.hf, "fast")
            exported = os.environ.get(stack.ENV_WEIGHTS_MEMO_DIR)
        self.assertIsNotNone(relay); self.assertEqual(exported, relay); self.assertNotEqual(relay, self.memo_dir)
        self.assertTrue(os.path.isfile(os.path.join(relay, digest_memo.MEMO_NAME)))
        stack._WEIGHTS_SEEN.clear()
        def must_not(*a, **k):
            raise AssertionError("a rank process hashed")
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=relay, TRITON_CACHE_DIR=None, ROWPAIR_RANK="1", ROWPAIR_WORLD="2"), \
             mock.patch.object(digest_memo, "sha256_file", must_not), mock.patch.object(digest_memo, "digest", must_not):
            why1, notes1 = stack.data_path_gate(None, variant="fast", pins=self.pins)
        self.assertIsNone(why1)
        ok = [x for x in notes1 if x.startswith("WEIGHTS pinned sha256")]; self.assertEqual(len(ok), 1, notes1); self.assertTrue(ok[0].endswith(" via=launcher"), ok[0])
        stack._WEIGHTS_SEEN.clear()
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=os.path.join(self.tmp.name, "empty"), TRITON_CACHE_DIR=None, ROWPAIR_RANK="1", ROWPAIR_WORLD="2"):
            why2, _ = stack.data_path_gate(None, variant="fast", pins=self.pins)     # a genuine miss still refuses by name
        self.assertIsNotNone(why2); self.assertIn("memo miss in a rank process", why2)

    def test_a_writable_configured_dir_needs_no_relay(self):
        rw = os.path.join(self.tmp.name, "rw2")
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=rw, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None):
            self.assertIsNone(stack.data_path_gate(None, variant="fast", pins=self.pins)[0])
            self.assertIsNone(stack.weights_memo_relay_for_ranks(self.hf, "fast")); self.assertEqual(os.environ.get(stack.ENV_WEIGHTS_MEMO_DIR), rw)

    def test_pred_on_a_readonly_memo_dir_with_entries_serves_the_hits_without_hashing(self):
        rw = os.path.join(self.tmp.name, "seed"); os.makedirs(rw)
        with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=rw, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None):
            self.assertIsNone(stack.data_path_gate(None, variant="fast", pins=self.pins)[0])         # a populated memo …
        os.chmod(rw, 0o555)                                                                             # … now read-only (as root this stays writable: then the branch under test is the writable one and the assertions still hold)
        stack._WEIGHTS_SEEN.clear(); calls = []
        try:
            with _Env(HF_HOME=self.hf, ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=rw, TRITON_CACHE_DIR=None, ROWPAIR_RANK=None), \
                 mock.patch.object(digest_memo, "sha256_file", lambda p, chunk=digest_memo.CHUNK: calls.append(p) or "0" * 64):
                why, notes = stack.data_path_gate(None, variant="fast", pins=self.pins)
        finally:
            os.chmod(rw, 0o755)
        self.assertIsNone(why); self.assertEqual(calls, [], "memo hits are served read-only, nothing hashed")
        self.assertIn("(cached digest ", [x for x in notes if x.startswith("WEIGHTS pinned sha256")][0])


if __name__ == "__main__":
    unittest.main()
