"""The weights digest goes through the kit-local on-disk memo (`digest_memo`, identical text in every kit): `check` hashes afresh
(refresh=True) and rewrites the entry, `pred` and an activation read the memo (refresh=False), a rank process of a multi-GPU run never
hashes (it reads the entry the launching process wrote), and a digest served from the memo says so on the weights line."""
import hashlib
import os

import pytest

from protenix_v1_opt import digest_memo as DM
from protenix_v1_opt import kit as K
from protenix_v1_opt import modes as M
from protenix_v1_opt import report as R
from protenix_v1_opt import stack
from protenix_v1_opt.tests.test_frozen_weights import _pin, _root


def test_the_memo_lives_in_the_kits_cache_root(monkeypatch):
    monkeypatch.setenv(K.DIGEST_MEMO_ENV, "/x/cache")
    assert K.digest_memo_dir() == "/x/cache" and K.DIGEST_MEMO_ENV == "TRITON_CACHE_DIR" and DM.MEMO_NAME == "weights_digests.json"
    monkeypatch.delenv(K.DIGEST_MEMO_ENV)
    assert K.digest_memo_dir() == K.DIGEST_MEMO_DEFAULT == os.path.join(os.path.expanduser("~"), ".cache", "protenix_v1_opt")
    assert not K.DIGEST_MEMO_DEFAULT.startswith("/tmp/")           # never a fixed name in the shared temporary directory


def _record_digest_calls(monkeypatch):
    calls = []

    def fake(path, memo_dir, refresh=False, hasher=None):
        calls.append({"refresh": refresh, "memo_dir": memo_dir, "path": path})
        return "ab" * 32, None
    monkeypatch.setattr(DM, "digest", fake)
    return calls


def test_check_hashes_afresh(tmp_path, monkeypatch):
    """`check` (cli.dry_run_report -> stack.gates(refresh_weights=True)) passes refresh=True to the memo."""
    from protenix_v1_opt import cli
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.setenv(_pin()["root_env"], _root(tmp_path))
    calls = _record_digest_calls(monkeypatch)
    cli.dry_run_report("exact")
    assert [c["refresh"] for c in calls] == [True] and calls[0]["memo_dir"] == K.digest_memo_dir()


def test_pred_and_the_activation_gate_read_the_memo(tmp_path, monkeypatch):
    """`pred` (the cli gate) and an activation (stack.gates) pass refresh=False."""
    from protenix_v1_opt import cli
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.setenv(_pin()["root_env"], _root(tmp_path))
    calls = _record_digest_calls(monkeypatch)
    monkeypatch.setattr(cli, "_run_stock", lambda stock_args, det: 0)
    assert cli.main(["pred", "--mode", "off", "--input", "x.json", "--out_dir", str(tmp_path / "o")]) == 0
    monkeypatch.setattr(K, "_DIGESTS", {})                            # a second process: the activation gate inside the model process
    stack.gates(M.resolve("exact"), False)
    assert [c["refresh"] for c in calls] == [False, False]


def test_a_second_process_reads_the_memo_and_the_line_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    root = _root(tmp_path, ckpt_bytes=777)
    w1 = K.frozen_weights_check(root)
    assert w1["cached_utc"] is None and len(w1["sha256"]) == 64
    memo = os.path.join(K.digest_memo_dir(), DM.MEMO_NAME)
    assert os.path.isfile(memo)                                       # written after the full hash
    monkeypatch.setattr(K, "_DIGESTS", {}); monkeypatch.setattr(K, "_ANNOUNCED", set())   # a later process on the same cache root
    w2 = K.frozen_weights_check(root)
    assert w2["sha256"] == w1["sha256"] and w2["cached_utc"] is not None
    err = capsys.readouterr().err.splitlines()
    assert err[-1] == f"{R.PREFIX} weights sha256={w1['sha256'][:12]} {K.NOT_PINNED_WORDS} (cached digest {w2['cached_utc']})"
    w3 = K.frozen_weights_check(root, refresh=True)                  # `check`: afresh, the entry rewritten, no cached word
    assert w3["cached_utc"] is None and w3["sha256"] == w1["sha256"]


def test_the_digest_decides_not_the_stat_fields(tmp_path, monkeypatch):
    """A memo entry whose digest is not the file's (same size/mtime/inode key) is what the memo serves — and `check` corrects it: the
    stat fields select the entry, the digest decides."""
    import json
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    payload = b"pinned weights for this test"
    monkeypatch.setitem(_pin(), "checkpoint_sha256", hashlib.sha256(payload).hexdigest())
    root = _root(tmp_path, payload=payload)
    ckpt = K.stock_checkpoint(root, [])
    memo = os.path.join(K.digest_memo_dir(), DM.MEMO_NAME); os.makedirs(os.path.dirname(memo), exist_ok=True)
    with open(memo, "w") as fh:
        json.dump({DM.stat_key(ckpt): {"sha256": "0" * 64, "utc": "2000-01-01T00:00:00Z"}}, fh)
    assert K.frozen_weights_check(root)["pinned"] is False              # the memo's digest is served (pred form) ...
    monkeypatch.setattr(K, "_DIGESTS", {})
    w = K.frozen_weights_check(root, refresh=True)                        # ... `check` hashes afresh: pinned by digest, entry rewritten
    assert w["pinned"] is True and json.load(open(memo))[DM.stat_key(ckpt)]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_rank_processes_never_hash(tmp_path, monkeypatch):
    """Under the rank environment (ROWPAIR_WORLD > 1) the digest is READ from the entry the launching process wrote; a miss raises by name."""
    from opt_core.mem.rowpair import launch as L
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    root = _root(tmp_path, ckpt_bytes=555)
    monkeypatch.setenv(L.ENV_WORLD, "2"); monkeypatch.setenv(L.ENV_RANK, "1")
    with pytest.raises(RuntimeError, match="digest memo miss in a rank process"):
        K.frozen_weights_check(root)
    monkeypatch.delenv(L.ENV_WORLD); monkeypatch.delenv(L.ENV_RANK); monkeypatch.setattr(K, "_DIGESTS", {})
    w0 = K.frozen_weights_check(root)                                     # the launching process digests ...
    monkeypatch.setenv(L.ENV_WORLD, "2"); monkeypatch.setenv(L.ENV_RANK, "1"); monkeypatch.setattr(K, "_DIGESTS", {})
    hashed = []
    monkeypatch.setattr(K, "sha256_file", lambda p: hashed.append(p) or "0" * 64)
    w1 = K.frozen_weights_check(root)                                     # ... the rank reads its entry, hashing nothing
    assert w1["sha256"] == w0["sha256"] and hashed == [] and w1["cached_utc"] is not None


def _uncreatable_dir(tmp_path):
    """A memo directory no process can create: its parent is a regular file (ENOTDIR for any user, root included)."""
    blocker = tmp_path / "not_a_dir"; blocker.write_bytes(b"")
    return str(blocker / "memo")


def test_an_unwritable_memo_dir_is_named_once_and_the_run_proceeds(tmp_path, monkeypatch, capsys):
    """A read-only / uncreatable memo directory is never a refusal: the digest is computed ONCE, the transcript names the directory and the
    error, the entry lands in a process-private writable directory exported as $PROTENIX_V1_OPT_WEIGHTS_MEMO, and `check` (refresh=True)
    behaves the same."""
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.delenv(K.WEIGHTS_MEMO_ENV, raising=False)
    bad = _uncreatable_dir(tmp_path); monkeypatch.setenv(K.DIGEST_MEMO_ENV, bad)
    hashed = []; real = K.sha256_file
    monkeypatch.setattr(K, "sha256_file", lambda p: hashed.append(p) or real(p))
    root = _root(tmp_path, ckpt_bytes=333)
    w = K.frozen_weights_check(root)                                       # pred / activation form
    err = capsys.readouterr().err
    assert w["cached_utc"] is None and len(w["sha256"]) == 64 and len(hashed) == 1
    assert err.count("weights digest memo unwritable at %s" % bad) == 1 and "NOT PINNED" in err
    private = os.environ[K.WEIGHTS_MEMO_ENV]
    assert private != bad and os.path.isfile(os.path.join(private, DM.MEMO_NAME)) and K.digest_memo_dir() == private
    monkeypatch.setattr(K, "_DIGESTS", {}); monkeypatch.delenv(K.WEIGHTS_MEMO_ENV)  # `check` on the unwritable root: afresh, named, proceeds
    w2 = K.frozen_weights_check(root, refresh=True)
    assert w2["sha256"] == w["sha256"] and w2["cached_utc"] is None and len(hashed) == 2
    assert capsys.readouterr().err.count("weights digest memo unwritable") == 1


def test_ranks_read_the_launchers_private_memo_when_the_configured_dir_is_unwritable(tmp_path, monkeypatch, capsys):
    """Launcher on an unwritable cache root, then a rank process (ROWPAIR_WORLD > 1) inheriting its environment: the rank gets the launcher's
    digest from the exported private directory without hashing; a genuine miss still raises by name."""
    from opt_core.mem.rowpair import launch as L
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.delenv(K.WEIGHTS_MEMO_ENV, raising=False)
    monkeypatch.setenv(K.DIGEST_MEMO_ENV, _uncreatable_dir(tmp_path))
    root = _root(tmp_path, ckpt_bytes=444)
    w0 = K.frozen_weights_check(root)                                      # the launching process: hashed once, private memo exported
    assert K.WEIGHTS_MEMO_ENV in os.environ
    monkeypatch.setenv(L.ENV_WORLD, "2"); monkeypatch.setenv(L.ENV_RANK, "1"); monkeypatch.setattr(K, "_DIGESTS", {})
    monkeypatch.setattr(K, "sha256_file", lambda p: pytest.fail("a rank process hashed the weights"))
    w1 = K.frozen_weights_check(root)                                      # the rank (inherits os.environ): reads the launcher's entry
    assert w1["sha256"] == w0["sha256"] and w1["cached_utc"] is not None
    other = _root(tmp_path / "other", ckpt_bytes=445)                      # a checkpoint the launcher never digested: a miss named, never a hash
    monkeypatch.setattr(K, "_DIGESTS", {})
    with pytest.raises(RuntimeError, match="digest memo miss in a rank process"):
        K.frozen_weights_check(other)


def test_the_memo_dir_precedence(monkeypatch):
    monkeypatch.setenv(K.DIGEST_MEMO_ENV, "/cache/root"); monkeypatch.delenv(K.WEIGHTS_MEMO_ENV, raising=False)
    assert K.digest_memo_dir() == "/cache/root"
    monkeypatch.setenv(K.WEIGHTS_MEMO_ENV, "/explicit/memo")
    assert K.digest_memo_dir() == "/explicit/memo"
