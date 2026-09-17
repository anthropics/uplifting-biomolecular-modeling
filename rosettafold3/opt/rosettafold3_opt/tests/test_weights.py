"""The WEIGHTS line: whether the checkpoint is the pinned one is decided by its sha256 DIGEST against the pin — `checkpoint=pinned …`, or
`checkpoint=unknown … proceeding` (a named warning, never a refusal); the digest goes through the on-disk memo (`digest_memo`): a hit reads
`(cached digest <utc>)`, `check` hashes afresh, and the kit-local helper is the shared reference text byte for byte."""
import hashlib
import io
import os
import stat

import pytest

from .. import cli, digest_memo, stack, weights


def _pins(sha: str, n: int) -> dict:
    return {"weights": {"rf3": {"filename": "rf3_foundry_01_24_latest_remapped.ckpt", "sha256": sha, "bytes": n}}}


def _env(tmp_path) -> dict:
    return {weights.ENV_DIGEST_DIR: str(tmp_path / "memo")}


def test_pinned_checkpoint_says_pinned(tmp_path):
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"weights-of-record")
    buf = io.StringIO()
    info = weights.announce(str(ck), _pins(hashlib.sha256(b"weights-of-record").hexdigest(), 17), file=buf, environ=_env(tmp_path))
    assert info["status"] == "pinned" and info["sha256_matches_pin"] is True and info["digest_cached_utc"] is None
    assert "WEIGHTS checkpoint=pinned sha256=" in buf.getvalue() and "the pinned checkpoint" in buf.getvalue()


def test_unknown_checkpoint_warns_by_name_and_proceeds(tmp_path):
    ck = tmp_path / "other.ckpt"; ck.write_bytes(b"some other weights")
    buf = io.StringIO()
    info = weights.announce(str(ck), _pins("0" * 64, 3038876446), file=buf, environ=_env(tmp_path))          # returns: no refusal
    out = buf.getvalue()
    assert info["status"] == "unknown" and info["sha256_matches_pin"] is False
    assert "WEIGHTS checkpoint=unknown" in out and "not the pinned checkpoint" in out and out.rstrip().endswith("proceeding")


def test_size_matching_but_digest_differing_is_unknown(tmp_path):
    """The digest alone decides: a file with the pin's byte count and name but other contents is unknown."""
    ck = tmp_path / "rf3_foundry_01_24_latest_remapped.ckpt"; ck.write_bytes(b"12345678901234567")
    info = weights.pin_status(str(ck), _pins(hashlib.sha256(b"weights-of-record").hexdigest(), 17), environ=_env(tmp_path))
    assert info["bytes"] == 17 and info["status"] == "unknown"


def test_no_pin_is_unknown_too(tmp_path):
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"x")
    buf = io.StringIO()
    assert weights.announce(str(ck), {}, file=buf, environ=_env(tmp_path))["status"] == "unknown" and "proceeding" in buf.getvalue()


def test_a_memo_hit_names_the_cached_time_and_check_hashes_afresh(tmp_path):
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"weights-of-record"); pins = _pins(hashlib.sha256(b"weights-of-record").hexdigest(), 17)
    first = weights.pin_status(str(ck), pins, verb="pred", environ=_env(tmp_path))
    assert first["digest_cached_utc"] is None and os.path.isfile(first["digest_memo"]) and first["digest_memo"].endswith(digest_memo.MEMO_NAME)
    buf = io.StringIO()
    second = weights.announce(str(ck), pins, verb="warm", file=buf, environ=_env(tmp_path))
    assert second["status"] == "pinned" and second["digest_cached_utc"] is not None
    assert "WEIGHTS checkpoint=pinned (cached digest " in buf.getvalue()
    third = weights.pin_status(str(ck), pins, verb="check", environ=_env(tmp_path))                              # check: afresh, entry rewritten
    assert third["digest_cached_utc"] is None and third["status"] == "pinned"


def test_check_passes_refresh_true_and_pred_warm_read_the_memo(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(digest_memo, "digest", lambda path, memo_dir, refresh=False, hasher=None: (calls.append((os.path.basename(path), memo_dir, refresh)) or ("0" * 64, None)))
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"x")
    for verb in ("pred", "warm", "check"):
        weights.pin_status(str(ck), {}, verb=verb, environ=_env(tmp_path))
    assert [c[2] for c in calls] == [False, False, True] and {c[1] for c in calls} == {str(tmp_path / "memo")}
    # the check VERB itself digests the checkpoint it is given, afresh
    calls.clear()
    monkeypatch.setenv(weights.ENV_DIGEST_DIR, str(tmp_path / "memo"))
    monkeypatch.setattr(stack, "activate", lambda mode, **kw: {"mode": mode, "reason": "stub activation", "gpu": None})
    monkeypatch.setattr(cli.report, "active_line", lambda rep: "[rosettafold3-opt] ACTIVE (stub)")
    cli.main(["check", "--mode", "exact", "--ckpt", str(ck)])
    assert calls == [("w.ckpt", str(tmp_path / "memo"), True)]


def test_memo_dir_sits_beside_the_jit_caches_by_default(tmp_path):
    assert weights.memo_dir({weights.ENV_DIGEST_DIR: "/x/memo"}) == "/x/memo"
    assert weights.memo_dir({"TRITON_CACHE_DIR": "/jitcache/torch2.13.0-sm90/triton"}) == "/jitcache/weights"
    assert weights.memo_dir({}).endswith(os.path.join(".cache", "rosettafold3_opt", "weights"))


def _read_only_dir(tmp_path, name="ro"):
    d = tmp_path / name; d.mkdir(); d.chmod(0o555)
    if os.access(str(d), os.W_OK):                                                # root writes through 0o555: the case cannot be staged here
        pytest.skip("cannot stage an unwritable directory as this user")
    return d


def test_an_unwritable_memo_dir_is_named_and_hashed_afresh_never_a_refusal(tmp_path):
    """An external caller mounts the JIT-cache root read-only: check (afresh) and pred (memo miss) name the memo state and proceed."""
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"weights-of-record"); pins = _pins(hashlib.sha256(b"weights-of-record").hexdigest(), 17)
    ro = _read_only_dir(tmp_path); env = {weights.ENV_DIGEST_DIR: str(ro / "weights")}
    try:
        for verb in ("check", "pred"):
            buf = io.StringIO()
            info = weights.announce(str(ck), pins, verb=verb, file=buf, environ=env)
            assert info["status"] == "pinned" and info["digest_cached_utc"] is None and info["digest_memo_state"].startswith("unwritable:")
            assert "memo=unwritable:" in buf.getvalue() and "hashed afresh, not memoised" in buf.getvalue() and "WEIGHTS checkpoint=pinned sha256=" in buf.getvalue()
        assert not (ro / "weights").exists()
    finally:
        ro.chmod(0o755)


def test_a_read_only_memo_with_an_entry_is_still_read(tmp_path):
    ck = tmp_path / "w.ckpt"; ck.write_bytes(b"weights-of-record"); pins = _pins(hashlib.sha256(b"weights-of-record").hexdigest(), 17)
    memo = tmp_path / "memo"; env = {weights.ENV_DIGEST_DIR: str(memo)}
    assert weights.pin_status(str(ck), pins, verb="pred", environ=env)["digest_memo_state"] == "ok"          # written by a process that could
    memo.chmod(0o555)
    if os.access(str(memo), os.W_OK):
        memo.chmod(0o755); pytest.skip("cannot stage an unwritable directory as this user")
    try:
        buf = io.StringIO()
        hit = weights.announce(str(ck), pins, verb="pred", file=buf, environ=env)
        assert hit["digest_cached_utc"] is not None and hit["digest_memo_state"].startswith("read_only:") and "(cached digest " in buf.getvalue()
        fresh = weights.pin_status(str(ck), pins, verb="check", environ=env)                                # check never reads the memo: afresh, named
        assert fresh["digest_cached_utc"] is None and fresh["digest_memo_state"].startswith("unwritable:")
    finally:
        memo.chmod(0o755)
