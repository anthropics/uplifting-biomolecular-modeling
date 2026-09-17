"""The checkpoint's equality is decided by its sha256 digest against stock/PINS.json and is a note, never a gate: an unknown digest is a
WARNING by name and every route proceeds; a file of the recorded SIZE with another digest is unknown too.

Tests of the kit-local weights digest memo (identical text in every kit except the import line).

The weights digest memo as this kit wires it: the reference helper carried byte-for-byte, `check` hashes afresh (refresh=True), `pred`
reads the memo (refresh=False), rank processes of a `--n_gpu P>1` run read the launcher's entry and never hash."""
import hashlib
import json
import os

import pytest

from opendde_opt import cli, digest_memo, frozen, manifest, stack
from opendde_opt.tests import _stubs


@pytest.fixture
def clean(tmp_path, monkeypatch):
    site = _stubs.make_site(str(tmp_path / "site"))
    monkeypatch.syspath_prepend(site)
    import importlib
    importlib.invalidate_caches()
    for k in list(os.environ):
        if k.startswith(("ODDE_", "OPENDDE_OPT", "DIT_", "CUEQ_", "CUBLAS_", "PYTORCH_CUDA")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    monkeypatch.setattr(stack, "_REPORT", None)
    return site


def _root(tmp_path, payload: bytes):
    root = _stubs.weights_root(str(tmp_path / "w"))                  # a complete frozen-weights root (every required file, managed sizes) ...
    with open(os.path.join(root, "checkpoint", "opendde.pt"), "wb") as fh:
        fh.write(payload)                                            # ... whose checkpoint carries this test's bytes
    return root


def test_an_unknown_digest_is_a_warning_by_name_and_check_proceeds_rc_0(tmp_path, monkeypatch, capsys, clean):
    root = _root(tmp_path, b"not the pinned weights")
    w = manifest.weights(root, tree=_stubs.TREE)
    assert w["checkpoint_match"] == "unknown" and w["files_hashed"] == ["checkpoint/opendde.pt"] and w["checkpoint_sha256"] == hashlib.sha256(b"not the pinned weights").hexdigest()
    line = manifest.weights_line(w)
    assert line.startswith("[opendde-opt] WEIGHTS sha256=" + w["checkpoint_sha256"][:12] + " UNKNOWN") and "not the pinned checkpoint" in line and "proceeding" in line
    assert frozen.problems(root, os.devnull, {"use_msa": "false"}, [], tree=None) == [] or all("checkpoint" not in p for p in frozen.problems(root, os.devnull, {"use_msa": "false"}, [], tree=None))
    monkeypatch.setenv("OPENDDE_ROOT_DIR", root)
    monkeypatch.setattr(stack, "shim_active", lambda: None)                          # this test's premise: no kit shim armed earlier in the session's process
    stack._REPORT = None; capsys.readouterr()
    assert cli.main(["check", "--mode", "off"]) == cli.EXIT_OK                       # proceeds: rc 0 ...
    out = capsys.readouterr().out
    assert "WEIGHTS sha256=" + w["checkpoint_sha256"][:12] + " UNKNOWN" in out, out[-800:]   # ... with the WARNING line by name
    stack._REPORT = None
    assert cli.main(["check", "--mode", "fast", "--json"]) == cli.EXIT_OK
    import json
    rep = json.loads(capsys.readouterr().out)
    assert rep["facts"]["weights"]["checkpoint_match"] == "unknown" and rep["facts"]["weights"]["checkpoint_sha256"] == w["checkpoint_sha256"]


def test_the_pinned_digest_prints_pinned_and_no_warning(tmp_path, monkeypatch):
    payload = b"the pinned weights (stub)"
    root = _root(tmp_path, payload)
    monkeypatch.setattr(manifest, "pinned_checkpoint", lambda tree: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    w = manifest.weights(root, tree=_stubs.TREE)
    line = manifest.weights_line(w)
    assert w["checkpoint_match"] == "pinned" and "(pinned)" in line and "UNKNOWN" not in line and "WARNING" not in line


def test_a_size_match_with_another_digest_is_unknown_not_pinned(tmp_path, monkeypatch):
    payload = b"same size, other bytes!!"
    root = _root(tmp_path, payload)
    monkeypatch.setattr(manifest, "pinned_checkpoint", lambda tree: {"bytes": len(payload), "sha256": "0" * 64})   # the recorded size, a different digest
    w = manifest.weights(root, tree=_stubs.TREE)
    assert w["checkpoint_bytes"] == len(payload) and w["checkpoint_match"] == "unknown" and "UNKNOWN" in manifest.weights_line(w)


def test_pred_check_and_serve_print_the_weights_line_and_the_gate_never_reads_the_match():
    pkg = os.path.dirname(manifest.__file__)
    assert open(os.path.join(pkg, "cli.py")).read().count("_manifest.weights_line(") == 2          # pred + check
    assert "checkpoint_match" not in open(os.path.join(pkg, "frozen.py")).read()                            # the frozen-weights gate refuses downloads, never an equality


def _write(p, data):
    p.write_bytes(data)
    return str(p)


def test_memo_is_written_only_after_a_full_hash(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)

    def interrupted(_):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        digest_memo.digest(f, str(tmp_path / "memo"), hasher=interrupted)
    assert not (tmp_path / "memo" / digest_memo.MEMO_NAME).exists()


def test_a_hit_skips_hashing_and_names_the_cached_time(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    h1, c1 = digest_memo.digest(f, memo)
    assert c1 is None and h1 == digest_memo.sha256_file(f)
    calls = []
    h2, c2 = digest_memo.digest(f, memo, hasher=lambda p: calls.append(p) or "0" * 64)
    assert calls == [] and h2 == h1 and c2 is not None
    assert digest_memo.word("pinned", c2) == "pinned (cached digest %s)" % c2
    assert digest_memo.word("unknown", None) == "unknown"


def test_a_changed_stat_key_is_hashed_again(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    digest_memo.digest(f, memo)
    _write(tmp_path / "w.bin", b"y" * 101)
    h, c = digest_memo.digest(f, memo)
    assert c is None and h == digest_memo.sha256_file(f)


def test_refresh_hashes_afresh_and_rewrites_the_entry(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    digest_memo.digest(f, memo, hasher=lambda p: "1" * 64)  # a stale entry
    calls = []
    h, c = digest_memo.digest(f, memo, refresh=True, hasher=lambda p: calls.append(p) or digest_memo.sha256_file(p))
    assert calls == [os.path.realpath(f)] and c is None and h == digest_memo.sha256_file(f)
    table = json.load(open(os.path.join(memo, digest_memo.MEMO_NAME)))
    assert table[digest_memo.stat_key(f)]["sha256"] == h


def test_the_match_is_by_digest_not_by_stat(tmp_path):
    a = _write(tmp_path / "a.bin", b"x" * 100)
    b = _write(tmp_path / "b.bin", b"z" * 100)  # same size, other bytes
    memo = str(tmp_path / "memo")
    ha, _ = digest_memo.digest(a, memo)
    hb, cb = digest_memo.digest(b, memo)
    assert ha != hb and cb is None


def test_rank_processes_read_the_launchers_entry_or_fail_by_name(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    with pytest.raises(RuntimeError, match="memo miss in a rank process"):
        digest_memo.rank_digest(f, memo)
    h, _ = digest_memo.digest(f, memo)
    assert digest_memo.rank_digest(f, memo)[0] == h


def test_a_corrupt_memo_file_is_ignored_not_trusted(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = tmp_path / "memo"
    memo.mkdir()
    (memo / digest_memo.MEMO_NAME).write_text("{not json")
    h, c = digest_memo.digest(f, str(memo))
    assert c is None and h == digest_memo.sha256_file(f)


REFERENCE_SHA256 = "d4500c2a8e7aea122cff64cdcd367e54ece302ed5dca7dcd31c6be788bda8354"   # the one text every kit carries


def test_the_helper_is_the_reference_text_byte_for_byte():
    with open(digest_memo.__file__, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == REFERENCE_SHA256

def test_check_hashes_afresh_and_pred_serve_read_the_memo(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, b"weights")
    calls = []
    real = digest_memo.digest

    def rec(path, memo_dir, refresh=False, hasher=digest_memo.sha256_file):
        calls.append(bool(refresh)); return real(path, memo_dir, refresh=refresh, hasher=hasher)
    monkeypatch.setattr(digest_memo, "digest", rec)
    site = _stubs.make_site(str(tmp_path / "site")); monkeypatch.syspath_prepend(site)
    for k in list(os.environ):
        if k.startswith(("ODDE_", "OPENDDE_OPT", "DIT_", "CUEQ_", "CUBLAS_", "PYTORCH_CUDA")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE); monkeypatch.setenv("OPENDDE_ROOT_DIR", root)
    monkeypatch.setattr(stack, "shim_active", lambda: None); monkeypatch.setattr(stack, "_REPORT", None)
    assert cli.main(["check", "--mode", "off"]) == cli.EXIT_OK and calls == [True]          # `check`: refresh=True (hashed afresh, entry rewritten)
    out = capsys.readouterr().out
    assert "WEIGHTS sha256=" in out and "(cached digest" not in out
    w = manifest.weights(root, tree=_stubs.TREE)                                            # the `pred` call form: refresh=False -> the memo serves it
    assert calls == [True, False] and w["digest_source"] == "memo" and w["digest_cached_utc"]
    assert "(cached digest " in manifest.weights_line(w)
    src_cli = open(os.path.join(os.path.dirname(manifest.__file__), "cli.py")).read()
    assert src_cli.count("refresh=True") == 1 and "cmd_check" in src_cli.split("refresh=True")[0].rsplit("def ", 1)[-1]   # only the `check` verb refreshes


def test_a_rank_process_reads_the_launchers_entry_and_never_hashes(tmp_path, monkeypatch):
    from opt_core.mem.rowpair import launch
    root = _root(tmp_path, b"weights of a P>1 run")
    monkeypatch.delenv(manifest.DIGEST_RELAY_ENV, raising=False)
    monkeypatch.setenv(launch.ENV_RANK, "1")
    with pytest.raises(RuntimeError, match="digest memo miss in a rank process"):          # no launcher entry yet: an error by name, never a re-hash
        manifest.weights(root, tree=_stubs.TREE)
    monkeypatch.delenv(launch.ENV_RANK)
    w0 = manifest.weights(root, tree=_stubs.TREE)                                           # the launcher (cmd_pred before run_ranks) digests ...
    monkeypatch.setenv(launch.ENV_RANK, "1")
    monkeypatch.setattr(digest_memo, "sha256_file", lambda *a, **k: pytest.fail("a rank process hashed"))
    w1 = manifest.weights(root, tree=_stubs.TREE)                                           # ... the rank reads its entry
    assert w1["checkpoint_sha256"] == w0["checkpoint_sha256"] and w1["digest_source"] == "rank:launcher_memo" and w1["checkpoint_match"] == w0["checkpoint_match"]


def test_a_read_only_memo_is_worded_without_exception_text_and_other_oserrors_stay_loud(tmp_path, monkeypatch):
    """K.29 — (a) a read-only / permission-denied memo directory (EROFS, EACCES, EPERM) is benign: `digest memo read-only — hashed afresh`, no
    exception class or errno text on the WEIGHTS line; (b) a writable memo: the line as always, no memo clause; (c) any other OSError on the memo
    (ENOSPC here) stays loud: hashed afresh AND its exception text on the line (the current contract; nothing is swallowed)."""
    import errno
    root = _root(tmp_path, b"weights"); fresh = hashlib.sha256(b"weights").hexdigest()
    for e in (OSError(errno.EROFS, "Read-only file system", "/jitcache/weights"), PermissionError(errno.EACCES, "Permission denied"), OSError(errno.EPERM, "Operation not permitted")):
        def ro(path, memo_dir, refresh=False, hasher=None, _e=e): raise _e
        monkeypatch.setattr(digest_memo, "digest", ro)
        w = manifest.weights(root, tree=_stubs.TREE); line = manifest.weights_line(w)
        assert w["digest_source"] == "hashed:memo_readonly" and w["digest_memo_readonly"] is True and w["digest_memo_error"] is None and w["checkpoint_sha256"] == fresh
        assert "(digest memo read-only — hashed afresh)" in line and not any(x in line for x in ("OSError", "Errno", "Error", "Traceback", "Permission")), line   # (a)
    monkeypatch.undo()
    w = manifest.weights(_root(tmp_path / "b", b"weights"), tree=_stubs.TREE); line = manifest.weights_line(w)                   # (b) a writable memo (the default dir under tmp HOME)
    assert w["digest_memo_readonly"] is False and w["digest_memo_error"] is None and "digest memo" not in line and w["digest_source"] in ("hashed", "memo"), line
    def full(path, memo_dir, refresh=False, hasher=None): raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(digest_memo, "digest", full)
    w = manifest.weights(root, tree=_stubs.TREE); line = manifest.weights_line(w)                                                # (c) not benign: loud, as before
    assert w["digest_source"] == "hashed:memo_unwritable" and w["digest_memo_readonly"] is False and w["checkpoint_sha256"] == fresh
    assert "(digest memo unwritable, hashed afresh: OSError: [Errno 28] No space left on device)" in line, line


def test_ranks_receive_the_launchers_digest_when_the_memo_is_read_only(tmp_path, monkeypatch):
    """A read-only memo directory (a cache volume mounted read-only): the launcher hashes afresh and names it, and its rank processes take the
    digest it relays through their inherited environment — no rank hashes, no rank dies on the memo miss."""
    import errno
    from opt_core.mem.rowpair import launch
    root = _root(tmp_path, b"weights of a P>1 run on a read-only cache")
    monkeypatch.delenv(manifest.DIGEST_RELAY_ENV, raising=False)
    def erofs(*a, **k): raise OSError(errno.EROFS, "Read-only file system")
    monkeypatch.setattr(digest_memo, "_store", erofs)                                       # the memo cannot be written (root bypasses a 0o555 dir, so the store itself refuses)
    w0 = manifest.weights(root, tree=_stubs.TREE)                                           # the launcher (cmd_pred's WEIGHTS line, before run_ranks)
    assert w0["digest_source"] == "hashed:memo_readonly" and "(digest memo read-only — hashed afresh)" in manifest.weights_line(w0)   # EROFS: benign wording, no exception text
    assert os.environ[manifest.DIGEST_RELAY_ENV].split("\t")[1] == w0["checkpoint_sha256"]  # relayed to the processes it launches
    monkeypatch.setenv(launch.ENV_RANK, "2")
    monkeypatch.setattr(digest_memo, "sha256_file", lambda *a, **k: pytest.fail("a rank process hashed"))
    w2 = manifest.weights(root, tree=_stubs.TREE)                                           # a rank: memo miss -> the launcher's relayed digest
    assert w2["checkpoint_sha256"] == w0["checkpoint_sha256"] and w2["digest_source"] == "rank:launcher_env" and w2["checkpoint_match"] == w0["checkpoint_match"]
    monkeypatch.setenv(manifest.DIGEST_RELAY_ENV, "[\"another file\", 1, 2, 3]\t" + "0" * 64 + "\tfresh")   # a relay for a different file is not this checkpoint's: the miss by name
    with pytest.raises(RuntimeError, match="digest memo miss in a rank process"):
        manifest.weights(root, tree=_stubs.TREE)
