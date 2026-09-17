"""The checkpoint's digest state is a NOTE, never a gate: a checkpoint that is not the pinned one (stock/PINS.json checkpoint.sha256)
prints `WEIGHTS unknown … proceeding` and every command proceeds; the pinned one prints `WEIGHTS pinned …`; the frozen-weights
gate (_frozen) refuses only ABSENT frozen inputs under its switch and never reads a sha."""
import hashlib
import os

import pytest

from protenix_opt import _frozen, cli, digest_memo, manifest, stack


@pytest.fixture(autouse=True)
def _memo_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv(manifest.CACHE_DIR_ENV, str(tmp_path / "cache"))       # the weights digest memo under the test's own cache root


def _root(tmp_path, payload=b"another checkpoint entirely"):
    ck = tmp_path / "checkpoint"; ck.mkdir()
    (ck / "protenix-v2.pt").write_bytes(payload)
    return str(tmp_path), hashlib.sha256(payload).hexdigest()


def test_unknown_checkpoint_is_named_and_proceeds(tmp_path):
    root, sha = _root(tmp_path)
    ws = manifest.weights_status(stack.pins()["checkpoint"]["sha256"], root_dir=root)
    assert ws["digest"] == "unknown" and ws["line"].startswith("WEIGHTS unknown sha256=" + sha[:16]) and "proceeding" in ws["line"]
    assert stack.pins()["checkpoint"]["sha256"][:16] in ws["line"]            # names the pinned checkpoint it is not


def test_pinned_checkpoint_says_pinned(tmp_path):
    root, sha = _root(tmp_path, b"stand-in bytes")
    ws = manifest.weights_status(sha, root_dir=root)                          # the pin equals this file's sha: pinned
    assert ws["digest"] == "pinned" and ws["line"].startswith("WEIGHTS pinned sha256=" + sha[:16])


def test_absent_checkpoint_is_named_and_proceeds(tmp_path):
    ws = manifest.weights_status("0" * 64, root_dir=str(tmp_path))
    assert ws["digest"] == "absent" and ws["line"].startswith("WEIGHTS absent") and "proceeding" in ws["line"]


def test_no_entry_point_gate_reads_the_checkpoint_sha(tmp_path, monkeypatch, capsys):
    root, _ = _root(tmp_path)
    for rel in _frozen.DATA_CACHES:                                            # the frozen inputs present: the gate has nothing to refuse, whatever the sha
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"")
    env = {"PROTENIX_ROOT_DIR": root, _frozen.SWITCH: "1"}
    assert _frozen.check(env, ["--model_name", manifest.CHECKPOINT_NAME]) is None
    assert _frozen.check({"PROTENIX_ROOT_DIR": root}, ["--model_name", manifest.CHECKPOINT_NAME]) is None
    monkeypatch.setenv("PROTENIX_ROOT_DIR", root)
    ws = cli.weights_note()                                                    # the CLI's one printer: a line on stderr, a record back, no exit
    assert ws["digest"] == "unknown" and "[protenix-opt] WEIGHTS unknown" in capsys.readouterr().err


def test_check_hashes_afresh_pred_takes_the_memo(monkeypatch):
    """`check` passes refresh=True to the memo (always re-hashes and rewrites the entry); `pred` passes refresh=False."""
    seen = []
    def fake_status(pins_sha256, root_dir=None, refresh=False, memo_dir=None):
        seen.append(refresh); return {"digest": "absent", "line": "WEIGHTS absent (mocked); proceeding"}
    monkeypatch.setattr(manifest, "weights_status", fake_status)
    src = open(cli.__file__).read()
    import re
    calls = {m.group(1): m.group(2) for m in re.finditer(r"^def (cmd_pred|cmd_check)\b[\s\S]*?^\s+weights_note\((refresh=True)?\)", src, re.M)}
    assert calls == {"cmd_pred": None, "cmd_check": "refresh=True"}, calls
    cli.weights_note(); cli.weights_note(refresh=True)
    assert seen == [False, True]


def test_the_memo_names_the_cached_time_and_the_state_stays_by_digest(tmp_path):
    root, sha = _root(tmp_path, b"stand-in bytes")
    memo = str(tmp_path / "memo")
    w1 = manifest.weights_status(sha, root_dir=root, memo_dir=memo)                     # hashed now: no cached word; the memo is written
    assert w1["digest"] == "pinned" and "(cached digest" not in w1["line"] and w1["digest_cached_utc"] is None
    assert os.path.isfile(os.path.join(memo, digest_memo.MEMO_NAME))
    w2 = manifest.weights_status(sha, root_dir=root, memo_dir=memo)                     # a hit: '… (cached digest <utc>)', same digest, same state
    assert w2["digest"] == "pinned" and w2["line"].startswith("WEIGHTS pinned (cached digest ") and w2["sha256"] == w1["sha256"]
    w3 = manifest.weights_status("0" * 64, root_dir=root, memo_dir=memo)                # by digest only: the same memo entry, another pin -> unknown, proceeding
    assert w3["digest"] == "unknown" and w3["line"].startswith("WEIGHTS unknown (cached digest ") and "proceeding" in w3["line"]
    w4 = manifest.weights_status(sha, root_dir=root, memo_dir=memo, refresh=True)       # check's form: afresh, entry rewritten, no cached word
    assert w4["digest_cached_utc"] is None and "(cached digest" not in w4["line"]


def test_an_unwritable_memo_directory_is_named_and_the_digest_is_computed_afresh(tmp_path):
    """A memo directory that cannot be written (a read-only mount, a path under a regular file, no permission) never stops a command and is
    never silent: the digest is computed afresh, the state is decided as always, and the WEIGHTS line names the memo failure."""
    root, sha = _root(tmp_path, b"stand-in bytes")
    blocker = tmp_path / "not-a-dir"; blocker.write_text("a regular file: nothing can be created beneath it (root or not)")
    ws = manifest.weights_status(sha, root_dir=root, memo_dir=str(blocker / "memo"))
    assert ws["digest"] == "pinned" and ws["sha256"] == sha and ws["digest_cached_utc"] is None
    assert ws["line"].startswith("WEIGHTS pinned sha256=" + sha[:16]) and "digest memo unwritable at" in ws["line"] and "hashed afresh" in ws["line"]
    assert "error" not in ws
    ws2 = manifest.weights_status("0" * 64, root_dir=root, memo_dir=str(blocker / "memo"))          # unknown stays unknown-and-proceeding, with the memo note
    assert ws2["digest"] == "unknown" and "proceeding" in ws2["line"] and "digest memo unwritable" in ws2["line"]
