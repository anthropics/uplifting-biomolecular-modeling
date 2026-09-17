"""L13's stored programs (af2ig_opt.programs.Store) are pickles to ``deserialize_and_load``: the sha256 line is checked BEFORE a byte is
deserialized, a directory or file another account could have written is refused, every refusal is one by-name stderr line (what, why, the
fix) followed by exactly the absent-file behaviour (traced, stored again), and what the store itself writes — under any umask — is never
refused. jax is a stand-in here (the suite runs on a box without the stack)."""
import hashlib
import os
import stat
import sys
import types

import pytest

from af2ig_opt import ccache, programs

SER_CALLS = []


class _Tree:
    num_leaves = 1


class _Apply:
    """``jit(f)`` as aot_compile uses it: ``.trace(*args).lower().compile()``."""

    def trace(self, *args):
        return self

    def lower(self):
        return self

    def compile(self):
        return "compiled"


@pytest.fixture(autouse=True)
def stand_in_jax(monkeypatch):
    jax = types.ModuleType("jax")
    jax.tree_util = types.SimpleNamespace(tree_structure=lambda x: _Tree(), tree_unflatten=lambda t, leaves: leaves[0])
    jax.devices = lambda: [types.SimpleNamespace(platform="cpu", device_kind="cpu")]
    ser = types.ModuleType("jax.experimental.serialize_executable")
    ser.serialize = lambda compiled: (b"EXE:" + compiled.encode(), _Tree(), _Tree())

    def deserialize_and_load(payload, in_tree, out_tree):
        SER_CALLS.append(payload)
        return ("loaded", payload)

    ser.deserialize_and_load = deserialize_and_load
    exp = types.ModuleType("jax.experimental")
    exp.serialize_executable = ser
    jax.experimental = exp
    for name, mod in (("jax", jax), ("jax.experimental", exp), ("jax.experimental.serialize_executable", ser)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(programs, "provider_facts", lambda device="": {})
    SER_CALLS.clear()
    old = os.umask(0o022)
    yield
    os.umask(old)


def _store(tmp_path):
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "driver.py").write_text("x = 1\n")
    return programs.Store(str(tmp_path / "jit" / "programs"), {"recycle": 3}, str(code))


def _stored(tmp_path):
    s = _store(tmp_path)
    compiled, rec = s.program(_Apply(), (1,), "sig", 8)
    assert (compiled, rec["source"], s.census()["stored"]) == ("compiled", "traced", 1)
    return os.path.join(s.dir, rec["file"])


def test_stored_program_is_its_sha256_line_then_the_executable_and_loads(tmp_path, capfd):
    p = _stored(tmp_path)
    head, payload = open(p, "rb").read().split(b"\n", 1)
    assert head == b"sha256:" + hashlib.sha256(payload).hexdigest().encode() and payload == b"EXE:compiled"
    s = _store(tmp_path)
    compiled, rec = s.program(_Apply(), (1,), "sig", 8)
    c = s.census()
    assert compiled == ("loaded", b"EXE:compiled") and rec["source"] == "loaded" and rec["bytes"] == len(payload)
    assert (c["loaded"], c["traced"], c["stored"], c["load_failed"], c["events"]) == (1, 0, 0, 0, [])
    assert "REFUSED" not in capfd.readouterr().err


@pytest.mark.parametrize("damage", ("no_line", "wrong_line", "truncated", "empty"))
def test_program_without_a_matching_line_is_never_deserialized_and_is_rebuilt_once(tmp_path, capfd, damage):
    p = _stored(tmp_path)
    good = open(p, "rb").read()
    blob = {"no_line": b"EXE:other", "wrong_line": b"sha256:" + b"0" * 64 + b"\nEXE:other", "truncated": good[:-3], "empty": b""}[damage]
    with open(p, "wb") as f:
        f.write(blob)
    capfd.readouterr()
    s = _store(tmp_path)
    compiled, rec = s.program(_Apply(), (1,), "sig", 8)
    err = capfd.readouterr().err
    c = s.census()
    assert SER_CALLS == [] and compiled == "compiled" and rec["source"] == "traced"
    assert err.count("REFUSED cache file " + p) == 1 and "sha256" in err and "fix:" in err and "treated as absent" in err
    assert (c["loaded"], c["traced"], c["stored"], c["load_failed"], c["events"]) == (0, 1, 1, 0, ["load_refused"])
    s2 = _store(tmp_path)                                            # rebuilt once: the next process loads the rewritten file, silently
    assert s2.program(_Apply(), (1,), "sig", 8)[1]["source"] == "loaded" and s2.census()["events"] == []
    assert "REFUSED" not in capfd.readouterr().err


@pytest.mark.parametrize("target", ("directory", "file"))
@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_group_or_other_writable_is_refused_with_the_fix_then_loads_after_it(tmp_path, capfd, target, bit):
    p = _stored(tmp_path)
    q = os.path.dirname(p) if target == "directory" else p
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) | bit)
    s = _store(tmp_path)
    assert s.program(_Apply(), (1,), "sig", 8)[1]["source"] == "traced" and SER_CALLS == []
    err = capfd.readouterr().err
    assert f"REFUSED cache file {p}: {target} {q} is writable by group or other" in err and f"fix: chmod go-w {q}" in err
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) & ~0o022)           # the printed fix
    assert _store(tmp_path).program(_Apply(), (1,), "sig", 8)[1]["source"] == "loaded"


def test_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0(tmp_path, monkeypatch, capfd):
    p = _stored(tmp_path)
    owner = os.stat(p).st_uid
    if owner != 0:
        monkeypatch.setattr(os, "geteuid", lambda: owner + 1)        # this process is another (non-root) user
        assert _store(tmp_path).program(_Apply(), (1,), "sig", 8)[1]["source"] == "traced"
        assert f"belongs to uid {owner}, not to this process (uid {owner + 1}) or root" in capfd.readouterr().err
    monkeypatch.setattr(os, "geteuid", lambda: 0)                    # a container's root reading a bind-mounted host directory
    assert _store(tmp_path).program(_Apply(), (1,), "sig", 8)[1]["source"] == "loaded"


@pytest.mark.parametrize("umask", (0o002, 0o000, 0o077))
def test_what_the_store_writes_is_never_refused_whatever_the_umask(tmp_path, capfd, umask):
    os.umask(umask)
    d, why = ccache.programs_dir({"state": "on", "source": "env", "root": str(tmp_path / "root"), "key": "k", "recipe": "default"}, str(tmp_path / "dflt"))
    assert why is None and os.stat(d).st_mode & 0o022 == 0
    code = tmp_path / "code"
    code.mkdir()
    s = programs.Store(d, {}, str(code))
    rec = s.program(_Apply(), (1,), "sig", 8)[1]
    assert os.stat(os.path.join(d, rec["file"])).st_mode & 0o022 == 0
    assert programs.Store(d, {}, str(code)).program(_Apply(), (1,), "sig", 8)[1]["source"] == "loaded"
    assert "REFUSED" not in capfd.readouterr().err
