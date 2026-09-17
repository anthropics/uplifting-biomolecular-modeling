"""The offload unit's stage checkpoints (ODDE_OFFLOAD_CKPT_DIR; `torch.load` unpickles them): the sha256 recorded beside the file is checked
BEFORE a byte is unpickled, a file or directory another account could have written is refused, every refusal is one by-name stderr line (what,
why, the fix) followed by exactly the absent-checkpoint behaviour (None: the stage runs), and what ckpt_save writes — under any umask — is
never refused."""
import hashlib
import importlib.util
import os
import pickle
import stat
import sys

import pytest

torch = pytest.importorskip("torch")

UNIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "fast_inference", "levers", "OFFLOAD", "odde_offload.py")
TRIPPED = []


def _trip(word):
    TRIPPED.append(word)
    return word


class _Trap:
    """Unpickling this calls `_trip` — the stand-in for a pickle that runs code."""

    def __reduce__(self):
        return (_trip, ("unpickled",))


@pytest.fixture()
def oo(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("odde_offload_under_test", UNIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(mod.CFG, "ckpt_dir", str(tmp_path / "ck"))
    TRIPPED.clear()
    old = os.umask(0o022)
    yield mod
    os.umask(old)


def _obj():
    return {"cycle_done": 3, "s": torch.arange(6.0).reshape(2, 3), "z": torch.ones(2, 2, 4), "z_is_host": True}


@pytest.mark.parametrize("umask", (0o022, 0o002, 0o000, 0o077))
def test_saved_checkpoint_has_its_sha256_record_and_loads_whatever_the_umask(oo, capfd, umask):
    os.umask(umask)
    oo.ckpt_save("trunk", _obj())
    d = oo.CFG["ckpt_dir"]; p = os.path.join(d, "trunk.pt")
    assert open(p + ".sha256").read().strip() == hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert all(os.stat(q).st_mode & 0o022 == 0 for q in (d, p, p + ".sha256")) and not os.path.exists(p + ".tmp")
    ck = oo.ckpt_load("trunk")
    assert ck["cycle_done"] == 3 and torch.equal(ck["s"], _obj()["s"]) and ck["z_is_host"] is True
    assert "REFUSED" not in capfd.readouterr().err


@pytest.mark.parametrize("damage", ("other_bytes", "truncated", "no_record"))
def test_checkpoint_whose_sha256_differs_or_is_missing_is_never_unpickled(oo, capfd, damage):
    oo.ckpt_save("trunk", _obj())
    p = os.path.join(oo.CFG["ckpt_dir"], "trunk.pt")
    if damage == "no_record":
        os.remove(p + ".sha256")
        with open(p, "wb") as f:
            f.write(pickle.dumps(_Trap()))
    else:
        good = open(p, "rb").read()
        with open(p, "wb") as f:
            f.write(pickle.dumps(_Trap()) if damage == "other_bytes" else good[:-9])
    assert oo.ckpt_load("trunk") is None and TRIPPED == []
    err = capfd.readouterr().err
    assert err.count("REFUSED checkpoint " + p) == 1 and "sha256" in err and "fix:" in err and "treated as absent" in err
    oo.ckpt_save("trunk", _obj())                                    # the stage ran and saved again: the next resume is served, silently
    assert oo.ckpt_load("trunk")["cycle_done"] == 3 and "REFUSED" not in capfd.readouterr().err


@pytest.mark.parametrize("target", ("directory", "file"))
@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_group_or_other_writable_is_refused_with_the_fix_then_loads_after_it(oo, monkeypatch, capfd, target, bit):
    oo.ckpt_save("trunk", _obj())
    d = oo.CFG["ckpt_dir"]; p = os.path.join(d, "trunk.pt"); q = d if target == "directory" else p
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) | bit)
    with monkeypatch.context() as m:
        m.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load ran on a refused checkpoint"))
        assert oo.ckpt_load("trunk") is None
    err = capfd.readouterr().err
    assert f"REFUSED checkpoint {p}: {target} {q} is writable by group or other" in err and f"fix: chmod go-w {q}" in err
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) & ~0o022)           # the printed fix
    assert oo.ckpt_load("trunk")["cycle_done"] == 3


def test_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0(oo, monkeypatch, capfd):
    oo.ckpt_save("trunk", _obj())
    owner = os.stat(os.path.join(oo.CFG["ckpt_dir"], "trunk.pt")).st_uid
    if owner != 0:
        monkeypatch.setattr(os, "geteuid", lambda: owner + 1)        # this process is another (non-root) user
        assert oo.ckpt_load("trunk") is None
        assert f"belongs to uid {owner}, not to this process (uid {owner + 1}) or root" in capfd.readouterr().err
    monkeypatch.setattr(os, "geteuid", lambda: 0)                    # a container's root reading a bind-mounted host directory
    assert oo.ckpt_load("trunk")["cycle_done"] == 3


def test_the_resume_sites_take_a_refused_checkpoint_as_no_checkpoint():
    """The three resume sites read `ckpt_load` inside their condition, so None (refused) falls through to the stage that runs."""
    here = os.path.dirname(UNIT)
    src = open(UNIT, encoding="utf-8").read() + open(os.path.join(here, "odde_offload_trunk.py"), encoding="utf-8").read()
    for name in ("refiner", "trunk", "trunk_cycle"):
        assert src.count(f'ckpt_load("{name}")) is not None:') == 1, name
    assert src.count("ckpt_load(") == 4                              # the three sites + the definition: no site reads it unguarded
