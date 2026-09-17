"""The cached in-place build of the fused residual+LayerNorm extension is imported as found, and its sha record sits beside it — whoever can
write the directory can rewrite both — so the owner / mode rule decides: a cache directory another account could have written is REFUSED on
stderr (what, why, the fix), is then no cached build, and the rebuild goes to a private directory; a cache of one's own — whatever modes the
builder left inside a directory closed to group and other — is imported silently."""
import hashlib
import json
import os
import stat

import pytest

from esmc_opt.kits import residual_ln as rl


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setitem(rl._state, "refused", None)
    old = os.umask(0o022)
    yield
    os.umask(old)


def _cache(tmp_path, dir_mode=0o700, so_mode=0o755):
    cdir = str(tmp_path / "home" / ".cache" / "esmc_sdkfused" / "torch2__cuda12__sm90__py311__abcdef012345")
    os.makedirs(os.path.join(cdir, "build"))
    so = os.path.join(cdir, "build", "ext.so")
    with open(so, "wb") as f:
        f.write(b"ELF:built")
    os.chmod(so, so_mode)
    with open(os.path.join(cdir, rl.BUILT_FILE), "w") as f:
        json.dump({"so_rel": "build/ext.so", "size": 9, "so_sha256": hashlib.sha256(b"ELF:built").hexdigest()}, f)
    os.chmod(cdir, dir_mode)
    return cdir, so


@pytest.mark.parametrize("so_mode", (0o755, 0o775))
def test_a_cache_of_ones_own_closed_to_others_is_the_cached_build_silently(tmp_path, capfd, so_mode):
    cdir, so = _cache(tmp_path, 0o700, so_mode)
    assert rl._cached_build(cdir)[0] == so and capfd.readouterr().err == ""
    assert rl._writable_cache_dir(cdir) == cdir


@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_a_group_or_other_writable_cache_is_refused_by_name_and_the_rebuild_goes_to_a_private_directory(tmp_path, capfd, bit):
    cdir, so = _cache(tmp_path, 0o755 | bit)
    assert rl._cached_build(cdir) is None
    err = capfd.readouterr().err
    assert err.count(f"[residual_ln] REFUSED cached build {cdir}: {cdir} is writable by group or other") == 1 and f"fix: chmod go-w {cdir}" in err
    private = rl._writable_cache_dir(cdir)
    assert private != cdir and stat.S_IMODE(os.stat(private).st_mode) == 0o700
    os.chmod(cdir, 0o755); rl._state["refused"] = None               # the printed fix
    assert rl._cached_build(cdir)[0] == so and rl._writable_cache_dir(cdir) == cdir


def test_a_binary_others_could_have_written_in_a_cache_they_can_enter_is_refused(tmp_path, capfd):
    cdir, so = _cache(tmp_path, 0o755, 0o775)
    assert rl._cached_build(cdir) is None and f"{so} is writable by group or other (mode 0775); fix: chmod go-w {so}" in capfd.readouterr().err


def test_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0(tmp_path, monkeypatch, capfd):
    cdir, so = _cache(tmp_path, 0o755)
    owner = os.stat(cdir).st_uid
    if owner != 0:
        monkeypatch.setattr(os, "geteuid", lambda: owner + 1)        # this process is another (non-root) user
        assert rl._cached_build(cdir) is None
        assert f"belongs to uid {owner}, not to this process (uid {owner + 1}) or root" in capfd.readouterr().err
    monkeypatch.setattr(os, "geteuid", lambda: 0)                    # a container's root reading a bind-mounted host directory
    assert rl._cached_build(cdir)[0] == so


@pytest.mark.parametrize("umask", (0o022, 0o002, 0o000))
def test_the_cache_directory_is_created_closed_to_group_and_other_whatever_the_umask(tmp_path, umask):
    os.umask(umask)
    cdir = str(tmp_path / "home" / ".cache" / "esmc_sdkfused" / "key")
    assert rl._writable_cache_dir(cdir) == cdir and stat.S_IMODE(os.stat(cdir).st_mode) == 0o700
