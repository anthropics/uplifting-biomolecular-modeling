"""configs/<card>.env and AF3_TORCH_CACHE_ROOT (the Triton / Inductor / JAX cache root the model processes read compiled code back from): an
explicit value wins; else <MODEL_OPT_JIT_ROOT>/<stack key> (run.sh exports that root); else — the file sourced by hand — a per-user root under the
temporary directory held to the private-directory rule: made 0700, used only when it is a directory this user owns that is neither a symbolic
link nor writable by group or others, otherwise refused by name on stderr and replaced by a fresh private directory for that shell. Never a
fixed name another account could have made first."""
import os
import stat
import subprocess

import pytest

from .conftest import HOME

CARDS = {"h100": "torch2.13.0-cu130-sm90", "h200": "torch2.13.0-cu130-sm90", "a100": "torch2.13.0-cu130-sm80"}


def _source(tmp_path, card, env=None, tmpdir=None):
    """Source configs/<card>.env in a clean bash with a stub `python` (the exports probe passes, prints nothing); returns (rc, AF3_TORCH_CACHE_ROOT, stderr)."""
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    py = b / "python"; py.write_text("#!/bin/sh\nexit 0\n"); py.chmod(py.stat().st_mode | stat.S_IXUSR)
    t = tmpdir or (tmp_path / "tmp"); t.mkdir(exist_ok=True)
    e = {"PATH": f"{b}:/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(t)}; e.update(env or {})
    p = subprocess.run(["bash", "-c", f'source "{HOME}/configs/{card}.env"; rc=$?; printf "%s\\n" "$rc" "${{AF3_TORCH_CACHE_ROOT-<unset>}}"'],
                       env=e, capture_output=True, text=True, timeout=60)
    rc, root = p.stdout.splitlines()[:2]
    return int(rc), root, p.stderr


@pytest.mark.parametrize("card", sorted(CARDS))
def test_explicit_value_and_jit_root(tmp_path, card):
    assert _source(tmp_path, card, {"AF3_TORCH_CACHE_ROOT": "/my/root", "MODEL_OPT_JIT_ROOT": "/a/jit"})[:2] == (0, "/my/root")   # an explicit value wins
    rc, root, err = _source(tmp_path, card, {"MODEL_OPT_JIT_ROOT": "/a/jit"})                                                        # run.sh's exported root: <root>/<stack key>
    assert (rc, root) == (0, "/a/jit/" + CARDS[card]) and "refused" not in err
    assert not (tmp_path / "tmp" / f"af3_torch_cache-uid{os.getuid()}").exists()                                                   # nothing made under the temporary directory then


@pytest.mark.parametrize("card", sorted(CARDS))
def test_by_hand_the_root_is_per_user_private_and_reused(tmp_path, card):
    mine = tmp_path / "tmp" / f"af3_torch_cache-uid{os.getuid()}"
    rc, root, err = _source(tmp_path, card)
    assert (rc, root) == (0, f"{mine}/{CARDS[card]}") and "refused" not in err, (rc, root, err)
    assert mine.is_dir() and not mine.is_symlink() and stat.S_IMODE(mine.stat().st_mode) == 0o700                                   # made here, owner-only, whatever the umask
    assert _source(tmp_path, card)[:2] == (0, f"{mine}/{CARDS[card]}")                                                              # a later shell of this user: the same root (its caches are reused)
    assert "/tmp/af3_torch_cache" not in open(os.path.join(HOME, "configs", f"{card}.env"), encoding="utf-8").read().split("_cache-uid")[0]   # no fixed shared name left in the file


@pytest.mark.parametrize("card", sorted(CARDS))
@pytest.mark.parametrize("plant", ["open", "link"])
def test_a_planted_root_is_refused_by_name(tmp_path, card, plant):
    """A directory at the per-user name that is open to group or others, or a symbolic link there (what another account could leave in a shared
    temporary directory), is refused by name; the shell gets a fresh private directory instead and nothing under the planted path is used."""
    t = tmp_path / "tmp"; t.mkdir()
    planted = t / f"af3_torch_cache-uid{os.getuid()}"
    if plant == "open":
        planted.mkdir(); planted.chmod(0o777)
    else:
        (tmp_path / "elsewhere").mkdir(); planted.symlink_to(tmp_path / "elsewhere")
    rc, root, err = _source(tmp_path, card)
    assert rc == 0 and f"configs/{card}.env: {planted} refused (another owner, open to group or others, a symbolic link, or not creatable)" in err, err
    assert root.endswith("/" + CARDS[card]) and not root.startswith(str(planted) + "/") and root.startswith(str(t) + "/af3_torch_cache."), root
    private = os.path.dirname(root)
    assert os.path.isdir(private) and not os.path.islink(private) and stat.S_IMODE(os.stat(private).st_mode) == 0o700 and os.stat(private).st_uid == os.getuid()
    if plant == "link":
        assert os.listdir(tmp_path / "elsewhere") == []                                                                              # nothing written through the link


def test_group_readable_own_directory_is_fine(tmp_path):
    """The rule is about who can WRITE: a per-user directory of one's own with mode 0750 (group may read) is used as it is, silently."""
    t = tmp_path / "tmp"; t.mkdir()
    mine = t / f"af3_torch_cache-uid{os.getuid()}"; mine.mkdir(); mine.chmod(0o750)
    rc, root, err = _source(tmp_path, "h100")
    assert (rc, root) == (0, f"{mine}/{CARDS['h100']}") and "refused" not in err
    mine.chmod(0o770)                                                                                                                 # group-writable: refused
    assert "refused" in _source(tmp_path, "h100")[2]
