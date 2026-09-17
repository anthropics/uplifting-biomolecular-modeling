"""The ranks' file store (c10d rendezvous data, read by every rank) lives under a per-user directory this user alone can have written when
ROWPAIR_STORE is not set: <tmp>/protenix_opt_rowpair-uid<uid>, made 0700; a planted one (symbolic link, another owner, group/other write
bit) is refused by name with the fix — never a fixed parent every account shares (tp_route.store_parent / bridge_env)."""
import os
import re
import stat

import pytest

from protenix_opt import tp_route

RANK_ENV = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29642", "TORCHELASTIC_RUN_ID": "run9"}


def _want(tmp):
    return os.path.join(str(tmp), "protenix_opt_rowpair-uid%d" % os.geteuid())


def test_no_fixed_shared_parent_in_the_source():
    src = open(tp_route.__file__, encoding="utf-8").read()
    assert not re.search(r"""gettempdir\(\), ["']protenix_opt_rowpair["']""", src)      # the old parent: one name for every account


def test_the_parent_is_made_private_and_the_store_keyed_by_run(tmp_path):
    old = os.umask(0o002)
    try:
        env = dict(RANK_ENV, TMPDIR=str(tmp_path))
        tp_route.bridge_env(env)
    finally:
        os.umask(old)
    parent = _want(tmp_path)
    assert env["ROWPAIR_STORE"] == os.path.join(parent, "run9", "c10d_store")
    st = os.lstat(parent)
    assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700 and st.st_uid == os.geteuid()
    assert stat.S_IMODE(os.stat(os.path.join(parent, "run9")).st_mode) & 0o022 == 0


def test_every_rank_of_a_run_derives_the_same_store(tmp_path):
    a = dict(RANK_ENV, TMPDIR=str(tmp_path)); b = dict(RANK_ENV, RANK="1", LOCAL_RANK="1", TMPDIR=str(tmp_path))
    tp_route.bridge_env(a); tp_route.bridge_env(b)
    assert a["ROWPAIR_STORE"] == b["ROWPAIR_STORE"]                     # the rendezvous needs one path; an earlier private parent of this user is reused silently


@pytest.mark.parametrize("mode", [0o777, 0o775, 0o702])
def test_a_parent_open_to_group_or_others_is_refused_by_name(tmp_path, mode):
    planted = _want(tmp_path); os.mkdir(planted); os.chmod(planted, mode)
    env = dict(RANK_ENV, TMPDIR=str(tmp_path))
    with pytest.raises(tp_route.StoreDirRefused) as e:
        tp_route.bridge_env(env)
    msg = str(e.value)
    assert planted in msg and "writable by group or other" in msg and "ROWPAIR_STORE" in msg
    assert "ROWPAIR_STORE" not in env and os.listdir(planted) == [] and stat.S_IMODE(os.stat(planted).st_mode) == mode


def test_a_symbolic_link_is_refused_by_name(tmp_path):
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(mode=0o700)
    planted = _want(tmp_path); os.symlink(elsewhere, planted)
    with pytest.raises(tp_route.StoreDirRefused) as e:
        tp_route.store_parent({"TMPDIR": str(tmp_path)})
    assert "symbolic link" in str(e.value) and list(elsewhere.iterdir()) == []


def test_a_named_store_is_used_as_given_and_nothing_is_made(tmp_path):
    env = dict(RANK_ENV, TMPDIR=str(tmp_path), ROWPAIR_STORE="/x/store")
    tp_route.bridge_env(env)
    assert env["ROWPAIR_STORE"] == "/x/store" and not os.path.lexists(_want(tmp_path))
