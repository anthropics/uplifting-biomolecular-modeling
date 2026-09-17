"""configs/<card>.env: the JIT caches (torch's extension builds, whose .so files are imported, and Triton's binaries, loaded as found)
live under a cache root this user alone can have written — MODEL_OPT_JIT_ROOT as named by the caller or by run.sh, else the per-user
root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` made 0700 and checked (owner this uid, no group/other write bit, not a symbolic link);
a root that fails the check is refused by name on stderr and a fresh private directory serves that run. Never a fixed name in the
shared temporary directory. The file is sourced here for real, with `python` and run.sh stubbed (the probes are not under test)."""
import os
import re
import stat
import subprocess

import pytest

KIT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CARDS = ["h100", "a100", "h200"]
KEY = "torch2.13.0-cu130-sm90"


def _rig(tmp_path):
    """A stub kit dir (run.sh probe -> 0) and a stub `python` first on PATH (the stack-key probe prints KEY; every other call -> 0)."""
    kit = tmp_path / "kit"; kit.mkdir()
    (kit / "run.sh").write_text("#!/bin/bash\nexit 0\n")
    bindir = tmp_path / "bin"; bindir.mkdir()
    py = bindir / "python"
    py.write_text('#!/bin/bash\ncase "$*" in *protenix_v1_opt._stackkey*) echo %s ;; esac\nexit 0\n' % KEY)
    py.chmod(0o755)
    tmp = tmp_path / "tmp"; tmp.mkdir()
    return kit, bindir, tmp


def _source(tmp_path, card, extra_env=None, strict=False, pre=None):
    kit, bindir, tmp = pre or _rig(tmp_path)
    env = {"PATH": str(bindir) + os.pathsep + "/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp), "MODEL_OPT": str(kit),
           "PROTENIX_ROOT_DIR": "/w"}
    env.update(extra_env or {})
    cfg = os.path.join(KIT, "configs", card + ".env")
    script = ("set -euo pipefail; " if strict else "") + f'source "{cfg}"; echo "RC=$?"; echo "ROOT=${{MODEL_OPT_JIT_ROOT:-}}"; ' \
             'echo "EXT=${TORCH_EXTENSIONS_DIR:-}"; echo "TRI=${TRITON_CACHE_DIR:-}"; echo "KEYED=${MODEL_OPT_STACK_KEY:-}"'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    got = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    return r, got, tmp


@pytest.mark.parametrize("card", CARDS)
def test_no_card_names_a_fixed_directory_in_the_shared_tmp(card):
    text = open(os.path.join(KIT, "configs", card + ".env"), encoding="utf-8").read()
    assert not re.search(r":-/tmp/\w", text), "a `${VAR:-/tmp/<name>}` default is a name every account shares"
    assert "/tmp/torch_extensions" not in text and "/tmp/triton_cache" not in text


@pytest.mark.parametrize("card", CARDS)
@pytest.mark.parametrize("strict", [False, True])
def test_without_a_named_root_the_per_user_root_is_made_private_and_keyed(tmp_path, card, strict):
    r, got, tmp = _source(tmp_path, card, strict=strict)
    want = tmp / ("model_opt_jit-uid%d" % os.getuid())
    assert r.returncode == 0 and got["RC"] == "0", r.stderr
    assert got["ROOT"] == str(want) and got["KEYED"] == KEY
    assert got["EXT"] == f"{want}/{KEY}/torch_extensions" and got["TRI"] == f"{want}/{KEY}/triton"
    st = os.lstat(want)
    assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700 and st.st_uid == os.getuid()
    assert "refused" not in r.stderr and "NOT ACTIVE" not in r.stderr, r.stderr


@pytest.mark.parametrize("card", CARDS)
def test_an_earlier_private_root_of_this_user_is_reused_silently(tmp_path, card):
    rig = _rig(tmp_path)
    want = rig[2] / ("model_opt_jit-uid%d" % os.getuid()); want.mkdir(mode=0o700); (want / "marker").write_text("x")
    r, got, _ = _source(tmp_path, card, pre=rig)
    assert got["ROOT"] == str(want) and "refused" not in r.stderr and (want / "marker").read_text() == "x"


@pytest.mark.parametrize("card", CARDS)
@pytest.mark.parametrize("mode", [0o777, 0o775, 0o702])
def test_a_root_open_to_group_or_others_is_refused_by_name(tmp_path, card, mode):
    rig = _rig(tmp_path)
    planted = rig[2] / ("model_opt_jit-uid%d" % os.getuid()); planted.mkdir(); os.chmod(planted, mode)
    r, got, tmp = _source(tmp_path, card, pre=rig)
    assert r.returncode == 0 and got["RC"] == "0", r.stderr
    assert got["ROOT"] != str(planted) and got["ROOT"].startswith(str(tmp / "protenix_v1_jit-"))
    assert stat.S_IMODE(os.stat(got["ROOT"]).st_mode) == 0o700
    assert got["EXT"] == f"{got['ROOT']}/{KEY}/torch_extensions" and got["TRI"] == f"{got['ROOT']}/{KEY}/triton"
    lines = [l for l in r.stderr.splitlines() if "refused" in l]
    assert len(lines) == 1 and lines[0].startswith(f"[protenix-v1-opt] jit cache: {planted} refused (") and got["ROOT"] in lines[0], r.stderr
    assert stat.S_IMODE(os.stat(planted).st_mode) == mode and list(planted.iterdir()) == []   # nothing chmod-ed, nothing written there


@pytest.mark.parametrize("card", CARDS)
def test_a_symbolic_link_is_refused_by_name(tmp_path, card):
    rig = _rig(tmp_path)
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(mode=0o700)
    planted = rig[2] / ("model_opt_jit-uid%d" % os.getuid()); os.symlink(elsewhere, planted)
    r, got, tmp = _source(tmp_path, card, pre=rig)
    assert got["RC"] == "0" and got["ROOT"] not in (str(planted), str(elsewhere)) and got["ROOT"].startswith(str(tmp / "protenix_v1_jit-"))
    assert f"jit cache: {planted} refused (" in r.stderr and list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("card", CARDS)
def test_a_named_root_is_used_as_given_and_nothing_is_made_in_tmp(tmp_path, card):
    named = tmp_path / "mine" / "jit"
    r, got, tmp = _source(tmp_path, card, {"MODEL_OPT_JIT_ROOT": str(named)})
    assert got["RC"] == "0" and got["ROOT"] == str(named) and got["EXT"] == f"{named}/{KEY}/torch_extensions" and got["TRI"] == f"{named}/{KEY}/triton"
    assert list(tmp.iterdir()) == [] and "refused" not in r.stderr


@pytest.mark.parametrize("card", CARDS)
def test_the_callers_own_cache_directories_are_kept(tmp_path, card):
    r, got, tmp = _source(tmp_path, card, {"TORCH_EXTENSIONS_DIR": "/my/ext", "TRITON_CACHE_DIR": "/my/triton"})
    assert got["RC"] == "0" and got["EXT"] == "/my/ext" and got["TRI"] == "/my/triton"
    assert got["ROOT"] == str(tmp / ("model_opt_jit-uid%d" % os.getuid()))                   # the root is still named (other caches key under it)


@pytest.mark.skipif(os.geteuid() == 0, reason="uid 0 can make a directory anywhere: the refusal cannot be staged")
@pytest.mark.parametrize("card", CARDS)
def test_no_private_directory_possible_is_not_active(tmp_path, card):
    rig = _rig(tmp_path)
    planted = rig[2] / ("model_opt_jit-uid%d" % os.getuid()); planted.mkdir(); os.chmod(planted, 0o777)
    os.chmod(rig[2], 0o500)                                                                   # the temporary directory itself admits no new entry
    try:
        r, got, tmp = _source(tmp_path, card, pre=rig)
    finally:
        os.chmod(rig[2], 0o700)
    assert got["RC"] == "3" and got["EXT"] == "" and got["TRI"] == ""                     # refused before anything is exported
    assert "[protenix-v1-opt] NOT ACTIVE: no private cache directory can be made under" in r.stderr, r.stderr
