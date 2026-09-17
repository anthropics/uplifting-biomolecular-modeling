"""The kits' run.sh [install]-jitcache seed block: one text in every kit; a writable root or image cache is used in place (nothing
made under /tmp); the writable copy of a read-only root or image cache goes to ${TMPDIR:-/tmp}/model_opt_jit-uid<uid>, made with mode
0700, and a path there that another user owns, a symbolic link or a plain file is refused (nothing seeded, nothing written through
it); with no preset root and no image cache that same private per-user directory is the exported cache root (refused the same way);
MODEL_OPT_JIT_SEED_MAX_FILES is digits or a usage error. Runs on the kits present beside common/ (an image carries one)."""
import glob
import os
import shutil
import stat
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))            # <release root>: <kit>/run.sh beside common/
HEAD = "# [install]-jitcache"
TAIL = 'jit cache: $J ($W)"'


def _blocks():
    out = {}
    for run_sh in sorted(glob.glob(os.path.join(RELEASE, "*", "run.sh"))):
        lines = open(run_sh, encoding="utf-8").read().splitlines()
        heads = [i for i, l in enumerate(lines) if l.startswith(HEAD) and "seed the compile caches" in l]
        if not heads:
            continue
        i = heads[0]
        j = next(k for k in range(i, len(lines)) if TAIL in lines[k])
        if lines[j].startswith(" "):                                         # the export line sits inside the image `if`: the block ends at its `fi`
            j = next(k for k in range(j, len(lines)) if lines[k] == "fi")
        out[run_sh.split(os.sep)[-2]] = lines[i:j + 1]
    return out


BLOCKS = _blocks()
V43 = {k: v for k, v in BLOCKS.items() if v[0].startswith(HEAD + " v4.3")}
pytestmark = pytest.mark.skipif(not BLOCKS or not shutil.which("bash"), reason="no kit run.sh beside common/ (or no bash)")


def _body(kit, lines):
    """The block's lines less its comment line and its final jit-cache line, the kit's literal log tag read as the generic one."""
    return [l.replace("[%s-kit] jit cache:" % kit, "[${KIT:-kit}-kit] jit cache:") for l in lines[1:] if not l.startswith("#") and TAIL not in l]


def test_one_block_text_in_every_kit():
    assert V43, sorted(BLOCKS)
    ref_kit, ref = next(iter(V43.items()))
    for kit, lines in V43.items():
        assert _body(kit, lines) == _body(ref_kit, ref), kit
        text = "\n".join(lines)
        assert ("[%s-kit] jit cache: $T refused" % kit in text) or ('KIT=' in open(os.path.join(RELEASE, kit, "run.sh"), encoding="utf-8").read() and "[${KIT:-kit}-kit] jit cache: $T refused" in text), kit   # the refusal names the kit
        assert lines[-1] in ['[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[%s-kit] jit cache: $J ($W)"; }' % t for t in (kit, "${KIT:-kit}")], kit
    for kit in BLOCKS:                                                       # no run.sh names the shared /tmp path without the uid suffix
        text = open(os.path.join(RELEASE, kit, "run.sh"), encoding="utf-8").read()
        assert text.count("model_opt_jit") == text.count("model_opt_jit-uid$U") >= 1, kit


def _run(kit, tmp_path, env=None, pre=""):
    scratch = tmp_path / "tmp"; scratch.mkdir(exist_ok=True)
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "TMPDIR": str(scratch), "KIT": kit}
    base.update(env or {})
    script = "set -euo pipefail\n" + pre + "\n".join(BLOCKS[kit]) + '\nprintf "\\nRESULT root=%s how=%s\\n" "${MODEL_OPT_JIT_ROOT:-<unset>}" "$W"\n'
    p = subprocess.run(["bash", "-c", script], env=base, capture_output=True, text=True, timeout=60)
    res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    return p, (res[-1] if res else "")


def _image(tmp_path, writable):
    img = tmp_path / "image"
    (img / "key1" / "triton").mkdir(parents=True)
    (img / "key1" / "triton" / "a.cubin").write_bytes(b"\0\1")
    (img / "key1" / "triton" / "__grp__a.json").write_text('{"child_paths": {"cubin": "%s/key1/triton/a.cubin"}}' % img)
    if not writable:
        for d in (img / "key1" / "triton", img / "key1", img):
            d.chmod(0o555)
    return img


def _restore(*dirs):
    for d in dirs:
        for root, subdirs, _ in os.walk(d):
            os.chmod(root, 0o755)


@pytest.fixture
def kit():
    """A kit whose block is the image-seed-only variant (no version tag) when one is present, else any kit."""
    older = [k for k in BLOCKS if k not in V43]
    return older[0] if older else next(iter(BLOCKS))


@pytest.fixture
def kit43():
    if not V43:
        pytest.skip("no v4.3 block kit present")
    return next(iter(V43))


def _t(tmp_path):
    return tmp_path / "tmp" / ("model_opt_jit-uid%d" % os.getuid())


@pytest.mark.parametrize("which", ["kit", "kit43"])
def test_writable_image_cache_is_used_in_place(which, request, tmp_path):
    k = request.getfixturevalue(which)
    if os.getuid() == 0:
        pytest.skip("root writes everywhere: the read-only cases are exercised as a user")
    img = _image(tmp_path, writable=True)
    p, res = _run(k, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img)})
    assert p.returncode == 0 and res == "RESULT root=%s how=in-image" % img, p.stdout + p.stderr
    assert not _t(tmp_path).exists() and os.listdir(tmp_path / "tmp") == []          # nothing made under TMPDIR


@pytest.mark.parametrize("which", ["kit", "kit43"])
def test_read_only_image_cache_is_seeded_per_user(which, request, tmp_path):
    k = request.getfixturevalue(which)
    if os.getuid() == 0:
        pytest.skip("root writes everywhere: the read-only cases are exercised as a user")
    img = _image(tmp_path, writable=False)
    try:
        p, res = _run(k, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img)})
        t = _t(tmp_path)
        assert p.returncode == 0 and res == "RESULT root=%s how=seeded from image" % t, p.stdout + p.stderr
        assert stat.S_IMODE(os.lstat(t).st_mode) == 0o700 and not os.path.islink(t)
        assert (t / ".seeded").exists() and (t / "key1" / "triton" / "a.cubin").read_bytes() == b"\0\1"
        assert str(img) not in (t / "key1" / "triton" / "__grp__a.json").read_text() and str(t) in (t / "key1" / "triton" / "__grp__a.json").read_text()
        p, res2 = _run(k, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img)})            # a second run of this user: the copy is reused
        assert p.returncode == 0 and res2 == res and "refused" not in p.stderr, p.stdout + p.stderr
    finally:
        _restore(img)


def test_preset_writable_root_takes_the_image_seed_and_tmp_is_not_touched(kit43, tmp_path):
    if os.getuid() == 0:
        pytest.skip("exercised as a user")
    img = _image(tmp_path, writable=False)
    root = tmp_path / "myjit"
    try:
        p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img), "MODEL_OPT_JIT_ROOT": str(root)})
        assert p.returncode == 0 and res == "RESULT root=%s how=seeded from image" % root, p.stdout + p.stderr
        assert (root / ".seeded").exists() and not _t(tmp_path).exists()
        p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img), "MODEL_OPT_JIT_ROOT": str(root)})
        assert p.returncode == 0 and res == "RESULT root=%s how=user" % root, p.stdout + p.stderr
    finally:
        _restore(img)


def _ro_root(tmp_path, nfiles=3):
    r = tmp_path / "shared_root"
    (r / "keyA" / "triton").mkdir(parents=True)
    for i in range(nfiles):
        (r / "keyA" / "triton" / ("f%d.bin" % i)).write_bytes(b"x")
    for d in (r / "keyA" / "triton", r / "keyA", r):
        d.chmod(0o555)
    return r


def test_read_only_preset_root_seeds_its_key_dir_per_user(kit43, tmp_path):
    if os.getuid() == 0:
        pytest.skip("exercised as a user")
    r = _ro_root(tmp_path)
    try:
        p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_ROOT": str(r), "MODEL_OPT_STACK_KEY": "keyA", "MODEL_OPT_JIT_IMAGE": str(tmp_path / "none")})
        t = _t(tmp_path)
        assert p.returncode == 0 and res == "RESULT root=%s how=seeded from read-only root" % t, p.stdout + p.stderr
        assert stat.S_IMODE(os.lstat(t).st_mode) == 0o700 and sorted(os.listdir(t / "keyA" / "triton")) == ["f0.bin", "f1.bin", "f2.bin"]
        # over MODEL_OPT_JIT_SEED_MAX_FILES: nothing is seeded and the root stays where the caller put it
        shutil.rmtree(t)
        p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_ROOT": str(r), "MODEL_OPT_STACK_KEY": "keyA", "MODEL_OPT_JIT_SEED_MAX_FILES": "2", "MODEL_OPT_JIT_IMAGE": str(tmp_path / "none")})
        assert p.returncode == 0 and res == "RESULT root=%s how=" % r, p.stdout + p.stderr
        assert not (t / "keyA").exists()
    finally:
        _restore(r)


@pytest.mark.parametrize("plant", ["symlink", "file", "group-writable", "other-writable", "other-owner"])
def test_a_planted_scratch_path_is_refused(kit43, kit, tmp_path, plant):
    t = _t(tmp_path); t.parent.mkdir(exist_ok=True)
    bait = tmp_path / "planted_target"; bait.mkdir()
    if plant == "symlink":
        os.symlink(bait, t)
    elif plant == "file":
        t.write_text("")
    elif plant.endswith("-writable"):                                        # this user's own directory, but open: whatever is inside cannot be trusted
        t.mkdir(); os.chmod(t, 0o720 if plant == "group-writable" else 0o702)
    else:
        if os.getuid() != 0:
            pytest.skip("chown needs root")
        t.mkdir(mode=0o700); os.chown(t, 4242, 4242)
    img = _image(tmp_path, writable=False)
    r = _ro_root(tmp_path)
    # a preset root the process cannot write; root writes through any mode, so a shim fails that one `mkdir -p` the way a read-only mount does
    shim = 'mkdir() { [ "${1-}" != -p ] || [ "${2-}" != "$MODEL_OPT_JIT_ROOT" ] || return 1; command mkdir "$@"; }\n'
    try:
        env = {"MODEL_OPT_JIT_ROOT": str(r), "MODEL_OPT_STACK_KEY": "keyA", "MODEL_OPT_JIT_IMAGE": str(tmp_path / "none")}
        p, res = _run(kit43, tmp_path, env, pre=shim)
        assert p.returncode == 0 and res == "RESULT root=%s how=" % r and "refused" in p.stderr and str(t) in p.stderr, p.stdout + p.stderr
        if os.getuid() != 0:                                                 # image seed refused: no root is exported (as root the image cache is writable and used in place)
            for k in (kit, kit43):
                p, res = _run(k, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img)})
                assert p.returncode == 0 and res == "RESULT root=<unset> how=" and "refused" in p.stderr, p.stdout + p.stderr
        assert os.listdir(bait) == [] and (os.path.islink(t) or not os.path.isdir(t) or os.listdir(t) == [])   # nothing landed behind the planted path
    finally:
        _restore(img, r)


@pytest.mark.parametrize("value", ["12x", " 5", "-1", "a[$(touch pwned)]"])
def test_seed_max_files_is_digits_or_a_usage_error(kit43, tmp_path, value):
    p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_SEED_MAX_FILES": value, "MODEL_OPT_JIT_IMAGE": str(tmp_path / "none")})
    assert p.returncode == 2 and res == "" and "MODEL_OPT_JIT_SEED_MAX_FILES" in p.stderr, p.stdout + p.stderr
    assert not (tmp_path / "pwned").exists() and not os.path.exists("pwned")


def test_a_missing_tmpdir_is_created_as_before(kit43, tmp_path):
    if os.getuid() == 0:
        pytest.skip("exercised as a user")
    img = _image(tmp_path, writable=False)
    deep = tmp_path / "tmp" / "a" / "b"
    try:
        p, res = _run(kit43, tmp_path, {"MODEL_OPT_JIT_IMAGE": str(img), "TMPDIR": str(deep)})
        t = deep / ("model_opt_jit-uid%d" % os.getuid())
        assert p.returncode == 0 and res == "RESULT root=%s how=seeded from image" % t and (t / ".seeded").exists(), p.stdout + p.stderr
    finally:
        _restore(img)


NO_IMAGE = lambda tmp_path: {"MODEL_OPT_JIT_IMAGE": str(tmp_path / "none")}   # an image or venv built from this tree: no populated image cache


@pytest.mark.parametrize("which", ["kit", "kit43"])
def test_no_preset_and_no_image_cache_exports_the_private_per_user_root(which, request, tmp_path):
    """Nothing preset and nothing to seed: the per-user root is still made (0700, this user's, not a link) and exported, so no kit reaches a
    fixed shared default; nothing is printed for it, and a second run reuses it."""
    k = request.getfixturevalue(which)
    t = _t(tmp_path)
    p, res = _run(k, tmp_path, NO_IMAGE(tmp_path))
    assert p.returncode == 0 and res == "RESULT root=%s how=" % t and p.stderr == "", p.stdout + p.stderr
    assert t.is_dir() and not os.path.islink(t) and os.lstat(t).st_uid == os.getuid() and stat.S_IMODE(os.lstat(t).st_mode) == 0o700
    assert "jit cache" not in p.stdout                                        # exported without a printed line: a kit's own first stdout line stays first
    p, res2 = _run(k, tmp_path, NO_IMAGE(tmp_path))
    assert p.returncode == 0 and res2 == res and p.stderr == "" and "jit cache" not in p.stdout, p.stdout + p.stderr
    # the stack key's Triton directory follows the exported root when the kit's block derives it (the v4.3 kits) and no TRITON_CACHE_DIR is preset
    p, res3 = _run(k, tmp_path, dict(NO_IMAGE(tmp_path), MODEL_OPT_STACK_KEY="keyZ"), pre='unset TRITON_CACHE_DIR\n')
    assert p.returncode == 0 and res3 == res, p.stdout + p.stderr


@pytest.mark.parametrize("which", ["kit", "kit43"])
@pytest.mark.parametrize("plant", ["symlink", "file", "group-writable", "other-writable", "other-owner"])
def test_no_preset_and_no_image_cache_a_planted_root_is_refused_by_name(which, plant, request, tmp_path):
    k = request.getfixturevalue(which)
    t = _t(tmp_path); t.parent.mkdir(exist_ok=True)
    bait = tmp_path / "planted_target"; bait.mkdir()
    if plant == "symlink":
        os.symlink(bait, t)
    elif plant == "file":
        t.write_text("")
    elif plant.endswith("-writable"):
        t.mkdir(); os.chmod(t, 0o720 if plant == "group-writable" else 0o702)
    else:
        if os.getuid() != 0:
            pytest.skip("chown needs root")
        t.mkdir(mode=0o700); os.chown(t, 4242, 4242)
    p, res = _run(k, tmp_path, NO_IMAGE(tmp_path))
    assert p.returncode == 0 and res == "RESULT root=<unset> how=" and "%s refused" % t in p.stderr and "[%s-kit] jit cache:" % k in p.stderr, p.stdout + p.stderr
    assert os.listdir(bait) == [] and (os.path.islink(t) or not os.path.isdir(t) or os.listdir(t) == [])       # nothing landed behind the planted path


def test_a_valid_preset_root_is_used_as_given_and_tmp_is_not_touched(kit43, kit, tmp_path):
    root = tmp_path / "myjit"
    for k in (kit, kit43):
        p, res = _run(k, tmp_path, dict(NO_IMAGE(tmp_path), MODEL_OPT_JIT_ROOT=str(root)))
        assert p.returncode == 0 and res == "RESULT root=%s how=" % root and p.stderr == "", p.stdout + p.stderr   # the caller's root, no seed word, nothing refused
        assert not _t(tmp_path).exists()
