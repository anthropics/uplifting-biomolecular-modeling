"""Runtime-compiled kernel caches of the core — triattn_exact's shared libraries, exactln's NVRTC cubins, the esm_v61 carried module's cubin
directory — are loaded from as found, so the directory must be one no other account could have written: a planted / group- or other-writable
/ foreign-owned directory is refused by name on stderr (what, why, the fix) and the process compiles again in a private directory; the
default under the shared temporary directory is per user and 0700; a private directory of one's own is used silently."""
from __future__ import annotations

import os
import stat
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")
from opt_core.kernels.ln import exactln  # noqa: E402
from opt_core.kernels.triattn_exact import _paths  # noqa: E402
from opt_core.kernels.trimul import esm_v61  # noqa: E402

LOADERS = (_paths, exactln, esm_v61)


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    for m in LOADERS:
        monkeypatch.setattr(m, "_CACHE_DIRS", {})
    for k in ("TRIATTN_EXACT_CACHE", "MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    shared = tmp_path / "shared_tmp"
    shared.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(shared))
    old = os.umask(0o022)
    yield str(shared)
    os.umask(old)


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


# ---------------------------------------------------------------------------------------------- the rule itself (one copy per loader)
@pytest.mark.parametrize("mod", LOADERS)
@pytest.mark.parametrize("umask", (0o022, 0o002, 0o000))
def test_a_directory_of_ones_own_is_created_0700_and_used_silently(mod, tmp_path, capfd, umask):
    os.umask(umask)
    d = str(tmp_path / "jit" / "cache")
    assert mod._private_cache_dir(d, "t") == d and _mode(d) == 0o700
    with open(os.path.join(d, "k.so"), "wb") as f:                  # whatever modes the builder leaves inside a closed directory
        f.write(b"x")
    os.chmod(os.path.join(d, "k.so"), 0o775)
    mod._CACHE_DIRS.clear()
    assert mod._private_cache_dir(d, "t") == d and capfd.readouterr().err == ""


@pytest.mark.parametrize("mod", LOADERS)
@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_a_group_or_other_writable_directory_is_refused_by_name_and_replaced_by_a_private_one(mod, tmp_path, capfd, bit):
    d = str(tmp_path / "planted")
    os.makedirs(d)
    os.chmod(d, 0o755 | bit)
    out = mod._private_cache_dir(d, "t")
    err = capfd.readouterr().err
    assert out != d and _mode(out) == 0o700 and os.stat(out).st_uid == os.geteuid()
    assert err.count(f"REFUSED cache directory {d}: {d} is writable by group or other") == 1 and f"fix: chmod go-w {d}" in err and out in err
    assert mod._private_cache_dir(d, "t") == out and capfd.readouterr().err == ""      # said once per process
    os.chmod(d, 0o755); mod._CACHE_DIRS.clear()                                        # the printed fix
    assert mod._private_cache_dir(d, "t") == d and capfd.readouterr().err == ""


@pytest.mark.parametrize("mod", LOADERS)
def test_a_file_others_could_have_written_refuses_a_directory_they_can_enter(mod, tmp_path, capfd):
    d = str(tmp_path / "open")
    os.makedirs(d)
    os.chmod(d, 0o755)
    so = os.path.join(d, "k.so")
    with open(so, "wb") as f:
        f.write(b"x")
    os.chmod(so, 0o664)
    out = mod._private_cache_dir(d, "t")
    assert out != d and f"{so} is writable by group or other (mode 0664); fix: chmod go-w {so}" in capfd.readouterr().err
    os.chmod(so, 0o644); mod._CACHE_DIRS.clear()
    assert mod._private_cache_dir(d, "t") == d


@pytest.mark.parametrize("mod", LOADERS)
def test_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0_except_the_per_user_default(mod, tmp_path, monkeypatch, capfd):
    d = str(tmp_path / "theirs")
    os.makedirs(d, mode=0o755)
    owner = os.stat(d).st_uid
    if owner != 0:
        monkeypatch.setattr(os, "geteuid", lambda: owner + 1)       # this process is another (non-root) user
        assert mod._private_cache_dir(d, "t") != d
        assert f"{d} belongs to uid {owner}, not to this process (uid {owner + 1}) or root" in capfd.readouterr().err
        mod._CACHE_DIRS.clear()
    monkeypatch.setattr(os, "geteuid", lambda: 0)                   # a container's root reading a bind-mounted host directory
    assert mod._private_cache_dir(d, "t") == d
    mod._CACHE_DIRS.clear()
    if owner != 0:                                                  # the per-user default under the shared temporary directory: this uid alone
        assert mod._private_cache_dir(d, "t", own_only=True) != d and "REFUSED cache directory" in capfd.readouterr().err


@pytest.mark.parametrize("mod", LOADERS)
def test_a_named_directory_reached_through_a_link_is_judged_by_where_it_leads(mod, tmp_path, capfd):
    target, link = str(tmp_path / "scratch"), str(tmp_path / "cache")
    os.makedirs(target, mode=0o700)
    os.symlink(target, link)
    assert mod._private_cache_dir(link, "t") == link and capfd.readouterr().err == ""
    mod._CACHE_DIRS.clear()
    os.chmod(target, 0o777)
    assert mod._private_cache_dir(link, "t") != link and "is writable by group or other (mode 0777)" in capfd.readouterr().err
    mod._CACHE_DIRS.clear()
    os.chmod(target, 0o700)
    assert mod._private_cache_dir(link, "t", own_only=True) != link and "is a symbolic link" in capfd.readouterr().err


# ---------------------------------------------------------------------------------------------- triattn_exact (shared libraries: cuda_mma)
def test_triattn_exact_default_is_per_user_and_a_planted_default_is_never_used(fresh, capfd):
    want = os.path.join(fresh, f"triattn_exact_cache-uid{os.getuid()}")
    assert _paths.cache_root() == want and _paths.cache_dir() == want and _mode(want) == 0o700
    sub = _paths.cache_dir("cuda_mma")
    assert sub == os.path.join(want, "cuda_mma") and _mode(sub) == 0o700 and capfd.readouterr().err == ""
    _paths._CACHE_DIRS.clear()
    os.chmod(want, 0o777)                                           # as planted by another account before this user's first run
    with open(os.path.join(sub, "cuda_mma_x.so"), "wb") as f:
        f.write(b"planted")
    got = _paths.cache_dir("cuda_mma")
    err = capfd.readouterr().err
    assert not got.startswith(want) and _mode(got) == 0o700 and os.listdir(got) == []      # nothing found: the route compiles again, privately
    assert err.count("REFUSED cache directory " + want) == 1 and "fix:" in err


def test_triattn_exact_named_root_of_ones_own_is_used_silently_and_a_group_writable_one_is_not(tmp_path, monkeypatch, capfd):
    root = str(tmp_path / "mine")
    monkeypatch.setenv("TRIATTN_EXACT_CACHE", root)
    d = _paths.cache_dir("cuda_mma")
    assert d == os.path.join(root, "cuda_mma") and capfd.readouterr().err == ""
    _paths._CACHE_DIRS.clear()
    os.chmod(d, 0o770)
    assert _paths.cache_dir("cuda_mma") != d and "REFUSED cache directory " + d in capfd.readouterr().err


# ---------------------------------------------------------------------------------------------- exactln (NVRTC cubin + its name file)
def _exactln_stubs(monkeypatch, calls):
    monkeypatch.setitem(exactln._STATE, "kernels", {})
    monkeypatch.setitem(exactln._STATE, "cubin", {})
    monkeypatch.setattr(exactln, "_device_facts", lambda: ("90", 1))
    monkeypatch.setattr(exactln, "_prebuilt_function", lambda expr, cc: None)
    monkeypatch.setattr(exactln, "_opts", lambda cc: [b"-O3"])
    monkeypatch.setattr(exactln, "_source", lambda: b"src")
    monkeypatch.setattr(exactln, "_nvrtc_version", lambda: "13.0")
    monkeypatch.setattr(exactln, "_nvrtc", lambda expr, opts: (calls.append("nvrtc") or b"CUBIN:compiled", b"lowered"))
    monkeypatch.setattr(exactln, "_ensure_ctx", lambda: None)
    monkeypatch.setattr(exactln, "_check", lambda res, what: res)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: None)

    class _Cu:
        def cuModuleLoadData(self, image):
            calls.append(("load", bytes(image)))
            return ("module",)

        def cuModuleGetFunction(self, module, name):
            return ("func",)

    monkeypatch.setattr(exactln, "_cu", lambda: _Cu())


@pytest.mark.parametrize("umask", (0o022, 0o002))
def test_exactln_cubin_of_ones_own_cache_is_loaded_silently_whatever_the_umask(tmp_path, monkeypatch, capfd, umask):
    os.umask(umask)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    calls = []
    _exactln_stubs(monkeypatch, calls)
    exactln._compile("float32", "float32", "float32", 64, 3, 1)
    d = str(tmp_path / "jit" / "exactln")
    files = sorted(os.listdir(d))
    assert calls == ["nvrtc", ("load", b"CUBIN:compiled")] and len(files) == 2 and all(_mode(os.path.join(d, n)) == 0o644 for n in files)
    calls.clear(); exactln._CACHE_DIRS.clear(); exactln._STATE["kernels"].clear()
    exactln._compile("float32", "float32", "float32", 64, 3, 1)                 # the next process: read from disk, nothing compiled, nothing said
    assert calls == [("load", b"CUBIN:compiled")] and capfd.readouterr().err == ""


def test_exactln_planted_cubin_is_never_loaded_and_the_kernel_is_compiled_again_privately(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    calls = []
    _exactln_stubs(monkeypatch, calls)
    exactln._compile("float32", "float32", "float32", 64, 3, 1)
    d = str(tmp_path / "jit" / "exactln")
    for n in os.listdir(d):
        if n.endswith(".cubin"):
            with open(os.path.join(d, n), "wb") as f:
                f.write(b"CUBIN:planted")
    os.chmod(d, 0o777)                                              # a directory anyone could have written that cubin into
    calls.clear(); exactln._CACHE_DIRS.clear(); exactln._STATE["kernels"].clear()
    exactln._compile("float32", "float32", "float32", 64, 3, 1)
    err = capfd.readouterr().err
    assert calls == ["nvrtc", ("load", b"CUBIN:compiled")]
    assert err.count("REFUSED cache directory " + d) == 1 and f"fix: chmod go-w {d}" in err


# ---------------------------------------------------------------------------------------------- esm_v61 (the carried module's cubin directory)
def test_esm_v61_shared_default_becomes_per_user_and_a_planted_one_is_never_read(fresh, capfd):
    shared = os.path.join(fresh, "esmfold2_opt_nvrtc")
    want = f"{shared}-uid{os.getuid()}"
    assert esm_v61._jit_dir_private(lambda: shared) == want and _mode(want) == 0o700 and not os.path.exists(shared)
    assert capfd.readouterr().err == ""
    esm_v61._CACHE_DIRS.clear()
    os.chmod(want, 0o777)
    got = esm_v61._jit_dir_private(lambda: shared)
    assert got != want and _mode(got) == 0o700 and "REFUSED cache directory " + want in capfd.readouterr().err


def test_esm_v61_named_jit_root_of_ones_own_is_used_silently_and_an_uncreatable_one_is_left_as_it_is(tmp_path, capfd):
    d = str(tmp_path / "jit" / "key" / "nvrtc")
    assert esm_v61._jit_dir_private(lambda: d) == d and _mode(d) == 0o700 and capfd.readouterr().err == ""
    ro = str(tmp_path / "ro")
    os.makedirs(ro)
    os.chmod(ro, 0o555)
    absent = os.path.join(ro, "key", "nvrtc")
    if os.geteuid() != 0:
        assert esm_v61._jit_dir_private(lambda: absent) == absent and not os.path.exists(absent)
    os.chmod(ro, 0o755)


def test_esm_v61_binds_the_checked_directory_into_the_carried_module():
    src = open(esm_v61.__file__, encoding="utf-8").read()
    assert src.count("v6m._jit_dir = lambda _carried=v6m._jit_dir: _jit_dir_private(_carried)") == 1
