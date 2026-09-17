"""The two scratch-directory fallbacks of the triangle-attention families are private: the exact package's cache_dir fallback (taken only when its
_paths module is unavailable) makes its per-user directory 0700 and refuses by name one another account could have written, returning a fresh
private directory instead; the native package's maintainer build script defaults TORCH_EXTENSIONS_DIR to a fresh mkdtemp directory."""
import os
import stat
import tempfile

import pytest

P = pytest.importorskip("opt_core.kernels.triattn_exact._prebuilt")
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def fallback(monkeypatch, tmp_path):
    """cache_dir on its fallback branch (no _paths), the temporary directory being tmp_path, $TRIATTN_EXACT_CACHE unset."""
    monkeypatch.setattr(P, "_paths", None)
    monkeypatch.delenv("TRIATTN_EXACT_CACHE", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_the_per_user_fallback_directory_is_made_private(fallback):
    d = P.cache_dir("build")
    assert d == os.path.join(str(fallback), "triattn_exact-%d" % os.getuid(), "build")
    for p in (os.path.dirname(d), d):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o700


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_group_writable_fallback_directory_is_refused_by_name_for_a_fresh_private_one(fallback, capsys):
    planted = fallback / ("triattn_exact-%d" % os.getuid())
    planted.mkdir()
    os.chmod(str(planted), 0o777)                                          # what another account (or an earlier umask) could have left there
    d = P.cache_dir()
    assert d != str(planted) and os.path.isdir(d)
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    err = capsys.readouterr().err
    assert err.startswith("[opt_core] triattn_exact: REFUSED cache directory %s: is writable by group or other (mode 0777)" % planted)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_symbolic_link_in_place_of_the_fallback_directory_is_refused(fallback, capsys):
    target = fallback / "elsewhere"
    target.mkdir(mode=0o700)
    os.symlink(str(target), str(fallback / ("triattn_exact-%d" % os.getuid())))
    d = P.cache_dir()
    assert os.path.realpath(d) != os.path.realpath(str(target))
    assert "is a symbolic link or belongs to uid" in capsys.readouterr().err


def test_a_named_cache_root_is_used_as_given_when_private(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_paths", None)
    root = tmp_path / "named"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("TRIATTN_EXACT_CACHE", str(root))
    assert P.cache_dir("k") == os.path.join(str(root), "k")


def test_the_native_build_script_defaults_its_extension_build_root_to_a_fresh_private_directory():
    text = open(os.path.join(CORE_DIR, "opt_core", "kernels", "triattn", "triattn_native", "build_prebuilt.py")).read()
    assert '"/tmp/' not in text and "'/tmp/" not in text.split('"""', 2)[2]        # no fixed path under the shared temporary directory in the code
    assert 'os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="te_triattn_native-")' in text
