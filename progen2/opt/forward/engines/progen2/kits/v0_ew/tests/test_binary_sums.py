"""The shipped kernel library is held to SHA256SUMS before it is mapped (ext.sums_refusal, in ext.load; build.py writes the line).

The tree: every compiled binary the kit ships under opt/ is a ``<sha256>  <name>`` line of the SHA256SUMS beside it, digest equal —
so nothing the kit ships is refused.  The rule: listed and equal passes (recorded in ext.BINARIES_HELD and in the load record);
unlisted, altered, or no SHA256SUMS at all in the kit's directory is a refusal by name with one stderr line, and ext.load refuses the
library exactly as it refuses an absent one — nothing is mapped.  A library outside the kit's directory with no SHA256SUMS beside it
(a build.py --out DIR of one's own) is not the shipped binary and is not checked.
"""
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))          # opt/forward
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
pytest.importorskip("torch")                                                     # ext imports torch

from engines.progen2.kits.v0_ew import ext  # noqa: E402

KIT_DIR = os.path.dirname(HERE)
OPT_DIR = os.path.dirname(ROOT)                                                  # progen2/opt: every binary the kit ships is under it
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _shipped_binaries():
    out = []
    for root, dirs, files in os.walk(OPT_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


@pytest.fixture()
def fresh(monkeypatch):
    """Empty verdict caches for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(ext, "_SUMS_VERDICTS", {})
    monkeypatch.setattr(ext, "BINARIES_HELD", {})
    return ext


@pytest.fixture()
def fakekit(tmp_path, monkeypatch, fresh):
    """A stand-in kit directory: the shipped library is the one in ext.HERE."""
    kit = tmp_path / "v0_ew"
    kit.mkdir()
    monkeypatch.setattr(ext, "HERE", str(kit))
    return kit


# ----------------------------------------------------------------------------------------------------------- the tree as shipped

def test_every_shipped_binary_is_a_sha256sums_line_with_its_digest(fresh):
    bins = _shipped_binaries()
    assert os.path.basename(OPT_DIR) == "opt" and os.path.join(KIT_DIR, ext.BINARY) in bins
    for p in bins:
        sums = os.path.join(os.path.dirname(p), ext.SUMS)
        assert os.path.isfile(sums), p
        assert ext._read_sums(sums).get(os.path.basename(p)) == ext._sha256_file(p), p        # the line and the bytes agree
    assert ext.sums_refusal(os.path.join(KIT_DIR, ext.BINARY)) is None                        # the loader's reading of the rule: the shipped library passes
    assert ext.BINARIES_HELD == {os.path.join(KIT_DIR, ext.BINARY): ext._sha256_file(os.path.join(KIT_DIR, ext.BINARY))}


def test_the_sha256sums_beside_the_library_is_in_sha256sum_format():
    for n, line in enumerate(open(os.path.join(KIT_DIR, ext.SUMS), encoding="utf-8"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.rstrip("\n").split(None, 1)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), n
        assert "/" not in name and name == name.strip(), n


# ------------------------------------------------------------------------------------------------------------------- the rule

def test_listed_and_equal_passes_and_is_held(fakekit, capsys):
    so = fakekit / ext.BINARY; so.write_bytes(b"\x7fELF-x")
    (fakekit / "SHA256SUMS").write_text("%s  %s\n" % (_sha(b"\x7fELF-x"), ext.BINARY))
    assert ext.sums_refusal(str(so)) is None
    assert ext.BINARIES_HELD == {str(so): _sha(b"\x7fELF-x")}
    assert capsys.readouterr().err == ""


def test_altered_bytes_are_refused_by_name_once_on_stderr(fakekit, capsys):
    so = fakekit / ext.BINARY; so.write_bytes(b"altered")
    (fakekit / "SHA256SUMS").write_text("%s  %s\n" % (_sha(b"original"), ext.BINARY))
    why = ext.sums_refusal(str(so))
    assert why == "sha256 %s != SHA256SUMS %s" % (_sha(b"altered")[:16], _sha(b"original")[:16])
    assert ext.sums_refusal(str(so)) == why                              # one verdict per path per process: no second hash, no second line
    assert capsys.readouterr().err == "[kit v0_ew] SHA256SUMS: refused %s: %s\n" % (ext.BINARY, why)
    assert ext.BINARIES_HELD == {}


def test_a_zeroed_digest_refuses(fakekit, capsys):
    so = fakekit / ext.BINARY; so.write_bytes(b"bytes")
    (fakekit / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, ext.BINARY))
    assert ext.sums_refusal(str(so)) == "sha256 %s != SHA256SUMS %s" % (_sha(b"bytes")[:16], "0" * 16)
    assert "refused %s: sha256 " % ext.BINARY in capsys.readouterr().err


def test_unlisted_library_beside_a_sha256sums_is_refused(fakekit, capsys):
    so = fakekit / ext.BINARY; so.write_bytes(b"y")
    (fakekit / "SHA256SUMS").write_text("%s  other.so\n" % _sha(b"o"))
    assert ext.sums_refusal(str(so)) == "not listed in SHA256SUMS"
    assert "refused %s: not listed" % ext.BINARY in capsys.readouterr().err


def test_a_malformed_sha256sums_lists_nothing(fakekit):
    so = fakekit / ext.BINARY; so.write_bytes(b"y")
    (fakekit / "SHA256SUMS").write_text("not-a-digest  %s\n" % ext.BINARY)
    assert ext.sums_refusal(str(so)) == "not listed in SHA256SUMS"


def test_the_shipped_library_with_no_sha256sums_beside_it_is_refused(fakekit, capsys):
    so = fakekit / ext.BINARY; so.write_bytes(b"z")
    (fakekit.parent / "SHA256SUMS").write_text("%s  v0_ew/%s\n" % (_sha(b"z"), ext.BINARY))   # above the kit's directory: never consulted
    assert ext.sums_refusal(str(so)) == "no SHA256SUMS beside it"
    assert "refused %s: no SHA256SUMS beside it" % ext.BINARY in capsys.readouterr().err


def test_a_library_outside_the_kit_is_checked_only_when_its_directory_carries_sums(fakekit, tmp_path, capsys):
    out = tmp_path / "elsewhere"; out.mkdir()
    so = out / ext.BINARY; so.write_bytes(b"own build")
    assert ext.sums_refusal(str(so)) is None and ext.BINARIES_HELD == {}         # not the shipped binary: unchecked, not held
    assert capsys.readouterr().err == ""
    out2 = tmp_path / "another"; out2.mkdir()
    so2 = out2 / ext.BINARY; so2.write_bytes(b"own build 2")
    (out2 / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, ext.BINARY))
    assert ext.sums_refusal(str(so2)).startswith("sha256 ")
    assert "refused %s: sha256 " % so2 in capsys.readouterr().err               # another library is named by path


# --------------------------------------------------------------------------------------- the loader refuses exactly where absence refuses

def test_load_refuses_an_altered_library_by_name_and_maps_nothing(fakekit, monkeypatch):
    so = fakekit / ext.BINARY; so.write_bytes(b"not the shipped bytes")
    (fakekit / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, ext.BINARY))
    monkeypatch.setattr(ext, "_lib", None)
    monkeypatch.setattr(ext, "_record", None)
    monkeypatch.setattr(ext.ctypes, "CDLL", lambda *a, **k: pytest.fail("a refused library must not be mapped"))
    monkeypatch.setattr(ext, "cuda_major_of_torch", lambda: pytest.fail("the hold comes before the stack check"))
    with pytest.raises(ext.KitRefused) as ei:
        ext.load(path=str(so))
    assert str(ei.value) == "kit v0_ew: %s at %s refused (sha256 %s != SHA256SUMS %s) — %s" % (ext.BINARY, so, _sha(b"not the shipped bytes")[:16], "0" * 16, ext.REBUILD)
    assert ext._lib is None
    with pytest.raises(ext.KitRefused, match="library not loaded"):
        ext.record()
    so.unlink()                                                          # and an absent library refuses at the same place, by the same type
    with pytest.raises(ext.KitRefused, match="missing"):
        ext.load(path=str(so))


def test_load_of_a_listed_library_goes_on_to_the_stack_check_and_records_the_digest(fakekit, monkeypatch):
    so = fakekit / ext.BINARY; so.write_bytes(b"fake library bytes")
    (fakekit / "SHA256SUMS").write_text("%s  %s\n" % (_sha(b"fake library bytes"), ext.BINARY))
    monkeypatch.setattr(ext, "_lib", None)
    monkeypatch.setattr(ext, "_record", None)
    monkeypatch.setattr(ext, "cuda_major_of_torch", lambda: 12)
    (fakekit / "BUILD.json").write_text('{"torch": "2.4.1+cu118", "nvcc": ["Cuda compilation tools, release 11.8, V11.8.89"]}')
    with pytest.raises(ext.KitRefused, match="binds the CUDA 11 runtime"):      # held, then the stack check refuses these pins
        ext.load(path=str(so))
    assert ext.BINARIES_HELD == {str(so): _sha(b"fake library bytes")}

    class _Lib:                                                          # the same major: the hold passed, the bytes are mapped, the digest is in the record
        def __getattr__(self, name):
            f = lambda *a, **k: 1                                        # noqa: E731
            setattr(self, name, f)
            return f
    (fakekit / "BUILD.json").write_text('{"torch": "2.9.1+cu128", "nvcc": ["Cuda compilation tools, release 12.8, V12.8.93"]}')
    monkeypatch.setattr(ext, "_cudart_preload", lambda: None)
    monkeypatch.setattr(ext.ctypes, "CDLL", lambda *a, **k: _Lib())
    ext.load(path=str(so))
    assert ext.record()["sha256"] == _sha(b"fake library bytes") and ext.record()["path"] == str(so)


def test_build_writes_the_sha256sums_line_of_the_library_it_built(tmp_path, monkeypatch):
    from engines.progen2.kits.v0_ew import build as B

    class _Run:
        def __init__(self, cmd, **k):
            self.returncode, self.stderr = 0, ""
            self.stdout = "nvcc: fake\nCuda compilation tools, release 12.8, V12.8.93\nBuild fake"
            if "-o" in cmd:
                open(cmd[cmd.index("-o") + 1], "wb").write(b"built bytes")
    monkeypatch.setattr(B.subprocess, "run", _Run)
    monkeypatch.setattr(B.shutil, "which", lambda n: "/fake/nvcc")
    assert B.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "SHA256SUMS").read_text() == "%s  %s\n" % (_sha(b"built bytes"), B.BINARY)
    monkeypatch.setattr(ext, "_SUMS_VERDICTS", {})
    monkeypatch.setattr(ext, "BINARIES_HELD", {})
    assert ext.sums_refusal(str(tmp_path / B.BINARY)) is None               # what build.py wrote is what ext.load holds the library to
