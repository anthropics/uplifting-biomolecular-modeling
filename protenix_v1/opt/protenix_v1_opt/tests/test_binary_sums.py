"""The kit's shipped compiled binaries are held to SHA256SUMS before anything maps them (opt_core.gates.binary_refusal).

The kit ships one: the stream-correct fast-LayerNorm extension under lib/fastln_prebuilt/, a line of the SHA256SUMS beside it, digest
equal (and equal to its manifest's so_sha256) — so the .so the tree carries is never refused.  install_prebuilt_fastln holds the .so to
that line before its loader maps it: a zeroed line or an absent SHA256SUMS refuses the .so by name with one stderr line, nothing is
loaded, and the report reads installed=False exactly as for an absent .so (the caller then takes the source rebuild).  The GPU stack is
not needed: torch, the stock LayerNorm module and the rebuild module are stand-ins; only the checks before the loader run.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import sys
import types

import pytest

from opt_core import gates

from .conftest import KIT

LIB = os.path.join(KIT, "lib")
PREBUILT = os.path.join(LIB, "fastln_prebuilt")
SO = "fast_layer_norm_cuda_v2_stream.so"
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")          # tools/binary_sums.py's set (opt_core)


def _sha_file(p: str) -> str:
    return gates.sha256_file(p)


def _binaries(root: str):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        out += [os.path.join(d, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


@pytest.fixture()
def fresh(monkeypatch):
    """Empty verdict caches for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])
    return gates


# ----------------------------------------------------------------------------------------------------------- the tree as shipped

def test_every_shipped_binary_is_a_sha256sums_line_beside_it_with_its_digest(fresh):
    bins = _binaries(KIT)
    assert [os.path.relpath(p, KIT) for p in bins] == [os.path.join("lib", "fastln_prebuilt", SO)]
    for p in bins:
        sums = os.path.join(os.path.dirname(p), "SHA256SUMS")
        assert os.path.isfile(sums), sums
        listed = gates.read_sums(sums)
        assert listed.get(os.path.basename(p)) == _sha_file(p), os.path.relpath(p, KIT)
        assert gates.binary_sums_for(p) == (sums, os.path.basename(p))
        assert gates.binary_refusal(p) is None                            # the loader's reading of the rule: nothing shipped is refused
    assert [d for _, d in gates.BINARIES_HELD] == [_sha_file(p)[:16] for p in bins]
    man = json.load(open(os.path.join(PREBUILT, "manifest.json"), encoding="utf-8"))
    assert man["so"] == SO and man["so_sha256"] == gates.read_sums(os.path.join(PREBUILT, "SHA256SUMS"))[SO]   # the manifest's own digest agrees


def test_every_sha256sums_under_the_kit_is_in_sha256sum_format():
    found = [os.path.join(d, "SHA256SUMS") for d, _, files in os.walk(KIT) if "SHA256SUMS" in files]
    assert os.path.join(PREBUILT, "SHA256SUMS") in found
    for s in found:
        for n, line in enumerate(open(s, encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (s, n)
            assert not rel.startswith("/") and ".." not in rel.split("/"), (s, n)


# --------------------------------------------------------------------------- install_prebuilt_fastln refuses exactly where absence refuses

def _stage(tmp_path, monkeypatch, sums_text):
    """A copy of lib/fastln_prebuilt with the given SHA256SUMS (None = absent), fastln_prebuilt.py loaded against stand-ins for torch, the
    stock LayerNorm module and the rebuild module (their versions and source digests agree with the manifest, so only the .so decides), and
    an extension loader that fails the test if it is ever constructed.  Returns (module, prebuilt dir)."""
    dst = tmp_path / "fastln_prebuilt"
    shutil.copytree(PREBUILT, str(dst))
    if sums_text is None:
        os.remove(str(dst / "SHA256SUMS"))
    else:
        (dst / "SHA256SUMS").write_text(sums_text, encoding="utf-8")
    man = json.load(open(dst / "manifest.json", encoding="utf-8"))
    torch = types.ModuleType("torch"); torch.__version__ = man["torch"]; torch.version = types.SimpleNamespace(cuda=man["cuda"])
    stubs = {"torch": torch}
    for name in ("protenix", "protenix.model", "protenix.model.layer_norm", "protenix.model.layer_norm.layer_norm",
                 "infopt_graphs", "infopt_graphs.protenix", "infopt_graphs.protenix.fastln_stream"):
        stubs[name] = types.ModuleType(name)
    for name, mod in stubs.items():
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(stubs[parent], leaf, mod)
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("_fastln_prebuilt_under_test", os.path.join(LIB, "fastln_prebuilt.py"))
    fpb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fpb)
    monkeypatch.setattr(fpb, "_kernel_source_shas", lambda: man["source_sha256"])

    class _NeverLoaded:
        def __init__(self, *a, **k):
            pytest.fail("the extension loader must not be constructed for a refused .so")

    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", _NeverLoaded)
    return fpb, dst


def test_a_zeroed_sha256sums_line_refuses_the_so_by_name_and_nothing_is_loaded(fresh, tmp_path, monkeypatch, capsys):
    fpb, dst = _stage(tmp_path, monkeypatch, "%s  %s\n" % ("0" * 64, SO))
    rep = fpb.install_prebuilt_fastln(str(dst))
    assert rep["installed"] is False
    assert rep["checks"]["so_exists"] is True and rep["checks"]["so_sha256sums"] is False and rep["checks"]["so_sha256"] is True
    assert rep["reason"].startswith("prebuilt checks failed: ")             # the words of an absent .so: the caller takes the source rebuild
    so = str(dst / SO)
    why = "sha256 %s != %s %s" % (_sha_file(so)[:16], str(dst / "SHA256SUMS"), "0" * 16)
    assert capsys.readouterr().err == "[opt_core] SHA256SUMS: refused %s: %s\n" % (so, why)
    assert gates.BINARIES_HELD == []


def test_an_unlisted_so_is_refused_by_name(fresh, tmp_path, monkeypatch, capsys):
    fpb, dst = _stage(tmp_path, monkeypatch, "%s  other.so\n" % hashlib.sha256(b"other").hexdigest())
    rep = fpb.install_prebuilt_fastln(str(dst))
    assert rep["installed"] is False and rep["checks"]["so_sha256sums"] is False
    assert capsys.readouterr().err == "[opt_core] SHA256SUMS: refused %s: not listed in %s\n" % (str(dst / SO), str(dst / "SHA256SUMS"))


def test_an_absent_sha256sums_refuses_the_so_as_an_absent_so_is_refused(fresh, tmp_path, monkeypatch, capsys):
    fpb, dst = _stage(tmp_path, monkeypatch, None)
    rep = fpb.install_prebuilt_fastln(str(dst))
    assert rep["installed"] is False and rep["checks"]["so_sha256sums"] is False
    assert capsys.readouterr().err == "[opt_core] SHA256SUMS: refused %s: no SHA256SUMS in its directory\n" % str(dst / SO)
    assert gates.BINARIES_HELD == []


def test_the_shipped_line_passes_every_check_and_reaches_the_loader(fresh, tmp_path, monkeypatch, capsys):
    fpb, dst = _stage(tmp_path, monkeypatch, open(os.path.join(PREBUILT, "SHA256SUMS"), encoding="utf-8").read())

    class _Reached(Exception):
        pass

    class _Loader:
        def __init__(self, *a, **k):
            raise _Reached()

    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", _Loader)
    rep = fpb.install_prebuilt_fastln(str(dst))
    assert rep["checks"] == {"so_exists": True, "so_sha256sums": True, "so_sha256": True, "torch_version": True, "cuda_version": True,
                             "kernel_sources_sha256": True}
    assert rep["installed"] is False and rep["reason"].startswith("load failed: _Reached")   # every check passed; only the (stand-in) loader stopped it
    assert capsys.readouterr().err == ""
    assert gates.BINARIES_HELD == [(str(dst / SO), _sha_file(str(dst / SO))[:16])]
