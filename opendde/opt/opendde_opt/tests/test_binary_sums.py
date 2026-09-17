"""The kit's shipped binaries are held to SHA256SUMS before anything maps them (opendde_opt/lnstream.py sums_refusal → opt_core.gates.binary_refusal;
common/opt_core/tools/binary_sums.py --package opendde/opt).

The tree: every compiled binary the kit ships under opt/ (.so, .cubin, .cubin.xz, .ptx, .fatbin) is a `<sha256>  <file>` line of the SHA256SUMS
beside it, digest equal — so nothing the kit ships is refused.  The rule: listed and equal passes (recorded in opt_core.gates.BINARIES_HELD);
unlisted, altered (a zeroed digest), or no SHA256SUMS beside it is a refusal by name with one `[opt_core] SHA256SUMS: refused <path>: <reason>`
stderr line, find_prebuilt words it `prebuilt_unlisted` and the JIT build serves exactly as for an absent binary, and the load site itself
(load_prebuilt) raises before an extension loader is ever constructed — nothing is mapped.
"""
import hashlib
import importlib.machinery
import json
import os

import pytest

from opt_core import gates
from opendde_opt import lnstream

OPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # opendde/opt: the package and the kit bytes it wraps (forward/)
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")                                 # the core tool's reading of "compiled binary"
KEY = "torch2.7.1-cu126-cp311-cxx11abi1"


def _binaries(root):
    out = []
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        out += [os.path.join(r, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


@pytest.fixture()
def fresh(monkeypatch):
    """Empty verdict caches for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])
    monkeypatch.delenv(lnstream.PREBUILT_ENV, raising=False)
    return gates


def _ship(root, body=b"\x7fELFfake-binary", sums="equal"):
    """A stand-in shipped binary with its manifest; `sums`: equal | zeroed | other | absent — the SHA256SUMS beside it."""
    d = os.path.join(root, KEY); os.makedirs(d, exist_ok=True)
    so = os.path.join(d, lnstream.EXT_CS + ".so")
    with open(so, "wb") as fh: fh.write(body)
    digest = hashlib.sha256(body).hexdigest()
    with open(os.path.join(d, lnstream.MANIFEST), "w") as fh: json.dump({"source_sha16": "abcd", "archs": ["sm_90"], "so_sha256": digest}, fh)
    if sums != "absent":
        line = {"equal": f"{digest}  {lnstream.EXT_CS}.so\n", "zeroed": f"{'0' * 64}  {lnstream.EXT_CS}.so\n", "other": f"{digest}  some_other_binary.so\n"}[sums]
        with open(os.path.join(d, lnstream.SUMS), "w") as fh: fh.write(line)
    return so


# ----------------------------------------------------------------------------------------------------------- the tree as shipped

def test_every_shipped_binary_is_a_sha256sums_line_with_its_digest(fresh):
    bins = _binaries(OPT_DIR)
    assert len(bins) >= 1 and any(p.startswith(lnstream.PREBUILT_DIR + os.sep) for p in bins)      # the tree ships the LNSTREAM prebuilt
    for p in bins:
        rel = os.path.relpath(p, OPT_DIR)
        found = gates.binary_sums_for(p)
        assert found is not None and found[1] is not None, f"{rel}: not a line of a {lnstream.SUMS} beside it"
        sums_path, key = found
        assert os.path.dirname(sums_path) == os.path.dirname(p) and key == os.path.basename(p), rel     # beside it, keyed by its own name
        assert gates.read_sums(sums_path)[key] == lnstream._file_sha256(p), f"{rel}: sha256 differs from its {lnstream.SUMS} line"
        assert lnstream.sums_refusal(p) is None, rel                                                    # the loader's reading: nothing shipped is refused
    assert [d for _r, d in gates.BINARIES_HELD] == [lnstream._file_sha256(p)[:16] for p in bins]


def test_every_sha256sums_beside_a_binary_is_in_sha256sum_format():
    seen = 0
    for r, _dirs, files in os.walk(OPT_DIR):
        if lnstream.SUMS not in files:
            continue
        seen += 1
        for n, line in enumerate(open(os.path.join(r, lnstream.SUMS), encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (r, n)
            assert not rel.startswith("/") and ".." not in rel.split("/") and os.path.isfile(os.path.join(r, rel)), (r, n)
    assert seen >= 1


def test_shipped_manifest_and_sums_agree():
    """The lever's own manifest digest and the SHA256SUMS line are the same sha256 of the same file (one binary, two readers, one number)."""
    for key in os.listdir(lnstream.PREBUILT_DIR):
        d = os.path.join(lnstream.PREBUILT_DIR, key)
        m = json.load(open(os.path.join(d, lnstream.MANIFEST)))
        assert gates.read_sums(os.path.join(d, lnstream.SUMS)) == {lnstream.EXT_CS + ".so": m["so_sha256"]}, key


# ------------------------------------------------------------------------------------------------------------------- the rule

def test_listed_and_equal_is_used(tmp_path, fresh):
    so = _ship(str(tmp_path))
    assert lnstream.find_prebuilt("abcd", "90", root=str(tmp_path), key=KEY) == (so, "used")
    assert fresh.BINARIES_HELD and fresh.BINARIES_HELD[-1][1] == lnstream._file_sha256(so)[:16]


@pytest.mark.parametrize("sums,clause", [("zeroed", "sha256 "), ("other", "not listed in "), ("absent", "no SHA256SUMS beside it")])
def test_unlisted_altered_or_absent_is_refused_by_name_and_nothing_is_mapped(tmp_path, fresh, monkeypatch, capsys, sums, clause):
    so = _ship(str(tmp_path), sums=sums)
    assert lnstream.find_prebuilt("abcd", "90", root=str(tmp_path), key=KEY) == (None, "prebuilt_unlisted")   # the word: the JIT build serves, as for an absent binary
    err = capsys.readouterr().err
    assert f"[opt_core] SHA256SUMS: refused {so}: {clause}" in err, err                                   # the core's one by-name line
    made = []
    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", lambda *a, **k: made.append(a) or pytest.fail("a refused binary reached the extension loader"))
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})                                                     # a direct call at the load site re-derives the verdict
    with pytest.raises(lnstream.LnstreamRefused, match="^prebuilt_unlisted:"):
        lnstream.load_prebuilt(so)
    assert made == [] and lnstream.EXT_CS not in __import__("sys").modules                                 # nothing mapped, nothing bound
    assert fresh.BINARIES_HELD == []


def test_build_falls_back_to_the_jit_build_on_a_refused_binary(tmp_path, fresh, monkeypatch):
    """build(): a refused shipped binary is the word `prebuilt_unlisted` and the JIT build is what serves — the same path an absent binary takes."""
    so = _ship(str(tmp_path), sums="zeroed")
    monkeypatch.setattr(lnstream, "PREBUILT_DIR", str(tmp_path))
    monkeypatch.setattr(lnstream, "stack_key", lambda: KEY)
    monkeypatch.setattr(lnstream, "check_source", lambda kdir: None)
    monkeypatch.setattr(lnstream, "write_source", lambda kdir: (os.path.join(kdir, "x.cu"), {"streamless_2": 25, "streamless_3": 0}, True))
    monkeypatch.setattr(lnstream, "source_digest", lambda kdir, cu: "abcd")
    monkeypatch.setattr(lnstream, "load_prebuilt", lambda path: pytest.fail("a refused binary was loaded"))
    import sys, types
    built = types.SimpleNamespace(__file__="/jit/fast_layer_norm_cuda_v2_cs/fast_layer_norm_cuda_v2_cs.so")
    tec = types.ModuleType("opendde.model.layer_norm.torch_ext_compile"); tec.compile = lambda **kw: built
    pkg = types.ModuleType("opendde.model.layer_norm"); pkg.torch_ext_compile = tec; pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "opendde", types.ModuleType("opendde")); monkeypatch.setitem(sys.modules, "opendde.model", types.ModuleType("opendde.model"))
    monkeypatch.setitem(sys.modules, "opendde.model.layer_norm", pkg); monkeypatch.setitem(sys.modules, "opendde.model.layer_norm.torch_ext_compile", tec)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True); monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    lm = types.SimpleNamespace(__file__=str(tmp_path / "layer_norm.py"))
    ext = lnstream.build(loader_module=lm)
    assert ext is built and lnstream.STATS["prebuilt"] == "prebuilt_unlisted" and lnstream.STATS["build"] == "jit"
