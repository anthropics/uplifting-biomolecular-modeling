"""The kit's shipped kernel binaries are held to SHA256SUMS lines before they are loaded (the shared core's rule,
``opt_core.gates.binary_refusal``): every compiled binary under the kit's own tree (``.so`` / ``.cubin`` / ``.xz`` / ``.ptx`` /
``.fatbin``; ``stock/`` is upstream's source, not loaded by the kit) is a ``<sha256>  <file>`` line of the SHA256SUMS in its
directory with that digest — none ships in this tree, and ``driver/prebuilt/sm_90a/SHA256SUMS`` lists none — and the kit's one
load path of its own cubins, ``ef2_nvjit.shipped_cubin``, refuses BY NAME a cubin whose SHA256SUMS is absent, does not list it, or
lists another digest, exactly as it refuses an absent cubin (the callers step aside to the statement the lever replaces)."""
import hashlib
import json
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(TESTS))                                    # esmfold2/opt
KIT_ROOT = os.path.dirname(OPT)                                                 # esmfold2/
DRIVER = os.path.join(OPT, "forward", "fast_inference", "driver")
SHIPPED = os.path.join(DRIVER, "prebuilt", "sm_90a")
SUMS = "SHA256SUMS"
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")
CORE_SRC = os.path.join(KIT_ROOT, "..", "common", "opt_core")                   # the release tree's core beside the kit


def _gates():
    try:
        import opt_core.gates as G  # noqa: PLC0415
        return G
    except ImportError:
        pass
    if os.path.isdir(os.path.join(CORE_SRC, "opt_core")):
        sys.path.insert(0, CORE_SRC)
        import opt_core.gates as G  # noqa: PLC0415
        return G
    pytest.skip("opt_core is neither installed nor beside the kit (installed copy without the release tree)")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _kit_files():
    for root, dirs, files in os.walk(KIT_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in ("stock", "__pycache__", ".git"))
        for f in sorted(files):
            yield os.path.join(root, f)


@pytest.fixture()
def gates(monkeypatch):
    G = _gates()
    monkeypatch.setattr(G, "_BINARY_VERDICTS", {})                             # one verdict per path per process: empty caches for the test
    monkeypatch.setattr(G, "_SUMS_READ", {})
    monkeypatch.setattr(G, "BINARIES_HELD", [])
    return G


@pytest.fixture(scope="module")
def J():
    sys.path.insert(0, DRIVER)
    try:
        import ef2_nvjit  # noqa: PLC0415
        yield ef2_nvjit
    finally:
        sys.modules.pop("ef2_nvjit", None)
        try:
            sys.path.remove(DRIVER)
        except ValueError:
            pass


# ----------------------------------------------------------------------------------------------------------- the tree as shipped

def test_every_kit_binary_is_a_sha256sums_line_with_its_digest(gates):
    bins = [p for p in _kit_files() if p.endswith(BINARY_SUFFIXES)]
    uncovered = []
    for p in bins:
        found = gates.binary_sums_for(p)
        if found is None or found[1] is None or gates.binary_refusal(p) is not None:
            uncovered.append(os.path.relpath(p, KIT_ROOT))
    assert uncovered == [], uncovered                                            # nothing the kit ships loads outside a SHA256SUMS line
    assert len(gates.BINARIES_HELD) == len(bins)
    assert os.path.isfile(os.path.join(SHIPPED, SUMS))                           # the shipped-cubin directory carries its SHA256SUMS even when it lists none:
    assert gates.read_sums(os.path.join(SHIPPED, SUMS)) == {}                    # a cubin placed there without a line is refused, not unchecked
    with open(os.path.join(SHIPPED, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    assert sorted(e["file"] for e in (man.get("cubins") or {}).values()) == sorted(gates.read_sums(os.path.join(SHIPPED, SUMS)))


def test_every_kit_sha256sums_is_in_sha256sum_format_and_names_present_files():
    for s in (p for p in _kit_files() if os.path.basename(p) == SUMS):
        base = os.path.dirname(s)
        for n, line in enumerate(open(s, encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (s, n)
            assert not rel.startswith("/") and ".." not in rel.split("/"), (s, n)
            p = os.path.join(base, *rel.split("/"))
            assert os.path.isfile(p), (s, n, rel)
            with open(p, "rb") as fh:
                assert _sha(fh.read()) == digest, (s, n, rel)


# ------------------------------------------------------------------------------------------------------- the loader's rule

SRC = "// a kernel source of the test\n__global__ void k() {}\n"


def _stage(J, root, data, sums=None):
    """A shipped-cubin directory under ``root``: x.cubin (``data``), a manifest entry the loader accepts, and SHA256SUMS = ``sums``
    ({file: digest}; None: no SHA256SUMS)."""
    d = os.path.join(root, J.ARCH)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "x.cubin"), "wb") as fh:
        fh.write(data)
    ent = dict(file="x.cubin", cubin_sha256=_sha(data), bytes=len(data), source_key=J.source_key(SRC, J.compile_options(J.ARCH, None, ())),
               nvrtc="13.0", stack_bytes=0, spill_stores=0, spill_loads=0, regs_ptxas=64, spill_free_required=True, diagnostics=[])
    with open(os.path.join(d, J.MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(dict(arch=J.ARCH, cubins=dict(x=ent)), fh)
    if sums is not None:
        with open(os.path.join(d, SUMS), "w", encoding="utf-8") as fh:
            fh.write("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
    return os.path.join(d, "x.cubin")


def test_a_listed_cubin_with_its_digest_loads(J, gates, tmp_path, monkeypatch):
    root = str(tmp_path / "listed"); data = b"\x7fELF-listed"
    _stage(J, root, data, sums={"x.cubin": _sha(data)})
    monkeypatch.setattr(J, "SHIPPED_ROOT", root)
    cub = J.shipped_cubin(SRC, "x.cu", verbose=False)
    assert cub.source == "shipped" and cub.sha256 == _sha(data) and cub.data == data
    assert gates.BINARIES_HELD == [(os.path.join(root, J.ARCH, "x.cubin"), _sha(data)[:16])]


def test_a_cubin_without_a_sha256sums_beside_it_is_refused_by_name(J, gates, tmp_path, monkeypatch):
    root = str(tmp_path / "nosums")
    _stage(J, root, b"\x7fELF-nosums", sums=None)
    monkeypatch.setattr(J, "SHIPPED_ROOT", root)
    with pytest.raises(J.NvjitUnavailable, match=r"refused: no SHA256SUMS in its directory .* — refusing by name"):
        J.shipped_cubin(SRC, "x.cu", verbose=False)


def test_a_cubin_its_sha256sums_does_not_list_is_refused_by_name(J, gates, tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "unlisted")
    _stage(J, root, b"\x7fELF-unlisted", sums={"y.cubin": _sha(b"y")})
    monkeypatch.setattr(J, "SHIPPED_ROOT", root)
    with pytest.raises(J.NvjitUnavailable, match=r"refused: not listed in .*SHA256SUMS — refusing by name"):
        J.shipped_cubin(SRC, "x.cu", verbose=False)
    assert "[opt_core] SHA256SUMS: refused" in capsys.readouterr().err


def test_an_altered_cubin_is_refused_by_name_before_the_manifest_digest(J, gates, tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "altered"); data = b"\x7fELF-altered"
    path = _stage(J, root, data, sums={"x.cubin": _sha(b"the audited bytes")})
    monkeypatch.setattr(J, "SHIPPED_ROOT", root)
    with pytest.raises(J.NvjitUnavailable, match=r"refused: sha256 %s != .*SHA256SUMS %s — refusing by name" % (_sha(data)[:16], _sha(b"the audited bytes")[:16])):
        J.shipped_cubin(SRC, "x.cu", verbose=False)
    err = capsys.readouterr().err
    assert err.count("[opt_core] SHA256SUMS: refused %s: sha256" % path) == 1
    assert gates.BINARIES_HELD == []


def test_without_the_shared_core_the_cubin_is_refused_not_loaded_unchecked(J, gates, tmp_path, monkeypatch):
    root = str(tmp_path / "nocore"); data = b"\x7fELF-nocore"
    _stage(J, root, data, sums={"x.cubin": _sha(data)})
    monkeypatch.setattr(J, "SHIPPED_ROOT", root)
    monkeypatch.setitem(sys.modules, "opt_core.gates", None)                    # `from opt_core.gates import …` raises ImportError
    with pytest.raises(J.NvjitUnavailable, match=r"SHA256SUMS check is not importable .* — refusing by name"):
        J.shipped_cubin(SRC, "x.cu", verbose=False)
