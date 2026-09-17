"""Shipped binaries are held to SHA256SUMS before any loader maps them (opt_core.gates.binary_refusal; tools/binary_sums.py).

The tree: every compiled binary under opt_core/ (.so, .cubin, .cubin.xz) is a line of the SHA256SUMS in its own directory or in the
nearest directory above it inside the package, digest equal — so nothing the package ships is refused.  The rule: listed and equal
passes (listed in gates.BINARIES_HELD); unlisted, altered, or no SHA256SUMS at all is a refusal by name with one stderr line, and the
loader steps aside exactly as it does for an absent binary.  A path outside the package with no SHA256SUMS beside it (a JIT build, a
developer's --prebuilt-root) is not a shipped binary and is not checked.  The loader families are exercised on their refusal branch.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import pytest

from opt_core import gates

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(CORE_DIR, "opt_core")
TOOL = os.path.join(CORE_DIR, "tools", "binary_sums.py")


def _tool():
    spec = importlib.util.spec_from_file_location("binary_sums_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def fresh(monkeypatch):
    """Empty verdict caches for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])
    return gates


@pytest.fixture()
def fakepkg(tmp_path, monkeypatch, fresh):
    """A stand-in package directory: binary_refusal bounds its upward SHA256SUMS search at gates._PACKAGE_DIR."""
    pkg = tmp_path / "pkg"
    (pkg / "a" / "b").mkdir(parents=True)
    monkeypatch.setattr(gates, "_PACKAGE_DIR", str(pkg))
    return pkg


# ----------------------------------------------------------------------------------------------------------- the tree as shipped

def test_every_shipped_binary_is_a_sha256sums_line_with_its_digest(fresh):
    tool = _tool()
    bins = tool.binaries(PKG)
    assert len(bins) > 0
    assert tool.check(PKG) == [], "\n".join(tool.check(PKG))            # the tool's reading of the rule
    refused = [(os.path.relpath(p, PKG), gates.binary_refusal(p)) for p in bins if gates.binary_refusal(p) is not None]
    assert refused == []                                                 # the loaders' reading of the rule: nothing shipped is refused
    assert len(gates.BINARIES_HELD) == len(bins)
    for p in bins:                                                       # and every one is covered by a SHA256SUMS at or above its own directory, inside the package
        sums_path, key = gates.binary_sums_for(p)
        assert sums_path.startswith(PKG + os.sep) and "/" + key == "/" + os.path.relpath(p, os.path.dirname(sums_path)).replace(os.sep, "/")


def test_every_sha256sums_beside_a_binary_is_in_sha256sum_format():
    tool = _tool()
    for s in tool.sums_files(PKG):
        for n, line in enumerate(open(s, encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (s, n)
            assert not rel.startswith("/") and ".." not in rel.split("/"), (s, n)


# ------------------------------------------------------------------------------------------------------------------- the rule

def test_listed_and_equal_passes_and_is_held(fakepkg, capsys):
    so = fakepkg / "a" / "b" / "x.so"; so.write_bytes(b"\x7fELF-x")
    (fakepkg / "a" / "b" / "SHA256SUMS").write_text("%s  x.so\n" % _sha(b"\x7fELF-x"))
    assert gates.binary_refusal(str(so)) is None
    assert gates.BINARIES_HELD == [("a/b/x.so", _sha(b"\x7fELF-x")[:16])]
    assert capsys.readouterr().err == ""


def test_the_nearest_sha256sums_above_the_binary_covers_it(fakepkg):
    cub = fakepkg / "a" / "b" / "k.cubin"; cub.write_bytes(b"cubin-bytes")
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  b/k.cubin\n" % _sha(b"cubin-bytes"))
    assert gates.binary_sums_for(str(cub)) == (str(fakepkg / "a" / "SHA256SUMS"), "b/k.cubin")
    assert gates.binary_refusal(str(cub)) is None


def test_a_sha256sums_that_does_not_list_the_binary_does_not_shadow_one_above_that_does(fakepkg):
    cub = fakepkg / "a" / "b" / "k.cubin"; cub.write_bytes(b"k")
    (fakepkg / "a" / "b" / "SHA256SUMS").write_text("%s  other.so\n" % _sha(b"o"))
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  b/k.cubin\n" % _sha(b"k"))
    assert gates.binary_sums_for(str(cub)) == (str(fakepkg / "a" / "SHA256SUMS"), "b/k.cubin")
    assert gates.binary_refusal(str(cub)) is None
    so = fakepkg / "a" / "b" / "u.so"; so.write_bytes(b"u")                # listed nowhere: named against the nearest SHA256SUMS
    assert gates.binary_sums_for(str(so)) == (str(fakepkg / "a" / "b" / "SHA256SUMS"), None)
    assert gates.binary_refusal(str(so)) == "not listed in a/b/SHA256SUMS"


def test_altered_bytes_are_refused_by_name_once_on_stderr(fakepkg, capsys):
    so = fakepkg / "a" / "x.so"; so.write_bytes(b"altered")
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  x.so\n" % _sha(b"original"))
    why = gates.binary_refusal(str(so))
    assert why == "sha256 %s != a/SHA256SUMS %s" % (_sha(b"altered")[:16], _sha(b"original")[:16])
    assert gates.binary_refusal(str(so)) == why                          # one verdict per path per process: no second hash, no second line
    err = capsys.readouterr().err
    assert err == "[opt_core] SHA256SUMS: refused a/x.so: %s\n" % why
    assert gates.BINARIES_HELD == []


def test_unlisted_binary_beside_a_sha256sums_is_refused(fakepkg, capsys):
    so = fakepkg / "a" / "y.so"; so.write_bytes(b"y")
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  x.so\n" % _sha(b"x"))
    assert gates.binary_refusal(str(so)) == "not listed in a/SHA256SUMS"
    assert "refused a/y.so: not listed" in capsys.readouterr().err


def test_a_binary_with_no_sha256sums_inside_the_package_is_refused(fakepkg, tmp_path):
    so = fakepkg / "a" / "b" / "z.so"; so.write_bytes(b"z")
    (tmp_path / "SHA256SUMS").write_text("%s  pkg/a/b/z.so\n" % _sha(b"z"))   # above the package directory: never consulted
    assert gates.binary_sums_for(str(so)) is None
    assert gates.binary_refusal(str(so)) == "no SHA256SUMS in or above its directory lists it"


def test_the_bytes_already_read_are_what_is_hashed(fakepkg):
    so = fakepkg / "a" / "x.so"; so.write_bytes(b"on-disk")
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  x.so\n" % _sha(b"on-disk"))
    assert gates.binary_refusal(str(so), data=b"in-hand") == "sha256 %s != a/SHA256SUMS %s" % (_sha(b"in-hand")[:16], _sha(b"on-disk")[:16])


def test_a_path_outside_the_package_is_checked_only_when_its_directory_carries_sums(fakepkg, tmp_path, capsys):
    out = tmp_path / "elsewhere"; out.mkdir()
    so = out / "jit.so"; so.write_bytes(b"jit")
    assert gates.binary_refusal(str(so)) is None and gates.BINARIES_HELD == []      # not a shipped binary: unchecked, not held
    so2 = out / "dev.so"; so2.write_bytes(b"dev")
    (out / "SHA256SUMS").write_text("%s  dev.so\n%s  jit2.so\n" % (_sha(b"dev"), "0" * 64))
    assert gates.binary_refusal(str(so2)) is None
    so3 = out / "jit2.so"; so3.write_bytes(b"jit2")
    assert gates.binary_refusal(str(so3)).startswith("sha256 ")
    assert "refused %s: sha256" % so3 in capsys.readouterr().err


def test_the_tool_writes_sums_only_where_no_sha256sums_above_lists_the_binary(tmp_path):
    tool = _tool()
    pkg = tmp_path / "pkg"; (pkg / "k" / "prebuilt" / "s1").mkdir(parents=True); (pkg / "p" / "v1" / "build").mkdir(parents=True)
    (pkg / "k" / "prebuilt" / "s1" / "e.so").write_bytes(b"e"); (pkg / "k" / "prebuilt" / "s1" / "manifest.json").write_text("{}")
    (pkg / "p" / "v1" / "build" / "u.cubin").write_bytes(b"u")
    (pkg / "p" / "v1" / "SHA256SUMS").write_text("%s  build/u.cubin\n%s  README.md\n" % (_sha(b"u"), "0" * 64))   # a payload root's own SHA256SUMS is left as it is
    assert [os.path.relpath(p, str(pkg)) for p, n in tool.write(str(pkg))] == ["k/prebuilt/s1/SHA256SUMS"]
    assert (pkg / "k" / "prebuilt" / "s1" / "SHA256SUMS").read_text() == "%s  e.so\n" % _sha(b"e")
    assert tool.check(str(pkg)) == []
    (pkg / "p" / "v1" / "build" / "u.cubin").write_bytes(b"u2")
    assert tool.check(str(pkg)) == ["p/v1/build/u.cubin: sha256 differs from p/v1/SHA256SUMS"]


# ------------------------------------------------------------------------------------- the loaders refuse exactly where absence refuses

def _refuse(monkeypatch, path, why="sha256 0000000000000000 != SHA256SUMS ffffffffffffffff"):
    """Seed the process verdict for one shipped path: what binary_refusal returns after a mismatch."""
    monkeypatch.setitem(gates._BINARY_VERDICTS, os.path.abspath(path), why)
    return why


def test_triattn_native_wrapper_holds_every_extension_of_a_stack(fresh, monkeypatch):
    from opt_core.kernels.triattn import triattn_native as W
    key = sorted(W.stacks_built())[0]
    rep = W.verify_manifests(key)                                        # as shipped: every extension of the stack passes and is listed as held
    held = {rel for rel, _ in gates.BINARIES_HELD}
    for ext in rep:
        assert os.path.relpath(os.path.join(W.payload_dir(), "prebuilt", key, ext + ".so"), PKG).replace(os.sep, "/") in held
    ext = sorted(rep)[0]
    why = _refuse(monkeypatch, os.path.join(W.payload_dir(), "prebuilt", key, ext + ".so"))
    with pytest.raises(W.Unavailable) as ei:
        W.verify_manifests(key)
    assert ei.value.kind == "digest:%s:SHA256SUMS" % ext and why in str(ei.value)


def test_esm_v61_wrapper_holds_the_payload_cubins(fresh, monkeypatch):
    from opt_core.kernels.trimul import esm_v61 as E
    E.verify_digests()
    cubins = sorted(rel for rel in E.DIGESTS if rel.endswith(".cubin"))
    assert cubins and {os.path.relpath(os.path.join(E.pkg_dir(""), rel), PKG) for rel in cubins} <= {rel for rel, _ in gates.BINARIES_HELD}
    _refuse(monkeypatch, os.path.join(E.pkg_dir(""), cubins[0]))
    with pytest.raises(E.Unavailable) as ei:
        E.verify_digests()
    assert str(ei.value.kind if hasattr(ei.value, "kind") else ei.value).startswith("digest:%s(" % cubins[0]) or ("digest:%s(" % cubins[0]) in str(ei.value)


def test_ln_ext_loader_refuses_before_its_manifest_digest(fakepkg):
    from opt_core.kernels.ln import ext_loader as L
    so = fakepkg / "a" / "fast_layernorm_ext.so"; so.write_bytes(b"so-bytes")
    m = {"_so": str(so), "_key": "torch9.9.9+cu130-cp311", "so_sha256": _sha(b"so-bytes")}
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  fast_layernorm_ext.so\n" % ("0" * 64))
    with pytest.raises(L.Unavailable) as ei:
        L._check_digest(m)
    assert "so_refused:torch9.9.9+cu130-cp311(sha256 " in str(ei.value)
    gates._BINARY_VERDICTS.clear(); gates._SUMS_READ.clear()
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  fast_layernorm_ext.so\n" % _sha(b"so-bytes"))
    L._check_digest(m)                                                   # listed and equal: on to the manifest's own digest, which agrees


def test_flash_transition_steps_past_a_refused_sealed_prebuilt_to_its_store(fresh, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels import transition as TR
    FT = TR.carried_module("flash_sm90a")
    src = os.path.join(PKG, "kernels", "transition", "flash_sm90a", "prebuilt")
    key = sorted(d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, "manifest.json")))[0]
    dst = tmp_path / "prebuilt" / key
    shutil.copytree(os.path.join(src, key), str(dst))
    monkeypatch.setattr(FT, "prebuilt_dir", lambda: str(dst))
    man = json.load(open(dst / "manifest.json"))
    assert TR._sealed_prebuilt_refusal(FT) is None                       # a copy outside the package with no SHA256SUMS beside it: not a shipped binary
    gates._BINARY_VERDICTS.clear(); gates._SUMS_READ.clear()
    (dst / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, man["so"]))
    why = TR._sealed_prebuilt_refusal(FT)
    assert why.startswith("prebuilt %s refused: sha256 " % man["so"])
    monkeypatch.setattr(FT, "load", lambda *a, **k: pytest.fail("the sealed loader must not be called for a refused prebuilt"))
    monkeypatch.setattr(TR, "_FLASH_EXT", {})
    # (a) an empty store: no binary for any interpreter, so the outcome names the sealed refusal on every stack
    monkeypatch.setattr(TR, "FLASH_STORE", "flash_prebuilt_none")
    with pytest.raises(RuntimeError) as ei:
        TR._flash_ext()
    assert str(ei.value) == "flash_transition: no prebuilt for %s (sealed loader: %s)" % (TR.flash_abi_key(), why)
    # (b) the store as shipped: an interpreter it holds a binary for (torch build + CPython ABI) is served from the store, past the refused
    #     sealed prebuilt; any other interpreter is refused by name with both reasons
    monkeypatch.undo()
    monkeypatch.setattr(FT, "prebuilt_dir", lambda: str(dst))
    monkeypatch.setattr(FT, "load", lambda *a, **k: pytest.fail("the sealed loader must not be called for a refused prebuilt"))
    monkeypatch.setattr(TR, "_FLASH_EXT", {})
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {}); monkeypatch.setattr(gates, "_SUMS_READ", {}); monkeypatch.setattr(gates, "BINARIES_HELD", [])
    ent = (TR.flash_store_manifest().get("binaries") or {}).get(TR.flash_abi_key())
    if ent is not None and ent.get("torch") == torch.__version__ and ent.get("cuda") == torch.version.cuda:
        try:
            mod = TR._flash_ext()
        except Exception as e:  # noqa: BLE001  the store binary could not be mapped in this process (no device libraries): still past the sealed refusal
            assert "no prebuilt for" not in str(e) and why not in str(e), str(e)
        else:
            assert hasattr(mod, "flash_transition")
            assert any(rel.endswith(ent["so"]) for rel, _ in gates.BINARIES_HELD)      # the store binary itself was held to its SHA256SUMS first
    else:
        with pytest.raises(RuntimeError) as ei:
            TR._flash_ext()
        msg = str(ei.value)
        assert msg.startswith("flash_transition:") and (why in msg or "store binary" in msg), msg


def test_cuda_sm90a_refuses_an_altered_prebuilt_by_name(fresh, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels.triattn import cuda_sm90a as C
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(C, "_EXT", None, raising=False)
    src = os.path.join(PKG, "kernels", "triattn", "cuda_sm90a", "prebuilt")
    akey = sorted(d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, "manifest.json")))[0]
    root = tmp_path / "root"
    dst = root / os.path.basename(C.prebuilt_dir(str(root)))              # the directory install() reads for this interpreter under a given root
    shutil.copytree(os.path.join(src, akey), str(dst))
    man = json.load(open(dst / "manifest.json"))
    (dst / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, man["so"]))
    with pytest.raises(C.Refused) as ei:
        C.install(prebuilt_root=str(root))
    assert "refused: sha256 " in str(ei.value)


def test_exactln_steps_aside_from_a_refused_bundle_as_from_an_absent_one(fresh, monkeypatch):
    from opt_core.kernels.ln import exactln as X
    idx = X._prebuilt_index()
    if not idx or not idx.get("bundles"):
        pytest.skip("no exactln bundle index in this tree")
    cc = sorted(idx["bundles"])[0][2:]
    b = idx["bundles"]["sm" + cc]
    expr = sorted(b["kernels"])[0]
    monkeypatch.setattr(X, "_cu", lambda: None)
    monkeypatch.setitem(X._STATE, "prebuilt_modules", {})
    why = _refuse(monkeypatch, os.path.join(X.PREBUILT_DIR, b["file"]))
    assert X._prebuilt_function(expr, cc) is None
    assert X._STATE["prebuilt_off"] == "bundle %s refused: %s" % (b["file"], why)


def test_triattn_xla_prechecks_report_a_refused_binary_as_they_report_a_missing_one(fresh, monkeypatch):
    K = pytest.importorskip("opt_core.kernels.triattn_xla._k2b")
    idx = K.cubin_index()
    if not idx:
        pytest.skip("no k2b cubins in this tree")
    akey = sorted(idx)[0]
    arch, key = akey.split("/", 1)
    why = _refuse(monkeypatch, os.path.join(K.PKG_DIR, idx[akey]["file"]))
    with pytest.raises(K.Refused) as ei:
        K._load_cubin(arch, key)
    assert "refused: %s; fallback:" % why in str(ei.value)
