"""lnstream's shipped-binary contract: the prebuilt extension is used only when its manifest names THIS source digest, THIS device architecture
and the file's own sha256, and the file is its SHA256SUMS line; anything else is a word and the JIT build serves (opendde_opt/lnstream.py
find_prebuilt; the SHA256SUMS hold itself: test_binary_sums.py)."""
import json, os
from opendde_opt import lnstream


def _ship(root, key, digest, archs=("sm_80", "sm_90"), body=b"\x7fELFfake"):
    d = os.path.join(root, key); os.makedirs(d, exist_ok=True)
    so = os.path.join(d, lnstream.EXT_CS + ".so")
    with open(so, "wb") as fh: fh.write(body)
    man = {"source_sha16": digest, "archs": list(archs), "so_sha256": lnstream._file_sha256(so)}
    with open(os.path.join(d, lnstream.MANIFEST), "w") as fh: json.dump(man, fh)
    with open(os.path.join(d, lnstream.SUMS), "w") as fh: fh.write(f"{lnstream._file_sha256(so)}  {lnstream.EXT_CS}.so\n")   # its SHA256SUMS line, as shipped
    return so


def test_prebuilt_contract(tmp_path, monkeypatch):
    root = str(tmp_path); key = "torch2.7.1-cu126-cp311-cxx11abi1"
    monkeypatch.delenv(lnstream.PREBUILT_ENV, raising=False)
    assert lnstream.find_prebuilt("abcd", "90", root=root, key=key) == (None, f"prebuilt_missing:{key}")      # nothing shipped for this stack: JIT by name
    so = _ship(root, key, "abcd")
    assert lnstream.find_prebuilt("abcd", "90", root=root, key=key) == (so, "used")                             # digest + arch + sha256 agree: the binary
    assert lnstream.find_prebuilt("abcd", "80", root=root, key=key) == (so, "used")                             # the fat binary carries sm_80 too
    assert lnstream.find_prebuilt("beef", "90", root=root, key=key) == (None, "prebuilt_stale_source")          # the pinned source changed: rebuild, never load
    assert lnstream.find_prebuilt("abcd", "89", root=root, key=key) == (None, "prebuilt_arch_missing:sm_89")
    with open(so, "ab") as fh: fh.write(b"tampered")
    assert lnstream.find_prebuilt("abcd", "90", root=root, key=key) == (None, "prebuilt_digest_mismatch")
    monkeypatch.setenv(lnstream.PREBUILT_ENV, "0")
    assert lnstream.find_prebuilt("abcd", "90", root=root, key=key) == (None, "prebuilt_off")


def test_stack_key_names_the_abi():
    import pytest
    torch = pytest.importorskip("torch")
    k = lnstream.stack_key()
    assert k.startswith("torch" + torch.__version__.split("+")[0]) and "-cp" in k and "cxx11abi" in k


def test_shipped_manifests_are_self_consistent():
    """Every binary the tree ships carries a manifest whose sha256 is the file's (a truncated / re-saved binary is refused by name at load)."""
    root = lnstream.PREBUILT_DIR
    if not os.path.isdir(root):
        return
    for key in os.listdir(root):
        d = os.path.join(root, key); so = os.path.join(d, lnstream.EXT_CS + ".so"); man = os.path.join(d, lnstream.MANIFEST)
        assert os.path.isfile(so) and os.path.isfile(man), key
        m = json.load(open(man))
        assert m["so_sha256"] == lnstream._file_sha256(so) and m["archs"] and len(m["source_sha16"]) == 16, key
