"""The kit's shipped kernel binaries are held to SHA256SUMS before any loader maps them (protenix_opt.binary_sums over
opt_core.gates.binary_refusal; the SHA256SUMS files are written by the core's tools/binary_sums.py --write --package <dir>).

Every compiled binary the kit ships under opt/ (today: the two fast-LayerNorm extensions of third_party/fastln_prebuilt*/ and the sm_90a
prologue library of third_party/protenix_fpf_triatt_procuda/prebuilt/) is a ``<sha256>  <name>`` line of the SHA256SUMS in its own
directory, digest equal; a binary whose line is zeroed, that is unlisted, or that has no SHA256SUMS beside it is refused by name on
stderr in the core's wording and nothing is loaded — the two load sites (fpf_stackgraph.ensure_stream_ln, ptx_trunk2_levers' prologue
lever) ask for the verdict before they reach the loader."""
import hashlib
import os
import re
import shutil

import pytest

from opt_core import gates
from protenix_opt import binary_sums, stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # opt/
KIT_HOME = os.path.abspath(stack.kit_home())                                                   # opt/forward/flashpairformer
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")                                  # == the core tool's
SHIPPED = (                                                                                     # the binaries the kit ships today, relative to opt/ (a new one must be added here AND get a SHA256SUMS line)
    "forward/flashpairformer/third_party/fastln_prebuilt/fast_layer_norm_cuda_v2_stream.so",
    "forward/flashpairformer/third_party/fastln_prebuilt_cu130/fast_layer_norm_cuda_v2_stream.so",
    "forward/flashpairformer/third_party/protenix_fpf_triatt_procuda/prebuilt/libtriatt_procuda_sm90a.so",
)
REFUSAL = re.compile(r"^\[opt_core\] SHA256SUMS: refused (?P<path>\S+): (?P<reason>.+)$", re.M)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _binaries():
    out = []
    for root, dirs, files in os.walk(OPT):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".venv") and not d.endswith(".egg-info"))
        out += [os.path.relpath(os.path.join(root, f), OPT).replace(os.sep, "/") for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


@pytest.fixture
def fresh(monkeypatch):
    """Verdict memos cleared (one verdict per path per process otherwise)."""
    monkeypatch.setattr(binary_sums, "_VERDICTS", {})
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])


@pytest.fixture
def copy(tmp_path):
    """A copy of the prologue library's prebuilt directory (binary + SHA256SUMS + manifest) to alter freely."""
    src = os.path.join(OPT, os.path.dirname(SHIPPED[2]))
    dst = tmp_path / "prebuilt"
    shutil.copytree(src, str(dst))
    return str(dst), os.path.basename(SHIPPED[2])


def test_the_kit_ships_exactly_the_known_binaries():
    assert tuple(_binaries()) == SHIPPED


def test_every_shipped_binary_is_a_line_of_the_sha256sums_beside_it_with_its_digest():
    problems = []
    for rel in SHIPPED:
        p = os.path.join(OPT, rel)
        sums = os.path.join(os.path.dirname(p), gates.BINARY_SUMS)
        if not os.path.isfile(sums):
            problems.append(f"{rel}: no {gates.BINARY_SUMS} beside it"); continue
        want = gates.read_sums(sums).get(os.path.basename(p))
        if want is None:
            problems.append(f"{rel}: not listed in {os.path.relpath(sums, OPT)}")
        elif want != _sha256(p):
            problems.append(f"{rel}: sha256 differs from {os.path.relpath(sums, OPT)}")
    assert not problems, problems


def test_every_sha256sums_beside_a_binary_is_in_sha256sum_format_and_lists_only_present_files():
    for rel in SHIPPED:
        d = os.path.dirname(os.path.join(OPT, rel))
        with open(os.path.join(d, gates.BINARY_SUMS), encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        assert lines, d
        for ln in lines:
            m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", ln)
            assert m, (d, ln)
            assert os.path.isfile(os.path.join(d, m.group(2))), (d, ln)


def test_the_sums_agree_with_the_loaders_own_manifests():
    """The SHA256SUMS line is the digest the package's manifest.json already pins (so_sha256): one fact, stated twice, equal."""
    import json
    for rel in SHIPPED:
        d = os.path.dirname(os.path.join(OPT, rel))
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["so"] == os.path.basename(rel)
        assert gates.read_sums(os.path.join(d, gates.BINARY_SUMS))[man["so"]] == man["so_sha256"]


def test_a_shipped_binary_passes_and_is_held(fresh, capsys):
    for rel in SHIPPED:
        p = os.path.join(OPT, rel)
        assert binary_sums.binary_refusal(p) is None
        assert gates.BINARIES_HELD[-1] == (p, _sha256(p)[:16])
    assert capsys.readouterr().err == ""


def test_a_zeroed_digest_is_refused_by_name_once_and_nothing_is_loaded(fresh, copy, capsys, monkeypatch):
    d, name = copy
    with open(os.path.join(d, gates.BINARY_SUMS), "w", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  " + name + "\n")
    import ctypes
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: loaded.append(a) or None)
    so = os.path.join(d, name)
    why = binary_sums.binary_refusal(so)
    assert why is not None and why.startswith("sha256 ") and "!=" in why
    assert binary_sums.binary_refusal(so) == why                        # one verdict per path per process, printed once
    err = capsys.readouterr().err
    hits = REFUSAL.findall(err)
    assert len(hits) == 1 and hits[0][0] == so and hits[0][1] == why, err
    assert loaded == [] and gates.BINARIES_HELD == []


def test_an_unlisted_binary_beside_a_sha256sums_is_refused(fresh, copy, capsys):
    d, name = copy
    other = os.path.join(d, "libother.so")
    shutil.copy(os.path.join(d, name), other)
    why = binary_sums.binary_refusal(other)
    assert why is not None and why.startswith("not listed in ")
    assert REFUSAL.search(capsys.readouterr().err)


def test_a_binary_with_no_sha256sums_beside_it_is_refused(fresh, copy, capsys):
    d, name = copy
    os.remove(os.path.join(d, gates.BINARY_SUMS))
    why = binary_sums.binary_refusal(os.path.join(d, name))
    assert why == "no SHA256SUMS in its directory"
    m = REFUSAL.search(capsys.readouterr().err)
    assert m and m.group("path") == os.path.join(d, name) and m.group("reason") == why
    assert gates.BINARIES_HELD == []


def test_manifest_binary_refusal_names_the_binary_the_loader_maps(fresh, copy, capsys):
    d, name = copy
    assert binary_sums.manifest_binary_refusal(d) is None                # intact copy: passes, held
    assert gates.BINARIES_HELD == [(os.path.join(d, name), _sha256(os.path.join(d, name))[:16])]
    with open(os.path.join(d, name), "ab") as fh:                       # altered bytes -> refused by name
        fh.write(b"\0")
    binary_sums._VERDICTS.clear(); gates._BINARY_VERDICTS.clear(); gates.BINARIES_HELD[:] = []
    why = binary_sums.manifest_binary_refusal(d)
    assert why is not None and why.startswith("sha256 ")
    assert gates.BINARIES_HELD == [] and REFUSAL.search(capsys.readouterr().err)
    os.remove(os.path.join(d, name))                                    # an absent binary / manifest is the loader's own by-name refusal: no verdict here
    assert binary_sums.manifest_binary_refusal(d) is None
    os.remove(os.path.join(d, "manifest.json"))
    assert binary_sums.manifest_binary_refusal(d) is None


def test_both_load_sites_ask_for_the_verdict_before_the_loader():
    """Source order at the two kit load sites: the verdict is taken before the sealed loader is reached."""
    sg = open(os.path.join(KIT_HOME, "src", "fpf_stackgraph", "stackgraph.py"), encoding="utf-8").read()
    a, b = sg.index("manifest_binary_refusal(pre)"), sg.index("FP.install_prebuilt_fastln(pre")
    assert 0 < a < b
    lv = open(os.path.join(KIT_HOME, "src", "ptx_trunk2_levers.py"), encoding="utf-8").read()
    a, b = lv.index("_binary_refusal(_so)"), lv.index("_PC.install(verbose=True)")
    assert 0 < a < b
