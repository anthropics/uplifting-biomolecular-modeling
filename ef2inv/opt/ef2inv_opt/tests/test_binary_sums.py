"""The cubin the design kit carries (k/ef2_t16/sm_90a/) is held to the SHA256SUMS beside it before the CUDA driver maps it — the shared core's
rule and helper (opt_core.gates.binary_refusal, carried byte-for-byte in k/ef2_t16_nvjit.py) at the kit's one load site,
ef2_t16_nvjit.shipped_cubin (-> Kernel -> cuModuleLoadData).

The tree: every compiled binary under opt/ is a ``<sha256>  <file>`` line of a SHA256SUMS in its own directory, digest equal — so nothing the
kit carries is refused, and the SHA256SUMS is always there (a binary outside the core package is looked up in its own directory only; the line's
presence beside the carried cubin is this file's first test).  The rule: listed and equal passes (recorded in opt_core.gates.BINARIES_HELD); a
zeroed digest, an unlisted file or altered bytes is a refusal by name with one stderr line before anything reaches the driver, and shipped_cubin
raises NvjitUnavailable exactly as it does for an absent cubin (the t16 lever then steps aside by name and K-D3's own kernels serve the
transition).  No device and no toolchain needed: only the binary's bytes are on trial here (the source key and the manifest record are
k/test_ef2_t16_transition's).
"""
import hashlib
import importlib.util
import json
import os
import shutil

import pytest

from opt_core import gates

from ._paths import KDIR, OPT

BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")
SHIPPED = os.path.join(KDIR, "ef2_t16", "sm_90a")
CUBIN = "ef2_transition_cute.cubin"
SUMS = "SHA256SUMS"


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _binaries(root):
    out = []
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(r, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


def _nvjit(kdir):
    """The loader module of a k/ directory (standard library only at import)."""
    spec = importlib.util.spec_from_file_location("ef2_t16_nvjit_under_test", os.path.join(kdir, "ef2_t16_nvjit.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def fresh(monkeypatch):
    """Empty verdict caches of the core's gate for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])
    return gates


def _carried_copy(tmp_path, monkeypatch):
    """A writable copy of the loader and its carried directory (k/ef2_t16_nvjit.py + k/ef2_t16/), loaded; the source key is answered from the
    manifest so that only the binary's bytes decide."""
    k = tmp_path / "k"
    k.mkdir()
    shutil.copy2(os.path.join(KDIR, "ef2_t16_nvjit.py"), str(k / "ef2_t16_nvjit.py"))
    shutil.copytree(os.path.join(KDIR, "ef2_t16"), str(k / "ef2_t16"))
    m = _nvjit(str(k))
    ent = m.shipped_manifest()["cubins"]["ef2_transition_cute"]
    monkeypatch.setattr(m, "source_key", lambda src, options: ent["source_key"])
    return m, ent, os.path.join(str(k), "ef2_t16", "sm_90a")


# ------------------------------------------------------------------------------------------------------------ the tree as carried

def test_every_carried_binary_is_a_sha256sums_line_with_its_digest(fresh):
    bins = _binaries(OPT)
    assert [os.path.relpath(b, OPT).replace(os.sep, "/") for b in bins] == ["forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_t16/sm_90a/" + CUBIN]
    for b in bins:
        sums = os.path.join(os.path.dirname(b), SUMS)
        assert os.path.isfile(sums), sums                                          # beside it: a binary outside the core package is looked up in its own directory only
        assert gates.read_sums(sums).get(os.path.basename(b)) == _sha(b), b        # listed, digest equal
        assert gates.binary_refusal(b) is None                                     # the core's reading of the same line
    assert gates.BINARIES_HELD == [(os.path.abspath(b), _sha(b)[:16]) for b in bins]
    with open(os.path.join(SHIPPED, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    assert gates.read_sums(os.path.join(SHIPPED, SUMS)) == {CUBIN: man["cubins"]["ef2_transition_cute"]["cubin_sha256"]}   # one line: the manifest's own digest of the one cubin


def test_the_carried_cubin_is_served_through_its_sha256sums_line(tmp_path, monkeypatch, capsys, fresh):
    m, ent, d = _carried_copy(tmp_path, monkeypatch)
    cub = m.shipped_cubin("", "ef2_transition_cute.cu", verbose=False)
    assert cub.source == "shipped" and cub.sha256 == ent["cubin_sha256"] == _sha(os.path.join(d, CUBIN))
    assert gates.BINARIES_HELD == [(os.path.join(d, CUBIN), ent["cubin_sha256"][:16])]
    assert SUMS not in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------------------- the refusals

@pytest.mark.parametrize("fault", ["zeroed", "unlisted", "altered"])
def test_a_cubin_off_its_sha256sums_line_is_refused_by_name_before_any_load(tmp_path, monkeypatch, capsys, fresh, fault):
    m, ent, d = _carried_copy(tmp_path, monkeypatch)
    sums, cubin = os.path.join(d, SUMS), os.path.join(d, CUBIN)
    good = _sha(cubin)
    if fault == "zeroed":
        with open(sums, "w", encoding="utf-8") as fh:
            fh.write("0" * 64 + "  " + CUBIN + "\n")
        reason = f"sha256 {good[:16]} != {sums} {'0' * 16}"
    elif fault == "unlisted":
        with open(sums, "w", encoding="utf-8") as fh:
            fh.write(good + "  another.cubin\n")
        reason = f"not listed in {sums}"
    else:                                                                          # the bytes moved under an unchanged line (and an unchanged manifest): the line refuses first
        with open(cubin, "ab") as fh:
            fh.write(b"\0")
        reason = f"sha256 {_sha(cubin)[:16]} != {sums} {good[:16]}"
    n_rec = len(m.records())
    with pytest.raises(m.NvjitUnavailable) as ei:
        m.shipped_cubin("", "ef2_transition_cute.cu", verbose=False)
    assert str(ei.value) == f"ef2_nvjit: ef2_transition_cute.cu: shipped cubin {cubin} refused: {reason} — refusing by name"
    assert [ln for ln in capsys.readouterr().err.splitlines() if SUMS in ln] == [f"[opt_core] {SUMS}: refused {cubin}: {reason}"]   # one line, the core's words
    assert gates.BINARIES_HELD == [] and len(m.records()) == n_rec                 # nothing recorded, nothing handed to the driver
