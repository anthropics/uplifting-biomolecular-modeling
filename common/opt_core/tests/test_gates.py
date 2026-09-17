"""Gates: sums with total accounting, the GPU checks, the version gate, the core pin gate."""
import hashlib
import os
import shutil

import pytest

from opt_core import gates

# ----------------------------------------------------------------------------------------------------- sums


def test_read_and_verify_sums_total_accounting(tmp_path):
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "a").write_bytes(b"A")
    (tmp_path / "k" / "b").write_bytes(b"B")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"# comment\n{hashlib.sha256(b'A').hexdigest()}  k/a\n{hashlib.sha256(b'X').hexdigest()}  k/b\n{'0' * 64}  k/gone\n")
    d = gates.read_sums(str(sums))
    assert list(d) == ["k/a", "k/b", "k/gone"]
    g = gates.verify_sums(str(tmp_path), d)
    assert not g.ok and g.details["checked"] == 2 and g.details["missing"] == ["k/gone"] and list(g.details["mismatched"]) == ["k/b"]
    assert g.reason == f"1 missing, 1 mismatched of 3 listed under {tmp_path}"
    assert gates.verify_sums(str(tmp_path), d, subset=["k/a"]).ok
    assert list(gates.verify_sums(str(tmp_path), d, subset=["k/unlisted"]).details["mismatched"]) == ["k/unlisted"]
    with pytest.raises(ValueError):
        sums.write_text("notasha  k/a\n")
        gates.read_sums(str(sums))


def test_carry_gate_reads_opt_SHA256SUMS_only(tmp_path):
    opt = tmp_path / "opt"
    (opt / "kit").mkdir(parents=True)
    (opt / "kit" / "f").write_bytes(b"F")
    assert gates.carry_gate(str(opt)).reason == f"carry manifest missing: {opt / 'SHA256SUMS'}"
    (opt / "KIT_SHA256SUMS").write_text(f"{hashlib.sha256(b'F').hexdigest()}  kit/f\n")      # another spelling is not read
    assert not gates.carry_gate(str(opt)).ok
    (opt / "SHA256SUMS").write_text(f"{hashlib.sha256(b'F').hexdigest()}  kit/f\n")
    g = gates.carry_gate(str(opt))
    assert g.ok and g.details["checked"] == 1 and g.details["path"] == str(opt / "SHA256SUMS")


# ----------------------------------------------------------------------------------------------------- GPU checks

H100 = {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"}
NVL = dict(H100, name="NVIDIA H100 NVL", memory_mib=95830)
NONE = {"name": None, "cc": None, "sm": None, "memory_mib": None, "probe": "nvidia-smi unavailable (FileNotFoundError)"}


def _clean(g):
    """A certified card: ok, no reason, no words, the comparison matched."""
    return g.ok and g.reason is None and g.words == () and g.details["match"] is True


def _named(g, *words):
    """An uncertified card: ok (the activation proceeds), no reason, exactly these words, the comparison recorded as not matched."""
    return g.ok and g.reason is None and g.words == tuple(words) and g.details["match"] in (False, None)


def test_gpu_name_and_cc_checks():
    assert _clean(gates.gpu_name_check(H100, "H100")) and _clean(gates.gpu_name_check(NVL, "H100"))
    nothing = gates.gpu_name_check(H100, None)
    assert nothing.ok and nothing.words == () and nothing.details["match"] is None                    # no requirement: nothing compared
    assert _named(gates.gpu_name_check(H100, "A100"), "card=uncertified(NVIDIA_H100_80GB_HBM3)")
    assert _named(gates.gpu_name_check(NONE, "H100"), "card=unprobed(nvidia-smi_unavailable_(FileNotFoundError))")
    assert gates.gpu_name_check(NONE, None).words == ()
    assert _clean(gates.gpu_cc_check(H100, "9.0")) and gates.gpu_cc_check(H100, None).ok and gates.gpu_cc_check(H100, None).words == ()
    assert _named(gates.gpu_cc_check(H100, "8.0"), "card=unlisted-cc(9.0)")
    assert _named(gates.gpu_cc_check(NONE, "8.0"), "card=unlisted-cc(none)")


A100_SXM = {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "sm": "sm80", "memory_mib": 81920, "probe": "nvidia-smi"}
A100_PCIE = dict(A100_SXM, name="NVIDIA A100 80GB PCIe")
A100_40 = dict(A100_SXM, name="NVIDIA A100-SXM4-40GB", memory_mib=40960)


A100_40_PCIE = dict(A100_40, name="NVIDIA A100-PCIE-40GB", memory_mib=40536)
A30 = dict(A100_SXM, name="NVIDIA A30", memory_mib=24576)


def test_gpu_class_check_compares_name_and_memory_and_never_refuses():
    h = gates.gpu_class_check(H100, "H100")
    assert _clean(h) and h.details["memory_member"] == 81559
    nothing = gates.gpu_class_check(NVL, None)
    assert nothing.ok and nothing.words == () and nothing.details["match"] is None                    # no target: nothing compared
    assert _named(gates.gpu_class_check(NVL, "H100"), "card=uncertified(NVIDIA_H100_NVL,95830MiB)")       # the name's token, another memory size: not the certified card
    unl = gates.gpu_class_check(H100, "B200")                                                          # a class this table does not know
    assert _named(unl, "target=unlisted(B200)") and unl.details["known"] == ["A100", "A100_40GB", "A100_80GB", "H100", "H200"]
    assert _clean(gates.gpu_class_check(A100_SXM, "A100")) and _clean(gates.gpu_class_check(A100_PCIE, "A100"))   # both 80 GB A100 variants are the A100 class
    assert _named(gates.gpu_class_check(A100_SXM, "H100"), "card=uncertified(NVIDIA_A100-SXM4-80GB,81920MiB)")
    assert _named(gates.gpu_class_check(H100, "A100"), "card=uncertified(NVIDIA_H100_80GB_HBM3,81559MiB)")
    assert gates.GPU_CLASSES["H100"] == {"name_contains": "H100", "memory_mib": 81559, "cc": "9.0"} and gates.GPU_CLASSES["A100"]["cc"] == "8.0"
    assert _named(gates.gpu_class_check(dict(H100, memory_mib=None), "H100"), "card=uncertified(NVIDIA_H100_80GB_HBM3,memory_unknown)")
    assert _named(gates.gpu_class_check(NONE, "H100"), "card=unprobed(nvidia-smi_unavailable_(FileNotFoundError))")
    for g in (gates.gpu_class_check(NVL, "H100"), gates.gpu_class_check(H100, "B200"), gates.gpu_class_check(NONE, "H100"), gates.gpu_name_check(H100, "A100")):
        assert all(" " not in w and w.isascii() for w in g.words) and g.as_dict()["words"] == list(g.words)   # blank-free ASCII tokens, carried by as_dict


def test_a100_class_admits_the_80_and_40_gb_parts_by_capability_and_probed_memory():
    """The A100 class (cc 8.0) is ONE class with two memory members: the 80 GB part (SXM4 / PCIe, 81920 MiB) and the 40 GB part (40960 MiB)
    both match the class; A100_80GB / A100_40GB name one part each; a card of the name with another memory (an A30-like probe) and an
    unknown-memory probe are named uncertified, never refused; the H100 row is unchanged (one member = its memory_mib)."""
    for probe in (A100_SXM, A100_PCIE, A100_40, A100_40_PCIE):
        g = gates.gpu_class_check(probe, "A100")
        assert _clean(g), (g.words, g.details)
        assert g.details["memory_member"] == (81920 if probe["memory_mib"] > 60000 else 40960)          # the member matched = the row a memory-keyed choice takes
    assert _clean(gates.gpu_class_check(A100_40, "A100_40GB")) and _clean(gates.gpu_class_check(A100_SXM, "A100_80GB"))
    assert _named(gates.gpu_class_check(A100_40, "A100_80GB"), "card=uncertified(NVIDIA_A100-SXM4-40GB,40960MiB)")
    assert _named(gates.gpu_class_check(A100_SXM, "A100_40GB"), "card=uncertified(NVIDIA_A100-SXM4-80GB,81920MiB)")
    assert _named(gates.gpu_class_check(A30, "A100"), "card=uncertified(NVIDIA_A30,24576MiB)")
    assert _named(gates.gpu_class_check(dict(A100_40, memory_mib=60000), "A100"), "card=uncertified(NVIDIA_A100-SXM4-40GB,60000MiB)")
    assert _named(gates.gpu_class_check(dict(A100_40, memory_mib=None), "A100"), "card=uncertified(NVIDIA_A100-SXM4-40GB,memory_unknown)")
    assert gates.class_memory_members(gates.GPU_CLASSES["A100"]) == (81920, 40960) and gates.class_memory_members(gates.GPU_CLASSES["H100"]) == (81559,)
    assert {k: v["cc"] for k, v in gates.GPU_CLASSES.items()} == {"H100": "9.0", "A100": "8.0", "A100_80GB": "8.0", "A100_40GB": "8.0", "H200": "9.0"}


def test_gpu_memory_member_keys_a_memory_dependent_choice_off_the_probed_memory():
    """A kit's memory-dependent table is written against the class members (MiB); gpu_memory_member picks the row from the PROBED memory —
    the 40 GB row on a ~40 GB probe whatever the card's name says, the 80 GB row on an 80 GB probe, None off-class / off-member / no GPU."""
    rows = {81920: "long_crop", 40960: "short_crop"}
    assert rows[gates.gpu_memory_member(A100_40, "A100")] == "short_crop" and rows[gates.gpu_memory_member(A100_40_PCIE, "A100")] == "short_crop"
    assert rows[gates.gpu_memory_member(A100_SXM, "A100")] == "long_crop" and rows[gates.gpu_memory_member(A100_PCIE, "A100")] == "long_crop"
    assert gates.gpu_memory_member(dict(A100_SXM, name="NVIDIA A100X", memory_mib=40200), "A100") == 40960           # keyed by the probed memory, never by the name's digits
    assert gates.gpu_memory_member(H100, "A100") is None and gates.gpu_memory_member(A30, "A100") is None and gates.gpu_memory_member(NONE, "A100") is None
    assert gates.gpu_memory_member(dict(A100_40, memory_mib=60000), "A100") is None and gates.gpu_memory_member(A100_40, "B200") is None
    assert gates.gpu_memory_member(H100, "H100") == 81559 and gates.gpu_memory_member(A100_40, "A100", tolerance_mib=0) == 40960


def test_nvidia_smi_probe_without_the_binary(monkeypatch):
    monkeypatch.setenv("PATH", "")
    p = gates.nvidia_smi_probe()
    assert set(p) == set(gates.PROBE_KEYS) and p["name"] is None and p["probe"].startswith("nvidia-smi unavailable")
    assert gates.nvidia_smi_probe(keys=("name", "cc", "sm", "probe")).keys() == {"name", "cc", "sm", "probe"}


def test_nvidia_smi_probe_parses_a_fake_binary(tmp_path, monkeypatch):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'NVIDIA H100 80GB HBM3, 9.0, 81559'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert gates.nvidia_smi_probe() == H100


# ----------------------------------------------------------------------------------------------------- versions, gates, force


def test_version_gate_stock_refuses_framework_drift_is_a_word_and_force():
    pins = {"openfold": "1.0.1", "torch": "2.4.1", "triton": "3.0.0"}
    exact = gates.version_gate(pins, dict(pins), stock=("openfold",))                     # the certified box: every pin met
    assert exact.ok and exact.reason is None and exact.words == () and not any(d["drift"] for k, d in exact.details.items() if k != "stock")
    assert exact.details["stock"] == ["openfold"] and exact.details["torch"] == {"pinned": "2.4.1", "found": "2.4.1", "drift": False}
    legacy = gates.version_gate({"torch": "2.4.1", "triton": "3.0.0"}, {"torch": "2.4.1", "triton": "3.0.0"})
    assert legacy.ok and legacy.words == () and legacy.details["stock"] == []               # no stock= : every entry is framework/toolchain
    drift = gates.version_gate(pins, {"openfold": "1.0.1", "torch": "2.5.0", "triton": None}, stock=("openfold",))
    assert drift.ok and not drift.forced and drift.reason is None and drift.words == ("stack=drift(torch:2.5.0!=2.4.1,triton:None!=3.0.0)",)
    assert drift.details["torch"]["drift"] is True and drift.details["openfold"]["drift"] is False
    g = gates.version_gate(pins, {"openfold": "1.2.0", "torch": "2.4.1", "triton": None}, stock=["openfold"])   # the stock engine at another version: refused
    assert not g.ok and not g.forced and g.reason == "version mismatch — openfold: pinned 1.0.1, found 1.2.0" and g.words == ("stack=drift(triton:None!=3.0.0)",)
    absent = gates.version_gate(pins, {"torch": "2.4.1", "triton": "3.0.0"}, stock=("openfold",))               # the stock engine not installed: refused
    assert not absent.ok and absent.reason == "version mismatch — openfold: pinned 1.0.1, found None"
    f = gates.version_gate(pins, {"torch": "2.5.0"}, force=True, stock=("openfold",))
    assert not f.ok and f.forced and gates.first_refusal([f]) is None and gates.first_refusal([g, f]) == g.reason
    assert gates.first_refusal([drift, exact, legacy]) is None                                # words never make a refusal


def test_run_gates_records_a_raising_check():
    def boom():
        raise RuntimeError("x")
    out = gates.run_gates([lambda: gates.Gate("a", True), boom])
    assert [g.ok for g in out] == [True, False] and out[1].reason == "boom raised RuntimeError('x')"
    assert out[0].as_dict() == {"name": "a", "ok": True, "reason": None, "details": {}, "forced": False, "words": []}
    positional = gates.Gate("gpu", True, None, {"gpu": "x"})                                # the kits' positional form keeps working; words default to ()
    assert positional.words == () and positional.as_dict()["words"] == []


# ----------------------------------------------------------------------------------------------------- the core pin

PYPROJECT = """[build-system]
requires = ["setuptools>=64"]
build-backend = "_build_backend"
backend-path = ["."]

[project]
name = "acme_opt"
version = "0.0.1"

[tool.opt_core]
path = "../../common/opt_core"
version = "{version}"
"""


def test_core_pin_check_match_mismatch_forced(tmp_path):
    core = gates.imported_core()
    assert core["version"] and set(core) == {"version", "package_dir"} and os.path.isdir(core["package_dir"])
    pp = tmp_path / "engine" / "opt" / "pyproject.toml"
    pp.parent.mkdir(parents=True)
    pp.write_text(PYPROJECT.format(version=core["version"]))
    g = gates.core_pin_check(str(pp))
    assert g.ok and g.details["pinned_abs_path"] == str(tmp_path / "common" / "opt_core") and set(g.details["pinned"]) == {"path", "version"}
    pp.write_text(PYPROJECT.format(version="0.0.1"))                                  # the pin is a floor: an older pin passes
    assert gates.core_pin_check(str(pp)).ok
    pp.write_text(PYPROJECT.format(version="9.9.9"))
    g = gates.core_pin_check(str(pp))
    assert not g.ok and g.reason == f"opt_core pinned >= v9.9.9 at ../../common/opt_core, imported v{core['version']} from {core['package_dir']}"
    assert gates.core_pin_check(str(pp), force=True).forced
    assert gates.version_tuple("0.5.10.0") > gates.version_tuple("0.5.9.9") and gates.version_tuple("1") == (1,)
    pp.write_text("[project]\nname='x'\n")
    u = gates.core_pin_check(str(pp))                                                     # the pin cannot be read: the imported core serves unpinned, named
    assert u.ok and u.reason is None and u.words == ("core_pin=unreadable",) and "lacks path, version" in u.details["unreadable"] and u.details["imported"] == core
    gone = gates.core_pin_check(str(tmp_path / "nowhere" / "pyproject.toml"))
    assert gone.ok and gone.words == ("core_pin=unreadable",) and gone.details["pinned"] is None


def test_read_pin_table_fallback_reader_matches_tomllib(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text(PYPROJECT.format(version="0.1.0"))
    full = gates.read_pin_table(str(pp))
    assert full == {"path": "../../common/opt_core", "version": "0.1.0"}
    import builtins
    real_import = builtins.__import__

    def no_tomllib(name, *a, **k):
        if name == "tomllib":
            raise ModuleNotFoundError(name)
        return real_import(name, *a, **k)

    builtins.__import__ = no_tomllib
    try:
        assert gates.read_pin_table(str(pp)) == full
        commented = pp.read_text().replace("[tool.opt_core]", "  [tool.opt_core]   # the core pin (a trailing comment is legal TOML)")
        commented = commented.replace('path = "../../common/opt_core"', '  path   =   "../../common/opt_core"    # a value-line comment, spaces around')
        assert commented.count("#") >= 2
        pp.write_text(commented)
        assert gates.read_pin_table(str(pp)) == full                          # the fallback reader: header comment, value comment, whitespace as tomllib
    finally:
        builtins.__import__ = real_import
    assert gates.read_pin_table(str(pp)) == full                              # and tomllib itself


def test_dist_version_reads_metadata_without_import():
    assert gates.dist_version("pytest") is not None and gates.dist_version("no-such-distribution-xyz") is None


def test_gpu_name_check_folds_case_only_on_request():
    card = {"name": "NVIDIA h100 80GB HBM3", "cc": "9.0", "probe": "nvidia-smi"}
    strict = gates.gpu_name_check(card, "H100")                                            # the case-sensitive form (most kits): no match, named, never refused
    assert strict.ok and strict.details["match"] is False and strict.words == ("card=uncertified(NVIDIA_h100_80GB_HBM3)",) and strict.details["fold_case"] is False
    folded = gates.gpu_name_check(card, "H100", fold_case=True)                            # the folding form a kit asks for by name
    assert folded.ok and folded.details["match"] is True and folded.words == () and folded.details["fold_case"] is True
    assert _named(gates.gpu_name_check(card, "A100", fold_case=True), "card=uncertified(NVIDIA_h100_80GB_HBM3)")
    assert gates.gpu_name_check(card, None, fold_case=True).ok and gates.gpu_name_check(card, None, fold_case=True).words == ()