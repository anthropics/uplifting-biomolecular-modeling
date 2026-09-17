"""kits/v1_25 class rule: a device is of a class BY COMPUTE CAPABILITY alone (_class_records.class_serves: sm equal); the row's memory_mib is
the part the row was measured on — a device of the same capability with another memory size (A100 40 GB vs the 80 GB row, H100 94 GB NVL vs the
80 GB row) is SERVED, the difference one clause on the apply line (memory_note), never a refusal. H100-80GB / A100-80GB: no clause (the line unchanged)."""
import json

from engines.flashzoi.kits.v1_25 import _class_records as R
from engines.flashzoi.kits.v1_25 import _wrap as W

H100_ROW = {"class": "h100_sxm", "sm": [9, 0], "memory_mib": 81559}
A100_REC = {"class": "a100_80gb", "sm": [8, 0], "memory_mib": 81920, "device_names": ["NVIDIA A100-SXM4-80GB"], "kit": "K"}


def test_class_serves_is_capability_only():
    assert R.class_serves(H100_ROW, "9.0", 81559) and R.class_serves(H100_ROW, (9, 0), 95830) and R.class_serves(H100_ROW, "9.0", None)
    assert not R.class_serves(H100_ROW, "8.0", 81559) and not R.class_serves(H100_ROW, None, 81559) and not R.class_serves({}, "9.0", 1)


def test_memory_note_words_the_difference_only_beyond_tolerance():
    assert R.memory_note(H100_ROW, 81559) is None and R.memory_note(H100_ROW, 81559 + R.MEMORY_TOL_MIB) is None and R.memory_note(H100_ROW, None) is None
    assert R.memory_note(H100_ROW, 95830) == "this device reports 95830 MiB, the class row h100_sxm was cut on a 81559 MiB part (memory is not a key: served by capability)"
    assert R.memory_note({"sm": [8, 0]}, 40536) is None                                   # a row without memory states nothing


def test_record_serves_a_same_capability_part_of_any_memory():
    assert R.record_serves(A100_REC, "K", "NVIDIA A100-SXM4-40GB", "8.0", 40536)          # the 40 GB part: same sm_80 kernels
    assert R.record_serves(A100_REC, "K", "NVIDIA A100 80GB PCIe", (8, 0), 81920)
    assert not R.record_serves(A100_REC, "K", "NVIDIA H100 80GB HBM3", "9.0", 81559)      # another capability: not this record's class
    assert not R.record_serves(A100_REC, "other-kit", "NVIDIA A100-SXM4-80GB", "8.0", 81920)   # another kit digest: never
    assert R.record_serves(A100_REC, "K", "NVIDIA A100-SXM4-80GB", None, None)            # no capability stated (no CUDA): by name


def test_assert_device_class_pinned_row_with_another_memory_size_is_served_with_a_note(monkeypatch):
    PINS = {"device_names": ["NVIDIA H100 80GB HBM3"], "device_classes": [H100_ROW], "fz_exact_sha256": "K"}
    dc = W.assert_device_class(PINS, "NVIDIA H100 80GB HBM3", "9.0", 81559)
    assert dc == {"device_name": "NVIDIA H100 80GB HBM3", "pinned": True, "record": None}    # the row's own part: no clause (the apply line unchanged)
    dc = W.assert_device_class(PINS, "NVIDIA H100 NVL", "9.0", 95830)
    assert dc["pinned"] is True and dc["memory_note"].startswith("this device reports 95830 MiB, the class row h100_sxm was cut on a 81559 MiB part")


def test_assert_device_class_record_of_the_capability_serves_the_40gb_part(monkeypatch, tmp_path):
    PINS = {"device_names": ["NVIDIA H100 80GB HBM3"], "device_classes": [H100_ROW], "fz_exact_sha256": "K"}
    rec = tmp_path / "a100_80gb.json"; rec.write_text(json.dumps(A100_REC))
    monkeypatch.setattr(W.glob, "glob", lambda pattern: [str(rec)])                       # the kit dir's class records = this one record (sm_80 for kit K)
    dc = W.assert_device_class(PINS, "NVIDIA A100-SXM4-40GB", "8.0", 40536)
    assert dc["record"] == str(rec) and dc["pinned"] is False
    assert dc["memory_note"] == "this device reports 40536 MiB, the class row a100_80gb was cut on a 81920 MiB part (memory is not a key: served by capability)"
    dc = W.assert_device_class(PINS, "NVIDIA A100-SXM4-80GB", "8.0", 81920)
    assert "memory_note" not in dc and dc["record"] == str(rec)
    dc = W.assert_device_class(PINS, "NVIDIA L40S", "8.9", 46068)                          # a capability no row / record carries: served UNPINNED — named on the line, never refused
    assert dc["pinned"] is False and dc["record"] is None and "NVIDIA L40S (cc 8.9, 46068 MiB)" in dc["unpinned"] and "bitwise vs stock unverified" in dc["unpinned"]


def test_class_row_words_the_measured_row_of_the_capability():
    rows = [{"class": "h100", "sm": [9, 0], "memory_mib": 81559}, {"class": "h200", "sm": [9, 0], "memory_mib": 143771}]
    assert R.class_row(rows, "9.0", 81559)["class"] == "h100" and R.class_row(rows, (9, 0), 143771)["class"] == "h200"   # two measured rows of one capability: the memory picks the WORD
    assert R.class_row(rows, "9.0", 95830)["class"] == "h100"                                # a memory no row was cut on: the capability's first row (served; memory_note words it)
    assert R.memory_note(R.class_row(rows, "9.0", 95830), 95830).startswith("this device reports 95830 MiB")
    assert R.class_row(rows, "8.0", 81920) is None and R.class_row([], "9.0", 1) is None and R.class_row(rows, None, None) is None
    from engines.flashzoi.kits import v1_25 as v25
    assert R.class_row(v25.PINS["device_classes"], (9, 0), 95830)["class"] == "h100"                 # the kit's pinned rows: an H100 NVL words as h100
