"""gates.GPU_CLASSES carries an H200 row: `card=` words read certified on an H200 under a configuration that targets it — and EVERY
lookup a process could make before the row existed (targets H100 / A100 / A100_80GB / A100_40GB / an unknown word / no target, on every probed
card shape) returns byte-identical words and details. The pre-row table is frozen below verbatim; the check runs both tables side by side."""
import itertools

import pytest

from opt_core import gates

# GPU_CLASSES as it stood through 0.5.223.0 (before the H200 row) — verbatim. Any edit of an existing row fails test_existing_rows_verbatim first.
PRE_H200 = {
    "H100": {"name_contains": "H100", "memory_mib": 81559, "cc": "9.0"},
    "A100": {"name_contains": "A100", "memory_mib": 81920, "cc": "8.0", "memory_mib_members": (81920, 40960)},
    "A100_80GB": {"name_contains": "A100", "memory_mib": 81920, "cc": "8.0"},
    "A100_40GB": {"name_contains": "A100", "memory_mib": 40960, "cc": "8.0"},
}

# probed card shapes: nvidia-smi and torch readings of the classes' members and their neighbours (NVL / PCIe / 40 GB / H200 / L40S / L4), no card, no memory
PROBES = [
    {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"},
    {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81089, "probe": "torch"},
    {"name": "NVIDIA H100 NVL", "cc": "9.0", "sm": "sm90", "memory_mib": 95830, "probe": "nvidia-smi"},
    {"name": "NVIDIA H100 PCIe", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"},
    {"name": "NVIDIA H200", "cc": "9.0", "sm": "sm90", "memory_mib": 143771, "probe": "nvidia-smi"},
    {"name": "NVIDIA H200", "cc": "9.0", "sm": "sm90", "memory_mib": 143156, "probe": "torch"},
    {"name": "NVIDIA H200 NVL", "cc": "9.0", "sm": "sm90", "memory_mib": 143771, "probe": "nvidia-smi"},
    {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "sm": "sm80", "memory_mib": 81920, "probe": "nvidia-smi"},
    {"name": "NVIDIA A100 80GB PCIe", "cc": "8.0", "sm": "sm80", "memory_mib": 81920, "probe": "nvidia-smi"},
    {"name": "NVIDIA A100-SXM4-40GB", "cc": "8.0", "sm": "sm80", "memory_mib": 40960, "probe": "nvidia-smi"},
    {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "sm": "sm80", "memory_mib": 81251, "probe": "torch"},
    {"name": "NVIDIA L40S", "cc": "8.9", "sm": "sm89", "memory_mib": 46068, "probe": "nvidia-smi"},
    {"name": "NVIDIA L4", "cc": "8.9", "sm": "sm89", "memory_mib": 22731, "probe": "torch"},
    {"name": "NVIDIA B200", "cc": "10.0", "sm": "sm100", "memory_mib": 183359, "probe": "nvidia-smi"},
    {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": None, "probe": "nvidia-smi"},
    {"name": None, "cc": None, "sm": None, "memory_mib": None, "probe": "nvidia-smi unavailable (FileNotFoundError)"},
    None,
    {},
]
PRE_TARGETS = list(PRE_H200) + [None, "", "L40S", "B200"]     # every target word a pre-row process could name: the four rows, no target, unknown words (NOT H200: its answer is the new row's, below)


def _gate(g):
    return (g.name, g.ok, dict(g.details), tuple(g.words))


def test_existing_rows_verbatim():
    for k, row in PRE_H200.items():
        assert gates.GPU_CLASSES[k] == row, k
    assert set(gates.GPU_CLASSES) - set(PRE_H200) == {"H200"}
    assert list(gates.GPU_CLASSES)[:len(PRE_H200)] == list(PRE_H200)          # insertion order kept: the new row is appended
    assert gates.GPU_MEMORY_TOLERANCE_MIB == 1024


@pytest.mark.parametrize("probe,target", list(itertools.product(range(len(PROBES)), PRE_TARGETS)))
def test_every_pre_row_lookup_is_byte_identical(probe, target):
    gpu = PROBES[probe]
    before = gates.gpu_class_check(gpu, target, classes=PRE_H200)
    after = gates.gpu_class_check(gpu, target)                                  # the shipped table
    b, a = _gate(before), _gate(after)
    if target in ("L40S", "B200"):                                              # an unknown target lists the table's keys in details.known — the one place the new key shows; words identical
        assert b[2].pop("known") == sorted(PRE_H200) and a[2].pop("known") == sorted(gates.GPU_CLASSES)
    assert b == a, (gpu, target, b, a)
    if target in PRE_H200:
        assert gates.gpu_memory_member(gpu, target, classes=PRE_H200) == gates.gpu_memory_member(gpu, target)


def test_h200_row_certifies_the_h200_and_nothing_else():
    smi, torch_, nvl = PROBES[4], PROBES[5], PROBES[6]
    for gpu in (smi, torch_, nvl):                                              # nvidia-smi and torch readings; an `H200 NVL` name of the same memory is the class by the table's own rule (name token + memory)
        g = gates.gpu_class_check(gpu, "H200")
        assert g.ok and tuple(g.words) == () and g.details["match"] is True and g.details["memory_member"] == 143771, _gate(g)
        assert gates.gpu_memory_member(gpu, "H200") == 143771
    h100 = gates.gpu_class_check(PROBES[0], "H200")                              # an H100 under --config h200: named, not refused
    assert h100.ok and tuple(h100.words) == ("card=uncertified(NVIDIA_H100_80GB_HBM3,81559MiB)",) and h100.details["match"] is False
    on_h200 = gates.gpu_class_check(smi, "H100")                                # an H200 under --config h100: exactly the word it read before the row existed
    assert tuple(on_h200.words) == ("card=uncertified(NVIDIA_H200,143771MiB)",) == tuple(gates.gpu_class_check(smi, "H100", classes=PRE_H200).words)
    assert tuple(gates.gpu_class_check(smi, "H200", classes=PRE_H200).words) == ("target=unlisted(H200)",)   # what --config h200 read before this row
