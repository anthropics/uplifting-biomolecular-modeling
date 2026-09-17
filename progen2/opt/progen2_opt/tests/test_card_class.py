"""The house card gate (stack.card_class / stack.gpu_info): name substring AND memory.total within 1 % of one of the class's card memories.
The A100 class is the 80 GB and the 40 GB card; the memory compared is nvidia-smi's memory.total whenever nvidia-smi answers for the card
torch names (torch's total_memory omits the driver's reservation)."""
import sys
import types

import pytest

from progen2_opt import stack


@pytest.mark.parametrize("name,mib,cls", [
    ("NVIDIA H100 80GB HBM3", 81559, "h100"),
    ("NVIDIA H100 80GB HBM3", 80998, "h100"),            # torch's total_memory on that card: inside 1 %
    ("NVIDIA H200", 143771, "h200"),
    ("NVIDIA B200", 183359, "b200"),
    ("NVIDIA A100-SXM4-80GB", 81920, "a100"),
    ("NVIDIA A100 80GB PCIe", 81920, "a100"),
    ("NVIDIA A100-SXM4-80GB", 81153, "a100"),           # torch's total_memory on that card: inside 1 % (by 52 MiB)
    ("NVIDIA A100-SXM4-40GB", 40960, "a100"),
    ("NVIDIA A100-PCIE-40GB", 40960, "a100"),
    ("NVIDIA A100-SXM4-40GB", 24576, "wrong_card"),      # the name alone is not the class
    ("NVIDIA L40S", 46068, "wrong_card"),
    ("NVIDIA A10G", 23028, "wrong_card"),
    (None, None, "none"),
    ("NVIDIA A100-SXM4-80GB", None, "wrong_card"),
])
def test_card_class(name, mib, cls):
    assert stack.card_class(name, mib) == cls


def test_classes_carry_memory_tuples():
    for cls, (sub, mems) in stack.GPU_CLASSES.items():
        assert sub and isinstance(mems, tuple) and mems and all(isinstance(m, int) and m > 0 for m in mems), cls
    assert set(stack.GPU_CLASSES["a100"][1]) == {81920, 40960}


def _fake_torch(name, total_memory_bytes, major=8, minor=0):
    props = types.SimpleNamespace(name=name, total_memory=total_memory_bytes, major=major, minor=minor)
    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: props, device_count=lambda: 1)
    return types.SimpleNamespace(cuda=cuda)


def _fake_smi(monkeypatch, stdout, rc=0):
    def run(argv, **kw):
        assert argv[0] == "nvidia-smi"
        return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="")
    monkeypatch.setattr(stack.subprocess, "run", run)


def test_gpu_info_torch_loaded_reads_memory_total_from_nvidia_smi(monkeypatch):
    """torch loaded: name / cc / count from torch, memory = nvidia-smi memory.total of the card torch names (a 40 GB A100 is the a100 class)."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("NVIDIA A100-SXM4-40GB", 42298834944))   # torch: 40340 MiB; memory.total: 40960
    _fake_smi(monkeypatch, "NVIDIA A100-SXM4-40GB, 40960, 8.0\n")
    g = stack.gpu_info()
    assert (g["name"], g["memory_mib"], g["cc"], g["sm"], g["count"], g["source"]) == ("NVIDIA A100-SXM4-40GB", 40960, "8.0", "sm80", 1, "torch")
    assert stack.card_class(g["name"], g["memory_mib"]) == "a100"


def test_gpu_info_torch_loaded_without_nvidia_smi_keeps_torch_memory(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("NVIDIA A100-SXM4-80GB", 85094825984))   # 81153 MiB

    def boom(argv, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(stack.subprocess, "run", boom)
    g = stack.gpu_info()
    assert (g["name"], g["memory_mib"], g["source"]) == ("NVIDIA A100-SXM4-80GB", 81153, "torch")
    assert stack.card_class(g["name"], g["memory_mib"]) == "a100"


def test_gpu_info_torch_loaded_other_card_named_by_nvidia_smi_keeps_torch_memory(monkeypatch):
    """nvidia-smi answers for a card of another name (a masked multi-card host): torch's own figure stands."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("NVIDIA A100-SXM4-80GB", 85094825984))
    _fake_smi(monkeypatch, "NVIDIA H100 80GB HBM3, 81559, 9.0\n")
    g = stack.gpu_info()
    assert (g["name"], g["memory_mib"]) == ("NVIDIA A100-SXM4-80GB", 81153)


def test_gpu_info_without_torch_reads_nvidia_smi(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    _fake_smi(monkeypatch, "NVIDIA A100-SXM4-80GB, 81920, 8.0\nNVIDIA A100-SXM4-80GB, 81920, 8.0\n")
    g = stack.gpu_info()
    assert g == {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": "8.0", "sm": "sm80", "count": 2, "source": "nvidia-smi"}


def test_gpu_info_no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    _fake_smi(monkeypatch, "", rc=9)
    assert stack.gpu_info() == {"name": None, "memory_mib": None, "cc": None, "sm": None, "count": 0, "source": None}
    assert stack.card_class(None, None) == "none"
