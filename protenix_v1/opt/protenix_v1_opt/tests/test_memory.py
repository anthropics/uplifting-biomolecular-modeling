"""The memory mode (`big`) around its adapter (big.py; the composition, the levers and the census are tests/test_big.py's): the MEMORY
line, the process's peak memory record, the adapter's attach name and report, one determinism row per mode."""
import types

import pytest

from protenix_v1_opt import big, modes, report as R, stack


def test_the_memory_line():
    rep = {"mode": "big", "memory": {"applied": {"infer_setting.chunk_size": 128, "infer_setting.sample_diffusion_chunk_size": 1}, "replaced": {"infer_setting.chunk_size": 256, "infer_setting.sample_diffusion_chunk_size": 5}}}
    line = R.memory_line(rep)
    assert line == "[protenix-v1-opt] MEMORY mode=big infer_setting.chunk_size=128(was 256) infer_setting.sample_diffusion_chunk_size=1(was 5)"
    assert line.startswith(R.PREFIX + " MEMORY ")


def test_gpu_peak_is_empty_without_torch_or_cuda(monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert stack.gpu_peak() == {}
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, max_memory_allocated=lambda d: 3 * 2**20 + 5, max_memory_reserved=lambda d: 7 * 2**20))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert stack.gpu_peak() == {"peak_allocated_mib": 3, "peak_reserved_mib": 7}


def test_the_adapter_is_the_one_install_and_names_its_attach(monkeypatch):
    """The memory install lives in the engine adapter (big.apply/big.report) with no second copy in stack.py, and one attach name per line."""
    assert not hasattr(stack, "apply_memory_preset")
    assert big.attach_name() == f"{stack.RUNNER_MODULE}.{stack.RUNNER_CLASS}.__init__"
    assert big.report({"mode": "big", "memory": {"applied": {"a": 1}, "replaced": {"a": 2}}}) == {"applied": {"a": 1}, "replaced": {"a": 2}}
    assert big.report({"mode": "exact"}) is None


def test_one_determinism_row_per_mode():
    rows = dict(modes.DETERMINISM)
    assert tuple(rows) == modes.MODES and all(isinstance(w, str) and w for w in rows.values())
    assert {m: (1 if w == "bitwise" else 2 if w == "band" else None) for m, w in rows.items()} == {m: modes.KIT_MODES[m].tier if m in modes.KIT_MODES else None for m in modes.MODES}
