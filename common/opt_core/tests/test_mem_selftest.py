"""opt_core.mem's CPU self-test (python -m opt_core.mem._selftest) as part of the suite: torch on CPU when importable, else skipped by name."""
import pytest


def test_mem_selftest_passes_on_cpu():
    pytest.importorskip("torch")
    from opt_core.mem import _selftest
    assert _selftest.main() == 0
