"""The capability-keyed settings rows of kernels/rfd_layernorm and kernels/dtk_kernels as their accessors answer them (torch, CPU: no CUDA device =
the capability-free defaults, which are cc 9.0's values; a CUDA box answers its device's row). The table literals themselves: test_a100_tables."""
import pytest

torch = pytest.importorskip("torch", reason="needs torch (CPU)")

from opt_core.kernels import safe_settings as S  # noqa: E402


def _expect(table, default):
    """The row this interpreter's current device is served: its capability's row on a CUDA box, else the default."""
    return S.resolve_row(table, S.device_cc(), S.triton_mm(), default=default)


def test_rfd_layernorm_settings_accessors():
    from opt_core.kernels import rfd_layernorm as M
    row = _expect(M._SETTINGS_BY_CC, M._SETTINGS)
    assert M.settings_for() is row and M.settings_for(torch.device("cpu")) is M._SETTINGS
    assert M._MIN_NUMEL is None and M.min_numel() == row["min_numel"]
    assert [M.num_warps(b, torch.device("cpu")) for b in (8, 32, 64, 128, 256, 512, 1024)] == [1, 1, 1, 2, 4, 4, 8]      # the default ladder (cc 9.0's)
    M._MIN_NUMEL = 0                                                              # the process pin a caller's serve layer may set: honoured, 0 included
    try:
        assert M.min_numel() == 0
        M._MIN_NUMEL = 12345
        assert M.min_numel() == 12345
    finally:
        M._MIN_NUMEL = None
    assert M.min_numel(torch.device("cpu")) == 1 << 18


def test_dtk_row_kernels_settings_accessors():
    pytest.importorskip("triton", reason="dtk_kernels imports triton at module import")
    from opt_core.kernels import dtk_kernels as D
    row = _expect(D._ROW_WARPS_BY_CC, D._ROW_WARPS)
    assert D.row_settings() is row and D.row_settings(torch.device("cpu")) is D._ROW_WARPS
    cpu = torch.device("cpu")                                                     # the default rule (cc 9.0's): 4 warps up to BC 1024, 8 above; swiglu 4
    assert [D.row_warps("ln_modulate", bc, cpu) for bc in (32, 1024, 2048, 8192)] == [4, 4, 8, 8]
    assert [D.row_warps("gate_residual", bc, cpu) for bc in (32, 1024, 2048)] == [4, 4, 8] and [D.row_warps("swiglu", bh, cpu) for bh in (32, 1024)] == [4, 4]
