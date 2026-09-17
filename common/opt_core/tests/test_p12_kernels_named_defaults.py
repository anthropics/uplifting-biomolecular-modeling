"""P12 SILENT-row fixes: a capability WITHOUT a row in a cc-keyed settings table (other than cc 9.0, whose measurements the defaults are) is served the
lever's defaults with ONE info line and a `default:no_row` note (opt_core.kernels.cell_words.name_default_row through the lever's SafeNet); an
unreadable lnl tiles table is named once.  CPU only."""
import os

import pytest

from opt_core.kernels import cell_words as CW
from opt_core.kernels import safe_settings as S


def test_name_default_row_says_once_and_only_for_unknown_capabilities(capfd):
    net = S.SafeNet("demo")
    table = {"8.0|*": {"x": 1}, "10.0|3.7": {"x": 2}}
    CW.name_default_row(net, table, (9, 0), "3.7")                      # cc 9.0: the defaults ARE its measurements
    CW.name_default_row(net, table, None, "3.7")                        # no CUDA device
    CW.name_default_row(net, table, (8, 0), "3.7")                      # a capability with a row
    CW.name_default_row(net, table, (10, 0), "3.7")                     # the exact row serves
    assert capfd.readouterr().err == "" and net.note() is None
    CW.name_default_row(net, table, (12, 0), "3.7")
    err = capfd.readouterr().err
    assert err.strip() == "[opt_core/demo] default settings (no row for cc 12.0, cc 12.0, triton 3.7); tuned rows exist for cc 8.0, 9.0, 10.0" and net.note() == "default:no_row"
    CW.name_default_row(net, table, (12, 0), "3.7"); CW.name_default_row(net, table, (10, 3), "3.7")
    assert capfd.readouterr().err == ""                                  # once per process


def test_dtk_row_settings_names_an_unknown_capability_once(monkeypatch, capfd):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton", reason="dtk_kernels imports triton at module import")
    from opt_core.kernels import dtk_kernels as D
    D._NET.reset(); D._rows_resolved.clear()
    monkeypatch.setattr(S, "row_for_device", lambda table, device=None, default=None, cache=None: default)
    monkeypatch.setattr(S, "device_cc", lambda device=None: (12, 0))
    assert D.row_settings("cuda:0") is D._ROW_WARPS and D.cells_note("cuda:0") == "default:no_row"
    err = capfd.readouterr().err
    assert err.startswith("[opt_core/dtk_kernels] default settings (no row for cc 12.0, cc 12.0, triton ") and "tuned rows exist for cc 8.0, 9.0" in err and err.count("\n") == 1
    D._NET.reset()
    monkeypatch.setattr(S, "device_cc", lambda device=None: (9, 0))
    assert D.row_settings("cuda:0") is D._ROW_WARPS and D.cells_note("cuda:0") is None and capfd.readouterr().err == ""
    D._NET.reset(); D._rows_resolved.clear()


def test_lnl_unreadable_tiles_table_is_named_once(monkeypatch, capfd, tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("triton", reason="lnl_fused imports triton at module import")
    try:                                                       # lnl_fused decorates with triton.autotune at import: a triton whose autotuner needs the GPU driver at
        from opt_core.kernels import lnl_fused as L            # decoration time (3.3 on a driverless host) cannot import it there — skipped by name
    except RuntimeError as e:
        pytest.skip("lnl_fused needs a triton that imports without a GPU driver here: %s" % e)
    bad = tmp_path / "tiles.json"; bad.write_text("{not json")
    monkeypatch.setenv("PF_LNL_TILES", str(bad))
    monkeypatch.setitem(L._TILES_STATE, "error", None)
    assert L._tile_table() == {} and L.tiles_state().startswith("unreadable:JSONDecodeError")
    err = capfd.readouterr().err
    assert err.startswith("[opt_core/lnl_fused] tiles table unreadable (") and err.count("\n") == 1
    assert L._tile_table() == {} and capfd.readouterr().err == ""          # once
    monkeypatch.setitem(L._TILES_STATE, "error", None)
    monkeypatch.delenv("PF_LNL_TILES")
    assert isinstance(L._tile_table(), dict) and L.tiles_state() is None and capfd.readouterr().err == ""   # the shipped table reads
