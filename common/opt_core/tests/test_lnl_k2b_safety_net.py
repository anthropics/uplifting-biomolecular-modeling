"""The safety net (opt_core.kernels.safe_settings) on the lnl_fused kernels (tiles_by_arch rows) and the fpf_triatt_k2b launch cell (K2B_CELLS):
no row / no cell for a compute capability -> the SAFE tile with ONE line; a BUILD failure of the tuned tile -> the SAFE tile with ONE line;
the SAFE tile failing to build -> the lever's BuildFailed; a capability with rows (H100) -> nothing changes. Pure CPU (no launches: fake
autotuner / launch callables)."""
import io
import types
import contextlib
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
try:                                                   # the kernels decorate with triton.autotune at import; a triton whose autotuner needs the GPU driver at
    import opt_core.kernels.lnl_fused as L             # decoration time (3.3 on a driverless host: "0 active drivers") cannot import them there — skipped by name
    import opt_core.kernels.fpf_triatt_k2b.triatt_k2b as K
except (ImportError, RuntimeError) as _e:              # noqa: BLE001 — collection on a host without a GPU driver
    pytest.skip("lnl_fused / triatt_k2b need a triton that imports without a GPU driver here: %s: %s" % (type(_e).__name__, _e), allow_module_level=True)
from opt_core.kernels import safe_settings as S


class FakeCompile(RuntimeError):
    def __init__(self, msg="PassManager::run failed"):
        super().__init__(msg)


def _cfg_sig(cfgs):
    return [(dict(c.kwargs), c.num_warps, c.num_stages) for c in cfgs]


@pytest.fixture
def lnl_clean():
    L._NET.reset(); saved = dict(L._ROW_STATE)
    yield
    L._NET.reset(); L._ROW_STATE.clear(); L._ROW_STATE.update(saved)


@pytest.fixture
def k2b_clean():
    K._NET.reset(); saved = dict(K._CELL)
    yield
    K._NET.reset(); K._CELL.update(saved)


# ---------------------------------------------------------------------------------------------------------------------- lnl_fused
def test_lnl_safe_tiles_are_the_safe_rows():
    assert _cfg_sig([L._safe_config("_fused_transition_kernel")]) == [({"BM": 64, "BH": 64}, 4, 1)]
    assert _cfg_sig([L._safe_config("_ln_linear_kernel")]) == [({"BM": 64}, 4, 1)]
    assert S.safe_row("lnl:transition", (12, 0))["status"] == "SAFE:inferred" and S.safe_row("lnl:ln_linear", "9.0")["settings"] == {"BM": 64, "num_warps": 4, "num_stages": 1}


def test_lnl_no_row_for_this_cc_serves_the_default_space_with_one_info_line(lnl_clean):
    """This CPU process has no tiles row (cc unknown): the autotuner holds the kernels' DEFAULT TILE ALONE (the module default, pinned: no
    search ever runs outside the sweep word PF_LNL_AUTOTUNE_ALL=1) and the first launch prints ONE info line naming the capabilities that have
    rows; no safe word; a pinned cache already in place is kept."""
    if torch.cuda.is_available():
        pytest.skip("a CUDA device with a tiles row may be present")
    assert L._ROW_STATE == {"_fused_transition_kernel": "none", "_ln_linear_kernel": "none"}
    assert _cfg_sig(L._fused_transition_kernel.configs) == _cfg_sig(L._TCFG_ALL[:1]) and _cfg_sig(L._ln_linear_kernel.configs) == _cfg_sig(L._LCFG_ALL[:1])
    assert isinstance(L._PINS["_fused_transition_kernel"], L._PinnedTiles) and isinstance(L._PINS["_ln_linear_kernel"], L._PinnedTiles)
    tuner = types.SimpleNamespace(configs=list(L._fused_transition_kernel.configs), cache=L._PinnedTiles({("k",): "x"}, L._TCFG_ALL[0]))
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert L._run("_fused_transition_kernel", tuner, lambda: "launched") == "launched"
        assert L._run("_ln_linear_kernel", types.SimpleNamespace(configs=list(L._ln_linear_kernel.configs), cache={}), lambda: "again") == "again"
    lines = [l for l in err.getvalue().splitlines() if l]
    tuned = ", ".join(S.table_ccs(L._tile_table()))
    assert lines == ["[opt_core/lnl_fused] default settings (no row for cc unknown, cc ?, triton %s); tuned rows exist for cc %s" % (S.triton_mm() or "?", tuned)], lines   # ONE line per lever
    assert "8.0" in tuned and "9.0" in tuned
    assert L.settings_word() is None and L.cells_note() == "default:no_row" and not L.safe_state()["on"]
    assert isinstance(tuner.cache, L._PinnedTiles) and dict.get(tuner.cache, ("k",)) == "x" and _cfg_sig(tuner.configs) == _cfg_sig(L._TCFG_ALL[:1])   # the tile set did not change: the pinned cache in place is kept


def test_lnl_no_row_then_a_build_failure_of_the_default_space_serves_the_safe_tile(lnl_clean):
    L._ROW_STATE["_fused_transition_kernel"] = "none"
    tuner = types.SimpleNamespace(configs=list(L._TCFG_ALL), cache={})
    def launch():
        if _cfg_sig(tuner.configs) != [({"BM": 64, "BH": 64}, 4, 1)]:
            raise FakeCompile()
        return "safe-launched"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert L._run("_fused_transition_kernel", tuner, launch) == "safe-launched"
    lines = [l for l in err.getvalue().splitlines() if l]
    assert len(lines) == 2 and lines[0].startswith("[opt_core/lnl_fused] default settings (no row for cc") and lines[1] == "[opt_core/lnl_fused] safe settings served (build_failed:FakeCompile, %s)" % S.where_word(None, S.triton_mm())
    assert L.settings_word() == "safe:build_failed:FakeCompile" and L.cells_note() == "default:no_row"


def test_lnl_build_failure_of_the_tuned_tiles_serves_the_safe_tile(lnl_clean):
    import triton
    L._ROW_STATE["_fused_transition_kernel"] = "row"                              # as on a capability WITH a tiles row (H100 / A100)
    tuned = [triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2)]
    tuner = types.SimpleNamespace(configs=list(tuned), cache={(0, 128, 512, False): tuned[0]})
    def launch():
        if _cfg_sig(tuner.configs) != [({"BM": 64, "BH": 64}, 4, 1)]:
            raise FakeCompile()
        return "safe-launched"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert L._run("_fused_transition_kernel", tuner, launch) == "safe-launched"
        assert L._run("_fused_transition_kernel", tuner, launch) == "safe-launched"          # the process stays on the safe tile, no second line
    assert [l for l in err.getvalue().splitlines() if l] == ["[opt_core/lnl_fused] safe settings served (build_failed:FakeCompile, %s)" % S.where_word(None, S.triton_mm())]
    assert _cfg_sig(tuner.configs) == [({"BM": 64, "BH": 64}, 4, 1)] and tuner.cache == {}    # tile set switched, the autotuner's per-key picks reset
    assert L.settings_word() == "safe:build_failed:FakeCompile"


def test_lnl_safe_tile_failing_too_is_build_failed_and_other_errors_propagate(lnl_clean):
    import triton
    L._ROW_STATE["_ln_linear_kernel"] = "row"
    tuner = types.SimpleNamespace(configs=[triton.Config({"BM": 256}, num_warps=8, num_stages=2)], cache={})
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(L.BuildFailed) as e:
            L._run("_ln_linear_kernel", tuner, lambda: (_ for _ in ()).throw(FakeCompile("failed to legalize operation")))
    assert e.value.kind == "build_failed" and "lnl_fused: the safe settings cannot build/run either" in str(e.value) and "--mode off" in str(e.value)
    L._NET.reset()
    with pytest.raises(ValueError):                                               # not a build failure: propagates, the net stays off
        L._run("_ln_linear_kernel", tuner, lambda: (_ for _ in ()).throw(ValueError("shape")))
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    with pytest.raises(RuntimeError):
        L._run("_ln_linear_kernel", tuner, lambda: (_ for _ in ()).throw(oom))
    assert not L.safe_state()["on"]


def test_lnl_row_present_launches_the_tuned_tiles_silently(lnl_clean):
    """A tiles row names this capability: the launch is silent, the tile set is the row's, and the autotuner's cache becomes the row's PINNED
    per-key tiles (serving mode: a plain benchmark cache is replaced, never searched); no settings word."""
    import triton
    L._ROW_STATE["_fused_transition_kernel"] = "row"
    tuned = [triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2)]
    tuner = types.SimpleNamespace(configs=list(tuned), cache={"k": 1})
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert L._run("_fused_transition_kernel", tuner, lambda: "tuned") == "tuned"
    assert err.getvalue() == "" and _cfg_sig(tuner.configs) == _cfg_sig(tuned) and L.settings_word() is None
    pins = L._PINS.get("_fused_transition_kernel")
    assert (tuner.cache is pins) if pins is not None else (tuner.cache == {"k": 1})       # the row's pinned tiles serve every key from here (no benchmark)


# ----------------------------------------------------------------------------------------------------------------- fpf_triatt_k2b
SAFE_K2B = {"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 1, "num_warps": 4, "num_stages": 1, "ORDER": 0, "MAXNREG": None}


def test_k2b_safe_cell_is_the_safe_row_for_every_capability(k2b_clean):
    assert K._safe_cfg(32, torch.bfloat16) == SAFE_K2B and S.safe_row("k2b", (8, 0))["status"] == "SAFE" and S.safe_row("k2b", (12, 0))["status"] == "SAFE:inferred"
    K._CELL["cc"] = (8, 0)
    assert K._safe_cfg(16, torch.float32) == SAFE_K2B


def test_k2b_capability_with_a_cell_is_unchanged(k2b_clean):
    """A cells entry exists (H100 / A100) or no table is named: pick_config = the table cell, no line, no fact."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        c = K.pick_config(32, 512, 512, 4, torch.bfloat16)
    assert c == dict(K._CONFIG_TABLE[32][0][1]) and err.getvalue() == "" and K.settings_word() is None and not K.safe_state()["on"]


def test_k2b_no_entry_for_this_cc_serves_the_package_default_with_one_info_line(k2b_clean, tmp_path, monkeypatch):
    """A K2B cells table WITHOUT an entry for the device's capability: the package default cell serves (the tuned settings for everyone), ONE info line at
    table load naming the capabilities that have entries, cells_note=default:no_row, no safe word; the table cells are untouched."""
    import json
    tab = {"format": "k2b_cells/v2", "by_cc": {"8.0": "sm80", "9.0": "sm90", "10.0": "sm100"}, "by_name": {}, "entries": {}}
    p = tmp_path / "K2B_CELLS.json"; p.write_text(json.dumps(tab))
    monkeypatch.setenv("PF_TRIATTN_TABLE", str(p))
    monkeypatch.setattr(K.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(K.torch.cuda, "get_device_capability", lambda i=0: (8, 6))          # e.g. a consumer Ampere card absent from by_cc / by_name
    monkeypatch.setattr(K.torch.cuda, "get_device_name", lambda i=0: "NVIDIA RTX A6000")
    before = dict(K._CONFIG_TABLE[32][0][1])
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert K._pf_load_device_table() is None
        assert K.pick_config(32, 512, 512, 4, torch.bfloat16) == before                   # the package default cell, not the safe cell
    lines = [l for l in err.getvalue().splitlines() if l]
    assert len(lines) == 1 and lines[0].startswith("[opt_core/fpf_triatt_k2b] default settings (no row for cc 8.6 (no cc/name match, 'NVIDIA RTX A6000', K2B_CELLS.json), cc 8.6, triton ") and lines[0].endswith("; tuned rows exist for cc 8.0, 9.0, 10.0"), lines
    assert K._CELL == {"no_cell": True, "cc": (8, 6)} and K.cells_note() == "default:no_row" and K.settings_word() is None and not K.safe_state()["on"]


def test_k2b_build_failure_serves_the_safe_cell_and_safe_failing_is_build_failed(k2b_clean):
    K._CELL["cc"] = (8, 0)
    tuned = dict(K._CONFIG_TABLE[32][0][1]); calls = []
    def launch(c):
        calls.append(dict(c))
        if c["num_stages"] != 1:
            raise FakeCompile()
        return "ok"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert K._NET.run(launch, tuned, lambda: K._safe_cfg(32, torch.bfloat16), where=K._where(), explicit=False) == "ok"
    assert calls == [tuned, SAFE_K2B] and [l for l in err.getvalue().splitlines() if l] == ["[opt_core/fpf_triatt_k2b] safe settings served (build_failed:FakeCompile, cc 8.0, triton %s)" % (S.triton_mm() or "?")]
    assert K.pick_config(32, 512, 512, 4, torch.bfloat16) == SAFE_K2B and K.settings_word() == "safe:build_failed:FakeCompile"
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(K.BuildFailed) as e:
            K._NET.run(lambda c: (_ for _ in ()).throw(FakeCompile()), tuned, lambda: K._safe_cfg(32, torch.bfloat16), where=K._where(), explicit=False)
    assert e.value.kind == "build_failed" and "fpf_triatt_k2b: the safe settings cannot build/run either" in str(e.value)
    K._NET.reset()
    with pytest.raises(FakeCompile):                                              # a caller-pinned cell is never retried
        K._NET.run(lambda c: (_ for _ in ()).throw(FakeCompile()), tuned, lambda: K._safe_cfg(32, torch.bfloat16), where=K._where(), explicit=True)
    assert not K.safe_state()["on"]


def test_sources_route_every_launch_through_the_net():
    import os
    here = os.path.dirname(os.path.abspath(__file__)); kern = os.path.join(os.path.dirname(here), "opt_core", "kernels")
    lsrc = open(os.path.join(kern, "lnl_fused.py")).read(); ksrc = open(os.path.join(kern, "fpf_triatt_k2b", "triatt_k2b.py")).read()
    assert lsrc.count('_run("_fused_transition_kernel", _fused_transition_kernel, lambda: _fused_transition_kernel[grid](') == 1
    assert lsrc.count('_run("_ln_linear_kernel", _ln_linear_kernel, lambda: _ln_linear_kernel[grid](') == 1 and "raise BuildFailed(" in lsrc
    assert "_NET.run(_attn, cfg, lambda: _safe_cfg(D, q.dtype), where=_where(), explicit=config is not None)" in ksrc and ksrc.count("if is_oom(e):") == 2 and "_NET.inform(" in ksrc and "_NET.inform(" in lsrc
