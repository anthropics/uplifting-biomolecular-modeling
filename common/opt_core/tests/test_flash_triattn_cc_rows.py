"""flash_triattn launch-configuration resolution by (compute capability, triton version) and the build-failure safety net: pure checks (no GPU)."""
import io
import contextlib
import pytest

torch = pytest.importorskip("torch")
FT = pytest.importorskip("opt_core.kernels.flash_triattn")

TUNED32 = dict(BLOCK_M=64, BLOCK_N=64, ROWS=1, num_warps=4, num_stages=3, ORDER=0)
TUNED16 = dict(BLOCK_M=64, BLOCK_N=64, ROWS=1, num_warps=4, num_stages=2, ORDER=0)
SAFE_FP32_64 = dict(BLOCK_M=64, BLOCK_N=64, ROWS=1, num_warps=4, num_stages=1, ORDER=0)


def test_cc90_has_no_row_any_triton():
    for tmm in ("2.3", "3.3", "3.6", "3.7", ""):
        assert FT._cc_row((9, 0), tmm) == {} and FT._cc_key((9, 0), tmm) is None
    assert FT._cc_row(None, "3.7") == {} and FT._cc_key(None, "3.7") is None


def test_sm80_default_row_is_the_tuned_set_and_2_3_is_the_named_exception():
    assert sorted(FT._CONFIG_TABLE_BY_CC) == ["10.0|*", "10.3|*", "8.0|*", "8.0|2.3"]
    for tmm in ("3.3", "3.6", "3.7", "4.0", ""):                                       # every triton without a named exception: the tuned default row
        assert FT._cc_key((8, 0), tmm) == "8.0|*"
        row = FT._cc_row((8, 0), tmm)
        assert row["16bit"][32][0][1] == TUNED32 and row["16bit"][16][0][1] == TUNED16 and "fp32" not in row   # other cells -> the default tables
    assert FT._cc_key((8, 0), "2.3") == "8.0|2.3" and FT._cc_row((8, 0), "2.3") is FT._SAFE_SINGLE_STAGE      # the exact key wins: single stage
    safe = FT._SAFE_SINGLE_STAGE
    assert sorted(safe) == ["16bit", "fp32"] and all(sorted(safe[c]) == [16, 32, 64, 128] for c in safe)
    assert all(cells[0][1]["num_stages"] == 1 for cls in safe.values() for cells in cls.values())
    assert safe["fp32"][64][0][1] == SAFE_FP32_64


def test_pick_config_and_facts_without_cuda_are_the_default_tables():
    """No CUDA device in this process -> no cc row applies: pick_config == the H100-tuned tables (the cc 9.0 behaviour); no cells / settings fact."""
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present; the GPU tests cover the device rows")
    for D in (16, 32, 64, 128):
        assert FT.pick_config(D, 512, 512, 4, torch.bfloat16) == dict(FT._CONFIG_TABLE[D][0][1])
        assert FT.pick_config(D, 512, 512, 4, torch.float32) == dict(FT._CONFIG_TABLE_F32[D][0][1])
    assert FT.cells_key() is None and FT.cells_note() is None and FT.settings_word() is None and FT.safe_state()["on"] is False


class _FakeCompilationError(RuntimeError):
    """Stands in for triton's compile-time failure: its message carries the MLIR words _is_build_failure recognises."""
    def __init__(self, msg="PassManager::run failed"):
        super().__init__(msg)


@pytest.fixture
def clean_state():
    FT._NET.reset()
    yield
    FT._NET.reset()


def test_build_failure_switches_to_safe_settings_once_with_one_line(clean_state):
    calls = []
    def launch(cfg):
        calls.append(dict(cfg))
        if cfg["num_stages"] != 1:
            raise _FakeCompilationError()
        return "ok"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert FT._launch_or_safe(launch, dict(TUNED32), 32, torch.bfloat16, explicit=False, where="triton 9.9, cc 8.0") == "ok"
    assert calls == [TUNED32, FT._SAFE_SINGLE_STAGE["16bit"][32][0][1]]
    lines = [l for l in err.getvalue().splitlines() if l.strip()]
    assert lines == ["[opt_core/flash_triattn] safe settings served (build_failed:_FakeCompilationError, triton 9.9, cc 8.0)"], lines
    assert FT.safe_state() == {"on": True, "reason": "build_failed:_FakeCompilationError", "where": "triton 9.9, cc 8.0", "note": None, "wide": True, "cells": {}} and FT.settings_word() == "safe:build_failed:_FakeCompilationError"
    assert FT.pick_config(32, 512, 512, 4, torch.bfloat16) == FT._SAFE_SINGLE_STAGE["16bit"][32][0][1]          # the process now picks the safe settings
    assert FT.pick_config(64, 512, 512, 16, torch.float32) == SAFE_FP32_64
    err2 = io.StringIO()
    with contextlib.redirect_stderr(err2):                                                                        # a later build failure WITH the safe settings serving:
        with pytest.raises(FT.BuildFailed) as e:                                                                  # the lever cannot run — BuildFailed, no second line
            FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(_FakeCompilationError()), dict(TUNED32), 32, torch.bfloat16, explicit=False, where="w")
    assert "--mode off" in str(e.value) and err2.getvalue() == ""


def test_safe_settings_failing_to_build_too_is_build_failed(clean_state):
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(FT.BuildFailed) as e:
            FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(_FakeCompilationError("failed to legalize operation")), dict(TUNED16), 16, torch.bfloat16, explicit=False, where="triton 9.9, cc 8.0")
    assert e.value.kind == "build_failed" and "flash_triattn: the safe settings cannot build/run either" in str(e.value)


def test_non_build_exceptions_and_explicit_configs_propagate_unchanged(clean_state):
    class Boom(ValueError):
        pass
    with pytest.raises(Boom):
        FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(Boom("x")), dict(TUNED32), 32, torch.bfloat16, explicit=False, where="w")
    with pytest.raises(RuntimeError):                                                       # a RuntimeError without the compiler's words (e.g. a CUDA runtime error): not a build failure
        FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(RuntimeError("CUDA error: an illegal memory access was encountered")), dict(TUNED32), 32, torch.bfloat16, explicit=False, where="w")
    with pytest.raises(_FakeCompilationError):                                              # a caller-pinned config: its build failure is the caller's
        FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(_FakeCompilationError()), dict(TUNED32), 32, torch.bfloat16, explicit=True, where="w")
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory") if hasattr(torch.cuda, "OutOfMemoryError") else RuntimeError("CUDA out of memory")
    with pytest.raises(type(oom)):                                                          # an out-of-memory is never a build failure
        FT._launch_or_safe(lambda cfg: (_ for _ in ()).throw(oom), dict(TUNED32), 32, torch.bfloat16, explicit=False, where="w")
    assert FT.safe_state()["on"] is False and FT.settings_word() is None


def test_is_build_failure_recognises_tritons_classes_when_importable():
    from opt_core.kernels import safe_settings as S
    for t in S.build_error_types():
        assert issubclass(t, BaseException)
    assert FT._is_build_failure(_FakeCompilationError()) and not FT._is_build_failure(ValueError("x"))


def test_an_unknown_capability_on_the_default_tables_is_named_once(monkeypatch, capfd):
    """P12: a CUDA device whose capability has no _CONFIG_TABLE_BY_CC row (and is not cc 9.0, whose measurements the default tables are) is served the
    default tables — engaged — with ONE info line and cells_note() == 'default:no_row'; cc 9.0 and a capability with a row say nothing."""
    FT._NET.reset()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(FT, "_cuda_device", lambda device=None: torch.device("cpu"))
    monkeypatch.setattr(FT, "_device_cc", lambda device: (12, 0))
    monkeypatch.setattr(FT, "_triton_mm", lambda: "3.7")
    assert FT._cc_rows(32, torch.bfloat16) == [] and FT.pick_config(32, 512, 512, 4) == dict(FT._CONFIG_TABLE[32][0][1]) if FT._CONFIG_TABLE[32][0][0] >= 512 else True
    err = capfd.readouterr().err
    assert err.startswith("[opt_core/flash_triattn] default settings (no row for cc 12.0, cc 12.0, triton 3.7); tuned rows exist for cc 8.0, 9.0, 10.0, 10.3") and err.count("\n") == 1
    assert FT.cells_key() is None and FT.cells_note() == "default:no_row" and capfd.readouterr().err == ""
    FT._NET.reset()
    monkeypatch.setattr(FT, "_device_cc", lambda device: (9, 0))
    assert FT._cc_rows(32, torch.bfloat16) == [] and FT.cells_note() is None and capfd.readouterr().err == ""        # cc 9.0: the default tables ARE its measurements
    monkeypatch.setattr(FT, "_device_cc", lambda device: (8, 0))
    assert FT._cc_rows(32, torch.bfloat16) and FT.cells_key() == "8.0|*" and FT.cells_note() is None and capfd.readouterr().err == ""
    FT._NET.reset()

