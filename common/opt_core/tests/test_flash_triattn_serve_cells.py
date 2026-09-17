"""flash_triattn_serve: the LEVER line carries cells=<row key> when a cc|triton row serves the device, cells_note=exception_row when a named
exception row serves, settings=safe:build_failed:<exc> once the kernel serves its safe settings; nothing extra when the default tables serve (cc 9.0);
the kernel's BuildFailed becomes the layer's Refusal('build_failed') — never a stock reroute."""
import types
import pytest

torch = pytest.importorskip("torch")
from opt_core.kernels import flash_triattn_serve as F1


def _line(monkeypatch, key, note, word=None):
    fake = types.SimpleNamespace(cells_key=lambda: key, cells_note=lambda: note, settings_word=lambda: word)
    monkeypatch.setattr(F1, "kernel_module", lambda: fake)
    monkeypatch.setattr(F1, "emit", lambda text: text)
    led = F1.ledger(min_tokens=0, impl="flash_triattn@deadbeef", origin="core")
    return F1.emit_line(led, tag="t")


def test_cells_and_settings_fact_forms(monkeypatch):
    assert " cells=8.0|*" in _line(monkeypatch, "8.0|*", None) and "settings=" not in _line(monkeypatch, "8.0|*", None)
    exc = _line(monkeypatch, "8.0|2.3", "exception_row")
    assert " cells=8.0|2.3" in exc and " cells_note=exception_row" in exc
    safe = _line(monkeypatch, "8.0|*", None, "safe:build_failed:CompilationError")
    assert " settings=safe:build_failed:CompilationError" in safe
    plain = _line(monkeypatch, None, None)                       # cc 9.0 / no device: the line is unchanged (no extra fact)
    assert "cells=" not in plain and "cells_note=" not in plain and "settings=" not in plain


def test_module_without_the_functions_gives_no_fact(monkeypatch):
    monkeypatch.setattr(F1, "kernel_module", lambda: types.SimpleNamespace())
    assert F1.cells_key() == (None, None) and F1.settings_word() is None


def test_build_failed_becomes_the_layers_refusal(monkeypatch):
    class BuildFailed(RuntimeError):
        kind = "build_failed"
    def boom(*a, **k):
        raise BuildFailed("safe single-stage settings failed to build too")
    fake = types.SimpleNamespace(BuildFailed=BuildFailed, flash_triangle_attention=boom, flash_supported=lambda q, k, v, b, m=None: (True, ""))
    monkeypatch.setattr(F1, "kernel_module", lambda: fake)
    monkeypatch.setattr(F1, "_supports_mask_arg", lambda mod: True)
    monkeypatch.setattr(F1, "_compute_dtype_name", lambda q: "bfloat16")
    led = F1.ledger(min_tokens=0, impl="flash_triattn@deadbeef", origin="core")
    q = torch.zeros(1, 4, 2, 8, 32, dtype=torch.bfloat16)
    with pytest.raises(F1.Refusal) as e:
        F1.triangle_attention(q, q, q, torch.zeros(1, 1, 2, 8, 8), ledger=led, stock=lambda *a: None)
    assert e.value.kind == "build_failed" and led.fields().get("served", 0) == 0
