"""opt_core.trimul under P12: the fpf_v4 provider turns the carried kernel's structural refusals into per-call fallback words and re-raises its
cannot-run refusals (fpf_trimul_v4 >= 4.3: `none:<Exc>`, `probe-failed`) so the MODE refuses by name; Lever.serve never reroutes a cannot-run
exception to the engine's forward; the tmk3 provider serves an UNKNOWN (dtype, C) class named (cells=unverified(...)) and keeps a MEASURED-OFF
class on the engine's forward by name.  No GPU: stub kernel modules planted in sys.modules."""
import sys
import types

import pytest

from opt_core import trimul as T


class _TU(ValueError):
    """fpf_trimul_v4.generic.TrimulUnsupported stand-in: .reason, and .cannot_run on the 4.3 reasons."""
    def __init__(self, reason, detail=""):
        super().__init__("fpf_trimul_v4.generic: unsupported (%s) %s" % (reason, detail)); self.reason = reason
        head = str(reason).split(":", 1)[0]
        if head in ("none", "probe-failed"):
            self.cannot_run = True


class _Z:
    is_cuda = True

    def __init__(self, n=512, c=128, dtype="bfloat16"):
        self.shape, self.dtype = (n, n, c), dtype
        self.device = types.SimpleNamespace(type="cuda")

    def dim(self):
        return 3


def _weights(m):
    return {k: types.SimpleNamespace(detach=lambda: None, dtype="bfloat16") for k in T.WEIGHT_KEYS}


@pytest.fixture
def stub_v4(monkeypatch):
    """A planted fpf_trimul_v4 package whose generic.trimul / .supported answer what the test sets."""
    top = types.ModuleType("fpf_trimul_v4"); top.__version__ = "4.3.0-stub"; top.__file__ = "/nonexistent/fpf_trimul_v4/__init__.py"
    G = types.ModuleType("fpf_trimul_v4.generic")
    G.TrimulUnsupported = _TU
    G.COUNTS = {"version": "4.3.0-stub"}
    G.state = {"supported": (True, ""), "raise": None, "served": 0}
    G.pack_weights = lambda **kw: {"packed": True}
    G.supported = lambda z, mask=None, weights=None: G.state["supported"]

    def trimul(z, mask, direction, weights, residual, pad):
        if G.state["raise"] is not None:
            raise G.state["raise"]
        G.state["served"] += 1
        return "kernel-out"
    G.trimul = trimul
    monkeypatch.setitem(sys.modules, "fpf_trimul_v4", top)
    monkeypatch.setitem(sys.modules, "fpf_trimul_v4.generic", G)
    monkeypatch.setattr(T, "_compute_input", lambda call, autocast_input: call.z)
    return G


def _lever(**kw):
    return T.Lever("acme-opt", "fast", provider=T.fpf_v4(_weights), expected=("mode_stock", "below_min_tokens", "c=64/64"), **kw)


def _call(**kw):
    return T.Call(module=types.SimpleNamespace(), z=_Z(**kw), mask=None, direction="outgoing", residual=False, orig=lambda: "engine-out")


def test_structural_refusal_mid_call_is_a_counted_fallback(stub_v4):
    L = _lever()
    assert L.serve(_call()) == "kernel-out"
    stub_v4.state["raise"] = _TU("c=64/64", "c_z=64 d=64")
    assert L.serve(_call()) == "engine-out"
    c = L.census()
    assert c["served"] == 1 and c["fallback"] == {"c=64/64": 1} and c["errors"] == {} and L.gate().ok


@pytest.mark.parametrize("reason", ["none:CompilationError", "probe-failed"])
def test_cannot_run_mid_call_propagates_and_is_never_rerouted(stub_v4, reason):
    L = _lever()
    stub_v4.state["raise"] = _TU(reason, "the SAFE cell failed to build")
    with pytest.raises(_TU) as e:
        L.serve(_call())
    assert T.cannot_run(e.value) and e.value.reason == reason
    c = L.census()
    assert c["served"] == 0 and c["fallback"] == {} and c["errors"] == {}          # nothing counted: not a fallback, not an error — the mode's refusal


def test_cannot_run_without_the_attribute_is_wrapped_by_reason_head(stub_v4):
    """A kernel whose exception lacks `cannot_run` but names a cannot-run reason: wrapped in opt_core.trimul.CannotRun (propagates)."""
    class _Old(ValueError):
        def __init__(self, reason): super().__init__(reason); self.reason = reason
    stub_v4.TrimulUnsupported = _Old
    L = _lever()
    stub_v4.state["raise"] = _Old("none:OutOfResources")
    with pytest.raises(T.CannotRun) as e:
        L.serve(_call())
    assert e.value.cannot_run and e.value.reason == "none:OutOfResources" and "fpf_v4" in str(e.value) and "refuses by name" in str(e.value)


def test_supported_false_with_a_cannot_run_reason_refuses_at_eligibility(stub_v4):
    L = _lever()
    stub_v4.state["supported"] = (False, "probe-failed")
    with pytest.raises(T.CannotRun) as e:
        L.serve(_call())
    assert e.value.reason == "probe-failed" and L.census()["fallback"] == {}
    stub_v4.state["supported"] = (False, "no-cell")                                    # structural: the engine's forward, counted
    assert L.serve(_call()) == "engine-out" and L.census()["fallback"] == {"no-cell": 1}


def test_the_packages_cannot_run_function_is_consulted_when_present(stub_v4):
    stub_v4.cannot_run = lambda reason: str(reason) == "custom-off"
    assert T._v4_cannot_run(stub_v4, reason="custom-off") and not T._v4_cannot_run(stub_v4, reason="none:X")   # the package's word wins over the head rule
    del stub_v4.cannot_run
    assert T._v4_cannot_run(stub_v4, reason="none:X") and T._v4_cannot_run(stub_v4, reason="probe-failed") and not T._v4_cannot_run(stub_v4, reason="c=64/64")
    assert T._v4_cannot_run(stub_v4, exc=_TU("none:E")) and not T._v4_cannot_run(stub_v4, exc=types.SimpleNamespace(cannot_run=False, reason="none:E"))


def test_generic_kernel_errors_still_reroute_per_call(stub_v4):
    L = _lever()
    stub_v4.state["raise"] = RuntimeError("illegal memory access")
    assert L.serve(_call()) == "engine-out"
    assert L.census()["errors"] == {"RuntimeError": 1} and not L.gate().ok


# ------------------------------------------------------------------------------------------------------------------ tmk3 cell verdicts
@pytest.fixture
def stub_tmk3(monkeypatch):
    top = types.ModuleType("fpf_trimul"); top.__version__ = "3.x-stub"; top.__file__ = "/nonexistent/fpf_trimul/__init__.py"
    K = types.ModuleType("fpf_trimul.kernels"); K.CUDA_GRID_Y_MAX = 65535; K.launch_grid_y = lambda N, C, cfg: 1
    K.trimul_forward = lambda zb, direction, mb, w, **kw: "tmk3-out"
    Tm = types.ModuleType("fpf_trimul.trimul"); Tm._MODE = "exact"; Tm.CONTRACT = "cublas"; Tm._select_cfg = lambda cdt, N, C: {"A": {"BM": 128}}
    Tm.verdicts = {("bfloat16", 128): ("verified", "record"), ("bfloat16", 64): ("off", "cell:bf16_C64+off(not-measured)"),
                   ("bfloat16", 512): ("unverified", "unverified(bf16_C512)"), ("float16", 128): ("unsupported", "dtype:float16")}
    Tm.cell_verdict = lambda cdt, C: Tm.verdicts[(str(cdt), int(C))]
    for name, m in (("fpf_trimul", top), ("fpf_trimul.kernels", K), ("fpf_trimul.trimul", Tm)):
        monkeypatch.setitem(sys.modules, name, m)
    return K, Tm


class _ZT(_Z):
    def __getitem__(self, i):
        return self

    def contiguous(self):
        return self


def _tmk3_lever(lines, wdtype="bfloat16"):
    import torch  # noqa: F401 — tmk3.fn stacks batched outputs with torch; the rank-3 path below never calls it
    prov = T.tmk3(lambda m: {k: types.SimpleNamespace(dtype=wdtype, detach=lambda: None) for k in T.WEIGHT_KEYS}, mode="exact")
    L = T.Lever("acme-opt", "exact", provider=prov, expected=("mode_stock", "below_min_tokens", "cell:bf16_C64"), stream=types.SimpleNamespace(write=lambda t: lines.append(t) if t.strip() else None, flush=lambda: None))
    return L, prov


def test_tmk3_measured_off_class_is_the_engines_forward_by_name(stub_tmk3, monkeypatch):
    pytest.importorskip("torch")
    lines = []
    L, prov = _tmk3_lever(lines)
    monkeypatch.setattr(prov, "fn", lambda call: "tmk3-out")                            # eligibility is what this test reads; the launch is the GPU suite's
    out = L.serve(T.Call(module=types.SimpleNamespace(), z=_ZT(512, 64), mask=None, direction="outgoing", residual=False, orig=lambda: "engine-out"))
    assert out == "engine-out" and L.census()["fallback"] == {"cell:bf16_C64+off(not-measured)": 1}
    assert L.serve(T.Call(module=types.SimpleNamespace(), z=_ZT(512, 128), mask=None, direction="outgoing", residual=False, orig=lambda: "engine-out")) == "tmk3-out"
    assert L.gate().ok                                                                   # the kit's expected word `cell:bf16_C64` still names it (prefix)
    assert "cells=" not in L.line() and lines == []


def test_tmk3_unknown_class_is_served_and_named_once(stub_tmk3, monkeypatch):
    pytest.importorskip("torch")
    lines = []
    L, prov = _tmk3_lever(lines)
    monkeypatch.setattr(prov, "fn", lambda call: "tmk3-out")
    for _ in range(3):
        out = L.serve(T.Call(module=types.SimpleNamespace(), z=_ZT(512, 512), mask=None, direction="incoming", residual=False, orig=lambda: "engine-out"))
        assert out == "tmk3-out"
    c = L.census()
    assert c["served"] == 3 and c["fallback"] == {} and c["cells"] == {"unverified(bf16_C512)": 3}
    assert len(lines) == 1 and "cells=unverified(bf16_C512)" in lines[0] and "provider tmk3.exact" in lines[0]
    assert " cells=unverified(bf16_C512):3 gate=ok" in L.line()
    _L16, prov16 = _tmk3_lever([], wdtype="float16")                                      # a dtype the kernels do not compute (the compute dtype is the weights' outside autocast): refused by name
    with pytest.raises(T.Refused) as e:
        prov16.eligible(T.Call(module=types.SimpleNamespace(), z=_ZT(512, 128, "float16"), mask=None, direction="incoming", residual=False, orig=lambda: None))
    assert e.value.reason == "dtype:float16"


def test_tmk3_verified_class_line_is_unchanged(stub_tmk3, monkeypatch):
    pytest.importorskip("torch")
    lines = []
    L, prov = _tmk3_lever(lines)
    monkeypatch.setattr(prov, "fn", lambda call: "tmk3-out")
    assert L.serve(T.Call(module=types.SimpleNamespace(), z=_ZT(512, 128), mask=None, direction="outgoing", residual=False, orig=lambda: "engine-out")) == "tmk3-out"
    line = L.line()
    assert "cells=" not in line and line.endswith("errors=none gate=ok") and lines == []
