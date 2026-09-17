"""Out-of-memory propagates through the served TriMul adapter: the FPF add-on's fast forward reroutes a kernel ERROR to the stock op (counted),
but an out-of-memory raised by the kernel launch is re-raised first (opt_core.oom.is_oom) — mocked on the CPU, no GPU."""
import os
import sys
import types

import pytest

import torch                                                     # the adapter imports torch; the kit's CPU suite carries it

ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "forward", "rf3_fpf_trimul_addon", "rf3fpf")


@pytest.fixture
def adapter(monkeypatch):
    """The real carried adapter module with the engine (foundry) and the core kernel package (fpf_trimul_v4.generic) stood in."""
    monkeypatch.syspath_prepend(os.path.abspath(ADAPTER_DIR))
    foundry = types.ModuleType("foundry"); foundry.SHOULD_USE_CUEQUIVARIANCE = True
    monkeypatch.setitem(sys.modules, "foundry", foundry)
    G = types.ModuleType("fpf_trimul_v4.generic")
    G.supported = lambda x, mask, weights=None: (True, "ok")
    G.pack_weights = lambda **kw: {"packed": True}
    G.TrimulUnsupported = type("TrimulUnsupported", (Exception,), {})
    G.raises = None
    def trimul(x, mask, **kw):
        raise G.raises
    G.trimul = trimul
    pkg = types.ModuleType("fpf_trimul_v4"); pkg.generic = G
    monkeypatch.setitem(sys.modules, "fpf_trimul_v4", pkg); monkeypatch.setitem(sys.modules, "fpf_trimul_v4.generic", G)
    sys.modules.pop("fpf_rf3_adapter", None)
    import fpf_rf3_adapter as adp
    monkeypatch.setitem(adp._ORIG, "forward", lambda self, pair: "stock-forward")
    adp.COUNTS.update({"served": 0, "fallback": {}, "errors": 0, "first": None})
    yield adp, G
    sys.modules.pop("fpf_rf3_adapter", None)


def _module():
    nn = torch.nn
    return types.SimpleNamespace(use_cuequivariance=True, d_pair=128, d_hidden=128, direction="outgoing",
                                 norm_in=nn.LayerNorm(128), p_in=nn.Linear(128, 256, bias=False), g_in=nn.Linear(128, 256, bias=False),
                                 norm_out=nn.LayerNorm(128), p_out=nn.Linear(128, 128, bias=False), g_out=nn.Linear(128, 128, bias=False))


def test_out_of_memory_in_the_trimul_launch_propagates_out_of_the_adapter(adapter):
    adp, G = adapter
    G.raises = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 13.10 GiB (mocked)")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        adp._fast_forward(_module(), torch.zeros(8, 8, 128))
    assert adp.COUNTS["errors"] == 0 and adp.COUNTS["served"] == 0          # not counted as a kernel error, not rerouted


def test_a_non_oom_kernel_error_still_takes_the_counted_stock_route(adapter):
    adp, G = adapter
    G.raises = RuntimeError("CUDA error: an illegal memory access was encountered (mocked)")
    assert adp._fast_forward(_module(), torch.zeros(8, 8, 128)) == "stock-forward"
    assert adp.COUNTS["errors"] == 1 and adp.COUNTS["served"] == 0
