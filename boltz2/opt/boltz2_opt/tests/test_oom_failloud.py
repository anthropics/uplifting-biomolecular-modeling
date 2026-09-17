"""On a GPU out-of-memory error the kit RAISES — no automatic fallback at any served site (no stock re-run of the call, no
lighter retry, no staged degradation); the one predicate is the core's ``opt_core.oom.is_oom``. Behavioural tests mock the callee to raise
torch's OutOfMemoryError and assert it reaches the caller while a non-OOM error keeps the site's named route; sites that need a built model or a
GPU to reach are held by a source contract (the handler's first statement is ``if is_oom(e): raise``). CPU only."""
import os
import re
import sys

import pytest

torch = pytest.importorskip("torch")
from opt_core.oom import is_oom  # noqa: E402

from .. import stack  # noqa: E402

OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
TRUNK = stack.kit_path("forward/trunk_levers")
DIT = stack.kit_path("forward/dit_hoist")
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST = re.compile(r"except Exception as (\w+):[^\n]*\n\s+if is_oom\(\1\): raise")      # the handler's first statement


def _import_kit(name, root=TRUNK, sub=""):
    d = os.path.join(root, sub) if sub else root
    if d not in sys.path:
        sys.path.insert(0, d)
    import importlib
    return importlib.import_module(name)


def _handlers_first(path, n_expected):
    """The file imports the core predicate once and at least n_expected broad handlers open with the re-raise."""
    src = open(path).read()
    assert src.count("from opt_core.oom import is_oom") == 1, path
    got = len(FIRST.findall(src))
    assert got >= n_expected, f"{os.path.basename(path)}: {got} handlers open with `if is_oom(e): raise`, expected >= {n_expected}"
    return src


def test_core_predicate_recognises_torch_oom():
    assert is_oom(OOM("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_oom(ValueError("out of range")) and not is_oom(KeyError("x"))


def test_adapters_reraise_oom_first():
    """pairblock / transition / msa_kernels: the kernel-error branch re-raises out-of-memory before counting the error and serving the call by stock."""
    for name in ("pairblock.py", "transition.py", "msa_kernels.py"):
        _handlers_first(os.path.join(PKG, name), 1)
    assert not os.path.exists(os.path.join(PKG, "oom.py")), "one predicate: opt_core.oom.is_oom — no kit-local spelling"


class _FakeCuda(torch.Tensor):
    is_cuda = True                                             # a CPU tensor that passes the levers' device guards (no kernel runs: the callee is mocked)


def _fake(shape=(1, 8, 8, 4)):
    return torch.zeros(shape).as_subclass(_FakeCuda)


def test_flash_patch_forward_propagates_oom(monkeypatch):
    FP = _import_kit("boltz_flash_triattn_patch")
    monkeypatch.setattr(FP, "_eligible", lambda self, x: (True, ""))
    monkeypatch.setattr(FP, "_site", lambda self: "test")
    orig_calls = []
    monkeypatch.setitem(FP._STATE, "orig_forward", lambda self, x, mask=None, chunk_size=None, use_kernels=False: orig_calls.append(1) or x)
    def core_oom(self, x, mask):
        raise OOM("CUDA out of memory. (flash core)")
    monkeypatch.setattr(FP, "_flash_core", core_oom)
    with pytest.raises(OOM):
        FP._patched_forward(object(), _fake())
    assert orig_calls == []                                    # no stock result served for the call
    def core_err(self, x, mask):
        raise ValueError("shape")
    monkeypatch.setattr(FP, "_flash_core", core_err)
    FP._patched_forward(object(), _fake())                     # a non-OOM error keeps the named stock-for-this-call route
    assert orig_calls == [1] and FP.STATS["stock|exception:ValueError"] >= 1
    _handlers_first(os.path.join(TRUNK, "boltz_flash_triattn_patch.py"), 2)   # the module forward and the kernel-function safety nets


def test_trunk_levers_and_graph_patch_handlers_reraise_oom_first():
    """boltz_trunk_levers (the mask-scope probe) and the graph patch's capture handler (a non-OOM capture failure keeps its named pre-draw
    route) — sites inside module forwards / CUDA-graph capture: source contract."""
    _handlers_first(os.path.join(TRUNK, "boltz_trunk_levers.py"), 1)
    src = _handlers_first(os.path.join(DIT, "src", "boltz_graph_patch.py"), 1)
    assert "CAPTURE FAILED -> predraw mode" in src
