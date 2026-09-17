"""Out of memory on the served path propagates; no fallback is applied. The kit's one test of the condition
(`_oom.py` `is_oom`, pure standard library) and the wrapper's capture route (`_wrap._GraphDispatch._graph_for`): an out-of-memory error raised by
the innermost capture callable re-raises (no `capture_refused`, no eager fused path), while a capture failure that is NOT out-of-memory keeps
the named `capture_refused` route. The wrapper cases need torch + numpy importable (CPU is enough) and skip by name otherwise."""
import importlib
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                 # flashzoi/
KITS_ROOT = os.path.join(TREE, "opt", "forward", "kits_v1_25")
KIT_DIR = os.path.join(KITS_ROOT, "engines", "flashzoi", "kits", "v1_25")
OOM_FILE = os.path.join(KIT_DIR, "_oom.py")


def _load_oom():
    spec = importlib.util.spec_from_file_location("kit_oom_under_test", OOM_FILE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_is_oom_recognises_the_out_of_memory_forms_without_torch():
    is_oom = _load_oom().is_oom

    class OutOfMemoryError(RuntimeError):      # the class name torch uses; recognised by name when torch is not imported
        pass

    class XlaRuntimeError(RuntimeError):
        pass

    assert is_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 2147483648 bytes"))
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert is_oom(MemoryError())
    assert is_oom(XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate"))
    assert not is_oom(RuntimeError("CUDA error: operation not permitted when stream is capturing"))
    assert not is_oom(RuntimeError("INTERNAL ASSERT FAILED"))
    assert not is_oom(ValueError("out of memory"))            # the text on a non-RuntimeError is not an allocation failure
    assert not is_oom(KeyError("x"))
    try:                                                       # one level of __cause__
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError as inner:
            raise KeyError("wrapped") from inner
    except KeyError as outer:
        assert is_oom(outer)
    try:                                                       # one level of __context__
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError:
            raise RuntimeError("raised while handling")
    except RuntimeError as outer:
        assert is_oom(outer)


def test_is_oom_recognises_torch_classes_by_isinstance_when_torch_is_present():
    torch = pytest.importorskip("torch")
    is_oom = _load_oom().is_oom
    assert is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    if hasattr(torch, "OutOfMemoryError"):
        assert is_oom(torch.OutOfMemoryError("CUDA out of memory"))
    assert not is_oom(torch.jit.Error("x")) if hasattr(torch.jit, "Error") else True


@pytest.fixture
def wrap():
    """`_wrap` imported as a submodule of a bare package rooted at the kit dir (its relative imports `_pins`, `_oom` resolve; the kit's
    `__init__` — background threads, JIT cache — is not executed)."""
    pytest.importorskip("torch"); pytest.importorskip("numpy")
    name = "fzkit_under_test"
    if name + "._wrap" in sys.modules:
        return sys.modules[name + "._wrap"]
    pkg = types.ModuleType(name); pkg.__path__ = [KIT_DIR]; sys.modules[name] = pkg
    if KITS_ROOT not in sys.path:
        sys.path.insert(0, KITS_ROOT)
    return importlib.import_module(name + "._wrap")


def _graphed_with(wrap, capture):
    """A _GraphDispatch-like object whose innermost capture callable is `capture` (a batch shape captures at its first call)."""
    import torch
    head = types.SimpleNamespace(weight=torch.zeros(3, 2, 1))
    runner = types.SimpleNamespace(capture_refused={}, counts={"lazy_captures": 0}, graphs={}, graph_keys={}, graph=None, model=types.SimpleNamespace(human_head=head))
    g = wrap._GraphDispatch.__new__(wrap._GraphDispatch)
    g.__dict__.update(runner=runner, r=runner, device="cpu")
    g._capture_now = capture
    return g, runner


def _call_graph_for(wrap, g, b):
    import inspect
    fn = wrap._GraphDispatch._graph_for
    params = inspect.signature(fn).parameters
    kw = {"force": False} if "force" in params else {}
    return fn(g, b, **kw)


def test_out_of_memory_during_a_capture_propagates_and_no_eager_route_is_armed(wrap):
    import torch

    def capture(b):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2147483648 bytes")

    g, runner = _graphed_with(wrap, capture)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _call_graph_for(wrap, g, 4)
    assert runner.capture_refused == {}                    # no capture_refused -> no eager fused fallback for this batch


def test_a_cuda_runtime_error_saying_out_of_memory_propagates_too(wrap):
    def capture(b):
        raise RuntimeError("CUDA error: out of memory")

    g, runner = _graphed_with(wrap, capture)
    with pytest.raises(RuntimeError, match="out of memory"):
        _call_graph_for(wrap, g, 4)
    assert runner.capture_refused == {}


def test_a_capture_failure_that_is_not_out_of_memory_keeps_the_named_capture_refused_route(wrap):
    def capture(b):
        raise RuntimeError("CUDA error: operation not permitted when stream is capturing")

    g, runner = _graphed_with(wrap, capture)
    assert _call_graph_for(wrap, g, 4) is None             # None = the eager fused path for this call
    assert 4 in runner.capture_refused and "capture refused at batch 4" in runner.capture_refused[4]
