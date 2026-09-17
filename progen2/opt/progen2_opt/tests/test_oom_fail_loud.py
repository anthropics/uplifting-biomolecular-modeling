"""Out of memory on the generation route is never answered by a fallback (the generation kit's `progen2_decode.py` OOM_POLICY).
The kit's one test of the condition (`_oom.py` `is_oom`, pure standard library) and a unit's ONE call (`progen2_decode.unit_call`):
an out-of-memory error from the innermost callable = the NAMED row (`error` 'OUT OF MEMORY at ...; no fallback applied ...', `oom` true),
the callable ran exactly once (no retry, no stock path, no re-install), and every other exception propagates. CPU only; the kit module
is imported by path (torch absent or present)."""
import importlib.util
import os
import sys

import pytest

from .conftest import OPT

SERVING = os.path.join(OPT, "serving", "pipeline_v0_4")
OOM_FILE = os.path.join(SERVING, "_oom.py")


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
def decode_mod():
    if SERVING not in sys.path:
        sys.path.insert(0, SERVING)
    spec = importlib.util.spec_from_file_location("progen2_decode_under_test", os.path.join(SERVING, "progen2_decode.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class _OutOfMemoryError(RuntimeError):
    pass


_OutOfMemoryError.__name__ = "OutOfMemoryError"          # torch's class name (torch.OutOfMemoryError), recognised without importing torch


def test_the_sample_units_one_call_fails_by_name_on_out_of_memory_and_is_never_retried(decode_mod):
    calls = []

    def call():
        calls.append(1)
        raise _OutOfMemoryError("CUDA out of memory. Tried to allocate 2147483648 bytes")

    result, row = decode_mod.unit_call(call, "sample N=8 L=256", torch=None, cuda=False, mem_policy={"fit_info": {}})
    assert result is None and len(calls) == 1                                   # ONE call: no evict-and-retry, no stock path, no re-install
    assert row["oom"] is True and row["error"].startswith("OUT OF MEMORY at sample N=8 L=256: ")
    assert "no fallback applied" in row["error"] and "block" not in row          # an error row, never a substitute result
    assert row["mem_policy"] == {"fit_info": {}}
    assert decode_mod.OOM_POLICY and "never answered by a fallback" in decode_mod.OOM_POLICY


def test_a_cuda_runtime_error_saying_out_of_memory_is_the_same_named_row(decode_mod):
    result, row = decode_mod.unit_call(lambda: (_ for _ in ()).throw(RuntimeError("CUDA error: out of memory")), "ll (len 12)")
    assert result is None and row["oom"] is True and row["error"].startswith("OUT OF MEMORY at ll (len 12)")


def test_every_other_exception_propagates_out_of_the_units_call(decode_mod):
    def call():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        decode_mod.unit_call(call, "sample N=1 L=64")
    with pytest.raises(KeyError):
        decode_mod.unit_call(lambda: {}["missing"], "sample N=1 L=64")


def test_a_result_passes_through_untouched(decode_mod):
    result, row = decode_mod.unit_call(lambda: (["MK"], {"ledger": 1}), "sample N=1 L=64")
    assert result == (["MK"], {"ledger": 1}) and row is None


def test_torch_out_of_memory_classes_take_the_named_row_when_torch_is_present(decode_mod):
    torch = pytest.importorskip("torch")

    def call():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    result, row = decode_mod.unit_call(call, "sample N=2 L=128", torch=torch, cuda=False)
    assert result is None and row["oom"] is True


def test_every_allocation_and_call_goes_through_the_one_test():
    """The kit file's slot allocation and the unit's sample call go through unit_call — every path calls the kit's one test, and a unit that cannot
    fit or whose call ran out raises OutOfMemory by name, never a substitute block (structure, read from the bytes)."""
    src = open(os.path.join(SERVING, "progen2_decode.py"), encoding="utf-8").read()
    assert src.count("from _oom import is_oom") == 1                               # ONE import of the ONE test
    assert "except torch.OutOfMemoryError" not in src and "OutOfMemoryError" not in src   # no second spelling of the test
    assert src.count("unit_call(") == 3                                            # the allocation, the sample call (+ the def)
    for where in ("static K/V allocation for", 'f"sample N={B} L={L}"'):
        assert where in src, where
    assert src.count("raise OutOfMemory(") == 2 and "except RuntimeError" not in src   # nothing on the route catches a RuntimeError but the one test inside unit_call
