"""opt_core.seq.varlen: offsets are host arithmetic; device tensors as lengths are refused by type; dense unpad/pad are reshapes."""
from __future__ import annotations

import pytest

from opt_core.seq import varlen as vl


def test_dense_offsets():
    assert vl.dense_offsets(4, 512) == [0, 512, 1024, 1536, 2048]
    assert vl.dense_offsets(0, 7) == [0]
    with pytest.raises(ValueError):
        vl.dense_offsets(-1, 3)


def test_offsets_from_lengths():
    assert vl.offsets_from_lengths([3, 5, 2]) == ([0, 3, 8, 10], 5)
    assert vl.offsets_from_lengths([]) == ([0], 0)
    with pytest.raises(ValueError):
        vl.offsets_from_lengths([1, -2])


def test_device_lengths_refused_by_type():
    class DevTensor:
        device = "cuda:0"

        def data_ptr(self):
            return 1

    with pytest.raises(TypeError) as e:
        vl.offsets_from_lengths([3, DevTensor()])
    assert "host ints" in str(e.value)


class FakeArr:
    """reshape-able stand-in: records the target shape (unpad/pad are pure reshapes)."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def reshape(self, *shape):
        n = 1
        for s in self.shape:
            n *= s
        m = 1
        for s in shape:
            m *= s
        assert n == m, (self.shape, shape)
        return FakeArr(shape)


def test_unpad_pad_dense_are_reshapes():
    q, k = vl.unpad_dense(FakeArr((2, 8, 4, 16)), FakeArr((2, 8, 4, 16)))
    assert q.shape == (16, 4, 16) and k.shape == (16, 4, 16)
    h = vl.pad_dense(FakeArr((16, 64)), 2, 8)
    assert h.shape == (2, 8, 64)
    with pytest.raises(ValueError) as e:
        vl.pad_dense(FakeArr((15, 64)), 2, 8)
    assert "not dense" in str(e.value)


def test_line_reports_what_served():
    for k in vl.COUNTERS:
        vl.COUNTERS[k] = 0
    assert vl.served() == "none" and vl.line_fields()["varlen"] == "none"
    assert set(vl.line_fields()) == {"varlen", "cu_built", "cu_hits", "cu_lengths"}
    vl.COUNTERS["cu_lengths"] = 2
    assert vl.line().startswith("varlen=lengths ")
    vl.COUNTERS["cu_hits"] = 1
    assert vl.served() == "both"
    vl.COUNTERS["cu_lengths"] = 0
    vl.COUNTERS["cu_uncached"] = 3
    assert vl.served() == "dense" and vl.line_fields()["cu_uncached"] == "3"
    for k in vl.COUNTERS:
        vl.COUNTERS[k] = 0


def test_float_length_refused_by_type():
    with pytest.raises(TypeError):
        vl.offsets_from_lengths([3, 7.9])


def test_cu_tensors_cached_per_key():
    torch = pytest.importorskip("torch")
    vl.clear()
    a = vl.dense_cu_seqlens(4, 512, "cpu")
    b = vl.dense_cu_seqlens(4, 512, "cpu")
    assert a is b and a.dtype == torch.int32 and a.tolist() == [0, 512, 1024, 1536, 2048]
    cu, mx = vl.cu_seqlens_from_lengths([3, 5, 2], "cpu")
    assert cu.tolist() == [0, 3, 8, 10] and mx == 5 and cu.dtype == torch.int32
    assert vl.census()["cached"] == 1 and vl.census()["cu_lengths"] == 1          # dense cached, lengths built per call
    vl.MAX_DENSE_SHAPES = 1
    try:
        c = vl.dense_cu_seqlens(2, 8, "cpu")
        assert c.tolist() == [0, 8, 16] and vl.census()["cached"] == 1 and vl.census()["cu_uncached"] == 1
    finally:
        vl.MAX_DENSE_SHAPES = 64
        vl.clear()
