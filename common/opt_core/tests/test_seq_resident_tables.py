"""opt_core.seq.resident_tables: build-once, freeze refuses by name, pins name what moved — on duck-typed tensors (no torch)."""
from __future__ import annotations

import pytest

from opt_core.seq import resident_tables as rt


class FakeTensor:
    def __init__(self, ptr, shape, dtype="float32", device="cuda:0"):
        self._ptr, self.shape, self.dtype, self.device = ptr, tuple(shape), dtype, device

    def data_ptr(self):
        return self._ptr


class FakeModule:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class FakeModel:
    def __init__(self, mods):
        self._mods = mods

    def named_modules(self):
        return list(self._mods.items())


def test_build_once_then_same_object():
    calls = []

    def builder(n, dim, device="cuda:0"):
        calls.append((n, dim, device))
        return (FakeTensor(100 + len(calls), (n, dim)), FakeTensor(200 + len(calls), (n, dim)))

    c = rt.ResidentTables(builder, name="rotary")
    a = c.get(1024, 64, device="cuda:0")
    b = c.get(1024, 64, device="cuda:0")
    assert a is b and len(calls) == 1
    c.get(2048, 64, device="cuda:0")
    assert len(calls) == 2 and c.census()["built"] == 2 and c.census()["hits"] == 1
    assert c.line() == "tables=rotary:2built/1hits frozen=0"
    assert c.line_fields() == {"tables": "rotary:2built/1hits", "frozen": "0"}


def test_key_by_shape_not_address():
    built = []
    c = rt.ResidentTables(lambda like: built.append(1) or FakeTensor(1, like.shape), name="pos")
    c.get(FakeTensor(111, (4, 8)))
    c.get(FakeTensor(222, (4, 8)))          # another tensor, same (shape, dtype, device) -> same table
    assert len(built) == 1
    c.get(FakeTensor(333, (4, 9)))
    assert len(built) == 2


def test_freeze_refuses_new_key_by_name_and_serves_old():
    c = rt.ResidentTables(lambda n: FakeTensor(n, (n,)), name="rel_k")
    c.get(1536)
    cen = c.freeze()
    assert cen["frozen"] is True
    assert c.get(1536).shape == (1536,)
    with pytest.raises(rt.PinDrift) as e:
        c.get(4096)
    assert "rel_k" in str(e.value) and "4096" in str(e.value) and "after the freeze" in str(e.value)
    assert c.census()["refused"] == 1
    c.clear()
    assert not c.frozen and c.keys() == []


def test_assert_pinned_names_the_moved_table():
    t = FakeTensor(10, (8, 2))
    c = rt.ResidentTables(lambda: t, name="tab")
    c.get()
    c.assert_pinned()
    t._ptr = 11                              # a .to() / rebuild moved the storage
    with pytest.raises(rt.PinDrift) as e:
        c.assert_pinned()
    assert "tab" in str(e.value) and "moved" in str(e.value)


def test_fingerprint_shapes():
    f = rt.fingerprint((FakeTensor(1, (2, 3)), [FakeTensor(2, (4,))], {"a": FakeTensor(3, (1,))}, 7))
    assert f[0] == "seq" and f[1] == ("tensor", 1, (2, 3), "float32", "cuda:0") and f[4][0] == "object"


def test_lazy_cache_pins_snapshot_prebuild_assert():
    rot0 = FakeModule(_cos_cached=FakeTensor(1000, (512, 32)), _sin_cached=FakeTensor(2000, (512, 32)), _seq_len_cached=512, inv_freq=FakeTensor(5, (32,)))
    rot1 = FakeModule(_cos_cached=None, _sin_cached=None, _seq_len_cached=0)
    lin = FakeModule(weight=FakeTensor(7, (3, 3)))
    model = FakeModel({"blocks.0.attn.rotary": rot0, "blocks.1.attn.rotary": rot1, "blocks.0.ffn": lin})
    ATTRS = ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached")      # the kit names its module's cache attributes
    with pytest.raises(ValueError):
        rt.find_modules(model, ())
    mods = rt.find_modules(model, ATTRS)
    assert [n for n, _ in mods] == ["blocks.0.attn.rotary", "blocks.1.attn.rotary"]
    pins = rt.LazyCachePins(mods, ATTRS, extra=("_seq_len_cached",), name="rotary")
    with pytest.raises(ValueError):                             # min_len without len_attr
        pins.pin(min_len=2048)
    with pytest.raises(rt.PinDrift) as e:                      # block 1 not built to 2048 yet; block 0 only to 512
        pins.pin(min_len=2048, len_attr="_seq_len_cached")
    assert "not built to 2048" in str(e.value)

    def prebuild():                                            # the kit calls the modules' own builder at max shape
        for _, m in mods:
            m._cos_cached, m._sin_cached, m._seq_len_cached = FakeTensor(id(m) + 1, (2048, 32)), FakeTensor(id(m) + 2, (2048, 32)), 2048

    snap = pins.pin(prebuild, min_len=2048, len_attr="_seq_len_cached")
    assert len(snap) == 4 and snap[0]["module"] == "blocks.0.attn.rotary" and snap[0]["attr"] == "_cos_cached" and snap[0]["_seq_len_cached"] == 2048
    pins.assert_pinned()
    pins.assert_pinned()
    assert pins.census() == {"name": "rotary", "modules": 2, "pins": 4, "checks": 2, "pinned": True}
    assert pins.line() == "pins=rotary:4 checks=2" and pins.line_fields() == {"pins": "rotary:4", "checks": "2"}
    rot1._sin_cached = FakeTensor(999, (4096, 32))              # a longer forward reallocated one table
    rot1._seq_len_cached = 4096
    with pytest.raises(rt.PinDrift) as e:
        pins.assert_pinned()
    msg = str(e.value)
    assert "blocks.1.attn.rotary" in msg and "changed" in msg


def test_lazy_cache_pins_refusals():
    with pytest.raises(ValueError) as e:
        rt.LazyCachePins([], ("cos",), name="x")                 # empty watch set: not a pin
    assert "no modules to watch" in str(e.value)
    with pytest.raises(ValueError):
        rt.LazyCachePins([("m", FakeModule(cos=None))], ())
    p = rt.LazyCachePins([("m", FakeModule(cos=None))], ("cos",))
    with pytest.raises(rt.PinDrift):
        p.assert_pinned()                                        # before pin
    with pytest.raises(rt.PinDrift) as e:
        p.pin()
    assert "nothing to pin" in str(e.value)
