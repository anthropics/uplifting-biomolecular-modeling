"""The whole-forward memory guard (stack._memory_guard): an out-of-memory inside a forward releases the levers' buffers, moves the shape to
the stock projections by name and reruns the SAME call once, outside the exception handler; a second out-of-memory is the stock model's own
and propagates; a shape already moved twice is not retried. torch and the levers are stand-ins here (no GPU)."""
import sys
import types

import pytest

from gpnstar_opt import stack


class FakeOOM(RuntimeError):
    pass


class Ids:
    def __init__(self, B, L):
        self.shape = (B, L)

    def dim(self):
        return 2

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]


class State:
    clade_species = [()] * 45
    memory_fallbacks = None


class Core:
    def __init__(self):
        self._exact_state = State()


@pytest.fixture()
def fakes(monkeypatch):
    calls = []
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=FakeOOM)
    monkeypatch.setitem(sys.modules, "torch", torch)
    core = Core()
    patches = types.ModuleType("gpnstar_opt.accel.patches")
    patches.core_model = lambda model: core

    def _memory_fallback(st, m_stock, where):
        calls.append((m_stock, where))
        st.memory_fallbacks = dict(st.memory_fallbacks or {})
        st.memory_fallbacks[m_stock] = st.memory_fallbacks.get(m_stock, 0) + 1
    patches._memory_fallback = _memory_fallback
    accel = types.ModuleType("gpnstar_opt.accel"); accel.__path__ = []; accel.patches = patches
    monkeypatch.setitem(sys.modules, "gpnstar_opt.accel", accel)
    monkeypatch.setitem(sys.modules, "gpnstar_opt.accel.patches", patches)
    stack._reset_for_tests()
    return core, calls


def _model(n_fail):
    m = types.SimpleNamespace(runs=0)

    def forward(*args, **kwargs):
        m.runs += 1
        if m.runs <= n_fail:
            raise FakeOOM("CUDA out of memory (stand-in)")
        return "logits"
    m.forward = forward
    return m


def test_one_oom_moves_the_shape_and_reruns_once(fakes):
    core, calls = fakes
    m = _model(1); stack._memory_guard(m)
    assert m.forward(input_ids=Ids(942, 128)) == "logits"
    assert m.runs == 2 and calls == [(942 * 128 * 45, "forward")]
    assert stack._STATE["kv_reason"]["942x128"] == "memory"


def test_second_oom_is_stocks_own_and_propagates(fakes, capsys):
    core, calls = fakes
    m = _model(2); stack._memory_guard(m)
    with pytest.raises(FakeOOM):
        m.forward(input_ids=Ids(942, 128))
    assert m.runs == 2 and len(calls) == 1
    assert "ERROR" in capsys.readouterr().err


def test_a_shape_moved_twice_is_not_retried(fakes):
    core, calls = fakes
    core._exact_state.memory_fallbacks = {942 * 128 * 45: 2}
    m = _model(1); stack._memory_guard(m)
    with pytest.raises(FakeOOM):
        m.forward(input_ids=Ids(942, 128))
    assert m.runs == 1 and calls == []


def test_no_oom_no_change(fakes):
    core, calls = fakes
    m = _model(0); stack._memory_guard(m)
    assert m.forward(input_ids=Ids(8, 128)) == "logits" and m.runs == 1 and calls == []
    stack._memory_guard(m)                      # idempotent: never wrapped twice
    assert getattr(m.forward, "_gpnstar_opt_memory_guard", False)
