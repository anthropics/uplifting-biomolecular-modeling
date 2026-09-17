"""Small batches: B·L below stack.DEDUP_MIN_TOKENS is routed to unified K/V by rule through the levers' per-shape route table
(reason=small_batch on the KV line); larger batches are left to the self-check. A fake state object stands in for the levers' state."""
import sys
import types

from gpnstar_opt import stack


class FakeIds:
    def __init__(self, B, L):
        self.shape = (B, L, 1)

    def dim(self):
        return 3


class FakeState:
    def __init__(self):
        self.dedup = True
        self.clade_species = [[0]] * 45
        self.mode_override = {}


class FakeModel:
    pass


def _install_fake_core(monkeypatch, state):
    mod = types.ModuleType("gpnstar_opt.accel.patches")
    model = FakeModel()
    model._exact_state = state
    mod.core_model = lambda m: model
    monkeypatch.setitem(sys.modules, "gpnstar_opt.accel.patches", mod)
    import gpnstar_opt.accel as accel_pkg
    monkeypatch.setattr(accel_pkg, "patches", mod, raising=False)
    return model


def test_small_batch_routes_to_unified_kv_by_rule(monkeypatch):
    stack._reset_for_tests()
    state = FakeState()
    model = _install_fake_core(monkeypatch, state)
    stack._pre_hook(model, (), {"input_ids": FakeIds(1, 128)})
    assert state.mode_override == {1 * 128 * 45: "p3b"}
    assert stack._STATE["kv_reason"]["1x128"] == "small_batch" and "1x128" in stack._STATE["kv_started"]


def test_a_batch_at_the_floor_is_left_to_the_kv_check(monkeypatch):
    stack._reset_for_tests()
    state = FakeState()
    model = _install_fake_core(monkeypatch, state)
    B = max(1, stack.DEDUP_MIN_TOKENS // 128)
    stack._pre_hook(model, (), {"input_ids": FakeIds(B, 128)})
    assert state.mode_override == {} and "small_batch" not in stack._STATE["kv_reason"].values()


def test_a_model_without_dedup_is_not_rerouted(monkeypatch):
    stack._reset_for_tests()
    state = FakeState()
    state.dedup = False
    model = _install_fake_core(monkeypatch, state)
    stack._pre_hook(model, (), {"input_ids": FakeIds(1, 128)})
    assert state.mode_override == {}
