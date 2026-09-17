"""RNG discipline: checkpoints restore every listed generator, registration is explicit and refuses by name, the capture contract gate,
offsets_moved, Philox positioning arithmetic and the one-generator cursor form — CPU, no torch (generators and graphs are stand-ins with the
duck-typed surface the module uses: get_state/set_state/get_offset/set_offset/manual_seed, register_generator_state)."""
import pytest

from opt_core.diffusion_loop import rng


class FakeGen:
    """A Philox-like generator: state = (seed, offset); draw(k) advances the offset by k."""

    def __init__(self, seed=0, offset=0):
        self.seed, self.offset = seed, offset

    def manual_seed(self, s):
        self.seed, self.offset = int(s), 0
        return self

    def get_state(self):
        return (self.seed, self.offset)

    def set_state(self, s):
        self.seed, self.offset = s

    def get_offset(self):
        return self.offset

    def set_offset(self, o):
        self.offset = int(o)

    def draw(self, k=4):
        self.offset += k
        return self.offset


class CpuLikeGen(FakeGen):
    """No Philox offset API (like torch's CPU generator)."""
    get_offset = None
    set_offset = None


class FakeGraph:
    def __init__(self):
        self.registered = []

    def register_generator_state(self, g):
        self.registered.append(g)


class OldGraph:
    """A CUDAGraph of a torch without register_generator_state."""


class Mod:
    def __init__(self, name, training, children=()):
        self._name, self.training, self._children = name, training, list(children)

    def named_modules(self):
        yield self._name, self
        for c in self._children:
            yield from c.named_modules()


def test_checkpoint_restores_every_listed_generator():
    led = rng.RngLedger()
    a, b = FakeGen(1, 8), FakeGen(2, 12)
    with rng.generator_checkpoint([a, b], ledger=led) as gens:
        assert gens == [a, b]
        a.draw(); b.draw(); b.draw()                        # warm-up consumption
        assert a.get_offset() == 12
    assert a.get_state() == (1, 8) and b.get_state() == (2, 12)
    assert (led.checkpoints, led.restores) == (1, 1)
    with pytest.raises(RuntimeError):
        with rng.generator_checkpoint([a], ledger=led):
            a.draw()
            raise RuntimeError("warm-up failed")
    assert a.get_state() == (1, 8) and led.restores == 2          # restored on the error path too


def test_registration_is_explicit_and_refuses_by_name():
    led, graph, gens = rng.RngLedger(), FakeGraph(), [FakeGen(1), FakeGen(2)]
    assert rng.register_generators(graph, gens, ledger=led) == gens
    assert graph.registered == gens and led.registered == 2
    with pytest.raises(rng.RngContractUnsupported):
        rng.register_generators(OldGraph(), gens, ledger=led)             # explicit non-default generators: refused by name
    assert led.refusals == 1
    dflt = FakeGen(0)
    

def test_default_generator_is_auto_registered_on_old_torch(monkeypatch):
    led, dflt = rng.RngLedger(), FakeGen(0)
    monkeypatch.setattr(rng, "default_generator", lambda device=None: dflt)
    monkeypatch.setattr(rng, "_is_default_generator", lambda g: g is dflt)
    assert rng.register_generators(OldGraph(), [None, dflt], ledger=led) == [dflt, dflt]
    assert (led.registered, led.registered_default_auto, led.refusals) == (0, 2, 0)
    with pytest.raises(rng.RngContractUnsupported):
        rng.register_generators(OldGraph(), [None, FakeGen(5)], ledger=led)
    graph = FakeGraph()
    assert rng.register_generators(graph, [None], ledger=led) == [dflt] and graph.registered == [dflt] and led.registered == 1


def test_capture_contract_gate():
    led = rng.RngLedger()
    evalnet = Mod("net", False, [Mod("net.block0", False), Mod("net.block1", False)])
    trainnet = Mod("net", False, [Mod("net.block0.dropout", True), Mod("net.block1.dropout", True)])
    g = rng.capture_contract_gate(rng.RNG_NONE, [evalnet], ledger=led)
    assert g.ok and g.details == {"contract": "rng-none", "train_mode_modules": [], "n_train_mode_modules": 0}
    g = rng.capture_contract_gate(rng.RNG_OUTSIDE, [trainnet], ledger=led)
    assert not g.ok and "2 module(s) in training mode (net.block0.dropout, net.block1.dropout)" in g.reason and "declare rng-inside" in g.reason
    g = rng.capture_contract_gate(rng.RNG_INSIDE, [trainnet], ledger=led)
    assert g.ok and g.details["n_train_mode_modules"] == 2
    g = rng.capture_contract_gate("rng-sometimes", [], ledger=led)
    assert not g.ok and g.name == "capture_contract"
    assert led.refusals == 2
    assert rng.train_mode_modules([trainnet], limit=1) == ["net.block0.dropout"]
    bare = type("Bare", (), {"training": True})()
    assert rng.train_mode_modules([bare]) == ["Bare"]               # an object without named_modules is inspected itself


def test_offsets_moved_reports_which_generators_a_callable_advanced():
    a, b, c = FakeGen(1, 0), FakeGen(2, 0), CpuLikeGen(3, 0)
    result, moved = rng.offsets_moved([a, b, c], lambda: (a.draw(4), c.draw(2)))
    assert result == (4, 2)
    assert moved == [(0, ("offset", 0), ("offset", 4)), (2, ("state", b"(3, 0)"), ("state", b"(3, 2)"))]


def test_philox_positioning_arithmetic(monkeypatch):
    made = []

    def fake_positioned(device, seed, offset=0):
        g = FakeGen().manual_seed(seed)
        if offset:
            g.set_offset(offset)
        made.append((device, seed, offset))
        return g

    monkeypatch.setattr(rng, "positioned_generator", fake_positioned)
    led = rng.RngLedger()
    inc = rng.philox_increment(lambda g: g.draw(4) and g.draw(4), "cuda:0", seed=0, ledger=led)   # one 'stock call' = two kernel draws
    assert inc == 8 and led.increments_measured == 1
    gens = rng.fork_at_offsets(12345, [0, 3 * inc, 7 * inc], "cuda:0", ledger=led)
    assert [g.get_state() for g in gens] == [(12345, 0), (12345, 24), (12345, 56)] and led.forks == 3
    g = gens[1]
    for _ in range(10):                                            # a padded replay: 10 calls where the item's stock run makes 6
        g.draw(inc)
    assert rng.reposition(g, 3 * inc, 6, inc, ledger=led) == 24 + 48 == g.get_offset()
    assert led.repositions == 1
    with pytest.raises(rng.RngContractUnsupported):
        rng.reposition(CpuLikeGen(), 0, 1, 4, ledger=led)
    assert led.refusals == 1


def test_stream_cursors_give_each_item_its_own_stream_over_one_generator():
    led = rng.RngLedger()
    shared = FakeGen(9, 100)
    cur = rng.StreamCursors(shared, [(9, 100), (9, 100), (9, 100)], ledger=led)      # the kit supplies each item's start state
    order = [0, 1, 0, 2, 1, 0]                                     # any interleaving
    seen = {0: [], 1: [], 2: []}
    for i in order:
        with cur.item(i) as g:
            assert g is shared
            seen[i].append(g.draw(4))
        assert shared.get_state() == (9, 100)                      # the shared generator's own state is untouched between items
    assert seen == {0: [104, 108, 112], 1: [104, 108], 2: [104]}   # item i gets the i-th... every item consumes ITS stream from 100
    assert cur.states() == [(9, 112), (9, 108), (9, 104)] and len(cur) == 3 and led.cursor_swaps == 6
    cur2 = rng.StreamCursors(shared, [(1, 0), (2, 0)], ledger=led)
    with cur2.item(1) as g:
        g.draw(4)
    assert cur2.states() == [(1, 0), (2, 4)]
    with pytest.raises(ValueError):
        rng.StreamCursors(shared, [])


def test_ledger_fields_and_contract_names():
    assert rng.CONTRACTS == ("rng-none", "rng-outside", "rng-inside")
    assert list(rng.RngLedger().evidence_fields()) == ["checkpoints", "restores", "registered", "registered_default_auto", "forks", "repositions",
                                                       "cursor_swaps", "increments_measured", "refusals"]
    assert isinstance(rng.ledger(), rng.RngLedger)


def test_module_imports_no_torch_at_top_level():
    import sys
    import importlib
    sys.modules.pop("torch", None)
    importlib.reload(rng)
    assert "torch" not in sys.modules
