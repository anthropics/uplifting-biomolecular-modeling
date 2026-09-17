"""The graphed diffusion step's keep-across-items decision (opt/forward/fast_inference/of3_levers/of3_graphs.py: keep_decision, key_names_differing,
_ReadView, keep_enabled, _read_values_differing, _static_moved, _copy_batch_into) on FAKE capture keys and CPU tensors — no capture, no GPU: a
resident generation is replayed by a later predict item exactly when the two capture keys differ only in feature entries the captured step never
read (and OPENFOLD3_OPT_GRAPHS_KEEP is not 0); a read entry, a step argument, the numerics mode or the kernel flags differing re-captures, by name."""
import importlib.util
import os
import types

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "of3_levers", "of3_graphs.py"))


@pytest.fixture(scope="module")
def G():
    spec = importlib.util.spec_from_file_location("of3_graphs_keep_under_test", SRC)      # a private name: other tests park a stand-in `of3_graphs` in sys.modules
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def key(items, extra=(), mode=("det", False)):
    """a capture key in _make_key's form: ((name, sig)...), extra) + (("mode", ...),)"""
    return (tuple(items), tuple(extra)) + (("mode", mode),)


SIG = lambda *shape: (tuple(shape), "torch.float32", 0)     # noqa: E731
BASE = [("xl", SIG(1, 5, 3256, 3)), ("t", SIG()), ("si", SIG(1, 1, 400, 449)), ("s", SIG(1, 1, 400, 384)), ("z", SIG(1, 1, 400, 400, 128)),
        ("token_mask", SIG(1, 400)), ("num_atoms_per_token", SIG(1, 400)), ("ref_pos", SIG(1, 3256, 3)), ("msa", SIG(1, 2011, 400, 34)), ("template_all_atom_positions", SIG(1, 4, 400, 24, 3))]
READ = frozenset({"token_mask", "num_atoms_per_token", "ref_pos", "atom_mask"})


def with_(name, sig, items=BASE):
    return [(k, (sig if k == name else v)) for k, v in items]


def test_identical_key_is_a_hit(G):
    assert G.keep_decision(key(BASE), READ, key(BASE)) == ("hit", [])
    assert G.keep_decision(key(BASE), None, key(BASE)) == ("hit", [])
    assert G.keep_decision(key(BASE), READ, key(BASE), keep=False) == ("hit", [])


def test_unread_entry_of_another_shape_keeps(G):
    new = key(with_("msa", SIG(1, 2010, 400, 34)))                                   # the MSA depth differs by one row: the step never reads the MSA
    assert G.keep_decision(key(BASE), READ, new) == ("keep", ["msa"])
    new2 = key(with_("template_all_atom_positions", SIG(1, 1, 400, 24, 3), with_("msa", SIG(1, 9, 400, 34))))
    assert G.keep_decision(key(BASE), READ, new2) == ("keep", ["msa", "template_all_atom_positions"])


def test_keep_off_drops_on_any_difference(G):
    new = key(with_("msa", SIG(1, 2010, 400, 34)))
    assert G.keep_decision(key(BASE), READ, new, keep=False) == ("drop", ["msa"])


def test_read_entry_of_another_shape_drops(G):
    new = key(with_("ref_pos", SIG(1, 3300, 3)))
    assert G.keep_decision(key(BASE), READ, new) == ("drop", ["ref_pos"])
    both = key(with_("ref_pos", SIG(1, 3300, 3), with_("msa", SIG(1, 7, 400, 34))))  # read names first, then the unread ones
    assert G.keep_decision(key(BASE), READ, both) == ("drop", ["ref_pos", "msa"])


def test_unknown_read_set_counts_every_entry_as_read(G):
    new = key(with_("msa", SIG(1, 2010, 400, 34)))
    assert G.keep_decision(key(BASE), None, new) == ("drop", ["msa"])                 # read_all (a whole-dict access during the warm-up): the full signature decides, as before


def test_step_arguments_mode_and_flags_always_drop(G):
    assert G.keep_decision(key(BASE), READ, key(with_("xl", SIG(1, 5, 3300, 3)))) == ("drop", ["xl"])       # atom count (the noisy coordinates)
    assert G.keep_decision(key(BASE), READ, key(with_("z", SIG(1, 1, 401, 401, 128)))) == ("drop", ["z"])   # token count (the pair embedding)
    assert G.keep_decision(key(BASE), READ, key(BASE, mode=("det", True))) == ("drop", ["mode"])             # numerics mode (deterministic / matmul precision / autocast ...)
    assert G.keep_decision(key(BASE), READ, key(BASE, extra=(("use_cueq_triangle_kernels", True),))) == ("drop", ["extra"])   # kernel flags


def test_entry_present_in_one_item_only(G):
    fewer = key([kv for kv in BASE if kv[0] != "msa"])
    assert G.keep_decision(key(BASE), READ, fewer) == ("keep", ["msa"])                # an unread entry absent from the new item: irrelevant
    fewer_read = key([kv for kv in BASE if kv[0] != "ref_pos"])
    assert G.keep_decision(key(BASE), READ, fewer_read) == ("drop", ["ref_pos"])       # a read entry absent (the step's .get default would now apply): re-capture
    assert G.keep_decision(fewer_read, READ | {"ref_pos"}, key(BASE)) == ("drop", ["ref_pos"])   # ... and a read-by-.get entry that appears: re-capture


def test_read_view_records_names_and_whole_dict_access(G):
    v = G._ReadView({"a": 1, "b": 2, "c": 3})
    assert v["a"] == 1 and v.get("b") == 2 and v.get("zz", 7) == 7
    assert v.read_names() == frozenset({"a", "b", "zz"})
    for whole in (lambda v: list(v), lambda v: v.keys(), lambda v: v.items(), lambda v: v.values(), lambda v: dict(v), lambda v: dict(**v), lambda v: v.copy()):
        v = G._ReadView({"a": 1, "b": 2}); v["a"]; whole(v)
        assert v.read_names() is None                                                # every entry counts as read
    v = G._ReadView({"a": 1}); "a" in v                                              # a membership test reads no value
    assert v.read_names() == frozenset()


def test_keep_env_words(G, monkeypatch):
    for word, want in (("", True), ("1", True), ("on", True), ("0", False), ("off", False), ("False", False), ("nonsense", True)):
        monkeypatch.setenv(G.KEEP_ENV, word)
        assert G.keep_enabled() is want, word
    monkeypatch.delenv(G.KEEP_ENV, raising=False)
    assert G.keep_enabled() is True                                                  # default: keep across items
    assert G.keep_enabled({G.KEEP_ENV: "0"}) is False
    assert G.KEEP_ENV == "OPENFOLD3_OPT_GRAPHS_KEEP"


def _fake_gen(G):
    g = G._Gen(key(BASE))
    g.static_batch = {"token_mask": torch.ones(1, 4), "ref_pos": torch.zeros(1, 9, 3), "n_chunks": 3, "flavour": "plain", "opts": (1, 2)}
    g.static_args = {"xl_noisy": torch.zeros(1, 5, 9, 3), "t": torch.zeros(()), "si_input": torch.zeros(1, 1, 4, 8), "si_trunk": torch.zeros(1, 1, 4, 8), "zij_trunk": torch.zeros(1, 1, 4, 4, 2)}
    g.static_out = torch.zeros(1, 5, 9, 3)
    g.static_ptrs = G._static_ptrs(g)
    return g


def test_read_python_values_compare_by_value(G):
    g = _fake_gen(G)
    same = {"token_mask": torch.ones(1, 4), "ref_pos": torch.ones(1, 9, 3), "n_chunks": 3, "flavour": "plain", "opts": (1, 2), "msa": torch.zeros(2, 2)}
    assert G._read_values_differing(g, same) == []
    other = dict(same, n_chunks=4)
    assert G._read_values_differing(g, other) == ["n_chunks"]
    gone = {k: v for k, v in same.items() if k != "flavour"}
    assert G._read_values_differing(g, gone) == ["flavour"]
    g.static_batch["obj"] = types.SimpleNamespace(x=1)                               # an arbitrary object: only the same object counts as equal
    assert G._read_values_differing(g, dict(same, obj=types.SimpleNamespace(x=1))) == ["obj"]
    assert G._read_values_differing(g, dict(same, obj=g.static_batch["obj"])) == []


def test_static_buffers_must_keep_their_addresses(G):
    g = _fake_gen(G)
    assert G._static_moved(g) == []
    g.static_batch["ref_pos"] = torch.zeros(1, 9, 3)                                 # re-allocated: another address -> named
    assert G._static_moved(g) == ["feat:ref_pos"]


def test_static_refresh_copies_values_and_refuses_another_shape(G):
    g = _fake_gen(G)
    live = {"token_mask": torch.full((1, 4), 5.0), "msa": torch.zeros(3, 3)}          # msa is not a static entry: ignored
    G._copy_batch_into(g.static_batch, live)
    assert torch.equal(g.static_batch["token_mask"], torch.full((1, 4), 5.0)) and g.static_batch["token_mask"].data_ptr() == g.static_ptrs[("feat", "token_mask")]
    with pytest.raises(RuntimeError, match="static feature 'token_mask'"):
        G._copy_batch_into(g.static_batch, {"token_mask": torch.ones(1, 5)})


def test_key_names_differing_orders_items_then_extra_then_mode(G):
    a = key(BASE, extra=(("chunk_size", None),), mode=("det", False))
    b = key(with_("msa", SIG(2,)), extra=(("chunk_size", 4),), mode=("det", True))
    assert G.key_names_differing(a, b) == ["msa", "extra", "mode"]
    assert G.key_names_differing(a, a) == []
