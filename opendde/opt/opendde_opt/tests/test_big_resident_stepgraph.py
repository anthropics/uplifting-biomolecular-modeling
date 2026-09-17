"""The big line per input size: below the offload size gate
(`MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`, 1,400 residue tokens on the 80 GB cards) the line IS fast's resident lever set minus `modes.BIG_DROP`
— and the sampler step graph `stepgraph` rides it up to the big word row (`stepgraph.WORD_ROWS["big"]`: 400 tokens, the size up to
which graph replay is ahead of the eager step under the big words; above it the lever is composed out by name,
`word_gate:big_graph_not_ahead_above:400`); at and above the offload gate the unit is installed and the graph leaves with it
(`BIG_KIT_ROWS_OFF`, `stepgraph.compose`'s `offload_line` word as belt and braces); the sm_80 card row caps the graph under big
exactly as under fast (1,024 tokens). The composition
is driven the way `cli.pred` drives it: resolve, then `big.plan` / `smalln.plan` / `chunklift.plan` / `stepgraph.plan` on a query of ONE
item of N residues, then resolve again (what `stack.activate` composes). CPU only; no upstream needed."""
import json
import os

import pytest

import opt_core.mem
from opendde_opt import big, chunklift, modes, smalln, stepgraph
from opendde_opt.tests import _stubs

pytestmark = pytest.mark.skipif(not hasattr(opt_core.mem, "register"),
                                reason="core_missing: opt_core.mem carries no lever registry (opt_core < 0.4.0) — every big line refuses by name on this core")

H100_ENV = {"MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS": "1400", "MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS": "2565", "MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS": "300"}   # configs/h100.env
SIZES = (17, 200, 201, 256, 400, 401, 448, 512, 513, 600, 800, 1024, 1025, 1399, 1400, 2048, 2565, 4096)
GATE = 1400                                                                                              # configs/h100.env MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS
WCAP, WWORD = stepgraph.WORD_ROWS["big"]                                                               # (400, "big_graph_not_ahead")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("ODDE_", "PYTORCH_CUDA", "OPENDDE_BIG", "MODEL_OPT_BIG", "MODEL_OPT_SMALL", "MODEL_OPT_LEVERS_OFF")):
            monkeypatch.delenv(k, raising=False)
    for k, v in H100_ENV.items():
        monkeypatch.setenv(k, v)
    big._reset(); smalln._reset(); chunklift._reset(); stepgraph._reset()
    yield
    big._reset(); smalln._reset(); chunklift._reset(); stepgraph._reset()


def _compose(word, n, tmp_path, sm="90"):
    """The line `cli.pred` activates for `--mode <word>` on a query of one item of `n` residues (H100 deployment parameters; `sm` = the card's
    compute capability word as stepgraph._card_sm reads it: "90" H100 — no CARD_ROWS row, the general gate only — or "80" A100)."""
    big._reset(); smalln._reset(); chunklift._reset(); stepgraph._reset()
    stepgraph._CARD.update(sm=sm, read=True)                                                             # after _reset (which forgets the card): this process's card, not the test box's
    q = tmp_path / f"q_{word}_{n}.json"
    q.write_text(json.dumps([{"name": f"n{n}", "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1, "unpairedMsaPath": "/x.a3m"}}]}]))
    env = dict(os.environ)
    res0 = modes.resolve(word, _stubs.TREE, env)
    big.plan(res0, str(q), env); smalln.plan(res0, str(q), env); chunklift.plan(res0, str(q)); stepgraph.plan(res0, str(q))
    return modes.resolve(word, _stubs.TREE, env).line


@pytest.mark.parametrize("n", SIZES)
def test_big_below_the_offload_gate_is_fasts_resident_set_with_the_step_graph(n, tmp_path):
    b = _compose("big", n, tmp_path)
    wb, gb = stepgraph.word(b), stepgraph.gated_off(b)                                                  # this call's plan (the next _compose re-plans)
    f = _compose("fast", n, tmp_path)
    assert b.name == "BIG_F" and f.name == "LSTAR2A" and b.allocator == modes.EXPANDABLE
    assert not set(modes.BIG_DROP) & set(b.levers)                                                    # alloc_auto, zprep_hoist, dit_fused, dit_lowp: off every big line at every size
    assert not {sw for x in modes.BIG_DROP for sw in modes.LEVER_SWITCHES.get(x, ())} & set(b.exports)  # with their switches
    offload = [x for x in b.levers if x in modes._OFFLOAD_LEVERS]
    if n < GATE:                                                                                         # the resident line: fast's own composition at this size minus BIG_DROP, in fast's order
        assert not offload and "sample_chunk" not in b.levers and b.path_order == f.path_order == modes.KIT_PATH_ORDER
        assert "stepgraph" in f.levers
        if n <= WCAP:                                                                                    # … with the sampler step graph up to the big word's row
            assert list(b.levers) == [x for x in f.levers if x not in modes.BIG_DROP], (n, b.levers, f.levers)
            assert "stepgraph" in b.levers and wb is None and gb == {}
        else:                                                                                            # … without it above the row, by that word's name
            assert list(b.levers) == [x for x in f.levers if x not in modes.BIG_DROP + ("stepgraph",)], (n, b.levers, f.levers)
            assert wb == f"word_gate:{WWORD}_above:{WCAP}" and gb == {"stepgraph": f"word_gate:{WWORD}_above:{WCAP}"}
        assert ("chunk_lift" in b.levers) == ("chunk_lift" in f.levers) and ("keep_pool" in b.levers) == ("keep_pool" in f.levers)
        want = {k: v for k, v in f.exports.items() if not any(k in modes.LEVER_SWITCHES.get(x, ()) for x in modes.BIG_DROP)}
        want.update({k: v for k, v in modes.BIG_WORD_SWITCHES.items() if k in want})                    # the providers bound by their own big tier word
        assert {k: v for k, v in b.exports.items() if k != modes.ALLOCATOR} == want, n                     # fast's switches exactly, minus the dropped levers', at the big words
    else:                                                                                                # the offload line: the unit installed, the graph off with it (no CUDA graph over streamed pair tensors)
        assert set(offload) == set(modes._OFFLOAD_LEVERS) and "sample_chunk" in b.levers and b.path_order == modes.OFFLOAD_PATH_ORDER
        assert "stepgraph" not in b.levers and "chunk_lift" not in b.levers and "keep_pool" not in b.levers and "tmpl_dedup" not in b.levers
        assert wb == "offload_line" and gb == {}                                                         # the offload line's LEVER row reads state=off with no gate word, as before
        assert b.exports["ODDE_OFFLOAD"] == "all"
        assert ("no_dit_hoist" in b.levers) == (n >= 2565) and ({"dit_hoist", "dit_align"} <= set(b.levers)) == (n < 2565)


@pytest.mark.parametrize("n", (200, 800, 1024, 1025, 1160, 1399))
def test_the_sm80_card_row_caps_the_graph_under_big_as_under_fast(n, tmp_path):
    cap, word = stepgraph.CARD_ROWS["80"]                                                                # A100: (1024, sm80_replay_not_bit_exact) — the card's row is asked before the word's
    b = _compose("big", n, tmp_path, sm="80")
    wb = stepgraph.word(b)
    f = _compose("fast", n, tmp_path, sm="80")
    assert ("stepgraph" in f.levers) is (n <= cap) and ("stepgraph" in b.levers) is (n <= min(cap, WCAP)), (n, b.levers)
    if n > cap:
        assert wb == f"card_gate:{word}_above:{cap}" == stepgraph.word(f)                                # the card's row is asked before the word's
    elif n > WCAP:
        assert wb == f"word_gate:{WWORD}_above:{WCAP}" and stepgraph.word(f) is None
    else:
        assert wb is None and stepgraph.word(f) is None


def test_word_rows_and_policy_words(tmp_path, monkeypatch):
    assert stepgraph.WORD_ROWS == {"big": (400, "big_graph_not_ahead")} and stepgraph.SAMPLER_WORD_SWITCH == "ODDE_DIT_ATTN" and modes.BIG_WORD_SWITCHES[stepgraph.SAMPLER_WORD_SWITCH] == modes.SAMPLER_BIG_WORD == "big"
    fast, bigf, s1 = modes.LINES["LSTAR2A"], modes.LINES["BIG_F"], modes.LINES["S1"]
    assert stepgraph.word_row(fast) is None and stepgraph.word_row(s1) is None and stepgraph.word_row(None) is None
    assert stepgraph.word_row(bigf) == (400, "big_graph_not_ahead", "big") == stepgraph.word_row(modes.LINES["BIG_TP"])
    toks = lambda *cs: {f"i{i}": {"residue_tokens": c, "ligands_uncounted": 0} for i, c in enumerate(cs)}
    p = stepgraph.policy(toks(200, 400), stepgraph.word_row(bigf))
    assert p["within"] and p["case"] == "within" and p["word_row"] == {"sampler_word": "big", "cap": 400, "word": "big_graph_not_ahead"} and "big word row <= 400" in p["reason"]
    p = stepgraph.policy(toks(200, 401), stepgraph.word_row(bigf))
    assert not p["within"] and p["case"] == "word_above" and "401 residue tokens > 400: big_graph_not_ahead" in p["reason"]
    monkeypatch.setitem(stepgraph.WORD_ROWS, "big", (200, "a_measured_row"))                             # the mechanism: a row is (cap, word) per sampler tier word
    assert stepgraph.word_row(bigf) == (200, "a_measured_row", "big") == stepgraph.word_row(modes.LINES["BIG_TP"]) and stepgraph.word_row(fast) is None
    p = stepgraph.policy(toks(200, 180), stepgraph.word_row(bigf))
    assert p["within"] and p["case"] == "within" and p["word_row"] == {"sampler_word": "big", "cap": 200, "word": "a_measured_row"} and "big word row <= 200" in p["reason"]
    p = stepgraph.policy(toks(200, 201), stepgraph.word_row(bigf))
    assert not p["within"] and p["case"] == "word_above" and "201 residue tokens > 200: a_measured_row" in p["reason"]
    assert stepgraph.policy(toks(200, 201), None)["within"] and stepgraph.policy(toks(5000), None)["within"]   # no row: the general gate only (GATE_MAX 99999)
    assert stepgraph.policy(None, stepgraph.word_row(bigf))["within"]                                         # no query read: composed in, the call's own admission decides
    b = _compose("big", 201, tmp_path)                                                                 # a row in the table composes the lever out by name above its cap, census reason word_gate:
    assert "stepgraph" not in b.levers and stepgraph.word(b) == "word_gate:a_measured_row_above:200" and stepgraph.gated_off(b) == {"stepgraph": "word_gate:a_measured_row_above:200"}


def test_the_static_rows_and_the_row_sharded_line_carry_no_graph():
    assert "stepgraph" not in modes.BIG_DROP and "stepgraph" in modes.BIG_KIT_ROWS_OFF["pair_offload"] and "stepgraph" in modes.BIG_TP_DROP
    assert "stepgraph" not in modes.LINES["BIG_F"].levers                                              # the static row = every gate on: the offload line
    assert "stepgraph" not in modes.LINES["BIG_TP"].levers                                             # --n_gpu P>1: the rank processes' sampler is not graphed
    assert "stepgraph" in modes.big_line("BIG_F", ()).levers and "stepgraph" in modes.big_line("BIG_F", ("no_dit_hoist",)).levers
    assert "stepgraph" not in modes.big_line("BIG_F", ("pair_offload", "sample_chunk")).levers
    assert "stepgraph" not in modes.big_line("BIG_TP", ("sample_chunk",)).levers
    assert modes.LEVER_SWITCHES["stepgraph"] == ()                                                       # no switch: the lever's presence changes the line's lever list only, never its exports


@pytest.mark.parametrize("word", ("exact", "fast"))
@pytest.mark.parametrize("n", (200, 400, 1400))
def test_exact_and_fast_keep_the_graph_at_every_size_on_h100(word, n, tmp_path):
    ln = _compose(word, n, tmp_path)
    assert "stepgraph" in ln.levers and stepgraph.word(ln) is None and {"zprep_hoist", "alloc_auto"} <= set(ln.levers)
    assert ({"dit_fused", "dit_lowp"} <= set(ln.levers)) == (word == "fast")
