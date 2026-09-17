"""The offload port's item gate on big/resident (openfold3_ob0_opt.modes.of3o_gate, OF3O_MIN_TOKENS): below the gate the port's O1 units step aside by
name and the line resolves to the FAST line as composed — the fast hooks, switches, cells, CUDA-graph levers and runner yaml; no `offload` hook, no
OF3O_* export (on this engine the fast line's sampler cells win only captured: CHANGES.md, the port's item gate); at or above it, and with no token
count, the port engages as written. Class contracts only: the kit's own gate constants and grammar."""
import pytest

from openfold3_ob0_opt import modes, report
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
LINE = modes.LINES[("big", "resident")]


def _r(n_tokens, environ=None):
    return modes.resolve("big", HOME, environ=dict(environ or {}), n_tokens=n_tokens)


def test_the_gate_value_is_the_kit_s_per_card_table_or_the_caller_s_knob():
    assert modes.of3o_min_tokens({}) == (modes.OF3O_GATE_DEFAULT, "default")
    for card, n in modes.OF3O_GATE_BY_CARD.items():
        assert modes.of3o_min_tokens({"MODEL_OPT_TARGET_GPU": card}) == (n, f"card:{card}")
        assert modes.of3o_min_tokens({"MODEL_OPT_TARGET_GPU": card.lower()}) == (n, f"card:{card}")
    assert modes.of3o_min_tokens({"OF3O_MIN_TOKENS": "777"}) == (777, "env")
    assert modes.of3o_min_tokens({"OF3O_MIN_TOKENS": "none"}) == (None, "env")
    for bad in ("many", "-1"):
        with pytest.raises(ValueError):
            modes.of3o_min_tokens({"OF3O_MIN_TOKENS": bad})
    assert "OF3O_MIN_TOKENS" in modes.NOT_LEVERS                                   # a caller's knob under big: never a conflict
    assert not _r(400, {"OF3O_MIN_TOKENS": "600"}).conflicts


def test_below_the_gate_the_units_step_aside_and_the_fast_line_runs_as_composed_graphs_included():
    g = modes.OF3O_GATE_DEFAULT
    fast = modes.LINES[("fast", None)]
    r = _r(g - 1)
    assert r.line == "resident" and r.mode == "big"                                   # the same line, gated — never a sub-line
    assert r.of3o_reason == "aside:lt_min" and r.of3o_gate == f"of3o_gate=aside:lt_min min_tokens={g}(default)" and r.of3o_min_tokens == g
    assert modes.OF3O_HOOK not in r.hooks and r.hooks == list(fast.hooks)                # the fast line's chain
    assert not any(k.startswith("OF3O_") for k in r.exports)                             # no port word
    assert not set(r.levers) & set(modes.OF3O_UNIT_LEVERS)                                # the units aside
    f = modes.resolve("fast", HOME, environ={}, n_tokens=g - 1)                          # == a fast call of the same size, word for word:
    assert r.levers == f.levers == [l for l in fast.levers if l in r.levers]              #  every fast lever in the fast line's order,
    assert set(modes.GRAPHS_LEVERS) | {"trunk_graph"} <= set(r.levers)                   #  the CUDA-graph levers included (this engine's sampler cells win only captured),
    assert r.exports == f.exports and sorted(r.unsets) == sorted(f.unsets)                #  every switch as fast spells it, nothing carried from the port,
    assert r.size_gate == f.size_gate and r.gate_reason == f.gate_reason                  #  the graph size gate decided as on fast
    aside = modes.effective_line("big", "resident", {}, g - 1)
    assert aside.hooks == fast.hooks and dict(aside.env) == dict(fast.env) and tuple(aside.unset) == tuple(fast.unset) and aside.levers == fast.levers
    assert aside.runner_yaml and aside.runner_yaml == fast.runner_yaml                    # the stock runner yaml (no attention chunking)
    assert modes.graphed_call("big", "resident", g - 1, {}) is modes.graphed_call("fast", None, g - 1, {}) is True   # the det recipe's graphed flag follows the call
    assert modes.graphed_call("big", "resident", g, {}) is False and modes.graphed_call("big", "resident", None, {}) is False and modes.graphed("big", "resident") is False
    assert modes.graphed_call("big", "resident", g - 1, {"MODEL_OPT_LEVERS_OFF": "cuda_graphs"}) is False   # the ablation door reaches below the gate
    assert modes.effective_line("big", "resident", {}, g) is modes.LINES[("big", "resident")] and modes.effective_line("big", "resident", {}, None) is modes.LINES[("big", "resident")]


def test_at_the_gate_and_without_a_token_count_the_port_engages_as_written():
    g = modes.OF3O_GATE_DEFAULT
    for n, word in ((g, "engaged"), (g + 1000, "engaged"), (None, "n_tok_unknown"), (0, "n_tok_unknown")):
        r = _r(n)
        assert r.of3o_reason == word, (n, r.of3o_gate)
        assert modes.OF3O_HOOK in r.hooks and r.exports.get("OF3O_LAYER") == "1" and set(modes.OF3O_UNIT_LEVERS) & set(r.levers)
    r = _r(400, {"OF3O_MIN_TOKENS": "none"})
    assert r.of3o_reason == "no_gate" and modes.OF3O_HOOK in r.hooks
    r = _r(400, {"OF3O_MIN_TOKENS": "300"})
    assert r.of3o_reason == "engaged" and r.of3o_gate == "of3o_gate=engaged min_tokens=300(env)"


def test_the_other_lines_carry_no_port_gate():
    for key in modes.LINES:
        if modes.OF3O_HOOK not in modes.LINES[key].hooks:
            assert modes.of3o_gate(modes.LINES[key], 400, {}) is None


def test_the_evidence_words():
    g = modes.OF3O_GATE_DEFAULT
    rep = dict(active=True, mode="big", line="resident", levers_requested=[], levers_applied=[], hooks=["cells"], n_tokens=g - 1,
               of3o_gate=f"of3o_gate=aside:lt_min min_tokens={g}(default)", of3o_reason="aside:lt_min", of3o_min_tokens=g, n_gpu=1)
    assert f" of3o_gate=aside:lt_min min_tokens={g}(default) n_tokens={g - 1} " in report.activation_line(dict(rep, line_spelling="x", hooks_spelling="y"))
    by = {l.split("name=")[1].split(" ")[0]: l for l in report.lever_lines(rep)}
    for unit in modes.OF3O_UNIT_LEVERS:
        if unit in LINE.levers:
            assert f"state=off reason=of3o_gate " in by[unit] and f" gate=aside:lt_min n_tokens={g - 1} min_tokens={g}" in by[unit], by[unit]
    rep2 = dict(rep, n_tokens=g, of3o_gate=f"of3o_gate=engaged min_tokens={g}(default)", of3o_reason="engaged", levers_applied=[u for u in modes.OF3O_UNIT_LEVERS if u in LINE.levers])
    by2 = {l.split("name=")[1].split(" ")[0]: l for l in report.lever_lines(rep2)}
    unit = next(u for u in modes.OF3O_UNIT_LEVERS if u in LINE.levers)
    assert "state=on " in by2[unit] and f" gate=engaged n_tokens={g} min_tokens={g}" in by2[unit]
