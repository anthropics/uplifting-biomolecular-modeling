"""The CUDA-graph size gate (modes.graphs_gate / resolve(n_tokens=...), the words of opt_core.mem.graph_gate): a graphed line captures graphs
only for a query of at most N polymer tokens when a cap N is given (OPENFOLD3_OPT_GRAPHS_MAX_TOKENS / --graphs-max-tokens: unset|always = no cap,
the default — available, not the default; 0|off = never capture); where the decision is eager/off the graph switches are not exported, the graph
levers not requested, and the report carries the decision (`graph=capture|eager:n_tok>N|off`, `gate=<reason>`); lines without graphs are
untouched and a cap with no token count is named (`gate=n_tok_unknown`)."""
import os

import pytest

from openfold3_opt import inputs, modes

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OPT",))}
CAP400 = {**CLEAN, modes.ENV_GRAPHS_MAX_TOKENS: "400"}


def test_gate_decisions():
    lv = modes.LINES[("fast", None)].levers
    assert modes.graphs_gate(lv, None, 400) is None                                                   # a cap needs a token count: the line as written (named by resolve)
    d = modes.graphs_gate(lv, 10_000, None)
    assert d.capture and d.reason == "no_cap" and d.fragment == "graph=capture"
    assert modes.graphs_gate(lv, None, None).capture                                                 # no cap: capture regardless of the count
    assert modes.graphs_gate(lv, 400, 400).capture and modes.graphs_gate(lv, 400, 400).reason == "within_cap"
    d = modes.graphs_gate(lv, 401, 400)
    assert not d.capture and d.reason == "n_tok>cap" and d.fragment == "graph=eager:n_tok>400"
    d = modes.graphs_gate(lv, 10, 0)
    assert not d.capture and d.reason == "off" and d.fragment == "graph=off"                          # 0 = never capture
    assert modes.graphs_gate(modes.LINES[("big", "resident")].levers, 5000, 400) is None           # no graphs on the line: nothing to gate
    d = modes.graphs_gate(modes.LINES[("exact", "cueq")].levers, 5000, 400)                          # the exact line carries graphs: gated like fast
    assert not d.capture and d.reason == "n_tok>cap"


def test_cap_from_the_environment():
    assert modes.graphs_cap({}) is None and modes.GRAPHS_MAX_TOKENS is None                          # no cap by default
    assert modes.graphs_cap({modes.ENV_GRAPHS_MAX_TOKENS: "always"}) is None and modes.graphs_cap({modes.ENV_GRAPHS_MAX_TOKENS: "off"}) == 0
    assert modes.graphs_cap({modes.ENV_GRAPHS_MAX_TOKENS: "0"}) == 0 and modes.graphs_cap({modes.ENV_GRAPHS_MAX_TOKENS: " 750 "}) == 750   # 0 = never capture (opt_core.mem.graph_gate.parse_cap)
    for bad in ("-1", "many", "4e2"):
        with pytest.raises(ValueError):
            modes.graphs_cap({modes.ENV_GRAPHS_MAX_TOKENS: bad})


def test_resolve_applies_the_gate():
    for key in (("fast", None),):
        mode, line = key
        kw = {}
        below = modes.resolve(mode, HOME, environ=CAP400, n_tokens=180, **kw)
        assert below.exports["OF3_CUDA_GRAPHS"] == "1" and below.exports["OF3_GRAPHS_STRICT"] == "1" and "cuda_graphs" in below.levers
        assert (below.size_gate, below.gate_reason, below.n_tokens) == ("graph=capture", "within_cap", 180)
        above = modes.resolve(mode, HOME, environ=CAP400, n_tokens=1201, **kw)
        assert "OF3_CUDA_GRAPHS" not in above.exports and "OF3_GRAPHS_STRICT" not in above.exports
        assert set(modes.GRAPHS_ENVS) <= set(above.unsets) and not set(modes.GRAPHS_LEVERS) & set(above.levers)
        assert (above.size_gate, above.gate_reason) == ("graph=eager:n_tok>400", "n_tok>cap") and not above.conflicts
        assert [l for l in below.levers if l not in modes.GRAPHS_LEVERS] == above.levers                # everything else of the line unchanged
        assert {k: v for k, v in below.exports.items() if k not in modes.GRAPHS_ENVS} == above.exports
        untold = modes.resolve(mode, HOME, environ=CAP400, **kw)                                     # a cap but no token count (warm, unreadable query): the line as written, named
        assert untold.exports == below.exports and (untold.size_gate, untold.gate_reason) == ("graph=capture", "n_tok_unknown")
        never = modes.resolve(mode, HOME, environ={**CLEAN, modes.ENV_GRAPHS_MAX_TOKENS: "off"}, n_tokens=50, **kw)
        assert never.exports == above.exports and (never.size_gate, never.gate_reason) == ("graph=off", "off")
        default = modes.resolve(mode, HOME, environ=CLEAN, n_tokens=5000, **kw)                       # the default: no cap, the tested line
        assert default.exports == below.exports and (default.size_gate, default.gate_reason) == ("graph=capture", "no_cap")
    big = modes.resolve("big", HOME, environ=CAP400, n_tokens=5000)
    assert big.size_gate is None and big.gate_reason is None
    ex = modes.resolve("exact", HOME, environ=CAP400, n_tokens=5000)                                 # the exact line above a cap: eager, named
    assert (ex.size_gate, ex.gate_reason) == ("graph=eager:n_tok>400", "n_tok>cap") and "OF3_CUDA_GRAPHS" not in ex.exports
    own = modes.resolve("exact", HOME, environ=CLEAN, n_tokens=400)                                   # no cap in the environment: the line's OWN cap (512) applies
    assert own.exports["OF3_CUDA_GRAPHS"] == "1" and own.exports["OF3_GRAPHS_STRICT"] == "1" and (own.size_gate, own.gate_reason) == ("graph=capture", "within_cap")
    above_own = modes.resolve("exact", HOME, environ=CLEAN, n_tokens=513)
    assert "OF3_CUDA_GRAPHS" not in above_own.exports and (above_own.size_gate, above_own.gate_reason) == ("graph=eager:n_tok>512", "n_tok>cap") and not above_own.conflicts
    untold = modes.resolve("exact", HOME, environ=CLEAN)                                              # no token count on a line capped by default: eager, fail-closed, named
    assert "OF3_CUDA_GRAPHS" not in untold.exports and (untold.size_gate, untold.gate_reason) == ("graph=eager:n_tok_unknown", "n_tok_unknown") and "cuda_graphs" not in untold.levers
    lifted = modes.resolve("exact", HOME, environ={**CLEAN, modes.ENV_GRAPHS_MAX_TOKENS: "always"}, n_tokens=3000)   # `always` lifts the line's own cap too
    assert lifted.exports["OF3_CUDA_GRAPHS"] == "1" and (lifted.size_gate, lifted.gate_reason) == ("graph=capture", "no_cap")


def test_a_preset_graph_switch_conflicts_above_the_cap():
    res = modes.resolve("fast", HOME, environ={**CAP400, "OF3_CUDA_GRAPHS": "1"}, n_tokens=2000)
    assert any("OF3_CUDA_GRAPHS" in c and "unset" in c for c in res.conflicts)


def test_graphed_call():
    assert modes.graphed_call("fast", None, 200, CAP400) and not modes.graphed_call("fast", None, 800, CAP400)
    assert modes.graphed_call("fast", None, 800, CLEAN)                                              # default: no cap
    assert modes.graphed_call("fast", None, None, CAP400)
    assert modes.graphed_call("big", "resident", 10, CAP400) and not modes.graphed_call("big", "resident", 800, CAP400)   # resident below the offload port's item gate = the fast line: graphed within the cap, eager above it
    assert not modes.graphed_call("big", "resident", modes.OF3O_GATE_DEFAULT, CLEAN) and not modes.graphed_call("big", "resident", None, CLEAN) and not modes.graphed_call("big", "tp", 10, CLEAN)   # the port engaged (at the gate, no token count) and tp: eager
    assert modes.graphed_call("exact", "cueq", 100, CAP400) and not modes.graphed_call("exact", "cueq", 5000, CLEAN)   # the exact line: graphed within the cap (its own 512 without one), eager above
    assert modes.graphed_call("exact", "cueq", 512, CLEAN) and not modes.graphed_call("exact", "cueq", 513, CLEAN) and not modes.graphed_call("exact", "cueq", None, CLEAN)   # no token count: eager (fail-closed)
    assert not modes.graphed_call("fast", None, 100, {**CLEAN, modes.ENV_GRAPHS_MAX_TOKENS: "0"})       # 0 = never


def test_polymer_tokens():
    qs = {"queries": {"a": {"chains": [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "M" * 100},
                                        {"molecule_type": "rna", "chain_ids": ["C"], "sequence": "ACGU"},
                                        {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "CCO"}]},
                      "b": {"chains": [{"molecule_type": "dna", "chain_ids": "D", "sequence": "ACGT" * 10}]}}}
    assert inputs.polymer_tokens(qs) == (204, 1)
    assert inputs.polymer_tokens({"queries": {}}) == (0, 0)
