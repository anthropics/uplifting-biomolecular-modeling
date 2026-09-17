"""The sampler step graph's plan-for-capture window.
`stepgraph.planning()` is True while a sampler call the lever ADMITTED runs its denoiser steps; two bindings inside the step whose choice
followed the live capture state alone read it and choose inside the window as they choose under capture — levers/SAMPLER `odde_apb_bind`
asks the provider's graph-timed column under a tier word (fast | big; an exact word keeps the literal capture state), `lncore` steps aside
to upstream's LayerNorm — so the eager oracle the capture is compared with serves the captured step's kernels. Outside the window
(no sampler call, a refused or stood-aside call, the lever off) every choice follows the live capture state. CPU only: the window is a host-side flag."""
import importlib
import os
import sys

import pytest

from opendde_opt import lncore, modes, report, stepgraph

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "levers", "SAMPLER"))


@pytest.fixture
def clean():
    stepgraph._reset()
    yield
    stepgraph._reset()
    stepgraph._CUR[0] = None


@pytest.fixture
def bind_module(monkeypatch):
    pytest.importorskip("torch")
    for k in modes.SAMPLER_SWITCHES + ("ODDE_DIT_ATTN", "ODDE_SAMPLER_PROBE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(UNIT)
    for name in ("odde_apb_bind", "odde_sampler"):
        sys.modules.pop(name, None)
    B = importlib.import_module("odde_apb_bind")
    B.reset()
    yield B
    B.reset()
    for name in ("odde_apb_bind", "odde_sampler"):
        sys.modules.pop(name, None)


def _open_window():
    c = stepgraph._Ctx(dm=None, det=False)
    stepgraph._CUR[0] = c
    return c


def test_the_window_is_an_admitted_live_sampler_call(clean):
    assert stepgraph.planning() is False                                                                 # no sampler call
    c = _open_window()
    assert stepgraph.planning() is False                                                                 # a call not (yet) admitted: its steps are the stock eager sampler's
    c.admitted = True
    assert stepgraph.planning() is True                                                                  # admitted: eager head, reference, warm-up, capture, held re-runs, replays
    c.refused = "aside:verify_mismatch_verify1_maxabs_1.0e-01"
    assert stepgraph.planning() is False                                                                 # stood aside mid-call: the rest of the call is eager, outside the window
    c.refused = None
    stepgraph.STATS["disabled"] = "aside:x"
    assert stepgraph.planning() is False                                                                 # the graph is off for the process
    stepgraph.STATS["disabled"] = None
    stepgraph._CUR[0] = None
    assert stepgraph.planning() is False and "planned_calls" in stepgraph.STATS


def test_step_opens_the_window_at_admission_before_the_eager_head_call(clean, monkeypatch):
    """`_Ctx.step`: the first denoiser call is admitted (or refused by name) BEFORE it runs; the window opens there, so the eager head call the
    DITFAST hoist records in already serves the capture's rows."""
    seen = []
    c = _open_window()
    monkeypatch.setattr(c, "_admit", lambda kwargs: None)
    monkeypatch.setitem(stepgraph._ORIG, "dm_call", lambda dm, **kw: seen.append(stepgraph.planning()) or "out")
    monkeypatch.setattr(stepgraph, "_make_cache", lambda det: None)
    assert c.step((), {"x_noisy": 0, "t_hat_noise_level": 0, "s_inputs": 0}) == "out"                    # head call (HEAD=1): eager, inside the window
    assert seen == [True] and c.admitted and stepgraph.STATS["planned_calls"] == 1 and stepgraph.STATS["eager_head"] == 1
    c2 = _open_window()
    monkeypatch.setattr(c2, "_admit", lambda kwargs: "memory_est_1MiB_over_0.5_of_free_0MiB")
    seen.clear()
    assert c2.step((), {"x_noisy": 0, "t_hat_noise_level": 0, "s_inputs": 0}) == "out"
    assert seen == [False] and not c2.admitted and c2.refused is not None                                # a refused call never opens it


def test_apb_binding_asks_the_graph_column_inside_the_window_under_the_big_word(clean, bind_module, monkeypatch):
    B = bind_module
    assert B.PLAN_WORDS == ("big",) and B.STEPGRAPH_MODULE == "opendde_opt.stepgraph" == lncore.STEPGRAPH_MODULE and lncore.WINDOW_WORDS == ("big",)
    assert B.planned() is False and [B.column(w, False) for w in ("exact", "fast", "big", None)] == ["eager"] * 4
    assert [B.column(w, True) for w in ("exact", "fast", "big")] == ["graph"] * 3                       # under capture: the provider's own rule, every word
    c = _open_window(); c.admitted = True
    assert B.planned() is True
    assert [B.column(w, False) for w in ("exact", "fast", "big", "fpf_apb")] == ["eager", "eager", "graph", "eager"]   # inside: the big word plans for the capture; exact / fast / a row pin literal (0.2.72)
    KA = B.KA
    monkeypatch.setattr(B, "_cc", lambda: (9, 0))
    seen = []

    def sel(cc, dtype, cell, n_tokens, *, word, **kw):
        seen.append((word, n_tokens, kw.get("timing"), kw.get("capture")))
        return KA.Selection("fpf_apb", None, word, f"9.0|fp32|dit_h16d48|S5|N<=800|{kw.get('timing')}|fwd", True, None, True, "tolerance", None, False, True, 3.0, None, "winner", None)
    monkeypatch.setattr(KA, "select", sel)
    monkeypatch.setattr(KA, "cell_family", lambda cc, dtype, cell, timing="eager": {(5, 800): "k"})
    B.COUNTS["dit"]["word"] = "big"
    s_in = B.selection("dit", 5, 448, "fp32", False)
    assert seen[-1] == ("big", 448, "graph", False)                                                    # an EAGER call inside the window: the graph-timed cells, capture= literal
    assert B.selection("dit", 5, 448, "fp32", True) is not s_in and seen[-1] == ("big", 448, "graph", True)
    stepgraph._CUR[0] = None                                                                             # window closed: the same class asks the eager column again (its own memo entry)
    s_out = B.selection("dit", 5, 448, "fp32", False)
    assert seen[-1] == ("big", 448, "eager", False) and s_out is not s_in and len(seen) == 3
    assert B.selection("dit", 5, 448, "fp32", False) is s_out and len(seen) == 3                         # memoised per (class, column)
    B.COUNTS["exact"]["word"] = "exact"
    c2 = _open_window(); c2.admitted = True
    monkeypatch.setattr(B, "_abi", lambda: None, raising=False)
    B.selection("exact", 5, 448, "fp32", False, "exact")
    assert seen[-1][2:] == ("eager", False)                                                              # an exact word inside the window: the literal capture state, as before


def test_apb_binding_reads_the_window_through_sys_modules_without_importing_the_package():
    src = open(os.path.join(UNIT, "odde_apb_bind.py")).read()
    assert "import opendde_opt" not in src and "from opendde_opt" not in src and 'sys.modules.get(STEPGRAPH_MODULE)' in src
    assert 'timing = column(w, capture)' in src and 'timing = "graph" if capture else "eager"' not in src


def test_lncore_steps_aside_inside_the_window_under_the_big_word_only(clean, monkeypatch):
    monkeypatch.setitem(lncore.COUNTS, "word", "big")
    assert lncore._stepgraph_window() is False
    c = _open_window(); c.admitted = True
    assert lncore._stepgraph_window() is True
    for w in ("fast", "fastln", "fastln:lp", None):                                                      # under fast / a row word / no word the lever ignores the window —
        monkeypatch.setitem(lncore.COUNTS, "word", w)                                                    # its calls follow the literal capture state alone
        assert lncore._stepgraph_window() is False, w
    monkeypatch.setitem(lncore.COUNTS, "word", "big")
    assert lncore._stepgraph_window() is True
    src = open(lncore.__file__).read()
    i, j = src.index('return "capturing"'), src.index('return "stepgraph_window"')
    assert 0 < i < j and "if _stepgraph_window():" in src[i:j]                                          # asked right after the live capture state, before the row is chosen
    c.refused = "x"
    assert lncore._stepgraph_window() is False


def test_census_tokens(clean):
    stats = {"accel_v2": {"apb_state_dit": {"calls": 10, "word": "big", "rows": {"fpf_apb": 10}, "stock_calls": 0, "plan_calls": 0, "refusals": {}, "notes": []}}}
    ev = report.lever_evidence("dit_attn_apb", stats)
    assert ev.get("plan_calls") is None and ev["calls"] == 10                                            # plan_calls=0: no token (every pinned census line unchanged)
    stats["accel_v2"]["apb_state_dit"]["plan_calls"] = 6
    assert report.lever_evidence("dit_attn_apb", stats)["plan_calls"] == 6


@pytest.mark.parametrize("n", (17, 200, 256, 400, 448, 800, 1024, 1399))
def test_fast_selections_ignore_the_window_at_every_size(n, clean, bind_module, monkeypatch):
    """Under the fast word every apb selection — eager calls inside the step graph's window included — is the literal-capture-state
    selection: same provider call (timing column "eager"), same memo entry, no plan_calls; the capture itself asks "graph" as
    always. Driven on the provider's real table for the H100 key (cc 9.0), both sites, fp32 statements and the fused stack's fp16 operands."""
    B = bind_module
    KA = B.KA
    monkeypatch.setattr(B, "_cc", lambda: (9, 0))
    asked = []
    real = KA.select

    def spy(cc, dtype, cell, n_tokens, *, word, **kw):
        asked.append((cell, word, n_tokens, kw.get("timing"), kw.get("capture")))
        return real(cc, dtype, cell, n_tokens, word=word, **kw)
    monkeypatch.setattr(KA, "select", spy)
    for site in ("dit", "atom"):
        B.COUNTS[site]["word"] = "fast"
    nat = n * 9                                                                                          # atoms ~ tokens x 9 (the provider keys atom cells on atoms // 8)
    out = {(site, dt): B.selection(site, 5, n if site == "dit" else nat, dt, False) for site in ("dit", "atom") for dt in ("fp32", "fp16")}
    k0 = len(asked)
    assert k0 == 4 and all(a[3:] == ("eager", False) for a in asked)
    c = _open_window(); c.admitted = True                                                                # a graphed fast sampler call: inside the window …
    assert B.planned() is True
    for (site, dt), sel in out.items():
        assert B.selection(site, 5, n if site == "dit" else nat, dt, False) is sel                       # … the SAME memoised selection: no new provider call, the eager column
    assert len(asked) == k0 and B.column("fast", False) == "eager"
    assert B.COUNTS["dit"]["plan_calls"] == 0 == B.COUNTS["atom"]["plan_calls"]
    cap = B.selection("dit", 5, n, "fp32", True)                                                         # the capture: the graph column, as always
    assert asked[-1][3:] == ("graph", True)
    B.COUNTS["dit"]["word"] = "big"                                                                    # the big word at the same size inside the window: the graph column, as the window chooses
    B.selection("dit", 5, n, "fp32", False)
    assert asked[-1][1:] == ("big", n, "graph", False)
