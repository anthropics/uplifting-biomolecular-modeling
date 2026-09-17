"""The pair-track driver on a card whose provider tables carry NO cell for a site's tier word (not even an inherited one): the site — or,
for the TriMul / transition sites, the whole stack — steps aside BY NAME (the block's own keyed core `default` / the per-layer path), named on
the LEVER line (`handed=` / `fallback_by=no_cell:…`), the fail-closed gate holds, the pass completes.  On the measured cards (9.0 / 8.0)
nothing changes: the cells exist, the provider's row serves, nothing is handed."""
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, OPT)
sys.path.insert(0, os.path.join(OPT, "forward", "pairfuse"))
import bz_pairfuse as D                                    # noqa: E402
from boltz2_opt import pairfuse as AD, modes               # noqa: E402
from opt_core.kernels import triattn as KA                 # noqa: E402
from opt_core.attn import pair_fused as PF                 # noqa: E402


def _fresh(monkeypatch, mode="fast"):
    r, picks = D.parse_word(modes.env_row(mode)["BOLTZ_PAIRFUSE"])
    monkeypatch.setitem(D._STATE, "residency", r); monkeypatch.setitem(D._STATE, "picks", picks); monkeypatch.setitem(D._STATE, "handed_triatt", {})
    for k in ("handed", "fallback", "core_cells", "errors", "sites", "served_by", "shapes"):
        monkeypatch.setitem(D.STATS, k, {} if k != "sites" else {w: 0 for w in D.SITE_WORDS})
    monkeypatch.setitem(D.STATS, "stack_calls", 0); monkeypatch.setitem(D.STATS, "stack_served", 0)
    monkeypatch.setitem(D._STATE, "applied", True); monkeypatch.setitem(D._STATE, "word", modes.env_row(mode)["BOLTZ_PAIRFUSE"])
    monkeypatch.setattr(D, "composed_ok", lambda: None)     # the hooks-in-force check imports boltz (absent on a CPU-only host)


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_a_triattn_tier_word_with_no_provider_cell_hands_the_site_to_the_blocks_keyed_core_by_name(mode, monkeypatch):
    _fresh(monkeypatch, mode)
    word = D.core_word(D._triatt_pick()); assert word == {"fast": "fast", "big": "big"}[mode]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))      # a card no table lists (and no inheritance answers for, below)
    def no_cell(cc, dtype, head_dim, heads, n_tokens, *a, **k):                                  # the face's refusal BY NAME for a tier word with no cell: kind no_cell:<key>, the library kernel named as the row to bind
        raise KA.Refusal("no_cell:bf16_D32", k.get("word", "fast"), "cueq", "no cell measured for this call class")
    monkeypatch.setattr(KA, "select", no_cell)
    m = types.SimpleNamespace(mha=types.SimpleNamespace(c_hidden=32, no_heads=4))
    zp = torch.empty((1, 400, 400, 128), dtype=torch.bfloat16)
    ok, reason = D._probe_core_triatt(m, zp, word)
    assert (ok, reason) == (True, ""), "no cell for the word on this card: the site steps aside by name, the probe does not refuse the stack"
    assert D._STATE["handed_triatt"] == {400: "default"} and D.STATS["handed"] == {f"triatt@400:{word}->default:no_cell:bf16_D32": 1}
    core, site_word = D._core_for(400)
    assert core == D.PF_DEFAULT_CORE == PF.DEFAULT_CORE and site_word == D.TRIATT_SITE_WORDS["default"], "the handed token count serves the block's own keyed core (flash_triattn on a card its table does not list)"
    assert PF.default_core_for((12, 0), "bf16", 32, 4, 400) == "flash_triattn"
    g = D.gate(); assert g["ok"], g
    text = D.line()
    assert f" handed=triatt@400:{word}->default:no_cell:bf16_D32=1 " in text and " gate=ok" in text and "REFUSED" not in text, text


def test_on_the_measured_cards_nothing_is_handed_the_providers_row_serves(monkeypatch):
    _fresh(monkeypatch, "fast")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))       # 8.0: the CUDA row's sm_80 member / the Triton cell — a measured cell at every ladder size
    m = types.SimpleNamespace(mha=types.SimpleNamespace(c_hidden=32, no_heads=4))
    for n in (128, 400, 800, 1200):
        zp = torch.empty((1, n, n, 128), dtype=torch.bfloat16)
        assert D._probe_core_triatt(m, zp, "fast") == (True, "")
    assert D._STATE["handed_triatt"] == {} and D.STATS["handed"] == {}, "cells exist: served by the provider's row, nothing handed"
    assert all(D._core_for(n)[1] == D.TRIATT_SITE_WORDS["core"] for n in (128, 400, 800, 1200))


def test_a_trimul_or_transition_tier_word_with_no_cell_hands_the_stack_to_the_per_layer_path_by_name_and_the_gate_holds(monkeypatch):
    _fresh(monkeypatch, "fast")
    assert D._no_cell_word("trimul", "fast:no_cell:cc=12.0") == "no_cell:trimul:cc=12.0" and D._no_cell_word("transition", "no_cell:pair_bf16_C128") == "no_cell:transition:pair_bf16_C128"
    assert D._no_cell_word("trimul", "fast:cc:8.0!=9.0") is None, "any other refusal keeps its own word (undeclared: the gate refuses, as before)"
    assert D.declared("no_cell:trimul:cc=12.0") and D.declared("no_cell:transition:stock:torch_swiglu") and D.declared("kernels_off") and D.declared("c:64") and not D.declared("trimul:fast:cc:8.0!=9.0")
    # a run whose every C=128 stack met a TriMul word with no cell on the card: each stack took the per-layer path by name — declared, the gate holds (idle by name), the line says so
    monkeypatch.setitem(D.STATS, "fallback", {"no_cell:trimul:cc=12.0": 12}); monkeypatch.setitem(D.STATS, "stack_calls", 12); monkeypatch.setitem(D.STATS, "stack_served", 0)
    g = D.gate()
    assert g["ok"] and g["idle"], g
    text = D.line()
    assert " state=skipped " in text and " fallback_by=no_cell:trimul:cc=12.0:12 " in text and " gate=ok" in text, text
    monkeypatch.setitem(AD._STATE, "driver", D); monkeypatch.setitem(AD._STATE, "applied", ["pairfuse"])
    assert AD.verdict()["ok"], "the adapter's verdict (stack.evidence reads it) treats the by-name hand-off as declared"
    # an UNDECLARED core refusal still refuses the gate (fail-closed unchanged)
    monkeypatch.setitem(D.STATS, "fallback", {"trimul:fast:cc:8.0!=9.0": 1})
    assert not D.gate()["ok"] and not AD.verdict()["ok"]


def test_stack_evidence_accepts_a_driver_that_handed_every_stack_off_by_name():
    from boltz2_opt import stack
    src = open(stack.__file__).read()
    assert "pairfuse_handed_off" in src and 'all(str(k).startswith("no_cell:") for k in pfb)' in src, "the trunk-graph check under the driver accepts served>0 OR graphs served OR every stack handed off by name"
