"""boltz2_opt.triattn_exact: the exact row's binding of the fused block's library attention core to the core provider's ``exact`` word.

Switch unset: pairblock's library core is handed on untouched (no line, nothing counted). Switch set: every call reaches
``opt_core.kernels.triattn.triangle_attention(..., word="exact", stock=<the library call>)``, a provider ``Refusal`` puts that call on the
library call by name, the LEVER line states the provider's row for this card and stack in the kit's grammar, the report carries the exact
row's evidence line, and the launcher renders the lever from the worker log. The mode table sets the switch on the exact row alone. CPU is
enough: the provider's serving entry is a stand-in; its table walk (select / describe) is the real one.
"""
import re

import pytest

from .. import modes, registry, report, stack, triattn_exact as TX
from ..worker_launch import ATTACH


def _fresh(monkeypatch, switch, cc="9.0", stack_key="9.0|torch0.0.0+cu000|cueq0.0.0"):
    """A clean adapter state with the switch `switch` ('1' | None), the card answering `cc`, the provider's exact-vouch stack key pinned to
    a key no table vouches (the exact word then names the library op: the provider's contract), no exit tally registered."""
    from opt_core import report as core_report
    from opt_core.kernels import triattn as T
    monkeypatch.setattr(TX, "_STATE", {"active": False, "applied": [], "facts": None, "stack": {}, "row_at": {}, "routed": {},
                                       "calls": 0, "served": 0, "refused": {}, "errors": {}, "rows": {}})
    if switch is None:
        monkeypatch.delenv(TX.SWITCH, raising=False)
    else:
        monkeypatch.setenv(TX.SWITCH, switch)
    monkeypatch.setattr(TX.RF, "card_cc", lambda: cc)
    monkeypatch.setattr(T, "exact_stack_key", lambda cc_=None: stack_key)
    monkeypatch.setattr(core_report, "register_exit_tally", lambda tag, line_of: True)
    return T


def _lib(seen):
    def cueq_core(q, k, v, bias, mask5, scale):                       # pairblock._cueq_core's signature (positional mask / scale)
        seen.append(("lib", mask5, scale))
        return v
    return cueq_core


def test_switch_unset_hands_the_library_core_on_untouched(monkeypatch, capsys):
    _fresh(monkeypatch, None)
    lib = _lib([])
    assert TX.enabled() is False and TX.variant() is None
    assert TX.core_for(lib) is lib, "the block's own core object, not a wrapper: fast / big / an exact row without the word run as before"
    assert TX.apply() == [] and TX.line() is None and TX.report()["applied"] == [] and TX.report()["gate"] is None
    assert "LEVER" not in capsys.readouterr().err


def test_switch_set_routes_every_call_through_the_exact_word_with_the_library_call_as_stock(monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    T = _fresh(monkeypatch, "1")
    calls, seen = [], []

    def fake_triangle_attention(q, k, v, bias, mask=None, scale=None, *, word, stock=None, **kw):
        calls.append({"word": word, "stock": stock, "mask": mask, "scale": scale, "kw": kw})
        if q.shape[-2] == 3:                                          # a structural refusal, by name
            raise T.Refusal("no_cell:heads=4", "exact", "cueq", "test")
        if q.shape[-2] == 5:                                          # a kernel error: counted, raised to the block
            raise RuntimeError("boom")
        return stock(q, k, v, bias, mask=mask, scale=scale)           # what the provider does where no exact row is vouched: the stock op

    monkeypatch.setattr(T, "triangle_attention", fake_triangle_attention)
    lib = _lib(seen)
    core = TX.core_for(lib)
    assert core is not lib and core.__wrapped__ is lib and TX.core_for(lib) is core, "one routed callable per library core (memoised)"
    err = capsys.readouterr().err
    assert err.count("LEVER name=triattn_exact state=on") == 1, "the LEVER line prints once, at activation"

    B, N, H, S, D = 1, 4, 4, 8, 32
    q = torch.zeros(B, N, H, S, D); k = torch.zeros(B, N, H, S, D); v = torch.ones(B, N, H, S, D); bias = torch.zeros(B, 1, H, S, S)
    mask5 = torch.ones(B, N, 1, 1, S, dtype=torch.bool)
    out = core(q, k, v, bias, mask5, 0.125)
    assert out is v and len(calls) == 1 and calls[0]["word"] == "exact" and calls[0]["mask"] is mask5 and calls[0]["scale"] == 0.125 and calls[0]["kw"] == {}
    st = calls[0]["stock"]
    assert st.__wrapped__ is lib and seen == [("lib", mask5, 0.125)], "stock= is the library call in the provider's convention (mask= / scale= keywords onto the positional core)"
    core(q, k, v, bias, None, 0.125)                                  # the bias-only form (no key mask reaches the core)
    assert calls[1]["mask"] is None and seen[-1] == ("lib", None, 0.125)

    q3 = torch.zeros(B, N, H, 3, D); v3 = torch.ones(B, N, H, 3, D)
    assert core(q3, q3, v3, torch.zeros(B, 1, H, 3, 3), None, 0.125) is v3, "a Refusal by name: THIS call on the library call"
    assert seen[-1] == ("lib", None, 0.125) and TX.census()["fallback"] == {"no_cell": 1}

    q5 = torch.zeros(B, N, H, 5, D)
    with pytest.raises(RuntimeError):
        core(q5, q5, q5, torch.zeros(B, 1, H, 5, 5), None, 0.125)
    c = TX.census()
    assert (c["calls"], c["served"], c["errors"], c["rows"]) == (4, 2, {"RuntimeError": 1}, {"cueq": 2}), c
    assert TX.verdict()["ok"] is False and "RuntimeError" in TX.verdict()["reason"], "an error refuses the fail-closed gate"
    assert capsys.readouterr().err.count("LEVER name=triattn_exact") == 0, "no second activation line"


LINE_RX = re.compile(r"^\[boltz2-opt\] LEVER name=triattn_exact state=on word=exact row=(?P<row>\S+) class=(?P<cls>exact|stock)( exact_vs=\S+)? cell=\S+ measured=(yes|no)"
                     r"( \S+=\S+)*? cc=(?P<cc>\S+) stack=(?P<stack>\S+) at=bf16xD32xH4xN\d+ calls=\d+ served=\d+ refused=\S+ errors=\d+ rows=\S+ kernel=\d+/\d+ kernel_refused=\S+"
                     r" impl=opt_core\.kernels\.triattn origin=core$")


def test_lever_line_grammar_states_the_providers_row_for_this_card_and_stack(monkeypatch, capsys):
    _fresh(monkeypatch, "1", cc="9.0")
    assert TX.apply() == ["triattn_exact"] and TX.apply() == ["triattn_exact"]
    ln = TX.line(); m = LINE_RX.match(ln)
    assert m, ln
    assert m.group("row") == "cueq" and m.group("cls") == "stock" and m.group("cc") == "9.0" and m.group("stack") == "9.0|torch0.0.0+cu000|cueq0.0.0", \
        "on a stack no exact-class row is vouched on, the exact word names the library op (the provider's cueq row)"
    assert capsys.readouterr().err.strip() == ln and all(" " not in tok for tok in ln.split(" ")), "printed once at activation; every value one token"
    f = TX.facts()
    assert (f["word"], f["row"], f["cc"], f["at"]) == ("exact", "cueq", "9.0", "bf16xD32xH4xN%d" % TX.LINE_TOKENS) and f["refused"] is None
    g = TX.verdict()
    assert g == {"ok": True, "idle": True, "reason": "no call reached the block's library core"}, "applied, no call yet: idle, not a failure"
    r = TX.report()
    assert r["applied"] == ["triattn_exact"] and r["variant"] == "1" and r["line"] == ln and r["census"]["calls"] == 0 and r["gate"] == g and r["impl"] == TX.IMPL


def test_lever_line_without_a_card_names_the_refusal(monkeypatch, capsys):
    _fresh(monkeypatch, "1", cc=None)
    TX.apply()
    ln = TX.line()
    assert ln.startswith("[boltz2-opt] LEVER name=triattn_exact state=on word=exact row=- cc=- stack=") and " refused=no_card fallback=- " in ln and ln.endswith("impl=opt_core.kernels.triattn origin=core"), ln


def test_the_report_carries_the_exact_rows_evidence_line_and_tally(monkeypatch):
    _fresh(monkeypatch, "1")
    from opt_core.kernels.triattn import exact_member as EM
    monkeypatch.setattr(EM, "counts", lambda: {"served": 7, "calls": 9, "refused": {"short_seq": 2}, "installed": True})
    monkeypatch.setattr(EM, "evidence_line", lambda: "triattn_exact: served 7/9 calls (refused: {'short_seq': 2})")
    TX.apply()
    r = TX.report()
    assert r["evidence"] == "triattn_exact: served 7/9 calls (refused: {'short_seq': 2})" == TX.evidence()
    assert r["census"]["kernel"] == {"served": 7, "calls": 9, "refused": {"short_seq": 2}, "installed": True}
    assert " kernel=7/9 kernel_refused=short_seq:2 " in TX.line()


def test_modes_table_sets_the_switch_on_the_exact_row_alone():
    assert registry.LEVERS["triattn_exact"]["switch"] == TX.SWITCH == "BOLTZ_TRIATTN_EXACT" and registry.LEVERS["triattn_exact"]["value"] == "1"
    assert registry.LEVERS["triattn_exact"]["origin"] == "core" and registry.LEVERS["triattn_exact"]["strategy"] == "LOCAL.boltz2.triattn_exact" and registry.LEVERS["triattn_exact"]["tier"] == 1
    ex = modes.env_row("exact")
    assert ex[TX.SWITCH] == "1" and TX.SWITCH in modes._EXACT_ONLY and "triattn_exact" in modes.levers("exact")
    assert list(ex).index(TX.SWITCH) == list(ex).index("BOLTZ_PAIRBLOCK") + 1 and modes.levers("exact").index("triattn_exact") == modes.levers("exact").index("pairblock") + 1
    for m in ("off", "fast", "big"):
        assert TX.SWITCH not in modes.env_row(m) and "triattn_exact" not in modes.levers(m) and "triattn_exact" not in modes.attachments(m), m
    att = modes.attachments("exact")
    assert att.index("triattn_exact") == att.index("pairblock") + 1 and modes.LEVER_ATTACH["triattn_exact"] == "triattn_exact" and modes.NEEDS["triattn_exact"] == ("pairblock",)
    a = ATTACH["triattn_exact"]
    assert a == {"module": "boltz2_opt.triattn_exact", "trigger": ATTACH["pairblock"]["trigger"], "report_key": "triattn_exact_report"}
    assert not [p for p in stack.attachment_problems("exact") if "triattn_exact" in p], "importable, LEVERS installed"
    row = modes._tabled("exact")                                       # the lever rides pairblock's library core: it leaves the row with the block, by name, word and attachment with it
    modes.take_off(row, "pairblock", "test", "run_off")
    assert "triattn_exact" not in row["levers"] and TX.SWITCH not in row["env"] and "triattn_exact" not in row["attach"] and row["run_off"]["triattn_exact"] == "test"


def test_pairblock_hands_its_cueq_core_through_the_seam(monkeypatch):
    import inspect
    from .. import pairblock as PB
    assert "TX.core_for(_cueq_core)" in inspect.getsource(PB) and PB.TX is TX
    _fresh(monkeypatch, None)
    assert TX.core_for(PB._cueq_core) is PB._cueq_core, "switch unset: the block's cueq variant hands pair_fused the library call itself"


def test_evidence_line_reaches_the_launcher(monkeypatch):
    """The worker log's triattn_exact_report is required evidence of the exact row (stack.evidence names its absence, a refused gate, a
    wrong word), and report.lever_lines renders the lever in the core's LEVER grammar with the provider's row and the kernel row's tally."""
    from .test_cli_and_evidence import _log
    kon = ["resid", "mask2"]
    log = _log(kon, mode="exact")
    assert log["triattn_exact_report"]["applied"] == ["triattn_exact"]
    ev, problems = stack.evidence("exact", log)
    assert problems == [] and ev["triattn_exact_report"]["variant"] == "1" and ev["triattn_exact_idle"] is False
    lines = report.lever_lines("exact", ev, {"n_gpu": 1})
    mine = [l for l in lines if " name=triattn_exact " in l]
    assert len(mine) == 1 and mine[0].startswith("[boltz2-opt] LEVER name=triattn_exact state=on impl=opt_core.kernels.triattn:triangle_attention(word=exact) origin=core strategy=LOCAL.boltz2.triattn_exact word=exact row=cueq cc=9.0 stack=") \
        and " kernel=0/0 kernel_refused=- " in mine[0] and " rows=cueq:192 " in mine[0] and mine[0].endswith(" idle=False gate=ok"), mine
    bad = dict(log); bad.pop("triattn_exact_report")
    assert any("no triattn_exact_report in the worker log" in p for p in stack.evidence("exact", bad)[1])
    bad = dict(log); bad["triattn_exact_report"] = dict(log["triattn_exact_report"], gate={"ok": False, "idle": False, "reason": "errors=RuntimeError:1"})
    assert any("exact triangle-attention word binding gate refused: errors=RuntimeError:1" in p for p in stack.evidence("exact", bad)[1])
    bad = dict(log); bad["triattn_exact_report"] = dict(log["triattn_exact_report"], variant="0")
    assert any("variant '0' != the row's BOLTZ_TRIATTN_EXACT='1'" in p for p in stack.evidence("exact", bad)[1])
    idle = dict(log); idle["triattn_exact_report"] = dict(log["triattn_exact_report"], census=dict(log["triattn_exact_report"]["census"], calls=0, served=0, rows={}), gate={"ok": True, "idle": True, "reason": "no call reached the block's library core"})
    ev, problems = stack.evidence("exact", idle)
    assert problems == [] and ev["triattn_exact_idle"] is True, "installed and idle (every call on the block's declared stock paths) is named, not a failure"
    st, reason, pairs = report.lever_state("triattn_exact", {}, {})
    assert (st, reason) == ("skipped", "not_in_triattn_exact_report.applied")
