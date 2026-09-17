"""No-refusal notes: a property decided up front that a line runs with is a NOTE line and the run proceeds (rc 0) — the tp launcher's chunk plan on
a card below its memory class, the offload port's pinned host pool over its budget (pageable, counted, never PARTIAL), the rowpair diffusion band
plan over its advisory whole-row count; an explicit `strict` pin policy and a pinned allocation the host refuses stay refusals by name. CPU only."""
import importlib
import os
import sys

import pytest

from openfold3_ob0_opt import modes, stack, tp

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")


def test_card_note_words_and_the_previously_allowed_region():
    assert tp.card_note(80 * 10 ** 9, environ={}) is None and tp.card_note(None, environ={}) is None          # the 80 GB class and an unreadable card: no note (unchanged)
    assert tp.card_note(40 * 10 ** 9, environ={"OF3TP_CHUNK": "32"}) is None                                  # the caller's chunk: no note
    note = tp.card_note(40 * 10 ** 9, environ={})
    assert note.startswith("NOTE chunk plan table (CHUNK_PLAN) is for >=79 GB cards; using it on 40.0 GB") and "may OOM" in note and "OF3TP_CHUNK=<chunk> tunes it" in note
    src = open(tp.__file__, encoding="utf-8").read()
    launch = src[src.index("def launch("):]
    assert "card_note(mem)" in launch and "MIN_CARD_BYTES" not in launch                                       # the launcher notes and proceeds; no card-size exit


@pytest.fixture
def of3o(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(OF3O)
    old = sys.modules.pop("of3_offload", None)
    sys.dont_write_bytecode, was = True, sys.dont_write_bytecode                                              # the port directory carries no __pycache__
    try:
        mod = importlib.import_module("of3_offload")
        mod.STATE["pageable"].clear(); mod._PIN_POOL.clear(); mod._PIN_BYTES["pinned"] = 0
        yield mod, torch
    finally:
        sys.dont_write_bytecode = was
        sys.modules.pop("of3_offload", None)
        if old is not None:
            sys.modules["of3_offload"] = old


def test_pin_pool_over_budget_is_a_note_and_pageable_under_census(of3o, monkeypatch, capsys):
    mod, torch = of3o
    monkeypatch.delenv("OF3O_PIN_POLICY", raising=False); monkeypatch.setenv("OF3O_PIN_BUDGET_GB", "0")      # every buffer is over the budget: decided up front
    assert mod.pin_policy() == "census" == modes.LINES[("big", "resident")].env["OF3O_PIN_POLICY"]
    t = torch.zeros(4, 3)
    buf = mod.pinned_like(t, "ztest")
    assert tuple(buf.shape) == (4, 3) and buf.dtype == t.dtype and mod.STATE["pageable"] == {"ztest": 1}
    err = capsys.readouterr().err
    assert "NOTE pinned host pool over OF3O_PIN_BUDGET_GB" in err and "buffer ztest" in err and "pageable" in err and '"event": "pageable"' in err
    assert stack.offload_partial({"fallbacks": {}, "pageable": {"ztest": 1}, "conf_census": {}}) == []       # counted (exit tally offload_pageable), never PARTIAL
    assert stack.offload_partial({"fallbacks": {"run_trunk": 1}, "pageable": {}, "conf_census": {}}) == ["offload fallback run_trunk x1"]   # a stock-body fallback stays PARTIAL
    monkeypatch.setenv("OF3O_PIN_POLICY", "strict")                                                             # the explicit opt-in: refused by name
    with pytest.raises(RuntimeError, match="OF3O_PIN_POLICY=strict refuses a pageable buffer"):
        mod.pinned_like(torch.zeros(5, 3), "ztest2")
    monkeypatch.setenv("OF3O_PIN_POLICY", "sometimes")
    with pytest.raises(ValueError, match="OF3O_PIN_POLICY"):
        mod.pin_policy()


def test_a_pinned_allocation_the_host_refuses_raises_by_name(of3o, monkeypatch):
    mod, torch = of3o
    monkeypatch.delenv("OF3O_PIN_POLICY", raising=False); monkeypatch.setenv("OF3O_PIN_BUDGET_GB", "600")
    real_empty = torch.empty

    def empty(*a, **k):
        if k.get("pin_memory"):
            raise RuntimeError("CUDA error: out of memory (cudaHostAlloc)")
        return real_empty(*a, **k)
    monkeypatch.setattr(mod.torch, "empty", empty)
    for policy in ("census", "strict"):
        monkeypatch.setenv("OF3O_PIN_POLICY", policy)
        with pytest.raises(RuntimeError, match="pinned allocation failed"):
            mod.pinned_like(torch.zeros(2, 2), f"zfail_{policy}")
    assert mod.STATE["pageable"] == {}                                                                          # nothing paged around the failure


def test_band_plan_extra_rows_are_advisory():
    """The adapter module imports upstream (torch, openfold3) — read its source: the note function is pure and runs here from its own text."""
    import ast
    path = os.path.join(HOME, "opt", "openfold3_ob0_opt", "tp_rowpair", "diffusion.py")
    src = open(path, encoding="utf-8").read(); tree = ast.parse(src)
    pieces = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(tg, "id", "") in ("ENV_BAND_W", "ENV_BAND_EXTRA_MAX", "BAND_EXTRA_UNCAPPED") for tg in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name == "band_extra_note")]
    ns = {"Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=pieces, type_ignores=[]), path, "exec"), ns)
    assert ns["band_extra_note"](10, 256, 32) is None and ns["band_extra_note"](256, 256, 32) is None       # within the advisory count: no note (unchanged plans)
    note = ns["band_extra_note"](300, 256, 32)
    assert note.startswith("NOTE diffusion band plan carries 300 whole pair rows outside the |k-q|<=32 band (> ROWPAIR_DIFF_BAND_EXTRA_MAX=256, advisory)") and "proceeding with the requested band" in note
    body = src[src.index("def band_plan_of("): src.index("BAND_EXTRA_UNCAPPED = ")]
    assert body.count("max_extra_rows=BAND_EXTRA_UNCAPPED") == 2 and "max_extra_rows=emax" not in body and ns["BAND_EXTRA_UNCAPPED"] >= 1 << 40   # the core's cap is never the adapter's argument
