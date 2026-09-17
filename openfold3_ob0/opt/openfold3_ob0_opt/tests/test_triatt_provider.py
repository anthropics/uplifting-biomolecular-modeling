"""The triangle-attention provider binding (openfold3_ob0_opt.of3_triattn; levers `triatt_provider` on the fast line, `triatt_exact` on the exact
line): the pair core's word grammar, the Router's selection through the core's cell table on the line's TIER word and the call's form
(opt_core.kernels.triattn.select — pure python, no GPU; no kit-side row order), the refusal-by-name path with the cell's named fallback, the census fields,
and the wiring (modes / registry / stack / the cells hook). No CUDA: the Router's `_facts` are given, never taken from tensors here."""
import os

import re

import pytest

from openfold3_ob0_opt import modes, of3_triattn as T3, registry, stack
from opt_core.kernels import triattn as T

H100, A100 = (9, 0), (8, 0)
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OF3_STACK = "torch2.10.0+cu128-cpython-311-x86_64-linux-gnu-sm90"       # the kit's torch stack key: the provider carries a cuda_sm90a prebuilt for it
NO_PREBUILT_STACK = "torch0.0.0+none-cpython-311-x86_64-linux-gnu-sm90"  # a torch ABI the sealed sm_90a extension has no prebuilt for: refused by name (no_prebuilt), the next row serves


def _router(word, stack=OF3_STACK, **kw):
    r = T3.Router("t", word, **kw)
    r._stack_key = stack                                                # what cuda_sm90a.stack_key() returns in the box (torch facts; not importable here)
    return r


def test_pair_spec_grammar():
    assert T3.parse_pair_spec("provider") == "fast" == T3.TIER_PAIR                 # the fast and big lines' word: the tier word, no kit-side row order
    assert T3.parse_pair_spec("provider:k2b") == "k2b"
    assert T3.parse_pair_spec(" provider:exact_headsplit ") == "exact_headsplit"
    assert T3.parse_pair_spec("provider:k2b@m128r2") == "k2b@m128r2"
    for bad in ("", "flash_triattn", "provider:", "prov"):
        with pytest.raises(ValueError):
            T3.parse_pair_spec(bad)


def test_no_kit_side_row_order():
    """The binding carries no row table: the tier word and the call form are the whole request (the cell table decides per card)."""
    for name in ("PREFER", "prefer_for"):
        assert not hasattr(T3, name), name
    assert T3.FORM_KEYPAD == "keypad" and T3.FORM_BIAS == "bias_only"


def _cc_str(cc):
    return cc if isinstance(cc, str) else f"{cc[0]}.{cc[1]}"


@pytest.mark.parametrize("stack,facts", [
    (OF3_STACK, (H100, "bf16", 32, 4, 384, False, "keypad")),           # pairformer / msa / confidence pair stacks on the kit's torch ABI, per key-count bucket
    (OF3_STACK, (H100, "bf16", 32, 4, 511, False, "keypad")),
    (OF3_STACK, (H100, "bf16", 32, 4, 768, False, "keypad")),
    (OF3_STACK, (H100, "bf16", 32, 4, 1152, False, "keypad")),
    (OF3_STACK, (H100, "bf16", 32, 4, 2000, False, "keypad")),
    (NO_PREBUILT_STACK, (H100, "bf16", 32, 4, 384, False, "keypad")),   # a torch ABI the sealed extensions carry no prebuilt for: the cell's next admitted row serves
    (NO_PREBUILT_STACK, (H100, "bf16", 32, 4, 768, False, "keypad")),
    (NO_PREBUILT_STACK, (H100, "bf16", 32, 4, 1152, False, "keypad")),
    (OF3_STACK, (H100, "bf16", 16, 4, 384, False, "keypad")),           # template pair stack (c=64)
    (OF3_STACK, (H100, "bf16", 16, 4, 1152, False, "keypad")),
    (OF3_STACK, (H100, "bf16", 32, 4, 768, False, "bias_only")),        # a call without a key mask reads the cell's bias-only order
    (OF3_STACK, (A100, "bf16", 32, 4, 768, False, "keypad")),           # A100 head dim 32
    (OF3_STACK, (A100, "bf16", 16, 4, 768, False, "keypad")),           # A100 template pair stack
])
def test_pair_router_serves_the_providers_tier_word_row(stack, facts):
    """On word fast the router serves exactly the row the provider's tier word names for the call class and form on this torch stack — a KERNEL
    row (never a stock row) — records the cell it read, and memoises per call class."""
    cc, dt, d, h, n, _grad, form = facts
    r = _router("fast", stack=stack)
    hit = r._select(facts)
    assert hit[0] == "row", hit
    sel = hit[1]
    assert sel.cls in ("fast", "exact") and sel.row in T.ROW_NAMES and sel.row not in ("cueq", "ds4sci", "sdpa"), T.describe(sel)
    want = T.select(_cc_str(cc), dt, d, h, n, word="fast", stack=stack if cc == H100 else None, form=form)
    assert sel.row == want.row and sel.cell == want.cell, (T.describe(sel), T.describe(want))
    assert sel.cell in r.cells and r.cells[sel.cell] == sel.row
    assert r._select(facts) is hit                                      # memoised per facts


def test_the_big_lines_word_is_the_providers_big_tier():
    """`provider:big` (the big lines' pair core) asks the provider's `big` tier word (core >= 0.5.114: this family's big cells are fast's —
    no triangle-attention row carries a memory cost); the kit's core pin floor admits it."""
    from openfold3_ob0_opt import _core_gate
    assert T3.parse_pair_spec("provider:big") == "big" and "big" in T.TIER_WORDS
    pin = _core_gate.read_table(os.path.join(HOME, "opt", "pyproject.toml"), "tool.opt_core")
    assert _core_gate.version_tuple(pin["version"]) >= (0, 5, 114, 0), pin
    for cc, stack in ((H100, OF3_STACK), (A100, None)):
        for d in (32, 16):
            for n in (400, 1200, 2000):
                b = _router("big", stack=stack)._select((cc, "bf16", d, 4, n, False, "keypad"))
                f = _router("fast", stack=stack)._select((cc, "bf16", d, 4, n, False, "keypad"))
                assert b[0] == f[0] == "row" and b[1].row == f[1].row, (cc, d, n, T.describe(b[1]), T.describe(f[1]))


def test_this_stacks_rows_at_the_ladder_shapes():
    """What the tier word serves on this kit's torch ABI at the ladder's key counts (the cell table at this core version; a cell that moves is the
    core's statement, read here so a move is visible in this kit's suite): H100 head dim 32: the sm_90a member up to 511 keys, the native CUDA member from 512; no
    UNCOVERED cell up to 2048 keys on either card at either head dim."""
    for n, row in ((400, "cuda_sm90a"), (511, "cuda_sm90a"), (512, "triattn_native"), (800, "triattn_native"), (1200, "triattn_native"), (2000, "triattn_native")):
        sel = _router("fast")._select((H100, "bf16", 32, 4, n, False, "keypad"))[1]
        assert sel.row == row, (n, T.describe(sel))
    for cc in (H100, A100):
        for d in (32, 16):
            for n in (400, 800, 1200, 2000):
                sel = _router("fast")._select((cc, "bf16", d, 4, n, False, "keypad"))[1]
                assert sel.measured or "beyond_measured" not in (sel.reason or ""), (cc, d, n, T.describe(sel))


def test_exact_router_defers_to_the_providers_exact_word():
    """Word exact: the router serves exactly what the provider's exact word names for the cell — an exact-class kernel row (bitwise vs a named
    stock row) or a stock row by name — on both cards and both head dims; never a tolerance-class row."""
    r = _router("exact")
    for cc in (H100, A100):
        for d, n in ((32, 768), (32, 1152), (32, 1536), (16, 768)):
            s = r._select((cc, "bf16", d, 4, n, False, "keypad"))[1]
            p = T.select(_cc_str(cc), "bf16", d, 4, n, word="exact", form="keypad")
            assert s.row == p.row and s.cls == p.cls and s.cls in ("exact", "stock"), (cc, d, n, T.describe(s))
            if s.cls == "exact":
                assert s.exact_vs in ("cueq", "ds4sci", "sdpa"), T.describe(s)      # bitwise against a named stock row
            else:
                assert s.row in ("cueq", "ds4sci", "sdpa"), T.describe(s)


def test_a_row_that_cannot_serve_is_refused_by_name_with_the_cells_fallback(capsys):
    r = _router("cuda_sm90a", stack=NO_PREBUILT_STACK)                  # the sealed sm_90a kernel asked for by name on a torch ABI it has no prebuilt for
    hit = r._select((H100, "bf16", 32, 4, 768, False, "keypad"))
    assert hit[0] == "fallback" and hit[1].startswith("no_prebuilt:") and hit[2] in T.ROW_NAMES and hit[2] != "cuda_sm90a", hit
    assert "refused by name" in capsys.readouterr().err
    hit = _router("cuda_sm90a")._select((A100, "bf16", 32, 4, 768, False, "keypad"))     # an sm_90a-only row on cc 8.0: refused with a reason, a named row as fallback
    assert hit[0] == "fallback" and isinstance(hit[1], str) and hit[1] and hit[2] in T.ROW_NAMES and hit[2] != "cuda_sm90a", hit


def test_census_fields():
    r = _router("fast", stack=NO_PREBUILT_STACK)
    r._select((H100, "bf16", 32, 4, 384, False, "keypad")); r._bump(r.served, "k2b"); r._bump(r.served, "k2b"); r._bump(r.refused, "smem"); r.fallback_calls = 1
    f = r.fields()
    assert re.match(r"^word=fast rows=k2b:2 fallback=1 fallback:smem=1 cells=9\.0\|bf16\|D32\|H4\|[^|:]+\|fwd:[A-Za-z0-9_]+", f), f   # grammar: word, served rows with counts, fallbacks by reason, cell key -> the row the cell named
    e = _router("exact", prove_exact=True)
    e.proven[((H100, "bf16", 32, 4, 612, False, "keypad"), (1, 612), True)] = True
    assert e.fields().endswith("proven=S612 bits_differ=none"), e.fields()


def test_wiring():
    fast, exact = modes.LINES[("fast", None)], modes.LINES[("exact", "cueq")]
    assert fast.env["OPENFOLD3_OB0_OPT_PAIR_CORE"] == "provider" and "triatt_provider" in fast.levers and fast.levers.index("triatt_provider") == fast.levers.index("triatt_block") + 1
    assert exact.env["OPENFOLD3_OB0_OPT_TRIATT_EXACT"] == "1" and "triatt_exact" in exact.levers and "triatt_exact" not in fast.levers
    assert modes.LEVER_SWITCHES["triatt_provider"] == ("OPENFOLD3_OB0_OPT_PAIR_CORE",) and modes.LEVER_SWITCHES["triatt_exact"] == modes.TRIATT_EXACT_ENVS
    assert modes.LEVERS_OFF_TAKES["triatt_block"] == ("triatt_provider",) and modes.TRIATT_EXACT_ENVS in modes.CELL_FAMILIES
    assert registry.LEVERS["triatt_provider"].tier == registry.TOLERANCE and registry.LEVERS["triatt_exact"].tier == registry.EXACT
    assert "triatt_provider" in stack._PROBES and "triatt_exact" in stack._PROBES
    assert stack.CORE_CELL_LEVERS["triatt_provider"] == ("fpf_triatt_k2b", "flash_triattn") and "triatt_exact" not in stack.CORE_CELL_LEVERS   # the provider is a package module, never a routed top-level name (a routed `triattn` would shadow the sealed package's own tree)
    hook = open(os.path.join(os.path.dirname(__file__), "..", "hooks", "cells", "sitecustomize.py"), encoding="utf-8").read()
    assert '"OPENFOLD3_OB0_OPT_TRIATT_EXACT"' in hook and "triatt_exact.install()" in hook
    # levers off: the provider alone leaves the block on its flash core (the word dropped and required unset); the block takes the provider along
    home = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    from openfold3_ob0_opt.tests import _stubs
    home = _stubs.tree_home()
    r = modes.resolve("fast", home, environ={"MODEL_OPT_LEVERS_OFF": "triatt_provider"}, n_tokens=612)
    assert r.conflicts == [] and "OPENFOLD3_OB0_OPT_PAIR_CORE" not in r.exports and "OPENFOLD3_OB0_OPT_PAIR_CORE" in r.unsets and "triatt_block" in r.levers
    r = modes.resolve("fast", home, environ={"MODEL_OPT_LEVERS_OFF": "triatt_block"}, n_tokens=612)
    assert r.levers_off == ["triatt_block", "triatt_provider"] and r.exports["OPENFOLD3_OB0_OPT_PAIR"] == "trimul_v4:pair_transition"
    r = modes.resolve("exact", home, environ={"MODEL_OPT_LEVERS_OFF": "triatt_exact"}, n_tokens=400)
    assert r.conflicts == [] and "OPENFOLD3_OB0_OPT_TRIATT_EXACT" not in r.exports and "triatt_exact" not in r.levers


def test_the_exact_cell_refuses_by_name_beside_the_trunk_kernels_dispatcher():
    from openfold3_ob0_opt.cells import triatt_exact as C
    prev = dict(C.STATE)
    try:
        C.STATE.update(installed=False, state="off", reason="")
        st = C.install(environ={"OPENFOLD3_OB0_OPT_TRIATT_EXACT": "1", "OF3T_TRIATT": "triton"})
        assert st["state"] == "refused" and st["reason"] == "of3t_triatt_routes" and not C.serving()
        assert "state=refused" in C.census_line()
        assert C.requested({"OPENFOLD3_OB0_OPT_TRIATT_EXACT": "1"}) and not C.requested({})
        with pytest.raises(ValueError):
            C.requested({"OPENFOLD3_OB0_OPT_TRIATT_EXACT": "yes"})
    finally:
        C.STATE.clear(); C.STATE.update(prev)


def test_pair_core_builds_the_block_router_with_the_flash_terminal():
    T3.PAIR.update(router=None, spec=None)
    try:
        r = T3.pair_core("provider:k2b")
        assert isinstance(r, T3.Router) and r.word == "k2b" and r.terminal[0] == T3.TERMINAL_PAIR == "flash_triattn"
        assert T3.pair_core("provider:k2b") is r                             # one router per process; the same spec returns it
        T3.PAIR.update(router=None, spec=None)
        p = T3.pair_core("provider")
        assert p.word == "fast" == T3.TIER_PAIR
    finally:
        T3.PAIR.update(router=None, spec=None)
