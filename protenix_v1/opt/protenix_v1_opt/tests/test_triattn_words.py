"""The triangle-attention provider words: `tricuda` (rides gflash; tolerance class; the provider's `fast` tier word, `big` under --mode big) and
`triexact` (rides gblock; exact class; the provider's `exact` tier word in every mode — the bit-identical row `triattn_exact` where the core's cell
table vouches it for the stack, the library op `cueq` by name everywhere else). The mode table carries them (fast / big: tricuda; exact: triexact),
the ablation grammar names them, the shared core's provider names the ROW per key on each card through the tier word (its measured cells: the kit
asserts the class contract — some row served or a refusal BY NAME answered by the stock statement — never the row of the day), and the report reads
their account — served (row=, served_by=, tier=, exact_stack=, member=), refused-by-name (gated, by design: never partial), error (partial). Each word
rides its own block only (gblock / gflash are exclusive): without its host it serves nothing, `aside=host_absent:<host>`."""
import os
import sys

import pytest

from protenix_v1_opt import ablation as A, kit as K, modes as M, report as R


def test_the_words_are_levers_of_the_grammar_and_the_registry():
    _, levers = K.lever_grammar(None)
    assert "tricuda" in levers and "triexact" in levers and "gblock" in levers and "gflash" in levers
    assert M.LEVERS["tricuda"]["class"] == "tolerance" and M.LEVERS["gblock"]["class"] == "exact" and M.LEVERS["triexact"]["class"] == "exact"
    assert R.STRATEGY_IDS["tricuda"] == "F1.flash_triatt" == R.STRATEGY_IDS["triexact"]
    assert R.lever_impl("triexact") == ("opt_core/kernels/triattn", "core") == R.lever_impl("tricuda")
    assert R.CORE_HOSTS == {"tricuda": ("fast", "gflash"), "triexact": ("exact", "gblock")}


def test_the_modes_carry_the_word_fast_and_big_not_exact():
    ex, fa = M.resolve("exact"), M.resolve("fast")
    assert "gblock" in ex.levers and "tricuda" not in ex.levers and "gflash" not in ex.levers
    assert "tricuda" in fa.levers and "gflash" in fa.levers and "gblock" not in fa.levers
    from protenix_v1_opt import big as B
    assert "tricuda" not in B.BASE_LEVERS_OFF and "+tricuda+" in B.compose()["arm"]          # big keeps the word (the pre-compiled CUDA row)


def test_the_word_is_individually_switchable(monkeypatch):
    monkeypatch.setenv(A.ENV, "tricuda")
    r = M.resolve("fast")
    assert "tricuda" not in r.levers and "gflash" in r.levers and r.ablated == ("tricuda",)
    monkeypatch.setenv(A.ENV, "gflash")                                                     # the host lever alone: the word stays on the arm and rides nothing (report: aside host_absent)
    r = M.resolve("fast")
    assert "gflash" not in r.levers and "tricuda" in r.levers


def _acct(word, counts, host_on=True, sel=None, facts=None, member=None):
    """A kit account (levers_ptx1.describe(): cfg + counts) for one provider word."""
    host = {"tricuda": "gflash", "triexact": "gblock"}[word]
    c = {"core:" + word: dict(counts)}
    if sel: c["coresel:" + word] = {s: 1 for s in sel}
    if facts: c["corefact:" + word] = {f: 1 for f in facts}
    if member: c["coremember:" + word] = dict(member)
    return {"cfg": {"trimul": "exact" if word == "triexact" else "fast", host: bool(host_on), word: True}, "counts": c}


def test_core_evidence_served_refused_by_name_and_error():
    e = R.core_evidence("tricuda", _acct("tricuda", {"cuda_sm90a": 1040}, sel=["9.0|bf16|D32|H4|N800|fwd=>row=cuda_sm90a x_stock=1.89"], facts=["form=mask_bias", "tier=fast"]))
    assert e["served"] == 1040 and not e["gated"] and not e["fallback"] and "aside" not in e and e["row"] == "cuda_sm90a" and e["host"] == "gflash" and e["tier"] == "fast"
    assert e["selections"] == ["9.0|bf16|D32|H4|N800|fwd=>row=cuda_sm90a x_stock=1.89"] and e["facts"] == ["tier=fast", "form=mask_bias"] and "served_by" not in e
    e = R.core_evidence("tricuda", _acct("tricuda", {"cueq:no_cell:bf16_D32": 1040}))          # a key the tier word refused by name -> the stock statement; by design
    assert e["served"] == 0 and e["gated"] == {"cueq:no_cell:bf16_D32": 1040} and e["aside"] == {"state": "aside", "word": "cueq:no_cell:bf16_D32"} and e["row"] == "none"
    e = R.core_evidence("tricuda", _acct("tricuda", {"cuda_sm90a": 1000, "cueq:strides": 40}))
    assert e["served"] == 1000 and e["gated"] == {"cueq:strides": 40} and "aside" not in e and e["tier"] == "fast"
    e = R.core_evidence("tricuda", _acct("tricuda", {"triattn_native": 10}, facts=["tier=big", "form=mask_bias"]))     # --mode big: the account names the tier word the kit bound
    assert e["tier"] == "big" and e["facts"] == ["tier=big", "form=mask_bias"]
    e = R.core_evidence("tricuda", _acct("tricuda", {}))                                        # gflash made no core call (every item gated)
    assert e["aside"]["word"] == "no_core_call"
    e = R.core_evidence("tricuda", _acct("tricuda", {}, host_on=False))                         # gflash ablated: the word rides nothing
    assert e["aside"]["word"] == "host_absent:gflash"
    e = R.core_evidence("tricuda", _acct("tricuda", {"cuda_sm90a": 10, "error:RuntimeError": 2}))
    assert e["fallback"] == {"error:RuntimeError": 2} and "aside" not in e
    e = R.core_evidence("tricuda", {})                                                           # no account (an older lever file): the default tier and host, no core call
    assert e["row"] == "none" and e["tier"] == "fast" and e["host"] == "gflash" and e["aside"]["word"] == "no_core_call"


def test_lever_lines_and_the_exit_rule_read_the_account():
    ev = {"gflash": {"served": 1040, "gated": {}, "fallback": {}, "retried": {}},
          "tricuda": R.core_evidence("tricuda", _acct("tricuda", {"cueq:arch:sm_80": 1040}, facts=["form=mask_bias"]))}
    lines = R.lever_lines(ev, {"mode": "fast", "mode_trimul": "fast", "mode_levers": ["gflash", "tricuda"]})
    tri = [l for l in lines if l.endswith(" lever=tricuda")][0]
    assert "state=skipped reason=aside" in tri and "served=0" in tri and "gated_by=cueq:arch:sm_80:1040" in tri and "aside=cueq:arch:sm_80" in tri and " row=none " in tri and " tier=fast " in tri and "form=mask_bias" in tri and "strategy=F1.flash_triatt" in tri
    assert R.partial_of(ev)[0] == []                                                             # refused by name for every call: by design, not partial
    ev["tricuda"] = R.core_evidence("tricuda", _acct("tricuda", {"cuda_sm90a": 10, "error:RuntimeError": 2}))
    assert R.partial_of(ev)[0] == ["tricuda"]                                                    # an error on a call is a fallback: partial
    tri = [l for l in R.lever_lines(ev, {"mode": "fast", "mode_trimul": "fast", "mode_levers": ["gflash", "tricuda"]}) if l.endswith(" lever=tricuda")][0]
    assert "state=skipped reason=fallback" in tri and "fallback=2" in tri and "fallback_by=error:RuntimeError:2" in tri


def test_kit_evidence_reads_the_words_off_the_carried_account():
    """stack.kit_lever_summary carries cfg + counts: the words' census rides counts (core:/coresel:/corefact:) and reaches the verdict."""
    fast = M.resolve("fast", environ={})
    acct = {"cfg": {"trimul": "fast", **{lv: True for lv in fast.levers}},
            "counts": {"trimul": {"fast": 12}, "triattn": {"gflash": 20}, "transition": {"ttr:C=128": 40}, "core:tricuda": {"cuda_sm90a": 20},
                       "coresel:tricuda": {"9.0|bf16|D32|H4|N800|fwd=>row=cuda_sm90a word=cuda_sm90a class=fast": 1}, "corefact:tricuda": {"prebuilt=torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90:built": 1}}}
    ev = R.kit_evidence(acct, "fast", fast.levers, [{"N_token": 800}], None)
    assert ev["tricuda"]["served"] == 20 and "aside" not in ev["tricuda"] and list(ev).index("tricuda") == list(ev).index("gflash") + 1
    assert ev["gflash"].get("floor") is None                                                     # the kit has no tri-attention size floor: no FLOOR facts
    line = [l for l in R.lever_lines(ev, {"mode": "fast"}) if l.endswith(" lever=tricuda")][0]
    assert " served=20 " in line and " row=cuda_sm90a host=gflash " in line and " tier=fast " in line and "prebuilt=torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90:built" in line and " selections=9.0|bf16|D32|H4|N800|fwd=>row=cuda_sm90a_word=cuda_sm90a_class=fast " in line


def test_the_tier_words_name_a_row_or_refuse_by_name_per_card():
    """The shared core's provider by TIER word (the kit's binding), as CLASS contracts: `exact` / `fast` / `big` name SOME row of the provider at the
    kit's keys on both cards (which row is the provider's measured cell, never asserted here), an exact-tier row is exact-class or the stock op, and a
    class the cells do not carry (5 heads) refuses BY NAME with a named fallback (an fp32 / fp16 stream never reaches the provider: the kit gates it, `stock:dtype`)."""
    T = pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    if "big" not in getattr(T, "TIER_WORDS", ()):
        pytest.skip("opt_core < 0.5.114: the provider has no `big` tier word")
    rows = set(getattr(T, "ROW_NAMES", ())) or None
    for cc in ("9.0", "8.0"):
        for n in (17, 256, 400, 448, 800, 1200, 2048):
            for word in ("fast", "big", "exact"):
                for hd in (32, 16):                                                               # the trunk (4 x 32) and the template pair stack (4 x 16)
                    try:
                        sel = T.select(cc, "bf16", hd, 4, n, word=word, form="mask_bias")
                    except T.Refusal as r:                                                        # a key the tier word cannot serve on this stack: by name, with a fallback row
                        assert r.kind and r.fallback, (cc, n, word, hd, r.kind)
                        continue
                    assert sel.word == word and sel.row and (rows is None or sel.row in rows), (cc, n, word, hd, sel)
                    if word == "exact":
                        assert sel.cls in ("exact", "stock"), (cc, n, sel.row, sel.cls)             # the exact tier: an exact-class row or the stock op by name — never a tolerance row
                    else:
                        assert sel.cls != "exact" or sel.row, (cc, n, sel.row)
    with pytest.raises(T.Refusal) as ri:                                                          # a class no cell carries (5 heads): refused BY NAME with a named fallback
        T.select("9.0", "bf16", 32, 5, 800, word="fast", form="mask_bias")
    assert ri.value.kind and ri.value.fallback


def test_the_provider_selects_by_name_per_card():
    """Row words stay a provider contract (the kit binds tier words only): a row word names its row where the card serves it and otherwise refuses BY
    NAME with a named fallback row; an fp32 stream refuses by dtype."""
    T = pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider (opt_core >= 0.5.24)")
    built = T.stacks_built()
    if built:
        assert T.select("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack=built[0]).row == "cuda_sm90a"
    KIT_CHAIN = tuple(getattr(T, "ROW_NAMES", ("cueq", "k2b", "cuda_sm90a", "triattn_native", "flash", "k2")))   # a refusal names one of the provider's rows as the fallback

    def serves_or_steps_aside(args, word, **kw):
        """CLASS contract of a row word at a key: the provider returns a selection of SOME row, or refuses BY NAME with a fallback the kit's chain serves
        (which of the two a card gets is the provider's table, not this kit's contract)."""
        try:
            sel = T.select(*args, word=word, **kw)
            assert sel.row in T.ROW_NAMES if hasattr(T, "ROW_NAMES") else bool(sel.row), (args, word, sel.row)
            return sel.row, None
        except T.Refusal as r:
            assert r.kind and r.fallback in KIT_CHAIN, (args, word, r.kind, r.fallback)
            return None, r.fallback

    for word in ("cuda_sm90a", "triattn_native"):                                        # cc 8.0: served (an sm_80 build) or refused by name -> the kit's own core; both are fine
        serves_or_steps_aside(("8.0", "bf16", 32, 4, 800), word)
    for word, fb in (("cuda_sm90a", "k2b"),):                                          # an fp32 stream: the bf16 rows refuse by dtype (row contract) -> the kit's core for that word
        with pytest.raises(T.Refusal) as ri:
            T.select("9.0", "fp32", 32, 4, 800, word=word)
        assert ri.value.kind.startswith("dtype") and ri.value.fallback == fb, (word, ri.value.kind, ri.value.fallback)
    row, fb = serves_or_steps_aside(("9.0", "bf16", 32, 4, 800), "cuda_sm90a", stack="torch0.0.0+cu0-cpython-30-none-sm90")   # a stack the row has no prebuilt for: refused by name (or served if the core builds one)
    assert row == "cuda_sm90a" or fb == "k2b"


def test_core_select_caches_the_verdict_and_names_the_refusal(capsys):
    pytest.importorskip("torch", reason="levers_ptx1 imports torch")
    pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    sys.path.insert(0, os.path.dirname(K.levers_file(None)))
    try:
        import levers_ptx1 as L
    except Exception as e:                                                       # protenix / the core's attention layer not importable here
        pytest.skip(f"levers_ptx1 not importable: {e!r}")
    finally:
        sys.path.pop(0)
    L.CORE_SEL.clear(); L.CORE_NOTES.clear()
    for k in ("coresel:tricuda", "corefact:tricuda", "coresel:triexact", "corefact:triexact"):
        L.COUNTS.pop(k, None)
    assert L.core_tier("tricuda") == "fast" and L.core_tier("triexact") == "exact"    # outside a kit mode: the words' default tier words (`big` for tricuda under --mode big)
    r = L.core_select("tricuda", "8.0", "bf16", 32, 4, 1200)                         # cc 8.0: the fast tier names its row for the card (whatever the cell says) or refuses by name
    assert (isinstance(r, L.CoreAside) and r.kind and r.fallback == L.LIBRARY_OP) or (not isinstance(r, L.CoreAside) and r.word == "fast" and r.row)
    assert L.core_select("tricuda", "8.0", "bf16", 32, 4, 1200) is r                 # cached per key: the provider is asked once
    out = capsys.readouterr().out
    assert ("refused by name at cc 8.0" in out) == isinstance(r, L.CoreAside)
    note = L.CORE_NOTES["tricuda"]["bf16:D32:H4:N1200"]
    assert (note.startswith("refused:") and note.endswith("->" + L.LIBRARY_OP)) if isinstance(r, L.CoreAside) else ("row=%s" % r.row) in note
    assert list(L.COUNTS["coresel:tricuda"])[-1].startswith("8.0|bf16|D32|H4|N1200=>")           # mirrored where the report reads it
    d = L.describe_cores()
    assert set(d) == {"tricuda", "triexact"} and d["tricuda"]["tier"] == "fast" and d["triexact"]["tier"] == "exact" and d["tricuda"]["fallback"] == d["triexact"]["fallback"] == L.LIBRARY_OP
    assert "tier=fast" in L.COUNTS["corefact:tricuda"] and "form=mask_bias" in L.COUNTS["corefact:tricuda"]


def test_tricuda_counts_every_provider_row_as_served_and_names_the_top_one():
    e = R.core_evidence("tricuda", _acct("tricuda", {"triattn_native": 900, "cuda_sm90a": 100, "cueq:no_cell": 40}))
    assert e["served"] == 1000 and e["row"] == "triattn_native" and e["served_by"] == {"triattn_native": 900, "cuda_sm90a": 100} and e["gated"] == {"cueq:no_cell": 40}
    e = R.core_evidence("tricuda", _acct("tricuda", {"k2b": 1000}))                             # a Triton row the fast tier names as a core row (9.0, 401-511 tokens): served like any row
    assert e["served"] == 1000 and e["row"] == "k2b" and "served_by" not in e
    T = pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's tri-attention provider")
    sel = T.select("9.0", "bf16", 32, 4, 1200, word="triattn_native")                    # the row word names the row (row-word contract)
    assert sel.row == "triattn_native"
    try:                                                                            # cc 8.0: served (the row's sm_80 build) or refused BY NAME with a fallback the kit's chain serves — the provider's table decides which
        assert T.select("8.0", "bf16", 32, 4, 1200, word="triattn_native").row == "triattn_native"
    except T.Refusal as r:
        assert r.kind and r.fallback in ("cuda_sm90a", "k2b", "cueq")


# ---- gflash's attention core = the provider's tier word per key under `tricuda`, the fused block's own default core with the word off;
# CLASS contracts (the row per key is the core's cell: never asserted by value).
def _levers():
    pytest.importorskip("torch", reason="levers_ptx1 imports torch")
    sys.path.insert(0, os.path.dirname(K.levers_file(None)))
    try:
        import levers_ptx1 as L
    except Exception as e:
        pytest.skip(f"levers_ptx1 not importable: {e!r}")
    finally:
        sys.path.pop(0)
    return L


def test_the_kit_carries_no_tri_attention_floor_row_word_or_cell(monkeypatch):
    L = _levers()
    for gone in ("GFLASH_CORE", "GFLASH_DEFAULT_CORE", "gflash_table_core", "_k2b_core", "TRIATTN_FLOOR"):
        assert not hasattr(L, gone), gone                                              # no kit K2B cell word, no kit default-core table, no kit size floor
    assert L.TRIATTN_GATE_TOKENS == 16 and L.TRIATTN_CEILING_TOKENS == 2048            # upstream-structural: its small-input torch route and its row-chunking point / the provider's last cell
    assert set(L.CORE_WORDS) == {"tricuda", "triexact"} and all(set(v) == {"host", "tier"} for v in L.CORE_WORDS.values())   # tier words only: no row names in the word table
    assert L.CORE_WORDS["tricuda"] == {"host": "gflash", "tier": "fast"} and L.CORE_WORDS["triexact"] == {"host": "gblock", "tier": "exact"} and L.LIBRARY_OP == "cueq"
    assert L.GBLOCK_PROLOGUE == "fpf" and L.GFLASH_PROLOGUE == "lnl"                    # the two block constructions (pair_fused impl words), not kernel rows
    st = type(sys)("protenix_v1_opt.stack"); st.status = lambda: {"mode": "big"}
    monkeypatch.setitem(sys.modules, "protenix_v1_opt.stack", st)
    assert L.core_tier("tricuda") == "big" and L.core_tier("triexact") == "exact"     # --mode big binds the provider's `big` word for tricuda; triexact binds `exact` in every mode
    st.status = lambda: {"mode": "fast"}
    assert L.core_tier("tricuda") == "fast" and L.core_tier("triexact") == "exact"


def test_gflash_core_without_the_word_is_the_fused_blocks_own_default(monkeypatch):
    L = _levers()
    PF = pytest.importorskip("opt_core.attn.pair_fused", reason="the shared core's fused block")
    import torch
    L.COUNTS.pop("gflash_core", None)
    monkeypatch.setitem(L.CFG, "tricuda", False)
    x = torch.zeros(1, 8, 8, 128)
    assert L.gflash_core(x, 4, 32, 800) == getattr(PF, "DEFAULT_CORE", "default")       # the word off: the core's own default core word, counted
    assert L.COUNTS["gflash_core"] == {getattr(PF, "DEFAULT_CORE", "default"): 1}


def test_gflash_core_under_the_word_serves_the_tier_row_or_the_stock_statement_by_name(monkeypatch):
    L = _levers()
    pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    import torch
    L.COUNTS.pop("gflash_core", None); L.CORE_SEL.clear()
    monkeypatch.setitem(L.CFG, "tricuda", True)
    monkeypatch.setattr(L, "device_cc", lambda dev: "9.0")
    x = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    core = L.gflash_core(x, 4, 32, 800)
    assert callable(core)                                                               # a provider row bound with its cached Selection, or the stock statement for a refused key
    (tok, n), = L.COUNTS["gflash_core"].items()
    assert n == 1 and (tok.startswith("tricuda:") or tok.startswith(L.LIBRARY_OP + ":")), tok
    monkeypatch.setattr(L, "_tier_select", lambda *a, **k: (_ for _ in ()).throw(L._T().Refusal("no_cell:test", "fast", "cueq")))
    L.CORE_SEL.clear(); L.COUNTS.pop("gflash_core", None)
    assert L.gflash_core(x, 4, 32, 801) is L._cueq_core                                 # a key the tier word refuses by name: the stock cuEquivariance statement inside the block
    assert L.COUNTS["gflash_core"] == {"cueq:no_cell:test": 1} and L.COUNTS["core:tricuda"]["cueq:no_cell:test"] >= 1


def test_the_gflash_lever_line_names_the_cores_served():
    acct = {"cfg": {"trimul": "fast", "gflash": True, "tricuda": True}, "gates": {"triattn_gate_tokens": 16, "triattn_ceiling_tokens": 2048},
            "counts": {"triattn": {"gflash": 960}, "gflash_core": {"tricuda:triattn_native": 900, "tricuda:cuda_sm90a": 40, "cueq:no_cell": 20}}}
    e = R.kit_evidence(acct, "fast", ["gflash", "tricuda"], [{"name": "x", "N_token": 800}], None)["gflash"]
    assert e["served"] == 960 and e["facts"] == ["cores=tricuda:triattn_native:900,tricuda:cuda_sm90a:40,cueq:no_cell:20"] and e.get("floor") is None
    assert e["gate"]["state"] == "served" and e["gate"]["word"] == "n<=16"                                    # upstream's small-input route, read against the items
    line = [l for l in R.lever_lines({"gflash": e}, {"mode": "fast", "mode_trimul": "fast", "mode_levers": ["gflash", "tricuda"]}) if l.endswith(" lever=gflash")][0]
    assert " cores=tricuda:triattn_native:900,tricuda:cuda_sm90a:40,cueq:no_cell:20 " in line + " " and "min_tokens=" not in line and " floor=" not in line, line
    tiny = R.kit_evidence({"cfg": acct["cfg"], "gates": acct["gates"], "counts": {"triattn": {"stock:gate": 96}}}, "fast", ["gflash", "tricuda"], [{"name": "p", "N_token": 12}], None)
    assert tiny["gflash"]["served"] == 0 and tiny["gflash"]["gate"]["state"] == "gate-off"                    # every item at or below 16 tokens: upstream's torch route by design, never partial
    assert not {"gflash", "tricuda"} & set(R.partial_of(tiny)[0])                                          # gflash gate-off + tricuda aside (no core call): accounted, never partial


# ---- `triexact`: gblock's attention core = the provider's `exact` tier word per key; the stock statement without the word. CLASS contracts: the
# exact tier names an exact-class row the core vouches for this stack or the library op (`cueq`) by name — on a stack without a vouch (this one,
# and the kit's pinned library build) the row is `cueq` at every key.
def test_the_exact_mode_carries_triexact_on_gblock_and_no_other_mode_does():
    ex, fa, bg = M.resolve("exact"), M.resolve("fast"), M.resolve("big")
    assert "triexact" in ex.levers and "gblock" in ex.levers and ex.levers.index("triexact") == ex.levers.index("gblock") + 1
    assert "+gblock+triexact+" in M.KIT_MODES["exact"].arm
    for r in (fa, bg):
        assert "triexact" not in r.levers and "gblock" not in r.levers and "tricuda" in r.levers          # fast / big: gflash + tricuda, untouched
    assert M.KIT_MODES["off"].arm == M.STOCK_ARM
    assert "triexact" in R.KNOB_LEVERS["triatt_kernel"] and "triexact" in R.KNOB_LEVERS["dtype"]        # --triatt_kernel torch / --dtype route the run off the word like its host


def test_triexact_is_individually_switchable_and_refused_outside_exact(monkeypatch):
    monkeypatch.setenv(A.ENV, "triexact")
    r = M.resolve("exact")
    assert "triexact" not in r.levers and "gblock" in r.levers and r.ablated == ("triexact",) and "+triexact" not in r.arm
    monkeypatch.setenv(A.ENV, "gblock")                                                     # the host lever alone: the word stays on the arm and rides nothing (report: aside host_absent:gblock)
    r = M.resolve("exact")
    assert "gblock" not in r.levers and "triexact" in r.levers
    monkeypatch.setenv(A.ENV, "triexact")                                                   # a lever of exact, not of fast / big: refused by name there
    with pytest.raises(RuntimeError, match=r"\['triexact'\] not a lever of mode fast"):
        M.resolve("fast")
    monkeypatch.setenv(A.ENV, "tricuda")                                                    # and the fast word is not a lever of exact
    with pytest.raises(RuntimeError, match=r"\['tricuda'\] not a lever of mode exact"):
        M.resolve("exact")


def test_triexact_core_evidence_served_aside_member_and_line():
    sel = ["9.0|bf16|D32|H4|N384=>row=cueq word=exact class=stock cell=9.0|bf16|D32|H4|N<=400|fwd measured=yes"]
    facts = ["tier=exact", "exact_stack=9.0|torch2.7.1+cu126|cueq0.8.0"]
    e = R.core_evidence("triexact", _acct("triexact", {"cueq": 960}, sel=sel, facts=facts, member={"served": 0, "calls": 0}))   # the pinned stack: no vouched cell -> the library op served every call through the provider
    assert e["served"] == 960 and e["row"] == "cueq" and e["host"] == "gblock" and e["tier"] == "exact" and not e["gated"] and not e["fallback"] and "aside" not in e
    assert e["facts"] == ["tier=exact", "exact_stack=9.0|torch2.7.1+cu126|cueq0.8.0", "member=0/0"] and e["selections"] == sel
    ev = {"gblock": {"served": 960, "gated": {}, "fallback": {}, "retried": {}}, "triexact": e}
    line = [l for l in R.lever_lines(ev, {"mode": "exact", "mode_trimul": "exact", "mode_levers": ["gblock", "triexact"]}) if l.endswith(" lever=triexact")][0]
    assert line == ("[protenix-v1-opt] LEVER name=F1.flash_triatt state=on impl=opt_core/kernels/triattn origin=core strategy=F1.flash_triatt served=960 row=cueq host=gblock "
                    "tier=exact exact_stack=9.0|torch2.7.1+cu126|cueq0.8.0 member=0/0 selections=9.0|bf16|D32|H4|N384=>row=cueq_word=exact_class=stock_cell=9.0|bf16|D32|H4|N<=400|fwd_measured=yes lever=triexact"), line
    assert R.partial_of(ev)[0] == []
    e = R.core_evidence("triexact", _acct("triexact", {"triattn_exact": 900, "cueq": 60}, member={"served": 880, "calls": 900, "refused:small_s": 20}))   # a vouched stack: the exact row above the library's threshold, the library op below it
    assert e["served"] == 960 and e["row"] == "triattn_exact" and e["served_by"] == {"cueq": 60, "triattn_exact": 900} and e["facts"] == ["tier=exact", "member=880/900", "member_refused=small_s:20"]
    line = [l for l in R.lever_lines({"gblock": ev["gblock"], "triexact": e}, {"mode": "exact"}) if l.endswith(" lever=triexact")][0]
    assert " state=on " in line and " served=960 row=triattn_exact served_by=cueq:60,triattn_exact:900 host=gblock tier=exact member=880/900 member_refused=small_s:20 lever=triexact" in line, line
    e = R.core_evidence("triexact", _acct("triexact", {}, host_on=False))                       # gblock ablated (or a hand arm without it): the word rides nothing — by name, never partial
    assert e["served"] == 0 and e["aside"] == {"state": "aside", "word": "host_absent:gblock"}
    e = R.core_evidence("triexact", _acct("triexact", {}))                                      # gblock made no core call (every item gated / above the ceiling)
    assert e["aside"]["word"] == "no_core_call" and e["row"] == "none"
    e = R.core_evidence("triexact", _acct("triexact", {"cueq:needs_stock": 12}))                 # a structural refusal by name per call: the stock statement, accounted
    assert e["gated"] == {"cueq:needs_stock": 12} and e["aside"]["word"] == "cueq:needs_stock"
    e = R.core_evidence("triexact", _acct("triexact", {"cueq": 10, "error:RuntimeError": 2}))
    assert e["fallback"] == {"error:RuntimeError": 2} and "aside" not in e and R.partial_of({"triexact": e})[0] == ["triexact"]
    e = R.core_evidence("triexact", {})                                                          # no account: the default tier and host, no core call
    assert e["row"] == "none" and e["tier"] == "exact" and e["host"] == "gblock" and e["aside"]["word"] == "no_core_call"


def test_kit_evidence_reads_triexact_after_gblock():
    exact = M.resolve("exact", environ={})
    acct = {"cfg": {"trimul": "exact", **{lv: True for lv in exact.levers}},
            "counts": {"trimul": {"exact": 12}, "triattn": {"gblock": 20}, "transition": {"xtr:C=128": 40}, "core:triexact": {"cueq": 20},
                       "coresel:triexact": {"9.0|bf16|D32|H4|N800=>row=cueq word=exact class=stock": 1}, "corefact:triexact": {"tier=exact": 1, "exact_stack=9.0|torch2.7.1+cu126|cueq0.8.0": 1},
                       "coremember:triexact": {"served": 0, "calls": 0}}}
    ev = R.kit_evidence(acct, "exact", exact.levers, [{"N_token": 800}], None)
    assert ev["triexact"]["served"] == 20 and ev["triexact"]["row"] == "cueq" and "aside" not in ev["triexact"] and list(ev).index("triexact") == list(ev).index("gblock") + 1
    assert "tricuda" not in ev
    line = [l for l in R.lever_lines(ev, {"mode": "exact"}) if l.endswith(" lever=triexact")][0]
    assert " served=20 row=cueq host=gblock tier=exact exact_stack=9.0|torch2.7.1+cu126|cueq0.8.0 member=0/0 selections=9.0|bf16|D32|H4|N800=>row=cueq_word=exact_class=stock lever=triexact" in line + "", line
    fast = M.resolve("fast", environ={})
    assert "triexact" not in R.kit_evidence({"cfg": {"trimul": "fast"}, "counts": {}}, "fast", fast.levers, [], None)   # not a lever of the fast arm: no evidence row (its LEVER line reads off not_in_mode)


def test_triexact_core_select_binds_the_exact_tier_and_names_its_vouch_key(capsys):
    L = _levers()
    T = pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    L.CORE_SEL.clear(); L.CORE_NOTES.pop("triexact", None)
    for k in ("coresel:triexact", "corefact:triexact"):
        L.COUNTS.pop(k, None)
    for cc, n in (("9.0", 800), ("8.0", 1200), ("9.0", 384)):
        r = L.core_select("triexact", cc, "bf16", 32, 4, n)                            # the exact tier: an exact-class row the core vouches for THIS stack, else the library op by name — a class contract
        if isinstance(r, L.CoreAside):
            assert r.kind and r.fallback == L.LIBRARY_OP
            continue
        assert r.word == "exact" and r.cls in ("exact", "stock") and (r.cls == "stock") == (r.row == L.LIBRARY_OP), (cc, n, T.describe(r))
        vouched = T.exact_stack_key(tuple(T.norm_cc(cc))) in (T.rows().get(r.row, {}).get("vouched_on") or []) or any(
            T.exact_stack_key(tuple(T.norm_cc(cc))) in v for v in ((T.cells().get(r.cell) or {}).get("vouched_on") or {}).values())
        assert r.row == L.LIBRARY_OP or vouched, (cc, n, r.row)                          # a kernel row only where the cell table records this process's vouch key
        assert L.core_select("triexact", cc, "bf16", 32, 4, n) is r                       # cached per key
        assert ("row=%s" % r.row) in L.CORE_NOTES["triexact"]["bf16:D32:H4:N%d" % n]
    facts = L.COUNTS["corefact:triexact"]
    assert "tier=exact" in facts and any(f.startswith("exact_stack=") for f in facts) and not any(f.startswith("form=") or f.startswith("prebuilt=") for f in facts)
    assert any(f == "exact_stack=%s" % T.exact_stack_key((9, 0)) for f in facts)          # the vouch key of this process, as the provider spells it
    assert all(k.split("=>", 1)[0].count("|") == 4 for k in L.COUNTS["coresel:triexact"])  # <cc>|<dtype>|D|H|N=><note>, where the report reads it


class _FakeFace(object):
    """opt_core.kernels.triattn, reduced to what _provider_core calls."""
    class Refusal(Exception):
        def __init__(self, kind, row=None, fallback=None, detail=""):
            Exception.__init__(self, kind); self.kind, self.row, self.fallback = kind, row, fallback

    def __init__(self, refuse=None, error=None):
        self.calls, self.refuse, self.error = [], refuse, error

    def triangle_attention(self, q, k, v, bias, mask=None, scale=None, **kw):
        self.calls.append(dict(kw, mask=mask, scale=scale))
        if self.refuse:
            raise _FakeFace.Refusal(self.refuse, "triattn_exact", "cueq")
        if self.error:
            raise self.error
        return q + 1


def test_gblock_core_is_the_stock_statement_without_the_word_and_the_exact_tier_word_with_it(monkeypatch):
    L = _levers()
    pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    import torch
    x = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    monkeypatch.setitem(L.CFG, "triexact", False)
    L.COUNTS.pop("core:triexact", None)
    assert L.gblock_core(x, 4, 32, 800) is L._cueq_core and "core:triexact" not in L.COUNTS and "gblock_core" not in L.COUNTS   # the word off: gblock exactly as before, no census
    monkeypatch.setitem(L.CFG, "triexact", True)
    monkeypatch.setattr(L, "device_cc", lambda dev: "9.0")
    L.CORE_SEL.clear()
    core = L.gblock_core(x, 4, 32, 800)
    sel = L.CORE_SEL[("triexact", "exact", "9.0", "bf16", 32, 4, 800, "fwd")]
    if isinstance(sel, L.CoreAside):                                                   # a key the tier word refused by name: the stock statement directly, counted
        assert core is L._cueq_core and L.COUNTS["core:triexact"] == {"cueq:%s" % sel.kind: 1}
        return
    assert callable(core) and core is not L._cueq_core                                 # the provider face bound with the cached Selection
    fake = _FakeFace(); monkeypatch.setattr(L, "_T", lambda: fake)
    q = torch.zeros(1, 8, 4, 8, 32, dtype=torch.bfloat16); bias = torch.zeros(1, 1, 4, 8, 8); mask5 = torch.ones(1, 8, 1, 1, 8, dtype=torch.bool)
    out = core(q, q, q, bias, mask5, 0.176)
    assert torch.equal(out, q + 1) and L.COUNTS["core:triexact"] == {sel.row: 1}
    (call,) = fake.calls
    assert call["word"] == "exact" and call["selection"] is sel and call["stock"] is L._stock_triattn and call["form"] == L.TRIATTN_FORM and call["mask"] is mask5 and call["scale"] == 0.176
    stock_calls = []
    monkeypatch.setattr(L, "_cueq_core", lambda *a: stock_calls.append(a) or "stock")
    fake.refuse = "layout"                                                             # a structural refusal by name on the call: the stock statement answers it, counted cueq:<kind>
    assert core(q, q, q, bias, mask5, 0.176) == "stock" and L.COUNTS["core:triexact"]["cueq:layout"] == 1 and len(stock_calls) == 1
    fake.refuse, fake.error = None, RuntimeError("boom")                                # any other error: counted error:<Type> (the report reads it as partial), the stock statement answers
    assert core(q, q, q, bias, mask5, 0.176) == "stock" and L.COUNTS["core:triexact"]["error:RuntimeError"] == 1
    assert L._stock_triattn.__defaults__ == (None, None)                                # the provider convention: f(q, k, v, bias, mask=None, scale=None)
    L.COUNTS.pop("core:triexact", None); L.CORE_SEL.clear()


def test_gblock_core_binds_the_stock_statement_for_a_key_the_word_refuses(monkeypatch):
    L = _levers()
    T = pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    import torch
    monkeypatch.setitem(L.CFG, "triexact", True)
    monkeypatch.setattr(L, "device_cc", lambda dev: "9.0")
    monkeypatch.setattr(L, "_tier_select", lambda *a, **k: (_ for _ in ()).throw(T.Refusal("needs_stock", "cueq", "cueq")))
    L.CORE_SEL.clear(); L.COUNTS.pop("core:triexact", None); L.COUNTS.pop("gblock_core", None)
    x = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    assert L.gblock_core(x, 4, 32, 801) is L._cueq_core
    assert L.COUNTS["core:triexact"] == {"cueq:needs_stock": 1} and L.CORE_NOTES["triexact"]["bf16:D32:H4:N801"] == "refused:needs_stock->cueq"
    assert "gblock_core" not in L.COUNTS                                               # the exact block's core choice adds no census family of its own (gblock's LEVER line is unchanged)
    L.COUNTS.pop("core:triexact", None); L.CORE_SEL.clear(); L.CORE_NOTES.pop("triexact", None)


def test_the_template_pair_stack_takes_the_same_core_word_through_the_block(monkeypatch):
    """tmpl_triatt sends the template Pairformer's c=64 triangle attention (4 heads x 32) through the SAME block construction as the trunk lever, so
    under gblock its core is gblock_core's: the exact tier word under triexact (keyed like the trunk's calls, its own row length), the stock statement
    without it — no template twin of the word."""
    L = _levers()
    pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    import torch
    seen = []
    monkeypatch.setattr(L, "core_select", lambda word, *key: seen.append((word,) + key) or L.CoreAside("test", L.LIBRARY_OP))
    monkeypatch.setattr(L, "device_cc", lambda dev: "9.0")
    x64 = torch.zeros(1, 40, 40, 64, dtype=torch.bfloat16)                                # a template pair tensor (c = 64)
    monkeypatch.setitem(L.CFG, "triexact", False)
    assert L.gblock_core(x64, 4, 32, 40) is L._cueq_core and seen == []
    monkeypatch.setitem(L.CFG, "triexact", True)
    assert L.gblock_core(x64, 4, 32, 40) is L._cueq_core and seen == [("triexact", "9.0", "bf16", 32, 4, 40, "fwd")]   # the word asked at the template call's own key (D32 H4, its row length)
    assert "tmpl_triexact" not in K.lever_grammar(None)[1] and "tmpl_triexact" not in M.LEVERS                          # no template twin: the template lever rides the block's core word
    L.COUNTS.pop("core:triexact", None)


def test_core_member_census_mirrors_the_exact_members_counts(monkeypatch):
    L = _levers()
    EM = pytest.importorskip("opt_core.kernels.triattn.exact_member", reason="the shared core's exact member (opt_core >= 0.5.225)")
    L.COUNTS.pop("coremember:triexact", None)
    monkeypatch.setitem(L.CFG, "triexact", False)
    assert L.core_member_census("triexact") == {} and L.core_member_census("tricuda") == {} and "coremember:triexact" not in L.COUNTS   # off / a fast word: nothing mirrored
    monkeypatch.setitem(L.CFG, "triexact", True)
    assert L.core_member_census("triexact") == {"served": 0, "calls": 0} and L.COUNTS["coremember:triexact"] == {"served": 0, "calls": 0}
    monkeypatch.setattr(EM, "counts", lambda: {"served": 880, "refused": {"small_s": 20, "no_proven_cell": 0}, "calls": 900, "installed": None})
    assert L.core_member_census("triexact") == {"served": 880, "calls": 900, "refused:small_s": 20}
    monkeypatch.setitem(L.CFG, "gblock", True)
    d = L.describe_cores()["triexact"]
    assert d["requested"] is True and d["host"] == "gblock" and d["host_on"] is True and d["tier"] == "exact" and d["member"] == {"served": 880, "calls": 900, "refused:small_s": 20}
    assert "member" not in L.describe_cores()["tricuda"]
    acct = L.describe()                                                                    # the kit account: the census rides `counts` (stack.kit_lever_summary carries counts, not cores)
    assert acct["counts"]["coremember:triexact"] == {"served": 880, "calls": 900, "refused:small_s": 20}
    e = R.core_evidence("triexact", acct)
    assert "member=880/900" in e["facts"] and "member_refused=small_s:20" in e["facts"]
    L.COUNTS.pop("coremember:triexact", None)


def test_each_core_word_rides_its_own_block_only(monkeypatch):
    """gblock / gflash are exclusive (gflash wins in _tri_forward); tricuda is asked by gflash_core only and triexact by gblock_core only: a word whose
    host is absent serves nothing and its line says so by name (aside=host_absent:<host>) — the kit's refusal for gflash+triexact / gblock+tricuda arms."""
    L = _levers()
    pytest.importorskip("opt_core.kernels.triattn", reason="the shared core's provider")
    import torch
    asked = []
    monkeypatch.setattr(L, "core_select", lambda word, *key: asked.append(word) or L.CoreAside("test", L.LIBRARY_OP))
    monkeypatch.setattr(L, "device_cc", lambda dev: "9.0")
    x = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    monkeypatch.setitem(L.CFG, "triexact", True); monkeypatch.setitem(L.CFG, "tricuda", False)
    L.COUNTS.pop("gflash_core", None)
    L.gflash_core(x, 4, 32, 800)                                                            # gflash + triexact: the flash block never asks the exact word
    assert asked == [] and "core:triexact" not in L.COUNTS
    monkeypatch.setitem(L.CFG, "triexact", False); monkeypatch.setitem(L.CFG, "tricuda", True)
    assert L.gblock_core(x, 4, 32, 800) is L._cueq_core and asked == []                     # gblock + tricuda: the exact block never asks the fast word
    L.COUNTS.pop("gflash_core", None); L.COUNTS.pop("core:triexact", None); L.COUNTS.pop("core:tricuda", None)
    for word, host, trimul in (("triexact", "gblock", "exact"), ("tricuda", "gflash", "fast")):
        other = {"gblock": "gflash", "gflash": "gblock"}[host]
        e = R.core_evidence(word, {"cfg": {"trimul": trimul, host: False, other: True, word: True}, "counts": {}})
        assert e["served"] == 0 and e["aside"] == {"state": "aside", "word": "host_absent:%s" % host}
        line = [l for l in R.lever_lines({word: e}, {"mode": trimul}) if l.endswith(" lever=" + word)][0]
        assert " state=skipped reason=aside " in line and (" aside=host_absent:%s " % host) in line
