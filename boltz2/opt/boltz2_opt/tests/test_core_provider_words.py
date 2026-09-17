"""The core's unified triangle-attention / triangle-multiplication providers (opt_core.kernels.triattn / .trimul, CORE 0.5.24-0.5.26) reached BY WORD
through this kit's existing mechanisms — the PAIRFUSE driver's per-site picks (`BOLTZ_PAIRFUSE=<residency>,<site>=core.<word>`), the fused block's
variant (`BOLTZ_PAIRBLOCK=core.<word>`), the TriMul class lever's provider / ceiling words (`BOLTZ_FPF_TRIMUL_PROVIDER`, `BOLTZ_FPF_TRIMUL_MAX_TOKENS`).
CPU: word grammar, class truth, refusals by name, the defaults' CLASS contracts (the TriMul site binds the provider's tier word: a tolerance-class
forward row, whichever its cell table measured; a row by name is honoured), the ceiling's fallback word.  The numerics are the providers' own
tests (common/opt_core) and the kit's CHANGES 0.3.6 measurements."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, OPT)
sys.path.insert(0, os.path.join(OPT, "forward", "pairfuse"))
import boltz2_opt.pairblock as PB   # noqa: E402
import boltz2_opt.trimul as TMA     # noqa: E402
from opt_core.kernels import triattn as KA, trimul as KT   # noqa: E402


def test_pairblock_core_words_are_the_faces_words_and_classed():
    assert PB.variant({PB.SWITCH: "cueq"}) == "cueq" and PB.variant({PB.SWITCH: "k2b"}) == "k2b"
    assert PB.variant({PB.SWITCH: "core.k2"}) == "core.k2"
    assert PB.variant({PB.SWITCH: "core.exact_headsplit"}) == "core.exact_headsplit"
    assert PB.variant({PB.SWITCH: "core.nope"}) is None and PB.variant({PB.SWITCH: "core."}) is None and PB.variant({}) is None
    assert PB.core_word("core.k2b@m128r2") == "k2b@m128r2"                 # a launch-setting word of the face rides along
    for w in KA.ROW_NAMES + KA.TIER_WORDS:
        assert PB.core_word("core." + w) == w
    assert PB.levers_of("cueq") == ("pairblock",) and PB.levers_of("k2b") == ("pairblock", "flash_triattn")
    assert PB.levers_of("core.exact_headsplit") == ("pairblock",) and PB.levers_of("core.exact") == ("pairblock",)
    assert PB.levers_of("core.k2") == ("pairblock", "flash_triattn") and PB.levers_of("core.fast") == ("pairblock", "flash_triattn")
    assert PB.levers_of("core.nope") == ()


def test_pairblock_core_callable_is_memoised_and_refuses_by_name(monkeypatch):
    f1, f2 = PB.core_callable("k2"), PB.core_callable("k2")
    assert f1 is f2 and f1.__name__ == "triattn_core_k2"

    def boom(*a, **k):
        raise KA.Refusal("no_prebuilt:test", "cuda_sm90a", "k2b")
    monkeypatch.setattr(KA, "triangle_attention", boom)
    PB._CORE_CALLABLES.pop("cuda_sm90a", None)
    with pytest.raises(PB.CoreRefused) as e:
        PB.core_callable("cuda_sm90a")(None, None, None, None, None, 1.0)
    assert (e.value.row, e.value.kind) == ("cuda_sm90a", "no_prebuilt:test") and str(e.value) == "core:cuda_sm90a:no_prebuilt:test"
    PB._CORE_CALLABLES.pop("cuda_sm90a", None)


def test_pairfuse_word_grammar_takes_core_picks_and_names_unknown_ones():
    import bz_pairfuse as D
    r0, p0 = D.parse_word("bf16")
    assert (r0, p0) == ("bf16", D.default_picks("bf16")) and set(p0) == {"trimul", "triatt", "transition", "pairbias"}
    assert p0["triatt"] == "core.fast" and D.core_word(p0["triatt"]) == "fast" and "fast" in KA.TIER_WORDS   # the bf16 residency word binds the TriangleAttention site to the core provider's
    for cc in ((8, 0), (9, 0)):                                                           # `fast` TIER word — class-true: whichever tolerance-class row its measured cell names per card and N
        for n in (256, 400, 512, 800, 1200):
            sel = KA.select(cc, "bf16", 32, 4, n, word="fast")
            assert sel.word == "fast" and KA.table()["rows"][sel.row]["class"] == "fast", (cc, n, sel)
    assert set(D.TIER_PICK_WORDS) == {"exact", "fast", "big"} and D.parse_word("bf16,triatt=core.big")[1]["triatt"] == "core.big"   # the tier vocabulary is a word of the grammar on every core version
    assert D.parse_word("bf16,triatt=k2b")[1]["triatt"] == "k2b" and D.parse_word("bf16,triatt=default")[1]["triatt"] == "default"        # the available words by name
    assert D.core_word(D.DEFAULT_PICKS["triatt"]) == "fast" and D.parse_word("fp32")[1]["triatt"] == D.DEFAULT_PICKS["triatt"] and D.default_picks(None) == D.DEFAULT_PICKS == D.default_picks("fp32")
    from boltz2_opt import modes as MZ
    rowpicks = {m: D.parse_word(MZ.MODES[m]["env"]["BOLTZ_PAIRFUSE"])[1] for m in ("fast", "big")}
    tiered = {k for k, v in rowpicks["fast"].items() if D.core_word(v) is not None}   # the sites a row binds to a core provider by word
    assert {"trimul", "triatt"} <= tiered and {k: v for k, v in rowpicks["fast"].items() if k not in tiered} == {k: v for k, v in rowpicks["big"].items() if k not in tiered}, "big's pair-track word is fast's but for the TIER words"
    assert {m: D.core_word(rowpicks[m]["triatt"]) for m in ("fast", "big")} == {"fast": "fast", "big": "big"}, "each ROW binds the TriangleAttention site to the core provider's TIER word of its own tier"
    tws = {m: D.core_word(rowpicks[m]["trimul"]) for m in ("fast", "big")}             # each ROW binds the TriMul site to the core provider's TIER word of its own tier (0.3.19): fast the
    assert tws == {"fast": "fast", "big": "big"} and set(tws.values()) <= set(KT.TIER_WORDS)   # provider's `fast` word, big its `big` word — whichever tolerance-class rows its table measured
    assert D.core_word(D.DEFAULT_PICKS["trimul"]) is None and "v4" in D.PROVIDERS["trimul"]   # the driver's bare default stays its own v4 binding (the fp32 residency word's route)
    fast_rows = {r for r in KT.ROW_NAMES if KT.table()["rows"].get(r, {}).get("class") == "fast"}
    for tw in tws.values():
        for cc in ((9, 0), (8, 0)):
            for n in (400, 800, 1200):
                for d in ("outgoing", "incoming"):
                    sel = KT.select(cc, "bf16", 128, 128, n, d, word=tw)
                    assert sel.word == tw and sel.row in fast_rows and not sel.backward, (tw, cc, n, d, sel)
                    assert KT.select(cc, "bf16", 128, 128, n, d, word="v4").row == "v4"     # a row by name is exactly that row
    assert D.parse_word("bf16,trimul=v4")[1]["trimul"] == "v4" and D.parse_word("bf16", environ={"BOLTZ_PAIRFUSE_TRIMUL": "v4"})[1]["trimul"] == "v4"
    assert D.parse_word("bf16,trimul=core.v4")[1]["trimul"] == "core.v4" and D.parse_word("bf16,trimul=core.big")[1]["trimul"] == "core.big"
    r, p = D.parse_word("bf16,triatt=core.k2,trimul=core.tmk3_fast")
    assert r == "bf16" and p["triatt"] == "core.k2" and p["trimul"] == "core.tmk3_fast" and p["transition"] == "fpf"
    assert D.parse_word("bf16,triatt=flash")[1]["triatt"] == "flash"
    assert D.parse_word("bf16,trimul=site")[1]["trimul"] == "site"
    for bad in ("bf16,triatt=core.nope", "bf16,transition=core.k2", "bf16,trimul=core.", "bf16,triatt=k9"):
        with pytest.raises(ValueError):
            D.parse_word(bad)
    assert D.core_word("core.native") == "native" and D.core_word("v4") is None
    assert set(D.core_words("triatt")) == set(KA.ROW_NAMES) | set(KA.TIER_WORDS)
    assert set(D.core_words("trimul")) == set(KT.ROW_NAMES) | set(KT.TIER_WORDS)
    assert {"trimul_core", "triatt_core", "triatt_flash", "triatt_default", "triatt_k2b", "triatt_cueq"} <= set(D.STATS["sites"]) == set(D.SITE_WORDS) and set(D.TRIATT_SITE_WORDS.values()) <= set(D.SITE_WORDS)


def test_trimul_provider_words_are_class_true_and_the_ceiling_is_named():
    assert TMA.provider_word({}) is None and TMA.max_tokens({}) is None
    assert TMA.provider_word({TMA.PROVIDER_ENV: "TMK3_FAST"}) == "tmk3_fast" and TMA.max_tokens({TMA.MAX_TOKENS_ENV: "800"}) == 800
    assert TMA.max_tokens({TMA.MAX_TOKENS_ENV: "x"}) is None
    assert not hasattr(TMA, "MIN_TOKENS_ENV") and not hasattr(TMA, "min_tokens"), "no token gate of this tree's: the provider decides every (card, shape) cell"
    assert TMA.TIER_OF == {"exact": "exact", "1": "fast"} and TMA.tier_word("exact", None) == "exact" and TMA.tier_word("1", None) == "fast" and TMA.tier_word("1", "big") == "big"
    rows = KT.table()["rows"]
    assert TMA.TIER_WORDS_OF == {"exact": ("exact",), "1": ("fast", "big")} and set(sum(TMA.TIER_WORDS_OF.values(), ())) <= set(KT.TIER_WORDS)
    for w in TMA.PROVIDER_WORDS_OF["exact"]:                                                # an exact lever names the provider's `exact` TIER word or exact-class KERNEL rows only (class-true)
        assert (w in KT.TIER_WORDS and w == "exact") or (w in KT.ROW_NAMES and rows[w]["class"] == "exact" and w not in KT.STOCK_ROWS), w
    for w in TMA.PROVIDER_WORDS_OF["1"]:                                                    # the fast lever: the tolerance-class tier words or tolerance-class rows
        assert (w in KT.TIER_WORDS and w in ("fast", "big")) or (w in KT.ROW_NAMES and rows[w]["class"] != "exact" and w not in KT.STOCK_ROWS), w
    assert "tmk3_exact" not in TMA.PROVIDER_WORDS_OF["1"] and "cueq" not in TMA.PROVIDER_WORDS_OF["1"] and "v4" not in TMA.PROVIDER_WORDS_OF["exact"] and "fast" not in TMA.PROVIDER_WORDS_OF["exact"]
    from opt_core import trimul as T
    with pytest.raises(ValueError):
        TMA._provider(T, "exact", "v4")                                                 # a fast row on the exact lever: refused at apply by name
    with pytest.raises(ValueError):
        TMA._provider(T, "1", "tmk3_exact")
    with pytest.raises(ValueError):
        TMA._provider(T, "exact", "fast")                                               # a tolerance tier word on the exact lever
    with pytest.raises(ValueError):
        TMA._provider(T, "1", "exact")
    p = TMA._provider(T, "1", None)
    assert p.name == "trimul:fast" and p.routes["word"] == "fast" and TMA._with_ceiling(T, p, None) is p
    assert TMA._provider(T, "1", "big").routes["word"] == "big" and TMA._provider(T, "exact", None).name == "trimul:exact"
    calls = []
    q = T.custom("probe", lambda call: None, lambda call: calls.append(call.n_tokens))
    q = TMA._with_ceiling(T, q, 800)

    class _Call:
        def __init__(self, n): self.n_tokens = n
    q.eligible(_Call(800)); assert calls == [800]
    with pytest.raises(T.Refused) as e:
        q.eligible(_Call(801))
    assert e.value.reason == TMA.ABOVE_MAX == "above_max_tokens" and calls == [800]


def test_defaults_are_todays_bindings():
    """Nothing any mode runs changes unless a row names a word: the mode table's words for these levers are today's."""
    from boltz2_opt import modes
    ex, fa = modes.MODES["exact"]["env"], modes.MODES["fast"]["env"]
    assert ex["BOLTZ_FPF_TRIMUL"] == "exact" and ex["BOLTZ_PAIRBLOCK"] == "cueq"
    from boltz2_opt import pairblock as PB
    assert fa["BOLTZ_FPF_TRIMUL"] == "1" and fa["BOLTZ_PAIRBLOCK"] in PB.TIER2_VARIANTS and fa["BOLTZ_PAIRBLOCK"] == "default", "fast's block variant: Tier 2, the block's own keyed core (0.3.23)"
    assert fa["BOLTZ_PAIRFUSE"].split(",")[0] == "bf16"


def test_trimul_card_refusal_is_the_providers_word(monkeypatch):
    """A provider row this card cannot serve is refused at install by the provider's own word (no launch)."""
    if not torch.cuda.is_available():
        assert TMA._card_refusal("tx_sm90a") is None                # no device: nothing asked
        return
    def refuse(*a, **k):
        raise KT.Refusal("cc:8.0!=9.0(test)", "tx_sm90a", "v4")
    monkeypatch.setattr(KT, "select", refuse)
    assert TMA._card_refusal("tx_sm90a") == "tx_sm90a:cc:8.0!=9.0(test)"


def test_pairblock_exact_core_words_take_the_cueq_row_rule():
    """An exact-class `core.<word>` is the cueq construction around another bitwise core: on a card with a `cueq` row rule the same rule applies."""
    src = open(PB.__file__).read()
    assert 'rule_v = "cueq" if (cw is not None and cw.split("@")[0] in EXACT_CORE_WORDS) else v' in src
    assert set(PB.EXACT_CORE_WORDS) >= {"exact_headsplit", "cueq", "exact"}


def test_pick_words_of_their_own_join_the_pairfuse_word():
    """A site's pick as its own word (an available word; no mode row sets one): BOLTZ_PAIRFUSE_TRIATT=core.k2b == `bf16,triatt=core.k2b`;
    the comma form wins when both name the site; an unknown pick is refused by name either way; no token floor word exists."""
    import bz_pairfuse as BZ
    assert BZ.PICK_ENV == {"triatt": "BOLTZ_PAIRFUSE_TRIATT", "trimul": "BOLTZ_PAIRFUSE_TRIMUL"} and not hasattr(BZ, "PICK_MIN_ENV")
    r, picks = BZ.parse_word("bf16", environ={"BOLTZ_PAIRFUSE_TRIATT": "core.k2b"})
    assert r == "bf16" and picks["triatt"] == "core.k2b" and picks["trimul"] == BZ.DEFAULT_PICKS["trimul"]   # one site's pick word leaves the other site at the driver's default
    assert BZ.parse_word("bf16,triatt=k2b", environ={"BOLTZ_PAIRFUSE_TRIATT": "core.triattn_native"})[1]["triatt"] == "k2b"
    assert BZ.parse_word("bf16", environ={})[1] == BZ.default_picks("bf16") and BZ.parse_word("fp32", environ={})[1] == BZ.DEFAULT_PICKS
    with pytest.raises(ValueError):
        BZ.parse_word("bf16", environ={"BOLTZ_PAIRFUSE_TRIATT": "core.nope"})
    assert "triattn_native" in BZ.HANDABLE_ROWS and "k2b" in BZ.HANDABLE_ROWS


def test_the_rows_bind_the_triattn_provider_by_tier_word_and_carry_no_row_pick_or_floor():
    """The fast / big rows name the TriangleAttention site by the provider's TIER word inside the PAIRFUSE word; no row-word pick, no token
    floor, no per-card drop of a tri-attention lever (the provider's cells decide per card); report.py carries no row-named lever."""
    from boltz2_opt import modes, registry, report as R
    assert "native_triattn" not in registry.LEVERS and "native_triattn" not in modes.LEVER_ATTACH and 'name == "native_triattn"' not in open(R.__file__).read()
    import bz_pairfuse as BZ
    for m, tier in (("fast", "fast"), ("big", "big")):
        env = modes.MODES[m]["env"]
        assert "BOLTZ_PAIRFUSE_TRIATT" not in env and "BOLTZ_PAIRFUSE_TRIATT_MIN_TOKENS" not in env and "BOLTZ_PAIRBLOCK_MIN_TOKENS" not in env, m
        assert BZ.core_word(BZ.parse_word(env["BOLTZ_PAIRFUSE"], environ={})[1]["triatt"]) == tier, m
        assert all("native_triattn" not in d.get(m, {}) for d in modes.CARD_DROPS.values())


def test_call_form_hint_words():
    """0.3.7: the provider's call-form hint for what the kit hands the core: no key mask -> bias_only; pair_fused's per-row key mask -> mask_bias."""
    assert PB.call_form(None) == "bias_only"
    assert PB.call_form(torch.ones(1, 8, 1, 1, 8, dtype=torch.bool)) == "mask_bias"
    import inspect
    assert "form" in inspect.signature(KA.triangle_attention).parameters, "opt_core >= 0.5.37.2 takes the hint (an older face is called without it)"


def test_every_row_binds_the_trimul_provider_by_tier_word_class_true():
    """The exact ROW names the trimul provider's `exact` TIER word, the fast row `fast`, the big row `big` (BOLTZ_FPF_TRIMUL_PROVIDER; the
    PAIRFUSE driver's C=128 site names the same words). By CPU resolution the exact word names, on 9.0 / 8.0 at every N in both directions, for
    the precision Boltz-2's pair stacks present (fp32-resident z under bf16 autocast) and for bf16-resident z, at c_z 128 (trunk) and 64 (template
    Pairformer), an exact-class row or the provider's stock row — never a tolerance-class row (the winner is the provider's to name; the test
    asserts the class) —; the adapter serves an exact-class KERNEL row through the face and hands a stock-row cell to the engine's own TriMul BY
    NAME (library_op:<row>, EXPECTED: the lever idles there, gate open); the tolerance words serve every cell through the face."""
    from opt_core import trimul as T
    import boltz2_opt.modes as modes
    assert {m: modes.env_row(m)[TMA.PROVIDER_ENV] for m in ("exact", "fast", "big")} == {"exact": "exact", "fast": "fast", "big": "big"}
    assert all(modes.env_row(m)["BOLTZ_FPF_TRIMUL"] == ("exact" if m == "exact" else "1") for m in ("exact", "fast", "big"))
    rows = KT.table()["rows"]; kernel_rows = set(TMA._exact_kernel_rows(KT))
    assert kernel_rows and all(rows[r]["class"] == "exact" for r in kernel_rows) and not kernel_rows & (set(KT.STOCK_ROWS) | set(getattr(KT, "MODULE_EXACT_ROWS", ())))
    seen = set()
    for cc, stack in (((9, 0), "H100:2.13.0+cu130/3.7.1/cueq0.11.1"), ((9, 0), "H100:2.12.0+cu130/3.7.0/cueq0.10.0"), ((8, 0), "A100:2.13.0+cu130/3.7.1/cueq0.11.1"), ((8, 0), "A100:2.12.0+cu130/3.7.0/cueq0.10.0")):
        for res in ("fp32", None):
            for c in (128, 64):
                for n in (20, 199, 256, 400, 800, 1200, 2048):
                    for d in ("outgoing", "incoming"):
                        try:
                            sel = KT.select(cc, "bf16", c, c, n, d, word="exact", residency=res, stack=stack)
                        except KT.Refusal:
                            continue                                                          # refused by name: the engine's TriMul serves that call, counted
                        assert sel.row in kernel_rows or sel.row in KT.STOCK_ROWS, (cc, stack, res, c, n, d, sel.row)
                        seen.add(rows.get(sel.row, {}).get("class", "stock"))
    assert seen <= {"exact", "stock"}, seen
    assert TMA.EXPECTED_OF == {"1": (), "exact": (TMA.LIBRARY_OP,)} and TMA.LIBRARY_OP == "library_op", "the rows' declared paths to the engine's TriMul: none under a tolerance word; the library-op cells under the exact word"
    p = TMA._provider(T, "exact", "exact")                                                  # routing on CPU with stand-ins for the face's selection / serving call
    assert p.name == "trimul:exact" and set(p.routes["kernel_rows"]) == kernel_rows and p.routes["word"] == "exact"

    class Sel:
        def __init__(self, row, cell="9.0|f32z_bf16|C128|H128|N<=400|out|fwd"): self.row, self.cell = row, cell

    class C:
        def __init__(self, ch=128): self.channels, self.extra, self.n_tokens = ch, {}, 400

    def face_select(row):                                                                   # by_word's own eligible leaves the Selection on call.extra (or raises Refused by name)
        def e(call):
            if row is None:
                raise T.Refused("cc:8.0_no_cell")
            call.extra["trimul_selection"] = Sel(row)
        return e
    # the adapter's own wrappers, end to end, over a stand-in face provider
    import boltz2_opt.trimul as TM2
    orig_by_word = T.by_word
    try:
        state = {"row": "tmk3_exact"}
        T.by_word = lambda weights_of, word, name=None, **k: T.Provider(name or f"trimul:{word}", lambda call: f"face-out:{word}", lambda call: face_select(state["row"])(call), kernel="trimul")
        q = TM2._provider(T, "exact", None)
        c = C(); q.eligible(c); assert q.fn(c) == "face-out:exact" and q.routes["face"] == {"tmk3_exact": 1} and q.routes["last_face"][0] == "tmk3_exact"
        state["row"] = "native_exact"; c = C(64); q.eligible(c); assert q.fn(c) == "face-out:exact" and q.routes["face"]["native_exact"] == 1     # the template Pairformer's C=64 TriMul: the same word, the provider's cell
        for stock in ("cueq", "torch_math"):
            state["row"] = stock; c = C()
            with pytest.raises(T.Refused) as e:
                q.eligible(c)
            assert e.value.reason == f"library_op:{stock}" and "trimul_selection" not in c.extra
        assert q.routes["aside"] == {"library_op:cueq": 1, "library_op:torch_math": 1}
        state["row"] = None
        with pytest.raises(T.Refused):                                                      # the face's own refusal by name stands: the engine's TriMul serves, counted (the gate refuses: not EXPECTED)
            q.eligible(C())
        w = TM2._provider(T, "1", "big")                                                  # a tolerance word: the face serves every cell it names, stock rows included
        for row in ("v4", "native", "cueq"):
            state["row"] = row; c = C(); w.eligible(c); assert w.fn(c) == "face-out:big"
        assert w.routes["face"] == {"v4": 1, "native": 1, "cueq": 1} and w.routes["aside"] == {} and w.routes["kernel_rows"] == []
        TM2._STATE["provider"] = q
        t = TM2._tier_facts()
        assert t["word"] == "exact" and t["row"] == "native_exact" and t["face"] == {"tmk3_exact": 1, "native_exact": 1} and "library_op:cueq" in t["aside"] and set(t["kernel_rows"]) == kernel_rows
    finally:
        T.by_word = orig_by_word; TM2._STATE["provider"] = None
    lever = T.Lever("boltz2-opt", "exact", provider=T.custom("x", lambda call: None, lambda call: None), expected=TM2.EXPECTED_OF["exact"])
    assert lever._expected("library_op:cueq") and lever._expected("library_op:torch_math") and not lever._expected("tmk3_exact:launch_grid_y>65535") and not lever._expected("below_min_tokens")


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_the_pair_track_driver_serves_the_provider_by_tier_word_at_every_token_count_and_tallies_its_line(mode, monkeypatch):
    """`_core_for` names the served block's attention core per token count: the row's pick — the core provider's face BY TIER WORD (`core.fast`
    on the fast row, `core.big` on the memory row) at EVERY served token count (no kit gate, no kit floor: the provider's cell decides per card
    and N) — and `triatt_site` tallies the site word it returns (`triatt_core`), seeded in STATS['sites'] from ONE table. A pick the provider
    handed to another row at a token count (handed_triatt, by name at the probe) serves that row's word there. CPU: the served call is a stand-in;
    the line renders for a 400-token shape."""
    torch = pytest.importorskip("torch")
    import types
    import boltz2_opt.modes as modes
    from opt_core.attn import pair_fused as PF
    import bz_pairfuse as D
    row = modes.env_row(mode)
    r, picks = D.parse_word(row["BOLTZ_PAIRFUSE"])
    assert D.core_word(picks["triatt"]) == {"fast": "fast", "big": "big"}[mode] and not hasattr(D, "CORE_GATE_DEFAULT") and not hasattr(D, "PICK_MIN_ENV")
    monkeypatch.setitem(D._STATE, "residency", r); monkeypatch.setitem(D._STATE, "picks", picks); monkeypatch.setitem(D._STATE, "handed_triatt", {800: "k2b"})
    monkeypatch.setitem(D.STATS, "sites", {w: 0 for w in D.SITE_WORDS})
    served = []
    def fake_block(z4, W, *, mask, ending, residual, impl, core, ln, scale):   # the core block's stand-in: records the core it was handed, returns z
        served.append(core if isinstance(core, str) else getattr(core, "__name__", "callable")); return z4
    monkeypatch.setattr(PF, "tri_attn_block", fake_block)
    monkeypatch.setattr(D, "_triatt_weights", lambda m, device: object())
    import boltz2_opt.pairblock as PB
    monkeypatch.setattr(PB, "core_callable", lambda w: (lambda *a, **k: None))   # the kit's one binding of the face by word (memoised per word on a GPU host); a stand-in here
    m = types.SimpleNamespace(mha=types.SimpleNamespace(c_hidden=32))
    expect = {}
    for n in (128, 256, 300, 400, 511, 512, 800, 1200):
        core, word = D._core_for(n)
        assert word in D.SITE_WORDS and word in D.STATS["sites"], (n, word)
        want = D.TRIATT_SITE_WORDS["k2b"] if n == 800 else D.TRIATT_SITE_WORDS["core"]   # the provider's word at every token count; the handed row's word where it handed one
        assert word == want, (mode, n, word, want)
        if n == 800: assert core == "k2b"
        z4 = torch.zeros(1, n, n, 8)
        out = D.triatt_site(m, z4, ending=False, mask=None, inplace=True)                                    # the served path: no exception, the word tallied
        assert out is z4
        expect[word] = expect.get(word, 0) + 1
    assert {w: c for w, c in D.STATS["sites"].items() if c} == expect == {D.TRIATT_SITE_WORDS["core"]: 7, D.TRIATT_SITE_WORDS["k2b"]: 1}, (D.STATS["sites"], expect)
    # the census line renders with these tallies (a 400-token shape among them) and names the providers by tier word
    monkeypatch.setitem(D._STATE, "applied", True); monkeypatch.setitem(D._STATE, "word", row["BOLTZ_PAIRFUSE"])
    monkeypatch.setattr(D, "composed_ok", lambda: None)                 # the hooks-in-force check imports boltz (absent on a CPU-only host); the line's tallies are what is read here
    text = D.line()
    assert text.startswith(f"[{D.TAG}] LEVER name={D.NAME} ") and f"{D.TRIATT_SITE_WORDS['core']}:" in text and "sites=" in text and " pf_cores=" in text and f"triatt:core.{ {'fast': 'fast', 'big': 'big'}[mode] }" in text, text
    assert " core_gate=" not in text and " pick_min=" not in text and " below_pick_min=" not in text, text
    _count_word = "no_such_site_word"; D._count(D.STATS["sites"], _count_word); assert D.STATS["sites"][_count_word] == 1   # a word outside the table is tallied, never a KeyError

