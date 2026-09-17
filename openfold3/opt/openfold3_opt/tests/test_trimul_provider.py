"""The triangle-multiplication provider binding of the pair cells (openfold3_opt.of3_trimul; lever `trimul_provider`, riding on the
`trimul_v4` cell): the word grammar (the line's tier word for every class, or the caller's word), the Router's decision ladder (the word's row
through the provider / the row v4 by name when a tier word's rows refuse the class / the line; refusals counted by word; the memo; a refused
(class, row) never asked again), the provider's static admission of rows at this engine's widths (opt_core.kernels.trimul.admits — pure python),
the census fields, and the wiring (modes / registry / stack / autoload).  No CUDA: the Router's `facts` are given, never read from tensors;
providers are stubs standing in for opt_core.trimul.by_word."""
import pytest

from openfold3_opt import _autoload, modes, of3_trimul as M, registry, stack
from openfold3_opt.tests import _stubs
from opt_core import trimul as CT
from opt_core.kernels import trimul as KT

HOME = _stubs.tree_home()


def test_word_grammar():
    words = M.known_words()
    assert set(KT.ROW_NAMES) <= set(words) and {"fast", "exact", "big"} <= set(words)
    assert M.check_word("tmk3_fast") == "tmk3_fast" and M.check_word("fast") == "fast"
    with pytest.raises(ValueError):
        M.check_word("not_a_row")
    assert M.KIT_ROW == "v4" and M.KIT_ROW in KT.ROW_NAMES
    assert set(M.STOCK_ROWS) <= set(KT.ROW_NAMES)
    assert M.requested({M.ENV: "1"}) and not M.requested({}) and not M.requested({M.ENV: "0"})
    assert M.spec({}) == "fast" and M.spec({M.ENV_WORD: "esm_v5_fwd"}) == "esm_v5_fwd"           # the caller's word, else the line's tier word
    assert M.spec({M.ENV_TIER: "big"}) == "big" and M.spec({M.ENV_TIER: "big", M.ENV_WORD: "v4"}) == "v4"
    assert M.tier_word({}) == "fast" and M.tier_word({M.ENV_TIER: "big"}) == "big" and M.tier_word({M.ENV_TIER: "exact"}) == "fast"
    assert not hasattr(M, "PREFER") and not hasattr(M, "PREFER_BY_TIER")                       # no kit row order: the core's cell table names the row per class and card


@pytest.mark.parametrize("row,facts", [
    ("esm_v5_fwd", ("9.0", "bf16", 128, 128, 400)),                # the shared core's `esm_v5_fwd` row at this kit's pair width in bf16
    ("esm_v5_fwd", ("9.0", "bf16", 64, 64, 400)),                  # the template width
    ("esm_v5_fwd", ("9.0", "fp32", 128, 128, 400)),                # fp32 operands
    ("esm_v61", ("9.0", "bf16", 128, 128, 400)),                   # the sealed CuTe-K3 line
    ("tmk3_fast", ("9.0", "bf16", 64, 64, 400)),                   # the template width's rows
    ("tmk3_exact", ("9.0", "bf16", 64, 64, 1200)),
    ("tmk3_fast", ("8.0", "bf16", 64, 64, 400)),
    ("v4", ("9.0", "bf16", 128, 128, 400)),                         # the kit row
    ("v4", ("9.0", "bf16", 64, 64, 400)),
])
def test_the_provider_admits_or_refuses_by_name_rows_at_this_engines_widths(row, facts):
    """Class contract, not the provider's current envelope: for provider rows at this engine's call facts the provider either ADMITS the row
    (None) or REFUSES it BY NAME — a `Refusal` carrying a refusal word and the row — never another exception and never a silent miss. Which
    widths / dtypes / capabilities a row admits is the provider's current envelope, revisited at any core release."""
    cc, dt, C, D, N = facts
    assert row in KT.ROW_NAMES, row
    try:
        assert KT.admits(row, cc, dt, C, D, N) is None
    except KT.Refusal as r:
        assert r.kind and isinstance(r.kind, str) and (r.row in (None, row)), (row, facts, r.kind, r.row)


def test_the_kit_row_serves_this_engines_trunk_width_on_the_pinned_part():
    """The one envelope this kit owns: its own row `v4` (the cell's statement + this kit's launch-cell table) admits the trunk pair width
    (c_z 128) in bf16 on cc 9.0 — the fast line's TriMul always has a server there."""
    assert KT.admits(M.KIT_ROW, "9.0", "bf16", 128, 128, 400) is None


def test_the_tier_words_resolve_at_this_engines_widths_on_both_cards():
    """The words the lines ask (fast, big) resolve through the core's cell table to a provider row (or a stock row naming the line's op) at
    the pairformer width (128,128) and the template width (64,64) on cc 9.0 and 8.0 at the ladder's token counts — never an unnamed miss."""
    stacks = {"9.0": "H100:2.10.0+cu128/3.6.0/cueq0.10.0", "8.0": "A100:2.10.0+cu128/3.6.0/cueq0.10.0"}
    for cc in ("9.0", "8.0"):
        for (C, H) in ((128, 128), (64, 64)):
            for word in ("fast", "big"):
                for N in (400, 800, 1200, 2000):
                    try:
                        sel = KT.select(cc, "bf16", C, H, N, "outgoing", word=word, stack=stacks[cc], has_cueq=True)
                    except KT.Refusal as e:
                        assert str(e).strip(), (cc, C, H, word, N); continue
                    assert sel.row in KT.ROW_NAMES, (cc, C, H, word, N, sel.row)


class _Sel:
    def __init__(self, row, cell="cell"):
        self.row, self.cell = row, cell


class _Prov:
    """Stands in for opt_core.trimul.by_word(word): eligible() sets the selection (row) or raises Refused."""
    def __init__(self, word, row=None, refuse=None):
        self.word, self.row, self.refuse, self.asked = word, row or word, refuse, 0

    def eligible(self, call):
        self.asked += 1
        if self.refuse:
            raise CT.Refused(self.refuse)
        call.extra["trimul_selection"] = _Sel(self.row)

    def fn(self, call):
        raise AssertionError("not served in the CPU tests")


def r_facts(r):
    """plan() positional stand-ins: (mod, z4, m3, direction, kit_why) for a class the kit row admits."""
    return object(), object(), None, "outgoing", None


def _router(word="fast", facts=("9.0", "bf16", 128, 128, 400), provs=None):
    r = M.Router(word, weights_of=lambda mod: {})
    r.facts = lambda z4, mod: facts
    provs = provs or {}
    r.provider = lambda w: provs.setdefault(w, _Prov(w))
    return r, provs


def test_router_the_tier_word_through_the_provider_for_every_class():
    # every class binds THROUGH THE PROVIDER by the line's tier word: whatever row the core's table names (a stand-in row here) is served as a
    # provider row — v4 included; a class the tier word's rows refuse asks the row v4 by name; the line only when that refuses too
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", row="some_measured_row")})
    assert r.by_tier and r.tier == "fast" and r.word == "fast"
    p = r.plan(*r_facts(r))
    assert p.kind == "row" and p.row == "some_measured_row" and p.reason == "fast>some_measured_row" and provs["fast"].asked == 1
    assert r.cells[p.key] == "fast>some_measured_row"
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", row=M.KIT_ROW)})     # the tier word names the v4 kernel: through the provider
    p = r.plan(*r_facts(r))
    assert p.kind == "row" and p.row == M.KIT_ROW and p.prov is provs["fast"] and p.reason == "fast>v4"
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", refuse="fast:no_cell"), M.KIT_ROW: _Prov(M.KIT_ROW, row=M.KIT_ROW)})
    p = r.plan(*r_facts(r))                                                                                # the tier word refuses the class: the row v4 BY NAME through the provider, the refusal counted
    assert p.kind == "row" and p.row == M.KIT_ROW and p.prov is provs[M.KIT_ROW] and p.reason is None and r.skips == {"fast(fast:no_cell)": 1}
    r, provs = _router("fast", ("9.0", "bf16", 64, 64, 400), {"fast": _Prov("fast", refuse="fast:no_cell"), M.KIT_ROW: _Prov(M.KIT_ROW, refuse="v4:c_z:64")})
    p = r.plan(object(), object(), None, "outgoing", "c:64x64")                                            # … and v4 refuses too: the line, by the class's reason, both refusals named
    assert p.kind == "line" and p.reason == "c:64x64" and r.skips == {"fast(fast:no_cell)": 1, "v4(v4:c_z:64)": 1} and r.cells[p.key] == "line(c:64x64)"
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", refuse="x"), M.KIT_ROW: _Prov(M.KIT_ROW, refuse="y")})
    p = r.plan(object(), object(), None, "outgoing", None)                                                 # no class reason: `refused`
    assert p.kind == "line" and p.reason == "refused"
    r, provs = _router("big", ("9.0", "bf16", 128, 128, 400), {"big": _Prov("big", row="lean_row")}); r.tier = "big"   # a big line's sites ask the big word
    p = r.plan(*r_facts(r))
    assert p.kind == "row" and p.reason == "big>lean_row" and "fast" not in provs
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", row="cueq")})        # the table names the line's own op: the line, by the cell's word
    p = r.plan(*r_facts(r))
    assert p.kind == "line" and p.reason == "cell:cueq"
    # the template width: the tier word through the provider on both cards (the core's table names the row per card and moves it)
    r, provs = _router("fast", ("9.0", "bf16", 64, 64, 400), {"fast": _Prov("fast", row="the_tables_row")})
    p = r.plan(object(), object(), None, "outgoing", "c:64x64")
    assert p.kind == "row" and p.row == "the_tables_row" and p.reason == "fast>the_tables_row" and provs["fast"].asked == 1
    r, provs = _router("big", ("9.0", "bf16", 64, 64, 2000), {"big": _Prov("big", row="the_big_cells_row")}); r.tier = "big"
    p = r.plan(object(), object(), None, "outgoing", "c:64x64")
    assert p.kind == "row" and p.row == "the_big_cells_row" and p.reason == "big>the_big_cells_row"   # no kit row list at the template width on a big line: the cell decides
    r, provs = _router("fast", ("8.0", "bf16", 64, 64, 800), {"fast": _Prov("fast", row="an_8_0_row")})
    p = r.plan(object(), object(), None, "outgoing", "c:64x64")
    assert p.kind == "row" and p.row == "an_8_0_row" and p.reason == "fast>an_8_0_row"
    r, provs = _router("fast", ("8.0", "bf16", 128, 128, 800), {"fast": _Prov("fast", row=M.KIT_ROW)})
    p = r.plan(object(), object(), None, "outgoing", None)
    assert p.kind == "row" and p.row == M.KIT_ROW and p.reason == "fast>" + M.KIT_ROW


def test_router_refused_class_row_is_never_asked_again_and_the_next_candidate_serves():
    # the tier word's row serves first; once serve_row records the (class, row) refused (bad numerics / serve-time refusal) the plan moves on —
    # the row v4 by name, then the line — and never asks the refused word again
    r, provs = _router("fast", ("9.0", "bf16", 128, 128, 400), {"fast": _Prov("fast", row="the_tables_row"), M.KIT_ROW: _Prov(M.KIT_ROW, row=M.KIT_ROW)})
    p = r.plan(object(), object(), None, "outgoing", None)
    assert p.kind == "row" and p.row == "the_tables_row"
    r.refused[(p.key, "fast")] = "bad_numerics(maxabs=nan)"             # what serve_row records for a non-finite / gross first statement
    r.refused[(p.key, p.row)] = "bad_numerics(maxabs=nan)"
    p2 = r.plan(object(), object(), None, "outgoing", None)
    assert p2.kind == "row" and p2.row == M.KIT_ROW
    r.refused[(p.key, M.KIT_ROW)] = "refused(x)"
    p3 = r.plan(object(), object(), None, "outgoing", None)
    assert p3.kind == "line" and p3.reason == "refused"


def test_router_explicit_word_serves_that_word_for_every_class_it_admits():
    r, provs = _router("esm_v5_fwd", ("9.0", "bf16", 128, 128, 400))
    assert not r.by_tier and r.word == "esm_v5_fwd"
    p = r.plan(object(), object(), None, "outgoing", None)
    assert p.kind == "row" and p.row == "esm_v5_fwd"
    r, provs = _router("esm_v5_fwd", ("9.0", "bf16", 64, 64, 400), {"esm_v5_fwd": _Prov("esm_v5_fwd", refuse="esm_v5_fwd:c_z:64(not_128|256)")})
    p = r.plan(object(), object(), None, "outgoing", "c:64x64")                 # refused where it cannot serve: the line, named (a row word asks no other row)
    assert p.kind == "line" and p.reason == "c:64x64" and list(r.skips) == ["esm_v5_fwd(esm_v5_fwd:c_z:64(not_128_256))"]
    r, provs = _router("cueq", ("9.0", "bf16", 128, 128, 400))                  # a stock row names the line's own op
    p = r.plan(object(), object(), None, "outgoing", None)
    assert p.kind == "line" and p.reason == "cell:cueq"
    r, provs = _router("big", ("9.0", "bf16", 128, 128, 400), {"big": _Prov("big", row="v4")})   # a tier word resolved to v4: served through the provider
    p = r.plan(object(), object(), None, "outgoing", None)
    assert p.kind == "row" and p.row == "v4"


def test_census_fields():
    r, _ = _router("fast", ("9.0", "bf16", 64, 64, 400), {"fast": _Prov("fast", row="tmk3_fast")})
    r.plan(object(), object(), None, "outgoing", "c:64x64")
    r._bump(r.rows, "tmk3_fast", 34); r.count_kit(); r.count_kit(); r.count_line("n<101")
    r.witness["k:tmk3_fast"] = "maxabs:0.0312/relrms:0.0021(vs:torch_math)"
    f = r.fields()
    assert f.startswith("word=fast tier=fast rows=tmk3_fast:34 kit=v4:2 line=n<101:2 skip=- witness=k:tmk3_fast:maxabs:0.0312/relrms:0.0021(vs:torch_math) cells=")
    assert " " not in f.split("cells=", 1)[1] and all("=" in w for w in f.split(" "))   # blank-free k=v words (the LEVER line grammar)
    assert ":fast>" in f.split("cells=", 1)[1]                                       # the class this plan decided names the tier word and the row it resolved to


def test_wiring_modes_registry_stack_autoload():
    fam = modes.TRIMUL_PROVIDER_ENVS
    assert fam == (M.ENV, M.ENV_WORD, M.ENV_TIER) and fam in modes.CELL_FAMILIES
    fast = modes.LINES[("fast", None)]
    assert fast.env[M.ENV] == "1" and M.ENV_WORD not in fast.env                # the line arms it; the word is the caller's knob
    lv = list(fast.levers)
    assert lv.index("trimul_provider") == lv.index("trimul_v4") + 1
    for key, ln in modes.LINES.items():                                         # every line that carries trimul_v4 (it rides on the cell): fast and big/resident
        assert ("trimul_provider" in ln.levers) == ("trimul_v4" in ln.levers), key
        if "trimul_provider" in ln.levers:
            assert ln.env.get(M.ENV) == "1" and ln.env.get(M.ENV_TIER) == ("big" if key[0] == "big" else "fast"), key
    assert "trimul_provider" in modes.LINES[("big", "resident")].levers
    assert modes.LEVER_SWITCHES["trimul_provider"] == (M.ENV, M.ENV_TIER) and modes.LEVERS_OFF_TAKES["trimul_v4"] == ("trimul_provider",)
    lever = registry.LEVERS["trimul_provider"]
    assert lever.tier == registry.TOLERANCE and lever.env_keys == (M.ENV,) and lever.marker.startswith("[openfold3-opt/pairfused] trimul_provider:")
    assert "trimul_provider" not in registry.A100_TESTED and registry.tested_sm("trimul_provider") == ("sm90",)   # H100 records only until the A100 pass measures it
    assert "trimul_provider" in stack._PROBES and stack.CORE_CELL_LEVERS["trimul_provider"] == ("fpf_trimul_v4", "fpf_trimul")
    assert M.ENV in _autoload.DECLARED and M.ENV_WORD in _autoload.DECLARED
    from openfold3_opt.cells import pairfused
    assert (pairfused.ENV_TRIMUL_PROVIDER, pairfused.ENV_TRIMUL_WORD, pairfused.ENV_TRIMUL_TIER) == (M.ENV, M.ENV_WORD, M.ENV_TIER)   # the cell spells the switches it consults


def test_levers_off_and_the_callers_word():
    OFF = modes.ENV_LEVERS_OFF
    base = modes.resolve("fast", HOME, environ={}, n_tokens=612)
    assert base.exports[M.ENV] == "1" and "trimul_provider" in base.levers
    r = modes.resolve("fast", HOME, environ={OFF: "trimul_provider"}, n_tokens=612)          # off alone: its switch leaves and is required unset; trimul_v4 stays in the pair value
    assert r.conflicts == [] and M.ENV not in r.exports and M.ENV in r.unsets and "trimul_provider" not in r.levers
    assert "trimul_v4" in r.levers and "trimul_v4" in r.exports[modes.PAIR_ENVS[0]].split(":")
    r = modes.resolve("fast", HOME, environ={OFF: "trimul_v4"}, n_tokens=612)                # the cell off takes the provider's row choice with it, named
    assert r.conflicts == [] and "trimul_v4" not in r.levers and "trimul_provider" not in r.levers and M.ENV not in r.exports and M.ENV in r.unsets
    r = modes.resolve("fast", HOME, environ={M.ENV_WORD: "esm_v5_fwd"}, n_tokens=612)      # the caller's word on the line that arms the family: theirs to set
    assert r.conflicts == [] and r.exports[M.ENV] == "1"
    r = modes.resolve("exact", HOME, environ={M.ENV_WORD: "esm_v5_fwd"}, n_tokens=400)     # on a line that does not carry the lever: refused by name
    assert r.conflicts and any(M.ENV_WORD in c or "TRIMUL" in c for c in r.conflicts)
