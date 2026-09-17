"""trimul_exact / trimul_v4 bound to the shared TriMul provider (opt_core.kernels.trimul behind opt_core.trimul.by_word) by the TIER WORD of the
mode — class contracts on CPU (the provider's select is pure; install binds classes on the stock module when the stock tree is importable):
each mode passes its tier word, a row word only through the developer variables, unknown / wrong-class words install skipped by name, a
provider row's named refusal is an expected step-aside to the stock op (gate ok, exit 0), and the kit carries no TriMul table, floor or pin."""
import importlib.util
import os

import pytest

from atlasfold_opt import registry
from atlasfold_opt.hooks import trimul as H

STACKS = {"9.0": "H100:2.7.1+cu128/3.3.1/cueq0.10.0", "8.0": "A100:2.7.1+cu128/3.3.1/cueq0.10.0"}   # the pinned stack's provider stack words per card (cc -> `<gpu>:<torch>/<triton>/cueq<version>`)


def test_each_mode_passes_its_tier_word(monkeypatch):
    monkeypatch.delenv("AFO_TRIMUL_WORD", raising=False); monkeypatch.delenv("AFO_TRIMUL_EXACT_WORD", raising=False)
    assert (H.word("exact"), H.word("fast"), H.word("big")) == ("exact", "fast", "big")          # big passes `big`, never the fast word
    monkeypatch.setenv("AFO_TRIMUL_WORD", "  "); monkeypatch.setenv("AFO_TRIMUL_EXACT_WORD", "")
    assert (H.word("exact"), H.word("fast"), H.word("big")) == ("exact", "fast", "big")          # blank = unset
    monkeypatch.setenv("AFO_TRIMUL_WORD", " v4 "); monkeypatch.setenv("AFO_TRIMUL_EXACT_WORD", "tmk3_exact")   # developer row words (engineering, never a mode)
    assert (H.word("exact"), H.word("fast"), H.word("big")) == ("tmk3_exact", "v4", "v4")


def test_no_kit_table_floor_or_pin():
    for name in ("MIN_TOKENS", "ESM_BIND", "ESM_RETIRED", "KIT_WORDS", "LEGACY_WORD", "DEFAULT_WORD", "FAST_EXPECTED", "EXACT_EXPECTED"):
        assert not hasattr(H, name), name                                                        # the provider decides small N / cards / rows; the kit names nothing by number
    assert importlib.util.find_spec("atlasfold_opt.hooks.trimul_esm") is None and importlib.util.find_spec("atlasfold_opt.hooks.widegrid") is None
    import atlasfold_opt
    assert not os.path.exists(os.path.join(os.path.dirname(atlasfold_opt.__file__), "cells", "fpf_trimul_v4"))   # no kit cells table: the provider's TRIMUL_CELLS / fpf_trimul_v4 table serve
    src = open(os.path.join(os.path.dirname(atlasfold_opt.__file__), "stack.py")).read()
    assert "FPF_TRIMUL_V4_CELLS" not in src


def test_expected_words_are_the_kit_words_plus_every_row_refusal():
    KT = pytest.importorskip("opt_core.kernels.trimul")
    words = H.expected_words(KT.ROW_NAMES)
    assert set(H.EXPECTED) <= set(words) and all(f"{r}:" in words for r in KT.ROW_NAMES)
    assert "below_min_tokens" not in words and not any(w.startswith("no-cell") or w.startswith("no_cell") for w in words)
    for lever in ("trimul_exact", "trimul_v4"):                                                 # the registry documents a subset of the install's vocabulary
        assert set(registry.LEVERS[lever]["expected"]) <= set(words), (lever, set(registry.LEVERS[lever]["expected"]) - set(words))
        assert registry.LEVERS[lever]["strategy"] == "F2.trimul"
    assert registry.LEVERS["trimul_exact"]["cls"] == "exact" and registry.LEVERS["trimul_v4"]["cls"] == "fast"


def test_provider_answers_every_tier_word_at_the_pair_geometry():
    """Class contract against the provider as installed: at this engine's pair geometry (bf16, c_z = c_hidden = 128, the 512 / 896 / 1280
    buckets, both directions) every tier word resolves to SOME provider row on both cards' stack words — the exact word to an exact-class or
    stock row only — and never raises. WHICH row is the provider's cell, not asserted here."""
    KT = pytest.importorskip("opt_core.kernels.trimul")
    for cc, stack in STACKS.items():
        for n in (256, 512, 896, 1280):
            for d in ("outgoing", "incoming"):
                for word in ("exact", "fast", "big"):
                    s = KT.select(cc, "bf16", 128, 128, n, d, word=word, stack=stack, has_cueq=None)
                    assert s.row in KT.ROW_NAMES, (cc, n, d, word, s)
                    if word == "exact":
                        assert s.row in tuple(KT.EXACT_ROWS) + tuple(KT.STOCK_ROWS), (cc, n, d, KT.describe(s))   # an exact word never selects a tolerance row


class _FakeProvider:                       # the Lever's census / line read only these
    name = "trimul:fast"; resolved_from = "core"; kernel = "trimul:v4"; version = "fast/no-cell"
    def describe(self): return "trimul:v4@fast"


def test_named_row_refusal_on_every_call_is_a_step_aside_not_a_refusal():
    """Every call refused BY NAME by a provider row (`<row>:<kind>`) -> the engine's forward (the stock cuEquivariance op) served them: the kit
    gate is ok with the boundary named and the LEVER line says gate=ok fallback_to=stock_cueq; an unlisted reason still refuses. No kit floor:
    the line carries min_tokens=0."""
    T = pytest.importorskip("opt_core.trimul"); KT = pytest.importorskip("opt_core.kernels.trimul")
    words = H.expected_words(KT.ROW_NAMES)
    lever = T.Lever("atlasfold-opt", "fast", provider=_FakeProvider(), min_tokens=0, expected=words)
    for _ in range(96):
        lever._count("outgoing", "fallback:v4:n<101"); lever._count("incoming", "fallback:v4:n<101")
    g = H._gate_with_boundary(lever)()
    assert g.ok and "v4:n<101" in (g.reason or ""), getattr(g, "reason", None)
    line = H._line_with_boundary(lever, H._gate_with_boundary(lever))()
    assert " gate=ok" in line and "gate=refused" not in line and "fallback_to=stock_cueq" in line and "reason=v4:n<101" in line and " min_tokens=0 " in line, line
    lever2 = T.Lever("atlasfold-opt", "fast", provider=_FakeProvider(), min_tokens=0, expected=words)
    lever2._count("outgoing", "fallback:kernel_exploded")                                           # an unlisted reason refuses the gate and the line says so
    assert not H._gate_with_boundary(lever2)().ok and "gate=refused" in H._line_with_boundary(lever2, H._gate_with_boundary(lever2))()


def _site():
    pytest.importorskip("torch"); pytest.importorskip("opt_core.trimul")
    try:
        import atlasfold.model.network.primitives.triangle_update as tu
    except Exception:
        pytest.skip("stock atlasfold not importable here")
    if getattr(tu, "_cueq_triangle_multiplicative_update", None) is None:
        pytest.skip("stock cuequivariance branch absent")
    return tu


def _restore(tu, saved):
    for cls_name, f in saved.items():
        getattr(tu, cls_name).forward = f


def test_install_binds_the_tier_word_and_skips_by_name(monkeypatch):
    tu = _site()
    saved = {c: getattr(tu, c).forward for c, _ in H.CLASSES}
    try:
        monkeypatch.delenv("AFO_TRIMUL_WORD", raising=False); monkeypatch.delenv("AFO_TRIMUL_EXACT_WORD", raising=False)
        for mode, prov in (("exact", "trimul:exact"), ("fast", "trimul:fast"), ("big", "trimul:big")):
            ins = H.install(mode, "[t]", {})
            assert ins.applied and ins.facts["word"] == mode and ins.facts["tier_word"] is True and ins.facts["provider"] == prov, (mode, ins)
            assert ins.lever == ("trimul_exact" if mode == "exact" else "trimul_v4")
            assert all(hasattr(getattr(tu, c).forward, "__wrapped_stock__") for c, _ in H.CLASSES)
            _restore(tu, saved)
        monkeypatch.setenv("AFO_TRIMUL_WORD", "nonsense")
        ins = H.install("fast", "[t]", {}); assert not ins.applied and ins.reason == "unknown_word:nonsense"
        monkeypatch.setenv("AFO_TRIMUL_WORD", "esm_shapes+")                                        # no kit words: anything outside the provider's ROW_NAMES / TIER_WORDS is unknown
        ins = H.install("big", "[t]", {}); assert not ins.applied and ins.reason.startswith("unknown_word:")
        monkeypatch.setenv("AFO_TRIMUL_EXACT_WORD", "tmk3_fast")                                      # a tolerance-class row word on the exact lever: skipped by name
        ins = H.install("exact", "[t]", {}); assert not ins.applied and ins.reason == "not_an_exact_word:tmk3_fast"
        monkeypatch.setenv("AFO_TRIMUL_EXACT_WORD", "fast")
        ins = H.install("exact", "[t]", {}); assert not ins.applied and ins.reason == "not_an_exact_word:fast"
        monkeypatch.delenv("AFO_TRIMUL_EXACT_WORD"); monkeypatch.setenv("AFO_TRIMUL_WORD", "v4")       # a developer row word binds that provider row (no pre-provider construction survives)
        ins = H.install("fast", "[t]", {}); assert ins.applied and ins.facts["word"] == "v4" and ins.facts["provider"] == "trimul:v4" and ins.facts["tier_word"] is False
        _restore(tu, saved); monkeypatch.setenv("AFO_TRIMUL_EXACT_WORD", "cueq"); monkeypatch.delenv("AFO_TRIMUL_WORD")   # the stock library op through the provider, by name
        ins = H.install("exact", "[t]", {}); assert ins.applied and ins.facts["provider"] == "trimul:cueq"
    finally:
        _restore(tu, saved)
