"""BOLTZ_TRANSITION = the core transition provider's TIER word (exact | fast | big) or `core.<row word>`: the word grammar, the construction
class (exact | fast) a word runs under, the declared census words, and — no shape / row-count / card table in the adapter."""
from .. import transition as TR


def test_variant_grammar():
    for w in ("exact", "fast", "big"):
        assert TR.variant({"BOLTZ_TRANSITION": w}) == w and TR.core_word(w) == w, "a tier word binds the provider by that very word"
    assert TR.variant({"BOLTZ_TRANSITION": "core.v1"}) == "core.v1" and TR.core_word("core.v1") == "v1"
    assert TR.variant({"BOLTZ_TRANSITION": "core.v2@bm64bh32w4s2il0"}) == "core.v2@bm64bh32w4s2il0"
    assert TR.variant({"BOLTZ_TRANSITION": "core."}) is None and TR.variant({"BOLTZ_TRANSITION": "v1"}) is None and TR.variant({}) is None
    assert TR.core_word(None) is None and TR.core_word("nope") is None


def test_class_of_words():
    assert TR.class_of("exact") == "exact" and TR.class_of("fast") == "fast" and TR.class_of("big") == "fast" and TR.class_of(None) is None
    assert TR.class_of("core.v1") == "exact" and TR.class_of("core.exact") == "exact"            # the caller's LayerNorm output projected: bitwise rows only
    for w in ("core.v1:lnfused", "core.v2", "core.v2:fast", "core.fast", "core.v2@bm64bh32w4s2il0", "core.af3_fused", "core.big"):
        assert TR.class_of(w) == "fast", w                                                      # LayerNorm in the kernel / tolerance rows
    assert TR.LN_OF[TR.class_of("core.v1")] == "stock" and TR.LN_OF["exact"] == "stock" and TR.LN_OF["fast"] == "kernel"
    assert TR.LEVERS_OF["exact"] == TR.LEVERS_OF["fast"] == ("fused_transition",)


def test_declared_words_and_no_kit_tables():
    assert set(TR.EXPECTED) == {"autocast_off", "chunked", "core_stock", "exact_unvouched", "no_cell"}
    assert TR.census_word("exact_vouch_not_recorded_on_A100:torch2.12.0+cu130/3.7.0") == "exact_unvouched"
    assert TR.census_word("exact_vouch_below_57_tokens_on_A100:torch2.12.0+cu130/3.7.0") == "exact_unvouched"
    assert TR.census_word("no_cell:9.0|bf16|pair_c128_n2|eager|fwd") == "no_cell"
    assert TR.census_word("dtype:fp32").startswith("core_refused:") and TR.census_word("dtype:fp32") not in TR.EXPECTED, "an undeclared refusal refuses the gate, by its word"
    for gone in ("CARD_ROWS", "KEPT_OUT", "CORE_EXACT_KEYS", "IMPL_ENV", "IMPLS", "kept_out"):
        assert not hasattr(TR, gone), f"transition.{gone}: which row serves which (card, C, hidden, N) is the provider's measured table, not the adapter's"
    assert TR.IMPL == "opt_core.kernels.transition"
