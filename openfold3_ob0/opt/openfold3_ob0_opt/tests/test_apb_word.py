"""The pair-bias attention family's provider word (cells/apb_word.py): the fast line exports the tier word `fast`, the big lines `big`;
the caller's knob names any provider word and wins; unknown words are refused BY NAME; `bind` hands the word to the tree's
`dit_rows.set_apb_word` once, before the levers' cores resolve, and `resolve_apb_core("auto")` then returns the provider-backed core;
no word set = the direct entry as before (nothing bound). CPU-only: no kernel runs here."""
import importlib

import pytest

from openfold3_ob0_opt import _autoload, modes
from openfold3_ob0_opt.cells import apb_word, dit_attn, dit_glue, apb_trunk
from opt_core.of3_sampler import dit_rows as DR


def _fresh():
    apb_word.STATE.update(word=None, source="none", bound=False, error=None)
    DR.set_apb_word(None)


def test_names_and_lines():
    assert (apb_word.ENV_TIER, apb_word.ENV_WORD) == modes.APB_WORD_ENVS == ("OPENFOLD3_OB0_OPT_APB_TIER", "OPENFOLD3_OB0_OPT_APB_WORD")
    assert modes.APB_WORD_ENVS in modes.CELL_FAMILIES and set(modes.APB_WORD_ENVS) <= set(_autoload.DECLARED)
    assert modes.LINES[("fast", None)].env[apb_word.ENV_TIER] == "fast"
    for key, ln in modes.LINES.items():
        if key[0] == "big" and dit_attn.ENV in ln.env:
            assert ln.env[apb_word.ENV_TIER] == "big", key                  # the big lines ask the big tier word literally
        if key[0] == "exact":
            assert apb_word.ENV_TIER not in ln.env, key                        # no pair-bias attention lever on the exact line: no word
        assert apb_word.ENV_WORD not in ln.env, key                            # the caller's knob is never a line's export


def test_word_precedence_and_validation():
    assert apb_word.word({}) is None and apb_word.requested({}) is False
    assert apb_word.word({apb_word.ENV_TIER: "fast"}) == "fast" and apb_word.word({apb_word.ENV_TIER: "big", apb_word.ENV_WORD: "apb_attn"}) == "apb_attn"
    assert apb_word.requested({apb_word.ENV_WORD: "sdpa:cudnn"}) is True and apb_word.requested({apb_word.ENV_WORD: "dtk_loop"}) is True
    for bad in ({apb_word.ENV_TIER: "exact"}, {apb_word.ENV_TIER: "on"}, {apb_word.ENV_WORD: "flash"}, {apb_word.ENV_WORD: "sdpa:nonsense_variant_x"}):
        with pytest.raises(ValueError) as ei:
            apb_word.requested(bad)
        assert list(bad)[0] in str(ei.value)                                  # refused BY NAME


def test_bind_hands_the_word_to_the_tree_and_resolve_returns_the_provider_core():
    _fresh()
    try:
        assert apb_word.bind({}) is None and DR.apb_word() is None
        core, why = DR.resolve_apb_core("auto")
        assert not isinstance(core, DR.ProviderCore)                            # nothing bound: the direct entry (or None off the GPU stack) — never the provider
        _fresh()
        assert apb_word.bind({apb_word.ENV_TIER: "big"}) == "big" and DR.apb_word() == "big" and apb_word.STATE["source"] == "tier"
        assert apb_word.bind({apb_word.ENV_TIER: "fast"}) == "big"             # bound once per process (idempotent)
        core, why = DR.resolve_apb_core("auto")
        assert isinstance(core, DR.ProviderCore) and core.word == "big" and why == "" and core.Unsupported is DR.ProviderUnsupported
        c = core.census()
        assert c["word"] == "big" and c["served"] == 0 and set(c) >= {"rows", "cells", "timing", "fallback", "refused"}
        assert DR.resolve_apb_core("dtk") == (None, "pinned:dtk")               # the per-sample pin keeps its meaning for the fused block
        pc, _ = DR.resolve_apb_core("dtk_loop")                                 # a row word for one caller (dit_attn's core knob dtk)
        assert isinstance(pc, DR.ProviderCore) and pc.word == "dtk_loop"
        _fresh()
        assert apb_word.bind({apb_word.ENV_TIER: "fast", apb_word.ENV_WORD: "fpf_apb"}) == "fpf_apb" and apb_word.STATE["source"] == "caller"
        with pytest.raises(ValueError):
            DR.set_apb_word("not_a_row")
    finally:
        _fresh()


def test_the_levers_bind_before_their_core_resolves():
    assert dit_attn.CORE_PREF == {"auto": "auto", "dtk": "dtk_loop"} and "flash_bias_attn(" not in open(dit_attn.__file__).read()   # no per-sample kernel loop in the kit: the core serves or the stock core does, by name
    for cell in (dit_glue, apb_trunk):
        assert cell.install is not cell._core.install and callable(cell.install)   # the adapter binds the word, then the tree installs
    src = open(dit_attn.__file__).read()
    assert src.index("apb_word.bind(environ)") < src.index("core0 = _apb_core()")
