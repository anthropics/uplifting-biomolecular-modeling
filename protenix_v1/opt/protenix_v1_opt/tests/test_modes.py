"""Modes resolve to the kit's arms under the kit's own grammar (read from levers_ptx1.py), and the registry covers every lever."""
import pytest

from .conftest import KIT
from protenix_v1_opt import kit as K
from protenix_v1_opt import modes as M


def test_grammar_is_read_from_the_kit_file():
    trimul, levers = K.lever_grammar(KIT)
    assert trimul == ("stock", "fast", "exact")
    assert set(levers) == {"gblock", "gflash", "tricuda", "triexact", "xtr", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "dit_attn_exact", "lazy_init", "template_dedupe", "tmpl_triatt", "tmpl_trimul", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "tmpl_triatt", "tmpl_xtr", "tmpl_xtr", "tmpl_pairfused", "tmpl_trimul_exact"}


def test_modes_and_arms():
    assert M.MODES == ("exact", "fast", "big", "off") and M.DEFAULT_MODE in M.MODES
    assert set(M.KIT_MODES) == set(M.MODES)
    assert M.KIT_MODES["exact"].arm == "exact+gblock+triexact+xtr+sg+hoist+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact" and M.KIT_MODES["exact"].tier == 1
    assert M.KIT_MODES["fast"].arm == "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and M.KIT_MODES["fast"].tier == 2
    assert M.KIT_MODES["off"].arm == M.STOCK_ARM
    for mode in ("exact", "fast"):
        res = M.resolve(mode, KIT)
        assert res.arm == M.KIT_MODES[mode].arm and res.trimul == mode and "hoist" in res.levers
        assert all(lv in M.LEVERS for lv in res.levers)
        assert all(M.LEVERS[lv]["class"] == "exact" for lv in M.resolve("exact", KIT).levers)          # the exact arm composes exact-class levers only
    assert M.resolve("exact", KIT).levers == ("gblock", "triexact", "xtr", "sg", "hoist", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "template_dedupe", "sampler_prep", "atom_attn_exact", "tmpl_triatt", "tmpl_xtr", "tmpl_trimul_exact") and M.resolve("fast", KIT).levers == ("gflash", "tricuda", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused")
    assert M.resolve("off", KIT).levers == ()
    big = M.resolve("big", KIT)                        # the composition sized on this process (no --input here: unsized, below the size gate; tests/test_big.py sizes it)
    assert big.arm == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and big.trimul == "fast" and big.levers == ("gflash", "tricuda", "ttr", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused") and big.tier == 2 and big.memory_preset == M.MEMORY_PRESET and big.line == "big"   # the fast arm without sg / hoist at every size
    assert all(M.resolve(m, KIT).memory_preset is None for m in ("exact", "fast", "off"))


def test_registry_covers_the_grammar():
    _, levers = K.lever_grammar(KIT)
    assert set(levers) | {"exact", "fast"} == set(M.LEVERS)
    for name, spec in M.LEVERS.items():
        assert spec["class"] in ("exact", "tolerance") and spec["what"] and spec["file"], name


def test_parse_arm_rejects_what_the_kit_rejects():
    assert K.parse_arm("stock", KIT) == ("stock", ())
    assert K.parse_arm("exact+sg+hoist", KIT) == ("exact", ("sg", "hoist"))
    assert K.parse_arm("exact+gblock+xtr+sg+hoist", KIT) == ("exact", ("gblock", "xtr", "sg", "hoist"))
    with pytest.raises(ValueError):
        K.parse_arm("exact+turbo", KIT)
    with pytest.raises(ValueError):
        K.parse_arm("sg+hoist", KIT)
    with pytest.raises(ValueError):
        K.parse_arm("fast+sg+sg", KIT)


def test_check_mode():
    assert M.check_mode(" Exact ") == "exact" and M.check_mode(None) is None
    with pytest.raises(ValueError):
        M.check_mode("turbo")
