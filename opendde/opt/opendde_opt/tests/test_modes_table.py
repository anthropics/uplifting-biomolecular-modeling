"""The mode table is the definitions' mode table: three modes, line S = exactly the kit's LEV_ENV switches (+ the house strict/warn),
`exact` = S1 (S + the FPF contract, in-process only), `fast` = LSTAR2A, `big` = BIG_F / BIG_TP, the ACCEL hook first on
every line's path, no test hook ever exported, host levers absent in S, and every kit line spelled as the kit README's row."""
import os

import pytest

from opendde_opt import modes, registry
from opendde_opt.tests._stubs import TREE

LEV_ENV = {"ODDE_SERVED_LEVERS": "1",                                                        # the served-levers hook's environment (modes._S_EXPORTS)
           "CUEQ_TRITON_CACHE_DIR": "$KIT/levers/KIT/cueq_cache_shipped", "ODDE_ADDON_LEVERS": "dit_hoist,dit_align", "ODDE_ARM_Z": "1"}
S_EXPORTS = {**LEV_ENV, "ODDE_SERVED_LEVERS_STRICT": "1"}
FPF = {"FPF_ENGINE": "opendde", "FPF_OPS": "trimul_out,trimul_in",
       "FPF_IMPL": "trimul_out=odde_trimul_bind:trimul_out,trimul_in=odde_trimul_bind:trimul_in"}          # the adapter's callables = the core-provider binding's (0.2.57: no kit TriMul construction)


def test_modes_are_the_three_shared_names():
    assert modes.MODES == ("off", "exact", "fast", "big")
    assert modes.DEFAULT_MODE in modes.MODES
    # the package default is fast (a fast mode ships); the env switch unset = off
    assert modes.MODE_LINES == {"off": None, "exact": "S1", "fast": "LSTAR2A", "big": "BIG_F"} and modes.DEFAULT_MODE == "fast"     # the kit owner's rows + big
    assert set(modes.MODE_LINES) == set(modes.MODES)


def test_exact_is_S1_the_kits_lev_env_plus_fpf_trimul_exact():
    assert modes._S_EXPORTS == S_EXPORTS                                          # the base composition S = the kit's LEV_ENV, value for value
    ex = modes.line_of("exact")                                                   # S1 = S + FPF TriMul EXACT, in-process only
    assert ex.name == "S1" and ex.tier == "exact" and ex.fpf == "exact"
    assert modes.line_of("EXACT") is ex                                           # a mode name is case-folded (a kit line goes by its exact upper-case name)
    assert {k: v for k, v in ex.exports.items() if k not in modes.FPF_SWITCHES + modes.XL_SWITCHES + tuple(modes._TRIATTN_EXACT) + modes.SAMPLER_SWITCHES + tuple(modes._SP_EXACT_EXPORTS) + tuple(modes._TRIMUL_EXACT) + tuple(modes._TRANSITION_EXACT)} == S_EXPORTS
    assert ex.exports["ODDE_TRIMUL"] == "exact" and "FPF_TRIMUL_MODE" not in ex.exports and "FPF_TRIMUL_OPENDDE" not in ex.exports and ex.exports["FPF_ENGINE"] == "opendde"
    assert ex.exports["FPF_OPS"] == "trimul_out,trimul_in"
    assert "ODDE_XL" not in ex.exports and "xl_tri_ln" not in ex.levers                              # no XL lever on the exact line since 0.2.48 (measured trunk cost; registry row)
    assert set(ex.levers) == set(modes._S_LEVERS) | {"fpf_trimul_exact", "alloc_auto", "stepgraph", "triattn_exact", "triattn_conf", "dit_attn_exact", "trimul_exact", "transition_exact"}   # + the allocator lever (placement only) + the step graph + the providers' exact attention / TriMul rows + the SAMPLER exact kernel (0.2.40)
    assert ex.path_order == modes.KIT_PATH_ORDER and modes.XL not in ex.path_order                     # no XL unit on the exact line since 0.2.48
    for k in (modes.ALLOCATOR, "ODDE_ARM_U", "ODDE_DIT_ATTN_EXACT", "ODDE_DIT_FUSED") + modes.TEST_HOOKS:
        assert k in ex.unset, k
    assert ex.exports["ODDE_DIT_ATTN"] == modes.SAMPLER_EXACT_WORD == "exact"                          # 0.2.57: the sampler's token-attention statement through the provider's exact tier by word
    assert modes.KIT_PATH_ORDER == (modes.ACCEL, modes.ARMT, modes.DITFAST_TOOLS, modes.KIT_LAYER, modes.SRC)
    assert set(modes._S_LEVERS) == {"served_levers_hook", "cueq_tuned_cache", "dit_hoist", "dit_align", "arm_z", "drop_bond_mask", "lnstream", "tmpl_dedup", "keep_pool", "sched_host", "structok_sync", "json_oneshot", "prefetch", "zprep_hoist"}   # the hook's levers + the tree's own (bond mask, LayerNorm stream, template de-dup, keep-pool)


def test_fast_is_LSTAR2A():
    fa = modes.line_of("fast")
    assert fa.name == "LSTAR2A" and fa.tier == "tier2" and fa.fpf is None
    assert fa.exports["ODDE_ARM_U2_TRIMUL"] == "1" and fa.exports["ODDE_ARM_T_SCOPE"] == "all" and "arm_u23" in fa.levers   # LSTAR + the U2 TriMul + the scope binds


def test_S1_is_S_plus_fpf_trimul_exact_in_process():
    s1 = modes.LINES["S1"]
    assert s1.tier == "exact" and s1.fpf == "exact"
    assert s1.exports == {**S_EXPORTS, **FPF, "ODDE_TRIATTN": "exact", "ODDE_ARM_T_SCOPE": "all", "ODDE_DIT_ATTN": "exact", **modes._TRIMUL_EXACT, **modes._TRANSITION_EXACT}   # + the SAMPLER exact kernel (0.2.40); triattn_conf on the exact line too (0.2.57: the confidence head binds the tier word on every line)
    assert set(s1.levers) == set(modes._S_LEVERS) | {"fpf_trimul_exact", "alloc_auto", "stepgraph", "triattn_exact", "triattn_conf", "dit_attn_exact", "trimul_exact", "transition_exact"}
    for k in list(FPF):
        assert k not in s1.unset
    for name in ("S1", "LSTAR2A", "BIG_F", "BIG_TP"):                              # every mode line: the XL switches absent, the unit off its path (no line carries xl_tri_ln since 0.2.44)
        ln = modes.LINES[name]
        assert "ODDE_XL" in ln.unset and modes.XL not in ln.path_order, name


def test_every_line_has_its_hook_first_and_exports_no_test_hook():
    for name, ln in modes.LINES.items():
        assert ln.tier in registry.TIERS
        assert ln.hook == modes.HOOK and ln.path_order[0] == ln.hook, name
        assert not (set(ln.exports) & set(modes.TEST_HOOKS)), name
        assert not (set(ln.exports) & set(ln.unset)), name
        assert all(n in registry.LEVERS for n in ln.levers), name
        if ln.fpf:
            assert ln.fpf == "exact" == ln.exports["ODDE_TRIMUL"] and "odde_trimul_bind:" in ln.exports["FPF_IMPL"] and "FPF_TRIMUL_MODE" not in ln.exports, name
        else:
            assert not (set(ln.exports) & set(modes.FPF_SWITCHES)), name


def test_resolve_materialises_paths_and_shim():
    r = modes.resolve("exact", TREE, environ={})
    assert r.mode == "exact" and r.line.name == "S1"
    assert r.exports["CUEQ_TRITON_CACHE_DIR"] == os.path.join(TREE, "opt", "forward", "fast_inference", "levers", "KIT", "cueq_cache_shipped")
    assert r.sys_path[0].endswith(os.path.join("opt", "forward", "fast_inference", "levers", "ACCEL"))
    assert r.shim == os.path.join(r.sys_path[0], "sitecustomize.py") and os.path.isfile(r.shim)
    assert all(os.path.isdir(d) for d in r.sys_path)
    assert r.pythonpath() == os.pathsep.join(r.sys_path)
    assert modes.describe_line(r).startswith("S1(hook=ACCEL; ODDE_SERVED_LEVERS=1 CUEQ_TRITON_CACHE_DIR=") and "ODDE_TRIMUL=exact" in modes.describe_line(r) and "FPF_TRIMUL_MODE" not in modes.describe_line(r)
    assert modes.describe_line(modes.resolve("fast", TREE, environ={})).startswith("LSTAR2A(hook=ACCEL; ")


def test_resolve_notes_switches_that_must_be_absent():
    r = modes.resolve("exact", TREE, environ={"ODDE_ARM_U": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert any("ODDE_ARM_U" in n for n in r.notes) and any("PYTORCH_CUDA_ALLOC_CONF" in n for n in r.notes)


def test_off_resolves_to_nothing():
    r = modes.resolve("off", TREE)
    assert r.line is None and r.exports == {} and r.sys_path == [] and r.shim is None
    assert modes.describe_line(r) == "off"


def test_lines_by_name_and_unknown_names():
    r = modes.resolve("lstar2a", TREE)
    assert r.mode is None and r.line.name == "LSTAR2A" and r.exports["ODDE_ARM_U"] == "1" and "ODDE_ARM_T_TRIMUL" in r.unset
    with pytest.raises(ValueError):
        modes.resolve("opt7x", TREE)
    with pytest.raises(ValueError):
        modes.check_mode("faster")


def test_kit_dirs_and_sub_paths_resolve_inside_the_tree():
    assert set(modes.KIT_DIRS) == {"fast_inference"}
    for key in modes.XL_PATH_ORDER + (modes.OFFLOAD, modes.DITFAST_TOOLS):
        assert os.path.isdir(modes.kit_dir(TREE, key)), key
    assert modes.kit_layer(TREE) == os.path.join(TREE, "opt", "forward", "fast_inference", "levers", "KIT")
    with pytest.raises(KeyError):
        modes.kit_dir(TREE, "no_such_kit")

