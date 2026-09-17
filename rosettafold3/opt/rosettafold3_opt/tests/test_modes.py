"""The mode table, locked to the kits' own files: modes.py is the ONE place a mode is defined; configs carry no switch."""
import os
import re
import sys

import pytest

from .. import modes, registry, stack

KIT_LINE_ROW = {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "1"}           # the patched files' two flags, on
FPF_ARM_FAST = "fast.fast+gflash+ttr+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm"                       # the fast mode: the FPF add-on's kernels with the Triton apb kernel and without the trunk graph (tg)
FPF_ARM_FAST_TG = "fast.fast+gflash+ttr+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm"                 # the add-on's line with apb: the fast kernels plus the pairformer-stack CUDA graph
FPF_LEVERS_FAST = ["fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res", "fpf_tg", "fpf_xmul", "fpf_xln", "fpf_msa", "warm"]     # the kernels + the pairformer-stack CUDA graph
FPF_LEVERS_FAST_NOTG = ["fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res", "fpf_xmul", "fpf_xln", "fpf_msa"]              # the fast mode: the kernels
KIT_LEVERS = ["graph", "graph_safe_ops", "hoist"]


def test_house_modes_are_off_exact_and_fast():
    assert modes.MODES == ("off", "exact", "fast", "big")
    assert set(modes.KIT_MODES) == set(modes.MODES)
    assert modes.KIT_MODES["fast"].fpf_arm == FPF_ARM_FAST
    assert modes.KIT_MODES["fast"].kit_levers == ("mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite") and modes.KIT_MODES["exact"].kit_levers == ("xtr", "confhoist", "hostlean", "prefetch", "awrite")
    assert modes.KIT_MODES["exact"].fpf_arm == modes.FPF_ARM_TG_SAPB == "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm" and modes.KIT_MODES["off"].fpf_arm is None
    assert modes.KIT_MODES["fast"].switches == modes.KIT_MODES["exact"].switches == KIT_LINE_ROW   # fast = the exact row + the FPF arm
    assert modes.KIT_MODES["big"].switches == {**KIT_LINE_ROW, "RF3_CUDAGRAPH": "0"} and modes.kit_mode("big").fpf_arm == "fast.big+gflash+ttr+apb.big+res+xmul.eager+xln+msa.pwa" and modes.kit_mode("big").kit_levers == ("dtk", "confhoist", "confln", "hostlean", "prefetch", "awrite")   # big = the fast row by reference + the memory levers; fast's arm minus the components they disengage (resolved parametrically)
    with pytest.raises(ValueError):
        modes.check_mode("faster")
    with pytest.raises(ValueError):
        modes.resolve("faster", stack.kit_home())


def test_default_mode_rule():
    """The default is fast (the standing rule: the default is always fast, judged at the amortized warm steady state; exact is the
    mode that reproduces stock's outputs); with the switch unset the package exports nothing (off)."""
    assert modes.DEFAULT_MODE == "fast"
    assert modes.DEFAULT_MODE in modes.MODES and modes.DEFAULT_MODE in modes.KIT_MODES
    assert modes.check_mode(None) == modes.DEFAULT_MODE and modes.check_mode("") == modes.DEFAULT_MODE
    from .. import _autoload, cli
    assert cli.mode_of(None) == modes.DEFAULT_MODE
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            sys.meta_path.remove(f)
    assert _autoload.install({}) is None                         # ROSETTAFOLD3_OPT unset: no hook, no row exported
    assert modes.resolve("off", stack.kit_home()).switches == {}


def test_exact_row_is_the_two_flags_on():
    assert modes.KIT_MODES["exact"].switches == KIT_LINE_ROW
    assert modes.resolve("exact", stack.kit_home()).switches == KIT_LINE_ROW


def test_fast_arm_is_the_component_stack_with_apb_and_tg_under_its_budget():
    """KIT_MODES["fast"] applies the adapter's component stack with the Triton apb kernel and the pairformer-stack graph (tg, under the
    fast-kernel token budget: tgbudget.MAX_I_FAST) with the warm sub-step; the exact mode's arm is tg+sapb@L1.warm."""
    assert modes.FPF_ARM == FPF_ARM_FAST == FPF_ARM_FAST_TG == modes.KIT_MODES["fast"].fpf_arm and modes.fpf_lever_subs(modes.FPF_ARM) == ["warm"]
    r = modes.resolve("fast", stack.kit_home(), fpf_home=stack.fpf_home())
    assert r.fpf_arm == FPF_ARM_FAST and r.fpf_components == ["fast.fast", "gflash", "ttr", "apb.fast", "res", "tg", "xmul.eager", "xln", "msa"]
    assert r.levers == KIT_LEVERS + FPF_LEVERS_FAST + ["mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite"] and r.levers_off == ["fpf_sapb", "fpf_dattn", "fpf_xatt", "fpf_smsa", "dtk", "xtr"]
    assert r.tree_state == "patched" and r.interpreter == "opt" and r.is_house_mode and r.notes == []
    assert modes.KIT_MODES["exact"].fpf_arm == modes.FPF_ARM_TG_SAPB == "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm"


def test_fpf_arm_grammar_is_the_adapters():
    """fpf_components parses what the adapter's apply_arm accepts (its component vocabulary, the @L suffix) and refuses the rest."""
    assert modes.fpf_components("fast+gflash+ttr+apb+tg+res@L1") == (["fast", "gflash", "ttr", "apb", "tg", "res"], True)
    assert modes.fpf_components("tg+sapb@L1") == (["tg", "sapb"], True)
    assert modes.fpf_components("fast") == (["fast"], None)
    assert modes.fpf_components("@L1") == (["stock"], True)
    for bad in ("fast+turbo@L1", "fast@L2", "fast+@Lx", "tg+sapb@L0", "exact@L1", "fast+capb@L1", "fast+ctr@L1", "fast+flash@L1"):
        with pytest.raises(ValueError):
            modes.fpf_components(bad)
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    for p in modes.FPF_COMPONENTS:                                # every component name this package knows is one the adapter parses
        assert re.search(r"[\"']%s[\"']" % re.escape(p), src), p
    assert modes.fpf_levers_of(None) == [] and modes.fpf_levers_of("stock@L1") == []
    assert modes.fpf_levers_of("tg@L1") == ["fpf_tg"] and modes.fpf_levers_of("stock+tg@L1") == ["fpf_tg"]


def test_an_arm_may_not_contradict_the_row():
    """An arm's @L state re-sets the kit levers in-process (set_kit_levers); a row whose switches say otherwise is refused by name."""
    km = modes.KitMode("bad", {"RF3_CUDAGRAPH": "0", "RF3_HOIST": "1"}, "patched", "test", fpf_arm="fast@L1")
    modes.KIT_MODES["_test_bad"] = km
    try:
        with pytest.raises(ValueError, match="set_kit_levers"):
            modes.resolve("_test_bad", stack.kit_home())
    finally:
        del modes.KIT_MODES["_test_bad"]


def test_the_modes_are_the_only_selections():
    """Four modes; any other name (a lever composition such as exact_graphoff) is refused by name."""
    assert set(modes.MODES) == {"off", "exact", "fast", "big"}
    with pytest.raises(ValueError, match="unknown mode"):
        modes.resolve("exact_graphoff", stack.kit_home())
    re_ = modes.resolve("exact", stack.kit_home(), fpf_home=stack.fpf_home())                                   # the exact mode carries the package lever xtr beside its graph arm
    assert re_.fpf_arm == modes.FPF_ARM_TG_SAPB == "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm" and re_.kit_levers == ["xtr", "confhoist", "hostlean", "prefetch", "awrite"] and re_.is_house_mode and re_.notes == []
    assert re_.levers == KIT_LEVERS + ["fpf_tg", "fpf_sapb", "fpf_xatt", "fpf_xmul", "fpf_xln", "fpf_smsa", "warm", "xtr", "confhoist", "hostlean", "prefetch", "awrite"] and re_.fpf_components == ["tg", "sapb", "xatt", "xmul.eager", "xln", "smsa"]


def test_the_kit_file_reads_the_switches_and_nothing_else():
    table = modes.flag_table(os.path.join(stack.kit_home(), modes.GRAPH_FLAGS_RELPATH))
    assert table == {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "0"}  # the default flip of the patched file (graph_flags.py CUDAGRAPH_MODE): stock on a patched tree is not stock
    assert set(table) == set(modes.SWITCHES)
    r = modes.resolve("exact", stack.kit_home())
    assert r.notes == []


def test_fpf_knobs_match_the_adapters_own_table():
    """FPF_KNOBS are the adapter's os.environ.get defaults, read by the same AST reader; none is exported by a mode."""
    table = modes.flag_table(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH))
    assert set(table) == set(modes.FPF_KNOBS), (set(table) ^ set(modes.FPF_KNOBS))
    for k, v in modes.FPF_KNOBS.items():
        assert table[k] == v, (k, table[k], v)
        assert k.startswith(modes.FPF_ENV_PREFIXES)
    r = modes.resolve("fast", stack.kit_home(), fpf_home=stack.fpf_home())
    assert r.fpf_knobs == modes.FPF_KNOBS
    assert modes.resolve("exact", stack.kit_home(), fpf_home=stack.fpf_home()).fpf_knobs == modes.FPF_KNOBS      # exact carries an arm (tg+sapb): the adapter's knob table applies


def test_no_mode_exports_a_knob_or_an_instrument():
    for km in list(modes.KIT_MODES.values()):
        assert set(km.switches) <= set(modes.SWITCHES)                          # the lever switches, nothing else
        assert not {k for k in km.switches if k.startswith(modes.FPF_ENV_PREFIXES)}
        assert "replay" not in km.switches.values()                # RF3_CUDAGRAPH=replay (graph_flags.py:6): a test-driver value, never a mode's


def test_registry_levers_cover_the_switches_and_the_arm_components():
    assert set(registry.switches()) >= set(modes.SWITCHES)
    for sw, names in modes.LEVERS_OF_SWITCH.items():
        for n in names:
            assert n in registry.LEVERS and registry.LEVERS[n].cls == registry.CLASS_FORWARD and registry.LEVERS[n].component is None
    for comp, names in modes.LEVERS_OF_FPF.items():
        assert comp in modes.FPF_COMPONENTS + ("fast",)
        for n in names:
            lv = registry.LEVERS[n]
            assert lv.kit == registry.FPF and lv.switch is None and lv.cls == registry.CLASS_FORWARD and lv.probe[0].startswith("fpf_")
    assert set(registry.FPF_LEVERS) == set(modes.FPF_LEVERS)
    for n, lv in registry.FPF_LEVERS.items():
        assert lv.component in modes.LEVERS_OF_FPF and n in modes.LEVERS_OF_FPF[lv.component]
        assert "Tier" in lv.tier
    for lv in registry.LEVERS.values():
        for f in lv.kit_files:
            assert os.path.exists(os.path.join(stack.opt_root(), lv.kit, f)), (lv.name, f)


def test_configs_carry_no_lever_switch():
    cfg_dir = os.path.join(stack.tree_root(), "configs")
    for fn in os.listdir(cfg_dir):
        text = open(os.path.join(cfg_dir, fn)).read()
        for ln in text.splitlines():
            ln = ln.split("#", 1)[0]
            assert not re.search(r"(^|\s|export\s+)(RF3_|RFD3_|FPF_RF3_|FPF_TRIMUL_|FOUNDRY_DET_SCATTER|CUBLAS_WORKSPACE_CONFIG|NVIDIA_TF32_OVERRIDE)[A-Z_0-9]*=", ln), (fn, ln)
            assert "ROSETTAFOLD3_OPT=" not in ln, (fn, ln)


def test_describe_line_grammar():
    assert modes.describe_line(modes.resolve("exact", stack.kit_home())) == "RF3_CUDAGRAPH=1,RF3_HOIST=1"
    assert modes.describe_line(modes.resolve("fast", stack.kit_home())) == "RF3_CUDAGRAPH=1,RF3_HOIST=1"    # the arm is printed beside the row
    assert modes.describe_line(modes.resolve("off", stack.kit_home())) == "none"
    assert modes.jit_cache_key("2.13.0", "9.0") == "torch2.13.0-sm90"


def test_the_row_sharded_levers_ride_every_mode_at_multi_gpu_only():
    """The row-sharded line's ROWPAIR_* levers (modes.TP_ENV) ride every mode at --n_gpu > 1 (fold.kit_env); at one GPU nothing is added and
    big's switch row is the shared exact / fast row."""
    from .. import fold, stack
    assert modes.reach_env(1) == {} and modes.reach_env(2) == dict(modes.TP_ENV) == modes.reach_env(8) == modes.tp_env()
    assert modes.kit_mode("big").switches == {**modes.KIT_MODES["fast"].switches, "RF3_CUDAGRAPH": "0"} and modes.KIT_MODES["exact"].switches == modes.KIT_MODES["fast"].switches   # the memory row runs no CUDA graph
    base = {k: v for k, v in os.environ.items() if k not in modes.TP_ENV}
    fast = modes.resolve("fast", stack.kit_home())
    assert all(fold.kit_env(fast, environ=base, n_gpu=2)[k] == v for k, v in modes.TP_ENV.items()) and not set(modes.TP_ENV) & set(fold.kit_env(fast, environ=base, n_gpu=1))


def test_a_row_names_at_most_one_diffusion_transformer_lever():
    """mkdit replaces the token diffusion transformer's blocks whole; dtk (the package) and dattn (the FPF arm component) serve the attention
    inside them: a mode / row carries at most one of the three (modes.DIT_LEVERS; mkdit.enable refuses by name otherwise); the fast mode is
    the only mode naming mkdit, the memory mode replaces it by dtk (big.REPLACED_KIT: the pair-bias layout is an [I, I]-class holder) and
    so no composition meets mkdit at --n_gpu > 1 (big only); no selectable name exists for 'fast with dtk'."""
    from .. import big
    assert modes.DIT_LEVERS == ("mkdit", "dtk") and set(modes.DIT_LEVERS) <= set(modes.KIT_LEVERS)
    for name, km in list(modes.KIT_MODES.items()):
        km = modes.kit_mode(name) if name == "big" else km
        comps = modes.fpf_components(km.fpf_arm)[0] if km.fpf_arm else []
        dit = [lv for lv in km.kit_levers if lv in modes.DIT_LEVERS] + (["dattn"] if "dattn" in comps else [])
        assert len(dit) <= 1, (name, dit)
    assert big.REPLACED_KIT == {"mkdit": "dtk"} and "mkdit" in big.DISENGAGED_KIT and modes.kit_mode("big").kit_levers == ("dtk", "confhoist", "confln", "hostlean", "prefetch", "awrite")
    assert [n for n, km in modes.KIT_MODES.items() if "mkdit" in km.kit_levers] == ["fast"]
    assert "fast_dtk" not in modes.MODES


def test_the_row_sharded_line_exports_its_levers_at_multi_gpu_only():
    """modes.TP_ENV (the ROWPAIR_* host-parking levers of the `--n_gpu P > 1` line) rides fold.kit_env / stack.activate at P > 1 in every mode
    and never at P == 1 (the adapter installs nothing there); the row is the same whatever the caller's environment carries."""
    from .. import fold, stack
    assert modes.TP_ENV["ROWPAIR_PARK_ZINIT"] == "1"
    assert all(k.startswith("ROWPAIR_") for k in modes.TP_ENV)                  # the core's names (opt_core/mem/rowpair/API.md), no kit alias
    assert modes.tp_env() == modes.TP_ENV and modes.reach_env(2) == modes.TP_ENV == modes.reach_env(8)
    assert not set(modes.TP_ENV) & set(modes.reach_env(1))
    base = {k: v for k, v in os.environ.items() if k not in modes.TP_ENV}
    big = modes.resolve("big", stack.kit_home())
    env2 = fold.kit_env(big, environ=base, n_gpu=2)
    assert {k: env2[k] for k in modes.TP_ENV} == modes.TP_ENV
    assert fold.kit_env(big, environ={**base, "ROWPAIR_PARK_ZINIT": "0"}, n_gpu=2)["ROWPAIR_PARK_ZINIT"] == "1"   # the row's value, whatever the caller's environment carries
    assert not set(modes.TP_ENV) & set(fold.kit_env(big, environ=base, n_gpu=1))
    for km in list(modes.KIT_MODES.values()):
        assert not set(km.switches) & set(modes.TP_ENV)                            # no mode's ROW carries them: they belong to the resource axis, not to a row's numerics


def test_fpf_subwords_are_the_rows_modules_component_words():
    """FPF_SUBWORDS restates each rows module's COMPONENT_WORDS (the adapter's truth) word for word; read back by AST so drift fails here."""
    import ast
    for comp, rel in modes.FPF_SUBWORD_MODULES.items():
        src = open(os.path.join(stack.fpf_home(), rel)).read()
        words = None
        for n in ast.parse(src).body:
            if isinstance(n, ast.Assign) and any(getattr(tg, "id", "") == "COMPONENT_WORDS" for tg in n.targets):
                words = ast.literal_eval(n.value)
        assert words is not None and comp in words, (comp, rel)
        assert tuple(words[comp]) == modes.FPF_SUBWORDS[comp], (comp, words[comp], modes.FPF_SUBWORDS[comp])
    assert set(modes.FPF_SUBWORDS) == set(modes.FPF_SUBWORD_MODULES)

