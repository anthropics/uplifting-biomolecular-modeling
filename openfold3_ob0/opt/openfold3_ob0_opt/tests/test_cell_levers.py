"""The package's cell levers (pairfused.py: the pair track on opt_core cells; rollout.py: the sampler under bf16 autocast; dit_attn.py: the
diffusion transformer's attention on the core's flash kernel): switch grammar, registry rows, the stack's core gates,
and the fast line's composition — CPU only (no model, no GPU)."""
import os

import pytest

from openfold3_ob0_opt import _autoload, modes, registry, stack
from openfold3_ob0_opt.cells import dit_attn, pairfused, rollout

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_switch_grammar():
    assert pairfused.levers_requested({}) == [] and pairfused.levers_requested({"OPENFOLD3_OB0_OPT_PAIR": " trimul_v4 : pair_transition "}) == ["trimul_v4", "pair_transition"]
    assert pairfused.levers_requested({"OPENFOLD3_OB0_OPT_PAIR": ":".join(pairfused.LEVER_NAMES)}) == list(pairfused.LEVER_NAMES)
    with pytest.raises(ValueError):
        pairfused.levers_requested({"OPENFOLD3_OB0_OPT_PAIR": "trimul_v5"})
    assert rollout.requested({}) is False and rollout.requested({"OPENFOLD3_OB0_OPT_ROLLOUT": "bf16"}) is True
    with pytest.raises(ValueError):
        rollout.requested({"OPENFOLD3_OB0_OPT_ROLLOUT": "fp16"})


def test_declared_variables():
    assert set(modes.PAIR_ENVS) | {modes.PAIRFUSED_CHAIN_ENV} | set(modes.ROLLOUT_ENVS) <= set(_autoload.DECLARED)   # the env route's mistyped-switch gate knows every cell-lever variable


def test_registry_rows_and_gates():
    for name in (*pairfused.LEVER_NAMES, "rollout_bf16"):
        lv = registry.LEVERS[name]
        assert lv.kit == "cells" and lv.tier == registry.TOLERANCE and lv.modes == ("fast", "big/resident"), name   # big/resident = fast minus the measured-cost levers (trimul_v4 among the pair cells)
        assert (name in stack.CORE_CELL_LEVERS or name == "rollout_bf16") and name in stack._PROBES                   # gated on the core's carried bytes at activation; probed after the model module executes
    assert stack.CORE_CELL_LEVERS["trimul_v4"] == ("fpf_trimul_v4",) and stack.core_kernel_exports() == {}   # the v4 TriMul kernel's launch cells are the core's own table: nothing exported
    assert not os.path.exists(os.path.join(HOME, "opt", "openfold3_ob0_opt", "cells", "fpf_trimul_v4_cells.json"))


def test_fast_line_composition():
    """The fast line = ONE line at stock's precision (bf16-mixed): the fast-inference kit's exact levers + the three
    pair-track cells (their triangle attention the core's provider's by the tier word) + rollout_bf16, on the stock runner yaml itself (the line differs from stock by its levers alone); no other line carries the cells' switches."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OB0_OPT",))}
    assert modes.LINES[("fast", None)].runner_yaml == modes.STOCK_YAML                    # the line runs on the stock runner yaml itself
    fast = modes.resolve("fast", HOME, environ=env)
    assert not fast.conflicts and not getattr(fast, "pending", [])
    assert fast.exports["OPENFOLD3_OB0_OPT_PAIR"] == "trimul_v4:triatt_block:pair_transition" and fast.exports["OPENFOLD3_OB0_OPT_PAIR_CORE"] == "provider" and "OF3T_TRIATT" in fast.unsets
    assert {"trimul_v4", "triatt_block", "triatt_provider", "pair_transition", "dit_attn"} <= set(fast.levers) and not any(l.startswith("triatt_flash") for l in fast.levers)
    big = modes.resolve("big", HOME, environ=env, n_tokens=2565)
    assert big.exports["OPENFOLD3_OB0_OPT_PAIR_CORE"] == "provider:big" and "OF3T_TRIATT" in big.unsets   # the big line asks the provider's big tier word
    assert fast.exports["OPENFOLD3_OB0_OPT_DIT"] == "flash_bias_attn" and "OPENFOLD3_OB0_OPT_DIT_MIN_TOKENS" not in fast.exports   # the lever's switch; its token gate stays the caller's (default 256)
    assert ("rollout_bf16" in fast.levers) == ("OPENFOLD3_OB0_OPT_ROLLOUT" in fast.exports)
    assert fast.hooks[0] == "cells" and fast.entry_hook.endswith(os.path.join("hooks", "cells", "sitecustomize.py"))
    assert fast.exports[modes.PAIRFUSED_CHAIN_ENV].endswith("of3t_hook")                                    # the cells hook chains to the trunk-kernels hook, which chains to the kit levers
    for key, ln in modes.LINES.items():                                                        # no other line carries the cells' switches
        if key != ("fast", None):
            if key == ("big", "resident"):                                                            # the memory line carries the fast line's pair and sampler cells (test_modes pins which)
                continue
            assert not any(k.startswith(("OPENFOLD3_OB0_OPT_PAIR", "OPENFOLD3_OB0_OPT_ROLLOUT")) for k in ln.env), key


def test_trunk_refusal_words_carry_no_token_ceiling():
    """The TriMul cell has no size ceiling (large N may exceed GPU memory; see the core's fpf_trimul_v4 HAZARDS for the workspace formula): the only
    size word the kit names is the cell's token floor (`n<`); no `n>` ceiling word is an expected route, so any other refusal on the trunk shape fails
    the run under strict."""
    assert pairfused.STRICT_EXEMPT == ("n<", "N<", "chunk", "training", "stream:float32", "cell:")   # `cell:`: a call class the core's tables give to the stock statement by name (trimul provider STOCK_ROWS on the N<=100 bucket; tri-attention engage cells)
    # + an fp32 pair stream (the confidence head's pairformer_dtype=float32 stack; the fp32 line): a route by design, no size word
    assert not any(w.startswith(("n>", "N>")) for w in pairfused.STRICT_EXEMPT)
    assert not any(w.lower().startswith("n>") for w in pairfused.STRICT_EXEMPT)
    import re
    src = open(pairfused.__file__).read()
    assert not re.search(r"\bN_MAX\b", src) and not re.search(r"""["']n>|f["']n>""", src)   # no ceiling constant read, no `n>` refusal word built


def test_multi_gpu_lines_are_big_lines_only():
    """P > 1 runs only under `big` (check_n_gpu_route): every multi-GPU line is a big line."""
    import pytest
    assert set(modes.TP_LINES) <= {line for (mode, line) in modes.LINES if mode == "big"}
    with pytest.raises(ValueError):
        modes.check_n_gpu_route("fast", None, n_gpu="2", environ={})


def test_ending_node_bias_frame_follows_upstream_0_5():
    """OpenFold3 >= 0.5.0 projects the ENDING node's triangle bias from the un-transposed pair rep (layers/triangular_attention.py
    `transpose_bias`, set True by latent/base_blocks.py PairBlock.tri_att_start_end) = opt_core.attn.pair_fused's bias_frame 'z'; the starting
    node's two frames coincide ('x'). The cell passes the frame per node — a port that dropped it would serve 0.4.x arithmetic silently."""
    import inspect
    assert pairfused.BIAS_FRAME == {False: "x", True: "z"}
    src = inspect.getsource(pairfused._install_triatt_block)
    assert "bias_frame=BIAS_FRAME[ending]" in src


def test_the_v4_cells_are_the_cores_table():
    """No kit TriMul cell table: the core's fpf_trimul_v4 table carries the launch cells per (cc, triton) and the activation exports no FPF_TRIMUL_V4_CELLS."""
    from opt_core.kernels.fpf_trimul_v4 import table as T
    core = T.load_table()
    assert any(k.startswith("9.0|") for k in T.rows(core)) and any(k.startswith("8.0|") for k in T.rows(core))
    assert "FPF_TRIMUL_V4_CELLS" not in {k for v in stack.core_kernel_exports().values() for k in v}


def test_dit_attn_is_in_the_fast_line():
    """dit_attn is a lever of the fast line: the line exports OPENFOLD3_OB0_OPT_DIT=flash_bias_attn and names the lever right after rollout_bf16;
    no other line carries or exports it; a disagreeing preset under fast is a conflict by name; `off` requests nothing; the switch grammar, the
    declared variables, the registry row, the core gate and the probe are the cell levers'."""
    var = "OPENFOLD3_OB0_OPT_DIT"
    assert dit_attn.ENV == var and modes.DIT_ENVS == (var, dit_attn.ENV_MIN, dit_attn.ENV_CORE) and not hasattr(modes, "OPT_IN_LEVERS")
    assert dit_attn.requested({}) is False and dit_attn.requested({var: "flash_bias_attn"}) is True and dit_attn.requested({var: "flash_bias_attn", dit_attn.ENV_CORE: "dtk"}) is True and dit_attn.VALUES == ("flash_bias_attn",)
    with pytest.raises(ValueError):
        dit_attn.requested({var: "flash_bias_attn", dit_attn.ENV_CORE: "sdpa"})
    with pytest.raises(ValueError):
        dit_attn.requested({var: "sdpa"})
    assert dit_attn.fallbacks() == {} and dit_attn.census_line() == "[openfold3_ob0-opt/dit_attn] LEVER name=dit_attn state=off impl=None gate=none high_precision_asked=0 high_precision_overridden=0 core=pending:auto core_served=0 core_refused=none modes=none"
    assert set(modes.DIT_ENVS) <= set(_autoload.DECLARED)                                      # the env route's mistyped-switch gate knows the lever's two variables
    lv = registry.LEVERS["dit_attn"]
    assert lv.kit == "cells" and lv.tier == registry.TOLERANCE and lv.modes == ("fast", "big/resident") and lv.env_keys == modes.DIT_ENVS
    assert stack.CORE_CELL_LEVERS["dit_attn"] == ("dtk_kernels", "apb_attn") and "dit_attn" in stack._PROBES   # gated on the core's carried bytes at activation; probed after the model module executes
    fast_line = modes.LINES[("fast", None)]
    assert fast_line.env[var] == "flash_bias_attn" and list(fast_line.levers).index("dit_attn") == list(fast_line.levers).index("rollout_bf16") + 1
    assert [key for key, ln in modes.LINES.items() if var in ln.env or "dit_attn" in ln.levers] == [("fast", None), ("big", "resident")]   # the fast line and the memory line (fast minus the measured-cost levers) carry it; no line unsets it
    assert not any(var in ln.unset for ln in modes.LINES.values())
    env = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OB0_OPT",))}
    fast = modes.resolve("fast", HOME, environ=env)
    assert "dit_attn" in fast.levers and fast.exports[var] == "flash_bias_attn" and f" {var}=flash_bias_attn " in modes.describe_line(fast) and not fast.conflicts
    same = modes.resolve("fast", HOME, environ={**env, var: "flash_bias_attn"})                 # the line's own value preset: the line as written
    assert same.levers == fast.levers and same.exports == fast.exports and not same.conflicts
    bad = modes.resolve("fast", HOME, environ={**env, var: "sdpa"})
    assert any(c == f"{var}='sdpa' preset, the fast line sets 'flash_bias_attn'" for c in bad.conflicts), bad.conflicts
    assert "dit_attn" in modes.resolve("big", HOME, environ=dict(env), n_tokens=3000).levers          # the one-GPU memory line carries it with the fast line's other memory-neutral cells
    for mode, kw in (("exact", {}), ("big", {"n_tokens": 3000, "n_gpu": 2})):
        r = modes.resolve(mode, HOME, environ={**env, **({"OPENFOLD3_OB0_OPT_N_GPU": "2", "OF3TP_RANK": "0", "OF3TP_WORLD": "2"} if kw.get("n_gpu") else {})}, **kw)
        assert "dit_attn" not in r.levers and var not in r.exports and "cells" in r.hooks, (mode, kw)   # the other lines do not name the lever; every kit line mounts the cells hook (exact: its exact-class cells, big resident / tp: the runner-side cells)
    off = modes.resolve("off", HOME, environ={**env, var: "flash_bias_attn"})
    assert off.levers == [] and not off.conflicts


def test_the_sampler_cells_are_in_the_lines_that_carry_them():
    """dit_glue / token_agg (fast class) on the fast line, atom_hoist (exact class) on the exact AND fast lines; switch grammar; registry rows;
    the stack's core gates and probes; the adapters bind this kit's names onto opt_core.of3_sampler."""
    from openfold3_ob0_opt.cells import apb_trunk, atom_hoist, atom_window, dit_glue, templ_embed, token_agg
    assert dit_glue.requested({}) is False and dit_glue.requested({"OPENFOLD3_OB0_OPT_DIT_GLUE": "1"}) is True
    assert dit_glue.requested({"OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_DIT_GLUE_CORE": "dtk", "OPENFOLD3_OB0_OPT_DIT_GLUE_MIN_TOKENS": "512"}) is True
    assert dit_glue.requested({"OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_DIT_GLUE_ROWS": "block"}) is True and dit_glue.ROWS_VALUES == ("all", "block")
    for bad in ({"OPENFOLD3_OB0_OPT_DIT_GLUE": "on"}, {"OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_DIT_GLUE_ROWS": "atom"}, {"OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_DIT_GLUE_CORE": "sdpa"}, {"OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_DIT_GLUE_MIN_TOKENS": "-3"}):
        with pytest.raises(ValueError):
            dit_glue.requested(bad)
    assert token_agg.requested({}) is False and token_agg.requested({"OPENFOLD3_OB0_OPT_TOKEN_AGG": "seg_reduce"}) is True
    assert templ_embed.requested({}) is False and templ_embed.requested({"OPENFOLD3_OB0_OPT_TEMPL_EMBED": "1"}) is True and (templ_embed.ENV,) == modes.TEMPL_EMBED_ENVS
    assert templ_embed._core.PREFIX == "[openfold3_ob0-opt/templ_embed]" and templ_embed._core.M_TEMPLATE == "openfold3.core.model.latent.template_module" and templ_embed.STATE is templ_embed._core.STATE
    with pytest.raises(ValueError):
        templ_embed.requested({"OPENFOLD3_OB0_OPT_TEMPL_EMBED": "on"})
    with pytest.raises(ValueError):
        token_agg.requested({"OPENFOLD3_OB0_OPT_TOKEN_AGG": "scatter"})
    assert atom_hoist.requested({}) is False and atom_hoist.requested({"OPENFOLD3_OB0_OPT_ATOM_HOIST": "1"}) is True
    assert atom_hoist.requested({"OPENFOLD3_OB0_OPT_ATOM_HOIST": "1", "OPENFOLD3_OB0_OPT_ATOM_HOIST_INV": "bias", "OPENFOLD3_OB0_OPT_ATOM_HOIST_MAX_GB": "0.5"}) is True
    for bad in ({"OPENFOLD3_OB0_OPT_ATOM_HOIST": "yes"}, {"OPENFOLD3_OB0_OPT_ATOM_HOIST": "1", "OPENFOLD3_OB0_OPT_ATOM_HOIST_INV": "all"}, {"OPENFOLD3_OB0_OPT_ATOM_HOIST": "1", "OPENFOLD3_OB0_OPT_ATOM_HOIST_MAX_GB": "-1"}):
        with pytest.raises(ValueError):
            atom_hoist.requested(bad)
    assert (dit_glue.ENV, dit_glue.ENV_MIN, dit_glue.ENV_CORE, dit_glue.ENV_ROWS) == modes.DIT_GLUE_ENVS and (token_agg.ENV,) == modes.TOKEN_AGG_ENVS
    assert (atom_hoist.ENV, atom_hoist.ENV_MAX_GB, atom_hoist.ENV_INV) == modes.ATOM_HOIST_ENVS
    assert atom_window.requested({}) is False and atom_window.requested({"OPENFOLD3_OB0_OPT_ATOM_WINDOW": "1"}) is True
    assert atom_window.requested({"OPENFOLD3_OB0_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION": "ieee", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV": "hoist"}) is True
    for bad in ({"OPENFOLD3_OB0_OPT_ATOM_WINDOW": "on"}, {"OPENFOLD3_OB0_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION": "bf16"}, {"OPENFOLD3_OB0_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV": "dit"}):
        with pytest.raises(ValueError):
            atom_window.requested(bad)
    assert (atom_window.ENV, atom_window.ENV_PRECISION, atom_window.ENV_INV) == modes.ATOM_WINDOW_ENVS
    assert atom_window._core.ENV == atom_window.ENV and atom_window._core.PREFIX == "[openfold3_ob0-opt/atom_window]" and atom_window.STATE is atom_window._core.STATE
    assert atom_window._core.M_DIT == "openfold3.core.model.layers.diffusion_transformer" and atom_window._core.M_DM == "openfold3.core.model.structure.diffusion_module"
    assert (apb_trunk.ENV, apb_trunk.ENV_MIN, apb_trunk.ENV_HIGH_PRECISION, apb_trunk.ENV_SCOPE, apb_trunk.ENV_PRODUCER) == modes.APB_TRUNK_ENVS and apb_trunk._core.HIGH_PRECISION == "override"
    assert apb_trunk.requested({"OPENFOLD3_OB0_OPT_APB_TRUNK": "1"}) is True and apb_trunk.requested({"OPENFOLD3_OB0_OPT_APB_TRUNK": "1", "OPENFOLD3_OB0_OPT_APB_TRUNK_HIGH_PRECISION": "honour", "OPENFOLD3_OB0_OPT_APB_TRUNK_SCOPE": "trunk"}) is True
    assert apb_trunk._core.PRODUCER_DEFAULT == "ln_proj" and apb_trunk.requested({"OPENFOLD3_OB0_OPT_APB_TRUNK": "1", "OPENFOLD3_OB0_OPT_APB_TRUNK_PRODUCER": "mm"}) is True
    for bad in ({"OPENFOLD3_OB0_OPT_APB_TRUNK": "1", "OPENFOLD3_OB0_OPT_APB_TRUNK_HIGH_PRECISION": "fp32"}, {"OPENFOLD3_OB0_OPT_APB_TRUNK": "1", "OPENFOLD3_OB0_OPT_APB_TRUNK_SCOPE": "confidence"},
                {"OPENFOLD3_OB0_OPT_APB_TRUNK": "1", "OPENFOLD3_OB0_OPT_APB_TRUNK_PRODUCER": "fused"}):
        with pytest.raises(ValueError):
            apb_trunk.requested(bad)
    assert set(modes.DIT_GLUE_ENVS) | set(modes.TOKEN_AGG_ENVS) | set(modes.ATOM_WINDOW_ENVS) | set(modes.ATOM_HOIST_ENVS) | set(modes.APB_TRUNK_ENVS) <= set(_autoload.DECLARED)   # the env route's mistyped-switch gate knows every cell-lever variable
    assert atom_hoist._core.ENV == atom_hoist.ENV and atom_hoist._core.PREFIX == "[openfold3_ob0-opt/atom_hoist]" and atom_hoist._memo.PREFIX == "[openfold3_ob0-opt/rollout_memo]"
    assert dit_glue._core.HIGH_PRECISION == "override" and dit_glue.STATE is dit_glue._core.STATE and token_agg.STATE is token_agg._core.STATE and atom_hoist.STATE is atom_hoist._core.STATE
    for name in ("dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk", "templ_embed"):
        lv = registry.LEVERS[name]
        assert lv.kit == "cells" and lv.tier == (registry.EXACT if name == "atom_hoist" else registry.TOLERANCE) and lv.modes == (("exact/cueq", "fast") if name == "atom_hoist" else ("fast",) if name == "dit_glue" else ("fast", "big/resident")), (name, lv.modes)   # derived from the mode table (atom_hoist / dit_glue: absent from big/resident by number)
        assert (name in stack.CORE_CELL_LEVERS or name == "atom_hoist") and name in stack._PROBES
    assert stack.CORE_CELL_LEVERS["token_agg"] == ("dtk_kernels",) and stack.CORE_CELL_LEVERS["dit_glue"] == ("dtk_kernels", "apb_attn") and stack.CORE_CELL_LEVERS["apb_trunk"] == ("apb_attn",)
    assert stack.CORE_CELL_LEVERS["atom_window"] == ("atom_window",) and stack.CORE_CELL_LEVERS["templ_embed"] == ("templ_embed",)
    fast = modes.resolve("fast", HOME, environ={})
    assert fast.exports["OPENFOLD3_OB0_OPT_DIT_GLUE"] == "1" and fast.exports["OPENFOLD3_OB0_OPT_TOKEN_AGG"] == "seg_reduce" and fast.exports["OPENFOLD3_OB0_OPT_ATOM_HOIST"] == "1" and fast.exports["OPENFOLD3_OB0_OPT_APB_TRUNK"] == "1"
    assert "apb_trunk" in fast.levers and "templ_embed" in fast.levers and fast.exports["OPENFOLD3_OB0_OPT_TEMPL_EMBED"] == "1"
    assert not any("OPENFOLD3_OB0_OPT_TEMPL_EMBED" in ln.env for key, ln in modes.LINES.items() if key not in (("fast", None), ("big", "resident")))   # the fast line and the memory line carry it
    assert {"dit_glue", "token_agg", "atom_window", "atom_hoist"} <= set(fast.levers)
    assert fast.exports["OPENFOLD3_OB0_OPT_ATOM_WINDOW"] == "1" and not any(k in fast.exports for k in ("OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV"))   # the kernel defaults: tf32rn dots, own invariants
    exact = modes.resolve("exact", HOME, environ={})
    assert exact.hooks[0] == "cells" and exact.exports["OPENFOLD3_OB0_OPT_ATOM_HOIST"] == "1" and "atom_hoist" in exact.levers and not {"dit_glue", "token_agg"} & set(exact.levers)
    assert not any(k.startswith(("OPENFOLD3_OB0_OPT_PAIR", "OPENFOLD3_OB0_OPT_DIT", "OPENFOLD3_OB0_OPT_ROLLOUT", "OPENFOLD3_OB0_OPT_TOKEN_AGG", "OPENFOLD3_OB0_OPT_ATOM_WINDOW")) for k in modes.LINES[("exact", "cueq")].env)



def test_dit_attn_has_no_command_line_switch():
    """The lever is the fast line's, not a flag: `--dit-attn` is not a word of pred / check / warm (argparse exit 2) and the CLI carries no setter."""
    from openfold3_ob0_opt import cli
    ap = cli.build_parser()
    for verb, extra in (("check", []), ("warm", []), ("pred", ["--query-json", "q.json", "--output-dir", "/tmp/o"])):
        with pytest.raises(SystemExit) as ei:
            ap.parse_args([verb, "--mode", "fast"] + extra + ["--dit-attn", "on"])
        assert ei.value.code == 2, (verb, ei.value.code)
    assert not hasattr(cli, "apply_dit_attn_flag") and not hasattr(cli, "DIT_ATTN_WORDS")


def test_a_preset_cell_switch_the_line_does_not_carry_is_refused_by_name():
    """The cells hook installs any cell lever whose own switch is set, and the exact line mounts that hook (atom_hoist): a fast-class cell switch
    preset in the caller's environment must be a conflict BY NAME on every line that does not export it (NOT ACTIVE), parameters of that family
    included; the line that carries the lever owns the family, so its parameters (size gates, caps, core pins) stay settable there."""
    for mode, kw in (("exact", {}), ("fast", {}), ("big", {"n_tokens": 3000})):
        base = modes.resolve(mode, HOME, environ={}, **kw)
        for fam in modes.CELL_FAMILIES:
            carried = fam[0] in base.exports
            for k in fam:
                r = modes.resolve(mode, HOME, environ={k: "1"}, **kw)
                hit = [c for c in r.conflicts if c.startswith(k + "=")]
                if carried:
                    assert bool(hit) == ((k in base.exports and base.exports[k] != "1") or k in base.unsets), (mode, k, r.conflicts)   # the carrying line owns the family: a knob it neither exports nor requires unset is the caller's
                    assert not any("does not carry this cell lever" in c for c in hit), (mode, k, hit)
                else:
                    assert hit and "does not carry this cell lever" in hit[0], (mode, k, r.conflicts)
    r = modes.resolve("exact", HOME, environ={"OPENFOLD3_OB0_OPT_ROLLOUT": "bf16", "OPENFOLD3_OB0_OPT_DIT_GLUE": "1", "OPENFOLD3_OB0_OPT_TOKEN_AGG": "seg_reduce", "OPENFOLD3_OB0_OPT_DIT": "flash_bias_attn"})
    assert len([c for c in r.conflicts if "does not carry this cell lever" in c]) == 4                                   # the audit's probe: four fast-class switches on --mode exact, four refusals
    assert modes.resolve("fast", HOME, environ={"OPENFOLD3_OB0_OPT_DIT_GLUE_MIN_TOKENS": "256", "OPENFOLD3_OB0_OPT_ATOM_HOIST_MAX_GB": "0.5"}).conflicts == []   # the fast line carries both: its knobs are the caller's
    assert modes.resolve("off", HOME, environ={"OPENFOLD3_OB0_OPT_DIT_GLUE": "1"}).conflicts == []          # off = the stock child with the kit's variables stripped: inert there
