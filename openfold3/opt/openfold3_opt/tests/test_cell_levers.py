"""The package's cell levers (pairfused.py: the pair track on opt_core cells; dit_attn.py: the diffusion transformer's attention on the core's flash
kernel; dit_glue.py / token_agg.py / atom_window.py / atom_hoist.py: the token DiT block schedule, the atom -> token aggregation, the fused atom-attention
windows, the atom path's per-rollout memo):
switch grammar, registry rows, the stack's core gates, and the fast line's composition by precision — CPU only (no model, no GPU)."""
import os

import pytest

from openfold3_opt import _autoload, modes, registry, stack
from openfold3_opt.cells import apb_trunk, atom_hoist, atom_window, dit_attn, dit_glue, pairfused, rollout, templ_embed, token_agg

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_switch_grammar():
    assert pairfused.levers_requested({}) == [] and pairfused.levers_requested({"OPENFOLD3_OPT_PAIR": " trimul_v4 : pair_transition "}) == ["trimul_v4", "pair_transition"]
    assert pairfused.levers_requested({"OPENFOLD3_OPT_PAIR": ":".join(pairfused.LEVER_NAMES)}) == list(pairfused.LEVER_NAMES)
    with pytest.raises(ValueError):
        pairfused.levers_requested({"OPENFOLD3_OPT_PAIR": "trimul_v5"})
    assert rollout.requested({}) is False and rollout.requested({"OPENFOLD3_OPT_ROLLOUT": "bf16"}) is True
    with pytest.raises(ValueError):
        rollout.requested({"OPENFOLD3_OPT_ROLLOUT": "fp16"})
    assert dit_attn.requested({}) is False and dit_attn.requested({"OPENFOLD3_OPT_DIT": "flash_bias_attn"}) is True
    with pytest.raises(ValueError):
        dit_attn.requested({"OPENFOLD3_OPT_DIT": "sdpa"})
    assert dit_glue.requested({}) is False and dit_glue.requested({"OPENFOLD3_OPT_DIT_GLUE": "1"}) is True
    assert dit_glue.requested({"OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_DIT_GLUE_CORE": "dtk", "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS": "512"}) is True
    assert dit_glue.requested({"OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_DIT_GLUE_ROWS": "block"}) is True and dit_glue.ROWS_VALUES == ("all", "block")
    for bad in ({"OPENFOLD3_OPT_DIT_GLUE": "on"}, {"OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_DIT_GLUE_ROWS": "atom"}, {"OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_DIT_GLUE_CORE": "sdpa"}, {"OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS": "-3"}):
        with pytest.raises(ValueError):
            dit_glue.requested(bad)
    assert token_agg.requested({}) is False and token_agg.requested({"OPENFOLD3_OPT_TOKEN_AGG": "seg_reduce"}) is True
    with pytest.raises(ValueError):
        token_agg.requested({"OPENFOLD3_OPT_TOKEN_AGG": "scatter"})
    assert atom_window.requested({}) is False and atom_window.requested({"OPENFOLD3_OPT_ATOM_WINDOW": "1"}) is True
    assert atom_window.requested({"OPENFOLD3_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OPT_ATOM_WINDOW_PRECISION": "tf32x3", "OPENFOLD3_OPT_ATOM_WINDOW_INV": "hoist"}) is True
    for bad in ({"OPENFOLD3_OPT_ATOM_WINDOW": "on"}, {"OPENFOLD3_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OPT_ATOM_WINDOW_PRECISION": "bf16"}, {"OPENFOLD3_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OPT_ATOM_WINDOW_INV": "dit"}):
        with pytest.raises(ValueError):
            atom_window.requested(bad)
    assert atom_hoist.requested({}) is False and atom_hoist.requested({"OPENFOLD3_OPT_ATOM_HOIST": "1"}) is True
    assert templ_embed.requested({}) is False and templ_embed.requested({"OPENFOLD3_OPT_TEMPL_EMBED": "1"}) is True
    with pytest.raises(ValueError):
        templ_embed.requested({"OPENFOLD3_OPT_TEMPL_EMBED": "on"})
    assert atom_hoist.requested({"OPENFOLD3_OPT_ATOM_HOIST": "1", "OPENFOLD3_OPT_ATOM_HOIST_INV": "bias", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB": "0.5"}) is True
    for bad in ({"OPENFOLD3_OPT_ATOM_HOIST": "yes"}, {"OPENFOLD3_OPT_ATOM_HOIST": "1", "OPENFOLD3_OPT_ATOM_HOIST_INV": "all"}, {"OPENFOLD3_OPT_ATOM_HOIST": "1", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB": "-1"}):
        with pytest.raises(ValueError):
            atom_hoist.requested(bad)


def test_declared_variables():
    assert set(modes.PAIR_ENVS) | {modes.PAIRFUSED_CHAIN_ENV} | set(modes.DIT_ENVS) | set(modes.ROLLOUT_ENVS) | set(modes.DIT_GLUE_ENVS) | set(modes.TOKEN_AGG_ENVS) | set(modes.ATOM_WINDOW_ENVS) | set(modes.TEMPL_EMBED_ENVS) | set(modes.ATOM_HOIST_ENVS) | set(modes.APB_TRUNK_ENVS) <= set(_autoload.DECLARED)
    assert (dit_glue.ENV, dit_glue.ENV_MIN, dit_glue.ENV_CORE, dit_glue.ENV_ROWS) == modes.DIT_GLUE_ENVS and (token_agg.ENV,) == modes.TOKEN_AGG_ENVS and (templ_embed.ENV,) == modes.TEMPL_EMBED_ENVS
    assert (dit_attn.ENV, dit_attn.ENV_MIN, dit_attn.ENV_CORE) == modes.DIT_ENVS and (apb_trunk.ENV, apb_trunk.ENV_MIN, apb_trunk.ENV_HIGH_PRECISION, apb_trunk.ENV_SCOPE, apb_trunk.ENV_PRODUCER) == modes.APB_TRUNK_ENVS
    assert apb_trunk.requested({"OPENFOLD3_OPT_APB_TRUNK": "1", "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION": "override", "OPENFOLD3_OPT_APB_TRUNK_SCOPE": "trunk"}) is True
    with pytest.raises(ValueError):
        apb_trunk.requested({"OPENFOLD3_OPT_APB_TRUNK": "1", "OPENFOLD3_OPT_APB_TRUNK_SCOPE": "confidence"})
    assert (atom_window.ENV, atom_window.ENV_PRECISION, atom_window.ENV_INV) == modes.ATOM_WINDOW_ENVS
    assert (atom_hoist.ENV, atom_hoist.ENV_MAX_GB, atom_hoist.ENV_INV) == modes.ATOM_HOIST_ENVS   # the env route's mistyped-switch gate knows every cell-lever variable


def test_registry_rows_and_gates():
    live = {m if l is None else f"{m}/{l}" for (m, l) in modes.LINES} | {m for (m, l) in modes.LINES}   # a registry row's `modes` names live lines only (mode or mode/line)
    for name, lv in registry.LEVERS.items():
        assert set(lv.modes) <= live, (name, lv.modes)
    EXACT_CELLS = ("atom_hoist", "castcache", "exactln", "apb_hoist", "trunk_graph", "post_release", "tuner_guard", "sync_hoist", "postfwd_mem", "loader_workers")   # exact-class cells (engine statements memoised / captured / released / re-blocked: no kernel of their own)
    RUNNER_CELLS = ("postfwd_mem", "loader_workers")                                          # the runner-side cells: every one-GPU kit line (exact, fast, big/resident)
    for name in (*pairfused.LEVER_NAMES, "rollout_bf16", "dit_attn", "dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk", "templ_embed", "castcache", "exactln", "ln_provider", "apb_hoist", "trunk_graph", "post_release", "tuner_guard", "sync_hoist", *RUNNER_CELLS):
        lv = registry.LEVERS[name]
        assert lv.kit == "cells" and lv.tier == (registry.EXACT if name in EXACT_CELLS else registry.TOLERANCE)
        BIG_ABSENT = ("dit_glue", "atom_hoist", "castcache", "trunk_graph")                       # the fast line's cells absent from big/resident; exactln rides it behind its 1024-token item gate; trimul_v4 rides it on the provider's word big
        want = ("exact", "fast", "big/resident") if name in RUNNER_CELLS else (("fast",) if name not in EXACT_CELLS or name in ("trunk_graph", "tuner_guard") else ("exact", "fast"))
        if name not in BIG_ABSENT and name not in RUNNER_CELLS:
            want = want + ("big/resident",)                                                   # every other cell rides the memory line too (big = fast minus the memory-cost levers)
        if name == "exactln":
            want = ("exact",)                                                                    # the LayerNorm binding's exact-class word rides the exact line alone; the fast / big lines carry the same binding as ln_provider (their tier word)
        assert lv.modes == want, name   # trunk_graph: exact class, on the fast line only (the exact trunk runs DS4Sci: uncapturable; the resident line's host-snapshot unit invalidates its capture)
        assert (name in stack.CORE_CELL_LEVERS or name in ("rollout_bf16",) + EXACT_CELLS) and name in stack._PROBES  # gated on the core's carried bytes at activation; probed after the model module executes
    assert stack.CORE_CELL_LEVERS["token_agg"] == ("dtk_kernels",) and stack.CORE_CELL_LEVERS["apb_trunk"] == ("apb_attn",) and stack.CORE_CELL_LEVERS["atom_window"] == ("atom_window",) and stack.CORE_CELL_LEVERS["templ_embed"] == ("templ_embed",)
    assert stack.CORE_CELL_LEVERS["dit_glue"] == stack.CORE_CELL_LEVERS["dit_attn"] == ("dtk_kernels", "apb_attn")   # the per-sample kernel and the all-samples pair-bias core both gated at activation
    assert stack.CORE_CELL_LEVERS["trimul_v4"] == ("fpf_trimul_v4",) and stack.CORE_KERNEL_EXPORTS == {}   # the v4 TriMul kernel's launch cells are the core's own table: nothing exported
    assert not os.path.exists(os.path.join(HOME, "opt", "openfold3_opt", "cells", "fpf_trimul_v4_cells.json"))


def test_fast_line_composition():
    env = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OPT",))}
    bf16 = modes.resolve("fast", HOME, environ=env)
    assert bf16.precision == modes.FAST_PRECISION == "bf16" and bf16.exports["OPENFOLD3_OPT_PAIR"] == "trimul_v4:triatt_block:pair_transition" and bf16.exports["OPENFOLD3_OPT_DIT"] == "flash_bias_attn" and bf16.exports["OPENFOLD3_OPT_ROLLOUT"] == "bf16"
    assert bf16.hooks[0] == "cells" and bf16.entry_hook.endswith(os.path.join("hooks", "cells", "sitecustomize.py"))
    assert bf16.exports[modes.PAIRFUSED_CHAIN_ENV].endswith(os.path.join("trunk_kernels", "of3t_hook"))          # the cells hook chains to the trunk-kernels hook, which chains to the kit levers
    assert bf16.exports["OPENFOLD3_OPT_DIT_GLUE"] == "1" and bf16.exports["OPENFOLD3_OPT_TOKEN_AGG"] == "seg_reduce" and bf16.exports["OPENFOLD3_OPT_ATOM_HOIST"] == "1"
    assert bf16.exports["OPENFOLD3_OPT_APB_TRUNK"] == "1" and bf16.exports["OPENFOLD3_OPT_TEMPL_EMBED"] == "1" and {"dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk", "templ_embed"} <= set(bf16.levers)
    assert bf16.exports["OPENFOLD3_OPT_ATOM_WINDOW"] == "1" and not any(k in bf16.exports for k in ("OPENFOLD3_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OPT_ATOM_WINDOW_INV"))   # the kernel defaults: tf32rn dots, own invariants
    assert all(bf16.exports[k] == "1" for k in ("OPENFOLD3_OPT_CASTCACHE", "OPENFOLD3_OPT_APB_HOIST", "OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_POST_RELEASE")) and {"castcache", "apb_hoist", "trunk_graph", "post_release"} <= set(bf16.levers)
    exact = modes.resolve("exact", HOME, environ=env)                                          # the exact line carries the cells hook for its exact-class cells (atom_hoist, castcache, apb_hoist, post_release)
    assert exact.hooks[0] == "cells" and all(exact.exports[k] == "1" for k in ("OPENFOLD3_OPT_ATOM_HOIST", "OPENFOLD3_OPT_CASTCACHE", "OPENFOLD3_OPT_APB_HOIST", "OPENFOLD3_OPT_POST_RELEASE"))
    assert {"atom_hoist", "castcache", "apb_hoist", "post_release"} <= set(exact.levers) and "trunk_graph" not in exact.levers and "OPENFOLD3_OPT_TRUNK_GRAPH" not in exact.exports
    assert not any(k.startswith(("OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_ROLLOUT", "OPENFOLD3_OPT_TOKEN_AGG", "OPENFOLD3_OPT_ATOM_WINDOW", "OPENFOLD3_OPT_APB_TRUNK", "OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_TEMPL")) for k in modes.LINES[("exact", "cueq")].env)
    for key, ln in modes.LINES.items():                                                        # no other line carries the cells' switches
        if key not in (("fast", None), ("exact", "cueq"), ("big", "resident")):              # big/resident carries the fast line's memory-neutral cells (test_modes pins which)
            assert not any(k.startswith(("OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_ROLLOUT", "OPENFOLD3_OPT_TOKEN_AGG", "OPENFOLD3_OPT_ATOM_WINDOW", "OPENFOLD3_OPT_ATOM_HOIST", "OPENFOLD3_OPT_APB", "OPENFOLD3_OPT_TEMPL", "OPENFOLD3_OPT_CASTCACHE", "OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_POST_RELEASE")) for k in ln.env), key


def test_trunk_refusal_words_carry_no_token_ceiling():
    """The TriMul cell has no size ceiling (large N may exceed GPU memory; see the core's fpf_trimul_v4 HAZARDS for the workspace formula): the only
    size word the kit names is the cell's token floor (`n<`); no `n>` ceiling word is an expected route, so any other refusal on the trunk shape fails
    the run under strict — except the core's `no-cell:` word (a shape its tables do not serve on this GPU runs the line's own statement, counted)
    and its `cell:` word (a call class the core's tables give to the stock statement by name: the trimul provider's STOCK_ROWS on the
    N<=100 bucket, the tri-attention engage cells)."""
    assert pairfused.STRICT_EXEMPT == ("n<", "N<", "chunk", "training", "no-cell:", "cell:")
    assert not any(w.lower().startswith("n>") for w in pairfused.STRICT_EXEMPT)
    import re
    src = open(pairfused.__file__).read()
    assert not re.search(r"\bN_MAX\b", src) and not re.search(r"""["']n>|f["']n>""", src)   # no ceiling constant read, no `n>` refusal word built


def test_the_v4_cells_are_the_cores_table():
    """No kit TriMul cell table: the core's fpf_trimul_v4 table carries the launch cells per (cc, triton) for cc 9.0 (H100) and cc 8.0 (A100), and the
    activation exports no FPF_TRIMUL_V4_CELLS."""
    from opt_core.kernels.fpf_trimul_v4 import table as T
    core = T.load_table()
    assert any(k.startswith("9.0|") for k in T.rows(core)) and any(k.startswith("8.0|") for k in T.rows(core))
    assert "FPF_TRIMUL_V4_CELLS" not in {k for v in stack.CORE_KERNEL_EXPORTS.values() for k in v}


def test_multi_gpu_lines_are_big_lines_only():
    """P > 1 runs only under `big` (check_n_gpu_route): every multi-GPU line is a big line."""
    import pytest
    assert set(modes.TP_LINES) <= {line for (mode, line) in modes.LINES if mode == "big"}
    with pytest.raises(ValueError):
        modes.check_n_gpu_route("fast", None, n_gpu="2", environ={})


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
    r = modes.resolve("exact", HOME, environ={"OPENFOLD3_OPT_ROLLOUT": "bf16", "OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_TOKEN_AGG": "seg_reduce", "OPENFOLD3_OPT_DIT": "flash_bias_attn"})
    assert len([c for c in r.conflicts if "does not carry this cell lever" in c]) == 4                                   # the audit's probe: four fast-class switches on --mode exact, four refusals
    assert modes.resolve("fast", HOME, environ={"OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS": "256", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB": "0.5"}).conflicts == []   # the fast line carries both: its knobs are the caller's
    assert modes.resolve("off", HOME, environ={"OPENFOLD3_OPT_DIT_GLUE": "1"}).conflicts == []          # off = the stock child with the kit's variables stripped: inert there
