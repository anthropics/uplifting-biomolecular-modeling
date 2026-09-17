"""The mode table is locked to the kits' own rows: switches, lever names, order, and tiers."""
import os
import subprocess
import sys

import pytest

from .. import modes, registry, stack

TRUNK = stack.kit_path("forward/trunk_levers")
DIT = stack.kit_path("forward/dit_hoist")


def test_mode_names_and_default():
    assert modes.MODE_NAMES == ("off", "exact", "fast", "big"), "the user-facing modes: exactly four, named by guarantee"
    assert set(modes.MODES) == set(modes.MODE_NAMES) == set(modes.EVIDENCE_ENV) == set(modes.CLI_ROUTE)
    assert modes.DEFAULT_MODE == "fast" and modes.PINNED_ROUTE == "worker", "the package default: fast wherever a fast mode ships (a literal value, never computed from the table)"
    for name in ("turbo", "exact_k", "exact_nk", "exact_fpf", "big_exact"):   # any other name is refused by name, never aliased
        assert name not in modes.MODES
        with pytest.raises(ValueError):
            modes.resolve(name)


def test_the_kits_kernels_off_default_row_is_no_modes_row():
    """Every mode's trunk row is resid,mask2 (kernels on) — the two levers the trunk add-on implements."""
    assert all(modes.env_row(m)["BOLTZ_LEVERS"] == "resid,mask2" for m in ("exact", "fast", "big"))


def test_exact_row_graph_and_hoist_switches_are_the_bz2dit_lgh_arm():
    row = modes.env_row("exact")
    assert {"BOLTZ_LEVERS", "BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_DIT_HOIST"} <= set(row) and modes._SAMPLER_ROW == {"BOLTZ_SAMPLER_ROLLOUT": "graph", "BOLTZ_DIT_HOIST": "2"}, "the roll-out replaces the per-step graph patch; the hoist stays"
    assert "BOLTZ_GRAPH_DIFFUSION" not in row and "graph_sampler" not in modes.levers("exact") and "graph_sampler" in registry.LEVERS, "the per-step graph patch is an available word in no row"


def test_fast_row_is_exact_plus_the_kits_opt_in_tier2_row():
    ex, fa = modes.env_row("exact"), modes.env_row("fast")
    assert set(ex) - set(fa) == {"BOLTZ_DIT_EXACT", "BOLTZ_TRIATTN_EXACT", "BOLTZ_EXACTLN_RESID"} == set(modes._EXACT_ONLY) and fa["BOLTZ_LEVERS"] == ex["BOLTZ_LEVERS"] and fa["BOLTZ_SAMPLER_ROLLOUT"] == ex["BOLTZ_SAMPLER_ROLLOUT"] and fa["BOLTZ_DIT_HOIST"] == ex["BOLTZ_DIT_HOIST"], "fast composes on the exact row (kernels on); the exact token-transformer schedule (BOLTZ_DIT_EXACT) is the exact row's alone — fast's token transformer is the fused bf16 step"
    delta = {k: v for k, v in fa.items() if ex.get(k) != v}
    from boltz2_opt import pairblock as PB
    assert delta.pop("BOLTZ_PAIRBLOCK") in PB.TIER2_VARIANTS, "fast's block variant is a Tier-2 one (0.3.23: `default`, the block's own keyed core)"
    assert delta == {"BOLTZ_SAMPLER_ALIGN": "jacobi64", "BOLTZ_PAIRBLOCK_C64": "1", "BOLTZ_TRANSITION": "fast", "BOLTZ_FPF_TRIMUL": "1", "BOLTZ_FPF_TRIMUL_PROVIDER": "fast", "BOLTZ_FPF_MSA": "opm,pwa",
                     "BOLTZ_PAIRFUSE": "bf16,trimul=core.fast,triatt=core.fast,transition=core.fast", "BOLTZ_SAMPLER_DIT": "bf16", "BOLTZ_ATOM": "keys,fused", "BOLTZ_ATOM_GEMM": "bf16", "BOLTZ_MSA2": "trans2", "BOLTZ_CONF": "condproj"}, "fast = exact's attachments in their Tier-2 variants (flash core inside the block at >= 300 tokens, chunked transitions served, fpf_trimul_v4) + the fused MSA-module kernels on every call + the layer driver with the triattn-native core row"
    assert {k: ex[k] for k in ("BOLTZ_FPF_TRIMUL", "BOLTZ_FPF_TRIMUL_PROVIDER", "BOLTZ_TRANSITION", "BOLTZ_PAIRBLOCK")} == {"BOLTZ_FPF_TRIMUL": "exact", "BOLTZ_FPF_TRIMUL_PROVIDER": "exact", "BOLTZ_TRANSITION": "exact", "BOLTZ_PAIRBLOCK": "cueq"}, "exact's pair-track cells"
    assert not any("TRIMUL_MIN_TOKENS" in k for m in modes.MODE_NAMES for k in modes.env_row(m)), "no TriMul token gate of this tree's: the core provider decides every (card, shape) cell"
    assert modes._F2_ROW == {"BOLTZ_TRIATTN": "flash", "BOLTZ_TRIATTN_MIN_TOKENS": "300"}, "the trunk add-on's flash patch row (the memory line's attention lever)"
    assert modes.resolve("fast")["worker_base"].endswith("bz_worker_levf2.py") and modes.resolve("exact")["worker_base"].endswith("bz_worker_lev.py")
    assert modes.resolve("fast")["kernels"] == "on" == modes.resolve("exact")["kernels"], "the flash kernel replaces cuEquivariance's triangle attention; the fused triangle multiplication stays upstream's"


def test_lever_lists_match_the_kit_constants_and_registry():
    assert stack._expand_levers("resid, mask2") == ["resid", "mask2"]              # the rows name the trunk levers literally
    assert modes.levers("exact")[2:4] == ["rollout", "dit_hoist"], "the roll-out attached at the sampler module's import, the hoist outermost (make_worker_variant.py)"
    EXACT_CELLS = {"align_aligncap", "dit_par", "dit_mask", "dit_sba", "dit_smx", "dit_glue", "fpf_trimul_exact", "fused_transition", "pairblock", "triattn_exact", "templ_skip", "exactln", "exactln_resid", "msa_pwa_exact", "msa_trans2_exact", "graph_trunk", "waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip", "atom_keys_gather", "atom_glue_hoist", "writer_overlap", "prefetch"}
    FAST_CELLS = {"align_jacobi64", "dit_fused", "pairblock", "flash_triattn", "pairblock_c64", "templ_skip", "fused_transition", "fpf_trimul", "fpf_opm", "fpf_pwa", "pairfuse", "exactln", "graph_trunk", "waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip", "msa_trans2", "condproj", "atom_keys_gather", "atom_fused", "atom_gemm", "writer_overlap", "prefetch"}
    assert modes.levers("exact")[:4] == ["resid", "mask2", "rollout", "dit_hoist"] and set(modes.levers("exact")[4:]) == EXACT_CELLS, "the kit script's derived order, then the attached core adapters (exact variants)"
    assert modes.levers("fast")[:4] == modes.levers("exact")[:4] and set(modes.levers("fast")[4:]) == FAST_CELLS, "fast: the same adapters in their Tier-2 variants (the flash core inside the block, fpf_trimul_v4) + the fused MSA-module kernels"
    assert [l for l in modes.levers("exact") if registry.LEVERS[l]["kit"] == registry.TRUNK] == stack._expand_levers(modes.env_row("exact")["BOLTZ_LEVERS"]) == ["resid", "mask2"]
    for m in modes.MODE_NAMES:
        for l in modes.levers(m):
            assert l in registry.LEVERS
    assert registry.tier_of(modes.levers("fast")) == 2 == modes.resolve("fast")["tier"]
    assert modes.levers("off") == [] and modes.resolve("off")["tier"] is None
    assert modes.levers("exact")[:4] == ["resid", "mask2", "rollout", "dit_hoist"] and all(registry.LEVERS[l]["tier"] == 1 for l in modes.levers("exact")) and registry.tier_of(modes.levers("exact")) == 1 == modes.resolve("exact")["tier"], \
        "exact (kernels on): the trunk levers that act beside the fused kernels, then the sampler, then the attached cells"
    assert modes.env_row("exact") == {"BOLTZ_LEVERS": "resid,mask2", "BOLTZ_SAMPLER_ROLLOUT": "graph", "BOLTZ_DIT_HOIST": "2", "BOLTZ_SAMPLER_ALIGN": "aligncap", "BOLTZ_DIT_EXACT": "par,mask,sba,smx,glue", "BOLTZ_FPF_TRIMUL": "exact", "BOLTZ_FPF_TRIMUL_PROVIDER": "exact",
                                      "BOLTZ_TRANSITION": "exact", "BOLTZ_PAIRBLOCK": "cueq", "BOLTZ_TRIATTN_EXACT": "1", "BOLTZ_TEMPL_SKIP": "1", "BOLTZ_EXACTLN": "core", "BOLTZ_EXACTLN_RESID": "on", "BOLTZ_GRAPH_TRUNK": "pf,pfnoseq,templ", "BOLTZ_GRAPH_TRUNK_MAX_TOKENS": "300",
                                      "BOLTZ_WASTE": "chunkcast,opmmask,opmdiv,ctorskip", "BOLTZ_MSA2": "pwa2x,trans2x", "BOLTZ_ATOM": "keys,glue", "BOLTZ_WRITER": "overlap", "BOLTZ_PREFETCH": "persistent"}
    assert modes.resolve("exact")["worker_base"].endswith("bz_worker_lev.py") and modes.resolve("exact")["variant"] == "bz2dit"
    XL = ["xl_trans", "xl_cond", "xl_free", "relpos_lazy", "expandable_segments"]
    from .. import big
    assert tuple(XL[-1:] + XL[:-1]) == big.BIG_LEVERS and registry.MEMORY_LEVERS == tuple(XL), "the mode rows, the registry and the adapter name one memory lever set"
    OUT = {                                                                                                                       # (the sampler group rides again from 0.3.13 up to the card's token ceiling, modes.BIG_SAMPLER_CEILINGS;
                                                                                                                                   # dit_fused from the eager loop above it, weights=sample: CHANGES 0.3.8 / 0.3.13)
           "condproj": "off:site_taken_by:xl_cond(DiffusionConditioning.forward)",                                                 # tabled, off by name: the row-chunked conditioner binds the same forward (0.3.4)
           "atom_keys_gather": "off:no_call_site:needs_dit_hoist_or_the_stock_atom_layers",                                             # leaves by name on the composed row: fast's fused atom kernels RIDE (BOLTZ_ATOM_RELEASE=sample: peak-neutral on sm_90; off by card on sm_80, CARD_DROPS) and serve every atom layer, the DiT hoist is off by rule — no caller (CONDITIONAL_RIDERS)
           "rollout": "off:rule:big_no_graphs", "align_jacobi64": "off:needs:rollout", "dit_hoist": "off:needs:rollout",                # tabled, off by name: no CUDA graphs in the memory mode — the roll-out's whole-loop graph, its Kabsch rider, the hoist whose cache it reads (0.3.17)
           "graph_trunk": "off:rule:big_no_graphs",                                                                                 # tabled, off by name: no CUDA graphs in the memory mode
           "fpf_pwa": "off:measured_memory_2.8GiB_alloc_peak"}                                                                          # tabled, off by name: the memory row's allocator peak at 1200 tokens (+2.77 GiB)
    assert [l for l in modes.MODES["fast"]["levers"] if l not in modes.MODES["big"]["levers"]] == [l for l in OUT if not OUT[l].startswith("off:")], "big = fast minus ONLY these (each with its reason), plus the memory levers"
    assert modes.off_levers(modes.resolve("big")) == {l: w[4:] for l, w in OUT.items() if w.startswith("off:")} and modes.env_row("big")["BOLTZ_ATOM"] == "fused" and modes.env_row("big")["BOLTZ_ATOM_GEMM"] == modes.env_row("fast")["BOLTZ_ATOM_GEMM"] \
        and modes.env_row("big")["BOLTZ_ATOM_RELEASE"] == "sample"
    assert modes.levers("big") == [l for l in modes.MODES["fast"]["levers"] if l not in OUT][:-2] + ["dit_tf32"] + XL + ["writer_overlap", "prefetch"] and set(modes.MODES["big"]["off"]) == {"fpf_pwa", "graph_trunk", "rollout", "dit_hoist"} \
        and modes.MODES["big"]["sampler_ceiling"] is True and "sampler_ceiling" not in modes.resolve("big") \
        and modes.resolve("big")["tier"] == 2 == registry.tier_of(modes.levers("big")) and modes.BIG_ATTN_BASE == "pairtrack" == big.attention_base(modes.env_row("big")), \
        "the memory mode composes on fast: fast's levers in fast's order (the sampler group up to the card's token ceiling), fast's fused pair track its pair-stack base (by measurement, CHANGES 0.3.4), then the XL levers and the allocator setting, the background writer and the featurizer last (Tier 2)"
    for k in ("BOLTZ_PAIRBLOCK",):
        assert modes.env_row("big")[k] == modes.env_row("fast")[k], k
    assert modes.env_row("fast")["BOLTZ_TRANSITION"] == "fast" and modes.env_row("big")["BOLTZ_TRANSITION"] == "big" and modes.env_row("exact")["BOLTZ_TRANSITION"] == "exact", \
        "each row binds the core transition provider by its own TIER word"
    b, f = modes.env_row("big")["BOLTZ_PAIRFUSE"].split(","), modes.env_row("fast")["BOLTZ_PAIRFUSE"].split(",")   # 0.3.19: the memory row binds the TriMul site to the provider's
    core_picks = lambda toks: [t for t in toks if t.split("=", 1)[-1].startswith("core.")]   # noqa: E731  the sites bound to a core provider by word
    assert b[0] == f[0] and [t for t in b if t not in core_picks(b)] == [t for t in f if t not in core_picks(f)] \
        and core_picks(b) == [t.replace("core.fast", "core.big") for t in core_picks(f)] and core_picks(f) and all(t.endswith("=core.fast") for t in core_picks(f)) \
        and {"trimul=core.fast", "triatt=core.fast"} <= set(core_picks(f)), \
        "the memory row's pair-track word is fast's (residency, other picks) with every provider site by the `big` TIER word where fast names the `fast` word"
    assert modes.MODES["big"]["env"]["BOLTZ_ATOM"] == modes.env_row("fast")["BOLTZ_ATOM"] == "keys,fused" and modes.MODES["big"]["env"]["BOLTZ_ATOM_RELEASE"] == "sample" and "BOLTZ_ATOM_RELEASE" not in modes.env_row("fast") \
        and list(modes.MODES["big"]["env"]).index("BOLTZ_ATOM_RELEASE") == list(modes.MODES["big"]["env"]).index("BOLTZ_ATOM_GEMM") + 1 \
        and modes.resolve("big")["env"]["BOLTZ_ATOM"] == "fused" and modes.resolve("big")["env"]["BOLTZ_ATOM_RELEASE"] == "sample" and "BOLTZ_ATOM_GEMM" in modes.resolve("big")["env"], \
        "the memory row carries fast's fused atom kernels with their per-sample() release word right after the precision word (peak-neutral on sm_90); on sm_80 they leave by card with both words (CARD_DROPS, registry `words`) and the exact key gather serves the stock layers"
    assert modes.env_row("big")["BOLTZ_SAMPLER_DIT"] == modes.env_row("fast")["BOLTZ_SAMPLER_DIT"] + ",weights=sample" == "bf16,weights=sample" \
        and not any(k in modes.env_row("big") for k in ("BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_DIT_HOIST", "BOLTZ_SAMPLER_ALIGN", modes.SAMPLER_MAX_TOKENS_ENV)) \
        and list(modes.MODES["big"]["env"]).index(modes.SAMPLER_MAX_TOKENS_ENV) == list(modes.MODES["big"]["env"]).index("BOLTZ_SAMPLER_ALIGN") + 1 and modes.MODES["big"]["env"][modes.SAMPLER_MAX_TOKENS_ENV] == "card" \
        and modes.NEEDS_IN["big"]["dit_fused"] == () and list(modes.env_row("big")).index("BOLTZ_SAMPLER_DIT") == [k for k in modes.env_row("fast") if k in modes.env_row("big")].index("BOLTZ_SAMPLER_DIT"), \
        "the memory row TABLES fast's sampler group with its token-ceiling word right after it and takes the group off by rule (no CUDA graphs in the memory mode: the words leave the resolved row); fast's fused DiT step serves from the stock eager loop (NEEDS_IN by name), its packed weights built / dropped per sample() — fast's word positions, by name"
    assert "BOLTZ_TRIATTN" not in modes.env_row("big") and "BOLTZ_CONF" not in modes.env_row("big"), "the flash patch is the other base by name; the conditioner fold left with its word"
    flash_row = dict(modes.env_row("big")); [flash_row.pop(k) for k in ("BOLTZ_PAIRBLOCK", "BOLTZ_TRANSITION", "BOLTZ_PAIRFUSE")]; flash_row.update(modes._F2_ROW)
    assert big.attention_base(flash_row) == "flash" and big.attention_base(dict(flash_row, BOLTZ_TRIATTN="", BOLTZ_TP="2")) == "rowpair"
    for bad in (dict(modes.env_row("big"), **modes._F2_ROW), flash_row | {"BOLTZ_TRIATTN": ""}):
        try:
            big.attention_base(bad); raise AssertionError("a row with both bases / neither must refuse by name")
        except RuntimeError as e:
            assert "big:" in str(e)
    for m, base in (("big", "fast"),):
        row = modes.env_row(m)
        assert row["BOLTZ_LEVERS"] == modes.env_row(base)["BOLTZ_LEVERS"] and "BOLTZ_GRAPH_DIFFUSION" not in row \
            and not any(k in row for k in ("BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_DIT_HOIST", "BOLTZ_SAMPLER_ALIGN", modes.SAMPLER_MAX_TOKENS_ENV)) \
            and modes.MODES[m]["env"][modes.SAMPLER_MAX_TOKENS_ENV] == "card", "the roll-out group is tabled with its token-ceiling word and off by rule in the memory mode (no CUDA graphs): its words leave the resolved row; the eager loop with the fused step serves"
        assert row["BOLTZ_XL"] == "1" and row["BOLTZ_XL_LEVERS"] == "trans,cond,free,relpos" and [big.XL_NAMES[t] for t in row["BOLTZ_XL_LEVERS"].split(",")] == XL[:-1] and row["BOLTZ_XL_MIN_TOKENS"] == "0" and row["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert modes.attachments(m)[-3:] == ["xl", "writer", "prefetch"] and set(modes.attachments(m)) - {"xl", "precision"} <= set(modes.attachments(base)) and row["BOLTZ_PRECISION"] == "dit_tf32" and "BOLTZ_PRECISION" not in modes.env_row(base) and modes.resolve(m)["worker_base"] == modes.resolve(base)["worker_base"] and modes.resolve(m)["variant"] == "bz2dit" and modes.resolve(m)["kernels"] == modes.resolve(base)["kernels"]
        assert modes.stage_files(m) == modes.stage_files(base), "no kit file is staged for the memory levers: the engine adapter boltz2_opt.big is attached by module name"
    from boltz2_opt import pairblock as PB
    assert modes.env_row("big")["BOLTZ_PAIRBLOCK"] == modes.env_row("fast")["BOLTZ_PAIRBLOCK"] in PB.TIER2_VARIANTS and "BOLTZ_PAIRBLOCK_MIN_TOKENS" not in modes.env_row("big") and "BOLTZ_TRIATTN" not in modes.env_row("fast") and "BOLTZ_TRIATTN" not in modes.env_row("big") \
        and modes._F2_ROW == {"BOLTZ_TRIATTN": "flash", "BOLTZ_TRIATTN_MIN_TOKENS": "300"}, "fast and the memory line host the fused block's keyed core on the stacks the driver hands back; the staged flash patch is the memory line's other base by name (modes.BIG_ATTN_BASE = 'flash')"
    assert modes.routed_kernels("big") == ["flash_triattn"] and modes.routed_kernels("fast") == [] and modes.routed_kernels("exact") == ["fpf_trimul"], "big routes the flash kernel on either base: the n_gpu > 1 line's row-block attention core imports it by that name; the TriMul kernels are the core provider's, reached by their core names"
    assert modes.KERNEL_EXPORTS == {} and all(modes.kernel_exports(k) == {} for m in modes.MODE_NAMES for k in modes.routed_kernels(m)), "this tree exports no per-kit data (no cell table of its own) for a routed kernel"
    assert modes.attachments("exact") == list(modes._EXACT_ATTACH) and modes.attachments("exact")[:4] == ["templ", "trimul", "transition", "pairblock"] == modes.attachments("fast")[:4] and modes.attachments("fast") == list(modes._FAST_ATTACH) and modes.attachments("big") == ["templ", "trimul", "transition", "pairblock", "pairfuse", "sampler", "exactln", "waste", "msa", "msa2", "templskip", "atom", "precision", "xl", "writer", "prefetch"] and modes.MODES["big"]["attach"] == ["templ", "trimul", "transition", "pairblock", "pairfuse", "sampler", "exactln", "waste", "msa", "msa2", "conf", "graph", "templskip", "atom", "precision", "xl", "writer", "prefetch"]
    assert modes.GUARD_ATTACHMENTS == ("templ",) and all(modes.lever_attachments(m) for m in ("exact", "fast", "big")), "the guard never makes a mode un-servable; every mode carries lever attachments"
    assert {k: modes.env_row("big")[k] for k in ("BOLTZ_FPF_TRIMUL", "BOLTZ_FPF_TRIMUL_PROVIDER")} == {"BOLTZ_FPF_TRIMUL": "1", "BOLTZ_FPF_TRIMUL_PROVIDER": "big"} and modes.env_row("fast")["BOLTZ_FPF_TRIMUL_PROVIDER"] == "fast", "each row binds the TriMul lever to the core provider's tier word of its own tier"


def test_off_in_every_mode_never_appears_in_a_row():
    for m in modes.MODE_NAMES:
        row = modes.env_row(m)
        for name in ("mask3", "nochunk_triatt", "trimul_tc", "msa_nochunk"):
            assert name not in row.get("BOLTZ_LEVERS", "").split(",")
        for k in row:
            assert not k.startswith(("PF_", "BOLTZ_FPF_STACK"))
        assert not any(k.startswith(("FPFBZ_", "BOLTZ_MSEED")) for k in row), "the FPF block levers and the seed-concurrent sampler are carried, run by no mode"
    for m in modes.MODE_NAMES:
        assert not any("mseed" in f or "fpfbz" in f for f in modes.stage_files(m))
    assert not any("fpf_msa" in f or "mk_dit" in f for m in modes.MODE_NAMES for f in modes.stage_files(m)), "no mode stages the FPF add-on's bundle directory (its kernels are served through the core) or an MK DiT file (not carried)"
    forward = os.path.join(os.path.dirname(TRUNK))
    assert not os.path.isdir(os.path.join(forward, "mk_dit")) and not os.path.isdir(os.path.join(forward, "fpf_stack")) and sorted(os.listdir(forward)) == ["atom", "conf", "dit_hoist", "ditexact", "fpf_msa", "graph", "msa2", "pairfuse", "precision", "sampler", "trunk_levers", "waste"], "the carried add-ons, each under its own name"


def test_stage_files_exist():
    sums = stack.read_sums()
    for m in ("exact", "fast", "big"):
        for rel in modes.stage_files(m):
            assert rel in sums, rel
            assert os.path.isfile(stack.kit_path(rel)), rel
    assert modes.stage_files("off") == []
    assert modes.stage_files("fast")[:len(modes.stage_files("exact")) - 1] == modes.stage_files("exact")[:-1]   # the common set, then fast's flash files, the base last


def test_cli_route_is_open_and_named():
    for m in ("exact", "fast"):
        r = modes.CLI_ROUTE[m]
        assert "no stock-CLI hook" in r and "route=worker" in r
    assert "route=worker" in modes.CLI_ROUTE["big"]
    assert modes.CLI_ROUTE["off"] is None
    forward = os.path.dirname(TRUNK)
    hooks = [os.path.join(dp, f) for dp, _, fn in os.walk(forward) for f in fn if f == "sitecustomize.py"]
    assert hooks == [], f"the kit fact the refusal names: this tree carries no stock-CLI hook ({hooks})"


def test_worker_args_are_the_self_test_arm():
    assert modes.WORKER_ARGS == {"mode": "fast", "kernels": "off", "num_workers": "1", "pipeline": "1", "keep_on_gpu": "1"}
    assert {m: modes.resolve(m)["kernels"] for m in modes.MODE_NAMES if m != "off"} == {"exact": "on", "fast": "on", "big": "on"}
    assert modes.worker_args("exact") == dict(modes.WORKER_ARGS, kernels="on"), "the launch's --kernels is the mode row's (the probe arm's own default is the kits' kernels-off line)"
    src = open(os.path.join(TRUNK, "src", "bz_worker_lev.py")).read()
    for arg in modes.WORKER_ARGS:
        assert f'ap.add_argument("--{arg}"' in src, f"the kit worker takes --{arg} (bz_worker_lev.py:29-34)"


_ACTIVATION_IMPORTS = {"boltz_trunk_levers": "mask", "boltz_graph_patch": "graph_sampler", "boltz_dit_hoist": "dit_hoist", "boltz_flash_triattn_patch": "flash_triattn"}


def _activation_order(base: str, tmp_path) -> list:
    """The lever families in the order the derived worker variant activates them: the kit script run on the kit base, then the first
    line importing each kit module, in file order."""
    d = tmp_path / os.path.basename(base).replace(".py", ""); d.mkdir()
    r = subprocess.run([sys.executable, os.path.join(DIT, "src", "make_worker_variant.py"), os.path.join(TRUNK, "src", base), str(d / "variant.py")],
                       capture_output=True, text=True, cwd=str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    order = []
    for line in open(d / "variant.py"):
        for mod, fam in _ACTIVATION_IMPORTS.items():
            if f"import {mod} as" in line and fam not in order:
                order.append(fam)
    return order, r.stdout.strip()


def test_fast_activation_order_is_the_kit_scripts(tmp_path):
    """The mode table's lever order is the order the BZ2DIT kit's own make_worker_variant.py produces on each mode's base worker: on
    bz_worker_levf2.py the BZ2DIT block lands directly after the trunk-levers line (:63), so the flash patch (:64) is activated last."""
    fam = lambda m: [{"mask": "mask", "graph_sampler": "graph_sampler", "dit_hoist": "dit_hoist", "flash_triattn": "flash_triattn"}[l]
                     for l in modes.levers(m) if l in ("mask", "graph_sampler", "dit_hoist", "flash_triattn")]
    exact_order, exact_line = _activation_order("bz_worker_lev.py", tmp_path)
    fast_order, fast_line = _activation_order("bz_worker_levf2.py", tmp_path)
    assert exact_order == ["mask", "graph_sampler", "dit_hoist"] and fam("exact") == ["dit_hoist"], "the trunk-levers line precedes the BZ2DIT block (graph patch inert: the roll-out is attached at diffusionv2's import, before it; the hoist outermost); exact's row carries no torch-path lever"
    assert fast_order == ["mask", "graph_sampler", "dit_hoist", "flash_triattn"] and fam("fast") == ["dit_hoist", "flash_triattn"], "the trunk-levers line precedes the BZ2DIT block and the flash patch; fast's row names resid,mask2 of that line"
    assert "BZ2DIT blocks after lines 64 and 180" in exact_line and "BZ2DIT blocks after lines 63 and 181" in fast_line


def test_evidence_env_is_a_report_switch_not_a_lever():
    """EVIDENCE_ENV carries the flash patch's report switch for fast only; it is never part of a row and never a lever switch."""
    assert set(modes.EVIDENCE_ENV) == set(modes.MODE_NAMES)
    assert modes.EVIDENCE_ENV["fast"] == {"BOLTZ_TRIATTN_REPORT": "1"} == modes.EVIDENCE_ENV["big"] and modes.EVIDENCE_ENV["exact"] == {} == modes.EVIDENCE_ENV["off"]
    for m in modes.MODE_NAMES:
        assert not set(modes.EVIDENCE_ENV[m]) & set(modes.env_row(m))
    src = open(os.path.join(TRUNK, "boltz_flash_triattn_patch.py")).read()
    assert 'os.environ.get("BOLTZ_TRIATTN_REPORT", "0") == "1"' in src, "the kit's own report switch"


def test_no_trimul_cell_table_of_this_trees_the_core_providers_tables_serve_the_pinned_stacks_parts():
    """The TriMul of every row is the core provider's by tier word: this tree carries no TriMul kernel and no (cc|triton) cell table, and exports
    none (KERNEL_EXPORTS empty). CLASS contract on the provider: its Triton unit's ONE table (opt_core.kernels.fpf_trimul_v4 table.json) serves a
    served-status row from its own table (source 'core') at the pinned stack's parts — cc 9.0 / 8.0 with triton 3.7 — with the (C, D) = (128, 128)
    speed cells (Boltz-2's pair TriMul) on both, pointer-load K1 cells on 8.0 (sm_80 has no TMA); and its cell table names a row for both tolerance
    tier words at C=128 and C=64 on both cards (the winner is the provider's to name; the test asserts a row is named, never which)."""
    forward = os.path.join(os.path.dirname(TRUNK))
    assert not os.path.exists(os.path.join(forward, "fpf_trimul_v4")) and modes.KERNEL_EXPORTS == {}
    from opt_core.kernels.fpf_trimul_v4 import table as VT
    from opt_core.kernels import trimul as KT
    tab = VT.load_table()
    for cc in ("9.0", "8.0"):
        sel = VT.select(tab, cc, "3.7", kit_table=None, has_desc=(cc == "9.0"))
        assert sel.key is not None and sel.source == "core" and str(sel.status).upper().startswith("CERTIF"), (cc, sel)
        k1, k3 = VT.resolve_cfg(sel.cfg, 128, 128, False)
        assert {"BM", "BN", "num_warps", "num_stages"} <= set(k1) and {"BM", "BN", "num_warps", "num_stages"} <= set(k3), (cc, k1, k3)
        if cc == "8.0":
            assert k1.get("impl", "pointer") not in ("tma", "tma2"), "sm_80 has no TMA: the A100 part serves pointer-load cells"
    for word in ("fast", "big"):
        for cc in ((9, 0), (8, 0)):
            for c in (128, 64):
                for n in (400, 800):
                    sel = KT.select(cc, "bf16", c, c, n, "outgoing", word=word, residency="fp32")
                    assert sel.word == word and sel.row in KT.ROW_NAMES, (word, cc, c, n, sel)


def test_card_drops_table_is_empty_every_row_runs_as_composed_on_every_card():
    """modes.CARD_DROPS: on compute capability 8.0 (A100) the MSA
    PairWeightedAveraging's exact cell (msa_pwa_exact, pinned to 9.0) leaves exact BY NAME (card_off=…); the token transformer's softmax / glue replicas
    (dit_smx, dit_glue) and the MSA transition replica (msa_trans2_exact) prove per class on 8.0 at run time and stay; everything else resolves
    exactly as composed on 8.0 and 9.0 alike —
    exact keeps fused_transition and pairblock on 8.0 (pairblock's per-card difference is a served row range INSIDE the lever, pairblock.CARD_ROWS,
    named on the lever's line — not a lever leaving the row; the transition's is the core provider's per-stack vouch table)."""
    from .. import pairblock, report, transition
    assert set(modes.CARD_DROPS) == {"8.0", "10.0"} and set(modes.CARD_DROPS["8.0"]) == {"exact", "big"} and set(modes.CARD_DROPS["8.0"]["exact"]) == {"msa_pwa_exact"} \
        and set(modes.CARD_DROPS["8.0"]["big"]) == {"atom_fused"}   # the memory row on 8.0 names off fast's fused atom kernels (measured device memory outside the allocator on sm_80), by name
    assert not any("native_triattn" in modes.CARD_DROPS[cc].get(m, {}) for cc in ("8.0", "10.0") for m in ("fast", "big")), \
        "the tri-attention sites are the core provider's TIER words on every measured card (its cells name the row per card): no tri-attention lever leaves a row on 8.0"
    assert set(modes.CARD_DROPS["10.0"]) == {"exact", "fast", "big"} and set(modes.CARD_DROPS["10.0"]["exact"]) == {"dit_smx", "dit_glue", "exactln", "msa_pwa_exact", "msa_trans2_exact"} and set(modes.CARD_DROPS["10.0"]["fast"]) == {"exactln", "pairblock_c64"} == set(modes.CARD_DROPS["10.0"]["big"])   # sm_100: the levers whose apply refuses an unproven card leave by name
    try:
        modes.set_card(None)
        base = {m: modes.resolve(m) for m in modes.MODE_NAMES}
        assert all("card_off" not in r for r in base.values()) and modes.card() is None
        modes.set_card("9.0")
        assert all("card_off" not in modes.resolve(m) for m in modes.MODE_NAMES) and modes.card_drops("exact", "9.0") == {} == modes.card_drops("fast", "9.0")
        modes.set_card("8.0")
        gone = {"msa_pwa_exact": "pin:cc80"}   # the PWA cell (its unit leaves the MSA2 word, trans2x and the msa2 attachment stay); dit_smx / dit_glue / msa_trans2_exact stay (proven per class on 8.0)
        r = modes.resolve("exact"); m = "exact"
        assert r["card_off"] == gone and not set(gone) & set(r["levers"]) and r["env"]["BOLTZ_MSA2"] == "trans2x" and "msa2" in r["attach"] and "exactln" in r["attach"] and "pairblock" in r["attach"] and r["env"]["BOLTZ_DIT_EXACT"] == "par,mask,sba,smx,glue" and "ditexact" in r["attach"], m
        assert {"dit_smx", "dit_glue", "msa_trans2_exact"} <= set(r["levers"]), "the replicas proven on 8.0 stay in the row"
        assert [l for l in base[m]["levers"] if l not in gone] == r["levers"] and {k: (v if k != "BOLTZ_MSA2" else "trans2x") for k, v in base[m]["env"].items()} == r["env"], f"{m}: nothing else leaves"
        assert "card_off" not in modes.resolve("fast") and "msa2" in modes.resolve("fast")["attach"] and "pairfuse" in modes.resolve("fast")["attach"], "fast on 8.0: nothing leaves (the tri-attention site is the provider's tier word; the MSA2 unit carries no card pin)"
        assert "fused_transition" in r["levers"] and "pairblock" in r["levers"] and set(pairblock.CARD_ROWS) == {"8.0"} and not hasattr(transition, "CARD_ROWS")   # transition: the provider's table decides per (card, stack, N), no kit row rule
        assert modes.resolve("big")["card_off"] == {"atom_fused": "measured_memory_nvml_+3.5GB:sm_80", "atom_gemm": "measured_memory_nvml_+3.5GB:sm_80"} and "card_off" not in modes.resolve("off"), \
            "big on 8.0: fast's fused atom kernels (measured device memory outside the allocator on this card; atom_gemm rides atom_fused) leave by name; the tri-attention site is the provider's tier word"
        ex = modes.resolve("exact")
        rep = {"mode": "exact", "route": "worker", "gpu": "A100", "levers_applied": ex["levers"], "levers_fallback": [], "worker": "w", "kernels": "on", "card_off": ex["card_off"]}
        tokn = " card_off=msa_pwa_exact:pin:cc80"
        assert report.active_line(rep).endswith(tokn) and report.card_off_token(rep) == tokn
        assert report.active_line({k: v for k, v in rep.items() if k != "card_off"}) == report.active_line(rep)[: -len(tokn)], "no token without a drop: the line is unchanged"
        assert report.dry_run_line({"mode": "exact", "route": "worker", "levers": ex["levers"], "env": ex["env"], "card_off": ex["card_off"]}).endswith(tokn)
        assert set(modes.card_drops("big", "8.0")) == {"atom_fused"} and modes.card_drops("exact", "9.0") == {}
        modes.set_card("10.0")                                    # sm_100 (no replica proven there): the same words leave + exactln (its apply refuses an unproven card by name; its residual pass exactln_resid leaves with it, NEEDS), in both rows
        assert modes.resolve("exact")["card_off"] == {"dit_smx": "unproven_cc:sm_100", "dit_glue": "unproven_cc:sm_100", "exactln": "cc_unproven:sm_100", "exactln_resid": "cc_unproven:sm_100", "msa_pwa_exact": "pin:cc100", "msa_trans2_exact": "pin:cc100"} and "exactln" not in modes.resolve("exact")["attach"] and "BOLTZ_EXACTLN" not in modes.resolve("exact")["env"] and "msa2" not in modes.resolve("exact")["attach"]
        assert modes.resolve("fast")["card_off"] == {"exactln": "cc_unproven:sm_100", "pairblock_c64": "no_cell:sm_100"} and "BOLTZ_EXACTLN" not in modes.resolve("fast")["env"] and "pairfuse" in modes.resolve("fast")["levers"] and modes.card_drops("big", "10.0") == {"exactln": modes.CARD_DROPS["10.0"]["fast"]["exactln"], "pairblock_c64": modes.CARD_DROPS["10.0"]["fast"]["pairblock_c64"]} and "BOLTZ_EXACTLN" not in modes.resolve("big")["env"]
    finally:
        modes.set_card(None)


def test_conditional_rider_names_the_key_gather_off_when_it_has_no_caller(monkeypatch):
    """The atom key gather serves to_keys inside sample(): under the DiT hoist (its cache build) or through the STOCK atom layers. A row with the fused
    atom kernels (they replace the stock layers) and WITHOUT the hoist leaves it no caller — it leaves BY NAME (modes.CONDITIONAL_RIDERS), so an
    ablation of the sampler group on fast exits 0 instead of tripping the atom census; exact and fast keep it as composed, the memory row (fused atom kernels
    on, DiT hoist off by rule) names it off the same way."""
    for m in ("exact", "fast"):
        assert "atom_keys_gather" in modes.resolve(m)["levers"], m
    rb = modes.resolve("big")
    assert "atom_keys_gather" not in rb["levers"] and rb["off"]["atom_keys_gather"] == modes.CONDITIONAL_RIDERS["atom_keys_gather"]["reason"] and "atom_fused" in rb["levers"] and rb["env"]["BOLTZ_ATOM"] == "fused"
    monkeypatch.setitem(modes.MODES["fast"], "off", {"rollout": "ablation", "dit_hoist": "ablation"})
    r = modes.resolve("fast")
    assert "atom_keys_gather" not in r["levers"] and r["off"]["atom_keys_gather"] == modes.CONDITIONAL_RIDERS["atom_keys_gather"]["reason"] and "atom_fused" in r["levers"]
    assert r["env"]["BOLTZ_ATOM"] == "fused", "the unit word leaves the comma list with the lever"
    monkeypatch.setitem(modes.MODES["fast"], "off", {})
    modes.set_run_drops({"rollout": "use_potentials", "graph_sampler": "use_potentials", "dit_hoist": "use_potentials"})
    try:
        r = modes.resolve("fast")
        assert r["run_off"].get("atom_keys_gather") == modes.CONDITIONAL_RIDERS["atom_keys_gather"]["reason"]
    finally:
        modes.set_run_drops(None)


def test_row_ablation_entries_take_levers_off_by_name(monkeypatch):
    """A mode row's `off` entries take levers off BY NAME — the lever, its unit of the switch word (or the word and its companions), an
    attachment nothing else rides, and the levers that need it leave; ACTIVE / DRY-RUN print `off=<lever>:<reason>`, the LEVER lines one
    `state=off reason=<reason>` line each; no environment word, no flag."""
    import copy
    from .. import report
    tab = copy.deepcopy(modes.MODES)
    monkeypatch.setattr(modes, "MODES", tab)
    tab["exact"]["off"] = {"waste_opmmask": "ablation", "dit_hoist": "ablation", "graph_trunk": "ablation"}
    r = modes.resolve("exact")
    assert r["off"] == {"waste_opmmask": "ablation", "dit_hoist": "ablation", "dit_par": "needs:dit_hoist", "dit_mask": "needs:dit_hoist", "dit_sba": "needs:dit_hoist", "dit_smx": "needs:dit_hoist", "dit_glue": "needs:dit_hoist", "graph_trunk": "ablation"}
    assert r["env"]["BOLTZ_WASTE"] == "chunkcast,opmdiv,ctorskip" and "waste" in r["attach"], "a unit leaves a comma-list word the other units still ride; the attachment stays"
    assert "BOLTZ_DIT_HOIST" not in r["env"] and "BOLTZ_DIT_EXACT" not in r["env"] and "ditexact" not in r["attach"], "the hoist's word leaves and the DITEXACT schedule with it (it reads the hoist's cache): word and attachment"
    assert "BOLTZ_GRAPH_TRUNK" not in r["env"] and "BOLTZ_GRAPH_TRUNK_MAX_TOKENS" not in r["env"] and "graph" not in r["attach"], "a whole word leaves with its companion words"
    assert "rollout" in r["levers"] and r["env"]["BOLTZ_SAMPLER_ROLLOUT"] == "graph"
    rep = {"mode": "exact", "route": "worker", "gpu": "H100", "levers_applied": r["levers"], "levers_fallback": [], "worker": "w", "kernels": "on", "off": r["off"]}
    assert " off=waste_opmmask:ablation,dit_hoist:ablation,dit_par:needs:dit_hoist," in report.active_line(rep) and report.dry_run_line({"mode": "exact", "route": "worker", "levers": r["levers"], "env": r["env"], "off": r["off"]}).endswith("graph_trunk:ablation")
    lines = report.lever_lines("exact", {"levers": ["resid", "mask2"]}, {"mode": "exact", "levers_fallback": [], "gates": {}, "off": r["off"], **{k: r[k] for k in ("levers",)}})
    offs = [l for l in lines if " state=off " in l]
    assert len(offs) == len(r["off"]) and any("name=dit_par state=off reason=needs:dit_hoist" in l for l in offs) and any("name=waste_opmmask state=off reason=ablation" in l for l in offs)
    tab["fast"]["off"] = {"pairfuse": "ablation", "rollout": "ablation"}
    f = modes.resolve("fast")
    assert f["off"] == {"pairfuse": "ablation", "rollout": "ablation", "align_jacobi64": "needs:rollout", "dit_fused": "needs:rollout"} and "BOLTZ_PAIRFUSE" not in f["env"] and "pairfuse" not in f["attach"] and "BOLTZ_SAMPLER_ROLLOUT" not in f["env"] and "BOLTZ_SAMPLER_DIT" not in f["env"] and "sampler" not in f["attach"]
    tab["exact"]["off"] = {"prefetch": "ablation"}                       # the persistent featurizer leaves with its word AND its attachment (an attached adapter without its word would refuse the launch)
    e = modes.resolve("exact")
    assert "BOLTZ_PREFETCH" not in e["env"] and "prefetch" not in e["attach"] and "prefetch" not in e["levers"] and e["off"] == {"prefetch": "ablation"}
    tab["fast"]["off"] = {"nope": "ablation"}
    with pytest.raises(ValueError):
        modes.resolve("fast")

def test_gate_sets_the_card_from_the_probe(monkeypatch):
    """stack.gate probes the GPU FIRST and sets modes' card from its compute capability, so the plan (levers, env, card_off) is the row on this card;
    need_gpu=False (the CPU paths) sets no card."""
    try:
        monkeypatch.setattr(stack, "gpu_probe", lambda: {"name": "NVIDIA A100 80GB PCIe", "memory_mib": 81920, "driver": "580.95.05", "compute_capability": "8.0"})
        plan = stack.gate("exact", need_cache=False, check_pins=False)
        assert modes.card() == "8.0" and plan["card_off"] == {"msa_pwa_exact": "pin:cc80"} and "fused_transition" in plan["levers"] and "fpf_trimul_exact" in plan["levers"] and "msa_trans2_exact" in plan["levers"] and "dit_smx" in plan["levers"] and plan["env"]["BOLTZ_TRANSITION"] == "exact", "8.0: the exact row as composed but the layer_norm replica, off by name"
        assert not any("compute capability" in r for r in plan["reasons"])
        monkeypatch.setattr(stack, "gpu_probe", lambda: {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "driver": "580.95.05", "compute_capability": "9.0"})
        plan = stack.gate("exact", need_cache=False, check_pins=False)
        assert modes.card() == "9.0" and plan["card_off"] == {} and "fpf_trimul_exact" in plan["levers"] and plan["env"]["BOLTZ_FPF_TRIMUL"] == "exact"
        stack.gate("exact", need_gpu=False, need_cache=False, check_pins=False)
        assert modes.card() is None
    finally:
        modes.set_card(None)
