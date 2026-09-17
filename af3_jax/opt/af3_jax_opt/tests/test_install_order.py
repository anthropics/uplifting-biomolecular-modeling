"""REVIEW F76 — the in-process levers' install order is one documented table (fpf_launch.TREE_LEVERS) and no lever silently shadows another:
static checks on the table, each module's REBINDS / INSTALL_AFTER_ADDON, and the mode table (CPU only; the composed result on a GPU is the SERVED
line's per-lever counts, judged by modes.lever_evidence)."""
import importlib.util, os, pytest
from af3_jax_opt import fpf_launch, modes

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"af3_jax_opt.inprocess.{name}", os.path.join(HERE, "inprocess", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)          # the modules import jax / alphafold3 only inside install(): loading is CPU-safe
    return mod


MODS = {name: _load(name) for name, _ in fpf_launch.TREE_LEVERS}


def test_the_table_is_the_mode_tables_tree_levers_in_order():
    assert [lv for _, lv in fpf_launch.TREE_LEVERS] == [lv for lv in modes.TREE_LEVER_ENV]          # one order, stated once in each place
    assert set(modes.INPROCESS_MODULES) >= {lv for _, lv in fpf_launch.TREE_LEVERS}
    for name, lv in fpf_launch.TREE_LEVERS:
        assert modes.INPROCESS_MODULES[lv] == f"inprocess/{name}.py"
        assert MODS[name].ENV_SWITCH == next(iter(modes.TREE_LEVER_ENV[lv]))                       # the switch the mode table writes is the one the module reads


SAMPLE = "alphafold3.model.network.diffusion_head:sample"
SAMPLE_CHAIN = ["cond_share", "atom_cond_hoist", "sampler_bf16"]        # the one shared target, in install order: REPLACED by the first, WRAPPED by the rest (scope(precompute(shared body)))


def test_every_module_declares_what_it_rebinds_and_shared_targets_are_the_declared_chain():
    by_target = {}
    for name, _ in fpf_launch.TREE_LEVERS:                                 # install order
        reb = getattr(MODS[name], "REBINDS", None)
        assert isinstance(reb, tuple) and reb and all(":" in t for t in reb), name
        for target in reb:
            by_target.setdefault(target, []).append(name)
    shared = {t: mods for t, mods in by_target.items() if len(mods) > 1}
    assert shared == {SAMPLE: SAMPLE_CHAIN}, f"levers sharing a rebinding target outside the declared chain (order between them would be load-bearing and is not in the table): {shared}"
    for target, mods in shared.items():
        replacers = [m for m in mods if target in getattr(MODS[m], "REPLACES", ())]
        assert replacers == mods[:1], f"{target}: exactly one lever may REPLACE a shared target and it must install first (got replacers={replacers}, order={mods})"


def test_add_on_relative_passes():
    late = {name for name in MODS if getattr(MODS[name], "INSTALL_AFTER_ADDON", False)}
    assert late == set(fpf_launch.INSTALL_AFTER_ADDON_MODULES) == {"hoist_logits"}                    # pass 2: subclasses the add-on's HoistTransformer
    assert all(t.startswith("af3_flashpairformer.") for t in MODS["hoist_logits"].REBINDS)
    assert all(not t.startswith("af3_flashpairformer.") for n in MODS if n not in late for t in MODS[n].REBINDS)   # pass-1 levers never touch the add-on
    order = [n for n, _ in fpf_launch.TREE_LEVERS]
    assert "alphafold3.model.network.modules:GridSelfAttention" in MODS["triatt_xla"].REBINDS and "triatt_xla" not in late   # before the add-on derives its class from that name
    assert "alphafold3.model.network.modules:TriangleMultiplication" in MODS["trimul_cd"].REBINDS and "trimul_cd" not in late
    assert "alphafold3.model.network.diffusion_head:DiffusionHead" in MODS["atom_cond_hoist"].REBINDS and "atom_cond_hoist" not in late   # likewise: HoistDiffusionHead derives from DiffusionHead at the add-on's import
    assert order[0] == "cond_share"                                                                     # the replacer of `sample` first
    assert order.index("hoist_logits") == len(order) - 1


@pytest.mark.parametrize("off", [lv for _, lv in fpf_launch.TREE_LEVERS])
def test_a_single_lever_ablation_leaves_the_others_switches(off):
    fast = modes.resolve("fast", "/c")
    abl = modes.with_levers_off(fast, [off])
    env = abl["env"]
    for name, lv in fpf_launch.TREE_LEVERS:
        var = MODS[name].ENV_SWITCH
        if lv == off:
            assert var not in env
        else:
            assert env.get(var) == modes.TREE_LEVER_ENV[lv][var], (off, lv, env)   # a TIER_WORD_SWITCHES lever's switch carries the composition's tier word
