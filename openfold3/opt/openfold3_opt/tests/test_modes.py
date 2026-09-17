"""The mode table (modes.py): its shape, the resolver's exports/conflicts/refusals, tier and lever-composition invariants across every
line, and the line selection by GPU count."""
import os
import re

import pytest

from openfold3_opt import modes, registry
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_mode_table_shape():
    assert modes.MODES == ("off", "exact", "fast", "big")
    assert modes.BIG_LINES == ("resident", "tp") and modes.DEFAULT_BIG_LINE == "resident"
    assert modes.LINE_PARAMS[("big", "tp")].values == ("2", "4", "8") and modes.LINE_PARAMS[("big", "tp")].env == "OPENFOLD3_OPT_N_GPU" and modes.TP_LINES == ("tp",)
    assert modes.DEFAULT_MODE == "fast"                       # the house default: fast wherever a fast mode ships
    assert modes.EXACT_LINES == ("cueq",) and modes.DEFAULT_EXACT_LINE == "cueq"
    assert ("exact", "cueq") not in modes.LINE_TIER and modes.tier("exact", "cueq") == "exact" and modes.tier("fast") == "tolerance"
    assert modes.graphed("exact", "cueq") and modes.graphed("fast")                     # both single-GPU kit lines capture the diffusion sampler (exact: up to its own 512-token cap)
    assert set(modes.LINES) == {("exact", "cueq"), ("fast", None), ("off", None), ("big", "resident"), ("big", "tp")}
    assert set(modes.KITS) == set(modes.HOOK_DIR) == {"fast_inference", "trunk_kernels", "offload"}                            # every carried kit and port is a hook
    assert set(modes.PORTS) == {"offload"}
    assert modes.LINES[("off", None)].env == {} and modes.LINES[("off", None)].hooks == ()


def test_no_tolerance_lever_in_exact():
    for key, ln in modes.LINES.items():
        tiers = {registry.LEVERS[n].tier for n in ln.levers}
        if key[0] == "exact":
            assert tiers <= {registry.EXACT}, key
            assert not any(k in ln.env for k in ("OF3T_TRIMUL", "OF3T_TRIATT", "OF3T_FLASH_IP")), key
        if key[0] == "fast":
            assert registry.TOLERANCE in tiers
            assert not any(k.startswith("OF3FPF_") for k in ln.env)
        if key[0] == "big":
            assert modes.tier(*key) in ("exact", "tolerance", "pending"), key                   # per line, from the equality runs (modes.LINE_TIER)
            if modes.tier(*key) == "exact":
                assert tiers <= {registry.EXACT}, key                                        # a line worded exact carries exact-class levers only
            assert "OF3_CUDA_GRAPHS" not in ln.env, key                                        # no big line captures CUDA graphs (the graph pools' memory)
            if "offload" in ln.hooks:                                                       # the offload port's lines: its pin policy and template fix pinned; the confidence phase over the chunked scorer,
                assert ln.env["OF3O_PIN_POLICY"] == "census" and ln.env["OF3O_TEMPL_FIX"] == "0", key          # the TM backend named (the line as written = at/above modes.CONF_MIN_TOKENS)
                assert ln.env["OF3O_CONF_MODE"] == "chunked" and ln.env["OF3O_CONF_TM_BACKEND"] == "blockreduce" and "conf_chunked" in ln.levers, key
            if key == ("big", "resident"):                                                # the row-block confidence head: its hook first, its switch, over the chunked scorer
                assert ln.hooks == ("confhead", "offload", "cells", "trunk_kernels", "fast_inference") and ln.env["OPENFOLD3_OPT_CONFHEAD"] == "1" and "confhead" in ln.levers, key   # the port's units install beneath the cells
                fast = modes.LINES[("fast", None)]                                               # the tier rule: big = fast minus ONLY the levers with a memory cost (CHANGES.md names each) —
                assert {l for l in fast.levers if l not in ln.levers} == {"cuda_graphs", "graphs_strict", "paircache", "dit_glue", "atom_hoist", "castcache", "trunk_graph"}, key   # a new fast lever lands here or on the line: each absence is for its memory cost (CHANGES.md)
                assert all(ln.env[k] == v for k, v in fast.env.items() if k in ln.env and k not in ("OPENFOLD3_OPT_PAIR_CORE", "OPENFOLD3_OPT_TRIMUL_TIER", "OPENFOLD3_OPT_LN_TIER", modes.APB_WORD_ENVS[0])), key   # every carried switch spelled as fast spells it, except the providers' TIER WORDS (pair core, trimul, LayerNorm, pair-bias attention: fast there, big here)
                assert ln.env["OPENFOLD3_OPT_PAIR_CORE"] == "provider:big" and fast.env["OPENFOLD3_OPT_PAIR_CORE"] == "provider", key   # `provider` = the tier word fast; the big line names its tier
                assert ln.env["OPENFOLD3_OPT_PAIR"] == fast.env["OPENFOLD3_OPT_PAIR"] and ln.env["OPENFOLD3_OPT_TRIMUL_TIER"] == "big", key
                assert ln.env["OPENFOLD3_OPT_LN_TIER"] == "big" and fast.env["OPENFOLD3_OPT_LN_TIER"] == "fast" and "OPENFOLD3_OPT_EXACTLN" not in ln.env and "OPENFOLD3_OPT_EXACTLN" not in fast.env, key   # the LayerNorm binding rides both lines as ln_provider with the line's tier word; the exact-class word is the exact line's alone; the graphed lines serve every item
                assert {"OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3T_PAIRCACHE"} <= set(ln.unset) and ln.kit_levers is None, key   # the graph family and the pair cache required unset; every hook chains the next (CHAIN_ENV)
            if key == ("big", "tp"):                                                      # the tensor-parallel line: the row-sharded pair representation's profile, the offload port off
                assert not any(k in ln.env for k in ("OF3T_TRIMUL", "OF3T_TRIATT")), key           # the sharded pair stack replaces the ops the trunk-kernels swaps act on
                assert {k for k in ln.env if k.startswith("OF3TP_")} == {"OF3TP_TRIATT_QBLOCK", "OF3TP_TRIATT_ROWBLOCK", "OF3TP_APB_QBLOCK", "OF3TP_APB_ROWBLOCK", "OF3TP_TRIMUL_SUB"} and not any(k.startswith("OF3O_") for k in ln.env), key   # the row-block sizes are the line's only OF3TP_ words


def test_resolver_exports_and_conflicts():
    r = modes.resolve("fast", HOME, environ={})
    fast = modes.LINES[("fast", None)]
    assert r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference") == os.path.join(HOME, modes.KITS["fast_inference"], "of3_levers")
    assert r.entry_hook == os.path.join(modes.hook_dir(HOME, "cells"), "sitecustomize.py") == os.path.join(HOME, modes.CELLS_HOOK_DIR, "sitecustomize.py")   # the package's cells hook first;
    assert r.exports[modes.PAIRFUSED_CHAIN_ENV] == modes.hook_dir(HOME, "trunk_kernels") == os.path.join(HOME, modes.KITS["trunk_kernels"], "of3t_hook")               #  it chains to the trunk-kernels hook
    assert modes.hook_spellings(r) == ["opt/openfold3_opt/hooks/cells", "of3t_hook", "of3_levers"]
    r = modes.resolve("exact", HOME, environ={})
    assert r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference")
    assert [os.path.basename(d) for d in r.hook_dirs] == ["cells", "of3t_hook", "of3_levers"] and r.line == "cueq" and r.exports["OPENFOLD3_OPT_ATOM_HOIST"] == "1"
    r = modes.resolve("fast", HOME, environ={"OF3T_TRIATT": "stock"})
    assert r.conflicts and "OF3T_TRIATT" in r.conflicts[0]
    r = modes.resolve("exact", HOME, environ={"OF3T_TRIMUL": "cueq"})
    assert r.conflicts and "requires it unset" in r.conflicts[0]
    with pytest.raises(ValueError):
        modes.resolve("faster", HOME)
    assert modes.describe_line(modes.resolve("off", HOME, environ={})) == "off"
    for (mode, line) in modes.LINES:                                                            # no chain variable (a directory) in any mode's spelling: the chain is the hook order after '@'
        res = modes.resolve(mode, HOME, environ=({"OF3TP_RANK": "0", "OF3TP_WORLD": "2"} if line == "tp" else {}), n_gpu=("2" if line == "tp" else None))
        spelled = modes.describe_line(res)
        kv = spelled.split(" @ ")[0]
        assert "/" not in kv and not any(k + "=" in kv for k in modes.KIT_LEVERS_ENVS), (mode, line, spelled)
    fast = modes.describe_line(modes.resolve("fast", HOME, environ={}))
    assert "OPENFOLD3_OPT_PAIR_CHAIN" not in fast and fast.split(" @ ")[1].startswith("opt/openfold3_opt/hooks/cells(cells)>")   # the cells hook still leads the chain after '@'


def test_graphed_follows_the_line_switches():
    """det.GRAPHED_EXTRA (OF3_GRAPHS_STRICT=1) is re-asserted on graphed lines only: the fast line captures graphs at every size, the exact line up to
    its own 512-token cap (graphed_call decides per call), the big lines do not."""
    assert modes.graphed("fast") and modes.graphed("exact", "cueq") and "OF3_CUDA_GRAPHS" not in modes.LINES[("exact", "cueq")].unset
    assert modes.LINES[("exact", "cueq")].env["OF3_CUDA_GRAPHS"] == "1" and modes.LINES[("exact", "cueq")].graphs_max_tokens == 512
    assert modes.graphed_call("exact", "cueq", 400, {}) and not modes.graphed_call("exact", "cueq", 513, {}) and not modes.graphed_call("exact", "cueq", None, {})   # fail-closed without a token count
    assert modes.graphed_call("exact", "cueq", 2000, {modes.ENV_GRAPHS_MAX_TOKENS: "always"}) and not modes.graphed_call("exact", "cueq", 400, {modes.ENV_GRAPHS_MAX_TOKENS: "0"})
    assert not modes.graphed("off")
    for ln in modes.BIG_LINES:
        assert not modes.graphed("big", ln), ln
        assert "OF3_GRAPHS_STRICT" in modes.LINES[("big", ln)].unset, ln


def test_big_lines_declare_no_late_rebinding_of_the_sampler():
    """The offload hook fires LAST on the model module (the chain installs its finder first; a later hook's finder sits ahead of it), so its
    model_forward is the outermost OpenFold3.forward binding and a prior hook's forward wrapper is superseded (of3_offload.apply_core records
    it as `forward_superseded`, a named event). The big lines therefore run with no hook that rebinds SampleDiffusion after import: the
    graph switches are declared unset on every big line (the trunk-kernels paircache late wrap has nothing to re-wrap)."""
    for ln in modes.BIG_LINES:
        line = modes.LINES[("big", ln)]
        assert {"OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT"} <= set(line.unset), ln
        assert "OF3_CUDA_GRAPHS" not in line.env, ln


def test_the_line_is_selected_by_the_gpu_count_alone():
    """No line selector: exact runs its one composition; under big `--n_gpu` (or OPENFOLD3_OPT_N_GPU) selects resident (1) or tp (2|4|8)."""
    assert modes.line_arg("exact") == "cueq" and modes.line_arg("fast") is None and modes.line_arg("off") is None
    assert modes.line_arg("big", environ={}) == "resident" == modes.line_arg("big", environ={}, n_gpu="1")
    assert modes.line_arg("big", environ={}, n_gpu="2") == "tp" == modes.line_arg("big", environ={"OPENFOLD3_OPT_N_GPU": "8"})
    p = modes.LINE_PARAMS[("big", "tp")]
    assert (p.name, p.flag, p.env, p.values, p.default) == ("n_gpu", "--n_gpu", "OPENFOLD3_OPT_N_GPU", ("2", "4", "8"), "1")
    assert modes.MODE_LINES == {"exact": ("cueq",), "big": ("resident", "tp")}


def test_every_switch_is_read_by_a_carried_hook():
    """Every variable a line exports is read by the kit whose hook the line runs (the kit's own os.environ reads)."""
    HOME = _stubs.tree_home()
    reads = {}
    for kit in list(modes.KITS) + list(modes.PACKAGE_HOOKS):
        text = ""
        if kit in modes.PACKAGE_HOOKS:                                                          # a package hook: its hook file + the module or package it installs (hooks/confhead -> confhead.py; tp_rowpair/hook -> tp_rowpair/)
            mod = os.path.join(HOME, "opt", "openfold3_opt", kit)
            for py in ([mod + ".py"] if os.path.isfile(mod + ".py") else [os.path.join(r, f) for r, _, fs in os.walk(mod) for f in fs if f.endswith(".py")]):
                text += open(py, encoding="utf-8").read()
        for root, _, files in os.walk(os.path.join(HOME, modes.KITS[kit]) if kit in modes.KITS else modes.hook_dir(HOME, kit)):
            for f in files:
                if f.endswith(".py"):                                                       # the add-ons' python
                    with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                        text += fh.read()
        reads[kit] = set(re.findall(r'\b((?:OF3|OF3T|OF3O|OF3TP|OF3FPF|OF3FLASHPF|CUBLAS|PYTORCH_CUDA|OPENFOLD3_OPT)_[A-Z0-9_]+)\b', text))   # every switch name the kit's code spells
        assert os.path.isfile(os.path.join(modes.hook_dir(HOME, kit), modes.HOOK_FILE)), kit
    runtime_reads = {"PYTORCH_CUDA_ALLOC_CONF"}                                             # torch's own allocator switch: read by torch, part of the tp line's profile (CHANGES.md `tp` contract), not spelled in kit code
    activation_text = ""                                                                    # the package's activation cells (modes.ACTIVATION_CELLS: installed by stack.activate on every kit line, no hook): their
    for cell in modes.ACTIVATION_CELLS:                                                     #  switches are read by opt/openfold3_opt/cells/<name>.py (+ its of3_<name>.py implementation) whatever hooks the line mounts
        for py in (os.path.join(HOME, "opt", "openfold3_opt", "cells", cell + ".py"), os.path.join(HOME, "opt", "openfold3_opt", "cells", "of3_" + cell + ".py")):
            if os.path.isfile(py):
                activation_text += open(py, encoding="utf-8").read()
    activation_reads = set(re.findall(r'\b(OPENFOLD3_OPT_[A-Z0-9_]+)\b', activation_text))
    assert "OPENFOLD3_OPT_FASTJSON" in activation_reads and "stack.activate" in activation_text
    for key, ln in modes.LINES.items():
        names = set(ln.hooks) | ({ln.kit_levers} if ln.kit_levers else set())
        union = (set().union(*(reads[k] for k in names)) if names else set()) | (activation_reads if key[0] != "off" else set())
        for k in ln.env:
            assert k in union or k in runtime_reads, (key, k)
    for name, lv in registry.LEVERS.items():
        for k in lv.env_keys:
            assert k in reads[lv.kit], (name, k)                                                # a package lever's switch (confhead) is read by the package's own hook


def test_offload_unit_switches_are_read_by_the_hook():
    """Every unit switch (modes.OFFLOAD_UNIT_SWITCHES) is a gated install in of3_offload.apply_core (composable units, NEED 1)."""
    HOME = _stubs.tree_home()
    src = open(os.path.join(HOME, "opt", "forward", "offload", "of3o", "of3_offload.py"), encoding="utf-8").read()
    for k in modes.OFFLOAD_UNIT_SWITCHES:
        assert f'env_int("{k}", 1)' in src, k


def test_exact_cueq_is_the_trunk_kernels_exact_levers_on_the_stock_configuration():
    """The exact line: stock's cuEquivariance triangle kernels come from the runner yaml (cli.kernel_policy, not a switch); the process carries fast
    init, the trunk-kernels add-on's two exact levers, the exact-class cells and the graphed sampler up to the line's own cap — no kernel swap in the trunk / confidence head
    (inside the graphed rollout the diffusion module's attention runs the torch path instead of DS4Sci: fast_inference/README.md C3)."""
    from openfold3_opt import cli
    ln = modes.LINES[("exact", "cueq")]
    assert dict(ln.env) == {"OF3_FAST_INIT": "1", "OF3_CUDA_GRAPHS": "1", "OF3_GRAPHS_STRICT": "1", "OF3T_TEMPL_DISTINCT": "1", "OF3T_PAIRCACHE": "1", "OPENFOLD3_OPT_ATOM_HOIST": "1",
                            "OPENFOLD3_OPT_CASTCACHE": "1", "OPENFOLD3_OPT_EXACTLN": "1", "OPENFOLD3_OPT_APB_HOIST": "1", "OPENFOLD3_OPT_POST_RELEASE": "1", "OPENFOLD3_OPT_SYNC_HOIST": "1", "OPENFOLD3_OPT_FASTJSON": "1",
                            "OPENFOLD3_OPT_POSTFWD_MEM": "1", "OPENFOLD3_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OPT_WRITER_OVERLAP": "1", "OPENFOLD3_OPT_HOSTFEAT": "1", "OPENFOLD3_OPT_CKPT_MMAP": "1", "OPENFOLD3_OPT_TRIATT_EXACT": "1", "OPENFOLD3_OPT_TRIMUL_EXACT": "1", "OPENFOLD3_OPT_TRANSITION_EXACT": "1"}
    assert ln.levers == ("fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "sync_hoist", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap")
    assert ln.hooks == ("cells", "trunk_kernels", "fast_inference") and ln.kit_levers == "fast_inference" and ln.runner_yaml is None   # the cells hook carries the exact-class cells and chains to the trunk-kernels hook
    assert {"OF3T_TRIATT", "OF3T_TRIMUL", "OF3T_APB"} <= set(ln.unset) and "OF3_CUDA_GRAPHS" not in ln.unset and ln.stock_kernels and ln.graphs_max_tokens == 512
    assert cli.kernel_policy("exact") == {"use_cueq_triangle_kernels": True, "use_deepspeed_evo_attention": True} and cli.kernel_policy("exact", 1)["use_deepspeed_evo_attention"] is False


def test_a_preset_addon_variable_the_line_does_not_set_is_refused_by_name():
    """The modes set their own switches: a variable under the add-ons' prefixes that the resolved line neither exports nor requires unset is a
    conflict (NOT ACTIVE), typo or latent name alike; the deterministic recipe's switch, the run variables and — on the row-sharded line — the
    documented OF3TP_* sizes stay settable."""
    for env in ({"OF3T_TRIAT": "flash"}, {"OF3T_PAIRCACHE_SCOPE": "cond"}, {"OF3O_CPU_THREADS": "8"}, {"OF3_GRAPHS_MAX_GEN": "2"}):
        r = modes.resolve("fast", HOME, environ=env, n_tokens=612)
        (k, v), = env.items()
        assert any(c.startswith(f"{k}={v!r} preset: not a switch of the fast line") for c in r.conflicts), (env, r.conflicts)
        assert not r.notes, r.notes
    assert modes.resolve("fast", HOME, environ={"OF3T_TRIAT": ""}, n_tokens=612).conflicts == []            # an empty value is unset
    assert modes.resolve("exact", HOME, environ={"OF3_DETERMINISTIC": "1"}, n_tokens=612).conflicts == []    # the deterministic recipe's switch (det.py)
    from openfold3_opt.tp_rowpair import env as tpenv
    rank = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2"}
    base = modes.resolve("big", HOME, environ=rank, n_tokens=2565, n_gpu="2")
    budget = next(k for k in tpenv.ENV_MAP if k.endswith("_GB") and k not in base.exports)                  # a documented row-sharded budget the line does not itself export
    assert modes.resolve("big", HOME, environ={**rank, budget: "20"}, n_tokens=2565, n_gpu="2").conflicts == [], budget
    r = modes.resolve("big", HOME, environ={**rank, "OF3T_PAIRCACHE_SCOPE": "cond"}, n_tokens=2565, n_gpu="2")
    assert any("OF3T_PAIRCACHE_SCOPE='cond' preset" in c for c in r.conflicts), r.conflicts

