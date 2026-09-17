"""The mode table (modes.py): its shape, the resolver's exports/conflicts/refusals, and the tier and lever-composition invariants across
every line."""
import os
import re

import pytest

from openfold3_ob0_opt import modes, registry
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_mode_table_shape():
    assert modes.MODES == ("off", "exact", "fast", "big")
    assert modes.BIG_LINES == ("resident", "tp") and modes.BIG_RESIDENT == "resident"       # one GPU: resident; --n_gpu P>1: tp — the resource decides, nothing else selects
    assert modes.LINE_PARAMS[("big", "tp")].values == ("2", "4", "8") and modes.LINE_PARAMS[("big", "tp")].env == "OPENFOLD3_OB0_OPT_N_GPU"
    assert modes.DEFAULT_MODE == "fast"                       # the house default: fast wherever a fast mode ships
    EXL = modes.EXACT_LINE
    assert EXL == "cueq" and modes.line_for("exact") == EXL                                   # one exact composition, named for the stock configuration's kernels
    assert ("exact", EXL) not in modes.LINE_TIER and modes.tier("exact", EXL) == "exact" and modes.tier("fast") == "tolerance"
    assert modes.graphed("exact", EXL) and modes.graphed("fast")             # both capture the diffusion sampler in CUDA graphs
    assert set(modes.LINES) == {("exact", EXL), ("fast", None), ("off", None)} | {("big", b) for b in modes.BIG_LINES}
    assert set(modes.KITS) == set(modes.HOOK_DIR) == {"fast_inference", "trunk_kernels", "offload"}   # every carried kit and port is a hook
    assert set(modes.PORTS) == {"offload"}
    assert modes.ERRATA == {}                                                                   # one hook directory per add-on
    for rel in (modes.STOCK_YAML, modes.STOCK_DET_YAML, modes.SHIPPED_YAML, modes.KERNELS_OFF_YAML):
        assert os.path.isfile(os.path.join(HOME, rel)), rel                                  # every configuration a route names is a file of the package
    assert modes.STOCK_DET_YAML == modes.STOCK_YAML != modes.SHIPPED_YAML                     # stock = the cuEquivariance bf16 configuration, det on the same file; shipped = the `default` arm
    assert modes.LINES[("big", "tp")].runner_yaml == modes.LINES[("big", "resident")].runner_yaml == modes.BIG_BF16_C16_YAML   # both big lines run the kernels-off c16 configuration (tp.pinned_yaml lays its row-shard plan on it)
    assert os.path.isfile(os.path.join(HOME, modes.BIG_BF16_C16_YAML))
    assert modes.LINES[("fast", None)].runner_yaml == modes.STOCK_YAML and modes.CUEQ_CAPTURES is True           # both candidate bases are capture-safe; the fast line names one
    assert ("use_cueq_triangle_kernels" in modes.CAPTURE_SAFE_KERNEL_FLAGS) == bool(modes.CUEQ_CAPTURES)
    assert modes.LINES[("off", None)].env == {} and modes.LINES[("off", None)].hooks == ()


def test_no_tolerance_lever_in_exact():
    for key, ln in modes.LINES.items():
        tiers = {registry.LEVERS[n].tier for n in ln.levers}
        if key[0] == "exact":
            assert tiers <= {registry.EXACT}, key
            assert not any(k in ln.env for k in ("OF3T_TRIMUL", "OF3T_TRIATT", "OF3T_FLASH_IP")), key
        if key[0] == "fast":
            assert registry.TOLERANCE in tiers
        if key[0] == "big":
            assert modes.tier(*key) in ("exact", "tolerance", "pending"), key                   # per line, from the equality runs (modes.LINE_TIER)
            if modes.tier(*key) == "exact":
                assert tiers <= {registry.EXACT}, key                                        # a line worded exact carries exact-class levers only
            assert not any(k in ln.env for k in ("OF3_CUDA_GRAPHS", "OF3T_TRIMUL")), key                 # no big line captures CUDA graphs; OF3T_TRIMUL stays stock's cuEquivariance on every line of this kit
            if "offload" in ln.hooks:                                                       # the offload port's lines: its pin policy and template fix pinned; the confidence phase over the chunked scorer,
                assert ln.env["OF3O_PIN_POLICY"] == "census", key          # the TM backend named (the line as written = at/above modes.CONF_MIN_TOKENS)
                assert ln.env["OF3O_CONF_MODE"] == "chunked" and ln.env["OF3O_CONF_TM_BACKEND"] == "blockreduce" and "conf_chunked" in ln.levers, key
            if key == ("big", "resident"):                                                # the row-block confidence head: its hook first, its switch, over the chunked scorer
                assert ln.hooks == ("confhead", "offload", "cells", "trunk_kernels", "fast_inference") and ln.env["OPENFOLD3_OB0_OPT_CONFHEAD"] == "1" and "confhead" in ln.levers and ln.env["OF3O_LAYER"] == "1", key   # the port's units install beneath the cells
                fast = modes.LINES[("fast", None)]                                               # the tier rule: big = fast minus ONLY the levers with a measured memory cost (CHANGES.md names each with its number) —
                assert {l for l in fast.levers if l not in ln.levers} == {"cuda_graphs", "graphs_strict", "paircache", "dit_glue", "atom_hoist", "castcache", "trunk_graph"}, key   # exactln rides the line behind its 1024-token item gate; trimul_v4 rides this kit's memory line (+21-28 MiB: its TriMul kernel) —   # exactln: pending its proof against the port's LN-SAFE guard on the same primitive (CHANGES.md);   # a new fast lever lands here or on the line
                assert all(ln.env[k] == v for k, v in fast.env.items() if k in ln.env and k not in ("OPENFOLD3_OB0_OPT_TRIMUL_TIER", "OPENFOLD3_OB0_OPT_PAIR_CORE", "OPENFOLD3_OB0_OPT_LN_TIER", modes.APB_WORD_ENVS[0])), key   # every carried switch spelled as fast spells it — except the providers' TIER WORDS (trimul, pair core, LayerNorm, pair-bias attention), which are the line's own (fast there, big here)
                assert ln.env.get("OPENFOLD3_OB0_OPT_TRIMUL_TIER") == "big" and fast.env.get("OPENFOLD3_OB0_OPT_TRIMUL_TIER") == "fast", key
                assert ln.env["OPENFOLD3_OB0_OPT_PAIR_CORE"] == "provider:big" and fast.env["OPENFOLD3_OB0_OPT_PAIR_CORE"] == "provider", key   # the triangle-attention provider: `provider` = the tier word fast; the big line names its tier
                assert ln.env["OPENFOLD3_OB0_OPT_PAIR"] == fast.env["OPENFOLD3_OB0_OPT_PAIR"], key                # the fast line's pair words verbatim
                assert {"OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3T_PAIRCACHE"} <= set(ln.unset) and ln.kit_levers is None, key   # the graph family and the pair cache required unset; every hook chains the next (CHAIN_ENV)
            if key == ("big", "tp"):                                                      # the tensor-parallel line: Mode S with the add-on's profile, the offload port off
                assert "OF3T_TRIATT" not in ln.env, key                                             # the sharded pair stack replaces the ops the trunk-kernels swap acts on
                assert ln.env["OF3TP_TRIMUL_SUB"] == "128" and not any(k.startswith("OF3O_") for k in ln.env), key


def test_resolver_exports_and_conflicts():
    EXL = modes.EXACT_LINE
    r = modes.resolve("fast", HOME, environ={})
    fast = modes.LINES[("fast", None)]
    assert r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference", line=fast) == os.path.join(HOME, modes.KITS["fast_inference"], modes.HOOK_DIR["fast_inference"])
    assert r.entry_hook == os.path.join(modes.hook_dir(HOME, "cells", line=fast), "sitecustomize.py") == os.path.join(HOME, modes.CELLS_HOOK_DIR, "sitecustomize.py")   # the package's cells hook first;
    assert r.exports[modes.PAIRFUSED_CHAIN_ENV] == modes.hook_dir(HOME, "trunk_kernels", line=fast) == os.path.join(HOME, modes.KITS["trunk_kernels"], modes.HOOK_DIR["trunk_kernels"])   #  it chains to the trunk-kernels hook
    assert modes.hook_spellings(r) == ["opt/openfold3_ob0_opt/hooks/cells", "of3t_hook", "of3_levers"]            # each add-on's one hook directory, spelled by its basename
    r = modes.resolve("exact", HOME, environ={})
    assert r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference")
    assert [os.path.basename(d) for d in r.hook_dirs] == ["cells", "of3t_hook", "of3_levers"] and r.exports["OPENFOLD3_OB0_OPT_ATOM_HOIST"] == "1"
    assert r.line == EXL
    r = modes.resolve("fast", HOME, environ={"OF3T_TRIATT": "stock"})
    assert r.conflicts and "OF3T_TRIATT" in r.conflicts[0]
    r = modes.resolve("exact", HOME, environ={"OF3T_TRIMUL": "cueq"})
    assert r.conflicts and "requires it unset" in r.conflicts[0]
    with pytest.raises(ValueError):
        modes.resolve("faster", HOME)
    assert modes.describe_line(modes.resolve("off", HOME, environ={})) == "off"
    for (mode, line) in modes.LINES:                                                            # no chain variable (a directory) in any mode's spelling: the chain is the hook order after '@'
        environ = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"} if line in modes.TP_LINES else {}
        res = modes.resolve(mode, HOME, environ=environ, n_gpu=2 if line in modes.TP_LINES else None)
        spelled = modes.describe_line(res)
        kv = spelled.split(" @ ")[0]
        assert "/" not in kv and not any(k + "=" in kv for k in modes.KIT_LEVERS_ENVS), (mode, line, spelled)
    fast = modes.describe_line(modes.resolve("fast", HOME, environ={}))
    assert "OPENFOLD3_OB0_OPT_PAIR_CHAIN" not in fast and fast.split(" @ ")[1].startswith("opt/openfold3_ob0_opt/hooks/cells(cells)>")   # the cells hook still leads the chain after '@'


def test_graphed_follows_the_line_switches():
    """det.GRAPHED_EXTRA (OF3_GRAPHS_STRICT=1) is re-asserted on graphed lines only: the fast and exact lines capture graphs, the big lines do not."""
    assert modes.graphed("fast") and modes.graphed("exact", modes.EXACT_LINE) and "OF3_CUDA_GRAPHS" not in modes.LINES[("exact", modes.EXACT_LINE)].unset
    assert not modes.graphed("off")
    for ln in modes.BIG_LINES:
        assert not modes.graphed("big", ln), ln
        assert "OF3_GRAPHS_STRICT" in modes.LINES[("big", ln)].unset, ln


def test_big_compositions_declare_no_late_rebinding_of_the_sampler():
    """The offload hook fires LAST on the model module (the chain installs its finder first; a later hook's finder sits ahead of it), so its
    model_forward is the outermost OpenFold3.forward binding and a prior hook's forward wrapper is superseded (of3_offload.apply_core records
    it as `forward_superseded`, a named event). The big lines therefore run with no hook that rebinds SampleDiffusion after import: the
    graph switches are declared unset on every big line (the trunk-kernels paircache late wrap has nothing to re-wrap)."""
    for ln in modes.BIG_LINES:
        line = modes.LINES[("big", ln)]
        assert {"OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3_TRUNK_GRAPHS"} <= set(line.unset), ln
        assert not any(k in line.env for k in ("OF3_CUDA_GRAPHS", "OF3_TRUNK_GRAPHS")), ln


def test_the_tp_parameter_slot_is_spelled():
    """The row-sharded composition's one parameter (`--n_gpu` / OPENFOLD3_OB0_OPT_N_GPU) is spelled in the table; no name selects a composition."""
    p = modes.LINE_PARAMS[("big", "tp")]
    assert (p.name, p.flag, p.env, p.values, p.default) == ("n_gpu", "--n_gpu", "OPENFOLD3_OB0_OPT_N_GPU", ("2", "4", "8"), "1")
    assert "tp" in modes.BIG_LINES and modes.line_for("big", n_gpu="2") == "tp" and modes.line_for("big") == modes.BIG_RESIDENT


def test_every_switch_is_read_by_a_carried_hook():
    """Every variable a line exports is read by the kit whose hook the line runs (the kit's own os.environ reads)."""
    HOME = _stubs.tree_home()
    reads = {}
    for kit in list(modes.KITS) + list(modes.PACKAGE_HOOKS):
        text = ""
        if kit in modes.PACKAGE_HOOKS:                                                          # a package hook: its hook file + the module or package it installs (hooks/confhead -> confhead.py; tp_rowpair/hook -> tp_rowpair/)
            mod = os.path.join(HOME, "opt", "openfold3_ob0_opt", kit)
            for py in ([mod + ".py"] if os.path.isfile(mod + ".py") else [os.path.join(r, f) for r, _, fs in os.walk(mod) for f in fs if f.endswith(".py")]):
                text += open(py, encoding="utf-8").read()
        for root, _, files in os.walk(os.path.join(HOME, modes.KITS[kit]) if kit in modes.KITS else modes.hook_dir(HOME, kit)):
            for f in files:
                if f.endswith(".py"):                                                       # the add-ons' python
                    with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                        text += fh.read()
        reads[kit] = set(re.findall(r'\b((?:OF3|OF3T|OF3O|OF3TP|CUBLAS|PYTORCH_CUDA|OPENFOLD3_OB0_OPT)_[A-Z0-9_]+)\b', text))   # every switch name the kit's code spells
        assert os.path.isfile(os.path.join(modes.hook_dir(HOME, kit), modes.HOOK_FILE)), kit
    runtime_reads = {"PYTORCH_CUDA_ALLOC_CONF"}                                             # torch's own allocator switch: read by torch, required by the tp add-on's documents (tp/SEAMS.md), not spelled in kit code
    from openfold3_ob0_opt import stack
    text = ""                                                                                # the package's activation installs these modules in every active process (stack.ACTIVATION_MODULES): their words need no hook
    for name in stack.ACTIVATION_MODULES:
        text += open(os.path.join(HOME, "opt", "openfold3_ob0_opt", name + ".py"), encoding="utf-8").read()
    reads[stack.PACKAGE] = set(re.findall(r'\b((?:OF3|OF3T|OF3O|OF3TP|CUBLAS|PYTORCH_CUDA|OPENFOLD3_OB0_OPT)_[A-Z0-9_]+)\b', text))
    for key, ln in modes.LINES.items():
        kits = set(ln.hooks) | ({ln.kit_levers} if ln.kit_levers else set()) | {stack.PACKAGE}
        union = set().union(*(reads[k] for k in kits)) if kits else set()
        for k in ln.env:
            assert k in union or k in runtime_reads, (key, k)
    for name, lv in registry.LEVERS.items():
        for k in lv.env_keys:
            assert k in reads[lv.kit], (name, k)                                                # a package lever's switch (confhead) is read by the package's own hook


def test_offload_unit_switches_are_read_by_the_hook():
    """Every unit switch (modes.OFFLOAD_UNIT_SWITCHES) is a gated install in of3_offload.apply_core (composable units, NEED 1)."""
    HOME = _stubs.tree_home()
    src = open(os.path.join(HOME, modes.KITS["offload"], modes.HOOK_DIR["offload"], "of3_offload.py"), encoding="utf-8").read()
    for k in modes.OFFLOAD_UNIT_SWITCHES:
        assert f'env_int("{k}", 1)' in src, k


def test_exact_line_is_the_trunk_kernels_exact_levers_on_the_stock_configuration():
    """The exact line: the stock configuration's kernels come from the runner yaml (cli.kernel_policy names the stock member by det level, not a
    switch); the process carries fast init, the diffusion sampler's CUDA graphs (strict) and the trunk-kernels add-on's two exact levers — no kernel swap."""
    from openfold3_ob0_opt import cli
    ln = modes.LINES[("exact", modes.EXACT_LINE)]
    assert dict(ln.env) == {"OF3_FAST_INIT": "1", "OF3T_TEMPL_DISTINCT": "1", "OF3T_PAIRCACHE": "1", "OF3_CUDA_GRAPHS": "1", "OF3_GRAPHS_STRICT": "1",
                            modes.ENV_CONF_DTYPE: "fp32", modes.ENV_Z_DTYPE: "fp32", "OPENFOLD3_OB0_OPT_ATOM_HOIST": "1",   # the confidence phase in upstream's fp32 (conf_dtype: the exact tier moves no number); the atom-path hoist (bitwise)
                            "OPENFOLD3_OB0_OPT_CASTCACHE": "1", "OPENFOLD3_OB0_OPT_EXACTLN": "1", "OPENFOLD3_OB0_OPT_APB_HOIST": "1", "OPENFOLD3_OB0_OPT_POST_RELEASE": "1",   # the host-side scheduling cells (exact class, bitwise)
                            "OPENFOLD3_OB0_OPT_FASTJSON": "1",                                                   # the confidence-JSON writer's text at C level (exact class, in bytes)
                            "OPENFOLD3_OB0_OPT_POSTFWD_MEM": "1", "OPENFOLD3_OB0_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OB0_OPT_WRITER_OVERLAP": "1", "OPENFOLD3_OB0_OPT_HOSTFEAT": "1", "OPENFOLD3_OB0_OPT_CKPT_MMAP": "1",   # the runner-side cells (exact class, bitwise)
                            "OPENFOLD3_OB0_OPT_TRIATT_EXACT": "1", "OPENFOLD3_OB0_OPT_TRIMUL_EXACT": "1", "OPENFOLD3_OB0_OPT_TRANSITION_EXACT": "1"}                                                  # the library's triangle-attention call on the core's provider, word exact (exact class, bitwise)
    assert ln.levers == ("fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap")
    assert ln.hooks == ("cells", "trunk_kernels", "fast_inference") and ln.kit_levers == "fast_inference" and ln.runner_yaml is None   # the cells hook carries atom_hoist (exact class) and chains to the trunk-kernels hook
    assert {"OF3T_TRIATT", "OF3T_TRIMUL", "OF3T_APB"} <= set(ln.unset) and "OF3_CUDA_GRAPHS" not in ln.unset
    assert cli.kernel_policy("exact") == modes.STOCK_YAML and cli.kernel_policy("exact", 1) == modes.STOCK_DET_YAML and cli.kernel_policy("off") == modes.STOCK_YAML
    assert cli.kernel_policy("fast") is None and cli.kernel_policy("big") is None


def test_a_preset_addon_variable_the_line_does_not_set_is_refused_by_name():
    """The modes set their own switches: a variable under the add-ons' prefixes that the resolved line neither exports nor requires unset is a
    conflict (NOT ACTIVE), typo or latent name alike; the deterministic recipe's switch, the run variables and — on the row-sharded line — the
    documented OF3TP_* sizes stay settable. `off` runs stock in a clean environment (env.py refuses kit names there), no line to contradict."""
    for env in ({"OF3T_TRIAT": "flash"}, {"OF3T_PAIRCACHE_SCOPE": "cond"}, {"OF3O_CPU_THREADS": "8"}, {"OF3_GRAPHS_MAX_GEN": "2"}, {"BFTP_ANY": "1"}):
        r = modes.resolve("fast", HOME, environ=env, n_tokens=612)
        (k, v), = env.items()
        assert any(c.startswith(f"{k}={v!r} preset: not a switch of the fast line") for c in r.conflicts), (env, r.conflicts)
        assert not r.notes, r.notes
    r = modes.resolve("exact", HOME, environ={"OF3T_PAIRCACHE_MAX_GB": "8"}, n_tokens=612)
    assert r.conflicts == [f"OF3T_PAIRCACHE_MAX_GB='8' preset: not a switch of the exact/{modes.EXACT_LINE} line (the mode sets its own switches; unset it)"], r.conflicts
    assert modes.resolve("fast", HOME, environ={"OF3T_TRIAT": ""}, n_tokens=612).conflicts == []            # an empty value is unset
    assert modes.resolve("exact", HOME, environ={"OF3_DETERMINISTIC": "1"}, n_tokens=612).conflicts == []    # the deterministic recipe's switch (det.py)
    assert modes.resolve("off", HOME, environ={"OF3T_PAIRCACHE_MAX_GB": "8"}).conflicts == []
    from openfold3_ob0_opt.tp_rowpair import env as tpenv
    rank = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}
    base = modes.resolve("big", HOME, environ=rank, n_tokens=2565, n_gpu="2")
    assert base.line in modes.TP_LINES and base.conflicts == [], base.conflicts
    budget = next(k for k in tpenv.ENV_MAP if k.endswith("_GB") and k not in base.exports)                  # a documented row-sharded budget the line does not itself export
    assert modes.resolve("big", HOME, environ={**rank, budget: "20"}, n_tokens=2565, n_gpu="2").conflicts == [], budget
    r = modes.resolve("big", HOME, environ={**rank, "OF3T_PAIRCACHE_SCOPE": "cond"}, n_tokens=2565, n_gpu="2")
    assert any("OF3T_PAIRCACHE_SCOPE='cond' preset" in c for c in r.conflicts), r.conflicts
    r = modes.resolve("big", HOME, environ={budget: "20"}, n_tokens=2565)                               # the resident line: the row-sharded line's sizes are not its switches
    assert r.conflicts == [f"{budget}='20' preset: not a switch of the big/{modes.BIG_RESIDENT} line (the mode sets its own switches; unset it)"], r.conflicts
