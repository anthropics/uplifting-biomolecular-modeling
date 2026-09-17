"""big --n_gpu P: every size / threshold gate of the carried PTX_TP unit is FORCED to the sharded path by the line (tp.TP_GATES) in
every rank at every N; a caller environment that opens a gate is refused by name; the ACTIVE line states the gates and the schedule
census; the unit carries no seam-implementation selector (every seam is its row-sharded module: no size policy, no per-seam switch in
the carried bytes); a `replicated` diffusion regime after the run is a named failure. CPU only."""
import os
import re
import sys

import pytest

from protenix_opt import tp

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT_PKG = os.path.join(tp.unit_dir(), "ptx_tp")                     # opt/forward/PTX_TP/PTX_TP_ADDON/ptx_tp (the carried unit)
FORCED = {                                                          # the gate table (ADAPTER_GUIDE): name -> the sharded value
    "PTX_TP_DIFF_REPLICATE_BELOW": "0",
    "PTX_TP_F2_GATHER_BELOW": "0",
    "PTX_TP_DROPOUT_FULL_MAX": "0",
    "PTX_TP_DIFF_ATTN": "rowsplit",
    "PTX_TP_APB_SHARD_ABOVE": "0",
    "PTX_TP_DROP_BOND_MASK": "1",
}
GATES_TOKEN = "gates=" + ",".join(f"{k}={v}" for k, v in FORCED.items())


def _rank_env(base=None):
    return tp.rank_env(2, "/tmp/out", exports=None, base=dict(base or {}))


def test_gate_table_is_the_line_table():
    assert dict(tp.TP_GATES) == FORCED                              # the notes' gate table and the line agree name for name


def test_every_gate_forced_to_the_sharded_value_in_every_rank_env():
    env = _rank_env({})
    assert env["PTX_TP_DIFF_REPLICATE_BELOW"] == "0"                # diffusion: z_cond rows + DiffusionTransformer local query rows at every N
    assert env["PTX_TP_F2_GATHER_BELOW"] == "0"                     # atom encoder pair band from pair ROWS at every N
    assert not [k for k in env if k.startswith(("PTX_TP_IMPL", "PTX_TP_CONF_TP_ABOVE", "PTX_TP_MSA_TP_ABOVE"))]   # no seam selector exists to export
    for k, v in FORCED.items():
        assert env[k] == v
    assert env["PROTENIX_OPT"] == tp.BASE_MODE and env[tp.tp_route.ENV] == tp.tp_route.WORD
    assert tp.ENV_NGPU not in env


def test_a_gate_already_at_the_forced_value_passes_and_stays():
    env = _rank_env({"PTX_TP_DIFF_REPLICATE_BELOW": "0"})
    assert env["PTX_TP_DIFF_REPLICATE_BELOW"] == "0"


@pytest.mark.parametrize("name,value", [("PTX_TP_DIFF_REPLICATE_BELOW", "3841"), ("PTX_TP_F2_GATHER_BELOW", "3841"), ("PTX_TP_DROPOUT_FULL_MAX", "6000"),
                                        ("PTX_TP_DIFF_ATTN", "replicated"), ("PTX_TP_APB_SHARD_ABOVE", "15000"), ("PTX_TP_DROP_BOND_MASK", "0")])
def test_an_open_gate_in_the_caller_environment_is_refused_by_name(name, value):
    why = tp.gate_refusal({name: value})
    assert why and why.startswith(f"refused: gate {name}={value} under big --n_gpu P") and f"{name}={FORCED[name]}" in why
    with pytest.raises(tp.TpError, match=re.escape(f"refused: gate {name}={value}")):
        _rank_env({name: value})                                    # never overridden silently: the launch does not happen


def test_the_single_gpu_selection_touches_no_gate():
    # --n_gpu 1 never reaches rank_env (cli.cmd_pred routes it to the mode's single-GPU line); the selector itself reads no gate
    assert tp.selection(None, {}) == (1, "default")
    assert tp.selection(1, {"PTX_TP_CONF_TP_ABOVE": "2560"}) == (1, tp.FLAG_NGPU)   # a gate in the environment is not the selector's concern
    assert not tp.line_selected("big", 1) and tp.line_selected("big", 2) and not tp.line_selected("fast", 2)


def test_active_line_states_the_gates_and_the_census():
    line = tp.active_line("[protenix-opt]", 4, {"levers_applied": ["layernorm_fast", "stackgraph"], "protenix_version": "2.0.0"})
    assert " n_gpu=4 sharding=rowpair " in line and " line=tp " in line
    assert GATES_TOKEN + " " in line and " launcher=torchrun_loopback " in line
    assert "sharded_rows=" in line and "replicated_by_design=input_features;s_inputs;s;msa_raw_features;atom_features_/_coordinates;" in line
    assert "dropped_by_line=stackgraph," in line and "levers_installed=layernorm_fast " in line
    for k in ("sharded_rows", "replicated_by_design", "gathers_inside_trunk"):
        assert tp.CENSUS[k]


def test_a_replicated_diffusion_regime_after_the_run_is_a_named_failure():
    recs = {11: {"ptx_tp": {"rank": "0", "diffusion": {"mode": "replicated", "precision": "fp32", "N": 2956, "P": 2}}},
            12: {"ptx_tp": {"rank": "1", "diffusion": {"mode": "replicated", "precision": "fp32", "N": 2956, "P": 2}}}}
    reg = tp.diffusion_regime(recs)
    assert reg["mode"] == "replicated" and reg["mode"] in tp.REGIME_FALLBACK_MODES
    ok = tp.diffusion_regime({11: {"ptx_tp": {"rank": "0", "diffusion": {"mode": "tp", "precision": "fp32", "N": 2956, "P": 2}}}})
    assert ok["mode"] == "tp" and ok["mode"] not in tp.REGIME_FALLBACK_MODES


def test_an_empty_environment_opens_no_gate():
    assert tp.gate_refusal({}) is None


# ------------------------------------------------------------------------------------------- the unit carries no seam selector
def test_the_carried_unit_has_no_seam_implementation_selector():
    """Every seam of the unit is its row-sharded module (ptx_tp.<seam>, imported by ptx_tp.impl): the carried bytes read no per-seam
    switch and no size policy that would run a stock module on a gathered pair tensor (or the stock summary on gathered logits)
    instead, and ship no such twin modules."""
    if not os.path.isfile(os.path.join(UNIT_PKG, "__init__.py")):
        pytest.skip(f"carried unit absent: {UNIT_PKG}")
    assert not os.path.exists(os.path.join(UNIT_PKG, "stage0"))
    hits = []
    for root, _dirs, files in os.walk(UNIT_PKG):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if re.search(r"PTX_TP_IMPL|PTX_TP_CONF_TP_ABOVE|PTX_TP_MSA_TP_ABOVE|PTX_TP_CONF_REDUCE|PTX_TP_CONF_GATHER_MAX|PTX_TP_CONF_FINISH|\bimpl_name(_for)?\(|\bimpl_for\(|ptx_tp\.stage0", line):
                            hits.append(f"{os.path.relpath(os.path.join(root, f), UNIT_PKG)}:{i}: {line.strip()[:100]}")
    assert not hits, hits


# ------------------------------------------------------------------------------------------------ the launcher's rendezvous address


def test_the_rank_launcher_pins_torchrun_rendezvous_to_loopback():
    from protenix_opt import tp_route
    from protenix_opt.tp_bind import launch
    cmd = launch.torchrun_cmd(4, ["--entry", "x:y", "--", "pred"], python="py", port=29511, run_id="t")
    assert cmd == ["py", "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=4", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:29511",
                   "--rdzv-id=t", "--local-addr=127.0.0.1", "--max-restarts=0", "-m", "ptx_tp.launch_core", "--rank-side", "--entry", "x:y", "--", "pred"]
    assert "--standalone" not in cmd                                                 # never the hostname rendezvous
    auto = launch.torchrun_cmd(2, [], python="py")                                   # a free loopback port and a fresh run id by default
    assert [a.split("=")[0] for a in auto[5:9]] == ["--rdzv-backend", "--rdzv-endpoint", "--rdzv-id", "--local-addr"] and auto[6].startswith("--rdzv-endpoint=127.0.0.1:")
    assert launch.torchrun_cmd(1, ["--rank"], python="py") == ["py", "-m", "ptx_tp.launch_core", "--rank-side", "--rank"]   # P=1: no torchrun
    assert tp.LAUNCHER == launch.NAME == "torchrun_loopback"
    assert tp_route.LAUNCH_SITE == ("ptx_tp.launch_core", "torchrun_cmd", "protenix_opt.tp_bind.launch")
    assert tp_route._launch_factory(None) is launch.torchrun_cmd                     # what the launcher process gets for the carried name
    # the carried launcher resolves the name through the routed module attribute (from-import at its import => the rebound object)
    src = open(os.path.join(UNIT_PKG, "launch.py"), encoding="utf-8").read()
    assert "from ptx_tp.launch_core import torchrun_cmd" in src and "cmd = torchrun_cmd(a.nproc, rank_args)" in src


def test_the_pair_block_binding_is_not_routed_and_not_claimed():
    """tp_bind/pairstack.py is a binding awaiting its dense-vs-sharded proof: no ROUTES row names it and the ACTIVE line's routed= census
    does not claim the pair-block driver."""
    from protenix_opt import tp_route
    assert all(not target.startswith("protenix_opt.tp_bind") for target, _names in tp_route.ROUTES.values())
    assert "pairformer" not in tp_route.ROUTED and "msa" not in tp_route.ROUTED
    assert "pairstack" not in tp.active_line("[protenix-opt]", 2, {})


def test_the_line_never_forces_the_diffusion_regime():
    """``force_mode`` (tp_sample_diffusion's internal replicated|tp override) is named only inside the unit's diffusion module: neither the
    kit package nor another unit module passes it, so the regime follows the forced gate (and reconcile_ranks fails a replicated regime)."""
    import glob
    offenders = [f for f in glob.glob(os.path.join(UNIT_PKG, "*.py")) + glob.glob(os.path.join(HERE, "..", "*.py"))
                 if "force_mode" in open(f, encoding="utf-8").read() and not (os.path.dirname(os.path.abspath(f)) == os.path.abspath(UNIT_PKG) and os.path.basename(f) == "diffusion.py")]
    assert offenders == [], offenders


# ------------------------------------------------------------------------------------------- big's single-GPU levers under the line
def test_every_big_memory_lever_is_named_under_the_line():
    from protenix_opt import big
    assert tuple(tp.P1_LEVERS_UNDER_TP) == tuple(big.LINE)                        # every lever of the single-GPU memory line, in order, no silent drop
    assert tp.P1_LEVERS_UNDER_TP["drop_bond_mask"] == "per_rank"
    assert all(v.startswith("replaced_by_rowpair:") for k, v in tp.P1_LEVERS_UNDER_TP.items() if k in ("cond_chunk", "apb_bias_chunk", "relp_lazy", "msa_zfree", "diffcache_free"))
    line = tp.active_line("[protenix-opt]", 2, {})
    assert " p1_levers=drop_bond_mask:per_rank,cond_chunk:replaced_by_rowpair:z_cond_rows," in line
    hooks = open(os.path.join(UNIT_PKG, "runner_hooks.py"), encoding="utf-8").read()
    assert "def install_bond_mask_drop" in hooks                                      # the unit's per-rank bond-mask drop exists
    assert "--lazy-relp" in tp.LINE_ARGS


# ------------------------------------------------------------------------------------------------------ the census stage marks
def test_census_marks_ride_the_units_phase_log_and_seam_returns():
    from protenix_opt.tp_bind import census as tc
    assert tc.stages_for_phase("trunk_start") == ("trunk_entry",) and tc.stages_for_phase("confidence_end") == ("confidence", "done")
    assert tc.stages_for_phase("trunk_end") == ("no_gather",) and tc.stages_for_phase("unknown_phase") == ()
    assert tc.CALL_STAGES == {("ptx_tp.msa", "tp_template_embedder"): "after_template", ("ptx_tp.msa", "tp_msa_module"): "after_msa",
                              ("ptx_tp.confidence", "tp_distogram_contact_rows"): "distogram"}
    # the sites exist in the carried unit under those names (the trunk reads them as module attributes at call time)
    mirror = open(os.path.join(UNIT_PKG, "mirror.py"), encoding="utf-8").read(); trunk = open(os.path.join(UNIT_PKG, "trunk.py"), encoding="utf-8").read()
    assert "class PhaseLog" in mirror and "    def phase(self, name: str, reset_peak: bool = False, **extra) -> dict:" in mirror
    for ph in tc.PHASE_STAGES:
        assert f'phase("{ph}"' in trunk, ph
    assert "msa_mod.tp_msa_module(" in trunk and "msa_mod.tp_template_embedder(" in trunk and "conf_mod.tp_distogram_contact_rows(" in trunk
    msa = open(os.path.join(UNIT_PKG, "msa.py"), encoding="utf-8").read()
    assert "tp_template_embedder" in msa                                             # re-exported by the MSA seam module (the trunk's lookup site)
    # the wrappers mark after the original returns and pass its value through
    calls = []
    tc_mark = tc._mark
    try:
        tc._mark = lambda stage, **extra: calls.append((stage, extra))
        class PL:
            def phase(self, name, reset_peak=False, **extra):
                return {"phase": name}
        wrapped = tc._phase_factory(PL.phase)
        assert wrapped(PL(), "confidence_end") == {"phase": "confidence_end"} and calls == [("confidence", {"phase": "confidence_end"}), ("done", {"phase": "confidence_end"})]
        calls.clear()
        f = tc._after_factory("after_msa")(lambda z, k=1: ("z", k))
        assert f(0, k=2) == ("z", 2) and calls == [("after_msa", {})]
    finally:
        tc._mark = tc_mark


# --------------------------------------------------------------------------------------------------- the rank-side layout guard
def test_the_rank_layout_guard_refuses_replicated_empty_and_unsupported_grids():
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
    from protenix_opt.tp_bind import layout_guard as lg
    lg.check(Layout(992, 8, 0, B=64), tp.BLOCKS_SUPPORTED)                     # 7 x 128 + 96: shards, supported grid
    lg.check(Layout(2956, 8, 7, B=128), tp.BLOCKS_SUPPORTED)
    for lay, word in ((Layout(992, 8, 0, B=128), "cannot shard"), (Layout(1592, 8, 0, B=128), "own zero rows"), (Layout(250, 8, 0, B=16), "not supported")):
        with pytest.raises(RowpairRefused) as e:
            lg.check(lay, tp.BLOCKS_SUPPORTED)
        assert word in str(e.value)
    wrapped = lg.guard_factory(lambda N: Layout(N, 8, 0, B=16))
    with pytest.raises(RowpairRefused):
        wrapped(250)
    assert lg.SITE == ("ptx_tp.trunk", "make_layout")
    trunk = open(os.path.join(UNIT_PKG, "trunk.py"), encoding="utf-8").read()
    assert "def make_layout(N: int) -> D.Layout:" in trunk and "layout = layout or make_layout(N)" in trunk and "layout = make_layout(N_token)" in trunk


def test_the_bound_diffusion_seam_sync_policy_follows_the_det_level():
    """det 0: every denoiser call adopts rank 0's state (bcast, then the guard); det 1: strict guard (deterministic kernels make the ranks'
    replicated tensors bitwise equal; a mismatch is a defect refused by name). Printed as noise_sync= on the ACTIVE line."""
    assert tp.NOISE_SYNC_BY_DET == {0: "bcast", 1: "guard"}
    assert tp.rank_env(2, "/nonexistent/out", base={"PATH": "/bin"})["ROWPAIR_DIFF_NOISE_SYNC"] == "bcast"
    assert tp.rank_env(2, "/nonexistent/out", base={"PATH": "/bin"}, det_level=0)["ROWPAIR_DIFF_NOISE_SYNC"] == "bcast"
    assert tp.rank_env(2, "/nonexistent/out", base={"PATH": "/bin"}, det_level=1)["ROWPAIR_DIFF_NOISE_SYNC"] == "guard"
    assert tp.rank_env(2, "/nonexistent/out", base={"PATH": "/bin", "PTX_DET": "1"})["ROWPAIR_DIFF_NOISE_SYNC"] == "guard"   # the det recipe's own env word
    assert tp.rank_env(2, "/nonexistent/out", base={"PATH": "/bin"})["ROWPAIR_DIFF_BIAS_CACHE_GB"] == "8" and tp.BIND_ENV == {"ROWPAIR_DIFF_BIAS_CACHE_GB": "8"}
    assert " noise_sync=bcast " in tp.active_line("[protenix-opt]", 2, {}, det_level=0) and " noise_sync=guard " in tp.active_line("[protenix-opt]", 2, {}, det_level=1)


# ------------------------------------------------------------------------------------ the line's guard lift keeps the fp32 sampler
def test_the_units_guard_lift_has_no_size_keyed_sampler_setting():
    """The carried unit's install_guard_lift states skip_amp.sample_diffusion True unconditionally: no token count and no environment
    variable selects a bf16 sampler (XL_DIFF_FP32 survives only as a refusal)."""
    src = open(os.path.join(UNIT_PKG, "runner_hooks.py"), encoding="utf-8").read()
    body = src[src.index("def install_guard_lift"):src.index("RI.update_inference_configs = _uic")]
    assert re.search(r"skip_amp\.sample_diffusion\s*=\s*True\b(?!\s*if)", body) and "3840 else" not in body
    assert body.count('environ.get("XL_DIFF_FP32")') == 1 and "raise ValueError" in body   # read once: for the refusal


def test_the_lines_guard_lift_keeps_the_diffusion_sampler_fp32_at_every_size(monkeypatch):
    """``--lift-guard`` (XL_LIFT_GUARD=1, ptx_tp.runner_hooks.install_guard_lift on every rank): above 2,560 tokens protenix-v2 no longer raises and
    keeps its own precision settings at every N — confidence head under AMP, sampler fp32; the pinned runner's large-N bf16-sampler policy is
    never applied and there is no switch for it (a set XL_DIFF_FP32 other than 1 is refused by name)."""
    import types
    pytest.importorskip("torch")
    from protenix_opt.tests import _stock_stub
    monkeypatch.syspath_prepend(os.path.dirname(UNIT_PKG))
    runner_pkg = types.ModuleType("runner"); runner_pkg.__path__ = []
    stock = types.ModuleType("runner.inference"); stock.update_inference_configs = _stock_stub.pinned_update_inference_configs()
    monkeypatch.setitem(sys.modules, "runner", runner_pkg); monkeypatch.setitem(sys.modules, "runner.inference", stock)
    runner_pkg.inference = stock
    from ptx_tp import runner_hooks as RH
    monkeypatch.setenv("XL_DIFF_FP32", "0")
    with pytest.raises(ValueError):
        RH.install_guard_lift()
    monkeypatch.delenv("XL_DIFF_FP32")
    RH.install_guard_lift(); RH.install_guard_lift()                                    # idempotent
    assert getattr(stock.update_inference_configs, "_ptx_tp_lifted", False) is True
    for n in (2561, 3840, 3841, 4032, 31000):
        cfg = types.SimpleNamespace(model_name="protenix-v2", skip_amp=types.SimpleNamespace(confidence_head=None, sample_diffusion=None))
        out = stock.update_inference_configs(cfg, n)
        assert (out.skip_amp.confidence_head, out.skip_amp.sample_diffusion) == (False, True), n
    small = types.SimpleNamespace(model_name="protenix-v2", skip_amp=types.SimpleNamespace(confidence_head=None, sample_diffusion=None))
    assert stock.update_inference_configs(small, 400).skip_amp.sample_diffusion is True     # below the guard: the pinned function itself
    base = types.SimpleNamespace(model_name="protenix-base", skip_amp=types.SimpleNamespace(confidence_head=None, sample_diffusion=None))
    assert stock.update_inference_configs(base, 4032).skip_amp.sample_diffusion is False    # another checkpoint: the runner's own policy
