"""rowpair.py — the TP adapter's CPU tests.

Pure (nothing but the package): the plan facts, the rank argv rule, ``install`` at ``n_gpu == 1`` INSTALLS NOTHING (no patch, no core rowpair
import), the output-rank words. Multi-process (torch + opt_core >= 0.4.3, else skipped BY NAME): P = 2 and P = 3 rank processes under gloo on
CPU through the core's own launcher run stand-in rf3 modules (the stock statements of RF3's pair-init, recycler, template, MSA module,
pairformer, distogram, confidence-head embedding, diffusion conditioning / transformer and atom-pair window classes at toy sizes, fp64)
once DENSE (before install) and once SHARDED (after install: the pair representation born as rows, never gathered); rank 0 asserts
sharded == dense within 1e-9 for every output, the seam census (every site's sharded call count, gathers inside the trunk = 0), the rows
census tiles 0:N, and the guards (a rank whose random draw differs is refused by name).

Run: ``python -m pytest rosettafold3_opt/tests/test_rowpair.py -q``.
"""
from __future__ import annotations

import math
import os
import re
import threading
import sys
import types

import pytest

from .. import rowpair
from . import _stubs
from . import test_templ_rows

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False
    nn = None

try:
    from .. import _core
    _core.load("mem.rowpair.heads")                               # a 0.4.3 module
    HAVE_ROWPAIR = True
except Exception:  # noqa: BLE001
    HAVE_ROWPAIR = False

needs_mp = pytest.mark.skipif(not (HAVE_TORCH and HAVE_ROWPAIR), reason="torch + opt_core.mem.rowpair >= 0.4.3 needed for the gloo rank tests")
HASHSEED_CORE = "mem.rowpair.rankdata"                            # the core module whose launcher forms start every rank under ONE PYTHONHASHSEED (the [tool.opt_core] floor carries it)
TOL = 1e-9
DENSE = {}
ENGINE = {}
NHEAVY_STUB = 2                                                   # atoms per token in the stand-in head's pLDDT logits ([.., n_token, atoms * bins])


# ============================================================================================================ pure (no torch)
def test_plan_facts():
    p1 = rowpair.plan(1)
    assert p1["sharding"] == "none" and p1["installs"] == [] and p1["conflicts"] == {}
    p2 = rowpair.plan(2, mode="big")
    assert p2["sharding"] == "rowpair" and p2["strategy"] == "F7.tensor_parallel" and p2["lever"] == "rowpair"
    sites = {s["site"] for s in p2["sites"]}
    assert sites == {"entry", "entry_rf3", "pair_init", "recycler", "distogram", "conf_head", "compile", "metric_ptm", "metric_iptm", "denoiser", "diff_cond",
                     "dit", "h2d", "noise_init", "noise_predraw", "templ_feats", "pair_windows", "featurise"}
    assert {x["site"]: x for x in p2["sites"]}["featurise"]["module"] == "foundry.inference_engines.base"   # rank-0 featurisation: the engine base's pipeline constructor (BaseInferenceEngine._construct_pipeline)
    assert {x["site"]: x for x in p2["sites"]}["h2d"]["module"] == "lightning.fabric.fabric"   # ROWPAIR_MSA_HOST's seam: the engine's H2D of the featurised example (Fabric.to_device)
    assert {x["site"]: x for x in p2["sites"]}["templ_feats"]["module"] == "rf3.data.ground_truth_template"   # the noised-template feature's birth: upstream's featurizer, rebound to the precursor form
    assert "ConfidenceHead.forward" in p2["installs"] and "Recycler.forward" in p2["installs"] and "FeatureInitializer.forward" in p2["installs"]
    assert set(p2["conflicts"]) == {"fpf_trimul", "fpf_res", "graph", "dtk"}          # the kit levers off by name under n_gpu>1
    assert any("MSA representation" in r for r in p2["replicated"])
    with pytest.raises(ValueError):
        rowpair.plan(0)


def test_rank_argv():
    argv = ["py", "-m", "rf3.cli", "fold", "inputs=/i/x.json", "out_dir=/o/run", "ckpt_path=/c.ckpt", "seed=1"]
    assert rowpair.rank_argv(argv, 0, "/s") == argv
    r1 = rowpair.rank_argv(argv, 1, "/s")
    assert r1[5] == "out_dir=/s/rank1" and r1[:5] == argv[:5] and r1[6:] == argv[6:]
    assert rowpair.rank_argv(argv, 3, "/s/x")[5] == "out_dir=/s/x/rank3"
    with pytest.raises(ValueError):
        rowpair.rank_argv(["py", "-m", "rf3.cli", "fold"], 1, "/s")
    with pytest.raises(ValueError):
        rowpair.rank_argv(argv + ["out_dir=/o/dup"], 1, "/s")


def test_launch_argv_contract():
    cmd = ["py", "-m", "rf3.cli", "fold", "inputs=/i/x.json", "out_dir=/o/run/seed-42", "ckpt_path=/c.ckpt", "seed=42"]
    runner, kw = rowpair.launch_argv(cmd, {"A": "1"}, 2)
    assert runner is rowpair.run_ranks and set(kw) == {"n_gpu", "argv_of", "env", "log_dir"}
    assert kw["n_gpu"] == 2 and kw["env"] == {"A": "1"} and kw["log_dir"] == "/o/run/seed-42.ranks" == rowpair.ranks_dir(cmd)
    assert kw["argv_of"](0) == cmd and kw["argv_of"](1)[5] == "out_dir=/o/run/seed-42.ranks/rank1"
    with pytest.raises(ValueError):
        rowpair.launch_argv(cmd, {}, 1)                              # P == 1 is fold.launch's own child, never the rank launcher
    with pytest.raises(ValueError):
        rowpair.launch_argv(["py", "-m", "rf3.cli", "fold"], {}, 2)


@pytest.mark.skipif(not HAVE_ROWPAIR, reason="opt_core.mem.rowpair >= 0.4.3 absent")
def test_run_ranks_passes_the_requested_P_to_the_core_launcher(monkeypatch):
    """n_gpu silent-drop class: the P that fold.launch hands to launch_argv reaches opt_core's run_rank_processes unchanged."""
    launch = _core.load("mem.rowpair.launch")
    seen = {}

    def fake_run(n, **kw):
        seen["n"] = n
        seen["argv1"] = kw["argv_of"](1)
        seen["isolate"] = kw.get("isolate_devices")
        seen["nccl_timeout_s"] = kw.get("nccl_timeout_s")
        return []

    monkeypatch.setattr(launch, "run_rank_processes", fake_run)
    cmd = ["py", "-m", "rf3.cli", "fold", "inputs=/i/x.json", "out_dir=/o/run/seed-42", "seed=42"]
    runner, kw = rowpair.launch_argv(cmd, {"A": "1"}, 3)
    assert runner(**kw) == 0 and seen["n"] == 3 and seen["isolate"] is True and seen["nccl_timeout_s"] == rowpair.TP_NCCL_TIMEOUT_S == 1800.0   # the collective timeout is the kit's constant, passed explicitly
    assert seen["argv1"][5] == "out_dir=/o/run/seed-42.ranks/rank1"


@pytest.mark.parametrize("preset", [None, "123"])
def test_run_ranks_fold_under_one_hash_seed(tmp_path, monkeypatch, capfd, preset):
    """feats_ranks_differ class: the P rank processes are fresh interpreters, so each would draw its own str-hash seed; through ``run_ranks`` the
    core's launcher starts every rank under ONE non-empty ``PYTHONHASHSEED`` — the base environment's value when it carries one (``123`` on both
    ranks, ``source=inherited``; an integer is kept, anything else gives the default), else ``rankdata.HASHSEED_DEFAULT`` for all (``source=default``) — so the data pipeline's set / dict iteration
    orders bytes alike on every rank, and says so ONCE on stderr under the kit's tag (``[rosettafold3-opt] RANKENV hashseed=<v> source=… ranks=2``).
    The REAL ``run_rank_processes`` runs here (CPU: no device census, no isolation), each rank's argv replaced by a probe that prints the seed it got."""
    launch = _core.load("mem.rowpair.launch")
    rankdata = _core.load(HASHSEED_CORE)
    real = launch.run_rank_processes
    probe = [sys.executable, "-c", "import os; print('seed', os.environ.get('PYTHONHASHSEED'), flush=True)"]

    def on_cpu(n, **kw):
        kw.update(argv_of=lambda r: list(probe), cpu_ok=True, isolate_devices=False, run_timeout_s=120)
        return real(n, **kw)

    monkeypatch.setattr(launch, "run_rank_processes", on_cpu)
    base = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    if preset is not None:
        base["PYTHONHASHSEED"] = preset
    cmd = ["py", "-m", "rf3.cli", "fold", "inputs=/i/x.json", f"out_dir={tmp_path}/run", "seed=42"]
    runner, kw = rowpair.launch_argv(cmd, base, 2)
    assert runner(**kw) == 0
    seeds = []
    for r in range(2):
        with open(os.path.join(kw["log_dir"], f"rank{r}.log"), encoding="utf-8") as fh:
            said = [ln.split()[1:] for ln in fh if ln.startswith("seed ")]
        assert len(said) == 1, (r, said)                              # one probe line per rank transcript
        seeds.append(said[0][0] if said[0] else "")
    assert seeds[0] == seeds[1] and seeds[0] not in ("", "None"), seeds   # one seed, set, on every rank
    assert seeds == ([preset] * 2 if preset is not None else [rankdata.HASHSEED_DEFAULT] * 2)   # an exported value is kept (inherited), else the core's default
    line = rankdata.rankenv_line(rowpair._report_prefix(), base, 2)   # the core's own words for this base: `[rosettafold3-opt] RANKENV hashseed=<v> source=<s> ranks=2`
    assert line.startswith("[rosettafold3-opt] RANKENV hashseed=%s source=%s " % (seeds[0], "inherited" if preset is not None else "default")) and line.endswith("ranks=2"), line
    assert capfd.readouterr().err.count(line) == 1                     # printed ONCE by the launch, under the kit's tag (run_ranks hands the core ROWPAIR_TAG)


def test_install_at_n_gpu_1_installs_nothing(monkeypatch):
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.delenv("ROWPAIR_RANK", raising=False)
    rowpair._RP.clear()
    rep = {}
    census = rowpair.install(rep, 1)
    assert census["installed"] is False and census["sharding"] == "none" and census["patched"] == [] and rep["rowpair"] == census
    assert census["reason"] == "n_gpu:1" and census["scheme"] == "none" and census["conflicts"] == {}
    assert rowpair.CTX["patches"] is None and rowpair.CTX["installed"] is False
    assert rowpair._RP == {}, "n_gpu == 1 must not import opt_core.mem.rowpair"
    assert rowpair.install(None) == census                        # ROWPAIR_WORLD unset -> P = 1
    st = rowpair.state()
    assert st["installed"] is False and st["patched"] == [] and st["sharding"] == "none"
    assert rowpair.lever_fields() == [("n_gpu", 1), ("sharding", "none")]


def test_output_rank_words(monkeypatch):
    monkeypatch.delenv("ROWPAIR_RANK", raising=False)
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    assert rowpair.is_output_rank() and rowpair.env_world() == 1
    monkeypatch.setenv("ROWPAIR_RANK", "0")
    monkeypatch.setenv("ROWPAIR_WORLD", "4")
    assert rowpair.is_output_rank() and rowpair.env_world() == 4
    monkeypatch.setenv("ROWPAIR_RANK", "3")
    assert not rowpair.is_output_rank()


@pytest.mark.skipif(not HAVE_ROWPAIR, reason="opt_core.mem.rowpair >= 0.4.3 absent")
def test_install_refuses_a_world_that_is_not_n_gpu(monkeypatch):
    monkeypatch.setenv("ROWPAIR_WORLD", "1")
    RP = _core.load("mem.rowpair")
    with pytest.raises(RP.RowpairRefused):
        rowpair.install(None, 2)
    assert rowpair.CTX["installed"] is False and rowpair.CTX["patches"] is None


def test_template_census_is_cumulative_over_the_process_items(monkeypatch):
    """The noised-template feature's census (``rowpair._templ_census``, the ``templ_*`` words of the ``LEVER name=rowpair`` line) counts ITEMS,
    cumulatively over the rank process: two items — the first templated (37 of 50 tokens filled; the template call of both recycles), the
    second untemplated — read ``templ_items=2 templ_real=1/2 templ_fill_max=37`` on the line after the second, not the last item's own zero."""
    try:
        EV = _core.load("mem.rowpair.evidence")
    except Exception as e:  # noqa: BLE001
        pytest.skip("opt_core.mem.rowpair.evidence absent: %s" % e)
    from .. import report
    ctx = dict(rowpair.CTX, item=0)
    ctx.pop("templ_feats", None)
    monkeypatch.setattr(rowpair, "CTX", ctx)                      # a fresh process: no item entered, no census yet
    held = EV.reset_schedule()
    try:
        seen = []
        for n_fill in (37, 0):                                    # item 1 templated, item 2 not
            rowpair.CTX["item"] += 1                              # the model entry's bump (_entry_forward): once per item
            for _cycle in range(2):                               # the template call of every recycle: the item is counted once
                rowpair._templ_census(n_fill, 50)
            seen.append((EV.schedule()["templ_items"], EV.schedule()["templ_real"], EV.schedule()["templ_fill_max"]))
        assert seen == [(1, "1/1", 37), (2, "1/2", 37)], seen
        sched = EV.schedule()
        assert [sched.get(k) for k in ("templ_form", "templ_items", "templ_real", "templ_fill_max", "templ_tokens")] == ["row_born", 2, "1/2", 37, 50], sched
        t = {"mode": "big", "graph_flags_imported": True, "n_gpu": 2, "sharding": "rowpair", "rowpair": {"installed": True, "sites": ["templ_feats"], "state": {"schedule": sched}}}
        t["levers"] = report.lever_states({"levers": ["rowpair"], "mode": "big"}, t)
        line = [l for l in report.lever_lines(t) if "name=rowpair" in l][0]
        assert " templ_form=row_born templ_items=2 templ_real=1/2 templ_fill_max=37" in line and " templ_fill=" not in line and " templ_tokens=" not in line, line
    finally:
        EV.reset_schedule()
        EV.record_schedule(**held)


# ================================================================================== stand-in rf3 modules (stock statements, toy sizes)
N_TOK, C_Z, C_S, C_M, N_SEQ, C_T, N_HEAD, R_MAX, S_MAX, C_SI, BINS, C_A, D_SMP, C_AP = 50, 16, 12, 8, 5, 8, 4, 4, 2, 10, 7, 8, 2, 4
N_CYC = 2                                                         # recycles = the raw MSA stack's leading dim (RF3: n_cycle = msa_stack.shape[0])
SHOULD_USE_CUEQUIVARIANCE = False                               # the stand-ins take stock's vanilla routes
NREL = 2 * (2 * R_MAX + 2) + 1 + (2 * S_MAX + 2)


@needs_mp
def test_zinit_park_source_rule():
    """ROWPAIR_PARK_ZINIT (rowpair._zinit_source): recycle_shard_ reads Z_init from THIS item's park when the stock control flow hands back the
    parked tensor object; a plain tensor when nothing is parked; a parked park meeting a foreign tensor is refused by name (its device storage
    is released — nothing else can serve the rows)."""
    RP = _core.load("mem.rowpair")
    z = torch.zeros(3, 6, 2)
    other = torch.zeros(3, 6, 2)
    saved = {k: rowpair.CTX.get(k) for k in ("zinit", "zinit_tensor")}
    try:
        rowpair.CTX["zinit"], rowpair.CTX["zinit_tensor"] = None, None
        assert rowpair._zinit_source(z) is z                                     # nothing parked: the tensor itself
        park = types.SimpleNamespace(park=True, release=lambda: None)
        rowpair.CTX["zinit"], rowpair.CTX["zinit_tensor"] = park, z
        assert rowpair._zinit_source(z) is park                                  # the item's parked shard: the park serves the rows
        with pytest.raises(RP.RowpairRefused, match="not this item's parked z_init shard"):
            rowpair._zinit_source(other)
        park.park = False                                                        # a resident park (lever off / CPU): a foreign tensor is simply itself
        assert rowpair._zinit_source(other) is other
        rowpair._release_zinit()
        assert rowpair.CTX["zinit"] is None and rowpair.CTX["zinit_tensor"] is None
    finally:
        rowpair.CTX.update(saved)


@needs_mp
def test_msa_host_placeholder_keeps_the_stock_reads(monkeypatch):
    """ROWPAIR_MSA_HOST (one process, no group): the H2D seam holds the raw MSA stack on the host and hands the model a zero-TOKEN placeholder
    whose n_cycle (dim 0), per-cycle selection and autocast cast read exactly as the stock tensor's; the cycle's rows the MSA module receives
    are the stock statement's values BITWISE (select on the host -> move -> cast == cast -> select); the lever off is the stock move; a stack that
    bypassed the seam is refused by name at the model entry (no fallback)."""
    RP = _core.load("mem.rowpair")
    stack = torch.randn(3, 5, 7, 4, dtype=torch.float32)
    S_inputs = torch.zeros(7, 2)
    lay = types.SimpleNamespace(N=7)
    fab = STUBS["h2d"]()
    saved = {k: rowpair.CTX.get(k) for k in ("installed", "msa_host", "cycle")}
    try:
        rowpair.CTX["installed"], rowpair.CTX["msa_host"] = True, None
        monkeypatch.setenv("ROWPAIR_MSA_HOST", "0")                              # off: the stock move, nothing held
        src = {"example_id": "q", "feats": {"msa_stack": stack, "x": torch.ones(2)}}
        out = rowpair._h2d_to_device(fab, type(fab).to_device, src)
        assert out["feats"]["msa_stack"].shape == stack.shape and rowpair.CTX["msa_host"] is None
        monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0")
        assert rowpair._h2d_to_device(fab, type(fab).to_device, torch.nn.Linear(2, 2)).__class__ is torch.nn.Linear   # a module: passthrough
        out = rowpair._h2d_to_device(fab, type(fab).to_device, src)
        assert src["feats"]["msa_stack"] is stack                                # the caller's mapping as it was
        ph = out["feats"]["msa_stack"]
        assert tuple(ph.shape) == (3, 5, 0, 4) and ph.dtype == stack.dtype and int(ph.shape[0]) == int(stack.shape[0])   # n_cycle = msa_stack.shape[0] (trainers/rf3.py) reads 3
        held = rowpair.CTX["msa_host"]
        assert held["mode"] == "rank0" and held["shape"] == (3, 5, 7, 4) and held["parked"] == ["msa_stack"] and torch.equal(held["feats"]["msa_stack"], stack)
        f = dict(out["feats"])
        f["msa_stack"] = f["msa_stack"].to(torch.bfloat16)                      # the stock autocast cast (RF3.py:394-404) on the placeholder: decides the working dtype for free
        assert f["msa_stack"].dtype == stack.to(torch.bfloat16).dtype and tuple(f["msa_stack"][1].shape) == (5, 0, 4)   # RF3.py:264's selection on the placeholder
        rowpair._msa_host_entry(f, torch.device("cpu"))                          # the entry check passes: placeholder + hold
        ref = stack.to(torch.bfloat16)                                           # stock: cast the whole stack, then select the cycle
        for i in range(3):
            rowpair.CTX["cycle"] = i
            f["msa"] = f["msa_stack"][i]
            rows = rowpair._msa_rows(f, S_inputs, lay)
            assert rows.dtype == torch.bfloat16 and tuple(rows.shape) == (5, 7, 4) and torch.equal(rows, ref[i])
        rowpair.CTX["cycle"] = 3
        with pytest.raises(RP.RowpairRefused, match="recycle index 3"):
            rowpair._msa_rows(f, S_inputs, lay)
        with pytest.raises(RP.RowpairRefused, match="did not engage"):             # the seam bypassed: a whole stack at the model entry under the lever
            rowpair._msa_host_entry({"msa_stack": stack}, torch.device("cpu"))
        rowpair.CTX["msa_host"] = None
        with pytest.raises(RP.RowpairRefused, match="did not engage"):             # ... or the placeholder without a hold
            rowpair._msa_host_entry(f, torch.device("cpu"))
    finally:
        rowpair.CTX.update(saved)


@needs_mp
def test_ckpt_probe_replay_rule():
    """ROWPAIR_CKPT_DIR resume: the recycle-0 early-stop probe's decision is REPLAYED (continue) when the item resumed past cycle 0 (cycle_k,
    k >= 1, or trunk_final) and JUDGED on cycle-0 state (the caller's function) on a cycle_000 resume or no resume."""
    CK = _core.load("mem.rowpair.ckpt")
    seen = []

    def user_fn(confidence_outputs=None, first_recycle_outputs=None):
        seen.append(1)
        return True, {"why": "user"}

    wrapped = rowpair._replayed_early_stop(user_fn)
    saved = rowpair.CTX.get("resume")
    try:
        rowpair.CTX["resume"] = None
        assert wrapped(confidence_outputs={}, first_recycle_outputs={}) == (True, {"why": "user"}) and seen == [1] and not rowpair._probe_replayed()
        rowpair.CTX["resume"] = {"tag": "cycle_000", "cycle": 0}
        assert wrapped() == (True, {"why": "user"}) and seen == [1, 1]
        rowpair.CTX["resume"] = {"tag": "cycle_003", "cycle": 3}
        assert wrapped() == (False, {}) and seen == [1, 1] and rowpair._probe_replayed()
        rowpair.CTX["resume"] = {"tag": CK.TRUNK_FINAL, "cycle": 1}
        assert wrapped() == (False, {}) and seen == [1, 1]
    finally:
        rowpair.CTX["resume"] = saved


def af3_noise_scale_to_noise_level(x):                          # stand-in of rf3.data.ground_truth_template's (elementwise)
    return torch.log1p(x)


def discretize_distance_matrix(distance_matrix, num_bins=38, min_distance=3.25, max_distance=50.75):   # stock body
    bin_width = (max_distance - min_distance) / num_bins
    bins = torch.arange(num_bins, device=distance_matrix.device) * bin_width + min_distance
    return torch.bucketize(distance_matrix, bins)


if HAVE_TORCH:
    def _batched4(fn):
        def call(x):
            if x.dim() == 4:
                return fn(x)
            o = fn(x[None])
            assert o.shape[0] == 1
            return o[0]
        return call

    class MultiDimLinear(nn.Linear):
        def __init__(self, in_features, out_shape, **kw):
            self.out_shape = tuple(out_shape)
            super().__init__(in_features, int(math.prod(out_shape)), **kw)

        def forward(self, x):
            return super().forward(x).reshape(x.shape[:-1] + self.out_shape)

    class Transition(nn.Module):
        def __init__(self, n, c):
            super().__init__()
            self.layer_norm_1 = nn.LayerNorm(c)
            self.linear_1 = nn.Linear(c, n * c, bias=False)
            self.linear_2 = nn.Linear(c, n * c, bias=False)
            self.linear_3 = nn.Linear(n * c, c, bias=False)

        def forward(self, X):
            X = self.layer_norm_1(X)
            return self.linear_3(F.silu(self.linear_1(X)) * self.linear_2(X))

    class TriangleMultiplication(nn.Module):
        def __init__(self, d_pair, d_hidden, direction):
            super().__init__()
            self.d_pair, self.d_hidden, self.direction, self.use_cuequivariance = d_pair, d_hidden, direction, True
            self.norm_in = nn.LayerNorm(d_pair)
            self.p_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)
            self.g_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)
            self.norm_out = nn.LayerNorm(d_hidden)
            self.p_out = nn.Linear(d_hidden, d_pair, bias=False)
            self.g_out = nn.Linear(d_pair, d_pair, bias=False)

        def forward(self, pair):                                # stock _forward_vanilla
            B, L = pair.shape[:2]
            pair_norm = self.norm_in(pair)
            p = self.p_in(pair_norm)
            left, right = p[..., : self.d_hidden], p[..., self.d_hidden:]
            g = torch.sigmoid(self.g_in(pair_norm))
            left = g[..., : self.d_hidden] * left
            right = g[..., self.d_hidden:] * right
            right = right / float(L)
            if self.direction == "outgoing":
                out = torch.einsum("bikd,bjkd->bijd", left, right)
            else:
                out = torch.einsum("bkid,bkjd->bijd", left, right)
            out = self.p_out(self.norm_out(out))
            return torch.sigmoid(self.g_out(pair_norm)) * out

    class TriangleAttention(nn.Module):
        def __init__(self, d_pair, n_head, d_hidden, start_node):
            super().__init__()
            self.norm = nn.LayerNorm(d_pair)
            self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
            self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
            self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
            self.to_b = nn.Linear(d_pair, n_head, bias=False)
            self.to_g = nn.Linear(d_pair, n_head * d_hidden)
            self.to_out = nn.Linear(n_head * d_hidden, d_pair)
            self.scaling = 1 / math.sqrt(d_hidden)
            self.h, self.dim, self.start_node, self.use_cuequivariance = n_head, d_hidden, start_node, True

        def forward(self, pair):                                # stock forward + _forward_vanilla
            pair = self.norm(pair)
            bias = self.to_b(pair)
            if not self.start_node:
                pair = pair.transpose(1, 2)
            out = self._forward_vanilla(pair, bias)
            if not self.start_node:
                out = out.transpose(1, 2)
            return self.to_out(out)

        def _forward_vanilla(self, pair, bias):
            B, L = pair.shape[:2]
            gate = torch.sigmoid(self.to_g(pair))
            query = self.to_q(pair).reshape(B, L, L, self.h, -1) * self.scaling
            key = self.to_k(pair).reshape(B, L, L, self.h, -1)
            value = self.to_v(pair).reshape(B, L, L, self.h, -1)
            attn = torch.einsum("bijhd,bikhd->bijkh", query, key) + bias.unsqueeze(1).expand(-1, L, -1, -1, -1)
            attn = F.softmax(attn, dim=-2)
            return gate * torch.einsum("bijkh,bikhd->bijhd", attn, value).reshape(B, L, L, -1)

    class AttentionPairBiasPairformerDeepspeed(nn.Module):
        def __init__(self, c_a, c_pair, n_head):
            super().__init__()
            self.n_head, self.c_a, self.c_pair, self.c = n_head, c_a, c_pair, c_a // n_head
            self.to_q = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_b = nn.Linear(c_pair, n_head, bias=False)
            self.to_g = nn.Sequential(MultiDimLinear(c_a, (n_head, self.c), bias=False), nn.Sigmoid())
            self.to_a = nn.Linear(c_a, c_a, bias=False)
            self.ln_0 = nn.LayerNorm((c_pair,))
            self.ln_1 = nn.LayerNorm((c_a,))
            self.use_deepspeed_evo = False
            self.force_bfloat16 = False                         # fp64 test (stock: True; the adapter mirrors the flag)

        def forward(self, A_I, S_I, Z_II, Beta_II=None):        # stock statement (non-deepspeed branch)
            assert S_I is None
            A_I = self.ln_1(A_I)
            if (self.use_deepspeed_evo or self.force_bfloat16) and A_I.device.type != "mps":
                A_I = A_I.to(torch.bfloat16)
            Q_IH, K_IH, V_IH = self.to_q(A_I), self.to_k(A_I), self.to_v(A_I)
            B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
            G_IH = self.to_g(A_I)
            Q_IH = Q_IH / torch.sqrt(torch.tensor(self.c).to(Q_IH.device, Q_IH.dtype))
            A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = (G_IH * A_I).flatten(start_dim=-2)
            return self.to_a(A_I)

    class OuterProductMean_AF3(nn.Module):
        def __init__(self, c_msa_embed, c_outer_product, c_out):
            super().__init__()
            self.norm = nn.LayerNorm(c_msa_embed)
            self.proj_left = nn.Linear(c_msa_embed, c_outer_product)
            self.proj_right = nn.Linear(c_msa_embed, c_outer_product)
            self.proj_out = nn.Linear(c_outer_product * c_outer_product, c_out)

        def forward(self, msa):
            B, N, L = msa.shape[:3]
            msa = self.norm(msa)
            left = self.proj_left(msa)
            right = self.proj_right(msa) / float(N)
            out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
            return self.proj_out(out)

    class MSAPairWeightedAverage(nn.Module):
        def __init__(self, c_pair, c_msa, c_avg, n_heads, separate):
            super().__init__()
            self.weighted_average_channels, self.n_heads, self.separate_gate_for_every_channel = c_avg, n_heads, separate
            self.norm_msa = nn.LayerNorm(c_msa)
            self.norm_pair = nn.LayerNorm(c_pair)
            self.to_v = nn.Linear(c_msa, c_avg * n_heads, bias=False)
            self.to_bias = nn.Linear(c_pair, n_heads, bias=False)
            self.to_gate = nn.Linear(c_msa, c_avg * n_heads if separate else n_heads, bias=False)
            self.to_out = nn.Linear(c_avg * n_heads, c_msa, bias=False)

        def forward(self, msa_SI, pair_II):
            S, I = msa_SI.shape[:2]
            msa_SI = self.norm_msa(msa_SI)
            v_SIH = self.to_v(msa_SI).reshape(S, I, self.n_heads, self.weighted_average_channels)
            w_IIH = F.softmax(self.to_bias(self.norm_pair(pair_II)), dim=-2)
            gate_SIH = torch.sigmoid(self.to_gate(msa_SI))
            weights = torch.einsum("ijh,sjhc->sihc", w_IIH, v_SIH)
            if self.separate_gate_for_every_channel:
                o = gate_SIH * weights.reshape(S, I, -1)
            else:
                o = gate_SIH[..., None] * weights
            return self.to_out(o.reshape(S, I, -1))

    class MSASubsampleEmbedder(nn.Module):
        def __init__(self, c_in, c_m):
            super().__init__()
            self.emb_msa = nn.Linear(c_in, c_m, bias=False)
            self.emb_S_inputs = nn.Linear(C_SI, c_m, bias=False)

        def forward(self, msa_SI, S_inputs):
            return self.emb_msa(msa_SI) + self.emb_S_inputs(S_inputs)

    class PairformerBlock(nn.Module):
        def __init__(self, c_s, c_z, n_head=N_HEAD):
            super().__init__()
            self.tri_mul_outgoing = TriangleMultiplication(c_z, c_z, "outgoing")
            self.tri_mul_incoming = TriangleMultiplication(c_z, c_z, "incoming")
            self.tri_attn_start = TriangleAttention(c_z, 2, 4, True)
            self.tri_attn_end = TriangleAttention(c_z, 2, 4, False)
            self.z_transition = Transition(2, c_z)
            if c_s > 0:
                self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(c_s, c_z, n_head)
                self.s_transition = Transition(2, c_s)

        def forward(self, S_I, Z_II):                            # stock body (dropout: a no-op)
            Z_II = Z_II + _batched4(self.tri_mul_outgoing)(Z_II)
            Z_II = Z_II + _batched4(self.tri_mul_incoming)(Z_II)
            Z_II = Z_II + _batched4(self.tri_attn_start)(Z_II)
            Z_II = Z_II + _batched4(self.tri_attn_end)(Z_II)
            Z_II = Z_II + self.z_transition(Z_II)
            if S_I is not None:
                S_I = S_I + self.attention_pair_bias(S_I, None, Z_II, Beta_II=torch.tensor([0.0], device=Z_II.device))
                S_I = S_I + self.s_transition(S_I)
            return S_I, Z_II

    class RF3TemplateEmbedder(nn.Module):
        def __init__(self, c_z, c):
            super().__init__()
            self.c = c
            self.emb_pair = nn.Linear(c_z, c, bias=False)
            self.norm_pair_before_pairformer = nn.LayerNorm(c_z)
            self.norm_after_pairformer = nn.LayerNorm(c)
            self.emb_templ = nn.Linear(66, c, bias=False)
            self.pairformer = nn.ModuleList([PairformerBlock(0, c) for _ in range(2)])
            self.agg_emb = nn.Linear(c, c_z, bias=False)

        def forward(self, f, Z_II):                               # stock body
            has, ns, cond = f["has_distogram_condition"], f["distogram_condition_noise_scale"], f["distogram_condition"]
            I = Z_II.shape[0]
            joint = (ns[None, :] ** 2 + ns[:, None] ** 2).sqrt()
            lvl = af3_noise_scale_to_noise_level(joint)
            feats = torch.cat([cond, has.unsqueeze(-1), lvl.unsqueeze(-1)], dim=-1) * has.unsqueeze(-1)
            ch = self.emb_templ(feats)
            u = torch.zeros(I, I, self.c, device=Z_II.device)
            v = self.emb_pair(self.norm_pair_before_pairformer(Z_II)) + ch
            for block in self.pairformer:
                _, v = block(None, v)
            u = u + self.norm_after_pairformer(v)
            return self.agg_emb(F.relu(u))

    class MSAModule(nn.Module):
        def __init__(self, c_z, c_m, n_block=2):
            super().__init__()
            self.n_block = n_block
            self.msa_subsampler = MSASubsampleEmbedder(C_M, c_m)
            self.outer_product = OuterProductMean_AF3(c_m, 4, c_z)
            self.msa_pair_weighted_averaging = MSAPairWeightedAverage(c_z, c_m, 3, 2, True)
            self.msa_transition = Transition(2, c_m)
            self.tri_mult_outgoing = TriangleMultiplication(c_z, c_z, "outgoing")
            self.tri_mult_incoming = TriangleMultiplication(c_z, c_z, "incoming")
            self.tri_attn_start = TriangleAttention(c_z, 2, 4, True)
            self.tri_attn_end = TriangleAttention(c_z, 2, 4, False)
            self.pair_transition = Transition(2, c_z)

        def forward(self, f, Z_II, S_inputs_I):                    # stock body (weight-shared blocks)
            msa_SI = self.msa_subsampler(f["msa"], S_inputs_I)
            for _ in range(self.n_block):
                Z_II = Z_II + _batched4(self.outer_product)(msa_SI)
                msa_SI = msa_SI + self.msa_pair_weighted_averaging(msa_SI, Z_II)
                msa_SI = msa_SI + self.msa_transition(msa_SI)
                Z_II = Z_II + _batched4(self.tri_mult_outgoing)(Z_II)
                Z_II = Z_II + _batched4(self.tri_mult_incoming)(Z_II)
                Z_II = Z_II + _batched4(self.tri_attn_start)(Z_II)
                Z_II = Z_II + _batched4(self.tri_attn_end)(Z_II)
                Z_II = Z_II + self.pair_transition(Z_II)
            return Z_II

    class RelativePositionEncoding(nn.Module):
        def __init__(self, r_max, s_max, c_z):
            super().__init__()
            self.r_max, self.s_max = r_max, s_max
            self.linear = nn.Linear(NREL, c_z, bias=False)

        def forward(self, f):                                     # stock body, non-cyclic branch
            b_same_chain = f["asym_id"][:, None] == f["asym_id"][None, :]
            b_same_residue = f["residue_index"][:, None] == f["residue_index"][None, :]
            b_same_entity = f["entity_id"][:, None] == f["entity_id"][None, :]
            d_res = torch.where(b_same_chain, torch.clip(f["residue_index"][:, None] - f["residue_index"][None, :] + self.r_max, 0, 2 * self.r_max), 2 * self.r_max + 1)
            A_relpos = F.one_hot(d_res.long(), 2 * self.r_max + 2)
            d_tok = torch.where(b_same_chain * b_same_residue, torch.clip(f["token_index"][:, None] - f["token_index"][None, :] + self.r_max, 0, 2 * self.r_max), 2 * self.r_max + 1)
            A_reltoken = F.one_hot(d_tok.long(), 2 * self.r_max + 2)
            d_chain = torch.where(b_same_entity, torch.clip(f["sym_id"][:, None] - f["sym_id"][None, :] + self.s_max, 0, 2 * self.s_max), 2 * self.s_max + 1)
            A_relchain = F.one_hot(d_chain.long(), 2 * self.s_max + 2)
            feats = torch.cat([A_relpos, A_reltoken, b_same_entity.unsqueeze(-1).long(), A_relchain], dim=-1).to(torch.get_default_dtype())
            return self.linear(feats)

    class InputFeatureEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(6, C_SI, bias=False)

        def forward(self, f):
            return self.lin(f["tok_feats"])

    class FeatureInitializer(nn.Module):
        def __init__(self, c_s, c_z):
            super().__init__()
            self.input_feature_embedder = InputFeatureEmbedder()
            self.to_s_init = nn.Linear(C_SI, c_s, bias=False)
            self.to_z_init_i = nn.Linear(C_SI, c_z, bias=False)
            self.to_z_init_j = nn.Linear(C_SI, c_z, bias=False)
            self.relative_position_encoding = RelativePositionEncoding(R_MAX, S_MAX, c_z)
            self.process_token_bonds = nn.Linear(1, c_z, bias=False)

        def forward(self, f):                                     # stock body
            S_inputs_I = self.input_feature_embedder(f)
            S_init_I = self.to_s_init(S_inputs_I)
            Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + self.to_z_init_j(S_inputs_I).unsqueeze(-2)
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            Z_init_II = Z_init_II + self.process_token_bonds(f["token_bonds"].unsqueeze(-1).to(torch.get_default_dtype()))
            return S_inputs_I, S_init_I, Z_init_II

    class Recycler(nn.Module):
        def __init__(self, c_s, c_z):
            super().__init__()
            self.process_zh = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, c_z, bias=False))
            self.template_embedder = RF3TemplateEmbedder(c_z, C_T)
            self.msa_module = MSAModule(c_z, C_M)
            self.process_sh = nn.Sequential(nn.LayerNorm(c_s), nn.Linear(c_s, c_s, bias=False))
            self.pairformer_stack = nn.ModuleList([PairformerBlock(c_s, c_z) for _ in range(2)])

        def forward(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II):   # stock body
            Z_II = Z_init_II + self.process_zh(Z_II)
            Z_II = Z_II + self.template_embedder(f, Z_II)
            Z_II = self.msa_module(f, Z_II, S_inputs_I)
            S_I = S_init_I + self.process_sh(S_I)
            for block in self.pairformer_stack:
                S_I, Z_II = block(S_I, Z_II)
            return S_I, Z_II

    Recycler.forward.__module__ = "rf3.model.RF3_structure"      # the adapter's owner check reads the stock owner name (classes= mode skips it)

    class DistogramHead(nn.Module):
        def __init__(self, c_z, bins):
            super().__init__()
            self.predictor = nn.Linear(c_z, bins)

        def forward(self, Z_II):                                  # stock body
            return self.predictor(Z_II + Z_II.transpose(-2, -3))

    class ConfidenceHead(nn.Module):
        def __init__(self, c_s, c_z, feature_ln):
            super().__init__()
            self.layer_norm_along_feature_dimension = feature_ln
            self.use_af3_style_binning_and_final_layer_norms, self.use_Cb_distances = True, False
            self.process_s_inputs_right = nn.Linear(C_SI, c_z, bias=False)
            self.process_s_inputs_left = nn.Linear(C_SI, c_z, bias=False)
            self.process_pred_distances = nn.Linear(40, c_z, bias=False)
            self.pairformer = nn.ModuleList([PairformerBlock(c_s, c_z) for _ in range(2)])
            self.layernorm_pae, self.predict_pae = nn.LayerNorm(c_z), nn.Linear(c_z, BINS, bias=False)
            self.layernorm_pde, self.predict_pde = nn.LayerNorm(c_z), nn.Linear(c_z, BINS, bias=False)
            self.predict_plddt = nn.Linear(c_s, NHEAVY_STUB * 3, bias=False)          # atoms * bins, as RF3's (its consumer reshapes by NHEAVY)

        def embed(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, rep_atoms):            # stock body :97-150
            S_trunk_I, Z_trunk_II, S_inputs_I = S_trunk_I.detach(), Z_trunk_II.detach(), S_inputs_I.detach()
            if X_pred_L is not None:
                X_pred_L = X_pred_L.detach()
            if self.layer_norm_along_feature_dimension:
                S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape[-1],))
                Z_trunk_II = F.layer_norm(Z_trunk_II, normalized_shape=(Z_trunk_II.shape[-1],))
                S_inputs_I = F.layer_norm(S_inputs_I, normalized_shape=(S_inputs_I.shape[-1],))
            else:
                S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=S_trunk_I.shape)
                Z_trunk_II = F.layer_norm(Z_trunk_II, normalized_shape=Z_trunk_II.shape)
                S_inputs_I = F.layer_norm(S_inputs_I, normalized_shape=S_inputs_I.shape)
            Z_trunk_II = Z_trunk_II + (self.process_s_inputs_right(S_inputs_I).unsqueeze(-2) + self.process_s_inputs_left(S_inputs_I).unsqueeze(-3))
            if X_pred_L is not None:
                X_rep = X_pred_L.index_select(1, rep_atoms)
                dist = torch.cdist(X_rep, X_rep)
                one_hot = F.one_hot(discretize_distance_matrix(dist, min_distance=3.25, max_distance=50.75, num_bins=39), num_classes=40)
                Z_trunk_II = Z_trunk_II + self.process_pred_distances(one_hot.to(torch.get_default_dtype()))
            return {"S": S_trunk_I, "Z": Z_trunk_II}

        def forward(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs=None):   # stock body up to the logits
            e = self.embed(S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, rep_atoms)
            S, Z = e["S"], e["Z"]
            for n in range(len(self.pairformer)):
                S, Z = self.pairformer[n](S, Z)
            pae = self.predict_pae(self.layernorm_pae(Z))
            left = self.predict_pde(self.layernorm_pde(Z))
            return dict(pae_logits=pae, pde_logits=left + left.transpose(-2, -3), plddt_logits=self.predict_plddt(S), exp_resolved_logits=S.sum() * 0)

    PLAN_WORDS = []                                               # the trunk-shard plan's per-pass words the stand-in saw (conf_ztrunk census)

    class TpConfStub:                                             # stands in for tp_conf.run_confidence_sharded (keyword interface): the stock
        @staticmethod                                             # embedding on the test-gathered rows, the CALLER's pair_stack_fn on the shard
        def run_confidence_sharded(head, *, z_trunk_shard, layout, S_inputs_I, S_trunk_I, X_pred_L, rep_atoms, pair_stack_fn, asym_id,
                                   is_ligand, rf3, pae=(32.0, 64), pde=(32.0, 64), tm=(32.0, 64), **kw):
            SH = _core.load("mem.rowpair.shard")
            plan = kw["plan"]                                                                 # the adapter's trunk-shard plan: this call is pass kw["pass_index"]; the stand-in
            src, _ = plan.begin(int(kw.get("pass_index", 0)), lead_out=(1,), last_use=kw.get("last_use"), out_dtype=z_trunk_shard.dtype)   # begins it as tp_conf does and reads the rows from what it hands back
            PLAN_WORDS.append(plan.words[int(kw.get("pass_index", 0))])
            loc = src.zrows(0, layout.R) if hasattr(src, "zrows") else src
            z = SH.unshard_rows(loc.contiguous(), layout, dim=0)
            dense = ConfidenceHead.embed(head, S_inputs_I, S_trunk_I, z, X_pred_L, rep_atoms)
            S = dense["S"]
            z_rows = dense["Z"][..., layout.r0:layout.r1, :, :]
            z_rows = z_rows[0] if z_rows.dim() == 4 else z_rows
            z_rows, S = pair_stack_fn(z_rows.contiguous(), S)
            zf = SH.unshard_rows(z_rows.contiguous(), layout, dim=0)
            if dense["Z"].dim() == 4 and zf.dim() == 3:
                zf = zf.unsqueeze(0)
            if X_pred_L is None:                                                              # the recycle-0 probe: no distance term, pLDDT only
                return dict(pae=None, pde=None, tm=None, plddt_logits=head.predict_plddt(S), exp_resolved_logits=S.sum() * 0)
            pae_l = head.predict_pae(head.layernorm_pae(zf))
            left = head.predict_pde(head.layernorm_pde(zf))
            return dict(pae=pae_l, pde=left + left.transpose(-2, -3), plddt_logits=head.predict_plddt(S), exp_resolved_logits=S.sum() * 0,
                        tm={"ptm": zf.sum().reshape(1)})

    class AdaLN(nn.Module):
        def __init__(self, c_a, c_s):
            super().__init__()
            self.ln_a = nn.LayerNorm(c_a, elementwise_affine=False)
            self.ln_s = nn.LayerNorm(c_s, bias=False)
            self.to_g = nn.Linear(c_s, c_a)
            self.to_b = nn.Linear(c_s, c_a, bias=False)

        def forward(self, a, s):
            s = self.ln_s(s)
            return torch.sigmoid(self.to_g(s)) * self.ln_a(a) + self.to_b(s)

    class ConditionedTransitionBlock(nn.Module):
        def __init__(self, c_token, c_s, n=2):
            super().__init__()
            self.ada_ln = AdaLN(c_token, c_s)
            self.linear_1 = nn.Linear(c_token, c_token * n, bias=False)
            self.linear_2 = nn.Linear(c_token, c_token * n, bias=False)
            self.linear_output_project = nn.Sequential(nn.Linear(c_s, c_token), nn.Sigmoid())
            self.linear_3 = nn.Linear(c_token * n, c_token, bias=False)

        def forward(self, Ai, Si):                                # stock body
            Ai = self.ada_ln(Ai, Si)
            Bi = F.silu(self.linear_1(Ai)) * self.linear_2(Ai)
            return self.linear_output_project(Si) * self.linear_3(Bi)

    class AttentionPairBiasDiffusion(nn.Module):
        def __init__(self, c_a, c_s, c_pair, n_head, kq_norm=True):
            super().__init__()
            self.n_head, self.c_a, self.c_pair, self.c = n_head, c_a, c_pair, c_a // n_head
            self.to_q = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False)
            self.to_b = nn.Linear(c_pair, n_head, bias=False)
            self.to_g = nn.Sequential(MultiDimLinear(c_a, (n_head, self.c), bias=False), nn.Sigmoid())
            self.to_a = nn.Linear(c_a, c_a, bias=False)
            self.linear_output_project = nn.Sequential(nn.Linear(c_s, c_a), nn.Sigmoid())
            self.ln_0 = nn.LayerNorm((c_pair,))
            self.ada_ln_1 = AdaLN(c_a, c_s)
            self.use_deepspeed_evo, self.force_bfloat16, self.kq_norm = False, False, kq_norm
            self.key_layer_norm = nn.LayerNorm((n_head * self.c,))
            self.query_layer_norm = nn.LayerNorm((n_head * self.c,))

        def forward(self, A_I, S_I, Z_II, Beta_II):               # stock (patched) body, token level, hoisted bias
            GF = sys.modules["rf3.graph_flags"]
            A_I = self.ada_ln_1(A_I, S_I)
            Q_IH, K_IH, V_IH = self.to_q(A_I), self.to_k(A_I), self.to_v(A_I)
            B_IIH = GF.hoist_get((id(self), "B_IIH"), lambda: self.to_b(self.ln_0(Z_II)))
            G_IH = self.to_g(A_I)
            Q_IH = self.query_layer_norm(Q_IH.reshape(-1, self.n_head * self.c)).reshape(Q_IH.shape)
            K_IH = self.key_layer_norm(K_IH.reshape(-1, self.n_head * self.c)).reshape(K_IH.shape)
            Q_IH = Q_IH / math.sqrt(self.c)
            A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = (G_IH * A_I).flatten(start_dim=-2)
            A_I = self.to_a(A_I)
            return self.linear_output_project(S_I) * A_I

    class DiffusionTransformerBlock(nn.Module):
        def __init__(self, c_token, c_s, c_pair, n_head, no_res=True):
            super().__init__()
            self.attention_pair_bias = AttentionPairBiasDiffusion(c_token, c_s, c_pair, n_head)
            self.conditioned_transition_block = ConditionedTransitionBlock(c_token, c_s)
            self.no_residual_connection_between_attention_and_transition = no_res

        def forward(self, A_I, S_I, Z_II, Beta_II):               # stock body
            if self.no_residual_connection_between_attention_and_transition:
                B_I = self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
                return A_I + B_I + self.conditioned_transition_block(A_I, S_I)
            A_I = A_I + self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
            return A_I + self.conditioned_transition_block(A_I, S_I)

    class DiffusionTransformer(nn.Module):
        def __init__(self, c_token, c_s, c_pair, n_block=3):
            super().__init__()
            self.blocks = nn.ModuleList([DiffusionTransformerBlock(c_token, c_s, c_pair, 2, no_res=(i % 2 == 0)) for i in range(n_block)])

        def forward(self, A_I, S_I, Z_II, Beta_II):
            for block in self.blocks:
                A_I = block(A_I, S_I, Z_II, Beta_II)
            return A_I

    class DiffusionModule(nn.Module):                             # the denoiser entry: conditioning + token transformer (X_noisy_L plays the token single)
        def __init__(self, c_z, c_s, c_a):
            super().__init__()
            self.diffusion_conditioning = DiffusionConditioning(c_z, c_s)
            self.diffusion_transformer = DiffusionTransformer(c_a, c_s, c_z)

        def forward(self, X_noisy_L, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):   # stock signature
            S_c, Z_c = self.diffusion_conditioning.forward_hoisted(t, f, S_inputs_I, S_trunk_I, Z_trunk_II)
            return self.diffusion_transformer(X_noisy_L, S_c, Z_c, None), S_c, Z_c

    class DiffusionConditioning(nn.Module):
        def __init__(self, c_z, c_s):
            super().__init__()
            self.sigma_data = 16.0
            self.relative_position_encoding = RelativePositionEncoding(R_MAX, S_MAX, c_z)
            self.to_zii = nn.Sequential(nn.LayerNorm(2 * c_z), nn.Linear(2 * c_z, c_z, bias=False))
            self.transition_1 = nn.ModuleList([Transition(2, c_z), Transition(2, c_z)])
            self.to_si = nn.Sequential(nn.LayerNorm(c_s + C_SI), nn.Linear(c_s + C_SI, c_s, bias=False))
            self.process_n = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, c_s, bias=False))
            self.transition_2 = nn.ModuleList([Transition(2, c_s), Transition(2, c_s)])

        def forward_hoisted(self, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):   # the kit's patched body
            GF = sys.modules["rf3.graph_flags"]

            def _pair():
                Z_II = torch.cat([Z_trunk_II, self.relative_position_encoding(f)], dim=-1)
                Z_II = self.to_zii(Z_II)
                for b in range(2):
                    Z_II = Z_II + self.transition_1[b](Z_II)
                return Z_II

            def _single_pre():
                return self.to_si(torch.cat([S_trunk_I, S_inputs_I], dim=-1))

            Z_II = GF.hoist_get((id(self), "Z_II"), _pair)
            S_pre = GF.hoist_get((id(self), "S_pre"), _single_pre)
            N_D = torch.cos(torch.arange(4.0) * (1 / 4 * torch.log(t / self.sigma_data)))
            S_I = self.process_n(N_D).unsqueeze(-2) + S_pre
            for b in range(2):
                S_I = S_I + self.transition_2[b](S_I)
            return S_I, Z_II

        forward = forward_hoisted

    class Fabric:                                                 # lightning Fabric.to_device-shaped: the engine's H2D of the featurised example (inference_engines/rf3.py:562)
        device = torch.device("cpu")

        def to_device(self, obj):
            def mv(x):
                return {k: mv(v) for k, v in x.items()} if isinstance(x, dict) else (x.to(self.device) if torch.is_tensor(x) else x)
            return mv(obj)

    class SampleDiffusion:
        def _get_initial_structure(self, c0, D, L, coord):     # stock statement
            return c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device) + coord

        def _predraw(self, noise_schedule, D, L, device):     # the kit-patched sampler's pre-draw (shapes as its docstring)
            T = int(noise_schedule.shape[0]) - 1
            init = torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device)
            E = torch.stack([torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device) for _ in range(T)])
            return init, E

    class Model(nn.Module):                                       # RF3WithConfidence-shaped entry: the control flow around the sites
        def __init__(self):
            super().__init__()
            self.feature_initializer = FeatureInitializer(C_S, C_Z)
            self.recycler = Recycler(C_S, C_Z)
            self.distogram_head = DistogramHead(C_Z, BINS)
            self.diffusion_module = DiffusionModule(C_Z, C_S, C_A)
            self.process_z = nn.Sequential(nn.LayerNorm(C_Z), nn.Linear(C_Z, C_AP, bias=False))   # the atom encoder's token-pair projection
            self.confidence_head = ConfidenceHead(C_S, C_Z, feature_ln=True)
            self.confidence_head_g = ConfidenceHead(C_S, C_Z, feature_ln=False)

        def forward(self, input, n_cycle=N_CYC):
            GF = sys.modules["rf3.graph_flags"]
            LV = sys.modules["rosettafold3_opt.levers"]
            f = input["f"]
            f["msa_stack"] = f["msa_stack"].to(torch.get_default_dtype())   # stock: the autocast cast of the raw MSA stack at the model entry (RF3.py:394-404)
            S_inputs_I, S_init_I, Z_init_II = self.feature_initializer(f)
            S_I, Z_II = torch.zeros_like(S_init_I), torch.zeros_like(Z_init_II)
            for i_cycle in range(n_cycle):
                f["msa"] = f["msa_stack"][i_cycle]                # stock: the cycle's i.i.d. MSA subsample (RF3.py:264)
                S_I, Z_II = self.recycler(f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II)
            out = {"S_I": S_I, "Z_II": Z_II, "distogram": self.distogram_head(Z_II)}
            GF.hoist_begin()
            a = input["a0"]
            for step, t in enumerate((torch.tensor(8.0), torch.tensor(3.0))):
                a, S_c, Z_c = self.diffusion_module(a, t, f, S_inputs_I, S_I, Z_II)
                win = LV.PAIR_WINDOWS(self.process_z, Z_c, input["tq"], input["tk"])
                out[f"S_c{step}"], out[f"a{step}"], out[f"win{step}"] = S_c, a, win
            out["Z_c"] = Z_c
            GF.hoist_end()
            for name, head in (("c", self.confidence_head), ("g", self.confidence_head_g)):
                c = head(S_inputs_I[None], S_I[None], Z_II, input["X"], None, torch.arange(N_TOK), frame_atom_idxs=None)   # RF3: batched S, X [1, L, 3]
                out.update({f"{name}_{k}": v for k, v in c.items()})
            return out

    def _graph_flags_stub():
        GF = types.ModuleType("rf3.graph_flags")
        GF.HOIST, GF.HOIST_CACHE, GF.GRAPH_SAFE_OPS, GF.CUDAGRAPH_MODE = True, None, True, "1"

        def hoist_begin(n_calls=None):
            GF.HOIST_CACHE = {}

        def hoist_end():
            GF.HOIST_CACHE = None

        def hoist_get(key, fn):
            c = GF.HOIST_CACHE
            if c is None:
                return fn()
            if key not in c:
                c[key] = fn()
            return c[key]

        def set_mode(mode, safe_ops=None, hoist=None, **_):
            GF.CUDAGRAPH_MODE = mode

        GF.hoist_begin, GF.hoist_end, GF.hoist_get, GF.set_mode = hoist_begin, hoist_end, hoist_get, set_mode
        return GF

    STUBS = {"entry": Model, "pair_init": FeatureInitializer, "recycler": Recycler, "distogram": DistogramHead, "conf_head": ConfidenceHead,
             "diff_cond": DiffusionConditioning, "dit": DiffusionTransformer, "denoiser": DiffusionModule, "noise_init": SampleDiffusion,
             "noise_predraw": SampleDiffusion, "h2d": Fabric}


def _inputs(seed, templated: bool = True):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    asym = torch.tensor([0] * 20 + [1] * 18 + [2] * 12)
    centers = torch.randn(N_TOK, 3, generator=g) * 6.0                                   # the template feature's per-token precursors (templ_feats site) …
    fill = torch.rand(N_TOK, generator=g) > 0.3
    if not templated:                                                                    # an untemplated query: the fill mask names no token (the draws above stay the seed's)
        fill = torch.zeros_like(fill)
    cond, has = test_templ_rows.dense_from_precursors(centers, fill, asym.to(torch.int64), test_templ_rows.EDGES)   # … and the dense pair the stock body reads, from them
    f = {"msa_stack": rnd(N_CYC, N_SEQ, N_TOK, C_M), "has_distogram_condition": has, "distogram_condition": cond.to(torch.float64),
         "distogram_condition_noise_scale": torch.rand(N_TOK, generator=g, dtype=torch.float64),
         "distogram_condition_centers": test_templ_rows.templ.float_bits(centers), "distogram_condition_fill": fill, "distogram_condition_molecule": asym.to(torch.int64),
         "distogram_condition_edges": test_templ_rows.templ.float_bits(test_templ_rows.EDGES.clone()),
         "tok_feats": rnd(N_TOK, 6), "token_bonds": torch.rand(N_TOK, N_TOK, generator=g) > 0.8,
         "asym_id": asym, "entity_id": torch.tensor([0] * 20 + [0] * 18 + [1] * 12), "sym_id": torch.tensor([0] * 20 + [1] * 18 + [0] * 12),
         "residue_index": torch.cat([torch.arange(20), torch.arange(18), torch.arange(12) // 3]),
         "token_index": torch.arange(N_TOK), "cyclic_asym_ids": []}
    n_atom = 3 * N_TOK                                            # 3 atoms per token; windows of 8 queries x 16 keys
    tok_idx = torch.arange(n_atom) // 3
    nq, q, k = n_atom // 8 + 1, 8, 16
    iQ = torch.clamp(torch.arange(nq)[:, None] * q + torch.arange(q)[None, :], 0, n_atom - 1)
    iK = torch.clamp(torch.arange(nq)[:, None] * q - 4 + torch.arange(k)[None, :], 0, n_atom - 1)
    return {"f": f, "t": rnd(D_SMP).abs() + 1.0, "X": rnd(1, N_TOK, 3) * 10, "a0": rnd(D_SMP, N_TOK, C_A), "tq": tok_idx[iQ], "tk": tok_idx[iK]}   # t [D]: the per-sample noise levels (RF3.py:406 reads D = t.shape[0])


def _cmp(a, b):
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return {"max_abs": float("inf"), "bitwise": False, "shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)]}
    d = (a - b).abs().max().item() if a.numel() else 0.0
    return {"max_abs": d, "bitwise": bool(torch.equal(a, b)), "shape": list(a.shape)}



def _early_stop(confidence_outputs, threshold=0.5, max_value=1.0):
    """RF3's ``should_early_stop_by_mean_plddt`` consumer, statement for statement: ``plddt_logits.unsqueeze(0)`` must be 3-D
    ``(B, n_token, max_atoms * n_bins)`` (``get_mean_atomwise_plddt``'s assertion, its exact text), then the expected pLDDT over the bins
    and the mean over (token, atom); returns ``(stop, mean_plddt)``."""
    pl = confidence_outputs["plddt_logits"].unsqueeze(0)
    assert pl.ndim == 3, "plddt_logits must be a 3D tensor (B, n_token, max_atoms_in_a_token * n_bins)"
    B, n_tok, last = pl.shape
    n_bins = last // NHEAVY_STUB
    logits = pl.reshape(B, n_tok, NHEAVY_STUB, n_bins).permute(0, 3, 1, 2).double()        # [B, n_bins, n_token, atoms]
    centers = (torch.arange(n_bins, dtype=torch.float64) + 0.5) * (max_value / n_bins)
    p = torch.softmax(logits, dim=1)
    plddt = (p * centers[None, :, None, None]).sum(dim=1)                                     # [B, n_token, atoms]
    mean = plddt.mean(dim=(-1, -2))
    return [bool((mean < threshold).item()), float(mean.item())]


def _h2d(inp):
    """The engine's H2D of the featurised example (``pipeline_output = trainer.fabric.to_device(pipeline_output)``, inference_engines/rf3.py:562)
    on the Fabric stand-in: under n_gpu>1 with ROWPAIR_MSA_HOST set the adapter's wrap holds ``msa_stack`` back; else a plain move."""
    moved = STUBS["h2d"]().to_device({"example_id": "q", "feats": inp["f"]})
    return dict(inp, f=moved["feats"])


def _ckpt_leg(model, _h2d, _inputs, RP, ckpt_dir):
    """ROWPAIR_CKPT_DIR through the adapter (core TrunkCheckpointer at Recycler.forward): run A writes cycle_000 + trunk_final; run B resumes
    trunk_final (every recycle fast-forwarded) and run C resumes cycle_000 (recycle 1 recomputed) — both bitwise identical to A in every output;
    a run whose features differ is refused by name before any byte loads, and allowed (named) under ROWPAIR_RESUME_ALLOW_FEATS=1."""
    CK = _core.load("mem.rowpair.ckpt")
    EV = _core.load("mem.rowpair.evidence")
    saved_env = {k: os.environ.get(k) for k in ("ROWPAIR_CKPT_DIR", "ROWPAIR_CKPT_EVERY", "ROWPAIR_RESUME", "ROWPAIR_RESUME_TAG", "ROWPAIR_RESUME_ALLOW_FEATS")}
    sys.argv = [a for a in sys.argv if not str(a).startswith("seed=")] + ["seed=7"]        # the rank command's seed token (fold.cmd_seed)
    out = {}
    try:
        os.environ["ROWPAIR_CKPT_DIR"] = ckpt_dir
        os.environ["ROWPAIR_CKPT_EVERY"] = "1"
        with torch.no_grad():
            A = model(_h2d(_inputs(1)))
        schedA = dict(EV.schedule())
        out["wrote"] = schedA.get("ckpt_wrote"); out["query"] = schedA.get("ckpt_query"); out["ckpt"] = schedA.get("ckpt")
        qdir = os.path.join(ckpt_dir, str(schedA.get("ckpt_query")))
        out["tags"] = sorted(os.listdir(qdir)) if os.path.isdir(qdir) else []
        calls0 = dict(rowpair.CTX["calls"].get("recycler", {}))
        with torch.no_grad():
            B = model(_h2d(_inputs(1)))                                                     # ROWPAIR_RESUME=1 (default): trunk_final
        schedB = dict(EV.schedule())
        out["resume_B"] = schedB.get("ckpt_resume"); out["ff_B"] = schedB.get("ckpt_ff_cycles")
        out["B_bitwise"] = {k: bool(torch.equal(A[k].cpu(), B[k].cpu())) for k in A if torch.is_tensor(A[k]) and torch.is_tensor(B.get(k))}
        out["B_resumed_calls"] = int(rowpair.CTX["calls"]["recycler"].get("resumed", 0)) - int(calls0.get("resumed", 0))
        os.environ["ROWPAIR_RESUME_TAG"] = "cycle_000"
        with torch.no_grad():
            C = model(_h2d(_inputs(1)))                                                     # after cycle 0: recycle 1 recomputed on the resumed shard
        schedC = dict(EV.schedule())
        out["resume_C"] = schedC.get("ckpt_resume"); out["ff_C"] = schedC.get("ckpt_ff_cycles")
        out["C_bitwise"] = {k: bool(torch.equal(A[k].cpu(), C[k].cpu())) for k in A if torch.is_tensor(A[k]) and torch.is_tensor(C.get(k))}
        os.environ.pop("ROWPAIR_RESUME_TAG")
        other = _h2d(_inputs(1))
        other["f"]["distogram_condition_noise_scale"] = other["f"]["distogram_condition_noise_scale"] + 1.0   # another item's features under the same query id
        try:
            with torch.no_grad():
                model(other)
            out["differ_refused"] = False
        except RP.RowpairRefused as e:
            out["differ_refused"] = str(e)[:160]
        os.environ["ROWPAIR_RESUME_ALLOW_FEATS"] = "1"
        other = _h2d(_inputs(1))                                                            # a fresh item (an item's host MSA hold ends with the item, refused or not)
        other["f"]["distogram_condition_noise_scale"] = other["f"]["distogram_condition_noise_scale"] + 1.0
        with torch.no_grad():
            model(other)                                                                    # the named allowance: resumes anyway
        out["differ_allowed"] = str(dict(EV.schedule()).get("ckpt_resume"))
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _entry_e2e(perturb_rank1: bool, msa_host: str = "", ckpt_dir=None):
    """Rank body: dense reference (stock stand-ins), then install + the sharded run of the same model object; rank 0 returns the comparison.
    ``msa_host``: the rank's ROWPAIR_MSA_HOST ('' = unset, rank0, all)."""
    os.environ["ROWPAIR_OPM_ROWS"] = "8"
    os.environ["ROWPAIR_TRIATT_ROWBLOCK"] = "4"
    if msa_host:
        os.environ["ROWPAIR_MSA_HOST"] = msa_host
    else:
        os.environ.pop("ROWPAIR_MSA_HOST", None)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    sys.modules["rf3.graph_flags"] = _graph_flags_stub()
    from .. import levers
    import importlib
    pkg = importlib.import_module(rowpair.__name__.rsplit(".", 1)[0])
    tpc = types.ModuleType(pkg.__name__ + ".tp_conf")             # the finish's fp64 stand-in (tp_conf.py is exercised by tests/test_tp_conf_cpu.py):
    tpc.run_confidence_sharded = TpConfStub.run_confidence_sharded  # this test covers the adapter's call contract + its pair_stack_fn on the shard
    tpc.RF3Fns = lambda **kw: kw
    saved_tpc = (sys.modules.get(tpc.__name__), getattr(pkg, "tp_conf", None))
    sys.modules[tpc.__name__] = tpc
    setattr(pkg, "tp_conf", tpc)
    mu = types.ModuleType("rf3.metrics.metric_utils")
    mu.find_bin_midpoints, mu.unbin_logits = (lambda *a, **k: None), (lambda *a, **k: None)
    sys.modules["rf3.metrics.metric_utils"] = mu
    model = Model().eval()
    for p in model.parameters():                                  # non-degenerate gates/outputs (stock inits zero some of them)
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.2)
    inp = _inputs(1)
    DENSE["conf_forward"] = STUBS["conf_head"].forward                                        # the stock body, captured before install patches the class
    ENGINE["feats"] = {"distogram_condition": inp["f"]["distogram_condition"].float().clone(),   # the engine's pre-cast batch outside f: an N x N twin of
                       "unrelated": inp["f"]["distogram_condition"].float().clone()}             # another dtype (reached by key) + a key it must not touch
    with torch.no_grad():
        dense = model(_h2d(inp))                                  # the stand-in Fabric unwrapped: a plain move
    P = rowpair.env_world()
    classes = dict(STUBS, pair_windows=levers)
    RP = _core.load("mem.rowpair")
    census = rowpair.install({}, P, classes=classes, init_group=False)
    again = rowpair.install({}, P, classes=classes, init_group=False)                        # ONCE per process: a second install (the watch's callback running
    assert again["reinstall"] == "skipped:already_installed" and again["installed"] and again["patched"] == census["patched"]   # again) is a named no-op, the sites as bound
    assert "reinstall" not in census and rowpair.CTX["installed"] and rowpair.CTX["n_gpu"] == P
    try:
        rowpair.install(None, P + 1, classes=classes, init_group=False)                     # ... asking for another P is refused by name
        raise AssertionError("a second install under another n_gpu must be refused")
    except RP.RowpairRefused as e:
        assert "installed under n_gpu=%d" % P in str(e), str(e)
    r = census["rank"]
    shardmod = _core.load("mem.rowpair.shard")
    if perturb_rank1 and r == 1:
        inp = _inputs(2)                                          # rank 1 disagrees on every replicated input (its raw MSA stack too); rank 0's must win
    inp = _h2d(inp)                                               # the engine's H2D through the wrapped seam: ROWPAIR_MSA_HOST holds msa_stack back (rank 0: pinned host; ranks > 0 under rank0: placeholder)
    msa = {"entry_shape": list(inp["f"]["msa_stack"].shape), "held": None if rowpair.CTX["msa_host"] is None else
           {"mode": rowpair.CTX["msa_host"]["mode"], "shape": list(rowpair.CTX["msa_host"]["feats"]["msa_stack"].shape), "parked": rowpair.CTX["msa_host"]["parked"],
            "placeholders": rowpair.CTX["msa_host"]["placeholders"]}}
    with torch.no_grad():
        sh = model(inp)
    msa["held_after"] = rowpair.CTX["msa_host"] is not None       # the item's hold is dropped at its exit
    msa["seam_refused"] = None
    if msa_host:                                                  # the lever set but the seam bypassed (a whole stack reaches the model): refused by name on every rank, no fallback
        raw = _inputs(1)
        try:
            with torch.no_grad():
                model(raw)
            msa["seam_refused"] = False
        except RP.RowpairRefused as e:
            msa["seam_refused"] = str(e)[:400]
    import copy
    _calls_main = copy.deepcopy(rowpair.CTX["calls"])              # the legs below run whole items of their own; the site-call census asserted by the test is the main item's
    ck = _ckpt_leg(model, _h2d, _inputs, RP, ckpt_dir) if ckpt_dir else None   # ROWPAIR_CKPT_DIR: write, resume (trunk_final / cycle_000) bitwise, digest refusal
    tr = {}                                                       # ROWPAIR_TRANSPOSE_INPLACE (the line exports 1): every pair block's ending-orientation transpose inside the shard's own
    _tin = os.environ.get("ROWPAIR_TRANSPOSE_INPLACE")            # storage (ring.transpose_shard_inplace_) — bitwise equal to the out-of-place transpose in every output
    try:
        forms = {}
        for v in ("0", "1"):
            os.environ["ROWPAIR_TRANSPOSE_INPLACE"] = v
            with torch.no_grad():
                forms[v] = model(_h2d(_inputs(1)))
            forms[v + "_word"] = _core.load("mem.rowpair.evidence").schedule().get("pairstack_transpose")
        tr = {"bitwise": {k: bool(torch.equal(forms["0"][k], forms["1"][k])) for k in forms["0"] if torch.is_tensor(forms["0"][k]) and torch.is_tensor(forms["1"].get(k))},
              "words": [forms["0_word"], forms["1_word"]]}
    finally:
        if _tin is None:
            os.environ.pop("ROWPAIR_TRANSPOSE_INPLACE", None)
        else:
            os.environ["ROWPAIR_TRANSPOSE_INPLACE"] = _tin
    pk = {}                                                       # ROWPAIR_FREE_ZTRUNK + ROWPAIR_CONF_PARK_ZTRUNK (the line exports both): the trunk shard parked at the ROLL-OUT
    _lev = {k: os.environ.get(k) for k in ("ROWPAIR_FREE_ZTRUNK", "ROWPAIR_CONF_PARK_ZTRUNK")}   # ENTRY (distogram), the diffusion conditioning and every confidence
    _pw_main = list(PLAN_WORDS)                                   # pass served from the park — bitwise equal to the resident shard in every output
    try:
        forms = {}
        for v in ("0", "1"):
            os.environ["ROWPAIR_FREE_ZTRUNK"] = os.environ["ROWPAIR_CONF_PARK_ZTRUNK"] = v
            del PLAN_WORDS[:]
            with torch.no_grad():
                res_v = model(_h2d(_inputs(1)))
            schv = _core.load("mem.rowpair.evidence").schedule()
            forms[v] = {k: t for k, t in res_v.items() if k != "Z_II"}              # Z_II: the parked shard's device storage stays released after the last pass (nothing
            forms[v + "_entry"] = schv.get("conf_ztrunk_entry")                     # in RF3 reads it after the confidence stage; the stand-in returns it for the dense check only)
            forms[v + "_words"] = list(PLAN_WORDS)
            forms[v + "_storage"] = int(res_v["Z_II"].untyped_storage().nbytes())
        pk = {"bitwise": {k: bool(torch.equal(forms["0"][k], forms["1"][k])) for k in forms["0"] if torch.is_tensor(forms["0"][k]) and torch.is_tensor(forms["1"].get(k))},
              "entry": [forms["0_entry"], forms["1_entry"]], "words": [forms["0_words"], forms["1_words"]], "storage": [forms["0_storage"], forms["1_storage"]]}
    finally:
        for k, v in _lev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        PLAN_WORDS[:] = _pw_main
    with torch.no_grad():                                         # an UNTEMPLATED item last: the template census on the LEVER line (templ_items / templ_real / templ_fill_max)
        model(_h2d(_inputs(1, templated=False)))                 # is cumulative over the process's items, so the templated items before it stay counted — asserted by the test
    rowpair.CTX["calls"] = _calls_main
    lay = rowpair.CTX["layouts"]["trunk"]
    L = _core.load("mem.rowpair.dist").ctx(N_TOK, align=rowpair.ALIGN)
    full = {}
    for k, v in sh.items():
        if k in ("Z_II", "distogram", "Z_c"):                      # shards -> gather for the comparison (the test's gather, not the adapter's)
            full[k] = shardmod.unshard_rows(v.contiguous(), L, dim=0)
        else:
            full[k] = v
    cmp = {k: _cmp(full[k], dense[k]) for k in dense if k in full}
    missing = sorted(set(dense) - set(full))
    rowpair.CTX["f"] = inp["f"]                                                                # (inside RF3.forward the entry holds f; this call is standalone)
    with torch.no_grad():                                                                      # the recycle-0 early-stop probe EXACTLY as RF3.forward calls it
        fi = model.feature_initializer(inp["f"])                                               # (RF3.py:419-431): UNBATCHED S_inputs_I / S_I / Z_II, X_pred_L=None
        pr = model.confidence_head(fi[0], fi[1], fi[2], None, None, torch.arange(N_TOK))
        z_full = shardmod.unshard_rows(fi[2].contiguous(), L, dim=0)
        pr_dense = DENSE["conf_forward"](model.confidence_head, fi[0], fi[1], z_full, None, None, torch.arange(N_TOK))
    probe = {"pae_none": pr["pae_logits"] is None and pr["pde_logits"] is None, "plddt": _cmp(pr["plddt_logits"], pr_dense["plddt_logits"]),
             "rank": [int(pr["plddt_logits"].dim()), int(pr_dense["plddt_logits"].dim())],
             "early_stop": [_early_stop(pr), _early_stop(pr_dense)]}                          # RF3's consumer on both: (decision, mean pLDDT)
    sd = SampleDiffusion()
    torch.manual_seed(7)                                          # identical seed on both ranks: the guards pass
    x0 = sd._get_initial_structure(torch.tensor(2.0), 3, 10, torch.zeros(3, 10, 3))
    pd = sd._predraw(torch.linspace(1, 0, 5), 3, 10, torch.device("cpu"))
    torch.manual_seed(7)
    ref0 = torch.tensor(2.0) * torch.normal(mean=0.0, std=1.0, size=(3, 10, 3)) + torch.zeros(3, 10, 3)   # rank 0's next draw under seed 7
    torch.manual_seed(7 + (r if perturb_rank1 else 0))           # rank 1 re-seeded differently: rank 0's draw replaces its own (bcast), no refusal
    refused = None
    try:
        x1 = sd._get_initial_structure(torch.tensor(2.0), 3, 10, torch.zeros(3, 10, 3))
        refused = False if torch.equal(x1, ref0) else "rank %d kept a draw that is not rank 0's" % r
    except RP.RowpairRefused as e:
        refused = str(e)[:160]
    train_refused = None
    model.train()
    try:
        with torch.no_grad():
            model(_h2d(_inputs(1)))
        train_refused = False
    except RP.RowpairRefused as e:
        train_refused = str(e)[:120]
    model.eval()
    cyc_refused = None
    inp_c = _h2d(_inputs(1))
    inp_c["f"]["cyclic_asym_ids"] = [1]
    try:
        with torch.no_grad():
            model(inp_c)
        cyc_refused = False
    except RP.RowpairRefused as e:
        cyc_refused = str(e)[:120]
    st = rowpair.state()
    names = rowpair.uninstall(destroy_group=False)               # the launcher's worker owns the group
    fills = [int(_inputs(1, templated=t)["f"]["distogram_condition_fill"].sum()) for t in (True, False)]   # the items' filled-token counts as THIS process drew them (fp64 default dtype: the draws differ from an fp32-default process's)
    return {"rank": r, "P": P, "census": census, "restored": names, "state": st, "noise_shape": list(x0.shape), "n_predraw": len(pd),
            "noise_refused": refused, "train_refused": train_refused, "cyclic_refused": cyc_refused, "probe": probe, "f_rows": list(inp["f"]["token_bonds"].shape), "engine_rows": [list(ENGINE["feats"]["distogram_condition"].shape), list(ENGINE["feats"]["unrelated"].shape)],
            "cmp": cmp, "missing": missing, "lay": lay, "shard_shape": list(sh["Z_II"].shape), "msa": msa, "plan_words": list(PLAN_WORDS),
            "ckpt": ck, "transpose": tr, "park": pk, "templ_fill": fills}


def _mp(P, entry, *args):
    launch = _core.load("mem.rowpair.launch")
    return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=900)



# ----------------------------------------------------------------------------------------------- rank-0 featurisation (site featurise)
FEATS_CALLS = {"n": 0}                                             # Transform-pipeline calls IN THIS PROCESS (receiving ranks add none)
FEATS_DIMS = (3, 5, 7, 4)                                          # the stand-in raw MSA stack [n_recycle, S, I, c]


def _dist_rank() -> int:
    return int(_core.load("mem.rowpair.dist").world()[1])


class _FeatPipeline:
    """The engine's Transform pipeline stand-in: one featurised example per call whose bytes depend on the RANK that runs it (+rank) and on
    the torch CPU generator (one draw, as the template noise / MSA subsample do); ``cfg == "raise"`` fails as a featurisation would."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, inp):
        import numpy as np
        FEATS_CALLS["n"] += 1
        if self.cfg == "raise":
            raise ValueError("no CCD entry for residue ZZZ")
        r = float(_dist_rank())
        n_rec, S, I, c = FEATS_DIMS
        feats = {"msa_stack": torch.arange(n_rec * S * I * c, dtype=torch.float32).reshape(n_rec, S, I, c) + r, "x": torch.arange(6.0).reshape(2, 3) + r,
                 "noise": torch.randn(4), "flag": torch.tensor([True, False]), "idx": torch.arange(I)}
        return {"example_id": inp["example_id"], "feats": feats, "coords": np.arange(I * 3, dtype=np.float64).reshape(I, 3) + r,
                "extra": {"name": "q", "chains": ["A", "B"], "n": int(r) + 1}}


class _FeatBase:                                                    # foundry BaseInferenceEngine-shaped: the constructor the site wraps sets self.pipeline
    def _construct_pipeline(self, cfg):
        self.pipeline = _FeatPipeline(cfg)


class _FeatEngine(_FeatBase):                                       # RF3InferenceEngine-shaped: inherits the constructor, calls self.pipeline(...) per item
    pass


H2D_LOCK = threading.Lock()                                          # the H2D seam records the item in the process-wide CTX: one threaded rank at a time takes its record


def _featurise_rank(rank, P, msa_host, fault, n_items, gate):
    """Rank body (a thread of ``opt_core.testing.run_ranks``: ``dist.comm()`` is this rank's): install (featurise + h2d + the two required
    sites; the second thread's install is the named no-op), build the engine, featurise ``n_items`` items through the wrapped pipeline, then
    the H2D seam and the model-entry placement statement on the last one; report this rank's view. ``fault``: '' | raise (rank 0's
    featurisation raises) | differ (rank 1's digest lies) | mismatch (rank 1 parsed another item than rank 0)."""
    RP, rd, MH, EV = (_core.load(m) for m in ("mem.rowpair", "mem.rowpair.rankdata", "mem.rowpair.msa_host", "mem.rowpair.evidence"))
    classes = {"pair_init": STUBS["pair_init"], "recycler": STUBS["recycler"], "h2d": STUBS["h2d"], "featurise": _FeatBase}
    census = rowpair.install({}, P, classes=classes, init_group=False)
    assert any(n.endswith("_FeatBase._construct_pipeline") for n in census["patched"]), census["patched"]
    out = {"rank": rank, "P": P, "data_form_at_install": dict(EV.schedule()).get("data_form"), "site_flag": rowpair.CTX.get("featurise_site")}
    gate.wait()                                                                             # both ranks have installed and read the schedule before either featurises
    eng = _FeatEngine()
    eng._construct_pipeline(fault or "ok")                                                 # the wrapped constructor: self.pipeline is the rank-0 form now
    assert getattr(eng.pipeline, "_rowpair_rank0", False) and isinstance(eng.pipeline.__wrapped__, _FeatPipeline)
    ex = None
    for k in range(n_items):
        item = f"q{k + 1}" if not (fault == "mismatch" and rank == 1) else "zz"
        try:
            ex = eng.pipeline({"example_id": item})
            out["refusal"] = None
        except RP.RowpairRefused as e:
            ex, out["refusal"] = None, str(e)
            break
    out["pipeline_calls"] = FEATS_CALLS["n"]
    if ex is None:
        return out
    f = ex["feats"]
    ms = f["msa_stack"]
    out.update(x=f["x"].tolist(), noise=f["noise"].tolist(), flag=[f["flag"].dtype == torch.bool, f["flag"].tolist()], coords=ex["coords"].tolist(), extra=ex["extra"],
               keys=sorted(ex), example_id=ex["example_id"], devices=sorted({str(v.device) for v in f.values() if torch.is_tensor(v)}),
               msa={"shape": list(ms.shape), "where": MH.where(ms), "rows": MH.parked_rows(ms, 0), "sum": float(ms.double().sum()) if ms.numel() else None})
    sched = dict(EV.schedule())
    out["sched"] = {k: sched.get(k) for k in ("data_form", "feats_ranks_equal", "feats_digest", "feats_digest_excludes", "feats_unhashed", "feats_items", "feats_src", "feats_rng",
                                             "feats_skipped", "feats_status", "feats_bcast_tensors", "msa_host_mode", "msa_host_where")}
    if msa_host and fault != "mismatch":                                                   # the engine's H2D of the (received) example, then the model entry's placement statement, on EVERY rank
        with H2D_LOCK:
            moved = STUBS["h2d"]().to_device(ex)
            held = rowpair.CTX["msa_host"]
            rowpair.CTX["msa_host"] = None
        out["h2d"] = {"shape": list(held["shape"]), "placeholders": list(held["placeholders"]), "parked": list(held["parked"]),
                      "model_ph": list(moved["feats"]["msa_stack"].shape), "held_is_received": held["feats"]["msa_stack"] is ms}
        rowpair._msa_host_entry(dict(moved["feats"]), torch.device("cpu"), held=held)       # rank0: the placement check; all: rank 0's stack streamed into this rank's host copy (a collective)
        filled = held["feats"]["msa_stack"]
        out["entry"] = {"shape": list(filled.shape), "sum": float(filled.double().sum()) if filled.numel() else None, "where": MH.where(filled)}
    return out


@needs_mp
@pytest.mark.parametrize("msa_host,fault,n_items", [("rank0", "", 2), ("all", "", 1), ("", "", 1), ("rank0", "raise", 1), ("rank0", "differ", 1), ("rank0", "mismatch", 1)])
def test_features_from_rank0_on_two_ranks(msa_host, fault, n_items, monkeypatch, capfd):
    """B-skip (data_form=rank0_bcast) on 2 threaded ranks (``opt_core.testing.run_ranks``: one process, the thread comm carries the store
    verbs and every collective): rank 0 alone runs the Transform pipeline; rank 1 receives its featurised example — every tensor, the numpy /
    str / list leaves — and both prove they hold rank 0's bytes and say so (one ``FEATS`` line per item and rank); with ROWPAIR_MSA_HOST set the
    raw MSA stack does not travel (a zero-row placeholder carrying n_recycle stands in, the receiver's H2D seam holds it as such, and under
    ``all`` the model entry fills it with rank 0's stack); two items advance the rendezvous ordinal alike on both ranks; a featurisation that
    raised on rank 0, and a receiver whose digest differs, are ONE refusal on both ranks; a receiver that parsed another item is refused by
    name. (The RNG carry is the core's: ``feats_rng=carried``; threads share one generator, so its effect is proven by the core's process tests
    and the trunk's per-recycle RNG guard on the GPUs.)"""
    testing = _core.load("testing")
    rd = _core.load("mem.rowpair.rankdata")
    if msa_host:
        monkeypatch.setenv("ROWPAIR_MSA_HOST", msa_host)
    else:
        monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)
    torch.manual_seed(0)
    FEATS_CALLS["n"] = 0
    rowpair.CTX.pop("featurise_calls", None)
    rowpair.CTX["calls"].pop("featurise", None)
    rowpair.CTX["msa_host"] = None
    _core.load("mem.rowpair.evidence").reset_schedule()                                 # the process-wide schedule record starts empty for this case
    if fault == "differ":
        real = rd.digest_features
        monkeypatch.setattr(rd, "digest_features", lambda feats, exclude=(): real(feats, exclude)._replace(digest="f" * 64) if _dist_rank() == 1 else real(feats, exclude))
    try:
        outs = testing.run_ranks(2, _featurise_rank, msa_host, fault, n_items, threading.Barrier(2), timeout_s=120)
        book = dict(rowpair.CTX.get("featurise_calls") or {})
    finally:
        restored = rowpair.uninstall(destroy_group=False) if rowpair.CTX["installed"] else []
    err = capfd.readouterr().err
    feats_lines = re.findall(r"\[rosettafold3-opt\] FEATS item=\d+ rank=\d+ example=\S+ data_form=\S+ feats_digest=\S+ feats_items=\d+ feats_digest_excludes=\S+ feats_unhashed=\S+ feats_ranks_equal=\w+", err)
    o0, o1 = outs
    assert (o0["rank"], o1["rank"]) == (0, 1) and o0["data_form_at_install"] is None and o0["site_flag"] is True   # data_form is recorded when the seam ENGAGES, not on the install
    assert any(n.endswith("_FeatBase._construct_pipeline") for n in restored) and "featurise_calls" not in rowpair.CTX
    if fault == "raise":
        assert FEATS_CALLS["n"] == 1 and feats_lines == []
        assert all(o["refusal"] and "feats_rank0_failed" in o["refusal"] and "ValueError" in o["refusal"] and "no CCD entry" in o["refusal"] for o in outs), [o["refusal"] for o in outs]
        return
    if fault == "differ":
        assert all(o["refusal"] and "feats_ranks_differ" in o["refusal"] for o in outs), [o["refusal"] for o in outs]
        assert len(feats_lines) == 2 and all("feats_ranks_equal=no" in ln and "data_form=rank0_bcast" in ln for ln in feats_lines), feats_lines   # every rank's line names its digest
        return
    if fault == "mismatch":                                                                 # rank 1 parsed 'zz', rank 0 featurised and named 'q1': refused on rank 1 by name, after the digest gate
        assert o0["refusal"] is None and o1["refusal"] and "feats_item_mismatch" in o1["refusal"] and "'zz'" in o1["refusal"] and "'q1'" in o1["refusal"], o1["refusal"]
        return
    assert o0["refusal"] is None and o1["refusal"] is None and FEATS_CALLS["n"] == n_items   # rank 1 never featurised
    assert book == {0: n_items, 1: n_items}                                                 # the rendezvous ordinal advanced alike on both ranks
    assert o0["x"] == o1["x"] == (torch.arange(6.0).reshape(2, 3)).tolist()                # rank 0's bytes (+0) on both ranks
    assert o0["noise"] == o1["noise"] and o0["flag"] == o1["flag"] == [True, [True, False]]
    assert o0["devices"] == o1["devices"] == ["cpu"]                                         # host tensors arrive on the host (the engine's H2D moves them, as on rank 0)
    assert o0["coords"] == o1["coords"] and o0["extra"] == o1["extra"] == {"name": "q", "chains": ["A", "B"], "n": 1}
    assert o0["keys"] == o1["keys"] == ["coords", "example_id", "extra", "feats"] and o0["example_id"] == o1["example_id"] == f"q{n_items}"
    for o in outs:                                                                          # the schedule record is the process's: both ranks wrote the same words
        sc = o["sched"]
        assert sc["data_form"] == "rank0_bcast" and sc["feats_ranks_equal"] == "yes" and len(sc["feats_digest"]) == 16 and sc["feats_src"] == 0
        assert sc["feats_rng"] == "carried" and sc["feats_status"].split()[0] == "ok" and sc["feats_items"] == n_items and sc["feats_unhashed"] == "none"
    assert o0["sched"]["feats_digest"] == o1["sched"]["feats_digest"]
    assert rowpair.CTX["calls"]["featurise"] == {"sharded": n_items, "replicated": n_items} # per item 1s on rank 0 + 1r on rank 1 (one census dict in this process)
    assert len(feats_lines) == 2 * n_items, feats_lines                                     # one FEATS line per item and rank, the digest and the gate's word on it
    for k in range(n_items):
        for r in (0, 1):
            mine = [ln for ln in feats_lines if f" item={k + 1} rank={r} example=q{k + 1} " in ln]
            assert len(mine) == 1 and f"data_form=rank0_bcast feats_digest={o0['sched']['feats_digest'] if k + 1 == n_items else ''}" in mine[0] and "feats_ranks_equal=yes" in mine[0], mine
    n_rec, S, I, c = FEATS_DIMS
    m0, m1 = o0["msa"], o1["msa"]
    assert m0["shape"] == [n_rec, S, I, c] and m0["where"] != "placeholder"
    if msa_host:                                                                            # ROWPAIR_MSA_HOST set (rank0 | all): the stack did not travel — a zero-row placeholder carrying n_recycle
        assert m1 == {"shape": [0, S, I, c], "where": "placeholder", "rows": n_rec, "sum": None}
        assert o0["sched"]["feats_skipped"] == "feats.msa_stack" == o0["sched"]["feats_digest_excludes"]
        assert o1["sched"]["msa_host_mode"] == msa_host and o1["sched"]["msa_host_where"] == "msa_stack:placeholder"   # the receiver's placement words name the mode in force
        h0, h1 = o0["h2d"], o1["h2d"]
        assert h0["shape"] == h1["shape"] == [n_rec, S, I, c] and h0["model_ph"] == h1["model_ph"] == [n_rec, S, 0, c]   # n_cycle = msa_stack.shape[0] reads n_recycle on both ranks
        assert h0["parked"] == ["msa_stack"] and h1 == {"shape": [n_rec, S, I, c], "placeholders": ["msa_stack"], "parked": [], "model_ph": [n_rec, S, 0, c], "held_is_received": True}
        e0, e1 = o0["entry"], o1["entry"]
        assert e0["shape"] == [n_rec, S, I, c] and e0["sum"] == m0["sum"]
        if msa_host == "rank0":                                                             # rank 0 alone holds the stack; the receiver keeps its placeholder (rank 0 serves the rows per recycle)
            assert e1 == {"shape": [0, S, I, c], "sum": None, "where": "placeholder"}
        else:                                                                               # all: the model entry streamed rank 0's stack into the receiver's host copy
            assert e1["shape"] == [n_rec, S, I, c] and e1["sum"] == m0["sum"] and e1["where"] != "placeholder"
    else:                                                                                   # lever off: the stack travels like every other feature (device-resident downstream, as stock)
        assert m1["shape"] == m0["shape"] and m1["sum"] == m0["sum"] and m1["where"] != "placeholder"
        assert o0["sched"]["feats_skipped"] == "none" == o0["sched"]["feats_digest_excludes"]


@needs_mp
def test_received_msa_placeholder_through_the_h2d_seam(monkeypatch):
    """One process, no group: rank 0's wire form hoists the raw MSA stack to the top-level key ``feats.msa_stack`` (the example untouched); a
    receiver's example built from the wire minus that key + its meta holds the zero-row placeholder carrying n_recycle; the H2D seam keeps
    it as the item's held stack (nothing parked), records the TRUE shape and hands the model its [n_recycle, S, 0, c] placeholder."""
    MH = _core.load("mem.rowpair.msa_host")
    n_rec, S, I, c = FEATS_DIMS
    stack = torch.randn(n_rec, S, I, c)
    example = {"example_id": "q", "feats": {"msa_stack": stack, "x": torch.ones(2)}, "extra": [1, 2]}
    wire = rowpair._feats_wire(example)
    key = rowpair._feats_wire_key()
    assert key == "feats.msa_stack" and wire[key] is stack and "msa_stack" not in wire["feats"] and example["feats"]["msa_stack"] is stack
    assert rowpair._feats_unwire(dict(wire), {})["feats"]["msa_stack"] is stack             # it travelled (all / off): back under feats as received
    received = {k: v for k, v in wire.items() if k != key}
    ex = rowpair._feats_unwire(received, {key: {"shape": (n_rec, S, I, c), "dtype": "float32", "dev": "cpu"}})
    ph = ex["feats"]["msa_stack"]
    assert tuple(ph.shape) == (0, S, I, c) and ph.dtype == torch.float32 and MH.where(ph) == "placeholder" and MH.parked_rows(ph, 0) == n_rec
    assert key not in ex and torch.equal(ex["feats"]["x"], torch.ones(2)) and ex["extra"] == [1, 2]
    saved = {k: rowpair.CTX.get(k) for k in ("installed", "msa_host")}
    try:
        rowpair.CTX["installed"], rowpair.CTX["msa_host"] = True, None
        monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0")
        fab = STUBS["h2d"]()
        out = rowpair._h2d_to_device(fab, type(fab).to_device, ex)
        held = rowpair.CTX["msa_host"]
        assert held["shape"] == (n_rec, S, I, c) and held["placeholders"] == ["msa_stack"] and held["parked"] == [] and held["feats"]["msa_stack"] is ph
        assert tuple(out["feats"]["msa_stack"].shape) == (n_rec, S, 0, c)                   # n_cycle = msa_stack.shape[0] reads n_recycle on a receiver too
        rowpair._msa_host_entry(dict(out["feats"]), torch.device("cpu"))
    finally:
        rowpair.CTX.update(saved)


@needs_mp
@pytest.mark.parametrize("P,perturb,msa_host,ckpt", [(2, False, "", True), (2, True, "rank0", True), (3, False, "rank0", False), (2, True, "all", False)])
@_stubs.CORE_GLOO_DENIED
def test_mp_sharded_equals_dense(P, perturb, msa_host, ckpt, tmp_path):
    out = _mp(P, _entry_e2e, perturb, msa_host, str(tmp_path / "ckpt") if ckpt else None)
    print("RESULT P=%d perturb=%s msa_host=%r" % (P, perturb, msa_host), {k: ("%.1e" % v["max_abs"], v["bitwise"]) for k, v in out["cmp"].items()})
    print("MSA", out["msa"])
    print("CENSUS", out["state"]["calls"], out["state"]["sync"], out["lay"], out["state"]["schedule"])
    assert out["rank"] == 0 and out["P"] == P and out["census"]["installed"] and out["census"]["sharding"] == "rowpair"
    assert set(out["restored"]) == set(out["census"]["patched"])
    assert any(n.endswith("levers.PAIR_WINDOWS") for n in out["census"]["patched"]) and "Recycler.forward" in out["census"]["patched"]
    assert out["shard_shape"][0] == out["lay"]["R"] < N_TOK and out["lay"]["N"] == N_TOK and out["lay"]["P"] == P
    bounds = out["lay"]["bounds"]
    assert tuple(bounds[0])[0] == 0 and tuple(bounds[-1])[-1] == N_TOK and len(bounds) == P          # [(r0, r1)] per rank, tiling 0:N
    assert all(tuple(bounds[i])[1] == tuple(bounds[i + 1])[0] for i in range(P - 1))
    for k, c in out["cmp"].items():
        assert c["max_abs"] <= TOL, (k, c)
    calls = out["state"]["calls"]
    for site in ("entry", "pair_init", "recycler", "template", "msa_module", "distogram", "conf_head", "diff_cond", "dit", "pair_windows"):
        assert calls.get(site, {}).get("sharded", 0) > 0, (site, calls)
    assert calls["recycler"]["sharded"] >= 2 and calls["recycler"].get("replicated", 0) == 0          # every cycle on the shard, never dense
    assert out["missing"] == [], out["missing"]
    for k in ("Z_II", "S_I", "distogram", "Z_c", "a1", "win1", "c_pae_logits", "c_pde_logits", "c_plddt_logits", "g_pae_logits", "g_plddt_logits"):
        assert k in out["cmp"], k
    assert calls["conf_head"]["sharded"] == 3 and calls["dit"]["sharded"] == 2 and calls["diff_cond"]["sharded"] == 2      # 2 heads + the probe
    assert calls["conf_probe"]["sharded"] == 1 and out["probe"]["pae_none"] and out["probe"]["plddt"]["max_abs"] <= TOL, out["probe"]
    assert out["probe"]["rank"][0] == out["probe"]["rank"][1] == 2, ("probe pLDDT logits rank (sharded, stock)", out["probe"]["rank"])   # RF3.forward's probe: [I, atoms*bins]
    assert out["probe"]["early_stop"][0][0] == out["probe"]["early_stop"][1][0], out["probe"]["early_stop"]                             # the early-stop decision is stock's
    assert abs(out["probe"]["early_stop"][0][1] - out["probe"]["early_stop"][1][1]) <= 1e-9, out["probe"]["early_stop"]
    assert calls["denoiser"]["sharded"] == 2 and out["state"]["schedule"].get("diff_noise") == "bcast_rank0_state", out["state"]["schedule"].get("diff_noise")
    assert int(out["state"]["schedule"].get("diff_state_inplace", 0)) == 1
    feats = out["state"]["schedule"].get("entry_pair_feats", "")
    assert "token_bonds" in feats and "distogram_condition" in feats and "has_distogram_condition" in feats, feats     # N x N inputs live as rows
    sched_t = out["state"]["schedule"]                          # the template census, cumulative over rank 0's items: those that reached the template embedder are the main item,
    n_templ = 1 + (2 if ckpt else 0) + 2 + 2 + 1                  # the checkpoint leg's A and C (B and the allowed resume fast-forward every recycle, the refused one loads nothing),
    assert sched_t.get("templ_form") == "row_born" and int(sched_t.get("templ_items", -1)) == n_templ, (n_templ, sched_t)   # the two transpose and the two park forms, the untemplated item last — each counted once over its N_CYC recycles
    assert sched_t.get("templ_real") == "%d/%d" % (n_templ - 1, n_templ) and out["templ_fill"][0] > 0 and out["templ_fill"][1] == 0, (sched_t, out["templ_fill"])   # all templated but the last
    assert int(sched_t.get("templ_fill_max", -1)) == out["templ_fill"][0] and "templ_fill" not in sched_t, (sched_t, out["templ_fill"])          # the most filled tokens of one item; no per-item word
    assert int(sched_t.get("templ_tokens", -1)) == N_TOK and int(sched_t.get("templ_row_blocks", 0)) >= 1, sched_t                                    # rows came from the precursors
    b0 = tuple(out["lay"]["bounds"][0])
    assert out["f_rows"] == [b0[1] - b0[0], N_TOK], out["f_rows"]                          # rank 0's token_bonds are its rows only
    assert out["engine_rows"][0][0] == b0[1] - b0[0] and out["engine_rows"][1][0] == N_TOK, out["engine_rows"]   # the engine twin -> rows; other keys untouched
    assert "distogram_condition" in out["state"]["schedule"].get("engine_pair_feats", ""), out["state"]["schedule"].get("engine_pair_feats")
    assert "nxn_whole_resident" in out["state"]["schedule"]
    assert float(out["state"]["schedule"].get("diff_rank_spread_A", -1)) == 0.0, out["state"]["schedule"].get("diff_rank_spread_A")   # X was made rank 0's at entry
    sched = out["state"]["schedule"]
    assert sched.get("msa_m") == "replicated" and int(sched.get("msa_blocks", 0)) >= 2
    assert sched.get("park_z_init") == "device"                  # ROWPAIR_PARK_ZINIT: the z_init shard went through trunk.ShardPark (a CPU shard stays resident; parked = host_pinned on a GPU)
    msa = out["msa"]                                              # ROWPAIR_MSA_HOST (rank 0's view): the raw MSA stack held on the host, the model read the zero-token placeholder,
    if msa_host:                                                  # the cycle's rows moved per recycle (rank0: broadcast to ranks > 0) — and sharded == dense above holds with it
        assert sched.get("msa_host") == msa_host and sched.get("msa_host_site") == "h2d" and sched.get("msa_host_stack") == "%dx%dx%dx%d:float64" % (N_CYC, N_SEQ, N_TOK, C_M)
        assert sched.get("msa_host_mode") == msa_host and str(sched.get("msa_host_rows", "")).startswith("%s:%dx%dx%d:" % (msa_host, N_SEQ, N_TOK, C_M))
        assert msa["entry_shape"] == [N_CYC, N_SEQ, 0, C_M]                             # what the model's own statements read (n_cycle = shape[0] intact)
        assert msa["held"] == {"mode": msa_host, "shape": [N_CYC, N_SEQ, N_TOK, C_M], "parked": ["msa_stack"], "placeholders": []}   # rank 0 keeps the host copy
        assert msa["held_after"] is False                                                # dropped with the item
        assert msa["seam_refused"] and ("ROWPAIR_MSA_HOST=%s but the raw MSA stack reached the model" % msa_host) in msa["seam_refused"], msa["seam_refused"]   # a whole stack under the lever: refused by name, no fallback
        assert int(out["state"]["calls"].get("h2d", {}).get("sharded", 0)) == 0          # the seam is a data statement, not a counted sharded site
    else:
        assert sched.get("msa_host") == "off" and msa["entry_shape"] == [N_CYC, N_SEQ, N_TOK, C_M] and msa["held"] is None
    pk = out["park"]                                              # ROWPAIR_CONF_PARK_ZTRUNK=1 + ROWPAIR_FREE_ZTRUNK=1 (exported): parked at the roll-out entry, the diffusion
    assert pk["bitwise"] and all(pk["bitwise"].values()) and {"S_I", "Z_c", "a1", "c_pae_logits", "c_plddt_logits", "g_plddt_logits"} <= set(pk["bitwise"]), pk   # conditioning
    assert str(pk["entry"][1]).startswith("parked:") and pk["entry"][0] == "resident", pk["entry"]          # + every confidence pass served from the park; bitwise vs resident
    assert pk["words"][0] == ["resident", "resident"] and all(str(w).startswith("parked") for w in pk["words"][1]) and len(pk["words"][1]) == 2, pk["words"]
    assert pk["storage"][1] == 0 < pk["storage"][0], pk["storage"]                                          # the parked shard's device storage: released, and left released
    tr = out["transpose"]                                         # ROWPAIR_TRANSPOSE_INPLACE=1 (exported on the line): in-place vs out-of-place transpose, bitwise in every output
    assert tr["bitwise"] and all(tr["bitwise"].values()) and {"Z_II", "S_I", "Z_c"} <= set(tr["bitwise"]), tr
    assert str(tr["words"][1]).startswith("inplace") and not str(tr["words"][0]).startswith("inplace"), tr["words"]
    ck = out["ckpt"]                                              # ROWPAIR_CKPT_DIR (core TrunkCheckpointer bound at Recycler.forward): written, resumed bitwise, refused by name
    if ckpt:
        print("CKPT", ck)
        assert ck["ckpt"] == "on" and ck["wrote"] == "trunk_final" and ck["query"] == "q_7" and ck["tags"] == ["cycle_000", "trunk_final"], ck
        assert ck["resume_B"] == "trunk_final" and int(ck["ff_B"]) == N_CYC and ck["B_resumed_calls"] == N_CYC, ck
        assert all(ck["B_bitwise"].values()) and set(ck["B_bitwise"]) >= {"Z_II", "S_I", "distogram", "Z_c", "a1", "c_plddt_logits", "c_pae_logits"}, ck["B_bitwise"]
        assert ck["resume_C"] == "cycle_000" and int(ck["ff_C"]) == 1 and all(ck["C_bitwise"].values()), ck
        assert ck["differ_refused"] and "resume_refused:features_differ" in ck["differ_refused"], ck["differ_refused"]
        assert ck["differ_allowed"] == "trunk_final", ck["differ_allowed"]
    # the confidence stage's trunk-shard plan (heads.ZTrunkPlan): D per-sample passes of the item's plan + the probe's
    assert out["plan_words"][:3] == ["resident", "resident", "resident"], out["plan_words"]   # own one-pass plan, both levers unset here -> resident (the e2e's
    assert sched.get("conf_ztrunk_probe") == "resident", sched                                # forms are proven bitwise in test_tp_conf_cpu.py::test_ztrunk_plan_forms)
    assert out["noise_shape"] == [3, 10, 3] and calls["noise_init"]["sharded"] >= 1 and calls["noise_predraw"]["sharded"] == 1 and out["n_predraw"] == 2
    assert bool(out["train_refused"]) and "training" in out["train_refused"], out["train_refused"]
    assert bool(out["cyclic_refused"]) and "cyclic" in out["cyclic_refused"], out["cyclic_refused"]
    sync = out["state"]["sync"]
    assert sync["entry"]["calls"] >= 1 and sync["s_inputs"]["calls"] >= 1 and sync["x_pred"]["calls"] >= 1
    assert out["noise_refused"] is False, out["noise_refused"]                                # bcast: every rank holds rank 0's draw, det or not
    assert out["state"]["schedule"].get("diff_noise_sync") == "bcast", out["state"]["schedule"].get("diff_noise_sync")
    assert out["state"]["schedule"].get("noise_sync") == "bcast" and int(out["state"]["schedule"].get("det", -1)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rs", "-x"]))


# ================================================================================ P>1 triangle kernels (opt_core "tpx"): the wiring at rest
import inspect                                                    # noqa: E402

from .. import stack                                              # noqa: E402

def _old_vanilla_rows(m, pair, bias):
    """The kit's row-batch statement of RF3's TriangleAttention._forward_vanilla as it read before the attention-core wiring (the reference
    the refactored statement must equal byte for byte)."""
    B, r, N = int(pair.shape[0]), int(pair.shape[1]), int(pair.shape[2])
    gate = torch.sigmoid(m.to_g(pair))
    query = m.to_q(pair).reshape(B, r, N, m.h, -1)
    key = m.to_k(pair).reshape(B, r, N, m.h, -1)
    value = m.to_v(pair).reshape(B, r, N, m.h, -1)
    query = query * m.scaling
    attn = torch.einsum("bijhd,bikhd->bijkh", query, key)
    attn = attn + bias.unsqueeze(1).expand(-1, r, -1, -1, -1)
    attn = torch.nn.functional.softmax(attn, dim=-2)
    out = torch.einsum("bijkh,bikhd->bijhd", attn, value).reshape(B, r, N, -1)
    return gate * out


def test_tp_kernel_words():
    """The row-sharded line's fused triangle kernels: two words from the core's one vocabulary (flash_triattn | cueq | torch; fpf_v4 | torch),
    one size gate; nothing is recorded until a pair block is bound (the rowpair LEVER line's tp_* fields appear only then)."""
    assert rowpair.TRIATT_KERNELS == ("torch", "flash_triattn", "cueq") and rowpair.TP_TRIATT_KERNEL == "flash_triattn"
    assert rowpair.TRIMUL_KERNELS == ("torch", "fpf_v4") and rowpair.TP_TRIMUL_KERNELS == "fpf_v4"
    assert rowpair.TP_KERNEL_MIN_TOKENS == 2048
    triatt = _core.load("mem.rowpair.triatt") if HAVE_ROWPAIR else None
    if triatt is not None:
        assert rowpair.TRIATT_KERNELS == tuple(sorted(triatt.CORE_KERNELS, key=rowpair.TRIATT_KERNELS.index))     # the core's words, exactly
    assert rowpair.TP_KERNELS == {"triatt": None, "trimul": None}                      # nothing bound in this process: the rowpair LEVER line carries no tp_* field
    src = inspect.getsource(rowpair.lever_fields)
    assert 'pairs += [(f"tp_{k}", w) for k, w in sorted(TP_KERNELS.items()) if w is not None]' in src


@pytest.mark.skipif(not HAVE_TORCH, reason="torch absent")
@pytest.mark.parametrize("starting", [True, False])
def test_triatt_stock_core_is_the_vanilla_statement_byte_for_byte(starting, monkeypatch):
    """On RF3's einsum route the attention core's fallback (_triatt_stock_core over the [B,r,H,N,D] views, q unscaled) == _forward_vanilla's
    statement (_triatt_vanilla_rows == the statement as it read before the wiring), bit for bit, starting node (contiguous bias) and ending node
    (the transposed bias view the driver hands in); _triatt_rows with a pass-through core likewise."""
    torch.manual_seed(3)
    N, r, C, H, D = 12, 5, 16, 4, 8
    m = TriangleAttention(C, H, D, starting).double()
    monkeypatch.setattr(rowpair, "_TP_STATE", {"triatt_core": {}, "emitted": set()})
    assert rowpair._tp_triatt_register(m, route_cueq=False) is None and rowpair._TP_STATE["triatt_stock"] == {"cueq": False, "scaling": {D: m.scaling}}
    pair = torch.randn(1, r, N, C, dtype=torch.float64)
    tb_full = torch.randn(N, N, H, dtype=torch.float64)
    bias = (tb_full if starting else tb_full.transpose(0, 1)).unsqueeze(0)
    with torch.no_grad():
        ref = _old_vanilla_rows(m, pair, bias)
        assert torch.equal(rowpair._triatt_vanilla_rows(m, pair, bias), ref)
        q, k, v = rowpair._triatt_qkv(m, pair)
        assert q.shape == (1, r, H, N, D) and not q.is_contiguous() and torch.equal(q.permute(0, 1, 3, 2, 4), m.to_q(pair).reshape(1, r, N, H, D))   # views, q unscaled
        tbv = rowpair._triatt_bias_view(bias)
        assert tbv.shape == (1, 1, H, N, N) and tbv.squeeze(1).permute(0, 2, 3, 1).data_ptr() == bias.data_ptr()
        o = rowpair._triatt_stock_core(q, k, v, (tbv,))
        assert o.shape == (1, r, H, N, D)
        assert torch.equal(torch.sigmoid(m.to_g(pair)) * o.permute(0, 1, 3, 2, 4).reshape(1, r, N, -1), ref)
        core_calls = []
        fake_core = lambda q_, k_, v_, b_: core_calls.append((q_.shape, len(b_), b_[0].shape)) or rowpair._triatt_stock_core(q_, k_, v_, b_)
        monkeypatch.setattr(rowpair, "_rp", lambda name=None: types.SimpleNamespace(torch=torch, attend_query_blocks=lambda core, q_, k_, v_, b_, qb: core(q_, k_, v_, b_)))
        assert torch.equal(rowpair._triatt_rows(m, pair, bias, fake_core), ref)
        assert core_calls == [((1, r, H, N, D), 1, (1, 1, H, N, N))]
        # two modules of one process: a second head width registers its own scaling; a route / scaling disagreement is named
        m2 = TriangleAttention(C, H, D, starting).double(); m2.scaling = 0.5
        assert rowpair._tp_triatt_register(m2, route_cueq=False) == "mixed_scaling"
        assert rowpair._tp_triatt_register(m, route_cueq=True) == "mixed_routes"


@pytest.mark.skipif(not HAVE_TORCH, reason="torch absent")
def test_triatt_stock_core_on_the_cuequivariance_route_is_the_modules_call(monkeypatch):
    """On the cuEquivariance route the fallback IS `cuequivariance_torch.triangle_attention(q, k, v, bias=[B,1,H,N,N], scale=m.scaling)` on the
    [B,r,H,N,D] views — the call TriangleAttention._forward_cuequivariance makes (attention.py) — so below the size gate the op still runs on
    cuEquivariance's kernel: a stand-in cuequivariance_torch records the call."""
    torch.manual_seed(4)
    N, r, C, H, D = 10, 3, 16, 4, 8
    m = TriangleAttention(C, H, D, True)
    calls = []
    def fake_triangle_attention(q, k, v, bias=None, scale=None, **kw):
        calls.append(dict(q=tuple(q.shape), bias=tuple(bias.shape), scale=scale, kw=kw, contiguous=q.is_contiguous(), bias_dtype=bias.dtype))
        att = torch.softmax(torch.einsum("brhid,brhjd->brhij", q * scale, k) + bias, dim=-1)          # any deterministic function of the arguments
        return torch.einsum("brhij,brhjd->brhid", att, v)
    monkeypatch.setitem(sys.modules, "cuequivariance_torch", types.SimpleNamespace(triangle_attention=fake_triangle_attention))
    monkeypatch.setattr(rowpair, "_TP_STATE", {"triatt_core": {}, "emitted": set()})
    assert rowpair._tp_triatt_register(m, route_cueq=True) is None
    pair = torch.randn(1, r, N, C); bias = torch.randn(1, N, N, H)
    with torch.no_grad():
        q, k, v = rowpair._triatt_qkv(m, pair); tbv = rowpair._triatt_bias_view(bias)
        o = rowpair._triatt_stock_core(q, k, v, (tbv,))
        want = fake_triangle_attention(q, k, v, bias=tbv, scale=m.scaling)
    assert torch.equal(o, want) and len(calls) == 2
    c = calls[0]
    assert c["q"] == (1, r, H, N, D) and c["bias"] == (1, 1, H, N, N) and c["scale"] == m.scaling and c["kw"] == {} and not c["contiguous"]


@pytest.mark.skipif(not HAVE_TORCH, reason="torch absent")
def test_tp_trimul_weights_are_the_fpf_addons_p1_mapping(monkeypatch):
    """One mapping of RF3's TriangleMultiplication tensors onto opt_core.trimul.WEIGHT_KEYS: rowpair._tp_trimul_weights == what the FPF add-on's
    P=1 adapter hands fpf_trimul_v4 (rf3fpf/fpf_rf3_adapter.py _weights_fast), key for key, tensor for tensor."""
    import importlib.util
    from opt_core import trimul as core_trimul
    m = TriangleMultiplication(16, 8, "outgoing")
    w = rowpair._tp_trimul_weights(m)
    assert tuple(w) == core_trimul.WEIGHT_KEYS
    got = {}
    fake = types.ModuleType("fpf_trimul_v4"); fake.generic = types.SimpleNamespace(pack_weights=lambda **kw: got.update(kw) or "packed")
    monkeypatch.setitem(sys.modules, "fpf_trimul_v4", fake); monkeypatch.setitem(sys.modules, "fpf_trimul_v4.generic", fake.generic)
    path = os.path.join(stack.fpf_home(), "rf3fpf", "fpf_rf3_adapter.py")
    src = open(path).read()
    body = src[src.index("def _weights_fast(m):"):src.index("def _fast_forward")]
    ns = {"os": os}
    exec(body, ns)                                                                          # the adapter's own function, verbatim (its module imports rf3 at top level)
    assert ns["_weights_fast"](m) == "packed"
    assert set(got) - {"cache_owner", "cache_key"} == set(w)
    for k_ in w:
        assert torch.equal(got[k_], w[k_]) and got[k_].shape == w[k_].shape, k_


@pytest.mark.skipif(not HAVE_TORCH, reason="torch absent")
def test_fused_words_without_the_core_fall_to_torch_by_name(monkeypatch):
    """A fused word with a pinned core that lacks the tpx modules: the torch statements serve and the rowpair LEVER line says why
    (tp_triatt=torch:core_missing(...), tp_trimul=torch:core_missing(...)); the vanilla TriMul route keeps torch by name too."""
    monkeypatch.setattr(rowpair, "TP_KERNELS", {"triatt": None, "trimul": None})
    monkeypatch.setattr(rowpair, "_TP_STATE", {"triatt_core": {}, "emitted": set()})
    monkeypatch.setattr(rowpair, "_tp_core_module", lambda name: None)                    # an older core: no attention_core / trimul_fused
    monkeypatch.setattr(rowpair, "TP_TRIATT_KERNEL", "flash_triattn")
    monkeypatch.setattr(rowpair, "TP_TRIMUL_KERNELS", "fpf_v4")
    ta = TriangleAttention(16, 4, 8, True)
    assert rowpair._tp_triatt_core(ta, False) is None and rowpair.TP_KERNELS["triatt"] == "torch:core_missing(opt_core.mem.rowpair.triatt.attention_core)"
    m = TriangleMultiplication(16, 8, "outgoing")
    assert rowpair._tp_trimul_fused(m, "stock-fns", vanilla_scale=False) is None
    assert rowpair.TP_KERNELS["trimul"] == "torch:core_missing(opt_core.mem.rowpair.trimul_fused)"
    # the core present but RF3 on its vanilla TriMul route (cuEquivariance off: the b operand carries 1/N): torch by name
    built = []
    fake_rf = types.SimpleNamespace(fused_trimul_fns=lambda weights, stock_fns, **kw: built.append((sorted(weights), stock_fns, kw)) or "fused-fns",
                                    emit_line=lambda tag: None)
    monkeypatch.setattr(rowpair, "_tp_core_module", lambda name: fake_rf if name == "trimul_fused" else None)
    assert rowpair._tp_trimul_fused(m, "stock-fns", vanilla_scale=True) is None and rowpair.TP_KERNELS["trimul"] == "torch:vanilla_route" and not built
    assert rowpair._tp_trimul_fused(m, "stock-fns", vanilla_scale=False) == "fused-fns" and rowpair.TP_KERNELS["trimul"] == "fpf_v4"
    keys, stock_fns, kw = built[0]
    assert stock_fns == "stock-fns" and kw == {"eps": 1e-5, "cells": None, "ledger": None, "min_tokens": 2048} and "trimul" in rowpair._TP_STATE["emitted"]
    # unknown words refuse by name
    monkeypatch.setattr(rowpair, "TP_TRIMUL_KERNELS", "t9")
    if HAVE_ROWPAIR:
        with pytest.raises(Exception, match="TP_TRIMUL_KERNELS='t9'"):
            rowpair._tp_trimul_fused(m, "stock-fns", vanilla_scale=False)


@pytest.mark.skipif(not (HAVE_TORCH and HAVE_ROWPAIR), reason="torch + opt_core.mem.rowpair needed")
def test_torch_words_leave_the_fns_as_they_were(monkeypatch):
    """With both words `torch` (the default) _trimul_fns is the plain torch TriMulFns (no fused hooks) and the attention row batch never asks
    the core for an attention core: today's code path, nothing recorded."""
    monkeypatch.setattr(rowpair, "TP_KERNELS", {"triatt": None, "trimul": None})
    monkeypatch.setattr(rowpair, "_TP_STATE", {"triatt_core": {}, "emitted": set()})
    monkeypatch.setattr(rowpair, "TP_TRIATT_KERNEL", "torch"); monkeypatch.setattr(rowpair, "TP_TRIMUL_KERNELS", "torch")
    asked = []
    monkeypatch.setattr(rowpair, "_tp_core_module", lambda name: asked.append(name))
    m = TriangleMultiplication(16, 8, "outgoing")
    fns = rowpair._trimul_fns(m, 12)
    trimul = _core.load("mem.rowpair.trimul")
    assert isinstance(fns, trimul.TriMulFns) and not hasattr(fns, "proj_into") and rowpair._tp_triatt_core(TriangleAttention(16, 4, 8, True), False) is None
    assert asked == [] and rowpair.TP_KERNELS == {"triatt": None, "trimul": None}


@pytest.mark.skipif(not (HAVE_TORCH and HAVE_ROWPAIR), reason="torch + opt_core.mem.rowpair needed")
def test_the_core_builds_the_fused_fns_over_todays_callables(monkeypatch):
    """With the pinned core's trimul_fused / attention_core: _trimul_fns hands pairstack a FusedTriMulFns whose proj / out / gate ARE the
    module's torch statements (the fallback of every declined unit) with the kit's size gate, and the attention core is one callable per
    process built over _triatt_stock_core; both record their word for the rowpair LEVER line."""
    RF = _core.load("mem.rowpair.trimul_fused"); RA = _core.load("mem.rowpair.triatt")
    monkeypatch.setattr(rowpair, "TP_KERNELS", {"triatt": None, "trimul": None})
    monkeypatch.setattr(rowpair, "_TP_STATE", {"triatt_core": {}, "emitted": set()})
    monkeypatch.setattr(rowpair, "_tp_emit_at_exit", lambda kind, mod: rowpair._TP_STATE["emitted"].add(kind))   # no atexit prints from a test
    monkeypatch.setattr(rowpair, "_cueq_on", lambda: True)                               # the engine's cuEquivariance route (the stand-in foundry module says vanilla)
    m = TriangleMultiplication(16, 8, "outgoing")
    fns = rowpair._trimul_fns(m, 12)
    assert isinstance(fns, RF.FusedTriMulFns) and fns.min_tokens == rowpair.TP_KERNEL_MIN_TOKENS == 2048 and fns.C_h == 8 and fns.C_z == 16
    assert rowpair.TP_KERNELS["trimul"] == "fpf_v4" and "trimul" in rowpair._TP_STATE["emitted"]
    z = torch.randn(3, 12, 16); out_a = fns.proj(z, None, True)
    x = m.norm_in(z); ref = (torch.sigmoid(m.g_in(x))[..., :8] * m.p_in(x)[..., :8])
    assert torch.equal(out_a, ref)                                                     # the fallback IS the module's statement
    ta = TriangleAttention(16, 4, 8, True)
    core = rowpair._tp_triatt_core(ta, True)
    assert callable(core) and rowpair._tp_triatt_core() is core and rowpair.TP_KERNELS["triatt"] == "flash_triattn" and "triatt" in rowpair._TP_STATE["emitted"]
    assert rowpair._TP_STATE["triatt_stock"] == {"cueq": True, "scaling": {8: ta.scaling}}
    monkeypatch.setattr(rowpair, "_cueq_on", lambda: False)                              # RF3's vanilla route (1/N on b): torch fns by name
    m2 = TriangleMultiplication(16, 8, "incoming")
    monkeypatch.setattr(rowpair, "TP_KERNELS", {"triatt": None, "trimul": None})
    plain = rowpair._trimul_fns(m2, 12)
    assert not isinstance(plain, RF.FusedTriMulFns) and rowpair.TP_KERNELS["trimul"] == "torch:vanilla_route"
