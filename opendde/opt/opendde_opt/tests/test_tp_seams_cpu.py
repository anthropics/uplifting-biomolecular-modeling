import os
"""The `--n_gpu P>1` adapter (opendde_opt.tp) end to end on CPU ranks (gloo), against the DENSE statements of a stub engine carrying OpenDDE's
attribute names (`_tp_stub_engine`): every residue-track seam runs on row shards — trunk (z_init rows, template rows, MSA rows, presharded
stacks), the trunk exit (the RowShard itself: nothing gathered; the structural stage on structural row shards is tests/test_tp_struct_cpu.py),
the distogram rows, the confidence head rows — and the per-rank results equal the dense
reference (fp32, tolerance: tiled contractions reorder sums). The seam CENSUS is asserted by count (no gather inside the trunk; every pair
stack call pre-sharded; every template / MSA block / confidence sample on rows). Skipped BY NAME without torch or without the 0.4.3 core
surface (`opt_core.mem.rowpair.msa`)."""
import copy
import os
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _triatt_torch_on_cpu_tensors(monkeypatch):
    """These seams run on CPU tensors (gloo ranks): the row-sharded line's tri-attention lever runs under the core's opt-out
    ``ROWPAIR_TRIATT_CORE=torch`` — the carried flash kernel serves CUDA tensors only and refuses CPU ones by name (test_tp_kernels);
    the spawned ranks inherit the variable."""
    monkeypatch.setenv("ROWPAIR_TRIATT_CORE", "torch")


torch = pytest.importorskip("torch", reason="the seam test runs the statements: torch (CPU) required")
pytest.importorskip("opt_core.mem.rowpair.msa", reason="the pinned core is older than the 0.4.3 rowpair surface (rowpair.msa): skipped by name until the re-pin")

P = 2
TOL = 2e-4
EXPECTED_CENSUS = {                      # stub engine: N_cycle=2, T=2 templates (1-block stack), 2 MSA blocks, trunk stack 2 blocks, confidence: 2 samples x 2-block stack
    "trunk_rows": 1, "gathers": 0, "unsharded": 0, "template_rows": 4, "template_feats_rows": 4, "msa_blocks_rows": 4,
    "presharded_calls": 2 * (2 + 1 + 2) + 2,   # per cycle: 2 template stack calls + the trunk stack + 2 MSA pair blocks (direct block calls, also counted in direct_blocks); + 2 confidence samples
    "direct_blocks": 4, "conf_rows": 1, "conf_samples_rows": 2, "distogram_rows": 1,
}                                                # (no relp counter: the pin's inference relp is lazy by itself, the adapter forces nothing)


def _dense_pae_pde_summaries(pae_logits, pde_logits, contact, has_frame, asym_id, nb, lo=0.0, hi=32.0):
    """AF3 §5.9.1-2 on DENSE [N, N, nb] logits (the statements sample_confidence applies to whole tensors): bin-centre expectations, pTM =
    max over framed i of mean_j TM-kernel-weighted PAE probability, ipTM = the same over inter-chain j."""
    N = pae_logits.shape[0]
    width = (hi - lo) / nb
    centers = lo + width * (torch.arange(nb, dtype=torch.float32) + 0.5)
    p_pae, p_pde = torch.softmax(pae_logits.float(), -1), torch.softmax(pde_logits.float(), -1)
    out = {"token_pair_pae": (p_pae * centers).sum(-1), "token_pair_pde": (p_pde * centers).sum(-1)}
    d0 = 1.24 * (max(N, 19) - 15) ** (1.0 / 3) - 1.8
    w = 1.0 / (1.0 + (centers / d0) ** 2)
    tm = (p_pae * w).sum(-1)                                           # [N, N]
    hf = has_frame.bool()
    out["ptm"] = tm.mean(-1)[hf].max()
    inter = (asym_id[:, None] != asym_id[None, :]).float()
    out["iptm"] = ((tm * inter).sum(-1) / inter.sum(-1).clamp(min=1))[hf].max()
    return out


def _rank_entry(seed, n_tok, templates="real"):
    """Runs in every rank process: dense reference (unpatched stubs) -> install the adapter -> the same pipeline sharded -> compare on THIS rank."""
    from opendde_opt import tp
    from opendde_opt.tests import _tp_stub_engine as E
    E.install_modules()
    model, feats, coords = E.build(seed, n_tok)                            # structural expansion disabled in the stub: the stock's equality — the residue RowShard is the exit (tests/test_tp_struct_cpu.py holds the structural stage)
    if templates == "dummy_only":                                              # the DUMMY template batch (aatype padded 0, every pair feature and mask zero): the embedder still runs
        for k in list(feats):                                                  # (n_blocks > 0) and the row path must feed it exactly what the dense path feeds it — no 'no real template' shortcut
            if k.startswith("template_"):
                feats[k] = torch.zeros_like(feats[k])
    with torch.no_grad():
        ref = model.pipeline(copy.copy(feats), coords)
    assert not tp.STATS["installed"]
    assert tp.in_rank_process()
    tp.install()
    assert tp.STATS["installed"], tp.STATS
    with torch.no_grad():
        feats_ = model.input_features(copy.copy(feats))                       # the stock forward's generate_relp (forced lazy by the adapter here)
        s_inputs, s, zrs = model.get_pairformer_output(feats_, model.N_cycle)
        assert isinstance(zrs, tp.RowShard), type(zrs)
        lay = zrs.lay
        r0, r1 = lay.r0, lay.r0 + lay.R
        _fd, _si, _s, z_exit = model.expand_to_structural_tokens(input_feature_dict=feats, s_inputs=s_inputs, s=s, z=zrs)
        assert isinstance(z_exit, tp.RowShard) and z_exit.z is zrs.z, type(z_exit)   # nothing gathered: the shard itself leaves the trunk
        from opt_core.mem.rowpair import shard as _shard
        z_full = _shard.unshard_rows(z_exit.z, z_exit.lay, dim=0)              # the TEST's own all-gather, for the dense comparison only
        contact = model.compute_distogram_contact_probs(zrs)
        plddt, pae, pde, resolved = model.confidence_head(input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s, z_trunk=zrs, pair_mask=None, x_pred_coords=coords)
        assert pae is None and pde is None, "the [N, N, bins] logits are never materialised: the head leaves row-block reducers"
        pend = tp._PENDING.pop("conf")
        assert len(pend["reducers"]) == 2 and pend["finish"] == "exact", pend
        red_stats = [red.finalize(lay.bounds, collect_full=True) for red in pend["reducers"]]      # collective on every rank; dicts on rank 0

    def md(a, b):
        return float((a - b).abs().max())
    scale = max(1.0, float(ref["z"].abs().max()))
    diffs = {"z_rows": md(zrs.z, ref["z"][r0:r1]), "z_exit_full": md(z_full, ref["z"]), "s": md(s, ref["s"]), "s_inputs": md(s_inputs, ref["s_inputs"]),
             "contact_rows": md(contact, ref["contact"][r0:r1]), "plddt": md(plddt, ref["plddt"]), "resolved": md(resolved, ref["resolved"])}
    assert tuple(contact.shape) == (r1 - r0, lay.N), contact.shape             # rows only: no [N, N] on any rank
    if tp.STATS["rank"] == 0:                                              # the reduced statistics == the AF3 formulas on the DENSE logits (independent reference below)
        for i, st_ in enumerate(red_stats):
            dense = _dense_pae_pde_summaries(ref["pae"][i], ref["pde"][i], ref["contact"], feats["has_frame"], feats["asym_id"], E.PBINS)
            diffs[f"ptm{i}"] = md(st_["ptm"].reshape(()), dense["ptm"])
            diffs[f"iptm{i}"] = md(st_["iptm"].reshape(()), dense["iptm"])
            diffs[f"token_pair_pde{i}"] = md(st_["token_pair_pde_f32"].reshape(n_tok, n_tok), dense["token_pair_pde"])
            diffs[f"contact_probs{i}"] = md(st_["contact_probs_f32"], ref["contact"])
            diffs[f"token_pair_pae_f16_{i}"] = md(st_["token_pair_pae_f16"].float(), dense["token_pair_pae"]) / 50.0   # fp16 collection: half-precision storage of [0, 32) values
    bad = {k: v for k, v in diffs.items() if not (v <= TOL * scale)}
    assert not bad, (tp.STATS["rank"], bad, diffs, scale)
    st = tp.kit_stats()
    st.update({k: tp.STATS[k] for k in ("template_feats_rows", "conf_samples_rows", "distogram_rows", "trunk_exit", "direct_blocks")})
    census = {k: st.get(k) for k in EXPECTED_CENSUS}
    assert census == EXPECTED_CENSUS, (tp.STATS["rank"], census)
    assert st["trunk_exit"] is None, st["trunk_exit"]                        # 'rowshard' is recorded by the structural stage's entry (expansion enabled); the stub's equality exit records nothing
    assert tp.STATS["schedule"]["pwa_schunk"] >= 1 and tp.STATS["schedule"]["hostgather_rows"] == lay.B
    return {"rank": tp.STATS["rank"], "layout": repr(lay), "diffs": diffs, "scale": scale, "census": census, "trunk_exit": st["trunk_exit"],
            "evidence": dict(tp.evidence_pairs()), "schedule": dict(tp.STATS["schedule"]),
            "zinit_blocks": tp.STATS["zinit_blocks"], "template_feats_rows": tp.STATS["template_feats_rows"]}


@pytest.mark.parametrize("geom,msa_m", [((40, 2), "token_sharded"), ((124, 4), "token_sharded"), ((40, 2), "replicated")],
                         ids=["N40_P2_B16_R32_last8", "N124_P4_B16x2_R32_last28", "N40_P2_msa_m_replicated"])
def test_residue_track_row_sharded_end_to_end_equals_dense(geom, msa_m, monkeypatch):
    """Two grid layouts with a ragged last rank: P=2 (rank 1 holds a partial block only) and P=4 with two blocks per rank and a partial last
    rank (the class of 1,012 tokens at P=8: B=64 x 2, last rank 116 rows); the MSA representation token-sharded (the line default) and
    replicated (``ROWPAIR_MSA_M_LAYOUT=replicated``)."""
    from opendde_opt.tests.conftest import run_sharded_or_skip
    from opendde_opt import tp
    n_tok, n_rank = geom
    monkeypatch.setenv(tp.ENV_MSA_M_LAYOUT, msa_m)                            # inherited by the rank processes
    out = run_sharded_or_skip(n_rank, _rank_entry, 0, n_tok, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=600)
    print("RANK0", out)
    assert out["census"] == EXPECTED_CENSUS
    assert out["schedule"]["msa_m"] == msa_m and out["schedule"]["diffz"] == "replaced_by_rowpair"
    assert out["schedule"]["inputs"].startswith("template_pair=") and "token_bonds=replicated" in out["schedule"]["inputs"]
    assert out["schedule"]["input_feats"] == "template_pair_rows_at_trunk_entry"   # the stub feeds the model itself (no stock runner in this process): whole HOST tensors cut at trunk entry
    assert out["zinit_blocks"] >= 1 and out["template_feats_rows"] == len(tp.TEMPLATE_PAIR_FEATS), out   # z_init born per row block; the template pair feats sliced to rows once
    assert max(out["diffs"].values()) <= TOL * out["scale"]
    ev = out["evidence"]
    assert ev["conf_logits"] == "reducer:exact" and out["schedule"]["struct_stage"].startswith("rowshard")
    assert "replicated_by_design" in ev and "struct_singles" in ev["replicated_by_design"] and "struct_stage_host" not in ev["replicated_by_design"]



def test_inplace_transpose_row_sharded_equals_dense(monkeypatch):
    """The BIG_TP export ROWPAIR_TRANSPOSE_INPLACE=1 (the ending-orientation transpose swaps blocks inside the shard's own storage) changes
    the schedule, not the arithmetic: sharded == dense end to end, same census."""
    from opendde_opt.tests.conftest import run_sharded_or_skip
    from opendde_opt import tp
    monkeypatch.setenv(tp.ENV_MSA_M_LAYOUT, "token_sharded")
    monkeypatch.setenv("ROWPAIR_TRANSPOSE_INPLACE", "1")
    out = run_sharded_or_skip(2, _rank_entry, 0, 40, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=600)
    print("RANK0 transpose_inplace=1", out)
    assert out["census"] == EXPECTED_CENSUS and out["schedule"]["transpose_inplace"] == "1"
    assert max(out["diffs"].values()) <= TOL * out["scale"], out["diffs"]


def test_dummy_only_template_batch_row_sharded_equals_dense(monkeypatch):
    """OpenDDE 1.1.1 runs the template embedder whenever its n_blocks > 0, also for the all-dummy template batch; under `--n_gpu P` the host
    row slabs of that batch reach the embedder exactly as the dense zeros do (sharded == dense end to end, same census)."""
    from opendde_opt.tests.conftest import run_sharded_or_skip
    from opendde_opt import tp
    monkeypatch.setenv(tp.ENV_MSA_M_LAYOUT, "token_sharded")
    out = run_sharded_or_skip(2, _rank_entry, 0, 40, "dummy_only", mode="big", backend="gloo", cpu_ok=True, run_timeout_s=600)
    print("RANK0 dummy_only", out)
    assert out["census"] == EXPECTED_CENSUS and out["template_feats_rows"] == len(tp.TEMPLATE_PAIR_FEATS)
    assert max(out["diffs"].values()) <= TOL * out["scale"], out["diffs"]


def test_n_gpu_1_installs_nothing_and_the_stub_dense_path_is_untouched(monkeypatch):
    """P = 1: the adapter installs no patch (structural rule); the engine's own statements run."""
    from opendde_opt import tp
    from opendde_opt.tests import _tp_stub_engine as E
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert not tp.in_rank_process()
    E.install_modules()
    model, feats, coords = E.build(1)
    before = E.OpenDDE.get_pairformer_output
    tp.install()
    assert E.OpenDDE.get_pairformer_output is before and not tp.STATS["installed"]
    with torch.no_grad():
        out = model.pipeline(feats, coords)
    assert torch.is_tensor(out["z"]) and tuple(out["z"].shape) == (E.N, E.N, E.C_Z)


def test_rowshard_refuses_a_trunk_checkpoint_by_name():
    from opendde_opt import tp
    from opt_core.mem.rowpair import RowpairRefused
    with pytest.raises(RowpairRefused, match="trunk checkpoint"):
        tp.RowShard(torch.zeros(2, 4, 3), None).cpu()


def _rank_entry_window_guard(seed, n_tok):
    """Every rank: the trunk under a TorchDispatchMode that records, between the census marks ``trunk_entry`` and ``after_template`` of every
    cycle (the recycle statement + the template embedder), every NON-VIEW op output shaped like this rank's z shard ``[R, N, C_z]``; with the
    block envs below (8 rows < R) the only such storage is z itself (allocated once, cycle 0)."""
    import copy
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from opendde_opt import tp
    from opendde_opt.tests import _tp_stub_engine as E
    torch.set_num_threads(1)
    E.install_modules()
    model, feats, coords = E.build(seed, n_tok)
    tp.install()
    assert tp.STATS["installed"], tp.STATS

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.window, self.storages, self.ops = False, {}, []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if self.window and torch.is_tensor(out) and out.dim() >= 3 and out._base is None and tuple(out.shape[-2:]) == (n_tok, E.C_Z) and int(out.shape[-3]) < n_tok:
                ptr = out.untyped_storage().data_ptr()
                if ptr not in self.storages:                                   # every distinct storage of a [rows, N, C_z] non-view output (blocks AND shards; filtered by rows == R below)
                    import traceback
                    where = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[:-1]
                             if f.filename.endswith(("/tp.py", "/trunk.py", "/template.py", "/shard.py", "/_tp_stub_engine.py"))][-3:]
                    self.storages[ptr] = (str(func), tuple(out.shape), where); self.ops.append((str(func), tuple(out.shape)))
            return out
    g = Guard()
    orig_mark = tp._mark

    def mark(stage):
        if stage == "trunk_entry":
            g.window = True
        elif stage == "after_template":
            g.window = False
        elif stage == "after_pairstack":
            g.window = True                                                    # the next cycle's recycle statement follows
        return orig_mark(stage)
    tp._mark = mark
    te_stack = model.template_embedder.pairformer_stack
    orig_stack_forward = te_stack.forward

    def stack_forward(*a, **k):                                                 # the template PAIR STACK's own transients (tri-mult projections / A block on the [R, N, c]
        was, g.window = g.window, False                                         # template shard) are the core drivers' budgeted operands, not glue: outside this assertion
        try:
            return orig_stack_forward(*a, **k)
        finally:
            g.window = was
    te_stack.forward = stack_forward
    try:
        feats_ = model.input_features(copy.copy(feats))
        with torch.no_grad(), g:
            s_inputs, s, zrs = model.get_pairformer_output(feats_, model.N_cycle)
        assert isinstance(zrs, tp.RowShard), type(zrs)
    finally:
        tp._mark = orig_mark
        g.window = False
    R = int(zrs.lay.R)
    shard_shaped = [v for v in g.storages.values() if int(v[1][-3]) == R]      # storages with exactly this rank's row count = shard-sized
    return {"rank": tp.rank(), "shard_shaped_storages": len(shard_shaped), "ops": shard_shaped[:8], "block_storages": len(g.storages) - len(shard_shaped), "R": R, "cycles": int(model.N_cycle),
            "schedule": {k: tp.STATS["schedule"].get(k) for k in ("zinit", "recycle", "template", "zres", "template_rows_after_trunk", "struct_trimul_rb")}, "recycle_rows_calls": tp.STATS["recycle_rows_calls"]}


def test_recycle_and_template_windows_allocate_no_shard_sized_temporary(monkeypatch):
    """sha C: the recycle statement and the template embedder run per ROW BLOCK in place (core trunk.recycle_shard_ / template.template_embed_rows):
    with 8-row blocks (< R) the only ``[R, N, C_z]``-shaped storage allocated in those windows is z itself, once (cycle 0) — no ``LN(z)``,
    ``linear(LN(z))``, ``z_init + …``, ``LN_z(z)``, feature slab, template term or ``z + term`` shard temporaries."""
    from opendde_opt.tests.conftest import run_sharded_or_skip
    from opendde_opt import tp
    for k in ("ROWPAIR_RECYCLE_ROWS", "ROWPAIR_INIT_ROWS", tp.ENV_TEMPL_ROWS):
        monkeypatch.setenv(k, "8")
    out = run_sharded_or_skip(2, _rank_entry_window_guard, 0, 40, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=600)
    print("RANK0", out)
    assert out["R"] > 8 and out["cycles"] >= 2 and out["recycle_rows_calls"] == out["cycles"]
    assert out["schedule"] == {"zinit": "device", "recycle": "rows_inplace", "template": "embed_rows_inplace", "zres": "device", "template_rows_after_trunk": "host", "struct_trimul_rb": "budget"}   # CPU: parking is a GPU mechanism (ROWPAIR_PARK_* / ODDE_TP_STRUCT_TRIMUL_RB unset here)
    assert out["shard_shaped_storages"] == 1, out["ops"]                       # z itself (aten.empty, cycle 0); every other storage in the window is an 8-row block
    assert out["block_storages"] >= 1
