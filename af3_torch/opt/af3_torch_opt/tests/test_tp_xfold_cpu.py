"""rowpair_xfold.py on the REAL xfold modules, CPU fp32, P = 2 rank-threads (opt_core.testing.run_ranks) vs the dense xfold statements on the
whole tensors: the pair block (with and without the single track), the distogram head, the confidence head, the trunk (Evoformer incl. the
template embedder and the MSA module), the diffusion conditioning rows / the sharded diffusion transformer / the banded atom-encoder statics.
Needs torch + einops + triton (xfold.fastnn imports it; the torch routes run on CPU) and opt_core >= 0.4.3 importable; skipped otherwise (the
package's other tests import neither). Random-initialised modules: the equivalence is sharded == dense of the SAME modules (no weights)."""
import dataclasses
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("triton")


def _triton_import_stub():
    """On a box WITHOUT a GPU driver, ``xfold.fastnn`` cannot import as is: its module-level ``@triton.autotune`` decorator probes the active
    driver at import time (``triton.jit`` does not). The fastnn routes exercised here are its TORCH implementations (``fastnn_config``: 'torch'
    for LayerNorm / attention / GLU), so on such a box the real triton is imported and ONLY ``triton.autotune`` / ``triton.heuristics`` are
    replaced by pass-through decorators (named here; never done when CUDA is present)."""
    if torch.cuda.is_available():
        return "real"
    import triton
    triton.autotune = lambda *a, **kw: (lambda fn: fn)
    triton.heuristics = lambda *a, **kw: (lambda fn: fn)
    return "autotune_noop"


TRITON = _triton_import_stub()

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.dirname(OPT)
KIT = os.path.join(HOME, "opt", "forward", "af3t", "af3_torch")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

testing = pytest.importorskip("opt_core.testing")
from opt_core.mem.rowpair import RowpairRefused, dist as D  # noqa: E402

N = int(os.environ.get("XFOLD_TP_TEST_N", "64"))          # geometry knobs for ragged / many-rank repros (defaults: N=64, P=2, align 16, row batch 16)
P = int(os.environ.get("XFOLD_TP_TEST_P", "2"))
ALIGN_T = os.environ.get("XFOLD_TP_TEST_ALIGN", "16")
if os.environ.get("XFOLD_TP_TEST_HANG_S"):                  # dump every thread's stack if a case runs longer than this (a collective desync hangs, it does not fail)
    import faulthandler; faulthandler.dump_traceback_later(int(os.environ["XFOLD_TP_TEST_HANG_S"]), repeat=True)
TOL = 1e-4          # fp32 CPU: GEMMs with a different M may reorder inner reductions in some BLAS builds; observed values are printed


def _load_rpx():
    if "rowpair_xfold" in sys.modules:
        return sys.modules["rowpair_xfold"]
    spec = importlib.util.spec_from_file_location("rowpair_xfold", os.path.join(OPT, "af3_torch_opt", "rowpair_xfold.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["rowpair_xfold"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def env():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    import af3_torch_api  # noqa: F401  (sets xfold.of3.OF3 = True — the OpenFold3 weight layout — before any module is built, as build_model does)
    from xfold import of3
    assert of3.OF3, "the kit's weight layout is OpenFold3"
    from xfold.alphafold3 import AlphaFold3
    model = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=2)
    model.eval()
    for prm in model.parameters():                       # LayerNorm / AdaLN-zero inits are degenerate (zeros): randomise everything (small)
        with torch.no_grad():
            prm.copy_(torch.randn_like(prm) * 0.1)
    model._af3t_kernels = None
    rpx = _load_rpx()
    rpx.QBLOCK = int(os.environ.get("XFOLD_TP_TEST_QBLOCK", "16"))     # several q-block launches per shard at this N (the module constant is 256)
    return dict(model=model, rpx=rpx, batch=_make_batch(N))


def _gi(idxs, mask, input_shape):
    from xfold.nn.atom_layout import GatherInfo
    return GatherInfo(gather_idxs=idxs.to(torch.int64), gather_mask=mask.to(torch.bool), input_shape=torch.tensor(input_shape))


def _make_batch(N):
    """A consistent synthetic Batch: N tokens (2 chains), 1 atom per token (dense slot 0), 32-query / 128-key atom subsets, 4 MSA rows,
    2 template slots (identical: exercises the slot de-duplication), no bonds."""
    from xfold import feat_batch, features
    g = torch.Generator().manual_seed(1)
    asym = torch.tensor([1] * (N // 2) + [2] * (N - N // 2), dtype=torch.int32)
    resid = torch.cat([torch.arange(N // 2), torch.arange(N - N // 2)]).to(torch.int32) + 1
    mask = torch.ones(N, dtype=torch.bool); mask[5] = False; mask[N - 3] = False
    tf = features.TokenFeatures(residue_index=resid, token_index=torch.arange(N, dtype=torch.int32) + 1, aatype=torch.randint(0, 20, (N,), generator=g).to(torch.int32),
                                mask=mask, seq_length=torch.tensor(N, dtype=torch.int32), asym_id=asym, entity_id=torch.ones(N, dtype=torch.int32),
                                sym_id=asym.clone(), is_protein=torch.ones(N, dtype=torch.bool), is_rna=torch.zeros(N, dtype=torch.bool),
                                is_dna=torch.zeros(N, dtype=torch.bool), is_ligand=torch.zeros(N, dtype=torch.bool),
                                is_nonstandard_polymer_chain=torch.zeros(N, dtype=torch.bool), is_water=torch.zeros(N, dtype=torch.bool))
    A = 24
    atom_mask = torch.zeros(N, A, dtype=torch.bool); atom_mask[:, 0] = True; atom_mask[:, 1] = True    # 2 atoms per token
    psi = features.PredictedStructureInfo(atom_mask=atom_mask, residue_center_index=torch.zeros(N, dtype=torch.int32))
    pb = features.PseudoBetaInfo(token_atoms_to_pseudo_beta=_gi(torch.arange(N) * A, torch.ones(N, dtype=torch.bool), (N, A)))
    ref = features.RefStructure(positions=torch.randn(N, A, 3, generator=g), mask=atom_mask.clone(), element=torch.randint(0, 20, (N, A), generator=g).to(torch.int32),
                                charge=torch.zeros(N, A), atom_name_chars=torch.randint(0, 60, (N, A, 4), generator=g).to(torch.int32),
                                ref_space_uid=torch.arange(N, dtype=torch.int32)[:, None].expand(N, A).contiguous())
    # atom subsets: n_atoms = 2N flat atoms in token order; queries 32 per subset, keys 128 per subset centred on the subset
    n_at = 2 * N; nq, nk = 32, 128
    ns = (n_at + nq - 1) // nq
    flat_tok = torch.arange(N).repeat_interleave(2)                   # token of flat atom a
    flat_slot = torch.arange(n_at) % 2                                # dense slot of flat atom a
    flat_dense = flat_tok * A + flat_slot                             # index into [N, A]
    q_at = torch.arange(ns * nq).view(ns, nq); q_valid = q_at < n_at; q_at_c = q_at.clamp(max=n_at - 1)
    t2q_atoms = _gi(flat_dense[q_at_c], q_valid, (N, A))
    tok2q = _gi(flat_tok[q_at_c], q_valid, (N,))
    k_start = (torch.arange(ns) * nq + nq // 2 - nk // 2).clamp(min=0, max=max(0, n_at - nk))
    k_at = k_start[:, None] + torch.arange(nk)[None, :]; k_valid = k_at < n_at; k_at_c = k_at.clamp(max=n_at - 1)
    tok2k = _gi(flat_tok[k_at_c], k_valid, (N,))
    q2k = _gi(k_at_c, k_valid, (ns, nq))                              # keys gathered from the flattened queries array (flat atom index == query index here)
    q2ta = _gi(torch.arange(N)[:, None] * 2 + torch.arange(A)[None, :].clamp(max=1), atom_mask.clone(), (ns, nq))
    aca = features.AtomCrossAtt(token_atoms_to_queries=t2q_atoms, tokens_to_queries=tok2q, tokens_to_keys=tok2k, queries_to_keys=q2k, queries_to_token_atoms=q2ta)
    S_msa = 4
    msa = features.MSA(rows=torch.randint(0, 21, (S_msa, N), generator=g).to(torch.int32), mask=torch.ones(S_msa, N, dtype=torch.bool),
                       deletion_matrix=torch.zeros(S_msa, N), profile=torch.rand(N, 31, generator=g), deletion_mean=torch.zeros(N), num_alignments=torch.tensor(S_msa, dtype=torch.int32))
    T = 2
    tpos = torch.randn(1, N, A, 3, generator=g).expand(T, N, A, 3).contiguous()
    tmask = torch.zeros(T, N, A, dtype=torch.bool); tmask[:, :, :4] = True
    templates = features.Templates(aatype=tf.aatype[None].expand(T, N).contiguous(), atom_positions=tpos, atom_mask=tmask)
    empty_b = _gi(torch.zeros(0, 2, dtype=torch.int64), torch.zeros(0, 2, dtype=torch.bool), (N,))
    plb = features.PolymerLigandBondInfo(tokens_to_polymer_ligand_bonds=empty_b, token_atoms_to_bonds=empty_b)
    llb = features.LigandLigandBondInfo(tokens_to_ligand_ligand_bonds=empty_b)
    return feat_batch.Batch(msa=msa, templates=templates, token_features=tf, ref_structure=ref, predicted_structure_info=psi,
                            polymer_ligand_bond_info=plb, ligand_ligand_bond_info=llb, pseudo_beta_info=pb, atom_cross_att=aca,
                            convert_model_output=None, frames=None)


def _maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


def _emb(N, g=2):
    gen = torch.Generator().manual_seed(g)
    return {"pair": torch.randn(N, N, 128, generator=gen), "single": torch.randn(N, 384, generator=gen), "target_feat": torch.randn(N, 447, generator=gen)}


def _shard(emb, lay):
    return {"pair": emb["pair"][lay.r0:lay.r1].clone().contiguous(), "single": emb["single"].clone(), "target_feat": emb["target_feat"].clone()}


# ----------------------------------------------------------------------------------------------------------------------- pair block
@pytest.mark.parametrize("with_single", [True, False])
def test_pair_block_sharded_equals_dense(env, with_single):
    model, rpx = env["model"], env["rpx"]
    blk = model.evoformer.trunk_pairformer[0] if with_single else model.evoformer.msa_stack[0]
    gen = torch.Generator().manual_seed(3)
    pair = torch.randn(N, N, 128, generator=gen); single = torch.randn(N, 384, generator=gen)
    seq_mask = env["batch"].token_features.mask
    pair_mask = (seq_mask[:, None] * seq_mask[None, :]).to(torch.float32)
    with torch.no_grad():
        if with_single:
            ref_pair, ref_single = blk(pair.clone(), pair_mask, single.clone(), seq_mask)
        else:
            zz = pair.clone()
            zz += blk.triangle_multiplication_outgoing(zz, mask=pair_mask); zz += blk.triangle_multiplication_incoming(zz, mask=pair_mask)
            zz += blk.pair_attention1(zz, mask=pair_mask); zz += blk.pair_attention2(zz, mask=pair_mask); zz += blk.pair_transition(zz)
            ref_pair, ref_single = zz, None

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T))
            lay = rpx.begin_item(N)
            z = pair[lay.r0:lay.r1].clone().contiguous()
            z, s = rpx.run_pair_stack_sharded([blk], z, pair_mask[lay.r0:lay.r1].contiguous(), lay, s=single.clone() if with_single else None,
                                              seq_mask=seq_mask, with_single=with_single)
            return (lay.r0, lay.r1, z, s, dict(rpx.STATE["stats"]))

    outs = testing.run_ranks(P, rank_fn)
    for r0, r1, z, s, st in outs:
        dz = _maxdiff(z, ref_pair[r0:r1])
        print(f"pair block with_single={with_single} rows {r0}:{r1} max|dz|={dz:.3e} equal={torch.equal(z, ref_pair[r0:r1])}")
        assert dz <= TOL
        if with_single:
            ds = _maxdiff(s, ref_single); print(f"  single max|ds|={ds:.3e}"); assert ds <= TOL
        assert st["presharded_calls"] >= 1 and st["pair_blocks"] >= 1


# -------------------------------------------------------------------------------------------------------------------------- heads
def test_distogram_sharded_equals_dense(env):
    model, rpx, batch = env["model"], env["rpx"], env["batch"]
    emb = _emb(N)
    with torch.no_grad():
        ref = model.distogram_head(batch, emb)

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T)); lay = rpx.begin_item(N)
            out = rpx.run_distogram_sharded(model, batch, _shard(emb, lay), lay)
            return out, dict(rpx.STATE["stats"])

    outs = testing.run_ranks(P, rank_fn)
    assert outs[1][0] is None
    got = outs[0][0]
    d = _maxdiff(got["contact_probs"], ref["contact_probs"]); print(f"distogram contact_probs max|d|={d:.3e} equal={torch.equal(got['contact_probs'], ref['contact_probs'])}")
    assert d <= TOL and torch.equal(got["bin_edges"], ref["bin_edges"])
    assert outs[0][1]["distogram_rows"] > 0


def test_confidence_sharded_equals_dense(env):
    model, rpx, batch = env["model"], env["rpx"], env["batch"]
    import af3_torch_api as A
    emb = _emb(N, 4)
    positions = torch.randn(N, 24, 3, generator=torch.Generator().manual_seed(5)) * 5
    with torch.no_grad():
        ref = A.run_confidence(model, batch, {k: v.clone() for k, v in emb.items()}, positions)

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T)); lay = rpx.begin_item(N)
            out = rpx.run_confidence_sharded(model, batch, _shard(emb, lay), lay, positions)
            return out, dict(rpx.STATE["stats"])

    outs = testing.run_ranks(P, rank_fn)
    assert outs[1][0] is None
    got = outs[0][0]
    assert set(got) == set(ref), (set(got) ^ set(ref))
    for k in ref:
        d = _maxdiff(got[k], ref[k]); print(f"confidence {k} {tuple(ref[k].shape)} max|d|={d:.3e} equal={torch.equal(got[k].float(), ref[k].float())}")
        assert d <= (TOL if k != "predicted_lddt" else 100 * TOL), k
    assert outs[0][1]["conf_rows"] > 0


def test_confidence_passes_on_the_trunk_plan_equal_dense(env):
    """Two confidence passes on one item's trunk-shard plan (heads.ZTrunkPlan: park at roll-out entry, begin / retire / end per pass — the
    P > 1 pred's protocol; on CPU the park's word is the library's CPU word and no bytes move) equal the dense head twice; the census
    records the plan's words and the placement defaults."""
    model, rpx, batch = env["model"], env["rpx"], env["batch"]
    import af3_torch_api as A
    emb = _emb(N, 4)
    positions = torch.randn(N, 24, 3, generator=torch.Generator().manual_seed(5)) * 5
    with torch.no_grad():
        ref = A.run_confidence(model, batch, {k: v.clone() for k, v in emb.items()}, positions)

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T)); lay = rpx.begin_item(N)
            loc = _shard(emb, lay)
            C = rpx.STATE["C"] if "C" in rpx.STATE else rpx._core()
            plan = rpx._ztrunk_new(C, loc["pair"], passes=2)
            entry = plan.park_now()
            outs = [rpx.run_confidence_sharded(model, batch, loc, lay, positions) for _ in range(2)]
            assert rpx.STATE["ztrunk_pass"] == 2 and plan.closed
            rpx.end_item(gate=False)
            return outs, entry, dict(rpx.STATE["stats"])

    res = testing.run_ranks(P, rank_fn)
    for (outs, entry, st) in res[:1]:                                         # rank 0 holds the outputs
        for got in outs:
            for k in ref:
                assert _maxdiff(got[k], ref[k]) <= (TOL if k != "predicted_lddt" else 100 * TOL), k
        print("plan words:", entry, st["levers"].get("ztrunk"), st["levers"].get("ztrunk_retire"), st.get("placement"))
        assert str(entry).startswith(("parked", "resident")) and "ztrunk" in st["levers"] and st["levers"].get("ztrunk_retire") == "pass1@embed"
        assert set(st["placement"]) >= {"ROWPAIR_CONF_PARK_ZTRUNK", "ROWPAIR_TRANSPOSE_INPLACE", "ROWPAIR_PARK_ZINIT", "ROWPAIR_TRIATT_ROWBLOCK"}


def test_placement_defaults_yield_to_the_environment(env, monkeypatch):
    """install() exports the kit's placement words where the variable is unset and records which came from the environment."""
    model, rpx = env["model"], env["rpx"]
    monkeypatch.setenv("ROWPAIR_TRANSPOSE_INPLACE", "0")
    monkeypatch.delenv("ROWPAIR_CONF_PARK_ZTRUNK", raising=False)
    rpx.install(model, 2, 0, align=int(ALIGN_T))
    pl = rpx.STATE["stats"]["placement"]
    assert pl["ROWPAIR_TRANSPOSE_INPLACE"] == "0:env"
    assert pl["ROWPAIR_CONF_PARK_ZTRUNK"] == "1:kit" and os.environ["ROWPAIR_CONF_PARK_ZTRUNK"] == "1"
    assert pl["ROWPAIR_TRIATT_ROWBLOCK"] == f"{rpx.QBLOCK}:kit"


# -------------------------------------------------------------------------------------------------------------------------- trunk
@pytest.mark.parametrize("opm_rows", [None, 8], ids=["opm_rows_budget", "opm_rows_8"])
def test_trunk_sharded_equals_dense(env, monkeypatch, opm_rows):
    """opm_rows: the MSA module's outer-product mean runs per OUTPUT ROW BLOCK of each rank (msa.opm_rows_budgeted): the core's budget (one
    block at this size) and a pinned 8-row block (ROWPAIR_OPM_ROWS) — the row block changes no (i, j) value (CPU fp32: the statement path)."""
    model, rpx, batch = env["model"], env["rpx"], env["batch"]
    import af3_torch_api as A
    from xfold.nn import featurization
    if opm_rows:
        monkeypatch.setenv("ROWPAIR_OPM_ROWS", str(opm_rows))
    else:
        monkeypatch.delenv("ROWPAIR_OPM_ROWS", raising=False)
    def _no_shuffle(msa):                                                  # the rows unpermuted in place of the random row permutation (rank-threads share one RNG)
        return msa
    monkeypatch.setattr(featurization, "shuffle_msa", _no_shuffle, raising=False)
    model.evoformer.num_msa = int(batch.msa.rows.shape[0])                # the fixture's MSA depth (the engine truncates to num_msa rows)
    with torch.no_grad():
        ref = A.run_trunk(model, batch, num_recycles=1)

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T)); lay = rpx.begin_item(N)
            out = rpx.run_trunk_sharded_xfold(A, model, batch, lay, num_recycles=1)
            return lay.r0, lay.r1, out, dict(rpx.STATE["stats"])

    outs = testing.run_ranks(P, rank_fn)
    for r0, r1, out, st in outs:
        dp = _maxdiff(out["pair"], ref["pair"][r0:r1]); ds = _maxdiff(out["single"], ref["single"]); dt = _maxdiff(out["target_feat"], ref["target_feat"])
        print(f"trunk rows {r0}:{r1} max|dpair|={dp:.3e} max|dsingle|={ds:.3e} max|dtf|={dt:.3e} equal_pair={torch.equal(out['pair'], ref['pair'][r0:r1])}")
        assert dp <= 10 * TOL and ds <= 10 * TOL and dt == 0.0
        assert st["trunk_rows"] > 0 and st["template_rows"] > 0 and st["msa_blocks_rows"] > 0 and st["gathers_in_trunk"] == 0
        assert st["levers"]["opm_rows"].startswith("statements:"), st["levers"]       # CPU: the module's statements per row block (the kernels serve CUDA + bf16 autocast)


# ---------------------------------------------------------------------------------------------------------------------- diffusion
def test_diffusion_conditioning_transformer_and_band_equal_dense(env):
    model, rpx, batch = env["model"], env["rpx"], env["batch"]
    from opt_core.mem.rowpair import diffusion as DF
    dh = model.diffusion_head
    emb = _emb(N, 6)
    gen = torch.Generator().manual_seed(7)
    act = torch.randn(N, dh.transformer.c_act if hasattr(dh.transformer, "c_act") else 768, generator=gen)
    single_cond = torch.randn(N, 384, generator=gen)
    seq_mask = batch.token_features.mask
    with torch.no_grad():
        pair_cond = dh._pair_conditioning(batch, emb, True)
        ref_act = dh.transformer(act.clone(), seq_mask, single_cond, pair_cond)
        ref_static = dh.atom_cross_att_encoder.compute_static(batch, trunk_single_cond=emb["single"], trunk_pair_cond=pair_cond)

    def rank_fn(rank, P_):
        with torch.no_grad():
            rpx.install(model, P_, rank, align=int(ALIGN_T)); lay = rpx.begin_item(N)
            ctx = rpx._DiffCtx(model, lay)
            rpx._prime_diffusion(ctx, batch, _shard(emb, lay), True)
            a = DF.diffusion_transformer_sharded(ctx.blocks, act.clone(), single_cond, ctx.zc, lay, schedule=ctx.sched, bias_cache=ctx.cache)
            return lay.r0, lay.r1, ctx.zc, a, ctx.enc_static, dict(rpx.STATE["stats"])

    outs = testing.run_ranks(P, rank_fn)
    for r0, r1, zc, a, st_enc, st in outs:
        dz = _maxdiff(zc, pair_cond[r0:r1]); da = _maxdiff(a, ref_act)
        print(f"diffusion rows {r0}:{r1} max|dzcond|={dz:.3e} max|dact|={da:.3e}")
        assert dz <= TOL and da <= 10 * TOL
        for k in ("pair_cond", "enc_pair_logits", "queries_single_cond", "keys_single_cond"):
            dk = _maxdiff(st_enc[k], ref_static[k]); print(f"  encoder static {k} max|d|={dk:.3e}"); assert dk <= TOL, k
        assert st["zcond_rows"] > 0 and st["band_rows"] > 0


# ----------------------------------------------------------------------------------------------------------------------- refusals
def test_p1_refused_by_name(env):
    model, rpx = env["model"], env["rpx"]
    with pytest.raises(RowpairRefused):
        rpx.install(model, 1, 0)
    lay1 = D.Layout.checked(N, 1, 0, B=16, align=int(ALIGN_T))
    with pytest.raises(RowpairRefused):
        rpx.run_distogram_sharded(model, env["batch"], _emb(N), lay1)


def test_census_names_replicated_tensors(env):
    c = env["rpx"].census()
    assert "conf_full_matrices_rank0" in c["xfold_replicated"] and "msa" in c["xfold_replicated"]


def test_item_exit_gate_by_declared_seam_set():
    """end_item() refuses by name unless every seam of the DECLARED set advanced on the item: 'all' (this kit) requires the trunk trio too,
    'heads' (a caller whose trunk runs elsewhere) does not; any other set name is refused at install."""
    rpx = _load_rpx()
    for seams, driven, ok in (("all", rpx.SEAMS_REQUIRED, True), ("heads", rpx.SEAMS_REQUIRED_HEADS, True),
                              ("all", rpx.SEAMS_REQUIRED_HEADS, False), ("heads", ("conf_rows",), False)):
        rpx.STATE.update(rpx._fresh_state()); rpx.STATE["seams"] = seams   # a fresh per-rank state (every key reset)
        rpx.STATE["stats"]["items"] += 1; rpx.STATE["stats_at_entry"] = {k: rpx.STATE["stats"].get(k, 0) for k in rpx.SEAMS_REQUIRED}
        for k in driven:
            rpx.STATE["stats"][k] += 1
        if ok:
            rpx.end_item()
        else:
            with pytest.raises(rpx.XfoldTPRefused, match="never ran"):
                rpx.end_item()
    with pytest.raises(Exception, match="seams="):
        rpx.install(object(), 2, 0, seams="trunk_only")



def test_stage_boundary_records_the_census_word():
    """stage_boundary() meets (no group here: barrier is a no-op) and records boundary_sync='barrier' + the boundary names in the census."""
    rpx = _load_rpx()
    rpx.STATE.update(rpx._fresh_state())
    assert rpx.STATE["stats"]["boundary_sync"] is None
    assert rpx.stage_boundary("trunk_done") == "barrier"
    rpx.stage_boundary("diffusion_done")
    assert rpx.STATE["stats"]["boundary_sync"] == "barrier" and rpx.STATE["stats"]["boundaries"] == ["trunk_done", "diffusion_done"]
    assert rpx.census()["boundary_sync"] == "barrier" and rpx.census()["boundaries"] == ["trunk_done", "diffusion_done"]
