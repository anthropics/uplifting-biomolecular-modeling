"""The tensor-parallel statement half (tp.py) on the CPU: protenix 1.1.0's own modules with random weights, DENSE (the stock forward) vs
ROW-SHARDED (tp.py's bindings over opt_core.mem.rowpair) on P threaded ranks (opt_core.testing.run_ranks), fp32. Every seam:
the trunk end to end (z born as rows, recycling, MSA module, Pairformer; no gather inside), the template embedder, the distogram -> contact
rows, the confidence head -> the stock summary / full_data keys on rank 0, diffusion conditioning rows, the diffusion transformer with local
query rows (plain and fused bias), the atom-encoder token-pair band; plus the seam census. Skipped when protenix is not importable."""
import os
import types

import pytest


@pytest.fixture(autouse=True)
def _tri_attention_units_on_the_module(monkeypatch):
    """CPU: the xP line's flash tri-attention core cannot run on CPU tensors — the core raises RowpairRefused by design (never a silent
    plain path). These dense-equality tests opt it out with the core's engineering word, so every tri-attention unit runs the module's own
    dispatch (the adapter's stock callable); the fused TriMul units decline below the size gate on their own (N here << 2048)."""
    monkeypatch.setenv("ROWPAIR_TRIATT_CORE", "torch")

os.environ.setdefault("LAYERNORM_TYPE", "torch")                     # protenix's torch LayerNorm (the CUDA extension is not built on a CPU box)
torch = pytest.importorskip("torch")
try:
    import protenix.model.protenix  # noqa: F401
except Exception as _e:                                                # named skip with the reason (a CPU box without the stock wheel)
    pytest.skip(f"protenix not importable here: {_e!r}", allow_module_level=True)
pytest.importorskip("opt_core.testing")

from opt_core.mem.rowpair import dist as D                            # noqa: E402
from opt_core.mem.rowpair import shard as SH                          # noqa: E402
from opt_core import testing as _T                                    # noqa: E402


class T(object):
    @staticmethod
    def run_ranks(P, fn, *args, **kw):                                  # a hung rank fails the test by name within 90 s (the hub's timeout), never a stalled job
        kw.setdefault("timeout_s", 90.0)
        return _T.run_ranks(P, fn, *args, **kw)
from opt_core.mem.rowpair import diffusion as DF                      # noqa: E402
from opt_core.mem.rowpair import evidence as EV                       # noqa: E402

from protenix_v1_opt import tp                                        # noqa: E402
tp.SYNC["mode"] = "guard"                                              # threaded CPU ranks: the strict policy (every replicated tensor here IS identical by construction)

import protenix.model.protenix as PX                                  # noqa: E402
from protenix.model import sample_confidence as SC                    # noqa: E402
from protenix.model.modules import pairformer as PF, embedders as EM, head as HDM, confidence as CO, diffusion as DM, transformer as TF  # noqa: E402
from protenix.model.modules.primitives import broadcast_token_to_local_atom_pair, rearrange_qk_to_dense_trunk                          # noqa: E402
from protenix.model.utils import permute_final_dims                   # noqa: E402

N, CZ, CS, CSI, CHUNK = 24, 16, 16, 12, 4
TOL = 2e-5


def _randomize(mod, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return mod.eval()


def _feats(seed=0, n=N, atoms_per_token=3, S=7, T=2):
    g = torch.Generator().manual_seed(seed)
    asym = torch.tensor([0] * (n // 2) + [1] * (n - n // 2))
    f = {"asym_id": asym, "entity_id": asym.clone(), "sym_id": torch.zeros(n, dtype=torch.long),
         "residue_index": torch.cat([torch.arange(n // 2), torch.arange(n - n // 2)]), "token_index": torch.arange(n),
         "token_bonds": (torch.rand(n, n, generator=g) < 0.1).float(), "has_frame": torch.ones(n, dtype=torch.bool),
         "msa": torch.randint(0, 32, (S, n), generator=g), "has_deletion": (torch.rand(S, n, generator=g) < 0.2).float(),
         "deletion_value": torch.rand(S, n, generator=g),
         "template_aatype": torch.randint(0, 32, (T, n), generator=g), "template_distogram": torch.rand(T, n, n, 39, generator=g),
         "template_pseudo_beta_mask": (torch.rand(T, n, n, generator=g) < 0.8).float(), "template_unit_vector": torch.randn(T, n, n, 3, generator=g),
         "template_backbone_frame_mask": (torch.rand(T, n, n, generator=g) < 0.8).float()}
    na = n * atoms_per_token
    f["atom_to_token_idx"] = torch.arange(na) // atoms_per_token
    f["atom_to_tokatom_idx"] = torch.arange(na) % atoms_per_token
    rep = torch.zeros(na, dtype=torch.bool); rep[::atoms_per_token] = True
    f["distogram_rep_atom_mask"] = rep
    f["is_ligand"] = torch.zeros(na)
    return f


def _configs():
    from ml_collections.config_dict import ConfigDict
    return ConfigDict({"triangle_attention": "torch", "triangle_multiplicative": "torch", "mc_dropout_rate": 0.0, "infer_setting": {"chunk_size": CHUNK},
                       "loss": {"distogram": {"min_bin": 2.3125, "max_bin": 21.6875, "no_bins": 64}, "pae": {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64},
                                "pde": {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}, "plddt": {"min_bin": 0.0, "max_bin": 1.0, "no_bins": 50}},
                       "metrics": {"clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75}}})


def _gather(z_loc, lay):
    return SH.gather_rows(z_loc.contiguous(), lay, 0)


def _close(a, b, tol=TOL, what=""):
    d = (a.float() - b.float()).abs().max().item()
    assert d <= tol, f"{what}: max|diff| {d} > {tol}"
    return d


# ---------------------------------------------------------------------------------------------------------------- relp rows
def test_relp_rows_equal_the_stock_plane():
    rpe = EM.RelativePositionEncoding(r_max=4, s_max=2, c_z=CZ)
    f = _feats()
    plane = EM.RelativePositionEncoding.generate_relp(rpe, dict(f))["relp"]
    rr = tp.RelpRows(rpe, f)
    assert rr.width == plane.shape[-1] == 4 * (4 + 1) + 1 + 2 * (2 + 1)
    assert torch.equal(rr.rows(0, N), plane) and torch.equal(rr.rows(5, 11), plane[5:11])


# ---------------------------------------------------------------------------------------------------------------- trunk end to end
class _FakeModel(object):
    """The attributes Protenix.get_pairformer_output reads (protenix.py:170-303), on protenix's own sub-modules with random weights."""
    training = False
    train_confidence_only = False

    def __init__(self, s_inputs, n_msa_blocks=2, n_pf_blocks=2, n_templ_blocks=0):
        self.configs = _configs()
        self._s_inputs = s_inputs
        self.input_embedder = lambda feats, inplace_safe=False, chunk_size=None: self._s_inputs
        self.linear_no_bias_sinit = _randomize(torch.nn.Linear(CSI, CS, bias=False), 1)
        self.linear_no_bias_zinit1 = _randomize(torch.nn.Linear(CS, CZ, bias=False), 2)
        self.linear_no_bias_zinit2 = _randomize(torch.nn.Linear(CS, CZ, bias=False), 3)
        self.relative_position_encoding = _randomize(EM.RelativePositionEncoding(r_max=4, s_max=2, c_z=CZ), 4)
        self.linear_no_bias_token_bond = _randomize(torch.nn.Linear(1, CZ, bias=False), 5)
        self.layernorm_z_cycle = _randomize(torch.nn.LayerNorm(CZ), 6)
        self.linear_no_bias_z_cycle = _randomize(torch.nn.Linear(CZ, CZ, bias=False), 7)
        self.layernorm_s = _randomize(torch.nn.LayerNorm(CS), 8)
        self.linear_no_bias_s = _randomize(torch.nn.Linear(CS, CS, bias=False), 9)
        self.template_embedder = _randomize(PF.TemplateEmbedder(n_blocks=n_templ_blocks, c=8, c_z=CZ, dropout=0.0), 10)
        self.msa_module = _randomize(PF.MSAModule(n_blocks=n_msa_blocks, c_m=8, c_z=CZ, c_s_inputs=CSI, msa_dropout=0.0, pair_dropout=0.0, blocks_per_ckpt=None,
                                                  msa_chunk_size=3, msa_configs={"enable": True, "strategy": "topk", "sample_cutoff": {"train": 7, "test": 7},
                                                                                 "min_size": {"train": 7, "test": 7}}), 11)   # topk + min == S: the subsample is RNG-independent (threaded ranks share one generator)
        self.pairformer_stack = _randomize(PF.PairformerStack(n_blocks=n_pf_blocks, n_heads=2, c_z=CZ, c_s=CS, dropout=0.0), 12)


def _trunk_dense(model, feats, n_cycle):
    f = dict(feats)
    f = EM.RelativePositionEncoding.generate_relp(model.relative_position_encoding, f)     # the stock plane
    torch.manual_seed(123)                                                                 # the MSA subsample draw
    with torch.no_grad():
        return PX.Protenix.get_pairformer_output(model, input_feature_dict=f, N_cycle=n_cycle, inplace_safe=True, chunk_size=CHUNK)


def _trunk_tp(rank, P, model, feats, n_cycle):
    f = dict(feats)
    f["relp"] = tp.RelpRows(model.relative_position_encoding, f)
    tp.STATE["chunk"] = CHUNK
    torch.manual_seed(123)
    with torch.no_grad():
        s_inputs, s, zsh = tp.get_pairformer_output_tp(model, f, n_cycle, inplace_safe=True, chunk_size=CHUNK)
    assert isinstance(zsh, tp.PairShard) and zsh.t.shape[0] == zsh.layout.R
    return s, _gather(zsh.t, zsh.layout), zsh.layout.bounds


@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("with_templates", [False, True])
def test_trunk_end_to_end_equals_dense(P, with_templates, monkeypatch):
    monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)                                 # the stock placement: the raw MSA features on the device
    torch.manual_seed(7)
    feats = _feats()
    s_inputs = torch.randn(N, CSI)
    model = _FakeModel(s_inputs, n_templ_blocks=(1 if with_templates else 0))
    if not with_templates:
        feats = {k: v for k, v in feats.items() if not k.startswith("template_")}
    c0 = dict(tp.COUNTS)
    s_d, s_dense, z_dense = None, *_trunk_dense(model, feats, 2)[1:]
    outs = T.run_ranks(P, _trunk_tp, model, feats, 2)
    for r, (s_tp, z_tp, bounds) in enumerate(outs):
        _close(z_tp, z_dense, what=f"trunk z rank{r} P={P}")
        _close(s_tp, s_dense, what=f"trunk s rank{r} P={P}")
    c = {k: tp.COUNTS[k] - c0.get(k, 0) for k in tp.COUNTS}
    assert c["trunk_inits"] == P and c["recycles"] == 2 * P and c["msa_blocks_rows"] == 2 * 2 * P and c["gathers_in_trunk"] == 0
    assert c["pairstack_calls"] >= 2 * P and c["template_calls"] == (2 * P if with_templates else 0)
    park = tp.PARKS["z_init"]                                                              # the z_init shard's park record (ROWPAIR_PARK_ZINIT; a CPU shard stays resident: where=device)
    assert park["where"] == "device" and park["parked"] is False
    assert park["gib"] in {round(z_dense[b0:b1].numel() * z_dense.element_size() / 2 ** 30, 4) for (b0, b1) in outs[0][2]}   # any rank's shard bytes (Layout.bounds)
    sched = EV.schedule()
    assert sched["park_z_init"] == "device" and sched["park_z_init_gib"] == round(park["gib"], 3) and sched["park_z_init_release"] == "resident"   # the core's census words (ShardPark.census_words)
    assert tp.record()["parks"]["z_init"] == park
    assert sched["msa_raw"] == "device"                                                    # ROWPAIR_MSA_HOST unset: the raw MSA features on the device, as the stock


@pytest.mark.parametrize("P", [2, 3])
def test_transpose_inplace_is_bitwise_neutral_through_the_trunk(P, monkeypatch):
    """ROWPAIR_TRANSPOSE_INPLACE=1 (the ×P line's export; read by opt_core.mem.rowpair.pairstack.pair_block_, which tp.pair_stack_tp / the MSA
    module's pair blocks go through): the ending-node triangle attention's distributed transposes swap row blocks into the shard's OWN storage
    instead of a second shard-sized buffer — placement only: the trunk (48-block-style Pairformer + MSA pair blocks on the shard) is BIT-IDENTICAL
    with the lever on and off, and the census names the form (`transpose_form=inplace:<k>` | `all`)."""
    monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)
    torch.manual_seed(7)
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    model = _FakeModel(torch.randn(N, CSI), n_templ_blocks=0)
    forms = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("ROWPAIR_TRANSPOSE_INPLACE", flag)
        EV.reset_schedule()
        outs = T.run_ranks(P, _trunk_tp, model, feats, 2)
        forms[flag] = (outs, EV.schedule().get("transpose_form"))
    for r in range(P):
        assert torch.equal(forms["1"][0][r][0], forms["0"][0][r][0]) and torch.equal(forms["1"][0][r][1], forms["0"][0][r][1]), f"rank {r}: transpose in place changed the trunk"
    assert str(forms["1"][1]).startswith("inplace:") and not str(forms["0"][1]).startswith("inplace:"), forms["0"][1:] + forms["1"][1:]


def _trunk_tp_host(rank, P, model, feats, n_cycle, mode):
    """The item-entry statements of a rank process (rowpair.replicate_inputs' MSA hold + tp.host_side_inputs) before the trunk, on this
    threaded rank: the raw MSA features leave the item; what this rank keeps on the host is by mode."""
    f = dict(feats)
    S, n = f["msa"].shape
    if rank != 0:                                                                         # ranks > 0 featurised their own item: whatever they hold is dropped, rank 0's lands
        f.update(msa=torch.zeros_like(f["msa"]), has_deletion=torch.zeros_like(f["has_deletion"]), deletion_value=torch.zeros_like(f["deletion_value"]))
    hold = tp.msa_host_hold(f)                                                            # the raw MSA features leave EVERY rank's item before the broadcast; rank 0 keeps its own aside
    assert hold is not None and hold["mode"] == mode
    assert all(k not in f for k in tp.MSA_HOST_KEYS) and set(hold["meta"]) == set(tp.MSA_HOST_KEYS) and hold["meta"]["msa"] == ((S, n), "int64")
    assert bool(hold["mine"]) == (rank == 0)
    tp.msa_host_land(f, hold, torch.device("cpu"))                                        # rank 0's tensors back; mode all: every rank's host copy made rank 0's (sync_host_features_); rank0: ranks > 0 hold none
    if mode == "all" or rank == 0:
        assert all(torch.equal(f[k], feats[k]) for k in tp.MSA_HOST_KEYS), f"rank {rank}: landed raw MSA differs from rank 0's"
    else:
        assert all(k not in f for k in tp.MSA_HOST_KEYS)
    tp.STATE["chunk"] = CHUNK
    facts = tp.host_side_inputs(f, None)                                                  # token_bonds + the raw MSA features -> this rank's host entry; the keys leave the item
    assert all(k not in f for k in tp.MSA_HOST_KEYS) and "token_bonds" not in f
    hm = tp.host_inputs()["msa"]
    assert hm["mode"] == mode and hm["n_rows"] == S and facts["tp_msa_host"].startswith(f"{mode}:S={S}:keys=3")
    if mode == "rank0" and rank != 0:
        assert hm["parked"] == {} and facts["tp_msa_host"].endswith("parked=0")
    else:
        assert set(hm["parked"]) == set(tp.MSA_HOST_KEYS) and hm["parked"]["msa"].shape == (S, n) and facts["tp_msa_host"].endswith("parked=3")
    f["relp"] = tp.RelpRows(model.relative_position_encoding, f)
    torch.manual_seed(123)
    with torch.no_grad():
        s_inputs, s, zsh = tp.get_pairformer_output_tp(model, f, n_cycle, inplace_safe=True, chunk_size=CHUNK)
    return s, _gather(zsh.t, zsh.layout), zsh.layout.bounds


@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("mode", ["all", "rank0"])
def test_msa_host_keeps_the_raw_msa_on_the_host_and_the_trunk_equals_dense(P, mode, monkeypatch):
    """ROWPAIR_MSA_HOST=all|rank0 (opt_core.mem.rowpair.msa_host; the ×P line exports rank0): the raw MSA features [S, N] are host-resident
    (rank 0's copy only in mode rank0), the per-cycle subsample statement draws its indices on a device stand-in and only the selected rows
    reach the device (mode rank0: gathered on rank 0, broadcast to the other ranks; mode all: every rank gathers its own) — placement only:
    the trunk equals the dense trunk exactly as with device-resident features, and the census names the placement."""
    monkeypatch.setenv("ROWPAIR_MSA_HOST", mode)
    EV.reset_schedule()
    torch.manual_seed(7)
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    s_inputs = torch.randn(N, CSI)
    model = _FakeModel(s_inputs, n_templ_blocks=0)
    c0 = dict(tp.COUNTS)
    s_dense, z_dense = _trunk_dense(model, feats, 2)[1:]
    outs = T.run_ranks(P, _trunk_tp_host, model, feats, 2, mode)
    for r, (s_tp, z_tp, bounds) in enumerate(outs):
        _close(z_tp, z_dense, what=f"msa_host={mode} trunk z rank{r} P={P}")
        _close(s_tp, s_dense, what=f"msa_host={mode} trunk s rank{r} P={P}")
    c = {k: tp.COUNTS[k] - c0.get(k, 0) for k in tp.COUNTS}
    assert c["msa_blocks_rows"] == 2 * 2 * P and c["msa_host_calls"] == 3 * 2 * P                      # 3 features x 2 cycles x P ranks served from the host
    assert sched["msa_sample_rows"] == 7 if (sched := EV.schedule()) else False                          # k_sel: topk with lower_bound == S keeps every row
    sched = EV.schedule()
    assert sched["msa_host_mode"] == mode and sched["msa_raw"] == f"host:{mode}" and sched["msa_host_rows"].startswith(f"{mode}:")
    assert "tp_msa_host" in sched and sched["tp_msa_host"].startswith(f"{mode}:S=7:keys=3")
    assert sched["msa_host_pageable_gib"] >= 0 and "msa_host_pinned_gib" in sched                         # token_bonds (+ the raw MSA) kept on the host through msa_host.to_host / park_features: the bytes are NAMED, pinned or pageable
    assert tp.REPLICATED["msa_raw"] in ({"rank0", "host:" + str(7 * N * (8 + 4 + 4))} if mode == "rank0" else {"host:" + str(7 * N * (8 + 4 + 4))})
    tp.STATE.pop("host_inputs", None)


def _msa_without_record(rank, P, model, feats, clear):
    f = dict(feats)
    hold = tp.msa_host_hold(f)
    tp.msa_host_land(f, hold, torch.device("cpu"))
    tp.STATE["chunk"] = CHUNK
    tp.host_side_inputs(f, None)
    lay = tp.layout_for(N)
    if clear == "entry":
        tp.STATE["host_inputs"].pop(rank, None)                                           # the bookkeeping slip: this rank's entry record gone
    else:
        tp.STATE["host_inputs"][rank]["N"] = N + 1                                        # ... or recorded for another item
    z_loc = torch.zeros(lay.R, N, CZ)
    with pytest.raises(tp.TPRefused, match="msa_host rank0: no host-side MSA record"):
        tp.msa_module_tp(model.msa_module, f, z_loc, model._s_inputs, lay, "torch", CHUNK)
    return True


@pytest.mark.parametrize("clear", ["entry", "N"])
def test_msa_host_without_the_entry_record_is_refused_by_name_never_a_skipped_msa_module(clear, monkeypatch):
    """Under ROWPAIR_MSA_HOST the adapter itself moved the raw MSA features out of the item, so `msa not in feats` is the normal state: the MSA
    module runs from the entry's explicit record or is REFUSED by name (a missing / foreign record never takes the stock's `no MSA -> z
    unchanged` exit silently); only an item recorded as carrying no MSA (`absent`) leaves z unchanged."""
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0")
    torch.manual_seed(7)
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    model = _FakeModel(torch.randn(N, CSI), n_templ_blocks=0)
    assert all(T.run_ranks(2, _msa_without_record, model, feats, clear))
    # an item WITHOUT MSA features: recorded absent at entry, z unchanged by name
    def absent(rank, P):
        f = {k: v for k, v in feats.items() if k not in tp.MSA_HOST_KEYS}
        tp.msa_host_land(f, tp.msa_host_hold(f), torch.device("cpu"))
        facts = tp.host_side_inputs(f, None)
        assert tp.host_inputs()["msa"] == {"mode": "rank0", "absent": True} and facts["tp_msa_host"] == "rank0:absent"
        lay = tp.layout_for(N)
        z_loc = torch.full((lay.R, N, CZ), 3.0)
        assert tp.msa_module_tp(model.msa_module, f, z_loc, model._s_inputs, lay, "torch", CHUNK) is z_loc
        return EV.schedule()["msa_raw"]
    assert T.run_ranks(2, absent) == ["absent:rank0"] * 2
    tp.STATE.pop("host_inputs", None)


# ---------------------------------------------------------------------------------------------------------------- distogram + confidence + summaries
def _heads():
    dh = _randomize(HDM.DistogramHead(c_z=CZ, no_bins=64), 20, 0.2)
    ch = _randomize(CO.ConfidenceHead(n_blocks=1, c_s=CS, c_z=CZ, c_s_inputs=CSI, b_pae=64, b_pde=64, b_plddt=50, b_resolved=2, max_atoms_per_token=3), 21, 0.2)
    return dh, ch


def _conf_dense(dh, ch, feats, s_inputs, s_trunk, z, coords, cfg):
    with torch.no_grad():
        contact = SC.compute_contact_prob(distogram_logits=dh(z).float(), **SC.get_bin_params(cfg.loss.distogram))
        plddt, pae, pde, resolved = CO.ConfidenceHead.forward.__wrapped__(ch, input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z, pair_mask=None,
                                                               x_pred_coords=coords, use_embedding=True, triangle_multiplicative="torch",
                                                               triangle_attention="torch", inplace_safe=False, chunk_size=CHUNK) \
            if hasattr(CO.ConfidenceHead.forward, "__wrapped__") else ch(input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z, pair_mask=None,
                                                                     x_pred_coords=coords, use_embedding=True, triangle_multiplicative="torch",
                                                                     triangle_attention="torch", inplace_safe=False, chunk_size=CHUNK)
        summ, full = SC.compute_full_data_and_summary(configs=cfg, pae_logits=pae, plddt_logits=plddt, pde_logits=pde, contact_probs=contact,
                                                      token_asym_id=feats["asym_id"], token_has_frame=feats["has_frame"], atom_coordinate=coords,
                                                      atom_to_token_idx=feats["atom_to_token_idx"], atom_is_polymer=1 - feats["is_ligand"], N_recycle=2,
                                                      interested_atom_mask=None, return_full_data=True, mol_id=None, elements_one_hot=None)
    return contact, plddt, resolved, summ, full


def _conf_tp(rank, P, dh, ch, feats, s_inputs, s_trunk, z, coords, cfg):
    tp.STATE["chunk"] = CHUNK
    tp.STATE["model"] = types.SimpleNamespace(configs=cfg)
    lay = tp.layout_for(N)
    zsh = tp.PairShard(SH.shard_rows(z, lay, 0).contiguous().clone(), lay)
    with torch.no_grad():
        cr = tp.distogram_tp(dh, zsh)
        tp.STATE["contact_rows"][rank] = cr
        plddt, th, th2, resolved = tp.confidence_tp(ch, dict(feats), s_inputs, s_trunk, zsh, None, coords, use_embedding=True, chunk_size=CHUNK)
        fn = tp.compute_full_data_and_summary_tp(SC.compute_full_data_and_summary)
        summ, full = fn(cfg, pae_logits=th, plddt_logits=plddt, pde_logits=th2, contact_probs=cr, token_asym_id=feats["asym_id"],
                        token_has_frame=feats["has_frame"], atom_coordinate=coords, atom_to_token_idx=feats["atom_to_token_idx"],
                        atom_is_polymer=1 - feats["is_ligand"], N_recycle=2, interested_atom_mask=None, return_full_data=True)
    return SH.gather_rows(cr.rows.contiguous(), lay, 0), plddt, resolved, summ, full


@pytest.mark.parametrize("P", [2, 3])
def test_distogram_confidence_and_summaries_equal_dense(P):
    torch.manual_seed(11)
    cfg = _configs()
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    dh, ch = _heads()
    s_inputs, s_trunk, z = torch.randn(N, CSI), torch.randn(N, CS), torch.randn(N, N, CZ)
    coords = torch.randn(2, N * 3, 3) * 6.0                                                # N_sample = 2
    contact_d, plddt_d, resolved_d, summ_d, full_d = _conf_dense(dh, ch, feats, s_inputs, s_trunk, z, coords, cfg)
    outs = T.run_ranks(P, _conf_tp, dh, ch, feats, s_inputs, s_trunk, z, coords, cfg)
    for r, (contact_t, plddt_t, resolved_t, summ_t, full_t) in enumerate(outs):
        _close(contact_t, contact_d, what=f"contact rank{r}")
        _close(plddt_t, plddt_d, what="plddt"); _close(resolved_t, resolved_d, what="resolved")
        assert len(summ_t) == len(summ_d) == 2
        if r != 0:
            assert summ_t == [{}, {}]                                                      # ranks > 0 write nothing
            continue
        for i in range(2):
            for k, v in summ_d[i].items():
                assert k in summ_t[i], f"summary key {k} missing"
                if torch.is_tensor(v):
                    _close(summ_t[i][k].float().cpu(), v.float().cpu(), 1e-4, what=f"summary[{i}].{k}")
            for k in ("token_pair_pae", "token_pair_pde", "contact_probs", "atom_plddt"):
                _close(full_t[i][k].float().cpu(), full_d[i][k].float().cpu(), 3e-2 if k != "atom_plddt" else 1e-4, what=f"full[{i}].{k}")   # fp16 planes


def _conf_tp_policy(rank, P, dh, ch, cond, feats, s_inputs, s_trunk, z, coords, cfg, entry):
    """One rank's post-trunk statements in main_inference_loop order: [roll-out entry: prepare_cache -> distogram rows first + park_now] ->
    distogram -> confidence passes; returns the outputs + the plan's record + the shard's storage size after the stage."""
    tp.STATE["chunk"] = CHUNK
    tp.STATE["model"] = types.SimpleNamespace(configs=cfg, distogram_head=dh)
    lay = tp.layout_for(N)
    zsh = tp.PairShard(SH.shard_rows(z, lay, 0).contiguous().clone(), lay)
    f = dict(feats)
    with torch.no_grad():
        zc = None
        if entry:
            rr = tp.RelpRows(cond.relpe, f)
            zc = tp.prepare_cache_cond_tp(cond, rr, zsh, False)                              # roll-out entry: contact rows taken from the device shard, then park_now
            assert isinstance(zsh.contact, tp.ContactRows) and zsh.plan is not None
            cr = zsh.contact
        else:
            cr = tp.distogram_tp(dh, zsh)
        tp.STATE["contact_rows"][rank] = cr
        plddt, th, th2, resolved = tp.confidence_tp(ch, f, s_inputs, s_trunk, zsh, None, coords, use_embedding=True, chunk_size=CHUNK)
        fn = tp.compute_full_data_and_summary_tp(SC.compute_full_data_and_summary)
        summ, full = fn(cfg, pae_logits=th, plddt_logits=plddt, pde_logits=th2, contact_probs=cr, token_asym_id=feats["asym_id"],
                        token_has_frame=feats["has_frame"], atom_coordinate=coords, atom_to_token_idx=feats["atom_to_token_idx"],
                        atom_is_polymer=1 - feats["is_ligand"], N_recycle=2, interested_atom_mask=None, return_full_data=True)
    rec = dict(tp.PARKS["z_trunk"])
    return (SH.gather_rows(cr.rows.contiguous(), lay, 0), plddt, resolved, summ, full, rec, int(zsh.t.untyped_storage().nbytes()),
            None if zc is None else _gather(zc.t, lay), dict(EV.schedule()))


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("free,park,entry", [(0, 0, False), (1, 0, False), (0, 1, False), (1, 1, False), (1, 1, True), (0, 0, True)])
def test_confidence_trunk_shard_policy_is_placement_only(S, free, park, entry, monkeypatch):
    """The trunk shard across the confidence passes (heads.ZTrunkPlan through tp.conf_plan; ROWPAIR_FREE_ZTRUNK / ROWPAIR_CONF_PARK_ZTRUNK, the
    ×P line exports both =1): per pass IN PLACE (one same-dtype last-use pass), PARKED on the host with the shard's storage released BEFORE
    the pass's pair input is allocated (from roll-out entry on when the conditioning runs first: the distogram rows are taken from the device
    shard first), or RESIDENT (both off) — placement only: every form gives outputs BIT-IDENTICAL to the resident form at S=1 and S=2, the
    words name the form per pass, and a parked / in-place shard is consumed (storage released / overwritten) after the stage."""
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", str(free)); monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", str(park))
    torch.manual_seed(11)
    cfg = _configs(); cfg.sample_diffusion = {"N_sample": S}
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    dh, ch = _heads()
    cond = _cond()
    s_inputs, s_trunk, z = torch.randn(N, CSI), torch.randn(N, CS), torch.randn(N, N, CZ)
    coords = torch.randn(S, N * 3, 3) * 6.0
    P = 2
    ref = T.run_ranks(P, _conf_tp_policy, dh, ch, cond, feats, s_inputs, s_trunk, z, coords, cfg, False) if (free, park, entry) != (0, 0, False) else None
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", "0"); monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", "0")
    res = T.run_ranks(P, _conf_tp_policy, dh, ch, cond, feats, s_inputs, s_trunk, z, coords, cfg, False)          # the resident reference (both levers off)
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", str(free)); monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", str(park))
    outs = T.run_ranks(P, _conf_tp_policy, dh, ch, cond, feats, s_inputs, s_trunk, z, coords, cfg, entry)
    contact_d, plddt_d, resolved_d, summ_d, full_d = _conf_dense(dh, ch, feats, s_inputs, s_trunk, z, coords, cfg)
    monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", "0")
    zc_ref = T.run_ranks(P, _cond_tp, cond, feats, z) if entry else None                                 # the conditioned pair rows read from the DEVICE shard
    for r in range(P):
        contact_t, plddt_t, resolved_t, summ_t, full_t, rec, st_bytes, zc_t, sched = outs[r]
        contact_0, plddt_0, resolved_0, summ_0, full_0, rec_0, st_0, _, sched_0 = res[r]
        # placement only: bit-identical to the resident form
        assert torch.equal(contact_t, contact_0) and torch.equal(plddt_t, plddt_0) and torch.equal(resolved_t, resolved_0), (r, rec["words"])
        if r == 0:
            for i in range(S):
                for k in ("token_pair_pae", "token_pair_pde", "contact_probs", "atom_plddt"):
                    assert torch.equal(full_t[i][k], full_0[i][k]), (k, rec["words"])
                for k, v in summ_0[i].items():
                    if torch.is_tensor(v):
                        assert torch.equal(summ_t[i][k], v), (k, rec["words"])
        # ... and equal to the dense stock statements (the existing tolerance of the row-sharded reductions)
        _close(contact_t, contact_d, what=f"contact rank{r}"); _close(plddt_t, plddt_d, what="plddt"); _close(resolved_t, resolved_d, what="resolved")
        # the per-pass words
        words = rec["words"]
        assert rec["passes"] == S and len(words) == S and rec["free"] == bool(free) and rec["park"] == bool(park)
        if entry and park:                                                                 # parked at roll-out entry: every pass served by the live park
            assert sched["conf_ztrunk_entry"] == "parked:host" and sched["zcond_src"] == "parked" and all(w == "parked:host" for w in words)
            assert sched["conf_ztrunk_entry_site"] == "prepare_cache:parked:host"
        elif free and S == 1:
            assert words == ["inplace"]
        elif free and not park:                                                            # S == 2: pass 0 is not the last use -> resident by name, pass 1 in place
            assert words == ["resident:free_declined:not_last", "inplace"]
        elif park:
            assert all(w == "parked:host" for w in words)
        else:
            assert all(w == "resident" for w in words) and (not entry or sched["conf_ztrunk_entry"] == "resident")
        consumed = any(w == "inplace" or w.startswith("parked") for w in words)
        assert rec["consumed"] is consumed
        if any(w.startswith("parked") for w in words):
            assert st_bytes == 0, "a parked trunk shard's storage is released and, after the last pass, not restored"
        assert sched["conf_pairstack"] == "inplace" and sched["conf_ztrunk"] == ",".join(words) and sched["conf_dtype"] == "float32"
        if entry:                                                                          # the conditioning read the trunk rows through the park: the same rows
            assert torch.equal(zc_t, zc_ref[r]), "z_cond from the parked source differs from the device-shard source"
    tp.STATE["contact_rows"].clear()


def _disto_bf16(rank, P, dh, z16):
    lay = tp.layout_for(N)
    zsh = tp.PairShard(SH.shard_rows(z16, lay, 0).contiguous().clone(), lay)
    assert not torch.is_autocast_enabled()                                                   # the failing regime: autocast OFF (prepare_cache's decorator), a bf16 shard, an fp32 head
    with torch.no_grad():
        cr = tp.distogram_tp(dh, zsh)
    fn = tp.head_rows_fn(dh.linear, zsh.t)
    probe = fn(zsh.t[0:1, 0:2, :])
    return SH.gather_rows(cr.rows.contiguous(), lay, 0), str(probe.dtype), fn.word, EV.schedule()["disto_gemm"]


BF16_GEMM_TOL = 2e-2      # CPU autocast bf16 GEMMs (oneDNN) round the fp32 accumulator to bf16 per output element; the whole-z GEMM and a row-block GEMM
                          # block the reduction differently, so a few logits land one bf16 ulp (2^-8 relative) apart and the 64-bin softmax sums that are the
                          # contact probabilities differ by up to ~6e-3 (measured) — the same shape dependence the n_gpu 1 path has between two batch shapes


@pytest.mark.parametrize("P", [2, 3])
def test_head_callbacks_run_bf16_shard_row_blocks_in_the_single_gpu_paths_arithmetic(P):
    """A bf16 trunk shard (the run under the runner's bf16 autocast) meets fp32 head weights with autocast OFF at roll-out entry
    (DiffusionConditioning.prepare_cache runs under autocasting_disable_decorator, which casts tensor arguments only — never a PairShard):
    the distogram / contact rows are computed per row block UNDER AUTOCAST OF THE SHARD'S DTYPE (tp.head_rows_fn) — exactly the ambient regime
    in which the stock computes `distogram_head(z)` at protenix.py:576 (bf16 GEMM + bf16 symmetric sum): no dtype clash (F.linear(bf16, fp32)
    raises on CPU as on GPU), no whole-shard cast, census disto_gemm=autocast:bfloat16, and the contact rows equal the dense statement computed
    under the same autocast to BF16_GEMM_TOL (why: see the constant). An fp32 shard runs plain (disto word fp32) and equals the dense fp32
    statement to TOL (test_distogram_confidence_and_summaries_equal_dense)."""
    torch.manual_seed(5)
    dh, _ = _heads()
    z16 = torch.randn(N, N, CZ).to(torch.bfloat16)
    cfg = _configs()
    tp.STATE["model"] = types.SimpleNamespace(configs=cfg, distogram_head=dh)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):     # the stock: distogram_head(z) under enable_amp (runner/inference.py:220-227)
        logits = dh(z16)
    assert logits.dtype == torch.bfloat16
    contact_ref = SC.compute_contact_prob(distogram_logits=logits.float(), **SC.get_bin_params(cfg.loss.distogram))
    outs = T.run_ranks(P, _disto_bf16, dh, z16)
    for r, (contact_t, probe_dtype, word, census) in enumerate(outs):
        assert probe_dtype == "torch.bfloat16" and word == census == "autocast:bfloat16", (probe_dtype, word, census)
        _close(contact_t, contact_ref, tol=BF16_GEMM_TOL, what=f"bf16-shard contact rows rank{r} P={P}")
    fn32 = tp.head_rows_fn(dh.linear, torch.zeros(2, 2, CZ))                                 # an fp32 shard: plain, the weight's dtype, no autocast
    assert fn32(torch.zeros(1, 2, CZ)).dtype == torch.float32 and fn32.word == "fp32"


def test_bf16_trunk_shard_through_roll_out_entry_and_confidence_with_fp32_heads(monkeypatch):
    """The whole post-trunk path of a bf16 shard under autocast OFF (the stock's skip_amp regime: conditioning and confidence in fp32) with the
    line's levers on: roll-out entry (distogram rows from the bf16 device shard, park), z_cond rows and every confidence pass cast PER ROW BLOCK
    to fp32 — it runs (no dtype clash anywhere) and equals the dense fp32 statements on the same bf16-valued z."""
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", "1"); monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", "1")
    torch.manual_seed(11)
    S, P = 2, 2
    cfg = _configs(); cfg.sample_diffusion = {"N_sample": S}
    feats = {k: v for k, v in _feats().items() if not k.startswith("template_")}
    dh, ch = _heads()
    cond = _cond()
    s_inputs, s_trunk = torch.randn(N, CSI), torch.randn(N, CS)
    z16 = torch.randn(N, N, CZ).to(torch.bfloat16)
    coords = torch.randn(S, N * 3, 3) * 6.0
    outs = T.run_ranks(P, _conf_tp_policy, dh, ch, cond, feats, s_inputs, s_trunk, z16, coords, cfg, True)
    contact_d, plddt_d, resolved_d, summ_d, full_d = _conf_dense(dh, ch, feats, s_inputs, s_trunk, z16.float(), coords, cfg)   # the fp32 statements on the bf16-valued z
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):     # the stock's distogram regime (ambient autocast) on the bf16 z
        logits16 = dh(z16)
    contact_ref = SC.compute_contact_prob(distogram_logits=logits16.float(), **SC.get_bin_params(cfg.loss.distogram))
    zc_ref = T.run_ranks(P, _cond_tp, cond, feats, z16.float())                              # z_cond rows from an fp32 copy of the same values (the decorator's whole-z cast at n_gpu 1)
    for r in range(P):
        contact_t, plddt_t, resolved_t, summ_t, full_t, rec, st_bytes, zc_t, sched = outs[r]
        _close(contact_t, contact_ref, tol=BF16_GEMM_TOL, what=f"contact rank{r}")
        assert sched["disto_gemm"] == "autocast:bfloat16"
        _close(plddt_t, plddt_d, what="plddt"); _close(resolved_t, resolved_d, what="resolved")
        _close(zc_t, zc_ref[r], what=f"z_cond rank{r}")
        if r == 0:
            for i in range(S):
                for k, v in summ_d[i].items():
                    if torch.is_tensor(v):
                        _close(summ_t[i][k].float().cpu(), v.float().cpu(), 1e-4, what=f"summary[{i}].{k}")     # ptm / iptm / chain tables from fp32 PAE rows
                for k in ("token_pair_pae", "token_pair_pde", "atom_plddt"):
                    _close(full_t[i][k].float().cpu(), full_d[i][k].float().cpu(), 3e-2 if k != "atom_plddt" else 1e-4, what=f"full[{i}].{k}")   # rank 0's [N, N] writer planes are fp16 by design
                _close(full_t[i]["contact_probs"].float().cpu(), contact_ref, 3e-2, what=f"full[{i}].contact_probs")
        assert all(w == "parked:host" for w in rec["words"]) and sched["conf_dtype"] == "float32" and sched["zcond_dtype"] == "float32" and st_bytes == 0
    tp.STATE["contact_rows"].clear()


def test_no_head_callback_hands_the_core_a_bare_module_on_shard_rows():
    """Source census over tp.py: every rows callback the core applies to pair-shard row blocks (heads.sym_logit_rows / logit_rows / embed_rows,
    diffusion.pair_cond_rows) is a function of this module that casts or autocasts the BLOCK (head_rows_fn, zrows_fn, embed_fn) — never a bare
    `head.linear`-style module whose weight dtype would meet a bf16 shard row unconverted."""
    import re
    src = open(tp.__file__).read()
    calls = re.findall(r"(?:sym_logit_rows|(?<![A-Za-z_])logit_rows|embed_rows|pair_cond_rows)\(\s*([^,\n]+),", src)
    assert calls, "no core rows-callback call sites found (the census pattern is stale)"
    bare = [c for c in calls if re.match(r"^[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9.]*$", c.strip()) and not c.strip().startswith("tp.")]
    assert bare == [], f"bare module callbacks on shard rows: {bare}"


# ---------------------------------------------------------------------------------------------------------------- diffusion conditioning rows + DiT
def _cond():
    return _randomize(DM.DiffusionConditioning(sigma_data=16.0, c_z=CZ, c_s=CS, c_s_inputs=CSI, c_noise_embedding=32), 30, 0.2)


def _cond_tp(rank, P, cond, feats, z):
    tp.STATE["chunk"] = CHUNK
    lay = tp.layout_for(N)
    zsh = tp.PairShard(SH.shard_rows(z, lay, 0).contiguous().clone(), lay)
    f = dict(feats); rr = tp.RelpRows(cond.relpe, f)
    with torch.no_grad():
        out = tp.prepare_cache_cond_tp(cond, rr, zsh, False)
    return _gather(out.t, lay)


@pytest.mark.parametrize("P", [2, 3])
def test_diffusion_conditioning_rows_equal_dense(P):
    torch.manual_seed(5)
    cond = _cond()
    feats = _feats(); z = torch.randn(N, N, CZ)
    plane = EM.RelativePositionEncoding.generate_relp(cond.relpe, dict(feats))["relp"]
    with torch.no_grad():
        dense = DM.DiffusionConditioning.prepare_cache.__wrapped__(cond, plane, z, False) if hasattr(DM.DiffusionConditioning.prepare_cache, "__wrapped__") \
            else cond.prepare_cache(plane, z, False)
    for r, got in enumerate(T.run_ranks(P, _cond_tp, cond, feats, z)):
        _close(got, dense, what=f"z_cond rank{r}")


def _dit_tp(rank, P, dt, a, s, z, fusion, normalize):
    tp.STATE["chunk"] = CHUNK
    lay = tp.layout_for(N)
    z_loc = SH.shard_rows(z, lay, 0).contiguous().clone()
    if fusion:
        z_loc = normalize(z_loc)
    blocks = [tp.dit_fns(b, fusion) for b in dt.blocks]
    with torch.no_grad():
        return DF.diffusion_transformer_sharded(blocks, a.clone(), s, z_loc, lay)


@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("fusion", [False, True])
def test_diffusion_transformer_local_queries_equal_dense(P, fusion):
    torch.manual_seed(3)
    dt = _randomize(TF.DiffusionTransformer(c_a=8, c_s=CS, c_z=CZ, n_blocks=2, n_heads=2), 40, 0.2)
    normalize = DM.DiffusionModule(c_z=CZ, c_s=CS, c_s_inputs=CSI, c_token=8, c_atom=8, c_atompair=4, atom_encoder={"n_blocks": 1, "n_heads": 2},
                                   transformer={"n_blocks": 1, "n_heads": 2, "drop_path_rate": 0}, atom_decoder={"n_blocks": 1, "n_heads": 2}).normalize
    a, s, z = torch.randn(2, N, 8), torch.randn(2, N, CS), torch.randn(N, N, CZ)
    with torch.no_grad():
        zz = permute_final_dims(normalize(z), [2, 0, 1]).contiguous().unsqueeze(0) if fusion else z.unsqueeze(0)
        dense = dt(a=a.clone(), s=s, z=zz, inplace_safe=False, chunk_size=None, enable_efficient_fusion=fusion)
    for r, got in enumerate(T.run_ranks(P, _dit_tp, dt, a, s, z, fusion, normalize)):
        _close(got, dense, 5e-5, what=f"DiT rank{r} fusion={fusion}")


# ---------------------------------------------------------------------------------------------------------------- atom-encoder token-pair band
def _band_tp(rank, P, proj, z, a2t, nq, nk):
    lay = tp.layout_for(N)
    z_loc = SH.shard_rows(z, lay, 0).contiguous().clone()
    idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(a2t, a2t, dim_q=-1, dim_k=-1, n_queries=nq, n_keys=nk, compute_mask=False)
    q3, k3 = idx_q.long().reshape(1, *idx_q.shape[-2:]), idx_k.long().reshape(1, *idx_k.shape[-2:])
    valid = torch.ones(q3.shape + (k3.shape[-1],), dtype=torch.bool)
    plan = DF.band_plan(q3, k3, valid, lay.N)
    with torch.no_grad():
        band, extras = DF.pair_band_rows(proj, z_loc, lay, plan)
        return DF.band_lookup(band, extras, plan)[0]


@pytest.mark.parametrize("P", [2])
def test_atom_pair_band_equals_the_stock_window_gather(P):
    torch.manual_seed(9)
    tp.STATE["chunk"] = CHUNK
    z = torch.randn(N, N, CZ)
    proj = _randomize(torch.nn.Sequential(torch.nn.LayerNorm(CZ), torch.nn.Linear(CZ, 4, bias=False)), 50)
    a2t = torch.arange(N * 3) // 3
    with torch.no_grad():
        dense = broadcast_token_to_local_atom_pair(z_token=proj(z), atom_to_token_idx=a2t, n_queries=8, n_keys=16, compute_mask=False)[0]
    for r, got in enumerate(T.run_ranks(P, _band_tp, proj, z, a2t, 8, 16)):
        _close(got, dense, what=f"band rank{r}")


# ---------------------------------------------------------------------------------------------------------------- refusals by name
def test_layout_refuses_at_world_one():
    with pytest.raises(tp.TPRefused):
        tp.layout_for(N)                                                                    # outside a group: world 1


def test_synced_denoiser_makes_rank_trajectories_identical_by_construction():
    """The sampler loop's state IN and the denoiser's update OUT pass through tp.sync (tp.synced_denoiser): under bcast (det 0) every rank's
    trajectory equals rank 0's EXACTLY although each rank's denoiser output carries rank-dependent recompute drift (emulated) — the exit
    spread is 0.0 at any step count; syncing the state IN alone leaves an O(drift x sigma) spread; under guard (det 1) the same drift is a
    defect refused by name. P is not parametrized here (unlike its neighbors above): this path only calls tp.sync/synced_denoiser, which
    bottom out in ThreadHub bcast_/allreduce_/allgather_obj -- a fixed loop over range(P) with no rank-position branching -- unlike the
    ring-transpose / all-to-all collectives the row-sharding tests above exercise, so P=3 would run the same code path once more, not a
    different one; P=2 is representative."""
    from opt_core.mem.rowpair import dist as D
    P = 2

    def rank_fn(rank, P_, mode, sync_out):
        tp.SYNC["mode"] = mode                                                              # module-global; every thread writes the same value
        gen = torch.Generator().manual_seed(7)                                             # identical per-rank generators = the synced RNG state of the real sampler
        def body(self, r_noisy, t_hat):                                                    # a denoiser with per-rank recompute drift (what non-det kernels do)
            return 0.9 * r_noisy + 1e-3 * rank * torch.ones_like(r_noisy) * t_hat
        if sync_out:
            den = tp.synced_denoiser(body)
        else:
            def den(self, r_noisy, t_hat):                                                 # the state-IN-only form (what tree v6 did)
                return body(self, tp.sync(r_noisy, "diffusion_state"), t_hat)
        sigmas = [160.0, 120.0, 60.0, 20.0, 4.0, 0.5, 0.0]                                  # a short EDM-like schedule (large first steps, as reach_probe's 2-step sampler)
        x = sigmas[0] * torch.randn(1, 64, 3, generator=gen)
        for s_cur, s_next in zip(sigmas[:-1], sigmas[1:]):
            x_noisy = x + 0.1 * s_cur * torch.randn(x.shape, generator=gen)
            x_den = den(None, r_noisy=x_noisy, t_hat=torch.tensor(s_cur))
            x = x_noisy + (s_next - s_cur) * (x_noisy - x_den) / max(s_cur, 1e-4)
        hi = x.clone(); lo = x.clone()
        D.allreduce_(hi, op="max"); D.allreduce_(lo, op="min")
        return float((hi - lo).abs().max())

    spread = T.run_ranks(P, rank_fn, "bcast", True)
    assert all(s == 0.0 for s in spread), spread                                           # identical BY CONSTRUCTION (not merely small)
    drift = T.run_ranks(P, rank_fn, "bcast", False)
    assert min(drift) > 0.0, drift                                                          # state-in sync alone: the ranks' own trajectories differ (sigma x drift) — not identical
    with pytest.raises(Exception) as ei:                                                    # det 1 (guard): the drift is a real defect, refused by name
        T.run_ranks(P, rank_fn, "guard", True)
    assert "diffusion_update" in str(ei.value), str(ei.value)[:300]
    tp.SYNC["mode"] = "guard"


def test_fp16_autocast_is_a_note_and_the_check_proceeds(capsys):
    """fp16 autocast under n_gpu>1 is NAMED once (`NOTE fp16 autocast …`) and the trunk-input check proceeds; bf16 / fp32 print nothing."""
    import types
    model = types.SimpleNamespace(training=False, constraint_embedder=None)
    feats = {"relp": tp.RelpRows.__new__(tp.RelpRows)}
    s_inputs = torch.zeros(8, 4)
    tp._NOTED.discard("fp16")
    prev_on, prev_dt = torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
    try:
        torch.set_autocast_gpu_dtype(torch.bfloat16); torch.set_autocast_enabled(True)
        tp._check_trunk_inputs(model, feats, s_inputs, False)
        assert "NOTE" not in capsys.readouterr().err
        torch.set_autocast_gpu_dtype(torch.float16)
        tp._check_trunk_inputs(model, feats, s_inputs, False)                              # proceeds: no TPRefused
        tp._check_trunk_inputs(model, feats, s_inputs, False)                              # the NOTE prints once per process
        err = capsys.readouterr().err
        assert err.count("[protenix-v1-opt] NOTE fp16 autocast under n_gpu>1: the line's numerics and memory statements were established on bf16 / fp32 only; proceeding in fp16") == 1, err
    finally:
        torch.set_autocast_enabled(prev_on); torch.set_autocast_gpu_dtype(prev_dt)


def test_a_leading_batch_dim_is_refused_naming_the_domain_and_the_knob():
    import types
    model = types.SimpleNamespace(training=False, constraint_embedder=None)
    with pytest.raises(tp.TPRefused, match=r"one item per fold call under n_gpu>1 .*run the items separately or --n_gpu 1"):
        tp._check_trunk_inputs(model, {"relp": tp.RelpRows.__new__(tp.RelpRows)}, torch.zeros(2, 8, 4), False)
