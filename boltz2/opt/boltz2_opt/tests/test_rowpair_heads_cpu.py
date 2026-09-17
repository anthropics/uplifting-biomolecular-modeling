"""Seams 16-19 of the row-sharded line (boltz2_opt.rowpair_heads, n_gpu > 1) on CPU: boltz 2.2.1's own DiffusionConditioning, AtomDiffusion
(2 sampler steps x 2 samples through the stock DiffusionModule: atom encoder / a 2-layer token DiffusionTransformer / atom decoder) and
ConfidenceModule (1-block Pairformer, ConfidenceHeads) — random weights, small dims, fp32, eval — driven by the three ``HEADS`` entries on P gloo
ranks (the core's launcher, one process per rank) against the STOCK modules called whole on the same replicated inputs in every rank process.
Asserted per rank: the conditioned pair rows, the atom-encoder outputs (its token-pair term through the band), the sampled coordinates, and
every entry of the confidence dict (pae / pde as [N, N] on rank 0's host, own rows elsewhere; ptm / iptm / ligand_iptm / protein_iptm /
pair_chains_iptm; complex_*; plddt) equal the dense statements to fp32 tolerance (max|d| / max|ref| <= 1e-5; torch.equal REPORTED), the head
census counts, and the allocation census: no tensor with two N-sized dims and >= 16 N^2 elements is created on any rank while the heads run
(nothing ``N x N x c``; the pair stack's triangle-attention logits ``[q_rows, H, N, N]`` — the stock chunked statement's transient, bounded by
the query block — are pinned to one query row there so the bound measures the heads). ``--n_gpu 1``: every entry refuses by name.

Skipped by name without the pure-python ``boltz`` model tree (``pip install --no-deps boltz==2.2.1 einops``) or without an opt_core that
carries the 0.4.3 heads / diffusion seams (``opt_core.mem.rowpair.diffusion``)."""
import os
import random
import sys
import types

import numpy as np
import pytest

@pytest.fixture(autouse=True)
def _cpu_ranks_only(monkeypatch):
    """These ranks run on the CPU (gloo) whatever the box carries: the spawned rank processes see no CUDA device, so the core launcher
    never maps rank -> GPU (a box with fewer GPUs than P would otherwise refuse by name)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield


torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.confidencev2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops)")
import opt_core.mem.rowpair.diffusion  # noqa: E402,F401  (the pinned core; an older core FAILS here, never skips)

from boltz2_opt import rowpair  # noqa: E402
from boltz2_opt import rowpair_heads as RH  # noqa: E402

DIMS = dict(token_s=32, token_z=16, atom_s=16, atom_z=8, W=8, H=16, heads=2, dit_layers=2, E=8, bins=64)
STEER = {"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False, "num_particles": 1, "fk_lambda": 4.0,
         "fk_resampling_interval": 3, "num_gd_steps": 0}
TOL = 1e-5


def build(N: int, seed: int = 0):
    """Stock heads (encodersv2 / diffusion_conditioning / diffusionv2 / confidencev2) at small dims + synthetic replicated inputs: trunk s, s_inputs,
    z [1,N,N,token_z], distogram logits, token and atom features (3 chains: two protein, one 3-token ligand; last 3 tokens padding)."""
    from torch import nn
    from boltz.data import const
    from boltz.model.modules.confidencev2 import ConfidenceModule
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    d = DIMS
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed); np.random.seed(seed)                               # boltz initialises Linear weights with scipy's truncnorm (numpy's global RNG): every rank process must build the SAME model
    m = nn.Module()
    m.rel_pos = RelativePositionEncoder(d["token_z"])
    m.diffusion_conditioning = DiffusionConditioning(token_s=d["token_s"], token_z=d["token_z"], atom_s=d["atom_s"], atom_z=d["atom_z"],
                                                     atoms_per_window_queries=d["W"], atoms_per_window_keys=d["H"], atom_encoder_depth=2,
                                                     atom_encoder_heads=2, token_transformer_depth=d["dit_layers"], token_transformer_heads=d["heads"],
                                                     atom_decoder_depth=2, atom_decoder_heads=2, atom_feature_dim=3 + 1 + d["E"],
                                                     conditioning_transition_layers=2, use_no_atom_char=True)
    m.structure_module = AtomDiffusion(score_model_args=dict(token_s=d["token_s"], atom_s=d["atom_s"], atoms_per_window_queries=d["W"],
                                                             atoms_per_window_keys=d["H"], sigma_data=16, dim_fourier=32, atom_encoder_depth=2,
                                                             atom_encoder_heads=2, token_transformer_depth=d["dit_layers"],
                                                             token_transformer_heads=d["heads"], atom_decoder_depth=2, atom_decoder_heads=2,
                                                             conditioning_transition_layers=2), num_sampling_steps=2)
    m.confidence_module = ConfidenceModule(d["token_s"], d["token_z"], pairformer_args=dict(num_blocks=1, num_heads=2, dropout=0.25),
                                           num_dist_bins=64, token_level_confidence=True, max_dist=22, add_s_to_z_prod=True, add_s_input_to_s=True,
                                           add_z_input_to_z=True, maximum_bond_distance=0, bond_type_feature=False,
                                           confidence_args=dict(num_plddt_bins=50, num_pde_bins=d["bins"], num_pae_bins=d["bins"], use_separate_heads=False),
                                           conditioning_cutoff_min=4.0, conditioning_cutoff_max=20.0)
    m.use_kernels = False
    m.steering_args = dict(STEER)
    with torch.no_grad():                                                  # zero / ones initialised parameters randomised so every statement matters
        for p in m.parameters():
            if p.dim() >= 1 and (float(p.abs().sum()) == 0.0 or bool((p == 1).all())):
                p.add_(torch.randn(p.shape, generator=g) * 0.2)
    m.eval()
    # ---- token features: chain 0 = tokens 0:18 protein, chain 1 = 18:34 protein, chain 2 = 34:37 ligand (1 atom each), 37:N padding (N >= 38)
    assert N >= 38, N
    PROT, LIG = const.chain_type_ids["PROTEIN"], const.chain_type_ids["NONPOLYMER"]
    asym = torch.tensor([0] * 18 + [1] * 16 + [2] * (N - 34))[None]
    mol = torch.tensor([PROT] * 34 + [LIG] * (N - 34))[None]
    pad = torch.ones(1, N); pad[0, 37:] = 0.0
    n_at = [3 if (i % 2 == 0) else 2 for i in range(34)] + [1, 1, 1] + [0] * (N - 37)   # atoms per token
    A = sum(n_at)
    NA = ((A + d["W"] - 1) // d["W"]) * d["W"] + d["W"]                                 # padded atom count (a whole extra window of padding)
    atom_to_token = torch.zeros(1, NA, N); token_to_rep_atom = torch.zeros(1, N, NA); frames_idx = torch.zeros(1, N, 3, dtype=torch.long)
    a = 0
    for t, k in enumerate(n_at):
        for j in range(k):
            atom_to_token[0, a + j, t] = 1.0
        if k:
            token_to_rep_atom[0, t, a] = 1.0
            frames_idx[0, t] = torch.tensor([a, a + min(1, k - 1), a + min(2, k - 1)])
        a += k
    atom_pad = torch.zeros(1, NA); atom_pad[0, :A] = 1.0
    feats = {
        "token_pad_mask": pad, "asym_id": asym, "entity_id": (asym > 1).long(), "sym_id": (asym == 1).long(), "mol_type": mol,
        "residue_index": torch.arange(N)[None] % 18, "token_index": torch.arange(N)[None], "cyclic_period": torch.zeros(1, N),
        "token_bonds": (torch.rand(1, N, N, 1, generator=g) < 0.05).float(),
        "contact_conditioning": torch.nn.functional.one_hot(torch.randint(0, len(const.contact_conditioning_info), (1, N, N), generator=g), len(const.contact_conditioning_info)).float(),
        "contact_threshold": 4.0 + 16.0 * torch.rand(1, N, N, generator=g),
        "atom_pad_mask": atom_pad, "atom_to_token": atom_to_token, "token_to_rep_atom": token_to_rep_atom, "frames_idx": frames_idx,
        "ref_pos": torch.randn(1, NA, 3, generator=g) * 3.0 * atom_pad[..., None], "ref_space_uid": atom_to_token.argmax(-1),
        "ref_charge": torch.randint(-1, 2, (1, NA), generator=g).float() * atom_pad,
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, d["E"], (1, NA), generator=g), d["E"]).float() * atom_pad[..., None],
    }
    inp = {"s_inputs": torch.randn(1, N, d["token_s"], generator=g), "s": torch.randn(1, N, d["token_s"], generator=g),
           "z": torch.randn(1, N, N, d["token_z"], generator=g), "pd": torch.randn(1, N, N, 1, d["bins"], generator=g) * 2.0}
    return m, feats, inp


def dense_heads(m, feats, inp, seed: int):
    """The stock statements on the whole tensors (boltz2.py:508-604): DiffusionConditioning, AtomDiffusion.sample (2 steps x 2 samples),
    ConfidenceModule (run_sequentially) — the reference; plus the stock conditioned pair and atom-pair term for the seam-18 checks."""
    s, s_inputs, z, pd = inp["s"], inp["s_inputs"], inp["z"], inp["pd"]
    dc = m.diffusion_conditioning
    relpos = m.rel_pos(feats)
    q, c, to_keys, eb, db, ttb = dc(s_trunk=s, z_trunk=z, relative_position_encoding=relpos, feats=feats)
    cond = {"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": eb, "atom_dec_bias": db, "token_trans_bias": ttb}
    zc = dc.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=relpos)
    _q, _c, p, _ = dc.atom_encoder(feats=feats, s_trunk=s, z=zc)
    torch.manual_seed(seed)
    struct = m.structure_module.sample(s_trunk=s, s_inputs=s_inputs, feats=feats, num_sampling_steps=2, atom_mask=feats["atom_pad_mask"],
                                       multiplicity=2, max_parallel_samples=None, steering_args=m.steering_args, diffusion_conditioning=cond)
    conf = m.confidence_module(s_inputs=s_inputs, s=s, z=z, x_pred=struct["sample_atom_coords"], feats=feats, pred_distogram_logits=pd[:, :, :, 0],
                               multiplicity=2, run_sequentially=True, use_kernels=False)
    return {"cond": cond, "zc": zc, "p": p, "struct": struct, "conf": conf}


class AllocCensus:
    """Every tensor an aten op returns while active: the largest numel, and the ops that produced a tensor with >= 2 dims of size N and
    >= 16 N^2 elements (an ``N x N x c`` tensor; [N, N] scalar planes such as the replicated pair mask are below it)."""

    def __init__(self, N: int):
        from torch.utils._python_dispatch import TorchDispatchMode
        cen = self
        self.N, self.max_numel, self.max_shape, self.offenders = int(N), 0, (), []

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in (out if isinstance(out, (list, tuple)) else (out,)):
                    if isinstance(t, torch.Tensor):
                        n = t.numel()
                        if n > cen.max_numel:
                            cen.max_numel, cen.max_shape = n, tuple(t.shape)
                        if sum(1 for s_ in t.shape if int(s_) == cen.N) >= 2 and n >= 16 * cen.N * cen.N:
                            cen.offenders.append((str(func), tuple(t.shape)))
                return out
        self.mode = _Mode()

    def __enter__(self):
        self.mode.__enter__(); return self

    def __exit__(self, *a):
        return self.mode.__exit__(*a)


def _rel(a, b) -> float:
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


def _abs(a, b) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


def _entry(N: int, census: bool = False):
    """One rank: the dense reference first (stock class forwards untouched), then the adapter installed, a TrunkShard cut from the SAME dense
    trunk tensors, and the three heads on the rows."""
    torch.set_num_threads(1)
    m, feats, inp = build(N)
    with torch.no_grad():
        ref = dense_heads(m, feats, inp, seed=11)
        os.environ[rowpair.ENV_P] = os.environ["ROWPAIR_WORLD"]
        assert rowpair.apply() == ["rowpair_tp"]
        fake = types.ModuleType("fake_boltz2_model_module")
        fake.Boltz2 = type("Boltz2", (), {"forward": lambda self, feats: None})
        rowpair._install_trunk(fake)
        assert set(k for k, v in rowpair.HEADS.items() if v is not None) == set(RH.HEADS), rowpair.HEADS      # bound at install
        lay = rowpair._layout(N)
        r0, r1 = int(lay.r0), int(lay.r1)
        rowpair._pair_planes_to_host(feats)
        mask = feats["token_pad_mask"]
        T = rowpair.TrunkShard(inp["s_inputs"], inp["s"], inp["z"][:, r0:r1].clone(), inp["pd"][:, r0:r1].clone(), mask,
                               mask[:, :, None] * mask[:, None, :], lay, feats)
        T.zplan = rowpair._ztrunk_plan(T, passes=2)                             # the trunk shard's placement plan, bound as _forward_sharded binds it (no word set here: resident)
        cen = AllocCensus(N) if census else None
        if cen is not None:
            os.environ["ROWPAIR_TRIATT_QBLOCK"] = "1"                           # the confidence Pairformer's triangle attention (the adapter's pair-stack driver, seams 8-13)
            cen.__enter__()                                                     # forms the stock chunked logits [q_rows, H, N, N]: pinned to 1 query row so the bound below
        try:                                                                    # measures what seams 16-19 allocate
            cond = RH.diffusion_conditioning_rows(m, T)
            sample_kw = dict(s_trunk=inp["s"], s_inputs=inp["s_inputs"], feats=feats, num_sampling_steps=2, atom_mask=feats["atom_pad_mask"],
                             multiplicity=2, max_parallel_samples=None, steering_args=m.steering_args, diffusion_conditioning=cond)
            zc = cond["token_trans_bias"]                                        # seam 18's rows, read BEFORE the roll-out (ROWPAIR_ZCOND_RELEASE default on)
            zc_rows = zc.z_loc.clone()                                              # sample_sharded releases the conditioned pair rows at its exit (their last reader on the predict line)
            torch.manual_seed(11)
            struct = RH.sample_sharded(m, T, **sample_kw)
            conf = RH.confidence_rows(m, T, ref["struct"]["sample_atom_coords"], 2, True)     # the reference coordinates: seams 16-17 isolated
        finally:
            if cen is not None:
                cen.__exit__(None, None, None)
        # ---- seam 18
        assert isinstance(zc, RH.ZCondRows) and tuple(zc_rows.shape) == (1, r1 - r0, N, DIMS["token_z"])
        assert zc.z_loc.numel() == 0, "sample_sharded left the conditioned pair rows allocated (ROWPAIR_ZCOND_RELEASE default: released at the roll-out's exit)"
        diffs = {"z_cond_rows": _rel(zc_rows, ref["zc"][:, r0:r1]), "q": _rel(cond["q"], ref["cond"]["q"]), "c": _rel(cond["c"], ref["cond"]["c"]),
                 "atom_enc_bias": _rel(cond["atom_enc_bias"], ref["cond"]["atom_enc_bias"]), "atom_dec_bias": _rel(cond["atom_dec_bias"], ref["cond"]["atom_dec_bias"])}
        equal = {"z_cond_rows": bool(torch.equal(zc_rows, ref["zc"][:, r0:r1])), "atom_enc_bias": bool(torch.equal(cond["atom_enc_bias"], ref["cond"]["atom_enc_bias"]))}
        # the band term alone: the stock einsum against the projection of the STOCK z_cond vs the encoder's p difference is inside atom_*_bias; check
        # the lookup mechanism bit-exactly on identical inputs: band of the reference z_cond rows vs the stock einsum
        C = RH._core()
        enc = m.diffusion_conditioning.atom_encoder
        _q2, _c2, p_rows, _ = RH._atom_encoder_banded(enc, feats, inp["s"], ref["zc"][:, r0:r1].contiguous(), lay, C, None)
        diffs["p_band_vs_einsum"] = _abs(p_rows, ref["p"]); equal["p_band_vs_einsum"] = bool(torch.equal(p_rows, ref["p"]))
        # ---- seam 19
        diffs["sample_atom_coords"] = _rel(struct["sample_atom_coords"], ref["struct"]["sample_atom_coords"])
        equal["sample_atom_coords"] = bool(torch.equal(struct["sample_atom_coords"], ref["struct"]["sample_atom_coords"]))
        # ---- seams 16-17
        cr = ref["conf"]
        for k in ("ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde", "plddt"):
            assert tuple(conf[k].shape) == tuple(cr[k].shape), (k, tuple(conf[k].shape), tuple(cr[k].shape))
            diffs["conf." + k] = _abs(conf[k], cr[k]) if k != "plddt" else _rel(conf[k], cr[k])
        pc = conf["pair_chains_iptm"]; pcr = cr["pair_chains_iptm"]
        assert {a: sorted(pc[a]) for a in pc} == {a: sorted(pcr[a]) for a in pcr}, (list(pc), list(pcr))
        diffs["conf.pair_chains_iptm"] = max(_abs(pc[a][b], pcr[a][b]) for a in pcr for b in pcr[a])
        if lay.rank == 0:                                                                       # the [S, N, N] matrices on rank 0's host
            assert conf["pae"].device.type == "cpu" and tuple(conf["pae"].shape) == (2, N, N), (conf["pae"].device, tuple(conf["pae"].shape))
            diffs["conf.pae[N,N]@rank0"] = _rel(conf["pae"], cr["pae"]); diffs["conf.pde[N,N]@rank0"] = _rel(conf["pde"], cr["pde"])
            equal["conf.pae"] = bool(torch.equal(conf["pae"], cr["pae"]))
        else:                                                                                   # own rows elsewhere (named)
            assert tuple(conf["pae"].shape) == (2, r1 - r0, N)
            diffs["conf.pae_rows"] = _rel(conf["pae"], cr["pae"][:, r0:r1]); diffs["conf.pde_rows"] = _rel(conf["pde"], cr["pde"][:, r0:r1])
        assert "pae_logits" not in conf and "pde_logits" not in conf
        calls = dict(rowpair._STATE["calls"])
        rep = rowpair.report()
    # tolerance classes: everything <= 1e-5 relative / absolute (GEMMs differ only in M; the gPDE / pTM sums are reordered row sums)
    bad = {k: v for k, v in diffs.items() if v > TOL}
    assert not bad, ("sharded heads vs stock modules (max diff)", bad, diffs, equal)
    expect = {"zcond_blocks": (1, None), "dit_transformer_calls": (2 * 1, None), "dit_layers_sharded": (2 * DIMS["dit_layers"], None), "conf_samples": (2, 2),
              "conf_transposes": (2, 2), "sample_coords_rank0": (1, 1), "denoiser_state_bcast": (1, None), "conf_pae_blocks": (2, None), "conf_pde_blocks": (2, None)}
    badc = {k: (calls.get(k), lo) for k, (lo, hi) in expect.items() if calls.get(k) is None or calls.get(k) < lo or (hi is not None and calls.get(k) > hi)}
    assert not badc, ("head census (got, expected>=)", badc, calls)
    assert calls.get("band_W") is not None and 0 <= calls["band_W"] < N and calls.get("band_extra_rows") == 0, calls
    out = {"rank": rep["rank"], "P": rep["n_gpu"], "N": N, "rows": (r0, r1), "diffs": diffs, "equal": equal, "calls": {k: calls.get(k) for k in expect} | {"band_W": calls.get("band_W")},
           "schedule": rep.get("schedule")}
    if cen is not None:
        out["alloc"] = {"max_numel": cen.max_numel, "max_shape": cen.max_shape, "bound": 16 * N * N, "offenders": cen.offenders[:8]}
        assert not cen.offenders, ("an N x N x c tensor was created on this rank", cen.offenders[:8], cen.max_shape)
    return out


@pytest.mark.parametrize("P,N", [(2, 40), (3, 40), (2, 41)])   # (3, 40) and (2, 41): ragged rows, N % P != 0
def test_heads_on_rows_equal_stock_modules(P, N):
    from opt_core.mem.rowpair import launch
    out = launch.run_sharded(P, _entry, N, False, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    print("ROWPAIR_HEADS", out)
    assert out["P"] == P and all(v <= TOL for v in out["diffs"].values())


def test_alloc_census_nothing_pair_shaped_whole():
    from opt_core.mem.rowpair import launch
    out = launch.run_sharded(2, _entry, 40, True, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    print("ROWPAIR_HEADS_ALLOC", out.get("alloc"))
    assert out["alloc"]["offenders"] == []


def test_p1_refuses_by_name():
    """--n_gpu 1: the adapter installs nothing and every HEADS entry refuses by name (the engine's own modules run)."""
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as D
    rowpair.reset_for_tests()
    N = 24
    lay = D.default_ctx(N)                                                                        # no group: the P == 1 layout
    assert lay.P == 1
    T = rowpair.TrunkShard(None, None, torch.zeros(1, N, N, 4), torch.zeros(1, N, N, 1, 8), None, None, lay, {})
    for name, fn, args in (("diffusion_conditioning_rows", RH.diffusion_conditioning_rows, ()), ("sample_sharded", RH.sample_sharded, ()),
                           ("confidence_rows", RH.confidence_rows, (None, 1, True))):
        with pytest.raises((RowpairRefused, rowpair.Refused), match="n_gpu=1"):
            fn(object(), T, *args)
    assert set(RH.HEADS) == {"diffusion_conditioning_rows", "sample_sharded", "confidence_rows"} and all(callable(f) for f in RH.HEADS.values())


def test_zcondrows_stands_in_for_the_bias_tensor():
    zc = RH.ZCondRows(torch.zeros(1, 2, 4, 3), None, [], None, 3, 6)
    assert zc.float() is zc
    rowpair.reset_for_tests()
    with pytest.raises(rowpair.Refused, match="ZCondRows.shape"):
        zc.shape
