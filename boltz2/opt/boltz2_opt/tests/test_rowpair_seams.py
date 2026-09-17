"""The row-sharded trunk (boltz2_opt.rowpair, n_gpu > 1) on CPU: boltz 2.2.1's own trunk modules (random weights, small dims, fp32, eval) driven
by ``rowpair._trunk_sharded`` on P gloo ranks (the core's launcher, one process per rank) against the DENSE stock statements
(boltz2.py:414-491 verbatim on the whole pair tensor) built identically in every rank process. Asserted per rank: the gathered rows equal the
dense z / s / distogram logits (fp32, max|d|/max|ref| <= 1e-4 — GEMMs differ only in M), and the seam census: z born as rows
(``trunk_rows``), no gather inside the trunk (``gathers_named == 0`` until the test's own gather), the template / MSA-block / Pairformer
stacks presharded, the distogram on rows, the confidence / z_cond row seams at their pending count (0, named). Two shapes: N below and
above boltz's chunk threshold (384: the MSA module's chunked PWA / OPM / triangle-attention statements).

Skipped by name without the pure-python ``boltz`` model tree (``pip install --no-deps boltz==2.2.1 einops``) or without an opt_core that
carries the 0.4.3 row-sharding surface (``opt_core.mem.rowpair.msa``)."""
import os
import sys
import types

import pytest

@pytest.fixture(autouse=True)
def _cpu_ranks_only(monkeypatch):
    """These ranks run on the CPU (gloo) whatever the box carries: the spawned rank processes see no CUDA device, so the core launcher
    never maps rank -> GPU (a box with fewer GPUs than P would otherwise refuse by name)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield


torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.trunkv2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops)")
pytest.importorskip("opt_core.mem.rowpair.msa", reason="opt_core >= 0.4.3 (opt_core.mem.rowpair.msa) is required")

from boltz2_opt import rowpair  # noqa: E402

DIMS = dict(Ds=24, token_s=32, token_z=16, msa_s=8, S=6, S_sub=4, T=1, tdim=8)


def build(N: int, seed: int = 0, subsample: bool = True, templates: bool = True):
    """A Boltz2-shaped trunk of stock modules (trunkv2 / pairformer / encodersv2) + synthetic feats, fp32, eval, seeded."""
    import random, numpy as np                                        # boltz's constructors draw initial weights from NUMPY's global RNG (initialize.trunc_normal_init_ ->
    random.seed(seed); np.random.seed(seed)                              # scipy truncnorm.rvs): every rank process must build the SAME model (torch is seeded below)
    from torch import nn
    from boltz.data import const
    from boltz.model.layers.pairformer import PairformerModule
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    from boltz.model.modules.trunkv2 import ContactConditioning, DistogramModule, MSAModule, TemplateV2Module
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = DIMS
    m = nn.Module()
    m.s_init = nn.Linear(d["Ds"], d["token_s"], bias=False)
    m.z_init_1 = nn.Linear(d["Ds"], d["token_z"], bias=False)
    m.z_init_2 = nn.Linear(d["Ds"], d["token_z"], bias=False)
    m.rel_pos = RelativePositionEncoder(d["token_z"])
    m.token_bonds = nn.Linear(1, d["token_z"], bias=False)
    m.bond_type_feature = False
    m.contact_conditioning = ContactConditioning(token_z=d["token_z"], cutoff_min=4.0, cutoff_max=20.0)
    m.s_norm = nn.LayerNorm(d["token_s"]); m.z_norm = nn.LayerNorm(d["token_z"])
    m.s_recycle = nn.Linear(d["token_s"], d["token_s"], bias=False); m.z_recycle = nn.Linear(d["token_z"], d["token_z"], bias=False)
    m.use_templates = templates; m.is_template_compiled = False; m.is_msa_compiled = False; m.is_pairformer_compiled = False
    m.template_module = TemplateV2Module(d["token_z"], d["tdim"], 1)
    m.msa_module = MSAModule(msa_s=d["msa_s"], token_z=d["token_z"], token_s=d["Ds"], msa_blocks=2, msa_dropout=0.15, z_dropout=0.25,
                             use_paired_feature=True, subsample_msa=subsample, num_subsampled_msa=d["S_sub"])
    m.pairformer_module = PairformerModule(d["token_s"], d["token_z"], 2, v2=True)
    m.distogram_module = DistogramModule(d["token_z"], 10)
    m.run_trunk_and_structure = True; m.use_kernels = False
    with torch.no_grad():                                                  # LayerNorm affine + zero-init output projections randomised so every statement matters
        for p in m.parameters():
            if p.dim() >= 1 and (float(p.abs().sum()) == 0.0 or bool((p == 1).all())):
                p.add_(torch.randn(p.shape, generator=g) * 0.2)
    m.eval()
    S, T = d["S"], d["T"]
    n_chain = 2
    asym = torch.arange(N)[None] // ((N + n_chain - 1) // n_chain)
    feats = {
        "s_inputs": torch.randn(1, N, d["Ds"], generator=g),
        "token_pad_mask": torch.ones(1, N), "token_bonds": (torch.rand(1, N, N, 1, generator=g) < 0.05).float(),
        "contact_conditioning": torch.nn.functional.one_hot(torch.randint(0, len(const.contact_conditioning_info), (1, N, N), generator=g), len(const.contact_conditioning_info)).float(),
        "contact_threshold": 4.0 + 16.0 * torch.rand(1, N, N, generator=g),
        "asym_id": asym, "entity_id": asym % 2, "sym_id": asym // 2, "residue_index": torch.arange(N)[None] % 50, "token_index": torch.arange(N)[None],
        "msa": torch.randint(0, const.num_tokens, (1, S, N), generator=g), "has_deletion": (torch.rand(1, S, N, generator=g) < 0.1).float(),
        "deletion_value": torch.rand(1, S, N, generator=g), "msa_paired": (torch.rand(1, S, N, generator=g) < 0.5).float(), "msa_mask": (torch.rand(1, S, N, generator=g) < 0.9).float(),   # zeros in the MSA mask: the OPM pair count (whole mask) vs its masked operands (this rank's columns) are value-tested

        "template_restype": torch.nn.functional.one_hot(torch.randint(0, const.num_tokens, (1, T, N), generator=g), const.num_tokens).float(),
        "template_frame_rot": torch.linalg.qr(torch.randn(1, T, N, 3, 3, generator=g))[0], "template_frame_t": torch.randn(1, T, N, 3, generator=g) * 5,
        "template_mask_frame": (torch.rand(1, T, N, generator=g) < 0.8).float(), "template_cb": torch.randn(1, T, N, 3, generator=g) * 8,
        "template_ca": torch.randn(1, T, N, 3, generator=g) * 8, "template_mask_cb": (torch.rand(1, T, N, generator=g) < 0.9).float(),
        "visibility_ids": asym[:, None].expand(1, T, N).clone(), "template_mask": (torch.rand(1, T, N, generator=g) < 0.9).float(),
    }
    feats["token_pad_mask"][0, -3:] = 0.0                                  # padded tokens exercise the masks
    m.input_embedder = lambda f: f["s_inputs"]
    return m, feats


def dense_trunk(model, feats, recycling_steps: int):
    """boltz2.py:414-491, the stock statements on the whole pair tensor (the reference)."""
    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init + relative_position_encoding
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    z_init = z_init + model.contact_conditioning(feats)
    s = torch.zeros_like(s_init); z = torch.zeros_like(z_init)
    mask = feats["token_pad_mask"].float(); pair_mask = mask[:, :, None] * mask[:, None, :]
    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        if model.use_templates:
            z = z + model.template_module(z, feats, pair_mask, use_kernels=False)
        z = z + model.msa_module(z, s_inputs, feats, use_kernels=False)
        s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)
    return s, z, model.distogram_module(z)


def _rank_entry(N: int, recycling_steps: int, variant: str = "full", host: str = "", msa_m: str = "", grids: str = ""):
    """One rank: dense reference first (stock class forwards untouched), then the adapter installed and its sharded trunk; returns the census.
    ``host``: the raw-MSA placement word (``ROWPAIR_MSA_HOST``) the sharded trunk runs under — the feats are parked as the batch hook parks them
    (``rank0``: ranks > 0 hold zero-row placeholders) after the dense reference read them on the device. ``msa_m``: the MSA representation's
    layout word (``ROWPAIR_MSA_M_LAYOUT``; unset = the default, token_sharded)."""
    torch.set_num_threads(1)
    if host:
        os.environ["ROWPAIR_MSA_HOST"] = host
    if msa_m:
        os.environ["ROWPAIR_MSA_M_LAYOUT"] = msa_m
    else:
        os.environ.pop("ROWPAIR_MSA_M_LAYOUT", None)
    for kv in filter(None, grids.split(",")):                               # e.g. "ROWPAIR_PWA_S_CHUNK=4,ROWPAIR_PWA_QBLOCK=7": several S-chunks and query blocks inside one rank's rows
        k, v = kv.split("="); os.environ[k] = v
    model, feats = build(N, subsample=(variant != "nosub"), templates=(variant != "notempl"))
    draws = {"dense": [], "sharded": []}
    _randperm = torch.randperm

    def recording(key):
        def rp(*a, **k):
            out = _randperm(*a, **k); draws[key].append(out.tolist()); return out
        return rp
    with torch.no_grad():
        torch.manual_seed(11)
        torch.randperm = recording("dense")
        s_ref, z_ref, pd_ref = dense_trunk(model, feats, recycling_steps)
        torch.randperm = _randperm
        os.environ[rowpair.ENV_P] = os.environ["ROWPAIR_WORLD"]
        assert rowpair.apply() == ["rowpair_tp"]                           # the model module is not imported here (lightning / rdkit): the after-import hook is armed;
        fake = types.ModuleType("fake_boltz2_model_module")                # the install itself is exercised on a stand-in module object carrying a Boltz2 class
        fake.Boltz2 = type("Boltz2", (), {"forward": lambda self, feats: None})
        rowpair._install_trunk(fake)
        assert fake.Boltz2.forward is rowpair._forward_sharded and rowpair._STATE["trunk_installed"]
        if host:                                                            # the batch hook's parking (rowpair_msa._transfer_batch_to_device -> core park_features), after the group exists
            from boltz2_opt import rowpair_msa
            from opt_core.mem.rowpair import msa_host as MH
            facts = MH.park_features(feats, rowpair_msa.HOST_KEYS, row_dims=rowpair_msa.ROW_DIM)
            assert facts["mode"] == host and (facts["placeholders"] if (host == "rank0" and int(os.environ["ROWPAIR_RANK"]) > 0) else facts["parked"])
        torch.manual_seed(11)
        torch.randperm = recording("sharded")
        s_inputs, s, z_loc, pd_loc, mask, pair_mask, lay = rowpair._trunk_sharded(model, feats, recycling_steps)
        torch.randperm = _randperm
        calls = dict(rowpair._STATE["calls"])
        assert calls["gathers_named"] == 0, calls                          # nothing gathered inside the trunk
        assert tuple(z_loc.shape) == (1, lay.r1 - lay.r0, N, DIMS["token_z"]), (tuple(z_loc.shape), lay.r0, lay.r1)
        z = rowpair._gather_rows(z_loc, lay, 1)
        pd = rowpair._gather_rows(pd_loc, lay, 1)

        def rel(a, b):
            return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))
        diffs = {"z": rel(z, z_ref), "s": rel(s, s_ref), "pdistogram": rel(pd, pd_ref), "z_equal": bool(torch.equal(z, z_ref))}
        rep = rowpair.report()
    cyc = recycling_steps + 1
    tp = 1 if model.use_templates else 0
    expect = {"trunk_rows": 1, "recycles": cyc, "template_rows": cyc * tp, "msa_blocks_rows": 2 * cyc, "presharded_calls": cyc * (tp + 2 + 1), "distogram_rows": 1,
              "conf_rows": 0, "zcond_rows": 0, "gathers_in_trunk": 0, "pwa_rows": 2 * cyc, "opm_rows": 2 * cyc, "layers": cyc * (tp + 2 + 2)}
    bad = {k: (calls.get(k), v) for k, v in expect.items() if calls.get(k) != v}
    assert not bad, f"seam census mismatch (got, expected): {bad}"
    assert all(v <= 1e-4 for k, v in diffs.items() if k != "z_equal"), (variant, diffs, "randperm draws equal:", draws["dense"] == draws["sharded"], draws)
    assert rep["installed"] and rep["trunk_installed"] and rep["guards"]["replicated_checked"] == (cyc if model.msa_module.subsample_msa else 0) + (1 if host else 0), rep["guards"]   # the draw per cycle (+ the msa_mask guard once per batch under a host word)
    assert set(rep["heads_bound"]) == {"diffusion_conditioning_rows", "sample_sharded", "confidence_rows"}, rep["heads_bound"]
    sched = rep["schedule"]                                                 # seam 4's placement census: the word in force, rows used of S, cycles served
    n_rows = DIMS["S_sub"] if model.msa_module.subsample_msa else DIMS["S"]
    assert sched.get("msa_host") == (host or "off") and sched.get("msa_rows") == f"{n_rows}/{DIMS['S']}", (host, sched.get("msa_host"), sched.get("msa_rows"))
    assert (sched.get("msa_host_cycles") == cyc) if host else ("msa_host_cycles" not in sched), sched.get("msa_host_cycles")
    want_m = msa_m or "token_sharded"                                       # the MSA representation's layout census: the word, this rank's columns, the OPM's local a
    assert sched.get("msa_m") == want_m, (want_m, sched.get("msa_m"))
    if want_m == "token_sharded":
        assert sched.get("msa_cols") == f"{lay.r0}:{lay.r1}/{N}" and sched.get("opm_a") == "local" and "pwa_s_chunk" in sched, (sched.get("msa_cols"), sched.get("opm_a"), sorted(sched))
        assert sched.get("msa_transition") == "token_sharded" and float(sched.get("opm_b_gather_gib", -1)) >= 0.0, (sched.get("msa_transition"), sched.get("opm_b_gather_gib"))
    else:
        assert sched.get("msa_cols") == "all" and "opm_a" not in sched, (sched.get("msa_cols"), sched.get("opm_a"))
    return {"rank": rep["rank"], "P": rep["n_gpu"], "N": N, "rows": (lay.r0, lay.r1), "diffs": diffs, "calls": calls, "schedule": rep["schedule"]}


@pytest.mark.parametrize("P,N,variant,host,msa_m", [(2, 40, "nosub", "", ""), (2, 40, "notempl", "", ""), (2, 40, "full", "", ""), (3, 40, "full", "", ""), (2, 41, "full", "", ""), (3, 41, "full", "", ""), (2, 392, "full", "", ""),   # N=41: ragged rows (N % P != 0) at P=2 and P=3; N=40 @P=3 ragged too; msa_m unset = the token-sharded MSA representation (the default)
                                                   (2, 40, "full", "rank0", ""), (2, 40, "nosub", "rank0", ""), (3, 41, "full", "all", ""),   # the raw MSA host-resident (ROWPAIR_MSA_HOST): rank0 = rank 0 stages the cycle's rows and broadcasts them (subsample on / off), all = every rank its copy
                                                   (2, 40, "full", "", "replicated"), (3, 41, "nosub", "", "replicated"), (2, 392, "full", "", "replicated"), (2, 40, "nosub", "rank0", "replicated")])   # ROWPAIR_MSA_M_LAYOUT=replicated: m whole on every rank (ragged / above the chunk threshold / under the host word)
@pytest.mark.parametrize("grids", ["", "ROWPAIR_PWA_S_CHUNK=4,ROWPAIR_PWA_QBLOCK=7"])   # the production grids in miniature: several gathered S-chunks (S=6 > 4) and query-row blocks (7 < R) per rank
def test_born_sharded_trunk_equals_dense_and_census(P, N, variant, host, msa_m, grids, monkeypatch):
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)
    monkeypatch.delenv("ROWPAIR_MSA_M_LAYOUT", raising=False)
    if grids and (msa_m == "replicated" or N > 41):                         # the grids matter to the token-sharded pass; the N=392 cases already walk several blocks
        pytest.skip("grid variants run on the small token-sharded cases")
    out = launch.run_sharded(P, _rank_entry, N, 1, variant, host, msa_m, grids, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    print("ROWPAIR_SEAMS", out)
    assert out["P"] == P and out["calls"]["gathers_named"] == 0 and out["diffs"]["z"] <= 1e-4


def test_refusals_by_name():
    rowpair.reset_for_tests()
    os.environ.pop(rowpair.ENV_P, None)
    with pytest.raises(rowpair.Refused, match="attached at n_gpu=1"):
        rowpair.apply()
    rep = rowpair.report()
    assert rep["installed"] is False and rep["applied"] == [] and "fpf_trimul" in " ".join(rep["replaced_levers"]) and "mask2" in " ".join(rep["bypassed_levers"])


def _units_entry(N: int, tin: str = ""):
    """One rank: each row statement ALONE against its stock module on the same dense inputs (localises a seam that disagrees). ``tin``: the
    pair block's ending-node transpose form (``ROWPAIR_TRANSPOSE_INPLACE``: ``"1"`` = pairwise block swaps into the shard's own storage, core
    ``ring.transpose_shard_inplace_``; unset = the all-to-all transpose into a second shard)."""
    torch.set_num_threads(1)
    if tin:
        os.environ["ROWPAIR_TRANSPOSE_INPLACE"] = tin
    model, feats = build(N)
    out = {}
    with torch.no_grad():
        os.environ[rowpair.ENV_P] = os.environ["ROWPAIR_WORLD"]
        rowpair.apply()
        lay = rowpair._layout(N)
        r0, r1 = lay.r0, lay.r1
        s_inputs = feats["s_inputs"]; mask = feats["token_pad_mask"].float(); pair_mask = mask[:, :, None] * mask[:, None, :]; pm = pair_mask[0]

        def rel(a, b):
            return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))
        # seam 1: pair init rows
        z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
        rpe = model.rel_pos(feats)
        z_init = z_init + rpe + model.token_bonds(feats["token_bonds"].float()) + model.contact_conditioning(feats)
        zi_rows = rowpair._z_init_rows(model, s_inputs, feats, r0, r1)
        out["init"] = rel(zi_rows, z_init[:, r0:r1])
        out["relpos_rows"] = rel(rowpair._relpos_rows(model.rel_pos, feats, r0, r1), rpe[:, r0:r1])
        TRK = rowpair._core_mod("trunk")
        zi_shard = TRK.init_pair_shard(lay, lambda g0, g1: rowpair._z_init_rows(model, s_inputs, feats, g0, g1), s_inputs.new_empty((1, N, z_init.shape[-1])), rows=r1 - r0)
        out["init_pair_shard"] = rel(zi_shard, z_init[:, r0:r1])
        # seam 2: recycle (cycle 0 from zeros, then cycle 1 from a random z)
        z_prev = torch.randn_like(z_init)
        z_rec = z_init + model.z_recycle(model.z_norm(z_prev))
        z_loc = z_prev[:, r0:r1].contiguous().clone()
        z_loc = TRK.recycle_shard_(z_loc, TRK.ShardPark(zi_shard.clone(), False) if hasattr(TRK, "ShardPark") else zi_shard.clone(), lambda zr: model.z_recycle(model.z_norm(zr)), lay)
        out["recycle_shard_"] = rel(z_loc, z_rec[:, r0:r1])
        z0 = z_rec
        z0_loc = z0[:, r0:r1].contiguous()
        # seam 3: template rows
        torch.manual_seed(5); u_ref = model.template_module(z0, feats, pair_mask, use_kernels=False)
        torch.manual_seed(5); u_loc = rowpair._template_rows(model.template_module, z0_loc.clone(), feats, pair_mask, lay, False)
        out["template_rows"] = rel(u_loc, u_ref[:, r0:r1])
        # seam 6: PWA alone (block 0), seam 5: OPM alone
        msa_mod = model.msa_module; layer = msa_mod.layers[0]
        m = torch.randn(1, DIMS["S_sub"], N, DIMS["msa_s"]); msa_mask = torch.ones(1, DIMS["S_sub"], N)
        pwa_ref = layer.pair_weighted_averaging(m, z0, pair_mask, False)
        pwa_loc = rowpair._pwa_update(layer.pair_weighted_averaging, m, z0_loc.clone(), pair_mask, False, lay)
        out["pwa_rows"] = rel(pwa_loc, pwa_ref)
        pwa_ref_c = layer.pair_weighted_averaging(m, z0, pair_mask, True)
        pwa_loc_c = rowpair._pwa_update(layer.pair_weighted_averaging, m, z0_loc.clone(), pair_mask, True, lay)
        out["pwa_rows_chunk_heads"] = rel(pwa_loc_c, pwa_ref_c)
        opm_ref = layer.outer_product_mean(m, msa_mask, None)
        zz = torch.zeros_like(z0_loc); rowpair._opm_add_(layer.outer_product_mean, zz, m, msa_mask, None, lay)
        out["opm_rows"] = rel(zz, opm_ref[:, r0:r1])
        opm_ref4 = layer.outer_product_mean(m, msa_mask, 4)
        zz = torch.zeros_like(z0_loc); rowpair._opm_add_(layer.outer_product_mean, zz, m, msa_mask, 4, lay)
        out["opm_rows_chunk4"] = rel(zz, opm_ref4[:, r0:r1])
        # seam 7-12: one MSA pair layer (PairformerNoSeqLayer) alone
        torch.manual_seed(5); zl_ref = layer.pairformer_layer(z0.clone(), pair_mask, 512, False)
        torch.manual_seed(5); zl = z0_loc.clone(); rowpair._presharded_stack([layer.pairformer_layer], zl[0], pm, lay, False)
        out["pair_layer"] = rel(zl, zl_ref[:, r0:r1])
        # per-sublayer of one pair layer: MSA block 0, trunk block 0, template block 0 (each statement alone vs its stock sub-module)
        def sublayers(tag, lyr, zfull, C):
            """each pair sub-layer of one layer alone through the bound PairBlockFns (in place on a shard copy) vs stock ``z + sub(z)`` rows;
            NOTE the module-level references above run the module CLASS forwards, which apply() has patched: these run the layer objects."""
            zf = zfull[..., :C].contiguous() if zfull.shape[-1] != C else zfull
            mrows = pm[r0:r1]; has_s = hasattr(lyr, "attention")
            fns = rowpair._block_fns(lyr, lay, False, mask if has_s else None)
            PS = rowpair._core_mod("pairstack")
            from opt_core.mem.rowpair import ring
            def one(name, fn, ref_delta, transposed=False):
                zs = zf[0, r0:r1].clone(memory_format=torch.contiguous_format)
                if transposed:
                    zT = ring.transpose_shard(zs, lay); mT = PS.mask_transposed(mrows, lay)
                    zT = fn(zT, mT, lay); got = ring.transpose_shard(zT, lay)
                else:
                    got = fn(zs, mrows, lay)
                out[f"{tag}.{name}"] = rel(got, (zf + ref_delta)[0, r0:r1])
            one("trimul_out", fns.trimul_out, lyr.tri_mul_out(zf, mask=pair_mask, use_kernels=False))
            one("trimul_in", fns.trimul_in, lyr.tri_mul_in(zf, mask=pair_mask, use_kernels=False))
            one("triatt_start", fns.triatt_start, lyr.tri_att_start(zf, mask=pair_mask, chunk_size=512, use_kernels=False))
            one("triatt_end", fns.triatt_end, lyr.tri_att_end(zf, mask=pair_mask, chunk_size=512, use_kernels=False), transposed=True)
            one("transition", lambda z, m, l: fns.transition(z, None, l), lyr.transition_z(zf))
            zs = zf[0, r0:r1].clone(memory_format=torch.contiguous_format)
            if has_s:
                torch.manual_seed(5); s_ref, z_ref = lyr(torch.zeros(1, N, DIMS["token_s"]), zf.clone(), mask, pair_mask, 512, False)
                torch.manual_seed(5); zgot, _s = rowpair._pair_layer(lyr, zs, pm, lay, False, s=torch.zeros(1, N, DIMS["token_s"]), mask=mask)
                out[f"{tag}.s"] = rel(_s, s_ref)
            else:
                torch.manual_seed(5); z_ref = lyr(zf.clone(), pair_mask, 512, False)
                torch.manual_seed(5); zgot, _s = rowpair._pair_layer(lyr, zs, pm, lay, False)
            out[f"{tag}.layer"] = rel(zgot, z_ref[0, r0:r1])
        sublayers("msa0", msa_mod.layers[0].pairformer_layer, z0, DIMS["token_z"])
        sublayers("msa1", msa_mod.layers[1].pairformer_layer, z0, DIMS["token_z"])
        sublayers("trunk0", model.pairformer_module.layers[0], z0, DIMS["token_z"])
        sublayers("templ0", model.template_module.pairformer.layers[0], z0, DIMS["tdim"])
        # whole MSA module
        torch.manual_seed(5); zm_ref = msa_mod(z0.clone(), s_inputs, feats, False)
        torch.manual_seed(5); zm = rowpair._msa_rows(msa_mod, z0_loc.clone(), s_inputs, feats, pair_mask, lay, False)
        out["msa_rows"] = rel(zm, zm_ref[:, r0:r1])
        # seam 13-14: trunk Pairformer (2 blocks, with s)
        s0 = torch.randn(1, N, DIMS["token_s"])
        torch.manual_seed(5); s_ref, zp_ref = model.pairformer_module(s0.clone(), z0.clone(), mask=mask, pair_mask=pair_mask, use_kernels=False)
        torch.manual_seed(5); zp = z0_loc.clone(); s_got, _ = rowpair._presharded_stack(model.pairformer_module.layers, zp[0], pm, lay, False, s=s0.clone(), mask=mask)
        out["pairformer_z"] = rel(zp, zp_ref[:, r0:r1]); out["pairformer_s"] = rel(s_got, s_ref)
        # seam 15: distogram rows
        pd_ref = model.distogram_module(z0)
        pd = rowpair._distogram_rows(model.distogram_module, z0_loc, lay)
        out["distogram_rows"] = rel(pd, pd_ref[:, r0:r1])
    from opt_core.mem.rowpair import evidence
    return {"rank": int(os.environ["ROWPAIR_RANK"]) if "ROWPAIR_RANK" in os.environ else -1, "rows": (r0, r1), "rel": out,
            "transpose": (evidence.schedule() or {}).get("pairstack_transpose"), "tensors": {"pair_layer": zl.clone(), "pairformer_z": zp.clone(), "pairformer_s": s_got.clone()}}


@pytest.mark.parametrize("P,N,tin", [(2, 40, ""), (2, 41, ""), (2, 41, "1")])   # 41: ragged last rank; tin: the ending-node transpose in place (ROWPAIR_TRANSPOSE_INPLACE=1, the ×P line's word)
def test_seam_units_equal_stock_modules(P, N, tin, monkeypatch):
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv("ROWPAIR_TRANSPOSE_INPLACE", raising=False)
    out = launch.run_sharded(P, _units_entry, N, tin, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    print("ROWPAIR_UNITS", out["rel"], out["transpose"])
    bad = {k: v for k, v in out["rel"].items() if not v <= 1e-4}
    assert not bad, bad
    assert out["transpose"] == ("inplace" if tin == "1" else "p2p"), out["transpose"]     # the census word the core printed: the form that RAN


def test_pair_block_transpose_inplace_is_bitwise(monkeypatch):
    """The pair block's ending-node transpose IN PLACE (``ROWPAIR_TRANSPOSE_INPLACE=1``: core ``pairstack.pair_block_`` -> ``ring.transpose_shard_inplace_``,
    read at ``rowpair._pair_layer``) is data movement: one pair layer and the 2-block Pairformer on the rows are BIT-IDENTICAL to the all-to-all
    transpose form (P = 2, N = 41: a ragged last rank)."""
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv("ROWPAIR_TRANSPOSE_INPLACE", raising=False)
    a = launch.run_sharded(2, _units_entry, 41, "", mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    b = launch.run_sharded(2, _units_entry, 41, "1", mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200)
    assert (a["transpose"], b["transpose"]) == ("p2p", "inplace")
    for k in ("pair_layer", "pairformer_z", "pairformer_s"):
        assert torch.equal(a["tensors"][k], b["tensors"][k]), k


def test_attention_contract_refused_by_name():
    """H-qkvg guard: a lever that removed a sub-module the row statement calls is refused by name before any shard is built."""
    model, feats = build(24)
    rowpair.reset_for_tests()
    rowpair._STATE["P"] = 2
    rowpair._check_attention_contract(model)                               # the stock modules satisfy the contract
    lyr = model.pairformer_module.layers[0]
    del lyr.tri_att_start.mha.linear_k                                     # what a fused-projection lever that dropped the originals would leave
    lyr.tri_att_start.mha.linear_k = None
    with pytest.raises(rowpair.Refused, match="attention_contract: Attention.linear_k"):
        rowpair._check_attention_contract(model)


class _Caught(Exception):
    """Raised by the recording ``a_proj`` to stop a template call right after the pair features are formed."""


def _template_pair_features(model, feats, pair_mask, z, rows=None):
    """The pre-projection template pair features ``a_tij`` exactly as they reach ``TemplateV2Module.a_proj``: the stock module's dense
    ``[1, T, N, N, F]`` (``rows=None``, trunkv2.py TemplateV2Module.forward) or the kit's row statement's ``[1, T, r1-r0, N, F]``
    (``rows=(P, rank)``: rowpair._template_rows on that rank's Layout). The module's ``a_proj`` is swapped for a recorder that raises
    _Caught, so neither call runs past the features (no pair stack, no collectives: one process, any P)."""
    from opt_core.mem.rowpair.dist import Layout
    tm = model.template_module
    got = {}

    class _Rec(torch.nn.Module):
        def forward(self, a):
            got["a"] = a.detach().clone(); raise _Caught()

    orig = tm.a_proj; tm.a_proj = _Rec()
    try:
        try:
            if rows is None:
                tm(z, feats, pair_mask, use_kernels=False)
            else:
                P, rank = rows
                lay = Layout(z.shape[1], P, rank, B=1, align=1)
                rowpair._template_rows(tm, z[:, lay.r0:lay.r1].contiguous().clone(), feats, pair_mask, lay, False)
        except _Caught:
            pass
    finally:
        tm.a_proj = orig
    return got["a"], (None if rows is None else (lay.r0, lay.r1))


@pytest.mark.parametrize("N,T,chains", [(33, 1, 2), (129, 4, 3), (300, 2, 2)])
def test_template_pair_features_are_born_as_rows_bitwise(N, T, chains):
    """Templates under ``--n_gpu P`` (the fleet contract, C3): the template pair features the kit computes for a rank's rows
    (rowpair._template_rows: distogram one-hot, cb / frame pair masks, unit vectors, the visibility mask, res_type_i / res_type_j — every
    channel ``a_proj`` reads) are torch.equal to the stock TemplateV2Module's dense features SLICED to those rows — for every rank of
    P in {2, 3, 4, 8}, with several chains (visibility ids), coordinate-free tokens (cb / frame masks) and one all-dummy slot when T > 1.
    No rank ever forms the dense ``[T, N, N, F]`` features; N = 33 is the shape where one-hot widths equal N, 129 the ``big --n_gpu 2``
    floor, 300 a mid size."""
    model, feats = build(48, seed=3)                                      # the module (weights are irrelevant to the features; a_proj is recorded, not applied)
    g = torch.Generator().manual_seed(100 + N)
    from boltz.data import const
    asym = (torch.arange(N)[None] * chains) // N                          # `chains` contiguous chains -> visibility ids
    tf = {"template_restype": torch.nn.functional.one_hot(torch.randint(0, const.num_tokens, (1, T, N), generator=g), const.num_tokens).float(),
          "template_frame_rot": torch.linalg.qr(torch.randn(1, T, N, 3, 3, generator=g))[0], "template_frame_t": torch.randn(1, T, N, 3, generator=g) * 5,
          "template_mask_frame": (torch.rand(1, T, N, generator=g) < 0.8).float(), "template_cb": torch.randn(1, T, N, 3, generator=g) * 12,
          "template_ca": torch.randn(1, T, N, 3, generator=g) * 12, "template_mask_cb": (torch.rand(1, T, N, generator=g) < 0.9).float(),
          "visibility_ids": asym[:, None].expand(1, T, N).clone(), "template_mask": (torch.rand(1, T, N, generator=g) < 0.9).float()}
    if T > 1:                                                              # one dummy slot, as boltz's load_dummy_templates_features makes it (all-zero masks and coordinates)
        for k in ("template_mask_frame", "template_mask_cb", "template_mask", "template_cb", "template_ca", "template_frame_t"):
            tf[k][:, T - 1] = 0.0
    feats = dict(feats, **tf)
    mask = torch.ones(1, N); mask[0, -2:] = 0.0; pair_mask = mask[:, :, None] * mask[:, None, :]
    z = torch.randn(1, N, N, DIMS["token_z"], generator=g)
    dense, _ = _template_pair_features(model, feats, pair_mask, z)
    F = 38 + 1 + 3 + 1 + 2 * const.num_tokens
    assert dense.shape == (1, T, N, N, F) and dense.dtype == torch.float32, dense.shape
    checked = 0
    for P in (2, 3, 4, 8):
        for rank in range(P):
            from opt_core.mem.rowpair.dist import Layout
            if Layout(N, P, rank, B=1, align=1).R == 0:
                continue                                                   # more ranks than rows at this N: that rank owns no rows (the kit refuses such a plan by name)
            rows, (r0, r1) = _template_pair_features(model, feats, pair_mask, z, rows=(P, rank))
            assert rows.shape == (1, T, r1 - r0, N, F), (P, rank, rows.shape)
            assert torch.equal(rows, dense[:, :, r0:r1]), (N, T, P, rank, float((rows - dense[:, :, r0:r1]).abs().max()))
            checked += 1
    assert checked >= 4 + (N >= 8) * 13, checked
