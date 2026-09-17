"""tp_rowpair.confidence vs the STOCK openfold3 (the pinned wheel) ``AuxiliaryHeadsAllAtom.forward`` + ``get_confidence_scores`` on CPU threaded ranks:
the stock heads (real ``model_config`` heads: pairformer embedding with its 4 blocks, PAE / PDE / pLDDT / resolved / distogram; random weights)
run dense in the main thread on ``[1, 1, N, N, 128]``; every rank runs ``aux_heads_rows`` on its ``[1, 1, n_loc, N, 128]`` rows and rank 0's
``_tp_confidence`` dict is compared key by key (ptm, iptm, pae, pde, gpde, plddt, chain_ptm, chain_pair_iptm, bespoke_iptm, has_clash,
sample_ranking_score; disorder is 0 on both sides: no protein tokens). Values fp32 ``max|diff| <= 1e-5`` (row-blocked softmax / expected
values; ``torch.equal`` reported), pLDDT / resolved logits ``<= 1e-5``; ranks > 0 hold empty ``pae`` / ``pde``; rank 0's matrices live on
the host. Cases: oracle | sharded confidence pairformer; the trunk shard's placement (``heads.ZTrunkPlan``): ``resident`` (levers off),
``inplace`` (``ROWPAIR_FREE_ZTRUNK=1``: S = 1 embeds in place — the shard is left with zero rows; S = 2 says ``resident:free_declined:samples``),
``parked`` (``ROWPAIR_CONF_PARK_ZTRUNK=1``: the shard sits on the host, its storage released before the pair representation exists, at S = 1 AND
S = 2), ``both`` (the line's setting: the park engages); S = 1 and S = 2 samples. The placement is bit-neutral: every form meets the same
tolerance against the stock, and the parked / in-place outputs equal the resident ones exactly (``forms_bitwise``)."""

import pytest

from openfold3_ob0_opt.tests.test_tp_rowpair_msa_cpu import (HAVE_CORE, HAVE_OF3, HAVE_TORCH, CHUNK, KW, P_CASES, PAIRSTACK_MODES, TOL32, PairGuard, diff, layout,
                                                          randomise_, run, use_pairstack)

if HAVE_TORCH:
    import torch

needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 + opt_core>=0.5.10 (rowpair.heads/confidence)")
N_TOK = 44


def build_heads(seed):
    import ml_collections as mlc
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    cfg = mlc.ConfigDict(model_config.to_dict())
    ah = randomise_(AuxiliaryHeadsAllAtom(cfg.architecture.heads), seed + 1, 0.15)
    stk = ah.pairformer_embedding.pairformer_stack
    stk.tune_chunk_size = False
    stk.chunk_size_tuner = None
    return ah, cfg


def make_batch(N, seed):
    """``[B=1, S=1, ...]`` batch as ``_rollout`` sees it (sample dim unsqueezed); 4 chains (2 ligand), all tokens atomized with one atom each."""
    g = torch.Generator().manual_seed(seed)
    n1, n2, n3 = N // 3, 2 * (N // 3), N - 1
    asym = torch.cat([torch.full((n1,), 1), torch.full((n2 - n1,), 2), torch.full((n3 - n2,), 3), torch.full((N - n3,), 4)]).float()
    is_ligand = ((asym == 3) | (asym == 4)).float()
    b = {"asym_id": asym, "token_mask": torch.ones(N), "restype": torch.nn.functional.one_hot(torch.randint(0, 32, (N,), generator=g), 32).float(),
         "is_protein": torch.zeros(N, dtype=torch.long), "is_rna": torch.zeros(N, dtype=torch.long), "is_dna": torch.zeros(N, dtype=torch.long),
         "is_ligand": is_ligand.long(), "is_atomized": torch.ones(N, dtype=torch.long), "num_atoms_per_token": torch.ones(N, dtype=torch.long),
         "start_atom_index": torch.arange(N), "atom_mask": torch.ones(N), "atom_to_token_index": torch.arange(N)}
    b = {k: v[None, None] for k, v in b.items()}
    b["atom_array"] = [None]
    return b


def compare_tree(a, b, pfx=""):
    """``{key: (max|diff|, equal)}`` over the stock dict's leaves; keys / shapes / dtypes must match exactly."""
    out = {}
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a.keys()) == set(b.keys()), (pfx, sorted(a.keys()) if isinstance(a, dict) else type(a), sorted(b.keys()))
        for k in b:
            out.update(compare_tree(a[k], b[k], f"{pfx}{k}."))
    elif isinstance(b, torch.Tensor):
        a = a.to(b.device)
        assert a.shape == b.shape and a.dtype == b.dtype, (pfx, a.shape, b.shape, a.dtype, b.dtype)
        d = (a.double() - b.double()).abs()
        same_nan = torch.isnan(a) == torch.isnan(b)
        d = torch.where(torch.isnan(d), torch.where(torch.isnan(a) & torch.isnan(b), torch.zeros_like(d), torch.full_like(d, float("inf"))), d)
        scale = max(1.0, float(torch.nan_to_num(b.detach()).abs().max().item())) if b.numel() else 1.0     # plddt lives in [0, 100], pae / pde in Angstrom
        out[pfx.rstrip(".")] = (d.max().item() if d.numel() else 0.0, bool(same_nan.all()) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)), scale)
    return out


ZTRUNK_FORMS = {"resident": ("0", "0"), "inplace": ("1", "0"), "parked": ("0", "1"), "both": ("1", "1")}   # form -> (ROWPAIR_FREE_ZTRUNK, ROWPAIR_CONF_PARK_ZTRUNK)


def expected_word(form: str, S: int) -> str:
    """The plan's census word for the one pass of a stand-alone ``aux_heads_rows`` call on a CPU shard that owns its storage."""
    free, park = ZTRUNK_FORMS[form]
    if free == "1" and S == 1:
        return "inplace"                                                        # the in-place form wins where it applies (one sample, last use)
    if park == "1":
        return "parked:host"                                                    # a CPU shard parks on (pageable) host
    if free == "1":
        return "resident:free_declined:samples"
    return "resident"


def run_case(P, S, form, pairstack, monkeypatch, offload_inference=False):
    """rank -> result dict of one (P, S, form, pairstack) case: diffs vs the stock, the plan word, the shard rows left, rank 0's raw confidence dict.
    ``offload_inference``: the sharded head's ``offload_inference`` argument (named, never a refusal; the stock reference runs without it)."""
    from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
    from openfold3_ob0_opt.tp_rowpair import confidence as CONF
    free, park = ZTRUNK_FORMS[form]
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", free)
    monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", park)
    monkeypatch.setenv("ROWPAIR_CONF_ROWS", "8")                   # several row blocks / all-to-all windows per rank at this N
    monkeypatch.setenv("ROWPAIR_FRAME_ROWS", "7")                  # several frame-search row blocks
    for k in ("ROWPAIR_TRIMUL_ROWS_A", "ROWPAIR_TRIMUL_ROWS_B", "ROWPAIR_TRIATT_QROWS"):
        monkeypatch.setenv(k, "8")                                 # pinned tile schedule (tiles_source=fixed): a free-memory-derived schedule changes GEMM M box to box
    use_pairstack(monkeypatch, pairstack)
    ah, cfg = build_heads(3)
    N = N_TOK
    batch = make_batch(N, 11)
    g = torch.Generator().manual_seed(5)
    c_s_input = ah.pairformer_embedding.linear_i.in_features
    c_z = ah.pae.linear.in_features
    c_s = int(cfg.architecture.shared.c_s)
    si_input = torch.randn(1, 1, N, c_s_input, generator=g)
    si_trunk = torch.randn(1, 1, N, c_s, generator=g)
    zij_trunk = torch.randn(1, 1, N, N, c_z, generator=g)
    x_pred = torch.randn(1, S, N, 3, generator=g) * 6.0
    kw = dict(KW, chunk_size=CHUNK, inplace_safe=True, offload_inference=False, _mask_trans=True)
    with torch.no_grad():
        output = {"si_trunk": si_trunk, "zij_trunk": zij_trunk.clone(), "atom_positions_predicted": x_pred}
        aux_ref = ah(batch=batch, si_input=si_input, output=output, use_zij_trunk_embedding=True, **kw)
        outputs_ref = dict(output)
        outputs_ref.update(aux_ref)
        conf_ref = get_confidence_scores(batch=batch, outputs=outputs_ref, config=cfg, compute_per_sample=False)

    def rank_fn(rank, P_):
        from opt_core.mem.rowpair import evidence
        lay = layout(N, P_, rank)
        with torch.no_grad(), PairGuard(N, c_z) as guard:
            zshard = zij_trunk[..., lay.r0:lay.r1, :, :].clone().contiguous()
            storage_before = int(zshard.untyped_storage().nbytes())
            out = {"si_trunk": si_trunk, "zij_trunk": zshard, "atom_positions_predicted": x_pred}
            aux = CONF.aux_heads_rows(ah, batch=batch, si_input=si_input, output=out, lay=lay, config=cfg, use_zij_trunk_embedding=True, **dict(kw, offload_inference=offload_inference))
        sched = evidence.schedule()
        res = {"rank": rank, "plddt_logits": diff(aux["plddt_logits"], aux_ref["plddt_logits"]),
               "resolved": diff(aux["experimentally_resolved_logits"], aux_ref["experimentally_resolved_logits"]),
               "pair_guard_hits": guard.hits if pairstack == "sharded" else [], "ztrunk_left": tuple(out["zij_trunk"].shape),
               "word": str(sched.get("conf_ztrunk")), "conf_pairstack": sched.get("conf_pairstack"), "conf_offload": sched.get("conf_offload"),
               "storage_before": storage_before, "storage_after": int(zshard.untyped_storage().nbytes())}
        conf = aux["_tp_confidence"]
        if rank == 0:
            res["cmp"] = compare_tree(conf, conf_ref)
            res["pae_device"] = str(conf["pae"].device)
            res["raw"] = {k: v.detach().clone() for k, v in conf.items() if isinstance(v, torch.Tensor)}
            res["plddt_raw"] = aux["plddt_logits"].detach().clone()
        else:
            res["pae_numel"] = int(conf["pae"].numel()) if "pae" in conf else -1
        return res

    return {r["rank"]: r for r in run(P, rank_fn)}


@needs_stack
@pytest.mark.parametrize("pairstack", PAIRSTACK_MODES)
@pytest.mark.parametrize("form", list(ZTRUNK_FORMS))
@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("P", P_CASES)
def test_aux_heads_and_confidence_vs_stock(P, S, form, pairstack, monkeypatch):
    res = run_case(P, S, form, pairstack, monkeypatch)
    word = expected_word(form, S)
    for rank, r in sorted(res.items()):
        assert r["plddt_logits"]["max_abs_diff"] <= TOL32 and r["resolved"]["max_abs_diff"] <= TOL32, (P, S, form, r["plddt_logits"], r["resolved"])
        assert not r["pair_guard_hits"], (rank, r["pair_guard_hits"][:3])
        assert r["word"] == word, (rank, form, S, r["word"])                       # the plan's census word for the one pass
        assert r["conf_pairstack"] == ("inplace" if pairstack == "sharded" else "copy"), (pairstack, r["conf_pairstack"])   # the core driver runs on the pass's own rows; the test's dense oracle returns a fresh tensor, named `copy`
        consumed = word == "inplace" or word.startswith("parked")
        assert (r["ztrunk_left"][-3] == 0) == consumed, (form, S, r["ztrunk_left"])  # output["zij_trunk"] holds zero rows once the plan consumed the shard
        assert (r["storage_after"] == 0) == word.startswith("parked"), (form, S, r["storage_before"], r["storage_after"])   # parked: the shard's storage released, not restored (last use)
        if rank == 0:
            worst = 0.0
            for k, (d, eq, scale) in sorted(r["cmp"].items()):
                print(f"P={P} S={S} {form} {pairstack} rank0 {k:34s} max|diff|={d:.3e} (scale {scale:.0f}) equal={eq}")
                worst = max(worst, d / scale)
            assert worst <= TOL32, (P, S, form, r["cmp"])                          # fp32 tolerance relative to the key's magnitude
            assert r["pae_device"] == "cpu", r["pae_device"]
        else:
            assert r["pae_numel"] == 0, r


@needs_stack
@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("P", P_CASES)
def test_ztrunk_forms_bitwise(P, S, monkeypatch):
    """Placement only: the in-place / parked / both forms return rank 0's confidence dict and the pLDDT logits BIT-IDENTICAL to the resident form."""
    ref = run_case(P, S, "resident", "sharded", monkeypatch)[0]
    for form in ("inplace", "parked", "both"):
        got = run_case(P, S, form, "sharded", monkeypatch)[0]
        assert got["raw"].keys() == ref["raw"].keys(), form
        for k in ref["raw"]:
            a, b = got["raw"][k], ref["raw"][k]
            assert a.shape == b.shape and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)) and bool((torch.isnan(a) == torch.isnan(b)).all()), (P, S, form, k)
        assert torch.equal(got["plddt_raw"], ref["plddt_raw"]), (P, S, form)


@needs_stack
@needs_stack
def test_confidence_offload_is_named_not_refused(monkeypatch):
    """``offload_inference.confidence_heads=true`` (the shipped predict preset above its token_cutoff) is not a refusal under row sharding: no
    ``[S, N, N, *]`` tensor exists on any rank to move — the head names it (census word ``conf_offload=stock_on_na``) and returns rank 0's
    confidence dict and the pLDDT logits BIT-IDENTICAL to the run without it (placement words aside, nothing differs)."""
    ref = run_case(2, 2, "resident", "sharded", monkeypatch)
    got = run_case(2, 2, "resident", "sharded", monkeypatch, offload_inference=True)
    assert [r["conf_offload"] for r in ref.values()] == ["off", "off"] and [r["conf_offload"] for r in got.values()] == ["stock_on_na", "stock_on_na"], (ref[0]["conf_offload"], got[0]["conf_offload"])
    for k in ref[0]["raw"]:
        a, b = got[0]["raw"][k], ref[0]["raw"][k]
        assert a.shape == b.shape and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)), k
    assert torch.equal(got[0]["plddt_raw"], ref[0]["plddt_raw"])


@needs_stack
def test_refusals_by_name(monkeypatch):
    from openfold3_ob0_opt.tp_rowpair import confidence as CONF
    ah, cfg = build_heads(3)
    N = 16
    batch = make_batch(N, 1)
    lay = layout(N, 2, 0)
    out = {"si_trunk": torch.zeros(1, 1, N, 4), "zij_trunk": torch.zeros(1, 1, lay.R, N, ah.pae.linear.in_features), "atom_positions_predicted": torch.zeros(1, 1, N, 3)}
    from openfold3_ob0_opt.tp_rowpair.pairstack import PairstackRefused
    with pytest.raises(PairstackRefused):
        CONF.aux_heads_rows(ah, batch=batch, si_input=torch.zeros(1, 1, N, 8), output=dict(out), lay=lay, config=cfg, **dict(KW, use_lma=True))
