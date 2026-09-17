"""tp_rowpair.template vs the STOCK openfold3 (the pinned wheel) ``TemplateEmbedderAllAtom`` on CPU threaded ranks (``opt_core.testing.run_ranks``):
``z0 + TemplateEmbedder(batch, z0, pair_mask)`` dense in the main thread vs ``template_embedder_add_rows_`` on each rank's rows.
Cases: T=2 DISTINCT real slots (dense pair features sliced to rows; two pair-stack passes), T=4 identical dummy slots dense (one pass after
de-duplication) and LAZY (``[T, 1, 1, F]`` placeholders + ``_lazy_template_pair``: zero rows synthesised per block; the line serves dummy
slots, so no all-dummy event is recorded), and the refusal of lazy rows for a REAL slot. fp32 ``max|diff| <= 1e-5``, ``torch.equal`` reported;
pair-shape guard in the sharded-pairstack mode. Helpers come from ``test_tp_rowpair_msa_cpu`` (one definition)."""

import pytest

from openfold3_ob0_opt.tests.test_tp_rowpair_msa_cpu import (HAVE_CORE, HAVE_OF3, HAVE_TORCH, CHUNK, KW, N_TOK, P_CASES, PAIRSTACK_MODES, TOL32,
                                                          PairGuard, arch_config, diff, dims, layout, randomise_, run, use_pairstack)

if HAVE_TORCH:
    import torch

needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 + opt_core>=0.5.10 (rowpair.template)")


def build_template_embedder(seed=0):
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    te = randomise_(TemplateEmbedderAllAtom(arch_config().template), seed + 2)
    te.template_pair_stack.tune_chunk_size = False
    te.template_pair_stack.chunk_size_tuner = None
    return te


def token_batch(N, seed, n_templ, real_slots=(), lazy=False):
    """Token-level template batch: 3 chains; slots in ``real_slots`` get random template content, the others are the predict-time dummies."""
    g = torch.Generator().manual_seed(seed)
    n1, n2 = N // 3, 2 * (N // 3)
    asym = torch.cat([torch.full((n1,), 1), torch.full((n2 - n1,), 2), torch.full((N - n2,), 3)]).float()
    token_mask = torch.ones(N)
    token_mask[n1 - 1] = 0.0
    b = {"asym_id": asym, "token_mask": token_mask,
         "template_restype": torch.nn.functional.one_hot(torch.full((n_templ, N), 31), 32).int(),
         "template_pseudo_beta_mask": torch.zeros(n_templ, N), "template_backbone_frame_mask": torch.zeros(n_templ, N)}
    if lazy:
        b["template_distogram"] = torch.zeros(n_templ, 1, 1, 39)
        b["template_unit_vector"] = torch.zeros(n_templ, 1, 1, 3)
        b["_lazy_template_pair"] = torch.ones(1)
    else:
        b["template_distogram"] = torch.zeros(n_templ, N, N, 39)
        b["template_unit_vector"] = torch.zeros(n_templ, N, N, 3)
        for t in real_slots:
            b["template_pseudo_beta_mask"][t] = (torch.rand(N, generator=g) > 0.3).float()
            b["template_backbone_frame_mask"][t] = (torch.rand(N, generator=g) > 0.3).float()
            b["template_restype"][t] = torch.nn.functional.one_hot(torch.randint(0, 32, (N,), generator=g), 32).int()
            b["template_distogram"][t] = torch.nn.functional.one_hot(torch.randint(0, 39, (N, N), generator=g), 39).float()
            b["template_unit_vector"][t] = torch.randn(N, N, 3, generator=g)
    return {k: v.unsqueeze(0) for k, v in b.items()}


def dense_of(batch_lazy, N):
    """The dense batch a lazy batch stands for (placeholders expanded to ``[T, N, N, F]`` zeros, flag dropped) — the stock reference input."""
    b = {k: v for k, v in batch_lazy.items() if k != "_lazy_template_pair"}
    for k in ("template_distogram", "template_unit_vector"):
        T, F = int(b[k].shape[1]), int(b[k].shape[-1])
        b[k] = torch.zeros(1, T, N, N, F, dtype=b[k].dtype)
    return b


CASES = {"two_real_slots": dict(n_templ=2, real_slots=(0, 1)), "four_dummies_dense": dict(n_templ=4), "four_dummies_lazy": dict(n_templ=4, lazy=True)}


@needs_stack
@pytest.mark.parametrize("pairstack", PAIRSTACK_MODES)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("P", P_CASES)
def test_template_embedder_vs_stock(P, case, pairstack, monkeypatch):
    from openfold3_ob0_opt.tp_rowpair import template as TEMPL
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    use_pairstack(monkeypatch, pairstack)
    te = build_template_embedder()
    N = N_TOK
    batch = token_batch(N, 11, **CASES[case])
    ref_batch = dense_of(batch, N) if CASES[case].get("lazy") else batch
    g = torch.Generator().manual_seed(5)
    z0 = torch.randn(1, N, N, dims()[1], generator=g) * 0.5
    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    with torch.no_grad():
        t_ref = z0 + te(batch=ref_batch, z=z0.clone(), pair_mask=pair_mask, chunk_size=CHUNK, _mask_trans=True, inplace_safe=True, **KW)

    def rank_fn(rank, P_):
        from opt_core.mem.rowpair.evidence import schedule
        lay = layout(N, P_, rank)
        z_loc = z0[:, lay.r0:lay.r1].clone().contiguous()
        pm_loc = pair_mask[:, lay.r0:lay.r1].contiguous()
        with torch.no_grad(), PairGuard(N, int(z0.shape[-1])) as guard:
            out = TEMPL.template_embedder_add_rows_(te, batch, z_loc, pm_loc, lay, census_tag="cycle0", chunk_size=CHUNK, _mask_trans=True, inplace_safe=True, **KW)
        sched = schedule()                                           # process-global census (threads share it; every rank records the same words)
        return {"rank": rank, "z": diff(out, t_ref[:, lay.r0:lay.r1]), "same_object": out is z_loc,
                "pair_guard_hits": guard.hits if pairstack == "sharded" else [], "templ_groups": sched.get("templ_groups"), "templ_event": sched.get("templ_event"),
                "templ_mode": sched.get("templ_mode")}

    for r in run(P, rank_fn):
        print(P, case, pairstack, r["rank"], r["z"], "groups", r["templ_groups"], "mode", r["templ_mode"], "event", r["templ_event"])
        assert r["z"]["max_abs_diff"] <= TOL32 and r["same_object"], (P, case, r)
        assert not r["pair_guard_hits"], (r["rank"], r["pair_guard_hits"][:3])
        if case.startswith("four_dummies"):
            assert r["templ_groups"] == "0+1+2+3", r                    # four identical dummies -> ONE pair-stack pass
            assert r["templ_event"] is None, r                           # dummy slots are the line's served case: no templated-request event
        if case == "two_real_slots":
            assert r["templ_groups"] == "0/1", r
        if case == "four_dummies_lazy":
            assert r["templ_mode"] == "lazy_dummy", r


@needs_stack
def test_lazy_rows_for_a_real_slot_refuse_by_name(monkeypatch):
    """Lazy zero rows may only stand in for DUMMY slots: a batch flagged lazy whose masks mark a real slot is refused by name (core RowpairRefused)."""
    from opt_core.mem.rowpair import RowpairRefused
    from openfold3_ob0_opt.tp_rowpair import template as TEMPL
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    N = 24
    batch = token_batch(N, 3, n_templ=2, lazy=True)
    batch["template_pseudo_beta_mask"][0, 1, :5] = 1.0                 # slot 1 is real

    def rank_fn(rank, P_):
        lay = layout(N, P_, rank)
        try:
            TEMPL.template_rows(batch, lay)
        except RowpairRefused as e:
            return str(e)
        return "no refusal"
    msgs = run(2, rank_fn)
    assert all("REAL templates" in m for m in msgs), msgs


