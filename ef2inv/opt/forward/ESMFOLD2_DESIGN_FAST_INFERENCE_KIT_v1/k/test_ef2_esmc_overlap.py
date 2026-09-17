"""Unit tests for ef2_esmc_overlap (CUDA + the Biohub transformers fork; a small random-init ESMC LM, no checkpoint).

    pytest -q k/test_ef2_esmc_overlap.py      or, without pytest:   python k/test_ef2_esmc_overlap.py

The lever is EXACT: on a miniature of the cookbook's step (fold under its own seed context with RNG draws inside, structure
gradient, then the pLM term under `seed_context(seed + step)` drawing its masks, then its gradient, then a draw from the outer
RNG stream) every tensor — structure grad, pLM loss, pLM grad, the outer draw — must be torch.equal with and without the lever
over consecutive steps (step 0 computes in place, later steps are served from the side stream), alone and on ef2_pppl_graph;
critic folds, argument changes and unseeded folds must compute in place and be counted by name; disable() must put the
module's two functions back.
"""
import os, sys, types, math
sys.path.insert(0, os.path.dirname(__file__))
try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn.functional as F
import transformers.models.esmc.modeling_esmc as ME
from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context
import ef2_esmc_overlap as eeo
import ef2_pppl_graph

_skipif = pytest.mark.skipif if pytest else (lambda cond, reason="": (lambda f: f))
cuda = _skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB, AA = 64, 20


def _make_module(lm):
    """A miniature cookbook module: the two functions the lever wraps + seed_context, with the real call shapes and RNG use."""
    bd = types.ModuleType("mini_binder_design")
    bd.seed_context = _seed_context
    bd.torch = torch

    def fold_and_get_distogram(model, target_seq, target_one_hot, design, num_loops=0, num_sampling_steps=1, calculate_confidence=False, seed=None):
        trunk = model
        ids = torch.cat([target_one_hot.argmax(-1), design.detach().argmax(-1) + 4], dim=1)
        seq_id = torch.zeros_like(ids); seq_id[:, target_one_hot.shape[1]:] = 1
        with torch.inference_mode():
            hs = trunk(input_ids=ids, sequence_id=seq_id, output_hidden_states=True).hidden_states
        feat = torch.stack([h.float() for h in hs], 0).mean(0).mean(-1).clone()          # [B, L] "structure features"
        with bd.seed_context(seed):
            noise = torch.randn_like(design)                                             # the sampler's RNG use inside the fold
            for _ in range(3 if calculate_confidence else 1):
                z = (design * (feat[:, target_one_hot.shape[1]:, None] + noise)).sum(-1)
        return {"distogram_logits": z, "seq_list": ["X"], "iptm": None}

    def compute_esmc_pseudoperplexity_nll(esmc_model, binder_design, score_mask, batch_size=4, n_passes=4):
        device = binder_design.device
        B, Lb, _ = binder_design.shape
        esmc = esmc_model.esmc
        emb = esmc.embed.weight                                                          # [VOCAB, D]
        onehot = F.one_hot(binder_design.argmax(-1), AA).to(binder_design.dtype)
        st = onehot + (binder_design - binder_design.detach())                           # straight-through
        losses = []
        for b in range(B):
            npos = int(score_mask.sum().item())                                          # a host sync, as the cookbook's nonzero()
            nmask = max(1, math.ceil(0.15 * npos))
            scores = torch.rand((n_passes, Lb), device=device)                           # THE mask draw under the caller's seed context
            masked = scores.topk(nmask, dim=-1, largest=False).indices
            pm = torch.zeros((n_passes, Lb), dtype=torch.bool, device=device)
            pm[torch.arange(n_passes, device=device)[:, None], masked] = True
            x = torch.zeros((n_passes, Lb, VOCAB), device=device, dtype=emb.dtype)
            x[:, :, 4:4 + AA] = st[b].to(emb.dtype).unsqueeze(0).expand(n_passes, -1, -1).clone()
            x = torch.where(pm[..., None], torch.zeros_like(x), x)
            rows = []
            for s0 in range(0, n_passes, batch_size):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden, *_ = esmc.transformer(x[s0:s0 + batch_size] @ emb, sequence_id=None, layers_to_collect=[], output_attentions=False)
                    rows.append(esmc_model.lm_head(hidden))
            logits = torch.cat(rows, 0)
            logp = logits.log_softmax(-1)[:, :, 4:4 + AA]
            nll = -(logp * binder_design[b].to(logp.dtype).unsqueeze(0)).sum(-1)
            losses.append(nll[pm].mean())                                               # boolean indexing: a host sync
        return torch.stack(losses, 0)

    bd.fold_and_get_distogram = fold_and_get_distogram
    bd.compute_esmc_pseudoperplexity_nll = compute_esmc_pseudoperplexity_nll

    class App:                                   # the design app's shape: its module is bd, it holds the LM
        pass
    App.__module__ = bd.__name__
    sys.modules[bd.__name__] = bd
    app = App(); app.esmc_model = lm
    return bd, app


def _lm(seed=0):
    cfg = ME.ESMCConfig(d_model=128, n_heads=4, n_layers=3, vocab_size=VOCAB)
    torch.manual_seed(seed)
    lm = ME.ESMCForMaskedLM(cfg).cuda().eval().requires_grad_(False)
    lm.esmc = lm.esmc.bfloat16()
    return lm


def _trajectory(bd, lm, steps, seed=7, B=1, Lt=30, Lb=12, critic_at=None, change_mask_at=None, unseeded_at=None):
    """The cookbook's run_step, in miniature; returns per-step tensors to compare."""
    torch.manual_seed(seed)
    logits0 = (0.5 * torch.randn(B, Lb, AA, device="cuda"))
    target = F.one_hot(torch.randint(0, AA, (B, Lt), device="cuda"), AA).float()
    score_mask = torch.ones(Lb, dtype=torch.bool, device="cuda")
    out = []
    logits = logits0.clone().requires_grad_(True)
    with bd.seed_context(1234):                                        # an outer RNG state the step must leave as stock leaves it
        for step in range(steps):
            design = F.softmax(logits / 0.5, dim=-1)
            critic = critic_at is not None and step == critic_at
            fr = bd.fold_and_get_distogram(lm.esmc, "T" * Lt, target, design, num_loops=3 if critic else 1,
                                           num_sampling_steps=50 if critic else 1, calculate_confidence=critic,
                                           seed=None if (unseeded_at is not None and step == unseeded_at) else seed + step)
            sl = fr["distogram_logits"].sum(-1)
            g_s, = torch.autograd.grad(sl.mean(), logits)
            design = F.softmax(logits / 0.5, dim=-1)
            sm = score_mask if not (change_mask_at is not None and step >= change_mask_at) else score_mask.clone().index_fill_(0, torch.tensor([0], device="cuda"), False)
            with bd.seed_context(seed + step):
                pl = bd.compute_esmc_pseudoperplexity_nll(esmc_model=lm, binder_design=design, score_mask=sm, batch_size=128, n_passes=4)
            g_p, = torch.autograd.grad(pl.mean(), logits)
            after = torch.rand(3, device="cuda")                             # the outer stream's next draw
            with torch.no_grad():
                logits -= 0.1 * (g_s + g_p)
            out.append(dict(sl=sl.detach().clone(), g_s=g_s.clone(), pl=pl.detach().clone(), g_p=g_p.clone(), after=after, logits=logits.detach().clone()))
    return out


def _compare(ref, got, label):
    assert len(ref) == len(got)
    for i, (r, g) in enumerate(zip(ref, got)):
        for k in r:
            assert torch.equal(r[k], g[k]), f"{label}: step {i} tensor {k} differs (max|d|={(r[k].float() - g[k].float()).abs().max().item():.3e})"


@cuda
def test_served_steps_are_bitwise_and_rng_neutral_eager_and_graphed():
    lm = _lm(); bd, app = _make_module(lm)
    ref = _trajectory(bd, lm, 4)
    ref2 = _trajectory(bd, lm, 4)
    _compare(ref, ref2, "stock repeat (determinism of the miniature itself)")
    h = eeo.enable(app)
    assert eeo.enable(bd, esmc_model=lm) is h                                    # idempotent, either spelling
    try:
        got = _trajectory(bd, lm, 4)
    finally:
        eeo.disable(app)
    _compare(ref, got, "overlap (eager LM)")
    assert h.stats["served"] == 3 and h.stats["computed"] == 1 and h.stats["fallback"] == {"no_reference_yet": 1}, h.stats
    assert bd.fold_and_get_distogram.__name__ == "fold_and_get_distogram" and not hasattr(bd.compute_esmc_pseudoperplexity_nll, "_ef2_esmc_overlap")
    # stacked on the pLM graph lever (the configuration it is meant for)
    ef2_pppl_graph.enable(lm)
    try:
        refg = _trajectory(bd, lm, 4)
        _compare(ref, refg, "pppl graph alone vs eager")
        h = eeo.enable(app)
        try:
            gotg = _trajectory(bd, lm, 4)
        finally:
            eeo.disable(app)
    finally:
        ef2_pppl_graph.disable(lm)
    _compare(ref, gotg, "overlap on pppl graph")
    assert h.stats["served"] == 3, h.stats


@cuda
def test_critic_fold_unseeded_fold_and_changed_args_compute_in_place_by_name():
    lm = _lm(1); bd, app = _make_module(lm)
    ref = _trajectory(bd, lm, 6, critic_at=2, change_mask_at=4, unseeded_at=5)
    h = eeo.enable(app)
    try:
        got = _trajectory(bd, lm, 6, critic_at=2, change_mask_at=4, unseeded_at=5)
    finally:
        eeo.disable(app)
    _compare(ref, got, "fallback paths")
    fb = h.stats["fallback"]
    # step0 no reference; step1 served; step2 critic fold -> not launched; step3 served; step4 mask changed -> inputs differ at the call,
    # computed in place (and the new mask becomes the reference); step5 unseeded fold -> not launched
    assert h.stats["served"] == 2 and h.stats["computed"] == 4, h.stats
    assert fb.get("no_reference_yet") == 1 and fb.get("not_design_fold") == 1 and fb.get("inputs_differ") == 1 and fb.get("unseeded_fold") == 1, fb


@cuda
def test_batch_of_two_designs_served_bitwise():
    lm = _lm(2); bd, app = _make_module(lm)
    ref = _trajectory(bd, lm, 3, B=2)
    h = eeo.enable(app)
    try:
        got = _trajectory(bd, lm, 3, B=2)
    finally:
        eeo.disable(app)
    _compare(ref, got, "B=2")
    assert h.stats["served"] == 2, h.stats


@cuda
def test_wrappers_stacked_over_ours_after_enable_run_early_without_recursion():
    """A (*a, **kw) wrapper stacked on BOTH module attributes after enable (the yardstick's per-step phase timer does exactly this; a plain
    wrapper hides the stock signature): the early launch goes through the stacked pLM wrapper, reaches the body through the lever's pass-through,
    and the loop's call is still served — bitwise."""
    lm = _lm(3); bd, app = _make_module(lm)
    ref = _trajectory(bd, lm, 3)
    h = eeo.enable(app)
    under_fold, under_pppl = bd.fold_and_get_distogram, bd.compute_esmc_pseudoperplexity_nll      # = the lever's wrappers
    calls = {"fold": 0, "pppl": 0}

    def fold_timer(*a, **k):
        calls["fold"] += 1
        return under_fold(*a, **k)

    def pppl_timer(*a, **k):
        calls["pppl"] += 1
        return under_pppl(*a, **k)

    bd.fold_and_get_distogram, bd.compute_esmc_pseudoperplexity_nll = fold_timer, pppl_timer
    try:
        got = _trajectory(bd, lm, 3)
    finally:
        bd.fold_and_get_distogram, bd.compute_esmc_pseudoperplexity_nll = under_fold, under_pppl
        eeo.disable(app)
    _compare(ref, got, "wrappers stacked over ours")
    assert h.stats["served"] == 2 and h.stats["computed"] == 1, h.stats
    assert calls == {"fold": 3, "pppl": 3 + 2}, calls              # the loop's 3 calls + the lever's 2 early launches through the stacked pLM wrapper
    assert bd.fold_and_get_distogram.__name__ == "fold_and_get_distogram"


if __name__ == "__main__":
    n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                n_fail += 1; print("FAIL", name, type(e).__name__, str(e)[:400], flush=True)
    print("RESULT", "OK" if not n_fail else f"{n_fail} failed")
    sys.exit(1 if n_fail else 0)
