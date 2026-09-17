"""CONF track — CPU unit tests of the template-module levers (opt/forward/conf/template_levers.py through the boltz2_opt.conf loader):

  tdummy  on an all-dummy template batch the stock module's update is +0.0 in every element, the elided call returns zeros of the same
          dtype/shape so `z + u` is bit-identical, and the RNG stream advances exactly as the stock call advances it (the eight
          get_dropout_mask draws are replayed); a batch with a live slot takes the stock path bit for bit.
  tfeat   the featurization computed at the first call of a forward generation and reused by the later calls gives outputs
          bit-identical to stock's per-call recomputation; a new generation (next item) recomputes.

Pure-python boltz 2.2.1 model tree required (pip install --no-deps boltz==2.2.1 einops scipy); no GPU, no pytorch_lightning (the
Boltz2.forward generation hook is driven by hand: install(model_hook=False) + begin_forward()).
"""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.trunkv2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops scipy)")

from boltz.data import const  # noqa: E402
from boltz.model.modules.trunkv2 import TemplateV2Module  # noqa: E402

from boltz2_opt import conf as ADAPTER  # noqa: E402

TOKEN_Z, TDIM, N = 16, 8, 24


@pytest.fixture()
def TL():
    tl = ADAPTER._module("template_levers")
    yield tl
    tl.uninstall()
    for k in list(tl.STATS):
        tl.STATS[k] = 0


def make_module(seed=0):
    torch.manual_seed(seed)
    m = TemplateV2Module(token_z=TOKEN_Z, template_dim=TDIM, template_blocks=2)
    with torch.no_grad():                      # stock inits u_proj / gating to zero in places; randomize everything so the checks are not vacuous
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return m.eval()


def dummy_feats(B=1, T=1, n=N):
    """featurizerv2.load_dummy_templates_features(tdim=1, num_tokens=n) with a batch dim (the collated form the model receives)."""
    z = lambda *s: torch.zeros(B, T, *s)
    return {"template_restype": torch.nn.functional.one_hot(torch.zeros(B, T, n, dtype=torch.long), const.num_tokens).float(),
            "template_frame_rot": z(n, 3, 3), "template_frame_t": z(n, 3), "template_cb": z(n, 3), "template_ca": z(n, 3),
            "template_mask_cb": z(n), "template_mask_frame": z(n), "template_mask": z(n), "query_to_template": torch.zeros(B, T, n, dtype=torch.long),
            "visibility_ids": z(n)}


def live_feats(B=1, T=2, n=N, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(B, T, *s, generator=g)
    rt = torch.randint(0, const.num_tokens, (B, T, n), generator=g)
    mask = (torch.rand(B, T, n, generator=g) > 0.3).float(); mask[:, :, 0] = 1.0
    rot = torch.linalg.qr(r(n, 3, 3)).Q
    return {"template_restype": torch.nn.functional.one_hot(rt, const.num_tokens).float(), "template_frame_rot": rot, "template_frame_t": r(n, 3),
            "template_cb": r(n, 3) * 8, "template_ca": r(n, 3) * 8, "template_mask_cb": mask.clone(), "template_mask_frame": mask.clone(),
            "template_mask": mask.clone(), "query_to_template": torch.arange(n).expand(B, T, n).clone(),
            "visibility_ids": (torch.arange(n) // (n // 2)).float().expand(B, T, n).clone()}


def run_calls(m, feats, zs, pair_mask):
    """z_k + m(z_k, ...) for each z in zs (the trunk's per-pass call), plus the RNG state after the last call."""
    outs = []
    with torch.no_grad():
        for z in zs:
            outs.append(z + m(z, feats, pair_mask, False))
    return outs, torch.get_rng_state()


def test_adapter_words():
    assert ADAPTER.parse_words("tdummy, tfeat") == ["tfeat", "tdummy"]
    assert ADAPTER.parse_words("") == []
    with pytest.raises(ValueError):
        ADAPTER.parse_words("tfeat,nope")
    assert set(ADAPTER.LEVERS) >= {"tfeat", "tdummy"} and all(ADAPTER.MODULE_OF[w] for w in ADAPTER.LEVERS)


def test_tdummy_stock_update_is_plus_zero_and_elision_is_bitwise_with_rng_replayed(TL):
    m = make_module(); feats = dummy_feats(); pair_mask = torch.ones(1, N, N)
    zs = [torch.randn(1, N, N, TOKEN_Z, generator=torch.Generator().manual_seed(10 + k)) for k in range(4)]
    zs[1][0, 3, 5, 2] = -0.0; zs[2][0, 0, 0, 0] = 0.0                       # signed zeros in z: stock's z + (+0.0) normalises -0.0
    torch.manual_seed(123)
    with torch.no_grad():
        u = m(zs[0], feats, pair_mask, False)
    assert bool((u == 0).all()) and not bool(torch.signbit(u).any()), "stock's all-dummy template update must be +0.0 in every element"
    torch.manual_seed(123); ref, ref_rng = run_calls(m, feats, zs, pair_mask)
    assert not bool(torch.signbit(ref[1][0, 3, 5, 2])), "z + (+0.0) turns -0.0 into +0.0 in stock"
    TL.install(["tdummy"], model_hook=False); TL.begin_forward()
    torch.manual_seed(123); got, got_rng = run_calls(m, feats, zs, pair_mask)
    for a, b in zip(ref, got):
        assert torch.equal(a, b) and torch.equal(torch.signbit(a), torch.signbit(b))
    assert torch.equal(ref_rng, got_rng), "the elided call must advance the RNG stream exactly as the stock template pairformer does"
    assert TL.STATS["elided"] == 4 and TL.STATS["live"] == 0 and TL.STATS["rng_draws_replayed"] == 4 * 8
    # without the replay the stream WOULD differ (guards the test itself)
    torch.manual_seed(123); s0 = torch.get_rng_state()
    assert not torch.equal(s0, ref_rng)


def test_tdummy_live_batch_takes_stock_path(TL):
    m = make_module(); feats = live_feats(); pair_mask = torch.ones(2, N, N)[:1].expand(1, N, N).contiguous()
    zs = [torch.randn(1, N, N, TOKEN_Z, generator=torch.Generator().manual_seed(20 + k)) for k in range(2)]
    torch.manual_seed(7); ref, ref_rng = run_calls(m, feats, zs, pair_mask)
    TL.install(["tdummy"], model_hook=False); TL.begin_forward()
    torch.manual_seed(7); got, got_rng = run_calls(m, feats, zs, pair_mask)
    assert all(torch.equal(a, b) for a, b in zip(ref, got)) and torch.equal(ref_rng, got_rng)
    assert TL.STATS["live"] == 2 and TL.STATS["elided"] == 0


def test_tfeat_reuse_is_bitwise_and_generation_scoped(TL):
    m = make_module(); feats = live_feats(); pair_mask = torch.ones(1, N, N)
    zs = [torch.randn(1, N, N, TOKEN_Z, generator=torch.Generator().manual_seed(30 + k)) for k in range(4)]
    torch.manual_seed(5); ref, ref_rng = run_calls(m, feats, zs, pair_mask)
    TL.install(["tfeat"], model_hook=False); TL.begin_forward()
    torch.manual_seed(5); got, got_rng = run_calls(m, feats, zs, pair_mask)
    assert all(torch.equal(a, b) for a, b in zip(ref, got)) and torch.equal(ref_rng, got_rng)
    assert TL.STATS["featurize_computed"] == 1 and TL.STATS["featurize_reused"] == 3
    # next item: a new generation and a new feats dict => recomputed, and correct for the NEW features
    feats2 = live_feats(seed=2); TL.begin_forward()
    torch.manual_seed(6); got2, _ = run_calls(m, feats2, zs[:2], pair_mask)
    TL.uninstall()
    torch.manual_seed(6); ref2, _ = run_calls(m, feats2, zs[:2], pair_mask)
    assert all(torch.equal(a, b) for a, b in zip(ref2, got2))
    assert TL.STATS["featurize_computed"] == 2


def test_tfeat_and_tdummy_compose(TL):
    m = make_module(); pair_mask = torch.ones(1, N, N)
    zs = [torch.randn(1, N, N, TOKEN_Z, generator=torch.Generator().manual_seed(40 + k)) for k in range(3)]
    for feats in (dummy_feats(), live_feats()):
        TL.uninstall(); torch.manual_seed(9); ref, ref_rng = run_calls(m, feats, zs, pair_mask)
        TL.install(["tfeat", "tdummy"], model_hook=False); TL.begin_forward()
        torch.manual_seed(9); got, got_rng = run_calls(m, feats, zs, pair_mask)
        assert all(torch.equal(a, b) for a, b in zip(ref, got)) and torch.equal(ref_rng, got_rng)


def test_end_forward_drops_cache(TL):
    m = make_module(); feats = live_feats(); pair_mask = torch.ones(1, N, N)
    TL.install(["tfeat"], model_hook=False); TL.begin_forward()
    with torch.no_grad():
        m(torch.randn(1, N, N, TOKEN_Z), feats, pair_mask, False)
    assert TL._CACHE_ATTR in m.__dict__

    class Model:  # the wrapper's view of Boltz2: .template_module
        template_module = m
    TL.end_forward(Model())
    assert TL._CACHE_ATTR not in m.__dict__
