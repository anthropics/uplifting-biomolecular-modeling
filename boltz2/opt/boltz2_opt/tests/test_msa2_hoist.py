"""msa_hoist (BOLTZ_MSA2=hoist, forward/msa2/exact_hoist.py): the hoisted forwards are BIT-IDENTICAL to the stock Boltz-2 2.2.1 modules under the
trunk's regime (eval, CUDA autocast bf16), chunked and unchunked, with a non-trivial MSA mask; and they route to the stock statements by name outside it.
GPU test: skipped without CUDA or without boltz importable (the kit's CPU suite stays stub-only)."""
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if importlib.util.find_spec("boltz") is None:
    pytest.skip("boltz not importable", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "msa2"))


def _hoist():
    if "msa2" not in sys.modules:
        spec = importlib.util.spec_from_file_location("msa2", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
        mod = importlib.util.module_from_spec(spec); sys.modules["msa2"] = mod; spec.loader.exec_module(mod)
    import importlib as il
    return il.import_module("msa2.exact_hoist")


def _rand_modules(seed=0):
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    from boltz.model.layers.outer_product_mean import OuterProductMean
    from boltz.model.layers.transition import Transition
    torch.manual_seed(seed)
    pwa = PairWeightedAveraging(c_m=64, c_z=128, c_h=32, num_heads=8).cuda().eval()
    opm = OuterProductMean(c_in=64, c_hidden=32, c_out=128).cuda().eval()
    trn = Transition(dim=64, hidden=256).cuda().eval()
    for mod in (pwa, opm, trn):
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.2)   # final_init_ zeroes proj_o: make every path carry signal
    return pwa, opm, trn


@pytest.mark.parametrize("S,N,chunked", [(48, 40, False), (200, 400, True), (333, 390, True)])
def test_hoist_bitwise_equals_stock(S, N, chunked):
    H = _hoist()
    pwa, opm, trn = _rand_modules()
    g = torch.Generator(device="cuda").manual_seed(1)
    m = (torch.randn(1, S, N, 64, device="cuda", generator=g) * 2).to(torch.bfloat16)
    z = torch.randn(1, N, N, 128, device="cuda", generator=g) * 3
    msa_mask = (torch.rand(1, S, N, device="cuda", generator=g) > 0.15).to(torch.int64)
    tok = (torch.rand(1, N, device="cuda", generator=g) > 0.05).float()
    token_mask = tok[:, :, None] * tok[:, None, :]
    kw = dict(chunk_heads=chunked, cs_trn=32 if chunked else None, cs_opm=4 if chunked else None)
    from boltz.model.layers.pair_averaging import PairWeightedAveraging as P
    from boltz.model.layers.outer_product_mean import OuterProductMean as O
    from boltz.model.layers.transition import Transition as T
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_p = P.forward(pwa, m, z, token_mask, kw["chunk_heads"])
        ref_t = T.forward(trn, m, kw["cs_trn"])
        ref_o = O.forward(opm, m, msa_mask, kw["cs_opm"])
        H.num_mask_cache_clear()
        got_p = H.pwa_forward(P.forward)(pwa, m, z, token_mask, kw["chunk_heads"])
        got_t = H.transition_forward(T.forward)(trn, m, kw["cs_trn"])
        got_o = H.opm_forward(O.forward)(opm, m, msa_mask, kw["cs_opm"])
        got_o2 = H.opm_forward(O.forward)(opm, m, msa_mask, kw["cs_opm"])   # second call: num_mask served from the cache
    for name, ref, got in (("pwa", ref_p, got_p), ("transition", ref_t, got_t), ("opm", ref_o, got_o), ("opm_cached", ref_o, got_o2)):
        assert ref.dtype == got.dtype and ref.shape == got.shape, (name, ref.dtype, got.dtype, ref.shape, got.shape)
        assert torch.equal(ref, got), f"{name}: max|diff|={(ref.float() - got.float()).abs().max().item()} (S={S} N={N} chunked={chunked})"
    if chunked:
        assert H._NUM_MASK["hits"] >= 1


def test_hoist_routes_to_stock_outside_regime():
    H = _hoist()
    pwa, opm, trn = _rand_modules()
    calls = []

    def stock(self, *a, **k):
        calls.append(1); return "stock"
    m = torch.zeros(1, 2, 2, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():   # autocast OFF: an explicit cast would change numerics -> stock statements by name
        assert H.pwa_forward(stock)(pwa, m, None, None, True) == "stock"
        assert H.transition_forward(stock)(trn, torch.zeros(2, 64, device="cuda"), 32) == "stock"
        assert H.opm_forward(stock)(opm, m, None, 4) == "stock"
    assert len(calls) == 3
    assert H.FALLBACK_BY["pwa"].get("autocast_off", 0) >= 1
    # the transition predicate: a dim the adapter does not accept goes to the previous forward uncounted
    before = dict(H.CENSUS["transition"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert H.transition_forward(stock, accept=lambda mod, x, cs: x.shape[-1] == 64)(trn, torch.zeros(2, 128, device="cuda"), None) == "stock"
    assert H.CENSUS["transition"] == before
