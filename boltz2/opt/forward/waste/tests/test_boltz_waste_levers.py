"""boltz_waste_levers (BOLTZ_WASTE=chunkcast,opmmask; opt/forward/waste/boltz_waste_levers.py): the patched chunked forwards are BIT-IDENTICAL to
the stock Boltz-2 2.2.1 modules under the trunk's regime (eval, CUDA autocast bf16) for m in {bf16 (layer-0 PWA), fp32 (every other call)}, S with
a ragged 64-row tail, a non-trivial MSA mask and token mask, chunked and unchunked (unchunked = pure delegation), under no_grad AND inference_mode
(inference tensors carry no version counter); the OPM num_mask cache serves a second call, keys on the live mask tensor object and empties when
that tensor dies; outside the regime (autocast off) the verbatim stock statements run; apply() steps aside by name from a foreign forward.
Parametrization and the random-weights/non-trivial-mask construction follow test_msa2_hoist.py. GPU test: skipped without CUDA or without boltz importable."""
import contextlib
import gc
import importlib.util
import itertools
import os
import sys

try:                                    # pytest when present (the kit's suite); plain `python3 test_boltz_waste_levers.py` on a box without it
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required", allow_module_level=True)
    if importlib.util.find_spec("boltz") is None:
        pytest.skip("boltz not importable", allow_module_level=True)
    parametrize = pytest.mark.parametrize
    raises = pytest.raises
except ModuleNotFoundError:
    pytest = None
    import torch
    assert torch.cuda.is_available() and importlib.util.find_spec("boltz") is not None, "CUDA + boltz required"

    def parametrize(names, values):
        def deco(fn):
            fn.__dict__.setdefault("_params", []).insert(0, (names, values)); return fn
        return deco

    @contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import boltz_waste_levers as W  # noqa: E402

from boltz.model.layers.pair_averaging import PairWeightedAveraging as P  # noqa: E402
from boltz.model.layers.outer_product_mean import OuterProductMean as O  # noqa: E402
from boltz.model.layers.transition import Transition as T  # noqa: E402

STOCK_P, STOCK_O, STOCK_T = P.forward, O.forward, T.forward     # captured before anything patches the classes


def _rand_modules(seed=0):
    torch.manual_seed(seed)
    pwa = P(c_m=64, c_z=128, c_h=32, num_heads=8).cuda().eval()
    opm = O(c_in=64, c_hidden=32, c_out=128).cuda().eval()
    trn = T(dim=64, hidden=256).cuda().eval()
    trz = T(dim=128, hidden=512).cuda().eval()          # the MSA-module pair transition (chunk 64)
    for mod in (pwa, opm, trn, trz):
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.2)            # final_init_ zeroes proj_o: make every path carry signal
    return pwa, opm, trn, trz


def _inputs(S, N, m_dtype, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = (torch.randn(1, S, N, 64, device="cuda", generator=g) * 2).to(m_dtype)
    z = torch.randn(1, N, N, 128, device="cuda", generator=g) * 3
    msa_mask = (torch.rand(1, S, N, device="cuda", generator=g) > 0.15).to(torch.int64)
    tok = (torch.rand(1, N, device="cuda", generator=g) > 0.05).float()
    token_mask = tok[:, :, None] * tok[:, None, :]
    return m, z, msa_mask, token_mask


@parametrize("grad_ctx", ["no_grad", "inference_mode"])
@parametrize("m_dtype", [torch.bfloat16, torch.float32])
@parametrize("S,N,chunked", [(48, 40, False), (200, 400, True), (333, 390, True)])
@parametrize("div_out", [False, True])
def test_patched_forwards_bitwise_equal_stock(S, N, chunked, m_dtype, grad_ctx, div_out):
    pwa, opm, trn, trz = _rand_modules()
    ctx = torch.no_grad if grad_ctx == "no_grad" else torch.inference_mode
    with ctx():
        m, z, msa_mask, token_mask = _inputs(S, N, m_dtype)
        cs_trn, cs_trz, cs_opm = (32, 64, 4) if chunked else (None, None, None)
        f_p = W._make_pwa_forward(STOCK_P); f_t = W._make_transition_forward(STOCK_T); f_o = W._make_opm_forward(STOCK_O, cast_hoist=True, mask_cache=True, div_out=div_out)
        W._NUMMASK.clear()
        hits0 = W.STATS["opm_nummask_reused"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref_p = STOCK_P(pwa, m, z, token_mask, chunked)
            ref_t = STOCK_T(trn, m, cs_trn)
            ref_tz = STOCK_T(trz, z, cs_trz)
            ref_o = STOCK_O(opm, m, msa_mask, cs_opm)
            got_p = f_p(pwa, m, z, token_mask, chunked)
            got_t = f_t(trn, m, cs_trn)
            got_tz = f_t(trz, z, cs_trz)
            got_o = f_o(opm, m, msa_mask, cs_opm)
            got_o2 = f_o(opm, m, msa_mask, cs_opm)          # second call: num_mask served from the cache (chunked)
        for name, ref, got in (("pwa", ref_p, got_p), ("transition_msa", ref_t, got_t), ("transition_z", ref_tz, got_tz), ("opm", ref_o, got_o), ("opm_cached", ref_o, got_o2)):
            assert ref.dtype == got.dtype and ref.shape == got.shape, (name, ref.dtype, got.dtype, ref.shape, got.shape)
            assert torch.equal(ref, got), f"{name}: max|diff|={(ref.float() - got.float()).abs().max().item()} (S={S} N={N} chunked={chunked} m={m_dtype} {grad_ctx})"
        if chunked:
            assert W.STATS["opm_nummask_reused"] == hits0 + 1
            assert len(W._NUMMASK) == 1 and next(iter(W._NUMMASK.values()))[0]() is msa_mask
            # a different mask tensor object (even with equal values) never hits: identity, not value, is the key
            msa_mask2 = msa_mask.clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                got_o3 = f_o(opm, m, msa_mask2, cs_opm)
            assert torch.equal(ref_o, got_o3) and W.STATS["opm_nummask_reused"] == hits0 + 1 and len(W._NUMMASK) == 2
            del msa_mask2, msa_mask
            gc.collect()
            assert len(W._NUMMASK) == 0, "num_mask entries must be released when the mask tensor dies"
        else:
            assert len(W._NUMMASK) == 0                     # unchunked = pure delegation, nothing cached


@parametrize("m_dtype", [torch.bfloat16, torch.float32])
def test_outside_regime_runs_stock_statements(m_dtype):
    """autocast OFF (not the trunk's regime): the chunked bodies run with no cast hoisted (xq = x) = the verbatim stock statements -> same bits as stock."""
    pwa, opm, trn, trz = _rand_modules()
    with torch.no_grad():
        m, z, msa_mask, token_mask = _inputs(130, 70, m_dtype)
        m32 = m.float()                                      # stock's chunked matmuls need matching dtypes without autocast: use fp32 everywhere
        c0 = (W.STATS["transition_cast_hoisted"], W.STATS["pwa_cast_hoisted"], W.STATS["opm_cast_hoisted"])
        ref_t = STOCK_T(trn, m32, 32); got_t = W._make_transition_forward(STOCK_T)(trn, m32, 32)
        ref_p = STOCK_P(pwa, m32, z, token_mask, True); got_p = W._make_pwa_forward(STOCK_P)(pwa, m32, z, token_mask, True)
        W._NUMMASK.clear()
        ref_o = STOCK_O(opm, m32, msa_mask, 4); got_o = W._make_opm_forward(STOCK_O, True, True, True)(opm, m32, msa_mask, 4)
        assert torch.equal(ref_t, got_t) and torch.equal(ref_p, got_p) and torch.equal(ref_o, got_o)
        assert (W.STATS["transition_cast_hoisted"], W.STATS["pwa_cast_hoisted"], W.STATS["opm_cast_hoisted"]) == c0, "no cast may be hoisted outside bf16 autocast"
        W._NUMMASK.clear()


def test_delegation_of_unchunked_calls_goes_to_previous_forward():
    calls = []

    def prev(self, *a, **k):
        calls.append((a, k)); return "prev"
    pwa, opm, trn, _ = _rand_modules()
    x = torch.zeros(2, 64, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert W._make_transition_forward(prev)(trn, x) == "prev"                 # chunk_size None -> previous forward (the fused-transition dispatcher in the kit)
        assert W._make_transition_forward(prev)(trn, x, None) == "prev"
        assert W._make_pwa_forward(prev)(pwa, x, x, x, False) == "prev"            # chunk_heads False
        assert W._make_opm_forward(prev, True, True)(opm, x, x) == "prev"          # chunk_size None
        assert W._make_opm_forward(prev, True, True)(opm, x, x, None) == "prev"
    assert len(calls) == 5


def test_apply_env_words_and_foreign_forward():
    W.remove()
    with raises(RuntimeError):
        W.apply({"BOLTZ_WASTE": "chunkcast,nonsense"})
    assert W.apply({}) == () and W.report()["applied"] and W.report()["levers"] == []
    W.remove()

    def foreign(self, m, z, mask, chunk_heads=False):        # e.g. a wholesale kernel replacement installed by another lever
        return None
    foreign.__module__ = "some_other_lever"
    P.forward = foreign
    try:
        got = W.apply({"BOLTZ_WASTE": "chunkcast,opmmask,opmdiv", "BOLTZ_WASTE_VERBOSE": "0"})
        assert got == ("chunkcast", "opmmask", "opmdiv")
        rep = W.report()
        assert P.forward is foreign, "a foreign forward must be left alone"
        assert "PairWeightedAveraging" not in rep["patched"] and "Transition" in rep["patched"] and "OuterProductMean" in rep["patched"]
        assert any("foreign_forward:some_other_lever" in s for s in rep["state"]["chunkcast"]["skipped"])
        lines = W.lever_lines()
        assert any(l.startswith("[boltz2-opt] LEVER name=W1.chunkcast state=on") and "foreign_forward:some_other_lever" in l for l in lines), lines
        assert any(l.startswith("[boltz2-opt] LEVER name=W1.opmmask state=on") for l in lines)
        assert T.forward is not STOCK_T and O.forward is not STOCK_O
    finally:
        W.remove()
        restored = (T.forward is STOCK_T and O.forward is STOCK_O and P.forward is foreign)   # remove() restores exactly what apply() replaced
        P.forward = STOCK_P
    assert restored


def test_skip_words():
    W.remove()
    try:
        W.apply({"BOLTZ_WASTE": "chunkcast,opmmask", "BOLTZ_MSA_KERNELS": "opm,pwa", "BOLTZ_WASTE_VERBOSE": "0"})
        rep = W.report()
        assert rep["state"]["opmmask"]["state"] == "skipped" and "BOLTZ_MSA_KERNELS=opm" in rep["state"]["opmmask"]["reason"]
        assert rep["patched"] == ["Transition"]
        assert O.forward is STOCK_O and P.forward is STOCK_P
    finally:
        W.remove()
    assert T.forward is STOCK_T


if __name__ == "__main__":          # pytest-free runner: expands the parametrize grids and runs every test_* in this module
    mod = sys.modules[__name__]; n_ok = 0
    for name in sorted(k for k in dir(mod) if k.startswith("test_")):
        fn = getattr(mod, name); params = fn.__dict__.get("_params", [])
        if not params:
            grids = [dict()]
        else:
            keys = [p[0] for p in params]; vals = [p[1] for p in params]
            grids = []
            for combo in itertools.product(*vals):
                kw = {}
                for k, v in zip(keys, combo):
                    ks = [x.strip() for x in k.split(",")]
                    kw.update(dict(zip(ks, v if len(ks) > 1 else (v,))))
                grids.append(kw)
        for kw in grids:
            fn(**kw); n_ok += 1; print(f"ok  {name} {kw}", flush=True)
    print(f"ALL PASSED: {n_ok} test invocations")
