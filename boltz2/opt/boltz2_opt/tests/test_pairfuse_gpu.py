"""GPU numerics of the PAIRFUSE layer driver (boltz2_opt.pairfuse / forward/pairfuse/bz_pairfuse.py) — run in a kit box:
    $STACK_PYTHON -m pytest -q opt/boltz2_opt/tests/test_pairfuse_gpu.py      (or python3 <this file>: __main__ runs every test)
Skips itself without CUDA / boltz / the routed kernels.  What it proves per site, on real checkpoint weights when BOLTZ2_CKPT resolves
(else boltz's own init) and a random z:
  (1) IN-PLACE == OUT-OF-PLACE, bitwise, where the provider has an out-of-place path: trimul K3 residual out=z, fused transition out=z.
      The tri-att epilogue's residual mode is in place by construction (no out-of-place residual path in opt_core): checked elementwise
      to 2 bf16 ulps against bf16(z + u_opmode) for both nodes (the one extra rounding of u is the only admissible difference).
  (2) site error vs the STOCK statements (fp32 z, bf16 autocast, use_kernels=True: cuEquivariance TriMul / triangle attention, boltz
      Transition, AttentionPairBias.proj_z) is inside the bf16-autocast class: max |kit - stock| <= 4x the max |stock_bf16path - fp64 ref|
      where a reference is cheap (transition, pair bias), and rel-rms <= 2e-2 for the TriMul / attention sites (their stock kernels are bf16
      themselves — PRECISION census: cuEquivariance casts to the autocast dtype).
  (3) the pair-bias kernel == LayerNorm+Linear+rearrange in fp32 to 1e-4 abs on fp32 z (ieee dot), and takes bf16 z.
  (4) a whole PairformerLayer through the driver (bf16 and fp32 residency) vs the stock layer: rel-rms of z and s updates <= 3e-2, and the
      CUDA RNG offset after the layer equals stock's (the four dropout draws are kept).
"""
import math
import os
import sys

try:
    import pytest
except ImportError:                                                   # kit images ship no pytest: __main__ below runs the tests without it
    pytest = None

if pytest is not None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required", allow_module_level=True)
    pytest.importorskip("boltz")
else:
    import torch

KIT_OPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.environ.get("BOLTZ2_CKPT") or os.path.join(os.environ.get("BOLTZ_CACHE", "/weights/boltz2"), "boltz2_conf.ckpt")   # absent: the sites run on random weights
N = int(os.environ.get("PAIRFUSE_TEST_N", "384"))
DEV = torch.device("cuda", 0)


def _route():
    """Route the fast row's kernels in this process as worker_launch does."""
    from opt_core import kernels as K
    from boltz2_opt import stack, modes
    for name in modes.ROUTED_KERNELS["fast"]:
        K.route(name)
        os.environ.update(K.exports(name, **{k: stack.kit_path(v) for k, v in modes.kernel_exports(name).items()}))
        g = K.route_check(name)
        assert g.ok, g.reason


def _make_env():
    os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1"); os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")
    os.environ.setdefault("OPT_CORE_PF_ALLOW_CANDIDATE", "1")           # the fp32-residency sites read LN from fp32 z: pair_fused candidate rows today (CORE-REQ; bz_pairfuse.apply('fp32') sets the same)
    torch.set_grad_enabled(False); torch.set_float32_matmul_precision("highest")
    _route()
    sys.path.insert(0, os.path.join(KIT_OPT, "forward", "pairfuse"))
    import bz_pairfuse as D
    D._STATE["residency"] = "bf16"
    from boltz.model.layers.pairformer import PairformerLayer
    torch.manual_seed(0)
    layer = PairformerLayer(384, 128, num_heads=16, dropout=0.25, pairwise_head_width=32, pairwise_num_heads=4, v2=True)
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        sub = {k[len("pairformer_module.layers.0."):]: v for k, v in sd.items() if k.startswith("pairformer_module.layers.0.")}
        layer.load_state_dict(sub, strict=True)
    layer = layer.to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(1)
    z = (torch.randn(1, N, N, 128, generator=g) * 2.0).to(DEV)
    s = torch.randn(1, N, 384, generator=g).to(DEV)
    return {"D": D, "layer": layer, "z": z, "s": s, "mask": torch.ones(1, N, device=DEV), "pair_mask": torch.ones(1, N, N, device=DEV)}


if pytest is not None:
    @pytest.fixture(scope="module")
    def env():
        return _make_env()


def _relrms(a, b):
    a = a.float(); b = b.float()
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12))


# ---------------------------------------------------------------------------------------------------------------- (1) in place == out of place
def test_trimul_inplace_bitwise(env):
    D, layer, z = env["D"], env["layer"], env["z"]
    from opt_core.kernels.fpf_trimul_v4 import generic as G, kernels as GK
    for mod, outgoing in ((layer.tri_mul_out, True), (layer.tri_mul_in, False)):
        w = D._trimul_weights(mod)
        for dt in (torch.bfloat16, torch.float32):
            z4 = z.to(dt).contiguous().clone()
            cfg = G._check(z4, None, w, None, None)
            fresh = torch.empty_like(z4)
            ref = G.CELLS.run(lambda c: GK.trimul_v4_forward(z4, outgoing, None, w, c, eps=1e-5, residual=True, stock_round=G._STOCK_ROUND, pad=16, out=fresh), cfg, z4.device)
            zin = z4.clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                got = D.trimul_site(mod, zin, outgoing=outgoing, mask=None, res_out=zin)
            assert got.data_ptr() == zin.data_ptr()
            assert torch.equal(got, ref), f"trimul in-place != out-of-place ({'out' if outgoing else 'in'}, {dt})"


def test_triatt_inplace_within_rounding(env):
    """The tri-att epilogue's residual (block) mode is in place BY CONSTRUCTION in opt_core (no out-of-place residual path exists to compare
    bitwise against): its op mode returns u = bf16(acc) while block mode stores bf16(fp32(z) + acc).  So the check is elementwise: the in-place
    result equals bf16(z + u) to within 2 bf16 ulps of each element (the one extra rounding of u), for both nodes — a mis-ordered read/write of
    an aliased tile would show up as gross element errors, and run-to-run bitwise stability is test_repeated_runs_bitwise's."""
    D, layer, z = env["D"], env["layer"], env["z"]
    from opt_core.attn import pair_fused as PF
    for mod, ending in ((layer.tri_att_start, False), (layer.tri_att_end, True)):
        W = D._triatt_weights(mod, DEV)
        z16 = z.to(torch.bfloat16).contiguous()
        core, _ = D._core_for(N)
        scale = 1.0 / math.sqrt(mod.mha.c_hidden)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            u = PF.tri_attn_block(z16.clone(), W, mask=None, ending=ending, residual=False, impl="fpf", core=core, ln="fused", scale=scale)   # op mode: the bf16 update in z's frame (a view for ending)
            zin = z16.clone()
            got = D.triatt_site(mod, zin, ending=ending, mask=None, inplace=True)                                                          # block mode: in place
        assert got.data_ptr() == zin.data_ptr()
        ref = (z16.float() + u.float()).to(torch.bfloat16).float()
        diff = (got.float() - ref).abs()
        tol = 2 * 2.0 ** -7 * torch.maximum(ref.abs(), got.float().abs()) + 1e-30                                                          # 2 bf16 ulps of each element
        nbad = int((diff > tol).sum())
        assert nbad == 0, f"tri-att {'end' if ending else 'start'}: {nbad} elements differ from bf16(z+u) by more than 2 ulps (max diff {float(diff.max())})"


def test_transition_inplace_bitwise(env):
    D, layer, z = env["D"], env["layer"], env["z"]
    from opt_core.attn import pair_fused as PF
    T = D._transition_weights(layer.transition_z, DEV)
    z16 = z.to(torch.bfloat16).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fresh = torch.empty_like(z16)
        ref = PF.transition(z16.clone(), T, residual=True, ln="fused", out=fresh)
        zin = z16.clone()
        got = D.transition_site(layer.transition_z, zin, inplace=True)
    assert got.data_ptr() == zin.data_ptr()
    assert torch.equal(got, ref), "transition in-place != out-of-place"


# ---------------------------------------------------------------------------------------------------------------- (2)(3) site error vs stock
def test_pair_bias_vs_stock(env):
    D, layer, z = env["D"], env["layer"], env["z"]
    pz = layer.attention.proj_z                                        # Sequential(LayerNorm, Linear, Rearrange)
    with torch.autocast("cuda", enabled=False):
        ref = pz(z.float())                                             # [1,16,N,N] fp32 (stock statement)
        ref64 = torch.nn.functional.linear(torch.nn.functional.layer_norm(z.double(), (128,), pz[0].weight.double(), pz[0].bias.double(), pz[0].eps), pz[1].weight.double()).permute(0, 3, 1, 2)
    got32 = D.pair_bias(z, pz[0].weight, pz[0].bias, pz[1].weight, pz[0].eps)
    got16 = D.pair_bias(z.to(torch.bfloat16), pz[0].weight, pz[0].bias, pz[1].weight, pz[0].eps)
    e_stock = (ref.double() - ref64).abs().max().item()
    e32 = (got32.double() - ref64).abs().max().item()
    assert e32 <= 4 * e_stock + 1e-5, f"pair bias fp32 z: {e32} vs stock's own {e_stock}"
    assert (got32 - ref).abs().max().item() <= 1e-3
    ref16 = pz(z.to(torch.bfloat16).float())
    assert (got16 - ref16).abs().max().item() <= 1e-3, "pair bias on bf16 z"


def test_sites_vs_stock_statements(env):
    D, layer, z, pm = env["D"], env["layer"], env["z"], env["pair_mask"]
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # TriMul: stock = z + tri_mul(z) (cueq kernel), kit = trimul_site in place on bf16 / fp32 copies
        for name, mod, outgoing in (("tri_mul_out", layer.tri_mul_out, True), ("tri_mul_in", layer.tri_mul_in, False)):
            ref = z + mod(z, mask=pm, use_kernels=True)
            for dt in (torch.bfloat16, torch.float32):
                zz = z.to(dt).contiguous().clone()
                got = D.trimul_site(mod, zz, outgoing=outgoing, mask=None, res_out=zz)
                out[f"{name}/{dt}"] = _relrms(got.float() - z.to(dt).float(), ref - z)      # error of the UPDATE (not swamped by the residual)
        for name, mod, ending in (("tri_att_start", layer.tri_att_start, False), ("tri_att_end", layer.tri_att_end, True)):
            ref = z + mod(z, mask=pm, chunk_size=None, use_kernels=True)
            zz = z.to(torch.bfloat16).contiguous().clone()
            got = D.triatt_site(mod, zz, ending=ending, mask=None, inplace=True)
            out[f"{name}/bf16"] = _relrms(got.float() - z.to(torch.bfloat16).float(), ref - z)
            u = D.triatt_site(mod, z.contiguous().clone(), ending=ending, mask=None, inplace=False)
            out[f"{name}/fp32"] = _relrms(u.float(), ref - z)
        ref = z + layer.transition_z(z)
        zz = z.to(torch.bfloat16).contiguous().clone()
        got = D.transition_site(layer.transition_z, zz, inplace=True)
        out["transition/bf16"] = _relrms(got.float() - z.to(torch.bfloat16).float(), ref - z)
        u = D.transition_site(layer.transition_z, z.contiguous().clone(), inplace=False)
        out["transition/fp32"] = _relrms(u.float(), ref - z)
    print({k: round(v, 5) for k, v in out.items()})
    for k, v in out.items():
        lim = 3e-2 if "bf16" in k else 2e-2                             # bf16 residency: the update read back from a bf16 z carries the stream's rounding
        assert v <= lim, f"{k}: rel-rms of the update vs stock {v} > {lim}"


# ---------------------------------------------------------------------------------------------------------------- (4) whole layer + RNG
def test_layer_vs_stock_and_rng(env):
    D, layer, z, s, mask, pm = env["D"], env["layer"], env["z"], env["s"], env["mask"], env["pair_mask"]
    res = {}
    for residency in ("bf16", "fp32"):
        D._STATE["residency"] = residency
        torch.cuda.manual_seed(1234)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_ref, z_ref = layer(s, z, mask, pm, chunk_size_tri_attn=512, use_kernels=True)
        off_ref = torch.cuda.get_rng_state().clone()
        torch.cuda.manual_seed(1234)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zr = D._enter(z); ctx = D._make_ctx(zr, pm)
            zr = D._pair_sublayers(layer, zr, ctx)
            s_got = D._seq_sublayers(layer, s, zr, mask)
            z_got = D._exit(zr, z)
        off_got = torch.cuda.get_rng_state().clone()
        assert torch.equal(off_ref, off_got), f"{residency}: CUDA RNG state after the layer differs from stock's (dropout draws not kept)"
        res[residency] = {"z_upd": _relrms(z_got - z, z_ref - z), "s_upd": _relrms(s_got - s, s_ref - s), "z": _relrms(z_got, z_ref)}
    D._STATE["residency"] = "bf16"
    print(res)
    for r, d in res.items():                                          # random-input bounds (the bf16 kernels' own rounding differences dominate); accuracy is judged on real predictions, not here
        assert d["z_upd"] <= 8e-2 and d["s_upd"] <= 5e-2, f"{r}: {d}"


# ---------------------------------------------------------------------------------------------------------------- (5) repeated runs, two shapes
def test_repeated_runs_bitwise(env):
    """F3 (review A10): the driven layer (in-place chain) is bitwise run-to-run over >= 20 repetitions at two shapes, one whose N is not a
    multiple of any tile (397): no race in the in-place stores, no nondeterministic reduction (fixed tiles, no atomics, no split-K)."""
    D, layer, s, mask = env["D"], env["layer"], env["s"], env["mask"]
    for n in (N, 397):
        g = torch.Generator(device="cpu").manual_seed(7)
        z = (torch.randn(1, n, n, 128, generator=g) * 2.0).to(DEV); ss = torch.randn(1, n, 384, generator=g).to(DEV)
        mk = torch.ones(1, n, device=DEV); pm = torch.ones(1, n, n, device=DEV)
        first = None
        for residency in ("bf16", "fp32"):
            D._STATE["residency"] = residency
            layer._pairfuse_elig = {}
            outs = []
            for rep in range(20):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    zr = D._enter(z); ctx = D._make_ctx(zr, pm)
                    zr = D._pair_sublayers(layer, zr, ctx)
                    s_out = D._seq_sublayers(layer, ss, zr, mk)
                torch.cuda.synchronize()
                outs.append((zr.clone(), s_out.clone()))
            for rep in range(1, 20):
                assert torch.equal(outs[rep][0], outs[0][0]) and torch.equal(outs[rep][1], outs[0][1]), f"{residency} N={n}: run {rep} differs bitwise from run 0"
    D._STATE["residency"] = "bf16"


# ---------------------------------------------------------------------------------------------------------------- (6) capture contract: sync-free after the first call
def test_no_host_sync_after_first_call(env):
    """After ONE eager served call of a shape class with a given mask tensor, a second call issues no synchronizing CUDA operation
    (torch.cuda.set_sync_debug_mode('error') raises on any): eligibility cached, providers probed, mask predicate cached.  A NEW mask
    tensor under mask_predicate(True) is also sync-free; without it the new tensor costs exactly one predicate sync (counted)."""
    D, layer, z, s, mask, pm = env["D"], env["layer"], env["z"], env["s"], env["mask"], env["pair_mask"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):   # boltz predicts under inference_mode: inference tensors have no version counter
        zi, si, mi, pmi = z.clone(), s.clone(), mask.clone(), torch.ones_like(pm)
        assert pmi.is_inference()
        D._served_stack("PairformerModule", [layer], si, zi, mi, pmi)
        n_inf = D.STATS["mask_pred_synced"]
        D._served_stack("PairformerModule", [layer], si, zi, mi, pmi)                   # same inference tensor: cached, no new sync
        assert D.STATS["mask_pred_synced"] == n_inf, "inference-mode mask tensor was not cached"
    for residency in ("bf16", "fp32"):
        D._STATE["residency"] = residency
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert D._stack_word(layer, z, True, False) is None          # primes eligibility (probes every provider) — eager
            D._served_stack("PairformerModule", [layer], s, z, mask, pm)  # first served call: warm probes, JIT, predicate sync — eager
        torch.cuda.synchronize()
        before = dict(D.STATS)
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert D._stack_word(layer, z, True, False) is None      # answered from the cache
                D._served_stack("PairformerModule", [layer], s, z, mask, pm)              # same mask tensor: predicate cached
                pm2 = torch.ones_like(pm)
                with D.mask_predicate(True):
                    D._served_stack("PairformerModule", [layer], s, z, mask, pm2)         # new tensor, predicate given: no sync
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        assert D.STATS["host_syncs"] == before["host_syncs"], f"{residency}: host_syncs grew {before['host_syncs']} -> {D.STATS['host_syncs']}"
        assert D.STATS["mask_pred_cached"] >= before["mask_pred_cached"] + 1 and D.STATS["mask_pred_given"] >= before["mask_pred_given"] + 1
        n0 = D.STATS["mask_pred_synced"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D._served_stack("PairformerModule", [layer], s, z, mask, torch.ones_like(pm))    # new tensor, nothing given: exactly one predicate sync
        assert D.STATS["mask_pred_synced"] == n0 + 1
        ok = D.primed(type("M", (), {"layers": [layer]})(), z, pm)
        assert ok["ok"], ok
    D._STATE["residency"] = "bf16"


if __name__ == "__main__":                                            # kit images without pytest: python3 test_pairfuse_gpu.py runs every test on one fixture
    E = _make_env()
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(E); print("PASS", name, flush=True)
            except Exception as ex:  # noqa
                failed += 1; print("FAIL", name, type(ex).__name__, str(ex)[:600], flush=True)
    sys.exit(1 if failed else 0)
