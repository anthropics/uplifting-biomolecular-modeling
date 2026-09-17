"""torch tests (CPU suffices; CUDA cases run when a device is present): policy apply/expect/signature, det_scatter vs the atomic ops,
cast cache equality / refresh / recycled-id safety, precast + misuse guard, apply_torch statements."""
from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")

from opt_core.precision import cast_cache, det_scatter, policy, recipe  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _restore_flags():
    saved = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    yield
    torch.set_float32_matmul_precision(saved[0]); torch.backends.cudnn.allow_tf32 = saved[1]; torch.backends.cudnn.benchmark = saved[2]
    torch.backends.cudnn.deterministic = saved[3]; torch.use_deterministic_algorithms(saved[4], warn_only=saved[5])
    det_scatter.reset()


# ----------------------------------------------------------------- policy -----------------------------------------------------------------
def test_apply_then_expect_and_mismatch():
    torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = True
    rec = policy.apply(policy.FP32_TF32)
    assert rec["policy"] == "fp32_tf32" and "matmul" in rec["changed"] and torch.get_float32_matmul_precision() == "high"
    assert policy.expect(policy.FP32_TF32)["changed"] == []
    with pytest.raises(policy.PolicyMismatch) as ei:
        policy.expect(policy.FP32_STRICT)
    assert ei.value.event == "policy_mismatch" and "matmul" in ei.value.record()["fields"]
    assert policy.apply(policy.UNTOUCHED)["changed"] == []


def test_policy_validation_and_input_precision_table():
    with pytest.raises(ValueError):
        policy.Policy("bad", matmul="tf32")
    with pytest.raises(ValueError):
        policy.Policy("bad", autocast="fp8")
    assert policy.input_precision_for("highest") == "ieee"
    assert policy.input_precision_for("highest", tf32x3=True) == "tf32x3"
    assert policy.input_precision_for("high") == "tf32" and policy.input_precision_for("high", tf32x3=True) == "tf32"
    assert policy.BF16_AUTOCAST.describe() == {"policy": "bf16_autocast", "matmul": "untouched", "cudnn_tf32": "untouched",
                                               "autocast": "bf16", "cudnn_benchmark": "untouched"}


def test_autocast_context_and_signature_guard():
    events = []
    g = policy.NumericsGuard(on_change=events.append, extra=lambda: (("backend", "eager"),))
    assert g.check() is None
    torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
    ev = g.check()
    assert ev is not None and ev["event"] == "numerics_signature_changed" and ev["changed"] == ["cudnn_benchmark"] and events == [ev]
    assert g.check() is None
    with policy.autocast_context(policy.BF16_AUTOCAST, device_type="cpu"):
        x = torch.randn(4, 4) @ torch.randn(4, 4)
        assert x.dtype == torch.bfloat16
    with policy.autocast_context(policy.FP32_STRICT, device_type="cpu"):
        assert (torch.randn(2, 2) @ torch.randn(2, 2)).dtype == torch.float32
    assert policy.signature_diff((("a", 1), ("b", 2)), (("a", 1), ("b", 3), ("c", 0))) == ["b", "c"]


# --------------------------------------------------------------- det_scatter --------------------------------------------------------------
def _atom_index(n_tok=37, max_per=14, gen=None):
    counts = torch.randint(1, max_per, (n_tok,), generator=gen)
    return torch.repeat_interleave(torch.arange(n_tok), counts).to(DEV), n_tok


def test_segment_reduce_matches_scatter_and_is_repeatable():
    gen = torch.Generator().manual_seed(0)
    idx, n_tok = _atom_index(gen=gen)
    src = torch.randn(3, idx.numel(), 8, generator=gen).to(DEV)
    ref_sum = torch.zeros(3, n_tok, 8, device=DEV).scatter_add_(1, idx[None, :, None].expand(3, -1, 8), src)
    got_sum = det_scatter.segment_reduce_rows(src, idx, n_tok, "sum")
    assert torch.allclose(got_sum, ref_sum, rtol=1e-5, atol=1e-5)
    ref_mean = torch.zeros(3, n_tok, 8, device=DEV).index_reduce(1, idx, src, "mean", include_self=False)
    got_mean = det_scatter.scatter_mean_into(torch.zeros(3, n_tok, 8, device=DEV), -2, idx, src)
    assert torch.allclose(got_mean, ref_mean, rtol=1e-5, atol=1e-5)
    again = det_scatter.scatter_mean_into(torch.zeros(3, n_tok, 8, device=DEV), 1, idx, src)
    assert torch.equal(got_mean, again)                                   # bit-exact run to run
    c = det_scatter.census()
    assert c["calls"] == 2 and c["served"] == 2 and c["fallbacks"] == 0 and c["table_builds"] == 1 and c["tables"] == 1


def test_scatter_reduce_broadcast_index_bf16_and_permuted_layout():
    gen = torch.Generator().manual_seed(1)
    idx, n_tok = _atom_index(gen=gen)
    perm = torch.randperm(idx.numel(), generator=gen).to(DEV)
    idx_p = idx[perm]                                                     # non-contiguous segments
    src = torch.randn(2, idx.numel(), 4, generator=gen).to(DEV).to(torch.bfloat16)
    idx_b = idx_p[None, :].expand(2, -1)                                  # identical rows
    got = det_scatter.scatter_reduce(src, idx_b, dim=-2, dim_size=n_tok, reduce="mean")
    ref = torch.zeros(2, n_tok, 4, device=DEV, dtype=torch.float32).index_reduce(1, idx_p, src.float(), "mean", include_self=False)
    assert got.dtype == torch.bfloat16 and torch.allclose(got.float(), ref, rtol=1e-2, atol=1e-2)
    got_src_acc = det_scatter.scatter_reduce(src, idx_p, dim=1, dim_size=n_tok, reduce="sum", accumulate="src")
    assert got_src_acc.shape == (2, n_tok, 4)


def test_unserved_layouts_are_counted_or_named():
    src = torch.randn(5, 3, device=DEV); idx = torch.tensor([0, 0, 1, 1, 2], device=DEV)
    calls = []

    def stock(src, index, dim, out, dim_size, reduce):
        calls.append(reduce); return "stock"
    assert det_scatter.scatter_reduce(src[None], idx, dim=-2, out=torch.zeros(1, 3, 3, device=DEV), reduce="sum", fallback=stock) == "stock"
    assert det_scatter.scatter_reduce(src[None], idx, dim=-1, reduce="sum", fallback=stock) == "stock"
    assert det_scatter.scatter_reduce(src[None], idx, dim=-2, reduce="max", fallback=stock) == "stock"
    ragged = torch.stack([idx, idx.flip(0)])
    assert det_scatter.scatter_reduce(src[None].expand(2, -1, -1), ragged, dim=-2, reduce="sum", fallback=stock) == "stock"
    c = det_scatter.census()
    assert c["fallbacks"] == 4 and c["served"] == 0 and c["fallback_reasons"] == {"out_given": 1, "dim": 1, "reduce_max": 1, "index_rows_differ": 1}
    with pytest.raises(det_scatter.LayoutUnsupported) as ei:
        det_scatter.scatter_mean_into(torch.zeros(3, 3, device=DEV), 0, idx[None], src)
    assert ei.value.event == "det_scatter_layout" and ei.value.record()["layout"] == "index_rank"


def test_table_refreshes_when_index_changes_in_place():
    idx = torch.tensor([0, 0, 1, 2, 2], device=DEV); src = torch.arange(10., device=DEV).reshape(5, 2)
    a = det_scatter.segment_reduce_rows(src, idx, 3, "sum")
    idx[2] = 0                                                            # in-place edit bumps _version
    b = det_scatter.segment_reduce_rows(src, idx, 3, "sum")
    assert det_scatter.census()["table_builds"] == 2
    assert torch.equal(a[1], src[2]) and torch.equal(b[1], torch.zeros(2, device=DEV)) and torch.equal(b[0], src[0] + src[1] + src[2])


@pytest.mark.skipif(DEV != "cuda", reason="CUDA graph replay needs a device")
def test_segment_reduce_under_cuda_graph_replay():
    idx, n_tok = _atom_index()
    static = torch.randn(idx.numel(), 16, device=DEV)
    det_scatter.segment_reduce_rows(static, idx, n_tok)                   # builds the table outside capture (host sync lives here)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = det_scatter.segment_reduce_rows(static, idx, n_tok)
    static.copy_(torch.randn_like(static)); g.replay(); torch.cuda.synchronize()
    assert torch.equal(out, det_scatter.segment_reduce_rows(static, idx, n_tok))


# ---------------------------------------------------------------- cast_cache --------------------------------------------------------------
def test_cast_cache_identity_hits_refresh_and_recycled_id():
    lin = torch.nn.Linear(8, 4).to(DEV)
    cc = cast_cache.CastCache("t")
    w16 = cc.get(lin.weight, "bf16")
    assert torch.equal(w16, lin.weight.detach().to(torch.bfloat16)) and cc.census()["misses"] == 1
    assert cc.get(lin.weight, "bf16") is w16 and cc.census()["hits"] == 1
    wt = cc.get(lin.weight, "bf16", transform=lambda t: t.t().contiguous(), key_extra="T")
    assert wt.shape == (8, 4) and cc.census()["entries"] == 2
    with torch.no_grad():
        lin.weight.add_(1.0)                                              # in-place update -> version bump -> refresh
    w16b = cc.get(lin.weight, "bf16")
    assert torch.equal(w16b, lin.weight.detach().to(torch.bfloat16)) and cc.census()["refreshes"] == 1
    tmp = lin.weight.detach() * 2                                         # a temporary: served uncached
    cc.get(tmp, "bf16")
    assert cc.census()["uncached"] == 1
    # recycled id: drop a Parameter, make new ones until CPython reuses the address; the cache must not serve the stale copy
    old = torch.nn.Parameter(torch.full((3,), 7.0, device=DEV)); cc.get(old, "bf16"); old_id = id(old)
    del old; gc.collect()
    for _ in range(200):
        new = torch.nn.Parameter(torch.full((3,), -1.0, device=DEV))
        if id(new) == old_id:
            assert torch.equal(cc.get(new, "bf16"), torch.full((3,), -1.0, device=DEV, dtype=torch.bfloat16))
            break
    assert cast_cache.verify_identical(lin.weight, "bf16", cc)
    assert cc.census()["bytes"] > 0 and cc.clear() >= 2


def test_precast_linear_inplace_guard_and_autocast_equivalence():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).to(DEV).eval()
    ref = torch.nn.Sequential(*[m for m in model])                      # same modules (shared) — take the reference BEFORE precast
    x = torch.randn(5, 16, device=DEV)
    dev_type = "cuda" if DEV == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(dev_type, dtype=torch.bfloat16):
        y_stock = ref(x).clone()                                          # fp32 masters, autocast casts per call
    rec = cast_cache.precast_linear_(model, "bf16", predicate=lambda name, m: name != "2")   # leave the last Linear fp32
    assert rec["modules"] == 1 and rec["params"] == 2 and rec["skipped"] == 1 and rec["guard"] == "on" and rec["bytes_after"] * 2 == rec["bytes_before"]
    with torch.no_grad(), torch.autocast(dev_type, dtype=torch.bfloat16):
        y_precast = model(x)
    assert torch.equal(y_stock, y_precast)                                # cast-identical operands -> bit-exact under autocast
    with torch.no_grad(), pytest.raises(cast_cache.PrecastMisuse) as ei:
        model(x)                                                          # fp32 input, autocast off -> named refusal, not silent bf16 math
    assert ei.value.event == "precast_misuse"


# ------------------------------------------------------------------ recipe ----------------------------------------------------------------
def test_apply_torch_levels():
    torch.use_deterministic_algorithms(False); torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
    r0 = recipe.apply_torch(0)
    assert r0["det"] == 0 and r0["changed"] == [] and not torch.are_deterministic_algorithms_enabled()
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    r1 = recipe.apply_torch(1, environ=env)
    assert r1["det"] == 1 and torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert set(r1["changed"]) >= {"deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic"} and r1["cublas_workspace"] == ":4096:8"
    r1w = recipe.apply_torch(1, warn_only=True, cudnn=False, environ=env)
    assert r1w["warn_only"] is True
    torch.use_deterministic_algorithms(False)
    rr = recipe.torch_recipe(1, switches={"KIT_DETERMINISTIC": "1"})
    assert recipe.apply_torch(rr, environ=env)["det"] == 1 and torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    env2 = {}
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        full = recipe.apply(rr, environ=env2)
        assert env2 == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "KIT_DETERMINISTIC": "1"} and full["env"] == ["CUBLAS_WORKSPACE_CONFIG", "KIT_DETERMINISTIC"]
        assert full["det"] == 1 and torch.are_deterministic_algorithms_enabled()
    assert recipe.apply(recipe.torch_recipe(0), environ={})["det"] == 0


@pytest.mark.skipif(DEV != "cuda", reason="order check needs an initialised CUDA context")
def test_apply_torch_order_check_is_named():
    torch.zeros(1, device="cuda")                                         # CUDA initialised
    with pytest.raises(recipe.RecipeOrderError) as ei:
        recipe.apply_torch(1, environ={})
    assert ei.value.event == "det_recipe_order"
    assert recipe.apply_torch(1, environ={}, check_order=False)["det"] == 1


# ------------------------------------------------------------ hardening cases -------------------------------------------------------------
def test_index_out_of_range_and_empty_index_and_lead_shape():
    src = torch.randn(4, 2, device=DEV)
    with pytest.raises(det_scatter.LayoutUnsupported) as ei:
        det_scatter.segment_reduce_rows(src, torch.tensor([0, 1, 2, 5], device=DEV), 3, "sum")
    assert ei.value.record()["layout"] == "index_out_of_range"
    with pytest.raises(det_scatter.LayoutUnsupported):
        det_scatter.segment_reduce_rows(src, torch.tensor([-1, 0, 1, 2], device=DEV), 3, "sum")
    empty = det_scatter.segment_reduce_rows(torch.zeros(2, 0, 5, device=DEV), torch.zeros(0, dtype=torch.long, device=DEV), 3, "mean")
    assert empty.shape == (2, 3, 5) and float(empty.abs().sum()) == 0.0
    idx = torch.tensor([0, 0, 1, 2], device=DEV)
    out = det_scatter.scatter_mean_into(torch.zeros(6, 3, 2, device=DEV), -2, idx, src)          # source [N, C] into zeros [B, I, C]
    assert out.shape == (6, 3, 2) and torch.equal(out[0], out[5]) and torch.allclose(out[0, 0], src[:2].mean(0))
    calls = []
    fb = lambda z, d, i, s: calls.append("fb") or z
    det_scatter.scatter_mean_into(torch.zeros(6, 3, 2, device=DEV), -2, idx, src[None].expand(5, -1, -1), fallback=fb)   # lead 5 vs 6
    assert calls == ["fb"] and det_scatter.census()["fallback_reasons"] == {"lead_shape": 1}
    # dim_size=None and a 2-D index: facts cached once per index tensor
    idx2 = idx[None, :].expand(3, -1)
    a = det_scatter.scatter_reduce(src[None].expand(3, -1, -1), idx2, dim=-2, reduce="sum")
    b = det_scatter.scatter_reduce(src[None].expand(3, -1, -1), idx2, dim=-2, reduce="sum")
    assert a.shape == (3, 3, 2) and torch.equal(a, b)


def test_precast_tied_weights_and_transform_sites():
    a = torch.nn.Linear(4, 4, bias=False).to(DEV); b = torch.nn.Linear(4, 4, bias=False).to(DEV)
    b.weight = a.weight                                                    # tied
    root = torch.nn.Sequential(a, b)
    rec = cast_cache.precast_linear_(root, "bf16", guard=False)
    assert rec["params"] == 1 and rec["modules"] == 2 and root[0].weight is root[1].weight and root[0].weight.dtype == torch.bfloat16
    lin = torch.nn.Linear(3, 5).to(DEV); cc = cast_cache.CastCache("sites")
    t1 = cc.get(lin.weight, "bf16", transform=lambda t: t.t().contiguous())
    t2 = cc.get(lin.weight, "bf16", transform=lambda t: t * 2)             # a different transform site -> a different entry
    assert t1.shape == (3, 5) and t2.shape == (5, 3) and cc.census()["entries"] == 2


def test_snapshot_restore_and_reader_extension_point():
    snap = policy.snapshot()
    torch.set_float32_matmul_precision("high" if snap["matmul"] == "highest" else "highest")
    torch.backends.cudnn.benchmark = not snap["cudnn_benchmark"]
    torch.use_deterministic_algorithms(not snap["deterministic_algorithms"], warn_only=True)
    rec = policy.restore(snap)
    assert set(rec["restored"]) >= {"matmul", "cudnn_benchmark", "deterministic_algorithms"} and rec["unrestorable"] == []
    assert policy.snapshot() == snap
    fake = lambda module=None: (("matmul_precision", "default"), ("x64", False))       # the shape a JAX reader will have
    sig = policy.numerics_signature(reader=fake, extra=(("backend", "xla"),))
    assert sig == (("matmul_precision", "default"), ("x64", False), ("backend", "xla"))
    g = policy.NumericsGuard(reader=fake)
    assert g.check() is None and g.snapshot() == {"matmul_precision": "default", "x64": False}
    assert "torch" in policy.READERS


def test_restore_default_record_and_coupled_fields():                     # CR-P2 amended (a) test (1)
    torch.set_float32_matmul_precision("highest")
    snap = policy.snapshot()
    assert "process_state" not in snap
    torch.set_float32_matmul_precision("high")
    rec = policy.restore(snap)
    assert rec["restored"] == ["matmul", "matmul_tf32"] and rec["unrestorable"] == [] and set(rec) == {"restored", "unrestorable"}
    assert torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32


def test_restore_opt_in_process_state_and_autocast_cache():               # CR-P2 amended (a) test (2) + (b)
    torch.set_float32_matmul_precision("highest")
    with torch.set_grad_enabled(True):
        snap = policy.snapshot(process_state=("grad_enabled",))
        assert snap["process_state"] == {"grad_enabled": True}
        torch.set_grad_enabled(False); torch.set_float32_matmul_precision("high")
        rec = policy.restore(snap, process_state=("grad_enabled",), clear_autocast_cache=True)
        assert torch.is_grad_enabled() and rec["process_state"] == {"restored": ["grad_enabled"]} and rec["autocast_cache"] == "cleared"
        assert rec["restored"] == ["matmul", "matmul_tf32"]
    with pytest.raises(ValueError):
        policy.restore(policy.snapshot(), process_state=("grad_enabled",))          # snapshot without the opt-in cannot restore it
    with pytest.raises(ValueError):
        policy.snapshot(process_state=("no_such_field",))
    assert policy.PROCESS_STATE_FIELDS == ("grad_enabled",) and "grad_enabled" not in policy.SIGNATURE_FIELDS
