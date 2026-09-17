"""CPU tests of opt_core.kernels.gather_attn (strategy F5.gather_attn): the module imports without Triton / CUDA, the carry is registered and
held, the strategy id is canonical, refusals are by name, the cell table gates untested cells behind the opt-in, and the reference (the
dense-masked formulation every served call is compared to) has the documented SET semantics: -1 padding, duplicates once, per-sample or
batch-shared bias / index sets, LQ != LK, an empty set -> NaN. The kernel itself runs in tests/gpu/test_gather_attn_gpu.py."""
import json
import math
import os

import pytest

torch = pytest.importorskip("torch", reason="needs torch (CPU)")

from opt_core import kernels  # noqa: E402
from opt_core.kernels import gather_attn as GA  # noqa: E402


def _masked_sdpa_fp32(q, k, v, bias, idx, H, gate=None):
    """Independent statement of the op: per (b, i, h) softmax over the listed keys (a Python loop; tiny shapes only)."""
    B, LQ, C = q.shape
    dh = C // H
    out = torch.zeros(B, LQ, C, dtype=torch.float32)
    for b in range(B):
        ib = idx[b if idx.shape[0] == B else 0]
        bb = bias[b if bias.shape[0] == B else 0].float()
        for i in range(LQ):
            keys = sorted({int(j) for j in ib[i].tolist() if int(j) >= 0})
            for h in range(H):
                qq = q[b, i, h * dh:(h + 1) * dh].float()
                if not keys:
                    out[b, i, h * dh:(h + 1) * dh] = float("nan")
                    continue
                kk = k[b, keys][:, h * dh:(h + 1) * dh].float()
                vv = v[b, keys][:, h * dh:(h + 1) * dh].float()
                s = kk @ qq / math.sqrt(dh) + bb[i, keys, h]
                o = torch.softmax(s, 0) @ vv
                if gate is not None:
                    o = o * gate[b, i, h * dh:(h + 1) * dh].float()
                out[b, i, h * dh:(h + 1) * dh] = o
    return out


def _case(B=2, LQ=7, LK=11, H=2, dh=4, k=5, shared_bias=False, shared_idx=False, seed=0, pad=False, dup=False):
    g = torch.Generator().manual_seed(seed)
    C = H * dh
    q = torch.randn(B, LQ, C, generator=g); kk = torch.randn(B, LK, C, generator=g)
    v = torch.randn(B, LK, C, generator=g).to(torch.bfloat16); gate = torch.rand(B, LQ, C, generator=g).to(torch.bfloat16)
    bias = torch.randn(1 if shared_bias else B, LQ, LK, H, generator=g).to(torch.bfloat16)
    ib = 1 if shared_idx else B
    idx = torch.stack([torch.stack([torch.randperm(LK, generator=g)[:k].sort().values for _ in range(LQ)]) for _ in range(ib)]).to(torch.int32)
    if pad:
        idx[:, ::2, 0] = -1                       # a padded slot (sorted rows keep -1 first)
    if dup:
        idx[:, 1::3, 1] = idx[:, 1::3, 2]         # an index listed twice
    return q, kk, v, bias, idx, H, gate


def test_module_imports_without_triton_and_names_its_strategy():
    assert GA.STRATEGY == "F5.gather_attn" and GA.KERNEL_VERSION.startswith("gather_attn/")
    assert isinstance(GA.HAVE_TRITON, bool)
    assert set(GA.__all__) >= {"gather_attn", "supported", "reference", "Refusal", "CELLS", "CONFIGS", "stats"}


def test_the_carry_is_registered_held_and_routable(tmp_path):
    assert "gather_attn" in kernels.names() and kernels.verify_carry("gather_attn") == []
    doc = kernels.sums("gather_attn")
    assert doc["kind"] == "module" and set(doc["files"]) == {"gather_attn.py"} and doc["exports"] == {} and doc["runtime_imports"] == []
    assert "F5.gather_attn" in doc["mechanism"]
    where = kernels.route("gather_attn")
    try:
        assert where == kernels.carried_path("gather_attn") and kernels.route_check("gather_attn").ok
    finally:
        kernels.unroute("gather_attn")


def test_strategy_id_is_canonical():
    with open(os.path.join(os.path.dirname(kernels.KERNELS_DIR), "STRATEGIES.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = [r for r in doc["canonical"] if r["id"] == "F5.gather_attn"]
    assert rows == [{"family": "F5", "id": "F5.gather_attn", "numerics_changing": True}]


def test_cells_and_configs_are_the_documented_tables():
    assert GA.CELLS == {(32, 128, "bfloat16"): "certified", (48, 32, "bfloat16"): "certified"}
    assert GA.cell_status(32, 128, torch.bfloat16) == "certified" and GA.cell_status(32, 128, torch.float16) == "candidate"
    assert GA.cell_status(48, 32, torch.bfloat16) == "certified" and GA.cell_status(64, 128, torch.bfloat16) == "candidate"
    assert set(GA.CONFIGS) == {32, 64, 128} and all(len(v) == 3 for v in GA.CONFIGS.values())


def test_capability_rows_resolve_through_safe_settings():
    """ROWS_BY_CC: "<cc>|*" rows (cc 8.0) resolved by the core's one mechanism; a capability without a row (9.0) and a CPU device are served CELLS /
    CONFIGS; cell_status / launch_config / cells_key read the row of the device they are asked about."""
    from opt_core.kernels import safe_settings as S
    assert "8.0|*" in GA.ROWS_BY_CC and not any(k.startswith("9.0") for k in GA.ROWS_BY_CC)
    assert S.resolve_row(GA.ROWS_BY_CC, (8, 0), "3.7") is GA.ROWS_BY_CC["8.0|*"] and S.resolve_row(GA.ROWS_BY_CC, (9, 0), "3.7") is None
    for row in GA.ROWS_BY_CC.values():
        assert set(row) == {"cells", "configs"} and set(row["configs"]) <= set(GA.CONFIGS) and all(len(v) == 3 for v in row["configs"].values())
        assert all(status in ("certified", "candidate") for status in row["cells"].values())
    cpu = torch.device("cpu")                                                     # no capability: the capability-free tables
    assert GA.cells_key(cpu) is None and GA.launch_config(32, cpu) == GA.CONFIGS[32] and GA.launch_config(48, cpu) == GA.CONFIGS[64]
    assert GA.cell_status(32, 128, torch.bfloat16, cpu) == "certified" and GA.cell_status(64, 128, torch.bfloat16, cpu) == "candidate"
    if not torch.cuda.is_available():                                             # a CPU interpreter: the current-device form answers the same
        assert GA.cells_key() is None and GA.launch_config(32) == GA.CONFIGS[32] and GA.cell_status(48, 32, torch.bfloat16) == "certified"


def test_candidate_opt_in(monkeypatch):
    monkeypatch.delenv(GA.ALLOW_CANDIDATE_ENV, raising=False)
    assert GA.candidate_allowed(None) is False and GA.candidate_allowed(True) is True
    monkeypatch.setenv(GA.ALLOW_CANDIDATE_ENV, "1")
    assert GA.candidate_allowed(None) is True and GA.candidate_allowed(False) is False


def test_refusals_are_by_name_on_cpu():
    q, k, v, bias, idx, H, gate = _case()
    why = GA.supported(q, k, v, bias, idx, H, gate=gate)
    assert why == ("not-cuda" if GA.HAVE_TRITON else "no-triton")
    with pytest.raises(GA.Refusal) as e:
        GA.gather_attn(q, k, v, bias, idx, H, gate=gate)
    assert e.value.reason == why and str(e.value).startswith("gather_attn refused: " + why)
    assert issubclass(GA.Refusal, ValueError)


@pytest.mark.skipif(not GA.HAVE_TRITON, reason="the refusal ladder below no-triton needs Triton importable")
def test_refusal_ladder_names(monkeypatch):
    class _T:                                          # a CPU stand-in that answers is_cuda True so the ladder below 'not-cuda' is reachable
        def __init__(self, t): self.t = t
        def __getattr__(self, n): return getattr(self.t, n)
        is_cuda = True
    q, k, v, bias, idx, H, gate = _case(dh=4, k=5)
    W = lambda *ts: [_T(t) for t in ts]
    assert GA.supported(*W(q, k, v.float(), bias, idx), H) == "v-dtype-float32"
    assert GA.supported(*W(q.double(), k.double(), v, bias, idx), H) == "qk-dtype-float64"
    assert GA.supported(*W(q, k, v, bias, idx.long()), H) == "idx-not-int32"
    assert GA.supported(*W(q, k, v, bias, idx), 3) == "layout"
    assert GA.supported(*W(q, k, v, bias.permute(0, 3, 1, 2), idx), H) == "bias-layout"
    assert GA.supported(*W(q, k, v, bias[:, :3], idx), H) == "bias-shape"
    assert GA.supported(*W(q, k, v, bias, idx[:, :3]), H) == "idx-shape"
    assert GA.supported(*W(q, k, v, bias, idx), H, gate=_T(gate[:, :3])) == "gate-shape"
    assert GA.supported(*W(q, k, v, bias, idx), H) == "cell-uncertified:dh4/k5/bfloat16+off(not-measured)"     # a cell no engine measured: the dense path BY NAME
    assert GA.supported(*W(q, k, v, bias, idx), H, allow_candidate=True) is None
    qg = q.clone().requires_grad_(True)
    assert GA.supported(*W(qg, k, v, bias, idx), H, allow_candidate=True) == "grad"
    big = torch.randn(1, 2, 2 * 129); kb = torch.randn(1, 3, 2 * 129)
    assert GA.supported(*W(big, kb, kb.to(torch.bfloat16), torch.zeros(1, 2, 3, 2), torch.zeros(1, 2, 1, dtype=torch.int32)), 2, allow_candidate=True) == "head-dim-129"


@pytest.mark.parametrize("kw", [dict(), dict(shared_bias=True), dict(shared_idx=True), dict(shared_bias=True, shared_idx=True, B=1), dict(pad=True), dict(dup=True),
                                dict(pad=True, dup=True, LQ=5, LK=9, k=9), dict(B=3, LQ=4, LK=4, k=1)])
def test_reference_is_the_set_semantics_op(kw):
    q, k, v, bias, idx, H, gate = _case(**kw)
    ref = GA.reference(q, k, v, bias, idx, H, gate=gate)
    want = _masked_sdpa_fp32(q, k, v, bias, idx, H, gate=gate)
    assert ref.shape == want.shape == q.shape and ref.dtype == torch.float32
    assert torch.allclose(ref, want, atol=1e-5, rtol=1e-5), (ref - want).abs().max()


def test_reference_empty_set_is_nan_like_the_dense_formulation():
    q, k, v, bias, idx, H, gate = _case(B=1, LQ=3, LK=6, k=2)
    idx[0, 1, :] = -1                                                    # query 1 lists nothing
    ref = GA.reference(q, k, v, bias, idx, H)
    assert torch.isnan(ref[0, 1]).all() and not torch.isnan(ref[0, 0]).any() and not torch.isnan(ref[0, 2]).any()


def test_reference_round_qk_is_the_kernels_rounding_point():
    q, k, v, bias, idx, H, gate = _case()
    a = GA.reference(q, k, v, bias, idx, H, round_qk=True)
    b = GA.reference(q.to(torch.bfloat16).float(), k.to(torch.bfloat16).float(), v, bias, idx, H)
    assert torch.equal(a, b)


def test_stats_shape():
    s = GA.stats()
    assert set(s) == {"calls", "served_certified", "served_candidate", "rows_sorted_here", "max_transient_bytes"}


def test_a_capability_without_a_row_is_served_the_defaults_named_once(monkeypatch, capfd):
    """P12: a CUDA device whose capability has no ROWS_BY_CC row (and is not cc 9.0, whose measurements CELLS / CONFIGS are) is served CELLS / CONFIGS with
    ONE info line and cells_note() == 'default:no_row'; cc 9.0 and CPU say nothing."""
    from opt_core.kernels import safe_settings as S
    GA._NET.reset(); GA._ROWS_RESOLVED.clear()
    monkeypatch.setattr(S, "row_for_device", lambda table, device=None, default=None, cache=None: None)
    monkeypatch.setattr(S, "device_cc", lambda device=None: (12, 0))
    assert GA._row("cuda:0") is None and GA.cells_note("cuda:0") == "default:no_row"
    err = capfd.readouterr().err
    assert err.startswith("[opt_core/gather_attn] default settings (no row for cc 12.0, cc 12.0, triton ") and "tuned rows exist for cc 8.0, 9.0" in err and err.count("\n") == 1
    GA._row("cuda:0")
    assert capfd.readouterr().err == ""                                              # once per process
    GA._NET.reset()
    monkeypatch.setattr(S, "device_cc", lambda device=None: (9, 0))
    assert GA.cells_note("cuda:0") is None and capfd.readouterr().err == ""           # cc 9.0: the defaults ARE its row
    monkeypatch.setattr(S, "device_cc", lambda device=None: None)
    assert GA.cells_note("cpu") is None and capfd.readouterr().err == ""
    GA._NET.reset()

