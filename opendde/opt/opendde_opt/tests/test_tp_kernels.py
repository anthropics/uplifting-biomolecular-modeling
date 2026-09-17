"""The row-sharded line's triangle-attention kernel lever (``tp_kernels``: tp_triatt = the core's row-block dispatch with the word
``flash_triattn``, the engine's torch statement kept as its fallback): the word is a constant of the line (no kit switch); on CPU tensors the
dispatch refuses the flash kernel BY NAME unless the core's opt-out ``ROWPAIR_TRIATT_CORE=torch`` is set, under which it runs the engine's
torch statement (equal to the module's own call byte for byte, the fallback counted by reason); the census line parses; the tri-mul seam
stays the engine's torch statements (the core's fused provider does not serve this trunk's pair width). Needs torch + the stock opendde
package + opt_core with the row-block kernels (skipped by name otherwise)."""
import os

import pytest

torch = pytest.importorskip("torch")
RA = pytest.importorskip("opt_core.mem.rowpair.triatt")

from opendde_opt import modes, registry, report, tp_kernels as TK  # noqa: E402

ROWS, N, C, H, D = 3, 20, 16, 4, 8


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for k in ("ROWPAIR_TRIATT_QBLOCK", "ROWPAIR_TRIATT_CORE"):
        monkeypatch.delenv(k, raising=False)
    saved = dict(TK.TRIATT)
    TK.TRIATT.update({"kernel": None, "bound": 0, "calls": 0, "door": None, "door_calls": 0, "door_rows": {}, "door_asides": {}})
    yield
    TK.TRIATT.update(saved)


def _layers():
    return pytest.importorskip("opendde.model.triangular.layers", reason="the stock opendde package is the statement the dispatch falls back to")


def _ta(seed=1):
    tri = pytest.importorskip("opendde.model.triangular.triangular", reason="the stock TriangleAttention module")
    try:
        ta = tri.TriangleAttention(c_in=C, c_hidden=D, no_heads=H, starting=True, inf=1e9)
    except (TypeError, AttributeError):                                        # a stand-in engine module of the CPU seam tests, not the stock package
        pytest.skip("opendde.model.triangular.triangular is a test stand-in here: the kernel tests run where the stock wheel is installed")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ta.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return ta.eval()


def _operands(seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, N, C, generator=g)
    mask = (torch.rand(ROWS, N, generator=g) > 0.15).float()
    tb = torch.randn(1, H, N, N, generator=g) * 0.5                          # the whole triangle bias as the driver hands it after the permute: [1, H, N, N]
    return x, mask, tb


def _run(core_word, monkeypatch, ta, x, mask, tb):
    if core_word is None:
        monkeypatch.delenv("ROWPAIR_TRIATT_CORE", raising=False)
    else:
        monkeypatch.setenv("ROWPAIR_TRIATT_CORE", core_word)
    run = TK.attention_dispatch(ta, None)
    mask_bias = (ta.inf * (mask - 1))[..., :, None, None, :]
    with torch.no_grad():
        return run(x.clone(), mask_bias, tb), mask_bias


def test_kernel_word_is_a_constant_of_the_line():
    assert TK.KERNEL_WORDS == {"triatt": "tier:big", "trimul": "fpf_v4"} and TK.KERNEL_WORDS["triatt"].startswith(RA.TIER_PREFIX)   # 0.2.67: the core tier word + the fused row-block TriMul provider word; ONE vocabulary with the core
    ln = modes.LINES["BIG_TP"]
    assert ln.levers[-2:] == ("rowpair_tp", "tp_triatt"), ln.levers
    assert not any(k.endswith(("_KERNEL", "_KERNELS", "TRIATT_CORE")) for k in ln.exports), ln.exports    # no kernel switch on the kit surface
    assert "tp_triatt" not in modes.LINES["BIG_F"].levers and "tp_triatt" not in modes.LINES["LSTAR2A"].levers
    lv = registry.LEVERS["tp_triatt"]
    assert (lv.kit, lv.cls, lv.tier) == (registry.HOUSE, "forward", "tier2") and registry.PIN_STATUS["tp_triatt"][0] == "tested"
    assert registry.ENGAGEMENT["tp_triatt"].multi_gpu and "tp_triatt" in report.CORE_LEVERS and "tp_trimul" not in registry.LEVERS


def test_the_trimul_seam_is_the_engines_torch_statements():
    """Since 0.2.67 ``_trimul_fns`` wraps the module's own torch statements in the core's fused row-block provider (``tp_kernels.trimul_provider``
    -> ``trimul_fused.FusedTriMulFns``; a plain ``TriMulFns`` where the provider cannot be built, counted in ``TRIMUL["refused"]``). What this
    test protects: the STOCK statements the provider falls back to — and runs for every unit under the opt-out ``ROWPAIR_TRIMUL_KERNELS=torch``
    and on CPU tensors — are bitwise the module's sub-layers."""
    from opt_core.mem.rowpair import trimul as RT
    tri = pytest.importorskip("opendde.model.triangular.triangular", reason="the stock TriangleMultiplication module")
    try:
        tm = tri.TriangleMultiplicationOutgoing(c_z=C, c_hidden=C).eval()
    except (TypeError, AttributeError):
        pytest.skip("opendde.model.triangular.triangular is a test stand-in here: the kernel tests run where the stock wheel is installed")
    from opendde_opt import tp
    fns = tp._trimul_fns(tm)
    assert isinstance(fns, RT.TriMulFns) and int(fns.C_h) == C, (type(fns).__name__, TK.TRIMUL)   # FusedTriMulFns IS a TriMulFns built on the stock proj / out / gate callables
    assert type(fns) is RT.TriMulFns or (type(fns).__name__ == "FusedTriMulFns" and TK.TRIMUL["kernels"] == "fpf_v4" and TK.TRIMUL["modules"] >= 1), TK.TRIMUL
    z = torch.randn(4, N, C); m = torch.ones(4, N, 1)
    with torch.no_grad():                                                          # .proj is the module's own statement (the provider's K1 replaces it per unit only on a served CUDA row block)
        assert torch.equal(fns.proj(z, m, True), (m * torch.sigmoid(tm.linear_a_g(tm.layer_norm_in(z))) * tm.linear_a_p(tm.layer_norm_in(z))).to(z.dtype))


def test_the_opt_out_runs_the_engines_statement_through_the_dispatch(monkeypatch):
    L = _layers(); ta = _ta(); x, mask, tb = _operands()
    out, mask_bias = _run("torch", monkeypatch, ta, x, mask, tb)                 # ROWPAIR_TRIATT_CORE=torch: the dispatch's torch statement, counted
    with torch.no_grad():
        ref = ta.mha(q_x=x, kv_x=x, biases=[mask_bias, tb], triangle_attention="torch")   # the module's own call on the row batch
    assert torch.equal(out, ref) and {k: TK.TRIATT[k] for k in ("kernel", "bound", "calls", "door", "door_calls")} == {"kernel": "tier:big", "bound": 1, "calls": 1, "door": None, "door_calls": 0}, TK.TRIATT   # 0.2.67: the line's word is the core tier word; 0.2.69: the core's torch opt-out shuts the kit door too (door None whatever add-on is loaded)
    c = TK.triatt_census()
    assert c["served"] == 0 and c["fallback"] == c["calls"] >= 1 and set(c["fallback_by"]) == {"kernel_torch"} and not c["unexpected"], c
    assert TK.triatt_census_line().startswith("TRIATT kernel=tier:big bound=1 calls=1 served=0 fallback=")
    assert TK.undeclared() == []
    del L


def test_flash_word_on_cpu_tensors_is_refused_by_name(monkeypatch):
    """CPU tensors: the carried kernel cannot run in this process -> refused by name (the lever, the reason, the opt-out
    ``ROWPAIR_TRIATT_CORE=torch`` / ``--mode off``) at bind or at the first call — never a silent torch statement."""
    from opt_core.mem.rowpair import RowpairRefused
    ta = _ta(); x, mask, tb = _operands()
    _run("torch", monkeypatch, ta, x, mask, tb)                                    # under the opt-out it binds and runs on CPU
    with pytest.raises((RowpairRefused, TK.TpKernelRefused)) as ei:
        _run(None, monkeypatch, ta, x, mask, tb)
    assert "torch" in str(ei.value) or "--mode off" in str(ei.value), str(ei.value)


def test_the_rowpair_lever_line_names_the_kernel_word(monkeypatch):
    """The rowpair_tp LEVER line carries the word bound (triatt_kernel=); the kernel lever's own line carries the counts as served= / declined=
    (the arm's `fallback=` audit word stays the core LEVER line's and the ACTIVE line's)."""
    TK.TRIATT.update(kernel="flash_triattn", bound=2)
    from opendde_opt import tp
    d = dict(tp.evidence_pairs())
    assert d["triatt_kernel"] == "flash_triattn" and d["trimul_kernels"] in ("-", "fpf_v4")   # 0.2.67: the fused row-block TriMul provider's word ("-" until a module is bound)
    ev = report.lever_evidence("tp_triatt", {})
    assert ev["kernel"] == "flash_triattn" and "declined" in ev and "fallback" not in ev
