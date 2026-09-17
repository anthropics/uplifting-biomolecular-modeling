"""structok_sync — the structural-token expander's role-pair projections with one host read per call (opendde_opt/structoksync.py): composition,
and the three served methods BITWISE the stock methods (real upstream class bodies on a stand-in expander: n_roles, c_z, pair_block_proj)."""
import sys
import types

import pytest

from opendde_opt import modes, ran, registry, structoksync


def test_composition_on_the_lines():
    assert registry.LEVERS["structok_sync"].tier == "exact"
    for name in ("S1", "LSTAR2A", "BIG_F"):
        assert "structok_sync" in modes.LINES[name].levers, name
    assert "structok_sync" not in modes.LINES["BIG_TP"].levers and "structok_sync" in modes.LEVER_SWITCHES and "structok_sync" in ran.COUNTERS
    out = modes.line_without(modes.LINES["S1"], ("structok_sync",))
    assert "structok_sync" not in out.levers and len(out.levers) == len(modes.LINES["S1"].levers) - 1


def _expander(torch, n_roles=3, c_z=8, device="cpu", dtype=None, seed=0):
    dtype = dtype or torch.float32
    g = torch.Generator().manual_seed(seed)
    ex = types.SimpleNamespace(n_roles=n_roles, c_z=c_z)
    ex.pair_block_proj = torch.nn.ModuleList([torch.nn.Linear(c_z, c_z) for _ in range(n_roles * n_roles)]).to(device=device, dtype=dtype)
    with torch.no_grad():
        for lin in ex.pair_block_proj:
            lin.weight.copy_(torch.randn(c_z, c_z, generator=g).to(device=device, dtype=dtype)); lin.bias.copy_(torch.randn(c_z, generator=g).to(device=device, dtype=dtype))
    return ex


def _stock_methods():
    ST = pytest.importorskip("opendde.model.modules.structural_tokens")
    cls = ST.StructuralTokenExpander
    def stock(name):
        f = getattr(cls, name)
        return getattr(f, "_orig", f)                                                          # the stock body even when this process already patched the class
    return {n: stock(n) for n in structoksync.METHODS}


@pytest.fixture(scope="module", autouse=True)
def _upstream_modules_leave_with_this_module():
    before = {k for k in sys.modules if k.split(".")[0] == "opendde"}
    yield
    for k in [k for k in sys.modules if k.split(".")[0] == "opendde" and k not in before]:
        del sys.modules[k]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("det", [False, True])
def test_the_served_methods_are_bitwise_the_stock_methods(device, det):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    stock = _stock_methods()
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(det)
    try:
        g = torch.Generator().manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16):
            for n_roles, n_struct, batch in ((3, 17, ()), (2, 9, (2,)), (4, 6, ())):        # 4 roles on 6 tokens: some role pairs empty (the dummy term)
                ex = _expander(torch, n_roles=n_roles, device=device, dtype=dtype)
                role = torch.randint(0, n_roles, (n_struct,), generator=g).to(device)
                if n_roles == 4:
                    role = role.clamp(max=1)                                                   # roles 2,3 absent: empty pairs
                z = torch.randn(*batch, n_struct, n_struct, ex.c_z, generator=g).to(device=device, dtype=dtype)
                row_index = torch.arange(3, min(n_struct, 3 + 5), device=device)
                col_index = torch.arange(0, n_struct, 2, device=device)
                structoksync._reset()
                with torch.no_grad():
                    a = stock["_pair_project_by_role_full"](ex, z, role); b = structoksync.FACTORIES["_pair_project_by_role_full"](None)(ex, z, role)
                    assert a.shape == b.shape and torch.equal(a, b), ("full", device, det, dtype, n_roles)
                    zr = z[..., row_index, :, :]
                    a = stock["_pair_project_by_role_full_chunk"](ex, zr, role, row_index); b = structoksync.FACTORIES["_pair_project_by_role_full_chunk"](None)(ex, zr, role, row_index)
                    assert a.shape == b.shape and torch.equal(a, b), ("chunk", device, det, dtype, n_roles)
                    zt = z[..., row_index, :, :][..., :, col_index, :]
                    a = stock["_pair_project_by_role_full_tile"](ex, zt, role, row_index, col_index); b = structoksync.FACTORIES["_pair_project_by_role_full_tile"](None)(ex, zt, role, row_index, col_index)
                    assert a.shape == b.shape and torch.equal(a, b), ("tile", device, det, dtype, n_roles)
                st = structoksync.kit_stats()
                assert st["calls"] == 3 and st["host_reads"] == 3 and st["pairs"] + st["empty"] == 3 * n_roles * n_roles, st
                if n_roles == 4:
                    assert st["empty"] > 0
    finally:
        torch.use_deterministic_algorithms(prev)


def test_install_binds_the_class_methods():
    pytest.importorskip("torch")
    ST = pytest.importorskip("opendde.model.modules.structural_tokens")
    structoksync.install()
    cls = ST.StructuralTokenExpander
    assert all(getattr(getattr(cls, m), "_structok_sync", False) for m in structoksync.METHODS)
    assert structoksync.kit_stats()["installed"] is True and structoksync.kit_stats()["patches"] == 3
