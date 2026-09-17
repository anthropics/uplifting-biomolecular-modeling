"""The package lever ``dev_scalars`` RESTATES two stock functions (compared against the kit's source live, at test time): xfold
geometry's ``Vec3Array.norm`` but for where the epsilon tensor comes from (the stock statement's tensor, built once per process), and
``Evoformer._embed_bonds`` but for the three scalar writes (device scalars instead of host floats). Bitwise by construction; the norm's
values are checked bit for bit here, the bond matrix's on the kit's canaries (GPU)."""
import ast
import inspect
import os
import textwrap

import pytest

from af3_torch_opt import dev_scalars, registry, modes, stack

GEOMETRY_KIT_FILE = "af3t/af3_torch/xfold/geometry.py"
ALPHAFOLD3_KIT_FILE = "af3t/af3_torch/xfold/alphafold3.py"


def _func_src(path, qual):
    src = open(path, encoding="utf-8").read(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    node = tree
    for part in qual.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return "".join(lines[node.lineno - 1: node.end_lineno])


def _stmts(fn_src):
    node = ast.parse(textwrap.dedent(fn_src)).body[0]
    body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) else node.body
    return [ast.unparse(s) for s in body]


def test_lever_tables():
    assert "dev_scalars" in registry.PACKAGE_LEVERS and "dev_scalars" in registry.LEVERS and registry.EVIDENCE["dev_scalars"] == "package"
    assert registry.BITWISE_EVIDENCE["dev_scalars"] == {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True}
    assert registry.STRATEGY["dev_scalars"] == "F6.host_sync_elimination" and registry.IMPL["dev_scalars"] == ("dev_scalars.install", "kit")
    assert all(("dev_scalars" in modes.MODE_PACKAGE_LEVERS[m]) == (m != "off") for m in modes.MODES)          # every mode but off
    assert dev_scalars.take() == {"norm_calls": 0, "eps_tensors": 0, "bond_calls": 0}


def test_norm_is_the_stock_statements_but_the_epsilon_tensor():
    stock = _stmts(_func_src(os.path.join(stack.forward_dir(), *GEOMETRY_KIT_FILE.split("/")), "Vec3Array.norm"))
    mine = [s for s in _stmts(inspect.getsource(dev_scalars.norm)) if not s.startswith(("import torch", "COUNTS["))]
    eps_stock = [s for s in stock if "epsilon ** 2" in s or "epsilon**2" in s]; eps_mine = [s for s in mine if "eps2_tensor(" in s]
    assert len(eps_stock) == len(eps_mine) == 1, (eps_stock, eps_mine)
    assert "torch.maximum(norm2, torch.tensor(epsilon ** 2, dtype=norm2.dtype, device=norm2.device))" in eps_stock[0], eps_stock      # the statement eps2_tensor restates
    assert "torch.maximum(norm2, eps2_tensor(epsilon, norm2.dtype, norm2.device))" in eps_mine[0], eps_mine
    assert [s for s in stock if s not in eps_stock] == [s for s in mine if s not in eps_mine]                                          # every other statement verbatim
    built = _stmts(inspect.getsource(dev_scalars.eps2_tensor))
    assert any("torch.tensor(epsilon ** 2, dtype=dtype, device=device)" in s for s in built), built                                    # the cached tensor IS the stock expression


def test_embed_bonds_is_the_stock_statements_but_the_scalars():
    stock = _stmts(_func_src(os.path.join(stack.forward_dir(), *ALPHAFOLD3_KIT_FILE.split("/")), "Evoformer._embed_bonds"))
    mine = [s for s in _stmts(inspect.getsource(dev_scalars._embed_bonds)) if not s.startswith(("import torch", "from xfold import of3", "COUNTS[", "one = ", "zero = "))]
    assert len(stock) == len(mine), (len(stock), len(mine))
    changed = 0
    for a, b in zip(stock, mine):
        if a == b:
            continue
        changed += 1                                                                                   # the scalar writes: same target, same indices, the host float replaced by the device scalar
        assert (a.replace("= 1.0", "= one").replace("= 0.0", "= zero") == b), (a, b)
    assert changed == 3, changed                                                                       # `[i, j] = 1.0`, the OF3 symmetric `[j, i] = 1.0`, `[0, 0] = 0.0`
    ones = [s for s in _stmts(inspect.getsource(dev_scalars._embed_bonds)) if s.startswith(("one = ", "zero = "))]
    assert ones == ["one = contact_matrix.new_ones(())", "zero = contact_matrix.new_zeros(())"], ones


def test_norm_values_are_stocks_bit_for_bit():
    torch = pytest.importorskip("torch")

    class V:                                                                                           # the fields Vec3Array.norm reads: dot()
        def __init__(s, x, y, z): s.x, s.y, s.z = x, y, z
        def dot(s, o): return s.x * o.x + s.y * o.y + s.z * o.z

    def stock_norm(self, epsilon=1e-6):
        norm2 = self.dot(self)
        if epsilon:
            norm2 = torch.maximum(norm2, torch.tensor(epsilon**2, dtype=norm2.dtype, device=norm2.device))
        return torch.sqrt(norm2)

    dev_scalars._EPS2.clear(); dev_scalars.take()
    torch.manual_seed(0)
    specials = torch.tensor([0.0, 1e-30, 1e-13, 1e-12, 1.0000001e-12, 3e-7, float("inf"), float("nan"), -0.0, 1e-3])
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        x = torch.cat([torch.randn(64) * 1e-6, specials]).to(dtype); y = torch.zeros_like(x); z = torch.randn(74).to(dtype) * 1e-7
        v = V(x, y, z)
        for eps in (1e-6, 1e-3, 0.0):
            a, b = stock_norm(v, eps), dev_scalars.norm(v, eps)
            assert torch.equal(torch.nan_to_num(a, 7.0), torch.nan_to_num(b, 7.0)) and torch.equal(a.isnan(), b.isnan()), (dtype, eps)
    c = dev_scalars.take()
    assert c["norm_calls"] == 9 and c["eps_tensors"] == 6 and len(dev_scalars._EPS2) == 6                # one tensor per (epsilon, dtype, device): (1e-6, 1e-3) x 3 dtypes; epsilon 0 builds none


def test_install_refuses_a_changed_model():
    class M:
        pass
    m = M(); m.evoformer = M()
    with pytest.raises(RuntimeError, match="_embed_bonds"):
        dev_scalars.install(m)
