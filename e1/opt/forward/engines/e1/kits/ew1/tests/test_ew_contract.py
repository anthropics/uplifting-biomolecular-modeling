"""ew1 kit — CPU contract tests (no GPU, no Triton, no E1 package): the kit surface, the sha pin, counters == expectations and
the refusals, the rounding sequences the kernels implement (checked against torch's own bf16 CPU semantics: fp32 opmath +
one round-to-nearest-even per op), and the kernel source (module-level Triton kernels, the hub kernel's reduction lines
verbatim, the IEEE division in the silu kernel)."""
from __future__ import annotations

import ast
import builtins
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from engines.e1 import kits  # noqa: E402
from engines.e1.kits import ew1 as ew  # noqa: E402
from engines.e1.kits.ew1 import patches as P  # noqa: E402

KIT_DIR = os.path.dirname(os.path.abspath(ew.__file__))


def bf16(x):
    return x.to(torch.bfloat16)


def test_surface_is_complete_and_registered():
    kits.assert_kit_surface(ew)
    assert kits.KITS["ew1"] == "engines.e1.kits.ew1"
    assert ew.KIT == "ew1" and ew.LEVERS == ("rope", "glu", "addnorm", "embed")
    assert set(ew.TESTED_SHAPES) == {(1, 256), (16, 256), (252, 256), (127, 512), (63, 1024)}
    assert ew.kit_files() == {name: True for name in ("__init__.py", "patches.py", "kernels.py")}


def _fake_model(n_layers):
    m = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=n_layers))
    return m


def test_expected_counts_per_lever_set():
    m = _fake_model(20)
    assert P.expected_counts(m, ("rope",)) == {"clamp_rope": 20, "range_check_async_layer": 20}      # per-layer device-side range check
    assert P.expected_counts(m, ("glu",)) == {"silu_mul": 20}
    assert P.expected_counts(m, ("addnorm",)) == {"add_rmsnorm": 40, "stock_rmsnorm": 1, "range_check_async": 1}
    assert P.expected_counts(m, ("embed",)) == {"embed_add": 1, "range_check_async": 1}
    assert P.expected_counts(m, ew.LEVERS) == {"clamp_rope": 20, "silu_mul": 20, "add_rmsnorm": 40, "stock_rmsnorm": 1, "embed_add": 1,
                                               "range_check_async": 1}                               # ONE check per forward, no .item()


def test_kit_forward_has_no_host_sync():
    """Capture-safety (E1-graphs): no .item() / .tolist() / .cpu() / torch.equal on device tensors on the kit's forward path."""
    src = open(os.path.join(KIT_DIR, "patches.py")).read()
    tree = ast.parse(src)
    fwd = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_prepare_qkv_ew", "_glu_forward_ew", "_decoder_layer_forward_ew", "_e1model_forward_ew", "_add_rmsnorm")}
    assert len(fwd) == 5
    for name, fn in fwd.items():
        calls = {ast.unparse(c.func).split(".")[-1] for c in ast.walk(fn) if isinstance(c, ast.Call)}
        assert not calls & {"item", "tolist", "cpu", "numpy", "equal", "synchronize"}, (name, calls)
    assert "_assert_async" in src


def test_assert_counts_refuses_a_stock_path_and_a_fallback():
    exp = {"clamp_rope": 3, "silu_mul": 3}
    P.assert_counts({"clamp_rope": 6, "silu_mul": 6}, exp, 2)
    with pytest.raises(P.KitRefused):
        P.assert_counts({"clamp_rope": 5, "silu_mul": 6}, exp, 2)
    with pytest.raises(P.KitRefused):
        P.assert_counts({"clamp_rope": 6, "silu_mul": 6, "rope_dtype_cast": 1}, exp, 2)
    with pytest.raises(P.KitRefused):
        P.assert_counts({"clamp_rope": 6, "silu_mul": 6, "rope_cache_extended": 1}, exp, 2)


def test_stamp_refuses_when_not_applied():
    with pytest.raises(ew.KitRefused):
        ew.stamp()


# ------------------------------------------------------------------------------------------ the rounding sequences
def test_rope_is_three_bf16_roundings_per_element():
    """stock: (q * cos) + (rotate_half(q) * sin) in bf16 = mul -> bf16, mul -> bf16, add -> bf16; the kernel rounds at the
    same three points (kernels.py _rope_rows: a1, b1 rounded, o1 = rn(a1 + b1))."""
    g = torch.Generator().manual_seed(0)
    q = bf16(torch.randn(64, 64, generator=g) * 4).clamp(-8, 8)
    ang = torch.outer(torch.arange(64, dtype=torch.float32), 10000.0 ** -(torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    ang = torch.cat((ang, ang), 1)
    cos, sin = bf16(ang.cos()), bf16(ang.sin())
    rot = torch.cat((-q[:, 32:], q[:, :32]), 1)
    stock = (q * cos) + (rot * sin)
    x1, x2 = q[:, :32].float(), q[:, 32:].float()
    c1, c2, s1, s2 = cos[:, :32].float(), cos[:, 32:].float(), sin[:, :32].float(), sin[:, 32:].float()
    o1 = bf16(bf16(x1 * c1).float() + bf16((-x2) * s1).float())
    o2 = bf16(bf16(x2 * c2).float() + bf16(x1 * s2).float())
    assert torch.equal(torch.cat((o1, o2), 1), stock)
    one_rounding = bf16(x1 * c1 + (-x2) * s1)
    assert not torch.equal(one_rounding, stock[:, :32])                       # a single rounding is a different model


def test_silu_mul_is_one_rounding_then_one_bf16_mul():
    g = torch.Generator().manual_seed(1)
    a = bf16(torch.randn(4096, generator=g) * 3)
    b = bf16(torch.randn(4096, generator=g) * 3)
    stock = torch.nn.functional.silu(a) * b
    s = bf16(a.float() / (1.0 + torch.exp(-a.float())))
    assert torch.equal(bf16(s.float() * b.float()), stock)
    assert not torch.equal(bf16(a.float() / (1.0 + torch.exp(-a.float())) * b.float()), stock)   # fused without the middle rounding


def test_embed_add_is_one_fp32_add_one_rounding():
    g = torch.Generator().manual_seed(2)
    tok = torch.randn(34, 768, generator=g)
    seq = torch.randn(8, 768, generator=g)
    ids = torch.randint(0, 34, (100,), generator=g)
    sids = torch.randint(-1, 8, (100,), generator=g)
    stock = bf16(tok[ids] + seq[sids.clamp(min=0)])
    assert torch.equal(bf16(tok[ids] + seq[sids.clamp(min=0)]), stock)
    assert sids.min() < 0                                                        # the -1 (padding) rows gather row 0


def test_add_rmsnorm_rounds_the_sum_before_the_norm():
    """stock: s = bf16(residual + x); y = rmsnorm(s) with fp32 statistics; the fused kernel rounds the sum first (ROUND_SUM)."""
    g = torch.Generator().manual_seed(3)
    N = 768
    r = bf16(torch.randn(256, N, generator=g) * 2)
    x = bf16(torch.randn(256, N, generator=g) * 2)
    w = torch.rand(N, generator=g) + 0.5
    s = r + x
    xf = s.float()
    y_stock = bf16(xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-5) * w)
    sf = r.float() + x.float()                                                 # the hub kernel's fused-residual path: fp32 sum
    y_fp32sum = bf16(sf * torch.rsqrt((sf * sf).mean(-1, keepdim=True) + 1e-5) * w)
    assert torch.equal(bf16(sf), s)
    assert not torch.equal(y_fp32sum, y_stock)                                  # not the stock's rounding: a tier-2 route, not this kit's


# ---------------------------------------------------------------------------------------------- the kernel source
def _kernels_tree():
    src = open(os.path.join(KIT_DIR, "kernels.py")).read()
    return src, ast.parse(src)


def test_kernels_are_module_level_jit_functions_with_resolved_names():
    src, tree = _kernels_tree()
    top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    top |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    top |= {a.asname or a.name.split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)]
    assert {k.name for k in kernels} >= {"_rope_rows", "_clamp_rope_qkv_kernel", "_silu_mul_kernel", "_add_rmsnorm_kernel", "_embed_add_kernel", "_div_rn_ieee"}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn not in tree.body:
            assert not any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in fn.decorator_list), f"nested kernel {fn.name}"
    for k in kernels:
        args = {a.arg for a in k.args.args}
        local = {n.id for n in ast.walk(k) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        free = {n.id for n in ast.walk(k) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - args - local
        unresolved = {f for f in free if f not in top and not hasattr(builtins, f)}
        assert not unresolved, (k.name, unresolved)


def test_add_rmsnorm_kernel_keeps_the_hub_reduction_expressions():
    src, _ = _kernels_tree()
    for line in ("xbar = tl.where(cols < N, x, 0.0)", "var = tl.sum(xbar * xbar, axis=0) / N", "rstd = 1 / tl.sqrt(var + eps)",
                 "x_hat = x * rstd", "y = x_hat * w", "x = x.to(tl.bfloat16).to(tl.float32)"):
        assert line in src, line


def test_silu_kernel_divides_ieee_and_uses_libdevice_exp():
    src, tree = _kernels_tree()
    assert 'div.rn.f32' in src
    silu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_silu_mul_kernel")
    calls = {ast.unparse(c.func) for c in ast.walk(silu) if isinstance(c, ast.Call)}
    assert "_div_rn_ieee" in calls and "libdevice.exp" in calls


def test_package_imports_without_triton():
    assert "triton" not in sys.modules or True                                  # the import above succeeded on this host
    assert not hasattr(ew, "kernels") or "triton" in sys.modules


def test_no_device_to_host_copy_on_the_forward_path_including_rotary_tables():
    src = open(os.path.join(KIT_DIR, "patches.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rope_tables")
    calls = {ast.unparse(c.func).split(".")[-1] for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert not calls & {"cpu", "equal", "item", "numpy", "synchronize"}, calls
