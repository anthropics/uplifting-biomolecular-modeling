"""The package lever ``tri_layout`` RESTATES ``TriangleMultiplication.forward`` (compared against the kit's source live): the same
statements but the three layout copies — the mask / gate products taken where the Linear wrote the projection, the einsum operands and
the product's channels-last layout written by the tiled copy kernel. Data movement only; the module's values are proved bit for bit on
the GPU (the kit's canaries under ``exact`` vs ``off``), here the restatement and the generic path are held to the kit's statements."""
import ast
import inspect
import os
import textwrap

import pytest

from af3_torch_opt import modes, registry, stack, tri_layout

KIT_FILE = "af3t/af3_torch/xfold/nn/triangle_multiplication.py"
ADAPTER_FILE = "af3t/kernels/af3_kernels.py"


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
    assert "tri_layout" in registry.PACKAGE_LEVERS and "tri_layout" in registry.LEVERS and registry.EVIDENCE["tri_layout"] == "package"
    assert registry.BITWISE_EVIDENCE["tri_layout"] == {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True}
    assert registry.STRATEGY["tri_layout"] == "F2.trimul_tperm_exact" and registry.IMPL["tri_layout"] == ("tri_layout.forward", "kit")
    assert all(("tri_layout" in modes.MODE_PACKAGE_LEVERS[m]) == (m != "off") for m in modes.MODES)
    assert tri_layout.take() == {"calls": 0, "tiled": 0, "generic": 0}


def test_forward_is_the_stock_statements_but_the_layout_copies():
    stock = _stmts(_func_src(os.path.join(stack.forward_dir(), *KIT_FILE.split("/")), "TriangleMultiplication.forward"))
    tail = [s for s in _stmts(inspect.getsource(tri_layout._stock_tail)) if not s.startswith("import torch")]
    mine = [s for s in _stmts(inspect.getsource(tri_layout.forward)) if not s.startswith(("import torch", "COUNTS[", "tiled = ", "if not tiled:"))]
    assert stock[:3] == ["pair = self.left_norm_input(pair)", "input_pair = pair", "projection = self.projection(pair)"], stock[:3]
    assert tail == stock[3:]                                                                    # the generic path: stock's statements from the channel-major view on, verbatim
    assert mine[:3] == stock[:3], mine[:3]                                                      # LN + projection: verbatim
    assert stock[-5:] == ["pair = self.center_norm(pair)", "pair = self.output_projection(pair)", "gate_out = self.gating_linear(input_pair)",
                          "pair *= torch.sigmoid(gate_out)", "return pair"], stock[-5:]
    assert mine[-5:] == ["pair = self.center_norm(pair)", "pair = self.output_projection(pair)", "gate_out = self.gating_linear(input_pair)",
                         "pair = _gate().mask_gate_(pair, gate_out)", "return pair"], mine[-5:]   # the output gate through gate_fuse.mask_gate_ (stock statement there)
    middle = mine[3:-5]
    assert middle == ["gate = self.gate(pair)",
                      "projection = _gate().mask_gate_(projection, gate, mask)",                # stock: `projection *= mask[None, ...]` + `projection *= torch.sigmoid(gate)` on the (ch, i, j) views
                      "a, b = operands(projection, self.c_pair, self.equation)",                 # stock: reshape(c, 2, N, N) + chunk + squeeze -> the strided halves einsum clones
                      "pair = torch.einsum(self.equation, a, b)",
                      "pair = channels_last(pair)"], middle                                      # stock: pair.permute(1, 2, 0), copied contiguous by center_norm
    stock_mid = stock[3:-5]
    assert "projection = projection.permute(2, 0, 1)" in stock_mid and "projection *= torch.sigmoid(gate)" in stock_mid and "pair = torch.einsum(self.equation, a, b)" in stock_mid and "pair = pair.permute(1, 2, 0)" in stock_mid, stock_mid
    from af3_torch_opt import gate_fuse
    g = _stmts(inspect.getsource(gate_fuse.mask_gate_))
    assert "x *= mask[..., None]" in "\n".join(g) and "x *= torch.sigmoid(gate)" in g, g            # the stock statements, run when the lever is off

def test_scope_follows_the_kernel_levers_fallback_table():
    """Under fast / big the trimul kernel lever owns the class forward (under exact: trimul_exact) and falls back BY NAME to the stock forward it saved in its table;
    tri_layout.install rebinds exactly that entry (scope=fallback). The adapter's statements this relies on, read from its source."""
    src = open(os.path.join(stack.forward_dir(), *ADAPTER_FILE.split("/")), encoding="utf-8").read()
    assert '_ORIG["trimul"] = TriangleMultiplication.forward' in src and '_ORIG["trimul"](self, pair, mask)' in src
    assert "TriangleMultiplication.forward = ((_trimul_forward_exact_aside if \"trimul_exact\" in _ON else _trimul_forward) if \"trimul\" in _ON" in src   # levers trimul / trimul_exact own the class forward; both fall back to _ORIG["trimul"] by name
    import types as _t, sys
    fake = _t.ModuleType("af3_kernels"); fake.__file__ = "/x/kernels/af3_kernels.py"; fake._ORIG = {}
    sys.modules["_tri_layout_fake_adapter"] = fake
    try:
        assert fake in tri_layout._kernel_adapters()
    finally:
        del sys.modules["_tri_layout_fake_adapter"]


def test_generic_path_on_cpu_is_stock_bit_for_bit():
    torch = pytest.importorskip("torch"); pytest.importorskip("triton")
    import sys
    kit = os.path.join(stack.forward_dir(), "af3t", "af3_torch")
    sys.path.insert(0, kit)
    try:
        from xfold.nn.triangle_multiplication import TriangleMultiplication as TM
    finally:
        sys.path.remove(kit)
    torch.manual_seed(0)
    for outgoing in (True, False):
        m = TM(c_pair=16, _outgoing=outgoing)
        for prm in m.parameters(): torch.nn.init.normal_(prm, std=0.3)
        pair = torch.randn(12, 12, 16); mask = (torch.rand(12, 12) > 0.2).float()
        tri_layout.take()
        assert torch.equal(TM.forward(m, pair.clone(), mask), tri_layout.forward(m, pair.clone(), mask))
        assert tri_layout.take() == {"calls": 1, "tiled": 0, "generic": 1}                    # CPU tensors: the stock statements, counted
