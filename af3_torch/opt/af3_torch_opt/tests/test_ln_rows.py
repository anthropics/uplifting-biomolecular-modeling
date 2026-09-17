"""Lever ln_rows: the row-blocked kernel is the kit's fastnn LayerNorm kernel body statement for statement; forward restates LayerNorm.forward."""
import inspect, os, re, textwrap
import pytest
from af3_torch_opt import ln_rows, stack

LN_FILE = os.path.join("af3t", "af3_torch", "xfold", "fastnn", "layer_norm.py")


def _src():
    return open(os.path.join(stack.forward_dir(), LN_FILE), encoding="utf-8").read()


def _body(src, name):
    """The statement lines of `def name(...)` in src after its signature (dedented; comments, blank lines dropped)."""
    lines = src.splitlines()
    k = next(i for i, l in enumerate(lines) if re.match(r"^[ \t]*def %s\(" % name, l))
    ind = len(lines[k]) - len(lines[k].lstrip())
    j = k
    while not lines[j].split("#")[0].rstrip().endswith(":"):        # the signature may span lines: it ends at the first `...):`
        j += 1
    out = []
    for l in lines[j + 1:]:
        code = l.split("#")[0].rstrip()
        if not code.strip():
            continue
        if len(l) - len(l.lstrip()) <= ind:
            break
        out.append(code)
    return [l.strip() for l in out]


def test_kernel_body_is_the_stock_kernel_body_per_row():
    stock = _body(_src(), "_layer_norm_fwd_fused")
    ours = _body(inspect.getsource(ln_rows._kernel), "_layer_norm_fwd_rows")
    # the stock body after its signature and row base pointers: from `mean = 0` to the store
    s0 = stock.index("mean = 0"); stock_stmts = stock[s0:]
    assert stock_stmts[-1] == "tl.store(Y + cols, y, mask=mask)" and "row = tl.program_id(0).to(tl.int64)" in stock and "Y += row * N" in stock and "X += row * N" in stock, stock
    o0 = ours.index("mean = 0"); our_stmts = ours[o0:]
    restated = [l.replace("Xr + cols", "X + cols").replace("Yr + cols", "Y + cols") for l in our_stmts]
    assert restated == stock_stmts, (restated, stock_stmts)
    head = ours[:o0]
    assert "pid = tl.program_id(0).to(tl.int64)" in head and "for r in range(ROWS):" in head and "row = pid * ROWS + r" in head and "if row < M:" in head and "Xr = X + row * N" in head and "Yr = Y + row * N" in head, head
    # the launch: the stock BLOCK_SIZE / num_warps choice, grid over row blocks
    launch = inspect.getsource(ln_rows.layer_norm_rows); stock_launch = _src()
    for stmt in ("MAX_FUSED_SIZE = 65536 // x.element_size()", "BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))", "num_warps = min(max(BLOCK_SIZE // 256, 1), 8)", "x = x.contiguous()"):
        assert stmt in launch and stmt in stock_launch, stmt


def test_forward_is_the_stock_statements_but_the_launch():
    src = _src()
    for stmt in ('if fastnn_config.layer_norm_implementation == "torch":', "return F.layer_norm(", "input, self.normalized_shape, self.weight, self.bias, self.eps",
                 'elif fastnn_config.layer_norm_implementation == "triton":', "fast_out = LayerNormTritonFunc.apply(", "f\"fastnn_config must be 'torch' or 'triton', got {fastnn_config}\""):
        assert stmt in src and stmt in inspect.getsource(ln_rows.forward), stmt


def test_generic_path_and_counts_on_cpu():
    torch = pytest.importorskip("torch"); pytest.importorskip("triton")
    import sys, types
    kit = os.path.join(stack.forward_dir(), "af3t", "af3_torch"); sys.path.insert(0, kit)
    try:
        from xfold.fastnn import config as fcfg
        from xfold.fastnn.layer_norm import LayerNorm as LN
    finally:
        sys.path.remove(kit)
    keep = fcfg.layer_norm_implementation; fcfg.layer_norm_implementation = "torch"
    try:
        m = LN(16); torch.nn.init.normal_(m.weight); torch.nn.init.normal_(m.bias); x = torch.randn(7, 16)
        ln_rows.take()
        assert torch.equal(LN.forward(m, x), ln_rows.forward(m, x)) and ln_rows.take() == {"calls": 1, "served": 0, "wide": 0, "generic": 1}
        model = torch.nn.Sequential(m)
        r = ln_rows.install(model); assert r["installed"] and not r["already"] and r["modules"] == 1 and r["reason"] is None
        assert LN.forward is ln_rows.forward and LN._stock_forward.__qualname__ == "LayerNorm.forward"
        assert torch.equal(model(x), LN._stock_forward(m, x)) and ln_rows.install(model)["already"]
    finally:
        if getattr(LN, "_stock_forward", None): LN.forward = LN._stock_forward
        for a in ("_ln_rows", "_ln_rows_reason", "_stock_forward"):
            if hasattr(LN, a): delattr(LN, a)
        fcfg.layer_norm_implementation = keep


def test_wide_rows_keep_the_stock_launch():
    assert ln_rows.MAX_ROW_BYTES == 512 and ln_rows.ROWS == 8
