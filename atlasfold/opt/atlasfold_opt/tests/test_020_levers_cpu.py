"""ln_bf16 and the det recipe on CPU: ln_bf16's CPU route, the registry's expected words == the module's literals, installers() coverage of
both rows (the DiT attention lever's contracts live in test_dit_apb_cpu.py)."""
import importlib

import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers


def _restore(cls, attr):
    fn = getattr(cls, attr)
    stock = getattr(fn, "__wrapped_stock__", None)
    if stock is not None:
        setattr(cls, attr, stock)


def test_rows_and_installers_cover_the_new_levers():
    inst = installers()
    for lever in ("dit_apb", "ln_bf16"):
        assert lever in registry.LEVERS and lever in inst
    assert "ln_bf16" in modes.MODES[modes.EXACT] and "ln_bf16" in modes.MODES[modes.FAST]
    assert "dit_sdpa" not in registry.LEVERS and "dit_sdpa" not in inst and all("dit_sdpa" not in row for row in modes.MODES.values())
    assert registry.LEVERS["ln_bf16"]["cls"] == "exact"
    for lever in modes.MODES[modes.EXACT] + modes.MODES[modes.FAST]:
        assert lever == "lever_report" or lever in inst, lever


def test_expected_words_are_the_modules_literals():
    from atlasfold_opt.hooks import ln_bf16
    assert tuple(registry.LEVERS["ln_bf16"]["expected"]) == ln_bf16.EXPECTED
    src = open(ln_bf16.__file__).read()
    for word in ln_bf16.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word


def test_ln_bf16_cpu_route_is_the_stock_statement():
    from atlasfold_opt.hooks import ln_bf16
    nm = importlib.import_module(ln_bf16.TARGET)
    ins = ln_bf16.install("exact", "atlasfold-opt", {})
    try:
        assert ins.applied and ins.lever == "ln_bf16"
        ln = nm.LayerNorm(128)
        x = torch.randn(2, 7, 128)
        stock = nm.LayerNorm.forward.__wrapped_stock__
        assert torch.equal(ln(x), stock(ln, x))                          # cpu -> the stock statement, counted
        assert torch.equal(ln(x.bfloat16()), stock(ln, x.bfloat16()))
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.ln_bf16" in line and "state=skipped reason=all_fallback:cpu" in line and "fallback_by=cpu:2" in line and "served=0" in line   # every call took a NAMED route: the family's all_fallback census word
    finally:
        _restore(nm.LayerNorm, "forward")


def test_ln_bf16_gates_leave_dtype_and_autocast_state_alone_on_cpu():
    """CPU: the gates only (the bitwise statement is CUDA-only: tests/test_020_levers_gpu.py). Output dtype == input dtype; an enclosing autocast
    region is untouched by the call."""
    from atlasfold_opt.hooks import ln_bf16
    nm = importlib.import_module(ln_bf16.TARGET)
    ins = ln_bf16.install("exact", "atlasfold-opt", {})
    try:
        ln = nm.LayerNorm(64).to(torch.bfloat16)
        x = torch.randn(3, 64).bfloat16()
        with torch.autocast("cpu", torch.bfloat16):
            y = ln(x)
            assert torch.is_autocast_enabled("cpu")                            # the call did not switch the enclosing region off
        assert y.dtype == torch.bfloat16 and ln(x.float()).dtype == torch.float32
        assert "all_fallback:cpu" in ins.lines[0]()
    finally:
        _restore(nm.LayerNorm, "forward")


def test_active_line_fill_word():
    from atlasfold_opt import stack
    assert stack._fill_word({"det": 0}) == "na"
    assert stack._fill_word({"det": 1, "fill_uninitialized_memory": False}) == "0" and stack._fill_word({"det": 1, "fill_uninitialized_memory": True}) == "1"


def _dit_modules(att, use_high_precision=False):
    torch.manual_seed(0)
    m = att.Attention(channel=768, num_heads=16, use_high_precision=use_high_precision) if "use_high_precision" in att.Attention.__init__.__code__.co_varnames \
        else att.Attention(768, 16)
    return m.eval()


def test_det_recipe_keeps_deterministic_algorithms_without_the_fill():
    from atlasfold_opt import det as D
    before = (torch.are_deterministic_algorithms_enabled(), torch.utils.deterministic.fill_uninitialized_memory)
    try:
        facts = D.apply(1)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.utils.deterministic.fill_uninitialized_memory is False and facts["fill_uninitialized_memory"] is False
        assert facts["deterministic_algorithms"] == "warn_only"
        assert D.apply(0) == {"det": 0}                                   # level 0 applies nothing
        assert D.env_for_subprocess(1) == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} and D.env_for_subprocess(0) == {}
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = before[1]
