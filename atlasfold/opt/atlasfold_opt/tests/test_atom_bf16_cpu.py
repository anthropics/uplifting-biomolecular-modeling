"""atom_bf16 on CPU: fast-row membership and order (after atom_kdedup / atom_tf32, before denoiser_graph, before relpos_lazy), registry /
installer coverage, the expected words == the module's literals, the window's depth counting, the CPU route and the disabled word are the
stock statement (torch.equal), the bf16 body on CPU tensors through the in-process override returns the caller's dtype and stays inside bf16
tolerance of the fp32 statement, the KEEP_COORD producer guard turns autocast off around AtomRelativePositionEncoding.forward inside the window."""
import importlib

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers

from .test_atom_tf32_cpu import _tiny_stack_inputs


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


def test_row_registry_installer():
    row = modes.MODES[modes.FAST]
    assert "atom_bf16" in row and "atom_bf16" not in modes.MODES[modes.EXACT] and "atom_bf16" in modes.MODES[modes.BIG]
    assert registry.LEVERS["atom_bf16"]["cls"] == "fast" and "atom_bf16" in installers()
    assert row.index("atom_tf32") < row.index("atom_bf16") < row.index("denoiser_graph") < row.index("relpos_lazy")
    assert row.index("atom_kdedup") < row.index("atom_bf16") and row.index("sampler_hoist") < row.index("atom_bf16")   # the outermost AtomAttentionStack.forward wrapper


def test_expected_words_are_the_module_literals():
    from atlasfold_opt.hooks import atom_bf16
    assert tuple(registry.LEVERS["atom_bf16"]["expected"]) == atom_bf16.EXPECTED
    src = open(atom_bf16.__file__).read()
    for word in atom_bf16.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word


def test_window_counts_depth_and_enables_autocast():
    from atlasfold_opt.hooks import atom_bf16
    assert atom_bf16._STATE["depth"] == 0
    with atom_bf16.window("cpu"):
        assert atom_bf16._STATE["depth"] == 1 and torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16
        with atom_bf16.window("cpu"):
            assert atom_bf16._STATE["depth"] == 2
        assert atom_bf16._STATE["depth"] == 1
    assert atom_bf16._STATE["depth"] == 0 and not torch.is_autocast_enabled("cpu")
    with pytest.raises(RuntimeError):
        with atom_bf16.window("cpu"):
            raise RuntimeError("boom")
    assert atom_bf16._STATE["depth"] == 0


def test_cpu_route_and_disabled_word_are_the_stock_statement(monkeypatch):
    from atlasfold_opt.hooks import atom_bf16
    AA = importlib.import_module(atom_bf16.TARGET); RP = importlib.import_module(atom_bf16.PRODUCER)
    monkeypatch.delenv("AFO_ATOM_BF16", raising=False); atom_bf16._STATE["override"] = None
    stock = AA.AtomAttentionStack.forward
    ins = atom_bf16.install("fast", "atlasfold-opt", {})
    try:
        assert ins.applied and AA.AtomAttentionStack.forward.__wrapped_stock__ is stock
        stack, batch, q, c = _tiny_stack_inputs(AA, RP)
        with torch.no_grad():
            ref = stock(stack, batch, q, c)
            out = stack(batch, q, c)                                          # CPU tensors: `cpu`, the stock statement
        assert torch.equal(out, ref)
        led = ins.facts["ledger"]
        assert led.served == 0 and led.fallbacks == {"cpu": 1}
        monkeypatch.setenv("AFO_ATOM_BF16", "0")
        with torch.no_grad():
            out2 = stack(batch, q, c)
        assert torch.equal(out2, ref) and led.fallbacks == {"cpu": 1, "disabled": 1}
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.atom_bf16 state=skipped reason=all_fallback:" in line and "switch=AFO_ATOM_BF16=0" in line, line
    finally:
        ins.facts["restore"](); atom_bf16._STATE["override"] = None
    assert AA.AtomAttentionStack.forward is stock and RP.AtomRelativePositionEncoding.forward is getattr(RP.AtomRelativePositionEncoding.forward, "__wrapped_stock__", RP.AtomRelativePositionEncoding.forward)


def test_bf16_body_on_cpu_tensors_returns_the_callers_dtype_within_bf16_tolerance(monkeypatch):
    """The window itself (device_type cpu here): Linear GEMMs in bf16, LayerNorm fp32, residual stream fp32 -> output dtype == input dtype,
    finite, inside bf16 tolerance of the fp32 statement (never bitwise: tolerance class)."""
    from atlasfold_opt.hooks import atom_bf16
    AA = importlib.import_module(atom_bf16.TARGET); RP = importlib.import_module(atom_bf16.PRODUCER)
    stack, batch, q, c = _tiny_stack_inputs(AA, RP)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                                                        # stock inits zero the output linears: randomise so the bf16 products matter
        for p in stack.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
        ref = stack(batch, q, c)
        with atom_bf16.window("cpu"):
            out = AA.AtomAttentionStack.forward(stack, batch, q, c)
    assert out.dtype == ref.dtype == torch.float32 and torch.isfinite(out).all()
    assert not torch.equal(out, ref)
    err = float((out - ref).abs().max()); scale = float(ref.abs().max())
    assert err <= 0.05 * scale + 0.05, (err, scale)


def test_producer_guard_turns_autocast_off_inside_the_window(monkeypatch):
    from atlasfold_opt.hooks import atom_bf16
    AA = importlib.import_module(atom_bf16.TARGET); RP = importlib.import_module(atom_bf16.PRODUCER)
    monkeypatch.delenv("AFO_ATOM_BF16", raising=False); atom_bf16._STATE["override"] = None
    seen = []
    rp_stock = RP.AtomRelativePositionEncoding.forward

    def spy(self, *a, **k):
        seen.append(torch.is_autocast_enabled("cpu"))
        return rp_stock(self, *a, **k)
    RP.AtomRelativePositionEncoding.forward = spy
    ins = atom_bf16.install("fast", "atlasfold-opt", {})
    try:
        _, batch, _, _ = _tiny_stack_inputs(AA, RP)                              # (builds the features through the producer once itself: seen[0])
        enc = RP.AtomRelativePositionEncoding()
        enc(batch)                                                               # outside the window: whatever the caller has (off)
        with atom_bf16.window("cpu"):
            assert torch.is_autocast_enabled("cpu")
            enc(batch)                                                           # inside: the guard turns autocast off around the producer
        assert seen == [False, False, False] and ins.facts["ledger"].get("guard_relpos") == 1   # inside the window the producer saw autocast OFF
    finally:
        ins.facts["restore"](); RP.AtomRelativePositionEncoding.forward = rp_stock; atom_bf16._STATE["override"] = None
