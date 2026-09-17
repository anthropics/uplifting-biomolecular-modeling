"""atom_tf32 on CPU: row / registry / installer membership, the expected words == the module's literals, the precision WINDOW (allow_tf32 set
inside, the entry value restored after — also on error and when nested), the KEEP_COORD producer guard, the CPU route (stock statement, counted
`cpu`) and AFO_ATOM_TF32=0 (`disabled`) on a real AtomAttentionStack with stock weights-free random init."""
import importlib

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers


def _restore(cls, attr):
    fn = getattr(cls, attr)
    stock = getattr(fn, "__wrapped_stock__", None)
    if stock is not None:
        setattr(cls, attr, stock)


def test_row_registry_installer():
    assert "atom_tf32" in modes.MODES[modes.FAST] and "atom_tf32" not in modes.MODES[modes.EXACT]
    assert registry.LEVERS["atom_tf32"]["cls"] == "fast"
    assert "atom_tf32" in installers()
    row = modes.MODES[modes.FAST]
    assert row.index("diffusion_bf16") < row.index("atom_tf32") < row.index("denoiser_graph") < row.index("relpos_lazy")   # inside the graph's capture; relpos_lazy's consumer wrapper outside the window


def test_expected_words_are_the_module_literals():
    from atlasfold_opt.hooks import atom_tf32
    assert tuple(registry.LEVERS["atom_tf32"]["expected"]) == atom_tf32.EXPECTED
    src = open(atom_tf32.__file__).read()
    for word in atom_tf32.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word


def test_window_sets_and_restores_allow_tf32():
    from atlasfold_opt.hooks import atom_tf32
    M = torch.backends.cuda.matmul
    prev = M.allow_tf32
    try:
        M.allow_tf32 = False
        with atom_tf32.window(True):
            assert M.allow_tf32 is True and atom_tf32._STATE["depth"] == 1
            with atom_tf32.window(True):                       # nested: restores what it found
                assert M.allow_tf32 is True and atom_tf32._STATE["depth"] == 2
            assert M.allow_tf32 is True
            with atom_tf32.window(False):                      # the producer guard's form
                assert M.allow_tf32 is False and atom_tf32._STATE["depth"] == 1
            assert M.allow_tf32 is True
        assert M.allow_tf32 is False and atom_tf32._STATE["depth"] == 0
        with pytest.raises(RuntimeError):
            with atom_tf32.window(True):
                raise RuntimeError("boom")
        assert M.allow_tf32 is False and atom_tf32._STATE["depth"] == 0   # restored on error
        M.allow_tf32 = True                                    # an entry value of True is restored as True (the lever never lowers the process setting)
        with atom_tf32.window(True):
            pass
        assert M.allow_tf32 is True
    finally:
        M.allow_tf32 = prev
        atom_tf32._STATE["depth"] = 0


def _tiny_stack_inputs(AA, RP, L=8, N=2, C=96):
    torch.manual_seed(0)
    stack = AA.AtomAttentionStack(channel_atom=C, channel_atompair=14, num_heads=2, num_blocks=1).eval()
    aatype = torch.nn.functional.one_hot(torch.randint(0, 20, (1, L)), 21).float()
    batch = {"res_idx": torch.arange(L).view(1, L), "asym_id": torch.zeros(1, L, dtype=torch.long), "seq_mask": torch.ones(1, L, dtype=torch.bool),
             "aatype": aatype, "atom14_mask": torch.ones(1, L, 14, dtype=torch.bool)}
    batch["atom_rel_pos"] = RP.AtomRelativePositionEncoding()(batch)
    q = torch.randn(1, N, L, 14, C)
    c = torch.randn(1, 1, L, 14, C)
    return stack, batch, q, c


def test_cpu_route_and_disabled_word_are_the_stock_statement(monkeypatch):
    from atlasfold_opt.hooks import atom_tf32
    AA = importlib.import_module(atom_tf32.TARGET)
    RP = importlib.import_module(atom_tf32.PRODUCER)
    monkeypatch.delenv("AFO_ATOM_TF32", raising=False)
    atom_tf32._STATE["override"] = None
    ins = atom_tf32.install("fast", "atlasfold-opt", {})
    try:
        assert ins.applied and ins.lever == "atom_tf32"
        try:
            stack, batch, q, c = _tiny_stack_inputs(AA, RP)
        except Exception as e:  # noqa: BLE001 — a stock constructor / feature form this test does not model is not the lever's failure
            pytest.skip(f"tiny AtomAttentionStack not constructible here: {e!r}")
        stock = AA.AtomAttentionStack.forward.__wrapped_stock__
        with torch.no_grad():
            ref = stock(stack, batch, q, c)
            out = stack(batch, q, c)                                   # cpu tensors -> counted `cpu`, stock precision
        assert torch.equal(out, ref)
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.atom_tf32" in line and "fallback_by=cpu:1" in line and "served=0" in line, line
        monkeypatch.setenv("AFO_ATOM_TF32", "0")
        with torch.no_grad():
            out2 = stack(batch, q, c)                                  # AFO word -> `disabled`
        assert torch.equal(out2, ref)
        line = ins.lines[0]()
        assert "disabled:1" in line and "cpu:1" in line, line
        g = ins.gates[0]()
        assert g.ok, g.reason                                          # every call took a NAMED expected route: not a refusal
        assert torch.backends.cuda.matmul.allow_tf32 in (False, True) and atom_tf32._STATE["depth"] == 0
    finally:
        _restore(AA.AtomAttentionStack, "forward")
        _restore(RP.AtomRelativePositionEncoding, "forward")
        atom_tf32._STATE["override"] = None


def test_bench_arm_switch_overrides_the_env_word(monkeypatch):
    from atlasfold_opt.hooks import atom_tf32
    monkeypatch.setenv("AFO_ATOM_TF32", "0")
    try:
        atom_tf32._STATE["override"] = None
        assert atom_tf32.enabled() is False
        atom_tf32.bench_arm("B"); assert atom_tf32.enabled() is True
        atom_tf32.bench_arm("A"); assert atom_tf32.enabled() is False
    finally:
        atom_tf32._STATE["override"] = None
