"""The dit_attn lever's dtype gate (cells/dit_attn.py): the kernel serves 16-bit operands only; fp32 operands run upstream's attention core
by design, counted `gated` (gated_by dtype_fp32) — never a fallback — and the LEVER line's counts close (calls == served + gated + fallback).
The operand dtype is decided before any projection the way upstream's Linear decides it. CPU only: no model, no kernel."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from openfold3_ob0_opt.cells import dit_attn as D


@pytest.fixture()
def gate():
    from opt_core.attn import size_gate
    saved = (D.GATE, dict(D.STATE))
    D.GATE = size_gate.from_env("dit_attn", min_var=D.ENV_MIN, default_min=D.MIN_DEFAULT, environ={})
    D.STATE.update({"dtype_gated": 0, "hp_asked": 0, "hp_overridden": 0, "first": None, "first_gated_dtype": None})
    yield D.GATE
    D.GATE = saved[0]; D.STATE.clear(); D.STATE.update(saved[1])


def test_serves_16bit_operands_only():
    assert D.serves_dtype(torch.bfloat16) and D.serves_dtype(torch.float16)
    assert not D.serves_dtype(torch.float32) and not D.serves_dtype(torch.float64)
    assert D.DTYPE_WORD == "dtype16"


def test_projected_dtype_follows_upstreams_linear():
    lin, lin_fp32 = SimpleNamespace(precision=None), SimpleNamespace(precision=torch.float32)
    x32, x16 = torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.bfloat16)
    assert D.projected_dtype(x32, lin) is torch.float32                                  # no autocast: the input's dtype (the fp32 roll-out) -> gated
    assert D.projected_dtype(x16, lin) is torch.bfloat16                                 # a bf16 input is projected in bf16 -> served
    with torch.autocast("cpu", dtype=torch.bfloat16):                                    # autocast on (the bf16 roll-out): nn.functional.linear runs in the autocast dtype -> served
        assert torch.is_autocast_enabled("cpu") and D.projected_dtype(x32, lin) is torch.bfloat16
        assert D.projected_dtype(x32, lin_fp32) is torch.float32                         # a Linear with a set precision returns the input's dtype whatever autocast says -> gated
        want = torch.nn.functional.linear(x32, torch.zeros(3, 4)).dtype                  # the prediction agrees with what the op does under autocast
        assert want is D.projected_dtype(x32, lin)
    assert D.projected_dtype(x32, lin) is torch.nn.functional.linear(x32, torch.zeros(3, 4)).dtype is torch.float32


def test_counts_close_and_the_line_names_both_gates(gate):
    gate.decide(100)                                   # below the size floor: gated_by lt_min
    D.STATE["dtype_gated"] += 2                        # two fp32-operand calls: gated_by dtype_fp32 (what forward() books before returning the stock core)
    gate.decide(400); gate.decide(400)                 # two 16-bit calls at 400 tokens: served
    pairs = dict(D.gate_pairs())
    assert pairs["gate.dit_attn"] == "min256,dtype16" and (pairs["calls"], pairs["served"], pairs["gated"], pairs["fallback"]) == (5, 2, 3, 0)
    assert pairs["calls"] == pairs["served"] + pairs["gated"] + pairs["fallback"] and pairs["gated_by"] == {"lt_min": 1, "dtype_fp32": 2}
    assert D.fallbacks() == {}                         # a gate is not a fallback: the kit's exit tally stays fallbacks=none
    D.STATE["installed"], D.STATE["impl"] = True, "/x/dtk_kernels.py"
    try:
        assert D.census_line() == ("[openfold3_ob0-opt/dit_attn] LEVER name=dit_attn state=on impl=/x/dtk_kernels.py gate.dit_attn=min256,dtype16 "
                                   "calls=5 served=2 gated=3 fallback=0 gated_by=lt_min:1,dtype_fp32:2 high_precision_asked=0 high_precision_overridden=0"
                                   " core=pending:auto core_served=0 core_refused=none modes=none")
    finally:
        D.STATE["installed"], D.STATE["impl"] = False, None
    gate.fallback("head_dim:256")                      # a served call the kernel refused: re-booked as a NAMED fallback, listed in full
    pairs = dict(D.gate_pairs())
    assert (pairs["calls"], pairs["served"], pairs["gated"], pairs["fallback"]) == (5, 1, 3, 1) and pairs["fallback_by"] == {"head_dim:256": 1} and D.fallbacks() == {"head_dim:256": 1}
