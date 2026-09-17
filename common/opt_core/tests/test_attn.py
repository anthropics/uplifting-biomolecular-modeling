"""opt_core.attn on CPU without torch: the size gate's counts close, from_env refuses by name, sdpa_bias is importable pure Python."""
import pytest
from opt_core.attn import size_gate as sg
from opt_core.attn import sdpa_bias as sb

def test_gate_counts_close():
    g = sg.SizeGate("t", min_tokens=4, max_tokens=10)
    assert [g.decide(n).event for n in (3, 4, 10, 11)] == ["gated:lt_min", "served", "served", "gated:gt_max"]
    g.fallback("no_kernel:flash")
    c = g.census()
    assert (c["calls"], c["served"], c["gated"], c["fallback"]) == (4, 1, 2, 1) and c["fallback_by"] == {"no_kernel:flash": 1}
    assert g.fields() == "gate.t=min4:max10 calls=4 served=1 gated=2 fallback=1 fallback_by=no_kernel:flash:1"
    assert g.problems(expect_served_min=1) and not g.problems(expect_served_min=1, allow_fallback=True)
    with pytest.raises(RuntimeError):
        sg.SizeGate("u").fallback("x")            # nothing served to re-book
    assert sg.SizeGate("u").fallback("x", rebook=False) == "fallback:x"

def test_from_env_named_refusal():
    assert sg.from_env("t", "T_MIN", 300, environ={}).min_tokens == 300
    g = sg.from_env("t", "T_MIN", 300, environ={"T_MIN": "off"})
    assert g.min_tokens is None and g.source == "env:T_MIN" and g.bounds_word() == "any"
    with pytest.raises(ValueError, match="T_MIN"):
        sg.from_env("t", "T_MIN", 300, environ={"T_MIN": "4k"})
    with pytest.raises(ValueError):
        sg.SizeGate("bad name")

def test_sdpa_bias_surface_is_pure_python():
    assert sb.BACKENDS == ("auto", "efficient", "flash", "cudnn", "math") and issubclass(sb.Refused, RuntimeError)


def test_size_gate_words_name_the_bound_kind():
    """R8: a largest-tested ceiling gates BY NAME as unmeasured (`gated:gt_max(unmeasured_above_<N>)`), a measured crossover as measured; the
    census keys stay `gated:lt_min` / `gated:gt_max`; the lever evidence labels an unmeasured bound."""
    g = sg.SizeGate("dit", min_tokens=400, max_tokens=2048)                         # defaults: floor measured, ceiling a largest-tested size
    lo, mid, hi = g.decide(10), g.decide(500), g.decide(4096)
    assert (lo.event, lo.word) == ("gated:lt_min", "gated:lt_min(measured_400)")
    assert (mid.event, mid.word, mid.served) == ("served", "served", True)
    assert (hi.event, hi.word, hi.served) == ("gated:gt_max", "gated:gt_max(unmeasured_above_2048)", False)
    ev = dict(g.lever_evidence())
    assert ev["max_tokens"] == 2048 and ev["max_kind"] == "unmeasured" and "min_kind" not in ev and ev["gated"] == 2
    m = sg.SizeGate("x", min_tokens=64, max_tokens=512, min_measured=False, max_measured=True)
    assert m.decide(1).word == "gated:lt_min(unmeasured_below_64)" and m.decide(9999).word == "gated:gt_max(measured_512)"
    assert dict(m.lever_evidence())["min_kind"] == "unmeasured" and "max_kind" not in dict(m.lever_evidence())

