"""precision.py — the sampler-autocast word `sampler_amp`: the per-item rule at upstream's policy site (engaged at <= the gate, upstream's
policy untouched above it and above upstream's own AMP size, inert under a stated non-bf16 dtype), the knob, the plan-time size gate
the registry / modes rows that carry the word (fast and the
single-card big line; never exact, never the row-sharded line), the ablation rule, and the pinned upstream sites the word binds."""
import os
import types

import pytest

from opendde_opt import modes, precision, registry

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(TREE, "stock", "src")


def upstream_policy(configs, n_token):
    """runner/inference.py:update_inference_configs restated (1.1.1): the size policy, then the two FORCE variables."""
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    if os.getenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP") == "1":
        configs.skip_amp.sample_diffusion = False
    if os.getenv("OPENDDE_FORCE_CONFIDENCE_AMP") == "1":
        configs.skip_amp.confidence_head = False
    return configs


def cfg(dtype="bf16"):
    return types.SimpleNamespace(dtype=dtype, skip_amp=types.SimpleNamespace(sample_diffusion=True, confidence_head=True))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(precision.GATE_ENV, raising=False)
    monkeypatch.delenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP", raising=False)
    monkeypatch.delenv("OPENDDE_FORCE_CONFIDENCE_AMP", raising=False)
    precision._reset()
    yield
    precision._reset()


# ---------------------------------------------------------------------------------------------------------------- the per-item rule
def test_engaged_at_and_below_the_gate_sets_upstreams_own_configs_value():
    w = precision.make_wrapper(upstream_policy)
    for n in (61, 400, 800, 1200):
        c = w(cfg(), n)
        assert c.skip_amp.sample_diffusion is False          # = what OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1 sets
        assert c.skip_amp.confidence_head is True            # the confidence head's flag is never touched
    st = precision.kit_stats()
    assert st["calls"] == 4 and st["engaged"] == 4 and st["above_gate"] == 0 and st["gate"] == precision.GATE_MAX
    assert getattr(w, "_sampler_amp", False) and w._orig is upstream_policy


def test_above_the_gate_upstreams_policy_stands(monkeypatch):
    assert precision.GATE_MAX == precision.UPSTREAM_AMP_ABOVE == 3840        # as shipped: no kit gate below upstream's own AMP size
    w = precision.make_wrapper(upstream_policy)
    c = w(cfg(), 3000)                                       # upstream's confidence-AMP regime, inside the word's: sampler under autocast, confidence flag upstream's
    assert c.skip_amp.sample_diffusion is False and c.skip_amp.confidence_head is False
    c = w(cfg(), 5000)                                       # above upstream's own sampler-AMP size: upstream runs it under autocast itself, the word changed nothing
    assert c.skip_amp.sample_diffusion is False
    monkeypatch.setenv(precision.GATE_ENV, "1000")           # a card's retune: above the knob upstream's fp32 sampler stands
    c = w(cfg(), 1200)
    assert c.skip_amp.sample_diffusion is True and c.skip_amp.confidence_head is True
    st = precision.kit_stats()
    assert (st["calls"], st["engaged"], st["above_gate"], st["upstream_amp"]) == (3, 1, 1, 1)


def test_a_stated_non_bf16_dtype_has_no_autocast_region_to_inherit():
    w = precision.make_wrapper(upstream_policy)
    c = w(cfg("fp32"), 400)
    assert c.skip_amp.sample_diffusion is True
    assert precision.kit_stats()["no_amp_dtype"] == 1 and precision.kit_stats()["engaged"] == 0


def test_the_knob(monkeypatch):
    monkeypatch.setenv(precision.GATE_ENV, "500")
    w = precision.make_wrapper(upstream_policy)
    assert w(cfg(), 500).skip_amp.sample_diffusion is False and w(cfg(), 501).skip_amp.sample_diffusion is True
    monkeypatch.setenv(precision.GATE_ENV, "0")              # 0 = never engaged
    assert w(cfg(), 1).skip_amp.sample_diffusion is True
    monkeypatch.setenv(precision.GATE_ENV, "many")
    with pytest.raises(ValueError, match=precision.GATE_ENV):
        precision.gate()
    assert precision.GATE_ENV in registry.KNOBS


def test_upstreams_variable_is_still_upstreams_and_still_absent_from_every_line(monkeypatch):
    """The word never exports the variable (every line unsets it); were a caller to state it, upstream's own branch honours it above the gate too."""
    for ln in modes.LINES.values():
        assert "OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP" not in (ln.exports or {})
        assert "OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP" in ln.unset
    monkeypatch.setenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP", "1")
    monkeypatch.setenv(precision.GATE_ENV, "1000")
    w = precision.make_wrapper(upstream_policy)
    assert w(cfg(), 2000).skip_amp.sample_diffusion is False      # upstream's branch, counted above_gate by the word (it changed nothing)
    assert precision.kit_stats()["above_gate"] == 1


# ---------------------------------------------------------------------------------------------------------------- the rows that carry the word
def test_registry_and_lines():
    lv = registry.LEVERS[precision.LEVER]
    assert (lv.kit, lv.cls, lv.tier) == (registry.HOUSE, "forward", "tier2")
    assert registry.PIN_STATUS[precision.LEVER][0] == "tested"
    assert precision.LEVER not in modes.LINES["LSTAR2A"].levers                        # no default line since 0.2.52 (registry row: speed ~0 on top of the sampler kernels)
    assert precision.LEVER not in modes.LINES["S1"].levers                            # never exact (bf16 sampler numerics)
    assert precision.LEVER not in modes.BIG_DROP and precision.LEVER not in modes.BIG_KIT_ROWS_OFF["pair_offload"]   # off fast's set, so nothing for the big lines to drop
    assert precision.LEVER not in modes.LINES["BIG_F"].levers and precision.LEVER not in modes.big_line("BIG_F", ("sample_chunk",)).levers   # neither the offload row nor the resident set

def test_individually_ablatable(monkeypatch):
    assert precision.LEVER in modes.LEVER_SWITCHES and modes.LEVER_SWITCHES[precision.LEVER] == ()
    fast = modes.LINES["LSTAR2A"]
    assert precision.LEVER not in fast.levers                                          # no default line since 0.2.52: the word is selectable by name only
    out = modes.line_without(fast, (precision.LEVER,))
    assert out.levers == fast.levers and out.exports == fast.exports                   # nothing to compose out: the line byte for byte


def test_counted_and_reported():
    from opendde_opt import ran, report
    assert precision.LEVER in ran.COUNTERS
    w = precision.make_wrapper(upstream_policy); w(cfg(), 400); w(cfg(), 5000)
    ev = report.lever_evidence(precision.LEVER, {"sampler_amp": precision.kit_stats()})
    assert ev["calls"] == 2 and ev["engaged"] == 1 and ev["above_gate"] == 0 and ev["upstream_amp"] == 1 and ev["gate"] == precision.GATE_MAX
    assert ev["policy"] == "skip_amp.sample_diffusion=False"


# ---------------------------------------------------------------------------------------------------------------- the pinned upstream sites
def test_the_pinned_sites_are_upstreams():
    inf = open(os.path.join(SRC, "runner", "inference.py")).read().splitlines()
    body = "\n".join(inf[1488:1516])                                                  # :1489-1516 — the policy function and the two FORCE variables
    assert "def update_inference_configs(configs: OpenDDEConfig, n_token: int)" in body
    assert 'os.getenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP") == "1"' in body and "configs.skip_amp.sample_diffusion = False" in body
    assert "if n_token > 3840:" in body and "elif n_token > 2560:" in body
    assert "update_inference_configs(configs, n_token)" in inf[1544]                  # :1545 — resolved by name at call time (_prepare_prediction_batch)
    mdl = open(os.path.join(SRC, "opendde", "model", "opendde.py")).read().splitlines()
    assert "autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)" in mdl[1331]   # :1332 — read per sampler call
    assert precision.UPSTREAM_AMP_ABOVE == 3840
