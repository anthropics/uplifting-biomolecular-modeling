"""The resident line's REACH gate (modes.reach_gate / apply_reach_gate): from REACH_GATE_BY_CARD[card] polymer tokens up the trunk's
full-N^2 pair cells and their provider rows (modes.REACH_LEVERS) step aside BY NAME on ('big', 'resident') and the offload port's row-block pair
statements serve; below the gate, without a token count, below the port's item gate and on every other line the resolution is the line as written
byte for byte. CPU contracts: the gate per card / N / caller word, the resolution above the gate (== MODEL_OPT_LEVERS_OFF=<those names>, the
ablation), the ACTIVE / LEVER words."""
import pytest

from openfold3_opt import _autoload, modes, report
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
ENV = modes.REACH_GATE_ENV
H100 = {"MODEL_OPT_TARGET_GPU": "H100"}


def _res(n, extra=None, mode="big"):
    env = dict(H100)
    env.update(extra or {})
    return modes.resolve(mode, HOME, environ=env, n_tokens=n)


def _shape(r):
    return (r.exports, r.unsets, r.hooks, r.levers, r.notes, r.conflicts, r.levers_off, r.reach_off, r.reach_gate, modes.describe_line(r))


@pytest.mark.parametrize("environ,expect", [({}, (modes.REACH_GATE_DEFAULT, "default")), ({"MODEL_OPT_TARGET_GPU": "H100"}, (4500, "card:H100")),
                                            ({"MODEL_OPT_TARGET_GPU": "a100"}, (4500, "card:A100")), ({"MODEL_OPT_TARGET_GPU": "L40S"}, (modes.REACH_GATE_DEFAULT, "default")),
                                            ({ENV: "6000", "MODEL_OPT_TARGET_GPU": "H100"}, (6000, "env")), ({ENV: "none"}, (None, "env")), ({ENV: " 0 "}, (0, "env"))])
def test_the_gate_per_card_and_the_callers_word(environ, expect):
    assert modes.reach_min_tokens(environ) == expect


@pytest.mark.parametrize("raw", ["-1", "4.5k", "yes"])
def test_a_malformed_word_is_named(raw):
    with pytest.raises(ValueError, match=ENV):
        modes.reach_min_tokens({ENV: raw})
    r = _res(5060, {ENV: raw})
    assert any(ENV in c for c in r.conflicts), r.conflicts                       # resolve: a conflict (NOT ACTIVE by name), nothing dropped
    assert r.reach_off == [] and all(n in r.levers for n in modes.REACH_LEVERS)


@pytest.mark.parametrize("n", [400, 1012, 1401, 1500, 2000, 2565, 4499])
def test_below_the_gate_the_line_is_as_written_byte_for_byte(n):
    assert _shape(_res(n)) == _shape(_res(n, {ENV: "none"}))
    r = _res(n)
    assert r.reach_gate is None and r.reach_reason is None and r.reach_min_tokens is None and r.reach_off == []
    assert "reach" not in report.activation_line(_rep(r, n))


def test_no_token_count_is_the_line_as_written():
    r = _res(None)
    assert r.reach_gate is None and r.reach_off == [] and all(n in r.levers for n in modes.REACH_LEVERS)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_other_lines_never_carry_the_gate(mode):
    r = _res(9000, mode=mode)
    assert r.reach_gate is None and r.reach_off == []


@pytest.mark.parametrize("n", [4500, 5060, 6144])
def test_at_or_above_the_gate_the_pair_cells_step_aside_by_name(n):
    below, r = _res(4499), _res(n)
    assert r.conflicts == []
    assert r.reach_off == [l for l in below.levers if l in modes.REACH_LEVERS] == list(modes.REACH_LEVERS)
    assert not any(l in r.levers for l in modes.REACH_LEVERS)
    assert all(l in r.levers for l in ("trimul_hostsnap", "triatt_lean", "trans_inplace"))     # the port's row-block pair statements stay: they serve
    assert r.reach_gate == f"reach=aside:n_tok>={modes.REACH_GATE_BY_CARD['H100']} min_tokens=4500(card:H100)" and r.reach_reason == "aside:n_tok>=4500" and r.reach_min_tokens == 4500
    assert not any(k in r.exports for k in modes.PAIR_ENVS) and all(k in r.unsets for k in modes.PAIR_ENVS)
    for name in ("trimul_provider", "triatt_provider"):
        assert not any(k in r.exports for k in modes.LEVER_SWITCHES[name])
    assert r.hooks == below.hooks                                                               # the cells hook stays (the sampler / apb / template cells still arm it)


def test_above_the_gate_equals_the_measured_ablation():
    """The gate's resolution is MODEL_OPT_LEVERS_OFF=<REACH_LEVERS> at the same size (arm B of the 5,060-token decision run) --
    exports, unsets, hooks and levers identical; only the bookkeeping differs (reach_off vs levers_off, the note)."""
    g = _res(5060)
    a = _res(5060, {modes.ENV_LEVERS_OFF: ",".join(modes.REACH_LEVERS), ENV: "none"})
    assert (g.exports, sorted(g.unsets), g.hooks, g.levers) == (a.exports, sorted(a.unsets), a.hooks, a.levers)
    assert g.reach_off == a.levers_off and g.levers_off == [] and a.reach_off == []


def test_the_callers_word_moves_or_removes_the_gate():
    assert _res(2000, {ENV: "1500"}).reach_off == list(modes.REACH_LEVERS)
    assert _res(2000, {ENV: "1500"}).reach_gate == "reach=aside:n_tok>=1500 min_tokens=1500(env)"
    r = _res(9000, {ENV: "none"})
    assert r.reach_off == [] and all(n in r.levers for n in modes.REACH_LEVERS)


def test_below_the_item_gate_nothing_applies_even_when_forced():
    r = _res(400, {ENV: "0"})                                                    # below OF3O_MIN_TOKENS the line is the fast composition (of3o_aside): the reach gate never reads it
    assert r.reach_gate is None and r.reach_off == []


def test_levers_off_and_the_gate_compose():
    r = _res(5060, {modes.ENV_LEVERS_OFF: "trimul_v4"})
    assert r.conflicts == []
    assert r.levers_off == ["trimul_v4", "trimul_provider"] and r.reach_off == ["triatt_block", "triatt_provider", "pair_transition"]
    assert not any(l in r.levers for l in modes.REACH_LEVERS)


def _rep(r, n):
    return {"active": True, "mode": r.mode, "line": r.line, "openfold3_version": "x", "gpu": {"name": "NVIDIA H100 80GB HBM3"},
            "levers_requested": list(r.levers), "levers_applied": list(r.levers), "levers_off": list(r.levers_off), "reach_off": list(r.reach_off),
            "reach_gate": r.reach_gate, "reach_reason": r.reach_reason, "reach_min_tokens": r.reach_min_tokens, "n_tokens": n,
            "of3o_gate": r.of3o_gate, "of3o_reason": r.of3o_reason, "of3o_min_tokens": r.of3o_min_tokens, "conf_gate": r.conf_gate, "conf_reason": r.conf_reason,
            "size_gate": r.size_gate, "gate_reason": r.gate_reason, "notes": list(r.notes)}


def test_the_words_on_the_active_and_lever_lines():
    r = _res(5060)
    rep = _rep(r, 5060)
    active = report.activation_line(rep)
    assert " reach=aside:n_tok>=4500 min_tokens=4500(card:H100) n_tokens=5060" in active, active
    assert " reach_off=trimul_v4,trimul_provider,triatt_block,triatt_provider,pair_transition" in active, active
    lines = {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep)}
    for name in modes.REACH_LEVERS:
        assert f"state=off reason={modes.REACH_GATE_REASON} " in lines[name], lines[name]
        assert "gate=aside:n_tok>=4500 n_tokens=5060 min_tokens=4500" in lines[name], lines[name]
    for name in ("trimul_hostsnap", "triatt_lean", "trans_inplace"):
        assert "state=on" in lines[name], lines[name]


def test_the_callers_word_is_a_declared_package_variable():
    assert ENV in _autoload.DECLARED                                             # an undeclared variable under the prefix is NOT ACTIVE by name (exit 3)
    assert ENV == modes.REACH_GATE_ENV and ENV.endswith("_REACH_GATE")
