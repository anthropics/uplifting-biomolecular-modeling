"""CPU tests for lever sampler_prep (lib/kit112_src/infopt_graphs/protenix/sampler_prep.py + the graphed.py sites): the parts word (all six or
none), the vectorised step scalars equal value for value to the per-step statement the loop runs without the lever, the shape key / index-value
comparison, and the report's reading of the loop's prep_report() (served; the by-design step-aside without the sampler graph; fallback on a
partial parts word or a failed poison probe)."""
import os
import sys

import pytest

from protenix_v1_opt import kit as K, modes as M, report as R

PKG = os.path.join(K.kit_home(), "lib", "kit112_src")
WORD = "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec"


def _sp():
    pytest.importorskip("torch")
    if PKG not in sys.path:
        sys.path.insert(0, PKG)
    from infopt_graphs.protenix import sampler_prep as SP
    return SP


def test_parts_word_all_or_none():
    SP = _sp()
    on, off = SP.PrepConfig(True), SP.PrepConfig(False)
    assert on.describe() == WORD == "+".join(SP.PARTS) and all(getattr(on, p) for p in SP.PARTS)
    assert off.describe() == "off" and off.parts == () and not any(getattr(off, p) for p in SP.PARTS)
    assert SP.PrepConfig(True, parts=("stepvec",)).describe() == "stepvec"          # a partial config prints a partial word (the kit's LEVER line reads it as a fallback)
    src = open(os.path.join(PKG, "infopt_graphs", "protenix", "graphed.py")).read()
    assert "def prep_report(self):" in src and src.index("class GraphedDenoiseLoop") < src.index("def prep_report(self):") < src.index("\ndef install(")
    assert "sampler_prep=False" in src and "prep=_sp.PrepConfig(bool(sampler_prep))" in src


def test_stepvec_equals_the_per_step_statement():
    SP = _sp(); import torch
    g = torch.Generator().manual_seed(0)
    sched = torch.sort(torch.rand(201, generator=g) * 160.0, descending=True).values.to(torch.float32)   # a decreasing 200-step schedule, fp32 like the sampler's
    gamma_min, gamma0 = 1.0, 0.8
    pattern = tuple(bool(v) for v in (sched[1:] > gamma_min).tolist())
    vec = SP.step_scalars(sched, pattern, gamma0)
    assert len(vec) == 200
    for i, (c_last, c_next) in enumerate(zip(sched[:-1], sched[1:])):                    # graphed.py's own per-step statement (the lever off)
        gamma = float(gamma0) if pattern[i] else 0
        t_hat = c_last * (gamma + 1); dnl = torch.sqrt(t_hat ** 2 - c_last ** 2); dt = c_next - t_hat
        assert torch.equal(vec[i][0], t_hat) and torch.equal(vec[i][1], dnl) and torch.equal(vec[i][2], dt), i


def test_shape_key_and_index_values():
    SP = _sp(); import torch
    f = {"asym_id": torch.tensor([0, 0, 1]), "ref_pos": torch.zeros(3, 3), "name": "x"}
    assert SP.feature_shape_key(f) == (("asym_id", (3,), "torch.int64"), ("ref_pos", (3, 3), "torch.float32"))
    ent = {"cond": {"input_feature_dict": {"asym_id": torch.tensor([0, 0, 1]), "ref_pos": torch.ones(3, 3)}}}
    assert SP.same_index_values(ent, f)                                                   # floating tensors are inputs by value: not compared
    assert not SP.same_index_values(ent, dict(f, asym_id=torch.tensor([0, 1, 1])))       # an integer feature with other values: a new capture
    assert not SP.same_index_values({"cond": {}}, f)


def _acct(res, prep, sampler=None):
    return {"cfg": {"trimul": res.trimul, **{lv: True for lv in res.levers}}, "counts": {"trimul": {res.trimul: 12}},
            "sampler": {"graphs": bool(prep.get("on")), "prep": prep, "hoist_installed": True,
                        "sampler": sampler if sampler is not None else {"captures": 1, "replays": 199, "eager_steps": 0, "bypass": 0}, "hoist": {"hits": 199, "records": 1}}}


def test_report_reads_the_prep_report():
    res = M.resolve("fast", environ={})
    full = {"on": True, "parts": WORD, "poison": "ok", "stats": {"pool_chained": 2, "key_value_miss": 0}, "aside": None}
    ev = R.kit_evidence(_acct(res, full), res.trimul, ("sampler_prep",))
    assert set(ev) == {"fast", "sampler_prep"} and ev["sampler_prep"]["served"] == 200 and ev["sampler_prep"]["fallback"] == {} and ev["sampler_prep"]["gated"] == {"pool_chained": 2}
    line = [l for l in R.lever_lines(ev, {"mode": "fast"}) if l.endswith(" lever=sampler_prep")][0]
    assert line == ("[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.sampler_prep state=on impl=lib/kit112_src/infopt_graphs origin=kit strategy=LOCAL.protenix_v1.sampler_prep "
                    "served=200 gated=2 gated_by=pool_chained:2 parts=" + WORD + " poison=ok lever=sampler_prep"), line
    aside = R.kit_evidence(_acct(res, {"on": False, "aside": "no_sampler_graph"}, sampler={}), res.trimul, ("sampler_prep",))["sampler_prep"]
    assert aside["served"] == 0 and aside["rows"] == {"state": "range-off", "why": "no_sampler_graph"} and R.partial_of({"sampler_prep": aside}) == ([], None)
    part = R.kit_evidence(_acct(res, dict(full, parts="rot_async+keycheck")), res.trimul, ("sampler_prep",))["sampler_prep"]
    assert part["fallback"] == {"parts": "rot_async+keycheck"} and R.partial_of({"sampler_prep": part})[0] == ["sampler_prep"]
    bad = R.kit_evidence(_acct(res, dict(full, poison="failed:ValueError")), res.trimul, ("sampler_prep",))["sampler_prep"]
    assert bad["fallback"] == {"poison": "failed:ValueError"} and R.partial_of({"sampler_prep": bad})[0] == ["sampler_prep"]
    assert M.GRAPH_CAP_LEVERS == ("sg", "hoist", "sampler_prep") and "sampler_prep" in M.resolve("exact", environ={}).levers and "sampler_prep" not in M.resolve("big", environ={}).levers
