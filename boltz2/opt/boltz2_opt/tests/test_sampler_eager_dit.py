"""The fused DiT step (dit_fused) served from the STOCK eager sampler loop — the memory row's composition (CHANGES 0.3.8): the sampler adapter's
fail-closed gate counts eager sample() calls, its LEVER line names the host, and the packed weights' lifetime word (weights=resident|sample) is a
value of bz_sampler_dit.install by source contract (the module needs triton + CUDA to install; its word grammar and release rule are read here)."""
import os
import re
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from .. import modes, sampler  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "sampler", "src")


def _stub(calls, samples, served, rollout_on):
    dit_stats = {"gemm": "bf16", "attn": "bf16", "layers": 24, "launches_per_layer": 10, "calls": calls * 200, "served": served, "scope": {}, "packs": calls,
                 "pack_gib": 1.03, "weights_mib": 513, "weights": "sample", "weights_builds": calls, "weights_releases": calls, "out_dtype_probe": True, "mask_fold": "bias"}
    mod = types.SimpleNamespace(STATS={"calls": calls, "samples": samples, "replays": 0, "captures": 0, "capture_failed": 0, "capture_s": [], "scope": {},
                                       "kabsch": "gesvd", "div_recipe": "recip", "predraw": None, "guard_prints": 0},
                                report=lambda: {"dit": {"stats": dit_stats, "cfg": {"weights": "sample"}}})
    levers = {"rollout": ({"state": "on", "word": "graph"} if rollout_on else {"state": "off", "reason": "word_absent"}),
              "align_jacobi64": {"state": "off", "reason": "word_absent"}, "align_aligncap": {"state": "off", "reason": "word_absent"},
              "dit_fused": {"state": "on", "word": "bf16,weights=sample"}}
    return mod, levers


def test_the_gate_counts_eager_sample_calls_for_the_fused_step(monkeypatch):
    """On the memory row no roll-out samples (samples stays 0); sample() calls still reach bz_sampler (calls). A fused step that served nothing over
    those calls is refused by name — the same rule the roll-out path had through `samples`."""
    mod, levers = _stub(calls=5, samples=0, served=0, rollout_on=False)
    monkeypatch.setitem(sampler._STATE, "mod", mod); monkeypatch.setitem(sampler._STATE, "applied", ["dit_fused"]); monkeypatch.setitem(sampler._STATE, "levers", levers)
    v = sampler.verdict()
    assert v["ok"] is False and v["reason"] == "dit_fused installed but served no token-transformer call", v
    mod, levers = _stub(calls=5, samples=0, served=1000, rollout_on=False)
    monkeypatch.setitem(sampler._STATE, "mod", mod); monkeypatch.setitem(sampler._STATE, "levers", levers)
    assert sampler.verdict() == {"ok": True, "idle": False, "reason": None}
    line = sampler.line()
    assert "LEVER name=dit_fused state=on" in line and "word=bf16,weights=sample" in line and "served=1000" in line and "weights=sample" in line and "weights_builds=5" in line and "host=eager" in line, line
    mod, levers = _stub(calls=5, samples=5, served=1000, rollout_on=True)
    monkeypatch.setitem(sampler._STATE, "mod", mod); monkeypatch.setitem(sampler._STATE, "applied", ["rollout", "dit_fused"]); monkeypatch.setitem(sampler._STATE, "levers", levers)
    assert "host=rollout" in sampler.line()


def test_the_memory_rows_word_and_the_modules_word_grammar_agree():
    """modes: big's BOLTZ_SAMPLER_DIT = fast's word + `weights=sample`; bz_sampler_dit.install parses `weights=resident|sample` (refusing other values
    by name), release() drops the packed weights under weights=sample, attach() builds them only under weights=resident, and prepare() packs the pair
    bias layer by layer (no [L, H, N, N] fp32 transient) — source contract."""
    assert modes.env_row("big")["BOLTZ_SAMPLER_DIT"].split(",") == [modes.env_row("fast")["BOLTZ_SAMPLER_DIT"], "weights=sample"]
    src = open(os.path.join(SRC, "bz_sampler_dit.py")).read()
    assert 'if w.startswith("weights="):' in src and 'if weights not in ("resident", "sample"):' in src and '"weights": "resident"' in src
    rel = src[src.index("def release():"):src.index("def _state_for(tt):")]
    assert 'if _CFG["weights"] == "sample":' in rel and '_STATE["st"] = None; _STATE["tt"] = None' in rel and 'STATS["weights_releases"] += 1' in rel
    att = src[src.index("def attach(score_model):"):src.index("def report():")]
    assert 'if _CFG["weights"] == "resident":' in att and "_state_for(tt)" in att
    prep = src[src.index("def prepare("):src.index("def release():")]
    assert "for l in range(st.L):" in prep and "pb[l] = (zv[l].to(dtype=torch.float32) + mterm).to(dt)" in prep and "torch.empty((st.L, st.H, N, N), dtype=dt" in prep
    roll = open(os.path.join(SRC, "bz_sampler.py")).read()
    i = roll.index("def sample_rollout("); body = roll[i:roll.index("sample_rollout._bzs_rollout = True")]
    assert body.index("dit.prepare(") < body.index('if why != "rollout_off":') < body.index("return inner(") < body.index("dit.release()"), \
        "bz_sampler packs the pair bias at sample() entry BEFORE it hands an out-of-scope / roll-out-off call to the inner (stock) sampler, and releases at exit: the eager loop's host"


def test_at_n_gpu_above_1_the_fused_step_leaves_the_row_sharded_line_by_name():
    modes.set_n_gpu(2)
    try:
        r2 = modes.resolve("big")
        assert "dit_fused" not in r2["levers"] and r2["tp_off"]["dit_fused"].startswith("xP_line_unchanged:") and "BOLTZ_SAMPLER_DIT" not in r2["env"] and "sampler" not in r2["attach"]
        assert not ({"atom_fused", "atom_gemm"} & set(r2["levers"])) and r2["tp_off"]["atom_fused"].startswith("xP_line_not_measured:") and r2["tp_off"]["atom_gemm"] == r2["tp_off"]["atom_fused"] and r2["env"].get("BOLTZ_ATOM") == "keys" and "atom_keys_gather" in r2["levers"] \
            and "BOLTZ_ATOM_RELEASE" not in r2["env"] and "BOLTZ_ATOM_GEMM" not in r2["env"]   # (registry `words`: the release word is the lever's own) the fused atom kernels leave the xP line by name with their words; the exact key gather serves the stock layers per rank
    finally:
        modes.set_n_gpu(1)
    assert "dit_fused" in modes.resolve("big")["levers"] and "sampler" in modes.resolve("big")["attach"]
