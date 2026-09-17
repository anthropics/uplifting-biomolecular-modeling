"""alloc_expandable beside the CUDA-graph lever: steps aside by name unless NO record of the run can capture (every padded bucket above
AFO_DENOISER_GRAPH_MAX_TOKENS) — then it engages as in the rows without the graph lever (CPU: the decision function + install on a stub allocator)."""
import types
import pytest

from atlasfold_opt.hooks import alloc_expandable as AX


def test_graph_gate_decision(monkeypatch):
    monkeypatch.delenv("AFO_DENOISER_GRAPH_MAX_TOKENS", raising=False)
    assert AX.graph_gate_exceeded({"min_bucket": 1280, "max_bucket": 1280}) == "exceeded:1280>1024"
    assert AX.graph_gate_exceeded({"min_bucket": 1024}) is None                    # 1024 captures: the graph lever keeps the allocator
    assert AX.graph_gate_exceeded({"min_bucket": 512, "max_bucket": 1280}) is None  # one capturable record is enough to step aside
    assert AX.graph_gate_exceeded({}) is None and AX.graph_gate_exceeded(None) is None   # inputs unknown at start-up: step aside as before
    monkeypatch.setenv("AFO_DENOISER_GRAPH_MAX_TOKENS", "1536")
    assert AX.graph_gate_exceeded({"min_bucket": 1280}) is None and AX.graph_gate_exceeded({"min_bucket": 1664}) == "exceeded:1664>1536"


def test_fast_row_steps_aside_or_engages_by_the_hint(monkeypatch):
    monkeypatch.delenv(AX.CONF_ENV, raising=False); monkeypatch.delenv(AX.SWITCH_ENV, raising=False); monkeypatch.delenv("AFO_DENOISER_GRAPH_MAX_TOKENS", raising=False)
    monkeypatch.setattr(AX, "graph_lever_in_row", lambda mode: "denoiser_graph" if mode == "fast" else None)
    def skipped_reason(ins):                                                     # a named step-aside: the lever reports itself (state=skipped reason=…), no gate
        return (ins.facts or {}).get("reason") if ins.applied else ins.reason
    ins = AX.install("fast", "[t]", {"mode": "fast", "det": 0, "inputs": {"min_bucket": 896}})
    assert skipped_reason(ins) == "graph_lever_in_row:denoiser_graph" and not ins.gates
    ins = AX.install("fast", "[t]", {"mode": "fast", "det": 0})                     # no hint: as before
    assert skipped_reason(ins) == "graph_lever_in_row:denoiser_graph"
    written = []
    TA = types.SimpleNamespace(write_conf=lambda conf, via="env": written.append((conf, via)), effective=lambda: {"expandable": None, "source": "pending"})
    import opt_core.mem as M
    monkeypatch.setattr(M, "torch_alloc", TA, raising=False)
    import sys; monkeypatch.setitem(sys.modules, "opt_core.mem.torch_alloc", TA)
    ins = AX.install("fast", "[t]", {"mode": "fast", "det": 0, "inputs": {"min_bucket": 1280, "max_bucket": 1280}})
    assert ins.applied, ins.reason
    assert ins.facts["graph_gate"] == "exceeded:1280>1024" and written and written[0][0] == AX.WANT
    line = ins.lines[0]()
    assert "name=F7.expandable_segments state=on " in line and " graph_gate=exceeded:1280>1024" in line and " lever=alloc_expandable" in line
