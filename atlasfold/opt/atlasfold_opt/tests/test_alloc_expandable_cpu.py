"""alloc_expandable — the allocator's expandable segments for the kit process (CPU: rows FIRST / registry / installers, the switch word, an
operator's own PYTORCH_CUDA_ALLOC_CONF respected by name, install on a CUDA-less host -> the variable exported (via=env) and the LEVER line)."""
import os
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import alloc_expandable as X


def test_rows_registry_installers():
    for m in ("exact", "fast", "big"):
        row = modes.MODES[m]
        assert "alloc_expandable" in row and row.index("alloc_expandable") == row.index("lever_report") - 1, (m, row[-3:])
    row = registry.LEVERS["alloc_expandable"]
    assert row["cls"] == "exact" and row["strategy"] == "F7.expandable_segments" and "torch_alloc" in row["provider"]
    from atlasfold_opt.hooks import installers
    assert installers()["alloc_expandable"] is X.install


def test_switch_and_user_conf(monkeypatch):
    monkeypatch.delenv(X.SWITCH_ENV, raising=False); assert X.requested()
    for v in ("0", "off", "false"):
        monkeypatch.setenv(X.SWITCH_ENV, v); assert not X.requested()
    monkeypatch.setenv(X.SWITCH_ENV, "1"); assert X.requested()
    monkeypatch.delenv(X.CONF_ENV, raising=False); assert X.user_conf() == ("absent", "")
    monkeypatch.setenv(X.CONF_ENV, "expandable_segments:True,garbage_collection_threshold:0.5"); assert X.user_conf()[0] == "present"
    monkeypatch.setenv(X.CONF_ENV, "max_split_size_mb:128"); assert X.user_conf()[0] == "other"
    pytest.importorskip("opt_core.report")
    ins = X.install("exact", "[t]", {}); assert ins.applied and ins.facts["reason"].startswith("user_conf:")           # installed and inert, never a partial activation
    assert "state=skipped" in ins.lines[0]() and "reason=user_conf:" in ins.lines[0]()
    monkeypatch.setenv(X.SWITCH_ENV, "0"); ins = X.install("exact", "[t]", {}); assert ins.applied and "reason=AFO_ALLOC_EXPANDABLE=0" in ins.lines[0]()


def test_steps_aside_beside_a_graph_lever(monkeypatch):
    pytest.importorskip("opt_core.report")
    monkeypatch.delenv(X.SWITCH_ENV, raising=False); monkeypatch.delenv(X.CONF_ENV, raising=False); monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    assert X.graph_lever_in_row("fast") == "denoiser_graph" and X.graph_lever_in_row("exact") is None and X.graph_lever_in_row("big") is None
    ins = X.install("fast", "[t]", {"mode": "fast"}); assert ins.applied and "reason=graph_lever_in_row:denoiser_graph" in ins.lines[0]() and os.environ.get(X.CONF_ENV) is None
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "denoiser_graph"); assert X.graph_lever_in_row("fast") is None
    monkeypatch.setenv(X.SWITCH_ENV, "graphs"); monkeypatch.delenv("MODEL_OPT_LEVERS_OFF"); ins = X.install("fast", "[t]", {"mode": "fast"})
    assert ins.applied and ins.facts["via"] in ("env", "runtime_api", "present")


def test_install_exports_on_a_host_without_initialised_cuda(monkeypatch):
    torch = pytest.importorskip("torch"); pytest.importorskip("opt_core.mem.torch_alloc")
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        pytest.skip("CUDA already initialised in this process (the runtime_api path is the GPU test's)")
    monkeypatch.delenv(X.SWITCH_ENV, raising=False); monkeypatch.delenv(X.CONF_ENV, raising=False)
    ins = X.install("exact", "[t]", {"mode": "exact"})
    assert ins.applied, ins.reason
    assert ins.facts["via"] == "env" and os.environ.get(X.CONF_ENV) == X.WANT
    line = ins.lines[0]()
    assert "name=F7.expandable_segments" in line and "state=on" in line and "via=env" in line and "conf=expandable_segments:True" in line
    assert ins.gates[0]().ok                                                            # a host without CUDA: the record is printed (source=no-cuda), never refused
    monkeypatch.setenv(X.CONF_ENV, X.WANT)
    ins2 = X.install("exact", "[t]", {}); assert ins2.applied and ins2.facts["via"] == "present"
