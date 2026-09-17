"""The sampler levers' resident state is bounded by input size: below BOLTZ_GRAPH_RELEASE_MIN_TOKENS (1024) the captured step graph (its
private pool, its static conditioning clones) and the hoist's cache live for the process; from it a sample() releases them when it returns, so
the confidence module and the next prediction's trunk run with the stock sampler's resident memory and the next sample() re-captures
(boltz_graph_patch.release_due / release_step_graph, boltz_dit_hoist.release_after_sample; the worker log carries `graph_released` /
`hoist_cache_released` per item, stack.evidence counts them, the LEVER lines say `release_min_tokens=… released=…`). CPU only."""
import importlib
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from .. import report as rep, stack  # noqa: E402

DIT_SRC = os.path.join(stack.kit_path("forward/dit_hoist"), "src")


def _fresh(name, monkeypatch, **env):
    for k in ("BOLTZ_GRAPH_RELEASE_MIN_TOKENS", "BOLTZ_GRAPH_DIFFUSION", "BOLTZ_DIT_HOIST"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if DIT_SRC not in sys.path:
        sys.path.insert(0, DIT_SRC)
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_the_release_rule_is_one_token_threshold(monkeypatch):
    bgp = _fresh("boltz_graph_patch", monkeypatch)
    assert bgp.RELEASE_MIN_TOKENS == 1024 and not bgp.release_due(1023) and bgp.release_due(1024) and bgp.release_due(2040)
    bgp = _fresh("boltz_graph_patch", monkeypatch, BOLTZ_GRAPH_RELEASE_MIN_TOKENS="0")
    assert bgp.RELEASE_MIN_TOKENS == 0 and not bgp.release_due(10 ** 6), "0 keeps the graph for the process at every size"
    bgp = _fresh("boltz_graph_patch", monkeypatch, BOLTZ_GRAPH_RELEASE_MIN_TOKENS="1")
    assert bgp.release_due(1) and bgp.release_due(199)
    assert "BOLTZ_GRAPH_RELEASE_MIN_TOKENS" .startswith(tuple(stack.load_pins()["stock_environment"]["must_be_absent_prefixes"])), "a caller's value is stripped: the process's default (or the row) rules"


def test_release_step_graph_drops_the_graph_once_and_counts(monkeypatch):
    bgp = _fresh("boltz_graph_patch", monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)          # CPU: the CUDA hygiene (sync, cuBLAS workspace map, empty_cache) is the GPU's
    diff = types.SimpleNamespace(_step_graph=types.SimpleNamespace(graph=object(), static={"dc": {}}))
    n0 = bgp.STATS["releases"]
    assert bgp.release_step_graph(diff) is True and diff._step_graph is None and bgp.STATS["releases"] == n0 + 1
    assert bgp.release_step_graph(diff) is False and bgp.STATS["releases"] == n0 + 1, "idempotent: nothing to drop the second time"
    assert bgp.release_step_graph(types.SimpleNamespace()) is False, "a module that never captured"


def test_the_hoist_releases_its_cache_with_the_graph_by_the_same_rule(monkeypatch):
    bgp = _fresh("boltz_graph_patch", monkeypatch, BOLTZ_GRAPH_RELEASE_MIN_TOKENS="10")
    dh = _fresh("boltz_dit_hoist", monkeypatch, BOLTZ_GRAPH_RELEASE_MIN_TOKENS="10")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _Net:                                                              # score_model: the cache store lives in its __dict__
        pass
    net = _Net(); net.__dict__["_dit_hoist_caches"] = {1: "cache", "_obj": "cache"}
    diff = types.SimpleNamespace(score_model=net, _step_graph=types.SimpleNamespace(graph=object(), static={"dc": {}}))
    n0, g0 = dh.STATS["cache_releases"], bgp.STATS["releases"]
    assert dh.release_after_sample(diff, 9) is False and "_dit_hoist_caches" in net.__dict__ and diff._step_graph is not None, "below the threshold: kept"
    assert dh.release_after_sample(diff, 10) is True
    assert "_dit_hoist_caches" not in net.__dict__ and diff._step_graph is None and dh.STATS["cache_releases"] == n0 + 1 and bgp.STATS["releases"] == g0 + 1
    assert dh.release_after_sample(diff, 10) is True and bgp.STATS["releases"] == g0 + 1, "the cache side is unconditional, the graph side idempotent"


def test_source_contract_release_sites_and_census_keys():
    """The graphed sampler releases at its own return, the hoist (outermost) in its finally after its census; the worker variant copies the
    census keys per item; stack.evidence counts them; the LEVER lines carry them."""
    g = open(os.path.join(DIT_SRC, "boltz_graph_patch.py")).read()
    tail = g[g.index("    STATS[\"sampler_s\"].append("):g.index("    return dict(sample_atom_coords=atom_coords")]
    assert 'n_tokens = int(network_condition_kwargs["s_trunk"].shape[1])' in tail and "released = release_step_graph(self) if release_due(n_tokens) else False" in tail
    assert '"release_min_tokens": RELEASE_MIN_TOKENS, "released": released,' in tail
    h = open(os.path.join(DIT_SRC, "boltz_dit_hoist.py")).read()
    fin = h[h.index("    finally:", h.index("def sample_hoisted(")):h.index("    return out", h.index("def sample_hoisted("))]
    assert fin.index('"cache_mb": cache_report(self)') < fin.index('STATS["last"]["cache_released"] = release_after_sample(self, int(network_condition_kwargs["s_trunk"].shape[1]))'), "release after the census"
    mv = open(os.path.join(DIT_SRC, "make_worker_variant.py")).read()
    assert '"release_min_tokens", "released", "headroom_gated"' in mv and '"cache_released", "headroom_gated"' in mv
    ev = {"graph_mode": "graph", "n_replay": 199, "n_items": 5, "release_min_tokens": 1024, "released": 5, "hoist_level": 2, "captured_with_cache": 5, "cache_released": 5}
    st, why, pairs = rep.lever_state("graph_sampler", ev, {"mode": "exact"})
    assert (st, why) == ("on", None) and pairs[:4] == [("n_replay", 199), ("items", 5), ("release_min_tokens", 1024), ("released", 5)] and ("headroom_gated", None) in pairs
    st, why, pairs = rep.lever_state("dit_hoist", ev, {"mode": "exact"})
    assert st == "on" and ("cache_released", 5) in pairs
    src = open(stack.__file__).read()
    assert 'ev["released"] = sum(1 for it in items if it.get("graph_released"))' in src and 'ev["cache_released"] = sum(1 for it in items if it.get("hoist_cache_released"))' in src
