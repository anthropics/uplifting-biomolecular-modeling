"""The sampler levers' memory-headroom gate (graph_sampler + dit_hoist): before a sample() builds any of their state, the levers' projected
working set — the hoist's cache (boltz_dit_hoist.projected_cache_bytes: _build_cache's terms by shape), the step graph's static clones
(boltz_graph_patch.projected_static_bytes), the eager step's transients twice — plus a margin of the card must fit in the free memory; when it
does not, that prediction samples on the stock path (the same arithmetic the graph replays: same bytes), counted `headroom_gated` per item, exempt
from the replay / captured-with-cache evidence, on a GATE line, on the LEVER lines and on EXIT (`levers_headroom_gated=`). CPU only."""
import importlib
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from .. import report as rep, stack  # noqa: E402

DIT_SRC = os.path.join(stack.kit_path("forward/dit_hoist"), "src")


def _mods(monkeypatch, **env):
    for k in ("BOLTZ_GRAPH_HEADROOM_MARGIN", "BOLTZ_GRAPH_RELEASE_MIN_TOKENS", "BOLTZ_GRAPH_DIFFUSION", "BOLTZ_DIT_HOIST"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if DIT_SRC not in sys.path:
        sys.path.insert(0, DIT_SRC)
    for n in ("boltz_graph_patch", "boltz_dit_hoist"):
        sys.modules.pop(n, None)
    return importlib.import_module("boltz_graph_patch"), importlib.import_module("boltz_dit_hoist")


def _card(monkeypatch, free_gib, total_gib=79.1, reserved_gib=0.0, allocated_gib=0.0):
    G = 2 ** 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(free_gib * G), int(total_gib * G)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: int(reserved_gib * G))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(allocated_gib * G))


def test_the_gate_is_projection_plus_two_transients_plus_a_margin_of_the_card_against_free(monkeypatch):
    bgp, _ = _mods(monkeypatch)
    assert bgp.HEADROOM_MARGIN_FRAC == 0.12
    G = 2 ** 30
    _card(monkeypatch, free_gib=60.0, total_gib=80.0, reserved_gib=10.0, allocated_gib=6.0)    # free = 60 (driver) + 4 (the allocator's unused reserve) = 64
    g = bgp.headroom(40 * G, 5 * G)                                                             # need = 40 + 2x5 + 9.6 = 59.6 <= 64
    assert g["gated"] is False and (g["projected_gib"], g["transient_gib"], g["margin_gib"], g["need_gib"], g["free_gib"], g["total_gib"]) == (40.0, 5.0, 9.6, 59.6, 64.0, 80.0)
    g = bgp.headroom(45 * G, 5 * G)                                                             # need = 64.6 > 64
    assert g["gated"] is True and g["need_gib"] == 64.6
    bgp2, _ = _mods(monkeypatch, BOLTZ_GRAPH_HEADROOM_MARGIN="0")
    _card(monkeypatch, free_gib=60.0, total_gib=80.0, reserved_gib=10.0, allocated_gib=6.0)
    assert bgp2.headroom(54 * G, 5 * G)["gated"] is False and bgp2.headroom(54 * G + 1, 5 * G)["gated"] is True
    assert "BOLTZ_GRAPH_" in stack.load_pins()["stock_environment"]["must_be_absent_prefixes"]


def test_step_transients_and_static_clones_by_shape(monkeypatch):
    bgp, _ = _mods(monkeypatch)
    nck = {"s_trunk": torch.zeros(1, 100, 384), "s_inputs": torch.zeros(1, 100, 384, dtype=torch.bfloat16),
           "diffusion_conditioning": {"token_trans_bias": torch.zeros(1, 100, 100, 384, dtype=torch.bfloat16), "to_keys": (lambda x: x), "q": torch.zeros(1, 800, 128)},
           "feats": {"atom_pad_mask": torch.ones(1, 800)}}
    assert bgp.step_transient_bytes(nck, 5, heads=16) == 3 * 1 * 5 * 16 * 100 * 100 * 4
    want = 100 * 384 * 4 + 100 * 384 * 2 + 100 * 100 * 384 * 2 + 800 * 128 * 4 + 5 * 1 * 5 * 800 * 3 * 4
    assert bgp.projected_static_bytes(nck, 5, 800) == want


class _Lin(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.l = torch.nn.Linear(i, o)


def _net(L_tok=24, L_atom=3, heads_tok=None, W=32, H=128):
    """The DiffusionModule attributes projected_cache_bytes reads, at Boltz-2's shapes: proj_z a rearrangement (no Linear) unless heads given."""
    def apb(heads):
        return types.SimpleNamespace(proj_z=(torch.nn.Linear(16, heads) if heads else torch.nn.Identity()), num_heads=16)
    def atom_layer():
        ad = lambda: types.SimpleNamespace(s_scale=torch.nn.Linear(128, 128), s_bias=torch.nn.Linear(128, 128))  # noqa: E731
        return types.SimpleNamespace(adaln=ad(), transition=types.SimpleNamespace(adaln=ad(), output_projection=torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Sigmoid())),
                                     output_projection=torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Sigmoid()), pair_bias_attn=apb(None))
    def atf():
        dtr = types.SimpleNamespace(layers=[atom_layer() for _ in range(L_atom)])
        return types.SimpleNamespace(attn_window_queries=W, attn_window_keys=H, diffusion_transformer=dtr)
    tok = types.SimpleNamespace(layers=[types.SimpleNamespace(pair_bias_attn=apb(heads_tok)) for _ in range(L_tok)])
    return types.SimpleNamespace(single_conditioner=types.SimpleNamespace(single_embed=torch.nn.Linear(768, 768)), token_transformer=tok,
                                 atom_attention_encoder=types.SimpleNamespace(atom_encoder=atf()), atom_attention_decoder=types.SimpleNamespace(atom_decoder=atf()))


def _nck(B=1, N=64, A=256, W=32, H=128, tok_bias_dtype=torch.bfloat16):
    dc = {"q": torch.zeros(B, A, 128), "c": torch.zeros(B, A, 128), "atom_enc_bias": torch.zeros(B, A // W, W, H, 12), "atom_dec_bias": torch.zeros(B, A // W, W, H, 12),
          "token_trans_bias": torch.zeros(B, N, N, 384, dtype=tok_bias_dtype), "to_keys": (lambda x: x)}
    return {"s_trunk": torch.zeros(B, N, 384), "s_inputs": torch.zeros(B, N, 384), "diffusion_conditioning": dc,
            "feats": {"token_pad_mask": torch.ones(B, N), "atom_pad_mask": torch.ones(B, A)}}


def test_the_cache_projection_is_build_caches_terms_by_shape(monkeypatch):
    """Level 2 at Boltz-2's shapes (proj_z a rearrangement, sampling with autocast off: every Linear output fp32): the dominant term is the
    token layers' expanded bias, m x numel(token_trans_bias) x 4 — 24 x [B*m, 16, N, N] fp32 — then the fp32 cast of a bf16 bias, the atom terms."""
    bgp, dh = _mods(monkeypatch)
    monkeypatch.setattr(dh, "_atom_layers", lambda net: [(p, i, atf, layer) for p, atf in (("enc", net.atom_attention_encoder.atom_encoder), ("dec", net.atom_attention_decoder.atom_decoder))
                                                       for i, layer in enumerate(atf.diffusion_transformer.layers)])
    dh.STATS["level"] = 2
    B, N, A, W, H, m = 1, 64, 256, 32, 128, 5
    net, nck = _net(), _nck(B, N, A, W, H)
    got = dh.projected_cache_bytes(net, nck, m)
    f4 = 4
    dcf = N * N * 384 * f4                                            # only token_trans_bias is not fp32 here
    s0 = B * m * N * 768 * f4
    c_m = A * 128 * m * f4
    rows = A * m
    adaln = 6 * (2 * rows * (128 + 128) * f4 + rows * (128 + 128) * f4)
    tok = 2 * B * N * m * f4 + N * N * 384 * m * f4
    nw = A * m // W
    per_atf = (B * (A // W) * W * H * 12) * m * f4
    atom = 2 * (per_atf + A * m * f4 + 2 * nw * H * f4 + per_atf)
    assert got == dcf + s0 + c_m + adaln + tok + atom, (got, dcf + s0 + c_m + adaln + tok + atom)
    dh.STATS["level"] = 1
    assert dh.projected_cache_bytes(net, nck, m) == dcf + s0 + c_m + adaln
    dh.STATS["level"] = 2
    big = dh.projected_cache_bytes(_net(), _nck(1, 2040, 15616, W, H), 5)     # an 8on7-class input: ~2,040 tokens, ~15.6k atoms, 5 samples
    assert 39e9 < big < 48e9, big                                              # 24 x [5, 16, 2040, 2040] fp32 = 32 GB of it


def test_a_gated_sample_runs_the_stock_sampler_counts_itself_and_frees_the_previous_predictions_state(monkeypatch):
    """sample_hoisted with the gate saying no: the hoist cache and the step graph a previous prediction left are dropped, AtomDiffusion.sample's
    original (the graph patch keeps it as _stock_sample) runs with the caller's arguments, both modules' per-call census say headroom_gated, no
    cache generation was opened (the denoiser calls take _stock_forward)."""
    bgp, dh = _mods(monkeypatch)
    dh.STATS["level"] = 2; bgp.STATS["mode"] = "graph"
    gate = {"gated": True, "projected_gib": 50.0, "projected_cache_gib": 45.0, "transient_gib": 4.0, "margin_gib": 6.3, "need_gib": 64.3, "free_gib": 60.0}
    monkeypatch.setattr(dh, "headroom_gate", lambda self, nck, m: dict(gate))
    released = []
    monkeypatch.setattr(bgp, "release_step_graph", lambda diff: released.append(diff) or True)
    calls = []

    class _Diff:
        training = False
        _dit_hoist_inner_sample = staticmethod(lambda self, *a, **k: calls.append(("inner", a, k)) or "inner-out")

        def __init__(self):
            self.score_model = types.SimpleNamespace(); self.score_model.__dict__["_dit_hoist_caches"] = {"x": object()}
            self._stock_sample = lambda *a, **k: calls.append(("stock", a, k)) or {"sample_atom_coords": "stock-out"}

    d = _Diff(); nck = _nck(1, 64, 256)
    gen0 = list(dh._GEN)
    out = dh.sample_hoisted(d, "mask", num_sampling_steps=200, multiplicity=5, max_parallel_samples=5,
                           steering_args={"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": True}, **nck)
    assert out == {"sample_atom_coords": "stock-out"} and [c[0] for c in calls] == ["stock"], "the stock sampler ran, not the inner (graphed) one"
    (kind, args, kw), = calls
    assert args == ("mask",) and kw["num_sampling_steps"] == 200 and kw["multiplicity"] == 5 and kw["s_trunk"] is nck["s_trunk"]
    assert released == [d] and "_dit_hoist_caches" not in d.score_model.__dict__ and dh._GEN == gen0 and dh._CUR["gen"] is None
    assert dh.STATS["headroom_gated"] == 1 and dh.STATS["last"]["headroom_gated"] is True and dh.STATS["last"]["cache_mb"] == 0.0 and dh.STATS["last"]["projected_cache_gib"] == 45.0
    assert bgp.STATS["headroom_gated"] == 1 and bgp.STATS["last"]["sampler_mode"] == "eager_headroom" and bgp.STATS["last"]["n_replay"] == 0 and bgp.STATS["last"]["headroom_gated"] is True
    assert (bgp.STATS["last"]["projected_gib"], bgp.STATS["last"]["free_gib"], bgp.STATS["last"]["n_tokens"]) == (50.0, 60.0, 64)


def test_source_contract_gate_before_state_and_census_plumbing():
    h = open(os.path.join(DIT_SRC, "boltz_dit_hoist.py")).read()
    body = h[h.index("def sample_hoisted("):]
    assert body.index("gate = headroom_gate(self, network_condition_kwargs, multiplicity)") < body.index("_GEN[0] += 1") < body.index("_ARMED.add(self)"), "the gate decides before any lever state of this sample() exists"
    g = open(os.path.join(DIT_SRC, "boltz_graph_patch.py")).read()
    assert 'HEADROOM_MARGIN_FRAC = float(os.environ.get("BOLTZ_GRAPH_HEADROOM_MARGIN", "0.12"))' in g and "need = float(projected_bytes) + 2.0 * float(transient_bytes) + margin" in g
    mv = open(os.path.join(DIT_SRC, "make_worker_variant.py")).read()
    assert '"headroom_gated", "projected_gib", "transient_gib", "need_gib", "free_gib"' in mv and '"headroom_gated", "projected_cache_gib"' in mv
    src = open(stack.__file__).read()
    assert 'gated = [it for it in items if it.get("graph_headroom_gated") or it.get("hoist_headroom_gated")]' in src and "it not in gated and" in src


def _wlog(items, S=200):
    per = []
    for name, gated in items:
        it = {"name": name, "seed": 0, "graph_release_min_tokens": 1024}
        if gated:
            it.update(graph_sampler_mode="eager_headroom", graph_n_replay=0, graph_headroom_gated=True, graph_projected_gib=50.1, graph_free_gib=61.0,
                      hoist_hoist_level=2, hoist_captured_with_cache=0, hoist_capture_stock_fallbacks=0, hoist_headroom_gated=True)
        else:
            it.update(graph_sampler_mode="graph", graph_n_replay=S - 1, graph_headroom_gated=False, graph_projected_gib=3.0, graph_free_gib=70.0,
                      hoist_hoist_level=2, hoist_captured_with_cache=1, hoist_capture_stock_fallbacks=0, hoist_headroom_gated=False)
        per.append(it)
    return per


def test_evidence_exempts_gated_items_and_the_lines_name_them(monkeypatch):
    """stack.evidence: a gated item (eager_headroom, no replay, nothing captured) is the lever's own rule — counted headroom_gated, not a problem;
    an ungated item still owes n_replay = S-1 and captured_with_cache >= 1. partial_activation records the GATE; the LEVER lines carry the count
    (skipped memory_headroom when every prediction was gated); the EXIT line carries levers_headroom_gated."""
    from .test_cli_and_evidence import _log
    kon = _wlog([("a", False), ("b", True)])
    base = _log(["resid", "mask2"], mode="exact")                                              # the exact row's trunk levers, two items (test_cli_and_evidence._log)
    for it, mine in zip(base["per_item"], kon):
        it.update(mine)
    ev, problems = stack.evidence("exact", base, sampling_steps=200)
    assert problems == [] and ev["headroom_gated"] == 1 and ev["headroom_last"] == {"projected_gib": 50.1, "free_gib": 61.0}
    bad = _log(base["env"]["boltz_levers"], mode="exact")
    for it, mine in zip(bad["per_item"], _wlog([("a", False), ("b", False)])):
        it.update(mine)
    bad["sampler_report"]["module"]["stats"]["replays"] = 199                                 # an UNGATED item that did not replay is still a problem (the roll-out replays S-1 boundaries per ungated prediction)
    ev2, problems2 = stack.evidence("exact", bad, sampling_steps=200)
    assert ev2["headroom_gated"] == 0 and any("roll-out: captures=2 replays=199 for 2 item(s)" in p for p in problems2), problems2
    fallbacks, gates = stack.partial_activation("exact", ev)
    assert fallbacks == [] and list(gates) == ["rollout,dit_hoist"] and gates["rollout,dit_hoist"].startswith("memory_headroom: 1/2 predictions sampled on the stock path")
    st, why, pairs = rep.lever_state("dit_hoist", ev, {"mode": "exact"})
    assert (st, why) == ("on", None) and ("headroom_gated", 1) in pairs
    allg = dict(ev, headroom_gated=2, n_items=2)
    assert rep.lever_state("dit_hoist", allg, {"mode": "exact"})[:2] == ("skipped", "memory_headroom"), "every prediction gated: the hoist reports skipped by the gate's name"
    rep.reset_tally(); rep.arm_tally("exact", "worker"); rep.count(ok=2); rep.set_rc(0); rep.tally_headroom(ev["headroom_gated"])
    assert rep.tally_line() == "[boltz2-opt] EXIT mode=exact route=worker n_gpu=1 sharding=none predictions=2 ok=2 failed=0 rc=0 levers_headroom_gated=rollout,dit_hoist:1"
    rep.reset_tally()
