"""CPU tests of lever denoiser_graph (no GPU, no weights): the call key, the size gate, the per-roll-out state machine with a stand-in for the CUDA
graph calls (capture / replay / recapture / storm / failed capture / lifetime), and the registry + installers coverage."""
import os

import pytest
import torch

from atlasfold_opt.hooks import denoiser_graph as DG
from atlasfold_opt.hooks.relpos import LazyFeat
from opt_core.counters import Ledger


# --------------------------------------------------------------------------------------------------------------- helpers
def _ledger():
    led = Ledger(DG.NAME, impl="fake", origin="kit", expected=DG.EXPECTED)
    for k in ("captures", "replays", "eager", "recaptures", "rollouts"):
        led.set(k, 0)
    return led


class FakeGraphs:
    """Stand-in for DG.CudaGraphs on CPU tensors: capture runs fn eagerly and returns a graph whose replay() recomputes the static output in place."""
    impl = "fake"

    def __init__(self):
        self.capturing = False
        self.raise_on_warmup = None                                              # the warm-up call raises (device state intact)
        self.raise_on_capture = None                                             # the capture raises (DG.CaptureAborted: state not restorable)
        self.captures = 0
        self.replays = 0

    def is_capturing(self):
        return self.capturing

    def capture(self, fn, device=None):
        if self.raise_on_warmup is not None:
            raise self.raise_on_warmup
        fn()                                                                      # the warm-up call (the real backend runs it on the side stream)
        if self.raise_on_capture is not None:
            raise DG.CaptureAborted(self.raise_on_capture)
        out = fn()
        self.captures += 1
        backend = self

        class G:
            def replay(self_inner):
                backend.replays += 1
                out.copy_(fn())
        return G(), out


class TinyDenoiser(torch.nn.Module):
    """forward(batch, r_noisy, single_cond, pair_bias) with the stock signature: a deterministic function of all four arguments."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(3, 3))
        self.calls = 0

    def forward(self, batch, r_noisy, single_cond, pair_bias):
        self.calls += 1
        m = batch["atom14_mask"].to(r_noisy.dtype)[:, None, :, :, None]                 # [B,1,L,14,1]
        return (r_noisy @ self.w) * m + single_cond.mean() + pair_bias.sum() * 0.0 + batch["nested"]["k"][0].sum() * 0.0


def _args(B=1, N=2, L=8, c=4):
    batch = {"atom14_mask": torch.ones(B, L, 14, dtype=torch.bool), "seq_mask": torch.ones(B, L, dtype=torch.bool),
             "nested": {"k": [torch.arange(L)]}, "n_int": 3, "name": "x"}
    return batch, torch.randn(B, N, L, 14, 3), torch.randn(B, 1, L, c), torch.randn(2, B, L, L, 4)


# --------------------------------------------------------------------------------------------------------------- the key
def test_key_same_tensors_same_key_and_refs_held():
    batch, r, sc, pb = _args()
    refs1, refs2 = [], []
    k1 = DG.call_key(batch, r, sc, pb, refs1)
    k2 = DG.call_key(batch, torch.randn_like(r), sc.clone(), pb, refs2)             # r / single_cond are keyed by layout only (copied, not referenced)
    assert k1 == k2 and hash(k1) == hash(k2)
    held = {id(x) for x in refs1}
    for t in (batch["atom14_mask"], batch["seq_mask"], batch["nested"]["k"][0], pb):
        assert id(t) in held
    assert id(r) not in held and id(sc) not in held


def test_key_changes_on_reallocation_stride_dtype_autocast_and_structure():
    batch, r, sc, pb = _args()
    k0 = DG.call_key(batch, r, sc, pb, [])
    assert DG.call_key(dict(batch, seq_mask=batch["seq_mask"].clone()), r, sc, pb, []) != k0          # same values, another address
    assert DG.call_key(batch, r, sc, pb.clone(), []) != k0                                            # pair_bias is held by reference
    assert DG.call_key(batch, r.transpose(3, 4).contiguous().transpose(3, 4), sc, pb, []) != k0        # r_noisy stride
    assert DG.call_key(batch, r[:, :1], sc, pb, []) != k0                                             # r_noisy shape (a ragged chunk)
    assert DG.call_key(batch, r, sc.double(), pb, []) != k0                                           # single_cond dtype
    assert DG.call_key(dict(batch, n_int=4), r, sc, pb, []) != k0                                     # a scalar in the batch
    b2 = dict(batch); b2["nested"] = {"k": [batch["nested"]["k"][0], torch.zeros(1)]}
    assert DG.call_key(b2, r, sc, pb, []) != k0                                                       # container structure
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        assert DG.call_key(batch, r, sc, pb, []) != k0                                                # autocast state
    assert DG.call_key(batch, r, sc, pb, []) == k0


def test_key_flattens_lazyfeat_closure_and_holds_it():
    batch, r, sc, pb = _args()
    keys_a = {"res_idx": torch.arange(8).unsqueeze(0), "aatype": torch.zeros(1, 8, 21)}
    enc = lambda d: d["res_idx"].float()                                        # noqa: E731 — stands in for the producer module
    lf = LazyFeat(lambda: enc(keys_a), device_type="cpu")
    b1 = dict(batch, atom_rel_pos=lf)
    refs = []
    k1 = DG.call_key(b1, r, sc, pb, refs)
    held = {id(x) for x in refs}
    assert id(lf) in held and id(keys_a["res_idx"]) in held and id(keys_a["aatype"]) in held
    assert DG.call_key(b1, r, sc, pb, []) == k1
    keys_b = {"res_idx": keys_a["res_idx"].clone(), "aatype": keys_a["aatype"]}    # a producer input re-allocated -> another key
    b2 = dict(batch, atom_rel_pos=LazyFeat(lambda: enc(keys_b), device_type="cpu"))
    assert DG.call_key(b2, r, sc, pb, []) != k1


# --------------------------------------------------------------------------------------------------------------- the size gate
def test_gate_word_and_env_knob(monkeypatch):
    r = torch.zeros(1, 2, 300, 14, 3)
    assert DG.gate_word(r, 1024) == DG.CPU                                       # no CUDA tensor -> cpu (before the size test)
    monkeypatch.delenv("AFO_DENOISER_GRAPH_MAX_TOKENS", raising=False)
    assert DG.max_tokens() == 1024
    monkeypatch.setenv("AFO_DENOISER_GRAPH_MAX_TOKENS", "256")
    assert DG.max_tokens() == 256
    fake_cuda = torch.zeros(1, 2, 300, 14, 3)
    class Dev:                                                                    # gate_word reads .device.type and .shape only
        type = "cuda"
    class R:
        device = Dev(); shape = fake_cuda.shape
    assert DG.gate_word(R(), 1024) is None and DG.gate_word(R(), 299) == DG.ABOVE_GATE and DG.gate_word(R(), 300) is None


# --------------------------------------------------------------------------------------------------------------- the state machine (stand-in graphs, CPU tensors)
def _rollout(limit=1 << 30):
    led = _ledger(); fake = FakeGraphs()
    ro = DG.RollOut(led, limit, graphs=fake)
    return ro, led, fake


def _no_cpu_gate(monkeypatch):
    monkeypatch.setattr(DG, "gate_word", lambda r, limit: DG.ABOVE_GATE if int(r.shape[2]) > int(limit) else None)


def test_first_call_captures_and_replays_then_replays_bitwise(monkeypatch):
    _no_cpu_gate(monkeypatch)
    torch.manual_seed(0)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout()
    batch, r, sc, pb = _args()
    with torch.no_grad():
        for step in range(5):
            r_t, sc_t = torch.randn_like(r), torch.randn_like(sc)
            want = stock(mod, batch, r_t, sc_t, pb)
            got = ro.call(stock, mod, batch, r_t, sc_t, pb)
            assert torch.equal(got, want), step
    assert fake.captures == 1 and fake.replays == 5 and led.served == 5
    f = led.facts()
    assert (f["captures"], f["replays"], f["eager"], f["recaptures"]) == (1, 5, 0, 0)
    assert led.fallbacks == {} and led.gate().ok
    assert len(ro.graphs) == 1
    ent = next(iter(ro.graphs.values()))
    assert ent.r is not r and ent.r.stride() == r.stride() and any(x is pb for x in ent.refs)


def test_ragged_chunks_capture_one_graph_per_shape_without_storm(monkeypatch):
    _no_cpu_gate(monkeypatch)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout()
    batch, r, sc, pb = _args(N=5)
    with torch.no_grad():
        for step in range(4):
            r_t = torch.randn_like(r)
            for st in (0, 2, 4):                                                  # chunk_size 2 over 5 samples: shapes 2, 2, 1
                chunk = r_t[:, st:st + 2]
                assert torch.equal(ro.call(stock, mod, batch, chunk, sc, pb), stock(mod, batch, chunk, sc, pb))
    f = led.facts()
    assert f["captures"] == 2 and f["recaptures"] == 0 and f["eager"] == 0 and f["replays"] == 12 and len(ro.graphs) == 2
    assert led.gate().ok and ro.dead is None


def test_recapture_storm_trips_to_named_eager(monkeypatch):
    _no_cpu_gate(monkeypatch)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout()
    batch, r, sc, pb = _args()
    with torch.no_grad():
        ro.call(stock, mod, batch, r, sc, pb)                                     # capture 1
        ro.call(stock, mod, batch, r, sc, pb.clone())                             # pair_bias moved: capture 2 (one recapture is tolerated)
        assert led.facts()["recaptures"] == 1 and ro.dead is None
        out = ro.call(stock, mod, batch, r, sc, pb.clone())                       # third capture of ONE shape -> storm
        assert torch.equal(out, stock(mod, batch, r, sc, pb))                     # (pair_bias enters TinyDenoiser as * 0.0)
        ro.call(stock, mod, batch, r, sc, pb)                                     # the rest of the roll-out is eager under the same word
    assert ro.dead == DG.RECAPTURE_STORM and led.fallbacks == {DG.RECAPTURE_STORM: 2} and len(ro.graphs) == 0
    assert led.facts()["eager"] == 2 and led.facts()["captures"] == 2
    g = led.gate()
    assert not g.ok and "recapture_storm" in g.reason                             # not an expected word: fail-loud


def test_warmup_exception_goes_eager_by_name(monkeypatch):
    _no_cpu_gate(monkeypatch)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout()
    fake.raise_on_warmup = RuntimeError("lazy init failed")
    batch, r, sc, pb = _args()
    with torch.no_grad():
        for _ in range(3):
            assert torch.equal(ro.call(stock, mod, batch, r, sc, pb), stock(mod, batch, r, sc, pb))
    assert ro.dead == "warmup_failed:RuntimeError" and led.fallbacks == {"warmup_failed:RuntimeError": 3} and not ro.aborted
    assert led.facts()["eager"] == 3 and led.facts()["captures"] == 0 and "capture_error" in led.facts() and " " not in led.facts()["capture_error"]
    assert not led.gate().ok


def test_capture_exception_is_counted_named_and_terminal(monkeypatch):
    _no_cpu_gate(monkeypatch)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout()
    fake.raise_on_capture = ValueError("operation not permitted when stream is capturing")
    batch, r, sc, pb = _args()
    with torch.no_grad(), pytest.raises(RuntimeError, match="cannot continue after a failed stream capture") as ei:
        ro.call(stock, mod, batch, r, sc, pb)
    assert isinstance(ei.value.__cause__, ValueError)                            # the ORIGINAL error is the cause
    assert ro.dead == "capture_failed:ValueError" and ro.aborted and led.fallbacks == {"capture_failed:ValueError": 1}
    assert led.facts()["captures"] == 0 and led.facts()["eager"] == 0 and led.facts()["capture_error"].startswith("ValueError:")
    assert not led.gate().ok and not led.gate(require_served=False).ok           # not an expected word under either gate form
    ro.close()
    assert ro.closed and len(ro.graphs) == 0


def test_above_gate_and_cpu_are_expected_eager_words(monkeypatch):
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    batch, r, sc, pb = _args(L=8)
    ro, led, fake = _rollout(limit=4)                                             # real gate_word: CPU tensors -> `cpu` first
    with torch.no_grad():
        ro.call(stock, mod, batch, r, sc, pb)
    assert led.fallbacks == {"cpu": 1} and fake.captures == 0
    _no_cpu_gate(monkeypatch)
    ro2, led2, fake2 = _rollout(limit=4)                                          # L=8 > 4 -> above_gate, no capture, no allocation
    with torch.no_grad():
        for _ in range(3):
            ro2.call(stock, mod, batch, r, sc, pb)
    assert led2.fallbacks == {"above_gate": 3} and fake2.captures == 0 and led2.served == 0 and led2.facts()["eager"] == 3
    assert led2.gate(require_served=False).ok                                    # hooks.size_gated: a run wholly above the gate is a legitimate run
    assert not led2.gate().ok                                                    # (the plain core gate would say `3 calls, 0 served`)


def test_capturing_guard_runs_inline(monkeypatch):
    _no_cpu_gate(monkeypatch)
    mod = TinyDenoiser(); stock = TinyDenoiser.forward
    ro, led, fake = _rollout(); fake.capturing = True
    batch, r, sc, pb = _args()
    with torch.no_grad():
        ro.call(stock, mod, batch, r, sc, pb)
    assert led.fallbacks == {"capturing": 1} and fake.captures == 0 and not led.gate().ok


# --------------------------------------------------------------------------------------------------------------- install(): class rebinding + lifetime on the stock classes
def _tiny_forward(self, batch, r_noisy, single_cond, pair_bias):
    return TinyDenoiser.forward(self, batch, r_noisy, single_cond, pair_bias)


def _install_over_fakes(monkeypatch, inner_sample):
    """install() against the stock diffusion_head classes whose two attributes were first replaced by CPU stand-ins (the lever wraps whatever it
    finds at install time), with the stand-in graph calls and the cpu gate lifted; monkeypatch restores the true stock attributes at teardown."""
    import atlasfold.model.network.diffusion_head as DH
    monkeypatch.setattr(DG, "CudaGraphs", FakeGraphs())
    _no_cpu_gate(monkeypatch)
    monkeypatch.setattr(DH.DiffusionModule, "forward", _tiny_forward)
    monkeypatch.setattr(DH.DiffusionHead, "sample", inner_sample)
    ins = DG.install("fast", "atlasfold-opt", {})
    assert ins.applied, ins.reason
    assert DH.DiffusionModule.forward.__wrapped_stock__ is _tiny_forward and DH.DiffusionHead.sample.__wrapped_stock__ is inner_sample
    assert DH.DiffusionModule.forward.__qualname__.endswith("[atlasfold_opt:denoiser_graph]")
    assert DH.DiffusionHead.sample.__qualname__.endswith("[atlasfold_opt:denoiser_graph]")

    class SM(TinyDenoiser):
        forward = DH.DiffusionModule.forward                                     # the rebound class attribute, as the stock DiffusionModule carries it

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score_model = SM()

        def sample(self, *a, **k):
            return DH.DiffusionHead.sample(self, *a, **k)
    return ins, Head()


def test_install_scopes_graphs_to_one_sample_call(monkeypatch):
    seen = []

    def inner_sample(self, batch, r, sc, pb, steps=3, raise_at=None):            # what DiffusionHead.sample was before install (stock, or diffusion_bf16's wrapper)
        ro = getattr(self.score_model, DG.STATE_ATTR)
        assert isinstance(ro, DG.RollOut) and not ro.closed
        out = None
        for i in range(steps):
            if raise_at == i:
                raise RuntimeError("sampler failed")
            out = self.score_model(batch, torch.randn_like(r), sc, pb)
            seen.append(len(ro.graphs))
        return out
    ins, head = _install_over_fakes(monkeypatch, inner_sample)
    led = ins.facts["ledger"]
    batch, r, sc, pb = _args()
    with torch.no_grad():
        head.sample(batch, r, sc, pb, steps=3)
    assert getattr(head.score_model, DG.STATE_ATTR) is None                      # dropped at sample() exit
    f = led.facts()
    assert seen == [1, 1, 1] and (f["captures"], f["replays"], f["eager"], f["recaptures"], f["rollouts"]) == (1, 3, 0, 0, 1)
    with torch.no_grad():                                                        # a forward outside any sample(): named eager, no state created
        head.score_model(batch, r, sc, pb)
    assert led.fallbacks == {"outside_sample": 1} and getattr(head.score_model, DG.STATE_ATTR) is None and led.facts()["eager"] == 1
    led.clear(); seen.clear()
    with torch.no_grad():                                                        # every roll-out captures afresh: no cross-roll-out graph cache
        head.sample(batch, r, sc, pb, steps=2); head.sample(batch, r, sc, pb, steps=2)
    f = led.facts()
    assert (f["captures"], f["rollouts"], f["replays"]) == (2, 2, 4) and led.gate().ok


def test_sample_that_raises_still_drops_the_graphs(monkeypatch):
    holder = {}

    def inner_sample(self, batch, r, sc, pb, steps=3, raise_at=1):
        holder["ro"] = getattr(self.score_model, DG.STATE_ATTR)
        for i in range(steps):
            if i == raise_at:
                raise RuntimeError("sampler failed")
            self.score_model(batch, torch.randn_like(r), sc, pb)
    ins, head = _install_over_fakes(monkeypatch, inner_sample)
    batch, r, sc, pb = _args()
    with torch.no_grad(), pytest.raises(RuntimeError, match="sampler failed"):
        head.sample(batch, r, sc, pb)
    ro = holder["ro"]
    assert getattr(head.score_model, DG.STATE_ATTR) is None and ro.closed and len(ro.graphs) == 0 and ins.facts["ledger"].facts()["captures"] == 1


def test_lever_line_fields(monkeypatch):
    ins, head = _install_over_fakes(monkeypatch, lambda self, *a, **k: None)
    line = ins.lines[0]()
    for word in ("name=LOCAL.atlasfold.denoiser_graph", "captures=0", "replays=0", "eager=0", "recaptures=0", "max_tokens=%d" % DG.max_tokens()):
        assert word in line, (word, line)
    assert ins.gates[0]().ok                                                     # no calls: ok


# --------------------------------------------------------------------------------------------------------------- registry / row coverage
def test_registry_row_and_installer():
    from atlasfold_opt.registry import LEVERS
    from atlasfold_opt.hooks import installers
    from atlasfold_opt.modes import MODES, FAST, EXACT
    row = LEVERS["denoiser_graph"]
    assert row["cls"] == "fast" and row["strategy"] == DG.NAME and tuple(row["expected"]) == DG.EXPECTED == ("above_gate", "cpu")
    assert installers()["denoiser_graph"] is DG.install
    assert "denoiser_graph" in MODES[FAST] and "denoiser_graph" not in MODES[EXACT]
    assert MODES[FAST].index("denoiser_graph") > MODES[FAST].index("diffusion_bf16")   # composes over diffusion_bf16's sample wrapper (autocast outermost)


def test_static_copy_keeps_layout():
    t = torch.randn(2, 5, 8, 14, 3)[:, 1:3]                                       # a non-dense chunk view (B=2)
    s = DG.static_copy(t)
    assert s.stride() == t.stride() and s.shape == t.shape and torch.equal(s, t) and s.data_ptr() != t.data_ptr()
    e = torch.randn(1, 3).expand(4, 3)                                            # overlapping -> dense copy
    se = DG.static_copy(e)
    assert torch.equal(se, e) and 0 not in se.stride()
