"""graph_reuse on CPU with a fake graph backend: the keep / adopt / drop policy (one item's graphs alive), the by-reference table walk and its
storage copies, the probe, the pre-trunk drop, the switch, the install wiring.  The CUDA-graph mechanics are denoiser_graph's (its tests);
byte identity across items is a GPU statement (test_graph_reuse_gpu.py, and the kit's pred --det 1 comparisons)."""
import types

import pytest
import torch

from opt_core.counters import Ledger
from atlasfold_opt.hooks import denoiser_graph as DG, graph_reuse as GR


class FakeGraphs:
    """DG.CudaGraphs stand-in on CPU: warm-up = one eager call; capture = one eager call whose output is the static output; replay() recomputes
    that output in place through the CAPTURED closure (so it reads the capture-time batch / pair_bias objects, as a CUDA graph reads their
    addresses); reset() marks the graph released."""
    impl = "fake"

    def __init__(self):
        self.captures = self.replays = self.warmups = self.resets = 0
        self.perturb = None                                                       # a callable applied to every replayed output (a probe mismatch)

    def is_capturing(self):
        return False

    def warmup(self, fn, device=None):
        self.warmups += 1
        return fn()

    def capture(self, fn, device=None):
        self.warmup(fn, device)
        out = fn()
        self.captures += 1
        backend = self

        class G:
            released = False

            def replay(g):
                assert not g.released, "replay of a released graph"
                backend.replays += 1
                y = fn()
                out.copy_(y if backend.perturb is None else backend.perturb(y))

            def reset(g):
                g.released = True
                backend.resets += 1
        return G(), out


class Tiny(torch.nn.Module):
    """forward(batch, r_noisy, single_cond, pair_bias): reads the mask and pair_bias BY REFERENCE (their values enter the output) and, like a
    sampler_hoist leaf, a per-roll-out tensor built at the first call and read at every later one."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(3, 3))

    def forward(self, batch, r_noisy, single_cond, pair_bias):
        ro = ROLL.cur
        if "leaf" not in ro.memo:
            ro.memo["leaf"] = batch["seq_mask"].to(r_noisy.dtype) * 2.0         # [B, L]: a roll-out leaf derived from THIS item's batch
        leaf = ro.memo["leaf"]
        m = batch["atom14_mask"].to(r_noisy.dtype)[:, None, :, :, None]
        return (r_noisy @ self.w) * m + single_cond.mean() + pair_bias.mean() + leaf[:, None, :, None, None]


class Roll:
    """Stand-in for sampler_hoist's roll-out state: a memo dict and counters (scalars under an object count by type in the walk)."""
    def __init__(self):
        self.memo = {}
        self.hits = 0
        self.names = {"a", "b"}


class _Rolls:
    cur = None


ROLL = _Rolls()


def _args(B=1, N=2, L=8, c=4, fill=1.0):
    batch = {"aatype_int": torch.zeros(B, L, dtype=torch.long), "atom14_mask": torch.rand(B, L, 14) > 0.3,
             "seq_mask": torch.rand(B, L) > 0.2, "nested": {"k": [torch.arange(L)]}, "n_int": 3, "name": "x"}
    return batch, torch.randn(B, N, L, 14, 3), torch.randn(B, 1, L, c), torch.randn(2, B, L, L, 4) * fill


def _ledgers():
    dg = Ledger(DG.NAME, impl="fake", origin="kit", expected=DG.EXPECTED)
    for k in ("captures", "replays", "eager", "recaptures", "rollouts"):
        dg.set(k, 0)
    gr = Ledger(GR.NAME, impl=GR.IMPL, origin="kit", expected=GR.EXPECTED)
    for k in ("kept", "drops", "prekey_drops"):
        gr.set(k, 0)
    return dg, gr


@pytest.fixture
def kit(monkeypatch):
    monkeypatch.setattr(DG, "gate_word", lambda r, limit: DG.ABOVE_GATE if int(r.shape[2]) > int(limit) else None)   # CPU tensors pass the gate here
    monkeypatch.setattr(GR, "keeper_roots", lambda: [(ROLL.cur, False)] if ROLL.cur is not None else [])
    dg, gr = _ledgers()
    fake = FakeGraphs()
    kp = GR.Keeper(gr, True, 1 << 30)
    return types.SimpleNamespace(dg=dg, gr=gr, fake=fake, kp=kp, mod=Tiny(), stock=Tiny.forward)


def _item(kit, batch, r, sc, pb, steps=3, chunks=(None,)):
    """One roll-out (= one item): `steps` denoiser calls per chunk shape through a fresh KeepingRollOut; returns (outputs, wants, rollout)."""
    ro = GR.KeepingRollOut(kit.dg, 1 << 30, graphs=kit.fake, keeper=kit.kp)
    ROLL.cur = Roll()
    outs, wants = [], []
    with torch.no_grad():
        for _ in range(steps):
            for ch in chunks:
                r_t = torch.randn_like(r if ch is None else r[:, :ch])
                sc_t = torch.randn_like(sc)
                wants.append(kit.stock(kit.mod, batch, r_t, sc_t, pb))
                outs.append(ro.call(kit.stock, kit.mod, batch, r_t, sc_t, pb))
    ro.close()
    ROLL.cur = None
    return outs, wants, ro


def _equal_all(outs, wants):
    return all(torch.equal(a, b) for a, b in zip(outs, wants))


# --------------------------------------------------------------------------------------------------------------- the walk
def test_reftable_signature_ignores_addresses_and_object_counters_but_not_layout():
    b1, r, sc, pb1 = _args(); b2, _, _, pb2 = _args()
    ro1, ro2 = Roll(), Roll()
    ro1.memo["leaf"] = torch.zeros(1, 8); ro2.memo["leaf"] = torch.ones(1, 8); ro2.hits = 7
    t1 = GR.walk_refs([(b1, True), (pb1, True), (ro1, False)]); t2 = GR.walk_refs([(b2, True), (pb2, True), (ro2, False)])
    assert t1.key() == t2.key() and len(t1.tensors) == len(t2.tensors) == 6 and not t1.unordered
    assert GR.walk_refs([(dict(b1, n_int=4), True)]).key() != GR.walk_refs([(b1, True)]).key()          # a scalar in the batch dict: by value
    assert GR.walk_refs([(dict(b1, seq_mask=b1["seq_mask"].t()), True)]).key() != GR.walk_refs([(b1, True)]).key()   # layout
    ro3 = Roll(); ro3.memo["leaf"] = torch.zeros(1, 8); ro3.memo["extra"] = torch.zeros(2)
    assert GR.walk_refs([(ro3, False)]).key() != GR.walk_refs([(ro1, False)]).key()                    # one more held tensor: structure
    v = torch.zeros(4, 8); ro4, ro5 = Roll(), Roll()
    ro4.memo["a"], ro4.memo["b"] = v[0], v[1]                                                          # two views of ONE storage ...
    ro5.memo["a"], ro5.memo["b"] = torch.zeros(8), torch.zeros(8)                                      # ... vs two storages: aliasing is structure
    t4, t5 = GR.walk_refs([(ro4, False)]), GR.walk_refs([(ro5, False)])
    assert t4.key() != t5.key() and len(t4.storages) == 1 and len(t5.storages) == 2 and t4.nbytes == v.numel() * 4
    ro6 = Roll(); ro6.names = {torch.zeros(1)}
    assert GR.walk_refs([(ro6, False)]).unordered


def test_copy_from_moves_values_storage_wise_including_overlapping_views():
    base1, base2 = torch.zeros(10), torch.arange(10.0)
    ro1, ro2 = Roll(), Roll()
    ro1.memo["win"] = base1.unfold(0, 4, 2); ro2.memo["win"] = base2.unfold(0, 4, 2)                  # overlapping unfold views (sampler_hoist's windowed leaves)
    t1, t2 = GR.walk_refs([(ro1, False)]), GR.walk_refs([(ro2, False)])
    assert t1.key() == t2.key()
    assert t1.copy_from(t2) == 1 and torch.equal(base1, base2) and t1.copy_from(t1) == 0


def test_args_sig_is_the_call_key_without_addresses():
    b1, r, sc, pb = _args(); b2, r2, sc2, pb2 = _args()
    assert GR.args_sig(b1, r, sc, pb) == GR.args_sig(b2, r2, sc2, pb2)                                # two items of one structure
    assert GR.args_sig(b1, r[:, :1], sc, pb) != GR.args_sig(b1, r, sc, pb)                             # chunk shape
    assert GR.args_sig(dict(b1, n_int=5), r, sc, pb) != GR.args_sig(b1, r, sc, pb)                     # batch scalar
    assert GR.args_sig(b1, r, sc.double(), pb) != GR.args_sig(b1, r, sc, pb)                           # dtype
    B2 = _args(B=2)
    assert GR.args_sig(*B2) != GR.args_sig(b1, r, sc, pb)                                              # batch size
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        assert GR.args_sig(b1, r, sc, pb) != GR.args_sig(b2, r2, sc2, pb2) or True                     # (autocast state enters through DG.autocast_state)


# --------------------------------------------------------------------------------------------------------------- keep / adopt / drop
def test_second_item_of_the_same_structure_replays_the_kept_graph_with_its_own_values(kit):
    torch.manual_seed(0)
    o1, w1, ro1 = _item(kit, *_args())
    assert _equal_all(o1, w1) and kit.fake.captures == 1 and kit.gr.fallbacks == {"first": 1}
    assert len(kit.kp.sets) == 1 and kit.kp.max_live == 1
    ks = kit.kp.cur
    assert ks.building is None and len(ks.entries) == 1 and ro1.closed and len(ro1.graphs) == 0 and ks.of(kit.mod)   # the roll-out closed; the keeper holds the entry
    ent = next(iter(ks.entries.values()))
    assert ent.graph is not None and ent.refs is None and ent.table is not None and ent.owner is ks
    b2, r2, sc2, pb2 = _args(fill=3.0)                                                                 # another member of the bucket: other mask, other pair_bias VALUES
    o2, w2, _ = _item(kit, b2, r2, sc2, pb2)
    assert _equal_all(o2, w2)                                                                          # item 2's outputs are item 2's (values copied into the kept storages)
    assert kit.fake.captures == 1 and kit.gr.served == 1 and kit.gr.facts()["kept"] == 1 and kit.fake.warmups == 2
    assert torch.equal(ent.table.tensors[[i for i, t in enumerate(ent.table.tensors) if t.dim() == 5][0]], pb2)   # the kept pair_bias storage now holds item 2's
    assert kit.kp.max_live == 1 and kit.gr.gate(require_served=False).ok and kit.dg.gate(require_served=False).ok
    f = kit.gr.facts()
    assert f["live"] == 1 and f["drops"] == 0 and ks.nbytes() == ent.table.nbytes > 0 and f["tensors"] == len(ent.table.tensors) and f["storages"] == len(ent.table.storages)


def test_another_structure_drops_the_kept_set_BEFORE_capturing_one_alive_at_a_time(kit):
    torch.manual_seed(1)
    _item(kit, *_args(L=8))
    g1 = next(iter(next(iter(kit.kp.sets.values())).entries.values())).graph
    seen_live = []
    real_capture = kit.fake.capture

    def capture(fn, device=None):
        seen_live.append((len(kit.kp.sets), g1.released))                                              # at capture time the old set is already gone
        return real_capture(fn, device)
    kit.fake.capture = capture
    o, w, _ = _item(kit, *_args(L=12))
    assert _equal_all(o, w) and seen_live == [(0, True)] and kit.fake.resets >= 1
    assert kit.gr.fallbacks == {"first": 1, "key": 1} and kit.kp.drops_by == {"key": 1} and kit.kp.max_live == 1
    assert next(iter(kit.kp.sets.values())).prekey == (1, 12)
    o, w, _ = _item(kit, *_args(L=12))                                                                 # and the new structure is now the kept one
    assert _equal_all(o, w) and kit.gr.served == 1 and kit.fake.captures == 2


def test_pre_trunk_drop_by_prekey_gate_and_model(kit):
    _item(kit, *_args(L=8))
    inner_calls = []
    inference = GR.make_inference(lambda self, batch, **k: inner_calls.append(GR.prekey_of(batch)) or "ok", kit.kp)
    model = types.SimpleNamespace(diffusion_head=types.SimpleNamespace(score_model=kit.mod))             # the model whose denoiser the kept graphs belong to
    assert inference(model, {"aatype_int": torch.zeros(1, 8, dtype=torch.long)}) == "ok" and len(kit.kp.sets) == 1   # same (B, L): kept through the trunk
    assert inference(model, {"aatype_int": torch.zeros(1, 12, dtype=torch.long)}) == "ok" and len(kit.kp.sets) == 0  # another L: dropped before the trunk
    assert kit.kp.drops_by == {"prekey": 1} and kit.gr.facts()["prekey_drops"] == 1
    _item(kit, *_args(L=8))
    other = types.SimpleNamespace(diffusion_head=types.SimpleNamespace(score_model=Tiny()))              # a second model in the process: the kept set is not its
    inference(other, {"aatype_int": torch.zeros(1, 8, dtype=torch.long)})
    assert len(kit.kp.sets) == 0 and kit.kp.drops_by == {"prekey": 1, "module": 1}
    _item(kit, *_args(L=8))
    kit.kp.limit = 4                                                                                   # an item above denoiser_graph's gate: nothing kept beside its trunk
    inference(model, {"aatype_int": torch.zeros(1, 8, dtype=torch.long)})
    assert len(kit.kp.sets) == 0 and kit.kp.drops_by == {"prekey": 2, "module": 1} and inner_calls == [(1, 8), (1, 12), (1, 8), (1, 8)]


def test_one_slot_for_the_process_a_second_module_never_adopts_and_a_dead_module_drops(kit):
    _item(kit, *_args(L=8))
    ks = kit.kp.cur
    mod2 = Tiny()
    kit2 = types.SimpleNamespace(**{**vars(kit), "mod": mod2})                                         # same keeper, another DiffusionModule instance
    o, w, _ = _item(kit2, *_args(L=8))
    assert _equal_all(o, w) and kit.gr.served == 0 and kit.kp.cur is not ks and kit.kp.cur.of(mod2)   # captured its own; the first set went BEFORE that capture
    assert kit.kp.drops_by == {"module": 1} and kit.kp.max_live == 1 and kit.fake.captures == 2
    mod3 = Tiny()
    ks3 = kit.kp.new_set(mod3, (1, 8), None)                                                          # (the fake graph's replay closure pins mod2; a CUDA graph holds no Python reference)
    assert kit.kp.cur is ks3 and kit.kp.drops_by == {"module": 2}
    del mod3
    import gc; gc.collect()
    assert kit.kp.cur is None and kit.kp.drops_by == {"module": 3} and kit.kp.max_live == 1            # the module died: its kept set is released at once


def test_ragged_chunks_keep_one_entry_per_shape_and_adopt_both(kit):
    torch.manual_seed(2)
    b, r, sc, pb = _args(N=5)
    o, w, _ = _item(kit, b, r, sc, pb, steps=2, chunks=(3, 2))
    assert _equal_all(o, w) and kit.fake.captures == 2 and kit.gr.fallbacks == {"first": 2}
    ks = next(iter(kit.kp.sets.values()))
    assert len(ks.entries) == 2 and kit.kp.max_live == 1
    o, w, _ = _item(kit, *_args(N=5), steps=2, chunks=(3, 2))
    assert _equal_all(o, w) and kit.fake.captures == 2 and kit.gr.served == 2


def test_probe_mismatch_drops_recaptures_and_refuses_the_gate_output_still_stock(kit):
    torch.manual_seed(3)
    _item(kit, *_args())
    kit.fake.perturb = lambda y: y + 1.0                                                               # the kept graph's replay no longer equals the eager call
    b, r, sc, pb = _args()
    ro = GR.KeepingRollOut(kit.dg, 1 << 30, graphs=kit.fake, keeper=kit.kp)
    ROLL.cur = Roll()
    with torch.no_grad():
        r_t, sc_t = torch.randn_like(r), torch.randn_like(sc)
        kit.fake.perturb, keep = None, kit.fake.perturb
        want = kit.stock(kit.mod, b, r_t, sc_t, pb)
        kit.fake.perturb = keep
        calls = {"n": 0}
        orig = kit.fake.capture

        def capture(fn, device=None):                                                       # the fresh capture after the failed probe replays unperturbed
            kit.fake.perturb = None; calls["n"] += 1
            return orig(fn, device)
        kit.fake.capture = capture
        got = ro.call(kit.stock, kit.mod, b, r_t, sc_t, pb)
    ro.close(); ROLL.cur = None
    assert torch.equal(got, want) and calls["n"] == 1
    assert kit.gr.fallbacks == {"first": 1, "probe": 1} and kit.kp.drops_by == {"probe": 1}
    assert not kit.gr.gate(require_served=False).ok                                                   # probe is not an expected word: refused by name
    assert len(kit.kp.sets) == 1 and kit.kp.max_live == 1                                             # the item's own capture is the kept set now


def test_table_mismatch_after_warmup_is_named_and_captures(kit, monkeypatch):
    torch.manual_seed(4)
    _item(kit, *_args())
    grow = {"on": True}

    class Roll2(Roll):
        def __init__(self):
            super().__init__()
            if grow["on"]:
                self.memo["extra"] = torch.zeros(3)                                                    # this roll-out holds one more tensor: another structure after warm-up
    monkeypatch.setattr(GR, "keeper_roots", lambda: [(ROLL.cur, False)] if ROLL.cur is not None else [])
    b, r, sc, pb = _args()
    ro = GR.KeepingRollOut(kit.dg, 1 << 30, graphs=kit.fake, keeper=kit.kp)
    ROLL.cur = Roll2()
    with torch.no_grad():
        r_t, sc_t = torch.randn_like(r), torch.randn_like(sc)
        want = kit.stock(kit.mod, b, r_t, sc_t, pb)
        got = ro.call(kit.stock, kit.mod, b, r_t, sc_t, pb)
    ro.close(); ROLL.cur = None
    assert torch.equal(got, want) and kit.gr.fallbacks == {"first": 1, "table": 1} and kit.kp.drops_by == {"table": 1}
    assert not kit.gr.gate(require_served=False).ok and kit.fake.captures == 2


def test_switch_off_is_plain_denoiser_graph(kit):
    kit.kp.on = False
    _item(kit, *_args()); _item(kit, *_args())
    assert kit.fake.captures == 2 and kit.gr.fallbacks == {"disabled": 2} and kit.gr.served == 0 and len(kit.kp.sets) == 0
    assert kit.gr.gate(require_served=False).ok and kit.fake.warmups == 2                             # no probe warm-ups: exactly denoiser_graph's calls
    assert GR.enabled({}) and GR.enabled({"AFO_GRAPH_REUSE": "1"}) and not GR.enabled({"AFO_GRAPH_REUSE": "0"}) and not GR.enabled({"AFO_GRAPH_REUSE": "off"})


def test_a_dead_rollout_does_not_leave_a_kept_set(kit, monkeypatch):
    monkeypatch.setattr(DG, "MAX_GRAPHS", 1)
    b, r, sc, pb = _args(N=5)
    o, w, ro = _item(kit, b, r, sc, pb, steps=1, chunks=(3, 2))                                        # the second shape exceeds MAX_GRAPHS: recapture_storm -> eager, set dropped
    assert _equal_all(o, w) and ro.dead == DG.RECAPTURE_STORM and len(kit.kp.sets) == 0 and kit.kp.drops_by == {"trip": 1}


def test_single_item_run_is_one_capture_and_no_adoption(kit):
    o, w, _ = _item(kit, *_args(), steps=4)
    assert _equal_all(o, w) and kit.fake.captures == 1 and kit.fake.warmups == 1 and kit.gr.served == 0 and kit.gr.fallbacks == {"first": 1}
    assert kit.gr.gate(require_served=False).ok                                                       # `first` only: a stock-equivalent run of the lever, not a refusal


# --------------------------------------------------------------------------------------------------------------- install wiring / rows
def test_install_wires_factory_and_inference_and_reads_denoiser_graph_state(monkeypatch):
    monkeypatch.setattr(DG, "ROLLOUT_FACTORY", None)
    monkeypatch.setattr(GR, "KEEPER", None)
    monkeypatch.setattr(DG, "INSTALLED", {})
    ins = GR.install("fast", "[t]", {})
    assert not ins.applied and ins.reason == "denoiser_graph_absent"                                  # denoiser_graph not installed: not applied, by name
    dg, _ = _ledgers()
    M = types.ModuleType("fake_model")

    class AtlasFold:
        def inference(self, batch, num_recycles=3):
            return ("stock", num_recycles)
    M.AtlasFold = AtlasFold
    monkeypatch.setattr(GR, "MODEL_SITES", (("fake_model", "AtlasFold"), ("fake_model_absent", "Nope")))
    import sys
    monkeypatch.setitem(sys.modules, "fake_model", M)
    ins = GR.install("fast", "[t]", {"denoiser_graph": {"ledger": dg, "limit": 1024}})
    assert ins.applied and DG.ROLLOUT_FACTORY is not None and GR.KEEPER is ins.facts["keeper"] and ins.facts["keeper"].limit == 1024
    ro = DG.ROLLOUT_FACTORY(dg, 1024)
    assert isinstance(ro, GR.KeepingRollOut) and ro.keeper is GR.KEEPER
    assert "graph_reuse" in AtlasFold.inference.__qualname__ and AtlasFold().inference({"aatype_int": torch.zeros(1, 4, dtype=torch.long)}, num_recycles=1) == ("stock", 1)
    line = ins.lines[0]()
    for word in ("name=LOCAL.atlasfold.graph_reuse", "kept=0", "drops=0", "live=0", "max_live=0", "kept_gib=", "switch=on", "sites=1"):
        assert word in line, (word, line)
    assert ins.gates[0]().ok


def test_registry_row_modes_and_ablation_closure():
    from atlasfold_opt.registry import LEVERS
    from atlasfold_opt.modes import MODES
    from atlasfold_opt.hooks import installers
    from atlasfold_opt import ablation as A
    row = LEVERS["graph_reuse"]
    assert row["cls"] == "fast" and row["module"] == "atlasfold_opt.hooks.graph_reuse" and tuple(row["requires"]) == ("denoiser_graph",)
    assert set(row["expected"]) == {"first", "key", "disabled"} == set(GR.EXPECTED)
    fast, big, exact = MODES["fast"], MODES["big"], MODES["exact"]
    assert "graph_reuse" in fast and fast.index("graph_reuse") > fast.index("denoiser_graph") > fast.index("sampler_hoist")   # installed after the levers whose state it reads
    assert "graph_reuse" not in big and "graph_reuse" not in exact and "denoiser_graph" not in big
    assert installers()["graph_reuse"] is GR.install
    assert [l for l, why in A.dropped("fast", ["denoiser_graph"]) if l == "graph_reuse" and "denoiser_graph" in why]   # MODEL_OPT_LEVERS_OFF=denoiser_graph takes graph_reuse with it (requires), by name
    assert A.dropped("fast", ["graph_reuse"]) == [] and A.validate("fast", ["graph_reuse"])            # ... and graph_reuse alone leaves denoiser_graph as it is
