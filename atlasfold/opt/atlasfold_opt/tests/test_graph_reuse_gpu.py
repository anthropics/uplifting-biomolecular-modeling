"""GPU tests of lever graph_reuse (skipped without CUDA): the stock DiffusionHead built tiny, sampled item after item the way the runners do —
a FRESH feature dict per item (new addresses, other values), one sample() roll-out each — with denoiser_graph + graph_reuse installed
(and once more with sampler_hoist under them, whose roll-out leaves the captured kernels read by reference): every item's coordinates
must be the bits of the eager stock sampler for THAT item; a second item of the same structure must be answered by replay (no capture);
an item of another structure must find the kept set dropped before its own capture (one alive); dropping the kept set must return the
allocator to where denoiser_graph alone leaves it."""
import pytest
import torch

from atlasfold_opt.tests.test_denoiser_graph_gpu import _tiny_head_and_batch, _sample, _Count

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _item_batch(device, B, L, seed):
    """A fresh feature dict of the (B, L) structure with its own values (another member of the bucket) + trunk outputs s, z."""
    torch.manual_seed(seed)
    _, batch, s, z = _tiny_head_and_batch(device, B=B, L=L)
    g = torch.Generator().manual_seed(seed)
    keep = (torch.rand(B, L, 14, generator=g) > 0.25).to(device)
    batch["atom14_mask"] = keep
    batch["seq_mask"] = torch.ones(B, L, dtype=torch.bool, device=device)
    batch["seq_mask"][:, L - 1 - seed % 3:] = False                               # a shorter member of the same bucket: other masks, same shapes
    batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}   # new addresses, as a runner's next model_run
    s = torch.randn(s.shape, generator=g).to(device); z = torch.randn(z.shape, generator=g).to(device)
    return batch, s, z


def _install(monkeypatch, with_hoist):
    import atlasfold.model.network.diffusion_head as DH
    from atlasfold_opt.hooks import denoiser_graph as DG, graph_reuse as GR
    monkeypatch.setattr(DG, "ROLLOUT_FACTORY", None)
    monkeypatch.setattr(DG, "INSTALLED", {})
    monkeypatch.setattr(GR, "KEEPER", None)
    monkeypatch.setattr(GR, "MODEL_SITES", ())                                    # the pre-trunk drop wrapper is a CPU-tested one-liner; no model class here
    monkeypatch.setenv("AFO_DENOISER_GRAPH_MAX_TOKENS", "1024")
    saved = [(DH.DiffusionModule, "forward", DH.DiffusionModule.forward), (DH.DiffusionHead, "sample", DH.DiffusionHead.sample)]
    installed = []
    ctx = {}
    if with_hoist:
        from atlasfold_opt.hooks import sampler_hoist as SH
        ins = SH.install("fast", "atlasfold-opt", ctx)
        assert ins.applied, ins.reason
        installed.append(ins)
    ins_dg = DG.install("fast", "atlasfold-opt", ctx); assert ins_dg.applied, ins_dg.reason
    ins_gr = GR.install("fast", "atlasfold-opt", ctx); assert ins_gr.applied, ins_gr.reason
    installed += [ins_dg, ins_gr]

    def restore():
        GR.KEEPER and GR.KEEPER.drop_all("test")
        for ins in reversed(installed):
            r = getattr(ins, "restore", None)
            if callable(r):
                try:
                    r()
                except Exception:  # noqa: BLE001
                    pass
        for cls, name, fn in saved:
            setattr(cls, name, fn)
    return ins_dg, ins_gr, restore


@cuda
@pytest.mark.parametrize("num_samples,chunk,bf16,B,with_hoist", [(3, 5, False, 1, False), (3, 2, True, 1, False), (3, 2, True, 2, False), (3, 5, True, 1, True), (3, 2, True, 2, True)])
def test_items_of_one_structure_replay_the_kept_graph_bitwise_and_one_set_is_alive(num_samples, chunk, bf16, B, with_hoist, monkeypatch):
    dev = torch.device("cuda")
    head, _, _, _ = _tiny_head_and_batch(dev, B=B, L=24)
    steps = 4
    plan = [("A", B, 24, 11), ("C", B, 24, 12), ("Bx", B, 40, 13), ("A2", B, 24, 11), ("C2", B, 24, 12)]   # same bucket twice, another bucket, back
    items = {name: _item_batch(dev, b, L, seed) for name, b, L, seed in plan}
    want = {name: _sample(head, *items[name], num_samples, steps, chunk, bf16=bf16) for name, *_ in plan}   # eager stock, per item
    assert not torch.equal(want["A"], want["C"])                                  # the two members differ (a stale by-reference tensor would show)
    ins_dg, ins_gr, restore = _install(monkeypatch, with_hoist)
    try:
        dg, gr = ins_dg.facts["ledger"], ins_gr.facts["ledger"]
        kp = ins_gr.facts["keeper"]
        shapes = len({min(chunk, num_samples - st) for st in range(0, num_samples, chunk)})
        enc = _Count(head.score_model.atom_encoder) if not with_hoist else None   # (sampler_hoist re-binds the encoder's forward per roll-out: not counted under it)
        from atlasfold_opt.hooks.denoiser_graph import CudaGraphs
        CudaGraphs.warmup(lambda: torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev), dev)   # the side stream's cuBLAS workspace exists before the base line is read (a first capture in a process allocates it for good, lever or not)
        torch.cuda.synchronize(); base_alloc = torch.cuda.memory_allocated()
        got = {}
        for name, *_ in plan:
            got[name] = _sample(head, *items[name], num_samples, steps, chunk, bf16=bf16)
            assert kp.max_live <= 1 and len(kp.sets) == 1, (name, kp.max_live, len(kp.sets))
        for name, *_ in plan:
            assert torch.equal(got[name], want[name]), (name, float((got[name] - want[name]).abs().max()))
        f, g = dg.facts(), gr.facts()
        # A captures (first), C adopts, Bx drops A/C's set then captures (key), A2 drops + captures (key), C2 adopts
        assert f["captures"] == 3 * shapes and f["recaptures"] == 0 and f["rollouts"] == 5, f
        assert gr.served == 2 * shapes and dict(gr.fallbacks) == {"first": 3 * shapes - 2, "key": 2}, (gr.served, dict(gr.fallbacks))   # first: A's graphs + the second chunk shape of Bx / A2; key: Bx, A2
        assert g["kept"] == 2 * shapes and g["drops"] == 2 and g["max_live"] == 1 and g["live"] == 1 and kp.drops_by == {"key": 2}, (g, kp.drops_by)
        assert enc is None or enc.n == 3 * shapes * 2 + 2 * shapes                 # eager Python: warm-up + capture per captured graph, ONE warm-up per adopted graph
        assert ins_gr.gates[0]().ok and ins_dg.gates[0]().ok
        line = ins_gr.lines[0]()
        for word in ("state=on", "kept=%d" % (2 * shapes), "max_live=1", "live=1", "kept_gib=", "adopt_s="):
            assert word in line, (word, line)
        kept_bytes = next(iter(kp.sets.values())).nbytes()
        assert kept_bytes > 0 and abs(float(g["kept_gib"]) * 2 ** 30 - kept_bytes) < 2 ** 21
        kp.drop_all("test")                                                       # what the keeper holds is all it holds: dropping it returns the allocator to the base line
        torch.cuda.synchronize()
        assert abs(torch.cuda.memory_allocated() - base_alloc) <= (1 << 21), (torch.cuda.memory_allocated(), base_alloc, kept_bytes)
        enc is not None and enc.restore()
    finally:
        restore()


@cuda
def test_switch_off_captures_per_item_as_denoiser_graph_alone(monkeypatch):
    dev = torch.device("cuda")
    head, _, _, _ = _tiny_head_and_batch(dev, B=1, L=24)
    a, c = _item_batch(dev, 1, 24, 21), _item_batch(dev, 1, 24, 22)
    want_a, want_c = _sample(head, *a, 3, 3, 5), _sample(head, *c, 3, 3, 5)
    monkeypatch.setenv("AFO_GRAPH_REUSE", "0")
    ins_dg, ins_gr, restore = _install(monkeypatch, False)
    try:
        got_a, got_c = _sample(head, *a, 3, 3, 5), _sample(head, *c, 3, 3, 5)
        assert torch.equal(got_a, want_a) and torch.equal(got_c, want_c)
        assert ins_dg.facts["ledger"].facts()["captures"] == 2 and dict(ins_gr.facts["ledger"].fallbacks) == {"disabled": 2}
        assert ins_gr.facts["keeper"].sets == {} and ins_gr.gates[0]().ok and "switch=off" in ins_gr.lines[0]()
    finally:
        restore()
