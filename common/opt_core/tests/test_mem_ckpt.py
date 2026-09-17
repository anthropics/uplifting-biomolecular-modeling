"""opt_core.mem.ckpt through opt_core.mem.apply on CPU with synthetic modules: equality of the levered paths against the un-levered
ones (torch.equal where the lever claims bit-exact; the draws bit-exact and the outputs within tolerance where it claims measured), the
record (marks, settings, flags), and every refusal by name."""
import pytest

torch = pytest.importorskip("torch")

from opt_core import mem  # noqa: E402
from opt_core.mem import ckpt, registry  # noqa: E402


def _cuda_present() -> bool:
    """These tests hold the no-CUDA path (refusal / named fallback); on a CUDA box they skip by name."""
    try:
        import torch                                       # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:                                      # noqa: BLE001
        return False


no_cuda_path = pytest.mark.skipif(_cuda_present(), reason="CUDA present: this test holds the no-CUDA refusal/fallback path")

N_TOK, C = 24, 8
N_SAMPLES, N_STEPS = 6, 4


def make_ctx(hooks=None, settings=None, environ=None, **kw):
    return mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch", hooks=hooks or {}, settings=settings or {},
                   environ={} if environ is None else environ, **kw)


# ------------------------------------------------------------------------------------------------------ synthetic modules


class PairBlock(torch.nn.Module):
    """One triangle-flavoured pair block: LN → row mixing → column mixing → residual. Deterministic, batch-free."""

    def __init__(self, c):
        super().__init__()
        self.ln = torch.nn.LayerNorm(c)
        self.row = torch.nn.Linear(c, c)
        self.col = torch.nn.Linear(c, c)

    def forward(self, z):
        h = self.ln(z)
        h = self.row(h) + self.col(h.transpose(0, 1)).transpose(0, 1)
        return z + torch.tanh(h)


class Trunk(torch.nn.Module):
    """A pair stack with recycles: z = z_init + LN(z_prev) each cycle, then the blocks."""

    def __init__(self, c, n_blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(PairBlock(c) for _ in range(n_blocks))
        self.recycle_ln = torch.nn.LayerNorm(c)
        self.single = torch.nn.Linear(c, c)

    def cycle_start(self, z_init, z_prev):
        return z_init + self.recycle_ln(z_prev)

    def cycle_body(self, z, s_init, s_prev):
        for b in self.blocks:
            z = b(z)
        s = s_init + torch.tanh(self.single(s_prev)) + z.mean(dim=1)
        return z, s

    @torch.no_grad()
    def forward_stock(self, z_init, s_init, n_cycles):
        z_prev, s_prev = torch.zeros_like(z_init), torch.zeros_like(s_init)
        for _ in range(n_cycles):
            z = self.cycle_start(z_init, z_prev)
            z_prev, s_prev = self.cycle_body(z, s_init, s_prev)
            del z
        return z_prev, s_prev

    @torch.no_grad()
    def forward_carry(self, z_init, s_init, n_cycles, carry):
        carry.hold(z_init=z_init, s_init=s_init, z=torch.zeros_like(z_init), s=torch.zeros_like(s_init))
        del z_init, s_init
        for cycle in range(n_cycles):
            zi, zp = carry.get("z_init"), carry.get("z")
            z = self.cycle_start(zi, zp)
            del zi, zp
            si, sp = carry.get("s_init"), carry.get("s")
            z, s = self.cycle_body(z, si, sp)
            del si, sp
            carry.hold(z=z, s=s)
            del z, s
            carry.seam(cycle)
        out = carry.get("z"), carry.get("s")
        carry.release()
        return out


CARRIED = ("z_init", "s_init", "z", "s")


def trunk_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    trunk = Trunk(C)
    for p in trunk.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return trunk, torch.randn(N_TOK, N_TOK, C, generator=g), torch.randn(N_TOK, C, generator=g)


# ------------------------------------------------------------------------------------------------------ recycle_carry


@pytest.mark.parametrize("park", ["host", "device"])
def test_recycle_carry_is_bitwise_vs_the_stock_loop_and_marks_the_record(park):
    trunk, z0, s0 = trunk_inputs()
    z_ref, s_ref = trunk.forward_stock(z0, s0, 3)
    ctx = make_ctx(hooks={"recycle_carry": {"device": "cpu", "carried": CARRIED}},
                   settings={"recycle_carry": {"park": park, "flush": False, "pin_max_gb": 1.0}})
    rec = mem.apply(["recycle_carry"], ctx, base="exact")
    a = rec.applied[0]
    assert rec.levers == ("recycle_carry",) and a.exact == "bitwise" and a.settings["park"] == park and a.settings["stock"] is False
    assert a.settings["pin_max_gb"] == (1.0 if park == "host" else None)
    assert (park == "device") == any("seam-only diagnostic arm" in n for n in a.notes)
    carry = ckpt.carry_of(ctx)
    rec.unit_begin("item-0")                                                # the adapter opens a unit per item around the seams
    z, s = trunk.forward_carry(z0.clone(), s0.clone(), 3, carry)
    rec.unit_end()
    assert torch.equal(z, z_ref) and torch.equal(s, s_ref)
    assert [e["cycle"] for e in carry.record["seams"]] == [0, 1, 2] and all(e["released"] is False for e in carry.record["seams"])
    assert a.settings["seams"] is carry.record["seams"] and carry.record["released"] is True
    if park == "host":
        assert set(carry.record["held"]) == set(CARRIED) and all(h["where"] == "host" for h in carry.record["held"].values())
        assert carry.record["host_park"]["closed"] is True and carry.record["host_park"]["pinned"] is False       # the CPU test line
        assert any(e["event"] == "park" for e in carry.record["host_park"]["events"])
    census = rec.census()
    assert census["ok"] and census["units"]["item-0"]["ran"] == ["recycle_carry"] and census["units"]["item-0"]["closed"]
    assert [e["detail"] for e in rec.units["item-0"].events] == ["seam cycle=0 released=False", "seam cycle=1 released=False", "seam cycle=2 released=False"]
    assert rec.exit_gate(0)["exit_code"] == 0 and a.scope == "unit"
    assert {(s["lever"], s["key"], s["source"]) for s in rec.settings} >= {("recycle_carry", "park", "ctx.settings"), ("recycle_carry", "flush", "ctx.settings")}


def test_recycle_carry_reads_the_kits_string_settings():
    given = {"park": "host", "pin_max_gb": "0.5", "flush": "0"}                                  # the strings the kit read from its own variables, cast by the lever
    ctx = make_ctx(hooks={"recycle_carry": {"device": "cpu", "carried": CARRIED}}, settings={"recycle_carry": dict(given)},
                   environ={"ACME_BIG_RECYCLE_CARRY_PARK": "device", "ACME_BIG_RECYCLE_CARRY": "0"})   # the environment is never read
    rec = mem.apply(["recycle_carry"], ctx, base="exact")
    a = rec.applied[0]
    assert a.settings["park"] == "host" and a.settings["pin_max_gb"] == 0.5 and a.settings["flush"] is False and rec.levers == ("recycle_carry",)
    assert {s["source"] for s in rec.settings if s["lever"] == "recycle_carry" and s["key"] in given} == {"ctx.settings"}
    off = make_ctx(hooks={"recycle_carry": {"device": "cpu", "carried": CARRIED}})
    rec = mem.apply(["recycle_carry"], off, base="exact", switches={"recycle_carry": False})
    assert rec.levers == () and rec.off_by_flag == ("recycle_carry",)
    with pytest.raises(mem.Refused, match="names no registered lever and no line lever"):
        mem.apply(["recycle_carry"], off, base="exact", switches={"recycle_park": True})


def test_recycle_carry_refuses_by_name():
    def refusal(hooks, settings=None, line=("recycle_carry",)):
        rec = mem.apply(list(line), make_ctx(hooks=hooks, settings=settings or {}), base="exact", strict=False)
        assert rec.refused, "expected a refusal"
        return rec.refused[-1]

    r = refusal({"recycle_carry": {"device": "cpu"}})
    assert r.precondition == "hooks.carried" and r.lever == "recycle_carry"
    r = refusal({"recycle_carry": {"device": "cpu", "carried": "z"}})
    assert r.precondition == "hooks.carried"
    r = refusal({"recycle_carry": {"device": "cpu", "carried": CARRIED}})
    assert r.precondition == "park" and "no park" in r.reason
    r = refusal({"recycle_carry": {"device": "cpu", "carried": CARRIED}}, {"recycle_carry": {"park": "disk"}})
    assert r.precondition == "park"
    r = refusal({"recycle_carry": {"device": "cpu", "carried": CARRIED}}, {"recycle_carry": {"park": "host", "flush": False}})
    assert r.precondition == "pin_max_gb"
    r = refusal({"recycle_carry": {"device": "cpu", "carried": CARRIED}}, {"recycle_carry": {"park": "device"}})
    assert r.precondition == "flush" and "needs a CUDA device" in r.reason
    r = refusal({"recycle_carry": {"device": "cuda:0", "carried": CARRIED}}, {"recycle_carry": {"park": "device"}})
    assert r.precondition == "flush" and "cache_release" in r.reason and "not applied" in r.reason
    r = refusal({"recycle_carry": {"device": "cuda:0", "carried": CARRIED}}, {"recycle_carry": {"park": "device"}, "cache_release": {"policy": "per_unit"}},
                line=("cache_release", "recycle_carry"))
    assert r.precondition == "flush" and "found: per_unit" in r.reason
    r = refusal({"recycle_carry": {"device": "tpu", "carried": CARRIED}}, {"recycle_carry": {"park": "device"}})
    assert r.precondition == "hooks.device"


@no_cuda_path
def test_recycle_carry_seam_releases_through_cache_release_only():
    """With cache_release per_stage in the line the seam routes through allocator.release: on a CPU box the primitive records its own
    fallback (torch.cuda unavailable) — the seam itself never calls empty_cache."""
    ctx = make_ctx(hooks={"recycle_carry": {"device": "cuda:0", "carried": CARRIED}},
                   settings={"recycle_carry": {"park": "device"}, "cache_release": {"policy": "per_stage"}})
    rec = mem.apply(["cache_release", "recycle_carry"], ctx, base="exact")
    assert rec.levers == ("cache_release", "recycle_carry")
    carry = ckpt.carry_of(ctx)
    rec.unit_begin("item-0")
    carry.hold(z=torch.ones(2))
    entry = carry.seam(0)
    rec.unit_end()
    assert entry["released"] is False and entry["before"]["max_allocated_gib"] is None          # counters: None when CUDA is unavailable, never zero
    u = rec.units["item-0"]
    assert "cache_release" in u.fallback and "torch.cuda unavailable" in u.fallback["cache_release"] and u.ran == ["recycle_carry"]


def test_recycle_carry_run_time_discipline_is_loud():
    ctx = make_ctx(hooks={"recycle_carry": {"device": "cpu", "carried": ("z",)}}, settings={"recycle_carry": {"park": "device", "flush": False}})
    mem.apply(["recycle_carry"], ctx, base="exact")
    carry = ckpt.carry_of(ctx)
    with pytest.raises(ckpt.CkptError, match="'s' is not a carried name"):
        carry.hold(s=torch.ones(2))
    with pytest.raises(ckpt.CkptError, match="not a tensor"):
        carry.hold(z=3)
    with pytest.raises(ckpt.CkptError, match="never held"):
        carry.get("z")
    z = torch.arange(4.0)
    carry.hold(z=z)
    assert carry.get("z").data_ptr() == z.data_ptr()                       # park=device: the held tensor itself
    carry.release()
    with pytest.raises(ckpt.CkptError, match="never held"):
        carry.get("z")
    with pytest.raises(ckpt.CkptError, match="not applied"):
        ckpt.carry_of(make_ctx())
    with pytest.raises(ckpt.CkptError, match="flush=1 needs the cache_release lever"):
        ckpt.RecycleCarry(None, "cpu", ("z",), park="device", flush=True)
    with pytest.raises(ckpt.CkptError, match="park=host needs pin_max_gb"):
        ckpt.RecycleCarry(None, "cpu", ("z",), park="host", flush=False)


def test_recycle_carry_host_park_returns_a_copy_and_the_primitive_refuses_by_name():
    ctx = make_ctx(hooks={"recycle_carry": {"device": "cpu", "carried": ("z",)}}, settings={"recycle_carry": {"park": "host", "flush": False, "pin_max_gb": 1.0}})
    mem.apply(["recycle_carry"], ctx, base="exact")
    carry = ckpt.carry_of(ctx)
    z = torch.arange(4.0)
    carry.hold(z=z)
    got = carry.get("z")
    got.add_(1.0)
    assert torch.equal(carry.get("z"), torch.arange(4.0)) and torch.equal(z, torch.arange(4.0))
    carry.hold(z=torch.arange(4.0) * 2)                                     # re-hold the same shape: written into the parked buffer
    assert torch.equal(carry.get("z"), torch.arange(4.0) * 2)
    carry.hold(z=torch.ones(2, 2))                                          # a new shape: re-parked
    assert torch.equal(carry.get("z"), torch.ones(2, 2))
    carry.release()
    from opt_core.mem import offload
    tiny = ckpt.RecycleCarry(None, "cpu", ("z",), park="host", flush=False, pin_max_gb=1e-9)
    with pytest.raises(offload.OffloadRefusal, match="pin_budget_exceeded"):
        tiny.hold(z=torch.ones(1024))


# ------------------------------------------------------------------------------------------------------ samples_per_pass


def sampler(den, x0, n_steps, chunk, draws, g):
    """A synthetic diffusion sampler over the samples ``x0[chunk]``: each step draws stock-shaped noise through ``draws`` and denoises."""
    x = x0[chunk]
    noises = []
    for _ in range(n_steps):
        eps = draws.draw(lambda: torch.randn(N_SAMPLES, N_TOK, 3, generator=g), chunk)
        noises.append(eps)
        x = x + 0.1 * den(x) + 0.05 * eps
    return {"x": x, "noise": torch.stack(noises, 1)}


def elementwise(x):
    return torch.tanh(x) * 0.5


def matmul_den(x):
    w = torch.linspace(-1, 1, N_TOK).view(N_TOK, 1) @ torch.linspace(-1, 1, N_TOK).view(1, N_TOK)
    return torch.einsum("ij,sjc->sic", w, x) / N_TOK


def stock_run(den, seed=7):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(N_SAMPLES, N_TOK, 3, generator=g)
    draws = ckpt.StockOrderDraws(generator=g)
    draws.open_chunk()
    out = sampler(den, x0, N_STEPS, slice(0, N_SAMPLES), draws, g)
    return x0, out, g.get_state()


def chunked_ctx(den, k, draw_order="stock", seed=7, **hooks):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(N_SAMPLES, N_TOK, 3, generator=g)
    run = lambda chunk, draws: sampler(den, x0, N_STEPS, chunk, draws, g)  # noqa: E731
    ctx = make_ctx(hooks={"samples_per_pass": dict({"n_samples": N_SAMPLES, "run": run, "generator": g, "sample_dim": 0}, **hooks)},
                   settings={"samples_per_pass": {"k": k, "draw_order": draw_order}})
    return ctx, g


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_samples_per_pass_keeps_stock_draws_bitwise_and_records_the_pass(k):
    x0, ref, end_state = stock_run(elementwise)
    ctx, g = chunked_ctx(elementwise, k)
    rec = mem.apply(["samples_per_pass"], ctx, base="fast")
    a = rec.applied[0]
    assert a.settings["k"] == k and a.settings["stock"] is False and a.scope == "unit"
    assert a.exact == "measured" == a.declared_exact and a.narrowed_by is None and "draw_order=stock" in a.exact_reason
    assert any(ckpt.IDENTITY_RECORD in n for n in a.notes) and rec.exact == "measured"
    rec.unit_begin("item-0")                                                # the adapter opens a unit per item around the sample pass
    out = ckpt.run_sample_chunks(ctx)
    rec.unit_end()
    assert torch.equal(out["noise"], ref["noise"])                          # every sample sees stock's noise
    assert torch.equal(out["x"], ref["x"])                                  # an elementwise denoiser is bit-exact
    assert torch.equal(g.get_state(), end_state)                            # the stream ends where stock's ends
    last = a.settings["last_pass"]
    assert last["draw_order"] == "stock" and last["census"] == {"draws_per_chunk": [N_STEPS] * len(last["chunks"]), "consistent": True, "end_states_equal": True}
    assert last["chunks"] == [[c.start, c.stop] for c in ckpt.sample_chunks(N_SAMPLES, k)]
    u = rec.units["item-0"]
    assert u.ran == ["samples_per_pass"] and u.events[0]["detail"] == f"n={N_SAMPLES} k={k} chunks={len(last['chunks'])} draw_order=stock"
    assert rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0


def test_samples_per_pass_measured_denoiser_keeps_the_draws_within_tolerance():
    _, ref, _ = stock_run(matmul_den)
    ctx, _ = chunked_ctx(matmul_den, 2)
    mem.apply(["samples_per_pass"], ctx, base="fast")
    out = ckpt.run_sample_chunks(ctx)
    assert torch.equal(out["noise"], ref["noise"])
    assert torch.allclose(out["x"], ref["x"], atol=1e-6, rtol=1e-6)


def test_samples_per_pass_chunked_draw_order_is_a_different_stream_measured_with_the_d82_reason():
    _, ref, _ = stock_run(elementwise)
    ctx, _ = chunked_ctx(elementwise, 2, draw_order="chunked")
    rec = mem.apply(["samples_per_pass"], ctx, base="fast")
    a = rec.applied[0]
    assert a.exact == "measured" == a.declared_exact and "D82" in a.exact_reason and "chunked" in a.exact_reason and a.narrowed_by is None
    rec.unit_begin("item-0")
    out = ckpt.run_sample_chunks(ctx)
    rec.unit_end()
    assert out["noise"].shape == ref["noise"].shape and not torch.equal(out["noise"], ref["noise"])
    assert a.settings["last_pass"]["draw_order"] == "chunked" and a.settings["last_pass"]["census"] is None


def test_samples_per_pass_not_applied_is_the_stock_pass_unrecorded():
    _, ref, _ = stock_run(elementwise)
    ctx, _ = chunked_ctx(elementwise, 2)
    rec = mem.apply([], ctx, base="fast")
    out = ckpt.run_sample_chunks(ctx)
    assert torch.equal(out["x"], ref["x"]) and rec.units == {}


def test_samples_per_pass_gates_are_loud():
    g = torch.Generator().manual_seed(1)

    def uneven(chunk, draws):                                               # the second chunk draws one step fewer
        steps = N_STEPS if chunk.start == 0 else N_STEPS - 1
        return torch.stack([draws.draw(lambda: torch.randn(N_SAMPLES, N_TOK, 3, generator=g), chunk) for _ in range(steps)], 1)

    with pytest.raises(ckpt.CkptError, match="draw census differs"):
        ckpt._run_chunks(N_SAMPLES, 3, uneven, generator=g)

    def other_shape(chunk, draws):                                          # the second chunk draws a different shape
        shape = (N_SAMPLES, N_TOK, 3) if chunk.start == 0 else (N_SAMPLES, N_TOK, 2)
        return draws.draw(lambda: torch.randn(shape, generator=g), chunk)

    with pytest.raises(ckpt.CkptError, match="draw census differs"):
        ckpt._run_chunks(N_SAMPLES, 3, other_shape, generator=g)

    def bypass(chunk, draws):                                               # a draw outside draws.draw on the second chunk: the stream diverges
        out = draws.draw(lambda: torch.randn(N_SAMPLES, N_TOK, 3, generator=g), chunk)
        if chunk.start:
            torch.randn(chunk.stop - chunk.start, N_TOK, 3, generator=g)
        return out

    with pytest.raises(ckpt.CkptError, match="RNG end state differs"):
        ckpt._run_chunks(N_SAMPLES, 3, bypass, generator=g)
    draws = ckpt.StockOrderDraws(generator=g)
    with pytest.raises(ckpt.CkptError, match="draw before rewind"):
        draws.draw(lambda: torch.randn(2), slice(0, 1))
    with pytest.raises(ckpt.CkptError, match="differ in keys"):
        ckpt._cat_outputs([{"a": torch.ones(1)}, {"b": torch.ones(1)}], 0)
    with pytest.raises(ckpt.CkptError, match="n_samples and k must be"):
        ckpt.sample_chunks(0, 1)
    with pytest.raises(ckpt.CkptError, match="needs run and n_samples"):
        ckpt.run_sample_chunks(make_ctx())


def test_samples_per_pass_refuses_by_name():
    def refusal(hooks, settings=None, environ=None):
        rec = mem.apply(["samples_per_pass"], make_ctx(hooks=hooks, settings=settings or {}, environ=environ), base="fast", strict=False)
        assert rec.refused
        return rec.refused[-1]

    run = lambda chunk, draws: None  # noqa: E731
    assert refusal({"samples_per_pass": {"run": run}}).precondition == "hooks.n_samples"
    assert refusal({"samples_per_pass": {"n_samples": 0, "run": run}}).precondition == "hooks.n_samples"
    assert refusal({"samples_per_pass": {"n_samples": 5, "run": "nope"}}).precondition == "hooks.run"
    r = refusal({"samples_per_pass": {"n_samples": 5, "run": run}})
    assert r.precondition == "k" and "ctx.settings['samples_per_pass']['k']" in r.reason
    assert refusal({"samples_per_pass": {"n_samples": 5, "run": run}}, {"samples_per_pass": {"k": 0}}).precondition == "k"
    for k in (5, 6):                                                       # k >= N is the stock pass: refused by name, never a narrowed label
        r = refusal({"samples_per_pass": {"n_samples": 5, "run": run}}, {"samples_per_pass": {"k": k}})
        assert r.precondition == "k" and "the stock pass" in r.reason and "switches={'samples_per_pass': False}" in r.reason
    r = refusal({"samples_per_pass": {"n_samples": 5, "run": run}}, {"samples_per_pass": {"k": "two"}})
    assert r.precondition == "settings.samples_per_pass.k"
    assert refusal({"samples_per_pass": {"n_samples": 5, "run": run}}, {"samples_per_pass": {"k": 1, "draw_order": "random"}}).precondition == "draw_order"
    assert refusal({"samples_per_pass": {"n_samples": 5, "run": run, "sample_dim": "0"}}, {"samples_per_pass": {"k": 1}}).precondition == "hooks.sample_dim"
    rec = mem.apply(["samples_per_pass"], make_ctx(hooks={"samples_per_pass": {"n_samples": 5, "run": run}}, settings={"samples_per_pass": {"k": "2"}},
                                                     environ={"ACME_BIG_SAMPLES_PER_PASS": "0", "ACME_BIG_SAMPLES_PER_PASS_K": "9"}), base="fast", strict=False)
    assert not rec.refused and rec.applied[0].settings["k"] == 2                              # a string k is cast; the environment is never read


# ------------------------------------------------------------------------------------------------------ seed_batch


def test_seed_batch_plans_in_stock_order_and_refuses_by_name():
    assert ckpt.seed_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]] and ckpt.seed_batches([1, 2], 1) == [[1], [2]]
    with pytest.raises(ckpt.CkptError):
        ckpt.seed_batches([1], 0)
    ctx = make_ctx(hooks={"seed_batch": {"seeds": [1, 2, 3], "capable": True}}, settings={"seed_batch": {"k": 2}})
    rec = mem.apply(["seed_batch"], ctx, base="fast")
    a = rec.applied[0]
    assert a.exact == "measured" == a.declared_exact and a.narrowed_by is None and "bitwise-claimed" in a.exact_reason and a.scope == "unit"
    assert a.settings == {"k": 2, "n_seeds": 3, "batches": [[1, 2], [3]], "stock": False}
    rec.unit_begin("item-0")                                                # the adapter plans the seeds per item, inside its unit
    assert ckpt.seed_batch_plan(ctx) == [[1, 2], [3]]
    rec.unit_end()
    assert rec.units["item-0"].ran == ["seed_batch"] and rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
    assert ckpt.seed_batch_plan(make_ctx(), [4, 5]) == [[4], [5]]           # not applied: the stock loop, unrecorded

    def refusal(hooks, settings=None):
        rec = mem.apply(["seed_batch"], make_ctx(hooks=hooks, settings=settings or {}), base="fast", strict=False)
        return rec.refused[-1]

    assert refusal({"seed_batch": {"seeds": [1]}}).precondition == "hooks.capable"
    assert refusal({"seed_batch": {"seeds": [], "capable": True}}).precondition == "hooks.seeds"
    r = refusal({"seed_batch": {"seeds": [1, 2], "capable": True}})
    assert r.precondition == "k" and "never a default" in r.reason and "ctx.settings['seed_batch']['k']" in r.reason
    assert refusal({"seed_batch": {"seeds": [1, 2], "capable": True}}, {"seed_batch": {"k": 0}}).precondition == "k"
    r = refusal({"seed_batch": {"seeds": [1, 2], "capable": True}}, {"seed_batch": {"k": 1}})
    assert r.precondition == "k" and "the stock loop" in r.reason and "switches={'seed_batch': False}" in r.reason    # k=1: refused, never a narrowed label
    r = refusal({"seed_batch": {"seeds": [1, 2], "capable": False}}, {"seed_batch": {"k": 2}})
    assert r.precondition == "hooks.capable" and "batched runner" in r.reason


# ------------------------------------------------------------------------------------------------------ graph_capture


def test_graph_policy_counts_refusals_per_site_and_fires_once_per_site_tokens():
    events = []
    pol = ckpt.GraphPolicy(policy="budget", budget=256, on_event=lambda s, t: events.append((s, t)))
    assert pol.admit("trunk", 200) and not pol.admit("trunk", 300) and not pol.admit("trunk", 300) and not pol.admit("sampler", 300)
    assert pol.events == {"trunk": 2, "sampler": 1} and events == [("trunk", 300), ("sampler", 300)] and pol.describe() == "budget:256"
    off = ckpt.GraphPolicy(policy="off")
    assert not off.admit("trunk", 1) and off.describe() == "off"
    assert ckpt.GraphPolicy().admit("trunk", 10 ** 6)
    with pytest.raises(ckpt.CkptError, match="policy budget needs budget"):
        ckpt.GraphPolicy(policy="budget", budget=0)
    with pytest.raises(ckpt.CkptError, match="policy must be"):
        ckpt.GraphPolicy(policy="maybe")


def test_graph_capture_is_typed_callable_only_and_recorded():
    installed = []
    env = {"ACME_BIG_GRAPH_CAPTURE_POLICY": "on", "ACME_BIG_GRAPH_CAPTURE_BUDGET": "1"}            # never read
    ctx = make_ctx(hooks={"graph_capture": {"switch": installed.append}}, settings={"graph_capture": {"policy": "budget", "budget": "300"}}, environ=env)
    rec = mem.apply(["graph_capture"], ctx, base="fast")
    a = rec.applied[0]
    assert a.exact == "measured" == a.declared_exact and "bitwise-claimed" in a.exact_reason and a.scope == "process"
    assert a.settings == {"policy": "budget", "budget": 300, "stock": False, "graphs": False}
    assert env == {"ACME_BIG_GRAPH_CAPTURE_POLICY": "on", "ACME_BIG_GRAPH_CAPTURE_BUDGET": "1"}             # no variable written
    pol = installed[0]
    assert isinstance(pol, ckpt.GraphPolicy) and pol.budget == 300 and pol is ctx.hooks["graph_capture"]["policy_object"]
    rec.unit_begin("item-0")
    assert pol.admit("trunk", 200) and not pol.admit("trunk", 400)
    rec.unit_end()
    ev = [e["detail"] for e in rec.units["process"].events]                 # a process-scope lever is marked on the process unit at apply, then per decision
    assert ev[0].startswith("applied (process scope)") and ev[1:] == ["trunk:200 captured", "trunk:400 eager"] and rec.units["process"].ran == ["graph_capture"]
    assert rec.units["item-0"].ran == [] and rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
    assert mem.undo(rec) == ["graph_capture"] and installed[-1].policy == "on"
    default = make_ctx(hooks={"graph_capture": {"switch": installed.append}})
    assert mem.apply(["graph_capture"], default, base="fast").applied[0].settings["policy"] == "off"

    def refusal(hooks, settings=None):
        rec = mem.apply(["graph_capture"], make_ctx(hooks=hooks, settings=settings or {}), base="fast", strict=False)
        return rec.refused[-1]

    assert refusal({}).precondition == "hooks.switch"
    assert refusal({"graph_capture": {"switch": "EF2_VAR"}}).precondition == "hooks.switch"
    r = refusal({"graph_capture": {"switch": installed.append}}, {"graph_capture": {"policy": "on"}})
    assert r.precondition == "policy" and "switches={'graph_capture': False}" in r.reason
    assert refusal({"graph_capture": {"switch": installed.append}}, {"graph_capture": {"policy": "budget"}}).precondition == "budget"
    assert refusal({"graph_capture": {"switch": installed.append, "on_event": 3}}).precondition == "hooks.on_event"


# ------------------------------------------------------------------------------------------------------ hoist_off


def test_hoist_off_drives_the_adapter_switch_and_restores_on_undo():
    state = {"hoist": True}

    def disable():
        state["hoist"] = False

        def restore():
            state["hoist"] = True
        return restore

    ctx = make_ctx(hooks={"hoist_off": {"disable": disable}})
    rec = mem.apply(["hoist_off"], ctx, base="fast")
    a = rec.applied[0]
    assert state["hoist"] is False and a.settings == {"hoist": "off", "stock": False} and a.undo is not None and "bitwise-claimed" in a.exact_reason
    assert a.exact == "measured" == a.declared_exact and a.scope == "process" and rec.units["process"].ran == ["hoist_off"] and a.notes == ["hoist cache disabled"]
    rec.unit_begin("item-0"); rec.unit_end()
    assert rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
    mem.undo(rec)
    assert state["hoist"] is True
    rec = mem.apply(["hoist_off"], make_ctx(hooks={"hoist_off": {"disable": lambda: None}}), base="fast")
    assert rec.applied[0].undo is None
    rec = mem.apply(["hoist_off"], make_ctx(hooks={"hoist_off": {"disable": "ACME_HOIST"}}), base="fast", strict=False)
    assert rec.refused[0].precondition == "hooks.disable"
    rec = mem.apply(["hoist_off"], make_ctx(), base="fast", strict=False)
    assert rec.refused[0].precondition == "hooks.disable"


# ------------------------------------------------------------------------------------------------------ the registry


def test_levers_are_registered_by_name_with_their_families_settings_and_hooks():
    assert set(registry.names(family="ckpt")) >= {"recycle_carry", "seed_batch"}
    assert "samples_per_pass" in registry.names(family="chunk") and set(registry.names(family="setting")) >= {"graph_capture", "hoist_off"}
    assert set(ckpt.HOOKS) == {"recycle_carry", "samples_per_pass", "seed_batch", "graph_capture", "hoist_off"}
    for name in ckpt.HOOKS:
        lv = registry.get(name)
        assert lv.module == "opt_core.mem.ckpt" and lv.frameworks == ("torch",) and lv.exact_reason and lv.describe().startswith(name)
        for hook in ckpt.HOOKS[name]:
            assert f"hooks.{hook}" in lv.preconditions or hook in ("generator", "device", "carry")
    assert registry.get("recycle_carry").settings == ("park", "pin_max_gb", "flush") and registry.get("samples_per_pass").settings == ("k", "draw_order")
    assert {n: registry.get(n).exact for n in ckpt.HOOKS} == {"recycle_carry": "bitwise", "samples_per_pass": "measured", "seed_batch": "measured",
                                                             "graph_capture": "measured", "hoist_off": "measured"}
    assert {n: registry.get(n).scope for n in ckpt.HOOKS} == {"recycle_carry": "unit", "samples_per_pass": "unit", "seed_batch": "unit",
                                                             "graph_capture": "process", "hoist_off": "process"}
    assert registry.setting_ref("graph_capture", "budget") == "ctx.settings['graph_capture']['budget']"
    disc = registry.discover()
    assert "ckpt" in disc["loaded"]



def test_graph_policy_decides_through_graph_gate():
    """GraphPolicy.admit is opt_core.mem.graph_gate.decide's capture bit; the decision object carries the gate's ONE wording."""
    from opt_core.mem import graph_gate
    from opt_core.mem.ckpt import GraphPolicy
    on, off, bud = GraphPolicy(policy="on"), GraphPolicy(policy="off"), GraphPolicy(policy="budget", budget=384)
    assert (on.cap, off.cap, bud.cap) == (None, 0, 384)
    for pol in (on, off, bud):
        for n in (1, 384, 385, 5000):
            d = pol.decision("pairformer", n)
            assert d.fragment == graph_gate.decide(n, pol.cap).fragment and pol.admit("pairformer", n) is d.capture
    assert bud.decision("s", 384).fragment == "graph=capture" and bud.decision("s", 385).reason == "n_tok>cap"
    assert off.decision("s", 1).fragment == "graph=off" and on.decision("s", 10 ** 6).reason == "no_cap"
    assert bud.events == {"pairformer": 2}                    # decision() alone records no event; admit() does
