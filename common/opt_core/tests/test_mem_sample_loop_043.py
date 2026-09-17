"""opt_core.mem.sample_loop on CPU with a stub diffusion roll-out + confidence head: chunked == batched torch.equal at samples in
{1, 2, 5} x chunk in {1, 2}, the RNG end state equal to the batched pass's, the alloc census (no all-samples tensor alive while the
confidence head runs; the head sees `chunk` samples), the fields, and every refusal by name (chunk > samples, needs_cross_sample,
identical chunks); the registered lever through opt_core.mem.Ctx.

The stub's statements are IEEE basic arithmetic, indexing and sqrt only (correctly rounded in SIMD and scalar code alike), so the test
isolates the SEAM's exactness (draws, slicing, assembly). Transcendental elementwise kernels and reductions are deliberately absent: their
CPU vectorisation is position-dependent (an element in a SIMD body vs a scalar tail), the same batch-dependence class a GPU stack's equality
row measures for the engine's real kernels."""
import weakref

import pytest

torch = pytest.importorskip("torch")

from opt_core import mem  # noqa: E402
from opt_core.mem import registry, sample_loop as SL  # noqa: E402

B, A, T = 1, 40, 6            # batch, atoms, denoising steps
N_TOK = 10                    # "tokens" of the stub confidence head (first N_TOK atoms)
SAMPLE_DIM = 1


def _schedule():
    return [4.0 * (0.7 ** i) for i in range(T + 1)]


def stub_rollout(S: int, weights, live_refs=None):
    """AF3-Alg.18-shaped loop over samples [chunk): init noise, per-step augmentation draw + noise draw, an elementwise 'denoiser'.
    Every batch-shaped draw is made at the STOCK shape [B, S, ...] through draws.draw and sliced to the chunk."""
    w, b = weights

    def stock_draw(shape):
        def fn():
            t = torch.randn(shape)
            if live_refs is not None:
                live_refs.append(weakref.ref(t))
            return t
        return fn

    def rollout(chunk, draws):
        sched = _schedule()
        x = sched[0] * draws.draw(stock_draw((B, S, A, 3)), chunk)
        for tau in range(T):
            rot = draws.draw(stock_draw((B, S, 1, 3)), chunk)                       # the augmentation draw (per sample)
            x = x - x[..., :1, :] + 0.1 * rot
            t_hat = sched[tau] * 1.2
            noise = (t_hat ** 2 - sched[tau] ** 2) ** 0.5 * draws.draw(stock_draw((B, S, A, 3)), chunk)
            x_noisy = x + noise
            u = x_noisy * w + b
            x_denoised = x_noisy * u / (1.0 + u * u)                                  # rational elementwise: IEEE-exact in SIMD and scalar
            delta = (x_noisy - x_denoised) / t_hat
            x = x_noisy + 1.5 * (sched[tau + 1] - t_hat) * delta
        return x
    return rollout


def stub_confidence(seen_chunks=None, live_refs=None, S=None):
    def confidence(x, chunk):
        if seen_chunks is not None:
            seen_chunks.append(int(x.shape[SAMPLE_DIM]))
        if live_refs is not None:
            alive = [r for r in live_refs if r() is not None]
            assert not alive, f"{len(alive)} all-samples draw(s) [B, {S}, ...] alive while the confidence head runs"
        xt = x[..., :N_TOK, :]                                                        # representative atoms
        z = xt[..., :, None, :] - xt[..., None, :, :]                                 # [B, k, n, n, 3] per-sample "pair rep"
        d = (z[..., 0] * z[..., 0] + z[..., 1] * z[..., 1] + z[..., 2] * z[..., 2] + 1e-8).sqrt()
        return {"atom_positions_predicted": x, "plddt": d[..., 0, :], "pae_logits": torch.stack([d, -d], dim=-1),
                "nested": {"ptm": (d[..., 0, 1] + d[..., 1, 2])[..., None]}}
    return confidence


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((3,), generator=g), torch.randn((3,), generator=g)


def _batched(S, seed=1234):
    """The stock pass: one chunk of every sample (plan chunk == S), same seed."""
    torch.manual_seed(seed)
    out = SL.run(SL.plan(S, S), stub_rollout(S, _weights()), stub_confidence(), sample_dim=SAMPLE_DIM, host=True)
    return out, torch.get_rng_state()


@pytest.mark.parametrize("S", [1, 2, 5])
@pytest.mark.parametrize("chunk", [1, 2])
def test_chunked_equals_batched_bitwise_and_rng_ends_where_one_batched_pass_ends(S, chunk):
    torch.set_num_threads(1)
    if chunk > S:
        with pytest.raises(SL.SampleLoopRefused) as e:
            SL.plan(S, chunk)
        assert e.value.reason.startswith(SL.REFUSE_CHUNK)
        return
    ref, ref_state = _batched(S)
    torch.manual_seed(1234)
    live, seen = [], []
    p = SL.plan(S, chunk)
    out = SL.run(p, stub_rollout(S, _weights(), live_refs=live), stub_confidence(seen_chunks=seen, live_refs=live, S=S), sample_dim=SAMPLE_DIM, host=True)
    assert p.passes == -(-S // chunk) and seen == [c.stop - c.start for c in p.chunks]
    assert torch.equal(torch.get_rng_state(), ref_state), "the generator must end where ONE batched pass ends"
    for key in ("atom_positions_predicted", "plddt", "pae_logits"):
        assert out.value[key].shape == ref.value[key].shape
        assert torch.equal(out.value[key], ref.value[key]), key
    assert torch.equal(out.value["nested"]["ptm"], ref.value["nested"]["ptm"])
    assert out.value["atom_positions_predicted"].shape[SAMPLE_DIM] == S
    f = out.fields()
    assert (f["samples"], f["sample_chunk"], f["sample_passes"], f["draw_order"]) == (S, chunk, p.passes, "stock")
    assert f["sample_chunk_source"] == ("one_pass" if chunk == S else "given") and f["sample_host"] in ("pageable", "pinned")
    assert out.record["census"]["consistent"] and out.record["census"]["end_states_equal"]
    assert out.record["census"]["draws_per_chunk"] == [1 + 2 * T] * p.passes


def test_default_chunk_is_one_and_one_pass_is_named():
    p = SL.plan(5)
    assert (p.chunk, p.source, p.passes) == (1, "default", 5)
    p = SL.plan(3, 3)
    assert (p.source, p.passes) == ("one_pass", 1)
    p = SL.plan(1)
    assert (p.chunk, p.source, p.passes) == (1, "one_pass", 1)


def test_refusals_are_by_name():
    with pytest.raises(SL.SampleLoopRefused) as e:
        SL.plan(2, 3)
    assert e.value.reason.startswith(SL.REFUSE_CHUNK) and e.value.lever == SL.LEVER
    with pytest.raises(SL.SampleLoopRefused) as e:
        SL.plan(5, 1, needs_cross_sample="sample-averaged pair logits")
    assert e.value.reason.startswith(SL.REFUSE_CROSS_SAMPLE)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(SL.SampleLoopRefused):
            SL.plan(4, bad)
    with pytest.raises(SL.SampleLoopRefused):
        SL.plan(0)
    with pytest.raises(SL.SampleLoopRefused):
        SL.plan(4, 1, draw_order="backwards")
    assert isinstance(SL.SampleLoopRefused("x"), mem.MemLeverRefused)


def test_identical_chunks_are_refused_by_name():
    """A roll-out that ignores `draws` (chunk-shaped draws of its own after the rewind) repeats samples: refused, never returned."""
    torch.manual_seed(7)

    def bad_rollout(chunk, draws):
        k = chunk.stop - chunk.start
        return torch.randn((B, k, A, 3))                     # NOT through draws.draw: the rewind makes every chunk draw the same numbers
    with pytest.raises(SL.SampleLoopRefused) as e:
        SL.run(SL.plan(3, 1), bad_rollout, None, sample_dim=SAMPLE_DIM)
    assert e.value.reason.startswith(SL.REFUSE_IDENTICAL)
    # the same roll-out under draw_order=chunked (no rewind) yields distinct samples: allowed (a named, band-class order)
    out = SL.run(SL.plan(3, 1, draw_order="chunked"), bad_rollout, None, sample_dim=SAMPLE_DIM)
    assert out.value.shape == (B, 3, A, 3) and out.fields()["draw_order"] == "chunked"


def test_draw_outside_draws_is_the_end_state_gate():
    """A draw made outside draws.draw that advances the generator differently per chunk: opt_core.mem.ckpt's end-state gate fires
    (CkptError). (A stray draw of EQUAL count per chunk passes that gate; the identical-chunks gate covers the repeat-samples case.)"""
    from opt_core.mem import ckpt
    torch.manual_seed(3)

    def leaky_rollout(chunk, draws):
        x = draws.draw(lambda: torch.randn((B, 3, A, 3)), chunk)
        return x + 0.01 * torch.randn(chunk.start + 1).sum()            # a stray draw whose count differs per chunk
    with pytest.raises(ckpt.CkptError):
        SL.run(SL.plan(3, 1), leaky_rollout, None, sample_dim=SAMPLE_DIM)


def test_registered_lever_through_ctx_applies_refuses_and_runs():
    assert SL.LEVER in registry.LEVERS and registry.LEVERS[SL.LEVER].family == "chunk"
    S = 4
    torch.manual_seed(11)
    hooks = {SL.LEVER: {"n_samples": S, "rollout": stub_rollout(S, _weights()), "confidence": stub_confidence(), "sample_dim": SAMPLE_DIM}}
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch", hooks=hooks, settings={SL.LEVER: {"chunk": 2}}, environ={})
    assert SL._applies(ctx) is None
    a = SL.sample_loop(ctx)
    assert a.lever == SL.LEVER and a.settings["chunk"] == 2 and a.settings["passes"] == 2
    bad = mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch", hooks=hooks, settings={SL.LEVER: {"chunk": 9}}, environ={})
    r = SL._applies(bad)
    assert r is not None and SL.REFUSE_CHUNK in str(r)
    flag = mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch", hooks=hooks, settings={SL.LEVER: {"chunk": "1"}}, environ={"ACME_BIG_SAMPLE_LOOP_CHUNK": "9"})   # the kit's string, cast; env never read
    assert SL.sample_loop(flag).settings["chunk"] == 1
