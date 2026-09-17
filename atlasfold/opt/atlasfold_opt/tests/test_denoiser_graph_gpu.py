"""GPU tests of lever denoiser_graph (skipped without CUDA): the stock DiffusionHead / DiffusionModule built with tiny dimensions and random weights,
sampled once with the lever installed and once stock under the same seed — the graph route (capture at the first call, replay every step, ragged
sample chunks -> one graph per chunk shape) must give the identical bits, and the roll-out state must be gone when sample() returns."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _tiny_head_and_batch(device, B=1, L=24):
    from atlasfold.model.network.diffusion_head import DiffusionHead
    from atlasfold.model.network.rel_pos_encoding import RelativePositionEncoding, AtomRelativePositionEncoding
    torch.manual_seed(0)
    head = DiffusionHead(channel_a=64, channel_atom=16, channel_cond=32, channel_s=32, channel_z=16, num_heads=4, num_blocks=2,
                         num_atom_heads=2, num_atom_blocks=1, seq_rel_pos_bins=73, atom_rel_pos_bins=14).to(device).eval()
    for p in head.parameters():                                   # random, non-degenerate weights (several stock inits are zero / 'final')
        torch.nn.init.normal_(p, std=0.2)
    aa = torch.randint(0, 20, (B, L))
    batch = {"res_idx": torch.arange(L).unsqueeze(0).repeat(B, 1), "asym_id": torch.cat([torch.zeros(B, L // 2), torch.ones(B, L - L // 2)], 1).long(),
             "entity_id": torch.zeros(B, L, dtype=torch.long), "sym_id": torch.zeros(B, L, dtype=torch.long), "seq_mask": torch.ones(B, L, dtype=torch.bool),
             "aatype": torch.nn.functional.one_hot(aa, 21).float(), "aatype_int": aa, "atom14_mask": torch.rand(B, L, 14) > 0.3}
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        batch["seq_rel_pos"] = RelativePositionEncoding(r_max=32, s_max=2).to(device)({k: batch[k] for k in ("res_idx", "asym_id", "entity_id", "sym_id")})
        batch["atom_rel_pos"] = AtomRelativePositionEncoding(max_r=4).to(device)({k: batch[k] for k in ("res_idx", "asym_id", "seq_mask", "aatype")})
    s = torch.randn(B, L, 32, device=device); z = torch.randn(B, L, L, 16, device=device)
    return head, batch, s, z


def _sample(head, batch, s, z, num_samples, steps, chunk, seed=101, bf16=False):
    from atlasfold.model.network.diffusion_head import SamplingConfig
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=bf16):   # bf16=True: as under diffusion_bf16 in the fast row
        out = head.sample(batch, s, z, num_samples=num_samples, config=SamplingConfig(num_steps=steps, chunk_size=chunk))
    return out.float()


class _Count:
    """Counts the eager executions of a submodule's forward (a replay runs its kernels, not its Python)."""
    def __init__(self, module):
        self.n, self.module, self.inner = 0, module, module.forward
        module.forward = self

    def __call__(self, *a, **k):
        self.n += 1
        return self.inner(*a, **k)

    def restore(self):
        self.module.forward = self.inner


@cuda
@pytest.mark.parametrize("num_samples,chunk,bf16,B", [(3, 5, False, 1), (3, 2, False, 1), (3, 2, True, 1), (3, 2, True, 2)])
# one chunk shape | ragged chunks (2, 1) -> two graphs | under bf16 autocast (as under diffusion_bf16) | B=2 items per call: chunk views are non-dense
def test_graph_route_is_bitwise_to_the_eager_sampler_and_state_is_scoped(num_samples, chunk, bf16, B, monkeypatch):
    import atlasfold.model.network.diffusion_head as DH
    from atlasfold_opt.hooks import denoiser_graph as DG
    dev = torch.device("cuda")
    head, batch, s, z = _tiny_head_and_batch(dev, B=B)
    steps = 5
    enc = _Count(head.score_model.atom_encoder)                                  # a module INSIDE the captured call (where dit_sdpa / ln_bf16 sit)
    want = _sample(head, batch, s, z, num_samples, steps, chunk, bf16=bf16)      # stock
    assert torch.isfinite(want).all()
    calls = steps * -(-num_samples // chunk)
    assert enc.n == calls                                                        # eager: one execution per score_model call
    stock_forward, stock_sample = DH.DiffusionModule.forward, DH.DiffusionHead.sample
    monkeypatch.setenv("AFO_DENOISER_GRAPH_MAX_TOKENS", "1024")
    ins = DG.install("fast", "atlasfold-opt", {})
    try:
        assert ins.applied, ins.reason
        enc.n = 0
        got = _sample(head, batch, s, z, num_samples, steps, chunk, bf16=bf16)
        led = ins.facts["ledger"]
        f = led.facts()
        shapes = len({min(chunk, num_samples - st) for st in range(0, num_samples, chunk)})
        assert torch.equal(got, want), float((got - want).abs().max())
        assert led.fallbacks == {} and (f["captures"], f["replays"], f["eager"], f["recaptures"], f["rollouts"]) == (shapes, calls, 0, 0, 1), (f, led.fallbacks)
        assert enc.n == 2 * shapes                                               # graph on: the inner Python runs at warm-up + capture only, per graph
        assert getattr(head.score_model, DG.STATE_ATTR) is None                 # graphs, pool buffers, held refs dropped at sample() exit
        assert ins.gates[0]().ok
        line = ins.lines[0]()
        assert "state=on" in line and "captures=%d" % shapes in line and "replays=%d" % calls in line and "max_tokens=1024" in line, line
        torch.cuda.synchronize(); mem0 = torch.cuda.memory_allocated(); res0 = torch.cuda.memory_reserved()
        again = _sample(head, batch, s, z, num_samples, steps, chunk, bf16=bf16)   # a further roll-out leaves nothing behind (pool, static buffers, refs)
        torch.cuda.synchronize(); mem1 = torch.cuda.memory_allocated(); res1 = torch.cuda.memory_reserved()
        assert torch.equal(again, want) and abs(mem1 - mem0) <= again.numel() * 4 + (1 << 20), (mem0, mem1, res0, res1)
        assert led.facts()["captures"] == 2 * shapes and led.facts()["rollouts"] == 2
        # above the gate: every call eager by name, no capture, gate ok (size-gated), outputs still the stock bits
        led.clear()
        for k in ("captures", "replays", "eager", "recaptures", "rollouts"):
            led.set(k, 0)
        monkeypatch.setattr(DG, "gate_word", lambda r, limit: DG.ABOVE_GATE)
        # gate_word is consulted per call by RollOut.call; a fresh roll-out picks the patched function up
        got2 = _sample(head, batch, s, z, num_samples, steps, chunk, bf16=bf16)
        assert torch.equal(got2, want) and led.fallbacks == {"above_gate": calls} and led.facts()["captures"] == 0 and ins.gates[0]().ok
    finally:
        enc.restore()
        DH.DiffusionModule.forward, DH.DiffusionHead.sample = stock_forward, stock_sample


_CHILD = r"""
import torch
from atlasfold_opt.hooks import denoiser_graph as DG
from opt_core.counters import Ledger
dev = torch.device("cuda"); W = torch.randn(3, 3, device=dev)
class Bad(torch.nn.Module):
    def forward(self, batch, r_noisy, single_cond, pair_bias):
        y = r_noisy @ W
        return y * 2 if float(y.sum().item()) > 1e30 else y          # a host sync: fine eagerly (warm-up passes), illegal under stream capture
led = Ledger(DG.NAME, impl="t", origin="kit", expected=DG.EXPECTED)
ro = DG.RollOut(led, 1024)
batch = {"m": torch.ones(1, device=dev)}; pb = torch.zeros(1, device=dev)
r = torch.randn(1, 2, 8, 14, 3, device=dev); sc = torch.randn(1, 1, 8, 4, device=dev)
with torch.inference_mode():
    try:
        ro.call(Bad.forward, Bad(), batch, r, sc, pb)
        print("NO_RAISE")
    except RuntimeError as e:
        print("RAISED", "cannot continue after a failed stream capture" in str(e), type(e.__cause__).__name__)
ro.close()
print("FALLBACKS", dict(led.fallbacks), "ABORTED", ro.aborted, "GATE_OK", led.gate(require_served=False).ok, "ERR", led.facts().get("capture_error", "")[:40])
try:
    torch.randn(2, device=dev); print("RNG_AFTER usable")
except RuntimeError as e:
    print("RNG_AFTER unusable:", str(e).splitlines()[0][:80])
"""


@cuda
def test_a_capture_that_fails_ends_the_item_loudly_by_name():
    """In a child interpreter (a failed capture leaves the process's CUDA RNG / allocator capture state unusable — the reason the lever ends the item
    instead of continuing eagerly; the child prints which): the ORIGINAL error is named in capture_failed:<Exc> and chained, the gate refuses."""
    import subprocess, sys, os
    env = dict(os.environ); env.setdefault("PYTHONPATH", os.pathsep.join(sys.path))
    p = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env, timeout=300)
    out = p.stdout
    assert "RAISED True RuntimeError" in out, (out, p.stderr[-2000:])
    assert "FALLBACKS {'capture_failed:RuntimeError': 1} ABORTED True GATE_OK False ERR RuntimeError:" in out, out
    assert "RNG_AFTER" in out, out                                               # informational: 'unusable: Offset increment outside graph capture ...' on torch 2.7


@cuda
def test_static_inputs_change_every_step_and_replay_tracks_them():
    """RollOut on a plain CUDA function: 5 steps with a new r_noisy each step -> 1 capture, 5 replays, each bitwise to the eager statement."""
    from atlasfold_opt.hooks import denoiser_graph as DG
    from opt_core.counters import Ledger
    dev = torch.device("cuda")
    torch.manual_seed(0)
    W = torch.randn(3, 8, device=dev); V = torch.randn(8, 3, device=dev)

    class M(torch.nn.Module):
        def forward(self, batch, r_noisy, single_cond, pair_bias):
            h = torch.nn.functional.layer_norm(r_noisy @ W + single_cond[..., None, :8], (8,))
            return (torch.softmax(h, -1) @ V) * batch["atom14_mask"][:, None, :, :, None].float() + pair_bias.mean()
    m = M().to(dev)
    led = Ledger(DG.NAME, impl="t", origin="kit", expected=DG.EXPECTED)
    ro = DG.RollOut(led, 1024)
    batch = {"atom14_mask": torch.rand(1, 16, 14, device=dev) > 0.2}
    pb = torch.randn(2, 1, 16, 16, 4, device=dev)
    with torch.inference_mode():
        for step in range(5):
            r = torch.randn(1, 2, 16, 14, 3, device=dev); sc = torch.randn(1, 1, 16, 8, device=dev)
            want = M.forward(m, batch, r, sc, pb)
            got = ro.call(M.forward, m, batch, r, sc, pb)
            assert torch.equal(got, want), step
    assert led.facts()["captures"] == 1 and led.facts()["replays"] == 5 and led.fallbacks == {}
    ro.close()
    assert len(ro.graphs) == 0 and ro.closed
