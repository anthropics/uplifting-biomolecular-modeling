"""lever 'sbatch' (the sample-batched diffusion sampler): its RNG plan reproduces stock's serial draws draw for draw (CPU generator: the
state walk; CUDA Philox: the offset plan), the batched loop over a linear stand-in denoiser is bitwise the serial loop's (so every difference
under the real denoiser is the batched kernels' reduction order, nothing else), canonical_noise's draw length rides into it, and the lever is
wired through the kit's tables (LEVER_SETS / registry / the n_gpu gate) like every other lever."""
import os
import types

import pytest

from af3_torch_opt import big, canonical_noise, modes, registry, stack


def _xfold():
    torch = pytest.importorskip("torch")
    import sys
    kit = os.path.join(stack.forward_dir(), "af3t", "af3_torch")        # the xfold package only: the kernels / DTK dirs stay OFF this process's path (a routed
    if kit not in sys.path:                                              # kernel name resolvable from a kit copy would change what the route gate sees)
        sys.path.insert(0, kit)
    from xfold import alphafold3, of3  # noqa: F401
    from xfold.nn import diffusion_head
    return torch, alphafold3, diffusion_head


def _serial_draws(torch, dev, S, T, rows, atoms):
    """stock's stream: the initial draw, then per sample per step (rotation 2x3, translation 3, noise)."""
    init = torch.randn((S, rows) + atoms + (3,), device=dev)
    units = {}
    for s in range(S):
        for t in range(T):
            units[(s, t)] = (torch.randn(size=(2, 3), dtype=torch.float32, device=dev), torch.randn(size=(3,), dtype=torch.float32, device=dev),
                             torch.randn(size=(rows,) + atoms + (3,), device=dev))
    after = torch.randn(11, device=dev)
    return init, units, after


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_draw_plan_reproduces_stocks_stream_in_batched_order(device):
    torch, _, DH = _xfold()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device(device, 0) if device == "cuda" else torch.device("cpu")
    S, T, rows, atoms = 3, 4, 7, (5,)

    def seed():
        torch.manual_seed(1234)
        if device == "cuda":
            torch.cuda.manual_seed_all(1234)
    seed(); init0, units0, after0 = _serial_draws(torch, dev, S, T, rows, atoms)
    seed()
    plan = DH.DrawPlan(dev, S, T, rows, atoms, torch.float32)
    init1 = plan.initial()
    assert plan.kind == ("philox_offsets" if device == "cuda" else "state_walk"), plan.kind
    got = {}
    for t in range(T):                       # BATCHED order: every sample of a step, then the next step
        for s in range(S):
            plan.seek(s, t)
            got[(s, t)] = (torch.randn(size=(2, 3), dtype=torch.float32, device=dev), torch.randn(size=(3,), dtype=torch.float32, device=dev),
                           torch.randn(size=(rows,) + atoms + (3,), device=dev))
    plan.finish()
    after1 = torch.randn(11, device=dev)
    assert torch.equal(init0, init1)
    for k in units0:
        for a, b in zip(units0[k], got[k]):
            assert torch.equal(a, b), k
    assert torch.equal(after0, after1)       # the generator is left where stock's loop leaves it


class _LinearDenoiser:
    """A stand-in diffusion head: elementwise arithmetic only (so batch order cannot change a bit), shape-polymorphic like the real one."""
    use_hoist = False
    use_step_graph = False

    def __call__(self, positions_noisy, noise_level, batch, embeddings, use_conditioning=True):
        return positions_noisy * (16.0 ** 2 / (noise_level ** 2 + 16.0 ** 2)) + 0.001 * torch_sin(positions_noisy)


def torch_sin(x):
    import torch
    return torch.sin(x)


def _stub_model(torch, alphafold3, num_samples, steps, n_tok, atoms=4, sample_batch=0):
    m = types.SimpleNamespace()
    m.num_samples, m.diffusion_steps = num_samples, steps
    m.gamma_0, m.gamma_min, m.noise_scale, m.step_scale = 0.8, 1.0, 1.003, 1.5
    m.diffusion_head = _LinearDenoiser()
    m.sample_batch = sample_batch
    m._sbatch_counts = {"batched_calls": 0, "single_calls": 0, "trajectories": 0, "chunk": None, "rng": None}
    for name in ("_sample_diffusion", "_sample_diffusion_batched", "_apply_denoising_step", "_apply_denoising_step_batched", "_draw_rows"):
        setattr(m, name, types.MethodType(getattr(alphafold3.AlphaFold3, name), m))
    mask = (torch.rand(n_tok, atoms) > 0.2).float()
    batch = types.SimpleNamespace(predicted_structure_info=types.SimpleNamespace(atom_mask=mask),
                                  token_features=types.SimpleNamespace(seq_length=torch.tensor([n_tok])))
    return m, batch


@pytest.mark.parametrize("S,sub", [(1, 1), (3, 3), (3, 1), (5, 2)])
def test_batched_loop_is_the_serial_loop_on_a_linear_denoiser(S, sub):
    torch, AF, _ = _xfold()
    m, batch = _stub_model(torch, AF, S, 6, 9)
    torch.manual_seed(7); serial = m._sample_diffusion(batch, {})["atom_positions"].clone(); after0 = torch.randn(5)
    m.sample_batch = sub
    torch.manual_seed(7); batched = m._sample_diffusion(batch, {})["atom_positions"].clone(); after1 = torch.randn(5)
    assert serial.shape == batched.shape == (S, 9, 4, 3)
    assert torch.equal(serial, batched)                      # draw for draw, statement for statement
    assert torch.equal(after0, after1)
    c = m._sbatch_counts
    per_step = -(-S // sub)
    assert c["trajectories"] == S and c["chunk"] == min(sub, S) and c["rng"] == "state_walk"
    assert c["batched_calls"] + c["single_calls"] == 6 * per_step and c["batched_calls"] == 6 * sum(1 for s0 in range(0, S, sub) if min(S, s0 + sub) - s0 > 1)


def test_canonical_noise_rides_into_the_batched_sampler():
    """canonical_noise installed + sbatch: the draws are made at the input's real token count C and laid into n rows (rows C.. get no draw and,
    masked, never move), the census books the trajectories, and the generator advances exactly as a model of length C advances it (every
    draw made at C rows: the run at n leaves the stream where the run at C leaves it)."""
    torch, AF, _ = _xfold()
    n, C, S = 9, 6, 2
    m, batch = _stub_model(torch, AF, S, 5, n, sample_batch=1 << 16)
    batch.token_features.seq_length = torch.tensor([C])
    batch.predicted_structure_info.atom_mask[:C] = 1.0; batch.predicted_structure_info.atom_mask[C:] = 0.0   # padding tokens carry no atom
    canonical_noise.take(); canonical_noise.install(m)
    torch.manual_seed(3); padded = m._sample_diffusion(batch, {})["atom_positions"].clone(); tail_padded = torch.randn(13)
    cn = canonical_noise.take()
    assert (cn["trajectories"], cn["model_len"], cn["canonical_len"]) == (S, n, C)
    assert padded.shape == (S, n, 4, 3) and not padded[:, C:].any()
    m2, batch2 = _stub_model(torch, AF, S, 5, C, sample_batch=1 << 16)                                     # the same input at its own length (no padding)
    batch2.predicted_structure_info.atom_mask[:] = 1.0
    canonical_noise.install(m2)
    torch.manual_seed(3); own = m2._sample_diffusion(batch2, {})["atom_positions"].clone(); tail_own = torch.randn(13)
    assert own.shape == (S, C, 4, 3)
    assert torch.equal(tail_padded, tail_own)
    assert torch.allclose(padded[:, :C], own, atol=1e-4, rtol=1e-4)      # same draws on the real rows (the masked centre of the augmentation is the same atoms': equal up to fp order)


def test_lever_is_wired_through_the_kits_tables():
    sets = modes.kit_lever_sets()
    assert "sbatch" in sets["fastest"] and sets["fastest"].index("sbatch") > sets["fastest"].index("stepgraph")   # the batched step is captured whole: after the graph lever in build order
    assert "sbatch" in registry.LEVERS and "sbatch" in registry.NOT_BITWISE and "sbatch" not in registry.EXACT
    assert registry.EVIDENCE["sbatch"] == "applied" and registry.STRATEGY["sbatch"].startswith("LOCAL.af3_torch.") and registry.IMPL["sbatch"][1] == "kit"
    assert registry.KERNEL_ROUTES["apb_attn"]["levers"] == ("sbatch", "apb") and any(t.startswith(registry.CORE) for t in registry.LEVERS["sbatch"]["touches"])   # the core's own pair-bias attention kernel: the batched step's direct entry, and a row its provider serves lever apb by the tier word
    fast = modes.resolve("fast"); big = modes.resolve("big"); exact = modes.resolve("exact")
    assert "sbatch" in fast["levers"] and "sbatch" in big["levers"] and "sbatch" not in exact["levers"]      # fast and big carry it; exact keeps the serial sampler (reduction order)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forward.py"), encoding="utf-8").read()
    assert '("sbatch", "skipped:rowpair_sampler")' in src
    rpx = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rowpair_xfold.py"), encoding="utf-8").read()
    assert "model.sample_batch = 0" in rpx and '_lever("sbatch", "skipped:rowpair_sampler")' in rpx


def test_build_model_source_sets_the_switch():
    src = open(os.path.join(stack.forward_dir(), "af3t", "af3_torch", "af3_torch_api.py"), encoding="utf-8").read()
    assert 'model.sample_batch = SAMPLE_BATCH_ALL if "sbatch" in levers else 0' in src


_GPU_SCRIPT = r"""
import sys, torch
sys.path.insert(0, DTK); sys.path.insert(0, CORE)
from opt_core.kernels import route
route("dtk_kernels"); route("apb_attn")
import dtk_modules as M
dev = torch.device("cuda", 0); g = torch.Generator().manual_seed(0)
nb, c, c_s, h, N, S = 4, 768, 384, 16, 448, 5
W = M.DiTWeights(nb, c, c_s, h, dev, g)
f = M.FusedDiT(W, torch.bfloat16, "xfold")
a = torch.randn(S, N, c, generator=g).to(dev); s = torch.randn(N, c_s, generator=g).to(dev)
bias = (0.5 * torch.randn(nb, h, N, N, generator=g)).to(dev).to(torch.bfloat16); km = torch.ones(N, device=dev); km[-37:] = 0
serial = torch.stack([f.forward(a[j].clone(), s, bias, km) for j in range(S)])
f.batched_attn = "flash_bias_attn:per_sample"
one = f.forward(a[:1].clone(), s, bias, km)
per = f.forward(a.clone(), s, bias, km)
f2 = M.FusedDiT(W, torch.bfloat16, "xfold"); f2.batched_attn = "apb_attn"
apb = f2.forward(a.clone(), s, bias, km)
scale = float(serial.abs().max())
W64 = M.DiTWeights(nb, c, c_s, h, dev, torch.Generator().manual_seed(0))          # the same weights (same generator seed) for the fp64 reference of the stock formulation
ref = torch.stack([M.transformer_ref("ref64", a[j].double(), s.double(), bias.double(), W64, torch.float64) for j in range(S)]).float()
e_serial, e_apb, e_per = (float((x - ref).abs().max()) for x in (serial, apb, per))
print("RESULT", int(torch.equal(one[0], serial[0])), f.batched_route, f2.batched_attn, f2.batched_attn_event,
      float((per - serial).abs().max()) / scale, float((apb - serial).abs().max()) / scale, e_serial / scale, e_apb / scale, e_per / scale)
"""


@pytest.mark.skipif(not os.environ.get("AF3T_GPU_TESTS"), reason="GPU test: set AF3T_GPU_TESTS=1 on a CUDA box (runs in its own interpreter: routing a kernel in this process would change what later tests resolve)")
def test_fused_dit_batched_matches_serial_on_gpu():
    """FusedDiT at batch S vs S serial calls, in a subprocess of this interpreter: a chunk of one on the per-sample flash route is bitwise the
    serial call; the periodic-row schedule with the per-sample flash attention and with the one-launch apb_attn kernel are the serial class
    (max |diff| relative to the activations' scale small: bf16 reduction order)."""
    import subprocess
    import sys
    script = _GPU_SCRIPT.replace("DTK", repr(os.path.join(stack.forward_dir(), "dtk"))).replace("CORE", repr(stack.core_dir()))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert r.returncode == 0 and line, (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    _, bit, route_, attn, event, rel_per, rel_apb, e_serial, e_apb, e_per = line[-1].split(maxsplit=9)
    assert bit == "1", line
    assert route_ == "periodic_rows" and attn and not str(attn).startswith("flash_bias_attn"), line   # the arm the shared core's provider served for the tier word (apb_attn | fpf_apb | l3a | sdpa:auto ...), or apb_attn directly
    assert float(rel_per) <= 2e-2 and float(rel_apb) <= 2e-2, line
    assert float(e_apb) <= 1.25 * float(e_serial) and float(e_per) <= 1.25 * float(e_serial), line   # op level: the batched step's error vs the fp64 reference is the serial step's class (<= 1.25x its max |err|)
