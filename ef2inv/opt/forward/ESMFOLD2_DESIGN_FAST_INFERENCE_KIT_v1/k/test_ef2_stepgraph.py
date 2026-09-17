"""Unit tests for ef2_stepgraph (CUDA machine with the Biohub transformers fork; random-init trunks, no weights).

    pytest -q k/test_ef2_stepgraph.py        or        python k/test_ef2_stepgraph.py   (pytest-free)

The claim under test is BITWISE: a graphed trunk pass (forward replay + backward replay) returns tensor-equal outputs and tensor-equal
input gradients to the eager pass it captured — for the stock trunk (every block checkpointed under grad), for torch.compile'd blocks (the
cookbook's COMPILE recipe), for the cuEquivariance triangle backend when it is installed (the `exact` mode's trunk), and for the kit's
kernel levers when they are importable (the `fast` mode's trunk); across several design steps, two recycle slots chained like the model's
recycle, and a mask that changes between steps at a fixed shape.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_stepgraph as sg

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def make_trunk(n_layers=3, seed=0):
    torch.manual_seed(seed)
    trunk = C.FoldingTrunk(n_layers=n_layers)
    for m in trunk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    return trunk.to(dev).eval().requires_grad_(False)


class Wrapper(torch.nn.Module):
    """A stand-in for the model: .folding_trunk + a forward that runs the fork's recycle loop verbatim under autocast (forward step 5-6:
    ``z = z_init + pair_loop_proj(z); z = folding_trunk(z, mask)`` per pass).  The exactness condition of a per-pass segment is the model's:
    the trunk's input is a fresh sum consumed only by the trunk (an input ALSO consumed outside the segment would receive its inside
    contributions pre-summed — a different float association than eager's interleaved accumulation, i.e. last-bit differences)."""
    def __init__(self, trunk, n_passes=2):
        super().__init__()
        self.folding_trunk, self.n_passes = trunk, n_passes
        torch.manual_seed(7)
        self.pair_loop_proj = torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, 256)).to(dev).requires_grad_(False)

    def forward(self, z_init, mask):
        with torch.autocast(**AMP):
            z = torch.zeros_like(z_init)
            for _ in range(self.n_passes):
                z = z_init + self.pair_loop_proj(z)
                z = self.folding_trunk(z, pair_attention_mask=mask)
            return z


def step(model, z_init, mask, gout):
    zz = z_init.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = model(zz, mask)
        g, = torch.autograd.grad(out, zz, gout)
    return out.detach().clone(), g.detach().clone()


def _data(N=48, B=1, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(B, N, N, 256, device=dev, generator=g) * 4
    mask = torch.ones(B, N, N, device=dev); mask[:, -3:, :] = 0; mask[:, :, -3:] = 0
    gout = torch.randn(B, N, N, 256, device=dev, generator=g).to(torch.bfloat16)
    return z, mask, gout


def _check_bitwise(model, n_steps=3, N=48, B=1):
    z, mask, gout = _data(N, B)
    ref = [step(model, z, mask, gout) for _ in range(2)]
    assert torch.equal(ref[0][0], ref[1][0]) and torch.equal(ref[0][1], ref[1][1]), "eager itself is not repeatable on this stack (test premise)"
    fwd_before = vars(model.folding_trunk).get("forward")      # None = the class forward; else an instance patch (a kit policy)
    pool = sg.enable(model, n_slots=model.n_passes)
    try:
        for _ in range(n_steps):
            o, g = step(model, z, mask, gout)
            st = sg.stats(model)
            assert st["disabled_reason"] is None, f"capture fell back to eager: {st['disabled_reason']}"
            assert torch.equal(o, ref[0][0]), f"graphed forward differs from eager (max |d| {(o.float() - ref[0][0].float()).abs().max().item():.3e})"
            assert torch.equal(g, ref[0][1]), f"graphed input-gradient differs from eager (max |d| {(g.float() - ref[0][1].float()).abs().max().item():.3e}, max |g| {ref[0][1].float().abs().max().item():.3e})"
        st = sg.stats(model)
        assert st["captures"] == model.n_passes and st["recaptures"] == 0 and st["disabled_reason"] is None, st
        assert st["replays_fwd"] == n_steps * model.n_passes and st["replays_bwd"] == n_steps * model.n_passes, st
        # a different mask at the same shape must replay with THAT mask (not the captured one)
        mask2 = mask.clone(); mask2[:, :5, :] = 0
        sg.disable(model); o_e, g_e = step(model, z, mask2, gout); sg.enable(model, n_slots=model.n_passes)
        step(model, z, mask, gout)                       # capture with mask ...
        o2, g2 = step(model, z, mask2, gout)             # ... replay with mask2
        assert torch.equal(o2, o_e) and torch.equal(g2, g_e), "mask is not a live input of the captured graph"
    finally:
        sg.disable(model)
    assert getattr(model, "_sg", None) is None and vars(model.folding_trunk).get("forward") is fwd_before, "disable did not restore the trunk's forward attribute"
    o3, g3 = step(model, z, mask, gout)
    assert torch.equal(o3, ref[0][0]) and torch.equal(g3, ref[0][1]), "disable did not restore the eager trunk"


def test_stock_trunk_checkpointed_bitwise():
    _check_bitwise(Wrapper(make_trunk(3)))


def test_batch2_bitwise():
    _check_bitwise(Wrapper(make_trunk(2)), N=40, B=2)


def test_compiled_blocks_bitwise():
    """The cookbook's COMPILE recipe: PairUpdateBlock.forward = torch.compile(forward) — AOTAutograd backward with donated buffers."""
    trunk = make_trunk(3)
    torch._dynamo.config.cache_size_limit = 512
    for b in trunk.blocks:
        b.forward = torch.compile(b.forward)
    _check_bitwise(Wrapper(trunk))


def test_cuequivariance_backend_compiled_bitwise():
    """The `exact` mode's trunk: cuEquivariance triangle kernels + compiled blocks, pair stack unchunked."""
    try:
        import cuequivariance_torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("SKIP (cuequivariance_torch not importable):", e); return
    trunk = make_trunk(3)
    trunk.set_kernel_backend(C.BACKEND_CUEQ); trunk.set_chunk_size(None)
    for b in trunk.blocks:
        b.forward = torch.compile(b.forward)
    _check_bitwise(Wrapper(trunk), N=64)


def test_kit_kernels_bitwise():
    """The `fast` mode's trunk: the kit's K-A2 / K-D3 kernels + a selective checkpoint policy (when ef2_autograd_kernels imports here)."""
    try:
        import ef2_autograd_kernels as agk
    except Exception as e:  # noqa: BLE001
        print("SKIP (ef2_autograd_kernels not importable):", e); return
    trunk = make_trunk(3)
    trunk.set_kernel_backend(None); trunk.set_chunk_size(None)
    agk.enable(trunk, trimul="bmm2", transition="refround", checkpoint="ckpt:1", only_under_grad=False)
    _check_bitwise(Wrapper(trunk), N=64)


def test_too_many_passes_named():
    model = Wrapper(make_trunk(1), n_passes=2)
    sg.enable(model, n_slots=1)
    z, mask, gout = _data(32)
    try:
        raised = None
        try:
            step(model, z, mask, gout)
        except RuntimeError as e:
            raised = str(e)
        # the pool names the defect and falls back to eager (disabled_reason set), it does not corrupt anything
        st = sg.stats(model)
        assert (raised and "n_slots" in raised) or (st["disabled_reason"] and "n_slots" in st["disabled_reason"]), (raised, st)
    finally:
        sg.disable(model)


if __name__ == "__main__":
    import traceback
    if dev.type != "cuda":
        print("CUDA required"); sys.exit(2)
    n_pass = n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); n_pass += 1; print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1; print("FAIL", name, type(e).__name__, str(e)[:400], flush=True); traceback.print_exc()
    print(f"RESULT passed={n_pass} failed={n_fail}")
    sys.exit(1 if n_fail else 0)
