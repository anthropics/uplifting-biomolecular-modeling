"""Unit tests for ef2_esmc_rope (CUDA + the Biohub transformers fork; random tensors / random-init modules, no checkpoint).

    pytest -q k/test_ef2_esmc_rope.py        or, without pytest:   python k/test_ef2_esmc_rope.py

The class under test is EXACT: rotated q/k AND the input gradients must be tensor-equal (torch.equal) to the fork's torch
RoPE branch (`_flash_attn_rotary_available` False — the opt-in upstream RoPE fix's state) at the design loop's shapes; inside a
small ESMC stack the last hidden state and d(out)/d(inputs_embeds) must be tensor-equal with and without the lever; calls the
lever does not claim (flash-attn Triton branch bound, interleaved layout) must reach the stock forward and be counted by name.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    import pytest
except ImportError:                                   # the kit's pytest-free runner style: only mark.skipif / mark.parametrize are used
    pytest = None
import torch
import transformers.models.esmc.modeling_esmc as ME
import ef2_esmc_rope as eer

_skipif = pytest.mark.skipif if pytest else (lambda cond, reason="": (lambda f: f))
_param = pytest.mark.parametrize if pytest else None
cuda = _skipif(not torch.cuda.is_available(), reason="CUDA required")
SHAPES = [(4, 82, 40, 64), (1, 199, 40, 64), (1, 435, 40, 64), (1, 704, 40, 64), (2, 50, 8, 64), (3, 17, 5, 128)]


class _TorchBranch:
    """Pin the fork's torch RoPE branch (what the opt-in upstream RoPE fix does) for the duration of a test."""
    def __enter__(self):
        self.prev = ME._flash_attn_rotary_available; ME._flash_attn_rotary_available = False; return self
    def __exit__(self, *a):
        ME._flash_attn_rotary_available = self.prev


class _Holder(torch.nn.Module):
    def __init__(self, rot):
        super().__init__(); self.rot = rot


def _run(mod, q0, k0, wq, wk):
    q = q0.detach().clone().requires_grad_(True); k = k0.detach().clone().requires_grad_(True)
    oq, ok = mod(q, k)
    ((oq * wq).sum() + (ok * wk).sum()).backward()
    return oq.detach(), ok.detach(), q.grad.detach(), k.grad.detach()


def _check_shape(B, S, H, D, dtype=torch.bfloat16, seed=0, strided=True):
    torch.manual_seed(seed)
    rot = ME.RotaryEmbedding(D).cuda()
    holder = _Holder(rot)
    if strided:      # q, k as the attention hands them over: chunks of one [B, S, 3*H*D] projection, unflattened — non-contiguous rows
        qkv = torch.randn(B, S, 3 * H * D, device="cuda", dtype=dtype)
        q0, k0, _ = (t.unflatten(-1, (H, D)) for t in qkv.chunk(3, dim=-1))
    else:
        q0 = torch.randn(B, S, H, D, device="cuda", dtype=dtype); k0 = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    wq = torch.randn(B, S, H, D, device="cuda", dtype=dtype); wk = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    with _TorchBranch():
        ref = _run(rot, q0, k0, wq, wk)
        h = eer.enable(holder)
        try:
            got = _run(rot, q0, k0, wq, wk)
            assert h.stats["served"] == 1 and not h.stats["fallback"], h.stats
        finally:
            eer.disable(holder)
        assert "forward" not in vars(rot)
        again = _run(rot, q0, k0, wq, wk)
    for name, r, g, a in zip(("q_rot", "k_rot", "dq", "dk"), ref, got, again):
        assert r.shape == g.shape and r.dtype == g.dtype, (name, r.shape, g.shape, r.dtype, g.dtype)
        assert torch.equal(r, g), f"{name} not bitwise at {(B, S, H, D)} {dtype}: max|d|={(r.float() - g.float()).abs().max().item():.3e}"
        assert torch.equal(r, a), f"{name}: disable() did not restore the stock forward"


if _param:
    @cuda
    @_param("B,S,H,D", SHAPES)
    def test_bitwise_equal_to_torch_branch_bf16(B, S, H, D):
        _check_shape(B, S, H, D, torch.bfloat16)
else:
    def test_bitwise_equal_to_torch_branch_bf16():
        for B, S, H, D in SHAPES:
            _check_shape(B, S, H, D, torch.bfloat16)


@cuda
def test_bitwise_equal_to_torch_branch_fp16_fp32_and_contiguous_inputs():
    _check_shape(2, 33, 6, 64, torch.float16)
    _check_shape(2, 33, 6, 64, torch.float32)
    _check_shape(1, 82, 40, 64, torch.bfloat16, strided=False)


@cuda
def test_unclaimed_calls_reach_stock_forward_and_are_named():
    torch.manual_seed(1)
    q = torch.randn(1, 9, 2, 64, device="cuda", dtype=torch.bfloat16); k = torch.randn_like(q)
    rot = ME.RotaryEmbedding(64, interleaved=True).cuda(); holder = _Holder(rot)
    with _TorchBranch():
        ref = rot(q, k)
        h = eer.enable(holder)
        got = rot(q, k)
        eer.disable(holder)
    assert h.stats == {"served": 0, "fallback": {"interleaved": 1}}, h.stats
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    if ME.apply_triton_rotary is not None:            # flash-attn's Triton rotary importable: the as-shipped branch is left alone
        rot = ME.RotaryEmbedding(64).cuda(); holder = _Holder(rot)
        prev = ME._flash_attn_rotary_available; ME._flash_attn_rotary_available = True
        try:
            ref = rot(q, k); h = eer.enable(holder); got = rot(q, k); eer.disable(holder)
        finally:
            ME._flash_attn_rotary_available = prev
        assert h.stats == {"served": 0, "fallback": {"flash_triton_branch": 1}}, h.stats
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@cuda
def test_enable_is_idempotent_and_finds_every_rotary_of_a_stack():
    cfg = ME.ESMCConfig(d_model=128, n_heads=4, n_layers=3, vocab_size=64)
    torch.manual_seed(2)
    esmc = ME.ESMCModel(cfg).to("cuda", torch.bfloat16).eval().requires_grad_(False)
    h1 = eer.enable(esmc); h2 = eer.enable(esmc)
    assert h1 is h2 and len(h1.modules) == cfg.n_layers, (len(h1.modules), cfg.n_layers)
    lm = torch.nn.Module(); lm.esmc = esmc                      # the ESMCForMaskedLM / ESMFold2 (._esmc) spellings resolve to the same trunk
    assert eer.enable(lm) is h1
    eer.disable(esmc)
    assert eer.handle(esmc) is None and all("forward" not in vars(m) for m in h1.modules)


@cuda
def test_small_esmc_stack_hidden_states_and_input_grads_bitwise():
    """The lever inside the fork's attention on a small random-init stack: (a) the fold's feature call form — masked SDPA
    path (chain mask), inference — every collected hidden state tensor-equal; (b) the pseudo-perplexity call form —
    `transformer(x, sequence_id=None)` under bf16 autocast with flash-attn pinned deterministic (the det recipe's attention
    element) — last hidden state AND d(sum(w*h))/d(x) tensor-equal, with and without the lever. The feature call runs
    FIRST on purpose: it builds the rotary cos/sin cache under inference_mode, which the grad pass must then live with (the
    design loop's order on the shared trunk)."""
    import functools
    cfg = ME.ESMCConfig(d_model=256, n_heads=4, n_layers=4, vocab_size=64)
    torch.manual_seed(3)
    esmc = ME.ESMCModel(cfg).to("cuda", torch.bfloat16).eval().requires_grad_(False)
    B, S = 3, 40
    ids = torch.randint(4, 24, (B, S), device="cuda"); seq_id = torch.ones(B, S, dtype=torch.bool, device="cuda"); seq_id[1, 30:] = False
    x0 = torch.randn(B, S, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(B, S, cfg.d_model, device="cuda", dtype=torch.float32)

    def feature_call():
        with torch.inference_mode():
            out = esmc.transformer(esmc.embed(ids), sequence_id=seq_id, layers_to_collect=list(range(cfg.n_layers + 1)))
        return [out[0]] + list(out[2])

    def loss_call():
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hdn, *_ = esmc.transformer(x, sequence_id=None, layers_to_collect=[], output_attentions=False)
        g, = torch.autograd.grad((hdn.float() * w).sum(), x)
        return [hdn.detach(), g.detach()]

    prev = (ME._xformers_available, ME.flash_attn_func)
    ME._xformers_available = False
    if ME.flash_attn_func is not None:
        ME.flash_attn_func = functools.partial(prev[1], deterministic=True)
    try:
        with _TorchBranch():
            ref_f, ref_l = feature_call(), loss_call()
            rep_f, rep_l = feature_call(), loss_call()
            det = all(torch.equal(a, b) for a, b in zip(ref_f + ref_l, rep_f + rep_l))
            assert det, "the stock path is not run-to-run deterministic in this process — comparison void, not a lever finding"
            hnd = eer.enable(esmc)
            try:
                new_f, new_l = feature_call(), loss_call()
            finally:
                eer.disable(esmc)
    finally:
        ME._xformers_available, ME.flash_attn_func = prev
    assert hnd.stats["served"] == 2 * cfg.n_layers and not hnd.stats["fallback"], hnd.stats
    for i, (a, b) in enumerate(zip(ref_f, new_f)):
        assert torch.equal(a, b), f"feature call: hidden state {i} differs, max|d|={(a.float() - b.float()).abs().max().item():.3e}"
    for name, a, b in zip(("last_hidden", "d/dx"), ref_l, new_l):
        assert torch.equal(a, b), f"loss call: {name} differs, max|d|={(a.float() - b.float()).abs().max().item():.3e}"


if __name__ == "__main__":                            # pytest-free: run every test, report, exit nonzero on failure
    n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if getattr(fn, "pytestmark", None) and any(m.name == "parametrize" for m in fn.pytestmark):
                    for args in [m for m in fn.pytestmark if m.name == "parametrize"][0].args[1]:
                        fn(*args)
                else:
                    fn()
                print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1; print("FAIL", name, type(e).__name__, str(e)[:300], flush=True)
    print("RESULT", "OK" if not n_fail else f"{n_fail} failed")
    sys.exit(1 if n_fail else 0)
