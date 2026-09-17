"""GPU tests of opt_core.kernels.dtk_kernels' periodic-operand row kernels (header DTK v0.1+kbg1; run on a CUDA box with Triton:
`python -m pytest common/opt_core/tests/gpu/test_dtk_kbg1_gpu.py -q`; skipped by name elsewhere).
(1) a periodic operand `[P, C]` (`ln_modulate(mod_period=P)`, `gate_residual(gate_period=P)`) is BIT-EXACT the same call on the operand
materialised over the R/P leading copies; (2) `swiglu(ab, c=c)` == `swiglu(ab) * c` (fp32: bit-exact; bf16 out: one rounding instead of two ->
within 1 bf16 ulp) and `out=` is written in place; (3) `sigmoid_gate=False` takes the gate post-sigmoid; (4) backward compatibility: with the
new keyword arguments left at their defaults every row kernel is BIT-EXACT the reference module named by `DTK_REF_MODULE` (an importable copy
of the previous carried bytes, e.g. `dtk_kernels_v01`; that test skips by name when the variable is unset).
Shapes are the AF3-family diffusion transformer's at the design engines' saturated batch: R = B x N token rows (B 128 samples x N 100 tokens,
B 16 x N 500), C = 768 (2 x token_s) and 384, conditioning period P = N."""
import importlib
import os

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("needs Triton", allow_module_level=True)
from opt_core.kernels import dtk_kernels as D  # noqa: E402

DEV = torch.device("cuda")
SHAPES = [(128, 100, 768), (16, 500, 768), (40, 154, 384)]          # (B samples, N tokens = period, C)


def _rows(B, N, C, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B * N, C, generator=g).to(DEV, dtype)
    blk = lambda: torch.randn(N, C, generator=g).to(DEV, dtype)       # a conditioning block [P = N, C]
    return x, blk(), blk(), (1 + 0.1 * torch.randn(C, generator=g)).to(DEV), (0.1 * torch.randn(C, generator=g)).to(DEV)


@pytest.mark.parametrize("B,N,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("rms", [False, True], ids=["ln", "rms"])
def test_ln_modulate_periodic_operand_is_bitwise_the_materialised_operand(B, N, C, dtype, rms):
    x, sc, sh, w, b = _rows(B, N, C, dtype, 1)
    y_per = D.ln_modulate(x, sc, sh, weight=w, bias=b, rms=rms, mod_period=N)
    y_mat = D.ln_modulate(x, sc.repeat(B, 1), sh.repeat(B, 1), weight=w, bias=b, rms=rms)      # row r modulated by block row r % N
    assert torch.equal(y_per, y_mat)
    assert torch.equal(y_per, D.ln_modulate(x, sc, sh, weight=w, bias=b, rms=rms, mod_period=N)), "run-to-run bitwise"
    assert torch.isfinite(y_per.float()).all()


@pytest.mark.parametrize("B,N,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("res", [False, True], ids=["nores", "res"])
def test_gate_residual_periodic_gate_is_bitwise_the_materialised_gate(B, N, C, dtype, res):
    x, g, r0, _w, _b = _rows(B, N, C, dtype, 2)
    resid = torch.randn_like(x) if res else None
    y_per = D.gate_residual(x, gate=g, res=resid, gate_period=N)
    y_mat = D.gate_residual(x, gate=g.repeat(B, 1), res=resid)
    assert torch.equal(y_per, y_mat)
    assert torch.equal(y_per, D.gate_residual(x, gate=g, res=resid, gate_period=N)), "run-to-run bitwise"


@pytest.mark.parametrize("B,N,C", SHAPES[:2])
def test_gate_given_post_sigmoid(B, N, C):
    x, g, _r, _w, _b = _rows(B, N, C, torch.float32, 3)
    y_pre = D.gate_residual(x, gate=g, gate_period=N)                                   # sigmoid inside the kernel (fp32)
    y_post = D.gate_residual(x, gate=torch.sigmoid(g), gate_period=N, sigmoid_gate=False)
    assert (y_pre - y_post).abs().max().item() <= 1e-5 * (1 + y_pre.abs().max().item())    # torch's sigmoid vs the kernel's 1/(1+exp(-g)): same value to fp32 rounding
    assert torch.equal(y_post, D.gate_residual(x, gate=torch.sigmoid(g), gate_period=N, sigmoid_gate=False))


@pytest.mark.parametrize("B,N,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_swiglu_third_factor_and_out(B, N, C, dtype):
    g = torch.Generator(device="cpu").manual_seed(4)
    h = 2 * C
    abc = torch.randn(B * N, 3 * h, generator=g).to(DEV, dtype)                         # ONE GEMM output [R, 3h]: a | b | c column blocks (row stride 3h)
    ab, c = abc[:, :2 * h], abc[:, 2 * h:]
    y3 = D.swiglu(ab, c=c)
    y2 = D.swiglu(ab, out_dtype=torch.float32).mul(c.float())                           # (silu(a) * b) in fp32, times c in fp32
    if dtype == torch.float32:
        assert torch.equal(y3, y2)                                                      # the same fp32 products in the same order
    else:
        d = (y3.float() - y2.to(dtype).float()).abs()
        ulp = torch.finfo(dtype).eps * y2.abs().clamp_min(1e-30)
        assert bool((d <= ulp).all()), float(d.max())                                   # one bf16 rounding instead of two: within 1 ulp
    out = torch.empty(B * N, h, device=DEV, dtype=dtype)
    r = D.swiglu(ab, c=c, out=out)
    assert r.data_ptr() == out.data_ptr() and torch.equal(out, y3)
    assert torch.equal(y3, D.swiglu(ab, c=c)), "run-to-run bitwise"


def _ref_module():
    name = os.environ.get("DTK_REF_MODULE")
    if not name:
        pytest.skip("DTK_REF_MODULE unset (name an importable copy of the previous carried dtk_kernels to hold backward compatibility bitwise)")
    return importlib.import_module(name)


@pytest.mark.parametrize("B,N,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_default_arguments_are_bitwise_the_reference_module(B, N, C, dtype):
    R = _ref_module()
    x, sc, sh, w, b = _rows(B, N, C, dtype, 5)
    scm, shm = sc.repeat(B, 1), sh.repeat(B, 1)
    for rms in (False, True):
        assert torch.equal(D.ln_modulate(x, scm, shm, weight=w, bias=b, rms=rms), R.ln_modulate(x, scm, shm, weight=w, bias=b, rms=rms))
        assert torch.equal(D.ln_modulate(x, None, None, weight=w, bias=b, rms=rms), R.ln_modulate(x, None, None, weight=w, bias=b, rms=rms))
    assert torch.equal(D.ln_modulate(x, scm, shm, sigmoid_scale=False), R.ln_modulate(x, scm, shm, sigmoid_scale=False))
    gm = torch.randn_like(x); res = torch.randn_like(x)
    assert torch.equal(D.gate_residual(x, gate=gm, res=res), R.gate_residual(x, gate=gm, res=res))
    assert torch.equal(D.gate_residual(x, gate=gm), R.gate_residual(x, gate=gm))
    assert torch.equal(D.gate_residual(x, res=res), R.gate_residual(x, res=res))
    ab = torch.randn(B * N, 2 * C, device=DEV, dtype=dtype)
    assert torch.equal(D.swiglu(ab), R.swiglu(ab))
    assert torch.equal(D.swiglu(ab, a_first_silu=False, out_dtype=torch.float32), R.swiglu(ab, a_first_silu=False, out_dtype=torch.float32))
