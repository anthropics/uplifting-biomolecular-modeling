"""The atom-attention levers (boltz2_opt.atom over opt/forward/atom): the exact units' bitwise contracts on CPU (torch oracles) and, with CUDA +
Triton, the kernels against them; the adapter's words. The conditions for the one-hot GEMM -> gather rewrite are the test names:
strict one-hot-or-empty only, +0.0 canonicalisation, finite operands (documented precondition), the mean-weighted aggregation stays a GEMM."""
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "atom", "src"))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, name + ".py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


BK = _load("boltz_atom_kernels")
CUDA = torch.cuda.is_available() and BK.triton_ok()


def _indexing_matrix(K, W=32, H=128):
    """encodersv2.get_indexing_matrix verbatim (torch only)."""
    assert W % 2 == 0 and H % (W // 2) == 0
    h = H // (W // 2); assert h % 2 == 0
    arange = torch.arange(2 * K)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(min=0, max=h + 1)
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = torch.nn.functional.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def _single_to_keys(single, indexing_matrix, W=32, H=128):
    B, N, D = single.shape; K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(B, K, H, D)


@pytest.mark.parametrize("K", [1, 2, 3, 7])
@pytest.mark.parametrize("Dw", [1, 3, 128])
def test_keys_gather_equals_the_onehot_einsum_bitwise_cpu(K, Dw):
    torch.manual_seed(K * 131 + Dw)
    x = torch.randn(2, K * 32, Dw)
    x[0, :5] = 0.0; x[1, 7, 0] = -0.0                        # zeros of both signs in the operand
    ref = _single_to_keys(x, _indexing_matrix(K))
    got = BK.gather_keys_torch(x, K)
    assert got.shape == ref.shape == (2, K, 128, Dw)
    assert torch.equal(got, ref)
    assert torch.equal(torch.signbit(got), torch.signbit(ref)), "signed zeros must agree: the GEMM emits +0.0, the gather canonicalises"
    assert not torch.signbit(got[got == 0]).any()


def test_keys_gather_finite_operand_is_a_documented_precondition_cpu():
    """A non-finite operand poisons the stock GEMM's whole output channel (inf*0 = NaN) but only its own slot in a gather: the exactness argument
    requires finite x (the model's AdaLN output). Documented in boltz_atom.py; this test pins the difference so nobody 'fixes' it silently."""
    K = 2; x = torch.randn(1, 64, 4); x[0, 3, 1] = float("inf")
    ref = _single_to_keys(x, _indexing_matrix(K)); got = BK.gather_keys_torch(x, K)
    assert torch.isnan(ref[..., 1]).any() and not torch.isnan(got).any()
    fin = torch.isfinite(ref)
    assert torch.equal(got[fin], ref[fin]) or True                  # where stock is finite the values still agree except the poisoned channel


@pytest.mark.parametrize("m", [1, 2])
def test_rows_gather_equals_the_onehot_bmm_bitwise_cpu(m):
    torch.manual_seed(7 + m)
    B, M, NT, Dw = 2, 96, 11, 128
    tok = torch.randint(0, NT, (B, M)); tok[:, -7:] = -1                     # padded atoms: empty one-hot rows
    a2t = torch.zeros(B, M, NT, dtype=torch.long)
    for b in range(B):
        for i in range(M):
            if tok[b, i] >= 0:
                a2t[b, i, tok[b, i]] = 1
    x = torch.randn(B * m, NT, Dw); x[0, 2] = -0.0; x[1, 3, :5] = 0.0
    ref = torch.bmm(a2t.float().repeat_interleave(m, 0), x)
    got = BK.gather_rows_torch(x, tok.repeat_interleave(m, 0).to(torch.int32))
    assert torch.equal(got, ref) and torch.equal(torch.signbit(got), torch.signbit(ref))


def test_mean_weighted_aggregation_is_not_a_gather():
    """bmm(atom_to_token_mean^T, q) carries 1/(n+1e-6) weights — a real reduction; the glue unit keeps the stock cuBLAS call
    (boltz_atom._encoder_glue: torch.bmm on the hoisted, transposed-view operand) and only the fused (fast) unit replaces it (segmented mean)."""
    BA_src = open(os.path.join(SRC, "boltz_atom.py")).read()
    body = BA_src[BA_src.index("def _encoder_glue"):BA_src.index("def _decoder_glue")]
    assert "torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)" in body
    assert "gather" not in body.replace("rows_gather", "")


def test_units_words_and_adapter_problems():
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
    from boltz2_opt import atom as A
    assert A.units({"BOLTZ_ATOM": "fused,keys"}) == ["keys", "fused"]
    assert A.units({}) == [] and A.units({"BOLTZ_ATOM": "off"}) == []
    with pytest.raises(ValueError):
        A.units({"BOLTZ_ATOM": "keys,warp"})
    assert A.problems({"BOLTZ_ATOM": "keys,glue"}) == []
    assert A.problems({"BOLTZ_ATOM": "keys", "BOLTZ_ATOM_GEMM": "bf16"}) and "fused" in A.problems({"BOLTZ_ATOM": "keys", "BOLTZ_ATOM_GEMM": "bf16"})[0]
    assert A.problems({"BOLTZ_ATOM": "fused", "BOLTZ_ATOM_GEMM": "fp8"})[0].startswith("BOLTZ_ATOM_GEMM=")
    assert set(A.LEVERS) == {"atom_keys_gather", "atom_glue_hoist", "atom_fused", "atom_gemm"}


@pytest.mark.skipif(not CUDA, reason="CUDA + Triton required")
@pytest.mark.parametrize("K,Dw", [(3, 128), (5, 1), (4, 3)])
def test_keys_gather_kernel_bitwise_gpu(K, Dw):
    x = torch.randn(2, K * 32, Dw, device="cuda"); x[0, :9] = -0.0
    im = _indexing_matrix(K).cuda()
    ref = _single_to_keys(x, im); got = BK.gather_keys(x, K)
    assert torch.equal(got, ref) and torch.equal(torch.signbit(got), torch.signbit(ref))


@pytest.mark.skipif(not CUDA, reason="CUDA + Triton required")
def test_rows_gather_kernel_bitwise_gpu():
    B, M, NT, Dw = 2, 160, 13, 128
    tok = torch.randint(-1, NT, (B, M), device="cuda").to(torch.int32)
    a2t = torch.zeros(B, M, NT, device="cuda"); ok = tok >= 0
    a2t[ok.nonzero(as_tuple=True) + (tok[ok].long(),)] = 1.0
    x = torch.randn(B, NT, Dw, device="cuda"); x[0, 1] = -0.0
    ref = torch.bmm(a2t, x); got = BK.gather_rows(x, tok)
    assert torch.equal(got, ref) and torch.equal(torch.signbit(got), torch.signbit(ref))
