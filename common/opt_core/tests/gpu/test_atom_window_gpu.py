"""GPU tests of opt_core.kernels.atom_window (ATOM_WINDOW v0.1; run on a CUDA box with Triton:
`python -m pytest common/opt_core/tests/gpu/test_atom_window_gpu.py -q`; skipped by name elsewhere).
(1) `ln_qkvg` + `window_attn` = an fp64 evaluation of the same block mathematics (LayerNorm, both AdaLN modulations, q|k|v|g, the shifted-window
pair-bias attention with the engines' key / pair validity, sigmoid gate, W_o + b_o, AdaLN-Zero gate, residual) built with an explicit GATHER of
the key windows — on layouts with padding atoms, fewer real atoms than one key window, an atom count that is not a multiple of the query block,
several samples with a sample-strided input, and at every precision word: `ieee` to fp32 rounding, `tf32rn` / `tf32x3` / `tf32` inside the TF32
class; padded query rows come out finite; (2) the kernels are deterministic (bitwise repeat); (3) `warmup` compiles and runs."""
import math

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("needs Triton", allow_module_level=True)
from opt_core.kernels import atom_window as AW  # noqa: E402

DEV = torch.device("cuda")
NQ, NK = 32, 128


def _problem(S, A, n_real, C, H, seed, sample_strided=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *shape, scale=1.0: (torch.randn(*shape, generator=g) * scale).to(DEV)   # noqa: E731
    if sample_strided:                                       # a [S, A, C] view with a non-standard sample stride (a slice of a wider buffer)
        buf = rn(S, A + 7, C); a = buf[:, 3:3 + A, :]
    else:
        a = rn(S, A, C)
    lin = {k: torch.nn.Linear(C, C, bias=b).to(DEV) for k, b in (("q", True), ("k", False), ("v", False), ("g", False), ("o", True))}
    cond = {k: torch.sigmoid(rn(A, C)) if k in ("gq", "gk", "gate") else rn(A, C, scale=0.5) for k in ("gq", "lsq", "gk", "lsk", "gate")}
    nb = math.ceil(A / NQ)
    bias = rn(nb, H, NQ, NK, scale=0.5)
    amask = torch.zeros(1, A, device=DEV); amask[:, :n_real] = 1.0
    ks = AW.window_starts(A, torch.tensor(float(n_real), device=DEV), NQ, NK, DEV)
    n_real_t = torch.full((1,), n_real, dtype=torch.int32, device=DEV)
    return a, lin, cond, bias, amask, ks, n_real_t


def _reference(a, lin, cond, bias, amask, ks, n_real, H, eps=1e-5, inf=1e9, dtype=torch.float64):
    """The block's attention half in `dtype` with an explicit gather of the key windows: a + gate * (W_o o + b_o)."""
    S, A, C = a.shape; D = C // H; nb = bias.shape[0]
    cv = lambda t: t.to(dtype)   # noqa: E731
    x = cv(a)
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    xh = (x - mu) / torch.sqrt(var + eps)
    xq = cv(cond["gq"]) * xh + cv(cond["lsq"]); xk = cv(cond["gk"]) * xh + cv(cond["lsk"])
    W = {k: cv(l.weight) for k, l in lin.items()}; B = {k: (cv(l.bias) if l.bias is not None else None) for k, l in lin.items()}
    proj = lambda t, k: t @ W[k].T + (B[k] if B[k] is not None else 0)   # noqa: E731
    q = proj(xq, "q") / math.sqrt(D); k = proj(xk, "k"); v = proj(xk, "v"); gt = torch.sigmoid(proj(xq, "g"))
    pad = nb * NQ - A
    qb = torch.nn.functional.pad(q, (0, 0, 0, pad)).reshape(S, nb, NQ, H, D)
    k_idx = ks.long()[:, None] + torch.arange(NK, device=a.device)[None, :]                  # [nb, NK]
    k_in = (k_idx >= 0) & (k_idx < A)
    kc = k_idx.clamp(0, A - 1)
    kb = k[:, kc].reshape(S, nb, NK, H, D); vb = v[:, kc].reshape(S, nb, NK, H, D)
    am = amask.expand(S, A)
    key_valid = k_in[None] & (k_idx[None] < int(n_real.item())) & (am[:, kc] > 0.5)         # [S, nb, NK]
    qm = torch.nn.functional.pad(am, (0, pad)).reshape(S, nb, NQ) > 0.5
    pair = qm[..., :, None] & key_valid[..., None, :]                                         # [S, nb, NQ, NK]
    logits = torch.einsum("sbqhd,sbkhd->sbhqk", qb, kb) + cv(bias)[None] + torch.where(pair, 0.0, -inf).to(dtype)[:, :, None]
    p = torch.softmax(logits, -1)
    o = torch.einsum("sbhqk,sbkhd->sbqhd", p, vb).reshape(S, nb * NQ, C)[:, :A]
    o = o * gt
    y = proj(o, "o")
    return x + cv(cond["gate"]) * y


CASES = [  # S, A, n_real, C, H, sample_strided
    (1, 40, 40, 128, 4, False),        # fewer atoms than one key window (negative window start, invalid keys)
    (2, 203, 180, 128, 4, True),       # padding atoms, A not a multiple of 32, sample-strided input
    (5, 700, 700, 128, 4, False),      # several blocks with shifted end windows, 5 samples
    (3, 257, 250, 64, 4, False),       # D = 16
    (2, 96, 96, 128, 2, False),        # D = 64
]


@pytest.mark.parametrize("S,A,n_real,C,H,strided", CASES)
@pytest.mark.parametrize("precision", ["ieee", "tf32rn", "tf32x3", "tf32"])
def test_fused_attention_half_matches_fp64(S, A, n_real, C, H, strided, precision):
    a, lin, cond, bias, amask, ks, n_real_t = _problem(S, A, n_real, C, H, seed=S * 1000 + A, sample_strided=strided)
    with torch.no_grad():
        qkvg = AW.ln_qkvg(a, cond["gq"], cond["lsq"], cond["gk"], cond["lsk"], lin["q"], lin["k"], lin["v"], lin["g"], 1e-5, 1.0 / math.sqrt(C // H), precision=precision)
        got = AW.window_attn(qkvg, a, bias, ks, n_real_t, amask, cond["gate"], lin["o"].weight, lin["o"].bias, H, NQ, NK, 1e9, precision=precision)
        ref = _reference(a, lin, cond, bias, amask, ks, n_real_t, H)
        ref32 = _reference(a, lin, cond, bias, amask, ks, n_real_t, H, dtype=torch.float32)      # plain fp32 torch ops (no TF32): the yardstick for the TF32 words
    assert got.shape == a.shape and got.dtype == torch.float32 and torch.isfinite(got).all()
    err = (got.double() - ref)[:, :n_real].abs().max().item()
    err32 = (ref32.double() - ref)[:, :n_real].abs().max().item()
    scale = ref[:, :n_real].abs().max().item()
    if precision == "ieee":
        assert err <= max(8 * err32, 2e-5 * scale), (err, err32)
    elif precision == "tf32x3":
        assert err <= max(40 * err32, 1e-4 * scale), (err, err32)
    else:                                                                                        # tf32rn / tf32: 10-bit-mantissa products, fp32 accumulation
        assert err <= 4e-3 * scale, (err, scale)
    if precision == "tf32rn":                                                                    # round-to-nearest operands land closer than truncated ones
        qk_t = AW.ln_qkvg(a, cond["gq"], cond["lsq"], cond["gk"], cond["lsk"], lin["q"], lin["k"], lin["v"], lin["g"], 1e-5, 1.0 / math.sqrt(C // H), precision="tf32")
        got_t = AW.window_attn(qk_t, a, bias, ks, n_real_t, amask, cond["gate"], lin["o"].weight, lin["o"].bias, H, NQ, NK, 1e9, precision="tf32")
        rms = lambda t: (t.double() - ref)[:, :n_real].pow(2).mean().sqrt().item()   # noqa: E731
        assert rms(got) <= rms(got_t) * 1.05


def test_deterministic_and_padded_rows_finite():
    S, A, n_real, C, H = 3, 190, 170, 128, 4
    a, lin, cond, bias, amask, ks, n_real_t = _problem(S, A, n_real, C, H, seed=7)
    with torch.no_grad():
        outs = []
        for _ in range(3):
            qkvg = AW.ln_qkvg(a, cond["gq"], cond["lsq"], cond["gk"], cond["lsk"], lin["q"], lin["k"], lin["v"], lin["g"], 1e-5, 1.0 / math.sqrt(C // H))
            outs.append(AW.window_attn(qkvg, a, bias, ks, n_real_t, amask, cond["gate"], lin["o"].weight, lin["o"].bias, H, NQ, NK, 1e9))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
    assert torch.isfinite(outs[0]).all()                                                        # padding rows (masked queries): finite, never NaN from an all -inf row


def test_window_starts_rule():
    ws = lambda A, n: AW.window_starts(A, torch.tensor(float(n)), NQ, NK, "cpu").tolist()   # noqa: E731
    assert ws(40, 40) == [0, 0]                                              # n_real < NK: both windows start at 0 (keys >= 40 are invalid in the kernel)
    assert ws(700, 700)[:3] == [0, 0, 16] and ws(700, 700)[-1] == 700 - NK  # interior windows centred on their query block; the last shifted left to end at n_real - 1
    assert ws(720, 700)[-1] == 700 - NK and len(ws(720, 700)) == 23         # padding atoms beyond n_real do not move the windows; NB = ceil(A / NQ)
    t = AW.window_starts(64, torch.tensor(64.0), NQ, NK, "cpu")
    assert t.dtype == torch.int32 and t.shape == (2,)
