"""Unit tests of lever `triatt`'s kernel — the provider row `cd_triatt` it binds through kernels/provider.py (`opt_core.kernels.pallas_triatt.triatt_attn`;
the kit's private copy kernels/triatt_attn.py + attbwd_dkdv.py left the tree in 0.10.3, the shared core carries them): forward AND gradients (dq, dk, dv, d pair-bias) against the pure-JAX fp32
reference on the same dtype-rounded inputs, at the AF-Multimer design step's call forms — triangle attention [B=N, H=4, S=N, D=32] at
N = 200/500/800, ragged sizes (S and B not multiples of any tile), key masks (random, tail, rows fully masked), D=16 (template stack) and D=8
(extra-MSA row attention: the zero-pad path), S=2 (MSA column attention: the small-XLA path), float32 inputs at tf32 and ieee — plus the op's
contract: unknown knobs raise, describe_op() reports the knobs, run-to-run bitwise (deterministic backward), jit + run-time mask dispatch.
GPU-only (the kernel is Pallas/Triton): every test is skipped by name on a host without a CUDA device."""
from __future__ import annotations

import unittest

import numpy as np

jax = jnp = T = None                                          # bound lazily by _require_gpu (no jax import, no backend probe at module import)
_GPU = None


def _require_gpu(case: unittest.TestCase) -> None:
    """Skip `case` by name unless a CUDA device is present; binds jax / jnp / the kernel module on first use (never at import)."""
    global jax, jnp, T, _GPU
    if _GPU is None:
        try:
            import jax as _jax
            import jax.numpy as _jnp
            _GPU = any(d.platform == "gpu" for d in _jax.devices())
        except Exception:  # noqa: BLE001
            _GPU = False
        if _GPU:
            import importlib
            from colabdesign_opt.kernels import provider as _P
            _T = importlib.import_module(_P.BINDINGS["triatt"].rows["cd_triatt"])   # the ROW module lever triatt binds (opt_core.kernels.pallas_triatt.triatt_attn); the kit's private copy left the tree in 0.10.3
            jax, jnp, T = _jax, _jnp, _T
    if not _GPU:
        case.skipTest("triatt_attn is a GPU (Pallas/Triton) kernel: no CUDA device here")


def _inputs(B, H, S, D, dtype, masked, seed=0, Sq=None):
    Sq = S if Sq is None else Sq
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    q = jax.random.normal(ks[0], (B, H, Sq, D), jnp.float32).astype(dtype)
    k = jax.random.normal(ks[1], (B, H, S, D), jnp.float32).astype(dtype)
    v = jax.random.normal(ks[2], (B, H, S, D), jnp.float32).astype(dtype)
    bias = jax.random.normal(ks[3], (H, Sq, S), jnp.float32).astype(dtype)
    if masked == "random":
        kmask = jax.random.uniform(ks[4], (B, S)) > 0.15
        kmask = kmask.at[:, 0].set(True)
    elif masked == "rows":                                    # some batch rows with every key masked (the template stack's binder rows)
        kmask = jnp.ones((B, S), bool).at[::3, :].set(False)
    elif masked == "tail":                                    # 0..4 trailing keys masked per row
        kmask = jnp.arange(S)[None, :] < (S - (jnp.arange(B) % 5))[:, None]
    else:
        kmask = jnp.ones((B, S), bool)
    do = jax.random.normal(ks[5], (B, H, Sq, D), jnp.float32).astype(dtype)
    return q, k, v, bias, kmask, do


def stock_xla_attention(q, k, v, bias, key_mask, scale):
    """Stock's math (modules.Attention: logits = einsum(q*scale, k) + mask bias + pair bias in the INPUT dtype, softmax, einsum) on heads-major
    tensors: the yardstick's 'stock bf16 class' arm (its error vs the fp32 reference is what the kernel's error is held against)."""
    mask_bias = (1e9 * (key_mask.astype(q.dtype) - 1.0))[:, None, None, :]
    logits = jnp.einsum("bhqd,bhkd->bhqk", q * scale, k) + mask_bias + bias[None]
    logits = jnp.clip(logits, -1e8, 1e8)                     # ColabDesign's patch (modules.py:365): masked logits are pinned, pass no gradient
    w = jax.nn.softmax(logits)
    return jnp.einsum("bhqk,bhkd->bhqd", w, v)


def _vjp_chunked(fn, q, k, v, bias, kmask, do, scale, chunk):
    """(o, (dq, dk, dv, dbias)) of fn over batch-row chunks (the fp32 reference materialises [B,H,S,S] logits: 8 GB at N=800)."""
    os_, dqs, dks, dvs, db = [], [], [], [], None
    for b0 in range(0, q.shape[0], chunk):
        sl = slice(b0, b0 + chunk)
        o, vjp = jax.vjp(lambda q_, k_, v_, b_: fn(q_, k_, v_, b_, kmask[sl], scale), q[sl], k[sl], v[sl], bias)
        g = vjp(do[sl])
        os_.append(o); dqs.append(g[0]); dks.append(g[1]); dvs.append(g[2])
        db = g[3].astype(jnp.float32) if db is None else db + g[3].astype(jnp.float32)
    cat = lambda xs: jnp.concatenate(xs, axis=0)             # noqa: E731
    return cat(os_), (cat(dqs), cat(dks), cat(dvs), db)


def _run(op, ref, q, k, v, bias, kmask, do, scale):
    o, vjp = jax.vjp(lambda q_, k_, v_, b_: op(q_, k_, v_, b_, kmask, scale), q, k, v, bias)
    g = vjp(do)
    big = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * 4 > 2 ** 31
    o_r, g_r = _vjp_chunked(ref, q, k, v, bias, kmask, do, scale, chunk=(100 if big else q.shape[0]))
    f32 = lambda x: np.asarray(x.astype(jnp.float32))       # noqa: E731
    return {"o": (f32(o), f32(o_r)), "dq": (f32(g[0]), f32(g_r[0])), "dk": (f32(g[1]), f32(g_r[1])), "dv": (f32(g[2]), f32(g_r[2])),
            "dbias": (f32(g[3]), f32(g_r[3]))}


class TestTriattVsReference(unittest.TestCase):
    def setUp(self):
        _require_gpu(self)

    # tolerance model: outputs/gradients are bf16 (2^-8 relative) roundings of fp32 accumulations whose summands are bf16-rounded P / dS;
    # max |err| <= tol_rel * max|ref| + tol_abs with the class's constants (stock's own bf16 XLA path sits in the same class).
    TOL = {"bfloat16": (2.5e-2, 2e-2), "float32_tf32": (4e-3, 2e-3), "float32_ieee": (2e-5, 1e-5)}

    def _check(self, res, cls, what):
        rel, ab = self.TOL[cls]
        for name, (a, b) in res.items():
            self.assertTrue(np.isfinite(a).all(), f"{what}: {name} has non-finite values")
            err = float(np.max(np.abs(a - b))); refm = float(np.max(np.abs(b)))
            self.assertLessEqual(err, rel * refm + ab, f"{what}: {name} max|err|={err:.3e} vs max|ref|={refm:.3e} (class {cls})")

    def _case(self, B, H, S, D, dtype=None, masked="none", knobs=None, Sq=None, cls=None, vs_stock=True):
        dtype = dtype or jnp.bfloat16
        q, k, v, bias, kmask, do = _inputs(B, H, S, D, dtype, masked, Sq=Sq)
        op = T.make_attention(**(knobs or {}))
        what = f"B{B} H{H} S{S} Sq{Sq} D{D} {jnp.dtype(dtype).name} mask={masked} knobs={knobs}"
        res = _run(op, T.reference_attention, q, k, v, bias, kmask, do, float(D) ** -0.5)
        cls = cls or ("bfloat16" if dtype == jnp.bfloat16 else "float32_tf32")
        self._check(res, cls, what)
        if vs_stock and dtype == jnp.bfloat16:               # the yardstick: no worse than stock's own bf16 XLA path against the same reference
            sto = _run(stock_xla_attention, T.reference_attention, q, k, v, bias, kmask, do, float(D) ** -0.5)
            for name in res:
                e_k = float(np.max(np.abs(res[name][0] - res[name][1]))); e_s = float(np.max(np.abs(sto[name][0] - sto[name][1])))
                self.assertLessEqual(e_k, 2.0 * e_s + 1e-3, f"{what}: {name} kernel max|err| {e_k:.3e} vs stock-bf16-XLA max|err| {e_s:.3e}")
        return res

    def test_triangle_attention_sizes(self):
        """[B=N, H=4, S=N, D=32] at the identity cases' and the K ladder's sizes: N mod tile != 0 and B mod G != 0 among them."""
        for n in (195, 256, 431, 500, 601, 800):
            with self.subTest(N=n):
                self._case(n, 4, n, 32, masked=("tail" if n in (195, 601) else "none"))

    def test_heads_and_head_dims(self):
        for H in (4, 8):
            for D in (8, 16, 32):
                with self.subTest(H=H, D=D):
                    self._case(37, H, 195, D, masked="random")

    def test_ragged_and_masks(self):
        self._case(5, 4, 131, 32, masked="random")
        self._case(7, 4, 257, 32, masked="tail")
        self._case(3, 8, 300, 32, masked="random", knobs=dict(mask_free="never"))
        self._case(9, 4, 257, 32, knobs=dict(G=4))                # forward rows-per-program 4 with a ragged batch (9 = 2x4+1: the clamped group)
        self._case(4, 4, 100, 32, Sq=64, masked="random")   # S_q != S_k

    def test_fully_masked_rows_follow_stock(self):
        """Batch rows with EVERY key masked (the template stack's binder rows, the padded second MSA row, a fully masked extra-MSA row):
        stock's bf16 logits absorb into -1e9 and the clip pins them, so the row is a uniform average of v, dV = P^T dO at uniform P, and NO
        gradient reaches q, k or the pair bias from it — the kernel serves exactly that (finite), checked against STOCK's bf16 XLA path."""
        for B, H, D in ((6, 4, 32), (6, 4, 16), (1, 8, 8), (2, 8, 32)):
            q, k, v, bias, kmask, do = _inputs(B, H, 195, D, jnp.bfloat16, "rows")
            res = _run(T.attention, stock_xla_attention, q, k, v, bias, kmask, do, D ** -0.5)
            for name, (a, b) in res.items():
                self.assertTrue(np.isfinite(a).all(), f"{name} non-finite on fully masked rows")
                err = float(np.max(np.abs(a - b))); refm = float(np.max(np.abs(b)))
                self.assertLessEqual(err, 2.5e-2 * refm + 2e-2, f"fully-masked rows B{B} H{H} D{D}: {name} max|err| {err:.3e} vs stock-bf16 max {refm:.3e}")

    def test_small_head_dims_and_small_seq(self):
        self._case(6, 4, 150, 16, masked="random")          # template pair stack (64 channels / 4 heads)
        self._case(2, 8, 120, 8, masked="tail")             # extra-MSA row attention (64 / 8 heads): zero-padded to 16, exact
        self._case(50, 8, 2, 32, masked="random")           # MSA column attention over N_seq = 2: the small-XLA path
        self._case(3, 4, 40, 32)                            # S below one tile

    def test_float32_precisions(self):
        self._case(6, 4, 200, 32, dtype=jnp.float32, masked="random")                                   # tf32 (stock's DEFAULT class)
        self._case(6, 4, 200, 32, dtype=jnp.float32, masked="random", knobs=dict(f32_precision="ieee"), cls="float32_ieee")

    def test_zero_pad_is_exact(self):
        """D=8 served by zero-padding to 16 == the kernel run on inputs that are already zero in those columns (bitwise)."""
        q, k, v, bias, kmask, do = _inputs(2, 8, 100, 8, jnp.bfloat16, "random")
        pad = lambda x: jnp.pad(x, [(0, 0)] * 3 + [(0, 8)])   # noqa: E731
        o8 = T.attention(q, k, v, bias, kmask, 8 ** -0.5)
        o16 = T.attention(pad(q), pad(k), pad(v), bias, kmask, 8 ** -0.5)[..., :8]
        np.testing.assert_array_equal(np.asarray(o8.astype(jnp.float32)), np.asarray(o16.astype(jnp.float32)))

    def test_prescaled_q_scale_one_is_the_same_op(self):
        """The op applies exactly `scale` and nothing else, at stock's rounding point (bf16(q * s) once, before Q.K^T; scale-free kernels):
        q pre-multiplied by s and rounded to bf16 — as a projection that folds the scale would hand it over — with scale=1.0 gives the SAME
        outputs and dK / dV / d(bias) bitwise, and dQ equal to bf16 tolerance (the chain rule's multiply by s is XLA's, rounded once)."""
        q, k, v, bias, kmask, do = _inputs(9, 4, 195, 32, jnp.bfloat16, "random")
        s = 32 ** -0.5
        qs = (q.astype(jnp.float32) * s).astype(q.dtype)
        o1, vjp1 = jax.vjp(lambda q_, k_, v_, b_: T.attention(q_, k_, v_, b_, kmask, s), q, k, v, bias); g1 = vjp1(do)
        o2, vjp2 = jax.vjp(lambda q_, k_, v_, b_: T.attention(q_, k_, v_, b_, kmask, 1.0), qs, k, v, bias); g2 = vjp2(do)
        f32 = lambda x: np.asarray(x.astype(jnp.float32))   # noqa: E731
        np.testing.assert_array_equal(f32(o1), f32(o2))
        for a, b, name in zip(g1[1:], g2[1:], ("dk", "dv", "dbias")):
            np.testing.assert_array_equal(f32(a), f32(b), err_msg=name)
        np.testing.assert_allclose(f32(g1[0]), f32(g2[0]) * s, rtol=0, atol=2.5e-2 * float(np.max(np.abs(f32(g1[0])))) + 1e-3)

    def test_deterministic(self):
        """Run-to-run bitwise, gradients included (no atomics anywhere)."""
        q, k, v, bias, kmask, do = _inputs(37, 4, 190, 32, jnp.bfloat16, "random")
        f = jax.jit(lambda q, k, v, b, do: jax.vjp(lambda *a: T.attention(*a, kmask, 32 ** -0.5), q, k, v, b)[1](do))
        g1 = [np.asarray(x.astype(jnp.float32)) for x in f(q, k, v, bias, do)]
        g2 = [np.asarray(x.astype(jnp.float32)) for x in f(q, k, v, bias, do)]
        for a, b in zip(g1, g2):
            np.testing.assert_array_equal(a, b)

    def test_under_jit_with_traced_mask(self):
        """The all(mask) dispatch is a run-time conditional: one compiled program serves masked and unmasked calls."""
        q, k, v, bias, kmask, do = _inputs(6, 4, 128, 32, jnp.bfloat16, "random")
        f = jax.jit(lambda q, k, v, b, m: T.attention(q, k, v, b, m, 32 ** -0.5))
        o_m = f(q, k, v, bias, kmask); o_u = f(q, k, v, bias, jnp.ones_like(kmask))
        r_m = T.reference_attention(q, k, v, bias, kmask, 32 ** -0.5); r_u = T.reference_attention(q, k, v, bias, jnp.ones_like(kmask), 32 ** -0.5)
        for o, r in ((o_m, r_m), (o_u, r_u)):
            err = float(jnp.max(jnp.abs(o.astype(jnp.float32) - r.astype(jnp.float32))))
            self.assertLess(err, 2.5e-2 * float(jnp.max(jnp.abs(r))) + 2e-2)


class TestTriattContract(unittest.TestCase):
    def setUp(self):
        _require_gpu(self)

    def test_knobs(self):
        with self.assertRaises(ValueError):
            T.make_attention(no_such_knob=1)
        with self.assertRaises(ValueError):
            T.make_attention(f32_precision="fp16")
        with self.assertRaises(ValueError):
            T.make_attention(grid_order="xyz")
        op = T.make_attention(G=2)
        d = T.describe_op(op)
        self.assertEqual(d["G"], 2)
        for key in ("bq", "bk", "G", "f32_precision", "dbias_partials", "mask_free", "grid_order"):
            self.assertIn(key, d)
        self.assertEqual(T.describe_op(T.attention)["f32_precision"], "tf32")


if __name__ == "__main__":
    unittest.main()
