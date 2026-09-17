"""Lever `proj` (kernels/proj_attn.py): the fused q|k|v|gate projection is the stock projections' algebra (CPU or GPU, real jax), the
rebound `Attention` keeps the stock parameter tree and matches the `pallas` lever's served path in value and gradient on the GPU (bf16
tolerance — the lever is precision-class, not bitwise), and the install contract holds (requires `pallas`, idempotent, restorable, one
LEVER line in the levers grammar). Skipped where real jax / haiku / a GPU are absent (the package's CPU suite runs on a stand-in jax)."""
import importlib
import re
import unittest

_PROBE = {}


def _real_jax():
    """Real jax importable (not the stand-in)? Probed lazily, INSIDE a test — never at module import."""
    if "jax" not in _PROBE:
        try:
            jax = importlib.import_module("jax"); jnp = importlib.import_module("jax.numpy")
            _PROBE["jax"] = jax if hasattr(jax, "jit") and hasattr(jnp, "einsum") and hasattr(jax, "devices") else None
        except Exception:  # noqa: BLE001
            _PROBE["jax"] = None
    return _PROBE["jax"]


def _gpu():
    if "gpu" not in _PROBE:
        jax = _real_jax()
        try:
            _PROBE["gpu"] = bool(jax is not None and jax.devices()[0].platform == "gpu")
        except Exception:  # noqa: BLE001
            _PROBE["gpu"] = False
    return _PROBE["gpu"]


def need(gpu: bool):
    """Skip (lazily) unless real jax [and a GPU with the pallas_attn kernel] is present."""
    if _real_jax() is None:
        raise unittest.SkipTest("real jax not importable (stand-in stack)")
    if gpu and not _gpu():
        raise unittest.SkipTest("needs a GPU with the pallas_attn kernel")


class FusedProjectionAlgebra(unittest.TestCase):
    def setUp(self):
        need(gpu=False)

    def test_fused_gemm_equals_the_four_stock_contractions(self):
        """y = x·[Wq|Wk|Wv|Wg] sliced and laid heads-major == the stock einsums 'bqa,ahc->bhqc' ×3 and 'bqc,chv->bqhv' (per element: the
        same 128-term dot products; equal to bf16 rounding — exactly equal on every backend tried, asserted to one bf16 ulp)."""
        import jax
        import jax.numpy as jnp
        from colabdesign_opt.kernels import proj_attn as P
        B, S, C, H, dk, dv = 3, 40, 128, 4, 32, 32
        ks = jax.random.split(jax.random.PRNGKey(0), 5)
        x = jax.random.normal(ks[0], (B, S, C), jnp.float32).astype(jnp.bfloat16)
        q_w, k_w = (jax.random.normal(k, (C, H, dk), jnp.float32).astype(jnp.bfloat16) * 0.09 for k in ks[1:3])
        v_w, g_w = (jax.random.normal(k, (C, H, dv), jnp.float32).astype(jnp.bfloat16) * 0.09 for k in ks[3:5])
        W = P.fused_projection_weights(q_w, k_w, v_w, g_w)
        self.assertEqual(W.shape, (C, 2 * H * dk + 2 * H * dv))
        y = jnp.einsum("bsa,ax->bsx", x, W)
        q, k, v, g = P.split_fused(y, H, dk, dv, gating=True)
        ref = {"q": jnp.einsum("bqa,ahc->bhqc", x, q_w), "k": jnp.einsum("bka,ahc->bhkc", x, k_w), "v": jnp.einsum("bka,ahc->bhkc", x, v_w),
               "g": jnp.einsum("bqc,chv->bqhv", x, g_w)}
        for name, got in (("q", q), ("k", k), ("v", v), ("g", g)):
            r = ref[name]
            self.assertEqual(got.shape, r.shape, name)
            d = float(jnp.max(jnp.abs(got.astype(jnp.float32) - r.astype(jnp.float32))))
            scale = float(jnp.max(jnp.abs(r.astype(jnp.float32))))
            self.assertLessEqual(d, scale * 2.0 ** -7, f"{name}: max|d|={d} scale={scale}")     # one bf16 ulp at the largest magnitude


class RebindingOnTheServedPath(unittest.TestCase):
    def setUp(self):
        need(gpu=True)
        from colabdesign_opt import pallas as KP
        from colabdesign_opt.kernels import proj_attn as P
        self.KP, self.P = KP, P
        KP.install()                                    # the pallas lever first (proj composes on it)
        P.uninstall()

    def tearDown(self):
        self.P.uninstall()

    def _transformed(self):
        import haiku as hk
        import ml_collections
        from colabdesign.af.alphafold.model import modules as CDM
        cfg = ml_collections.ConfigDict({"num_head": 4, "gating": True, "dropout_rate": 0.0, "orientation": "per_row", "shared_dropout": True})
        gc = ml_collections.ConfigDict({"zero_init": False, "subbatch_size": None, "deterministic": True})

        def fwd(x, bias, nb):
            return CDM.Attention(cfg, gc, output_dim=int(x.shape[-1]))(x, x, bias, nb)
        return hk.transform(fwd)

    def _transformed_two_tracers(self):
        import haiku as hk
        import ml_collections
        from colabdesign.af.alphafold.model import modules as CDM
        cfg = ml_collections.ConfigDict({"num_head": 4, "gating": True, "dropout_rate": 0.0, "orientation": "per_row", "shared_dropout": True})
        gc = ml_collections.ConfigDict({"zero_init": False, "subbatch_size": None, "deterministic": True})

        def fwd(x, bias, nb):
            return CDM.Attention(cfg, gc, output_dim=int(x.shape[-1]))(x, x + 0, bias, nb)     # m_data: a distinct tracer with the same values
        return hk.transform(fwd)

    def test_same_parameter_tree_same_values_same_gradients(self):
        import jax
        import jax.numpy as jnp
        B, S, C, H = 6, 96, 128, 4
        ks = jax.random.split(jax.random.PRNGKey(1), 4)
        x = (jax.random.normal(ks[0], (B, S, C), jnp.float32) * 0.7).astype(jnp.bfloat16)
        mask = (jax.random.uniform(ks[1], (B, S)) > 0.1).astype(jnp.float32)
        bias = (1e9 * (mask - 1.0))[:, None, None, :].astype(jnp.bfloat16)
        nb = (jax.random.normal(ks[2], (H, S, S), jnp.float32) * 0.5).astype(jnp.bfloat16)
        served = self._transformed()                    # modules.Attention = the pallas lever's class
        params = served.init(ks[3], x, bias, nb)
        self.P.install()
        fused = self._transformed()                     # modules.Attention = this lever's class on top
        params_fused = fused.init(ks[3], x, bias, nb)
        flat = lambda p: sorted((f"{m}/{n}", tuple(v.shape), str(v.dtype)) for m, d in p.items() for n, v in d.items())  # noqa: E731
        self.assertEqual(flat(params), flat(params_fused))          # the stock parameter tree, unchanged
        w = jax.random.normal(jax.random.PRNGKey(7), (B, S, C), jnp.float32)

        def loss(apply, p, xx):
            return jnp.sum(apply(p, None, xx, bias, nb).astype(jnp.float32) * w)
        out_s = served.apply(params, None, x, bias, nb).astype(jnp.float32)
        out_f = fused.apply(params, None, x, bias, nb).astype(jnp.float32)
        scale = float(jnp.max(jnp.abs(out_s)))
        self.assertLessEqual(float(jnp.max(jnp.abs(out_s - out_f))), 4 * scale * 2.0 ** -7)
        gx_s, gp_s = jax.grad(lambda xx, p: loss(served.apply, p, xx), argnums=(0, 1))(x, params)
        gx_f, gp_f = jax.grad(lambda xx, p: loss(fused.apply, p, xx), argnums=(0, 1))(x, params)

        def close(a, b, what):
            a, b = a.astype(jnp.float32), b.astype(jnp.float32)
            cos = float(jnp.vdot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))
            rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-30))
            self.assertGreater(cos, 0.9999, f"{what}: cosine {cos}")
            self.assertLess(rel, 0.05, f"{what}: max rel {rel}")
        close(gx_s, gx_f, "d q_data")
        for m in gp_s:
            for n in gp_s[m]:
                close(gp_s[m][n], gp_f[m][n], f"d {m}/{n}")
        ev = self.P.evidence()
        self.assertGreaterEqual(ev["served"], 1)
        self.assertEqual(ev["fallback"], 0)
        # the two-tracer form (stock's sub-batched forward passes q_data and m_data as separate slices): TWO GEMMs, same values
        fused2 = self._transformed_two_tracers()
        out_2 = fused2.apply(params, None, x, bias, nb).astype(jnp.float32)
        self.assertLessEqual(float(jnp.max(jnp.abs(out_2 - out_f))), 4 * scale * 2.0 ** -7)


class ComposesOnTheActiveOp(unittest.TestCase):
    def setUp(self):
        need(gpu=True)

    """proj serves through WHICHEVER op the active kernel lever enabled on the serve layer (pallas: F1; triatt: its own) and the softmax
    scale is applied exactly once: the spy op records that proj passes key_dim**-0.5 exactly as the served call does."""

    def test_spy_op_receives_the_calls_and_scale_once(self):
        import jax
        import jax.numpy as jnp
        import haiku as hk
        import ml_collections
        from colabdesign.af.alphafold.model import modules as CDM
        from opt_core.kernels import pallas_attn_serve as F1
        from colabdesign_opt.kernels import proj_attn as P
        P.uninstall(); F1.disable([CDM])
        seen = []
        ref = F1.kernel_module().reference_attention                # a pure-JAX op with the kernel's signature: stands in for 'another lever's op'

        def spy(q, k, v, bias, kmask, scale):
            seen.append(float(scale))
            return ref(q, k, v, bias, kmask, scale).astype(q.dtype)
        led = F1.ledger(min_tokens=0, expected=(F1.HEAD_DIM_LT_16,))
        F1.enable([CDM], ledger=led, all_calls=True, op=spy)      # what a superseding kernel lever does: its own op on the serve layer
        cfg = ml_collections.ConfigDict({"num_head": 4, "gating": True, "dropout_rate": 0.0, "orientation": "per_row", "shared_dropout": True})
        gc = ml_collections.ConfigDict({"zero_init": False, "subbatch_size": None, "deterministic": True})
        B, S, C, H = 3, 40, 128, 4
        ks = jax.random.split(jax.random.PRNGKey(3), 3)
        x = jax.random.normal(ks[0], (B, S, C), jnp.float32).astype(jnp.bfloat16)
        bias = jnp.zeros((B, 1, 1, S), jnp.bfloat16)
        nb = (jax.random.normal(ks[1], (H, S, S), jnp.float32) * 0.5).astype(jnp.bfloat16)
        mk = lambda: hk.transform(lambda xx, bb, nn: CDM.Attention(cfg, gc, output_dim=C)(xx, xx, bb, nn))  # noqa: E731
        served = mk(); params = served.init(ks[2], x, bias, nb)
        seen.clear(); out_served = served.apply(params, None, x, bias, nb).astype(jnp.float32)
        self.assertEqual(seen, [32 ** -0.5])                        # the served path: the op applies key_dim**-0.5
        try:
            P.install()
            before = led.served
            seen.clear(); out_proj = mk().apply(params, None, x, bias, nb).astype(jnp.float32)
            self.assertEqual(seen, [32 ** -0.5])                    # proj passes the scale exactly as the served call does; the op applies it once
            self.assertEqual(led.served, before + 1)                 # proj counted its kernel-served call in the ACTIVE lever's ledger
        finally:
            P.uninstall(); F1.disable([CDM])
        scale = float(jnp.max(jnp.abs(out_served)))
        self.assertLessEqual(float(jnp.max(jnp.abs(out_served - out_proj))), 4 * scale * 2.0 ** -7)     # same op, fused projections
        self.assertEqual(P.evidence()["fallback"], 0)


class InstallContract(unittest.TestCase):
    def setUp(self):
        need(gpu=True)

    def test_requires_pallas_idempotent_restorable_and_one_line(self):
        from colabdesign.af.alphafold.model import modules as CDM
        from opt_core.kernels import pallas_attn_serve as F1
        from colabdesign_opt import pallas as KP
        from colabdesign_opt.kernels import proj_attn as P
        P.uninstall(); F1.disable([CDM])
        with self.assertRaises(P.LeverError):
            P.install()                                 # refuses without the pallas lever's class in place
        KP.install()
        a = P.install(); b = P.install()
        self.assertTrue(a["installed"] and b["installed"])
        self.assertTrue(getattr(CDM.Attention, P.MARKER, False) and getattr(CDM.Attention, "_opt_core_pallas_attn", False))
        self.assertEqual(CDM.Attention.__name__, "Attention")       # haiku derives the module name from the class name
        line = P.lever_line()
        self.assertTrue(re.match(r"^\[colabdesign-opt\] LEVER name=proj state=on impl=fused_qkvg@kit origin=kit .*\bnumerics=precision precision=bf16 requires=attn_kernel active=F1\.pallas_attn traced=\d+$", line), line)
        self.assertTrue(P.off_line("ablated").startswith("[colabdesign-opt] LEVER name=proj state=off reason=ablated impl=fused_qkvg@kit origin=kit"), P.off_line("ablated"))
        for attr in ("install", "installed", "uninstall", "off_line", "evidence", "REFUSALS", "NUMERICS", "EXPECTED_FALLBACKS"):
            self.assertTrue(hasattr(P, attr), attr)               # the kit's lever-module protocol (registry.py)
        self.assertTrue(P.uninstall())
        self.assertFalse(getattr(CDM.Attention, P.MARKER, False))
        self.assertTrue(getattr(CDM.Attention, "_opt_core_pallas_attn", False))   # the pallas lever's class is back


try:
    import pytest  # noqa: F401
except ImportError:                                              # no pytest on the image: the kit's stand-in runs the file with unittest
    from colabdesign_opt.tests import _noptest as pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
