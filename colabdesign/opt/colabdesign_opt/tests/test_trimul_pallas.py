"""Fused TriangleMultiplication kernels (the provider row `cd_trimul` the lever binds through kernels/provider.py: `opt_core.kernels.pallas.cd_trimul.trimul_pallas`,
byte-identical to the kit's private copy kernels/trimul_pallas.py, which left the tree in 0.10.3) and their lever (kernels/trimul_fused.py).

GPU tests (skipped without the GPU backend): the custom_vjp op — forward AND input / mask / parameter gradients — against the op-by-op replica of
stock (`reference`, stock's dtypes) and against an f32/HIGHEST yardstick, both equations, N = 200/500/800 at cz=c=128 (the Evoformer pair
stack), ragged N at cz=c=64 (the template pair stack), f32 activations; the installed lever against the STOCK haiku module (outputs, gradients,
parameter tree, the not_fused fallback bitwise) and the LEVER line's census. CPU tests: the module protocol and the plane assignment.
"""
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:                                                            # pragma: no cover
    HAVE_JAX = False
try:
    import haiku as hk
    import ml_collections
    from colabdesign.af.alphafold.model import modules as CDM, utils as CDU
    HAVE_STACK = HAVE_JAX
except Exception:                                                            # pragma: no cover
    HAVE_STACK = False

from colabdesign_opt.kernels import trimul_fused as L
K = None
if HAVE_JAX:                                                                  # the kernels module imports jax; the protocol tests below need only the lever module
    import importlib
    from colabdesign_opt.kernels import provider as _P
    K = importlib.import_module(_P.BINDINGS[L.LEVER].rows["cd_trimul"])       # the ROW module the lever binds (opt_core.kernels.pallas.cd_trimul.trimul_pallas); the kit's private copy left the tree in 0.10.3

F32 = np.float32
_GPU = None


def gpu() -> bool:
    """The GPU backend is present — asked lazily (never at module import: under one `unittest discover` / pytest process the stand-in tests and
    lever parcompile must not find a jax backend already created by another module's import)."""
    global _GPU
    if _GPU is None:
        try:
            _GPU = bool(HAVE_JAX and jax.default_backend() == "gpu")
        except Exception:                                                    # a CUDA plugin with no visible device raises instead of falling back
            _GPU = False
    return _GPU


def _needs_gpu(cls, stack=False):
    orig = cls.__dict__.get("setUpClass")

    @classmethod
    def setUpClass(k):
        if not gpu():
            raise unittest.SkipTest("needs the GPU backend (Pallas/Triton kernels)")
        if stack and not HAVE_STACK:
            raise unittest.SkipTest("needs the colabdesign stack")
        if orig is not None:
            orig.__func__(k)
    cls.setUpClass = setUpClass
    return cls


def make_params(key, cz, c):
    ks = jax.random.split(key, 12)
    n = lambda k, shape, s: (s * jax.random.normal(k, shape)).astype(jnp.float32)
    return {
        "left_norm_input": {"scale": 1.0 + n(ks[0], (cz,), 0.1), "offset": n(ks[1], (cz,), 0.1)},
        "projection": {"weights": n(ks[2], (cz, 2 * c), cz ** -0.5), "bias": n(ks[3], (2 * c,), 0.1)},
        "gate": {"weights": n(ks[4], (cz, 2 * c), cz ** -0.5), "bias": 1.0 + n(ks[5], (2 * c,), 0.1)},
        "center_norm": {"scale": 1.0 + n(ks[6], (c,), 0.1), "offset": n(ks[7], (c,), 0.1)},
        "output_projection": {"weights": n(ks[8], (c, cz), c ** -0.5), "bias": n(ks[9], (cz,), 0.1)},
        "gating_linear": {"weights": n(ks[10], (cz, cz), cz ** -0.5), "bias": 1.0 + n(ks[11], (cz,), 0.1)},
    }


def make_inputs(seed, N, cz, dtype):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = (2.0 * jax.random.normal(k1, (N, N, cz), jnp.float32) + 0.5).astype(dtype)          # a mean offset: LN statistics matter
    mask = (jax.random.uniform(k2, (N, N)) > 0.05).astype(dtype)
    ct = jax.random.normal(k3, (N, N, cz), jnp.float32).astype(dtype)
    return x, mask, ct


def rel(a, b):
    a = np.asarray(jnp.asarray(a, jnp.float32)); b = np.asarray(jnp.asarray(b, jnp.float32))
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def cos(a, b):
    a = np.asarray(jnp.asarray(a, jnp.float32)).ravel(); b = np.asarray(jnp.asarray(b, jnp.float32)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ============================================================ protocol / pure (CPU)
class TestProtocol(unittest.TestCase):
    def test_module_protocol(self):
        for attr in ("install", "installed", "uninstall", "off_line", "evidence", "REFUSALS", "NUMERICS"):
            self.assertTrue(hasattr(L, attr), attr)
        self.assertEqual(L.NUMERICS, "precision")
        line = L.off_line("ablated")
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=trimul_pallas state=off reason=ablated impl=trimul_pallas@"), line)
        self.assertIn(" origin=core", line)                                                   # the kernels are the shared core's provider row (kernels/provider.py); the adapter is the kit's
        self.assertRegex(L.kernel_impl(), r"^trimul_pallas@[0-9a-f]{8}$")

    @unittest.skipUnless(HAVE_JAX, "jax")
    def test_plane_assignment(self):
        p = make_params(jax.random.PRNGKey(0), 16, 16)
        wo = K.split_params(p, K.EQ_OUT, jnp.float32); wi = K.split_params(p, K.EQ_IN, jnp.float32)
        np.testing.assert_array_equal(np.asarray(wo["p"][0]), np.asarray(p["projection"]["weights"][:, :16]))   # outgoing: plane p = left
        np.testing.assert_array_equal(np.asarray(wi["p"][0]), np.asarray(p["projection"]["weights"][:, 16:]))   # incoming: plane p = right
        with self.assertRaises(ValueError):
            K.make_triangle_multiplication("ikc,kjc->ijc")


# ============================================================ the op (GPU)
@_needs_gpu
class TestKernelsAgainstReference(unittest.TestCase):
    def _arms(self, p, eq):
        hi = jax.lax.Precision.HIGHEST
        return {"kit": lambda x, m: K.triangle_multiplication(x, m, p, equation=eq),
                "stockref": lambda x, m: K.reference(x, m, p, equation=eq),
                "yard": lambda x, m: K.reference(x, m, p, equation=eq, compute_dtype=jnp.float32, precision=hi)}

    def _check(self, N, cz, c, dtype, eq, seed=0, tag=""):
        p = make_params(jax.random.PRNGKey(seed + 1), cz, c)
        x, mask, ct = make_inputs(seed, N, cz, dtype)
        outs = {}
        for name, f in self._arms(p, eq).items():
            fj = jax.jit(lambda x, m, ct, f=f: (lambda y, vjp: (y, vjp(ct.astype(y.dtype))[0]))(*jax.vjp(lambda xx: f(xx, m), x)))
            outs[name] = fj(x, mask, ct)
        (yk, gk), (ys, gs), (yr, gr) = outs["kit"], outs["stockref"], outs["yard"]
        e = dict(y_kit=rel(yk, yr), y_stock=rel(ys, yr), g_kit=rel(gk, gr), g_stock=rel(gs, gr), y_ks=rel(yk, ys), g_ks=rel(gk, gs), cy=cos(yk, ys), cg=cos(gk, gs))
        print(f"[test_trimul] {tag:10s} N={N:4d} cz={cz} c={c} {str(jnp.dtype(dtype)):8s} {eq}: y kit/stock vs yard {e['y_kit']:.2e}/{e['y_stock']:.2e} "
              f"dx {e['g_kit']:.2e}/{e['g_stock']:.2e} | kit-vs-stock y {e['y_ks']:.2e} dx {e['g_ks']:.2e} cos {e['cy']:.6f}/{e['cg']:.6f}")
        self.assertTrue(np.isfinite(np.asarray(yk, F32)).all() and np.isfinite(np.asarray(gk, F32)).all())
        self.assertLess(e["y_kit"], 2.0 * e["y_stock"] + 2e-3, e); self.assertLess(e["g_kit"], 2.0 * e["g_stock"] + 2e-3, e)
        self.assertGreater(e["cy"], 0.999, e); self.assertGreater(e["cg"], 0.999, e)
        return e

    def test_evoformer_sizes_bf16(self):
        print()
        for N in (200, 500, 800):
            for eq in K.EQUATIONS:
                self._check(N, 128, 128, jnp.bfloat16, eq, tag="evoformer")

    def test_template_stack_ragged_and_f32(self):
        print()
        for eq in K.EQUATIONS:
            self._check(331, 64, 64, jnp.bfloat16, eq, seed=3, tag="template")          # ragged N (not a multiple of the tile), cz=c=64
            self._check(75, 128, 128, jnp.bfloat16, eq, seed=5, tag="tiny")              # N·N < a few tiles per row
            e = self._check(150, 128, 128, jnp.float32, eq, seed=7, tag="f32")           # f32 activations (tf32-class products, as XLA DEFAULT)
            self.assertLess(e["y_kit"], 5e-3)

    def test_mask_and_parameter_gradients(self):
        N, cz, c = 96, 64, 64
        for eq in K.EQUATIONS:
            p = make_params(jax.random.PRNGKey(11), cz, c)
            x, mask, ct = make_inputs(13, N, cz, jnp.bfloat16)
            mask32 = mask.astype(jnp.float32)
            loss_k = lambda m, pp: jnp.sum(K.triangle_multiplication(x, m, pp, equation=eq).astype(jnp.float32) * ct)
            loss_r = lambda m, pp: jnp.sum(K.reference(x, m, pp, equation=eq).astype(jnp.float32) * ct)
            (gm_k, gp_k), (gm_r, gp_r) = jax.jit(jax.grad(loss_k, argnums=(0, 1)))(mask32, p), jax.jit(jax.grad(loss_r, argnums=(0, 1)))(mask32, p)
            self.assertLess(rel(gm_k, gm_r), 3e-2, f"{eq} dmask")
            self.assertGreater(cos(gm_k, gm_r), 0.999, f"{eq} dmask")
            flat_k = jax.tree_util.tree_leaves_with_path(gp_k); flat_r = dict(jax.tree_util.tree_leaves_with_path(gp_r))
            for path, gk in flat_k:
                gr = flat_r[path]
                self.assertLess(rel(gk, gr), 3e-2, f"{eq} d{jax.tree_util.keystr(path)}")
            # integer mask: float0 cotangent, no error
            mi = (mask32 > 0).astype(jnp.int32)
            y = jax.jit(jax.grad(lambda xx: jnp.sum(K.triangle_multiplication(xx, mi, p, equation=eq).astype(jnp.float32))))(x)
            self.assertTrue(np.isfinite(np.asarray(y, F32)).all())

    def test_deterministic(self):
        p = make_params(jax.random.PRNGKey(1), 128, 128)
        x, mask, ct = make_inputs(0, 256, 128, jnp.bfloat16)
        f = jax.jit(lambda x: jax.vjp(lambda xx: K.triangle_multiplication(xx, mask, p, equation=K.EQ_OUT), x)[1](ct)[0])
        a, b = np.asarray(f(x), F32), np.asarray(f(x), F32)
        np.testing.assert_array_equal(a, b)


# ============================================================ the lever against the STOCK haiku module (GPU + the stack)
def _cfgs(eq, c, fused=True):
    cfg = ml_collections.ConfigDict({"equation": eq, "num_intermediate_channel": c, "fuse_projection_weights": fused, "dropout_rate": 0.25,
                                     "orientation": "per_row", "shared_dropout": True})
    gc = ml_collections.ConfigDict({"bfloat16": True, "bfloat16_output": False, "multimer_mode": True, "subbatch_size": None, "use_remat": True,
                                    "zero_init": True, "use_dgram": False, "deterministic": False})
    return cfg, gc


def _transformed(klass, eq, c, name, fused=True):
    cfg, gc = _cfgs(eq, c, fused)

    def f(act, mask):
        with CDU.bfloat16_context():
            return klass(cfg, gc, name=name)(act, mask)
    return hk.transform(f)


def _randomised(params, seed):
    leaves, td = jax.tree_util.tree_flatten_with_path(params)
    ks = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    out = []
    for k, (path, v) in zip(ks, leaves):
        nm = jax.tree_util.keystr(path)
        r = jax.random.normal(k, v.shape, jnp.float32)
        r = 1.0 + 0.1 * r if "scale" in nm else (r * (v.shape[0] ** -0.5) if "weights" in nm else (1.0 + 0.1 * r if ("gat" in nm and "bias" in nm) else 0.1 * r))
        out.append(r.astype(v.dtype))
    return jax.tree_util.tree_unflatten(td, out)


@lambda c: _needs_gpu(c, stack=True)
class TestLeverAgainstStockModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.uninstall()
        cls.stock_cls = CDM.TriangleMultiplication
        L.install()
        assert L.installed() and CDM.TriangleMultiplication is not cls.stock_cls
        cls.kit_cls = CDM.TriangleMultiplication

    @classmethod
    def tearDownClass(cls):
        L.uninstall()
        assert CDM.TriangleMultiplication is cls.stock_cls and not L.installed()

    def test_parameter_tree_outputs_gradients_and_census(self):
        print()
        before = dict(L.LEDGER.counts())
        for eq, name in ((K.EQ_OUT, "triangle_multiplication_outgoing"), (K.EQ_IN, "triangle_multiplication_incoming")):
            for N, c in ((200, 128), (177, 64)):
                cz = c
                stock, kit = _transformed(self.stock_cls, eq, c, name), _transformed(self.kit_cls, eq, c, name)
                x, mask, ct = make_inputs(21, N, cz, jnp.bfloat16)
                p_stock, p_kit = stock.init(jax.random.PRNGKey(1), x, mask), kit.init(jax.random.PRNGKey(1), x, mask)
                self.assertEqual(jax.tree_util.tree_structure(p_stock), jax.tree_util.tree_structure(p_kit))
                for (pa, a), (pb, b) in zip(jax.tree_util.tree_leaves_with_path(p_stock), jax.tree_util.tree_leaves_with_path(p_kit)):
                    self.assertEqual((jax.tree_util.keystr(pa), a.shape, a.dtype), (jax.tree_util.keystr(pb), b.shape, b.dtype))
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))                        # same initialisers, same key → same init
                params = _randomised(p_stock, 5)
                run = lambda tr: jax.jit(lambda x, m, ct: (lambda y, vjp: (y, vjp(ct)[0]))(*jax.vjp(lambda xx: tr.apply(params, None, xx, m), x)))(x, mask, ct)
                (ys, gs), (yk, gk) = run(stock), run(kit)
                p_rel = {k.split("/")[-1]: v for k, v in params.items()}
                yard = jax.jit(lambda x, m, ct: (lambda y, vjp: (y, vjp(ct.astype(jnp.float32))[0]))(*jax.vjp(
                    lambda xx: K.reference(xx, m, p_rel, equation=eq, compute_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST), x)))(x, mask, ct)
                e = (rel(yk, yard[0]), rel(ys, yard[0]), rel(gk, yard[1]), rel(gs, yard[1]), rel(yk, ys), rel(gk, gs), cos(yk, ys), cos(gk, gs))
                print(f"[test_trimul] haiku N={N} c={c} {eq}: y kit/stock vs yard {e[0]:.2e}/{e[1]:.2e} dx {e[2]:.2e}/{e[3]:.2e} | kit-vs-stock y {e[4]:.2e} dx {e[5]:.2e} cos {e[6]:.6f}/{e[7]:.6f}")
                self.assertLess(e[0], 2 * e[1] + 2e-3); self.assertLess(e[2], 2 * e[3] + 2e-3)
                self.assertGreater(e[6], 0.999); self.assertGreater(e[7], 0.999)
        after = L.LEDGER.counts()
        served = sum(v for k, v in after.items() if k.startswith("served")) - sum(v for k, v in before.items() if k.startswith("served"))
        self.assertEqual(served, 4 * 2, after)                                                               # init + apply trace per (eq, size)
        line = L.exit_line()
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=trimul_pallas state=on impl=trimul_pallas@"), line)
        for part in (" origin=core ", " numerics=precision ", " precision=bf16 ", " served=", "xC128xE0:", "xC64xE1:", " provider=opt_core.kernels.pallas@", " word=", " row=cd_trimul"):
            self.assertIn(part, line)
        self.assertTrue(L.evidence()["gate_ok"], L.evidence())

    def test_not_fused_config_runs_stock_program(self):
        eq, name = K.EQ_OUT, "triangle_multiplication_outgoing"
        stock, kit = _transformed(self.stock_cls, eq, 64, name, fused=False), _transformed(self.kit_cls, eq, 64, name, fused=False)
        x, mask, _ = make_inputs(4, 64, 64, jnp.bfloat16)
        params = _randomised(stock.init(jax.random.PRNGKey(2), x, mask), 9)
        before = L.LEDGER.counts().get(f"fallback:{L.NOT_FUSED}", 0)
        ys = jax.jit(lambda x, m: stock.apply(params, None, x, m))(x, mask)
        yk = jax.jit(lambda x, m: kit.apply(params, None, x, m))(x, mask)
        np.testing.assert_array_equal(np.asarray(ys, F32), np.asarray(yk, F32))
        self.assertEqual(L.LEDGER.counts().get(f"fallback:{L.NOT_FUSED}", 0) - before, 1)


try:
    import pytest
except ImportError:                                                                                            # pragma: no cover — the image has no pytest
    from colabdesign_opt.tests import _noptest as pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
