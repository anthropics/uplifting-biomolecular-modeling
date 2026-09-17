"""F8's op (mosaic_fast/trimul_provider.py): joltz's triangle multiplication bound to the shared core's kernel provider by TIER WORD.
The wiring on any host (constants; configure/reset words refused by name; P7 halfpair installs no dtype route over the op's body), the
parameter mapping and the unserved-layout names on CPU, the call-kind dispatch on CPU with the provider's answer replaced by named stand-ins
(the un-differentiated call runs the `fwd` selection, a differentiated call the `fwdbwd` selection inside jax.vjp — F6's body by name where a
selection says xla — with gradients equal to F6's, under jax.checkpoint too, float and bool masks, the census words), the provider's CLASS
CONTRACTS for this module's call classes on both cards read from the core's own table (a tier word resolves to a row the provider carries for
the op; `exact` is XLA for this family; a kernel row the tier word names measured at or above XLA in its cell — no test names the row of the
day), and a GPU-gated numerics proof in a child process (values and checkpointed gradients vs F6 for both dtypes, both directions, both tier
words; the kernels land in exactly the executables whose selection names a kernel row) that skips at ~0 cost on CPU-only hosts."""
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

from mosaic_opt.tests import _stubs

OP_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "trimul_provider.py")


def _importable(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _load_op(name="trimul_provider_under_test"):
    spec = importlib.util.spec_from_file_location(name, OP_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stubs._standins()):
        spec.loader.exec_module(mod)
    return mod


def _gpu_visible():
    """(visible, why not) for THIS process without importing jax here: the launch environment must not hide the GPU or pin the CPU platform,
    and a CUDA device must be enumerable (nvidia-smi)."""
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() in ("cpu",):
        return False, "JAX_PLATFORMS=cpu at launch"
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):
        return False, "CUDA_VISIBLE_DEVICES hides every device"
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"nvidia-smi not runnable ({type(e).__name__})"
    ok = r.returncode == 0 and "GPU 0" in r.stdout
    return ok, ("" if ok else "nvidia-smi lists no GPU")


GPU_OK, GPU_WHY = _gpu_visible()


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestProviderOpWiring(unittest.TestCase):

    def test_constants_and_words(self):
        OP = _load_op("trimul_provider_constants")
        self.assertEqual((OP.LEVER, OP.MODULE, OP.SETTING, OP.WORDS, OP.UNIT), ("F8", "mosaic.fast.trimul_provider", "fast", ("fast", "big"), "pair"))
        self.assertEqual(OP.FORMS, {"bf16": "af2", "f32": "joltz"}); self.assertEqual(OP.CALL_KINDS, ("fwd", "fwdbwd")); self.assertEqual(OP.LN_EPS, 1e-5)
        self.assertTrue(OP.VERSION.startswith("trimul_provider/"))
        self.assertEqual(OP.trimul.__serves_dtypes__, ("float32", "bfloat16"))          # P7 halfpair reads it: no dtype route over this body
        src = open(OP_KIT_FILE, encoding="utf-8").read()
        for pinned in ("cd_trimul", "fpf_trimul", "native_xla", "prefer=", "min_kernel_pixels", "MIN_KERNEL", "TILES"):   # the kit names no row, no pin, no size rule, no tile table
            self.assertNotIn(pinned, src, pinned)

    def test_configure_words_and_reset(self):
        OP = _load_op("trimul_provider_words")
        self.assertEqual((OP.settings()["on"], OP.settings()["word"]), (False, None))
        for w in (None, "fast", "big", " BIG "):
            s = OP.configure(w) if w is None else OP.configure(word=w)
            self.assertEqual((s["on"], s["word"]), (True, (w or "fast").strip().lower()))
        for bad in ("exact", "cd", "primal_kernel", "xla", "tier", ""):
            with self.assertRaises(OP.Refusal) as cm:
                OP.configure(word=bad)
            self.assertEqual(cm.exception.reason, "unknown_word", bad); self.assertEqual(cm.exception.kind, "unknown_word")
        OP.CENSUS["x:body=kernel"] += 2; OP.CENSUS["y:body=xla(cell)"] += 1; OP.ROWS["f32:a/b"] += 3; OP.CELLS.add("k")
        c = OP.census()
        self.assertEqual((c["kernel_calls"], c["xla_calls"], c["calls"], c["rows"], c["cells"]), (2, 1, 3, {"f32:a/b": 3}, ["k"]))
        s = OP.reset()
        self.assertEqual((s["on"], s["word"], OP.census()["calls"], OP.census()["cells"]), (False, None, 0, []))
        with self.assertRaises(OP.Refusal) as cm:                                           # a body carried under F8's name must be configured first
            OP.trimul(object(), types.SimpleNamespace(shape=(1, 8, 8, 16), dtype="float32"), None, "outgoing")
        self.assertEqual(cm.exception.reason, "not_configured")

    def test_hlo_kernel_call_counter(self):
        OP = _load_op("trimul_provider_hlo")
        text = "\n".join([
            'ROOT %cc0 = f32[1] custom-call(%a), custom_call_target="__gpu$xla.gpu.triton", backend_config={name = "trimul_k1_prologue"}',
            '%cc1 = bf16[1] custom-call(%b), custom_call_target="__gpu$xla.gpu.triton", backend_config={name = "trimul_epilogue_f32"}',
            '%cc2 = f32[1] custom-call(%c), custom_call_target="__gpu$xla.gpu.triton", backend_config={name = "triattn_fwd_f32"}',
            '%cc3 = f32[1] custom-call(%d), custom_call_target="__cublas$lt$matmul"'])
        c = OP.hlo_kernel_calls(text)
        self.assertEqual((c["kernel_calls"], c["named_calls"]), (3, 2)); self.assertEqual(c["targets"], {"__gpu$xla.gpu.triton": 3})
        self.assertEqual(c["by_name"], {"trimul_k1_prologue": 1, "trimul_epilogue_f32": 1})


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestHalfpairInstallsNoRouteOverTheOp(unittest.TestCase):
    def test_p7_reads_serves_dtypes(self):
        """P7 halfpair's trimul route: a served body declaring bfloat16 among its dtypes gets NO dtype route (word f8_<F8's word>)."""
        hp_file = os.path.join(_stubs.KIT, "mosaic_fast", "halfpair.py")
        spec = importlib.util.spec_from_file_location("halfpair_route_under_test", hp_file)
        H = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, _stubs._standins()):
            spec.loader.exec_module(H)
            body = lambda module, x, mask, direction: None  # noqa: E731
            body.__serves_dtypes__ = ("float32", "bfloat16")
            TL = types.SimpleNamespace(installed=lambda: True, STATE={"on": True, "body": body, "body_word": "fast"}, set_body=mock.Mock())
            with mock.patch.object(H, "_module", lambda name: TL):
                self.assertEqual(H._engage_trimul_route(), "f8_fast")
            TL.set_body.assert_not_called()


@unittest.skipUnless(_importable("jax", "equinox", "joltz", "opt_core", "mosaic"), "needs the kit venv (jax + joltz + opt_core; CPU is enough)")
class TestProviderParamsCPU(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import numpy as np
        import jax.numpy as jnp
        from joltz import TriangleMultiplicationOutgoing as TO
        from joltz.backend import Linear, LayerNorm
        from mosaic.fast import trimul_provider as OP
        self.np, self.jnp, self.OP, self.TO = np, jnp, OP, TO
        rng = np.random.default_rng(0)
        self.C = C = 16
        self.lin = lambda o, i, bias=False: Linear(weight=jnp.asarray(rng.normal(size=(o, i)), jnp.float32), bias=jnp.asarray(rng.normal(size=(o,)), jnp.float32) if bias else None)
        self.lnm = lambda c, eps=1e-5, bias=True: LayerNorm(weight=jnp.full((c,), 2.0, jnp.float32), bias=jnp.full((c,), .5, jnp.float32) if bias else None, eps=eps)
        self.mod = lambda bias=False, eps=1e-5, lnbias=True: TO(norm_in=self.lnm(C, eps, lnbias), p_in=self.lin(2 * C, C, bias), g_in=self.lin(2 * C, C, bias),
                                                             norm_out=self.lnm(C, eps, lnbias), p_out=self.lin(C, C, bias), g_out=self.lin(C, C, bias))

    def test_math_layout_planes_and_the_incoming_swap(self):
        m = self.mod(); C = self.C
        eq = lambda a, b: bool((self.jnp.asarray(a) == self.jnp.asarray(b)).all())  # noqa: E731
        po, pi = self.OP.provider_params(m, "outgoing"), self.OP.provider_params(m, "incoming")
        self.assertTrue(eq(po["left_w"], m.p_in.weight[:C].T) and eq(po["right_w"], m.p_in.weight[C:].T))          # plane a = the first C output channels, as x @ w
        self.assertTrue(eq(po["left_gate_w"], m.g_in.weight[:C].T) and eq(po["right_gate_w"], m.g_in.weight[C:].T))
        self.assertTrue(eq(pi["left_w"], po["right_w"]) and eq(pi["right_w"], po["left_w"]) and eq(pi["left_gate_w"], po["right_gate_w"]))   # incoming: left = plane b
        self.assertTrue(eq(po["out_w"], m.p_out.weight.T) and eq(po["gate_w"], m.g_out.weight.T))
        self.assertTrue(eq(po["ln_in_scale"], m.norm_in.weight) and eq(po["ln_c_offset"], m.norm_out.bias))
        for k in ("left_w", "right_w", "left_gate_w", "right_gate_w", "out_w", "gate_w"):
            self.assertEqual(tuple(po[k].shape), (C, C), k)
        self.assertNotIn("left_b", po)                                                                            # the module has no biases: none passed
        z = self.OP.provider_params(self.mod(lnbias=False), "outgoing")
        self.assertTrue(bool((z["ln_in_offset"] == 0).all()) and bool((z["ln_c_offset"] == 0).all())); self.assertEqual(str(z["ln_in_scale"].dtype), "float32")

    def test_unserved_layouts_are_named(self):
        x = self.jnp.zeros((1, 8, 8, self.C), self.jnp.float32)
        self.assertIsNone(self.OP.unserved_reason(self.mod(), x))
        self.assertIsNone(self.OP.unserved_reason(self.mod(), x.astype(self.jnp.bfloat16)))
        self.assertEqual(self.OP.unserved_reason(self.mod(bias=True), x), "bias_p_in")
        self.assertEqual(self.OP.unserved_reason(self.mod(eps=1e-6), x), "norm_in_eps_1e-06")
        self.assertEqual(self.OP.unserved_reason(self.mod(), x.astype(self.jnp.float16)), "dtype_float16")
        self.assertEqual(self.OP.unserved_reason(self.mod(), self.jnp.zeros((1, 8, 6, self.C))), "x_not_square_pair")


DISPATCH_CHILD = r'''
import json, os, sys, types
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from joltz import TriangleMultiplicationOutgoing as TO, TriangleMultiplicationIncoming as TI
from joltz.backend import Linear, LayerNorm
from mosaic.fast import trimul_layout as F6, trimul_provider as OP
from opt_core.kernels import pallas as P
MARK = {"fwd": 1000.0, "fwdbwd": 0.0}                                  # the un-differentiated stand-in row is visible in the value; the differentiated one must keep F6's value AND gradients
rng = np.random.default_rng(3)
lin = lambda o, i: Linear(weight=jnp.asarray(rng.normal(0, i ** -.5, (o, i)), jnp.float32), bias=None)
lnm = lambda c: LayerNorm(weight=jnp.asarray(1 + .1 * rng.normal(size=(c,)), jnp.float32), bias=jnp.asarray(.1 * rng.normal(size=(c,)), jnp.float32), eps=1e-5)
ARMS = {}                                                              # kind -> arm the stand-in provider answers
def fake_select(dt, n, c, equation, kind, word=None):
    arm = ARMS[kind]
    return types.SimpleNamespace(row=arm.split(":")[0].split("@")[0], arm=arm, cell_key=f"test|{dt}|{equation}|N<={n}|{kind}", word=word or OP.STATE["word"], candidates=(arm,))
def fake_face(sel, kind, module, x, mask, equation):
    return F6.trimul_cmajor(module, x, mask, equation) + MARK[kind], sel.arm
OP.select, OP.face_call = fake_select, fake_face
tol = 1e-5
out = {"cases": 0}
B, N, C = 2, 12, 16
for cls, d in ((TO, "outgoing"), (TI, "incoming")):
    m = cls(norm_in=lnm(C), p_in=lin(2 * C, C), g_in=lin(2 * C, C), norm_out=lnm(C), p_out=lin(C, C), g_out=lin(C, C))
    params, static = eqx.partition(m, eqx.is_array)
    x = jnp.asarray(rng.normal(size=(B, N, N, C)), jnp.float32); cot = jnp.asarray(rng.normal(size=(B, N, N, C)), jnp.float32)
    for mk in ("float", "bool"):
        mask_f = jnp.asarray(rng.random((B, N, N)), jnp.float32); mask = mask_f if mk == "float" else (mask_f > .4)
        ref = F6.trimul_cmajor(m, x, mask, d)
        for arms in ({"fwd": "rowK", "fwdbwd": "rowK"}, {"fwd": "rowK", "fwdbwd": "xla"}, {"fwd": "xla", "fwdbwd": "rowK"}, {"fwd": "xla", "fwdbwd": "xla"}):
            ARMS.clear(); ARMS.update(arms)
            for word in ("fast", "big"):
                OP.reset(); OP.configure(word=word)
                # (1) the un-differentiated call: the fwd selection (marked stand-in) or F6's body by name
                y = jax.jit(lambda xx: OP.trimul(m, xx, mask, d))(x)
                want = ref + (MARK["fwd"] if arms["fwd"] != "xla" else 0.0)
                assert float(jnp.abs(y - want).max()) <= tol * max(1.0, float(jnp.abs(want).max())), (d, mk, arms, word, "fun")
                # (2) the differentiated call: primal from the fwdbwd selection (no fwd mark), cotangents == jax.vjp(F6) to reassociation
                f_op = lambda xx, pp: OP.trimul(eqx.combine(pp, static), xx, mask, d)
                val, g = jax.jit(jax.value_and_grad(lambda xx, pp: jnp.sum(f_op(xx, pp) * cot), argnums=(0, 1)))(x, params)
                _, vjp = jax.vjp(lambda xx, pp: F6.trimul_cmajor(eqx.combine(pp, static), xx, mask, d), x, params)
                gw = vjp(cot)
                assert abs(float(val) - float(jnp.sum(ref * cot))) <= tol * max(1.0, abs(float(val))) * N * N, (d, mk, arms, word, "primal", float(val), float(jnp.sum(ref * cot)))
                close = lambda a, b: float(jnp.abs(a - b).max()) <= tol * max(1.0, float(jnp.abs(b).max()))
                assert close(g[0], gw[0]), (d, mk, arms, word, "dx")
                for a_, b_ in zip(jax.tree_util.tree_leaves(g[1]), jax.tree_util.tree_leaves(gw[1])):
                    assert close(a_, b_), (d, mk, arms, word, "dparams")
                if mk == "float":                                       # the float mask's cotangent is real (a wrong transpose or zeros would show)
                    gm = jax.jit(jax.grad(lambda mm: jnp.sum(OP.trimul(m, x, mm, d) * cot)))(mask)
                    _, vjm = jax.vjp(lambda mm: F6.trimul_cmajor(m, x, mm, d), mask)
                    assert close(gm, vjm(cot)[0]) and float(jnp.abs(gm).max()) > 0, (d, arms, word, "dmask")
                # (3) under jax.checkpoint (the trunk's per-block remat): same value, same gradients
                vc, gc = jax.jit(jax.value_and_grad(lambda xx: jnp.sum(jnp.sin(jax.checkpoint(lambda z: OP.trimul(m, z, mask, d))(xx)) * cot)))(x)
                vw, gw2 = jax.value_and_grad(lambda xx: jnp.sum(jnp.sin(F6.trimul_cmajor(m, xx, mask, d)) * cot))(x)
                assert abs(float(vc) - float(vw)) <= tol * max(1.0, abs(float(vw))) * N and close(gc, gw2), (d, mk, arms, word, "checkpoint")
                cen = OP.census()
                kernel = arms["fwd"] != "xla" or arms["fwdbwd"] != "xla"
                assert cen["calls"] >= 1 and (cen["kernel_calls"] == cen["calls"] if kernel else cen["xla_calls"] == cen["calls"]), (arms, cen)
                assert all((f":word={word}:" in k) and (f":fwd={arms['fwd']}:ad={arms['fwdbwd']}:" in k) for k in cen["words"]), cen["words"]
                assert all(k.endswith(":body=kernel") if kernel else k.endswith(":body=xla(cell)") for k in cen["words"]), cen["words"]
                tr = cen["traces"]
                if kernel:
                    assert tr.get("fun:%s" % arms["fwd"], 0) >= 1 and tr.get("fwd:%s" % arms["fwdbwd"], 0) >= 1 and tr.get("bwd", 0) >= 1, (arms, tr)
                assert set(cen["rows"]) == {"f32:%s/%s" % (arms["fwd"], arms["fwdbwd"])}, cen["rows"]
                out["cases"] += 1
# a module layout outside the mapping and a provider refusal: F6's body, counted by name
OP.reset(); OP.configure(word="fast"); ARMS.update({"fwd": "rowK", "fwdbwd": "rowK"})
mb = TO(norm_in=lnm(C), p_in=Linear(weight=lin(2 * C, C).weight, bias=jnp.zeros((2 * C,), jnp.float32)), g_in=lin(2 * C, C), norm_out=lnm(C), p_out=lin(C, C), g_out=lin(C, C))
xb = jnp.asarray(rng.normal(size=(1, N, N, C)), jnp.float32); maskb = jnp.ones((1, N, N), jnp.float32)
yb = OP.trimul(mb, xb, maskb, "outgoing"); assert float(jnp.abs(yb - F6.trimul_cmajor(mb, xb, maskb, "outgoing")).max()) == 0.0
assert any(k.endswith(":body=xla(bias_p_in)") for k in OP.census()["words"]), OP.census()["words"]
def refusing_select(*a, **k): raise P.Refusal("no GPU here", kind="backend_not_gpu", row=None, fallback="xla")
OP.select = refusing_select; OP.reset(); OP.configure(word="fast")
m0 = TO(norm_in=lnm(C), p_in=lin(2 * C, C), g_in=lin(2 * C, C), norm_out=lnm(C), p_out=lin(C, C), g_out=lin(C, C))
y0 = OP.trimul(m0, xb, maskb, "outgoing"); assert float(jnp.abs(y0 - F6.trimul_cmajor(m0, xb, maskb, "outgoing")).max()) == 0.0
assert any(k.endswith(":body=xla(refused:backend_not_gpu)") for k in OP.census()["words"]) and OP.census()["xla_calls"] == 1, OP.census()
print(json.dumps(out))
'''


@unittest.skipUnless(_importable("jax", "equinox", "joltz", "opt_core", "mosaic"), "needs the kit venv (jax + joltz + opt_core; CPU is enough)")
class TestCallKindDispatchCPU(unittest.TestCase):
    def test_fun_runs_the_fwd_selection_and_grad_runs_the_fwdbwd_selection(self):
        """In a child interpreter (module-level stand-ins for the provider's answer must not leak): 2 directions x 2 masks x 4 selection pairs x
        2 tier words = 32 cases, plus the unserved-layout and provider-refusal bodies."""
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        r = subprocess.run([sys.executable, "-c", DISPATCH_CHILD], capture_output=True, text=True, timeout=1800, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-6000:])
        self.assertIn('"cases": 32', r.stdout)


@unittest.skipUnless(_importable("jax", "opt_core"), "needs jax + opt_core importable (the provider's table is read through its own select())")
class TestProviderCellContract(unittest.TestCase):
    """What the tier words resolve to for THIS module's call classes, per card, from the core's own table — asserted as class contracts (the
    row is one the provider carries for the op; `exact` is XLA for this family; a kernel row a tier word names inside a measured cell measured
    at or above XLA there), printed as a table; never the name of the row of the day."""

    def test_tier_words_resolve_per_card(self):
        import jax.numpy as jnp
        from opt_core.kernels import pallas as P
        line = P.jax_line_of(__import__("jax").__version__) if hasattr(P, "jax_line_of") else None
        rows_known = set(getattr(P, "ROW_NAMES", ())) | set(getattr(P, "ROWS", {}).keys()) | {"xla"}
        table, n_checked = [], 0
        for cc in ("9.0", "8.0"):
            for dt, form in (("bf16", "af2"), ("f32", "joltz")):
                for equation in ("outgoing", "incoming"):
                    fam = P.family("trimul", form=form, unit="pair", c=128, c_hidden=128, equation=equation)
                    for kind in ("fwd", "fwdbwd"):
                        for n in (200, 500, 800):
                            for word in ("fast", "big", "exact"):
                                sel = P.select(line, cc, jnp.bfloat16 if dt == "bf16" else jnp.float32, "trimul", fam, n, kind, word=word)
                                row = sel.row
                                self.assertIn(row.split("@")[0], rows_known, (cc, dt, equation, kind, n, word, sel))
                                if word == "exact":
                                    self.assertEqual(row, "xla", ("exact is XLA (F6's body) for this family", cc, dt, equation, kind, n, sel))
                                cell = getattr(sel, "cell", None) or {}
                                if row != "xla" and cell.get("measured") and sel.x_stock is not None:
                                    self.assertGreaterEqual(float(sel.x_stock), 0.95, ("a tier word's kernel row measured at or above XLA in its cell", cc, dt, equation, kind, n, word, sel))
                                if word != "exact":
                                    table.append(f"{cc}|{dt}|{equation}|{kind}|N{n}|{word} -> {sel.arm} @ {getattr(sel, 'cell_key', None)} x_stock={getattr(sel, 'x_stock', None)}")
                                n_checked += 1
        print("\n" + "\n".join(table))
        self.assertEqual(n_checked, 2 * 2 * 2 * 2 * 3 * 3)


PROVIDER_NUMERICS_CHILD = r'''
import itertools, json, sys
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from joltz import TriangleMultiplicationOutgoing as TO, TriangleMultiplicationIncoming as TI
from joltz.backend import Linear, LayerNorm
from mosaic.fast import trimul_layout as F6, trimul_provider as OP
assert jax.default_backend() == "gpu", jax.default_backend()
print("PRECISION " + json.dumps({"jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision), "x64": bool(jax.config.jax_enable_x64)}))
rng = np.random.default_rng(0)
rel = lambda a, b: float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)) / max(np.linalg.norm(np.asarray(b, np.float64)), 1e-30))
lin = lambda o, i: Linear(weight=jnp.asarray(rng.normal(0, i ** -.5, (o, i)), jnp.float32), bias=None)
lnm = lambda c: LayerNorm(weight=jnp.asarray(1 + .1 * rng.normal(size=(c,)), jnp.float32), bias=jnp.asarray(.1 * rng.normal(size=(c,)), jnp.float32), eps=1e-5)
cast = lambda tree, dt: jax.tree_util.tree_map(lambda a: a.astype(dt) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a, tree)
TOL = {"float32": dict(y=3e-3, g=3e-3), "bfloat16": dict(y=3e-2, g=6e-2)}      # the rows' classes: tf32 products on f32 operands; bf16 products (f32 accumulation, one rounding) on bf16 operands vs F6's bf16 XLA arithmetic
worst, cases, table = {}, 0, []
C = 128
for (cls, d), N, dtn, word in itertools.product([(TO, "outgoing"), (TI, "incoming")], (48, 256), ("float32", "bfloat16"), ("fast", "big")):
    dt = jnp.dtype(dtn)
    m32 = cls(norm_in=lnm(C), p_in=lin(2 * C, C), g_in=lin(2 * C, C), norm_out=lnm(C), p_out=lin(C, C), g_out=lin(C, C))
    m = cast(m32, dt) if dtn == "bfloat16" else m32
    x = jnp.asarray(rng.normal(0, 2, (1, N, N, C)) + 1, jnp.float32).astype(dt); cot = jnp.asarray(rng.normal(size=(1, N, N, C)), jnp.float32).astype(dt)
    live = np.ones((N,), bool); live[-5:] = False
    mask = jnp.asarray(np.broadcast_to(live[:, None] & live[None, :], (1, N, N))).astype(dt)
    OP.reset(); OP.configure(word=word)
    p, st = eqx.partition(m, eqx.is_array)
    y_ref = jax.jit(lambda xx: F6.trimul_cmajor(m, xx, mask, d))(x); y_op = jax.jit(lambda xx: OP.trimul(m, xx, mask, d))(x)
    step = lambda f: jax.jit(jax.grad(lambda xx, pp: jnp.sum(jax.checkpoint(lambda z, q: f(eqx.combine(q, st), z, mask, d))(xx, pp).astype(jnp.float32) * cot), argnums=(0, 1)))
    g_op = step(OP.trimul)(x, p); g_ref = step(F6.trimul_cmajor)(x, p)
    r = dict(y=rel(y_op.astype(jnp.float32), y_ref.astype(jnp.float32)), dx=rel(g_op[0].astype(jnp.float32), g_ref[0].astype(jnp.float32)),
             dp=max(rel(a.astype(jnp.float32), b.astype(jnp.float32)) for a, b in zip(jax.tree_util.tree_leaves(g_op[1]), jax.tree_util.tree_leaves(g_ref[1]))))
    assert y_op.shape == y_ref.shape and y_op.dtype == y_ref.dtype and bool(jnp.isfinite(y_op.astype(jnp.float32)).all()), (d, N, dtn, word)
    cen = OP.census()
    fwd_hlo = OP.hlo_kernel_calls(jax.jit(lambda xx: OP.trimul(m, xx, mask, d)).lower(x).compile().as_text())["kernel_calls"]
    grad_hlo = OP.hlo_kernel_calls(step(OP.trimul).lower(x, p).compile().as_text())["kernel_calls"]
    rows = list(cen["rows"])
    assert len(rows) == 1, cen
    fwd_arm, ad_arm = rows[0].split(":", 1)[1].split("/")
    t = TOL[dtn]
    ok_y = r["y"] < (t["y"] if fwd_arm != "xla" else 1e-5)
    ok_g = r["dx"] < (t["g"] if ad_arm != "xla" else 1e-4) and r["dp"] < (t["g"] if ad_arm != "xla" else 1e-4)
    hlo_ok = (fwd_hlo >= 1) == (fwd_arm != "xla") and (grad_hlo >= 1) == (ad_arm != "xla")   # the kernels land in exactly the executables whose selection names a kernel row
    table.append(dict(direction=d, N=N, dtype=dtn, word=word, fwd=fwd_arm, ad=ad_arm, rel=r, fwd_kernel_calls=fwd_hlo, grad_kernel_calls=grad_hlo, traces=cen["traces"], cells=cen["cells"]))
    assert ok_y and ok_g and hlo_ok, table[-1]
    for k, v in r.items(): worst[k] = max(worst.get(k, 0.0), v)
    cases += 1
print("TABLE " + json.dumps(table))
print(json.dumps({"cases": cases, "worst": worst, "probe": OP.probe()}))
'''


@unittest.skipUnless(_importable("jax", "opt_core", "mosaic", "equinox", "joltz") and GPU_OK, f"needs the kit venv on a GPU host ({GPU_WHY or 'no GPU'})")
class TestProviderNumericsGPU(unittest.TestCase):
    def test_values_and_checkpointed_gradients_vs_f6_both_dtypes_both_words(self):
        env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}          # the CPU classes pin JAX_PLATFORMS=cpu in-process; the child picks the GPU itself
        r = subprocess.run([sys.executable, "-c", PROVIDER_NUMERICS_CHILD], capture_output=True, text=True, timeout=2400, env=env)
        sys.stdout.write(r.stdout[-12000:])
        self.assertEqual(r.returncode, 0, r.stderr[-6000:])
        self.assertIn('"cases": 16', r.stdout)


if __name__ == "__main__":
    unittest.main()
