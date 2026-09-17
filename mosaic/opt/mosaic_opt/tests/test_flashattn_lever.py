"""Lever K1 (triattn): the kit module mosaic/fast/flashattn.py maps joltz's TriangleAttention core onto the shared core's JAX-family provider
(opt_core.kernels.pallas) by the row's TIER WORD — the fast row pins `fast`, the big row `big` (modes.KIT_MODES specs); the module names no
kernel, tile or launch setting; provider row / arm words pin one row for ablation; the spec grammar refuses by name; off, the rebound call IS the
stock body. The served mapping is checked on CPU against the stock body through the provider's stock-statement row (TestServedMappingCPU); the
tier words' rows, their numerics class and the refusal-by-name paths are checked on a GPU (TestProviderBindingGPU)."""
import importlib.util
import os
import subprocess
import sys
import types
import unittest

from . import _stubs
from mosaic_opt import registry

KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "flashattn.py")
TOOLS_FILE = os.path.join(_stubs.KIT, "tools", "flashattn.py")
DRIVER = os.path.join(_stubs.KIT, "tools", "public_design_run.py")


def _load(path, name="flashattn_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gpu_visible():
    """A GPU is on this host (nvidia-smi lists one) — asked without importing jax in this interpreter (the suite may pin its own platform)."""
    import shutil
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def _have(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestFlashAttnLever(unittest.TestCase):
    def test_registry_entry_and_row(self):
        """K1 is a per-step lever of the fast tier: wired, class fast, origin core (every row lives in opt_core), switched by the ONE flag
        `--levers fast` (row T_fast) / `levers.install("fast")`; the fast row pins the tier word `fast`, the big row `big` (a tier-following
        lever: modes.TIER_FOLLOWING) — no row word, no kernel setting in the mode table; no tools/ twin and no driver flag of its own."""
        from mosaic_opt import modes
        lv = registry.LEVERS["K1"]
        self.assertEqual(lv.kit_file, f"{registry.KIT_FAST_DIR}/flashattn.py")
        self.assertEqual((lv.klass, lv.route, lv.module, lv.flag, lv.origin, lv.wired, lv.tier), ("fast", "install", "mosaic.fast.flashattn", "--levers fast", "core", True, "tier2"))
        self.assertEqual(lv.probe, registry.probe_key("K1")); self.assertIn("K1", registry.WIRED); self.assertIn("K1", registry.IN_PROCESS)
        self.assertIn("K1", modes.TIER_FOLLOWING)
        for w in ("fast", "big"):
            self.assertIn("K1", modes.KIT_MODES[w]["levers"]); self.assertEqual(modes.KIT_MODES[w]["specs"].get("K1"), w)   # the pinned spec IS the row's tier word
            self.assertEqual(modes.specs_of(w)["K1"], modes.KIT_MODES[w]["klass"])
        self.assertNotIn("K1", modes.KIT_MODES["exact"]["levers"])                                # the exact row carries no kernel lever
        self.assertFalse(os.path.exists(TOOLS_FILE))
        drv = open(DRIVER, encoding="utf-8").read()
        self.assertNotIn("--kernels", drv); self.assertNotIn("flashattn", drv)                     # the ONE flag --levers reaches it through mosaic_opt.levers

    def test_module_surface_names_no_kernel_of_its_own(self):
        """The module is a binding: the provider package and face, the op / family kind / direction / layout it maps onto, the tier words, the
        record word `fast`; no kit tile table, launch setting, cuDNN op or bridge import remains (those rows live in the provider)."""
        fa = _load(KIT_FILE, "flashattn_surface")
        self.assertEqual((fa.LEVER, fa.NAME, fa.MODULE, fa.KLASS, fa.RECORD), ("K1", "triattn", "mosaic.fast.flashattn", "fast", "fast"))
        self.assertEqual((fa.PROVIDER, fa.FACES, fa.OP, fa.KIND, fa.DIRECTION, fa.LAYOUT), ("opt_core.kernels.pallas", "opt_core.kernels.pallas.serve", "attn", "tri", "fwdbwd", "BHSD"))
        self.assertEqual(fa.TIER_WORDS, ("fast", "big", "exact")); self.assertEqual(fa.ENV_REQUIRED, {}); self.assertEqual(fa.MIN_TOKENS, 0)
        self.assertEqual(tuple(fa.EXPECTED_FALLBACKS), ())
        for gone in ("TILES", "FWD_TILES", "STAGES", "DBIAS", "DBIAS_KERNEL_MIN_BYTES", "DQ", "CHUNK", "PRECISION", "CUDNN_DTYPES", "FOLD_FLAGS", "XLA_WORD", "SERVER",
                     "_cudnn_op", "_xla_op", "_auto_dbias_op", "server"):
            self.assertFalse(hasattr(fa, gone), gone)
        src = open(KIT_FILE, encoding="utf-8").read()
        for gone in ("pallas_attn_serve", "kernels.triattn_xla", "import triattn_xla", "dot_product_attention", "make_flash_attention"):
            self.assertNotIn(gone, src, gone)

    @unittest.skipUnless(_have("opt_core"), "parse() names the provider's rows: needs opt_core importable")
    def test_spec_grammar_and_refusals(self):
        fa = _load(KIT_FILE, "flashattn_grammar")
        off = {"on": False, "spec": "stock", "word": None, "kind": None}
        self.assertEqual(fa.parse("stock"), off); self.assertEqual(fa.parse("off"), off)
        self.assertEqual(fa.parse(None), fa.parse("fast")); self.assertEqual(fa.parse(""), fa.parse("fast"))   # None = RECORD = the fast tier word, ON (never a silent off)
        for w in fa.TIER_WORDS:
            self.assertEqual(fa.parse(w), {"on": True, "spec": w, "word": w, "kind": "tier"})
        rows = fa.rows()
        for r in ("triattn_xla", "pallas_attn", "cd_triatt", "cudnn", "xla"):
            self.assertIn(r, rows)
            self.assertEqual(fa.parse(r), {"on": True, "spec": r, "word": r, "kind": "row"})
        self.assertNotIn("fpf_trimul", rows)
        self.assertEqual(fa.parse("triattn_xla@vjp")["kind"], "arm"); self.assertEqual(fa.parse("pallas_attn:tf32@bwd_tf32")["kind"], "arm")
        for bad in ("triattn", "triattn:xla", "triattn:cudnn+fold", "triattn@f64x32+s3", "flash", "fpf_trimul", "proj+pallas_attn", "pallas_attn@@", "FAST"):
            with self.assertRaises(fa.Refusal, msg=bad) as cm:
                fa.parse(bad)
            self.assertEqual(cm.exception.reason, "unknown_spec", bad)
        from opt_core.kernels import pallas as P
        self.assertEqual(tuple(w for w in fa.TIER_WORDS if w not in P.TIER_WORDS), ())              # the kit's tier words are the provider's

    def test_install_rebinds_and_off_is_the_stock_body(self):
        """With a stand-in `joltz` module (a class with __call__), install() rebinds, the rebound call delegates to the ORIGINAL function
        object while the lever is off, configure('stock') keeps it off, uninstall() restores the class attribute."""
        fa = _load(KIT_FILE, "flashattn_standins")
        calls = []

        class TriangleAttention:                       # stand-in: records that the stock body ran
            starting = True
            def __call__(self, x, mask):
                calls.append((x, mask)); return ("stock", x)

        fake = types.ModuleType("joltz"); fake.TriangleAttention = TriangleAttention
        stock_fn = TriangleAttention.__call__
        saved = {k: sys.modules.get(k) for k in ("joltz", "jax", "equinox")}
        sys.modules["joltz"] = fake
        jax_stub = types.ModuleType("jax"); jax_stub.clear_caches = lambda: None
        eqx_stub = types.ModuleType("equinox"); eqx_stub.clear_caches = lambda: None
        sys.modules.setdefault("jax", jax_stub); sys.modules.setdefault("equinox", eqx_stub)
        try:
            fa.install(); fa.install()                                   # idempotent
            self.assertIsNot(TriangleAttention.__call__, stock_fn)
            self.assertIs(fa._ORIG["triatt"], stock_fn)
            self.assertEqual(TriangleAttention()("x", "m"), ("stock", "x"))     # off: the stock body
            rec = fa.configure("stock")
            self.assertEqual((rec["on"], rec["spec"], rec["installed"]), (0, "stock", 1))
            for k in ("name", "state", "tag", "reason"):
                self.assertNotIn(k, rec)
            self.assertEqual(TriangleAttention()("y", "m"), ("stock", "y"))
            self.assertEqual(len(calls), 2)
            self.assertIn("state=off", fa.emit_line("t")) if importlib.util.find_spec("opt_core") else None
            with self.assertRaises(fa.Refusal) as cm:                    # installed but never configured ON: the gate refuses by name
                fa.gate()
            self.assertEqual(cm.exception.reason, "not_configured")
            fa.uninstall()
            self.assertIs(TriangleAttention.__call__, stock_fn)
            self.assertEqual(fa.describe()["on"], 0)
            fa.gate()                                                    # not installed: nothing ran under the lever's name
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_gate_prices_the_ledger_verdict(self):
        """The configured-on gate is fail-closed on the census: the core Ledger's `gate()` RETURNS a verdict (ok, reason) and `flashattn.gate()`
        raises `lever_gate` on a refused one (an undeclared fallback reason, a kernel error) and `nothing_served` when no triangle attention went
        through the op. A stand-in ledger states the contract; the real `opt_core.counters.Ledger` (when importable) shows the declared and the
        undeclared fallback end to end."""
        fa = _load(KIT_FILE, "flashattn_gate_verdict")

        class Verdict:                                                   # the two fields of opt_core.gates.Gate the kit reads
            def __init__(self, ok, reason=None):
                self.ok, self.reason = ok, reason

        class StandInLedger:
            def __init__(self, ok, served, reason=None):
                self.verdict, self.served, self.names = Verdict(ok, reason), served, []
            def gate(self, name=None):
                self.names.append(name); return self.verdict

        try:
            L = StandInLedger(False, 7, "unexpected fallback: fallback_by=unserved_no_backward:1")
            fa.STATE["ledger"] = L
            with self.assertRaises(fa.Refusal) as cm:
                fa.gate()
            self.assertEqual((cm.exception.reason, L.names), ("lever_gate", [f"{fa.LEVER}.{fa.NAME}"]))
            self.assertIn("unserved_no_backward", cm.exception.detail)
            fa.STATE["ledger"] = StandInLedger(True, 0)                  # the census closes but nothing was served: not a run under the lever's name
            with self.assertRaises(fa.Refusal) as cm:
                fa.gate()
            self.assertEqual(cm.exception.reason, "nothing_served")
            fa.STATE["ledger"] = StandInLedger(True, 3)
            fa.gate()                                                    # served, census closed: passes
            if importlib.util.find_spec("opt_core") is not None:
                from opt_core.counters import Ledger
                real = Ledger(f"{fa.LEVER}.{fa.NAME}", impl="stand-in", origin="core", min_tokens=0, expected=tuple(fa.EXPECTED_FALLBACKS) + ("below_size_rule",))
                for _ in range(3):
                    real.serve("B4xH4xS169xD32")
                fa.STATE["ledger"] = real
                fa.gate()
                real.fallback("below_size_rule")                         # a DECLARED reason (the size rule when a gate is set): counted, the gate holds
                fa.gate()
                real.fallback("unserved_no_backward")                    # an undeclared one (a row word's envelope): refused by name
                with self.assertRaises(fa.Refusal) as cm:
                    fa.gate()
                self.assertEqual(cm.exception.reason, "lever_gate"); self.assertIn("unserved_no_backward:1", cm.exception.detail)
                empty = Ledger(f"{fa.LEVER}.{fa.NAME}", impl="stand-in", origin="core", min_tokens=0, expected=())
                fa.STATE["ledger"] = empty
                with self.assertRaises(fa.Refusal) as cm:
                    fa.gate()
                self.assertEqual(cm.exception.reason, "nothing_served")
        finally:
            fa.STATE["ledger"] = None

    def test_unserved_word_of_a_provider_refusal(self):
        """A provider Refusal raised while tracing one call becomes the counted fallback word `unserved_<kind>`; any other exception is not a
        fallback (None: re-raised by the caller)."""
        fa = _load(KIT_FILE, "flashattn_unserved")
        Ref = type("Refusal", (LookupError,), {})
        r = Ref("no measured cell"); r.kind = "cell_unmeasured"
        self.assertEqual(fa._unserved(r), "unserved_cell_unmeasured")
        self.assertIsNone(fa._unserved(ValueError("boom")))
        k = fa.Refusal("unknown_spec", "x")                              # the kit's own configure refusal is not a per-call fallback either
        self.assertIsNone(fa._unserved(k))


@unittest.skipUnless(_stubs.tree_present() and _have("jax", "equinox", "einops", "joltz", "opt_core"), "needs jax + equinox + joltz + opt_core importable (the kit venv)")
class TestServedMappingCPU(unittest.TestCase):
    """Lever ON through the REAL binding with the provider's stock-statement row (`xla`, selected for cc 9.0 — the provider's OPT_CORE_PALLAS_CC
    form of tracing off-target) on CPU: the rebound TriangleAttention equals the stock body to float32 rounding for the starting and the ending node
    at Boltz-2's 4 heads x 32 and a non-Boltz 2 x 16, N=70, with padded tokens — rows→batch, linear(x)→pair bias, mask rows→key masks, heads-major
    layout, scale, gating and the output Linear are mapped right on every live row; the padded tokens' all-masked rows are finite and isolated —
    the gradient of a loss over live rows equals stock's everywhere; the census names the arm that served (`f32.xla`). Child interpreter (the real
    jax/joltz stack must not leak into this session's stand-ins)."""

    SCRIPT = r"""
import importlib.util, sys, numpy as np
import jax, jax.numpy as jnp
import joltz
from joltz import TriangleAttention, Attention
from joltz.backend import Linear, LayerNorm
spec = importlib.util.spec_from_file_location("flashattn_cpu_mapping", sys.argv[1]); fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)
N, C = 70, 32
ks = jax.random.split(jax.random.PRNGKey(0), 10)
lin = lambda k, o, i, bias=False: Linear(weight=jax.random.normal(k, (o, i), jnp.float32) / np.sqrt(i), bias=(0.1 * jax.random.normal(jax.random.fold_in(k, 1), (o,), jnp.float32)) if bias else None)
x = jax.random.normal(ks[0], (1, N, N, C), jnp.float32)
tok = jnp.arange(N) < N - 3
mask = (tok[:, None] & tok[None, :]).astype(jnp.float32)[None]
live = np.asarray(tok); sel = live[:, None] & live[None, :]
stock_call = joltz.TriangleAttention.__call__
for starting, H, D in ((True, 4, 32), (False, 4, 32), (True, 2, 16)):
    mha = Attention(c_q=C, c_k=C, c_v=C, c_hidden=D, no_heads=H, gating=True, linear_q=lin(ks[1], H * D, C), linear_k=lin(ks[2], H * D, C),
                    linear_v=lin(ks[3], H * D, C), linear_o=lin(ks[4], C, H * D, True), linear_g=lin(ks[5], H * D, C, True), sigmoid=jax.nn.sigmoid)
    mod = TriangleAttention(c_in=C, c_hidden=D, no_heads=H, starting=starting, inf=1e9, layer_norm=LayerNorm(weight=jnp.ones(C), bias=jnp.zeros(C), eps=1e-5), linear=lin(ks[6], H, C), mha=mha)
    live_loss = lambda f: (lambda xx: (f(xx) * jnp.asarray(sel, xx.dtype)[..., None]).sum())         # a loss that reads live rows only (as every consumer does)
    with jax.default_matmul_precision("highest"):
        stock = np.asarray(stock_call(mod, x, mask)); g_stock = np.asarray(jax.grad(live_loss(lambda xx: stock_call(mod, xx, mask)))(x))
        fa.install()
        try:
            rec = fa.configure("xla", cc="9.0"); assert (rec["on"], rec["spec"], rec["word"], rec["kind"], rec["cc"], rec["origin"]) == (1, "xla", "xla", "row", "9.0", "core"), rec
            assert rec["impl"].startswith(fa.FACES + "@"), rec
            served = np.asarray(mod(x, mask)); g_served = np.asarray(jax.grad(live_loss(lambda xx: mod(xx, mask)))(x)); d = fa.describe()
            assert d["served"] >= 1 and d["fallback"] == 0 and d["calls"] == d["served"], d
            assert "f32.xla:" in d["facts"], d                                                # the census names the arm that served each traced call
            fa.gate()
        finally:
            fa.configure("stock"); fa.uninstall()
    err = float(np.abs(served[0] - stock[0])[sel].max()); scale = float(np.abs(stock[0])[sel].max())     # live rows: the stock body's numbers
    assert err < 1e-4 * max(1.0, scale), (starting, H, D, err)
    assert bool(np.isfinite(served[0]).all()), (starting, H, D, "padded rows must be finite")           # padded (all-masked) rows: finite, isolated
    gerr = float(np.abs(g_served - g_stock).max()); gscale = float(np.abs(g_stock).max())
    assert bool(np.isfinite(g_served).all()) and gerr < 1e-4 * max(1.0, gscale), (starting, H, D, gerr)   # live loss ⇒ the same gradient everywhere, padded rows included
    print("MAPPING_OK", "starting" if starting else "ending", f"H={H}", f"D={D}", f"max_abs_err={err:.3e}", f"grad_max_abs_err={gerr:.3e}", f"served={d['served']}", f"facts={d['facts']}")
if jax.default_backend() != "gpu":                                    # a tier word off a GPU: refused BY NAME before anything is traced (the provider's word)
    try:
        fa.configure("fast"); raise SystemExit("configure('fast') did not refuse on a non-GPU backend")
    except fa.Refusal as e:
        assert e.reason == "provider_refused" and "backend_not_gpu" in e.detail, (e.reason, e.detail); print("TIER_REFUSAL_OK", e.reason, e.detail.split(":")[0])
else:
    print("TIER_REFUSAL_OK", "gpu-host-skips-the-cpu-refusal")
"""

    def test_served_body_equals_stock_body_through_the_stock_statement_row(self):
        env = dict(os.environ); env.setdefault("JAX_PLATFORMS", "cpu"); env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env.pop("OPT_CORE_PALLAS_CC", None)
        r = subprocess.run([sys.executable, "-c", self.SCRIPT, KIT_FILE], capture_output=True, text=True, timeout=900, env=env)
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-3000:])
        oks = [l for l in r.stdout.splitlines() if l.startswith("MAPPING_OK")]
        self.assertEqual(len(oks), 3, r.stdout[-2000:])
        self.assertIn("TIER_REFUSAL_OK", r.stdout)
        print("\n".join(oks + [l for l in r.stdout.splitlines() if l.startswith("TIER_REFUSAL_OK")]))


@unittest.skipUnless(_stubs.tree_present() and _have("jax", "equinox", "einops", "joltz", "opt_core"), "needs the kit venv")
class TestProviderBindingGPU(unittest.TestCase):
    """On a GPU (skipped elsewhere): the tier words `fast` and `big` configure, select a DIFFERENTIABLE row for bfloat16 operands (P7's pair track)
    and for float32 operands at the pairformer geometry, and the served op agrees with the float32 reference within each operand class forward AND
    backward (dq and the pair-bias gradient) at a token count off every tile multiple with masked keys; `exact` serves the stock statement; a row word
    and an arm word pin one row; the provider's refusal of a row for this call class (`cudnn`: differentiated, key-masked, shared bias) is a configure
    refusal BY NAME; the module end to end under `fast` agrees with joltz's body within the TF32 class and its census names the arm that served.
    Child interpreter."""

    SCRIPT = r"""
import importlib.util, sys, numpy as np
import jax, jax.numpy as jnp
assert jax.default_backend() == "gpu", jax.default_backend()
spec = importlib.util.spec_from_file_location("flashattn_gpu", sys.argv[1]); fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)
from opt_core.kernels.pallas import serve as PS
HI = jax.lax.Precision.HIGHEST
B, H, S, D = 6, 4, 72, 32
ks = jax.random.split(jax.random.PRNGKey(3), 6)
q32, k32, v32 = (jax.random.normal(kk, (B, H, S, D), jnp.float32) for kk in ks[:3])
b32 = 0.5 * jax.random.normal(ks[3], (H, S, S), jnp.float32)
kmask = jnp.ones((B, S), bool).at[:, S - 5:].set(False).at[1, :3].set(False)
w = jax.random.normal(ks[4], (B, H, S, D), jnp.float32)
scale = 1.0 / np.sqrt(D)
def ref_fg(q, k, v, b):
    f = lambda qq, bb: PS.reference_attention(qq, k, v, bb, kmask, scale, precision=HI, layout="BHSD")
    return f(q, b), jax.grad(lambda qq, bb: (f(qq, bb) * w).sum(), argnums=(0, 1))(q, b)
ro, (rgq, rgb) = ref_fg(q32, k32, v32, b32)
def check(op, dt, tol, tag):
    q, k, v, b = (t.astype(dt) for t in (q32, k32, v32, b32))
    f = lambda qq, bb: op(qq, k, v, bb, kmask, scale).astype(jnp.float32)
    o = f(q, b); gq, gb = jax.grad(lambda qq, bb: (f(qq, bb) * w).sum(), argnums=(0, 1))(q, b)
    assert o.shape == (B, H, S, D) and o.dtype == jnp.float32, (o.shape, o.dtype)
    errs = [float(jnp.abs(a.astype(jnp.float32) - r).max() / jnp.abs(r).max()) for a, r in ((o, ro), (gq, rgq), (gb, rgb))]
    assert all(np.isfinite(errs)) and max(errs) < tol, (tag, str(dt), errs, tol)
    return errs
for word in ("fast", "big"):
    rec = fa.configure(word)
    assert (rec["on"], rec["word"], rec["kind"], rec["origin"], rec["direction"], rec["layout"]) == (1, word, "tier", "core", "fwdbwd", "BHSD"), rec
    assert not str(rec["sel_bf16"]).startswith("refused") and not str(rec["sel_f32"]).startswith("refused"), rec      # a measured cell on this card for both operand classes
    assert rec["sel_bf16"] != "xla", rec                                                                                 # the tier serves a kernel row for the bf16 pair track, not the stock statement
    from opt_core.counters import Ledger
    fa.STATE["ledger"].clear()
    e16 = check(fa.current_op(), jnp.bfloat16, 6e-2, word); e32 = check(fa.current_op(), jnp.float32, 1e-2, word)
    facts = fa.describe()["facts"]
    assert "bf16." in facts and "f32." in facts, facts
    print("TIER_OK", word, "cc=%s" % rec["cc"], "line=%s" % rec["jax_line"], "sel_bf16=%s" % rec["sel_bf16"], "sel_f32=%s" % rec["sel_f32"], "cell_bf16=%s" % rec["cell_bf16"],
          "bf16_errs=%s" % ["%.2e" % x for x in e16], "f32_errs=%s" % ["%.2e" % x for x in e32], "facts=%s" % facts)
rec = fa.configure("exact")
assert not str(rec["sel_f32"]).startswith("refused"), rec
e32 = check(fa.current_op(), jnp.float32, 1e-2, "exact")
print("EXACT_OK", "sel_bf16=%s" % rec["sel_bf16"], "sel_f32=%s" % rec["sel_f32"], "facts=%s" % fa.describe()["facts"], "f32_errs=%s" % ["%.2e" % x for x in e32])
for word, kind in (("triattn_xla@vjp", "arm"), ("pallas_attn", "row")):
    rec = fa.configure(word); assert (rec["word"], rec["kind"]) == (word, kind), rec
    e16 = check(fa.current_op(), jnp.bfloat16, 6e-2, word)
    print("ROW_OK", word, "facts=%s" % fa.describe()["facts"], "bf16_errs=%s" % ["%.2e" % x for x in e16])
try:
    fa.configure("cudnn"); raise SystemExit("configure('cudnn') served a differentiated key-masked call")
except fa.Refusal as e:
    assert e.reason == "provider_refused" and "cudnn_shared_bias_grad" in e.detail, (e.reason, e.detail); print("REFUSAL_OK cudnn", e.detail.split(":")[0])
for old in ("triattn:xla", "triattn:cudnn+fold", "triattn"):
    try:
        fa.configure(old); raise SystemExit("old word %r accepted" % old)
    except fa.Refusal as e:
        assert e.reason == "unknown_spec", (old, e.reason)
print("REFUSAL_OK old_words")
# the module end to end under the fast tier word: joltz's TriangleAttention (float32 parameters and activations: the TF32 class) vs its own body
import joltz
from joltz import TriangleAttention, Attention
from joltz.backend import Linear, LayerNorm
N, C = 70, 32
lin = lambda k, o, i, bias=False: Linear(weight=jax.random.normal(k, (o, i), jnp.float32) / np.sqrt(i), bias=(0.1 * jax.random.normal(jax.random.fold_in(k, 1), (o,), jnp.float32)) if bias else None)
kk = jax.random.split(jax.random.PRNGKey(0), 10)
x = jax.random.normal(kk[0], (1, N, N, C), jnp.float32)
tok = jnp.arange(N) < N - 3
mask = (tok[:, None] & tok[None, :]).astype(jnp.float32)[None]
live = np.asarray(tok); sel = live[:, None] & live[None, :]
stock_call = joltz.TriangleAttention.__call__
fa.install()
try:
    rec = fa.configure("fast")
    for starting in (True, False):
        mha = Attention(c_q=C, c_k=C, c_v=C, c_hidden=32, no_heads=4, gating=True, linear_q=lin(kk[1], 128, C), linear_k=lin(kk[2], 128, C),
                        linear_v=lin(kk[3], 128, C), linear_o=lin(kk[4], C, 128, True), linear_g=lin(kk[5], 128, C, True), sigmoid=jax.nn.sigmoid)
        mod = TriangleAttention(c_in=C, c_hidden=32, no_heads=4, starting=starting, inf=1e9, layer_norm=LayerNorm(weight=jnp.ones(C), bias=jnp.zeros(C), eps=1e-5), linear=lin(kk[6], 4, C), mha=mha)
        live_loss = lambda f: (lambda xx: (f(xx) * jnp.asarray(sel, xx.dtype)[..., None]).sum())
        with jax.default_matmul_precision("highest"):
            stock = np.asarray(stock_call(mod, x, mask)); g_stock = np.asarray(jax.grad(live_loss(lambda xx: stock_call(mod, xx, mask)))(x))
        served = np.asarray(jax.jit(lambda xx: mod(xx, mask))(x)); g_served = np.asarray(jax.jit(jax.grad(live_loss(lambda xx: mod(xx, mask))))(x))
        err = float(np.abs(served[0] - stock[0])[sel].max() / max(1.0, np.abs(stock[0])[sel].max()))
        gerr = float(np.abs(g_served - g_stock).max() / max(1.0, np.abs(g_stock).max()))
        assert np.isfinite(served).all() and err < 1e-2 and gerr < 1e-2, (starting, err, gerr)
        print("MODULE_OK", "starting" if starting else "ending", "rel_err=%.2e grad_rel_err=%.2e" % (err, gerr))
    d = fa.describe(); assert d["served"] >= 4 and d["fallback"] == 0, d
    fa.gate()
    line = fa.emit_line("gpu-test")
    assert " state=on " in line and " origin=core " in line and "sel_bf16=" in line and "word=fast" in line, line
    print("CENSUS_OK", "served=%s facts=%s shapes=%s" % (d["served"], d["facts"], d["shapes"]))
finally:
    fa.configure("stock"); fa.uninstall()
"""

    def test_tier_words_rows_numerics_and_refusals_on_gpu(self):
        if not _gpu_visible():
            self.skipTest("needs a GPU (nvidia-smi lists none)")
        env = dict(os.environ); env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p); env.pop("OPT_CORE_PALLAS_CC", None); env.pop("JAX_PLATFORMS", None)
        r = subprocess.run([sys.executable, "-c", self.SCRIPT, KIT_FILE], capture_output=True, text=True, timeout=1800, env=env)
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-4000:])
        tags = [l for l in r.stdout.splitlines() if l.split(" ")[0] in ("TIER_OK", "EXACT_OK", "ROW_OK", "REFUSAL_OK", "MODULE_OK", "CENSUS_OK")]
        for t in ("TIER_OK fast", "TIER_OK big", "EXACT_OK", "ROW_OK triattn_xla@vjp", "ROW_OK pallas_attn", "REFUSAL_OK cudnn", "REFUSAL_OK old_words", "MODULE_OK starting", "MODULE_OK ending", "CENSUS_OK"):
            self.assertTrue(any(l.startswith(t) for l in tags), (t, r.stdout[-3000:]))
        print("\n".join(tags))


def _token_ok(v):
    """The kit driver's lever-evidence rule (mosaic_opt.levers._token_ok on the kit line; replicated here when that module is absent):
    every describe() value is a scalar AND one blank-free token."""
    try:
        from mosaic_opt.levers import _token_ok as rule          # the package's rule when importable
        return rule(v)
    except ImportError:
        import re
        return isinstance(v, (str, int, float, bool, type(None))) and re.fullmatch(r"\S+", str(v)) is not None


@unittest.skipUnless(_stubs.tree_present() and _have("opt_core"), "needs opt_core importable")
class TestDescribeTokens(unittest.TestCase):
    """describe() before install, after configure, and after served + fallback traffic (the ledger populated the way served calls populate it,
    the per-arm facts the way the census records them): every value passes the driver's token rule; the LEVER line renders with the provider's
    identity and the selection facts."""

    def test_every_describe_value_is_a_single_token_scalar(self):
        fa = _load(KIT_FILE, "flashattn_tokens")
        from opt_core.counters import Ledger

        def check(d, when):
            bad = {k: v for k, v in d.items() if not _token_ok(v)}
            self.assertEqual(bad, {}, f"{when}: {bad}")
        check(fa.describe(), "before install")
        saved = dict(fa.STATE)
        try:
            fa.STATE.update(spec="fast", impl="provider", word="fast", kind="tier", cc="9.0", jax_line="0.10", op=lambda *a: None, override=True,
                            ledger=Ledger(f"{fa.LEVER}.{fa.NAME}", impl=f"{fa.FACES}@0.0.0", origin="core", min_tokens=64, expected=("below_size_rule",) + fa.EXPECTED_FALLBACKS),
                            sel={"bf16": {"arm": "triattn_xla@vjp", "cell": "0.10|9.0|bf16|attn|tri_h4_d32|N<=500|fwdbwd", "order": "triattn_xla@vjp+cd_triatt+xla"},
                                 "f32": {"arm": "pallas_attn:tf32@bwd_tf32", "cell": "0.10|9.0|f32|attn|tri_h4_d32|N<=500|fwdbwd", "order": "x"}})
            d = fa.describe(); check(d, "configured, no traffic")
            self.assertEqual((d["sel_bf16"], d["cell_f32"], d["word"], d["kind"], d["describe_at"]), ("triattn_xla@vjp", "0.10|9.0|f32|attn|tri_h4_d32|N<=500|fwdbwd", "fast", "tier", fa.DESCRIBE_AT))
            L = fa.STATE["ledger"]                                        # traffic, as _triangle_attention_call and _census record it
            L.serve("B169xH4xS169xD32"); L.serve("B169xH4xS169xD32"); L.serve("B12xH4xS12xD32x2"); L.fallback("below_size_rule")
            import numpy as np
            fa._census({"served:attention:triattn_xla@vjp": 1, "refused:cudnn:cudnn_shared_bias_grad": 0},
                       {"served:attention:triattn_xla@vjp": 3, "refused:cudnn:cudnn_shared_bias_grad": 1, "served:attention:pallas_attn:tf32@bwd_tf32": 1, "served:transition:xla": 5}, np.float32)
            d = fa.describe(); check(d, "after traffic")
            self.assertEqual((d["served"], d["fallback"], d["calls"]), (3, 1, 4))
            self.assertEqual(d["shapes"], "B12xH4xS12xD32x2:1+B169xH4xS169xD32:2")
            self.assertEqual(d["fallback_by"], "below_size_rule:1")
            self.assertEqual(d["facts"], "aside.cudnn.cudnn_shared_bias_grad:1+f32.pallas_attn~tf32@bwd_tf32:1+f32.triattn_xla@vjp:2")   # another face's counter is not this lever's fact
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                line = fa.emit_line("tokens")
            self.assertIn(f" impl={fa.FACES}@0.0.0 origin=core ", line); self.assertIn(" served=3 ", line); self.assertIn(" sel_bf16=triattn_xla@vjp ", line)
            self.assertIn(" word=fast ", line); self.assertIn(" override=1", line)
        finally:
            fa.STATE.clear(); fa.STATE.update(saved)
        self.assertEqual(fa._token({}), "none"); self.assertEqual(fa._token("a b\tc"), "a_b_c"); self.assertEqual(fa._token(True), 1)
        self.assertEqual(fa._token([1, "x y"]), "1+x_y"); self.assertIsNone(fa._token(None))


if __name__ == "__main__":
    unittest.main()
