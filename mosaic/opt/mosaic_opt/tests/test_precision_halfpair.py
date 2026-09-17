"""Lever P7 (halfpair): the trunk pairformer's pair track in bfloat16 — the kit module mosaic/fast/halfpair.py (one home, no tools/ twin);
the registry entry names the module, its class (fast: bf16 operand rounding, never bitwise) and the ONE flag that reaches it (`--levers fast|big`
→ mosaic_opt.levers → install() + configure(None)); the row installs it LAST (after F6/F8/K1, and after P5 on big: it wraps the Pairformer2 call
it finds); the spec grammar (region tokens, the `pf` alias, canonical spelling, named refusals); describe() is FLAT single tokens; off, the rebound
calls ARE joltz's own (function identity) and configure('stock') routes every block to them. The arithmetic is checked on CPU when jax + equinox +
joltz import (TestHalfpairCPU): a block under `pf` equals joltz's float32 block to bf16 rounding and its parameters / residual stream stay float32;
`carry` hands blocks a bf16 activation and refuses `carry_bypassed` by name when the Pairformer2 wrapper was displaced."""
import importlib.util
import os
import unittest

from . import _stubs
from mosaic_opt import registry

LID = "P7"
KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "halfpair.py")
TOOLS_FILE = os.path.join(_stubs.KIT, "tools", "halfpair.py")
HAVE_STACK = all(importlib.util.find_spec(m) is not None for m in ("jax", "equinox", "joltz"))


def _load(path=KIT_FILE, name="halfpair_under_test"):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in _stubs.STANDINS}
    sys.modules.update(_stubs._standins())
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestHalfpairLever(unittest.TestCase):
    def test_registry_entry_and_rows(self):
        """P7 is a per-step lever of the fast tier AND the memory tier (fast ⊂ big): wired, class fast, origin kit, route install, switched by
        the ONE flag `--levers fast`; installed last of each row — after F6/F8/K1 (it reads their state at configure) and after P5 on big
        (it wraps whatever Pairformer2.__call__ it finds)."""
        from mosaic_opt import modes
        lv = registry.LEVERS[LID]
        self.assertEqual(lv.kit_file, f"{registry.KIT_FAST_DIR}/halfpair.py")
        self.assertEqual((lv.klass, lv.route, lv.module, lv.flag, lv.origin, lv.wired, lv.tier),
                         ("fast", "install", "mosaic.fast.halfpair", "--levers fast", "kit", True, "tier2"))
        self.assertIn(LID, registry.WIRED); self.assertIn(LID, registry.IN_PROCESS); self.assertIn(LID, registry.INSTALL)
        for mode in ("fast", "big"):
            row = modes.KIT_MODES[mode]["levers"]
            self.assertIn(LID, row)
            for before in ("F6", "F8", "K1"):
                self.assertLess(row.index(before), row.index(LID), (mode, before))
        big = modes.KIT_MODES["big"]["levers"]
        self.assertLess(big.index("P5"), big.index(LID))
        self.assertNotIn(LID, modes.KIT_MODES["exact"]["levers"])                       # a fast-class lever is never in the exact tier
        M = _load(name="halfpair_grammar_for_rows")
        for mode in ("fast", "big"):                                                      # each tier PINS its P7 word (a valid spec naming a pair region)
            word = modes.specs_of(mode).get(LID); self.assertTrue(word, mode); self.assertTrue(M.parse(word)["on"], (mode, word))
        self.assertEqual(modes.specs_of("fast")[LID], "pf+msa")                             # fast: all three pair sub-layer families + the MSA module's pair blocks (0.3.16)
        self.assertLessEqual(set(M.parse(modes.specs_of("big")[LID])["regions"]), set(M.parse(modes.specs_of("fast")[LID])["regions"]))   # big's regions ⊆ fast's
        self.assertFalse(os.path.exists(TOOLS_FILE))                                     # a per-step lever module has ONE home

    def test_grammar(self):
        M = _load()
        self.assertEqual(M.LEVER, LID); self.assertEqual(M.ENV_REQUIRED, {}); self.assertEqual(M.KLASS, "fast"); self.assertEqual(M.DTYPE, "bf16")
        self.assertEqual(M.parse(None), M.parse(M.SETTING))                              # configure(None) = the module's default setting
        self.assertEqual(M.parse("pf")["regions"], ("tm", "ta", "tz")); self.assertEqual(M.parse("pf")["spec"], "pf")
        self.assertEqual(M.parse("tz+tm+ta")["spec"], "pf")                              # the alias is the canonical spelling of its three tokens
        self.assertEqual(M.parse("ta+tm+carry")["spec"], "tm+ta+carry")                  # canonical order, not the caller's
        self.assertEqual(M.parse("pf+msa+carry")["regions"], ("tm", "ta", "tz", "carry", "msa"))
        self.assertEqual(M.parse("PF + Carry")["spec"], "pf+carry")
        for off in ("stock", "off", " Stock "):
            self.assertEqual(M.parse(off), {"on": False, "spec": "stock", "regions": ()})
        for bad in ("bf16", "pf+fp8", "carry", "msa+carry", "pf,carry", "tm=1"):
            with self.assertRaises(M.Refusal) as cm:
                M.parse(bad)
            self.assertEqual(cm.exception.reason, "unknown_spec", bad)

    def test_configure_before_install(self):
        M = _load()
        self.assertFalse(M.installed())
        with self.assertRaises(M.Refusal) as cm:
            M.configure("pf")
        self.assertEqual(cm.exception.reason, "not_installed")
        rec = M.configure("stock")                                                       # off before install: fine, nothing to do
        self.assertEqual((rec["on"], rec["spec"], rec["dtype"]), (False, "stock", "f32"))
        M.gate()                                                                         # off: the gate has nothing to refuse

    def test_describe_is_flat_tokens(self):
        M = _load()
        M.STATE.update(installed=True, on=True, spec="pf+carry", regions=("tm", "ta", "tz", "carry"), ta_path="aside:k1_pallas_dtype not served", trimul="route_primal_kernel")
        M._ORIG["pf2"] = object()                                                        # installed() reads _ORIG
        try:
            rec = M.describe()
            for k, v in rec.items():
                self.assertTrue(v is None or isinstance(v, (bool, int, float, str)), k)
                if isinstance(v, str):
                    self.assertNotIn(" ", v, k); self.assertTrue(v, k)
            self.assertEqual((rec["tm"], rec["tz"], rec["carry"], rec["msa"], rec["dtype"], rec["accumulate"], rec["residual"], rec["seq_track"]),
                             ("on", "on", "bf16", "off", "bf16", "f32", "f32", "f32"))
            self.assertTrue(rec["ta"].startswith("aside:k1_pallas"))                     # a region that stepped aside says so by name
            line = M.emit_line("t")
            self.assertIn("HALFPAIR tag=t lever=P7 spec=pf+carry", line); self.assertIn("ta_path=aside:k1_pallas_dtype_not_served", line)
            with self.assertRaises(M.Refusal) as cm:                                     # on and nothing traced: fail-closed
                M.gate()
            self.assertEqual(cm.exception.reason, "nothing_served")
        finally:
            M._ORIG.clear()


@unittest.skipUnless(_stubs.tree_present() and HAVE_STACK, "needs jax + equinox + joltz importable (the kit venv)")
class TestHalfpairCPU(unittest.TestCase):
    """The block arithmetic on CPU against joltz's own float32 block (small random modules; bf16 tolerance)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax  # noqa: F401
        import joltz
        cls.joltz = joltz
        cls.M = _load(name="halfpair_cpu_under_test")

    def tearDown(self):
        if self.M.installed():
            self.M.uninstall()

    def _layer(self, key, c_z=16, c_s=24, heads=2):
        """A random PairformerLayer2 with joltz's field structure (built through equinox, no torch checkpoint)."""
        import jax
        import jax.numpy as jnp
        J = self.joltz
        ks = iter(jax.random.split(key, 64))
        lin = lambda i, o, bias=False: J.Linear(weight=jax.random.normal(next(ks), (o, i), jnp.float32) / (i ** 0.5),  # noqa: E731
                                                bias=(jax.random.normal(next(ks), (o,), jnp.float32) * 0.1) if bias else None)
        ln = lambda c: J.LayerNorm(weight=jnp.ones((c,), jnp.float32) + 0.1 * jax.random.normal(next(ks), (c,), jnp.float32), bias=0.1 * jax.random.normal(next(ks), (c,), jnp.float32), eps=1e-5)  # noqa: E731

        def trimul(cls_):
            return cls_(norm_in=ln(c_z), p_in=lin(c_z, 2 * c_z), g_in=lin(c_z, 2 * c_z), norm_out=ln(c_z), p_out=lin(c_z, c_z), g_out=lin(c_z, c_z))

        def triatt(starting):
            mha = J.Attention(c_q=c_z, c_k=c_z, c_v=c_z, c_hidden=8, no_heads=heads, gating=True,
                              linear_q=lin(c_z, 8 * heads), linear_k=lin(c_z, 8 * heads), linear_v=lin(c_z, 8 * heads), linear_o=lin(8 * heads, c_z, True),
                              linear_g=lin(c_z, 8 * heads, True), sigmoid=jax.nn.sigmoid)
            return J.TriangleAttention(c_in=c_z, c_hidden=8, no_heads=heads, starting=starting, inf=1e9, layer_norm=ln(c_z), linear=lin(c_z, heads), mha=mha)
        trans = lambda c: J.Transition(norm=ln(c), fc1=lin(c, 4 * c), fc2=lin(c, 4 * c), fc3=lin(4 * c, c), silu=jax.nn.silu)  # noqa: E731
        apb = J.AttentionPairBias2(c_s=c_s, num_heads=heads, head_dim=c_s // heads, inf=1e6, proj_q=lin(c_s, c_s, True), proj_k=lin(c_s, c_s), proj_v=lin(c_s, c_s),
                                   proj_g=lin(c_s, c_s), proj_z=J.Sequential(_modules={"0": ln(c_z), "1": lin(c_z, heads), "2": J._rearrange("b i j h -> b h i j")}), proj_o=lin(c_s, c_s))
        return J.PairformerLayer2(token_z=c_z, dropout=0.25, num_heads=heads, attention=apb,
                                  tri_mul_out=trimul(J.TriangleMultiplicationOutgoing), tri_mul_in=trimul(J.TriangleMultiplicationIncoming),
                                  tri_att_start=triatt(True), tri_att_end=triatt(False),
                                  pre_norm_s=ln(c_s), transition_s=trans(c_s), transition_z=trans(c_z), s_post_norm=ln(c_s))

    def _skip_unless_constructible(self):
        import jax
        try:
            return self._layer(jax.random.PRNGKey(0))
        except TypeError as e:                                                           # joltz's field set moved: the test names it instead of guessing
            self.skipTest(f"joltz module constructors changed: {e}")

    def test_install_identity_and_stock_word(self):
        J, M = self.joltz, self.M
        own = (J.PairformerLayer2.__call__, J.PairformerNoSeqLayer.__call__, J.Pairformer2.__call__)
        M.install()
        self.assertTrue(M.installed()); self.assertIsNot(J.PairformerLayer2.__call__, own[0])
        with self.assertRaises(RuntimeError):
            M.install()
        rec = M.configure("stock")
        self.assertEqual((rec["on"], rec["spec"]), (False, "stock"))
        M.uninstall()
        self.assertEqual((J.PairformerLayer2.__call__, J.PairformerNoSeqLayer.__call__, J.Pairformer2.__call__), own)   # restored byte-for-byte

    def test_block_pf_matches_float32_to_bf16_rounding(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        layer = self._skip_unless_constructible()
        M = self.M
        B, N, c_z, c_s = 1, 12, 16, 24
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        s = jax.random.normal(k1, (B, N, c_s), jnp.float32); z = jax.random.normal(k2, (B, N, N, c_z), jnp.float32)
        mask = jnp.ones((B, N), jnp.float32).at[:, -2:].set(0.0); pair_mask = mask[:, :, None] * mask[:, None, :]
        ref_s, ref_z, _ = layer(s, z, mask, pair_mask, deterministic=True, key=k3)      # joltz's own float32 block
        M.install()
        rec = M.configure("pf")
        self.assertEqual((rec["on"], rec["tm"], rec["ta"], rec["tz"], rec["carry"], rec["ta_path"], rec["trimul"]), (True, "on", "on", "on", "f32", "stock_body", "stock_body"))
        out_s, out_z, _ = layer(s, z, mask, pair_mask, deterministic=True, key=k3)
        self.assertEqual(out_z.dtype, jnp.float32); self.assertEqual(out_s.dtype, jnp.float32)   # the residual stream and the sequence track stay float32
        live = np.asarray(pair_mask[..., None] > 0) * np.ones((1, 1, 1, c_z), bool)
        dz = np.abs(np.asarray(out_z) - np.asarray(ref_z))[live]; scale = np.abs(np.asarray(ref_z))[live].mean()
        self.assertGreater(dz.max(), 0.0)                                                # bf16 sub-layers: not bitwise with float32 …
        self.assertLess(dz.mean() / scale, 2e-2, (dz.mean(), scale))                    # … and inside bf16 rounding of a handful of sub-layers
        self.assertGreaterEqual(M.describe()["blocks_traced"], 1)
        M.gate()                                                                          # served: the gate passes
        rec = M.configure("stock")                                                        # the stock word: joltz's own body again, bitwise
        s2, z2, _ = layer(s, z, mask, pair_mask, deterministic=True, key=k3)
        np.testing.assert_array_equal(np.asarray(z2), np.asarray(ref_z)); np.testing.assert_array_equal(np.asarray(s2), np.asarray(ref_s))

    def test_grad_flows_in_float32(self):
        import jax
        import jax.numpy as jnp
        layer = self._skip_unless_constructible()
        M = self.M
        B, N, c_z, c_s = 1, 10, 16, 24
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
        s = jax.random.normal(k1, (B, N, c_s)); z = jax.random.normal(k2, (B, N, N, c_z))
        mask = jnp.ones((B, N)); pair_mask = mask[:, :, None] * mask[:, None, :]
        f = lambda z_: sum(jnp.sum(o) for o in layer(s, z_, mask, pair_mask, deterministic=True, key=k3)[:2])  # noqa: E731
        g_ref = jax.grad(f)(z)
        M.install(); M.configure("tm+tz")
        g = jax.grad(f)(z)
        self.assertEqual(g.dtype, jnp.float32)
        cos = float(jnp.vdot(g, g_ref) / (jnp.linalg.norm(g) * jnp.linalg.norm(g_ref)))
        self.assertGreater(cos, 0.98, cos)

    def test_carry_bypassed_is_named(self):
        import jax
        import jax.numpy as jnp
        layer = self._skip_unless_constructible()
        M = self.M
        M.install(); M.configure("pf+carry")
        B, N = 1, 8
        s = jnp.zeros((B, N, 24)); z = jnp.zeros((B, N, N, 16)); mask = jnp.ones((B, N)); pm = mask[:, :, None] * mask[:, None, :]
        with self.assertRaises(M.Refusal) as cm:                                          # a float32 activation under `carry`: the wrapper was displaced
            layer(s, z, mask, pm, deterministic=True, key=jax.random.PRNGKey(0))
        self.assertEqual(cm.exception.reason, "carry_bypassed")
        out_s, out_z, _ = layer(s, z.astype(jnp.bfloat16), mask, pm, deterministic=True, key=jax.random.PRNGKey(0))
        self.assertEqual(out_z.dtype, jnp.bfloat16); self.assertEqual(out_s.dtype, jnp.float32)   # carried in bf16, handed back in bf16


if __name__ == "__main__":
    unittest.main()
