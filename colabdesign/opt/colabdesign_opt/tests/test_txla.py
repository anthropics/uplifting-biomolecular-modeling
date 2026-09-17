"""Lever `txla` (txla.py): opt_core's forward-only triangle-attention bridge `opt_core.kernels.triattn_xla` on the design step's forward-only
attention calls, composed on lever `triatt` as a jax.custom_vjp (primal = the bridge, fwd/bwd rules = triatt's).

CPU (stand-in stack): the lever protocol surface and its lines' grammar, the per-call routing rule, the registry row (when the registry names
the lever), and a tripwire on the serve-layer call of lever triatt this lever mirrors. GPU (real jax + a CUDA device + the bridge importable):
the bridged op's forward against the fp32 reference, its GRADIENT bitwise equal to triatt's own (the rules ARE triatt), the primal served by
the bridge under jit (census), a masked call honoured; the install contract on the real modules package (after triatt + proj: installed, proj
re-stacked, idempotent, restorable; without triatt: steps aside by name `skipped reason=needs_triatt`). Skipped by name where real jax / a GPU are absent.
"""
from __future__ import annotations

import importlib
import os
import re
import unittest

_PROBE = {}


def _real_jax():
    if "jax" not in _PROBE:
        try:
            jax = importlib.import_module("jax"); jnp = importlib.import_module("jax.numpy")
            _PROBE["jax"] = jax if hasattr(jax, "jit") and hasattr(jnp, "einsum") and hasattr(jax, "devices") else None
        except Exception:  # noqa: BLE001
            _PROBE["jax"] = None
    return _PROBE["jax"]


def _gpu_with_bridge():
    if "gpu" not in _PROBE:
        jax = _real_jax()
        ok = False
        try:
            ok = bool(jax is not None and jax.devices()[0].platform == "gpu")
            if ok:
                importlib.import_module("opt_core.kernels.triattn_xla")
                importlib.import_module("colabdesign.af.alphafold.model.modules")
        except Exception:  # noqa: BLE001
            ok = False
        _PROBE["gpu"] = ok
    return _PROBE["gpu"]


def need_gpu():
    if _real_jax() is None:
        raise unittest.SkipTest("real jax not importable (stand-in stack)")
    if not _gpu_with_bridge():
        raise unittest.SkipTest("needs a CUDA device with opt_core.kernels.triattn_xla and colabdesign importable")


KIT_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # opt/colabdesign_opt


class TestProtocolAndRule(unittest.TestCase):
    """CPU: the module surface levers.install drives, the lines' grammar, the routing rule — no jax import."""

    def test_protocol_surface(self):
        from colabdesign_opt import txla as X
        for attr in ("install", "installed", "uninstall", "off_line", "evidence", "exit_line", "REFUSALS", "NUMERICS", "EXPECTED_FALLBACKS", "ATTENTION_FWD_KERNELS", "make_op", "route"):
            self.assertTrue(hasattr(X, attr), attr)
        self.assertEqual((X.LEVER, X.NUMERICS, X.REQUIRES, X.LAYOUT, X.IMPL_WORD, X.REFUSALS, X.ATTENTION_FWD_KERNELS), ("txla", "precision", "triatt", "BNHSD", "auto", (X.NeedsTriatt, X.TxUnavailable), ()))
        self.assertFalse(X.ATTENTION_FWD_KERNELS)                                # falsy: not an attention-kernel lever (levers.ATTENTION_KERNEL_LEVERS stays pallas, triatt)
        for gone in ("ROW_PREF", "CUDA_ROWS", "ROW_FLOOR", "MIN_TOKENS", "BELOW_SIZE_RULE", "GATED", "row_word", "row_floor"):   # class contract: no row word, no floor table, no size rule of the kit's — the face's own table names the rows
            self.assertFalse(hasattr(X, gone), gone)

    def test_select_row_is_the_faces_table(self):
        """The row per arm of a call is the face's answer under its `auto` word (class contract: the lever hands the face `auto` on every
        call, whatever the face's table names is the row; TX.Refused only when no row serves — the caller hands the call to triatt by name)."""
        from colabdesign_opt import txla as X

        class FakeTX:
            class Refused(Exception):
                pass
            calls = []
            @staticmethod
            def cells():
                return {"by_cc": {"9.0": {"order": ["triattn_native", "cuda_sm90a", "k2b_aot"], "rules": {"triattn_native": {"min_S": 512}, "cuda_sm90a": {"min_S": 384}}}}}
            @staticmethod
            def select(cc, dtype, D, S, N=1, H=1, B=1, has_mask=True, impl="auto", bias_dtype=None, SK=None):
                FakeTX.calls.append((impl, D, S, has_mask))
                if cc != "9.0" and impl in ("triattn_native", "cuda_sm90a"):
                    raise FakeTX.Refused("cc")
                if impl == "triattn_native":
                    if D != 32: raise FakeTX.Refused("head_dim")
                    return "triattn_native", {}
                if impl == "cuda_sm90a":
                    return "cuda_sm90a", {}
                if cc == "9.0" and D == 32 and S >= 512: return "triattn_native", {}
                if cc == "9.0" and D == 32 and S >= 384: return "cuda_sm90a", {}
                if D in (16, 32): return "k2b_aot", {}
                raise FakeTX.Refused("no row")
        TX = FakeTX
        for cc, D, S, mask in (("9.0", 32, 800, False), ("9.0", 32, 431, True), ("9.0", 32, 200, False), ("9.0", 16, 800, True), ("8.0", 32, 800, False), ("8.0", 16, 200, True)):
            FakeTX.calls.clear()
            row = X.select_row(TX, cc, "bf16", D, S, S, 4, 4, None, mask)
            self.assertEqual(FakeTX.calls, [("auto", D, S, mask)], (cc, D, S))            # ONE question to the face, with the word auto — never a row word
            self.assertEqual(row, TX.select(cc, "bf16", D, S, has_mask=mask, impl="auto")[0])   # = the face's own answer (its table), whatever row that is
            self.assertTrue(isinstance(row, str) and row, row)
        self.assertEqual(X.select_row(TX, "9.0", "bf16", 32, 800, 800, 4, 4, None, False, impl="cuda_sm90a"), "cuda_sm90a")   # a row word handed in (tests) goes to the face as given
        with self.assertRaises(TX.Refused):                                            # no row serves 8 per head unpadded: refused by name (the caller counts `refused`, triatt's op serves)
            X.select_row(TX, "9.0", "bf16", 8, 800, 800, 4, 4, None, False)
        with self.assertRaises(TX.Refused):                                            # a CUDA row word on cc 8.0: the face refuses it by name (never a silent substitute in select_row)
            X.select_row(TX, "8.0", "bf16", 32, 800, 800, 4, 4, None, False, impl="triattn_native")
        self.assertEqual(set(X.EXPECTED_FALLBACKS), {"small_call", "refused"})

    def test_off_and_skipped_lines_follow_the_lever_grammar(self):
        from colabdesign_opt import txla as X
        off = X.off_line("ablated")
        self.assertTrue(off.startswith("[colabdesign-opt] LEVER name=txla state=off reason=ablated impl=triattn_xla origin=core"), off)
        self.assertNotIn("  ", off)
        X.reset_for_tests()
        self.assertTrue(X.exit_line().startswith("[colabdesign-opt] LEVER name=txla state=off reason=not_installed impl="), X.exit_line())   # before install: nothing claimed
        X._ledger(); X._STATE["skipped"] = X.NEEDS_TRIATT                                                                                     # what install() records before raising NeedsTriatt
        line = X.exit_line()
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=txla state=skipped reason=cannot_run impl=triattn_xla@"), line)   # the protocol's words (levers.install prints the line proper)
        self.assertRegex(line, r" detail=needs_triatt\b"); self.assertRegex(line, r" served=0 fallback=0 fallback_by=none shapes="); self.assertNotIn("  ", line)
        for tok in ("served_fn=0", "served_grad=0", "vjp_traced=0", "vmap=0", "refused=none", "layout=BNHSD", "impl_word=auto", "requires=triatt", "grad=triatt", "numerics=precision", "precision=bf16", "source=exit"):
            self.assertIn(" " + tok, line)
        for gone in (" min_tokens=", " row_floor=", " words=", " gate.txla="):            # no size rule / floor / row-word tokens: the face's table holds the rows' rules
            self.assertNotIn(gone, line)
        X.reset_for_tests()

    def test_route_rule(self):
        from colabdesign_opt import txla as X
        self.assertEqual(X.route(500, 500, 32, 32), "bridge")                     # triangle attention / MSA row attention
        self.assertEqual(X.route(500, 500, 16, 16), "bridge")                     # template pair stack; the extra-MSA row attention once the seam padded 8 -> 16
        self.assertEqual(X.route(384, 384, 32, 32), "bridge"); self.assertEqual(X.route(384, 2, 32, 32), "small_call")
        self.assertEqual(X.route(383, 383, 32, 32), "bridge"); self.assertEqual(X.route(200, 200, 16, 16), "bridge"); self.assertEqual(X.route(16, 500, 16, 16), "bridge")   # no size rule of the kit's: every eligible call reaches the face, whose table names the row
        self.assertEqual(X.route(500, 2, 32, 32), "small_call")                   # MSA column attention over 2 sequences (keys)
        self.assertEqual(X.route(2, 500, 32, 32), "small_call")
        self.assertEqual(X.route(500, 500, 8, 8), "small_call")                   # an unpadded head dim below 16 (proj hands D=8 calls down; the seam pads before the op)
        self.assertEqual(X.route(500, 500, 32, 8), "small_call")

    def test_registry_row_when_registered(self):
        from colabdesign_opt import modes, registry
        if "txla" not in registry.LEVERS:
            raise unittest.SkipTest("the registry does not name lever txla (yet)")
        row = registry.LEVERS["txla"]
        self.assertEqual((row.numerics, row.module, row.line_name, row.supersedes), ("precision", "colabdesign_opt.txla", "txla", ()))
        order = list(registry.ORDER)
        self.assertLess(order.index("triatt"), order.index("txla")); self.assertLess(order.index("proj"), order.index("txla"))   # installed after triatt (its op) and after proj (re-stacked)
        self.assertIn("txla", modes.levers_of("fast")); self.assertNotIn("txla", modes.levers_of("exact"))
        r = modes.resolve("fast-no-txla")
        self.assertEqual((r.base, r.ablated), ("fast", ("txla",))); self.assertNotIn("txla", r.levers)
        self.assertIn("txla", modes.resolve("fast-no-triatt").levers)             # still in the set: it steps aside at install, by name (needs_triatt)

    def test_tripwire_on_triatt_levers_serve_call(self):
        """txla re-serves the modules package with triatt's ledger / scope / rules: the call it mirrors must still be the one triatt makes."""
        src = open(os.path.join(KIT_PKG, "kernels", "triatt_lever.py")).read()
        self.assertIn("F1.enable([CDM], ledger=LEDGER, all_calls=True, op=K.attention, pad_head_dim_below_min=True, min_keys=MIN_KEYS)", src)
        self.assertRegex(src, r"(?m)^SCOPE = \"all_calls\""); self.assertRegex(src, r"(?m)^MIN_KEYS = 16\b")
        from colabdesign_opt.kernels import provider as P
        ksrc = open(P.row_file("triatt")).read()                                  # the ROW module lever triatt binds (cd_triatt: the shared core's triatt_attn.py), located without importing it — the kit's private copy left the tree in 0.10.3
        from colabdesign_opt import txla as X
        self.assertRegex(ksrc, rf"(?m)^MIN_HEAD_DIM = {X.MIN_HEAD_DIM}\b"); self.assertRegex(ksrc, rf"(?m)^MIN_SEQ = {X.MIN_SEQ}\b")
        self.assertTrue(X.PAD_HEAD_DIM_BELOW_MIN)
        psrc = open(os.path.join(KIT_PKG, "kernels", "proj_attn.py")).read()
        self.assertIn("op = active_op(rec)", psrc)                                # proj reads the serve layer's record AT TRACE TIME: re-serving the record's op reaches proj's calls


class TestEvidenceCensusReadsTxla(unittest.TestCase):
    """CPU: the kit's evidence census (evidence.py) classifies lever txla from its ONE line — applied / gated / fallback / missing — with the
    fallback words the lever declares, prints its census pair on the EVIDENCE line, and names a gated txla with the lever's gate text."""

    ON = ("[colabdesign-opt] LEVER name=txla state=on impl=triattn_xla@0.5.116.0 origin=core served=8 fallback=1 fallback_by=small_call:1 "
          "shapes=B500xH4xS500xD32:4 rows=triattn_native:10,k2b_aot:6 served_fn=8 served_grad=0 vjp_traced=9 vmap=0 refused=none layout=BNHSD impl_word=auto requires=triatt "
          "grad=triatt proj_restacked=1 numerics=precision precision=bf16 cc=9.0 tx=1.6.1 source=exit pid=7")
    ALL_ASIDE = ON.replace("served=8 fallback=1 fallback_by=small_call:1", "served=0 fallback=9 fallback_by=refused:8,small_call:1").replace("rows=triattn_native:10,k2b_aot:6 served_fn=8", "rows=none served_fn=0")
    ODD = ON.replace("fallback=1 fallback_by=small_call:1", "fallback=2 fallback_by=small_call:1,mystery:1")
    NONE_SERVED = ON.replace("state=on ", "state=skipped reason=no_calls ").replace("served=8 fallback=1 fallback_by=small_call:1", "served=0 fallback=0 fallback_by=none")

    def test_declared_words_are_the_levers(self):
        from colabdesign_opt import evidence as E, txla as X
        self.assertEqual(E.TXLA_DECLARED_FALLBACKS, tuple(X.EXPECTED_FALLBACKS)); self.assertEqual(E.TXLA_LINE_NAME, X.LEVER)
        self.assertIn(X.LEVER, E.CENSUS_LEVERS); self.assertEqual(list(E.CENSUS_LEVERS)[-1], X.LEVER); self.assertNotIn(X.LEVER, E.GATES)   # no size gate of the kit's: txla is never `gated`

    def _classify(self, txla_line):
        from colabdesign_opt import evidence as E, modes
        res = modes.resolve("fast")
        lines = [l for l in (txla_line,) if l]
        return E.classify(lines, res.levers), res

    def test_states(self):
        from colabdesign_opt import evidence as E, report
        ev, _ = self._classify(self.ON)
        self.assertEqual(ev["state"]["txla"], "applied"); self.assertIn("txla", ev["applied"]); self.assertEqual((ev["txla_served"], ev["txla_fallback"], ev["txla_fallback_by"]), (8, 1, {"small_call": 1}))
        self.assertIn(" txla_served=8 txla_fallback_by=small_call:1 pallas_served=", report.evidence_line(ev))          # the census pair, after proj's, before pallas'
        self.assertNotIn("txla", E.verdict(ev)["partial"])
        ev, _ = self._classify(self.ALL_ASIDE)                                                                             # every call handed to triatt by DECLARED words: named aside, not a defect
        self.assertEqual(ev["state"]["txla"], "skipped"); self.assertNotIn("txla", ev.get("gated", {})); self.assertEqual(ev["txla_served"], 0)
        v = E.verdict(ev); self.assertNotIn("txla", v["partial"])
        self.assertIn(" txla_served=0 txla_fallback_by=refused:8,small_call:1 ", report.evidence_line(ev))
        ev, _ = self._classify(self.ODD)
        self.assertEqual(ev["state"]["txla"], "fallback"); self.assertTrue(E.verdict(ev)["partial"]["txla"].startswith("fallback: undeclared reason(s) "), E.verdict(ev)["partial"])
        ev, _ = self._classify(self.NONE_SERVED)
        self.assertEqual(ev["state"]["txla"], "missing"); self.assertEqual(E.verdict(ev)["partial"]["txla"], "missing: installed, the kernels served no call (served=0)")
        ev, _ = self._classify(None)                                                                                       # no line of its own
        self.assertEqual(ev["state"]["txla"], "missing"); self.assertTrue(E.verdict(ev)["partial"]["txla"].startswith("missing: no line of its own"))
        self.assertIn(" txla_served=n/a txla_fallback_by=n/a ", report.evidence_line(ev))

    def test_stepped_aside_is_skipped_not_missing(self):
        from colabdesign_opt import evidence as E
        aside = "[colabdesign-opt] LEVER name=txla state=skipped reason=cannot_run impl=txla@kit origin=kit detail=NeedsTriatt:_lever_triatt_does_not_serve source=install pid=7"
        ev, _ = self._classify(aside)
        self.assertEqual(ev["state"]["txla"], "skipped"); self.assertEqual(ev["skipped_by"]["txla"], "cannot_run: NeedsTriatt:_lever_triatt_does_not_serve"); self.assertNotIn("txla", E.verdict(ev)["partial"])


class TestBridgedOpOnGpu(unittest.TestCase):
    """GPU: the custom_vjp op — forward = the bridge (close to the fp32 reference), gradient = triatt's bit for bit, census, masks."""

    def setUp(self):
        need_gpu()

    def _inputs(self, B=6, H=4, S=400, D=32, seed=0, mask_tail=0):
        import jax
        import jax.numpy as jnp
        ks = jax.random.split(jax.random.PRNGKey(seed), 4)
        q = jax.random.normal(ks[0], (B, H, S, D), jnp.float32).astype(jnp.bfloat16)
        k = jax.random.normal(ks[1], (B, H, S, D), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(ks[2], (B, H, S, D), jnp.float32).astype(jnp.bfloat16)
        bias = (0.5 * jax.random.normal(ks[3], (H, S, S), jnp.float32)).astype(jnp.bfloat16)
        mask = jnp.ones((B, S), bool)
        if mask_tail:
            mask = mask.at[:, S - mask_tail:].set(False)
        return q, k, v, bias, mask, float(D) ** -0.5

    def test_forward_close_to_reference_gradient_bitwise_triatt(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from colabdesign_opt import txla as X
        from colabdesign_opt.kernels import provider as P
        K = importlib.import_module(P.BINDINGS["triatt"].rows["cd_triatt"])      # the ROW module lever triatt binds (opt_core.kernels.pallas_triatt.triatt_attn)
        X.reset_for_tests()
        op = X.make_op(K.attention)
        for D, tail in ((32, 0), (16, 0), (32, 24)):
            q, k, v, bias, mask, scale = self._inputs(D=D, mask_tail=tail)
            out = jax.jit(op, static_argnums=(5,))(q, k, v, bias, mask, scale)
            ref = K.reference_attention(q, k, v, bias, mask, scale)
            err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
            self.assertTrue(bool(jnp.all(jnp.isfinite(out.astype(jnp.float32)))), f"D={D} tail={tail}")
            self.assertLess(err, 3e-2, f"D={D} tail={tail} max|bridge-ref|={err}")
            tri = K.attention(q, k, v, bias, mask, scale)
            self.assertLess(float(jnp.max(jnp.abs(out.astype(jnp.float32) - tri.astype(jnp.float32)))), 3e-2)
            w = jax.random.normal(jax.random.PRNGKey(7), out.shape, jnp.float32)
            loss = lambda f: (lambda q_, k_, v_, b_: jnp.sum(f(q_, k_, v_, b_, mask, scale).astype(jnp.float32) * w))  # noqa: E731
            g_b = jax.jit(jax.grad(loss(op), argnums=(0, 1, 2, 3)))(q, k, v, bias)
            g_t = jax.jit(jax.grad(loss(K.attention), argnums=(0, 1, 2, 3)))(q, k, v, bias)
            for a, b, n in zip(g_b, g_t, ("dq", "dk", "dv", "dbias")):
                self.assertTrue(np.array_equal(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))), f"{n} D={D} tail={tail}: the bwd rule must be triatt's, bit for bit")
            g_b2 = jax.jit(jax.grad(loss(op), argnums=(0, 1, 2, 3)))(q, k, v, bias)          # the graded path run to run bitwise either way
            for a, b in zip(g_b, g_b2):
                self.assertTrue(np.array_equal(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))), "gradient run-to-run bitwise")
            again = jax.jit(op, static_argnums=(5,))(q, k, v, bias, mask, scale)
            self.assertTrue(np.array_equal(np.asarray(out.astype(jnp.float32)), np.asarray(again.astype(jnp.float32))), "run-to-run bitwise")
        ev = X.evidence()
        self.assertGreaterEqual(ev["served"], 2, ev); self.assertGreaterEqual(ev["vjp_traced"], 2, ev)   # TRACED primal calls: one per distinct (shape, dtype) under jit
        self.assertTrue(X.tx_rows(), X.tx_rows())                                 # the bridge bound kernels (its own census)
        small = self._inputs(S=400)                                                 # S_k = 2: the small path is triatt's op, counted
        q, k, v, bias, mask, scale = small
        o2 = op(q, k[:, :, :2], v[:, :, :2], bias[:, :, :2], mask[:, :2], scale)
        self.assertEqual(tuple(o2.shape), tuple(q.shape)); self.assertGreaterEqual(X.evidence()["fallback_by"].get("small_call", 0), 1)
        q, k, v, bias, mask, scale = self._inputs(S=200)                            # 200 queries: no size rule of the kit's — the face serves it (its table's row), same class as triatt's forward
        served0 = X.evidence()["served"]
        o3 = jax.jit(op, static_argnums=(5,))(q, k, v, bias, mask, scale); t3 = jax.jit(K.attention, static_argnums=(5,))(q, k, v, bias, mask, scale)
        e3 = float(jnp.max(jnp.abs(o3.astype(jnp.float32) - t3.astype(jnp.float32)))); m3 = float(jnp.max(jnp.abs(t3.astype(jnp.float32))))
        self.assertLessEqual(e3, 2.5e-2 * m3 + 2e-2, f"S=200 bridge vs triatt forward max|err| {e3:.3e}"); self.assertEqual(X.evidence()["served"], served0 + 1); self.assertNotIn("below_size_rule", X.evidence()["fallback_by"])
        q, k, v, bias, mask, scale = self._inputs(B=4, S=400)                       # a caller under jax.vmap (hk.vmap): served slice by slice (custom_vmap -> lax.map), = the unmapped calls
        Q = jnp.stack([q, q[::-1], q * 0.5]); KK = jnp.stack([k, k * 0.5, k[::-1]]); VV = jnp.stack([v, v[::-1], v])
        om = jax.jit(jax.vmap(lambda a, b_, c: op(a, b_, c, bias, mask, scale)))(Q, KK, VV)
        os_ = jnp.stack([jax.jit(op, static_argnums=(5,))(Q[i], KK[i], VV[i], bias, mask, scale) for i in range(3)])
        self.assertTrue(np.array_equal(np.asarray(om.astype(jnp.float32)), np.asarray(os_.astype(jnp.float32))), "vmapped = per-slice, bit for bit"); self.assertGreaterEqual(X.evidence()["vmap"], 1)
        X.reset_for_tests()


class TestInstallContractOnGpu(unittest.TestCase):
    def setUp(self):
        need_gpu()

    def test_after_triatt_and_proj_idempotent_restorable_and_steps_aside_without_triatt(self):
        from colabdesign.af.alphafold.model import modules as CDM
        from opt_core.kernels import pallas_attn_serve as F1
        from colabdesign_opt import pallas as KP, txla as X
        from colabdesign_opt.kernels import proj_attn as PJ, triatt_lever as TL
        PJ.uninstall(); F1.disable([CDM]); X.reset_for_tests()
        TL.install(); PJ.install()
        a = X.install(); b = X.install()
        self.assertTrue(a["installed"] and b["installed"] and a["skipped"] is None, (a, b))
        self.assertEqual(a["proj_restacked"], 1)
        self.assertTrue(PJ.installed() and TL.installed() and X.installed())         # proj back on top, triatt's ledger still behind the class, our op in the record
        rec = F1._ENABLED[id(CDM)]
        self.assertIs(rec["ledger"], TL.LEDGER); self.assertTrue(getattr(rec["op"], X.MARKER, False)); self.assertEqual(CDM.Attention.__name__, "Attention")
        line = X.lever_line()
        self.assertTrue(re.match(r"^\[colabdesign-opt\] LEVER name=txla state=on impl=triattn_xla@\S+ origin=core served=\d+ fallback=\d+ fallback_by=\S+ shapes=\S+ "
                                 r"rows=\S+ served_fn=\d+ served_grad=0 vjp_traced=\d+ vmap=\d+ refused=\S+ layout=BNHSD impl_word=auto requires=triatt grad=triatt proj_restacked=1 "
                                 r"numerics=precision precision=bf16 cc=\S+ tx=\S+ source=exit pid=\d+$", line), line)
        self.assertTrue(X.uninstall())
        self.assertFalse(X.installed()); self.assertTrue(PJ.installed() and TL.installed())
        self.assertFalse(getattr(F1._ENABLED[id(CDM)]["op"], X.MARKER, False))   # triatt's own op is back in the record
        PJ.uninstall(); F1.disable([CDM]); X.reset_for_tests()
        KP.install()                                                                # = fast-no-triatt: pallas restored, no triatt behind the class
        with self.assertRaises(X.NeedsTriatt):                                       # the kit's step-aside protocol: levers.install prints state=skipped reason=cannot_run detail=NeedsTriatt:...
            X.install()
        c = X.info()
        self.assertFalse(c["installed"]); self.assertEqual(c["skipped"], "needs_triatt"); self.assertIn(X.NeedsTriatt, X.REFUSALS)
        self.assertTrue(X.exit_line().startswith("[colabdesign-opt] LEVER name=txla state=skipped reason=cannot_run impl=triattn_xla@"), X.exit_line()); self.assertIn(" detail=needs_triatt", X.exit_line())
        F1.disable([CDM]); X.reset_for_tests()


try:
    import pytest  # noqa: F401
except ImportError:                                              # no pytest on the image: the kit's stand-in runs the file with unittest
    from colabdesign_opt.tests import _noptest as pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
