"""Lever pair F6+F8 — one pipeline: F6 (trimul_layout, channel-major layout) provides the served call and its extension point; F8
(trimul_fused) dispatches that extension point to its op (trimul_provider: the shared core's kernel provider bound by tier word) when
installed after F6. Covers: registry wiring and install order for both per-step levers (F8 must install after F6; a third lever wrapping
F6's call before or after F8 composes or is refused by name); both levers' spec grammars and describe()/emit_line() evidence; F6 serving
joltz's own arithmetic on CPU (values and gradients, both directions, batch, odd sizes, masks with padded rows); F8's refusal chain
(not_installed/needs_F6/op_missing/probe_failed/op_refused) and its dispatch through F6's extension point with the op's kernel-vs-xla
accounting as LEVER words. The op's own wiring, the provider's class contracts per card and the GPU numerics live in test_trimul_provider.py."""


import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

from . import _stubs
from mosaic_opt import levers, modes, registry


LID_F6 = "F6"
LID_F8 = "F8"

F6_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "trimul_layout.py")
F6_TOOLS_FILE = os.path.join(_stubs.KIT, "tools", "trimul_layout.py")

KIT_FAST = os.path.join(_stubs.KIT, "mosaic_fast")
F8_KIT_FILE = os.path.join(KIT_FAST, "trimul_fused.py")
OP_FILE = os.path.join(KIT_FAST, "trimul_provider.py")


def _importable(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _gpu():
    try:
        import jax  # noqa: WPS433
        return jax.default_backend() == "gpu"
    except Exception:  # noqa: BLE001
        return False


def _load_trimul_layout(name="trimul_layout_under_test"):
    spec = importlib.util.spec_from_file_location(name, F6_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stubs._standins()):
        spec.loader.exec_module(mod)
    return mod


def _load_trimul_fused(path=F8_KIT_FILE, name="trimul_fused_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stubs._standins()):
        spec.loader.exec_module(mod)
    return mod


def _standin_joltz(calls):
    """A stand-in `joltz` with the two triangle-multiplication classes whose `__call__`s read as joltz's own (module and qualname, which is how
    the lever tells joltz's call from another lever's wrapper), plus inert `jax` / `equinox` (clear_caches only)."""
    class Out:
        def __call__(self, x, mask):
            calls.append(("out", x, mask)); return ("stock-out", x)

    class In:
        def __call__(self, x, mask):
            calls.append(("in", x, mask)); return ("stock-in", x)
    for cls, name in ((Out, "TriangleMultiplicationOutgoing"), (In, "TriangleMultiplicationIncoming")):
        cls.__call__.__module__ = "joltz"; cls.__call__.__qualname__ = f"{name}.__call__"
    fake = types.ModuleType("joltz"); fake.TriangleMultiplicationOutgoing = Out; fake.TriangleMultiplicationIncoming = In
    jx = types.ModuleType("jax"); jx.clear_caches = lambda: None
    eq = types.ModuleType("equinox"); eq.clear_caches = lambda: None
    return Out, In, fake, {"joltz": fake, "jax": jx, "equinox": eq}


def _wrap_like_p5(cls):
    """What the memory lever's `sub` rematerialisation does at ITS install: replace `cls.__call__` by a wrapper around the call it finds."""
    orig = cls.__call__

    def call(self, *args, **kwargs):
        return ("remat", orig(self, *args, **kwargs))
    call.__name__ = getattr(orig, "__name__", "__call__"); call.__qualname__ = getattr(orig, "__qualname__", "__call__"); call.__module__ = "mosaic.fast.memlevers"
    cls.__call__ = call
    return orig


class _Shape:
    """An activation stand-in: the served call reads x.shape only (census words)."""
    def __init__(self, shape):
        self.shape = tuple(shape)


def _op_module(ok=True, kind=None, calls=None, words=("fast", "big"), xla_below=0):
    """A stand-in op with the op's surface: probe() (ok/kind), configure(word) refusing a word outside `words` BY NAME, reset(), settings(),
    census() (kernel_calls / xla_calls / calls / words / rows / cells / traces), trimul(); calls below `xla_below` tokens count as xla(cell)."""
    op = types.ModuleType("mosaic.fast.trimul_provider")
    op.VERSION = "vtest"; op.WORDS = tuple(words); op.SETTING = "fast"
    op.probe = lambda: {"ok": ok, "kind": kind or ("ok" if ok else "unknown"), "cc": "test", "bf16_fwdbwd": "rowA@cellA"}
    state = {"on": False, "word": None}
    class OpRefusal(RuntimeError):
        def __init__(self, kind, detail=""):
            self.kind = self.reason = kind; super().__init__(f"{kind}: {detail}")
    def configure(word=None):
        w = word or "fast"
        if w not in words:
            raise OpRefusal("unknown_word", w)
        state.update(on=True, word=w)
        return dict(state)
    op.configure = configure
    counter = {}
    def reset():
        state.update(on=False, word=None); counter.clear(); return dict(state)
    op.reset = reset
    op.settings = lambda: dict(state, version="vtest")
    def trimul(module, x, mask, direction):
        n = x.shape[-2]
        body = "kernel" if n >= xla_below else "xla(cell)"
        key = f"{direction}:N{n}:word={state['word']}:body={body}"
        calls.append(("op", direction, x.shape, state["word"])); counter[key] = counter.get(key, 0) + 1; return ("op-out", direction)
    trimul.__serves_dtypes__ = ("float32", "bfloat16")
    op.trimul = trimul
    def census():
        k = sum(v for kk, v in counter.items() if kk.endswith("body=kernel")); xl = sum(v for kk, v in counter.items() if "body=xla(" in kk)
        return {"kernel_calls": k, "xla_calls": xl, "calls": sum(counter.values()), "words": dict(counter), "rows": {"f32:rowA/rowB": sum(counter.values())} if counter else {},
                "cells": ["cellA"] if counter else [], "traces": {}}
    op.census = census
    return op



@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestTrimulLayoutLever(unittest.TestCase):

    def test_registry_entry(self):
        """F6 is a per-step lever (route install) of the fast numerics class, kit-origin, wired into the `fast` and `big` rows after E10
        (before P5 on `big`); one home (no tools/ twin)."""
        lv = registry.LEVERS[LID_F6]
        self.assertEqual((lv.route, lv.klass, lv.origin, lv.wired, lv.module, lv.flag), ("install", "fast", "kit", True, "mosaic.fast.trimul_layout", "--levers fast"))
        self.assertIn(LID_F6, registry.INSTALL); self.assertIn(LID_F6, registry.WIRED)
        self.assertTrue(os.path.isfile(F6_KIT_FILE)); self.assertFalse(os.path.exists(F6_TOOLS_FILE), "F6 has one home: no tools/ twin")
        self.assertEqual(lv.kit_file, "mosaic_fast/trimul_layout.py")
        fast, big = list(modes.install_levers_of("fast")), list(modes.install_levers_of("big"))
        self.assertEqual(fast[fast.index("E10") + 1], LID_F6); self.assertEqual(fast[fast.index(LID_F6) + 1], "F8")   # F8 (the kernel lever on this layout) follows F6
        self.assertEqual(tuple(big[big.index(LID_F6) + 1 : big.index(LID_F6) + 4]), ("F8", "F9", "P5"))                         # F8 gives the rebound call its kernel body; P5's sub-block remat wraps the rebound triangle-multiplication calls last
        plan = levers.plan("fast")
        self.assertIn(LID_F6, plan["levers"]); self.assertIsNone(plan["specs"][LID_F6])           # configure(None) = the default setting

    def test_spec_grammar(self):
        T = _load_trimul_layout("trimul_grammar")
        self.assertEqual(T.SETTING, "cmajor"); self.assertEqual(T.ENV_REQUIRED, {}); self.assertEqual(T.KLASS, "fast")
        for spec in (None, "", "cmajor", "CMAJOR"):
            self.assertEqual(T.parse(spec), {"on": True, "spec": "cmajor"}, spec)
        for spec in ("stock", "off"):
            self.assertEqual(T.parse(spec), {"on": False, "spec": "stock"}, spec)
        with self.assertRaises(T.Refusal) as cm:
            T.parse("rows64")
        self.assertEqual(cm.exception.reason, "unknown_spec")

    def test_install_roundtrip_over_a_standin_joltz(self):
        """install() rebinds both triangle-multiplication calls; off, the rebound call delegates to the ORIGINAL function; configure(None) before
        install refuses `not_installed`; gate() refuses `not_configured` when installed and never turned on; uninstall() restores the originals."""
        T = _load_trimul_layout("trimul_standins")
        calls = []

        Out, In, fake, standins = _standin_joltz(calls)
        stock_out, stock_in = Out.__call__, In.__call__
        with mock.patch.dict(sys.modules, standins):
            with self.assertRaises(T.Refusal) as cm:
                T.configure(None)
            self.assertEqual(cm.exception.reason, "not_installed")
            T.install(); T.install()                                                    # idempotent
            self.assertTrue(T.installed()); self.assertIsNot(Out.__call__, stock_out)
            self.assertEqual(Out()("x", "m"), ("stock-out", "x")); self.assertEqual(In()("y", "n"), ("stock-in", "y"))   # off: the stock bodies
            rec = T.configure("stock")
            self.assertEqual((rec["on"], rec["installed"]), (0, 1))
            with self.assertRaises(T.Refusal) as cm:
                T.gate()
            self.assertEqual(cm.exception.reason, "not_configured")
            T.uninstall()
            self.assertIs(Out.__call__, stock_out); self.assertIs(In.__call__, stock_in); self.assertFalse(T.installed())
        self.assertEqual([c[0] for c in calls], ["out", "in"])


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestInstallOrder(unittest.TestCase):

    def test_set_body_extension_point(self):
        """set_body(): refuses `not_installed` before install and `body_not_callable`; None ⇒ the served call runs trimul_cmajor with
        (module, x, mask, direction); a body ⇒ that body with the same operands; describe()/the census word `body=`; uninstall resets to xla."""
        T = _load_trimul_layout("trimul_setbody")
        calls = []
        Out, In, fake, standins = _standin_joltz(calls)
        with mock.patch.dict(sys.modules, standins):
            with self.assertRaises(T.Refusal) as cm:
                T.set_body(lambda *a: None)
            self.assertEqual(cm.exception.reason, "not_installed")
            T.install(); T.configure(None)
            with self.assertRaises(T.Refusal) as cm:
                T.set_body("not a function")
            self.assertEqual(cm.exception.reason, "body_not_callable")
            seen = []
            T.trimul_cmajor = lambda module, x, mask, direction: seen.append(("cmajor", direction, x.shape)) or ("cmajor", direction)   # the XLA body, recorded
            class X:                                                       # an activation stand-in: the served call reads .shape for its census
                shape = (1, 8, 8, 16)
            self.assertEqual(T.describe()["body"], "xla"); self.assertIsNone(T.STATE["body"])
            self.assertEqual(Out()(X(), "m"), ("cmajor", "outgoing")); self.assertEqual(In()(X(), "m"), ("cmajor", "incoming"))
            body_calls = []
            T.set_body(lambda module, x, mask, direction: body_calls.append((type(module).__name__, x.shape, mask, direction)) or ("body", direction), "test kernel")
            self.assertEqual(T.describe()["body"], "test_kernel")
            self.assertEqual(Out()(X(), "m"), ("body", "outgoing")); self.assertEqual(In()(X(), "n"), ("body", "incoming"))
            self.assertEqual(body_calls, [("Out", (1, 8, 8, 16), "m", "outgoing"), ("In", (1, 8, 8, 16), "n", "incoming")])
            self.assertIn("body=test_kernel", T.emit_line("t"))
            T.set_body(None)
            self.assertEqual(T.describe()["body"], "xla"); self.assertEqual(Out()(X(), "m"), ("cmajor", "outgoing"))
            T.set_body(lambda *a: ("again",), "k2"); T.uninstall()
            self.assertEqual((T.STATE["body"], T.STATE["body_word"]), (None, "xla"))
        self.assertEqual([c[0] for c in seen], ["cmajor", "cmajor", "cmajor"])

    def test_wrapper_installed_first_is_refused_by_name(self):
        """Order P5→F6 (a lever wrapping the triangle multiplications installed BEFORE this one — e.g. `mode:big(plus='F6')`): install() refuses
        `install_order` and touches neither class (the served body would bypass the wrapper silently at these two sites)."""
        T = _load_trimul_layout("trimul_order_refused"); calls = []
        Out, In, fake, standins = _standin_joltz(calls)
        with mock.patch.dict(sys.modules, standins):
            _wrap_like_p5(Out)                                                          # only one of the two sites wrapped is enough to refuse
            wrapped_out, own_in = Out.__call__, In.__call__
            with self.assertRaises(T.Refusal) as cm:
                T.install()
            self.assertEqual(cm.exception.reason, "install_order"); self.assertIn("TriangleMultiplicationOutgoing", str(cm.exception))
            self.assertIs(Out.__call__, wrapped_out); self.assertIs(In.__call__, own_in); self.assertFalse(T.installed())   # nothing touched

    def test_wrapper_installed_after_composes(self):
        """Order F6→P5 (this lever in the tier tuple BEFORE the memory lever): the wrapper wraps the SERVED call — a call reaches wrapper → served
        call → (lever off) joltz's own body; uninstalling in reverse order restores joltz's own calls."""
        T = _load_trimul_layout("trimul_order_composes"); calls = []
        Out, In, fake, standins = _standin_joltz(calls)
        own_out, own_in = Out.__call__, In.__call__
        with mock.patch.dict(sys.modules, standins):
            T.install()
            served_out = Out.__call__
            self.assertEqual(getattr(served_out, "__trimul_direction__", None), "outgoing")
            found_by_p5 = _wrap_like_p5(Out); _wrap_like_p5(In)
            self.assertIs(found_by_p5, served_out)                                      # the wrapper holds the served call, not joltz's own
            self.assertEqual(Out()("x", "m"), ("remat", ("stock-out", "x")))            # wrapper → served call → (off) the stock body
            self.assertEqual(In()("y", "n"), ("remat", ("stock-in", "y")))
            Out.__call__ = served_out; In.__call__ = T._CALLS["incoming"]              # the memory lever's uninstall (reverse order) puts back what it found
            T.uninstall()
            self.assertIs(Out.__call__, own_out); self.assertIs(In.__call__, own_in)
        self.assertEqual([c[0] for c in calls], ["out", "in"])


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestTrimulFusedLever(unittest.TestCase):

    def test_registry_entry(self):
        """F8 is a per-step lever (route install) of the fast numerics class, kit-origin, wired into the `fast` and `big` rows right after F6
        (before P5 on `big`); one home (no tools/ twin); its op file is carried, the retired op files are not."""
        lv = registry.LEVERS[LID_F8]
        self.assertEqual((lv.route, lv.klass, lv.origin, lv.wired, lv.module, lv.flag), ("install", "fast", "kit", True, "mosaic.fast.trimul_fused", "--levers fast"))
        self.assertIn(LID_F8, registry.INSTALL); self.assertIn(LID_F8, registry.WIRED)
        self.assertLess(registry.INSTALL.index("F6"), registry.INSTALL.index(LID_F8), "F8 installs after F6 in registry row order")
        for word in ("fast", "big"):
            row = modes.KIT_MODES[word]["levers"]; self.assertEqual(row[row.index("F6") + 1], LID_F8)
        self.assertEqual(modes.KIT_MODES["big"]["levers"][modes.KIT_MODES["big"]["levers"].index(LID_F8) + 1 : modes.KIT_MODES["big"]["levers"].index(LID_F8) + 3], ("F9", "P5"))   # F9 (the fused transition) rides between F8 and P5
        for p in (F8_KIT_FILE, OP_FILE):
            self.assertTrue(os.path.isfile(p), p)
            self.assertFalse(os.path.exists(os.path.join(_stubs.KIT, "tools", os.path.basename(p))), "one home: no tools/ twin")
        for gone in ("trimul_pallas_op.py", "trimul_pallas_fwd.py", "trimul_vjp.py", "trimul_bf16.py"):     # the kit carries no kernel adapter, backward or row pin for this family
            self.assertFalse(os.path.exists(os.path.join(KIT_FAST, gone)), gone)

    def test_tier_rows_carry_the_tier_word(self):
        """The fast row asks the provider with `fast`, the big row with `big` (the one fast-row lever the big row runs at another setting,
        named in TIER_FOLLOWING); `--levers fast+F8=big` is the by-hand A/B and is labelled."""
        self.assertEqual(modes.specs_of("fast").get(LID_F8), "fast"); self.assertEqual(modes.specs_of("big").get(LID_F8), "big")
        self.assertIn(LID_F8, modes.TIER_FOLLOWING); self.assertNotIn(LID_F8, modes.BIG_SPECS_DIFFER)   # the tier word itself is the one stated difference (modes.TIER_FOLLOWING), not a setting
        self.assertEqual(levers.plan("fast")["specs"][LID_F8], "fast"); self.assertEqual(levers.plan("big")["specs"][LID_F8], "big")
        q = levers.plan("fast", F8="big")
        self.assertEqual((q["label"], q["specs"][LID_F8]), ("fast+F8=big", "big"))

    def test_spec_grammar(self):
        F = _load_trimul_fused(name="trimul_fused_grammar")
        self.assertEqual(F.SETTING, "fast"); self.assertEqual(F.WORDS, ("fast", "big")); self.assertEqual(F.OP_MODULE, "mosaic.fast.trimul_provider")
        self.assertEqual(F.parse(None), {"on": True, "spec": "fast", "word": "fast"})                      # the default setting
        for w in F.WORDS:
            self.assertEqual(F.parse(w), {"on": True, "spec": w, "word": w}); self.assertEqual(F.parse(f" {w.upper()} ")["word"], w)
        for off in ("stock", "off", "OFF"):
            self.assertEqual(F.parse(off), {"on": False, "spec": "stock", "word": None})
        for bad in ("primal_kernel", "primal_kernel/cd", "xla", "vjp_kernel", "cd", "planes", "exact", "fast/cd", "tier", "pallas", "a=b", "", "kernel"):   # the retired words and row names are refused by name
            with self.assertRaises(F.Refusal) as cm:
                F.parse(bad)
            self.assertEqual(cm.exception.reason, "unknown_spec", bad)

    def test_configure_refusals_and_dispatch_through_f6(self):
        """not_installed → needs_F6 (F6 absent; F6 installed but off) → op_missing → probe_failed:<reason> (F6's body untouched) → op_refused:<kind>
        → on: F6's served call runs the op under the word, census counts, gate passes, the LEVER line names it; 'stock' and uninstall restore
        F6's XLA body and reset the op."""
        F = _load_trimul_fused(name="trimul_fused_dispatch")
        T = _load_trimul_layout("trimul_layout_for_f8")
        calls = []
        Out, In, fake, standins = _standin_joltz(calls)
        standins = dict(standins)
        standins.pop("mosaic.fast.trimul_layout", None); standins.pop("mosaic.fast.trimul_provider", None)
        with mock.patch.dict(sys.modules, standins):
            sys.modules.pop("mosaic.fast.trimul_layout", None); sys.modules.pop("mosaic.fast.trimul_provider", None)
            with self.assertRaises(F.Refusal) as cm:
                F.configure(None)
            self.assertEqual(cm.exception.reason, "not_installed")
            F.install(); F.install(); self.assertTrue(F.installed())
            with self.assertRaises(F.Refusal) as cm:                      # F6's module not importable at all
                F.configure(None)
            self.assertEqual(cm.exception.reason, "needs_F6")
            sys.modules["mosaic.fast.trimul_layout"] = T                   # F6 present: installed but OFF
            T.install(); T.configure("stock")
            with self.assertRaises(F.Refusal) as cm:
                F.configure(None)
            self.assertEqual(cm.exception.reason, "needs_F6")
            T.configure(None)                                              # F6 on
            sys.modules["mosaic.fast.trimul_provider"] = None                 # no op module in the installed mosaic (an import that fails)
            with self.assertRaises(F.Refusal) as cm:
                F.configure(None)
            self.assertEqual(cm.exception.reason, "op_missing")
            sys.modules["mosaic.fast.trimul_provider"] = _op_module(ok=False, kind="backend_not_gpu", calls=calls)
            with self.assertRaises(F.Refusal) as cm:
                F.configure(None)
            self.assertEqual(cm.exception.reason, "probe_failed:backend_not_gpu")
            self.assertIsNone(T.STATE["body"]); self.assertEqual(T.describe()["body"], "xla")        # refused: F6 untouched
            self.assertEqual(F.describe()["probe"], "backend_not_gpu"); self.assertEqual(F.serves_dtypes(), ())
            sys.modules["mosaic.fast.trimul_provider"] = _op_module(ok=True, calls=calls, words=("fast",))   # an op that does not know the big word refuses it by name
            with self.assertRaises(F.Refusal) as cm:
                F.configure("big")
            self.assertEqual(cm.exception.reason, "op_refused:unknown_word"); self.assertIsNone(T.STATE["body"])
            sys.modules["mosaic.fast.trimul_provider"] = op_ok = _op_module(ok=True, calls=calls, xla_below=8)   # N8 served by a kernel row, smaller by F6's body by cell
            rec = F.configure("big")                                        # the big row's word
            self.assertEqual((rec["on"], rec["spec"], rec["word"], rec["body"]), (1, "big", "big", "big")); self.assertIs(T.STATE["body"], F._body)
            rec = F.configure(None)                                           # the default setting = the fast row's word
            self.assertEqual((rec["on"], rec["spec"], rec["word"], rec["body"], rec["probe"], rec["op_version"], rec["kernel_calls"], rec["xla_calls"]),
                             (1, "fast", "fast", "fast", "ok", "vtest", 0, 0))
            self.assertIn("bf16_fwdbwd:rowA@cellA", rec["probe_facts"])                                       # the probe's facts ride the record as one token
            self.assertIs(T.STATE["body"], F._body); self.assertEqual(T.describe()["body"], "fast")           # F6's census names the body by F8's word
            self.assertEqual(F.serves_dtypes(), ("float32", "bfloat16"))                                       # both dtypes: P7 installs no route
            with self.assertRaises(F.Refusal) as cm:                      # on, nothing traced yet
                F.gate()
            self.assertEqual(cm.exception.reason, "nothing_served")
            out = Out()(_Shape((1, 8, 8, 16)), "m"); inn = In()(_Shape((2, 3, 8, 8, 16)), "m")     # joltz's calls → F6's served call → the op
            self.assertEqual((out, inn), (("op-out", "outgoing"), ("op-out", "incoming")))
            self.assertEqual([c for c in calls if c[0] == "op"], [("op", "outgoing", (1, 8, 8, 16), "fast"), ("op", "incoming", (2, 3, 8, 8, 16), "fast")])
            d = F.describe(); self.assertEqual((d["served"], d["fallback"], d["op_calls"], d["kernel_calls"], d["xla_calls"]), (2, 0, 2, 2, 0)); self.assertIn("incoming:B6xN8xC16", d["shapes"])
            self.assertEqual((d["rows"], d["cells"]), ("f32:rowA/rowB:2", "cellA"))
            F.gate()
            line = F.emit_line("t")
            for tok in ("name=F8.trimul_fused state=on", "served=2", "body=fast", "word=fast", "kernel_calls=2", "xla_calls=0", "rows=f32:rowA/rowB:2", "cells=cellA", "op_version=vtest"):
                self.assertIn(tok, line)
            for gone in ("minpx=", "backward=", "bf16=", "bf16_kernel_calls=", "xla_small_calls=", "precision="):   # the retired words
                self.assertNotIn(gone, line)
            Out()(_Shape((1, 4, 4, 16)), "m")                                 # a call the provider's cell hands to F6's body: served, counted under xla — never silent
            d = F.describe(); self.assertEqual((d["served"], d["op_calls"], d["kernel_calls"], d["xla_calls"]), (3, 3, 2, 1)); F.gate()
            op_ok.census = lambda: {"kernel_calls": 1, "xla_calls": 1, "calls": 3, "words": {}, "rows": {}, "cells": [], "traces": {}}   # a third, unnamed body: the gate fails closed
            with self.assertRaises(F.Refusal) as cm:
                F.gate()
            self.assertEqual(cm.exception.reason, "op_census_open")
            rec = F.configure("stock")
            self.assertEqual((rec["on"], rec["body"], rec["word"]), (0, "xla", "none")); self.assertIsNone(T.STATE["body"]); self.assertFalse(op_ok.settings()["on"])   # off resets the op
            F.configure(None); self.assertIs(T.STATE["body"], F._body)
            F.uninstall(); self.assertIsNone(T.STATE["body"]); self.assertFalse(F.installed()); self.assertFalse(op_ok.settings()["on"])
            T.uninstall()
            sys.modules.pop("mosaic.fast.trimul_layout", None); sys.modules.pop("mosaic.fast.trimul_provider", None)



F6_NUMERICS_CHILD = r'''
import itertools, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import joltz
from joltz import TriangleMultiplicationOutgoing as TO, TriangleMultiplicationIncoming as TI
from joltz.backend import Linear, LayerNorm
from mosaic.fast import trimul_layout as T
stock = {"outgoing": joltz.TriangleMultiplicationOutgoing.__call__, "incoming": joltz.TriangleMultiplicationIncoming.__call__}
rng = np.random.default_rng(0)
def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
def lin(o, i): return Linear(weight=jnp.asarray(rng.normal(0, i ** -0.5, (o, i)), jnp.float32), bias=jnp.asarray(rng.normal(0, .1, (o,)), jnp.float32))
def ln(c): return LayerNorm(weight=jnp.asarray(1 + .1 * rng.normal(size=(c,)), jnp.float32), bias=jnp.asarray(.1 * rng.normal(size=(c,)), jnp.float32), eps=1e-5)
def mod(cls, c): return cls(norm_in=ln(c), p_in=lin(2 * c, c), g_in=lin(2 * c, c), norm_out=ln(c), p_out=lin(c, c), g_out=lin(c, c))
worst = 0.0; n = 0
for (cls, d), (B, N, C), mk in itertools.product([(TO, "outgoing"), (TI, "incoming")], [(1, 7, 8), (2, 11, 16), (1, 20, 32)], ["ones", "random", "padrows"]):
    m = mod(cls, C)
    x = jnp.asarray(rng.normal(0, 2, (B, N, N, C)) + rng.normal(0, 3, (1, 1, 1, C)), jnp.float32)
    if mk == "ones": mask = jnp.ones((B, N, N), bool)
    elif mk == "random": mask = jnp.asarray(rng.random((B, N, N)) > .3)
    else:
        live = np.ones((N,), bool); live[-3:] = False; mask = jnp.asarray(np.broadcast_to(live[:, None] & live[None, :], (B, N, N)))
    cot = jnp.asarray(rng.normal(size=(B, N, N, C)), jnp.float32)
    f_ref = lambda x, m: jnp.sum(stock[d](m, x, mask) * cot)
    f_got = lambda x, m: jnp.sum(T.trimul_cmajor(m, x, mask, d) * cot)
    y_ref, y_got = stock[d](m, x, mask), T.trimul_cmajor(m, x, mask, d)
    assert y_got.shape == y_ref.shape and bool(jnp.isfinite(y_got).all()), (d, B, N, C, mk)
    g_ref = jax.grad(f_ref, argnums=(0, 1))(x, m); g_got = jax.grad(f_got, argnums=(0, 1))(x, m)
    r = max([rel(y_got, y_ref), rel(g_got[0], g_ref[0])] + [rel(a, b) for a, b in zip(jax.tree_util.tree_leaves(g_got[1]), jax.tree_util.tree_leaves(g_ref[1]))])
    # 2e-4: XLA-CPU codegen differs by CPU generation (ISA / vector width) for the same jaxlib and the same trimul_cmajor bytes — case
    # (outgoing, B1, N20, C32, random mask) reads rel 1.066e-4 on AMD hosts (aws/gcp) and ~4e-7 on others; f32 rounding of a re-associated
    # contraction, not an arithmetic change (GPU runs give the lever's numerics class)
    assert r < 2e-4, (d, B, N, C, mk, r, "tolerance 2e-4 rel: XLA-CPU codegen varies by CPU generation; observed max 1.066e-4 (outgoing,1,20,32,random) on AMD hosts")
    worst = max(worst, r); n += 1
T.install(); T.configure(None)
m = mod(TO, 16); x = jnp.asarray(rng.normal(size=(1, 9, 9, 16)), jnp.float32); mask = jnp.ones((1, 9, 9), bool)
assert rel(joltz.TriangleMultiplicationOutgoing.__call__(m, x, mask), stock["outgoing"](m, x, mask)) < 2e-4, "tolerance 2e-4 rel (XLA-CPU codegen varies by CPU generation; observed max 1.066e-4)"
d = T.describe(); assert d["served"] == 1 and d["shapes"] == "outgoing:B1xN9xC16:1" and d["norm_out"] == "channel_axis0", d
T.gate(); line = T.emit_line("t")
assert "name=F6.trimul_cmajor state=on" in line and "served=1" in line, line
T.configure("stock"); assert bool((joltz.TriangleMultiplicationOutgoing.__call__(m, x, mask) == stock["outgoing"](m, x, mask)).all())
T.uninstall(); assert joltz.TriangleMultiplicationOutgoing.__call__ is stock["outgoing"]
print(json.dumps({"cases": n, "worst_rel": worst}))
'''


@unittest.skipUnless(_stubs.tree_present() and _importable("jax", "equinox", "einops", "joltz", "opt_core"), "needs jax + equinox + joltz + opt_core importable (the kit venv)")
class TestChannelMajorNumericsCPU(unittest.TestCase):

    def test_served_arithmetic_equals_stock_bodies(self):
        """Runs in a child interpreter (the real jax/joltz stack must not leak into this session's stand-ins): values and gradients of the
        channel-major body vs joltz's stock bodies, both directions × batch/odd N/width × masks (incl. padded rows) × norm_out form."""
        r = subprocess.run([sys.executable, "-c", F6_NUMERICS_CHILD], capture_output=True, text=True, timeout=1200)
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn('"cases": 18', r.stdout)


if __name__ == "__main__":
    unittest.main()
