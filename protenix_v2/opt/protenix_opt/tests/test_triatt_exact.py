"""triatt_exact — the pair stacks' triangle attention bound to the shared core's provider opt_core.kernels.triattn by the tier word `exact`
(opt/forward/flashpairformer/src/fpf/triatt_exact.py): the registry row (EXACT, marker probe, extra + row_dependent) and the report rows, membership
(MODES exact only; the exact composition of the cc-9.0 and cc-8.0 README rows exports its switch, no fast / big / cc-10.x row does, env.sh never does), the
marker grammar the kit classifies by, the binding itself on stub sites with the real provider on CPU tensors (switch off: the sites are untouched; on: every
call reaches T.triangle_attention with word='exact' and the site's own callable as stock, bytes unchanged; a refusal by name takes the site's callable, counted;
a call outside the convention passes through by declared route; unapply puts the callables back; a core below the floor raises by name), the hook in
ptx_trunk2_levers, the LEVER line pairs off a record and the member's evidence line."""
import importlib
import math
import os
import re
import sys
import types
import unittest
from unittest import mock

from protenix_opt import kits, modes, registry, report, stack, tp
from protenix_opt.registry import LEVERS
from opt_core.kernels import triattn as T                      # after protenix_opt (it places the core); before any test patches sys.modules: one provider module object for the tests and the binding
from opt_core.kernels.triattn import exact_member as EM

LEVER = "triatt_exact"
SWITCH = "PTX_TRIATT_EXACT"
MOD = "fpf.triatt_exact"
FPF = kits.kit_dir("flashpairformer")
SRC = os.path.join(FPF, "src")
TL_NAME = "protenix.model.triangular.layers"
ON = ("TRIATT_EXACT:on(opt_core.kernels.triattn 0.5.225.0 word=exact bind=tier:exact sites=tl+pad8 cc=9.0 select=S256:cueq,S512:cueq,S1536:cueq; "
      "no cell vouches an exact-class row on 9.0|torch2.13.0+cu130|cueq0.11.1: every call takes the library op BY NAME through the provider; "
      "EXACT-BITWISE vs the library triangle_attention D=32)")
MEMBER_LINE = re.compile(r"^triattn_exact: served \d+/\d+ calls \(refused: \{.*\}\)$")


def _module():
    """The binding module imported fresh from the kit's src (its record is per process: unapply() in the tests' cleanups resets it)."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    sys.modules.pop(MOD, None)
    return importlib.import_module(MOD)


def _torch(tc):
    try:
        import torch
    except ImportError:
        tc.skipTest("torch absent")
    return torch


def _reference(torch):
    """A CPU stand-in for the library op in its own convention: softmax_j(scale q.k + bias, masked) @ v, in q's dtype; like the library it returns
    [B,N,H,S,D] for 4-D ([N,H,S,D]) and 5-D operands alike (B=1 added for the 4-D call)."""
    def ref(q, k, v, bias, mask, scale):
        s = torch.einsum("...hid,...hjd->...hij", q.float() * float(scale), k.float()) + bias.float()
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        out = (s.softmax(-1) @ v.float()).to(q.dtype)
        return out.unsqueeze(0) if q.dim() == 4 else out
    return ref


class _Sites:
    """Stub engine + pad8 modules in sys.modules: protenix.model.triangular.layers.cuequivariance_triangular_attn (positional signature, as stock) and
    fpf_cueq_pad8exact._cue_tri (the library's keyword signature); every call is logged with the site name and operand shapes."""
    def __init__(self, torch):
        self.log = []
        ref = _reference(torch)

        def cuequivariance_triangular_attn(q, k, v, bias, mask, scale):
            self.log.append(("tl", tuple(q.shape), None if mask is None else tuple(mask.shape)))
            return ref(q, k, v, bias, mask, scale)

        def _cue_tri(q, k, v, bias, mask=None, scale=None, return_aux=False):
            self.log.append(("pad8", tuple(q.shape), None if mask is None else tuple(mask.shape)))
            return ref(q, k, v, bias, mask, 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale)

        self.tl_fn, self.pad8_fn = cuequivariance_triangular_attn, _cue_tri
        pkg = types.ModuleType("protenix"); pkg.__path__ = []
        model = types.ModuleType("protenix.model"); model.__path__ = []
        tri = types.ModuleType("protenix.model.triangular"); tri.__path__ = []
        self.TL = types.ModuleType(TL_NAME); self.TL.cuequivariance_triangular_attn = cuequivariance_triangular_attn
        self.P8 = types.ModuleType("fpf_cueq_pad8exact"); self.P8._cue_tri = _cue_tri
        self.modules = {"protenix": pkg, "protenix.model": model, "protenix.model.triangular": tri, TL_NAME: self.TL, "fpf_cueq_pad8exact": self.P8}


class TestRows(unittest.TestCase):
    def test_registry_and_report_rows(self):
        lv = LEVERS[LEVER]
        self.assertEqual((lv.cls, lv.tier, lv.env_keys, lv.probe, lv.conditional, lv.extra, lv.row_dependent, lv.replaced_by, lv.words),
                         (registry.FORWARD, registry.EXACT, (SWITCH,), "marker", (), True, True, (), ()))
        self.assertIn("kit README", lv.source); self.assertIn("src/fpf/triatt_exact.py", lv.source); self.assertIn("opt_core.kernels.triattn", lv.description)
        self.assertTrue(os.path.isfile(os.path.join(SRC, *MOD.split(".")) + ".py"))
        self.assertEqual(report.STRATEGY_IDS[LEVER], "LOCAL.protenix_v2.triatt_exact")
        self.assertEqual(report.IMPL[LEVER], ("opt_core.kernels.triattn", "core"))
        self.assertEqual(stack.MARKERS[LEVER], ("TRIATT_EXACT:", ("TRIATT_EXACT:on(",)))
        self.assertEqual(tp.TALLY[LEVER], ("fpf", (("triatt_exact", "calls"),)))
        self.assertEqual(list(LEVERS).index(LEVER), list(LEVERS).index("dit_attn_exact") + 1, "the LEVER lines keep registry order: right after dit_attn_exact")

    def test_membership_and_rows(self):
        self.assertIn(LEVER, modes.MODES["exact"]); self.assertNotIn(LEVER, modes.MODES["fast"]); self.assertNotIn(LEVER, modes.MODES["big"]); self.assertNotIn(LEVER, modes.big_levers())
        self.assertEqual(modes.MODES["exact"].index(LEVER), modes.MODES["exact"].index("dit_attn_exact") + 1)
        for key, row in modes.README_ROWS.items():
            listed = key.split("|")[0] in ("9.0", "8.0")
            self.assertEqual(row["exact"]["post"].get(SWITCH), "1" if listed else None, key)          # the cc-9.0 and cc-8.0 rows' exact composition, no cc-10.x row
            self.assertNotIn(SWITCH, row["exact"]["pre"], key); self.assertNotIn(SWITCH, row["fast"]["pre"], key); self.assertNotIn(SWITCH, row["fast"]["post"], key)
            self.assertEqual(registry.in_row(LEVERS[LEVER], row["exact"]), listed, key); self.assertFalse(registry.in_row(LEVERS[LEVER], row["fast"]), key)
        self.assertNotIn(SWITCH, modes.OTHER_ROW["exact"]["post"]); self.assertFalse(registry.in_row(LEVERS[LEVER], modes.OTHER_ROW["exact"]))

    def test_resolve_exports_the_switch_under_exact_only(self):
        base = {"PATH": os.environ.get("PATH", "")}
        for cc, triton, row_key in (("9.0", "3.3.1", "9.0|3.3"), ("9.0", "3.7.1", "9.0|3.7"), ("9.0", "3.5.0", "9.0|*"), ("8.0", "3.7.1", "8.0|3.7"), ("8.0", "3.3.1", "8.0|*")):
            r = modes.resolve("exact", dict(base), stack.kit_home(), compute_cap=cc, triton=triton, probe_gpu=False)
            self.assertEqual((r.row_key, r.exports.get(SWITCH), r.extras.get(SWITCH)), (row_key, "1", "1"), (cc, triton))       # an extra: set by resolve() after env.sh, never by env.sh
            for mode in ("fast", "big"):
                self.assertNotIn(SWITCH, modes.resolve(mode, dict(base), stack.kit_home(), compute_cap=cc, triton=triton, probe_gpu=False).exports, (cc, triton, mode))
        for cc in ("10.0", "10.3", "8.9"):
            for mode in ("exact", "fast", "big"):
                self.assertNotIn(SWITCH, modes.resolve(mode, dict(base), stack.kit_home(), compute_cap=cc, triton="3.7.1", probe_gpu=False).exports, (cc, mode))
        env_sh = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
        self.assertNotIn(SWITCH, env_sh, "a README row switch: resolve() exports it around env.sh, env.sh never names it")

    def test_marker_grammar(self):
        family, good = stack.MARKERS[LEVER]
        self.assertTrue(ON.startswith(family)); self.assertTrue(any(g in ON for g in good)); self.assertFalse(any(b in ON for b in stack.BAD))
        env = modes.resolve("exact", {"PATH": os.environ.get("PATH", "")}, stack.kit_home(), compute_cap="9.0", triton="3.7.1", probe_gpu=False).exports
        row = modes.readme_row("9.0|3.7", "exact")
        on, fb, skipped, why = stack._classify("exact", [ON], env, row=row, row_key="9.0|3.7")
        self.assertIn(LEVER, on); self.assertNotIn(LEVER, fb); self.assertNotIn(LEVER, skipped)
        bad = "TRIATT_EXACT:unavailable(RuntimeError('triatt_exact: opt_core 0.5.220.3 is older than 0.5.225.0'))"
        on2, fb2, _s, why2 = stack._classify("exact", [bad], env, row=row, row_key="9.0|3.7")
        self.assertIn(LEVER, fb2); self.assertNotIn(LEVER, on2); self.assertEqual(why2[LEVER], bad)
        _o, fb3, _s3, why3 = stack._classify("exact", [], env, row=row, row_key="9.0|3.7")
        self.assertIn(LEVER, fb3); self.assertEqual(why3[LEVER], "no marker in the kit's applied list")
        on4, fb4, sk4, why4 = stack._classify("exact", [ON], env, row=modes.OTHER_ROW["exact"], row_key="other")                 # a card whose row does not list it: not in the arm, never a fallback
        self.assertIn(LEVER, sk4); self.assertNotIn(LEVER, fb4); self.assertIn(stack.NOT_IN_ROW, why4[LEVER])


class TestBinding(unittest.TestCase):
    """The binding on stub sites; the provider is the real opt_core.kernels.triattn deciding over CPU tensors (no cell vouches the kernel row here: the
    stock op serves every call) with a spy in front of its face recording the word and the stock callable."""

    def setUp(self):
        self.torch = _torch(self)
        self.sites = _Sites(self.torch)
        patcher = mock.patch.dict(sys.modules, self.sites.modules)
        patcher.start(); self.addCleanup(patcher.stop)
        self.X = _module()
        self.addCleanup(self.X.unapply)
        self.T = T
        self.seen = []
        real = T.triangle_attention

        def spy(q, k, v, bias, mask=None, scale=None, **kw):
            self.seen.append({"q": tuple(q.shape), "bias": tuple(bias.shape), "mask": None if mask is None else tuple(mask.shape), "word": kw.get("word"), "stock": kw.get("stock"),
                              "prefer": kw.get("prefer")})
            return real(q, k, v, bias, mask, scale, **kw)
        sp = mock.patch.object(T, "triangle_attention", spy)
        sp.start(); self.addCleanup(sp.stop)

    def _operands(self, N=6, H=8, S=24, D=32):
        torch = self.torch
        g = torch.Generator().manual_seed(7)
        q = torch.randn(N, H, S, D, generator=g).to(torch.bfloat16); k = torch.randn(N, H, S, D, generator=g).to(torch.bfloat16); v = torch.randn(N, H, S, D, generator=g).to(torch.bfloat16)
        bias = torch.randn(1, H, S, S, generator=g); mask = torch.rand(N, 1, 1, S, generator=g) > 0.25
        return q, k, v, bias, mask, 1.0 / math.sqrt(D)

    def test_switch_off_touches_nothing(self):
        X, TL, P8 = self.X, self.sites.TL, self.sites.P8
        for environ in ({}, {SWITCH: ""}, {SWITCH: "0"}):
            self.assertIsNone(X.apply(environ), environ)
            self.assertIs(TL.cuequivariance_triangular_attn, self.sites.tl_fn); self.assertIs(P8._cue_tri, self.sites.pad8_fn)
            self.assertFalse(X.report()["installed"]); self.assertEqual(X.report()["sites"], [])
        with self.assertRaises(ValueError) as cm:
            X.apply({SWITCH: "2"})
        self.assertIn(SWITCH, str(cm.exception)); self.assertIs(TL.cuequivariance_triangular_attn, self.sites.tl_fn)
        self.assertFalse(X.enabled({})); self.assertTrue(X.enabled({SWITCH: "1"}))

    def test_on_every_call_reaches_the_provider_by_the_word_exact_with_the_sites_callable_as_stock(self):
        torch, X, TL, P8 = self.torch, self.X, self.sites.TL, self.sites.P8
        marker = X.apply({SWITCH: "1"})
        self.assertEqual(marker, X.apply({SWITCH: "1"}), "a second apply returns the same marker and binds nothing twice")
        self.assertRegex(marker, r"^TRIATT_EXACT:on\(opt_core\.kernels\.triattn [0-9.]+ word=exact bind=tier:exact sites=tl\+pad8 cc=(none|\d+\.\d+) select=\S+; .+; EXACT-BITWISE vs the library triangle_attention D=32\)$")
        self.assertFalse(any(b in marker for b in stack.BAD))
        self.assertIsNot(TL.cuequivariance_triangular_attn, self.sites.tl_fn); self.assertTrue(TL.cuequivariance_triangular_attn._ptx_triatt_exact)
        self.assertIsNot(P8._cue_tri, self.sites.pad8_fn); self.assertTrue(P8._cue_tri._ptx_triatt_exact)
        q, k, v, bias, mask, scale = self._operands()
        ref_tl, ref_p8 = self.sites.tl_fn, self.sites.pad8_fn
        # the statement's layout ([N,H,S,D] / [1,H,S,S] / [N,1,1,S]), the stock module's ([1,N,H,S,D] / [1,1,H,S,S] / [1,N,1,1,S]), the NOMASK form, the pad8 handle
        out4 = TL.cuequivariance_triangular_attn(q, k, v, bias, mask, scale)
        out5 = TL.cuequivariance_triangular_attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.unsqueeze(0), mask.unsqueeze(0), scale)
        outn = TL.cuequivariance_triangular_attn(q, k, v, bias, None, scale)
        outp = P8._cue_tri(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.unsqueeze(0), mask=None, scale=scale)
        self.assertEqual((tuple(out4.shape), tuple(out5.shape), tuple(outn.shape), tuple(outp.shape)), ((1, 6, 8, 24, 32), (1, 6, 8, 24, 32), (1, 6, 8, 24, 32), (1, 6, 8, 24, 32)), "the library's rank comes back: [1,N,H,S,D] for 4-D and 5-D statement calls alike")
        self.assertEqual([s for s, *_ in self.sites.log], ["tl", "tl", "tl", "pad8"], "every served call ran the site's own callable (no cell vouches the kernel row on this interpreter)")
        self.assertTrue(torch.equal(out4, ref_tl(q, k, v, bias, mask, scale))); self.assertTrue(torch.equal(out5, out4)); self.assertTrue(torch.equal(outn, ref_tl(q, k, v, bias, None, scale)))
        self.assertTrue(torch.equal(outp, ref_p8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.unsqueeze(0), mask=None, scale=scale)))
        self.assertEqual(len(self.seen), 4)
        for call in self.seen:                                                                     # the provider convention, the tier word, no row preference
            self.assertEqual((call["q"], call["bias"], call["word"], call["prefer"]), ((1, 6, 8, 24, 32), (1, 1, 8, 24, 24), "exact", None))
            self.assertIn(call["mask"], (None, (1, 6, 1, 1, 24)))
        self.assertEqual([c["mask"] for c in self.seen], [(1, 6, 1, 1, 24), (1, 6, 1, 1, 24), None, None])
        # stock = the site's own callable in the provider convention: calling it lands on the callable that was found at the site
        n = len(self.sites.log)
        q5, k5, v5, b5 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.unsqueeze(0)
        self.seen[0]["stock"](q5, k5, v5, b5, mask=None, scale=scale); self.seen[3]["stock"](q5, k5, v5, b5, mask=None, scale=scale)
        self.assertEqual([s for s, *_ in self.sites.log[n:]], ["tl", "pad8"])
        r = X.report()
        self.assertEqual((r["unit"], r["installed"], r["word"], r["bind"], r["sites"], r["calls"], r["provider"], r["refused"], r["passthrough"], r["site_calls"]),
                         (LEVER, True, "exact", "tier:exact", ["tl", "pad8"], 4, 4, {}, {}, {"tl": 3, "pad8": 1}))
        self.assertRegex(r["opt_core"], r"^\d+\.\d+\.\d+"); self.assertEqual(r["marker"], marker)
        self.assertRegex(r["member_line"], MEMBER_LINE); self.assertEqual(set(r["member"]), {"served", "calls", "refused", "installed"})
        self.assertTrue(X.exit_line().startswith("[FPF] triatt_exact: triattn_exact: served "), X.exit_line())
        for key in ("sites=tl+pad8", "calls=4", "provider=4", "refused=none", "passthrough=none", "opt_core="):
            self.assertIn(key, X.exit_line())

    def test_a_refusal_by_name_takes_the_sites_callable_counted(self):
        torch, X, TL, T = self.torch, self.X, self.sites.TL, self.T
        X.apply({SWITCH: "1"})
        q, k, v, bias, mask, scale = self._operands()

        def refuse(*a, **kw):
            raise T.Refusal("exact_vouch_not_recorded:9.0|torch|cueq", "triattn_exact", "cueq", "no vouch key for this stack")
        with mock.patch.object(T, "triangle_attention", refuse):
            X.unapply(); X.apply({SWITCH: "1"})                                                    # bind again so the wrapper closes over the refusing face
            out = TL.cuequivariance_triangular_attn(q, k, v, bias, mask, scale)
            out2 = TL.cuequivariance_triangular_attn(q, k, v, bias, None, scale)
        self.assertEqual([e[0] for e in self.sites.log], ["tl", "tl"], "both calls ran the site's own callable")
        self.assertEqual(self.sites.log[0][1:], ((1, 6, 8, 24, 32), (1, 6, 1, 1, 24)), "in the provider convention (the refusal came after the layout step)")
        self.assertTrue(torch.equal(out, self.sites.tl_fn(q, k, v, bias, mask, scale))); self.assertTrue(torch.equal(out2, self.sites.tl_fn(q, k, v, bias, None, scale)))
        self.assertEqual(tuple(out.shape), (1, 6, 8, 24, 32), "the library's rank comes back on the refused route too")
        r = X.report()
        self.assertEqual((r["calls"], r["provider"], r["refused"], r["passthrough"]), (2, 2, {"exact_vouch_not_recorded": 2}, {}))

    def test_calls_outside_the_convention_pass_through_by_declared_route(self):
        torch, X, TL, P8 = self.torch, self.X, self.sites.TL, self.sites.P8
        X.apply({SWITCH: "1"})
        q, k, v, bias, mask, scale = self._operands()
        out = TL.cuequivariance_triangular_attn(q[:, :4], k[:, :4], v[:, :4], bias[:, :4], None, scale)   # H4 operands with an H4 bias: the convention holds, the face is asked
        self.assertEqual(tuple(out.shape), (1, 6, 4, 24, 32)); self.assertEqual(len(self.seen), 1); self.assertEqual(self.seen[0]["q"], (1, 6, 4, 24, 32))
        out3 = TL.cuequivariance_triangular_attn(q[0], k[0], v[0], bias[0], None, scale)             # rank-3 operands: not the convention -> the site's callable as called
        self.assertTrue(torch.equal(out3, self.sites.tl_fn(q[0], k[0], v[0], bias[0], None, scale))); self.assertEqual(self.sites.log[1][1], (8, 24, 32))
        out2 = TL.cuequivariance_triangular_attn(q, k, v, bias[0, 0], None, scale)                  # a rank-2 bias
        self.assertTrue(torch.equal(out2, self.sites.tl_fn(q, k, v, bias[0, 0], None, scale)))
        q5, k5, v5, b5 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.unsqueeze(0)
        P8._cue_tri(q5, k5, v5, b5, scale=scale, return_aux=False)                                    # an argument beyond the six operands
        self.assertEqual(self.sites.log[-1][:2], ("pad8", (1, 6, 8, 24, 32)))
        self.assertEqual(len(self.seen), 1, "the face was asked once: the three other calls passed through")
        r = X.report()
        self.assertEqual((r["calls"], r["provider"], r["passthrough"], r["refused"]), (4, 1, {"rank": 1, "bias_rank": 1, "arguments": 1}, {}))

    def test_unapply_puts_the_callables_back(self):
        X, TL, P8 = self.X, self.sites.TL, self.sites.P8
        X.apply({SWITCH: "1"})
        self.assertTrue(X.unapply())
        self.assertIs(TL.cuequivariance_triangular_attn, self.sites.tl_fn); self.assertIs(P8._cue_tri, self.sites.pad8_fn)
        self.assertFalse(X.report()["installed"]); self.assertEqual(X.report()["calls"], 0)
        self.assertFalse(X.unapply(), "nothing bound: nothing restored")
        self.assertTrue(X.apply({SWITCH: "1"}).startswith("TRIATT_EXACT:on(")); self.assertTrue(TL.cuequivariance_triangular_attn._ptx_triatt_exact)

    def test_pad8_is_bound_only_when_loaded(self):
        X, TL = self.X, self.sites.TL
        with mock.patch.dict(sys.modules, {}):
            sys.modules.pop("fpf_cueq_pad8exact", None)
            marker = X.apply({SWITCH: "1"})
        self.assertIn("sites=tl ", marker); self.assertEqual(X.report()["sites"], ["tl"]); self.assertIs(self.sites.P8._cue_tri, self.sites.pad8_fn)

    def test_a_core_below_the_floor_raises_by_name_and_binds_nothing(self):
        X, TL = self.X, self.sites.TL
        import opt_core
        with mock.patch.object(opt_core, "__version__", "0.5.224.9"):
            with self.assertRaises(RuntimeError) as cm:
                X.apply({SWITCH: "1"})
        self.assertIn("triatt_exact: opt_core 0.5.224.9 is older than 0.5.225.0", str(cm.exception))
        self.assertIs(TL.cuequivariance_triangular_attn, self.sites.tl_fn); self.assertFalse(X.report()["installed"])
        self.assertEqual(X.FLOOR, (0, 5, 225, 0))

    def test_the_engine_module_absent_raises_by_name(self):
        X = self.X
        with mock.patch.dict(sys.modules, {TL_NAME: None}):                                       # an interpreter without the engine: ImportError out of apply, nothing bound
            with self.assertRaises(ImportError):
                X.apply({SWITCH: "1"})
        self.assertFalse(X.report()["installed"])


class TestHook(unittest.TestCase):
    def test_trunk_levers_hook_and_record(self):
        src = open(os.path.join(SRC, "ptx_trunk2_levers.py"), encoding="utf-8").read()
        for needle in ('if os.environ.get("PTX_TRIATT_EXACT", "0") not in ("", "0"):', "from fpf import triatt_exact as _tx", '_STATS["applied"].append(_tx.apply())',
                       "_TXA_REPORT = _tx.report", 'f"TRIATT_EXACT:unavailable({_e!r})"', '[FPF] triatt_exact: ', '_STATS["triatt_exact"] = _TXA_REPORT()'):
            self.assertIn(needle, src, needle)
        mine = src.index('if os.environ.get("PTX_TRIATT_EXACT", "0")')
        self.assertLess(src.index('if os.environ.get("PTX_DIT_ATTN_EXACT", "0")'), mine, "bound after dit_attn_exact")
        self.assertLess(src.index('if os.environ.get("PTX_E_PAD8", "")'), mine, "bound after the block levers: a loaded pad8 provider is bound too")

    def test_module_constants(self):
        X = _module()
        self.assertEqual((X.NAME, X.SWITCH, X.WORD, X.FACE, X.MARK, X.ROW, X.HEAD_DIM, X.PROBE_SIZES), (LEVER, SWITCH, "exact", "opt_core.kernels.triattn", "TRIATT_EXACT:", "triattn_exact", 32, (256, 512, 1536)))
        self.assertEqual((X.TL_MODULE, X.TL_NAME, X.PAD8_MODULE, X.PAD8_NAME), (TL_NAME, "cuequivariance_triangular_attn", "fpf_cueq_pad8exact", "_cue_tri"))
        self.assertEqual(stack.MARKERS[LEVER][0], X.MARK)


class TestEvidence(unittest.TestCase):
    REC = {"installed": True, "word": "exact", "bind": "tier:exact", "sites": ["tl", "pad8"], "calls": 12, "provider": 11, "refused": {"exact_vouch_not_recorded": 2},
           "passthrough": {"arguments": 1}, "member": {"served": 7, "calls": 9, "refused": {"small_s": 2}, "installed": True},
           "select": {"256": "cueq", "512": "triattn_exact", "1536": "triattn_exact"}, "exact_stack": "9.0|torch2.13.0+cu130|cueq0.11.1", "opt_core": "0.5.225.0"}
    LINE = ("[protenix-opt] LEVER name=LOCAL.protenix_v2.triatt_exact state=on impl=opt_core.kernels.triattn origin=core strategy=LOCAL.protenix_v2.triatt_exact installed=1 word=exact "
            "bind=tier:exact sites=tl+pad8 calls=12 provider=11 refused=exact_vouch_not_recorded:2 passthrough=arguments:1 kernel=7 kernel_refused=small_s:2 "
            "select=S256:cueq,S512:triattn_exact,S1536:triattn_exact exact_stack=9.0|torch2.13.0+cu130|cueq0.11.1 opt_core=0.5.225.0 mode=exact lever=triatt_exact")

    def _rep(self, **over):
        rep = {"active": True, "mode": "exact", "levers_applied": list(modes.MODES["exact"]), "levers_fallback": [], "levers_not_in_arm": [], "fallback_reasons": {}, "fpf": {"triatt_exact": dict(self.REC)}}
        rep.update(over); return rep

    def test_lever_line_off_the_trunk_record(self):
        self.assertEqual(report.triatt_exact_evidence(self._rep()), [("installed", 1), ("word", "exact"), ("bind", "tier:exact"), ("sites", "tl+pad8"), ("calls", 12), ("provider", 11),
                                                                     ("refused", "exact_vouch_not_recorded:2"), ("passthrough", "arguments:1"), ("kernel", 7), ("kernel_refused", "small_s:2"),
                                                                     ("select", "S256:cueq,S512:triattn_exact,S1536:triattn_exact"), ("exact_stack", "9.0|torch2.13.0+cu130|cueq0.11.1"), ("opt_core", "0.5.225.0")])
        line = [l for l in report.lever_lines(self._rep()) if l.endswith(" lever=" + LEVER)]
        self.assertEqual(line, [self.LINE])
        empty = dict(self.REC, refused={}, passthrough={}, member=None, select={}, exact_stack=None, sites=["tl"])
        pairs = dict(report.triatt_exact_evidence(self._rep(fpf={"triatt_exact": empty})))
        self.assertEqual((pairs["refused"], pairs["passthrough"], pairs["kernel"], pairs["kernel_refused"], pairs["select"], pairs["exact_stack"], pairs["sites"]), ("none", "none", 0, "none", "none", "none", "tl"))
        with mock.patch.dict(sys.modules, {}):
            sys.modules.pop(MOD, None)
            self.assertEqual(report.triatt_exact_evidence({}), [("unit", "none")], "no record, no module: unit=none")
            self.assertEqual(report.triatt_exact_evidence(self._rep(fpf={"triatt_exact": dict(self.REC, installed=False)})), [("unit", "none")])
            off = [l for l in report.lever_lines(dict(self._rep(fpf={}), mode="fast", levers_applied=list(modes.MODES["fast"]))) if l.endswith(" lever=" + LEVER)]
        self.assertEqual(off, ["[protenix-opt] LEVER name=LOCAL.protenix_v2.triatt_exact state=off reason=not_in_mode:fast impl=opt_core.kernels.triattn origin=core strategy=LOCAL.protenix_v2.triatt_exact mode=fast lever=triatt_exact"])

    def test_live_module_record_when_the_report_lacks_it(self):
        X = _module()
        fake = types.ModuleType(MOD); fake.report = lambda: dict(self.REC, calls=3, provider=3)
        with mock.patch.dict(sys.modules, {MOD: fake}):
            pairs = dict(report.triatt_exact_evidence({"fpf": {}}))
        self.assertEqual((pairs["calls"], pairs["provider"], pairs["installed"]), (3, 3, 1))
        broken = types.ModuleType(MOD); broken.report = lambda: (_ for _ in ()).throw(RuntimeError("x y"))
        with mock.patch.dict(sys.modules, {MOD: broken}):
            self.assertEqual(report.triatt_exact_evidence(None)[0], ("unit", "report_error"))
        self.assertEqual(X.report()["unit"], LEVER)

    def test_the_members_evidence_line_is_reachable(self):
        from opt_core.kernels.triattn import exact_member as EM
        self.assertRegex(EM.evidence_line(), MEMBER_LINE)
        c = EM.counts()
        self.assertEqual({"served", "refused", "calls", "installed"} - set(c), set())
        X = _module()
        m = X.member_counts()
        self.assertRegex(m["line"], MEMBER_LINE); self.assertEqual((m["served"], m["calls"]), (int(c["served"]), int(c["calls"])))
        self.assertIsNone(X.report()["member_line"], "not installed: no member line in the record"); self.assertIn("triattn_exact: no member record", X.exit_line())


if __name__ == "__main__":
    unittest.main()
