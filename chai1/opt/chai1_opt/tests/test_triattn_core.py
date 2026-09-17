"""Pair-track lever `triattn` (chai1_opt.triattn_core): Chai-1's triangle attention on the shared core's ONE provider (opt_core.kernels.triattn) BY THE
MODE'S TIER WORD at every crop, head dim and card — no kit cell table.  CPU tests: the words (tier -> provider word; the call form; the position
stride; the launch-word format), the per-call-class decision against the LIVE provider table as CLASS CONTRACTS (some servable fast-class row the
provider admits on that card for Chai-1's strided bias_only call, decided by a provider cell of that shape, whatever row and launch cell the table
names today — never the shared core's current row by name; a flipped table still satisfies them), the declared step-asides (the provider naming the stock
op; a card without cells) vs the undeclared refusal, the exact tier's rule on this engine (statement by name: the provider's exact rows are vouched
against another reference op), the cfg_fn wrapping and the counters with stand-ins (no GPU), and the registration (pairtrack / registry / modes /
core floor; the kit cell table is gone)."""
import io
import json
import os
import re
import sys
import types
import unittest
from contextlib import redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))


def _core():
    try:
        import opt_core  # noqa: F401
        from opt_core.kernels import triattn  # noqa: F401
    except ImportError:                                                            # pragma: no cover
        raise unittest.SkipTest("opt_core (with kernels.triattn) not importable: PYTHONPATH=../common/opt_core:opt")



def _fb(led):                                                                      # opt_core.counters.Ledger: `fallbacks` is a dict (a method on older cores)
    f = getattr(led, "fallbacks")
    return f() if callable(f) else dict(f)


def _served(led):
    s = getattr(led, "served")
    return s() if callable(s) else int(s)

LAUNCH_RE = r"^(m\d+n\d+R\d+w\d+s\d+(o\d+)?(r\d+)?)?$"                            # launch_word()'s format ('' = the row's own cell)
CROPS = (256, 384, 512, 768, 1024, 1536, 2048)                                     # chai_lab's crops
STACKS = {"9.0": "H100:2.13.0+cu130/3.7.1/nocueq", "8.0": "A100:2.13.0+cu130/3.7.1/nocueq"}   # the release stack words (cell-style; offline they stand for every built key of the torch build)
UNMEASURED_CANDIDATES = ((8, 9), (12, 0), (11, 0), (8, 6), (13, 0), (9, 5))              # cards tried, in order, for "a cc WITHOUT a measured column" — resolved against the provider's
                                                                                             #  table at test time, so the shared core adding a measured column (10.0, 10.3, …) never reds these tests


def _measured_columns(T):
    """The cc columns the provider's table carries measured cells for ('8.0', '9.0', '10.0', …)."""
    return sorted({k.split("|")[0] for k in T.table().get("cells", {}) if "|" in k})


def _unmeasured_ccs(T):
    cols = set(_measured_columns(T))
    return [cc for cc in UNMEASURED_CANDIDATES if f"{cc[0]}.{cc[1]}" not in cols]


def _served_re(T, dh=64):
    """A coverage word for a SERVED call, built from the provider's own row vocabulary (never a literal row list)."""
    rows = "|".join(re.escape(r) for r in T.rows() if r not in T.STOCK_ROWS)
    return r"^(fast|big) -> (%s)( m\d+n\d+R\d+w\d+s\d+(o\d+)?(r\d+)?)? \(core \d+\.\d+\|bf16\|D%d\|H4\|N<=\d+\|fwd\)$" % (rows, dh)


class Words(unittest.TestCase):
    def setUp(self):
        _core()
        from chai1_opt import triattn_core as tc
        self.tc = tc

    def test_tier_words(self):
        """Every tier asks its own word (core >= 0.5.114 lists exact | fast | big on every face); unknown tiers raise by name."""
        tc = self.tc
        lists_big = types.SimpleNamespace(TIER_WORDS=("exact", "fast", "big")); no_big = types.SimpleNamespace(TIER_WORDS=("exact", "fast"))
        self.assertEqual([tc.word_for(t, lists_big) for t in ("exact", "fast", "big")], ["exact", "fast", "big"])
        self.assertEqual(tc.word_for("big", no_big), "big")                     # the word is the tier; a face without it refuses by name at select
        with self.assertRaises(LookupError):
            tc.word_for("turbo", no_big)
        self.assertEqual(tc.word_for("big"), "big")
        self.assertEqual(tc.FORM, "bias_only"); self.assertEqual(tc.DTYPE_WORD, "bf16"); self.assertEqual(tc.REFERENCE, "sdpa")
        self.assertEqual(tc.position_stride(64, 4), 1024); self.assertEqual(tc.position_stride(32, 4), 512)
        self.assertEqual(tc.cc_word((9, 0)), "9.0"); self.assertEqual(tc.cc_word("8.0"), "8.0")
    def test_launch_word_format(self):
        tc = self.tc
        self.assertEqual(tc.launch_word({"BLOCK_M": 128, "BLOCK_N": 32, "ROWS": 1, "num_warps": 4, "num_stages": 3, "ORDER": 0, "MAXNREG": 168}), "m128n32R1w4s3r168")
        self.assertEqual(tc.launch_word({"BLOCK_M": 64, "BLOCK_N": 64, "ROWS": 2, "num_warps": 4, "num_stages": 2, "ORDER": 1}), "m64n64R2w4s2o1")
        self.assertEqual(tc.launch_word(None), ""); self.assertEqual(tc.launch_word({}), "")

    def test_decide_with_stand_in_selects(self):
        """The decision's words on test-local answers: a stock row -> stock_statement (declared); no_cell / no_row refusals -> no_cell (declared);
        any other refusal -> refused:<kind> (undeclared); the exact tier serves only an exact-class row vouched against THIS engine's reference."""
        tc = self.tc
        from opt_core.kernels import triattn as T
        S = lambda **kw: types.SimpleNamespace(**{"row": "k2b", "cell": "9.0|bf16|D64|H4|N<=800|fwd", "config": None, "cls": "fast", "exact_vs": "cueq", "measured": True, "reason": "test", **kw})
        seen = {}
        def spy(*a, **kw):
            seen["args"] = a; seen.update(kw); return S()
        sel, fb, word = tc.decide((9, 0), 64, 4, 512, "fast", select=spy)
        self.assertEqual((fb, word, sel.row), (None, "fast", "k2b"))
        self.assertEqual(seen["args"], ("9.0", "bf16", 64, 4, 512))
        self.assertEqual((seen["word"], seen["form"], seen["position_stride"], seen["stack"]), ("fast", "bias_only", 1024, None))
        self.assertNotIn("prefer", seen)                                              # no kit preference list: the provider's cell order decides
        for stock in T.STOCK_ROWS:
            self.assertEqual(tc.decide((9, 0), 64, 4, 512, "fast", select=lambda *a, **kw: S(row=stock, cls="stock"))[:2], (None, "stock_statement"))
        def refuse(kind):
            def f(*a, **kw):
                raise T.Refusal(kind, "k2b", "cueq", "test")
            return f
        self.assertEqual(tc.decide((8, 9), 64, 4, 512, "fast", select=refuse("no_cell:bf16_D64"))[1:], ("no_cell", "fast"))
        self.assertEqual(tc.decide((9, 0), 16, 4, 512, "fast", select=refuse("no_row:head_dim16"))[1:], ("no_cell", "fast"))
        self.assertEqual(tc.decide((9, 0), 64, 4, 4096, "fast", select=refuse("int32_offset:k=4194304>=2**31"))[1:], ("refused:int32_offset:k=4194304>=2**31", "fast"))
        self.assertNotIn("refused:int32_offset:k=4194304>=2**31", tc.EXPECTED_FALLBACKS); self.assertIn("no_cell", tc.EXPECTED_FALLBACKS); self.assertIn("stock_statement", tc.EXPECTED_FALLBACKS)
        # the exact tier on this engine
        self.assertEqual(tc.decide((9, 0), 64, 4, 512, "exact", select=lambda *a, **kw: S(row="exact_row", cls="exact", exact_vs="cueq"))[:2], (None, "stock_statement"))
        self.assertEqual(tc.decide((9, 0), 64, 4, 512, "exact", select=lambda *a, **kw: S(row="k2b", cls="fast"))[:2], (None, "stock_statement"))
        sel, fb, word = tc.decide((9, 0), 64, 4, 512, "exact", select=lambda *a, **kw: S(row="exact_native", cls="exact", exact_vs="sdpa"))
        self.assertEqual((fb, word, sel.row), (None, "exact", "exact_native"))          # an exact-class row vouched against SDPA WOULD serve the exact tier


class ProviderContracts(unittest.TestCase):
    """The live provider table read as CLASS CONTRACTS: never the shared core's current row by name."""

    def setUp(self):
        _core()
        from chai1_opt import triattn_core as tc
        from opt_core.kernels import triattn as T
        self.tc, self.T = tc, T

    def _assert_servable(self, sel, cc: str, dh: int, n: int):
        tc, T = self.tc, self.T
        self.assertIsNotNone(sel, (cc, dh, n))
        self.assertNotIn(sel.row, T.STOCK_ROWS, (cc, dh, n, sel.row)); self.assertIn(sel.row, T.ROW_NAMES)
        self.assertEqual(sel.cls, "fast", (cc, dh, n)); self.assertEqual(T.rows()[sel.row]["class"], "fast", sel.row)
        ok, why = T.admits(sel.row, cc, "bf16", dh, 4, n, position_stride=tc.position_stride(dh, 4))
        self.assertTrue(ok, f"{cc} D{dh} N={n} row {sel.row}: {why}")
        self.assertTrue(sel.cell and sel.cell.startswith(f"{cc}|bf16|D{dh}|H4|"), (cc, dh, n, sel.cell))
        self.assertTrue(sel.config is None or isinstance(sel.config, dict), sel.config)
        self.assertRegex(tc.launch_word(sel.config), LAUNCH_RE)
        row_cc = T.rows()[sel.row].get("cc")
        if isinstance(row_cc, list):                                                   # a row the provider lists named cards for must list this card to serve it
            self.assertIn(cc, row_cc, (cc, sel.row, row_cc))

    def test_fast_tier_serves_every_chai_lab_crop_on_both_cards_trunk_and_template_heads(self):
        """No stock_statement / no_cell / refusal at any ladder crop on cc 9.0 / 8.0 for the trunk's D64 H4 and the template stack's D32 H4 (an UNCOVERED
        cell or a refusal there is a matter for the shared core, not a shipped step-aside)."""
        tc = self.tc
        for cc in ("9.0", "8.0"):
            for dh in (64, 32):
                for stack in (None, STACKS[cc]):
                    for n in CROPS:
                        sel, fb, word = tc.decide(cc, dh, 4, n, "fast", stack=stack)
                        self.assertIsNone(fb, f"{cc} D{dh} N={n} stack={stack}: {fb}"); self.assertEqual(word, "fast")
                        self._assert_servable(sel, cc, dh, n)
                        self.assertTrue(getattr(sel, "measured", False), f"{cc} D{dh} N={n}: the provider cell is not a measured one ({sel.cell})")

    def test_big_asks_the_provider_word_and_is_served_like_fast(self):
        tc, T = self.tc, self.T
        word = tc.word_for("big")
        self.assertIn(word, T.TIER_WORDS)
        for cc in ("9.0", "8.0"):
            for n in (384, 768, 1200):
                sel, fb, w = tc.decide(cc, 64, 4, n, "big")
                self.assertEqual((fb, w), (None, word)); self._assert_servable(sel, cc, 64, n)

    def test_exact_tier_keeps_the_statement_on_this_engine(self):
        """The provider's exact rows are vouched against the cuEquivariance op (exact_vs=cueq) or ARE the stock op: none is exact vs Chai-1's SDPA
        statement, so the exact tier answers stock_statement at every crop on both cards (the kit's exact mode does not compose the lever)."""
        tc = self.tc
        for cc in ("9.0", "8.0"):
            for n in CROPS:
                sel, fb, word = tc.decide(cc, 64, 4, n, "exact", stack=STACKS[cc])
                self.assertEqual(word, "exact")
                if sel is not None:                                                     # only ever an exact-class row vouched against SDPA
                    self.assertEqual((sel.cls, str(sel.exact_vs)), ("exact", "sdpa"), (cc, n, sel.row))
                else:
                    self.assertIn(fb, ("stock_statement", "no_cell"), (cc, n, fb))
        from chai1_opt import modes
        self.assertNotIn("triattn", modes.KIT_MODES["exact"].pairtrack)

    def test_unmeasured_card_inherits_portable_rows_for_fast_and_big(self):
        """The provider's contract (kernels.triattn >= 0.5.154): on a cc WITHOUT a measured column, fast / big INHERIT the nearest arch-compatible measured
        column's portable rows — a served Selection whose cell is a measured column's (not the card's own), ``measured=False``, the reason naming the
        inheritance — or, where that inherited cell names the stock op, the statement by name; never a refusal.  The cc is picked against the live table."""
        tc, T = self.tc, self.T
        unmeasured = _unmeasured_ccs(T)
        if not unmeasured:
            self.skipTest(f"every candidate card {UNMEASURED_CANDIDATES} has a measured column now ({_measured_columns(T)})")
        cols = _measured_columns(T); inherited = 0
        for cc in unmeasured[:2]:
            for tier in ("fast", "big"):
                for dh in (64, 32):
                    for n in (256, 800, 1536):
                        sel, fb, word = tc.decide(cc, dh, 4, n, tier)
                        self.assertNotIn(str(fb), ("no_cell",), (cc, tier, dh, n)); self.assertFalse(str(fb).startswith("refused:"), (cc, tier, dh, n, fb))   # never a refusal
                        if sel is None:                                                                # the inherited cell names the stock op: the statement, by name (declared)
                            self.assertEqual(fb, "stock_statement", (cc, tier, dh, n, fb)); continue
                        inherited += 1
                        self.assertNotIn(sel.row, T.STOCK_ROWS); self.assertEqual(sel.cls, "fast")
                        self.assertFalse(getattr(sel, "measured", True), (cc, tier, dh, n, sel.cell))
                        self.assertIn("inherit", str(sel.reason).lower(), sel.reason)
                        col = str(sel.cell).split("|")[0]
                        self.assertIn(col, cols, (cc, sel.cell)); self.assertNotEqual(col, tc.cc_word(cc))   # a measured column's cell, not the card's own
                        row_cc = T.rows()[sel.row].get("cc")
                        if isinstance(row_cc, list):                                                   # a row built for named cards only must name the column it is inherited through
                            self.assertTrue(tc.cc_word(cc) in row_cc or col in row_cc, (sel.row, row_cc, col))
        self.assertGreater(inherited, 0, "no inherited cell served on any unmeasured candidate")

    def test_exact_on_an_unmeasured_card_keeps_the_statement(self):
        """EXACT never inherits a kernel row: on an unmeasured cc it names the library / stock op — for this engine the statement BY NAME."""
        tc = self.tc
        from opt_core.kernels import triattn as T
        for cc in list(_unmeasured_ccs(T))[:2] + [(7, 5)]:
            for n in (256, 1024, 2048):
                sel, fb, word = tc.decide(cc, 64, 4, n, "exact")
                self.assertEqual(word, "exact")
                self.assertIsNone(sel, (cc, n, getattr(sel, "row", None))); self.assertIn(fb, ("stock_statement", "no_cell"), (cc, n, fb))

    def test_spy_sees_form_stride_word_stack_and_no_prefer(self):
        tc, T = self.tc, self.T
        seen = {}
        def spy(*a, **kw):
            seen.update(kw); return T.select(*a, **kw)
        tc.decide((9, 0), 64, 4, 2048, "fast", select=spy, stack=STACKS["9.0"])
        self.assertEqual((seen.get("form"), seen.get("position_stride"), seen.get("word"), seen.get("stack")), ("bias_only", 1024, "fast", STACKS["9.0"]))
        self.assertNotIn("prefer", seen)
        tc.decide((9, 0), 32, 4, 512, "fast", select=spy)
        self.assertEqual(seen.get("position_stride"), 512)

    def test_coverage_strings(self):
        tc, T = self.tc, self.T
        served = _served_re(T, 64)
        for cc in ("9.0", (8, 0)):                                                                   # this engine's cards: fast SERVED at every chai_lab crop (an aside there is a matter for the shared core)
            cov = tc.coverage(cc, 64, 4, "fast")
            self.assertEqual(set(cov), set(CROPS))
            for n, w in cov.items():
                self.assertRegex(w, served, (cc, n))
        cov_x = tc.coverage("9.0", 64, 4, "exact")
        self.assertTrue(all(w.startswith(("stock_statement", "no_cell")) or "exact -> " in w for w in cov_x.values()), cov_x)
        others = [c for c in _measured_columns(T) if c not in ("9.0", "8.0")] + [tc.cc_word(c) for c in _unmeasured_ccs(T)[:2]]
        for cc in others:                                                                            # any other column the shared core adds, and unmeasured cards (inheritance): a served row or the
            for n, w in tc.coverage(cc).items():                                                     #  statement BY NAME where the (inherited) cell names the stock op — never a refusal / no_cell
                self.assertTrue(re.match(served, w) or w.startswith("stock_statement"), (cc, n, w))
        for n, w in tc.coverage((7, 5)).items():                                                     # below every row's architecture floor: a DECLARED step-aside by name, never an undeclared refusal
            self.assertTrue(w.startswith(("no_cell", "stock_statement")), (n, w))

    def test_contracts_hold_when_the_provider_flips_a_cell(self):
        """A test-local copy of the provider's table with the 9.0 D64 2048 and 1200 cells' fast order reversed drives decide(): the binding follows the
        table it is given (no kit word pins a row) and every contract holds; the live table is restored and re-checked."""
        import copy
        tc, T = self.tc, self.T
        live = T.table(); fake = copy.deepcopy(live)
        flipped = {}
        for key in ("9.0|bf16|D64|H4|N<=2048|fwd", "9.0|bf16|D64|H4|N<=1200|fwd"):
            c = fake["cells"][key]
            for blk in [c] + ([c["strided"]] if isinstance(c.get("strided"), dict) else []) + [f for f in (c.get("forms") or {}).values() if isinstance(f, dict)]:
                order = [r for r in blk.get("fast_order", []) if r in T.ROW_NAMES and r not in T.STOCK_ROWS]
                if len(order) >= 2:
                    blk["fast_order"] = list(reversed(order)); blk["fast"] = blk["fast_order"][0]; flipped[key] = True
        T._TABLE = fake; T.select_cache_clear()
        try:
            for n in (1024, 2048):
                sel, fb, word = tc.decide((9, 0), 64, 4, n, "fast")
                self.assertIsNone(fb, (n, fb)); self._assert_servable(sel, "9.0", 64, n)
            self._assert_servable(tc.decide((8, 0), 64, 4, 1024, "fast")[0], "8.0", 64, 1024)
        finally:
            T._TABLE = live; T.select_cache_clear()
        self.assertTrue(flipped, "the live table offered no cell with two servable rows to flip (test needs a refresh)")
        self.assertEqual(tc.decide((9, 0), 64, 4, 2048, "fast")[0].row, T.select("9.0", "bf16", 64, 4, 2048, word="fast", position_stride=1024, form=tc.FORM).row)   # restored


class Binding(unittest.TestCase):
    """install() + the impl's step-aside paths with stand-ins (no GPU): the plug, the chain word, the tier, the declared fallbacks, the gate."""

    def setUp(self):
        _core()
        from chai1_opt import triattn_core as tc
        self.tc = tc
        tc._ANNOUNCED.clear()
        self.calls = []
        def lowcopy(mod, zn, mask):
            self.calls.append("lowcopy"); return "lowcopy-out"
        lowcopy.__name__ = "line_statement"
        import torch
        self.TR = types.SimpleNamespace(CFG={"triattn_impl": None, "trimul_impl": None}, bfw=lambda w: w.to(torch.bfloat16))
        TR = self.TR
        def line_cfg():                                                              # the stack's per-call fixer: the tier1 line sets its low-copy statement
            TR.CFG["triattn_impl"] = lowcopy
        line_cfg.chai1_opt_lever = "tier1"
        self.tw = types.SimpleNamespace(cfg_fn=line_cfg)
        self.lowcopy = lowcopy
        self.led = tc.new_ledger()

    C = 64                                                                                    # the pair channel count of these stand-in modules (real tensors, CPU)

    def _mod(self, H=4, dh=64):
        import torch
        g = torch.Generator().manual_seed(0)
        lin = lambda o: types.SimpleNamespace(weight=torch.randn(o, self.C, generator=g) * 0.2)
        return types.SimpleNamespace(H=H, dh=dh, pair2b=lin(2 * H), pair2qkvg1=lin(H * 4 * dh), pair2qkvg2=lin(H * 4 * dh))

    def _mask(self, B, N):
        import torch
        m = torch.ones(B, N, N, dtype=torch.bool)
        m[:, :, N - 2:] = False; m[:, N - 2:, :] = False                                      # two padded tokens, as a crop pads
        return m

    def _zn(self, B=1, N=16):
        import torch
        return torch.randn(B, N, N, self.C, generator=torch.Generator().manual_seed(1))

    def test_install_wraps_cfg_fn_and_pins_the_step_aside_target(self):
        tc, TR = self.tc, self.TR
        fact = tc.install(self.tw, TR, self.led, mode="big", cc=(9, 0))
        impl = TR.CFG["triattn_impl"]
        self.assertEqual(getattr(impl, "chai1_opt_lever", None), "triattn")
        self.assertIs(impl.chai1_opt_state["prev"], self.lowcopy); self.assertEqual(impl.chai1_opt_state["tier"], "big")
        self.assertEqual(self.tw.cfg_fn.chai1_opt_lever, "tier1+triattn")
        self.assertIn("by tier word (tier=big word=", fact); self.assertIn("step-aside target: line_statement", fact)
        self.assertEqual((self.led.get("tier"), self.led.get("word")), ("big", tc.word_for("big")))
        TR.CFG["triattn_impl"] = "clobbered"; self.tw.cfg_fn()                        # the stack re-runs the fixer per trunk call: the binding is back on the plug
        self.assertIs(TR.CFG["triattn_impl"], impl)
        later = lambda mod, zn, mask: "chunked"                                       # a wrapper installed AFTER this lever (big's chunker) is never adopted as the step-aside target
        TR.CFG["triattn_impl"] = later; self.tw.cfg_fn()
        self.assertIs(impl.chai1_opt_state["prev"], self.lowcopy)

    def test_install_refuses_without_plug_points_or_a_tier(self):
        tc = self.tc
        with self.assertRaises(LookupError):
            tc.install(types.SimpleNamespace(cfg_fn=None), types.SimpleNamespace(CFG={}), self.led)
        with self.assertRaises(LookupError):
            tc.install(types.SimpleNamespace(), self.TR, self.led)
        with self.assertRaises(LookupError):
            tc.install(self.tw, self.TR, self.led, mode="off")

    def _served_cc(self, dh=32, n=16):
        """A card the provider SERVES for (dh, H4, n): the first unmeasured candidate (inheritance), else this engine's 9.0."""
        from opt_core.kernels import triattn as T
        for cc in list(_unmeasured_ccs(T)) + [(9, 0), (8, 0)]:
            if self.tc.decide(cc, dh, 4, n, "fast")[0] is not None:
                return cc
        self.skipTest("no served card for the stand-in shape")

    def _decide_as(self, word):
        """Pin the provider's decision for this test (its answer for a given card is the shared core's to change; the binding's handling of each answer is the contract)."""
        tc = self.tc
        orig = tc.decide
        tc.decide = lambda cc, dh, H, N, tier="fast", select=None, stack=None: (None, word, tc.word_for(tier))
        self.addCleanup(setattr, tc, "decide", orig)

    def test_a_card_without_cells_steps_aside_by_name_once_on_stderr(self):
        tc, TR = self.tc, self.TR
        tc.install(self.tw, TR, self.led, mode="fast", cc=(12, 0))                    # a card the provider answers no_cell / no_row for
        impl = TR.CFG["triattn_impl"]
        impl.chai1_opt_state["stack"] = None
        self._decide_as("no_cell")
        err = io.StringIO()
        with redirect_stderr(err):
            for _ in range(3):
                self.assertEqual(impl(self._mod(), self._zn(), self._mask(1, 512)), "lowcopy-out")
        self.assertEqual(_fb(self.led).get("no_cell"), 3)
        self.assertEqual(err.getvalue().count("TRIATTN tier=fast word=fast form=bias_only statement=line reason=no_cell"), 1, err.getvalue())
        self.assertEqual(self.led.unexpected(), {})                                    # declared: the gate passes as skipped (nothing served in this CPU process)
        self.assertEqual(_served(self.led), 0)

    def test_the_stock_op_named_by_the_provider_is_a_declared_step_aside(self):
        tc, TR = self.tc, self.TR
        tc.install(self.tw, TR, self.led, mode="exact", cc=(9, 0))                    # the exact tier on this engine: stock_statement at every crop
        impl = TR.CFG["triattn_impl"]; impl.chai1_opt_state["stack"] = None
        self._decide_as("stock_statement")
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(impl(self._mod(), self._zn(), self._mask(1, 384)), "lowcopy-out")
            self.assertEqual(impl(self._mod(), self._zn(), self._mask(1, 384)), "lowcopy-out")
        self.assertEqual(_fb(self.led).get("stock_statement"), 2); self.assertEqual(self.led.unexpected(), {})
        self.assertEqual(err.getvalue().count("reason=stock_statement"), 1, err.getvalue())

    def test_an_undeclared_refusal_refuses_the_gate(self):
        tc, TR = self.tc, self.TR
        from opt_core.kernels import triattn as T
        tc.install(self.tw, TR, self.led, mode="fast", cc=(9, 0))
        impl = TR.CFG["triattn_impl"]; impl.chai1_opt_state["stack"] = None
        orig = tc.decide
        tc.decide = lambda cc, dh, H, N, tier="fast", select=None, stack=None: (None, "refused:install_failed:test", "fast")
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(impl(self._mod(), self._zn(), self._mask(1, 512)), "lowcopy-out")
        finally:
            tc.decide = orig
        un = self.led.unexpected()
        self.assertEqual(len(un), 1); self.assertTrue(next(iter(un)).startswith("refused:"), un)
        self.assertIn("REFUSED install_failed:test", err.getvalue())
        self.assertFalse(self.led.gate(require_served=True).ok)

    def test_served_path_runs_the_statement_on_the_provider_row_with_real_tensors(self):
        """A card the provider serves (an unmeasured one by inheritance when the table leaves one, else 9.0): the binding's torch path runs end to end on real CPU tensors — pair bias with the mask
        folded, strided [B, N, H, S, D] views, the provider call (a CPU reference stands in for the GPU row, honouring the provider's tensor contract),
        gating into the cat buffer — and reproduces the statement (SDPA per direction, the ending direction on the transposed pair track, kept in
        its transposed orientation) to bf16 tolerance; the ledger counts a served call, no fallback; the cell announce names the inherited cell."""
        import torch
        import torch.nn.functional as F
        tc, TR, T = self.tc, self.TR, __import__("opt_core.kernels.triattn", fromlist=["x"])
        calls = []
        def cpu_row(q, k, v, bias, mask=None, scale=None, **kw):                             # the provider's contract: q/k/v [B, N, H, S, D], bias [B, 1, H, S, S] fp32 -> [B, N, H, S, D]
            calls.append((tuple(q.shape), q.stride(), kw.get("word"), getattr(kw.get("selection"), "row", None), kw.get("form")))
            self.assertEqual(q.stride(-1), 1); self.assertIsNone(mask); self.assertEqual(bias.dtype, torch.float32)
            a = torch.einsum("bnhsd,bnhtd->bnhst", q.float(), k.float()) / (q.shape[-1] ** 0.5) + bias
            return torch.einsum("bnhst,bnhtd->bnhsd", a.softmax(-1), v.float()).to(q.dtype).contiguous()
        orig = T.triangle_attention; T.triangle_attention = cpu_row; self.addCleanup(setattr, T, "triangle_attention", orig)
        H, dh, B, N = 4, 32, 1, 16
        tc.install(self.tw, TR, self.led, mode="fast", cc=self._served_cc(dh, N))
        impl = TR.CFG["triattn_impl"]; impl.chai1_opt_state["stack"] = None
        mod, zn, mask = self._mod(H, dh), self._zn(B, N), self._mask(B, N)
        err = io.StringIO()
        with redirect_stderr(err):
            out = impl(mod, zn, mask)
        self.assertEqual(tuple(out.shape), (B, N, N, 2 * H * dh)); self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(len(calls), 2 * B); self.assertTrue(all(c[2] == "fast" and c[4] == "bias_only" and c[3] not in T.STOCK_ROWS for c in calls), calls)
        self.assertEqual(calls[0][0], (1, N, H, N, dh)); self.assertEqual(calls[0][1][1], N * H * 4 * dh); self.assertEqual(calls[0][1][3], H * 4 * dh)   # strided views of the projection buffer, no copies
        self.assertEqual((_served(self.led), _fb(self.led)), (1, {}))
        self.assertRegex(err.getvalue(), r"TRIATTN tier=fast word=fast form=bias_only row=\S+ core_cell=\d+\.\d+\|bf16\|D32\|H4\|")   # the serving (inherited) cell, announced once
        # the statement, independently (fp32 reference of chai1_eager.trunk.TriangleAttention between its LayerNorm and linear_out)
        zb = zn.to(torch.bfloat16).float()
        b = F.linear(zb, mod.pair2b.weight.to(torch.bfloat16).float()).masked_fill(~mask.unsqueeze(-1), -10000)          # [B, N, N, 2H]
        ref = torch.empty(B, N, N, 2 * H * dh)
        for d, W in enumerate((mod.pair2qkvg1.weight, mod.pair2qkvg2.weight)):
            zin = zb if d == 0 else zb.transpose(1, 2)
            x = F.linear(zin, W.to(torch.bfloat16).float()).view(B, N, N, H, 4, dh)
            q, k, v, g = (x[..., i, :].permute(0, 3, 1, 2, 4) for i in range(4))                                          # [B, H, r, s, D]
            bias = b[..., d * H:(d + 1) * H].permute(0, 3, 1, 2).unsqueeze(2)                                              # [B, H, 1, q, k] broadcast over rows
            a = (torch.einsum("bhrsd,bhrtd->bhrst", q, k) / dh ** 0.5 + bias).softmax(-1)
            o = torch.einsum("bhrst,bhrtd->bhrsd", a, v) * torch.sigmoid(g)
            ref[..., d * H * dh:(d + 1) * H * dh] = o.permute(0, 2, 3, 1, 4).reshape(B, N, N, H * dh)
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 3e-2, rel)

    def test_a_row_that_raises_at_first_call_steps_aside_by_name_for_the_process(self):
        """An inherited row that fails to build / launch on this card (a raw exception, not a provider Refusal): traceback ONCE, announce once,
        booked fallback:row_error (declared: the gate holds, the fold proceeds), LEVER fact error=<ExcType>, and the statement serves this call and
        every later call of the process without asking the row again."""
        import torch
        tc, TR, T = self.tc, self.TR, __import__("opt_core.kernels.triattn", fromlist=["x"])
        n_calls = []
        def broken_row(*a, **kw):
            n_calls.append(1); raise RuntimeError("0 active drivers ([]). There should only be one.")
        orig = T.triangle_attention; T.triangle_attention = broken_row; self.addCleanup(setattr, T, "triangle_attention", orig)
        tc.install(self.tw, TR, self.led, mode="fast", cc=self._served_cc(32, 16))
        impl = TR.CFG["triattn_impl"]; impl.chai1_opt_state["stack"] = None
        mod, zn, mask = self._mod(4, 32), self._zn(1, 16), self._mask(1, 16)
        err = io.StringIO()
        with redirect_stderr(err):
            for _ in range(3):
                self.assertEqual(impl(mod, zn, mask), "lowcopy-out")                        # the step-aside target (the line's statement stand-in) served every call
        self.assertEqual(len(n_calls), 1)                                                    # the row was asked once; the process stays on the statement afterwards
        self.assertEqual(_fb(self.led), {"row_error": 3}); self.assertEqual(_served(self.led), 0)
        self.assertEqual(self.led.unexpected(), {}); self.assertEqual(self.led.get("error"), "RuntimeError")
        self.assertEqual(impl.chai1_opt_state.get("row_error"), "RuntimeError")
        e = err.getvalue()
        self.assertEqual(e.count("Traceback (most recent call last)"), 1, e)                # the traceback once
        self.assertEqual(e.count("TRIATTN tier=fast word=fast form=bias_only row="), 1, e)   # the cell announce once
        self.assertEqual(e.count(" ERROR RuntimeError: "), 1, e)                            # the ERROR announce once
        self.assertIn("ERROR RuntimeError: 0 active drivers", e); self.assertIn("stepped aside", e)

    def test_without_a_line_plug_the_module_statement_serves(self):
        tc = self.tc
        TR = types.SimpleNamespace(CFG={"triattn_impl": None}, bfw=lambda w: w)
        tw = types.SimpleNamespace(cfg_fn=None)                                      # a wrapper whose line sets no plug
        tc.install(tw, TR, self.led, mode="fast", cc=(12, 0))
        TR.CFG["triattn_impl"].chai1_opt_state["stack"] = None
        self._decide_as("no_cell")
        with redirect_stderr(io.StringIO()):
            self.assertIs(TR.CFG["triattn_impl"](self._mod(), self._zn(), self._mask(4, 256)), NotImplemented)


class Registration(unittest.TestCase):
    def setUp(self):
        _core()

    def test_pairtrack_registry_modes_carry_the_lever(self):
        from chai1_opt import modes, pairtrack, registry, triattn_core as tc
        self.assertIn("triattn", pairtrack.LEVERS)
        self.assertEqual(tuple(pairtrack.EXPECTED_FALLBACKS["triattn"]), tuple(tc.EXPECTED_FALLBACKS))
        self.assertIn("triattn", registry.LEVERS); self.assertEqual(registry.LEVERS["triattn"].probe, ("pairtrack", "triattn"))
        self.assertEqual(registry.FAMILY["triattn"], "F1"); self.assertEqual(registry.ORIGIN["triattn"], "core")
        self.assertEqual(registry.STRATEGY["triattn"], tc.STRATEGY)
        self.assertIn("triattn", modes.KIT_MODES["fast"].pairtrack); self.assertIn("triattn", modes.KIT_MODES["big"].pairtrack)
        self.assertNotIn("triattn", modes.KIT_MODES["exact"].pairtrack)             # the exact tier keeps the statement by name on this engine
        self.assertEqual(tuple(modes.PAIRTRACK_LEVERS), tuple(pairtrack.LEVERS))
        self.assertIn("triattn", modes.KIT_MODES["fast"].in_process)

    def test_no_kit_cell_table(self):
        """The shared core's rule: no kit-side cell table for a family the shared core serves — the file and its reader are gone."""
        from chai1_opt import triattn_core as tc
        self.assertFalse(os.path.exists(os.path.join(PKG, "cells", "triattn.cells.json")))
        for name in ("cells", "validate", "resolve", "CELLS_FILE", "PREFER", "STATEMENT"):
            self.assertFalse(hasattr(tc, name), name)
        src = open(os.path.join(PKG, "triattn_core.py"), encoding="utf-8").read()
        self.assertNotIn("triattn.cells.json", src.split('"""', 2)[-1])              # the code below the module docstring never names the retired table

    def test_strategy_id_is_canonical_in_the_core_table(self):
        from chai1_opt import triattn_core as tc
        import opt_core
        tab = json.load(open(os.path.join(os.path.dirname(opt_core.__file__), "STRATEGIES.json")))
        self.assertIn(tc.STRATEGY, [c["id"] for c in tab["canonical"]])

    def test_core_floor_carries_the_provider(self):
        """The kit's core pin (a floor) is at least the first core whose provider takes form= / position_stride= / stack=, and the producer list names the module."""
        from chai1_opt import _core, triattn_core as tc
        sect = open(os.path.join(PKG, "..", "pyproject.toml"), encoding="utf-8").read().split("[tool.opt_core]", 1)[1]
        pin = [ln for ln in sect.splitlines() if ln.strip().startswith("version")][0].split("=", 1)[1].split("#")[0].strip().strip('"')
        v = lambda s: tuple(int(x) for x in s.split("."))
        self.assertGreaterEqual(v(pin), v(tc.CORE_FLOOR), pin)
        self.assertIn("opt_core.kernels.triattn", _core.REQUIRED_PRODUCERS)
        import opt_core
        self.assertGreaterEqual(v(opt_core.__version__.split("+")[0]), v(tc.CORE_FLOOR))


if __name__ == "__main__":
    unittest.main()
