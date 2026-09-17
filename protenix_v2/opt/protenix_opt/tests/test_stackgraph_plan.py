"""stackgraph plan-for-capture: the trunk stack graph takes a first-sighted signature's eager
call as its bitwise oracle, and two levers inside the stack (transition_core over opt_core.kernels.transition, pf_attn over opt_core.kernels.apb)
bind providers whose cells are keyed by TIMING COLUMN (eager | graph).  Class contracts, CPU only (no device, no engine):

  * the window: fpf_stackgraph.stackgraph.planning() is True inside `_planning(sig)` and False after it -- also when the body raises; the
    reference call, the side-stream warm-up and the capture of a first sighting sit INSIDE the window (source structure); the SUMMARY line keeps
    every field it had and appends planned_refs;
  * the column per word: under a tier word (fast | big) a call inside the window is selected from the GRAPH column exactly like a call inside a
    capture; under exact the column follows the literal capture state alone (ptx_transition_core.timing_word, apb_core._timing); serve() keys its
    memo and asks the provider with that column while capture= stays the literal capture state;
  * the provider grid on the H100 cu130 stack (opt_core.kernels.transition.select, CPU): the set of classes whose REFERENCE-CALL row
    changes with this fix (eager column -> graph column) is exactly fast | big x {pair c256/h1024, single c384/h1536} x every N in 17..448, and no
    exact cell; pf_attn's cell pf_h16d24 names different rows in the two columns under fast | big (the second source the fix covers).
"""
import ast
import os
import re
import sys
import types
import unittest

from protenix_opt import stack

_SRC = os.path.join(stack.opt_home(), "forward", "flashpairformer", "src")          # the FlashPairformer tree's src (env.sh PYTHONPATH on the worker)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SG_FILE = os.path.join(_SRC, "fpf_stackgraph", "stackgraph.py")
STACK = "H100:torch2.13.0+cu130/3.7.1"                                                # the K row's stack label (IMG_C: torch 2.13.0+cu130, triton 3.7.1)
SIZES = (17, 64, 128, 200, 256, 257, 300, 400, 448)                                  # the stack graph's token range: 17 <= N <= PTX_BLK_GRAPH_MAXTOK (448)
FAMILIES = (("pair", 256, 1024), ("single", 384, 1536), ("pair", 128, 512), ("rows", 64, 128))   # protenix-v2: trunk pair c_z 256 / single c_s 384; c128 / c64 rows: the MSA / template stacks


def _torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _stackgraph():
    """The real module (imports torch; no device touched at import)."""
    import importlib
    return importlib.import_module("fpf_stackgraph.stackgraph")


class PlanningWindow(unittest.TestCase):
    def setUp(self):
        if not _torch():
            self.skipTest("torch absent")
        self.SG = _stackgraph()

    def test_window_sets_and_clears(self):
        SG = self.SG
        self.assertFalse(SG.planning())
        n0 = SG.ST.get("planned_refs", 0)
        with SG._planning(("main", 200)):
            self.assertTrue(SG.planning())
        self.assertFalse(SG.planning())
        self.assertEqual(SG.ST["planned_refs"], n0 + 1)
        self.assertIs(sys.modules["fpf_stackgraph"].planning, SG.planning)          # the package re-exports the predicate the levers look up

    def test_window_clears_when_the_reference_call_raises(self):
        SG = self.SG
        with self.assertRaises(RuntimeError):
            with SG._planning(("main", 200)):
                self.assertTrue(SG.planning())
                raise RuntimeError("the eager reference call failed")
        self.assertFalse(SG.planning())
        with self.assertRaises(MemoryError):                                          # an OOM re-raised from inside the window (the capture path re-raises OOM) leaves no window behind either
            with SG._planning(None):
                raise MemoryError("oom")
        self.assertFalse(SG.planning())


class WindowCoversReferenceWarmupCapture(unittest.TestCase):
    """Source structure of install()'s forward: the first sighting's eager reference (`s_ref, z_ref = _orig(self, s, z, **kw)`), the side-stream
    warm-up and the torch.cuda.graph capture + check all sit inside `with _planning(sig):`; the SUMMARY line keeps its fields and appends planned_refs."""

    def setUp(self):
        self.src = open(_SG_FILE).read()
        self.tree = ast.parse(self.src)

    def _forward(self):
        inst = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "install")
        return next(n for n in ast.walk(inst) if isinstance(n, ast.FunctionDef) and n.name == "forward")

    def test_reference_warmup_capture_inside_the_window(self):
        fwd = self._forward()
        withs = [n for n in ast.walk(fwd) if isinstance(n, ast.With)
                 and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", "") == "_planning" for i in n.items)]
        self.assertEqual(len(withs), 1, "exactly one plan-for-capture window in forward")
        inside = ast.unparse(withs[0]) if hasattr(ast, "unparse") else self.src
        self.assertIn("_orig(self, s, z, **kw)", inside)                             # the eager reference call (the oracle)
        self.assertIn("torch.cuda.stream(side)", inside)                             # the side-stream warm-up
        self.assertIn("torch.cuda.graph(g", inside)                                  # the capture
        self.assertIn("torch.equal(e.z_out, z_ref)", inside)                         # the replay-vs-reference check
        # the replay and eager fall-through paths of a KNOWN signature sit above the window (they select nothing new): every early `return _orig(...)` is outside it
        pre = self.src[self.src.index("    def forward(self, s, z"):self.src.index("with _planning(sig):")]
        self.assertGreaterEqual(pre.count("return _orig(self, s, z, **kw)"), 3)     # not ours / replay-or-eager of a known signature / cache full
        self.assertNotIn("_planning(", pre.split("# ---- first sighting of this signature")[0])

    def test_summary_line_keeps_its_fields_and_appends_planned_refs(self):
        line = next(l for l in self.src.splitlines() if "[fpf_stackgraph] SUMMARY full_eager=" in l)
        keys = re.findall(r"([a-z_]+)=\{", line)
        want = ["full_eager", "live_graphs", "replay_fail", "captures", "replays", "eager", "refused", "oom_skips", "oom_skips_by_ntok", "evicted", "disabled",
                "mode", "budget_gb", "pool_gb", "distinct_ntok", "budget_eager", "pool_bytes_by_ntok", "max_sig_desc", "budget_src", "planned_refs"]
        self.assertEqual(keys, want)


class TimingColumnPerWord(unittest.TestCase):
    """transition_core / pf_attn: the provider column of a call = graph inside a capture; under a tier word also graph inside the plan-for-capture
    window; exact keys on the literal capture state alone."""

    def setUp(self):
        if not _torch():
            self.skipTest("torch absent")
        self.SG = _stackgraph()
        import ptx_transition_core as TCORE
        from protenix_opt import apb_core
        self.TCORE, self.APB = TCORE, apb_core

    def test_columns_outside_the_window(self):
        for word in ("fast", "big", "exact"):
            self.assertEqual(self.TCORE.timing_word(word, False), "eager")
            self.assertEqual(self.TCORE.timing_word(word, True), "graph")
            self.assertEqual(self.APB._timing(word, False), "eager")
            self.assertEqual(self.APB._timing(word, True), "graph")

    def test_columns_inside_the_window(self):
        with self.SG._planning(("main", 200)):
            for word in ("fast", "big"):
                self.assertEqual(self.TCORE.timing_word(word, False), "graph", word)   # the planned eager reference / warm-up: the graph column
                self.assertEqual(self.TCORE.timing_word(word, True), "graph", word)
                self.assertEqual(self.APB._timing(word, False), "graph", word)
                self.assertEqual(self.APB._timing(word, True), "graph", word)
            self.assertEqual(self.TCORE.timing_word("exact", False), "eager")            # exact: the literal capture state, window or not
            self.assertEqual(self.TCORE.timing_word("exact", True), "graph")
            self.assertEqual(self.APB._timing("exact", False), "eager")
        self.assertEqual(self.TCORE.timing_word("fast", False), "eager")                 # and the window closed

    def test_no_stackgraph_module_no_window(self):
        saved = {k: sys.modules.pop(k) for k in ("fpf_stackgraph.stackgraph", "fpf_stackgraph") if k in sys.modules}
        try:
            self.assertFalse(self.TCORE._stackgraph_planning())
            self.assertFalse(self.APB._stackgraph_planning())
            sys.modules["fpf_stackgraph.stackgraph"] = types.ModuleType("fpf_stackgraph.stackgraph")   # a module that predates the window: no planning() -> no window
            self.assertFalse(self.TCORE._stackgraph_planning())
            self.assertEqual(self.TCORE.timing_word("fast", False), "eager")
        finally:
            sys.modules.pop("fpf_stackgraph.stackgraph", None)
            sys.modules.update(saved)

    def test_serve_keys_and_asks_the_provider_with_the_planned_column(self):
        """serve(): memo key, select(timing=) and transition(timing=) carry the planned column; capture= stays the literal (False) state."""
        import torch
        TCORE = self.TCORE
        calls = []

        class Sel:
            row, variant, cfg, cell_key = "esm_t16", None, None, ("cc9.0", "pair_c256_n4")
            def line(self): return "TRANSITION served=esm_t16"

        class FakeT:
            STOCK_ROWS = ("torch_swiglu", "engine_module", "compile")
            class Refusal(Exception):
                kind, fallback = "x", "torch_swiglu"
            @staticmethod
            def cell_word(c, hidden, family="pair"): return "%s_c%d_n%d" % (family, c, hidden // c)
            @staticmethod
            def cfg_word(cfg): return ""
            @staticmethod
            def select(word, **kw): calls.append(("select", word, kw["timing"], kw["capture"])); return Sel()
            @staticmethod
            def pack(**kw): return object()
            @staticmethod
            def transition(x, W, **kw): calls.append(("transition", kw["word"], kw["timing"], kw["capture"])); return x.clone(), Sel()

        class Mod(torch.nn.Module):
            def __init__(self, c=256, n=4):
                super().__init__()
                self.layernorm1 = torch.nn.LayerNorm(c); self.linear_no_bias_a = torch.nn.Linear(c, n * c, bias=False)
                self.linear_no_bias_b = torch.nn.Linear(c, n * c, bias=False); self.linear_no_bias = torch.nn.Linear(n * c, c, bias=False)

        saved = (TCORE._T, TCORE._eligible, dict(TCORE._ST), dict(TCORE._SEL))
        cap_saved = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: False                       # a CPU interpreter: no stream is capturing (the CUDA build's own answer off-device)
        TCORE._T = lambda: FakeT
        TCORE._eligible = lambda module, x: None                                      # a CPU tensor is "eligible" here: the column logic is what is under test
        mod, x = Mod().eval(), torch.zeros(2, 8, 8, 256)
        try:
            for word, want in (("fast", "graph"), ("big", "graph"), ("exact", "eager")):
                TCORE._SEL.clear(); del calls[:]
                TCORE._ST.update(word=word, installed=True)
                with self.SG._planning(("main", 8)):
                    y = TCORE.serve(mod, x)
                self.assertIsNotNone(y, word)
                self.assertEqual([c[0] for c in calls], ["select", "transition"], word)
                self.assertEqual({c[2] for c in calls}, {want}, word)                # the column asked of the provider
                self.assertEqual({c[3] for c in calls}, {False}, word)               # capture= is the literal capture state (no capture on CPU)
                self.assertEqual([k[-1] for k in TCORE._SEL], [want], word)          # the memo key carries the column
                del calls[:]; TCORE._SEL.clear()
                y = TCORE.serve(mod, x)                                              # the same call outside the window: the eager column under every word
                self.assertEqual({c[2] for c in calls}, {"eager"}, word)
        finally:
            torch.cuda.is_current_stream_capturing = cap_saved
            TCORE._T, TCORE._eligible = saved[0], saved[1]
            TCORE._ST.clear(); TCORE._ST.update(saved[2]); TCORE._SEL.clear(); TCORE._SEL.update(saved[3])


def _row(T, word, fam, c, hidden, n, timing):
    """The reference-call row the provider names for one class in one column ('module:<why>' when the statement runs), CPU select()."""
    try:
        s = T.select(word, c=c, hidden=hidden, n_tokens=n, dtype="bf16", family=fam, residual=False, timing=timing, capture=False, cc="9.0",
                     stack=STACK, ln_given=(word == "exact"))
    except T.Refusal as r:
        return "module:refused:%s" % r.kind
    arm = s.row + ((":" + s.variant) if s.variant else "")
    return ("module:" + arm) if s.row in T.STOCK_ROWS else arm


class ProviderGridOnTheStackOfRecord(unittest.TestCase):
    """Which reference-call rows change with the fix, read off the shared core's own select() on the K row's stack (CPU): before = the eager
    column (what the reference served), after = the column timing_word names for a planned, non-capturing call of the word."""

    def test_reference_rows_change_exactly_for_tier_words_pair_c256_single_c384(self):
        from opt_core.kernels import transition as T
        planned_col = {"fast": "graph", "big": "graph", "exact": "eager"}             # ptx_transition_core.timing_word(word, capturing=False) inside the window
        if _torch():
            import ptx_transition_core as TCORE
            SG = _stackgraph()
            with SG._planning(("main", 200)):
                self.assertEqual({w: TCORE.timing_word(w, False) for w in planned_col}, planned_col)
        changed, differ_exact = set(), set()
        for word in ("fast", "big", "exact"):
            for fam, c, hidden in FAMILIES:
                for n in SIZES:
                    before = _row(T, word, fam, c, hidden, n, "eager")
                    after = _row(T, word, fam, c, hidden, n, planned_col[word])
                    if before != after:
                        changed.add((word, fam, c, n))
                    if word == "exact" and before != _row(T, word, fam, c, hidden, n, "graph"):
                        differ_exact.add((fam, c, n))
        # pair c256: the columns differ up to the N<=400 cell (<= 256: v2:fast | esm_t16; 257-400: flash_sm90a:kernel_ln | esm_t16) and agree from the N<=800 cell on
        # (448: esm_t16 in both); single c384: v2:fast | the statement at every size -- so every signature in 17..448 met at least one differing class before the fix
        want = ({(w, "pair", 256, n) for w in ("fast", "big") for n in SIZES if n <= 400}
                | {(w, "single", 384, n) for w in ("fast", "big") for n in SIZES})
        self.assertEqual(changed, want)
        self.assertEqual({n for n in SIZES if any((w, fam, c, n) in changed for w in ("fast", "big") for fam, c, _h in FAMILIES)}, set(SIZES))
        self.assertFalse(any(w == "exact" for w, *_ in changed))                        # exact: no reference-call row moves (the column stays the capture state's)
        # what the tier words' reference now serves inside the window on this stack: the graph column's rows (pair c256 esm_t16, single c384 the statement)
        self.assertEqual({_row(T, w, "pair", 256, 1024, n, "graph") for w in ("fast", "big") for n in SIZES}, {"esm_t16"})
        self.assertTrue(all(_row(T, w, "single", 384, 1536, n, "graph").startswith("module:") for w in ("fast", "big") for n in SIZES))
        # exact's two columns do differ in name at <= 256 tokens (eager: the stock cell; graph: v1) -- immaterial: both are the statement's bytes by vouch and the
        # exact word keeps the literal capture state; recorded so a later change of that decision is a visible edit of this line
        self.assertTrue(differ_exact <= {(fam, c, n) for fam, c, _h in FAMILIES for n in SIZES if n <= 256}, sorted(differ_exact))

    def test_pf_attn_columns_differ_under_tier_words(self):
        """The second source: kernels.apb's pf_h16d24 cell names sdpa (eager column) vs fpf_apb (graph column) under fast | big on cc 9.0 -- the planned
        reference must read the graph column too (apb_core._timing); under exact both columns are the statement."""
        if not _torch():
            self.skipTest("torch absent (kernels.apb.select takes a torch dtype)")
        import torch
        from opt_core.kernels import apb as A
        for word in ("fast", "big"):
            for n in SIZES:
                e = A.select("9.0", torch.bfloat16, "pf_h16d24", n, word=word, samples=1, capture=False, heads=16, head_dim=24, stack=STACK, timing="eager")
                g = A.select("9.0", torch.bfloat16, "pf_h16d24", n, word=word, samples=1, capture=False, heads=16, head_dim=24, stack=STACK, timing="graph")
                c = A.select("9.0", torch.bfloat16, "pf_h16d24", n, word=word, samples=1, capture=True, heads=16, head_dim=24, stack=STACK)
                self.assertEqual((g.row, g.variant), (c.row, c.variant), (word, n))       # the planned eager call names the captured call's row
                self.assertNotEqual((e.row, e.variant), (g.row, g.variant), (word, n))    # ... which the plain eager column does not (the refusal's cause)
                self.assertNotIn(g.row, A.STOCK_ROWS)
        for n in SIZES:
            e = A.select("9.0", torch.bfloat16, "pf_h16d24", n, word="exact", samples=1, capture=False, heads=16, head_dim=24, stack=STACK)
            c = A.select("9.0", torch.bfloat16, "pf_h16d24", n, word="exact", samples=1, capture=True, heads=16, head_dim=24, stack=STACK)
            self.assertEqual((e.row, e.variant), (c.row, c.variant), n)


if __name__ == "__main__":
    unittest.main()
