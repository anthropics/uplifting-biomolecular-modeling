"""The fused transition's named fallback (stack.report_fzt over the patched module's fzt_describe()): a cell of the route the core's cell
table does not serve on this stack — its compute capability / Triton line carries no row — ran the stock transition; the package prints the
APPLIED line, then ONE `FALLBACK lever=RFD3_FZT reason=no-cell …` line, and the activation is partial (RFD3_FZT fell
back). With every cell served (the H100 case) the evidence is the APPLIED line alone, byte for byte, and nothing is partial. The patched
module keeps the refusal words per cell (FZT_UNSERVED), counts those calls as cells_stock['no-cell:<cell>'], and lets every other refusal of a
served call propagate. A card with a served ROW RANGE (FZT_ROWS_BY_CC: compute capability 8.0; none on 9.0) routes a call outside its cell's
range to the stock body BY NAME (cells_stock['rows<MIN:<cell>'] / ['rows>MAX:<cell>']) — by design, not a fallback, never partial — and the
APPLIED line names the ranges (rows_cc= rows_served=) on that card only: the 9.0 line is byte for byte the one above."""
import ast
import contextlib
import io
import os
import sys
import types
import unittest

from .. import modes, report, stack

PATCHED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "xattempt_addon", "patched",
                       "rfd3", "model", "layers", "layer_utils.py")

SERVED = {"RFD3_FZT": True, "installs": 1, "fused_calls": 40, "stock_calls": 2, "cells_fused": {"128x256": 20, "256x1024": 20}, "cells_stock": {"not-cuda:128x256": 2},
          "modules_packed": 4, "cells": "128x256,128x512,256x512,256x1024", "cells_unserved": {}, "opt_core_cells_sha256": "ab" * 32}   # the keys of patched layer_utils.fzt_describe(), in its order
UNSERVED = dict(SERVED, fused_calls=0, stock_calls=42, cells_fused={}, cells_stock={"no-cell:128x256": 20, "no-cell:256x1024": 20, "not-cuda:128x256": 2},
                cells_unserved={"128x256": "no-cell:fpf:transition:128x256:8.0|3.3", "256x1024": "no-cell:fpf:transition:256x1024:8.0|3.3"})
ROWS_SERVED_80 = "128x256:16384-1048544,128x512:4096-1048544,256x512:4096-1048544,256x1024:4096-1048544"   # FZT_ROWS_BY_CC["8.0"] rendered (fzt_describe's rows_served word)
SERVED_80 = {}                                                                                                # an 8.0 card: the same tally, the card's row words after `cells`, two calls below a floor (a stock route by name)
for _k, _v in SERVED.items():
    SERVED_80[_k] = _v
    if _k == "cells":
        SERVED_80["rows_cc"], SERVED_80["rows_served"] = "8.0", ROWS_SERVED_80
SERVED_80.update(fused_calls=38, stock_calls=4, cells_stock={"not-cuda:128x256": 2, "rows<16384:128x256": 2})


class TestFztFallback(unittest.TestCase):
    def setUp(self):
        self._saved = (stack._STATE["report"], sys.modules.get(modes.KIT_FZT_MODULE), report._MODULES["fzt"])

    def tearDown(self):
        stack._STATE["report"], mod, report._MODULES["fzt"] = self._saved
        if mod is None:
            sys.modules.pop(modes.KIT_FZT_MODULE, None)
        else:
            sys.modules[modes.KIT_FZT_MODULE] = mod

    def _run(self, stats):
        mod = types.ModuleType(modes.KIT_FZT_MODULE)
        mod.fzt_describe = lambda: dict(stats)
        sys.modules[modes.KIT_FZT_MODULE] = mod
        report._MODULES["fzt"] = None                                            # the package finds the module afresh (sys.modules), as in a run
        rep = {"active": True, "mode": "exact", "env": dict(modes.KIT_MODES["exact"].env), "levers_planned": sorted(modes.KIT_MODES["exact"].env),
               "levers_applied": sorted(modes.KIT_MODES["exact"].env), "levers_fallback": [], "levers_unavailable": [], "partial": False, "partial_reason": None}
        stack._STATE["report"] = rep
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            line = stack.report_fzt()
            again = stack.report_fzt()                                        # once per process: the second call returns the recorded line, prints nothing
        self.assertEqual(line, again)
        return rep, line, [ln for ln in err.getvalue().splitlines() if ln.strip()]

    def test_every_cell_served_is_the_applied_line_alone(self):
        rep, line, lines = self._run(SERVED)
        self.assertEqual(lines, ["[rfdiffusion3-opt] APPLIED rfd3.model.layers.layer_utils RFD3_FZT=True installs=1 fused_calls=40 stock_calls=2 modules_packed=4 "
                                 "cells=128x256,128x512,256x512,256x1024 opt_core_cells_sha256=" + "ab" * 32])
        self.assertEqual((rep["partial"], rep["levers_fallback"], rep.get("fzt_fallback_line")), (False, [], None))

    def test_an_unserved_cell_is_named_once_and_partial(self):
        rep, line, lines = self._run(UNSERVED)
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith("[rfdiffusion3-opt] APPLIED rfd3.model.layers.layer_utils RFD3_FZT=True installs=1 fused_calls=0 stock_calls=42 "))
        self.assertEqual(lines[1], "[rfdiffusion3-opt] FALLBACK lever=RFD3_FZT reason=no-cell cells=128x256,256x1024 "
                                   "core=no-cell:fpf:transition:128x256:8.0|3.3;no-cell:fpf:transition:256x1024:8.0|3.3 route=stock-transition calls=40")
        self.assertEqual(lines[1], rep["fzt_fallback_line"])
        self.assertTrue(rep["partial"])
        self.assertEqual(rep["levers_fallback"], [modes.KIT_FZT_SWITCH])                    # the one lever that fell back, by name
        self.assertFalse(hasattr(modes, "CORE_CANDIDATE_SWITCH"))
        self.assertIn("opt_core serves no fused transition cell for 128x256,256x1024 on this stack (no-cell:fpf:transition:128x256:8.0|3.3; no-cell:fpf:transition:256x1024:8.0|3.3): "
                      "those calls ran the stock transition (40 call(s))", rep["partial_reason"])
        v = report.verdict(0, rep, False)                                          # the family rule: exit 3 unless --allow-partial records it
        self.assertEqual((v["status"], v["exit_code"]), ("partial", 3))
        self.assertEqual(report.verdict(0, rep, True)["status"], "partial_allowed")

    def test_a_card_row_range_is_named_on_the_applied_line_and_is_not_partial(self):
        """An 8.0 card (FZT_ROWS_BY_CC row): the APPLIED line carries rows_cc= / rows_served= right after cells=, calls below a floor are a
        counted stock route, and nothing is partial or a fallback (no FALLBACK line)."""
        rep, line, lines = self._run(SERVED_80)
        self.assertEqual(lines, ["[rfdiffusion3-opt] APPLIED rfd3.model.layers.layer_utils RFD3_FZT=True installs=1 fused_calls=38 stock_calls=4 modules_packed=4 "
                                 "cells=128x256,128x512,256x512,256x1024 rows_cc=8.0 rows_served=" + ROWS_SERVED_80 + " opt_core_cells_sha256=" + "ab" * 32])
        self.assertEqual((rep["partial"], rep["levers_fallback"], rep.get("fzt_fallback_line")), (False, [], None))
        import re                                                                  # the activation expectation of this line (installs, fused_calls, cells) holds on both cards
        for ln in (line, self._run(SERVED)[1]):
            self.assertRegex(ln, r"\[rfdiffusion3\-opt\] APPLIED rfd3\.model\.layers\.layer_utils RFD3_FZT=True (?=.*\binstalls=[1-9])(?=.*\bfused_calls=[1-9])(?=.*\bcells=128x256,128x512,256x512,256x1024\b)")

    def test_the_row_range_table_is_per_capability_and_read_from_the_device(self):
        """Source facts of the carried file: FZT_ROWS_BY_CC has an 8.0 row covering every cell of FZT_CELLS with min < max and NO 9.0 row (H100:
        every row count served, its APPLIED line unchanged); the range is judged by a pure function of (cell, rows, cc) whose words name the
        bound; the capability comes from the tensor's device (torch.cuda.get_device_capability), never from MODEL_OPT_TARGET_GPU; the row check
        sits after the autocast check and before any weight packing; fzt_describe adds the words through _fzt_rows_fields()."""
        src = open(PATCHED, encoding="utf-8").read()
        tree = ast.parse(src)
        assigns = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        cells = ast.literal_eval(assigns["FZT_CELLS"])
        table = ast.literal_eval(assigns["FZT_ROWS_BY_CC"])
        self.assertEqual(set(table), {"8.0"})                                        # no 9.0 row: H100 serves every row count, as before this table existed
        self.assertEqual(set(table["8.0"]), set(cells))
        for cell, (lo, hi) in table["8.0"].items():
            self.assertTrue(0 < lo < hi <= (1 << 20) - 16, (cell, lo, hi))            # below the floors cuBLAS re-orders in bands on sm_80; from 2^20-16 rows up it splits the row dimension
        self.assertEqual(table["8.0"][(128, 256)][0], 16384); self.assertEqual(table["8.0"][(128, 512)][0], 4096); self.assertEqual({hi for _, hi in table["8.0"].values()}, {1048544})
        self.assertEqual(",".join(f"{c}x{h}:{table['8.0'][(c, h)][0]}-{table['8.0'][(c, h)][1]}" for c, h in cells), ROWS_SERVED_80)   # the fixture above IS the table rendered
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fzt_rows_word")
        ns = {"FZT_ROWS_BY_CC": table}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), PATCHED, "exec"), ns)     # a pure function of (cell, rows, cc): no torch
        word = ns["_fzt_rows_word"]
        self.assertEqual([word((128, 256), r, "8.0") for r in (0, 2500, 16383, 16384, 90000, 1048544, 1048545, 1280000)],
                         ["rows<16384", "rows<16384", "rows<16384", None, None, None, "rows>1048544", "rows>1048544"])
        self.assertEqual([word((128, 512), r, "8.0") for r in (2500, 4095, 4096, 720000, 2000000)], ["rows<4096", "rows<4096", None, None, "rows>1048544"])
        self.assertEqual([word(c, r, "9.0") for c in cells for r in (1, 2500, 10**7)], [None] * 12)   # H100: every row count served
        self.assertEqual(word((384, 1536), 10, "8.0"), None)                          # a cell outside the table is not this function's business (cell: route above it)
        body = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_transition_forward_fzt"))
        self.assertLess(body.index("_fzt_autocast_bf16()"), body.index("_fzt_rows_word(cell, X.numel() // C, _fzt_cc(X.device))"))
        self.assertLess(body.index("_fzt_rows_word(cell, X.numel() // C, _fzt_cc(X.device))"), body.index("_fzt_weights(self, X.device)"))
        self.assertIn('_fzt_count("stock", rows_word + ":" + key)', body)
        cc_fn = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fzt_cc"))
        self.assertIn("torch.cuda.get_device_capability(", cc_fn)
        self.assertNotIn("MODEL_OPT_TARGET_GPU", src)                                # the probed device, never the configuration's word
        self.assertIn("d.update(_fzt_rows_fields())", src)

    def test_the_patched_module_falls_back_on_no_cell_only(self):
        """Source facts of the carried file (it imports upstream and torch; read, not imported): the served call sits in a try whose handler
        keeps the refusal words, re-raises anything but `no-cell:`, and the unserved cell is checked before any packing."""
        src = open(PATCHED, encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_transition_forward_fzt")
        body = ast.get_source_segment(src, fn)
        self.assertLess(body.index("if key in FZT_UNSERVED:"), body.index("_fzt_weights(self, X.device)"))
        handler = next(n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler))
        self.assertEqual(ast.get_source_segment(src, handler.type), "_FZT_PF.Unsupported")
        hsrc = ast.get_source_segment(src, handler)
        self.assertIn('if not words.startswith("no-cell:")', hsrc)
        self.assertIn("raise", hsrc)
        self.assertIn('_fzt_count("stock", "no-cell:" + key)', body)
        from opt_core.attn import pair_fused as PF                                   # the words the handler keys on are the core's own: find_cell's refusal token and Unsupported.reason
        import inspect
        self.assertEqual(PF.Unsupported("no-cell:fpf:transition:128x256:8.0|3.3").reason, "no-cell:fpf:transition:128x256:8.0|3.3")
        d = PF.lookup_cell("fpf", "transition", (64, 256), stack=("9.0", "3.3"))            # a key the core's table does not list on a capability WITH rows: the per-call refusal word (the core's CPU form, stack= names the box)
        self.assertIsNone(d.row); self.assertTrue(d.reason.startswith("no-cell:fpf:transition:64x256:"), d.reason)
        with self.assertRaises(PF.Unsupported) as cm:                                # find_cell raises it with that word as .reason — what the handler's startswith("no-cell:") keys on
            PF.find_cell("fpf", "transition", (64, 256), None, stack=("9.0", "3.3"))
        self.assertTrue(cm.exception.reason.startswith("no-cell:"), cm.exception.reason)
        self.assertTrue(callable(getattr(PF, "lookup_cell", None)) and "Unsupported(" in inspect.getsource(PF.find_cell))
        self.assertIn("ranked_logger = RankedLogger(__name__", src)                    # the module-level logger _fzt_unserved writes to
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertLessEqual({"_fzt_unserved", "fzt_describe", "_transition_forward_fzt"}, names)
        self.assertIn('d["cells_unserved"] = dict(FZT_UNSERVED)', src)


if __name__ == "__main__":
    unittest.main()
