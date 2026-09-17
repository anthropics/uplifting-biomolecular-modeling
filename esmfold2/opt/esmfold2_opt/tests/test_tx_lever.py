"""Optimization `tx` (trimul group, every class): the pair TriMul served by the shared core's TriMul provider bound by the MODE's TIER word
(`exact` | `fast` | `big`) -- the provider's measured cell table decides the row per call class on every card; a class whose cell names the
stock op and a call the provider refuses BY NAME are served by the upstream fused TriMul (counted, printed once).
CPU tests: registry / MODES membership and strategy, the composition + word rules, the per-class selection + refusal-by-name path on a stub provider,
enable_tx without a CUDA device (bound, deferred), and the provider face's own resolution of the three tier words at this kit's cells (class
contracts: SOME served row, never a named winner)."""
import os, sys, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
if DRV not in sys.path:
    sys.path.insert(0, DRV)
from esmfold2_opt import registry, modes          # noqa: E402
import ef2_server as srv                           # noqa: E402

KIT = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference"))


def _ef2_w4(tc):
    """ef2_w4 imports the esm / transformers forks and triton at module level: present in the kit's stack (the GPU image runs these tests there); a bare
    CPU interpreter without them skips the route tests by name instead of erroring."""
    try:
        import ef2_w4
    except ModuleNotFoundError as e:                                   # noqa: PERF203
        tc.skipTest(f"ef2_w4 needs the kit stack ({e.name})")
    return ef2_w4


class TestTxLever(unittest.TestCase):
    def test_registry_row_and_strategy(self):
        lv = registry.LEVERS["tx"]
        self.assertEqual((lv.field, lv.kit_file, lv.probe, lv.classes, lv.tier_vs_kit_line), (registry.FIELD_TRIMUL, "ef2_w4.py", ("w4", "tx"), None, "T2"))
        self.assertEqual(registry.STRATEGY["tx"], "F2.fpf_trimul_fast")
        self.assertIsNone(registry.class_excludes(lv, "8.0")); self.assertIsNone(registry.class_excludes(lv, "9.0"))
        for gone in ("t9", "sigmoid", "incnt", "formtab", "k3cute", "lnfold"):
            self.assertNotIn(gone, registry.LEVERS); self.assertNotIn(gone, registry.STRATEGY)
        for gone in ("ef2_trimul_v5.py", "ef2_trimul_v6.py", "ef2_w4_fpf_trimul_v4_cells.json", os.path.join("prebuilt", "sm_90a", "ef2_trimul_v6_k3.cubin")):
            self.assertFalse(os.path.exists(os.path.join(DRV, gone)), gone)              # no kit TriMul kernel, cubin or cell table ships: the shared core's provider serves

    def test_modes_carry_tx_on_every_line_and_class(self):
        self.assertEqual([g for g in srv.MODES["opt14_msa"][3].split(";") if g.startswith("trimul:")], ["trimul:tx"])
        self.assertNotIn("trimul:", srv.MODES["opt7x"][3])                              # exact: the upstream fused TriMul is the exact statement itself (no byte-vouched exact-class row to bind today)
        self.assertNotIn("t9", srv.MODES["opt14_msa"][2].split(","))
        for mode in ("fast", "big"):
            for cc in ("9.0", "8.0", "10.0", None):
                res = modes.resolve(mode, "full_msa", KIT, cc=cc)
                self.assertIn("tx", res.levers_for_variant, (mode, cc)); self.assertNotIn("tx", res.not_for_class, (mode, cc))
        self.assertNotIn("tx", modes.resolve("exact", "full_msa", KIT, cc="9.0").levers)
        ab = modes.resolve("fast", "fast", KIT, cc="9.0", ablate="tx")                  # individually switchable through the kit's ablation word
        self.assertNotIn("tx", ab.levers_for_variant)
        self.assertEqual(srv.check_composition("opt14_msa", off=set(ab.levers_off))["trimul"], [])

    def test_composition_and_word_rules(self):
        with mock.patch.dict(srv.MODES, {"x_tx_only": ("fused", "tg", "", "trimul:tx")}):
            self.assertEqual(srv.check_composition("x_tx_only")["trimul"], ["tx"])      # tx without any W4 lever is a valid set (enable_tx installs the TriMul entry points itself)
        with mock.patch.dict(srv.MODES, {"x_unknown": ("fused", "tg", "t3", "trimul:k3cute")}):
            with self.assertRaises(ValueError) as cm:
                srv.check_composition("x_unknown")
            self.assertIn("unknown levers", str(cm.exception))
        env = dict(os.environ)
        try:
            os.environ.pop(srv.TRIMUL_TIER_ENV, None); os.environ.pop(srv.PACKAGE_MODE_ENV, None)
            self.assertEqual([srv.trimul_tier(k) for k in ("opt7x", "opt14_msa", "opt7_msa")], ["exact", "fast", "fast"])
            os.environ[srv.PACKAGE_MODE_ENV] = "big"; self.assertEqual(srv.trimul_tier("opt14_msa"), "big")   # the memory mode binds `big` literally
            os.environ[srv.TRIMUL_TIER_ENV] = "exact"; self.assertEqual(srv.trimul_tier("opt14_msa"), "exact")     # the engineering override wins
        finally:
            os.environ.clear(); os.environ.update(env)

    def test_selection_and_refusal_by_name(self):
        import torch
        W = _ef2_w4(self)

        class _Refusal(Exception):
            def __init__(self, kind, row, fallback):
                super().__init__(f"{kind} [row {row}] -> fallback row {fallback}"); self.kind, self.row, self.fallback = kind, row, fallback

        class _Sel:
            def __init__(self, row, cell, word): self.row, self.cell, self.word = row, cell, word

        class _StubProvider:                                                       # the provider face's surface ef2_w4 uses: select / describe / call_precision / triangle_multiplication
            Refusal = _Refusal
            STOCK_ROWS = ("cueq", "torch_math"); EXACT_ROWS = ("cueq", "torch_math", "native_exact"); TIER_WORDS = ("fast", "exact", "big"); ROW_NAMES = ("native", "torch_math")
            calls = 0; words = []; prefers = []

            @staticmethod
            def call_precision(z): return ("bf16", z.dtype)

            @staticmethod
            def select(cc, dtype, c_z, c_hidden, n_tokens, direction, *, word, prefer=None, **kw):   # the stub cell table: `exact` -> the stock row everywhere; else a kernel row from 101 tokens;
                row = "torch_math" if (word == "exact" or n_tokens < 101) else ("native" if n_tokens >= 300 else "tx_sm90a")   # this stub cell measures no preferred row at >= 300 tokens
                return _Sel(row, f"{cc}|bf16|C256|H256|N<={n_tokens}|{direction[:3]}|fwd", word)

            @staticmethod
            def describe(sel): return f"trimul row={sel.row} word={sel.word} cell={sel.cell}"

            @classmethod
            def triangle_multiplication(cls, z, mask, direction, weights, word, residual, cache, prefer=None):
                cls.calls += 1; cls.words.append(word); cls.prefers.append(prefer)
                if mask is not None and not bool(mask.all()):                       # a call the word's chain refuses with the tensors in hand (stub: any masked pair)
                    raise _Refusal("stub_mask", "native", "torch_math")
                n = int(z.shape[1]); cache["_last"] = _Sel("native" if n >= 300 else "tx_sm90a", "cell", word); return z + 1.0   # a recognisable output: the stub 'served'

        C, D = 256, 256
        mk = lambda: dict(norm_in_weight=torch.ones(C), norm_in_bias=torch.zeros(C), p_in_weight=torch.zeros(2 * D, C), g_in_weight=torch.zeros(2 * D, C),   # noqa: E731
                          norm_out_weight=torch.ones(C), norm_out_bias=torch.zeros(C), p_out_weight=torch.zeros(C, D), g_out_weight=torch.zeros(C, C))
        saved = dict(W._TX), dict(W.STATS), dict(W._STATE)
        try:
            W._STATE["tx"] = True                                                       # the lever is bound in this process (enable_tx sets it; this test binds the stub provider by hand)
            for word in ("fast", "big"):
                _StubProvider.words = []; _StubProvider.prefers = []
                W._TX.clear(); W._TX.update(saved[0])
                W._TX.update(on=True, bound=True, mod=_StubProvider, word=word, cc="9.0", stack="H100:stub", abi="stub-abi", has_cueq=False, classes={}, rows={}, refused={}, printed=set(), stock_classes=0)
                w = W._tx_pack(**mk())
                self.assertEqual(tuple(w["tx_w"]["w_ag"].shape), (D, C)); self.assertEqual(tuple(w["tx_w"]["w_bp"].shape), (D, C)); self.assertEqual(w["tx_w"]["ln_in_w"].dtype, torch.float32)
                self.assertEqual(w["tx_w"]["w_o"].dtype, torch.bfloat16); self.assertEqual((w["C"], w["CH"]), (C, D))
                small = torch.zeros(1, 20, 20, C); big = torch.zeros(1, 128, 128, C); huge = torch.zeros(1, 300, 300, C)
                holed = torch.ones(128, 128, dtype=torch.bool); holed[0, 1] = False
                self.assertIsNone(W._tx_forward(small, "outgoing", torch.ones(20, 20, dtype=torch.bool), w))   # the cell names the stock op for this class: None = the upstream module by name, no provider call
                self.assertIsNone(W._tx_forward(small, "incoming", None, w))                                    # another class of the same kind: its own `tx:` line
                o3 = W._tx_forward(big, "outgoing", None, w)                                                    # a kernel row serves, by the TIER word
                o4 = W._tx_forward(huge, "incoming", None, w)
                self.assertIsNone(W._tx_forward(big, "outgoing", holed, w))                                     # the chain refuses this call with the tensors in hand: named once, upstream serves
                self.assertIsNone(W._tx_forward(big, "outgoing", holed, w))                                     # same kind again: counted, not re-printed
                for o, z in ((o3, big), (o4, huge)):
                    self.assertTrue(torch.equal(o, z + 1.0)); self.assertEqual(o.shape, z.shape)
                self.assertEqual(_StubProvider.words, [word] * 4, word)                                         # served by the TIER word, one provider call each (two refused)
                # w4.19: under `big` the eager calls of classes N <= 512 carry prefer=("tx_sm90a",) WHEN the cell serves that row (N=128 here); the stub cell measures no
                # tx_sm90a at N=300 -> the word alone (prefer None on the serving call); `fast` never carries a preference
                self.assertEqual(_StubProvider.prefers, ([("tx_sm90a",), None, ("tx_sm90a",), ("tx_sm90a",)] if word == "big" else [None] * 4), word)
                st = W.tx_state()
                self.assertEqual(st["word"], word); self.assertEqual(st["rows"], {"tx_sm90a": 1, "native": 1})
                self.assertEqual(st["prefer_rule"], "tx_sm90a:eager:N<=512" if word == "big" else None)
                self.assertEqual({(c["N"], c["prefer"], c["prefer_asked"]) for c in st["classes"]},
                                 ({(20, None, "tx_sm90a"), (128, "tx_sm90a", "tx_sm90a"), (300, None, "tx_sm90a")} if word == "big" else {(20, None, None), (128, None, None), (300, None, None)}), word)
                self.assertEqual(st["refused"], {"stub_mask": 2}); self.assertEqual(W._TX["printed"], {"stub_mask"})
                self.assertEqual([(c["N"], c["direction"], c["row"], c["cell_row"]) for c in st["classes"]],
                                 [(20, "incoming", None, "torch_math"), (20, "outgoing", None, "torch_math"), (128, "outgoing", "tx_sm90a", "tx_sm90a"), (300, "incoming", "native", "native")])
                self.assertEqual(st["stock_classes"], 2); self.assertEqual(st["class_stock_calls"], 2)
                self.assertIn("_last", w["tx_cache"])
                self.assertTrue(W.describe()["tx"]); self.assertEqual(W.describe()["tx_state"]["word"], word); self.assertEqual(W.describe()["tx_state"]["stack"], "H100:stub")
            _StubProvider.words = []                                                # the exact word: the stub's table vouches no exact-class kernel row -> every class is the upstream module's, no provider call
            W._TX.update(word="exact", classes={}, rows={}, refused={}, printed=set(), stock_classes=0)
            w = W._tx_pack(**mk())
            for z in (torch.zeros(1, 128, 128, C), torch.zeros(1, 300, 300, C)):
                self.assertIsNone(W._tx_forward(z, "outgoing", None, w)); self.assertIsNone(W._tx_forward(z, "incoming", None, w))
            self.assertEqual(_StubProvider.words, []); self.assertEqual(W.tx_state()["stock_classes"], 4)
            self.assertTrue(all(c["row"] is None and c["cell_row"] == "torch_math" for c in W.tx_state()["classes"]))
        finally:
            W._TX.clear(); W._TX.update(saved[0]); W._STATE.clear(); W._STATE.update(saved[2])

    def test_enable_tx_without_cuda_binds_and_defers(self):
        import torch
        W = _ef2_w4(self)
        if torch.cuda.is_available():
            self.skipTest("CPU path only")
        saved = dict(W._STATE)
        try:
            st = W.enable_tx("fast")                                                   # the kit's no-device FORCE path: bound, provider selection + canary deferred, named -> applied
            self.assertFalse(st["on"]); self.assertTrue(st["bound"]); self.assertTrue(st["installed"]); self.assertTrue(st["why"].startswith("no_cuda_device")); self.assertTrue(W.describe()["tx"])
            self.assertEqual(st["word"], "fast"); self.assertTrue(W._STATE["enabled"]); self.assertTrue(W._STATE["tx"])
            st = W.enable_tx("no_such_word")                                           # an unknown word: off by name, the upstream fused TriMul serves
            self.assertFalse(st["on"]); self.assertFalse(W.describe()["tx"]); self.assertIn("unknown word", st["why"])
            self.assertEqual(W.enable_tx("big")["word"], "big"); self.assertEqual(W.enable_tx("exact")["word"], "exact")
        finally:
            W.disable(); W._STATE.clear(); W._STATE.update(saved); W._TX.update(on=False, bound=False, why=None)

    def test_provider_face_resolves_the_tier_words_at_this_kits_cells(self):
        """CONTRACT, not the cell table's winner of the day: the shared core's pure selection resolves each tier word at this kit's shape (c 256, bf16
        forward), for every size class the trunk issues, on class 9.0 AND class 8.0, to SOME row it names with a cell and a describe() line: `fast` /
        `big` -> a member of ROW_NAMES (a kernel row or, where the table measures the stock op fastest, a stock row -- ef2_w4 serves the upstream
        module by name there); `exact` -> a stock row (= the upstream statement by name) or an EXACT_ROWS member (a vouched exact-class kernel row).
        Which row a size class gets is the core's cell decision and is deliberately NOT asserted.  ef2_w4 uses exactly this surface."""
        from opt_core.kernels import trimul as TRI
        for n in ("TIER_WORDS", "ROW_NAMES", "STOCK_ROWS", "EXACT_ROWS", "select", "describe", "call_precision", "stack_word", "tx_abi_tag", "cueq_present", "triangle_multiplication", "Refusal"):
            self.assertTrue(hasattr(TRI, n), n)
        for word in ("exact", "fast", "big"):
            self.assertIn(word, TRI.TIER_WORDS)
        for cc, stack in (("9.0", "H100:2.13.0+cu130/3.7.1/nocueq"), ("8.0", "A100:2.13.0+cu130/3.7.1/nocueq")):
            for word in ("exact", "fast", "big"):
                for N in (20, 128, 400, 800, 1200, 1536):
                    for d in ("outgoing", "incoming"):
                        sel = TRI.select(cc, "bf16", 256, 256, N, d, word=word, stack=stack, has_cueq=False)
                        self.assertIn(sel.row, TRI.ROW_NAMES, (cc, word, N, d)); self.assertTrue(sel.cell, (cc, word, N, d)); self.assertIn("word=" + word, TRI.describe(sel))
                        if word == "exact":
                            self.assertIn(sel.row, tuple(TRI.EXACT_ROWS) + tuple(TRI.STOCK_ROWS), (cc, N, d, TRI.describe(sel)))   # a stock row (the upstream statement by name) or a vouched exact-class row
                        if cc == "8.0":
                            self.assertNotIn(sel.row, ("tx_sm90a", "tx_sm90a_exact"), TRI.describe(sel))   # never an sm_90a object on class 8.0
        with self.assertRaises(TRI.Refusal):
            TRI.select("8.0", "bf16", 256, 256, 800, "outgoing", word="tx_sm90a")


if __name__ == "__main__":
    unittest.main()
