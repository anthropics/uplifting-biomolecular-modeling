"""The package's implied levers (modes.OPTIN_LEVERS: tf32, alloc; a row carries them as KitMode.implied_optin — there is no request
surface beside a mode): under `fast` tf32 is applied (precision.py), probed from torch's own flag and named on the ACTIVE line (`optin=tf32,alloc`,
`levers_applied=...,tf32`) and in the manifest; under `exact` and `off` it is refused by name; an unknown name is refused by name; the dry run
names the levers; one LEVER line per lever. Exercised through the internal stack.activate(optin=...)."""
import contextlib
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout

import chai1_opt
from chai1_opt import cli, modes, ngpu, precision, registry, report, stack
from chai1_opt.tests import _stubs


class TestOptIn(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests()
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)
        self.eager_saved, self.eager = _stubs.eager_stub(stack)
        self.err = io.StringIO()

    def tearDown(self):
        _stubs.eager_restore(stack, self.eager_saved)
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)
        stack.reset_for_tests()

    def test_off_unless_requested_or_implied(self):
        with redirect_stderr(self.err):
            rep = chai1_opt.enable("fast")
        self.assertTrue(rep["active"], rep)
        self.assertEqual(rep["optin"], ["tf32", "alloc"]); self.assertEqual(rep["optin_applied"]["tf32"], True)   # fast's row implies tf32 + the allocator policy (KitMode.implied_optin)
        self.assertIn("tf32", rep["levers_applied"]); self.assertIn("alloc", rep["levers_applied"])
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")   # exported into this process before the eager stack installed
        self.assertTrue(self.torch.backends.cuda.matmul.allow_tf32)
        self.assertIn("optin=tf32,alloc", self.err.getvalue())
        self.assertEqual(precision.applied(self.torch), ("tf32",))

    def test_fast_optin_tf32_applies_probes_and_reports(self):
        with redirect_stderr(self.err):
            rep = stack.activate("fast", optin="tf32")
        self.assertTrue(rep["active"], rep)
        self.assertEqual(rep["optin"], ["tf32", "alloc"]); self.assertEqual(rep["optin_applied"]["tf32"], True)
        self.assertTrue(self.torch.backends.cuda.matmul.allow_tf32)                      # torch's own flag holds it
        self.assertEqual(rep["levers_applied"], ["W1", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "tf32", "alloc"]); self.assertFalse(rep["partial"])
        self.assertIn("[chai1-opt] ACTIVE mode=fast levels=W1,W2,W5 eager=tier1 dstep=hoist2,compiled,dit_attn optin=tf32,alloc levers_applied=W1,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,tf32,alloc not_applicable=W2,prefetch", self.err.getvalue())
        self.assertEqual(precision.applied(self.torch), ("tf32",)); self.assertEqual(rep["optin_applied"]["tf32_policy"]["policy"], "tf32"); self.assertEqual(rep["optin_applied"]["tf32_policy"]["changed"], ["matmul", "matmul_tf32"])
        self.assertIsNotNone(stack._STATE["eager"])                                       # the stack installed after the flag (graph capture reads it)

    def test_optin_accepts_lists_and_dedups(self):
        self.assertEqual(modes.parse_optin("tf32, tf32"), ("tf32",)); self.assertEqual(modes.parse_optin(["tf32"]), ("tf32",))
        self.assertEqual(modes.parse_optin(None), ()); self.assertEqual(modes.parse_optin(""), ())

    def test_exact_refuses_tf32_by_name_and_applies_nothing(self):
        with redirect_stderr(self.err):
            rep = stack.activate("exact", optin="tf32")
        self.assertFalse(rep["active"]); self.assertIn("opt-in lever tf32 does not join mode exact", rep["reason"])
        self.assertFalse(self.torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(self.chai1.load_exported.__name__, "load_exported")              # refused before the kit's statements ran
        self.assertIn("NOT ACTIVE: opt-in lever tf32 does not join mode exact", self.err.getvalue())
        with self.assertRaises(chai1_opt.ActivationError):
            stack.activate("exact", optin="tf32", strict=True)

    def test_unknown_optin_is_refused_by_name(self):
        with redirect_stderr(self.err):
            rep = stack.activate("fast", optin="bf16")
        self.assertFalse(rep["active"]); self.assertIn("unknown opt-in lever(s) bf16", rep["reason"])
        self.assertIsNone(stack._STATE["eager"])

    def test_dry_run_names_the_request(self):
        with redirect_stderr(self.err):
            rep = stack.activate("fast", dry_run=True, optin="tf32")
        self.assertIsNone(rep["reason"]); self.assertEqual(rep["optin"], ["tf32", "alloc"])
        self.assertIn("DRY-RUN mode=fast levels=W1,W2,W5 eager=tier1 dstep=hoist2,compiled,dit_attn optin=tf32,alloc levers_applied=W1,W2,W5,tier1,hoist2,compiled,dit_attn,templ_empty,v4trimul,exactln,triattn,msa_pad,transition,trunk_n,rankcc,tailasync,confmemo,prefetch,tf32,alloc", self.err.getvalue())
        self.assertIsNone(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))                      # a dry run exports nothing
        self.assertFalse(self.torch.backends.cuda.matmul.allow_tf32)                     # a dry run applies nothing

    def test_registry_describes_every_optin_lever(self):
        want = {"tf32": ("flag", "chai1_opt/precision.py", "F4"), "alloc": ("env", "chai1_opt/alloc.py", "F7")}
        for n in modes.OPTIN_LEVERS:
            lv = registry.LEVERS[n]
            self.assertEqual((lv.probe[0], lv.kit_file, registry.FAMILY[n]), want[n]); self.assertFalse(lv.default_on)
        self.assertNotIn("CHAI1_OPT_OPTIN", stack.DECLARED_ENV)                          # no request surface: a row's implied levers only

    def test_lever_lines_one_per_lever(self):
        with redirect_stderr(self.err):
            rep = stack.activate("fast", optin="tf32")
        lines = [l for l in self.err.getvalue().splitlines() if l.startswith("[chai1-opt] LEVER ")]
        self.assertEqual(len(lines), len(registry.LEVERS), lines)                      # one line per lever the house composes or carries
        by = {l.split("name=")[1].split(" ")[0]: l for l in lines}
        self.assertEqual(by["tf32"], "[chai1-opt] LEVER name=tf32 state=on impl=chai1_opt/precision.py origin=kit strategy=F4.autocast_policy family=F4 mode=fast route=in-process")
        self.assertIn("[chai1-opt] NUMERICS matmul=high matmul_tf32=True cudnn_tf32=True", self.err.getvalue()); self.assertEqual(rep["numerics"]["matmul_tf32"], True)
        self.assertIn("name=tier1 state=on impl=chai1_eager/stack.py origin=kit strategy=LOCAL.chai1_ts2eager family=F3", by["tier1"]); self.assertIn("name=W1 state=on impl=kit/chai_worker.py origin=kit strategy=F6.weights_residency_init family=F6", by["W1"])
        self.assertIn("name=W2 state=skipped reason=not_applicable_on_route:in-process", by["W2"])
        self.assertEqual(set(by), set(registry.LEVERS)); self.assertNotIn("W4", by); self.assertNotIn("lin3xT", by)   # no line for a lever the house does not carry
        self.assertIn("name=alloc state=on impl=chai1_opt/alloc.py", by["alloc"])
        self.assertEqual(report.lever_state("tf32", rep), ("on", None))

    def test_lever_lines_on_a_refusal_name_the_refusal(self):
        with redirect_stderr(self.err):
            rep = stack.activate("exact", optin="tf32")
        lines = [l for l in self.err.getvalue().splitlines() if l.startswith("[chai1-opt] LEVER ")]
        self.assertEqual(len(lines), len(registry.LEVERS))
        w1 = [l for l in lines if "name=W1 " in l][0]
        self.assertIn("state=skipped reason=activation_refused", w1)

    def test_no_lever_lines_on_a_dry_run(self):
        with redirect_stderr(self.err):
            stack.activate("fast", dry_run=True)
        self.assertNotIn(" LEVER ", self.err.getvalue())

    def test_alloc_refused_by_name_when_the_core_has_no_mem_module_else_exports(self):
        from chai1_opt import alloc
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            have_core = True
            try:
                from chai1_opt import _core; _core.ensure_importable()
                import opt_core.mem.torch_alloc  # noqa: F401
            except ImportError:
                have_core = False
            with redirect_stderr(self.err):
                rep = stack.activate("exact", optin="alloc")
            if not have_core:
                self.assertFalse(rep["active"]); self.assertIn("opt-in lever alloc needs opt_core.mem.torch_alloc, absent from the installed core", rep["reason"])
                self.assertFalse(alloc.applied())
            else:
                self.assertTrue(rep["active"], rep); self.assertIn("alloc", rep["levers_applied"])
                self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True"); self.assertTrue(alloc.applied())
        finally:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

class TestWords(unittest.TestCase):
    """The ``--n_gpu P`` axis (ngpu.py over opt_core.mem.ngpu): P ∈ {1} on chai1. The ACTIVE / DRY-RUN line carries exactly
    ``n_gpu=1 sharding=none`` (callers grep these tokens); ``--n_gpu`` absent == ``--n_gpu 1``; P>1 under exact / fast is the
    core's mode refusal, P>1 under big is chai1's not-applicable refusal, both exit 3 with the sentence on the NOT ACTIVE line; a malformed
    value is usage (exit 2)."""
    def test_supported_set_and_tokens(self):
        self.assertEqual(ngpu.SUPPORTED, (1,))
        self.assertEqual(ngpu.active_fields(1), "n_gpu=1 sharding=none")                 # the exact token text (P=1)
        self.assertEqual(ngpu.active_pairs(1), [("n_gpu", 1), ("sharding", "none")])
        self.assertEqual(ngpu.core().active_fields(2, ngpu.SCHEME), "n_gpu=2 sharding=rowpair")   # what a P>1 kit prints (the core's words)

    def test_resolve(self):
        for mode in modes.MODES:
            self.assertEqual(ngpu.resolve(None, mode), 1)
            self.assertEqual(ngpu.resolve("1", mode), 1)
            self.assertEqual(ngpu.resolve(1, mode), 1)
        self.assertEqual(ngpu.refusal("2", "fast"), "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)")
        self.assertEqual(ngpu.refusal("2", "exact"), "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)")
        self.assertEqual(ngpu.refusal("2", "big"), "refused: n_gpu>1 not applicable on chai1: the trunk's pair stack is shardable but pair init, the "
                         "diffusion module and the confidence head are sealed TorchScript exports that run whole per rank, and the model input "
                         "limit of 2048 tokens fits one card under --mode big (row sharding buys no reach)")
        self.assertEqual(ngpu.refusal("8", "big"), ngpu.REFUSE_NOT_APPLICABLE)
        for bad in ("0", "-1", "x", "1.5", ""):
            with self.assertRaises(ValueError):
                ngpu.resolve(bad, "big")


class TestActivationLine(unittest.TestCase):
    def test_tokens_on_the_active_and_dry_run_lines(self):
        rep = {"active": True, "mode": "fast", "levels": ["W1", "W2", "W5"], "eager": "tier1", "dstep": ("hoist2", "compiled", "dit_attn"), "levers_applied": ["W1"],
               "det": 0, "route": "driver", "chai_lab_version": "0.6.1", "torch_version": "2.13.0",
               "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": (9, 0)}, "hoist": {"keyed": "item", "release": "trunk_boundary"}, "n_gpu": 1}
        line = report.activation_line(rep)
        self.assertIn(" n_gpu=1 sharding=none", line)
        self.assertLess(line.index(" gpu="), line.index(" n_gpu="))                        # after the fixed head of the line
        rep["big"] = {"line": "lean", "levers": ["msa_rows"], "exact": "measured"}; rep["mode"] = "big"
        line = report.activation_line(rep)
        self.assertLess(line.index(" big="), line.index(" n_gpu="))
        rep["active"] = False
        line = report.activation_line(rep, dry_run=True)
        self.assertTrue(line.startswith("[chai1-opt] DRY-RUN mode=big "), line)
        self.assertIn(" n_gpu=1 sharding=none", line)


class TestNgpuCli(unittest.TestCase):
    def setUp(self):
        self.saved = _stubs.install()[-1]

    def tearDown(self):
        _stubs.remove(self.saved)

    def run_cli(self, argv):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = cli.main(argv)
        return rc, err.getvalue(), out.getvalue()

    def test_check_refuses_p2_by_name_under_every_mode(self):
        for mode, sentence in (("exact", ngpu.core().REFUSE_MODE), ("fast", ngpu.core().REFUSE_MODE), ("big", ngpu.REFUSE_NOT_APPLICABLE)):
            rc, err, _ = self.run_cli(["check", "--mode", mode, "--n_gpu", "2"])
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, (mode, err))
            self.assertIn(f"[chai1-opt] NOT ACTIVE: {sentence}", err)

    def test_check_json_names_the_refusal(self):
        rc, err, out = self.run_cli(["check", "--mode", "big", "--n_gpu", "4", "--json"])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE)
        self.assertEqual(json.loads(out)["reason"], ngpu.REFUSE_NOT_APPLICABLE)

    def test_malformed_value_is_usage(self):
        rc, err, _ = self.run_cli(["check", "--mode", "fast", "--n_gpu", "0"])
        self.assertEqual(rc, cli.EXIT_USAGE, err)

    def test_pred_refuses_before_touching_inputs(self):
        rc, err, _ = self.run_cli(["pred", "--mode", "fast", "--n_gpu", "2", "--input", "/nonexistent.fasta", "--out_dir", "/tmp/x"])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err)
        self.assertIn("requires --mode big", err)


if __name__ == "__main__":
    unittest.main()


class TestTrimulBindingWords(unittest.TestCase):
    """v4trimul's LEVER / EXIT words name HOW the lever is bound: the shared core's triangle-multiplication provider BY TIER WORD (strategy
    F2.trimul; `binding=by_word word=<tier> row_<dir>=trimul:<row> cell_<dir>=<word>/<cell>` from the provider object, `pending` before a
    direction decided a call); the tier word per kit mode (fast -> fast, big -> big: big never shares fast's word)."""
    def setUp(self):
        from chai1_opt import pairtrack
        self.pt = pairtrack; self.saved = pairtrack._STATE["trimul"]

    def tearDown(self):
        self.pt._STATE["trimul"] = self.saved

    class _Prov:
        def __init__(self, kernel="trimul", version=None): self.kernel, self.version = kernel, version     # opt_core.trimul.by_word: kernel 'trimul' until the first decided call
        def describe(self): return f"{self.kernel}@{self.version}"

    class _Lever:
        def __init__(self, prov, word): self.provider, self.chai1_word = prov, word

    def test_words_name_the_resolved_row_per_direction(self):
        P, L = self._Prov, self._Lever
        self.pt._STATE["trimul"] = (None, L(P("trimul:some_row", "fast/9.0|bf16|C256|H256|N<=512|out|fwd"), "fast"), L(P(), "fast"))
        self.assertEqual(self.pt.trimul_binding(), {"binding": "by_word", "word": "fast", "row_out": "trimul:some_row", "cell_out": "fast/9.0|bf16|C256|H256|N<=512|out|fwd",
                                                    "row_in": "pending", "cell_in": "pending"})

    def test_big_asks_the_big_word(self):
        P, L = self._Prov, self._Lever
        self.pt._STATE["trimul"] = (None, L(P(), "big"), L(P(), "big"))
        b = self.pt.trimul_binding()
        self.assertEqual((b["binding"], b["word"], b["row_out"], b["row_in"]), ("by_word", "big", "pending", "pending"))

    def test_tier_word_per_mode(self):
        self.assertEqual(self.pt.TRIMUL_WORDS, {"exact": "exact", "fast": "fast", "big": "big"})
        self.assertEqual([self.pt.trimul_word(m) for m in ("fast", "big", "exact")], ["fast", "big", "exact"])
        self.assertEqual(self.pt.TRIMUL_LADDER, {"exact": "exact", "fast": "fast", "big": "fast"})          # opt_core.trimul.Lever's ladder classes (stock | exact | fast)
        with self.assertRaises(self.pt.LeverUnavailable):
            self.pt.trimul_word("off")

    def test_levers_bind_the_provider_by_the_modes_word(self):
        """_trimul_levers builds the two core Levers over opt_core.trimul.by_word with the MODE's tier word (CPU: the provider is pure at
        construction; nothing is selected before a CUDA call arrives) — no kernel routed by name, no kit cell table read."""
        import opt_core.trimul as T
        seen = []
        real = T.by_word

        def spy(weights_of, word, **kw):
            seen.append((word, kw.get("cache_key")))
            return real(weights_of, word, **kw)
        T.by_word = spy
        try:
            for mode, word in (("fast", "fast"), ("big", "big")):
                seen.clear()
                _T, L_out, L_in = self.pt._trimul_levers(mode)
                self.assertEqual([w for w, _ in seen], [word, word])
                self.assertEqual(sorted(k for _, k in seen), sorted([f"F2.by_word.{word}.out", f"F2.by_word.{word}.in"]))
                self.assertEqual((L_out.mode, L_in.mode), ("fast", "fast"))                        # big is fast-class on the core ladder; the WORD differs
                self.assertEqual((L_out.chai1_word, L_out.provider.name, L_out.expected), (word, f"trimul:{word}", ()))
                self.assertEqual(L_out.min_tokens, 0)
        finally:
            T.by_word = real

    def test_not_installed_is_empty(self):
        self.pt._STATE["trimul"] = None
        self.assertEqual(self.pt.trimul_binding(), {})

    def test_no_kit_cell_table_for_trimul(self):
        """The kit carries no triangle-multiplication cell table and pairtrack reads none (the provider's TRIMUL_CELLS decide)."""
        import os
        cells = os.path.join(os.path.dirname(self.pt.__file__), "cells")
        self.assertFalse([f for f in (os.listdir(cells) if os.path.isdir(cells) else []) if "trimul" in f])
        self.assertFalse(hasattr(self.pt, "CELLS")); self.assertFalse(hasattr(self.pt, "trimul_row")); self.assertFalse(hasattr(self.pt, "TRIMUL_ROW_OVERRIDE"))


class TestTrimulTierWordsCoverChai1Cells(unittest.TestCase):
    """Class contract on the shared provider (CPU, pure): at Chai-1's pair-stack class (bf16, c_z = c_hidden = 256, both directions) the tier
    words `fast` and `big` resolve WITHOUT a refusal to a MEASURED cell of the provider's table on both cards' image stacks (cc 9.0 / 8.0,
    torch 2.13.0+cu130 / triton 3.7.1 / no cuequivariance) at every crop chai-lab folds — never a row asserted by name here."""
    STACKS = {"9.0": "H100:2.13.0+cu130/3.7.1/nocueq", "8.0": "A100:2.13.0+cu130/3.7.1/nocueq"}
    CROPS = (256, 384, 512, 768, 1024, 1536, 2048)

    def test_fast_and_big_words_resolve_to_measured_cells(self):
        from opt_core.kernels import trimul as KT
        for cc, stack in self.STACKS.items():
            for word in ("fast", "big"):
                for n in self.CROPS:
                    for d in ("outgoing", "incoming"):
                        sel = KT.select(cc, "bf16", 256, 256, n, d, word=word, stack=stack)
                        self.assertIsNotNone(sel.cell, (cc, word, n, d, sel.reason))               # a measured family serves the class (no UNCOVERED_CELL at chai-lab's crops)
                        self.assertNotIn(sel.row, KT.STOCK_ROWS, (cc, word, n, d, sel.reason))    # a kernel row, not the library / module-math statement
                        KT.admits(sel.row, cc, "bf16", 256, 256, n, residual=False, batch=1)      # the row admits the call's static envelope (raises Refusal by name otherwise)
