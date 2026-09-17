"""The fused blocks (L10 -fused_triattn, L11 -fused_trimul) and flash attention (L8 -flash_attn) when the block CANNOT RUN in the process —
the serve layer's refusal at setup: no tile rows for the GPU's compute capability (`no_tiles`), a table without float32 rows, jax below the
Pallas floor, Pallas / the kernel absent, a backend that is not gpu. A mode is all of its levers: the adapter prints its LEVER line
(`state=skipped reason=cannot_run:<why>(<block>)`) and the kit's `NOT ACTIVE: lever <id> (<flag>) cannot run in this process: …` line, and ends
the driver with status 3 before any design — never a run under the mode's name with a subset, nothing rebound. An exception that names no kind
is re-raised as is. With tile rows nothing changes. No jax, no GPU."""
import contextlib, io, json, os, re, tempfile, unittest

from af2ig_opt import fused_trimul, registry, stack, triattn


class Refusal(RuntimeError):
    def __init__(self, kind, detail=""):
        self.kind, self.detail = kind, detail
        super().__init__(f"fpf_pallas: {kind}" + (f" — {detail}" if detail else ""))


class StubServe:
    """The serve layer's surface the adapters use at setup, on a part whose compute capability has no tile table."""
    Refusal = Refusal
    NO_TILES = "no_tiles"
    F32_DTYPE = "float32"
    def __init__(self, kind="no_tiles", cc="8.0", f32_rows=True):
        self.kind, self.cc, self.f32 = kind, cc, f32_rows
    def compute_capability(self):
        return self.cc
    def require(self, require_gpu=True):
        if self.kind is None:
            return {"ok": True, "cc": self.cc, "tiles": f"own:{self.cc}", "jax": "0.10.2", "backend": "gpu", "own_table": True}
        raise Refusal(self.kind, f"no-tiles:{self.cc} (tile tables: ['10.0', '10.3', '9.0'])" if self.kind == "no_tiles" else "jax.default_backend() = 'cpu'")
    def required_multiple(self, kind, n=None, cc=None, dtype="bfloat16"):
        if str(getattr(dtype, "name", dtype)) == "float32" and not self.f32:
            raise Refusal("no_tiles", f"no-f32-tiles:{self.cc} ({'attn' if kind == 'triattn' else kind}; f32 rows: ['9.0'])")
        return 64
    def f32_precision(self, word):
        return word


def timers(records):
    d = tempfile.mkdtemp(); p = os.path.join(d, "timers.jsonl")
    with open(p, "w") as fh:
        for r in records: fh.write(json.dumps(r) + "\n")
    return p


class CardFallback(unittest.TestCase):
    def setUp(self):
        for m in (triattn, fused_trimul):
            m._STATE.update(orig=None, patches=None, serve=None, ledger=None, probe=None)

    tearDown = setUp

    def refuses(self, fn, lever, word):
        """`fn()` ends the process by name: SystemExit(3) after the LEVER skipped line and the NOT ACTIVE line naming `lever` and `word`."""
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            fn()
        self.assertEqual(cm.exception.code, 3)
        err = buf.getvalue()
        self.assertRegex(err, rf"\[af2ig-opt\] LEVER name=\S+ state=skipped reason={re.escape(word)} .* lever={lever} ")   # the driver-side line: name = the kernel ledger's, lever = the kit id
        self.assertIn(f"[af2ig-opt] NOT ACTIVE: lever {lever} (", err); self.assertIn(f") cannot run in this process: {word} — ", err)
        self.assertIn("a mode is all of its levers: select a mode without", err)
        return err

    def test_no_tiles_refuses_the_mode_by_name(self):
        for m, lever, word in ((triattn, registry.L10, "cannot_run:no_tiles_cc80(triattn)"), (fused_trimul, registry.L11, "cannot_run:no_tiles_cc80(trimul)")):
            m._STATE["serve"] = StubServe()
            self.refuses(m.setup, lever, word)
            self.assertIsNone(m._STATE["patches"])                                            # nothing was rebound
            self.assertIsNone(m._STATE["probe"])

    def test_a_table_without_f32_rows_refuses_by_name(self):
        """A part WITH a bf16 tile table and no float32 rows (cc 10.0 / 10.3 in the core's tables): the f32 blocks cannot run there."""
        for m, lever, kind in ((triattn, registry.L10, "triattn"), (fused_trimul, registry.L11, "trimul")):
            m._STATE["serve"] = StubServe(kind=None, cc="10.0", f32_rows=False)
            self.refuses(m.setup, lever, f"cannot_run:no_f32_tiles_cc100({kind})")
            self.assertIsNone(m._STATE["patches"])

    def test_any_other_named_refusal_refuses_by_name(self):
        for m, lever, kind in ((triattn, registry.L10, "triattn"), (fused_trimul, registry.L11, "trimul")):
            m._STATE["serve"] = StubServe(kind="backend_not_gpu")
            self.refuses(m.setup, lever, f"cannot_run:backend_not_gpu({kind})")

    def test_a_kindless_exception_is_reraised(self):
        class Odd(StubServe):
            def require(self, require_gpu=True):
                raise Refusal("", "no kind")
        triattn._STATE["serve"] = Odd()
        with self.assertRaises(Refusal):
            triattn.setup()
        self.assertIsNone(triattn._STATE["probe"])

    def test_a_table_with_f32_rows_installs(self):
        """With a table and its float32 rows present: setup() proceeds to install (stubbed out here) and the probe is the serve layer's."""
        for m in (triattn, fused_trimul):
            m._STATE["serve"] = StubServe(kind=None, cc="9.0", f32_rows=True)
            saved = m.install; m.install = lambda: None
            try:
                probe = m.setup()
            finally:
                m.install = saved
            self.assertTrue(probe["ok"]); self.assertNotIn("reason=cannot_run", m.line())

    def test_a_served_record_is_applied(self):
        recs = [{"kind": "proc_start", "argv": ["predict_pdb.py", registry.LEVERS[registry.L10].switch]},
                {"kind": "fused_triattn", "served": 96, "fallback": 0, "fallback_by": {}, "impl": "fpf_pallas_f32", "origin": "core", "precision": "tf32", "tiles": "own:9.0"}]
        got = stack.applied({"levers_planned": [registry.L10], "mode": "fast"}, timers(recs))
        self.assertEqual((got["partial"], got["levers_applied"]), ([], [registry.L10]))

    def test_a_kernel_error_in_a_fused_census_is_partial(self):
        """One fail-closed rule for the kernel levers: an `errors` entry of the census record is an undeclared event like an undeclared fallback reason."""
        recs = [{"kind": "proc_start", "argv": ["predict_pdb.py", registry.LEVERS[registry.L10].switch]},
                {"kind": "fused_triattn", "served": 96, "fallback": 0, "fallback_by": {}, "errors": {"XlaRuntimeError": 1}, "impl": "fpf_pallas_f32", "origin": "core"}]
        got = stack.applied({"levers_planned": [registry.L10], "mode": "fast"}, timers(recs))
        self.assertEqual(got["partial"], [registry.L10]); self.assertIn("'error:XlaRuntimeError': 1", " ".join(got["partial_reasons"].values()))

    def flash_records(self, **census):
        rec = dict({"kind": "flash_attn", "served": 5, "fallback": 0, "fallback_by": {}, "errors": {}, "impl": "pallas_attn", "origin": "core", "min_tokens": 0, "precision": "tf32"}, **census)
        return [{"kind": "proc_start", "argv": ["predict_pdb.py", registry.LEVERS[registry.L8].switch]}, rec]

    def test_flash_attn_declared_fallbacks_are_applied(self):
        """-flash_attn (L8): the declared reasons of a healthy run (no pair bias, per-head width < 16, below the size rule — flash_attn.EXPECTED_FALLBACKS over the
        serve layer's words) leave the lever applied."""
        from af2ig_opt import flash_attn
        F1 = flash_attn._serve()
        self.assertEqual(stack._flash_expected(), (F1.BELOW_SIZE_RULE, F1.NO_PAIR_BIAS, F1.HEAD_DIM_LT_16))
        got = stack.applied({"levers_planned": [registry.L8], "mode": "fast"}, timers(self.flash_records(fallback=20, fallback_by={F1.NO_PAIR_BIAS: 4, F1.HEAD_DIM_LT_16: 16})))
        self.assertEqual((got["partial"], got["levers_applied"]), ([], [registry.L8]), got.get("partial_reasons"))
        self.assertIn("5 Attention call(s) traced onto the kernel pallas_attn", got["evidence"][registry.L8]); self.assertIn("reasons: head_dim_lt_16 x16, no_pair_bias x4", got["evidence"][registry.L8])

    def test_flash_attn_undeclared_fallback_is_partial(self):
        """-flash_attn (L8) is fail-closed like L10/L11: an Attention call left on the stock ops for a reason OUTSIDE the declared set (here the serve layer's
        `bias_form`) makes the lever partial even though other calls were served — never `on` over an undeclared reason."""
        got = stack.applied({"levers_planned": [registry.L8], "mode": "fast"}, timers(self.flash_records(fallback=1, fallback_by={"bias_form": 1})))
        self.assertEqual(got["partial"], [registry.L8]); self.assertEqual(got["levers_applied"], [])
        self.assertRegex(got["partial_reasons"][registry.L8], r"^Attention call\(s\) left on the stock ops for undeclared reason\(s\) / kernel error\(s\) \{'bias_form': 1\} \(5 served\)")
        got = stack.applied({"levers_planned": [registry.L8], "mode": "fast"}, timers(self.flash_records(fallback=5, fallback_by={"no_pair_bias": 4, "kernel_unavailable": 1})))
        self.assertEqual(got["partial"], [registry.L8]); self.assertIn("{'kernel_unavailable': 1}", " ".join(got["partial_reasons"].values()))          # the declared reason beside it is not listed
        got = stack.applied({"levers_planned": [registry.L8], "mode": "fast"}, timers(self.flash_records(errors={"XlaRuntimeError": 2})))
        self.assertEqual(got["partial"], [registry.L8]); self.assertIn("{'error:XlaRuntimeError': 2}", " ".join(got["partial_reasons"].values()))

    def test_flash_attn_unreadable_declared_set_is_fail_closed(self):
        """A core whose serve layer is not importable in THIS process leaves L8's declared set unreadable: every fallback reason counts as undeclared and the reason says why."""
        from af2ig_opt import flash_attn
        saved = dict(flash_attn._STATE)
        def absent():
            raise RuntimeError("-flash_attn: opt_core.kernels.pallas_attn_serve is not importable from this opt_core (test)")
        orig = flash_attn._serve; flash_attn._serve = absent
        try:
            self.assertEqual(stack._flash_expected(), ())
            got = stack.applied({"levers_planned": [registry.L8], "mode": "fast"}, timers(self.flash_records(fallback=4, fallback_by={"no_pair_bias": 4})))
        finally:
            flash_attn._serve = orig; flash_attn._STATE.clear(); flash_attn._STATE.update(saved)
        self.assertEqual(got["partial"], [registry.L8]); self.assertIn("the declared set is unreadable in this process", " ".join(got["partial_reasons"].values()))

    def test_flash_attn_refuses_by_name_when_the_kernel_cannot_serve(self):
        """-flash_attn (L8): the serve layer's refusal at setup (here: jax below the Pallas floor) is the same named refusal of the mode."""
        from af2ig_opt import flash_attn
        class StubF1:
            Refusal = Refusal
            BELOW_SIZE_RULE, NO_PAIR_BIAS, HEAD_DIM_LT_16 = "below_size_rule", "no_pair_bias", "head_dim_lt_16"
            def kernel_origin(self): return "core"
            def ledger(self, min_tokens=0, expected=(), origin=None):
                from opt_core.counters import Ledger
                return Ledger("pallas_attn", impl="pallas_attn", origin=origin or "core", expected=tuple(expected))
            def require(self, require_gpu=True): raise Refusal("jax_older_than_floor", "jax 0.4.20 < 0.5.0")
        saved = dict(flash_attn._STATE)
        try:
            flash_attn._STATE.update(serve=StubF1(), op=None, ledger=None, gate=None)
            self.refuses(flash_attn.setup, "L8", "cannot_run:jax_older_than_floor(pallas_attn)")
            self.assertIsNone(flash_attn._STATE["op"])
        finally:
            flash_attn._STATE.clear(); flash_attn._STATE.update(saved)
