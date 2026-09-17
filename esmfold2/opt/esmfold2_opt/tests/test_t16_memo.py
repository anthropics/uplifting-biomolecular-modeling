"""t16 transition_cute.2.1: the provider glue resolves each call CLASS once -- in eager use the first call of a class (word, family, form,
residual, n_tokens, rows, device) goes through select + transition (the core's census record, the priming), the served Selection is kept, and
the class's later calls dispatch that same Selection directly: the SAME row as select names for the class, on every call (bytes unchanged),
without the two selections per call.  A class that resolves to the statement is kept as such (the module's own call, by name, without
re-selecting); a kept row that refuses a call is dropped and the class resolves afresh; EF2_T16_MEMO=0 = every call resolves (2.0 behaviour).
The memo is live under the `big` word only (transition_cute.2.2): fast / exact / row words keep the 2.0 path and LEVER tokens at every N.
CPU tests on a stub provider face (torch tensors on the host; no device, no forks)."""
import collections, os, sys, types, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
if DRV not in sys.path:
    sys.path.insert(0, DRV)


_LOADED_HERE = []                                        # the driver module this file imported (popped from sys.modules at class teardown: the package's EXIT tally reads
                                                         # sys.modules for lever modules, and a bare import with empty counters is not a process that ran the lever)


def _mod(tc):
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        tc.skipTest(f"torch not importable: {e!r}")
    fresh = "ef2_transition_cute" not in sys.modules
    import ef2_transition_cute as M
    if fresh and not _LOADED_HERE:
        _LOADED_HERE.append("ef2_transition_cute")
    return M


class _Refusal(Exception):
    def __init__(self, kind, word=None, fallback=None):
        super().__init__(kind); self.kind, self.word, self.fallback = kind, word, fallback


class _Sel:
    def __init__(self, row, variant=None, word=None):
        self.row, self.variant, self.word, self.cell_key, self.tier, self.words = row, variant, word, "stub|" + row, None, ()


class _Face:
    """The provider face's surface ef2_transition_cute uses: select / transition / _dispatch / STOCK_ROWS / Refusal.  The stub cell table: family
    esmpair -> esm_t16 everywhere; family pair -> v2:fast at N<=256, flash_sm90a:kernel_ln at N<=400, esm_t16 above; N < 32 -> the statement."""
    Refusal = _Refusal
    STOCK_ROWS = ("torch_swiglu", "engine_module")

    def __init__(self):
        self.selects, self.transitions, self.dispatches = [], [], []
        self.refuse_dispatch = set()

    def _row(self, family, n):
        if n < 32:
            return ("torch_swiglu", None)
        if family == "esmpair":
            return ("esm_t16", None)
        return ("v2", "fast") if n <= 256 else (("flash_sm90a", "kernel_ln") if n <= 400 else ("esm_t16", None))

    def select(self, word, *, c, hidden, n_tokens, dtype, direction, timing, family, residual, device, capture, rows_count, form):
        self.selects.append((word, family, form, residual, n_tokens, rows_count, timing))
        row, var = self._row(family, n_tokens)
        return _Sel(row, var, word)

    def transition(self, x2d, W, *, word, residual, n_tokens, family, form, timing, capture):
        self.transitions.append((word, family, n_tokens))
        row, _, var = word.partition(":")
        sel = _Sel(row, var or None, word)
        return self._serve(x2d, sel, residual), sel

    def _dispatch(self, x, W, sel, word, residual, x_ln, mask, binary_mask, out, form):
        self.dispatches.append((word, sel.row, residual, form))
        if word in self.refuse_dispatch:
            raise _Refusal("stub:refuses_now", word, "torch_swiglu")
        return self._serve(x, sel, residual), sel

    @staticmethod
    def _serve(x2d, sel, residual):                     # a recognisable per-row output: the row's index added
        k = {"esm_t16": 1.0, "v2": 2.0, "flash_sm90a": 3.0}[sel.row]
        return (x2d + k) if residual else (x2d * 0 + k)


class TestT16Memo(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for name in _LOADED_HERE:
            sys.modules.pop(name, None)

    def setUp(self):
        self.M = _mod(self)
        import torch
        self.torch = torch
        M = self.M
        self.saved = (dict(M._STATE), collections.Counter(M.STATS))
        self.face = _Face()
        M._STATE.update(face=self.face, word="big", memo={}, memo_on=None, rows=collections.Counter(), refused=collections.Counter(), printed=set())
        M.STATS.clear()
        self._pack = mock.patch.object(M, "_face_pack_for", lambda mod: "W"); self._pack.start()
        self.mod = types.SimpleNamespace()
        os.environ.pop(M.MEMO_ENV, None)

    def tearDown(self):
        M = self.M
        self._pack.stop()
        M._STATE.clear(); M._STATE.update(self.saved[0]); M._STATE["memo"] = {}; M._STATE["memo_on"] = None
        M.STATS.clear(); M.STATS.update(self.saved[1])
        os.environ.pop(M.MEMO_ENV, None)

    def _x(self, n, pair=True):
        torch = self.torch
        return torch.zeros((1, n, n, 256) if pair else (n, 256), dtype=torch.bfloat16)

    def test_memo_serves_the_same_row_as_select_for_a_sweep_of_classes(self):
        M, face = self.M, self.face
        classes = [(n, residual) for n in (48, 200, 256, 300, 400, 512, 800) for residual in (True, False)]
        for rep in range(3):                                                        # three passes over every class: pass 0 resolves, passes 1-2 are memo hits
            for n, residual in classes:
                x = self._x(n)
                with mock.patch("sys.stderr"):
                    y = M._run(self.mod, x, residual=residual, kind=("transition" if residual else "pair_transition"))
                row, var = face._row("esmpair" if residual else "pair", n)
                k = {"esm_t16": 1.0, "v2": 2.0, "flash_sm90a": 3.0}[row]
                self.assertTrue(bool((y == (x + k if residual else x * 0 + k)).all()), (rep, n, residual, row))   # the row select names for the class served, on every pass
        self.assertEqual(len(face.selects), len(classes)); self.assertEqual(len(face.transitions), len(classes))   # ONE resolution per class (select + transition once)
        self.assertEqual(len(face.dispatches), 2 * len(classes))                                                   # the later calls dispatched the kept Selection
        self.assertEqual(M.STATS["memo_classes"], len(classes)); self.assertEqual(M.STATS["memo_hits"], 2 * len(classes)); self.assertEqual(M.STATS["face_calls"], 3 * len(classes))
        for (word, row, residual, form) in face.dispatches:
            self.assertEqual(form, "esmfused" if residual else "swiglu"); self.assertIn(row, ("esm_t16", "v2", "flash_sm90a"))
            self.assertEqual(word, row + {"esm_t16": "", "v2": ":fast", "flash_sm90a": ":kernel_ln"}[row])          # the row word the class resolved to, verbatim
        rows = dict(M._STATE["rows"])
        self.assertEqual(rows["transition:esm_t16"], 3 * 7); self.assertEqual(rows["pair_transition:v2:fast"], 3 * 3)
        self.assertEqual(rows["pair_transition:flash_sm90a:kernel_ln"], 3 * 2); self.assertEqual(rows["pair_transition:esm_t16"], 3 * 2)
        self.assertEqual(len(M._STATE["memo"]), len(classes)); self.assertEqual(M.kernel_evidence()["memo"], "per_class")

    def test_class_key_separates_word_family_size_rows_and_device(self):
        M, face = self.M, self.face
        with mock.patch("sys.stderr"):
            M._run(self.mod, self._x(200), residual=True, kind="transition")
            M._run(self.mod, self._x(200), residual=True, kind="transition")                  # same class: memo
            M._run(self.mod, self.torch.zeros((2, 200, 200, 256), dtype=self.torch.bfloat16), residual=True, kind="transition")   # same N, twice the rows (a batched pair): its own class
            M._run(self.mod, self._x(200), residual=False, kind="pair_transition")           # the plain pair family: its own class
            M._STATE["word"] = "fast"
            M._run(self.mod, self._x(200), residual=True, kind="transition")                  # another bound word (fast: no memo under it, 2.2) -> resolves through select
        self.assertEqual(len(face.selects), 4); self.assertEqual(M.STATS["memo_hits"], 1); self.assertEqual(len(M._STATE["memo"]), 3)
        self.assertEqual({k[0] for k in M._STATE["memo"]}, {"big"})

    def test_a_class_resolving_to_the_statement_is_kept_by_name(self):
        M, face = self.M, self.face
        with mock.patch("sys.stderr"):
            for _ in range(4):
                self.assertIsNone(M._run(self.mod, self._x(16), residual=True, kind="transition"))   # N < 32: the stub cell names the statement -> None, the module's own call
        self.assertEqual(len(face.selects), 1); self.assertEqual(face.transitions, []); self.assertEqual(face.dispatches, [])
        self.assertEqual(M.STATS["face_refused"], 4); self.assertEqual(M.STATS["memo_hits"], 3); self.assertEqual(M._STATE["refused"]["transition:cell_names_torch_swiglu"], 4)

    def test_a_kept_row_that_refuses_is_dropped_and_the_class_resolves_afresh(self):
        M, face = self.M, self.face
        with mock.patch("sys.stderr"):
            M._run(self.mod, self._x(300), residual=False, kind="pair_transition")           # resolves: flash_sm90a:kernel_ln
            face.refuse_dispatch.add("flash_sm90a:kernel_ln")
            y = M._run(self.mod, self._x(300), residual=False, kind="pair_transition")       # the kept row refuses now -> dropped, full path (select + transition) serves this very call
        self.assertTrue(bool((y == 3.0).all())); self.assertEqual(M.STATS["memo_dropped"], 1)
        self.assertEqual(len(face.selects), 2); self.assertEqual(len(face.transitions), 2); self.assertEqual(M.STATS["face_calls"], 2); self.assertEqual(M.STATS["face_refused"], 0)

    def test_memo_off_by_env_resolves_every_call(self):
        M, face = self.M, self.face
        os.environ[M.MEMO_ENV] = "0"; M._STATE["memo_on"] = None
        with mock.patch("sys.stderr"):
            for _ in range(3):
                M._run(self.mod, self._x(400), residual=True, kind="transition")
        self.assertEqual(len(face.selects), 3); self.assertEqual(len(face.transitions), 3); self.assertEqual(face.dispatches, [])
        self.assertEqual(M.STATS["memo_hits"], 0); self.assertEqual(M._STATE["memo"], {}); self.assertNotIn("memo", M.kernel_evidence())   # off: no token (2.2)

    def test_fast_exact_and_row_words_keep_the_2_0_path_and_census_at_every_size(self):
        """transition_cute.2.2: under `fast`, `exact` and any row word NOTHING of the memo is active at any N -- every call resolves through
        select + transition (no dispatch of a kept Selection), no memo counter exists, and the t16 LEVER evidence is the 2.0 token set (no `memo=`)."""
        M, face = self.M, self.face
        from esmfold2_opt import report
        sys.modules["ef2_transition_cute"] = M
        for word in ("fast", "exact", "esm_t16", "faithful"):
            face.selects.clear(); face.transitions.clear(); face.dispatches.clear(); M.STATS.clear(); M._STATE["memo"].clear()
            M._STATE["word"] = word; M._STATE["memo_on"] = None; M._STATE["rows"].clear(); M._STATE["refused"].clear()
            self.assertFalse(M.memo_on(), word)
            ev0 = M.kernel_evidence()
            self.assertEqual(list(ev0), ["word", "provider", "core", "family"], (word, ev0)); self.assertEqual(ev0["word"], word)      # the 2.0 LEVER tokens, in order; no memo=
            self.assertNotIn("memo", report.lever_evidence("t16"), word)
            n = 0
            with mock.patch("sys.stderr"):
                for N in (200, 400, 800):
                    for residual in (True, False):
                        for _ in range(3):
                            M._run(self.mod, self._x(N), residual=residual, kind=("transition" if residual else "pair_transition")); n += 1
            self.assertEqual(len(face.selects), n, word); self.assertEqual(len(face.transitions), n, word); self.assertEqual(face.dispatches, [], word)   # the 2.0 glue path, every call
            self.assertEqual([k for k in M.STATS if k.startswith("memo")], [], word); self.assertEqual(M._STATE["memo"], {}, word)
            ev = M.kernel_evidence()
            self.assertNotIn("memo", ev, word); self.assertEqual(list(ev)[:4], ["word", "provider", "core", "family"]); self.assertIn("rows", ev)
            self.assertNotIn("memo", report.lever_evidence("t16"), word)
        # the big word: the memo and its token
        M._STATE["word"] = "big"; M._STATE["memo_on"] = None
        self.assertTrue(M.memo_on()); self.assertEqual(M.kernel_evidence().get("memo"), "per_class"); self.assertEqual(report.lever_evidence("t16").get("memo"), "per_class")

    def test_memo_env_1_is_the_ab_knob_under_any_word_and_0_wins_under_big(self):
        M, face = self.M, self.face
        try:
            os.environ[M.MEMO_ENV] = "1"
            for word in ("fast", "esm_t16"):
                M._STATE["word"] = word; M._STATE["memo_on"] = None
                self.assertTrue(M.memo_on(), word); self.assertEqual(M.kernel_evidence().get("memo"), "per_class")
            os.environ[M.MEMO_ENV] = "0"
            M._STATE["word"] = "big"; M._STATE["memo_on"] = None
            self.assertFalse(M.memo_on()); self.assertNotIn("memo", M.kernel_evidence())
            os.environ.pop(M.MEMO_ENV)
            M._STATE["memo_on"] = None
            self.assertTrue(M.memo_on())                                                     # unset: big -> on
            M._STATE["word"] = "fast"
            self.assertFalse(M.memo_on())                                                    # re-decided when the bound word changes
        finally:
            os.environ.pop(M.MEMO_ENV, None); M._STATE["memo_on"] = None

    def test_a_core_without_the_serve_entry_serves_through_transition(self):
        M, face = self.M, self.face
        d = face._dispatch
        try:
            face._dispatch = None                                                        # getattr(TR, "_dispatch", None) is None: transition(<row word>) serves the memoised class
            with mock.patch("sys.stderr"):
                for _ in range(3):
                    y = M._run(self.mod, self._x(400), residual=True, kind="transition")
            self.assertTrue(bool((y == self._x(400) + 1.0).all()))
            self.assertEqual(len(face.selects), 1); self.assertEqual(len(face.transitions), 3); self.assertEqual(M.STATS["memo_hits"], 2)
        finally:
            face._dispatch = d


if __name__ == "__main__":
    unittest.main()
