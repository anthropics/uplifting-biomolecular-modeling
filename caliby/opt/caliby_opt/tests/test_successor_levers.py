"""The successor levers' package side, on CPU: X003/X011 background CIF writers (`caliby_opt.bg_writers` over
`opt_core.host.outputs.AsyncWriter`) and X010 concurrent multi-sequence sampling's words (`caliby_opt.multiseq`) — the switch words and
their usage errors, the chunking, real forked and threaded writers writing the same bytes, a FAILING forked writer named on stderr and
raised (non-zero exit in a real process), the serial-fallback line, and the exit line's census words (`report.multiseq_state`,
`report.cif_writers_state`). The lever files' GPU side (same sequences, energies and CIF bytes as the serial calls / the synchronous
writes) is not this file's; here the kit file is held to being glue only (no fork/wait/pool code of its own)."""
import ast
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

from .. import bg_writers, env_words, multiseq, report, stack

SERVED_SDU = "caliby.eval.eval_utils.seq_des_utils"
SERVED_POTTS = "caliby.model.seq_denoiser.denoisers.seq_design.potts"
SWITCHES = ("CALIBY_X_BG_CIF", "CALIBY_X_CIF_WORKERS", "CALIBY_X_MULTISEQ")


def _write_texts(items):                      # a background writer's chunk in these tests: [(path, text), ...]
    for path, text in items:
        with open(path, "w") as fh:
            fh.write(text)


def _write_fail(items):                       # a writer that dies on its chunk
    raise OSError(f"no space left on device (test) while writing {len(items)} files")


def _clean_env():
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in SWITCHES}, clear=True)


def _reset_states():
    bg_writers.STATE.update(mode=None, workers=0, batches=0, chunks=0, files=0, failed=0, joined=False)
    multiseq.STATE.update(calls=0, sequences=0, serial_calls=0, serial_reasons={}, rng=None)


class TestSwitchWords(unittest.TestCase):
    def test_int_and_choice_words(self):
        with _clean_env():
            self.assertEqual(env_words.int_word("CALIBY_X_CIF_WORKERS"), 0)
            self.assertIsNone(env_words.choice_word("CALIBY_X_BG_CIF", ("fork", "thread")))
            os.environ["CALIBY_X_CIF_WORKERS"] = "8"
            self.assertEqual(env_words.int_word("CALIBY_X_CIF_WORKERS"), 8)
            os.environ["CALIBY_X_CIF_WORKERS"] = ""
            self.assertEqual(env_words.int_word("CALIBY_X_CIF_WORKERS", default=3), 3)
            for bad in ("x", "8w", "1.5", "-1"):
                os.environ["CALIBY_X_CIF_WORKERS"] = bad
                with self.assertRaises(env_words.UsageError) as cm:
                    env_words.int_word("CALIBY_X_CIF_WORKERS")
                self.assertTrue(str(cm.exception).startswith("[CALIBY_X_CIF_WORKERS] malformed value"), str(cm.exception))
            os.environ["CALIBY_X_BG_CIF"] = "frok"
            with self.assertRaises(env_words.UsageError):
                env_words.choice_word("CALIBY_X_BG_CIF", ("fork", "thread", "0"))
            self.assertTrue(issubclass(env_words.UsageError, ValueError))

    def test_writer_settings(self):
        with _clean_env():
            self.assertEqual(bg_writers.settings(), (None, 1))                   # unset: the stock synchronous writes
            os.environ["CALIBY_X_BG_CIF"] = "0"
            self.assertEqual(bg_writers.settings(), (None, 1))
            os.environ["CALIBY_X_BG_CIF"] = "fork"
            self.assertEqual(bg_writers.settings(), ("fork", 1))                 # X003: one background writer
            os.environ["CALIBY_X_CIF_WORKERS"] = "8"
            self.assertEqual(bg_writers.settings(), ("fork", 8))                 # X011
            os.environ["CALIBY_X_BG_CIF"] = "thread"
            self.assertEqual(bg_writers.settings(), ("thread", 8))
            os.environ["CALIBY_X_CIF_WORKERS"] = "many"
            with self.assertRaises(env_words.UsageError):
                bg_writers.settings()

    def test_multiseq_level(self):
        with _clean_env():
            self.assertEqual(multiseq.level(), 0)
            for word, want in (("0", 0), ("1", 1)):
                os.environ["CALIBY_X_MULTISEQ"] = word
                self.assertEqual(multiseq.level(), want)
            for bad in ("2", "on", "-1", "1.0"):
                os.environ["CALIBY_X_MULTISEQ"] = bad
                with self.assertRaises(env_words.UsageError) as cm:
                    multiseq.level()
                self.assertTrue(str(cm.exception).startswith("[CALIBY_X_MULTISEQ] malformed value"), str(cm.exception))


class TestChunks(unittest.TestCase):
    def test_every_item_exactly_once_in_order(self):
        for n in (0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 100, 320, 801):
            for workers in (1, 2, 3, 8, 16):
                items = list(range(n))
                parts = bg_writers.chunks(items, workers)
                self.assertEqual([x for p in parts for x in p], items, (n, workers))
                self.assertLessEqual(len(parts), max(1, workers), (n, workers))
                if len(parts) > 1:
                    self.assertTrue(all(len(p) >= bg_writers.CHUNK_MIN for p in parts), (n, workers, [len(p) for p in parts]))
                    self.assertLessEqual(max(len(p) for p in parts) - min(len(p) for p in parts), 1)     # balanced
                self.assertEqual(len(parts), max(1, min(workers, n // bg_writers.CHUNK_MIN)) if n else 0, (n, workers))
                self.assertEqual(len(parts) == 0, n == 0)


class TestBackgroundWriters(unittest.TestCase):
    def setUp(self):
        _reset_states()

    def _items(self, d, n):
        return [(os.path.join(d, f"design_{i}.cif"), f"data_{i}\n" * (i + 1)) for i in range(n)]

    def test_fork_and_thread_writers_write_every_file_and_count(self):
        for mode in ("fork", "thread"):
            _reset_states()
            with _clean_env(), tempfile.TemporaryDirectory() as d:
                os.environ.update(CALIBY_X_BG_CIF=mode, CALIBY_X_CIF_WORKERS="2")
                err = io.StringIO()
                with redirect_stderr(err):
                    w = bg_writers.from_env(_write_texts)
                self.assertIsNotNone(w)
                line = w.lever_line()
                self.assertIn(line, err.getvalue().splitlines())                 # announced once on stderr at creation (report.say)
                self.assertTrue(line.startswith("[caliby-opt] LEVER name=F6.output_overlap state=on impl=host.outputs origin=core"), line)
                self.assertIn(f"mode={mode}", line)
                self.assertIn("workers=2", line)
                self.assertEqual(w.submit_batch(self._items(d, 20)), 2)          # 20 files, 2 workers -> 2 chunks of 10
                self.assertEqual(w.submit_batch(self._items(d, 5)[:0]), 0)
                more = [(os.path.join(d, f"b2_{i}.cif"), "x") for i in range(5)]
                self.assertEqual(w.submit_batch(more), 1)                        # fewer than CHUNK_MIN files -> one chunk
                w.join()
                for path, text in self._items(d, 20) + more:
                    with open(path) as fh:
                        self.assertEqual(fh.read(), text, path)
                self.assertEqual(bg_writers.census_word(), f"{mode}:2w/3chunks/25files")
                st = bg_writers.STATE
                self.assertEqual((st["batches"], st["chunks"], st["files"], st["failed"], st["joined"]), (3, 3, 25, 0, True))

    def test_unset_is_the_stock_synchronous_path(self):
        with _clean_env():
            self.assertIsNone(bg_writers.from_env(_write_texts))
            self.assertEqual(bg_writers.census_word(), "sync")
        _reset_states()
        self.assertEqual(bg_writers.census_word(), "unused")                    # run_seq_des never ran (the ensemble route)

    def test_a_failing_forked_writer_is_named_on_stderr_and_raised(self):
        with _clean_env(), tempfile.TemporaryDirectory() as d:
            os.environ.update(CALIBY_X_BG_CIF="fork", CALIBY_X_CIF_WORKERS="2")
            w = bg_writers.from_env(_write_fail)
            w.submit_batch(self._items(d, 16))                                   # 2 chunks, both die
            err = io.StringIO()
            with redirect_stderr(err), self.assertRaises(RuntimeError) as cm:
                w.join()
            text = err.getvalue()
            self.assertIn("[CALIBY_X_BG_CIF] 2 background CIF write(s) failed (fork mode, 2 worker(s)): the outputs of this run are incomplete", text)
            self.assertIn("write(s) failed", str(cm.exception))                  # the core writer's own error, raised through
            self.assertEqual(bg_writers.STATE["failed"], 2)
            self.assertTrue(bg_writers.STATE["joined"])
            self.assertEqual(bg_writers.census_word(), "fork:2w/2chunks/16files,failed:2")
            with mock.patch.dict(sys.modules, {SERVED_SDU: types.SimpleNamespace(_x_write_cifs=_write_fail)}):
                self.assertEqual(report.cif_writers_state(), "fork:2w/2chunks/16files,failed:2")   # the exit line's word
            self.assertEqual(sorted(os.listdir(d)), [])                          # nothing rewritten behind the failure

    def test_a_failing_writer_is_a_nonzero_exit_in_a_real_process(self):
        code = ("import os, sys, tempfile\n"
                "os.environ.update(CALIBY_X_BG_CIF='fork', CALIBY_X_CIF_WORKERS='2')\n"
                "from caliby_opt import bg_writers\n"
                "from caliby_opt.tests.test_successor_levers import _write_fail\n"
                "d = tempfile.mkdtemp()\n"
                "w = bg_writers.from_env(_write_fail)\n"
                "w.submit_batch([(os.path.join(d, 'a%d.cif' % i), 'x') for i in range(9)])\n"
                "w.join()\n"
                "print('UNREACHABLE: returned after a failed writer')\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("[CALIBY_X_BG_CIF] 1 background CIF write(s) failed", r.stderr)
        self.assertNotIn("UNREACHABLE", r.stdout)

    def test_exit_word_stock_and_not_loaded(self):
        with mock.patch.dict(sys.modules, {SERVED_SDU: types.SimpleNamespace()}):
            self.assertEqual(report.cif_writers_state(), "stock_module")
        with mock.patch.dict(sys.modules, {k: v for k, v in sys.modules.items() if k != SERVED_SDU}, clear=True):
            self.assertEqual(report.cif_writers_state(), "not_loaded")


class TestMultiseqWords(unittest.TestCase):
    def setUp(self):
        _reset_states()

    def test_refusal_is_the_fast_paths_conditions(self):
        served = dict(fast_sampler_level=2, proposal="dlmc", rejection_step=False, differentiable_penalty=True, is_cuda=True)
        self.assertIsNone(multiseq.refusal(**served))
        self.assertIn("CALIBY_FAST_SAMPLER=1", multiseq.refusal(**dict(served, fast_sampler_level=1)))
        self.assertIn("CALIBY_FAST_SAMPLER=0", multiseq.refusal(**dict(served, fast_sampler_level=0)))
        for change in (dict(proposal="chromatic"), dict(rejection_step=True), dict(differentiable_penalty=False), dict(return_trajectory=True)):
            self.assertIn("outside the fast path", multiseq.refusal(**dict(served, **change)), change)
        self.assertEqual(multiseq.refusal(**dict(served, is_cuda=False)), "CPU tensors")

    def test_serial_fallback_line_and_census_words(self):
        err = io.StringIO()
        with redirect_stderr(err):
            multiseq.note_serial("CPU tensors")
            multiseq.note_serial("CPU tensors")
        lines = err.getvalue().splitlines()
        self.assertEqual(lines, ["[CALIBY_X_MULTISEQ] CPU tensors -> serial fallback (the sequences of this call sampled one call at a time)"] * 2)
        self.assertIn(report.FALLBACK_MARKERS["CALIBY_X_MULTISEQ"], lines[0])  # counted by the stderr tee -> the run ends NOT ACTIVE (partial)
        multiseq.note_served(8, {"seed": 11, "base": 0, "increment": 4, "unit": 2004, "after": 16032})
        self.assertEqual(multiseq.census_word(), "1calls/8seqs,serial:2")
        self.assertEqual(multiseq.STATE["serial_reasons"], {"CPU tensors": 2})
        self.assertEqual(multiseq.STATE["rng"]["after"], 16032)
        with mock.patch.dict(sys.modules, {SERVED_POTTS: types.SimpleNamespace(sample_potts_multiseq=object())}):
            self.assertEqual(report.multiseq_state(), "1calls/8seqs,serial:2")
        with mock.patch.dict(sys.modules, {SERVED_POTTS: types.SimpleNamespace()}):
            self.assertEqual(report.multiseq_state(), "stock_module")
        _reset_states()
        multiseq.note_served(8)
        self.assertEqual(multiseq.census_word(), "1calls/8seqs")             # the expectations table's shape: calls > 0, no serial suffix


class TestKitFilesAreGlue(unittest.TestCase):
    """The lever files delegate to the package: `seq_des_utils.py` keeps no process/pool code of its own and hands `run_seq_des`'s CIFs to
    `bg_writers`; `potts.py` reads its switch word and prints its fallback line through `multiseq`."""

    def _tree(self, name):
        path = os.path.join(stack.kit_dir(stack.KIT_ADDON), "fast", name)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        return src, ast.parse(src)

    def test_seq_des_utils_writer_glue(self):
        src, tree = self._tree("seq_des_utils.py")
        top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("_x_write_cifs", top)                                     # module level: a forked worker can name it
        for needle in ("_x_bg.from_env(_x_write_cifs)", "_x_writers.submit_batch(_x_items)", "_x_writers.join()", "import caliby_opt.bg_writers as _x_bg"):
            self.assertIn(needle, src)
        for banned in ("os.fork(", ".fork()", "waitpid", "ThreadPoolExecutor", "multiprocessing", "_x_os"):
            self.assertNotIn(banned, src, banned)

    def test_potts_words_come_from_the_package(self):
        src, tree = self._tree("potts.py")
        self.assertIn("import caliby_opt.multiseq as _x_ms", src)
        for needle in ("_x_ms.level()", "_x_ms.refusal(", "_x_ms.note_serial(reason)", "_x_ms.note_served("):
            self.assertIn(needle, src)
        self.assertNotIn('os.environ.get("CALIBY_X_MULTISEQ"', src)


if __name__ == "__main__":
    unittest.main()
