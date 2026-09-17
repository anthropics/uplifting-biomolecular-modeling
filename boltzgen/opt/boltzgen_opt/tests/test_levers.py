"""Kit-carried and house-written levers, exercised on the stub upstream with real CPU torch and the pinned core (no GPU needed):

* seed fix (opt/forward/fast_inference/src/bg_hook.py, TestSeedFix) — the featurizer seed derives from the global numpy stream's
  state without a draw, so binder lengths drawn after it equal stock's at the same seed; derived seeds are deterministic and distinct.
* OOM trace (opt/forward/size_levers/src/sz_levers.py, TestOomTrace) — a CUDA OOM inside Boltz.forward prints the `[sz] {"event":
  "oom", …}` evidence line the caller's reader matches (stack.RUNNER_OOM_LINE) and re-raises the SAME exception: upstream's own
  predict_step handler then skips the batch exactly as stock does (no exit code of the lever's own; the run's lines say what happened).
* async_writer (opt/host/writer_levers/src/hl_levers.py, TestHostLevers) — the background write is upstream's own writer body on
  host tensors; the files a run writes are byte for byte the synchronous writer's in every worker form (fork, spawn, thread).
* fast's own levers (opt/forward/fast_levers/src/fl_levers.py, TestFastLevers) — the sampler-context precondition, cond_dedup
  numerics against the stock statements on B identical rows, the per-site census and fail-closed gate, the attention lever's
  eligibility rules, and the process-form lines the caller's census reads.
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError as e:                                                           # a bare venv: every lever here needs numpy + torch (CPU)
    raise unittest.SkipTest(f"numpy + torch (CPU) are the test floor of these levers: {e}")

from boltzgen_opt import modes, registry, stack

from . import _stubs


# ---------------------------------------------------------------------------------------------- seed fix

HOOK_DIR = os.path.join(stack.kit_dir(modes.KIT_PARTNER), "src")


def _hook_default_rng():
    """bg_hook's `_seeded_default_rng` on a fresh import of the carried module (its other patches need the kits' stack: they fail soft)."""
    sys.path.insert(0, HOOK_DIR)
    try:
        sys.modules.pop("bg_hook", None)
        saved = np.random.default_rng
        hook = importlib.import_module("bg_hook")
        patched = np.random.default_rng
        np.random.default_rng = saved
        return hook, patched
    finally:
        sys.path.remove(HOOK_DIR)


class TestSeedFix(unittest.TestCase):
    def test_binder_length_draws_equal_stock_at_the_same_seed(self):
        hook, patched = _hook_default_rng()
        self.assertTrue(hook._acc.get("seed_fix_active"), "seed_fix_active is recorded at import")
        seed = 12345
        np.random.seed(seed)                                                          # pl.seed_everything's numpy line
        stock_lengths = [int(np.random.randint(60, 121)) for _ in range(4)]           # schema.py l.543 for designs r = 0..3, no featurizer between
        np.random.seed(seed)
        arm_lengths = []
        for _ in range(4):
            patched(None); patched(None)                                              # two featurizer RNG constructions per design (the DataLoader worker's)
            arm_lengths.append(int(np.random.randint(60, 121)))
        self.assertEqual(arm_lengths, stock_lengths)                                  # the global stream untouched by the seed derivation

    def test_derived_seeds_are_deterministic_and_distinct(self):
        hook, patched = _hook_default_rng()
        np.random.seed(7)
        a = [patched(None).integers(0, 2**31) for _ in range(3)]
        np.random.seed(7)
        hook._seed_calls.clear()
        b = [patched(None).integers(0, 2**31) for _ in range(3)]
        self.assertEqual([int(x) for x in a], [int(x) for x in b])                    # deterministic under the seed (and the call's ordinal)
        self.assertEqual(len({int(x) for x in a}), 3)                                 # distinct per call: no two featurizer RNGs share a stream
        self.assertEqual(int(patched(5).integers(0, 10**6)), int(np.random.default_rng(5).integers(0, 10**6)))   # an explicit seed passes through

    def test_calls_before_the_seed_line_do_not_shift_the_seeds_derived_after_it(self):
        """The binder-path defect: on the kit arms scipy.stats constructs four generators while bg_hook imports cuequivariance_torch — before
        any seed line —, on `off` nothing does; 0.5.2's process-wide ordinal made every later seed differ between the arms. The ordinal is per
        stream key now: what was drawn under another key (unseeded, or another step's seed) does not count."""
        hook, patched = _hook_default_rng()
        hook._seed_calls.clear()
        np.random.seed(7)
        clean = [int(patched(None).integers(0, 2**31)) for _ in range(3)]
        hook._seed_calls.clear()
        np.random.seed(99)                                                            # another key: an earlier step, or the unseeded state at import
        for _ in range(4):
            patched(None)
        np.random.seed(7)                                                             # this step's seed line
        after = [int(patched(None).integers(0, 2**31)) for _ in range(3)]
        self.assertEqual(after, clean)

    def test_a_forked_worker_counts_from_its_own_first_call(self):
        """The featurizer's generator is drawn in a forked DataLoader worker: its seeds must not depend on how many generators the parent drew
        (the kits' runner and the stock child differ there). The ordinal is keyed by process id: under another pid (the worker's — here the
        hook's `os.getpid` answered by a stand-in, the inherited stream state being the parent's by construction) the first call derives what
        a process with no earlier call derives, the parent's first, however many the parent drew."""
        from unittest import mock
        hook, patched = _hook_default_rng()
        hook._seed_calls.clear()
        np.random.seed(7)
        parent = [int(patched(None).integers(0, 2**31)) for _ in range(2)]
        with mock.patch("os.getpid", return_value=os.getpid() + 100003):              # the forked worker's view: same stream state, its own pid
            child = [int(patched(None).integers(0, 2**31)) for _ in range(2)]
        self.assertEqual(child, parent)
        self.assertNotEqual(parent[0], parent[1])
        self.assertEqual(int(patched(None).integers(0, 2**31)) in parent, False)       # back in the parent the count went on: a third, new seed




# ------------------------------------------------------------------------------------------------ OOM trace

SIZE_LEVERS_SRC = stack.kit_src(modes.KIT_SIZE_LEVERS)                       # <opt>/forward/size_levers/src: the PYTHONPATH entry the mode table gives big (stack.mode_env)

OOM_CODE = r"""
import json, sys, torch
import boltzgen.model.models.boltz as bz
def oom_forward(self, *a, **k):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
bz.Boltz.forward = oom_forward                    # the model's forward raises the real OOM class (no GPU needed): installed BEFORE the lever wraps it
import sz_levers
sz_levers.install()
before = dict(sz_levers.STATS)
def stock_shaped_catch(fn):                       # the shape of upstream's own predict_step handler (boltz.py: `except RuntimeError` + "out of memory" in the text -> warn, skip the batch)
    try:
        fn(); return "no_exception"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return "skipped"
        raise
caught = None
try:
    bz.Boltz.forward(bz.Boltz())
except torch.cuda.OutOfMemoryError as e:          # the SAME exception comes back out of the wrapper
    caught = [type(e).__name__, isinstance(e, RuntimeError), str(e).split(".")[0]]
except BaseException as e:                        # anything else (a SystemExit, a wrapped error) is a different design and fails below
    caught = [type(e).__name__, None, str(e)]
print("RECORD " + json.dumps({"caught": caught, "upstream": stock_shaped_catch(lambda: bz.Boltz.forward(bz.Boltz())),
                             "before": [before["oom_trace_enabled"], before["oom_trace_site"], before["oom_hit"]], "hit": sz_levers.STATS["oom_hit"]}))
"""


class TestOomTrace(unittest.TestCase):
    """sz_levers' OOM trace on the stub model with real CPU torch, in a fresh interpreter: the wrapper is evidence only — one `[sz]` event
    line, STATS["oom_hit"], and the original OutOfMemoryError re-raised (a RuntimeError subclass, so upstream's own handler one frame up
    skips the batch as it does without the levers). The event line is the one stack.levers_from_lines reads as `oom` for the run record."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_oomtrace_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_trace(self, **env):
        rc, out, err = _stubs.run_py(OOM_CODE, _stubs.clean_env(self.site, extra_path=SIZE_LEVERS_SRC, **env))
        self.assertEqual(rc, 0, err)
        return next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD ")), out, err

    def test_the_trace_prints_its_line_and_reraises_the_same_oom(self):
        rec, out, err = self.run_trace(SZ_OOM_TRACE="1")
        self.assertEqual(rec["before"], [True, "boltzgen.model.models.boltz.Boltz.forward", False])     # armed at install, named site, nothing hit yet
        self.assertEqual(rec["caught"], ["OutOfMemoryError", True, "CUDA out of memory"])                 # the same exception, still a RuntimeError: upstream's handler sees exactly what stock raises
        self.assertEqual(rec["upstream"], "skipped")                                                   # and a handler shaped like upstream's own skips the batch, as stock does
        self.assertTrue(rec["hit"])
        events = [ln for ln in out.splitlines() if stack.RUNNER_OOM_LINE.match(ln)]
        self.assertEqual(len(events), 2, out)                                                          # one `[sz] {"event": "oom", ...}` line per OOM (two calls here)
        ev = json.loads(events[0][len("[sz] "):])
        self.assertEqual((ev["event"], ev["msg"]), ("oom", "CUDA out of memory. Tried to allocate 1.00 GiB"))
        self.assertIn("where", ev); self.assertIn("max_alloc_GB", ev)
        res = modes.resolve("big", HOME)
        self.assertTrue(stack.levers_from_lines(out.splitlines(), res)["oom"])                          # the caller's reader records it for the run (design.process_report: rep["oom"]); the exit stays the run's own
        self.assertIsInstance(torch.cuda.OutOfMemoryError("x"), RuntimeError)
        src = open(os.path.join(SIZE_LEVERS_SRC, "sz_levers.py"), encoding="utf-8").read()
        self.assertNotIn("SystemExit", src); self.assertNotIn("SZ_OOM_EXIT", src)                       # no exit of the lever's own: evidence, then upstream's control flow


# --------------------------------------------------------------------------------------------- async_writer

HOST_PRELUDE = r"""
import json, os, sys
import torch
import boltzgen.task.predict.writer as W
class _Cfg:
    multiplicity = 4
class _DM:
    cfg = _Cfg()
class _Trainer:
    datamodule = _DM()
def run(outdir, batches=3, rows=4, seed=0):
    torch.manual_seed(seed)
    w = W.DesignWriter(outdir)
    _orig = w.write_on_batch_end
    def timed(*a, **kw):                      # an instance-bound closure like the partner kit's timing wrapper: unpicklable, must not travel to a worker
        return _orig(*a, **kw)
    w.write_on_batch_end = timed
    for b in range(batches):
        pred = {"exception": False, "coords": torch.randn(rows, 7, 3)}
        batch = {"id": ["spec"], "mask": torch.ones(1, 7)}
        w.on_predict_batch_end(_Trainer(), None, pred, batch, b)
    w.on_predict_end(_Trainer(), None)
    return sorted(os.listdir(outdir))
"""


def _digest(d):
    return {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))}


def hl_levers_modes():
    return ("fork",)                                                  # hl_levers.MODE: the one worker form


class TestHostLevers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_host_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, code, **env):
        e = _stubs.clean_env(self.site, extra_path=stack.kit_src(modes.KIT_HOST_LEVERS), **env)
        return _stubs.run_py(HOST_PRELUDE + code, e)

    def test_the_host_levers_kit_is_laid_out_like_every_kit(self):
        self.assertIn(modes.KIT_HOST_LEVERS, modes.KITS)
        self.assertEqual(stack.kit_missing(modes.KIT_HOST_LEVERS), [])
        self.assertTrue(os.path.isfile(os.path.join(stack.kit_src(modes.KIT_HOST_LEVERS), "hl_levers.py")))
        lv = registry.LEVERS["async_writer"]
        self.assertEqual((lv.kit, lv.switch, lv.tier), (modes.KIT_HOST_LEVERS, "HL_ASYNC_WRITER", "exact"))

    def test_switch_off_installs_nothing(self):
        rc, out, err = self._run('import hl_levers\nprint(json.dumps([hl_levers.STATS["writer_enabled"], W.DesignWriter.write_on_batch_end.__qualname__]))\n')
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out.strip().splitlines()[-1]), [False, "DesignWriter.write_on_batch_end"])
        self.assertNotIn("async_writer installed", out + err)

    def test_the_background_write_is_the_synchronous_writers_bytes(self):
        ref = os.path.join(self.tmp, "sync")
        rc, out, err = self._run(f"print(json.dumps(run({ref!r})))\n")
        self.assertEqual(rc, 0, err)
        want = _digest(ref)
        self.assertEqual(len(want), 3 * 4 * 2)                       # 3 batches x 4 designs x (cif, npz)
        for mode in ("fork",):                                       # the one worker form (hl_levers.MODE): forked worker processes
            d = os.path.join(self.tmp, "async_" + mode)
            rc, out, err = self._run(f"import hl_levers\nprint(json.dumps(run({d!r})))\n", HL_ASYNC_WRITER="1")
            self.assertEqual(rc, 0, f"{mode}: {err}")
            self.assertEqual(_digest(d), want, mode)                 # same names, same bytes
            self.assertRegex(out, stack.RUNNER_LINES["async_writer"].pattern.lstrip("^"), mode)
            self.assertIn(f"[hl_levers] CENSUS async_writer[W=4,mode={mode},submitted=3,written=3,failed=0]", out, mode)   # hl_levers.WORKERS = 4
            self.assertIn("HOST host.outputs TALLY part=writer name=design_writer", out, mode)
            stats = [l for l in out.splitlines() if stack.RUNNER_STATS_LINE.match(l) and "async_writer stats" in l]
            self.assertEqual(len(stats), 1, out)
            rec = json.loads(stack.RUNNER_STATS_LINE.match(stats[0]).group(2))
            self.assertEqual((rec["batches"], rec["designs"], rec["drains"], rec["writer"]["written"], rec["writer"]["failed"]), (3, 12, 1, 3, 0), mode)
            self.assertNotIn("GATE-FAIL", out + err, mode)

    def test_a_failed_write_fails_the_run_by_name(self):
        d = os.path.join(self.tmp, "async_fail")
        code = "import hl_levers\ntry:\n    run(%r)\nexcept RuntimeError as e:\n    print('DRAIN-RAISED', str(e).splitlines()[0])\n    sys.exit(7)\n" % (d,)
        for mode in hl_levers_modes():
            rc, out, err = self._run(code, HL_ASYNC_WRITER="1", BGSTUB_WRITER_FAIL="1")
            self.assertNotEqual(rc, 0, f"{mode}: {err}")                     # the drain at on_predict_end re-raised the write failure (the caller saw it: DRAIN-RAISED) and the core's exit guard keeps the status a loss even over the caller's own exit code
            self.assertIn("DRAIN-RAISED", out, mode)
            self.assertRegex(out + err, stack.RUNNER_DISABLED_LINES["async_writer"].pattern.lstrip("^"), mode)
            self.assertIn("failed=3", out + err, mode)
            self.assertIn("HOST host.outputs TALLY", out + err, mode)


# ------------------------------------------------------------------------------------------------ fast levers

HOME = stack.opt_home()
FAST_PRELUDE = r'''
import json, os, sys, threading
import torch
os.environ.setdefault("FL_COND_DEDUP", "1"); os.environ.setdefault("FL_ATTN_BF16", "1")
import fl_levers as fl
from boltzgen.model.modules import transformers as T, encoders as E, diffusion as D
torch.manual_seed(0)
def sampling(fn, rows=4):
    """Run fn inside the sampler context exactly as the wrapper establishes it: preconditioned_network_forward(training=False), batch-1 trunk."""
    ad = D.AtomDiffusion(); ad.eval()
    return ad.preconditioned_network_forward(torch.zeros(rows, 3), 1.0, {"s_trunk": torch.zeros(1, 5, 8), "multiplicity": rows, "fn": fn}, training=False)
'''


def _run(code, site, **env):
    e = _stubs.clean_env(site, extra_path=stack.kit_src(modes.KIT_FAST_LEVERS), **env)
    return _stubs.run_py(FAST_PRELUDE + code, e)


class TestFastLevers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_fast_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_fast_levers_kit_is_laid_out_like_every_kit(self):
        src = stack.kit_src(modes.KIT_FAST_LEVERS)
        self.assertEqual(src, os.path.join(HOME, "forward", "fast_levers", "src"))
        self.assertTrue(os.path.isfile(os.path.join(src, "fl_levers.py")))
        for name in ("cond_dedup", "attn_bf16", "attn_cudnn", "dit_fused"):
            self.assertEqual((registry.LEVERS[name].kit, registry.LEVERS[name].file, registry.LEVERS[name].tier), (modes.KIT_FAST_LEVERS, "src/fl_levers.py", "2"))
        res = modes.resolve("fast", HOME)
        self.assertEqual(res.imports[-3:], ("fl_levers", "sz_levers", "hl_levers"))   # the mode's own imports: the fast levers, the out-of-memory trace, the background writer
        self.assertIn(src, stack.mode_env(res, {"PATH": os.environ.get("PATH", "")})[0]["PYTHONPATH"].split(os.pathsep))
        pins = json.load(open(os.path.join(os.path.dirname(HOME), "stock", "PINS.json")))
        for rel in ("boltzgen/model/modules/diffusion.py", "boltzgen/model/modules/encoders.py", "boltzgen/model/modules/transformers.py", "boltzgen/model/layers/attention.py"):
            self.assertIn(rel, pins["touched_modules"])                       # every upstream module the levers patch or read is pinned against the stock wheel

    def test_install_patches_exactly_the_named_targets_and_prints_the_proof_lines(self):
        rc, out, err = _run(r'''
print("RECORD " + json.dumps({
  "pnf": getattr(D.AtomDiffusion.preconditioned_network_forward, "_fl_levers", False), "sc": getattr(E.SingleConditioning.forward, "_fl_levers", False),
  "adaln": T.AdaLN.forward is fl.adaln_forward, "ctb": T.ConditionedTransitionBlock.forward is fl.transition_forward, "layer": T.DiffusionTransformerLayer.forward is fl.layer_forward,
  "sdpa": getattr(torch.nn.functional.scaled_dot_product_attention, "_fl_levers", False), "sites": sorted(fl.SITES),
  "stats": {k: fl.STATS[k] for k in ("dedup_enabled", "attn_enabled")}}))
fl.install()                                      # idempotent: no second line
''', self.site)
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertEqual(rec, {"pnf": True, "sc": True, "adaln": True, "ctb": True, "layer": True, "sdpa": True, "sites": sorted(fl_sites()),
                               "stats": {"dedup_enabled": True, "attn_enabled": True}})
        self.assertEqual(out.count("[fl_levers] cond_dedup installed"), 1); self.assertEqual(out.count("[fl_levers] attn_bf16 installed"), 1)
        self.assertRegex(out, stack.RUNNER_LINES["cond_dedup"].pattern.lstrip("^")); self.assertRegex(out, stack.RUNNER_LINES["attn_bf16"].pattern.lstrip("^"))

    def test_switches_off_install_nothing(self):
        rc, out, err = _run(r'''
print("RECORD " + json.dumps({"pnf": getattr(D.AtomDiffusion.preconditioned_network_forward, "_fl_levers", False), "sites": sorted(fl.SITES), "dedup": fl.STATS["dedup_enabled"], "attn": fl.STATS["attn_enabled"]}))
''', self.site, FL_COND_DEDUP="0", FL_ATTN_BF16="0")
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertEqual(rec, {"pnf": False, "sites": [], "dedup": False, "attn": False})
        self.assertNotIn("[fl_levers]", out)

    def test_cond_dedup_serves_the_s_path_on_one_row_inside_the_sampler_and_matches_the_stock_statements(self):
        """B identical conditioning rows: SingleConditioning runs once and is broadcast (the token layers then see a stride-0 view and serve
        their own sites); the result equals the stock statements on the B materialized rows to fp32 rounding (CPU: a GEMM with 1 row vs B
        rows may round differently — Tier 2, no bitwise claim); outside the sampler context and on real (non-broadcast) rows every site runs
        the stock statement inline, counted by reason; nothing is served, nothing falls back."""
        rc, out, err = _run(r'''
B, N, C = 4, 5, 8
sc = E.SingleConditioning(token_s=C // 2); layer = T.DiffusionTransformerLayer(heads=2, dim=C); sc.eval(); layer.eval()
times = torch.full((B,), 0.37); st1 = torch.randn(1, N, C // 2); si1 = torch.randn(1, N, C // 2); a = torch.randn(B, N, C)
with torch.no_grad():
    # outside the sampler: stock statements everywhere (the wrapper is inert), on materialized rows
    s_ref, f_ref = sc(times, st1.repeat_interleave(B, 0), si1.repeat_interleave(B, 0))
    y_ref = layer(a, s_ref, None, torch.ones(B, N))
    before = {n: dict(fl.SITES[n].stats()) for n in fl.SITES}
    inline_before = dict(fl.STATS["inline_by"])
    # inside the sampler: served
    def body():
        s, f = sc(times, st1.repeat_interleave(B, 0), si1.repeat_interleave(B, 0))
        return s, f, layer(a, s, None, torch.ones(B, N))
    s_new, f_new, y_new = sampling(body, rows=B)
after = {n: fl.SITES[n].stats() for n in fl.SITES}
print("RECORD " + json.dumps({
  "s_stride0": s_new.stride(0), "s_close": torch.allclose(s_new, s_ref, atol=1e-5, rtol=1e-5), "f_close": torch.allclose(f_new, f_ref, atol=1e-6),
  "y_close": torch.allclose(y_new, y_ref, atol=1e-5, rtol=1e-5), "y_maxdiff": float((y_new - y_ref).abs().max()),
  "served_before": {n: before[n]["served"] for n in before}, "served_after": {n: after[n]["served"] for n in after},
  "fallback_after": {n: after[n]["fallback"] for n in after}, "rows_saved_cond": after["cond"]["rows_saved"],
  "inline_before": inline_before, "inline_after": dict(fl.STATS["inline_by"]), "context": dict(fl.STATS["context_by"]), "max_rows": fl.STATS["max_rows"],
  "gate": fl.gate()}))
''', self.site)
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertEqual(rec["s_stride0"], 0)                                       # broadcast, not recomputed
        self.assertTrue(rec["s_close"] and rec["f_close"] and rec["y_close"], rec)
        self.assertEqual(rec["served_before"], {n: 0 for n in fl_sites()})           # outside the sampler nothing is served ...
        self.assertGreater(rec["inline_before"]["not_sampling"], 0)                  # ... and the inline stock statements are counted by reason
        self.assertEqual(rec["served_after"], {"cond": 1, "adaln": 2, "layer_gate": 1, "transition_gate": 1, "attn_mask": 0})   # one layer: AdaLN twice (layer + transition), two gates
        self.assertEqual(rec["fallback_after"], {n: 0 for n in fl_sites()})
        self.assertEqual((rec["rows_saved_cond"], rec["max_rows"], rec["context"]["sampling"]), (3, 4, 1))
        self.assertEqual(rec["gate"], [])

    def test_real_rows_inside_the_sampler_take_the_stock_statement_counted_not_a_fallback(self):
        """The atom transformers' layers are the same classes with a real (B*NW, W, C) conditioning: `rows_real`, the stock statement inline."""
        rc, out, err = _run(r'''
B, N, C = 3, 5, 8
layer = T.DiffusionTransformerLayer(heads=2, dim=C); layer.eval()
a = torch.randn(B, N, C); s = torch.randn(B, N, C)                       # materialized, non-uniform rows: a real copy, like the atom layers' c
with torch.no_grad():
    y0 = layer(a, s, None, torch.ones(B, N))                              # outside: stock
    y1 = sampling(lambda: layer(a, s, None, torch.ones(B, N)), rows=B)    # inside, real rows: stock inline
print("RECORD " + json.dumps({"equal": torch.equal(y0, y1), "served": {n: fl.SITES[n].stats()["served"] for n in fl.SITES}, "rows_real": fl.STATS["inline_by"]["rows_real"], "gate": fl.gate()}))
''', self.site)
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertTrue(rec["equal"]); self.assertEqual(set(rec["served"].values()), {0}); self.assertGreater(rec["rows_real"], 0)
        self.assertTrue(any("no served call" in p or "served" in p for p in rec["gate"]), rec["gate"])   # rows > 1 seen in the sampler and nothing served: the gate names it

    def test_a_site_bypassed_inside_a_capture_fails_the_gate(self):
        """Under the graph sampler the whole denoiser step is captured and replayed: a trusted site must be SERVED inside the capture (the core's
        contract); were it bypassed, the replayed graph would carry the stock statement — so any bypassed count is a gate problem by name."""
        rc, out, err = _run(r'''
from opt_core.capture import hoist
B, N, C = 4, 5, 8
sc = E.SingleConditioning(token_s=C // 2); sc.eval()
st1 = torch.randn(1, N, C // 2); si1 = torch.randn(1, N, C // 2)
hoist.set_capturing(True)                       # the core's own capture flag (torch's stream-capture state is the other source, GPU only)
try:
    with torch.no_grad():
        s, _ = sampling(lambda: sc(torch.full((B,), 0.5), st1.repeat_interleave(B, 0), si1.repeat_interleave(B, 0)), rows=B)
finally:
    hoist.set_capturing(False)
st = fl.SITES["cond"].stats()
served_in_capture = st["served"] == 1 and st["bypassed"] == 0
fl.SITES["cond"].stats_["bypassed"] += 7                                    # what an un-served capture would have counted
print("RECORD " + json.dumps({"served_in_capture": served_in_capture, "gate": fl.gate()}))
''', self.site)
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertTrue(rec["served_in_capture"], "opt_core RowDedup: a trusted apply must be served inside a capture (narrow/expand are capturable)")
        self.assertTrue(any("bypassed=7" in p for p in rec["gate"]), rec["gate"])

    def test_attn_bf16_eligibility_rules_and_passthrough_census(self):
        """The sampler's fp32 attention with a 4-d float mask is eligible — the token layers' full (queries == keys) calls AND the atom
        transformers' 32x128 windowed calls; the trunk (outside the sampler) and mask-less calls pass through to the stock call, counted by
        reason. On a CPU box the bf16 memory-efficient kernel does not exist: the first eligible call disables the lever BY NAME and the stock
        call serves (the GATE-FAIL / DISABLED lines the caller reads as a partial run) — the GPU self-test holds the served path."""
        rc, out, err = _run(r'''
F = torch.nn.functional
q = torch.randn(4, 2, 6, 4); kw = torch.randn(4, 2, 12, 4); m = torch.zeros(4, 2, 6, 6)
o0 = F.scaled_dot_product_attention(q, q, q, attn_mask=m)                                   # outside the sampler
def inside():
    F.scaled_dot_product_attention(q, kw, kw, attn_mask=torch.zeros(4, 2, 6, 12))           # windowed (queries != keys): eligible (the atom layers)
    F.scaled_dot_product_attention(q, q, q)                                                 # no mask
    return F.scaled_dot_product_attention(q, q, q, attn_mask=m)                              # eligible
o1 = sampling(inside, rows=4)
import atexit; atexit.unregister(fl.report_lines); fl.report_lines()
print("RECORD " + json.dumps({"same_outside": torch.equal(o0, F.scaled_dot_product_attention.__wrapped_stock__(q, q, q, attn_mask=m)) if hasattr(F.scaled_dot_product_attention, "__wrapped_stock__") else True,
                             "close_inside": torch.allclose(o1, o0, atol=5e-2), "eligible": fl.STATS["attn_eligible"], "calls": fl.STATS["attn_calls"],
                             "pt": dict(fl.STATS["passthrough_by"]), "disabled": fl.STATS["attn_disabled_reason"], "gate": fl.gate()}))
''', self.site)
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertEqual((rec["pt"]["not_sampling"], rec["pt"]["no_float_mask"], rec["eligible"]), (1, 1, 2)); self.assertNotIn("windowed", rec["pt"])
        self.assertTrue(rec["close_inside"])
        if rec["calls"] == 0:                                                        # CPU: no bf16 memory-efficient kernel -> disabled by name, stock served, gate + lines say so
            self.assertTrue(rec["disabled"]); self.assertTrue(any(p.startswith("attn_bf16 disabled") for p in rec["gate"]), rec["gate"])
            self.assertRegex(err, stack.RUNNER_DISABLED_LINES["attn_bf16"].pattern.lstrip("^"))
            self.assertIn("[fl_levers] attn_bf16 GATE-FAIL attn_bf16 disabled:", out)
        m = stack.RUNNER_STATS_LINE.match(next(ln for ln in out.splitlines() if ln.startswith("[fl_levers] attn_bf16 stats:")))
        self.assertEqual(m.group(1), "attn_bf16"); self.assertEqual(json.loads(m.group(2))["attn_eligible"], 2)
        gate_census = json.loads(m.group(2))["attn_gate"]                            # the core size gate's census: every eligible call it decided (one on CPU: the first disables the lever by name), served or re-booked as a named fallback
        self.assertIn(gate_census["calls"], (1, 2)); self.assertEqual(gate_census["served"] + gate_census["fallback"], gate_census["calls"])
        if rec["calls"] == 0:
            self.assertRegex(out, r"\[fl_levers\] LEVER name=fl_levers\.attn_bf16 state=skipped reason=\S+ impl=opt_core\.attn\.sdpa_bias origin=core strategy=F5\.flash_attn_dense compute=bf16 backend=efficient event=none served=0 fallback=1 ")
        self.assertEqual(rec["pt"]["reentrant"], 1)                                  # the core's one inner SDPA call (served, or refused by the kernel on this box) passed through the wrapper, never counted eligible

    def test_a_per_row_sigma_is_not_trusted(self):
        """The denoiser call's sigma must be ONE value for the batch for any site to serve: a Python float (stock sample()), a one-element
        tensor, a stride-0 expanded view, or the partner graph sampler's static buffer while its marked step body runs; a per-row (B,) tensor —
        which upstream's signature admits — makes the call untrusted: counted `context_by.sigma_per_row`, every site runs the stock statement."""
        rc, out, err = _run(r'''
B, N, C = 4, 5, 8
sc = E.SingleConditioning(token_s=C // 2); sc.eval()
ad = D.AtomDiffusion(); ad.eval()
args = lambda: (torch.full((B,), 0.5), torch.zeros(1, N, C // 2).repeat_interleave(B, 0), torch.zeros(1, N, C // 2).repeat_interleave(B, 0))
call = lambda sigma: ad.preconditioned_network_forward(torch.zeros(B, 3), sigma, {"s_trunk": torch.zeros(1, N, C), "multiplicity": B, "fn": lambda: sc(*args())}, training=False)
with torch.no_grad():
    call(torch.linspace(0.1, 0.9, B))                                            # per-row: untrusted
    served0 = fl.SITES["cond"].stats()["served"]; inline0 = fl.STATS["inline_by"]["not_sampling"]
    call(0.5); call(torch.tensor(0.5)); call(torch.tensor(0.5).expand(B))       # float, 0-d, expanded view: trusted
    class G:                                                                     # the partner's step-graph object: its static (B,) sigma buffer, marked while the body runs
        static = {"sigma": torch.full((B,), 0.5)}
    fl._graph_body_marked(lambda self: call(self.static["sigma"]))(G())
    call(G.static["sigma"])                                                      # the same buffer outside the marked body: per-row again
print("RECORD " + json.dumps({"context": dict(fl.STATS["context_by"]), "sigma": dict(fl.STATS["sigma_by"]), "served0": served0, "inline0": inline0,
                             "served": fl.SITES["cond"].stats()["served"], "sampling_calls": fl.STATS["sampling_calls"], "gate": fl.gate()}))
''', self.site, FL_ATTN_BF16="0")
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertEqual((rec["served0"], rec["inline0"]), (0, 1))                    # the per-row call served nothing: SingleConditioning ran its stock statement, counted
        self.assertEqual((rec["context"]["sigma_per_row"], rec["context"]["sampling"]), (2, 4))
        self.assertEqual(rec["sigma"], {"float": 1, "scalar": 1, "expanded": 1, "graph_static": 1})
        self.assertEqual((rec["served"], rec["sampling_calls"]), (4, 4))
        self.assertFalse([p for p in rec["gate"] if "sigma" in p or p.startswith("cond:")], rec["gate"])   # trusted calls were seen and the cond site served: no sigma / cond problem (the idle layer sites are named, as in the exit-lines test; a run with ONLY per-row calls is a sigma problem, below)
        rc, out, err = _run(r'''
B, N, C = 4, 5, 8
sc = E.SingleConditioning(token_s=C // 2); sc.eval(); ad = D.AtomDiffusion(); ad.eval()
with torch.no_grad():
    ad.preconditioned_network_forward(torch.zeros(B, 3), torch.linspace(0.1, 0.9, B), {"s_trunk": torch.zeros(1, N, C), "multiplicity": B,
                                      "fn": lambda: sc(torch.full((B,), 0.5), torch.zeros(B, N, C // 2), torch.zeros(B, N, C // 2))}, training=False)
print("RECORD " + json.dumps({"gate": fl.gate()}))
''', self.site, FL_ATTN_BF16="0")
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertTrue(any(p.startswith("cond_dedup: 1 denoiser call(s) carried a per-row sigma") for p in rec["gate"]), rec["gate"])

    def test_exit_lines_are_the_ones_the_callers_census_reads(self):
        rc, out, err = _run(r'''
B, N, C = 4, 5, 8
sc = E.SingleConditioning(token_s=C // 2); sc.eval()
with torch.no_grad():
    sampling(lambda: sc(torch.full((B,), 0.5), torch.zeros(1, N, C // 2).repeat_interleave(B, 0), torch.zeros(1, N, C // 2).repeat_interleave(B, 0)), rows=B)
''', self.site, FL_ATTN_BF16="0")
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        stats = [ln for ln in lines if stack.RUNNER_STATS_LINE.match(ln)]
        self.assertEqual([stack.RUNNER_STATS_LINE.match(ln).group(1) for ln in stats], ["cond_dedup"])
        d = json.loads(stack.RUNNER_STATS_LINE.match(stats[0]).group(2))
        self.assertEqual((d["sites"]["cond"]["served"], d["sites"]["cond"]["rows_saved"], d["max_rows"]), (1, 3, 4))
        ev = [ln for ln in lines if ln.startswith("[fl_levers] LEVER name=")]
        self.assertEqual(len(ev), 5)                                                # one evidence line per RowDedup site (attn_mask's site exists, idle)
        self.assertTrue(all("impl=row_dedup" in ln and " class=tolerance " in ln for ln in ev), ev)   # the numerics class the mode table declares for every site, on the core's evidence line
        gate_fail = [ln for ln in lines if "GATE-FAIL" in ln]
        self.assertTrue(gate_fail, "the layer sites saw rows > 1 (max_rows 4) and never served: a GATE-FAIL line")   # only SingleConditioning ran: adaln/gates idle -> named
        self.assertTrue(all(stack.RUNNER_DISABLED_LINES["cond_dedup"].match(ln) for ln in gate_fail), gate_fail)
        res = modes.resolve("fast", HOME)
        installed = "[fl_levers] cond_dedup installed sites=cond,adaln,layer_gate,transition_gate strategy=F7.row_dedup class=tolerance graph_integration=0"
        evd = stack.levers_from_lines([installed] + gate_fail, res)
        self.assertIn("cond_dedup", evd["levers_fallback"])                          # installed AND gate-failed: the run is partial by name
        evd = stack.levers_from_lines([installed], res)
        self.assertIn("cond_dedup", evd["levers_applied"])

    def test_dit_fused_without_cond_dedup_is_refused_by_name_at_install(self):
        rc, out, err = _run("print('unreachable')\n", self.site, FL_COND_DEDUP="0", FL_ATTN_BF16="0", FL_DIT_FUSED="1")
        self.assertNotEqual(rc, 0)
        self.assertIn("dit_fused REFUSED reason=requires_cond_dedup", err)
        self.assertNotIn("unreachable", out)

    def test_the_attention_backend_pin_is_part_of_attn_bf16s_install_line(self):
        """FL_ATTN_BACKEND is exported by the fast row (cudnn; `efficient` is the other pin the lever takes); the pin appears on attn_bf16's install line,
        which is what stack.RUNNER_LINES["attn_cudnn"] reads, and in STATS["attn_backend"] (the in-process probe compares it to the export)."""
        for pin in ("cudnn", "efficient"):
            rc, out, err = _run("import atexit; atexit.unregister(fl.report_lines); fl.report_lines()\n"
                                 "print('RECORD ' + json.dumps(fl.STATS['attn_backend']))\n", self.site, FL_ATTN_BACKEND=pin)
            self.assertEqual(rc, 0, err)
            rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
            self.assertEqual(rec, pin)
            line = [l for l in (out + err).splitlines() if stack.RUNNER_LINES["attn_bf16"].match(l)]
            self.assertEqual(len(line), 1, out + err)
            self.assertIn(f"backend={pin}", line[0])
            self.assertEqual(bool(stack.RUNNER_LINES["attn_cudnn"].match(line[0])), pin == "cudnn")

    def test_dit_fused_off_the_gpu_is_disabled_by_name_not_silently(self):
        """On the CPU stub stack the routed core kernel cannot serve (no triton / no CUDA): the lever is either off by name at install (the core's
        route refusal or an import error, printed as a DISABLED line the caller's census reads) or installed; a served count without a GPU is
        never claimed."""
        rc, out, err = _run("import atexit; atexit.unregister(fl.report_lines); fl.report_lines()\n"
                             "print('RECORD ' + json.dumps([fl.STATS['dit_enabled'], fl.STATS['dit_active'], fl.STATS['dit_disabled_reason']]))\n",
                             self.site, FL_DIT_FUSED="1")
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        enabled, active, why = rec
        self.assertTrue(enabled)
        if not active:
            self.assertTrue(why, "dit inactive without a named reason")
            self.assertTrue(any(stack.RUNNER_DISABLED_LINES["dit_fused"].match(l) or "dit_fused DISABLED" in l for l in (out + err).splitlines()), out + err)


def fl_sites():
    return ("cond", "adaln", "layer_gate", "transition_gate", "attn_mask")


if __name__ == "__main__":
    unittest.main()
