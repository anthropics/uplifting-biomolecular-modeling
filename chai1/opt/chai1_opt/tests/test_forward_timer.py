"""The FORWARD timer line (one grammar on the stock and the kit routes) and the W5 ESM memo scope switch (CHAI1_OPT_ESM_MEMO_SCOPE)."""
import io
import json
import os
import re
import sys
import types
import unittest

from chai1_opt.tests import _stubs  # noqa: F401  (torch / chai_lab stand-ins importable)
from chai1_opt import _autoload, forward_timer, report, stack

GRAMMAR = re.compile(r"^\[chai1-opt(?: stock)?\] FORWARD item=\S+ out=\S+ tokens=(\d+|na) crop=(\d+|na) forward_s=\d+\.\d{3} \(run_folding_on_context, cuda-synced\)$")


def _fake_chai1():
    mod = types.ModuleType("chai_lab.chai1")
    mod.calls = []
    def run_folding_on_context(feature_context, *, output_dir=None, num_trunk_recycles=3, **kw):
        mod.calls.append((feature_context, os.fspath(output_dir), num_trunk_recycles)); return ("candidates", feature_context)
    def make_all_atom_feature_context(*args, **kw):
        mod.calls.append(("features", kw.get("fasta_file"))); return "ctx"
    mod.run_folding_on_context = run_folding_on_context; mod.make_all_atom_feature_context = make_all_atom_feature_context
    return mod


def _reset_timer_state():
    del forward_timer.LOG[:]


class _Ctx:                                                                        # a feature context with the one attribute the timer reads
    class structure_context:  # noqa: N801
        num_tokens = 400


class TestForwardLine(unittest.TestCase):
    def setUp(self):
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        _reset_timer_state()

    def tearDown(self):
        _stubs.remove(self.saved); _reset_timer_state()

    def test_one_grammar_on_both_routes_and_pass_through(self):
        for prefix in (report.PREFIX, report.STOCK_PREFIX):
            mod = _fake_chai1(); err = io.StringIO()
            self.assertTrue(forward_timer.install(mod, prefix, stream=err)); self.assertFalse(forward_timer.install(mod, prefix, stream=err))   # idempotent
            r = mod.run_folding_on_context(_Ctx(), output_dir="/out/pred/n0400_43ls__1/seed_42", num_trunk_recycles=3)
            self.assertEqual(r, ("candidates",) + (r[1],)); self.assertEqual(mod.calls[-1][1:], ("/out/pred/n0400_43ls__1/seed_42", 3))        # arguments and result untouched
            ln = [l for l in err.getvalue().strip().splitlines() if " FORWARD " in l][-1]
            self.assertRegex(ln, GRAMMAR); self.assertTrue(ln.startswith(f"{prefix} FORWARD item=n0400_43ls__1 out=seed_42 tokens=400 crop="))
            self.assertEqual(forward_timer.LOG[-1]["item"], "n0400_43ls__1"); self.assertEqual(forward_timer.LOG[-1]["tokens"], 400)
            forward_timer.uninstall(mod); self.assertFalse(hasattr(mod.run_folding_on_context, "chai1_opt_lever"))
        self.assertEqual(forward_timer.tally_fields()[0].split()[0], "forward_calls=2")
        self.assertEqual(forward_timer.line("[chai1-opt]", "x", "seed_0", 12, 256, 1.5), "[chai1-opt] FORWARD item=x out=seed_0 tokens=12 crop=256 forward_s=1.500 (run_folding_on_context, cuda-synced)")

    def test_a_context_without_tokens_is_timed_all_the_same(self):
        mod = _fake_chai1(); err = io.StringIO(); forward_timer.install(mod, report.PREFIX, stream=err)
        mod.run_folding_on_context(object(), output_dir="/o/item/seed_1")
        self.assertRegex([l for l in err.getvalue().splitlines() if " FORWARD " in l][0], GRAMMAR); self.assertIn(" tokens=na crop=na ", err.getvalue())

    def test_one_line_per_fold_and_the_peak_counters_restart_at_fold_entry(self):
        """The wrapper prints the FORWARD line and nothing else, and restarts the allocator's peak counters before the fold body (the routes'
        rows read ``max_mem_alloc_gb`` after the fold through ``read_peak`` / ``torch.cuda``)."""
        events = []; mem = {"alloc": 0.0, "reserved": 0.0}
        tc = self.torch.cuda
        tc.reset_peak_memory_stats = lambda *a, **k: (events.append("reset"), mem.update(alloc=0.0, reserved=0.0))
        tc.max_memory_allocated = lambda *a, **k: mem["alloc"]; tc.max_memory_reserved = lambda *a, **k: mem["reserved"]
        mod = _fake_chai1(); inner = mod.run_folding_on_context
        def run_folding_on_context(feature_context, **kw):
            events.append("fold"); mem.update(alloc=12.25 * 2**30, reserved=14.5 * 2**30); return inner(feature_context, **kw)
        mod.run_folding_on_context = run_folding_on_context
        err = io.StringIO(); forward_timer.install(mod, report.STOCK_PREFIX, stream=err)
        mod.run_folding_on_context(_Ctx(), output_dir="/out/pred/n0400_43ls__1/seed_42"); mod.run_folding_on_context(_Ctx(), output_dir="/out/pred/itemB/seed_42")
        self.assertEqual(events, ["reset", "fold", "reset", "fold"])                                  # once per fold, before the fold body
        out = err.getvalue().strip().splitlines()
        self.assertEqual(len(out), 2); self.assertTrue(all(GRAMMAR.match(l) for l in out), out)         # one FORWARD line per fold, no other line
        self.assertEqual(forward_timer.read_peak(), {"alloc_gib": 12.25, "reserved_gib": 14.5})           # GiB since the fold's entry
        self.assertEqual(sorted(forward_timer.LOG[-1]), ["crop", "forward_s", "item", "out", "tokens"])
        forward_timer.uninstall(mod); self.assertIs(mod.run_folding_on_context, run_folding_on_context)

    def test_no_cuda_allocator_times_all_the_same_and_reads_no_peak(self):
        _stubs.remove(self.saved); self.torch, self.chai1, self.esm, self.saved = _stubs.install(cuda_available=False)
        events = []; self.torch.cuda.reset_peak_memory_stats = lambda *a, **k: events.append("reset")
        mod = _fake_chai1(); err = io.StringIO(); forward_timer.install(mod, report.PREFIX, stream=err)
        mod.run_folding_on_context(_Ctx(), output_dir="/out/pred/n0400_43ls__1/seed_42")
        self.assertEqual(events, []); self.assertIsNone(forward_timer.read_peak()); self.assertFalse(forward_timer.reset_peak())
        self.assertRegex(err.getvalue().strip(), GRAMMAR)


class TestEsmMemoScope(unittest.TestCase):
    def setUp(self):
        stack.ESM_SCOPE_STATS.update(scope=None, clears=0)

    def test_the_word(self):
        self.assertEqual(stack.esm_memo_scope({}), "global"); self.assertEqual(stack.esm_memo_scope({"CHAI1_OPT_ESM_MEMO_SCOPE": ""}), "global")
        self.assertEqual(stack.esm_memo_scope({"CHAI1_OPT_ESM_MEMO_SCOPE": "input"}), "input"); self.assertEqual(stack.esm_memo_scope({"CHAI1_OPT_ESM_MEMO_SCOPE": "GLOBAL"}), "global")
        with self.assertRaises(stack.ActivationError) as cm:
            stack.esm_memo_scope({"CHAI1_OPT_ESM_MEMO_SCOPE": "item"})
        self.assertIn("CHAI1_OPT_ESM_MEMO_SCOPE='item' is not one of global|input", str(cm.exception))

    def test_input_scope_empties_the_memo_per_feature_context_and_global_keeps_it(self):
        mod = _fake_chai1(); ns = {"levels": {"W1", "W2", "W5"}, "ESM_CACHE": {"SEQA": 1, "SEQB": 2}}
        stack.install_esm_memo_scope(mod, ns, "global")
        mod.make_all_atom_feature_context(fasta_file="a.fasta"); self.assertEqual(len(ns["ESM_CACHE"]), 2)          # global: nothing wrapped, the memo serves every input
        self.assertFalse(getattr(mod.make_all_atom_feature_context, "chai1_opt_esm_scope", False))
        stack.install_esm_memo_scope(mod, ns, "input"); stack.install_esm_memo_scope(mod, ns, "input")              # idempotent
        self.assertEqual(mod.make_all_atom_feature_context(fasta_file="b.fasta"), "ctx"); self.assertEqual(ns["ESM_CACHE"], {})   # emptied when the next input's features are built
        ns["ESM_CACHE"]["SEQC"] = 3                                                                                   # filled while that input folds (chains inside one input still hit)
        mod.make_all_atom_feature_context(fasta_file="c.fasta"); self.assertEqual(ns["ESM_CACHE"], {})
        self.assertEqual(stack.ESM_SCOPE_STATS, {"scope": "input", "clears": 2}); self.assertEqual(stack.esm_scope_tally_fields(), ["esm_memo_scope=input esm_memo_clears=2"])

    def test_the_msa_form_mark_stays_visible_under_the_scope_wrapper(self):
        mod = _fake_chai1(); mod.make_all_atom_feature_context.chai1_opt_lever = "msa_form"
        stack.install_esm_memo_scope(mod, {"levels": {"W1", "W5"}, "ESM_CACHE": {}}, "input")
        self.assertEqual(mod.make_all_atom_feature_context.chai1_opt_lever, "msa_form"); self.assertTrue(mod.make_all_atom_feature_context.chai1_opt_esm_scope)

    def test_a_namespace_that_is_not_the_kits_is_named_not_ignored(self):
        """scope=input on a dict without the kit statements' shape (no `levels`), or with W5 among the levels but no ESM_CACHE, raises by
        name — the memo is never left global silently."""
        mod = _fake_chai1()
        with self.assertRaises(stack.ActivationError) as cm:
            stack.install_esm_memo_scope(mod, {"name": "NVIDIA H100", "cc": "9.0"}, "input")
        self.assertIn("not the kit's (no `levels`)", str(cm.exception))
        with self.assertRaises(stack.ActivationError):
            stack.install_esm_memo_scope(mod, {"levels": {"W1", "W5"}}, "input")
        stack.install_esm_memo_scope(mod, {"levels": {"W1"}}, "input")                      # no W5 in the process: nothing memoised, nothing to scope — fine
        self.assertTrue(mod.make_all_atom_feature_context.chai1_opt_esm_scope)
        stack.install_esm_memo_scope(_fake_chai1(), {"anything": 1}, "global")               # global wraps nothing and needs nothing

    def test_the_driver_hands_the_workers_namespace(self):
        """driver.py scopes the memo on the worker's own namespace (the dict its statements run in), never on another dict of the window."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(stack.__file__)), "driver.py")).read()
        self.assertIn("worker_ns = g", src); self.assertIn('stack.install_esm_memo_scope(chai1_mod, worker_ns, rep["esm_memo_scope"])', src)

    def test_the_kit_routes_depth_line_reads_the_alignment_rows(self):
        """install_msa_form's wrapper prints `[chai1-opt] MSA item=… chains=2 depth=3,5 dir=…` from the .aligned.pqt rows in the run's dir."""
        import tempfile
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "q_1to1.fasta"); open(fa, "w").write(">protein|name=A\nACDE\n>protein|name=B\nFGHIK\n")
            msas = _msa_fixture(self, td, {"ACDE": 3, "FGHIK": 5})
            calls = []; fake = types.SimpleNamespace(make_all_atom_feature_context=lambda **kw: calls.append(kw) or "ctx")
            stack.MSA_FORMS.clear(); stack.install_msa_form(fake); err = io.StringIO()
            with redirect_stderr(err):
                fake.make_all_atom_feature_context(fasta_file=fa, msa_directory=msas)
            self.assertIn(f"[chai1-opt] MSA item=q_1to1.fasta chains=2 depth=3,5 dir={msas}", err.getvalue().splitlines())
            self.assertRegex([l for l in err.getvalue().splitlines() if " MSA item=" in l][0], report.MSA_DEPTH_GRAMMAR)
            self.assertEqual(os.fspath(calls[0]["msa_directory"]), msas); stack.MSA_FORMS.clear()

    def test_the_name_is_declared(self):
        self.assertIn("CHAI1_OPT_ESM_MEMO_SCOPE", stack.DECLARED_ENV); self.assertIn("CHAI1_OPT_ESM_MEMO_SCOPE", _autoload.DECLARED_ENV)
        self.assertEqual(_autoload.undeclared({"CHAI1_OPT_ESM_MEMO_SCOPE": "input", "CHAI1_OPT": "fast"}), [])
        cfg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__)))), "configs", "h100.env")).read()
        self.assertIn("CHAI1_OPT_ESM_MEMO_SCOPE", cfg); self.assertNotRegex(cfg, r"(?m)^\s*export\s+CHAI1_OPT_ESM_MEMO_SCOPE=")     # documented there, never exported by the config


def _msa_fixture(case, td, rows_by_seq):
    """An MSA directory holding one `<seq>.aligned.pqt` per sequence with the given row counts (chai-lab's naming and its FASTA reader are
    stood in: expected_basename(seq) = f"{seq}.aligned.pqt"; read_inputs parses `>` records). Real parquet files when pyarrow is importable,
    else stock_fold.pqt_rows is stood in with the counts (the row counter itself is a one-liner over parquet metadata)."""
    from chai1_opt import stock_fold
    msas = os.path.join(td, "msas"); os.makedirs(msas, exist_ok=True)
    pm = types.ModuleType("chai_lab.data.parsing.msas.aligned_pqt"); pm.expected_basename = lambda seq: f"{seq}.aligned.pqt"
    saved = {n: sys.modules.get(n) for n in ("chai_lab.data.parsing", "chai_lab.data.parsing.msas", "chai_lab.data.parsing.msas.aligned_pqt", "chai_lab.data.dataset.inference_dataset")}
    sys.modules["chai_lab.data.parsing"] = types.ModuleType("chai_lab.data.parsing"); sys.modules["chai_lab.data.parsing.msas"] = types.ModuleType("chai_lab.data.parsing.msas")
    sys.modules["chai_lab.data.parsing.msas.aligned_pqt"] = pm
    ds = types.ModuleType("chai_lab.data.dataset.inference_dataset")
    ds.read_inputs = lambda p: [types.SimpleNamespace(sequence=l.strip()) for l in open(p) if l.strip() and not l.startswith(">")]
    sys.modules["chai_lab.data.dataset.inference_dataset"] = ds
    def restore():
        for n, m in saved.items():
            if m is None: sys.modules.pop(n, None)
            else: sys.modules[n] = m
    case.addCleanup(restore)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        for seq, n in rows_by_seq.items():
            pq.write_table(pa.table({"sequence": [seq] * n}), os.path.join(msas, f"{seq}.aligned.pqt"))
    except ImportError:
        for seq, n in rows_by_seq.items():
            open(os.path.join(msas, f"{seq}.aligned.pqt"), "wb").write(b"")
        real = stock_fold.pqt_rows; stock_fold.pqt_rows = lambda p: rows_by_seq[os.path.basename(str(p))[:-len(".aligned.pqt")]]
        case.addCleanup(setattr, stock_fold, "pqt_rows", real)
    return msas


class TestStockProcessLoopsTheItems(unittest.TestCase):
    """stock_fold.main runs every item of the plan in ONE process: one environment proof, one FORWARD line per fold, one row per (item, seed)
    — an item that fails (its fold raises, its output directory is not empty, its inputs cannot be read) is that item's failed row by name and
    the next items still fold (exit 2 = some row failed, never a silent drop)."""

    def setUp(self):
        self.torch, self.chai1, self.esm, self.saved = _stubs.install()
        del forward_timer.LOG[:]
        import chai1_opt.outputs as outputs
        self.outputs = outputs; self.saved_out = outputs.n_samples
        outputs.n_samples = lambda cand: 1
        tc = self.torch.cuda
        for name, fn in (("is_available", lambda: False), ("synchronize", lambda: None), ("max_memory_allocated", lambda: 0), ("max_memory_reserved", lambda: 0),
                         ("reset_peak_memory_stats", lambda: None)):
            if not hasattr(tc, name): setattr(tc, name, fn)
        chai1 = sys.modules["chai_lab.chai1"]; self.saved_c1 = {k: getattr(chai1, k, None) for k in ("run_inference", "run_folding_on_context")}
        def run_folding_on_context(feature_context, *, output_dir=None, **kw):
            open(os.path.join(os.fspath(output_dir), "pred.model_idx_0.cif"), "w").write("data_x\n"); return types.SimpleNamespace(pae=[[0.0]])
        def run_inference(fasta_file, *, output_dir, msa_directory=None, seed=None, device=None, num_trunk_recycles=3, num_diffn_timesteps=200,
                          num_diffn_samples=5, num_trunk_samples=1, use_esm_embeddings=True, use_msa_server=False, use_templates_server=False,
                          recycle_msa_subsample=0, low_memory=True, msa_server_url="https://api.colabfold.com", constraint_path=None, template_hits_path=None, **kw):
            if "bad" in os.fspath(fasta_file): raise RuntimeError("boom in the fold")
            return chai1.run_folding_on_context(types.SimpleNamespace(structure_context=types.SimpleNamespace(num_tokens=7)), output_dir=output_dir)
        chai1.run_inference = run_inference; chai1.run_folding_on_context = run_folding_on_context
        ds = types.ModuleType("chai_lab.data.dataset.inference_dataset"); ds.read_inputs = lambda p: [types.SimpleNamespace(sequence="ACDE")]
        self.saved_ds = sys.modules.get("chai_lab.data.dataset.inference_dataset"); sys.modules["chai_lab.data.dataset.inference_dataset"] = ds
        self.saved_env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("CHAI1_OPT") or k in ("CHAI_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")}
        self.saved_tally = dict(report._TALLY)                                                             # stock_fold.main registers its own exit tally; other tests' tallies must not inherit it
        _reset_timer_state()

    def tearDown(self):
        chai1 = sys.modules.get("chai_lab.chai1")
        for k, v in self.saved_c1.items():
            if v is not None and chai1 is not None: setattr(chai1, k, v)
        if self.saved_ds is None: sys.modules.pop("chai_lab.data.dataset.inference_dataset", None)
        else: sys.modules["chai_lab.data.dataset.inference_dataset"] = self.saved_ds
        self.outputs.n_samples = self.saved_out
        os.environ.update(self.saved_env); _stubs.remove(self.saved); _reset_timer_state()
        report._TALLY.clear(); report._TALLY.update(self.saved_tally)

    def test_noesm_reaches_the_stock_call_and_leaves_the_msa_directory(self):
        """ESM off + low_memory off on the stock route: run_inference receives use_esm_embeddings=False and low_memory=False, the MSA directory
        exactly as the plan names it, and the pass-through gate accepts the plan (signature defaults + exactly the two flags given)."""
        self._noesm_family_reaches_the_stock_call(low_memory=False, departures={"low_memory": False, "use_esm_embeddings": False})

    def test_noesm_lowmem_reaches_the_stock_call_with_upstreams_low_memory(self):
        """ESM off alone on the stock route: the composed run_inference call receives use_esm_embeddings=False and low_memory=True (the signature's
        own default, chai1.py:502), the MSA directory untouched, and the pass-through gate accepts it (its one flag: use_esm_embeddings)."""
        self._noesm_family_reaches_the_stock_call(low_memory=True, departures={"use_esm_embeddings": False})

    def _noesm_family_reaches_the_stock_call(self, low_memory, departures):
        import tempfile
        from contextlib import redirect_stderr
        from chai1_opt import settings, stock_fold
        real_proof = stock_fold.env_proof
        stock_fold.env_proof = lambda *a, **k: {"schema": "chai1_opt.stock_env_proof/1", "clean": True, "torch_imported_before_proof": True}
        self.addCleanup(setattr, stock_fold, "env_proof", real_proof)
        chai1 = sys.modules["chai_lab.chai1"]; inner = chai1.run_inference; seen = []
        def run_inference(fasta_file, *, output_dir, msa_directory=None, seed=None, device=None, num_trunk_recycles=3, num_diffn_timesteps=200,
                          num_diffn_samples=5, num_trunk_samples=1, use_esm_embeddings=True, use_msa_server=False, use_templates_server=False,
                          recycle_msa_subsample=0, low_memory=True, msa_server_url="https://api.colabfold.com", constraint_path=None, template_hits_path=None):
            seen.append({"use_esm_embeddings": use_esm_embeddings, "low_memory": low_memory, "msa_directory": os.fspath(msa_directory) if msa_directory else None})
            return inner(fasta_file, output_dir=output_dir, msa_directory=msa_directory, seed=seed)
        chai1.run_inference = run_inference
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "a_1to1.fasta"); open(fa, "w").write(">protein|name=A\nACDE\n>protein|name=B\nFGHIK\n")
            msas = _msa_fixture(self, td, {"ACDE": 3, "FGHIK": 5})                                            # two chains: 3 and 5 alignment rows
            st = settings.from_values(departures)
            plan = {"schema": "chai1_opt.plan/2", "fold": dict(st["fold"]), "non_default": dict(st["non_default"]), "det": 0, "msa_dir": msas,
                    "items": [{"id": "a", "key": "a_1to1", "fasta": fa, "seeds": [42]}]}
            pp = os.path.join(td, "plan.json"); json.dump(plan, open(pp, "w"))
            err = io.StringIO()
            with redirect_stderr(err):
                rc = stock_fold.main(["--plan", pp, "--out_dir", td, "--tag", "t"])
            self.assertEqual(rc, 0, err.getvalue()[-600:])
            self.assertEqual(seen, [{"use_esm_embeddings": False, "low_memory": low_memory, "msa_directory": msas}])  # the call received them; the MSA dir untouched
            self.assertEqual([f for f in os.listdir(os.path.join(td, "t")) if f.endswith(".json") or f.endswith(".jsonl")], [])   # no proof / rows file: the pass writes upstream's outputs only
            lines = err.getvalue().splitlines()
            kw = [l for l in lines if " SETTINGS " in l]; dep = [l for l in lines if l.startswith("[chai1-opt stock] MSA item=")]
            self.assertEqual(len(kw), 1); self.assertRegex(kw[0], report.SETTINGS_GRAMMAR)                      # once per pass, the stock prefix
            self.assertIn(" use_esm_embeddings=False", kw[0]); self.assertIn(f" low_memory={low_memory}", kw[0]); self.assertTrue(kw[0].startswith("[chai1-opt stock] SETTINGS constraint_path=None device=cuda:0 "), kw[0]); self.assertNotIn("preset=", kw[0])
            self.assertFalse([l for l in lines if " PEAK" in l or " PHASE" in l], lines)                        # the kit prints the FORWARD line only
            self.assertEqual(dep, [f"[chai1-opt stock] MSA item=a_1to1 chains=2 depth=3,5 dir={msas}"]); self.assertRegex(dep[0], report.MSA_DEPTH_GRAMMAR)
            self.assertIn("[chai1-opt stock] MSA form=directory item=a_1to1 aligned_pqt=2/2 ", err.getvalue())        # the form line, unchanged, still there

    def test_record_settings_line_and_the_gate_refuses_a_tampered_keyword(self):
        """low_memory off on the stock route prints SETTINGS with use_esm_embeddings=True; a plan whose keywords differ from the installed defaults
        beyond the flags it names (use_esm_embeddings flipped off without the flag) is refused by name before any fold."""
        import tempfile
        from contextlib import redirect_stderr
        from chai1_opt import settings, stock_fold
        real_proof = stock_fold.env_proof
        stock_fold.env_proof = lambda *a, **k: {"schema": "chai1_opt.stock_env_proof/1", "clean": True, "torch_imported_before_proof": True}
        self.addCleanup(setattr, stock_fold, "env_proof", real_proof)
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "a_1to1.fasta"); open(fa, "w").write(">protein|name=A\nACDE\n")
            rec = settings.from_values({"low_memory": False})
            for tag, name, fold, want_rc in (("r", "record", dict(rec["fold"]), 0),
                                            ("x", "tampered", dict(rec["fold"], use_esm_embeddings=False), 2)):
                plan = {"schema": "chai1_opt.plan/2", "fold": fold, "non_default": dict(rec["non_default"]), "det": 0, "msa_dir": None,
                        "items": [{"id": "a", "key": "a_1to1", "fasta": fa, "seeds": [42]}]}
                pp = os.path.join(td, f"plan_{tag}.json"); json.dump(plan, open(pp, "w")); err = io.StringIO()
                with redirect_stderr(err):
                    rc = stock_fold.main(["--plan", pp, "--out_dir", td, "--tag", tag])
                self.assertEqual(rc, want_rc, err.getvalue()[-600:])
                if name == "record":
                    st = [l for l in err.getvalue().splitlines() if " SETTINGS " in l]
                    self.assertEqual(len(st), 1); self.assertRegex(st[0], report.SETTINGS_GRAMMAR); self.assertIn(" use_esm_embeddings=True", st[0]); self.assertIn(" low_memory=False", st[0])
                else:
                    self.assertIn("[chai1-opt stock] refusing: the plan's run_inference keywords are not the installed signature's defaults with exactly the flags given {'low_memory': False}", err.getvalue())
                    self.assertNotIn(" SETTINGS ", err.getvalue()); self.assertNotIn(" FORWARD ", err.getvalue())        # refused before any fold

    def test_a_role_less_input_folds_on_the_stock_route(self):
        """A monomer FASTA with no role field, translated by the package (inputs.load / resolve_keys with the kit's own fasta_spec / to_plan):
        the stock caller folds it — one run_inference call per seed into <key>/seed_<s>/, key = the FASTA stem — exit 0."""
        import tempfile
        from contextlib import redirect_stderr
        from chai1_opt import inputs, settings, stock_fold
        from chai1_opt.tests.test_locks import _kit_proto
        real_proof = stock_fold.env_proof
        stock_fold.env_proof = lambda *a, **k: {"schema": "chai1_opt.stock_env_proof/1", "clean": True, "torch_imported_before_proof": True}
        self.addCleanup(setattr, stock_fold, "env_proof", real_proof)
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "mono.fasta"); open(fa, "w").write(">protein|name=A\nACDE\n")
            its = inputs.resolve_keys(inputs.load(fa), _kit_proto())
            plan = inputs.to_plan(its, its.items, {its.items[0].key: [9]}, settings.from_values({}), 0, None)
            self.assertEqual(plan["items"], [{"id": "mono", "key": "mono", "fasta": fa, "seeds": [9]}])
            pp = os.path.join(td, "plan.json"); json.dump(plan, open(pp, "w")); err = io.StringIO()
            with redirect_stderr(err):
                rc = stock_fold.main(["--plan", pp, "--out_dir", td, "--tag", "s"])
            e = err.getvalue()
            self.assertEqual(rc, 0, e[-600:])
            self.assertTrue(os.path.isfile(os.path.join(td, "s", "mono", "seed_9", "pred.model_idx_0.cif")))
            self.assertIn("[chai1-opt stock] SEED mono 9 ok wall=", e); self.assertIn("DONE 1/1 seed-folds ok", e)

    def test_no_seed_named_is_stocks_unseeded_call_and_device_passes_through(self):
        """An item with no seed named on the stock route: ONE run_inference call with seed=None — what `chai-lab fold` without --seed does
        (chai1.py:570-572 sets no seed) — into <key>/seed_none/, its row seed null; `--device` reaches run_inference as given, else cuda:0."""
        import tempfile
        from contextlib import redirect_stderr
        from chai1_opt import settings, stock_fold
        real_proof = stock_fold.env_proof
        stock_fold.env_proof = lambda *a, **k: {"schema": "chai1_opt.stock_env_proof/1", "clean": True, "torch_imported_before_proof": True}
        self.addCleanup(setattr, stock_fold, "env_proof", real_proof)
        chai1 = sys.modules["chai_lab.chai1"]; inner = chai1.run_inference; seen = []
        def run_inference(fasta_file, *, output_dir, msa_directory=None, seed=None, device=None, num_trunk_recycles=3, num_diffn_timesteps=200,
                          num_diffn_samples=5, num_trunk_samples=1, use_esm_embeddings=True, use_msa_server=False, use_templates_server=False,
                          recycle_msa_subsample=0, low_memory=True, msa_server_url="https://api.colabfold.com", constraint_path=None, template_hits_path=None):                        # the installed signature the pass-through gate reads
            seen.append({"seed": seed, "device": device, "out": os.fspath(output_dir)})
            return inner(fasta_file, output_dir=output_dir, msa_directory=msa_directory, seed=seed, device=device, num_trunk_recycles=num_trunk_recycles,
                         num_diffn_timesteps=num_diffn_timesteps, num_diffn_samples=num_diffn_samples, num_trunk_samples=num_trunk_samples,
                         use_esm_embeddings=use_esm_embeddings, use_msa_server=use_msa_server, use_templates_server=use_templates_server,
                         recycle_msa_subsample=recycle_msa_subsample, low_memory=low_memory, msa_server_url=msa_server_url,
                         constraint_path=constraint_path, template_hits_path=template_hits_path)
        chai1.run_inference = run_inference
        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "a.fasta"); open(fa, "w").write(">protein|name=A\nACDE\n")
            for tag, device, want_device in (("u", None, "cuda:0"), ("d", "cuda:1", "cuda:1")):
                plan = {"schema": "chai1_opt.plan/2", "fold": dict(settings.library_defaults()), "non_default": {}, "device": device, "det": 0, "msa_dir": None,
                        "items": [{"id": "a", "key": "a_1to1", "fasta": fa, "seeds": None}]}
                pp = os.path.join(td, f"plan_{tag}.json"); json.dump(plan, open(pp, "w")); err = io.StringIO(); seen.clear()
                with redirect_stderr(err):
                    rc = stock_fold.main(["--plan", pp, "--out_dir", td, "--tag", tag])
                e = err.getvalue()
                self.assertEqual(rc, 0, e[-600:])
                self.assertEqual(seen, [{"seed": None, "device": want_device, "out": os.path.join(td, tag, "a_1to1", "seed_none")}])   # one unseeded call, the device as given / cuda:0
                self.assertIn("[chai1-opt stock] SEED a_1to1 none ok wall=", e); self.assertIn(f" device={want_device} ", e); self.assertIn("[chai1-opt stock] SETTINGS ", e)
                self.assertFalse(os.path.exists(os.path.join(td, tag, "a_1to1.stock_rows.jsonl"))); self.assertNotIn("SEED FAIL", e); self.assertIn("DONE 1/1 seed-folds ok", e)

    def test_one_process_folds_every_item_and_accounts_for_each(self):
        import tempfile
        from contextlib import redirect_stderr
        from chai1_opt import stock_fold
        real_proof = stock_fold.env_proof                                                                     # the proof itself is test_locks' subject; this process (pytest) is not a clean one
        stock_fold.env_proof = lambda *a, **k: {"schema": "chai1_opt.stock_env_proof/1", "clean": True, "torch_imported_before_proof": True}
        self.addCleanup(setattr, stock_fold, "env_proof", real_proof)
        with tempfile.TemporaryDirectory() as td:
            good, bad = os.path.join(td, "good.fasta"), os.path.join(td, "bad.fasta")
            for f in (good, bad): open(f, "w").write(">protein|name=A\nACDE\n")
            os.makedirs(os.path.join(td, "t", "itemD", "seed_0")); open(os.path.join(td, "t", "itemD", "seed_0", "leftover"), "w").write("x")   # a non-empty output dir
            plan = {"schema": "chai1_opt.plan/2", "fold": {"num_trunk_recycles": 3}, "non_default": {}, "det": 0, "msa_dir": None,
                    "items": [{"id": "itemA", "key": "itemA", "fasta": good, "seeds": [0, 1]}, {"id": "itemB", "key": "itemB", "fasta": bad, "seeds": [0]},
                              {"id": "itemC", "key": "itemC", "fasta": good, "seeds": [0]}, {"id": "itemD", "key": "itemD", "fasta": good, "seeds": [0]}]}
            pp = os.path.join(td, "plan.json"); json.dump(plan, open(pp, "w"))
            err = io.StringIO()
            with redirect_stderr(err):
                rc = stock_fold.main(["--plan", pp, "--out_dir", td, "--tag", "t"])
            e = err.getvalue()
            self.assertEqual(rc, 2, e)                                                                          # some row failed → 2, by count
            self.assertEqual(e.count("ENV-CLEAN ok"), 1); self.assertEqual(e.count(" ready route=stock"), 1)     # ONE process: one proof, one ready line
            fw = [l for l in e.splitlines() if " FORWARD " in l]
            self.assertEqual(len(fw), 3, fw); self.assertTrue(all(GRAMMAR.match(l) for l in fw), fw)             # itemA ×2 seeds + itemC; every line in the one grammar
            self.assertEqual([l.split("item=")[1].split()[0] for l in fw], ["itemA", "itemA", "itemC"])
            self.assertFalse([l for l in e.splitlines() if " PEAK" in l or " PHASE" in l])                        # the kit prints the FORWARD line only
            seed_lines = [l for l in e.splitlines() if l.startswith("[chai1-opt stock] SEED ") and not l.startswith("[chai1-opt stock] SEED FAIL ")]
            self.assertEqual([tuple(l.split()[3:6]) for l in seed_lines], [("itemA", "0", "ok"), ("itemA", "1", "ok"), ("itemB", "0", "fail"), ("itemC", "0", "ok"), ("itemD", "0", "fail")])   # every seed-fold's status on its SEED line, in order
            fails = [l for l in e.splitlines() if l.startswith("[chai1-opt stock] SEED FAIL ")]                                     # a failed seed-fold is named with its error …
            self.assertEqual([l.split()[4] for l in fails], ["itemB", "itemD"]); self.assertIn("boom in the fold", fails[0]); self.assertIn("exists and is not empty", fails[1])
            self.assertLess(e.index("SEED FAIL itemB 0"), e.index("SEED itemB 0 fail"))                                            # … before its SEED status line; the next item still folded
            self.assertEqual([f for f in os.listdir(os.path.join(td, "t")) if f.endswith(".jsonl") or f.endswith(".json")], [])   # no rows / proof file beside upstream's outputs
            self.assertIn("DONE 3/5 seed-folds ok", e)


if __name__ == "__main__":
    unittest.main()
