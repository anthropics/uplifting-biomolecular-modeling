"""The big mode (M1, memory): fast's composition (the parallel precompile refused) + the memory levers -trimul_chunk and the XLA pool
fraction, manifests and lines."""
import os
import tempfile
import glob
import unittest
from unittest import mock

from af2ig_opt import cli, modes, registry

from . import _stubs

MEM = ["-trimul_chunk", "256:1473"]


class TestBig(unittest.TestCase):
    def test_resolution(self):
        from af2ig_opt import big
        b = modes.resolve("big", _stubs.KIT); f = modes.resolve("fast", _stubs.KIT); e = modes.resolve("exact", _stubs.KIT)
        nol18 = [x for i, x in enumerate(f.flags) if x != "-tmpl_pointwise_sub" and (i == 0 or f.flags[i - 1] != "-tmpl_pointwise_sub")]
        self.assertEqual((b.flags, b.levers, b.precompile, f.folded_into, b.folded_into), (nol18, [l for l in f.levers if l not in ("L18", "L13", "L15", "L16", "ccache")] + ["mem_fraction", "L13", "L15", "L16", "ccache"], 6, None, None))   # 0.7.4: big = fast minus L18 (measured memory cost at 8192) plus the pool fraction; un-folded   # 0.7.2: fast FOLDED into big — one composed line (0.7.1: big = fast + the pool fraction; -trimul_chunk only when opted in)
        bo = modes.resolve("big", _stubs.KIT, trimul=modes.trimul_flag_value())                                        # the opt-in (AF2IG_OPT_TRIMUL_CHUNK=on): the 0.7.0 composition
        self.assertEqual((bo.flags, bo.levers), (b.flags[:-6] + MEM + b.flags[-6:], [l for l in b.levers if l not in ("mem_fraction", "L13", "L15", "L16", "ccache")] + ["trimul_chunk", "mem_fraction", "L13", "L15", "L16", "ccache"]))
        self.assertEqual([modes.trimul_opt_in({}), modes.trimul_opt_in({"AF2IG_OPT_TRIMUL_CHUNK": "on"}), modes.trimul_opt_in({"AF2IG_OPT_TRIMUL_CHUNK": "128:2000"}), modes.trimul_opt_in({"AF2IG_OPT_TRIMUL_CHUNK": "off"})], [None, "256:1473", "128:2000", None])
        with self.assertRaises(ValueError): modes.trimul_opt_in({"AF2IG_OPT_TRIMUL_CHUNK": "lots"})   # the program line and the deployment lever close every kit mode's set
        self.assertEqual(b.flags, ["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(MEM[1], big.trimul_flag_value())   # big inherits the fast line's levers
        self.assertEqual(big.child_env(), {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}); self.assertEqual((registry.MEMORY_LEVERS, registry.MEMORY_DEFAULT, registry.MEMORY_OPT_IN), (("trimul_chunk", "mem_fraction"), ("mem_fraction",), ("trimul_chunk",)))
        for bad in ():                                            # 0.6.0: the memory mode precompiles too (AOT, no forward): no --precompile refusal left
            with self.assertRaises(ValueError, msg=str(bad)):
                modes.resolve("big", _stubs.KIT, **bad)

    def test_pairstack_switches(self):
        from af2ig_opt import pairstack
        self.assertEqual(pairstack.parse_trimul("256:1473"), (256, 1473)); self.assertIsNone(pairstack.parse_trimul("")); self.assertIsNone(pairstack.parse_trimul(None))
        for bad in ("256", "0:1473", "a:b", "256:-1"):
            with self.assertRaises(ValueError): pairstack.parse_trimul(bad)
        self.assertEqual(pairstack.producers(None), [])
        self.assertEqual(pairstack.producers((256, 1473)), [pairstack.PRODUCER_HAIKU, pairstack.PRODUCER_ROWCHUNK])
        self.assertEqual(pairstack.census(), {"installed": False, "trimul_chunk": None}); self.assertEqual(pairstack.lines(), [])

    def test_monomer_designs_are_accepted(self):
        """A monomer design (one chain, or `-force_monomer`) is an input stock runs: the memory line has no input refusal for it — the driver's
        monomer branch templates nothing and predicts every residue exactly as stock's, and the `pairstack` adapter carries no template gate."""
        from af2ig_opt import pairstack
        self.assertFalse(hasattr(pairstack, "template_gate")); self.assertNotIn("template_gate", pairstack.census())
        drv = open(os.path.join(_stubs.KIT, "patches", "patched_files", "af2_initial_guess", "predict_pdb.py"), encoding="utf-8").read()
        mono = drv[drv.index("        if feat_holder.monomer:\n            # For monomers predict all residues"):drv.index("        template_dict = af2_util.generate_template_features(")]
        self.assertEqual(mono, "        if feat_holder.monomer:\n            # For monomers predict all residues\n            feat_holder.residue_mask = [False for i in range(len(feat_holder.seq))]\n"
                               "        else:\n            # For interfaces fix the target and predict the binder\n"
                               "            feat_holder.residue_mask = [int(i) > feat_holder.binderlen for i in range(len(feat_holder.seq))]\n\n")   # stock predict.py's statements, nothing between them and the template features
        for pth in glob.glob(os.path.join(_stubs.KIT, "patches", "[0-9][0-9]_*.diff")):
            self.assertNotIn("template_gate", open(pth, encoding="utf-8").read(), pth)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 2)
            out = os.path.join(tmp, "b")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "big", "--pdbdir", in_dir, "--out", out], env)
            missing = [ln for ln in err.splitlines() if "producer_missing" in ln]
            if missing:                                          # a core without the row-chunk producer: refused by name, nothing ran (a core between cuts)
                self.assertEqual(rc, 0, err); self.assertIn("skipped=trimul_chunk", err); self.assertIn("reason=cannot_run:producer_missing:opt_core.mem.rowpair_jax.rowchunk", err)   # 0.6.0: the lever steps aside by name, the mode runs
                return
            self.assertEqual(rc, 0, err)
            start = _stubs.records(so)[0]; argv = start["argv"]                                   # the driver's own proc_start record (relayed to stdout): its argv and environment
            self.assertEqual(argv[-15:], ["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-program_cache", "<programs>", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(start["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.95")
            self.assertRegex(err, r"EXIT pid=\d+ mode=big .* partial=none allow_partial=off\n"); self.assertEqual(_stubs.run_files(out), ["check.point", "out.sc", "pdbs"])   # <out> holds the driver's outputs only
            self.assertIn("ACTIVE mode=big route=cli line=-fast -precompile 6 -subbatch 128 -flash_attn -fused_triattn -fused_trimul -opm_reassoc -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L9,L8,L10,L11,L12,L19,mem_fraction,L13,L15,L16,ccache precompile=6", err)
            self.assertRegex(err, r"ACTIVE mode=big .* applied=argv\n"); self.assertNotIn("n_gpu=", err); self.assertNotIn("sharding=", err)   # one GPU: no resource-axis tokens
            self.assertRegex(err, r"LEVER name=trimul_chunk state=off reason=\S*not_in_the_lever_set_of_big")   # 0.7.1: the row-chunked TriangleMultiplication is an opt-in no shipped mode composes
            self.assertIn("LEVER name=mem_fraction state=on impl=xla_client_mem_fraction origin=kit strategy=F7.jax_memory_flags flag=XLA_PYTHON_CLIENT_MEM_FRACTION arm=big evidence=proc_start.env:_XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", err)
            self.assertNotIn("LEVER name=rowpair", err)
            self.assertNotIn(" base=", err); self.assertEqual(len(os.listdir(os.path.join(out, "pdbs"))), 2)
            rc, _, err = _stubs.run_cli(["pred", "--mode", "big", "--precompile", "--pdbdir", in_dir, "--out", os.path.join(tmp, "xp")], env)
            self.assertEqual(rc, 0, err); self.assertIn("ACTIVE mode=big route=cli line=-fast -precompile 6 ", err); self.assertTrue(os.path.exists(os.path.join(tmp, "xp", "pdbs")))   # 0.7.2: one composed line — big accepts --precompile [N] as fast does

    def test_above_the_chunk_floor_l11_is_superseded_off_not_on(self):
        """A directory whose every compiled length reaches the -trimul_chunk floor (1473): the row-chunked TriangleMultiplication body takes every trace and no
        call reaches fast's fused block (L11) — by the mode's definition. The run is complete (exit 0, partial=none); L11 prints `state=off
        reason=superseded_by:trimul_chunk_…`, never `state=on` over zero calls and never `skipped`; trimul_chunk is `on` with its engaged traces."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir = os.path.join(tmp, "big"); _stubs.long_input(in_dir, "cplx1500", 1500)
            out = os.path.join(tmp, "b")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "big", "--pdbdir", in_dir, "--out", out], dict(env, STUB_FTRIMUL_SERVED="0", STUB_FTRIMUL_FALLBACK="0", AF2IG_OPT_TRIMUL_CHUNK="on"))   # 0.7.1: the lever is an opt-in
            if any("producer_missing" in ln for ln in err.splitlines()):        # a core without the row-chunk producer refuses big by name (TestBigWithoutProducers): nothing to supersede here
                self.assertEqual(rc, 0, err); self.skipTest("big refused by name on this core: producer_missing:opt_core.mem.rowpair_jax (the superseded line needs the row-chunk producer)")
            self.assertEqual(rc, 0, err)
            self.assertRegex(err, r"EXIT pid=\d+ mode=big .* items=1/1 .* partial=none allow_partial=off\n"); self.assertNotIn("NOT ACTIVE", err); self.assertNotIn("PARTIAL allowed", err)
            l11 = [ln for ln in err.splitlines() if ln.startswith("[af2ig-opt] LEVER name=L11 ")]
            self.assertEqual(len(l11), 1, err)
            self.assertIn("LEVER name=L11 state=off reason=superseded_by:trimul_chunk_—_0_TriangleMultiplication_calls_reached_the_fused_block,_2_trace(s)_ran_the_row-chunked_body_(rows=256_min_residues=1473);", l11[0])
            self.assertIn(" impl=opt_core.pallas.serve.triangle_multiplication origin=core strategy=F2.fpf_trimul_fast flag=-fused_trimul arm=big", l11[0]); self.assertNotIn("evidence=", l11[0])
            self.assertRegex(err, r"LEVER name=trimul_chunk state=on .* flag=-trimul_chunk arm=big evidence=\S*engaged_traces=2_disengaged_traces=0")
            self.assertEqual(sorted(l.split(" name=")[1].split()[0] for l in err.splitlines() if " state=on " in l and "arm=big" in l), sorted(["L6", "L1", "U1", "L7", "L9", "L8", "L10", "L12", "L19", "trimul_chunk", "mem_fraction", "L13", "L15", "L16", "ccache"]))


class TestBigWithoutProducers(unittest.TestCase):
    def test_refused_without_producers(self):
        """On a core that passes the pin gate (same version) but lacks the memory line's producer (opt_core.mem.rowpair_jax — a shadow copy with it
        stubbed out, first on PYTHONPATH), the lever steps aside BY NAME (0.6.0 doctrine: `skipped=trimul_chunk`, LEVER state=skipped reason=cannot_run:producer_missing:…) and big runs
        its remaining levers: never a traceback, never a refusal; exact / fast are untouched on that core. Runs at every pin."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 1)
            env = _stubs.prepend_pythonpath(env, _stubs.shadow_core(tmp, "older_mem", drop=("mem/rowpair_jax",)))
            rc, _, err = _stubs.run_cli(["pred", "--mode", "big", "--pdbdir", in_dir, "--out", os.path.join(tmp, "b")], dict(env, AF2IG_OPT_TRIMUL_CHUNK="on"))   # 0.7.1: opted in — the step-aside is the lever's
            self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE:", err); self.assertNotIn("Traceback", err)          # 0.6.0 doctrine: the memory line's pair-stack lever steps aside BY NAME; big runs its remaining levers
            self.assertIn(" skipped=trimul_chunk precompile=", err); self.assertNotIn("-trimul_chunk", err.split(" levers=")[0].split(" line=")[-1])
            self.assertRegex(err, r"LEVER name=trimul_chunk state=skipped reason=cannot_run:producer_missing:opt_core\.mem\.rowpair_jax\.\S* impl=rowpair_jax\.rowchunk origin=core strategy=F7\.chunked_eval flag=-trimul_chunk arm=big\n")   # every missing producer named
            self.assertIn("stepped aside by name", err); self.assertTrue(os.path.exists(os.path.join(tmp, "b", "pdbs")))
            rc, _, err = _stubs.run_cli(["check", "--mode", "big", "--n_gpu", "2"], env)
            self.assertEqual(rc, cli.EXIT_USAGE, err)                                                      # no resource axis: --n_gpu is not a switch of this kit
            rc, _, err = _stubs.run_cli(["check", "--mode", "fast"], env); self.assertEqual(rc, 0, err)      # exact / fast untouched on that core


def _xla_oom():
    """An exception shaped like jaxlib's device OOM: class ``XlaRuntimeError`` (module ``jaxlib.xla_extension``) whose text starts with the
    ``RESOURCE_EXHAUSTED: Out of memory`` status line XLA writes for a failed device allocation."""
    cls = type("XlaRuntimeError", (RuntimeError,), {"__module__": "jaxlib.xla_extension"})
    return cls("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 68719476736 bytes.")


class TestOomPropagation(unittest.TestCase):
    """``pairstack.install`` is the driver's served entry for the memory levers (patch 06 calls it before the model runner is built): every other
    exception of the installation is the named refusal ``[af2ig-opt] pairstack refused: <reason>`` + exit 3; an OOM is re-raised first — it is never
    rerouted into a refusal, a skipped input or a failed-design record. The classifier is the model-opt tree's one: ``opt_core.oom.is_oom`` (JAX
    ``RESOURCE_EXHAUSTED``, CUDA OOM, ``MemoryError``). No GPU: the allocation site is mocked with an exception class named like jaxlib's."""

    def test_oom_propagates_out_of_the_served_entry(self):
        from af2ig_opt import pairstack
        err = _xla_oom()
        with mock.patch("af2ig_opt._core_gate.gate", lambda *a, **k: None), mock.patch.object(pairstack, "_install", side_effect=err):
            with self.assertRaises(type(err)) as cm:
                pairstack.install((256, 1473))
        self.assertIs(cm.exception, err)                                    # the OOM itself, not a wrapper and not SystemExit(3)

    def test_memoryerror_propagates(self):
        from af2ig_opt import pairstack
        with mock.patch("af2ig_opt._core_gate.gate", lambda *a, **k: None), mock.patch.object(pairstack, "_install", side_effect=MemoryError("host")):
            with self.assertRaises(MemoryError):
                pairstack.install((256, 1473))

    def test_other_errors_keep_the_named_refusal(self):
        from af2ig_opt import pairstack
        with mock.patch("af2ig_opt._core_gate.gate", lambda *a, **k: None), mock.patch.object(pairstack, "_install", side_effect=RuntimeError("modules.py not pinned")):
            with self.assertRaises(SystemExit) as cm:
                pairstack.install((256, 1473))
        self.assertEqual(cm.exception.code, 3)


class TestCensusDeltaUnderThreads(unittest.TestCase):
    def test_parallel_traces_each_get_their_own_delta(self):
        """0.6.0: the census snapshot is read inside TRACE_LOCK around the trace — three programs prepared on three threads each store the calls of
        THEIR trace (8), not the running total another thread's trace had reached while this one compiled (the replayed L12 count was 3x before)."""
        import threading, time as _time
        from af2ig_opt import programs
        count = {"served": 0}
        class _Lowered:
            def compile(self): _time.sleep(0.05); return "exe"
        class _Traced:
            def lower(self): return _Lowered()
        class _Fn:
            def trace(self, *a): count["served"] += 8; _time.sleep(0.01); return _Traced()
        snap = lambda: {"opm_reassoc": {"served": count["served"], "fallback": 0, "errors": 0, "fallback_by": {}, "errors_by": {}}}
        out = []
        def work(): c, rec = programs.aot_compile(_Fn(), (1,), snap); out.append(programs.census_delta(rec["census_before"], rec["census_after"])["opm_reassoc"]["served"])
        th = [threading.Thread(target=work) for _ in range(3)]; [x.start() for x in th]; [x.join() for x in th]
        self.assertEqual(sorted(out), [8, 8, 8])
