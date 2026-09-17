"""`big`'s own levers reach the process that runs the model. The mode's size levers live in the house kit `opt/forward/size_levers/src/sz_levers.py`
(on PYTHONPATH as `<kit>/src` like every kit) and no vendor kit file imports them, so both forms must import them themselves: the in-process
form in `activate()`'s import loop, the process form (`boltzgen-opt design --mode big`: the add-on's runner as a child) by handing the runner
child the mode (`BOLTZGEN_OPT=big`, stack.mode_env), so that the child activates the package at interpreter start through the hook-first route
(site → the partner's `sitecustomize` → `bg_hook` imports `boltzgen` → the installed `.pth`'s finder → `activate()` imports `sz_levers` last)
and `install()` prints its `[levers] …` lines — the proof `stack.levers_from_lines` reads. The hand-over is one-shot (`BOLTZGEN_OPT_HANDOVER`,
consumed by the child's activation): afterwards the child's environment names no mode, so the runner's stock CPU-step subprocesses
(`xa_run.py` l.62-65: its environment minus PYTHONPATH, each under its own step name) stay stock: unactivated and outside the accelerator census. A planned lever no line proves is a named fallback
(never dropped from the account), so a child that did not install them makes the run partial: exit 3 unless allowed. `exact` hands its child
nothing of the package: its child environment is held to a literal here. Real kit modules and the package on this tree, stub
upstream (`_stubs`), CPU torch; the process-form cases need the package installed (`pip install -e boltzgen/opt`: the child's route is the
installed `.pth`)."""
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

from boltzgen_opt import cli, codes, design, modes, registry, report, stack

from . import _stubs

HOME = stack.opt_home()
TD_LINE = "[levers] TokenDistanceModule row-chunked, rows/block=64"       # sz_levers.install() under the mode's SZ_TD_CHUNK=64
RUNNER_BODY = r"""
import json, os, sys
import xa_fastinit                                        # what the add-on's runner imports for a GPU step under XA_FAST_INIT=1 (xa_run.py l.37; XA_HOIST=0 under big: no xa_hoist)
rec = {"env_mode": os.environ.get("BOLTZGEN_OPT"), "env_handover": os.environ.get("BOLTZGEN_OPT_HANDOVER"),
       "pythonpath": os.environ.get("PYTHONPATH", "").split(os.pathsep),
       "kit_modules": sorted(n for n in sys.modules if n.startswith(("xa_", "bg_", "sz_", "hl_"))), "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None)}
if os.environ.get("EMULATE_CPU_STEP"):                       # the runner's stock CPU-step subprocess, composed exactly as xa_run.py l.58-65 does it after the GPU steps
    import subprocess
    os.environ["BOLTZGEN_PIPELINE_STEP"] = "design_folding"                   # the last GPU step's name, left on the environment by the in-process pipeline (bg_inproc.py l.33)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}         # xa_run.py l.62
    env["BOLTZGEN_PIPELINE_STEP"] = "analysis"                                # xa_run.py l.64: the step's own name in the child's environment
    cpu = ("import json, os, sys\nimport boltzgen, boltzgen.resources.main\nimport boltzgen_opt\nfrom boltzgen_opt import census\nst = boltzgen_opt.status()\n"
           "print('CPU ' + json.dumps({'env_mode': os.environ.get('BOLTZGEN_OPT'), 'step': os.environ.get('BOLTZGEN_PIPELINE_STEP'), 'active': bool(st.get('active')), "
           "'kit_modules': sorted(n for n in sys.modules if n.startswith(('xa_', 'bg_', 'sz_', 'hl_'))), 'finder': any(type(f).__name__ == 'Finder' for f in sys.meta_path), "
           "'census_armed': census.armed() is not None}))\n")
    r = subprocess.run([sys.executable, "-c", cpu], env=env, capture_output=True, text=True, timeout=300)
    rec["cpu_step"] = {"rc": r.returncode, "out": r.stdout[-4000:], "err": r.stderr[-4000:]}
sz = sys.modules.get("sz_levers")
rec["sz_file"] = getattr(sz, "__file__", None)
rec["sz_stats"] = {k: sz.STATS[k] for k in ("td_enabled", "oom_trace_enabled")} if sz else None
import boltzgen_opt
st = boltzgen_opt.status()
rec["status"] = {k: st.get(k) for k in ("active", "mode", "form", "levers_applied", "levers_fallback", "partial", "hook_first")}
print("RECORD " + json.dumps(rec))
"""


def _record(out: str) -> dict:
    return next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))


def _active_line_levers(err: str) -> set:
    """The `levers=` field of the one `[boltzgen-opt] ACTIVE mode=big …` line on stderr (report.py's activation line)."""
    lines = [ln for ln in err.splitlines() if ln.startswith("[boltzgen-opt] ACTIVE mode=big form=inproc")]
    assert len(lines) == 1, err[-2000:]
    m = re.search(r" levers=([^ ]*)", lines[0])
    assert m, lines[0]
    return set(m.group(1).split(","))


class TestBigLeversReachTheModelProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_big_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------------------- the tree
    def test_the_size_levers_kit_is_laid_out_like_every_kit(self):
        """`<kit>/src/<module>.py`: the PYTHONPATH entry the mode table gives big (stack.kit_src) holds sz_levers.py, and the registry rows say so."""
        src = stack.kit_src(modes.KIT_SIZE_LEVERS)
        self.assertTrue(os.path.isfile(os.path.join(src, "sz_levers.py")), src)
        self.assertEqual(src, os.path.join(HOME, "forward", "size_levers", "src"))
        for name in ("td_chunk",):
            self.assertEqual((registry.LEVERS[name].kit, registry.LEVERS[name].file), (modes.KIT_SIZE_LEVERS, "src/sz_levers.py"))
        res = modes.resolve("big", HOME)
        self.assertEqual(res.imports[-1], "hl_levers")                             # the universal async_writer, imported after every mode's own
        self.assertEqual(res.imports[-2], "sz_levers")                             # the mode's own module, imported last before it
        self.assertIn(src, stack.mode_env(res, {"PATH": os.environ.get("PATH", "")})[0]["PYTHONPATH"].split(os.pathsep))

    # ------------------------------------------------------------------------------------------------------- in-process form
    def test_enable_big_installs_the_size_levers_in_process(self):
        res = modes.resolve("big", HOME)
        code = ("import json, sys, boltzgen_opt\nrep = boltzgen_opt.enable('big')\nsz = sys.modules.get('sz_levers')\n"
                "print('RECORD ' + json.dumps({'active': rep['active'], 'reason': rep.get('reason'), 'applied': rep.get('levers_applied'), 'partial': rep.get('partial'), "
                "'modules': sorted(n for n in sys.modules if n.startswith(('xa_', 'bg_', 'sz_', 'hl_'))), "
                "'sz_file': getattr(sz, '__file__', None), 'stats': {k: sz.STATS[k] for k in ('td_enabled', 'oom_trace_enabled')} if sz else None}))\n")
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site))
        self.assertEqual(rc, 0, err)
        rec = _record(out)
        self.assertTrue(rec["active"], rec)
        self.assertTrue(set(res.imports) <= set(rec["modules"]), (res.imports, rec["modules"]))   # every module the mode table lists, imported through activate()'s own loop
        self.assertTrue({"td_chunk"} <= _active_line_levers(err), err[-1500:])       # and the activation line says so
        self.assertTrue({"td_chunk"} <= set(rec["applied"]), rec)
        self.assertFalse(rec["partial"], rec)
        self.assertEqual(rec["stats"], {"td_enabled": True, "oom_trace_enabled": True})   # the trace is armed with the size levers
        self.assertTrue(rec["sz_file"].startswith(stack.kit_src(modes.KIT_SIZE_LEVERS) + os.sep), rec["sz_file"])
        self.assertIn(TD_LINE, out)
        self.assertIn("[boltzgen-opt] ACTIVE mode=big form=inproc", err)

    # --------------------------------------------------------------------------------------------------------- process form
    def _runner_child(self, res, drop_mode=False, cpu_step=False):
        """The runner child of `design --mode <res.mode>` up to its lever imports: the environment exactly as cmd_design builds it
        (design.kit_env → stack.mode_env; the caller's PYTHONPATH replaced by the kits' src directories), the interpreter the runner
        command names (design.runner_cmd: this one, its installed .pth processed at start), the runner's own first import."""
        if not _stubs.package_installed():
            raise unittest.SkipTest("boltzgen_opt is not installed in this interpreter (the runner child activates through the installed "
                                    "boltzgen_opt_autoload.pth): run in a venv with `pip install -e boltzgen/opt` to exercise this")
        run_dir = tempfile.mkdtemp(prefix="run_", dir=self.tmp)
        env, dropped = design.kit_env(res, run_dir, base=_stubs.clean_env(self.site))
        self.assertIn("PYTHONPATH", dropped)                                        # the caller's PYTHONPATH (stubs, package, core) is replaced by the kits' entries ...
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [stack.kit_src(k) for k in res.kits] + stack.package_roots())   # ... then the package and the core this process runs (stack.package_roots): the child imports the same copies
        cmd = design.runner_cmd(res, run_dir, 0)
        self.assertEqual(cmd[1:], [os.path.join(HOME, res.runner), run_dir, "0"])  # ... and the child is `<this python> <add-on>/src/xa_run.py <run_dir> <seed>`
        if drop_mode:
            env.pop(modes.ENV, None)
        if cpu_step:
            env["EMULATE_CPU_STEP"] = "1"
        pth = _stubs.site_pth(self.site)                                            # the stub upstream importable without PYTHONPATH, as the installed wheel is on a box
        try:
            rc, out, err = _stubs.run_py(RUNNER_BODY, env, cwd=run_dir)
        finally:
            _stubs.remove_site_pth(pth)
        return rc, out, err, env

    def test_design_big_hands_the_runner_child_the_mode_and_the_child_installs_the_size_levers(self):
        res = modes.resolve("big", HOME)
        rc, out, err, env = self._runner_child(res, cpu_step=True)
        self.assertEqual((env.get(modes.ENV), env.get(stack.ENV_HANDOVER)), ("big", "1"))   # the mode handed over, one-shot, for a mode with levers of its own
        self.assertEqual(rc, 0, err)
        rec = _record(out)
        self.assertEqual((rec["env_mode"], rec["env_handover"]), (None, None), rec)   # consumed: after its activation the child's environment names no mode
        self.assertTrue(rec["sitecustomize"] and rec["sitecustomize"].startswith(stack.kit_src(modes.KIT_PARTNER)), rec)   # the child started on the partner's sitecustomize line
        self.assertIn("sz_levers", rec["kit_modules"], rec)
        self.assertTrue(set(res.imports) <= set(rec["kit_modules"]), (res.imports, rec["kit_modules"]))   # every module the mode table lists, imported in the child through the real route
        self.assertTrue({"td_chunk", "fastinit"} <= _active_line_levers(err), err[-1500:])   # the child's activation line lists them
        self.assertTrue(rec["sz_file"].startswith(stack.kit_src(modes.KIT_SIZE_LEVERS) + os.sep), rec)
        self.assertEqual(rec["sz_stats"], {"td_enabled": True, "oom_trace_enabled": True})
        self.assertEqual((rec["status"]["active"], rec["status"]["mode"], rec["status"]["hook_first"], rec["status"]["partial"]), (True, "big", True, False), rec)
        self.assertTrue({"td_chunk", "fastinit"} <= set(rec["status"]["levers_applied"]), rec)
        self.assertIn(TD_LINE, out)
        self.assertIn("[boltzgen-opt] ACTIVE mode=big form=inproc", err)         # the child's own activation line, hook-first
        ev = stack.levers_from_lines((out + err).splitlines(), res)                  # what design.evidence reads from the child's log
        self.assertTrue({"td_chunk"} <= set(ev["levers_applied"]), ev)
        self.assertFalse({"td_chunk"} & set(ev["levers_fallback"]), ev)
        # (a) the runner's stock CPU-step subprocess after that activation: inert — no mode inherited, nothing armed, nothing applied, no line
        cpu = rec["cpu_step"]
        self.assertEqual(cpu["rc"], 0, cpu["err"])
        crec = next(json.loads(ln[4:]) for ln in cpu["out"].splitlines() if ln.startswith("CPU "))
        self.assertEqual(crec, {"env_mode": None, "step": "analysis", "active": False, "kit_modules": [], "finder": False, "census_armed": False}, crec)   # its own step name; the census payload it inherits arms nothing in a CPU step
        self.assertNotIn("[boltzgen-opt", cpu["err"] + cpu["out"])                  # no line of the package at all from a stock step: ACTIVE / NOT ACTIVE / EXIT, nor a `[boltzgen-opt <mode>] KERNELS` / PEAK census line
        self.assertNotIn("[levers]", cpu["out"] + cpu["err"])
        self.assertEqual(err.count("[boltzgen-opt] ACTIVE "), 1, err[-2000:])       # the runner child's one activation line; its CPU step added none

    def test_design_fast_hands_the_runner_child_the_mode_and_the_child_resolves_fasts_row(self):
        """`design --mode fast` exports BOLTZGEN_OPT=fast + the one-shot marker beside the row's own switches (FL_COND_DEDUP=1, FL_ATTN_BF16=1,
        FL_ATTN_BACKEND=cudnn, FL_DIT_FUSED=1) to its runner child (design.kit_env → stack.mode_env); the child, activated hook-first through
        the installed .pth, resolves the SAME row — one ACTIVE mode=fast line, cond_dedup and attn_bf16 installed — and consumes the names."""
        res = modes.resolve("fast", HOME)
        rc, out, err, env = self._runner_child(res)
        self.assertEqual((env.get(modes.ENV), env.get(stack.ENV_HANDOVER), env.get("FL_COND_DEDUP"), env.get("FL_ATTN_BF16"), env.get("FL_ATTN_BACKEND"), env.get("FL_DIT_FUSED")),
                         ("fast", "1", "1", "1", "cudnn", "1"), env)
        self.assertNotIn("BOLTZGEN_OPT_VARIANT", env)                                # no variant name: the mode is one row
        self.assertEqual(rc, 0, err[-3000:])
        rec = _record(out)
        self.assertEqual((rec["env_mode"], rec["env_handover"]), (None, None), rec)   # consumed by the child's activation (stack.activate)
        active = [ln for ln in err.splitlines() if ln.startswith("[boltzgen-opt] ACTIVE mode=fast form=inproc")]
        self.assertEqual(len(active), 1, err[-2000:])
        self.assertNotIn(" variant=", active[0])
        self.assertIn("[fl_levers] cond_dedup installed sites=", out)
        self.assertIn("[fl_levers] attn_bf16 installed", out)
        self.assertEqual((rec["status"]["active"], rec["status"]["mode"], rec["status"]["hook_first"]), (True, "fast", True), rec)
        self.assertTrue({"cond_dedup", "attn_bf16", "attn_cudnn", "fastinit"} <= set(rec["status"]["levers_applied"]), rec)
        ev = stack.levers_from_lines((out + err).splitlines(), res)                  # the caller's census of that child log
        self.assertTrue({"cond_dedup", "attn_bf16", "attn_cudnn"} <= set(ev["levers_applied"]), ev)
        with mock.patch.dict(os.environ, env, clear=True):                           # the hook-first rule on exactly that environment: the row's own line, no mismatch
            self.assertIsNone(stack.hook_line_mismatch(res))

    def test_a_runner_child_that_never_activates_the_package_makes_the_run_partial_and_exit_3(self):
        """The fail-closed side: a child that does not install the size levers (here: the mode withheld from it — what a child whose
        interpreter lacks the installed .pth does) prints no `[levers]` line, the caller's census names the lever as fallen back, and the
        verb exits 3 with the partial refusal line — a mode is all of its levers; never ACTIVE-and-silent."""
        res = modes.resolve("big", HOME)
        rc, out, err, env = self._runner_child(res, drop_mode=True)
        self.assertEqual(rc, 0, err)
        rec = _record(out)
        self.assertEqual((rec["env_mode"], rec["sz_file"], rec["status"]["active"]), (None, None, False), rec)
        self.assertNotIn("[levers]", out + err)
        ev = stack.levers_from_lines((out + err).splitlines(), res)
        self.assertTrue({"td_chunk"} <= set(ev["levers_fallback"]), ev)
        self.assertEqual({ev["fallback_reasons"][k] for k in ("td_chunk",)}, {"no kit line proves it"})
        rep = {"levers_applied": ev["levers_applied"], "levers_fallback": ev["levers_fallback"], "fallback_reasons": ev["fallback_reasons"], "partial": bool(ev["levers_fallback"])}
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            code = cli.exit_for(0, rep)
        self.assertEqual(code, codes.EXIT_NOT_ACTIVE)
        self.assertIn("NOT ACTIVE: partial activation", buf.getvalue()); self.assertIn("td_chunk", buf.getvalue())

    def test_the_hook_first_rule_recognises_bigs_own_line_in_a_childs_environment(self):
        """A child started with the mode's exports (stack.mode_env) carries the mode's own line: for `big` that includes names outside the
        kits' switch prefixes (SZ_TD_CHUNK, the allocator setting), read with their values; a changed value is named.
        `exact`'s line is kit switches only, judged exactly as before."""
        from unittest import mock
        for mode in ("exact", "big", "fast"):
            res = modes.resolve(mode, HOME)
            env, _ = stack.mode_env(res, {"PATH": "/usr/bin"})
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertIsNone(stack.hook_line_mismatch(res), mode)
                if mode == "big":
                    os.environ["SZ_TD_CHUNK"] = "32"
                    self.assertIn("SZ_TD_CHUNK='32'", stack.hook_line_mismatch(res))
                    del os.environ["SZ_TD_CHUNK"]
                    self.assertIn("SZ_TD_CHUNK=None", stack.hook_line_mismatch(res))
        exact = modes.resolve("exact", HOME)
        env, _ = stack.mode_env(exact, {"PATH": "/usr/bin"})
        with mock.patch.dict(os.environ, dict(env, PYTORCH_CUDA_ALLOC_CONF="x", CUDA_VISIBLE_DEVICES="0"), clear=True):
            self.assertIsNone(stack.hook_line_mismatch(exact))                        # names outside the kits' switch prefixes that exact does not export stay out of its judgement
        with mock.patch.dict(os.environ, dict(env, SZ_TD_CHUNK="64"), clear=True):
            self.assertIn("kit switches SZ_TD_CHUNK='64' where the mode's line is BG_GRAPH=graph,", stack.hook_line_mismatch(exact))   # a size-lever switch in an exact child is a line exact never wrote: named (SZ_ is a kit switch prefix)
        with mock.patch.dict(os.environ, dict(env, BG_GRAPH="predraw"), clear=True):
            self.assertIn("BG_GRAPH='predraw'", stack.hook_line_mismatch(exact))
        with mock.patch.dict(os.environ, dict(env, BG_TIMING_FILE="/t.jsonl"), clear=True):
            self.assertIsNone(stack.hook_line_mismatch(exact))                        # observability switches are not levers (modes.OBSERVABILITY_SWITCHES)

    # ------------------------------------------------------------------------------------------------------------ the census
    def test_every_planned_lever_of_every_mode_has_a_proof_route_in_the_process_form(self):
        """Each lever a mode plans is either proven by a kit line (stack.RUNNER_LINES) or by the caller from files (stack.CALLER_PROVEN_LEVERS);
        nothing planned can leave the account silently."""
        for mode in modes.MODES:
            res = modes.resolve(mode, HOME)
            for name in res.levers:
                self.assertTrue(name in stack.RUNNER_LINES or name in stack.CALLER_PROVEN_LEVERS, f"{mode}: {name} has no proof route")
        self.assertEqual(set(stack.CALLER_PROVEN_LEVERS), {"inproc"})
        big = modes.resolve("big", HOME)
        ev = stack.levers_from_lines([], big)                                     # a log proving nothing: every line-proven planned lever is named, none dropped
        self.assertEqual(set(ev["levers_fallback"]), set(big.levers) - set(stack.CALLER_PROVEN_LEVERS))
        ev = stack.levers_from_lines([TD_LINE, "[xa_fastinit] coverage OK 12/12"], big)
        self.assertEqual((set(ev["levers_applied"]), ev["levers_fallback"]), ({"fastinit", "td_chunk"}, ["async_writer"]))


    def test_the_oom_trace_says_where_and_the_design_census_counts(self):
        """Every kit mode's model process carries the size levers' out-of-memory trace (sz_levers on every kit mode's path and own imports):
        one `[sz] {"event": "oom", …}` line WHERE a CUDA out-of-memory was raised inside Boltz.forward, re-raised unchanged, read by
        stack.levers_from_lines into `oom`. HOW MANY batches upstream's own handler then skipped is the design census's `oom_skipped`
        (design.designs_census), counted from upstream's own `| WARNING: ran out of memory, skipping batch` line in every mode, `off`
        included — one count, no second one from the trace (the trace prints no summary line)."""
        for mode in ("exact", "fast", "big"):
            res = modes.resolve(mode, stack.opt_home())
            self.assertIn("sz_levers", res.imports, mode); self.assertIn(modes.KIT_SIZE_LEVERS, res.kits, mode)
            self.assertEqual(res.env.get("SZ_TD_CHUNK"), "64" if mode == "big" else None, mode)   # the size lever itself is big's; the trace needs no switch
        big = modes.resolve("big", stack.opt_home())
        ev = stack.levers_from_lines(['[sz] {"event": "oom", "where": "x", "msg": "CUDA out of memory", "max_alloc_GB": 70.1}'], big)
        self.assertTrue(ev["oom"]); self.assertNotIn("oom_skipped", ev)
        self.assertFalse(stack.levers_from_lines(["[sz] oom skipped=3", "ordinary line"], big)["oom"])
        src = open(os.path.join(stack.kit_dir(modes.KIT_SIZE_LEVERS), "src", "sz_levers.py"), encoding="utf-8").read()
        self.assertNotIn("oom skipped=", src); self.assertFalse(hasattr(stack, "RUNNER_OOM_SUMMARY"))
        lines = ["Predicting: 3/4\r| WARNING: ran out of memory, skipping batch", "| WARNING: ran out of memory, skipping batch", "WARNING: Skipping batch. Exception for x_0", "unrelated"]
        c = design.designs_census(os.path.join(stack.opt_home(), "no", "such", "run"), lines)
        self.assertEqual((c["oom_skipped"], c["featurizer_skipped"], c["step"], c["requested"], c["produced"]), (2, 1, None, None, None))

    def test_a_big_runner_log_with_the_add_ons_lines_and_no_levers_line_is_a_partial_run_exit_3(self):
        """The shape of a runner child that ran the add-on's levers and never installed the mode's own (its log: `[xa_fastinit] coverage OK`,
        the partner's lines, no `[levers] …`): the census names td_chunk as fallen back, the report is partial, the verb exits 3
        — an `ACTIVE mode=big … levers=fastinit,inproc` line with nothing named cannot be produced from such a log."""
        res = modes.resolve("big", HOME)
        log = ["[xa_fastinit] coverage OK 438/438", "[bg_hook] seed fix active", "[xa_run] xa_fastinit stats: {'enabled': True}", "inproc: design 12 designs"]
        ev = stack.levers_from_lines(log, res)
        self.assertEqual((ev["levers_applied"], sorted(ev["levers_fallback"])), (["fastinit"], ["async_writer", "td_chunk"]))
        rep = {"levers_applied": ev["levers_applied"] + list(stack.CALLER_PROVEN_LEVERS), "levers_fallback": ev["levers_fallback"],
               "fallback_reasons": ev["fallback_reasons"], "partial": bool(ev["levers_fallback"])}          # design.evidence: the caller's two proven from their files, partial = any fallback
        self.assertEqual(sorted(stack.partial_levers(rep)), ["async_writer", "td_chunk"])
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(cli.exit_for(0, rep), codes.EXIT_NOT_ACTIVE)             # a mode is all of its levers: refused by name
        self.assertRegex(buf.getvalue(), r"(?m)^\[boltzgen-opt\] NOT ACTIVE: .*td_chunk")   # the line starts at column 0 (print_fresh writes a line feed first)

    def test_each_modes_child_environment_is_this_literal_and_every_kit_mode_hands_the_mode_over(self):
        base = {"PATH": "/usr/bin:/bin", "HOME": "/home/u", "LANG": "C.UTF-8", "PYTHONPATH": "/foreign/path", "BOLTZGEN_OPT": "exact",
                "BG_GRAPH": "predraw", "XA_FAST_INIT": "1", "XA_VERBOSE_KEEP": "x", "SZ_TD_CHUNK": "8", "HL_WRITER_MODE": "thread", "BG_TIMING_FILE": "/t.jsonl", "CUDA_VISIBLE_DEVICES": "0"}
        kept = {"BG_TIMING_FILE": "/t.jsonl", "CUDA_VISIBLE_DEVICES": "0", "HOME": "/home/u", "LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"}
        xa, fi, sz, fl, wl = (os.path.join(HOME, "forward", d, "src") if d != "writer_levers" else os.path.join(HOME, "host", d, "src")
                              for d in ("xattempt_addon", "fast_inference", "size_levers", "fast_levers", "writer_levers"))
        roots = stack.package_roots()                                               # the package's and the core's own directories follow the kits' entries in every activating mode (stack.package_roots)
        want = {                                                                    # (env, dropped) per mode: off keeps the caller's PYTHONPATH and loses every kit switch; every activating mode hands its own imports' mode over, one-shot, and carries the universal async_writer
            "off": (dict(kept, PYTHONPATH="/foreign/path"), ["BG_GRAPH", "BOLTZGEN_OPT", "HL_WRITER_MODE", "SZ_TD_CHUNK", "XA_FAST_INIT", "XA_VERBOSE_KEEP"]),
            "exact": (dict(kept, BG_GRAPH="graph", XA_FAST_INIT="1", XA_HOIST="1", HL_ASYNC_WRITER="1", PYTHONPATH=os.pathsep.join((xa, fi, sz, wl, *roots)),
                           BOLTZGEN_OPT="exact", BOLTZGEN_OPT_HANDOVER="1"),
                      ["BG_GRAPH", "HL_WRITER_MODE", "PYTHONPATH", "SZ_TD_CHUNK", "XA_VERBOSE_KEEP"]),          # a caller value equal to the mode's own export (XA_FAST_INIT=1) is not reported
            "big": (dict(kept, BG_GRAPH="off", XA_FAST_INIT="1", XA_HOIST="0", SZ_TD_CHUNK="64", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                           HL_ASYNC_WRITER="1", PYTHONPATH=os.pathsep.join((xa, fi, sz, wl, *roots)), BOLTZGEN_OPT="big", BOLTZGEN_OPT_HANDOVER="1"),
                      ["BG_GRAPH", "BOLTZGEN_OPT", "HL_WRITER_MODE", "PYTHONPATH", "SZ_TD_CHUNK", "XA_VERBOSE_KEEP"]),   # the caller's SZ_TD_CHUNK=8 is dropped for the mode's own 64
            "fast": (dict(kept, BG_GRAPH="graph", XA_FAST_INIT="1", XA_HOIST="1", FL_COND_DEDUP="1", FL_ATTN_BF16="1", FL_DIT_FUSED="1", FL_ATTN_BACKEND="cudnn",
                          HL_ASYNC_WRITER="1", PYTHONPATH=os.pathsep.join((xa, fi, fl, sz, wl, *roots)), BOLTZGEN_OPT="fast", BOLTZGEN_OPT_HANDOVER="1"),
                     ["BG_GRAPH", "BOLTZGEN_OPT", "HL_WRITER_MODE", "PYTHONPATH", "SZ_TD_CHUNK", "XA_VERBOSE_KEEP"]),
        }
        self.assertEqual(set(want), set(modes.MODES))
        for mode, (env_want, dropped_want) in want.items():
            env, dropped = stack.mode_env(modes.resolve(mode, HOME), dict(base))
            self.assertEqual(env, env_want, mode); self.assertEqual(dropped, dropped_want, mode)
            hands = mode != "off" and bool(modes.KIT_MODES[mode].own_imports)
            self.assertEqual((modes.ENV in env, stack.ENV_HANDOVER in env), (hands, hands), mode)   # only a mode with levers of its own hands the mode over, one-shot
            self.assertFalse([k for k in env if k == "BOLTZGEN_OPT_VARIANT" or (mode != "big" and k.startswith("SZ_"))], (mode, env))
        self.assertEqual([m for m in modes.MODES if m != "off" and modes.KIT_MODES[m].own_imports], ["exact", "fast", "big"])   # async_writer gives every activating mode an own import now
        env, _ = stack.mode_env(modes.resolve("big", HOME), {"PATH": "/usr/bin"})   # a caller that never set the variable (`--mode big`): the child still gets the mode
        self.assertEqual((env[modes.ENV], env[stack.ENV_HANDOVER]), ("big", "1"))
        env, dropped = stack.mode_env(modes.resolve("exact", HOME), {"PATH": "/usr/bin", stack.ENV_HANDOVER: "1"})   # a mode with its own imports reissues the marker itself: the caller's own value never survives untouched either way
        self.assertEqual((stack.ENV_HANDOVER in env, dropped), (True, [stack.ENV_HANDOVER]))

    def test_the_hand_over_is_consumed_by_the_childs_activation(self):
        """In-process, the marker's contract alone: an activation that finds BOLTZGEN_OPT_HANDOVER removes it together with BOLTZGEN_OPT
        from the environment (the activated process keeps its levers; what it spawns inherits no mode); without the marker
        an activated process keeps exporting the mode to its children (the in-process form's rule)."""
        for marker in (True, False):
            extra = {stack.ENV_HANDOVER: "1"} if marker else {}
            code = ("import json, os, boltzgen_opt\nrep = boltzgen_opt.enable('big')\n"
                    "print('RECORD ' + json.dumps({'active': rep['active'], 'env': [os.environ.get(k) for k in ('BOLTZGEN_OPT', 'BOLTZGEN_OPT_HANDOVER')]}))\n")
            rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, **extra))
            self.assertEqual(rc, 0, err)
            self.assertEqual(_record(out), {"active": True, "env": [None, None] if marker else ["big", None]}, (marker, err[-800:]))


if __name__ == "__main__":
    unittest.main()
