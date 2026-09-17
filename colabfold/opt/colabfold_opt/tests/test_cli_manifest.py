"""The CLI on a stub `colabfold.batch` (records its argv and environment, writes colabfold's file names and completion markers, and —
when told — the manifest an activated model process would write into the launch's work directory): the launch forms (the console script
beside the interpreter; the import form; never `-m`), the stock route's cleaned environment and proof (the STOCK line, the launch record's
stock_proof), the fast route's exported switch with the caller's kit variables stripped, the gates before the launch, the verdict after it
(activation report, calls >= 1, completion markers), the composed argv (colabfold_batch's options verbatim, --data, its two positionals as
written), colabfold's job names, the launch record `cli.run` returns (nothing of the kit written under the results directory), the usage
refusals (unknown mode, disagreement, no parameters root, positionals missing) and the exit codes."""
import contextlib
import io
from contextlib import redirect_stderr
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from colabfold_opt import _autoload, cli, inputs, manifest, modes, settings, stack, stock_pred
from colabfold_opt.tests import _stubs

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opt_core                                                                       # noqa: E402 — the shared core's import root, handed to child interpreters explicitly
CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
import importlib.util                                                                 # noqa: E402
try:
    HAVE_AXIS = all(importlib.util.find_spec(n) is not None for n in ("opt_core.mem.ngpu", "opt_core.mem.rowpair_jax"))
except ImportError:
    HAVE_AXIS = False
TREE = os.path.dirname(OPT)
A3M = os.path.join(TREE, "tests", "inputs", "1BRS_AD.a3m")
WHEEL = os.path.join(TREE, "stock", "colabfold-1.6.1-py3-none-any.whl")                # the pinned upstream (stock/PINS.json): its colabfold/input.py names the jobs
STUB_BODY = textwrap.dedent('''\
    """stand-in for colabfold.batch (tests only): records argv + environment and lays the results directory out as colabfold_batch does —
    its file names, and its completion facts exactly (batch.py:1405-1412 a job whose <job>.result.zip or <job>.done.txt exists is skipped
    unless --overwrite-existing-results; :1450-1453 the input features pickle under --num-models 0 / --msa-only; :1643-1654 --zip moves the
    job's files into <job>.result.zip and writes no marker, else <job>.done.txt when a model ran; :2184-2200 --af3-json writes <job>.json and
    returns before run()); with STUB_MANIFEST (json: active, reason, calls | calls_per_job) it writes the manifest an activated model process
    would write, through the package's own writer, where that process writes it (the launch's work directory `pred` names), carrying the
    decision the run hook takes from run()'s arguments before the stock run (manifest.no_model_run, over the same facts)."""
    import json, os, sys, zipfile


    def run(queries=None, result_dir=None, **kw):                                   # never called by main(): the hook is not exercised here
        raise RuntimeError("stub run called")


    def _value(opts, flag, default):                                                # `--flag v` / `--flag=v`, the last occurrence
        val = default
        for i, a in enumerate(opts):
            if a == flag and i + 1 < len(opts):
                val = opts[i + 1]
            elif a.startswith(flag + "="):
                val = a.split("=", 1)[1]
        return val


    def main(argv=None):
        argv = sys.argv[1:] if argv is None else argv
        inp, res = argv[-2], argv[-1]
        opts = argv[:-2]
        os.makedirs(res, exist_ok=True)
        from colabfold.input import safe_filename                                         # colabfold's own job naming (batch.py:1399)
        files = sorted(os.listdir(inp)) if os.path.isdir(inp) else [os.path.basename(inp)]      # colabfold_batch's first positional: a directory or one file
        names = [safe_filename(os.path.splitext(f)[0]) for f in files if f.endswith((".a3m", ".fasta"))]
        num_models = 0 if "--msa-only" in opts else int(_value(opts, "--num-models", 5))     # batch.py:2160-2161: --msa-only is num_models = 0
        keep = "--overwrite-existing-results" not in opts                                 # :2222 keep_existing_results
        zip_results = "--zip" in opts                                                     # :2233
        rc = int(os.environ.get("STUB_RC", "0"))
        json.dump({"argv": argv, "env": dict(os.environ), "cwd": os.getcwd()}, open(os.path.join(res, "stub_call.json"), "w"))
        if "--af3-json" in opts:                                                          # :2184-2200: the AlphaFold 3 input JSON per job, and main() returns before run() — no model, no marker, no hook
            for n in names:
                open(os.path.join(res, n + ".json"), "w").write("{}"); open(os.path.join(res, n + ".a3m"), "w").write("")
            sys.stderr.write("stub colabfold_batch ran (--af3-json)\\n")
            return rc
        work = os.environ.get("COLABFOLD_OPT_WORK_DIR")
        m = json.loads(os.environ["STUB_MANIFEST"]) if (os.environ.get("STUB_MANIFEST") and work) else None   # as the real hook: no work directory named (the stock route, a bare env-route run), no manifest
        if m is not None:
            from colabfold_opt import manifest
            if "launch_id" in m:                                                        # another launch's manifest (a stale file): its id, not this launch's
                os.environ[manifest.ENV_LAUNCH_ID] = m["launch_id"]
            case = m["no_model_run"] if "no_model_run" in m else manifest.no_model_run(num_models, names, res, keep_existing=keep)   # the hook's decision before the stock run (stack.no_model_run_census), over run()'s facts
            rep = {"active": m.get("active", False), "mode": os.environ.get("COLABFOLD_OPT", "off"), "reason": m.get("reason"),
                   "route": "env", "levers": ["AF_PALLAS_ATTN"], "levers_applied": ["AF_PALLAS_ATTN"] if m.get("active") else [],
                   "levers_unavailable": [] if m.get("active") else ["AF_PALLAS_ATTN"], "queries": len(names), "no_model_run": case,
                   "tokens_min": (m.get("tokens") or [None])[0], "tokens_max": (m.get("tokens") or [None, None])[1]}
            if "n_gpu" in m:                                                            # the GPU count this (stub) model process says it ran at
                rep["n_gpu"] = m["n_gpu"]
            manifest.write(manifest.work_dir(), rep)                                        # where the model process writes it: the launch's work directory
        predicted = 0
        for n in names:                                                                   # colabfold.batch.run's job loop — its completion facts only
            if keep and any(os.path.isfile(os.path.join(res, n + x)) for x in (".result.zip", ".done.txt")):
                continue                                                                  # :1405-1412 a finished job is skipped
            written = [n + ".a3m"]
            open(os.path.join(res, n + ".a3m"), "w").write("")                            # :1456-1458 the MSA
            if num_models == 0:
                open(os.path.join(res, n + ".pickle"), "wb").write(b""); written.append(n + ".pickle")   # :1450-1453 the input features, no model
            for r in range(1, num_models + 1):                                            # :1506-1640 the ranked model files
                for f in (f"{n}_unrelaxed_rank_00{r}_alphafold2_multimer_v3_model_{r}_seed_000.pdb", f"{n}_scores_rank_00{r}_alphafold2_multimer_v3_model_{r}_seed_000.json"):
                    open(os.path.join(res, f), "w").write("ATOM\\n" if f.endswith(".pdb") else "{}"); written.append(f)
            predicted += 1 if num_models > 0 else 0
            if zip_results:                                                               # :1643-1651 the job's files into <job>.result.zip, deleted after — no marker
                with zipfile.ZipFile(os.path.join(res, n + ".result.zip"), "w") as z:
                    for f in written:
                        z.write(os.path.join(res, f), arcname=f)
                for f in written:
                    os.unlink(os.path.join(res, f))
            elif num_models > 0 and n not in os.environ.get("STUB_SKIP_DONE", "").split(","):
                open(os.path.join(res, n + ".done.txt"), "w").write("")                   # :1653-1654 the marker, only when a model ran
        if m is not None and ("calls" in m or "calls_per_job" in m):                     # the exit record: the counters given, or calls_per_job × the jobs that ran a model (a skipped job reaches no lever)
            calls = m["calls"] if "calls" in m else int(m["calls_per_job"]) * predicted
            tp = {"ROWPAIR": {"enabled": True, "n_gpu": m["tp_n_gpu"], "calls": 3}} if "tp_n_gpu" in m else None   # the row-sharded pair stack installed at exit, at that count
            manifest.record_exit(manifest.work_dir(), {"enabled": True, "calls": calls, "fallbacks": 0}, lever_states=tp)
        sys.stderr.write("stub colabfold_batch ran\\n")
        return rc


    if __name__ == "__main__":
        sys.exit(main())
    ''')
STUB_SCRIPT = "#!{python}\nimport sys\nfrom colabfold.batch import main\nsys.exit(main())\n"      # the console script's body (entry point colabfold.batch:main)


def write_stub_package(root):
    """<root>/colabfold/{__init__,batch,input,utils}.py: the stub batch, and colabfold's OWN input.py from the pinned wheel (the package
    names jobs through colabfold.input.get_queries / safe_filename) with the one name it imports from colabfold.utils (MolType)."""
    import ast, zipfile
    os.makedirs(os.path.join(root, "colabfold"), exist_ok=True)
    open(os.path.join(root, "colabfold", "__init__.py"), "w").write("")
    open(os.path.join(root, "colabfold", "batch.py"), "w").write(STUB_BODY)
    with zipfile.ZipFile(WHEEL) as z:
        open(os.path.join(root, "colabfold", "input.py"), "w", encoding="utf-8").write(z.read("colabfold/input.py").decode("utf-8"))
        utils = z.read("colabfold/utils.py").decode("utf-8")
    mol = next(n for n in ast.parse(utils).body if isinstance(n, ast.ClassDef) and n.name == "MolType")
    open(os.path.join(root, "colabfold", "utils.py"), "w", encoding="utf-8").write("from enum import Enum\n\n" + ast.get_source_segment(utils, mol) + "\n")
    return root


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stub = write_stub_package(os.path.join(self.tmp, "stubpkg"))
        self.data = os.path.join(self.tmp, "params_root"); os.makedirs(self.data)
        self.saved_env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT", "STUB_")):
                os.environ.pop(k)
        os.environ["PYTHONPATH"] = os.pathsep.join([OPT, self.stub, CORE_ROOT])      # the package, the stub colabfold and the shared core for the child (importable wherever this interpreter found them)
        sys.path.insert(0, self.stub)                                                # and importable here: launch_form probes `colabfold`
        stack.reset_for_tests()
        self.mans = {}                                                               # results directory -> the launch record `pred` returned (cli.run)

    def main(self, argv):
        """cli.main with the launch record kept: `pred` / `warm` return it in memory (nothing of it is written under the results directory)."""
        rc, rec = cli.run(argv)
        if rec is not None:
            self.mans[rec.get("result_dir")] = rec
        return rc

    def tearDown(self):
        os.environ.clear(); os.environ.update(self.saved_env)
        if self.stub in sys.path:
            sys.path.remove(self.stub)
        for n in [n for n in sys.modules if n == "colabfold" or n.startswith("colabfold.")]:
            sys.modules.pop(n, None)
        stack.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def out(self, name="o"):
        return os.path.join(self.tmp, name)

    def call(self, rd):
        return json.load(open(os.path.join(rd, "stub_call.json")))


class TestLaunchForm(Base):
    """stock_pred.launch_form: the console script beside the interpreter (its shebang naming it), else the import form on the interpreter
    (no script, a foreign / wrapper / env shebang) with the reason recorded; LaunchError only when neither form exists; never `-m`."""

    def interpreter_dir(self, shebang=None, script=True):
        """A `bin/` holding this interpreter (a symlink) and, on request, a `colabfold_batch` beside it. Under a virtual environment the
        directory is made a view of THAT environment (its `pyvenv.cfg` one level up, its `lib`/`lib64`/`include` beside `bin`): CPython finds a
        venv by the `pyvenv.cfg` next to or above the executable's own path, so a bare symlink elsewhere would start the base interpreter
        without the environment's site-packages (the kit and its core not importable)."""
        b = os.path.join(self.tmp, "bin"); os.makedirs(b, exist_ok=True)
        py = os.path.join(b, "python")
        if not os.path.exists(py):
            os.symlink(sys.executable, py)
            cfg = os.path.join(sys.prefix, "pyvenv.cfg")
            if sys.prefix != getattr(sys, "base_prefix", sys.prefix) and os.path.isfile(cfg):
                shutil.copy(cfg, os.path.join(self.tmp, "pyvenv.cfg"))
                for sub in ("lib", "lib64", "include"):
                    src, dst = os.path.join(sys.prefix, sub), os.path.join(self.tmp, sub)
                    if os.path.exists(src) and not os.path.lexists(dst):
                        os.symlink(src, dst)
        if script:
            p = os.path.join(b, "colabfold_batch")
            open(p, "w").write(STUB_SCRIPT.format(python=shebang or py)); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        return py

    def test_script_beside_the_interpreter(self):
        py = self.interpreter_dir()
        self.assertEqual(stock_pred.cli_argv(py), [os.path.join(os.path.dirname(py), "colabfold_batch")])
        self.assertEqual(stock_pred.cli_name(stock_pred.cli_argv(py)), "colabfold_batch")

    def test_foreign_or_wrapper_shebang_takes_the_import_form(self):
        for shebang in ("/usr/bin/env python3", "/bin/sh", "/some/other/python"):
            py = self.interpreter_dir(shebang=shebang)
            d = stock_pred.launch_form(py)
            self.assertEqual((d["form"], d["argv"]), (stock_pred.IMPORT_FORM_NAME, [py, "-c", stock_pred.IMPORT_FORM]), shebang)
            self.assertEqual(d["shebang"], shebang.split()[0]); self.assertIn(f"has the shebang {shebang.split()[0]!r}, not this interpreter", d["reason"])
        p = os.path.join(os.path.dirname(py), "colabfold_batch")
        open(p, "w").write("#!/bin/sh\n" + "'" * 3 + "exec' /very/long/path/bin/python3.11 \"$0\" \"$@\"\n' " + "'" * 3 + "\n")   # pip's wrapper for a long path
        d = stock_pred.launch_form(py)
        self.assertEqual((d["form"], d["shebang"]), (stock_pred.IMPORT_FORM_NAME, "/bin/sh"))

    def test_launch_error_only_when_neither_form_exists(self):
        """No colabfold importable on this interpreter (the probe answered No, whatever the interpreter running the tests carries) and no
        usable script beside it: LaunchError, and `pred` refuses by name (rc 3) before anything is launched."""
        py = self.interpreter_dir(script=False)
        self.assertTrue(stock_pred.importable("colabfold"))                            # the stub (or the wheel) is importable here
        saved = stock_pred.importable; stock_pred.importable = lambda name: False; self.addCleanup(setattr, stock_pred, "importable", saved)
        with self.assertRaises(stock_pred.LaunchError) as cm:
            stock_pred.launch_form(py)
        self.assertIn("no launch form for the stock command line", str(cm.exception)); self.assertIn("colabfold is not importable", str(cm.exception))
        self.interpreter_dir(shebang="/bin/sh")                                        # a script with a wrapper shebang does not rescue it
        with self.assertRaises(stock_pred.LaunchError):
            stock_pred.launch_form(py)
        stock_pred.importable = saved
        self.assertEqual(stock_pred.launch_form(py)["form"], stock_pred.IMPORT_FORM_NAME)
        # pred (gates passed): the named refusal, rc 3, nothing launched — the resolution above, wired: launch_form raises for this interpreter
        def none(executable=None):
            raise stock_pred.LaunchError("no launch form for the stock command line: (simulated) no colabfold_batch beside the interpreter, and colabfold is not importable")
        saved_lf = stock_pred.launch_form; stock_pred.launch_form = none; self.addCleanup(setattr, stock_pred, "launch_form", saved_lf)
        gates = _stubs.gates_pass(stack); self.addCleanup(_stubs.gates_restore, stack, gates)
        with __import__("contextlib").redirect_stderr(__import__("io").StringIO()) as err:
            rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out(), "pred"), "--data", self.data])
        self.assertEqual(rc, 3); self.assertIn("NOT ACTIVE: no launch form for the stock command line", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.out(), "pred", "stub_call.json")))

    def test_import_form_without_a_script(self):
        py = self.interpreter_dir(script=False)
        d = stock_pred.launch_form(py); argv = d["argv"]
        self.assertEqual(argv, [py, "-c", stock_pred.IMPORT_FORM]); self.assertNotIn("-m", argv)
        self.assertEqual((d["form"], d["script"], d["shebang"]), (stock_pred.IMPORT_FORM_NAME, None, None)); self.assertIn("no colabfold_batch beside", d["reason"])
        self.assertEqual(stock_pred.cli_name(argv), stock_pred.IMPORT_FORM_NAME); self.assertNotIn(" ", stock_pred.IMPORT_FORM_NAME)
        self.assertNotIn("-m", stock_pred.cli_argv(sys.executable))                   # this interpreter: the import form or its own script, never -m
        src = open(stock_pred.__file__, encoding="utf-8").read()
        self.assertNotIn('"-m"', src.replace("Never `python -m colabfold.batch`", ""))

    def test_pred_through_the_script_beside_the_interpreter(self):
        """`python -m colabfold_opt pred --mode off` under an interpreter with the console script beside it launches that script."""
        py = self.interpreter_dir()
        env = {k: v for k, v in os.environ.items()}
        r = subprocess.run([py, "-m", "colabfold_opt", "pred", "--mode", "off", A3M, os.path.join(self.out(), "pred"), "--data", self.data],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, r.stderr)                                    # the parameters gate refuses first (no marker under --data)
        self.assertIn("NOT ACTIVE: parameters under", r.stderr); self.assertFalse(os.path.exists(self.out()))
        # with the gate passed (a stub check_pins answer is not reachable from a child): the launch form recorded in the proof
        code = ("import json, sys, colabfold_opt.stack as st, colabfold_opt.cli as cli; st.weights_check = lambda d, refresh=False: ([], {'ok': True}); "
                "rc, rec = cli.run(sys.argv[1:]); print(json.dumps({'proof': rec['stock_proof'], 'argv': rec['argv']})); sys.exit(rc)")
        r = subprocess.run([py, "-c", code, "pred", "--mode", "off", A3M, os.path.join(self.out(), "pred"), "--data", self.data],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        rd = os.path.join(self.out(), "pred")
        rec = json.loads(r.stdout.strip().splitlines()[-1]); proof = rec["proof"]               # the launch record's proof, printed by the child above (no proof file exists)
        self.assertEqual(proof["cli"], "colabfold_batch"); self.assertEqual(os.path.basename(rec["argv"][0]), "colabfold_batch")
        self.assertEqual(os.path.dirname(rec["argv"][0]), os.path.dirname(py)); self.assertFalse(os.path.exists(os.path.join(rd, "stock_env_proof.json")))
        self.assertTrue(self.call(rd)["argv"][-1].endswith("pred")); self.assertIn("STOCK cli=colabfold_batch", r.stderr)


class TestPred(Base):
    def setUp(self):
        super().setUp()
        self.gates = _stubs.gates_pass(stack)                                         # the parent's gates pass on this CPU box

    def tearDown(self):
        _stubs.gates_restore(stack, self.gates)
        super().tearDown()

    def test_off_is_the_stock_cli_in_a_cleaned_environment(self):
        os.environ.update(COLABFOLD_OPT="off", AF_PALLAS_ATTN="1", AF_PALLAS_ATTN_ALL="1")
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + os.path.join(OPT, "forward", "af2_pallas_flash", "af2_pallas_flash")
        rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out(), "pred"), "--data", self.data, "--random-seed", "3"])
        self.assertEqual(rc, 0)
        rd = os.path.join(self.out(), "pred")
        c = self.call(rd)
        self.assertEqual(c["argv"][-2], A3M); self.assertEqual(c["argv"][-1], rd)                              # colabfold_batch's positionals: the input as given, the result directory
        self.assertEqual(c["argv"][:-2], ["--data", self.data, "--random-seed", "3"])                                  # colabfold_batch's options verbatim, in the caller's order
        self.assertEqual([k for k in c["env"] if k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN"))], [])
        pp = c["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(pp, [OPT, self.stub, CORE_ROOT])                                                       # the kit directory stripped, the rest kept, nothing of the kit appended
        self.assertEqual(sorted(os.listdir(self.out())), ["pred"])                                                  # nothing but the results directory under its parent
        self.assertEqual(sorted(f for f in os.listdir(rd) if not f.startswith("1BRS_AD")), ["stub_call.json"])          # nothing of the kit in the results directory: colabfold's files only (no manifest, no log, no proof file)
        man = self.mans[rd]; proof = man["stock_proof"]                                       # the launch record `pred` returned: in memory, never on disk
        form = stock_pred.launch_form()                                                # this interpreter's own form: the script beside it, or the import form
        self.assertTrue(proof["ok"]); self.assertEqual(proof["cli"], stock_pred.cli_name(form["argv"])); self.assertEqual(proof["env_present"], []); self.assertEqual(proof["kit_dirs"], [])
        self.assertEqual(man["argv"][:len(form["argv"])], form["argv"]); self.assertEqual(proof["argv0"], form["argv"][0])
        self.assertEqual(proof["launch"]["form"], form["form"]); self.assertIn(proof["cli"], stock_pred.FORMS)
        self.assertEqual(proof["prefixes"], ["COLABFOLD_OPT", "AF_PALLAS_ATTN"])
        self.assertEqual((man["mode"], man["active"], man["exit_code"], man["command"], man["result_dir"]), ("off", False, 0, "pred", rd))
        self.assertEqual(man["verdict"]["ok"], True); self.assertEqual(man["verdict"]["rc"], 0)
        self.assertEqual(man["settings"]["stock_flags"], ["--data", self.data, "--random-seed", "3"]); self.assertEqual((man["settings"]["models_per_seed"], man["settings"]["num_seeds"]), (5, 1))
        self.assertEqual(man["settings"]["defaults"]["--num-models"], 5); self.assertEqual(man["settings"]["model_config_defaults"]["num_recycle"], 20)
        self.assertNotIn("name", man["settings"]); self.assertNotIn("seed", man["settings"])                   # no preset: stock's flags only
        self.assertEqual(man["stock_proof"]["ok"], True); self.assertEqual(man["inputs"], [{"id": "1BRS_AD", "input": A3M}])
        self.assertEqual(len([o for o in man["outputs"] if o["file"].endswith(".pdb")]), 5)
        self.assertEqual(sorted(o["file"] for o in man["outputs"]), sorted(os.listdir(rd)))                           # the listing is the results directory's own content

    def test_kit_dirs_scan_sees_any_kit_copy(self):
        other = os.path.join(self.tmp, "kitcopy", "af2_pallas_flash"); os.makedirs(other)
        open(os.path.join(other, "af2_pallas_attn.py"), "w").write("")
        self.assertEqual(stock_pred._kit_dirs_on(os.pathsep.join([OPT, other, os.path.dirname(other)])), [other, os.path.dirname(other)])
        env = stock_pred.stock_env({"PYTHONPATH": os.pathsep.join([OPT, other]), "AF_PALLAS_ATTN_ALL": "1", "COLABFOLD_OPT": "fast", "HOME": "/h"})
        self.assertEqual(env, {"PYTHONPATH": OPT, "HOME": "/h", "PYTHONUNBUFFERED": "1"})

    def test_fast_refuses_the_widening_switch_by_name(self):
        os.environ["AF_PALLAS_ATTN_ALL"] = "1"
        self.addCleanup(os.environ.pop, "AF_PALLAS_ATTN_ALL", None)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "t"), "--data", self.data])
        self.assertEqual(rc, 3); self.assertIn("NOT ACTIVE: AF_PALLAS_ATTN_ALL=1 refused:", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.out(), "t", "call.json")))       # nothing launched

    def test_fast_exports_the_switch_and_strips_the_callers_kit_variables(self):
        os.environ.update(AF_PALLAS_ATTN_PRECISE_BWD="1", AF_PALLAS_ATTN="0")
        self.addCleanup(os.environ.pop, "AF_PALLAS_ATTN_PRECISE_BWD", None)
        env = stock_pred.fast_env("fast")
        self.assertEqual([k for k in env if k.startswith("AF_PALLAS_ATTN")], [])
        self.assertEqual(env["COLABFOLD_OPT"], "fast"); self.assertEqual([k for k in env if k.startswith("COLABFOLD_OPT")], ["COLABFOLD_OPT"])   # no launch id / work dir / opt-out unless pred names them
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 7})
        rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "t"), "--data", self.data])
        self.assertEqual(rc, 0)
        rd = os.path.join(self.out(), "t")
        c = self.call(rd)
        self.assertEqual(c["env"]["COLABFOLD_OPT"], "fast")
        self.assertTrue(c["env"]["COLABFOLD_OPT_WORK_DIR"]); self.assertFalse(os.path.exists(c["env"]["COLABFOLD_OPT_WORK_DIR"]))   # the work directory is the launch's and is gone when pred returns
        self.assertEqual([k for k in c["env"] if k.startswith("AF_PALLAS_ATTN")], [])   # the kit's switch is the activation's, inside the child
        self.assertFalse(os.path.exists(os.path.join(rd, "stock_env_proof.json")))
        man = self.mans[rd]
        self.assertEqual((man["mode"], man["exit_code"], man["active"]), ("fast", 0, True)); self.assertIsNone(man["stock_proof"])
        self.assertEqual((man["verdict"]["ok"], man["verdict"]["calls"], man["kit_state_exit"]["calls"]), (True, 7, 7))
        self.assertEqual(man["launch_id"], c["env"]["COLABFOLD_OPT_LAUNCH_ID"]); self.assertEqual(man["verdict"]["launch_id"], man["launch_id"])   # this launch's id, from the child's env
        self.assertEqual(len(man["launch_id"]), 32); self.assertEqual(man["gates"]["mode"], "fast"); self.assertIsNone(man["gates"].get("would_refuse"))

    def test_verdict_is_the_manifest_not_the_child_rc(self):
        """A fast child that exits 0 without activating (hook_never_fired, 3), or with calls=0 (kernel_not_engaged: PARTIAL, 3 by name),
        or without a completion marker (missing_outputs, 1), is a named state with its own code; the partial state prints the house line."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("a"), "pred"), "--data", self.data])
        self.assertEqual(rc, 3)                                                        # no activation report: the hook never fired
        v = self.mans[os.path.join(self.out("a"), "pred")]["verdict"]
        self.assertEqual((v["reason"], v["rc"], v["child_rc"], v["partial"]), ("hook_never_fired", 3, 0, []))
        self.assertIn("NOT ACTIVE: no activation report of this launch", err.getvalue())
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 0})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("b"), "pred"), "--data", self.data])
        self.assertEqual(rc, 3)
        man = self.mans[os.path.join(self.out("b"), "pred")]; v = man["verdict"]
        self.assertEqual((v["reason"], v["calls"], v["rc"], v["partial"]), ("kernel_not_engaged", 0, 3, ["AF_PALLAS_ATTN"])); self.assertNotIn("allow_partial", v)
        self.assertEqual((man["partial"], man["exit_code"], man["child_rc"]), (["AF_PALLAS_ATTN"], 3, 0))
        self.assertIn("[colabfold-opt] NOT ACTIVE: partial activation — AF_PALLAS_ATTN: EXIT calls=0 (the kernel ran no attention call); exit 3\n", err.getvalue())
        self.assertEqual(v["exit_by"], "verdict")
        os.environ.pop("STUB_MANIFEST"); os.environ["STUB_SKIP_DONE"] = "1BRS_AD"
        rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("d"), "pred"), "--data", self.data])
        self.assertEqual(rc, 1)
        v = self.mans[os.path.join(self.out("d"), "pred")]["verdict"]
        self.assertEqual((v["reason"], v["missing"], v["partial"]), ("missing_outputs", ["1BRS_AD"], []))

    @unittest.skipIf(not HAVE_AXIS, "the pinned core has no opt_core.mem.ngpu / rowpair_jax: --n_gpu > 1 is refused by name there (test_core_older)")
    def test_n_gpu_reaches_the_model_process_and_a_dropped_axis_is_refused(self):
        """`pred --mode big --n_gpu 2`: the model process receives P (its environment carries COLABFOLD_OPT_N_GPU=2) and the verdict is
        FAIL-CLOSED on the count that process reports — a run whose ACTIVE report says another P (an inner call dropped the axis), or that
        reports P > 1 without the row-sharded pair stack installed at exit, is `n_gpu_mismatch requested=P active=Q`, rc 3, never a pass."""
        stack.gpu_info = lambda index=0: {**_stubs.GPU_H100, "count": 2}                # two cards visible to the command line's gate
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 5, "n_gpu": 2, "tp_n_gpu": 2})
        rc = self.main(["pred", "--mode", "big", "--n_gpu", "2", A3M, os.path.join(self.out("a"), "pred"), "--data", self.data])
        rd = os.path.join(self.out("a"), "pred"); c = self.call(rd)
        self.assertEqual((c["env"]["COLABFOLD_OPT"], c["env"]["COLABFOLD_OPT_N_GPU"]), ("big", "2"))     # the requested P reached the model process
        v = self.mans[rd]["verdict"]
        self.assertEqual((rc, v["ok"], v["n_gpu_requested"], v["n_gpu_active"]), (0, True, 2, 2), v)
        for tag, stub, active in (("b", {"active": True, "calls": 5, "n_gpu": 1}, 1),                    # the model process ran at P=1: the axis was dropped inside
                                  ("c", {"active": True, "calls": 5, "n_gpu": 2}, None),                 # P=2 reported, no row-sharded stack installed at exit
                                  ("d", {"active": True, "calls": 5, "n_gpu": 2, "tp_n_gpu": 1}, None),  # the lever disagrees with the report
                                  ("e", {"active": True, "calls": 5}, 1)):                               # a report without the axis at all
            os.environ["STUB_MANIFEST"] = json.dumps(stub)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = self.main(["pred", "--mode", "big", "--n_gpu", "2", A3M, os.path.join(self.out(tag), "pred"), "--data", self.data])
            v = self.mans[os.path.join(self.out(tag), "pred")]["verdict"]
            self.assertEqual((rc, v["reason"], v["n_gpu_requested"], v["n_gpu_active"]), (3, "n_gpu_mismatch", 2, active), (tag, v))   # never a pass, never a partial state
            self.assertIn(f"[colabfold-opt] NOT ACTIVE: n_gpu_mismatch requested=2 active={active}\n", err.getvalue(), tag)
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 5})                            # P=1 (the default): a report without the axis counts as 1
        rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("f"), "pred"), "--data", self.data])
        v = self.mans[os.path.join(self.out("f"), "pred")]["verdict"]; self.assertEqual((rc, v["n_gpu_requested"], v["n_gpu_active"]), (0, 1, 1))
        self.assertNotIn("COLABFOLD_OPT_N_GPU", self.call(os.path.join(self.out("f"), "pred"))["env"])    # fast exports no axis variable

    @unittest.skipIf(not HAVE_AXIS, "the pinned core has no opt_core.mem.ngpu / rowpair_jax: --n_gpu > 1 is refused by name there (test_core_older)")
    def test_n_gpu_8_starts_the_model_process_at_its_pool_fraction(self):
        """`pred --mode big --n_gpu 8`: the model process starts with XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 (modes.N_GPU_MEM_FRACTION: NCCL's
        communicators allocate outside jax's pool; the P = 8 line runs at 0.90) when the caller left the fraction unset or at the image's own
        preset (stock/PINS.json image.env, 0.95); a fraction the caller chose — another value, or jax's other name XLA_CLIENT_MEM_FRACTION —
        reaches the model process untouched; P = 2 leaves the environment as it is."""
        stack.gpu_info = lambda index=0: {**_stubs.GPU_H100, "count": 8}
        F, ALT = modes.MEM_FRACTION_ENV, modes.MEM_FRACTION_ENV_ALT
        preset = stack.pins()["image"]["env"][F]
        self.assertEqual((preset, modes.N_GPU_MEM_FRACTION, F, ALT), ("0.95", {8: "0.90"}, "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION"))
        cases = (("a", 8, {}, "0.90", None),               # unset: the P = 8 fraction
                 ("b", 8, {F: preset}, "0.90", None),      # the image's preset is the stack's value, not the user's: the P = 8 fraction
                 ("c", 8, {F: "0.85"}, "0.85", None),      # the caller's own fraction wins
                 ("d", 8, {ALT: "0.8"}, None, "0.8"),      # jax's other name set by the caller: theirs, nothing added
                 ("e", 2, {}, None, None),                 # P = 2: the environment as it is
                 ("f", 2, {F: preset}, preset, None))
        for tag, p, extra, want, want_alt in cases:
            for k in (F, ALT):
                os.environ.pop(k, None)
            os.environ.update(extra)
            os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 5, "n_gpu": p, "tp_n_gpu": p})
            rc = self.main(["pred", "--mode", "big", "--n_gpu", str(p), A3M, os.path.join(self.out(tag), "pred"), "--data", self.data])
            c = self.call(os.path.join(self.out(tag), "pred"))
            self.assertEqual((rc, c["env"].get(F), c["env"].get(ALT), c["env"]["COLABFOLD_OPT_N_GPU"]), (0, want, want_alt, str(p)), tag)

    def test_a_partial_state_is_exit_3_and_recorded(self):
        """A lever applied whose kernel no call reached (`partial`) is `kernel_not_engaged`, rc 3 — a mode is all of its levers, there is no
        opt-out — and a failed model process keeps its own state (1) with the partial recorded beside it; `COLABFOLD_OPT_ALLOW_PARTIAL` is no
        name of this kit (refused as undeclared, rc 2)."""
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 0})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("a"), "pred"), "--data", self.data])
        self.assertEqual(rc, 3)
        man = self.mans[os.path.join(self.out("a"), "pred")]; v = man["verdict"]
        self.assertEqual((v["ok"], v["rc"], v["reason"], v["partial"]), (False, 3, "kernel_not_engaged", ["AF_PALLAS_ATTN"])); self.assertNotIn("allow_partial", man)
        self.assertEqual(err.getvalue().count("[colabfold-opt] NOT ACTIVE: partial activation — AF_PALLAS_ATTN: EXIT calls=0 (the kernel ran no attention call); exit 3\n"), 1)
        os.environ["STUB_RC"] = "1"
        rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("d"), "pred"), "--data", self.data])
        self.assertEqual(rc, 1)                                                        # a failed model process keeps its own state, the partial recorded
        v = self.mans[os.path.join(self.out("d"), "pred")]["verdict"]
        self.assertEqual((v["reason"], v["partial"]), ("model_process_failed", ["AF_PALLAS_ATTN"]))
        os.environ.pop("STUB_RC"); os.environ["COLABFOLD_OPT_ALLOW_PARTIAL"] = "1"
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("e"), "pred"), "--data", self.data])
            self.assertEqual(rc, 2); self.assertIn("usage: undeclared variable(s) COLABFOLD_OPT_ALLOW_PARTIAL ", err.getvalue())
        finally:
            os.environ.pop("COLABFOLD_OPT_ALLOW_PARTIAL")

    def test_warm_exits_as_pred_does(self):
        """warm's exit code is pred's own — a partial activation is 3 (`WARM NOT ACTIVE`), never collapsed to 1, the levers named (`partial=`)."""
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 0})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = self.main(["warm", "--mode", "fast", "--out", self.out("w1"), "--data", self.data])
        self.assertEqual(rc, 3)
        line = [l for l in err.getvalue().splitlines() if "] WARM " in l][-1]
        self.assertTrue(line.startswith("[colabfold-opt] WARM NOT ACTIVE mode=fast rc=3 reason=kernel_not_engaged"), line)
        self.assertIn(" partial=AF_PALLAS_ATTN out=", line)
        os.environ.pop("STUB_MANIFEST"); os.environ["STUB_SKIP_DONE"] = "1BRS_AD"
        with contextlib.redirect_stderr(err):
            rc = self.main(["warm", "--mode", "off", "--out", self.out("w3"), "--data", self.data])
        self.assertEqual(rc, 1)
        self.assertIn("[colabfold-opt] WARM FAIL mode=off rc=1 reason=missing_outputs", err.getvalue())

    def test_rerun_into_a_result_dir_and_a_stale_manifest_never_vouches(self):
        """A result_dir carrying a previous run is colabfold's to resume: it skips a job whose <jobname>.done.txt (or, under --zip,
        <jobname>.result.zip) exists, so a second launch into a complete directory builds no model and no lever call — that launch is
        `no_model_run=all_jobs_done` by name (the IDLE line), its census its own (calls=0), rc 0 as stock's, never a partial activation;
        a manifest that does not carry this launch's id is not this launch's activation report (hook_never_fired, rc 3); an
        opt_manifest.json planted under <results> vouches for nothing."""
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls_per_job": 112})   # the stub's exit census: 112 kernel calls per job that ran a model, 0 for a skipped job
        rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "pred"), "--data", self.data])
        self.assertEqual(rc, 0)
        rd = os.path.join(self.out(), "pred"); first = self.mans[rd]
        self.assertEqual((first["kit_state_exit"]["calls"], first["verdict"]["no_model_run"], first["verdict"]["completion"]), (112, None, {"1BRS_AD": "done.txt"}))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "pred"), "--data", self.data])
        self.assertEqual(rc, 0, err.getvalue()); second = self.mans[rd]
        self.assertNotEqual(second["launch_id"], first["launch_id"])                   # launched again into the same directory; this launch's manifest
        v = second["verdict"]
        self.assertEqual((v["no_model_run"], v["calls"], v["partial"], v["idle"], second["kit_state_exit"]["calls"]), ("all_jobs_done", 0, [], ["AF_PALLAS_ATTN"], 0))   # colabfold skipped the finished job: zero lever calls, idle by name
        self.assertIn("[colabfold-opt] IDLE no_model_run=all_jobs_done levers=not_applicable:AF_PALLAS_ATTN jobs=1: every job was complete in the results directory "
                      "before the run (1/1: 1BRS_AD.done.txt)", err.getvalue())
        self.assertNotIn("NOT ACTIVE", err.getvalue())
        json.dump({"active": True, "launch_id": second["launch_id"]}, open(os.path.join(rd, "opt_manifest.json"), "w"))   # a manifest planted under <results>: nothing reads it
        os.environ.pop("STUB_MANIFEST")                                                # the child writes none of its own —
        self.assertEqual(self.main(["pred", "--mode", "fast", A3M, rd, "--data", self.data]), 3)                        # — hook_never_fired, complete directory or not
        self.assertEqual(self.mans[rd]["verdict"]["reason"], "hook_never_fired")
        # a stale manifest the child finds cannot vouch: the stub writes a previous run's manifest (another launch id)
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 112, "launch_id": first["launch_id"]})
        rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("b"), "pred"), "--data", self.data])
        self.assertEqual(rc, 3)
        v = self.mans[os.path.join(self.out("b"), "pred")]["verdict"]
        self.assertEqual(v["reason"], "hook_never_fired"); self.assertNotEqual(v["launch_id"], first["launch_id"])
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 112, "launch_id": ""})           # no id at all: the same
        self.assertEqual(self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("c"), "pred"), "--data", self.data]), 3)
        os.environ.pop("STUB_MANIFEST"); os.environ["STUB_SKIP_DONE"] = "1BRS_AD"
        rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("d"), "pred"), "--data", self.data])
        self.assertEqual(rc, 1)
        os.environ.pop("STUB_SKIP_DONE")                                               # a marker-less failed run leaves its files; the next launch into the directory runs and reports itself
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("d"), "pred"), "--data", self.data]), 0)
        self.assertEqual(self.mans[os.path.join(self.out("d"), "pred")]["verdict"]["no_model_run"], None)

    def test_pred_prints_no_dry_run_line(self):
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 3})
        err = __import__("io").StringIO()
        with __import__("contextlib").redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "pred"), "--data", self.data])
        self.assertEqual(rc, 0); self.assertNotIn("DRY-RUN", err.getvalue())
        self.assertEqual(self.mans[os.path.join(self.out(), "pred")]["gates"].get("would_refuse"), None)

    def test_gates_run_before_the_launch(self):
        """The parameters gate in every mode (colabfold would fetch the parameters, batch.py:2164) and every gate for a kit mode, before
        anything is staged or launched."""
        stack.weights_check = lambda d, refresh=False: ([f"parameters under {d}: marker missing"], {"ok": False})
        for mode in ("off", "fast"):
            rc = self.main(["pred", "--mode", mode, A3M, os.path.join(self.out(mode), "pred"), "--data", self.data])
            self.assertEqual(rc, 3, mode); self.assertFalse(os.path.exists(self.out(mode)), mode)
        _stubs.gates_restore(stack, self.gates); self.gates = _stubs.gates_pass(stack, gpu=None)      # no GPU: a kit mode refuses, off does not
        self.assertEqual(self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out("f"), "pred"), "--data", self.data]), 3)
        self.assertFalse(os.path.exists(self.out("f")))
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("g"), "pred"), "--data", self.data]), 0)

    def test_default_mode_and_env_mode(self):
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 1})
        rc = self.main(["pred", A3M, os.path.join(self.out(), "pred"), "--data", self.data])                       # no mode: the default, fast (modes.DEFAULT_MODE)
        self.assertEqual(rc, 0); self.assertEqual(self.call(os.path.join(self.out(), "pred"))["env"]["COLABFOLD_OPT"], "fast")
        self.assertEqual(self.mans[os.path.join(self.out(), "pred")]["mode"], "fast")
        os.environ["COLABFOLD_OPT"] = "off"                                                                           # the environment route selects off by name
        rc = self.main(["pred", A3M, os.path.join(self.out("o2"), "pred"), "--data", self.data])
        self.assertEqual(rc, 0); self.assertNotIn("COLABFOLD_OPT", self.call(os.path.join(self.out("o2"), "pred"))["env"])
        self.assertEqual(self.mans[os.path.join(self.out("o2"), "pred")]["mode"], "off")

    def test_stock_flags_pass_through_verbatim(self):
        rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out(), "pred"), "--data", self.data, "--num-recycle", "3", "--num-seeds", "2"])
        self.assertEqual(rc, 0)
        argv = self.call(os.path.join(self.out(), "pred"))["argv"]
        self.assertEqual(argv[:-2], ["--data", self.data, "--num-recycle", "3", "--num-seeds", "2"])                # verbatim, in the caller's order
        self.assertEqual(self.mans[os.path.join(self.out(), "pred")]["settings"]["num_seeds"], 2)
        rc = self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("o0"), "pred"), "--data", self.data])   # no flag: upstream's defaults exactly — nothing but --data is added
        self.assertEqual(rc, 0); self.assertEqual(self.call(os.path.join(self.out("o0"), "pred"))["argv"][:-2], ["--data", self.data])
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("o2"), "pred"), "--data", self.data, "--random-seed", "5"]), 0)   # stock's seed flag is stock's: passed through
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("o3"), "pred"), "--data", self.data, "--random-seed", "1"]), 0)   # stock's own --data after `--`: the parameters root, verbatim, once
        c3 = self.call(os.path.join(self.out("o3"), "pred")); self.assertEqual(c3["argv"][:-2], ["--data", self.data, "--random-seed", "1"]); self.assertEqual(self.mans[os.path.join(self.out("o3"), "pred")]["data_dir"], self.data)
        self.assertEqual(self.main(["pred", "--mode", "off", "--seed", "3", A3M, os.path.join(self.out("o5"), "pred"), "--data", self.data]), 2)          # the kit has no --seed of its own: an option it does not know ahead of the positionals is a usage refusal (behind them it is colabfold_batch's to judge)
        self.assertEqual(self.main(["pred", "--mode", "off", "--settings", "record", A3M, os.path.join(self.out("o4"), "pred"), "--data", self.data, "--num-recycle", "3"]), 2)   # the kit has no --settings: usage refusal

    def test_colabfold_batchs_command_line_is_taken_as_its_parser_takes_it(self):
        """`pred <input> <results> [colabfold_batch options]`: the two positionals are found wherever colabfold_batch's own parser finds
        them (its valued options take one token, `--opt=value` and its bare options none, everything after a bare `--` is positional) and
        the composed argv is the options verbatim in the caller's order, `--data` when they carry none, then the positionals as written;
        the kit's own flags may stand anywhere. A row's command in the layer's form (`run.sh pred --mode M <dir> <results> --random-seed 0
        --num-models 1`) composes the argv the retired `--input <dir> --out_dir <out> --tag pred -- --random-seed 0 --num-models 1` form
        composed for it: `[--random-seed 0 --num-models 1 --data <root> <dir> <out>/pred]`."""
        rd = os.path.join(self.out(), "pred")
        self.assertEqual(self.main(["pred", "--num-models", "1", A3M, "--mode", "off", rd, "--random-seed", "0", "--data", self.data, "--templates"]), 0)
        self.assertEqual(self.call(rd)["argv"], ["--num-models", "1", "--random-seed", "0", "--data", self.data, "--templates", A3M, rd])   # options in order, positionals last, the kit's --mode gone
        rd2 = os.path.join(self.out("row"), "pred")
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, rd2, "--random-seed", "0", "--num-models", "1", "--data", self.data]), 0)   # the row form with the root as colabfold_batch's own --data
        self.assertEqual(self.call(rd2)["argv"], ["--random-seed", "0", "--num-models", "1", "--data", self.data, A3M, rd2])            # = the retired form's argv for the same row
        os.environ["COLABFOLD_OPT_DATA_DIR"] = self.data
        rd2b = os.path.join(self.out("rowenv"), "pred")
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, rd2b, "--random-seed", "0", "--num-models", "1"]), 0)             # the row form exactly, the root from the config variable
        self.assertEqual(self.call(rd2b)["argv"], ["--random-seed", "0", "--num-models", "1", "--data", self.data, A3M, rd2b])
        os.environ.pop("COLABFOLD_OPT_DATA_DIR")
        self.assertEqual(self.mans[rd2]["settings"]["stock_flags"], ["--random-seed", "0", "--num-models", "1", "--data", self.data]); self.assertEqual(self.mans[rd2]["settings"]["models_per_seed"], 1)
        rd3 = os.path.join(self.out("dd"), "pred")
        self.assertEqual(self.main(["pred", "--mode", "off", "--data", self.data, "--", A3M, rd3]), 0)                                # a bare `--`: colabfold_batch's own, kept where the caller put it
        self.assertEqual(self.call(rd3)["argv"], ["--data", self.data, "--", A3M, rd3])
        rd4 = os.path.join(self.out("eq"), "pred")
        self.assertEqual(self.main(["pred", "--mode=off", "--num-models=2", A3M, rd4, "--data", self.data]), 0)                       # `--opt=value` forms, the kit's and colabfold's
        self.assertEqual(self.call(rd4)["argv"], ["--num-models=2", "--data", self.data, A3M, rd4]); self.assertEqual(self.mans[rd4]["settings"]["models_per_seed"], 2)
        rel = os.path.relpath(os.path.join(self.out("rel"), "pred"))                                                                  # relative positionals mean what the caller wrote: the model process runs in the caller's directory
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, rel, "--data", self.data]), 0)
        self.assertEqual(self.call(os.path.abspath(rel))["argv"][-1], rel); self.assertEqual(self.call(os.path.abspath(rel))["cwd"], os.getcwd())
        self.assertIn(os.path.abspath(rel), self.mans)                                                                                # the record names the results directory it made
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.main(["pred", "--mode", "off", A3M, "--data", self.data]), 2)                                      # one positional: usage, by name
            self.assertEqual(self.main(["pred", "--mode", "off", A3M, rd, "third", "--data", self.data]), 2)                          # three
            self.assertEqual(self.main(["pred", "--mode", "off", "--nonesuch", "x", A3M, rd, "--data", self.data]), 2)                 # an option outside colabfold_batch's table ahead of the positionals: its value cannot be told from a positional
            self.assertEqual(self.main(["pred", "--mode", "off", "--out", rd, A3M, rd, "--data", self.data]), 2)                       # --out is warm's
            self.assertEqual(self.main(["check", "--mode", "off", A3M]), 2)                                                            # check takes no colabfold_batch arguments
        self.assertIn("usage: pred: colabfold_batch's two positionals, <input> <results>, are required (found 1:", err.getvalue())
        self.assertIn("usage: pred: --nonesuch: not an option of colabfold_batch this package knows", err.getvalue())

    def test_the_option_table_is_colabfold_batchs_own(self):
        """settings.options_table() is read from the pinned wheel's own `main()` parser (stock/PINS.json upstream.colabfold.wheel.file), never
        copied: colabfold_batch's two positionals, its valued / bare options and the one whose value is optional (`--initial-guess`)."""
        tb = settings.options_table()
        self.assertEqual(tb["positionals"], ["input", "results"])
        self.assertTrue({"--num-models", "--num-seeds", "--random-seed", "--data", "--model-type", "--num-recycle", "--jobname-prefix"} <= tb["valued"])
        self.assertTrue({"--templates", "--amber", "--zip", "--save-all"} <= tb["bare"]); self.assertEqual(tb["optional_value"], {"--initial-guess"})
        self.assertFalse(tb["valued"] & tb["bare"] - tb["optional_value"]); self.assertGreaterEqual(len(tb["valued"]), 30); self.assertGreaterEqual(len(tb["bare"]), 15)
        self.assertTrue(set(settings.UPSTREAM_DEFAULTS) <= tb["valued"] | tb["bare"])                                  # every default the package cites is an option of that parser
        self.assertEqual(inputs.split_argv(["in.a3m", "out", "--templates", "--num-recycle", "3"]), ("in.a3m", "out", ["--templates", "--num-recycle", "3"], False))
        self.assertEqual(inputs.split_argv(["--initial-guess", "guess.pdb", "in", "out"]), ("in", "out", ["--initial-guess", "guess.pdb"], False))   # nargs='?' takes the following token, as argparse does
        self.assertEqual(inputs.split_argv(["in", "out", "--initial-guess"]), ("in", "out", ["--initial-guess"], False))
        self.assertEqual(inputs.split_argv(["in", "out", "--future-flag", "v"]), ("in", "out", ["--future-flag", "v"], False))               # behind the positionals an unknown option is colabfold_batch's to judge
        with self.assertRaises(inputs.UsageError):
            inputs.split_argv(["--future-flag", "v", "in", "out"])

    def test_det_0_changes_nothing_and_det_1_is_refused_before_anything_runs(self):
        """--det 0 (the default) leaves the composed argv byte-identical to a call without it and is recorded as det 0 in the manifest;
        --det 1 is a usage refusal by name (rc 2, the package carries no deterministic recipe) before the gates or the launch — for every
        command, not only pred; any other value is a usage error."""
        base = ["pred", "--mode", "off", A3M, "--data", self.data]
        self.assertEqual(self.main(base + [os.path.join(self.out("a"), "pred"), "--random-seed", "3"]), 0)
        self.assertEqual(self.main(base + [os.path.join(self.out("b"), "pred"), "--det", "0", "--random-seed", "3"]), 0)
        ca, cb = self.call(os.path.join(self.out("a"), "pred")), self.call(os.path.join(self.out("b"), "pred"))
        self.assertEqual(ca["argv"][:-2], cb["argv"][:-2]); self.assertEqual(cb["argv"][:-2], ["--data", self.data, "--random-seed", "3"])   # colabfold_batch's options in the caller's order
        self.assertNotIn("--det", cb["argv"])
        self.assertEqual((self.mans[os.path.join(self.out("a"), "pred")]["det"], self.mans[os.path.join(self.out("b"), "pred")]["det"]), (0, 0))
        err = __import__("io").StringIO()
        with __import__("contextlib").redirect_stderr(err):
            rc = self.main(base + [os.path.join(self.out("c"), "pred"), "--det", "1"])
        self.assertEqual(rc, 2); self.assertIn("--det 1 is refused: the package carries no deterministic recipe", err.getvalue())
        self.assertFalse(os.path.exists(self.out("c")))                                   # nothing staged, nothing launched
        with __import__("contextlib").redirect_stderr(__import__("io").StringIO()):
            self.assertEqual(self.main(base + [os.path.join(self.out("d"), "pred"), "--det", "2"]), 2)
            self.assertEqual(self.main(base + [os.path.join(self.out("d"), "pred"), "--det", "x"]), 2)
            self.assertEqual(self.main(["check", "--mode", "off", "--det", "1"]), 2)
        self.assertFalse(os.path.exists(self.out("d")))
        self.assertIn("[--det 0|1]", cli.USAGE)

    def test_undeclared_variable_name_is_refused(self):
        """A variable under the kit's prefix that nothing reads (COLABFOLD_OPT_MODE=fast: a mistyped name) is refused by the command line
        before anything runs — rc 2, the usage line naming it and the names the kit reads; never ignored, never stripped silently."""
        base = ["pred", "--mode", "off", A3M, "--data", self.data]
        for name in ("COLABFOLD_OPT_MODE", "COLABFOLD_OPTS", "COLABFOLD_OPT_SIZE_RULE"):        # the last: a retired name, refused like any other
            with mock.patch.dict(os.environ, {name: "fast"}):
                self.assertEqual(_autoload.undeclared(), [name])
                err = __import__("io").StringIO()
                with __import__("contextlib").redirect_stderr(err):
                    rc = self.main(base + [os.path.join(self.out("u"), "pred")])
                self.assertEqual(rc, 2); self.assertIn(f"usage: undeclared variable(s) {name} (the names this kit reads: COLABFOLD_OPT, COLABFOLD_OPT_DATA_DIR, ", err.getvalue())
                self.assertFalse(os.path.exists(self.out("u")))
                with __import__("contextlib").redirect_stderr(__import__("io").StringIO()):
                    self.assertEqual(self.main(["check", "--mode", "off"]), 2)
        with mock.patch.dict(os.environ, {"COLABFOLD_OPT_DATA_DIR": self.data}):   # a declared name: read, not refused
            self.assertEqual(_autoload.undeclared(), [])

    def test_data_dir_from_env_or_flag(self):
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out(), "pred")]), 2)
        os.environ["COLABFOLD_OPT_DATA_DIR"] = self.data
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out(), "pred")]), 0)
        self.assertEqual(self.call(os.path.join(self.out(), "pred"))["argv"][-4:-2], ["--data", self.data])

    def test_child_exit_codes(self):
        os.environ["STUB_RC"] = "3"
        self.assertEqual(self.main(["pred", "--mode", "fast", A3M, os.path.join(self.out(), "pred"), "--data", self.data]), 3)
        man = self.mans[os.path.join(self.out(), "pred")]
        self.assertEqual((man["exit_code"], man["verdict"]["reason"]), (3, "not_active"))
        os.environ["STUB_RC"] = "1"
        self.assertEqual(self.main(["pred", "--mode", "off", A3M, os.path.join(self.out("o2"), "pred"), "--data", self.data]), 1)
        self.assertEqual(self.mans[os.path.join(self.out("o2"), "pred")]["verdict"]["reason"], "model_process_failed")

    def test_input_forms_are_colabfolds_and_job_names_are_colabfolds(self):
        d = os.path.join(self.tmp, "in"); os.makedirs(d)
        shutil.copyfile(A3M, os.path.join(d, "x1.a3m")); shutil.copyfile(A3M, os.path.join(d, "second-2.b.a3m")); shutil.copyfile(A3M, os.path.join(d, "odd id.a3m"))
        rc = self.main(["pred", "--mode", "off", d, os.path.join(self.out(), "pred"), "--data", self.data])
        self.assertEqual(rc, 0); self.assertEqual(self.call(os.path.join(self.out(), "pred"))["argv"][-2], d)          # the directory itself, verbatim
        self.assertEqual(sorted(i["id"] for i in self.mans[os.path.join(self.out(), "pred")]["inputs"]), ["odd_id", "second-2.b", "x1"])   # colabfold's names (safe_filename of the stems)
        fasta = os.path.join(self.tmp, "q.fasta"); open(fasta, "w").write(">q one\nMKTAYIAKQR\n>r|two\nGSHM:AAAA\n")
        self.assertEqual(inputs.jobs(fasta), ["r_two", "q_one"])                                                          # a fasta: one job per record, named by colabfold from the header, in its length order
        self.assertEqual(inputs.jobs(fasta, ["--jobname-prefix", "job", "--sort-queries-by", "none"]), ["job_0", "job_1"])
        ten = os.path.join(self.tmp, "ten.csv")
        open(ten, "w").write("id,sequence\n" + "".join(f"s{i},{'ACDEFGHIKL'[i:] + 'MNPQ'}\n" for i in range(10)))
        self.assertEqual(inputs.jobs(ten, ["--jobname-prefix", "job", "--sort-queries-by", "none"]), [f"job_{n:02d}" for n in range(10)])   # batch.py:1393-1396: zero-filled to the width of the query count  # batch.py:1393-1396
        csv = os.path.join(self.tmp, "q.csv"); open(csv, "w").write("id,sequence\nc1,MKTAYIAKQR\n")
        self.assertEqual(inputs.jobs(csv), ["c1"])
        for p, exc in ((os.path.join(self.tmp, "items.json"), ValueError), (os.path.join(self.tmp, "nosuch.a3m"), OSError), (os.path.join(self.tmp, "empty.a3m"), ValueError)):
            if p.endswith("items.json"): open(p, "w").write("[]")
            if p.endswith("empty.a3m"): open(p, "w").write("")
            with self.assertRaises(exc):
                inputs.jobs(p)                                                                                            # colabfold's own errors: unknown format, not found, empty
        self.assertEqual(self.main(["pred", "--mode", "off", os.path.join(self.tmp, "items.json"), os.path.join(self.out("o3"), "pred"), "--data", self.data]), 2)


class TestUsage(Base):
    def test_unknown_and_disagreement(self):
        self.assertEqual(self.main(["pred", "--mode", "turbo", A3M, os.path.join(self.out(), "pred"), "--data", self.data]), 2)
        self.assertEqual(self.main(["check", "--mode", "turbo"]), 2)
        self.assertEqual(self.main(["check", "--mode", "exact"]), 3)        # a kit mode: gated like fast (no colabfold, no GPU here: would refuse), never a usage error
        os.environ["COLABFOLD_OPT"] = "exact"
        self.assertEqual(self.main(["check", "--mode", "fast"]), 2)                              # the environment names another mode: a disagreement
        os.environ["COLABFOLD_OPT"] = "off"
        self.assertEqual(self.main(["check", "--mode", "fast"]), 2)
        self.assertEqual(self.main(["check", "--mode", "off"]), 0)
        self.assertEqual(self.main(["bogus"]), 2); self.assertEqual(self.main([]), 2); self.assertEqual(self.main(["--help"]), 0)
        self.assertEqual(self.main(["pred", "--mode", "off"]), 2)

    def test_check_exit_codes(self):
        self.assertEqual(self.main(["check", "--mode", "off"]), 0)
        self.assertEqual(self.main(["check", "--mode", "fast"]), 3)          # no colabfold, no GPU here: would refuse
        r = subprocess.run([sys.executable, "-m", "colabfold_opt", "check", "--mode", "fast", "--json"], capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(r.returncode, 3)
        rep = json.loads(r.stdout); self.assertIn("would_refuse", rep); self.assertIn("DRY-RUN mode=fast", r.stderr)


class TestPublicInputForm(unittest.TestCase):
    """The bundled example is ColabFold's complex a3m form (the paired block's first record = the full query, `>101\t102`, then the
    per-chain unpaired records padded to the complex) — stock's own serialization, read back by stock (test_bundled_inputs)."""

    def test_public_input_form(self):
        lines = open(A3M, encoding="utf-8").read().splitlines()
        self.assertEqual(lines[0], "#110,89\t1,1"); self.assertEqual(lines[1], ">101\t102")
        self.assertEqual(len(lines[2]), 199); self.assertTrue(lines[2].isalpha())                    # the full query, no gaps
        self.assertEqual(lines[3], ">101"); self.assertEqual(lines[4], lines[2][:110] + "-" * 89)  # the per-chain unpaired records, padded
        self.assertEqual(lines[5], ">102"); self.assertEqual(lines[6], "-" * 110 + lines[2][110:])
