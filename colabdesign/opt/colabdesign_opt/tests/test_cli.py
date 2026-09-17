"""The CLI end to end on the stub stack (a subprocess per command, the tree as COLABDESIGN_OPT_HOME, an nvidia-smi stand-in on PATH):
`design` off / fast with the report lines and the file set, the refusals by name (`exact` is no mode of this kit), `check`, the mode-vs-environment rule, the
weights warn-and-run and digest-memo behavior; in-process with the arm stubbed, the partial-activation / fail-loud grammar (a stock arm
printing a lever line, a missing or fallen-back lever, a gated lever, incomplete outputs); the core pin gate at every
entry point (the console script, enable()/status(), the arm launchers, run.sh -- an absent or stale core is one NOT ACTIVE line and exit 3,
never a traceback); and the further verb built on the same arm: `warm` (the kit's own target-writer block located
in the kit file and executed verbatim, a design through `design`, the result JSON and the WARM line)."""

import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import pytest
except ImportError:                                        # an interpreter without pytest (the stack image): the kit's stand-in
    from colabdesign_opt.tests import _noptest as pytest

from . import _stubs
from .test_levers import APPLIED, APPLIED_FN, CACHE_OK, DEFECT, EXACT_OK, GATED, GATED_FN, HOIST_OK, PALLAS_OK, PALLAS_REPLACED, PARCOMPILE_OK, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRIMUL_OK, nosub_fn_line, nosub_line, pallas_line, triatt_line, TXLA_OK, TXLA_GATED, TRANSITION_OK, TRANSITION_SMALL
from colabdesign_opt import bindcraft, cli, driver, inputs, names, outputs, report, settings, stack, warm

_IT = settings.iteration_counts(settings.load(_stubs.TREE)["advanced"]) if _stubs.tree_present() else {"soft": 0, "temp": 0, "hard": 0, "greedy": 0}
STEPS, GREEDY = _IT["soft"] + _IT["temp"] + _IT["hard"], _IT["greedy"]          # the RUN line's estimate, read from the pinned settings file
nvidia_smi_stub = _stubs.nvidia_smi_stub
GATED_112 = nosub_line(112, "none", "kit", "none", "skipped", "gated")
GATED_FN_112 = nosub_fn_line(112, "skipped", "gated")


# ============================================================ design / check: subprocess end to end on the stub stack
@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf(_stubs.bindcraft_stack_missing(), _stubs.SKIP_BINDCRAFT)
class TestCliProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_cli_")
        cls.site = _stubs.make_stub_stack(cls.tmp)
        cls.target = _stubs.write_target(os.path.join(cls.tmp, "t.pdb"), ("A",), 108)
        cls.params = _stubs.make_params(os.path.join(cls.tmp, "params"))
        cls.env = _stubs.child_env(cls.site, PATH=nvidia_smi_stub(cls.tmp) + os.pathsep + os.environ["PATH"], COLABDESIGN_OPT_HOME=_stubs.TREE, XDG_CACHE_HOME=os.path.join(cls.tmp, "xdg"))   # the design start's weights digest memo lands under the test's tmp, never the real ~/.cache

    def cli(self, *args, **env):
        r = subprocess.run([sys.executable, "-m", "colabdesign_opt", *args], env=dict(self.env, **env), capture_output=True, text=True, cwd=self.tmp)
        return r.returncode, [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt]")], r

    def design(self, mode, out, *extra, binder_len="100", **env):
        return self.cli("design", "--mode", mode, *extra, "--starting-pdb", self.target, "--chains", "A", "--binder-len", binder_len, "--seed", "0", "--out", out, "--params-dir", self.params, **env)

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_off(self):
        out = os.path.join(self.tmp, "off")
        rc, lines, r = self.design("off", out, AF2M_LEVERS="nosub", COLABDESIGN_OPT="off")
        self.assertEqual(rc, 0, r.stderr[-2000:])
        weights = [l for l in lines if l.startswith("[colabdesign-opt] weights=")]              # the stand-in params are not the pinned weights: each named NOT PINNED on the design's line, and the design proceeds (warn-and-run)
        self.assertEqual(len(weights), 5, lines); self.assertTrue(all(l.endswith("NOT PINNED — this tree was tested with the pinned weights only") for l in weights), weights)
        self.assertEqual(lines.index(weights[0]), 2, lines); lines = [l for l in lines if l not in weights]
        self.assertTrue(lines[0].startswith("[colabdesign-opt] ACTIVE mode=off route=subprocess tier=none class=stock colabdesign=1.1.3@e31a56fe jax=0.6.0 haiku=0.0.17 gpu=NVIDIA-H100-80GB-HBM3(sm90,81559MiB) levers=none skipped=none arm=stock settings=default_4stage_multimer line=stock"), lines[0])
        self.assertEqual(lines[1], f"[colabdesign-opt] RUN mode=off target={self.target} chains=A binder_len=100 tokens=208 hotspot=none seed=0 steps<={STEPS} greedy_rounds={GREEDY}")
        self.assertTrue(lines[2].startswith("[colabdesign-opt] SETTINGS default_4stage_multimer file=stock/src/bindcraft/settings_advanced/default_4stage_multimer.json sha256="), lines[2])
        self.assertTrue(lines[3].startswith(f"[colabdesign-opt] [run] design_l100_s0: tokens 208 steps {STEPS} (soft "), lines[3])
        self.assertEqual(lines[4], f"[colabdesign-opt] EXIT rc=0 mode=off levers=none evidence=n/a out={out}")
        self.assertNotIn("[colabdesign-opt] LEVER", r.stderr + r.stdout)
        self.assertEqual({"design.fasta", "design.pdb", "run.log", "trajectory.jsonl", bindcraft.FAILURE_CSV}, _files(out), sorted(os.listdir(out)))   # the outputs, the arm's log, BindCraft's csv (its directories beside): no record file
        self.assertIn("[colabdesign-opt stock] ENV-CLEAN ok", open(os.path.join(out, "run.log")).read())

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_given_settings_files_reach_the_arm(self):
        """`design --advanced FILE --filters FILE` (bindcraft.py's own flags): the arm designs at the given files — its SETTINGS line names the
        advanced file (name = the file's stem, file = its absolute path) and BindCraft's failure census takes its columns from the filters file;
        without them the arm runs BindCraft's defaults; a file that does not exist is refused by name (exit 3) before any arm starts."""
        adv = os.path.abspath(os.path.join(self.tmp, "my_4stage.json")); flt = os.path.abspath(os.path.join(self.tmp, "my_filters.json"))
        shutil.copy(settings.path(_stubs.TREE), adv)
        filters = json.load(open(settings.filters_path(_stubs.TREE))); filters["Average_KitGivenFilter"] = {"threshold": None, "higher": False}
        json.dump(filters, open(flt, "w"))
        out = os.path.abspath(os.path.join(self.tmp, "off_given"))
        rc, lines, r = self.design("off", out, "--advanced", adv, "--filters", flt)
        self.assertEqual(rc, 0, r.stderr[-2000:])
        st = [l for l in lines if l.startswith("[colabdesign-opt] SETTINGS ")]
        self.assertEqual(len(st), 1, lines); self.assertTrue(st[0].startswith(f"[colabdesign-opt] SETTINGS my_4stage file={adv} sha256={names.sha256_file(adv)[:16]} "), st[0])   # printed BY THE ARM at its own settings
        header = open(os.path.join(out, bindcraft.FAILURE_CSV)).readline()
        self.assertIn("KitGivenFilter", header)                                                     # generate_filter_pass_csv named its columns from the given filters file
        rc, lines, r = self.design("off", os.path.join(self.tmp, "off_default2"))
        self.assertEqual(rc, 0, r.stderr[-2000:])
        self.assertNotIn("KitGivenFilter", open(os.path.join(self.tmp, "off_default2", bindcraft.FAILURE_CSV)).readline())
        self.assertTrue(any(l.startswith(f"[colabdesign-opt] SETTINGS {settings.NAME} file={settings.SETTINGS_RELPATH} ") for l in lines), lines)
        rc, lines, r = self.design("off", os.path.join(self.tmp, "off_absent"), "--advanced", os.path.join(self.tmp, "absent.json"))
        self.assertEqual(rc, 3, r.stderr[-800:]); self.assertIn("--advanced", r.stderr); self.assertIn("no such file", r.stderr)

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_fast_on_the_stub_stack(self):
        """`fast` = every lever: the ACTIVE line names the whole set and class precision; the stub stack has no Pallas kernel / GPU backend and
        the stand-in design surface is not ColabDesign's pinned loop, so in the arm those levers STEP ASIDE BY NAME at install (`LEVER …
        state=skipped reason=cannot_run detail=…`; proj / txla `no_attention_kernel`) and the design runs with the rest — the kit rule,
        never a refusal and never silent: `design` names each again (`skipped: <lever>: <why>`), EVIDENCE carries `skipped=`, exit 0."""
        out = os.path.join(self.tmp, "fast")
        rc, lines, r = self.design("fast", out)
        self.assertTrue(lines[0].startswith("[colabdesign-opt] ACTIVE mode=fast route=subprocess tier=2 class=precision "), lines[0])
        self.assertIn(" levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla skipped=none arm=kit settings=default_4stage_multimer line=every lever: the exact-class ones + nosub/nosub_fn + the kit's Pallas kernels (flash attention on every eligible attention call, fused triangle multiplication on every call): never bitwise vs stock", lines[0])   # the parent's dry run (a card visible): the set; the arm's lines are the record of what stepped aside
        self.assertEqual(rc, 0, r.stderr[-2500:]); self.assertNotIn("REFUSED", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertRegex(r.stderr, r"\[colabdesign-opt\] LEVER name=hoist_prev state=skipped reason=cannot_run impl=hoist_prev@kit origin=kit detail=HoistPrevError:\S+defines_no_`_recycle`:\S+ source=install pid=\d+")
        self.assertIn("[colabdesign-opt] skipped: hoist_prev: cannot_run: HoistPrevError:", r.stderr); self.assertIn("[colabdesign-opt] skipped: proj: no_attention_kernel: ", r.stderr)
        self.assertIn("[colabdesign-opt] LEVER name=compilecache state=on ", r.stderr)                 # the levers that can run here engaged
        ev = [l for l in lines if l.startswith("[colabdesign-opt] EVIDENCE ")][0]
        self.assertIn(" missing=none skipped=hoist_prev,trimul,triatt,", ev); self.assertTrue(ev.startswith("[colabdesign-opt] EVIDENCE levers=compilecache,lowercache,parcompile "), ev)
        self.assertTrue(lines[-1].startswith("[colabdesign-opt] EXIT rc=0 mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla evidence=applied=compilecache,lowercache,parcompile"), lines[-1])
        self.assertIn("run.log", os.listdir(out)); self.assertIn("design.pdb", os.listdir(out))     # the design ran

    def test_exact_dry_run(self):
        """`exact` = the exact-class levers (compilecache + hoist_prev), tier 1, class 1: `check --mode exact` names them on the ACTIVE line; an unknown
        word is refused before any arm starts (NOT ACTIVE, exit 3) with the one unknown-mode text naming both bases."""
        r = subprocess.run([sys.executable, "-m", "colabdesign_opt", "check", "--mode", "exact"], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("[colabdesign-opt] ACTIVE mode=exact route=check tier=1 class=exact ", r.stderr); self.assertIn(" levers=compilecache+lowercache+parcompile+hoist_prev skipped=none arm=kit settings=default_4stage_multimer line=identical outputs to stock, from the kit's process ", r.stderr)
        rc, lines, r = self.design("turbo", os.path.join(self.tmp, "turbo"))
        self.assertEqual(rc, 3, r.stderr[-1500:])
        self.assertIn("[colabdesign-opt] NOT ACTIVE reason=unknown mode 'turbo' (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "turbo", "run.log")))

    def test_refusals(self):
        rc, lines, r = self.design("turbo", os.path.join(self.tmp, "x"))
        self.assertEqual(rc, 3); self.assertIn("NOT ACTIVE reason=", lines[0]); self.assertIn("turbo", lines[0])
        rc, lines, r = self.design("fast", os.path.join(self.tmp, "x"), "--levers", "nosub")
        self.assertEqual(rc, 2)                                                                     # no lever axis: a mode is its lever set
        rc, lines, _ = self.design("fast", os.path.join(self.tmp, "x"), COLABDESIGN_OPT="off")
        self.assertEqual(rc, 2); self.assertIn("disagrees with COLABDESIGN_OPT=off", lines[0])
        rc, lines, _ = self.cli("design", "--mode", "off", "--starting-pdb", self.target, "--chains", "A", "--binder-len", "100", "--out", os.path.join(self.tmp, "x"))
        self.assertEqual(rc, 2); self.assertIn("no params/ directory", lines[0])                         # BindCraft's default root (stock/src/bindcraft) holds no params in the tree
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "x", "run.log")))

    def test_check(self):
        rc, lines, r = self.cli("check", "--mode", "fast", "--json")
        self.assertEqual(rc, 0, r.stderr[-1500:]); self.assertIn("ACTIVE mode=fast route=check tier=2 class=precision", lines[0]); self.assertIn("levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla skipped=none arm=kit", lines[0])
        rep = json.loads(r.stdout)
        self.assertEqual((rep["active"], rep["levers"], rep["numerics_class"], rep["arm"]), (True, ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], "precision", "kit"))
        self.assertTrue(rep["core"]["ok"]); self.assertTrue(rep["card"]["ok"]); self.assertNotIn("levers_installed", rep)     # a dry run installs nothing
        rc, lines, r = self.cli("check", "--json", COLABDESIGN_OPT="exact")                          # the exact tier through the variable: tier 1, class 1, its two levers
        self.assertEqual(rc, 0, r.stderr[-1500:]); self.assertIn("ACTIVE mode=exact route=check tier=1 class=exact", lines[0]); self.assertIn("levers=compilecache+lowercache+parcompile+hoist_prev skipped=none arm=kit", lines[0])
        rep = json.loads(r.stdout)
        self.assertEqual((rep["active"], rep["mode"], rep["levers"], rep["tier"], rep["numerics_class"]), (True, "exact", ["compilecache", "lowercache", "parcompile", "hoist_prev"], 1, "exact"))
        rc, lines, r = self.cli("check", COLABDESIGN_OPT="turbo")                                    # an unknown word: exit 3, the one text
        self.assertEqual(rc, 3); self.assertIn("NOT ACTIVE reason=unknown mode 'turbo' (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>[-no-<lever>...] without the named levers of fast's compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla)", r.stderr)

    @unittest.skipIf(_stubs.host_nvidia_smi(), "a real nvidia-smi is on PATH: the no-GPU cases need a host without one")
    def test_check_without_a_gpu(self):
        rc, lines, _ = self.cli("check", "--mode", "fast", PATH=os.environ["PATH"])              # no nvidia-smi stand-in: no GPU — the levers that need the GPU backend will step aside by name in the arm: foretold, the mode is served
        self.assertEqual(rc, 0, lines); self.assertIn("ACTIVE mode=fast route=check tier=2 class=precision", lines[0])
        self.assertIn(" levers=parcompile+hoist_prev+nosub+nosub_fn+opm_fold skipped=compilecache,trimul,triatt,ln,proj,transition,txla arm=kit ", lines[0]); self.assertIn(" gpu=none ", lines[0])
        rc, lines, _ = self.cli("check", "--mode", "off", PATH=os.environ["PATH"])
        self.assertEqual(rc, 0)                                                                    # stock needs no GPU gate


def _files(d):
    """The regular files directly under d (BindCraft's own directories beside them are not listed)."""
    return {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestCliInProcess(unittest.TestCase):
    """The arm stubbed at `driver.launch`: the package's handling of what the arm printed and wrote."""

    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="cd_opt_cli_inproc_")
        self.site = _stubs.make_stub_stack(self.tmp); sys.path.insert(0, self.site)
        for k in ("COLABDESIGN_OPT", "AF2M_LEVERS", "COLABDESIGN_PARAMS_DIR"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("COLABDESIGN_OPT_HOME", _stubs.TREE); self.mp.setenv("XDG_CACHE_HOME", os.path.join(self.tmp, "xdg"))   # no digest memo under the real ~/.cache
        self.mp.setattr(stack, "nvidia_smi_probe", lambda: dict(_stubs.GPU))
        self.target = _stubs.write_target(os.path.join(self.tmp, "t.pdb"), ("A",), 108)
        self.params = _stubs.make_params(os.path.join(self.tmp, "params"))

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def fake_arm(self, lines, rc=0, write_outputs=True):
        """driver.launch replaced: the arm's lines (stamped 0.0) and its outputs (or none)."""
        def launch(argv, env, *, cwd, log_path, timeout, echo=None):
            out = cwd
            if write_outputs:
                for n in ("design.pdb", "design.fasta", "trajectory.jsonl"):
                    open(os.path.join(out, n), "w").write(n + "\n")
            open(log_path, "w").write("\n".join(lines) + "\n")
            return {"exit_code": rc, "timed_out": False, "wall_s": 0.1, "lines": list(lines), "stamps": [0.0] * len(lines), "log": log_path}
        self.mp.setattr(driver, "launch", launch)

    def run_cli(self, mode, out, *extra):
        """(rc, the package's lines, design()'s in-memory record) for one in-process `design`."""
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            rec = cli.design_argv(["--mode", mode, *extra, "--starting-pdb", self.target, "--chains", "A", "--binder-len", "100", "--seed", "0", "--out", out, "--params-dir", self.params])
        self.rec = rec
        self.assertEqual(sorted(f for f in (os.listdir(out) if os.path.isdir(out) else []) if f.endswith(".json")), [])   # no record file under --out, ever
        return rec["exit_code"], [l for l in buf.getvalue().splitlines() if l.startswith("[colabdesign-opt]")]

    def verdict(self):
        """The last run's (exit_code, partial levers, gated levers, missing outputs) from design()'s record."""
        r = self.rec
        return {"exit_code": r["exit_code"], "partial": (r.get("verdict") or {}).get("partial") or {}, "gated": (r.get("verdict") or {}).get("gated") or {},
                "incomplete": outputs.complete(r.get("files") or {}) if r.get("files") else list(names.OUTPUTS), "evidence": r.get("evidence") or {}}

    def test_stock_arm_printing_a_lever_line_is_a_defect(self):
        self.fake_arm(["[colabdesign-opt stock] ENV-CLEAN ok", GATED])
        out = os.path.join(self.tmp, "o1")
        rc, lines = self.run_cli("off", out)
        self.assertEqual(rc, 3); self.assertTrue(any("NOT ACTIVE reason=the stock arm printed 1 lever line(s): " + GATED[:60] in l for l in lines), lines)
        self.assertEqual(self.verdict()["exit_code"], 3)

    def test_missing_lever_is_not_active(self):
        self.fake_arm(["[colabdesign-opt kit] ENV-CLEAN ok"])
        out = os.path.join(self.tmp, "o2")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 3); self.assertTrue(any("EVIDENCE levers=none gated=none fallback=none missing=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,triatt,opm_fold,ln,proj,transition,txla " in l for l in lines), lines)
        m = self.verdict()
        self.assertTrue(m["partial"]["nosub"].startswith("missing: no line of its own"), m["partial"])
        self.assertIn(f"{names.PREFIX} {report.partial_line(', '.join(k + ': ' + v for k, v in m['partial'].items()))}", lines); self.assertEqual(list(m["partial"]), ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"])
        self.assertEqual((m["evidence"]["state"], list(m["partial"]), m["exit_code"]), ({"compilecache": "missing", "lowercache": "missing", "parcompile": "missing", "hoist_prev": "missing", "nosub": "missing", "nosub_fn": "missing", "trimul": "missing", "triatt": "missing", "opm_fold": "missing", "ln": "missing", "proj": "missing", "transition": "missing", "txla": "missing"}, ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], 3))

    def test_fallback_is_partial_not_active(self):
        """A nosub line that is not the shipped decision (the lever has no fallback: such a line is a defect) is a partial activation: exit 3
        by name, the record's `partial` naming the lever and the reason."""
        self.fake_arm([*EXACT_OK, DEFECT, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK])
        out = os.path.join(self.tmp, "o3")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 3); self.assertTrue(any("fallback=nosub" in l for l in lines))
        self.assertTrue(any(l.startswith(f"{names.PREFIX} NOT ACTIVE: partial activation — nosub: fallback: grad_subbatch=4 source=kit") and l.endswith(report.PARTIAL_TAIL) for l in lines), lines)
        self.assertIn("EXIT rc=3", lines[-1]); self.assertIn("evidence=applied=compilecache,lowercache,parcompile,hoist_prev,nosub_fn,trimul,triatt,opm_fold,ln,proj,transition,txla;fallback=nosub;missing=none", lines[-1])
        m = self.verdict()
        self.assertEqual((m["exit_code"], list(m["partial"]), m["gated"], m["incomplete"]), (3, ["nosub"], {}, []))
        self.assertNotIn("--allow-partial", " ".join(lines)); self.assertFalse([a for a in ("allow_partial",) if a in self.rec])   # no opt-out: an installed lever that did not engage is a defect

    def test_partial_line_is_the_family_grammar(self):
        """The partial-exit line is one fixed grammar, printed through report.partial_line by every entry point: the fixed parts
        are byte-literal (the head is the tree's `NOT ACTIVE: partial activation — `), <detail> is this package's `lever: reason` list; no opt-out line exists."""
        self.assertEqual(report.partial_line("nosub: fallback: x"), "NOT ACTIVE: partial activation — nosub: fallback: x; exit 3 (an installed lever that did not engage is a defect, named - a lever that cannot run here steps aside at install instead, state=skipped, exit 0)")
        self.assertFalse(hasattr(report, "partial_allowed_line"))
        self.fake_arm([*EXACT_OK, DEFECT, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK])
        out = os.path.join(self.tmp, "o3d")
        rc, lines = self.run_cli("fast", out)
        printed = [l for l in lines if "NOT ACTIVE" in l]
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertEqual(len(printed), 1)
        self.assertEqual(printed[0], f"{names.PREFIX} {report.partial_line('nosub: ' + self.verdict()['partial']['nosub'])}")

    def test_partial_with_a_failed_arm_keeps_its_code(self):
        self.fake_arm([*EXACT_OK, DEFECT, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK], rc=7)
        out = os.path.join(self.tmp, "o3f")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 7); self.assertTrue(any("PARTIAL: nosub: fallback" in l and "the arm failed, rc=7" in l for l in lines))
        self.assertEqual((self.verdict()["exit_code"], list(self.verdict()["partial"])), (7, ["nosub"]))

    def test_gated_below_the_size_gate_is_exit_0(self):
        """nosub at or below 384 tokens: `gated`, recorded by name, exit 0 — not partial (stock's programs by construction)."""
        self.fake_arm([*EXACT_OK, GATED, GATED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_SMALL, TXLA_GATED])
        out = os.path.join(self.tmp, "o3g")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 0); self.assertTrue(any(l.startswith("[colabdesign-opt] gated: nosub: tokens at or below the size gate") for l in lines), lines)
        m = self.verdict()
        self.assertEqual((m["partial"], list(m["gated"]), m["evidence"]["state"]), ({}, ["nosub", "nosub_fn"], {"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied", "nosub": "gated", "nosub_fn": "gated", "trimul": "applied", "triatt": "applied", "opm_fold": "applied", "ln": "applied", "proj": "applied", "transition": "applied", "txla": "applied"}))

    def test_attention_kernel_states_in_process(self):
        self.fake_arm([*EXACT_OK, APPLIED, APPLIED_FN, TRIMUL_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK, PALLAS_REPLACED, triatt_line(0, {})])
        out = os.path.join(self.tmp, "o3h")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 3)
        self.assertIn(f"{names.PREFIX} {report.partial_line('triatt: missing: installed, the kernels served no call (served=0)')}", lines)
        self.assertEqual(list(self.verdict()["partial"]), ["triatt"])
        self.fake_arm([*EXACT_OK, APPLIED, APPLIED_FN, TRIMUL_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK, PALLAS_REPLACED, triatt_line(96, {"below_keys_rule": 8})])
        out = os.path.join(self.tmp, "o3i")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 0, lines)
        m = self.verdict()
        self.assertEqual((m["evidence"]["state"], m["evidence"]["triatt_served"], m["evidence"]["triatt_fallback_by"], m["partial"]), ({"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied", "nosub": "applied", "nosub_fn": "applied", "trimul": "applied", "triatt": "applied", "opm_fold": "applied", "ln": "applied", "proj": "applied", "transition": "applied", "txla": "applied"}, 96, {"below_keys_rule": 8}, {}))
        self.assertTrue(any("EVIDENCE levers=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,triatt,opm_fold,ln,proj,transition,txla gated=none fallback=none missing=none skipped=none grad_subbatch=none fn_subbatch=none compile_cache=0/104 compile_threads=16 prev_on_device=141 trimul_served=12 trimul_fallback_by=none triatt_served=96 triatt_fallback_by=below_keys_rule:8 opm_fold_served=52 opm_fold_fallback_by=none ln_served=830 ln_fallback_by=channels_not_pow2:96 proj_served=88 proj_fallback_by=head_dim_lt_16:8 transition_served=5 transition_fallback_by=none txla_served=8 txla_fallback_by=small_call:1 pallas_served=n/a pallas_fallback=n/a pallas_fallback_by=none" in l for l in lines), lines)
        self.fake_arm([*EXACT_OK, APPLIED, APPLIED_FN, TRIMUL_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK, PALLAS_REPLACED, triatt_line(95, {"below_keys_rule": 8, "backend_not_gpu": 1})])
        out = os.path.join(self.tmp, "o3j")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual((rc, list(self.verdict()["partial"])), (3, ["triatt"]))                  # an undeclared fallback reason is partial

    def test_incomplete_outputs_fail(self):
        """A short file set is `incomplete` — EXIT_FAIL with the missing names on a line, never `partial`."""
        self.fake_arm([*EXACT_OK, APPLIED, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK], write_outputs=False)
        out = os.path.join(self.tmp, "o4")
        rc, lines = self.run_cli("fast", out)
        self.assertEqual(rc, 1); self.assertTrue(any("design incomplete: missing design.pdb" in l for l in lines))
        m = self.verdict()
        self.assertEqual((m["exit_code"], m["incomplete"][:1], m["partial"]), (1, ["design.pdb"], {}))
        # both states at once: partial precedes incomplete (3, both recorded)
        self.fake_arm([*EXACT_OK, DEFECT, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK], write_outputs=False)
        out = os.path.join(self.tmp, "o5")
        rc, lines = self.run_cli("fast", out)
        m = self.verdict()
        self.assertEqual((rc, m["exit_code"], list(m["partial"]), m["incomplete"][:1]), (3, 3, ["nosub"], ["design.pdb"]))

    def test_arm_exit_code_propagates(self):
        self.fake_arm([*EXACT_OK, GATED, GATED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_SMALL, TXLA_GATED], rc=7)
        rc, lines = self.run_cli("fast", os.path.join(self.tmp, "o7"))
        self.assertEqual(rc, 7); self.assertTrue(lines[-1].startswith("[colabdesign-opt] EXIT rc=7"))


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestCoreGateAtEntries(unittest.TestCase):
    """The core pin gate (colabdesign_opt/_core_gate.py, the tree's kit_template copy) is statement one of every entry — `python -m
    colabdesign_opt <verb>` (= the console script; run.sh execs it after the same probe), `colabdesign_opt.enable()` / `status()`, the arm
    launchers, the autoload trigger — BEFORE any opt_core import: an ABSENT core and a STALE core are one `[colabdesign-opt] NOT ACTIVE: reason=…`
    line and exit 3, never a traceback; the exit code equals the CLI's and the report's."""
    GATE = "[colabdesign-opt] NOT ACTIVE: reason="

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cd_opt_gate_")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("COLABDESIGN_OPT", "AF2M_", "AF_PALLAS_"))}

    def shadow_core(self, version="0.0.1"):
        d = os.path.join(self.tmp, "shadow"); os.makedirs(os.path.join(d, "opt_core"), exist_ok=True)
        open(os.path.join(d, "opt_core", "__init__.py"), "w").write(f'__version__ = "{version}"\nraise ImportError("the shadow core must never be imported: the gate locates, it does not import")\n')
        return d

    def test_exit_codes_agree(self):
        from colabdesign_opt import __main__ as entry, _core_gate
        self.assertEqual(entry.EXIT_INACTIVE, report.EXIT_NOT_ACTIVE); self.assertEqual(_core_gate.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE)

    def test_gate_copy_is_the_template(self):
        tpl = os.path.join(_stubs.TREE, "..", "common", "opt_core", "kit_template", "_core_gate.py")
        if not os.path.isfile(tpl):
            self.skipTest("the core's kit_template is not beside this tree")
        mine = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_core_gate.py")
        self.assertEqual(open(tpl, "rb").read(), open(mine, "rb").read())

    def test_core_absent(self):
        env = dict(self.env, PYTHONPATH=_stubs.OPT_DIR)                                           # nothing but the package on the path; with -S no opt_core is importable anywhere
        for argv in (["-S", "-m", "colabdesign_opt", "check", "--mode", "fast"], ["-S", "-m", "colabdesign_opt", "design", "--mode", "off", "--starting-pdb", "t.pdb", "--chains", "A", "--binder-len", "10", "--out", os.path.join(self.tmp, "o")],
                     ["-S", "-c", "import colabdesign_opt; colabdesign_opt.enable('fast')"], ["-S", "-c", "import colabdesign_opt; colabdesign_opt.status()"],
                     ["-S", "-s", "-m", "colabdesign_opt.kit_launch", "--pins", _stubs.PINS, "--mode", "fast", "--", "--out", os.path.join(self.tmp, "o2")],
                     ["-S", "-s", "-m", "colabdesign_opt.stock_launch", "--pins", _stubs.PINS, "--", "--out", os.path.join(self.tmp, "o3")]):
            r = subprocess.run([sys.executable, *argv], env=env, capture_output=True, text=True, cwd=self.tmp)
            self.assertEqual(r.returncode, 3, (argv, r.stderr[-600:])); self.assertIn(self.GATE + "core_missing:opt_core (pinned >= v", r.stderr); self.assertNotIn("Traceback", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))

    def test_core_stale(self):
        env = dict(self.env, PYTHONPATH=self.shadow_core() + os.pathsep + _stubs.OPT_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
        for argv in (["-m", "colabdesign_opt", "check", "--mode", "fast"], ["-c", "import colabdesign_opt; colabdesign_opt.enable('fast')"],
                     ["-s", "-m", "colabdesign_opt.kit_launch", "--pins", _stubs.PINS, "--mode", "fast", "--", "--out", os.path.join(self.tmp, "o4")]):
            r = subprocess.run([sys.executable, *argv], env=env, capture_output=True, text=True, cwd=self.tmp)
            self.assertEqual(r.returncode, 3, (argv, r.stderr[-600:])); self.assertIn(self.GATE + "core_mismatch: opt_core pinned ", r.stderr); self.assertIn("installed v0.0.1 at ", r.stderr); self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run(["bash", os.path.join(_stubs.TREE, "run.sh"), "check", "--config", "h100", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr[-800:]); self.assertIn(self.GATE + "core_mismatch: opt_core pinned ", r.stderr); self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr)                              # the config's probe refuses; run.sh stops there



@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestWeightsWarnAndRun(unittest.TestCase):
    """Warn-and-run by sha256: cli.weights_words = stock/check_pins.py's one wording; unknown checkpoints are named, never refused; the design start's
    digest memo hashes an unchanged file once (keyed by path, size, mtime_ns) and re-hashes on any change — never size-only."""
    def _params(self, tmp, payload=b"not the pinned weights"):
        d = os.path.join(tmp, "params"); os.makedirs(d)
        for i in range(1, 6):
            with open(os.path.join(d, f"params_model_{i}_multimer_v3.npz"), "wb") as fh:
                fh.write(payload + b" %d" % i)
        return d

    def test_unknown_checkpoint_is_named_not_pinned_by_sha256_and_not_refused(self):
        tmp = tempfile.mkdtemp(prefix="cd_opt_w_")
        try:
            d = self._params(tmp)
            cert, unc = cli.weights_words(tmp, cached=False)
            self.assertEqual((len(cert), len(unc)), (0, 5))
            self.assertTrue(all(l.startswith("weights=params_model_") and " sha256=" in l and l.endswith("NOT PINNED — this tree was tested with the pinned weights only") for l in unc), unc)
            os.remove(os.path.join(d, "params_model_5_multimer_v3.npz"))                   # an ABSENT pinned file: named on its own line (the design refuses it by name — bindcraft.refusals), no exception here
            unc = cli.weights_words(tmp, cached=False)[1]
            self.assertEqual(len(unc), 5); self.assertEqual(unc[-1], "weights=params_model_5_multimer_v3.npz ABSENT — pinned params file not found"); self.assertEqual(sum("NOT PINNED" in l for l in unc), 4)
        finally:
            import shutil; shutil.rmtree(tmp)

    def test_digest_memo_hashes_once_and_rehashes_on_change_never_size_only(self):
        tmp = tempfile.mkdtemp(prefix="cd_opt_wc_")
        try:
            d = self._params(tmp); m, pins = cli.check_pins_module(); cache = os.path.join(tmp, "memo.json"); calls = []
            real = m.sha256_file; m.sha256_file = lambda p: (calls.append(p), real(p))[1]
            _bad, first = m.check_weights(pins, tmp, cache=cache); self.assertEqual(len(calls), 5)
            _bad, second = m.check_weights(pins, tmp, cache=cache); self.assertEqual(len(calls), 5)          # unchanged files: memo hits, no re-hash
            self.assertEqual(first, second); self.assertEqual(first, m.check_weights(pins, tmp)[1])          # memo == the authoritative digest (no memo: 5 more digests)
            self.assertEqual(len(calls), 10); p1 = os.path.join(d, "params_model_1_multimer_v3.npz")
            with open(p1, "wb") as fh:                                                                   # SAME size, other bytes, new mtime: re-hashed (one digest) and its digest moves
                fh.write(b"NOT the pinned weights 1")
            st = os.stat(p1); os.utime(p1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            _bad, third = m.check_weights(pins, tmp, cache=cache); self.assertEqual(len(calls), 11)
            self.assertNotEqual(third["params_model_1_multimer_v3.npz"]["sha256"], first["params_model_1_multimer_v3.npz"]["sha256"])
            self.assertEqual(third["params_model_1_multimer_v3.npz"]["bytes"], first["params_model_1_multimer_v3.npz"]["bytes"])
            _bad, ro = m.check_weights(pins, tmp, cache=os.path.join(tmp, "no", "such", "dir", os.devnull, "memo.json")); self.assertEqual(ro, third)   # an unwritable memo still digests
        finally:
            import shutil; shutil.rmtree(tmp)


# ============================================================ warm: BindCraft's bundled example, then one design
EXAMPLE_TOKENS = 115 + 65                                                        # example/PDL1.pdb chain A: 115 residues with a CA atom; settings_target/PDL1.json lengths[0] = 65


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestExampleCase(unittest.TestCase):
    def test_bindcrafts_own_example_as_its_settings_file_names_it(self):
        case = warm.example_case(_stubs.TREE)
        self.assertEqual((case["name"], case["chains"], case["hotspot"], case["binder_len"], case["settings"]), ("PDL1", "A", "56", 65, "stock/src/bindcraft/settings_target/PDL1.json"))
        self.assertEqual(case["target"], os.path.join(_stubs.TREE, "stock", "src", "bindcraft", "example", "PDL1.pdb")); self.assertTrue(os.path.isfile(case["target"]))
        want = json.load(open(os.path.join(_stubs.TREE, names.EXAMPLE_SETTINGS)))              # nothing typed in the kit: every value is the vendored file's
        self.assertEqual((case["name"], case["chains"], case["hotspot"], case["binder_len"], os.path.basename(case["target"])),
                         (want["binder_name"], want["chains"], want["target_hotspot_residues"], want["lengths"][0], os.path.basename(want["starting_pdb"])))
        self.assertEqual(inputs.target(case["target"], case["chains"]).target_res + case["binder_len"], EXAMPLE_TOKENS)

    def test_a_tree_without_the_example_is_named(self):
        tmp = tempfile.mkdtemp(prefix="cd_opt_example_")
        with self.assertRaises(warm.WarmError):
            warm.example_case(tmp)                                                              # no settings file
        d = os.path.join(tmp, os.path.dirname(names.EXAMPLE_SETTINGS)); os.makedirs(d)
        json.dump({"binder_name": "X", "starting_pdb": "/elsewhere/X.pdb", "chains": "A", "target_hotspot_residues": "", "lengths": [70, 150]}, open(os.path.join(tmp, names.EXAMPLE_SETTINGS), "w"))
        with self.assertRaises(warm.WarmError) as cm:
            warm.example_case(tmp)                                                              # settings present, target absent
        self.assertIn("X.pdb is absent", str(cm.exception))
        os.makedirs(os.path.join(tmp, names.EXAMPLE_DIR)); open(os.path.join(tmp, names.EXAMPLE_DIR, "X.pdb"), "w").write("END\n")
        case = warm.example_case(tmp)
        self.assertEqual((case["name"], case["hotspot"], case["binder_len"]), ("X", None, 70))         # an empty hotspot string = none; the first length


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestWarmInProcess(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="cd_opt_warm_")
        self.site = _stubs.make_stub_stack(self.tmp); sys.path.insert(0, self.site)
        for k in ("COLABDESIGN_OPT", "AF2M_LEVERS", "COLABDESIGN_PARAMS_DIR"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("COLABDESIGN_OPT_HOME", _stubs.TREE)
        self.mp.setattr(stack, "nvidia_smi_probe", lambda: dict(_stubs.GPU))
        self.params = _stubs.make_params(os.path.join(self.tmp, "params"))
        self.launches = []

        def launch(argv, env, *, cwd, log_path, timeout, echo=None):
            """The arm replaced: its outputs, its lever lines, then what the design script prints — the SETTINGS line as the design call starts
            (arriving 32.0 s after the launch) and the [run] line as it ends (report.run_summary_line of a run record)."""
            self.launches.append(argv)
            for n in ("design.pdb", "design.fasta", "trajectory.jsonl"):
                open(os.path.join(cwd, n), "w").write(n + "\n")
            seed = argv[argv.index("--seed") + 1]; tag = f"{argv[argv.index('--binder-name') + 1]}_l{argv[argv.index('--binder-len') + 1]}_s{seed}"
            run = {"tokens": 112, "steps": 210, "n_steps": {"soft": 75, "temp": 45, "hard": 5, "greedy_rounds": 15, "greedy_forward": 16}, "terminate": {"verdict": "", "gate": None},
                   "final": {"loss": 1.0}, "files": {"design.pdb": {"sha256": "ab" * 32}},
                   "timing": {"first_calls_s": [40.0, 30.0, 20.0], "phase_steady_s": {"soft": 0.42, "temp": 0.42, "hard": 0.43, "greedy_forward": 0.2}, "steady_s": 0.42, "total_s": 180.0}}
            settings_line = f"{names.PREFIX} {report.settings_line({'name': settings.NAME, 'file': settings.SETTINGS_RELPATH, 'sha256': '0' * 64}, {'design_models': [0, 1, 2, 3, 4], 'helicity_value': -0.3, 'design_name': tag}, None)}"
            lines = ([*EXACT_OK, GATED_112, GATED_FN_112, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_SMALL, TXLA_GATED] if "colabdesign_opt.kit_launch" in argv else ["[colabdesign-opt stock] ENV-CLEAN ok"]) + [settings_line, "Stage 1: Test Logits", f"{names.PREFIX} {report.run_summary_line(tag, run)}"]
            stamps = [0.5] * (len(lines) - 3) + [32.0, 33.0, 212.5]
            open(log_path, "w").write("\n".join(lines) + "\n")
            return {"exit_code": 0, "timed_out": False, "wall_s": 213.0, "lines": lines, "stamps": stamps, "log": log_path}
        self.mp.setattr(driver, "launch", launch)

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def test_example_case_fast(self):
        lines = []
        res = warm.run("fast", os.path.join(self.tmp, "w"), self.params, log=lines.append)
        self.assertEqual((res["status"], res["exit_code"], res["mode"], res["levers"], res["tokens"]), ("PASS", 0, "fast", ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], 112))
        self.assertEqual(res["timing"], {"ready_s": 32.0, "first_calls_s": [40.0, 30.0, 20.0], "phase_steady_s": {"soft": 0.42, "temp": 0.42, "hard": 0.43, "greedy_forward": 0.2}, "steady_s": 0.42, "total_s": 180.0})   # the [run] line read back; ready = the SETTINGS line's arrival
        self.assertEqual((res["steps"], res["files"]), (210, {n: names.sha256_file(os.path.join(res["out_dir"], n)) for n in names.OUTPUTS}))   # files = the design command's own digests of --out
        self.assertEqual(res["evidence"], {"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied", "nosub": "gated", "nosub_fn": "gated", "trimul": "applied", "triatt": "applied", "opm_fold": "applied", "ln": "applied", "proj": "applied", "transition": "applied", "txla": "applied"}); self.assertEqual((res["input"]["binder_len"], res["input"]["chains"], res["input"]["hotspot"]), (65, "A", "56"))
        self.assertEqual(res["input"]["target"], os.path.join(_stubs.TREE, "stock", "src", "bindcraft", "example", "PDL1.pdb")); self.assertEqual(res["input"]["case"]["name"], "PDL1")
        self.assertEqual(res["out_dir"], os.path.join(self.tmp, "w", "warm"))
        self.assertEqual(os.listdir(os.path.join(self.tmp, "w")), ["warm"]); self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "w", "warm"))), ["design.fasta", "design.pdb", "run.log", "trajectory.jsonl"])   # nothing but the outputs and the log
        self.assertEqual(lines[0], "warm case: BindCraft's example (stock/src/bindcraft/settings_target/PDL1.json): PDL1.pdb chains A hotspot 56 binder length 65")
        self.assertEqual(lines[-1], f"WARM PASS mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla tokens=112 ready=32.0 first_calls=40.0,30.0,20.0 steady=0.420 total=180.0 rc=0 out={res['out_dir']}")
        argv = self.launches[-1]; self.assertIn("--binder-name", argv); self.assertEqual(argv[argv.index("--binder-name") + 1], "warm"); self.assertEqual(argv[argv.index("--binder-len") + 1], "65")
        self.assertEqual((argv[argv.index("--starting-pdb") + 1], argv[argv.index("--chains") + 1], argv[argv.index("--target-hotspot-residues") + 1]), (res["input"]["target"], "A", "56"))

    def test_partial_follows_design(self):
        """warm exits as its design does: a partial arm (a lever line that is not the shipped decision) is NOT ACTIVE rc 3 with `partial` in the result
        and on the WARM line; there is no opt-out (an installed lever that did not engage is a defect)."""
        fallback = DEFECT
        launch = driver.launch

        def partial_launch(argv, env, **kw):
            r = launch(argv, env, **kw)
            lines = [*EXACT_OK, fallback, GATED_FN_112, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_SMALL, TXLA_GATED] + r["lines"][len(EXACT_OK) + 10:]                                   # the nosub line replaced by one that is not the shipped decision; the design's lines kept
            open(kw["log_path"], "w").write("\n".join(lines) + "\n")
            return dict(r, lines=lines)
        self.mp.setattr(driver, "launch", partial_launch)
        lines = []
        res = warm.run("fast", os.path.join(self.tmp, "w3"), self.params, log=lines.append)
        self.assertEqual((res["status"], res["exit_code"], list(res["partial"])), ("NOT ACTIVE", 3, ["nosub"])); self.assertFalse([k for k in res if k.startswith("allow")])
        self.assertTrue(lines[-1].startswith("WARM NOT ACTIVE mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla"), lines[-1]); self.assertTrue(lines[-1].endswith(" partial=nosub"), lines[-1])

    def test_override_and_refusals(self):
        tgt = _stubs.write_target(os.path.join(self.tmp, "t.pdb"), ("A", "B"), 20)
        res = warm.run("off", os.path.join(self.tmp, "w2"), self.params, target=tgt, chain="A,B", binder_len=50, seed=3)
        self.assertEqual((res["status"], res["levers"], res["input"]["case"], res["input"]["hotspot"], res["input"]["seed"]), ("PASS", [], None, None, 3))
        self.assertEqual(self.launches[-1][self.launches[-1].index("--binder-len") + 1], "50"); self.assertNotIn("--target-hotspot-residues", self.launches[-1])
        res = warm.run("turbo", os.path.join(self.tmp, "w3"), self.params)
        self.assertEqual((res["status"], res["exit_code"]), ("NOT ACTIVE", 3)); self.assertNotIn("timing", res)
        with self.assertRaises(warm.WarmError):
            warm.run("off", os.path.join(self.tmp, "w4"), self.params, target=tgt)
        with self.assertRaises(warm.WarmError):
            warm.run("off", os.path.join(self.tmp, "w5"), self.params, chain="A")

    def test_cli(self):
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = cli.main(["warm", "--mode", "off", "--out", os.path.join(self.tmp, "c1"), "--params-dir", self.params])
        self.assertEqual(rc, 0); self.assertIn("[colabdesign-opt] WARM PASS mode=off levers=none", buf.getvalue())
        with redirect_stderr(buf):
            rc = cli.main(["warm", "--mode", "off", "--out", os.path.join(self.tmp, "c2"), "--params-dir", self.params, "--starting-pdb", "x.pdb"])
        self.assertEqual(rc, 2)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf(_stubs.bindcraft_stack_missing(), _stubs.SKIP_BINDCRAFT)
class TestWarmProcess(unittest.TestCase):
    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_full_route_on_the_stub_stack(self):
        tmp = tempfile.mkdtemp(prefix="cd_opt_warm_proc_")
        site = _stubs.make_stub_stack(tmp)
        params = _stubs.make_params(os.path.join(tmp, "params"))
        env = _stubs.child_env(site, PATH=_stubs.nvidia_smi_stub(tmp) + os.pathsep + os.environ["PATH"], COLABDESIGN_OPT_HOME=_stubs.TREE)
        r = subprocess.run([sys.executable, "-m", "colabdesign_opt", "warm", "--mode", "off", "--out", os.path.join(tmp, "w"), "--params-dir", params],   # off: the stub stack has no Pallas kernel for fast; the stock arm runs the whole route
                           env=env, capture_output=True, text=True, cwd=tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        warm_line = [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt] WARM ")]
        self.assertEqual(len(warm_line), 1, r.stderr[-2000:]); self.assertTrue(warm_line[0].startswith(f"[colabdesign-opt] WARM PASS mode=off levers=none tokens={EXAMPLE_TOKENS} ready="), warm_line)   # PDL1 chain A (115 CA residues) + 65
        m = re.search(r" first_calls=([0-9.,]+) steady=([0-9.]+) total=", warm_line[0]); self.assertIsNotNone(m, warm_line)
        self.assertEqual(len(m.group(1).split(",")), 5)                                                                      # one first call per stage call: logits ×2, softmax, one-hot, greedy
        self.assertIn("warm case: BindCraft's example (stock/src/bindcraft/settings_target/PDL1.json): PDL1.pdb chains A hotspot 56 binder length 65", r.stderr)
        self.assertEqual(os.listdir(os.path.join(tmp, "w")), ["warm"]); self.assertTrue(os.path.isfile(os.path.join(tmp, "w", "warm", "design.pdb")))
        self.assertEqual(_files(os.path.join(tmp, "w", "warm")), {"design.fasta", "design.pdb", "run.log", "trajectory.jsonl", bindcraft.FAILURE_CSV})


if __name__ == "__main__":
    unittest.main()
