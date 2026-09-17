"""Activation: enable("exact") exports the delta only when every gate passes; the refusals are named; the upstream pin and the caller's
deterministic recipe are reported, never gated; a CFG / low-memory design line is refused by name (nothing exported, exit 3 under strict); late activation is
refused once the kit's module has read the switch; the autoload finder fires BEFORE the trigger package's body; the .pth is inert without
RFDIFFUSION3_OPT. All on fake interpreters (fixtures) and a fake nvidia-smi, in subprocesses of this interpreter."""
import json
import os
import re
import tempfile
import unittest

from .. import modes, registry
from . import _fixtures as fx

ENABLE = '''
import json, os, sys
import rfdiffusion3_opt
mode = sys.argv[1]
strict = "--strict" in sys.argv
try:
    rep = rfdiffusion3_opt.enable(mode, strict=strict)
except rfdiffusion3_opt.ActivationError as e:
    print(json.dumps({"error": str(e), "env": os.environ.get("RFD3_HOIST")})); sys.exit(3)
print(json.dumps({"active": rep.get("active"), "reason": rep.get("reason"), "findings": rep.get("findings"), "env": os.environ.get("RFD3_HOIST"),
                  "interpreter": (rep.get("interpreter") or {}).get("label"), "gpu": rep.get("gpu"), "applied": rep.get("applied"),
                  "levers": {k: rep.get(k) for k in ("levers_planned", "levers_applied", "levers_fallback", "levers_unavailable", "partial")},
                  "env_present": rep.get("env_present"), "has_notes": "notes" in rep, "has_readme_check": "readme_check" in rep, "attn_pin": rep.get("attn_pin"), "has_run_parameters": "run_parameters" in rep,
                  "det_env": rep.get("det_env"), "pins": rep.get("pins"), "env_line": rep.get("env_line"),
                  "levers_in_environ": sorted(k for k in ("RFD3_HOIST", "RFD3_FZT", "RFD3_CUDAGRAPH", "RFD3_INIT_CHUNK", "RFD3_COMPILE", "RFD3_TOKEN_SDPA", "RFD3_GATHER_ATTN") if k in os.environ),
                  "hook": rfdiffusion3_opt.stack.applied_hook() is not None, "status_same": rfdiffusion3_opt.status() is rep}))
'''


class TestEnable(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.make_site(os.path.join(self.td.name, "p"), "patched")
        self.pristine = fx.make_site(os.path.join(self.td.name, "s"), "pristine")

    def tearDown(self):
        self.td.cleanup()

    def _enable(self, site, mode="exact", env=None, strict=False, bin_dir=None):
        r = fx.run_py(ENABLE, site, env=env, bin_dir=self.bin if bin_dir is None else bin_dir, argv=[mode] + (["--strict"] if strict else []))
        try:
            data = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            self.fail(f"no JSON from the child: rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        return r, data

    def test_exact_activates_on_a_patched_interpreter(self):
        r, d = self._enable(self.patched)
        self.assertTrue(d["active"], d)
        self.assertEqual(d["env"], "1")
        self.assertEqual(d["interpreter"], "patched")
        self.assertEqual(d["gpu"]["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(d["gpu"]["sm"], "90")
        self.assertEqual(d["applied"], "at-import")
        self.assertEqual(d["levers"], {"levers_planned": ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"], "levers_applied": ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"], "levers_fallback": [], "levers_unavailable": [], "partial": False})
        self.assertTrue(d["status_same"]); self.assertTrue(d["hook"])                # the APPLIED hook is installed on activation
        self.assertEqual((d["has_notes"], d["has_readme_check"]), (False, False))    # no GPU-class note, no README check: the mode table is the one source
        self.assertEqual(d["pins"]["ok"], True); self.assertEqual(d["pins"]["notes"], [])   # the fake distribution sits at the pin: nothing to report
        self.assertNotIn("] PINS ", r.stderr)
        self.assertIn("[rfdiffusion3-opt] ACTIVE mode=exact env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1 interpreter=patched(10/10 patched)", r.stderr)
        self.assertIn("gpu=NVIDIA H100 80GB HBM3(sm90)", r.stderr)
        self.assertIn("[rfdiffusion3-opt] EXIT pid=", r.stderr)                    # the exit tally is registered at activation
        self.assertIn("rfd3.model.hoist was never loaded in this process", r.stderr)

    def test_exact_refused_on_a_pristine_interpreter(self):
        r, d = self._enable(self.pristine)
        self.assertFalse(d["active"])
        self.assertIsNone(d["env"])                                                 # nothing exported on refusal
        self.assertIn("the kit is not installed in this interpreter (rfd3 tree pristine", d["reason"])
        self.assertIn("install.sh", d["reason"])
        self.assertIn("[rfdiffusion3-opt] NOT ACTIVE: the kit is not installed", r.stderr)
        self.assertEqual(d["levers"], {"levers_planned": ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"], "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False})
        r, d = self._enable(self.pristine, strict=True)
        self.assertEqual(r.returncode, 3)
        self.assertIn("error", d)

    def test_exact_and_fast_proceed_on_any_gpu_class_without_a_note(self):
        """A visible GPU other than sm_90 activates exactly as sm_90 does: the delta exported, strict returns 0, the gpu reported, and no
        NOTE line of any kind (the report carries no `notes` key); `off` likewise."""
        a100 = fx.fake_gpu(os.path.join(self.td.name, "bin80"), line="NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB, 580.95.05")
        for mode, env_line in (("exact", "RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1"), ("fast", "RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1")):
            r, d = self._enable(self.patched, mode, bin_dir=a100)
            self.assertTrue(d["active"], d)
            self.assertEqual(d["env"], "1")
            self.assertEqual(d["applied"], "at-import")
            self.assertFalse(d["has_notes"])
            self.assertEqual(d["gpu"]["sm"], "80")
            self.assertIn(f"[rfdiffusion3-opt] ACTIVE mode={mode} env={env_line} interpreter=patched(10/10 patched)", r.stderr)
            self.assertIn("gpu=NVIDIA A100-SXM4-80GB(sm80)", r.stderr)
            self.assertNotIn("NOTE", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr); self.assertNotIn("sm_90", r.stderr)
            r, d = self._enable(self.patched, mode, bin_dir=a100, strict=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("error", d)
        r, d = self._enable(self.patched, "off", bin_dir=a100)
        self.assertNotIn("NOTE", r.stderr)

    def test_exact_refused_without_gpu(self):
        """PATH holds only an empty directory and the interpreter's own: no nvidia-smi anywhere, on a GPU box too."""
        nobin = os.path.join(self.td.name, "nobin")
        os.makedirs(nobin)
        r, d = self._enable(self.patched, bin_dir=nobin, env={"PATH": nobin + os.pathsep + os.path.dirname(fx.sys.executable)})
        self.assertFalse(d["active"])
        self.assertIn("no visible GPU", d["reason"])
        self.assertIsNone(d["gpu"])

    def test_off_exports_nothing(self):
        r, d = self._enable(self.pristine, "off")
        self.assertFalse(d["active"])
        self.assertIsNone(d["env"])
        self.assertIn("mode off: stock", d["reason"])
        self.assertIn("[rfdiffusion3-opt] NOT ACTIVE mode=off env=none interpreter=pristine(", r.stderr)
        self.assertIn("(stock: nothing exported)", r.stderr)
        self.assertEqual(d["env_present"], [])
        self.assertEqual(d["levers"], {"levers_planned": [], "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False})
        r, d = self._enable(self.pristine, "off", env={"RFD3_HOIST": "1", "RFD3_LOW_MEMORY_MODE": "1"})   # report only: the stock routes strip them
        self.assertFalse(d["active"])
        self.assertEqual(d["env_present"], ["RFD3_HOIST", "RFD3_LOW_MEMORY_MODE"])
        self.assertIn("env_present=RFD3_HOIST,RFD3_LOW_MEMORY_MODE", r.stderr)

    def test_the_pin_word_prints_on_the_active_line(self):
        """The kit routes pin the pre-flight atom-attention decision (routes.Route.attn_pin): `attn_pin=1` on the ACTIVE line after `applied=`
        under exact and fast. There is no run-parameter: a lever is part of a mode or it is not, and no other word follows `applied=`."""
        r, d = self._enable(self.patched, "fast")
        self.assertTrue(d["active"], (d, r.stderr))
        self.assertIn("ACTIVE mode=fast env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1 interpreter=patched(10/10 patched) ", r.stderr)
        self.assertRegex(r.stderr, r" applied=at-import attn_pin=1 alloc_conf=\S+")   # report.activation_line: the pin word right after applied=
        self.assertNotIn("dynamic=", r.stderr); self.assertNotIn("withheld=", r.stderr)
        self.assertTrue(d["attn_pin"]); self.assertFalse(d["has_run_parameters"])
        r, d = self._enable(self.patched, "exact")
        self.assertIn(" applied=at-import attn_pin=1 alloc_conf=", r.stderr); self.assertTrue(d["attn_pin"])
        for name in ("RFD3_OPT_ATTN_PIN", "RFD3_COMPILE_DYNAMIC", "RFD3_HOIST_PARTS"):      # names no file of the kit reads: not the kit's, nothing to refuse or print
            r, d = self._enable(self.patched, "exact", env={name: "0"})
            self.assertTrue(d["active"], (name, d)); self.assertIn(" applied=at-import attn_pin=1 alloc_conf=", r.stderr)

    def test_environment_disagreement_is_refused(self):
        for env, needle in (({"RFD3_HOIST": "0"}, "disagrees with mode exact"),
                            ({"RFD3_COMPILE": "1"}, "not part of mode exact"),
                            ({"RFD3_GATHER_ATTN": "1"}, "RFD3_GATHER_ATTN='1' in the environment is not part of mode exact"),   # a kit name the mode does not export: refused like any other
                            ({"RFD3_DENSE_SDPA_ATTENTION": "0"}, "upstream's own switch"),        # upstream's force-sparse switch: the kit's levers run on the dense path only
                            ({"RFD3_LOW_MEMORY_MODE": "1"}, "upstream's own switch")):
            r, d = self._enable(self.patched, env=env)
            self.assertFalse(d["active"], (env, d))
            self.assertIn(needle, d["reason"], env)
            self.assertEqual(d["levers_in_environ"], sorted(k for k in env if k in ("RFD3_HOIST", "RFD3_COMPILE", "RFD3_GATHER_ATTN")), env)   # nothing exported on refusal (only what the caller had set)
        r, d = self._enable(self.patched, env={"RFD3_HOIST": "1"})                  # the mode's own value pre-set: agreed, idempotent
        self.assertTrue(d["active"])

    def test_the_deterministic_recipe_is_tolerated_and_reported(self):
        """FOUNDRY_DET_SCATTER / CUBLAS_WORKSPACE_CONFIG are the caller's deterministic recipe (torch's / foundry's names): set under a kit
        mode they refuse nothing — the delta is exported, strict returns 0 — and the report names them as set under `det_env`."""
        det = {"FOUNDRY_DET_SCATTER": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        for mode in ("exact", "fast"):
            r, d = self._enable(self.patched, mode, env=det)
            self.assertTrue(d["active"], (mode, d))
            self.assertEqual(d["env"], "1")
            self.assertEqual(d["findings"], [])
            self.assertEqual(d["det_env"], det)
            self.assertIn(f"[rfdiffusion3-opt] ACTIVE mode={mode} env=RFD3_HOIST=1,", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
            r, d = self._enable(self.patched, mode, env=det, strict=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        r, d = self._enable(self.patched, "exact", env={"FOUNDRY_DET_SCATTER": "1"})
        self.assertEqual(d["det_env"], {"FOUNDRY_DET_SCATTER": "1"})                 # only the names that are set
        r, d = self._enable(self.patched, "exact")
        self.assertEqual(d["det_env"], {})

    def test_pins_are_reported_never_gated(self):
        """The upstream distribution off its pin (an index install here: no direct_url.json) refuses nothing under a kit mode: the delta is
        exported, strict returns 0, `findings` is empty; the pin checker's finding rides in `pins.notes` (`pins.ok` False) and prints as one
        `PINS <finding>` line before the ACTIVE line. At the pin: `pins.ok`, no notes, no PINS line."""
        site = fx.unpin_distribution(fx.make_site(os.path.join(self.td.name, "unpinned"), "patched"))
        for mode in ("exact", "fast"):
            r, d = self._enable(site, mode)
            self.assertTrue(d["active"], (mode, d, r.stderr))
            self.assertEqual((d["env"], d["findings"], d["applied"]), ("1", [], "at-import"))
            self.assertFalse(d["pins"]["ok"]); self.assertEqual(len(d["pins"]["notes"]), 1, d["pins"])
            self.assertIn("rc-foundry 0.2.1.dev13+g4010e3e2e: installed from a non-git, non-archive source; want version 0.2.1.dev13+g4010e3e2e from", d["pins"]["notes"][0])
            pins = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] PINS ")]
            self.assertEqual(pins, [f"[rfdiffusion3-opt] PINS {d['pins']['notes'][0]}"])   # one line per finding, the finding verbatim
            self.assertLess(r.stderr.index("] PINS "), r.stderr.index("] ACTIVE mode="))   # reported before the activation line
            self.assertNotIn("NOT ACTIVE", r.stderr); self.assertNotIn("pins:", d.get("reason") or "")
            r, d = self._enable(site, mode, strict=True)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertNotIn("error", d)
        r, d = self._enable(site, "off")                                              # the stock arm reports it too (its child proves the TREE, not the pin)
        self.assertEqual(len([l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] PINS ")]), 1, r.stderr)
        self.assertFalse(d["pins"]["ok"])
        r, d = self._enable(self.patched, "exact")                                    # at the pin: nothing to report
        self.assertEqual((d["pins"]["ok"], d["pins"]["notes"]), (True, [])); self.assertNotIn("] PINS ", r.stderr)

    def test_an_unserved_line_is_refused_by_name(self):
        """A design line that turns classifier-free guidance (or the low-memory tokenization) on, gated by activate(argv=...): the mode
        REFUSES it by name — the finding is modes.unserved_reason(...), nothing is exported to os.environ, no APPLIED hook, one NOT ACTIVE line
        (never ACTIVE / DECLINED), ActivationError under strict (exit 3); the same line under `off` is stock's own business (NOT ACTIVE
        mode=off, no finding), and the same line without the switch — or with the CFG parameters / the driven sampler options only — activates."""
        code = ENABLE.replace("rfdiffusion3_opt.enable(mode, strict=strict)", "rfdiffusion3_opt.stack.activate(mode, strict=strict, trigger='design', argv=[t for t in sys.argv[2:] if t != '--strict'])")
        base = ["out_dir=/o", "inputs=/s.json", "ckpt_path=/w.ckpt", "diffusion_batch_size=8"]
        for tok, feature, asked in (("inference_sampler.use_classifier_free_guidance=true", "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true"),
                                    ("low_memory_mode=True", "low_memory_mode", "low_memory_mode=true")):
            line = base + [tok]
            for mode in ("exact", "fast"):
                why = modes.unserved_reason(mode, feature, asked)
                r = fx.run_py(code, self.patched, bin_dir=self.bin, argv=[mode] + line)
                self.assertEqual(r.returncode, 0, r.stderr)                             # not strict: the report says no
                d = json.loads(r.stdout.strip().splitlines()[-1])
                self.assertFalse(d["active"]); self.assertEqual(d["applied"], "no")
                self.assertEqual((d["findings"], d["reason"]), ([why], why))
                self.assertEqual((d["env"], d["levers_in_environ"], d["hook"]), (None, [], False))   # nothing exported, no APPLIED hook, no decline record
                na = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] NOT ACTIVE: ")]
                self.assertEqual(na, [f"[rfdiffusion3-opt] NOT ACTIVE: {why}"], r.stderr)
                self.assertNotIn("] ACTIVE ", r.stderr); self.assertNotIn("DECLINED", r.stderr)
                r = fx.run_py(code, self.patched, bin_dir=self.bin, argv=[mode, "--strict"] + line)
                self.assertEqual(r.returncode, 3, r.stderr)                             # strict: ActivationError
                d = json.loads(r.stdout.strip().splitlines()[-1])
                self.assertEqual((d["error"], d["env"]), (why, None))
            r = fx.run_py(code, self.pristine, bin_dir=self.bin, argv=["off"] + line)  # off: upstream's keys are upstream's; nothing to refuse
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertFalse(d["active"]); self.assertEqual(d["findings"], []); self.assertIn("NOT ACTIVE mode=off", r.stderr)
        for extra in ([], ["inference_sampler.cfg_scale=2.0", "inference_sampler.cfg_t_max=0.5"], ["inference_sampler.use_classifier_free_guidance=false", "low_memory_mode=false"],
                      ["inference_sampler.allow_realignment=true", "inference_sampler.s_jitter_origin=0.5", "inference_sampler.fraction_of_steps_to_fix_motif=0.3"]):
            r = fx.run_py(code, self.patched, bin_dir=self.bin, argv=["exact"] + base + extra)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(d["active"], (extra, d)); self.assertEqual(d["levers_in_environ"], ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"])
            self.assertIn("] ACTIVE mode=exact ", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)

    def test_unknown_modes(self):
        for word in ("exact_lowmem", "turbo"):                                    # a lever subset and an unknown word alike: not a mode, one sentence (the core's)
            r, d = self._enable(self.patched, word)
            self.assertFalse(d["active"])
            self.assertIn(f"'{word}' is not a mode (exact|fast|off)", d["reason"])

    def test_fast_exports_both_switches(self):
        r, d = self._enable(self.patched, "fast")
        self.assertTrue(d["active"], d.get("reason"))
        self.assertEqual(d["env"], "1")                                          # RFD3_HOIST as the child's environment holds it after enable()
        self.assertEqual(sorted(d["levers"]["levers_planned"]), ["RFD3_COMPILE", "RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_GATHER_ATTN", "RFD3_HOIST", "RFD3_INIT_CHUNK", "RFD3_TOKEN_SDPA"])
        self.assertEqual(d["levers_in_environ"], ["RFD3_COMPILE", "RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_GATHER_ATTN", "RFD3_HOIST", "RFD3_INIT_CHUNK", "RFD3_TOKEN_SDPA"])   # every planned lever is in the environment after enable()
        self.assertIn("ACTIVE mode=fast env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1 interpreter=patched(10/10 patched)", r.stderr)
        r, d = self._enable(self.patched, "")                                        # no mode named: the default is fast (modes.DEFAULT_MODE)
        self.assertEqual(modes.DEFAULT_MODE, "fast"); self.assertTrue(d["active"], d); self.assertIn("ACTIVE mode=fast env=", r.stderr)
        r, d = self._enable(self.patched, "exact", env={"RFD3_COMPILE": "1"})     # the compile switch pre-set under exact: not part of the mode, refused
        self.assertFalse(d["active"])

    def test_second_mode_and_idempotence(self):
        code = ENABLE + '''
rep2 = rfdiffusion3_opt.enable("exact")
rep3 = rfdiffusion3_opt.enable("off")
print(json.dumps({"same": rep2 is rep, "second": rep3.get("reason")}))
'''
        r = fx.run_py(code, self.patched, bin_dir=self.bin, argv=["exact"])
        last = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(last["same"])
        self.assertIn("a second mode is refused", last["second"])

    def test_late_activation_is_refused(self):
        code = '''
import json, os, sys, types
m = types.ModuleType("rfd3.model.hoist"); m.HOIST = False
sys.modules["rfd3.model.hoist"] = m                       # the kit's module already imported: RFD3_HOIST was read at its import
import rfdiffusion3_opt
rep = rfdiffusion3_opt.enable("exact")
print(json.dumps({"active": rep["active"], "reason": rep.get("reason"), "env": os.environ.get("RFD3_HOIST")}))
'''
        r = fx.run_py(code, self.patched, bin_dir=self.bin)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertFalse(d["active"])
        self.assertIn("late activation", d["reason"])
        hoist_src = open(os.path.join(registry.kit_home(), "patched", "rfd3", "model", "hoist.py"), encoding="utf-8").read().splitlines()
        read_at = 1 + next(i for i, l in enumerate(hoist_src) if l.startswith("HOIST = os.environ.get(\"RFD3_HOIST\""))   # the line the kit file reads the switch on
        for cited in re.findall(r"hoist\.py:(\d+)", d["reason"]):                       # a source pointer in the refusal names that line (or none is given)
            self.assertEqual(int(cited), read_at, d["reason"])
        self.assertIsNone(d["env"])

    def test_kit_bytes_gate(self):
        """A kit tree missing one of its carried files refuses activation, the file named (check_tree checks tree shape — every
        carried path present, nothing stray beside them — not byte content: the git commit identifies the carried files)."""
        with tempfile.TemporaryDirectory() as td:
            import shutil
            tree2 = os.path.join(td, "tree")
            shutil.copytree(registry.tree_home(), tree2, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            os.remove(os.path.join(tree2, registry.KIT_RELPATH, "patched", "rfd3", "model", "hoist.py"))
            r = fx.run_py(ENABLE, self.patched, env={"MODEL_OPT": tree2}, bin_dir=self.bin, argv=["exact"])
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertFalse(d["active"])
            self.assertIn("kit bytes", d["reason"])
            self.assertIn("patched/rfd3/model/hoist.py", d["reason"])


class TestAutoload(unittest.TestCase):
    """The .pth route: RFDIFFUSION3_OPT=exact + `import rfd3` -> the delta is in the environment when the package body runs."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.make_site(os.path.join(self.td.name, "p"), "patched")
        self.pristine = fx.make_site(os.path.join(self.td.name, "s"), "pristine")
        self.seen = os.path.join(self.td.name, "seen.json")

    def tearDown(self):
        self.td.cleanup()

    def _run(self, site, env, tail=""):
        code = ("import os, sys, json, rfdiffusion3_opt._autoload as a\nimport rfd3\n" + tail +
                "print(json.dumps({'finder': a.FINDER is not None, 'armed': getattr(a.FINDER, 'armed', None), 'seen': rfd3.SEEN, 'after': os.environ.get('RFD3_HOIST'), "
                "'hoist': getattr(sys.modules.get('rfd3.model.hoist'), 'HOIST', None)}))")
        return fx.run_py(code, site, env=dict(env, RFD3_FAKE_SEEN=self.seen), bin_dir=self.bin)

    def test_fires_at_the_trigger_before_the_kit_module(self):
        """The core's finder fires when `rfd3` is imported, after that package's own body (upstream's imports pydantic and packaging
        only): the delta is in os.environ before anything imports `rfd3.model.hoist`, which reads it at its import (hoist.py:22)."""
        fx.fake_torch(self.patched)
        r = self._run(self.patched, {"RFDIFFUSION3_OPT": "exact"}, tail="import rfd3.model.hoist\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual((d["finder"], d["armed"]), (True, False))            # installed at start, fired once
        self.assertIsNone(d["seen"]["RFD3_HOIST"])                            # the trigger's own body runs first …
        self.assertEqual((d["after"], d["hoist"]), ("1", True))               # … the delta is exported right after it, and the kit's module read it on
        self.assertIn("ACTIVE mode=exact env=RFD3_HOIST=1", r.stderr)
        self.assertIn("trigger=rfd3", r.stderr)
        self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=True", r.stderr)   # the activation's own hook, on this route too

    def test_any_gpu_class_proceeds_without_a_note(self):
        """On a visible GPU other than sm_90 the trigger activates exactly as on sm_90: the delta exported, the kit's module reads it,
        exit 0, the ACTIVE line — and no NOTE line on either."""
        fx.fake_torch(self.patched)
        sm90_bin, self.bin = self.bin, fx.fake_gpu(os.path.join(self.td.name, "bin80"), line="NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB, 580.95.05")
        try:
            for mode in ("exact", "fast"):
                if os.path.exists(self.seen):
                    os.remove(self.seen)
                r = self._run(self.patched, {"RFDIFFUSION3_OPT": mode}, tail="import rfd3.model.hoist\n")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f"ACTIVE mode={mode} env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1", r.stderr)
                self.assertIn("gpu=NVIDIA A100-SXM4-80GB(sm80)", r.stderr)
                self.assertNotIn("NOTE", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
                d = json.loads(r.stdout.strip().splitlines()[-1])
                self.assertEqual((d["after"], d["hoist"]), ("1", True))       # the delta was exported and the kit's module read it
        finally:
            self.bin = sm90_bin
        r = self._run(self.patched, {"RFDIFFUSION3_OPT": "exact"}, tail="import rfd3.model.hoist\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ACTIVE mode=exact env=RFD3_HOIST=1", r.stderr)
        self.assertNotIn("NOTE", r.stderr)

    def test_a_cfg_line_at_the_trigger_is_refused_and_ends_the_process(self):
        """RFDIFFUSION3_OPT=exact on upstream's own command line carrying the CFG switch: at the trigger the mode refuses the line by name —
        one NOT ACTIVE line naming RFDIFFUSION3_OPT=off as the stock path, and the process ENDS 3 there and then: nothing after `import rfd3`
        runs (no stdout), nothing is exported, no ACTIVE / DECLINED / APPLIED line. The CFG parameters without the switch are served."""
        fx.fake_torch(self.patched)
        code = ("import os, sys, json, rfdiffusion3_opt._autoload as a\nsys.argv = ['rfd3', 'design', 'out_dir=/o', 'inputs=/s.json'] + sys.argv[1:]\nimport rfd3\nimport rfd3.model.hoist\n"
                "print(json.dumps({'seen': rfd3.SEEN, 'after': os.environ.get('RFD3_HOIST'), 'hoist': getattr(sys.modules.get('rfd3.model.hoist'), 'HOIST', None)}))")
        r = fx.run_py(code, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, bin_dir=self.bin, argv=["inference_sampler.use_classifier_free_guidance=true", "inference_sampler.cfg_scale=1.5"])
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout.strip(), "", r.stdout)
        na = [l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] NOT ACTIVE: ")]
        self.assertEqual(na, ["[rfdiffusion3-opt] NOT ACTIVE: " + modes.unserved_reason("exact", "classifier-free guidance", "inference_sampler.use_classifier_free_guidance=true")], r.stderr)
        self.assertIn("RFDIFFUSION3_OPT=off", na[0])
        for absent in ("DECLINED", "] ACTIVE ", "] APPLIED "):
            self.assertNotIn(absent, r.stderr)
        r = fx.run_py(code, self.patched, env={"RFDIFFUSION3_OPT": "exact", "RFD3_FAKE_SEEN": self.seen}, bin_dir=self.bin, argv=["inference_sampler.cfg_scale=1.5"])
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual((d["after"], d["hoist"]), ("1", True))                    # served: the delta exported at the trigger, the kit's module read it
        self.assertIn("] ACTIVE mode=exact ", r.stderr); self.assertIn("trigger=rfd3", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)

    def test_refusal_stops_the_process(self):
        r = self._run(self.pristine, {"RFDIFFUSION3_OPT": "exact"})
        self.assertEqual(r.returncode, 3)
        self.assertIn("NOT ACTIVE: the kit is not installed", r.stderr)
        self.assertEqual(r.stdout.strip(), "")                                # the process ended at the trigger: stock never runs silently

    def test_inert_without_the_switch(self):
        for env in ({}, {"RFDIFFUSION3_OPT": "off"}):
            r = self._run(self.pristine, env)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertFalse(d["finder"])
            self.assertIsNone(d["seen"]["RFD3_HOIST"])
            self.assertNotIn("[rfdiffusion3-opt]", r.stderr)

    def test_unknown_values_are_refused(self):
        """A value that is not a mode is refused when `_autoload` runs (interpreter start on the real route): NOT ACTIVE, exit 3, nothing
        of upstream runs — stock never runs silently under RFDIFFUSION3_OPT (README), on the patched and the pristine interpreter alike;
        a lever subset (`exact_lowmem`) is refused like any other word that is not a mode."""
        for site in (self.patched, self.pristine):
            for word in ("turbo", "exact_lowmem"):
                if os.path.exists(self.seen):
                    os.remove(self.seen)
                r = self._run(site, {"RFDIFFUSION3_OPT": word})
                self.assertEqual(r.returncode, 3, r.stderr)
                self.assertEqual(r.stderr.strip(), f"[rfdiffusion3-opt] NOT ACTIVE: '{word}' is not a mode (exact|fast|off)")
                self.assertEqual(r.stdout.strip(), "")
                self.assertFalse(os.path.exists(self.seen))                  # refused before any import of upstream

    def test_applied_line_from_the_kit_module(self):
        """The one-shot hook on rfd3.model.hoist reports what the module read (a synthetic hoist module standing in for the kit's,
        which needs torch)."""
        with tempfile.TemporaryDirectory() as td:
            site = fx.make_site(td, "patched")
            with open(os.path.join(site, "rfd3", "model", "hoist.py"), "w") as fh:     # stands in for the kit file in this fake site only
                fh.write('import os\nHOIST = os.environ.get("RFD3_HOIST", "0") == "1"\nPARTS = {"pll", "dedup"}\nHOIST_STATS = {"rollouts": 0}\n'
                         'COMPILE = False\nTOKEN_SDPA = False\n'
                         'def describe():\n    return {"RFD3_HOIST": HOIST, "parts": ",".join(sorted(PARTS)), "RFD3_COMPILE": COMPILE, "RFD3_TOKEN_SDPA": TOKEN_SDPA}\n')
            code = '''
import sys, json
from rfdiffusion3_opt import stack
f = stack.AppliedHook.install()
import rfd3.model.hoist as h
print(json.dumps({"applied": f.applied, "HOIST": h.HOIST, "in_meta_path": f in sys.meta_path}))
'''
            r = fx.run_py(code, site, env={"RFD3_HOIST": "1"}, bin_dir=self.bin)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(d["applied"])
            self.assertTrue(d["HOIST"])
            self.assertFalse(d["in_meta_path"])
            self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=True parts=dedup,pll RFD3_COMPILE=False RFD3_TOKEN_SDPA=False", r.stderr)   # report.applied_line: the switches the module read, no instrument word
            self.assertNotIn("assert=", r.stderr)


if __name__ == "__main__":
    unittest.main()
