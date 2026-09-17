"""The stack-word observer and the KERNELS line's stack words (census.py, report-only): upstream's `Boltz.predict_step` is wrapped once per
armed process — at the census arm when its module is imported, else the moment it is — by a pass-through observer that reads the TF32
switches each call runs under and nothing else (no clock, no synchronisation, no line of its own: the package times nothing),
passes arguments, the return value and any exception through untouched; the KERNELS line ends `compile=… torch=… cuda=… py=… tf32=…
cudnn_tf32=… alloc_conf=… seed=…`, in that order, each word formed without raising; the ACTIVE line's last word is `fallbacks=`."""
import json
import re
import shutil
import tempfile
import unittest

from boltzgen_opt import census, design, modes, report, stack
from boltzgen_opt.tests import _stubs

HOME = stack.opt_home()
TAIL_RE = re.compile(r" thresholds=\S+ compile=(?P<compile>\S+) torch=(?P<torch>\S+) cuda=(?P<cuda>\S+) py=(?P<py>\S+) tf32=(?P<tf32>\S+) "
                     r"cudnn_tf32=(?P<cudnn_tf32>\S+) alloc_conf=(?P<alloc_conf>\S+) seed=(?P<seed>\S+)$")


class TestCallObserver(unittest.TestCase):
    """A fresh interpreter on the stubs: arm, import upstream's model module, call predict_step."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_observer_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # argv: payload; a fake CUDA seam (synchronize counted: the observer must never call it); upstream's checkpoint state as boltz.py sets it up (l.446-455)
    CODE = ("import json, sys, torch\n"
            "calls = []\n"
            "torch.cuda.is_available = lambda: True\n"
            "torch.cuda.synchronize = lambda *a, **k: calls.append(1)\n"
            "from boltzgen_opt import census\n"
            "if IMPORT_FIRST: import boltzgen.model.models.boltz\n"
            "census.arm(sys.argv[1])\n"
            "hooked = any(type(f).__name__ == '_Hook' and f.name == census.UPSTREAM_MODEL_MODULE for f in sys.meta_path)\n"
            "import boltzgen.model.models.boltz as bz\n"
            "w = bz.Boltz.predict_step\n"
            "out = dict(hooked_before_import=hooked, marker=getattr(w, '_boltzgen_opt_census', None), again=census.install_call_observer(),\n"
            "           still_same=bz.Boltz.predict_step is w, wraps_stub=w.__wrapped__.__code__.co_filename.endswith('boltz.py'),\n"
            "           hook_left=any(type(f).__name__ == '_Hook' and f.name == census.UPSTREAM_MODEL_MODULE for f in sys.meta_path))\n"
            "m = bz.Boltz(); m.predict_args = dict(diffusion_samples=16); m.current_checkpoint_index = -1; m.checkpoint_paths = ['adherence.ckpt']\n"
            "torch.set_float32_matmul_precision('high')\n"                      # what upstream's design step sets (matmul_precision of the step config)
            "b1, b2, b4 = dict(id=1), dict(id=2, switch=True), dict(id=4, switch=True)\n"
            "r1 = m.predict_step(b1, 0); r2 = m.predict_step(b2, 1)\n"
            "try:\n"
            "    m.predict_step(dict(id=3, **{'raise': 'boom'}), 2)\n"
            "except RuntimeError as e:\n"
            "    out['raised'] = str(e)\n"
            "r4 = m.predict_step(batch=b4, batch_idx=3)\n"                       # a second switch point past the paths: index advances, nothing loads (upstream's own condition)
            "torch.set_float32_matmul_precision('highest')\n"                   # the in-process runner restores the interpreter default after the step (bg_inproc.py)
            "out.update(identity=[r1 is b1, r2 is b2, r4 is b4], syncs=len(calls), loaded=m.loaded, index=m.current_checkpoint_index)\n"
            "print(json.dumps(out))\n")

    def run_child(self, import_first=False, **env):
        pl = census.payload("off", "on", step="design", item="pdl1 ref", seed=42)
        rc, so, se = _stubs.run_py(self.CODE.replace("IMPORT_FIRST", str(bool(import_first))), _stubs.clean_env(self.site, **env), args=[pl])
        self.assertEqual(rc, 0, se)
        self.assertFalse([ln for ln in se.splitlines() if ln.startswith("[boltzgen-opt") and " KERNELS " not in ln and " PEAK " not in ln], se)   # the observer prints nothing: KERNELS and PEAK are the process's only census lines
        return json.loads(so.strip().splitlines()[-1]), se

    def test_arm_then_import_wraps_once_and_passes_every_call_through(self):
        out, se = self.run_child()
        self.assertEqual(out["hooked_before_import"], True)
        self.assertEqual(out["marker"], "observer")
        self.assertEqual((out["again"], out["still_same"], out["wraps_stub"], out["hook_left"]), (True, True, True, False))   # idempotent: one wrapper, the hook retired
        self.assertEqual(out["identity"], [True, True, True])               # the return value is the original's, untouched
        self.assertEqual(out["raised"], "boom")                              # the exception is the caller's, unchanged
        self.assertEqual(out["syncs"], 0)                                    # no device interaction: the observer never synchronises
        self.assertEqual((out["loaded"], out["index"]), (["adherence.ckpt"], 1))   # upstream's own checkpoint switch ran inside the wrapped call, untouched
        (rec,) = census.parse_lines(se.splitlines())                        # the KERNELS line at exit, with the stack words
        m = TAIL_RE.search(rec["raw"])
        self.assertIsNotNone(m, rec["raw"])
        self.assertEqual(m["compile"], "off")
        self.assertEqual(m["seed"], "42")
        self.assertEqual(m["tf32"], "1/high")                                # what the calls ran under, not the restored exit-time default
        self.assertEqual(m["alloc_conf"], "unset")
        self.assertRegex(m["py"], r"^\d+\.\d+\.\d+$")
        self.assertNotIn("unknown", m["torch"] + m["cuda"] + m["cudnn_tf32"])

    def test_module_imported_before_the_arm_is_wrapped_at_the_arm(self):
        out, se = self.run_child(import_first=True, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        self.assertEqual(out["hooked_before_import"], False)                # nothing to hook: wrapped at once
        self.assertEqual((out["marker"], out["again"], out["still_same"], out["hook_left"], out["syncs"]), ("observer", True, True, False, 0))
        (rec,) = census.parse_lines(se.splitlines())
        self.assertEqual(TAIL_RE.search(rec["raw"])["alloc_conf"], "expandable_segments:True")


class TestPayload(unittest.TestCase):
    def test_payload_carries_the_seed(self):
        p = census.parse_payload(census.payload("off", "on", step="design", item="a b", seed=3))
        self.assertEqual((p["item"], p["seed"]), ("a_b", "3"))
        self.assertNotIn("seed=", census.payload("off", "on"))
        self.assertEqual(census.parse_payload(census.payload("off", "on", step="design", item="a b", seed=3))["route"], "default")   # off is upstream alone: route=default
        res = modes.resolve("exact", HOME)
        env, _ = design.kit_env(res, "/r/job7", {"PATH": "/bin"}, seed=11)
        self.assertEqual(census.parse_payload(env[census.ENV_KERNELS])["seed"], "11")
        env, _ = design.kit_env(res, "/r/job7", {"PATH": "/bin"})
        self.assertNotIn("seed", census.parse_payload(env[census.ENV_KERNELS]))


class TestStackWords(unittest.TestCase):
    """The KERNELS tail in a fresh interpreter: TorchDynamo's own graph count, the torch facts, never raising."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_words_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def words(self, code, **env):
        rc, so, se = _stubs.run_py(code, _stubs.clean_env(self.site, **env), args=[census.payload("off", "on", step="design", seed=5)])
        self.assertEqual(rc, 0, se)
        return json.loads(so.strip().splitlines()[-1])

    def test_compile_word(self):
        out = self.words("import json, sys, torch, torch._dynamo.utils as u; from boltzgen_opt import census\n"
                         "a = census.compile_word(); u.counters['stats']['unique_graphs'] += 2; b = census.compile_word()\n"
                         "print(json.dumps([a, b]))\n")
        self.assertEqual(out, ["off", "engaged[graphs=2]"])
        out = self.words("import json; from boltzgen_opt import census; print(json.dumps([census.compile_word(), census.stack_words(census.parse_payload(__import__('sys').argv[1]))]))\n")
        self.assertEqual(out[0], "unknown:torch-not-imported")             # named, never raised
        self.assertIn("torch=unknown:torch-not-imported ", out[1])
        self.assertTrue(out[1].endswith(" alloc_conf=unset seed=5"), out[1])

    def test_words_read_torch_and_the_line_ends_with_them_in_order(self):
        out = self.words("import json, sys, platform, torch; from boltzgen_opt import census; census.arm(sys.argv[1], print_if_unused=False)\n"
                         "torch.backends.cudnn.allow_tf32 = False\n"
                         "print(json.dumps([census.stack_words(), census.line(), torch.__version__, str(torch.version.cuda or 'none'), platform.python_version()]))\n",
                         PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")
        words, line, tv, cv, pv = out
        m = TAIL_RE.search(line)
        self.assertIsNotNone(m, line)
        self.assertTrue(line.endswith(" " + words), line)
        self.assertEqual((m["torch"], m["cuda"], m["py"], m["cudnn_tf32"], m["alloc_conf"], m["seed"]), (tv, cv, pv, "0", "max_split_size_mb:128", "5"))
        self.assertRegex(m["tf32"], r"^[01]/(highest|high|medium)$")
        self.assertIn(m["compile"], ("off",))


class TestLinesStartAtColumnZero(unittest.TestCase):
    """Behind upstream's progress bar (tqdm: `\\r`, the bar, no line feed) every census / activation / refusal line still starts a line of
    its own: the stream split on line feeds yields each line matching its ^-anchored grammar, never glued to a bar fragment."""

    FRAG = "[Step 1/1] design - Predicting DataLoader 0: :  33%|\u2588\u2588\u2588\u258e      | 1/3 [00:29<00:59,  0.03it/s]"
    CODE = ("import json, sys, types, torch\n"
            "from unittest import mock\n"
            "from boltzgen_opt import census, report\n"
            "census.arm(sys.argv[1])\n"
            "import boltzgen.model.models.boltz as bz\n"
            "m = bz.Boltz(); m.current_checkpoint_index = -1; m.checkpoint_paths = []\n"
            "def bar(): sys.stderr.write('\\r' + sys.argv[2]); sys.stderr.flush()      # tqdm's refresh: carriage return, the bar, no line feed\n"
            "bar(); m.predict_step({'x': 1}, 0)\n"
            "bar(); m.predict_step({'x': 2}, 1)\n"
            "bar(); report.emit({'active': True, 'mode': 'exact', 'form': 'process', 'switches': 'BG_GRAPH=graph', 'boltzgen_version': '0.3.2', 'gpu': None, "
            "'levers_applied': ['inproc'], 'levers_fallback': [], 'dropped': []})\n"
            "bar(); report.say(report.not_active_line('mode exact routes through the cuEquivariance kernels and this process cannot provide them (test)'))\n"
            "cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, current_device=lambda: 0, synchronize=lambda: None,\n"
            "                             max_memory_allocated=lambda dev=None: 2 ** 30, max_memory_reserved=lambda dev=None: 2 ** 31,\n"
            "                             memory_allocated=lambda dev=None: 2 ** 29, memory_reserved=lambda dev=None: 2 ** 30, reset_peak_memory_stats=lambda dev=None: None)\n"
            "bar()\n"
            "with mock.patch.object(torch, 'cuda', cuda):\n"
            "    census.print_line()               # the exit-time printer (KERNELS + PEAK), here behind a bar fragment\n")

    def test_bar_fragments_never_glue_a_line(self):
        tmp = tempfile.mkdtemp(prefix="bgopt_col0_")
        try:
            site = _stubs.materialize(tmp)
            pl = census.payload("off", "on", step="design", item="x", seed=1)
            rc, so, se = _stubs.run_py(self.CODE, _stubs.clean_env(site), args=[pl, self.FRAG])
            self.assertEqual(rc, 0, se[-1500:])
            lines = se.split("\n")
            self.assertEqual(len(census.parse_lines(lines)), 1)
            self.assertEqual(len(census.parse_peak_lines(lines)), 1)
            self.assertEqual(sum(1 for l in lines if re.match(r"^\[boltzgen-opt\] ACTIVE mode=exact .* fallbacks=none$", l)), 1)
            self.assertEqual(sum(1 for l in lines if re.match(r"^\[boltzgen-opt\] NOT ACTIVE: mode exact routes through the cuEquivariance kernels and this process cannot provide them \(test\)$", l)), 1)
            frags = [l for l in lines if "it/s]" in l]
            self.assertEqual(len(frags), 5, se[-1500:])                       # every bar refresh ended up on a line of its own …
            self.assertFalse([l for l in frags if "[boltzgen-opt" in l])        # … with nothing of the package glued to it
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestExamplesParse(unittest.TestCase):
    """census.EXAMPLES / report.EXAMPLES are real H100 lines of this tree; each parses under the module's own grammar (an outside reader parses
    the same strings under its own grammar; the two cannot drift apart unnoticed)."""

    def test_kernels_examples(self):
        ex = census.EXAMPLES
        self.assertEqual(set(ex), {"kernels_default", "kernels_exact", "kernels_big", "kernels_default_compiled", "peak"})   # no stock route: upstream alone is route=default, compiled or not
        for key, line in ex.items():
            self.assertEqual(line, line.strip(), key)
            if key.startswith("kernels_"):
                (rec,) = census.parse_lines([line])
                route = {"kernels_default_compiled": "default"}.get(key, key[len("kernels_"):])
                self.assertEqual(rec["route"], route, key)
                self.assertIn(rec["route"], census.ROUTES, key)
                self.assertNotIn("settings", rec["fields"], key)               # no settings= word on any line
                self.assertEqual(rec["mode"], "off" if route == "default" else route, key)
                m = TAIL_RE.search(line)
                self.assertIsNotNone(m, key)
                if key == "kernels_default_compiled":
                    self.assertRegex(m["compile"], r"^engaged\[graphs=[1-9]\d*\]$")
                else:
                    self.assertEqual(m["compile"], "off", key)
                self.assertNotIn("unknown", line, key)
            elif key == "peak":
                self.assertEqual(len(census.parse_peak_lines([line])), 1)

    def test_active_examples(self):
        for key in ("active_exact", "active_big"):
            line = report.EXAMPLES[key]
            self.assertTrue(line.startswith(f"{report.PREFIX} ACTIVE mode={key[len('active_'):]} form=process switches="), line)
            self.assertTrue(line.endswith(" fallbacks=none"), line)


class TestActiveLineNamesFallbacks(unittest.TestCase):
    def test_last_word(self):
        rep = {"active": True, "mode": "exact", "form": "inproc", "switches": "BG_GRAPH=graph", "boltzgen_version": "0.3.2", "gpu": None,
               "levers_applied": ["inproc", "graph_sampler"], "levers_fallback": [], "dropped": []}
        self.assertTrue(report.activation_line(rep).endswith(" levers=inproc,graph_sampler fallbacks=none"), report.activation_line(rep))
        rep = dict(rep, levers_fallback=["hoist", "fastinit"], partial=True)
        line = report.activation_line(rep)
        self.assertTrue(line.endswith(" fallbacks=hoist,fastinit"), line)
        self.assertIn(" fallback=hoist,fastinit ", line)
        self.assertNotIn("fallbacks=", report.activation_line(dict(rep, active=False, reason="x")))


if __name__ == "__main__":
    unittest.main()
