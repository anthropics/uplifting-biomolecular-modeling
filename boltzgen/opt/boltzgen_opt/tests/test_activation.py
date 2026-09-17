"""Activation on the kits' own bytes (stubbed upstream, real torch on CPU), each scenario in a fresh interpreter:
the in-process form applies the four in-process levers through the kit modules' own import-time patches and reports `inproc` as the
runner's; idempotent; a second mode is refused; late activation after `import boltzgen` is allowed, after a model instance or an
already-imported kit module it is refused by name; the GPU gate is by memory (wrong card refused, FORCE reported); the env route
(.pth finder emulated by `import boltzgen_opt._autoload`) activates on the first `import boltzgen`, exits 3 on refusal, skips the
CPU steps, installs nothing when unset/off, refuses a non-mode at the trigger; the exit tally prints the kit modules' own counters."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import re

from boltzgen_opt import modes, stack
from boltzgen_opt.tests import _stubs

PRELUDE = "import os, sys, json\nimport boltzgen_opt\nfrom boltzgen_opt import stack, modes, report\n"


def clean_lines(out):
    """stdout with every kit module's own printed banner dropped (``[<module>] ...``, printed at import/install time by whichever
    levers the mode's activation passes through, including hl_levers' own atexit report) — the lines the test itself printed,
    in the order it printed them."""
    return [ln for ln in out.strip().splitlines() if not re.match(r"^\[\w+\]", ln)]


class TestActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_act_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_code(self, code, **env_more):
        env = _stubs.clean_env(self.site, **env_more)
        return _stubs.run_py(PRELUDE + code, env)

    def test_exact_applies_the_kit_levers_in_process(self):
        rc, out, err = self.run_code(r'''
rep = boltzgen_opt.enable("exact")
print(json.dumps({k: rep.get(k) for k in ("active", "levers_applied", "levers_fallback", "levers_unavailable", "gpu_gate", "instance_counter", "switches", "form")}))
print(json.dumps({k: os.environ.get(k) for k in ("BG_GRAPH", "XA_FAST_INIT", "XA_HOIST")}))
print(json.dumps(sys.path[:2]))
import xa_fastinit, xa_hoist, bg_graph_patch, bg_hook
print(json.dumps([xa_fastinit.STATS["enabled"], xa_hoist.STATS["enabled"], bg_graph_patch.STATS["mode"], bool(bg_hook._acc.get("seed_fix_active"))]))
print(json.dumps(boltzgen_opt.enable("exact") is rep))
report.emit(boltzgen_opt.status())
''')
        self.assertEqual(rc, 0, err)
        lines = clean_lines(out)
        rep = json.loads(lines[0])
        self.assertTrue(rep["active"])
        self.assertEqual(rep["levers_applied"], ["graph_sampler", "fastinit", "hoist", "async_writer"])
        self.assertEqual(rep["levers_fallback"], [])
        self.assertEqual(rep["levers_unavailable"], ["inproc"])
        self.assertEqual((rep["gpu_gate"], rep["instance_counter"], rep["form"]), ("ok", "new-wrap", "inproc"))
        self.assertEqual(rep["switches"], "BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1")
        self.assertEqual(json.loads(lines[1]), {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1"})
        self.assertTrue(json.loads(lines[2])[0].endswith(os.path.join("forward", "xattempt_addon", "src")))
        self.assertTrue(json.loads(lines[2])[1].endswith(os.path.join("forward", "fast_inference", "src")))
        self.assertEqual(json.loads(lines[3]), [True, True, "graph", True])
        self.assertTrue(json.loads(lines[4]))
        self.assertIn("[boltzgen-opt] ACTIVE mode=exact form=inproc switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1 boltzgen=0.3.2 gpu=NVIDIA H100 80GB HBM3(81559MiB) levers=graph_sampler,fastinit,hoist,async_writer unavailable=inproc", err)
        self.assertIn("[bg_graph_patch] mode=graph", err)
        self.assertIn("[xa_hoist] installed (with partner graph-sampler integration)", err)
        # the exit tally: the kit modules' own counters, at interpreter exit
        tally = [ln for ln in err.splitlines() if "EXIT tally" in ln]
        self.assertEqual(len(tally), 1)
        for key in ("xa_fastinit.enabled=True", "xa_hoist.enabled=True", "bg_graph_patch.mode=graph", "bg_hook.seed_fix_active=1"):
            self.assertIn(key, tally[0])

    def test_an_activated_process_arms_its_children_with_the_modes_census_payload(self):
        """stack.activate exports BOLTZGEN_OPT_KERNELS for the processes it launches (upstream's per-step children re-execute site and
        arm the census there): the payload of THIS mode — route = the mode, expect on, no step/item words — so a child that cannot import
        the accelerator library refuses by name (kernels.refuse_if_absent reads mode and expect from it)."""
        rc, out, err = self.run_code(r'''
from boltzgen_opt import census
before = os.environ.get(stack.ENV_KERNELS)
rep = boltzgen_opt.enable("exact")
print(json.dumps({"before": before, "active": rep["active"], "payload": census.parse_payload(os.environ[stack.ENV_KERNELS])}))
''')
        self.assertEqual(rc, 0, err)
        rec = json.loads(clean_lines(out)[0])
        self.assertEqual(rec, {"before": None, "active": True, "payload": {"mode": "exact", "route": "exact", "expect": "on"}}, rec)

    def test_exit_tally_says_so_when_nothing_loaded(self):
        rc, out, err = self.run_code("report.register_exit_tally()\n")
        self.assertEqual(rc, 0)
        self.assertIn("EXIT tally: no lever module loaded in this process", err)
        rc, out, err = self.run_code("report.register_exit_tally()\n", BOLTZGEN_OPT_QUIET="1")
        self.assertNotIn("EXIT tally", err)

    def test_second_mode_refused_off_then_exact_allowed(self):
        rc, out, err = self.run_code(r'''
r0 = boltzgen_opt.enable("off")
print(json.dumps([r0["active"], r0["reason"].startswith("mode off")]))
r1 = boltzgen_opt.enable("exact")
print(json.dumps(r1["active"]))
try:
    boltzgen_opt.enable("off"); print("NO-ERROR")
except boltzgen_opt.ActivationError as e:
    print("REFUSED " + str(e))
''')
        self.assertEqual(rc, 0, err)
        lines = clean_lines(out)
        self.assertEqual(json.loads(lines[0]), [False, True])
        self.assertTrue(json.loads(lines[1]))
        self.assertTrue(lines[2].startswith("REFUSED already active in mode 'exact'"))

    def test_fast_applies_exacts_levers_and_its_own_two_in_process(self):
        rc, out, err = self.run_code(r'''
import json, sys
rep = boltzgen_opt.enable("fast")
fl = sys.modules.get("fl_levers")
print("RECORD " + json.dumps({"active": rep["active"], "mode": rep["mode"], "applied": rep.get("levers_applied"), "fallback": rep.get("levers_fallback"),
                             "stats": {k: fl.STATS[k] for k in ("dedup_enabled", "attn_enabled")} if fl else None,
                             "modules": sorted(n for n in sys.modules if n.startswith(("xa_", "bg_", "fl_")))}))
''')
        self.assertEqual(rc, 0, err)
        rec = next(json.loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
        self.assertTrue(rec["active"], (rec, err[-800:]))
        self.assertEqual(rec["mode"], "fast")
        self.assertTrue({"fastinit", "hoist", "graph_sampler", "cond_dedup", "attn_bf16"} <= set(rec["applied"]), rec)
        self.assertEqual(rec["stats"], {"dedup_enabled": True, "attn_enabled": True})
        self.assertIn("fl_levers", rec["modules"])
        self.assertIn("[fl_levers] cond_dedup installed sites=cond,adaln,layer_gate,transition_gate strategy=F7.row_dedup class=tolerance graph_integration=1", out)   # in the mode's process the partner graph sampler is importable: its step body is marked
        self.assertIn("[fl_levers] attn_bf16 installed core=opt_core.attn.sdpa_bias strategy=F5.flash_attn_dense compute=bf16 backend=cudnn mask_align=16 gate=any", out)
        self.assertIn("[boltzgen-opt] ACTIVE mode=fast form=inproc", err)

    def test_late_activation_after_import_allowed_after_instance_refused(self):
        rc, out, err = self.run_code(r'''
import boltzgen, boltzgen.model.models.boltz as B
rep = boltzgen_opt.enable("exact")
print(json.dumps([rep["active"], stack.instance_count()]))
m = B.Boltz()
print(json.dumps(stack.instance_count()))
''')
        self.assertEqual(rc, 0, err)
        lines = clean_lines(out)
        self.assertEqual(json.loads(lines[0]), [True, 0])
        self.assertEqual(json.loads(lines[1]), 1)
        rc, out, err = self.run_code(r'''
import boltzgen.model.models.boltz as B
m = B.Boltz()
rep = boltzgen_opt.enable("exact")
print(json.dumps([rep["active"], rep["reason"]]))
try:
    boltzgen_opt.enable("exact", strict=True); print("NO-ERROR")
except boltzgen_opt.ActivationError as e:
    print("STRICT " + str(e))
print(json.dumps(sorted(n for n in sys.modules if n.startswith(("xa_", "bg_")))))
''')
        self.assertEqual(rc, 0, err)
        lines = clean_lines(out)
        active, reason = json.loads(lines[0])
        self.assertFalse(active)
        self.assertIn("1 Boltz model instance(s) already exist in this process", reason)
        self.assertTrue(lines[1].startswith("STRICT 1 Boltz model instance"))
        self.assertEqual(json.loads(lines[2]), [])                           # nothing of the kit was imported by a refused activation

    def test_kit_module_already_imported_refused(self):
        kit_src = os.path.join(_stubs.opt_home(), "forward", "fast_inference", "src")
        rc, out, err = self.run_code(r'''
import bg_hook
rep = boltzgen_opt.enable("exact")
print(json.dumps([rep["active"], rep["reason"]]))
''', extra_path=kit_src, BG_GRAPH="off")
        self.assertEqual(rc, 0, err)
        active, reason = json.loads(out.strip().splitlines()[0])
        self.assertFalse(active)
        self.assertIn("kit module(s) already imported in this process (bg_hook,sitecustomize)", reason)   # the kit's own route: PYTHONPATH -> sitecustomize -> bg_hook
        self.assertIn("the package does not re-activate", reason)

    def test_gpu_gate_by_capability(self):
        """The card is never a gate: the configured H100 80GB (the pinned card, stock/PINS.json) is `ok` with no extra line; every other card is
        `noted` — active, one NOTE line naming it, which also says when the card is below the tested floor (stock/PINS.json
        gpu_of_record.min_compute_capability = 8.0) or its capability cannot be read; only no GPU at all is a refusal (`none`)."""
        code = r'''
rep = boltzgen_opt.enable("exact")
print(json.dumps([rep["active"], rep.get("gpu_gate"), rep.get("reason")]))
'''
        rc, out, err = self.run_code(code)                                   # the pinned card (NVIDIA H100 80GB HBM3, 81559 MiB): ok, no NOTE
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((active, gate, reason), (True, "ok", None))
        self.assertNotIn("NOTE", err)
        self.assertNotIn("gpu_gate=", err)
        rc, out, err = self.run_code(code, gpu="NVIDIA A100-SXM4-80GB,81920,8.0")   # another card at the floor: admitted, named once
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((active, gate, reason), (True, "noted", None))
        self.assertIn("[boltzgen-opt] NOTE gpu=NVIDIA A100-SXM4-80GB mem=81920MiB differs from the pinned card NVIDIA H100 80GB HBM3 (81559MiB, stock/PINS.json); proceeding (capability sm_80)", clean_lines(err))
        rc, out, err = self.run_code(code, gpu="NVIDIA H200,143771,9.0")
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((active, gate), (True, "noted"))
        rc, out, err = self.run_code(code, gpu="NVIDIA H100 80GB PCIe,81559,9.0")   # the pinned memory under another name: admitted, noted (the name is never a refusal)
        self.assertEqual(tuple(json.loads(clean_lines(out)[0])[:2]), (True, "noted"))
        rc, out, err = self.run_code(code, gpu="Tesla V100-SXM2-32GB,32768,7.0")    # below the tested floor: admitted, the NOTE says so — the card is never a refusal
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((rc, active, gate, reason), (0, True, "noted", None), err)
        self.assertIn("[boltzgen-opt] NOTE gpu=Tesla V100-SXM2-32GB mem=32768MiB differs from the pinned card NVIDIA H100 80GB HBM3 (81559MiB, stock/PINS.json) and its "
                      "capability sm_70 is below the tested floor 8.0; proceeding — a lever that cannot run on this card names itself on its own line", clean_lines(err))
        self.assertNotIn("gpu_gate=", err); self.assertNotIn("NOT ACTIVE", err)
        rc, out, err = self.run_code(code, gpu="NVIDIA H200,143771")                # a capability that cannot be read: admitted, named, never guessed
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((active, gate, reason), (True, "noted", None), err)
        self.assertIn("capability unreadable", err); self.assertNotIn("NOT ACTIVE", err)
        rc, out, err = self.run_code(code, gpu="")
        active, gate, reason = json.loads(clean_lines(out)[0])
        self.assertEqual((active, gate), (False, "none"))
        self.assertIn("no GPU visible", reason)

    def test_dry_run_applies_nothing(self):
        rc, out, err = self.run_code(r'''
rep = boltzgen_opt.enable("exact", dry_run=True)
print(json.dumps([rep["active"], rep["dry_run"], rep.get("reason"), rep.get("would_export"), sorted(n for n in sys.modules if n.startswith(("xa_", "bg_"))), os.environ.get("BG_GRAPH"), boltzgen_opt.status()["active"]]))
''')
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out.strip()), [False, True, None, {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1", "HL_ASYNC_WRITER": "1"}, [], None, False])
        self.assertIn("[boltzgen-opt] DRY-RUN", err) if "DRY-RUN" in err else None

    def test_wrong_pin_refused(self):
        site2 = _stubs.materialize(os.path.join(self.tmp, "v2"))
        meta = os.path.join(site2, "boltzgen-0.3.2.dist-info")
        os.rename(meta, os.path.join(site2, "boltzgen-0.3.1.dist-info"))
        open(os.path.join(site2, "boltzgen-0.3.1.dist-info", "METADATA"), "w").write("Metadata-Version: 2.1\nName: boltzgen\nVersion: 0.3.1\n")
        env = _stubs.clean_env(site2)
        rc, out, err = _stubs.run_py(PRELUDE + 'rep = boltzgen_opt.enable("exact"); print(json.dumps([rep["active"], rep["reason"]]))\n', env)
        self.assertEqual(rc, 0, err)
        active, reason = json.loads(out.strip())
        self.assertFalse(active)
        self.assertIn("boltzgen 0.3.1 is installed, the pin is 0.3.2", reason)

    # ---------------------------------------------------------------------------------------------------------- env route
    def test_env_route_activates_on_first_import_of_boltzgen(self):
        code = "import boltzgen_opt._autoload\nimport sys\nprint('finder', any(type(f).__name__ == 'Finder' for f in sys.meta_path))\nimport boltzgen\nimport boltzgen_opt\nprint('active', boltzgen_opt.status()['active'], boltzgen_opt.status().get('levers_applied'))\n"
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 0, err)
        self.assertIn("finder True", out)
        self.assertIn("active True ['graph_sampler', 'fastinit', 'hoist', 'async_writer']", out)
        self.assertIn("[boltzgen-opt] ACTIVE mode=exact form=inproc", err)
        self.assertIn("EXIT tally", err)

    def test_env_route_refusal_exits_3(self):
        code = "import boltzgen_opt._autoload\nimport boltzgen\nprint('REACHED')\n"
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", gpu=""))   # no GPU visible: the one card condition that cannot activate
        self.assertEqual(rc, 3, err)
        self.assertNotIn("REACHED", out)
        self.assertIn("[boltzgen-opt] NOT ACTIVE: no GPU visible", err)
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", gpu="Tesla V100-SXM2-32GB,32768,7.0"))   # a card below the tested floor is named, never refused
        self.assertEqual(rc, 0, err)
        self.assertIn("REACHED", out); self.assertIn("below the tested floor 8.0; proceeding", err); self.assertNotIn("NOT ACTIVE", err)

    def test_env_route_partial_refuses_by_name(self):
        """A mode is all of its levers: a PARTIAL activation on the env route (a lever of the mode could not run — here classify() is made to say
        so) is a refusal by name — `partial=hoist` on the activation line, the NOT ACTIVE partial line, exit 3 before the trigger import
        completes; there is no opt-out. A library caller (enable()) gets the report with `partial` and the line, never an exit; strict raises."""
        patch = ("import boltzgen_opt._autoload, boltzgen_opt\nfrom boltzgen_opt import stack\n_c = stack.classify\n"
                 "stack.classify = lambda res: dict(_c(res), levers_fallback=['hoist'], fallback_reasons={'hoist': 'test: fell back'}, partial=True)\n")
        code = patch + "import boltzgen\nprint('REACHED')\n"
        line = "[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: test: fell back); a mode is all of its levers: exit 3"
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 3, err)
        self.assertNotIn("REACHED", out)
        self.assertIn(" partial=hoist", err); self.assertIn(" fallbacks=hoist", err); self.assertIn(line, err)
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", BOLTZGEN_OPT_ALLOW_PARTIAL="1"))   # yesterday's opt-out is no switch: nothing runs under the mode's name with a subset
        self.assertEqual(rc, 3, err); self.assertNotIn("REACHED", out); self.assertNotIn("allow_partial", err)
        rc, out, err = _stubs.run_py(patch + "rep = boltzgen_opt.enable('exact')\nprint(rep['active'], rep['partial'], 'allow_partial' in rep)\n", _stubs.clean_env(self.site))
        self.assertEqual(rc, 0, err)
        self.assertEqual(clean_lines(out)[-1], "True True False")
        self.assertIn(line, err)
        rc, out, err = _stubs.run_py(patch + "try:\n    boltzgen_opt.enable('exact', strict=True)\nexcept boltzgen_opt.ActivationError as e:\n    print('RAISED', 'partial activation' in str(e))\n", _stubs.clean_env(self.site))
        self.assertEqual((rc, clean_lines(out)[-1]), (0, "RAISED True"), err)

    def test_a_lever_that_falls_back_during_the_run_ends_the_process_3(self):
        """The in-process routes (BOLTZGEN_OPT=<mode> at the trigger import; enable()) judge the run at interpreter exit as the launching verb
        judges a child from its lines: a planned lever whose module's records show it stopped serving during the run (stack.runtime_fallbacks:
        here xa_hoist disabled itself) makes the process's outputs not the mode's — one NOT ACTIVE partial line and exit 3, after the run
        (stack._exit_verdict). No fallback: the process's own exit. No model built after the activation: nothing ran under the mode's name, the
        exit is the caller's. An uncaught exception keeps its own status (1). The process form's child (BOLTZGEN_OPT_HANDOVER) is judged by
        its launching verb, not by itself."""
        act = "import boltzgen_opt._autoload, boltzgen, sys\nfrom boltzgen.model.models.boltz import Boltz\n"
        build = "m = Boltz.load_from_checkpoint('x')\n"
        fall = "sys.modules['xa_hoist'].STATS['disabled_reason'] = 'self-check mismatch at layer 3 (test)'\n"
        line = ("[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: [xa_hoist] disabled_reason: self-check mismatch at layer 3 (test)) "
                "in the run: this process's outputs are not the mode's; a mode is all of its levers: exit 3")
        rc, out, err = _stubs.run_py(act + build + fall + "print('RAN')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 3, err); self.assertIn("RAN", out); self.assertIn(line, err.splitlines())
        self.assertLess(err.find("] EXIT tally:"), err.find("] NOT ACTIVE: partial activation"))      # the kit's exit lines first, the verdict last
        rc, out, err = _stubs.run_py(act + build + "print('RAN')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)
        rc, out, err = _stubs.run_py(act + fall + "print('RAN')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)                                # no model was built: no model work happened under the mode's name
        rc, out, err = _stubs.run_py(act + build + fall + "raise RuntimeError('the run failed')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 1, err); self.assertNotIn("NOT ACTIVE: partial", err)
        rc, out, err = _stubs.run_py(act + build + fall + "print('RAN')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", BOLTZGEN_OPT_HANDOVER="1"))
        self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)
        from boltzgen_opt import census
        armed = _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", BOLTZGEN_OPT_KERNELS=census.payload("exact", "on"))   # a model process of the environment route: the census armed at interpreter start, before the activation
        rc, out, err = _stubs.run_py(act + build + fall + "print('RAN')\n", armed)
        self.assertEqual(rc, 3, err)
        lines = err.splitlines()
        kern = [i for i, ln in enumerate(lines) if ln.startswith("[boltzgen-opt") and " KERNELS route=exact " in ln]
        self.assertEqual(len(kern), 1, err)                                                     # this process's census line is printed (once) before the verdict ends the process
        self.assertLess(kern[0], lines.index(line))
        rc, out, err = _stubs.run_py(act + "from boltzgen.task.predict.predict import Predict\nclass D: num_workers = 0\nPredict(data=D()).run()\n", armed)
        self.assertEqual(rc, 3, err)                                                             # the step gate ends the process the same way: the census line first
        lines = err.splitlines()
        self.assertEqual(sum(ln.startswith("[boltzgen-opt") and " KERNELS route=exact " in ln for ln in lines), 1, err)
        self.assertLess(next(i for i, ln in enumerate(lines) if " KERNELS route=exact " in ln), next(i for i, ln in enumerate(lines) if "NOT ACTIVE: mode=exact cannot serve this step" in ln))
        replay = "sys.modules['xa_fastinit']._replayed(7, 'COVERAGE INCOMPLETE (1 missing params, 0 shape mismatches): replayed 7 stock inits in order; the lever is a no-op for this load')\n"
        rc, out, err = _stubs.run_py(act + build + replay, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 3, err)                                                             # a load whose stock initialisers were replayed is fast-init not serving: both judges read it (STATS fallback_reason; the COVERAGE INCOMPLETE line)
        self.assertIn("NOT ACTIVE: partial activation — fastinit fell back (fastinit: [xa_fastinit] fallback_reason: COVERAGE INCOMPLETE (1 missing params, 0 shape mismatches): replayed 7 stock inits in order; the lever is a no-op for this load) in the run", err)
        self.assertRegex("[xa_fastinit] COVERAGE INCOMPLETE (1 missing params, 0 shape mismatches) -> replayed 7 stock inits in order; lever is a no-op for this load", stack.RUNNER_DISABLED_LINES["fastinit"])
        rc, out, err = _stubs.run_py(act + build + "import boltzgen_opt.stack as S\nS.runtime_fallbacks = None\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))   # a verdict that cannot be reached is a refusal, never a silent pass
        self.assertEqual(rc, 3, err); self.assertIn("[boltzgen-opt] NOT ACTIVE: exit verdict failed: TypeError(", err)
        rc, out, err = _stubs.run_py("import boltzgen_opt, sys\nrep = boltzgen_opt.enable('fast')\nprint(rep['step_gate'], rep['exit_verdict'])\n"
                                     "from boltzgen.model.models.boltz import Boltz\n" + build +
                                     "sys.modules['fl_levers'].STATS['attn_disabled_reason'] = 'no_kernel:cudnn'\n", _stubs.clean_env(self.site))
        self.assertEqual(rc, 3, err); self.assertEqual(clean_lines(out)[-1], "True True")
        self.assertRegex(err, r"NOT ACTIVE: partial activation — attn_bf16,attn_cudnn fell back \(attn_bf16: \[fl_levers\] attn_disabled_reason: no_kernel:cudnn; attn_cudnn: \[fl_levers\] attn_disabled_reason: no_kernel:cudnn\) in the run")

    def test_runtime_fallbacks_read_each_modules_own_records(self):
        """stack.runtime_fallbacks over the registry's `fell_back` grammar: a `set` record is the module's own reason; a module's gate() problems
        go to the lever whose prefix they carry, the unclaimed ones to the module's lever with the empty prefix (fl_levers: cond_dedup)."""
        import types
        from unittest import mock
        fl = types.ModuleType("fl_levers"); fl.STATS = {"attn_disabled_reason": None, "dit_disabled_reason": "route refused: sha"}
        fl.gate = lambda: ["attn_bf16: 3 eligible call(s), served=0", "s_path: fallback=2 of 10 calls", "dit_fused: served=0"]
        hl = types.ModuleType("hl_levers"); hl.STATS = {}; hl.gate = lambda: ["async_writer: 1 write(s) failed"]
        gp = types.ModuleType("bg_graph_patch"); gp.STATS = {"capture_error": "Traceback (most recent call last):\n  ...\nRuntimeError: capture failed: op not capturable"}
        res = modes.resolve("fast", stack.opt_home())
        with mock.patch.dict(sys.modules, {"fl_levers": fl, "hl_levers": hl, "bg_graph_patch": gp}):
            sys.modules.pop("xa_hoist", None); sys.modules.pop("xa_fastinit", None)
            fallen, why = stack.runtime_fallbacks(res)
        self.assertEqual(fallen, ["graph_sampler", "async_writer", "cond_dedup", "attn_bf16", "dit_fused"])   # the mode table's lever order; hoist / fastinit: module absent, nothing to read; attn_cudnn: no disabled reason set
        self.assertEqual(why["graph_sampler"], "[bg_graph_patch] capture_error: RuntimeError: capture failed: op not capturable")
        self.assertEqual(why["cond_dedup"], "[fl_levers] GATE-FAIL s_path: fallback=2 of 10 calls")
        self.assertEqual(why["attn_bf16"], "[fl_levers] GATE-FAIL attn_bf16: 3 eligible call(s), served=0")
        self.assertEqual(why["dit_fused"], "[fl_levers] dit_disabled_reason: route refused: sha")           # the module's own reason first; its gate words would say the same
        self.assertEqual(why["async_writer"], "[hl_levers] GATE-FAIL async_writer: 1 write(s) failed")
        with mock.patch.dict(sys.modules, {"hl_levers": hl}):
            for m in ("fl_levers", "bg_graph_patch", "xa_hoist", "xa_fastinit"):
                sys.modules.pop(m, None)
            self.assertEqual(stack.runtime_fallbacks(modes.resolve("exact", stack.opt_home())), (["async_writer"], {"async_writer": "[hl_levers] GATE-FAIL async_writer: 1 write(s) failed"}))

    def test_a_step_a_planned_lever_cannot_serve_is_refused_before_its_model_loads(self):
        """stack.arm_step_gate wraps upstream's Predict.run in every activated process: fast-init cannot serve a step whose DataLoader runs
        in-process (num_workers 0: xa_fastinit.unserved_reason, its RNG contract) — the mode refuses the step by name, exit 3, before the model
        loads and before the module's own per-step disable; a step it can serve runs."""
        act = "import boltzgen_opt._autoload, boltzgen\nfrom boltzgen.task.predict.predict import Predict\n"
        step = "class D: num_workers = {nw}\nprint('BEFORE', flush=True)\nt = Predict(data=D()); t.debug = {debug}\nr = t.run()\nprint('AFTER', r, flush=True)\n"
        shift = "the featurizer's RNG stream would shift from stock's (fast-init skips the initialisers' draws on numpy's global stream, which an in-process DataLoader then reads)"
        tail = "; a mode is all of its levers, nothing of the step ran: exit 3 (the stock path is --mode off / BOLTZGEN_OPT=off)"
        for env_more in ({}, {"BOLTZGEN_OPT_HANDOVER": "1"}):                                    # the environment route's model process, and the `design` verb's child (its launching verb then exits 3 on the child's 3)
            rc, out, err = _stubs.run_py(act + step.format(nw=0, debug=False), _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", **env_more))
            self.assertEqual(rc, 3, err)
            self.assertEqual([ln for ln in clean_lines(out) if ln in ("BEFORE",) or ln.startswith("AFTER")], ["BEFORE"])
            self.assertIn(f"[boltzgen-opt] NOT ACTIVE: mode=exact cannot serve this step: fastinit — in-process DataLoader (num_workers=0): {shift}{tail}", err.splitlines(), env_more)
            rc, out, err = _stubs.run_py(act + step.format(nw=2, debug=True), _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", **env_more))   # upstream's `debug` flag: Predict.run sets num_workers 0 itself, after the entry — refused on the flag
            self.assertEqual(rc, 3, err); self.assertNotIn("AFTER", out)
            self.assertIn(f"[boltzgen-opt] NOT ACTIVE: mode=exact cannot serve this step: fastinit — in-process DataLoader (debug=true: upstream sets num_workers 0 inside Predict.run): {shift}{tail}", err.splitlines(), env_more)
            rc, out, err = _stubs.run_py(act + step.format(nw=2, debug=False), _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", **env_more))
            self.assertEqual(rc, 0, err); self.assertIn("AFTER ran", out); self.assertNotIn("NOT ACTIVE", err)
        src = open(os.path.join(stack.kit_src(modes.KIT_XATTEMPT), "xa_fastinit.py"), encoding="utf-8").read()
        self.assertNotIn("Predict.run = ", src)                                                  # the package's step gate is the one reader of unserved_reason: the add-on wraps no step itself

    def test_env_route_a_non_mode_refused_at_the_trigger(self):
        rc, out, err = _stubs.run_py("import boltzgen_opt._autoload\nprint('START-OK')\nimport boltzgen\nprint('REACHED')\n", _stubs.clean_env(self.site, BOLTZGEN_OPT="faster"))
        self.assertEqual(rc, 3)
        self.assertIn("START-OK", out)
        self.assertNotIn("REACHED", out)
        self.assertIn("NOT ACTIVE: BOLTZGEN_OPT='faster' is not a mode (off|exact|fast|big)", err)

    def test_env_route_unset_or_off_installs_nothing(self):
        code = "import boltzgen_opt._autoload\nimport sys\nprint(any(type(f).__name__ == 'Finder' for f in sys.meta_path), 'torch' in sys.modules)\nimport boltzgen\nprint(sorted(n for n in sys.modules if n.startswith(('xa_', 'bg_'))))\n"
        for value in (None, "", "off"):
            env = _stubs.clean_env(self.site)
            if value is not None:
                env["BOLTZGEN_OPT"] = value
            rc, out, err = _stubs.run_py(code, env)
            self.assertEqual(rc, 0, err)
            self.assertEqual(out.strip().splitlines(), ["False False", "[]"], value)

    def test_env_route_skips_cpu_steps(self):
        code = "import boltzgen_opt._autoload\nimport boltzgen, sys\nprint(sorted(n for n in sys.modules if n.startswith(('xa_', 'bg_'))), 'torch' in sys.modules)\n"
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", BOLTZGEN_PIPELINE_STEP="analysis"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.strip(), "[] False")
        self.assertIn("NOTE CPU step 'analysis'", err); self.assertNotIn("NOT ACTIVE", err)   # nothing was asked of a CPU step: a note, not a refusal
        rc, out, err = _stubs.run_py(code, _stubs.clean_env(self.site, BOLTZGEN_OPT="exact", BOLTZGEN_PIPELINE_STEP="design"))
        self.assertEqual(rc, 0, err)
        self.assertIn("'xa_fastinit'", out)

    def test_in_process_form_keeps_the_env_switch_and_hands_the_mode_to_children(self):
        rc, out, err = self.run_code(r'''
rep = boltzgen_opt.enable("exact")
print(json.dumps([rep["active"], rep["dropped_env"], os.environ.get("BOLTZGEN_OPT"), os.environ["PYTHONPATH"].split(os.pathsep)[:2] == rep["pythonpath"][:2], rep.get("hook_first")]))
''', BOLTZGEN_OPT="exact", BG_GRAPH="graph", XA_HOIST="0")
        self.assertEqual(rc, 0, err)
        active, dropped, env_mode, pp, hook_first = json.loads(clean_lines(out)[0])
        self.assertEqual((active, env_mode, pp, hook_first), (True, "exact", True, False))
        self.assertEqual(dropped, ["XA_HOIST"])                                      # a caller value equal to the mode's own export is not "dropped"
        rc, out, err = self.run_code('boltzgen_opt.enable("exact"); print(os.environ.get("BOLTZGEN_OPT"))\n')
        self.assertEqual(rc, 0, err)
        self.assertEqual(clean_lines(out)[-1], "exact")                              # the explicit route exports the mode for the process's children too

    # ------------------------------------------------------------------------------- the env route across upstream's process tree
    def _upstream_topology(self, mode_env: dict, child_steps=("design", "inverse_folding", "analysis")):
        """The shape of `BOLTZGEN_OPT=<mode> boltzgen run`: the parent CLI process imports `boltzgen` (the finder fires there), then starts
        one interpreter per pipeline step with BOLTZGEN_PIPELINE_STEP set and its own environment inherited (cli/boltzgen.py l.806-817);
        the `analysis` step additionally starts a multiprocessing 'spawn' pool (task/analyze/analyze.py l.195). Each step process runs
        the stub step and records what it saw; the parent prints the records. Needs the installed .pth in the children."""
        if not _stubs.package_installed():
            raise unittest.SkipTest("boltzgen_opt is not installed in this interpreter (upstream's step processes activate through the "
                                    "installed boltzgen_opt_autoload.pth): run in a venv with `pip install -e boltzgen/opt` to exercise this")
        parent = r'''
import json, os, subprocess, sys
import boltzgen                                           # the finder fires here, as in `boltzgen run` (boltzgen.cli.boltzgen:main)
child = r"""
import json, os, sys
import boltzgen.resources.main                            # a step's first import of its own (the stub step)
def seen():
    return {"step": os.environ.get("BOLTZGEN_PIPELINE_STEP"), "env_mode": os.environ.get("BOLTZGEN_OPT"),
            "kit_modules": sorted(n for n in sys.modules if n.startswith(("xa_", "bg_"))),
            "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None), "finder_left": any(type(f).__name__ == "Finder" for f in sys.meta_path)}
def pool_worker(_):
    return seen()
if __name__ == "__main__":
    rec = seen()
    import boltzgen_opt
    rec["status"] = {k: boltzgen_opt.status().get(k) for k in ("active", "levers_applied", "hook_first")}
    if rec["step"] == "analysis":
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            rec["pool_worker"] = pool.map(pool_worker, [0])[0]
    print("RECORD " + json.dumps(rec))
"""
script = os.path.join(sys.argv[1], "step_stub.py"); open(script, "w").write(child)
for step in sys.argv[2:]:
    os.environ["BOLTZGEN_PIPELINE_STEP"] = step
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    print(json.dumps({"step": step, "rc": r.returncode, "record": next((json.loads(ln[7:]) for ln in r.stdout.splitlines() if ln.startswith("RECORD ")), None)}))
'''
        env = _stubs.clean_env(self.site, **mode_env)
        env.pop("PYTHONPATH")                                                   # the children's route is the installed .pth: nothing on PYTHONPATH but what the parent exports
        pth = _stubs.site_pth(self.site)                                        # the stubs importable without PYTHONPATH (as test_stock_route does)
        try:
            rc, out, err = _stubs.run_py(parent, env, args=[self.tmp] + list(child_steps))
        finally:
            _stubs.remove_site_pth(pth)
        return rc, [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")], err

    def test_env_route_parent_hands_every_step_process_the_whole_line(self):
        rc, rows, err = self._upstream_topology({"BOLTZGEN_OPT": "exact"})
        self.assertEqual(rc, 0, err)
        self.assertEqual([r["step"] for r in rows], ["design", "inverse_folding", "analysis"])
        for r in rows:
            self.assertEqual(r["rc"], 0, err)
            rec = r["record"]
            self.assertEqual(rec["env_mode"], "exact", rec)                                # BOLTZGEN_OPT reaches every step process
            self.assertTrue(rec["sitecustomize"].startswith(_stubs.opt_home()), rec)      # every step starts on the partner's sitecustomize line
            self.assertFalse(rec["finder_left"], rec)
        for r in rows[:2]:                                                                  # the GPU steps: the whole line, completed after the hook
            rec = r["record"]
            self.assertEqual(rec["kit_modules"], ["bg_graph_patch", "bg_hook", "xa_fastinit", "xa_hoist"], rec)
            self.assertEqual(rec["status"], {"active": True, "levers_applied": ["graph_sampler", "fastinit", "hoist", "async_writer"], "hook_first": True}, rec)
        rec = rows[2]["record"]                                                             # the CPU step: the hook's own route only, as under the runner
        self.assertEqual(rec["kit_modules"], ["bg_graph_patch", "bg_hook"], rec)
        self.assertEqual(rec["status"]["active"], False)
        self.assertEqual(rec["pool_worker"]["kit_modules"], ["bg_graph_patch", "bg_hook"], rec)
        self.assertEqual(err.count("ACTIVE mode=exact form=inproc"), 3)                    # the parent + the two GPU steps, once each
        self.assertIn("levers=graph_sampler,fastinit,hoist,async_writer unavailable=inproc after=sitecustomize step=design", err)
        self.assertIn("after=sitecustomize step=inverse_folding", err)
        self.assertEqual(err.count("NOTE CPU step 'analysis'"), 1)                  # the step's process, not its spawn-pool worker

    def test_env_route_hook_first_refuses_a_line_that_is_not_the_modes(self):
        # a process started on the partner's route by hand (PYTHONPATH = the kits, BG_GRAPH unset: the kits' bare PYTHONPATH line, sampler off) with
        # BOLTZGEN_OPT=exact: the hook has already read a switch value that is not the mode's — refused by name, exit 3, nothing else runs
        kits = os.pathsep.join(stack.kit_src(k) for k in (modes.KIT_XATTEMPT, modes.KIT_PARTNER))
        code = "import boltzgen_opt._autoload\nimport site\nsite.main()\nimport boltzgen.resources.main\nprint('REACHED')\n"
        env = _stubs.clean_env(self.site, extra_path=kits, BOLTZGEN_OPT="exact")
        r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertNotIn("REACHED", r.stdout)
        self.assertIn("NOT ACTIVE: kit module(s) already imported in this process (bg_hook,sitecustomize) — kit switches BG_GRAPH=None,HL_ASYNC_WRITER=None,XA_FAST_INIT=None,XA_HOIST=None where the mode's line is BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1: the kits were activated by their own route (PYTHONPATH/sitecustomize); the package does not re-activate", r.stderr)
        # the same start under the mode's own line completes it (what an activated parent exports)
        env = _stubs.clean_env(self.site, extra_path=kits, BOLTZGEN_OPT="exact", BG_GRAPH="graph", XA_FAST_INIT="1", XA_HOIST="1", HL_ASYNC_WRITER="1")
        r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("REACHED", r.stdout)
        self.assertIn("levers=graph_sampler,fastinit,hoist,async_writer unavailable=inproc after=sitecustomize", r.stderr)



if __name__ == "__main__":
    unittest.main()
