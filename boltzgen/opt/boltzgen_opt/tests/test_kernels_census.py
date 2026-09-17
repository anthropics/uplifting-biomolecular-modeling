"""The accelerator census (census.py; the modes' policy over it: kernels.py): one KERNELS line per model process on every route (route words default | exact | fast | big),
words from the library objects bound at upstream's call sites and a per-call served / by-rule / fallback count. What the route expects
comes from the mode table and upstream's own `Using kernels:` line; `--use_kernels false` (upstream's switch) makes every route expect no
library call — accepted, never refused. The account of a run (`verdict`: `ok`, `findings`) is a RECORD in opt_manifest.json and never an
exit: a reference-path call the library's rules do not explain, an absent library on the stock route, a missing line are findings, exit
unchanged. The one refusal left is a KIT mode's child that cannot provide the kernels the mode routes through: NOT ACTIVE, exit 3, before
the step (`refuse_if_absent`; None for upstream's own processes). The library's documented dispatch rules (S_qo / seq <= threshold) are
`byrule`: counted, named."""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from boltzgen_opt import _autoload, census, codes, design, kernels, modes, report, stack, stock_design
from boltzgen_opt.tests import _stubs

SPEC = "entities:\n  - protein:\n      id: B\n      sequence: 60..80\n"
H100 = {"BOLTZGEN_STUB_KERNELS_CC": "9,0"}


def res(use, cc=(9, 0)):
    return {"use_kernels": use, "cc": cc, "line": f"Using kernels: {use} [device capability: {cc}]"}


class TestPeakLine(unittest.TestCase):
    """The per-item allocator PEAK line (report-only): torch's own maxima, reset at the census install (= item start), printed beside KERNELS."""

    def test_grammar_reset_at_item_start_and_parse(self):
        import types
        calls = []
        cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, current_device=lambda: 0,
                                     max_memory_allocated=lambda dev=None: int(21.5 * 2 ** 30), max_memory_reserved=lambda dev=None: 24 * 2 ** 30,
                                     memory_allocated=lambda dev=None: 2 ** 30, memory_reserved=lambda dev=None: 2 ** 31,
                                     reset_peak_memory_stats=lambda dev=None: calls.append("reset"))     # read through opt_core.mem.allocator (reset_peak / counters)
        fake_torch = types.SimpleNamespace(cuda=cuda)
        p = census.parse_payload(census.payload("exact", "on", step="design", item="L100"))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            census._peak_reset()                                            # what install() does first: item START
            self.assertEqual(calls, ["reset"])
            ln = census.peak_line(p)
        self.assertEqual(ln, "[boltzgen-opt exact] PEAK item=L100 alloc_gib=21.50 reserved_gib=24.00")
        m = re.search(r'PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$', ln)     # an outside reader's regex, verbatim
        self.assertEqual(m.groups(), ("L100", "21.50", "24.00"))
        (rec,) = census.parse_peak_lines(["noise", ln, "[boltzgen-opt exact] KERNELS route=exact step=design"])
        self.assertEqual((rec["mode"], rec["item"], rec["alloc_gib"], rec["reserved_gib"]), ("exact", "L100", 21.5, 24.0))
        cpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False))
        with mock.patch.dict(sys.modules, {"torch": cpu}):
            self.assertIsNone(census.peak_line(p))                          # a process that never initialised CUDA prints no PEAK line
            census._peak_reset()                                            # and resetting there is a no-op, never an error
        kv = kernels.census_of([ln], kernels.expectation("exact", res(True)), 0)
        self.assertEqual([r["item"] for r in kv["peak"]], ["L100"])         # the run record carries the PEAK records beside the account
        self.assertEqual(design.item_id("/x/runs/007_L100/"), "007_L100")


class TestTablesAndGrammar(unittest.TestCase):
    """No subprocess: the tables agree, the grammar round-trips, the account's rules, the one refusal's rule."""

    def test_exit_codes_are_0_1_2_3_and_nothing_else(self):
        self.assertEqual((codes.EXIT_OK, codes.EXIT_FAIL, codes.EXIT_USAGE, codes.EXIT_NOT_ACTIVE), (0, 1, 2, 3))
        for mod in (codes, kernels, stock_design, _autoload):
            self.assertFalse(hasattr(mod, "EXIT_KERNELS"), mod.__name__)     # the census never exits: no code of its own anywhere
        self.assertFalse(hasattr(codes, "EXIT_OOM")); self.assertFalse(hasattr(codes, "BIG_OOM_CHILD_RC"))
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, codes.EXIT_NOT_ACTIVE)   # the import-free copy the kit child's refusal exits with
        self.assertEqual(stock_design.EXIT_NOT_STOCK, codes.EXIT_NOT_ACTIVE)
        self.assertFalse(hasattr(kernels, "refusal_line")); self.assertFalse(hasattr(kernels, "KernelsRefused"))

    def test_env_name_is_one(self):
        self.assertEqual(_autoload.ENV_KERNELS, census.ENV_KERNELS)
        self.assertEqual(stack.ENV_KERNELS, census.ENV_KERNELS)
        self.assertIn(census.ENV_KERNELS, stack.pins()["stock_environment"]["must_be_absent_names"])   # a leaked payload never reaches the stock child unproven

    def test_every_mode_declares_every_accelerator(self):
        self.assertEqual(modes.UPSTREAM_ACCELERATORS, census.ACCELERATORS)
        for mode in modes.MODES:
            routed, replaced = modes.kernels_of(mode)
            self.assertEqual(set(routed) | set(replaced), set(census.ACCELERATORS), mode)
            self.assertFalse(set(routed) & set(replaced), mode)
            exp = kernels.expectation(mode, res(True))
            self.assertEqual(set(exp), {"expect", "words"}, mode)            # an expectation is a record: no `refusal` in it, on any route
            for a, w in exp["words"].items():
                self.assertIn(w.split(":", 1)[0], census.WORD_KINDS)
                self.assertEqual(w, "engaged", (mode, a))                    # no mode of this kit replaces an upstream accelerator: all route through cuEquivariance
        self.assertEqual(modes.kernels_of("off"), (census.ACCELERATORS, {}))   # the stock arm routes through everything upstream engages

    def test_routes_are_the_modes_and_default(self):
        """The route word: `default` for upstream alone (`off`), else the kit mode's own name; the four words are the whole set (census.ROUTES)."""
        self.assertEqual(census.ROUTES, ("default", "exact", "fast", "big"))
        self.assertEqual(census.route_of("off"), "default")
        for mode in ("exact", "fast", "big"):
            self.assertEqual(census.route_of(mode), mode)
        with self.assertRaises(ValueError):
            census.payload("off", "on", route="stock")                        # not a route word
        reads = set(stack.pins()["stock_environment"]["reads"])
        self.assertTrue({"CUEQ_TRIATTN_FALLBACK_THRESHOLD", "CUEQ_TRIMUL_FALLBACK_THRESHOLD"} <= reads)   # the library's own switches are names upstream's process reads (STOCK.md): a caller's reach the stock child

    def test_payload_round_trip(self):
        p = census.payload("off", "on", step="design")
        self.assertEqual(p, "mode=off,route=default,expect=on,step=design")
        self.assertEqual(census.parse_payload(p), {"mode": "off", "route": "default", "expect": "on", "step": "design"})
        p = census.parse_payload(census.payload("exact", "off:use_kernels=false", item="j7", seed=4))
        self.assertEqual(p, {"mode": "exact", "route": "exact", "expect": "off:use_kernels=false", "item": "j7", "seed": "4"})
        self.assertNotIn("settings", census.payload("big", "on", step="design", item="x", seed=0))   # no settings word anywhere: the payload, the line, the parsed record
        with self.assertRaises(ValueError):
            census.payload("off", "on", route="campaign")                     # not a route word
        with self.assertRaises(ValueError):
            census.parse_payload("mode=off,route=default")                    # expect is required
        import inspect
        self.assertEqual(list(inspect.signature(census.payload).parameters), ["mode", "expect", "step", "item", "seed", "route"])

    def test_upstream_resolution_and_expectation(self):
        tmp = tempfile.mkdtemp()
        try:
            log = os.path.join(tmp, "opt_configure.log")
            open(log, "w").write("noise\nUsing kernels: True [device capability: (9, 0)]\nmore\n")
            self.assertEqual(kernels.upstream_resolution(log), {"use_kernels": True, "cc": (9, 0), "line": "Using kernels: True [device capability: (9, 0)]"})
            open(log, "w").write("nothing here\n")
            self.assertIsNone(kernels.upstream_resolution(log)["line"])
            open(log, "w").write("Using kernels: False [device capability: (9, 0)]\n")
            exp = design.kernel_expectation("fast", tmp)                     # the caller's side reads the same log: `--use_kernels false` on an H100, a kit mode — expected off, recorded
            self.assertEqual((exp["expect"], exp["resolution"]["use_kernels"]), ("off:use_kernels=false", False))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        e = kernels.expectation("exact", res(True))
        self.assertEqual(e, {"expect": "on", "words": {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}})
        for mode in modes.MODES:                                                # `--use_kernels false` on an H100: every route expects the torch paths — accepted, named, no refusal
            e = kernels.expectation(mode, res(False, (9, 0)))
            self.assertEqual(e, {"expect": "off:use_kernels=false", "words": {"cueq_triatt": "off-by-route:use_kernels=False(use_kernels=false)", "cueq_trimul": "off-by-route:use_kernels=False(use_kernels=false)"}}, mode)
        e = kernels.expectation("off", res(False, (7, 5)))                   # upstream ships the kernels off below cc 8: named
        self.assertEqual(e["words"]["cueq_triatt"], "off-by-route:use_kernels=False(upstream(cc=7.5<8))")
        e = kernels.expectation("big", {"use_kernels": None, "cc": None, "line": None})   # no `Using kernels:` line: unknown, the words stay engaged, nothing refused
        self.assertEqual(e, {"expect": "unknown", "words": {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}})

    def test_line_grammar_and_verdict(self):
        eng = "[boltzgen-opt off] KERNELS route=default step=design cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=0,byrule=96(s_qo<=100:96)] cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=0,byrule=48(seq<=100:48)] cueq=0.11.1-cu13 caps=sm100f:0,sm107f:0,sm120f:0 cc=9.0 thresholds=triatt:100,trimul:100"
        self.assertRegex(eng, r"\] KERNELS route=(default|exact|fast|big) ")   # the cross-engine grep token
        parsed = census.parse_lines(["x", eng, "y"])
        self.assertEqual(len(parsed), 1)
        self.assertEqual((parsed[0]["mode"], parsed[0]["route"], parsed[0]["step"]), ("off", "default", "design"))
        self.assertNotIn("settings", parsed[0]); self.assertNotIn("settings", parsed[0]["fields"])
        self.assertEqual(parsed[0]["kinds"], {"cueq_triatt": "engaged", "cueq_trimul": "engaged"})
        exp = kernels.expectation("off", res(True))
        v = kernels.verdict(parsed, exp, 1)
        self.assertEqual(set(v), {"ok", "findings", "lines", "words", "expected"})   # the account's keys: `findings`, never `refusals`
        self.assertTrue(v["ok"], v)                                            # a rung entirely at S<=threshold: served=0, byrule=N — counted, named, PASSES
        self.assertEqual(v["findings"], [])
        short = kernels.verdict(parsed, exp, 2)                               # a model process without a census line is a finding
        self.assertFalse(short["ok"])
        self.assertIn("1 KERNELS line(s) where 2 model process(es) ran", short["findings"][0])
        fb = eng.replace("cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=0,byrule=96(s_qo<=100:96)]", "cueq_triatt=fallback:reference@s_qo=200:4[served=10,byrule=0,fallback=4]")
        v = kernels.verdict(census.parse_lines([fb]), exp, 1)
        self.assertFalse(v["ok"])
        self.assertEqual(v["findings"], ["route=default step=design cueq_triatt=fallback:reference@s_qo=200:4[served=10,byrule=0,fallback=4] (expected engaged)"])
        ab = eng.replace("cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=0,byrule=48(seq<=100:48)]", "cueq_trimul=absent:triton-import-failed")
        self.assertFalse(kernels.verdict(census.parse_lines([ab]), exp, 1)["ok"])
        off = kernels.expectation("off", res(False, (7, 5)))
        self.assertFalse(kernels.verdict(parsed, off, 1)["ok"])               # the gate said off and the line says engaged: kinds differ, a finding
        offline = eng.replace("cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=0,byrule=96(s_qo<=100:96)]", "cueq_triatt=off-by-route:use_kernels=False(use_kernels=false)").replace(
            "cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=0,byrule=48(seq<=100:48)]", "cueq_trimul=off-by-route:use_kernels=False(use_kernels=false)")
        self.assertTrue(kernels.verdict(census.parse_lines([offline]), kernels.expectation("off", res(False, (9, 0))), 1)["ok"])   # `--use_kernels false`: off expected, off found — the account passes
        na = ("[boltzgen-opt exact] KERNELS route=exact step=inverse_folding cueq_triatt=n/a-upstream:step=inverse_folding(inverse-fold network: no trunk, no call site) "
              "cueq_trimul=n/a-upstream:step=inverse_folding(inverse-fold network: no trunk, no call site) cueq=0.11.1-cu13 caps=sm100f:0,sm107f:0,sm120f:0 cc=9.0 thresholds=triatt:100,trimul:100")
        two = census.parse_lines([eng.replace("[boltzgen-opt off]", "[boltzgen-opt exact]").replace("route=default", "route=exact"), na])
        self.assertEqual(two[1]["kinds"], {"cueq_triatt": "n/a-upstream", "cueq_trimul": "n/a-upstream"})
        self.assertTrue(kernels.verdict(two, kernels.expectation("exact", res(True)), 2)["ok"])       # design engaged + inverse folding n/a: the pipeline passes
        self.assertFalse(kernels.verdict(census.parse_lines([na.replace("step=inverse_folding", "step=design", 1)]), kernels.expectation("exact", res(True)), 1)["ok"])   # the same word on a design process is a finding
        kv = kernels.census_of(["noise", fb], exp, 1, where=" (/r/opt_run.log)")
        self.assertEqual((kv["ok"], len(kv["findings"]), kv["peak"]), (False, 1, []))

    def test_refuse_if_absent_is_the_kit_modes_gate_only(self):
        """A KIT mode's model process that cannot provide the kernels the mode routes through (payload expect=on, the library absent) gets
        the NOT ACTIVE line — the caller (_autoload.arm_census) prints it and exits 3 before the step; upstream's own processes (mode=off)
        and a route that expects no library call (`--use_kernels false`) get None: an absent library there is upstream's to meet."""
        gone = {"cueq_triatt": "import cuequivariance_ops_torch failed", "cueq_trimul": "import cuequivariance_ops_torch failed"}
        with mock.patch.dict(census._STATE, {"absent": gone}):
            self.assertEqual(census.absent(), gone)
            self.assertIsNone(kernels.refuse_if_absent(census.parse_payload(census.payload("off", "on", step="design"))))
            self.assertIsNone(kernels.refuse_if_absent(census.parse_payload(census.payload("exact", "off:use_kernels=false", step="design"))))
            for mode in ("exact", "fast", "big"):
                line = kernels.refuse_if_absent(census.parse_payload(census.payload(mode, "on", step="design")))
                self.assertEqual(line, f"[boltzgen-opt] NOT ACTIVE: mode {mode} routes through the cuEquivariance kernels and this process cannot provide them "
                                       f"(step=design cueq_triatt=absent:import_cuequivariance_ops_torch_failed cueq_trimul=absent:import_cuequivariance_ops_torch_failed); "
                                       f"exit 3 before the step — `--use_kernels false` (upstream's switch) runs the torch paths, `--mode off` runs upstream alone")
                self.assertEqual(line, report.not_active_line(line[len("[boltzgen-opt] NOT ACTIVE: "):]))   # the package's one refusal formatter
        with mock.patch.dict(census._STATE, {"absent": {}}):
            self.assertIsNone(kernels.refuse_if_absent(census.parse_payload(census.payload("exact", "on"))))   # nothing absent: nothing to refuse

    def test_word_rules(self):
        facts = {"version": "0.11.1", "build": "cu13", "cc": (9, 0)}
        c = lambda s=None, b=None, f=None: {"served": dict(s or {}), "byrule": dict(b or {}), "fallback": dict(f or {})}
        self.assertEqual(census.word("cueq_triatt", "on", facts, {}, c(s={"generic": 3}, b={"s_qo<=100": 2})), "engaged:generic-sm90@0.11.1-cu13[served=3,byrule=2(s_qo<=100:2)]")
        self.assertEqual(census.word("cueq_triatt", "on", facts, {}, c(b={"s_qo<=100": 2})), "engaged:none-sm90@0.11.1-cu13[served=0,byrule=2(s_qo<=100:2)]")
        self.assertTrue(census.word("cueq_triatt", "on", facts, {}, c()).startswith("fallback:no-call"))
        self.assertTrue(census.word("cueq_triatt", "on", facts, {}, c(), step="design").startswith("fallback:no-call"))            # a design process that never reached the library fell back
        self.assertEqual(census.word("cueq_triatt", "on", facts, {}, c(), step="inverse_folding"), "n/a-upstream:step=inverse_folding(inverse-fold network: no trunk, no call site)")
        self.assertTrue(census.word("cueq_triatt", "on", facts, {}, c(s={"generic": 1}), step="inverse_folding").startswith("engaged:"))   # a call is a call, whatever the step
        self.assertEqual(census.KERNEL_FREE_STEPS, ("inverse_folding",))
        self.assertIn("inverse_folding", stack.GPU_STEPS)
        self.assertTrue(census.word("cueq_triatt", "on", facts, {}, c(s={"generic": 3}, f={"reference@s_qo=200": 1})).startswith("fallback:reference@s_qo=200:1[served=3,byrule=0,fallback=1]"))
        self.assertTrue(census.word("cueq_triatt", "on", facts, {"cueq_triatt": "import failed"}, c()).startswith("absent:import_failed"))
        self.assertEqual(census.word("cueq_triatt", "off:upstream(cc=7.5<8)", facts, {}, c()), "off-by-route:use_kernels=False(upstream(cc=7.5<8))")
        self.assertEqual(census.word("cueq_triatt", "off:use_kernels=false", facts, {}, c()), "off-by-route:use_kernels=False(use_kernels=false)")
        self.assertTrue(census.word("cueq_triatt", "off:upstream(cc=7.5<8)", facts, {}, c(s={"generic": 1})).startswith("fallback:calls-under-use_kernels=False"))
        self.assertEqual(census.word("cueq_triatt", "unknown", facts, {}, c(s={"generic": 1})), "engaged:generic-sm90@0.11.1-cu13[served=1,byrule=0]")


class TestCensusOnTheStubLibrary(unittest.TestCase):
    """The census bound to a stubbed cuequivariance in a fresh interpreter: served / byrule counted from real calls; a build whose kernels
    never serve reads `fallback`; the threshold switch moves the 100-token rung onto the kernels; the package's triton raise-stub reads `absent`."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_kern_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    CODE = ("import sys, torch; from boltzgen_opt import census; census.arm(sys.argv[1]); "
            "from cuequivariance_torch.primitives.triangle import triangle_attention, triangle_multiplicative_update\n"
            "for n in [int(x) for x in sys.argv[2].split(',')]:\n"
            "    q = torch.zeros(1, 1, 2, n, 32); triangle_attention(q, q, q, torch.zeros(1, 1, 2, n, n)); triangle_multiplicative_update(torch.zeros(1, n, n, 8))\n"
            "print(census.line())")

    def census(self, sizes, **env):
        rc, so, se = _stubs.run_py(self.CODE, _stubs.clean_env(self.site, **env), args=[census.payload("off", "on", step="design"), sizes])
        self.assertEqual(rc, 0, se)
        lines = census.parse_lines(so.splitlines())
        self.assertEqual(len(lines), 1, so + se)
        return lines[0]

    def test_served_and_byrule_are_counted(self):
        rec = self.census("128,100,100")
        self.assertEqual(rec["words"]["cueq_triatt"], "engaged:generic@0.11.1-cu13[served=1,byrule=2(s_qo<=100:2)]")   # no CUDA on the test box: no -sm tag
        self.assertEqual(rec["words"]["cueq_trimul"], "engaged:triton@0.11.1-cu13[served=1,byrule=2(seq<=100:2)]")
        self.assertEqual(rec["fields"]["cueq"], "0.11.1-cu13")
        self.assertEqual(rec["fields"]["caps"], "sm100f:0,sm107f:0,sm120f:0")
        self.assertEqual(rec["fields"]["thresholds"], "triatt:100,trimul:100")
        self.assertEqual((rec["route"], rec["step"]), ("default", "design"))
        self.assertNotIn("settings", rec["fields"])                          # the line has no settings= word
        self.assertTrue(kernels.verdict([rec], kernels.expectation("off", res(True)), 1)["ok"])

    def test_reference_where_the_rules_say_served_is_a_finding(self):
        rec = self.census("128,100", STUB_CUEQ_BROKEN_BUILD="1")
        self.assertEqual(rec["kinds"], {"cueq_triatt": "fallback", "cueq_trimul": "fallback"})
        self.assertIn("reference@s_qo=128:1[served=0,byrule=1,fallback=1]", rec["words"]["cueq_triatt"])
        v = kernels.verdict([rec], kernels.expectation("off", res(True)), 1)
        self.assertFalse(v["ok"])
        self.assertEqual(len(v["findings"]), 2)

    def test_threshold_switch_moves_the_rung(self):
        rec = self.census("100", CUEQ_TRIATTN_FALLBACK_THRESHOLD="99", CUEQ_TRIMUL_FALLBACK_THRESHOLD="99")
        self.assertTrue(rec["words"]["cueq_triatt"].startswith("engaged:generic@0.11.1-cu13[served=1,byrule=0]"), rec["words"])
        self.assertEqual(rec["fields"]["thresholds"], "triatt:99,trimul:99")

    def test_triton_raise_stub_is_absent(self):
        rc, so, se = _stubs.run_py("import sys, torch; from boltzgen_opt import census; census.arm(sys.argv[1]); import cuequivariance_ops_torch; print(census.line())",
                                   _stubs.clean_env(self.site, STUB_CUEQ_TRITON_BROKEN="1"), args=[census.payload("exact", "on")])
        self.assertEqual(rc, 0, se)
        rec = census.parse_lines(so.splitlines())[0]
        self.assertEqual(rec["kinds"]["cueq_trimul"], "absent")
        self.assertIn("triton_components_unavailable", rec["words"]["cueq_trimul"])
        self.assertEqual(rec["route"], "exact")

    def test_armed_and_never_imported_is_no_call(self):
        rc, so, se = _stubs.run_py("import sys; from boltzgen_opt import census; census.arm(sys.argv[1]); print(census.line())",
                                   _stubs.clean_env(self.site), args=[census.payload("exact", "on")])
        self.assertEqual(rc, 0, se)
        rec = census.parse_lines(so.splitlines())[0]
        self.assertTrue(rec["words"]["cueq_triatt"].startswith("fallback:no-call"), rec["words"])
        self.assertIn("KERNELS route=exact", se)                              # the exit line itself, on stderr
        self.assertNotIn("settings=", se)

    def test_eager_arm_on_an_absent_library_refuses_for_a_kit_mode_only(self):
        """`arm(eager=True)` (the stock child's form, after its proof) imports the library at once: absent for both accelerators on a path
        without it; `refuse_if_absent` then answers None for mode=off and the NOT ACTIVE line for a kit mode — in the same process state."""
        site = _stubs.materialize(os.path.join(self.tmp, "nocueq"), cueq=False)
        code = ("import json, sys; from boltzgen_opt import census, kernels; census.arm(sys.argv[1], eager=True)\n"
                "print('RECORD ' + json.dumps({'absent': census.absent(), 'refusal': kernels.refuse_if_absent()}))\n")
        for mode, refused in (("off", False), ("exact", True), ("big", True)):
            rc, so, se = _stubs.run_py(code, _stubs.clean_env(site), args=[census.payload(mode, "on", step="design")])
            self.assertEqual(rc, 0, se)
            rec = next(json.loads(ln[7:]) for ln in so.splitlines() if ln.startswith("RECORD "))
            self.assertEqual(set(rec["absent"]), set(census.ACCELERATORS), mode)
            if refused:
                self.assertTrue(rec["refusal"].startswith(f"[boltzgen-opt] NOT ACTIVE: mode {mode} routes through the cuEquivariance kernels and this process cannot provide them (step=design cueq_triatt=absent:import_cuequivariance_ops_torch_failed"), rec["refusal"])
            else:
                self.assertIsNone(rec["refusal"])                               # upstream's own process: an absent library is upstream's to meet, never the kit's refusal
            self.assertIn(f"KERNELS route={census.route_of(mode)} step=design cueq_triatt=absent:import_cuequivariance_ops_torch_failed", se)   # the line prints at exit either way

    def test_upstreams_own_process_carries_nothing_of_the_kit(self):
        """Upstream's own model process (mode off: `python -s -m boltzgen_opt.stock_design …`) proves its environment and runs the step — no
        census, counter, wrap or timer: the caller names neither the census nor any policy module (AST), has no option of its own beyond the
        core's stock-proof contract, and importing it loads no other package module."""
        site = _stubs.materialize(os.path.join(self.tmp, "nocueq_stock"), cueq=False)
        code = ("import json, sys\n"
                "from boltzgen_opt import stock_design\n"
                "loaded = sorted(m for m in sys.modules if m.startswith('boltzgen_opt') and m != 'boltzgen_opt._autoload')\n"
                "print('RECORD ' + json.dumps({'loaded': loaded, 'own': [n for n in ('split_kernels', 'arm_census') if hasattr(stock_design, n)]}))\n")
        rc, so, se = _stubs.run_py(code, _stubs.clean_env(site))
        self.assertEqual(rc, 0, se)
        rec = next(json.loads(ln[7:]) for ln in so.splitlines() if ln.startswith("RECORD "))
        self.assertEqual(rec["own"], [])                                                                # no `--kernels`, no arm_census
        self.assertNotIn("boltzgen_opt.census", rec["loaded"]); self.assertNotIn("boltzgen_opt.kernels", rec["loaded"])
        tree = ast.parse(open(stock_design.__file__, encoding="utf-8").read())
        named = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(any(x.split(".")[-1] in ("kernels", "census") for x in named), named)        # the stock caller names no monitor and no policy module anywhere

class TestRoutesEndToEnd(unittest.TestCase):
    """`boltzgen-opt design --mode off` on the stubs: upstream alone after the proof — no census line, no kernels account, the exit ALWAYS the
    run's own (an absent library or `--use_kernels false` are upstream's to meet). A kit child (BOLTZGEN_OPT_KERNELS at interpreter start)
    that cannot provide the kernels its mode routes through exits 3 before the step."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_kroute_")
        cls.site = _stubs.materialize(cls.tmp)
        cls.spec = os.path.join(cls.tmp, "spec.yaml")
        open(cls.spec, "w").write(SPEC)
        if not _stubs.package_installed():
            raise unittest.SkipTest("boltzgen_opt is not installed in this interpreter: run in a venv with `pip install -e boltzgen/opt` to exercise this")
        cls.pth = _stubs.site_pth(cls.site)

    @classmethod
    def tearDownClass(cls):
        _stubs.remove_site_pth(cls.pth)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, args, **env_more):
        env = _stubs.clean_env(self.site, **env_more)
        r = subprocess.run([sys.executable, "-m", "boltzgen_opt"] + args, env=env, capture_output=True, text=True, timeout=300, cwd=self.tmp)
        return r.returncode, r.stdout, r.stderr

    def design(self, name, *extra, **env):
        out = os.path.join(self.tmp, name)
        rc, so, se = self.cli(["design", "--mode", "off", self.spec, "--output", out, "--seed", "3", "--num_designs", "2", *extra], **env)
        return rc, se, out

    def test_stock_route_prints_no_census_line_and_records_none(self):
        """`design --mode off`: every GPU step is upstream alone after its proof — no `KERNELS` / `PEAK` line anywhere in the run, no `kernels`
        account in opt_manifest.json (the configure record keeps upstream's own `Using kernels:` resolution), the exit the run's own."""
        rc, se, out = self.design("d1", **H100)
        self.assertEqual(rc, 0, se)
        log = open(os.path.join(out, "opt_run.log")).read()
        self.assertNotIn("] KERNELS ", se + log); self.assertNotIn("] PEAK ", se + log)
        self.assertEqual(census.parse_lines(log.splitlines()), [])
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertNotIn("kernels", man["activation_report"])
        self.assertIsNone(man["run"]["kernels"])
        self.assertEqual(man["configure"]["kernels_resolution"]["line"], "Using kernels: True [device capability: (9, 0)]")
        self.assertEqual(man["exit_code"], 0)
        self.assertTrue(os.path.exists(os.path.join(out, "analysis.done")))
        proof = json.load(open(os.path.join(out, "stock_env_proof_design.json")))
        self.assertTrue(proof["ok"], proof)
        self.assertEqual((proof["autoload_armed"], proof["torch_loaded_before_proof"], proof["step"]), ([], False, "design"))
    def test_absent_library_on_the_stock_route_is_upstreams_to_meet(self):
        root = os.path.join(self.tmp, "nocueq")
        site = _stubs.materialize(root, cueq=False)
        _stubs.remove_site_pth(self.pth)
        pth = _stubs.site_pth(site)
        try:
            out = os.path.join(root, "run")
            env = _stubs.clean_env(site, **H100)
            r = subprocess.run([sys.executable, "-m", "boltzgen_opt", "design", "--mode", "off", self.spec, "--output", out, "--steps", "design", "--num_designs", "2"],
                               env=env, capture_output=True, text=True, timeout=300, cwd=root)
            self.assertEqual(r.returncode, 0, r.stderr)                        # upstream alone: an absent library is upstream's to meet — the step runs, the exit is the run's
            self.assertTrue(os.path.exists(os.path.join(out, "design.done")))
            self.assertNotIn("routes through the cuEquivariance kernels", r.stderr); self.assertNotIn("REFUSED", r.stderr); self.assertNotIn("] KERNELS ", r.stderr)
            man = json.load(open(os.path.join(out, "opt_manifest.json")))
            self.assertEqual((man["exit_code"], man["seed"]), (0, None)); self.assertNotIn("kernels", man["activation_report"])
        finally:
            _stubs.remove_site_pth(pth)
            type(self).pth = _stubs.site_pth(self.site)
    def test_kit_child_arms_early_and_refuses_an_absent_library_before_anything(self):
        """The kit routes' children (BOLTZGEN_OPT_KERNELS in their environment, the package's .pth at interpreter start): armed early, so an
        absent library is NOT ACTIVE, exit 3, before the child's own code runs; upstream's CPU steps are not model processes and stay unarmed;
        a mode=off payload in the same place refuses nothing."""
        pl = census.payload("exact", "on")
        body = "import boltzgen; print('step body ran')"                       # upstream's package import is the early trigger (before any step)
        r = subprocess.run([sys.executable, "-c", body], env=_stubs.clean_env(self.site, BOLTZGEN_OPT_KERNELS=pl), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)                            # the stub library imports: armed, the body runs, the line at exit
        self.assertIn("step body ran", r.stdout)
        self.assertIn("KERNELS route=exact step=all", r.stderr); self.assertNotIn("settings=", r.stderr)
        r = subprocess.run([sys.executable, "-c", "import boltzgen, sys; m = sys.modules.get('cuequivariance_ops_torch'); print(getattr(getattr(m, 'triangle_attention', None), '_boltzgen_opt_census', None))"],
                           env=_stubs.clean_env(self.site, BOLTZGEN_OPT_KERNELS=pl), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.stdout.strip(), "cueq_triatt", r.stderr)           # early: importing upstream's package installed and wrapped the library before any step code
        r = subprocess.run([sys.executable, "-c", "import sys; print('cuequivariance_ops_torch' in sys.modules or 'torch' in sys.modules)"],
                           env=_stubs.clean_env(self.site, BOLTZGEN_OPT_KERNELS=pl), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.stdout.strip(), "False", r.stderr)                 # nothing imported at interpreter start-up itself
        root = os.path.join(self.tmp, "nocueq_kit")
        site = _stubs.materialize(root, cueq=False)
        _stubs.remove_site_pth(self.pth)
        pth = _stubs.site_pth(site)
        try:
            r = subprocess.run([sys.executable, "-c", body], env=_stubs.clean_env(site, BOLTZGEN_OPT_KERNELS=pl), capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, codes.EXIT_NOT_ACTIVE, r.stderr)
            self.assertNotIn("step body ran", r.stdout)                     # nothing of the child's own code ran
            self.assertIn("[boltzgen-opt] NOT ACTIVE: mode exact routes through the cuEquivariance kernels and this process cannot provide them (step=? cueq_triatt=absent:import_cuequivariance_ops_torch_failed", r.stderr)
            self.assertIn("exit 3 before the step", r.stderr)
            self.assertIn("KERNELS route=exact", r.stderr)                     # the census line itself, with the absent words
            r = subprocess.run([sys.executable, "-c", body], env=_stubs.clean_env(site, BOLTZGEN_OPT_KERNELS=pl, BOLTZGEN_PIPELINE_STEP="analysis"), capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)                        # upstream's CPU step: not a model process, unarmed, no line
            self.assertIn("step body ran", r.stdout)
            self.assertNotIn("KERNELS route=", r.stderr)
            r = subprocess.run([sys.executable, "-c", body], env=_stubs.clean_env(site, BOLTZGEN_OPT_KERNELS=census.payload("off", "on", step="design")), capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)                        # a mode=off payload: upstream's own process — the body runs, the line names the absent library, nothing refused
            self.assertIn("step body ran", r.stdout)
            self.assertNotIn("NOT ACTIVE", r.stderr)
            self.assertIn("KERNELS route=default step=design cueq_triatt=absent:import_cuequivariance_ops_torch_failed", r.stderr)
        finally:
            _stubs.remove_site_pth(pth)
            type(self).pth = _stubs.site_pth(self.site)

    def test_runner_starts_each_cpu_step_under_its_own_name_unarmed(self):
        """The kits' runner on its own bytes (xa_run.py l.58-68): a CPU step is upstream's `main.py <config>` in the runner's environment minus
        PYTHONPATH under THAT step's name — not the GPU steps' name the in-process pipeline leaves on the runner's environment (bg_inproc.py l.33)
        — so the census payload every kit child carries (design.kit_env) arms nothing there: the runner prints its KERNELS line, the step none."""
        res_ = modes.resolve("exact", stack.opt_home())
        run = os.path.join(self.tmp, "runner_cpu")
        r = subprocess.run([sys.executable, "-m", "boltzgen.cli.boltzgen", "configure", self.spec, "--output", run, "--steps", "design", "analysis", "filtering", "--num_designs", "2"],
                           env=_stubs.clean_env(self.site), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        pl = census.payload("exact", "on")
        env = _stubs.clean_env(self.site, extra_path=stack.kit_src(modes.KIT_PARTNER), BOLTZGEN_OPT_KERNELS=pl,
                               BOLTZGEN_PIPELINE_STEP="design_folding")            # the name the GPU steps leave behind in the runner (bg_inproc.py l.33), set in their place: only the CPU steps run here
        r = subprocess.run([sys.executable, os.path.join(stack.opt_home(), res_.runner), run, "0", "analysis,filtering"], env=env, capture_output=True, text=True, timeout=300, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        for step in ("analysis", "filtering"):                                  # both CPU steps, in pipeline order, each from the one child-environment dict the loop rewrites
            self.assertIn(f"[xa_run] step {step} (stock subprocess) rc=0", r.stderr)
            rec = json.load(open(os.path.join(run, step + ".done")))          # what upstream's step process saw
            self.assertEqual((rec["step_env"], rec["pythonpath"]), (step, None), rec)
            self.assertNotIn("boltzgen_opt.kernels", rec["opt_modules"], rec) # the installed .pth ran there and armed nothing: a CPU step is not a model process
        lines = census.parse_lines(r.stderr.splitlines())
        self.assertEqual([x["step"] for x in lines], ["design_folding"], r.stderr)   # one KERNELS line, the runner's own (it imports upstream's package); none labelled with a step that never touches the library

    def test_use_kernels_false_is_passed_through(self):
        """`-- --use_kernels false` reaches upstream's configure verbatim on the stock route as on every kit route (test_cli_forms); the step
        runs, exit 0; upstream's own resolution line records it."""
        rc, se, out = self.design("d5", "--steps", "design", "--", "--use_kernels", "false", **H100)
        self.assertEqual(rc, 0, se)
        self.assertTrue(os.path.exists(os.path.join(out, "design.done")))
        self.assertIn("--use_kernels false", open(os.path.join(out, "configure_argv.txt")).read().splitlines()[0])
        self.assertIn("Using kernels: False [device capability: (9, 0)]", open(os.path.join(out, "opt_configure.log")).read())
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertEqual((man["exit_code"], man["configure"]["kernels_resolution"]["use_kernels"]), (0, False))

    def test_below_cc8_upstream_ships_kernels_off_and_the_stock_step_runs(self):
        rc, se, out = self.design("d6", "--steps", "design", BOLTZGEN_STUB_KERNELS_CC="7,5")
        self.assertEqual(rc, 0, se)
        self.assertIn("Using kernels: False [device capability: (7, 5)]", open(os.path.join(out, "opt_configure.log")).read())


if __name__ == "__main__":
    unittest.main()
