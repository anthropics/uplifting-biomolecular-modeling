"""Lever decision logic and the evidence classification built from it: nosub's two decisions (nosub.py: the graded executable
unchunked, source `kit`; the forward-only executable at stock's size rule) and its LEVER line at either side of the size gate (no jax needed:
the decisions are pure) -- nothing in the package changes them; then the EVIDENCE classification those LEVER lines feed
(opt_core.report.lever_line -- the grammar nosub.py and pallas.py print): the four states per lever, the verdict, the stock-arm check."""

import re
import unittest

from opt_core import report as core_report
from opt_core.kernels import pallas_attn_serve as F1

from colabdesign_opt import evidence, levers, modes, nosub, pallas


# ============================================================ nosub's decisions (nosub.py, pure -- no jax needed)
class TestNosubDecisions(unittest.TestCase):
    def setUp(self):
        levers.reset_for_tests(); nosub._BUILDS.clear(); nosub._LINES.clear()

    tearDown = setUp

    def test_the_shipped_decisions_at_every_size(self):
        """The graded executable unchunked (source `kit`), the forward-only one at stock's rule (chunks of 4 above 384, none at or below); the
        size gate decides the LEVER word: `skipped reason=gated` at or below 384 tokens (stock's programs), `on` above."""
        for tokens in (293, 343, 384, 385, 393, 443, 900, 1400):
            d = nosub.decide(tokens)
            self.assertEqual(sorted(d), ["fn", "gate", "gate_fn", "grad", "policy", "tokens"]); self.assertEqual(d["policy"], {"grad": "kit", "fn": "stock"})
            self.assertEqual((d["grad"].value, d["grad"].source, d["grad"].estimated_bytes, d["grad"].device_bytes), (None, "kit", None, None))
            self.assertEqual((d["fn"].value, d["fn"].source), (4 if tokens > 384 else None, "stock"))
            self.assertEqual(d["gate"].served, tokens > 384)
            line = nosub.line_of(d)
            self.assertTrue(line.startswith(f"[colabdesign-opt] LEVER name=nosub state={'on' if tokens > 384 else 'skipped reason=gated'} impl=subbatch_policy@"), line)
            for part in (" origin=core ", f" tokens={tokens} ", " grad_subbatch=none grad_subbatch_source=kit ", f" fn_subbatch={4 if tokens > 384 else 'none'} fn_subbatch_source=stock ", " gate.nosub=min385 "):
                self.assertIn(part, line)
            for absent in ("dev.", "requested", "_est_gb", "_dev_gb", "auto"):
                self.assertNotIn(absent, line)
            ev = evidence.classify([line], ["nosub"])
            self.assertEqual((ev["state"]["nosub"], ev["grad_subbatch"], ev["grad_subbatch_source"], ev["fn_subbatch_source"]), ("applied" if tokens > 384 else "gated", "none", "kit", "stock"))
            self.assertEqual(evidence.verdict(ev)["partial"], {})

    def test_nothing_in_the_package_changes_a_decision(self):
        """No setter, no parse, no module request state: the lever's surface is decide / line_of / builds / install / installed."""
        for gone in ("set_dev_request", "parse_dev_request", "dev_settings", "dev_overrides", "reset_dev", "DEV_DEFAULTS", "DEV_FLAGS", "_DEV"):
            self.assertFalse(hasattr(nosub, gone), gone)
        decisive = lambda line: re.sub(r" calls=\d+ served=\d+ gated=\d+", "", line)         # the size gate's running census aside, a size's line never varies
        self.assertEqual(decisive(nosub.line_of(nosub.decide(900))), decisive(nosub.line_of(nosub.decide(900))))

    def test_builds_record(self):
        d = nosub.decide(443); d["line"] = nosub.line_of(d); d["lines"] = [d["line"]]; nosub._BUILDS.append(d)
        b = nosub.builds()
        self.assertEqual(len(b), 1); self.assertEqual(sorted(b[0]), ["fn", "gate", "gate_fn", "grad", "line", "lines", "policy", "tokens"]); self.assertEqual((b[0]["tokens"], b[0]["line"], b[0]["lines"]), (443, d["line"], [d["line"]]))
        d = nosub.decide(700, grad="kit", fn="kit")                                                            # fast: nosub + nosub_fn
        self.assertEqual((d["grad"].value, d["grad"].source, d["fn"].value, d["fn"].source, d["gate_fn"].served), (None, "kit", None, "kit", True))
        line = nosub.line_of_fn(d)
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=nosub_fn state=on impl=subbatch_policy@"), line)
        self.assertIn(" origin=core tokens=700 fn_subbatch=none fn_subbatch_source=kit stock_fn_subbatch=4 gate.nosub_fn=min385 ", line)
        d = nosub.decide(300, grad="stock", fn="kit")                                                          # fast-no-nosub: the graded executable at stock's rule, fn unchunked (= stock's here: gated)
        self.assertEqual((d["grad"].source, d["fn"].source, d["gate_fn"].served), ("stock", "kit", False))
        self.assertTrue(nosub.line_of_fn(d).startswith("[colabdesign-opt] LEVER name=nosub_fn state=skipped reason=gated impl="), nosub.line_of_fn(d)); self.assertIn(" stock_fn_subbatch=none ", nosub.line_of_fn(d))



# ============================================================ the EVIDENCE classification built from the levers' LEVER lines
TAG = "colabdesign-opt"


def nosub_line(tokens, grad, grad_src, fn, state, reason=None):
    pairs = [("tokens", tokens), ("grad_subbatch", grad), ("grad_subbatch_source", grad_src),
             ("fn_subbatch", fn), ("fn_subbatch_source", "stock"), ("gate.nosub", "min385"), ("calls", 1), ("served", int(tokens > 384)), ("gated", int(reason == "gated")), ("fallback", 0)]
    return core_report.lever_line(TAG, "nosub", state, *pairs, reason=reason, impl="subbatch_policy@0.3.2", origin="core")


def pallas_line(served, fallbacks: dict):
    L = F1.ledger(min_tokens=0, expected=pallas.EXPECTED_FALLBACKS)
    for _ in range(served):
        L.serve("B4xH8xS256xD32")
    for reason, n in fallbacks.items():
        L.fallback(reason, n)
    L.impl = "pallas_attn@db23dd4b"; L.origin = "core"
    return L.line(TAG, scope=pallas.SCOPE, source="exit")


APPLIED = nosub_line(571, "none", "kit", 4, "on")
GATED = nosub_line(208, "none", "kit", "none", "skipped", "gated")


def nosub_fn_line(tokens, state, reason=None):
    """Lever nosub_fn's line as nosub.line_of_fn prints it (the forward-only executable unchunked, source kit; stock's rule beside it)."""
    pairs = [("tokens", tokens), ("fn_subbatch", "none"), ("fn_subbatch_source", "kit"), ("stock_fn_subbatch", 4 if tokens > 384 else None), ("gate.nosub_fn", "min385"),
             ("calls", 1), ("served", int(tokens > 384)), ("gated", int(reason == "gated")), ("fallback", 0)]
    return core_report.lever_line(TAG, "nosub_fn", state, *pairs, reason=reason, impl="subbatch_policy@0.3.2", origin="core")


APPLIED_FN = nosub_fn_line(571, "on")
GATED_FN = nosub_fn_line(208, "skipped", "gated")


def compilecache_line(requests=104, hits=0, misses=104, state="on", dir_source="default"):
    """Lever compilecache's exit line as compilecache_jax.line_of prints it."""
    if state != "on":
        return core_report.lever_line(TAG, "compilecache", state, impl="jax_design.pcc@0.5.18.20", origin="core")
    return core_report.lever_line(TAG, "compilecache", "on", impl="jax_design.pcc@0.5.18.20", origin="core", dir="/x/cache/colabdesign_opt/pcc/KEY/xla", dir_source=dir_source,
                                  key="KEY", autotune="jax_xla_cache", requests=requests, hits=hits, misses=misses, compiled_s="94.1", saved_s="0.0", entries="0->104", reinit=0, source="exit", pid=7)


def hoist_prev_line(device_prev_steps=141, multi=0, source="exit"):
    """Lever hoist_prev's line (install / exit census) as hoist_prev.line prints it."""
    return core_report.lever_line(TAG, "hoist_prev", "on", ("patched", "_recycle,run"), ("zero_builds", 1), ("zero_inits", device_prev_steps), ("device_prev_steps", device_prev_steps),
                                  ("multi_model_stock_path", multi), ("run_calls", device_prev_steps + multi), impl="hoist_prev@kit", origin="kit", source=source, pid=7)


def lowercache_line(calls=2, loads=0, stores=2, traced=2, state="on"):
    """Lever lowercache's exit line as lowercache.line_of prints it."""
    if state != "on":
        return core_report.lever_line(TAG, "lowercache", state, impl="serialize_executable@kit", origin="kit")
    return core_report.lever_line(TAG, "lowercache", "on", impl="serialize_executable@kit", origin="kit", dir="/x/cache/colabdesign_opt/pcc/KEY/lowered", calls=calls, memo_hits=0, loads=loads, stores=stores,
                                  traced=traced, fallbacks=0, load_s="0.0", retrace_s="0.0", traced_s="41.0", store_s="1.2", mb_stored="39", entries=f"0->{stores}", relower="off", why="none", source="exit", pid=7)


LOWERCACHE_OK = lowercache_line()                    # a cold process: both programs traced and stored
LOWERCACHE_WARM = lowercache_line(2, 2, 0, 0)        # a warm process: both programs loaded
CACHE_OK = compilecache_line()                       # a cold-populating design process: requests>0, all misses
CACHE_WARM = compilecache_line(104, 104, 0)
CACHE_IDLE = compilecache_line(0, 0, 0)              # requests=0: the cache was never consulted (missing, fail-closed)
HOIST_OK = hoist_prev_line()
HOIST_IDLE = hoist_prev_line(0, 0)                   # installed, no design step ran through it (missing)
HOIST_STOCKPATH = hoist_prev_line(0, 3)              # ran, always on stock's multi-model path (fallback)
PARCOMPILE_OK = "[colabdesign-opt] LEVER name=parcompile state=on impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact threads=16 cpus=16 flags=--xla_gpu_enable_llvm_module_compilation_parallelism=true,--xla_gpu_force_compilation_parallelism=16 xla_flags_kept=--xla_gpu_enable_triton_gemm=false source=install pid=7"
EXACT_OK = [CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK]                      # the exact-class levers' lines of a design that engaged them
DEFECT = nosub_line(900, 4, "kit", 4, "skipped", "kept")           # not a line the lever can print (it has no fallback): classified `fallback` = named, never passed
def trimul_line(served, fallbacks: dict, state="on"):
    """kernels/trimul_fused.py's exit line for a design process (opt_core.counters.Ledger under the lever's LINE_NAME)."""
    from opt_core.counters import Ledger
    L = Ledger("trimul_pallas", impl="trimul_pallas@0badc0de", origin="kit", expected=("not_fused",))
    for i in range(served):
        L.serve("N256xC128xE%d" % (i % 2))
    for reason, n in fallbacks.items():
        L.fallback(reason, n)
    return L.line(TAG, numerics="precision", precision="bf16", source="exit")


def triatt_line(served, fallbacks: dict, state="on"):
    """kernels/triatt_lever.py's exit line for a design process (opt_core.counters.Ledger under the lever id)."""
    from opt_core.counters import Ledger
    L = Ledger("triatt", impl="triatt_attn@0d15ea5e", origin="kit", expected=("below_keys_rule",))
    for i in range(served):
        L.serve("B4xH4xS256xD32")
    for reason, n in fallbacks.items():
        L.fallback(reason, n)
    return L.line(TAG, numerics="precision", precision="bf16", source="exit")


TRIATT_OK = triatt_line(96, {"below_keys_rule": 8})


def census_line(name, impl, precision, served, fallbacks: dict, shapes="none"):
    """The census levers' exit line shape (kernels/layers_*.py lever_line: opt_core.report.lever_line with numerics/precision/served/fallback/fallback_by/shapes)."""
    from opt_core import report as _r
    fb = ",".join(f"{k}:{v}" for k, v in sorted(fallbacks.items())) or "none"
    return _r.lever_line(TAG, name, "on", ("numerics", "precision"), ("precision", precision), ("served", served), ("fallback", sum(fallbacks.values())),
                         ("fallback_by", fb), ("shapes", shapes), impl=impl, origin="kit")


OPM_OK = census_line("opm_fold", "opm_fold@kit", "tf32", 52, {}, "S1xN200xc32xF128:4,S2xN200xc32xF128:48")     # every OuterProductMean call re-associated
LN_OK = census_line("ln", "ln_pallas@kit", "none", 830, {"channels_not_pow2": 96}, "C128xbf16:400,C256xbf16:334,C64xbf16:96")   # the 384-channel single norms on stock's method (declared)                                   # every attention call served, the 2-key MSA column attention on stock's method (declared)


def proj_line(served, fallbacks: dict, active="triatt"):
    """kernels/proj_attn.py's exit line (opt_core.counters.Ledger under the lever id, with its evidence pairs)."""
    from opt_core.counters import Ledger
    L = Ledger("proj", impl="fused_qkvg@kit", origin="kit", expected=("head_dim_lt_16",))
    for i in range(served):
        L.serve("S256xC128xH4")
    for reason, n in fallbacks.items():
        L.fallback(reason, n)
    return L.line(TAG, "on", numerics="precision", precision="bf16", requires="attn_kernel", active=active, traced=served + sum(fallbacks.values()))


PROJ_OK = proj_line(88, {"head_dim_lt_16": 8})                                              # every kernel-served call fused; the extra-MSA row attention below (declared)
PALLAS_REPLACED = pallas.off_line("replaced")                                             # pallas in a fast arm: registered, replaced by triatt (registry Supersession)
TRIMUL_OK = trimul_line(12, {})                              # the triangle-multiplication kernels' exit line of a design process: 12 traced calls served, no fallback

def txla_line(served, fallbacks: dict, state="on", reason=None, rows="triattn_native:10,k2b_aot:6", shape="B500xH4xS500xD32"):
    """txla.py's exit line (opt_core.counters.Ledger under the lever id with the lever's evidence pairs): `on` whenever the bridge served (no
    size rule of the kit's: the bridge's own table names the row per call); `skipped` only when every call was handed to triatt by a declared word."""
    from opt_core.counters import Ledger
    L = Ledger("txla", impl="triattn_xla@0.5.116.0", origin="core", expected=("small_call", "refused"))
    for i in range(served):
        L.serve(shape)
    for r, n in fallbacks.items():
        for _ in range(n):
            L.fallback(r)
    return L.line(TAG, state, reason, rows=rows if served else "none", served_fn=served, served_grad=0, vjp_traced=served + 1 if served else 0, vmap=0, refused="none", layout="BNHSD",
                  impl_word="auto", requires="triatt", grad="triatt", proj_restacked=1, numerics="precision", precision="bf16", cc="9.0", tx="1.6.1", source="exit")


TXLA_OK = txla_line(8, {"small_call": 1})                        # a design's forward-only program: 8 traced calls on the bridge, the 2-key column attention on triatt
TXLA_GATED = txla_line(8, {"small_call": 1}, rows="k2b_aot:16", shape="B112xH4xS112xD32")   # a design below 384 tokens: the bridge serves it too (its table's row there) — the lever has no size rule, the line is `on` (name kept for the importers' below-384 scenarios)


def txla_aside_line():
    """The line levers.install prints when txla steps aside BY NAME at install (fast-no-triatt: no triatt behind the served class)."""
    from colabdesign_opt import levers as _lv
    return _lv.skipped_line("txla", _lv.CANNOT_RUN, "NeedsTriatt: lever triatt does not serve colabdesign.af.alphafold.model.modules in this run")


TXLA_ASIDE = txla_aside_line()


def transition_line(served, fallbacks: dict, state="on", reason=None, shapes="C128:2,C256:1,C64:2", rows="cd_transition"):
    """kernels/layers_transition.py's exit line (opt_core.report.lever_line with the adapter's evidence pairs, as its lever_line() renders them): the
    Transition calls served by the provider row the tier word names for their cells (`row=` / `tier=`), calls whose cell names the stock statement
    counted `cell_xla` (a declared per-call step-aside BY NAME, like every provider-bound lever's)."""
    from opt_core import report as _r
    fb = ",".join(f"{k}:{v}" for k, v in sorted(fallbacks.items())) or "none"
    n_xla = int(fallbacks.get("cell_xla", 0))
    row = ",".join(f"{k}:{v}" for k, v in ((rows, served), ("xla", n_xla)) if v) or "none"
    pairs = (("numerics", "precision"), ("precision", "bf16"), ("variant", "fused"), ("served", served), ("fallback", sum(fallbacks.values())), ("fallback_by", fb),
             ("shapes", shapes if served else "none"), ("provider", "opt_core.kernels.pallas@0.5.116.0"), ("word", "fast"), ("row", row), ("tier", row), ("source", "exit"))
    return _r.lever_line(TAG, "transition", state, *pairs, reason=reason, impl="layers_transition@0a1b2c3d", origin="core")


TRANSITION_OK = transition_line(5, {})                                # the 5 traced Transition calls (pair x2, MSA, extra-MSA, template) on the provider row their cells name
TRANSITION_SMALL = transition_line(3, {"cell_xla": 2}, shapes="C128:2,C64:1")   # a small design: the pair / template cells name the row, the MSA / extra-MSA cells the stock statement (`cell_xla`, by name) — applied
PALLAS_OK = pallas_line(8, {"head_dim_lt_16": 1})          # the kernel's exit line on the AF-Multimer design model: 8 calls served, the extra-MSA row attention the one declared fallback


class TestEvidence(unittest.TestCase):
    def test_lines_are_the_levers_grammar(self):
        """The test's lines come from the same formatter the levers use (nosub.line_of → lever_line; pallas.exit_line → Ledger.line)."""
        self.assertTrue(APPLIED.startswith("[colabdesign-opt] LEVER name=nosub state=on impl=subbatch_policy@0.3.2 origin=core tokens=571 grad_subbatch=none"))
        self.assertIn("LEVER name=F1.pallas_attn state=on", pallas_line(8, {"head_dim_lt_16": 1}))
        self.assertEqual(evidence.PALLAS_LINE_NAME, F1.LEVER); self.assertEqual(evidence.DECLARED_FALLBACKS, tuple(pallas.EXPECTED_FALLBACKS))
        self.assertEqual(nosub.GATE.name, "nosub"); self.assertEqual(nosub.GATE.bounds_word(), "min385"); self.assertEqual(modes.SIZE_GATE_TOKENS + 1, 385)

    def test_nosub_states(self):
        rec = evidence.classify([APPLIED], ("nosub",))
        self.assertEqual(rec["state"], {"nosub": "applied"}); self.assertEqual((rec["tokens"], rec["grad_subbatch"], rec["fn_subbatch"]), (571, "none", "4"))
        rec = evidence.classify(["junk before " + GATED], ("nosub",))
        self.assertEqual(rec["state"], {"nosub": "gated"}); self.assertEqual(rec["tokens"], 208)
        rec = evidence.classify([DEFECT], ("nosub",))
        self.assertEqual(rec["state"], {"nosub": "fallback"}); self.assertEqual(rec["grad_subbatch"], "4")
        self.assertEqual(evidence.classify(["nothing from a lever"], ("nosub",))["state"], {"nosub": "missing"})
        self.assertEqual(evidence.classify([core_report.lever_line(nosub.TAG, "nosub", "off", reason="mode:other", impl=nosub.IMPL, origin="core")], ("nosub",))["state"], {"nosub": "missing"})   # state=off is not evidence of a selected lever

    def test_pallas_states(self):
        rec = evidence.classify([APPLIED, pallas_line(8, {"head_dim_lt_16": 1})], ("nosub", "pallas"))
        self.assertEqual(rec["state"], {"nosub": "applied", "pallas": "applied"})
        self.assertEqual((rec["pallas_served"], rec["pallas_fallback"], rec["pallas_fallback_by"], rec["pallas_impl"], rec["pallas_shapes"]), (8, 1, {"head_dim_lt_16": 1}, "pallas_attn@db23dd4b", "B4xH8xS256xD32:8"))
        rec = evidence.classify([APPLIED, pallas_line(0, {})], ("nosub", "pallas"))
        self.assertEqual(rec["state"]["pallas"], "missing")                  # installed, no call served
        rec = evidence.classify([APPLIED, pallas_line(7, {"head_dim_lt_16": 1, "backend_not_gpu": 1})], ("nosub", "pallas"))
        self.assertEqual(rec["state"]["pallas"], "fallback")                 # a reason outside the declared set
        rec = evidence.classify([APPLIED], ("nosub", "pallas"))
        self.assertEqual(rec["state"]["pallas"], "missing")                  # no exit line: the process did not exit normally
        self.assertEqual(evidence.classify([], ("nosub", "pallas"))["state"], {"nosub": "missing", "pallas": "missing"})

    def test_modes_name_only_classified_levers(self):
        """Every mode names only levers `classify` knows: a lever `verdict` cannot see never enters a mode."""
        for m in modes.MODES.values():
            for lever in m.levers:
                self.assertIn(lever, evidence.LINE_NAMES, f"mode {m.name!r} names {lever!r}")
        self.assertEqual(set(evidence.GATES), {"nosub", "nosub_fn"})                         # the levers with a by-number size gate of their own (transition and txla: the providers' cells, no kit gate)

    def test_exact_levers_classify(self):
        """compilecache: applied iff its exit line has requests>0 (a design process always compiles); hoist_prev: applied iff the device path was taken."""
        ev = evidence.classify([CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK], ("compilecache", "lowercache", "parcompile", "hoist_prev"))
        self.assertEqual((ev["state"], ev["compile_requests"], ev["compile_hits"], ev["prev_device_steps"]), ({"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied"}, 104, 0, 141)); self.assertEqual(ev["parcompile_threads"], 16)
        self.assertEqual(evidence.verdict(ev), {"partial": {}, "gated": {}, "skipped": {}})
        ev = evidence.classify([CACHE_IDLE, HOIST_IDLE], ("compilecache", "lowercache", "parcompile", "hoist_prev"))
        self.assertEqual(ev["state"], {"compilecache": "missing", "lowercache": "missing", "parcompile": "missing", "hoist_prev": "missing"})
        v = evidence.verdict(ev)["partial"]
        self.assertEqual(v["compilecache"], "missing: installed, no compile request consulted the cache (requests=0)"); self.assertEqual(v["hoist_prev"], "missing: installed, no design step ran through it (run_calls=0)")
        ev = evidence.classify([CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_STOCKPATH], ("compilecache", "lowercache", "parcompile", "hoist_prev"))   # every design step on stock's multi-model path BY NAME: a named aside (exit 0), never a refusal on a settings file stock accepts
        self.assertEqual(ev["state"]["hoist_prev"], "skipped"); v = evidence.verdict(ev); self.assertEqual(v["partial"], {}); self.assertTrue(v["skipped"]["hoist_prev"].startswith("aside:multi_model_stock_path:3"), v)
        ev = evidence.classify([compilecache_line(state="off"), HOIST_OK], ("compilecache", "lowercache", "parcompile", "hoist_prev"))
        self.assertEqual(ev["state"]["compilecache"], "missing")
        self.assertEqual(evidence.classify([], ("compilecache", "lowercache", "parcompile", "hoist_prev"))["state"], {"compilecache": "missing", "lowercache": "missing", "parcompile": "missing", "hoist_prev": "missing"})

    def test_verdict(self):
        """partial = fallback | missing (each with a reason); gated = nosub at or below the size gate; applied = neither."""
        self.assertEqual(evidence.verdict(evidence.classify([APPLIED], ("nosub",))), {"partial": {}, "gated": {}, "skipped": {}})
        v = evidence.verdict(evidence.classify([GATED], ("nosub",)))
        self.assertEqual((v["partial"], list(v["gated"])), ({}, ["nosub"]))
        v = evidence.verdict(evidence.classify([DEFECT], ("nosub",)))
        self.assertEqual(list(v["partial"]), ["nosub"]); self.assertEqual(v["partial"]["nosub"], "fallback: grad_subbatch=4 source=kit (not the shipped decision)")
        v = evidence.verdict(evidence.classify([], ("nosub", "pallas")))
        self.assertEqual(list(v["partial"]), ["nosub", "pallas"]); self.assertTrue(v["partial"]["pallas"].startswith("missing:"))
        v = evidence.verdict(evidence.classify([APPLIED, pallas_line(0, {})], ("nosub", "pallas")))
        self.assertEqual((list(v["partial"]), v["partial"]["pallas"]), (["pallas"], "missing: installed, no attention call served by the kernel (served=0)"))
        v = evidence.verdict(evidence.classify([APPLIED, pallas_line(7, {"backend_not_gpu": 1})], ("nosub", "pallas")))
        self.assertTrue(v["partial"]["pallas"].startswith("fallback: undeclared reason(s) {'backend_not_gpu': 1}"), v)
        self.assertEqual(evidence.verdict(evidence.classify([APPLIED, pallas_line(8, {"head_dim_lt_16": 1})], ("nosub", "pallas"))), {"partial": {}, "gated": {}, "skipped": {}})

    def test_buckets_and_stock_lines(self):
        rec = evidence.classify([APPLIED], ("nosub", "pallas"))
        self.assertEqual((rec["applied"], rec["gated"], rec["fallback"], rec["missing"]), (["nosub"], [], [], ["pallas"]))
        self.assertEqual(evidence.stock_lever_lines(["ok", GATED]), [GATED])
        self.assertEqual(evidence.summary(rec), "nosub=applied;pallas=missing"); self.assertEqual(evidence.summary(None), "n/a")



class TestStepAside(unittest.TestCase):
    """The kit rule (levers.py): a lever that cannot engage here steps aside BY NAME — one `state=skipped reason=cannot_run|no_attention_kernel
    detail=<blank-free token> source=install` line — and evidence classifies it `skipped` (named on EVIDENCE `skipped=`, verdict `skipped`,
    exit 0), never `missing` / `partial`; the attention-kernel lever literal equals the modules' own declarations."""

    def test_skipped_line_grammar_and_classification(self):
        from colabdesign_opt import levers, report
        line = levers.skipped_line("hoist_prev", levers.CANNOT_RUN, "HoistPrevError: colabdesign.af.design defines no `_recycle` (a = b)")
        self.assertRegex(line, r"^\[colabdesign-opt\] LEVER name=hoist_prev state=skipped reason=cannot_run impl=hoist_prev@kit origin=kit detail=HoistPrevError:_colabdesign\.af\.design_defines_no_`_recycle`_\(a_:_b\) source=install pid=\d+$")
        self.assertNotIn("= ", line.split("detail=")[1]); self.assertLessEqual(len(levers.detail_token("x " * 500)), levers.DETAIL_MAX)   # blank-free, bounded
        k = levers.skipped_line("proj", levers.NO_ATTENTION_KERNEL, "needs an attention-kernel lever")
        self.assertIn(" numerics=precision ", k); self.assertIn(" reason=no_attention_kernel ", k)      # impl / origin / numerics read off the module's own off line
        ev = evidence.classify([APPLIED, line], ("nosub", "hoist_prev"))
        self.assertEqual(ev["state"], {"hoist_prev": "skipped", "nosub": "applied"}); self.assertEqual((ev["skipped"], ev["missing"], ev["applied"]), (["hoist_prev"], [], ["nosub"]))
        self.assertTrue(ev["skipped_by"]["hoist_prev"].startswith("cannot_run: HoistPrevError:"), ev["skipped_by"])
        v = evidence.verdict(ev)
        self.assertEqual((v["partial"], v["gated"], list(v["skipped"])), ({}, {}, ["hoist_prev"]))       # exit 0: named, never partial
        self.assertIn(" missing=none skipped=hoist_prev ", report.evidence_line(ev))
        gated = line.replace("reason=cannot_run", "reason=gated")                                       # a size / memory gate's skipped word is NOT a step-aside: classified by the lever's own rules
        self.assertNotIn("hoist_prev", evidence.classify([APPLIED, gated], ("nosub", "hoist_prev"))["skipped"])

    def test_attention_kernel_literal_matches_the_modules(self):
        import importlib
        from colabdesign_opt import levers, registry
        declared = tuple(l for l in registry.ORDER if getattr(importlib.import_module(registry.LEVERS[l].module), "ATTENTION_FWD_KERNELS", ()))
        self.assertEqual(levers.ATTENTION_KERNEL_LEVERS, declared)                                       # pallas, triatt: the levers proj / txla compose on
        self.assertEqual(levers.attention_kernel_levers(), declared)
        for l in levers.NEEDS_ATTENTION_KERNEL + levers.NEEDS_GPU + levers.ATTENTION_KERNEL_LEVERS:
            self.assertIn(l, registry.ORDER)
        for l in levers.NEEDS_ATTENTION_KERNEL:                                                          # registry order installs the kernels before the levers that need them
            self.assertGreater(registry.ORDER.index(l), max(registry.ORDER.index(k) for k in levers.ATTENTION_KERNEL_LEVERS))
        self.assertEqual(set(levers.STEP_ASIDE_REASONS), {"cannot_run", "no_attention_kernel", "no_gpu"}); self.assertNotIn("gated", levers.STEP_ASIDE_REASONS)

if __name__ == "__main__":
    unittest.main()


class TestDeclaredAsideIsNotADefect(unittest.TestCase):
    """A census lever whose EVERY call stepped aside by a DECLARED word (served=0, fallback_by=<declared>:n) is ACTIVE with a NAMED aside
    (state `skipped`, verdict `skipped`, exit code unchanged) — e.g. `trimul` on BindCraft's AF2-ptm monomer design (use_multimer_design
    false: TriangleMultiplication without fused projections → not_fused:n), `opm_fold` when every call is gated_s_gt_32; an installed lever
    that served nothing WITHOUT a declared word, or with an undeclared one, stays a defect (exit 3)."""

    def test_every_call_aside_by_a_declared_word_exits_0_named(self):
        from colabdesign_opt import cli
        ev = evidence.classify([APPLIED, trimul_line(0, {"not_fused": 12})], ("nosub", "trimul"))
        self.assertEqual(ev["state"]["trimul"], "skipped"); self.assertEqual((ev["skipped"], ev["missing"], ev["fallback"]), (["trimul"], [], []))
        self.assertTrue(ev["skipped_by"]["trimul"].startswith("aside:not_fused:12"), ev["skipped_by"])
        v = evidence.verdict(ev)
        self.assertEqual((v["partial"], v["gated"], list(v["skipped"])), ({}, {}, ["trimul"]))
        self.assertEqual(cli.partial_exit(0, v), 0)                                            # rc 0: named, never NOT ACTIVE
        self.assertEqual(evidence.summary(ev), "nosub=applied;trimul=skipped")
        ev = evidence.classify([APPLIED, census_line("opm_fold", "opm_fold@kit", "tf32", 0, {"gated_s_gt_32": 52}, "none")], ("nosub", "opm_fold"))
        self.assertEqual(ev["state"]["opm_fold"], "skipped"); self.assertEqual(evidence.verdict(ev)["partial"], {})

    def test_served_nothing_without_a_declared_word_still_exits_3(self):
        from colabdesign_opt import cli
        ev = evidence.classify([APPLIED, trimul_line(0, {})], ("nosub", "trimul"))                # served=0, no reason at all: a defect
        self.assertEqual(ev["state"]["trimul"], "missing"); v = evidence.verdict(ev)
        self.assertEqual(v["partial"]["trimul"], "missing: installed, the kernels served no call (served=0)"); self.assertEqual(cli.partial_exit(0, v), cli.EXIT_NOT_ACTIVE)
        ev = evidence.classify([APPLIED, trimul_line(0, {"mystery": 12})], ("nosub", "trimul"))     # served=0 by an UNDECLARED word: a defect
        self.assertIn(ev["state"]["trimul"], ("missing", "fallback")); self.assertNotEqual(evidence.verdict(ev)["partial"], {})
        ev = evidence.classify([APPLIED, trimul_line(6, {"not_fused": 2})], ("nosub", "trimul"))    # served some, declared asides on the rest: applied
        self.assertEqual(ev["state"]["trimul"], "applied")


class TestStructuralAsideWordsAreDeclared(unittest.TestCase):
    """Every STRUCTURAL per-call step-aside word a lever can print is declared (the call ran stock's op BY NAME — never a defect): e.g. lever
    `proj` on BindCraft's AF2-ptm monomer design prints `fallback_by=head_dim_lt_16:1,no_gating:4` (template-stack attention has no gating)
    with served > 0 → applied, exit 0; a provider cell word (`cell_xla`) is declared by prefix; environment words stay fail-closed."""

    def test_proj_no_gating_on_the_monomer_model_is_applied(self):
        from colabdesign_opt import cli
        ev = evidence.classify([APPLIED, TRIATT_OK, proj_line(8, {"head_dim_lt_16": 1, "no_gating": 4})], ("nosub", "triatt", "proj"))
        self.assertEqual(ev["state"]["proj"], "applied", ev); v = evidence.verdict(ev); self.assertEqual(v["partial"], {}); self.assertEqual(cli.partial_exit(0, v), 0)

    def test_structural_words_and_cell_words_are_declared_environment_words_are_not(self):
        for lever, mk, ok_words, bad in (("trimul", lambda fb: trimul_line(6, fb), {"not_fused": 2, "dtype": 1, "cell_xla": 3}, "backend_not_gpu"),
                                          ("ln", lambda fb: census_line("ln", "layers_ln@kit", "none", 20, fb, "C128:20"), {"channels_not_pow2": 4, "rank": 1, "cell_xla": 2}, "platform"),
                                          ("transition", lambda fb: transition_line(5, fb), {"cell_xla": 1, "channels": 1}, "platform"),
                                          ("triatt", lambda fb: triatt_line(9, fb), {"below_keys_rule": 1, "no_pair_bias": 2, "cell_xla": 1}, "kernel_unavailable")):
            ev = evidence.classify([APPLIED, mk(ok_words)], ("nosub", lever))
            self.assertEqual(ev["state"][lever], "applied", (lever, ev["state"], ev.get(lever + "_fallback_by")))
            ev = evidence.classify([APPLIED, mk({**ok_words, bad: 1})], ("nosub", lever))
            self.assertEqual(ev["state"][lever], "fallback", (lever, bad)); self.assertIn(lever, evidence.verdict(ev)["partial"])

    def test_the_declared_sets_mirror_the_kernels(self):
        self.assertEqual(evidence.CELL_FALLBACK_PREFIX, "cell_")
        self.assertIn("no_gating", evidence.PROJ_DECLARED_FALLBACKS); self.assertIn("no_pair_bias", evidence.TRIATT_DECLARED_FALLBACKS)
        self.assertIn("dtype", evidence.TRIMUL_DECLARED_FALLBACKS); self.assertIn("rank", evidence.LN_DECLARED_FALLBACKS); self.assertIn("channels", evidence.TRANSITION_DECLARED_FALLBACKS)
