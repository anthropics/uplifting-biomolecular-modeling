"""Levers templ_trimul_tmk3 (EXACT; exact), templ_trimul_esm (TOLERANCE; fast | big) and templ_triatt_core (TOLERANCE; fast | big): the template embedder's c = 64 pair
stack through the shared core's providers (opt/forward/flashpairformer/src/ptx_c64_routes.py; protenix_opt 0.3.40).  protenix_opt 0.3.44: both TriMul levers bind the shared
core's TIER WORD (ARM E 'exact'; ARM T 'fast' | 'big' after the kit mode) — the tests state CLASS contracts (an exact-class row under 'exact', a tolerance-or-better row under
'fast' / 'big', the by-name routes intact), never a row name: rows flip with the core's cells.

CPU facts only (no torch, no GPU): the registry / modes / report rows, env.sh's exports per arm (the module's FPF_OPS trimul entries FOLLOW the
arm's own; the attention word is ARM T only; =0|off leaves a lever out), the module's pure helpers (the inner-provider parse, the shape
predicates, the core floor), and its import-time behaviour in an interpreter without protenix (the TriMul lever is built from the core; the
attention site steps aside BY NAME — never an exception out of the import)."""
import importlib
import os
import subprocess
import sys
import unittest

from protenix_opt import kits, modes, registry, report
from protenix_opt.registry import LEVERS

FPF = kits.kit_dir("flashpairformer")
SRC = os.path.join(FPF, "src")
TM, TE, TA = "templ_trimul_tmk3", "templ_trimul_esm", "templ_triatt_core"


def _source(arm, extra=None):
    """FPF_OPS / PTX_TEMPL_TRIMUL / PTX_TEMPL_TRIATT after sourcing env.sh under ARM=<arm> in a clean bash (no nvidia-smi, no torch probe needed)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "ARM": arm, "FPF_HOME": FPF}
    env.update(extra or {})
    out = subprocess.run(["bash", "-c", 'source "$FPF_HOME/env.sh" >/dev/null 2>&1; printf "%s\\n%s\\n%s\\n" "${FPF_OPS:-}" "${PTX_TEMPL_TRIMUL:-<unset>}" "${PTX_TEMPL_TRIATT:-<unset>}"'],
                         env=env, capture_output=True, text=True, check=True).stdout.split("\n")
    return {"FPF_OPS": out[0], "PTX_TEMPL_TRIMUL": out[1], "PTX_TEMPL_TRIATT": out[2]}


def _module():
    """ptx_c64_routes imported fresh with the auto-apply guard set (apply() is then called by the test with its own environ)."""
    os.environ["PTX_C64_ROUTES_NO_AUTOAPPLY"] = "1"
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    sys.modules.pop("ptx_c64_routes", None)
    return importlib.import_module("ptx_c64_routes")


class TestRows(unittest.TestCase):
    def test_registry_rows(self):
        self.assertEqual(LEVERS[TM].tier, registry.EXACT); self.assertEqual(LEVERS[TM].env_keys, ("PTX_TEMPL_TRIMUL",)); self.assertEqual(LEVERS[TM].probe, "env")
        self.assertEqual(LEVERS[TE].tier, registry.TOLERANCE); self.assertEqual(LEVERS[TE].env_keys, ("PTX_TEMPL_TRIMUL",)); self.assertEqual(LEVERS[TE].probe, "env")
        self.assertEqual((LEVERS[TM].words, LEVERS[TE].words), ((("PTX_TEMPL_TRIMUL", "tmk3"),), (("PTX_TEMPL_TRIMUL", "esm"),))); self.assertEqual(LEVERS[TM].replaced_by, (TE,))
        self.assertEqual(LEVERS[TA].tier, registry.TOLERANCE); self.assertEqual(LEVERS[TA].env_keys, ("PTX_TEMPL_TRIATT",)); self.assertEqual(LEVERS[TA].probe, "env")
        for n in (TM, TE, TA):
            self.assertFalse(LEVERS[n].conditional); self.assertFalse(LEVERS[n].extra); self.assertFalse(LEVERS[n].row_dependent)
            self.assertIn("ptx_c64_routes.py", LEVERS[n].source); self.assertIn("env.sh", LEVERS[n].source)

    def test_modes_membership(self):
        """The EXACT-class TriMul route is in all three tiers; the TOLERANCE-class attention route in fast and big only (R3: exact = exact-class levers; fast ⊃ exact; big ⊇ fast's set)."""
        self.assertIn(TM, modes.MODES["exact"]); self.assertNotIn(TM, modes.MODES["fast"]); self.assertNotIn(TM, modes.big_levers())
        self.assertNotIn(TE, modes.MODES["exact"]); self.assertIn(TE, modes.MODES["fast"]); self.assertIn(TE, modes.big_levers())
        self.assertNotIn(TA, modes.MODES["exact"]); self.assertIn(TA, modes.MODES["fast"]); self.assertIn(TA, modes.big_levers())

    def test_report_rows(self):
        self.assertEqual(report.STRATEGY_IDS[TM], "F2.fpf_trimul_exact"); self.assertEqual(report.IMPL[TM], ("opt_core/kernels/trimul", "core"))
        self.assertEqual(report.STRATEGY_IDS[TE], "F2.fpf_trimul_fast"); self.assertTrue(report.IMPL[TE][0].startswith("opt_core/kernels/trimul"), report.IMPL[TE]); self.assertEqual(report.IMPL[TE][1], "core")   # class: a shared-core TriMul kernel (the tier row's family), not one package's name
        self.assertEqual(report.STRATEGY_IDS[TA], "F1.flash_triatt"); self.assertEqual(report.IMPL[TA], ("opt_core/kernels/triattn", "core"))


class TestEnvSh(unittest.TestCase):
    MINE = "trimul_out=ptx_c64_routes:trimul_out,trimul_in=ptx_c64_routes:trimul_in"

    def test_arm_e_exports_the_trimul_word_only_and_the_entries_follow_the_arms(self):
        r = _source("E")
        self.assertEqual(r["PTX_TEMPL_TRIMUL"], "tmk3"); self.assertEqual(r["PTX_TEMPL_TRIATT"], "<unset>")
        self.assertTrue(r["FPF_OPS"].endswith("," + self.MINE), r["FPF_OPS"])
        self.assertTrue(r["FPF_OPS"].startswith("trimul_out=ptx_trimul_routes:trimul_out,"), r["FPF_OPS"])      # the arm's own exact provider first (the module's inner provider)

    def test_arm_t_exports_both_words(self):
        r = _source("T")
        self.assertEqual(r["PTX_TEMPL_TRIMUL"], "esm"); self.assertEqual(r["PTX_TEMPL_TRIATT"], "core")
        self.assertEqual(_source("T", {"PTX_TEMPL_TRIMUL": "tmk3"})["PTX_TEMPL_TRIMUL"], "tmk3")               # a caller's word is kept (the exact row under a T mode)
        self.assertEqual(_source("E", {"PTX_TEMPL_TRIMUL": "esm"})["PTX_TEMPL_TRIMUL"], "esm")                 # and the other way (the exact tier's det pair decides whether that is wise)
        self.assertEqual(_source("T", {"PTX_TEMPL_TRIMUL": "bogus"})["PTX_TEMPL_TRIMUL"], "esm")               # an unknown word is named on stderr and reads as the arm's default
        self.assertTrue(r["FPF_OPS"].endswith("," + self.MINE), r["FPF_OPS"]); self.assertIn("fpf_smalln:trimul_c256", r["FPF_OPS"])

    def test_arm_e_never_keeps_a_stale_attention_word(self):
        self.assertEqual(_source("E", {"PTX_TEMPL_TRIATT": "core"})["PTX_TEMPL_TRIATT"], "<unset>")

    def test_attention_words(self):
        for w in ("native", "sm90a", "k2b"):
            self.assertEqual(_source("T", {"PTX_TEMPL_TRIATT": w})["PTX_TEMPL_TRIATT"], w)
        self.assertEqual(_source("T", {"PTX_TEMPL_TRIATT": "bogus"})["PTX_TEMPL_TRIATT"], "core")                  # an unknown word is named on stderr and reads as core

    def test_off_words_leave_the_levers_out(self):
        r = _source("T", {"PTX_TEMPL_TRIMUL": "0", "PTX_TEMPL_TRIATT": "off"})
        self.assertEqual((r["PTX_TEMPL_TRIMUL"], r["PTX_TEMPL_TRIATT"]), ("<unset>", "<unset>")); self.assertNotIn("ptx_c64_routes", r["FPF_OPS"])
        r = _source("T", {"PTX_TEMPL_TRIMUL": "off"})                                   # the attention word alone still needs the module imported: the entries stay (pure pass-through for TriMul)
        self.assertEqual(r["PTX_TEMPL_TRIMUL"], "<unset>"); self.assertIn(self.MINE, r["FPF_OPS"])

    def test_extra_ops_still_merge_last(self):
        r = _source("T", {"FPF_OPS_EXTRA": "outer_product_mean=x.y:fn"})
        self.assertTrue(r["FPF_OPS"].endswith(self.MINE + ",outer_product_mean=x.y:fn"), r["FPF_OPS"])


class TestModule(unittest.TestCase):
    def test_pure_helpers(self):
        M = _module()
        arm_t = "trimul_out=fpf_smalln:trimul_c256,trimul_in=fpf_smalln:trimul_c256,trimul_out=ptx_c64_routes:trimul_out,trimul_in=ptx_c64_routes:trimul_in"
        self.assertEqual(M.parse_inner(arm_t), {"trimul_out": "fpf_smalln:trimul_c256", "trimul_in": "fpf_smalln:trimul_c256"})
        self.assertEqual(M.parse_inner("trimul_out=ptx_c64_routes:trimul_out,trimul_in=ptx_c64_routes:trimul_in"), {"trimul_out": "stock", "trimul_in": "stock"})
        arm_e = "trimul_out=ptx_trimul_routes:trimul_out,trimul_in=ptx_trimul_routes:trimul_in," + arm_t.split(",", 2)[2]
        self.assertEqual(M.parse_inner(arm_e)["trimul_in"], "ptx_trimul_routes:trimul_in")
        self.assertTrue(M.core_ok("0.5.47.0")); self.assertTrue(M.core_ok("0.5.48.1")); self.assertTrue(M.core_ok("0.6.0")); self.assertFalse(M.core_ok("0.5.46.9")); self.assertFalse(M.core_ok("0.5.35.1")); self.assertFalse(M.core_ok(None))
        self.assertEqual(M.version_tuple("0.5.32rc1"), (0, 5)); self.assertFalse(M.core_ok("0.5.32rc1"))

        class Mod:
            c_z, c_hidden = 64, 64
        self.assertTrue(M.is_template_trimul(Mod(), "cuequivariance")); self.assertFalse(M.is_template_trimul(Mod(), "torch"))
        Mod.c_z = 256; self.assertFalse(M.is_template_trimul(Mod(), "cuequivariance"))
        self.assertTrue(M.is_template_triatt(2, 32)); self.assertFalse(M.is_template_triatt(8, 32)); self.assertFalse(M.is_template_triatt(1, 32)); self.assertFalse(M.is_template_triatt(2, 64))
        self.assertIs(M.TRIATT_PREFER, M.TRIATT_WORDS["core"]); self.assertEqual(M.TRIMUL_MIN_TOKENS, 101); self.assertEqual(M.MIN_CORE, (0, 5, 47))
        # class contracts (protenix_opt 0.3.44): the switch words name a Lever CLASS word of opt_core.trimul and a registry lever; ARM T binds the core's tier word after the kit mode
        from opt_core import trimul as T
        self.assertEqual(M.TRIMUL_WORDS["tmk3"], ("exact", TM)); self.assertEqual(M.TRIMUL_WORDS["esm"][1], TE); self.assertIn(M.TRIMUL_WORDS["esm"][0], tuple(m for m in T.MODES if m not in ("stock", "exact")))
        self.assertEqual((M.MODE_ENV, M.ARM_T_TIERS, M.TIER_CORE), ("PROTENIX_OPT", ("fast", "big"), (0, 5, 86))); self.assertGreaterEqual(M.TIER_CORE, M.MIN_CORE)
        self.assertEqual([M.tier_word(e) for e in ({"PROTENIX_OPT": "big"}, {"PROTENIX_OPT": "fast"}, {}, {"PROTENIX_OPT": "exact"}, {"PROTENIX_OPT": " BIG"}, {"PROTENIX_OPT": "off"})], ["big", "fast", "fast", "fast", "big", "fast"])
        self.assertEqual(M.arm_t_binding("0.5.86.0", {"PROTENIX_OPT": "big"}), ("tier", "big")); self.assertEqual(M.arm_t_binding("0.5.88.0", {}), ("tier", "fast")); self.assertEqual(M.arm_t_binding("0.6", {"PROTENIX_OPT": "fast"}), ("tier", "fast"))
        self.assertEqual(M.arm_t_binding("0.5.85.9", {"PROTENIX_OPT": "big"})[0], "package"); self.assertIn("trimul_esm_shapes", M.arm_t_binding("0.5.47.0", {})[1]); self.assertEqual(M.arm_t_binding(None, {})[0], "package")
        self.assertTrue(M.core_at("0.5.86.0", M.TIER_CORE)); self.assertFalse(M.core_at("0.5.85.12", M.TIER_CORE)); self.assertFalse(M.core_at(None, M.TIER_CORE)); self.assertFalse(M.core_at("0.5.86rc1", M.TIER_CORE))
        self.assertEqual(M.ESM_KERNEL.rsplit(".", 1)[0], "opt_core.kernels"); self.assertTrue(callable(M._esm_provider))            # the package route below TIER_CORE is kept by name
        from protenix_opt import _core
        from protenix_opt._core_gate import read_table, version_tuple
        pin = read_table(_core.PYPROJECT, "tool.opt_core")["version"]
        self.assertGreaterEqual(version_tuple(pin)[:3], M.TIER_CORE, pin)                                                            # the shipped pairing always binds the tier word (the package route is for a tree run against an older core by hand)
        # class contract (protenix_opt 0.3.49): every attention switch word resolves through the provider (opt_core.kernels.triattn) by its TIER WORD — the row is select()'s per
        # cell; a word's row preference (None = the cell alone decides) only narrows the provider's own rows in the kit's order; the one-row words pin one provider row each
        from opt_core.kernels import triattn as KA
        self.assertEqual((M.TRIATT_TIER, M.TRIATT_BIND, M.TRIATT_GATE_ENV), ("fast", "tier:fast", "PTX_T_MIN_TOKENS")); self.assertIn(M.TRIATT_TIER, KA.TIER_WORDS)
        self.assertEqual(sorted(M.TRIATT_WORDS), ["core", "k2b", "native", "sm90a"]); self.assertIs(M.TRIATT_WORDS["core"], M.TRIATT_CORE_PREFER)
        for word, prefer in M.TRIATT_WORDS.items():
            self.assertTrue(prefer is None or (isinstance(prefer, tuple) and prefer and all(r.split("@")[0] in KA.KERNEL_ROWS for r in prefer)), (word, prefer))
            if word != "core":
                self.assertEqual(len(prefer), 1, word)                                                                          # attribution words: exactly one row
        self.assertEqual([M.TRIATT_WORDS[w][0].split("@")[0] for w in ("native", "sm90a", "k2b")], ["triattn_native", "cuda_sm90a", "k2b"])
        self.assertEqual((M.prefer_token(None), M.prefer_token(()), M.prefer_token(("k2b",)), M.prefer_token(("triattn_native@v11", "k2b"))), ("none", "none", "k2b", "triattn_native@v11+k2b"))

    def test_apply_without_protenix_builds_the_trimul_lever_and_steps_the_attention_site_aside_by_name(self):
        M = _module()
        st = M.apply({"FPF_OPS": "trimul_out=ptx_trimul_routes:trimul_out,trimul_in=ptx_trimul_routes:trimul_in,trimul_out=ptx_c64_routes:trimul_out,trimul_in=ptx_c64_routes:trimul_in",
                      "PTX_TEMPL_TRIMUL": "tmk3", "PTX_TEMPL_TRIATT": "core", "PROTENIX_OPT": "exact"})
        self.assertEqual(st["trimul"], "tmk3"); self.assertIsNotNone(M._TRIMUL_LEVER)                       # opt_core.trimul.Lever built (stdlib-only construction)
        self.assertEqual((st["trimul_bind"], M._TRIMUL_LEVER.mode, M._TRIMUL_LEVER.provider.kernel), ("tier:exact", "exact", "trimul"))   # ARM E: the core's exact tier row BY WORD (class: opt_core.trimul.by_word over kernels.trimul)
        self.assertTrue(st["triatt"].startswith("aside:"), st["triatt"])                                    # no protenix in this interpreter: aside by name, no exception
        st2 = _module().apply({"PTX_TEMPL_TRIATT": "k2b", "PTX_T_MIN_TOKENS": "300"})                          # the word's rows and ARM T's size gate are read before the install
        self.assertEqual((st2["triatt_prefer"], st2["triatt_min_tokens"]), (("k2b",), 300)); self.assertTrue(st2["triatt"].startswith("aside:"), st2["triatt"])
        self.assertEqual(st["inner"], {"trimul_out": "ptx_trimul_routes:trimul_out", "trimul_in": "ptx_trimul_routes:trimul_in"})
        tl, ta = M._trimul_line(), M._triatt_line()
        self.assertTrue(tl.startswith("[ptx_c64_routes] LEVER name=F2.trimul state=on ")); self.assertIn(" lever=templ_trimul_tmk3", tl); self.assertIn(" min_tokens=101", tl); self.assertIn(" word=tmk3 bind=tier:exact esm_rerouted=none", tl); self.assertIn(" mode=exact", tl)
        self.assertTrue(ta.startswith("[ptx_c64_routes] LEVER name=F1.flash_triatt state=skipped reason=aside:")); self.assertIn(" lever=templ_triatt_core", ta)

    def _core_version(self):
        import opt_core
        return getattr(opt_core, "__version__", None)

    def test_the_attention_lever_line_names_the_tier_binding_and_the_preference(self):
        """protenix_opt 0.3.49: the F1.flash_triatt LEVER line carries ``prefer=<rows|none> bind=tier:fast`` (like the TriMul line's bind=) next to the row SERVED (impl=)."""
        M = _module()
        M.STATE.update(triatt="core", triatt_switch="core", triatt_prefer=("triattn_native@v11", "cuda_sm90a", "k2b"), triatt_bind=M.TRIATT_BIND, triatt_min_tokens=300)
        M.STATE["triatt_served"]["triattn_native"] = 80; M.STATE["triatt_shapes"]["400x2x32/bfloat16"] = 80
        line = M._triatt_line()
        self.assertTrue(line.startswith("[ptx_c64_routes] LEVER name=F1.flash_triatt state=on impl=triattn:triattn_native@fast origin=core served=80 "), line)
        self.assertIn(" word=core prefer=triattn_native@v11+cuda_sm90a+k2b bind=tier:fast gate=ok lever=templ_triatt_core", line)
        M.STATE.update(triatt_prefer=None); M.STATE["triatt_served"].clear(); M.STATE["triatt_served"]["cuda_sm90a"] = 80                # the tier word alone: prefer=none, the row served still named
        line = M._triatt_line()
        self.assertIn(" impl=triattn:cuda_sm90a@fast ", line); self.assertIn(" prefer=none bind=tier:fast ", line)
        M.STATE.update(triatt=None, triatt_switch=None, triatt_bind="none"); M.STATE["triatt_served"].clear(); M.STATE["triatt_shapes"].clear()   # the word absent: nothing installed, no bind token
        line = M._triatt_line()
        self.assertTrue(line.startswith("[ptx_c64_routes] LEVER name=F1.flash_triatt state=skipped "), line); self.assertNotIn("bind=tier", line)

    def test_every_attention_word_resolves_through_the_provider_by_the_tier_word(self):
        """Class contract on the core's cell table (offline select, no device): under the tier word each switch word's preference yields a FAST-class row of the provider on both
        cards at the template sizes, or the provider refuses BY NAME with a fallback row (a one-row word off its card: cuda_sm90a on cc 8.0) — no row name is pinned for ``core``."""
        M = _module()
        from opt_core.kernels import triattn as KA
        for cc in ((9, 0), (8, 0)):
            for word, prefer in M.TRIATT_WORDS.items():
                for n in (400, 800, 1200):
                    try:
                        sel = KA.select(cc, "bf16", 32, 2, n, word=M.TRIATT_TIER, prefer=prefer)
                    except KA.Refusal as r:
                        self.assertTrue(r.fallback, (cc, word, n, str(r))); continue
                    self.assertEqual((sel.word, sel.cls), (M.TRIATT_TIER, "fast"), (cc, word, n, sel))
                    self.assertIn(sel.row, KA.KERNEL_ROWS, (cc, word, n, sel))
                    if prefer is not None:
                        self.assertIn(sel.row, {r.split("@")[0] for r in prefer}, (cc, word, n, sel))
        for cc in ((9, 0), (8, 0)):                                                                                             # opt_core >= 0.5.114.0 carries the big word on this face (== fast's rows today): the route binds the
            sel_b = KA.select(cc, "bf16", 32, 2, 800, word="big"); sel_f = KA.select(cc, "bf16", 32, 2, 800, word="fast")   # kit mode's own word under big (tier_word()), never fast-under-big
            self.assertIn(sel_b.row, KA.KERNEL_ROWS, (cc, sel_b)); self.assertEqual(sel_b.row, sel_f.row, (cc, sel_b, sel_f))
        self.assertEqual(M.tier_word({"PROTENIX_OPT": "big"}), "big"); self.assertEqual(M.tier_word({"PROTENIX_OPT": "fast"}), "fast"); self.assertEqual(M.tier_word({"PROTENIX_OPT": "exact"}), "fast")

    def test_the_esm_word_binds_the_core_tier_row_after_the_kit_mode(self):
        """ARM T (protenix_opt 0.3.44): PTX_TEMPL_TRIMUL=esm builds the TOLERANCE-class lever over ``opt_core.trimul.by_word(weights_of, <tier>)`` — tier ``big`` under the kit
        mode big, ``fast`` under fast / any other / no mode word; stdlib-only construction (no torch needed); the line names the binding and, once a call is served, the row."""
        M = _module()
        if not M.core_at(self._core_version(), M.TIER_CORE):
            self.skipTest(f"opt_core {self._core_version()} < {M.TIER_CORE}: the word esm keeps the package route there (test_below_tier_core_...)")
        for mode, tier in (("big", "big"), ("fast", "fast"), ("", "fast"), ("exact", "fast")):
            M = _module()
            st = M.apply({"PTX_TEMPL_TRIMUL": "esm", "PROTENIX_OPT": mode} if mode else {"PTX_TEMPL_TRIMUL": "esm"})
            self.assertFalse(str(st["trimul"]).startswith("aside:"), st["trimul"])
            self.assertEqual((st["trimul"], st["trimul_lever"], st["trimul_bind"]), ("esm", "templ_trimul_esm", f"tier:{tier}")); self.assertEqual(M._TRIMUL_LEVER.mode, "fast")   # the Lever's class word stays the tolerance class
            self.assertEqual((M._TRIMUL_LEVER.provider.name, M._TRIMUL_LEVER.provider.kernel), (f"trimul:{tier}", "trimul"))                                                 # by_word over kernels.trimul — the served row fills impl=trimul:<row>@<tier>/<cell>
            tl = M._trimul_line(); self.assertIn(" lever=templ_trimul_esm", tl); self.assertIn(f" word=esm bind=tier:{tier} esm_rerouted=none", tl); self.assertIn(" mode=fast", tl); self.assertNotIn("trimul_esm_shapes", tl)

    def test_below_tier_core_the_esm_word_keeps_the_package_route_by_name(self):
        """opt_core older than TIER_CORE (0.5.86): the word esm builds the earlier route — the ``opt_core.kernels.trimul_esm_shapes`` package directly (``_esm_provider``) — BY NAME
        (one stderr line, ``bind=package:trimul_esm_shapes``); never silent, never a refusal."""
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch absent: the package route imports it at apply")
        import opt_core
        real = opt_core.__version__
        try:
            opt_core.__version__ = "0.5.69.2"
            M = _module(); st = M.apply({"PTX_TEMPL_TRIMUL": "esm", "PROTENIX_OPT": "big"})
        finally:
            opt_core.__version__ = real
        if str(st["trimul"]).startswith("aside:"):
            self.skipTest(f"this interpreter's opt_core cannot build the package provider: {st['trimul']}")
        self.assertEqual((st["trimul"], st["trimul_lever"], st["trimul_bind"]), ("esm", "templ_trimul_esm", "package:trimul_esm_shapes")); self.assertEqual(M._TRIMUL_LEVER.mode, "fast")
        self.assertEqual((M._TRIMUL_LEVER.provider.name, M._TRIMUL_LEVER.provider.kernel), ("trimul_esm_shapes", "trimul_esm_shapes"))
        tl = M._trimul_line(); self.assertIn(" word=esm bind=package:trimul_esm_shapes esm_rerouted=none lever=templ_trimul_esm", tl); self.assertIn(" mode=fast", tl)

    def test_tier_words_resolve_by_class_on_the_core_cell_table(self):
        """CLASS contracts on the shared core's own selection for the template cells (cc, bf16, C64, H64, N, direction), CPU-side (kernels.trimul.select is pure): 'exact' -> an
        EXACT-class row on a stack where one is vouched (the kit's H100 image; cc 8.0), 'fast' / 'big' -> a row of the fast tier (tolerance-or-better); a stack the table never
        byte-tested keeps an exact-class row too (tmk3_exact is bitwise by construction) — no row NAME is pinned here."""
        M = _module()
        if not M.core_at(self._core_version(), M.TIER_CORE):
            self.skipTest(f"opt_core {self._core_version()} < {M.TIER_CORE}")
        from opt_core.kernels import trimul as KT
        tol_or_better, exact_rows = tuple(KT.table()["tiers"]["fast"]), tuple(KT.EXACT_ROWS)
        stacks = {"H100:2.13.0+cu130/3.7.1/cueq0.11.1": (9, 0), "H100:2.12.0+cu130/3.7.0/cueq0.10.0": (9, 0), "A100:2.13.0+cu130/3.7.1/cueq0.11.1": (8, 0)}   # the kit image's stack word (kernels.trimul stack grammar), a sibling H100 stack, the A100 image
        for st, cc in stacks.items():
            for N in (256, 400, 612, 800, 1200):
                for d in ("outgoing", "incoming"):
                    ex = KT.select(cc, "bf16", 64, 64, N, d, word="exact", stack=st)
                    self.assertIn(ex.row, exact_rows, (st, N, d, KT.describe(ex)))
                    for tier in M.ARM_T_TIERS:
                        sel = KT.select(cc, "bf16", 64, 64, N, d, word=tier, stack=st)
                        self.assertIn(sel.row, tol_or_better, (st, tier, N, d, KT.describe(sel))); self.assertIsNotNone(sel.cell, (st, tier, N, d))

    def test_absent_words_install_nothing(self):
        M = _module()
        st = M.apply({"FPF_OPS": "trimul_out=ptx_c64_routes:trimul_out,trimul_in=ptx_c64_routes:trimul_in"})
        self.assertEqual((st["trimul"], st["triatt"]), ("off", "off")); self.assertIsNone(M._TRIMUL_LEVER); self.assertEqual(st["inner"]["trimul_out"], "stock")

    def test_an_old_core_or_an_unknown_word_steps_aside_by_name(self):
        M = _module()
        self.assertFalse(M.core_ok("0.5.26.2"))
        st = M.apply({"PTX_TEMPL_TRIMUL": "v9", "PTX_TEMPL_TRIATT": "zzz"})
        self.assertEqual(st["trimul"], "aside:unknown_word:v9"); self.assertEqual(st["triatt"], "aside:unknown_word:zzz"); self.assertIsNone(M._TRIMUL_LEVER)


if __name__ == "__main__":
    unittest.main()
