"""The multi-GPU route's row-chunking levers (esmfold2_opt.rowchunk) through the kit's own mechanisms, on CPU: registry names filed under
strategies, the big set gaining them at n_gpu > 1 only (rowpair.FOR_ROUTE -> modes.route_add), the ablation variable subtracting any
one by name, the installer binding members by name and refusing by name, stack.settle_route turning a member that did not bind into a
partial application (exit 3), the LEVER lines, the knobs, and the n_gpu 1 import graph (no rowchunk module loads). No GPU, no kit server."""
import os
import subprocess
import sys
import types
import unittest

from esmfold2_opt import ablation, modes, registry, report, rowpair, stack, tp
from esmfold2_opt.rowchunk import install as rci

MEMBERS = ("injrows", "confrows", "confbf16", "pdeskip", "confmem", "biasfree", "zbf16")


def _kit():
    from esmfold2_opt.tests._stubs import kit_or_none
    return kit_or_none()


class RegistryFilesTheMembers(unittest.TestCase):
    def test_names_fields_strategies(self):
        self.assertEqual(tuple(rowpair.FOR_ROUTE), MEMBERS); self.assertEqual(rci.MEMBERS, MEMBERS)
        for n in MEMBERS:
            lv = registry.LEVERS[n]
            self.assertEqual((lv.field, lv.probe), (registry.FIELD_RC, ("rc", n)))
            self.assertIn(n, registry.STRATEGY); self.assertTrue(lv.kit_file.startswith("esmfold2_opt."))
        self.assertEqual({n: registry.LEVERS[n].tier_vs_kit_line for n in MEMBERS},
                         {"injrows": "T2", "confrows": "T2", "confbf16": "T2", "zbf16": "T2", "pdeskip": "bitwise", "confmem": "bitwise", "biasfree": "bitwise"})

    def test_flags_ride_on_confrows_and_confbf16_on_confmem(self):
        self.assertEqual(ablation.DEPENDS["confbf16"], ("confrows", "confmem"))
        for n in ("pdeskip", "confmem", "zbf16"):
            self.assertEqual(ablation.DEPENDS[n], ("confrows",))
        self.assertEqual(rci.RIDES_ON_CONFROWS, ("confbf16", "pdeskip", "confmem", "zbf16"))
        self.assertEqual(rci.RIDES_ON, {k: ablation.DEPENDS[k] for k in ("confbf16", "pdeskip", "confmem", "zbf16")})

    def test_route_tables(self):
        self.assertEqual(rowpair.for_route(1), ()); self.assertEqual(rowpair.for_route(2), MEMBERS)
        self.assertEqual(tp.for_route(1), ()); self.assertEqual(tp.for_route(4), MEMBERS)
        self.assertEqual(modes.route_add(1), []); self.assertEqual(modes.route_add(8), list(MEMBERS))


class TheSetGainsThemAtNGpuAbove1Only(unittest.TestCase):
    def setUp(self):
        self.kit = _kit()
        if self.kit is None:
            self.skipTest("the kit tree (driver/ef2_server.py) is not importable here")

    def test_big_n_gpu_2_names_every_member_n_gpu_1_none(self):
        r2 = modes.resolve("big", "fast", self.kit, n_gpu=2)
        self.assertEqual(r2.for_route, MEMBERS); self.assertEqual(tuple(r2.levers[-7:]), MEMBERS)
        self.assertTrue(set(MEMBERS) <= set(r2.levers_for_variant))
        for variant in ("fast", "full_msa", "full_nomsa"):
            r1 = modes.resolve("big", variant, self.kit)
            self.assertFalse(set(MEMBERS) & set(r1.levers)); self.assertEqual(r1.for_route, ())
        for mode in ("fast", "exact"):
            self.assertFalse(set(MEMBERS) & set(modes.resolve(mode, "fast", self.kit).levers))

    # The declared validity of subtracting members by name at n_gpu 2 (ablation.DEPENDS): a member may leave the set alone unless another
    # member still in the set rides on it — then the AblationError names that member, and naming both is valid.
    SINGLES_VALID = ("injrows", "confbf16", "pdeskip", "biasfree", "zbf16")
    SINGLES_REFUSED = {"confrows": ("confbf16", "pdeskip", "confmem", "zbf16"), "confmem": ("confbf16",)}
    PAIRS_VALID = (("confmem", "confbf16"), ("confrows", "confbf16", "pdeskip", "confmem", "zbf16"), ("confmem", "confbf16", "zbf16"), ("injrows", "biasfree"),
                   ("pdeskip", "zbf16"), ("confbf16", "zbf16", "pdeskip"))
    PAIRS_REFUSED = {("confrows", "confmem"): ("confbf16", "pdeskip", "zbf16"), ("confmem", "pdeskip"): ("confbf16",), ("confrows", "confbf16", "confmem"): ("pdeskip", "zbf16")}

    def test_every_single_subtraction_has_its_declared_outcome(self):
        self.assertEqual(set(self.SINGLES_VALID) | set(self.SINGLES_REFUSED), set(MEMBERS))
        for n in self.SINGLES_VALID:
            r = modes.resolve("big", "fast", self.kit, n_gpu=2, ablate=n)
            self.assertEqual(r.ablate, (n,)); self.assertEqual(set(MEMBERS) - set(r.levers), {n}, n); self.assertIn(n, r.levers_off)
        for n, riders in self.SINGLES_REFUSED.items():
            with self.assertRaises(ablation.AblationError) as cm:
                modes.resolve("big", "fast", self.kit, n_gpu=2, ablate=n)
            msg = str(cm.exception)
            self.assertIn("rides on", msg); self.assertTrue(any(f"lever {d!r}" in msg for d in riders), (n, msg))

    def test_the_pairs_the_table_implies(self):
        for combo in self.PAIRS_VALID:
            r = modes.resolve("big", "fast", self.kit, n_gpu=2, ablate=",".join(combo))
            self.assertEqual(set(MEMBERS) - set(r.levers), set(combo), combo)
        for combo, riders in self.PAIRS_REFUSED.items():
            with self.assertRaises(ablation.AblationError) as cm:
                modes.resolve("big", "fast", self.kit, n_gpu=2, ablate=",".join(combo))
            self.assertTrue(any(f"lever {d!r}" in str(cm.exception) for d in riders), (combo, str(cm.exception)))

    def test_confrows_alone_names_its_dependents_and_n_gpu_1_refuses_the_name(self):
        with self.assertRaises(ablation.AblationError) as cm:
            modes.resolve("big", "fast", self.kit, n_gpu=2, ablate="confrows")
        self.assertIn("rides on confrows", str(cm.exception))
        with self.assertRaises(ablation.AblationError) as cm:
            modes.resolve("big", "fast", self.kit, n_gpu=1, ablate="pdeskip")
        self.assertIn("not in mode big's set", str(cm.exception))

    def test_dry_run_line_names_the_members(self):
        r2 = modes.resolve("big", "fast", self.kit, n_gpu=2)
        rep = {"active": True, "applied": "dry-run", "mode": "big", "variant": "fast", "server_mode": r2.server_mode, "levers_planned": list(r2.levers),
               "levers_applied": [], "levers_fallback": [], "gpu": {"name": None, "sm": None}, "resolution": r2, "n_gpu": 2}
        old = report.N_GPU["P"]
        try:
            report.set_n_gpu(2)
            line = report.activation_line(rep)
        finally:
            report.set_n_gpu(old)
        levers = [w for w in line.split() if w.startswith("levers=")][0]
        for n in MEMBERS:
            self.assertIn(n, levers.split("=", 1)[1].split(","))


class _SH(object):
    def __init__(self):
        self.calls = 0
    def sample(self, *a, **k):
        self.calls += 1
        return {"sample_atom_coords": None}


class _Model(object):
    def __init__(self):
        self.structure_head = _SH()


class TheInstallerBindsAndRefusesByName(unittest.TestCase):
    def setUp(self):
        from esmfold2_opt.rowchunk import confmem, confrows, zbf16rows
        self.mods = (confmem, confrows, zbf16rows)
        self.state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rci.STATE.items()}
        self.inj = dict(rowpair.INJ)
        self.model = _Model()
        for k in list(os.environ):
            if k.startswith("EF2_ROWPAIR_") and k.endswith(("_MB", "_MIN_TOKENS")):
                del os.environ[k]

    def tearDown(self):
        confmem, confrows, zbf16rows = self.mods
        zbf16rows.unapply(); confrows.unapply(); confmem.unapply(self.model)
        rci.STATE.clear(); rci.STATE.update(self.state); rowpair.INJ.clear(); rowpair.INJ.update(self.inj)

    def test_p1_binds_nothing_and_says_why(self):
        rec = rci.install(self.model, 1)
        self.assertFalse(any(rec["levers"].values())); self.assertTrue(all("n_gpu=1" in w for w in rec["why"].values()))
        self.assertFalse(rowpair.INJ["on"]); self.assertIsNone(rci.counters())

    def test_p2_binds_every_named_member(self):
        os.environ["EF2_ROWPAIR_CONF_MB"] = "256"
        try:
            rec = rci.install(self.model, 2, MEMBERS)
        finally:
            del os.environ["EF2_ROWPAIR_CONF_MB"]
        self.assertEqual(rec["levers"], {n: True for n in MEMBERS}); self.assertEqual(rci.levers_on(), {n: True for n in MEMBERS})
        self.assertEqual((rowpair.INJ["on"], rowpair.INJ["min_tokens"]), (True, 1024))
        self.assertEqual(rec["knobs"], {"min_tokens": 1024, "inject_mb": 512.0, "conf_mb": 256.0, "zbf16_mb": 1024.0})
        self.assertEqual(rci.evidence("confrows"), {"min_tokens": 1024, "block_mb": 256}); self.assertEqual(rci.evidence("injrows")["block_mb"], 512)
        self.model.structure_head.sample(); self.assertEqual(self.mods[0].STATS["sample_calls"] >= 1, True)     # biasfree wrapped the instance's sample
        c = rci.counters(); self.assertEqual(c["levers"], "+".join(MEMBERS)); self.assertIn("confrows_floor_skips", c); self.assertIn("injrows_floor_skips", c)
        self.assertEqual(rci.fold_fields(512)[:3], [("rowchunk", ",".join(MEMBERS)), ("rowchunk_floor", 1024), ("rowchunk_rows", "whole:below_floor")])
        self.assertEqual(rci.fold_fields(4096)[2], ("rowchunk_rows", "blocked"))

    def test_a_subset_binds_only_the_subset(self):
        rec = rci.install(self.model, 2, ("injrows", "confrows", "pdeskip"))
        self.assertEqual({n for n, v in rec["levers"].items() if v}, {"injrows", "confrows", "pdeskip"})
        self.assertEqual(rec["why"]["zbf16"], "not in the set"); self.assertNotIn("sample", self.mods[0]._ORIG)

    def test_refusals_by_name(self):
        from opt_core.mem.rowpair import RowpairRefused
        with self.assertRaises(RowpairRefused) as cm:
            rci.install(self.model, 2, ("injrows", "zbf16"))
        self.assertIn("ride on confrows", str(cm.exception))
        with self.assertRaises(RowpairRefused) as cm:                                # confbf16's bf16 pair would reach the whole-shard pooling's fp32 Linear
            rci.install(self.model, 2, ("injrows", "confrows", "confbf16", "pdeskip", "zbf16"))
        self.assertIn("confbf16 rides on confmem", str(cm.exception))
        rec = rci.install(self.model, 2, ("confrows", "pdeskip", "zbf16")); self.assertTrue(rec["levers"]["zbf16"])   # confmem and confbf16 out together: binds
        for m in self.mods[1:]:
            m.unapply()
        rci.STATE["installed"] = False
        with self.assertRaises(RowpairRefused) as cm:
            rci.install(self.model, 2, ("injrows", "nosuch"))
        self.assertIn("unknown lever name", str(cm.exception))
        with self.assertRaises(RowpairRefused) as cm:
            rci.install(object(), 2, ("biasfree",))
        self.assertIn("biasfree cannot bind", str(cm.exception))
        os.environ["EF2_ROWPAIR_INJECT_MB"] = "-5"
        try:
            with self.assertRaises(RowpairRefused) as cm:
                rci.install(self.model, 2, ("injrows",))
            self.assertIn("EF2_ROWPAIR_INJECT_MB", str(cm.exception))
        finally:
            del os.environ["EF2_ROWPAIR_INJECT_MB"]

    def test_knobs(self):
        self.assertEqual(rci.read_knobs({}), {"min_tokens": 1024, "inject_mb": 512.0, "conf_mb": 512.0, "zbf16_mb": 1024.0})
        self.assertEqual(rci.read_knobs({"EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS": "2048", "EF2_ROWPAIR_ZBF16_MB": "64"})["min_tokens"], 2048)
        from opt_core.mem.rowpair import RowpairRefused
        for bad in ({"EF2_ROWPAIR_CONF_MB": "abc"}, {"EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS": "10.5"}, {"EF2_ROWPAIR_ZBF16_MB": "0"}):
            with self.assertRaises(RowpairRefused) as cm:
                rci.read_knobs(bad)
            self.assertIn(list(bad)[0], str(cm.exception))


class SettleRouteJudgesTheDeferredMembers(unittest.TestCase):
    def setUp(self):
        self.kit = _kit()
        if self.kit is None:
            self.skipTest("the kit tree (driver/ef2_server.py) is not importable here")
        self.saved = stack._REPORT
        self.res = modes.resolve("big", "fast", self.kit, n_gpu=2, ablate="pdeskip")
        rc = [n for n in self.res.levers if n in MEMBERS]; other = [n for n in self.res.levers_for_variant if n not in MEMBERS]
        self.rep = {"active": True, "mode": "big", "variant": "fast", "server_mode": self.res.server_mode, "levers_planned": list(self.res.levers),
                    "levers_applied": list(other), "levers_fallback": [], "fallback_reasons": {}, "levers_deferred": list(rc), "partial": [],
                    "gpu": {"name": None, "sm": None}, "resolution": self.res, "ablate": list(self.res.ablate), "n_gpu": 2}
        self.rec = {"levers_applied": list(other), "levers_fallback": [], "fallback_reasons": {}, "levers_deferred": list(rc), "model_index": 0}
        self.rc = rc

    def tearDown(self):
        stack._REPORT = self.saved

    def _line(self, lines, name):
        return [l for l in lines if f" name={name} " in l][0]

    def test_classify_time_lines_defer_and_apply_to_leaves_them_out(self):
        all_lines = report.lever_lines(self.rep, self.rec); cfg_lines = report.lever_lines(self.rep, self.rec, include_deferred=False)
        self.assertEqual(len(all_lines), len(registry.LEVERS)); self.assertEqual(len(all_lines) - len(cfg_lines), len(self.rc))
        self.assertIn(" state=skipped reason=deferred:rowpair.install_rank ", self._line(all_lines, "confrows"))
        self.assertIn(" state=off reason=ablated ", self._line(all_lines, "pdeskip"))
        self.assertIn(" impl=esmfold2_opt.rowchunk.zbf16rows ", self._line(all_lines, "zbf16")); self.assertIn(" impl=esmfold2_opt.rowpair ", self._line(all_lines, "injrows"))

    def test_a_member_that_did_not_bind_is_partial_exit_3(self):
        stack._REPORT = dict(self.rep)
        out = stack.settle_route({"levers": {n: n != "zbf16" for n in self.rc}, "why": {"zbf16": "the confrows lever is not installed in this process"}})
        self.assertEqual(out["partial"], ["zbf16"]); self.assertEqual(out["levers_deferred"], []); self.assertIn("confrows", out["levers_applied"])
        self.assertEqual(out["fallback_reasons"]["zbf16"], "the confrows lever is not installed in this process")
        v = report.verdict(report.EXIT_OK, stack.status())
        self.assertEqual((v["exit_code"], v["refused_partial"], v["partial"]), (report.EXIT_NOT_ACTIVE, True, ["zbf16"]))

    def test_every_member_bound_is_exit_0_and_inactive_process_is_untouched(self):
        stack._REPORT = dict(self.rep)
        out = stack.settle_route({"levers": {n: True for n in self.rc}, "why": {}})
        self.assertEqual(out["partial"], []); self.assertTrue(set(self.rc) <= set(out["levers_applied"]))
        self.assertEqual(report.verdict(report.EXIT_OK, stack.status())["exit_code"], report.EXIT_OK)
        stack._REPORT = None
        self.assertEqual(stack.settle_route({"levers": {}, "why": {}}), {})

    def test_no_record_at_all_refuses_every_deferred_member(self):
        stack._REPORT = dict(self.rep)
        out = stack.settle_route({})
        self.assertEqual(set(out["partial"]), set(self.rc))


class NGpu1ImportsNoRowchunkModule(unittest.TestCase):
    def test_fresh_interpreter(self):
        code = ("import sys, esmfold2_opt, esmfold2_opt.cli, esmfold2_opt.stack, esmfold2_opt.report\n"
                "from esmfold2_opt import modes, rowpair\n"
                "from esmfold2_opt.tests._stubs import kit_or_none\n"
                "kit = kit_or_none()\n"
                "if kit:\n"
                "    [modes.resolve(m, 'fast', kit) for m in ('fast', 'exact', 'big')]\n"
                "print('ROWCHUNK_MODULES=' + ','.join(sorted(m for m in sys.modules if m.startswith('esmfold2_opt.rowchunk'))))\n"
                "print('EXIT=' + esmfold2_opt.report.exit_tally_line(pid=1))\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, KMP_AFFINITY="disabled"), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ROWCHUNK_MODULES=\n", r.stdout)                                   # nothing of the package is imported at n_gpu 1
        self.assertIn("no lever counters", r.stdout); self.assertNotIn("rowchunk=", r.stdout)


if __name__ == "__main__":
    unittest.main()
