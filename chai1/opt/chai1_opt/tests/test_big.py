"""The big mode (big.py) on the eager-stack stubs, no torch: the mode row composes fast by reference; the lean line applies through
opt_core.mem with the graphed override set (the allocator policy is the package's `alloc` lever, implied by the mode row); a flag turns a
lever off by name, an unknown flag refuses;
the installers land on the adopted stack instances (the MSA rows sliced to the rows that carry any mask, the chunk rows set, the un-graphed
step marked), the census counts one unit per item and the exit gate fails closed on a partial unit unless the opt-out is set; the ACTIVE
line carries the big field after gpu= and hoist=; the library absent is a refusal by name (core_missing)."""
import io
import os
import sys
import unittest
from contextlib import redirect_stderr

import numpy as np

from chai1_opt import big, hoist, modes, report
from chai1_opt.tests import _stubs

LEAN = ("msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph")


class FakeBool:
    """A torch-like bool tensor over numpy: any(dim) / nonzero() / numel() / max().item() / slicing — what big's slice reads."""
    def __init__(self, a):
        self.a = np.asarray(a)
        self.ndim, self.shape = self.a.ndim, self.a.shape

    def any(self, dim):
        return FakeBool(self.a.any(axis=dim))

    def nonzero(self):
        return FakeBool(np.argwhere(self.a))

    def numel(self):
        return int(self.a.size)

    def max(self):
        return FakeBool(self.a.max())

    def item(self):
        return self.a.item()

    def __getitem__(self, s):
        return FakeBool(self.a[s])


class FakeFeats(FakeBool):
    pass


class FakeTensor:
    """shape / dtype / data_ptr like a torch tensor (the hoist's item key reads them); ``tag`` is what the stub denoiser caches."""
    def __init__(self, shape, ptr, dtype="torch.float32", tag="t"):
        self.shape, self.ptr, self.dtype, self.tag = tuple(shape), ptr, dtype, tag

    def data_ptr(self):
        return self.ptr


class Anchor(FakeTensor):
    """One item's static tensor: a new object per item (the object is the unit anchor); ``n`` its pair extent (the crop)."""
    _n = 0

    def __init__(self, n=16):
        Anchor._n += 1
        super().__init__((1, n, n, 128), 1000 + Anchor._n, tag=f"item{Anchor._n}")


def denoiser_kw(anchor, n=16):
    return {"token_pair_trunk_repr": anchor, "atom_single_input_feats": FakeTensor((1, 8 * n, 64), anchor.ptr + 1),
            "atom_noised_coords": FakeTensor((1, 5, 8 * n, 3), anchor.ptr + 2), "noise_sigma": FakeTensor((1, 5), anchor.ptr + 3), "num_steps": 200}


def msa(S=8, N=4, rows_true=(0,)):
    mask = np.zeros((1, S, N), dtype=bool)
    for r in rows_true:
        mask[0, r, :] = True
    return {"msa_input_feats": FakeFeats(np.zeros((1, S, N, 3))), "msa_mask": FakeBool(mask)}


class PWA:
    """A stand-in MSAPairWeightedAveraging: the carried forward's signature; the chunked class wraps it."""
    CH = 8192

    def forward(self, msa, z, pair_mask, msa_mask):
        return msa


class OPM:
    """a stand-in OuterProductMean (the real one is re-classed onto the core chunker; its forward is never called in these tests)."""
    CH = 4096


class FakeTrunkModule:
    def __init__(self):
        import types
        self.msa_module = types.SimpleNamespace(msa_pair_weighted_averaging=[PWA(), PWA()], outer_product_mean=[OPM(), OPM()])


class TestModeRow(unittest.TestCase):
    def test_big_composes_fast_by_reference(self):
        km, fast = modes.KIT_MODES["big"], modes.KIT_MODES["fast"]
        self.assertEqual((km.levels, km.eager, km.dstep, km.stack), (fast.levels, fast.eager, fast.dstep, fast.stack))
        self.assertEqual(modes.MODES, ("off", "exact", "fast", "big"))
        self.assertEqual(modes.KIT_MODE_NAMES, ("exact", "fast", "big"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertEqual(big.LINE, LEAN)
        self.assertEqual(km.memory, LEAN)                                               # the mode row and the module name the same line
        self.assertEqual(km.lever_names, fast.lever_names + LEAN)
        self.assertEqual(big.BASE, "fast")
        self.assertEqual(km.implied_optin, ("tf32", "alloc"))                         # fast's implied tf32 + the allocator policy: one producer, the package's alloc lever
        self.assertEqual(km.pairtrack, fast.pairtrack)                                  # fast's trunk levers by reference
        self.assertEqual(modes.with_implied("big", ()), ("tf32", "alloc"))
        self.assertEqual(modes.with_implied("big", ("tf32", "alloc")), ("tf32", "alloc"))
        self.assertEqual(modes.with_implied("fast", ("tf32",)), ("tf32", "alloc")); self.assertEqual(modes.with_implied("exact", ()), ("alloc",))
        self.assertIsNone(modes.optin_refusal("big", ("alloc",)))


class TestApplyLine(unittest.TestCase):
    def setUp(self):
        big.reset_for_tests()

    def tearDown(self):
        big.reset_for_tests()

    def test_lean_line_applies_and_records(self):
        env, rep = {}, {}
        rec = big.apply_line(rep, out_dir="/tmp/big-test", environ=env)
        self.assertEqual(tuple(rec.levers), LEAN)
        self.assertEqual(rec.refused, [])
        self.assertIs(big.graphed_override(), False)
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)                               # the allocator is the alloc lever's, never the line's
        self.assertEqual(rep["big"]["levers"], list(LEAN))
        self.assertEqual(big.mode_field(), "lean:msa_rows,msa_chunk,trunk_chunk,opm_chunk,nograph")
        self.assertEqual(rec.exact, "measured")
        self.assertEqual(rec.exact_per_lever["nograph"], "bitwise")
        self.assertEqual(rec.exact_per_lever["msa_rows"], "measured")
        self.assertEqual(big.apply_line(rep, environ=env), rec)                       # idempotent within a process

    def test_the_levers_are_fixed_and_switch_words_are_refused_by_name(self):
        """A mode's memory levers are fixed: nothing in the environment removes or adds one or sets a lever setting, except the two documented
        size gates (big.GATE_WORDS). Every other CHAI1_BIG_* name is refused BY NAME — by apply_line itself and by the package's
        undeclared-variable gate (stack.gates / the .pth hook) — never ignored in silence."""
        from chai1_opt import _autoload, stack
        self.assertEqual((tuple(big.GATE_WORDS), stack.DECLARED_BIG_ENV, _autoload.DECLARED_BIG_ENV),
                         (("CHAI1_BIG_MSA_CHUNK_MIN_N", "CHAI1_BIG_NOGRAPH_MIN_N", "CHAI1_BIG_TRUNK_CHUNK_MIN_N", "CHAI1_BIG_OPM_CHUNK_MIN_N", "CHAI1_BIG_HOIST2_MAX_N"),) * 3)
        for env in ({"CHAI1_BIG_MSA_CHUNK": "0"}, {"CHAI1_BIG_NOGRAPH": "0"}, {"CHAI1_BIG_MSA_CHUNK_ROWS": "256"}, {"CHAI1_BIG_TURBO": "1"},
                    {"CHAI1_BIG_ALLOW_PARTIAL": "1"}, {"CHAI1_BIG_LINE": "lean"}, {"CHAI1_BIG_PAIR_CHUNK_MIN_N": "0"}):
            big.reset_for_tests()
            with self.assertRaises(ValueError) as cm:
                big.apply_line({}, environ=env)
            name = next(iter(env))
            self.assertIn(f"{name}: not a variable this package reads", str(cm.exception)); self.assertIn("the only CHAI1_BIG_* words are CHAI1_BIG_MSA_CHUNK_MIN_N, CHAI1_BIG_NOGRAPH_MIN_N", str(cm.exception))
            self.assertEqual(_autoload.undeclared(env), [name])                              # the gate names it too (NOT ACTIVE: undeclared variable(s) …)
        self.assertEqual(_autoload.undeclared({"CHAI1_BIG_MSA_CHUNK_MIN_N": "9", "CHAI1_BIG_NOGRAPH_MIN_N": "9"}), [])
        big.reset_for_tests()
        rec = big.apply_line({}, environ={})
        self.assertEqual((tuple(rec.levers), list(rec.off_by_flag), list(rec.on_by_flag)), (LEAN, [], []))
        self.assertIs(big.graphed_override(), False)                                        # nograph is always in the big line: the step runs un-graphed

    def test_the_two_size_gate_words_reach_the_lever(self):
        """CHAI1_BIG_MSA_CHUNK_MIN_N / CHAI1_BIG_NOGRAPH_MIN_N are read by the package (big.gate_words) and handed to the core as the
        lever's ``min_n`` setting; the core casts the value and refuses a non-integer by name."""
        rec = big.apply_line({}, environ={"CHAI1_BIG_MSA_CHUNK_MIN_N": "512", "CHAI1_BIG_NOGRAPH_MIN_N": " 64 "})
        ap = {a.lever: a for a in rec.applied}
        self.assertEqual((ap["msa_chunk"].settings["min_n"], ap["msa_chunk"].settings["rows"], ap["nograph"].settings["min_n"]), (512, big.MSA_CHUNK_ROWS, 64))
        big.reset_for_tests()
        M = big.mem()
        with self.assertRaises(M.Refused) as cm:
            big.apply_line({}, environ={"CHAI1_BIG_MSA_CHUNK_MIN_N": "abc"})
        self.assertIn("msa_chunk", str(cm.exception)); self.assertIn("min_n", str(cm.exception))


class TestHooksAndCensus(unittest.TestCase):
    def setUp(self):
        big.reset_for_tests()
        self.saved = _stubs.install()[-1]
        self.S = _stubs.make_eager_module()
        self.env = {}
        self.rec = big.apply_line({}, environ=self.env)
        self.parts = self.S.build_parts(None, "tier1", graphed=big.graphed_override())
        self.parts["trunk"].trunk = FakeTrunkModule()
        hoist.adopt(self.parts, self.S)
        self.done = big.on_adopted(self.parts, self.S)
        self.tw, self.dw = self.parts["trunk"], self.parts["diffusion"]

    def tearDown(self):
        big.reset_for_tests()
        _stubs.remove(self.saved)

    def fold(self, anchor, steps=2, S=8, rows_true=(0,)):
        for _ in range(3):
            self.tw.forward(256, move_to_device="cuda:0", token_pair_trunk_initial_repr=anchor, **msa(S=S, rows_true=rows_true))
        for _ in range(steps):
            self.dw.forward(256, **denoiser_kw(anchor))

    def test_installers_land_on_the_adopted_instances(self):
        self.assertEqual(type(self.tw).__name__, "BigTrunk")
        self.assertEqual(type(self.dw).__name__, "BigDenoiser")
        self.assertIs(self.dw.graphed, False)
        self.assertEqual(sorted(self.done), ["hoist2", "msa_chunk", "msa_rows", "nograph", "opm_chunk", "trunk_chunk"])   # hoist2: the default (a100, 40 GB-safe) table's per-item stand-down
        self.assertTrue(all(type(m).__name__ == "BigPWA" for m in big._pwa_modules(self.tw)))
        self.assertTrue(all(isinstance(m, PWA) for m in big._pwa_modules(self.tw)))     # re-classed in place: the carried instance, its state kept

    def test_msa_rows_sliced_to_the_last_masked_row(self):
        kw = big._slice_msa_rows(self.tw, msa(S=8, rows_true=(0, 2)), self.rec)
        self.assertEqual(kw["msa_input_feats"].shape, (1, 3, 4, 3))
        self.assertEqual(kw["msa_mask"].shape, (1, 3, 4))

    def test_census_one_unit_per_item_and_the_unmarked_lever_fails_closed(self):
        a, b = Anchor(), Anchor()
        self.fold(a); self.fold(b)
        v = big.exit_gate(0)
        c = self.rec.census()
        self.assertEqual(sorted(c["units"]), ["item0", "item1"])
        self.assertEqual(sorted(c["units"]["item0"]["ran"]), ["msa_rows", "nograph"])
        self.assertEqual(c["units"]["item0"]["absent"], ["msa_chunk"])                   # the stub trunk never calls the PWA: no chunker entry, no mark
        self.assertEqual(c["partial"], ["msa_chunk"])
        self.assertEqual(v["exit_code"], 3)                                              # fail-closed: an unmarked lever is a partial unit
        self.assertEqual(big.manifest_block()["census"]["n_units"], 2)

    def test_nothing_to_drop_is_a_named_skip(self):
        self.fold(Anchor(), S=1)
        big.exit_gate(0)
        u = self.rec.census()["units"]["item0"]
        self.assertIn("msa_rows", u["skipped"])

    def test_pwa_chunked_class_not_in_force_is_a_fallback(self):
        for m in big._pwa_modules(self.tw):
            m.__class__ = PWA
        self.fold(Anchor())
        big.exit_gate(0)
        u = self.rec.census()["units"]["item0"]
        self.assertIn("msa_chunk", u["fallback"])

    def test_graphed_step_is_a_fallback_and_fails_closed(self):
        self.dw.graphed = True
        self.fold(Anchor())
        v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 3)
        self.assertIn("nograph", self.rec.census()["partial"])

    def test_allow_partial_is_a_recorded_opt_out(self):
        """The route's opt-out of a partial item is the value the launcher hands in (the driver's --allow-partial; on the CHAI1_OPT route the
        word is CHAI1_OPT_ALLOW_PARTIAL=1) and the core's PARTIAL line names that route's word (Ctx.opt_out); CHAI1_BIG_ALLOW_PARTIAL is
        no word of this package (refused by name, test_the_levers_are_fixed…)."""
        self.fold(Anchor())
        err = io.StringIO()
        with redirect_stderr(err):
            v = big.exit_gate(0)                                                        # no flag: refused (exit 3), the line names --allow-partial
        self.assertEqual(v["exit_code"], 3); self.assertFalse(v.get("allow_partial")); self.assertEqual(v.get("opt_out"), "--allow-partial")
        self.assertIn("(--allow-partial records and proceeds)", err.getvalue())
        with redirect_stderr(err):
            v = big.exit_gate(0, allow_partial=True)                                    # the flag: recorded, exit 0
        self.assertEqual(v["exit_code"], 0); self.assertTrue(v.get("allow_partial")); self.assertEqual(v.get("allow_partial_source"), "kit")
        self.assertIn("(--allow-partial, recorded)", err.getvalue())
        # the CHAI1_OPT (in-process) route: stack.activate hands its own word and its own opt-out value (CHAI1_OPT_ALLOW_PARTIAL=1)
        big.reset_for_tests()
        self.rec = big.apply_line({}, environ={}, allow_partial=True, opt_out="CHAI1_OPT_ALLOW_PARTIAL=1")
        self.assertEqual((self.rec.opt_out, self.rec.allow_partial), ("CHAI1_OPT_ALLOW_PARTIAL=1", True))
        self.parts = self.S.build_parts(None, "tier1", graphed=big.graphed_override())
        self.parts["trunk"].trunk = FakeTrunkModule()
        hoist.adopt(self.parts, self.S); big.on_adopted(self.parts, self.S)
        self.tw, self.dw = self.parts["trunk"], self.parts["diffusion"]
        self.dw.graphed = True
        self.fold(Anchor())
        err2 = io.StringIO()
        with redirect_stderr(err2):
            v = big.exit_gate(0, allow_partial=True)
        self.assertEqual(v["exit_code"], 0); self.assertEqual(v.get("opt_out"), "CHAI1_OPT_ALLOW_PARTIAL=1")
        self.assertIn("(CHAI1_OPT_ALLOW_PARTIAL=1, recorded)", err2.getvalue()); self.assertNotIn("--allow-partial", err2.getvalue())

    def test_trunk_chunk_plug_survives_the_stacks_cfg_fixer(self):
        """the stack's cfg_fn runs before every trunk call and resets CFG; the lever's wrapper re-plugs after it."""
        self.fold(Anchor())
        cfg = self.S.trunk.CFG
        self.assertEqual(getattr(cfg["trimul_impl"], "__name__", None), "trimul_chunked")
        self.assertGreaterEqual(len(self.S.cfg_calls), 1)
        c = self.rec.census()
        self.assertIn("trunk_chunk", c["units"]["item0"]["skipped"])              # the fake trunk's 16-row pair is below min_n: a named skip

    def test_no_fold_means_every_lever_absent(self):
        v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 3)


class TestGatedLine(unittest.TestCase):
    """K21: exact and fast carry msa_chunk + nograph GATED at pair extent (crop) 2048 (modes.MEMORY_GATE_N) through the same opt_core.mem
    machinery: below the gate every item runs the mode's own path (the carried pair-weighted averaging, the stack's own graphed rule) as a
    named skip; at the gate the PWA runs in row blocks and the hoisted step un-graphed; each item's decision is printed (MEMORY line) and
    counted (gate census, the exit's MEMORY-GATE line, the manifest block). The levers are not in lever_names / levers_applied= (they are
    conditional per item): the ACTIVE line names them as big=gate2048:msa_chunk,nograph."""
    GATED = ("msa_chunk", "nograph")

    def setUp(self):
        big.reset_for_tests()
        self.saved = _stubs.install()[-1]
        self.S = _stubs.make_eager_module()
        self.err = io.StringIO()

    def tearDown(self):
        big.reset_for_tests()
        _stubs.remove(self.saved)

    def adopt(self, mode="fast", environ=None, graphed=True):
        self.rep = {}
        self.rec = big.apply_line(self.rep, environ={} if environ is None else environ, mode=mode)
        self.assertIsNone(big.graphed_override())                                       # the stack's own graphed rule stays in force (nograph acts per item)
        self.parts = self.S.build_parts(None, "tier1", graphed=graphed)
        hoist.adopt(self.parts, self.S)
        with redirect_stderr(self.err):
            self.done = big.on_adopted(self.parts, self.S)
        self.tw, self.dw = self.parts["trunk"], self.parts["diffusion"]

    def fold(self, anchor, steps=2):
        with redirect_stderr(self.err):
            for _ in range(3):
                self.tw.forward(256, move_to_device="cuda:0", token_pair_trunk_initial_repr=anchor, **msa(S=8, rows_true=(0,)))
            for _ in range(steps):
                self.dw.forward(256, **denoiser_kw(anchor))

    def test_the_rows_carry_the_gated_levers_outside_lever_names(self):
        for m in ("exact", "fast"):
            km = modes.KIT_MODES[m]
            self.assertEqual((km.memory_gated, km.memory_gate, km.memory), (self.GATED, 2048, ()))
            self.assertFalse(set(self.GATED) & set(km.lever_names))                           # conditional per item: never on levers_applied=
            self.assertEqual(big.dry_run_fields(m), {"line": "gate2048", "levers": (["msa_rows"] if m == "fast" else []) + list(self.GATED), "gates": {"msa_chunk": 2048, "nograph": 2048}})
        self.assertEqual((modes.KIT_MODES["big"].memory_gated, modes.KIT_MODES["big"].memory_gate), ((), 0))   # big: its line at every size, unchanged
        self.assertEqual(big.dry_run_fields("big"), {"line": "lean", "levers": list(LEAN), "gates": {"config": "a100", "trunk_chunk": 1536, "opm_chunk": 1536, "hoist2_max_n": 1536}})   # no MODEL_OPT_TARGET_GPU: the 40 GB-safe table
        self.assertEqual((modes.MEMORY_GATE_N, modes.GATED_MEMORY, big.MSA_CHUNK_MIN_N, big.NOGRAPH_MIN_N), (2048, self.GATED, 0, 0))
        with self.assertRaises(ValueError):
            big.apply_line({}, environ={}, mode="off")

    def test_apply_records_the_gate_and_the_flags_reach_it(self):
        rep = {}
        rec = big.apply_line(rep, environ={}, mode="fast")
        self.assertEqual((tuple(rec.levers), rec.base, rec.refused), (("msa_rows",) + self.GATED, "fast", []))   # fast: msa_rows at every size + the gated pair
        self.assertEqual(rep["big"]["line"], "gate2048"); self.assertEqual(rep["big"]["gates"], {"msa_rows": 0, "msa_chunk": 2048, "nograph": 2048})
        ap = {a.lever: a for a in rec.applied}
        self.assertEqual((ap["msa_chunk"].settings["min_n"], ap["msa_chunk"].settings["rows"], ap["nograph"].settings["min_n"]), (2048, 1024, 2048))
        self.assertEqual(big.mode_field(), "gate2048:msa_rows,msa_chunk,nograph")
        self.assertEqual(report.lever_state("msa_chunk", dict(rep, mode="fast", active=True, levers_applied=["W1"])), ("on", None))   # the LEVER line: in force
        big.reset_for_tests()
        rec = big.apply_line(rep, environ={"CHAI1_BIG_MSA_CHUNK_MIN_N": "1024", "CHAI1_BIG_NOGRAPH_MIN_N": "4096"}, mode="exact")   # the two size-gate words move each lever's gate; nothing removes a lever
        self.assertEqual((tuple(rec.levers), list(rec.off_by_flag), rep["big"]["gates"]), (("msa_chunk", "nograph"), [], {"msa_chunk": 1024, "nograph": 4096}))
        big.reset_for_tests()
        with self.assertRaises(ValueError):
            big.apply_line({}, environ={"CHAI1_BIG_NOGRAPH": "0"}, mode="exact")           # a lever-off word: refused by name on the gated line too

    def test_below_the_gate_every_item_runs_the_mode_unchanged(self):
        self.adopt("fast")
        self.assertEqual(sorted(self.done), ["msa_chunk", "msa_rows", "nograph"])
        pwa = big._pwa_modules(self.tw)[0]
        self.assertEqual(type(pwa).__name__, "BigPWA")
        feats = msa(S=8, N=4)
        self.assertIs(pwa.forward(feats["msa_input_feats"], None, None, feats["msa_mask"]), feats["msa_input_feats"])   # N=4 < 2048: the carried forward itself (the core chunker never entered)
        self.assertEqual(big._STATE["chunk_entries"], 0)
        self.fold(Anchor(n=1536))
        self.assertIs(self.dw.graphed, True)                                                  # the stack's own rule (det 0: graphed) untouched below the gate
        with redirect_stderr(self.err):
            v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 0)                                                   # both levers a named skip on the unit: complete, not partial
        u = self.rec.census()["units"]["item0"]
        self.assertEqual(sorted(u["skipped"]), ["msa_chunk", "nograph"]); self.assertEqual((list(u["ran"]), list(u["fallback"]), list(u["absent"])), (["msa_rows"], [], []))
        self.assertEqual(v["gate_census"], {"units": 1, "applied": {"msa_rows": 1, "msa_chunk": 0, "nograph": 0}, "stood_down": {"msa_rows": 0, "msa_chunk": 1, "nograph": 1}, "min_n": {"msa_chunk": 2048, "msa_rows": 0, "nograph": 2048}})
        e = self.err.getvalue()
        self.assertIn("\n[chai1-opt] MEMORY unit=item0 pair_extent=1536 applied=msa_rows standing_down=msa_chunk,nograph alloc=none (min_n msa_rows>=0 msa_chunk>=2048 nograph>=2048)\n", e)   # a line of its own (leading newline: tqdm's bar holds the current one); alloc=none: nothing exported in this test process
        self.assertEqual(e.count(" MEMORY unit="), 1)                                         # once per item, not per recycle
        self.assertIn("[chai1-opt] MEMORY-GATE units=1 applied msa_rows=1/1 msa_chunk=0/1 nograph=0/1 (min_n msa_rows>=0 msa_chunk>=2048 nograph>=2048)\n", e)
        self.assertEqual(big.manifest_block()["gate_census"]["stood_down"], {"msa_rows": 0, "msa_chunk": 1, "nograph": 1})

    def test_at_the_gate_the_levers_apply_per_item_and_the_rule_comes_back_below(self):
        self.adopt("fast")
        below, top, again = Anchor(n=1024), Anchor(n=2048), Anchor(n=512)
        self.fold(below); self.assertIs(self.dw.graphed, True)
        self.fold(top); self.assertIs(self.dw.graphed, False)                                 # the 2048 item: the hoisted step un-graphed for THIS item
        self.fold(again); self.assertIs(self.dw.graphed, True)                                # back below: the stack's own rule again
        with redirect_stderr(self.err):
            v = big.exit_gate(0)
        c = self.rec.census()["units"]
        self.assertEqual(sorted(c), ["item0", "item1", "item2"])
        self.assertEqual((sorted(c["item0"]["skipped"]), list(c["item1"]["ran"]), list(c["item1"]["absent"])), (["msa_chunk", "nograph"], ["msa_rows", "nograph"], ["msa_chunk"]))   # the stub trunk never calls the PWA: no chunker entry above the gate → absent, fail-closed
        self.assertEqual(v["exit_code"], 3)
        self.assertEqual(v["gate_census"]["applied"], {"msa_rows": 3, "msa_chunk": 1, "nograph": 1}); self.assertEqual(v["gate_census"]["stood_down"], {"msa_rows": 0, "msa_chunk": 2, "nograph": 2})
        e = self.err.getvalue()
        self.assertIn("MEMORY unit=item1 pair_extent=2048 applied=msa_rows,msa_chunk,nograph standing_down=none alloc=none (min_n msa_rows>=0 msa_chunk>=2048 nograph>=2048)", e)
        self.assertIn("MEMORY unit=item2 pair_extent=512 applied=msa_rows standing_down=msa_chunk,nograph alloc=none", e)
        self.assertIn("MEMORY-GATE units=3 applied msa_rows=3/3 msa_chunk=1/3 nograph=1/3", e)

    def test_under_the_recipe_the_step_is_already_ungraphed(self):
        self.adopt("exact", graphed=False)                                                    # det 1: the stack builds the hoisted step un-graphed
        self.fold(Anchor(n=2048)); self.assertIs(self.dw.graphed, False)
        self.fold(Anchor(n=256)); self.assertIs(self.dw.graphed, False)                       # nothing to put back: the rule's own value is False
        with redirect_stderr(self.err):
            big.exit_gate(0)
        c = self.rec.census()["units"]
        self.assertEqual((list(c["item0"]["ran"]), sorted(c["item1"]["skipped"])), (["nograph"], ["msa_chunk", "nograph"]))


class TestActiveLine(unittest.TestCase):
    def test_big_field_after_gpu(self):
        rep = {"active": True, "mode": "big", "levels": ["W1", "W2", "W5"], "eager": "tier1", "dstep": ("hoist2", "compiled", "dit_attn"),
               "levers_applied": ["W1"], "det": 0, "route": "driver", "chai_lab_version": "0.6.1", "torch_version": "2.5.1",
               "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": (9, 0)}, "hoist": {"keyed": "item", "release": "trunk_boundary"},
               "big": {"line": "lean", "levers": list(LEAN), "exact": "measured"}}
        line = report.activation_line(rep)
        self.assertIn(" hoist=item/trunk_boundary big=lean:msa_rows,msa_chunk,trunk_chunk,opm_chunk,nograph big_exact=measured", line)
        self.assertTrue(line.startswith("[chai1-opt] ACTIVE mode=big "))
        self.assertLess(line.index(" gpu="), line.index(" big="))                       # after the gpu= tail: the held fields stay byte-identical


class TestLibraryAbsent(unittest.TestCase):
    def test_refusal_by_name(self):
        import opt_core
        saved = {k: sys.modules.get(k) for k in ("opt_core.mem",)}
        attr = opt_core.__dict__.pop("mem", None)
        sys.modules["opt_core.mem"] = None                                               # an import of the library now raises ImportError
        try:
            why = big.refusal()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            if attr is not None:
                opt_core.mem = attr
        self.assertTrue(why and why.startswith(big.NOT_PRESENT), why)
        self.assertEqual(big.NOT_PRESENT, "reason=producer_missing:opt_core.mem")


if __name__ == "__main__":
    unittest.main()


class TestMemclass(unittest.TestCase):
    """big's composed line takes its size gates from the device's memory class at adoption; the config word is the prior; explicit words win."""
    def setUp(self):
        big.reset_for_tests()

    def _prime(self, config="a100", explicit=()):
        big._STATE.update(is_line=True, hoist2_in_row=True, config=config, explicit_gates=set(explicit), hoist2_max_n=big.GATE_TABLE[config]["hoist2_max_n"])
        big._STATE["min_n"].update(trunk_chunk=big.GATE_TABLE[config]["trunk_chunk"], opm_chunk=big.GATE_TABLE[config]["opm_chunk"])

    def test_80g_device_under_a100_config_takes_the_h100_row(self):
        self._prime("a100"); orig = big._device_total_gib
        try:
            big._device_total_gib = lambda: 79.2
            big._resolve_memclass()
        finally:
            big._device_total_gib = orig
        self.assertEqual(big._STATE["memclass"], "80g"); self.assertEqual(big._STATE["memclass_row"], "h100")
        self.assertEqual(big._STATE["min_n"]["trunk_chunk"], big.GATE_TABLE["h100"]["trunk_chunk"]); self.assertEqual(big._STATE["hoist2_max_n"], 0)
        self.assertEqual(big._STATE["min_n_override"], {"trunk_chunk": 2048, "opm_chunk": 2048})

    def test_40g_device_keeps_the_40gb_row_and_explicit_words_win(self):
        self._prime("h100", explicit=("trunk_chunk", "hoist2")); big._STATE["min_n"]["trunk_chunk"] = 1024; big._STATE["hoist2_max_n"] = 777
        orig = big._device_total_gib
        try:
            big._device_total_gib = lambda: 39.4
            big._resolve_memclass()
        finally:
            big._device_total_gib = orig
        self.assertEqual(big._STATE["memclass"], "40g"); self.assertEqual(big._STATE["min_n"]["trunk_chunk"], 1024)   # explicit word kept
        self.assertEqual(big._STATE["min_n"]["opm_chunk"], 1536); self.assertEqual(big._STATE["hoist2_max_n"], 777)  # opm from the 40g row; hoist2 explicit

    def test_no_device_keeps_the_prior_and_gated_rows_are_untouched(self):
        self._prime("a100"); orig = big._device_total_gib
        try:
            big._device_total_gib = lambda: None
            big._resolve_memclass()
        finally:
            big._device_total_gib = orig
        self.assertEqual(big._STATE["memclass"], "prior"); self.assertEqual(big._STATE["min_n"]["trunk_chunk"], 1536)
        big.reset_for_tests(); big._STATE.update(is_line=False)
        big._resolve_memclass(); self.assertIsNone(big._STATE["memclass"])


class TestGatedNamedSkipAtUnitClose(unittest.TestCase):
    """A size-gated lever whose site is never entered on an item below its gate is a NAMED skip at unit close (exit 0 with the word), never
    'never marked' (exit 3) — the external evaluation's cases: fast at 1600 tokens (`levers=msa_chunk: item0: never marked`) and big's
    trunk_chunk / opm_chunk below the 80 GB card's 2048 gate; a lever whose gate engages and that still leaves no event stays exit 3."""

    def setUp(self):
        big.reset_for_tests()

    def tearDown(self):
        big.reset_for_tests()

    def _item(self, rec, n, name="item0"):
        rec.unit_begin(name)
        big._STATE.update(unit=name, n_units=1, unit_n=n)

    def test_fast_gated_levers_below_the_gate_with_no_site_event_exit_0_named(self):
        rec = big.apply_line({}, environ={}, mode="fast")                                   # fast: msa_rows (size-free) + msa_chunk, nograph gated at 2048
        self.assertGreater(big._STATE["min_n"]["msa_chunk"], 1600)
        self._item(rec, 1600)
        rec.mark("msa_rows", detail="S=4096->4000")                                           # the size-free lever's site ran; msa_chunk / nograph sites never entered
        big.unit_end()
        u = rec.units["item0"]
        self.assertIn("msa_chunk", u.skipped); self.assertIn("gated: pair extent 1600 < min_n", u.skipped["msa_chunk"])
        self.assertIn("nograph", u.skipped)
        v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 0, v.get("reasons"))
        self.assertEqual(v.get("partial") or [], [])

    def test_big_chunk_levers_below_the_80g_gate_exit_0_named(self):
        rec = big.apply_line({}, environ={}, mode="big")
        big._STATE["min_n"].update(trunk_chunk=2048, opm_chunk=2048)                        # the 80 GB memclass row (resolved lazily at adoption on a GPU)
        self._item(rec, 1600)
        for lv in rec.levers:                                                                   # every other lever's site ran on the item
            if lv not in ("trunk_chunk", "opm_chunk"):
                rec.mark(lv, detail="ran")
        big.unit_end()
        u = rec.units["item0"]
        for lv in ("trunk_chunk", "opm_chunk"):
            self.assertIn(lv, u.skipped); self.assertIn("gated: pair extent 1600 < min_n 2048", u.skipped[lv])
        v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 0, v.get("reasons"))

    def test_a_lever_whose_gate_engages_and_never_marks_stays_exit_3(self):
        rec = big.apply_line({}, environ={}, mode="big")
        big._STATE["min_n"].update(trunk_chunk=2048, opm_chunk=2048)
        self._item(rec, 2048)                                                                   # at the gate: the chunk sites must run
        for lv in rec.levers:
            if lv != "opm_chunk":
                rec.mark(lv, detail="ran")
        big.unit_end()
        self.assertNotIn("opm_chunk", rec.units["item0"].skipped)
        v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 3)
        self.assertTrue(any("opm_chunk" in str(r) and "never marked" in str(r) for r in (v.get("reasons") or [])) or "opm_chunk" in (v.get("partial") or []), v)


class TestNarrowedTrunkSiteSkips(unittest.TestCase):
    """The item gate reads the CROP (2048) and applies the chunk levers; trunk_n then narrows the trunk's inputs to N' = ceil64(live) = 1600,
    so the chunk SITES meet an extent below min_n and take the carried branch — they name it (skip), the verdict is exit 0; an engaged gate
    whose site is entered at the gate extent and leaves no mark is still exit 3 (MSAMOD's root cause of the external eval's fast@1600 exit 3)."""

    def setUp(self):
        big.reset_for_tests()

    def tearDown(self):
        big.reset_for_tests()

    class _Shape:
        def __init__(self, *shape): self.shape = tuple(shape)

    def _open(self, rec, crop):
        rec.unit_begin("item0"); big._STATE.update(unit="item0", n_units=1, unit_n=crop)

    def test_fast_msa_chunk_site_below_min_n_after_narrowing_is_a_named_skip_exit_0(self):
        rec = big.apply_line({}, environ={}, mode="fast")
        min_n = big._STATE["min_n"]["msa_chunk"]; self.assertEqual(min_n, 2048)
        self._open(rec, 2048)                                                                   # the gate applied msa_chunk at the crop
        rec.mark("msa_rows", detail="S=4096->4000"); rec.mark("nograph", detail="graphed=False")
        Base = type("PWA", (), {"forward": lambda self, msa, z, pm, mm: "carried"})
        PWA = big._pwa_class(Base, rows=512, min_n=min_n)
        out = PWA().forward(self._Shape(1, 4000, 1600, 64), None, None, None)                  # the site sees N' = 1600
        self.assertEqual(out, "carried")
        u = rec.units["item0"]; self.assertIn("msa_chunk", u.skipped); self.assertIn("extent 1600 < min_n 2048 at the site", u.skipped["msa_chunk"])
        big.unit_end(); v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 0, v.get("reasons")); self.assertEqual(v.get("partial") or [], [])

    def test_big_opm_and_pwa_sites_below_min_n_after_narrowing_exit_0(self):
        rec = big.apply_line({}, environ={}, mode="big")
        big._STATE["min_n"].update(trunk_chunk=2048, opm_chunk=2048, msa_chunk=2048)
        self._open(rec, 2048)
        for lv in rec.levers:
            if lv not in ("opm_chunk", "msa_chunk"):
                rec.mark(lv, detail="ran")
        BaseO = type("OPM", (), {"forward": lambda self, msa, mm: "carried"})
        try:
            import torch  # noqa: F401 — BigOPM.forward imports torch / chai1_eager.trunk before its size test
            have_torch = True
        except Exception:
            have_torch = False
        if have_torch:
            try:
                OPM = big._opm_class(BaseO, rows=256, min_n=2048)
                self.assertEqual(OPM().forward(self._Shape(1, 4000, 1600, 64), None), "carried")
                self.assertIn("opm_chunk", rec.units["item0"].skipped)
            except ImportError:
                rec.skip("opm_chunk", "extent 1600 < min_n 2048 at the site (test: eager trunk not importable)")
        else:
            rec.skip("opm_chunk", "extent 1600 < min_n 2048 at the site (test: no torch)")
        BaseP = type("PWA", (), {"forward": lambda self, msa, z, pm, mm: "carried"})
        PWA = big._pwa_class(BaseP, rows=512, min_n=2048)
        PWA().forward(self._Shape(1, 4000, 1600, 64), None, None, None)
        self.assertIn("msa_chunk", rec.units["item0"].skipped)
        big.unit_end(); v = big.exit_gate(0)
        self.assertEqual(v["exit_code"], 0, v.get("reasons"))

    def test_site_helper_names_once_and_never_over_a_mark(self):
        rec = big.apply_line({}, environ={}, mode="big")
        self._open(rec, 2048)
        big._site_skip(rec, "trunk_chunk", 1600, 2048, "triangle_multiplication")
        big._site_skip(rec, "trunk_chunk", 1600, 2048, "triangle_attention")               # once per (unit, lever)
        self.assertEqual(sum(1 for e in rec.units["item0"].events if e.get("lever") == "trunk_chunk" and e.get("kind") == "skip"), 1)
        rec.mark("opm_chunk", detail="ran"); big._STATE["marked"].add(("item0", "opm_chunk"))
        big._site_skip(rec, "opm_chunk", 1600, 2048, "outer_product_mean")                  # never over a mark of the same unit
        self.assertNotIn("opm_chunk", rec.units["item0"].skipped)

    def test_engaged_gate_site_entered_at_the_gate_extent_without_a_mark_stays_exit_3(self):
        rec = big.apply_line({}, environ={}, mode="big")
        big._STATE["min_n"].update(trunk_chunk=2048, opm_chunk=2048)
        self._open(rec, 2048)
        for lv in rec.levers:
            if lv != "trunk_chunk":
                rec.mark(lv, detail="ran")
        big.unit_end(); v = big.exit_gate(0)                                              # trunk_chunk: gate engaged, no site skip, no mark
        self.assertEqual(v["exit_code"], 3)


class TestChunkCensusExitFields(unittest.TestCase):
    """EXIT `msa_chunk=` / `opm_chunk=` / `trunk_chunk=`: `<items served>/<items>` when the chunked site ran, `skipped:<word>` when it only stood
    down by name (narrowed_extent | below_gate) — engagement vs named skip is evidenced on the exit line (chai1_opt 0.4.30)."""

    def setUp(self):
        big.reset_for_tests()

    def tearDown(self):
        big.reset_for_tests()

    def test_no_line_no_fields(self):
        self.assertEqual(big.chunk_census_fields(), [])

    def test_served_and_named_skips_on_a_synthetic_census(self):
        rec = big.apply_line({}, environ={}, mode="big")
        big._STATE["min_n"].update(trunk_chunk=2048, opm_chunk=2048, msa_chunk=1024)
        rec.unit_begin("item0"); big._STATE.update(unit="item0", n_units=1, unit_n=1600)
        big._mark_once(rec, "msa_chunk", detail="rows=512 S=4000 n_chunks=8")                 # the PWA site chunked
        big._site_skip(rec, "opm_chunk", 1600, 2048, "outer_product_mean")                     # the OPM site met the narrowed extent
        for lv in rec.levers:
            if lv not in ("msa_chunk", "opm_chunk", "trunk_chunk"):
                rec.mark(lv, detail="ran")
        big.unit_end()                                                                          # trunk_chunk: below its gate, site not entered -> unit-close named skip
        f = big.chunk_census_fields()
        self.assertIn("msa_chunk=1/1", f); self.assertIn("opm_chunk=skipped:narrowed_extent", f); self.assertIn("trunk_chunk=skipped:below_gate", f)
        from chai1_opt import report
        self.assertTrue({"msa_chunk=1/1", "opm_chunk=skipped:narrowed_extent", "trunk_chunk=skipped:below_gate"} <= set(report.optin_tally_fields()))
        self.assertEqual(big.exit_gate(0)["exit_code"], 0)

    def test_fast_line_names_only_its_chunk_lever(self):
        rec = big.apply_line({}, environ={}, mode="fast")                                      # fast: msa_rows + msa_chunk, nograph gated
        rec.unit_begin("item0"); big._STATE.update(unit="item0", n_units=1, unit_n=512)
        rec.mark("msa_rows", detail="S=4096->4000"); big.unit_end()
        f = big.chunk_census_fields()
        self.assertEqual([x.split("=")[0] for x in f], ["msa_chunk"]); self.assertEqual(f, ["msa_chunk=skipped:below_gate"])
