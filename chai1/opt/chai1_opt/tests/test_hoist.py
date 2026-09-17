"""The item-keyed hoist (hoist.py) on the eager-stack stubs, no torch: an adopted denoiser hoists once per item however many steps the
item takes; a new item whose static tensors land where the previous item's were (the stub's data-pointer key matches) is hoisted anew
(the stack's own key would have served the stale precompute); a new item is hoisted anew after the previous one was released at the
trunk boundary, with the previous item's cache / graph / static tensors dropped; the counters and the verdict (one hoist per item, a
mismatch named either way); the activation line and the exit tally carry the state; adopt refuses parts that are not the stack's own;
the carried stack's names this module binds to are present in its bytes."""
import gc
import os
import unittest
import weakref

from chai1_opt import hoist, report, stack
from chai1_opt.tests import _stubs


class FakeTensor:
    """shape / dtype / data_ptr like a torch tensor; ``ptr`` stands for the address the allocator hands out."""
    def __init__(self, tag, shape, ptr, dtype="torch.float32"):
        self.tag, self.shape, self.ptr, self.dtype = tag, tuple(shape), ptr, dtype

    def data_ptr(self):
        return self.ptr


def item(tag, n=16, ptr=1000):
    return {"token_pair_trunk_repr": FakeTensor(tag, (1, n, n, 128), ptr), "atom_single_input_feats": FakeTensor(tag, (1, 8 * n, 64), ptr + 1),
            "atom_noised_coords": FakeTensor(tag, (1, 5, 8 * n, 3), ptr + 2), "noise_sigma": FakeTensor(tag, (1, 5), ptr + 3),
            "num_steps": 200}


class TestItemKeyedHoist(unittest.TestCase):
    def setUp(self):
        self.S = _stubs.make_eager_module()
        self.parts = self.S.build_parts(None, "tier1", graphed=True)
        self.dw, self.tw = self.parts["diffusion"], self.parts["trunk"]

    def test_adopt_reclasses_in_place_and_keeps_state(self):
        self.dw.ln_policy = _stubs.PolicyStub("big")
        st = hoist.adopt(self.parts, self.S)
        self.assertEqual(st, {"keyed": "item", "release": "trunk_boundary"})
        self.assertIs(self.parts["diffusion"], self.dw); self.assertIs(self.parts["trunk"], self.tw)
        self.assertIsInstance(self.dw, self.S.HoistedDiffusionWrapper); self.assertIsInstance(self.tw, self.S.EagerTrunkWrapper)
        self.assertEqual(type(self.dw).__name__, "ItemKeyedDenoiser"); self.assertEqual(type(self.tw).__name__, "ItemBoundaryTrunk")
        self.assertEqual(self.dw.ln_policy.mode, "big")
        self.assertEqual(hoist.stats(self.dw), {"keyed": "item", "release": "trunk_boundary", "n_precompute": 0, "n_release": 0})
        self.assertIsNone(hoist.stats(_stubs.DiffusionStub()))

    def test_adopt_refuses_foreign_parts(self):
        with self.assertRaises(TypeError):
            hoist.adopt({"diffusion": object(), "trunk": self.tw}, self.S)

    def test_one_hoist_per_item_over_the_steps(self):
        hoist.adopt(self.parts, self.S)
        a = item("a")
        self.tw.forward(256, **{})
        for _ in range(398):
            self.assertEqual(self.dw.forward(256, **a), "a")
        self.assertEqual((self.dw.n_precompute, self.dw.precomputes), (1, ["a"]))

    def test_new_item_at_the_same_addresses_is_hoisted_anew(self):
        hoist.adopt(self.parts, self.S)
        a = item("a", ptr=1000)
        self.assertEqual(self.dw.forward(256, **a), "a")
        del a; gc.collect()                                           # the previous item's tensors are freed ...
        b = item("b", ptr=1000)                                       # ... and the next item's land at the same addresses: the stub's own key matches
        self.assertEqual(self.dw.key, _stubs.DiffusionStub.key_of(256, b))
        self.assertEqual(self.dw.forward(256, **b), "b")              # hoisted anew: b's conditioning, not a's
        self.assertEqual((self.dw.n_precompute, self.dw.precomputes), (2, ["a", "b"]))

    def test_same_addresses_same_item_alive_is_one_hoist(self):
        hoist.adopt(self.parts, self.S)
        a = item("a")
        self.dw.forward(256, **a); self.dw.forward(256, **a)
        self.assertEqual(self.dw.n_precompute, 1)

    def test_trunk_boundary_releases_the_previous_item(self):
        hoist.adopt(self.parts, self.S)
        a = item("a")
        self.tw.forward(256); self.dw.forward(256, **a)
        hf = self.dw.hf[256]
        self.assertIsNotNone(hf.cache); self.assertIsNotNone(hf._graph); self.assertIsNotNone(hf._static_kw)
        self.assertEqual(self.tw.forward(256), ("trunk", 256))         # item b's first trunk call
        self.assertEqual((hf.cache, hf._graph, hf._static_kw, hf._static_out), (None, None, None, None))
        self.assertIsNone(self.dw.key); self.assertEqual(self.dw.n_release, 1)
        self.tw.forward(256); self.tw.forward(256)                    # recycles: nothing resident, nothing counted
        self.assertEqual(self.dw.n_release, 1)
        b = item("b")
        self.assertEqual(self.dw.forward(256, **b), "b")
        self.assertEqual((self.dw.n_precompute, self.dw.n_release), (2, 1))

    def test_an_item_that_dies_in_the_trunk_is_accounting_not_a_stale_hoist(self):
        """Several items of one process die in the trunk (out of memory) — `hoist_precomputes=2!=6`: a seed-fold that raises inside the trunk never
        reaches its denoiser — no hoist, nothing resident (the boundary release ran at its first trunk call) — and the NEXT item is hoisted
        anew from its own tensors (never served the earlier item's precompute). The counters then read n_precompute = completed folds <
        attempted folds: hoist.verdict(…, ok=completed) is None and hoist.unreached names the shortfall; a completed fold WITHOUT its own
        hoist stays a MISMATCH."""
        hoist.adopt(self.parts, self.S)
        a, c = item("a"), item("c", ptr=1000)                                # c lands on a's addresses (the carried data-pointer key alone would call it the same item)
        self.tw.forward(256); self.assertEqual(self.dw.forward(256, **a), "a")
        hf = self.dw.hf[256]
        self.tw.forward(256)                                                 # item b's first trunk call releases a's hoist ...
        self.assertEqual((hf.cache, hf._graph, self.dw.key, self.dw.n_release), (None, None, None, 1))
        # ... and b dies inside the trunk (OOM) before any denoiser call: nothing to hoist, nothing resident
        self.tw.forward(256)                                                 # item c's first trunk call: nothing resident, nothing counted
        self.assertEqual(self.dw.n_release, 1)
        self.assertEqual(self.dw.forward(256, **c), "c")                    # hoisted anew from c's tensors — a stale serve would have answered "a"
        self.assertEqual((self.dw.n_precompute, self.dw.precomputes), (2, ["a", "c"]))
        st = hoist.stats(self.dw)
        self.assertIsNone(hoist.verdict(st, 3, 2)); self.assertEqual(hoist.unreached(st, 3, 2), 1)      # 3 attempted, 2 completed, 2 hoists: accounting, named
        self.assertIn("stale", hoist.verdict(st, 3, 3))                                                   # had all 3 completed with 2 hoists: a stale precompute served
        f = report.eager_tally_fields({"diffusion_events": [], "hoist": st}, "tier1", 3, 2)
        self.assertEqual(f[-2:], ["hoist_precomputes=2/3 hoist_releases=1", "hoist_unreached=1"])

    def test_release_all_covers_every_adopted_denoiser(self):
        hoist.adopt(self.parts, self.S)
        other = self.S.build_parts(None, "tier1", graphed=False); hoist.adopt(other, self.S)
        self.dw.forward(256, **item("a")); other["diffusion"].forward(512, **item("c", n=32))
        self.assertEqual(hoist.release_all(), 2)
        self.assertEqual(hoist.release_all(), 0)

    def test_confidence_head_entry_releases_the_item(self):
        """adopt(parts, S, handle=h) wraps the handle's loader ONCE: the part served for confidence_head.pt releases every adopted denoiser's item
        before its forward (the diffusion phase is over), the second sample's call is a no-op, every other component is served untouched, the
        loader identity the activation probes read moves with the wrap (h.loader is C1.load_exported), and the trunk-boundary release still
        covers an item whose confidence head never ran."""
        import types
        served = []

        class Part:
            def __init__(self, key): self.key, self.flat = key, f"flat:{key}"
            def forward(self, *a, **kw): served.append(self.key); return ("out", self.key)

        C1 = types.SimpleNamespace()
        loader = lambda comp_key, device: Part(comp_key)                                       # noqa: E731 — the stack's make_loader shape: (comp_key, device) -> part
        C1.load_exported = loader
        h = self.S.StackHandle(C1, None, loader, self.parts)
        st = hoist.adopt(self.parts, self.S, handle=h)
        self.assertEqual(st, {"keyed": "item", "release": "trunk_boundary"})                    # the activation word is unchanged
        self.assertIsNot(h.loader, loader); self.assertIs(C1.load_exported, h.loader)           # wrapped once, both names moved (the W1 / eager probes read the pair)
        self.assertIs(h.loader.__wrapped__, loader); self.assertTrue(getattr(h.loader, hoist.LOADER_ATTR))
        self.assertFalse(hoist.serve(h, self.parts))                                            # idempotent
        emb = h.loader("token_embedder.pt", "cuda:0"); conf = h.loader("confidence_head.pt", "cuda:0")
        self.assertIsInstance(emb, Part); self.assertIsInstance(conf, hoist.ConfidenceEntry)
        self.assertEqual(conf.flat, "flat:confidence_head.pt")                                 # attributes pass through (pairtrack's exactln bind reads .flat)
        n0 = hoist.conf_entry_releases()
        a = item("a")
        self.tw.forward(256); self.dw.forward(256, **a); self.dw.forward(256, **a)                   # the item's steps: one hoist
        hf = self.dw.hf[256]
        self.assertIsNotNone(hf.cache); self.assertEqual((self.dw.n_precompute, self.dw.n_release), (1, 0))
        self.assertEqual(conf.forward(crop_size=256), ("out", "confidence_head.pt"))            # sample 0: the item's state is dropped BEFORE the head runs
        self.assertEqual((hf.cache, hf._graph, hf._static_kw, self.dw.key, self.dw.n_release), (None, None, None, None, 1))
        for _ in range(4):
            conf.forward(crop_size=256)                                                          # samples 1..4: nothing resident, nothing counted
        self.assertEqual((self.dw.n_release, hoist.conf_entry_releases() - n0, served.count("confidence_head.pt")), (1, 1, 5))
        self.tw.forward(256)                                                                     # the next item's trunk call: nothing left to release
        self.assertEqual(self.dw.n_release, 1)
        self.assertEqual(self.dw.forward(256, **item("b")), "b"); self.assertEqual(self.dw.n_precompute, 2)
        self.tw.forward(256)                                                                     # an item whose confidence head never ran: the trunk boundary releases it
        self.assertEqual(self.dw.n_release, 2)
        no_flat = dict(self.parts, flat_rest=False)
        h2 = self.S.StackHandle(types.SimpleNamespace(load_exported=loader), None, loader, no_flat)
        self.assertFalse(hoist.serve(h2, no_flat)); self.assertIs(h2.loader, loader)             # a line serving upstream's TorchScript head: nothing to wrap

    def test_verdict(self):
        st = {"n_precompute": 3}
        self.assertIsNone(hoist.verdict(st, 3)); self.assertIsNone(hoist.verdict(None, 3)); self.assertIsNone(hoist.verdict(st, None))
        self.assertIn("stale", hoist.verdict(st, 4)); self.assertIn("more than once", hoist.verdict(st, 2))
        # fewer hoists than ATTEMPTED seed-folds is accounting, not a stale hoist, when every COMPLETED fold is covered
        self.assertIsNone(hoist.verdict({"n_precompute": 2}, 6, 2)); self.assertEqual(hoist.unreached({"n_precompute": 2}, 6, 2), 4)   # 4 folds died before their denoiser
        self.assertIn("stale", hoist.verdict({"n_precompute": 2}, 6, 3)); self.assertIsNone(hoist.unreached({"n_precompute": 2}, 6, 3))   # a completed fold without its own hoist
        self.assertIsNone(hoist.unreached(st, 3, 3)); self.assertIsNone(hoist.unreached(None, 3)); self.assertIn("stale", hoist.verdict(st, 4, None))   # no ok count: today's strict rule

    def test_static_key_and_same_item(self):
        a = item("a")
        k = hoist.static_key(256, a)
        self.assertEqual(k[0], 256); self.assertEqual([x[0] for x in k[1:]], sorted(n for n in a if n != "num_steps"))
        self.assertNotEqual(k, hoist.static_key(256, item("a", n=17)))
        refs = tuple(weakref.ref(a[n]) for n in hoist.ITEM_TENSORS)
        self.assertTrue(hoist.same_item(refs, a)); self.assertFalse(hoist.same_item((), a))
        del a; gc.collect()
        self.assertFalse(hoist.same_item(refs, item("a")))

    def test_lines_carry_the_state(self):
        rep = {"active": True, "mode": "fast", "levels": "W1,W2,W5", "eager": "tier1", "dstep": "hoist2,compiled,dit_attn", "levers_applied": ["W1"],
               "det": 1, "route": "driver", "chai_lab_version": "0.6.1", "torch_version": "2.13.0",
               "gpu": {"visible": True, "name": "H100", "cc": "9.0"}, "hoist": dict(hoist.STATE)}
        line = report.activation_line(rep)
        self.assertRegex(line, r" gpu=H100\(sm90\) hoist=item/trunk_boundary$")
        ns = {"summary": [{"n": 2, "n_ok": 2, "rows": []}, {"n": 1, "n_ok": 1, "rows": [{}]}]}
        self.assertEqual(report.items_folded(ns), 3); self.assertIsNone(report.items_folded({}))
        stats = {"diffusion_events": [], "hoist": {"keyed": "item", "release": "trunk_boundary", "n_precompute": 3, "n_release": 2}}
        f = report.eager_tally_fields(stats, "tier1", 3)
        self.assertEqual(f[-1], "hoist_precomputes=3/3 hoist_releases=2")
        f = report.eager_tally_fields(stats, "tier1", 4)                    # no ok count: the strict rule (a completed fold without its hoist)
        self.assertEqual(f[-2:], ["hoist_precomputes=3/4 hoist_releases=2", "HOIST MISMATCH hoist_precomputes=3<4 completed seed-folds: a completed item was denoised without a hoist of its own (a stale precompute served)"])
        f = report.eager_tally_fields(stats, "tier1", 4, 3)                 # 4 attempted, 3 completed, 3 hoists: one fold died before its denoiser — accounting, named, no MISMATCH
        self.assertEqual(f[-2:], ["hoist_precomputes=3/4 hoist_releases=2", "hoist_unreached=1"])
        f = report.eager_tally_fields(stats, "tier1", 4, 4)                 # 4 completed but 3 hoists: a stale precompute served one of them
        self.assertEqual(f[-1], "HOIST MISMATCH hoist_precomputes=3<4 completed seed-folds: a completed item was denoised without a hoist of its own (a stale precompute served)")
        self.assertEqual(report.eager_tally_fields(stats, "tier1", None)[-1], "hoist_precomputes=3/? hoist_releases=2")
        ns2 = {"summary": [{"n": 3, "n_ok": 1, "rows": []}]}
        self.assertEqual((report.items_folded(ns2), report.items_ok(ns2)), (3, 1)); self.assertIsNone(report.items_ok({}))
        ns3 = {"summary": [{"n": 2, "n_ok": 2, "trunk_samples": 2, "rows": []}, {"n": 1, "n_ok": 0, "rows": []}]}   # two trunk samples: one hoist per fold call = seed-folds x trunk samples
        self.assertEqual((report.items_folded(ns3), report.items_ok(ns3)), (5, 4))
        stats2 = {"diffusion_events": [], "hoist": {"keyed": "item", "release": "trunk_boundary", "n_precompute": 4, "n_release": 3}}
        self.assertEqual(report.eager_tally_fields(stats2, "tier1", 5, 4)[-2:], ["hoist_precomputes=4/5 hoist_releases=3", "hoist_unreached=1"])   # 2 seeds x 2 trunks hoisted, one failed fold never reached its denoiser
        self.assertIsNone(hoist.verdict(stats2["hoist"], 4, 4)); self.assertIn("HOIST MISMATCH", " ".join(report.eager_tally_fields(stats2, "tier1", 3, 3)))   # 4 hoists for 3 fold calls: one hoisted twice = MISMATCH
        self.assertNotIn("hoist", " ".join(report.eager_tally_fields({"diffusion_events": []}, "tier1", 3)))

    def test_carried_stack_names(self):
        """The names this module binds to in the carried eager stack, present in its bytes: the wrapper classes, ``_move``, the
        data-pointer key the item key replaces, and HoistedForward's per-item fields."""
        home = os.path.join(stack.eager_home(), "chai1_eager")
        st = open(os.path.join(home, "stack.py"), encoding="utf-8").read(); hf = open(os.path.join(home, "hoist.py"), encoding="utf-8").read()
        for s in ("class HoistedDiffusionWrapper", "class EagerTrunkWrapper", "def _move(kw, dev)",
                  'key = (crop_size, kw["token_pair_trunk_repr"].data_ptr(), kw["atom_single_input_feats"].data_ptr(), tuple(kw["atom_noised_coords"].shape))',
                  "if key != self.key:", "def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):"):
            self.assertIn(s, st, s)
        for s in ("self.cache = None", "self._graph = None", "self._static_kw = dict(kw)", "self._static_out = self._step("):
            self.assertIn(s, hf, s)
        self.assertEqual(hoist.HOIST_FIELDS, ("cache", "_graph", "_static_kw", "_static_out"))


if __name__ == "__main__":
    unittest.main()


class TestJitCaches(unittest.TestCase):
    """chai1_opt/jit.py: the levers' JIT caches keyed under MODEL_OPT_JIT_ROOT (the kits' convention), set only where the user has not."""
    def test_root_keys_the_unset_cache_dirs_and_says_so(self):
        from chai1_opt import jit
        class T:                                     # a torch stand-in: version words only
            __version__ = "2.13.0+cu130"
            class version: cuda = "13.0"
            class cuda:
                @staticmethod
                def is_available(): raise AssertionError("jit.cache_key must not touch CUDA (the alloc lever reads its setting at CUDA init)")
                get_device_capability = is_available
        self.assertEqual(jit.cache_key(T, {"MODEL_OPT_TARGET_GPU": "H100"}), "torch2.13.0-cu130-h100"); self.assertEqual(jit.cache_key(T, {}), "torch2.13.0-cu130-gpu")
        env = {}
        f = jit.apply(env, T); self.assertIsNone(f["root"]); self.assertEqual(env, {}); self.assertIn("MODEL_OPT_JIT_ROOT unset", jit.note(f))
        env = {"MODEL_OPT_JIT_ROOT": "/jit", "TRITON_CACHE_DIR": "/mine", "MODEL_OPT_TARGET_GPU": "H100"}
        f = jit.apply(env, T)
        self.assertEqual(env["TRITON_CACHE_DIR"], "/mine"); self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], "/jit/torch2.13.0-cu130-h100/inductor"); self.assertEqual(f["set"], ["TORCHINDUCTOR_CACHE_DIR"])
        self.assertIn("TORCHINDUCTOR_CACHE_DIR=/jit/torch2.13.0-cu130-h100/inductor", jit.note(f)); self.assertIn("key=torch2.13.0-cu130-h100", jit.note(f))
        env = {"MODEL_OPT_JIT_ROOT": "/jit", "MODEL_OPT_STACK_KEY": "k1"}
        jit.apply(env, T); self.assertEqual(env["TRITON_CACHE_DIR"], "/jit/k1/triton")


class TestHoist2ScalarFold(unittest.TestCase):
    """hoist2 stores int(x)/float(x)/bool(x) for cached CPU scalars the step reads only through that cast (chai1_eager 0.4.5): no data-dependent
    tensor read left in a trace of the step; a name also read as a tensor, read through two casts, protected or stored stays a tensor."""
    def test_scalar_only_uses(self):
        import ast, sys, os
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "..", "forward", "eager_trunk"))
        from chai1_eager.hoist import scalar_only_uses
        body = ast.parse("y = torch.layer_norm(x, [int(a)], None, None, 0.1)\nz = torch.reshape(y, [int(b), int(b), c])\nw = torch.mul(d, 2) + int(d)\nu = float(e) + int(e)\nf = int(f)\nreturn (z, bool(g))").body
        got = scalar_only_uses(body, ["a", "b", "c", "d", "e", "f", "g", "h"], protect=("h",))
        self.assertEqual(got, {"a": "int", "b": "int", "g": "bool"})     # c: read bare (tensor operand); d: also a tensor operand; e: two casts; f: stored; h: protected


class TestDitAttnStepAsideWords(unittest.TestCase):
    """`dit_attn` asks opt_core's kernels.apb by the MODE's tier word (chai1_fastln 0.3.x): one selection per (tokens, samples) per process; a
    provider row that IS the library statement (sdpa rows) or a Refusal leaves the statement serving BY NAME; an install-time Refusal for the
    whole cell steps the lever aside by a word built from the Refusal kind; anything else propagates."""
    @classmethod
    def setUpClass(cls):
        import importlib.util, os
        try:
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"torch not importable here: {e}")
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "forward", "dstep_megakernel", "chai1_fastln", "dit_attn.py")
        spec = importlib.util.spec_from_file_location("chai1_fastln_dit_attn_undertest", path)
        cls.DA = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.DA)

    def _fake_apb(self, rows=None, refuse_kind=None, unexpected=None):
        """kernels.apb stand-in: select() returns Sel(row=rows[n_tokens] or "fpf_apb"), or raises Refusal(kind) / an unexpected error."""
        class Refusal(Exception):
            def __init__(self, kind, row=None, fallback=None):
                Exception.__init__(self, kind); self.kind, self.row, self.fallback = kind, row, fallback
        class Sel:
            def __init__(self, row): self.row, self.variant = row, None
        class Fake:
            pass
        F = Fake(); F.Refusal = Refusal; F.calls = []
        def select(cc, dtype, cell, n_tokens, *, word, samples=1, timing="eager", capture=False, head_dim=None, heads=None, **kw):
            F.calls.append((word, tuple(cc), cell, int(n_tokens), int(samples), timing, capture, head_dim, heads))
            if unexpected: raise unexpected
            if refuse_kind: raise Refusal(refuse_kind, None, None)
            return Sel((rows or {}).get(int(n_tokens), "fpf_apb"))
        F.select = select
        return F

    def setUp(self):
        self.DA._STATE.update(selections={}, rows={}, apb=None, cc=(9, 0)); self.DA.WORD = "fast"
        self.DA.STATS.update(served_by=None, refusal=None, rows={}, statement=0, served=0, calls=0)

    def test_module_surface(self):                                   # the Router's call path needs these names at fold time (a lost one is a NameError inside the fold)
        for n in ("_op", "admits", "Router", "available", "step_aside_word", "_row_for", "row_decision", "_serve4", "_stock4", "patch_flat", "unpatch_flat", "report", "shim_of", "STATEMENT_ROWS"):
            self.assertTrue(hasattr(self.DA, n), n)
        import torch
        q = torch.zeros(1, 16, 5, 8, 48); m = torch.zeros(1, 16, 1, 8, 8)
        self.assertEqual(self.DA.admits(q, q, q, m, 0.0, False, None), "device")          # the DiT form, but CPU: takes the statement
        self.assertEqual(self.DA.admits(q[0], q[0], q[0], m[0], 0.0, False, None), "not_5d")
        r = self.DA.Router(torch.nn.functional.scaled_dot_product_attention, aside="no_apb_row_cc80")
        self.assertEqual(tuple(r(q, q, q, m).shape), (1, 16, 5, 8, 48))                   # stepped aside: the original statement serves

    def test_tier_word_selection_per_token_count(self):
        DA = self.DA
        F = self._fake_apb(rows={1024: "sdpa", 2048: "sdpa"}); DA._STATE["apb"] = F; DA.WORD = "big"; DA._STATE["cc"] = (8, 0)
        self.assertEqual(DA._row_for_impl(512, 5, 48, 16), "fpf_apb"); self.assertEqual(DA.row_decision("fpf_apb"), "provider")
        self.assertEqual(DA._row_for_impl(1024, 5, 48, 16), "sdpa"); self.assertEqual(DA.row_decision("sdpa"), "statement")   # the row IS the library statement: the original call serves, by name
        self.assertEqual(DA._row_for_impl(512, 5, 48, 16), "fpf_apb"); self.assertEqual(len(F.calls), 2)                    # one select per (tokens, samples) per process
        self.assertEqual(F.calls[0], ("big", (8, 0), "dit_h16d48", 512, 5, "graphed", True, 48, 16))                 # asked by the tier word, literally
        self.assertEqual(DA.STATS["served_by"], "fpf_apb"); self.assertIn("sdpa", DA.STATS["rows"])
        F2 = self._fake_apb(refuse_kind="head_dim:48 not routed"); DA._STATE.update(apb=F2, selections={}, rows={})
        self.assertEqual(DA._row_for_impl(768, 5, 48, 16), "refused:head_dim"); self.assertEqual(DA.row_decision("refused:head_dim"), "statement")

    def test_words(self):
        DA = self.DA
        self.assertEqual(DA.step_aside_word("cc:8.0!=9.0(sm_90 binary)", (8, 0)), "no_apb_row_cc80")
        self.assertEqual(DA.step_aside_word("dtype:bf16 (fp32 statement only)"), "no_apb_row_dtype")
        F = self._fake_apb(refuse_kind="cc:7.5 (no row)")
        self.assertEqual(DA.available(cc=(7, 5), apb=F, word="fast"), "no_apb_row_cc75")                             # no row at all for the cell: steps aside by name
        F = self._fake_apb(); self.assertIsNone(DA.available(cc=(8, 0), apb=F, word="big"))                          # a kernel row or the statement row: either serves
        self.assertEqual([c[3] for c in F.calls], [256, 2048]); self.assertEqual({c[0] for c in F.calls}, {"big"})
        with self.assertRaises(ValueError):                                                                            # unexpected failures still refuse loudly
            DA.available(cc=(9, 0), apb=self._fake_apb(unexpected=ValueError("boom")), word="fast")
