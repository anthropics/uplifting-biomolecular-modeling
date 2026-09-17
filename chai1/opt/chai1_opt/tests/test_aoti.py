"""chai1_fastln.aoti — the compiled step ahead of time (chai1_fastln 0.3.3): CPU-only checks of everything but the export/compile/load themselves
(those need the card and the CUDA headers; `run.sh warm` and the fold prove them on the box). Naming and lookup (`package_name`, `find`, `listing`:
a package serves only its own crop, n_samples, lever word and code digest — another count or digest is ABSENT by name, a .partial write is never
listed), the input/constant partition of a step's arguments and cache (`_plan`: device tensors and CPU tensors with elements are inputs, cached
0-d CPU tensors / ints / lists / None are per-crop constants whose rendering is digested — `_consts_digest` changes with a constant, not with an
input's values), `Dispatch` (serves a package when one is found, else builds the Dynamo step lazily ONCE through the injected factory and names the
absence; an Unusable package falls to Dynamo by name), and the kit's words (`modes.compile_word` on:aoti|on:dynamo — see test_activation_rules)."""
import json
import os
import sys
import tempfile
import unittest

from chai1_opt import stack

torch = None   # imported in setUpModule (the suite collects without torch; chai1_opt imports it lazily too)
A = None       # chai1_fastln.aoti, imported in setUpModule through the kit's own add-on homes (the documented test command puts opt/ and the core on the path, not the add-ons)


def setUpModule():  # noqa: N802
    """chai1_fastln (and chai1_eager beside it) live under opt/forward/…: reach them the way the kit does at run time (stack.dstep_home / eager_home)."""
    global A, torch
    try:
        import torch as _torch
    except ImportError as e:                                                  # named, never silent: the module's checks need torch's CPU tensors
        raise unittest.SkipTest(f"chai1_fastln.aoti tests need torch (CPU is enough): {e}")
    torch = _torch
    for home in (stack.eager_home(), stack.dstep_home()):
        home in sys.path or sys.path.insert(0, home)
    from chai1_fastln import aoti as _aoti
    A = _aoti


class FakeHF:
    """The hoisted step's surface aoti reads: argnames, cache, src, method, root, _step (the eager statements), events."""
    method = "forward_512"

    def __init__(self):
        self.argnames = ["atom_coords", "noise_sigma", "mask"]
        self.cache = {"pair": torch.zeros(2, 3), "n_blocks": 4, "shape0": torch.tensor(7), "biases": [torch.ones(2), None], "flag": None}
        self.src = "def _step(root, atom_coords, noise_sigma, mask, cache):\n    return atom_coords * 2\n"
        self.root = torch.nn.Linear(1, 1)
        self.compiled, self._compile_mode, self.stats = "default", "default", {}
        self.calls = []
        self._step = lambda root, *rest: ("eager", len(rest))


class TestNaming(unittest.TestCase):
    def test_package_name_and_find(self):
        with tempfile.TemporaryDirectory() as d:
            name = A.package_name(512, 5, "hoist2+dit_attn", "0123456789ab")
            self.assertEqual(name, "dstep_c512_s5_hoist2+dit_attn_0123456789ab.pt2")
            for n in (name, A.package_name(512, 4, "hoist2+dit_attn", "0123456789ab"), A.package_name(512, 5, "hoist2", "0123456789ab"),
                      A.package_name(768, 5, "hoist2+dit_attn", "0123456789ab"), A.package_name(512, 5, "hoist2+dit_attn", "ffffffffffff")):
                open(os.path.join(d, n), "w").close(); open(os.path.join(d, n[:-4] + ".meta.json"), "w").write(json.dumps({"name": n[:-4]}))
            open(os.path.join(d, "dstep_c1024_s5_hoist2_0123456789ab.pt2.partial"), "w").close()          # a write in progress: never listed
            hits = A.find(d, 512, 5, "hoist2+dit_attn", "0123456789ab")
            self.assertEqual([os.path.basename(h) for h in hits], [name])
            self.assertEqual(len(A.find(d, 512, None, "hoist2+dit_attn", "0123456789ab")), 2)         # n_samples unknown yet: both counts are candidates
            self.assertEqual(A.find(d, 512, 3, "hoist2+dit_attn", "0123456789ab"), [])              # another --num-diffn-samples: absent by name
            self.assertEqual(A.find(d, 512, 5, "hoist2+dit_attn", "000000000000"), [])              # other code: absent by name
            self.assertEqual(A.find(None, 512, 5, "hoist2+dit_attn", "0123456789ab"), [])            # no root: nothing
            self.assertNotIn("dstep_c1024_s5_hoist2_0123456789ab.pt2.partial", [os.path.basename(p) for p in A.listing(d)])
            self.assertEqual(len(A.listing(d)), 5)

    def test_package_dir_follows_the_jit_root_and_key(self):
        self.assertIsNone(A.package_dir({}))
        d = A.package_dir({"MODEL_OPT_JIT_ROOT": "/x/jit", "MODEL_OPT_STACK_KEY": "torch2.13.0-cu130-sm90"})
        self.assertEqual(d, "/x/jit/torch2.13.0-cu130-sm90/aoti")

    def test_levers_word(self):
        self.assertEqual(A.levers_word_of(hoister="hoist2", dit_attn=True), "hoist2+dit_attn")
        self.assertEqual(A.levers_word_of(hoister="hoist2", dit_attn=False), "hoist2")

    def test_step_digest_moves_with_the_source(self):
        hf = FakeHF(); d1 = A.step_digest(hf)
        self.assertRegex(d1, r"^[0-9a-f]{12}$")
        hf.src += "    # changed\n"
        self.assertNotEqual(A.step_digest(hf), d1)


class TestPlan(unittest.TestCase):
    def test_inputs_and_constants(self):
        hf = FakeHF()
        args = [torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool)]
        leaves, kinds, spec = A._plan(args, hf.cache)
        n_in = sum(1 for k in kinds if k == "i")
        # inputs: the 3 args + cache 'pair' (2x3) + cache biases[0] (2,) = 5; constants: n_blocks, shape0 (0-d CPU) (+ the None leaves as the pytree renders them)
        self.assertEqual(n_in, 5)
        self.assertGreaterEqual(len(kinds) - n_in, 2)
        dig = A._consts_digest(leaves, kinds, spec)
        args2 = [a + 1 if a.dtype != torch.bool else ~a for a in args]                                  # input VALUES never enter the digest
        l2, k2, s2 = A._plan(args2, hf.cache); self.assertEqual(A._consts_digest(l2, k2, s2), dig)
        hf.cache["n_blocks"] = 5                                                                          # a constant does
        l3, k3, s3 = A._plan(args, hf.cache); self.assertNotEqual(A._consts_digest(l3, k3, s3), dig)
        hf.cache["n_blocks"] = 4; hf.cache["shape0"] = torch.tensor(8)                                    # a cached 0-d CPU tensor is a constant too
        l4, k4, s4 = A._plan(args, hf.cache); self.assertNotEqual(A._consts_digest(l4, k4, s4), dig)


class TestWarmCli(unittest.TestCase):
    """`python -m chai1_opt.warm_aoti` imports cleanly without CUDA and names why it skips (applicable: no lever / no JIT root / no CUDA headers)."""
    def test_import_and_applicable_words(self):
        from chai1_opt import warm_aoti as W
        ok, why = W.applicable("exact", environ={})
        self.assertFalse(ok); self.assertIn("does not carry the compiled step", why)
        ok, why = W.applicable("fast", environ={})
        self.assertFalse(ok); self.assertIn("MODEL_OPT_JIT_ROOT", why)
        ok, why = W.applicable("fast", environ={"MODEL_OPT_JIT_ROOT": "/x", "CUDA_HOME": "/nonexistent-cuda"})
        self.assertFalse(ok); self.assertIn("CUDA development headers", why)
        self.assertIn("skipped", W.step_line({"status": "skipped", "reason": "x", "wall_s": 0.0, "crops": []}))


class TestDispatch(unittest.TestCase):
    def test_absent_builds_dynamo_once_by_name(self):
        hf = FakeHF(); built = []; events = []

        def factory():
            built.append(1)
            return lambda root, *rest: ("dynamo", len(rest))
        with tempfile.TemporaryDirectory() as d:
            disp = A.Dispatch(hf, d, "hoist2+dit_attn", factory, events=events)
            args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
            before = len(A.STATS["absent"])
            out1 = disp(hf.root, *args, hf.cache); out2 = disp(hf.root, *args, hf.cache)
            self.assertEqual(out1, ("dynamo", 4)); self.assertEqual(out2, ("dynamo", 4))
            self.assertEqual(len(built), 1)                                                               # the Dynamo step is built once, lazily
            self.assertEqual(len(A.STATS["absent"]) - before, 1)
            self.assertTrue(any(str(e[1] if isinstance(e, tuple) else e).startswith("aoti_absent:c512_s5_") for e in events), events)

    def test_leaf_names_follow_plan_order(self):
        hf = FakeHF()
        args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
        leaves, kinds, spec = A._plan(args, hf.cache)
        names = A._leaf_names(args, hf.cache, ["atom_noised_coords", "noise_sigma", "mask"])
        self.assertEqual(len(names), len(leaves))
        self.assertEqual(names[:3], ["arg:atom_noised_coords", "arg:noise_sigma", "arg:mask"])
        self.assertIn("cache:pair", names); self.assertTrue(any(n.startswith("cache:biases[0]") for n in names), names)

    def test_unaligned_example_and_alignment_probe(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        self.assertTrue(A._aligned(x))
        u = A._unaligned_example(x)
        self.assertFalse(A._aligned(u)); self.assertTrue(torch.equal(u, x)); self.assertEqual(u.stride(), x.stride())
        v = x.reshape(-1)[1:5]                                                                     # a view at a 4-byte offset: unaligned
        self.assertFalse(A._aligned(v)); self.assertTrue(A._aligned(v.clone()))

    def test_realign_replaces_cache_leaves_once_and_args_per_call(self):
        """A package compiled for aligned inputs gets aligned copies: a cache leaf is swapped IN the cache once per item, an argument per call — named once each."""
        hf = FakeHF()
        big = torch.zeros(2 * 3 + 4); hf.cache["pair"] = big[1:7].view(2, 3)                     # an unaligned view in the cache
        pk = A.Package.__new__(A.Package)                                                          # no launcher: exercise _inputs alone
        args = (torch.zeros(1, 5, 3), torch.ones(16)[1:6].view(1, 5), torch.zeros(4, dtype=torch.bool))   # arg 1 unaligned
        leaves, kinds, spec = A._plan(args, hf.cache)
        pk.hf, pk.name, pk.kinds = hf, "dstep_cX", "".join(kinds)
        pk.meta = {"consts_digest": A._consts_digest(leaves, kinds, spec)}
        pk._cache_id, pk._fix_args, pk._said, pk.free = None, (), set(), set()
        pk.in_names = [n for n, k in zip(A._leaf_names(args, hf.cache, ["atom_noised_coords", "noise_sigma", "mask"]), kinds) if k == "i"]
        pk.in_strides = []
        n0 = len(A.STATS["realigned"])
        out = pk._inputs(args, hf.cache)
        self.assertTrue(all(A._aligned(t) for t in out))
        self.assertTrue(A._aligned(hf.cache["pair"])); self.assertTrue(torch.equal(hf.cache["pair"], big[1:7].view(2, 3)))   # swapped in the cache, same values
        out2 = pk._inputs(args, hf.cache)                                                          # same item, next step: the arg is copied again, nothing re-announced
        self.assertTrue(all(A._aligned(t) for t in out2))
        self.assertEqual(len(A.STATS["realigned"]) - n0, 2)                                        # one cache leaf + one arg, each named once
        pk.free = set(pk.in_names)                                                                 # a package compiled to take them unaligned: nothing is copied
        pk._cache_id = None; hf.cache["pair"] = big[1:7].view(2, 3)
        out3 = pk._inputs(args, hf.cache)
        self.assertFalse(A._aligned(out3[1])); self.assertFalse(A._aligned(hf.cache["pair"]))

    def test_foreign_strides_are_materialised_not_refused(self):
        """A mode hands `noise_sigma` as an expanded view (strides (1, 0)) while the package was built on a contiguous (5, 1): the loader copies it per call
        into the recorded layout (a compiled launcher does not re-stride) instead of setting the package aside (chai1_fastln 0.3.8)."""
        hf = FakeHF()
        pk = A.Package.__new__(A.Package)
        built_args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
        leaves, kinds, spec = A._plan(built_args, hf.cache)
        pk.hf, pk.name, pk.kinds = hf, "dstep_cX", "".join(kinds)
        pk.meta = {"consts_digest": A._consts_digest(leaves, kinds, spec)}
        pk._cache_id, pk._fix_args, pk._said, pk.free = None, (), set(), set()
        pk.in_names = [n for n, k in zip(A._leaf_names(built_args, hf.cache, ["atom_noised_coords", "noise_sigma", "mask"]), kinds) if k == "i"]
        pk.in_strides = [tuple(v.stride()) for v, k in zip(leaves, kinds) if k == "i"]
        sigma = torch.tensor([[7.0]]).expand(1, 5)                                                   # strides (1, 0): one value shared by the 5 samples
        self.assertEqual(sigma.stride(), (1, 0))
        n0 = len(A.STATS["realigned"])
        out = pk._inputs((built_args[0], sigma, built_args[2]), hf.cache)
        self.assertEqual(out[1].stride(), (5, 1)); self.assertTrue(torch.equal(out[1], torch.full((1, 5), 7.0)))
        out2 = pk._inputs((built_args[0], torch.tensor([[3.0]]).expand(1, 5), built_args[2]), hf.cache)   # next step: copied again, correct values
        self.assertTrue(torch.equal(out2[1], torch.full((1, 5), 3.0))); self.assertEqual(len(A.STATS["realigned"]) - n0, 1)
        # a size-1 axis with an arbitrary stride is NOT foreign (no copy)
        weird = torch.ones(5).as_strided((1, 5), (99, 1))
        out3 = pk._inputs((built_args[0], weird, built_args[2]), hf.cache)
        self.assertIs(out3[1], weird)



    # ---- chai1_fastln 0.3.11: layouts by META only — no stride heuristics in the release path
    def _metad(self, built_args, cache=None, names=("atom_noised_coords", "noise_sigma", "mask")):
        hf = FakeHF()
        if cache is not None: hf.cache = cache
        pk = A.Package.__new__(A.Package)
        leaves, kinds, spec = A._plan(built_args, hf.cache)
        pk.hf, pk.name, pk.kinds = hf, "dstep_cX_meta", "".join(kinds)
        pk.meta = {"consts_digest": A._consts_digest(leaves, kinds, spec)}
        pk._cache_id, pk._fix_args, pk._said, pk.free = None, (), set(), set()
        pk.in_names = [n for n, k in zip(A._leaf_names(built_args, hf.cache, list(names)), kinds) if k == "i"]
        pk.in_strides = [tuple(v.stride()) for v, k in zip(leaves, kinds) if k == "i"]           # the RECORDED layouts (as warm_aoti writes them at the build)
        return hf, pk

    def test_a_metaless_package_is_refused_by_name_at_load_never_launched(self):
        """No inputs / input_strides in the meta (the sets built before layouts were recorded) -> Refused('no_layout_meta') before the launcher
        is opened; the Dispatch counts it, says it, and the card's no-package route serves (dynamo here; the eager aside on cc 8.0)."""
        with tempfile.TemporaryDirectory() as d:
            hf = FakeHF()
            name = A.package_name(A.crop_of(hf), 5, "hoist2", A.step_digest(hf))
            open(os.path.join(d, name), "w").close()
            open(os.path.join(d, name[:-4] + ".meta.json"), "w").write(json.dumps({"name": name[:-4], "kinds": "iic", "consts_digest": "x", "constants": {}}))
            with self.assertRaises(A.Refused) as cm:
                A.Package(hf, os.path.join(d, name))
            self.assertEqual(cm.exception.word, "no_layout_meta")
            built, events = [], []
            n0 = len(A.STATS["refused"])
            class W0: compile_aside = None
            w0 = W0()
            disp = A.Dispatch(hf, d, "hoist2", lambda: built.append(1) or (lambda root, *rest: ("dynamo", len(rest))), events=events, wrapper=w0, eager=lambda root, *rest: ("eager", len(rest)))
            out = disp(hf.root, torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool), hf.cache)
            self.assertEqual(out[0], "eager"); self.assertIsNone(disp.pkg); self.assertEqual(built, [])       # the EAGER hoisted step, NO Inductor build (the factory was never called)
            self.assertEqual(w0.compile_aside, "no_layout_meta"); self.assertEqual(A.STATS["aside_reason"], "no_layout_meta")   # EXIT compiled=stepped_aside:no_layout_meta
            self.assertTrue(any(w == "compile_aside:no_layout_meta" for _, w, _ in events), events)
            self.assertEqual([r for r in A.STATS["refused"][n0:]], [f"{name[:-4]}:no_layout_meta"])
            self.assertTrue(any(w == "aoti_refused:no_layout_meta" for _, w, _ in events), events)
            # cc 8.0: the named eager aside, not a per-process compile
            hf2 = FakeHF(); built2, events2 = [], []
            class W: compile_aside = None
            disp2 = A.Dispatch(hf2, d, "hoist2", lambda: built2.append(1) or (lambda root, *rest: ("dynamo", 0)), events=events2, aside_reason=A.ASIDE_CC80, wrapper=W(), eager=lambda root, *rest: ("eager", len(rest)))
            out2 = disp2(hf2.root, torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool), hf2.cache)
            self.assertEqual(out2[0], "eager"); self.assertEqual(built2, [])
        s = A.report(); self.assertGreaterEqual(s["n_refused"], 2); self.assertTrue(all(r.endswith(":no_layout_meta") for r in s["refused"][-2:]))

    def test_b_metad_package_restrides_an_expanded_sigma_to_the_recorded_layout_per_sample_bitwise(self):
        built_args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
        hf, pk = self._metad(built_args)
        self.assertEqual(pk.in_strides[1], (5, 1))
        for case in ("0,0", "1,0"):
            for value in (7.0, 3.0):
                sigma = torch.tensor(value).expand(1, 5) if case == "0,0" else torch.tensor([[value]]).expand(1, 5)
                self.assertEqual(sigma.stride(), tuple(int(x) for x in case.split(",")))
                out = pk._inputs((built_args[0], sigma, built_args[2]), hf.cache)
                ref = pk._inputs((built_args[0], torch.full((1, 5), value), built_args[2]), hf.cache)
                self.assertEqual(out[1].stride(), (5, 1))
                for smp in range(5):
                    self.assertTrue(torch.equal(out[1][:, smp], ref[1][:, smp]), (case, value, smp))

    def test_c_metad_package_refuses_by_name_an_arg_whose_recorded_layout_cannot_be_produced(self):
        built_sigma = torch.tensor([[1.0]]).expand(1, 5)                                             # recorded as a broadcast layout (1, 0)
        built_args = (torch.zeros(1, 5, 3), built_sigma, torch.zeros(4, dtype=torch.bool))
        hf, pk = self._metad(built_args)
        self.assertEqual(pk.in_strides[1], (1, 0))
        distinct = torch.arange(5.0).reshape(1, 5)                                                   # per-sample values a zero-stride axis cannot hold
        with self.assertRaises(A.Refused) as cm:
            pk._inputs((built_args[0], distinct, built_args[2]), hf.cache)
        self.assertEqual(cm.exception.word, "layout_unproducible:arg:noise_sigma")

    def test_d_metad_package_passes_a_permuted_arg_whose_strides_match_the_recorded_ones_untouched(self):
        coords = torch.zeros(1, 3, 5).transpose(1, 2)                                                # [1, 5, 3] view, strides (15, 1, 5): the build's own layout
        built_args = (coords, torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
        hf, pk = self._metad(built_args)
        self.assertEqual(pk.in_strides[0], (15, 1, 5))
        live = torch.ones(1, 3, 5).transpose(1, 2)                                                   # same strides, new tensor
        n0 = len(A.STATS["realigned"])
        out = pk._inputs((live, torch.full((1, 5), 2.0), built_args[2]), hf.cache)
        self.assertIs(out[0], live); self.assertEqual(len(A.STATS["realigned"]), n0)                 # untouched, zero copies

    def test_e_cache_leaves_must_match_the_recorded_layout_untouched_or_refused_never_copied(self):
        base = torch.zeros(5, 46, 256)
        cache = {"scale0": base[:, :, :128], "expand_mask": torch.zeros(1, 46, 1, dtype=torch.bool).expand(5, 46, 1), "n_blocks": 4}   # a stride-256 slice + a (0,1,1) expand
        self.assertEqual(cache["scale0"].stride(), (11776, 256, 1)); self.assertEqual(cache["expand_mask"].stride(), (0, 1, 1))
        built_args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
        hf, pk = self._metad(built_args, cache=cache)
        n0 = len(A.STATS["realigned"])
        out = pk._inputs(built_args, hf.cache)                                                       # presented exactly as recorded: untouched, zero copies
        self.assertIs(hf.cache["scale0"], cache["scale0"]); self.assertIs(hf.cache["expand_mask"], cache["expand_mask"]); self.assertEqual(len(A.STATS["realigned"]), n0)
        # a new item presenting the slice CONTIGUOUS (a different producer): refused by name, not copied
        hf2, pk2 = self._metad(built_args, cache=dict(cache))
        pk2.in_strides = list(pk.in_strides)                                                         # the recorded (build) layouts
        hf2.cache["scale0"] = cache["scale0"].contiguous()
        with self.assertRaises(A.Refused) as cm:
            pk2._inputs(built_args, hf2.cache)
        self.assertEqual(cm.exception.word, "layout_unproducible:cache:scale0")
        self.assertTrue(hf2.cache["scale0"].is_contiguous()); self.assertEqual(len(A.STATS["realigned"]), n0)   # nothing copied / stored

    def test_card_aside_reason_words(self):
        from chai1_opt import modes
        self.assertEqual(A.ASIDE_CC80, modes.COMPILE_ASIDE_CC80)                                     # one word on the ACTIVE token and on the step-aside line
        self.assertEqual(A.card_aside_reason(cc=(8, 0)), "no_aoti_packages_cc80")
        self.assertIsNone(A.card_aside_reason(cc=(9, 0)))
        self.assertEqual(A.card_aside_reason({"MODEL_OPT_TARGET_GPU": "A100"}, cc=None) if not torch.cuda.is_initialized() else "no_aoti_packages_cc80", "no_aoti_packages_cc80")
        self.assertIsNone(A.card_aside_reason({"MODEL_OPT_TARGET_GPU": "H100"}, cc=None) if not torch.cuda.is_initialized() else None)

    def test_absent_on_cc80_serves_the_eager_step_not_dynamo(self):
        hf = FakeHF(); built = []; events = []

        class W: compile_aside = None
        w = W()
        eager = lambda root, *rest: ("eager", len(rest))
        with tempfile.TemporaryDirectory() as d:
            disp = A.Dispatch(hf, d, "hoist2", lambda: built.append(1) or (lambda root, *rest: ("dynamo", 0)), events=events, aside_reason=A.ASIDE_CC80, wrapper=w, eager=eager)
            args = (torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool))
            n0 = len(A.STATS["aside"])
            self.assertEqual(disp(hf.root, *args, hf.cache), ("eager", 4)); self.assertEqual(disp(hf.root, *args, hf.cache), ("eager", 4))
            self.assertEqual(built, [])                                                                   # no Dynamo compile on this card
            self.assertEqual(len(A.STATS["aside"]) - n0, 1); self.assertEqual(A.STATS["aside_reason"], A.ASIDE_CC80)
            self.assertEqual(w.compile_aside, A.ASIDE_CC80); self.assertIsNone(hf.compiled)
            self.assertTrue(any(str(e[1]).startswith("compile_aside:no_aoti_packages_cc80") for e in events), events)
            self.assertGreaterEqual(A.report()["n_aside"], 1)

    def test_attach_without_a_directory_changes_nothing(self):
        hf = FakeHF()
        self.assertFalse(A.attach(hf, dirpath=None, levers_word="hoist2"))
        self.assertFalse(hasattr(hf, "compiled") and hf.compiled == "aoti")


if __name__ == "__main__":
    unittest.main()
