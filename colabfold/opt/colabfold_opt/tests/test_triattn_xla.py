"""The pair-biased attention binding (triattn_xla.py: the carried adapter's FLASH_OP = ONE call of the shared core provider's attention face by the
mode's tier word; the site-naming class; the lever TRIATTN_XLA = the provider's pre-compiled triangle-attention row inside that binding):
the site / family words, the op's call contract (heads-major layout, the tier word, the MSA row count, `prefer=` without the bridge lever),
the two census dicts, the provider's tier-word resolution for colabfold's call classes on both cards (the real cell table, CPU: pure Python),
the ablation words (`pallas:<row>`, `triattn_xla:<row>`, the bridge lever ablated with the binding), the superseded census, the registry /
LEVER grammar, the ROWPAIR step-aside at --n_gpu > 1. CPU only: the bridge and the provider's call face are stand-ins; GPU numbers live in
the provider's cells and CHANGES.md."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from contextlib import redirect_stderr

from colabfold_opt import ablation, manifest, modes, registry, stack, triattn_xla
from colabfold_opt.tests import _stubs

PROVIDER_ROWS_TRI = {"triattn_xla", "pallas_attn", "cd_triatt", "rowshared", "cudnn", "xla_sdpa", "xla", "fpf_core"}   # the provider's rows that serve op attn for the tri / tmpl / msarow families


class _Cfg:
    """Stand-in for the Attention module's ConfigDict (num_head, gating, optional key_dim / value_dim); shared with test_msa_col_cudnn."""

    def __init__(self, num_head=4, gating=True, **kw):
        self.num_head, self.gating = num_head, gating
        self._kw = dict(kw)

    def get(self, k, default=None):
        return self._kw.get(k, getattr(self, k, default) if k in ("num_head", "gating") else default)


class _Arr:
    """A shape/dtype-carrying stand-in array: einsum-free tests read shapes only; `bias[:, 0, 0, :] > -1e8` yields a [b, k] stand-in."""

    def __init__(self, shape, dtype="bfloat16"):
        self.shape, self.ndim, self.dtype = tuple(shape), len(shape), dtype

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 4:                              # bias[:, 0, 0, :]
            return _Arr((self.shape[0], self.shape[3]), self.dtype)
        raise IndexError(idx)

    def __gt__(self, other):
        return _Arr(self.shape, "bool")


class _Sel:
    def __init__(self, arm, cell_key, candidates):
        self.arm, self.row, self.cell_key, self.candidates = arm, arm.split("@")[0], cell_key, list(candidates)


def _serve_stub(serves="pallas_attn@s3", order=("triattn_xla", "pallas_attn@s3", "xla")):
    """A stand-in for opt_core.kernels.pallas.serve: family / resolve / attention record their arguments; attention 'serves' `serves` (COUNTS)."""
    m = types.ModuleType("stub_pallas_serve_face")
    m.COUNTS, m.calls, m.resolved = {}, [], []

    def family(op, fam=None, **shape):
        kind = shape.get("kind"); s = shape.get("n_seq")
        return f"{kind}_s{s}_h{shape.get('heads')}_d{shape.get('head_dim')}" if s else f"{kind}_h{shape.get('heads')}_d{shape.get('head_dim')}"

    def resolve(op, fam, dtype, n_tokens, *, word, direction="fwd", prefer=None, key_masked=True, **kw):
        m.resolved.append(dict(op=op, fam=fam, n=n_tokens, word=word, prefer=prefer))
        cands = [a for a in order if prefer is None or a.split("@")[0] in prefer or a == "xla"]
        return _Sel(m.serves if m.serves in cands else cands[0], f"0.5|9.0|bf16|{op}|{fam}|N<={n_tokens}|{direction}", cands)

    def attention(q, k, v, bias, key_mask=None, scale=None, *, word, kind="tri", n_seq=None, direction="fwd", layout="BSHD", prefer=None, selection=None, **kw):
        m.calls.append(dict(word=word, kind=kind, n_seq=n_seq, direction=direction, layout=layout, prefer=prefer, selection=selection))
        arm = selection.arm if selection is not None else m.serves
        m.COUNTS["served:attention:" + arm] = m.COUNTS.get("served:attention:" + arm, 0) + 1
        return ("out", arm)

    m.family, m.resolve, m.attention, m.serves = family, resolve, attention, serves
    m.report = lambda: {"counts": dict(m.COUNTS)}
    return m


class TestWords(unittest.TestCase):
    def test_site_words_from_the_haiku_module_path(self):
        M = lambda p: types.SimpleNamespace(module_name=p)  # noqa: E731
        self.assertEqual(triattn_xla.site_of(M("evoformer_iteration/triangle_attention_starting_node/attention")), "triangle_start")
        self.assertEqual(triattn_xla.site_of(M("evoformer_iteration/triangle_attention_ending_node/attention")), "triangle_end")
        self.assertEqual(triattn_xla.site_of(M("evoformer_iteration/msa_row_attention_with_pair_bias/attention")), "msa_row")
        self.assertEqual(triattn_xla.site_of(M("extra_msa_stack/triangle_attention_starting_node/attention")), "extra_triangle_start")
        self.assertEqual(triattn_xla.site_of(M("template_embedding/single_template_embedding/template_embedding_iteration/triangle_attention_ending_node/attention")), "template_triangle_end")
        self.assertEqual(triattn_xla.site_of(M("")), "other"); self.assertEqual(triattn_xla.site_of(object()), "other")

    def test_kind_words_are_the_providers_attention_kinds(self):
        self.assertEqual([triattn_xla.kind_of(s) for s in ("triangle_start", "triangle_end", "extra_triangle_start", "msa_row", "template_triangle_start", "other")],
                         ["tri", "tri", "tri", "msarow", "tmpl", "tri"])

    def test_constants(self):
        self.assertEqual((triattn_xla.NAME, triattn_xla.ROW, triattn_xla.LAYOUT, triattn_xla.PROVIDER, triattn_xla.PROVIDER_SERVE, triattn_xla.SERVE),
                         ("TRIATTN_XLA", "triattn_xla", "BHSD", "opt_core.kernels.pallas", "opt_core.kernels.pallas.serve", "opt_core.kernels.triattn_xla"))
        self.assertFalse(hasattr(triattn_xla, "FLOORS")); self.assertFalse(hasattr(triattn_xla, "ROW_PIN"))      # no kit-side size floor and no row pin: the provider's cells decide
        self.assertEqual(triattn_xla.CELL_RULE, "cell"); self.assertIn(triattn_xla.CELL_RULE, registry.STEP_ASIDE_RULES)


class TestProviderOp(unittest.TestCase):
    def setUp(self):
        triattn_xla.reset_for_tests()
        self.face = _serve_stub()
        self.saved = triattn_xla.PROVIDER_SERVE
        sys.modules[self.face.__name__] = self.face
        triattn_xla.PROVIDER_SERVE = self.face.__name__

    def tearDown(self):
        triattn_xla.PROVIDER_SERVE = self.saved
        sys.modules.pop(self.face.__name__, None)
        triattn_xla.reset_for_tests()

    def _call(self, op, site, B=128, S=800, H=4, D=32):
        q = _Arr((B, H, S, D))
        with triattn_xla.site_context(site):
            return op(q, q, q, _Arr((H, S, S)), _Arr((B, S), "bool"), 0.176)

    def test_the_op_is_one_provider_call_by_tier_word_heads_major_forward(self):
        triattn_xla._STATE["enabled"] = True                                               # the bridge lever applied: the full order of the word
        op = triattn_xla.provider_op("fast")
        self.assertEqual(op.word, "fast")
        out = self._call(op, "triangle_start")
        c = self.face.calls[-1]
        self.assertEqual((c["word"], c["kind"], c["n_seq"], c["layout"], c["direction"], c["prefer"]), ("fast", "tri", None, "BHSD", "fwd", None))
        self.assertEqual(out, ("out", "pallas_attn@s3"))
        self._call(op, "msa_row", B=128, H=8)                                              # MSA row attention: the family carries the row count of the call (the sub-batch chunk)
        c = self.face.calls[-1]
        self.assertEqual((c["kind"], c["n_seq"]), ("msarow", 128)); self.assertEqual(self.face.resolved[-1]["fam"], "msarow_s128_h8_d32")
        self._call(op, "template_triangle_end", H=4, D=16)
        self.assertEqual((self.face.calls[-1]["kind"], self.face.resolved[-1]["fam"]), ("tmpl", "tmpl_h4_d16"))
        op_b = triattn_xla.provider_op("big")                                            # --mode big binds the word big, literally
        self._call(op_b, "triangle_end"); self.assertEqual(self.face.calls[-1]["word"], "big")

    def test_without_the_bridge_lever_the_word_is_narrowed_to_the_other_rows(self):
        triattn_xla._STATE["enabled"] = False                                              # TRIATTN_XLA not applied (ablated): prefer= every provider row but the bridge's
        op = triattn_xla.provider_op("fast")
        self._call(op, "triangle_start")
        prefer = self.face.calls[-1]["prefer"]
        self.assertIsInstance(prefer, tuple); self.assertNotIn("triattn_xla", prefer); self.assertIn("pallas_attn", prefer); self.assertIn("xla", prefer)
        self.assertEqual(set(prefer), set(triattn_xla.rows_without_bridge()))

    def test_census_binding_and_bridge_lever(self):
        triattn_xla._STATE["enabled"] = True
        op = triattn_xla.provider_op("fast")
        self.face.serves = "triattn_xla"                                                   # the provider serves the bridge row: the lever's calls, its sites / shapes
        self._call(op, "triangle_start"); self._call(op, "triangle_end")
        self.face.serves = "pallas_attn@s3"                                                # the cell names another row although the bridge headed the order: refused at the call
        self._call(op, "msa_row", H=8)
        st, b = dict(triattn_xla._STATE), triattn_xla.bind_state()
        self.assertEqual((st["calls"], st["fallbacks"], st["fallback_by"]), (2, 1, "refused:1"))
        self.assertEqual((st["sites"], st["shapes"]), ("triangle_end:1,triangle_start:1", "128x800x4x32:2"))
        self.assertEqual(st["rows"], "served:2")                                           # the bridge stub carries no per-row report here: counted under 'served'
        self.assertEqual((b["provider_calls"], b["served"]), (3, "pallas_attn@s3:1,triattn_xla:2"))
        self.assertEqual(b["sites"], "msa_row:pallas_attn:1,triangle_end:triattn_xla:1,triangle_start:triattn_xla:1")
        self.assertTrue(all(k.startswith("0.5|9.0|bf16|attn|") for k in triattn_xla._BCOUNTS["cells"]))
        # the cell named another row with the bridge NOT at the head of the order: fallback_by=cell (registry.STEP_ASIDE_RULES: by design)
        triattn_xla.reset_for_tests(); triattn_xla._STATE["enabled"] = True
        face = _serve_stub(serves="cd_triatt", order=("cd_triatt", "triattn_xla", "xla")); sys.modules[self.face.__name__] = face
        self._call(triattn_xla.provider_op("fast"), "triangle_start")
        self.assertEqual((triattn_xla._STATE["calls"], triattn_xla._STATE["fallbacks"], triattn_xla._STATE["fallback_by"]), (0, 1, "cell:1"))
        self.assertEqual(manifest.stepped_aside_rule(dict(triattn_xla._STATE)), "cell")   # every call another row by the cell: by design, rc 0

    def test_bind_puts_the_op_in_the_adapters_slot_and_the_site_class_over_the_bound_class(self):
        mods = types.ModuleType("alphafold.model.modules")

        class Attention:                                                                   # whatever class is bound (the carried adapter's)
            def __init__(self, name="attention"):
                self.module_name = name

            def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
                return triattn_xla.current_site()

        mods.Attention = Attention
        kit = types.SimpleNamespace(FLASH_OP=["default_op"], _STATE={"enabled": True, "calls": 0, "fallbacks": 0})
        word = triattn_xla.bind(kit, "big", modules=mods)
        self.assertEqual(word, "big"); self.assertEqual(kit.FLASH_OP[0].word, "big"); self.assertTrue(triattn_xla.bound(mods))
        self.assertTrue(getattr(mods.Attention, triattn_xla.MARKER)); self.assertEqual(mods.Attention.__name__, "Attention")
        a = mods.Attention("evoformer/msa_row_attention_with_pair_bias/attention")
        self.assertEqual(a(None, None, None, nonbatched_bias=object()), "msa_row")         # a pair-biased call: the site named for the op below
        self.assertEqual(a(None, None, None), "other")                                     # no pair bias: passed through, no site
        self.assertEqual(triattn_xla.bind_state()["word"], "big")
        triattn_xla.bind(kit, "big", modules=mods)                                       # idempotent
        self.assertIs(mods.Attention._pallas_word_wrapped, Attention)
        triattn_xla.unbind(); self.assertIs(mods.Attention, Attention); self.assertEqual(kit.FLASH_OP[0], "default_op")


@unittest.skipUnless(__import__("importlib").util.find_spec("opt_core.kernels.pallas") is not None, "needs the shared core's provider table")
class TestProviderResolvesColabfoldsCallClasses(unittest.TestCase):
    """The tier words resolve, on the provider's own cell table, for every call class colabfold makes (bf16; triangle start/end h4 d32, the
    template pair stack h4 d16, MSA row attention h8 d32 at the sub-batch chunk's rows) at the identity canary's and the ladder's sizes on both
    cards — to a row of the triangle-attention family, never a refusal; `exact` names the stock statement; without the bridge lever no bridge row."""

    def setUp(self):
        self._env = unittest.mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop(ablation.ENV, None)                                                 # no row switched off by an earlier test's variable: the provider reads it at select()

    def tearDown(self):
        self._env.stop()

    def test_fast_big_exact_resolve_on_both_cards(self):
        import importlib
        P = importlib.import_module("opt_core.kernels.pallas")
        no_bridge = triattn_xla.rows_without_bridge()
        self.assertNotIn("triattn_xla", no_bridge); self.assertIn("pallas_attn", no_bridge)
        for cc in ("9.0", "8.0"):
            for kind, H, D, s in (("tri", 4, 32, None), ("tmpl", 4, 16, None), ("msarow", 8, 32, 128), ("msarow", 8, 32, 508)):
                fam = P.family("attn", kind=kind, heads=H, head_dim=D, n_seq=s)
                for n in (199, 400, 800, 1200):
                    for word in ("fast", "big"):
                        sel = P.select("0.5.3", cc, "bf16", "attn", fam, n, "fwd", word=word)
                        self.assertIn(P.parse_arm(sel.arm)[0], PROVIDER_ROWS_TRI, (cc, fam, n, word, sel.arm))
                        nb = P.select("0.5.3", cc, "bf16", "attn", fam, n, "fwd", word=word, prefer=no_bridge)
                        self.assertNotEqual(P.parse_arm(nb.arm)[0], "triattn_xla", (cc, fam, n, word))
                    self.assertEqual(P.select("0.5.3", cc, "bf16", "attn", fam, n, "fwd", word="exact").arm, "xla", (cc, fam, n))   # exact: the stock statement by name (no Pallas row is bitwise the stock op)

    def test_ablation_words_of_the_provider_validate_against_its_rows(self):
        FAST = list(modes.TABLE["fast"][0])
        self.assertEqual(ablation.validate("fast", ["pallas:cd_triatt"], FAST), ["pallas:cd_triatt"])          # a provider row off inside the core, AF_PALLAS_ATTN applied and kept
        self.assertEqual(ablation.validate("fast", ["triattn_xla:cuda_80"], FAST), ["triattn_xla:cuda_80"])    # a bridge row off inside the core, TRIATTN_XLA applied and kept
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("fast", ["pallas:nonesuch"], FAST)
        self.assertIn("pallas:nonesuch: not a row of ", str(cm.exception)); self.assertIn("AF_PALLAS_ATTN", str(cm.exception))
        self.assertEqual(ablation.validate("fast", ["pallas:cd_triatt", "AF_PALLAS_ATTN"], FAST), ["pallas:cd_triatt", "AF_PALLAS_ATTN", "TRIATTN_XLA"])   # another provider-bound lever kept is enough
        with self.assertRaises(ablation.AblationError) as cm:                                                 # every provider-bound lever of the mode ablated: nothing left to switch a row of
            ablation.validate("fast", ["pallas:cd_triatt", *[l for l in ablation.PROVIDER_LEVERS if l in FAST]], FAST)
        self.assertIn("is not applied by this run (mode fast, ablated)", str(cm.exception))
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("fast", ["pallas"], FAST)
        self.assertIn("pallas: names every row of ", str(cm.exception))


class TestLeverInTheKit(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests()
        self.tmp = tempfile.mkdtemp()
        self.env_saved = {k: os.environ.get(k) for k in ("COLABFOLD_OPT_KIT", "AF_PALLAS_ATTN", "COLABFOLD_OPT", "COLABFOLD_OPT_N_GPU", "MODEL_OPT_LEVERS_OFF")}
        for k in self.env_saved:
            os.environ.pop(k, None)
        os.environ["COLABFOLD_OPT_KIT"] = _stubs.make_kit_dir(self.tmp)
        self.mods, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)

    def tearDown(self):
        import shutil
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)
        stack.reset_for_tests()
        for k, v in self.env_saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(self.tmp, True)

    def test_registry_row_and_lever_line_grammar(self):
        lv = registry.LEVERS[modes.TRIATTN_LEVER]
        self.assertEqual((lv.impl, lv.origin, lv.strategy, lv.probe), ("opt_core.kernels.triattn_xla", "core", "F1.flash_triatt", ("module_state", modes.TRIATTN_LEVER_MODULE, "enabled")))
        self.assertEqual(registry.LEVERS[modes.LEVER].impl, "opt_core.kernels.pallas")                          # AF_PALLAS_ATTN: the provider by tier word
        self.assertEqual(registry.MARKERS[modes.TRIATTN_LEVER], ("class_attr", "alphafold.model.modules", "Attention", triattn_xla.MARKER))
        self.assertIn(modes.TRIATTN_LEVER, modes.TABLE["fast"][0]); self.assertIn(modes.TRIATTN_LEVER, modes.TABLE["big"][0]); self.assertNotIn(modes.TRIATTN_LEVER, modes.TABLE["exact"][0])
        self.assertEqual([modes.tier_word(m) for m in ("exact", "fast", "big")], ["exact", "fast", "big"])

    def test_fast_activation_binds_the_word_and_big_p_gt_1_drops_the_lever_by_name(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rep = stack.activate("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"], err.getvalue())
        self.assertEqual(rep.get("attn_word"), "fast"); self.assertTrue(triattn_xla._STATE["enabled"]); self.assertEqual(triattn_xla._STATE["word"], "fast")
        self.assertTrue(triattn_xla.bound()); self.assertEqual(sys.modules["af2_pallas_attn"].FLASH_OP[0].word, "fast")
        self.assertIn(modes.TRIATTN_LEVER, rep["levers_applied"])
        st = stack.lever_states()
        self.assertEqual(st[modes.LEVER]["word"], "fast"); self.assertEqual(st[modes.LEVER]["served"], "none")   # AF_PALLAS_ATTN's line carries the binding's census after the adapter's own counters
        self.assertNotIn("floors", st[modes.TRIATTN_LEVER])

    def test_big_binds_big(self):
        if not all(__import__("importlib").util.find_spec(n) is not None for n in ("opt_core.mem.ngpu", "opt_core.mem.rowpair_jax")):
            self.skipTest("needs the axis' core modules")
        err = io.StringIO()
        with redirect_stderr(err):
            rep = stack.activate("big", queries=_stubs.queries(600), n_gpu=1)
        self.assertTrue(rep["active"], err.getvalue()); self.assertEqual(rep.get("attn_word"), "big"); self.assertEqual(triattn_xla.bind_state()["word"], "big")


class TestSupersededCensus(unittest.TestCase):
    """modes.SUPERSEDES / manifest.superseded_by: a lever whose kernel no call reached because the lever bound over its class served them ends BY
    NAME (`superseded_by=<lever>:<calls>`, rc 0), never as a partial activation; without a superseding call served it stays partial (rc 3)."""

    def test_table(self):
        self.assertEqual(tuple(modes.SUPERSEDES[modes.TRIATTN_LEVER]), (modes.LEVER,))

    def _man(self, kit_calls, kit_fallbacks, tx_calls, tx_enabled=True):
        rep = {"active": True, "mode": "fast", "levers_applied": ["DEVICE_RESIDENT", modes.LEVER, modes.TRIATTN_LEVER]}
        return {"activation_report": rep,
                "kit_state_exit": {"enabled": True, "calls": kit_calls, "fallbacks": kit_fallbacks},
                "lever_states_exit": {modes.LEVER: {"enabled": True, "calls": kit_calls, "fallbacks": kit_fallbacks},
                                      modes.TRIATTN_LEVER: {"enabled": tx_enabled, "calls": tx_calls, "fallbacks": 0, "fallback_by": "none"}}}

    def test_nothing_reached_the_flash_kernel_and_triattn_served_is_by_design_rc_0(self):
        man = self._man(0, 8, 36)
        self.assertEqual(manifest.superseded_by(man, modes.LEVER), f"{modes.TRIATTN_LEVER}:36")
        self.assertNotIn(modes.LEVER, manifest.partial_levers(man))

    def test_calls_0_without_a_superseding_call_served_stays_partial_rc_3(self):
        for man in (self._man(0, 8, 0), self._man(0, 0, 0)):
            self.assertIsNone(manifest.superseded_by(man, modes.LEVER))
            self.assertIn(modes.LEVER, manifest.partial_levers(man))


if __name__ == "__main__":
    unittest.main()
