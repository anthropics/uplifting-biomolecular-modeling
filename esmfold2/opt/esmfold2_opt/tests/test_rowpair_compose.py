"""How the ``n_gpu > 1`` row-sharded line (esmfold2_opt.rowpair*) meets the single-GPU levers of the H100 sets: the route declarations
(``rowpair.NOT_FOR_ROUTE`` / ``EAGER_UNDER_SHARDING``, ``tp.not_for_route``) are registry lever names with reason tokens; the sampler binding
keeps ef2_dit's roll-out (lever ro) EAGER through ef2_opt's sampler-site graph budget and restores it on uninstall; the MSA row form issues
the composing kernels itself (m15 = ef2_msa_v2.mtr_fused per token block IN PLACE on the replicated m; t15msa = ef2_pair_v2's fused
PairTransition + residual per local row block IN PLACE) — tested with EXACT stand-in modules registered under the lever modules' names, so
the plumbing (aliasing, blocking, counters, fall-through) is checked bit for bit against the plain row statements; the real kernels'
numerics are their own tracks' evidence."""
import sys
import types
import unittest
from unittest import mock

try:
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
except ImportError as _e:                                                   # an older / absent core: named skip
    raise unittest.SkipTest(f"producer_missing:opt_core.mem.rowpair ({_e})")
from esmfold2_opt import registry, rowpair, tp
from esmfold2_opt import rowpair_msa as RM


def _torch_or_skip():
    try:
        import torch
    except (ImportError, RuntimeError) as e:
        raise unittest.SkipTest(f"torch missing ({e})")
    torch.set_num_threads(1)
    return torch


def _registry_names():
    names = set()
    for row in getattr(registry, "REGISTRY", ()):
        n = row.get("name") if isinstance(row, dict) else getattr(row, "name", None)
        if n:
            names.add(n)
    if not names and hasattr(registry, "names"):
        names = set(registry.names())
    return names


class RouteDeclarations(unittest.TestCase):
    def test_not_for_route_names_are_registry_levers_with_reason_tokens(self):
        names = _registry_names()
        if not names:
            self.skipTest("registry has no name list in this tree")
        self.assertTrue(rowpair.NOT_FOR_ROUTE)
        for lever, reason in rowpair.NOT_FOR_ROUTE.items():
            self.assertIn(lever, names, lever)
            self.assertTrue(reason and " " not in reason, f"{lever}: the reason is ONE token (it is a LEVER-line field value): {reason!r}")
        for lever in rowpair.EAGER_UNDER_SHARDING:
            self.assertIn(lever, names, lever)
        self.assertFalse(set(rowpair.NOT_FOR_ROUTE) & set(rowpair.EAGER_UNDER_SHARDING), "a lever is either off the route or eager on it, not both")
        for lever in ("ax", "af", "fz", "t15", "t15msa", "m15", "kd", "t9", "t10", "x4"):   # the levers that ENGAGE under sharding are in neither table
            self.assertNotIn(lever, rowpair.NOT_FOR_ROUTE)

    def test_tp_accessors(self):
        self.assertEqual(tp.not_for_route(1), {})
        self.assertEqual(tp.not_for_route(2), rowpair.NOT_FOR_ROUTE)
        self.assertEqual(tp.not_for_route(8), rowpair.not_for_route(8))
        self.assertIsNot(tp.not_for_route(2), rowpair.NOT_FOR_ROUTE)        # a copy: callers may edit it
        self.assertEqual(tp.eager_under_sharding(1), {})
        self.assertEqual(set(tp.eager_under_sharding(4)), set(rowpair.EAGER_UNDER_SHARDING))
        self.assertEqual(rowpair.SAMPLER_SITE_BUDGET_SHARDED, 1)


# ---------------------------------------------------------------------------------------------------------------- stand-in lever modules
def _fake_msa_v2(torch, on=("m15",), ok=True):
    """A stand-in `ef2_msa_v2`: levers_on() and an EXACT mtr_fused (m + tr(m) written into `out`, which may alias m) with the module's counters."""
    mod = types.ModuleType("ef2_msa_v2")
    mod.STATS = __import__("collections").Counter()
    mod.levers_on = lambda: list(on)
    mod._mtr_ok = lambda tr, m: bool(ok)

    def mtr_fused(tr, m, out=None):
        y = m + tr(m)
        if out is None:
            out = torch.empty_like(m)
        out.copy_(y)
        mod.STATS["m15_calls"] += 1
        mod.shapes = getattr(mod, "shapes", []) + [tuple(m.shape)]
        return out
    mod.mtr_fused = mtr_fused
    return mod


def _fake_pair_v2(torch, on=True):
    """A stand-in `ef2_pair_v2`: levers_on() and an EXACT _pair_transition_residual_v2 (pair + pt(pair), out of place) with a call log."""
    mod = types.ModuleType("ef2_pair_v2")
    mod.STATS = __import__("collections").Counter()
    mod.levers_on = lambda: {"t15": bool(on), "t15msa": bool(on), "t6s": False, "t6i": False}

    def _pair_transition_residual_v2(pt, pair):
        mod.STATS["t15_msa_pair_transition_calls"] += 1
        mod.shapes = getattr(mod, "shapes", []) + [tuple(pair.shape)]
        return pair + pt(pair)
    mod._pair_transition_residual_v2 = _pair_transition_residual_v2
    return mod


class _Tr(object):
    """a per-(token, row) transition: any elementwise map serves the blocking argument"""

    def __call__(self, m):
        return 0.5 * m * m - 0.25 * m


class ComposingKernels(unittest.TestCase):
    def setUp(self):
        self.torch = _torch_or_skip()
        RM.STATS.clear()

    def test_kit_msa_kernels_reads_the_modules_lever_records(self):
        torch = self.torch
        with mock.patch.dict(sys.modules, {"ef2_msa_v2": _fake_msa_v2(torch), "ef2_pair_v2": _fake_pair_v2(torch)}):
            k = RM.kit_msa_kernels()
            self.assertIs(k["m15"], sys.modules["ef2_msa_v2"])
            self.assertIs(k["t15msa"], sys.modules["ef2_pair_v2"])
            self.assertEqual(RM.kit_kernel_words(k), {"msa_m15": "token_blocks_inplace", "msa_t15msa": "row_blocks_inplace"})
            self.assertEqual(RM.census_facts()["msa_m15"], "token_blocks_inplace")
        with mock.patch.dict(sys.modules, {"ef2_msa_v2": _fake_msa_v2(torch, on=("m16", "m17")), "ef2_pair_v2": _fake_pair_v2(torch, on=False)}):
            k = RM.kit_msa_kernels()                                        # installed modules with these levers OFF: nothing composes (m16 / m17 are rowpair.NOT_FOR_ROUTE's)
            self.assertEqual((k["m15"], k["t15msa"]), (None, None))
            self.assertEqual(RM.kit_kernel_words(k), {"msa_m15": "off", "msa_t15msa": "off"})
        sys.modules.pop("ef2_msa_v2", None); sys.modules.pop("ef2_pair_v2", None)
        self.assertEqual(RM.kit_msa_kernels(), {"m15": None, "t15msa": None})   # modules absent (the Fast model, or a set without them)

    def test_m15_on_the_replicated_m_per_token_block_in_place_is_the_plain_statement(self):
        torch = self.torch
        lay = Layout.checked(600, 2, 0, 8)
        tr = _Tr()
        m = torch.randn(1, 600, 5, 8)
        ref = m + tr(m)                                                       # the plain replicated statement (msa_transition_replicated + residual)
        v2 = _fake_msa_v2(torch)
        mm = m.clone()
        plain = mock.patch.object(RM, "msa_transition_replicated", lambda tr_, m_, lay_: tr_(m_))   # the core's replicated statement needs a process group; its value is tr(m)
        plain.start(); self.addCleanup(plain.stop)
        out = RM.msa_transition_residual_replicated_(tr, mm, lay, {"m15": v2, "t15msa": None})
        self.assertIs(out, mm)                                                # in place over m: no [N, M, d] copy
        self.assertTrue(torch.equal(out, ref))
        nblk = -(-600 // RM.MSA_TRANSITION_TOKENS)
        self.assertEqual(v2.STATS["m15_calls"], nblk)                        # one kernel issue per token block (600 tokens / 256 = 3)
        self.assertEqual(RM.STATS["m15_row_calls"], nblk)
        self.assertEqual({s[1] for s in v2.shapes}, {RM.MSA_TRANSITION_TOKENS, 600 - 2 * RM.MSA_TRANSITION_TOKENS})
        # the kernel declines (its own _mtr_ok) -> the replicated stock statement, counted, same values
        v2no = _fake_msa_v2(torch, ok=False)
        out2 = RM.msa_transition_residual_replicated_(tr, m.clone(), lay, {"m15": v2no, "t15msa": None})
        self.assertTrue(torch.equal(out2, ref))
        self.assertEqual((v2no.STATS["m15_calls"], RM.STATS["m15_row_fallthrough"]), (0, 1))
        # no m15: the engine's own transition + residual
        out3 = RM.msa_transition_residual_replicated_(tr, m.clone(), lay, None)
        self.assertTrue(torch.equal(out3, ref))

    def test_t15msa_per_local_row_block_in_place_is_the_plain_statement(self):
        torch = self.torch
        lay = Layout.checked(96, 2, 1, 16)                                    # rank 1 of 2: rows 48:96
        pt = _Tr()
        pair = torch.randn(1, lay.R, lay.N, 8)
        ref = pair + pt(pair)
        p2 = _fake_pair_v2(torch)
        prev = (rowpair.CTX.active, rowpair.CTX.layout)
        rowpair.CTX.active, rowpair.CTX.layout = True, lay
        try:
            z = pair.clone()
            out = RM.pair_transition_fused_rows_(p2, pt, z, 20)
            self.assertIs(out, z)                                             # written back in place per row block
            self.assertTrue(torch.equal(out, ref))
            nblk = -(-lay.R // 20)
            self.assertEqual(p2.STATS["t15_msa_pair_transition_calls"], nblk)
            self.assertEqual({s[:3] for s in p2.shapes}, {(1, 20, lay.N), (1, lay.R - 20 * (nblk - 1), lay.N)})   # [1, rows, N, C] blocks, the tail shorter
            self.assertEqual(RM.STATS["t15msa_row_calls"], 1)
            with self.assertRaises(RowpairRefused):                          # a whole pair (not this rank's shard) is refused by name, never transformed
                RM.pair_transition_fused_rows_(p2, pt, torch.randn(1, lay.N, lay.N, 8), 20)
        finally:
            rowpair.CTX.active, rowpair.CTX.layout = prev

    def test_pair_ops_take_the_kernel_outside_and_inside_the_sharded_context(self):
        torch = self.torch
        p2 = _fake_pair_v2(torch)

        class _PT(object):
            _chunk_size = None

            def __call__(self, pair):
                return 0.1 * pair

        class _Blk(object):
            def __init__(self):
                self.pair_transition = _PT()

            def tri_mul_out(self, pair, mask=None):
                return 0.0 * pair

            def tri_mul_in(self, pair, mask=None):
                return 0.0 * pair
        blk = _Blk()
        x = torch.randn(1, 12, 12, 4)
        prev = (rowpair.CTX.active, rowpair.CTX.layout)
        rowpair.CTX.active, rowpair.CTX.layout = False, None
        try:
            got = RM.pair_ops(blk, 0, {"m15": None, "t15msa": p2})(x)          # outside a sharded fold: the block's statements, the transition through the module's fused residual
            self.assertTrue(torch.equal(got, x + 0.1 * x))
            self.assertEqual(p2.STATS["t15_msa_pair_transition_calls"], 1)
            got0 = RM.pair_ops(blk, 0, None)(x)                                # no kernels: the stock three statements
            self.assertTrue(torch.equal(got0, x + 0.1 * x))
        finally:
            rowpair.CTX.active, rowpair.CTX.layout = prev


# ---------------------------------------------------------------------------------------------------------------- the sampler binding under sharding
class SamplerBindingUnderSharding(unittest.TestCase):
    """rowpair.install keeps ef2_dit's roll-out as the sampler (lever ro), sets ef2_opt's sampler-site graph budget to 1 token so the roll-out
    and the graphed sampler take their EAGER forms by the kit's one capture gate, names it, and puts the budget back on uninstall."""

    def setUp(self):
        self.torch = _torch_or_skip()

    def _model_and_modules(self, with_dit: bool):
        torch = self.torch
        try:
            from transformers.models.esmfold2 import modeling_esmfold2_common as CMN  # noqa: F401 — install reads the common module's classes
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"transformers fork (stock/PINS.json) not importable ({type(e).__name__}: {e})")
        eo = types.ModuleType("ef2_opt")

        class _CFG(object):
            graph_budget_tokens_sampler = 0
            loop_static = False
            recycle_graph = False
        eo.CFG = _CFG()
        dit = types.ModuleType("ef2_dit")

        def _sample_dit(self, *a, **k):
            return "dit"
        dit._sample_dit = _sample_dit
        dit._dm_forward_dit = lambda self, *a, **k: None
        dit.levers_on = lambda: {"ro": True, "kd": True, "dit": False}
        sh = torch.nn.Module()
        sh.diffusion_module = torch.nn.Module()

        def eager(*a, **k):
            return "eager"
        sh._ef2opt_eager_sample = eager
        if with_dit:
            sh.sample = types.MethodType(_sample_dit, sh)
        else:
            sh.sample = lambda *a, **k: "graphed"
        model = torch.nn.Module()
        model.structure_head = sh
        model.parcae_input_norm = torch.nn.LayerNorm(rowpair.D_PAIR)             # install reads the model's d_pair (and the trunk backend, patched below) BEFORE the sampler binding —
        model.eval()                                                             # its by-name refusals come before any rebinding; a stub without it never reached the binding
        return model, eo, dit

    def _install_p2(self, model, modules):
        from opt_core.mem.rowpair import dist as RD
        rowpair._modules()                                                       # the fork's model + common modules and everything THEY import (transformer_engine -> onnx ...) loaded
        CMN = sys.modules[rowpair.COMMON_MODULE]                                 # BEFORE patch.dict snapshots sys.modules: on exit it drops every module first imported inside the
                                                                                 # block, and the next test's re-import of onnx's protobuf extension aborts the interpreter (SIGABRT)
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(RD, "is_dist", lambda: True), mock.patch.object(RD, "world", lambda: (2, 0)), \
                mock.patch.object(rowpair, "_guard_weights_replicated", lambda model: None), \
                mock.patch.object(rowpair, "_trunk_backend", lambda model: getattr(CMN, "BACKEND_REFERENCE", None)):   # the reference backend: no `_kernel_backend` word on the blocks (the fork names only BACKEND_FUSED / BACKEND_CUEQ)
            try:
                return rowpair.install(model, 2)
            except Exception as e:  # noqa: BLE001 — the stub model has no trunks / loop: what install does PAST the sampler binding is not under test here
                return {"refused": f"{type(e).__name__}: {e}"}

    def test_dit_rollout_stays_the_sampler_eager_by_budget_and_uninstall_restores(self):
        model, eo, dit = self._model_and_modules(with_dit=True)
        sh = model.structure_head
        bound_before = vars(sh)["sample"]
        try:
            self._install_p2(model, {"ef2_opt": eo, "ef2_dit": dit})
            self.assertEqual(eo.CFG.graph_budget_tokens_sampler, rowpair.SAMPLER_SITE_BUDGET_SHARDED)   # the capture gate's budget on this rank: every sampler shape eager
            self.assertIs(vars(sh)["sample"], bound_before)                   # the roll-out is still the sampler (not rebound to the eager chain)
            self.assertEqual(rowpair.STATE.get("sampler"), "ef2_dit.rollout:eager")
            self.assertEqual(rowpair.STATE.get("sampler_budget_prev"), 0)
        finally:
            with mock.patch.dict(sys.modules, {"ef2_opt": eo, "ef2_dit": dit}):
                rowpair.uninstall()
        self.assertEqual(eo.CFG.graph_budget_tokens_sampler, 0)              # ef2_opt's budget as it was
        self.assertNotIn("sampler_budget_prev", rowpair.STATE)

    def test_without_a_rollout_the_eager_sampler_serves_as_before(self):
        model, eo, dit = self._model_and_modules(with_dit=False)
        sh = model.structure_head
        try:
            self._install_p2(model, {"ef2_opt": eo})
            self.assertEqual(rowpair.STATE.get("sampler"), "ef2_opt.eager")
            self.assertEqual(sh.sample(), "eager")                            # rebound to ef2_opt's eager sampler, restorable
            self.assertIn("model.structure_head.sample", rowpair.PATCHES.names())
        finally:
            rowpair.uninstall()
        self.assertEqual(sh.sample(), "graphed")


if __name__ == "__main__":
    unittest.main()
