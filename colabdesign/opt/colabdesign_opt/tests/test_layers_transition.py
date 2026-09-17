"""Lever `transition` (kernels/layers_transition.py): the ADAPTER that binds `modules.Transition` to the shared core's JAX-family provider row by the
tier word (kernels/provider.py BINDINGS['transition']).  What is tested here is the adapter's contract, never the provider's current cell winner by
name: the binding names op `transition` under the tier word with rows the adapter's two call forms cover; every row it names is a tolerance-class
row WITH a backward (the design step differentiates through it); a cell that names the stock statement comes back as the declared per-call word
`cell_xla`; the module carries no kernel source and no size / card / tile table; the LEVER line grammar; the evidence census reads the line.  GPU
images add the numerics check of the bound row against stock's module math (skipped by name elsewhere).
`python test_layers_transition.py` runs the same functions without pytest (an image that lacks it).
"""
import os
import re
import sys
import unittest

import numpy as np

try:
    import pytest
except ImportError:
    try:
        from colabdesign_opt.tests._noptest import pytest
    except ImportError:
        import importlib as _il, types as _ty
        pytest = _ty.SimpleNamespace(importorskip=_il.import_module, mark=_ty.SimpleNamespace(parametrize=lambda names, values: (lambda fn: fn)))

from colabdesign_opt.kernels import provider as P

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(os.path.dirname(HERE), "kernels", "layers_transition.py")


def _skip(why):
    sk = getattr(pytest, "skip", None)
    if sk is not None:
        sk(why)
    raise unittest.SkipTest(why)


# ─────────────────────────────────────────── CPU: the binding and the adapter's contract ───────────────────────────────────────────
def test_binding_is_the_tier_word_over_backward_capable_tolerance_rows():
    core = P.provider()
    b = P.BINDINGS["transition"]
    assert b.op == "transition" and b.word == P.TIER_WORD and b.word in core.TIER_WORDS, (b.op, b.word)
    assert b.direction == "fwdbwd" and b.dtype == "bf16"
    assert core.family(b.op, **b.main) == "af2_relu_c128_x4", core.family(b.op, **b.main)          # the model's main cell: the Evoformer pair transition
    rows = core.TABLE["rows"]
    for row, path in b.rows.items():
        assert row in rows and any(str(o).startswith("transition") for o in (rows[row].get("ops") or [])), (row, rows[row].get("ops"))
        assert rows[row].get("backward") and str(rows[row]["backward"]).lower() not in ("none", "no", "false"), (row, rows[row].get("backward"))    # differentiated calls: a row without a backward is never bound
        assert str(rows[row].get("cls", "")) not in ("exact", "stock"), row                                                                          # class precision (registry): a tolerance-class kernel, never claimed bitwise
        assert path.startswith("opt_core.kernels.") and P.row_file("transition", row) is not None, path
    assert re.match(r"^[a-z_]+@[0-9a-f]{8}$", P.kernel_word("transition")), P.kernel_word("transition")


def test_cells_naming_the_stock_statement_come_back_as_the_declared_cell_word():
    b = P.BINDINGS["transition"]
    try:
        for cc in ("9.0", "8.0"):
            b.reset(); b.stack = ("0.6", cc)
            for n in (112, 195, 431, 500, 800, 3000):
                for shape in (b.main, dict(form="af2", activation="relu", c=64, factor=2), dict(form="af2", activation="relu", c=256, factor=4), dict(form="af2", activation="relu", c=64, factor=4)):
                    arm, cfg = P._cell("transition", n, "bf16", None, 1, **shape)
                    row = arm.split("@")[0].split(":")[0]
                    assert row in b.rows or row == "xla", (cc, n, shape, arm)                       # a tier word never hands the adapter a row it cannot bind except the stock statement, by name
                    assert isinstance(cfg, dict)
                    if row == "xla":
                        assert P.fallback_word("transition", n, "bf16", **shape) == P.CELL_FALLBACK + "xla"
            f = P.facts("transition")
            for k in ("provider", "word", "row", "tier"):
                assert k in f and not re.search(r"\s", str(f[k])), f
            assert f["word"] == P.TIER_WORD
    finally:
        b.reset()


def test_the_adapter_carries_no_kernel_and_no_size_card_or_tile_table():
    src = open(ADAPTER).read()
    for token in ("pallas_call", "plgpu", "_TILES", "SERVED_CC", "MIN_TOKENS", "below_size_rule", "cc80_speed", "custom_vjp", "BlockSpec"):
        assert token not in src, token
    from colabdesign_opt import evidence as E, registry
    from colabdesign_opt.kernels import layers_transition as T
    assert T.NAME == "transition" and T.ORIGIN == "core" and T.NUMERICS == "precision" == registry.LEVERS["transition"].numerics
    assert tuple(T.EXPECTED_FALLBACKS) == E.TRANSITION_DECLARED_FALLBACKS == ("dtype_not_bf16", "channels", "intermediate_width", "rank")
    assert T.NAME in E.CENSUS_LEVERS and list(E.CENSUS_LEVERS).index(T.NAME) == list(E.CENSUS_LEVERS).index("proj") + 1 and T.NAME not in E.GATES
    assert T.FORMS == ("fused", "lnkeep") and T.SERVED_C == (64, 128, 256)
    off = T.off_line("ablated")
    assert off.startswith(f"[colabdesign-opt] LEVER name=transition state=off reason=ablated impl={P.kernel_word('transition')} origin=core"), off
    assert T.lever_line().startswith("[colabdesign-opt] LEVER name=transition state=off reason=not_installed "), T.lever_line()

    class _Fused:  # noqa: D101 - a row module of the LN-in-kernel form
        @staticmethod
        def transition(*a): return None
    class _Keep:   # noqa: D101,E306 - a row module of the LN-kept form
        @staticmethod
        def mlp_2d(*a): return None
    class _Odd:    # noqa: D101,E306
        pass
    assert T.form_of(_Fused) == "fused" and T.form_of(_Keep) == "lnkeep"
    with pytest.raises(T.Refusal) if hasattr(pytest, "raises") else _Raises(T.Refusal):
        T.form_of(_Odd)


class _Raises:                                   # a minimal pytest.raises for the no-pytest image
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        if et is None: raise AssertionError(f"{self.exc.__name__} not raised")
        return issubclass(et, self.exc)


# ─────────────────────────────────────────── CPU: the evidence census reads the adapter's line ───────────────────────────────────────────
class TestEvidenceCensusReadsTransition(unittest.TestCase):
    """the kit's evidence census (evidence.py) classifies lever transition from its ONE line: applied (served, declared step-asides only) /
    skipped (every call a declared step-aside: a card or size whose cells all name the stock statement) / fallback (an undeclared word) / missing."""

    ON = ("[colabdesign-opt] LEVER name=transition state=on impl=layers_transition@0a1b2c3d origin=core numerics=precision precision=bf16 variant=fused served=5 fallback=0 "
          "fallback_by=none shapes=C128:2,C256:1,C64:2 provider=opt_core.kernels.pallas@0.5.116.0 word=fast row=cd_transition:5 tier=cd_transition:5 source=exit pid=7")
    SOME_XLA = ON.replace("served=5 fallback=0 fallback_by=none", "served=3 fallback=2 fallback_by=cell_xla:2").replace("row=cd_transition:5 tier=cd_transition:5", "row=cd_transition:3,xla:2 tier=cd_transition:3,xla:2 uncovered=2:af2_relu_c256_x4/fwdbwd,af2_relu_c64_x4/fwdbwd")
    ALL_XLA = ON.replace("served=5 fallback=0 fallback_by=none", "served=0 fallback=5 fallback_by=cell_xla:5").replace("shapes=C128:2,C256:1,C64:2", "shapes=none")
    STRUCT = ON.replace("fallback=0 fallback_by=none", "fallback=1 fallback_by=channels:1")
    ODD = ON.replace("fallback=0 fallback_by=none", "fallback=1 fallback_by=platform:1")          # an ENVIRONMENT word (undeclared, fail-closed)
    ASIDE = "[colabdesign-opt] LEVER name=transition state=skipped reason=cannot_run impl=layers_transition@0a1b2c3d origin=core numerics=precision detail=Refusal:lever_transition:_word_'fast'_names_row_'xla' source=install pid=7"

    def _classify(self, line):
        from colabdesign_opt import evidence as E, modes
        res = modes.resolve("fast")
        return E.classify([l for l in (line,) if l], res.levers), res

    def test_states(self):
        from colabdesign_opt import evidence as E, report
        ev, _ = self._classify(self.ON)
        self.assertEqual(ev["state"]["transition"], "applied"); self.assertEqual((ev["transition_served"], ev["transition_fallback"], ev["transition_fallback_by"]), (5, 0, {}))
        self.assertIn(" proj_fallback_by=n/a transition_served=5 transition_fallback_by=none txla_served=", report.evidence_line(ev))   # the census pair, after proj's, before txla's
        self.assertNotIn("transition", E.verdict(ev)["partial"])
        ev, _ = self._classify(self.SOME_XLA)                                           # some cells name the stock statement: applied, the cell word declared by its prefix
        self.assertEqual(ev["state"]["transition"], "applied"); self.assertEqual(ev["transition_fallback_by"], {"cell_xla": 2}); self.assertNotIn("transition", E.verdict(ev)["partial"])
        ev, _ = self._classify(self.STRUCT)                                             # a structural word: applied
        self.assertEqual(ev["state"]["transition"], "applied")
        ev, _ = self._classify(self.ALL_XLA)                                            # every call the stock statement by its cell: stepped aside by name (exit 0), never partial
        self.assertEqual(ev["state"]["transition"], "skipped"); self.assertNotIn("transition", E.verdict(ev)["partial"])
        self.assertIn(" transition_served=0 transition_fallback_by=cell_xla:5 ", report.evidence_line(ev))
        ev, _ = self._classify(self.ASIDE)                                              # the install-time refusal by name (a card whose main cell names the stock statement)
        self.assertEqual(ev["state"]["transition"], "skipped"); self.assertNotIn("transition", E.verdict(ev)["partial"])
        ev, _ = self._classify(self.ODD)                                                # an undeclared fallback word: partial (fail-closed)
        self.assertEqual(ev["state"]["transition"], "fallback"); self.assertIn("transition", E.verdict(ev)["partial"])
        ev, _ = self._classify("")                                                      # no line: missing (partial)
        self.assertEqual(ev["state"]["transition"], "missing"); self.assertIn("transition", E.verdict(ev)["partial"])

    def test_fixture_is_the_levers_grammar(self):
        """test_levers' TRANSITION_OK / TRANSITION_SMALL (the fake arms' lines) parse to `applied`."""
        try:
            from .test_levers import TRANSITION_OK, TRANSITION_SMALL
        except ImportError:                                                             # run as a script: no parent package
            from colabdesign_opt.tests.test_levers import TRANSITION_OK, TRANSITION_SMALL
        for line in (TRANSITION_OK, TRANSITION_SMALL):
            ev, _ = self._classify(line); self.assertEqual(ev["state"]["transition"], "applied", line)
            self.assertRegex(line, r" provider=\S+ word=fast row=\S+ tier=\S+ source=exit$")
        self.assertIn(" fallback_by=cell_xla:2 ", TRANSITION_SMALL)


# ─────────────────────────────────────────── GPU: the bound row against stock's module math ───────────────────────────────────────────
CASES = [(333, 64), (1000, 128), (517, 256)]


def _stack():
    g = globals()
    if g.get("_STACK_READY"):
        return
    try:
        import jax as _jax
    except ImportError:
        _skip("no jax")
    if not hasattr(_jax, "jit") or not hasattr(_jax, "numpy"):
        _skip("stand-in jax")
    try:
        if _jax.default_backend() != "gpu":
            _skip("no gpu backend")
    except Exception as e:      # noqa: BLE001
        _skip(f"backend: {e!r}")
    try:
        import haiku as _hk
        from colabdesign.af.alphafold.model import modules as _mods
        from colabdesign_opt.kernels import layers_transition as _T
    except ImportError as e:
        _skip(f"real stack not importable: {e}")
    g.update(jax=_jax, jnp=_jax.numpy, hk=_hk, modules=_mods, LT=_T)
    g["_STACK_READY"] = True


def _case(M, C, seed=0):
    rng = np.random.RandomState(seed)
    H = 4 * C
    x = rng.standard_normal((M, C)).astype(np.float32) * 1.5 + 0.2
    return dict(x=x, scale=(1.0 + 0.1 * rng.standard_normal(C)).astype(np.float32), offset=(0.05 * rng.standard_normal(C)).astype(np.float32),
                w1=(rng.standard_normal((C, H)) * (2.0 / C) ** 0.5).astype(np.float32), b1=(0.02 * rng.standard_normal(H)).astype(np.float32),
                w2=(rng.standard_normal((H, C)) * (1.0 / H) ** 0.5).astype(np.float32), b2=(0.02 * rng.standard_normal(C)).astype(np.float32),
                dy=rng.standard_normal((M, C)).astype(np.float32))


@pytest.mark.parametrize("M,C", CASES)
def test_bound_row_matches_stock_within_stock_error(M, C):
    """|row - fp32 reference| <= 1.25 x |stock bf16 path - fp32 reference| + 1e-3 (max-abs), forward and d(act): the class-precision pass rule."""
    _stack()
    try:
        K = P.admit("transition")
    except P.ProviderRefusal as e:
        _skip(f"the tier word names no bindable row here: {e}")
    form = LT.form_of(K)
    c = _case(M, C)
    bf = lambda a: jnp.asarray(a, jnp.bfloat16)           # noqa: E731
    x, w1, b1, w2, b2, dy = (bf(c[k]) for k in ("x", "w1", "b1", "w2", "b2", "dy"))
    scale, offset = jnp.asarray(c["scale"]), jnp.asarray(c["offset"])
    ref32 = lambda xx: LT.reference_transition(xx.astype(jnp.float32), scale, offset, w1.astype(jnp.float32), b1.astype(jnp.float32), w2.astype(jnp.float32), b2.astype(jnp.float32))   # noqa: E731
    stock = lambda xx: LT.reference_transition(xx, scale, offset, w1, b1, w2, b2)      # noqa: E731
    if form == "fused":
        LT._sm80_tiles(K)
        row = lambda xx: K.transition(xx, scale, offset, w1, b1, w2, b2, LT.LN_EPS)     # noqa: E731
    else:
        row = lambda xx: K.mlp_2d(LT.reference_layer_norm(xx, scale, offset), w1, b1, w2, b2, LT.LN_EPS)   # noqa: E731
    y_ref = np.asarray(ref32(x), np.float32); y_stock = np.asarray(stock(x).astype(jnp.float32)); y_row = np.asarray(row(x).astype(jnp.float32))
    e_stock = float(np.max(np.abs(y_stock - y_ref))); e_row = float(np.max(np.abs(y_row - y_ref)))
    assert e_row <= 1.25 * e_stock + 1e-3, (M, C, form, e_row, e_stock)
    g_ref = np.asarray(jax.vjp(ref32, x.astype(jnp.float32))[1](dy.astype(jnp.float32))[0], np.float32)
    g_stock = np.asarray(jax.vjp(stock, x)[1](dy)[0].astype(jnp.float32)); g_row = np.asarray(jax.vjp(row, x)[1](dy)[0].astype(jnp.float32))
    ge_stock = float(np.max(np.abs(g_stock - g_ref))); ge_row = float(np.max(np.abs(g_row - g_ref)))
    assert ge_row <= 1.25 * ge_stock + 1e-3, (M, C, form, ge_row, ge_stock)


def test_module_under_haiku_serves_by_the_row_and_keeps_stocks_parameter_tree():
    _stack()
    from colabdesign_opt.kernels import layers_transition as T
    try:
        info = T.install()
    except T.Refusal as e:
        _skip(f"stepped aside by name here: {e}")
    try:
        assert info["word"] == P.TIER_WORD and info["variant"] in T.FORMS and getattr(modules.Transition, T.MARKER, False)
        from colabdesign.af.alphafold.model import config
        cfg = config.model_config("model_1_multimer_v3").model
        c, gc = cfg.embeddings_and_evoformer.evoformer.pair_transition, cfg.global_config
        def f(act, mask):
            return modules.Transition(c, gc, name="pair_transition")(act, mask)
        fwd = hk.transform(f)
        rng = jax.random.PRNGKey(0)
        act = jax.random.normal(rng, (512, 8, 128), jnp.bfloat16); mask = jnp.ones((512, 8), jnp.bfloat16)
        params = fwd.init(rng, act, mask)
        assert set(params) == {"pair_transition/input_layer_norm", "pair_transition/transition1", "pair_transition/transition2"}, sorted(params)
        T.reset_census()
        y = fwd.apply(params, rng, act, mask); jax.block_until_ready(y)
        cen = T.census()
        assert y.shape == act.shape and (cen["served"] == 1 or cen["fallback_by"].get("cell_xla") == 1), cen      # served by the row, or the stock class by the cell's name — never silent
        S = T.stock_class()
        y0 = hk.transform(lambda a, m: S(c, gc, name="pair_transition")(a, m)).apply(params, rng, act, mask)
        err = float(jnp.max(jnp.abs(y.astype(jnp.float32) - y0.astype(jnp.float32))))
        assert err <= 0.05, err
        line = T.lever_line()
        assert re.match(r"^\[colabdesign-opt\] LEVER name=transition state=on impl=\S+@[0-9a-f]{8} origin=core numerics=precision precision=bf16 variant=(fused|lnkeep) served=\d+ fallback=\d+ fallback_by=\S+ shapes=\S+ admit=(main|cells) provider=\S+ word=fast row=\S+ tier=\S+(?: uncovered=\S+)? source=exit$", line), line
    finally:
        T.uninstall(); T.reset_census()
    assert not getattr(modules.Transition, T.MARKER, False)


def _main():
    """`python test_layers_transition.py` on an image without pytest: every test function and the census TestCase, one summary line."""
    fails = 0
    def run(name, fn, *a):
        nonlocal fails
        try:
            fn(*a); print(f"ok   {name}{a if a else ''}")
        except unittest.SkipTest as e:
            print(f"skip {name}{a if a else ''}: {e}")
        except BaseException as e:  # noqa: BLE001 - pytest.skip outcomes included
            if type(e).__name__ in ("Skipped", "SkipTest"):
                print(f"skip {name}{a if a else ''}: {e}"); return
            fails += 1; print(f"FAIL {name}{a if a else ''}: {type(e).__name__}: {e}")
    run("test_binding_is_the_tier_word_over_backward_capable_tolerance_rows", test_binding_is_the_tier_word_over_backward_capable_tolerance_rows)
    run("test_cells_naming_the_stock_statement_come_back_as_the_declared_cell_word", test_cells_naming_the_stock_statement_come_back_as_the_declared_cell_word)
    run("test_the_adapter_carries_no_kernel_and_no_size_card_or_tile_table", test_the_adapter_carries_no_kernel_and_no_size_card_or_tile_table)
    for M, C in CASES:
        run("test_bound_row_matches_stock_within_stock_error", test_bound_row_matches_stock_within_stock_error, M, C)
    run("test_module_under_haiku_serves_by_the_row_and_keeps_stocks_parameter_tree", test_module_under_haiku_serves_by_the_row_and_keeps_stocks_parameter_tree)
    res = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestEvidenceCensusReadsTransition))
    fails += len(res.failures) + len(res.errors)
    print(f"test_layers_transition: {'all passed' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
