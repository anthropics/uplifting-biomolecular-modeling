"""Lever ln (kernels/layers_ln.py = the adapter; the kernels are the shared core's provider row `cd_ln` =
opt_core.kernels.pallas.cd_layers.layers_ln): the row's Pallas LayerNorm kernels against haiku's formula in f32 (the row's reference_layer_norm) —
forward, the activation gradient d_x (random cotangent), and the parameter gradients d_scale/d_offset (plain-JAX part of the VJP) — for bf16 and
f32 rows, row counts that are NOT a multiple of the row tile (masked tail), C in {64, 128, 256}; the adapter's serving rule through the real
common_modules.LayerNorm module under haiku with the ROW word pinned for the test (served bf16 input; a 384-channel input falls back by name);
and the tier-word contract of the tree's own binding (a call is served by the bound row or handed to stock's method by the cell's name — never
silent). GPU only (Pallas/Triton); skipped by name elsewhere. Run as a script with its timing argument (see `__main__`) it prints XLA-vs-Pallas fwd and fwd+bwd
times (DCE-safe) at pair sizes N=200/500/800.
"""
import contextlib
import re
import sys, time

import numpy as np


try:
    import pytest
except ImportError:                      # an image without pytest: `python test_layers_<x>.py` runs the same functions (see _main)
    try:
        from colabdesign_opt.tests._noptest import pytest          # the kit's shared stand-in, when present
    except ImportError:
        import importlib as _il, types as _ty
        pytest = _ty.SimpleNamespace(importorskip=_il.import_module, mark=_ty.SimpleNamespace(parametrize=lambda names, values: (lambda fn: fn)))


def _skip(why):
    import unittest
    sk = getattr(pytest, "skip", None)
    if sk is not None:
        sk(why)
    raise unittest.SkipTest(why)


def _stack():
    """Import the real stack lazily (never at module import: the kit suite imports every test module on CPU images with stand-in packages) and bind
    the names the tests use as module globals; skip by name when the real jax / haiku / colabdesign / a GPU is not there."""
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
        import haiku as _hk
        from colabdesign.af.alphafold.model import common_modules as _cm, utils as _au
        from colabdesign_opt.kernels import layers_ln as _L, provider as _P
        from opt_core.kernels.pallas.cd_layers import layers_ln as _K            # the provider row module: the kernels this lever serves
    except ImportError as e:
        _skip(f"real stack not importable: {e}")
    g.update(jax=_jax, jnp=_jax.numpy, hk=_hk, common_modules=_cm, af_utils=_au, L=_L, P=_P, K=_K)
    g["_STACK_READY"] = True


@contextlib.contextmanager
def _row_word():
    """Pin the ROW word on the lever for the duration (default rule: the provider serves exactly that row in every cell, measured or not) —
    the kernel-serving path under test, independent of the provider's cell table; the tree's own word is restored after."""
    P.reset_for_tests()
    b = P.BINDINGS[L.NAME]
    keep = b.word
    b.word = next(iter(b.rows))
    try:
        yield b.word
    finally:
        b.word = keep
        P.reset_for_tests()


def _gpu() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:      # noqa: BLE001
        return False





def _relerr(x, ref):
    x = x.astype(jnp.float32); ref = ref.astype(jnp.float32)
    return float(jnp.linalg.norm((x - ref).ravel()) / (jnp.linalg.norm(ref.ravel()) + 1e-30))


@pytest.mark.parametrize("M,C,dtype", [(1000, 128, "bfloat16"), (777, 128, "bfloat16"), (513, 64, "bfloat16"), (333, 256, "float32"), (4096, 128, "float32")])
def test_kernel_matches_reference(M, C, dtype):
    _stack()
    if not _gpu():
        print("skipped: no gpu"); return
    dt = getattr(jnp, dtype)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    x = (2.0 * jax.random.normal(k1, (M, C)) + 0.5).astype(dt)
    scale = 1.0 + 0.2 * jax.random.normal(k2, (C,)); offset = 0.1 * jax.random.normal(k3, (C,))
    ct = jax.random.normal(k4, (M, C)).astype(dt)
    fv = (M == 777)                                                                    # one bf16 case runs the fast-variance form
    y, vjp = jax.vjp(lambda a, s, o: K.layer_norm_2d(a, s, o, 1e-5, fv), x, scale, offset)
    dx, ds, do = vjp(ct)
    yr, vjpr = jax.vjp(lambda a, s, o: K.reference_layer_norm(a, s, o, 1e-5, fv), x, scale, offset)
    dxr, dsr, dor = vjpr(ct)
    tol = 1e-2 if dtype == "bfloat16" else 2e-5
    e = {k: _relerr(a, b) for k, a, b in (("y", y, yr), ("dx", dx, dxr), ("dscale", ds, dsr), ("doffset", do, dor))}
    print(f"M={M} C={C} {dtype}: " + " ".join(f"{k}={v:.2e}" for k, v in e.items()))
    assert y.dtype == dt and dx.dtype == dt
    for k, v in e.items():
        assert v < tol, (k, v, tol)


@pytest.mark.parametrize("fast_variance", [False, True])
def test_each_variance_form_matches_stock_haiku_module(fast_variance):
    """The served class against the STOCK common_modules.LayerNorm (haiku) with the same use_fast_variance flag: y (bf16 rows with a large mean,
    where the two variance forms differ most) and d_x."""
    _stack()
    if not _gpu():
        print("skipped: no gpu"); return
    x = (3.0 + 0.5 * jax.random.normal(jax.random.PRNGKey(3), (29, 31, 128))).astype(jnp.bfloat16)
    ct = jax.random.normal(jax.random.PRNGKey(4), (29, 31, 128)).astype(jnp.bfloat16)

    def make(cls):
        def f(a):
            with af_utils.bfloat16_context():
                return cls([-1], True, True, use_fast_variance=fast_variance, name="norm")(a)
        return hk.without_apply_rng(hk.transform(f))
    with _row_word():
        _variance_form_case(make, fast_variance, x, ct)


def _variance_form_case(make, fast_variance, x, ct):
    L.install()
    try:
        served, stock = make(common_modules.LayerNorm), make(L.stock_class())
        params = {"norm": {"scale": 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(5), (128,)), "offset": 0.1 * jax.random.normal(jax.random.PRNGKey(6), (128,))}}
        n0 = L.census()["served"]
        ys, vs = jax.vjp(lambda a: jax.jit(served.apply)(params, a), x)
        yr, vr = jax.vjp(lambda a: jax.jit(stock.apply)(params, a), x)
        assert L.census()["served"] > n0, "the served class did not see the call"
        (dxs,), (dxr,) = vs(ct), vr(ct)
        ey, ed = _relerr(ys, yr), _relerr(dxs, dxr)
        neq = float(jnp.mean((ys != yr).astype(jnp.float32)))
        print(f"use_fast_variance={fast_variance}: y relerr vs stock haiku {ey:.2e} (fraction of bf16 elements not bitwise-equal {neq:.4f}), dx relerr {ed:.2e}")
        assert ey < 4e-3 and ed < 1e-2, (ey, ed)
    finally:
        L.uninstall()


def test_module_serving_rule_under_haiku():
    _stack()
    if not _gpu():
        print("skipped: no gpu"); return
    with _row_word():
        _serving_rule_case()


def _serving_rule_case():
    L.install()
    try:
        def f(pair, single):
            with af_utils.bfloat16_context():
                a = common_modules.LayerNorm([-1], True, True, name="pair_norm")(pair)        # served (C=128, bf16)
                b = common_modules.LayerNorm([-1], True, True, name="single_norm")(single)    # 384 channels: fallback by name
            return a, b
        t = hk.without_apply_rng(hk.transform(f))
        pair = jax.random.normal(jax.random.PRNGKey(1), (37, 41, 128)).astype(jnp.bfloat16)
        single = jax.random.normal(jax.random.PRNGKey(2), (41, 384))
        before = L.census()
        params = t.init(None, pair, single)
        assert params["pair_norm"]["scale"].shape == (128,) and params["pair_norm"]["scale"].dtype == jnp.float32
        a, b = jax.jit(t.apply)(params, pair, single)
        c = L.census()
        assert c["served"] > before["served"] and c["fallback_by"].get("channels_not_pow2", 0) > before["fallback_by"].get("channels_not_pow2", 0), c
        ar = K.reference_layer_norm(pair, params["pair_norm"]["scale"], params["pair_norm"]["offset"])
        assert a.dtype == jnp.bfloat16 and _relerr(a, ar) < 1e-2
        line = L.lever_line()
        assert re.match(r"^\[colabdesign-opt\] LEVER name=ln state=on impl=layers_ln@[0-9a-f]{8} origin=core numerics=precision precision=none served=", line), line   # impl = the provider row module @ sha8 (kernels/provider.py)
    finally:
        L.uninstall()


def test_tier_word_binding_serves_the_row_or_names_the_cell():
    """The tree's own binding (the provider's TIER word): a model-shaped pair LayerNorm is either served by the bound row (census `served`) or
    handed to stock's method BY THE CELL'S NAME (`fallback_by=cell_<arm>`), or the lever refuses at install BY NAME — the provider's table decides
    which; the kit accounts for every call either way (a class contract, not a statement about the table's current winner)."""
    _stack()
    if not _gpu():
        print("skipped: no gpu"); return
    P.reset_for_tests()
    assert P.BINDINGS[L.NAME].word == P.TIER_WORD, P.BINDINGS[L.NAME].word
    try:
        L.install()
    except L.Refusal as e:
        assert "provider" in str(e) and L.NAME in str(e), str(e)                      # refused by name (the tier word names no bindable row for the main cell here)
        print(f"tier word: install refused by name: {e}")
        return
    try:
        def f(pair):
            with af_utils.bfloat16_context():
                return common_modules.LayerNorm([-1], True, True, name="pair_norm")(pair)
        t = hk.without_apply_rng(hk.transform(f))
        pair = jax.random.normal(jax.random.PRNGKey(7), (64, 64, 128)).astype(jnp.bfloat16)
        before = L.census()
        params = t.init(None, pair)
        y = jax.jit(t.apply)(params, pair)
        c = L.census()
        served = c["served"] - before["served"]
        cell_fallbacks = sum(v for k, v in c["fallback_by"].items() if k.startswith(P.CELL_FALLBACK)) - sum(v for k, v in before["fallback_by"].items() if k.startswith(P.CELL_FALLBACK))
        assert served > 0 or cell_fallbacks > 0, (before, c)                          # accounted for, one way or the other
        assert y.dtype == jnp.bfloat16 and _relerr(y, K.reference_layer_norm(pair, params["pair_norm"]["scale"], params["pair_norm"]["offset"])) < 1e-2
        f = P.facts(L.NAME)
        assert f["word"] == P.TIER_WORD and f["provider"].startswith(P.PROVIDER + "@"), f
        print(f"tier word: served={served} cell_fallbacks={cell_fallbacks} facts={f}")
    finally:
        L.uninstall()
        P.reset_for_tests()


def _bench_impl(sizes=(200, 500, 800), C=128, reps=30):
    print("device", jax.devices()[0])
    for N in sizes:
        M = N * N
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
        x = jax.random.normal(k1, (M, C)).astype(jnp.bfloat16)
        scale = 1.0 + 0.2 * jax.random.normal(k2, (C,)); offset = 0.1 * jax.random.normal(k3, (C,))
        ct = jax.random.normal(k4, (M, C)).astype(jnp.bfloat16)
        for label, fn in (("xla", K.reference_layer_norm), ("pallas", K.layer_norm_2d)):
            fwd = jax.jit(lambda a, s, o: fn(a, s, o, 1e-5))

            @jax.jit
            def fwdbwd(a, s, o, ct):
                y, vjp = jax.vjp(lambda a_: fn(a_, s, o, 1e-5), a)
                (dx,) = vjp(ct)
                return jnp.sum(y[:7, :3].astype(jnp.float32)) + jnp.sum(dx.astype(jnp.float32))     # depends on both leaves
            for f, args, nm in ((fwd, (x, scale, offset), "fwd"), (fwdbwd, (x, scale, offset, ct), "fwd+bwd")):
                jax.block_until_ready(f(*args)); jax.block_until_ready(f(*args))
                t0 = time.perf_counter()
                for _ in range(reps):
                    r = f(*args)
                jax.block_until_ready(r)
                dt = (time.perf_counter() - t0) / reps
                gbs = (M * C * 2 * (2 if nm == "fwd" else 3)) / dt / 1e9
                print(f"N={N} [{M}x{C} bf16] {label:6s} {nm:8s} {dt*1e6:9.1f} us  ~{gbs:6.0f} GB/s")


def _main():
    for M, C, d in [(1000, 128, "bfloat16"), (777, 128, "bfloat16"), (513, 64, "bfloat16"), (333, 256, "float32"), (4096, 128, "float32")]:
        test_kernel_matches_reference(M, C, d)
    for fv in (False, True):
        test_each_variance_form_matches_stock_haiku_module(fv)
    test_module_serving_rule_under_haiku()
    test_tier_word_binding_serves_the_row_or_names_the_cell()
    print("test_layers_ln: all passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        _stack(); _bench_impl()
    else:
        _main()
