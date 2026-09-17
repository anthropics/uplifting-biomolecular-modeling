"""Lever opm_fold (kernels/layers_opm.py): the re-associated OuterProductMean against (1) stock's module bytes (modules.OuterProductMean, chunked scan) and
(2) an fp32 HIGHEST-precision reference of the same formula — forward AND the activation gradient (vjp w.r.t. the MSA activations with a random
cotangent; the design step differentiates w.r.t. the sequence only, never the weights). Pass rule (class precision): the lever's error vs the fp32
reference equals stock's own bf16-path error within 5 % (ratios measured: out 1.00-1.02, d_act 0.99-1.04), on out and on d_act, at every size and S in {1,2,4,5} incl. all-zero mask rows/columns. Run as a script with its timing argument
(see `__main__`) it prints fwd and fwd+bwd timings (DCE-safe: the timed value depends on every output leaf) at N = 200/500/800 on the visible GPU.
"""
import re
import contextlib
import functools, sys, time

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
        from colabdesign.af.alphafold.model import modules as _mods, config as _cfg, utils as _au
        from colabdesign_opt.kernels import layers_opm as _O, provider as _P
    except ImportError as e:
        _skip(f"real stack not importable: {e}")
    g.update(jax=_jax, jnp=_jax.numpy, hk=_hk, modules=_mods, af_config=_cfg, af_utils=_au, layers_opm=_O, P=_P)
    g["_STACK_READY"] = True


@contextlib.contextmanager
def _row_word():
    """Pin the ROW word on the lever for the duration (default rule: the provider serves exactly that row in every cell, measured or not) —
    the re-association under test at every S / N below, independent of the provider's cell table; the tree's own word is restored after."""
    P.reset_for_tests()
    b = P.BINDINGS[layers_opm.NAME]
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



def _configs():
    cfg = af_config.model_config("model_1_multimer_v3")
    c = cfg.model.embeddings_and_evoformer.evoformer.outer_product_mean
    gc = cfg.model.global_config
    return c, gc


def _make(S, N, C_m=256, F=128, seed=0):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(k, 3)
    act = jax.random.normal(k1, (S, N, C_m), jnp.float32)
    mask = jnp.ones((S, N), jnp.float32)
    ct = jax.random.normal(k3, (N, N, F), jnp.float32)          # cotangent for the vjp
    return act, mask, ct, k2


def _transformed(use_lever: bool, F=128):
    c, gc = _configs()

    def f(act, mask):
        with af_utils.bfloat16_context():
            cls = modules.OuterProductMean if use_lever else layers_opm.stock_class()      # install() rebinds modules.OuterProductMean
            assert bool(getattr(cls, layers_opm.MARKER, False)) == bool(use_lever)
            return cls(c, gc, num_output_channel=F, name="outer_product_mean")(act, mask)
    return hk.without_apply_rng(hk.transform(f))


def _params(S, N, C_m, F, key):
    """Random (non-zero) parameters with stock's names, f32 (the bf16 getter casts at use)."""
    ks = jax.random.split(key, 6)
    c = 32
    return {
        "outer_product_mean/layer_norm_input": {"scale": 1.0 + 0.1 * jax.random.normal(ks[0], (C_m,)), "offset": 0.1 * jax.random.normal(ks[1], (C_m,))},
        "outer_product_mean/left_projection": {"weights": jax.random.normal(ks[2], (C_m, c)) / np.sqrt(C_m), "bias": 0.1 * jax.random.normal(ks[3], (c,))},
        "outer_product_mean/right_projection": {"weights": jax.random.normal(ks[4], (C_m, c)) / np.sqrt(C_m), "bias": jnp.zeros((c,))},
        "outer_product_mean": {"output_w": jax.random.normal(ks[5], (c, c, F)) / c, "output_b": 0.05 * jnp.ones((F,))},
    }


def _reference(params, act, mask):
    """fp32, HIGHEST precision, stock's association (sum over a first, then the K=c*c contraction)."""
    p = params
    x = act
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    x = (x - mu) * jax.lax.rsqrt(var + 1e-5) * p["outer_product_mean/layer_norm_input"]["scale"] + p["outer_product_mean/layer_norm_input"]["offset"]
    m = mask[..., None]
    hp = jax.lax.Precision.HIGHEST
    l = m * (jnp.einsum("snc,cd->snd", x, p["outer_product_mean/left_projection"]["weights"], precision=hp) + p["outer_product_mean/left_projection"]["bias"])
    r = m * (jnp.einsum("snc,cd->snd", x, p["outer_product_mean/right_projection"]["weights"], precision=hp) + p["outer_product_mean/right_projection"]["bias"])
    P = jnp.einsum("abc,ade->bdce", l, r, precision=hp)
    out = jnp.einsum("bdce,cef->bdf", P, p["outer_product_mean"]["output_w"], precision=hp) + p["outer_product_mean"]["output_b"]
    norm = jnp.einsum("abc,adc->bdc", m, m)
    return out / (1e-3 + norm)


def _run(fn, params, act, mask, ct):
    """out (f32) and d_act for a bf16 activation input (the Evoformer's dtype), cotangent ct."""
    a16 = act.astype(jnp.bfloat16)
    out, vjp = jax.vjp(lambda a: fn(params, a, mask).astype(jnp.float32), a16)
    (da,) = vjp(ct)
    return out, da.astype(jnp.float32)


def _relerr(x, ref):
    return float(jnp.linalg.norm((x - ref).ravel()) / (jnp.linalg.norm(ref.ravel()) + 1e-30))


@pytest.mark.parametrize("S,N", [(2, 48), (1, 40), (4, 72), (5, 40), (2, 130)])      # S=1 (extra-MSA stack), 2 (the design MSA), 4, 5; 130 > chunk 128: stock's scan + remainder path
def test_opm_matches_reference_within_stock_error(S, N):
    _stack()
    with _row_word():
        _reference_case(S, N)


def _reference_case(S, N):
    layers_opm.install()            # records the stock __call__ for the A side
    try:
        act, mask, ct, kp = _make(S, N)
        if N == 40:
            mask = mask.at[:, -5:].set(0.0)                          # padded columns exercise the mask/norm path
        if S == 4:
            mask = mask.at[1, :9].set(0.0).at[3, 20:].set(0.0)       # per-sequence masks differ -> norm[b,d] takes every value 1..4
        if S == 5:
            mask = mask.at[:, 7].set(0.0).at[2, :].set(0.0)          # an all-zero mask COLUMN (residue 7 in every sequence: norm=0 there, out = bias/1e-3) and an all-zero mask ROW (sequence 2)
        params = _params(S, N, 256, 128, kp)
        ref_out, ref_vjp = jax.vjp(lambda a: _reference(params, a, mask), act)
        (ref_da,) = ref_vjp(ct)
        so, sda = _run(jax.jit(_transformed(False).apply), params, act, mask, ct)
        lo, lda = _run(jax.jit(_transformed(True).apply), params, act, mask, ct)
        e_so, e_lo = _relerr(so, ref_out), _relerr(lo, ref_out)
        e_sd, e_ld = _relerr(sda, ref_da), _relerr(lda, ref_da)
        print(f"S={S} N={N}: out relerr stock={e_so:.3e} lever={e_lo:.3e} | d_act relerr stock={e_sd:.3e} lever={e_ld:.3e}")
        # PARITY, not strictly below: with T in f32 the lever's error equals stock's own within a few % (measured ratios out 1.00-1.02, d_act
        # 0.99-1.04 over S in {1,2,4,5}, N to 300); both are dominated by the bf16 roundings they SHARE (inputs, projections, parameters, the bf16
        # output) — running the lever's two f32 products at precision HIGHEST instead of DEFAULT (TF32) changes nothing (5.281e-3 -> 5.272e-3),
        # so the residual is rounding alignment, not lost precision. Rule: within 5 % of stock's error.
        assert e_lo <= 1.05 * e_so + 1e-6, (e_lo, e_so)
        assert e_ld <= 1.05 * e_sd + 1e-6, (e_ld, e_sd)
        assert e_lo < 2e-2 and e_ld < 2e-2
        cos = float(jnp.vdot(lda, sda) / (jnp.linalg.norm(lda) * jnp.linalg.norm(sda)))
        assert cos > 0.999, cos
    finally:
        layers_opm.uninstall()


def test_lever_line_grammar():
    _stack()
    with _row_word() as row:
        layers_opm.install()
        try:
            line = layers_opm.lever_line()
            assert re.match(r"^\[colabdesign-opt\] LEVER name=opm_fold state=on impl=layers_opm@[0-9a-f]{8} origin=core numerics=precision precision=tf32 served=", line), line   # impl = the provider row module @ sha8 (kernels/provider.py)
            assert f" word={row} " in line and " provider=opt_core.kernels.pallas@" in line and " row=" in line and " tier=" in line, line
        finally:
            layers_opm.uninstall()


def test_tier_word_binding_installs_or_refuses_by_name():
    """The tree's own binding is the provider's TIER word: install either admits a row this adapter binds for the model's main cell (the LEVER
    line then names word=<tier> and the row served) or refuses BY NAME — the provider's table decides which (a class contract)."""
    _stack()
    P.reset_for_tests()
    assert P.BINDINGS[layers_opm.NAME].word == P.TIER_WORD, P.BINDINGS[layers_opm.NAME].word
    try:
        layers_opm.install()
    except layers_opm.Refusal as e:
        assert "provider" in str(e) and layers_opm.NAME in str(e), str(e)
        print(f"tier word: install refused by name: {e}")
        return
    try:
        line = layers_opm.lever_line()
        assert f" word={P.TIER_WORD} " in line and re.search(r" row=cd_opm\b", line), line
        print(f"tier word: {line}")
    finally:
        layers_opm.uninstall()
        P.reset_for_tests()


# ─────────────────────────────── GPU microbench (not a test) ───────────────────────────────
def _bench_impl(sizes=(200, 500, 800), S=2, reps=20):
    P.reset_for_tests(); P.BINDINGS[layers_opm.NAME].word = next(iter(P.BINDINGS[layers_opm.NAME].rows))      # the row, at every size (microbench of the kernel, not of the table)
    layers_opm.install()
    print("device", jax.devices()[0])
    for N in sizes:
        act, mask, ct, kp = _make(S, N)
        params = _params(S, N, 256, 128, kp)
        a16 = act.astype(jnp.bfloat16)
        for label, use in (("stock", False), ("opm", True)):
            fn = _transformed(use).apply
            fwd = jax.jit(lambda p, a, m: fn(p, a, m))

            @jax.jit
            def fwdbwd(p, a, m, ct):
                out, vjp = jax.vjp(lambda a_: fn(p, a_, m), a)
                (da,) = vjp(ct.astype(out.dtype))
                return jnp.sum(out.astype(jnp.float32)) + jnp.sum(da.astype(jnp.float32))   # depends on every output leaf (no DCE of the backward)
            for f, args, nm in ((fwd, (params, a16, mask), "fwd"), (fwdbwd, (params, a16, mask, ct), "fwd+bwd")):
                jax.block_until_ready(f(*args)); jax.block_until_ready(f(*args))
                t0 = time.perf_counter()
                for _ in range(reps):
                    r = f(*args)
                jax.block_until_ready(r)
                dt = (time.perf_counter() - t0) / reps
                print(f"N={N} S={S} {label:5s} {nm:8s} {dt*1e3:8.3f} ms")
        # numerics at size
        ref_out, ref_vjp = jax.vjp(lambda a: _reference(params, a, mask), act)
        (ref_da,) = ref_vjp(ct)
        so, sda = _run(jax.jit(_transformed(False).apply), params, act, mask, ct)
        lo, lda = _run(jax.jit(_transformed(True).apply), params, act, mask, ct)
        print(f"N={N}: out relerr stock={_relerr(so, ref_out):.3e} lever={_relerr(lo, ref_out):.3e} | d_act relerr stock={_relerr(sda, ref_da):.3e} lever={_relerr(lda, ref_da):.3e} | max|out| diff lever-stock={float(jnp.max(jnp.abs(lo-so))):.3e}")


def _main():
    for S, N in [(2, 48), (1, 40), (4, 72), (5, 40), (2, 130)]:
        test_opm_matches_reference_within_stock_error(S, N)
    test_lever_line_grammar()
    test_tier_word_binding_installs_or_refuses_by_name()
    print("test_layers_opm: all passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        _stack(); _bench_impl()
    else:
        _main()
