"""pallas_attn_serve per-call rules: the N_keys floor (``below_keys_rule``) and the head-dim zero-pad (``pad_head_dim``), CPU logic with a
stubbed probe; on a GPU (CUDA jax) the two call classes they open — an MSA column attention without pair bias (H=8, D=32, hundreds of keys)
and an extra-MSA row attention with head dim 8 (zero-padded to 16) — against the materialised XLA reference, with a timing print."""
import time

import pytest

from opt_core.kernels import pallas_attn_serve as S


@pytest.fixture
def gpu_probe(monkeypatch):
    monkeypatch.setattr(S, "probe", lambda refresh=False, require_gpu=True: {"ok": True, "kind": None, "backend": "gpu"})


def test_min_keys_floor_is_a_named_rule(gpu_probe):
    assert S.ineligible(32, 32, False, (128, 1, 1, 508), all_calls=True, n_keys=508, min_keys=256) is None
    assert S.ineligible(32, 32, False, (128, 1, 1, 64), all_calls=True, n_keys=64, min_keys=256) == S.BELOW_KEYS_RULE
    assert S.ineligible(32, 32, False, (128, 1, 1, 64), all_calls=True, n_keys=64, min_keys=0) is None            # no floor declared
    assert S.ineligible(32, 32, False, (128, 1, 1, 508), all_calls=False, n_keys=508, min_keys=256) == S.NO_PAIR_BIAS


def test_head_dim_below_16_is_refused_unless_padding_is_asked(gpu_probe):
    assert S.ineligible(8, 8, True, (4, 1, 1, 256)) == S.HEAD_DIM_LT_16
    assert S.ineligible(8, 8, True, (4, 1, 1, 256), pad_head_dim=True) is None
    assert S.ineligible(8, 16, True, (4, 1, 1, 256), pad_head_dim=True) == S.KEY_DIM_NE_VALUE_DIM
    assert S.PADDED_HEAD_DIM_SUFFIX == "p16" and S.MIN_HEAD_DIM == 16


def test_probe_vocabulary_names_a_removed_pallas_api():
    assert S.PALLAS_API_REMOVED == "pallas_api_removed" and "pallas_call" in S.PALLAS_API and S.PALLAS_IO_API == ("load", "store")


jnp = None
try:
    import jax
    import jax.numpy as jnp  # noqa: F811
    GPU = jax.default_backend() == "gpu"
except Exception:  # noqa: BLE001
    GPU = False
gpu = pytest.mark.skipif(not GPU, reason="needs a CUDA jax backend (Pallas-Triton lowering)")


@pytest.mark.skipif(jnp is None, reason="jax not importable")
def test_pad_head_dim_is_a_noop_at_16_and_zero_pads_below():
    q = jnp.ones((2, 8, 32, 8), jnp.bfloat16); k = jnp.ones((2, 8, 40, 8), jnp.bfloat16); v = jnp.full((2, 8, 40, 8), 2.0, jnp.bfloat16)
    qp, kp, vp, dv = S.pad_head_dim(q, k, v)
    assert qp.shape == (2, 8, 32, 16) and kp.shape == (2, 8, 40, 16) and vp.shape == (2, 8, 40, 16) and dv == 8
    assert bool(jnp.all(qp[..., 8:] == 0)) and bool(jnp.all(vp[..., :8] == 2.0))
    q16 = jnp.ones((2, 8, 32, 16), jnp.bfloat16)
    assert S.pad_head_dim(q16, q16, q16)[0] is q16


def _rel_rms(a, b):
    a = a.astype(jnp.float32); b = b.astype(jnp.float32)
    return float(jnp.sqrt(jnp.mean(jnp.square(a - b))) / (jnp.sqrt(jnp.mean(jnp.square(b))) + 1e-30))


def _timeit(fn, *args, reps=20):
    out = fn(*args); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps * 1e3, out


@gpu
def test_msa_column_attention_without_pair_bias_is_served_in_class():
    """[B=128 columns, H=8, S=508 sequences, D=32] bf16, zero pair bias, all keys valid (an MSA column attention chunk)."""
    B, H, S_, D = 128, 8, 508, 32
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(ks[i], (B, H, S_, D), jnp.float32).astype(jnp.bfloat16) for i in range(3))
    mask_bias = jnp.zeros((B, 1, 1, S_), jnp.float32)
    scale = D ** -0.5
    ref_fn = jax.jit(lambda q, k, v: S.reference_attention(q, k, v, jnp.zeros((H, S_, S_), q.dtype), jnp.ones((B, S_), bool), scale))
    ker_fn = jax.jit(lambda q, k, v: S.attention_core(q, k, v, mask_bias, None, scale))
    t_ref, ref = _timeit(ref_fn, q, k, v)
    t_ker, out = _timeit(ker_fn, q, k, v)
    out2 = ker_fn(q, k, v)
    e = _rel_rms(out, ref)
    print(f"msa column attention B{B}xH{H}xS{S_}xD{D} bf16 no pair bias: rel_rms vs XLA bf16 body {e:.3e}; pallas {t_ker:.2f} ms vs XLA {t_ref:.2f} ms ({t_ref / t_ker:.2f}x)")
    assert bool(jnp.all(out == out2)) and e <= 2.0e-2, e


@gpu
@pytest.mark.parametrize("n", [512, 1000])
def test_extra_msa_row_attention_head_dim_8_is_served_by_zero_padding(n):
    """[B=128 extra sequences (a chunk), H=8, S=N residues, D=8] bf16 with a pair bias [8,N,N] and a padded key tail."""
    B, H, D = 128, 8, 8
    ks = jax.random.split(jax.random.PRNGKey(1), 4)
    q, k, v = (jax.random.normal(ks[i], (B, H, n, D), jnp.float32).astype(jnp.bfloat16) for i in range(3))
    pair_bias = jax.random.normal(ks[3], (H, n, n), jnp.float32).astype(jnp.bfloat16)
    kmask = jnp.broadcast_to(jnp.arange(n) < n - 24, (B, n))
    mask_bias = jnp.where(kmask, 0.0, -1e9)[:, None, None, :].astype(jnp.float32)
    scale = D ** -0.5
    ref_fn = jax.jit(lambda q, k, v, b: S.reference_attention(q, k, v, b, kmask, scale))
    ker_fn = jax.jit(lambda q, k, v, b: S.attention_core(q, k, v, mask_bias, b, scale, pad_head_dim_below_min=True))
    t_ref, ref = _timeit(ref_fn, q, k, v, pair_bias, reps=10)
    t_ker, out = _timeit(ker_fn, q, k, v, pair_bias, reps=10)
    assert out.shape == (B, H, n, D)
    e = _rel_rms(out, ref)
    print(f"extra-msa row attention B{B}xH{H}xS{n}xD{D}(p16) bf16 + pair bias: rel_rms vs XLA bf16 body {e:.3e}; pallas {t_ker:.2f} ms vs XLA {t_ref:.2f} ms ({t_ref / t_ker:.2f}x)")
    assert e <= 2.0e-2, e


def test_precision_is_stamped_by_the_serve_layer():
    """0.5.2: emit_line stamps precision=<word> itself; without jax the word names the refusal instead of failing."""
    from opt_core.kernels import pallas_attn_serve as S
    try:
        import jax  # noqa: F401
        have_jax = True
    except Exception:
        have_jax = False
    w = S.served_precision()
    if have_jax:
        assert w in ("ieee", "tf32", "bf16")
    else:
        assert w.startswith("unavailable:")
    src = open(S.__file__).read()
    assert "evidence[key] = served_word(field, ledger)" in src and "F32_PRECISIONS = (" not in src           # stamped by the serve layer (precision, bwd_precision, dq: one grammar), never re-typed there
    assert '("precision", "f32_precision"), ("bwd_precision", "bwd_f32_precision"), ("dq", "dq"), ("dbias", "dbias")' in src
