"""opt_core.mem.rowpair_jax.shard — the all_to_all element limit: `shard.all_to_all` issues ONE `lax.all_to_all` below
`A2A_MAX_ELEMS` local elements and k channel pieces at or above it; the pieces concatenated must equal the one call BIT FOR BIT (class moves_bytes).

The arithmetic tests (`a2a_plan`, the env limit, the evidence word at state=off) need no jax. The device tests force the split on small operands with
``ROWPAIR_JAX_A2A_MAX_ELEMS`` and compare against the single call on a multi-device CPU jax (``XLA_FLAGS=--xla_force_host_platform_device_count=4``:
P in {2, 4}) or a multi-GPU box; they skip BY NAME without jax (the core's own test interpreter has none).
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)

_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")             # four host devices for THIS process's jax backend on a CPU box (harmless on gpu): set, initialise, restore
if "xla_force_host_platform_device_count" not in (_XLA_FLAGS_BEFORE or ""):
    os.environ["XLA_FLAGS"] = ((_XLA_FLAGS_BEFORE or "") + " --xla_force_host_platform_device_count=4").strip()
try:
    import jax as _jax_backend_init  # noqa: E402
    _jax_backend_init.devices()
except Exception:  # noqa: BLE001 — no jax here: the device tests skip by name
    pass
finally:
    if _XLA_FLAGS_BEFORE is None:
        os.environ.pop("XLA_FLAGS", None)
    else:
        os.environ["XLA_FLAGS"] = _XLA_FLAGS_BEFORE

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
from opt_core.mem import MemLeverRefused  # noqa: E402
from opt_core.mem.rowpair_jax import evidence, mesh, shard  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    HAVE_JAX, _JAX_WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE_JAX, _JAX_WHY = False, repr(_e)
needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="rowpair_jax device tests need jax (%s): run with xla_force_host_platform_device_count=4 or on a multi-GPU box" % _JAX_WHY)
if HAVE_JAX:
    PLATFORM, NDEV = jax.devices()[0].platform, jax.device_count()
else:
    PLATFORM, NDEV = "none", 1
PS = [p for p in (2, 4, 8) if p <= NDEV]                       # P>1 only: P=1 builds no mesh and never reaches a collective


# ------------------------------------------------------------------------------------------------ arithmetic (no jax)
def test_plan_below_limit_is_the_single_call():
    assert shard.A2A_MAX_ELEMS == 2 ** 31 and shard.A2A_MAX_ELEMS_ENV == "ROWPAIR_JAX_A2A_MAX_ELEMS"
    assert shard.a2a_plan((2530, 5060, 128), 1, 0, limit=2 ** 31) == (None, [])            # the JAX design engine's passing shape: one call
    assert shard.a2a_plan((2047, 8192, 128), 1, 0, limit=2 ** 31) == (None, [])            # 2**31 - 2**20 elements: one call
    assert shard.a2a_plan((4, 8, 5), 1, 0, limit=161) == (None, [])                      # 160 < 161
    dim, sizes = shard.a2a_plan((4, 8, 5), 1, 0, limit=160)                              # 160 >= 160: split (pieces must be < limit)
    assert dim == 2 and sum(sizes) == 5 and all(4 * 8 * s < 160 for s in sizes)


def test_plan_at_the_limit_cuts_the_channel_dim():
    dim, sizes = shard.a2a_plan((2048, 8192, 128), 1, 0, limit=2 ** 31)                 # the failing shape: 2**31 elements → 2 pieces of 64 channels
    assert (dim, sizes) == (2, [64, 64])
    dim, sizes = shard.a2a_plan((4096, 8192, 128), 1, 0, limit=2 ** 31)                 # 2**32 → 63 channels fit under the limit → 3 near-equal pieces
    assert (dim, sizes) == (2, [43, 43, 42]) and all(4096 * 8192 * s < 2 ** 31 for s in sizes)
    assert shard.a2a_plan((8192, 4096, 128), 0, 1, limit=2 ** 31) == (2, [43, 43, 42])   # cols_to_rows of the column block: the same plan
    assert shard.a2a_plan((128, 2048, 8192), 2, 1, limit=2 ** 31) == (0, [64, 64])       # channel-first [C, N/P, N] (ckj,cki->cij): the free dim is 0
    dim, sizes = shard.a2a_plan((4, 8, 5), 1, 0, limit=100)                              # per channel 32 elements → 3 fit → ceil(5/3)=2 pieces, near-equal 3+2
    assert (dim, sizes) == (2, [3, 2])
    assert shard.a2a_plan((4, 8, 5), 1, 0, limit=33) == (2, [1, 1, 1, 1, 1])             # width 1: k == C
    assert shard.a2a_plan((4, 8, 5), 1, 0, limit=32) == (None, [])                      # one channel slice (32) is not < 32: not cuttable → the one call, as today
    assert shard.a2a_plan((64, 128), 1, 0, limit=100) == (None, [])                     # rank 2: no free dim → the one call, as today
    dim, sizes = shard.a2a_plan((6, 12, 4, 7), 1, 0, limit=500)                          # rank 4: the largest free extent (dim 3, 7) is cut
    assert dim == 3 and sum(sizes) == 7 and max(sizes) - min(sizes) <= 1 and all(6 * 12 * 4 * s < 500 for s in sizes)
    dim, sizes = shard.a2a_plan((6, 12, 7, 7), 1, 0, limit=1000)                         # tie → the later dim
    assert dim == 3


def test_env_limit_and_refusals(monkeypatch):
    monkeypatch.delenv(shard.A2A_MAX_ELEMS_ENV, raising=False)
    assert shard.a2a_max_elems() == 2 ** 31 and shard.a2a_max_elems({}) == 2 ** 31 and shard.a2a_max_elems({shard.A2A_MAX_ELEMS_ENV: " "}) == 2 ** 31
    assert shard.a2a_max_elems({shard.A2A_MAX_ELEMS_ENV: "1000"}) == 1000
    monkeypatch.setenv(shard.A2A_MAX_ELEMS_ENV, "160")
    assert shard.a2a_max_elems() == 160 and shard.a2a_plan((4, 8, 5), 1, 0)[0] == 2      # the plan reads the env when no limit is passed
    for bad in ("0", "-5", "2e9", "lots"):
        with pytest.raises(MemLeverRefused) as ei:
            shard.a2a_max_elems({shard.A2A_MAX_ELEMS_ENV: bad})
        assert ei.value.lever == "rowpair" and ei.value.reason.startswith("refused: ROWPAIR_JAX_A2A_MAX_ELEMS=")


def test_record_and_evidence_word_off_line_unchanged():
    shard.a2a_reset()
    assert shard.a2a_record() == {"kmax": 1, "calls": 0} and shard.a2a_fields() == {}
    off = evidence.line("kit-opt", "off", 1, reason="n_gpu=1")                          # the P=1 line never carries the word (x1 composition unchanged)
    assert "a2a_split" not in off
    shard._A2A_RECORD.update(kmax=3, calls=2)                                            # as if two split calls were traced
    try:
        assert shard.a2a_fields() == {"a2a_split": 3}
        assert "a2a_split" not in evidence.line("kit-opt", "off", 1, reason="n_gpu=1")   # state=off: still absent whatever the record holds
        assert "a2a_split" not in evidence.line("kit-opt", "skipped", 2, reason="activation_refused")
    finally:
        shard.a2a_reset()


# ------------------------------------------------------------------------------------------------ device tests (jax)
def _rm(p):
    return mesh.build(p, platform=PLATFORM)


def _values(shape, dtype):
    """Distinct-valued operand (every element identifies its own position) so a misplaced byte cannot hide: arange in f32 (exact below 2**24), cast."""
    n = 1
    for s in shape:
        n *= s
    assert n < 2 ** 24
    return jnp.asarray(np.arange(n, dtype=np.float32).reshape(shape) * (1.0 if dtype == "float32" else 0.25)).astype(dtype)


def _run(p, x, limit, row_dim, col_dim, monkeypatch):
    """rows_to_cols, cols_to_rows∘rows_to_cols and transpose_block of the row-sharded ``x`` under the env limit ``limit`` (None = unset: single calls)."""
    if limit is None:
        monkeypatch.delenv(shard.A2A_MAX_ELEMS_ENV, raising=False)
    else:
        monkeypatch.setenv(shard.A2A_MAX_ELEMS_ENV, str(int(limit)))
    rm, ax = _rm(p), "row"
    nd = x.ndim
    rows, cols = shard.rows_spec(ax, nd, row_dim), shard.cols_spec(ax, nd, col_dim)

    def body(xl):                                                                        # fresh closures per call: every call retraces (the env is read at trace time)
        c = shard.rows_to_cols(xl, ax, row_dim, col_dim)
        back = shard.cols_to_rows(c, ax, row_dim, col_dim)
        tb = shard.transpose_block(xl, ax, row_dim, col_dim)
        return c, back, tb

    outs = shard.shard_map(body, rm, (rows,), (cols, rows, rows))(x)
    return [np.asarray(o.astype(jnp.float32)) for o in outs]                             # bf16 → f32 is lossless: f32 equality == bf16 bit equality


@needs_jax
@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("layout,n_mult,c", [("rows_first", 8, 8), ("rows_first", 5, 5), ("rows_first", 6, 3), ("channel_first", 8, 8), ("channel_first", 5, 5)])
def test_split_all_to_all_equals_single_call_bitwise(p, dtype, layout, n_mult, c, monkeypatch):
    """For the pair layouts the recipes use ([N/P, N, C] and the channel-first [C, N/P, N]), even and odd N/P and C: every forced piece count k
    (2 pieces, an uneven cut, k == C) gives rows_to_cols / cols_to_rows / transpose_block BITWISE equal to the single call, and equal to numpy."""
    shard.a2a_reset()
    n = n_mult * p
    if layout == "rows_first":
        shape, row_dim, col_dim, ch_dim = (n, n, c), 0, 1, 2
    else:
        shape, row_dim, col_dim, ch_dim = (c, n, n), 1, 2, 0
    x = _values(shape, dtype)
    xn = np.asarray(x.astype(jnp.float32))
    ref_cols, ref_back, ref_tb = _run(p, x, None, row_dim, col_dim, monkeypatch)       # today's single calls
    assert shard.a2a_record()["calls"] == 0                                              # the limit unset: nothing split
    assert np.array_equal(ref_back, xn) and np.array_equal(ref_cols, xn) and np.array_equal(ref_tb, np.swapaxes(xn, row_dim, col_dim))   # the collectives are right (global view)
    local = (n // p) * n * c                                                             # local elements of the row block (= of the column block)
    per_channel = local // c
    ks_seen = set()
    for width in sorted({max(1, c // 2), 2 if c >= 3 else 1, 1}):                       # pieces of `width` channels: k = ceil(c / width) → 2 pieces, an uneven cut, k == c
        limit = per_channel * width + 1                                                  # the widest slice under this limit is exactly `width` channels
        assert limit <= local                                                            # so the split engages
        k = -(-c // width)
        shard.a2a_reset()
        cols_, back_, tb_ = _run(p, x, limit, row_dim, col_dim, monkeypatch)
        rec = shard.a2a_record()
        assert rec["calls"] == 3 and rec["kmax"] == k, (rec, k, width)                   # rows_to_cols + cols_to_rows + transpose_block each split in k pieces
        assert shard.a2a_fields() == {"a2a_split": k}
        assert np.array_equal(cols_, ref_cols), "rows_to_cols split k=%d differs (P=%d %s %s)" % (k, p, dtype, layout)
        assert np.array_equal(back_, ref_back), "cols_to_rows split k=%d differs (P=%d %s %s)" % (k, p, dtype, layout)
        assert np.array_equal(tb_, ref_tb), "transpose_block split k=%d differs (P=%d %s %s)" % (k, p, dtype, layout)
        ks_seen.add(k)
    assert len(ks_seen) >= (2 if c >= 3 else 1)
    shard.a2a_reset()


@needs_jax
@pytest.mark.parametrize("p", PS[:1])
def test_split_inside_jit_and_line_word(p, monkeypatch):
    """Under jax.jit (the kit's program) the split traces the same way, the result is bitwise the single call's, and the family's state=on line carries
    a2a_split=<k> — while a line rendered before any split (or at state=off) does not."""
    shard.a2a_reset()
    n, c = 8 * p, 6
    x = _values((n, n, c), "float32")
    rm, ax = _rm(p), "row"
    rows = shard.rows_spec(ax)

    def prog(limit_env):
        if limit_env is None:
            monkeypatch.delenv(shard.A2A_MAX_ELEMS_ENV, raising=False)
        else:
            monkeypatch.setenv(shard.A2A_MAX_ELEMS_ENV, str(limit_env))
        f = shard.shard_map(lambda xl: shard.transpose_block(xl, ax) * 2.0 + 1.0, rm, (rows,), rows)
        return np.asarray(jax.jit(f)(x))

    ref = prog(None)
    line_kw = dict(rmesh=rm, schedule="gather", kernel="jnp", kernel_reason="none", sites=("a", "b"))
    assert " a2a_split=" not in evidence.line("kit-opt", "on", p, **dict(line_kw))
    got = prog((n // p) * n * 2 + 1)                                                     # 2-channel pieces → k = 3
    assert np.array_equal(ref, got) and np.array_equal(ref, np.swapaxes(np.asarray(x), 0, 1) * 2.0 + 1.0)
    assert shard.a2a_record() == {"kmax": 3, "calls": 1}
    on = evidence.line("kit-opt", "on", p, **dict(line_kw))
    assert " a2a_split=3 " in on and on.index(" a2a_split=3 ") < on.index(" schedule=gather"), on
    assert on.endswith(" schedule=gather kernel=jnp kernel_reason=none sites=a,b")
    assert "a2a_split" not in evidence.line("kit-opt", "off", 1, reason="n_gpu=1")
    shard.a2a_reset()
