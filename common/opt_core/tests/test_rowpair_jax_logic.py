"""opt_core.mem.rowpair_jax — logic and numerics tests of the row-sharded pair stack primitives.

No jax is installed in the core's test interpreter: every test here skips BY NAME without jax. They run for real on a CPU box with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (four host devices; every P in {1,2,4}) and on a multi-GPU box (P up to the device
count). Assertions: refusal words exact; ``moves_bytes``-class primitives BIT-EXACT; ``complete_contraction`` / ``reordered`` within the stated
tolerance (``ROWPAIR_TOL`` relative, default 1e-4; ``ROWPAIR_TOL_REORDERED`` 1e-5) with the measured max-abs-diff of every case written to
``$ROWPAIR_RESULTS`` (json) for the qualification table. The import-hygiene test runs everywhere (a subprocess without jax imported).
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)

_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")             # four host devices for THIS process's jax backend on a CPU box (harmless on gpu):
if "xla_force_host_platform_device_count" not in (_XLA_FLAGS_BEFORE or ""):   # set, initialise the backend, restore — the variable never leaks into the session
    os.environ["XLA_FLAGS"] = ((_XLA_FLAGS_BEFORE or "") + " --xla_force_host_platform_device_count=4").strip()
try:
    import jax as _jax_backend_init  # noqa: E402
    _jax_backend_init.devices()
except Exception:  # noqa: BLE001 — no jax in this interpreter: the device tests below skip by name
    pass
finally:
    if _XLA_FLAGS_BEFORE is None:
        os.environ.pop("XLA_FLAGS", None)
    else:
        os.environ["XLA_FLAGS"] = _XLA_FLAGS_BEFORE

RESULTS = {"cases": [], "env": {}}


def test_package_is_stdlib_at_import():
    """Clean-interpreter contract: with jax / haiku / numpy / the model libraries / torch NOT importable, `import opt_core`, `opt_core.mem` and every
    rowpair_jax module (recipes included) import, pull none of them, and the recipes / mesh refuse BY NAME (MemLeverRefused) instead of ImportError."""
    code = ("import sys; sys.path.insert(0, %r); import importlib, importlib.abc\n"
            "class Deny(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('jax','jaxlib','numpy','haiku','alphafold','alphafold3','colabfold','torch'):\n"
            "            raise ImportError('clean interpreter: ' + name + ' is not installed here')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Deny())\n"
            "import opt_core, opt_core.mem\n"
            "mods = ['opt_core.mem.rowpair_jax.' + m for m in ('mesh','shard','trimul','triatt','transition','evidence','haiku','_lazy','rowchunk','alphafold','alphafold3')]\n"
            "[importlib.import_module(m) for m in ['opt_core.mem.rowpair_jax'] + mods]\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('jax','jaxlib','numpy','haiku','alphafold','alphafold3','colabfold','torch'))\n"
            "from opt_core.mem import MemLeverRefused\n"
            "from opt_core.mem.rowpair_jax import alphafold3, rowchunk, alphafold, mesh\n"
            "refused = []\n"
            "for name, call in (('alphafold3.library', alphafold3.library), ('rowchunk.check', rowchunk.check), ('mesh.build', lambda: mesh.build(2)),\n"
            "                   ('alphafold.install', lambda: alphafold.install(type('RM', (), {'n_gpu': 2, 'axis': 'row'})(), object()))):\n"
            "    try:\n        call()\n    except MemLeverRefused as e:\n        refused.append(name + ':' + e.lever)\n"
            "print('HEAVY=' + ','.join(bad)); print('REFUSED=' + ','.join(refused)); sys.exit(1 if bad or len(refused) != 4 else 0)\n") % CORE_DIR
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_refusal_words_without_jax_semantics():
    """The gates that need no device: explicit P, the mode gate, the extent gate, the active tokens — exact words."""
    sys.path.insert(0, CORE_DIR) if CORE_DIR not in sys.path else None
    from opt_core.mem import MemLeverRefused
    from opt_core.mem.rowpair_jax import evidence, mesh, shard, triatt
    from opt_core.mem import ngpu
    for bad in (None, "", "auto", "all", 0, -1, "x", 2.5):
        with pytest.raises(ValueError):                                  # a usage error (opt_core.mem.ngpu.check_n_gpu is the producer)
            mesh.explicit_n_gpu(bad)
    assert mesh.explicit_n_gpu("4") == 4 and mesh.explicit_n_gpu(2) == 2
    assert mesh.refuse_mode("big", 4) == 4 and mesh.refuse_mode("exact", 1) == 1 and mesh.refuse_mode("fast", "1") == 1
    for m in ("exact", "fast", "off"):
        with pytest.raises(MemLeverRefused) as ei:
            mesh.refuse_mode(m, 2)
        assert ei.value.lever == "n_gpu" and ei.value.reason == ngpu.REFUSE_MODE
    assert evidence.active_fields(3) == ngpu.active_pairs(3) and evidence.active_text(3) == ngpu.active_fields(3) and evidence.SCHEMES == ngpu.SCHEMES
    assert shard.next_multiple(1000, 8) == 1000 and shard.next_multiple(1001, 8) == 1008 and shard.next_multiple(0, 3) == 0
    assert shard.local_extent(1024, 4) == 256
    with pytest.raises(MemLeverRefused) as ei:
        shard.local_extent(1001, 4)
    assert ei.value.reason == "refused: num_rows=1001 not divisible by n_gpu=4 (pad or bucket to 1004)"
    assert evidence.active_fields(1) == [("n_gpu", 1), ("sharding", "none")]
    assert evidence.active_fields(8) == [("n_gpu", 8), ("sharding", "rowpair")]
    assert evidence.active_text(2) == "n_gpu=2 sharding=rowpair" and evidence.active_text(1) == "n_gpu=1 sharding=none"
    assert evidence.active_fields(4, "foldcp2d") == [("n_gpu", 4), ("sharding", "foldcp2d")]
    with pytest.raises(ValueError):
        evidence.active_fields(2, "columns")
    line = evidence.line("kit-opt", "off", 1, reason="n_gpu=1")
    assert line.startswith("[kit-opt] LEVER name=rowpair state=off reason=n_gpu=1 impl=opt_core.mem.rowpair_jax@") and " origin=core n_gpu=1 sharding=none prealloc=" in line
    env = mesh.xla_memory_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_CLIENT_MEM_FRACTION": "0.85"})
    assert env == {"prealloc": "false", "mem_fraction": "unset", "client_mem_fraction": "0.85", "allocator": "unset", "xla_effective_fraction": "0.85"}
    assert mesh.xla_memory_env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8", "XLA_CLIENT_MEM_FRACTION": "0.9"})["xla_effective_fraction"] == "0.9"   # peak's precedence table
    from opt_core import arch as _arch                    # the shared id is declared by the core (mem.ngpu's owner); this package only reads the registry
    assert _arch.supports(evidence.TP_LEVER_ID, "sm90").word.split(":")[0] in ("undeclared", "uncertified", "supported")
    class _M:                                             # card_fields reads lever_state for the mesh's sm
        facts = [{"sm": "sm90"}]
    cf = evidence.card_fields(_M()); assert cf["card_state"] in ("on", "off") and "card_support" in cf
    class _StubMesh:                                   # the gate reads the LIVE pool fraction off the mesh facts (bytes_limit / device total)
        lever = "rowpair"
        def __init__(self, n, limit, total):
            self.n_gpu = n
            self.facts = [{"id": i, "bytes_limit": limit, "total_bytes": total} for i in range(n)]
        pool_fraction = mesh.RowMesh.pool_fraction
    assert mesh.mem_fraction_gate(_StubMesh(4, 76, 80), {8: 0.85}, {}) == {"mem_fraction_ceiling": "none", "xla_pool_fraction": 0.95}
    assert mesh.mem_fraction_gate(_StubMesh(8, 64, 80), {8: 0.85}, {})["mem_fraction_ceiling"] == 0.85
    with pytest.raises(MemLeverRefused) as ei:            # live pool 0.95 > ceiling, whatever the env string says
        mesh.mem_fraction_gate(_StubMesh(8, 76, 80), {8: 0.85}, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"})
    assert ei.value.reason.startswith("refused: n_gpu=8 needs the XLA pool fraction <= 0.85 (live 0.950; env XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 ")
    with pytest.raises(MemLeverRefused) as ei:            # unreadable live fraction under an applicable ceiling: fail-closed
        mesh.mem_fraction_gate(_StubMesh(8, None, None), {8: 0.85}, {})
    assert "unreadable" in ei.value.reason
    with pytest.raises(MemLeverRefused) as ei:            # P=1 builds no mesh (structural)
        mesh.build(1)
    assert ei.value.lever == "n_gpu" and ei.value.reason == mesh.REFUSE_P1
    assert shard.pad_plan(1500, 8) == {"n": 1500, "n_padded": 1504, "pad": 4, "n_loc": 188} and shard.pad_plan(4000, 4)["pad"] == 0
    assert all(shard.pad_plan(b, q)["n_padded"] % q == 0 for b in (1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000) for q in (2, 4, 8))
    with pytest.raises(ValueError):
        evidence.line("kit-opt", "on", 2, schedule="gather", kernel="jnp", kernel_reason="none")      # state=on is fail-closed: rmesh + sites missing
    assert triatt.kernel_gate(256, 1024) == (True, "unconstrained")
    assert triatt.kernel_gate(256, 1024, {"row_multiple": 128, "min_cols": 16}) == (True, "fits")
    assert triatt.kernel_gate(250, 1000, {"row_multiple": 128}) == (False, "row_multiple=128:250")
    assert triatt.kernel_gate(256, 1024, {"bogus": 1}) == (False, "unknown_constraint=bogus")
    assert triatt.kernel_fields("pallas_attn", False, "row_multiple=128:250") == {"kernel": "jnp", "kernel_reason": "row_multiple=128:250"}
    pf = evidence.peak_fields([{"id": 0, "kind": "k", "peak_bytes_in_use": 3_000_000_000, "bytes_in_use": 1, "bytes_limit": 80},
                               {"id": 1, "kind": "k", "peak_bytes_in_use": None, "bytes_in_use": None, "bytes_limit": None}])
    assert pf == {"xla_peak_bytes": "d0:3000000000", "xla_peak_gb_max": "3.000", "peak_scope": "xla_allocator"}
    assert evidence.peak_fields([])["xla_peak_bytes"] == "unavailable"
    assert trimul.transient_bytes(1024, 128, 2, 4, "ring", "incoming")["blocks"] == 4 and trimul.transient_bytes(1024, 128, 2, 4, "ring")["blocks"] == 3
    assert set(r[3] for r in evidence.plan_rows()) <= set(evidence.CLASSES)


# ================================================================================================ device tests (jax required)
try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402  (comes with jax)
    HAVE_JAX, _JAX_WHY = True, ""
except Exception as _e:  # noqa: BLE001 — the core's own test interpreter has no jax: the device tests skip by name
    HAVE_JAX, _JAX_WHY = False, repr(_e)
needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="rowpair_jax device tests need jax (%s): run on a CPU box with xla_force_host_platform_device_count=4 or a multi-GPU box" % _JAX_WHY)

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
from opt_core.mem import MemLeverRefused  # noqa: E402
from opt_core.mem.rowpair_jax import evidence, mesh, shard, transition, triatt, trimul  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402

if HAVE_JAX:
    PLATFORM = jax.devices()[0].platform
    NDEV = jax.device_count()
    TOL_REL = {"float32": float(os.environ.get("ROWPAIR_TOL_F32", "1e-5")),                        # row_local / complete_contraction at exact-f32 matmul precision, relative to max|ref|;
               "float32:default": float(os.environ.get("ROWPAIR_TOL_F32_DEFAULT", str(2.0 ** -10))),   # f32 operands at the GPU backend's DEFAULT matmul precision
               "bfloat16": float(os.environ.get("ROWPAIR_TOL_BF16", str(2.0 ** -7)))}             # (tf32-eligible; cpu = exact f32); bf16. Every case's max|d| is RECORDED.
    TOL_REORDERED = float(os.environ.get("ROWPAIR_TOL_REORDERED", "1e-5"))                          # psum means: relative, any platform
    RESULTS["env"] = {"platform": PLATFORM, "device_count": NDEV, "device_kind": str(jax.devices()[0].device_kind), "jax": jax.__version__,
                      "shard_map": shard.shard_map_flavour(), "tol_rel": TOL_REL, "tol_reordered": TOL_REORDERED, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                      **mesh.xla_memory_env()}
else:
    PLATFORM, NDEV, TOL_REL, TOL_REORDERED = "none", 1, {"float32": 0.0, "float32:default": 0.0, "bfloat16": 0.0}, 0.0
PS = [p for p in (1, 2, 4, 8) if p <= NDEV]


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path:
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True)


def _record(name, p, cls, ref, got, **kw):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    mad = float(np.max(np.abs(ref - got))) if ref.size else 0.0
    scale = float(np.max(np.abs(ref))) if ref.size else 1.0
    bitwise = bool(np.array_equal(ref, got))
    RESULTS["cases"].append({"case": name, "P": p, "class": cls, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, **kw})
    return mad, scale, bitwise


def _tol_key(dtype, precision):
    """f32 at the backend's default matmul precision on a GPU is tf32-eligible (its own tolerance class); f32 at HIGHEST and f32 on cpu share the f32 class."""
    if dtype == "float32" and precision == "default" and PLATFORM == "gpu":
        return "float32:default"
    return dtype


def _hold(name, p, cls, ref, got, dtype="float32", precision="default", **kw):
    """moves_bytes: BIT-EXACT (the only exact class). row_local / complete_contraction: max|d| <= TOL_REL[dtype(:precision)] * max|ref|; reordered:
    TOL_REORDERED. Every case's measured max|d|, bit-exact flag, dtype and matmul precision is recorded whatever the class."""
    mad, scale, bitwise = _record(name, p, cls, ref, got, dtype=dtype, precision=precision, **kw)
    if cls == evidence.CLASS_MOVES_BYTES:
        assert bitwise, "%s P=%d class=%s: not bitwise (max|d|=%g)" % (name, p, cls, mad)
    else:
        key = _tol_key(dtype, precision)
        tol = TOL_REORDERED if cls == evidence.CLASS_REORDERED else TOL_REL[key]
        assert mad <= tol * scale, "%s P=%d class=%s tol_class=%s: max|d|=%g > tol %g*%g" % (name, p, cls, key, mad, tol, scale)


def _rm(p):
    return mesh.build(p, platform=PLATFORM, _allow_single_device_mesh=True)     # the platform pinned to what the box runs; P=1 meshes are a test-only hook


def _rng(seed, shape, dtype="float32"):
    return np.random.RandomState(seed).standard_normal(shape).astype(dtype)


# ------------------------------------------------------------------------------------------------ mesh
@needs_jax
def test_mesh_build_and_refusals():
    k = NDEV
    with pytest.raises(MemLeverRefused) as ei:
        mesh.build(k + 1, platform=PLATFORM)
    assert ei.value.lever == "n_gpu" and ei.value.reason == "refused: n_gpu=%d visible=%d" % (k + 1, k)
    with pytest.raises(MemLeverRefused) as ei:
        mesh.build(3, platform=PLATFORM, devices=jax.devices()[:2], lever="mykit_tp")
    assert ei.value.lever == "n_gpu" and ei.value.reason == "refused: n_gpu=3 visible=%d" % len(jax.devices()[:2])   # 2 on the CPU / multi-GPU box, 1 on a 1-GPU box
    with pytest.raises(ValueError):
        mesh.build("auto", platform=PLATFORM)
    other = "cpu" if PLATFORM == "gpu" else "gpu"
    with pytest.raises(MemLeverRefused) as ei:            # the platform pin: a jax that fell back to another backend is refused by name
        mesh.build(2, platform=other)
    assert ei.value.reason == "refused: n_gpu=2 platform=%s expected=%s" % (PLATFORM, other), ei.value.reason
    with pytest.raises(MemLeverRefused) as ei:
        mesh.build(1, platform=PLATFORM)
    assert ei.value.reason == mesh.REFUSE_P1
    for p in PS:
        rm = _rm(p)
        assert rm.n_gpu == p and rm.axis == "row" and rm.visible == k and [d.id for d in rm.devices] == [d.id for d in jax.devices()[:p]]
        assert tuple(rm.mesh.devices.shape) == (p,) and rm.mesh.axis_names == ("row",)
        d = rm.describe()
        assert d["n_gpu"] == p and d["visible"] == k and d["platform"] == PLATFORM and " " not in d["device_kind"] and d["devices"] == ",".join("d%d" % x.id for x in rm.devices)
        assert all(set(f) >= {"id", "platform", "kind", "bytes_limit", "peak_bytes_in_use", "total_bytes", "sm"} for f in rm.facts)
        if PLATFORM == "gpu":
            assert all(isinstance(f["bytes_limit"], int) and f["bytes_limit"] > 0 for f in rm.facts), rm.facts
            assert all(isinstance(f["total_bytes"], int) and f["total_bytes"] >= f["bytes_limit"] for f in rm.facts), rm.facts
            assert all(str(f["sm"]).startswith("sm") for f in rm.facts), rm.facts
            frac = rm.pool_fraction()
            assert frac is not None and 0.0 < frac <= 1.0
            RESULTS["env"]["xla_pool_fraction_live"] = frac
            assert mesh.mem_fraction_gate(rm, {p: 1.0})["mem_fraction_ceiling"] == 1.0
            with pytest.raises(MemLeverRefused):
                mesh.mem_fraction_gate(rm, {p: max(frac - 0.05, 0.01)})
    line = evidence.line("kit-opt", "on", PS[-1], rmesh=_rm(PS[-1]), peaks=evidence.device_peaks(_rm(PS[-1])), schedule="gather", kernel="jnp",
                         kernel_reason="no_fused_kernel_in_suite", sites=("triangle_multiplication_outgoing", "pair_transition"))
    assert (" n_gpu=%d sharding=%s axis=row visible=%d platform=%s devices=" % (PS[-1], "rowpair" if PS[-1] > 1 else "none", k, PLATFORM)) in line
    assert " xla_peak_bytes=" in line and " peak_scope=xla_allocator " in line and " shard_map=" in line
    assert line.endswith(" schedule=gather kernel=jnp kernel_reason=no_fused_kernel_in_suite sites=triangle_multiplication_outgoing,pair_transition"), line
    assert line.count(" ") == len(line.split()) - 1
    RESULTS["env"]["lever_line_example"] = line


# ------------------------------------------------------------------------------------------------ collectives
@needs_jax
@pytest.mark.parametrize("p", PS)
def test_collectives_roundtrip_bitwise(p):
    rm, ax = _rm(p), "row"
    n, c = 8 * p, 3
    x = _rng(1, (n, n, c))
    rows, rep = shard.rows_spec(ax), shard.replicated_spec()

    def body(xl, xfull):
        nl = xl.shape[0]
        g = shard.gather(xl, ax, 0)                                   # == full
        cols = shard.rows_to_cols(xl, ax)                             # [n, n/p, c] = my column block
        back = shard.cols_to_rows(cols, ax)                           # == xl
        tb = shard.transpose_block(xl, ax)                            # rows of x^T
        lb = shard.local_block(xfull, ax, nl, 0)                      # == xl
        me = shard.axis_index(ax)
        colref = jax.lax.dynamic_slice_in_dim(xfull, me * nl, nl, axis=1)
        ring = shard.ppermute_shift(xl, ax, 1)                        # I receive (me-1)'s rows
        return g, cols, colref, back, tb, lb, ring

    outs = shard.shard_map(body, rm, (rows, rep), (rep, rows, rows, rows, rows, rows, rows))(x, x)
    g, cols, colref, back, tb, lb, ring = [np.asarray(o) for o in outs]
    _hold("gather", p, evidence.CLASS_MOVES_BYTES, x, g)
    _hold("rows_to_cols", p, evidence.CLASS_MOVES_BYTES, colref, cols)
    _hold("cols_to_rows∘rows_to_cols", p, evidence.CLASS_MOVES_BYTES, x, back)
    _hold("transpose_block", p, evidence.CLASS_MOVES_BYTES, np.swapaxes(x, 0, 1), tb)
    _hold("local_block", p, evidence.CLASS_MOVES_BYTES, x, lb)
    nl = n // p
    expect_ring = np.concatenate([x[((d - 1) % p) * nl:((d - 1) % p + 1) * nl] for d in range(p)], axis=0)
    _hold("ppermute_shift", p, evidence.CLASS_MOVES_BYTES, expect_ring, ring)


# ------------------------------------------------------------------------------------------------ trimul
@needs_jax
@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("schedule", trimul.SCHEDULES)
@pytest.mark.parametrize("equation", sorted(trimul.EQUATIONS))
@pytest.mark.parametrize("dtype,precision", [("float32", "highest"), ("float32", "default"), ("bfloat16", "default")])
def test_trimul_contract_equals_dense(p, schedule, equation, dtype, precision):
    rm, ax = _rm(p), "row"
    prec = jax.lax.Precision.HIGHEST if precision == "highest" else None
    direction, row_dim, col_dim = trimul.classify(equation)
    n, c = 16 * p, 5
    shape = [0, 0, 0]
    shape[row_dim], shape[col_dim] = n, n
    shape[[i for i in range(3) if i not in (row_dim, col_dim)][0]] = c
    a = jnp.asarray(_rng(2, tuple(shape)), dtype=dtype)
    b = jnp.asarray(_rng(3, tuple(shape)), dtype=dtype)
    dense = jnp.einsum(equation, a, b, precision=prec)
    rows = shard.rows_spec(ax, 3, row_dim)
    f = shard.shard_map(lambda al, bl: trimul.contract(equation, al, bl, ax, schedule=schedule, precision=prec), rm, (rows, rows), rows)
    got = jax.jit(f)(a, b)
    assert got.shape == dense.shape and got.dtype == dense.dtype
    _hold("trimul[%s,%s,%s,%s]" % (equation, schedule, dtype, precision), p, evidence.CLASS_COMPLETE, np.asarray(dense.astype(jnp.float32)), np.asarray(got.astype(jnp.float32)),
          dtype=dtype, precision=precision, direction=direction)


@needs_jax
def test_trimul_refusals():
    with pytest.raises(MemLeverRefused) as ei:
        trimul.classify("abc,abd->acd")
    assert "is not one of" in ei.value.reason
    rm = _rm(PS[-1])
    rows = shard.rows_spec("row")
    a = jnp.zeros((4 * PS[-1], 4 * PS[-1], 2))
    with pytest.raises(MemLeverRefused):
        shard.shard_map(lambda al, bl: trimul.contract("ikc,jkc->ijc", al, bl, "row", schedule="spiral"), rm, (rows, rows), rows)(a, a)
    tb = trimul.transient_bytes(4096, 128, 2, 4, "gather", "incoming")
    assert tb["full_operand"] == 4096 * 4096 * 128 * 2 and tb["column_block"] == tb["full_operand"] // 4
    assert trimul.transient_bytes(4096, 128, 2, 4, "gather")["column_block"] == 0
    assert trimul.transient_bytes(4096, 128, 2, 4, "ring")["total"] == 3 * (4096 * 4096 * 128 * 2 // 4)


# ------------------------------------------------------------------------------------------------ triatt + transition
def _attention_rows(x, bias_hnn, mask_rows):
    """A plain per-row attention (starting node): q,k,v from row i; logits [h, i, j, k] = q·k + bias[h, j, k]; mask on keys."""
    h = bias_hnn.shape[0]
    c = x.shape[-1]
    q = x.reshape(x.shape[0], x.shape[1], h, c // h)
    logits = jnp.einsum("iqhc,ikhc->hiqk", q, q) / np.sqrt(c // h) + bias_hnn[:, None, :, :]
    logits = jnp.where(mask_rows[None, :, None, :] > 0, logits, -1e9)
    w = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum("hiqk,ikhc->iqhc", w, q).reshape(x.shape)
    return o


@needs_jax
@pytest.mark.parametrize("p", PS)
def test_triatt_start_and_end_equal_dense(p):
    rm, ax = _rm(p), "row"
    n, c, h = 8 * p, 8, 2
    x = jnp.asarray(_rng(4, (n, n, c)))
    wb = jnp.asarray(_rng(5, (c, h)))
    mask = jnp.asarray((np.random.RandomState(6).rand(n, n) > 0.2).astype(np.float32))
    # dense references
    bias = jnp.transpose(jnp.einsum("ijc,ch->ijh", x, wb), (2, 0, 1))
    start_ref = _attention_rows(x, bias, mask)
    xt = jnp.swapaxes(x, 0, 1)
    bias_t = jnp.transpose(jnp.einsum("ijc,ch->ijh", xt, wb), (2, 0, 1))
    end_ref = jnp.swapaxes(_attention_rows(xt, bias_t, jnp.swapaxes(mask, 0, 1)), 0, 1)
    rows, rep = shard.rows_spec(ax), shard.replicated_spec()

    def start_body(xl, mfull):
        nl = xl.shape[0]
        b = jnp.transpose(triatt.bias_full(jnp.einsum("ijc,ch->ijh", xl, wb), ax), (2, 0, 1))
        return _attention_rows(xl, b, triatt.mask_rows(mfull, ax, nl))

    def end_body(xl, mfull):
        nl = xl.shape[0]
        xlt = triatt.enter_transposed(xl, ax)
        b = jnp.transpose(triatt.bias_full(jnp.einsum("ijc,ch->ijh", xlt, wb), ax), (2, 0, 1))
        return triatt.exit_transposed(_attention_rows(xlt, b, triatt.mask_cols(mfull, ax, nl)), ax)

    start = jax.jit(shard.shard_map(start_body, rm, (rows, rep), rows))(x, mask)
    end = jax.jit(shard.shard_map(end_body, rm, (rows, rep), rows))(x, mask)
    cls = evidence.CLASS_COMPLETE                   # the toy attention's own GEMMs tile by the row count (the primitives inside are moves_bytes, held bit-exact above/below)
    _hold("triatt_starting", p, cls, np.asarray(start_ref), np.asarray(start))
    _hold("triatt_ending", p, cls, np.asarray(end_ref), np.asarray(end))
    bias_rt = jax.jit(shard.shard_map(lambda xl: triatt.bias_full(jnp.transpose(xl[..., :h], (2, 0, 1)), ax, row_dim=1), rm, (rows,), rep))(x)   # [H, N/P, N] -> [H, N, N]: data movement only
    _hold("bias_full", p, evidence.CLASS_MOVES_BYTES, np.asarray(jnp.transpose(x[..., :h], (2, 0, 1))), np.asarray(bias_rt))
    rows2 = shard.rows_spec(ax, 2)
    masks = shard.shard_map(lambda mf: (triatt.mask_rows(mf, ax, n // p), triatt.mask_cols(mf, ax, n // p)), rm, (rep,), (rows2, rows2))(mask)
    _hold("mask_rows", p, evidence.CLASS_MOVES_BYTES, np.asarray(mask), np.asarray(masks[0]))
    _hold("mask_cols", p, evidence.CLASS_MOVES_BYTES, np.asarray(jnp.swapaxes(mask, 0, 1)), np.asarray(masks[1]))
    rt = jax.jit(shard.shard_map(lambda xl: triatt.exit_transposed(triatt.enter_transposed(xl, ax), ax), rm, (rows,), rows))(x)
    _hold("enter∘exit_transposed", p, evidence.CLASS_MOVES_BYTES, np.asarray(x), np.asarray(rt))


@needs_jax
@pytest.mark.parametrize("p", PS)
def test_transition_consumers_equal_dense(p):
    rm, ax = _rm(p), "row"
    n, s, c, co = 8 * p, 3, 4, 6
    x = jnp.asarray(_rng(7, (n, n, c)))
    left = jnp.asarray(_rng(8, (s, n, c)))
    right = jnp.asarray(_rng(9, (s, n, c)))
    m = jnp.asarray((np.random.RandomState(10).rand(s, n, 1) > 0.3).astype(np.float32))
    wo = jnp.asarray(_rng(11, (c, c, co)))
    rows, rep = shard.rows_spec(ax), shard.replicated_spec()
    # outer-product mean (the AlphaFold form: act[i,j] = sum_s left[s,i] (x) right[s,j], normalised by the mask outer product)
    def opm_dense(lf, lm, rf, rm_):
        outer = jnp.einsum("sia,sjb->ijab", lf * lm, rf * rm_)
        norm = jnp.einsum("sia,sja->ij", lm, rm_)[:, :, None]
        return jnp.einsum("ijab,abo->ijo", outer, wo) / (1e-3 + norm)
    opm_ref = opm_dense(left, m, right, m)
    def opm_body(lf, lm, rf, rm_, xl):
        nl = xl.shape[0]
        lfl, lml = transition.opm_operands(lf, lm, ax, nl, res_dim=1)
        return opm_dense(lfl, lml, rf, rm_)
    opm = jax.jit(shard.shard_map(opm_body, rm, (rep, rep, rep, rep, rows), rows))(left, m, right, m, x)
    _hold("outer_product_mean", p, evidence.CLASS_COMPLETE, np.asarray(opm_ref), np.asarray(opm))     # the toy OPM's einsums tile by rows; the operand slicing is moves_bytes:
    ops = shard.shard_map(lambda lf, lm, xl: transition.opm_operands(lf, lm, ax, xl.shape[0], res_dim=1), rm, (rep, rep, rows), (shard.rows_spec(ax, 3, 1), shard.rows_spec(ax, 3, 1)))(left, m, x)
    _hold("opm_operands", p, evidence.CLASS_MOVES_BYTES, np.asarray(left), np.asarray(ops[0]))
    # pair logits gather, symmetrize, rows_full, masked mean
    wl = jnp.asarray(_rng(12, (c, 2)))
    logits_ref = jnp.einsum("ijc,ch->ijh", x, wl)
    outs = jax.jit(shard.shard_map(lambda xl: (transition.pair_logits_full(jnp.einsum("ijc,ch->ijh", xl, wl), ax), transition.symmetrize(xl, ax),
                                                transition.rows_full(xl, ax)), rm, (rows,), (rep, rows, rep)))(x)
    _hold("pair_logits_full", p, evidence.CLASS_MOVES_BYTES, np.asarray(logits_ref), np.asarray(outs[0]))
    _hold("symmetrize", p, evidence.CLASS_MOVES_BYTES, np.asarray(x + jnp.swapaxes(x, 0, 1)), np.asarray(outs[1]))
    _hold("rows_full", p, evidence.CLASS_MOVES_BYTES, np.asarray(x), np.asarray(outs[2]))
    pm = jnp.asarray((np.random.RandomState(13).rand(n, n, 1) > 0.5).astype(np.float32))
    mm_ref = jnp.sum(pm * jnp.ones_like(x) * x) / (jnp.sum(pm * jnp.ones_like(x)) + 1e-10)          # the stock mask-mean form with its eps
    mm = jax.jit(shard.shard_map(lambda xl, ml: transition.masked_mean(xl, ml * jnp.ones_like(xl), ax, 1e-10), rm, (rows, rows), rep))(x, pm)
    xp = shard.pad_pair(x, shard.pad_plan(n + 3, p)["n_padded"] if (n + 3) % p else n + 3 + p)      # padding: original block bit-exact, cut back bit-exact
    _hold("pad_unpad", p, evidence.CLASS_MOVES_BYTES, np.asarray(x), np.asarray(shard.unpad_pair(xp, n)))
    assert int(xp.shape[0]) % p == 0 and int(xp.shape[0]) == int(xp.shape[1]) and float(jnp.abs(xp[n:]).max()) == 0.0
    _hold("masked_mean", p, evidence.CLASS_REORDERED, np.asarray(mm_ref).reshape(1), np.asarray(mm).reshape(-1)[:1])
    # carry constraint under jit: the pair leaf comes out row-sharded, the other leaf untouched
    carry = jax.jit(lambda t: transition.constrain_carry(t, rm, ("pair",)))({"pair": x, "single": left})
    spec = carry["pair"].sharding.spec if hasattr(carry["pair"], "sharding") else None
    RESULTS["env"]["carry_spec_P%d" % p] = str(spec)
    if p > 1:
        assert tuple(spec)[0] == "row", spec
    _hold("constrain_carry", p, evidence.CLASS_MOVES_BYTES, np.asarray(x), np.asarray(carry["pair"]))


# ------------------------------------------------------------------------------------------------ haiku recipe
@needs_jax
def test_haiku_recipe_sharded_model_equals_dense():
    hk = pytest.importorskip("haiku", reason="the haiku recipe test needs dm-haiku")
    p = PS[-1]
    rm, ax = _rm(p), "row"
    n, c = 8 * p, 8

    def _epilogue(out):
        """A row-chunked epilogue in the stock style: the body once whole at init (parameters cannot be created inside hk.scan), an hk.scan over
        row blocks at apply (hk.scan threads Haiku state — the rng-leak fact the region's body_frame answers)."""
        def chunk(carry, blk):
            return carry, hk.Linear(c, name="out")(hk.LayerNorm(-1, True, True, name="ln_out")(blk))
        if hk.running_init():
            return chunk(None, out)[1]
        _, ys = hk.scan(chunk, None, out.reshape((2, out.shape[0] // 2) + out.shape[1:]))
        return ys.reshape(out.shape)

    class TriMul(hk.Module):                       # a stock-shaped sub-layer: LN, projections, ONE cross-row einsum, a row-chunked output projection
        def __init__(self, equation, name=None):
            super().__init__(name=name)
            self.equation = equation

        def __call__(self, act, mask):
            act = hk.LayerNorm(-1, True, True, name="ln_in")(act)
            left = mask[..., None] * hk.Linear(c, name="left")(act)
            right = mask[..., None] * hk.Linear(c, name="right")(act)
            out = jnp.einsum(self.equation, left, right)
            return _epilogue(out)

    class Block(hk.Module):
        def __call__(self, pair, mask):
            pair = pair + TriMul("ikc,jkc->ijc", name="tri_out")(pair, mask)
            pair = pair + TriMul("kjc,kic->ijc", name="tri_in")(pair, mask)
            return pair + hk.Linear(c, name="transition")(jax.nn.relu(hk.Linear(2 * c, name="transition_in")(pair)))

    def forward(pair, mask):
        return Block(name="block")(pair, mask)

    model = hk.transform(forward)
    pair = jnp.asarray(_rng(20, (n, n, c)))
    mask = jnp.asarray((np.random.RandomState(21).rand(n, n) > 0.1).astype(np.float32))
    key = jax.random.PRNGKey(0)
    params = model.init(key, pair, mask)
    dense = jax.jit(model.apply)(params, key, pair, mask)
    names_before = sorted(params)

    # --- the adapter: rebind TriMul.__call__ (sharded inside a region, stock outside) and Block.__call__ (the region)
    patches = rp_hk.PatchSet("rowpair")
    STOCK_TRIMUL = rp_hk.stock_body(TriMul, "__call__")

    def trimul_rows(self, act, mask_rows):
        if not rp_hk.in_region():
            return STOCK_TRIMUL(self, act, mask_rows)
        act = hk.LayerNorm(-1, True, True, name="ln_in")(act)
        left = mask_rows[..., None] * hk.Linear(c, name="left")(act)
        right = mask_rows[..., None] * hk.Linear(c, name="right")(act)
        out = trimul.contract(self.equation, left, right, ax)
        return _epilogue(out)

    got_stock = rp_hk.rebind(patches, TriMul, "__call__", trimul_rows)
    assert got_stock is STOCK_TRIMUL
    STOCK_BLOCK = rp_hk.stock_body(Block, "__call__")
    rows, rep = shard.rows_spec(ax), shard.replicated_spec()

    def block_rows(self, pair_, mask_):
        def inner(pl, mfull):
            return STOCK_BLOCK(self, pl, triatt.mask_rows(mfull, ax, pl.shape[0]))
        return rp_hk.region(inner, rm, (rows, rep), rows)(shard.constrain(pair_, rm), mask_)

    rp_hk.rebind(patches, Block, "__call__", block_rows)
    assert sorted(patches.names()) == sorted(["TriMul.__call__", "Block.__call__"]), patches.names()
    params2 = model.init(key, pair, mask)                      # name scopes kept by wrap_method: the same parameter tree
    assert sorted(params2) == names_before, (sorted(params2), names_before)
    apply = rp_hk.jit_apply(model.apply, rm, n_args=4)
    sharded = apply(shard.put(params, rm), shard.put(key, rm), shard.put(pair, rm), shard.put(mask, rm))
    _hold("haiku_block[trimul out+in, transition, hk.scan epilogue]", p, evidence.CLASS_COMPLETE, np.asarray(dense), np.asarray(sharded))
    assert not rp_hk.in_region()
    # P=1 mesh through the same rebound classes vs the plain dense apply (the primitives on a 1-device mesh)
    rm1 = _rm(1)
    def block_rows1(self, pair_, mask_):
        def inner(pl, mfull):
            return STOCK_BLOCK(self, pl, triatt.mask_rows(mfull, "row", pl.shape[0]))
        return rp_hk.region(inner, rm1, (shard.rows_spec("row"), shard.replicated_spec()), shard.rows_spec("row"))(pair_, mask_)
    patches.restore()
    patches1 = rp_hk.PatchSet("rowpair")
    rp_hk.rebind(patches1, TriMul, "__call__", trimul_rows)
    rp_hk.rebind(patches1, Block, "__call__", block_rows1)
    one = rp_hk.jit_apply(model.apply, rm1, n_args=4)(params, key, pair, mask)
    _record("haiku_block P=1 mesh vs plain apply", 1, evidence.CLASS_COMPLETE, np.asarray(dense), np.asarray(one))
    patches1.restore()
    assert rp_hk.stock_body(TriMul, "__call__") is STOCK_TRIMUL           # restore() put the stock attribute back
    plain_again = jax.jit(hk.transform(forward).apply)(params, key, pair, mask)      # a FRESH transform: jax's trace cache keys on the function object
    _hold("restore→stock", p, evidence.CLASS_MOVES_BYTES, np.asarray(dense), np.asarray(plain_again))
    peaks = evidence.device_peaks(rm)
    RESULTS["env"]["peaks_after_suite"] = evidence.peak_fields(peaks)
    if PLATFORM == "gpu":
        assert all(pk["peak_bytes_in_use"] is not None and pk["peak_bytes_in_use"] > 0 for pk in peaks), peaks


def test_peak_fields_names_an_absent_peak(monkeypatch):
    """No positive allocator peak is NAMED (platform allocator / no stats / zero), never printed as 0.000."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    f = evidence.peak_fields([{"id": 0, "peak_bytes_in_use": 0}, {"id": 1, "peak_bytes_in_use": 0}])
    assert f["xla_peak_gb_max"] == "unavailable" and f["xla_peak_unavailable"] == "platform_allocator" and f["xla_peak_bytes"] == "d0:0,d1:0"
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR")
    assert evidence.peak_fields([{"id": 0, "peak_bytes_in_use": None}])["xla_peak_unavailable"] == "no_memory_stats"
    assert evidence.peak_fields([{"id": 0, "peak_bytes_in_use": 0}])["xla_peak_unavailable"] == "backend_reports_zero"
    g = evidence.peak_fields([{"id": 0, "peak_bytes_in_use": 5_000_000_000}])
    assert g["xla_peak_gb_max"] == "5.000" and "xla_peak_unavailable" not in g
    ln = evidence.line("t", "off", 1, reason="n_gpu=1", peaks=[{"id": 0, "peak_bytes_in_use": 0}])      # the word reaches the line
    assert "xla_peak_unavailable=backend_reports_zero" in ln, ln
