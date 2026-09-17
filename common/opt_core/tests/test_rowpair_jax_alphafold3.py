"""opt_core.mem.rowpair_jax.alphafold3 — body tests per site group on the AF3 image: each patched body on a 1-device mesh (test-only hook) equals
the STOCK body BIT-EXACT (transcription fidelity), and on P=2/4 row blocks equals it within the stated tolerance class; non-zero-mean inputs.
Groups covered here: trunk (TriangleMultiplication out/in, OuterProductMean, MSAAttention, PairFormerIteration = GridSelfAttention row/col +
TransitionBlock + trimul ×2), heads (DistogramHead). The model group (Model.__call__, recycle carry, heads region) and the confidence / diffusion
heads run only with weights + a featurised batch: the AF3 JAX kit's end-to-end dev check (P=2 vs P=1) is its record. Skips BY NAME without alphafold3."""
import json
import os
import sys
import types

import pytest

try:                                                                     # the model library logs through absl; an unparsed absl FLAGS would parse pytest's argv
    from absl import flags as _absl_flags
    _absl_flags.FLAGS.mark_as_parsed()
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
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

from opt_core.mem import MemLeverRefused  # noqa: E402
from opt_core.mem.patchset import PatchSet  # noqa: E402
from opt_core.mem.rowpair_jax import alphafold3 as af3, mesh, shard, triatt  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    L = af3.library()
    from alphafold3.model import model_config  # noqa: E402
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold3 (the AF3 image): %s" % WHY)
RESULTS = {"cases": [], "env": {}, "skipped_by_name": []}
TOL_F32 = float(os.environ.get("ROWPAIR_TOL_F32", "1e-5"))
TOL_F32_DEFAULT = float(os.environ.get("ROWPAIR_TOL_F32_DEFAULT", str(2.0 ** -10)))
OFFSET = float(os.environ.get("ROWPAIR_AF3_OFFSET", "40.0"))
LEVER = "rowpair"


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path and (RESULTS["cases"] or RESULTS["skipped_by_name"]):
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True, default=str)


def _gc():
    gc = model_config.GlobalConfig()
    for k, v in (("flash_attention_implementation", "xla"),):          # the attention kernel that runs on every platform
        if hasattr(gc, k):
            try:
                setattr(gc, k, v)
            except Exception:  # noqa: BLE001 — frozen config: rebuild
                gc = type(gc)(**{**gc.__dict__, k: v}) if hasattr(gc, "__dict__") else gc
    return gc


def _rm(p):
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps():
    return [1] + [p for p in (2, 4) if p <= jax.device_count()]


def _hold(name, p, ref, got, cls):
    ref, got = np.asarray(jnp.asarray(ref, jnp.float32), np.float64), np.asarray(jnp.asarray(got, jnp.float32), np.float64)
    mad, scale = float(np.max(np.abs(ref - got))), float(np.max(np.abs(ref)))
    bitwise = bool(np.array_equal(ref, got))
    tol_class = "bitwise(1-device mesh)" if p == 1 else ("float32:default" if jax.devices()[0].platform == "gpu" else "float32")
    tol = 0.0 if p == 1 else (TOL_F32_DEFAULT if tol_class == "float32:default" else TOL_F32)
    RESULTS["cases"].append({"case": name, "P": p, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, "class": cls, "tol_class": tol_class, "offset": OFFSET})
    if p == 1:
        assert bitwise, "%s on a 1-device mesh is not bitwise vs the stock body (max|d|=%g): a transcription differs from the pinned tree" % (name, mad)
    else:
        assert mad <= tol * scale, "%s P=%d tol_class=%s: max|d|=%g > %g*%g" % (name, p, tol_class, mad, tol, scale)


def _config(cls, **kw):
    """The module's nested Config (AF3 style); a construction failure is a NAMED skip of this test (recorded), never a silent pass."""
    C = getattr(cls, "Config", None)
    if C is None:
        pytest.skip("%s has no nested Config class in this tree" % cls.__name__)
    try:
        return C(**kw)
    except Exception as e:  # noqa: BLE001
        RESULTS["skipped_by_name"].append({"test": cls.__name__, "reason": "Config(**%s): %r" % (sorted(kw), e)})
        pytest.skip("%s.Config(%s): %r" % (cls.__name__, sorted(kw), e))


def _run(make_stock, make_rows, args, in_specs, out_spec, p, sites=("trunk",)):
    """params from a plain init of the stock forward; the stock apply; then the recipe installed on a P-device mesh and the rows forward inside ONE
    region entered as a row-sharded body (what PairFormerIteration/EvoformerIteration do for their sub-layers)."""
    key = jax.random.PRNGKey(0)
    params = hk.transform(make_stock).init(key, *args)
    stock = jax.jit(hk.transform(make_stock).apply)(params, key, *args)
    rm = _rm(p)
    ps = PatchSet(LEVER)
    rec = af3.install(rm, ps, heads="replicated", conf="sharded", b21=None, sites=sites, _test_single_device_mesh=(p == 1))
    try:
        def fwd(*a):
            def body(*al):
                with af3._sharded_body(LEVER):
                    return make_rows(rm.axis, *al)
            return rp_hk.region(body, rm, in_specs(rm.axis), out_spec(rm.axis))(*a)
        got = rp_hk.jit_apply(hk.transform(fwd).apply, rm, n_args=2 + len(args))(shard.put(params, rm), shard.put(key, rm), *[shard.put(a, rm) for a in args])
    finally:
        ps.restore()
    RESULTS["env"].setdefault("records", []).append({k: rec[k] for k in ("heads", "conf", "diffusion", "library") if k in rec})
    return stock, got


@needs_lib
def test_install_record_reports_effective_head_state():
    RESULTS["env"].update({"jax": jax.__version__, "platform": jax.devices()[0].platform, "library": af3.library_shas(L)})
    rm = _rm(2) if jax.device_count() >= 2 else None
    if rm is None:
        pytest.skip("needs >= 2 devices")
    ps = PatchSet(LEVER)
    rec = af3.install(rm, ps, heads="sharded", conf="sharded", b21=None)
    try:
        assert rec["heads"] == "replicated" and rec["diffusion"] == "absent" and rec["heads_requested"] == "sharded", rec   # effective, not requested
        assert rec["conf"] == "sharded" and len(rec["sites"]) >= 14
    finally:
        ps.restore()
    with pytest.raises(MemLeverRefused):
        af3.install(_rm(1), PatchSet("x"))                               # P=1 refused without the test hook


@needs_lib
@pytest.mark.parametrize("equation", ["ikc,jkc->ijc", "kjc,kic->ijc"])
def test_triangle_multiplication_rows_equal_stock(equation):
    M = L["modules"]; gc = _gc()
    cfg = _config(M.TriangleMultiplication, equation=equation, use_glu_kernel=False)
    n, c = 32, 16
    rs = np.random.RandomState(1)
    act = jnp.asarray(rs.standard_normal((n, n, c)).astype("float32") + OFFSET); mask = jnp.asarray((rs.rand(n, n) > 0.15).astype("float32"))
    stock_f = lambda a, m: M.TriangleMultiplication(cfg, gc, name="tm")(a, m)  # noqa: E731
    rows_f = lambda ax, a_l, m_full: M.TriangleMultiplication(cfg, gc, name="tm")(a_l, triatt.mask_rows(m_full, ax, int(a_l.shape[0])))  # noqa: E731
    for p in _ps():
        stock, got = _run(stock_f, rows_f, (act, mask), lambda ax: (shard.rows_spec(ax), shard.replicated_spec()), lambda ax: shard.rows_spec(ax), p)
        _hold("af3_trimul[%s]" % equation, p, stock, got, "complete_contraction")


@needs_lib
def test_outer_product_mean_and_msa_attention_rows_equal_stock():
    M = L["modules"]; gc = _gc()
    n, s_, cz, cm = 32, 6, 16, 12
    rs = np.random.RandomState(3)
    pair = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET)
    msa = jnp.asarray(rs.standard_normal((s_, n, cm)).astype("float32") + OFFSET); msa_mask = jnp.asarray((rs.rand(s_, n) > 0.1).astype("float32"))
    cfg_opm = _config(M.OuterProductMean, chunk_size=8, num_outer_channel=4)
    stock_f = lambda a, m: M.OuterProductMean(config=cfg_opm, global_config=gc, num_output_channel=cz, name="opm")(a, m)  # noqa: E731
    def rows_f(ax, a, m):
        n_loc = n // shard.axis_size(ax)
        return M.OuterProductMean(config=cfg_opm, global_config=gc, num_output_channel=cz, name="opm")(a, m, shard.axis_index(ax), n_loc)
    for p in _ps():
        stock, got = _run(stock_f, rows_f, (msa, msa_mask), lambda ax: (shard.replicated_spec(), shard.replicated_spec()), lambda ax: shard.rows_spec(ax), p)
        _hold("af3_outer_product_mean", p, stock, got, "row_local")
    cfg_msa = _config(M.MSAAttention, num_head=2)
    stock_g = lambda a, m, pr: M.MSAAttention(cfg_msa, gc, name="ma")(a, m, pair_act=pr)  # noqa: E731
    rows_g = lambda ax, a, m, pr_l: M.MSAAttention(cfg_msa, gc, name="ma")(a, m, pair_act=pr_l)  # noqa: E731
    for p in _ps():
        stock, got = _run(stock_g, rows_g, (msa, msa_mask, pair), lambda ax: (shard.replicated_spec(), shard.replicated_spec(), shard.rows_spec(ax)), lambda ax: shard.replicated_spec(), p)
        _hold("af3_msa_attention_pair_logits", p, stock, got, "row_local")


@needs_lib
def test_pairformer_iteration_region_equals_stock():
    """The PairFormerIteration body (trimul out/in, GridSelfAttention row + column, TransitionBlock) through the recipe's OWN region (the patched
    __call__ builds the shard_map) vs the stock block."""
    M = L["modules"]; gc = _gc()
    cfg = _config(M.PairFormerIteration, num_layer=1)
    n, c = 32, 16
    rs = np.random.RandomState(4)
    act = jnp.asarray(rs.standard_normal((n, n, c)).astype("float32") + OFFSET); mask = jnp.asarray((rs.rand(n, n) > 0.1).astype("float32"))
    mask = mask * mask.T                                                  # the tree's callers pass seq⊗seq masks (symmetric): the named hazard
    fwd = lambda a, m: M.PairFormerIteration(cfg, gc, name="pfi")(a, m)  # noqa: E731
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, act, mask)
    stock = jax.jit(hk.transform(fwd).apply)(params, key, act, mask)
    for p in _ps():
        rm = _rm(p); ps = PatchSet(LEVER)
        af3.install(rm, ps, sites=("trunk",), _test_single_device_mesh=(p == 1))
        try:
            got = rp_hk.jit_apply(hk.transform(fwd).apply, rm, n_args=4)(shard.put(params, rm), shard.put(key, rm), shard.put(act, rm), shard.put(mask, rm))
        finally:
            ps.restore()
        _hold("af3_pairformer_iteration[trimul×2,gsa row+col,transition]", p, stock, got, "row_local+complete_contraction")


@needs_lib
def test_distogram_head_rows_equal_stock():
    DG = L["distogram_head"]; gc = _gc()
    cfg = _config(DG.DistogramHead, first_break=2.3125, last_break=21.6875, num_bins=64) if getattr(DG.DistogramHead, "Config", None) else None
    n, c = 32, 16
    rs = np.random.RandomState(5)
    pair = jnp.asarray(rs.standard_normal((n, n, c)).astype("float32") + OFFSET)
    seq_mask = jnp.asarray((rs.rand(n) > 0.1).astype("float32"))
    def batch_of(m):
        return types.SimpleNamespace(token_features=types.SimpleNamespace(mask=m))
    fwd = lambda pr, m: DG.DistogramHead(cfg, gc)(batch_of(m), {"pair": pr}, return_distogram=True)  # noqa: E731
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, pair, seq_mask)
    stock = jax.jit(hk.transform(fwd).apply)(params, key, pair, seq_mask)
    for p in _ps():
        rm = _rm(p); ps = PatchSet(LEVER)
        af3.install(rm, ps, heads="replicated", conf="sharded", sites=("heads",), _test_single_device_mesh=(p == 1))
        try:
            def fwd_rows(pr, m):
                def body(pr_l, m_full):
                    with af3._manual_body(None, LEVER):
                        af3._S["heads_rows_conf"] = True
                        try:
                            out = DG.DistogramHead(cfg, gc)(batch_of(m_full), {"pair": pr_l}, return_distogram=True)
                        finally:
                            af3._S["heads_rows_conf"] = False
                        return out["contact_probs"], out["distogram"]
                return shard.shard_map(body, rm, (shard.rows_spec(rm.axis), shard.replicated_spec()), (shard.replicated_spec(), shard.replicated_spec()))(shard.constrain(pr, rm), m)
            cp, dg = rp_hk.jit_apply(hk.transform(fwd_rows).apply, rm, n_args=4)(shard.put(params, rm), shard.put(key, rm), shard.put(pair, rm), shard.put(seq_mask, rm))
        finally:
            ps.restore()
        _hold("af3_distogram[contact_probs]", p, stock["contact_probs"], cp, "row_local")
        _hold("af3_distogram[distogram]", p, stock["distogram"], dg, "row_local")


@needs_lib
def test_evoformer_prologue_pair_producers_at_p2():
    """The Evoformer prologue's pair producers (the embedding group) run at P=2 through the installed constraints: `_seq_pair_embedding` returns
    (pair_activations, pair_mask) in the pinned tree — the transcription table's pair index is checked against the method's source at install and the
    wrapped call equals the stock (a constraint moves no values)."""
    Ev = L["evoformer"]
    assert dict(af3.PROLOGUE_PAIR_OUTPUT)["_seq_pair_embedding"] == 0
    for meth, oi in af3.PROLOGUE_PAIR_OUTPUT:                                # the table matches the pinned source (what install enforces)
        assert (oi is None) == (af3._return_arity(getattr(Ev.Evoformer, meth)) == 1), (meth, oi)
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices")
    n, c_tf, c_z = 24, 10, 8
    rs = np.random.RandomState(8)
    target_feat = jnp.asarray(rs.standard_normal((n, c_tf)).astype("float32"))
    mask = jnp.asarray((rs.rand(n) > 0.1).astype("float32"))
    fake = types.SimpleNamespace(config=types.SimpleNamespace(pair_channel=c_z), global_config=_gc())
    tf = types.SimpleNamespace(mask=mask, residue_index=jnp.arange(n), asym_id=jnp.zeros(n, jnp.int32), entity_id=jnp.zeros(n, jnp.int32),
                               sym_id=jnp.zeros(n, jnp.int32), token_index=jnp.arange(n))
    stock_method = Ev.Evoformer._seq_pair_embedding
    fwd = lambda t: stock_method(fake, tf, t)  # noqa: E731
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, target_feat)
    stock_pair, stock_mask = jax.jit(hk.transform(fwd).apply)(params, key, target_feat)
    rm = _rm(2); ps = PatchSet(LEVER)
    af3.install(rm, ps, sites=("trunk",))
    try:
        assert Ev.Evoformer._seq_pair_embedding is not stock_method
        fwd2 = lambda t: Ev.Evoformer._seq_pair_embedding(fake, tf, t)  # noqa: E731
        got_pair, got_mask = rp_hk.jit_apply(hk.transform(fwd2).apply, rm, n_args=3)(shard.put(params, rm), shard.put(key, rm), shard.put(target_feat, rm))
    finally:
        ps.restore()
    assert np.array_equal(np.asarray(stock_mask), np.asarray(got_mask))            # the mask is data movement only
    # the pair: the constraint lets GSPMD partition the stock's projections over the mesh (GEMM shape per device differs) → the precision class, not bit-exact
    _hold("af3_evoformer_prologue[_seq_pair_embedding]", 2, np.asarray(stock_pair), np.asarray(got_pair), "row_local")
