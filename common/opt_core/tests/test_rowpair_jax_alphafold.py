"""opt_core.mem.rowpair_jax.alphafold — the AF2 recipe installs on the pinned library, refuses by name otherwise, and its sub-layer bodies on
row blocks equal the stock sub-layers (TriangleMultiplication fused+unfused × out/in, TriangleAttention per_row/per_column, OuterProductMean,
MSARowAttentionWithPairBias) at small N for every P the box offers. Needs jax + dm-haiku + ``alphafold.model`` (skips BY NAME without them);
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` on a CPU box. Measured max-abs-diffs go to ``$ROWPAIR_RESULTS``."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
if os.environ.get("ROWCHUNK_AF_PATH"):
    sys.path.insert(0, os.environ["ROWCHUNK_AF_PATH"])
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
from opt_core.mem.rowpair_jax import alphafold as rp_af2
from opt_core.mem.rowpair_jax import evidence, mesh, shard, triatt  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    import ml_collections  # noqa: E402
    from alphafold.model import modules as M  # noqa: E402
    try:
        from alphafold.model import modules_multimer as MM  # noqa: E402
    except Exception:  # noqa: BLE001
        MM = None
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model: %s" % WHY)
RESULTS = {"cases": [], "env": {}}
TOL_F32 = float(os.environ.get("ROWPAIR_TOL_F32", "1e-5"))                       # f32 at exact f32 matmul precision (cpu; gpu at HIGHEST)
TOL_F32_DEFAULT = float(os.environ.get("ROWPAIR_TOL_F32_DEFAULT", str(2.0 ** -10)))  # f32 operands at the GPU backend's DEFAULT matmul precision (tf32-eligible): the
                                                                                     # library's sub-layers run at their own (default) precision, as the stock does


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path and RESULTS["cases"]:
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True)


def _rm(p):
    if p > jax.device_count():                                          # a P-device mesh needs P devices on THIS backend (1-GPU boxes run P=1 only)
        pytest.skip("needs %d jax devices on the default backend (%s offers %d): the CPU box with XLA_FLAGS=--xla_force_host_platform_device_count=8, "
                    "or a %d-GPU box" % (p, jax.devices()[0].platform, jax.device_count(), p))
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps():
    """P=1 (a 1-device mesh through the test-only hook: the plan must equal the stock BIT-EXACT — the fidelity class that catches a mis-transcribed body)
    then every P>1 the box offers (tolerance class)."""
    return [1] + [p for p in (2, 4) if p <= jax.device_count()]


OFFSET = float(os.environ.get("ROWPAIR_AF2_OFFSET", "40.0"))          # non-zero-mean activations: a one-pass vs two-pass LayerNorm shows at rel ~1e-3 here


def _gc():
    return ml_collections.ConfigDict({"zero_init": False, "subbatch_size": 4, "deterministic": True, "use_remat": False, "eval_dropout": False,
                                       "use_flash_attention": False})   # a kit tree whose Attention reads this flag runs its stock ops; other trees ignore the key


def _hold(name, p, ref, got, cls):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    mad, scale = float(np.max(np.abs(ref - got))), float(np.max(np.abs(ref)))
    bitwise = bool(np.array_equal(ref, got))
    tol_class = "bitwise(1-device mesh)" if p == 1 else ("float32:default" if jax.devices()[0].platform == "gpu" else "float32")
    tol = 0.0 if p == 1 else (TOL_F32_DEFAULT if tol_class == "float32:default" else TOL_F32)
    RESULTS["cases"].append({"case": name, "P": p, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, "class": cls,
                             "dtype": "float32", "precision": "default", "tol_class": tol_class, "offset": OFFSET})
    if p == 1:
        assert bitwise, "%s on a 1-device mesh is not bitwise vs the stock body (max|d|=%g): a transcription differs from the pinned tree" % (name, mad)
    else:
        assert mad <= tol * scale, "%s P=%d tol_class=%s: max|d|=%g > %g*%g" % (name, p, tol_class, mad, tol, scale)


@needs_lib
def test_install_refusals_and_restore():
    RESULTS["env"].update({"jax": jax.__version__, "platform": jax.devices()[0].platform, "modules_file": M.__file__, "modules_sha": rp_af2.file_sha(M),
                           "multimer": bool(MM), "fused_tree": hasattr(M.TriangleMultiplication, "_fused_triangle_multiplication")})
    rm = _rm(2)
    with pytest.raises(MemLeverRefused) as ei:
        rp_af2.install(rm, M, MM, pin={"0" * 64: "nothing"})
    assert "is not a transcribed tree" in ei.value.reason
    stock_call = M.TriangleMultiplication.__call__
    patches = rp_af2.install(rm, M, MM)
    try:
        assert rp_af2.installed() and rp_af2.pad_multiple() == 2
        d = rp_af2.describe()
        assert "TriangleMultiplication.__call__" in d["sites"] and "EvoformerIteration.__call__" in d["sites"] and d["heads"] in rp_af2.HEAD_WORDS and d["msa"] == "replicated"
        assert d["library"] != "none"
        RESULTS["env"]["describe"] = d
        d = rp_af2.describe()                             # the ON vocabulary: evidence.line renders it as is (schedule / kernel / kernel_reason / sites present)
        for k in evidence.ON_REQUIRED:
            assert k in d, (k, sorted(d))
        ln = evidence.line("t", "on", int(rm.n_gpu), rmesh=rm, peaks=evidence.device_peaks(rm), **d)
        assert "schedule=%s" % rp_af2.TRIMUL_DEFAULT in ln and "kernel=jnp" in ln and "EvoformerIteration.__call__" in ln and "n_gpu=%d sharding=rowpair" % int(rm.n_gpu) in ln, ln
        RESULTS["env"]["line"] = ln
    finally:
        names = rp_af2.uninstall()
    assert not rp_af2.installed() and M.TriangleMultiplication.__call__ is stock_call and names


def _apply_pair_module(make, p, act, mask, region_mask="rows"):
    """params from a plain init; stock apply; then the installed plan: the module inside a region on row blocks (mask rows local)."""
    fwd = lambda a, m: make()(a, m, False) if region_mask != "opm" else make()(a, m, False)  # noqa: E731
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, act, mask)
    stock = jax.jit(hk.transform(fwd).apply)(params, key, act, mask)
    rm = _rm(p)
    ax = rm.axis
    rows, rep = shard.rows_spec(ax), shard.replicated_spec()
    patches = rp_af2.install(rm, M, MM, _test_single_device_mesh=(p == 1))
    try:
        def fwd_rows(a, m):
            mod = make()
            def body(a_l, m_full):
                return mod(a_l, triatt.mask_rows(m_full, ax, int(a_l.shape[0])), False)
            return rp_hk.region(body, rm, (rows, rep), rows)(shard.constrain(a, rm), m)
        got = rp_hk.jit_apply(hk.transform(fwd_rows).apply, rm, n_args=4)(shard.put(params, rm), shard.put(key, rm), shard.put(act, rm), shard.put(mask, rm))
    finally:
        rp_af2.uninstall()
    return stock, got


@needs_lib
@pytest.mark.parametrize("equation", ["ikc,jkc->ijc", "kjc,kic->ijc"])
@pytest.mark.parametrize("fused", [True, False])
def test_triangle_multiplication_rows_equal_stock(equation, fused):
    if fused and not hasattr(M.TriangleMultiplication, "_fused_triangle_multiplication"):
        pytest.skip("this library has no fused body")
    n, cz = 32, 16
    rs = np.random.RandomState(1)
    act = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET); mask = jnp.asarray((rs.rand(n, n) > 0.15).astype("float32"))
    cfg = ml_collections.ConfigDict({"equation": equation, "num_intermediate_channel": 8, "fuse_projection_weights": bool(fused), "dropout_rate": 0.0,
                                      "orientation": "per_row", "shared_dropout": True})
    for p in _ps():
        stock, got = _apply_pair_module(lambda: M.TriangleMultiplication(cfg, _gc()), p, act, mask)
        _hold("af2_trimul[%s,%s]" % (equation, "fused" if fused else "unfused"), p, stock, got, "complete_contraction")


@needs_lib
@pytest.mark.parametrize("orientation", ["per_row", "per_column"])
def test_triangle_attention_rows_equal_stock(orientation):
    n, cz = 32, 16
    rs = np.random.RandomState(2)
    act = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET); mask = jnp.asarray((rs.rand(n, n) > 0.15).astype("float32"))
    cfg = ml_collections.ConfigDict({"num_head": 2, "key_dim": 8, "value_dim": 8, "gating": True, "orientation": orientation, "dropout_rate": 0.0, "shared_dropout": True})
    def make():
        mod = M.TriangleAttention(cfg, _gc())
        if orientation == "per_column":                   # the stock swaps to per_row internally on the transposed tensor; the plan's body handles per_column itself
            return lambda a, m, t: mod(a, m, t)
        return mod
    for p in _ps():
        stock, got = _apply_pair_module(make, p, act, mask)
        _hold("af2_triatt[%s]" % orientation, p, stock, got, "row_local")


@needs_lib
def test_msa_row_attention_and_opm_rows_equal_stock():
    n, s_, cz, cm = 32, 6, 16, 12
    rs = np.random.RandomState(3)
    pair = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET)
    msa = jnp.asarray(rs.standard_normal((s_, n, cm)).astype("float32") + OFFSET); msa_mask = jnp.asarray((rs.rand(s_, n) > 0.1).astype("float32"))
    cfg_att = ml_collections.ConfigDict({"num_head": 2, "key_dim": 8, "value_dim": 8, "gating": True, "orientation": "per_row", "dropout_rate": 0.0, "shared_dropout": True})
    cfg_opm = ml_collections.ConfigDict({"chunk_size": 8, "num_outer_channel": 4, "dropout_rate": 0.0, "orientation": "per_row", "shared_dropout": True, "first": False})
    key = jax.random.PRNGKey(0)
    for p in _ps():
        rm = _rm(p); ax = rm.axis
        rows, rep = shard.rows_spec(ax), shard.replicated_spec()
        # MSARowAttentionWithPairBias(msa, msa_mask, pair rows) -> msa (replicated out)
        f = lambda ma, mm, pa: M.MSARowAttentionWithPairBias(cfg_att, _gc())(ma, mm, pa, False)  # noqa: E731
        params = hk.transform(f).init(key, msa, msa_mask, pair)
        stock = jax.jit(hk.transform(f).apply)(params, key, msa, msa_mask, pair)
        rp_af2.install(rm, M, MM, _test_single_device_mesh=(p == 1))
        try:
            def f_rows(ma, mm, pa):
                mod = M.MSARowAttentionWithPairBias(cfg_att, _gc())
                return rp_hk.region(lambda ma_, mm_, pa_l: mod(ma_, mm_, pa_l, False), rm, (rep, rep, rows), rep)(ma, mm, shard.constrain(pa, rm))
            got = rp_hk.jit_apply(hk.transform(f_rows).apply, rm, n_args=5)(shard.put(params, rm), shard.put(key, rm), shard.put(msa, rm), shard.put(msa_mask, rm), shard.put(pair, rm))
        finally:
            rp_af2.uninstall()
        _hold("af2_msa_row_att_pair_bias", p, stock, got, "row_local")
        # OuterProductMean(msa, msa_mask) -> pair rows
        g = lambda ma, mm: M.OuterProductMean(cfg_opm, _gc(), cz)(ma, mm, False)  # noqa: E731
        params = hk.transform(g).init(key, msa, msa_mask)
        stock = jax.jit(hk.transform(g).apply)(params, key, msa, msa_mask)
        rp_af2.install(rm, M, MM, _test_single_device_mesh=(p == 1))
        try:
            def g_rows(ma, mm):
                mod = M.OuterProductMean(cfg_opm, _gc(), cz)
                return rp_hk.region(lambda ma_, mm_: mod(ma_, mm_, False), rm, (rep, rep), rows)(ma, mm)
            got = rp_hk.jit_apply(hk.transform(g_rows).apply, rm, n_args=4)(shard.put(params, rm), shard.put(key, rm), shard.put(msa, rm), shard.put(msa_mask, rm))
        finally:
            rp_af2.uninstall()
        _hold("af2_outer_product_mean", p, stock, got, "row_local")


@needs_lib
@pytest.mark.parametrize("tree", ["monomer", "multimer"])
def test_evoformer_iteration_region_pads_odd_n(tree):
    """N % P != 0 (an unpadded multimer complex): the EvoformerIteration region pads the residue axes with masked zeros to a multiple of P and slices
    back — the real positions equal the STOCK block at N within the tolerance class (P=2, 4); the plan records the padded calls."""
    from alphafold.model import config as af_config, prng
    name = "model_1_multimer_v3" if tree == "multimer" else "model_1_ptm"
    try:
        mc = af_config.model_config(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip("alphafold.model.config.model_config(%r): %r" % (name, e))
    c, gc = mc.model.embeddings_and_evoformer.evoformer, mc.model.global_config
    fused = bool(getattr(c.triangle_multiplication_outgoing, "fuse_projection_weights", False))
    if fused and not hasattr(M.TriangleMultiplication, "_fused_triangle_multiplication"):
        pytest.skip("this tree has no fused TriangleMultiplication body")
    n, s_, cm, cz = 19, 6, 32, 16
    rs = np.random.RandomState(7)
    msa = jnp.asarray(rs.standard_normal((s_, n, cm)).astype("float32") + OFFSET); pair = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET)
    seq_mask = (rs.rand(n) > 0.1).astype("float32")
    msa_mask = jnp.asarray(np.tile(seq_mask[None], (s_, 1)) * (rs.rand(s_, n) > 0.1)); pair_mask = jnp.asarray(seq_mask[:, None] * seq_mask[None, :])
    def fwd(m, z, mm, zm):
        out = M.EvoformerIteration(c, gc, is_extra_msa=False, name="ei")({"msa": m, "pair": z}, {"msa": mm, "pair": zm}, is_training=False,
                                                                          safe_key=prng.SafeKey(jax.random.PRNGKey(1)))
        return out["msa"], out["pair"]
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, msa, pair, msa_mask, pair_mask)
    stock_msa, stock_pair = jax.jit(hk.transform(fwd).apply)(params, key, msa, pair, msa_mask, pair_mask)
    for p in [q for q in (2, 4) if q <= jax.device_count()]:
        assert n % p != 0
        rm = _rm(p)
        rp_af2.install(rm, M, MM)
        try:
            got_msa, got_pair = rp_hk.jit_apply(hk.transform(fwd).apply, rm, n_args=6)(shard.put(params, rm), shard.put(key, rm), *[shard.put(a, rm) for a in (msa, pair, msa_mask, pair_mask)])
            assert rp_af2.describe()["padded_calls"] >= 1, rp_af2.describe()
        finally:
            rp_af2.uninstall()
        assert got_pair.shape == stock_pair.shape and got_msa.shape == stock_msa.shape
        keep = np.asarray(seq_mask) > 0                                     # compare real positions (masked rows are don't-care in the stock too)
        _hold("af2_evoformer_iteration_odd_n[%s,pair]" % tree, p, np.asarray(stock_pair)[keep][:, keep], np.asarray(got_pair)[keep][:, keep], "padded_region")
        _hold("af2_evoformer_iteration_odd_n[%s,msa]" % tree, p, np.asarray(stock_msa)[:, keep], np.asarray(got_msa)[:, keep], "padded_region")


def _while_body_collectives(hlo_text):
    """{while-body computation name: [collective ops]} for every while loop in a compiled HLO module whose body (or anything it calls) holds a collective —
    a per-iteration collective. Bodies are followed through body=/condition=/calls=/to_apply=/branch_computations= references."""
    import re
    comps, name, buf = {}, None, []
    for line in hlo_text.splitlines():
        m = re.match(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s+->\s+.*\{\s*$", line)
        if m:
            name, buf = m.group(1), []
        elif line.strip() == "}" and name is not None:
            comps[name] = "\n".join(buf); name = None
        elif name is not None:
            buf.append(line)
    ref = re.compile(r"(?:body|condition|calls|to_apply|branch_computations)=\{?%?([\w.\-]+(?:,\s*%?[\w.\-]+)*)\}?")
    coll = re.compile(r"\b(all-gather|all-to-all|all-reduce|reduce-scatter|collective-permute)(?:-start)?\b")
    def closure(n, seen):
        if n in seen or n not in comps:
            return seen
        seen.add(n)
        for m in ref.finditer(comps[n]):
            for x in re.split(r",\s*%?", m.group(1)):
                closure(x.lstrip("%"), seen)
        return seen
    out = {}
    for n, txt in comps.items():
        for m in re.finditer(r"while\([^)]*\),\s*condition=%?([\w.\-]+),\s*body=%?([\w.\-]+)", txt):
            body = m.group(2)
            ops = sorted({c.group(1) for k in closure(body, set()) for c in coll.finditer(comps.get(k, ""))})
            if ops:
                out[body] = ops
    return out


def _while_bodies(hlo_text):
    """{while-body computation name: the text of the body and everything it calls} for every while loop in a compiled HLO module."""
    import re
    comps, name, buf = {}, None, []
    for line in hlo_text.splitlines():
        m = re.match(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s+->\s+.*\{\s*$", line)
        if m:
            name, buf = m.group(1), []
        elif line.strip() == "}" and name is not None:
            comps[name] = "\n".join(buf); name = None
        elif name is not None:
            buf.append(line)
    ref = re.compile(r"(?:body|condition|calls|to_apply|branch_computations)=\{?%?([\w.\-]+(?:,\s*%?[\w.\-]+)*)\}?")
    def closure(n, seen):
        if n in seen or n not in comps:
            return seen
        seen.add(n)
        for m in ref.finditer(comps[n]):
            for x in re.split(r",\s*%?", m.group(1)):
                closure(x.lstrip("%"), seen)
        return seen
    out = {}
    for n, txt in comps.items():
        for m in re.finditer(r"while\([^)]*\),\s*condition=%?([\w.\-]+),\s*body=%?([\w.\-]+)", txt):
            body = m.group(2)
            out[body] = "\n".join(comps.get(k, "") for k in sorted(closure(body, set())))
    return out


@needs_lib
def test_template_pointwise_scan_has_no_per_iteration_collective_at_p2():
    """E2D-F7: the monomer TemplateEmbedding tail (modules.py: flatten the pair to [N*N,1,c] and run the template-pointwise Attention through
    mapping.inference_subbatch — an hk.scan with dynamic_slice over N*N) fed by the recipe's TemplatePairStack region at P=2 compiles to while loops with
    NO collective inside any loop body (the region hands the template pair back replicated), and equals the stock within the class."""
    if MM is not None and not hasattr(M, "TemplatePairStack"):
        pytest.skip("multimer-only tree: no monomer TemplatePairStack / pointwise template attention")
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices")
    from alphafold.model import config as af_config, prng
    try:
        mc = af_config.model_config("model_1_ptm")
    except Exception as e:  # noqa: BLE001
        pytest.skip("alphafold.model.config.model_config('model_1_ptm'): %r" % (e,))
    tc, gc = mc.model.embeddings_and_evoformer.template, mc.model.global_config
    try:
        gc.use_flash_attention = False
    except Exception:  # noqa: BLE001
        pass
    n, ct, cq = 32, int(tc.template_pair_stack.triangle_attention_starting_node.value_dim), 128
    rs = np.random.RandomState(11)
    pair = jnp.asarray(rs.standard_normal((n, n, ct)).astype("float32") + OFFSET); query = jnp.asarray(rs.standard_normal((n, n, cq)).astype("float32"))
    mask = jnp.asarray((rs.rand(n, n) > 0.05).astype("float32"))
    def fwd(p_, m_, q_):
        act = M.TemplatePairStack(tc.template_pair_stack, gc)(p_, m_, is_training=False, safe_key=prng.SafeKey(jax.random.PRNGKey(3)))
        flat_query = jnp.reshape(q_, [n * n, 1, cq])                                      # the stock tail, monomer modules.py TemplateEmbedding.__call__
        flat_templates = jnp.reshape(jnp.transpose(act[None], [1, 2, 0, 3]), [n * n, 1, ct])
        bias = jnp.zeros([1, 1, 1, 1], flat_query.dtype)
        attn = M.Attention(tc.attention, gc, cq)
        emb = M.mapping.inference_subbatch(attn, int(tc.subbatch_size), batched_args=[flat_query, flat_templates], nonbatched_args=[bias], low_memory=True)
        return jnp.reshape(emb, [n, n, cq])
    key = jax.random.PRNGKey(0)
    f = hk.transform(fwd)
    params = f.init(key, pair, mask, query)
    stock_out = jax.jit(f.apply)(params, key, pair, mask, query)
    rm = _rm(2)
    rp_af2.install(rm, M, MM)
    try:
        jitted = rp_hk.jit_apply(hk.transform(fwd).apply, rm, n_args=5)
        args = (shard.put(params, rm), shard.put(key, rm), shard.put(pair, rm), shard.put(mask, rm), shard.put(query, rm))
        got = jitted(*args)
        hlo = jitted.lower(*args).compile().as_text()
    finally:
        rp_af2.uninstall()
    loops = _while_body_collectives(hlo)                       # {body: [collectives]} — the template pair stack's layer_stack body legitimately holds the region's
    bodies = _while_bodies(hlo)                                 # collectives (once per block); the POINTWISE scan bodies (chunks of [subbatch, 1, c]) must hold none
    marker = "[%d,1," % int(tc.subbatch_size)
    pointwise = sorted(b for b, txt in bodies.items() if marker in txt)
    RESULTS["env"]["template_pointwise_scan"] = {"bodies_with_collectives": loops, "pointwise_bodies": pointwise}
    assert pointwise, "the template-pointwise scan loop was not found in the compiled HLO (marker %s); bodies=%s" % (marker, sorted(bodies)[:12])
    bad = {b: loops[b] for b in pointwise if b in loops}
    assert not bad, "per-iteration collectives inside the template-pointwise scan at P=2: %s" % (bad,)
    _hold("af2_template_stack+pointwise[monomer]", 2, np.asarray(stock_out), np.asarray(got), "row_local")
