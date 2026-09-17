"""opt_core.mem.rowpair_jax.alphafold_template — the AF2 template embedding as ONE rows region equals the stock embedding on the full pair for
both pinned trees (monomer ``TemplateEmbedding`` on the tree in this interpreter; ``modules_multimer.TemplateEmbedding`` where the tree has it),
``dgram_rows`` is the stock ``dgram_from_positions`` row block bit-exact, the dispatch-off program is today's program bit-exact, the region compiles to NO
collective when the pair stack has zero blocks (rows in, rows out — the zero-block guard) and to no collective inside the template-pointwise scan when it
has blocks, and the refusals are by name. Needs jax + dm-haiku + ``alphafold.model`` (skips BY NAME without them); ``ROWCHUNK_AF_PATH=<dir with
alphafold/>`` selects another tree; ``XLA_FLAGS=--xla_force_host_platform_device_count=8`` on a CPU box. Measured values go to ``$ROWPAIR_RESULTS``."""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
if os.environ.get("ROWCHUNK_AF_PATH"):
    sys.path.insert(0, os.environ["ROWCHUNK_AF_PATH"])
_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")
if "xla_force_host_platform_device_count" not in (_XLA_FLAGS_BEFORE or ""):
    os.environ["XLA_FLAGS"] = ((_XLA_FLAGS_BEFORE or "") + " --xla_force_host_platform_device_count=8").strip()
try:
    import jax as _jax_backend_init  # noqa: E402
    _jax_backend_init.devices()
except Exception:  # noqa: BLE001
    pass
finally:
    if _XLA_FLAGS_BEFORE is None:
        os.environ.pop("XLA_FLAGS", None)
    else:
        os.environ["XLA_FLAGS"] = _XLA_FLAGS_BEFORE

from opt_core.mem import MemLeverRefused  # noqa: E402
from opt_core.mem.rowpair_jax import alphafold as rp_af2  # noqa: E402
from opt_core.mem.rowpair_jax import alphafold_template as rp_t  # noqa: E402
from opt_core.mem.rowpair_jax import mesh, shard  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    from alphafold.model import modules as M  # noqa: E402
    from alphafold.model import config as af_config, prng  # noqa: E402
    try:
        from alphafold.model import modules_multimer as MM  # noqa: E402
    except Exception:  # noqa: BLE001
        MM = None
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model: %s" % WHY)
needs_monomer = pytest.mark.skipif(not HAVE or not hasattr(M, "TemplateEmbedding") or not hasattr(M, "TemplatePairStack"),
                                   reason="this tree has no monomer TemplateEmbedding / TemplatePairStack")
needs_multimer = pytest.mark.skipif(not HAVE or MM is None, reason="this tree has no modules_multimer")
RESULTS = {"cases": [], "env": {}, "guards": {}, "refusals": {}}
TOL_F32 = float(os.environ.get("ROWPAIR_TOL_F32", "1e-5"))
OFFSET = float(os.environ.get("ROWPAIR_AF2_OFFSET", "40.0"))
COLL = re.compile(r"\b(all-gather|all-to-all|all-reduce|reduce-scatter|collective-permute)(?:-start)?\b")


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path:
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True, default=str)


def _rm(p):
    if p > jax.device_count():                                          # a P-device mesh needs P devices on THIS backend (1-GPU boxes run P=1 only)
        pytest.skip("needs %d jax devices on the default backend (%s offers %d): the CPU box with XLA_FLAGS=--xla_force_host_platform_device_count=8, "
                    "or a %d-GPU box" % (p, jax.devices()[0].platform, jax.device_count(), p))
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps(n):
    return [1] + [p for p in (2, 4, 8) if p <= jax.device_count() and n % p == 0]


def _record(name, p, ref, got, cls, tol=TOL_F32, extra=None):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    assert ref.shape == got.shape, (name, ref.shape, got.shape)
    mad, scale = float(np.max(np.abs(ref - got))), float(np.max(np.abs(ref)))
    bitwise = bool(np.array_equal(ref, got))
    row = {"case": name, "P": p, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, "class": cls, "dtype": "float32",
           "tol": 0.0 if p == 1 else tol, "platform": jax.devices()[0].platform}
    row.update(extra or {})
    RESULTS["cases"].append(row)
    if p == 1:
        assert bitwise, "%s on a 1-device mesh is not bitwise vs the stock body (max|d|=%g): a transcription differs from the pinned tree" % (name, mad)
    else:
        assert mad <= tol * max(scale, 1.0), "%s P=%d: max|d|=%g > %g*%g" % (name, p, mad, tol, scale)
    return row


def _monomer_cfg(num_block=1):
    mc = af_config.model_config("model_1_ptm")
    c, gc = mc.model.embeddings_and_evoformer.template, mc.model.global_config
    c.template_pair_stack.num_block = int(num_block)
    gc.use_remat = False
    gc.deterministic = True
    gc.subbatch_size = 4
    for k, v in (("use_flash_attention", False), ("eval_dropout", False), ("bfloat16", False)):
        try:
            setattr(gc, k, v)
        except Exception:  # noqa: BLE001 — the tree's global_config does not carry the key
            pass
    return c, gc


def _monomer_batch(n, t, seed):
    """T template slots of an N-residue target: real-looking random atoms; slot 1 (when T > 1) is an all-dummy slot (mask 0, zeros)."""
    rs = np.random.RandomState(seed)
    aatype = rs.randint(0, 21, size=(t, n)).astype("int32")
    pos = (rs.standard_normal((t, n, 37, 3)) * 3.0 + np.arange(n)[None, :, None, None] * np.array([3.8, 0., 0.])[None, None, None, :]).astype("float32")
    amask = (rs.rand(t, n, 37) > 0.1).astype("float32")
    pb_mask = (rs.rand(t, n) > 0.1).astype("float32")
    pb = pos[:, :, 3, :] + rs.standard_normal((t, n, 3)).astype("float32") * 0.1
    tmask = np.ones((t,), "float32")
    if t > 1:
        tmask[1] = 0.
        pos[1], amask[1], pb_mask[1], pb[1], aatype[1] = 0., 0., 0., 0., 0
    return {"template_mask": jnp.asarray(tmask), "template_aatype": jnp.asarray(aatype), "template_pseudo_beta": jnp.asarray(pb),
            "template_pseudo_beta_mask": jnp.asarray(pb_mask), "template_all_atom_positions": jnp.asarray(pos), "template_all_atom_masks": jnp.asarray(amask)}


def _install_both(rm, p, active=None):
    patches = rp_af2.install(rm, M, MM, sites=("trunk",), _test_single_device_mesh=(p == 1))
    act = active if active is not None else (lambda: rp_af2.installed() and not rp_hk.in_region())
    sites = rp_t.install_template(patches, M, MM, rmesh=rm, axis=rm.axis, n_gpu=p, active=act, in_region=rp_hk.in_region, _test_single_device_mesh=(p == 1))
    return patches, sites


def _uninstall():
    rp_af2.uninstall()
    rp_t.forget()


def _jit_rows(apply_fn, rm, n_args, row_args):
    """jax.jit with the pair-shaped positional args ROW-sharded in, everything else replicated; the output sharding left to propagation (a fact under test)."""
    rep, rows = shard.named(rm, shard.replicated_spec()), shard.named(rm, shard.rows_spec(rm.axis, 3, 0))
    ins = tuple(rows if i in row_args else rep for i in range(n_args))
    return jax.jit(apply_fn, in_shardings=ins), ins


def _hlo_census(text):
    ops = {}
    for m in re.finditer(r"^\s*%?[\w.\-]+ = (\w+\[[^\]]*\](?:\{[^}]*\})?) (all-gather|all-to-all|all-reduce|reduce-scatter|collective-permute)(?:-start)?\(", text, re.M):
        ops.setdefault(m.group(2), []).append(m.group(1))
    return {k: {"count": len(v), "shapes": sorted(set(v))} for k, v in ops.items()}, len(COLL.findall(text))


def _while_bodies(hlo_text):
    """{while-body computation: the text of the body and everything it calls} for every while loop of a compiled HLO module."""
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


# ================================================================================================ (3) dgram_rows == stock rows, bit-exact
@needs_lib
def test_dgram_rows_equals_stock_rows_bitwise():
    RESULTS["env"].update({"jax": jax.__version__, "platform": jax.devices()[0].platform, "device_count": jax.device_count(), "modules_file": M.__file__,
                           "modules_sha": rp_t.file_sha(M), "multimer_sha": rp_t.file_sha(MM) if MM is not None else None, "pin_label": rp_t.TEMPL_PIN.get(rp_t.file_sha(M)),
                           "tree_facts": rp_t.tree_facts(M) if hasattr(M, "SingleTemplateEmbedding") else None})
    assert rp_t.file_sha(M) in rp_t.TEMPL_PIN, ("this interpreter's alphafold.model.modules is not a pinned tree", M.__file__, rp_t.file_sha(M))
    rs = np.random.RandomState(3)
    pos = jnp.asarray((rs.standard_normal((40, 3)) * 8.0).astype("float32"))
    cfgs = [dict(num_bins=39, min_bin=3.25, max_bin=50.75), dict(num_bins=15, min_bin=3.25, max_bin=20.75)]   # template dgram_features; recycle prev_pos
    worst = 0.0
    for cfg in cfgs:
        full = np.asarray(M.dgram_from_positions(pos, **cfg))
        for a, b in ((0, 40), (0, 10), (10, 20), (35, 40), (7, 8)):
            got = np.asarray(rp_t.dgram_rows(pos[a:b], pos, **cfg))
            assert got.shape == (b - a, 40, cfg["num_bins"])
            assert np.array_equal(got, full[a:b]), (cfg, a, b, float(np.max(np.abs(got - full[a:b]))))
            worst = max(worst, float(np.max(np.abs(got - full[a:b]))))
    RESULTS["cases"].append({"case": "dgram_rows_vs_dgram_from_positions_rows", "P": None, "max_abs_diff": worst, "bitwise": worst == 0.0, "class": "row_local"})


# ================================================================================================ (6) refusals by name
@needs_monomer
def test_install_template_refusals_by_name():
    rm = _rm(2)
    ref = {}
    patches = rp_af2.install(rm, M, MM, sites=("trunk",))                  # the trunk plan only: the template term is installed explicitly below
    try:
        with pytest.raises(MemLeverRefused) as ei:                     # unpinned tree
            rp_t.install_template(patches, M, MM, rmesh=rm, axis=rm.axis, n_gpu=2, active=lambda: True, in_region=rp_hk.in_region, pin={"0" * 64: "nothing"})
        assert "is not a transcribed tree" in ei.value.reason, ei.value.reason
        ref["unpinned"] = ei.value.reason
        assert not rp_t.installed()
        sites = rp_t.install_template(patches, M, MM, rmesh=rm, axis=rm.axis, n_gpu=2, active=lambda: rp_af2.installed() and not rp_hk.in_region(), in_region=rp_hk.in_region)
        assert rp_t.installed() and "TemplateEmbedding.__call__" in sites and "SingleTemplateEmbedding.__call__" in sites
        assert (MM is None) == ("multimer.TemplateEmbedding.__call__" not in sites)
        d = rp_t.describe()
        assert d["template"] == "rows" and d["template_masks"] == "replicated" and d["template_library"] != "none"
        ref["describe"] = d
        # N % P != 0 at the region entry
        c, gc = _monomer_cfg(1)
        n, t, cq = 18, 1, 32
        assert n % 4 != 0
        _uninstall()
        tb = _monomer_batch(n, t, 1)
        rs = np.random.RandomState(2)
        query = jnp.asarray(rs.standard_normal((n, n, cq)).astype("float32"))
        seq_mask = np.ones((n,), "float32"); mask_2d = jnp.asarray(seq_mask[:, None] * seq_mask[None, :])
        def fwd(q, m, b):
            return M.TemplateEmbedding(c, gc)(q, b, m, is_training=False)
        f = hk.transform(fwd)
        key = jax.random.PRNGKey(0)
        params = f.init(key, query, mask_2d, tb)                         # the stock init (nothing installed)
        rm4 = _rm(4)
        patches, _sites = _install_both(rm4, 4)
        with pytest.raises(MemLeverRefused) as ei:
            jax.jit(f.apply)(params, key, query, mask_2d, tb)
        assert "not divisible" in ei.value.reason, ei.value.reason
        ref["n_mod_p"] = ei.value.reason
    finally:
        _uninstall()
    assert not rp_t.installed() and not rp_af2.installed()
    stock_call = rp_hk.stock_body(M.TemplateEmbedding, "__call__")
    assert stock_call is not rp_t._template_embedding_call
    RESULTS["refusals"].update(ref)


# ================================================================================================ (1) (4) (5) monomer TemplateEmbedding
def _monomer_case(n, t, cq, num_block, seed):
    c, gc = _monomer_cfg(num_block)
    tb = _monomer_batch(n, t, seed)
    rs = np.random.RandomState(seed + 100)
    query = jnp.asarray(rs.standard_normal((n, n, cq)).astype("float32") + OFFSET)
    seq_mask = (rs.rand(n) > 0.1).astype("float32")
    mask_2d = jnp.asarray(seq_mask[:, None] * seq_mask[None, :])

    def fwd(q, m, b):
        return M.TemplateEmbedding(c, gc)(q, b, m, is_training=False)
    f = hk.transform(fwd)
    key = jax.random.PRNGKey(seed)
    params = f.init(key, query, mask_2d, tb)
    stock = jax.jit(f.apply)(params, key, query, mask_2d, tb)              # nothing installed: the pinned single-device program
    return c, gc, tb, query, mask_2d, f, key, params, stock


@needs_monomer
@pytest.mark.parametrize("n,t,cq", [(16, 1, 32), (24, 2, 32), (32, 4, 64)])
def test_monomer_template_embedding_rows_equal_stock(n, t, cq):
    """(1): stock TemplateEmbedding on the full pair vs the installed rows region (trunk plan + template plan) at P = 1 (bit-exact: fidelity) and every P | N;
    (5): the region's output leaves row-sharded (propagated output sharding = rows); the collective census of the P>1 program (the nested trunk
    stack's named sites — the zero-block case below proves this module adds none) is recorded per while-loop body. The template-pointwise scan runs
    INSIDE the manual region on local values, where no sharding propagation exists: a per-iteration reshard (the defect class of a scan over a
    row-sharded GLOBAL operand) cannot be expressed there."""
    c, gc, tb, query, mask_2d, f, key, params, stock = _monomer_case(n, t, cq, 1, seed=n + t)
    for p in _ps(n):
        rm = _rm(p)
        _install_both(rm, p)
        try:
            jitted, ins = _jit_rows(f.apply, rm, 5, row_args=(2,))
            args = (shard.put(params, rm), shard.put(key, rm), jax.device_put(query, ins[2]), shard.put(mask_2d, rm), shard.put(tb, rm))
            got = jitted(*args)
            spec = tuple(getattr(got.sharding, "spec", ())) if p > 1 else None
            hlo = jitted.lower(*args).compile().as_text() if p > 1 else ""
            regions = rp_t.describe()["template_regions_built"]
        finally:
            _uninstall()
        extra = {"N": n, "T": t, "c_z": cq, "num_block": 1, "out_spec": str(spec), "template_regions_built": regions}
        if p > 1:
            census, ncoll = _hlo_census(hlo)
            bodies = _while_bodies(hlo)                                  # {while body: closure text}: which loops hold the (nested trunk stack's) collectives — recorded
            per_body = {b: len(COLL.findall(txt)) for b, txt in bodies.items() if COLL.search(txt)}
            extra.update({"collectives": census, "n_collectives": ncoll, "while_bodies_with_collectives": per_body, "n_while_bodies": len(bodies)})
            RESULTS["guards"]["monomer_N%d_T%d_P%d" % (n, t, p)] = extra
            assert spec and spec[0] == rm.axis and all(s is None for s in spec[1:]), ("the template embedding must leave the program ROW-sharded", spec)
            assert regions >= 1, "the template region was not built (dispatch never reached the rows body)"
        _record("af2_template_embedding[monomer,N=%d,T=%d]" % (n, t), p, stock, got, "row_local+complete_contraction(nested pair stack)", extra=extra)


@needs_monomer
@pytest.mark.parametrize("n,t,cq", [(16, 2, 32), (32, 4, 64)])
def test_monomer_template_region_zero_blocks_has_no_collective(n, t, cq):
    """(5) THE ZERO-BLOCK GUARD: with ``num_block = 0`` (the stock stack returns its input) everything in the region is this module's code — the feature rows,
    ``embedding2d``, ``output_layer_norm``, the pointwise attention on local points, the mask product: the compiled P>1 program holds NO collective of any
    kind (rows in, rows out; nothing is gathered or replicated) and equals the stock within the row_local class (bit-exact recorded)."""
    c, gc, tb, query, mask_2d, f, key, params, stock = _monomer_case(n, t, cq, 0, seed=7 * n + t)
    for p in _ps(n):                                                        # P=1: this module's code alone vs the stock, BIT-EXACT (fidelity)
        rm = _rm(p)
        _install_both(rm, p)
        try:
            jitted, ins = _jit_rows(f.apply, rm, 5, row_args=(2,))
            args = (shard.put(params, rm), shard.put(key, rm), jax.device_put(query, ins[2]), shard.put(mask_2d, rm), shard.put(tb, rm))
            got = jitted(*args)
            hlo = jitted.lower(*args).compile().as_text() if p > 1 else ""
            spec = tuple(getattr(got.sharding, "spec", ()))
        finally:
            _uninstall()
        if p > 1:
            census, ncoll = _hlo_census(hlo)
            RESULTS["guards"]["monomer_zero_blocks_N%d_T%d_P%d" % (n, t, p)] = {"collectives": census, "n_collectives": ncoll, "out_spec": str(spec)}
            assert ncoll == 0, "the template region (no pair-stack blocks) at P=%d holds collectives: %s" % (p, census)
            assert spec and spec[0] == rm.axis and all(s is None for s in spec[1:]), spec
        _record("af2_template_embedding_zero_blocks[monomer,N=%d,T=%d]" % (n, t), p, stock, got, "row_local", extra={"N": n, "T": t, "num_block": 0})


@needs_monomer
def test_monomer_dispatch_off_is_todays_program_bitwise():
    """(4): with ``active() == False`` the rebound TemplateEmbedding / SingleTemplateEmbedding run the STOCK bodies: the P=2 program equals the trunk-plan-only
    program (today's) BIT-EXACT, and on a 1-device mesh the stock program BIT-EXACT."""
    n, t, cq = 24, 2, 32
    c, gc, tb, query, mask_2d, f, key, params, stock = _monomer_case(n, t, cq, 1, seed=5)
    out = {}
    for p in (1, 2):
        rm = _rm(p)
        rep = shard.named(rm, shard.replicated_spec())
        for tag, act in (("trunk_only", None), ("off", (lambda: False))):
            if tag == "trunk_only":
                rp_af2.install(rm, M, MM, sites=("trunk",), _test_single_device_mesh=(p == 1))
            else:
                _install_both(rm, p, active=act)
            try:
                jitted = jax.jit(f.apply, in_shardings=(rep,) * 5, out_shardings=rep)
                out[(tag, p)] = np.asarray(jitted(*[shard.put(a, rm) for a in (params, key, query, mask_2d, tb)]))
                if tag == "off":
                    assert rp_t.describe()["template_regions_built"] == 0
            finally:
                _uninstall()
    for p in (1, 2):                                                        # the rebinds are transparent when off: the SAME program as the trunk plan alone
        d = float(np.max(np.abs(out[("off", p)] - out[("trunk_only", p)])))
        RESULTS["cases"].append({"case": "dispatch_off[monomer]", "P": p, "max_abs_diff": d, "bitwise": d == 0.0, "class": "stock", "vs": "trunk_plan_only_program"})
        assert np.array_equal(out[("off", p)], out[("trunk_only", p)]), (p, d)
    d1 = float(np.max(np.abs(out[("off", 1)] - np.asarray(stock))))          # the trunk plan's own 1-device fidelity through the stock template embedding (recorded)
    RESULTS["cases"].append({"case": "trunk_plan_only_vs_stock[monomer,template_embedding]", "P": 1, "max_abs_diff": d1, "bitwise": d1 == 0.0, "class": "trunk_region(1-device)"})
    assert np.array_equal(out[("off", 1)], np.asarray(stock)), d1


# ================================================================================================ (2) multimer TemplateEmbedding
def _multimer_cfg(num_block=1):
    mc = af_config.model_config("model_1_multimer_v3")
    c, gc = mc.model.embeddings_and_evoformer.template, mc.model.global_config
    c.template_pair_stack.num_block = int(num_block)
    gc.use_remat = False
    gc.deterministic = True
    gc.subbatch_size = 4
    gc.bfloat16 = False                                                    # f32 numerics on the CPU box (the bf16 casts of the stock body are transcribed and off here)
    for k, v in (("use_flash_attention", False), ("eval_dropout", False), ("bfloat16_output", False)):
        try:
            setattr(gc, k, v)
        except Exception:  # noqa: BLE001
            pass
    return c, gc


def _multimer_case(n, t, cq, num_block, seed):
    c, gc = _multimer_cfg(num_block)
    rs = np.random.RandomState(seed)
    aatype = rs.randint(0, 21, size=(t, n)).astype("int32")
    pos = (rs.standard_normal((t, n, 37, 3)) * 3.0 + np.arange(n)[None, :, None, None] * np.array([3.8, 0., 0.])[None, None, None, :]).astype("float32")
    amask = (rs.rand(t, n, 37) > 0.1).astype("float32")
    if t > 1:
        pos[1], amask[1], aatype[1] = 0., 0., 0                               # an all-dummy slot
    tb = {"template_aatype": jnp.asarray(aatype), "template_all_atom_positions": jnp.asarray(pos), "template_all_atom_mask": jnp.asarray(amask)}
    query = jnp.asarray(rs.standard_normal((n, n, cq)).astype("float32") + OFFSET)
    seq_mask = (rs.rand(n) > 0.1).astype("float32")
    pad = jnp.asarray(seq_mask[:, None] * seq_mask[None, :])
    asym = np.concatenate([np.zeros(n // 2 + 1), np.ones(n - n // 2 - 1)]).astype("int32")
    multi = jnp.asarray(asym[:, None] == asym[None, :])

    def fwd(q, pm, mm, b):
        return MM.TemplateEmbedding(c, gc)(q, b, pm, mm, is_training=False, safe_key=prng.SafeKey(jax.random.PRNGKey(seed + 1)))
    f = hk.transform(fwd)
    key = jax.random.PRNGKey(seed)
    params = f.init(key, query, pad, multi, tb)
    stock = jax.jit(f.apply)(params, key, query, pad, multi, tb)
    return c, gc, tb, query, pad, multi, f, key, params, stock


@needs_multimer
@pytest.mark.parametrize("n,t,cq", [(16, 1, 32), (24, 2, 64), (32, 4, 64)])
def test_multimer_template_embedding_rows_equal_stock(n, t, cq):
    """(2): stock modules_multimer.TemplateEmbedding on the full pair vs the installed rows region at P = 1 (bit-exact) and every P | N; the output leaves
    row-sharded; the collective census of the P>1 program is recorded."""
    c, gc, tb, query, pad, multi, f, key, params, stock = _multimer_case(n, t, cq, 1, seed=3 * n + t)
    for p in _ps(n):
        rm = _rm(p)
        _install_both(rm, p)
        try:
            jitted, ins = _jit_rows(f.apply, rm, 6, row_args=(2,))
            args = (shard.put(params, rm), shard.put(key, rm), jax.device_put(query, ins[2]), shard.put(pad, rm), shard.put(multi, rm), shard.put(tb, rm))
            got = jitted(*args)
            spec = tuple(getattr(got.sharding, "spec", ())) if p > 1 else None
            hlo = jitted.lower(*args).compile().as_text() if p > 1 else ""
            regions = rp_t.describe()["template_regions_built"]
        finally:
            _uninstall()
        extra = {"N": n, "T": t, "c_z": cq, "num_block": 1, "out_spec": str(spec), "template_regions_built": regions}
        if p > 1:
            census, ncoll = _hlo_census(hlo)
            extra.update({"collectives": census, "n_collectives": ncoll})
            RESULTS["guards"]["multimer_N%d_T%d_P%d" % (n, t, p)] = extra
            assert spec and spec[0] == rm.axis and all(s is None for s in spec[1:]), ("the template embedding must leave the program ROW-sharded", spec)
            assert regions >= 1
        _record("af2_template_embedding[multimer,N=%d,T=%d]" % (n, t), p, stock, got, "row_local+complete_contraction(nested pair stack)", extra=extra)


@needs_multimer
def test_multimer_template_region_without_stack_has_no_collective():
    """(5) for the multimer body: with the template stack made the equality (``layer_stack`` of the library replaced by an equality combinator for this
    test — ``layer_stack(0)`` itself is not traceable) everything in the region is this module's code (``construct_input`` rows, the nine Linears,
    ``query_embedding_norm``, ``output_layer_norm``, the template scan-sum, ReLU, ``output_linear``): the compiled P>1 program holds NO collective (rows
    in, rows out) and equals the stock BIT-EXACT at P=1 / within the row_local class at P>1."""
    n, t, cq = 32, 2, 64
    real = MM.layer_stack.layer_stack
    MM.layer_stack.layer_stack = lambda num_layers, **kw: (lambda f: (lambda x: x))   # noqa: E731 — the equality stack for BOTH the stock and the rows body
    try:
        c, gc, tb, query, pad, multi, f, key, params, stock = _multimer_case(n, t, cq, 1, seed=41)
        for p in _ps(n):
            rm = _rm(p)
            _install_both(rm, p)
            try:
                jitted, ins = _jit_rows(f.apply, rm, 6, row_args=(2,))
                args = (shard.put(params, rm), shard.put(key, rm), jax.device_put(query, ins[2]), shard.put(pad, rm), shard.put(multi, rm), shard.put(tb, rm))
                got = jitted(*args)
                hlo = jitted.lower(*args).compile().as_text() if p > 1 else ""
                spec = tuple(getattr(got.sharding, "spec", ()))
            finally:
                _uninstall()
            if p > 1:
                census, ncoll = _hlo_census(hlo)
                RESULTS["guards"]["multimer_no_stack_N%d_T%d_P%d" % (n, t, p)] = {"ran": True, "collectives": census, "n_collectives": ncoll, "out_spec": str(spec)}
                assert ncoll == 0, "the multimer template region (identity stack) at P=%d holds collectives: %s" % (p, census)
                assert spec and spec[0] == rm.axis and all(s is None for s in spec[1:]), spec
            _record("af2_template_embedding_no_stack[multimer,N=%d,T=%d]" % (n, t), p, stock, got, "row_local", extra={"N": n, "T": t, "stack": "identity"})
    finally:
        MM.layer_stack.layer_stack = real


@needs_multimer
def test_multimer_dispatch_off_is_todays_program_bitwise():
    n, t, cq = 24, 2, 32
    c, gc, tb, query, pad, multi, f, key, params, stock = _multimer_case(n, t, cq, 1, seed=9)
    out = {}
    for p in (1, 2):
        rm = _rm(p)
        rep = shard.named(rm, shard.replicated_spec())
        for tag, act in (("trunk_only", None), ("off", (lambda: False))):
            if tag == "trunk_only":
                rp_af2.install(rm, M, MM, sites=("trunk",), _test_single_device_mesh=(p == 1))
            else:
                _install_both(rm, p, active=act)
            try:
                jitted = jax.jit(f.apply, in_shardings=(rep,) * 6, out_shardings=rep)
                out[(tag, p)] = np.asarray(jitted(*[shard.put(a, rm) for a in (params, key, query, pad, multi, tb)]))
            finally:
                _uninstall()
    for p in (1, 2):
        d = float(np.max(np.abs(out[("off", p)] - out[("trunk_only", p)])))
        RESULTS["cases"].append({"case": "dispatch_off[multimer]", "P": p, "max_abs_diff": d, "bitwise": d == 0.0, "class": "stock", "vs": "trunk_plan_only_program"})
        assert np.array_equal(out[("off", p)], out[("trunk_only", p)]), (p, d)
    d1 = float(np.max(np.abs(out[("off", 1)] - np.asarray(stock))))
    RESULTS["cases"].append({"case": "trunk_plan_only_vs_stock[multimer,template_embedding]", "P": 1, "max_abs_diff": d1, "bitwise": d1 == 0.0, "class": "trunk_region(1-device)"})
    assert np.array_equal(out[("off", 1)], np.asarray(stock)), d1
