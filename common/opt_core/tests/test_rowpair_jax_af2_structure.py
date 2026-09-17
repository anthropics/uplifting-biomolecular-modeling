"""opt_core.mem.rowpair_jax.alphafold — the structure-module share of the AF2 recipe: ``InvariantPointAttention.__call__`` of the pinned
``alphafold.model.folding`` (QuatAffine IPA, both monomer trees) and ``alphafold.model.folding_multimer`` (Rigid3Array IPA) rebound to bodies that
run the stock arithmetic with the QUERY residues restricted to this device's pair ROW BLOCK; the rebound method is the stock body bit for bit outside
the heads region; the whole ``StructureModule`` fed pair rows inside a region equals the dense module per output leaf; and the pinned folding
sources hold no pair consumer beyond the audited ones. Needs jax + dm-haiku + ``alphafold.model`` (skips BY NAME without them);
``XLA_FLAGS=--xla_force_host_platform_device_count=8`` on a CPU box (P ∈ {1, 2, 4, 8}); the tree under test is ``$ROWCHUNK_AF_PATH`` (a directory
holding ``alphafold/``) when set, else site-packages. Measured max-abs-diffs go to ``$ROWPAIR_RESULTS``."""
import contextlib
import hashlib
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
_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")             # eight host devices for THIS process's jax backend on a CPU box (harmless on gpu):
if "xla_force_host_platform_device_count" not in (_XLA_FLAGS_BEFORE or ""):   # set, initialise the backend, restore — the variable never leaks into the session
    os.environ["XLA_FLAGS"] = ((_XLA_FLAGS_BEFORE or "") + " --xla_force_host_platform_device_count=8").strip()
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

from opt_core.mem.rowpair_jax import alphafold as rp_af2  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402
from opt_core.mem.rowpair_jax import mesh, shard  # noqa: E402

try:
    import haiku as hk  # noqa: E402
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import ml_collections  # noqa: E402
    import numpy as np  # noqa: E402
    from alphafold.model import folding as FD  # noqa: E402
    from alphafold.model import modules as M  # noqa: E402
    try:
        from alphafold.model import folding_multimer as FDM  # noqa: E402
        from alphafold.model import modules_multimer as MM  # noqa: E402
    except Exception:  # noqa: BLE001 — the monomer tree has neither
        FDM, MM = None, None
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
    FD = FDM = M = MM = None
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model: %s" % WHY)
needs_multimer = pytest.mark.skipif(not HAVE or FDM is None, reason="needs alphafold.model.folding_multimer (the 2.3.x tree): %s" % (WHY or "monomer tree"))
RESULTS = {"cases": [], "env": {}, "sites": {}, "audit": {}, "refusals": []}
TOL_F32 = float(os.environ.get("ROWPAIR_TOL_F32", "1e-5"))                       # f32 at exact f32 matmul precision (cpu; gpu at HIGHEST)
TOL_F32_DEFAULT = float(os.environ.get("ROWPAIR_TOL_F32_DEFAULT", str(2.0 ** -10)))  # f32 operands at the GPU backend's DEFAULT matmul precision
OFFSET = float(os.environ.get("ROWPAIR_AF2_OFFSET", "3.0"))            # non-zero-mean activations (the structure module's LayerNorms see them)


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path and (RESULTS["cases"] or RESULTS["audit"]):
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True)


def _rm(p):
    if p > jax.device_count():                                          # a P-device mesh needs P devices on THIS backend (1-GPU boxes run P=1 only)
        pytest.skip("needs %d jax devices on the default backend (%s offers %d): the CPU box with XLA_FLAGS=--xla_force_host_platform_device_count=8, "
                    "or a %d-GPU box" % (p, jax.devices()[0].platform, jax.device_count(), p))
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps():
    """P=1 (a 1-device mesh through the test-only hook: the rows body must equal the stock BIT-EXACT) then every P>1 the box offers."""
    return [1] + [p for p in (2, 4, 8) if p <= jax.device_count()]


def _gc():
    return ml_collections.ConfigDict({"zero_init": False, "subbatch_size": 4, "deterministic": True, "use_remat": False, "eval_dropout": False,
                                       "use_flash_attention": False, "bfloat16": False, "bfloat16_output": False})


def _sm_cfg():
    """A small structure-module config with every field the pinned StructureModule / FoldIteration / IPA / MultiRigidSidechain read at inference."""
    return ml_collections.ConfigDict({"num_layer": 2, "num_channel": 32, "num_head": 4, "num_scalar_qk": 8, "num_point_qk": 4, "num_scalar_v": 8,
                                       "num_point_v": 4, "num_layer_in_transition": 2, "dropout": 0.0, "position_scale": 10.0,
                                       "sidechain": {"num_channel": 16, "num_residual_block": 1}, "compute_in_graph_metrics": False})


@contextlib.contextmanager
def _heads_flag(on):
    """Force the heads-region flag the rebound IPA dispatches on: through the recipe's trace state when it carries one (``_S.heads_rows``), else by
    binding the module-level predicate ``heads_rows`` the bodies call."""
    S = getattr(rp_af2, "_S", None)
    if S is not None and hasattr(S, "heads_rows") and callable(getattr(rp_af2, "heads_rows", None)):
        old = S.heads_rows
        S.heads_rows = bool(on)
        try:
            yield
        finally:
            S.heads_rows = old
    else:
        had, old = hasattr(rp_af2, "heads_rows"), getattr(rp_af2, "heads_rows", None)
        rp_af2.heads_rows = (lambda: True) if on else (lambda: False)
        try:
            yield
        finally:
            if had:
                rp_af2.heads_rows = old
            else:
                delattr(rp_af2, "heads_rows")


@contextlib.contextmanager
def _installed(p):
    """The AF2 plan on a P-device mesh (P=1 through the test-only hook) plus the structure-module sites; everything restored on exit."""
    rm = _rm(p)
    patches = rp_af2.install(rm, M, MM, sites=("trunk", "model"), _test_single_device_mesh=(p == 1))   # the structure sites are installed explicitly below (install(sites⊇heads) installs them itself)
    try:
        sites = rp_af2._install_structure(patches, {"folding": FD, "folding_multimer": FDM}, patches.lever)
        RESULTS["sites"][str(p)] = sites
        yield rm, sites
    finally:
        rp_af2.uninstall()


def _hold(name, p, ref, got, cls, n):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    assert ref.shape == got.shape, (name, ref.shape, got.shape)
    mad, scale = float(np.max(np.abs(ref - got))) if ref.size else 0.0, float(np.max(np.abs(ref))) if ref.size else 0.0
    bitwise = bool(np.array_equal(ref, got))
    tol_class = "bitwise(1-device mesh)" if p == 1 else ("float32:default" if jax.devices()[0].platform == "gpu" else "float32")
    tol = 0.0 if p == 1 else (TOL_F32_DEFAULT if tol_class == "float32:default" else TOL_F32)
    RESULTS["cases"].append({"case": name, "P": p, "N": n, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, "class": cls,
                             "dtype": "float32", "precision": "default", "tol_class": tol_class, "offset": OFFSET})
    if p == 1:
        assert bitwise, "%s on a 1-device mesh is not bitwise vs the stock body (max|d|=%g): a transcription differs from the pinned tree" % (name, mad)
    else:
        assert mad <= tol * max(scale, 1e-30), "%s P=%d tol_class=%s: max|d|=%g > %g*%g" % (name, p, tol_class, mad, tol, scale)


def _leaves(tree):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in flat]


def _ipa_inputs(n, seed, c1=32, cz=16):
    rs = np.random.RandomState(seed)
    x1 = jnp.asarray(rs.standard_normal((n, c1)).astype("float32") + OFFSET)
    x2 = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32") + OFFSET)
    m = jnp.asarray((rs.rand(n, 1) > 0.15).astype("float32"))
    quat = rs.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    frame = jnp.asarray(np.concatenate([quat, 3.0 * rs.standard_normal((n, 3))], axis=-1).astype("float32"))   # [N, 7]: unit quaternion + translation
    return x1, x2, m, frame


def _quat_affine(frame):
    return FD.quat_affine.QuatAffine.from_tensor(frame)                 # as FoldIteration builds it from activations['affine'] (folding.py:320)


def _rigid(frame):
    g = FDM.geometry
    rot = g.Rot3Array.from_quaternion(frame[:, 0], frame[:, 1], frame[:, 2], frame[:, 3], normalize=True)
    return g.Rigid3Array(rot, g.Vec3Array(frame[:, 4], frame[:, 5], frame[:, 6]))


# ----------------------------------------------------------------------------------------------------------------- pins + install surface
@needs_lib
def test_folding_pins_and_install_surface():
    pins = dict(rp_af2.PIN)
    pins.update(getattr(rp_af2, "SM_PIN", {}))
    RESULTS["env"].update({"jax": jax.__version__, "haiku": hk.__version__, "platform": jax.devices()[0].platform, "device_count": jax.device_count(),
                           "modules_file": M.__file__, "modules_sha": rp_af2.file_sha(M), "folding_file": FD.__file__, "folding_sha": rp_af2.file_sha(FD),
                           "folding_multimer_sha": rp_af2.file_sha(FDM) if FDM is not None else None, "multimer": FDM is not None,
                           "sm_bodies": {k: list(v) for k, v in rp_af2.SM_BODIES.items()}})
    assert rp_af2.file_sha(M) in rp_af2.PIN, ("modules.py is not a pinned tree", rp_af2.file_sha(M))
    assert rp_af2.file_sha(FD) in pins, ("folding.py is not a pinned tree", rp_af2.file_sha(FD))
    if FDM is not None:
        assert rp_af2.file_sha(FDM) in pins, ("folding_multimer.py is not a pinned tree", rp_af2.file_sha(FDM))
    stock_fd = FD.InvariantPointAttention.__call__
    stock_fdm = FDM.InvariantPointAttention.__call__ if FDM is not None else None
    with _installed(2) as (_rm2, sites):
        assert "folding." + rp_af2.SM_SITE in sites and FD.InvariantPointAttention.__call__ is not stock_fd
        assert (FDM is None) == ("folding_multimer." + rp_af2.SM_SITE not in sites)
        for s in sites:
            assert s in rp_af2.SM_BODIES, (s, sorted(rp_af2.SM_BODIES))
    assert FD.InvariantPointAttention.__call__ is stock_fd and (FDM is None or FDM.InvariantPointAttention.__call__ is stock_fdm)


# ----------------------------------------------------------------------------------------------------------------- (1) IPA rows == stock IPA
def _ipa_case(flavour, n, p):
    """stock IPA on the full pair vs the rebound IPA inside a region over (1-D: replicated, pair: rows, mask: replicated, frame: replicated) with the heads flag on."""
    Fm, make_frame, word = (FD, _quat_affine, "affine") if flavour == "affine" else (FDM, _rigid, "rigid")
    cfg, gc = _sm_cfg(), _gc()
    x1, x2, m, frame = _ipa_inputs(n, seed=10 + n)

    def fwd(x1_, x2_, m_, fr_):
        return Fm.InvariantPointAttention(cfg, gc)(inputs_1d=x1_, inputs_2d=x2_, mask=m_, **{word: make_frame(fr_)})
    key = jax.random.PRNGKey(0)
    params = hk.transform(fwd).init(key, x1, x2, m, frame)
    stock = jax.jit(hk.transform(fwd).apply)(params, key, x1, x2, m, frame)
    with _installed(p) as (rm, _sites):
        ax = rm.axis
        rows, rep = shard.rows_spec(ax), shard.replicated_spec()

        def fwd_rows(x1_, x2_, m_, fr_):
            mod = Fm.InvariantPointAttention(cfg, gc)

            def body(x1_b, x2_l, m_b, fr_b):
                return mod(inputs_1d=x1_b, inputs_2d=x2_l, mask=m_b, **{word: make_frame(fr_b)})
            return rp_hk.region(body, rm, (rep, rows, rep, rep), rep)(x1_, shard.constrain(x2_, rm), m_, fr_)
        with _heads_flag(True):
            got = rp_hk.jit_apply(hk.transform(fwd_rows).apply, rm, n_args=6)(
                shard.put(params, rm), shard.put(key, rm), shard.put(x1, rm), shard.put(x2, rm), shard.put(m, rm), shard.put(frame, rm))
            got = np.asarray(got)
    return stock, got


@needs_lib
@pytest.mark.parametrize("n", [16, 32])
def test_ipa_affine_rows_equal_stock(n):
    for p in _ps():
        stock, got = _ipa_case("affine", n, p)
        _hold("af2_ipa[folding,%d]" % n, p, stock, got, "row_local", n)


@needs_multimer
@pytest.mark.parametrize("n", [16, 32])
def test_ipa_rigid_rows_equal_stock(n):
    for p in _ps():
        stock, got = _ipa_case("rigid", n, p)
        _hold("af2_ipa[folding_multimer,%d]" % n, p, stock, got, "row_local", n)


# ----------------------------------------------------------------------------------------------------------------- (3) dispatch: flag off = the stock body
@needs_lib
def test_rebound_ipa_is_stock_outside_the_heads_region():
    """Installed, heads flag OFF, called on the FULL pair outside any region: the rebound method must be the stock body bit-exact (both flavours)."""
    n = 16
    for flavour in (["affine", "rigid"] if FDM is not None else ["affine"]):
        Fm, make_frame, word = (FD, _quat_affine, "affine") if flavour == "affine" else (FDM, _rigid, "rigid")
        cfg, gc = _sm_cfg(), _gc()
        x1, x2, m, frame = _ipa_inputs(n, seed=7)

        def fwd(x1_, x2_, m_, fr_):
            return Fm.InvariantPointAttention(cfg, gc)(inputs_1d=x1_, inputs_2d=x2_, mask=m_, **{word: make_frame(fr_)})
        key = jax.random.PRNGKey(1)
        params = hk.transform(fwd).init(key, x1, x2, m, frame)
        stock = np.asarray(jax.jit(hk.transform(fwd).apply)(params, key, x1, x2, m, frame))
        with _installed(2):
            with _heads_flag(False):
                got = np.asarray(jax.jit(hk.transform(fwd).apply)(params, key, x1, x2, m, frame))   # a fresh function object: traced under the patch
        RESULTS["cases"].append({"case": "af2_ipa_dispatch_off[%s]" % flavour, "P": 2, "N": n, "max_abs_diff": float(np.max(np.abs(stock - got))),
                                 "ref_max_abs": float(np.max(np.abs(stock))), "bitwise": bool(np.array_equal(stock, got)), "class": "stock_dispatch",
                                 "dtype": "float32", "precision": "default", "tol_class": "bitwise", "offset": OFFSET})
        assert np.array_equal(stock, got), "the rebound %s IPA with the heads flag off is not the stock body bitwise" % flavour


# ----------------------------------------------------------------------------------------------------------------- (2) the whole StructureModule on pair rows
def _monomer_batch(n, rs):
    return {"seq_mask": jnp.asarray((rs.rand(n) > 0.1).astype("float32")), "aatype": jnp.asarray(rs.randint(0, 20, size=(n,)).astype("int32")),
            "atom14_atom_exists": jnp.asarray((rs.rand(n, 14) > 0.3).astype("float32")), "atom37_atom_exists": jnp.asarray((rs.rand(n, 37) > 0.3).astype("float32")),
            "residx_atom37_to_atom14": jnp.asarray(rs.randint(0, 14, size=(n, 37)).astype("int32"))}


def _structure_module_case(flavour, n, p):
    """dense StructureModule vs the same module inside a region fed the pair ROW BLOCK (single / batch / key replicated), heads flag on; per output leaf."""
    Fm = FD if flavour == "affine" else FDM
    cfg, gc = _sm_cfg(), _gc()
    rs = np.random.RandomState(100 + n)
    single = jnp.asarray(rs.standard_normal((n, 24)).astype("float32") + OFFSET)
    pair = jnp.asarray(rs.standard_normal((n, n, 16)).astype("float32") + OFFSET)
    batch = _monomer_batch(n, rs)
    names = sorted(batch)
    key = jax.random.PRNGKey(0)
    skey = jax.random.PRNGKey(42)                                        # the structure module's safe_key, explicit (drawn outside the region — the stock draws it only when absent)

    def call(single_, pair_, skey_, *fields):
        b = dict(zip(names, fields))
        if flavour == "affine":
            return Fm.StructureModule(cfg, gc, compute_loss=True)({"single": single_, "pair": pair_}, b, is_training=False, safe_key=Fm.prng.SafeKey(skey_))
        return Fm.StructureModule(cfg, gc)({"single": single_, "pair": pair_}, b, is_training=False, safe_key=Fm.prng.SafeKey(skey_), compute_loss=True)

    def fwd(single_, pair_, skey_, *fields):
        return call(single_, pair_, skey_, *fields)
    args = [batch[k] for k in names]
    params = hk.transform(fwd).init(key, single, pair, skey, *args)
    dense = jax.jit(hk.transform(fwd).apply)(params, key, single, pair, skey, *args)
    with _installed(p) as (rm, _sites):
        ax = rm.axis
        rows, rep = shard.rows_spec(ax), shard.replicated_spec()

        def fwd_rows(single_, pair_, skey_, *fields):
            def body(single_b, pair_l, skey_b, *fields_b):
                return call(single_b, pair_l, skey_b, *fields_b)
            return rp_hk.region(body, rm, (rep, rows, rep) + (rep,) * len(fields), rep)(single_, shard.constrain(pair_, rm), skey_, *fields)
        with _heads_flag(True):
            got = rp_hk.jit_apply(hk.transform(fwd_rows).apply, rm, n_args=5 + len(args))(
                shard.put(params, rm), shard.put(key, rm), shard.put(single, rm), shard.put(pair, rm), shard.put(skey, rm), *[shard.put(a, rm) for a in args])
            got = jax.device_get(got)
    return _leaves(dense), _leaves(got)


def _compare_tree(tag, p, n, dense, got):
    assert [k for k, _ in dense] == [k for k, _ in got], ([k for k, _ in dense], [k for k, _ in got])
    assert dense, "the structure module returned no leaves"
    for (k, a), (_k, b) in zip(dense, got):
        if not np.issubdtype(np.asarray(a).dtype, np.floating):
            assert np.array_equal(np.asarray(a), np.asarray(b)), (tag, k)
            continue
        _hold("%s%s" % (tag, k), p, a, b, "row_local", n)


@needs_lib
@pytest.mark.parametrize("n", [16, 32])
def test_structure_module_affine_pair_rows_equal_dense(n):
    for p in _ps():
        dense, got = _structure_module_case("affine", n, p)
        _compare_tree("af2_structure_module[folding,%d]" % n, p, n, dense, got)


@needs_multimer
@pytest.mark.parametrize("n", [16, 32])
def test_structure_module_rigid_pair_rows_equal_dense(n):
    for p in _ps():
        dense, got = _structure_module_case("rigid", n, p)
        _compare_tree("af2_structure_module[folding_multimer,%d]" % n, p, n, dense, got)


# ----------------------------------------------------------------------------------------------------------------- (4) no other pair consumer in the pinned trees
PAIR_TOKENS = ("inputs_2d", "static_feat_2d", "act_2d", "['pair']", '["pair"]')
PAIR_READS = {   # sha256 of the pinned folding source → every line (number, text) naming a pair-shaped array: the audited set (docstrings and comments included)
    "85295f2ca8fa9b7ff7a369e645ee4e948e33132f246d509cf002dbf9ff75c1ac": [          # the 2.3.x multimer-era tree's folding.py (PyPI alphafold 2.3.13 build)
        [72, "def __call__(self, inputs_1d, inputs_2d, mask, affine):"], [90, "inputs_2d: (N, M, C') 2D input embedding, used for biases and values."],
        [211, "inputs_2d)"], [262, "# c = inputs_2d channels"], [265, "result_attention_over_2d = jnp.einsum('hij, ijc->ihc', attn, inputs_2d)"],
        [305, "static_feat_2d=None,"], [327, "inputs_2d=static_feat_2d,"], [436, "act_2d = common_modules.LayerNorm("], [441, "representations['pair'])"],
        [449, "static_feat_2d=act_2d,"]],
    "6eb571fba2812c54ae65c465bc1c1f8bdb92131e2a635dabd7834cefe6f028d5": [          # the 2.3.x multimer-era tree's folding_multimer.py
        [224, "inputs_2d: jnp.ndarray,"], [244, "inputs_2d: (N, M, C') 2D input embedding, used for biases values in the"],
        [309, "num_head, name='attention_2d')(inputs_2d)"], [358, "# c = inputs_2d channels"],
        [361, "result_attention_over_2d = jnp.einsum('ijh, ijc->ihc', attn, inputs_2d)"], [400, "static_feat_2d: Optional[jnp.ndarray] = None,"],
        [423, "inputs_2d=static_feat_2d,"], [526, "act_2d = common_modules.LayerNorm("], [531, "representations['pair'])"], [538, "static_feat_2d=act_2d,"]],
    "63e943be2e57b5da47fe0ee13a41fbd21f497fa2bdc9e7b67941b14df648a06e": [          # dl_binder_design cafa3853 alphafold/model/folding.py
        [74, "def __call__(self, inputs_1d, inputs_2d, mask, affine):"], [92, "inputs_2d: (N, M, C') 2D input embedding, used for biases and values."],
        [213, "inputs_2d)"], [264, "# c = inputs_2d channels"], [267, "result_attention_over_2d = jnp.einsum('hij, ijc->ihc', attn, inputs_2d)"],
        [307, "static_feat_2d=None,"], [329, "inputs_2d=static_feat_2d,"], [438, "act_2d = hk.LayerNorm("], [443, "representations['pair'])"],
        [451, "static_feat_2d=act_2d,"]],
}


def _pair_reads(path):
    with open(path, "rb") as fh:
        src = fh.read()
    sha = hashlib.sha256(src).hexdigest()
    hits = [[i, line.strip()] for i, line in enumerate(src.decode("utf-8").splitlines(), 1) if any(t in line for t in PAIR_TOKENS)]
    return sha, hits


@needs_lib
def test_no_other_pair_consumer_in_the_pinned_structure_module():
    """Every statement of the folding sources under test that names a pair-shaped array is one of the audited reads (pair_layer_norm, the FoldIteration
    pass-through, IPA's attention_2d / result_attention_over_2d and their docstring/comment mentions). A differently pinned tree — or a new consumer —
    fails here with the observed list, so it is audited before the rows body serves it."""
    for mod in [FD] + ([FDM] if FDM is not None else []):
        sha, hits = _pair_reads(mod.__file__)
        RESULTS["audit"][mod.__name__] = {"sha": sha, "reads": hits}
        assert sha in PAIR_READS, "%s sha256 %s is not an audited folding tree; observed pair reads: %r" % (mod.__file__, sha, hits)
        assert hits == PAIR_READS[sha], (mod.__file__, hits)
