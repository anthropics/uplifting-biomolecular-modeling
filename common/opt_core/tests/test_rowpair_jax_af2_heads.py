"""opt_core.mem.rowpair_jax.alphafold_heads — the AF2 pair-reading heads on row blocks: ``install_heads`` installs on the pinned trees and refuses by name
otherwise; with the rows flag DOWN the rebound classes are the stock program bit for bit; with the flag UP inside a region over (pair rows, rest replicated)
the DistogramHead / PredictedAlignedErrorHead logits assembled from the row blocks equal the stock heads on the whole pair, and — on a tree that computes
confidence IN the program — ``get_confidence_metrics`` over the aligned-error logit rows equals the stock (pLDDT / PAE rows / max / pTM / ipTM / ranking);
``transition.symmetrize`` equals the stock ``x + swapaxes(x, -2, -3)`` bit-exact; a whole pair under the rows flag is refused at trace. N in {16, 32},
P in {1, 2, 4, 8} (P=1 through the 1-device test mesh must be BIT-EXACT: the fidelity class). Needs jax + dm-haiku + ``alphafold.model`` (skips BY NAME
without them); ``ROWCHUNK_AF_PATH=<dir holding alphafold/>`` selects a tree ahead of site-packages; ``XLA_FLAGS=--xla_force_host_platform_device_count=8``
on a CPU box. Measured max-abs-diffs and bit-exact-or-not per output go to ``$ROWPAIR_RESULTS``.
"""
import json
import os
import sys
import types

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

from opt_core.mem import MemLeverRefused  # noqa: E402
from opt_core.mem.patchset import PatchSet  # noqa: E402
from opt_core.mem.rowpair_jax import alphafold_heads as H  # noqa: E402
from opt_core.mem.rowpair_jax import mesh, shard, transition  # noqa: E402
from opt_core.mem.rowpair_jax import haiku as rp_hk  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    import ml_collections  # noqa: E402
    from alphafold.model import modules as M  # noqa: E402
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model: %s" % WHY)
LEVER = "rowpair"
RESULTS = {"cases": [], "env": {}, "refusals": {}}
TOL_F32 = float(os.environ.get("ROWPAIR_TOL_F32", "1e-5"))    # row_local class at f32 (cpu; the bodies' arithmetic is the stock's per element)
NS = (16, 32)
FLAG = {"on": False}                                           # the caller's rows flag: raised INSIDE the test's regions only


def rows_active():
    return bool(FLAG["on"])


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path and (RESULTS["cases"] or RESULTS["env"]):
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True)


def _rm(p):
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps():
    """P=1 (a 1-device mesh: the row bodies must equal the stock BIT-EXACT) then every P in {2, 4, 8} the box offers (tolerance class, bit-exact recorded)."""
    return [1] + [p for p in (2, 4, 8) if p <= jax.device_count()]


def _gc():
    return ml_collections.ConfigDict({"zero_init": False, "subbatch_size": 4, "deterministic": True, "use_remat": False, "eval_dropout": False})   # zero_init False: the heads' Linears get non-zero weights


def _in_graph():
    return bool(H.CONFIDENCE_IN_GRAPH[H.file_sha(M)])


def _install(p, ax):
    ps = PatchSet(LEVER)
    sites = H.install_heads(ps, M, lever=LEVER, rows_active=rows_active, axis=ax, n_gpu=p)
    return ps, sites


def _hold(name, p, ref, got, cls, expect_bitwise=False):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    assert ref.shape == got.shape, (name, ref.shape, got.shape)
    mad, scale = float(np.max(np.abs(ref - got))) if ref.size else 0.0, float(np.max(np.abs(ref))) if ref.size else 0.0
    bitwise = bool(np.array_equal(ref, got))
    tol_class = "bitwise(1-device mesh)" if p == 1 else ("bitwise(moves_bytes)" if expect_bitwise else "float32")
    RESULTS["cases"].append({"case": name, "P": p, "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bitwise, "class": cls, "dtype": "float32",
                             "tol_class": tol_class, "shape": list(ref.shape)})
    if p == 1 or expect_bitwise:
        assert bitwise, "%s P=%d is not bitwise vs the stock (max|d|=%g): a transcription differs from the pinned tree" % (name, p, mad)
    else:
        assert mad <= TOL_F32 * max(scale, 1e-30), "%s P=%d: max|d|=%g > %g*%g" % (name, p, mad, TOL_F32, scale)


# ----------------------------------------------------------------------------------------------------------------- pin, refusals, restore
@needs_lib
def test_pin_facts_refusals_and_restore(tmp_path):
    sha = H.file_sha(M)
    assert sha in H.PIN_MODULES and sha in H.HEADS_PIN, sha
    in_graph = _in_graph()
    assert ("confidence" in vars(M)) == in_graph
    env = {"jax": jax.__version__, "haiku": hk.__version__, "platform": jax.devices()[0].platform, "device_count": jax.device_count(),
           "modules_file": M.__file__, "modules_sha": sha, "modules_label": H.PIN_MODULES[sha], "confidence_location": "in_graph" if in_graph else "host"}
    if in_graph:
        C = H.confidence_module(M)
        env.update({"confidence_file": C.__file__, "confidence_sha": H.file_sha(C)})
        assert env["confidence_sha"] in H.PIN_CONFIDENCE
    RESULTS["env"].update(env)
    # a module surface lacking a class of the plan → refused by name (before the pin)
    fake = types.ModuleType("fakepkg.model.modules")
    for nm in H.REQUIRED:
        if nm != "DistogramHead":
            setattr(fake, nm, getattr(M, nm))
    fake.__file__ = M.__file__
    with pytest.raises(MemLeverRefused) as ei:
        H.install_heads(PatchSet(LEVER), fake, lever=LEVER, rows_active=rows_active, axis="row", n_gpu=2)
    assert "lacks DistogramHead" in ei.value.reason
    RESULTS["refusals"]["lacks_class"] = ei.value.reason
    # other bytes → refused by name (the pin)
    other = tmp_path / "modules.py"
    other.write_bytes(b"# not a transcribed tree\n")
    fake2 = types.ModuleType("fakepkg.model.modules")
    for nm in H.REQUIRED:
        setattr(fake2, nm, getattr(M, nm))
    fake2.__file__ = str(other)
    with pytest.raises(MemLeverRefused) as ei:
        H.install_heads(PatchSet(LEVER), fake2, lever=LEVER, rows_active=rows_active, axis="row", n_gpu=2)
    assert "is not a transcribed tree" in ei.value.reason
    RESULTS["refusals"]["unpinned"] = ei.value.reason
    with pytest.raises(MemLeverRefused):
        H.install_heads(PatchSet(LEVER), M, lever=LEVER, rows_active=rows_active, axis="row", n_gpu=0)
    with pytest.raises(MemLeverRefused):
        H.install_heads(PatchSet(LEVER), M, lever=LEVER, rows_active=True, axis="row", n_gpu=2)      # a flag value, not a callable
    # install: the sites of this tree; the attributes change; a second install is refused; restore puts every stock attribute back
    disto_attr, pae_attr = M.DistogramHead.__call__, M.PredictedAlignedErrorHead.__call__
    conf_attrs = {}
    if in_graph:
        C = H.confidence_module(M)
        conf_attrs = {nm: getattr(C, nm) for nm in ("compute_predicted_aligned_error", "predicted_tm_score", "get_confidence_metrics", "compute_plddt")}
    ps = PatchSet(LEVER)
    sites = H.install_heads(ps, M, lever=LEVER, rows_active=rows_active, axis="row", n_gpu=2)
    try:
        want = list(H.SITES) if in_graph else list(H.SITES[:2])
        assert sites == want, sites
        assert M.DistogramHead.__call__ is not disto_attr and M.PredictedAlignedErrorHead.__call__ is not pae_attr
        d = H.describe_heads()
        assert d["confidence"] == env["confidence_location"] and d["conf_sites"] == ",".join(want) and d["heads_library"].startswith("confidence:" if in_graph else "modules:")
        assert d["outputs_rows"] == ",".join(H.OUTPUTS_ROWS[d["confidence"]])
        RESULTS["env"]["describe_heads"] = d
        RESULTS["env"]["sites"] = sites
        with pytest.raises(MemLeverRefused) as ei:
            H.install_heads(PatchSet(LEVER), M, lever=LEVER, rows_active=rows_active, axis="row", n_gpu=2)
        assert "already rebound" in ei.value.reason
    finally:
        restored = ps.restore()
    assert M.DistogramHead.__call__ is disto_attr and M.PredictedAlignedErrorHead.__call__ is pae_attr and len(restored) == len(want)
    for nm, fn in conf_attrs.items():
        assert getattr(H.confidence_module(M), nm) is fn, nm


# ----------------------------------------------------------------------------------------------------------------- the symmetrisation primitive
@needs_lib
@pytest.mark.parametrize("n", NS)
def test_symmetrize_equals_stock_swapaxes(n):
    """``transition.symmetrize`` on row blocks == the DistogramHead's ``x + jnp.swapaxes(x, -2, -3)`` on the whole map, BIT-EXACT at every P (moves_bytes)."""
    rs = np.random.RandomState(10 + n)
    x = jnp.asarray(rs.standard_normal((n, n, 8)).astype("float32"))
    ref = np.asarray(jax.jit(lambda v: v + jnp.swapaxes(v, -2, -3))(x))
    for p in _ps():
        rm = _rm(p); ax = rm.axis
        rows = shard.rows_spec(ax)
        f = jax.jit(shard.shard_map(lambda xl: transition.symmetrize(xl, ax), rm, (rows,), rows), out_shardings=shard.named(rm, rows))
        got = f(shard.put(x, rm))
        assert got.sharding.is_equivalent_to(shard.named(rm, rows), got.ndim), got.sharding
        _hold("symmetrize[N=%d]" % n, p, ref, np.asarray(got), "moves_bytes", expect_bitwise=True)


# ----------------------------------------------------------------------------------------------------------------- the two heads on rows
def _head_rows_vs_stock(n, head_name, make, out_key, edges_key, cz=32):
    rs = np.random.RandomState(20 + n)
    pair = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32"))
    key = jax.random.PRNGKey(0)
    fwd = lambda pr: make()({"pair": pr}, None, False)  # noqa: E731 — (representations, batch, is_training): batch unused by both heads
    params = hk.transform(fwd).init(key, pair)
    stock = jax.jit(hk.transform(fwd).apply)(params, key, pair)
    ref_logits, ref_edges = np.asarray(stock[out_key]), np.asarray(stock[edges_key])
    assert np.any(ref_logits != 0.0), "vacuous reference (zero weights)"
    for p in _ps():
        rm = _rm(p); ax = rm.axis
        rows, rep = shard.rows_spec(ax), shard.replicated_spec()
        ps, _sites = _install(p, ax)
        try:
            FLAG["on"] = False                                             # flag DOWN: the rebound class is the stock program, bit for bit
            off = jax.jit(hk.transform(fwd).apply)(params, key, pair)
            assert np.array_equal(np.asarray(off[out_key]), ref_logits) and np.array_equal(np.asarray(off[edges_key]), ref_edges), "%s: flag down != stock" % head_name
            RESULTS["cases"].append({"case": "%s.flag_down[N=%d]" % (head_name, n), "P": p, "bitwise": True, "max_abs_diff": 0.0, "class": "stock_dispatch"})

            def fwd_rows(pr):                                             # flag UP inside a region over (pair rows) → (logits rows, edges replicated)
                mod = make()
                def body(pair_rows):
                    FLAG["on"] = True
                    try:
                        out = mod({"pair": pair_rows}, None, False)
                    finally:
                        FLAG["on"] = False
                    return out[out_key], out[edges_key]
                return rp_hk.region(body, rm, (rows,), (rows, rep))(shard.constrain(pr, rm))
            reps = shard.named(rm, rep)
            jitted = jax.jit(hk.transform(fwd_rows).apply, in_shardings=(reps, reps, reps), out_shardings=(shard.named(rm, rows), reps))
            got_logits, got_edges = jitted(shard.put(params, rm), shard.put(key, rm), shard.put(pair, rm))
            assert got_logits.sharding.is_equivalent_to(shard.named(rm, rows), got_logits.ndim), got_logits.sharding       # the logits LEAVE the program row-sharded; np.asarray below is the writer boundary
            assert tuple(got_logits.shape) == tuple(ref_logits.shape)
            _hold("%s.logits[N=%d]" % (head_name, n), p, ref_logits, np.asarray(got_logits), H.HEADS_BODIES[head_name + ".__call__"][0])
            _hold("%s.%s[N=%d]" % (head_name, edges_key, n), p, ref_edges, np.asarray(got_edges), "replicated", expect_bitwise=True)
        finally:
            FLAG["on"] = False
            ps.restore()


@needs_lib
@pytest.mark.parametrize("n", NS)
def test_distogram_head_rows_equal_stock(n):
    cfg = ml_collections.ConfigDict({"num_bins": 64, "first_break": 2.3125, "last_break": 21.6875, "weight": 1.0})
    _head_rows_vs_stock(n, "DistogramHead", lambda: M.DistogramHead(cfg, _gc()), "logits", "bin_edges")


@needs_lib
@pytest.mark.parametrize("n", NS)
def test_aligned_error_head_rows_equal_stock(n):
    cfg = ml_collections.ConfigDict({"num_bins": 64, "max_error_bin": 31.0, "weight": 1.0})
    _head_rows_vs_stock(n, "PredictedAlignedErrorHead", lambda: M.PredictedAlignedErrorHead(cfg, _gc()), "logits", "breaks")


# ----------------------------------------------------------------------------------------------------------------- in-program confidence on the logits rows
@needs_lib
@pytest.mark.parametrize("with_asym", [False, True])
@pytest.mark.parametrize("n", NS)
def test_confidence_rows_equal_stock(n, with_asym):
    """``get_confidence_metrics`` (the caller's region form: aligned-error logits as row blocks, the rest replicated) with the rebound ``predicted_tm_score`` /
    ``compute_predicted_aligned_error`` == the stock function on the whole logits: pLDDT / mean (replicated inputs), aligned_confidence_probs + predicted_aligned_error
    (rows, assembled), max, pTM, ipTM (asym_id), ranking. Trees that compute confidence on the host skip BY NAME (nothing runs in their program)."""
    if not _in_graph():
        RESULTS["env"]["confidence_rows_test"] = "skipped: this tree computes confidence on the host from the returned logits (nothing in the program to rebind)"
        pytest.skip("this tree's confidence.py runs on the host (NumPy/SciPy): no in-program confidence to test")
    C = H.confidence_module(M)
    rs = np.random.RandomState(30 + n + int(with_asym))
    logits = jnp.asarray((2.0 * rs.standard_normal((n, n, 64))).astype("float32"))
    breaks = jnp.linspace(0.0, 31.0, 63).astype(jnp.float32)
    mask = (rs.rand(n) > 0.2).astype("float32"); mask[0] = 1.0
    mask = jnp.asarray(mask)
    plddt_logits = jnp.asarray(rs.standard_normal((n, 50)).astype("float32"))
    asym_id = jnp.asarray(np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype("int32"))
    rank_by = "multimer" if with_asym else "ptm"

    def make_pr(lg):
        pae = {"logits": lg, "breaks": breaks}
        if with_asym:
            pae["asym_id"] = asym_id
        return {"predicted_lddt": {"logits": plddt_logits}, "predicted_aligned_error": pae}

    stock_fn = lambda pr, m: C.get_confidence_metrics(pr, m, rank_by=rank_by, use_jnp=True, keep_pae=False)  # noqa: E731
    ref = jax.jit(stock_fn)(make_pr(logits), mask)
    ref = {k: np.asarray(v) for k, v in ref.items()}
    ref_host_ptm = float(C.predicted_tm_score(np.asarray(logits), np.asarray(breaks), residue_weights=np.asarray(mask), use_jnp=False))
    keys_rows3, keys_rows2 = ("aligned_confidence_probs",), ("predicted_aligned_error",)
    for p in _ps():
        rm = _rm(p); ax = rm.axis
        rows3, rows2, rep = shard.rows_spec(ax, 3, 0), shard.rows_spec(ax, 2, 0), shard.replicated_spec()
        ps, _sites = _install(p, ax)
        try:
            FLAG["on"] = False                                             # flag down: the rebound functions are the stock ones (program AND host)
            off = jax.jit(stock_fn)(make_pr(logits), mask)
            for k in ref:
                assert np.array_equal(np.asarray(off[k]), ref[k]), k
            assert float(C.predicted_tm_score(np.asarray(logits), np.asarray(breaks), residue_weights=np.asarray(mask), use_jnp=False)) == ref_host_ptm
            RESULTS["cases"].append({"case": "confidence.flag_down[N=%d,asym=%d]" % (n, int(with_asym)), "P": p, "bitwise": True, "max_abs_diff": 0.0, "class": "stock_dispatch"})

            specs = {"predicted_lddt": {"logits": rep}, "predicted_aligned_error": {"logits": rows3, "breaks": rep}}
            out_specs = {"plddt": rep, "mean_plddt": rep, "ranking_confidence": rep, "aligned_confidence_probs": rows3, "predicted_aligned_error": rows2,
                         "max_predicted_aligned_error": rep, "ptm": rep}
            if with_asym:
                specs["predicted_aligned_error"]["asym_id"] = rep
                out_specs["iptm"] = rep

            def body(pr_, m_):
                FLAG["on"] = True
                try:
                    return C.get_confidence_metrics(pr_, m_, rank_by=rank_by, use_jnp=True, keep_pae=False)
                finally:
                    FLAG["on"] = False
            pr = make_pr(logits)
            f = jax.jit(shard.shard_map(body, rm, (specs, rep), out_specs),
                        out_shardings={k: shard.named(rm, v) for k, v in out_specs.items()})
            got = f(shard.put(pr, rm), shard.put(mask, rm))
            assert set(got) == set(ref), (sorted(got), sorted(ref))
            for k in sorted(ref):
                if k in keys_rows3:
                    assert got[k].sharding.is_equivalent_to(shard.named(rm, rows3), 3), (k, got[k].sharding)
                if k in keys_rows2:
                    assert got[k].sharding.is_equivalent_to(shard.named(rm, rows2), 2), (k, got[k].sharding)
                cls = ("row_local(stock_body)" if k in ("aligned_confidence_probs", "predicted_aligned_error", "max_predicted_aligned_error")
                       else "row_local+moves_bytes(gather:[N])" if k in ("ptm", "iptm") else "replicated(stock)")
                _hold("confidence.%s[N=%d,asym=%d]" % (k, n, int(with_asym)), p, ref[k], np.asarray(got[k]), cls)
        finally:
            FLAG["on"] = False
            ps.restore()


# ----------------------------------------------------------------------------------------------------------------- fail loud: a whole pair under the rows flag
@needs_lib
def test_rows_gate_refuses_whole_pair_under_flag():
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices")
    n, cz, p = 16, 8, 2
    rm = _rm(p); ax = rm.axis
    rep = shard.replicated_spec()
    cfg = ml_collections.ConfigDict({"num_bins": 8, "first_break": 2.3125, "last_break": 21.6875, "weight": 1.0})
    pair = jnp.zeros((n, n, cz), jnp.float32)
    key = jax.random.PRNGKey(0)
    fwd = lambda pr: M.DistogramHead(cfg, _gc())({"pair": pr}, None, False)["logits"]  # noqa: E731
    params = hk.transform(fwd).init(key, pair)
    ps, _sites = _install(p, ax)
    try:
        def fwd_whole(pr):                                                # the pair enters the region REPLICATED (whole [N, N, C] per device) under the flag: refused at trace
            mod = M.DistogramHead(cfg, _gc())
            def body(pair_whole):
                FLAG["on"] = True
                try:
                    return mod({"pair": pair_whole}, None, False)["logits"]
                finally:
                    FLAG["on"] = False
            return rp_hk.region(body, rm, (rep,), rep)(pr)
        with pytest.raises(MemLeverRefused) as ei:
            jax.jit(hk.transform(fwd_whole).apply)(shard.put(params, rm), shard.put(key, rm), shard.put(pair, rm))
        assert "rows are active" in ei.value.reason and "DistogramHead" in ei.value.reason
        RESULTS["refusals"]["whole_pair_under_flag"] = ei.value.reason
        if _in_graph():
            C = H.confidence_module(M)
            logits = jnp.zeros((n, n, 8), jnp.float32); breaks = jnp.linspace(0.0, 31.0, 7)
            def body2(lg, br):
                FLAG["on"] = True
                try:
                    return C.predicted_tm_score(lg, br, use_jnp=True)
                finally:
                    FLAG["on"] = False
            with pytest.raises(MemLeverRefused) as ei:
                jax.jit(shard.shard_map(body2, rm, (rep, rep), rep))(logits, breaks)
            assert "predicted_tm_score" in ei.value.reason
            RESULTS["refusals"]["whole_logits_under_flag"] = ei.value.reason
    finally:
        FLAG["on"] = False
        ps.restore()
