"""opt_core.mem.rowpair_jax.rowchunk — the AF2-family row-chunked TriangleMultiplication equals the stock sub-layer (fused and unfused forms).

Needs jax + dm-haiku + the AlphaFold-2 model library (``alphafold.model.modules``): skips BY NAME without them. ``ROWCHUNK_AF_PATH`` (optional)
is prepended to ``sys.path`` so the same file runs against a source tree of the library (the earlier class form whose ``__call__`` is the
unfused body) in a separate process. Measured max-abs-diffs go to ``$ROWPAIR_RESULTS`` (json) when set.
"""
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

from opt_core.mem import Ledger, MemLeverRefused  # noqa: E402
from opt_core.mem.rowpair_jax import rowchunk  # noqa: E402
from opt_core.mem.patchset import PatchSet  # noqa: E402

RESULTS = {"cases": [], "env": {}}


def test_rowchunk_refuses_without_library_by_name():
    """In a process without the model library the lever is a refusal naming it (never an ImportError)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import builtins\n_imp = builtins.__import__\n"
            "def fake(name, *a, **k):\n    if name.startswith('alphafold'):\n        raise ImportError('no alphafold here')\n    return _imp(name, *a, **k)\n"
            "builtins.__import__ = fake\n"
            "from opt_core.mem import MemLeverRefused\nfrom opt_core.mem.rowpair_jax import rowchunk\n"
            "try:\n    rowchunk.check()\nexcept MemLeverRefused as e:\n    print('REFUSED', e.lever, e.reason); sys.exit(0)\nsys.exit(1)\n") % CORE_DIR
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "REFUSED trimul_chunk alphafold.model.modules not importable" in r.stdout, r.stdout + r.stderr


_XLA_FLAGS_BEFORE = os.environ.get("XLA_FLAGS")             # four host devices for THIS process's jax backend on a CPU box; set, initialise, restore
if "xla_force_host_platform_device_count" not in (_XLA_FLAGS_BEFORE or ""):
    os.environ["XLA_FLAGS"] = ((_XLA_FLAGS_BEFORE or "") + " --xla_force_host_platform_device_count=4").strip()
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

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    from alphafold.model import modules as af_modules  # noqa: E402
    import ml_collections  # noqa: E402
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model (the AF2 model library): %s" % WHY)


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path and RESULTS["cases"]:
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True)


def _configs(equation, fused):
    cfg = ml_collections.ConfigDict({"equation": equation, "num_intermediate_channel": 8, "fuse_projection_weights": bool(fused),
                                     "dropout_rate": 0.0, "orientation": "per_row", "shared_dropout": True})
    gc = ml_collections.ConfigDict({"zero_init": False, "subbatch_size": 4, "deterministic": True, "use_remat": False, "eval_dropout": False})
    return cfg, gc


@needs_lib
@pytest.mark.parametrize("equation", [rowchunk.OUTGOING, rowchunk.INCOMING])
@pytest.mark.parametrize("fused", [True, False])
def test_chunked_equals_stock(equation, fused):
    form = rowchunk.form_of(af_modules.TriangleMultiplication)
    if form == "call" and fused:
        pytest.skip("this library form has no fused body (form=call): the unfused case covers it")
    facts = rowchunk.check()
    RESULTS["env"].update({"jax": jax.__version__, "platform": jax.devices()[0].platform, "form": form, **facts,
                           "library_file": getattr(af_modules, "__file__", "?")})
    n, cz = 32, 16
    cfg, gc = _configs(equation, fused)
    rs = np.random.RandomState(0)
    act = jnp.asarray(rs.standard_normal((n, n, cz)).astype("float32"))
    mask = jnp.asarray((rs.rand(n, n) > 0.15).astype("float32"))

    def forward(a, m):
        return af_modules.TriangleMultiplication(cfg, gc)(a, m, is_training=False)

    def fresh():                                    # a NEW transform per phase: jax's trace cache keys on the function object, and the class is looked up at trace
        return hk.transform(lambda a, m: forward(a, m))
    model = fresh()
    key = jax.random.PRNGKey(0)
    params = model.init(key, act, mask)
    stock = jax.jit(model.apply)(params, key, act, mask)

    ledger, shapes = Ledger(), []
    patches = PatchSet(rowchunk.LEVER)
    base = af_modules.TriangleMultiplication
    patches.replace(af_modules, "TriangleMultiplication", rowchunk.chunked_class(base, rows=n // 4, ledger=ledger, shapes=shapes))
    try:
        assert af_modules.TriangleMultiplication.__name__ == "TriangleMultiplication"
        params2 = fresh().init(key, act, mask)                             # init through the chunked class: the stock path (names unchanged)
        assert jax.tree_util.tree_structure(params2) == jax.tree_util.tree_structure(params), (sorted(params2), sorted(params))
        chunked = jax.jit(fresh().apply)(params, key, act, mask)
        patches.restore()
        big = PatchSet("t2")
        big.replace(af_modules, "TriangleMultiplication", rowchunk.chunked_class(base, rows=n, ledger=ledger, shapes=shapes))   # rows >= N: the stock body, counted
        whole = jax.jit(fresh().apply)(params, key, act, mask)
        big.restore()
    finally:
        patches.restore()
    assert af_modules.TriangleMultiplication is base
    ref, got = np.asarray(stock, np.float64), np.asarray(chunked, np.float64)
    mad = float(np.max(np.abs(ref - got)))
    scale = float(np.max(np.abs(ref)))
    RESULTS["cases"].append({"case": "rowchunk[%s,%s,%s]" % (equation, "fused" if fused else "unfused", form), "rows": n // 4, "n": n,
                             "max_abs_diff": mad, "ref_max_abs": scale, "bitwise": bool(np.array_equal(ref, got)), "class": rowchunk.NUMERICS_CLASS})
    assert mad <= 1e-4 * scale, (mad, scale)
    assert np.array_equal(np.asarray(whole), np.asarray(stock))            # rows >= N is the stock body exactly
    facts_l = ledger.facts()
    assert facts_l.get("trimul_chunk_chunked_traces", 0) >= 1 and facts_l.get("trimul_chunk_stock_traces", 0) >= 2, facts_l
    assert shapes and shapes[-1]["form"] in ("fused", "unfused", "call") and any(s["chunked"] for s in shapes)


@needs_lib
def test_chunked_refusals():
    base = af_modules.TriangleMultiplication
    with pytest.raises(MemLeverRefused):
        rowchunk.chunked_class(base, rows=0)
    cfg, gc = _configs("abc,abd->acd", False)
    n = 16
    act, mask = jnp.zeros((n, n, 4)), jnp.ones((n, n))
    model = hk.transform(lambda a, m: af_modules.TriangleMultiplication(cfg, gc)(a, m, is_training=False))
    patches = PatchSet(rowchunk.LEVER)
    patches.replace(af_modules, "TriangleMultiplication", rowchunk.chunked_class(base, rows=4))
    try:
        cfg_ok, _ = _configs(rowchunk.OUTGOING, False)
        params = hk.transform(lambda a, m: af_modules.TriangleMultiplication(cfg_ok, gc)(a, m, is_training=False)).init(jax.random.PRNGKey(0), act, mask)   # same parameter names
        with pytest.raises(MemLeverRefused) as ei:
            model.apply(params, None, act, mask)
        assert "is neither" in ei.value.reason
    finally:
        patches.restore()
