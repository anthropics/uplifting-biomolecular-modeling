"""opt_core.mem.rowpair_jax.alphafold — the model / heads groups of the AF2 recipe: install words refused by name, every stock attribute restored,
and the WHOLE MODEL (``modules.AlphaFold`` monomer pTM on both pinned trees; ``modules_multimer.AlphaFold`` on the 2.3.x tree) at P=1 (nothing installed)
vs P∈{2,4,8} installed with heads=conf=sharded requested: every output leaf compared (max|diff| recorded per leaf), the SHARDING FACTS of the program
(the recycled pair in, the prologue producers, the trunk→heads pair, the pair-shaped outputs are ``P('row', …)``) asserted from inside the jit, the recycle
carry across two passes, and the named refusal of an N that does not divide by P. CPU box: ``XLA_FLAGS=--xla_force_host_platform_device_count=8``; tree (b)
with ``ROWCHUNK_AF_PATH=<dir containing alphafold/>``. Results JSON to ``$ROWPAIR_RESULTS``."""
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
from opt_core.mem.rowpair_jax import mesh, shard  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import haiku as hk  # noqa: E402
    from alphafold.model import modules as M  # noqa: E402
    from alphafold.model import config as af_config  # noqa: E402
    from alphafold.model import folding as F  # noqa: E402
    from alphafold.common import residue_constants as RC  # noqa: E402
    try:
        from alphafold.model import modules_multimer as MM  # noqa: E402
        from alphafold.model import folding_multimer as FM  # noqa: E402
    except Exception:  # noqa: BLE001
        MM, FM = None, None
    HAVE, WHY = True, ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
needs_lib = pytest.mark.skipif(not HAVE, reason="needs jax + dm-haiku + alphafold.model: %s" % WHY)
RESULTS = {"cases": [], "env": {}, "leaves": [], "sharding": [], "refusals": []}
TOL = float(os.environ.get("ROWPAIR_AF2_MODEL_TOL", "1e-4"))        # whole model on CPU f32: complete contractions per element, sums reordered by the row split
import inspect  # noqa: E402
IN_GRAPH_RECYCLE = HAVE and "initial_guess" in inspect.signature(M.AlphaFold.__call__).parameters   # the dl_binder_design tree: recycling inside the program (hk.while_loop)
TREE = "dl" if IN_GRAPH_RECYCLE else "cf"


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    path = os.environ.get("ROWPAIR_RESULTS")
    if path:
        with open(path, "w") as fh:
            json.dump(RESULTS, fh, indent=1, sort_keys=True, default=str)


@pytest.fixture(autouse=True)
def _uninstalled():
    rp_af2.uninstall()
    yield
    rp_af2.uninstall()


def _rm(p):
    if p > jax.device_count():                                          # a P-device mesh needs P devices on THIS backend (1-GPU boxes run P=1 only)
        pytest.skip("needs %d jax devices on the default backend (%s offers %d): the CPU box with XLA_FLAGS=--xla_force_host_platform_device_count=8, "
                    "or a %d-GPU box" % (p, jax.devices()[0].platform, jax.device_count(), p))
    return mesh.build(p, platform=jax.devices()[0].platform, _allow_single_device_mesh=(p == 1))


def _ps():
    return [p for p in (2, 4, 8) if p <= jax.device_count()]


# ---------------------------------------------------------------------------------------------------------------- install words
@needs_lib
def test_install_words_refusals_and_restore():
    RESULTS["env"].update({"jax": jax.__version__, "haiku": hk.__version__, "platform": jax.devices()[0].platform, "devices": jax.device_count(),
                           "modules_file": M.__file__, "modules_sha": rp_af2.file_sha(M), "folding_sha": rp_af2.file_sha(F), "tree": TREE,
                           "multimer": bool(MM), "pin_labels": sorted(set(rp_af2.PIN.values()))})
    assert rp_af2.file_sha(M) in rp_af2.PIN and rp_af2.file_sha(F) in rp_af2.PIN
    rm = _rm(2)
    owners = {"AlphaFoldIteration.__call__": M.AlphaFoldIteration.__call__, "EmbeddingsAndEvoformer.__call__": M.EmbeddingsAndEvoformer.__call__,
              "EvoformerIteration.__call__": M.EvoformerIteration.__call__, "modules.hk": M.hk, "folding.InvariantPointAttention.__call__": F.InvariantPointAttention.__call__,
              "DistogramHead.__call__": M.DistogramHead.__call__, "PredictedAlignedErrorHead.__call__": M.PredictedAlignedErrorHead.__call__}
    if hasattr(M, "confidence"):
        owners["modules.confidence"] = M.confidence
    if MM is not None:
        owners.update({"multimer.AlphaFoldIteration.__call__": MM.AlphaFoldIteration.__call__, "multimer.EmbeddingsAndEvoformer.__call__": MM.EmbeddingsAndEvoformer.__call__,
                       "multimer.hk": MM.hk, "folding_multimer.InvariantPointAttention.__call__": FM.InvariantPointAttention.__call__})
    with pytest.raises(MemLeverRefused) as ei:
        rp_af2.install(rm, M, MM, pin={"0" * 64: "nothing"})
    assert "is not a transcribed tree" in ei.value.reason
    patches = rp_af2.install(rm, M, MM, heads="sharded", conf="sharded")
    try:
        d = rp_af2.describe()
        RESULTS["env"]["describe_default"] = d
        for k in ("heads", "conf", "structure", "structure_reason", "conf_reason", "recycle", "template", "template_reason", "msa", "masks", "outputs_rows"):
            assert k in d, k
        assert d["recycle"] == "rows" and d["msa"] == "replicated" and d["masks"] == "replicated"
        assert d["structure"] in ("sharded", "replicated") and d["conf"] in ("sharded", "replicated")
        assert (d["structure"] == "sharded") == ("_install_structure" in vars(rp_af2)), d              # effective word follows the presence of the IPA row bodies
        assert (d["conf"] == "sharded") == (rp_af2._optional_module(rp_af2.HEADS_MODULE) is not None), d
        assert (d["template"] == "rows") == (rp_af2._optional_module(rp_af2.TEMPLATE_MODULE) is not None), d
        assert "AlphaFoldIteration.__call__" in d["sites"] and "EmbeddingsAndEvoformer.__call__" in d["sites"] and "modules.hk" in d["sites"], d["sites"]
        assert M.AlphaFoldIteration.__call__ is not owners["AlphaFoldIteration.__call__"] and M.hk is not owners["modules.hk"]
    finally:
        names = rp_af2.uninstall()
    RESULTS["env"]["restored"] = names
    for name, obj in owners.items():
        assert _owner_now(name) is obj, name
    # the trunk group alone: today's words
    rp_af2.install(rm, M, MM, sites=("trunk",))
    d = rp_af2.describe()
    assert d["heads"] == "not_installed" and d["recycle"] == "replicated" and d["template"] == "replicated" and M.hk is owners["modules.hk"]
    rp_af2.uninstall()
    rp_af2.install(rm, M, MM, heads="replicated", conf="replicated")
    d = rp_af2.describe()
    assert d["structure"] == "replicated" and d["structure_reason"].startswith("requested") and d["conf"] == "replicated" and d["conf_reason"] == "requested"
    assert ("multimer_stock_outside_region" in d["structure_reason"]) == (MM is not None), d["structure_reason"]


def _owner_now(name):
    F_ = F
    table = {"AlphaFoldIteration.__call__": lambda: M.AlphaFoldIteration.__call__, "EmbeddingsAndEvoformer.__call__": lambda: M.EmbeddingsAndEvoformer.__call__,
             "EvoformerIteration.__call__": lambda: M.EvoformerIteration.__call__, "modules.hk": lambda: M.hk, "modules.confidence": lambda: M.confidence,
             "folding.InvariantPointAttention.__call__": lambda: F_.InvariantPointAttention.__call__, "DistogramHead.__call__": lambda: M.DistogramHead.__call__,
             "PredictedAlignedErrorHead.__call__": lambda: M.PredictedAlignedErrorHead.__call__,
             "multimer.AlphaFoldIteration.__call__": lambda: MM.AlphaFoldIteration.__call__, "multimer.EmbeddingsAndEvoformer.__call__": lambda: MM.EmbeddingsAndEvoformer.__call__,
             "multimer.hk": lambda: MM.hk, "folding_multimer.InvariantPointAttention.__call__": lambda: FM.InvariantPointAttention.__call__}
    return table[name]()


# ---------------------------------------------------------------------------------------------------------------- the whole model
def _tiny_monomer_config():
    cfg = af_config.model_config("model_3_ptm")             # pTM heads (distogram, aligned error, pLDDT, masked MSA, experimentally resolved, structure); no templates
    c, gc = cfg.model, cfg.model.global_config
    c.embeddings_and_evoformer.evoformer_num_block = 1
    c.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    c.heads.structure_module.num_layer = 2
    c.num_recycle = 1
    if "resample_msa_in_recycling" in c:
        c.resample_msa_in_recycling = False
    for k, v in (("deterministic", True), ("use_remat", False), ("subbatch_size", 4), ("zero_init", False), ("bfloat16", False), ("bfloat16_output", False),
                 ("use_flash_attention", False), ("eval_dropout", False)):
        if k in gc:
            gc[k] = v
    return cfg


def _monomer_batch(n, s=6, se=10, seed=0):
    """The monomer model's processed features for ONE ensemble slice (leading dim E=1), synthetic and self-consistent (per-residue atom tables from
    residue_constants for the drawn aatype)."""
    rng = np.random.RandomState(seed)
    aatype = rng.randint(0, 20, size=(n,)).astype(np.int32)
    f = {}
    f["aatype"] = aatype[None]
    f["residue_index"] = np.arange(n, dtype=np.int32)[None]
    f["seq_length"] = np.array([n], np.int32)
    tf_ = np.zeros((n, 22), np.float32)
    tf_[np.arange(n), aatype + 1] = 1.0
    f["target_feat"] = tf_[None]
    f["msa_feat"] = rng.rand(1, s, n, 49).astype(np.float32)
    f["seq_mask"] = np.ones((1, n), np.float32)
    f["msa_mask"] = np.ones((1, s, n), np.float32)
    f["msa_row_mask"] = np.ones((1, s), np.float32)
    f["extra_msa"] = rng.randint(0, 22, size=(1, se, n)).astype(np.int32)
    f["extra_has_deletion"] = np.zeros((1, se, n), np.float32)
    f["extra_deletion_value"] = np.zeros((1, se, n), np.float32)
    f["extra_msa_mask"] = np.ones((1, se, n), np.float32)
    f["atom14_atom_exists"] = RC.restype_atom14_mask[aatype].astype(np.float32)[None]
    f["atom37_atom_exists"] = RC.restype_atom37_mask[aatype].astype(np.float32)[None]
    f["residx_atom37_to_atom14"] = RC.restype_atom37_to_atom14[aatype].astype(np.int32)[None] if hasattr(RC, "restype_atom37_to_atom14") else _atom37_to_atom14(aatype)[None]
    f["residx_atom14_to_atom37"] = RC.restype_atom14_to_atom37[aatype].astype(np.int32)[None] if hasattr(RC, "restype_atom14_to_atom37") else np.zeros((1, n, 14), np.int32)
    return {k: jnp.asarray(v) for k, v in f.items()}


def _atom37_to_atom14(aatype):
    out = np.zeros((len(aatype), 37), np.int32)
    for i, a in enumerate(aatype):
        names = RC.restype_name_to_atom14_names[RC.restype_1to3[RC.restypes[a]]]
        for j, nm in enumerate(names):
            if nm:
                out[i, RC.atom_order[nm]] = j
    return out


class _Spy:
    """A Haiku method interceptor recording, from INSIDE the jitted program, the sharding XLA assigns at the prologue sites (operand and output of the modules
    named in ``rp_af2.PROLOGUE_PAIR_MODULES``), the recycled pair entering ``EmbeddingsAndEvoformer`` and the pair it returns (the trunk → heads boundary)."""
    def __init__(self):
        self.rec = []

    def _inspect(self, tag, x):
        def cb(s):
            spec = getattr(s, "spec", None)
            self.rec.append({"site": tag, "sharding": type(s).__name__, "spec": None if spec is None else [None if p is None else str(p) for p in spec]})
        try:
            jax.debug.inspect_array_sharding(x, callback=cb)
        except Exception as e:  # noqa: BLE001
            self.rec.append({"site": tag, "inspect_error": repr(e)})

    def __call__(self, next_f, args, kwargs, context):
        out = next_f(*args, **kwargs)
        if getattr(context, "method_name", "") != "__call__" or rp_af2.in_manual_region():
            return out
        name = str(getattr(context.module, "module_name", "")).split("/")[-1]
        if name in rp_af2.PROLOGUE_PAIR_MODULES:
            self._inspect(name + ".operand", args[0])
            self._inspect(name + ".out", out)
        elif name == rp_af2.EVOFORMER_SCOPE and isinstance(out, dict) and "pair" in out:
            b = args[0] if args else kwargs.get("batch")
            if isinstance(b, dict) and "prev_pair" in b:
                self._inspect("evoformer.prev_pair_in", b["prev_pair"])
            self._inspect("evoformer.pair_out(trunk_to_heads)", out["pair"])
        return out


def _transformed(cfg, spy, in_graph_recycle):
    """``types.SimpleNamespace(init=, apply=, spy=)`` of the transformed whole model with the sharding spy interceptor active around it."""
    def fwd(batch):
        with hk.intercept_methods(spy):
            model = M.AlphaFold(cfg.model)
            if in_graph_recycle:
                return model(batch, is_training=False, compute_loss=False, ensemble_representations=False, return_representations=True, initial_guess=None)
            return model(batch, is_training=False, compute_loss=False, ensemble_representations=False, return_representations=True)
    tr = hk.transform(fwd)
    return types.SimpleNamespace(init=tr.init, apply=tr.apply, spy=spy)


def _leaves(tree):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): v for path, v in flat}


def _compare(case, p, ref_tree, got_tree):
    ref, got = _leaves(ref_tree), _leaves(got_tree)
    assert sorted(ref) == sorted(got), (sorted(set(ref) ^ set(got)))
    worst = 0.0
    for k in sorted(ref):
        a, b = np.asarray(ref[k], np.float64), np.asarray(got[k], np.float64)
        assert a.shape == b.shape, (k, a.shape, b.shape)
        mad = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        spec = getattr(getattr(got[k], "sharding", None), "spec", None)
        RESULTS["leaves"].append({"case": case, "P": p, "leaf": k, "shape": list(a.shape), "max_abs_diff": mad, "ref_max_abs": scale,
                                  "bitwise": bool(np.array_equal(a, b)), "out_spec": None if spec is None else [None if q is None else str(q) for q in spec]})
        assert np.all(np.isfinite(b)) or not np.all(np.isfinite(a)), k
        assert mad <= TOL * max(1.0, scale), "%s P=%d leaf %s: max|d|=%g > %g*max(1,%g)" % (case, p, k, mad, TOL, scale)
        worst = max(worst, mad / max(1.0, scale))
    return worst


def _place(tree, rm):
    return shard.put(tree, rm)


def _run_sharded(case, p, transformed, params, key, batches, ref_outs, expect_rows_out):
    """Install at P, jit through wrap_runner's placement, run every batch of ``batches`` (a recycle sequence feeds ``prev``), compare each pass to ``ref_outs``."""
    rm = _rm(p)
    rp_af2.install(rm, M, MM, heads="sharded", conf="sharded")
    try:
        d = rp_af2.describe()
        for word in ("conf_reason", "structure_reason", "template_reason"):    # the family is whole: no sibling module may be absent (an *_absent word is a packaging defect)
            assert "absent" not in str(d[word]), (word, d[word])
        assert d["conf"] == "sharded" and d["structure"] == "sharded" and d["template"] == "rows" and d["recycle"] == "rows" and d["heads"] == "sharded", d
        spy = transformed.spy
        spy.rec.clear()
        runner = types.SimpleNamespace(apply=transformed.apply)
        rp_af2.wrap_runner(runner, rm, n_args=3)
        params_p = _place(params, rm)
        prev, outs = None, []
        for i, batch in enumerate(batches):
            b = dict(_place(batch, rm))
            if prev is not None:
                b["prev"] = prev                                               # as returned: the recycled pair stays in its row layout between the passes
            out = runner.apply(params_p, _place(key, rm), b)
            outs.append(out)
            prev = out.get("prev") if isinstance(out, dict) else None
            worst = _compare(case + "[pass%d]" % i, p, ref_outs[i], out)
            RESULTS["cases"].append({"case": case, "pass": i, "P": p, "worst_rel": worst, "describe": d if i == 0 else None, "tree": TREE})
        d = rp_af2.describe()
        RESULTS["cases"][-1]["describe_after"] = d
        got = _leaves(outs[-1])
        for leaf in expect_rows_out:                                            # the pair-shaped outputs LEAVE row-sharded
            spec = getattr(getattr(got[leaf], "sharding", None), "spec", None)
            assert spec is not None and str(spec[0]) == rm.axis, (leaf, spec)
        rows_sites = [r for r in spy.rec if "spec" in r]
        RESULTS["sharding"].append({"case": case, "P": p, "sites": spy.rec[:], "outputs_rows": d["outputs_rows"], "inputs_rows": d["inputs_rows"]})
        assert rows_sites, "no sharding facts recorded from inside the program"
        for r in rows_sites:                                                   # SHARDING FACTS: prologue operands/outputs, prev_pair in, trunk→heads pair are P('row', …)
            assert r["spec"] and r["spec"][0] == rm.axis, r
        seen = {r["site"] for r in rows_sites}
        for must in ("prev_pair_norm.out", "evoformer.pair_out(trunk_to_heads)"):
            assert must in seen, (must, sorted(seen))
        if len(batches) > 1 or IN_GRAPH_RECYCLE:
            assert "evoformer.prev_pair_in" in seen, sorted(seen)                  # the recycled pair ENTERS the embedder row-sharded (jit boundary or in-graph loop carry)
        return outs, d
    finally:
        rp_af2.uninstall()


@needs_lib
@pytest.mark.parametrize("n", [16, 32])
def test_whole_model_monomer_ptm(n):
    """``modules.AlphaFold`` (monomer pTM) on this tree: P=1 stock vs every P>1 installed (heads=conf=sharded requested; effective words recorded), two passes
    (the second feeds ``prev`` back = the out-of-graph recycle) on the 2.3.x tree, one call with the in-graph recycle loop (num_recycle=1) on the dl_binder_design tree."""
    cfg = _tiny_monomer_config()
    spy = _Spy()
    transformed = _transformed(cfg, spy, IN_GRAPH_RECYCLE)
    batch = _monomer_batch(n)
    key = jax.random.PRNGKey(0)
    params = jax.jit(transformed.init)(jax.random.PRNGKey(42), batch)
    n_params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
    ref_apply = jax.jit(transformed.apply)
    batches = [batch] if IN_GRAPH_RECYCLE else [batch, batch]
    ref_outs, prev = [], None
    for b in batches:
        b = dict(b)
        if prev is not None:
            b["prev"] = prev
        out = ref_apply(params, key, b)
        ref_outs.append(out)
        prev = out.get("prev")
    leaves = sorted(_leaves(ref_outs[-1]))
    RESULTS["env"]["monomer_n%d" % n] = {"n_params": n_params, "leaves": leaves, "spy_p1": spy.rec[:]}
    expect = ["representations.pair", "distogram.logits"]
    expect += ["prev.prev_pair", "predicted_aligned_error", "aligned_confidence_probs"] if not IN_GRAPH_RECYCLE else ["predicted_aligned_error.logits"]
    for leaf in expect:
        assert leaf in leaves, (leaf, leaves)
    for p in _ps():
        if n % p:
            with pytest.raises(MemLeverRefused) as ei:                          # N not a multiple of P under the model group: refused BY NAME (the kit pads)
                _run_sharded("monomer_ptm[n=%d]" % n, p, transformed, params, key, batches, ref_outs, expect)
            assert "not divisible by n_gpu=%d" % p in ei.value.reason, ei.value.reason
            RESULTS["refusals"].append({"case": "monomer_ptm[n=%d]" % n, "P": p, "reason": ei.value.reason})
            continue
        _run_sharded("monomer_ptm[n=%d]" % n, p, transformed, params, key, batches, ref_outs, expect)


@needs_lib
def test_n_not_multiple_of_p_is_refused_by_name():
    if 8 > jax.device_count():
        pytest.skip("needs 8 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=8); have %d" % jax.device_count())
    cfg = _tiny_monomer_config()
    spy = _Spy()
    transformed = _transformed(cfg, spy, IN_GRAPH_RECYCLE)
    batch = _monomer_batch(20)
    key = jax.random.PRNGKey(0)
    params = jax.jit(transformed.init)(jax.random.PRNGKey(1), batch)
    with pytest.raises(MemLeverRefused) as ei:
        _run_sharded("monomer_ptm[n=20]", 8, transformed, params, key, [batch], [None], [])
    assert "num_rows=20 not divisible by n_gpu=8" in ei.value.reason, ei.value.reason
    RESULTS["refusals"].append({"case": "n=20,P=8", "reason": ei.value.reason})


# ---------------------------------------------------------------------------------------------------------------- the multimer model (2.3.x tree)
def _tiny_multimer_config():
    cfg = af_config.model_config("model_1_multimer_v3")
    c, gc = cfg.model, cfg.model.global_config
    c.embeddings_and_evoformer.evoformer_num_block = 1
    c.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    c.embeddings_and_evoformer.num_msa = 4
    c.embeddings_and_evoformer.num_extra_msa = 5
    c.embeddings_and_evoformer.template.enabled = False
    c.heads.structure_module.num_layer = 2
    c.num_recycle = 1
    c.num_ensemble_eval = 1
    if "resample_msa_in_recycling" in c:
        c.resample_msa_in_recycling = False
    if "calc_extended_ptm" not in c:                                          # the 2.3.x multimer model reads it (modules_multimer.py:468); the kit's config builder sets it
        with c.unlocked():
            c.calc_extended_ptm = False
    for k, v in (("deterministic", True), ("use_remat", False), ("subbatch_size", 4), ("zero_init", False), ("bfloat16", False), ("bfloat16_output", False),
                 ("eval_dropout", False)):
        if k in gc:
            gc[k] = v
    return cfg


def _multimer_batch(n, s=9, t_=1, seed=0):
    rng = np.random.RandomState(seed)
    aatype = rng.randint(0, 20, size=(n,)).astype(np.int32)
    half = n // 2
    f = {"aatype": aatype, "residue_index": np.concatenate([np.arange(half), np.arange(n - half)]).astype(np.int32),
         "asym_id": np.concatenate([np.ones(half), 2 * np.ones(n - half)]).astype(np.int32), "sym_id": np.ones(n, np.int32),
         "entity_id": np.concatenate([np.ones(half), 2 * np.ones(n - half)]).astype(np.int32), "seq_mask": np.ones(n, np.float32),
         "msa": rng.randint(0, 21, size=(s, n)).astype(np.int32), "msa_mask": np.ones((s, n), np.float32), "deletion_matrix": np.zeros((s, n), np.float32),
         "cluster_bias_mask": np.concatenate([np.ones(1), np.zeros(s - 1)]).astype(np.float32), "bert_mask": np.zeros((s, n), np.float32),
         "seq_length": np.array(n, np.int32), "template_aatype": np.zeros((t_, n), np.int32), "template_all_atom_positions": np.zeros((t_, n, 37, 3), np.float32),
         "template_all_atom_mask": np.zeros((t_, n, 37), np.float32), "all_atom_positions": (rng.rand(n, 37, 3) * 10).astype(np.float32),
         "all_atom_mask": RC.restype_atom37_mask[aatype].astype(np.float32), "entity_mask": np.ones(n, np.float32), "iter": np.array(0, np.int32)}
    f["msa"][0] = aatype
    out = {k: jnp.asarray(v) for k, v in f.items()}
    out["prev"] = {"prev_msa_first_row": jnp.zeros([n, 256]), "prev_pair": jnp.zeros([n, n, 128]), "prev_pos": jnp.zeros([n, 37, 3])}   # the 2.3.x multimer model expects `prev` in the batch on every pass (modules_multimer.py:446-451)
    return out


@needs_lib
@pytest.mark.parametrize("n", [16, 32])
def test_whole_model_multimer(n):
    """``modules_multimer.AlphaFold`` on the 2.3.x tree: P=1 stock vs every P>1 installed, two passes (prev fed back); a tree without ``modules_multimer`` skips BY NAME."""
    if MM is None:
        pytest.skip("this tree has no alphafold.model.modules_multimer (monomer library): the multimer model is the 2.3.x tree's")
    cfg = _tiny_multimer_config()
    spy = _Spy()

    def fwd(batch):
        with hk.intercept_methods(spy):
            return MM.AlphaFold(cfg.model)(batch, is_training=False, return_representations=True)
    tr = hk.transform(fwd)
    transformed = types.SimpleNamespace(init=tr.init, apply=tr.apply, spy=spy)
    batch = _multimer_batch(n)
    key = jax.random.PRNGKey(0)
    try:
        params = jax.jit(transformed.init)(jax.random.PRNGKey(7), batch)
        ref_apply = jax.jit(transformed.apply)
        ref_outs, prev = [], None
        for b in (batch, batch):
            b = dict(b)
            if prev is not None:
                b["prev"] = prev
            out = ref_apply(params, key, b)
            ref_outs.append(out)
            prev = out.get("prev")
        jax.tree_util.tree_map(np.asarray, ref_outs)
    except Exception as e:  # noqa: BLE001 — the STOCK model refused the synthetic features: a named non-result, not a verdict on the plan
        RESULTS["cases"].append({"case": "multimer[n=%d]" % n, "P": 1, "stock_run_failed": repr(e)[:400]})
        pytest.skip("stock multimer model did not run on the synthetic features (named non-result): %s" % repr(e)[:200])
    leaves = sorted(_leaves(ref_outs[-1]))
    RESULTS["env"]["multimer_n%d" % n] = {"leaves": leaves}
    expect = [leaf for leaf in ("representations.pair", "distogram.logits", "prev.prev_pair", "predicted_aligned_error", "aligned_confidence_probs") if leaf in leaves]
    assert "representations.pair" in leaves and "distogram.logits" in leaves, leaves
    for p in _ps():
        if n % p:
            continue
        _run_sharded("multimer[n=%d]" % n, p, transformed, params, key, [batch, batch], ref_outs, expect)


# ---------------------------------------------------------------------------------------------------------------- M1: the prologue is PLACED (HLO census)
def _full_pair_shapes(hlo, n, min_c=15):
    """Instruction result shapes in the partitioned HLO text that are a WHOLE pair-class tensor: ``[n, n, C]`` / ``[*, n, n, C]`` with ``C >= min_c`` (the partitioned
    program's shapes are per device, so a whole ``[N, N, c]`` shape anywhere = a replicated materialisation). ``[N, N]`` masks and ``[N, N, H<min_c]`` biases are the named classes."""
    import re
    found = {}
    body = "\n".join(l for l in hlo.splitlines() if " parameter(" not in l and not l.startswith(("HloModule", "ENTRY")))   # activations only: parameters are replicated by design; the module header repeats every shape
    for m in re.finditer(r"(?:f32|bf16|f16|s32|pred)\[([0-9,]+)\]", body):
        dims = [int(d) for d in m.group(1).split(",") if d]
        if len(dims) == 3 and dims[0] == n and dims[1] == n and dims[2] >= min_c:
            found[tuple(dims)] = found.get(tuple(dims), 0) + 1
        elif len(dims) == 4 and dims[1] == n and dims[2] == n and dims[3] >= min_c:
            found[tuple(dims)] = found.get(tuple(dims), 0) + 1
    return found


def _collectives(hlo):
    import re
    out = {}
    for m in re.finditer(r"=\s*(?:f32|bf16|f16|s32|pred|\()[^\n]*?\b(all-gather|all-reduce|all-to-all|reduce-scatter|collective-permute)(?:-start)?\(", hlo):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


@needs_lib
@pytest.mark.parametrize("n", [40])                                          # 40: divisible by 2, 4, 8 and equal to no channel size of the model (a shape census keys on N)
def test_prologue_and_trunk_hold_no_whole_pair_hlo_census(n):
    """``EmbeddingsAndEvoformer`` alone (prologue + extra-MSA stack + Evoformer; templates off) jitted at every P>1 with the recycled pair entering row-sharded: the COMPILED,
    partitioned program holds NO whole ``[N, N, c>=15]`` tensor anywhere (no replicated materialisation of the pair, the recycled pair, the relative-position one-hot,
    the recycled-position distogram) — the census that the prologue is placed, not propagated by luck; the ``[N, N]`` masks and ``[N, N, H]`` biases are the
    named classes and are recorded."""
    cfg = _tiny_monomer_config()
    ec, gc = cfg.model.embeddings_and_evoformer, cfg.model.global_config

    def fwd(batch):
        return M.EmbeddingsAndEvoformer(ec, gc)(batch, is_training=False)
    tr = hk.transform(fwd)
    full = _monomer_batch(n)
    b0 = {k: v[0] for k, v in full.items()}
    b0.update({"prev_pos": jnp.zeros([n, 37, 3]), "prev_msa_first_row": jnp.zeros([n, ec.msa_channel]), "prev_pair": jnp.zeros([n, n, ec.pair_channel])})
    key = jax.random.PRNGKey(0)
    params = jax.jit(tr.init)(jax.random.PRNGKey(3), b0)
    ref = jax.jit(tr.apply)(params, key, b0)
    for p in _ps():
        if n % p:
            continue
        rm = _rm(p)
        rp_af2.install(rm, M, MM, heads="sharded", conf="sharded")
        try:
            args = (_place(params, rm), _place(key, rm), _place(b0, rm))
            ins, outs = rp_af2.apply_shardings(rm, args, jax.eval_shape(tr.apply, *args))
            args = tuple(jax.device_put(a, s) for a, s in zip(args, ins))     # the recycled pair enters in its row layout (jit_sharded does this for a runner)
            jitted = jax.jit(tr.apply, in_shardings=ins, out_shardings=outs)
            hlo = jitted.lower(*args).compile().as_text()
            got = jitted(*args)
        finally:
            rp_af2.uninstall()
        whole = _full_pair_shapes(hlo, n)
        masks_biases = _full_pair_shapes(hlo, n, min_c=0)
        RESULTS["sharding"].append({"case": "hlo_census[E&E,n=%d]" % n, "P": p, "whole_pair_shapes": {str(k): v for k, v in whole.items()},
                                    "nn_class_shapes(named: masks/biases)": {str(k): v for k, v in masks_biases.items() if k not in whole}, "collectives": _collectives(hlo),
                                    "out_pair_spec": [str(x) for x in got["pair"].sharding.spec], "hlo_bytes": len(hlo)})
        assert not whole, "P=%d: whole [N,N,c] tensors in the partitioned program: %s" % (p, whole)
        assert str(got["pair"].sharding.spec[0]) == rm.axis
        _compare("E&E[n=%d]" % n, p, ref, got)


# ---------------------------------------------------------------------------------------------------------------- M2: loop carries are selected BY NAME
@needs_lib
@pytest.mark.parametrize("n,s", [(24, 24), (24, 7)])
def test_loop_carry_constrained_by_name_not_shape(n, s):
    """``_constrain_loop_carry``: ``prev_pair`` / ``pair`` leaves come out row-sharded; an ``msa`` leaf of shape ``[S, N, c]`` is untouched even when S == N; an array under an
    unknown name that looks pair-shaped is refused BY NAME at trace."""
    ps = _ps()
    if not ps:
        pytest.skip("needs >= 2 jax devices on the default backend (%s offers %d)" % (jax.devices()[0].platform, jax.device_count()))
    p = ps[0]
    rm = _rm(p)
    rp_af2.install(rm, M, MM)
    try:
        carry = (jnp.zeros([], jnp.int32), {"prev_pair": jnp.ones([n, n, 8]), "pair": jnp.ones([n, n, 8]), "msa": jnp.ones([s, n, 8]), "prev_pos": jnp.ones([n, 37, 3]),
                                             "prev_msa_first_row": jnp.ones([n, 8])})
        out = jax.jit(rp_af2._constrain_loop_carry)(carry)
        specs = {k: getattr(getattr(v, "sharding", None), "spec", None) for k, v in out[1].items()}
        RESULTS["cases"].append({"case": "carry_by_name[n=%d,s=%d]" % (n, s), "P": p, "specs": {k: (None if v is None else [str(x) for x in v]) for k, v in specs.items()}})
        for k in ("prev_pair", "pair"):
            assert specs[k] is not None and len(specs[k]) >= 1 and str(specs[k][0]) == rm.axis, (k, specs[k])
        for k in ("msa", "prev_pos", "prev_msa_first_row"):
            assert specs[k] is None or len(specs[k]) == 0 or all(x is None for x in specs[k]), (k, specs[k])     # untouched: no row axis on any dim
        assert bool(jnp.all(out[1]["msa"] == 1.0))
        with pytest.raises(MemLeverRefused) as ei:
            jax.jit(rp_af2._constrain_loop_carry)((jnp.zeros([], jnp.int32), {"mystery_pair": jnp.ones([n, n, 8])}))
        assert "mystery_pair" in ei.value.reason and "looks pair-shaped" in ei.value.reason, ei.value.reason
        RESULTS["refusals"].append({"case": "carry unknown pair-shaped leaf", "reason": ei.value.reason})
    finally:
        rp_af2.uninstall()
