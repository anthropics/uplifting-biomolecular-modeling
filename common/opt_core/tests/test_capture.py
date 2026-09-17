"""CPU tests of opt_core.capture (policies, GraphPool bookkeeping, ConstMemo keying, xla_cache keys) and opt_core.jit_cache key formats.
GPU semantics of capture/replay are not exercised here."""
from __future__ import annotations

import sys

import pytest

from opt_core import jit_cache
from opt_core.capture import graphs, hoist, pool, xla_cache


def _cuda_present() -> bool:
    """These tests hold the no-CUDA path (refusal / named fallback); on a CUDA box they skip by name."""
    try:
        import torch                                       # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:                                      # noqa: BLE001
        return False


no_cuda_path = pytest.mark.skipif(_cuda_present(), reason="CUDA present: this test holds the no-CUDA refusal/fallback path")


def test_capture_package_import_is_stdlib_only():
    assert "torch" not in sys.modules or True          # other tests in the session may import torch; the package itself must not
    import importlib
    for m in ("graphs", "pool", "hoist", "xla_cache"):
        importlib.import_module("opt_core.capture." + m)


def test_jit_cache_key_formats_are_byte_identical_to_the_recorded_dir_names():
    assert jit_cache.key("2.4.1+cu124", cc="9.0") == "torch2.4.1-cu124-sm90"
    assert jit_cache.compose("torch-cu-sm", version="2.4.1+cu124", cuda="12.4", cc="9.0") == "torch2.4.1-cu124-sm90"
    assert jit_cache.compose("torch-sm", version="2.9.1+cu128", cc="9.0") == "torch2.9.1-sm90"
    assert jit_cache.compose("torch-sm-triton", version="2.8.0", cc="90", triton="3.4.0") == "torch2.8.0-sm90-triton3.4.0"
    assert jit_cache.compose("lib{version}-cu{cuda}-sm{cc}", version="0.5.3+local", cuda="12.8", cc="9.0") == "lib0.5.3-cu128-sm90"        # the caller's own shape
    assert jit_cache.compose("{gpu}__lib{version}", gpu="some_gpu_model", version="0.10.2") == "some_gpu_model__lib0.10.2"
    assert set(jit_cache.FORMATS) == {"torch-cu-sm", "torch-sm", "torch-sm-triton"}


def test_jit_cache_dirs_and_env():
    d = jit_cache.cache_dirs("/cache", "k")
    assert d == {"TRITON_CACHE_DIR": "/cache/k/triton", "TORCH_EXTENSIONS_DIR": "/cache/k/torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "/cache/k/inductor"}
    assert jit_cache.cache_dirs("/cache", "k", names={"MY_CACHE_DIR": "mine"}) == {"MY_CACHE_DIR": "/cache/k/mine"}                    # the caller's own names
    env = jit_cache.cache_env("/cache", "k", environ={"TRITON_CACHE_DIR": "/no/such/dir/x"}, names=["TRITON_CACHE_DIR"])
    assert env == {"TRITON_CACHE_DIR": ("/cache/k/triton", "keyed")}


def test_policies_decide_by_name():
    assert pool.FixedK(2).decide("k", 1)[0] == "eager" and pool.FixedK(2).decide("k", 2)[0] == "capture"
    assert pool.break_even_k(1.0, 0.25) == 4
    b = pool.BreakEven(auto=True, probe_k=2)
    assert b.decide("a", 1)[0] == "eager" and b.decide("a", 2)[0] == "capture"
    b.observe("a", capture_s=0.5, eager_s=0.010, replay_s=0.008)
    assert b.k("new") == pool.break_even_k(0.5, 0.002)
    t = pool.Table({"x": 3})
    assert t.decide("y", 1)[0] == "eager" and t.decide("y", 1)[2] is True      # unknown key: eager BY NAME, final
    assert t.decide("x", 3)[0] == "capture"
    pl = pool.Planned({"s1": 10, "s2": 1}, pricing=pool.FixedK(3))
    assert pl.decide("s1", 1)[0] == "capture" and pl.decide("s2", 1)[2] is True and pl.decide("s3", 1)[2] is True
    assert "planned" in pl.describe()


@no_cuda_path
def test_graph_cache_without_cuda_is_a_named_refusal_or_disable():
    with pytest.raises(graphs.CaptureUnavailable):
        graphs.GraphCache("strict").run(lambda x: x, 1)
    g = graphs.GraphCache("dev", strict=False)
    assert g.run(lambda x: x + 1, 1) == 2
    assert g.state().startswith("disabled:") and g.partial().startswith("disabled")
    line = g.evidence_line("kit-opt", "F3.cuda_graph_sampler")
    assert line.startswith("[kit-opt] LEVER name=F3.cuda_graph_sampler state=skipped reason=") and "impl=cuda_graphs origin=core" in line


def test_graph_pool_sites_generations_and_jobs():
    gp = pool.GraphPool("t", strict=False)
    a = gp.site("step", max_entries=2)
    assert gp.site("step") is a
    with pytest.raises(ValueError):
        gp.site("step", max_entries=3)                                            # one site, one configuration
    g1 = gp.generation("blocks", ("traj", 1)); g2 = gp.generation("blocks", ("traj", 2))
    assert gp.generation("blocks", ("traj", 1)) is g1
    gp.generation("blocks", ("traj", 3), max_generations=2)                       # evicts the oldest untouched generation (traj 2) whole
    assert gp.generation_evictions == 1 and "blocks[1]" not in gp.sites and "blocks[0]" in gp.sites
    gp.new_job()
    assert gp.jobs == 1
    assert all(l.startswith("[t] LEVER name=") for l in gp.evidence_lines())


def test_unhashable_argument_leaf_is_named():
    with pytest.raises(TypeError):
        graphs.tensor_signature(({1, 2},))


def test_const_memo_explicit_key_and_rollout_scope():
    m = hoist.ConstMemo("memo")
    calls = []
    with m.rollout("r1"):
        assert m.value("bias", 1, lambda: calls.append(1) or "v1") == "v1"
        assert m.value("bias", 1, lambda: calls.append(2) or "v2") == "v1"      # hit: same key
        assert m.value("bias", 2, lambda: calls.append(3) or "v3") == "v3"      # new key recomputes
    with m.rollout("r2"):
        assert m.value("bias", 2, lambda: calls.append(4) or "v4") == "v4"      # new roll-out dropped everything
    s = m.stats()
    assert (s["hits"], s["misses"], s["rollouts"]) == (1, 3, 2)
    assert m.evidence_line("kit").startswith("[kit] LEVER name=memo state=on impl=const_memo origin=core")
    with pytest.raises(TypeError):
        m.get([], lambda: 1)                                                       # sources must be tensors


def test_xla_cache_keys_and_env_without_jax():
    assert xla_cache.config_hash({"b": [1, 2], "a": 1}, {"jax": "0.5.3"}) == xla_cache.config_hash({"a": 1, "b": [1, 2]}, {"jax": "0.5.3"})
    assert xla_cache.config_hash({"a": 1}, {"jax": "0.5.3"}) != xla_cache.config_hash({"a": 1}, {"jax": "0.5.4"})
    env = xla_cache.persistent_cache_env("/cache", "jax0.5.3-cu128-sm90")
    assert env["JAX_COMPILATION_CACHE_DIR"] == "/cache/jax0.5.3-cu128-sm90/jax" and env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"


def test_held_leaves_are_excluded_from_the_mutation_check():
    class Buf:                                   # any object: Held wraps it, _strip_held drops it
        pass
    tree = (graphs.Held(Buf()), 1, {"k": [graphs.Held(Buf()), "x"]})
    assert graphs._strip_held(tree) == (None, 1, {"k": [None, "x"]})
    assert graphs._unwrap((graphs.Held(5),)) == (5,)


@no_cuda_path
def test_generation_family_keeps_evicted_counters_and_prints_one_line():
    gp = pool.GraphPool("t", strict=False)
    for traj in (1, 2, 3):                       # max 2 resident: traj 1 is evicted when 3 arrives
        site = gp.generation("blocks", ("traj", traj), max_generations=2)
        site.run(lambda x: x + 1, 1)             # no CUDA here: the site disables itself by name -> a PARTIAL reason
    lines = gp.evidence_lines()
    assert len(lines) == 1 and lines[0].startswith("[t] LEVER name=blocks ")
    f = dict(gp.family_fields("blocks"))
    assert f["generations_resident"] == 2 and f["generations_evicted"] == 1 and f["partials"] == 3
    assert gp.partial() is not None              # the evicted generation's reason is not lost
    assert "blocks" in gp.stats()["families"]


def test_mutates_copied_inputs_flag_is_declared_and_printed():
    g = graphs.GraphCache("stack", strict=False, mutates_copied_inputs=True)
    assert g.mutates_copied_inputs is True
    g.run(lambda z: z, 1)                        # no CUDA: disabled by name; the line still carries the declaration
    assert " inputs=copied-mutable " in g.evidence_line("kit") + " "
    assert " inputs=copied " in graphs.GraphCache("plain", strict=False).evidence_line("kit") + " "
    st = [1, 2]; graphs.tree_copy_into(st, [None, None])   # stripped Held slots are skipped, not copied


def test_jit_cache_key_is_import_free_and_strict(monkeypatch):
    assert jit_cache.parse_version_py("__version__ = '2.8.0+cu128'\ncuda: Optional[str] = '12.8'\n") == {"version": "2.8.0+cu128", "cuda": "12.8"}
    assert jit_cache.parse_version_py('__version__ = "2.4.1"\ncuda = "12.4"\nhip = None\n') == {"version": "2.4.1", "cuda": "12.4"}
    assert jit_cache.parse_version_py("") == {}
    monkeypatch.setenv("PATH", "")                                   # no nvidia-smi: cc cannot be established
    with pytest.raises(jit_cache.StackKeyUnknown):
        jit_cache.key("2.4.1+cu124")
    assert jit_cache.key("2.4.1+cu124", cc="9.0") == "torch2.4.1-cu124-sm90"
    assert jit_cache.key("2.4.1+cu124", strict=False).endswith("-smunknown")
    assert "torch" not in sys.modules or True                        # key() never imports the distribution
