"""opt_core.mem.jax_mem through opt_core.mem.apply on CPU: the config transforms on synthetic attribute- and mapping-style configs, the
bucket policy, the XLA environment record / collision refusal, the memory-fraction lever through allocator.jax_export, the flash
feasibility refusal — and every refusal by name. No JAX."""
import ast
import dataclasses
import functools
import os
import subprocess
import sys

import pytest

from opt_core import mem
from opt_core.mem import allocator, jax_mem, registry

ISOLATED_ENV = "OPT_CORE_TEST_ISOLATED"          # names the test running in the fresh interpreter (recursion stop)


def in_fresh_interpreter_if_xla_is_up(fn):
    """The memory-fraction variables are read at XLA backend initialisation, so ``allocator.jax_export`` refuses by name once a backend is
    initialised in THIS process — which any JAX test earlier in the same session does. Such a test runs its body in a fresh interpreter then
    (this file :: this test, nothing else); in a process without an initialised backend it runs inline."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if allocator.jax_state().get("backends_initialized") and os.environ.get(ISOLATED_ENV) != fn.__name__:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{os.path.abspath(__file__)}::{fn.__name__}"],
                               env=dict(os.environ, **{ISOLATED_ENV: fn.__name__}), capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            assert r.returncode == 0, f"{fn.__name__} in a fresh interpreter:\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}"
            return None
        return fn(*args, **kwargs)
    return wrapper


class Cfg:
    """An attribute config (the dataclass-like family)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class DictCfg(dict):
    """A mapping config with attribute access (the ml_collections-like family)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def attr_config():
    return Cfg(model=Cfg(global_config=Cfg(subbatch_size=4, pair_transition_shard_spec=((2048, None), (None, 1024)))))


def dict_config():
    return DictCfg(model=DictCfg(global_config=DictCfg(subbatch_size=4)))


PINS = {"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
TABLE = (256, 512, 1024, 2048)


def make_ctx(hooks=None, settings=None, environ=None, framework="jax"):
    return mem.Ctx(prefix="ACME", tag="acme-opt", framework=framework, hooks=hooks or {}, settings=settings or {},
                   environ={} if environ is None else environ)


def refused(line, hooks=None, settings=None, environ=None, framework="jax"):
    rec = mem.apply(line, make_ctx(hooks, settings, environ, framework), base="exact", strict=False)
    assert rec.refused, "expected a refusal"
    return rec.refused[-1]


# ------------------------------------------------------------------------------------------------------ config transforms


@pytest.mark.parametrize("make", [attr_config, dict_config])
def test_set_config_path_sets_an_existing_knob_and_records_stock(make):
    cfg = make()
    rec = jax_mem.set_config_path(cfg, "model.global_config.subbatch_size", 1)
    assert rec == {"path": "model.global_config.subbatch_size", "stock_value": 4, "value": 1}
    assert jax_mem.get_config_path(cfg, "model.global_config.subbatch_size") == 1


@pytest.mark.parametrize("make", [attr_config, dict_config])
def test_config_path_absent_is_named_never_invented(make):
    cfg = make()
    with pytest.raises(jax_mem.JaxMemError, match="'model.global_config.chunk' is absent"):
        jax_mem.get_config_path(cfg, "model.global_config.chunk")
    with pytest.raises(jax_mem.JaxMemError, match="'model.nope' is absent"):
        jax_mem.set_config_path(cfg, "model.nope.x", 1)
    with pytest.raises(jax_mem.JaxMemError, match="empty config path"):
        jax_mem.get_config_path(cfg, "")
    assert not hasattr(cfg.model.global_config, "chunk") and "chunk" not in getattr(cfg.model.global_config, "__dict__", {})


def test_shape_spec_resolution_and_whole_spec_validation():
    spec = ((2048, None), (None, 1024))
    assert jax_mem.shape_spec_value(spec, 2048) is None and jax_mem.shape_spec_value(spec, 2049) == 1024
    assert jax_mem.shape_spec_value(((1536, 128), (None, 32)), 1000) == 128
    with pytest.raises(jax_mem.JaxMemError, match="no row"):
        jax_mem.shape_spec_value(((1536, 128),), 2000)
    with pytest.raises(jax_mem.JaxMemError, match="row 0 .* is not \\(max_tokens, value\\)"):
        jax_mem.shape_spec_value(((1, 2, 3),), 1)
    with pytest.raises(jax_mem.JaxMemError, match="row 1 .* is not \\(max_tokens, value\\)"):
        jax_mem.check_shape_spec(((1536, 128), (1, 2, 3)))
    with pytest.raises(jax_mem.JaxMemError, match="row 1 max_tokens 1000 is not above"):
        jax_mem.check_shape_spec(((1536, 128), (1000, 64)))
    with pytest.raises(jax_mem.JaxMemError, match="None before the last row"):
        jax_mem.check_shape_spec(((None, 128), (1536, 64)))
    with pytest.raises(jax_mem.JaxMemError, match="not a positive int or None"):
        jax_mem.check_shape_spec((("1536", 128),))
    with pytest.raises(jax_mem.JaxMemError, match="empty"):
        jax_mem.check_shape_spec(())
    assert jax_mem.parse_value("4") == 4 and jax_mem.parse_value("[[2048, null], [null, 1024]]") == ((2048, None), (None, 1024))
    assert jax_mem.parse_value("null") is None


# ------------------------------------------------------------------------------------------------------ subbatch


def test_subbatch_lever_transforms_records_and_undoes():
    cfg = attr_config()
    ctx = make_ctx(hooks={"subbatch": {"config": cfg, "path": "model.global_config.subbatch_size"}}, settings={"subbatch": {"value": 1}})
    rec = mem.apply(["subbatch"], ctx, base="exact")
    a = rec.applied[0]
    assert a.exact == "measured" == a.declared_exact and a.narrowed_by is None and a.scope == "unit"
    assert a.settings == {"path": "model.global_config.subbatch_size", "stock_value": 4, "value": 1, "kind": "count", "stock": False}
    assert a.sites == ("model.global_config.subbatch_size",) and cfg.model.global_config.subbatch_size == 1
    for item in ("item-0", "item-1"):                                       # the adapter marks the sub-batched op per item, inside its unit
        rec.unit_begin(item)
        jax_mem.mark_subbatch(ctx, detail="pair transition sub-batched")
        rec.unit_end()
    assert rec.units["item-1"].ran == ["subbatch"] and rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
    assert mem.undo(rec) == ["subbatch"] and cfg.model.global_config.subbatch_size == 4
    jax_mem.mark_subbatch(make_ctx(), detail="not applied")                 # the lever not applied: nothing recorded, no error
    r = refused(["subbatch"], {"subbatch": {"config": attr_config(), "path": "model.global_config.subbatch_size"}}, {"subbatch": {"value": 4}})
    assert r.precondition == "value" and "stock's own" in r.reason and "switches={'subbatch': False}" in r.reason    # stock's value: refused, never a narrowed label
    cfg = attr_config()
    env = {"ACME_BIG_SUBBATCH_VALUE": "7"}                                                     # never read
    rec = mem.apply(["subbatch"], make_ctx(hooks={"subbatch": {"config": cfg, "path": "model.global_config.pair_transition_shard_spec"}},
                                           settings={"subbatch": {"value": "[[null, 512]]"}}, environ=env), base="exact")   # the JSON string the kit read from its variable
    a = rec.applied[0]
    assert a.settings["kind"] == "shape_spec" and cfg.model.global_config.pair_transition_shard_spec == ((None, 512),)
    assert a.settings["stock_value"] == ((2048, None), (None, 1024)) and a.settings["stock"] is False
    assert [s for s in rec.settings if s["lever"] == "subbatch"][0]["source"] == "ctx.settings"
    dcfg = dict_config()
    mem.apply(["subbatch"], make_ctx(hooks={"subbatch": {"config": dcfg, "path": "model.global_config.subbatch_size"}}, settings={"subbatch": {"value": 2}}), base="exact")
    assert dcfg["model"]["global_config"]["subbatch_size"] == 2


def test_subbatch_refuses_by_name():
    cfg = attr_config()
    path = "model.global_config.subbatch_size"
    assert refused(["subbatch"], {"subbatch": {"path": path}}).precondition == "hooks.config"
    assert refused(["subbatch"], {"subbatch": {"config": cfg}}).precondition == "hooks.path"
    r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": path}})
    assert r.precondition == "value" and "ctx.settings['subbatch']['value']" in r.reason
    assert refused(["subbatch"], {"subbatch": {"config": cfg, "path": path}}, {"subbatch": {"value": 0}}).precondition == "value"
    r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": path}}, {"subbatch": {"value": ((1536, 128), (1, 2, 3))}})
    assert r.precondition == "value" and "malformed" in r.reason
    r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": "model.global_config.chunk"}}, {"subbatch": {"value": 1}})
    assert r.precondition == "hooks.path" and "is absent" in r.reason
    r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": path}}, {"subbatch": {"value": "four"}})
    assert r.precondition in ("settings.subbatch.value", "value") and "four" in r.reason
    assert cfg.model.global_config.subbatch_size == 4                                             # nothing moved
    r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": path}}, {"subbatch": {"value": 1}}, framework="torch")
    assert r.precondition == "framework"


def test_subbatch_refuses_a_frozen_or_locked_config_by_name():
    @dataclasses.dataclass(frozen=True)
    class Frozen:
        subbatch_size: int = 4

    class Locked(dict):
        def __setitem__(self, k, v):
            raise KeyError("locked")

    for cfg, word in ((Cfg(model=Frozen()), "FrozenInstanceError"), (DictCfg(model=Locked(subbatch_size=4)), "locked")):
        r = refused(["subbatch"], {"subbatch": {"config": cfg, "path": "model.subbatch_size"}}, {"subbatch": {"value": 1}})
        assert r.precondition == "hooks.path" and "config knob 'subbatch_size'" in r.reason and "is not settable" in r.reason and word in r.reason
        assert jax_mem.get_config_path(cfg, "model.subbatch_size") == 4                          # the probe changed nothing


# ------------------------------------------------------------------------------------------------------ bucket_policy


def test_resolve_bucket_names_the_case():
    assert jax_mem.resolve_bucket(300, TABLE) == {"tokens": 300, "bucket": 512, "padding": 212, "case": "shipped"}
    assert jax_mem.resolve_bucket(1024, TABLE)["padding"] == 0
    assert jax_mem.resolve_bucket(3000, TABLE) == {"tokens": 3000, "bucket": 3000, "padding": 0, "case": "exact"}
    assert jax_mem.resolve_bucket(3000, TABLE, 3072) == {"tokens": 3000, "bucket": 3072, "padding": 72, "case": "probe"}
    with pytest.raises(jax_mem.JaxMemError, match="not beyond the shipped table"):
        jax_mem.resolve_bucket(3000, TABLE, 2048)
    with pytest.raises(jax_mem.JaxMemError, match="below the size"):
        jax_mem.resolve_bucket(4000, TABLE, 3072)
    with pytest.raises(jax_mem.JaxMemError, match="strictly increasing"):
        jax_mem.resolve_bucket(10, (512, 256))
    with pytest.raises(jax_mem.JaxMemError, match="tokens must be"):
        jax_mem.resolve_bucket(0, TABLE)


def test_bucket_policy_is_the_probe_form_and_marks_each_item():
    r = refused(["bucket_policy"], {"bucket_policy": {"buckets": TABLE}})
    assert r.precondition == "probe_bucket" and "nothing to apply" in r.reason and "ctx.settings['bucket_policy']['probe_bucket']" in r.reason
    probe = make_ctx(hooks={"bucket_policy": {"buckets": TABLE}}, settings={"bucket_policy": {"probe_bucket": "3072"}}, environ={"ACME_BIG_BUCKET_POLICY_PROBE_BUCKET": "1"})   # env never read
    rec = mem.apply(["bucket_policy"], probe, base="exact")
    a = rec.applied[0]
    assert a.exact == "measured" == a.declared_exact and a.narrowed_by is None and a.settings == {"table": list(TABLE), "probe_bucket": 3072, "stock": False}
    assert a.notes == ["probe bucket 3072 named beyond the shipped table (last shipped 2048)"] and a.scope == "process"
    assert jax_mem.bucket_for(probe, 300)["case"] == "shipped" and jax_mem.bucket_for(probe, 2048)["case"] == "shipped"
    assert jax_mem.bucket_for(probe, 3000) == {"tokens": 3000, "bucket": 3072, "padding": 72, "case": "probe"}
    assert a.notes[-1].startswith("probe bucket 3072 at 3000 tokens (padding 72)")
    ev = [e["detail"] for e in rec.units["process"].events]                 # a process-scope lever is marked on the process unit at apply, then per item
    assert ev[0].startswith("applied (process scope)") and ev[1:] == ["tokens=300 bucket=512 case=shipped", "tokens=2048 bucket=2048 case=shipped",
                                                                      "tokens=3000 bucket=3072 case=probe"]
    rec.unit_begin("item-0"); rec.unit_end()                                # a process-scope lever: the item's unit expects nothing of it
    assert rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
    with pytest.raises(jax_mem.JaxMemError, match="below the size"):
        jax_mem.bucket_for(probe, 4000)
    with pytest.raises(jax_mem.JaxMemError, match="not applied"):
        jax_mem.bucket_for(make_ctx(), 10)
    assert refused(["bucket_policy"], {}).precondition == "hooks.buckets"
    assert refused(["bucket_policy"], {"bucket_policy": {"buckets": ()}}).precondition == "hooks.buckets"
    assert refused(["bucket_policy"], {"bucket_policy": {"buckets": (5, 5)}}, {"bucket_policy": {"probe_bucket": 9}}).precondition == "probe_bucket"
    assert refused(["bucket_policy"], {"bucket_policy": {"buckets": TABLE}}, {"bucket_policy": {"probe_bucket": 1024}}).precondition == "probe_bucket"


# ------------------------------------------------------------------------------------------------------ xla_env


def test_xla_env_records_never_overrides_and_refuses_a_collision_by_name():
    env = dict(PINS, JAX_TRACEBACK_FILTERING="off", HOME="/x")
    assert jax_mem.xla_env_record(env) == {"JAX_TRACEBACK_FILTERING": "off", **{k: PINS[k] for k in sorted(PINS)}}
    chk = jax_mem.xla_env_check(PINS, env)
    assert chk["ok"] and chk["collisions"] == {} and chk["live_differs"] == {} and chk["unset_pins"] == []
    chk = jax_mem.xla_env_check(PINS, {"XLA_FLAGS": "--other"}, requested={"XLA_CLIENT_MEM_FRACTION": "0.5"})
    assert not chk["ok"] and set(chk["collisions"]) == {"XLA_CLIENT_MEM_FRACTION"} and set(chk["live_differs"]) == {"XLA_FLAGS"}
    assert chk["unset_pins"] == ["XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"]
    ctx = make_ctx(hooks={"xla_env": {"pinned": PINS, "requested": {"XLA_FLAGS": " --XLA_GPU_ENABLE_TRITON_GEMM=FALSE "}}}, environ=env)
    rec = mem.apply(["xla_env"], ctx, base="exact")
    a = rec.applied[0]
    assert a.exact == "bitwise" and a.settings["record"] == jax_mem.xla_env_record(env) and a.settings["stock"] is True and a.notes == []
    assert a.scope == "process" and rec.units["process"].ran == ["xla_env"] and rec.exit_gate(0)["exit_code"] == 0
    assert env == dict(PINS, JAX_TRACEBACK_FILTERING="off", HOME="/x")                            # nothing written
    r = refused(["xla_env"], {"xla_env": {"pinned": PINS, "requested": {"XLA_FLAGS": "--x"}}}, environ=env)
    assert r.precondition == "xla.XLA_FLAGS" and "never overridden" in r.reason and r.details == {"pinned": PINS["XLA_FLAGS"], "requested": "--x"}
    assert refused(["xla_env"], {}).precondition == "hooks.pinned"
    assert refused(["xla_env"], {"xla_env": {"pinned": "x"}}).precondition == "hooks.pinned"
    drifted = mem.apply(["xla_env"], make_ctx(hooks={"xla_env": {"pinned": PINS}}, environ=dict(PINS, XLA_FLAGS="--x")), base="exact").applied[0]
    assert drifted.notes == ["pinned variable differs live: XLA_FLAGS=--x (pinned --xla_gpu_enable_triton_gemm=false)"]
    assert drifted.settings["live_differs"] == {"XLA_FLAGS": {"pinned": PINS["XLA_FLAGS"], "live": "--x"}}


# ------------------------------------------------------------------------------------------------------ mem_fraction


@in_fresh_interpreter_if_xla_is_up
def test_mem_fraction_exports_through_jax_export_and_records_every_write():
    env = {}
    ctx = make_ctx(hooks={"mem_fraction": {"pinned": {}}}, settings={"mem_fraction": {"fraction": 0.95, "preallocate": False}}, environ=env)
    rec = mem.apply(["mem_fraction"], ctx, base="exact")
    a = rec.applied[0]
    assert env == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"} and a.exact == "bitwise" and a.scope == "process"
    assert a.settings["exported"] == {"XLA_PYTHON_CLIENT_MEM_FRACTION": {"value": "0.95", "state": "exported"},
                                      "XLA_PYTHON_CLIENT_PREALLOCATE": {"value": "false", "state": "exported"}} and a.settings["stock"] is False
    assert [w["name"] for w in rec.allocator["writes"]] == ["XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"]
    assert all(w["lever"] == "mem_fraction" and w["via"] == "env" for w in rec.allocator["writes"])
    kept = mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {}}}, settings={"mem_fraction": {"fraction": 0.95, "preallocate": False}},
                                                environ=dict(env)), base="exact")
    assert kept.applied[0].settings["stock"] is True and kept.allocator["writes"] == []
    e2 = {}
    mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {}}}, settings={"mem_fraction": {"fraction": "0.5", "preallocate": "1"}}, environ=e2), base="exact")
    e3 = {}
    mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {}}}, settings={"mem_fraction": {"fraction": 0.5}}, environ=e3), base="exact")
    assert e3 == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"}
    e4 = {}
    mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {}}}, settings={"mem_fraction": {"fraction": 4.0, "unified_memory": True}}, environ=e4), base="exact")
    assert e4 == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"}
    drift = mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {"XLA_FLAGS": "--a"}}}, settings={"mem_fraction": {"fraction": 0.5}},
                                                 environ={"XLA_FLAGS": "--b"}), base="exact").applied[0]
    assert drift.notes == ["pinned variable differs live: XLA_FLAGS=--b (pinned --a)"]
    assert jax_mem._fraction_str(4.0) == "4.0" and jax_mem._fraction_str(0.9) == "0.9" and jax_mem._fraction_str(0.75) == "0.75"


@in_fresh_interpreter_if_xla_is_up
def test_mem_fraction_refuses_by_name():
    assert refused(["mem_fraction"], {}).precondition == "hooks.pinned"
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}})
    assert r.precondition == "fraction" and "ctx.settings['mem_fraction']['fraction']" in r.reason
    assert refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 0}}).precondition == "fraction"
    assert refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 4.0}}).precondition == "unified_memory"
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 0.5, "preallocate": "maybe"}})
    assert r.precondition in ("settings.mem_fraction.preallocate", "preallocate") and "maybe" in r.reason
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75"}}}, {"mem_fraction": {"fraction": 0.9}})
    assert r.precondition == "pins.XLA_PYTHON_CLIENT_MEM_FRACTION" and r.details == {"pinned": "0.75", "wanted": "0.9"}
    live = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 0.9}}, environ=live)
    assert r.precondition == "xla.XLA_PYTHON_CLIENT_MEM_FRACTION" and live == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}     # jax_export's refusal; nothing written


def test_the_xla_table_is_peaks_and_the_record_extends_it():
    """ONE XLA client table (opt_core.mem.peak): jax_mem reads names / prefixes / MEM_FRACTION precedence from it and adds only the
    device / communication prefixes of its wider record."""
    from opt_core.mem import peak

    assert jax_mem.XLA_PREFIXES[:len(peak.XLA_PREFIXES)] == tuple(peak.XLA_PREFIXES) and set(jax_mem.RECORD_EXTRA_PREFIXES) == {"TF_", "NCCL_", "CUDA_VISIBLE_DEVICES", "TPU_"}
    assert jax_mem.MEM_FRACTION_VAR in peak.MEM_FRACTION_PRECEDENCE and jax_mem.MEM_FRACTION_VAR in peak.XLA_NAMES and jax_mem.PREALLOCATE_VAR in peak.XLA_NAMES
    env = {"XLA_FLAGS": "--a", "JAX_PLATFORMS": "cuda", "TF_FORCE_UNIFIED_MEMORY": "1", "NCCL_DEBUG": "warn", "CUDA_VISIBLE_DEVICES": "0,1", "HOME": "/x"}
    rec = jax_mem.xla_env_record(env)
    assert rec == {k: env[k] for k in sorted(env) if k != "HOME"} and set(peak.xla_env(env)) <= set(rec) and list(rec) == sorted(rec)


@in_fresh_interpreter_if_xla_is_up
def test_mem_fraction_lives_under_one_name_and_both_names_refuse():
    """The fraction is compared / written under the MEM_FRACTION name already present (client read order), else the pinned one, else
    XLA_PYTHON_CLIENT_MEM_FRACTION — never the second name beside the first; both present refuses by name (CUDA plugin conflict)."""
    new, old = "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION"
    assert jax_mem.mem_fraction_name({}, {}) == old and jax_mem.mem_fraction_name({new: "0.9"}, {}) == new and jax_mem.mem_fraction_name({old: "0.9"}, {new: "0.9"}) == old
    assert jax_mem.mem_fraction_name({}, {new: "0.95"}) == new and jax_mem.mem_fraction_conflict({new: "1", old: "1"}) == {new: "1", old: "1"} and jax_mem.mem_fraction_conflict({new: "1"}) is None
    # the newer name present and equal: kept under that name, nothing exported beside it
    live = {new: "0.95"}
    rec = mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {new: "0.95"}}}, settings={"mem_fraction": {"fraction": 0.95}}, environ=live), base="exact")
    assert live == {new: "0.95"} and rec.applied[0].settings["exported"] == {new: {"value": "0.95", "state": "kept"}} and rec.applied[0].settings["stock"] is True
    # the newer name present and different: the primitive's refusal under THAT name; nothing written
    live = {new: "0.95"}
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 0.5}}, environ=live)
    assert r.precondition == "xla." + new and live == {new: "0.95"}
    # the newer name pinned (absent live) and different: pins.<name>; pinned and equal, absent live: exported under the pinned name
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {new: "0.95"}}}, {"mem_fraction": {"fraction": 0.5}})
    assert r.precondition == "pins." + new and r.details == {"pinned": "0.95", "wanted": "0.5"}
    e = {}
    mem.apply(["mem_fraction"], make_ctx(hooks={"mem_fraction": {"pinned": {new: "0.95"}}}, settings={"mem_fraction": {"fraction": 0.95}}, environ=e), base="exact")
    assert e == {new: "0.95"}
    # both names present: refuses by name before any write
    live = {new: "0.95", old: "0.95"}
    r = refused(["mem_fraction"], {"mem_fraction": {"pinned": {}}}, {"mem_fraction": {"fraction": 0.95}}, environ=live)
    assert r.precondition == "xla.MEM_FRACTION" and r.details["present"] == {new: "0.95", old: "0.95"} and live == {new: "0.95", old: "0.95"}


# ------------------------------------------------------------------------------------------------------ flash feasibility


@pytest.mark.parametrize("hooks,fused", [({"attention_family": "pairformer", "flash_implementation": "triton"}, True),
                                         ({"attention_family": "pairformer", "flash_implementation": "xla"}, False),
                                         ({"attention_family": "evoformer"}, None), ({}, None), ({"flash_implementation": "cudnn"}, True)])
def test_flash_triattn_always_refuses_by_design_with_the_verdict(hooks, fused):
    r = refused(["flash_triattn_jax"], {"flash_triattn_jax": hooks})
    assert r.precondition == "kernel" and "feasibility only" in r.reason and r.details["applies"] is False and r.details["stock_fused"] is fused
    assert r.details["family"] == (hooks.get("attention_family") or "unknown")
    lever = registry.get("flash_triattn_jax")
    with pytest.raises(registry.RefusalError, match="feasibility only"):
        lever.apply(make_ctx(hooks={"flash_triattn_jax": hooks}))


# ------------------------------------------------------------------------------------------------------ the registry


def test_levers_are_registered_by_name_for_jax_only():
    assert set(registry.names(family="jax")) >= {"subbatch", "bucket_policy", "xla_env", "mem_fraction", "flash_triattn_jax"}
    assert set(jax_mem.HOOKS) == {"subbatch", "bucket_policy", "xla_env", "mem_fraction", "flash_triattn_jax"}
    for name in jax_mem.HOOKS:
        lv = registry.get(name)
        assert lv.module == "opt_core.mem.jax_mem" and lv.frameworks == ("jax",) and lv.exact_reason
    assert registry.get("mem_fraction").settings == ("fraction", "preallocate", "unified_memory")
    assert {n: registry.get(n).exact for n in jax_mem.HOOKS} == {"subbatch": "measured", "bucket_policy": "measured", "xla_env": "bitwise",
                                                                "mem_fraction": "bitwise", "flash_triattn_jax": "measured"}
    assert {n: registry.get(n).scope for n in jax_mem.HOOKS} == {"subbatch": "unit", "bucket_policy": "process", "xla_env": "process",
                                                                "mem_fraction": "process", "flash_triattn_jax": "unit"}
    assert registry.setting_ref("subbatch", "value") == "ctx.settings['subbatch']['value']"
    assert "jax_mem" in registry.discover()["loaded"]


def test_module_imports_no_jax():
    tree = ast.parse(open(jax_mem.__file__, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(getattr(node, "func", None), "attr", "") == "import_module":
            names.add("<dynamic import>")
    assert "jax" not in names and "jaxlib" not in names and "<dynamic import>" not in names
