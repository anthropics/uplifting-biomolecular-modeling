"""runner_hooks: the switches' parsing, the runner patch (installed now or at import) for the chunk and the guard lift, the failure record, the record in reconcile."""
import os
import sys
import tempfile
import textwrap
import types

import pytest

from protenix_opt import runner_hooks as dc
from protenix_opt import stack
from protenix_opt.tests import _stock_stub


class _Cfg:
    def __init__(self):
        self.infer_setting = types.SimpleNamespace(sample_diffusion_chunk_size=5)
        self.skip_amp = types.SimpleNamespace(confidence_head=False, sample_diffusion=True)   # the runner's configs shape (configs_base: skip_amp)
        self.model_name = "protenix-v2"


def _stub_module():
    m = types.ModuleType(dc.TARGET)

    def update_inference_configs(configs, n_token):
        configs.seen_n_token = n_token
        return configs
    m.update_inference_configs = update_inference_configs
    return m


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(dc, "_STATE", {"guard_lift": False, "patched": False, "guard_lifted_items": 0, "sampler_fp32_items": 0, "max_n_token": 0, "finder": None, "failures": [], "handler": None, "listeners": []})
    saved = list(sys.meta_path); had = sys.modules.pop(dc.TARGET, None)
    yield
    sys.meta_path[:] = saved
    sys.modules.pop(dc.TARGET, None)
    if had is not None: sys.modules[dc.TARGET] = had


def test_install_patches_an_imported_runner_and_counts_the_items():
    sys.modules[dc.TARGET] = m = _stub_module()
    assert dc.install(guard_lift=True) == ["GUARD_LIFT:1(patched)"]
    cfg = m.update_inference_configs(_Cfg(), 1956)
    assert cfg.infer_setting.sample_diffusion_chunk_size == 5 and cfg.seen_n_token == 1956   # the stock's own adjustments run; the stock chunking untouched
    m.update_inference_configs(_Cfg(), 400)
    st = dc.state(); assert (st["patched"], st["guard_lift"], st["max_n_token"]) == (True, True, 1956)
    dc.install(guard_lift=True)                                                              # idempotent: one wrapper
    assert m.update_inference_configs.__wrapped__.__name__ == "update_inference_configs" and not hasattr(m.update_inference_configs.__wrapped__, "__wrapped__")


def test_install_before_import_patches_at_the_runners_import(monkeypatch):
    assert dc.install(guard_lift=True) == ["GUARD_LIFT:1(armed)"] and isinstance(sys.meta_path[0], dc._PostImportFinder)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "runner"))
        open(os.path.join(d, "runner", "__init__.py"), "w").write("")
        open(os.path.join(d, "runner", "inference.py"), "w").write(textwrap.dedent("""
            def update_inference_configs(configs, n_token):
                configs.seen_n_token = n_token
                return configs
        """))
        monkeypatch.syspath_prepend(d); sys.modules.pop("runner", None)
        import importlib
        m = importlib.import_module(dc.TARGET)
        assert not any(isinstance(f, dc._PostImportFinder) for f in sys.meta_path)          # the finder leaves at the import
        cfg = m.update_inference_configs(_Cfg(), 100)
        assert cfg.infer_setting.sample_diffusion_chunk_size == 5 and dc.state()["patched"] is True and dc.state()["guard_lift"] is True
    sys.modules.pop("runner", None)


def _guarded_stub():
    m = types.ModuleType(dc.TARGET)

    def update_inference_configs(configs, n_token):                                          # the stock form: the assertion, then the > 2560 settings
        if n_token > 2560 and configs.model_name in ["protenix-v2"]:
            raise AssertionError("protenix-v2 model does not support n_token > 2560. It might cause OOM.")
        configs.big = n_token > 2560
        return configs
    m.update_inference_configs = update_inference_configs
    return m


def test_guard_lift_parses_and_lifts_the_assertion_for_the_call_only():
    assert dc.guard_lift_from_env({}) is False and dc.guard_lift_from_env({dc.ENV_GUARD: "0"}) is False and dc.guard_lift_from_env({dc.ENV_GUARD: "1"}) is True
    with pytest.raises(ValueError):
        dc.guard_lift_from_env({dc.ENV_GUARD: "yes"})
    sys.modules[dc.TARGET] = m = _guarded_stub()
    with pytest.raises(AssertionError):
        m.update_inference_configs(_Cfg(), 2956)
    assert dc.install(guard_lift=True) == ["GUARD_LIFT:1(patched)"]
    cfg = m.update_inference_configs(_Cfg(), 2956)
    assert cfg.big is True and cfg.model_name == "protenix-v2" and cfg.infer_setting.sample_diffusion_chunk_size == 5   # restored; no chunk asked
    m.update_inference_configs(_Cfg(), 400)
    st = dc.state(); assert (st["guard_lift"], st["guard_lifted_items"], st["max_n_token"]) == (True, 1, 2956)
    c2 = _Cfg(); c2.model_name = "protenix-base"                                            # another model: the stock function untouched
    assert m.update_inference_configs(c2, 2956).big is True


def _policy_stub():
    """A runner.inference whose update_inference_configs IS the pinned stock function (compiled from stock/src by _stock_stub)."""
    m = types.ModuleType(dc.TARGET)
    m.update_inference_configs = _stock_stub.pinned_update_inference_configs()
    return m


def _amp_cfg(model_name="protenix-v2"):
    c = _Cfg(); c.model_name = model_name
    c.skip_amp.confidence_head = c.skip_amp.sample_diffusion = None                          # set by the runner's policy alone
    return c


@pytest.mark.parametrize("n_token", [2561, 3840, 3841, 4032, 7000, 31000])
def test_guard_lift_keeps_the_diffusion_sampler_fp32_at_every_size_above_the_guard(n_token):
    """Lifted, an item of any size runs with protenix-v2's own precision settings (= the stock settings at every size stock accepts):
    confidence head under autocast, diffusion sampler fp32 — where the pinned runner's large-N policy says bf16 the lift puts the
    setting back, counted per item; another checkpoint keeps the stock policy untouched."""
    sys.modules[dc.TARGET] = m = _policy_stub()
    with pytest.raises(AssertionError):
        m.update_inference_configs(_amp_cfg(), n_token)                                     # the pinned function refuses protenix-v2 above the guard
    ref = m.update_inference_configs(_amp_cfg(), 400)                                       # what stock protenix-v2 runs at every size it accepts
    assert (ref.skip_amp.confidence_head, ref.skip_amp.sample_diffusion) == (False, True)
    stock_large_n_bf16 = m.update_inference_configs(_amp_cfg("protenix-base"), n_token).skip_amp.sample_diffusion is False   # the runner's policy at this size
    assert dc.install(guard_lift=True) == ["GUARD_LIFT:1(patched)"]
    cfg = m.update_inference_configs(_amp_cfg(), n_token)
    assert (cfg.skip_amp.confidence_head, cfg.skip_amp.sample_diffusion) == (False, True) and cfg.model_name == "protenix-v2"
    st = dc.state()
    assert (st["guard_lifted_items"], st["sampler_fp32_items"]) == (1, int(stock_large_n_bf16))
    other = m.update_inference_configs(_amp_cfg("protenix-base"), n_token)                 # another checkpoint: the stock policy untouched by the lift
    assert other.skip_amp.sample_diffusion is (not stock_large_n_bf16) and dc.state()["guard_lifted_items"] == 1


def test_failure_record_names_the_runners_item_failures_by_class():
    import logging
    dc.install_failure_record(); dc.install_failure_record()                                 # idempotent
    log = logging.getLogger(dc.TARGET)
    log.error("[Rank 0] 8G5T_2000 failed: CUDA out of memory. Tried to allocate 12.00 GiB")
    log.error("[Rank 0] 1L1L_3000 failed: RuntimeError: shape mismatch")
    log.info("[Rank 0] other line"); log.error("no item here")
    f = dc.failures()
    assert [(x["item"], x["reason_class"]) for x in f] == [("8G5T_2000", "OOM"), ("1L1L_3000", "other")] and "12.00 GiB" in f[0]["text"]
    assert len([h for h in log.handlers if isinstance(h, dc._FailureHandler)]) == 1
    log.removeHandler(dc._STATE["handler"])




def test_reconcile_records_the_lift_and_names_an_unapplied_one():
    dc.install(guard_lift=True)                                                              # armed, runner never imported
    r = stack.reconcile({"levers_applied": ["guard_lift", "nomask"], "levers_fallback": [], "fallback_reasons": {}}, {})
    assert r["runner_hooks"]["patched"] is False and r["runner_hooks"]["guard_lift"] is True
    assert "guard_lift" not in r["levers_applied"] and "guard_lift" in r["levers_fallback"] and "never imported" in r["fallback_reasons"]["guard_lift"]
    sys.modules[dc.TARGET] = _stub_module(); dc.install(guard_lift=True)
    r2 = stack.reconcile({"levers_applied": ["guard_lift"], "levers_fallback": [], "fallback_reasons": {}}, {})
    assert r2["runner_hooks"]["patched"] is True and "guard_lift" in r2["levers_applied"]


def test_guard_lift_rides_the_big_row_and_is_refused_elsewhere():
    """The lift is part of big's row (PTX_GUARD_LIFT=1 exported by the mode); set under a mode whose row does not list it, activation
    refuses by name."""
    from protenix_opt import modes, registry
    assert "guard_lift" in modes.MODES["big"] and "guard_lift" not in modes.MODES["exact"]
    assert stack._own_levers("big", {"PTX_GUARD_LIFT": "1"}) is True and stack._own_levers("exact", {}) is False
    with pytest.raises(stack.ActivationError):
        stack._own_levers("fast", {"PTX_GUARD_LIFT": "1"})
