"""The `loader_workers` cell (exact class): the predict DataLoader's worker count is capped at the query set's item count — max(1, min(items,
configured)), never 0, never above the configured count; training / validation loaders untouched; `_CAP=none` keeps the engine's count on
every loader, by name. Stub engine: a DataModule class with the engine's method names (no torch, no openfold3)."""
import enum
import sys
import types

import pytest

from openfold3_opt import of3_loader as L
from openfold3_opt.cells import loader_workers as cell


class _Mode(enum.Enum):
    train = "train"
    predict = "predict"


def _engine(monkeypatch, name="of3opt_test_data_module"):
    mod = types.ModuleType(name)

    class DataModule:
        def __init__(self, workers, items):
            self.num_workers = workers
            self.datasets_by_mode = {_Mode.predict: list(range(items)), _Mode.train: list(range(50))}

        def generate_dataloader(self, mode, sampler=None):
            return {"mode": mode, "num_workers": self.num_workers, "sampler": sampler}

        def predict_dataloader(self):
            return self.generate_dataloader(_Mode.predict)

    mod.DataModule = DataModule
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def bound(monkeypatch):
    prev = {k: getattr(L, k) for k in L.CONFIGURABLE}
    saved = dict(L.STATE)
    L.STATE.update(installed=False, state="off", reason="", cap=None, loaders=0, configured=[], used=[], items=[])
    mod = _engine(monkeypatch)
    L.configure(M_DATA=mod.__name__)
    yield mod
    try:
        L.uninstall()
    except Exception:
        pass
    L.configure(**prev)
    L.STATE.clear(); L.STATE.update(saved)


def test_plan_is_min_items_configured_floored_at_one():
    assert [L.plan(i, 10) for i in (0, 1, 3, 10, 12)] == [1, 1, 3, 10, 10]
    assert L.plan(5, 0) == 0 and L.plan(1, 1) == 1                                            # 0 configured: in-process featurisation, as configured


def test_the_cell_binds_this_kits_switch_names():
    assert cell.ENV == "OPENFOLD3_OPT_LOADER_WORKERS" and cell.ENV_CAP == "OPENFOLD3_OPT_LOADER_WORKERS_CAP" and cell.STATE is L.STATE
    assert L.M_DATA == "openfold3.core.data.framework.data_module" and L.PREFIX == "[openfold3-opt/loader_workers]"


def test_predict_loader_capped_train_loader_untouched_count_restored(bound):
    st = L.install({"OPENFOLD3_OPT_LOADER_WORKERS": "1"})
    assert st["state"] == "on" and st["cap"] == "items"
    dm = bound.DataModule(workers=10, items=1)
    assert dm.predict_dataloader()["num_workers"] == 1 and dm.num_workers == 10               # capped for the loader, the module's setting restored
    assert dm.generate_dataloader(_Mode.train)["num_workers"] == 10                          # training: the engine's count
    dm3 = bound.DataModule(workers=10, items=3)
    assert dm3.predict_dataloader()["num_workers"] == 3
    dm12 = bound.DataModule(workers=10, items=12)
    assert dm12.predict_dataloader()["num_workers"] == 10
    line = L.census_line()
    assert " LEVER name=loader_workers state=on cap=items loaders=3 configured=10,10,10 used=1,3,10 items=1,3,12" in line, line
    assert L.install({"OPENFOLD3_OPT_LOADER_WORKERS": "1"}) is st                              # idempotent
    assert getattr(bound.DataModule.generate_dataloader, "_of3opt_loader_workers", False)


def test_cap_none_keeps_the_engines_count_by_name(bound):
    st = L.install({"OPENFOLD3_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OPT_LOADER_WORKERS_CAP": "none"})
    assert st["state"] == "on" and st["reason"] == "cap_none"
    dm = bound.DataModule(workers=10, items=1)
    assert dm.predict_dataloader()["num_workers"] == 10
    assert "reason=cap_none cap=none loaders=1 configured=10 used=10 items=1" in L.census_line()


def test_unrequested_installs_nothing_and_bad_words_raise(bound):
    assert L.install({}) is L.STATE and L.STATE["installed"] is False
    assert not getattr(bound.DataModule.generate_dataloader, "_of3opt_loader_workers", False)
    with pytest.raises(ValueError):
        L.requested({"OPENFOLD3_OPT_LOADER_WORKERS": "yes"})
    with pytest.raises(ValueError):
        L.cap_word({"OPENFOLD3_OPT_LOADER_WORKERS_CAP": "0"})


def test_an_engine_without_the_data_module_is_refused_by_name(bound, monkeypatch):
    L.configure(M_DATA="of3opt_test_no_such_engine_module")
    st = L.install({"OPENFOLD3_OPT_LOADER_WORKERS": "1"})
    assert st["state"] == "refused" and st["reason"].startswith("no_datamodule:")
