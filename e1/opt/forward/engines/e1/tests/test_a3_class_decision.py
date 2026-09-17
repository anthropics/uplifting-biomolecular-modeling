"""The A3 lever's class decision (kits/v1_2): a listed class keeps its table entry (h100 / h200 on, a100 / b200 / l40s off — the
class's lever set); a GPU of NO tested class takes the lever set of the class the kit's config names (MODEL_OPT_TARGET_GPU, h100 when
unset) and is NAMED as untested — an untested GPU is never a reason to drop a lever."""
import importlib


def _dec():
    return importlib.import_module("engines.e1.kits.v1_2").a3_decision


def test_tested_classes_keep_their_record(monkeypatch):
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    d = _dec()
    assert d("NVIDIA H100 80GB HBM3") == {"gpu_class": "h100", "A3": True, "cite": d("NVIDIA H100 80GB HBM3")["cite"], "untested": False}
    assert d("NVIDIA H200")["A3"] is True and d("NVIDIA H200")["untested"] is False
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe"):
        r = d(name); assert (r["gpu_class"], r["A3"], r["untested"]) == ("a100", False, False), r
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "h100")                       # the config's class never overrides a listed class's own entry
    assert d("NVIDIA A100-SXM4-80GB")["A3"] is False


def test_an_untested_class_takes_the_configs_lever_set_and_is_named(monkeypatch):
    d = _dec()
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    r = d("NVIDIA L4")
    assert r["untested"] is True and r["A3"] is True and r["cite"].startswith("untested class ") and "the h100 lever set" in r["cite"], r   # unset config -> h100's set: A3 engages, named
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "a100")
    r = d("NVIDIA L4")
    assert r["untested"] is True and r["A3"] is False and "the a100 lever set" in r["cite"], r                                             # the a100 config's set leaves A3 out
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "H200")
    assert d("Some Future GPU")["A3"] is True
