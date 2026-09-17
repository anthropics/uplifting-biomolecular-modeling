"""The card gate (stack.gate_records' `gpu` gate): admission is by compute capability — a class of stock/PINS.json supported_gpus whose
cc matches the running GPU — never by name or memory size; a card admitted under another name is a NOTE line, and a cc with no class is
admitted untested with a NOTE line (never refused). H100/H200 (9.0), B200 (10.0), B300 (10.3), A100 (8.0)."""
import json, os

from protenix_v1_opt import modes as M, report as R, stack

KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "v05_addon")


def test_supported_gpus_classes_and_capabilities():
    table = stack.pins()["supported_gpus"]
    assert {k: v["compute_capability"] for k, v in table.items()} == {"H100": "9.0", "H200": "9.0", "B200": "10.0", "B300": "10.3", "A100": "8.0"}
    assert [k for k, v in table.items() if v.get("pinned")] == ["H100"]                       # one pinned class; the others are admitted, not pinned


def test_supported_class_is_by_capability_with_names_as_labels():
    sc = stack.supported_class
    assert sc({"name": "NVIDIA H100 80GB HBM3", "sm": "9.0"}) == "H100" and sc({"name": "NVIDIA H200", "sm": "9.0"}) == "H200"
    assert sc({"name": "NVIDIA A100-SXM4-80GB", "sm": "8.0"}) == "A100" and sc({"name": "NVIDIA A100-PCIE-40GB", "sm": "8.0"}) == "A100"
    assert sc({"name": "NVIDIA A800-SXM4-80GB", "sm": "8.0"}) == "A100"                       # another name, same capability: admitted as the class
    assert sc({"name": "NVIDIA H800", "sm": "9.0"}) == "H100"                                  # first 9.0 class in table order
    assert sc({"name": "NVIDIA A10G", "sm": "8.6"}) is None and sc({"name": "NVIDIA L4", "sm": "8.9"}) is None and sc(None) is None and sc({"error": "x"}) is None


def _gpu_gate(monkeypatch, capsys, gpu):
    monkeypatch.setattr(stack, "torch_gpu_info", lambda: gpu)
    gates = {g.name: g for g in stack.gate_records(M.resolve("fast", KIT), det=False)}
    return gates["gpu"], capsys.readouterr().err


def test_gate_admits_a100_by_capability_without_a_note(monkeypatch, capsys):
    g, err = _gpu_gate(monkeypatch, capsys, {"name": "NVIDIA A100-SXM4-80GB", "sm": "8.0", "count": 1})
    assert g.ok and g.reason is None and g.details["class"] == "A100" and "NOTE" not in err


def test_gate_admits_another_name_of_the_capability_with_a_note(monkeypatch, capsys):
    g, err = _gpu_gate(monkeypatch, capsys, {"name": "NVIDIA A800-SXM4-80GB", "sm": "8.0", "count": 1})
    assert g.ok and g.details["class"] == "A100"
    assert f"{R.PREFIX} NOTE GPU NVIDIA A800-SXM4-80GB (sm 8.0) admitted by compute capability as class A100" in err.splitlines()


def test_gate_admits_an_unlisted_capability_untested_and_names_it(monkeypatch, capsys):
    g, err = _gpu_gate(monkeypatch, capsys, {"name": "NVIDIA L4", "sm": "8.9", "count": 1})
    assert g.ok and g.reason is None and g.details["class"] is None and g.details["untested"] is True
    assert (f"{R.PREFIX} NOTE GPU NVIDIA L4 (sm 8.9) is not a tested class ['A100', 'B200', 'B300', 'H100', 'H200'] (compute capabilities ['10.0', '10.3', '8.0', '9.0']): "
            "admitted untested — a lever without a kernel cell for this card takes the stock statement by name") in err.splitlines()


def test_h100_admission_prints_nothing_new(monkeypatch, capsys):
    g, err = _gpu_gate(monkeypatch, capsys, {"name": "NVIDIA H100 80GB HBM3", "sm": "9.0", "count": 1})
    assert g.ok and g.details["class"] == "H100" and "NOTE GPU" not in err
