"""CPU test: the class row a card is served. A card named in the class tables (H100 / H200 / A100 / L40S / B200) resolves by its name; any other card is served the DEFAULT row of its compute capability (fastdefault.CC_DEFAULT_CLASS), said in the name field; a cc with no
default row stays 'unknown'. Run: python3 -m pytest opt/kit_ho/tf/test_class_by_cc_cpu.py -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrombpnet_fastkit import fastdefault as fd


def _detect_with(name, cc, monkeypatch):
    monkeypatch.delenv("CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE", raising=False)
    monkeypatch.setattr(fd, "nvsmi_query", lambda *a, **k: {"name": name, "cc": cc})
    return fd.detect_gpu()


def test_named_cards_resolve_by_name_unchanged(monkeypatch):
    for name, cc, cls in (("NVIDIA H100 80GB HBM3", "9.0", "H100"), ("NVIDIA H200", "9.0", "H200"), ("NVIDIA A100-SXM4-80GB", "8.0", "A100"),
                          ("NVIDIA A100 80GB PCIe", "8.0", "A100"), ("NVIDIA L40S", "8.9", "L40S"), ("NVIDIA B200", "10.0", "B200")):
        got = _detect_with(name, cc, monkeypatch)
        assert got == (cls, name, cc), got          # the name field untouched for a named card


def test_unnamed_card_is_served_its_cc_default_row(monkeypatch):
    for name, cc, cls in (("NVIDIA H800", "9.0", "H100"), ("NVIDIA H800 PCIe", "9.0", "H100"), ("NVIDIA A800-SXM4-80GB", "8.0", "A100"), ("NVIDIA A30", "8.0", "A100")):
        got_cls, got_name, got_cc = _detect_with(name, cc, monkeypatch)
        assert got_cls == cls and got_cc == cc, (got_cls, got_cc)
        assert got_name.startswith(name) and f"class {cls} by cc {cc}" in got_name, got_name   # the basis is SAID, never silent


def test_a_cc_without_a_default_row_stays_unknown(monkeypatch):
    assert _detect_with("Tesla T4", "7.5", monkeypatch)[0] == "unknown"
    assert set(fd.CC_DEFAULT_CLASS.values()) <= set(fd._CLASS_CC) and all(fd._CLASS_CC[v] == k for k, v in fd.CC_DEFAULT_CLASS.items())
