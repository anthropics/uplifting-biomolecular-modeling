"""kits/v1_25 at 0 GPU: the exact route's numerics class. The kit is pinned to torch's own defaults (cuDNN TF32 on, matmul TF32 off,
float32 matmul precision 'highest'); inside each call the head GEMM's TF32 follows cudnn.allow_tf32 at rest — the switch stock's cuDNN conv
head obeys — and the switches are restored after the call. Any other class at rest is read, NAMED on the apply line as drift and
served the same way; nothing is refused over it (route_numerics never raises). An autocast scope is the caller's and keeps the class word's base."""
import importlib
import inspect

import pytest
import torch

v25 = importlib.import_module("engines.flashzoi.kits.v1_25")
W = importlib.import_module("engines.flashzoi.kits.v1_25._wrap")
lane = importlib.import_module("engines.flashzoi.kits.v1_25._pins")


@pytest.fixture
def torch_defaults(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield monkeypatch
    torch.set_float32_matmul_precision(prec)


def test_the_pinned_class_is_the_stock_default(torch_defaults):
    assert W.TF32_CLASS_SERVED == "torch-default"
    d = W.route_numerics("test")
    assert d["class"] == "torch-default" and d["pinned_class"] is True and d["read_at"] == "test"
    assert "pinned to the stock's torch-default TF32 class" in d["kit_decision"] and "named on the apply line as drift" in d["kit_decision"]


def test_autocast_keeps_the_base_class(torch_defaults):
    with torch.autocast("cpu"):
        d = W.route_numerics("test")
    assert d["class"].startswith("torch-default+autocast:") and d["pinned_class"] is True


def test_other_classes_are_named_as_drift_never_refused(torch_defaults):
    torch_defaults.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    d = W.route_numerics("KitRunner(model)"); assert d["class"] == "tf32-on" and d["pinned_class"] is False
    torch_defaults.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.set_float32_matmul_precision("high")
    try:
        d = W.route_numerics("forward() through the kit"); assert d["class"] == "tf32-on" and d["pinned_class"] is False and d["float32_matmul_precision"] == "high"
    finally:
        torch.set_float32_matmul_precision("highest")
    torch_defaults.setattr(torch.backends.cudnn, "allow_tf32", False)
    d = W.route_numerics("KitRunner(model)"); assert d["class"] == "tf32-off" and d["pinned_class"] is False and d["cudnn_allow_tf32"] is False


def test_apply_and_every_call_read_the_class_and_the_call_mirrors_it():
    init_src = inspect.getsource(v25.KitRunner.__init__)
    assert 'self.numerics = route_numerics("KitRunner(model)")' in init_src                 # at apply, before any patch
    assert 'self.drift = [] if self.numerics["pinned_class"] else' in init_src            # a class other than the default is named on the line
    chk_src = inspect.getsource(W._predict_tensor_checks)
    assert 'route_numerics("forward() through the kit")' in chk_src and "raise" not in chk_src.split('route_numerics("forward() through the kit")')[1]   # per call: read, never a refusal
    wsrc = inspect.getsource(W)
    body = wsrc[wsrc.index("        def predict_tensor(self, xt):"):wsrc.index("        def predict(self, x")]
    assert body.count("_predict_tensor_checks(self, xt)") == 1
    assert "torch.backends.cuda.matmul.allow_tf32 = bool(_tf[0])" in body and "torch.backends.cudnn.allow_tf32 = True" not in body   # the head GEMM's TF32 = cudnn.allow_tf32 at rest; cudnn's own switch is never written
