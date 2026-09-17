"""opt_core.precision.policy reader when torch (>= 2.9) refuses the allow_tf32-style TF32 getters because the per-backend ``fp32_precision``
setting was used in the process: the values come from ``fp32_precision``; the setters still set (no real torch needed)."""
from __future__ import annotations

import types

from opt_core.precision import policy


class _Refusing:
    def __init__(self, fp32_precision, **attrs):
        object.__setattr__(self, "fp32_precision", fp32_precision)
        for k, v in attrs.items():
            object.__setattr__(self, k, v)

    def __getattribute__(self, name):
        if name == "allow_tf32":
            raise RuntimeError("mixed allow_tf32 and fp32_precision APIs")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            object.__setattr__(self, "fp32_precision", "tf32" if value else "ieee")
        else:
            object.__setattr__(self, name, value)


def _torch(mm="tf32", conv="ieee"):
    t = types.ModuleType("torch")

    def refuse():
        raise RuntimeError("mixed allow_tf32 and fp32_precision APIs")
    t.get_float32_matmul_precision = refuse
    t.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=_Refusing(mm)),
                                       cudnn=_Refusing(conv, benchmark=False, deterministic=False, conv=types.SimpleNamespace(fp32_precision=conv)))
    t.are_deterministic_algorithms_enabled = lambda: False
    t.is_deterministic_algorithms_warn_only_enabled = lambda: False
    t.is_autocast_enabled = lambda device_type=None: False
    t.get_autocast_dtype = lambda device_type: "torch.float16"
    return t


def test_reader_falls_back_to_fp32_precision():
    sig = dict(policy.torch_reader(_torch("tf32", "ieee")))
    assert (sig["matmul"], sig["matmul_tf32"], sig["cudnn_tf32"]) == ("high", True, False)
    sig = dict(policy.torch_reader(_torch("ieee", "tf32")))
    assert (sig["matmul"], sig["matmul_tf32"], sig["cudnn_tf32"]) == ("highest", False, True)
    assert policy.live(_torch("tf32", "tf32")) == {"matmul": "high", "cudnn_tf32": True, "matmul_tf32": True, "cudnn_benchmark": False}


def test_restore_sets_through_the_allow_tf32_setters():
    t = _torch("tf32", "tf32")
    out = policy.restore({"matmul_tf32": False, "cudnn_tf32": False}, t)
    assert set(out["restored"]) == {"matmul_tf32", "cudnn_tf32"}
    assert t.backends.cuda.matmul.fp32_precision == "ieee" and t.backends.cudnn.fp32_precision == "ieee"
