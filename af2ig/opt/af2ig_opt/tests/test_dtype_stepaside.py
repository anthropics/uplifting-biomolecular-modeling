"""K10 (0.6.0): the fused float32 blocks (L10 TriangleAttention, L11 TriangleMultiplication) take float32 activations only — any other dtype stays on the
stock body, counted under the declared fallback reason `dtype` (never handed to the f32 kernel: a Pallas LoweringError killed the design before)."""
import unittest
from unittest import mock


class _Cfg(dict):
    __getattr__ = dict.__getitem__


class _Act:
    def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype


class _DT:
    def __init__(self, name): self.name = name
    def __str__(self): return self.name


class TestDtypeStepAside(unittest.TestCase):
    def test_trimul_and_triattn_route_non_f32_to_the_stock_body_by_name(self):
        from af2ig_opt import fused_trimul, triattn
        served = mock.Mock(); served.served_reason = mock.Mock(return_value=None)
        with mock.patch.object(fused_trimul, "_serve", return_value=served), mock.patch.object(triattn, "_serve", return_value=served):
            cfg_m = _Cfg(equation="ikc,jkc->ijc", num_intermediate_channel=128)
            self.assertEqual(fused_trimul.reason_for(cfg_m, _Act((256, 256, 128), _DT("bfloat16")), _Act((256, 256), _DT("bfloat16"))), "dtype")
            self.assertIsNone(fused_trimul.reason_for(cfg_m, _Act((256, 256, 128), _DT("float32")), _Act((256, 256), _DT("float32"))))
            cfg_a = _Cfg(gating=True, num_head=4, orientation="per_row"); cfg_a.get = dict.get.__get__(cfg_a)
            self.assertEqual(triattn.reason_for(cfg_a, _Act((256, 256, 128), _DT("float16")), _Act((256, 256), _DT("float16"))), "dtype")
            self.assertIsNone(triattn.reason_for(cfg_a, _Act((256, 256, 128), _DT("float32")), _Act((256, 256), _DT("float32"))))
        self.assertEqual(served.served_reason.call_count, 2)                                   # the f32 calls reached the serve layer's own rule; the others never did
        self.assertIn("dtype", fused_trimul.EXPECTED_FALLBACKS); self.assertIn("dtype", triattn.EXPECTED_FALLBACKS)   # declared: named as fallback_by on the LEVER line, the run keeps its exit code


if __name__ == "__main__":
    unittest.main()
