"""The gate: refused only for no CUDA device / evo2 or vtx absent or off their pins; everything else named on the line."""
import unittest
from unittest import mock

from evo2_opt import activation as A

PINS = {"upstream": {"evo2": {"version": "0.6.0"}, "vtx": {"version": "1.1.0"}},
        "stacks": {"img_full": {"pins": {"torch": "2.7.1+cu128", "triton": "3.3.1", "flash_attn": "2.8.0.post2", "transformer_engine": "2.3.0"}},
                   "img_a100": {"pins": {"torch": "2.7.1+cu128", "triton": "3.3.1", "flash_attn": "2.8.0.post2", "transformer_engine": "absent"}}},
        "gpus": {"h100": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm_90", "memory_mib": 81559}, "a100_80gb": {"name": "NVIDIA A100-SXM4-80GB", "sm": "sm_80", "memory_mib": 81152}}}
H100 = {"count": 1, "torch": True, "devices": [{"index": 0, "name": "NVIDIA H100 80GB HBM3", "mib": 81559, "sm": "sm_90"}]}
L40 = {"count": 1, "torch": True, "devices": [{"index": 0, "name": "NVIDIA L40S", "mib": 45589, "sm": "sm_89"}]}


def gate(versions, gpu=H100, te="2.3.0"):
    with mock.patch.object(A.P, "dist_version", side_effect=lambda n: versions.get(n)), mock.patch.object(A.P, "te_version", return_value=te), \
            mock.patch.object(A, "gpu_info", return_value=gpu):
        return A.gate(PINS)


class TestGate(unittest.TestCase):
    def test_pinned_stack_on_h100_is_clean(self):
        g = gate({"evo2": "0.6.0", "vtx": "1.1.0", "torch": "2.7.1", "triton": "3.3.1", "flash_attn": "2.8.0.post2"})
        self.assertTrue(g["ok"], g); self.assertEqual(g["notes"], []); self.assertEqual(g["stack"], "img_full")
        self.assertIn("listed(h100)", g["gpu"])
        self.assertIn("ACTIVE mode=exact evo2=0.6.0 vtx=1.1.0 stack=img_full transformer_engine=2.3.0", A.active_line(g))

    def test_refusals(self):
        self.assertIn("evo2 is not installed", gate({"vtx": "1.1.0"})["reason"])
        self.assertIn("another version is another stock", gate({"evo2": "0.7.0", "vtx": "1.1.0"})["reason"])
        g = gate({"evo2": "0.6.0", "vtx": "1.1.0"}, gpu={"count": 0, "torch": True, "devices": []})
        self.assertFalse(g["ok"]); self.assertIn("no CUDA device", g["reason"])

    def test_environment_is_named_not_refused(self):
        g = gate({"evo2": "0.6.0", "vtx": "1.1.0", "torch": "2.8.0+cu129", "triton": "3.3.1"}, gpu=L40)
        self.assertTrue(g["ok"], g)
        self.assertIn("unlisted: torch 2.8.0+cu129 != pinned 2.7.1+cu128 (stock/PINS.json stacks.img_full)", g["notes"])
        self.assertIn("unlisted(sm_89, 45 GiB", g["gpu"])
        g = gate({"evo2": "0.6.0", "vtx": "1.1.0", "torch": "2.7.1"}, te=None)
        self.assertTrue(g["ok"]); self.assertEqual(g["stack"], "img_a100"); self.assertTrue(any("transformer_engine absent" in n for n in g["notes"]))
        self.assertIn("transformer_engine=absent", A.active_line(g))

    def test_enable_rejects_other_modes_and_off_is_stock(self):
        self.assertFalse(A.enable("off")["active"])
        self.assertEqual(A.MODES, ("exact", "fast"))
        with self.assertRaises(A.Evo2OptRefused):
            A.enable("faster")


if __name__ == "__main__":
    unittest.main()
