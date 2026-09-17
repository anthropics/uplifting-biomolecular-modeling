"""CPU tests of the package surface that needs no TensorFlow: the .pth text is the build backend's, the autoload switch grammar, the weights
gate refuses by name, the line grammar, the pool rewrite on a synthetic graph."""
import hashlib
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.dirname(OPT)


class TestPackage(unittest.TestCase):
    def test_pth_is_generated_text(self):
        sys.path.insert(0, OPT)
        import _build_backend as bb
        want = bb.pth_text("enformer_deepmind_opt", "ENFORMER_DEEPMIND_OPT", "enformer-deepmind-opt")
        with open(os.path.join(OPT, "enformer_deepmind_opt_autoload.pth"), encoding="utf-8") as f:
            self.assertEqual(f.read(), want)

    def test_check_pins_weights_gate(self):
        import tempfile
        d = tempfile.mkdtemp()
        r = subprocess.run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--weights", d], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("WEIGHTS REFUSED", r.stderr)

    def test_unknown_switch_value_exits_3(self):
        env = dict(os.environ, ENFORMER_DEEPMIND_OPT="fast", PYTHONPATH=OPT)
        r = subprocess.run([sys.executable, "-c", "import enformer_deepmind_opt._autoload"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("NOT ACTIVE mode=fast reason=unknown ENFORMER_DEEPMIND_OPT value", r.stderr)

    def test_ops_build_record(self):
        sys.path.insert(0, OPT)
        from enformer_deepmind_opt import ops
        info = ops.build_info()                                                   # BUILD.json alone: no TensorFlow needed
        self.assertEqual(sorted(info["ops"]), ["EdmBias2Residual", "EdmBiasAct", "EdmBiasGelu", "EdmBiasResidual", "EdmBiasScaleShiftGelu", "EdmLayerNorm", "EdmPoolLogits", "EdmQScaleBias", "EdmRelShiftSoftmax", "EdmScaleShiftGelu", "EdmSoftmaxPool2", "EdmSoftmaxPool2Gelu"]); self.assertEqual(info["build"], "sm_80+sm_90")
        self.assertEqual(info["tensorflow"], "2.17.1"); self.assertEqual([tuple(a) for a in info["archs"]], [(8, 0), (9, 0)])
        import hashlib
        for name, digest in info["sources"].items():
            with open(os.path.join(os.path.dirname(ops.__file__), name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, name)   # the sources beside the object are the ones it was built from
        with open(ops.SO_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), info["object"]["sha256"])

    def test_lines(self):
        sys.path.insert(0, OPT)
        from enformer_deepmind_opt import _runtime as rt
        self.assertEqual(rt.line_of({"active": False, "reason": "no CUDA device visible"}), "[enformer-deepmind-opt] NOT ACTIVE mode=exact reason=no CUDA device visible")
        rep = {"active": True, "dry_run": True, "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": (9, 0)}, "kit_version": "0.1.0", "levers": ["pool", "poolgemm", "bngelu", "bnconst", "poscache", "relsoftmax", "biasres", "biasgelu", "poolgelu", "hostshape", "layernorm"], "build": "sm_90", "notes": []}
        self.assertEqual(rt.line_of(rep), "[enformer-deepmind-opt] DRY mode=exact gpu=NVIDIA H100 80GB HBM3(sm90) kit=0.1.0 levers=pool,poolgemm,bngelu,bnconst,poscache,relsoftmax,biasres,biasgelu,poolgelu,hostshape,layernorm build=sm_90")
        rep.update(dry_run=False, notes=["tensorflow 2.18.0 differs from the tested 2.17.1 (the levers engage; byte-identity to stock was established on that build)"])
        self.assertTrue(rt.line_of(rep).startswith("[enformer-deepmind-opt] ACTIVE mode=exact gpu=NVIDIA H100 80GB HBM3(sm90) kit=0.1.0 levers=pool,poolgemm,bngelu,bnconst,poscache,relsoftmax,biasres,biasgelu,poolgelu,hostshape,layernorm build=sm_90 applied=deferred notes=tensorflow 2.18.0"))


if __name__ == "__main__":
    unittest.main()
