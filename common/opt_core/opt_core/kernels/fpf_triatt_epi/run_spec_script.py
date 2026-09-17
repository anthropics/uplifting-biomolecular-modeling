#!/usr/bin/env python3
"""run_spec_script.py <path/to/bench.py|test_numerics.py> [args...] — installs the fp64 LayerNorm fallback (lead broadcast 05:52Z: fast_layernorm has no fp64 kernel,
ref64_call would raise and the spec scripts silently fall back to stock-as-reference) and then executes the spec script unchanged."""
import sys, os, runpy
import torch, torch.nn.functional as F
from protenix.model.layer_norm.layer_norm import FusedLayerNorm
if not getattr(FusedLayerNorm, "_fpf_fp64_patched", False):
    _orig = FusedLayerNorm.forward
    def _fwd(self, x):
        if x.dtype == torch.float64:
            w = None if self.weight is None else self.weight.to(x.dtype); b = None if self.bias is None else self.bias.to(x.dtype)
            return F.layer_norm(x, self.normalized_shape, w, b, self.eps)
        return _orig(self, x)
    FusedLayerNorm.forward = _fwd; FusedLayerNorm._fpf_fp64_patched = True
    print("[run_spec_script] fp64 LayerNorm fallback installed (ref64 only; stock paths untouched)", flush=True)
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
