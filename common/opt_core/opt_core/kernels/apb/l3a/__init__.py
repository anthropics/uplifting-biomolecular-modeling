"""l3a -- the attention / trunk kernels of lever L3a, one module per lever, each behind ONE flag; nothing here is imported by the frozen
modes.  Standard library only at import time: submodules (which import torch / triton inside functions or under a guard) load on first attribute
access.

    fab_batched   lever L3a  shared-bias batched pair-bias flash attention (flag KOPT_ATTN_L3A = 0 | a | c)
    bias_layout   lever L3a  PairBiasLayout: ONE padded [L, H, N, Np] bias buffer (aligned rows, power-of-two stride skew)
"""
import importlib

_LAZY = ("fab_batched", "bias_layout")


def __getattr__(name):          # PEP 562: lazy submodules, no torch / triton import at package import
    if name in _LAZY:
        return importlib.import_module(__name__ + "." + name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
