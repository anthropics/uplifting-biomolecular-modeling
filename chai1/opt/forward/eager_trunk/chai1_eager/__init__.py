# Derived from Chai-1 (chai-lab 0.6.1), Copyright 2024 Chai Discovery, Inc., Apache-2.0 — see NOTICE.
"""chai1_eager — eager-PyTorch re-expression of the Chai-1 trunk (exported TorchScript `models_v2/trunk.pt`).

    from chai1_eager.trunk import load_trunk, CFG          # structured nn.Modules (bitwise-identical to the scripted trunk)
    from chai1_eager.ts2eager import load_eager_component   # generic TorchScript-code-tree -> eager transpiler (any Chai-1 component)
    from chai1_eager import kernels                         # copy-free trimul layout (bitwise-identical, faster)
    from chai1_eager.hoist import HoistedForward, HoistedForward2   # step-invariant hoisting (name taint | value taint) + CUDA graph for the diffusion module
"""
__version__ = "0.4.8+errata01"
