"""Carried modules of the ESM-family kits (byte-identical to their kit copies):

    ef2_pair_v2            row esm_t15: the inference kit's one-kernel residual pair transition (Triton; sm90 and sm80 rows)
    ef2_autograd_kernels   row esm_kd3: the design kit's differentiable TransitionRefround (K-D3: fast | lean, an out-only forward kernel pluggable
                           under lean) and the two SwiGLU rounding kernels its forward / backward use
    ef2_t16_transition     row esm_t16: the sm_90a CUDA C++ / CuTe persistent pair-transition FORWARD (cubin carried under ef2_t16/sm_90a with its
    ef2_t16_nvjit          manifest; loaded through the CUDA driver, no compiler at run time) and its loader

``ef2_pair_v2`` and ``ef2_autograd_kernels`` import the ESM-family image's ``transformers.models.esmfold2`` fork at module level: import them only
where that image runs (the face turns the ImportError into a named refusal).  The design kit's modules import one another by their kit's
top-level names (``import ef2_t16_transition``, ``import ef2_t16_nvjit``); ``bind_names()`` binds those two names to THIS package's copies in
``sys.modules`` before the face imports them (a process whose kit already imported its own byte-identical copies keeps those: setdefault)."""
import importlib
import sys

SIBLING_NAMES = ("ef2_t16_nvjit", "ef2_t16_transition")     # import order: the loader first (the transition module imports it at module level)


def bind_names():
    """Make the design kit's sibling imports resolve to the carried copies (idempotent; an existing binding is kept)."""
    bound = {}
    for name in SIBLING_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            mod = importlib.import_module(__name__ + "." + name)
            mod = sys.modules.setdefault(name, mod)
        bound[name] = getattr(mod, "__file__", None)
    return bound
