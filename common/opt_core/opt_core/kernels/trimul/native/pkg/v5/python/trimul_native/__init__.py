"""trimul_native -- triangle multiplication (the pair-stack "triangle multiplicative update", outgoing and incoming) as arch-keyed CUDA
cubins launched through the CUDA driver from Python, behind one provider face.

    from trimul_native import face
    face.admits(...)  face.check(...)  face.pack_weights(...)  face.serve(...)      # see face.py
    python -m trimul_native.build --archs sm_90a,sm_80                             # compile csrc/ -> build/<arch>/<unit>.cubin + manifest

Modules: ``face`` (the boundary), ``launch`` (module cache, argument packing, tensor-map encoding, launches on the framework's stream),
``_driver`` (the driver binding: ctypes over libcuda.so.1, or the cuda.bindings wheel), ``manifest`` (build record + digests), ``build``
(the nvcc driver script).  Importing the package imports nothing of the framework; ``face.serve`` / ``face.check`` / ``launch`` import torch
when called.
"""
from .face import Refusal, admits, describe, resolve_cell, cells, notices, add_notice_sink, VERSION            # noqa: F401

__all__ = ["Refusal", "admits", "describe", "resolve_cell", "cells", "notices", "add_notice_sink", "VERSION"]
__version__ = VERSION
