"""infopt_graphs — model-agnostic CUDA-graph capture for PyTorch co-folding inference."""
from .core import GraphedFunction, StaticGraph, signature_of, static_like, copy_into, graphs_enabled
from opt_core.tools.graph_audit.audit import SyncCensus, LaunchCounter, RNGGuard, rng_fingerprint, audit_capture_safety, sync_census

__version__ = "0.1.0"
__all__ = ["GraphedFunction", "StaticGraph", "signature_of", "static_like", "copy_into", "graphs_enabled", "SyncCensus",
           "LaunchCounter", "RNGGuard", "rng_fingerprint", "audit_capture_safety", "sync_census", ]
