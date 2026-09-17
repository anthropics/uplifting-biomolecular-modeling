"""infopt_graphs — engine-agnostic CUDA-graph capture + static-shape bucketing for PyTorch AF3-class inference."""
from .core import GraphedFunction, StaticGraph, signature_of, static_like, copy_into, graphs_enabled
from opt_core.tools.graph_audit.audit import SyncCensus, LaunchCounter, RNGGuard, rng_fingerprint, audit_capture_safety, sync_census
from .bucket import Bucketer, pad_to_bucket, batch_pad

__version__ = "0.1.0"
__all__ = ["GraphedFunction", "StaticGraph", "signature_of", "static_like", "copy_into", "graphs_enabled", "SyncCensus",
           "LaunchCounter", "RNGGuard", "rng_fingerprint", "audit_capture_safety", "sync_census", "Bucketer", "pad_to_bucket", "batch_pad"]
