"""The one exception every opt_core.ops cell raises for a call outside its served domain (rank, device, dims, dtype): the engine
adapter catches it and runs the stock module for that call, counted -- never an error. When the FPF add-on's ``fpf_engines`` registry is
importable its ``FPFFallback`` IS this class (one class for both producers); otherwise a local class of the same name and shape."""
try:
    from fpf_engines import FPFFallback          # the FPF add-on's registry (FPF_ENGINES_v0.1), when present
except Exception:                                 # pragma: no cover
    class FPFFallback(Exception):
        def __init__(self, reason):
            super().__init__(reason)
            self.reason = reason

__all__ = ["FPFFallback"]
