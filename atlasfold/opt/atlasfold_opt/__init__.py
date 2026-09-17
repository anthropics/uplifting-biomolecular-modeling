"""atlasfold_opt — the AtlasFold inference optimization kit (upstream github.com/SeonghwanSeo/atlasfold v1.0.0, commit 992067e).

    import atlasfold_opt
    report = atlasfold_opt.enable("exact")      # or "fast" | "big"; idempotent, once per process ("off" = stock: pred --mode off)
    atlasfold_opt.status()

or, without code changes, ``ATLASFOLD_OPT=fast atlasfold monomer ...``: the package's .pth installs a lazy import hook that activates the
mode the first time the atlasfold model family (``atlasfold.model``, ``atlasfold.runner``, ``atlasfold.runner_multimer``,
``atlasfold.pretrained``) is imported.

Modes (``modes.MODES`` lists the levers per mode; ``registry.LEVERS`` describes each lever; ``stack.py`` applies them):
  "off"   = stock AtlasFold, nothing patched (``pred --mode off`` runs the stock CLI in a clean subprocess).
  "exact" = stock arithmetic, reorganized — byte-identical outputs to "off" under --det 1: the TriangleMultiplication, triangle attention,
            SwiGLU transition and LayerNorm sites go through the opt_core providers by the tier word `exact` (an exact-class row where the
            provider records one bitwise for this stack, else the stock op by name), plus the memory levers and the lever report.
  "fast"  = the tier word `fast` on the same providers, the fused pair-stack kernels, SDPA softmax*V in AtlasLM's logit-exporting attention,
            bf16 / TF32 arithmetic in the diffusion module and the denoiser captured in a CUDA graph: small numeric differences, faster.
  "big" = the memory mode: fast's levers minus the CUDA-graph levers (denoiser_graph, graph_reuse), the providers asked for their `big` rows.
Importing the package loads neither torch nor atlasfold; both load when a mode is activated.
"""
__version__ = "0.2.44"
TAG = "atlasfold-opt"
ENV = "ATLASFOLD_OPT"
STOCK = {"name": "atlasfold", "tag": "v1.0.0", "commit": "992067e67df29b665c501e0d2e9ead9dd4ba9b69", "version": "1.0.0"}
from .hooks import LeverAborted   # noqa: E402 — a lever that ends an item by name raises this (API callers map it to the refusal class, exit 3)

__all__ = ["enable", "status", "MODES", "ActivationError", "LeverAborted", "registry", "modes", "stack", "TAG", "ENV", "STOCK"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (core/stock/GPU gate, or a lever failed to apply)."""


def _refuse_without_core() -> None:
    """First statement of every in-process entry: the opt_core pin gate (``_core_gate``; an absent or older opt_core prints
    ``[atlasfold-opt] NOT ACTIVE: reason=...`` and raises SystemExit(3)) — never a traceback, never a kit mode on an unpinned core."""
    from . import _core
    _core.gate()


def enable(mode: str, *, strict: bool = False, trigger: str = None, det: int = None, n_gpu: int = 1, allow_partial: bool = None, inputs: dict = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") in this process; returns the activation report. Idempotent."""
    _refuse_without_core()
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger, det=det, n_gpu=n_gpu, allow_partial=allow_partial, inputs=inputs)


def status() -> dict:
    _refuse_without_core()
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import atlasfold_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
