"""Lever diffusion_bf16.

diffusion_bf16 — stock runs the whole diffusion sampler with autocast DISABLED (model.py L316-319; model_multimer.py L381) and fp32
``precision=32`` linears; here DiffusionHead.sample runs under bf16 autocast, so the 12-block token DiffusionTransformer (the O(steps x L^2)
part) computes in bf16 while the atom encoder/decoder keep their own inner fp32 islands (diffusion_head.py L151, L178) and every LayerNorm
keeps fp32 statistics (primitives/normalization.py). Class: fast, accuracy-gated."""
from . import Installed, rebind


def install_bf16(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    import torch
    from opt_core.counters import Ledger
    try:
        dh = importlib.import_module("atlasfold.model.network.diffusion_head")
    except Exception as e:  # noqa: BLE001
        return Installed("diffusion_bf16", False, reason=f"import:diffusion_head:{type(e).__name__}")
    cls = dh.DiffusionHead
    stock = cls.sample
    ledger = Ledger("LOCAL.atlasfold.diffusion_bf16", impl="torch.autocast(bf16)", origin="kit", expected=("cpu",))

    def sample(self, *args, **kwargs):
        dev = next(self.parameters()).device
        if dev.type != "cuda":
            ledger.fallback("cpu")
            return stock(self, *args, **kwargs)
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            out = stock(self, *args, **kwargs)
        ledger.serve("sample")
        return out.float() if hasattr(out, "float") else out
    sample.__qualname__ = "DiffusionHead.sample[atlasfold_opt:diffusion_bf16]"
    rebind(cls, "sample", sample, stock)
    return Installed("diffusion_bf16", True, lines=[lambda: ledger.line(tag)], gates=[ledger.gate])

