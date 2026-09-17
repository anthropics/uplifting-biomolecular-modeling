"""Lever distogram_offload (class exact): model.predict computes the distogram logits (fp32 [B, L, L, 64], model.py:311-313) BEFORE diffusion
sampling and the confidence head and keeps them on the GPU until the runner copies them to the host (runner.py model_run: `.cpu().float().numpy()`,
and only when return_distogram).  This lever moves the logits to (pinned) host memory as soon as they are produced, so L²·64·4 bytes (1.07 GB at
2,048 tokens, 4.3 GB at 4,096) leave the peak window of sampling + confidence.  Values are untouched (a device-to-host copy).  Site:
atlasfold.model.network.distogram_head:DistogramHead.forward."""
from __future__ import annotations

import torch

from opt_core.counters import Ledger

from . import Installed

NAME = "LOCAL.atlasfold.distogram_offload"


def install(mode: str, tag: str, settings: dict):
    from atlasfold.model.network import distogram_head as DH
    ledger = Ledger(NAME, impl="logits.to(cpu,pinned)", origin="kit", expected=("not_cuda",))
    stock = DH.DistogramHead.forward

    def forward(self, z, *a, **k):
        out = stock(self, z, *a, **k)
        lg = out.get("logits") if isinstance(out, dict) else None
        if isinstance(lg, torch.Tensor) and lg.is_cuda:
            host = torch.empty(lg.shape, dtype=lg.dtype, device="cpu", pin_memory=True)
            host.copy_(lg, non_blocking=False)
            out["logits"] = host
            del lg
            ledger.serve("L%d" % out["logits"].shape[-2])
        else:
            ledger.fallback("not_cuda")
        return out

    forward.__wrapped_stock__ = stock
    DH.DistogramHead.forward = forward
    return Installed("distogram_offload", True, lines=[lambda: ledger.line(tag)], gates=[ledger.gate])
