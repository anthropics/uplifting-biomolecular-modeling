"""ef2_bf16_confidence — run ESMFold2-Experimental's confidence head under bf16 autocast (FAST/BIG-class candidate lever).

`ESMFold2ExperimentalModel.forward` runs the trunk under bf16 autocast but calls the confidence head AFTER the autocast region, on
`z.detach().float()`: its 4 PairUpdateBlocks (triangle multiplication, transitions, pair-biased attention), the pLDDT / PAE / PDE heads
and the TM-score reductions execute in fp32, at several times the cost of the same block in the bf16 trunk.
`enable(model)` wraps `model.confidence_head.forward` (instance-level)
in `torch.autocast("cuda", dtype=torch.bfloat16)` and casts every floating output back to float32, so consumers see the stock dtypes.

Numerics class: FAST (bf16 contraction inside the confidence trunk; the design gradient path, the sampled coordinates and every
denoising-network output are untouched — the head consumes detached tensors and feeds nothing back). The bf16 head's iptm / ptm / plddt
differ from the fp32 head's by less than the fp32 head's own run-to-run variation under the production (non-deterministic) recipe. The design loop reads `iptm` on its confidence steps to pick the best
step, so this lever belongs to fast / memory-lean compositions only, never to an exact one.

Usage:
    import ef2_bf16_confidence as bc
    bc.enable(model)      # idempotent;  bc.disable(model) restores;  a model without a confidence head is left untouched (returns False)
"""
from __future__ import annotations

import types

import torch


def _cast_back(out):
    if torch.is_tensor(out):
        return out.float() if out.is_floating_point() and out.dtype != torch.float32 else out
    if isinstance(out, dict):
        return {k: _cast_back(v) for k, v in out.items()}
    if isinstance(out, (list, tuple)):
        return type(out)(_cast_back(v) for v in out)
    return out


def _bf16_forward(self, *args, **kw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = self._bc_orig_forward(*args, **kw)
    return _cast_back(out)


def enable(model) -> bool:
    head = getattr(model, "confidence_head", None)
    if head is None:
        return False
    if not hasattr(head, "_bc_orig_forward"):
        head._bc_orig_forward = head.forward
        head.forward = types.MethodType(_bf16_forward, head)
    return True


def disable(model) -> None:
    head = getattr(model, "confidence_head", None)
    if head is None:                                            # skip-unused-confidence parks the head at None only inside a call
        return
    if hasattr(head, "_bc_orig_forward"):
        if vars(head).get("forward") is not None and getattr(head.forward, "__func__", None) is not _bf16_forward:
            raise RuntimeError("ef2_bf16_confidence.disable: confidence_head.forward was re-wrapped after enable(); disable that lever first (LIFO)")
        del head.forward                                        # back to the class's forward
        del head._bc_orig_forward
