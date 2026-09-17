"""ef2_esmc_graph — K-D stretch: CUDA-graph capture of the ESMC-6B forward used inside the ESMFold2 design step.

`compute_lm_hidden_states()` calls `esmc(input_ids=[B, max_len], sequence_id=[B, max_len], output_hidden_states=True)`
under `torch.inference_mode()` once per design step; the LM input length is fixed for a whole design trajectory
(target + binder + BOS/EOS per chain), so the 80-layer forward (thousands of kernel launches, launch-bound at design
sizes) can be captured once per (B, max_len) and replayed.  Replay is bitwise-identical to eager as long as
the same SDPA backend is selected (the capture records whatever eager dispatches to).

Usage:
    import ef2_esmc_graph as eeg
    eeg.enable(model)        # wraps model._esmc.forward; captures lazily per (shape, dtype, flags); eeg.disable(model)
Only calls with output_hidden_states=True and tensors on CUDA are graphed; anything else falls through to eager.
"""
from __future__ import annotations
import torch


class _GraphedESMC:
    def __init__(self, esmc, n_warmup=2, max_entries=2):
        self.esmc, self.n_warmup, self.max_entries = esmc, n_warmup, max_entries
        self.orig_forward = esmc.forward
        self.entries = {}          # key -> dict(graph, ids, seq, out)
        self.stats = dict(captures=0, replays=0, eager=0)

    def __call__(self, *args, **kw):
        input_ids = kw.get("input_ids", args[0] if args else None)
        sequence_id = kw.get("sequence_id"); ohs = kw.get("output_hidden_states")
        graphable = (input_ids is not None and torch.is_tensor(input_ids) and input_ids.is_cuda and ohs is True
                     and not torch.is_grad_enabled() and len(args) <= 1
                     and set(kw) <= {"input_ids", "sequence_id", "output_hidden_states"})
        if not graphable:
            self.stats["eager"] += 1
            return self.orig_forward(*args, **kw)
        key = (tuple(input_ids.shape), input_ids.dtype, None if sequence_id is None else (tuple(sequence_id.shape), sequence_id.dtype),
               torch.is_autocast_enabled(), torch.is_inference_mode_enabled())
        ent = self.entries.get(key)
        if ent is None:
            if len(self.entries) >= self.max_entries:
                self.entries.pop(next(iter(self.entries)))
            ids = input_ids.clone(); seq = sequence_id.clone() if sequence_id is not None else None
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.n_warmup):
                    self.orig_forward(input_ids=ids, sequence_id=seq, output_hidden_states=True)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.orig_forward(input_ids=ids, sequence_id=seq, output_hidden_states=True)
            torch.cuda.synchronize()
            ent = dict(graph=g, ids=ids, seq=seq, out=out); self.entries[key] = ent; self.stats["captures"] += 1
        ent["ids"].copy_(input_ids)
        if ent["seq"] is not None: ent["seq"].copy_(sequence_id)
        ent["graph"].replay(); self.stats["replays"] += 1
        out = ent["out"]
        # return a copy-free view object of the same type; consumers (compute_lm_hidden_states) read .hidden_states immediately
        return out


def enable(model, **kw) -> _GraphedESMC:
    esmc = model._esmc
    if hasattr(esmc, "_eeg"):
        return esmc._eeg
    g = _GraphedESMC(esmc, **kw); esmc._eeg = g; esmc.forward = g
    return g


def disable(model) -> None:
    esmc = model._esmc
    if hasattr(esmc, "_eeg"):
        esmc.forward = esmc._eeg.orig_forward; del esmc._eeg


def release(model):
    """Free captured LM-forward graphs (wrapper stays installed; next call re-captures)."""
    g = getattr(getattr(model, "_esmc", None), "_eeg", None)
    if g is not None:
        g.entries.clear()
    import gc; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
