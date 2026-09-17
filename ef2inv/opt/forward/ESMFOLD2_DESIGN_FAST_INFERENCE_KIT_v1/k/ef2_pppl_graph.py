"""ef2_pppl_graph — exact lever for the ESM cookbook's ESMC pseudo-perplexity loss (compute_esmc_pseudoperplexity_nll):
CUDA-graph capture of the ESMC-6B `TransformerStack.forward` FORWARD AND BACKWARD (grad w.r.t. the input embeddings;
all LM weights are frozen) at a fixed input shape [n_passes*(design batch), L_binder+2, d_model].

Why: per design step the cookbook runs the 80-layer 6B LM forward+backward on 4 masked copies of the binder
(~96-275 tokens each) under autograd; that is ~10k small kernel launches and mostly host-bound (like the fold trunk was).
The masking RNG, one-hot/straight-through construction, lm_head, log-softmax and NLL stay eager and untouched.

Exactness: replay executes the kernels recorded from the eager code path on the same shapes/dtypes -> bitwise identical
loss and gradient, provided the eager path is itself deterministic (it is in the tested builds: the stock cookbook with
det_scatter reproduces bitwise run-to-run, which includes this backward).  Checked at
build time: loss and d loss/d binder_design torch.equal eager vs graphed for 3 shapes, and end-to-end (identical designed sequences).

Usage (cookbook):   import ef2_pppl_graph; ef2_pppl_graph.enable(esmc_model)      # ESMCForMaskedLM or its .esmc
Only calls with grad enabled, an input that requires grad, no inference-mode, sequence_id=None, layers_to_collect empty and
output_attentions False are graphed (exactly the cookbook's pPPL call); everything else (the fold model's own use of the
shared ESMC trunk under inference_mode, the K-D LM-forward graph capture) falls through to the original forward.
Memory: one captured graph per input shape (LRU, max_entries=2) holding that shape's saved activations.
"""
from __future__ import annotations
import torch
from torch import Tensor


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ent):
        ent["x"].copy_(x)
        ent["fwd"].replay()
        ctx.ent = ent
        return ent["norm_x"].detach().clone()

    @staticmethod
    def backward(ctx, g):
        ent = ctx.ent
        ent["g_out"].copy_(g)
        ent["bwd"].replay()
        return ent["g_in"].clone(), None


class GraphedTransformerStack:
    def __init__(self, stack: torch.nn.Module, n_warmup: int = 2, max_entries: int = 1, check_replay: bool = True):
        self.stack, self.n_warmup, self.max_entries, self.check_replay = stack, n_warmup, max_entries, check_replay
        self.orig_forward = stack.forward
        self.entries: dict = {}
        self.stats = dict(captures=0, replays=0, eager=0, replay_ok=0, replay_fail=0, pool_gb=[], evictions=0)
        self.disabled_reason = ""

    def _graphable(self, x, sequence_id, layers_to_collect, output_attentions) -> bool:
        return (not self.disabled_reason and torch.is_tensor(x) and x.is_cuda and torch.is_grad_enabled() and x.requires_grad
                and not torch.is_inference_mode_enabled() and not torch.cuda.is_current_stream_capturing()
                and sequence_id is None and not layers_to_collect and not output_attentions)

    def __call__(self, x: Tensor, sequence_id=None, layers_to_collect=None, output_attentions: bool = False):
        if not self._graphable(x, sequence_id, layers_to_collect, output_attentions):
            self.stats["eager"] += 1
            return self.orig_forward(x, sequence_id=sequence_id, layers_to_collect=layers_to_collect, output_attentions=output_attentions)
        key = (tuple(x.shape), x.dtype, tuple(x.stride()), torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None, x.device.index)
        ent = self.entries.get(key)
        if ent is None:
            try:
                ent = self._capture(x, key)
            except Exception as e:                      # capture unsupported (backend, allocator state...) -> eager forever, loudly
                self.disabled_reason = f"capture failed: {type(e).__name__}: {str(e)[:200]}"
                import warnings; warnings.warn("ef2_pppl_graph: " + self.disabled_reason + " -> eager")
                torch.cuda.synchronize()
                self.stats["eager"] += 1
                return self.orig_forward(x, sequence_id=sequence_id, layers_to_collect=layers_to_collect, output_attentions=output_attentions)
        self.stats["replays"] += 1
        norm_x = _Replay.apply(x, ent)
        # (norm_x, x_prenorm, collected, attentions): the cookbook uses only norm_x; x_prenorm is returned detached (static buffer)
        return norm_x, ent["x_pre"].detach(), (), None

    def _capture(self, x: Tensor, key):
        import gc
        while len(self.entries) >= self.max_entries:          # release the evicted entry (graphs + private pool) BEFORE capturing a new one
            _, old = self.entries.popitem()
            for k_ in ("bwd", "fwd", "norm_x", "x_pre", "g_out", "g_in", "x"):
                old.pop(k_, None)
            del old
            self.stats["evictions"] += 1
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        m0 = torch.cuda.memory_allocated()
        # static input: a LEAF buffer copied into; the stack gets a non-leaf view (see note in module docstring)
        xs = x.detach().clone().requires_grad_(True)
        def fwd():
            x_nl = xs.view_as(xs)                       # non-leaf view, zero kernels; grads flow to xs
            return self.orig_forward(x_nl, sequence_id=None, layers_to_collect=[], output_attentions=False)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                      # warm-up: workspaces, autotune, allocator
            for _ in range(self.n_warmup):
                out = fwd(); norm_x = out[0]
                g = torch.ones_like(norm_x)
                gin, = torch.autograd.grad(norm_x, xs, g)
                del out, norm_x, g, gin
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        fwd_graph = torch.cuda.CUDAGraph(); bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd_graph, pool=pool):
            out = fwd()
        norm_x, x_pre = out[0], out[1]
        g_out = torch.empty_like(norm_x)
        with torch.cuda.graph(bwd_graph, pool=pool):
            g_in, = torch.autograd.grad(norm_x, xs, g_out, retain_graph=True)
        torch.cuda.synchronize()
        ent = dict(x=xs, fwd=fwd_graph, bwd=bwd_graph, norm_x=norm_x, x_pre=x_pre, g_out=g_out, g_in=g_in, pool=pool, key=key)
        if self.check_replay:                                 # replay vs eager on the capture input, fwd and bwd, bitwise
            xe = x.detach().clone().requires_grad_(True)
            oe = self.orig_forward(xe.view_as(xe), sequence_id=None, layers_to_collect=[], output_attentions=False)[0]
            ge = torch.randn_like(oe)
            gie, = torch.autograd.grad(oe, xe, ge)
            with torch.no_grad():
                xs.copy_(x.detach()); g_out.copy_(ge)
            fwd_graph.replay(); bwd_graph.replay(); torch.cuda.synchronize()
            ok = torch.equal(oe, norm_x) and torch.equal(gie, g_in)
            self.stats["replay_ok" if ok else "replay_fail"] += 1
            if not ok:
                self.disabled_reason = f"replay != eager at capture (max|d out| {(oe - norm_x).abs().max().item():.3e}, max|d grad| {(gie - g_in).abs().max().item():.3e})"
                import warnings; warnings.warn("ef2_pppl_graph: " + self.disabled_reason + " -> eager")
                raise RuntimeError(self.disabled_reason)
        torch.cuda.synchronize()
        self.stats["pool_gb"].append(round((torch.cuda.memory_allocated() - m0) / 2**30, 2))   # live bytes held by this entry
        self.entries[key] = ent; self.stats["captures"] += 1
        return ent


def _stack_of(obj):
    if hasattr(obj, "esmc") and hasattr(obj.esmc, "transformer"):     # ESMCForMaskedLM
        return obj.esmc.transformer
    if hasattr(obj, "transformer"):                                   # ESMCModel
        return obj.transformer
    return obj                                                        # the stack itself


def enable(esmc_or_lm, **kw) -> GraphedTransformerStack:
    stack = _stack_of(esmc_or_lm)
    if hasattr(stack, "_pppl_graph"):
        return stack._pppl_graph
    g = GraphedTransformerStack(stack, **kw)
    stack._pppl_graph = g
    stack.forward = g                    # instance attribute: Module.__call__ -> this wrapper; the class forward stays intact
    return g


def disable(esmc_or_lm) -> None:
    stack = _stack_of(esmc_or_lm)
    g = getattr(stack, "_pppl_graph", None)
    if g is not None:
        stack.forward = g.orig_forward
        del stack._pppl_graph


def stats(esmc_or_lm):
    g = getattr(_stack_of(esmc_or_lm), "_pppl_graph", None)
    return None if g is None else dict(g.stats, disabled_reason=g.disabled_reason, entries=len(g.entries))


def release(esmc_or_lm) -> None:
    """Free captured pPPL graphs (wrapper stays installed; next grad-enabled call re-captures)."""
    g = getattr(_stack_of(esmc_or_lm), "_pppl_graph", None)
    if g is not None:
        g.entries.clear()
        import gc; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
