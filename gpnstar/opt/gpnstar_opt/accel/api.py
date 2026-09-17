"""One-call entry points.

    from gpnstar_opt.accel.api import make_exact, verify_identity
    model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-star-hg38-v100-200m", revision=...).cuda().eval()
    make_exact(model)                       # in place; large-batch default (P0+P1+P2+P4 dedup, eager)
    runner = make_exact(model, dedup=False, cuda_graph_batch=batch)   # small-batch: static P3b + CUDA graph

Works on any GPNStarForMaskedLM / GPNStarModel instance, e.g. the `.model` inside
gpn.star.inference.MLMforVEPModel / MLMforLogitsModel / MLMforEmbeddingModel (the HF Trainer then
drives the patched module unchanged; the CUDA-graph runner is for custom loops with a fixed shape).

Contract (true for gpn's inference paths): one target row (T=1), target_species == 0 for every
window, eval mode, fp32, TF32 off (stock defaults).  The first forward for a new batch shape
validates target_species and caches the shape-dependent constants.
"""

from __future__ import annotations

import torch

from .colattn import patch_col_attention_layout  # noqa: F401
from .fused_col import fused_report, patch_fused_col_attention  # noqa: F401
from .graphs import graph_report, patch_graphs, unpatch_graphs  # noqa: F401
from .patches import apply_exact, apply_fast_policy, kv_mode_report, patch_unified_kv, set_static_capacities, static_overflow  # noqa: F401
from .runtime import GraphRunner, compare_logits


def make_exact(model, *, dedup: bool = True, min_rows: int = 2048, cuda_graph_batch: dict | None = None,
               on_dedup_reject: str = "p3b", graph: bool = True, graph_max_tokens: int | None = None):
    """Patch `model` in place.  dedup=True (P4) gives the largest gain at batch >= 8 but has
    data-dependent shapes (eager only).  dedup=False (P3b) is static and CUDA-graph capturable:
    pass `cuda_graph_batch` (example batch dict with the production shape) to get a GraphRunner.

    AUTO-SELECT: the layer-0 reduced-vs-full K/V GEMM self-check decides per batch shape.  If it rejects
    the de-duplicated GEMM for a shape `dedup_max_rejects` (=2) times -- observed e.g. at eval batch 128 x 128 bp on
    H100, where cuBLAS picks a different kernel for the small de-duplicated M -- that shape switches to the static
    unified K/V path (P3b: slower than de-dup, faster than the stock projections; validated on its own) instead of running at stock speed; if P3b is rejected too
    the shape runs the stock projections without building any reduced input (~stock memory).  Outputs are exact in
    every case; `patches.kv_mode_report(model)` shows which path each shape ended on.
    on_dedup_reject: "p3b" (default) | "stock" | "ladder" (legacy: keep padding the dedup GEMM up to stock rows).
    graph=True (P6): batch shapes up to graphs.GRAPH_MAX_TOKENS tokens are served from one whole-forward CUDA graph per shape (captured
    at the shape's first forward after a clean eager one, on the static unified-K/V route; larger shapes keep the eager de-dup route)."""
    apply_exact(model, 3)
    patch_unified_kv(model, dedup=dedup and cuda_graph_batch is None, min_rows=min_rows, on_dedup_reject=on_dedup_reject)
    patch_col_attention_layout(model)  # P5 colattn: K/V built once in the attention matmuls' layout (same cuBLAS calls as stock's SDPA-math)
    patch_fused_col_attention(model)   # P5b fusedattn: on top of colattn, the attention read straight from the distinct K/V rows in cuBLAS's own summation order where the batch shape's first forward proves it equal (else colattn's operands)
    if cuda_graph_batch is not None:
        return GraphRunner(model, cuda_graph_batch)
    if graph:
        patch_graphs(model, **({} if graph_max_tokens is None else {"max_tokens": int(graph_max_tokens)}))
    return model


@torch.no_grad()
def verify_identity(stock_model, exact_fn_or_model, batch: dict, center: int | None = None) -> dict:
    """Run both on `batch` and return identity evidence (bitwise / max|dlogits| / argmax / LLR)."""
    stock_model.eval()
    ref = stock_model(**batch).logits
    if isinstance(exact_fn_or_model, torch.nn.Module):
        with torch.inference_mode():
            out = exact_fn_or_model(**batch).logits
    else:
        out = exact_fn_or_model(batch)
    L = batch["input_ids"].shape[1]
    return compare_logits(out, ref, L // 2 if center is None else center)


def make_exact_static(model, example_batch: dict, caps, *, margin: float = 2.0, min_rows: int = 2048):
    """Exact, CUDA-graph-captured, de-duplicated variant (P4s) for a FIXED batch shape.
    `caps` = per-multi-clade unique-row counts observed on calibration batches with the eager dedup
    path (ExactState.last_unique_counts) -- use the max over >= 16 batches (margin 1.25-1.5) or margin >= 2 with fewer;
    capacity = caps * margin.  After each replay call
    `static_overflow(model, B, L)` -- if True the batch exceeded a capacity and must be re-run eagerly."""
    B, L = example_batch["input_ids"].shape[:2]
    apply_exact(model, 3)
    patch_unified_kv(model, dedup="static", min_rows=min_rows)
    set_static_capacities(model, B, L, caps, margin=margin)
    return GraphRunner(model, example_batch)


def make_fast(model, *, policy: str = "tf32_all", dedup=True, min_rows: int = 2048, cuda_graph_batch: dict | None = None,
              caps=None, margin: float = 1.5, graph: bool = True):
    """FAST mode = exact kit + TF32 tensor-core matmuls under `policy` (see patches.apply_fast_policy).
    NOT bit-identical to stock; the precision contract is stated in the kit README.
    TF32 is scoped to THIS model: a forward pre/post hook on the top-level module sets
    torch.backends.cuda.matmul.allow_tf32 = True on entry and restores the previous value on exit, and the policy pins
    selected sub-modules back to fp32 inside.  The process-wide flag is NOT changed, so other models in the same process
    keep strict fp32 -- except code that runs concurrently (other threads) *during* this model's forward, because the
    backend flag is global state.  `model.gpnstar_fast_undo()` removes all hooks (the model is then the exact kit)."""
    apply_exact(model, 3)
    if cuda_graph_batch is not None and caps is not None:
        B, L = cuda_graph_batch["input_ids"].shape[:2]
        patch_unified_kv(model, dedup="static", min_rows=min_rows)
        set_static_capacities(model, B, L, caps, margin=margin)
    else:
        patch_unified_kv(model, dedup=(dedup and cuda_graph_batch is None), min_rows=min_rows)
    patch_col_attention_layout(model)
    apply_fast_policy(model, policy)
    from .patches import _TF32Scope
    outer = _TF32Scope(True)
    outer_handles = [model.register_forward_pre_hook(outer.pre), model.register_forward_hook(outer.post)]
    # NOTE: pre-hooks registered later run later, so the outer scope is registered on the top-level module while the
    # policy pins live on sub-modules (entered after the outer hook, exited before it) -- nesting is well-formed.

    def undo(_m=model, _hs=outer_handles):
        for h in _hs:
            h.remove()
        for h in getattr(_m, "_fast_handles", []):
            h.remove()
        _m._fast_handles = []
    model.gpnstar_fast_undo = undo
    if cuda_graph_batch is not None:
        return GraphRunner(model, cuda_graph_batch)
    if graph:
        patch_graphs(model)   # installed last: the TF32 scope hooks run around the graphed forward (their flag flips are host-side; the capture bakes the kernels in)
    return model
