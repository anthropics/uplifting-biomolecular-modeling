"""The ``big`` mode — the kit's memory line for inputs past what ``fast`` fits: ``fast``'s own lever set recomposed for headroom plus
memory levers composed on the shared core's primitives (``opt_core.mem``), ONE mode on the fast base. Slower than ``fast`` where
``fast`` fits is expected and stated; the point is the token count that completes on one GPU.

Levers, in ``LEVER_ORDER`` — one composition, all of them in force on every ``big`` pred (the ACTIVE line's ``levers=``, the BIG line and
the LEVER lines record it):

  graph_drop           setting    NO CUDA graph in this mode, at any size: the kit graph lever ``stepgraph`` (the whole-denoiser-step CUDA
                                  graph of ``fast``) is dropped from the BUILD and the part it keeps without a graph stays (``hoist``: the
                                  step-invariant conditioning computed once per trajectory, the step eager on it) -- ``GRAPH_DROP_MIN_TOKENS
                                  = 0``. A graph's private memory pool is memory this mode does not spend, whatever the size; the price is
                                  speed where the batched step is launch-bound (small inputs); at sizes where the step is kernel-bound the
                                  graph's replay saves nothing. A positive ``GRAPH_DROP_MIN_TOKENS`` is the per-item
                                  size gate this setting also supports (the graph built and replayed below the gate, switched off per item at or
                                  above it: ``forward.py``); ``None`` = never dropped. Class: the same kernels either way -- outputs of the
                                  levers that remain.
  diff_free            setting    the diffusion statics (the hoist caches, a whole-step graph if any, the kernel kit's pools) released the
                                  moment the sampler returns, before the confidence / distogram heads (``forward._reset_item_state``, the
                                  kit's own eviction sequence). Class: frees only — bitwise vs the mode without it by construction.
  expandable_segments  allocator  ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` through ``opt_core.mem.torch_alloc`` (exported in the
                                  model process before torch initialises CUDA, declared ``graphs_on`` / ``allow_with_graphs`` while the step
                                  graph is built: below graph_drop's gate the graph's private pool is memory the policy cannot return,
                                  composed on purpose at sizes where peak is not the constraint; the graph captures and replays beside the
                                  policy on torch 2.13; the allocator's own record read back as the evidence). Class: placement only — bitwise by construction.
  prev_free            setting    the recycle's fp32 prev embeddings (pair ``[N, N, 128]``, single) released the moment the Evoformer has
                                  embedded them, instead of living beside the new pair until the pass returns (``forward_impl._run_trunk_prev_free``:
                                  the kit api's recycle loop with a consume-once prev). Class: frees only — bitwise by construction.

Every lever prints one ``[af3-torch-opt] LEVER name=F7.<lever> state=on|off|skipped …`` line per pred with the model process's own
record as evidence (``report.lever_line``); ``opt_core.mem`` absent from the imported core is a refusal of the mode by name.

``paircond_chunk``: the sampler's step-invariant pair conditioning (``DiffusionHead._pair_conditioning``: the relative-encoding one-hots, the
[N, N, 267] concat, LayerNorm, projection and two transitions — about 5 kB of transients per token pair when whole, the statement that runs out
of memory first as inputs grow) evaluated per block of ``PAIRCOND_CHUNK_ROWS`` rows into one preallocated result (``xfold/nn/paircond_rows.py``;
one-hots written in the pair dtype, ``True * pair`` read in place). Tolerance-class (a GEMM's leading size changes per block).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

BASE_SET = "fastest"                     # the kit lever set big recomposes (modes.MODE_SETS['big'])
LEVER_ORDER: Tuple[str, ...] = ("graph_drop", "diff_free", "expandable_segments", "prev_free", "paircond_chunk")
PAIRCOND_CHUNK_ROWS = 256                # paircond_chunk: rows of the token grid per block of the sampler's step-invariant pair conditioning (an input at or
                                         # below it runs whole, as stock); the block's transients are rows x N x ~5 kB
DROP: Dict[str, Tuple[str, ...]] = {"stepgraph": ("hoist",)}     # kit graph lever graph_drop switches off (per item, size-gated) -> the non-graph part it keeps
ALLOC_POLICY = "expandable"              # opt_core.mem.torch_alloc policy under expandable_segments
GRAPH_DROP_MIN_TOKENS: Optional[int] = 0      # graph_drop: 0 = the graph levers dropped from the build (this mode runs NO CUDA graph at any size: a graph's private
                                              # pool is memory the memory line does not spend; costs forward 3.884 -> 4.956 s per item at 448 tokens on H100, nothing
                                              # measurable at 832 / 1216); a positive padded token count = the per-item size gate (graph below it, eager on the hoist at
                                              # or above it: forward.py --graph-drop-tokens); None = never dropped.

def mem_present() -> Tuple[bool, Optional[str]]:
    """Whether the imported core carries ``opt_core.mem`` (else the refusal text)."""
    try:
        import opt_core.mem  # noqa: F401
        from opt_core.mem import torch_alloc  # noqa: F401
    except Exception as e:                     # noqa: BLE001
        return False, f"mode 'big' needs opt_core.mem (the shared core's memory primitives), which the imported core does not carry ({type(e).__name__}: {e})"
    return True, None


def selection() -> dict:
    """The memory composition of ``big``: every lever of LEVER_ORDER in force — {'levers': [...in LEVER_ORDER], 'settings': {...}}
    (``graph_drop`` size-gated: settings['graph_drop_min_tokens'])."""
    return {"levers": list(LEVER_ORDER), "settings": {"graph_drop_min_tokens": GRAPH_DROP_MIN_TOKENS, "paircond_rows": PAIRCOND_CHUNK_ROWS}}


def graph_drop_always(sel_or_settings: Optional[Mapping]) -> bool:
    """Whether graph_drop drops the graph levers from the BUILD (gate 0: every item) rather than per item."""
    st = (sel_or_settings or {})
    st = st.get("settings", st) if isinstance(st, Mapping) else {}
    return st.get("graph_drop_min_tokens", GRAPH_DROP_MIN_TOKENS) == 0


def graph_dropped_for(bucket: Optional[int], min_tokens: Optional[int]) -> bool:
    """graph_drop's per-item decision: the step graph is switched off for an item whose padded token count is at or above the gate."""
    return min_tokens is not None and bucket is not None and int(bucket) >= int(min_tokens)


def build_levers(base: Sequence[str], levers: Sequence[str], settings: Optional[Mapping] = None) -> Tuple[str, ...]:
    """The kit lever tuple the model is built with: the base set — its graph levers replaced by what they keep only when graph_drop's gate is 0
    (every item); under the size gate the graph levers stay built and forward.py switches the step graph off per item."""
    if "graph_drop" not in levers or not graph_drop_always(settings if settings is not None else selection()):
        return tuple(base)
    out: List[str] = []
    for l in base:
        if l in DROP:
            out.extend(k for k in DROP[l] if k not in out and k not in base)
        else:
            out.append(l)
    return tuple(out)


def dropped(base: Sequence[str], levers: Sequence[str], settings: Optional[Mapping] = None) -> List[str]:
    """The graph levers absent from the BUILD (gate 0 only; under the size gate nothing is dropped from the build)."""
    return [l for l in base if l in DROP] if ("graph_drop" in levers and graph_drop_always(settings if settings is not None else selection())) else []


def forward_argv(sel: Optional[dict]) -> List[str]:
    """What big adds to forward.py's argv: the levers the MODEL process acts on (graph_drop's gate, diff_free, prev_free; the allocator policy)."""
    if not sel:
        return []
    argv = ["--big", ",".join(sel["levers"])]
    if "expandable_segments" in sel["levers"]:
        argv += ["--alloc", ALLOC_POLICY]
    gate = (sel.get("settings") or {}).get("graph_drop_min_tokens")
    if "graph_drop" in sel["levers"] and gate:
        argv += ["--graph-drop-tokens", str(int(gate))]
    if "paircond_chunk" in sel["levers"]:
        argv += ["--paircond-rows", str(int((sel.get("settings") or {}).get("paircond_rows") or PAIRCOND_CHUNK_ROWS))]
    return argv


def lever_state(lever: str, rep: Mapping) -> Tuple[str, Optional[str], dict]:
    """(state, reason, evidence) of one big lever in one pred's records: `off` outside big; `on` with the model process's record
    (forward.json `big`); `skipped` when selected but the record does not show it."""
    sel = rep.get("big")
    if not sel:
        return "off", "not_in_arm_lever_set", {}
    if lever not in sel.get("levers", ()):
        return "off", "not_selected", {}
    rec = rep.get("big_record") or {}                 # the model process's record (forward.json 'big'); {} when the forward did not start
    if lever == "graph_drop":
        built = rep.get("levers_applied")
        if built is None:
            return "skipped", "no_levers_applied_record(forward.json)", {}
        gate = (sel.get("settings") or {}).get("graph_drop_min_tokens", GRAPH_DROP_MIN_TOKENS)
        if gate == 0:                                     # dropped from the build: the graph levers must be absent from what the model was built with
            bad = [l for l in DROP if l in built]
            if bad:
                return "skipped", f"graph_levers_still_built:{','.join(bad)}", {}
            return "on", None, {"dropped": [l for l in DROP if l not in built], "kept": [k for v in DROP.values() for k in v if k in built], "gate_tokens": 0}
        n, items = int(rec.get("graph_drop_items") or 0), int(rec.get("items") or 0)
        if gate is None:
            return "skipped", "size_gate:none", {}
        # in force = the gate armed in the model process and evaluated per item: items_dropped of items ran the step eager on the hoist (0 when no
        # item of this pred reached the gate — the step graph replayed for every item, which is the lever's decision below the gate, not a skip)
        return "on", None, {"gate_tokens": gate, "items_dropped": n, "items": items}
    if lever == "diff_free":
        n = rec.get("diff_free_calls")
        if not n:
            if rec and not rec.get("items_ok", 1):
                return "skipped", "no_item_completed_its_sampler", {}
            return "skipped", "no_release_recorded(forward.json:big.diff_free_calls)", {}
        return "on", None, {"calls": n}
    if lever == "paircond_chunk":
        if not rec.get("paircond_rows"):
            return "skipped", "no_rows_recorded(forward.json:big.paircond_rows)", {}
        return "on", None, {"rows": rec.get("paircond_rows"), "items_chunked": rec.get("paircond_items", 0), "blocks": rec.get("paircond_blocks", 0), "items": rec.get("items", 0)}
    if lever == "prev_free":
        n = rec.get("prev_free_calls")
        if not n:
            if rec and not rec.get("items_ok", 1):
                return "skipped", "no_item_completed_its_trunk", {}
            return "skipped", "no_release_recorded(forward.json:big.prev_free_calls)", {}
        return "on", None, {"calls": n}
    if lever == "expandable_segments":
        a = rec.get("alloc") or {}
        if a.get("effective") is not True:
            return "skipped", f"alloc_record_not_expandable(effective={a.get('effective')},source={a.get('source')})", {}
        return "on", None, {"alloc": a.get("policy"), "alloc_conf": a.get("conf"), "effective": 1}
    return "skipped", "unknown_lever", {}
