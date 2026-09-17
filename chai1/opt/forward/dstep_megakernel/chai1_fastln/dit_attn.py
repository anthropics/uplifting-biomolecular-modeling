"""dit_attn — lever `dit_attn` (FAST class): the diffusion transformer's 16 pair-biased token attentions per denoiser step, served through
opt_core's pair-bias-attention provider (``opt_core.kernels.apb``) instead of SDPA's MATH path.

The transpiled denoiser calls ``torch.scaled_dot_product_attention(q, k, v, attn_mask)`` with 5-D fp32 operands — q/k/v ``[1, H=16, S, N, 48]``
(S diffusion samples batched), attn_mask a fp32 additive bias ``[1, 16, 1, N, N]`` shared by the S samples. No fused SDPA kernel takes 5-D inputs, so
stock runs the MATH path: the ``[S, 16, N, N]`` logits are materialised (bias add, masked fill, softmax, two GEMMs). This lever rebinds
``scaled_dot_product_attention`` inside the denoiser's transpiled namespace (an instance attribute on its TorchShim; other components untouched): a call of exactly that form is viewed as ``q[0].transpose(0, 1)`` = ``[S, H, N, D]`` and a
``[1, H, N, N]`` bias (views, no copies) and served by ``kernels.apb.pair_bias_attention`` asked by the MODE'S TIER WORD (``fast`` in fast,
``big`` in big; exact does not carry this lever): the provider serves the row its cell table names for (card, fp32, cell ``dit_h16d48``, the
item's token count, S samples, graphed) — one selection per token count per process (``_row_for``, a constant inside the compiled step). Where
the tier word resolves to a row that IS the library statement (``STATEMENT_ROWS``: the provider's sdpa rows), the statement itself serves, by name
(inside the compiled step Inductor fuses it); the row word and the counts are recorded (``report()``), never silent. Every other SDPA call of the step (the atom transformer's windowed
attention) takes the original statement.

Class: FAST, not exact — stock's statement is the math path; the memory-efficient kernel (and its replica) reorder the softmax reductions
(within the TF32 numerics class of the fast tier). Under the `compiled` lever the
served call is one opaque custom op (``torch.ops.chai1_fastln.dit_attn``) in Dynamo's graph: Inductor does not trace into the provider.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

SURFACE = ("WORD", "STATS", "STATEMENT_ROWS", "available", "step_aside_word", "admits", "_op", "_row_for", "row_decision", "_serve4", "_stock4", "_apb", "Router", "shim_of", "patch_flat", "unpatch_flat", "report")   # the names the kit / this package's callers reach (chai1_opt tests/test_lever_surfaces.py checks they exist, statically)

NAME = "dit_attn"
WORD: Optional[str] = None                              # the provider's TIER word this process asks by: the mode's own ("fast" | "big"), set at patch_flat
TIER_WORDS = ("fast", "big", "exact")
STATEMENT_ROWS = ("sdpa", "sdpa_upcast", "sdpa_gather", "naive", "naive_module", "torch_module", "torch_dit_rows")   # provider rows that are the library / statement op itself
CELL = "dit_h16d48"
STATS: Dict[str, Any] = {"installed": False, "word": None, "stepped_aside": None, "served_by": None, "refusal": None, "calls": 0, "served": 0, "statement": 0,
                         "rows": {}, "original": {}, "errors": 0}
_STATE: Dict[str, Any] = {"selections": {}, "rows": {}, "apb": None, "cc": None}


def _apb():
    if _STATE["apb"] is None:
        from opt_core.kernels import apb as A          # the shared core's pair-bias-attention provider (the kit's install pins a core that carries it)
        _STATE["apb"] = A
    return _STATE["apb"]


def _stock4(q4, k4, v4, b4):
    """The statement's own arithmetic (SDPA math path on the 5-D operands) from the 4-D views — what a call the provider refuses at launch gets."""
    import torch.nn.functional as F
    o = F.scaled_dot_product_attention(q4.transpose(0, 1).unsqueeze(0), k4.transpose(0, 1).unsqueeze(0), v4.transpose(0, 1).unsqueeze(0), b4.transpose(0, 1).unsqueeze(0))
    return o[0].transpose(0, 1).contiguous()


def _row_for_impl(n_tokens: int, samples: int, head_dim: int, heads: int) -> str:
    """The provider's row word for this call class under WORD (one ``kernels.apb.select`` per (tokens, samples) per process, cached with its
    Selection for the launches); a Refusal is cached as ``refused:<kind>`` (the statement serves, counted by that name)."""
    key = (int(n_tokens), int(samples))
    row = _STATE["rows"].get(key)
    if row is not None:
        return row
    A = _apb()
    cc = _STATE.get("cc") or (tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0))
    _STATE["cc"] = cc
    try:
        sel = A.select(tuple(cc), torch.float32, CELL, int(n_tokens), word=WORD or "fast", samples=int(samples), timing="graphed", capture=True,
                       head_dim=int(head_dim), heads=int(heads))
        row = str(getattr(sel, "row", None) or "unknown")
        _STATE["selections"][key] = sel
    except A.Refusal as r:                                  # no row for this call class under the word: the statement serves, named
        row = "refused:" + (str(getattr(r, "kind", None) or (r.args[0] if r.args else "refused")).split(":")[0].split(" ")[0] or "refused")
        STATS["refusal"] = f"{WORD}:{getattr(r, 'kind', None) or (r.args[0] if r.args else 'refused')}"[:160]
    _STATE["rows"][key] = row
    STATS["rows"].setdefault(row, 0)
    if STATS["served_by"] is None or (STATS["served_by"] in STATEMENT_ROWS and row not in STATEMENT_ROWS and not row.startswith("refused:")):
        STATS["served_by"] = row                           # the EXIT word names the provider row (the first kernel row met, else the statement row)
    return row


try:                                                        # inside the compiled step the row decision is a trace-time constant (evaluated once in Python)
    from torch._dynamo import assume_constant_result as _const
    _row_for = _const(_row_for_impl)
except Exception:                                           # noqa: BLE001 — torch without dynamo: the plain function
    _row_for = _row_for_impl


def row_decision(row: str) -> str:
    """``provider`` (launch the provider's row through the opaque op) | ``statement`` (the row IS the library statement, or refused: the original call)."""
    return "statement" if (row in STATEMENT_ROWS or row.startswith("refused:") or row == "unknown") else "provider"


def _serve4(q4, k4, v4, b4):
    """[S,H,N,D] fp32 views + [1,H,N,N] fp32 bias -> [S,H,N,D] fp32 (contiguous) through the provider row selected for this token count (``_row_for``
    ran first, outside the op). A launch-time Refusal (shape / stride outside the row's route) takes the statement's own arithmetic, counted."""
    A = _apb()
    S, H, N, D = int(q4.shape[0]), int(q4.shape[1]), int(q4.shape[2]), int(q4.shape[3])
    sel = _STATE["selections"].get((N, S))
    try:
        if sel is None:                                     # not selected yet in this process (an eager call before any Router decision): select now
            _row_for_impl(N, S, D, H); sel = _STATE["selections"].get((N, S))
        if sel is None:
            return _stock4(q4, k4, v4, b4)
        out, _ = A.pair_bias_attention(q4, k4, v4, b4, selection=sel, layout="shnd", cell=CELL, capture=True)
    except A.Refusal as r:
        STATS["route_refused"] = STATS.get("route_refused", 0) + 1; STATS["refusal"] = f"{getattr(sel, 'row', WORD)}:{r.args[0] if r.args else 'refused'}"[:160]
        return _stock4(q4, k4, v4, b4)
    return out.contiguous()


def step_aside_word(kind: str, cc=None) -> str:
    """The EXIT / LEVER word when the provider cannot select ANY row for the cell under the tier word at install: ``no_apb_row_<kind head>`` —
    the Refusal kind's first token (``cc``, ``dtype``, ``head_dim``, ``word`` …), ``cc`` carrying the card's digits (``no_apb_row_cc80``)."""
    head = (str(kind or "refused").split(":")[0].split(" ")[0].strip() or "refused")
    if head == "cc" and cc is not None:
        head = "cc%d%d" % (int(cc[0]), int(cc[1]))
    return "no_apb_row_" + head


def available(cc=None, apb=None, word=None) -> Optional[str]:
    """None when the provider selects a row for (cc, fp32, cell dit_h16d48, S=5, graphed, capture) under the tier word at the smallest and the
    largest crop (256, 2048) — a kernel row or its statement row, either serves — else a NAMED reason word from the Refusal kind, decided once at
    install, before any fold (the lever then steps aside BY NAME: every call keeps its statement; ``report()`` / the EXIT tally say why).
    Anything but a Refusal is unexpected and propagates (the activation refuses loudly by name)."""
    A = apb if apb is not None else _apb()
    w = word or WORD or "fast"
    if cc is None:
        cc = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0)
    try:
        for n in (256, 2048):
            A.select(tuple(cc), torch.float32, CELL, n, word=w, samples=5, timing="graphed", capture=True, head_dim=48, heads=16)
    except A.Refusal as r:
        return step_aside_word(getattr(r, "kind", None) or str(r), cc)
    return None

# One opaque op for torch.compile: Dynamo/Inductor keep it as an extern call (no tracing into the provider / its prebuilt kernel). Registered once
# per process at first use (idempotent: a second load of this file in one process reuses the registered op).
_OP = None


def _op():
    global _OP
    if _OP is None:
        try:
            @torch.library.custom_op("chai1_fastln::dit_attn", mutates_args=())
            def _dit_attn_op(q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor, b4: torch.Tensor) -> torch.Tensor:
                return _serve4(q4, k4, v4, b4)

            @_dit_attn_op.register_fake
            def _(q4, k4, v4, b4):
                return torch.empty(q4.shape, dtype=q4.dtype, device=q4.device)
        except RuntimeError:                                    # already registered in this process
            pass
        _OP = torch.ops.chai1_fastln.dit_attn
    return _OP


def admits(q, k, v, attn_mask, dropout_p, is_causal, scale) -> Optional[str]:
    """None if the call is the DiT token attention this lever serves, else the reason it takes the original statement (counted)."""
    if not (torch.is_tensor(attn_mask) and q.dim() == 5 and k.dim() == 5 and v.dim() == 5 and attn_mask.dim() == 5):
        return "not_5d"
    if q.shape[0] != 1 or q.dtype != torch.float32 or attn_mask.dtype != torch.float32:
        return "dtype_or_batch"
    if k.shape != q.shape or v.shape != q.shape or q.shape[-1] not in (24, 48):
        return "shape"
    H, S, N = q.shape[1], q.shape[2], q.shape[3]
    if tuple(attn_mask.shape) != (1, H, 1, N, N):
        return "bias_form"
    if dropout_p or is_causal or scale is not None:
        return "sdpa_args"
    if not q.is_cuda:
        return "device"
    return None


class Router:
    """The denoiser namespace's ``scaled_dot_product_attention``: serves the DiT token form, passes everything else to the original."""

    def __init__(self, original, aside=None):
        self.original = original
        self.aside = aside                                      # reason the lever stepped aside in this process (no provider row selects here), or None

    def __call__(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        counting = not torch.compiler.is_compiling()            # under torch.compile the counters stay out of the trace (a guard on a changing Python int
        if counting:                                            # would recompile the step at every call); the eager / graphed paths count every call
            STATS["calls"] += 1
        why = ("stepped_aside" if self.aside else None) or admits(q, k, v, attn_mask, dropout_p, is_causal, scale)
        if why is not None:
            if counting:
                STATS["original"][why] = STATS["original"].get(why, 0) + 1
            return self.original(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)
        row = _row_for(int(q.shape[3]), int(q.shape[2]), int(q.shape[4]), int(q.shape[1]))  # the provider's row for (tokens, samples) under WORD — a constant in the trace
        if row_decision(row) == "statement":                                                  # the row IS the library statement (or refused): the original call, fused by Inductor
            if counting:
                STATS["statement"] += 1; STATS["rows"][row] = STATS["rows"].get(row, 0) + 1
            return self.original(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)
        q4, k4, v4 = q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1)        # [S, H, N, D] views
        b4 = attn_mask[0].transpose(0, 1)                                                     # [1, H, N, N] view (shared by the S samples)
        out = _op()(q4, k4, v4, b4)                                                             # [S, H, N, D] (one opaque op under torch.compile)
        if counting:
            STATS["served"] += 1; STATS["rows"][row] = STATS["rows"].get(row, 0) + 1
        return out.transpose(0, 1).unsqueeze(0)                                               # [1, H, S, N, D]: the statement's own layout (a view)


def shim_of(flat):
    return object.__getattribute__(flat, "_rt").globals["torch"]


def patch_flat(flat, word: str = "fast"):
    """Rebind scaled_dot_product_attention inside `flat`'s transpiled namespace (instance attribute on its TorchShim; other components untouched);
    ``word`` = the mode's tier word the provider is asked by."""
    global WORD
    if word not in TIER_WORDS:
        raise ValueError(f"dit_attn: tier word {word!r} is not one of {TIER_WORDS}")
    WORD = word
    _STATE["selections"].clear(); _STATE["rows"].clear()
    shim = shim_of(flat)
    aside = available(word=word)
    r = Router(type(shim).scaled_dot_product_attention, aside=aside)
    shim.__dict__["scaled_dot_product_attention"] = r
    STATS.update(installed=True, word=word, stepped_aside=aside)
    return r


def unpatch_flat(flat):
    shim = shim_of(flat)
    shim.__dict__.pop("scaled_dot_product_attention", None)
    STATS["installed"] = False


def report() -> Dict[str, Any]:
    return dict(unit=NAME, **{k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()})
