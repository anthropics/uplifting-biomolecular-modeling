"""of3_transition — the binding of OpenFold3 0.5.0's SwiGLU transition (AF3 Algorithm 11: LayerNorm -> [Linear a | Linear b] -> swish(a)·b ->
Linear out -> ·mask) to the shared core's ONE transition provider (`opt_core.kernels.transition`; cell table TRANSITION_CELLS.json) by the TIER
WORD `exact` under this engine's statement form.  No kernel, tile table or token floor lives in this kit: the provider's table names, per
card, stack, (c, hidden) and row count, the row the exact tier serves and from which token count its bitwise vouch holds.  This module is the
exact line's `transition_exact` lever (cells/transition_exact.py names it; hooks/cells/sitecustomize.py installs it after
`openfold3.core.model.layers.transition` executes).

Form: OpenFold3 0.5.0's `primitives.activations.SwiGLU.forward(x, use_kernel=False)` runs `self.swish(self.linear_a(x)) * self.linear_b(x)` —
silu rounded to bf16, then the product rounded to bf16: the provider's plain form (`form='swiglu'`).

Per call the binding asks the provider, once per (c, hidden, row class, device, eager|graph), what the exact tier serves on THIS process's
stack (`select('exact', form=FORM, ln_given=True)`).  A kernel row — `v1` = kernels/fpf_transition fed the module's own LayerNorm output: a|b
one bf16 MMA chain over K = c with fp32 accumulation rounded to bf16 (the cuBLAS bf16 GEMM's bits), h in the form above, out accumulated over
ascending hidden chunks (cuBLAS's K order); bitwise equal to the statement on the stacks and from the token counts the table lists
— is launched through `transition(x, W, word='exact', x_ln=…, form=FORM)`.  Where the
table's exact-tier winner is the stock statement (small row counts; c 384, where Triton pads K to 512), where this stack or row count carries
no bitwise vouch, or where the shape has no cell (the MSA transition's 64×256), the call runs the module's OWN statement, counted by the
provider's word — never a substitute composition, never silent.  The row class of a call is its row count M keyed the table's way (an N×N
pair layout has N = isqrt(M); a chunked call counts the chunk's rows), so the stack's token floor applies to the rows the call launches.

Served: eval-mode `SwiGLUTransition` instances whose three projections are bias-free `primitives.linear.Linear` without a precision override,
input on the GPU in bf16 or fp32 under bf16 autocast (the engine's Linear casts the LayerNorm's fp32 output to bf16 under autocast; the
binding hands the provider that bf16 tensor).  Everything else runs the module's own statement, counted: `dtype:<dtype>` (fp32 outside bf16
autocast: the diffusion module's transitions inside upstream's fp32 autocast island), `exact:<provider word>` (no cell | no vouch on this
stack | below this stack's token floor), `exact_winner:<stock row>:c<C>`, `bias`, `precision`, `training`, `cpu`.
"""
from __future__ import annotations

import math
import sys
import threading
from typing import Any, Dict

PREFIX = "[openfold3_ob0-opt/transition_exact]"
WORD = "exact"                                  # the provider's tier word: the measured exact-tier winner of the cell on this stack, bitwise-vouched, else refused by name
FORM = "swiglu"                                 # the provider's form key of this engine's statement: swish(a) * b, two bf16 roundings (OpenFold3 0.5.0, use_kernel=False)
STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": None, "served": 0, "fallback": 0, "reasons": {}, "widths": {},
                         "rows": {}, "cells": {}, "first": None, "word": WORD, "form": FORM, "core": None, "vouch": None}
_LOCK = threading.Lock()
_PACKS: Dict[int, Any] = {}                    # id(module) -> (provider Weights, device, weight versions)
_WINNERS: Dict[Any, Any] = {}                  # (c, hidden, n, device, timing, capture) -> None (serve) | the fallback WORD (a str: a cached exception instance re-raised per call would chain every caller's frame into its traceback and keep their tensors alive)
_MISS = object()


def _log(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


class Fallback(Exception):
    """This call runs the module's own statement; ``reason`` is the census word."""

    def __init__(self, reason: str):
        super().__init__(reason); self.reason = reason


def _count(reason: str) -> None:
    with _LOCK:
        STATE["fallback"] += 1
        STATE["reasons"][reason] = STATE["reasons"].get(reason, 0) + 1


def _autocast_bf16(torch) -> bool:
    """True under CUDA autocast to bf16 (the trunk's `bf16-mixed` region)."""
    if hasattr(torch, "get_autocast_dtype"):                        # torch >= 2.4
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") is torch.bfloat16
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() is torch.bfloat16


def _eligible(module, x):
    """(C, NH) for a call the provider may serve, else raise Fallback(reason)."""
    import torch
    if module.training:
        raise Fallback("training")
    if not (torch.is_tensor(x) and x.is_cuda):
        raise Fallback("cpu")
    if x.dtype is torch.bfloat16:
        pass
    elif x.dtype is torch.float32 and _autocast_bf16(torch):
        pass                                                       # the engine's Linear casts the fp32 LayerNorm output to bf16 under autocast: the binding applies the same cast
    else:
        raise Fallback(f"dtype:{str(x.dtype).replace('torch.', '')}")
    sw = getattr(module, "swiglu", None)
    lins = (getattr(sw, "linear_a", None), getattr(sw, "linear_b", None), getattr(module, "linear_out", None))
    if any(l is None or getattr(l, "weight", None) is None for l in lins):
        raise Fallback("module-variant")
    if any(getattr(l, "bias", None) is not None for l in lins):
        raise Fallback("bias")
    if any(getattr(l, "precision", None) is not None for l in lins):
        raise Fallback("precision")
    C = int(x.shape[-1]); NH = int(lins[0].weight.shape[0])
    if lins[1].weight.shape[0] != NH or tuple(lins[2].weight.shape) != (C, NH) or lins[0].weight.shape[1] != C:
        raise Fallback("module-variant")
    return C, NH


def _packed(module, device):
    """The provider's packed weights (`kernels.transition.pack`), built once per module; rebuilt if a weight was replaced or moved."""
    import torch
    from opt_core.kernels import transition as TP
    wa, wb, wo = module.swiglu.linear_a.weight, module.swiglu.linear_b.weight, module.linear_out.weight
    ver = (wa.data_ptr(), wb.data_ptr(), wo.data_ptr(), wa._version, wb._version, wo._version)
    hit = _PACKS.get(id(module))
    if hit is not None and hit[1] == str(device) and hit[2] == ver:
        return hit[0]
    ln = module.layer_norm
    with torch.no_grad():
        W = TP.pack(w_a=wa.detach(), w_b=wb.detach(), w_o=wo.detach(),
                    ln_w=(ln.weight.detach() if getattr(ln, "weight", None) is not None else None),
                    ln_b=(ln.bias.detach() if getattr(ln, "bias", None) is not None else None),
                    eps=float(getattr(ln, "eps", 1e-5)), device=device)
    _PACKS[id(module)] = (W, str(device), ver)
    return W


def _winner(TP, C: int, NH: int, n: int, device, timing: str, capture: bool) -> None:
    """Serve only where the provider's exact tier names a KERNEL row for this cell on this process's stack (its table: bitwise against the
    statement there, from the stack's token floor up, and not slower than the stock arm); a stock winner (`exact_winner:<row>:c<C>`) or a
    refusal (`exact:<word>`: no cell, no vouch on this stack, below the floor) is this call's route to the module's own statement, by name.
    Resolved once per (c, hidden, n, device, timing, capture)."""
    key = (C, NH, n, str(device), timing, capture)
    v = _WINNERS.get(key, _MISS)
    if v is _MISS:
        try:
            sel = TP.select(WORD, c=C, hidden=NH, n_tokens=n, dtype="bf16", family="pair", device=device, ln_given=True, form=FORM,
                            timing=timing, capture=capture)
        except TP.Refusal as r:
            v = "exact:" + str(getattr(r, "kind", None) or r).split(" ")[0]
        else:
            row = getattr(sel, "row", None)
            if row in getattr(TP, "STOCK_ROWS", ("torch_swiglu", "engine_module", "compile")):
                v = f"exact_winner:{row}:c{C}"
            else:
                v = None
                STATE["vouch"] = str(getattr(sel, "stack", None) or "none")
        _WINNERS[key] = v
    if v is not None:
        raise Fallback(v)                                          # a fresh instance per call (see _WINNERS)


def fused_update(module, x):
    """The transition's UPDATE u = Linear_out(silu(a)·b) [..., c] bf16 for x [..., c] (the mask and the residual are the caller's): the module's own
    LayerNorm, then the provider's exact-tier row.  Raises Fallback(reason) for a call the tier does not serve here (nothing computed yet)."""
    import torch
    from opt_core.kernels import transition as TP
    C, NH = _eligible(module, x)
    M = x.numel() // C
    if M < 1:
        raise Fallback("rows:0")
    n = max(1, math.isqrt(M))                                      # the table's size key: token count N of an N x N pair layout with these rows
    capture = bool(torch.cuda.is_current_stream_capturing())
    timing = "graph" if capture else "eager"
    _winner(TP, C, NH, n, x.device, timing, capture)
    xl = module.layer_norm(x)                                      # the engine's own LayerNorm call (castcache serves its casts on the exact line)
    y_ln = xl if xl.dtype is torch.bfloat16 else xl.to(torch.bfloat16)
    if not y_ln.is_contiguous():
        y_ln = y_ln.contiguous()
    W = _packed(module, x.device)
    try:
        u, sel = TP.transition(y_ln, W, word=WORD, x_ln=y_ln, n_tokens=n, family="pair", timing=timing, capture=capture, form=FORM)
    except TP.Refusal as r:                                        # by name: `kind` is the blank-free machine word; the fallback row is the module's statement here
        raise Fallback("exact:" + str(getattr(r, "kind", None) or r).split(" ")[0]) from None
    if STATE["first"] is None:
        STATE["first"] = f"{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"
    with _LOCK:
        STATE["served"] += 1
        STATE["widths"][f"c{C}"] = STATE["widths"].get(f"c{C}", 0) + 1
        arm = (getattr(sel, "row", None) or "?") + ((":" + sel.variant) if getattr(sel, "variant", None) else "")
        STATE["rows"][arm] = STATE["rows"].get(arm, 0) + 1
        ck = getattr(sel, "cell_key", None)
        if ck:
            STATE["cells"][str(ck)] = STATE["cells"].get(str(ck), 0) + 1
    want = tuple(x.shape[:-1]) + (C,)
    return u if tuple(u.shape) == want else u.view(want)


def install(transition_module) -> Dict[str, Any]:
    """Patch ``SwiGLUTransition._transition`` class-wide (the chunked path calls it per chunk): served calls run the provider's exact-tier row,
    every other call the module's own statement, counted.  Idempotent."""
    cls = getattr(transition_module, "SwiGLUTransition", None)
    if cls is None or not hasattr(cls, "_transition"):
        STATE.update(state="refused", reason="no-SwiGLUTransition._transition")
        _log(f"REFUSED: {transition_module.__name__} has no SwiGLUTransition._transition — the transitions run the engine's statement")
        return STATE
    if getattr(cls._transition, "_of3_transition_exact", False):
        return STATE
    try:
        from opt_core.kernels import transition as TP    # noqa: F401 — the provider (the kit's core pin gate names an older core before this point)
        import opt_core
        STATE["core"] = getattr(opt_core, "__version__", "?")
        if FORM not in getattr(TP, "FORMS", ()):
            raise ImportError(f"form {FORM} not in kernels.transition.FORMS")
    except Exception as e:
        STATE.update(state="refused", reason=f"no-provider:{type(e).__name__}")
        _log(f"REFUSED: opt_core.kernels.transition not usable ({type(e).__name__}: {e}) — the transitions run the engine's statement")
        return STATE
    stock = cls._transition

    def _transition(self, x, mask):
        if STATE["state"] != "on":
            return stock(self, x, mask)
        try:
            u = fused_update(self, x)
        except Fallback as f:
            _count(f.reason)
            return stock(self, x, mask)
        return u * mask                                            # the module's own epilogue statement (`x = x * mask`), same promotion

    _transition._of3_transition_exact = True
    _transition.__wrapped__ = stock
    cls._transition = _transition
    STATE.update(installed=True, state="on", reason=None)
    _log(f"installed: SwiGLUTransition._transition class-wide -> the core's transition provider (opt_core {STATE['core']} kernels.transition) by the tier word "
         f"{WORD} under form {FORM}, the module's own LayerNorm feeding it; cells the tier does not serve on this stack / fp32 calls run the module's statement, counted")
    return STATE


def census() -> str:
    """` word=exact form=<form> core=… vouch=<stack|none> served=<n> fallback=<n>[ fallback:<reason>=<n>…] rows=<arm>:<n>,…|none cells=<cell>:<n>,…|none widths=c<C>:<n>,…|none first=…`."""
    rs = "".join(f" fallback:{k}={v}" for k, v in sorted(STATE["reasons"].items()))
    rows = ",".join(f"{k}:{v}" for k, v in sorted(STATE["rows"].items())) or "none"
    cells = ",".join(f"{k}:{v}" for k, v in sorted(STATE["cells"].items())) or "none"
    widths = ",".join(f"{k}:{v}" for k, v in sorted(STATE["widths"].items())) or "none"
    return (f" word={WORD} form={FORM} core={STATE['core']} vouch={STATE.get('vouch') or 'none'} served={STATE['served']} fallback={STATE['fallback']}{rs}"
            f" rows={rows} cells={cells} widths={widths} first={STATE['first'] or 'none'}")
