"""odde_transition_bind -- OpenDDE's pair-transition sites bound to the core's transition provider (opt_core.kernels.transition) by TIER WORD.

The site: ``Transition.forward`` of every ``pair_transition`` module the trunk arm binds (c_in 384, n 4 -> hidden 1536; the 48-block
pairformer, the MSA-module pair stacks, the template pairformer and, under ODDE_ARM_T_SCOPE, the structural refiner and the confidence
head's stack), reached through the kit's transition adapter (third_party/fpf_transition_odde, applied by odde_arm_t.bind under this unit's
word).  Per call the adapter asks ``decide()``: the provider's ``select(<word>)`` for the cell (compute capability, bf16, pair_c384_n4,
N bucket) --

  * a CARRIED row (one of the provider's fused rows) is served through the provider face ``transition()`` chunk by chunk (``serve_chunk``);
  * a STOCK row (``torch_swiglu`` = the statement, ``engine_module`` = the caller's own module) or a refusal by name (``no_winner`` above
    the table's sizes, ``no_cell``) hands the call to the kit's single-engine
    composition of the SAME statement, ``sep16`` (fpf_transition_odde: the module's LayerNorm cast to bf16 once, the two cuBLAS projections,
    one fused SiLU*gate elementwise kernel with the statement's two rounding points, the output projection written in place; bitwise to the
    engine's module under bf16 autocast at the op on both cards) -- NAMED on the LEVER line (``singleton=sep16:<n>``), counted, never silent.

One switch, read once at import:

  ODDE_TRANSITION=exact    the provider's exact tier (line S1): a row serves only where the cell records it bitwise to the statement on this
                           stack; a cell that names the statement itself -> sep16 by name.
  ODDE_TRANSITION=fast     the provider's fast tier (line LSTAR2A): the cell's measured winner (a stock-row winner -> sep16 by name).
  ODDE_TRANSITION=big    the provider's big tier (the memory mode's lines).
  ODDE_TRANSITION=<row>    pin one provider row by name (<row> | <row>@<cfg>, the provider's row names); a refusal is an aside by name.
  unset / 0 / off          inert: the adapter is not applied; the engine's module runs.

Census: ``COUNTS`` / ``describe()`` -- the LEVER line's evidence (word, calls, provider rows served, singleton calls, cells, asides, refusals,
excluded rows); ``opendde_opt.report`` prints it.  Nothing under site-packages is edited.
"""
from __future__ import annotations

import os

__version__ = "0.1"

_RAW = os.environ.get("ODDE_TRANSITION", "").strip()
WORD = None if _RAW.lower() in ("", "0", "off", "none", "stock") else _RAW
TIER_WORDS = ("fast", "exact", "big")          # the provider's tier words this kit's lines export (opt_core.kernels.transition TIER_WORDS also has 'faithful')
SINGLETON = "sep16"                              # the kit's single-engine composition of the statement (third_party/fpf_transition_odde), by name
CELL = (384, 1536)                               # OpenDDE's pair transition: c_z 384, n 4
FAMILY = "pair"

COUNTS: dict = {"version": __version__, "word": WORD, "active": WORD is not None, "calls": 0, "prov_calls": 0, "rows": {}, "singleton": {},
                "cells": {}, "asides": {}, "refusals": {}, "excluded": [], "errors": {}, "stack": None, "decisions": 0}
_EXCLUDED: set = set()      # provider rows that raised at call time in this process: never served again (COUNTS['excluded'] / ['errors'] name them)
_MEMO: dict = {}            # (cc, N, zdtype) -> Decision: the provider's pure selection asked once per call class (the census records it there)
_STACK: dict = {}


class Aside(Exception):
    """The call (or the rest of it) is the singleton's by a named rule; .reason names it."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Decision(object):
    """What ``decide`` resolved for a call class: kind 'provider' (``sel`` = the provider's Selection, ``row``) or 'singleton' (``reason``)."""
    __slots__ = ("kind", "reason", "sel", "row", "cell", "n_tokens")

    def __init__(self, kind, reason=None, sel=None, row=None, cell=None, n_tokens=None):
        self.kind, self.reason, self.sel, self.row, self.cell, self.n_tokens = kind, reason, sel, row, cell, n_tokens

    def aside(self, reason):
        return Decision("singleton", reason=reason, sel=self.sel, row=self.row, cell=self.cell, n_tokens=self.n_tokens)


def active() -> bool:
    return WORD is not None


def _bump(key, sub=None, n=1):
    if sub is None:
        COUNTS[key] = int(COUNTS.get(key) or 0) + n
    else:
        d = COUNTS.setdefault(key, {})
        d[sub] = int(d.get(sub) or 0) + n


def _cc(device):
    import torch
    return "%d.%d" % torch.cuda.get_device_capability(device)


def decide(module, x) -> Decision:
    """The provider's answer for this call's class (memoized per (cc, N, z dtype)): a carried row -> Decision('provider'); a stock row, a
    refusal by name or an excluded row -> Decision('singleton', reason).  ``x`` = the call's input [.., N, N, c] (fp32 master under bf16
    autocast, or bf16); the cell's dtype axis is the statement's compute dtype under autocast (bf16)."""
    from opt_core.kernels import transition as T
    N = int(x.shape[-2]) if x.dim() >= 2 else int(x.shape[0])
    key = (str(x.device), N, str(x.dtype))
    d = _MEMO.get(key)
    if d is not None and (d.kind != "provider" or d.row not in _EXCLUDED):
        return d
    _bump("decisions")
    c, hidden = int(module.c_in), int(module.linear_no_bias.weight.shape[1])
    try:
        if "word" not in _STACK:
            _STACK["word"] = T.stack_word(x.device); COUNTS["stack"] = _STACK["word"]
        sel = T.select(WORD, c=c, hidden=hidden, n_tokens=N, dtype="bf16", family=FAMILY, device=x.device, stack=_STACK["word"],
                       rows_count=int(x.numel() // c), ln_given=True)
    except T.Refusal as r:
        _bump("refusals", f"{r.row}:{r.kind}"[:120])
        d = Decision("singleton", reason=f"refused:{str(r.kind).split(':')[0]}"[:80], n_tokens=N)
    else:
        cell = sel.cell_key or f"c{c}_h{hidden}|N={N}|no-cell"
        _bump("cells", f"{cell}->{sel.row}" + ((":" + sel.variant) if sel.variant else ""))
        if sel.row in T.STOCK_ROWS:
            d = Decision("singleton", reason=f"cell_stock:{sel.row}", sel=sel, row=sel.row, cell=cell, n_tokens=N)
        elif sel.row in _EXCLUDED:
            d = Decision("singleton", reason=f"excluded:{sel.row}", sel=sel, row=sel.row, cell=cell, n_tokens=N)
        else:
            d = Decision("provider", sel=sel, row=sel.row, cell=cell, n_tokens=N)
    _MEMO[key] = d
    return d


def weights(module, device):
    """The provider's packed weights for this module (cached on the module; rebuilt when the parameters are replaced)."""
    from opt_core.kernels import transition as T
    wa, wb, wo = module.linear_no_bias_a.weight, module.linear_no_bias_b.weight, module.linear_no_bias.weight
    key = (wa.data_ptr(), wb.data_ptr(), wo.data_ptr(), wa._version, wb._version, wo._version, str(device))
    c = getattr(module, "_odde_transition_W", None)
    if c is None or c[0] != key:
        ln = module.layernorm1
        W = T.pack(w_a=wa, w_b=wb, w_o=wo, ln_w=getattr(ln, "weight", None), ln_b=getattr(ln, "bias", None), eps=float(getattr(ln, "eps", 1e-5)), device=device)
        module._odde_transition_W = c = (key, W)
    return c[1]


def serve_chunk(module, chunk, y16, dec: Decision):
    """Serve one row chunk through the provider row ``dec`` resolved: ``chunk`` [rows, c] the pre-LayerNorm rows (the call's dtype), ``y16``
    [rows, c] bf16 = the module's own LayerNorm output cast once (rows that take the caller's LayerNorm read it; rows that carry their own
    normalise ``chunk``).  Returns the bf16 update [rows, c] or raises ``Aside`` (a refusal with the tensors in hand, or a row that cannot
    run here: excluded by name for the process)."""
    import torch
    from opt_core.kernels import transition as T
    from opt_core.oom import is_oom
    W = weights(module, chunk.device)
    x16 = chunk if chunk.dtype == torch.bfloat16 else chunk.to(torch.bfloat16)
    try:
        out, sel = T.transition(x16, W, word=WORD, x_ln=y16, n_tokens=dec.n_tokens, family=FAMILY, stack=_STACK.get("word"))
    except T.Refusal as r:
        _bump("refusals", f"{r.row}:{r.kind}"[:120]); raise Aside(f"refused:{str(r.kind).split(':')[0]}"[:80])
    except Exception as e:  # noqa: BLE001 -- a row that cannot run on this stack is excluded BY NAME for the process; OOM is the caller's
        if is_oom(e):
            raise
        row = dec.row or "?"
        _EXCLUDED.add(row); COUNTS["excluded"] = sorted(_EXCLUDED); COUNTS["errors"][row] = repr(e)[:300]
        _bump("refusals", f"{row}:error:{type(e).__name__}")
        raise Aside(f"error:{row}:{type(e).__name__}")
    if sel.row in T.STOCK_ROWS:                                   # a word that resolved to the statement inside the face: never served here (the singleton is that statement, leaner)
        raise Aside(f"cell_stock:{sel.row}")
    _bump("prov_chunks"); _bump("rows_chunks", sel.row)
    return out


def note_call(dec: Decision, served_by: str):
    """The adapter records one transition call decided under the word: ``served_by`` = a provider row name or SINGLETON."""
    _bump("calls")
    if served_by == SINGLETON:
        _bump("singleton", SINGLETON); _bump("asides", dec.reason or "unknown")
    else:
        _bump("prov_calls"); _bump("rows", served_by)


def describe() -> dict:
    """The LEVER line's evidence: word, calls decided under it, provider rows served, singleton calls, the distinct cell decisions, asides,
    refusals, excluded rows."""
    return {"word": WORD, "calls": int(COUNTS.get("calls") or 0), "prov_calls": int(COUNTS.get("prov_calls") or 0), "rows": dict(COUNTS.get("rows") or {}),
            "singleton": dict(COUNTS.get("singleton") or {}), "cells": dict(COUNTS.get("cells") or {}), "asides": dict(COUNTS.get("asides") or {}),
            "refusals": dict(COUNTS.get("refusals") or {}), "excluded": list(COUNTS.get("excluded") or []), "errors": dict(COUNTS.get("errors") or {}),
            "stack": COUNTS.get("stack"), "decisions": int(COUNTS.get("decisions") or 0)}


def reset_counts():
    for k in ("calls", "prov_calls", "decisions"):
        COUNTS[k] = 0
    for k in ("rows", "singleton", "cells", "asides", "refusals"):
        COUNTS[k] = {}
    _MEMO.clear()
