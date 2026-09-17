"""odde_triattn_bind -- OpenDDE's triangle-attention site served by the core's triangle-attention provider (opt_core.kernels.triattn).

The ARM-T add-on (odde_arm_t) replaces upstream's ``layers.cuequivariance_triangular_attn(q, k, v, bias, mask, scale)`` with its own wrapper
(design gate, module scope, dtype / rank checks). With this unit's word set, the wrapper hands every admitted call to ``serve()`` below: the TIER
WORD goes straight to the provider's ``select()``, which names the row per (compute capability, dtype, head dim, heads, row count, call form) cell
of its measured table and carries that cell's launch setting; a refusal serves the provider's NAMED fallback row; a cell (or fallback) that names a
stock row is served by the stock op (``Aside``: the wrapper counts the reason and calls upstream's function). No kit-side cell table, launch setting
or row preference lives here -- the row is the provider's decision, stated per cell on the LEVER line. Nothing under site-packages is edited.

One switch, read once at import:

  ODDE_TRIATTN=fast    the provider's fast tier (line LSTAR2A).
  ODDE_TRIATTN=exact   the provider's exact tier (line S1): rows bitwise to the stock cuEquivariance op only, else the stock op by the cell's name; the
                       provider's vouch gate is live (select(exact_stack=<cc|torch|cueq key of this process>): an exact row serves only on a stack the
                       table records it byte-vouched on, else `unavailable=<row>:exact_vouch_not_recorded:<key>` and the stock op serves).
  ODDE_TRIATTN=big   the provider's memory tier (the big lines: fast's resolution minus rows with a recorded peak above the stock op's).
                       On a core without the word the fast tier serves, stated (word_served=fast).
  ODDE_TRIATTN=<row>   pin one provider row by name (a user knob; kit modes export tier words only); a refusal serves its named fallback row.
  unset / 0 / off      inert: the arm's attention site keeps the stock op (attention=stock).
  ODDE_TRIATTN_CONF=stock  ablation only (MODEL_OPT_LEVERS_OFF=triattn_conf; no line exports it): the confidence head's 4-block pair stack keeps the
                       stock op, counted `asides=conf_stock`; absent = the tier word serves that stack like every other pair stack (lever triattn_conf on).

Census: ``COUNTS`` (calls per row, per cell line, asides by reason, refusals / exclusions by name) -- ``describe()`` is the LEVER line's evidence;
``opendde_opt.report`` prints it.
"""
from __future__ import annotations

import os
import re

__version__ = "0.2"

_RAW = os.environ.get("ODDE_TRIATTN", "").strip()
WORD = None if _RAW.lower() in ("", "0", "off", "none", "stock") else _RAW
CONF = "stock" if os.environ.get("ODDE_TRIATTN_CONF", "").strip().lower() == "stock" else None   # ablation of lever triattn_conf only (see above)
MAX_HOPS = 4                # named-fallback hops per call before the call is handed to the stock op (asides 'exhausted')

COUNTS: dict = {"version": __version__, "word": WORD, "active": WORD is not None, "word_served": None, "exact_stack": None, "calls": 0, "rows": {}, "cells": {}, "stock_calls": 0,
                "asides": {}, "refusals": {}, "excluded": [], "unavailable": {}, "errors": {}, "cuda_prebuilt": None, "shapes": {},
                "bias_forms": {}, "stack_key": None, "passed": {}, "forms": {}}
_REDIRECT: dict = {}        # row -> the provider's named fallback row, for rows that refused or failed AT CALL TIME in this process (COUNTS['excluded'] / ['refusals'])
_SEL: dict = {}             # (dtype, D, H, S, form, n_redirects) -> (provider Selection | None, aside reason | None)
_FACTS: dict = {}           # device facts read once: cc tuple, cc word, cuda_sm90a stack key, the word served, form= support


class Aside(Exception):
    """The call is the stock op's by design (the cell's winner is a stock row, an unsupported form, or every row stepped aside); .reason names it."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def active() -> bool:
    return WORD is not None


def _bump(key, sub=None, n=1):
    if sub is None:
        COUNTS[key] = int(COUNTS.get(key) or 0) + n
    else:
        d = COUNTS.setdefault(key, {})
        d[sub] = int(d.get(sub) or 0) + n


def _facts(T=None):
    if _FACTS:
        return _FACTS
    import torch
    cc = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0)
    _FACTS["cc"] = cc
    _FACTS["cc_word"] = f"{cc[0]}.{cc[1]}"
    stack = None
    if cc == (9, 0):
        try:
            from opt_core.kernels.triattn import cuda_sm90a as C
            stack = C.stack_key()
            from opt_core.kernels import triattn as T_
            COUNTS["cuda_prebuilt"] = stack in T_.stacks_built()
        except Exception as e:  # noqa: BLE001
            COUNTS["errors"]["stack_key"] = repr(e)[:200]
    _FACTS["stack"] = stack
    COUNTS["stack_key"] = stack
    xst = None                                                                                  # the provider's exact-vouch key of this process ('<cc>|torch<v>|cueq<v>'): an exact-class
    if T is not None and hasattr(T, "exact_stack_key"):                                          # row serves only on a stack the table records it byte-vouched on, else the tier passes it
        try:                                                                                    # over BY NAME (exact_vouch_not_recorded:<key>) and the stock op serves -- the gate is the provider's
            xst = T.exact_stack_key(cc)
        except Exception as e:  # noqa: BLE001
            COUNTS["errors"]["exact_stack"] = repr(e)[:200]
    _FACTS["exact_stack"] = xst
    COUNTS["exact_stack"] = xst
    word = WORD
    if T is not None and word == "big" and "big" not in tuple(getattr(T, "TIER_WORDS", ())):   # a core older than the memory tier word: fast's tier serves, stated
        word = "fast"
    _FACTS["word"] = word
    COUNTS["word_served"] = word
    return _FACTS


def _dtype_word(dt) -> str:
    import torch
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dt, str(dt).replace("torch.", ""))


def _form_kw(T, m5) -> dict:
    """The call-form hint the provider face reads where a cell measured the forms apart (``form=``): OpenDDE's site passes upstream's per-row key
    mask with the pair bias on every call (``mask_bias``; ``bias_only`` when a caller passes no mask)."""
    if "form_kw" not in _FACTS:
        import inspect
        try:
            ok = "form" in inspect.signature(T.triangle_attention).parameters
        except (TypeError, ValueError):
            ok = False
        _FACTS["form_kw"] = ok
    if not _FACTS["form_kw"]:
        return {}
    form = "bias_only" if m5 is None else "mask_bias"
    _bump("forms", form)
    return {"form": form}


def _why_not(T, f, row, dtype, D, H, S) -> str:
    """The provider's admission verdict for ``row`` on this call, as one census token (observation only)."""
    try:
        r = T.admits(row, f["cc"], dtype, int(D), int(H), int(S), stack=f.get("stack"))
        ok, why = (r[0], r[1]) if isinstance(r, tuple) else (bool(r), "")
        return "admitted" if ok else ("_".join(str(why).split(":")[0].split()))[:40]
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def _row_fallback(T, row):
    """The provider's named fallback of ``row`` (its rows block), or the stock op."""
    try:
        fb = (T.rows().get(str(row).split("@")[0]) or {}).get("fallback")
    except Exception:  # noqa: BLE001
        fb = None
    return fb or "cueq"


def _select(T, f, dtype, D, H, S, form=None):
    """(Selection, None) = the provider's row for the word at this cell; (None, reason) = the stock op serves it, by the named reason."""
    key = (dtype, D, H, S, form, len(_REDIRECT))
    if key in _SEL:
        return _SEL[key]
    word = f["word"]
    fkw = {"form": form} if (form and _FACTS.get("form_kw")) else {}
    if f.get("exact_stack"):
        fkw["exact_stack"] = f["exact_stack"]
    hops, sel, why = 0, None, None
    while True:
        try:
            sel = T.select(f["cc_word"], dtype, D, H, S, word=word, stack=f["stack"], **fkw)
            for row_, key_ in re.findall(r"([a-z0-9_]+):exact_vouch_not_recorded:([^'\]\s,]+)", str(getattr(sel, "reason", "") or "")):
                _bump("refusals", f"{row_}:exact_vouch_not_recorded")      # the provider's vouch gate: this exact row is not recorded byte-vouched on this stack -> passed over, the stock op by name
                COUNTS["unavailable"][row_] = f"exact_vouch_not_recorded:{key_}"
        except T.Refusal as r:                                  # the word's row (or every candidate of the tier) cannot serve here: its NAMED fallback row, or the stock op
            _bump("refusals", f"{r.row}:{str(r.kind).split(':')[0] if str(r.kind).startswith('exact_vouch_not_recorded') else r.kind}")
            if str(r.kind).startswith("exact_vouch_not_recorded"):
                COUNTS["unavailable"][str(r.row).split("@")[0]] = str(r.kind)[:120]
            fb = getattr(r, "fallback", None)
            if fb and fb in T.ROW_NAMES and fb not in T.STOCK_ROWS and hops < MAX_HOPS:
                word = fb; hops += 1; continue
            sel, why = None, ("cell_stock" if not _REDIRECT else "cell_stock_after_exclusions")
            break
        if sel.row in T.STOCK_ROWS:                             # the cell names a stock row for this word: the stock op, by the cell's rule
            sel, why = None, ("cell_stock" if not (_REDIRECT or hops) else "cell_stock_after_exclusions")
            break
        if sel.row in _REDIRECT:                                # the row refused / failed at call time earlier in this process: the provider's named fallback for it
            fb = _REDIRECT[sel.row]
            if fb in T.ROW_NAMES and fb not in T.STOCK_ROWS and hops < MAX_HOPS:
                word = fb; hops += 1; continue
            sel, why = None, "cell_stock_after_exclusions"
            break
        break
    if f["word"] in getattr(T, "TIER_WORDS", ("fast", "exact")):   # observation: the cell's named winner for the tier was passed over (refused by name at select, excluded, ...)
        try:
            _k, cell, _m, _n = T.cell_for(f["cc_word"], dtype, D, H, S)
            col = "fast" if f["word"] == "big" and "big" not in (cell or {}) else f["word"]
            win = str((cell or {}).get(col) or "").split("@")[0]
            if win and win not in T.STOCK_ROWS and (sel is None or win != str(sel.row).split("@")[0]):
                whyn = "excluded" if win in _REDIRECT else _why_not(T, f, win, dtype, D, H, S)
                _bump("passed", f"{win}@{str(_k).split('|')[4] if _k and str(_k).count('|') >= 4 else S}:{whyn}")
        except Exception:  # noqa: BLE001
            pass
    _SEL[key] = (sel, why)
    return _SEL[key]


def _exclude(T, row, kind, fallback=None):
    row = str(row).split("@")[0]
    _REDIRECT[row] = fallback or _row_fallback(T, row)
    COUNTS["excluded"] = sorted(_REDIRECT)
    _bump("refusals", f"{row}:{kind}")


def serve(q5, k5, v5, b5, m5, scale, *, stock5):
    """Serve one admitted call of the arm's attention wrapper: q5/k5/v5 [B,N,H,S,D] (the compute dtype already applied), b5 [B,1,H,S,S] fp32,
    m5 [B,N,1,1,S] bool | None, scale float; ``stock5`` = the stock op on these 5-D tensors (cuequivariance signature, returns the tensor) for the
    rows that call it per head.  Returns the output [B,N,H,S,D] in q5.dtype, or raises ``Aside(reason)``: the caller serves the stock op and counts the reason."""
    from opt_core.kernels import triattn as T
    from opt_core.oom import is_oom
    f = _facts(T)
    B, N, H, S_q, D = (int(x) for x in q5.shape)
    S = int(k5.shape[-2])
    if b5.dim() == 5 and b5.shape[1] != 1:                     # upstream's chunk_layer expands the one [H,I,J] triangle bias over the chunk's rows:
        if b5.stride(1) == 0:                                   # a zero-stride view is the same map for every row -- take it once, no copy
            b5 = b5[:, :1]; _bump("bias_forms", "rows_expanded_view")
        else:
            _bump("bias_forms", "per_row"); raise Aside("bias_per_row")
    if S_q != S:
        raise Aside("sq_ne_sk")
    dtype = _dtype_word(q5.dtype)
    _bump("shapes", f"{dtype}|H{H}|D{D}")
    fk = _form_kw(T, m5); form = fk.get("form")
    for _hop in range(MAX_HOPS):
        sel, why = _select(T, f, dtype, D, H, S, form)
        if sel is None:
            raise Aside(why or "cell_stock")
        try:
            out = T.triangle_attention(q5, k5, v5, b5, m5, scale=scale, word=sel.row, stock=stock5, selection=sel, **fk)
        except T.Refusal as r:                                  # the row refused at call time (an install gate, driver bindings absent, ...): unavailable BY NAME, its named fallback serves
            _exclude(T, sel.row, r.kind, getattr(r, "fallback", None))
            COUNTS["unavailable"][str(sel.row).split("@")[0]] = str(r.kind)[:120]
            continue
        except Exception as e:  # noqa: BLE001 -- OOM is the caller's; anything else excludes the row by name for this process and its named fallback serves
            if is_oom(e):
                raise
            _exclude(T, sel.row, type(e).__name__)
            if isinstance(e, (ImportError, OSError)):              # the row cannot run on this stack (driver bindings / compiler library absent): unavailable BY NAME
                COUNTS["unavailable"][str(sel.row).split("@")[0]] = f"{type(e).__name__}:{str(e)[:100]}"
            else:                                                   # a row that failed while running: a defect, named in errors (the lever is then never 'inert by aside')
                COUNTS["errors"][str(sel.row).split("@")[0]] = repr(e)[:300]
            continue
        _bump("calls"); _bump("rows", sel.row); _bump("cells", T.describe(sel))
        return out
    raise Aside("exhausted")


def describe() -> dict:
    """The LEVER line's evidence: word (and the word served), calls per row, asides, refusals, excluded rows, the distinct cell lines."""
    return {"word": WORD, "word_served": COUNTS.get("word_served"), "conf": CONF, "calls": int(COUNTS.get("calls") or 0), "rows": dict(COUNTS.get("rows") or {}),
            "stock_calls": int(COUNTS.get("stock_calls") or 0), "asides": dict(COUNTS.get("asides") or {}), "refusals": dict(COUNTS.get("refusals") or {}),
            "excluded": list(COUNTS.get("excluded") or []), "unavailable": dict(COUNTS.get("unavailable") or {}),
            "errors": {k: str(v)[:160] for k, v in (COUNTS.get("errors") or {}).items()}, "cells": dict(COUNTS.get("cells") or {}),
            "cuda_prebuilt": COUNTS.get("cuda_prebuilt"), "stack_key": COUNTS.get("stack_key"), "exact_stack": COUNTS.get("exact_stack"), "shapes": dict(COUNTS.get("shapes") or {}),
            "bias_forms": dict(COUNTS.get("bias_forms") or {}), "passed": dict(COUNTS.get("passed") or {}), "forms": dict(COUNTS.get("forms") or {})}


def note_aside(reason: str):
    """The arm's wrapper records a call it handed to the stock op on this unit's account (reason from ``Aside``)."""
    _bump("stock_calls"); _bump("asides", reason)


def reset_counts():
    for k in ("calls", "stock_calls"):
        COUNTS[k] = 0
    for k in ("rows", "cells", "asides", "refusals", "shapes", "bias_forms", "passed", "forms"):
        COUNTS[k] = {}
