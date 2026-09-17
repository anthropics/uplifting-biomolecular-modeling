"""Lever exactln — ``atlasfold.model.network.primitives.normalization:LayerNorm.forward`` (normalization.py L31-43) bound BY TIER WORD to the core's
LayerNorm provider ``opt_core.kernels.ln``: the word is the mode's tier — ``exact`` in the exact row, ``fast`` in the fast row, ``big`` in the
big row (``AFO_EXACTLN_WORD`` names another word: a tier word, or a provider row word ``exactln`` / ``fastln`` / ``ln_rows`` / ... — the
operator's explicit choice, printed ``pinned=1``).

Stock computes ``F.layer_norm(x.float(), (C,), w.float(), b.float(), eps).to(x.dtype)``.  The three forms that statement takes in this model
(fp32 x / fp32 parameters; bf16 x / bf16 parameters — the lm_stack / main_stack modules load_model casts —; bf16 x / fp32 parameters with the
``.to(bfloat16)`` on the result: the provider's fp32 | bf16 | bf16-out widened forms) are each resolved by the provider per CALL CLASS (cc, form,
C, the call's ROW count = numel / C — the quantity the provider's cells are keyed by —, alignment, CUDA-graph capture state, and for the
exact word the running torch / triton / cuEquivariance stack): the measured cell's winner for the tier.  ``exact`` resolves the carried bit-for-bit
ATen replica ``exactln`` only where the provider records it BITWISE on THIS stack, else ATen BY NAME; ``fast`` / ``big`` resolve the cell's
tolerance-class winner (``fastln``, ``ln_rows``, ``exactln``, ... or ATen).  The resolution is cached per call class (one provider decision per
class per process; the provider's census records it once); nothing is chosen here — no kit-side row floor, no row word by name.

A call whose resolved row is a STOCK row (``aten`` / ``aten_autocast`` / a named extension nothing carries) runs the statement BELOW this wrapper
(``ln_bf16``'s when that lever is in the row — this lever is installed after it — else stock), counted ``stock_row:<row>``: the module's own
statement, never a re-statement.  Other named step-asides to the statement below: ``disabled`` (``AFO_EXACTLN=0``), ``cpu``, ``dtype:<t>`` (fp16 /
fp64 activations), ``param_form`` (parameters of two dtypes, or fp32 x with bf16 parameters), ``normalized_shape`` (a multi-axis norm),
``no_affine:<row>`` (a no-affine norm — the diffusion module's AdaLN inner norms — resolved to a row whose no-affine form this kit has not run:
only exactln / fastln / dtk_ln and the stock rows take gamma = beta = None here), ``refused:<kind>`` (the provider's Refusal of THIS call's form: a
width its row does not serve, a mis-aligned view, ...), ``no_core:<reason>`` (the provider absent).  A row that RAISES is counted ``error:<Type>``
for that call (the statement below serves it; the gate refuses at exit).  Fields: word= pinned= rows=<row:n> plan=<form/C/cell:row | stock:row>.
``MODEL_OPT_LEVERS_OFF=exactln`` removes the lever from the row before activation.  Class: exact under the exact word (bitwise rows or ATen by
name); tolerance under fast / big."""
from __future__ import annotations

import os

from . import Installed, graph_context, rebind

LEVER = "exactln"
NAME = "LOCAL.atlasfold.exactln"
ENV = "AFO_EXACTLN"
WORD_ENV = "AFO_EXACTLN_WORD"
TIER_OF_MODE = {"exact": "exact", "fast": "fast", "big": "big"}
TARGET = "atlasfold.model.network.primitives.normalization"
NO_AFFINE_ROWS = ("exactln", "fastln", "dtk_ln")                       # carried rows this kit runs with gamma = beta = None (exactln: the diffusion module's AdaLN inner norms; fastln / dtk_ln take None by construction)
EXPECTED = ("disabled", "cpu", "dtype:", "param_form", "normalized_shape", "stock_row:", "no_affine:", "refused:", "no_core:")
_STATE = {"override": None}                    # None = read AFO_EXACTLN; True / False = the scratch A/B switch


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def word(mode: str = "exact") -> str:
    """AFO_EXACTLN_WORD, else the mode's tier word (`exact` | `fast` | `big`)."""
    return (os.environ.get(WORD_ENV) or "").strip() or TIER_OF_MODE.get(mode, "exact")


def bench_arm(label: str) -> None:
    """A/B switch for developer timing scripts (never a kit switch): arm 'A' = the statement below (lever inert), any other label = the provider."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"exactln": enabled()}   # type: ignore[attr-defined]


def provider():
    """(module, version, refusal) — opt_core.kernels.ln, its version word, or (None, None, '<reason>') when it does not import."""
    try:
        from opt_core.kernels import ln as LN
        import opt_core
        return LN, str(getattr(opt_core, "__version__", "?")), None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}:{str(e)[:40]}".replace(" ", "_")


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    import torch
    from opt_core.counters import Ledger
    try:
        nm = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{TARGET}:{type(e).__name__}")
    cls = nm.LayerNorm
    below = cls.forward                                                       # ln_bf16's wrapper when that lever is in the row (installed before this one), else stock
    LN, version, why_not = provider()
    w = word(mode)
    if LN is not None:
        row = LN.split_word(w)[0]
        if w not in LN.TIER_WORDS and row not in LN.ROW_NAMES:
            return Installed(LEVER, False, reason=f"unknown_word:{w}")
        if row in LN.STOCK_ROWS:
            return Installed(LEVER, False, reason=f"word_names_no_kernel:{w}")
    pinned = LN is not None and w not in LN.TIER_WORDS
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    ledger = Ledger(NAME, impl=(f"opt_core.kernels.ln@{version}:{w}" if LN is not None else f"aside({why_not})"), origin="core", expected=words)
    ledger.set("word", w); ledger.set("pinned", int(pinned)); ledger.set("rows", "none"); ledger.set("plan", "none")
    if LN is None:
        ledger.word("no_core", why_not)
    bf16, f32 = torch.bfloat16, torch.float32
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        has_triton = False
    dev = {}                                                                 # device index -> (cc word, stack word for the exact word | None)
    cache = {}                                                               # call class -> ("serve", Selection, arm) | ("below", reason)
    rows_served, plan = {}, {}

    def device_facts(x):
        i = x.device.index if x.device.index is not None else torch.cuda.current_device()
        got = dev.get(i)
        if got is None:
            cc = "%d.%d" % torch.cuda.get_device_capability(x.device)
            st = LN.stack_word(x.device) if LN.split_word(w)[0] == "exact" else None   # the exact vouch is per stack; tolerance words need no stack
            got = dev[i] = (cc, st)
        return got

    def resolve(x, C, rows, w_, b_, capture):
        """One provider decision per call class, cached: ("serve", Selection, arm) or ("below", <named reason>)."""
        xdt = x.dtype
        pdt = w_.dtype if w_ is not None else (b_.dtype if b_ is not None else None)
        widen = (xdt == bf16 and w_ is not None and w_.dtype == f32)
        aligned = LN._rows_aligned(x, C) if hasattr(LN, "_rows_aligned") else True
        key = (xdt, pdt, w_ is None and b_ is None, C, rows, aligned, bool(capture))
        got = cache.get(key)
        if got is not None:
            return got
        cc, st = device_facts(x)
        n = int(round(rows ** 0.5))
        cw = LN.cell_for_rows(rows, n, C)
        form = "bf16o" if widen else str(xdt).replace("torch.", "").replace("bfloat16", "bf16").replace("float32", "fp32")
        try:
            sel = LN.select(cc, xdt, cw, n, word=w, widen=widen, out=("bf16" if widen else None), capture=bool(capture), C=C, has_triton=has_triton,
                            aligned=aligned, stack=st, rows=rows)
        except LN.Refusal as e:
            got = ("below", "refused:" + str(e.kind).split(":")[0].split("(")[0].replace(" ", "_")[:40])
        except ValueError as e:                                              # an unknown word reaching select (install checks first): named
            got = ("below", "refused:" + str(e)[:24].replace(" ", "_"))
        else:
            arm = LN.arm_word(sel.row, sel.variant)
            if sel.row in LN.STOCK_ROWS:
                got = ("below", f"stock_row:{sel.row}")
            elif key[2] and sel.row not in NO_AFFINE_ROWS:
                got = ("below", f"no_affine:{sel.row}")
            else:
                got = ("serve", sel, arm)
        cache[key] = got
        cellw = str(cw) if cw else "none"
        plan[f"{form}/C{C}/{cellw}"] = (got[2] if got[0] == "serve" else got[1].replace("stock_row:", "stock:"))
        ledger.set("plan", ",".join(f"{k}:{v}" for k, v in sorted(plan.items())[:12]))
        return got

    def forward(self, x):
        if not enabled():
            ledger.fallback("disabled")
            return below(self, x)
        if LN is None:
            ledger.fallback("no_core:" + str(why_not)[:24])
            return below(self, x)
        if not x.is_cuda:
            ledger.fallback("cpu")
            return below(self, x)
        ns = self.normalized_shape
        C = int(ns[-1]) if isinstance(ns, (tuple, list)) else int(ns)
        if (isinstance(ns, (tuple, list)) and len(ns) != 1) or x.dim() < 1 or int(x.shape[-1]) != C or C == 0:
            ledger.fallback("normalized_shape")
            return below(self, x)
        if x.dtype not in (bf16, f32):
            ledger.fallback("dtype:" + str(x.dtype).replace("torch.", ""))
            return below(self, x)
        wt, bs = self.weight, self.bias
        pdt = {p.dtype for p in (wt, bs) if p is not None}
        if len(pdt) > 1 or (pdt and x.dtype == f32 and next(iter(pdt)) != f32) or (pdt and next(iter(pdt)) not in (bf16, f32)):
            ledger.fallback("param_form")
            return below(self, x)
        rows = x.numel() // C
        capture = graph_context(x)
        got = resolve(x, C, rows, wt, bs, capture)
        if got[0] == "below":
            ledger.fallback(got[1])
            return below(self, x)
        sel, arm = got[1], got[2]
        widen = (x.dtype == bf16 and wt is not None and wt.dtype == f32)
        try:
            y, _ = LN.layer_norm(x, (C,), wt, bs, self.eps, word=w, selection=sel, out_dtype=(bf16 if widen else None), capture=capture)
        except LN.Refusal as e:                                              # the provider's named refusal of THIS call's form -> the statement below, by name
            ledger.fallback("refused:" + str(e.kind).split(":")[0].split("(")[0].replace(" ", "_")[:40])
            return below(self, x)
        except Exception as e:  # noqa: BLE001 — a row that raised: this call takes the statement below; the gate refuses at exit
            ledger.error(e)
            return below(self, x)
        if y.dtype != x.dtype:                                               # the statement returns x.dtype; a row answering the widened form in fp32 is cast as the statement casts
            y = y.to(x.dtype)
        rows_served[arm] = rows_served.get(arm, 0) + 1
        ledger.set("rows", ",".join(f"{k}:{v}" for k, v in sorted(rows_served.items())))
        ledger.serve(f"C{C}:{arm}")
        return y
    forward.__qualname__ = "LayerNorm.forward[atlasfold_opt:exactln]"
    rebind(cls, "forward", forward, below)
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[_gate(ledger, words)],
                     facts={"impl": getattr(ledger, "impl", None), "env": os.environ.get(ENV, "1"), "word": w, "pinned": pinned, "provider": version, "refusal": why_not, "ledger": ledger})


def _gate(ledger, words):
    """The lever's exit gate: every fallback reason must equal or start with a registry word (parametric reasons `stock_row:<row>`, `refused:<kind>`,
    `no_affine:<row>`, `dtype:<t>`, `no_core:<reason>` match by prefix); a row error refuses; a run whose calls ALL took a named route below is a
    legitimate stock run (ok)."""
    from . import pair_cells as PC
    return PC.gate_for(ledger, words)
