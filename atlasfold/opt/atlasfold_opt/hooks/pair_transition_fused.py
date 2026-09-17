"""Lever pair_transition — the fused SwiGLU transition of atlasfold's pair track (stock primitives/transition.py: ``Transition(channel, n)`` =
nn.Sequential(LayerNorm(C), SwiGLU(LinearNoBias C -> 2*n*C; chunk order a | b, out = silu(a) * b: activation.py L15), LinearNoBias n*C -> C);
``forward(self, x)`` returns the UPDATE the caller adds: block.py L281 ``z = z + transition_z(z)``) bound BY TIER WORD to the core's transition
provider ``opt_core.kernels.transition``.

Word: the mode's tier word — ``fast`` in the fast row, ``big`` in the big row (``AFO_PAIR_TRANSITION_WORD`` names another word: a tier word or
a provider row word ``v2`` / ``v2:fast`` / ``v1:lnfused`` / ... — the operator's explicit choice, printed ``pinned=1``).  The provider
resolves the word per call class (cc, dtype, c, hidden, n_tokens): the measured cell's winner for the tier (a kernel row or the statement — the
provider's tables choose, nothing is chosen here); ``opt_core.kernels.transition.transition(x, W, word=…, residual=…)`` serves it (the weights packed
once per module by the provider's ``pack``, cached on the module).  Served: the bf16 pair transitions at c_z = 128, factor 4 (LM stack, Pairformer;
the confidence pair stack under the runners' bf16 autocast) at every size the provider's cells serve a kernel row.  A non-contiguous x is served
through a counted copy (copies=<n>), never a silent one.
ORDER: this lever installs AFTER conf_transition_chunk / pair_transition_chunk / transition_exact (modes.MODES row order), which route the SAME
attribute — so the stock callable captured here IS the exact-class binding over the row-block wrapper: this binding is outermost and every call it
declines runs the statement beneath (the exact lever's construction, the chunked statement or plain stock), counted on each census.
Fallbacks, counted by name: `training`, `dtype:<t>` (bf16 only: fp32 inputs — the diffusion PairConditioning transitions, or the confidence pair
stack without autocast), `rank`, `c:<n>` (C != 128: the single track's 384, the LM stack's 768), `factor:<n>` (n != 4: the diffusion module's
factor-2 transitions), `cpu`, `stock_row:<cell>` (the provider's cell names the statement for this size), `refused:<kind>` (ANY provider Refusal —
no cell, a row outside its envelope: the statement beneath runs, by name).  A kernel that raises is counted `error:<Type>` for that call (the gate
refuses at exit).  Fields: word= rows=<row:n> plan=<C/N:row/class | stock | refused:kind> pinned= copies=.  Class: fast (tolerance-class rows)."""
import os
from typing import Optional

from . import Installed, rebind
from . import transition_chunk as TC
from . import pair_cells as PC

LEVER = "pair_transition"
NAME = "LOCAL.atlasfold.pair_transition"
TARGET = "atlasfold.model.network.primitives.transition"
WORD_ENV = "AFO_PAIR_TRANSITION_WORD"
TIER_OF_MODE = {"fast": "fast", "big": "big"}                     # the mode's tier word (any other mode installing this lever: `fast`)


def word(mode: str = "fast") -> str:
    """AFO_PAIR_TRANSITION_WORD, else the mode's tier word (`fast` | `big`)."""
    return (os.environ.get(WORD_ENV) or "").strip() or TIER_OF_MODE.get(mode, "fast")


def kt_weights(module):
    """The module's parameters packed once for the provider (opt_core.kernels.transition.pack), cached on the module (the exact-class lever beneath
    shares the same cache attribute: one pack per module)."""
    W = getattr(module, "_afo_kt_transition_w", None)
    if W is None:
        from opt_core.kernels import transition as KT
        for name, lin in (("swiglu", module[1].linear), ("out", module[2])):
            if getattr(lin, "bias", None) is not None:
                raise KT.Refusal("bias:" + name, None, "torch_swiglu")
        ln = module[0]
        W = KT.pack(w_o=module[2].weight, w_ab=module[1].linear.weight, ln_w=ln.weight, ln_b=ln.bias, eps=float(ln.eps), device=module[2].weight.device)
        module._afo_kt_transition_w = W
    return W


STATE = {"ledger": None, "word": None, "rows": {}, "plan": {}}                # this lever's ledger once installed (pair_block_residual counts the transitions it runs with residual=True on it)


def _note_plan(C, N, text):
    plan = STATE.setdefault("plan", {})
    key = f"{C}/{N}"
    if plan.get(key) != text:
        plan[key] = text
        L = STATE.get("ledger")
        if L is not None:
            L.set("plan", ",".join(f"{k}:{v}" for k, v in sorted(plan.items())))


def fused(module, x, residual: bool):
    """ONE fused transition of x by the lever's word through opt_core.kernels.transition.transition(x, W, word=..., residual=...) (the kernel rows
    fold the residual in their epilogue).  Raises opt_core.attn.pair_fused.Unsupported (the vocabulary both callers catch:
    this lever and hooks/pair_block_residual) carrying `stock_row:<cell>` (the statement is the provider's answer) or `refused:<kind>` (a named Refusal)."""
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import transition as KT
    w = STATE.get("word") or "fast"
    N, C = int(x.shape[-2]), int(x.shape[-1])
    try:
        y, sel = KT.transition(x, kt_weights(module), word=w, residual=residual, n_tokens=N, family="pair", timing="eager")
    except KT.Refusal as e:
        reason = "refused:" + str(e.kind).split(":")[0]
        _note_plan(C, N, reason)
        raise PF.Unsupported(reason)
    arm = sel.row + (f":{sel.variant}" if sel.variant else "")
    if sel.row in KT.STOCK_ROWS:                                             # the provider names the statement for this cell: the statement BENEATH runs (never the provider's re-statement)
        cellw = (sel.cell_key or "").split("|")
        reason = "stock_row:" + (("n" + cellw[3].split("=")[-1]) if len(cellw) > 3 and cellw[3].startswith("N") else f"n{N}")
        _note_plan(C, N, "stock")
        raise PF.Unsupported(reason)
    _note_plan(C, N, f"{arm}/{sel.cls}")
    rows = STATE.setdefault("rows", {})
    rows[arm] = rows.get(arm, 0) + 1
    L = STATE.get("ledger")
    if L is not None:
        L.set("rows", ",".join(f"{k}:{v}" for k, v in sorted(rows.items())))
    return y


def geometry(module):
    """(C, factor) of a stock Transition: C = the out projection's rows, factor = the SwiGLU linear's rows / 2C."""
    C = int(module[2].weight.shape[0])
    return C, int(module[1].linear.weight.shape[0]) // (2 * C)


def refusal(module, x) -> Optional[str]:
    """The named reason this call runs the statement beneath before the provider is asked, or None (ask the provider).  The words are the lever's
    census vocabulary (registry.LEVERS["pair_transition"]["expected"] lists the expected ones by prefix)."""
    import torch
    if module.training:
        return "training"
    if not torch.is_tensor(x):
        return "rank"
    if x.dtype != torch.bfloat16:
        return "dtype:" + str(x.dtype).replace("torch.", "")
    if x.dim() < 3:
        return "rank"
    C, factor = geometry(module)
    if C != PC.TRUNK_C or int(x.shape[-1]) != PC.TRUNK_C:
        return f"c:{int(x.shape[-1])}"
    if factor != PC.TRUNK_N:
        return f"factor:{factor}"
    if not x.is_cuda:
        return "cpu"
    return None


def note_served(x) -> None:
    """Count ONE served fused transition on this lever's ledger (pair_block_residual runs the same kernel with the residual in its epilogue)."""
    L = STATE["ledger"]
    if L is not None:
        L.serve(f"{'x'.join(str(int(d)) for d in x.shape[:-2])}x{int(x.shape[-2])}")
        TC.note_upstream(x, "upstream_pair_transition")


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        TR = importlib.import_module(TARGET)
        import torch  # noqa: F401
        import opt_core
        from opt_core.attn import pair_fused as PF
        from opt_core.kernels import transition as KT
        from opt_core.counters import Ledger
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}")
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    w = word(mode)
    row = KT.split_word(w)[0]
    if w not in KT.TIER_WORDS and row not in KT.ROW_NAMES:
        return Installed(LEVER, False, reason=f"unknown_word:{w}")
    if row in KT.STOCK_ROWS or row in KT.SCHEDULE_ROWS:
        return Installed(LEVER, False, reason=f"word_names_no_kernel:{w}")
    pinned = w not in KT.TIER_WORDS
    ledger = Ledger(NAME, impl=f"opt_core.kernels.transition@{opt_core.__version__}:{w}", origin="core", expected=words)
    STATE["ledger"] = ledger; STATE["word"] = w; STATE["rows"] = {}; STATE["plan"] = {}
    ledger.set("copies", 0); ledger.set("word", w); ledger.set("pinned", int(pinned)); ledger.set("rows", "none"); ledger.set("plan", "none")
    stock = TR.Transition.forward                    # the exact-class binding / transition_chunk wrapper when those levers are in the row (installed before this one), else nn.Sequential.forward

    def forward(self, x):
        why = refusal(self, x)
        if why is not None:
            ledger.fallback(why)
            return stock(self, x)
        xin = x
        if not xin.is_contiguous():                   # the core reshapes x to rows: a strided x would copy silently there — copy here, counted
            xin = xin.contiguous(); ledger.count("copies")
        word_ = err = None
        try:
            u = fused(self, xin, residual=False)                                  # by word through the provider
        except PF.Unsupported as e:                   # the provider's named answer (statement / refusal), raised before any launch: keep the WORD only
            word_ = e.reason
        except Exception as e:  # noqa: BLE001 — a kernel that raised: this call runs the statement beneath, the gate refuses at exit
            err = type(e).__name__
        if word_ is not None or err is not None:
            if word_ is not None:
                ledger.fallback(word_)
            else:
                ledger.error(err)
            return stock(self, x)
        ledger.serve(f"{'x'.join(str(int(d)) for d in x.shape[:-2])}x{int(x.shape[-2])}")
        TC.note_upstream(x, "upstream_pair_transition")                     # the row-block wrapper below never sees this call: its ledger names who served it
        return u
    forward.__qualname__ = f"Transition.forward[atlasfold_opt:{LEVER}]"
    rebind(TR.Transition, "forward", forward, stock)
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[PC.gate_for(ledger, words)],
                     facts={"impl": ledger.impl, "ledger": ledger, "word": w, "pinned": pinned})
