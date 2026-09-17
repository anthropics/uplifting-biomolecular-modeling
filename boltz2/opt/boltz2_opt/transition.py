"""boltz2_opt.transition — the engine adapter of the core's transition provider (``opt_core.kernels.transition``: one provider over
every carried implementation of the SwiGLU transition, LayerNorm -> W_a|W_b -> silu(a)*b -> W_out in one kernel, the 4C hidden
never reaching HBM, and the measured per-(card, stack, C, hidden, N) cell table that names which row serves which tier word) onto Boltz-2's
``Transition`` (``boltz.model.layers.transition``: the pair transition of every Pairformer stack, the MSA module's m- and z-transitions,
the sequence transition).  Attached by ``boltz2_opt.worker_launch --attach transition`` right after that module imports.

Switch ``BOLTZ_TRANSITION`` (the mode table sets it, modes.py) = the provider's TIER word: ``exact`` = the exact-class construction on the
calls whose stock rounding chain the cell reproduces — unchunked calls under bf16 autocast: the module's own LayerNorm (fp32, as stock),
autocast's bf16 cast applied once, the cell's projections / silu / product / output projection at stock's rounding points, the update
returned in bf16 for the caller's fp32 residual add — wherever the provider's table vouches its kernel arm bitwise for this stack and size
(else the module's own forward, by name: ``exact_unvouched``; a cell whose exact winner is the statement itself: ``core_stock``); the
hidden-CHUNKED calls (``chunk_size`` set: boltz chunks the MSA module's transitions above 384 tokens and accumulates the partial outputs in
bf16 — another rounding chain) and the autocast-off fp32 calls (the Pairformer's sequence transition) take the module's original forward,
counted by name.  ``fast`` = the cell's measured fast winner (LayerNorm in the kernel, tolerance class), the chunked calls served unchunked
too (Tier 2 there: memory-safe, the hidden stays on chip).  ``big`` = the cell's memory-tier winner, same construction as fast.
``core.<word>`` = a provider ROW word, for measurement.  A call shape without a cell takes the original forward by the provider's own
refusal (``no_cell``).  ``EXPECTED`` lists the reasons a healthy run may show; anything else, or a kernel error, refuses the fail-closed
gate (``verdict()``).  No shape, row-count or card table lives in this adapter: the provider decides, or names why not.
"""
import os
import sys
from typing import Any, Dict, List, Optional

from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no fallback applied

from . import exactln as XLN                                         # the bitwise LayerNorm replica's fused-cast entry (ln_to_bf16: the stock statement `module(x).to(bfloat16)` when that lever is off) [EXACTLN]

TAG = "boltz2-opt"
NAME = "F5.transition"
SWITCH = "BOLTZ_TRANSITION"
VARIANTS = ("exact", "fast", "big")                 # the TIER words of the core's transition provider (opt_core.kernels.transition: one provider over every carried row and
                                                      # the measured cell table): `exact` = the cell's bitwise arm fed the module's own LayerNorm output (row v1 where the table
                                                      # vouches it for this stack and size, else the module's own forward BY NAME), `fast` = the cell's measured fast winner
                                                      # (LayerNorm in the kernel: tolerance class), `big` = the cell's memory-tier winner; + `core.<word>` (CORE_PREFIX): a
                                                      # provider ROW word for measurement (`core.v1`, `core.v2:fast`, `core.v2@<cfg>`, `core.v1:lnfused`, ...). A word the provider
                                                      # does not serve for a call (its by-name refusal, or a cell whose winner is the stock arm) is counted by name and that call
                                                      # takes the module's original forward
CORE_PREFIX = "core."
CORE_EXACT_ROWS = ("v1", "exact")                     # words of the exact class here (the caller's LayerNorm output projected at the stock rounding points; `v1:lnfused` is NOT):
                                                      # they feed the module's own LayerNorm output and keep the hidden-chunked calls on the module (the chunked rule)
LEVERS_OF = {"exact": ("fused_transition",), "fast": ("fused_transition",)}
LN_OF = {"exact": "stock", "fast": "kernel"}          # exact class: the module's own LayerNorm output (bf16) is the cell's input; fast class: the row normalizes in the kernel
LEVERS = ("fused_transition",)                        # the registry levers this adapter installs (worker_launch reads it)
IMPL = "opt_core.kernels.transition"                  # the LEVER line's impl=: the provider; the row that served each call is the line's core_rows=<row>:<n>
EXPECTED = ("autocast_off", "chunked", "core_stock", "exact_unvouched", "no_cell")   # declared stock paths, by name: the fp32 sequence transition (autocast off); the
# hidden-chunked MSA-module calls under an exact-class word; a cell whose winner for the word on this card is the stock arm (core_stock: the
# module's own forward serves — e.g. `exact` below the size the kernel arm beats the statement); the exact tier on a stack / size the table holds
# no bitwise vouch for (exact_unvouched: the module's own forward is the exact tier there); a call shape the table has no cell for (no_cell:
# the diffusion pairwise-conditioner's (128, 256) transitions). Which row serves which (card, C, hidden, N) is the provider's measured table,
# not this adapter's: no shape, row-count or card table lives here
_STATE: Dict[str, Any] = {"ledger": None, "applied": [], "patched": [], "variant": None, "errors": {}, "served_by": {}, "core_rows": {}}


def variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    if v.startswith(CORE_PREFIX) and len(v) > len(CORE_PREFIX):
        return v
    return v if v in VARIANTS else None


def core_word(v: Optional[str]) -> Optional[str]:
    """The provider word a variant binds: the TIER word itself for exact | fast | big, the ROW word of a `core.<word>` variant, None for no word."""
    if (v or "").startswith(CORE_PREFIX):
        return v[len(CORE_PREFIX):]
    return v if v in VARIANTS else None


def class_of(v: Optional[str]) -> Optional[str]:
    """`exact` | `fast`: the construction class a variant runs under (LN_OF, LEVERS_OF and the chunked rule key on it): `exact` for CORE_EXACT_ROWS (the
    word before any `@cfg`, exactly), else `fast` (`big` and every other row word)."""
    w = core_word(v)
    if w is None:
        return None
    return "exact" if w.split("@")[0] in CORE_EXACT_ROWS else "fast"


def requested(environ=None) -> bool:
    return variant(environ) is not None


def weights_of(m):
    """boltz Transition -> the core's canonical transition tensors (opt_core.attn.pair_fused.TRANSITION_KEYS)."""
    for lin in (m.fc1, m.fc2, m.fc3):                                              # every parameter of the module the pack is not given must not exist (never a silently dropped bias)
        assert lin.bias is None, f"{type(m).__name__}: a projection bias the fused transition is not given ({lin})"
    return dict(ln_w=m.norm.weight, ln_b=m.norm.bias, w_a=m.fc1.weight, w_b=m.fc2.weight, w_out=m.fc3.weight, eps=getattr(m.norm, "eps", 1e-5))


def census_word(kind: str) -> str:
    """The declared census word of a provider refusal kind (opt_core.kernels.transition.Refusal.kind): `exact_unvouched` for the exact tier's
    vouch refusals (not recorded on this stack / below the vouched size), `no_cell` for a call shape without a cell, else `core_refused:<kind>`
    (undeclared: the gate refuses the run and the word names why)."""
    k = str(kind)
    if k.startswith("exact_vouch_"):
        return "exact_unvouched"
    if k.startswith("no_cell"):
        return "no_cell"
    return "core_refused:" + k[:60]


def row_word(sel) -> str:
    """`<row>[:<variant>][@<launch>]` of a provider Selection — the LEVER line's core_rows= key (the launch is the provider's own word for a
    per-cell configuration, cfg_word; None on a card whose cell runs the package's own configuration)."""
    from opt_core.kernels import transition as KT
    cfg = getattr(sel, "cfg", None)
    cw = cfg if isinstance(cfg, str) else (KT.cfg_word(cfg) if cfg else None)
    return str(sel.row) + ((":" + sel.variant) if getattr(sel, "variant", None) else "") + (("@" + cw) if cw else "")


def apply(spec: Optional[str] = None) -> List[str]:
    """Patch Transition.forward class-wide (idempotent). ``spec`` overrides the env switch."""
    if _STATE["ledger"] is not None:
        return list(_STATE["applied"])
    v = variant() if spec is None else variant({SWITCH: str(spec)})
    if v is None:
        return []
    import torch
    from opt_core.attn import pair_fused as PF
    import boltz.model.layers.transition as TRm

    cls = class_of(v)                                                # exact | fast: the construction class (CORE_EXACT_ROWS)
    cword = core_word(v)                                             # the provider word: the tier word itself, or the row word of `core.<word>`
    from opt_core.kernels import transition as KT                   # standard library at import; torch inside the serving calls
    ln = LN_OF[cls]
    L = PF.ledger(NAME, expected=EXPECTED)
    cc = "%d.%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    stack = KT.stack_word() if torch.cuda.is_available() else None   # '<card>:torch<version>/<triton>': the table's vouch key for the exact tier, resolved once
    orig = TRm.Transition.forward

    def _core_weights(m, device):                                    # the provider's canonical pack (opt_core.kernels.transition.pack), once per module
        W = getattr(m, "_opt_transition_W", None)
        if W is None:
            w = weights_of(m)
            W = KT.pack(w_a=w["w_a"], w_b=w["w_b"], w_o=w["w_out"], ln_w=w["ln_w"], ln_b=w["ln_b"], eps=w["eps"], device=device)
            m._opt_transition_W = W
        return W

    def forward(self, x, chunk_size: int = None):
        if not (x.is_cuda and torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16 and x.dtype in (torch.float32, torch.bfloat16)):
            L.fallback("autocast_off"); return orig(self, x, chunk_size)
        if chunk_size is not None and cls == "exact":
            L.fallback("chunked"); return orig(self, x, chunk_size)
        refused = None                                               # the refusal WORD; the stock path runs outside the except block (a bound exception keeps the frame's tensors alive)
        try:
            W = _core_weights(self, x.device)
            c, hidden, n = int(x.shape[-1]), int(W.hidden), int(x.shape[-2])
            family = "rows" if c == 64 else "pair"                       # the table's cell families: the pair transition (N·N rows of C=128) / the MSA rows of width 64
            rows = int(x.numel() // c)
            sel0 = KT.select(cword, c=c, hidden=hidden, n_tokens=n, dtype="bf16", family=family, residual=False, mask=False, cc=cc, stack=stack,
                             rows_count=rows, ln_given=(ln == "stock"))   # the provider decides for this (card, stack, C, hidden, N): a row, the stock arm, or a Refusal by name (below)
            if sel0.row in getattr(KT, "STOCK_ROWS", ("torch_swiglu", "engine_module", "compile")):
                refused = "core_stock"                                   # the cell's winner for this word here is the statement: the module's own forward serves, by name
            else:
                x_ln = None
                if ln == "stock":                                        # exact class: the module's own LayerNorm (fp32 under autocast), autocast's bf16 cast applied once
                    x_ln = XLN.ln_to_bf16(self.norm, x)                   # = self.norm(x).to(torch.bfloat16): the stock statement unless the exactln lever is on, which fuses the cast into its bitwise LayerNorm replica's store [EXACTLN]
                    if not x_ln.is_contiguous():
                        x_ln = x_ln.contiguous()
                y, sel = KT.transition(x, W, word=cword, residual=False, x_ln=x_ln, n_tokens=n, family=family, stack=stack)
                rw = row_word(sel)
                _STATE["core_rows"][rw] = _STATE["core_rows"].get(rw, 0) + 1     # which provider row / launch served (the LEVER line's core_rows=)
        except KT.Refusal as e:                                          # the provider's by-name refusal (kind, row, fallback): counted under its declared census word, this call takes the original forward
            refused = census_word(getattr(e, "kind", e))
        except Exception as e:  # noqa: BLE001 — a kernel error: counted (the gate refuses), this call served by the original forward —
            if is_oom(e): raise     # except a GPU out-of-memory error, which propagates: no fallback on out-of-memory
            k = type(e).__name__
            _STATE["errors"][k] = _STATE["errors"].get(k, 0) + 1; L.error(e)
            if _STATE["errors"][k] <= 2:
                print(f"[{TAG}] {NAME} kernel error {k}: {str(e)[:300]} — this call takes the stock forward", file=sys.stderr, flush=True)
            refused = "error:" + k
        if refused is not None:
            if not refused.startswith("error:"):
                L.fallback(refused)
            return orig(self, x, chunk_size)
        kind = "unchunked" if chunk_size is None else "chunked_as_one"
        L.serve(f"{int(x.shape[-2])}x{int(x.shape[-1])}", impl=kind)
        _STATE["served_by"][kind] = _STATE["served_by"].get(kind, 0) + 1
        return y

    TRm.Transition.forward = forward
    _STATE.update(ledger=L, applied=list(LEVERS_OF[cls]), patched=["Transition.forward"], variant=v)
    try:
        from opt_core import report as _report
        _report.register_exit_tally(TAG + "/" + NAME, line)
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write(line() + "\n")
    return list(LEVERS_OF[cls])


def line() -> Optional[str]:
    L = _STATE["ledger"]
    if L is None:
        return None
    from opt_core.attn import pair_fused as PF
    return PF.emit_line(L, TAG, variant=_STATE["variant"], impl=IMPL, ln=LN_OF.get(class_of(_STATE["variant"]) or "", "-"),
                        core_rows=",".join(f"{k}:{n}" for k, n in sorted(_STATE["core_rows"].items())) or "-")


IDLE = "idle: every transition call of the run took a declared stock path — the lever is installed and served nothing"


def verdict() -> Dict[str, Any]:
    """Fail-closed (the ledger's gate): refused on any kernel error, on any fallback reason outside EXPECTED, or when calls arrived and none
    was served — except the idle case (every call took a DECLARED stock path: the lever is installed, the input never reached it)."""
    L = _STATE["ledger"]
    if L is None:
        return {"ok": False, "idle": False, "reason": "not applied"}
    if L.calls > 0 and L.served == 0 and not L.errors and not L.unexpected():
        return {"ok": True, "idle": True, "reason": IDLE}
    g = L.gate(NAME)
    return {"ok": bool(g.ok), "idle": False, "reason": getattr(g, "reason", None)}


def census() -> Dict[str, Any]:
    L = _STATE["ledger"]
    return None if L is None else {"served": dict(_STATE["served_by"]), "served_total": L.served, "fallback": L.fallbacks, "errors": L.errors, "shapes": L.shapes, "calls": L.calls}


def report() -> Dict[str, Any]:
    L = _STATE["ledger"]
    if L is None:
        return {"applied": [], "disabled": {}, "line": None, "census": None, "gate": None, "patched": []}
    return {"applied": list(_STATE["applied"]), "disabled": {}, "line": line(), "census": census(), "gate": verdict(), "patched": list(_STATE["patched"]),
            "variant": _STATE["variant"], "cls": class_of(_STATE["variant"]), "core_word": core_word(_STATE["variant"]), "core_rows": dict(_STATE["core_rows"]),
            "impl": IMPL, "expected": list(EXPECTED)}
