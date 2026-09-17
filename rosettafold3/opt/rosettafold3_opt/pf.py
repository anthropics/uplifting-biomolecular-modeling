"""The xtr lever: rf3's Transition (LayerNorm -> linear_1 / linear_2 SwiGLU -> linear_3, no biases) served by the shared core's transition
PROVIDER ``opt_core.kernels.transition`` under its EXACT TIER WORD.  Per call the provider resolves ``exact`` for (compute capability, dtype,
width c x hidden, token count, this stack) to the row it has BITWISE-VOUCHED there -- ``v1`` (the carried fpf transition kernel reading
the module's own LayerNorm output: ``pair_fused.transition(impl=fpf, ln=stock, x_ln)``) above the cell's size floor on cc 9.0 and
cc 8.0 for the 128-wide pair transitions -- or to the stock statements BY NAME (the 384-wide single transition, the c=64 tracks, sizes below
the vouch, a width without a cell): those calls run the module's own forward, counted ``route:<key>:<the provider's word>``.  No exactness
table lives in this kit: the vouches (stack, size floor, width) are the provider's cells.  A refusal the provider raises at LAUNCH of a row it
selected is a counted ``fallback:<key>:<reason>`` and fails the run by name (:func:`problems`; fold.lever_failures)."""
import sys
import threading
from typing import Dict, Optional, Tuple

from . import _core

KERNELS = ("fpf_transition", "fpf_glue_v2")       # the carried core cells the provider's v1 row launches (routed + sum-checked by opt_core.attn.pair_fused itself)
WORD = "exact"                                    # the tier word this lever asks: the exact mode's transition
PROVIDER = "opt_core.kernels.transition"
STATE = {"on": False, "impl": None, "serve": (), "routes": {}, "rows": {}, "error": None, "cells_sha256": None, "reason": None, "cc": None, "stack": None}
CPU_PROCESS = "no CUDA device in this process: no GPU call to serve (a CPU helper process of a GPU run; the GPU process installs the lever and accounts for it)"
CENSUS: Dict[str, int] = {}                       # served:<key> | route:<key>:<why> | fallback:<key>:<reason> -> calls
_LOCK = threading.Lock()
_PREV = {}                                        # the class forward this lever wrapped (stock, or the FPF add-on's transition forward)
_DECISIONS: Dict[tuple, tuple] = {}               # (key, n_tokens, rows) -> (serve?, census word, row): the provider's answer, asked once per distinct call shape


class XtrRefused(RuntimeError):
    """The lever cannot apply in this process (named precondition); stack.fpf_apply turns it into the NOT ACTIVE line."""


def _count(key: str) -> None:
    with _LOCK:
        CENSUS[key] = CENSUS.get(key, 0) + 1


def _key(C: int, HID: int) -> str:
    return f"{C}x{HID}"


def _word(text: str) -> str:
    """A census-safe token from a provider reason (no spaces / commas / '=' so the LEVER line's by_key stays one token per entry)."""
    return str(text).split(" ")[0].replace(",", ";").replace("=", "~")[:96]


def family_of(TR, shape, C: int, HID: int) -> str:
    """The provider's row family for one Transition call, from the call's own shape: square token dims ``[.., I, I, c]`` (the pair track, the
    template pair stack) are 'pair'; a non-square input whose width the provider tables as a SINGLE-track cell (``[.., I, c_s]``: rows = I) is
    'single'; anything else (the MSA rows ``[.., S, I, 64]``) asks 'pair', under which the provider names its 64-wide rows family itself."""
    shape = tuple(int(d) for d in shape)
    square = len(shape) >= 3 and shape[-3] == shape[-2]
    if not square and TR.cell_word(int(C), int(HID), "single") is not None:
        return "single"
    return "pair"


def decide(TR, C: int, HID: int, n_tokens: int, rows: int, cc: str, stack: Optional[str], family: str = "pair") -> Tuple[bool, str, Optional[str]]:
    """(serve, census word, row) for one call shape from the provider's ``select(word='exact', ..., ln_given=True)``: a non-stock row it has
    vouched bitwise on this stack at this size serves (``served:<key>``, row named); a stock row is the module's own forward BY NAME
    (``route:<key>:stock:<row>@<cell>``); a Refusal is the module's own forward BY NAME (``route:<key>:<refusal kind>``).  Memoised per shape."""
    memo = (C, HID, int(n_tokens), int(rows), cc, stack, family)
    hit = _DECISIONS.get(memo)
    if hit is not None:
        return hit
    key = _key(C, HID)
    try:
        sel = TR.select(WORD, c=C, hidden=HID, n_tokens=int(n_tokens), dtype="bf16", direction="fwd", timing="eager", family=family,
                        residual=False, cc=cc, stack=stack, rows_count=int(rows), ln_given=True)
    except TR.Refusal as e:
        out = (False, f"route:{key}:{_word(e)}", None)
    else:
        if sel.row in TR.STOCK_ROWS:
            out = (False, f"route:{key}:stock:{sel.row}", sel.row)
        else:
            out = (True, f"served:{key}", sel.row + ((":" + sel.variant) if sel.variant else ""))
    _DECISIONS[memo] = out
    return out


TTR_ROUTE = "route:prev:fpf_ttr"                                  # a width the FPF add-on's fused transition serves on this arm: left to it (xtr serves what ttr does not)
TTR_ADAPTER = "fpf_rf3_adapter"                                  # the add-on's module (rf3fpf/fpf_rf3_adapter.py): its CFG2["transition"] == "triton" is the ttr component on


def ttr_keys(prev, routes=("128x512", "128x256", "64x256", "64x128")) -> Tuple[str, ...]:
    """The transition widths the FPF add-on's ``ttr`` serves in this process — C in (64, 128) with HID % 128 == 0 (fpf_rf3_adapter
    _transition_forward_v2; rf3's widths of that class by default) — when the forward this lever wraps IS the add-on's and its transition component is on; else ().
    Those keys leave xtr's serve set under the named route ``route:prev:fpf_ttr`` (the exact mode carries no ttr: nothing changes there)."""
    adp = sys.modules.get(TTR_ADAPTER)
    if adp is None or getattr(prev, "__module__", None) != TTR_ADAPTER:
        return ()
    if (getattr(adp, "CFG2", None) or {}).get("transition") != "triton":
        return ()
    out = []
    for key in routes:
        try:
            C, HID = (int(x) for x in key.split("x"))
        except ValueError:
            continue
        if C in (64, 128) and HID % 128 == 0:
            out.append(key)
    return tuple(out)


def _weights(TR, m):
    c = m.__dict__.get("_rf3opt_xtr")
    if c is None:
        for lin in (m.linear_1, m.linear_2, m.linear_3):               # the mapper rule: a parameter the provider is not handed must not exist (never dropped silently)
            if getattr(lin, "bias", None) is not None:
                raise TR.Refusal("mapper:linear-bias", WORD, "torch_swiglu")
        c = TR.pack(w_o=m.linear_3.weight, w_a=m.linear_1.weight, w_b=m.linear_2.weight, ln_w=m.layer_norm_1.weight, ln_b=m.layer_norm_1.bias,
                    eps=m.layer_norm_1.eps)
        m.__dict__["_rf3opt_xtr"] = c
    return c


def forward(self, X):
    """Transition.forward under the lever: the provider's exact row where it vouches one for this call, everything else the previous forward BY NAME."""
    import torch
    C = int(self.linear_1.weight.shape[1]); HID = int(self.linear_1.weight.shape[0])
    key = _key(C, HID)
    if key in STATE["routes"] and STATE["routes"][key] == TTR_ROUTE:  # a width the FPF add-on's fused transition serves on this arm: left to it, by name
        _count("route:" + key + ":" + TTR_ROUTE); return _PREV["forward"](self, X)
    if not (X.is_cuda and torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        _count("route:" + key + ":no-bf16-autocast"); return _PREV["forward"](self, X)
    TR = sys.modules[PROVIDER]
    n_tok = int(X.shape[-2]) if X.dim() >= 2 else 1                # the pair / single track's token count I (z: [.., I, I, c]; s: [.., I, c]; msa rows: [.., S, I, c])
    rows = X.numel() // C
    fam = family_of(TR, X.shape, C, HID)
    serve, word, row = decide(TR, C, HID, n_tok, rows, STATE["cc"], STATE["stack"], fam)
    if not serve:
        _count(word); return _PREV["forward"](self, X)
    n32 = self.layer_norm_1(X)                                   # the module's own LayerNorm, as stock runs it (fp32 under autocast)
    n16 = n32.to(torch.bfloat16)                                 # the rounding autocast applies in front of linear_1
    if n16.stride(-1) != 1:
        n16 = n16.contiguous()
    try:
        Y, sel = TR.transition(n16, _weights(TR, self), word=WORD, residual=False, x_ln=n16, n_tokens=n_tok, family=fam, stack=STATE["stack"])
    except TR.Refusal as e:                                      # refused at launch of a row it selected: counted, fails the run by name (problems)
        _count("fallback:" + key + ":" + _word(e)); return _PREV["forward"](self, X)
    _count("served:" + key)
    with _LOCK:
        STATE["rows"][key] = sel.row + ((":" + sel.variant) if sel.variant else "")
        if key not in STATE["serve"]:
            STATE["serve"] = tuple(STATE["serve"]) + (key,)
    return Y


def enable() -> dict:
    """Install the lever class-wide (once). Needs torch+CUDA and the core's transition provider importable. Returns :func:`describe`.
    Raises :class:`XtrRefused` naming the failed precondition."""
    if STATE["on"]:
        return describe()
    try:
        import torch
        TR = _core.load("kernels.transition")
    except Exception as e:
        raise XtrRefused(f"{PROVIDER} not importable: {type(e).__name__}: {e}") from e
    if not torch.cuda.is_available():                            # a CPU helper process of a GPU run (the served worker line's spawned prefetch process imports rf3 too):
        STATE.update(on=False, reason=CPU_PROCESS)               # no GPU call can reach the lever here -- a named no-op, not a refusal; the GPU process installs and accounts
        return describe()
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = "%d.%d" % tuple(torch.cuda.get_device_capability(dev))
    try:
        stack = TR.stack_word(dev)                               # the provider's stack word (its vouches are per stack); None: the reference stack's numbers, flagged by it
    except Exception:
        stack = None
    import rf3.model.layers.layer_utils as LU
    cls = LU.Transition
    _PREV["forward"] = cls.forward                               # stock (activation_checkpointing-wrapped) or the FPF add-on's transition forward
    routes: Dict[str, str] = {}
    for k in ttr_keys(_PREV["forward"]):                          # composed with the FPF add-on's fused transition (arm component ttr): the widths ttr serves stay ttr's, by name
        routes[k] = TTR_ROUTE
    cls.forward = forward
    sha = None
    try:
        PF = _core.load("attn.pair_fused"); sha = PF.cells_sha256()
    except Exception:
        pass
    STATE.update(on=True, impl=getattr(TR, "__file__", None), serve=(), routes=routes, rows={}, cells_sha256=sha, cc=cc, stack=stack,
                 prev=getattr(_PREV["forward"], "__module__", "?") + "." + getattr(_PREV["forward"], "__qualname__", "?"))
    return describe()


def census() -> dict:
    with _LOCK:
        c = dict(CENSUS)
    served = sum(v for k, v in c.items() if k.startswith("served:"))
    routed = sum(v for k, v in c.items() if k.startswith("route:"))
    fallback = sum(v for k, v in c.items() if k.startswith("fallback:"))
    return {"calls": served + routed + fallback, "served": served, "routed": routed, "fallback": fallback, "by_key": c}


def problems() -> list:
    """The fail-closed gate as sentences (empty = clean): any counted fallback (the provider refused a row it had selected, at launch); a
    lever that no call reached while on.  A lever whose every call the provider routed to the statements BY NAME (below its vouched size,
    an unvouched width, a stock cell) served nothing by the provider's decision -- a named state (``routed_by_name``), not a failure."""
    if not STATE["on"]:
        return []
    c = census(); out = []
    if c["fallback"]:
        out.append("xtr fallbacks (the provider refused a selected row at launch): " + ", ".join(f"{k}={v}" for k, v in c["by_key"].items() if k.startswith("fallback:")))
    if c["calls"] == 0:
        out.append("no Transition call reached the xtr lever in this process (installed but never ran)")
    return out


def routed_by_name(c: Optional[dict] = None) -> bool:
    """True when the lever served no call because the provider routed every call to the statements by name (no fallback among them)."""
    c = census() if c is None else c
    return bool(STATE["on"] and c.get("calls") and not c.get("served") and not c.get("fallback"))


below_floor = routed_by_name                       # the exit tally's older name for the same named state (report.lever_states: state=skipped reason=below_floor)


def describe() -> dict:
    """The exit tally's ``xtr`` block: on, impl (the provider), construction, the widths served + their rows, routes, the census, ok + reason."""
    bad = problems()
    return {"on": STATE["on"], "impl": STATE["impl"], "origin": "core", "construction": f"kernels.transition(word={WORD},x_ln=given)",
            "serve": list(STATE["serve"]), "rows": dict(STATE.get("rows") or {}), "routes": STATE["routes"], "stack": STATE.get("stack"),
            "cells_sha256": STATE["cells_sha256"], "prev": STATE.get("prev"),
            **({"below_floor": True} if STATE["on"] and routed_by_name() else {}),               # every call routed to the statements by the provider's decision (LEVER state skipped)
            "census": census() if STATE["on"] else None,
            "ok": STATE["on"] and not bad, "reason": ("; ".join(bad) if bad else (None if STATE["on"] else (STATE.get("reason") or "not installed")))}


def lever_evidence() -> list:
    c = census()
    rows = ",".join(f"{k}:{v}" for k, v in sorted((STATE.get("rows") or {}).items()))
    return [("impl", STATE["impl"]), ("origin", "core"), ("word", WORD), ("serve", ",".join(STATE["serve"])), ("rows", rows or "none"),
            ("calls", c["calls"]), ("served", c["served"]), ("routed", c["routed"]), ("fallback", c["fallback"])]
