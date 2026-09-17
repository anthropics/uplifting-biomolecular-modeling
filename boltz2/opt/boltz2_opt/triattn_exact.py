"""boltz2_opt.triattn_exact — the exact mode's triangle-attention core bound to the shared core's provider by its ``exact`` tier word
(registry lever ``triattn_exact``; exact class).

Under ``--mode exact`` the fused triangle-attention block (boltz2_opt.pairblock, variant ``cueq``) hands the attention core the library call
stock makes (cuEquivariance's ``triangle_attention``).  With this lever's switch set that core goes through
``opt_core.kernels.triattn.triangle_attention(q, k, v, bias, mask, scale, word="exact", stock=<the library call>)`` instead: the provider
serves its exact-class row ``triattn_exact`` on a (card, torch, library) stack its cell table vouches for and the ``stock=`` callable — the
same library call, its ``cueq`` row — everywhere else, per call and by name; an exact-class row is bit-identical to the library op wherever it
is served, so the block's output is the library kernel's bits either way.  A structural refusal the provider raises by name (``Refusal``) puts
THAT call on the library call, counted by kind; the row's own per-call words (short sequences, no proven cell) are the provider's business,
tallied on its evidence line (``opt_core.kernels.triattn.exact_member.evidence_line``).

Switch ``BOLTZ_TRIATTN_EXACT=1`` — the exact row alone sets it (modes._EXACT_ONLY); fast and big bind the provider's tolerance-class words
through pairblock's own variants.  Attached by ``worker_launch --attach triattn_exact`` right after the attention module, beside ``pairblock``
whose ``cueq`` core it routes (``core_for``: the seam pairblock calls per forward; modes.NEEDS).  Lines (stderr; the second at exit)::

    [boltz2-opt] LEVER name=triattn_exact state=on word=exact row=<row> class=<exact|stock> [exact_vs=<row>] cell=<cell key> measured=<yes|no>
                 cc=<M.m|-> stack=<cc|torch|library key> at=bf16xD32xH4xN<LINE_TOKENS> [refused=<kind> fallback=<row>]
                 calls=<n> served=<n> refused=<kind:n,…|-> errors=<n> rows=<row:n,…|-> kernel=<served>/<calls> kernel_refused=<reason:n,…|->
                 impl=opt_core.kernels.triattn origin=core

``row=`` … ``measured=`` are the provider's selection for the exact word on this card and stack at the pair track's shape class (bf16,
head_dim 32, 4 heads) and LINE_TOKENS tokens (``select`` / ``describe``); ``rows=`` the rows that served the run's calls, by the calls' own
token counts; ``kernel=`` / ``kernel_refused=`` the exact row's own tally (``exact_member.counts``).
"""
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from . import rowfloor as RF                                          # the kit's one card probe (card_cc: 'M.m' of the worker's device; None without CUDA)

TAG = "boltz2-opt"
NAME = "triattn_exact"
SWITCH = "BOLTZ_TRIATTN_EXACT"
WORD = "exact"                                                         # the provider's exact tier word
IMPL = "opt_core.kernels.triattn"
LEVERS = (NAME,)
LINE_SHAPE = ("bf16", 32, 4)                                           # the pair track's triangle attention under autocast: bf16 operands, head_dim 32, 4 heads (C=128 and the template's C=64 alike)
LINE_TOKENS = 512                                                      # the token count the activation line states its selection at; a served call's own count decides that call (rows=)

_STATE: Dict[str, Any] = {"active": False, "applied": [], "facts": None, "stack": {}, "row_at": {}, "routed": {},
                          "calls": 0, "served": 0, "refused": {}, "errors": {}, "rows": {}}


def enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """The switch: ``BOLTZ_TRIATTN_EXACT=1`` in the (worker's) environment."""
    return str((os.environ if env is None else env).get(SWITCH, "")).strip() == "1"


def variant(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The row word (``"1"``) when the switch is set, else None."""
    return "1" if enabled(env) else None


def _tok(v) -> str:
    return str(v if v is not None else "-").replace(" ", "_") or "-"


def _kv(d: Dict[str, int]) -> str:
    return ",".join(f"{_tok(k)}:{n}" for k, n in sorted(d.items())) or "-"


def _stack(cc: Optional[str]) -> str:
    """The provider's exact-vouch stack key for this card ('<cc>|torch<version>|cueq<version>'), read once per cc."""
    if cc not in _STATE["stack"]:
        try:
            from opt_core.kernels import triattn as T
            _STATE["stack"][cc] = str(T.exact_stack_key(cc))
        except Exception as e:  # noqa: BLE001 — named on the line, never raised into a served call
            _STATE["stack"][cc] = f"?:{type(e).__name__}"
    return _STATE["stack"][cc]


def _select_facts(cc: Optional[str], n_tokens: int) -> Dict[str, Any]:
    """The provider's answer for the exact word at (cc, bf16, D32, H4, n_tokens, fwd) on this stack: ``row`` (the fallback row on a refusal),
    ``words`` (describe's k=v words after row= / word=), ``refused`` / ``fallback`` (a Refusal's kind and named fallback)."""
    f: Dict[str, Any] = {"row": "-", "words": "", "refused": None, "fallback": None}
    if cc is None:
        f["refused"] = "no_card"
        return f
    try:
        from opt_core.kernels import triattn as T
    except Exception as e:  # noqa: BLE001
        f["refused"] = f"core:{type(e).__name__}"
        return f
    dt, D, H = LINE_SHAPE
    try:
        sel = T.select(cc, dt, D, H, int(n_tokens), "fwd", word=WORD, exact_stack=_stack(cc))
        f["row"] = str(sel.row)
        f["words"] = " ".join(w for w in str(T.describe(sel)).split() if not w.startswith(("row=", "word=")))
    except T.Refusal as e:
        f.update(row=str(e.fallback or "-"), refused=str(e.kind), fallback=e.fallback)
    except Exception as e:  # noqa: BLE001 — an argument the table cannot place (an unlisted capability word): named, the calls are the provider's to decide
        f["refused"] = f"select:{type(e).__name__}"
    return f


def _kernel_counts() -> Dict[str, Any]:
    """The exact row's own per-process tally (exact_member.counts(): served / calls / refused by reason)."""
    try:
        from opt_core.kernels.triattn import exact_member as EM
        return dict(EM.counts())
    except Exception:  # noqa: BLE001
        return {}


def evidence() -> Optional[str]:
    """The exact row's evidence line (``triattn_exact: served n/m calls (refused: {...})``), None when the core does not carry the row."""
    try:
        from opt_core.kernels.triattn import exact_member as EM
        return str(EM.evidence_line())
    except Exception:  # noqa: BLE001
        return None


def _activate() -> None:
    """Once per process (the attach hook's apply() or the first core_for under the switch): resolve the line's facts (card, stack, the
    provider's selection), mark the lever applied, register the exit tally, print the LEVER line."""
    if _STATE["active"]:
        return
    cc = RF.card_cc()
    f = {"word": WORD, "cc": cc, "stack": _stack(cc), "at": "%sxD%dxH%dxN%d" % (LINE_SHAPE[0], LINE_SHAPE[1], LINE_SHAPE[2], LINE_TOKENS)}
    f.update(_select_facts(cc, LINE_TOKENS))
    _STATE.update(active=True, applied=[NAME], facts=f)
    try:
        from opt_core import report as _report
        _report.register_exit_tally(TAG + "/" + NAME, line)
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write((line() or "") + "\n")


def _note_row(k) -> None:
    """rows=: the provider's row for this call's token count (its memoised selection at the call's own size; '-' without a card)."""
    try:
        n = int(k.shape[-2])
    except Exception:  # noqa: BLE001
        n = 0
    row = _STATE["row_at"].get(n)
    if row is None:
        cc = (_STATE["facts"] or {}).get("cc")
        row = _STATE["row_at"][n] = _select_facts(cc, n)["row"] if (cc is not None and n > 0) else "-"
    _STATE["rows"][row] = _STATE["rows"].get(row, 0) + 1


def core_for(lib: Callable) -> Callable:
    """pairblock's seam: the attention core the block hands pair_fused where it would hand ``lib``, its library call
    (``core(q, k, v, bias, mask5, scale)``) — ``lib`` itself with the switch unset; with it set, a callable of the same signature that serves
    each call through the provider's exact word with ``lib`` as its ``stock=`` op (memoised per ``lib``)."""
    if not enabled():
        return lib
    fn = _STATE["routed"].get(lib)
    if fn is not None:
        return fn
    _activate()
    from opt_core.kernels import triattn as T
    from opt_core.oom import is_oom

    def stock(q, k, v, bias, mask=None, scale=None):                 # `lib` in the provider's stock= convention
        return lib(q, k, v, bias, mask, scale)

    def routed(q, k, v, bias, mask5, scale):
        _STATE["calls"] += 1
        try:
            out = T.triangle_attention(q, k, v, bias, mask5, scale, word=WORD, stock=stock)
        except T.Refusal as e:                                       # structural, by name: THIS call on the library call, counted by kind
            kind = _tok(str(getattr(e, "kind", "") or type(e).__name__).split(":")[0])
            _STATE["refused"][kind] = _STATE["refused"].get(kind, 0) + 1
            return stock(q, k, v, bias, mask=mask5, scale=scale)
        except Exception as e:  # noqa: BLE001 — counted here (the gate refuses), raised to the block, whose own error path takes the call
            if not is_oom(e):                                        # (an out-of-memory error propagates uncounted: no fallback on out-of-memory)
                name = type(e).__name__
                _STATE["errors"][name] = _STATE["errors"].get(name, 0) + 1
            raise
        _STATE["served"] += 1
        try:
            _note_row(k)
        except Exception:  # noqa: BLE001 — the census never costs a served call
            pass
        return out

    stock.__wrapped__ = lib
    routed.__wrapped__ = lib
    routed.stock = stock
    _STATE["routed"][lib] = routed
    return routed


def apply(spec: Optional[str] = None) -> List[str]:
    """The attach hook's entry (worker_launch --attach triattn_exact, after the attention module): with the switch set (``spec`` overrides
    the environment) the lever is active — its facts resolved, its LEVER line printed — and pairblock's ``cueq`` core is served through
    ``core_for`` from the first call on; returns the levers installed ([] with the switch unset)."""
    if not _STATE["applied"] and (enabled() if spec is None else str(spec).strip() == "1"):
        _activate()
    return list(_STATE["applied"])


def facts() -> Dict[str, Any]:
    """The activation facts: word / cc / stack / at / row / words / refused / fallback ({} before activation)."""
    return dict(_STATE["facts"] or {})


def census() -> Dict[str, Any]:
    return {"calls": _STATE["calls"], "served": _STATE["served"], "fallback": dict(_STATE["refused"]), "errors": dict(_STATE["errors"]),
            "rows": dict(_STATE["rows"]), "kernel": _kernel_counts()}


def verdict() -> Dict[str, Any]:
    """Fail-closed gate: ok unless not applied or an error was counted; idle = applied and no call reached the block's library core (every
    triangle-attention call of the run on one of the block's declared stock paths)."""
    if not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    if _STATE["errors"]:
        return {"ok": False, "idle": False, "reason": "errors=" + _kv(_STATE["errors"])}
    if _STATE["calls"] == 0:
        return {"ok": True, "idle": True, "reason": "no call reached the block's library core"}
    return {"ok": True, "idle": False, "reason": None}


def line() -> Optional[str]:
    if not _STATE["active"]:
        return None
    f = _STATE["facts"] or {}
    kc = _kernel_counts()
    parts = [f"[{TAG}] LEVER name={NAME} state=on word={WORD} row={_tok(f.get('row'))}"]
    if f.get("words"):
        parts.append(str(f["words"]))
    parts.append(f"cc={_tok(f.get('cc'))} stack={_tok(f.get('stack'))} at={f.get('at')}")
    if f.get("refused"):
        parts.append(f"refused={_tok(f['refused'])} fallback={_tok(f.get('fallback'))}")
    parts.append(f"calls={_STATE['calls']} served={_STATE['served']} refused={_kv(_STATE['refused'])} errors={sum(_STATE['errors'].values())} "
                 f"rows={_kv(_STATE['rows'])} kernel={kc.get('served', 0)}/{kc.get('calls', 0)} kernel_refused={_kv(dict(kc.get('refused') or {}))}")
    parts.append(f"impl={IMPL} origin=core")
    return " ".join(parts)


def report() -> Dict[str, Any]:
    """The worker log's ``triattn_exact_report`` (the block adapters' shape: applied / variant / line / census / gate, + facts / evidence)."""
    return {"applied": list(_STATE["applied"]), "disabled": {}, "variant": "1" if _STATE["applied"] else None, "line": line(),
            "census": census() if _STATE["active"] else None, "gate": verdict() if _STATE["applied"] else None,
            "facts": facts(), "evidence": evidence(), "impl": IMPL, "patched": []}
