"""fpf.triatt_exact -- lever ``triatt_exact`` (EXACT class; the exact composition of the kit README rows of cc 9.0 and cc 8.0): the pair
stacks' triangle attention bound to the shared core's triangle-attention provider ``opt_core.kernels.triattn`` BY THE TIER WORD ``exact`` ALONE.
Every library call the engine makes at a bound site asks

    T.triangle_attention(q, k, v, bias, mask, scale, word="exact", stock=<the site's own library callable>)

in the provider's convention (q / k / v [B,N,H,S,D], bias [B,1,H,S,S], mask [B,N,1,1,S] bool or None): the statement's [N,H,S,D] / [1,H,S,S] /
[N,1,1,S] layout gains the leading axis on the way in (views), and the served tensor comes back exactly as the library returns it -- [1,N,H,S,D]
for a 4-D statement call too (the engine indexes the leading axis itself).  No row preference, no kit floor, no kit cell table.  Under the word ``exact`` the provider serves its
exact-class row ``triattn_exact`` -- the carried CUDA member, bit-identical to the library op on the cells its own table proves, compiled at
first use -- only on a (cc | torch | library) stack its cell table vouches, and the stock op (row ``cueq`` = the site's own callable) BY NAME
everywhere else; the member's per-call refusals inside the provider take the library op for that call and are counted by the member
(``exact_member.counts()``).  A structural refusal the face raises by name (``T.Refusal``) takes the site's callable for that call, counted by
kind here.  Either way the bytes are the library op's: the lever is bitwise the mode without it by construction.

Sites (the names through which every trunk path reaches the library op; each wrapped once, ``_ptx_triatt_exact`` on the wrapper, the callable
found there kept as that site's ``stock``):
  tl    protenix.model.triangular.layers.cuequivariance_triangular_attn -- the stock Attention module's call (Pairformer / MSA-module /
        confidence-head / template pair stacks on the stock path), the block statement's full, row-chunked and lean cores, fpf_pad8exact;
  pad8  fpf_cueq_pad8exact._cue_tri -- the padded-buffer provider's own library handle (PTX_E_PAD8 rows), bound when that module is loaded.
A call outside the convention (extra arguments, operand ranks that do not map, bias or mask axes that are not the convention's broadcast axes)
goes to the site's callable unchanged by declared route (``passthrough``, counted by reason) -- never guessed at.

Switch ``PTX_TRIATT_EXACT=1`` (modes.README_ROWS: the exact post exports of the cc-9.0 and cc-8.0 rows; absent / 0 = nothing touched; any other
value raises by name).  Hook: ptx_trunk2_levers.apply_from_env imports this module after the block levers (so a loaded pad8 provider is bound
too) and appends the marker :func:`apply` returns (protenix_opt.stack.MARKERS ``TRIATT_EXACT:on(``); a bind failure is written by the hook as
``TRIATT_EXACT:unavailable(<exc>)`` (BAD: the kit refuses the mode by name).  opt_core older than ``FLOOR`` -- the first core whose
kernels.triattn carries the row and its ``exact_member`` glue -- raises by name.  :func:`unapply` puts the found callables back.

Evidence: :func:`report` is the trunk record's ``triatt_exact`` entry (protenix_opt.report reads it for the LEVER line): word, bind, sites,
calls, provider (calls handed to the face), refused {kind: n}, passthrough {reason: n}, the member's counts and its own line
(``triattn_exact: served n/m calls (refused: {...})``), the install-time selection per probe size (``select``: the row the word resolves to on
this device's exact stack; nothing is served there), the exact stack key and the core version.  :func:`exit_line` is the one
``[FPF] triatt_exact: ...`` stderr line printed at interpreter exit.
"""
from __future__ import annotations

import atexit
import functools
import importlib
import os
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple

NAME = "triatt_exact"                          # registry lever (protenix_opt.registry) and the trunk record key
SWITCH = "PTX_TRIATT_EXACT"
WORD = "exact"                                 # the provider's tier word this lever binds; nothing else is ever asked
FACE = "opt_core.kernels.triattn"
MARK = "TRIATT_EXACT:"                         # protenix_opt.stack.MARKERS family: 'TRIATT_EXACT:on(' = installed
FLOOR = (0, 5, 225, 0)                         # oldest opt_core whose kernels.triattn carries row triattn_exact and exact_member
ROW = "triattn_exact"                          # the provider's exact-class kernel row (named on the marker where a probe size resolves to it)
HEAD_DIM = 32                                  # the pair stacks' per-head width (trunk H8, template stack H2)
PROBE_HEADS = 8
PROBE_SIZES = (256, 512, 1536)                 # token counts the install-time selection is stated at (marker / LEVER line select=)
TL_MODULE, TL_NAME = "protenix.model.triangular.layers", "cuequivariance_triangular_attn"
PAD8_MODULE, PAD8_NAME = "fpf_cueq_pad8exact", "_cue_tri"


def _fresh() -> Dict[str, Any]:
    return {"installed": False, "marker": None, "word": WORD, "bind": f"tier:{WORD}", "opt_core": None, "cc": None, "exact_stack": None,
            "select": {}, "exact_rows": [], "describe": None, "sites": [], "site_calls": {}, "calls": 0, "provider": 0, "refused": {}, "passthrough": {}}


STATE: Dict[str, Any] = _fresh()
_BOUND: Dict[str, Dict[str, Any]] = {}         # site -> {"owner": module, "attr": name, "orig": callable, "wrapper": callable}
_EXIT = {"registered": False}


# ----------------------------------------------------------------------------------------------------------------- switch / core
def enabled(environ=None) -> bool:
    """PTX_TRIATT_EXACT: absent / 0 -> False; 1 -> True; anything else raises by name."""
    v = (os.environ if environ is None else environ).get(SWITCH, "")
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{SWITCH}={v!r}: 1 (the pair stacks' triangle attention through {FACE} by the word {WORD}) or unset/0")
    return True


def _version_tuple(text) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(text))[:4])


def core_version() -> str:
    import opt_core
    return str(getattr(opt_core, "__version__", "0"))


def _face():
    """(T, EM): the provider module and its exact-member glue.  Raises RuntimeError by name below FLOOR or when the row is not carried."""
    v = core_version()
    if _version_tuple(v) < FLOOR:
        raise RuntimeError(f"{NAME}: opt_core {v} is older than {'.'.join(str(x) for x in FLOOR)} (the first core whose kernels.triattn carries row {ROW})")
    from opt_core.kernels import triattn as T
    from opt_core.kernels.triattn import exact_member as EM
    if ROW not in tuple(getattr(T, "ROW_NAMES", ())):
        raise RuntimeError(f"{NAME}: {FACE} of opt_core {v} carries no row {ROW}")
    return T, EM


def _count(d: Dict[str, int], key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def _kind(exc) -> str:
    """One token for a face refusal: its kind up to the first ':' (no blanks, no parentheses)."""
    k = str(getattr(exc, "kind", None) or type(exc).__name__).split(":", 1)[0]
    return re.sub(r"[\s()]+", "_", k) or "refusal"


# ----------------------------------------------------------------------------------------------------------------- layout
def _five(q, k, v, bias, mask):
    """One call in the provider's convention: ((q5, k5, v5, bias5, mask5), None), or (None, <passthrough reason>).  Leading axes are added
    with unsqueeze (views, never copies); the output keeps the library's own rank ([B,N,H,S,D], B=1 for a 4-D statement call)."""
    dim = getattr(q, "dim", None)
    if dim is None or not hasattr(k, "dim") or not hasattr(v, "dim"):
        return None, "operands"
    r = q.dim()
    if r not in (4, 5) or k.dim() != r or v.dim() != r:
        return None, "rank"
    if bias is None or not hasattr(bias, "dim") or not 3 <= bias.dim() <= 5:
        return None, "bias_rank"
    if mask is not None and (not hasattr(mask, "dim") or not 4 <= mask.dim() <= 5):
        return None, "mask_rank"
    lead = 5 - r
    q5, k5, v5 = (t if lead == 0 else t.unsqueeze(0) for t in (q, k, v))
    b5 = bias
    while b5.dim() < 5:
        b5 = b5.unsqueeze(0)
    B, N, H, S, D = q5.shape
    if b5.shape[0] != B or b5.shape[1] != 1 or b5.shape[2] != H or b5.shape[3] != S or b5.shape[4] != k5.shape[3]:
        return None, "bias_layout"
    m5 = None
    if mask is not None:
        m5 = mask if mask.dim() == 5 else mask.unsqueeze(0)
        if m5.shape[0] != B or m5.shape[1] != N or m5.shape[2] != 1 or m5.shape[3] != 1 or m5.shape[4] != k5.shape[3]:
            return None, "mask_layout"
    return (q5, k5, v5, b5, m5), None


# ----------------------------------------------------------------------------------------------------------------- binding
def _bind(site: str, owner, attr: str, keyword: bool, T) -> None:
    """Wrap ``owner.<attr>`` once for ``site``; the callable found there is the site's ``stock`` (``keyword``: it takes mask= / scale= by
    keyword -- the library's own signature; else the engine's positional (q, k, v, bias, mask, scale))."""
    orig = getattr(owner, attr, None)
    if not callable(orig):
        raise RuntimeError(f"{NAME}: site {site}: {getattr(owner, '__name__', owner)}.{attr} is not a callable")
    if getattr(orig, "_ptx_triatt_exact", False):              # bound already in this process: keep that binding
        return
    if keyword:
        def stock(Q, K, V, Bi, mask=None, scale=None, _o=orig):
            return _o(Q, K, V, Bi, mask=mask, scale=scale)
    else:
        def stock(Q, K, V, Bi, mask=None, scale=None, _o=orig):
            return _o(Q, K, V, Bi, mask, scale)
    Refusal = T.Refusal
    serve = T.triangle_attention

    @functools.wraps(orig)
    def bound(q, k, v, bias, mask=None, scale=None, *rest, **kw):
        STATE["calls"] += 1
        _count(STATE["site_calls"], site)
        if rest or kw:                                            # not the six-operand convention: the site's callable, unchanged
            _count(STATE["passthrough"], "arguments")
            return orig(q, k, v, bias, mask, scale, *rest, **kw)
        five, why = _five(q, k, v, bias, mask)
        if five is None:
            _count(STATE["passthrough"], why)
            return stock(q, k, v, bias, mask=mask, scale=scale)
        q5, k5, v5, b5, m5 = five
        STATE["provider"] += 1
        try:
            out = serve(q5, k5, v5, b5, m5, scale, word=WORD, stock=stock)
        except Refusal as e:                                      # structural, by name, before anything ran: this call takes the site's callable
            _count(STATE["refused"], _kind(e))
            out = stock(q5, k5, v5, b5, mask=m5, scale=scale)
        return out                                                     # the library's own rank: [1,N,H,S,D] for a 4-D statement call, as the site returned before

    bound._ptx_triatt_exact = True
    bound._ptx_triatt_exact_site = site
    setattr(owner, attr, bound)
    _BOUND[site] = {"owner": owner, "attr": attr, "orig": orig, "wrapper": bound}


def _unbind(site: str) -> bool:
    b = _BOUND.pop(site, None)
    if b is None:
        return False
    if getattr(b["owner"], b["attr"], None) is b["wrapper"]:
        setattr(b["owner"], b["attr"], b["orig"])
        return True
    return False


def _probe(T) -> None:
    """The install-time selection stated on the marker / LEVER line: per probe size the row the word resolves to on this device's exact stack
    (a refusal by name reads <fallback>[<kind>]).  No CUDA device in the process: nothing to key, the calls decide."""
    torch = sys.modules.get("torch")
    try:
        has_cuda = bool(torch is not None and torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        has_cuda = False
    if not has_cuda:
        STATE.update(cc=None, exact_stack=None, select={}, exact_rows=[], describe=None)
        return
    cc = tuple(torch.cuda.get_device_capability())
    try:
        xs = T.exact_stack_key(cc)
    except Exception as e:  # noqa: BLE001
        xs = f"unkeyed[{_kind(e)}]"
    select, describe = {}, None
    for S in PROBE_SIZES:
        try:
            sel = T.select(cc, "bf16", HEAD_DIM, PROBE_HEADS, S, "fwd", word=WORD, exact_stack=xs)
            select[S] = str(sel.row)
            if describe is None:
                describe = str(T.describe(sel))
        except T.Refusal as e:
            select[S] = f"{e.fallback or 'stock'}[{_kind(e)}]"
    try:
        classes = {name: str(rec.get("class")) for name, rec in T.rows().items()}
    except Exception:  # noqa: BLE001
        classes = {}
    exact_rows = sorted({r for r in select.values() if classes.get(r) == "exact"})
    STATE.update(cc=f"{cc[0]}.{cc[1]}", exact_stack=xs, select=select, exact_rows=exact_rows, describe=describe)


def _select_token(select: Dict[Any, str]) -> str:
    return ",".join(f"S{s}:{r}" for s, r in select.items()) or "none"


def _marker() -> str:
    if STATE["exact_rows"]:
        sizes = sorted(int(s) for s, r in STATE["select"].items() if r in STATE["exact_rows"])
        note = (f"exact-class row {','.join(STATE['exact_rows'])} at S{'/S'.join(str(s) for s in sizes)} of the probe sizes on {STATE['exact_stack']} "
                f"(bit-identical to the library op; a per-call refusal by name takes the library op)")
    elif STATE["cc"] is None:
        note = "no cuda device in this process: the provider decides per call"
    else:
        note = f"no cell vouches an exact-class row on {STATE['exact_stack']}: every call takes the library op BY NAME through the provider"
    return (f"{MARK}on({FACE} {STATE['opt_core']} word={WORD} bind=tier:{WORD} sites={'+'.join(STATE['sites'])} cc={STATE['cc'] or 'none'} "
            f"select={_select_token(STATE['select'])}; {note}; EXACT-BITWISE vs the library triangle_attention D={HEAD_DIM})")


def apply(environ=None) -> Optional[str]:
    """Bind the sites when the switch is set and return the ``TRIATT_EXACT:on(...)`` marker (the same one on a later call); None with nothing
    touched when the switch is absent / 0.  Raises by name when it cannot bind here (the hook names it on the marker; the kit refuses the mode)."""
    if not enabled(environ):
        return None
    if STATE["installed"]:
        return STATE["marker"]
    T, _EM = _face()
    TL = importlib.import_module(TL_MODULE)
    sites = ["tl"]
    _bind("tl", TL, TL_NAME, keyword=False, T=T)
    P8 = sys.modules.get(PAD8_MODULE)                             # the padded-buffer provider holds its own library handle: bound only when loaded
    if P8 is not None and callable(getattr(P8, PAD8_NAME, None)):
        try:
            _bind("pad8", P8, PAD8_NAME, keyword=True, T=T)
        except Exception:
            _unbind("tl")
            raise
        sites.append("pad8")
    STATE.update(installed=True, sites=sites, opt_core=core_version())
    _probe(T)
    STATE["marker"] = _marker()
    if not _EXIT["registered"]:
        _EXIT["registered"] = True
        atexit.register(_print_exit_line)
    return STATE["marker"]


def unapply() -> bool:
    """Put the found callables back at every bound site and forget the record (a fresh apply binds again).  True when a site was restored."""
    restored = False
    for site in list(_BOUND):
        restored = _unbind(site) or restored
    STATE.clear()
    STATE.update(_fresh())
    return restored


# ----------------------------------------------------------------------------------------------------------------- evidence
def member_counts() -> Optional[Dict[str, Any]]:
    """The exact member's own counts in this process ({served, calls, refused {word: n}, installed}) or None when the core lacks it."""
    try:
        from opt_core.kernels.triattn import exact_member as EM
        c = EM.counts()
    except Exception:  # noqa: BLE001
        return None
    return {"served": int(c.get("served") or 0), "calls": int(c.get("calls") or 0), "refused": dict(c.get("refused") or {}),
            "installed": bool(c.get("installed")), "line": EM.evidence_line()}


def report() -> Dict[str, Any]:
    """Plain data: the trunk record's ``triatt_exact`` entry and the source of protenix_opt.report's LEVER pairs."""
    out: Dict[str, Any] = {"unit": NAME}
    for k, v in STATE.items():
        out[k] = dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
    out["select"] = {str(s): r for s, r in STATE["select"].items()}
    m = member_counts() if STATE["installed"] else None
    out["member"] = {k: m[k] for k in ("served", "calls", "refused", "installed")} if m else None
    out["member_line"] = m["line"] if m else None
    return out


def _pairs_token(d: Optional[Dict[str, int]]) -> str:
    return ",".join(f"{k}:{n}" for k, n in sorted((d or {}).items())) or "none"


def exit_line() -> str:
    """``[FPF] triatt_exact: <the member's line> sites=.. calls=.. provider=.. refused=.. passthrough=.. select=.. exact_stack=.. opt_core=..``."""
    r = report()
    return (f"[FPF] {NAME}: {r.get('member_line') or ROW + ': no member record'} sites={'+'.join(r.get('sites') or []) or 'none'} "
            f"calls={r.get('calls', 0)} provider={r.get('provider', 0)} refused={_pairs_token(r.get('refused'))} passthrough={_pairs_token(r.get('passthrough'))} "
            f"select={_select_token(r.get('select') or {})} exact_stack={r.get('exact_stack') or 'none'} opt_core={r.get('opt_core')}")


def _print_exit_line() -> None:
    if STATE.get("installed"):
        try:
            print(exit_line(), file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass
