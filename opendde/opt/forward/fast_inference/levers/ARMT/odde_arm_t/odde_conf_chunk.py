"""odde_arm_t.odde_conf_chunk -- lever ``triattn_conf`` on the confidence head's CHUNKED and ROW-BLOCK attention paths.

The arm (``odde_arm_t``) marks the confidence head's pair stack by a forward pre-hook on ``model.confidence_head.pairformer_stack``: inside that
module's own ``forward`` every triangle-attention call reaches the provider binding (``odde_triattn_bind.serve``: the TIER WORD's row per cell)
with ``STATE["in_conf"] > 0`` -- counted ``att_conf_calls`` / ``att_conf_prov_calls``, and the lever's ablation (``ODDE_TRIATTN_CONF=stock``) keeps
that stack on the stock op BY NAME (``asides=conf_stock``). Upstream's chunk loop (``TriangleAttention._chunk`` -> ``chunk_layer`` -> ``self.mha``
per row chunk) runs INSIDE that forward, so it is marked. The offload unit's confidence stage is not: above its size gate on the big lines
``levers/OFFLOAD`` ``off_confidence_head`` -> ``off_pairformer_stack`` -> ``off_triatt`` walks the stack's BLOCKS over host-streamed row blocks and
issues ``block.tri_att_{start,end}.mha(q_x=<rows>, kv_x=<rows>, biases=[<mask rows>, <full triangle bias>])`` once per row block -- the same
``Attention.forward`` -> ``layers.cuequivariance_triangular_attn`` site the arm binds, reached WITHOUT the stack-level mark (the stack's ``forward``
never runs) -- without the mark those calls would be served unattributed and the lever's must-run counter would read 0 there.

This unit binds the mark one level down, where every path meets: a forward pre/post hook pair on each ``blocks[i].tri_att_{start,end}.mha``
(upstream's ``Attention`` module) of the confidence head's pair stack. Per call it raises the arm's ``in_conf`` for the duration of the call --
so the provider serves each row block by the tier word with that block's mask slice and the full triangle bias under the SAME rule as the
module path (a refusal serves the provider's NAMED fallback row or the stock op by the cell's name; the ablation keeps the stock op by name) --
and keeps the census of the path that issued the call: ``module`` (the stack's forward is on the call stack: the arm's mark was already up;
upstream's statement, whole or chunked) or ``offload_rows`` (a driver outside that forward walked the blocks: the offload unit's confidence
stage, one call per row block), and per path whether a provider row served the call (``core``: the arm's ``att_conf_prov_calls`` moved across
it), the stock op served it by a named rule (``stock``: ``att_conf_calls`` moved, the provider count did not), or the arm's wrapper handed it to
the stock op before the confidence mark / never ran (``unmarked``: below the arm's design gate, no provider word, or a ``triangle_attention``
implementation other than cuequivariance -- upstream's torch statement). ``rows`` = the row-block sizes of the offload-row calls a provider row
served, ``blocks`` = the offload row blocks seen. Nothing here chooses a kernel, a row or a launch setting, allocates, or changes a shape: the
row blocks, their slices and the peak are the driver's; the row is the provider's (``opt_core.kernels.triattn.select`` by the word).

No switch of its own: bound by the arm's ``bind(model)`` whenever the arm binds a model (the mark is read only with the provider word live);
left out with its lever (``MODEL_OPT_LEVERS_OFF=triattn_conf`` -> ``ODDE_TRIATTN_CONF=stock``: every marked call is the stock op's by name, on
every path alike). Nothing under site-packages is edited.
"""
import atexit
import json
import sys

__version__ = "0.1.0"

PATHS = ("module", "offload_rows")
COUNTS = {"version": __version__, "bound": 0, "models": 0, "calls": 0, "blocks": 0,
          "paths": {}, "core": {}, "stock": {}, "unmarked": {}, "rows": {}, "module_chunks": 0, "errors": {}}
#   bound          Attention modules of the confidence head's pair stack carrying the hook pair (4 blocks x tri_att_start/end = 8 per model)
#   calls          marked calls, all paths;  paths / core / stock / unmarked: per path (PATHS) -- see the module docstring
#   blocks         offload row blocks seen (== paths["offload_rows"]);  rows: {row-block size: n} of the offload-row calls a provider row served
#   module_chunks  calls on the module path whose query rows were a chunk of the pair (upstream's chunk loop under the stack's own forward)
_BOUND = set()          # id() of the pair stacks already carrying the hooks
_HOOKS = []             # torch hook handles (removable; tests)
_STACK = []             # per-call (path, att_conf_calls before, att_conf_prov_calls before, rows) from the pre-hook to the post-hook (calls never nest)


def _log(msg):
    print(f"[odde_conf_chunk] {msg}", file=sys.stderr, flush=True)


def _bump(key, sub=None, n=1):
    if sub is None:
        COUNTS[key] = int(COUNTS.get(key) or 0) + n
    else:
        d = COUNTS.setdefault(key, {})
        d[sub] = int(d.get(sub) or 0) + n


def _arm():
    return sys.modules.get("odde_arm_t")


def _form(args, kwargs):
    """(query rows, keys) of an Attention call: q_x is [*, rows, keys, c] (a row block / chunk: rows < keys; the whole pair: rows == keys)."""
    q = (kwargs or {}).get("q_x")
    if q is None and args:
        q = args[0]
    try:
        return (int(q.shape[-3]) if q.dim() >= 3 else 1), int(q.shape[-2])
    except Exception:  # noqa: BLE001
        return None, None


def _pre(mod, args, kwargs=None):
    A = _arm()
    st = getattr(A, "STATE", None)
    c = getattr(A, "COUNTS", None) or {}
    path = "module" if (st is not None and int(st.get("in_conf") or 0) > 0) else "offload_rows"   # the stack's own forward already raised the mark, or a driver outside it issued the call
    rows, keys = _form(args, kwargs)
    _STACK.append((path, int(c.get("att_conf_calls") or 0), int(c.get("att_conf_prov_calls") or 0), rows, keys))
    if st is not None:
        st["in_conf"] = int(st.get("in_conf") or 0) + 1
    return None


def _post(mod, args, *rest):
    A = _arm()
    st = getattr(A, "STATE", None)
    if st is not None and int(st.get("in_conf") or 0) > 0:
        st["in_conf"] -= 1
    if not _STACK:
        _bump("errors", "unpaired_post")
        return None
    path, calls0, prov0, rows, keys = _STACK.pop()
    c = getattr(A, "COUNTS", None) or {}
    _bump("calls"); _bump("paths", path)
    if path == "offload_rows":
        _bump("blocks")
    elif rows is not None and keys is not None and rows < keys:
        _bump("module_chunks")
    if int(c.get("att_conf_prov_calls") or 0) > prov0:                     # a provider row served this call inside the confidence mark
        _bump("core", path)
        if path == "offload_rows" and rows is not None:
            _bump("rows", rows)
    elif int(c.get("att_conf_calls") or 0) > calls0:                       # the stock op by a named rule (cell_stock / a refusal's stock fallback / the ablation conf_stock)
        _bump("stock", path)
    else:                                                                  # the arm's wrapper served stock before the mark (design gate, no word, dtype / rank) or never ran (triangle_attention != cuequivariance)
        _bump("unmarked", path)
    return None


def bind(model) -> dict:
    """Hook the confidence head's pair stack's triangle-attention modules (``confidence_head.pairformer_stack.blocks[*].tri_att_{start,end}.mha``).
    Idempotent per stack; returns this unit's bind record for the arm's census."""
    stack = getattr(getattr(model, "confidence_head", None), "pairformer_stack", None)
    if stack is None:
        COUNTS["errors"]["bind"] = "model has no confidence_head.pairformer_stack"
        return {"bound": 0, "reason": COUNTS["errors"]["bind"]}
    if id(stack) in _BOUND:
        return {"bound": COUNTS["bound"], "already": True}
    n = 0
    for blk in list(getattr(stack, "blocks", None) or []):
        for name in ("tri_att_start", "tri_att_end"):
            mha = getattr(getattr(blk, name, None), "mha", None)
            if mha is None or not hasattr(mha, "register_forward_pre_hook"):
                continue
            try:
                _HOOKS.append(mha.register_forward_pre_hook(_pre, with_kwargs=True))
            except TypeError:                                              # torch < 2.0: positional-only hooks (q_x unreadable there: rows census None, the mark intact)
                _HOOKS.append(mha.register_forward_pre_hook(lambda m, a: _pre(m, a, None)))
            try:
                _HOOKS.append(mha.register_forward_hook(_post, always_call=True))   # the mark comes down even when the call raises
            except TypeError:
                _HOOKS.append(mha.register_forward_hook(_post))
            n += 1
    _BOUND.add(id(stack))
    COUNTS["bound"] += n; COUNTS["models"] += 1
    if n == 0:
        COUNTS["errors"]["bind"] = "confidence_head.pairformer_stack has no blocks[*].tri_att_{start,end}.mha"
    _log(f"bound {n} triangle-attention modules of the confidence head's pair stack ({len(list(getattr(stack, 'blocks', None) or []))} blocks): "
         f"upstream's chunk loop and the offload unit's row blocks are marked for lever triattn_conf")
    return {"bound": n}


def unbind():
    """Remove every hook (tests)."""
    for h in _HOOKS:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    _HOOKS.clear(); _BOUND.clear(); _STACK.clear()
    COUNTS["bound"] = 0; COUNTS["models"] = 0


def reset_counts():
    for k in ("calls", "blocks", "module_chunks"):
        COUNTS[k] = 0
    for k in ("paths", "core", "stock", "unmarked", "rows"):
        COUNTS[k] = {}


def describe() -> dict:
    """This unit's census for the LEVER line (report.lever_evidence: conf_path / rows / blocks) and the ran-or-refuse predicate (registry)."""
    rows = dict(sorted(((int(k), int(v)) for k, v in (COUNTS.get("rows") or {}).items()), key=lambda kv: (-kv[1], kv[0])))
    return {"version": __version__, "bound": int(COUNTS.get("bound") or 0), "calls": int(COUNTS.get("calls") or 0), "blocks": int(COUNTS.get("blocks") or 0),
            "paths": dict(COUNTS.get("paths") or {}), "core": dict(COUNTS.get("core") or {}), "stock": dict(COUNTS.get("stock") or {}),
            "unmarked": dict(COUNTS.get("unmarked") or {}), "rows": rows, "module_chunks": int(COUNTS.get("module_chunks") or 0),
            "errors": {k: str(v)[:160] for k, v in (COUNTS.get("errors") or {}).items()}}


def conf_path() -> str | None:
    """The one census word of the path that served the confidence head's triangle attention this process: ``offload_rows_core`` (the offload unit's
    row blocks, a provider row served >= 1 of them), ``offload_rows_stock`` (row blocks, every one the stock op's by a named rule), ``offload_rows_unmarked``
    (row blocks the arm's wrapper never marked: stock before the mark or another attention implementation), ``module`` (the stack's own forward), None (no call)."""
    p = COUNTS.get("paths") or {}
    if p.get("offload_rows"):
        if (COUNTS.get("core") or {}).get("offload_rows"):
            return "offload_rows_core"
        if (COUNTS.get("stock") or {}).get("offload_rows"):
            return "offload_rows_stock"
        return "offload_rows_unmarked"
    if p.get("module"):
        return "module"
    return None


if not getattr(sys, "_odde_conf_chunk_atexit", False):                   # one census line per process (a re-import in the same process does not stack handlers)
    sys._odde_conf_chunk_atexit = True
    atexit.register(lambda: _log("COUNTS@exit " + json.dumps(sys.modules[__name__].describe() if hasattr(sys.modules.get(__name__), "describe") else describe(), default=str, sort_keys=True)))
