"""fpf_rf3_triattn — RoseTTAFold3's triangle-attention calls on the shared core's ONE provider (``opt_core.kernels.triattn``).

Two seams, each a class-level re-binding switched from the FPF arm (``fpf_rf3_adapter.apply_arm``) and undone by it, no file or weight edit:

* ``gflash[.<word>]`` — the fast tier's fused triangle attention (``fpf_rf3_adapter._tri_forward_v2``: LN+cast(+transpose)+to_b prologue, one
  q|k|v|g GEMM, the attention core, the gate epilogue).  The attention core is the provider's row named by the sub-word: ``flash`` (the default:
  the flash_triattn Triton kernel), ``k2b`` / ``k2`` (the K2B Triton kernels), ``cuda_sm90a`` (the sealed sm_90a extension: bf16, head_dim 32, cc 9.0 and a prebuilt for this interpreter's ABI — refused BY NAME
  ``no_prebuilt:<stack key>`` otherwise and bound to the cell's fallback row, k2b), ``fast`` (the row the cell table names for this card).
* ``xatt[.<word>]`` — the exact tier: the STOCK ``TriangleAttention`` statement untouched (its LayerNorm, projections, casts, rearranges, gate) with
  its one ``cuet.triangle_attention`` call served by the provider's exact row: word ``exact`` (default) = the cell's exact row —
  ``exact_headsplit`` where the core's table names it (per-head stock calls above its split size; the stock bytes below), else the stock op
  itself BY NAME (``cueq``) — or a row word (``exact_headsplit`` | ``cueq``).  The seam is the module attribute ``rf3.model.layers.attention.cuet``: a proxy whose
  ``triangle_attention`` is the routed call and whose every other name is the library's (the triangle multiplication beside it is untouched).

Every selection is the provider's (``select`` / ``describe``): one ``[fpf_rf3 triattn] SELECT`` line per distinct (component, word, dtype, heads,
head_dim, size bucket) names the row that serves, its cell and — when the asked row refused — the refusal word and the fallback row bound instead
(never a silent substitute); ``census()`` carries the per-row call counts for the adapter's ``describe_v2()`` and the kit's LEVER line.  A row that
raises anything but a refusal on a call is counted as an error once (printed) and that call is served by the fallback row (a prediction never dies of
it; an out-of-memory propagates — it is the caller's to see).
"""
import os
import sys

COMPONENT_WORDS = {                                      # arm sub-words per component: <component>.<word>; the first is the default (no sub-word)
    "gflash": ("card", "flash", "k2b", "k2", "cuda_sm90a", "triattn_native", "fast"),
    "xatt": ("exact", "exact_headsplit", "cueq"),
}
DEFAULT_WORD = {c: w[0] for c, w in COMPONENT_WORDS.items()}
TIER_DEFAULT = {"gflash": "fast"}                        # the word a bare component (no sub-word: the word "card") asks of the provider on EVERY device: the FAST
                                                         # tier word -- the provider's cells name the row per card / dtype / head_dim / heads / size class
                                                         # (opt_core's CUDA triangle-attention kernel on sm_90, the sm_90a kernel, K2B, the flash Triton kernel,
                                                         # or the stock op BY NAME where the table names no kernel row); no per-card table in this kit
FORM = "bias_only"                                       # RF3's call form at both sites: q/k/v + the pair bias (1,1,H,N,N), no mask (rf3/model/layers/attention.py
                                                         # `cuet.triangle_attention(query, key, value, bias=bias_cueq, scale=self.scaling)`; the adapter's gflash
                                                         # component hands the same operands) -- the provider's `form=` hint (its sub-512-token cells differ by form)


def card_word(comp, cc=None):
    """The provider word a bare ``comp`` asks on any device: its tier word (``TIER_DEFAULT``)."""
    return TIER_DEFAULT.get(comp, COMPONENT_WORDS[comp][-1])


PREFIX = "[fpf_rf3 triattn]"

STATE = {
    "gflash": {"word": DEFAULT_WORD["gflash"], "on": False},
    "xatt": {"word": DEFAULT_WORD["xatt"], "on": False},
}
_SEL = {}                                                # (component, cc, dtype, D, H, S-bucket-free S, direction) -> entry (selection, refusal, counts)
CENSUS = {"gflash": {}, "xatt": {}}                      # component -> {"<row>|<dtype>|D<d>|H<h>": calls}; plus "refused:<row>:<kind>-><fallback>", "error:<row>"
ERRORS = {"n": 0, "first": None}
_ORIG = {}                                               # "cuet": the library module rf3's attention.py imported; "stock": its triangle_attention


class WordError(ValueError):
    """An arm sub-word outside the component's vocabulary (``COMPONENT_WORDS``) — refused by name at ``apply_arm`` time."""


def parse_component(token: str):
    """'gflash' -> ('gflash', None); 'gflash.k2b' -> ('gflash', 'k2b'); 'xatt.exact_headsplit' -> ('xatt', 'exact_headsplit'); (None, None) for a
    token that is not one of this module's components; WordError for a known component with an unknown (or empty) word."""
    base, dot, word = str(token).partition(".")
    if base not in COMPONENT_WORDS:
        return None, None
    if not dot:
        return base, None
    if word not in COMPONENT_WORDS[base]:
        raise WordError(f"FPF arm component {token!r}: word {word!r} is not one of {COMPONENT_WORDS[base]} ({base}.<word>)")
    return base, word


def strip_components(arm: str):
    """Split an arm string into (the arm without this module's sub-words / xatt, {"gflash": <sub-word or None>, "xatt": <word or None when absent>}).
    ``gflash.k2b`` is handed on as ``gflash`` (the adapter's own component: prologue / GEMM / epilogue unchanged); ``xatt[.w]`` is removed (this
    module installs it; a bare ``xatt`` is the default word)."""
    body, at, tail = arm.partition("@")
    words = {"gflash": None, "xatt": None}
    keep = []
    for p in [x for x in body.split("+") if x]:
        base, word = parse_component(p)
        if base is None:
            keep.append(p)
            continue
        if base == "gflash":
            words["gflash"] = word
            keep.append("gflash")
        else:
            words["xatt"] = word or DEFAULT_WORD["xatt"]
    out = "+".join(keep)
    if not out:
        out = "stock"
    return out + (at + tail if at else ""), words


# ----------------------------------------------------------------------------------------------------------------- the provider
def _T():
    from opt_core.kernels import triattn as T
    return T


def _stock():
    """The stock op (cuequivariance_torch.triangle_attention) of this process, or None when the library is absent."""
    if "stock" in _ORIG:
        return _ORIG["stock"]
    try:
        A = sys.modules.get("rf3.model.layers.attention")
        cuet = getattr(A, "cuet", None) if A is not None else None
        real = getattr(cuet, "_real", cuet)
        fn = getattr(real, "triangle_attention", None) if real is not None else None
        if fn is None:
            import cuequivariance_torch as cqt
            fn = cqt.triangle_attention
    except Exception:
        fn = None
    _ORIG["stock"] = fn
    return fn


def _stack_key(cc, word, prefer=None):
    """The cuda_sm90a ABI key of this process when the word could select that row on this card (what the face itself computes), else None."""
    if tuple(cc) != (9, 0):
        return None
    if not (word.split("@")[0] == "cuda_sm90a" or (prefer and "cuda_sm90a" in prefer) or (word == "fast" and not prefer)):
        return None
    try:
        from opt_core.kernels.triattn import cuda_sm90a as C
        return C.stack_key()
    except Exception as e:                                # pragma: no cover - the module is pure python at import
        return f"unimportable:{type(e).__name__}"


def _bucket(T, sel, S):
    return sel.cell.split("|")[4] if (sel is not None and sel.cell) else f"N={S}"


def _select(comp, facts):
    """The provider's selection for (component word, tensor facts), memoised; a refusing row is bound to its named fallback and the refusal kept."""
    T = _T()
    cc, dtype, D, H, S, direction = facts
    word = STATE[comp]["word"]
    if word == "card":                                    # a bare component: its tier word (TIER_DEFAULT)
        word = card_word(comp, cc)
    key = (comp, word, tuple(cc), dtype, D, H, S, direction)
    ent = _SEL.get(key)
    if ent is not None:
        return ent
    stack = _stack_key(cc, word)
    refusal = None
    hint = _form_kw(T)
    try:
        sel = T.select(cc, dtype, D, H, S, direction, word=word, stack=stack, **hint)
    except T.Refusal as r:
        refusal = r
        fb = r.fallback or "cueq"
        try:
            sel = T.select(cc, dtype, D, H, S, direction, word=fb, stack=stack, **hint)
        except T.Refusal as r2:                           # the fallback itself refuses (never for the table's own fallbacks): the stock op by name
            refusal = r2
            sel = T.select(cc, dtype, D, H, S, direction, word="cueq")
    ent = {"sel": sel, "refusal": refusal, "calls": 0, "key": key, "call_refusal": None}
    _SEL[key] = ent
    _say_select(comp, ent, S)
    return ent


_FORM_KW = {}


def _form_kw(T):
    """``{"form": FORM}`` when the core's ``select`` takes the call-form hint, else ``{}``."""
    if "kw" not in _FORM_KW:
        try:
            import inspect
            _FORM_KW["kw"] = {"form": FORM} if "form" in inspect.signature(T.select).parameters else {}
        except Exception:                                 # pragma: no cover
            _FORM_KW["kw"] = {}
    return _FORM_KW["kw"]


def _say_select(comp, ent, S):
    T = _T()
    sel, r = ent["sel"], ent["refusal"] or ent.get("call_refusal")
    cc, dtype, D, H = ent["key"][2], ent["key"][3], ent["key"][4], ent["key"][5]
    asked = STATE[comp]["word"]
    if asked == "card":
        asked = f"card->{ent['key'][1]}"                 # the bare component's tier word (TIER_DEFAULT), then what the provider bound
    line = (f"{PREFIX} SELECT component={comp} asked={asked} form={FORM} cc={cc[0]}.{cc[1]} dtype={dtype} H={H} D={D} n={S} "
            f"{T.describe(sel)}")
    if r is not None:
        line += f" REFUSED row={r.row} kind={r.kind} -> bound {sel.row} by name"
    print(line, flush=True)


def _count(comp, ent, dtype, D, H):
    ent["calls"] += 1
    k = f"{ent['sel'].row}|{dtype}|D{D}|H{H}"
    CENSUS[comp][k] = CENSUS[comp].get(k, 0) + 1
    r = ent["refusal"] or ent.get("call_refusal")
    if r is not None:
        rk = f"refused:{r.row}:{r.kind}->{ent['sel'].row}"
        CENSUS[comp][rk] = CENSUS[comp].get(rk, 0) + 1


def serve(comp, q, k, v, bias, mask=None, scale=None):
    """The component's triangle attention call through the provider (cuequivariance calling convention: q/k/v [B,N,H,S,D], bias [B,1,H,S,S]).
    A row's run-time refusal binds the named fallback for this shape.  An error (anything but a refusal or an out-of-memory) is the caller's for
    ``gflash`` (the adapter's own kernel-error path: retry contiguous, else stock for the call, counted) and, for ``xatt`` (the stock statement has
    no handler), counted + printed once here and the call served by the fallback row."""
    from opt_core.oom import is_oom
    T = _T()
    facts = T._tensor_facts(q, k)
    cc, dtype, D, H, S, direction = facts
    ent = _select(comp, facts)
    stock = _stock()
    for _attempt in (0, 1, 2):
        sel = ent["sel"]
        try:
            out = T.triangle_attention(q, k, v, bias, mask, scale, word=sel.word, stock=stock, selection=sel)
        except T.Refusal as r:                            # a run-time refusal of the row (strides / smem / load gate): bind the fallback for this shape, by name
            _rebind(comp, ent, r, S)
            continue
        except Exception as e:
            if is_oom(e):
                raise
            if isinstance(e, (OSError, ImportError)) and ent["calls"] == 0:      # the row's run-time needs absent (Triton, or the prebuilt extension's shared objects): a named refusal
                fb = T.rows().get(sel.row, {}).get("fallback") or "cueq"
                _rebind(comp, ent, T.Refusal(f"runtime_absent:{type(e).__name__}", sel.row, fb, str(e)[:120]), S)
                continue
            if comp != "xatt" or sel.row in ("cueq",):
                raise
            ERRORS["n"] += 1
            CENSUS[comp][f"error:{sel.row}"] = CENSUS[comp].get(f"error:{sel.row}", 0) + 1
            if ERRORS["first"] is None:
                ERRORS["first"] = f"{sel.row}: {type(e).__name__}: {str(e)[:200]}"
                print(f"{PREFIX} ROW ERROR component={comp} row={sel.row} served by the stock op for this call: {e!r}"[:400], flush=True)
            fsel = T.select(cc, dtype, D, H, S, direction, word="cueq")
            out = T.triangle_attention(q, k, v, bias, mask, scale, word="cueq", stock=stock, selection=fsel)
            ent["calls"] += 1
            return out
        _count(comp, ent, dtype, D, H)
        return out
    raise RuntimeError(f"{PREFIX} {comp}: row {ent['sel'].row} refused repeatedly for one shape ({ent.get('call_refusal')})")   # pragma: no cover


def _rebind(comp, ent, refusal, S):
    T = _T()
    cc, dtype, D, H, S_, direction = ent["key"][2:]
    fb = refusal.fallback or "cueq"
    if fb == ent["sel"].row:
        fb = "cueq"
    try:
        ent["sel"] = T.select(cc, dtype, D, H, S_, direction, word=fb)
    except T.Refusal:
        ent["sel"] = T.select(cc, dtype, D, H, S_, direction, word="cueq")
    ent["call_refusal"] = refusal
    _say_select(comp, ent, S)


# ----------------------------------------------------------------------------------------------------------------- the gflash seam
def gflash_kernel(q, k, v, bias=None, mask=None, scale=None):
    """The attention core of the adapter's gflash component (flash_triangle_attention's signature): the provider's row for the ``gflash`` word."""
    return serve("gflash", q, k, v, bias, mask, scale)


def set_gflash_word(word):
    word = word or DEFAULT_WORD["gflash"]
    if word not in COMPONENT_WORDS["gflash"]:
        raise WordError(f"gflash word {word!r} is not one of {COMPONENT_WORDS['gflash']}")
    STATE["gflash"]["word"] = word
    STATE["gflash"]["on"] = True
    return word


# ----------------------------------------------------------------------------------------------------------------- the xatt seam
def cueq_fallback_threshold(default=100):
    """cuEquivariance's own small-sequence rule: ``cuequivariance_ops_torch.triangle_attention.CUEQ_TRIATTN_FALLBACK_THRESHOLD`` — at S <= it the
    STOCK op takes the library's torch reference path (``_triangle_attention_torch``), not its CUDA kernel.
    Read from the loaded module (the constant the kit's kernels census reports as the library's rule); ``default`` when the module or the constant is absent."""
    m = sys.modules.get("cuequivariance_ops_torch.triangle_attention")
    if m is None:
        try:
            import importlib
            m = importlib.import_module("cuequivariance_ops_torch.triangle_attention")
        except Exception:
            m = None
    t = getattr(m, "CUEQ_TRIATTN_FALLBACK_THRESHOLD", None) if m is not None else None
    return int(t) if isinstance(t, int) else int(default)


class CuetProxy(object):
    """``rf3.model.layers.attention.cuet`` with ``triangle_attention`` served by the provider's exact row; every other attribute is the library's.
    Calls at or below the library's own fallback threshold (``cueq_fallback_threshold()``: S <= CUEQ_TRIATTN_FALLBACK_THRESHOLD, where the stock op
    is its torch reference path) go to the stock op BY NAME (census ``passthrough:S<=<t>``): the provider's rows stand in for the CUDA kernel, not that path."""

    def __init__(self, real):
        self._real = real
        self._thr = cueq_fallback_threshold()

    def triangle_attention(self, q, k, v, bias=None, mask=None, scale=None, **kw):
        if kw:                                            # a keyword this seam does not know: the library's own call, counted (never met in rf3's statement)
            CENSUS["xatt"]["passthrough:kwargs"] = CENSUS["xatt"].get("passthrough:kwargs", 0) + 1
            return self._real.triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale, **kw)
        S = int(q.shape[-2]) if hasattr(q, "shape") and len(q.shape) >= 2 else None
        if S is not None and S <= self._thr:                # the library's reference-path sizes: the stock op by name (bitwise with stock by construction)
            key = "passthrough:S<=%d" % self._thr
            CENSUS["xatt"][key] = CENSUS["xatt"].get(key, 0) + 1
            return self._real.triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale)
        return serve("xatt", q, k, v, bias, mask, scale)

    def __getattr__(self, name):
        return getattr(self._real, name)


def enable_xatt(word=None):
    """Install (word given or default) / keep the exact-tier seam: ``attention.cuet`` -> CuetProxy.  Returns the word.  Refuses by name (RuntimeError)
    when rf3's attention module has no ``cuet`` (cuequivariance not in use: nothing to route)."""
    word = word or DEFAULT_WORD["xatt"]
    if word not in COMPONENT_WORDS["xatt"]:
        raise WordError(f"xatt word {word!r} is not one of {COMPONENT_WORDS['xatt']}")
    import rf3.model.layers.attention as A
    cuet = getattr(A, "cuet", None)
    if cuet is None:
        raise RuntimeError("xatt: rf3.model.layers.attention has no `cuet` (cuequivariance_torch not imported by upstream): nothing to route")
    if not isinstance(cuet, CuetProxy):
        _ORIG["cuet"] = cuet
        _ORIG["stock"] = cuet.triangle_attention
        A.cuet = CuetProxy(cuet)
    STATE["xatt"]["word"] = word
    STATE["xatt"]["on"] = True
    return word


def disable_xatt():
    A = sys.modules.get("rf3.model.layers.attention")
    if A is not None and isinstance(getattr(A, "cuet", None), CuetProxy):
        A.cuet = A.cuet._real
    STATE["xatt"]["on"] = False
    return False


def reset_census():
    for c in CENSUS:
        CENSUS[c] = {}
    _SEL.clear()
    ERRORS["n"] = 0; ERRORS["first"] = None


def census(comp=None):
    """{component: {"word", "on", "calls": {row|dtype|D|H: n, refused:...: n, error:...: n}, "served": n, "selections": [describe lines]}}."""
    T = None
    out = {}
    for c in (COMPONENT_WORDS if comp is None else (comp,)):
        sels = []
        for key, ent in _SEL.items():
            if key[0] != c:
                continue
            try:
                T = T or _T()
                d = T.describe(ent["sel"])
            except Exception:                             # pragma: no cover
                d = f"row={ent['sel'].row}"
            r = ent["refusal"] or ent.get("call_refusal")
            sels.append(f"H{key[5]}D{key[4]}n{key[6]}:{d}" + (f" REFUSED {r.row}:{r.kind}" if r is not None else ""))
        served = sum(v for k, v in CENSUS[c].items() if not k.startswith(("refused:", "error:", "passthrough:")))
        word = STATE[c]["word"]
        if word == "card":                                # the bare component: name the row(s) this device's card row resolved to (one per process in practice)
            asked = sorted({key[1] for key in _SEL if key[0] == c})
            word = "card->" + ("|".join(asked) if asked else "?")
        out[c] = {"word": word, "on": bool(STATE[c]["on"]), "calls": dict(CENSUS[c]), "served": int(served), "selections": sels,
                  "errors": sum(v for k, v in CENSUS[c].items() if k.startswith("error:"))}
    return out if comp is None else out[comp]


def census_line(comp):
    """One line for the kit's exit census: ``[fpf_rf3 triattn] CENSUS component=<c> word=<w> served=<n> rows=<row|dtype|D|H:n,...> [refused=...] selections=<...>``."""
    c = census(comp)
    rows = ",".join(f"{k}:{v}" for k, v in sorted(c["calls"].items()) if not k.startswith(("refused:", "error:")))
    ref = ",".join(f"{k[len('refused:'):]}:{v}" for k, v in sorted(c["calls"].items()) if k.startswith("refused:"))
    err = ",".join(f"{k[len('error:'):]}:{v}" for k, v in sorted(c["calls"].items()) if k.startswith("error:"))
    line = f"{PREFIX} CENSUS component={comp} word={c['word']} on={c['on']} served={c['served']} rows={rows or '-'}"
    if ref:
        line += f" refused={ref}"
    if err:
        line += f" errors={c['errors']}({err})"
    line += " selections=" + (" ; ".join(c["selections"]) or "-")
    return line


def print_census():
    for comp in COMPONENT_WORDS:
        if STATE[comp]["on"] or CENSUS[comp]:
            print(census_line(comp), file=sys.stderr, flush=True)


_ATEXIT = {"registered": False}


def register_exit_census():
    if not _ATEXIT["registered"]:
        import atexit
        atexit.register(print_census)
        _ATEXIT["registered"] = True
