"""fpf_rf3_trimul_rows — RoseTTAFold3's triangle-multiplication calls on the shared core's ONE provider (``opt_core.kernels.trimul``).

Two seams, each a class-level re-binding switched from the FPF arm (``fpf_rf3_adapter.apply_arm``) and undone by it, no file or weight edit:

* ``fast[.<word>]`` — the fast tier's fused TriMul (the arm's trimul head word; ``fpf_rf3_adapter._fast_forward``: the stock statement's own
  bf16 cast, then ONE call over the provider's op boundary — LN_in, the a|b projections and gates, the contraction, LN_out, the gated output
  projection; the caller adds the residual, or the ``res`` component fuses it into the row's epilogue).  The row is named by the sub-word:
  ``v4`` (the default = ``fpf_trimul_v4`` ``generic.trimul``),
  ``tmk3_fast`` / ``tmk3_exact`` (the ``fpf_trimul`` kernel line with its fast / exact launch table; c a multiple of 64 — these rows also serve
  the c=64 template pair track every other row steps aside on), ``tx_sm90a`` (the sealed sm_90a package: c 256 only — refused BY NAME for this
  model's widths), ``fast`` / ``big`` (the row the cell table names fastest / lowest-peak for this card
  and stack, OPT-IN).  A call no admitted row serves runs the STOCK statement, counted under the adapter's census word (``N<floor>``: the token
  floor — under a TIER word the provider table's per-class floor (``fast_floor``: ``N<2`` on the cc-9.0 c_z-128 classes whose N<=100 cells name
  row v4, ``N<101`` on every other class and under a ROW word); ``c=<c>/d=<d>``: a pair track the word's row does not serve; ``refused:<row>:<kind>``: a machine / ABI refusal) — never silent.
* ``xmul[.<word>]`` — the exact tier: the STOCK ``TriangleMultiplication`` statement untouched (its cast, its weights, its one
  ``cuet.triangle_multiplicative_update`` call) with that call served by the provider's exact row: word ``exact`` (default) = the cell's exact
  row — ``tmk3_exact`` where the core's table names it for this card (bitwise == the library op; c 128 up to the table's largest size,
  the c=64 template track at every size, cc 9.0 and 8.0), the library op itself BY NAME (``cueq``) in every other cell — or a row word
  (``tmk3_exact`` | ``cueq``), or the scope word ``eager``: the cell's exact row for calls made OUTSIDE CUDA-graph capture (the template
  embedder, the MSA module's pair stack, the confidence head's pairformer, and the whole trunk when no graph arm is on) and the library op BY
  NAME for calls being captured into a graph (the ``tg`` arm's pairformer stack: a replayed graph ranks rows by device time alone, which the
  provider's eager-measured cell table does not carry; both rows are the library op's bits, so a warm-up pass and its captured replay agree to
  the bit).  Calls at N <= the library's own fallback threshold (``CUEQ_TRIMUL_FALLBACK_THRESHOLD``, read from the loaded module: below it the
  stock op is its torch reference path, which no kernel row reproduces bit for bit) go to the stock op BY NAME (census ``passthrough:N<=<t>``;
  the fast seam serves them from the table's floor for the class up and declines them below it — ``fast_floor``).  The seam is the module attribute ``rf3.model.layers.attention.cuet``: a proxy whose
  ``triangle_multiplicative_update`` is the routed call and whose every other name is the object beneath it (the library module, or the
  triangle-attention module's proxy (fpf_rf3_triattn.CuetProxy) when its ``xatt`` component is on — the two seams nest in either order and each undoes only itself).

Selection is the provider's ``select()`` per (component, word, cc, precision, c, d, N, direction), memoised, printed once per new key as
``[fpf_rf3 trimul] SELECT component=<c> asked=<word> ... <the provider's describe()>``; a row that refuses this card / stack / shape (its
``admits()`` or a run-time ``Refusal``) is bound to the refusal's NAMED fallback row for that key and the line says so.  Per-row call counts:
``census()`` / ``census_line()`` (the kit's LEVER census reads ``fpf_rf3_adapter.describe()['rows']`` / ``describe_v2()['xmul']``).  A row
that raises anything but a refusal on a call is the adapter's own kernel-error path for ``fast`` (stock for that call, counted, printed once)
and, for ``xmul`` (the stock statement has no handler), counted + printed once here and that call served by the library op (a prediction never
dies of it; an out-of-memory propagates — it is the caller's to see).
"""
import re
import sys

COMPONENT_WORDS = {                                      # arm sub-words per component: <component>.<word>; the first is the default (no sub-word)
    "fast": ("v4", "tmk3_fast", "tmk3_exact", "tx_sm90a", "esm_v5_fwd", "esm_v61", "fast", "big"),
    "xmul": ("exact", "eager", "tmk3_exact", "esm_v5_fwd", "esm_v61", "cueq"),
}
EXACT_TIER_CLASSES = ("bitwise", "stock")                # the xmul seam serves a row whose numerics class is the stock op's bits (``bitwise≡cueq``) or a named stock
                                                         # row; a tolerance-class row (the ESM-family lines, v4) named under xmul is refused BY CLASS -> cueq
DEFAULT_WORD = {c: w[0] for c, w in COMPONENT_WORDS.items()}
PREFIX = "[fpf_rf3 trimul]"
STOCK_ROWS = ("cueq", "torch_math")                      # the provider's named stock rows: under the fast seam = the stock statement BY NAME (declined, counted)

STATE = {
    "fast": {"word": DEFAULT_WORD["fast"], "on": False},
    "xmul": {"word": DEFAULT_WORD["xmul"], "on": False},
}
_SEL = {}                                                # (component, word, cc, prec, C, D, N, direction) -> entry (selection, refusal, counts)
PAIR_SITE = (128, 128)                                   # (d_pair, d_hidden) of the fast head word's site under a TIER word: rf3's pair track; a ROW word serves
                                                         # every module its row admits, the template track's (64, 64) included
CENSUS = {"fast": {}, "xmul": {}}                        # component -> {"<row>|<prec>|C<c>|D<d>": calls}; plus "refused:<row>:<kind>-><fallback>", "declined:<word>", "error:<row>"
ERRORS = {"n": 0, "first": None}
_ORIG = {}                                               # "beneath": the object this seam's proxy wraps; "stock": the library's triangle_multiplicative_update
_PROC = {}                                               # per-process facts: capability per device, the provider's stack word


class WordError(ValueError):
    """An arm sub-word outside the component's vocabulary (``COMPONENT_WORDS``) — refused by name at ``apply_arm`` time."""


class Declined(Exception):
    """The fast seam serves nothing for this call: ``word`` is the adapter's census key (``N<floor>`` | ``c=<c>/d=<d>`` | ``refused:<row>:<kind>``);
    the stock statement runs, counted under it."""

    def __init__(self, word):
        Exception.__init__(self, word)
        self.word = word


def parse_component(token: str):
    """'fast' -> ('fast', None); 'fast.tmk3_fast' -> ('fast', 'tmk3_fast'); 'xmul.tmk3_exact' -> ('xmul', 'tmk3_exact'); (None, None) for a token that is
    not one of this module's components; WordError for a known component with an unknown (or empty) word."""
    base, dot, word = str(token).partition(".")
    if base not in COMPONENT_WORDS:
        return None, None
    if not dot:
        return base, None
    if word not in COMPONENT_WORDS[base]:
        raise WordError(f"FPF arm component {token!r}: word {word!r} is not one of {COMPONENT_WORDS[base]} ({base}.<word>)")
    return base, word


def strip_components(arm: str):
    """Split an arm string into (the arm without this module's sub-words / xmul, {"fast": <sub-word or None>, "xmul": <word or None when absent>}).
    ``fast.tmk3_fast`` is handed on as ``fast`` (the adapter's own trimul head word: the cast, the binding, the census unchanged); ``xmul[.w]`` is
    removed (this module installs it; a bare ``xmul`` is the default word)."""
    body, at, tail = arm.partition("@")
    words = {"fast": None, "xmul": None}
    keep = []
    for p in [x for x in body.split("+") if x]:
        base, word = parse_component(p)
        if base is None:
            keep.append(p)
            continue
        if base == "fast":
            words["fast"] = word
            keep.append("fast")
        else:
            words["xmul"] = word or DEFAULT_WORD["xmul"]
    out = "+".join(keep)
    if not out:
        out = "stock"
    return out + (at + tail if at else ""), words


# ----------------------------------------------------------------------------------------------------------------- the provider
def _T():
    from opt_core.kernels import trimul as T
    return T


def _stock():
    """The stock op (cuequivariance_torch.triangle_multiplicative_update) of this process, or None when the library is absent."""
    if "stock" in _ORIG:
        return _ORIG["stock"]
    fn = None
    try:
        A = sys.modules.get("rf3.model.layers.attention")
        obj = getattr(A, "cuet", None) if A is not None else None
        while obj is not None and hasattr(obj, "_real"):          # beneath every seam proxy (this module's, fpf_rf3_triattn's)
            obj = obj._real
        fn = getattr(obj, "triangle_multiplicative_update", None) if obj is not None else None
        if fn is None:
            import cuequivariance_torch as cqt
            fn = cqt.triangle_multiplicative_update
    except Exception:
        fn = None
    _ORIG["stock"] = fn
    return fn


def cueq_fallback_threshold(default=100):
    """cuEquivariance's own small-N rule: ``cuequivariance_ops_torch.triangle_multiplicative_update.CUEQ_TRIMUL_FALLBACK_THRESHOLD`` — at N <= it the
    STOCK op takes the library's torch reference path (``_tri_mul_torch``), not the CUDA kernels the provider's exact rows reproduce bit for bit (its
    cell table starts above it).  Read from the loaded module (the constant the kit's KERNELS census prints as ``thresholds=trimul:<t>``); ``default``
    when the module or the constant is absent."""
    if "thr" in _PROC:
        return _PROC["thr"]
    m = sys.modules.get("cuequivariance_ops_torch.triangle_multiplicative_update")
    if m is None:
        try:
            import importlib
            m = importlib.import_module("cuequivariance_ops_torch.triangle_multiplicative_update")
        except Exception:
            m = None
    t = getattr(m, "CUEQ_TRIMUL_FALLBACK_THRESHOLD", None) if m is not None else None
    _PROC["thr"] = int(t) if isinstance(t, int) else int(default)
    return _PROC["thr"]


def _facts(x, C, D, direction):
    """(cc, precision word, C, D, N, direction word) of a live call — the provider's cell grammar; capability and stack memoised per process."""
    T = _T()
    dev = str(x.device)
    cc = _PROC.get(("cc", dev))
    if cc is None:
        cc = _PROC[("cc", dev)] = tuple(T._device_cc(x))
    if "stack" not in _PROC:
        _PROC["stack"] = T.stack_word(x)
    prec, _cdt = T.call_precision(x)
    return cc, prec, int(C), int(D), int(x.shape[-2]), "out" if str(direction).startswith("out") else "in"


def _admit(T, sel, facts, residual, batch):
    cc, prec, C, D, N, _d = facts
    T.admits(sel.row, cc, prec, C, D, N, residual=residual, batch=batch, has_cueq=(True if _stock() is not None else None),
             abi=(T.tx_abi_tag() if sel.row.startswith("tx_sm90a") else None))


def _select(comp, facts, residual=False, batch=1, capturing=False):
    """The provider's selection for (component word, tensor facts[, capture context]), memoised; a refusing row is bound to its named fallback
    and the refusal kept.  ``capturing`` = the call is being recorded into a CUDA graph (the ``eager`` scope word reads it; the key carries it)."""
    T = _T()
    cc, prec, C, D, N, direction = facts
    word = STATE[comp]["word"]
    ctx = "capture" if capturing else "eager"
    key = (comp, word, tuple(cc), prec, C, D, N, direction) + ((ctx,) if word == "eager" else ())
    ent = _SEL.get(key)
    if ent is not None:
        return ent
    has_cueq = True if _stock() is not None else None
    asked = word
    if word == "eager":                                   # scope word: the cell's exact row outside graph capture, the library op by name inside it
        asked = "cueq" if capturing else "exact"
        if capturing and has_cueq is False:
            asked = "torch_math"
    kw = dict(word=asked, stack=_PROC.get("stack"), has_cueq=has_cueq)
    refusal = None
    try:
        sel = T.select(cc, prec, C, D, N, direction, **kw)
        _admit(T, sel, facts, residual, batch)
    except T.Refusal as r:
        refusal = r
        fb = r.fallback or "cueq"
        try:
            sel = T.select(cc, prec, C, D, N, direction, **dict(kw, word=fb))
            _admit(T, sel, facts, residual, batch)
        except T.Refusal as r2:                           # the fallback itself refuses (v4 <- tx_sm90a on the c=64 track ...): the stock op by name
            refusal = r2 if r2.row else refusal
            sel = T.select(cc, prec, C, D, N, direction, **dict(kw, word="cueq" if has_cueq is not False else "torch_math"))
    if comp == "xmul" and not str(sel.cls or "").startswith(EXACT_TIER_CLASSES) and sel.row not in STOCK_ROWS:   # the exact tier: the stock op's bits or the stock op
        refusal = T.Refusal("class:%s(the exact tier serves rows measured bitwise to the stock op; %s is %s class)" % (str(sel.cls).split("(")[0], sel.row, str(sel.cls).split("(")[0]),
                            sel.row, "cueq" if has_cueq is not False else "torch_math")
        sel = T.select(cc, prec, C, D, N, direction, **dict(kw, word=refusal.fallback))
    ent = {"sel": sel, "refusal": refusal, "calls": 0, "key": key, "call_refusal": None, "first_refusal": refusal}
    _SEL[key] = ent
    _say_select(comp, ent)
    return ent


def _say_select(comp, ent):
    T = _T()
    sel, r = ent["sel"], ent.get("call_refusal") or ent["refusal"]
    _c, word, cc, prec, C, D, N, direction = ent["key"][:8]
    ctx = (" ctx=%s" % ent["key"][8]) if len(ent["key"]) > 8 else ""
    line = (f"{PREFIX} SELECT component={comp} asked={word}{ctx} cc={cc[0]}.{cc[1]} dtype={prec} c={C} d={D} n={N} dir={direction} "
            f"{T.describe(sel)}")
    if ctx == " ctx=capture":
        line += " (eager: a call captured into a CUDA graph keeps the library op by name; the cell table ranks eager calls)"
    if r is not None:
        line += f" REFUSED row={r.row} kind={r.kind} -> bound {sel.row} by name"
    print(line, flush=True)


def _count(comp, ent):
    ent["calls"] += 1
    _c, _w, _cc, prec, C, D, _n, _d = ent["key"][:8]
    k = f"{ent['sel'].row}|{prec}|C{C}|D{D}"
    CENSUS[comp][k] = CENSUS[comp].get(k, 0) + 1
    r = ent.get("call_refusal") or ent["refusal"]
    if r is not None:
        rk = f"refused:{r.row}:{r.kind}->{ent['sel'].row}"
        CENSUS[comp][rk] = CENSUS[comp].get(rk, 0) + 1


_FLOOR = re.compile(r"^n<(\d+)")


def decline_word(ent):
    """The adapter's census key for a fast-seam call bound to a stock row: ``N<k>`` (a token floor: report.py TRIMUL_TOKEN_FLOOR), ``c=<c>/d=<d>``
    (a pair track the row does not serve: TRIMUL_TRACK_CLASS), else ``refused:<row>:<kind>`` (capability / ABI / stack: a machine word)."""
    _c, _w, _cc, _p, C, D, _n, _d = ent["key"][:8]
    r = ent.get("call_refusal") or ent["refusal"]
    kind = r.kind if r is not None else ""
    m = _FLOOR.match(kind)
    if m:
        return "N<%s" % m.group(1)
    if (C, D) != (128, 128) and (r is None or kind.startswith(("c_z", "c:", "c=", "d_hidden", "c_z="))):
        return "c=%d/d=%d" % (C, D)
    if r is None:
        return "refused:%s:no_row" % ent["sel"].word
    return ("refused:%s:%s" % (r.row, kind)).replace(" ", "_")[:120]


def _rebind(comp, ent, refusal):
    T = _T()
    _c, _w, cc, prec, C, D, N, direction = ent["key"][:8]
    fb = refusal.fallback or "cueq"
    if fb == ent["sel"].row:
        fb = "cueq"
    has_cueq = True if _stock() is not None else None
    try:
        ent["sel"] = T.select(cc, prec, C, D, N, direction, word=fb, stack=_PROC.get("stack"), has_cueq=has_cueq)
    except T.Refusal:
        ent["sel"] = T.select(cc, prec, C, D, N, direction, word="cueq" if has_cueq is not False else "torch_math", has_cueq=has_cueq)
    ent["call_refusal"] = refusal
    _say_select(comp, ent)


def _store(m):
    """Per-module store: the ten canonical weight tensors (stable objects: the provider keys its packed copies on them) + the provider's cache."""
    s = getattr(m, "_fpf_trimul_rows", None)
    if s is None:
        D = int(m.d_hidden)
        s = m._fpf_trimul_rows = {
            "w10": dict(ln_in_w=m.norm_in.weight.detach(), ln_in_b=m.norm_in.bias.detach(),
                        w_ag=m.g_in.weight.detach()[:D], w_ap=m.p_in.weight.detach()[:D],
                        w_bg=m.g_in.weight.detach()[D:], w_bp=m.p_in.weight.detach()[D:],
                        ln_out_w=m.norm_out.weight.detach(), ln_out_b=m.norm_out.bias.detach(),
                        w_o=m.p_out.weight.detach(), w_og=m.g_out.weight.detach()),
            "cache": {}}
    return s


# ----------------------------------------------------------------------------------------------------------------- the fast seam
def fast_floor(m, x, thr=None):
    """The fast seam's token floor for one call of module ``m`` on the cast pair tensor ``x``: below it the stock statement runs BY NAME (census
    word ``N<floor>``).  Under a TIER word (fast | big) the floor is THE PROVIDER TABLE'S: row v4's per-class token floor
    ``opt_core.kernels.trimul.v4_n_min(cc, precision, c, d)`` (TRIMUL_CELLS ``rows.v4.admits.n_min_cells``: 2 tokens on the cc-9.0 c_z-128 classes
    whose small-N cells name row v4; ``n_min`` = 101 on every unlisted
    class -- every A100 class, fp32 (the (64, 64) template track never asks: a tier word declines it by track, ``c=64/d=64``) -- and 101 on a core without the reader) -- the very number the provider's own
    admits() and serving call hold row v4 to, so the seam asks select() exactly where the table answers with a kernel row and keeps the word
    ``N<101`` everywhere else.  Never above the library's threshold + 1 (``cueq_fallback_threshold() + 1`` = 101, the seam's floor when the
    table carries none); a ROW word keeps exactly that."""
    T = _T()
    thr = cueq_fallback_threshold() if thr is None else int(thr)
    old = thr + 1
    if STATE["fast"]["word"] not in T.TIER_WORDS:
        return old
    floor_of = getattr(T, "v4_n_min", None)
    if floor_of is None:                                  # a core without the table-floor reader: the library threshold + 1
        return old
    cc, prec, C, D, _n, _dw = _facts(x, int(x.shape[-1]), m.d_hidden, m.direction)
    return min(old, int(floor_of(cc, prec, C, D)))


def serve_fast(m, x, residual=False, pad=16):
    """The fast tier's TriMul of module ``m`` (rf3 TriangleMultiplication: d_pair, d_hidden, direction, norm_in/p_in/g_in/norm_out/p_out/g_out) on
    the already-cast pair tensor ``x`` through the provider's row for the ``fast`` word.  Returns the update (z + update when ``residual``: the
    ``res`` component's fused epilogue) or raises ``Declined(word)`` when no admitted row serves this call (the adapter runs the stock statement,
    counted under ``word``): below the call's token floor (``N<floor>``, :func:`fast_floor` -- ``N<2`` on the classes the table floors at 2, ``N<101``
    on every other), on the template track's (64, 64) modules under a TIER word (``c=64/d=64`` at every size), on a width no row admits under a ROW word.  A call ABOVE the
    library's threshold (N >= 101) never reads the table floor.  Kernel
    errors propagate (the adapter's error path); a run-time ``Refusal`` re-binds the named fallback row."""
    T = _T()
    thr = cueq_fallback_threshold()
    if int(x.shape[-2]) <= thr:                           # at or below the library's threshold:
        if STATE["fast"]["word"] in T.TIER_WORDS and (int(m.d_pair), int(m.d_hidden)) != PAIR_SITE:   # the template track's (64, 64) modules are not the tier
            k = "declined:c=%d/d=%d" % (int(m.d_pair), int(m.d_hidden))                              # word's site at ANY size: declined with the one word the
            CENSUS["fast"][k] = CENSUS["fast"].get(k, 0) + 1                                          # calls from 101 tokens carry (below), the stock statement runs
            raise Declined("c=%d/d=%d" % (int(m.d_pair), int(m.d_hidden)))
        floor = fast_floor(m, x, thr)                     # the pair track: served from the table's floor for this call's class up (fast_floor), the stock
        if int(x.shape[-2]) < floor:                      # statement BY NAME below it -- census word N<floor> (N<2 on cc 9.0; N<101 wherever the table carries no floor)
            k = "declined:N<%d" % floor
            CENSUS["fast"][k] = CENSUS["fast"].get(k, 0) + 1
            raise Declined("N<%d" % floor)
    if STATE["fast"]["word"] in T.TIER_WORDS and (int(m.d_pair), int(m.d_hidden)) != PAIR_SITE:   # a TIER word serves the lever's site — the pair track's
        k = "declined:c=%d/d=%d" % (int(m.d_pair), int(m.d_hidden))                              # TriangleMultiplication (c_z = d_hidden = 128: pairformer, MSA
        CENSUS["fast"][k] = CENSUS["fast"].get(k, 0) + 1                                          # module, confidence head); the template track's (64, 64)
        raise Declined("c=%d/d=%d" % (int(m.d_pair), int(m.d_hidden)))                          # modules stay the stock statement's (xmul's site), counted under one census word
    facts = _facts(x, m.d_pair, m.d_hidden, m.direction)
    batch = int(x.shape[0]) if x.dim() == 4 else 1
    ent = _select("fast", facts, residual=residual, batch=batch)
    st = None
    for _attempt in (0, 1, 2):
        sel = ent["sel"]
        if sel.row in STOCK_ROWS:                         # no fused row admitted for this call on this card / stack / shape: the stock statement BY NAME
            word = decline_word(ent)
            ent["calls"] += 1
            CENSUS["fast"]["declined:" + word] = CENSUS["fast"].get("declined:" + word, 0) + 1
            raise Declined(word)
        st = st or _store(m)
        try:
            out = T.triangle_multiplication(x, None, direction=m.direction, weights=st["w10"], selection=sel, residual=residual,
                                            stock=_stock(), cache=st["cache"], pad=pad)
        except T.Refusal as r:                            # a run-time refusal of the row (the kernel's own supported(), a load gate): bind the fallback, by name
            _rebind("fast", ent, r)
            continue
        _count("fast", ent)
        return out
    raise RuntimeError(f"{PREFIX} fast: row {ent['sel'].row} refused repeatedly for one shape ({ent.get('call_refusal')})")   # pragma: no cover


def note_default(m, x):
    """The default word's record for a call the adapter served with its direct launch (``v4`` = fpf_trimul_v4 generic.trimul): the provider's
    SELECT line once per new shape key and the row census.  A table read only -- launches nothing and never raises into the forward."""
    try:
        if not getattr(x, "is_cuda", False):
            return
        ent = _select("fast", _facts(x, m.d_pair, m.d_hidden, m.direction))
        _count("fast", ent)
    except Exception as e:                                # pragma: no cover - census only
        k = "note:unavailable:%s" % type(e).__name__
        CENSUS["fast"][k] = CENSUS["fast"].get(k, 0) + 1


def set_fast_word(word):
    word = word or DEFAULT_WORD["fast"]
    if word not in COMPONENT_WORDS["fast"]:
        raise WordError(f"fast word {word!r} is not one of {COMPONENT_WORDS['fast']}")
    STATE["fast"]["word"] = word
    STATE["fast"]["on"] = True
    return word


# ----------------------------------------------------------------------------------------------------------------- the xmul seam
class CuetTrimulProxy(object):
    """``rf3.model.layers.attention.cuet`` with ``triangle_multiplicative_update`` served by the provider's exact row; every other attribute is
    the wrapped object's (the library module, or another component's proxy over it)."""

    def __init__(self, real):
        self._real = real

    def triangle_multiplicative_update(self, x, direction="outgoing", mask=None, norm_in_weight=None, norm_in_bias=None, p_in_weight=None,
                                         g_in_weight=None, norm_out_weight=None, norm_out_bias=None, p_out_weight=None, g_out_weight=None,
                                         eps=1e-5, **kw):
        key = None
        if kw or any(t is None for t in (norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight)):
            key = "passthrough:kwargs"                        # a form this seam does not know: the library's own call, counted
        elif hasattr(x, "shape") and len(x.shape) >= 2 and int(x.shape[-2]) <= cueq_fallback_threshold():
            key = "passthrough:N<=%d" % cueq_fallback_threshold()   # the library's reference-path sizes: the stock op BY NAME (bitwise with stock by construction)
        if key is not None:
            CENSUS["xmul"][key] = CENSUS["xmul"].get(key, 0) + 1
            return self._real.triangle_multiplicative_update(x, direction=direction, mask=mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias,
                                                              p_in_weight=p_in_weight, g_in_weight=g_in_weight, norm_out_weight=norm_out_weight,
                                                              norm_out_bias=norm_out_bias, p_out_weight=p_out_weight, g_out_weight=g_out_weight, eps=eps, **kw)
        return serve_xmul(x, direction, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight,
                          g_out_weight, eps)

    def __getattr__(self, name):
        return getattr(self._real, name)


_XW = {}                                                 # id(p_in_weight) -> {"ref": p_in_weight, "w10": {...}, "cache": {}} (the statement's own tensors, split once)


def _xstore(norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight):
    s = _XW.get(id(p_in_weight))
    if s is None or s["ref"] is not p_in_weight:
        D = int(p_in_weight.shape[0]) // 2
        s = _XW[id(p_in_weight)] = {
            "ref": p_in_weight,
            "w10": dict(ln_in_w=norm_in_weight.detach(), ln_in_b=norm_in_bias.detach(), w_ap=p_in_weight.detach()[:D], w_bp=p_in_weight.detach()[D:],
                        w_ag=g_in_weight.detach()[:D], w_bg=g_in_weight.detach()[D:], ln_out_w=norm_out_weight.detach(), ln_out_b=norm_out_bias.detach(),
                        w_o=p_out_weight.detach(), w_og=g_out_weight.detach()),
            "cache": {}}
    return s


def serve_xmul(x, direction, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, eps):
    """The stock statement's one library call through the provider's exact row for the ``xmul`` word.  The ``cueq`` row IS the library call with
    the statement's own arguments (bound by name: nothing re-packed, nothing substituted)."""
    from opt_core.oom import is_oom
    T = _T()
    C = int(x.shape[-1]); D = int(p_in_weight.shape[0]) // 2
    facts = _facts(x, C, D, direction)
    batch = int(x.shape[0]) if x.dim() == 4 else 1
    capturing = False
    if STATE["xmul"]["word"] == "eager" and x.is_cuda:
        import torch
        capturing = bool(torch.cuda.is_current_stream_capturing())
    ent = _select("xmul", facts, residual=False, batch=batch, capturing=capturing)
    stock = _stock()

    def _library():
        return stock(x, direction=direction, mask=mask, norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias, p_in_weight=p_in_weight,
                     g_in_weight=g_in_weight, norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias, p_out_weight=p_out_weight,
                     g_out_weight=g_out_weight, eps=eps)
    for _attempt in (0, 1, 2):
        sel = ent["sel"]
        if sel.row == "cueq":
            out = _library()
            _count("xmul", ent)
            return out
        st = _xstore(norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight)
        try:
            out = T.triangle_multiplication(x, mask, direction=direction, weights=st["w10"], selection=sel, residual=False, stock=stock,
                                            cache=st["cache"], eps=eps)
        except T.Refusal as r:                            # a run-time refusal of the row: bind the fallback for this shape, by name
            _rebind("xmul", ent, r)
            continue
        except Exception as e:
            if is_oom(e):
                raise
            if isinstance(e, (OSError, ImportError)) and ent["calls"] == 0:      # the row's run-time needs absent (triton / cuequivariance_ops_torch): a named refusal
                fb = (T.table().get("rows", {}).get(sel.row, {}) or {}).get("fallback") or "cueq"
                _rebind("xmul", ent, T.Refusal(f"runtime_absent:{type(e).__name__}", sel.row, fb))
                continue
            ERRORS["n"] += 1
            CENSUS["xmul"][f"error:{sel.row}"] = CENSUS["xmul"].get(f"error:{sel.row}", 0) + 1
            if ERRORS["first"] is None:
                ERRORS["first"] = f"{sel.row}: {type(e).__name__}: {str(e)[:200]}"
                print(f"{PREFIX} ROW ERROR component=xmul row={sel.row} served by the stock op for this call: {e!r}"[:400], flush=True)
            out = _library()
            ent["calls"] += 1
            return out
        _count("xmul", ent)
        return out
    raise RuntimeError(f"{PREFIX} xmul: row {ent['sel'].row} refused repeatedly for one shape ({ent.get('call_refusal')})")   # pragma: no cover


def _chain(A):
    """[(holder, attr-or-None, obj), ...] down the ``cuet`` proxy chain: the module attribute first, then each proxy's ``_real``."""
    out = []
    obj = getattr(A, "cuet", None)
    holder, attr = A, "cuet"
    while obj is not None:
        out.append((holder, attr, obj))
        if not hasattr(obj, "_real") or isinstance(obj, type):
            break
        holder, attr, obj = obj, "_real", obj._real
    return out


def enable_xmul(word=None):
    """Install (word given or default) / keep the exact-tier seam: ``attention.cuet`` -> CuetTrimulProxy over whatever is there.  Returns the word.
    Refuses by name (RuntimeError) when rf3's attention module has no ``cuet`` (cuequivariance not in use: nothing to route)."""
    word = word or DEFAULT_WORD["xmul"]
    if word not in COMPONENT_WORDS["xmul"]:
        raise WordError(f"xmul word {word!r} is not one of {COMPONENT_WORDS['xmul']}")
    import rf3.model.layers.attention as A
    cuet = getattr(A, "cuet", None)
    if cuet is None:
        raise RuntimeError("xmul: rf3.model.layers.attention has no `cuet` (cuequivariance_torch not imported by upstream): nothing to route")
    if not any(isinstance(obj, CuetTrimulProxy) for _h, _a, obj in _chain(A)):
        _ORIG["beneath"] = cuet
        _ORIG.pop("stock", None); _stock()
        A.cuet = CuetTrimulProxy(cuet)
    STATE["xmul"]["word"] = word
    STATE["xmul"]["on"] = True
    return word


def disable_xmul():
    """Remove this seam's proxy wherever it sits in the ``cuet`` chain (outermost: the attribute; beneath another component's proxy: that proxy's ``_real``)."""
    A = sys.modules.get("rf3.model.layers.attention")
    if A is not None:
        for holder, attr, obj in _chain(A):
            if isinstance(obj, CuetTrimulProxy):
                setattr(holder, attr, obj._real)
                break
    STATE["xmul"]["on"] = False
    return False


# ----------------------------------------------------------------------------------------------------------------- census
def reset_census():
    for c in CENSUS:
        CENSUS[c] = {}
    _SEL.clear()
    ERRORS["n"] = 0; ERRORS["first"] = None


def census(comp=None):
    """{component: {"word", "on", "calls": {row|prec|C|D: n, refused:...: n, declined:...: n, error:...: n}, "served": n, "selections": [describe lines]}}."""
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
            r = ent.get("call_refusal") or ent["refusal"]
            sels.append(f"c{key[4]}d{key[5]}n{key[6]}{key[7]}{(':' + key[8]) if len(key) > 8 else ''}:{d}" + (f" REFUSED {r.row}:{r.kind}" if r is not None else ""))
        served = sum(v for k, v in CENSUS[c].items() if not k.startswith(("refused:", "error:", "passthrough:", "declined:")))
        out[c] = {"word": STATE[c]["word"], "on": bool(STATE[c]["on"]), "calls": dict(CENSUS[c]), "served": int(served), "selections": sels,
                  "declined": sum(v for k, v in CENSUS[c].items() if k.startswith("declined:")),
                  "errors": sum(v for k, v in CENSUS[c].items() if k.startswith("error:"))}
    return out if comp is None else out[comp]


def census_line(comp):
    """One line for the kit's exit census: ``[fpf_rf3 trimul] CENSUS component=<c> word=<w> served=<n> rows=<row|prec|C|D:n,...> [declined=...] [refused=...] selections=<...>``."""
    c = census(comp)
    rows = ",".join(f"{k}:{v}" for k, v in sorted(c["calls"].items()) if not k.startswith(("refused:", "error:", "declined:")))
    dec = ",".join(f"{k[len('declined:'):]}:{v}" for k, v in sorted(c["calls"].items()) if k.startswith("declined:"))
    ref = ",".join(f"{k[len('refused:'):]}:{v}" for k, v in sorted(c["calls"].items()) if k.startswith("refused:"))
    err = ",".join(f"{k[len('error:'):]}:{v}" for k, v in sorted(c["calls"].items()) if k.startswith("error:"))
    line = f"{PREFIX} CENSUS component={comp} word={c['word']} on={c['on']} served={c['served']} rows={rows or '-'}"
    if dec:
        line += f" declined={dec}"
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
    return True
