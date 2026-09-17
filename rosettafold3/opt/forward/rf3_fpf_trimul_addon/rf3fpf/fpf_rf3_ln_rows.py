"""fpf_rf3_ln_rows — RoseTTAFold3's LayerNorm calls on the shared core's ONE provider (``opt_core.kernels.ln``).

One seam, the FPF arm component ``xln[.<word>]``: a class-level re-binding of ``torch.nn.LayerNorm.forward`` switched from the arm
(``fpf_rf3_adapter.apply_arm``) and undone by it — no file or weight edit.  Every ``nn.LayerNorm`` module RF3 builds (the pairformer's
triangle-attention / transition / attention-pair-bias norms on the c=128 pair tensor, the MSA module's, the c=64 template track's, the
confidence head's pairformer, the atom transformers', the diffusion transformer's AdaLN parts, the diffusion conditioning's) reaches
``F.layer_norm`` through that one method; the seam serves the call through the provider when the call's FORM is one the provider serves
with stock's own semantics, and runs the module's own forward under a NAMED route otherwise (counted, printed in the census — never silent):

    fp32    fp32 x, fp32 gamma/beta (autocast on or off: ``layer_norm`` is on autocast's fp32 list, the statement runs fp32 either way)
    bf16w   bf16 x, fp32 gamma/beta UNDER bf16 autocast (autocast's own form: fp32 arithmetic, fp32 OUT) -> the provider's widen form
    routes  ``form:<why>`` for everything else (bf16 x outside autocast, a missing gamma under autocast [AdaLN's affine-free norm: autocast
            returns fp32, the provider's bf16 form would not], fp16, a normalized_shape that is not the last dim, CPU, autograd) -> the
            module's own forward BY NAME

Words (``COMPONENT_WORDS["xln"]``; the first is the default):
    exact            the provider's exact TIER word per cell: exactln / exactln:widen (bitwise to ATen) where the core's table (LN_CELLS.json)
                     names one for this card's cell, ATen BY NAME at the cells where it names none (bf16 N^2 rows, N-row cells in eager)
                     — the word this kit's exact mode asks
    fast | big     the tier words (tolerance-class rows admitted: fastln / ln_rows on fp32 cells; bf16w cells = exactln:widen)
    exactln | exactln:triton | fastln | ln_rows | aten
                     row words = exactly that row (the provider's default rule); a refusing row binds its NAMED fallback
                     (``SELECT ... REFUSED row=<r> kind=<k> -> bound <fb> by name``), never a silent substitute

Selection is the provider's ``select()``, memoised per (word, cc, form, C, x shape, aligned, capture): once per distinct call class, printed
as a ``SELECT`` line with the provider's ``describe()``.  Under CUDA stream capture (the ``tg`` trunk graph, the sampler graph) the class is
selected with ``capture=True`` — the provider never names a capture-unsafe row a winner there (its ``CAPTURE_UNSAFE_ROWS``; exactln /
fastln / ln_rows all replay under capture).  The capture
query (``torch.cuda.is_current_stream_capturing``) is made only for a call class whose eager and graph selections differ.

Census at exit / describe(): ``CENSUS component=xln word=<w> on=<bool> served=<n> rows=<row|form|C<c>:n,...> routes=<form:why:n,...>
refused=<row:kind->fb:n> selections=<describe(); ...>``.
"""
import os
import sys

COMPONENT_WORDS = {"xln": ("exact", "fast", "big", "exactln", "exactln:triton", "exactln:widen", "fastln", "ln_rows", "aten")}
DEFAULT_WORD = {c: w[0] for c, w in COMPONENT_WORDS.items()}
PREFIX = "[fpf_rf3 ln]"
STOCK_ROWS = ("aten", "aten_autocast")                     # the provider's named stock rows = the module's own forward BY NAME
STATE = {"xln": {"on": False, "word": None}}
_SEL = {}                                                   # (word, cc, form, C, shape, aligned, capture) -> entry
_CLASS = {}                                                 # (word, cc, form, C, shape, aligned) -> entry | ("ask",) when eager/graph selections differ
CENSUS = {"xln": {}}                                        # "<row>|<form>|C<c>": n ; "route:form:<why>": n ; "refused:<row>:<kind>-><fb>": n ; "error:<cls>": n
ERRORS = {"n": 0, "first": None}
_ORIG = {}                                                  # "forward": torch.nn.LayerNorm.forward as this process imported it
_PROC = {}


def parse_component(token: str):
    """'xln' -> ('xln', None); 'xln.fastln' -> ('xln', 'fastln'); 'xln.exactln:triton' -> ('xln', 'exactln:triton'); anything else -> (None, None)."""
    base, dot, word = token.partition(".")
    if base not in COMPONENT_WORDS:
        return None, None
    if dot and word not in COMPONENT_WORDS[base]:
        raise ValueError("%s: unknown word %r for component %s (words: %s)" % (PREFIX, word, base, ", ".join(COMPONENT_WORDS[base])))
    return base, (word or None)


def strip_components(arm: str):
    """(the arm without ``xln[.w]``, {"xln": <word, or None when absent>}); a bare ``xln`` is the default word."""
    body, at, tail = arm.partition("@")
    words = {"xln": None}
    keep = []
    for p in [x for x in body.split("+") if x]:
        base, word = parse_component(p)
        if base is None:
            keep.append(p)
            continue
        words["xln"] = word or DEFAULT_WORD["xln"]
    out = "+".join(keep) or "stock"
    return out + (at + tail if at else ""), words


# ----------------------------------------------------------------------------------------------------------------- the provider
def _L():
    from opt_core.kernels import ln as L
    return L


def _proc(torch, dev):
    k = ("cc", dev)
    if k not in _PROC:
        _PROC[k] = tuple(torch.cuda.get_device_capability(dev))
        if "stack" not in _PROC:
            try:
                _PROC["stack"] = _L().stack_word(dev)
            except Exception:                                # noqa: BLE001
                _PROC["stack"] = None
            try:
                import triton  # noqa: F401
                _PROC["has_triton"] = True
            except ImportError:
                _PROC["has_triton"] = False
    return _PROC[k]


def form_of(torch, x, weight, bias):
    """(form word, route reason): 'fp32' | 'bf16w' served; None + why otherwise (the module's own forward by name)."""
    ac = torch.is_autocast_enabled()
    if not x.is_cuda:
        return None, "cpu"
    if torch.is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad)):
        return None, "autograd"
    if x.dtype == torch.float32 and (weight is None or weight.dtype == torch.float32) and (bias is None or bias.dtype == torch.float32):
        return "fp32", None
    if x.dtype == torch.bfloat16:
        if ac and _autocast_dtype(torch) == torch.bfloat16:
            if weight is not None and weight.dtype == torch.float32 and (bias is None or bias.dtype == torch.float32):
                return "bf16w", None
            return None, ("bf16_noaffine_autocast" if weight is None else "bf16_params_%s_autocast" % str(weight.dtype).replace("torch.", ""))
        return None, "bf16_no_autocast"
    return None, "dtype_%s" % str(x.dtype).replace("torch.", "")


def _autocast_dtype(torch):
    try:
        return torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype()


def _n_tokens(x, C):
    rows = x.numel() // C
    if x.dim() >= 3 and x.shape[-2] == x.shape[-3]:
        return int(x.shape[-2])                              # [.., I, I, C] pair rows
    if x.dim() >= 2:
        return int(x.shape[-2])                              # [.., I, C] rows (single, msa, atom, dit: the provider's cell_for_rows sorts the family by rows vs N)
    return max(int(rows), 1)


def select_facts(cc, dtype, C, rows, n_tokens, form, word, capture=False, aligned=True):
    """The provider's selection for one call class (pure: table facts only) -> (Selection, Refusal|None, cell word).  A refusing row word is
    bound to its NAMED fallback (the refusal kept for the SELECT line and the census); the last resort is the form's stock row by name."""
    L = _L()
    cell = L.cell_for_rows(rows, n_tokens, C)
    widen = form == "bf16w"
    kw = dict(word=word, widen=widen, stack=_PROC.get("stack"), capture=capture, C=C, has_triton=_PROC.get("has_triton"), aligned=aligned)
    refusal = None
    try:
        sel = L.select(cc, dtype, cell, n_tokens, **kw)
    except L.Refusal as r:
        refusal = r
        fb = r.fallback or ("aten_autocast" if widen else "aten")
        try:
            sel = L.select(cc, dtype, cell, n_tokens, **dict(kw, word=fb))
        except L.Refusal as r2:
            refusal = r2 if r2.row else refusal
            sel = L.select(cc, dtype, cell, n_tokens, **dict(kw, word=("aten_autocast" if widen else "aten")))
    return sel, refusal, cell


def _select_one(L, cc, x, weight, C, form, word, capture, aligned):
    n = _n_tokens(x, C)
    rows = x.numel() // C
    sel, refusal, cell = select_facts(cc, x.dtype, C, rows, n, form, word, capture=capture, aligned=aligned)
    return sel, refusal, cell, n


def _say_select(ent):
    L = _L()
    sel, r = ent["sel"], ent.get("call_refusal") or ent["refusal"]
    word, cc, form, C, shape, aligned, capture = ent["key"]
    line = ("%s SELECT component=xln asked=%s cc=%d.%d form=%s C=%d shape=%s aligned=%s capture=%s cell=%s -> %s"
            % (PREFIX, word, cc[0], cc[1], form, C, "x".join(str(s) for s in shape), aligned, capture, ent["cell"], L.describe(sel)))
    if r is not None:
        line += " REFUSED row=%s kind=%s -> bound %s by name" % (r.row, r.kind, L.arm_word(sel.row, sel.variant))
    print(line, flush=True)


def _entry(torch, x, weight, C, form, aligned):
    """The memoised selection for this call class; the capture state is queried only when the eager and graph selections differ."""
    L = _L()
    cc = _proc(torch, x.device)
    word = STATE["xln"]["word"] or DEFAULT_WORD["xln"]
    ck = (word, cc, form, C, tuple(x.shape), aligned)
    cls = _CLASS.get(ck)
    if cls is None:
        ents = []
        for capture in (False, True):
            sel, refusal, cell, n = _select_one(L, cc, x, weight, C, form, word, capture, aligned)
            ent = {"sel": sel, "refusal": refusal, "first_refusal": refusal, "call_refusal": None, "calls": 0, "cell": cell, "n": n, "key": ck + (capture,)}
            _SEL[ck + (capture,)] = ent
            ents.append(ent)
        same = (ents[0]["sel"].row, ents[0]["sel"].variant) == (ents[1]["sel"].row, ents[1]["sel"].variant)
        cls = ents[0] if same else ("ask",)
        _CLASS[ck] = cls
        _say_select(ents[0])
        if not same:
            _say_select(ents[1])
    if isinstance(cls, tuple):
        capture = bool(torch.cuda.is_current_stream_capturing())
        return _SEL[ck + (capture,)]
    return cls


def _count(ent, form, C):
    ent["calls"] += 1
    L = _L()
    k = "%s|%s|C%d" % (L.arm_word(ent["sel"].row, ent["sel"].variant), form, C)
    c = CENSUS["xln"]
    c[k] = c.get(k, 0) + 1
    r = ent.get("call_refusal") or ent["refusal"]
    if r is not None:
        rk = "refused:%s:%s->%s" % (r.row, r.kind, L.arm_word(ent["sel"].row, ent["sel"].variant))
        c[rk] = c.get(rk, 0) + 1


def _route(why):
    c = CENSUS["xln"]
    k = "route:form:%s" % why
    c[k] = c.get(k, 0) + 1


def _ln_forward(self, input):
    """torch.nn.LayerNorm.forward under the xln component."""
    import torch
    orig = _ORIG["forward"]
    ns = self.normalized_shape
    if len(ns) != 1 or input.dim() < 1 or input.shape[-1] != ns[0]:
        _route("normalized_shape"); return orig(self, input)
    form, why = form_of(torch, input, self.weight, self.bias)
    if form is None:
        _route(why); return orig(self, input)
    C = int(ns[0])
    L = _L()
    try:
        aligned = bool(L._rows_aligned(input, C)) if input.dim() >= 2 else True
    except Exception:                                        # noqa: BLE001
        aligned = input.is_contiguous()
    try:
        ent = _entry(torch, input, self.weight, C, form, aligned)
    except Exception as e:                                   # noqa: BLE001  (table / import trouble: named, counted, the module's own forward)
        ERRORS["n"] += 1; ERRORS["first"] = ERRORS["first"] or repr(e)[:200]
        k = "error:select:%s" % type(e).__name__
        CENSUS["xln"][k] = CENSUS["xln"].get(k, 0) + 1
        return orig(self, input)
    sel = ent["sel"]
    if sel.row in STOCK_ROWS:                                # the provider names the statement: the module's own forward BY NAME (its bytes, its autocast handling)
        _count(ent, form, C)
        return orig(self, input)
    try:
        y, _ = L.layer_norm(input, ns, self.weight, self.bias, self.eps, selection=sel)
    except L.Refusal as r:                                   # the row's own admission at call time (exactln: misaligned rows ...): rebind this class to the named fallback
        y = None; refusal = r
    except Exception as e:                                   # noqa: BLE001  a row that raises (build / driver): named + counted, this class rebound to the statement
        ERRORS["n"] += 1; ERRORS["first"] = ERRORS["first"] or ("%s: %s" % (type(e).__name__, str(e)[:160]))
        k = "error:%s:%s" % (L.arm_word(sel.row, sel.variant), type(e).__name__)
        CENSUS["xln"][k] = CENSUS["xln"].get(k, 0) + 1
        refusal = L.Refusal("raised:%s" % type(e).__name__, sel.row, "aten_autocast" if form == "bf16w" else "aten"); y = None
    if y is None:
        r = refusal
        ent["call_refusal"] = r
        fbw = r.fallback or ("aten_autocast" if form == "bf16w" else "aten")
        cc = _proc(torch, input.device)
        try:
            ent["sel"] = L.select(cc, input.dtype, ent["cell"], ent["n"], word=fbw, widen=(form == "bf16w"), stack=_PROC.get("stack"), C=C,
                                  has_triton=_PROC.get("has_triton"), aligned=aligned)
        except L.Refusal:
            ent["sel"] = L.select(cc, input.dtype, ent["cell"], ent["n"], word=("aten_autocast" if form == "bf16w" else "aten"), widen=(form == "bf16w"), C=C)
        _say_select(ent)
        _count(ent, form, C)
        if ent["sel"].row in STOCK_ROWS:
            return orig(self, input)
        y, _ = L.layer_norm(input, ns, self.weight, self.bias, self.eps, selection=ent["sel"])
        return y
    _count(ent, form, C)
    return y


# ----------------------------------------------------------------------------------------------------------------- switching
def enable_xln(word=None):
    """Install the seam (idempotent); returns the word in force."""
    import torch
    w = word or DEFAULT_WORD["xln"]
    if w not in COMPONENT_WORDS["xln"]:
        raise ValueError("%s: unknown xln word %r (words: %s)" % (PREFIX, w, ", ".join(COMPONENT_WORDS["xln"])))
    _L()                                                     # the provider must import: a missing face fails HERE, by name, before any binding
    if "forward" not in _ORIG:
        _ORIG["forward"] = torch.nn.LayerNorm.forward
    if STATE["xln"]["word"] != w:
        _SEL.clear(); _CLASS.clear()
    STATE["xln"]["word"] = w
    STATE["xln"]["on"] = True
    torch.nn.LayerNorm.forward = _ln_forward
    print("%s LEVER name=fpf_xln state=on word=%s seam=torch.nn.LayerNorm.forward provider=opt_core.kernels.ln rows=%s"
          % (PREFIX, w, ",".join(_L().ROW_NAMES[:4])), flush=True)
    return w


def disable_xln():
    import torch
    if "forward" in _ORIG and torch.nn.LayerNorm.forward is _ln_forward:
        torch.nn.LayerNorm.forward = _ORIG["forward"]
    STATE["xln"]["on"] = False
    return False


def reset_census():
    CENSUS["xln"].clear(); _SEL.clear(); _CLASS.clear()
    ERRORS["n"] = 0; ERRORS["first"] = None


def census(comp=None):
    L = None
    try:
        L = _L()
    except Exception:                                        # noqa: BLE001
        pass
    c = CENSUS["xln"]
    served = sum(n for k, n in c.items() if not k.startswith(("route:", "refused:", "error:")) and k.split("|")[0] not in STOCK_ROWS)
    stock = sum(n for k, n in c.items() if not k.startswith(("route:", "refused:", "error:")) and k.split("|")[0] in STOCK_ROWS)
    routes = sum(n for k, n in c.items() if k.startswith("route:"))
    sels = []
    for ent in _SEL.values():
        if ent["calls"]:
            s = (L.describe(ent["sel"]) if L else str(ent["sel"])) + (" capture" if ent["key"][-1] else "")
            if s not in sels:
                sels.append(s)
    out = {"word": STATE["xln"]["word"], "on": STATE["xln"]["on"], "served": served, "stock_by_name": stock, "routes": routes, "calls": dict(c),
           "errors": ERRORS["n"], "first_error": ERRORS["first"], "selections": sels}
    return {"xln": out} if comp is None else out


def census_line(comp="xln"):
    c = census(comp)
    rows = ",".join("%s:%d" % (k, v) for k, v in sorted(c["calls"].items(), key=lambda kv: -kv[1]) if not k.startswith(("route:", "refused:", "error:")))
    routes = ",".join("%s:%d" % (k[len("route:"):], v) for k, v in sorted(c["calls"].items()) if k.startswith("route:"))
    ref = ",".join("%s:%d" % (k[len("refused:"):], v) for k, v in sorted(c["calls"].items()) if k.startswith("refused:"))
    err = ",".join("%s:%d" % (k[len("error:"):], v) for k, v in sorted(c["calls"].items()) if k.startswith("error:"))
    line = "%s CENSUS component=%s word=%s on=%s served=%d stock_by_name=%d rows=%s routes=%s" % (PREFIX, comp, c["word"], c["on"], c["served"], c["stock_by_name"], rows or "-", routes or "-")
    if ref:
        line += " refused=%s" % ref
    if err or c["errors"]:
        line += " errors=%s(%s) first=%s" % (c["errors"], err, c["first_error"])
    line += " selections=" + (" ; ".join(c["selections"]) or "-")
    return line


def print_census():
    if STATE["xln"]["on"] or CENSUS["xln"]:
        print(census_line("xln"), file=sys.stderr, flush=True)


_ATEXIT = {"registered": False}


def register_exit_census():
    if not _ATEXIT["registered"]:
        import atexit
        atexit.register(print_census)
        _ATEXIT["registered"] = True
