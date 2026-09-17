"""fpf_rf3_apb_rows — RoseTTAFold3's pairformer attention-with-pair-bias core on the shared core's one provider (``opt_core.kernels.apb``).

The seam is the FPF arm's ``apb`` component (``fpf_rf3_adapter._apb_forward_v2``, the fast tier's AttentionPairBiasPairformerDeepspeed
forward: the pair-side LayerNorm + to_b served by the carried ``lnl_fused.ln_linear`` producer, the rest the module's statements).  Its head
sub-word ``apb.<word>`` names what computes the attention core — softmax_j(Q.K/sqrt(c) + b_ij) V, gated — after that producer:

    stmt (default)   the module's own statements, byte for byte: Q/sqrt(c) as a bf16 tensor division, the
                     [I, J, H] logits by einsum (+ the fp32 pair bias -> fp32 logits), softmax over j in fp32, the second einsum, the gate
    sdpa | fpf_apb | apb_attn
                     the provider's row of that name (row word = exactly that row, the provider's default rule): the logits never reach HBM.
                     TOLERANCE class against the statement (online softmax; the scalar ``Beta_II`` offset the pairformer passes — one element,
                     softmax-invariant — is not added; a per-pair ``Beta_II`` [I, I] rides the bias).  A refusing row (int32 offsets, dtype,
                     geometry, capture state) is DECLINED BY NAME to the statement (``SELECT ... REFUSED row=<r> kind=<k> -> stmt``), counted.
    fast | big     the provider's tier words: the row the cell table names on this card (APB_CELLS.json ``pf_h16d24``; the table's stock
                     row there is the SDPA statement, itself a row the provider serves)

There is no exact-class row for this statement (the provider's bitwise rows are fp32: ``dit_exact`` == ``sdpa_upcast``; RF3's core runs bf16
under autocast -> ``dit_exact`` refuses ``dtype`` by name), so the exact tier's ``sapb`` keeps the statement and takes no sub-word; the exact
tier's LayerNorms inside it are the ``xln`` component's (fpf_rf3_ln_rows).  The diffusion transformer's attention (``dattn`` / the package's
``dtk`` and ``mkdit`` levers) is not this module's seam.

Selection = the provider's ``select()``, memoised per (word, cc, dtype, H, D, I, S, capture) and printed once as a ``SELECT`` line with the
provider's ``describe()``; under CUDA stream capture (the ``tg`` trunk graph) the class is selected with ``capture=True`` (capture-unsafe rows —
``ds4sci`` — are never winners there), the capture query made only when the eager and graph selections differ.
Census at exit / describe(): ``CENSUS component=apb word=<w> served=<n> rows=<row|bf16|H<h>|D<d>:n> declined=<why:n> refused=<row:kind:n>
selections=<describe(); ...>``.
"""
import math
import sys

COMPONENT_WORDS = {"apb": ("stmt", "sdpa", "fpf_apb", "apb_attn", "fast", "big")}
DEFAULT_WORD = {c: w[0] for c, w in COMPONENT_WORDS.items()}
PREFIX = "[fpf_rf3 apb]"
STATE = {"apb": {"on": False, "word": None}}
_SEL = {}                                                   # (word, cc, dt, H, D, I, S, capture) -> entry
_CLASS = {}                                                 # (word, cc, dt, H, D, I, S) -> entry | ("ask",)
CENSUS = {"apb": {}}
ERRORS = {"n": 0, "first": None}
_PROC = {}


def parse_component(token: str):
    """'apb' -> ('apb', None); 'apb.fpf_apb' -> ('apb', 'fpf_apb'); anything else -> (None, None)."""
    base, dot, word = token.partition(".")
    if base not in COMPONENT_WORDS:
        return None, None
    if dot and word not in COMPONENT_WORDS[base]:
        raise ValueError("%s: unknown word %r for component %s (words: %s)" % (PREFIX, word, base, ", ".join(COMPONENT_WORDS[base])))
    return base, (word or None)


def strip_components(arm: str):
    """(the arm with ``apb.<word>`` handed on as ``apb`` — the adapter's own component: the producer, the binding, its census unchanged —,
    {"apb": <sub-word or None>})."""
    body, at, tail = arm.partition("@")
    words = {"apb": None}
    keep = []
    for p in [x for x in body.split("+") if x]:
        base, word = parse_component(p)
        if base is None:
            keep.append(p)
            continue
        words["apb"] = word
        keep.append("apb")
    out = "+".join(keep) or "stock"
    return out + (at + tail if at else ""), words


def set_apb_word(word):
    w = word or DEFAULT_WORD["apb"]
    if w not in COMPONENT_WORDS["apb"]:
        raise ValueError("%s: unknown apb word %r (words: %s)" % (PREFIX, w, ", ".join(COMPONENT_WORDS["apb"])))
    if w != DEFAULT_WORD["apb"]:
        _A()                                                 # the provider must import for a non-default word: fails HERE by name, before any binding
    if STATE["apb"]["word"] != w:
        _SEL.clear(); _CLASS.clear()
    STATE["apb"]["word"] = w
    return w


def core_word():
    """None under the default word (the adapter runs the statement untouched), else the sub-word in force."""
    w = STATE["apb"]["word"]
    return None if (w is None or w == DEFAULT_WORD["apb"]) else w


# ----------------------------------------------------------------------------------------------------------------- the provider
def _A():
    from opt_core.kernels import apb as A
    return A


def _proc(torch, dev):
    k = ("cc", dev)
    if k not in _PROC:
        _PROC[k] = tuple(torch.cuda.get_device_capability(dev))
        if "stack" not in _PROC:
            A = _A()
            try:
                _PROC["stack"] = A.stack_word(dev)
            except Exception:                                # noqa: BLE001
                _PROC["stack"] = None
    return _PROC[k]


def _select_one(A, cc, dt, H, D, I, S, word, capture):
    cell = A.cell_word("pf", heads=H, head_dim=D)
    kw = dict(word=word, samples=S, stack=_PROC.get("stack"), capture=capture, head_dim=D, heads=H)
    refusal = None
    sel = None
    try:
        sel = A.select(cc, dt, cell, I, **kw)
    except A.Refusal as r:                                   # a refusing row: declined BY NAME to the statement, the refusal kept
        refusal = r
    return sel, refusal, cell


def _say_select(ent):
    A = _A()
    sel, r = ent["sel"], ent.get("call_refusal") or ent["refusal"]
    word, cc, dt, H, D, I, S, capture = ent["key"]
    line = "%s SELECT component=apb asked=%s cc=%d.%d dtype=%s h=%d d=%d n=%d s=%d capture=%s cell=%s -> %s" % (
        PREFIX, word, cc[0], cc[1], str(dt).replace("torch.", ""), H, D, I, S, capture, ent["cell"], A.describe(sel) if sel is not None else "stmt (the module's statement)")
    if r is not None:
        line += " REFUSED row=%s kind=%s -> %s by name" % (r.row, r.kind, (A.arm_word(sel.row, sel.variant) if sel is not None else "stmt"))
    print(line, flush=True)


def _entry(torch, Q, H, D, I, S):
    A = _A()
    cc = _proc(torch, Q.device)
    word = STATE["apb"]["word"]
    ck = (word, cc, Q.dtype, H, D, I, S)
    cls = _CLASS.get(ck)
    if cls is None:
        ents = []
        for capture in (False, True):
            sel, refusal, cell = _select_one(A, cc, Q.dtype, H, D, I, S, word, capture)
            ent = {"sel": sel, "refusal": refusal, "call_refusal": None, "calls": 0, "cell": cell, "key": ck + (capture,)}
            _SEL[ck + (capture,)] = ent
            ents.append(ent)
        r0 = ents[0]["sel"]; r1 = ents[1]["sel"]
        same = (r0 is None and r1 is None) or (r0 is not None and r1 is not None and (r0.row, r0.variant) == (r1.row, r1.variant))
        cls = ents[0] if same else ("ask",)
        _CLASS[ck] = cls
        _say_select(ents[0])
        if not same:
            _say_select(ents[1])
    if isinstance(cls, tuple):
        return _SEL[ck + (bool(torch.cuda.is_current_stream_capturing()),)]
    return cls


def _count(ent, dt, H, D, served):
    ent["calls"] += 1
    c = CENSUS["apb"]
    if served:
        A = _A()
        k = "%s|%s|H%d|D%d" % (A.arm_word(ent["sel"].row, ent["sel"].variant), str(dt).replace("torch.", "").replace("bfloat16", "bf16").replace("float32", "fp32"), H, D)
    else:
        r = ent.get("call_refusal") or ent["refusal"]
        k = "declined:%s" % ("refused:%s:%s" % (r.row, r.kind) if r is not None else "stmt")
    c[k] = c.get(k, 0) + 1


def plan_shapes(q_shape, bias_shape, H):
    """The statement's broadcasting, planned once per call: Q/K/V/G [..q, I, H, D] against the bias planes [..b, I, J, >=H] (RF3's einsum
    '...ihd,...jhd->...ijh' + B then '...ijh,...jhc->...ihc' broadcasts the leading dims: the confidence head's UNBATCHED single [I, H, D]
    against its per-sample pair [S, I, J, H] yields a per-sample output) -> (lead, S, Sb, I, J, D): the output's leading dims, the sample
    count the rows see, the bias's own sample count (1 = shared by the S samples, else == S), or None when the leading dims do not broadcast
    to a form the rows serve (declined by name: 'lead:<q lead>x<bias lead>')."""
    I, D = int(q_shape[-3]), int(q_shape[-1])
    J = int(bias_shape[-2])
    lq, lb = tuple(int(v) for v in q_shape[:-3]), tuple(int(v) for v in bias_shape[:-3])
    n = max(len(lq), len(lb))
    lq_, lb_ = (1,) * (n - len(lq)) + lq, (1,) * (n - len(lb)) + lb
    lead = []
    for a, b in zip(lq_, lb_):
        if a != b and 1 not in (a, b):
            return None
        lead.append(max(a, b))
    lead = tuple(lead)
    S = 1
    for v in lead:
        S *= v
    Sb = 1
    for v in lb_:
        Sb *= v
    if Sb not in (1, S):                                      # a bias batched along a dim the queries broadcast AND unbatched along one they carry: not a [1|S] plane set
        return None
    return lead, S, Sb, I, J, D


def serve_core(mod, Q_IH, K_IH, V_IH, b16, Beta_II, G_IH):
    """The gated attention core [..lead, I, H*D] through the provider's row, or None = declined by name (the caller runs the statement).
    Q/K/V/G [..q, I, H, D] bf16 (G after the sigmoid), b16 [..b, I, J, NOUT>=H] = the producer's planes, Beta_II None | [1] | [..b, I, I];
    ..lead = the broadcast of ..q and ..b exactly as the statement's einsums broadcast them (``plan_shapes``)."""
    import torch
    H = int(mod.n_head); D = int(mod.c)
    bias = b16[..., :H]
    if Beta_II is not None and Beta_II.numel() > 1:         # per-pair offsets ride the bias; a one-element offset is softmax-invariant (not added)
        bias = bias + Beta_II[..., None].to(bias.dtype)
    plan = plan_shapes(tuple(Q_IH.shape), tuple(bias.shape), H)
    if plan is None:
        k = "declined:lead:%sx%s" % ("x".join(str(int(v)) for v in Q_IH.shape[:-3]) or "-", "x".join(str(int(v)) for v in bias.shape[:-3]) or "-")
        CENSUS["apb"][k] = CENSUS["apb"].get(k, 0) + 1
        return None
    lead, S, Sb, I, J, D_ = plan
    try:
        ent = _entry(torch, Q_IH, H, D, I, S)
    except Exception as e:                                   # noqa: BLE001
        ERRORS["n"] += 1; ERRORS["first"] = ERRORS["first"] or ("select: %s: %s" % (type(e).__name__, str(e)[:160]))
        k = "error:select:%s" % type(e).__name__
        CENSUS["apb"][k] = CENSUS["apb"].get(k, 0) + 1
        return None
    if ent["sel"] is None or ent["sel"].row in ("naive", "sdpa_upcast"):
        _count(ent, Q_IH.dtype, H, D, False)
        return None
    A = _A()
    bias4 = bias.reshape((Sb, I, J, H)).permute(0, 3, 1, 2).contiguous()      # [1|S, H, I, J] head-major planes, the layout the rows read
    if bias4.dtype != Q_IH.dtype:
        bias4 = bias4.to(Q_IH.dtype)
    tail = (I, H, D)
    q4 = Q_IH.expand(lead + tail).reshape((S,) + tail); k4 = K_IH.expand(lead + tail).reshape((S,) + tail); v4 = V_IH.expand(lead + tail).reshape((S,) + tail)
    try:
        o, _ = A.pair_bias_attention(q4, k4, v4, bias4, selection=ent["sel"], layout="snhd", cell=ent["cell"] or "pf_h16d24", scale=1.0 / math.sqrt(D))
    except A.Refusal as r:                                   # the row's own admission at call time: this class declines by name from here on
        ent["call_refusal"] = r; ent["sel"] = None
        _say_select(ent); _count(ent, Q_IH.dtype, H, D, False)
        return None
    except Exception as e:                                   # noqa: BLE001  (build / launch failure: named, counted, declined)
        ERRORS["n"] += 1; ERRORS["first"] = ERRORS["first"] or ("%s: %s: %s" % (A.arm_word(ent["sel"].row, ent["sel"].variant), type(e).__name__, str(e)[:160]))
        k = "error:%s:%s" % (A.arm_word(ent["sel"].row, ent["sel"].variant), type(e).__name__)
        CENSUS["apb"][k] = CENSUS["apb"].get(k, 0) + 1
        ent["call_refusal"] = A.Refusal("raised:%s" % type(e).__name__, ent["sel"].row, "stmt"); ent["sel"] = None
        _say_select(ent)
        return None
    if tuple(o.shape) != (S,) + tail:                         # a row answering in another layout is not this statement's: declined by name, never reshaped into place
        k = "declined:out_shape:%s" % "x".join(str(int(v)) for v in o.shape)
        CENSUS["apb"][k] = CENSUS["apb"].get(k, 0) + 1
        return None
    _count(ent, Q_IH.dtype, H, D, True)
    o = o.reshape(lead + tail)
    return (G_IH * o).flatten(start_dim=-2)


# ----------------------------------------------------------------------------------------------------------------- census
def reset_census():
    CENSUS["apb"].clear(); _SEL.clear(); _CLASS.clear()
    ERRORS["n"] = 0; ERRORS["first"] = None


def census(comp=None):
    c = CENSUS["apb"]
    served = sum(n for k, n in c.items() if not k.startswith(("declined:", "error:")))
    declined = sum(n for k, n in c.items() if k.startswith("declined:"))
    sels = []
    if _SEL:
        try:
            A = _A()
        except Exception:                                    # noqa: BLE001
            A = None
        for ent in _SEL.values():
            if ent["calls"]:
                s = (A.describe(ent["sel"]) if (A and ent["sel"] is not None) else "stmt") + (" capture" if ent["key"][-1] else "")
                r = ent.get("call_refusal") or ent["refusal"]
                if r is not None:
                    s += " REFUSED %s:%s" % (r.row, r.kind)
                if s not in sels:
                    sels.append(s)
    out = {"word": STATE["apb"]["word"], "on": STATE["apb"]["on"], "served": served, "declined": declined, "calls": dict(c), "errors": ERRORS["n"],
           "first_error": ERRORS["first"], "selections": sels}
    return {"apb": out} if comp is None else out


def census_line(comp="apb"):
    c = census(comp)
    rows = ",".join("%s:%d" % (k, v) for k, v in sorted(c["calls"].items(), key=lambda kv: -kv[1]) if not k.startswith(("declined:", "error:")))
    dec = ",".join("%s:%d" % (k[len("declined:"):], v) for k, v in sorted(c["calls"].items()) if k.startswith("declined:"))
    err = ",".join("%s:%d" % (k[len("error:"):], v) for k, v in sorted(c["calls"].items()) if k.startswith("error:"))
    line = "%s CENSUS component=%s word=%s on=%s served=%d rows=%s declined=%s" % (PREFIX, comp, c["word"], c["on"], c["served"], rows or "-", dec or "-")
    if err or c["errors"]:
        line += " errors=%s(%s) first=%s" % (c["errors"], err, c["first_error"])
    line += " selections=" + (" ; ".join(c["selections"]) or "-")
    return line


def print_census():
    if STATE["apb"]["on"] and (STATE["apb"]["word"] not in (None, DEFAULT_WORD["apb"]) or CENSUS["apb"]):
        print(census_line("apb"), file=sys.stderr, flush=True)


_ATEXIT = {"registered": False}


def register_exit_census():
    if not _ATEXIT["registered"]:
        import atexit
        atexit.register(print_census)
        _ATEXIT["registered"] = True
