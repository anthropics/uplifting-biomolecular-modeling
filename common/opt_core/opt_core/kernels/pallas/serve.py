"""Call faces of the JAX-family provider (opt_core.kernels.pallas): one function per op, every row behind it.  jax is imported when a
face is called, never at import.  Each face takes ``word=`` (row | arm | tier word, see the package doc), resolves it with ``select``
(``jax_line`` / ``cc`` default to the running jax and the default backend's device), serves the first candidate that engages and records
a refusing row BY NAME (``COUNTS``, ``LAST_REFUSAL``; ``strict=True`` raises the Refusal instead of walking on).

    attention(q, k, v, bias, key_mask=None, scale=None, *, word, kind="tri", direction="fwd", layout="BSHD", ...)     [B,S,H,D] | [B,H,S,D]
    triangle_attention_block(act, pair_mask, params, *, orientation, form, unit=None, word, ...)                          act [N,N,C]
    triangle_multiplication(act, mask, params, *, equation, form, unit=None, word, ...)                                   act [N,N,C]
    glu_transposed_masked(x, weight, mask_t, *, word, ...)                                                                 the AF3 contraction prologue
    transition(x, params, *, activation, form, word, ...)                                                                  x [..., C]
    layer_norm(x, scale, offset, *, unit, word, ...)                                                                       x [..., C]
    outer_product_mean(act, mask, params, *, form, word, ...)                                                              act [S,N,C_m]

Parameter dicts are the math layout of ``opt_core.kernels.fpf_pallas_serve`` (``attn_params`` keywords: ln_scale, ln_offset, q_w/k_w/v_w
[C,H,D], bias_w [C,H], gate_w [C,H,D], out_w [H,D,C], gate_b, out_b; ``trimul_params`` keywords: ln_in_scale, ln_in_offset, left_w,
right_w, left_gate_w, right_gate_w [C,Ch], ln_c_scale, ln_c_offset [Ch], out_w [Ch,C], gate_w [C,C] + the six optional biases);
transition: ln_scale, ln_offset, w1 [C,F] (relu) | [C,2F] (swiglu: columns [:F] gate a, [F:] b), b1, w2 [F,C], b2; opm: ln_scale,
ln_offset (optional: LayerNorm applied when given), left_w [C_m,c], left_b, right_w, right_b, out_w [c,c,F], out_b.
"""
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import census_stepaside                                          # the cell-coverage census: a serving step-aside re-recorded by name
from . import (Refusal, Selection, select, family, families, attn_family_facts, jax_line_of, cc_of, dtype_word, parse_arm, has_backward, CC_ENV, PAD_HEAD_DIM_TO,
               NO_BACKWARD, NEEDS_TOKAMAX, NEEDS_JAX_FFI, NEEDS_HAIKU, N_NOT_TILE_MULTIPLE, PRECISION_REQUIRED, ROW_ERROR, STOCK_ROWS)

COUNTS: Dict[str, int] = {}
LAST_REFUSAL: Dict[str, Any] = {}
LN_EPS = 1e-5


# ----------------------------------------------------------------------------------------------------------------- environment facts
def running_jax_line() -> str:
    import jax  # noqa: WPS433
    return jax_line_of(jax.__version__)


def running_cc() -> str:
    env = os.environ.get(CC_ENV)
    if env:
        return cc_of(env)
    import jax  # noqa: WPS433
    d = jax.devices()[0]
    cc = getattr(d, "compute_capability", None)
    if cc is None:
        raise Refusal("no GPU device on the default backend (platform %s): the Pallas rows lower for GPUs only; the stock rows serve" % d.platform,
                      kind="backend_not_gpu", row=None, fallback="xla")
    return cc_of(cc)


def _record(ref: Refusal, face: str) -> None:
    COUNTS["refused:%s:%s" % (ref.row, ref.kind)] = COUNTS.get("refused:%s:%s" % (ref.row, ref.kind), 0) + 1
    LAST_REFUSAL.clear()
    LAST_REFUSAL.update(face=face, row=ref.row, kind=ref.kind, fallback=ref.fallback, message=str(ref))


def _served(arm: str, face: str) -> None:
    COUNTS["served:%s:%s" % (face, arm)] = COUNTS.get("served:%s:%s" % (face, arm), 0) + 1


def kinds_with_depth() -> List[str]:
    """attention kinds whose family carries a sequence depth (_s<n>): msarow, msacol, extramsa_slab512, af3_dit, ..."""
    out = set()
    for f in families("attn"):
        fx = attn_family_facts(f)
        if fx["n_seq"] is not None:
            out.add(fx["kind"])
    return sorted(out)


def resolve(op: str, fam: str, dtype: Any, n_tokens: int, *, word: str, direction: str = "fwd", jax_line: Any = None, cc: Any = None,
            prefer: Optional[Sequence[str]] = None, key_masked: bool = True) -> Selection:
    line = jax_line if jax_line is not None else running_jax_line()
    ccw = cc if cc is not None else running_cc()
    return select(line, ccw, dtype, op, fam, n_tokens, direction, word=word, prefer=prefer, key_masked=key_masked)


DEAD_ARMS: Dict[tuple, str] = {}          # (face, arm, cc) -> "error:<ExcType>": an INHERITED (unmeasured-cc) row that failed to build / compile / launch on this
PROBED_ARMS: set = set()                  # part in this process -- never retried; (face, arm, cc) probed once (an eager launch on zeros of the call's shapes)
LAUNCH_ERROR = "launch_error"


def _is_oom(e: BaseException) -> bool:
    from opt_core.oom import is_oom  # noqa: WPS433
    return is_oom(e)


def _inherited(sel: Selection) -> bool:
    from . import INHERITED_CC_TOKEN  # noqa: WPS433
    return INHERITED_CC_TOKEN in str(getattr(sel, "note", "") or "")


def _concrete_zeros(tree):
    """Concrete zero arrays with the shapes / dtypes of `tree`'s array leaves (others pass through) -- the probe's inputs, also under an enclosing trace."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    def z(v):
        return jnp.zeros(tuple(v.shape), v.dtype) if hasattr(v, "shape") and hasattr(v, "dtype") else v
    with jax.ensure_compile_time_eval():
        return jax.tree_util.tree_map(z, tree)


def _walk(sel: Selection, face: str, serve_one, strict: bool, probe=None):
    """Serve the first candidate that engages; a refusing row is recorded by name and the next measured arm serves ('xla' last).

    INHERITED rows (a cc without a measured cell for this key, served by another column's source-compiled rows -- unmeasured_cc_rule) are GUARDED:
    the row's first use in the process is probed (``probe(arm)``: the face re-entered EAGERLY on concrete zeros of the call's shapes with the row word,
    so a kernel that does not build / compile / launch on this part fails here, inside the provider, also under an enclosing jit trace), and any
    exception that is not an out-of-memory -- at the probe or at the call -- marks the arm dead for the process (census 'stepped_aside:error:<ExcType>',
    once), the walk serving the next candidate (the donor cell's next portable row, the statement last) BY NAME.  A dead arm is never retried.
    Out-of-memory is re-raised as is.  Measured columns are unchanged (their rows were measured to launch there; a raw error is a real error)."""
    last = first = None
    passed = []
    inherited = _inherited(sel)
    cands = list(sel.candidates)
    for i, arm in enumerate(cands):
        row, ip, setting = parse_arm(arm)
        dk = (face, arm, sel.cc)
        if inherited and dk in DEAD_ARMS:
            ref = Refusal("%s: %s on cc %s at first use in this process -- not retried; next candidate by name" % (arm, DEAD_ARMS[dk], sel.cc),
                          kind=DEAD_ARMS[dk], row=row, fallback=cands[i + 1] if i + 1 < len(cands) else "xla")
            _record(ref, face); last = ref; first = ref if first is None else first; passed.append(arm)
            if strict or len(cands) == 1:
                raise ref
            continue
        try:
            if inherited and probe is not None and row not in STOCK_ROWS and dk not in PROBED_ARMS:
                PROBED_ARMS.add(dk)
                try:
                    import jax  # noqa: WPS433
                    with jax.ensure_compile_time_eval():
                        jax.block_until_ready(probe(arm))
                except Refusal:
                    raise
                except Exception as e:                     # noqa: BLE001 - is_oom first; anything else = the inherited row does not run on this part
                    if _is_oom(e):
                        raise
                    DEAD_ARMS[dk] = "error:%s" % type(e).__name__
                    raise Refusal("%s: failed to build / launch on cc %s (inherited row; %s: %s) -- stepped aside by name" % (arm, sel.cc, type(e).__name__, str(e)[:200]),
                                  kind=DEAD_ARMS[dk], row=row, fallback=cands[i + 1] if i + 1 < len(cands) else "xla")
            out = serve_one(arm, row, ip, setting)
            _served(arm, face)
            if passed:
                census_stepaside(sel, arm, first.row, first.kind, passed)
            return out
        except Refusal as ref:
            _record(ref, face)
            last = ref
            first = ref if first is None else first
            passed.append(arm)
            if strict or len(cands) == 1:
                raise
            continue
        except Exception as e:                             # noqa: BLE001 - is_oom first; inherited rows only: a raw build / launch error becomes a named step-aside
            if not inherited or row in STOCK_ROWS or _is_oom(e):
                raise
            DEAD_ARMS[dk] = "error:%s" % type(e).__name__
            ref = Refusal("%s: failed at the call on cc %s (inherited row; %s: %s) -- stepped aside by name" % (arm, sel.cc, type(e).__name__, str(e)[:200]),
                          kind=DEAD_ARMS[dk], row=row, fallback=cands[i + 1] if i + 1 < len(cands) else "xla")
            _record(ref, face); last = ref; first = ref if first is None else first; passed.append(arm)
            if strict or len(cands) == 1:
                raise ref
            continue
    raise last if last is not None else Refusal("%s: no candidate served" % face, kind=ROW_ERROR)


def forward_only(fn, row: str, fallback: Optional[str]):
    """Wrap a forward-only row so that differentiating THROUGH it raises the row's no_backward refusal by name (jax.custom_vjp)."""
    import jax  # noqa: WPS433

    @jax.custom_vjp
    def f(*args):
        return fn(*args)

    def f_fwd(*args):
        raise Refusal("%s: forward only (no VJP); a differentiated call is refused by name -- rows with a backward: see select(direction='fwdbwd')" % row,
                      kind=NO_BACKWARD, row=row, fallback=fallback)

    def f_bwd(res, ct):  # pragma: no cover - never reached
        raise Refusal("%s: forward only" % row, kind=NO_BACKWARD, row=row, fallback=fallback)

    f.defvjp(f_fwd, f_bwd)
    return f


def _split_static(d: Dict[str, Any]):
    """({array leaves}, {python scalars / strings}) -- only arrays ride through jit / custom_vjp argument lists."""
    arrays = {k: v for k, v in d.items() if hasattr(v, "dtype") and hasattr(v, "shape")}
    return arrays, {k: v for k, v in d.items() if k not in arrays}


def _tokamax():
    try:
        import tokamax  # noqa: WPS433
    except Exception as e:  # noqa: BLE001 - any import failure of the optional library is the named refusal
        raise Refusal("tokamax is not importable on this stack (%s: %s)" % (type(e).__name__, e), kind=NEEDS_TOKAMAX, row="tokamax", fallback="xla")
    return tokamax


# ----------------------------------------------------------------------------------------------------------------- op: attention core
def reference_attention(q, k, v, bias=None, key_mask=None, scale=None, *, layout="BSHD", precision=None):
    """The XLA statement (row 'xla'): logits = scale*q.k (+ bias[h] shared over B) (+ -1e9 where key_mask is False) -> softmax -> @ v,
    softmax in f32 for 16-bit inputs' logits as jax.nn.softmax computes, products at ``precision`` (None = the backend default)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    dt = q.dtype
    D = int(q.shape[-1])
    sc = (D ** -0.5) if scale is None else scale
    qs = (q * sc).astype(dt)
    eq_s, eq_o = ("bqhc,bkhc->bhqk", "bhqk,bkhc->bqhc") if layout == "BSHD" else ("bhqc,bhkc->bhqk", "bhqk,bhkc->bhqc")
    logits = jnp.einsum(eq_s, qs, k, precision=precision)
    if bias is not None:
        b = bias if bias.ndim == 4 else bias[None]
        logits = logits + b.astype(logits.dtype)
    if key_mask is not None:
        logits = logits + (1e9 * (key_mask.astype(jnp.float32) - 1.0)).astype(logits.dtype)[:, None, None, :]
    w = jax.nn.softmax(logits, axis=-1).astype(dt)
    return jnp.einsum(eq_o, w, v, precision=precision)


def cudnn_lengths_attention(q, k, v, key_mask, scale):
    """Row 'cudnn' on a BIAS-FREE call with a key mask: ``jax.nn.dot_product_attention(implementation="cudnn")`` in its padding-mask form --
    ``query_seq_lengths`` = Sq for every batch row, ``key_value_seq_lengths`` = the row's count of attended keys -- so no ``[B,1,Sq,Sk]``
    mask / bias operand is built (the library folds a ``mask=`` into a dense bias for cuDNN, and cuDNN refuses a ``[B,1,1,Sk]`` one: the
    dense form is what a biased call runs).  q, k, v ``[B,S,H,D]``; key_mask ``[B,Sk]`` bool (True = attend).  The lengths form attends the
    FIRST ``n`` keys of a row, so: a row whose attended keys are not a prefix (a hole) is detected in graph and the whole call then runs on
    keys / values stably permuted attended-first (``lax.cond``; the same attention set, never a wrong answer); a row with NO attended key
    gets the full length with its queries zeroed -- uniform weights over every key, which is the stock statement's answer there (-1e9 on
    every logit).  Same softmax / accumulation as the dense form; the library's own refusals (dtype, head dim, card, a jax without the
    length operands) surface as this row's ``row_error`` refusal in the caller."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    B, Sq, Sk = int(q.shape[0]), int(q.shape[1]), int(k.shape[1])
    km = key_mask.astype(bool)
    lens = jnp.sum(km, axis=-1).astype(jnp.int32)                                          # [B] attended keys per row
    empty = lens == 0
    qz = jnp.where(empty[:, None, None, None], jnp.zeros((), q.dtype), q)                   # no attended key: zero logits -> uniform over the full length
    kv_len = jnp.where(empty, jnp.int32(Sk), lens)
    q_len = jnp.full((B,), Sq, dtype=jnp.int32)
    holes = jnp.any(km[:, 1:] & ~km[:, :-1])                                                # some row's attended keys are not a prefix

    def run(kk, vv):
        return jax.nn.dot_product_attention(qz, kk, vv, scale=scale, query_seq_lengths=q_len, key_value_seq_lengths=kv_len, implementation="cudnn")

    def permuted():
        order = jnp.argsort(~km, axis=-1, stable=True)                                      # attended keys first, in their original order
        idx = order[:, :, None, None]
        return run(jnp.take_along_axis(k, idx, axis=1), jnp.take_along_axis(v, idx, axis=1))

    return jax.lax.cond(holes, permuted, lambda: run(k, v))


def pad_head_dim(q, k, v, to: int = PAD_HEAD_DIM_TO):
    import jax.numpy as jnp  # noqa: WPS433
    D = int(q.shape[-1])
    if D >= to:
        return q, k, v, D
    w = [(0, 0)] * (q.ndim - 1) + [(0, to - D)]
    return jnp.pad(q, w), jnp.pad(k, w), jnp.pad(v, w), D


def attention(q, k, v, bias=None, key_mask=None, scale=None, *, word: str, kind: str = "tri", n_seq: Optional[int] = None, direction: str = "fwd",
              layout: str = "BSHD", jax_line: Any = None, cc: Any = None, prefer: Optional[Sequence[str]] = None, strict: bool = False,
              config: Optional[Dict[str, Any]] = None, selection: Optional[Selection] = None):
    """Attention core with a pair bias shared over the batch rows and a key mask.  ``layout`` "BSHD" (q [B,S,H,D], the module layout) or
    "BHSD"; bias [H,Sq,Sk] (or [1,H,Sq,Sk]) or None (a bias-free call -- MSA column attention: the Pallas rows get a zeros bias operand, the
    library rows none, row 'cudnn' its key-lengths form: ``cudnn_lengths_attention``); key_mask [B,Sk] bool (True = attend) or None; scale
    None = D**-0.5 (a caller that scaled q itself passes 1.0).  ``kind`` + q's (heads, head_dim) (+ ``n_seq``: the MSA kinds' depth, default
    B) name the family.  Returns q's layout / dtype."""
    if layout not in ("BSHD", "BHSD"):
        raise ValueError("layout %r: BSHD | BHSD" % (layout,))
    B, S, H, D = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3])) if layout == "BSHD" else (int(q.shape[0]), int(q.shape[2]), int(q.shape[1]), int(q.shape[3]))
    if selection is None:
        with_depth = kind in kinds_with_depth()
        fam = family("attn", kind=kind, heads=H, head_dim=D, n_seq=((n_seq if n_seq is not None else B) if with_depth else None))
        selection = resolve("attn", fam, q.dtype, S, word=word, direction=direction, jax_line=jax_line, cc=cc, prefer=prefer, key_masked=key_mask is not None)
    sel = selection

    def one(arm, row, ip, setting):
        cfg = dict(sel.config if arm == sel.arm else {})
        cfg.update(config or {})
        return _attention_row(row, ip, cfg, q, k, v, bias, key_mask, scale, layout=layout, direction=direction, pad_to=sel.pad_head_dim_to or (PAD_HEAD_DIM_TO if D < PAD_HEAD_DIM_TO else None))
    return _walk(sel, "attention", one, strict, probe=lambda arm: attention(*_concrete_zeros((q, k, v, bias, key_mask)), scale=scale, word=arm, kind=kind, n_seq=n_seq, direction=direction,
                                                                       layout=layout, jax_line=jax_line, cc=cc, strict=True))


def _attention_row(row, ip, cfg, q, k, v, bias, key_mask, scale, *, layout, direction, pad_to):
    """One row on one call.  head_dim below ``pad_to`` (16): q/k/v zero-padded on the head dim for every row but the stock statement and the
    output sliced back -- exact (zero columns add nothing to q.k; the padded output columns are dropped; the scale is the caller's)."""
    D = int(q.shape[-1])
    sc = (D ** -0.5) if scale is None else scale
    if row == "xla" or not pad_to or D >= pad_to:
        return _attention_row_at(row, ip, cfg, q, k, v, bias, key_mask, sc, layout=layout, direction=direction)
    qp, kp, vp, D0 = pad_head_dim(q, k, v, pad_to)
    return _attention_row_at(row, ip, cfg, qp, kp, vp, bias, key_mask, sc, layout=layout, direction=direction)[..., :D0]


def _attention_row_at(row, ip, cfg, q, k, v, bias, key_mask, sc, *, layout, direction):
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    dt = q.dtype
    D = int(q.shape[-1])
    scale = sc
    f32 = dtype_word(dt) == "f32"
    if row == "xla":
        return reference_attention(q, k, v, bias, key_mask, scale, layout=layout)
    if row in ("xla_sdpa", "cudnn"):
        qs, kk, vv = (q, k, v) if layout == "BSHD" else (jnp.swapaxes(q, 1, 2), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2))
        b = None if bias is None else (bias if bias.ndim == 4 else bias[None]).astype(dt)
        m = None if key_mask is None else key_mask[:, None, None, :]
        try:
            if row == "cudnn" and b is None and key_mask is not None:
                o = cudnn_lengths_attention(qs, kk, vv, key_mask, sc)                    # bias-free + key mask: cuDNN's padding-mask (per-row key LENGTHS) form, no [B,1,Sq,Sk] operand
            else:
                o = jax.nn.dot_product_attention(qs, kk, vv, bias=b, mask=m, scale=sc, implementation=("cudnn" if row == "cudnn" else "xla"))
        except Refusal:
            raise
        except Exception as e:  # noqa: BLE001 - the library's own refusal of a shape / dtype / card is this row's named refusal
            raise Refusal("%s: jax.nn.dot_product_attention refused: %s: %s" % (row, type(e).__name__, str(e)[:200]), kind=ROW_ERROR, row=row, fallback="xla")
        return o if layout == "BSHD" else jnp.swapaxes(o, 1, 2)
    if row == "tokamax":
        tk = _tokamax()
        qs, kk, vv = (q, k, v) if layout == "BSHD" else (jnp.swapaxes(q, 1, 2), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2))
        b = None if bias is None else (bias if bias.ndim == 4 else bias[None])
        m = None if key_mask is None else key_mask[:, None, None, :]
        o = tk.dot_product_attention(qs, kk, vv, bias=b, mask=m, scale=sc, implementation=cfg.get("implementation"))
        return o if layout == "BSHD" else jnp.swapaxes(o, 1, 2)
    if row == "triattn_xla":
        try:
            from .. import triattn_xla as X  # noqa: WPS433
        except Exception as e:  # noqa: BLE001 - an import failure of the bridge (no jax.ffi on this jax) is the named refusal
            raise Refusal("triattn_xla is not importable here (%s: %s)" % (type(e).__name__, e), kind=NEEDS_JAX_FFI, row=row, fallback="pallas_attn")
        qh = q if layout == "BHSD" else jnp.swapaxes(q, 1, 2)
        kh = k if layout == "BHSD" else jnp.swapaxes(k, 1, 2)
        vh = v if layout == "BHSD" else jnp.swapaxes(v, 1, 2)
        b = bias if bias.ndim == 3 else bias[0]
        vjp = cfg.get("vjp")                                                      # 'triattn_xla@vjp': the bridge's differentiable row (forward cubin + Pallas backward), by word only
        try:
            o = X.triangle_attention(qh[None], kh[None], vh[None], b[None], mask=(None if key_mask is None else key_mask[None]), scale=sc, layout="BNHSD", **({"vjp": vjp} if vjp else {}))[0]
        except X.Refused as e:
            raise Refusal("triattn_xla refused by name: %s" % e, kind=getattr(e, "kind", NEEDS_JAX_FFI), row=row, fallback=("cd_triatt" if vjp else "pallas_attn"))
        return o if layout == "BHSD" else jnp.swapaxes(o, 1, 2)
    if row == "fpf_core":
        from .. import fpf_pallas_serve as S  # noqa: WPS433
        qb, kb, vb = (q, k, v) if layout == "BSHD" else (jnp.swapaxes(q, 1, 2), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2))
        if abs(sc - D ** -0.5) > 1e-12:
            qb = (qb.astype(jnp.float32) * (sc * math.sqrt(D))).astype(dt)          # the kernel scales by D**-0.5 itself
        S_ = int(qb.shape[1])
        mult = 128
        Sp = ((S_ + mult - 1) // mult) * mult
        if bias is None:                                                                # a bias-free call (MSA column attention): the kernel's bias operand is zeros [1,H,Sq,Sk]
            bias = jnp.zeros((1, int(qb.shape[2]), S_, int(kb.shape[1])), dt)
        b = bias if bias.ndim == 4 else bias[None]
        km = key_mask if key_mask is not None else jnp.ones((qb.shape[0], S_), dtype=bool)
        if Sp != S_:                                                                    # zero-pad queries and keys to the tile multiple; padded keys masked: exact
            pw = [(0, 0), (0, Sp - S_), (0, 0), (0, 0)]
            qb, kb, vb = jnp.pad(qb, pw), jnp.pad(kb, pw), jnp.pad(vb, pw)
            b = jnp.pad(b, [(0, 0), (0, 0), (0, Sp - S_), (0, Sp - S_)])
            km = jnp.pad(km, [(0, 0), (0, Sp - S_)])
        fn = (lambda a, b_, c_, d_, e_: S.attention_core(a, b_, c_, d_, e_))
        fn = forward_only(fn, "fpf_core", "pallas_attn")
        o = fn(qb, kb, vb, b.astype(dt), km)
        o = o[:, :S_]
        return o if layout == "BSHD" else jnp.swapaxes(o, 1, 2)
    # the Pallas rows: heads-major [B,H,S,D], bias [H,Sq,Sk] in the activation dtype, key mask [B,Sk] bool, scale
    qh, kh, vh = (q, k, v) if layout == "BHSD" else (jnp.swapaxes(q, 1, 2), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2))
    if bias is None:                                                                    # a bias-free call (MSA column attention): the kernels' bias operand is zeros [H,Sq,Sk] (what the msacol cells timed)
        bias = jnp.zeros((int(qh.shape[1]), int(qh.shape[2]), int(kh.shape[2])), dt)
    b = bias if bias.ndim == 3 else bias[0]
    km = key_mask if key_mask is not None else jnp.ones((qh.shape[0], kh.shape[2]), dtype=bool)
    if row == "pallas_attn":
        from .. import pallas_attn_serve as PS  # noqa: WPS433
        M = PS.kernel_module(require_gpu=False)
        kw = {k_: v_ for k_, v_ in cfg.items() if k_ in ("bq", "bk", "num_warps", "num_stages", "dbias_dtype", "bq_bwd", "bk_bwd", "batch_chunk", "precise_bwd", "bwd_f32_precision", "dq", "dbias")}
        if f32 and ip is not None:
            kw["f32_precision"] = ip
        op = M.make_flash_attention(**kw) if kw else M.make_flash_attention()
        o = op(qh, kh, vh, b.astype(dt), km, sc)
    elif row == "cd_triatt":
        from ..pallas_triatt import triatt_attn as CT  # noqa: WPS433  (carried by the core beside this package)
        kw = {k_: v_ for k_, v_ in cfg.items() if k_ in ("bq", "bk", "num_warps", "num_stages")}
        if f32 and ip is not None:
            kw["f32_precision"] = ip
        op = CT.make_attention(**kw) if kw else CT.attention
        o = op(qh, kh, vh, b.astype(dt), km, sc)
    elif row == "rowshared":
        from . import rowshared_flash_pallas as RS  # noqa: WPS433
        kw = dict(rows=int(cfg.get("rows", 2)), order=cfg.get("order", "grouped"))
        if f32:
            kw["f32_precision"] = ip or "tf32"
        try:
            op = RS.make_rowshared_attention(**kw)
            o = op(qh, kh, vh, b.astype(dt), km, sc)
        except RS.RowAttnRefusal as e:
            raise Refusal("rowshared refused by name: %s" % e, kind=getattr(e, "kind", ROW_ERROR), row=row, fallback="pallas_attn")
    else:
        raise Refusal("row %r does not serve the attention core" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return o if layout == "BHSD" else jnp.swapaxes(o, 1, 2)


# ----------------------------------------------------------------------------------------------------------------- op: triangle attention module
def _ln(x, scale, offset, eps=LN_EPS):
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    f32 = jnp.float32
    xf = x.astype(f32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    y = (xf - mean) * jax.lax.rsqrt(var + eps)
    if scale is not None:
        y = y * scale.astype(f32)
    if offset is not None:
        y = y + offset.astype(f32)
    return y


def reference_triangle_attention(act, pair_mask, params: Dict[str, Any], *, orientation: str, core=None, precision=None):
    """The AF2-family TriangleAttention module statement at act.dtype (row 'xla' when ``core`` is None; the 'proj+<core>' rows pass the
    core row): LayerNorm (f32 statistics) -> [transpose for the ending node] -> q/k/v (q scaled by D**-0.5) / pair bias / gate
    projections -> core -> sigmoid gate -> output projection (+ biases when params carries gate_b / out_b) -> [transpose back]."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    dt = act.dtype
    T = {"starting": False, "ending": True, "per_row": False, "per_column": True}[orientation]
    C, H, D = (int(s) for s in params["q_w"].shape)
    c = lambda w: w.astype(dt)  # noqa: E731
    x = _ln(act, params["ln_scale"], params["ln_offset"]).astype(dt)
    m = pair_mask
    if T:
        x = jnp.swapaxes(x, 0, 1)
        m = jnp.swapaxes(m, 0, 1)
    kmask = m > 0                                                                  # [b, k]
    nb = jnp.einsum("qkc,ch->hqk", x, c(params["bias_w"]), precision=precision)  # [H, S, S] pair bias from the (transposed) normalised act
    q = jnp.einsum("bsc,chd->bshd", x, c(params["q_w"]), precision=precision) * (D ** -0.5)
    k = jnp.einsum("bsc,chd->bshd", x, c(params["k_w"]), precision=precision)
    v = jnp.einsum("bsc,chd->bshd", x, c(params["v_w"]), precision=precision)
    q = q.astype(dt)
    if core is None:
        o = reference_attention(q, k, v, nb, kmask, 1.0, layout="BSHD", precision=precision)
    else:
        o = core(q, k, v, nb, kmask, 1.0)
    g = jnp.einsum("bsc,chd->bshd", x, c(params["gate_w"]), precision=precision)
    if params.get("gate_b") is not None:
        g = g + c(params["gate_b"])[None, None]
    o = o * jax.nn.sigmoid(g)
    out = jnp.einsum("bshd,hdc->bsc", o.astype(dt), c(params["out_w"]), precision=precision)
    if params.get("out_b") is not None:
        out = out + c(params["out_b"])[None, None]
    if T:
        out = jnp.swapaxes(out, 0, 1)
    return out.astype(dt)


def triangle_attention_block(act, pair_mask, params: Dict[str, Any], *, orientation: str, form: str = "af2", unit: Optional[str] = None, word: str,
                             direction: str = "fwd", jax_line: Any = None, cc: Any = None, prefer: Optional[Sequence[str]] = None, strict: bool = False,
                             input_precision: Optional[str] = None, config: Optional[Dict[str, Any]] = None, selection: Optional[Selection] = None, **fpf_kw):
    """Triangle attention module.  ``params``: attn_params keywords (math layout).  ``form`` af2 | af3 | joltz and ``unit`` pair | tmpl
    (None: c 64 -> tmpl, else pair) name the family with (c, heads, head_dim, orientation).  ``fpf_kw`` ride to fpf_pallas_serve.tri_attn_block
    (ending_bias_transposed, key_mask, cfg, pad)."""
    N, C = int(act.shape[0]), int(act.shape[-1])
    Cw, H, D = (int(s) for s in params["q_w"].shape)
    unit = unit or ("tmpl" if C <= 64 else "pair")
    if selection is None:
        fam = family("triattn", form=form, unit=unit, c=C, heads=H, head_dim=D, orientation={"per_row": "starting", "per_column": "ending"}.get(orientation, orientation))
        selection = resolve("triattn", fam, act.dtype, N, word=word, direction=direction, jax_line=jax_line, cc=cc, prefer=prefer)
    sel = selection

    def one(arm, row, ip, setting):
        ip_ = ip or input_precision
        cfg = dict(sel.config if arm == sel.arm else {})
        cfg.update(config or {})
        if row == "xla":
            return reference_triangle_attention(act, pair_mask, params, orientation=orientation)
        if row == "fpf_block":
            from .. import fpf_pallas_serve as S  # noqa: WPS433
            f32 = dtype_word(act.dtype) == "f32"
            if f32 and ip_ is None:
                raise Refusal("fpf_block: f32 activations need the operand-precision word (tf32 | tf32x3 | ieee): word='fpf_block:tf32' or input_precision=",
                              kind=PRECISION_REQUIRED, row=row, fallback="xla")
            kp = S.attn_params(weights_dtype=(act.dtype if f32 else None), **{k_: params[k_] for k_ in ("ln_scale", "ln_offset", "q_w", "k_w", "v_w", "bias_w", "gate_w", "out_w")},
                               **({"gate_b": params["gate_b"], "out_b": params["out_b"]} if params.get("gate_b") is not None else {}))
            kw = dict(fpf_kw)
            if cfg:
                kw["cfg"] = dict(kw.get("cfg") or {}, **{k_: v_ for k_, v_ in cfg.items() if k_ in ("t1", "w1", "t2", "w2", "bq", "bk", "wa", "sa")})   # fpf_pallas_serve.attn_cfg keys
            kpa, kps = _split_static(kp)
            try:
                fn = lambda a, m_, p_: S.tri_attn_block(a, m_, dict(p_, **kps), orientation=orientation, precision=(ip_ if f32 else cfg.get("precision", "std")), **kw)  # noqa: E731
                return forward_only(fn, "fpf_block", "proj+pallas_attn")(act, pair_mask, kpa)
            except S.Refusal as e:
                raise Refusal("fpf_block refused by name: %s" % e, kind=getattr(e, "kind", ROW_ERROR), row=row, fallback="xla")
        if row.startswith("proj+"):
            core_row = row.split("+", 1)[1]

            def core(q, k, v, nb, kmask, scale):
                return _attention_row(core_row, ip_, cfg, q, k, v, nb, kmask, scale, layout="BSHD", direction=direction, pad_to=(PAD_HEAD_DIM_TO if D < PAD_HEAD_DIM_TO else None))
            return reference_triangle_attention(act, pair_mask, params, orientation=orientation, core=core)
        raise Refusal("row %r does not serve the triangle-attention module" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return _walk(sel, "triangle_attention_block", one, strict, probe=lambda arm: triangle_attention_block(*_concrete_zeros((act, pair_mask, params)), orientation=orientation, form=form,
                                                                                      unit=unit, word=arm, direction=direction, jax_line=jax_line, cc=cc, strict=True))


# ----------------------------------------------------------------------------------------------------------------- op: triangle multiplication module
TRIMUL_KEYS = ("ln_in_scale", "ln_in_offset", "left_w", "right_w", "left_gate_w", "right_gate_w", "ln_c_scale", "ln_c_offset", "out_w", "gate_w")
TRIMUL_BIAS_KEYS = ("left_b", "right_b", "left_gate_b", "right_gate_b", "out_b", "gate_b")
EQUATIONS = {"outgoing": "ikc,jkc->ijc", "incoming": "kjc,kic->ijc", "ikc,jkc->ijc": "ikc,jkc->ijc", "kjc,kic->ijc": "kjc,kic->ijc"}


def _fpf_trimul_params(params, act_dtype):
    from .. import fpf_pallas_serve as S  # noqa: WPS433
    kw = {k_: params[k_] for k_ in TRIMUL_KEYS}
    if params.get("left_b") is not None:
        kw.update({k_: params[k_] for k_ in TRIMUL_BIAS_KEYS})
    return S.trimul_params(**kw)


def cd_trimul_tree(params: Dict[str, Any], dt):
    """Math-layout trimul params -> the AF2-multimer fused-projection module tree cd_trimul reads (projection = [left | right] columns,
    biases zero when absent, everything in the activation dtype but the LayerNorms)."""
    import jax.numpy as jnp  # noqa: WPS433
    c = lambda w: w.astype(dt)  # noqa: E731
    cat = lambda a, b: jnp.concatenate([a, b], axis=-1)  # noqa: E731
    T = {"left_norm_input": {"scale": params["ln_in_scale"], "offset": params["ln_in_offset"]}, "projection": {"weights": c(cat(params["left_w"], params["right_w"]))},
         "gate": {"weights": c(cat(params["left_gate_w"], params["right_gate_w"]))}, "center_norm": {"scale": params["ln_c_scale"], "offset": params["ln_c_offset"]},
         "output_projection": {"weights": c(params["out_w"])}, "gating_linear": {"weights": c(params["gate_w"])}}
    Ch, C = int(params["left_w"].shape[1]), int(params["gate_w"].shape[0])
    if params.get("left_b") is not None:
        T["projection"]["bias"] = c(cat(params["left_b"], params["right_b"]))
        T["gate"]["bias"] = c(cat(params["left_gate_b"], params["right_gate_b"]))
        T["output_projection"]["bias"] = c(params["out_b"])
        T["gating_linear"]["bias"] = c(params["gate_b"])
    else:
        T["projection"]["bias"] = jnp.zeros((2 * Ch,), dt)
        T["gate"]["bias"] = jnp.zeros((2 * Ch,), dt)
        T["output_projection"]["bias"] = jnp.zeros((C,), dt)
        T["gating_linear"]["bias"] = jnp.zeros((C,), dt)
    return T


def reference_triangle_multiplication(act, mask, params: Dict[str, Any], *, equation: str, precision=None, compute_dtype="act"):
    """The stock TriangleMultiplication statement (row 'xla'), op by op at act.dtype (compute_dtype='act') or in f32 (None)."""
    from .. import fpf_pallas_serve as S  # noqa: WPS433
    p = _fpf_trimul_params(params, act.dtype)
    cd = act.dtype if compute_dtype == "act" else compute_dtype
    return S.trimul_reference(act, mask, p, equation=EQUATIONS[equation], compute_dtype=cd, precision=precision)


def triangle_multiplication(act, mask, params: Dict[str, Any], *, equation: str, form: str = "af2", unit: Optional[str] = None, word: str, direction: str = "fwd",
                            jax_line: Any = None, cc: Any = None, prefer: Optional[Sequence[str]] = None, strict: bool = False, input_precision: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None, selection: Optional[Selection] = None, **fpf_kw):
    """Triangle multiplication module.  ``params``: trimul_params keywords (math layout; the six biases optional).  ``equation`` outgoing |
    incoming (or the einsum strings)."""
    import jax  # noqa: WPS433
    N, C = int(act.shape[0]), int(act.shape[-1])
    Ch = int(params["left_w"].shape[1])
    unit = unit or ("tmpl" if C <= 64 else "pair")
    if selection is None:
        fam = family("trimul", form=form, unit=unit, c=C, c_hidden=Ch, equation=equation)
        selection = resolve("trimul", fam, act.dtype, N, word=word, direction=direction, jax_line=jax_line, cc=cc, prefer=prefer)
    sel = selection
    f32 = dtype_word(act.dtype) == "f32"

    def one(arm, row, ip, setting):
        ip_ = ip or input_precision
        cfg = dict(sel.config if arm == sel.arm else {})
        cfg.update(config or {})
        if row == "xla":
            return reference_triangle_multiplication(act, mask, params, equation=equation)
        if row in ("fpf_trimul", "fpf_trimul_xlabwd"):
            from .. import fpf_pallas_serve as S  # noqa: WPS433
            if f32 and ip_ is None:
                raise Refusal("%s: f32 activations need the operand-precision word (tf32 | tf32x3 | ieee)" % row, kind=PRECISION_REQUIRED, row=row, fallback="xla")
            p_all = _fpf_trimul_params(params, act.dtype)
            p, p_static = _split_static(p_all)
            prec = ip_ if f32 else cfg.get("precision", "std")
            kw = dict(fpf_kw)
            lc = {k_: v_ for k_, v_ in cfg.items() if k_ in ("t1", "w1", "s1", "t2", "w2", "s2", "ein")}          # fpf_pallas_serve.trimul_cfg keys
            if lc:
                kw["cfg"] = dict(kw.get("cfg") or {}, **lc)

            def fwd_fn(a, m_, p_):
                try:
                    return S.trimul_block(a, m_, dict(p_, **p_static), equation=EQUATIONS[equation], precision=prec, **kw)
                except S.Refusal as e:
                    raise Refusal("%s refused by name: %s" % (row, e), kind=getattr(e, "kind", ROW_ERROR), row=row, fallback="xla")
            if row == "fpf_trimul":
                return forward_only(fwd_fn, "fpf_trimul", "cd_trimul")(act, mask, p)

            @jax.custom_vjp
            def op(a, p_):
                return fwd_fn(a, mask, p_)

            def op_fwd(a, p_):
                return fwd_fn(a, mask, p_), (a, p_)

            def op_bwd(res, ct):
                a, p_ = res
                _, vjp = jax.vjp(lambda a2, p2: S.trimul_reference(a2, mask, dict(p2, **p_static), equation=EQUATIONS[equation], compute_dtype=None, precision=None).astype(a2.dtype), a, p_)
                return vjp(ct)
            op.defvjp(op_fwd, op_bwd)
            return op(act, p)
        if row == "native_xla":
            from .. import trimul_xla as TX  # noqa: WPS433
            eq = EQUATIONS[equation]
            direction_word = "outgoing" if eq == EQUATIONS["outgoing"] else ("incoming" if eq == EQUATIONS["incoming"] else None)
            if direction_word is None:
                raise Refusal("native_xla: equation %r is neither the outgoing nor the incoming update" % (equation,), kind=ROW_ERROR, row=row, fallback="cd_trimul")
            try:
                w = TX.weights_from_math_layout(params, direction_word)
                return TX.triangle_multiplication(act, mask, weights=w, direction=direction_word, cc=(str(cc) if cc is not None else None))
            except TX.Refused as e:
                raise Refusal("native_xla refused by name: %s" % e, kind=getattr(e, "kind", ROW_ERROR), row=row, fallback="cd_trimul")
        if row == "cd_trimul":
            from .cd_trimul import trimul_pallas as CM  # noqa: WPS433
            tree = cd_trimul_tree(params, act.dtype)
            lc = {k_: v_ for k_, v_ in cfg.items() if k_ in CM.DEFAULT_CFG}          # t1 w1 t3 w3 (forward tiles / warps), tb1 wb1 tb3 wb3 (backward)
            return CM.triangle_multiplication(act, mask, tree, equation=EQUATIONS[equation], cfg=(lc or None))
        raise Refusal("row %r does not serve the triangle-multiplication module" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return _walk(sel, "triangle_multiplication", one, strict, probe=lambda arm: triangle_multiplication(*_concrete_zeros((act, mask, params)), equation=equation, form=form, unit=unit,
                                                                                     word=arm, direction=direction, jax_line=jax_line, cc=cc, strict=True))


# ----------------------------------------------------------------------------------------------------------------- op: GLU prologue (AF3 form)
def glu_transposed_masked(x, weight, mask_t, *, activation=None, word: str = "glut", form: str = "af3", jax_line: Any = None, cc: Any = None, strict: bool = False,
                          config: Optional[Dict[str, Any]] = None, selection: Optional[Selection] = None):
    """gated_linear_unit(x [N,N,C], weight [C,2,F]) -> transpose to [F,N,N] -> * mask_t: rows glut (Pallas, bitwise the tokamax statement) |
    tokamax_glu_T (the stock statement through tokamax) | xla (jnp)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    N, C = int(x.shape[0]), int(x.shape[-1])
    F = int(weight.shape[-1])
    act_fn = activation if activation is not None else jax.nn.swish
    if selection is None:
        selection = resolve("glut", family("glut", form=form, c=C, out=F), x.dtype, N, word=word, jax_line=jax_line, cc=cc)
    sel = selection

    def one(arm, row, ip, setting):
        if row == "glut":
            from .. import fpf_pallas_serve as S  # noqa: WPS433
            try:
                return forward_only(lambda a, w_, m_: S.glu_transposed_masked(a, w_, m_, activation=act_fn, config=config), "glut", "tokamax_glu_T")(x, weight, mask_t)
            except S.Refusal as e:
                raise Refusal("glut refused by name: %s" % e, kind=getattr(e, "kind", ROW_ERROR), row=row, fallback="tokamax_glu_T")
        w3 = weight if weight.ndim == 3 else weight.reshape(C, 2, F)
        if row == "tokamax_glu_T":
            tk = _tokamax()
            y = tk.gated_linear_unit(x, w3, activation=act_fn, implementation=None)
        elif row == "xla":
            ab = jnp.einsum("ijc,ckf->ijkf", x, w3.astype(x.dtype))
            y = act_fn(ab[..., 0, :]) * ab[..., 1, :]
        else:
            raise Refusal("row %r does not serve the GLU prologue" % row, kind=ROW_ERROR, row=row, fallback="xla")
        return jnp.transpose(y, (2, 0, 1)) * mask_t
    return _walk(sel, "glu_transposed_masked", one, strict, probe=lambda arm: glu_transposed_masked(*_concrete_zeros((x, weight, mask_t)), activation=activation, word=arm, form=form,
                                                                                   jax_line=jax_line, cc=cc, strict=True))



# ----------------------------------------------------------------------------------------------------------------- cc 8.x tile table of cd_transition
SM80_TRANSITION_TILE = (32, 4)          # (rows per program, num_warps) for C >= 128 on cc 8.x: the module's (64, 4|8) requests 49,408 B of shared memory
                                        # there and fails to launch; (32, 4) launches: x1.02 / 1.07 forward, x0.88 / 0.86 forward+backward of the
                                        # stock statement at 400 / 800 tokens (c 128, measured); the tiles are bitwise-identical in output (measured)


def _cd_transition_sm80_tiles(LT) -> None:
    """Install the cc 8.x entry of cd_transition's tile table once (the carried module has none): C >= 128 on cc 8.x -> SM80_TRANSITION_TILE;
    everything else -> the module's own table.  Idempotent; a no-op on other cards."""
    if getattr(LT, "_tiles_sm80_installed", False):
        return
    try:
        major = running_cc().split(".")[0]
    except Refusal:
        return
    if major != "8":
        LT._tiles_sm80_installed = True
        return
    own = LT._tiles

    def _tiles(C, H, _own=own):
        bm, bh, nw = _own(C, H)
        if C >= 128:
            return SM80_TRANSITION_TILE[0], bh, SM80_TRANSITION_TILE[1]
        return bm, bh, nw
    LT._tiles = _tiles
    LT._tiles_sm80_installed = True

# ----------------------------------------------------------------------------------------------------------------- op: transition
def reference_transition(x, params: Dict[str, Any], *, activation: str, precision=None):
    """The stock Transition statement at x.dtype (row 'xla'): LayerNorm -> W1 (+b1) -> relu | swiglu -> W2 (+b2)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    dt = x.dtype
    c = lambda w: w.astype(dt)  # noqa: E731
    xn = _ln(x, params.get("ln_scale"), params.get("ln_offset")).astype(dt)
    if activation == "relu":
        h = jnp.einsum("...c,cf->...f", xn, c(params["w1"]), precision=precision)
        if params.get("b1") is not None:
            h = h + c(params["b1"])
        h = jax.nn.relu(h)
    else:
        F = int(params["w2"].shape[0])
        w1 = params["w1"] if params["w1"].ndim == 2 else params["w1"].reshape(params["w1"].shape[0], -1)
        ab = jnp.einsum("...c,cf->...f", xn, c(w1), precision=precision)
        h = jax.nn.swish(ab[..., :F]) * ab[..., F:]
    out = jnp.einsum("...f,fc->...c", h.astype(dt), c(params["w2"]), precision=precision)
    if params.get("b2") is not None:
        out = out + c(params["b2"])
    return out.astype(dt)


def transition(x, params: Dict[str, Any], *, activation: str, form: str = "af2", word: str, direction: str = "fwd", n_tokens: Optional[int] = None,
               jax_line: Any = None, cc: Any = None, prefer: Optional[Sequence[str]] = None, strict: bool = False, input_precision: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None, selection: Optional[Selection] = None):
    """Transition (pair / MSA / single).  ``activation`` relu | swiglu; factor = F / C names the family with (form, c); ``n_tokens`` (default:
    the second-to-last axis) picks the size bucket."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    C = int(x.shape[-1])
    F = int(params["w2"].shape[0])
    n = int(n_tokens if n_tokens is not None else (x.shape[-2] if x.ndim >= 2 else x.shape[0]))
    if selection is None:
        selection = resolve("transition", family("transition", form=form, activation=activation, c=C, factor=max(1, F // C)), x.dtype, n, word=word, direction=direction,
                            jax_line=jax_line, cc=cc, prefer=prefer)
    sel = selection
    f32 = dtype_word(x.dtype) == "f32"

    def one(arm, row, ip, setting):
        ip_ = ip or input_precision
        cfg = dict(sel.config if arm == sel.arm else {})
        cfg.update(config or {})
        if row == "xla":
            return reference_transition(x, params, activation=activation)
        if row == "fpf_transition":
            from ..fpf_pallas import transition_pallas as FT  # noqa: WPS433
            try:
                fn = lambda a, s_, o_, w1, w2, b2: FT.fused_transition(a, s_, o_, w1, w2, b2)  # noqa: E731
                return forward_only(fn, "fpf_transition", "xla")(x, params.get("ln_scale"), params.get("ln_offset"), params["w1"], params["w2"], params.get("b2"))
            except ValueError as e:
                raise Refusal("fpf_transition: %s" % e, kind="not_supported", row=row, fallback="xla")
        if row == "mlp_transition":
            from .mlp_transition import mlp_transition_pallas as MT  # noqa: WPS433
            kw = dict(activation=activation)
            if f32:
                kw["f32_precision"] = ip_ or "tf32"
            try:
                if cfg.get("bwd") == "reference":
                    op = MT.make_transition_op(bwd="reference", **kw)
                    return op(x, params.get("ln_scale"), params.get("ln_offset"), params["w1"], params["w2"], params.get("b1"), params.get("b2"))
                fn = lambda a, s_, o_, w1, w2, b1, b2: MT.fused_transition(a, s_, o_, w1, w2, b1, b2, **kw)  # noqa: E731
                return forward_only(fn, "mlp_transition", "xla")(x, params.get("ln_scale"), params.get("ln_offset"), params["w1"], params["w2"], params.get("b1"), params.get("b2"))
            except MT.NotServed as e:
                raise Refusal("mlp_transition refused by name: %s" % e, kind=getattr(e, "kind", "not_supported"), row=row, fallback="xla")
        if row == "cd_transition":
            from .cd_layers import layers_transition as LT  # noqa: WPS433
            if x.dtype != jnp.bfloat16:                                        # the module's contract: bf16 activations, bf16 operands (its products are
                raise Refusal("cd_transition: bf16 activations only (the module's operand contract); %s activations: mlp_transition (f32: its "
                              "precision words) or the stock statement" % dtype_word(x.dtype), kind="not_supported", row=row, fallback="xla")
            _cd_transition_sm80_tiles(LT)
            wd = x.dtype                                                       # bf16 x bf16, f32 accumulation): weights / biases arrive in the activation
            b1 = params.get("b1").astype(wd) if params.get("b1") is not None else jnp.zeros((F,), wd)     # dtype as the AlphaFold-2 bf16 getter hands
            b2 = params.get("b2").astype(wd) if params.get("b2") is not None else jnp.zeros((C,), wd)     # them; f32 masters are cast here (a mixed
            sc = params.get("ln_scale") if params.get("ln_scale") is not None else jnp.ones((C,), jnp.float32)   # bf16 x f32 tile product does not
            of = params.get("ln_offset") if params.get("ln_offset") is not None else jnp.zeros((C,), jnp.float32)  # lower); LayerNorm stays f32
            return LT.transition(x, sc, of, params["w1"].astype(wd), b1, params["w2"].astype(wd), b2)
        if row == "tokamax_glu":
            tk = _tokamax()
            xn = _ln(x, params.get("ln_scale"), params.get("ln_offset")).astype(x.dtype)
            w1 = params["w1"]
            w3 = w1 if w1.ndim == 3 else w1.reshape(C, 2, F)
            h = tk.gated_linear_unit(xn, w3.astype(x.dtype), activation=jax.nn.swish, implementation=cfg.get("implementation"))
            out = jnp.einsum("...f,fc->...c", h.astype(x.dtype), params["w2"].astype(x.dtype))
            if params.get("b2") is not None:
                out = out + params["b2"].astype(x.dtype)
            return out.astype(x.dtype)
        raise Refusal("row %r does not serve the transition" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return _walk(sel, "transition", one, strict, probe=lambda arm: transition(*_concrete_zeros((x, params)), activation=activation, form=form, word=arm, direction=direction,
                                                                        n_tokens=n_tokens, jax_line=jax_line, cc=cc, strict=True))


# ----------------------------------------------------------------------------------------------------------------- op: LayerNorm
def reference_layer_norm(x, scale, offset, *, eps: float = LN_EPS):
    return _ln(x, scale, offset, eps).astype(x.dtype)


def layer_norm(x, scale, offset, *, unit: str = "pair", word: str, direction: str = "fwd", n_tokens: Optional[int] = None, n_seq: Optional[int] = None,
               t: Optional[int] = None, jax_line: Any = None, cc: Any = None, prefer: Optional[Sequence[str]] = None, strict: bool = False,
               selection: Optional[Selection] = None):
    """LayerNorm over the last axis (eps 1e-5, f32 statistics), rows cd_ln | xla.  ``unit`` pair | tmpl | single | atom | dit | msa (+ n_seq / t)
    names the family with c."""
    C = int(x.shape[-1])
    n = int(n_tokens if n_tokens is not None else (x.shape[-2] if x.ndim >= 2 else x.shape[0]))
    if selection is None:
        selection = resolve("ln", family("ln", unit=unit, c=C, n_seq=n_seq, t=t), x.dtype, n, word=word, direction=direction, jax_line=jax_line, cc=cc, prefer=prefer)
    sel = selection

    def one(arm, row, ip, setting):
        if row == "xla":
            return reference_layer_norm(x, scale, offset)
        if row == "cd_ln":
            from .cd_layers import layers_ln as L  # noqa: WPS433
            if getattr(L, "jax", None) is None:
                raise Refusal("cd_ln: its module imports did not resolve on this stack", kind=NEEDS_HAIKU, row=row, fallback="xla")
            return L.layer_norm(x, scale, offset)
        raise Refusal("row %r does not serve LayerNorm" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return _walk(sel, "layer_norm", one, strict, probe=lambda arm: layer_norm(*_concrete_zeros((x, scale, offset)), unit=unit, word=arm, direction=direction, n_tokens=n_tokens,
                                                                        n_seq=n_seq, jax_line=jax_line, cc=cc, strict=True))


# ----------------------------------------------------------------------------------------------------------------- op: outer-product mean
def _opm_project(act, mask, params, dt, precision=None):
    import jax.numpy as jnp  # noqa: WPS433
    c = lambda w: w.astype(dt)  # noqa: E731
    a = act
    if params.get("ln_scale") is not None:
        a = _ln(act, params["ln_scale"], params["ln_offset"]).astype(dt)
    m = mask.astype(dt)[..., None]
    left = m * (jnp.einsum("sna,ac->snc", a.astype(dt), c(params["left_w"]), precision=precision) + c(params["left_b"]))
    right = m * (jnp.einsum("sna,ac->snc", a.astype(dt), c(params["right_w"]), precision=precision) + c(params["right_b"]))
    return left, right


def reference_outer_product_mean(act, mask, params: Dict[str, Any], *, chunk: int = 128, precision=None):
    """The stock OuterProductMean statement at act.dtype (row 'xla'): (LayerNorm) -> left/right projections * mask -> outer product over the
    sequence axis in chunks of ``chunk`` rows -> output projection + bias -> / (1e-3 + mask norm)."""
    import jax.numpy as jnp  # noqa: WPS433
    dt = act.dtype
    left, right = _opm_project(act, mask, params, dt, precision)
    S_, N, cc_ = (int(s) for s in left.shape)
    Fo = int(params["out_w"].shape[-1])
    ow = params["out_w"].astype(dt).reshape(cc_ * cc_, Fo)
    outs = []
    for i in range(0, N, chunk):
        lc = left[:, i:i + chunk]
        o4 = jnp.einsum("sic,sje->ijce", lc, right, precision=precision)
        outs.append(jnp.einsum("ijk,kf->ijf", o4.reshape(o4.shape[0], N, cc_ * cc_), ow, precision=precision) + params["out_b"].astype(dt))
    out = jnp.concatenate(outs, 0)
    m = mask.astype(dt)
    norm = jnp.einsum("si,sj->ij", m, m, precision=precision)[..., None]
    return (out / (1e-3 + norm)).astype(dt)


def outer_product_mean(act, mask, params: Dict[str, Any], *, form: str = "af2", word: str, direction: str = "fwd", jax_line: Any = None, cc: Any = None,
                       prefer: Optional[Sequence[str]] = None, strict: bool = False, input_precision: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                       selection: Optional[Selection] = None):
    """Outer-product mean: act [S,N,C_m] (MSA), mask [S,N]; params ln_scale / ln_offset (optional), left_w / left_b, right_w / right_b [C_m,c],[c],
    out_w [c,c,F], out_b [F].  Family: form, c_m, c, f, n_seq (= S, nearest measured)."""
    import jax.numpy as jnp  # noqa: WPS433
    S_, N, Cm = (int(s) for s in act.shape)
    c_ = int(params["left_w"].shape[-1])
    Fo = int(params["out_w"].shape[-1])
    if selection is None:
        depths = []
        for f in families("opm"):
            mm = f.split("_S")
            if f.startswith("%s_opm_cm%d_c%d_f%d_S" % (form, Cm, c_, Fo)):
                depths.append(int(mm[-1]))
        pick = (sorted(d for d in depths if d >= S_) or [max(depths)])[0] if depths else S_
        selection = resolve("opm", family("opm", form=form, c_m=Cm, c=c_, f=Fo, n_seq=pick), act.dtype, N, word=word, direction=direction, jax_line=jax_line, cc=cc, prefer=prefer)
    sel = selection
    f32 = dtype_word(act.dtype) == "f32"

    def one(arm, row, ip, setting):
        ip_ = ip or input_precision
        cfg = dict(sel.config if arm == sel.arm else {})
        cfg.update(config or {})
        dt = act.dtype
        if row == "xla":
            return reference_outer_product_mean(act, mask, params)
        if row in ("opm_two_launch", "opm_reassoc", "opm_stockbody"):
            from .opm_pallas import opm_pallas as OP  # noqa: WPS433
            left, right = _opm_project(act, mask, params, dt)
            w, b = params["out_w"].astype(dt), params["out_b"].astype(dt)
            if row == "opm_stockbody":
                return OP.opm_stock(left, right, mask.astype(dt), w, b)
            m = mask.astype(jnp.float32)
            norm = jnp.einsum("si,sj->ij", m, m)
            if row == "opm_reassoc":
                return OP.opm_reassoc(left, right, norm, w, b, precision=(ip_ if f32 else None) and None)
            kw = {}
            if f32 and ip_:
                kw["precision"] = ip_
            fn = lambda l_, r_, n_, w_, b_: OP.opm_two_launch(l_, r_, n_, w_, b_, **kw)  # noqa: E731
            return forward_only(fn, "opm_two_launch", "opm_reassoc")(left, right, norm, w, b)
        if row == "cd_opm":
            from .cd_layers import layers_opm as LO  # noqa: WPS433
            if getattr(LO, "jax", None) is None:
                raise Refusal("cd_opm: needs haiku importable (its module imports haiku with jax)", kind=NEEDS_HAIKU, row=row, fallback="xla")
            a = _ln(act, params["ln_scale"], params["ln_offset"]).astype(dt) if params.get("ln_scale") is not None else act
            lw, lb, rw, rb = params["left_w"].astype(dt), params["left_b"].astype(dt), params["right_w"].astype(dt), params["right_b"].astype(dt)
            left_lin = lambda x: jnp.einsum("sna,ac->snc", x, lw) + lb    # noqa: E731  (the stock Linear statement at the activations' dtype)
            right_lin = lambda x: jnp.einsum("sna,ac->snc", x, rw) + rb   # noqa: E731
            return LO.outer_product_mean(a, mask, left_lin, right_lin, params["out_w"].astype(dt), params["out_b"].astype(dt))
        raise Refusal("row %r does not serve the outer-product mean" % row, kind=ROW_ERROR, row=row, fallback="xla")
    return _walk(sel, "outer_product_mean", one, strict, probe=lambda arm: outer_product_mean(*_concrete_zeros((act, mask, params)), form=form, word=arm, direction=direction,
                                                                                jax_line=jax_line, cc=cc, strict=True))


def report() -> Dict[str, Any]:
    """What was served and refused in this process."""
    return {"counts": dict(COUNTS), "last_refusal": dict(LAST_REFUSAL)}
