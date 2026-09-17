"""F6 `trimul_cmajor` — joltz's triangle multiplication kept in ONE channel-major layout, so XLA materialises no [N,N,C]↔[C,N,N] transpose.

What it does. `install()` rebinds `joltz.TriangleMultiplicationOutgoing.__call__` and `joltz.TriangleMultiplicationIncoming.__call__` (the two
classes behind the trunk pairformer's, the MSA module's and the template module's triangle multiplications). Stock projects the normalised pair
activation to `p_in(x)·sigmoid(g_in(x))` [.., N, N, 2C] (channel-minor), masks it, splits it into a | b and contracts `einsum('bikd,bjkd->bijd')`
(outgoing; `'bkid,bkjd->bijd'` incoming): a matmul batched over the CHANNEL d, which cuBLAS takes only with d as the leading (major) dimension —
so XLA writes physical transposes of both operands to [.., 2C, N, N], of the LayerNorm output that feeds them, and of the [.., C, N, N] product
back to channel-minor, forward, in the rematerialised forward, and for the cotangents. With
the lever configured on, the same arithmetic runs in one layout: the two input projections are contracted so the GEMM WRITES channel-major
(`p[e, b, i, k] = Σ_c W_p[e, c]·x[b, i, k, c]` — a plain `lax.dot_general` whose output order is the layout, no copy), the sigmoid gate and the
pair mask apply in that layout, a | b are the two contiguous channel halves, the triangle product is the batched matmul it already is with its
batch dims (channel, batch) leading (`out[d, b, i, j] = Σ_k a[d, b, i, k]·b[d, b, j, k]`; incoming contracts the first pair index), `norm_out`
normalises over the LEADING channel axis (`_layer_norm(t, …, axis=0)`: joltz's LayerNorm arithmetic — mean, variance of the centred values,
rsqrt(var + ε), weight·x̂ + bias — written along that axis), and the output projection contracts the channel
axis, which lands the result channel-minor (`out[b, i, j, c'] = Σ_d t[d, b, i, j]·W_out[c', d]`) for the output gate `sigmoid(g_out(x_in))` and the
residual — no transpose anywhere, any N, any batch, any channel count (nothing here names a size). Numerics class `fast`: the dots are the stock einsums'
contractions with other dimension numbers and the channel LayerNorm reduces along a major axis, so XLA's kernel and algorithm picks differ —
not bitwise with stock; no dtype or precision changes (every dot at the ambient
`jax.default_matmul_precision`, as the einsums it replaces). Every traced call is counted (`served`, by direction and shape); the LEVER line
(`emit_line`) is the census and `gate()` refuses a run under the lever's name in which no triangle multiplication was served.

    from mosaic.fast import trimul_layout
    trimul_layout.install(); rec = trimul_layout.configure(None)    # = "cmajor", the default setting, BEFORE the loss is traced; "stock" = every call on
    ...                                                              # joltz's own bodies (off)
    print(trimul_layout.emit_line(tag)); trimul_layout.gate()        # after the run: the census line, the fail-closed gate

`configure` clears jax's and equinox's trace caches (the previously traced bodies would otherwise stay) and returns `describe()`. ENV_REQUIRED is
empty: nothing of this lever lives in the environment.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

LEVER = "F6"
NAME = "trimul_cmajor"
MODULE = "mosaic.fast.trimul_layout"
SPECS = ("stock", "cmajor")                            # configure() words
SETTING = "cmajor"                                     # the default setting: configure(None) applies it
ENV_REQUIRED: Dict[str, str] = {}                       # nothing must be in the environment before jax import
KLASS = "fast"                                         # not bitwise with stock (other dimension numbers → other XLA kernel and algorithm picks)
EXPECTED_FALLBACKS = ()                                # the lever declares no fallback: every triangle multiplication is served when on
TARGETS = (("TriangleMultiplicationOutgoing", "outgoing"), ("TriangleMultiplicationIncoming", "incoming"))   # joltz class → direction word

STATE: Dict[str, Any] = {"spec": None, "on": False, "ledger": None, "body": None, "body_word": "xla"}   # body: the served arithmetic (None = trimul_cmajor, the XLA path); set_body() installs another with the same contract
_ORIG: Dict[str, Any] = {}


class Refusal(RuntimeError):
    """A named refusal (`.reason` is the word)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# --------------------------------------------------------------------------------------------------------------------------- the grammar
def parse(spec: Optional[str]) -> Dict[str, Any]:
    """`None` / `''` / `'cmajor'` → ON (the default setting); `'stock'` / `'off'` → every call on joltz's own bodies. Anything else is
    `Refusal('unknown_spec')` naming the words."""
    word = (spec or "").strip().lower() or SETTING
    if word in ("stock", "off"):
        return {"on": False, "spec": "stock"}
    if word == "cmajor":
        return {"on": True, "spec": "cmajor"}
    raise Refusal("unknown_spec", f"{spec!r}: one of {', '.join(SPECS)} (None = the default setting {SETTING!r})")


# --------------------------------------------------------------------------------------------------------------------------- the arithmetic
def _bcast(v, axis: int, ndim: int):
    """A per-channel vector [C] shaped to broadcast against x along `axis`."""
    shape = [1] * ndim
    shape[axis] = v.shape[0]
    return v.reshape(shape)


def _layer_norm(x, weight, bias, eps, axis: int):
    """joltz's LayerNorm arithmetic (backend.LayerNorm = eqx.nn.LayerNorm: mean, variance of the centred values clamped at 0, rsqrt(var + eps),
    `weight * x̂`, `+ bias`, statistics in x's dtype promoted to at least float32, the result cast back) written along `axis` — norm_out over the
    LEADING channel axis of the channel-major layout (joltz's own __call__ normalises the last axis only)."""
    import jax
    import jax.numpy as jnp
    orig = x.dtype
    dt = jnp.promote_types(x.dtype, jnp.float32)
    x = x.astype(dt)
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.maximum(0.0, jnp.var(x, axis=axis, keepdims=True))
    out = (x - mean) * jax.lax.rsqrt(var + eps)
    if weight is not None:
        out = _bcast(weight.astype(dt), axis, x.ndim) * out
    if bias is not None:
        out = out + _bcast(bias.astype(dt), axis, x.ndim)
    return out.astype(orig)


def _project_cmajor(linear, xs):
    """joltz Linear (weight [E, C], bias [E] | None) applied to xs [B, i, k, C] with the result WRITTEN channel-major: [E, B, i, k]."""
    import jax
    y = jax.lax.dot_general(linear.weight, xs, (((1,), (3,)), ((), ())))          # [E, C]·[B, i, k, C] over C → [E, B, i, k]
    if linear.bias is not None:
        y = y + linear.bias[:, None, None, None]
    return y


def _project_from_cmajor(linear, t):
    """joltz Linear (weight [C', D], bias [C'] | None) applied over the LEADING axis of t [D, B, i, j]: the result lands channel-minor [B, i, j, C']."""
    import jax
    y = jax.lax.dot_general(t, linear.weight, (((0,), (1,)), ((), ())))            # [D, B, i, j]·[C', D] over D → [B, i, j, C']
    if linear.bias is not None:
        y = y + linear.bias
    return y


def trimul_cmajor(module, x, mask, direction: str):
    """The triangle multiplication of `module` (a joltz TriangleMultiplication{Outgoing,Incoming}: norm_in, p_in, g_in, norm_out, p_out, g_out)
    on x [..., N, N, C] with pair mask [..., N, N], in the channel-major layout (module docstring). `direction` ∈ {'outgoing', 'incoming'}."""
    import jax
    import jax.numpy as jnp
    if direction not in ("outgoing", "incoming"):
        raise Refusal("unknown_direction", repr(direction))
    x = module.norm_in(x)                                                           # LayerNorm over C, channel-minor: joltz's own call
    x_in = x
    lead = x.shape[:-3]
    n1, n2, c = x.shape[-3:]
    xs = x.reshape((-1,) + (n1, n2, c))                                             # [B, i, k, C]: the leading dims flattened into one batch
    ms = jnp.reshape(mask, (-1, n1, n2)).astype(xs.dtype)                           # stock multiplies by the mask as given (bool or float): same values
    v = _project_cmajor(module.p_in, xs) * jax.nn.sigmoid(_project_cmajor(module.g_in, xs))   # [2D, B, i, k]
    v = v * ms[None]
    a, b = jnp.split(v, 2, axis=0)                                                  # the two contiguous channel halves [D, B, i, k]
    if direction == "outgoing":                                                     # out[d, b, i, j] = Σ_k a[d, b, i, k]·b[d, b, j, k]   (stock 'bikd,bjkd->bijd')
        t = jax.lax.dot_general(a, b, (((3,), (3,)), ((0, 1), (0, 1))))
    else:                                                                           # out[d, b, i, j] = Σ_k a[d, b, k, i]·b[d, b, k, j]   (stock 'bkid,bkjd->bijd')
        t = jax.lax.dot_general(a, b, (((2,), (2,)), ((0, 1), (0, 1))))
    t = _layer_norm(t, module.norm_out.weight, module.norm_out.bias, module.norm_out.eps, 0)   # norm_out over the LEADING channel axis (joltz's arithmetic along axis 0)
    out = _project_from_cmajor(module.p_out, t)                                     # [B, i, j, C']: channel-minor again, by the contraction itself
    out = out * jax.nn.sigmoid(module.g_out(x_in)).reshape(out.shape)
    return out.reshape(lead + out.shape[1:])


# --------------------------------------------------------------------------------------------------------------------------- the lever API
def _clear_trace_caches():
    """joltz's filter_jit'd trunk and mosaic's jitted loss step keep the previously traced body: clear jax's and equinox's caches, or refuse."""
    import jax
    import equinox as eqx
    try:
        jax.clear_caches(); eqx.clear_caches()
    except Exception as e:                                   # a configuration change that cannot evict the traced bodies is refused, never ignored
        raise Refusal("cache_clear_failed", f"{type(e).__name__}: {e}") from None


def _ledger():
    from opt_core.counters import Ledger
    return Ledger(f"{LEVER}.{NAME}", impl=MODULE, origin="kit", expected=tuple(EXPECTED_FALLBACKS))


def _is_joltz_own(fn, cls_name: str) -> bool:
    """True when `fn` is joltz's own `<cls_name>.__call__` (not another lever's wrapper or served call)."""
    return getattr(fn, "__module__", None) == "joltz" and getattr(fn, "__qualname__", "") == f"{cls_name}.__call__"


def install():
    """Rebind the two triangle-multiplication `__call__`s (idempotent). Until `configure` turns the lever on, the rebound calls ARE the stock bodies.
    Refuses `install_order` — touching nothing — when either `__call__` is already another lever's wrapper (the memory lever P5's `sub`
    rematerialisation wraps the call it finds at ITS install): the served body does not delegate to the call it replaced, so a wrapper installed
    first would be bypassed silently at these two sites. Install this lever BEFORE any lever that wraps the triangle multiplications; a wrapper
    installed after it wraps the served call and composes."""
    import joltz
    if _ORIG:
        return
    found = {cls_name: getattr(joltz, cls_name).__call__ for cls_name, _ in TARGETS}
    foreign = [f"joltz.{c}.__call__ is {getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', getattr(f, '__name__', '?'))}" for c, f in found.items() if not _is_joltz_own(f, c)]
    if foreign:
        raise Refusal("install_order", f"{LEVER} must be installed before any lever that wraps the triangle multiplications (found: {'; '.join(foreign)}) — "
                                       f"its served body would bypass that wrapper; install {LEVER} first so the wrapper wraps the served call")
    for cls_name, direction in TARGETS:
        cls = getattr(joltz, cls_name)
        _ORIG[direction] = found[cls_name]
        cls.__call__ = _CALLS[direction]


def uninstall():
    import joltz
    for cls_name, direction in TARGETS:
        if direction in _ORIG:
            getattr(joltz, cls_name).__call__ = _ORIG.pop(direction)
    STATE.update(spec=None, on=False, ledger=None, body=None, body_word="xla")
    _clear_trace_caches()


def installed() -> bool:
    return all(direction in _ORIG for _, direction in TARGETS)


def set_body(body=None, word: str = "xla") -> None:
    """The ONE extension point of this lever: the served arithmetic. `body(module, x, mask, direction) -> out` with EXACTLY `trimul_cmajor`'s
    contract (same math to rounding, same shapes and dtypes, leading dims preserved, mask multiplied as given); `None` restores `trimul_cmajor`
    (word `xla`). A kernel lever (F8 `trimul_fused`) sets its op here instead of rebinding joltz a second time, so the class rebinding, the
    install order and P5's wrapping stay this module's; `word` is the single token describe()/the census line print as `body=`. Refuses
    `not_installed`; clears the trace caches so the next trace serves the new body."""
    if not installed():
        raise Refusal("not_installed", "set_body() before install(): there is no served call to give a body to")
    if body is not None and not callable(body):
        raise Refusal("body_not_callable", repr(body))
    STATE["body"] = body
    STATE["body_word"] = ("_".join(str(word).split()) or "unnamed") if body is not None else "xla"
    _clear_trace_caches()


def configure(spec: Optional[str]) -> Dict[str, Any]:
    """Set the lever's layout for every later trace (`None` = the default setting, ON; 'stock' = off): refuses by name BEFORE anything is traced
    (an unknown word; `not_installed` when the layout could not take effect; install() itself refuses `install_order`; `set_body()` — the
    served-arithmetic extension point a kernel lever uses — refuses `not_installed` / `body_not_callable`); clears the trace caches; starts a fresh census; returns `describe()`."""
    cfg = parse(spec)
    if cfg["on"] and not installed():
        raise Refusal("not_installed", f"configure({spec!r}) before install(): the triangle-multiplication calls are not rebound, the layout could not take effect")
    _clear_trace_caches()
    STATE.update(spec=cfg["spec"], on=cfg["on"], ledger=_ledger() if installed() else None)
    return describe()


def describe() -> Dict[str, Any]:
    """The effective configuration in force, FLAT (single-token scalar values): lever, module, klass, on, spec, layout, installed, norm_out's axis,
    and — once configured — the census so far (served, fallback, fallback_by, shapes, calls)."""
    out: Dict[str, Any] = {"lever": LEVER, "lever_name": NAME, "module": MODULE, "klass": KLASS, "on": int(bool(STATE["on"])),
                           "spec": STATE["spec"] or "stock", "layout": "cmajor" if STATE["on"] else "stock", "installed": int(installed()),
                           "impl": MODULE, "origin": "kit", "norm_out": "channel_axis0" if STATE["on"] else "joltz", "body": STATE["body_word"]}
    L = STATE["ledger"]
    if L is not None:
        f = L.fields()
        for k in ("served", "fallback", "fallback_by", "shapes", "calls"):
            out[k] = _token(f.get(k))
        out["expected_fallbacks"] = ",".join(L.expected) or "none"
    return {k: _token(v) for k, v in out.items()}


def _token(v):
    """One describe() value as a scalar single token: mappings → `key:value+key:value` ('none' when empty), sequences → `a+b`, bool → int,
    strings with blanks → joined by '_', numbers and None as they are."""
    if isinstance(v, bool):
        return int(v)
    if v is None or isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return "+".join(f"{_token(k)}:{_token(v[k])}" for k in sorted(v, key=str)) or "none"
    if isinstance(v, (list, tuple, set, frozenset)):
        return "+".join(str(_token(x)) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)) or "none"
    return "_".join(str(v).split()) or "none"


def census() -> Dict[str, Any]:
    L = STATE["ledger"]
    return L.fields() if L is not None else {}


def emit_line(tag: str = "") -> str:
    """The LEVER line of this process's census (the core Ledger's grammar); the off line when the lever is not configured on."""
    from opt_core.report import emit, lever_line
    L = STATE["ledger"]
    if L is None or not STATE["on"]:
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason="not_configured" if L is None else "mode_stock", impl=MODULE, origin="kit"))
    return emit(L.line(tag, layout="cmajor", norm_out="channel_axis0", body=STATE["body_word"]))


def gate():
    """Fail closed: raises when the lever is installed but not configured ON (a run under the lever's name on the stock bodies), when the census
    holds an error or an undeclared fallback, or when no triangle multiplication was served."""
    L = STATE["ledger"]
    if L is None or not STATE["on"]:
        if installed():
            raise Refusal("not_configured", f"{LEVER} is installed but configure() never turned it on: the run would carry the lever's name with every triangle multiplication on the stock body")
        return
    g = L.gate(f"{LEVER}.{NAME}")
    if not g.ok:
        raise Refusal("lever_gate", str(g.reason))
    if L.served == 0:
        raise Refusal("nothing_served", f"{LEVER} configured on and no triangle multiplication was traced through it")


# --------------------------------------------------------------------------------------------------------------------------- the calls
def _served_call(direction: str):
    def call(self, x, mask):
        """`joltz.TriangleMultiplication{Outgoing,Incoming}.__call__` with the lever: the stock body when off, `trimul_cmajor` when on."""
        if not STATE["on"]:
            return _ORIG[direction](self, x, mask)
        L = STATE["ledger"]
        if L is not None:
            lead = x.shape[:-3]
            batch = 1
            for d in lead:
                batch *= int(d)
            L.serve(f"{direction}:B{batch}xN{x.shape[-2]}xC{x.shape[-1]}")
        body = STATE["body"]
        return trimul_cmajor(self, x, mask, direction) if body is None else body(self, x, mask, direction)
    call.__name__ = f"trimul_{direction}_cmajor"
    call.__trimul_direction__ = direction
    return call


_CALLS = {direction: _served_call(direction) for _, direction in TARGETS}
