"""F8's op · `trimul_provider` — joltz's triangle multiplication through the shared core's JAX-family kernel provider
(`opt_core.kernels.pallas.serve.triangle_multiplication`) BY TIER WORD, with exactly F6's call contract:

    out = trimul(module, x, mask, direction)          # == trimul_layout.trimul_cmajor(module, x, mask, direction): same shapes/dtypes, leading dims kept

Nothing here names a kernel, a row, a tile or a size threshold. For every traced call the provider is asked twice — once per CALL KIND —
with the mode's tier word (`fast` in mode fast, `big` in mode big; `configure(word=...)`):

    resolve("trimul", family(form, unit=pair, c, c_hidden=c, equation), x.dtype, N, word=<tier>, direction="fwd")     -> the un-differentiated call's row
    resolve(...,                                                                          direction="fwdbwd")  -> the differentiated call's row

and the answer is bound behind ONE `jax.custom_vjp` per (word, dtype, N, C, equation, the two rows): the un-differentiated call (the refold,
inference) runs the `fwd` selection; a differentiated call (the design step: the trunk under the per-block remat) runs the `fwdbwd`
selection's forward inside `jax.vjp` and its pullback as the backward rule — the row's own fused backward when it has one. A selection whose
row is `xla` (the provider's cell table selects XLA for this call on this GPU, or no cell covers the call) is F6's OWN channel-major body
(`trimul_layout.trimul_cmajor`) BY NAME — counted `xla(cell)`, never silent; a provider refusal at trace time (no GPU backend, a row that
cannot engage this call) is F6's body counted `xla(refused:<kind>)`; a module layout the provider's parameter mapping does not cover is F6's
body counted `xla(<reason>)`. A kernel row that steps aside at trace time hands over to the provider's next candidate (the provider
records the step-aside by name); the arm that actually served is read back from the provider's own counters and is the census word.

Family words (the provider's cell tables, `opt_core/kernels/pallas/PALLAS_CELLS.json`): bfloat16 activations (P7 `halfpair`'s pair blocks)
are asked under form `af2` (unit pair, c = c_hidden = C: the cells that cover this module's fused forward+backward rows per GPU and
size), float32 activations under form `joltz` (the bias-free f32 cells). The module has no biases; LayerNorm epsilon must be the provider's
(1e-5) and concrete at trace time.

Parameter layout (`provider_params`): torch-layout Linear weights [C_out, C_in] become the provider's math layout `x @ w` [C_in, C_out];
plane a = p_in's FIRST C output channels, plane b = the last C; outgoing: left = a, right = b; incoming (joltz contracts 'bkid,bkjd->bijd',
the provider's incoming equation is 'kjc,kic->ijc'): left = b, right = a; LayerNorm vectors float32 (a missing bias is zeros).

Census (`census()`, F8's LEVER words): per traced call site `<equation>:B<b>xN<n>xC<c>:<dtype>:word=<tier>:fwd=<arm>:ad=<arm>:body=<kernel|xla(<why>)>`;
`TRACES` counts which rule bodies were traced and what served them (`fun:<arm>`, `fwd:<arm>`, `bwd`). `probe()` imports the provider and
names what the word resolves to at N=512, C=128 on this card for both dtypes and call kinds (facts for the LEVER line, decided by the provider).
"""
from collections import Counter
from functools import partial
from typing import Any, Dict, Optional, Tuple

LEVER = "F8"
NAME = "trimul_provider"
MODULE = "mosaic.fast.trimul_provider"
VERSION = "trimul_provider/1"
WORDS = ("fast", "big")                                        # the provider's tier words this op is asked with (the mode carries the word)
SETTING = "fast"
UNIT = "pair"
FORMS = {"bf16": "af2", "f32": "joltz"}                         # dtype word -> the provider family form whose cells cover this module
CALL_KINDS = ("fwd", "fwdbwd")                                  # provider `direction`: the un-differentiated call | the differentiated call
LN_EPS = 1e-5                                                   # the provider rows' LayerNorm epsilon (joltz's modules carry the same constant)
FACE = "triangle_multiplication"                                # the provider face this op calls (its served-arm counters are keyed by it)

STATE: Dict[str, Any] = {"on": False, "word": None}
CENSUS: "Counter[str]" = Counter()                               # census word -> traced calls
TRACES: "Counter[str]" = Counter()                               # rule body:arm -> traces
ROWS: "Counter[str]" = Counter()                                 # "<dtype>:<fwd arm>/<ad arm>" -> traced calls (the LEVER line's rows= word)
CELLS: set = set()                                              # the provider cell keys the answers came from
_SEL: Dict[Tuple, Any] = {}                                     # (word, dtype word, n, c, equation, kind) -> Selection
_OPS: Dict[Tuple, Any] = {}                                     # (word, dtype word, n, c, equation, fwd arm, ad arm) -> the custom_vjp op


class Refusal(RuntimeError):
    """A named refusal: `.reason` (== `.kind`) is the word."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = self.kind = reason
        self.detail = detail
        super().__init__(f"{LEVER} {NAME}: {reason}" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------------------------------------------------------------------ the provider
def _provider():
    """(opt_core.kernels.pallas, opt_core.kernels.pallas.serve); an opt_core without the provider is `provider_missing` by name."""
    try:
        from opt_core.kernels import pallas as P  # noqa: WPS433
        from opt_core.kernels.pallas import serve as PS  # noqa: WPS433
    except ImportError as e:
        raise Refusal("provider_missing", f"opt_core.kernels.pallas is not in the installed opt_core ({type(e).__name__}: {e})") from None
    return P, PS


def _dtype_word(dtype) -> Optional[str]:
    import jax.numpy as jnp  # noqa: WPS433
    dt = jnp.dtype(dtype)
    return "bf16" if dt == jnp.bfloat16 else ("f32" if dt == jnp.float32 else None)


def _lead_count(x) -> int:
    b = 1
    for d in x.shape[:-3]:
        b *= int(d)
    return b


def _cell_token(key) -> str:
    return "none" if key is None else str(key).replace("|", "/").replace(" ", "")


def _concrete_eps(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:  # noqa: BLE001 — a traced eps cannot be checked: the call is unserved by name
        return None


def select(dt: str, n: int, c: int, equation: str, kind: str, word: Optional[str] = None):
    """The provider's Selection for one call class under the tier word (cached): `.row` is a kernel row or 'xla' (F6's body by cell)."""
    import jax.numpy as jnp  # noqa: WPS433
    word = word or STATE["word"] or SETTING
    key = (word, dt, int(n), int(c), equation, kind)
    sel = _SEL.get(key)
    if sel is None:
        P, PS = _provider()
        fam = P.family("trimul", form=FORMS[dt], unit=UNIT, c=int(c), c_hidden=int(c), equation=equation)
        sel = _SEL[key] = PS.resolve("trimul", fam, jnp.bfloat16 if dt == "bf16" else jnp.float32, int(n), word=word, direction=kind)
        CELLS.add(_cell_token(getattr(sel, "cell_key", None)))
    return sel


# -------------------------------------------------------------------------------------------------------------------------- parameter layout
def unserved_reason(module, x) -> Optional[str]:
    """`None` when the provider's parameter mapping covers this module on this activation, else the reason's NAME (the call then runs F6's
    body, counted)."""
    if _dtype_word(x.dtype) is None:
        return f"dtype_{x.dtype}"
    if len(x.shape) < 3 or int(x.shape[-3]) != int(x.shape[-2]):
        return "x_not_square_pair"
    for name in ("p_in", "g_in", "p_out", "g_out"):
        if getattr(getattr(module, name), "bias", None) is not None:
            return f"bias_{name}"
    for name in ("norm_in", "norm_out"):
        ln = getattr(module, name)
        if getattr(ln, "weight", None) is None:
            return f"{name}_no_weight"
        eps = _concrete_eps(getattr(ln, "eps", LN_EPS))
        if eps is None:
            return f"{name}_eps_traced"
        if abs(eps - LN_EPS) > 1e-12:
            return f"{name}_eps_{eps:g}"
    c = int(x.shape[-1])
    if tuple(int(s) for s in module.p_in.weight.shape) != (2 * c, c) or tuple(int(s) for s in module.g_in.weight.shape) != (2 * c, c):
        return "weight_shape_in"
    if tuple(int(s) for s in module.p_out.weight.shape) != (c, c) or tuple(int(s) for s in module.g_out.weight.shape) != (c, c):
        return "weight_shape_out"
    return None


def operands(module) -> Dict[str, Any]:
    """The joltz module's parameters as `x @ w` matrices [C_in, C_out] (plane a = p_in's first C output channels, plane b = the last C) and
    float32 LayerNorm vectors (a missing bias is zeros)."""
    import jax.numpy as jnp  # noqa: WPS433
    f32 = jnp.float32
    c = int(module.norm_in.weight.shape[0])
    wp, wg = module.p_in.weight, module.g_in.weight                     # [2C, C]
    zeros = jnp.zeros((c,), f32)

    def vec(ln, attr):
        v = getattr(ln, attr, None)
        return zeros if v is None else v.astype(f32)
    return dict(c=c, ln_in_scale=vec(module.norm_in, "weight"), ln_in_offset=vec(module.norm_in, "bias"),
                ln_c_scale=vec(module.norm_out, "weight"), ln_c_offset=vec(module.norm_out, "bias"),
                wpa=wp[:c].T, wga=wg[:c].T, wpb=wp[c:].T, wgb=wg[c:].T, w_out=module.p_out.weight.T, w_gate=module.g_out.weight.T)


def provider_params(module, equation: str) -> Dict[str, Any]:
    """The module's parameters in the provider's math layout (module docstring: which plane is `left` per equation)."""
    k = operands(module)
    a = dict(left_w=k["wpa"], right_w=k["wpb"], left_gate_w=k["wga"], right_gate_w=k["wgb"])
    if equation == "incoming":
        a = dict(left_w=k["wpb"], right_w=k["wpa"], left_gate_w=k["wgb"], right_gate_w=k["wga"])
    return dict(ln_in_scale=k["ln_in_scale"], ln_in_offset=k["ln_in_offset"], ln_c_scale=k["ln_c_scale"], ln_c_offset=k["ln_c_offset"],
                out_w=k["w_out"], gate_w=k["w_gate"], **a)


# ------------------------------------------------------------------------------------------------------------------------------ the bodies
def _served_arm(PS, before: Dict[str, int]) -> str:
    """The arm the provider's walk served since `before` (its own counters `served:<face>:<arm>`), or 'unknown'."""
    pre = f"served:{FACE}:"
    for k, v in PS.COUNTS.items():
        if k.startswith(pre) and v > before.get(k, 0):
            return k[len(pre):]
    return "unknown"


def face_call(sel, kind: str, module, x, mask, equation: str):
    """F6's contract through the provider face under Selection `sel` (one [N, N, C] call per leading-batch element). Returns (out, arm):
    `arm` is what the provider's walk actually served (its next candidate when the head steps aside at trace time, by name)."""
    import jax.numpy as jnp  # noqa: WPS433
    _, PS = _provider()
    lead = x.shape[:-3]
    n1, n2, c = (int(v) for v in x.shape[-3:])
    xs = x.reshape((-1, n1, n2, c))
    ms = jnp.reshape(mask, (-1, n1, n2)).astype(xs.dtype)
    params = provider_params(module, equation)
    dt = _dtype_word(x.dtype)
    outs, arm = [], sel.arm
    for b in range(int(xs.shape[0])):
        before = dict(PS.COUNTS)
        outs.append(PS.triangle_multiplication(xs[b], ms[b], params, equation=equation, form=FORMS[dt], unit=UNIT, word=sel.word or STATE["word"] or SETTING,
                                               direction=kind, selection=sel, strict=False))
        arm = _served_arm(PS, before)
    out = jnp.stack(outs).reshape(lead + tuple(int(v) for v in outs[0].shape))
    return out.astype(x.dtype), arm


def _build(word: str, dt: str, n: int, c: int, equation: str, sel_fwd, sel_ad):
    """The custom_vjp op for one call class: op(static, params, x, mask) -> out. `fun` = the `fwd` selection (F6's body when its row is xla);
    the forward RULE = the `fwdbwd` selection inside jax.vjp (F6's body when xla), the backward rule = that pullback."""
    import equinox as eqx  # noqa: WPS433
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    from mosaic.fast.trimul_layout import trimul_cmajor  # noqa: WPS433

    def body(sel, kind, static, params, x, mask):
        module = eqx.combine(params, static)
        if sel is None or sel.row == "xla":
            TRACES[f"{'fun' if kind == 'fwd' else 'fwd'}:xla"] += 1
            return trimul_cmajor(module, x, mask, equation)
        out, arm = face_call(sel, kind, module, x, mask, equation)
        TRACES[f"{'fun' if kind == 'fwd' else 'fwd'}:{arm}"] += 1
        return out

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def op(static, params, x, mask):                                   # the un-differentiated call
        return body(sel_fwd, "fwd", static, params, x, mask)

    def fwd(static, params, x, mask):                                  # the differentiated call: the fwdbwd row's forward, its pullback saved
        if jnp.issubdtype(jnp.asarray(mask).dtype, jnp.inexact):
            out, pull = jax.vjp(lambda p_, x_, m_: body(sel_ad, "fwdbwd", static, p_, x_, m_), params, x, mask)
            return out, (pull, None)
        out, pull = jax.vjp(lambda p_, x_: body(sel_ad, "fwdbwd", static, p_, x_, mask), params, x)
        return out, (pull, mask)

    def bwd(static, res, d_out):
        pull, zmask = res
        TRACES["bwd"] += 1
        got = pull(d_out)
        if zmask is None:
            return got
        d_params, d_x = got
        return d_params, d_x, np.zeros(np.shape(zmask), dtype=jax.dtypes.float0)   # a bool / integer mask: no cotangent (float0), as jax spells it

    op.defvjp(fwd, bwd)
    return op


# -------------------------------------------------------------------------------------------------------------------------------- the entry
def trimul(module, x, mask, direction: str):
    """F6's contract, served by the provider's answer for this call under the configured tier word (module doc)."""
    if not STATE["on"]:
        raise Refusal("not_configured", f"{MODULE}.trimul called before configure(): F8 would carry this op's name on an unconfigured body")
    if direction not in ("outgoing", "incoming"):
        raise Refusal("unknown_direction", repr(direction))
    import equinox as eqx  # noqa: WPS433
    from mosaic.fast.trimul_layout import trimul_cmajor  # noqa: WPS433
    word = STATE["word"]
    n, c, b = int(x.shape[-2]), int(x.shape[-1]), _lead_count(x)
    head = f"{direction}:B{b}xN{n}xC{c}:{_dtype_word(x.dtype) or x.dtype}:word={word}"
    reason = unserved_reason(module, x)
    if reason is not None:
        CENSUS[f"{head}:fwd=xla:ad=xla:body=xla({reason})"] += 1; ROWS[f"{_dtype_word(x.dtype) or x.dtype}:xla/xla"] += 1
        return trimul_cmajor(module, x, mask, direction)
    dt = _dtype_word(x.dtype)
    P, _ = _provider()
    try:
        sel_fwd = select(dt, n, c, direction, "fwd", word)
        sel_ad = select(dt, n, c, direction, "fwdbwd", word)
    except P.Refusal as e:                                              # the provider cannot answer here (no GPU backend, ...): F6's body BY NAME
        CENSUS[f"{head}:fwd=xla:ad=xla:body=xla(refused:{getattr(e, 'kind', type(e).__name__)})"] += 1; ROWS[f"{dt}:xla/xla"] += 1
        return trimul_cmajor(module, x, mask, direction)
    tok = f"{head}:fwd={sel_fwd.arm}:ad={sel_ad.arm}:cells={_cell_token(sel_fwd.cell_key)}+{_cell_token(sel_ad.cell_key)}"
    ROWS[f"{dt}:{sel_fwd.arm}/{sel_ad.arm}"] += 1
    if sel_fwd.row == "xla" and sel_ad.row == "xla":
        CENSUS[tok + ":body=xla(cell)"] += 1
        return trimul_cmajor(module, x, mask, direction)
    CENSUS[tok + ":body=kernel"] += 1                                   # once per TRACE of the call site; TRACES names what each rule body ran
    key = (word, dt, n, c, direction, sel_fwd.arm, sel_ad.arm)
    fn = _OPS.get(key)
    if fn is None:
        fn = _OPS[key] = _build(word, dt, n, c, direction, sel_fwd, sel_ad)
    params, static = eqx.partition(module, eqx.is_array)
    return fn(static, params, x, mask)


trimul.__serves_dtypes__ = ("float32", "bfloat16")               # read by P7 halfpair: both activation dtypes are this op's — no dtype route


# --------------------------------------------------------------------------------------------------------------------------------- surface
def configure(word: Optional[str] = None) -> Dict[str, Any]:
    """Turn the op on under a tier word (None = SETTING); clears the cached selections and ops. Refuses `unknown_word` by name."""
    w = SETTING if word is None else str(word).strip().lower()
    if w not in WORDS:
        raise Refusal("unknown_word", f"{word!r}: the op is asked with one of the provider's tier words {', '.join(WORDS)}")
    STATE.update(on=True, word=w)
    _SEL.clear(); _OPS.clear()
    return settings()


def reset() -> Dict[str, Any]:
    STATE.update(on=False, word=None)
    _SEL.clear(); _OPS.clear(); CENSUS.clear(); TRACES.clear(); ROWS.clear(); CELLS.clear()
    return settings()


def settings() -> Dict[str, Any]:
    return {"on": bool(STATE["on"]), "word": STATE["word"], "version": VERSION, "forms": dict(FORMS), "unit": UNIT, "provider": "opt_core.kernels.pallas"}


def census() -> Dict[str, Any]:
    """kernel_calls / xla_calls (traced calls per body), words (the census), traces (rule bodies), rows (dtype:fwd-arm/ad-arm), cells."""
    return {"kernel_calls": int(sum(v for k, v in CENSUS.items() if k.endswith(":body=kernel"))),
            "xla_calls": int(sum(v for k, v in CENSUS.items() if ":body=xla(" in k)),
            "calls": int(sum(CENSUS.values())), "words": dict(CENSUS), "traces": dict(TRACES), "rows": dict(ROWS), "cells": sorted(CELLS)}


def probe() -> Dict[str, Any]:
    """Import the provider and ask it, at N=512 / C=128 / outgoing, what the word resolves to for both dtypes and call kinds on this card:
    {"ok", "kind", "cc", "bf16_fwd", "bf16_fwdbwd", "f32_fwd", "f32_fwdbwd"}. Not ok (by name) when the provider is missing or cannot answer
    (no GPU backend: `backend_not_gpu`); a word that resolves to xla everywhere is still ok — the calls run F6's body, counted."""
    out: Dict[str, Any] = {"ok": True, "kind": "ok", "cc": None, "version": VERSION}
    try:
        P, PS = _provider()
    except Refusal as e:
        return dict(out, ok=False, kind=e.reason)
    try:
        out["cc"] = PS.running_cc()
    except Exception as e:  # noqa: BLE001 — the provider's refusal (backend_not_gpu on a CPU host) by name
        return dict(out, ok=False, kind=str(getattr(e, "kind", None) or type(e).__name__))
    for dt in ("bf16", "f32"):
        for kind in CALL_KINDS:
            try:
                sel = select(dt, 512, 128, "outgoing", kind, STATE["word"] or SETTING)
                out[f"{dt}_{kind}"] = f"{sel.arm}@{_cell_token(sel.cell_key)}"
            except Exception as e:  # noqa: BLE001
                out[f"{dt}_{kind}"] = f"unavailable({getattr(e, 'kind', None) or getattr(e, 'reason', None) or type(e).__name__})"
    return out


KERNEL_NAME_MARKERS = ("trimul", "tri_mul")                              # kernel names of the provider's triangle-multiplication rows, as they appear in HLO custom calls
CUSTOM_CALL_MARKERS = ("__gpu$xla.gpu.triton", "pallas", "mosaic_gpu", "trimul")


def hlo_kernel_calls(hlo_text: str) -> Dict[str, Any]:
    """Count kernel custom calls in an HLO text dump (shows which executable a kernel landed in): `kernel_calls` = custom calls carrying a
    Pallas/Triton/FFI marker, `named_calls` = those whose name carries a triangle-multiplication kernel marker, `by_name`, `targets`."""
    import re  # noqa: WPS433
    text = hlo_text if isinstance(hlo_text, str) else (hlo_text.as_text() if hasattr(hlo_text, "as_text") else str(hlo_text))
    targets: Counter = Counter(); by_name: Counter = Counter(); kernel_calls = 0
    for line in text.splitlines():
        if "custom-call" not in line and "custom_call" not in line:
            continue
        m = re.search(r'custom_call_target="([^"]+)"', line)
        target = m.group(1) if m else ""
        nm = re.search(r'name\s*=\s*\\?"([^"\\]+)\\?"', line)
        name = nm.group(1) if nm else ""
        if any(k in target for k in CUSTOM_CALL_MARKERS) or any(k in name for k in KERNEL_NAME_MARKERS):
            kernel_calls += 1; targets[target] += 1
            if any(k in name for k in KERNEL_NAME_MARKERS) or any(k in target for k in KERNEL_NAME_MARKERS):
                by_name[name or target] += 1
    return {"kernel_calls": kernel_calls, "named_calls": int(sum(by_name.values())), "by_name": dict(by_name), "targets": dict(targets)}
