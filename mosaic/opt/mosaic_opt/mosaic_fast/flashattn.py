"""K1 `triattn` — joltz's TriangleAttention core served by the shared core's JAX-family provider (opt_core.kernels.pallas) by TIER WORD.

What it does. `install()` rebinds `joltz.TriangleAttention.__call__` (the one class behind the trunk pairformer's, the MSA module's, the template
module's and the confidence module's triangle attentions, starting AND ending node). With the lever configured on, a call keeps the module's own
LayerNorm, q/k/v/gate/output Linears and bias Linear and replaces only the attention core — `q·kᵀ/√d + mask bias + triangle bias → softmax → ·v`,
which stock materialises as an [rows, H, N, N] float32 array — by ONE call of the provider's attention face
(`opt_core.kernels.pallas.serve.attention(..., kind="tri", direction="fwdbwd", layout="BHSD", word=<WORD>)`: forward and backward, the design
step is a gradient). The provider owns every implementation ("row") of that core and the cell table (PALLAS_CELLS.json, per jax line ×
compute capability × operand dtype × head geometry × token bucket × direction); this module owns only the mapping: rows of the pair representation
= the batch, the triangle bias `linear(x)[q, k, h]` = the pair bias `[H, N_q, N_k]` shared over the batch, the pair mask row `mask[i, :]` = the
boolean key mask of batch element i (the model's pair mask is binary: token padding), `1/√c_hidden` = the scale, heads-major operands
[rows, H, tokens, D] (the layout the kernel rows read; the head split is this module's). A FULLY masked row (a padded token's row: every key
masked) is served over the live key columns instead (finite, and a sane backward for every row); stock's such row is exactly uniform attention
(its −1e9 mask absorbs the logits in float32), so padded rows' pair activations DIFFER from stock's. They are isolated: every consumer of the pair
activation masks padded rows and keys, so they never reach a live pair position or the loss, and they carry zero cotangent.

Words (`configure(SPEC)`; `parse` refuses anything else BY NAME before a trace):
  stock | off           every call on joltz's own body (the lever off).
  fast | big | exact  a TIER word, handed to the provider as it is: per call the provider serves the cell of (this jax line, this card's
                        compute capability, the operands' dtype — bfloat16 under P7's pair track, float32 elsewhere —, heads × head dim, the token
                        bucket, fwd+bwd): `fast` = the cell's fastest row, `big` = its lowest-peak-memory row not slower than the stock statement,
                        `exact` = a row bitwise with the stock statement where the cell names one, else the stock statement re-stated through
                        the face (`xla`) — this kit's exact mode configures no kernel lever; the word is accepted for completeness. A row that
                        cannot engage on a call steps aside BY NAME inside the provider (its census) and the cell's next row serves; the walk
                        ends at the stock statement, so a tier word serves every call of a family the cell table covers.
  <row> | <arm>         ONE provider row pinned in place of the tier's choice — a row name (`triattn_xla`, `cd_triatt`, `pallas_attn`,
                        `rowshared`, `tokamax`, `cudnn`, `xla_sdpa`, `xla`, …: `rows()`) or an arm word `row[:f32 class][@setting]`
                        (`triattn_xla@vjp`, `pallas_attn@s3`, `pallas_attn:tf32@bwd_tf32`, `tokamax@triton`, …); a row that refuses this
                        module's call class (differentiated, key-masked, shared pair bias) is refused BY NAME at configure with the provider's
                        word (e.g. `cudnn`: cudnn_shared_bias_grad), a row that refuses one call's shape at trace is a counted fallback BY NAME to
                        joltz's body and the lever's gate then fails the run naming it.
Mode fast of the kit configures the word `fast`, mode big `big` (mosaic_opt.modes KIT_MODES specs): the kernel follows the tier, this
module names no kernel, tile or launch setting of its own.

Uniform lever API (mosaic_opt.levers): install / uninstall / configure(None = RECORD) / describe / gate / emit_line / ENV_REQUIRED. describe()
is FLAT (single-token values) and carries the provider's facts: the word, the compute capability and jax line it selected for, the arm the
provider selects at `DESCRIBE_AT` tokens for bfloat16 and for float32 operands (`sel_bf16`, `sel_f32`, with the cell keys), and the LIVE census —
`served`, `shapes`, and one `<dtype>.<arm>` fact per traced call naming the arm that served it (`aside.<row>.<reason>` for a row that stepped
aside on the way). Refusals of configure, by name (`Refusal.reason`): unknown_spec, core_too_old (the importable opt_core has no JAX-family
provider with the tier words), provider_refused (the provider's own word: backend_not_gpu off a GPU, cudnn_shared_bias_grad, needs_tokamax,
jax_line_unmeasured, …), cache_clear_failed.
"""
import math
import os
from typing import Any, Dict, Optional, Tuple

LEVER = "K1"
NAME = "triattn"
MODULE = "mosaic.fast.flashattn"
PROVIDER = "opt_core.kernels.pallas"                     # the shared core's JAX-family provider: rows, words, the cell table, select() (no framework import)
FACES = "opt_core.kernels.pallas.serve"                  # its call faces (jax imported when a face is called): attention(), resolve(), running_cc(), running_jax_line()
OP = "attn"                                              # the provider op this module maps onto: the attention CORE (the module keeps its own LayerNorm and projections)
KIND = "tri"                                             # the family kind: triangle attention rows of the pair stack (family = attn|H<heads>|D<head dim>, e.g. tri_h4_d32)
DIRECTION = "fwdbwd"                                      # every call is served by a differentiable row: the design step is a gradient (the refold's forward-only calls ride the same rows)
LAYOUT = "BHSD"                                          # operands heads-major [rows, H, tokens, D]: the layout the kernel rows read (the stock-statement rows swap inside the face)
TIER_WORDS = ("fast", "big", "exact")                  # the provider's tier words (opt_core.kernels.pallas TIER_WORDS), checked against the importable provider at parse
RECORD = "fast"                                          # configure(None): the fast tier word
SPECS = ("stock",) + TIER_WORDS + ("<row>", "<row>[:<f32 class>][@<setting>]")   # the spec grammar, for the registry's words
MIN_TOKENS = 0                                           # size rule on N_q (0 = every call handed to the provider; its rows tile and mask edge tiles themselves)
DESCRIBE_AT = 500                                        # the token count describe() / the LEVER line quote the provider's selection at (sel_bf16 / sel_f32); every call selects at its own size
KLASS = "fast"                                           # numerics class: the kernel rows re-associate the softmax reduction (not bitwise with stock)
ENV_REQUIRED: Dict[str, str] = {}                       # nothing must be in the environment before jax import
EXPECTED_FALLBACKS = ()                                  # the fallback reasons this lever declares for Boltz-2: none — a tier word serves every call (a geometry or token bucket with no
                                                         #   cell is served by the provider's stock-statement row, named in its census); a provider refusal of one call
                                                         #   (a row word's envelope, a jax line without cells) is counted BY NAME (`unserved_<reason>`) and the gate fails the run

STATE: Dict[str, Any] = {"spec": None, "impl": None, "word": None, "kind": None, "cc": None, "jax_line": None, "op": None, "ledger": None, "override": False,
                         "selections": {}, "sel": {}}
_ORIG: Dict[str, Any] = {}


class Refusal(RuntimeError):
    """A named refusal of `configure` (`.reason` is the word)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


def provider():
    """(the provider package, its faces module), imported — `Refusal('core_too_old')` naming what the importable core lacks."""
    import importlib
    try:
        P = importlib.import_module(PROVIDER)
        PS = importlib.import_module(FACES)
    except ImportError as e:
        raise Refusal("core_too_old", f"lever {LEVER} needs {PROVIDER} (the JAX-family provider with the tier words) from the shared core; the importable core does not carry it: {e}") from None
    missing = [w for w in TIER_WORDS if w not in getattr(P, "TIER_WORDS", ())]
    if missing or not callable(getattr(PS, "attention", None)) or "selection" not in PS.attention.__code__.co_varnames:
        raise Refusal("core_too_old", f"{PROVIDER} at opt_core {_core_version()} lacks the tier words {missing} or the attention face with selection=")
    return P, PS


def _core_version() -> str:
    try:
        import opt_core
        return str(getattr(opt_core, "__version__", "unknown"))
    except ImportError:
        return "absent"


def rows() -> Tuple[str, ...]:
    """The provider rows that serve this module's op (the attention core), in the provider's order — the row / arm words `parse` accepts."""
    P, _ = provider()
    return tuple(r for r, d in P.ROWS.items() if any(o.split(" ")[0] == OP for o in d.get("ops", ())))


def parse(spec: Optional[str]) -> Dict[str, Any]:
    """The spec grammar (module docstring "Words"): None / '' = RECORD; 'stock' | 'off' = the lever off; a tier word; a provider row or arm word
    for the attention core. Anything else is `Refusal('unknown_spec')` naming the words; a provider that is not importable is `core_too_old`.
    `kind` = tier | row | arm; `spec` is the canonical spelling (the word itself)."""
    word = (spec or "").strip() or RECORD
    if word in ("stock", "off"):
        return {"on": False, "spec": "stock", "word": None, "kind": None}
    P, _ = provider()
    if word in TIER_WORDS:
        return {"on": True, "spec": word, "word": word, "kind": "tier"}
    served = rows()
    try:
        row, ip, setting = P.parse_arm(word)
    except P.Refusal as r:
        raise Refusal("unknown_spec", f"{spec!r}: {r} — the words: stock | {' | '.join(TIER_WORDS)} (a tier: the provider's measured cell per call) | "
                                      f"a provider row or arm word of the attention core ({', '.join(served)}; row[:f32 class][@setting])") from None
    if row not in served:
        raise Refusal("unknown_spec", f"{spec!r}: provider row {row!r} does not serve the attention core (op {OP!r}); the rows that do: {', '.join(served)}")
    return {"on": True, "spec": word, "word": word, "kind": "row" if (ip is None and setting is None) else "arm"}


def _clear_trace_caches():
    """joltz's filter_jit'd trunk and mosaic's jitted loss step keep the previously traced body: clear jax's and equinox's caches, or refuse."""
    import jax
    import equinox as eqx
    try:
        jax.clear_caches(); eqx.clear_caches()
    except Exception as e:                                   # a configuration change that cannot evict the traced bodies is refused, never ignored
        raise Refusal("cache_clear_failed", f"{type(e).__name__}: {e}") from None


# --------------------------------------------------------------------------------------------------------------------------- the lever API
def install():
    """Rebind `joltz.TriangleAttention.__call__` (idempotent). Until `configure` turns the lever on, the rebound call IS the stock body."""
    import joltz
    if "triatt" in _ORIG:
        return
    _ORIG["triatt"] = joltz.TriangleAttention.__call__
    joltz.TriangleAttention.__call__ = _triangle_attention_call


def _cleared() -> Dict[str, Any]:
    return dict(spec=None, impl=None, word=None, kind=None, cc=None, jax_line=None, op=None, ledger=None, override=False, selections={}, sel={})


def uninstall():
    import joltz
    if "triatt" in _ORIG:
        joltz.TriangleAttention.__call__ = _ORIG.pop("triatt")
    STATE.update(_cleared())
    _clear_trace_caches()


def configure(spec: Optional[str], *, min_tokens: Optional[int] = None, op=None, cc: Optional[str] = None) -> Dict[str, Any]:
    """Set the lever's configuration for every later trace (`None` = RECORD, ON; 'stock' = off): refuses by name (`Refusal`) BEFORE anything of the
    model is traced; clears the trace caches; returns `describe()`. `cc` selects for a compute capability other than the running device's (as the
    provider's OPT_CORE_PALLAS_CC does; tracing off-target); `op` overrides the served op (tests: a pure-jnp reference over the same layout)."""
    cfg = parse(spec)
    _clear_trace_caches()
    if not cfg["on"]:
        STATE.update(_cleared(), spec="stock")
        return describe()
    P, PS = provider()
    gate_n = int(MIN_TOKENS if min_tokens is None else min_tokens)
    override = op is not None
    cc = cc or os.environ.get(getattr(P, "ENV_CC", "OPT_CORE_PALLAS_CC")) or None
    if cc is None:
        try:
            cc = PS.running_cc()                                            # backend_not_gpu — by name (an override op may still run: it selects nothing)
        except P.Refusal as r:
            if not override:
                raise Refusal("provider_refused", f"{r.kind}: {r}") from None
    line = PS.running_jax_line()
    STATE.update(spec=cfg["spec"], impl="provider", word=cfg["word"], kind=cfg["kind"], cc=cc, jax_line=line, selections={}, sel={}, override=override)
    if op is None:
        op = _provider_op(P, PS, cfg["word"], cc, line)                     # provider_refused — by name, from a probe trace at a served-class shape
    expected = tuple(EXPECTED_FALLBACKS) + (("below_size_rule",) if gate_n > 0 else ())
    from opt_core.counters import Ledger
    STATE.update(op=op, ledger=Ledger(f"{LEVER}.{NAME}", impl=f"{FACES}@{_core_version()}", origin="core", min_tokens=gate_n, expected=expected))
    for dt in ("bf16", "f32"):                                              # the provider's selection at DESCRIBE_AT tokens, quoted as facts (every call selects at its own size)
        STATE["sel"][dt] = _selection_words(P, cfg["word"], cc, line, dt, DESCRIBE_AT)
    return describe()


def current_op():
    return STATE["op"]


def _selection_words(P, word: str, cc: Optional[str], line: str, dt: str, n: int) -> Dict[str, str]:
    """{arm, cell} the provider selects for the pairformer geometry (4 heads × 32) at `n` tokens, or {arm: refused:<kind>}."""
    try:
        fam = P.family(op=OP, kind=KIND, heads=4, head_dim=32)
        s = P.select(line, str(cc), dt, OP, fam, int(n), DIRECTION, word=word, key_masked=True)
        return {"arm": s.arm, "cell": s.cell_key or "none", "order": "+".join(s.candidates)}
    except Exception as r:  # noqa: BLE001 — the provider's Refusal (cell_unmeasured, backend …): quoted, not raised (the call decides per call)
        return {"arm": "refused:" + str(getattr(r, "kind", type(r).__name__)), "cell": "none", "order": "none"}


def _provider_op(P, PS, word: str, cc: Optional[str], line: str):
    """The served op over the kernel layout: q, k, v [rows, H, tokens, D] (bfloat16 | float32) · pair bias [H, T, S] · boolean key mask [rows, S] ·
    scale -> [rows, H, T, D] in q's dtype = ONE call of the provider's attention face under `word`, with the Selection resolved once per (dtype,
    heads, head dim, tokens) and the arm that served censused per traced call. Probed here once by a trace with a gradient (q and the pair
    bias) at a served-class shape in both operand dtypes: a word the provider refuses for this call class is `Refusal('provider_refused')` with the
    provider's own word."""
    import jax
    import jax.numpy as jnp

    def select(dtype, H: int, D: int, S: int):
        key = (jnp.dtype(dtype).name, int(H), int(D), int(S))
        sel = STATE["selections"].get(key)
        if sel is None:
            fam = P.family(op=OP, kind=KIND, heads=int(H), head_dim=int(D)) # attn|H<h>|D<d>: a geometry without a cell is refused at select for a tier word (cell_unmeasured)
            sel = PS.resolve(OP, fam, dtype, int(S), word=word, direction=DIRECTION, cc=cc, jax_line=line, key_masked=True)
            STATE["selections"][key] = sel
        return sel

    def op(q, k, v, bias, kmask, scale):
        H, S, D = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
        sel = select(q.dtype, H, D, S)
        before = dict(PS.COUNTS)
        out = PS.attention(q, k, v, bias, key_mask=kmask, scale=scale, word=word, kind=KIND, direction=DIRECTION, layout=LAYOUT, selection=sel,
                           strict=False, cc=cc, jax_line=line)
        _census(before, PS.COUNTS, q.dtype)
        return out.astype(q.dtype)

    try:                                                                    # lowerability probe at served-class shapes (bf16 and f32, bias, mask, grad incl. d(bias)) — once, here, by name
        for dt in (jnp.bfloat16, jnp.float32):
            z = jnp.zeros((2, 4, 32, 32), dt); m = jnp.ones((2, 32), bool).at[:, -1].set(False)
            jax.grad(lambda qq, bb: op(qq, z, z, bb, m, 0.25).astype(jnp.float32).sum(), argnums=(0, 1))(z, jnp.zeros((4, 32, 32), dt))
    except P.Refusal as r:
        raise Refusal("provider_refused", f"{r.kind}: {str(r)[:400]}") from None
    except Exception as e:                                                  # noqa: BLE001 — a lowering / launch error of the selected row on this stack, named
        raise Refusal("provider_refused", f"{type(e).__name__}: {str(e)[:400]}") from None
    finally:
        L = STATE.get("ledger")
        if L is not None:
            L.clear()
    STATE["selections"].clear()                                             # the probe's selections were at the probe's size
    op.layout = "kernel"
    return op


def _census(before: Dict[str, int], after: Dict[str, int], dtype) -> None:
    """One fact per traced call on the ledger in force: `<dtype>.<arm>` for the arm whose served count rose in the provider's counters,
    `aside.<row>.<reason>` for every row that stepped aside on the way (the provider's refusal words)."""
    L = STATE.get("ledger")
    if L is None:
        return
    import jax.numpy as jnp
    dt = "bf16" if jnp.dtype(dtype) == jnp.bfloat16 else ("f32" if jnp.dtype(dtype) == jnp.float32 else jnp.dtype(dtype).name)
    for key, n in after.items():
        d = int(n) - int(before.get(key, 0))
        if d <= 0:
            continue
        parts = key.split(":")
        if parts[0] == "served" and len(parts) >= 3 and parts[1] == "attention":
            L.count(f"{dt}.{':'.join(parts[2:])}".replace(":", "~"), d)      # an arm's f32-class colon spelled '~' (the census token grammar keeps ':' for key:value)
        elif parts[0] == "refused" and len(parts) >= 3:
            L.count(f"aside.{parts[1]}.{parts[2]}", d)


def describe() -> Dict[str, Any]:
    """The effective configuration in force, FLAT (single-token values): lever, module, klass, on, spec, and — when on — the provider's identity
    (impl = the faces module @ the core's version, origin core), the word and its kind, the compute capability and jax line selected for, the
    direction and layout of the call, the selection at DESCRIBE_AT tokens per operand dtype (sel_bf16 / cell_bf16 / sel_f32 / cell_f32), the
    declared fallbacks, and the LIVE census (served, fallback, fallback_by, min_tokens, shapes, facts = the per-arm counts, calls)."""
    on = STATE["op"] is not None
    out: Dict[str, Any] = {"lever": LEVER, "lever_name": NAME, "module": MODULE, "klass": KLASS, "on": int(on),
                           "spec": STATE["spec"] or "stock", "installed": int("triatt" in _ORIG)}
    if not on:
        return out
    L = STATE["ledger"]
    out.update(impl=L.impl, origin=L.origin, word=STATE["word"], kind=STATE["kind"], cc=STATE["cc"] or "none", jax_line=STATE["jax_line"] or "none",
               direction=DIRECTION, layout=LAYOUT, describe_at=DESCRIBE_AT, override=int(STATE["override"]), expected_fallbacks=",".join(L.expected) or "none")
    for dt, s in STATE["sel"].items():
        out[f"sel_{dt}"] = s.get("arm"); out[f"cell_{dt}"] = s.get("cell")
    f = L.fields()                                                    # LIVE counters of everything traced since configure(): served / fallback / fallback_by / shapes / facts
    for k, v in f.items():
        if k not in ("name", "state", "tag", "reason", "impl", "origin"):
            out[k] = _token(v)
    out["calls"] = int(f.get("served", 0) or 0) + int(f.get("fallback", 0) or 0)
    return {k: _token(v) for k, v in out.items()}                    # every value a scalar AND one blank-free token (the LEVER line splits fields on blanks)


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
    """The LEVER line of this process's census (the core's grammar, `opt_core.counters.Ledger.line`): the provider's identity, the counters, and the
    selection facts; the off line when the lever is off."""
    from opt_core.report import emit, lever_line
    L = STATE["ledger"]
    if L is None:
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason="not_configured", impl=MODULE, origin="kit"))
    ev = dict(word=STATE["word"], kind=STATE["kind"], cc=STATE["cc"] or "none", jax_line=STATE["jax_line"] or "none", direction=DIRECTION, layout=LAYOUT,
              describe_at=DESCRIBE_AT)
    for dt, s in STATE["sel"].items():
        ev[f"sel_{dt}"] = s.get("arm"); ev[f"cell_{dt}"] = s.get("cell")
    if STATE["override"]:
        ev["override"] = 1
    return emit(L.line(tag, **ev))


def gate():
    """Fail closed: raises when the lever is installed but not configured ON (a run under the lever's name with every call on the stock
    body), when the census holds an error or a fallback reason the lever does not declare, or when calls happened with none served."""
    L = STATE["ledger"]
    if L is None:
        if "triatt" in _ORIG:
            raise Refusal("not_configured", "K1 is installed but configure() never turned it on: the run would carry the lever's name with every call on the stock body")
        return
    g = L.gate(f"{LEVER}.{NAME}")                                        # the core Ledger's gate result (opt_core.gates.Gate) returns, never raises — raised here
    if not g.ok:
        raise Refusal("lever_gate", str(g.reason))
    if L.served == 0:
        raise Refusal("nothing_served", f"{LEVER} configured on and no triangle attention was traced through the op")


# --------------------------------------------------------------------------------------------------------------------------- the call
def _unserved(exc) -> Optional[str]:
    """The fallback word of a provider refusal raised while tracing one call (`unserved_<kind>`), None for any other exception (re-raised)."""
    kind = getattr(exc, "kind", None)
    if kind is None or not type(exc).__name__ == "Refusal":
        return None
    return "unserved_" + "_".join(str(kind).split())[:40]


def _triangle_attention_call(self, x, mask):
    """`joltz.TriangleAttention.__call__` with the attention core served (see the module docstring for the mapping)."""
    if STATE["op"] is None:
        return _ORIG["triatt"](self, x, mask)
    L = STATE["ledger"]
    x0, mask0 = x, mask
    n = x.shape[-2] if self.starting else x.shape[-3]                   # N_q = N_k after the ending-node transpose
    if L.min_tokens and int(n) < int(L.min_tokens):
        L.fallback("below_size_rule")
        return _ORIG["triatt"](self, x, mask)
    import einops
    import jax
    import jax.numpy as jnp
    if not self.starting:
        x = einops.rearrange(x, "... I J C_in -> ... J I C_in")
        mask = einops.rearrange(mask, "... I J -> ... J I")
    x = self.layer_norm(x)
    mha = self.mha
    H, D = int(mha.no_heads), int(mha.c_hidden)
    lead = x.shape[:-3]
    I, J = x.shape[-3], x.shape[-2]
    xs = x.reshape((-1, I, J, x.shape[-1]))                             # [L, I, J, C]: L = the leading (batch / template) dims flattened
    ms = mask.reshape((-1, I, J))
    op = STATE["op"]
    lay = "l i j (h d) -> l i h j d"                                    # heads-major [rows, H, tokens, D] (LAYOUT): rows i = the batch
    q = einops.rearrange(mha.linear_q(xs), lay, h=H)
    k = einops.rearrange(mha.linear_k(xs), lay, h=H)
    v = einops.rearrange(mha.linear_v(xs), lay, h=H)
    bias = einops.rearrange(self.linear(xs), "l q k h -> l h q k").astype(q.dtype)   # triangle_bias[h, q, k] = linear(x)[q, k, h], shared over rows
    kmask = ms > 0.5                                                    # mask_bias = inf·(mask−1): key k of row i is dropped where mask[i, k] = 0
    col = kmask.any(axis=-2, keepdims=True)                             # the keys live in SOME row (for the pairformer's outer-product pair mask: the token mask)
    col = col | ~col.any(axis=-1, keepdims=True)                        # nothing live anywhere: every key (finite outputs either way)
    kmask = jnp.where(kmask.any(axis=-1, keepdims=True), kmask, col)    # a row with EVERY key masked (a padded token's row) is served over the
    #   live columns instead: finite outputs and a sane backward for every row. Stock's such row is EXACTLY uniform attention (−1e9 absorbs the
    #   logits in float32), so padded rows DIFFER from stock — and are isolated: every consumer of the pair activation masks padded rows/keys,
    #   they never reach a live pair position or the loss, and they carry zero cotangent.
    scale = 1.0 / math.sqrt(D)
    try:
        o = jnp.stack([op(q[l], k[l], v[l], bias[l], kmask[l], scale) for l in range(xs.shape[0])])   # [L, I, H, J, D]
    except Exception as e:                                              # noqa: BLE001 — a provider refusal of THIS call (no family, no cell, a row word's envelope):
        why = _unserved(e)                                              #   counted BY NAME, the call on joltz's body; anything else is raised
        if why is None:
            raise
        L.fallback(why)
        return _ORIG["triatt"](self, x0, mask0)
    L.serve(f"B{I}xH{H}xS{J}xD{D}" + (f"x{xs.shape[0]}" if xs.shape[0] > 1 else ""))   # B<rows>xH<h>xS<n>xD<d> [x<leading dims>]
    o = einops.rearrange(o, "l i h j d -> l i j h d")
    if mha.linear_g is not None:
        g = einops.rearrange(jax.nn.sigmoid(mha.linear_g(xs)), "l i j (h d) -> l i j h d", h=H)
        o = o * g
    out = mha.linear_o(einops.rearrange(o, "l i j h d -> l i j (h d)")).reshape(lead + (I, J, -1))
    if not self.starting:
        out = einops.rearrange(out, "... J I C_in -> ... I J C_in")
    return out
