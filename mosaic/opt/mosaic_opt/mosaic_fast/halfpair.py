"""P7 `halfpair` — the trunk pairformer's PAIR track computed in bfloat16, the way upstream Boltz-2 runs its trunk (torch `bf16-mixed` autocast,
`boltz/main.py predict: Trainer(precision="bf16-mixed")`), inside joltz's float32 model: a dtype lever, beside P6's matmul-precision words.

What it does. `install()` rebinds `joltz.PairformerLayer2.__call__` (the block of the trunk's 64-block Pairformer2 and of the confidence module's
pairformer), `joltz.PairformerNoSeqLayer.__call__` (the pair-only block of the MSA module's 4 layers and of the template module) and wraps
`joltz.Pairformer2.__call__` (the stacked scan; whatever call is bound there at install — joltz's own, or P5's grouped-remat scan when P5 was installed
first — is the call wrapped, never re-stated). With the lever configured on, a block runs each pair sub-layer whose REGION token is in the spec —
  tm  the two triangle multiplications      ta  the two triangle attentions      tz  the pair transition      (pf = tm+ta+tz)
— on bfloat16 operands: the sub-layer's own module (its LayerNorms, Linears, gates: joltz's classes, or the call another lever serves there — F6's
channel-major triangle multiplication, K1's fused triangle attention, P5's `sub` rematerialisation wrapper) is applied with its floating parameters cast
to bfloat16 to the block's pair activation cast to bfloat16 (and the pair mask cast alike), and its output is cast back to float32 BEFORE the residual
add. What stays float32, as torch autocast keeps it: LayerNorm statistics (equinox's LayerNorm promotes to float32 inside and casts back), the residual
stream and every residual accumulation inside a block (`z + dropout·f(z)` in float32), the whole sequence track (pre_norm_s, the pair-biased sequence
attention — which reads the float32 pair activation for its bias, upstream's `z.float()` — transition_s, s_post_norm: upstream disables autocast
there), softmax statistics inside K1's cuDNN attention (float32 accumulate), and everything outside the pairformer blocks (input embedder, MSA
averaging / outer-product mean, distogram head, diffusion conditioning, the fp32 sampler, the loss). GEMM products accumulate in float32 (cuBLAS bf16
tensor-core GEMMs; XLA's default for bfloat16 dots). Token `carry`: the pair activation is CARRIED in bfloat16 between the 64 blocks (cast once before
the stacked scan, back to float32 after it; inside a block the five residual adds still accumulate in float32 and the sum is rounded to bfloat16 once, at
the block's exit) — the scan's per-block checkpoints (and P5's group carries) are then half-size; without it the carry and the checkpoints stay float32
and only the sub-layer intermediates are bfloat16. Token `msa`: the MSA module's (and the template module's) pair-only blocks run the same policy
(their pair-weighted averaging and outer-product mean stay float32).

Composition (by construction, each named on describe() / the LEVER line):
  F6 trimul_cmajor      dtype-generic: serves the bfloat16 call as is (trimul=f6_xla).
  F8 trimul_fused       holds F6's body (trimul_layout.set_body). A body that declares bfloat16 in `__serves_dtypes__` (F8's op does) takes the bfloat16
                        calls as they are: no route, word `f8_<word>`, F8's census counts them. A body that does not gets a dtype ROUTE at F6's one
                        extension point: bfloat16 calls → F6's XLA arithmetic (counted `trimul_bf16_xla`), float32 calls (the MSA / template blocks without
                        `msa`; every call when this lever is off) → that body unchanged (trimul=route_<word>); its census then counts only the float32
                        calls it still serves, and with `msa` on no pair block would call it in the design step, so its own gate would fail the run by
                        name (nothing_served) — refused here first, by name (msa_starves_f8): pin F8 off (`F8=stock`) or drop `msa` in that case.
  K1 flashattn          the shared core's provider row K1's word selects (the modes' tier words `fast` / `big`, or a pinned row): its operand dtype is
                        probed once at configure on a bf16 call with a gradient; served → ta=k1_pallas,
                        refused → the `ta` region steps aside BY NAME (ta=aside:k1_pallas_<reason>, the two triangle attentions
                        stay float32) — never silent, never a refusal of the run. K1 off: joltz's own attention body in bfloat16 (logits and softmax in
                        bf16 — upstream's softmax_no_cast) (ta=stock_body).
  P5 memlevers          pf<G> group scan: wrapped (install P5 FIRST — mode big's order; installed after this lever P5 would discard the carry wrapper:
                        detected at trace and refused by name, carry_bypassed); `sub`: its checkpoint wrappers sit on the sub-layer classes this lever calls
                        through — composes; tri<C> chunking: composes (chunks see bf16 rows).
  P6 precision          orthogonal (matmul precision words apply to float32 dots; a `trunk=` word still governs the float32 sequence track and MSA module).
  E1 / E10 / P1-P3      untouched classes.
Numerics class `fast` by construction: bf16 operand rounding — not bitwise with stock.

    from mosaic.fast import halfpair
    halfpair.install(); rec = halfpair.configure(None)      # = SETTING, BEFORE the loss is traced; "stock" = every block on joltz's own body (off)
    ...                                                     # words: pf | tm+ta+tz+carry+msa (any '+' subset) | stock
    print(halfpair.emit_line(tag)); halfpair.gate()         # after the run: the census line; the fail-closed gate (on and no pair block traced)

`configure` clears jax's and equinox's trace caches (joltz's filter_jit'd trunk and mosaic's jitted loss step keep traced bodies otherwise) and returns
`describe()`. ENV_REQUIRED is empty: nothing of this lever lives in the environment.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, Optional

LEVER = "P7"
NAME = "halfpair"
MODULE = "mosaic.fast.halfpair"
TOKENS = ("tm", "ta", "tz", "carry", "msa")             # region tokens a spec joins with '+'
ALIASES = {"pf": ("tm", "ta", "tz")}                    # pf = the three pair sub-layer regions of a block
SETTING = "pf"                                          # the default setting: configure(None) applies it
SPEC_WORDS = "'stock' | 'pf' | any '+'-joined subset of tm ta tz carry msa (pf = tm+ta+tz), e.g. 'pf+carry'"
ENV_REQUIRED: Dict[str, str] = {}                       # nothing must be in the environment before jax import
KLASS = "fast"                                          # bf16 operand rounding: not bitwise with stock
DTYPE = "bf16"                                          # the one operand dtype this lever names (jnp.bfloat16)
LAYOUT_MODULE = "mosaic.fast.trimul_layout"             # F6 (its extension point set_body, its XLA arithmetic trimul_cmajor)
FUSED_MODULE = "mosaic.fast.trimul_fused"               # F8 (whether it holds F6's body)
ATTN_MODULE = "mosaic.fast.flashattn"                   # K1 (which op serves the triangle attention)

STATE: Dict[str, Any] = {"installed": False, "on": False, "spec": None, "regions": (), "ta_path": "none", "trimul": "none", "route": None}
CENSUS: "collections.Counter[str]" = collections.Counter()   # traced block calls: "pf2:N<n>:<dtype>" / "noseq:N<n>:<dtype>" / "trimul_bf16_xla" / "carry:N<n>"
_ORIG: Dict[str, Any] = {}


class Refusal(RuntimeError):
    """A named refusal (`.reason` is the word)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# --------------------------------------------------------------------------------------------------------------------------- the grammar
def parse(spec: Optional[str]) -> Dict[str, Any]:
    """`None` / '' → SETTING; 'stock' / 'off' → off; '+'-joined TOKENS / ALIASES → on with that region set (canonical order). Anything else refuses
    `unknown_spec` naming the words."""
    word = (spec or "").strip().lower() or SETTING
    if word in ("stock", "off"):
        return {"on": False, "spec": "stock", "regions": ()}
    got = set()
    for tok in word.split("+"):
        tok = tok.strip()
        if tok in ALIASES:
            got.update(ALIASES[tok])
        elif tok in TOKENS:
            got.add(tok)
        else:
            raise Refusal("unknown_spec", f"{spec!r}: token {tok!r}; words: {SPEC_WORDS}")
    regions = tuple(t for t in TOKENS if t in got)
    if not any(t in regions for t in ("tm", "ta", "tz")):
        raise Refusal("unknown_spec", f"{spec!r}: names no pair sub-layer region (tm, ta, tz or pf); `carry` / `msa` qualify a region set")
    return {"on": True, "spec": canonical(regions), "regions": regions}


def canonical(regions) -> str:
    r = tuple(t for t in TOKENS if t in set(regions))
    if not r:
        return "stock"
    head = ["pf"] if all(t in r for t in ALIASES["pf"]) else [t for t in ("tm", "ta", "tz") if t in r]
    return "+".join(head + [t for t in ("carry", "msa") if t in r])


# --------------------------------------------------------------------------------------------------------------------------- the arithmetic
def _half():
    import jax.numpy as jnp
    return jnp.bfloat16


def _cast_module(mod, dtype):
    """`mod` (an equinox module: a joltz sub-layer, possibly with another lever's __call__ bound on its class) with every floating array leaf cast to dtype."""
    import equinox as eqx
    import jax
    params, static = eqx.partition(mod, eqx.is_inexact_array)
    return eqx.combine(jax.tree_util.tree_map(lambda a: a.astype(dtype), params), static)


def _cast_arg(a, dtype):
    import jax.numpy as jnp
    if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating):
        return a.astype(dtype)
    return a


def _sub(mod, on: bool, z32, *args, **kwargs):
    """One pair sub-layer applied to the float32 pair activation: in bfloat16 (module, activation and floating arguments cast; result back to float32)
    when its region is on, else exactly as the block would (float32)."""
    import jax.numpy as jnp
    if not on:
        return mod(z32, *args, **kwargs)
    h = _half()
    out = _cast_module(mod, h)(z32.astype(h), *[_cast_arg(a, h) for a in args], **{k: _cast_arg(v, h) for k, v in kwargs.items()})
    return out.astype(jnp.float32)


TZ_SLOT: Dict[str, Any] = {"fn": None, "word": "none"}   # the ONE extension point of the transition line: F9 (`mosaic.fast.transition_fused`, route `zres`) sets a
                                                        # body fn(module_bf16, z32) -> z32 + f32(bf16(transition)) (casts + residual inside its kernels), or NotImplemented


def set_tz_body(fn, word: Optional[str] = None):
    """F9's door: `fn` serves the pair transition LINE of a `tz` block (see TZ_SLOT); None releases it (this lever's own line again)."""
    TZ_SLOT["fn"] = fn
    TZ_SLOT["word"] = (word or getattr(fn, "__module__", "set")) if fn is not None else "none"


def _tz_line(mod, on: bool, z32):
    """`z + transition_z(z)` of a block: under `tz` with a body set, the body's fused line (counted `tz_body`); else this lever's own `z32 + _sub(...)`."""
    fn = TZ_SLOT["fn"]
    if on and fn is not None:
        out = fn(_cast_module(mod, _half()), z32)
        if out is not NotImplemented:
            CENSUS["tz_body"] += 1
            return out
    return z32 + _sub(mod, on, z32)


def _regions():
    r = STATE["regions"]
    ta_on = "ta" in r and not str(STATE["ta_path"]).startswith("aside")
    return "tm" in r, ta_on, "tz" in r, "carry" in r, "msa" in r


def _pairformer_layer2_call(self, s, z, mask, pair_mask, *, deterministic: bool = False, key):
    """`joltz.PairformerLayer2.__call__` with the lever: joltz's own body when off; the pair sub-layers of the spec's regions in bfloat16 when on (module
    docstring). The sequence track is joltz's own lines on the float32 pair activation."""
    if not STATE["on"]:
        return _ORIG["pf2_layer"](self, s, z, mask, pair_mask, deterministic=deterministic, key=key)
    import jax.numpy as jnp
    import joltz
    tm, ta, tz, carry, _ = _regions()
    zin = z.dtype
    if carry and zin != _half():
        raise Refusal("carry_bypassed", f"`carry` is on and the block received a {zin} pair activation: joltz.Pairformer2.__call__ was rebound after "
                                        f"{LEVER}'s install (install {LEVER} after P5 — the big row's order)")
    CENSUS[f"pf2:N{z.shape[-2]}:{jnp.dtype(zin).name}"] += 1
    z = z.astype(jnp.float32)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_mul_out, tm, z, pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_mul_in, tm, z, pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_att_start, ta, z, pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key, columnwise=True)
    z = z + dropout * _sub(self.tri_att_end, ta, z, pair_mask)
    z = _tz_line(self.transition_z, tz, z)
    s_normed = self.pre_norm_s(s)
    s = s + self.attention(s=s_normed, z=z, mask=mask, k_in=s_normed)
    s = s + self.transition_s(s)
    s = self.s_post_norm(s)
    return s, z.astype(zin), key


def _noseq_layer_call(self, z, pair_mask, *, key, deterministic=False):
    """`joltz.PairformerNoSeqLayer.__call__` (the MSA / template modules' pair-only block) with the lever: bfloat16 sub-layers under `msa`, else joltz's own."""
    tm, ta, tz, _, msa = _regions()
    if not (STATE["on"] and msa):
        return _ORIG["noseq_layer"](self, z, pair_mask, key=key, deterministic=deterministic)
    import jax.numpy as jnp
    import joltz
    zin = z.dtype
    CENSUS[f"noseq:N{z.shape[-2]}:{jnp.dtype(zin).name}"] += 1
    z = z.astype(jnp.float32)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_mul_out, tm, z, mask=pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_mul_in, tm, z, mask=pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_att_start, ta, z, mask=pair_mask)
    dropout, key = joltz.get_dropout_mask(self.dropout, z, not deterministic, key=key)
    z = z + dropout * _sub(self.tri_att_end, ta, z, mask=pair_mask)
    z = _tz_line(self.transition_z, tz, z)
    return z.astype(zin)


def _pairformer2_call(self, s, z, mask, pair_mask, *, key, deterministic=False):
    """`joltz.Pairformer2.__call__` wrapped: the call bound at install (joltz's scan or P5's grouped scan), with the pair activation carried in bfloat16
    across the blocks under `carry` (cast before, float32 after)."""
    inner = _ORIG["pf2"]
    _, _, _, carry, _ = _regions()
    if not (STATE["on"] and carry):
        return inner(self, s, z, mask, pair_mask, key=key, deterministic=deterministic)
    zin = z.dtype
    CENSUS[f"carry:N{z.shape[-2]}"] += 1
    s, z = inner(self, s, z.astype(_half()), mask, pair_mask, key=key, deterministic=deterministic)
    return s, z.astype(zin)


# --------------------------------------------------------------------------------------------------------------------------- companions (F6/F8, K1)
def _module(name):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _engage_trimul_route():
    """When another lever's body holds F6's served call (F8): no route if that body declares bfloat16 in `__serves_dtypes__`; otherwise route by
    dtype at F6's extension point (bf16 → F6's XLA arithmetic, float32 → that body). Returns the `trimul` word for describe()."""
    TL = _module(LAYOUT_MODULE)
    if TL is None or not TL.installed() or not TL.STATE.get("on"):
        return "stock_body"                                             # F6 off: joltz's einsum bodies, dtype-generic
    prev_body, prev_word = TL.STATE.get("body"), TL.STATE.get("body_word", "xla")
    if prev_body is None:
        return "f6_xla"                                                 # F6's own trimul_cmajor: dtype-generic
    if "bfloat16" in tuple(getattr(prev_body, "__serves_dtypes__", ("float32",))):
        return f"f8_{prev_word}"                                        # the body serves bfloat16 calls itself (F8's op does): no route, its own census counts them
    import jax.numpy as jnp

    def routed(module, x, mask, direction):
        if x.dtype != jnp.float32:
            CENSUS["trimul_bf16_xla"] += 1
            return TL.trimul_cmajor(module, x, mask, direction)
        return prev_body(module, x, mask, direction)
    routed.__halfpair_route__ = True
    STATE["route"] = (prev_body, prev_word)
    TL.set_body(routed, word=f"p7route:{prev_word}")
    return f"route_{prev_word}"


def _release_trimul_route():
    r = STATE.get("route")
    if not r:
        return
    TL = _module(LAYOUT_MODULE)
    STATE["route"] = None
    if TL is None or not TL.installed():
        return
    cur = TL.STATE.get("body")
    if getattr(cur, "__halfpair_route__", False):                       # only undo our own routing (F8 may have reset the body since)
        TL.set_body(r[0], word=r[1])


def _resolve_ta_path(regions) -> str:
    """Which body serves a bf16 triangle attention here: K1's op (the shared core's provider row its word selects) if it serves bf16 operands (probed
    once, by a traced call with a gradient at a served-class shape), else `aside:k1_pallas_<reason>` (the region stays float32); K1 off → joltz's own body."""
    if "ta" not in regions:
        return "off"
    FA = _module(ATTN_MODULE)
    if FA is None or not FA._ORIG or FA.STATE.get("op") is None:
        return "stock_body"
    try:                                                                # probe: does K1's op lower a bf16 call with a backward?
        import jax
        import jax.numpy as jnp
        op = FA.STATE["op"]
        native = getattr(op, "layout", "kernel") == "native"
        B, H, S, D = 2, 4, 32, 32
        shp = (B, S, H, D) if native else (B, H, S, D)
        q = jnp.ones(shp, jnp.bfloat16); bias = jnp.zeros((H, S, S), jnp.bfloat16); km = jnp.ones((B, S), bool)
        jax.grad(lambda q_: op(q_, q, q, bias, km, 0.1).astype(jnp.float32).sum())(q).block_until_ready()
        return "k1_pallas"
    except Exception as e:  # noqa: BLE001  the kernel's own refusal / lowering error, named: the region steps aside
        why = getattr(e, "reason", None) or getattr(e, "kind", None) or type(e).__name__
        return "aside:k1_pallas_" + "_".join(str(why).split())[:40]


def _check_msa_vs_f8(regions):
    if "msa" not in regions:
        return
    TF = _module(FUSED_MODULE)
    if TF is not None and TF.STATE.get("installed") and TF.STATE.get("on") and "bfloat16" not in (TF.serves_dtypes() if callable(getattr(TF, "serves_dtypes", None)) else ()):     # a float32-only F8 body: the template blocks (skipped under E1) would be its only float32 callers left
        raise Refusal("msa_starves_f8", "with `msa` on, no pair block hands F8's float32 kernels a call in the design step and F8's own gate would fail the run "
                                        "(nothing_served): pin F8 off on this row (F8=stock) or drop `msa`")


# --------------------------------------------------------------------------------------------------------------------------- the lever API
def _clear_trace_caches():
    try:
        import jax
        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass
    try:
        import equinox as eqx
        eqx.clear_caches()
    except Exception:  # noqa: BLE001
        pass


def install():
    """Rebind the three joltz attributes (once per process; a second install() raises). Nothing changes numerically until configure() turns a region on."""
    if _ORIG:
        raise RuntimeError(f"{NAME}.install() called twice in one process (uninstall() first)")
    import joltz
    _ORIG["pf2_layer"] = joltz.PairformerLayer2.__call__
    _ORIG["noseq_layer"] = joltz.PairformerNoSeqLayer.__call__
    _ORIG["pf2"] = joltz.Pairformer2.__call__                           # joltz's own scan, or P5's grouped scan when P5 installed first
    joltz.PairformerLayer2.__call__ = _pairformer_layer2_call
    joltz.PairformerNoSeqLayer.__call__ = _noseq_layer_call
    joltz.Pairformer2.__call__ = _pairformer2_call
    STATE["installed"] = True
    return sorted(_ORIG)


def installed() -> bool:
    return bool(_ORIG)


def uninstall():
    """joltz's attributes restored (the ones install() found), the F6 route released, the configuration back to stock, trace caches cleared."""
    _release_trimul_route()
    if _ORIG:
        import joltz
        joltz.PairformerLayer2.__call__ = _ORIG.pop("pf2_layer")
        joltz.PairformerNoSeqLayer.__call__ = _ORIG.pop("noseq_layer")
        joltz.Pairformer2.__call__ = _ORIG.pop("pf2")
    STATE.update(installed=False, on=False, spec=None, regions=(), ta_path="none", trimul="none", route=None)
    CENSUS.clear()
    _clear_trace_caches()


def configure(spec: Optional[str] = None) -> Dict[str, Any]:
    """Set the region set from a spec (parse), resolve the companions (F6/F8 route, K1 path) BEFORE any trace, start a fresh census, clear the trace
    caches, return describe(). Refuses by name: unknown_spec, not_installed, msa_starves_f8."""
    cfg = parse(spec)
    if not installed():
        if cfg["on"]:
            raise Refusal("not_installed", f"configure({spec!r}) before install()")
        STATE.update(on=False, spec="stock", regions=())
        return describe()
    _release_trimul_route()
    CENSUS.clear()
    if not cfg["on"]:
        STATE.update(on=False, spec="stock", regions=(), ta_path="none", trimul="none")
        _clear_trace_caches()
        return describe()
    _check_msa_vs_f8(cfg["regions"])
    ta_path = _resolve_ta_path(cfg["regions"])
    trimul = _engage_trimul_route() if "tm" in cfg["regions"] else "off"
    STATE.update(on=True, spec=cfg["spec"], regions=cfg["regions"], ta_path=ta_path, trimul=trimul)
    _clear_trace_caches()
    return describe()


def _token(v) -> str:
    s = "_".join(str(v).split())
    return s if s else "none"


def describe() -> Dict[str, Any]:
    """The FLAT record (single-token scalars): spec, on, per region on|off (ta may read aside:…), dtype/accumulate/residual words, the carry dtype, which
    body serves the bf16 triangle multiplication / attention, blocks traced so far."""
    r = set(STATE["regions"]) if STATE["on"] else set()
    ta_word = "off" if "ta" not in r else ("on" if not str(STATE["ta_path"]).startswith("aside") else _token(STATE["ta_path"]))
    return {
        "spec": STATE["spec"] if STATE["spec"] else ("stock" if installed() else "none"),
        "on": bool(STATE["on"]),
        "tm": "on" if "tm" in r else "off", "ta": ta_word, "tz": "on" if "tz" in r else "off",
        "carry": DTYPE if "carry" in r else "f32", "msa": "on" if "msa" in r else "off",
        "dtype": DTYPE if r else "f32", "accumulate": "f32", "residual": "f32", "seq_track": "f32", "layernorm_stats": "f32",
        "ta_path": _token(STATE["ta_path"]), "trimul": _token(STATE["trimul"]),
        "blocks_traced": int(sum(v for k, v in CENSUS.items() if k.startswith(("pf2:", "noseq:")))),
        "trimul_bf16_xla": int(CENSUS.get("trimul_bf16_xla", 0)),
        "tz_body": _token(TZ_SLOT["word"]), "tz_body_calls": int(CENSUS.get("tz_body", 0)),
        "upstream": "boltz2_bf16-mixed_trunk",
        "impl": MODULE, "origin": "kit",
    }


def census() -> Dict[str, int]:
    return dict(CENSUS)


def emit_line(tag: str = "") -> str:
    """`[mosaic-opt] HALFPAIR tag=<t> spec=<s> blocks=<pf2:N…:dtype=count+…> trimul_bf16_xla=<n> ta_path=<w> trimul=<w>` — the census line, printed and returned."""
    blocks = "+".join(f"{k}={v}" for k, v in sorted(CENSUS.items()) if k.startswith(("pf2:", "noseq:", "carry:"))) or "none"
    line = (f"[mosaic-opt] HALFPAIR tag={_token(tag) if tag else 'none'} lever={LEVER} spec={STATE['spec'] or 'stock'} blocks={blocks} "
            f"trimul_bf16_xla={CENSUS.get('trimul_bf16_xla', 0)} ta_path={_token(STATE['ta_path'])} trimul={_token(STATE['trimul'])} "
            f"tz_body={_token(TZ_SLOT['word'])}:{CENSUS.get('tz_body', 0)}")
    print(line, flush=True)
    return line


def gate():
    """Fail-closed: configured on and no pair block traced through this lever → `nothing_served` (the run would carry the lever's name on stock's blocks)."""
    if not STATE["on"]:
        return
    if describe()["blocks_traced"] == 0:
        raise Refusal("nothing_served", f"{LEVER} configured on ({STATE['spec']}) and no pairformer block was traced through it")


PATCHED = ("joltz.PairformerLayer2.__call__", "joltz.PairformerNoSeqLayer.__call__", "joltz.Pairformer2.__call__")
