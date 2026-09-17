"""
P5 (opt-in) memory levers for joltz (Boltz-2 in JAX) as used by mosaic — both mathematically exact re-schedulings:
  MEM["triatt_chunk"] = C   : TriangleAttention evaluated over row-chunks of C rows (rows are independent attention problems); each chunk is
                              jax.checkpoint-ed and chunks are visited with lax.map, so at most one chunk's [C, H, N, N] logits (+ its backward) is live
                              instead of the full [N, H, N, N] tensor. The triangle bias (shared across rows) is computed once from the full input, as stock.
                              A call whose row count is <= C runs as one chunk (the stock call) and says so in the trace ledger. Un-chunked calls
                              (no C, or C >= rows) are the call install() found — joltz's own, or a fused attention kernel another lever serves
                              (install that lever FIRST); a chunk over a served attention is REFUSED by name (the kernel tiles rows itself).
  MEM["pf_group"] = G       : every Pairformer2 stack (L stacked blocks: the Boltz-2 trunk's 64, the confidence module's 8) scanned as floor(L/G)
                              groups x G blocks with checkpoint at both levels, then the L mod G remaining blocks one level deep: the backward keeps
                              about L/G + G block-input carries instead of L (two-level rematerialisation); every block is applied exactly as stock, in
                              stock order, same keys. A stack with L <= G runs as one group (the stock scan) and says so in the trace ledger.
  MEM["sub_remat"] = True     : every pair sub-layer call of the trunk and the MSA module (TriangleMultiplicationOutgoing / Incoming, Transition,
                              PairWeightedAveraging, OuterProductMean; TriangleAttention through its own lever above) wrapped in jax.checkpoint: inside a
                              rematerialised block's backward only ONE sub-layer's intermediates ([N, N, 4·c] transition hidden, the triangle
                              projections and gates) are live at a time instead of the whole block's — a third rematerialisation level below pf_group,
                              paid with one more forward of each sub-layer. Spec word 'sub'.
Call memlevers.install() once (before tracing) and memlevers.configure(spec) BEFORE tracing; configure() clears jax/equinox jit caches because joltz
caches inner filter_jit traces (a config change would otherwise be silently ignored inside an already-traced process) and raises when a cache cannot
be cleared. Stock behaviour when both are None. Every trace of a patched call appends one record to the trace ledger (traces(): site, extent, setting, mode) — the run's account of what the
levers did to the executables it built. Numerics: the same arithmetic in a different schedule (chunked GEMM shapes, rematerialised recomputation):
reduction order can differ from the stock executable's, so bitwise identity with stock is not assumed.
"""
from typing import Optional

import jax, jax.numpy as jnp, einops, equinox as eqx
MEM = {"triatt_chunk": None, "pf_group": None, "sub_remat": False}                  # + "fit": True while the `fit` word is in force (absent otherwise)
_ORIG = {}
TRACES = []                                                                          # one record per trace of a patched call: {"site", "n", "setting", "mode"}
STATE = {"requested": None, "configured_at": 0}                                      # the spec configure() was asked for (canonical words) and the trace-ledger index at that moment
SPEC_WORDS = "'stock' | 'tri<C>' | 'pf<G>' | 'sub' | 'fit' | any '+'-joined combination, e.g. 'tri64+pf8+sub', 'pf8+sub+fit' (C, G integers >= 1)"
SETTING = "pf8+sub"                                                                  # the default setting (mode big's P5 word): configure(None) applies it; 'stock' is the explicit off word
ENV_REQUIRED = {}                                                                    # the lever needs no process environment (nothing to export before the JAX backend starts)
FIT_GIB_PER_TOK2, FIT_GIB_CONST, FIT_FRACTION = 4.74e-5, 3.0, 0.90                   # `fit`: a peak-memory model of the un-rescheduled (mode fast) step, GiB = a·N² + b, and the pool fraction it may fill before the levers reschedule
FIT_ASIDE = "aside:fits"                                                             # the ledger mode of a call the `fit` gate left on the stock schedule (named, counted, accepted by the gate)


def _pool_limit_gib():
    """The device pool's limit in GiB (jax memory_stats bytes_limit); None when the backend states none (CPU) — `fit` then never steps aside."""
    try:
        bl = (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit")
        return float(bl) / 2**30 if bl else None
    except Exception:  # noqa: BLE001  a backend without memory statistics
        return None


def _fits(n) -> bool:
    """`fit` in force AND the un-rescheduled step at this call's token count is predicted to fit the pool: the sub-levers step aside for this trace (FIT_ASIDE)."""
    if not MEM.get("fit"):
        return False
    limit = _pool_limit_gib()
    return limit is not None and FIT_GIB_PER_TOK2 * float(n) * float(n) + FIT_GIB_CONST <= FIT_FRACTION * limit


def _trace(site, n, setting, mode):
    TRACES.append({"site": site, "n": int(n), "setting": setting, "mode": mode, "spec": spec_of()})


def _stock_triatt(fn):
    """True when `fn` is joltz's own TriangleAttention.__call__ (not another lever's served call)."""
    return getattr(fn, "__module__", None) == "joltz" and getattr(fn, "__qualname__", "") == "TriangleAttention.__call__"


def _triatt_call(self, x, mask):
    C = MEM["triatt_chunk"]
    I = x.shape[-3] if self.starting else x.shape[-2]                                # the query-row count of this call (ending node: transposed inside)
    owner = _ORIG["triatt"]                                                          # the call install() found: joltz's own, or another lever's served call (a fused kernel)
    if C is None:
        _trace("triangle_attention", I, None, "stock")
        return owner(self, x, mask)                                                 # un-chunked: the owner's call as is
    if _fits(x.shape[-2]):
        _trace("triangle_attention", I, C, FIT_ASIDE)                                # `fit`: the pool holds the un-chunked call at this token count — the owner's call, named
        return owner(self, x, mask)
    if C >= I:
        _trace("triangle_attention", I, C, "single_chunk")                          # the chunk covers every row: the owner's call, named
        return owner(self, x, mask)
    if not _stock_triatt(owner):
        raise ValueError(f"memlevers_refused: triatt_chunk={C} over a served TriangleAttention ({getattr(owner, '__module__', '?')}.{getattr(owner, '__qualname__', '?')}) — "
                         "row chunking re-schedules joltz's own attention body; a fused attention kernel tiles the rows itself: compose pf<G> / sub with it instead")
    if not self.starting:
        x = einops.rearrange(x, "... I J C_in -> ... J I C_in")
        mask = einops.rearrange(mask, "... I J -> ... J I")
    x = self.layer_norm(x)
    mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]                      # [..., I, 1, 1, J]  (row-dependent)
    triangle_bias = einops.rearrange(self.linear(x), "... J I H -> ... 1 H J I")     # [..., 1, H, N, N]  (shared by all rows)
    _trace("triangle_attention", I, C, "chunked")
    lead = x.shape[:-3]; nl = len(lead); J, Cin = x.shape[-2], x.shape[-1]
    n_full, rem = I // C, I % C
    f = jax.checkpoint(lambda xc, mbc: self.mha(q_x=xc, kv_x=xc, biases=[mbc, triangle_bias]))
    xm = x[..., : n_full * C, :, :].reshape(*lead, n_full, C, J, Cin)
    mm = mask_bias[..., : n_full * C, :, :, :].reshape(*lead, n_full, C, 1, 1, mask_bias.shape[-1])
    xm = jnp.moveaxis(xm, nl, 0); mm = jnp.moveaxis(mm, nl, 0)                   # chunk axis first for lax.map
    ym = jax.lax.map(lambda a: f(a[0], a[1]), (xm, mm))                          # [n_full, ..., C, J, Cin]
    ym = jnp.moveaxis(ym, 0, nl).reshape(*lead, n_full * C, J, Cin)
    if rem:
        yr = f(x[..., n_full * C :, :, :], mask_bias[..., n_full * C :, :, :, :])
        x = jnp.concatenate([ym, yr], axis=-3)
    else:
        x = ym
    if not self.starting:
        x = einops.rearrange(x, "... J I C_in -> ... I J C_in")
    return x


def _pairformer2_call(self, s, z, mask, pair_mask, *, key, deterministic=False):
    G = MEM["pf_group"]
    L = jax.tree_util.tree_leaves(self.stacked_parameters)[0].shape[0]

    @jax.checkpoint
    def _body(carry, params):                                                        # joltz.Pairformer2's own scan body: one block, checkpointed
        s, z, key = carry
        return eqx.combine(self.static, params)(s, z, mask, pair_mask, key=key, deterministic=deterministic), None

    if G is not None and L > G and _fits(z.shape[-2]):
        _trace("pairformer", L, G, FIT_ASIDE)                                        # `fit`: the un-grouped scan's residuals fit the pool at this token count — joltz's own schedule, named
        (s, z, key), _ = jax.lax.scan(_body, (s, z, key), self.stacked_parameters)
        return s, z
    if G is None or L <= G:
        _trace("pairformer", L, G, "stock" if G is None else "single_group")         # L <= G: one group = the stock scan, named
        (s, z, key), _ = jax.lax.scan(_body, (s, z, key), self.stacked_parameters)
        return s, z
    n_full, rem = L // G, L % G
    _trace("pairformer", L, G, "grouped" if rem == 0 else "grouped+remainder")
    grouped = jax.tree_util.tree_map(lambda a: a[: n_full * G].reshape(n_full, G, *a.shape[1:]), self.stacked_parameters)

    @jax.checkpoint
    def _group(carry, gparams):                                                      # one group of G blocks: the inner scan, checkpointed as a whole
        return jax.lax.scan(_body, carry, gparams)[0], None

    (s, z, key), _ = jax.lax.scan(_group, (s, z, key), grouped)
    if rem:                                                                          # the last L mod G blocks, stock order, one level deep (per-block checkpoint)
        tail = jax.tree_util.tree_map(lambda a: a[n_full * G :], self.stacked_parameters)
        (s, z, key), _ = jax.lax.scan(_body, (s, z, key), tail)
    return s, z


SUB_SITES = {"trimul_out": "TriangleMultiplicationOutgoing", "trimul_in": "TriangleMultiplicationIncoming", "transition": "Transition",
             "pair_weighted_averaging": "PairWeightedAveraging", "outer_product_mean": "OuterProductMean"}   # site word -> joltz class whose __call__ `sub` rematerialises


def _remat_call(site, orig):
    """joltz's own `orig` __call__, wrapped in jax.checkpoint when MEM['sub_remat'] is on (the module's arrays and the call's array arguments are the
    checkpoint's inputs; keyword arguments ride along); the stock call otherwise. One trace record per trace either way."""
    def call(self, *args, **kwargs):
        n = args[0].shape[-2] if args and hasattr(args[0], "shape") and len(args[0].shape) >= 2 else -1
        if not MEM["sub_remat"]:
            _trace(site, n, None, "stock")
            return orig(self, *args, **kwargs)
        if _fits(n):
            _trace(site, n, "sub", FIT_ASIDE)                                            # `fit`: the block's intermediates fit the pool at this token count — the stock call, named
            return orig(self, *args, **kwargs)
        _trace(site, n, "sub", "remat")
        return eqx.filter_checkpoint(lambda s_, *a: orig(s_, *a, **kwargs))(self, *args)   # filter_: the module carries non-array leaves (activation callables, ints)
    call.__name__ = getattr(orig, "__name__", "__call__"); call.__qualname__ = getattr(orig, "__qualname__", "__call__")
    return call


def install():
    import joltz
    if "installed" in _ORIG: return
    _ORIG["triatt"] = joltz.TriangleAttention.__call__; _ORIG["pf2"] = joltz.Pairformer2.__call__
    joltz.TriangleAttention.__call__ = _triatt_call
    joltz.Pairformer2.__call__ = _pairformer2_call
    for site, cls_name in SUB_SITES.items():
        cls = getattr(joltz, cls_name)
        _ORIG["sub:" + site] = cls.__call__
        cls.__call__ = _remat_call(site, cls.__call__)
    _ORIG["installed"] = True


def installed():
    return "installed" in _ORIG


def parse(spec: Optional[str]):
    """`None` = SETTING (the default setting). {'triatt_chunk': C | None, 'pf_group': G | None} from a spec ('stock', 'tri64', 'pf8', 'tri64+pf8', 'tri32+pf4'); a part that is not one of
    SPEC_WORDS, or a C / G below 1, is refused by name (`memlevers_refused: ...`)."""
    out = {"triatt_chunk": None, "pf_group": None, "sub_remat": False}                 # + "fit": True when the spec carries the word (absent otherwise: the shape every other spec had)
    spec = SETTING if spec is None else spec
    for part in str(spec).split("+"):
        try:
            if part == "sub": out["sub_remat"] = True
            elif part == "fit": out["fit"] = True
            elif part.startswith("tri"): out["triatt_chunk"] = int(part[3:])
            elif part.startswith("pf"): out["pf_group"] = int(part[2:])
            elif part in ("stock", ""): continue
            else: raise ValueError(part)
        except ValueError:
            raise ValueError(f"memlevers_refused: spec part {part!r} of {spec!r} is not one of {SPEC_WORDS}") from None
    bad = {k: v for k, v in out.items() if k in ("triatt_chunk", "pf_group") and v is not None and v < 1}
    if bad:
        raise ValueError(f"memlevers_refused: {bad} in {spec!r}: chunk / group sizes are integers >= 1")
    if out.get("fit") and out["triatt_chunk"] is None and out["pf_group"] is None and not out["sub_remat"]:
        raise ValueError(f"memlevers_refused: spec {spec!r}: 'fit' gates the sub-levers it is joined with (tri<C> / pf<G> / sub) and names none")
    return out


def configure(spec: Optional[str] = None):
    """`None` = SETTING (the default setting). spec examples: 'stock', 'tri64', 'pf8', 'tri64+pf8', 'tri32+pf4' (SPEC_WORDS). Returns the configuration now in force (a copy of MEM)."""
    new = parse(spec)
    MEM.update({"triatt_chunk": None, "pf_group": None, "sub_remat": False}); MEM.pop("fit", None)
    # joltz wraps trunk_iteration / the sampler in eqx.filter_jit and mosaic wraps the loss step: their traced jaxprs are cached per process and
    # would silently keep the previous configuration -> clear all staging/compilation caches whenever the configuration changes; a cache that
    # cannot be cleared is an error (the configuration might not be in force), never swallowed.
    jax.clear_caches()
    eqx.clear_caches()
    MEM.update(new)
    STATE["requested"] = spec_of(new); STATE["configured_at"] = len(TRACES)
    return dict(MEM)


def uninstall():
    """joltz's own TriangleAttention / Pairformer2 calls restored, the configuration back to stock, jit caches cleared (executables built under the
    levers are dropped); a no-op when install() never ran."""
    if "installed" not in _ORIG: return
    import joltz
    joltz.TriangleAttention.__call__ = _ORIG.pop("triatt"); joltz.Pairformer2.__call__ = _ORIG.pop("pf2")
    for site, cls_name in SUB_SITES.items():
        getattr(joltz, cls_name).__call__ = _ORIG.pop("sub:" + site)
    _ORIG.pop("installed")
    MEM.update({"triatt_chunk": None, "pf_group": None, "sub_remat": False}); MEM.pop("fit", None)
    STATE["requested"] = None; STATE["configured_at"] = 0
    jax.clear_caches(); eqx.clear_caches()


def spec_of(config=None):
    """The canonical spec word of a configuration ({'triatt_chunk': C, 'pf_group': G}; MEM when None): 'stock' | 'tri<C>' | 'pf<G>' | 'tri<C>+pf<G>'."""
    c = MEM if config is None else config
    parts = (([f"tri{c['triatt_chunk']}"] if c.get("triatt_chunk") is not None else []) + ([f"pf{c['pf_group']}"] if c.get("pf_group") is not None else [])
             + (["sub"] if c.get("sub_remat") else []) + (["fit"] if c.get("fit") else []))
    return "+".join(parts) or "stock"


PATCHED = ("joltz.TriangleAttention.__call__", "joltz.Pairformer2.__call__") + tuple(f"joltz.{c}.__call__" for c in SUB_SITES.values())   # the attributes install() replaces (restored by uninstall())


def describe():
    """The lever's state in this process, for the LEVER line — every value ONE blank-free token (the installer's rule, mosaic_opt.levers: the line
    splits fields on blanks): installed (0|1), the configuration in force (values + canonical spec; absent = 'none'), patched = the count of rebound
    attributes and patched_sites = their short names '+'-joined, traces = ledger length, trace_census = 'site.mode=count' items ','-joined."""
    census = {}
    for t in TRACES:
        k = f"{t['site']}.{t['mode'].replace('+', '_')}"; census[k] = census.get(k, 0) + 1
    sites = [p.split(".")[1] for p in PATCHED] if installed() else []
    tok = lambda v: "none" if v is None else (int(v) if isinstance(v, bool) else v)
    return {"lever": "P5", "lever_name": "memlevers", "setting": SETTING, "installed": int(installed()), "triatt_chunk": tok(MEM["triatt_chunk"]), "pf_group": tok(MEM["pf_group"]),
            "sub_remat": int(bool(MEM["sub_remat"])), "spec": spec_of(), "requested": STATE["requested"] or "none", "patched": len(sites), "patched_sites": "+".join(sites) or "none",
            "traces": len(TRACES), "trace_census": ",".join(f"{k}={v}" for k, v in sorted(census.items())) or "none",
            **({"fit": 1, "fit_model": f"{FIT_GIB_PER_TOK2:g}xN2+{FIT_GIB_CONST:g}GiB<={FIT_FRACTION:g}xpool", "pool_gib": (f"{_pool_limit_gib():.1f}" if _pool_limit_gib() else "none")} if MEM.get("fit") else {})}


def activate(ctx=None, spec=None):
    """The activation contract (mosaic_opt.levers): install(), configure(spec) (SETTING when None), and hand back
    {'deactivate': uninstall, 'patched': [...], 'describe': {...}}. `ctx` (the caller's context) is not read: the lever patches joltz's classes,
    so a model already built runs the reschedule at its next trace (configure() cleared the caches)."""
    install()
    cfg = configure(None if spec is None else str(spec))
    return {"deactivate": uninstall, "patched": list(PATCHED), "describe": describe(), "config": cfg, "lever_module": __name__, "spec": spec_of(cfg)}


LEVER, NAME = "P5", "memlevers"
IN_FORCE = {"triatt_chunk": ("triangle_attention:chunked", lambda site, mode: site == "triangle_attention" and mode == "chunked"),
            "pf_group": ("pairformer:grouped[+remainder]", lambda site, mode: site == "pairformer" and mode.startswith("grouped")),
            "sub_remat": ("<sub-layer site>:remat", lambda site, mode: site in SUB_SITES and mode == "remat")}   # per sub-lever: the trace words that show it reshaped an executable
SPEC_PART = {"triatt_chunk": "tri", "pf_group": "pf", "sub_remat": "sub"}


class GateRefusal(RuntimeError):
    """The fail-closed gate's named refusal (`memlevers_gate:<reason> ...`)."""


def _census_since_configure():
    census = {}
    for t in TRACES[STATE["configured_at"]:]:
        k = f"{t['site']}:{t['mode']}"; census[k] = census.get(k, 0) + 1
    return census


def emit_line(tag: str = "") -> str:
    """The LEVER line of this process in the core's grammar (opt_core.report.lever_line): the spec in force vs the one configure() was asked
    for, installed, and the trace census since configure() as <site>.<mode>=<count> items; the off line when nothing is configured."""
    from opt_core.report import emit, lever_line
    spec = spec_of()
    if not installed() or spec == "stock":
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason=("not_installed" if not installed() else "not_configured"), impl="mosaic.fast.memlevers", origin="kit"))
    ev = {k.replace(":", ".").replace("+", "_"): v for k, v in sorted(_census_since_configure().items())}
    return emit(lever_line(tag, f"{LEVER}.{NAME}", "on", impl="mosaic.fast.memlevers", origin="kit", spec=spec, requested=STATE["requested"] or "none",
                           installed=int(installed()), traces=len(TRACES) - STATE["configured_at"], **ev))


def gate():
    """Fail closed — GateRefusal `memlevers_gate:<reason>`: install() without configure() (`not_configured`); the spec in force differs from
    the one configure() was asked for (`spec_mismatch`); an executable traced since configure() carries another spec (`stale_spec_traces`); a
    configured sub-lever never reshaped an executable traced since configure() (`<tri|pf|sub>_not_in_force` — a chunk that covered every
    row everywhere, a group no stack was deeper than, a run that never traced the patched sites). Silent when not installed or configured off."""
    if not installed():
        return
    if STATE["requested"] is None:
        raise GateRefusal("memlevers_gate:not_configured — install() ran but configure() never did: the run would carry the lever's name on the stock schedule")
    spec = spec_of()
    if spec != STATE["requested"]:
        raise GateRefusal(f"memlevers_gate:spec_mismatch — in force {spec!r}, configured {STATE['requested']!r}")
    if spec == "stock":
        return
    recent = TRACES[STATE["configured_at"]:]
    stale = sorted({str(t.get("spec")) for t in recent if t.get("spec") != spec})
    if stale:
        raise GateRefusal(f"memlevers_gate:stale_spec_traces — executables traced under {stale} since configure({spec!r})")
    census = _census_since_configure()
    for key, (words, holds) in IN_FORCE.items():
        if MEM[key] in (None, False):
            continue
        if not any(holds(t["site"], t["mode"]) or t["mode"] == FIT_ASIDE for t in recent):   # `fit`: a call left on the stock schedule by the capacity gate is the sub-lever's named decision
            raise GateRefusal(f"memlevers_gate:{SPEC_PART[key]}_not_in_force — configured {spec!r} but no executable traced since configure() carries "
                              f"{words} (ledger since configure: {census if census else 'nothing traced'})")


def traces():
    """The trace ledger: one record per trace of a patched call since install() — what the levers did to every executable built in this process."""
    return [dict(t) for t in TRACES]
