"""Rebinds alphafold3.model.network.modules.{TriangleMultiplication,GridSelfAttention} to Haiku subclasses that read the STOCK parameters
(same names/shapes/dtypes) and call the fused Pallas kernels. All AF3 call sites (trunk pairformer, MSA stack, template stack, confidence head)
resolve these classes through the `modules` global at trace time, so rebinding before the first `hk.transform(...).apply` trace is sufficient."""
import collections
import logging
import os

import haiku as hk
import jax
import jax.numpy as jnp

# The kernels and their tile tables live in the shared core (opt_core.kernels.fpf_pallas = trimul_pallas.py / triattn_pallas.py,
# opt_core.kernels.fpf_pallas_serve = TILE_TABLES / tables_for / served_reason); this module is the AlphaFold 3 Haiku adapter over them.
# The core is reached by path (the kit's [tool.opt_core] pin; the tree's launchers put it on sys.path before this import) — never installed
# into the model's interpreter.
try:
    from opt_core.kernels.fpf_pallas import trimul_pallas as K, triattn_pallas as A
    from opt_core.kernels import fpf_pallas_serve as S
    from opt_core.oom import is_oom
except ImportError as _e:
    raise ImportError(f"af3_flashpairformer: the core's carried kernels are not importable ({_e}); put the pinned core directory "
                      f"(<tree>/common/opt_core, the kit's [tool.opt_core] path) on sys.path before importing this package") from _e

log = logging.getLogger("af3_flashpairformer")
MODES = ("off", "trimul", "triatt", "both")
TILE_TABLES, TILE_TABLES_SOURCE = S.TILE_TABLES, S.TILE_TABLES_SOURCE      # per compute capability; the core's tables


def _compute_capability():
    try:
        d = jax.devices()[0]
        return str(getattr(d, "compute_capability", "") or "")
    except Exception as e:  # pragma: no cover
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return ""


def tables_for(cc=None):
    """(cc, table): the core's tile table for this part. A part without a table is the core's refusal ``no_tiles`` (detail
    ``no-tiles:<cc>``) unless OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1 opts in to the sm_90 table — then status()['tile_table'] and the launcher's
    SERVED line say ``fallback:<cc>->9.0``; never silent."""
    cc = cc or _compute_capability()
    _, t, _own = S.tables_for(cc)
    return cc, t


try:
    CC, _T = tables_for()
    NO_TILES = None
except S.Refusal as _no_tiles:          # this part has no tile table and no OPT_CORE_PALLAS_ALLOW_FALLBACK_CC opt-in: install() refuses by name
    CC, _T, NO_TILES = _compute_capability(), S.TILE_TABLES[S.FALLBACK_CC], _no_tiles
TRIMUL_CFG = dict(_T["trimul"])
TRIMUL_CFG_BY_N = {int(k): dict(v) for k, v in _T.get("trimul_by_n", {}).items()}
ATTN_CFG_DEFAULT = dict(_T["attn_default"])
ATTN_CFG_BY_N = {int(k): dict(v) for k, v in _T["attn_by_n"].items()}

HI = False                                                                # the kernels' f32 hand-off variant (f32_tri / f32_o) is not used: bf16 between kernels
STATE = dict(mode="off", modules=None, stock_tm=None, stock_ga=None, sites=collections.Counter())
KERNEL_KINDS = {"trimul": "trimul", "triatt": "triattn"}                    # the add-on's kernel names -> the core serve layer's table kinds
HELD = {}                                                                 # kernel -> "fallback:no_tiles_cc<NN>(<kind>)": the core REFUSED this part (below cc 8.0) and install() left that class STOCK, by name —
                                                                          #  the tree launcher refuses the run before launch by these words (af3_jax_opt.fpf_launch.refuse_held): a mode is all of its levers


def _served_common(act, mask, kind="trimul", num_head=None):
    """True when the core serves this call AT THIS N (S.served_reason(pad=False) answers None: this model pads its whole input to a bucket
    that is a tile multiple, so the blocks never pad per call here); the refusal's name is the core's."""
    return S.served_reason(kind, act.shape, act.dtype, None if mask is None else mask.shape, num_head=num_head, pad=False) is None   # whole-input scope: the bucket itself is the tile multiple


def make_classes(stock_tm, stock_ga):
    """Build the two patched classes on top of the given stock classes (from any alphafold3 modules module object)."""
    from alphafold3.model.components import haiku_modules as hm

    class FlashTriangleMultiplication(stock_tm):
        """TriangleMultiplication with fused Pallas prologue (LN+GLU+mask -> channel-major planes) / cuBLAS contraction / fused epilogue
        (center-LN + output projection + gating). Same parameters as stock."""

        def __call__(self, act, mask):
            eq = getattr(self.config, "equation", None)
            row = getattr(self, "trimul_cd_row", None)                       # the tree's TRIMUL_CD lever (af3_jax_opt/inprocess/trimul_cd.py) rebound TriangleMultiplication beneath this class:
            if row is not None:                                               #  the whole module served by the core's provider row (opt_core.kernels.pallas); None = the row stepped aside by name
                out = row(act, mask, site="fused")                            #  for this call (counted on the SERVED line's tcd_aside=) and the fused block below runs exactly as it was
                if out is not None:
                    STATE["sites"][("trimul", "fused", tuple(act.shape), str(act.dtype))] += 1   # the add-on's site engaged; tcd= on the SERVED line names the row that computed it
                    return out
            if not (_served_common(act, mask) and eq in ("ikc,jkc->ijc", "kjc,kic->ijc")):
                STATE["sites"][("trimul", "fallback", tuple(act.shape), str(act.dtype))] += 1
                return super().__call__(act, mask)
            STATE["sites"][("trimul", "fused", tuple(act.shape), str(act.dtype))] += 1
            c = act.shape[-1]
            w_proj, _ = hm.haiku_linear_get_params(act, num_output=2 * c, name="projection")
            w_gate, _ = hm.haiku_linear_get_params(act, num_output=2 * c, initializer=self.global_config.final_init, name="gate")
            with hk.name_scope("left_norm_input"):
                s_in = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o_in = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            with hk.name_scope("center_norm"):
                s_c = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o_c = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            w_out, _ = hm.haiku_linear_get_params(act, num_output=c, initializer=self.global_config.final_init, name="output_projection")
            w_gl, _ = hm.haiku_linear_get_params(act, num_output=c, name="gating_linear")
            p = dict(ln_in_scale=s_in, ln_in_offset=o_in, w_proj=w_proj, w_gate=w_gate, ln_c_scale=s_c, ln_c_offset=o_c, w_out=w_out, w_gl=w_gl)
            return K.triangle_multiplication_fused(act, mask, p, equation=eq, cfg=dict(TRIMUL_CFG, **TRIMUL_CFG_BY_N.get(int(act.shape[0]), {}), f32_tri=HI)).astype(act.dtype)

    class FlashGridSelfAttention(stock_ga):
        """GridSelfAttention with fused Pallas prologue (LN + q/k/v/pair-bias projections, transposed reads for the ending-node variant),
        Pallas flash attention forward, fused epilogue (gating + output projection, transposed writes). Same parameters as stock; no row chunking."""

        def __call__(self, act, pair_mask):
            h = self.config.num_head
            c = act.shape[-1]
            d = S.head_dim(c, h)
            if not _served_common(act, pair_mask, "triattn", num_head=h):
                STATE["sites"][("triatt", "fallback", tuple(act.shape), str(act.dtype))] += 1
                return super().__call__(act, pair_mask)
            STATE["sites"][("triatt", "fused", tuple(act.shape), str(act.dtype))] += 1
            n = act.shape[0]; hd = h * d; dt = act.dtype
            with hk.name_scope("act_norm"):
                s = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)

            def w(name, shape):
                with hk.name_scope(name):
                    return hk.get_parameter("weights", shape, dt, init=jnp.zeros)
            hp = {"act_norm": {"scale": s, "offset": o}, "pair_bias_projection": {"weights": w("pair_bias_projection", (c, h))},
                  "q_projection": {"weights": w("q_projection", (h, d, c))}, "k_projection": {"weights": w("k_projection", (h, d, c))},
                  "v_projection": {"weights": w("v_projection", (c, h, d))}, "gating_query": {"weights": w("gating_query", (hd, c))},
                  "output_projection": {"weights": w("output_projection", (hd, c))}}
            kp = A.attn_params_from_haiku(hp)
            cfg = {**ATTN_CFG_DEFAULT, **ATTN_CFG_BY_N.get(n, {}), "f32_o": HI}
            of3 = bool(getattr(self.global_config, "of3_weights", False))     # sokrypton port: pair bias transposed for the ending node
            core = getattr(self, "triatt_xla_core", None)                    # the tree's TRIATT_XLA lever (af3_jax_opt/inprocess/triatt_xla.py) rebound GridSelfAttention beneath this class:
            if core is None:                                                   #  absent = the fused block exactly as it was (one core function, the Pallas flash-attention kernel inside)
                return A.grid_self_attention_fused(act, pair_mask, kp, transpose=self.transpose, ending_bias_transposed=of3, cfg=cfg).astype(dt)
            return fused_with_core(act, pair_mask, kp, core, transpose=self.transpose, ending_bias_transposed=of3, cfg=cfg).astype(dt)

    return FlashTriangleMultiplication, FlashGridSelfAttention


def fused_with_core(act, pair_mask, kp, core, *, transpose, ending_bias_transposed=True, cfg=None):
    """The core's ``grid_self_attention_fused`` (opt_core.kernels.fpf_pallas.triattn_pallas) line for line — Pallas prologue (LayerNorm + q/k/v/pair-bias
    projections, transposed reads for the ending node), attention, Pallas epilogue (gating + output projection) — with the attention between them handed to
    ``core(q, k, v, bias, mask)`` ([N, S, H, D] bf16 in the prologue's own layout, bias [H, S, S], key mask [N, S] booleans): the tree's TRIATT_XLA lever.
    When ``core`` steps aside (returns None: a size or card its kernels do not serve, named in its own report) the add-on's Pallas flash-attention kernel
    serves the call exactly as ``grid_self_attention_fused`` does."""
    N, _, C = act.shape; H, D = kp["H"], kp["D"]
    c = {**A.DEFAULT_ATTN_CFG, **A.ATTN_CFG_BY_N.get(N, {}), **(cfg or {})}
    q, k, v, braw = A.attn_prologue(act, kp["ln_scale"], kp["ln_offset"], kp["wq_t"], kp["wk_t"], kp["wv2"], kp["wb16"], transpose=transpose, t=c["t1"], num_warps=c["w1"])
    bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))                       # (H, N, N) from the un-transposed normalised act, as stock
    if transpose and ending_bias_transposed:
        bias = jnp.swapaxes(bias, -1, -2)
    mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0                            # stock: pair_mask = swapaxes(pair_mask); mask = pair_mask[:, None, None, :]
    q4, k4, v4 = q.reshape(N, N, H, D), k.reshape(N, N, H, D), v.reshape(N, N, H, D)
    o = core(q4, k4, v4, bias, mask2)
    if o is None:
        o = A.flash_attention_bshd(q4, k4, v4, bias[None], mask2, bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"], out_dtype=(A.F32 if c.get("f32_o") else None))
    return A.attn_epilogue(o.reshape(N, N, H * D), act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"], transpose=transpose, t=c["t2"], num_warps=c["w2"])


def install(mode="both", modules=None):
    """mode in off|trimul|triatt|both. `modules`: the alphafold3.model.network.modules module object to patch (default: import it)."""
    mode = (mode or "both").lower()
    if mode not in MODES:
        raise ValueError(f"AF3_FLASHPAIRFORMER mode must be one of {MODES}, got {mode!r}")
    if modules is None:
        import alphafold3.model.network.modules as modules
    if STATE["modules"] is not None and STATE["modules"] is not modules:
        uninstall()
    if STATE["modules"] is None:
        STATE.update(modules=modules, stock_tm=modules.TriangleMultiplication, stock_ga=modules.GridSelfAttention)
    HELD.clear()
    wanted = {"trimul": mode in ("trimul", "both"), "triatt": mode in ("triatt", "both")}
    if NO_TILES is not None:                                              # the core refuses this part (no_tiles: below cc 8.0 — an 8.x/9.x/10.x part without its own table is served the
        cc = (CC or "unknown").replace(".", "")                           #  generation's safe rows and never lands here): each requested kernel is HELD by name, its class stays stock
        for k, want in wanted.items():
            if want:
                HELD[k] = f"fallback:no_tiles_cc{cc}({KERNEL_KINDS[k]})"
    ftm, fga = make_classes(STATE["stock_tm"], STATE["stock_ga"]) if any(want and k not in HELD for k, want in wanted.items()) else (None, None)
    modules.TriangleMultiplication = ftm if (wanted["trimul"] and "trimul" not in HELD) else STATE["stock_tm"]
    modules.GridSelfAttention = fga if (wanted["triatt"] and "triatt" not in HELD) else STATE["stock_ga"]
    STATE["mode"] = mode; STATE["sites"].clear()
    try:
        backend = jax.default_backend()
    except Exception as e:  # pragma: no cover
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        backend = "?"
    log.warning("af3_flashpairformer: mode=%s (TriangleMultiplication=%s, GridSelfAttention=%s), jax %s backend %s", mode,
                modules.TriangleMultiplication.__name__, modules.GridSelfAttention.__name__, jax.__version__, backend)
    if HELD:                                                              # one named line per refused part (nothing prints when the core serves tiles — own or safe)
        log.warning("af3_flashpairformer: held=%s (the core refuses compute capability %s: the stock class serves every call of a held kernel)",
                    ",".join(f"{k}:{w}" for k, w in sorted(HELD.items())), CC or "unknown")
    return mode


def uninstall():
    m = STATE["modules"]
    if m is not None:
        m.TriangleMultiplication = STATE["stock_tm"]; m.GridSelfAttention = STATE["stock_ga"]
    STATE.update(mode="off"); HELD.clear()
    return "off"


def status():
    return dict(mode=STATE["mode"], patched_module=getattr(STATE["modules"], "__name__", None), compute_capability=CC, tile_table=(S.tiles_label(CC) if NO_TILES is None else f"none:{NO_TILES.detail.split()[0]}"),
                trimul_cfg=TRIMUL_CFG, trimul_cfg_by_n=TRIMUL_CFG_BY_N, attn_cfg=ATTN_CFG_DEFAULT, attn_cfg_by_n=ATTN_CFG_BY_N, held=dict(HELD))


def served_report():
    """Counter of (kernel, fused|fallback, act shape, dtype) seen while TRACING (one entry per traced call site, not per executed call)."""
    return dict(STATE["sites"])
