"""TRIATT_XLA — the pair stack's triangle attention CORE (``modules.GridSelfAttention``: the trunk pairformer's 48 layers per recycle, the MSA
and template stacks, the confidence head's pairformer) served by the shared core's JAX-family provider ``opt_core.kernels.pallas``
(``serve.attention(q, k, v, bias, key_mask, word=<the mode's TIER word>, kind="tri" | "tmpl")``): the word is the mode's tier — ``fast``
under ``--mode fast``, ``big`` under ``--mode big`` — and the provider's cell for this head count / head dim / size / compute capability /
jax version names the row that serves (``PALLAS_CELLS.json`` cells ``attn:tri_h4_d32`` / ``attn:tmpl_h4_d16``). One of those rows, the bridge
row ``triattn_xla`` (BRIDGE_ROW), is ``opt_core.kernels.triattn_xla``: pre-compiled triangle-attention forward kernels (bf16 tensor-core
products, fp32 online softmax; several builds per GPU architecture, that package's own per-call choice among them); the Pallas rows, the
library rows ``tokamax`` / ``cudnn`` and the stock statement ``xla`` follow in the cell's order. This module names no row, no card and no
size. Only the attention CORE changes hands: softmax_k(q·k / sqrt(d) + pair bias, padded keys excluded) · v. It is bound at the two places
the core runs —

  (a) inside the FlashPairformer add-on's fused block (``af3_flashpairformer.patch.FlashGridSelfAttention``, modes fast / big at the sizes
      the fused Pallas prologue / epilogue serve): the prologue's q / k / v ([N, S, H·D] in the module's own layout), its pair bias [H, S, S]
      and the key mask go to the provider in place of the add-on's Pallas flash-attention kernel; prologue and epilogue stay;
  (b) in the stock module's ``_attention`` (every other call: the add-on's fallback shapes, the memory mode's sizes where the add-on is
      disengaged, the row-sharded program whose transcription calls ``self._attention`` on the row block): the stock LayerNorm /
      projections / gating / output projection lines with ``tokamax.dot_product_attention`` replaced by the same provider call —

by rebinding ``modules.GridSelfAttention`` to a subclass (``XlaGridSelfAttention``, ``__wrapped__`` = the stock class: the KERNELS census
words it ``stock+w:XlaGridSelfAttention``) BEFORE the add-on builds its classes on top of whatever ``modules.GridSelfAttention`` is; the
fused class inherits ``triatt_xla_core`` from it. The bridge row, when the provider's word selects it, is launched through the bridge
package's own face in the layout the tensors already have (``layout="BNSHD"``: no transposes); every other row through ``serve.attention``.
A call the provider does not serve (a dtype other than bf16, a rank the faces do not take, a word refused by name, a transform the faces do
not cover) STEPS ASIDE BY NAME: the arithmetic of that call site runs (the add-on's Pallas kernel in (a), tokamax in (b)) and the reason is
counted (``report()["aside"]``, the SERVED line's ``txla_aside=``), never silently and never a refusal of the run; inside the provider a row
that cannot engage (switched off, a size / head dim outside its binaries) steps aside by name to the cell's next row.

Numerics: NOT bitwise with either stock or the add-on's Pallas kernel (bf16 tensor-core products, fp32 online softmax and bias, another
accumulation order; the same tolerance class as the add-on's Pallas kernel) — a fast-class lever; deterministic run to run. ``--mode
exact`` does not name it: the provider's ``exact`` word at every attention cell is the stock statement by name.
Switch: ``AF3_JAX_TRIATT_XLA`` = the provider word (the mode table sets the mode's tier word; ``1`` = ``fast``; a row / arm word such as
``triattn_xla`` or ``rowshared@r2`` names exactly that row — an ablation affordance); ``MODEL_OPT_LEVERS_OFF=TRIATT_XLA`` removes the lever
(nothing is rebound); ``MODEL_OPT_LEVERS_OFF=triattn_xla:<kernel>`` (one of the bridge package's own kernel words) passes the kit's
ablation gate by name (ablation.py validate_row_words), reaches the model process and switches that kernel off inside the bridge — its
next kernel or the cell's next row serves, ``txla_rows=`` names it — worded ``rows_off=<words>`` on the ACTIVE / DONE lines;
``pallas:<row>`` switches a provider row off the same way.
Prints: the launcher's SERVED line carries ``txla=<served>|off txla_rows=<row:n,…|none> txla_aside=<reason:n,…|none>`` from ``report()``
(installed / word / traced (trace-time served calls) / rows / sites (fused | module) / aside); the wrapper reads it (modes.lever_evidence).
"""
import os

ENV_SWITCH = "AF3_JAX_TRIATT_XLA"
DEFAULT_WORD = "fast"                                                     # the provider word for switch value "1"; the mode table passes the mode's tier word (modes.TIER_WORD: fast -> fast, big -> big)
TIER_WORDS = ("fast", "exact", "big")                                    # opt_core.kernels.pallas TIER_WORDS (repeated here so the wrapper can read this module without the model stack)
BRIDGE_ROW = "triattn_xla"                                                # the provider's row served by opt_core.kernels.triattn_xla: launched through that package's own face in LAYOUT (the tensors' layout) when the word selects it
FACE = "attention"                                                        # the provider's serve face (serve.COUNTS keys 'served:<face>:<arm>')
REBINDS = ("alphafold3.model.network.modules:GridSelfAttention",)   # what install() rebinds (module:attribute) — a subclass; the FlashPairformer add-on builds FlashGridSelfAttention on top of whatever this name is at ITS import, hence BEFORE the add-on; the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
LAYOUT = "BNSHD"                                                          # [rows, seq, heads, dim]: what the module's projections and the add-on's prologue emit — no transpose (the provider's "BSHD")
_STATE = {"installed": False, "traced": 0, "rows": {}, "sites": {}, "aside": {}, "stock": None, "cls": None, "word": None}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") not in ("", "0")


def word(environ=os.environ) -> str:
    """The provider word this process names: switch value ``1`` = DEFAULT_WORD; any other non-empty value is passed to the provider verbatim."""
    v = (environ.get(ENV_SWITCH, "") or "").strip()
    return DEFAULT_WORD if v in ("", "1") else v


def _word(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-@") else "_" for ch in str(text))[:60].strip("_") or "unknown"


def _aside(reason: str) -> None:
    key = _word(reason)
    _STATE["aside"][key] = _STATE["aside"].get(key, 0) + 1


def refusal_word(exc) -> str:
    """One short word for a refusal: ``<row>:<kind>`` (the provider's Refusal), the bridge's first reason, or the exception's class."""
    row, kind = getattr(exc, "row", None), getattr(exc, "kind", None)
    if kind:
        return f"{row or 'none'}:{kind}"
    reasons = getattr(exc, "reasons", None) or {}
    if reasons:
        row, why = next(iter(reasons.items()))
        return f"{row}:{_word(str(why).split('(')[0].strip())[:40]}"
    return type(exc).__name__


def attention_core(q, k, v, bias, mask, *, site: str):
    """q, k, v [N, S, H, D] (rows, seq, heads, dim — no batch axis) bf16; bias [H, S, S] or [1, H, S, S]; mask [N, S] bool (True = attend).
    Returns softmax_k(q·k/√D + bias)·v in q's shape and dtype from the row the provider's word names, or None when the provider steps aside for
    this call (counted)."""
    import jax.numpy as jnp
    from opt_core.kernels import pallas as P
    from opt_core.kernels.pallas import Refusal
    from opt_core.kernels.pallas import serve as PS
    if q.dtype != jnp.bfloat16:
        _aside(f"dtype:{q.dtype}"); return None
    if q.ndim != 4 or bias.ndim not in (3, 4) or mask is None or mask.ndim != 2:
        _aside(f"rank:q{q.ndim}b{bias.ndim}"); return None
    N, S, H, D = (int(x) for x in q.shape)
    w = _STATE["word"] or word()
    kind = "tmpl" if H * D <= 64 else "tri"                               # the provider's attention kinds: the template pair stack (c 64: 4 heads x 16) | the pair stack (c 128: 4 x 32)
    bias3 = bias if bias.ndim == 3 else bias.reshape(bias.shape[-3:])
    keys = mask > 0 if mask.dtype != jnp.bool_ else mask
    try:                                                                  # the provider's decision for this word / cell (pure Python, before any launch); an unknown word is refused by name here
        sel = PS.resolve("attn", P.family("attn", kind=kind, heads=H, head_dim=D), q.dtype, S, word=w, direction="fwd", key_masked=True)
    except Refusal as e:
        _aside(refusal_word(e)); return None
    arm = None
    out = None
    if sel.row == BRIDGE_ROW and not (sel.config or {}).get("vjp"):      # the bridge's kernels head the word's order here: launched through the bridge's own face in the tensors' layout (serve.attention would transpose to heads-major and back)
        from opt_core.kernels import triattn_xla as TX
        try:
            out, row = TX.triangle_attention(q, k.astype(q.dtype), v.astype(q.dtype), bias3, keys.astype(jnp.uint8), layout=LAYOUT, return_row=True)
            arm = f"{BRIDGE_ROW}.{row}"
            PS.COUNTS[f"served:{FACE}:{BRIDGE_ROW}"] = PS.COUNTS.get(f"served:{FACE}:{BRIDGE_ROW}", 0) + 1
        except TX.Refused:                                                # a size / head dim / switch its binaries do not serve: the provider's walk below records the refusal by name and its next row serves
            out = None
        except NotImplementedError as e:                                  # a transform the bridge's face does not cover: by name
            _aside("trace:" + type(e).__name__); return None
    if out is None:
        before = {k_: v_ for k_, v_ in PS.COUNTS.items() if k_.startswith(f"served:{FACE}:")}
        try:
            out = PS.attention(q, k, v, bias3, keys, None, word=w, kind=kind, direction="fwd", layout="BSHD", selection=sel)
        except Refusal as e:                                              # refused by name (every candidate switched off, a dtype rule, a row word that cannot serve): this call site keeps its own core
            _aside(refusal_word(e)); return None
        except NotImplementedError as e:                                  # a transform the provider's rows do not cover: by name
            _aside("trace:" + type(e).__name__); return None
        served = [k_.split(":", 2)[2] for k_, v_ in PS.COUNTS.items() if k_.startswith(f"served:{FACE}:") and v_ > before.get(k_, 0)]
        arm = served[0] if served else (sel.arm or w)
    _STATE["traced"] += 1
    _STATE["rows"][arm] = _STATE["rows"].get(arm, 0) + 1
    _STATE["sites"][site] = _STATE["sites"].get(site, 0) + 1
    return out.astype(q.dtype)


def _make_class(stock_ga, modules):
    """``GridSelfAttention`` whose ``_attention`` runs the provider's row; every other line is the stock's (alphafold3/model/network/modules.py)."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    hm = modules.hm

    class XlaGridSelfAttention(stock_ga):
        __wrapped__ = stock_ga                                            # the KERNELS census: the stock class owns the site, this wrapper is named (kernels_probe._site_word)

        def triatt_xla_core(self, q, k, v, bias, mask, site="fused"):
            """For the FlashPairformer add-on's fused block (variant a): the attention core between its Pallas prologue and epilogue."""
            return attention_core(q, k, v, bias, mask, site=site)

        @hk.transparent
        def _attention(self, act, mask, bias):
            num_channels = act.shape[-1]
            assert num_channels % self.config.num_head == 0
            qkv_dim = max(num_channels // self.config.num_head, 16)     # stock: Triton requires a minimum dimension of 16 for doing matmul
            qkv_shape = (self.config.num_head, qkv_dim)
            q = hm.Linear(qkv_shape, use_bias=False, name="q_projection", transpose_weights=True)(act)
            k = hm.Linear(qkv_shape, use_bias=False, name="k_projection", transpose_weights=True)(act)
            v = hm.Linear(qkv_shape, use_bias=False, name="v_projection")(act)
            # core: q/k/v [rows, S, H, D] straight from the projections (layout BNSHD, no batch axis); bias = the nonbatched pair bias [H, S, S];
            # mask = the stock key mask [rows, 1, 1, S] as [rows, S] booleans; scale D**-0.5 (the faces' default = tokamax's)
            weighted_avg = None
            if q.ndim == 4 and mask is not None and mask.ndim == 4:
                weighted_avg = attention_core(q, k, v, bias, jnp.reshape(mask, (mask.shape[0], mask.shape[-1])) > 0, site="module")
            else:
                _aside(f"rank:q{q.ndim}")
            if weighted_avg is None:                                      # stepped aside (counted by name): the stock core of this call site
                weighted_avg = modules.tokamax.dot_product_attention(q, k, v, mask=mask, bias=jnp.expand_dims(bias, 0),
                                                                     implementation=self.global_config.flash_attention_implementation)
            weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))
            gate_values = hm.Linear(self.config.num_head * qkv_dim, bias_init=1.0, initializer="zeros", transpose_weights=True, name="gating_query")(act)
            weighted_avg *= jax.nn.sigmoid(gate_values)
            return hm.Linear(num_channels, initializer=self.global_config.final_init, name="output_projection")(weighted_avg)

    XlaGridSelfAttention.__name__ = XlaGridSelfAttention.__qualname__ = "XlaGridSelfAttention"
    return XlaGridSelfAttention


def install() -> bool:
    """Rebind ``modules.GridSelfAttention`` (idempotent); imports the provider and the bridge eagerly so a core without them is named at start."""
    if _STATE["installed"]:
        return True
    import alphafold3.model.network.modules as modules
    from opt_core.kernels.pallas import serve as PS                      # noqa: F401 — the core on the model process's PYTHONPATH (the kit's [tool.opt_core] path), as the add-on's kernels
    from opt_core.kernels import triattn_xla as TX                       # noqa: F401 — the bridge's launcher: a jaxlib it has no build for is named at start
    _STATE["word"] = word()
    _STATE["stock"] = modules.GridSelfAttention
    _STATE["cls"] = _make_class(modules.GridSelfAttention, modules)
    modules.GridSelfAttention = _STATE["cls"]
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        import alphafold3.model.network.modules as modules
        modules.GridSelfAttention = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    core = {}
    try:
        from opt_core.kernels import triattn_xla as TX
        from opt_core.kernels.pallas import serve as PS
        r = TX.report()
        pr = PS.report()
        core = {"version": r.get("version"), "counts": r.get("counts"), "levers_off": r.get("levers_off"), "last_refusal": r.get("last_refusal"),
                "launcher": (r.get("launcher") or {}).get("api") if isinstance(r.get("launcher"), dict) else r.get("launcher"),
                "provider_counts": {k: v for k, v in (pr.get("counts") or {}).items() if f":{FACE}:" in k or k.startswith("refused:")}}
    except Exception as e:  # noqa: BLE001 — the report is words; a core that does not import is named here and is txla=off on the line
        core = {"error": f"{type(e).__name__}: {e}"}
    return {"installed": _STATE["installed"], "word": _STATE["word"] or word(), "traced": _STATE["traced"], "rows": dict(_STATE["rows"]),
            "sites": dict(_STATE["sites"]), "aside": dict(_STATE["aside"]), "core": core}
