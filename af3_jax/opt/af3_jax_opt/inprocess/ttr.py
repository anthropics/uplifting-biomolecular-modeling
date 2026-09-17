"""TTR — the Transition (``modules.TransitionBlock``: LayerNorm -> SwiGLU (``tokamax.gated_linear_unit``) -> output projection) served by the
shared core's JAX-family provider ``opt_core.kernels.pallas`` (``serve.transition(x, params, activation="swiglu", form="af3", word=<tier
word>)``): the provider resolves each call's cell (``PALLAS_CELLS.json``: jax version x compute capability x dtype x family
``af3_swiglu_c<C>_x<factor>`` x token bucket) to the fastest row of the tier the composition names — ``fast`` in mode fast, ``big`` in mode
big (both regions) — and serves it; this module names no row, no card floor and no tile. With the shipped table: the pair transition
c=128 x4 -> ``fpf_transition`` on cc 9.0 / ``mlp_transition`` on cc 8.0 (fused LayerNorm + SwiGLU + output projection in ONE Pallas kernel:
the [N, N, 4C] intermediate never reaches HBM), the template / MSA transitions c=64 -> ``mlp_transition`` / ``fpf_transition``, the single
transitions c=384 -> the provider's stock statement BY NAME (``tokamax_glu``: no Pallas row serves C > 256) — counted as ``routed``. A family
with no cell in the table is the provider's stock statement by name (its census prints the UNCOVERED_CELL token); a call this module does
not hand to the provider (an activation dtype other than bf16) or that the provider refuses by name is a ``fallback`` counted by shape and
reason: that call runs the stock arithmetic (the stock class, or the provider's reference statement on the parameters already read) —
named, never silent. ``--mode exact`` does not name this lever: every transition row is tolerance class, so exact keeps the stock
``TransitionBlock``.

Every ``TransitionBlock`` call site resolves the class at call time, so the rebinding serves the trunk pairformer (48 blocks per recycle),
the MSA stack's transitions, the template pair stack, the confidence head's pairformer and the single transitions.

Switch: ``AF3_JAX_TTR=<tier word>`` in the model process's environment (the mode table sets the composition's tier word,
modes.TIER_WORD_SWITCHES; ``1`` = DEFAULT_WORD; ``MODEL_OPT_LEVERS_OFF=TTR`` removes the lever, ``MODEL_OPT_LEVERS_OFF=pallas[:<row>]``
switches provider rows off by name inside the provider).
Prints: the launcher's SERVED line carries ``ttr=<served by a kernel row>|off ttr_routed=<served by the provider's stock statement by name>
ttr_fallback=<shape:reason:n,...|none>`` from ``report()`` -> {"installed", "word", "fused", "routed", "rows", "fallback", "uncovered"}
(trace-time call sites); the provider's own census line and tokens (``[opt_core] CELLS ...``, UNCOVERED_CELL / NAMED_FALLBACK) name the
cells.
"""
import os

ENV_SWITCH = "AF3_JAX_TTR"
REBINDS = ("alphafold3.model.network.modules:TransitionBlock",)   # what install() rebinds (module:attribute) — the transition class; the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
TIER_WORDS = ("fast", "exact", "big")                             # opt_core.kernels.pallas TIER_WORDS (repeated here so the wrapper can read this module without the model stack)
DEFAULT_WORD = "fast"                                               # switch value "1": the lever's own class (tolerance) = the fast tier's word
FORM, ACTIVATION = "af3", "swiglu"                                  # the provider's family words for this model's module (families af3_swiglu_c<C>_x<factor>)
FACE = "transition"                                                 # the provider's serve face (serve.COUNTS keys 'served:<face>:<arm>')
STOCK_ROWS = ("xla", "tokamax_glu")                                 # the provider's stock statements for this op (opt_core.kernels.pallas STOCK_ROWS restricted to the transition): served there = `routed`
_STATE = {"installed": False, "stock": None, "cls": None, "word": None, "rows": {}, "routed": 0, "fused": 0, "fallback": {}, "uncovered": {}}


def wanted(environ=os.environ) -> bool:
    return (environ.get(ENV_SWITCH, "") or "").strip() not in ("", "0")


def word(environ=os.environ) -> str:
    """The provider word this process names: the switch's value when it is a tier word; ``1`` = DEFAULT_WORD. Anything else is handed to the
    provider verbatim and is its refusal by name (every call falls back, ttr=0)."""
    v = (environ.get(ENV_SWITCH, "") or "").strip()
    return DEFAULT_WORD if v in ("", "1") else v


def _word(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-@") else "_" for ch in str(text))[:60].strip("_") or "unknown"


def _fallback(shape, reason: str, dtype) -> None:
    key = "x".join(str(int(d)) for d in shape) + ":" + _word(reason) + ":" + str(dtype).rsplit(".", 1)[-1]
    _STATE["fallback"][key] = _STATE["fallback"].get(key, 0) + 1


def refusal_word(exc) -> str:
    """One short word for a provider refusal: ``<row>=<kind>`` (opt_core.kernels.pallas.Refusal) or the exception's class."""
    row, kind = getattr(exc, "row", None), getattr(exc, "kind", None)
    if kind:
        return f"{row or 'none'}={kind}"
    return type(exc).__name__


def transition_row(act, params):
    """act [..., C] bf16, params = the provider's math layout (ln_scale, ln_offset [C]; w1 [C, 2F] columns [:F] = a (swish), [F:] = b; w2 [F, C]).
    Returns the module's output in act's dtype from the row the tier word resolves, or None when the call falls back (counted by name)."""
    from opt_core.kernels.pallas import Refusal
    from opt_core.kernels.pallas import serve as PS
    w = _STATE["word"] or word()
    before = {k: v for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:")}
    try:                                                            # the provider walks the tier word's candidates in the cell's order; a row that cannot engage steps aside by name inside it, its stock statement last
        out = PS.transition(act, params, activation=ACTIVATION, form=FORM, word=w, direction="fwd")
    except Refusal as e:                                            # refused by name (an unknown word, a dtype the rows do not serve): this call site keeps the stock module
        _fallback(act.shape, refusal_word(e), act.dtype); return None
    except NotImplementedError as e:                                # a transform the row's kernels do not cover: by name
        _fallback(act.shape, "trace_" + type(e).__name__, act.dtype); return None
    served = [k.split(":", 2)[2] for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:") and v > before.get(k, 0)]
    arm = served[0] if served else w
    _STATE["rows"][arm] = _STATE["rows"].get(arm, 0) + 1
    if arm.split(":")[0].split("@")[0] in STOCK_ROWS:
        _STATE["routed"] += 1
    else:
        _STATE["fused"] += 1
    return out.astype(act.dtype)


def _make_class(stock_tb, modules):
    """``TransitionBlock`` served by the provider's transition face; the stock class where a call falls back."""
    import haiku as hk
    import jax.numpy as jnp
    hm = modules.hm

    class ProviderTransitionBlock(stock_tb):
        __wrapped__ = stock_tb                                      # the KERNELS census: the stock class owns the site, this wrapper is named (kernels_probe._site_word)

        def __call__(self, act, broadcast_dim=0):
            if act.dtype != jnp.bfloat16:                           # decided before any parameter is fetched: the stock class fetches its own
                _fallback(act.shape, f"dtype={act.dtype}", act.dtype)
                return super().__call__(act, broadcast_dim=broadcast_dim)
            c = act.shape[-1]
            f = int(c * self.config.num_intermediate_factor)
            with hk.name_scope("input_layer_norm"):
                s = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones)
                o = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            w1, _ = hm.haiku_linear_get_params(act, num_output=2 * f, initializer="relu", name="transition1")   # [C, 2F]: the stock reshape(C, 2, F) split = columns [:F] a (swish), [F:] b
            with hk.name_scope("transition2"):
                w2 = hk.get_parameter("weights", (f, c), act.dtype, init=jnp.zeros)
            out = transition_row(act, dict(ln_scale=s, ln_offset=o, w1=w1, w2=w2))
            if out is None:                                         # fell back by name after the parameters were read: the stock statement on the same parameters (the provider's reference, this dtype)
                from opt_core.kernels.pallas import serve as PS
                return PS.reference_transition(act, dict(ln_scale=s, ln_offset=o, w1=w1, w2=w2), activation=ACTIVATION).astype(act.dtype)
            return out

    ProviderTransitionBlock.__name__ = ProviderTransitionBlock.__qualname__ = "ProviderTransitionBlock"
    return ProviderTransitionBlock


def install() -> bool:
    """Rebind ``modules.TransitionBlock`` (idempotent); imports the provider eagerly so a core without it is named at start."""
    if _STATE["installed"]:
        return True
    import alphafold3.model.network.modules as modules
    from opt_core.kernels.pallas import serve as PS                # noqa: F401 — the core on the model process's PYTHONPATH (the kit's [tool.opt_core] path)
    _STATE["word"] = word()
    _STATE["stock"] = modules.TransitionBlock
    _STATE["cls"] = _make_class(modules.TransitionBlock, modules)
    modules.TransitionBlock = _STATE["cls"]
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        import alphafold3.model.network.modules as modules
        modules.TransitionBlock = _STATE["stock"]
        _STATE["installed"] = False


def uncovered() -> dict:
    """{family: n} — transition call classes the provider decided WITHOUT a measured cell of their own (its census: outcome inherited), by shape word."""
    out = {}
    try:
        from opt_core import cell_census as C
        for e in C.table():                                         # flat entries: cc / stack / dtype / shape ('<op>:<family>') / bucket / form / word / outcome / served / cell / n
            shape = str(e.get("shape", ""))
            if e.get("outcome") == "inherited" and shape.startswith("transition:"):
                fam = shape.split(":", 1)[1]
                out[fam] = out.get(fam, 0) + int(e.get("n", 1) or 1)
    except Exception:                                               # noqa: BLE001 — the census is the provider's; its own exit line and tokens are the record
        return dict(_STATE["uncovered"])
    return out


def report() -> dict:
    """{"installed", "word", "fused": trace-time calls a kernel row served, "routed": calls the provider's stock statement served by name (C > 256: the
    single transitions; a family without a measured cell), "rows": {arm: n}, "fallback": {"<shape>:<reason>:<dtype>": n} (the stock module ran),
    "uncovered": {family: n}}."""
    if not _STATE["installed"]:
        return {"installed": False, "word": _STATE["word"] or word(), "fused": 0, "routed": 0, "rows": {}, "fallback": {}, "uncovered": {}}
    return {"installed": True, "word": _STATE["word"] or word(), "fused": int(_STATE["fused"]), "routed": int(_STATE["routed"]), "rows": dict(_STATE["rows"]),
            "fallback": dict(_STATE["fallback"]), "uncovered": uncovered()}
