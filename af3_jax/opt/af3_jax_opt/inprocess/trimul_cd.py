"""TRIMUL_CD — the pair stack's triangle multiplication (``modules.TriangleMultiplication``: the trunk pairformer's 48 blocks x (outgoing,
incoming) per recycle, the MSA stack's 4, the template stack's 2 (c=64), the confidence head's pairformer) served WHOLE (input LayerNorm ->
gated left/right projections x mask -> the contraction -> centre LayerNorm -> gated output projection) by the shared core's JAX-family
provider ``opt_core.kernels.pallas`` (``serve.triangle_multiplication(act, mask, params, equation=, form="af3", word=<the mode's TIER
word>)``): the word is the mode's tier — ``fast`` under ``--mode fast``, ``big`` under ``--mode big`` — and the provider's cell for this
form / channel width / equation / size / compute capability / jax version names the row that serves (``PALLAS_CELLS.json``: ``native_xla``,
``cd_trimul``, ``fpf_trimul``, …, the stock statement ``xla`` last); this module names no row, no card and no size. A call whose cell is not
in the table is served by the provider's declared fallback and printed by the provider as an ``UNCOVERED_CELL`` token
(``opt_core.cell_census``); the kit counts those calls on its SERVED line (``tcd_uncovered=<family:n>``).

Bound at the two places the module runs, by rebinding ``modules.TriangleMultiplication`` to a subclass (``CdTriangleMultiplication``,
``__wrapped__`` = the stock class) BEFORE the FlashPairformer add-on derives ``FlashTriangleMultiplication`` from whatever that name is at
ITS import (fpf_launch.TREE_LEVERS: pass 1):
  (a) inside the add-on's fused class: ``FlashTriangleMultiplication.__call__`` calls the inherited ``trimul_cd_row`` first — the provider
      serves the whole module (the add-on's own fused block is one of the provider's rows, ``fpf_trimul``; the site counts as engaged on the
      SERVED line; ``tcd_rows=`` says which row computed it);
  (b) as the module itself wherever the add-on's class is not installed (FPF_TRIMUL ablated: ``AF3_FLASHPAIRFORMER=triatt``): ``__call__`` =
      the provider, else the stock lines.
A call the provider does not serve (an activation dtype other than bf16, a non-square / unmasked call, an equation that is neither update,
a word the provider refuses by name) STEPS ASIDE BY NAME: the arithmetic that call site had runs (the add-on's fused block in (a), the stock
module in (b)) and the reason is counted (``report()["aside"]``, the SERVED line's ``tcd_aside=``) — never silently, never a refusal of the
run. Inside the provider a row that cannot engage (switched off by ``MODEL_OPT_LEVERS_OFF=pallas[:<row>]``, a shape / card rule) steps aside
by name to the cell's next row (``serve.COUNTS`` records it). The memory mode's reach region does not name this lever
(modes.BIG_DISENGAGED: TRIMUL_CHUNK owns the site there); region fast engages it with the mode's word.

Parameters: the STOCK tree (same names / shapes / dtypes the stock class and the add-on read: left_norm_input/{scale,offset},
projection/weights [C,2C], gate/weights [C,2C], center_norm/{scale,offset}, output_projection/weights, gating_linear/weights; AlphaFold 3's
Linear layers carry no bias) handed to the provider in its math layout (left = the even columns of projection / gate, right = the odd ones:
the stock ``reshape(C, 2, ...)`` split).

Numerics: NOT bitwise with stock (bf16 tensor-core products with fp32 accumulation, fp32 LayerNorm statistics, fewer bf16 rounding points)
— a fast-class (tolerance) lever; deterministic run to run (no atomics). ``--mode exact`` does not name it: the provider's ``exact`` word at
every af3 triangle-multiplication cell is the stock statement by name, so exact keeps the stock module.
Switch: ``AF3_JAX_TRIMUL_CD`` = the provider word (the mode table sets the mode's tier word; ``1`` = ``fast``; a row / arm word such as
``cd_trimul`` or ``fpf_trimul@w4`` names exactly that row — an ablation affordance, printed as ``tcd_word=``);
``MODEL_OPT_LEVERS_OFF=TRIMUL_CD`` removes the lever (nothing is rebound).
Prints: the launcher's SERVED line carries ``tcd=<served>|off tcd_rows=<arm:n,...|none> tcd_aside=<reason:n,...|none> tcd_word=<word>
tcd_uncovered=<family:n,...|none>`` from ``report()`` (installed / word / traced (trace-time served calls) / rows (the provider arm that
served, per call) / sites (fused | module) / aside / uncovered); the wrapper reads it (modes.lever_evidence: zero served calls =
levers_short by name).
"""
import os

ENV_SWITCH = "AF3_JAX_TRIMUL_CD"
REBINDS = ("alphafold3.model.network.modules:TriangleMultiplication",)   # what install() rebinds (module:attribute) — a subclass; the FlashPairformer add-on builds FlashTriangleMultiplication on top of whatever this name is at ITS import, hence BEFORE the add-on (fpf_launch.TREE_LEVERS pass 1; tests/test_install_order.py)
DEFAULT_WORD = "fast"                                                     # the provider word for switch value "1"; the mode table passes the mode's tier word (modes.TIER_WORD: fast -> fast, big -> big)
TIER_WORDS = ("fast", "exact", "big")                                    # opt_core.kernels.pallas TIER_WORDS (repeated here so the wrapper can read this module without the model stack)
FORM = "af3"                                                              # the provider's family form word for this model's module (families: af3_pair_c128_ch128_{outgoing,incoming}, af3_tmpl_c64_ch64_{outgoing,incoming})
EQUATIONS = ("ikc,jkc->ijc", "kjc,kic->ijc")                              # outgoing | incoming (modules.TriangleMultiplication.Config.equation)
FACE = "triangle_multiplication"                                          # the provider's serve face (serve.COUNTS keys 'served:<face>:<arm>')
_STATE = {"installed": False, "traced": 0, "rows": {}, "sites": {}, "aside": {}, "uncovered": {}, "stock": None, "cls": None, "word": None}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") not in ("", "0")


def word(environ=os.environ) -> str:
    """The provider word this process names: switch value ``1`` = DEFAULT_WORD; any other non-empty value is passed to the provider verbatim
    (a tier word, a row name, an arm '<row>@<setting>') — an unknown word is the provider's refusal by name (every call steps aside, tcd=0)."""
    v = (environ.get(ENV_SWITCH, "") or "").strip()
    return DEFAULT_WORD if v in ("", "1") else v


def _word(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-@") else "_" for ch in str(text))[:60].strip("_") or "unknown"


def _aside(reason: str) -> None:
    key = _word(reason)
    _STATE["aside"][key] = _STATE["aside"].get(key, 0) + 1


def refusal_word(exc) -> str:
    """One short word for a provider refusal: ``<row>:<kind>`` (opt_core.kernels.pallas.Refusal) or the exception's class."""
    row, kind = getattr(exc, "row", None), getattr(exc, "kind", None)
    if kind:
        return f"{row or 'none'}:{kind}"
    return type(exc).__name__


def early_aside(act, mask, equation):
    """The step-asides decidable from the call alone, BEFORE the caller fetches the module's parameters (a caller that then runs its own arithmetic
    fetches them itself; fetching twice under one module scope is what must not happen): dtype, rank, mask shape, equation.
    Returns the aside reason (the caller counts it through ``note_aside``) or None."""
    import jax.numpy as jnp
    if act.dtype != jnp.bfloat16:
        return f"dtype:{act.dtype}"
    if act.ndim != 3 or int(act.shape[0]) != int(act.shape[1]):
        return f"rank:act{act.ndim}"
    if mask is None or mask.ndim != 2 or tuple(int(x) for x in mask.shape) != (int(act.shape[0]), int(act.shape[1])):
        return "mask:" + ("none" if mask is None else "x".join(str(int(x)) for x in mask.shape))
    if equation not in EQUATIONS:
        return f"equation:{equation}"
    return None


def note_aside(reason: str) -> None:
    _aside(reason)


def trimul_row(act, mask, params, equation, *, site: str):
    """act [N, N, C] bf16 pair activations, mask [N, N], params = the provider's math layout (ln_in_*, left_w / right_w, left_gate_w / right_gate_w,
    ln_c_*, out_w, gate_w). Returns the module's output [N, N, C] in act's dtype from the row the provider's word names for this call, or None when
    the provider steps aside for this call (counted by name)."""
    from opt_core.kernels import pallas as P
    from opt_core.kernels.pallas import Refusal
    from opt_core.kernels.pallas import serve as PS
    why = early_aside(act, mask, equation)
    if why is not None:
        _aside(why); return None
    w = _STATE["word"] or word()
    C, Ch = int(act.shape[-1]), int(params["left_w"].shape[1])
    fam = P.family("trimul", form=FORM, unit=("tmpl" if C <= 64 else "pair"), c=C, c_hidden=Ch, equation=equation)   # the provider's own family word for this call (serve.triangle_multiplication derives the same)
    try:                                                                  # the provider's decision for this word / cell — pure Python, before any launch; an unknown word or a row word that cannot serve is refused by name here
        selection = PS.resolve("trimul", fam, act.dtype, int(act.shape[0]), word=w, direction="fwd")
    except Refusal as e:
        _aside(refusal_word(e)); return None
    if getattr(selection, "cell_key", None) is None:                     # no cell in the table for this form on this card / stack: the provider serves its declared fallback and prints UNCOVERED_CELL; counted here by family
        _STATE["uncovered"][fam] = _STATE["uncovered"].get(fam, 0) + 1
    before = {k: v for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:")}
    try:                                                                  # the provider walks the word's candidates; a row that cannot engage steps aside by name inside it (tier words) or raises (a row word)
        out = PS.triangle_multiplication(act, mask, params, equation=equation, form=FORM, word=w, direction="fwd", selection=selection)
    except Refusal as e:                                                  # refused by name (levers_off, a dtype / shape rule, an unknown word): this call site keeps the arithmetic it had
        _aside(refusal_word(e)); return None
    except NotImplementedError as e:                                      # a transform the row's kernels do not cover: by name
        _aside("trace:" + type(e).__name__); return None
    served = [k.split(":", 2)[2] for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:") and v > before.get(k, 0)]
    arm = served[0] if served else (getattr(selection, "arm", None) or w)
    _STATE["traced"] += 1
    _STATE["rows"][arm] = _STATE["rows"].get(arm, 0) + 1
    _STATE["sites"][site] = _STATE["sites"].get(site, 0) + 1
    return out.astype(act.dtype)


def _make_class(stock_tm, modules):
    """``TriangleMultiplication`` served by the provider; the stock lines (alphafold3/model/network/modules.py:258-344) where it steps aside."""
    import haiku as hk
    import jax.numpy as jnp
    hm = modules.hm

    class CdTriangleMultiplication(stock_tm):
        __wrapped__ = stock_tm                                            # the KERNELS census: the stock class owns the site, this wrapper is named (kernels_probe._site_word)

        @hk.transparent                                                  # called from a __call__ already inside this module's scope (ours or the add-on's fused class): the STOCK parameter names resolve as the stock lines' do
        def trimul_cd_row(self, act, mask, site="fused"):
            """For the FlashPairformer add-on's fused class (site fused) and this class's own __call__ (site module): the whole module from the
            provider, or None (stepped aside by name; the caller's own arithmetic runs)."""
            eq = getattr(self.config, "equation", None)
            why = early_aside(act, mask, eq)                              # decided before any parameter is fetched: the caller (the add-on's fused class or the stock lines) fetches them itself when it runs instead
            if why is not None:
                note_aside(why); return None
            c = act.shape[-1]
            w_proj, _ = hm.haiku_linear_get_params(act, num_output=2 * c, name="projection")
            w_gate, _ = hm.haiku_linear_get_params(act, num_output=2 * c, initializer=self.global_config.final_init, name="gate")
            with hk.name_scope("left_norm_input"):
                s_in = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o_in = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            with hk.name_scope("center_norm"):
                s_c = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o_c = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            w_out, _ = hm.haiku_linear_get_params(act, num_output=c, initializer=self.global_config.final_init, name="output_projection")
            w_gl, _ = hm.haiku_linear_get_params(act, num_output=c, name="gating_linear")
            p = dict(ln_in_scale=s_in, ln_in_offset=o_in, left_w=w_proj[:, 0::2], right_w=w_proj[:, 1::2],          # the stock reshape(C, 2, ...) split: column 2c = left channel c, 2c+1 = right channel c
                     left_gate_w=w_gate[:, 0::2], right_gate_w=w_gate[:, 1::2], ln_c_scale=s_c, ln_c_offset=o_c, out_w=w_out, gate_w=w_gl)
            return trimul_row(act, mask, p, eq, site=site)

        def __call__(self, act, mask):
            out = self.trimul_cd_row(act, mask, site="module")
            if out is None:                                               # stepped aside (counted by name): the stock module of this call site
                return super().__call__(act, mask)
            return out

    CdTriangleMultiplication.__name__ = CdTriangleMultiplication.__qualname__ = "CdTriangleMultiplication"
    return CdTriangleMultiplication


def install() -> bool:
    """Rebind ``modules.TriangleMultiplication`` (idempotent); imports the provider eagerly so a core without it is named at start."""
    if _STATE["installed"]:
        return True
    import alphafold3.model.network.modules as modules
    from opt_core.kernels.pallas import serve as PS                      # noqa: F401 — the core on the model process's PYTHONPATH (the kit's [tool.opt_core] path), as the add-on's kernels
    _STATE["word"] = word()
    _STATE["stock"] = modules.TriangleMultiplication
    _STATE["cls"] = _make_class(modules.TriangleMultiplication, modules)
    modules.TriangleMultiplication = _STATE["cls"]
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        import alphafold3.model.network.modules as modules
        modules.TriangleMultiplication = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    core = {}
    try:
        from opt_core.kernels.pallas import serve as PS
        r = PS.report()
        core = {"counts": {k: v for k, v in (r.get("counts") or {}).items() if FACE in k or k.startswith("refused:")}, "last_refusal": r.get("last_refusal")}
    except Exception as e:  # noqa: BLE001 — the report is words; a core that does not import is named here and is tcd=off on the line
        core = {"error": f"{type(e).__name__}: {e}"}
    return {"installed": _STATE["installed"], "word": _STATE["word"] or word(), "traced": _STATE["traced"], "rows": dict(_STATE["rows"]),
            "sites": dict(_STATE["sites"]), "aside": dict(_STATE["aside"]), "uncovered": dict(_STATE["uncovered"]), "core": core}
