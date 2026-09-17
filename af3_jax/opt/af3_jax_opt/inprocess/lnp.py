"""LNP — the model's standalone LayerNorms (``haiku_modules.LayerNorm``, the class every AF3 module reaches as ``hm.LayerNorm``) served
through the shared core's JAX-family provider ``opt_core.kernels.pallas`` (``serve.layer_norm``; row ``cd_ln`` = a Pallas row-LayerNorm
kernel, one program per row with f32 statistics in registers and one store per element; row ``xla`` = the provider's
``reference_layer_norm`` statement) by the mode's tier word (``fast`` under --mode fast, ``big`` under --mode big): per call the
provider serves the row its cell table names for the call's cell (PALLAS_CELLS.json op ``ln``: unit x channels x card x jax version x size).
Only the arithmetic of ``__call__`` changes: the parameters keep the stock names, shapes and dtypes (``scale`` / ``offset``, [C], f32 for an
upcast 16-bit input), fetched under the module's own scope.

Units routed (UNITS): ``pair`` — the square [N, N, 128] pair LayerNorms outside the fused pair blocks (and the pair-stack modules' own
norms wherever the fused classes step aside); ``tmpl`` — the [.., N, N, 64] template pair planes; ``msa`` — the [S, N, 64] MSA activations
(family depth by :func:`msa_depth`: the nearest depth in the table at or above S, else the deepest). The unit is decided from the call
alone (channels, rank, squareness; the module's scope name separates an MSA plane from a template plane) — :func:`classify`.

Kept on the stock class BY RULE, counted by reason (``routed=``), never silently: no ``scale`` + ``offset`` pair (the adaptive LayerNorms
of the diffusion / atom transformers and the conditioning norms: ``adaptive``), a normalisation or parameter axis other than the last (the
triangle multiplication's centre norm: ``axis``), a channel count outside UNIT_CHANNELS or MIN_C..MAX_C or not a power of two (single
C = 384, diffusion-transformer C = 768: ``channels``), non-square C = 128 planes and non-square template planes (row blocks: ``rows``), rank
below 3 (``rank``), a 16-bit input the module does not upcast (``no_upcast``), an epsilon other than EPS (``eps``), a dtype other than
bf16 / f32 (``dtype_<name>``), a backend other than gpu (``platform``). A provider refusal by name (``MODEL_OPT_LEVERS_OFF=pallas[:cd_ln]``,
an unknown word, a stack without the row's imports) is an ``aside`` counted by the refusal's word; the stock formula serves that call.

Numerics: NOT bitwise with stock — row ``cd_ln``: f32 statistics, two-pass variance where stock uses E[x^2] - E[x]^2, another summation
order; row ``xla``: the provider's statement of the same formula — a fast-class (tolerance) lever. ``--mode exact`` does not name it.

Switch: ``AF3_JAX_LNP=<word>`` in the model process's environment (the mode table sets it for a composition that names LNP: ``1`` = the
tier word ``fast``; ``big`` under --mode big; any other value is passed to the provider verbatim — a tier word or a row ``cd_ln`` |
``xla``; an unknown word is the provider's refusal by name at every call). ``MODEL_OPT_LEVERS_OFF=LNP`` removes the switch.
Prints (the launcher, from :func:`line`): ``[af3-jax-opt] LNP served=<traced calls the provider served>|off word=<word> rows=<row:n,...|none>
units=<unit:n,...|none> routed=<reason:n,...|none> aside=<reason:n,...|none> uncovered=<family:n,...|none>`` (trace-time call sites;
``uncovered`` = served calls whose family has no cell on this card / jax version: the provider names the XLA statement there).
"""
import os
import re

ENV_SWITCH = "AF3_JAX_LNP"
REBINDS = ("alphafold3.model.components.haiku_modules:LayerNorm",)   # what install() rebinds (module:attribute) — the class name every AF3 module resolves as hm.LayerNorm at trace time, to a same-name subclass; fpf_launch.TREE_LEVERS / tests/test_install_order.py read it
DEFAULT_WORD = "fast"                                                     # the provider's TIER word for switch value "1" (opt_core.kernels.pallas TIER_WORDS)
TIER_WORDS = ("fast", "exact", "big")
FACE = "layer_norm"                                                       # the provider's serve face (serve.COUNTS keys 'served:<face>:<arm>')
UNITS = ("pair", "tmpl", "msa")                                           # the provider LayerNorm units this lever routes (PALLAS_CELLS.json ln families pair_c128, tmpl_c64[_t4], msa_c64_s<depth>)
UNIT_CHANNELS = {128: "pair", 64: "tmpl_or_msa"}                          # channel count -> unit family (64: square plane = tmpl unless the module scope says msa; [S, N, 64] = msa)
MIN_C, MAX_C = 32, 1024                                                   # row cd_ln's channel range (powers of two); outside it no row but xla exists
EPS = 1e-5                                                                # the provider's LayerNorm epsilon (serve.LN_EPS); a module with another eps keeps the stock lines
PREFIX = "[af3-jax-opt]"
_STATE = {"installed": False, "word": None, "stock": None, "traced": 0, "rows": {}, "units": {}, "routed": {}, "aside": {}, "uncovered": {}, "families": None}


def wanted(environ=os.environ) -> bool:
    return (environ.get(ENV_SWITCH, "") or "").strip() not in ("", "0")


def word(environ=os.environ) -> str:
    """The provider word this process names: ``1`` = DEFAULT_WORD (tier word fast); any other non-empty value verbatim (``big`` under --mode big)."""
    v = (environ.get(ENV_SWITCH, "") or "").strip()
    return DEFAULT_WORD if v in ("", "1") else v


def _word(text) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-@") else "_" for ch in str(text))[:60].strip("_") or "unknown"


def _count(table: str, key: str) -> None:
    _STATE[table][key] = _STATE[table].get(key, 0) + 1


def classify(*, shape, create_scale: bool, create_offset: bool, axis, param_axis, eps: float, upcast: bool, is_16bit: bool, scope: str = "",
             measured=None):
    """Decide the call from its facts alone -> (unit, kwargs for serve.layer_norm, None) when routed to the provider, or (None, None, reason)
    when the stock lines keep it BY RULE. ``shape`` = the input's, ``scope`` = the module's scope name (haiku module_name), ``measured`` =
    the provider's ln family words (families('ln')) for the depth / template-count rule (None = pass the facts as they are)."""
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    if not (create_scale and create_offset):
        return None, None, "adaptive"
    ax = tuple(sorted((a % nd) if nd else a for a in (axis if isinstance(axis, (tuple, list)) else (axis,))))
    if nd and ax != (nd - 1,):
        return None, None, "axis"
    if param_axis:
        pa = tuple((a % nd) if nd else a for a in (param_axis if isinstance(param_axis, (tuple, list)) else (param_axis,)))
        if pa != (nd - 1,):
            return None, None, "axis"
    if abs(float(eps) - EPS) > 1e-12:
        return None, None, "eps"
    if is_16bit and not upcast:
        return None, None, "no_upcast"
    if nd < 3:
        return None, None, "rank"
    C = shape[-1]
    if C < MIN_C or C > MAX_C or (C & (C - 1)) or C not in UNIT_CHANNELS:
        return None, None, "channels"
    square = shape[-2] == shape[-3]
    scope_l = (scope or "").lower()
    is_msa_scope = ("msa" in scope_l) and ("pair" not in scope_l.rsplit("/", 1)[-1])
    if C == 128:
        if not square:
            return None, None, "rows"
        return "pair", {"unit": "pair", "n_tokens": shape[-2]}, None
    # C == 64: template pair plane [.., N, N, 64] or MSA activations [S, N, 64]
    if square and not is_msa_scope:
        kw = {"unit": "tmpl", "n_tokens": shape[-2]}
        if nd >= 4 and measured is not None and ("tmpl_c%d_t%d" % (C, shape[-4])) in measured:
            kw["t"] = shape[-4]                                            # a batched template stack at a measured template count: its own family
        return "tmpl", kw, None
    if not square and not is_msa_scope and "template" in scope_l:
        return None, None, "rows"
    S = shape[-3]
    return "msa", {"unit": "msa", "n_tokens": shape[-2], "n_seq": msa_depth(C, S, measured)}, None


def msa_depth(C: int, S: int, measured=None) -> int:
    """The provider's measured-depth rule for MSA families (its docstring's n_seq rule): the nearest measured depth at or above S, else the
    deepest measured; no measured msa family at this C (or no table) -> S itself (the provider then names the XLA statement, uncovered)."""
    depths = []
    for f in (measured or ()):
        m = re.match(r"^msa_c%d_s(\d+)$" % int(C), f)
        if m:
            depths.append(int(m.group(1)))
    if not depths:
        return int(S)
    at_or_above = sorted(d for d in depths if d >= int(S))
    return at_or_above[0] if at_or_above else max(depths)


def refusal_word(exc) -> str:
    row, kind = getattr(exc, "row", None), getattr(exc, "kind", None)
    return f"{row or 'none'}:{kind}" if kind else type(exc).__name__


def _families():
    if _STATE["families"] is None:
        try:
            from opt_core.kernels import pallas as P
            _STATE["families"] = tuple(P.families("ln"))
        except Exception:                                                 # noqa: BLE001 — a core without the table: the facts pass as they are (the provider names what it has)
            _STATE["families"] = ()
    return _STATE["families"]


def _platform_ok() -> bool:
    try:
        import jax
        return jax.default_backend() == "gpu"
    except Exception:                                                     # noqa: BLE001
        return False


def _stock_formula(self, x, scale, offset):
    """The stock arithmetic with the parameters already fetched (haiku's LayerNorm formula over the last axis: the module's upcast, its variance
    form, its eps) — for a call the provider refused by name after the parameters were read (they are read once per module scope)."""
    import jax
    import jax.numpy as jnp
    dtype = x.dtype
    xf = x.astype(jnp.float32) if dtype in (jnp.bfloat16, jnp.float16) else x
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    if bool(getattr(self, "use_fast_variance", True)):
        var = jnp.maximum(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) - jnp.square(mean), 0.0)
    else:
        var = jnp.var(xf, axis=-1, keepdims=True)
    y = (xf - mean) * jax.lax.rsqrt(var + float(getattr(self, "eps", EPS))) * scale.astype(xf.dtype) + offset.astype(xf.dtype)
    return y.astype(dtype)


def _lnp_call(self, x, stock_call):
    """``haiku_modules.LayerNorm.__call__`` served by the provider's row: the unit decided from the call, the module's own parameters (stock
    names / shapes / dtypes, under its own scope: this runs as the same-name subclass's wrapped method), the provider's face by the process's
    word; the stock method by name where the call is kept by rule, the stock formula where the provider refuses by name."""
    import haiku as hk
    import jax.numpy as jnp
    dtype = x.dtype
    is16 = dtype in (jnp.bfloat16, jnp.float16)
    unit, kw, why = classify(shape=x.shape, create_scale=bool(getattr(self, "_temp_create_scale", False)), create_offset=bool(getattr(self, "_temp_create_offset", False)),
                             axis=getattr(self, "axis", -1), param_axis=getattr(self, "param_axis", None), eps=getattr(self, "eps", EPS), upcast=bool(getattr(self, "upcast", True)),
                             is_16bit=is16, scope=str(getattr(self, "module_name", "") or ""), measured=_families())
    if why is None and dtype not in (jnp.bfloat16, jnp.float32):
        why = "dtype_" + jnp.dtype(dtype).name
    if why is None and not _platform_ok():
        why = "platform"
    if why is not None:                                                   # kept on the stock lines by rule: counted, the stock method serves (it fetches its own parameters)
        _count("routed", _word(why))
        return stock_call(self, x)
    from opt_core.kernels import pallas as P
    from opt_core.kernels.pallas import Refusal
    from opt_core.kernels.pallas import serve as PS
    C = int(x.shape[-1])
    pdtype = jnp.float32 if is16 else dtype                                # the stock parameter dtype: x's after the module's upcast (haiku_modules.LayerNorm.__call__)
    scale = hk.get_parameter("scale", (C,), pdtype, init=self.scale_init)
    offset = hk.get_parameter("offset", (C,), pdtype, init=self.offset_init)
    w = _STATE["word"] or word()
    fam, sel = None, None
    try:
        fam = P.family("ln", unit=kw["unit"], c=C, n_seq=kw.get("n_seq"), t=kw.get("t"))
        if w in TIER_WORDS:                                               # a tier word: the provider's selection for this cell, read once here so an uncovered family is counted by name
            sel = PS.resolve("ln", fam, x.dtype, int(kw["n_tokens"]), word=w, direction="fwd")
    except Refusal as e:                                                  # refused by name before anything ran (levers_off, an unmeasured jax line, an unknown word): the stock formula serves this call
        _count("aside", _word(refusal_word(e)))
        return _stock_formula(self, x, scale, offset)
    if sel is not None and getattr(sel, "cell_key", None) is None:        # no measured cell for this family on this card / line: the provider names the XLA statement (its census prints the token); counted by family
        _count("uncovered", _word(fam))
    before = {k: v for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:")}
    try:
        out = PS.layer_norm(x, scale.astype(jnp.float32), offset.astype(jnp.float32), word=w, direction="fwd", selection=sel, **kw)
    except Refusal as e:                                                  # a row that cannot engage and names no next arm: by name, the stock formula serves this call
        _count("aside", _word(refusal_word(e)))
        return _stock_formula(self, x, scale, offset)
    except NotImplementedError as e:                                      # a transform the row's kernel does not cover (a batching rule): by name
        _count("aside", "trace_" + type(e).__name__)
        return _stock_formula(self, x, scale, offset)
    served = [k.split(":", 2)[2] for k, v in PS.COUNTS.items() if k.startswith(f"served:{FACE}:") and v > before.get(k, 0)]
    _STATE["traced"] += 1
    _count("rows", _word(served[0] if served else w))
    _count("units", unit)
    return out.astype(dtype)


def _make_class(A):
    """The same-name subclass of ``haiku_modules.LayerNorm`` whose ``__call__`` is _lnp_call (haiku's metaclass wraps the subclass method, so the
    parameters keep the module's own scope and names — the opt_core cd_layers install pattern)."""
    stock_call = A.__call__

    class LayerNorm(A):                                                   # noqa: D101
        __wrapped__ = A                                                   # the KERNELS census: the stock class owns the site, this wrapper is named
        _stock_cls = A

        def __call__(self, x):
            return _lnp_call(self, x, stock_call)

    LayerNorm.__qualname__ = "LayerNorm"
    LayerNorm.__module__ = A.__module__
    return LayerNorm


def install() -> bool:
    """Rebind ``haiku_modules.LayerNorm`` (idempotent; every ``hm.LayerNorm(...)`` site resolves the name at trace time); imports the provider
    eagerly so a core without it is named at start."""
    if _STATE["installed"]:
        return True
    from alphafold3.model.components import haiku_modules as HM
    from opt_core.kernels.pallas import serve as PS                      # noqa: F401 — the core on the model process's PYTHONPATH, as the other tree levers' kernels
    _STATE["word"] = word()
    _STATE["stock"] = HM.LayerNorm
    HM.LayerNorm = _make_class(HM.LayerNorm)
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.components import haiku_modules as HM
        HM.LayerNorm = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    return {"installed": _STATE["installed"], "word": _STATE["word"] or word(), "traced": _STATE["traced"], "rows": dict(_STATE["rows"]), "units": dict(_STATE["units"]),
            "routed": dict(_STATE["routed"]), "aside": dict(_STATE["aside"]), "uncovered": dict(_STATE["uncovered"])}


def line(rep=None) -> str:
    """The lever's ONE census line (the launcher prints it at exit next to SERVED; modes.LNP_LINE_RX reads it)."""
    r = report() if rep is None else rep
    j = lambda d: ",".join(f"{k}:{v}" for k, v in sorted((d or {}).items())) or "none"  # noqa: E731
    served = str(r["traced"]) if r.get("installed") else "off"
    return (f"{PREFIX} LNP served={served} word={r.get('word') or 'none'} rows={j(r.get('rows'))} units={j(r.get('units'))} "
            f"routed={j(r.get('routed'))} aside={j(r.get('aside'))} uncovered={j(r.get('uncovered'))}")
