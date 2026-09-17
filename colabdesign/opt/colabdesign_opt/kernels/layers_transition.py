"""Lever `transition` — every `modules.Transition` of the design model (AlphaFold-Multimer's Transition: LayerNorm -> Linear(C->4C) -> relu ->
Linear(4C->C); the Evoformer's pair_transition [N,N,128] and msa_transition [S,N,256] per block, the extra-MSA stack's [.,N,64], the template
pair stack's factor-2 [N,N,64]) served by the fused ReLU-transition kernel THE SHARED CORE'S PROVIDER NAMES for the call's cell — the JAX-family
provider `opt_core.kernels.pallas`, op `transition`, reached BY THE TIER WORD through `kernels/provider.py` (the LEVER line names `provider= word=
row= tier= [uncovered=]`). This module is the ADAPTER ONLY: the haiku class rebinding under stock's parameter names / scopes / dtypes /
initializers (input_layer_norm/{scale,offset}, transition1/{weights,bias}, transition2/{weights,bias} — the parameter tree is unchanged, the
model's params load as they are), the structural per-call step-aside words, the census and the LEVER line. It carries no kernel and no size / card /
tile table: which row serves a (compute capability, channels, tokens, direction) cell — or the stock statement `xla`, by name — is the provider's
measured cell table, and a call whose cell names a row this adapter does not bind runs the stock class BY NAME, counted `fallback_by=cell_<row>:n`.

Row forms this adapter binds (read off the row module the word resolves to, printed `variant=`):
* `fused`  — a row module exposing `transition(x, scale, offset, w1, b1, w2, b2, eps)`: LayerNorm + MLP in ONE kernel per direction (the provider's
             `cd_transition`: bf16 products / f32 statistics and accumulation, the 4C intermediate on-chip; class `precision`, never bitwise);
* `lnkeep` — a row module exposing `mlp_2d(xn2, w1, b1, w2, b2, eps)`: the MLP alone in one kernel per direction, the LayerNorm left to the run's
             own `common_modules.LayerNorm` module call (lever `ln`'s row when that lever is on).
The design step differentiates the sequence only: the rows' activation gradient is the kernel's; parameter cotangents are the rows' plain-JAX
expressions (dead code there).

Served: GPU backend, bf16 activations, C in SERVED_C (64/128/256: every Transition of the model), the intermediate a multiple of HIDDEN_CHUNK,
rank >= 2 — the structural words `dtype_not_bf16` / `channels` / `intermediate_width` / `rank` (EXPECTED_FALLBACKS; `platform` stays undeclared =
fail-closed) — then the provider's cell for (C, factor, x.shape[-2] = the design's token count, fwdbwd): a bound row serves it, `xla` (or any row
this adapter does not bind) hands it to the stock class as `cell_<row>`. Admission at install (provider.admit): the word resolved for the model's
MAIN cell (pair transition, C=128, 512 tokens) -> that row module (`admit=main`); a main cell that names the stock statement while the binding's
rows import is NOT a refusal for this lever — the template pair stack's C=64 transitions have cells of their own that can name a bound row on the
same card — so every call resolves by its own cell (`admit=cells`); a word the provider refuses, or a bound row that does not import here, is the
lever's refusal BY NAME at install (`state=skipped reason=cannot_run detail=…`, levers.install) and the mode runs the rest of its set.

Install: `install()` rebinds `colabdesign.af.alphafold.model.modules.Transition` to a same-name subclass (marker MARKER; haiku builds the class
through the module attribute at trace time) before the model is traced. The LEVER line prints once at exit (kernels.register_exit_line):
    [colabdesign-opt] LEVER name=transition state=on impl=<row module>@<sha8> origin=core numerics=precision precision=bf16 variant=<fused|lnkeep> served=<n> fallback=<n> fallback_by=<reason:n,..|none> shapes=<C..:n,..|none> admit=<main|cells> provider=opt_core.kernels.pallas@<core> word=fast row=<row:n,…> tier=<arm:cells,…> [uncovered=<n>:<family/direction>,…] source=exit
Ablation: the mode word `fast-no-transition`. Lever protocol (registry.py): NUMERICS, REFUSALS, EXPECTED_FALLBACKS, install() -> dict, installed(),
uninstall(), off_line(reason), evidence(), lever_line().
"""
from __future__ import annotations

import collections

try:                                     # the kit's CPU test images carry a stand-in jax: import-tolerant like kernels/layers_ln.py; _require() refuses by name
    import jax
    import jax.numpy as jnp
except Exception:                        # noqa: BLE001  # pragma: no cover - ImportError, or AttributeError when a real haiku meets a stand-in jax
    jax = jnp = None

from ..names import TAG
from . import provider                  # noqa: E402 - the kit's binding of this lever to the shared core's provider (pure Python)

NAME = "transition"                      # registry id = the LEVER line's name=
IMPL = provider.kernel_word(NAME)        # `<row module>@<sha8>`: the provider's row module (located, not imported: the same word on a GPU-less host)
ORIGIN = "core"                          # the implementation lives in the shared core; the adapter (class rebinding, gates, census) is this module
MARKER = "_colabdesign_opt_layers_transition"
NUMERICS = "precision"
PRECISION = "bf16"
FORMS = ("fused", "lnkeep")              # the row forms this adapter binds (module doc); `variant=` on the LEVER line names the bound row's
EXPECTED_FALLBACKS = ("dtype_not_bf16", "channels", "intermediate_width", "rank")   # every structural per-call step-aside word (stock's Transition serves the call BY NAME); a provider cell word `cell_<row>` is declared by its prefix (evidence.CELL_FALLBACK_PREFIX); `platform` stays undeclared = fail-closed
SERVED_C = (64, 128, 256)
HIDDEN_CHUNK = 128                       # the rows process the 4C (or 2C) intermediate in on-chip chunks of this width; H must be a multiple of it (every Transition of the model is)
LN_EPS = 1e-5                            # modules.Transition's LayerNorm eps (haiku default), handed to the fused form's in-kernel LayerNorm

_CENSUS = {"served": collections.Counter(), "fallback": collections.Counter()}
_STATE = {"installed": False, "stock_cls": None, "K": None, "admit": "none"}   # K: the row module the word resolved to for the main cell at install (provider.admit) | the first bound row (admit=cells)


def form_of(K) -> str:
    """The call form of a row module: `lnkeep` when it serves the MLP alone (`mlp_2d`), `fused` when it serves LayerNorm + MLP (`transition`)."""
    if K is not None and callable(getattr(K, "mlp_2d", None)):
        return "lnkeep"
    if K is not None and callable(getattr(K, "transition", None)):
        return "fused"
    raise Refusal(f"lever {NAME}: row module {getattr(K, '__name__', K)!r} exposes neither transition(x, scale, offset, w1, b1, w2, b2, eps) nor mlp_2d(xn2, w1, b1, w2, b2, eps)")


def _sm80_tiles(K) -> None:
    """A `fused` row module carries no compute-capability 8.x tile entry; the provider's face installs one before it serves the row there
    (serve._cd_transition_sm80_tiles: idempotent, a no-op on other cards). The adapter calls the row module directly, so it asks the face once."""
    try:
        from opt_core.kernels.pallas import serve as _serve  # noqa: WPS433
        fix = getattr(_serve, "_cd_transition_sm80_tiles", None)
        if fix is not None and hasattr(K, "_tiles"):
            fix(K)
    except Exception:                    # noqa: BLE001 - an older face without the helper: the row's own tiles
        pass


def reference_layer_norm(x, scale, offset, eps=LN_EPS):
    xf = x.astype(jnp.float32)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.var(xf, axis=-1, keepdims=True)
    return (scale * jax.lax.rsqrt(var + eps) * (xf - mu) + offset).astype(x.dtype)


def reference_transition(x, scale, offset, w1, b1, w2, b2, eps=LN_EPS):
    """Stock's Transition on arrays in stock's dtypes (bf16 operands, XLA's f32 accumulation): the unit tests' A side without haiku."""
    xn = reference_layer_norm(x, scale, offset, eps)
    h = jax.nn.relu(jnp.dot(xn, w1, preferred_element_type=jnp.float32).astype(x.dtype) + b1)
    return jnp.dot(h, w2, preferred_element_type=jnp.float32).astype(x.dtype) + b2


# ─────────────────────────────────────────── serving rule ───────────────────────────────────────────
def _eligibility(act, C, H):
    """None when the call is structurally one the rows serve, else the fallback reason word (the stock class BY NAME)."""
    if jax.default_backend() != "gpu":
        return "platform"
    if act.dtype != jnp.bfloat16:
        return "dtype_not_bf16"
    if C not in SERVED_C:
        return "channels"
    if H < HIDDEN_CHUNK or H % HIDDEN_CHUNK:
        return "intermediate_width"
    if act.ndim < 2:
        return "rank"
    return None


def _resolve(act, C: int, H: int):
    """(row module | None, its fallback word) for this call's cell under the bound word (provider.resolve): family (form af2, relu, C, factor),
    tokens = x.shape[-2] (the residue axis every Transition input of the model carries second to last), direction fwdbwd."""
    n = int(act.shape[-2])
    shape = dict(form="af2", activation="relu", c=int(C), factor=max(1, int(H) // int(C)))
    K, _cfg = provider.resolve(NAME, n, "bf16", **shape)
    return K, (None if K is not None else provider.fallback_word(NAME, n, "bf16", **shape))


def _name_scope(name):
    import haiku as hk
    ns = getattr(hk, "name_scope", None) or hk.experimental.name_scope
    return ns(name)


def _transition_call(self, act, mask, stock_call):
    """modules.Transition.__call__ (lever transition): stock's parameters and names; the row's kernel in the row's form."""
    import haiku as hk
    from colabdesign.af.alphafold.model import common_modules
    c, gc = self.config, self.global_config
    C = int(act.shape[-1])
    H = int(C * c.num_intermediate_factor)
    why = _eligibility(act, C, H)
    if why is not None:
        _CENSUS["fallback"][why] += 1
        return stock_call(self, act, mask)
    K, cell_word = _resolve(act, C, H)
    if K is None:                                                                                # the word names a row this adapter does not bind for this cell (`xla`): the stock class, by the cell's name
        _CENSUS["fallback"][cell_word] += 1
        return stock_call(self, act, mask)
    form = form_of(K)
    w_init1 = hk.initializers.TruncatedNormal(stddev=float((2.0 / C) ** 0.5))                    # stock initializer='relu' (values matter at init only)
    w_init2 = hk.initializers.Constant(0.0) if gc.zero_init else hk.initializers.TruncatedNormal(stddev=float((1.0 / H) ** 0.5))   # 'final_init'
    if form == "fused":                                                                          # LayerNorm inside the row's kernel: stock's LN parameters under stock's scope
        with _name_scope("input_layer_norm"):
            scale = hk.get_parameter("scale", (C,), jnp.float32, init=jnp.ones)
            offset = hk.get_parameter("offset", (C,), jnp.float32, init=jnp.zeros)
    else:                                                                                        # lnkeep: stock's own LayerNorm module call, verbatim (the `ln` lever's class when that lever is on)
        act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='input_layer_norm')(act)
    with _name_scope("transition1"):
        w1 = hk.get_parameter("weights", (C, H), act.dtype, init=w_init1)
        b1 = hk.get_parameter("bias", (H,), act.dtype, init=hk.initializers.Constant(0.0))
    with _name_scope("transition2"):
        w2 = hk.get_parameter("weights", (H, C), act.dtype, init=w_init2)
        b2 = hk.get_parameter("bias", (C,), act.dtype, init=hk.initializers.Constant(0.0))
    _CENSUS["served"][f"C{C}"] += 1
    if form == "fused":
        _sm80_tiles(K)
        return K.transition(act, scale.astype(jnp.float32), offset.astype(jnp.float32), w1.astype(act.dtype), b1.astype(act.dtype),
                            w2.astype(act.dtype), b2.astype(act.dtype), LN_EPS)
    y2 = K.mlp_2d(act.reshape(-1, C), w1.astype(act.dtype), b1.astype(act.dtype), w2.astype(act.dtype), b2.astype(act.dtype), LN_EPS)
    return y2.reshape(act.shape)


def install() -> dict:
    _require()
    b = provider.BINDINGS[NAME]
    try:
        _STATE["K"] = provider.admit(NAME)                            # the word -> the row module, by the provider's selection for the model's main cell
        _STATE["admit"] = "main"
    except provider.ProviderRefusal as e:
        if getattr(e, "kind", None) != "row_unbindable":              # the provider refused the word, or a bound row does not import here: the lever's refusal BY NAME
            raise Refusal(f"lever {NAME}: {e}") from None
        # The main cell (pair transition, C=128) names the stock statement on this card / stack while other cells of the same model (the template
        # pair stack's C=64 transitions) can name a bound row: not a refusal for this lever — every call resolves by ITS OWN cell (`admit=cells`
        # on the line; the main-cell calls come back `cell_xla` by name).  The bound rows are imported now, so a row that cannot import refuses here,
        # never at trace time.
        mods = []
        for row in b.rows:
            try:
                mods.append(provider.row_module(b, row))
            except provider.ProviderRefusal as e2:
                raise Refusal(f"lever {NAME}: {e2}") from None
        _STATE["K"] = mods[0] if mods else None
        _STATE["admit"] = "cells"
        b.notes["admit"] = str(e)[:300]
    form_of(_STATE["K"])                                              # a row of a form this adapter does not bind: refused by name here, not at trace time
    from colabdesign.af.alphafold.model import modules
    A = modules.Transition
    if not getattr(A, MARKER, False):
        stock_call = A.__call__

        class Transition(A):                    # noqa: D101
            _stock_cls = A

            def __call__(self, act, mask, *args, **kwargs):
                return _transition_call(self, act, mask, lambda s_, a_, m_: stock_call(s_, a_, m_, *args, **kwargs))
        setattr(Transition, MARKER, True)
        Transition.__qualname__ = "Transition"; Transition.__module__ = A.__module__
        _STATE["stock_cls"] = A
        modules.Transition = Transition
    _STATE["installed"] = True
    _register_exit_line()
    return {"lever": NAME, "impl": provider.kernel_word(NAME), "origin": ORIGIN, "numerics": NUMERICS, "precision": PRECISION, "variant": variant_word(),
            "row": provider.bound_row(NAME), "word": provider.BINDINGS[NAME].word, "provider": provider.provider_word()}


def uninstall() -> None:
    import sys
    modules = sys.modules.get("colabdesign.af.alphafold.model.modules")   # nothing to undo in a process that never imported the model
    A = getattr(modules, "Transition", None) if modules else None
    if A is not None and getattr(A, MARKER, False):
        modules.Transition = A._stock_cls
    _STATE["installed"] = False
    _STATE["K"] = None
    _STATE["admit"] = "none"


def installed() -> bool:
    return bool(_STATE["installed"])


def stock_class():
    """Stock's Transition class (before install: the module's; after: the one install() displaced)."""
    from colabdesign.af.alphafold.model import modules
    A = modules.Transition
    return A._stock_cls if getattr(A, MARKER, False) else A


def variant_word() -> str:
    """The bound row's call form (`fused` | `lnkeep`), `none` before a row is admitted."""
    try:
        return form_of(_STATE["K"]) if _STATE.get("K") is not None else "none"
    except Refusal:
        return "unknown"


def reset_census() -> None:
    for c in _CENSUS.values():
        c.clear()


def census() -> dict:
    return {"served": int(sum(_CENSUS["served"].values())), "fallback": int(sum(_CENSUS["fallback"].values())),
            "shapes": dict(_CENSUS["served"]), "fallback_by": dict(_CENSUS["fallback"])}


# ─────────────────────────────────────────── kit lever protocol (registry.py) ───────────────────────────────────────────
class Refusal(RuntimeError):
    """install() raises this when the lever cannot run here (levers.install turns it into `state=skipped reason=cannot_run`)."""


REFUSALS = (Refusal,)


def _require() -> None:
    """The lever's floor: jax with the GPU backend (the rows' Pallas/Triton kernels lower on it only; a row that does not import here is the
    provider's refusal by name at admit)."""
    if jax is None:
        raise Refusal(f"lever {NAME}: jax is not importable here")
    try:
        backend = jax.default_backend()
    except Exception as e:                                            # noqa: BLE001
        raise Refusal(f"lever {NAME}: backend: {e!r}") from None
    if backend != "gpu":
        raise Refusal(f"lever {NAME}: backend={backend} (gpu required)")


def evidence() -> dict:
    c = census()
    return {"lever": NAME, "line_name": NAME, "installed": bool(_STATE["installed"]), "impl": provider.kernel_word(NAME), "origin": ORIGIN, "numerics": NUMERICS,
            "precision": PRECISION, "variant": variant_word(), **c, "binding": provider.report(NAME)}


def off_line(reason: str) -> str:
    from opt_core import report as _r
    return _r.lever_line(TAG, NAME, "off", reason=reason, impl=provider.kernel_word(NAME), origin=ORIGIN)


def lever_line() -> str:
    """This lever's ONE LEVER line (state=on with its census; the kit's exit printer calls it at process exit)."""
    from opt_core import report as _r
    if not _STATE["installed"]:
        return off_line("not_installed")
    c = census()
    shapes = ",".join(f"{k}:{v}" for k, v in sorted(c["shapes"].items())) or "none"
    fb = ",".join(f"{k}:{v}" for k, v in sorted(c["fallback_by"].items())) or "none"
    return _r.lever_line(TAG, NAME, "on", ("numerics", NUMERICS), ("precision", PRECISION), ("variant", variant_word()), ("served", c["served"]),
                         ("fallback", c["fallback"]), ("fallback_by", fb), ("shapes", shapes), ("admit", _STATE.get("admit", "none")), *provider.facts(NAME).items(), ("source", "exit"),
                         impl=provider.kernel_word(NAME), origin=ORIGIN)


def _register_exit_line() -> None:
    """Hand lever_line to the package's one exit printer (kernels/__init__.py) — a no-op until the registry names this lever."""
    try:
        from . import register_exit_line
        register_exit_line(NAME, lever_line)
    except (ImportError, ValueError):
        pass
