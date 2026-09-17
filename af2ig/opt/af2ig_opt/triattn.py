"""L10 ``-fused_triattn``: AlphaFold's ``TriangleAttention`` module (starting and ending node; the Evoformer pair stack's and the template pair
stack's) served by the model-opt tree's kernel PROVIDER, ``opt_core.kernels.pallas.serve.triangle_attention_block``, BY TIER WORD : the kit
hands the provider the mode's tier word (``fast`` | ``big``, modes.tier_word) and the tier's float32 product class (modes.F32_PRODUCTS), and the
provider serves the fastest measured row of the call's cell (compute capability, dtype, family = form af2 / unit pair|tmpl / c / heads / head dim /
orientation, length) — the fused Pallas block, the XLA-FFI triangle-attention bridge behind the stock projections, another attention core, or the
stock statement (``xla``, always last): a row that cannot engage steps aside BY NAME inside the provider and the next measured arm serves. No kit-side
row table, length row or named-row word: ``MODEL_OPT_LEVERS_OFF=<row>`` / ``=pallas`` (the provider's words) and this kit's ``L10`` are the ablation.

L19 (``AF2IG_OPT_TRIATTN_CORE_DTYPE=bf16``, composed by fast / big) is this module's precision lever: the module is handed to the provider in
bfloat16 (activations and parameters cast at the module boundary, the output cast back to float32) so the provider's bf16 cell serves it; ``fp32`` /
unset keeps float32 (TF32-class products). Evidence: the fused_triattn census (served / fallback by reason, ``providers=<arm>=<calls>`` in the
provider's own names, ``bridge=on:<arm>`` when an FFI-bridge arm served else ``none``, ``+core_bfloat16`` under L19, ``word=<tier>``).

Numerics: not bitwise with the stock module (fused softmax / re-associated products; TF32-class products under float32) — tier 2, inside the
engine's identity band; exact binds nothing here (stock XLA by name).
"""
from __future__ import annotations

import os
from typing import Optional

from . import _fused

TAG = _fused.TAG
LEVER = "L10"
FLAG = "-fused_triattn"
NAME = "F1.fpf_pallas_triattn"                    # the Ledger's name= (the tree's kernel-lever naming: F1.<kernel>)
IMPL = "opt_core.pallas.serve.triangle_attention_block"   # bound BY TIER WORD through the provider — the module row per cell (fpf_block | proj+triattn_xla | proj+<core> | xla) is the provider's
FACE = "triangle_attention_block"
FORM = "af2"
KEY_MASK = "by_line"                              # AlphaFold masks the keys of line b with pair_mask[b, k] (the attended line's own row) in both orientations
ENDING_BIAS_TRANSPOSED = True                     # AlphaFold layout of the ending node's pair bias
F32, DTYPE = "float32", "dtype"                    # the one activation dtype this kit's blocks serve, and the fallback reason word for any other (declared: named on the LEVER line as fallback_by, the run keeps its own exit code)


def _dtype_word(dt) -> str:
    return getattr(dt, "name", None) or str(dt)


EXPECTED_FALLBACKS = (DTYPE,)                        # a healthy af2ig run refuses nothing: every TriangleAttention call of the model is f32, square, C in {64, 128}, H = 4, D in {16, 32}
NO_GATING = "no_gating"                           # this module's own refusals (never met by AlphaFold's configs; named so a foreign config is counted, not mis-served)
PROJECTION_DIMS = "projection_dims"

ENV_CORE_DTYPE = "AF2IG_OPT_TRIATTN_CORE_DTYPE"          # L19: the module's operand dtype word — unset | fp32 = float32; bf16 = bfloat16 (fast / big place it; MODEL_OPT_LEVERS_OFF=L19 or a caller's fp32 keeps float32)
CORE_DTYPE_WORDS = {"": "float32", "fp32": "float32", "f32": "float32", "float32": "float32", "bf16": "bfloat16", "bfloat16": "bfloat16"}
BRIDGE = "triattn_xla"                                   # the provider arm word of the XLA-FFI bridge rows (proj+triattn_xla…): named on the census as bridge=on:<arm> when one served
_STATE = {"orig": None, "patches": None, "serve": None, "ledger": None, "probe": None, "bridge": None, "core_dtype_applied": None}


def _dtype_word(dt) -> str:
    return getattr(dt, "name", None) or str(dt)


def core_dtype(environ=None) -> str:
    """``float32`` | ``bfloat16``: the bridge core's operand dtype from ``AF2IG_OPT_TRIATTN_CORE_DTYPE`` (an unknown word = float32, named in the probe as ``core_dtype_note``)."""
    import os as _os
    environ = _os.environ if environ is None else environ
    return CORE_DTYPE_WORDS.get(str(environ.get(ENV_CORE_DTYPE, "") or "").strip().lower(), "float32")


def core_dtype_note(environ=None) -> Optional[str]:
    import os as _os
    environ = _os.environ if environ is None else environ
    raw = str(environ.get(ENV_CORE_DTYPE, "") or "").strip().lower()
    return None if raw in CORE_DTYPE_WORDS else f"unknown_word:{raw}->float32"


def _serve():
    if _STATE["serve"] is None:
        _STATE["serve"] = _fused.serve("triattn")                         # the serve layer's probe (jax / Pallas / gpu / tile rows): a MODE refuses by name without it
    return _STATE["serve"]


def _ledger():
    return _fused.ledger(_STATE, NAME, IMPL, EXPECTED_FALLBACKS)


def setup(environ=None) -> dict:
    """Resolve the lever for this process and install the route: the serve layer's ``require()`` (a named ``Refusal`` on a missing jax / Pallas, a CPU
    backend, a GPU without tile rows makes the MODE refuse by name, :func:`_fused.refuse`), the provider importable, the bridge's FFI targets registered
    (so a stored program that launches them LOADS in this process), the module rebound. Returns the probe ``{ok, jax, backend, cc, tiles, precision,
    key_mask, word, core_dtype, bridge}``."""
    from . import modes
    S = _serve()
    try:
        probe = dict(S.require())
        _fused.f32_rows(S, "triattn")
    except S.Refusal as r:
        word = _fused.refusal_word(S, r, "triattn")
        if word is None:
            raise
        _fused.refuse(LEVER, FLAG, word, r, _ledger().line(TAG, state="skipped", reason=word, lever=LEVER, flag=FLAG, dtype="f32"))
    _fused.provider()                                                    # core_missing is named here, before any design
    _STATE["bridge"] = bridge_probe(probe.get("cc"), environ, register=True)
    _STATE["core_dtype_applied"] = core_dtype(environ)
    _ledger()
    install()
    probe.update(precision=modes.F32_PRODUCTS, key_mask=KEY_MASK, word=modes.tier_word(environ), core_dtype=_STATE["core_dtype_applied"], bridge=dict(_STATE["bridge"]))
    _STATE["probe"] = probe
    return probe


def bridge_probe(cc, environ=None, register: bool = False) -> dict:
    """The XLA-FFI bridge's PROCESS facts (no row decision — that is the provider's, per call): ``{launcher: registered | <why not>, portable, cc,
    core_dtype, core_dtype_note}``. ``register`` loads the core's launcher and registers its FFI targets so a stored program that calls them loads here."""
    environ = os.environ if environ is None else environ
    reg = bridge_register() if register else "not_requested"
    portable, why = bridge_programs_portable()
    return {"launcher": reg, "portable": bool(portable), "process_local": (None if portable else why), "cc": (str(cc) if cc is not None else None),
            "core_dtype": core_dtype(environ), "core_dtype_note": core_dtype_note(environ)}


def bridge_on() -> bool:
    """True once a bridge arm has served a call in this process (the provider chose it for a cell)."""
    return any(BRIDGE in arm for arm in _fused.served_arms(FACE))


PROCESS_LOCAL = "triattn_xla_ffi_specs"     # the L13 word for programs that launch the bridge through per-process launch specs (an old core): traced in every process, never stored
SELF_DESCRIBING_TARGET = "triattn_xla_run"  # the core's self-describing launch target: programs that call it deserialize AND run in another process


def bridge_programs_portable(TX=None) -> tuple:
    """(True, ``portable:<target>@<triattn_xla version>``) when the installed core launches the bridge through the self-describing FFI target
    (``triattn_xla._launch.TARGET_RUN``: opt_core >= 0.5.39.0) — its programs deserialize AND run in another process —, else (False, :data:`PROCESS_LOCAL`)
    (a 1.0-1.2 launcher: launch specs registered in the tracing process, ``unknown spec id`` elsewhere; or no core). Arrays-free; imports the core's launcher
    module only (no library is loaded here — :func:`bridge_register` does that at setup)."""
    try:
        if TX is None:
            from opt_core.kernels import triattn_xla as TX
        launch = getattr(TX, "_launch", None)
        if launch is None:
            import importlib
            launch = importlib.import_module(TX.__name__ + "._launch")
        target = getattr(launch, "TARGET_RUN", None)
        ver = str(getattr(TX, "__version__", "") or "?")
    except Exception:  # noqa: BLE001 — no core / a core without the launcher module: the old word stands
        return False, PROCESS_LOCAL
    if target:
        return True, f"portable:{target}@{ver}"
    return False, PROCESS_LOCAL


def process_local_word() -> Optional[str]:
    """The L13 store's word for this process's programs: None (portable: the installed core launches the bridge through its self-describing target) or
    :data:`PROCESS_LOCAL` (an older core with per-process launch specs — programs traced with L10 are never stored or loaded, by name)."""
    if _STATE.get("orig") is None and _STATE.get("probe") is None:      # L10 not installed in this process: nothing of the bridge can be in a program
        return None
    portable, why = bridge_programs_portable()
    return None if portable else PROCESS_LOCAL


def programs_word() -> str:
    """``portable`` | ``process_local:<why>`` for the census."""
    portable, why = bridge_programs_portable()
    return "portable" if portable else f"process_local:{why}"


def bridge_register(TX=None) -> str:
    """Load the core's launcher and register its XLA-FFI targets in this process (``triattn_xla._launch.load()`` — what the first ``triangle_attention``
    call would do, done at setup so ``deserialize_and_load`` of a stored bridge program finds the target): ``registered`` | ``<ExceptionType>:<text>``."""
    try:
        if TX is None:
            from opt_core.kernels import triattn_xla as TX
        loader = getattr(TX, "load_launcher", None)
        if loader is None:
            from opt_core.kernels.triattn_xla import _launch
            loader = _launch.load
        loader()
        return "registered"
    except Exception as e:  # noqa: BLE001 — a launcher that does not load / register (jaxlib without a matching FFI build, dlopen failure): the bridge steps aside by name
        return f"{type(e).__name__}:{str(e)[:160]}"


def bridge_facts(cc: str = "", environ=None) -> dict:
    """The bridge's process facts without registering anything: see :func:`bridge_probe`."""
    return bridge_probe(cc or None, environ, register=False)


def install() -> None:
    """Rebind ``alphafold.model.modules.TriangleAttention.__call__`` to :func:`fused_call` WITH haiku's method wrapper (the module's name scope:
    the tree's shared haiku recipe ``opt_core.mem.rowpair_jax.haiku.rebind``); idempotent; :func:`uninstall` restores."""
    from alphafold.model import modules
    _fused.rebind(_STATE, modules.TriangleAttention, fused_call, "triattn")


def uninstall() -> None:
    _fused.unbind(_STATE)


def reason_for(config, pair_act, pair_mask) -> Optional[str]:
    """Why a TriangleAttention call stays on the stock body (a reason name), or None when the block takes it."""
    S = _serve()
    C = int(pair_act.shape[-1])
    if _dtype_word(pair_act.dtype) != F32:                                  # this kit runs the float32 blocks only — an activation of another dtype (a bf16 experiment upstream of the pair stack) stays on the stock body, COUNTED by name (`dtype`), never handed to the f32 kernel (a Pallas LoweringError before 0.6.0)
        return DTYPE
    if not bool(config.gating):
        return NO_GATING
    if int(config.get("key_dim", C)) != C or int(config.get("value_dim", C)) != C:
        return PROJECTION_DIMS
    return S.served_reason("triattn", pair_act.shape, pair_act.dtype, pair_mask.shape, num_head=int(config.num_head))


def _shape_key(pair_act, num_head: int) -> str:
    n, _, c = (int(s) for s in pair_act.shape)
    return f"N{n}xC{c}xH{num_head}"


def fused_call(self, pair_act, pair_mask, is_training=False):
    """``TriangleAttention.__call__`` through the provider: same arguments, same parameter names, same output shape ``[N, N, C]``."""
    import haiku as hk
    import jax.numpy as jnp
    from . import modes
    _serve()
    led = _ledger()
    c = self.config
    assert len(pair_act.shape) == 3 and len(pair_mask.shape) == 2, (pair_act.shape, pair_mask.shape)
    assert c.orientation in ("per_row", "per_column"), c.orientation
    why = reason_for(c, pair_act, pair_mask)
    if why is not None:
        led.fallback(why)
        return _STATE["orig"](self, pair_act, pair_mask, is_training)
    C = int(pair_act.shape[-1]); H = int(c.num_head); D = C // H
    dt = pair_act.dtype
    zeros, ones = hk.initializers.Constant(0.0), hk.initializers.Constant(1.0)   # placeholders: under RunModel.apply every array comes from the pinned checkpoint
    Reader = _fused.param_reader()
    ln = Reader(name="query_norm")({"scale": ([C], ones), "offset": ([C], zeros)}, dt)                 # = hk.LayerNorm(name='query_norm')'s parameters
    bias_w = hk.get_parameter("feat_2d_weights", [C, H], dt, init=zeros)
    at = Reader(name="attention")({"query_w": ([C, H, D], zeros), "key_w": ([C, H, D], zeros), "value_w": ([C, H, D], zeros), "gating_w": ([C, H, D], zeros),
                                   "gating_b": ([H, D], ones), "output_w": ([H, D, C], zeros), "output_b": ([C], zeros)}, dt)   # = the Attention module's parameters
    params = dict(ln_scale=ln["scale"], ln_offset=ln["offset"], q_w=at["query_w"], k_w=at["key_w"], v_w=at["value_w"], bias_w=bias_w,
                  gate_w=at["gating_w"], out_w=at["output_w"], gate_b=at["gating_b"], out_b=at["output_b"])          # the provider's math layout (attn_params keywords)
    core_dt = _STATE.get("core_dtype_applied") or core_dtype()
    act_in, mask_in = pair_act, pair_mask.astype(dt)
    if core_dt == "bfloat16":                                                                          # L19: the module handed to the provider in bfloat16 (its bf16 cell serves it); output back to float32
        act_in = pair_act.astype(jnp.bfloat16); mask_in = pair_mask.astype(jnp.bfloat16)
        params = {k_: (v_.astype(jnp.bfloat16) if v_ is not None else None) for k_, v_ in params.items()}
    P, PS = _fused.provider()
    word = modes.tier_word()
    try:                                                                # the provider serves the tier word's fastest measured row for this cell; a row that cannot engage steps aside BY NAME
        out = PS.triangle_attention_block(act_in, mask_in, params, orientation=c.orientation, form=FORM, word=word, input_precision=modes.F32_PRODUCTS,
                                          ending_bias_transposed=ENDING_BIAS_TRANSPOSED, key_mask=KEY_MASK, pad=True)   # inside it and the next measured arm serves ('xla' = the stock statement last)
    except P.Refusal as r:                                              # refused by name after the parameters were read (levers-off words, an unknown word): counted and raised — the stock
        led.error(r)                                                    # body's submodule names are taken in this call, a silent stock run is not possible here
        raise RuntimeError(f"af2ig_opt.triattn: provider refused by name: {getattr(r, 'kind', type(r).__name__)} ({r}); shape {_shape_key(pair_act, H)} orientation {c.orientation} word {word}") from None
    led.serve(_shape_key(pair_act, H), first={"orientation": c.orientation, "word": word, "core_dtype": core_dt} if led.first is None else None)
    return out.astype(dt)


def census() -> dict:
    """The Ledger's fields for the driver's ``fused_triattn`` timer record: ``{name, impl, origin, state, served, fallback, fallback_by, errors,
    shapes, partial, calls, first}`` + ``precision``, ``key_mask``, ``word``, ``providers`` (the provider's served arms), ``bridge``, ``programs``."""
    from . import modes
    out = _fused.census(_ledger(), _STATE["probe"], precision=modes.F32_PRODUCTS, key_mask=KEY_MASK)
    out.update(word=modes.tier_word(), providers=_fused.served_arms(FACE), bridge=bridge_word(), bridge_report=bridge_report(), programs=programs_word(),
               core_dtype=_STATE.get("core_dtype_applied") or core_dtype())
    return out


def bridge_word() -> str:
    """``on:<arm>[+core_bfloat16]`` when an FFI-bridge arm served a call in this process, ``none[+core_bfloat16]`` otherwise (the provider chose other rows)."""
    arms = _fused.served_arms(FACE)
    br = sorted(a for a in arms if BRIDGE in a)
    core = "" if (_STATE.get("core_dtype_applied") or core_dtype()) == "float32" else "+core_bfloat16"
    return (f"on:{br[0]}{core}" if br else f"none{core}")


def bridge_report() -> Optional[dict]:
    """The bridge core's own account (``triattn_xla.report()``: launcher, cubins loaded, served per row, last refusal) — None when it is not importable."""
    try:
        from opt_core.kernels import triattn_xla as TX
        r = dict(TX.report())
        return {k: r.get(k) for k in ("served", "launcher", "cubins", "last_refusal", "ffi_api") if k in r} or r
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:120]}


def providers_word() -> str:
    """``<arm>=<calls>,…`` in the provider's own arm names, or ``none``."""
    return _fused.arms_word(_fused.served_arms(FACE))


def line() -> str:
    """The per-lever evidence line of this process (the core's LEVER grammar, from the Ledger) + the tier word and the provider arms that served."""
    from . import modes
    p = _STATE["probe"] or {}
    return _ledger().line(TAG, lever=LEVER, flag=FLAG, dtype="f32", precision=modes.F32_PRODUCTS, tiles=p.get("tiles"), word=modes.tier_word(), providers=providers_word(), bridge=bridge_word())


def emit() -> str:
    """Print :func:`line` once on stderr (the driver's exit)."""
    return _fused.emit(line())
