"""L11 (``-fused_trimul``): AlphaFold's ``TriangleMultiplication`` (outgoing and incoming — the 48 Evoformer blocks' and the template pair
stack's) computed by the model-opt tree's fused triangle-multiplication block, ``opt_core.kernels.fpf_pallas_serve.trimul_block``: a Pallas
prologue (input LayerNorm, the left|right projections and their sigmoid gates, the mask), ONE batched TF32 GEMM for the triangle contraction, and
a Pallas epilogue (centre LayerNorm, output projection, output gate) — float32 in and out, float32 accumulation, tensor-core TF32 products
(the tier's float32 product class, modes.F32_PRODUCTS, handed to the provider once). The stock body it replaces (``alphafold/model/modules.py`` ``TriangleMultiplication.__call__``, the
unfused 8-Linear layout of the monomer weights) is left untouched and still serves every call the block refuses.

How it engages: :func:`setup` (the driver, before any design) resolves the block for this process — a core without it, a jax below the serve
layer's floor, a CPU backend or a GPU without a tile table stops the run with the reason named, never a silent stock run — and rebinds
``modules.TriangleMultiplication.__call__`` to :func:`fused_call` with haiku's own method wrapper (``af2ig_opt._fused.rebind``: the module keeps
its name scope; the vendored ``alphafold`` files are not edited). At trace time each call is routed by rule: the serve layer's ``served_reason``
(dtype, square pair, channel count), plus this module's one condition (``num_intermediate_channel`` = channels — true in every AlphaFold config);
a refused call runs the stock body and is COUNTED by reason, a served call is counted with its shape. The parameters are read with the stock haiku
names (``layer_norm_input``, ``left_projection``, ``right_projection``, ``left_gate``, ``right_gate``, ``center_layer_norm``,
``output_projection``, ``gating_linear``: ``weights``/``bias``, ``scale``/``offset``) so the pinned checkpoint loads unchanged. Pair sizes
off the kernel tile are padded and masked per call inside the block; the model runs at the design's own length. The memory line's row-chunked
TriangleMultiplication (``-trimul_chunk``, ``af2ig_opt.pairstack``) subclasses the stock class and dispatches to ``TriangleMultiplication.__call__``
below its size floor — i.e. to this block when both are on — and to the chunked body above it.

Numerics: not bitwise with the stock body (fused LayerNorm/projection arithmetic, TF32 products with float32 accumulation = the class of the
stock's default-precision XLA einsums) — a tier-2 lever, composed on the fast line after ``-fused_triattn``. Deterministic run to run.

Evidence: ``fused_trimul`` timer records (the Ledger's fields per compiled length and at exit) and the tree's per-lever line (:func:`emit`);
``af2ig_opt.stack.applied``: ``served >= 1`` and no fallback reason outside :data:`EXPECTED_FALLBACKS` = the lever is on; 0 calls served AND 0 fallbacks
with the pairstack census counting ``trimul_chunk`` engaged traces = superseded above the memory line's floor (evidenced by that census, not a
partial); anything else is a partial activation.
"""
from typing import Optional

from . import _fused

TAG = _fused.TAG
LEVER = "L11"
FLAG = "-fused_trimul"
NAME = "F2.fpf_pallas_trimul"
IMPL = "opt_core.pallas.serve.triangle_multiplication"   # bound BY TIER WORD through the provider (rows per cell are the provider's: fpf_trimul | cd_trimul | … | xla)
FACE = "triangle_multiplication"
FORM = "af2"
F32, DTYPE = "float32", "dtype"                    # the one activation dtype this kit's blocks serve, and the fallback reason word for any other (declared: named on the LEVER line as fallback_by, the run keeps its own exit code)


def _dtype_word(dt) -> str:
    return getattr(dt, "name", None) or str(dt)


EXPECTED_FALLBACKS = (DTYPE,)                        # a healthy af2ig run refuses nothing: every TriangleMultiplication call is f32, square, C = Ci in {64, 128}
INTERMEDIATE_DIMS = "intermediate_dims"           # this module's own refusal (num_intermediate_channel != channels; never met by AlphaFold's configs)
EQUATIONS = {"ikc,jkc->ijc": "outgoing", "kjc,kic->ijc": "incoming"}
UNKNOWN_EQUATION = "unknown_equation"

_STATE = {"orig": None, "patches": None, "serve": None, "ledger": None, "probe": None}


def _serve():
    if _STATE["serve"] is None:
        _STATE["serve"] = _fused.serve("trimul")                          # a pinned core without the float32 block: named, before any design
    return _STATE["serve"]


def _ledger():
    return _fused.ledger(_STATE, NAME, IMPL, EXPECTED_FALLBACKS)


def setup(environ=None) -> dict:
    """Resolve the block for this process and install the route (the serve layer's ``require()``, the precision word validated, :func:`install`).
    Returns the probe ``{ok, jax, backend, cc, own_table, tiles, precision}``."""
    S = _serve()
    try:
        probe = dict(S.require())
        _fused.f32_rows(S, "trimul")                                        # the float32 rows this block reads at trace time: refused here, by name, when the table has none
    except S.Refusal as r:                                               # the block cannot run in this process (no tile rows for the part, jax below the floor, Pallas / the kernel absent, a non-gpu
        word = _fused.refusal_word(S, r, "trimul")                          # backend): the MODE refuses by name — never a run under the mode's name without this lever
        if word is None:
            raise
        _fused.refuse(LEVER, FLAG, word, r, _ledger().line(TAG, state="skipped", reason=word, lever=LEVER, flag=FLAG, dtype="f32"))
    _ledger()
    install()
    probe.update(precision=_precision_word(), word=_word())
    _STATE["probe"] = probe
    return probe


def install() -> None:
    """Rebind ``alphafold.model.modules.TriangleMultiplication.__call__`` to :func:`fused_call` (haiku method wrapper kept); idempotent."""
    from alphafold.model import modules
    _fused.rebind(_STATE, modules.TriangleMultiplication, fused_call, "fused_trimul")


def uninstall() -> None:
    _fused.unbind(_STATE)


def reason_for(config, act, mask) -> Optional[str]:
    """Why a TriangleMultiplication call stays on the stock body (a reason name), or None when the block takes it."""
    S = _serve()
    C = int(act.shape[-1])
    if _dtype_word(act.dtype) != F32:                                  # this kit runs the float32 blocks only — an activation of another dtype (a bfloat16 activation produced upstream of the pair stack) stays on the stock body, COUNTED by name (`dtype`), never handed to the f32 kernel (a Pallas LoweringError before 0.6.0)
        return DTYPE
    if config.equation not in EQUATIONS:
        return UNKNOWN_EQUATION
    if int(config.num_intermediate_channel) != C:
        return INTERMEDIATE_DIMS
    return S.served_reason("trimul", act.shape, act.dtype, mask.shape)


def _shape_key(act) -> str:
    n, _, c = (int(s) for s in act.shape)
    return f"N{n}xC{c}"


def fused_call(self, act, mask, is_training=True):
    """``TriangleMultiplication.__call__`` on the fused block: same arguments, same parameter names, same output shape ``[N, N, C]``."""
    import haiku as hk
    S = _serve()
    led = _ledger()
    c = self.config
    assert len(act.shape) == 3 and len(mask.shape) == 2, (act.shape, mask.shape)
    why = reason_for(c, act, mask)
    if why is not None:
        led.fallback(why)
        return _STATE["orig"](self, act, mask, is_training)
    C = int(act.shape[-1]); Ci = int(c.num_intermediate_channel)
    dt = act.dtype
    zeros, ones = hk.initializers.Constant(0.0), hk.initializers.Constant(1.0)   # placeholders: under RunModel.apply every array comes from the pinned checkpoint
    Reader = _fused.param_reader()
    ln_in = Reader(name="layer_norm_input")({"scale": ([C], ones), "offset": ([C], zeros)}, dt)
    lin = {name: Reader(name=name)({"weights": ([C, Ci], zeros), "bias": ([Ci], ones if name.endswith("gate") else zeros)}, dt)
           for name in ("left_projection", "right_projection", "left_gate", "right_gate")}
    ln_c = Reader(name="center_layer_norm")({"scale": ([Ci], ones), "offset": ([Ci], zeros)}, dt)
    out_p = Reader(name="output_projection")({"weights": ([Ci, C], zeros), "bias": ([C], zeros)}, dt)
    gate = Reader(name="gating_linear")({"weights": ([C, C], zeros), "bias": ([C], ones)}, dt)
    params = dict(ln_in_scale=ln_in["scale"], ln_in_offset=ln_in["offset"],
                  left_w=lin["left_projection"]["weights"], left_b=lin["left_projection"]["bias"],
                  right_w=lin["right_projection"]["weights"], right_b=lin["right_projection"]["bias"],
                  left_gate_w=lin["left_gate"]["weights"], left_gate_b=lin["left_gate"]["bias"],
                  right_gate_w=lin["right_gate"]["weights"], right_gate_b=lin["right_gate"]["bias"],
                  ln_c_scale=ln_c["scale"], ln_c_offset=ln_c["offset"],
                  out_w=out_p["weights"], out_b=out_p["bias"], gate_w=gate["weights"], gate_b=gate["bias"])   # the provider's math layout (serve.TRIMUL_KEYS / TRIMUL_BIAS_KEYS)
    P, PS = _fused.provider()
    from . import modes
    word = modes.tier_word()
    try:                                                                # the provider serves the tier word's fastest measured row for this cell (cc, dtype, family, N); a row that cannot
        out = PS.triangle_multiplication(act, mask.astype(dt), params, equation=EQUATIONS[c.equation], form=FORM, word=word, input_precision=modes.F32_PRODUCTS)   # engage steps aside BY NAME inside it; the next measured arm serves ('xla' = the stock statement last)
    except P.Refusal as r:                                              # refused by name after the parameters were read (levers-off words, an unknown word): counted and raised — the stock
        led.error(r)                                                    # body's submodule names are taken in this call, a silent stock run is not possible here
        raise RuntimeError(f"af2ig_opt.fused_trimul: provider refused by name: {getattr(r, 'kind', type(r).__name__)} ({r}); shape {_shape_key(act)} equation {c.equation} word {word}") from None
    led.serve(_shape_key(act), first={"equation": EQUATIONS[c.equation]} if led.first is None else None)
    return out


def census() -> dict:
    """The Ledger's fields for the driver's ``fused_trimul`` timer record + ``precision`` and the probe's ``cc`` / ``tiles`` / ``jax``."""
    out = _fused.census(_ledger(), _STATE["probe"], precision=_precision_word())
    out.update(providers=_fused.served_arms(FACE), word=_word())
    return out


def line() -> str:
    p = _STATE["probe"] or {}
    return _ledger().line(TAG, lever=LEVER, flag=FLAG, dtype="f32", precision=_precision_word(), tiles=p.get("tiles"), word=_word(), providers=providers_word())


def emit() -> str:
    return _fused.emit(line())


def _word() -> str:
    from . import modes
    return modes.tier_word()


def _precision_word() -> str:
    """The float32 product class the tier hands the provider (modes.F32_PRODUCTS) — one word for every adapter."""
    from . import modes
    return modes.F32_PRODUCTS


def providers_word() -> str:
    """``<arm>=<calls>,…`` — the provider rows that served this process's TriangleMultiplication calls (the provider's own names), or ``none``."""
    return _fused.arms_word(_fused.served_arms(FACE))
