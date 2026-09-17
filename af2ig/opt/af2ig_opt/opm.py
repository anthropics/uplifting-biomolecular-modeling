"""L12 (``-opm_reassoc``): AlphaFold's ``OuterProductMean`` (Jumper et al. 2021, Suppl. Alg. 10 — one per Evoformer block, 48, and per
extra-MSA block, 4) computed RE-ASSOCIATED by this kit adapter (pure JAX, float32; no kernel, no tile table, no card rule).

Stock (``alphafold/model/modules.py`` ``OuterProductMean.__call__``) contracts the two 32-channel projections into the outer product first,
``P[j,c,e,i] = sum_s L[s,i,c] * R[s,j,e]`` — a [N, 32, 32, rows] float32 intermediate per 128-row chunk of the left projection (``chunk_size``,
``low_memory``): N^2 * 4 KiB written and re-read through HBM per call — and then applies the output weight, ``out[i,j,f] = sum_{c,e} P * W[c,e,f]``
(2 * N^2 * 1024 * 128 FLOPs). At this engine's MSA depth (S = 5 cluster rows in the Evoformer stack and 5 extra-MSA rows: STOCK.md; the served
shapes below record the S the model actually ran, per stack) the other association is ~6.6x cheaper and has no N^2-sized intermediate: fold the
weight into the right projection once per MSA row, ``T[s,c,j,f] = sum_e R[s,j,e] * W[c,e,f]`` ([S, 32, N, c_z] float32 = S * N * 16 KiB), then ONE
GEMM with K = 32 * S, ``out[i,(j,f)] = sum_{(s,c)} L[i,(s,c)] * T[(s,c),(j,f)]``, + ``output_b``; the mask product, the normalisation
``norm[i,j] = sum_s mask[s,i] * mask[s,j]`` and ``epsilon = 1e-3`` are stock's statements unchanged (:func:`reassociated` / :func:`stock_order` are
the two orderings side by side; ``tests/test_opm_reassoc.py`` measures both against a float64 reference). FLOPs per call: 2 * N^2 * c_z * 32 * S
against stock's 2 * N^2 * (1024 * S + 1024 * c_z).

Numerics: a re-association of the same float32 sums — tier 2, not bitwise with the stock body; both orderings run XLA's default-precision float32
products (TF32-class on cc >= 8.0 NVIDIA parts, the class of every stock einsum of the model: the driver prints ``jax_default_matmul_precision``
at setup). Deterministic run to run. Composed on the fast line after ``-fused_trimul`` and on big's fast base (it LOWERS the per-call working
set: no [rows, N, 1024] chunk intermediate).

How it engages: :func:`setup` (the driver's ``-opm_reassoc`` hook, before any design) rebinds ``modules.OuterProductMean.__call__`` to
:func:`opm_call` WITH haiku's method wrapper (``af2ig_opt._fused.rebind`` over the tree's ``opt_core.mem.rowpair_jax.haiku``: the module keeps its
name scope, and the body below creates its parameters under the stock names ``layer_norm_input``, ``left_projection``, ``right_projection``,
``output_w``, ``output_b`` — the pinned checkpoint loads unchanged; the vendored ``alphafold`` files are not edited). Every call is counted once
(``opt_core.counters.Ledger``): served, with its shape ``S<msa rows>xN<residues>xM<c_m>``, or — for an input the body does not expect (not
[S, N, c_m] activations with an [S, N] mask: never produced by AlphaFold's monomer model) — left on the stock body and counted by reason
(:data:`BAD_INPUT`; declared set empty, so such a call makes the run partial, never a silent stock run). Stock's ``chunk_size`` has no
counterpart here (there is no N^2 * 1024 intermediate to bound); the sub-batch lever (L9, ``global_config.subbatch_size``) does not touch this
module in either body.

Evidence: ``opm_reassoc`` timer records (the Ledger's fields per compiled length and at exit: served / fallback / fallback_by / shapes) and
the per-lever line on stderr (:func:`emit`); ``af2ig_opt.stack.applied``: ``served >= 1`` and no fallback = the lever is on.
"""
from typing import Optional

from . import _fused

TAG = _fused.TAG
LEVER = "L12"
FLAG = "-opm_reassoc"
NAME = "LOCAL.af2ig.opm_reassoc"                 # the Ledger's name (the adapter's own LEVER line); registry.L12 carries the same strategy id
IMPL = "opm_reassoc_jax"                         # pure JAX in this kit (origin=kit)
ORIGIN = "kit"
PRECISION = "default"                            # the products' precision word: XLA's default for float32 (no precision= is passed, exactly as stock's einsums)
EXPECTED_FALLBACKS = ()                          # a healthy af2ig run refuses nothing: every OuterProductMean call is [S, N, c_m] with an [S, N] mask
BAD_INPUT = "unexpected_input_rank"              # the one refusal: activations / mask not of the ranks the body expects -> the stock body, counted
EPSILON = 1e-3                                   # stock's (modules.py OuterProductMean: epsilon = 1e-3)

_STATE = {"orig": None, "patches": None, "ledger": None, "probe": None}


def _ledger():
    if _STATE["ledger"] is None:
        from opt_core.counters import Ledger
        _STATE["ledger"] = Ledger(NAME, impl=IMPL, origin=ORIGIN, expected=tuple(EXPECTED_FALLBACKS))
    return _STATE["ledger"]


def precision_state() -> dict:
    """The numerics facts the driver prints at setup: jax version, backend, ``jax_default_matmul_precision`` (unset = XLA's default: TF32-class
    float32 products on cc >= 8.0 parts — the class stock's einsums run in), the model dtype."""
    import jax
    try:
        word = jax.config.jax_default_matmul_precision
    except Exception:  # noqa: BLE001 — a jax without the option: the backend default, unnamed
        word = None
    return {"jax": getattr(jax, "__version__", None), "backend": jax.default_backend(), "matmul_precision": str(word) if word else PRECISION, "dtype": "float32"}


def setup(environ=None) -> dict:
    """Install the route (:func:`install`) and return the probe ``{jax, backend, matmul_precision, dtype, impl, lever}``. Nothing card-specific
    to resolve: the body is plain XLA ops on any backend."""
    probe = precision_state()
    _ledger()
    install()
    probe.update(impl=IMPL, lever=LEVER)
    _STATE["probe"] = probe
    return probe


def install() -> None:
    """Rebind ``alphafold.model.modules.OuterProductMean.__call__`` to :func:`opm_call` (haiku method wrapper kept); idempotent."""
    from alphafold.model import modules
    _fused.rebind(_STATE, modules.OuterProductMean, opm_call, "opm_reassoc")


def uninstall() -> None:
    _fused.unbind(_STATE)


def reason_for(act, mask) -> Optional[str]:
    """Why an OuterProductMean call stays on the stock body (a reason name), or None when the re-associated body takes it."""
    if len(act.shape) != 3 or len(mask.shape) != 2 or tuple(int(d) for d in mask.shape) != tuple(int(d) for d in act.shape[:2]):
        return BAD_INPUT
    return None


def _shape_key(act) -> str:
    s, n, m = (int(d) for d in act.shape)
    return f"S{s}xN{n}xM{m}"


def reassociated(left, right, output_w, output_b, xp):
    """``out[i,j,f] = sum_{s,c,e} left[s,i,c] * right[s,j,e] * output_w[c,e,f] + output_b[f]`` contracted WEIGHT FIRST: ``T[s,c,j,f] =
    sum_e right[s,j,e] * W[c,e,f]`` ([S, C, N, F]), then one [N, S*C] x [S*C, N*F] product, reshaped [N, N, F] (a view). ``xp`` is
    ``jax.numpy`` in the model and ``numpy`` in the CPU test; no ``precision=`` is passed (XLA's default, as stock)."""
    S, N, C = (int(d) for d in left.shape)
    F = int(output_w.shape[-1])
    t = xp.einsum("sje,cef->scjf", right, output_w).reshape(S * C, N * F)      # the weight folded into the right projection: [(s,c), (j,f)]
    lf = xp.transpose(left, (1, 0, 2)).reshape(N, S * C)                       # [i, (s,c)]
    return xp.matmul(lf, t).reshape(N, N, F) + output_b


def stock_order(left, right, output_w, output_b, xp):
    """Stock's association (``OuterProductMean`` ``compute_chunk`` over every row at once): the [N, C, C, N] outer product, then the weight."""
    act = xp.einsum("acb,ade->dceb", xp.transpose(left, (0, 2, 1)), right)
    act = xp.einsum("dceb,cef->dbf", act, output_w) + output_b
    return xp.transpose(act, (1, 0, 2))


def opm_call(self, act, mask, is_training=True):
    """``OuterProductMean.__call__`` re-associated: same arguments ([S, N, c_m] MSA activations, [S, N] mask), same parameter names, same
    output [N, N, c_z]; the LayerNorm, the two projections, the mask product, the normalisation and epsilon are stock's statements."""
    import haiku as hk
    import jax.numpy as jnp
    from alphafold.model import common_modules
    led = _ledger()
    why = reason_for(act, mask)
    if why is not None:
        led.fallback(why)
        return _STATE["orig"](self, act, mask, is_training)
    gc, c = self.global_config, self.config
    key = _shape_key(act)
    mask = mask[..., None]
    act = hk.LayerNorm([-1], True, True, name="layer_norm_input")(act)
    left_act = mask * common_modules.Linear(c.num_outer_channel, initializer="linear", name="left_projection")(act)
    right_act = mask * common_modules.Linear(c.num_outer_channel, initializer="linear", name="right_projection")(act)
    init_w = hk.initializers.Constant(0.0) if gc.zero_init else hk.initializers.VarianceScaling(scale=2., mode="fan_in")
    output_w = hk.get_parameter("output_w", shape=(c.num_outer_channel, c.num_outer_channel, self.num_output_channel), init=init_w)
    output_b = hk.get_parameter("output_b", shape=(self.num_output_channel,), init=hk.initializers.Constant(0.0))
    act = reassociated(left_act, right_act, output_w, output_b, jnp)
    norm = jnp.einsum("abc,adc->bdc", mask, mask)
    act /= EPSILON + norm
    led.serve(key, first={"S": int(left_act.shape[0]), "C": int(c.num_outer_channel), "F": int(self.num_output_channel)} if led.first is None else None)
    return act


def census() -> dict:
    """The Ledger's fields for the driver's ``opm_reassoc`` timer record + the probe's precision / jax / backend."""
    out = dict(_ledger().fields())
    out.pop("facts", None); out.pop("min_tokens", None)
    p = _STATE["probe"] or {}
    out.update(precision=p.get("matmul_precision", PRECISION), jax=p.get("jax"), backend=p.get("backend"), lever=LEVER)
    return out


def line() -> str:
    p = _STATE["probe"] or {}
    return _ledger().line(TAG, lever=LEVER, flag=FLAG, dtype="f32", precision=p.get("matmul_precision", PRECISION))


def emit() -> str:
    return _fused.emit(line())
