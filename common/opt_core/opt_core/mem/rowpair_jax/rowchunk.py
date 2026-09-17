"""The single-device ROW-CHUNK of the AlphaFold-2-family ``TriangleMultiplication`` (the P=1 memory lever ``trimul_chunk`` of the AF2 kits):
the sub-layer's ``[N, N, c]`` intermediates (normalised input, projections, gates, the einsum output, the output projection and gate) exist
for ONE block of ``rows`` output rows at a time; the operand the einsum needs whole (outgoing ``ikc,jkc->ijc``: the right projection;
incoming ``kjc,kic->ijc``: the left projection) is built once, in row blocks. Same modules, same parameter names, the same per-element
arithmetic; the loops are the model library's own ``mapping.sharded_apply`` / ``mapping.inference_subbatch`` (``hk.scan`` +
``dynamic_update_slice``) — the seam its attention sub-batching already uses. ONE producer for both AF2 kits.

:func:`chunked_class(base, rows)` returns a subclass of the stock class that KEEPS THE STOCK NAME (Haiku derives module names from the class
name: the parameter tree is unchanged) and covers the two forms the AF2 model library ships:

  * the fused form (``_fused_triangle_multiplication``: one ``projection`` / ``gate`` Linear of width ``2*c_i``; ``fuse_projection_weights=True``
    in the multimer-v3-era configs) and the unfused form (``_triangle_multiplication``: ``left_/right_projection``, ``left_/right_gate``) of a
    class that dispatches on ``config.fuse_projection_weights`` — both methods are overridden;
  * the earlier class whose ``__call__`` IS the unfused body (no ``_fused_triangle_multiplication`` attribute) — ``__call__`` is overridden.

Guards (named, counted in the adapter's :class:`opt_core.mem.Ledger`): ``rows >= N`` → the stock body (``<lever>_stock_traces``); Haiku
``init`` → the stock body (initialisers cannot run inside ``hk.scan``; the kits load trained parameters); an equation other than the two
above → :class:`opt_core.mem.MemLeverRefused` AT TRACE (never a silent stock run); chunked traces count ``<lever>_chunked_traces`` and append
``{n, rows, equation, form}`` to ``shapes`` (bounded). Numerics class ``complete_contraction`` (:mod:`evidence`): every output element's
einsum has the dense length; the per-block GEMM shape differs from the whole-tensor call, so bit-exact-vs-stock is the kit's equality record
per stack, not a construction (the projections / norms / gates are row-local). Install: ``PatchSet.replace(modules, "TriangleMultiplication",
chunked_class(modules.TriangleMultiplication, rows, ledger))`` BEFORE the model is traced (the Evoformer looks the class up in the module's
globals at call time). Inside a row-sharded region (``big --n_gpu P>1``) the recipe owns ``TriangleMultiplication`` (:mod:`alphafold`): a chunked
class traced there is REFUSED BY NAME (``trimul_chunk inside a row-sharded region …``) — the kit installs one or the other.

ONE producer for the fused and the unfused body;
the model library (``alphafold.model.{modules,mapping,common_modules,utils}``) is imported inside the functions — a process
without it gets a refusal naming the lever. Standard library at import.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

from .. import Ledger, MemLeverRefused
from . import _lazy

LEVER = "trimul_chunk"
OUTGOING = "ikc,jkc->ijc"
INCOMING = "kjc,kic->ijc"
MAX_SHAPES = 64
NUMERICS_CLASS = "complete_contraction"
LIBRARY = ("alphafold.model.modules", "alphafold.model.mapping", "alphafold.model.common_modules", "alphafold.model.utils")


def stock_modules(lever: str = LEVER) -> Tuple[Any, Any, Any, Any]:
    """``(modules, mapping, common_modules, utils)`` of the AlphaFold-2 model library in this process, or a refusal naming the lever."""
    out = []
    for name in LIBRARY:
        try:
            out.append(importlib.import_module(name))
        except ImportError as e:
            raise MemLeverRefused(lever, f"{name} not importable ({e})") from None
    return tuple(out)  # type: ignore[return-value]


def form_of(base: Any, lever: str = LEVER) -> str:
    """``fused+unfused`` (a class with ``_fused_triangle_multiplication`` and ``_triangle_multiplication``) or ``call`` (``__call__`` is the unfused
    body); a class with neither shape is refused by name."""
    if callable(getattr(base, "_fused_triangle_multiplication", None)) and callable(getattr(base, "_triangle_multiplication", None)):
        return "fused+unfused"
    if callable(getattr(base, "__call__", None)) and hasattr(base, "__init__"):
        return "call"
    raise MemLeverRefused(lever, f"{getattr(base, '__name__', base)!s}: neither _fused_triangle_multiplication/_triangle_multiplication nor a __call__ body to chunk")


def check(lever: str = LEVER) -> Dict[str, str]:
    """The preconditions, by name: the library importable, ``modules.TriangleMultiplication`` present, ``mapping.sharded_apply`` /
    ``mapping.inference_subbatch`` callable, ``common_modules.Linear`` present, ``modules._layer_norm`` present for the fused form. Returns
    ``{"form": …, "layer_norm": "common_modules.LayerNorm" | "hk.LayerNorm", "library": "modules:<sha12>", "library_sha256": …}`` (the evidence line's
    ``library=`` names the exact bytes chunked) or raises :class:`MemLeverRefused`."""
    modules, mapping, common_modules, _utils = stock_modules(lever)
    base = getattr(modules, "TriangleMultiplication", None)
    if base is None:
        raise MemLeverRefused(lever, "alphafold.model.modules has no TriangleMultiplication")
    for fn in ("sharded_apply", "inference_subbatch"):
        if not callable(getattr(mapping, fn, None)):
            raise MemLeverRefused(lever, f"alphafold.model.mapping has no {fn}")
    if not callable(getattr(common_modules, "Linear", None)):
        raise MemLeverRefused(lever, "alphafold.model.common_modules has no Linear")
    form = form_of(base, lever)
    if form == "fused+unfused" and not callable(getattr(modules, "_layer_norm", None)):
        raise MemLeverRefused(lever, "alphafold.model.modules has no _layer_norm (the fused form's norm constructor)")
    import hashlib  # noqa: PLC0415
    with open(getattr(modules, "__file__", ""), "rb") as fh:
        library_sha = hashlib.sha256(fh.read()).hexdigest()
    return {"form": form, "layer_norm": "common_modules.LayerNorm" if hasattr(common_modules, "LayerNorm") else "hk.LayerNorm",
            "library": "modules:" + library_sha[:12], "library_sha256": library_sha}


def chunked_class(base: Any, rows: int, ledger: Optional[Ledger] = None, shapes: Optional[List[Dict[str, Any]]] = None, lever: str = LEVER):
    """The row-chunked subclass of the stock ``TriangleMultiplication`` ``base`` with block ``rows`` (see the module docstring). ``ledger`` counts
    ``<lever>_stock_traces`` / ``<lever>_chunked_traces``; ``shapes`` (a list the adapter owns) receives one ``{n, rows, chunked, equation, form}``
    entry per trace, up to :data:`MAX_SHAPES`."""
    rows = int(rows)
    if rows < 1:
        raise MemLeverRefused(lever, f"rows={rows}: the block must be a positive row count")
    modules, mapping, common_modules, utils = stock_modules(lever)
    hk = _lazy.haiku(lever)
    jax = _lazy.jax(lever)
    jnp = _lazy.jnp(lever)
    form = form_of(base, lever)
    ledger = ledger if ledger is not None else Ledger()
    shapes = shapes if shapes is not None else []
    unfused_norm = getattr(common_modules, "LayerNorm", None) or hk.LayerNorm     # each tree's own constructor for the unfused path's norms

    def _record(n: int, chunked: bool, equation: str, which: str) -> None:
        ledger.count(f"{lever}_{'chunked' if chunked else 'stock'}_traces")
        if len(shapes) < MAX_SHAPES:
            shapes.append({"n": int(n), "rows": rows, "chunked": bool(chunked), "equation": equation, "form": which})

    def _guard(self, act, which: str) -> Optional[str]:
        """The equation when this trace is chunked; None when the stock body runs (rows >= N, or Haiku init)."""
        n = int(act.shape[0])
        equation = str(self.config.equation).replace(" ", "")
        from . import haiku as _rp_haiku  # noqa: PLC0415
        if _rp_haiku.in_region():
            raise MemLeverRefused(lever, "trimul_chunk inside a row-sharded region (n_gpu>1): the AlphaFold-2 recipe owns TriangleMultiplication there — install one or the other")
        if rows >= n or hk.running_init():
            _record(n, False, equation, which)
            return None
        if equation not in (OUTGOING, INCOMING):
            raise MemLeverRefused(lever, f"equation {equation!r} is neither {OUTGOING!r} nor {INCOMING!r} — not chunked")
        _record(n, True, equation, which)
        return equation

    def _unfused_rows(self, act, mask3, equation: str):
        """The unfused body in row chunks: names layer_norm_input / left_projection / right_projection / left_gate / right_gate /
        center_layer_norm / output_projection / gating_linear (the stock's)."""
        c, gc = self.config, self.global_config
        outgoing = equation == OUTGOING
        layer_norm_input = unfused_norm(axis=[-1], create_scale=True, create_offset=True, name="layer_norm_input")
        left_projection = common_modules.Linear(c.num_intermediate_channel, name="left_projection")
        right_projection = common_modules.Linear(c.num_intermediate_channel, name="right_projection")
        left_gate = common_modules.Linear(c.num_intermediate_channel, bias_init=1., initializer=utils.final_init(gc), name="left_gate")
        right_gate = common_modules.Linear(c.num_intermediate_channel, bias_init=1., initializer=utils.final_init(gc), name="right_gate")
        center_layer_norm = unfused_norm(axis=[-1], create_scale=True, create_offset=True, name="center_layer_norm")
        output_channel = int(act.shape[-1])
        output_projection = common_modules.Linear(output_channel, initializer=utils.final_init(gc), name="output_projection")
        gating_linear = common_modules.Linear(output_channel, bias_init=1., initializer=utils.final_init(gc), name="gating_linear")

        def left(a, m):
            a = layer_norm_input(a)
            return (m * left_projection(a)) * jax.nn.sigmoid(left_gate(a))

        def right(a, m):
            a = layer_norm_input(a)
            return (m * right_projection(a)) * jax.nn.sigmoid(right_gate(a))

        whole = mapping.sharded_apply(right if outgoing else left, rows, in_axes=0, out_axes=0)(act, mask3)     # the whole operand, in row chunks

        def output_rows(act_op, act_rows, mask_op):                       # act_op/mask_op: the chunk's operand slice (rows i | columns i); act_rows: rows i for the gate
            if outgoing:
                out = jnp.einsum(c.equation, left(act_op, mask_op), whole)
            else:
                out = jnp.einsum(c.equation, whole, right(act_op, mask_op))
            out = output_projection(center_layer_norm(out))
            return out * jax.nn.sigmoid(gating_linear(layer_norm_input(act_rows)))

        op_axis = 0 if outgoing else 1
        return mapping.sharded_apply(output_rows, rows, in_axes=(op_axis, 0, op_axis), out_axes=0)(act, act, mask3)

    def _fused_rows(self, left_act, mask3, equation: str):
        """The fused body in row chunks: names left_norm_input / projection / gate / center_norm / output_projection / gating_linear (the stock's)."""
        c, gc = self.config, self.global_config
        ci = int(c.num_intermediate_channel)
        output_channel = int(left_act.shape[-1])
        norm_in = modules._layer_norm(axis=-1, name="left_norm_input")
        projection = common_modules.Linear(2 * ci, name="projection")
        gate = common_modules.Linear(2 * ci, name="gate", bias_init=1., initializer=utils.final_init(gc))
        center_norm = modules._layer_norm(axis=-1, name="center_norm")
        output_projection = common_modules.Linear(output_channel, initializer=utils.final_init(gc), name="output_projection")
        gating_linear = common_modules.Linear(output_channel, bias_init=1., initializer=utils.final_init(gc), name="gating_linear")

        def proj_rows(z_rows, mask_rows):                                   # [R, N, c_z], [R, N, 1] -> ([R, N, c_i], [R, N, c_i])
            a = norm_in(z_rows)
            p = mask_rows * projection(a)
            p = p * jax.nn.sigmoid(gate(a))
            return p[..., :ci], p[..., ci:]

        left_proj, right_proj = mapping.inference_subbatch(proj_rows, rows, batched_args=[left_act, mask3], nonbatched_args=[], low_memory=True)

        if equation == OUTGOING:
            def out_rows(z_rows, l_rows):                                    # rows i of the output: left[i, k] against every right[j, k]
                act = jnp.einsum(c.equation, l_rows, right_proj)
                act = output_projection(center_norm(act))
                return act * jax.nn.sigmoid(gating_linear(norm_in(z_rows)))
            return mapping.sharded_apply(out_rows, shard_size=rows, in_axes=(0, 0), out_axes=0)(left_act, left_proj)

        def out_rows_in(z_rows, r_cols):                                     # rows i of the output: right[k, i] (a column block) against every left[k, j]
            act = jnp.einsum(c.equation, left_proj, r_cols)
            act = output_projection(center_norm(act))
            return act * jax.nn.sigmoid(gating_linear(norm_in(z_rows)))
        return mapping.sharded_apply(out_rows_in, shard_size=rows, in_axes=(0, 1), out_axes=0)(left_act, right_proj)

    if form == "fused+unfused":
        class TriangleMultiplication(base):  # noqa: N801 — the stock name, on purpose (Haiku's module name = the parameter tree)
            _rowchunk_rows = rows

            @hk.transparent                                                  # as the stock methods: sub-modules stay in the module's own scope
            def _fused_triangle_multiplication(self, left_act, left_mask):
                equation = _guard(self, left_act, "fused")
                if equation is None:
                    return super()._fused_triangle_multiplication(left_act, left_mask)
                return _fused_rows(self, left_act, left_mask[..., None], equation)

            @hk.transparent
            def _triangle_multiplication(self, left_act, left_mask):
                equation = _guard(self, left_act, "unfused")
                if equation is None:
                    return super()._triangle_multiplication(left_act, left_mask)
                return _unfused_rows(self, left_act, left_mask[..., None], equation)
    else:
        class TriangleMultiplication(base):  # noqa: N801
            _rowchunk_rows = rows

            def __call__(self, act, mask, is_training=True):
                equation = _guard(self, act, "call")
                if equation is None:
                    return super().__call__(act, mask, is_training=is_training)
                return _unfused_rows(self, act, mask[..., None], equation)

    TriangleMultiplication.__qualname__ = getattr(base, "__qualname__", "TriangleMultiplication")
    TriangleMultiplication.__module__ = getattr(base, "__module__", TriangleMultiplication.__module__)
    return TriangleMultiplication


def transient_bytes(n_rows: int, rows: int, c_z: int, c_i: int, itemsize: int) -> Dict[str, int]:
    """Per call: the whole operand (``N*N*c_i``), one block's intermediates (``rows*N*(c_z + 2*c_i + c_z)``-class), the output (``N*N*c_z``) —
    arithmetic for the kit's block choice."""
    n, r, cz, ci, b = int(n_rows), int(rows), int(c_z), int(c_i), int(itemsize)
    whole = n * n * ci * b
    block = r * n * (2 * cz + 2 * ci) * b
    return {"whole_operand": whole, "block_intermediates": block, "output": n * n * cz * b, "total": whole + block + n * n * cz * b}
