"""Layouts, constraints, the shard_map call and the collectives of the row-sharded pair stack.

Layouts. The pair representation ``[N, N, C]`` (or the channel-first ``[C, N, N]`` some sub-layers transpose to) is ROW-sharded: device ``d``
holds rows ``[d*N/P, (d+1)*N/P)`` of every column — :func:`rows_spec`. The COLUMN layout (:func:`cols_spec`) is the transient of the ending-node
attention and of the incoming multiplication. Everything not pair-shaped (single/MSA activations, masks, parameters) is REPLICATED
(:func:`replicated_spec`) so propagation never leaks a row sharding into ops with uneven dims. Inside a ``shard_map`` region a function sees LOCAL
blocks (``[N/P, N, C]``) and the axis name; outside, global arrays carry ``NamedSharding`` and :func:`constrain` pins the layout at the pair sites
(the trunk prologue's pair outputs, the recycle carry, the trunk → heads boundary).

Collectives (the enumerable set of the plan; class ``moves_bytes`` — values are moved, never combined — except :func:`psum`, class ``reordered``):
:func:`gather` (all_gather of the row blocks → the full axis), :func:`rows_to_cols` / :func:`cols_to_rows` (all_to_all re-layout between the row
block ``[N/P, N, …]`` and the column block ``[N, N/P, …]``), :func:`transpose_block` (the rows of ``x^T`` this device owns, from the row block of
``x``: one all_to_all + a swapaxes — the symmetrisation ``x + x^T`` and the ending-node transpose need exactly this), :func:`local_block` (my slice
of a REPLICATED operand along a dim: ``dynamic_slice`` at ``axis_index * n_loc`` — the outer-product-mean's left operand, mask blocks),
:func:`psum`. :func:`shard_map` is the one call form over jax 0.5 … 0.10 (``jax.shard_map(check_vma=False)`` or
``jax.experimental.shard_map.shard_map(check_rep=False)`` — the replication checker is off: the bodies are stock sub-layer code whose invariants
the checker cannot infer through custom calls). :func:`local_extent` is the ``N % P`` gate (refused by name; :func:`next_multiple` is the arithmetic
of the kit's bucket rule). Standard library at import; jax inside the functions.

The all_to_all element limit (:func:`all_to_all`, the ONE binding of ``lax.all_to_all(tiled)`` that :func:`rows_to_cols` / :func:`cols_to_rows` /
:func:`transpose_block` issue). XLA's tiled all_to_all on jax 0.5-class GPU runtimes fails ``INVALID_ARGUMENT`` ('All buffers must have the same element
type and count') once the LOCAL operand holds ``>= 2**31`` elements (``[2048, 8192, 128]`` = 2**31 fails; ppermute / all_gather are unaffected; jax 0.10
is unaffected) — the pair block ``[N/P, N, 128]`` reaches it at ``N >= 4096 * sqrt(P)``. Below :data:`A2A_MAX_ELEMS` local elements (env
:data:`A2A_MAX_ELEMS_ENV` lowers it for tests) the call is the ONE ``lax.all_to_all`` it always was; at or above it the operand is cut into consecutive
slices along a FREE dim (one that is neither split nor concatenated — the channel dim of the pair block: the exchange is independent per index of it),
one all_to_all per slice, the results concatenated along that dim == the one call element for element (class ``moves_bytes``: the same bytes arrive by
k collectives instead of one; :func:`a2a_plan` is the arithmetic, :func:`a2a_fields` the evidence word ``a2a_split=<k>`` the family line carries when a
split was traced in this process). P=1 never reaches a collective (no mesh, no region).
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import MemLeverRefused
from . import LEVER, _lazy

A2A_MAX_ELEMS = 2 ** 31                          # a piece of ONE lax.all_to_all holds FEWER local elements than this (XLA/NCCL all_to_all: INVALID_ARGUMENT at >= 2**31 on jax 0.5-class runtimes)
A2A_MAX_ELEMS_ENV = "ROWPAIR_JAX_A2A_MAX_ELEMS"      # a positive integer lowering the limit (the unit tests force the split on small operands); unset = A2A_MAX_ELEMS
_A2A_RECORD: Dict[str, int] = {"kmax": 1, "calls": 0}   # trace-time record of the split all_to_alls of this process: the largest piece count, how many split calls were traced


# ------------------------------------------------------------------------------------------------------------------ specs
def rows_spec(axis: str, ndim: int = 3, row_dim: int = 0):
    """``PartitionSpec`` sharding dim ``row_dim`` of an ``ndim`` array over ``axis``, every other dim replicated: rows ``P(axis, None, None)`` for
    ``[N, N, C]``; ``rows_spec(axis, 3, 1)`` = ``P(None, axis, None)`` for the channel-first ``[C, N, N]``."""
    _Mesh, _NS, P = _lazy.sharding()
    rd = row_dim % ndim
    return P(*[axis if i == rd else None for i in range(ndim)])


def cols_spec(axis: str, ndim: int = 3, col_dim: int = 1):
    """The column layout: dim ``col_dim`` sharded (``P(None, axis, None)`` for ``[N, N, C]``)."""
    return rows_spec(axis, ndim, col_dim)


def replicated_spec():
    """``P()`` — replicated on every device of the mesh."""
    _Mesh, _NS, P = _lazy.sharding()
    return P()


def named(rmesh: Any, spec: Any):
    """``NamedSharding(mesh, spec)``; ``rmesh`` is a :class:`mesh.RowMesh` or a bare ``jax.sharding.Mesh``."""
    _Mesh, NamedSharding, _P = _lazy.sharding()
    return NamedSharding(getattr(rmesh, "mesh", rmesh), spec)


def constrain(x: Any, rmesh: Any, row_dim: int = 0, axis: Optional[str] = None):
    """``with_sharding_constraint`` of a GLOBAL pair array to the row layout (inside jit, outside shard_map): the pair sites of the plan."""
    jax = _lazy.jax()
    ax = axis or getattr(rmesh, "axis", None)
    if ax is None:
        raise MemLeverRefused(LEVER, "constrain: an axis name is required (pass a RowMesh or axis=)")
    return jax.lax.with_sharding_constraint(x, named(rmesh, rows_spec(ax, x.ndim, row_dim)))


def replicate(x: Any, rmesh: Any):
    """``with_sharding_constraint`` to the replicated layout (confines a non-pair array)."""
    jax = _lazy.jax()
    return jax.lax.with_sharding_constraint(x, named(rmesh, replicated_spec()))


def put(tree: Any, rmesh: Any, spec: Any = None):
    """``jax.device_put`` of a pytree onto the mesh with ONE spec for every leaf (default replicated): the model inputs / parameters of the jit."""
    jax = _lazy.jax()
    sh = named(rmesh, replicated_spec() if spec is None else spec)
    return jax.device_put(tree, sh)


# ------------------------------------------------------------------------------------------------------------------ shard_map
def shard_map(f: Callable, rmesh: Any, in_specs: Any, out_specs: Any, check: bool = False) -> Callable:
    """``f`` mapped over the mesh: inside, arguments are the LOCAL blocks per ``in_specs`` and results are assembled per ``out_specs``. ``check=False``
    turns the replication checker off under both API generations (``check_vma`` / ``check_rep``)."""
    fn, _flavour = _lazy.shard_map_impl()
    mesh = getattr(rmesh, "mesh", rmesh)
    params = inspect.signature(fn).parameters
    flag = "check_vma" if "check_vma" in params else "check_rep"
    if flag not in params:
        raise MemLeverRefused(LEVER, f"shard_map has neither check_vma nor check_rep ({sorted(params)})")
    return fn(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **{flag: bool(check)})


def shard_map_flavour() -> str:
    """``jax.shard_map`` or ``jax.experimental.shard_map`` — which call :func:`shard_map` binds on this jax (an evidence value)."""
    return _lazy.shard_map_impl()[1]


# ------------------------------------------------------------------------------------------------------------------ extents
def next_multiple(n: int, p: int) -> int:
    """The smallest multiple of ``p`` that is ``>= n`` (the arithmetic of a kit's bucket / padding rule under P devices)."""
    n, p = int(n), int(p)
    if p < 1 or n < 0:
        raise ValueError(f"next_multiple: need n >= 0 and p >= 1 (got n={n}, p={p})")
    return ((n + p - 1) // p) * p


def pad_plan(n_rows: int, n_gpu: int) -> Dict[str, int]:
    """``{"n": N, "n_padded": next multiple of P, "pad": rows to add, "n_loc": rows per device}`` — every size has a plan (a ladder bin is padded, never
    refused): the kit pads the pair map AND its masks by ``pad`` zero rows/columns (or picks a bucket that is a multiple of P) before the region."""
    n, p = int(n_rows), int(n_gpu)
    npad = next_multiple(n, p)
    return {"n": n, "n_padded": npad, "pad": npad - n, "n_loc": npad // p}


def pad_pair(x: Any, n_padded: int, dims: Sequence[int] = (0, 1), value: float = 0.0):
    """``x`` zero-padded at the END of each of ``dims`` up to ``n_padded`` (the pair map on both token dims; a ``[N, N]`` mask the same; per-token arrays on
    one dim). Class ``moves_bytes`` for the original elements; the padded rows must be masked by the kit's (equally padded) masks. :func:`unpad_pair` cuts back."""
    jnp = _lazy.jnp()
    widths = [(0, 0)] * x.ndim
    for d in dims:
        d = d % x.ndim
        if x.shape[d] > int(n_padded):
            raise MemLeverRefused(LEVER, f"pad_pair: dim {d} has {x.shape[d]} > n_padded={n_padded}")
        widths[d] = (0, int(n_padded) - int(x.shape[d]))
    return jnp.pad(x, widths, mode="constant", constant_values=value)


def unpad_pair(x: Any, n_rows: int, dims: Sequence[int] = (0, 1)):
    """The leading ``n_rows`` of each of ``dims`` (undoes :func:`pad_pair`)."""
    jax = _lazy.jax()
    out = x
    for d in dims:
        out = jax.lax.slice_in_dim(out, 0, int(n_rows), axis=d % x.ndim)
    return out


def local_extent(n_rows: int, n_gpu: int, lever: str = LEVER) -> int:
    """``N / P`` — the row count per device for an N the kit has padded (:func:`pad_plan`); an unpadded ``N % P != 0`` reaching here is refused by name."""
    n, p = int(n_rows), int(n_gpu)
    if p < 1:
        raise MemLeverRefused(lever, f"refused: n_gpu={p} — must be >= 1")
    if n % p != 0:
        raise MemLeverRefused(lever, f"refused: num_rows={n} not divisible by n_gpu={p} (pad or bucket to {next_multiple(n, p)})")
    return n // p


def axis_index(axis: str):
    """This device's position on the mesh axis (a traced scalar; inside shard_map only)."""
    jax = _lazy.jax()
    return jax.lax.axis_index(axis)


def axis_size(axis: str) -> int:
    """P inside a shard_map region (``lax.psum(1, axis)``; ``lax.axis_size`` where the jax has it)."""
    jax = _lazy.jax()
    fn = getattr(jax.lax, "axis_size", None)
    if fn is not None:
        return int(fn(axis))
    return int(jax.lax.psum(1, axis))


# ------------------------------------------------------------------------------------------------------------------ collectives (inside shard_map)
def gather(x_local: Any, axis: str, dim: int = 0):
    """all_gather (tiled) of the per-device blocks along ``dim``: ``[N/P, …]`` → ``[N, …]`` on every device. Class ``moves_bytes``."""
    jax = _lazy.jax()
    return jax.lax.all_gather(x_local, axis, axis=dim % x_local.ndim, tiled=True)


def a2a_max_elems(environ: Optional[Mapping[str, str]] = None, lever: str = LEVER) -> int:
    """The element limit of ONE all_to_all piece: env :data:`A2A_MAX_ELEMS_ENV` when set (a positive integer — refused by name otherwise), else
    :data:`A2A_MAX_ELEMS` (2**31). Read at trace time (each :func:`all_to_all` call), never at import."""
    env = os.environ if environ is None else environ
    raw = env.get(A2A_MAX_ELEMS_ENV)
    if raw is None or str(raw).strip() == "":
        return A2A_MAX_ELEMS
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise MemLeverRefused(lever, f"refused: {A2A_MAX_ELEMS_ENV}={raw!r} is not a positive integer") from None
    if value < 1:
        raise MemLeverRefused(lever, f"refused: {A2A_MAX_ELEMS_ENV}={raw!r} is not a positive integer")
    return value


def a2a_plan(shape: Sequence[int], split_axis: int, concat_axis: int, limit: Optional[int] = None) -> Tuple[Optional[int], List[int]]:
    """How ONE all_to_all of a LOCAL operand of ``shape`` is issued under the element ``limit`` (default :func:`a2a_max_elems`): ``(None, [])`` = the one
    ``lax.all_to_all`` call (fewer than ``limit`` elements — today's path, untouched); ``(dim, sizes)`` = consecutive slices of widths ``sizes`` along ``dim``,
    a dim that is neither ``split_axis`` nor ``concat_axis`` (the all_to_all moves data independently per index of such a dim, so all_to_all per slice and a
    concatenate along ``dim`` IS the one call, element for element), every slice holding fewer than ``limit`` elements, the widths near-equal (they differ by
    at most one; ``sum(sizes) == shape[dim]``; the fewest slices that fit). The dim chosen is the largest free extent (ties → the later dim: the channel dim of
    ``[N/P, N, C]``). An operand with no free dim (rank 2) or whose single index-slice already holds ``>= limit`` elements cannot be cut this way → ``(None,
    [])``: the one call as today (a pair block reaches that only at ``N*N/P >= limit`` — beyond any card)."""
    dims = [int(s) for s in shape]
    nd = len(dims)
    lim = a2a_max_elems() if limit is None else int(limit)
    total = 1
    for s in dims:
        total *= s
    if total < lim:
        return None, []
    sa, ca = split_axis % nd, concat_axis % nd
    free = [d for d in range(nd) if d not in (sa, ca) and dims[d] > 1]
    if not free:
        return None, []
    dim = max(free, key=lambda d: (dims[d], d))
    per_index = total // dims[dim]                                        # elements of a width-1 slice along dim
    if per_index >= lim:
        return None, []
    width = (lim - 1) // per_index                                        # the widest slice under the limit (>= 1)
    k = -(-dims[dim] // width)                                            # the fewest slices (ceil)
    base, extra = divmod(dims[dim], k)
    return dim, [base + 1] * extra + [base] * (k - extra)


def a2a_record() -> Dict[str, int]:
    """``{"kmax": the largest piece count a split all_to_all used in this process (1 = none engaged), "calls": split all_to_all calls traced}`` — a copy."""
    return dict(_A2A_RECORD)


def a2a_reset() -> None:
    """Forget the split record (the unit tests; a kit never needs it)."""
    _A2A_RECORD.update(kmax=1, calls=0)


def a2a_fields() -> Dict[str, Any]:
    """The evidence word of the split: ``{"a2a_split": kmax}`` when a split all_to_all was traced in this process, else ``{}`` (the family line then reads
    exactly as before — :func:`evidence.line` folds this in under ``state=on``)."""
    rec = a2a_record()
    return {"a2a_split": int(rec["kmax"])} if int(rec["calls"]) > 0 else {}


def all_to_all(x: Any, axis: str, split_axis: int, concat_axis: int, lever: str = LEVER):
    """The ONE binding of ``jax.lax.all_to_all(x, axis, split_axis, concat_axis, tiled=True)`` the re-layouts issue: below the element limit the single call,
    verbatim; at or above it the :func:`a2a_plan` pieces — ``lax.slice_in_dim`` along the free dim, one all_to_all per piece, ``jnp.concatenate`` along the
    same dim (recorded for :func:`a2a_fields`). Class ``moves_bytes`` either way; the transient of the split form is one extra block (the pieces' outputs
    live until the concatenate)."""
    jax = _lazy.jax(lever)
    nd = x.ndim
    sa, ca = split_axis % nd, concat_axis % nd
    dim, sizes = a2a_plan(x.shape, sa, ca)
    if dim is None:
        return jax.lax.all_to_all(x, axis, split_axis=sa, concat_axis=ca, tiled=True)
    jnp = _lazy.jnp(lever)
    pieces, start = [], 0
    for width in sizes:
        blk = jax.lax.slice_in_dim(x, start, start + width, axis=dim)
        pieces.append(jax.lax.all_to_all(blk, axis, split_axis=sa, concat_axis=ca, tiled=True))
        start += width
    _A2A_RECORD["calls"] = int(_A2A_RECORD["calls"]) + 1
    _A2A_RECORD["kmax"] = max(int(_A2A_RECORD["kmax"]), len(sizes))
    return jnp.concatenate(pieces, axis=dim)


def rows_to_cols(x_rows: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """Re-layout the ROW block ``[N/P, N, …]`` into the COLUMN block ``[N, N/P, …]`` (one all_to_all: split ``col_dim`` across devices, concatenate
    the received pieces along ``row_dim``; :func:`all_to_all` cuts it into channel pieces at the element limit). Class ``moves_bytes``."""
    nd = x_rows.ndim
    return all_to_all(x_rows, axis, split_axis=col_dim % nd, concat_axis=row_dim % nd)


def cols_to_rows(x_cols: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """The inverse of :func:`rows_to_cols`: the COLUMN block ``[N, N/P, …]`` back to the ROW block ``[N/P, N, …]`` (:func:`all_to_all`). Class ``moves_bytes``."""
    nd = x_cols.ndim
    return all_to_all(x_cols, axis, split_axis=row_dim % nd, concat_axis=col_dim % nd)


def transpose_block(x_rows: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """The row block of ``x^T`` this device owns, from its row block of ``x``: ``(x^T)[i, j] = x[j, i]`` for my rows ``i`` = my COLUMN block of ``x``
    (:func:`rows_to_cols`) with the two dims swapped. One all_to_all. Class ``moves_bytes``. ``x + transpose_block(x)`` is the symmetrisation of a
    row-sharded map; ``transpose_block`` in and out brackets a computation written for ``x^T`` (the ending-node attention)."""
    jnp = _lazy.jnp()
    return jnp.swapaxes(rows_to_cols(x_rows, axis, row_dim, col_dim), row_dim % x_rows.ndim, col_dim % x_rows.ndim)


def local_block(x_full: Any, axis: str, n_loc: int, dim: int = 0):
    """My ``n_loc``-long slice of a REPLICATED array along ``dim`` (``dynamic_slice_in_dim`` at ``axis_index * n_loc``): an operand whose rows must
    match the local pair rows (the outer-product-mean's left activations, a mask block, per-token features). Class ``moves_bytes``."""
    jax = _lazy.jax()
    start = jax.lax.axis_index(axis) * int(n_loc)
    return jax.lax.dynamic_slice_in_dim(x_full, start, int(n_loc), axis=dim % x_full.ndim)


def psum(x: Any, axis: str):
    """Cross-device sum (class ``reordered``: the P partials are added in the collective's order)."""
    jax = _lazy.jax()
    return jax.lax.psum(x, axis)


def ppermute_shift(x: Any, axis: str, shift: int = 1):
    """Send my block to device ``(me + shift) % P`` and receive from ``(me - shift) % P`` (the ring step). Class ``moves_bytes``."""
    jax = _lazy.jax()
    p = axis_size(axis)
    perm = [(j, (j + int(shift)) % p) for j in range(p)]
    return jax.lax.ppermute(x, axis, perm)
