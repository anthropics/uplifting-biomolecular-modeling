"""Rows of a stock full-tensor CUDA Philox draw — ``nn.init.trunc_normal_`` and fused ``F.dropout`` — bit-identical to the single-device call,
with the CUDA generator left exactly where the stock call leaves it. A rank materialises ONLY its global rows ``[g0, g1)`` of a pair-shaped
draw ``[1, L, L, C]``; the draw is never built whole. Nothing is emulated: the stock kernels are RE-ISSUED at the stock launch geometry with
the generator offset set to the value the stock launch would see, and the overlap with the wanted rows is copied out.

Why re-issuing reproduces the bits (torch CUDA, ``aten/src/ATen/native/cuda/DistributionTemplates.h``, ``CUDAGeneratorImpl.cpp``,
``TensorIterator.cpp``, ``torch/nn/init.py``, ``native/cuda/Dropout.cu``):

* ``Tensor.normal_`` on a contiguous CUDA tensor is ``distribution_nullary_kernel``: block = 256 threads, ``grid = min(ceil(N / 256),
  multiProcessorCount * (maxThreadsPerMultiProcessor // 256))``, and the generator advances by ``counter_offset = ((N - 1) // (256 * grid *
  unroll) + 1) * 4`` (``unroll`` = 4 for fp32) taken BEFORE the 32-bit-indexing check. Inside one launch the value at linear index ``i`` is a
  pure function of (seed, philox offset, grid, i) (``curand_init(seed, idx, offset)`` + ``curand_normal4``; the Box–Muller uses the hardware
  ``__sincosf``, which is why a torch-ops Philox emulation cannot be bit-exact and the kernel itself is re-issued). A 1-D iterator that
  cannot use 32-bit indexing (``N > 2**31 - 1`` elements or byte extent above it, i.e. ``N > 2**29`` for fp32) is split recursively (first
  half = ``floor(n / 2)``, pieces yielded in address order) and EVERY piece re-enters the kernel with its own ``philox_cuda_state`` and grid;
  the whole-tensor state drawn first is then unused but its increment IS consumed. :func:`launch_plan` computes exactly these launches.
* ``nn.init.trunc_normal_`` (:func:`trunc_normal_algorithm` = ``"rejection"``) with acceptance mass ``p = Phi((b - mean) / std) - Phi((a -
  mean) / std) > 0.3`` is a rejection sampler over WHOLE-TENSOR normal draws: ``result = t.normal_(); loop: mask = (result < lo) | (result >
  hi); if not mask.any(): break; result = where(mask, empty_like(result).normal_(), result)`` with ``lo, hi`` the fp32-rounded bounds.
  Element ``e`` therefore equals the first in-bounds value among the draws ``D_0[e], D_1[e], ...`` and the generator consumption is ``1 +
  max_e(first accepted round)`` whole draws — a GLOBAL, data-dependent count: every rank reports its rows' round count and the MAXIMUM over
  ranks (:func:`agree_rounds`, one int64 ``allreduce(max)`` through :mod:`.dist`) finishes the generator (:func:`finish_generator`). A torch
  whose ``trunc_normal_`` is the inverse-CDF form (``uniform_ + erfinv_``) is REFUSED BY NAME (no rows replica of that form is offered).
* ``F.dropout(x, p, training=True)`` on a contiguous fp32 CUDA tensor with ``N % 4 == 0`` is the fused ``native_dropout`` kernel
  (``fused_dropout_kernel_vec<VEC=4>``): the same block/grid formula, ONE philox state (no 32-bit splitting), thread ``idx`` handles elements
  ``[i, i + 4)`` for ``i = idx * 4 + k * (grid * 256 * 4)`` from its k-th ``curand_uniform4``. Once the grid is SATURATED (``N >= (grid_max
  - 1) * 256 + 1``) the element -> (thread, call, component) map is independent of the launch size, so re-issuing ``native_dropout`` on a
  buffer holding flat elements ``[F0, F0 + n)`` with ``F0`` a multiple of the period ``4 * 256 * grid`` and the generator offset set to
  ``base + 4 * (F0 / period)`` reproduces output and mask bit for bit; the stock call advances the generator by ``counter_offset(N)``. A
  non-saturating size is refused by name (the stock launch is then small enough to run whole).

Memory: at most one launch piece (<= 2 GiB fp32) plus about ``3 x elems_budget`` fp32 transients, independent of ``L**2``. CUDA only: CPU
generators have no Philox offset (refused by name). Every rank must enter with the same generator state (the trunk's RNG guard proves it:
:func:`opt_core.mem.rowpair.trunk.guard_rng_replicated`).

API (refusals are :class:`opt_core.mem.rowpair.RowpairRefused`):
    cuda_default_generator(device) · get_offset(gen) · set_offset(gen, offset)      the CUDA generator and its Philox offset (a multiple of 4)
    launch_plan(numel, device, itemsize=4) -> LaunchPlan                              the stock launches of ONE ``normal_`` / ``uniform_``: ``launches
                                                                                      ((flat_start, n, rel_offset), ...)``, ``round_increment``
    stock_round_increment(numel, device, itemsize=4) -> int                          generator advance of one whole-tensor draw
    draw_normal_flat(plan, f0, f1, *, round_base_offset, mean, std, gen, device)     flat elements ``[f0, f1)`` of one whole-tensor ``normal_`` draw
    trunc_normal_algorithm() -> "rejection" | "inverse_cdf" | "unknown"               which ``nn.init.trunc_normal_`` this torch has
    trunc_normal_rows(L, C, g0, g1, *, std, mean, a, b, device, generator, base_offset, out_dtype, elems_budget)
        -> (rows ``[1, g1 - g0, L, C]``, info{rounds, base_offset, round_increment, ...})   rows of the stock ``trunc_normal_`` on ``[1, L, L, C]``
    agree_rounds(rounds, device) -> int                                                 MAX over ranks (equality without a group)
    finish_generator(gen, base_offset, round_increment, global_rounds) -> final        leave ``gen`` where the stock call leaves it
    trunc_normal_shard(layout, C, *, std, mean, a, b, device, generator, out_dtype)    this rank's rows + agree_rounds + finish_generator in one
                                                                                        call (a sharded :class:`.dist.Layout`; refused at P == 1)
    dropout_plan(numel, device) -> DropoutPlan · dropout_rows(x_rows, L, C, g0, g1, *, p, generator, base_offset, device, return_mask, finish)
        -> (out_rows fp32, mask_rows | None, info{base_offset, round_increment, final_offset, ...})
    self-test(device, which=("trunc_normal", "dropout")) -> dict                        bit-exact probe of the replicas against the stock calls at
                                                                                        small sizes; the generator offset is restored
"""
from __future__ import annotations

import inspect
import math
from typing import Dict, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch

__all__ = ["LaunchPlan", "launch_plan", "stock_round_increment", "draw_normal_flat", "trunc_normal_algorithm", "trunc_normal_rows",
           "agree_rounds", "finish_generator", "trunc_normal_shard", "cuda_default_generator", "get_offset", "set_offset",
           "DropoutPlan", "dropout_plan", "dropout_rows", "selfcheck", "INT32_MAX", "BLOCK_SIZE"]

LEVER = "rowpair.rng"
INT32_MAX = 2 ** 31 - 1
BLOCK_SIZE = 256                      # block_size_bound of the distribution kernels
CURAND4_ENGINE_CALLS = 4              # generator offsets consumed per curand_*4 call
UNROLL_FP32 = 4                       # float4 unroll of the fp32 distribution functors


def _refuse(msg: str) -> RowpairRefused:
    return RowpairRefused(msg, LEVER)


# ================================================================================================================ generator helpers
def cuda_default_generator(device):
    """The default CUDA generator of ``device`` (the one ``normal_`` / ``dropout`` use without ``generator=``)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise _refuse(f"cuda_default_generator: device {dev} is not CUDA (CPU generators have no Philox offset; the rows replica is CUDA-only)")
    torch.cuda.init()                                                       # a cold process has no default generators before CUDA's lazy init
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[idx]


def get_offset(gen) -> int:
    """The Philox offset of a CUDA generator."""
    try:
        if hasattr(gen, "get_offset"):
            return int(gen.get_offset())
        st = gen.get_state()
        return int(st[8:16].clone().view(torch.int64).item())
    except RuntimeError as exc:
        raise _refuse(f"get_offset: {exc} (a CUDA generator is required)") from None


def set_offset(gen, offset: int) -> None:
    """Set the Philox offset of a CUDA generator (a non-negative multiple of 4, as every stock launch leaves it)."""
    offset = int(offset)
    if offset < 0 or offset % 4:
        raise _refuse(f"set_offset: offset {offset} must be a non-negative multiple of 4")
    try:
        if hasattr(gen, "set_offset"):
            gen.set_offset(offset)
            return
        st = gen.get_state().clone()
        st[8:16] = torch.tensor([offset], dtype=torch.int64).view(torch.uint8)
        gen.set_state(st)
    except RuntimeError as exc:
        raise _refuse(f"set_offset: {exc} (a CUDA generator is required)") from None


# ================================================================================================================ launch geometry
_PROPS_CACHE: Dict[tuple, Tuple[int, int]] = {}


def _mp_props(device) -> Tuple[int, int]:
    """``(multiProcessorCount, maxThreadsPerMultiProcessor)`` of a CUDA device — the two device facts torch's grid formula reads."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise _refuse(f"launch geometry: device {dev} is not CUDA")
    key = (dev.type, dev.index if dev.index is not None else torch.cuda.current_device())
    if key not in _PROPS_CACHE:
        p = torch.cuda.get_device_properties(dev)
        mtp = getattr(p, "max_threads_per_multi_processor", None)
        if mtp is None:
            raise _refuse("torch.cuda.get_device_properties lacks max_threads_per_multi_processor: the stock grid cannot be derived exactly")
        _PROPS_CACHE[key] = (int(p.multi_processor_count), int(mtp))
    return _PROPS_CACHE[key]


def _grid(numel: int, sm: int, mtp: int) -> int:
    return min((int(numel) + BLOCK_SIZE - 1) // BLOCK_SIZE, int(sm) * (int(mtp) // BLOCK_SIZE))


def _counter_offset(numel: int, sm: int, mtp: int, unroll: int = UNROLL_FP32) -> int:
    g = _grid(numel, sm, mtp)
    return ((int(numel) - 1) // (BLOCK_SIZE * g * int(unroll)) + 1) * CURAND4_ENGINE_CALLS


def _fits_32bit(n: int, itemsize: int) -> bool:
    """``TensorIteratorBase::can_use_32bit_indexing`` for a 1-D single-operand iterator."""
    return int(n) <= INT32_MAX and (1 + (int(n) - 1) * int(itemsize)) <= INT32_MAX


def _split_until_32bit(numel: int, itemsize: int) -> list:
    """The pieces ``[(start, n), ...]`` a coalesced 1-D contiguous iterator of ``numel`` elements is split into until every piece can use
    32-bit indexing, in the order the sub-iterators are yielded (address order; first half = ``floor(n / 2)``)."""
    vec: list = [[0, int(numel)], None]

    def advance():
        vec.pop()
        while vec and not _fits_32bit(vec[-1][1], itemsize):
            s, n = vec[-1]
            copy = n // 2
            this = n - copy
            vec[-1] = [s + copy, this]      # the second half stays on the stack
            vec.append([s, copy])           # the first half is examined next
    advance()
    out = []
    while vec:
        out.append((vec[-1][0], vec[-1][1]))
        advance()
    return out


class LaunchPlan(object):
    """The launches of ONE stock ``t.normal_()`` / ``t.uniform_()`` on a contiguous CUDA tensor: ``launches = ((flat_start, n,
    generator_offset_relative_to_the_call), ...)`` and ``round_increment`` = the generator advance of the whole call."""

    __slots__ = ("numel", "itemsize", "launches", "round_increment", "sm_count", "max_threads_per_sm")

    def __init__(self, numel: int, itemsize: int, launches: tuple, round_increment: int, sm_count: int, max_threads_per_sm: int):
        self.numel, self.itemsize, self.launches = int(numel), int(itemsize), tuple(launches)
        self.round_increment, self.sm_count, self.max_threads_per_sm = int(round_increment), int(sm_count), int(max_threads_per_sm)

    def describe(self) -> str:
        ns = [l[1] for l in self.launches[:4]]
        return (f"LaunchPlan(numel={self.numel}, launches={len(self.launches)} x n={ns}{'...' if len(self.launches) > 4 else ''}, "
                f"round_increment={self.round_increment}, sm={self.sm_count}, mtp={self.max_threads_per_sm})")


def _launch_plan_for(numel: int, itemsize: int, sm: int, mtp: int) -> LaunchPlan:
    unroll = 4 if int(itemsize) <= 4 else 2          # float4 / double2 distribution functors
    first = _counter_offset(numel, sm, mtp, unroll)
    if _fits_32bit(numel, itemsize):
        return LaunchPlan(numel, itemsize, ((0, int(numel), 0),), first, sm, mtp)
    launches = []
    rel = first                                       # the whole-tensor philox state is drawn first and then not used
    for (s, n) in _split_until_32bit(numel, itemsize):
        launches.append((s, n, rel))
        rel += _counter_offset(n, sm, mtp, unroll)
    return LaunchPlan(numel, itemsize, tuple(launches), rel, sm, mtp)


def launch_plan(numel: int, device, itemsize: int = 4) -> LaunchPlan:
    """The launches of one ``normal_`` / ``uniform_`` on a contiguous CUDA tensor of ``numel`` elements of ``itemsize`` bytes on ``device``."""
    sm, mtp = _mp_props(device)
    return _launch_plan_for(int(numel), int(itemsize), sm, mtp)


def stock_round_increment(numel: int, device, itemsize: int = 4) -> int:
    """The generator offset advance of one stock whole-tensor ``normal_`` / ``uniform_`` over ``numel`` elements."""
    return launch_plan(numel, device, itemsize).round_increment


# ================================================================================================================ normal / trunc_normal rows
def draw_normal_flat(plan: LaunchPlan, f0: int, f1: int, *, round_base_offset: int, mean: float, std: float, gen, device, out=None):
    """Flat elements ``[f0, f1)`` of the fp32 tensor ``torch.empty(plan.numel).normal_(mean, std)`` would hold had the generator offset been
    ``round_base_offset`` at call time: only the stock sub-launches overlapping ``[f0, f1)`` are re-issued (the stock kernel on a fresh 1-D
    buffer of the piece's numel) and the overlap copied. The generator is left at an unspecified offset (see :func:`finish_generator`)."""
    f0, f1 = int(f0), int(f1)
    if not (0 <= f0 < f1 <= plan.numel):
        raise _refuse(f"draw_normal_flat: [{f0}, {f1}) outside [0, {plan.numel})")
    if out is None:
        out = torch.empty(f1 - f0, dtype=torch.float32, device=device)
    for (s, n, rel) in plan.launches:
        a = max(f0, s)
        b = min(f1, s + n)
        if a >= b:
            continue
        set_offset(gen, int(round_base_offset) + rel)
        buf = torch.empty(n, dtype=torch.float32, device=device)
        buf.normal_(mean, std, generator=gen)                 # the stock kernel at the stock numel: stock grid and offsets
        out[a - f0:b - f0].copy_(buf[a - s:b - s])
        del buf
    return out


_TN_ALGO: Optional[str] = None


def trunc_normal_algorithm() -> str:
    """``"rejection"`` when this torch's ``nn.init.trunc_normal_`` is the whole-tensor ``normal_`` rejection sampler (the form the rows replica
    reproduces), ``"inverse_cdf"`` for the ``uniform_ + erfinv_`` form, ``"unknown"`` otherwise. Read once from ``torch.nn.init`` source."""
    global _TN_ALGO
    if _TN_ALGO is None:
        try:
            src = inspect.getsource(torch.nn.init._no_grad_trunc_normal_)
        except (OSError, TypeError, AttributeError):
            src = ""
        if "erfinv_" in src:
            _TN_ALGO = "inverse_cdf"
        elif ".normal_(" in src and "torch.where(" in src and "mask.any()" in src:
            _TN_ALGO = "rejection"
        else:
            _TN_ALGO = "unknown"
    return _TN_ALGO


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def trunc_normal_rows(L: int, C: int, g0: int, g1: int, *, std: float, mean: float = 0.0, a: Optional[float] = None,
                      b: Optional[float] = None, device="cuda", generator=None, base_offset: Optional[int] = None, out_dtype=None,
                      elems_budget: int = 1 << 28, B: int = 1):
    """Rows ``[g0, g1)`` of ``state = torch.empty(1, L, L, C, fp32, cuda); nn.init.trunc_normal_(state, mean, std, a, b)`` (then
    ``.to(out_dtype)``), bit-identical to the stock call issued with ``generator`` (default: the device's default generator) at ``base_offset``
    (default: its current offset). ``a`` / ``b`` default to torch's ``-2, 2``. Rows are produced in super-blocks of at most ``elems_budget``
    elements. Returns ``(rows [1, g1 - g0, L, C], info)`` with ``info["rounds"]`` = the number of EXTRA whole-tensor draws these rows needed
    (the stock call consumed ``1 + max over ALL rows``: :func:`agree_rounds` then :func:`finish_generator`). The generator is left at an
    unspecified offset."""
    if int(B) != 1:
        raise _refuse("trunc_normal_rows: the flat-index replica is written for a leading batch of 1")
    algo = trunc_normal_algorithm()
    if algo != "rejection":
        raise _refuse(f"trunc_normal_rows: this torch's nn.init.trunc_normal_ is the {algo!r} form (torch {torch.__version__}); only the whole-tensor "
                      "normal_ rejection form has a rows replica")
    a = -2.0 if a is None else float(a)
    b = 2.0 if b is None else float(b)
    std, mean = float(std), float(mean)
    p = _norm_cdf((b - mean) / std) - _norm_cdf((a - mean) / std)
    if not (p > 0.3):
        raise _refuse(f"trunc_normal_rows: acceptance mass p={p:.4f} <= 0.3 takes trunc_normal_'s log-pdf branch, which has no rows replica")
    device = torch.device(device)
    gen = generator if generator is not None else cuda_default_generator(device)
    lo = torch.tensor(a, dtype=torch.float32).item()          # tensor.new_tensor(a).item(): the fp32-rounded bounds
    hi = torch.tensor(b, dtype=torch.float32).item()
    L, C, g0, g1 = int(L), int(C), int(g0), int(g1)
    if not (0 <= g0 < g1 <= L):
        raise _refuse(f"trunc_normal_rows: rows [{g0}, {g1}) outside [0, {L})")
    N = L * L * C
    plan = launch_plan(N, device, 4)
    base = get_offset(gen) if base_offset is None else int(base_offset)
    out_dtype = torch.float32 if out_dtype is None else out_dtype
    R = g1 - g0
    out = torch.empty((1, R, L, C), dtype=out_dtype, device=device)
    row_elems = L * C
    rows_per = max(1, min(R, int(elems_budget) // row_elems))
    max_rounds = 0
    n_super = 0
    for s in range(g0, g1, rows_per):
        e = min(s + rows_per, g1)
        f0, f1 = s * row_elems, e * row_elems
        res = draw_normal_flat(plan, f0, f1, round_base_offset=base, mean=mean, std=std, gen=gen, device=device)
        r = 0
        while True:
            mask = (res < lo) | (res > hi)                   # trunc_normal_'s own statements: elementwise, hence the same bits
            if not bool(mask.any()):
                break
            r += 1
            cand = draw_normal_flat(plan, f0, f1, round_base_offset=base + r * plan.round_increment, mean=mean, std=std, gen=gen, device=device)
            res = torch.where(mask, cand, res)
            del cand
        del mask
        max_rounds = max(max_rounds, r)
        blk = res.view(1, e - s, L, C)
        out[:, s - g0:e - g0].copy_(blk if out_dtype == torch.float32 else blk.to(dtype=out_dtype))
        del res, blk
        n_super += 1
    info = dict(rounds=max_rounds, base_offset=base, round_increment=plan.round_increment, launches_per_draw=len(plan.launches),
                superblocks=n_super, plan=plan.describe(), lo=lo, hi=hi)
    return out, info


def agree_rounds(rounds: int, device=None) -> int:
    """The MAXIMUM of ``rounds`` over the ranks of the group (one int64 ``allreduce(max)`` through :mod:`.dist`); ``rounds`` itself without a
    group. Every rank of a sharded draw calls it (the count is data dependent, the generator advance must be the stock one everywhere)."""
    from .dist import allreduce_, is_dist
    if not is_dist():
        return int(rounds)
    dev = torch.device(device) if device is not None else (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
                                                           else torch.device("cpu"))
    t = torch.tensor([int(rounds)], dtype=torch.int64, device=dev)
    allreduce_(t, "max")
    return int(t.item())


def finish_generator(gen, base_offset: int, round_increment: int, global_rounds: int) -> int:
    """Leave ``gen`` where the stock ``trunc_normal_`` leaves it: ``1 + global_rounds`` whole-tensor draws after ``base_offset``. Returns the offset."""
    final = int(base_offset) + (1 + int(global_rounds)) * int(round_increment)
    set_offset(gen, final)
    return final


def trunc_normal_shard(layout, C: int, *, std: float, mean: float = 0.0, a: Optional[float] = None, b: Optional[float] = None, device="cuda",
                       generator=None, out_dtype=None, elems_budget: int = 1 << 28):
    """This rank's shard ``[1, R, L, C]`` (``L = layout.N``, rows ``layout.r0:layout.r1``) of the stock ``trunc_normal_`` pair-state draw, with the
    generator advanced by the stock amount on every rank: :func:`trunc_normal_rows` + :func:`agree_rounds` + :func:`finish_generator`. Every rank
    enters with the same generator state and leaves with the same (stock) state. A sharded :class:`.dist.Layout` is required (refused by name
    at ``P == 1``: the engine's own call runs). Returns ``(z_loc, info)`` with ``info["global_rounds"]`` and ``info["final_offset"]``."""
    from .dist import require_sharded
    require_sharded(layout, "trunc_normal_shard", LEVER)
    device = torch.device(device)
    gen = generator if generator is not None else cuda_default_generator(device)
    base = get_offset(gen)
    rows, info = trunc_normal_rows(layout.N, int(C), layout.r0, layout.r1, std=std, mean=mean, a=a, b=b, device=device, generator=gen,
                                   base_offset=base, out_dtype=out_dtype, elems_budget=elems_budget)
    info["global_rounds"] = agree_rounds(info["rounds"], device)
    info["final_offset"] = finish_generator(gen, base, info["round_increment"], info["global_rounds"])
    return rows, info


# ================================================================================================================ fused dropout rows
class DropoutPlan(object):
    """The one launch of a stock fused dropout on ``numel`` fp32 elements: ``grid``, ``period = 4 * 256 * grid`` elements (one ``curand_uniform4``
    per thread), ``round_increment`` (the generator advance of the call), ``min_saturating`` (the smallest numel with this grid)."""

    __slots__ = ("numel", "grid", "period", "round_increment", "min_saturating")

    def __init__(self, numel: int, grid: int, period: int, round_increment: int, min_saturating: int):
        self.numel, self.grid, self.period, self.round_increment, self.min_saturating = numel, grid, period, round_increment, min_saturating

    def describe(self) -> str:
        return f"DropoutPlan(numel={self.numel}, grid={self.grid}, period={self.period}, incr={self.round_increment})"


def _dropout_plan_for(numel: int, sm: int, mtp: int) -> DropoutPlan:
    g = _grid(numel, sm, mtp)
    gmax = int(sm) * (int(mtp) // BLOCK_SIZE)
    if g < gmax:
        raise _refuse(f"dropout_rows: numel={numel} does not saturate the grid ({g} < {gmax}); the stock launch is small — run F.dropout whole")
    incr = ((int(numel) - 1) // (BLOCK_SIZE * g * 4) + 1) * 4
    return DropoutPlan(int(numel), g, 4 * BLOCK_SIZE * g, incr, (gmax - 1) * BLOCK_SIZE + 1)


def dropout_plan(numel: int, device) -> DropoutPlan:
    """The stock fused-dropout launch for ``numel`` fp32 elements on ``device`` (refused by name when the grid is not saturated)."""
    sm, mtp = _mp_props(device)
    return _dropout_plan_for(int(numel), sm, mtp)


def dropout_rows(x_rows, L: int, C: int, g0: int, g1: int, *, p: float, generator=None, base_offset: Optional[int] = None, device="cuda",
                 return_mask: bool = True, finish: bool = True, elems_budget: int = 1 << 28, B: int = 1):
    """Rows ``[g0, g1)`` of ``F.dropout(x, p, training=True)`` for a contiguous fp32 CUDA tensor ``x [1, L, L, C]`` whose rows ``[g0, g1)`` are
    ``x_rows [1, g1 - g0, L, C]`` (``None`` -> ones: the scaled keep mask), bit-identical to the stock call made with ``generator`` (default: the
    device's default generator) at ``base_offset`` (default: its current offset). Returns ``(out_rows fp32, mask_rows bool | None, info)``.
    ``finish=True`` leaves the generator where the stock call leaves it (``base + info["round_increment"]``); a caller producing several row
    chunks passes ``finish=False`` and calls ``set_offset(gen, info["final_offset"])`` once. No collective: the consumption is data independent."""
    if int(B) != 1:
        raise _refuse("dropout_rows: the flat-index replica is written for a leading batch of 1")
    device = torch.device(device)
    gen = generator if generator is not None else cuda_default_generator(device)
    L, C, g0, g1 = int(L), int(C), int(g0), int(g1)
    N = L * L * C
    if N % 4:
        raise _refuse(f"dropout_rows: numel {N} is not a multiple of 4 (the VEC=4 fused kernel path of a 16-byte aligned fp32 input is assumed)")
    plan = dropout_plan(N, device)
    base = get_offset(gen) if base_offset is None else int(base_offset)
    if not (0 <= g0 < g1 <= L):
        raise _refuse(f"dropout_rows: rows [{g0}, {g1}) outside [0, {L})")
    R = g1 - g0
    row_elems = L * C
    if x_rows is not None and (x_rows.dtype != torch.float32 or tuple(x_rows.shape) != (1, R, L, C)):
        raise _refuse(f"dropout_rows: x_rows must be fp32 [1, {R}, {L}, {C}], got {x_rows.dtype} {tuple(x_rows.shape)}")
    out = torch.empty((1, R, L, C), dtype=torch.float32, device=device)
    mask = torch.empty((1, R, L, C), dtype=torch.bool, device=device) if return_mask else None
    rows_per = max(1, min(R, int(elems_budget) // row_elems))
    n_launch = 0
    for s in range(g0, g1, rows_per):
        e = min(s + rows_per, g1)
        f0, f1 = s * row_elems, e * row_elems
        F0 = (f0 // plan.period) * plan.period
        n = max(f1 - F0, plan.min_saturating)
        n = -(-n // 4) * 4
        padded = x_rows is None or (f0 - F0) > 0 or n > (f1 - F0)
        buf = torch.zeros(n, dtype=torch.float32, device=device) if padded else torch.empty(n, dtype=torch.float32, device=device)
        if x_rows is None:
            buf[f0 - F0:f1 - F0].fill_(1.0)
        else:
            buf[f0 - F0:f1 - F0].copy_(x_rows[:, s - g0:e - g0].reshape(-1))
        if buf.data_ptr() % 16:
            raise _refuse("dropout_rows: the staging buffer is not 16-byte aligned (the VEC=4 kernel path needs it)")
        set_offset(gen, base + 4 * (F0 // plan.period))
        o, m = torch.native_dropout(buf, float(p), True)          # the stock kernel (F.dropout(training=True) takes this path)
        out[:, s - g0:e - g0].copy_(o[f0 - F0:f1 - F0].view(1, e - s, L, C))
        if return_mask:
            mask[:, s - g0:e - g0].copy_(m[f0 - F0:f1 - F0].view(1, e - s, L, C))
        del buf, o, m
        n_launch += 1
    final = base + plan.round_increment
    if finish:
        set_offset(gen, final)
    info = dict(base_offset=base, round_increment=plan.round_increment, final_offset=final, launches=n_launch, plan=plan.describe())
    return out, mask, info


# ================================================================================================================ bit-exact self-test
def selfcheck(device="cuda", which: Sequence[str] = ("trunc_normal", "dropout"), *, L: int = 40, C: int = 8, p: float = 0.25) -> Dict[str, object]:
    """Probe the replicas against the stock calls on ``device`` at a small size and report ``{<name>: {"equal": bool, "offset_equal": bool,
    ...}}``; the generator offset is restored afterwards. ``trunc_normal`` on a torch without the rejection form reports ``{"refused": <reason>}``.
    A kit runs it once per process before relying on the replica (fail loud: a False here is a refusal for the caller to raise)."""
    device = torch.device(device)
    gen = cuda_default_generator(device)
    base0 = get_offset(gen)
    rep: Dict[str, object] = {"device": str(device), "torch": torch.__version__, "trunc_normal_algorithm": trunc_normal_algorithm()}
    try:
        if "trunc_normal" in which:
            std = math.sqrt(2.0 / (5.0 * C))
            try:
                set_offset(gen, base0)
                t = torch.empty((1, L, L, C), dtype=torch.float32, device=device)
                torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2 * std, b=2 * std)
                off_stock = get_offset(gen)
                parts = [(0, L // 3), (L // 3, L - 1), (L - 1, L)]
                eq, rounds, incr = True, 0, None
                for g0, g1 in parts:
                    rows, info = trunc_normal_rows(L, C, g0, g1, std=std, mean=0.0, a=-2 * std, b=2 * std, device=device, generator=gen, base_offset=base0)
                    eq = eq and bool(torch.equal(rows, t[:, g0:g1]))
                    rounds, incr = max(rounds, info["rounds"]), info["round_increment"]
                final = finish_generator(gen, base0, incr, rounds)
                rep["trunc_normal"] = {"equal": eq, "offset_equal": final == off_stock, "rounds": rounds, "stock_offset_advance": off_stock - base0}
            except RowpairRefused as r:
                rep["trunc_normal"] = {"refused": r.reason}
        if "dropout" in which:
            try:
                sm, mtp = _mp_props(device)
                period = 4 * BLOCK_SIZE * sm * (mtp // BLOCK_SIZE)
                Ld = max(L, int(math.ceil(math.sqrt(2.2 * period / C))) + 1)         # > 2 periods: probes the per-period offset path too
                g = torch.Generator(device=device)
                g.manual_seed(11)
                x = torch.randn((1, Ld, Ld, C), generator=g, device=device)
                set_offset(gen, base0)
                o = torch.nn.functional.dropout(x, p=p, training=True)
                off_stock = get_offset(gen)
                parts = [(0, 1), (1, Ld // 2), (Ld // 2, Ld)]
                eq, final = True, None
                for g0, g1 in parts:
                    orow, _mrow, info = dropout_rows(x[:, g0:g1].contiguous(), Ld, C, g0, g1, p=p, generator=gen, base_offset=base0, device=device,
                                                     finish=False, elems_budget=max(Ld * C, period // 2))
                    eq = eq and bool(torch.equal(orow, o[:, g0:g1]))
                    final = info["final_offset"]
                rep["dropout"] = {"equal": eq, "offset_equal": final == off_stock, "L": Ld, "stock_offset_advance": off_stock - base0}
            except RowpairRefused as r:
                rep["dropout"] = {"refused": r.reason}
    finally:
        set_offset(gen, base0)
    return rep
