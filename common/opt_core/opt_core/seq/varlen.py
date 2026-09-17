"""Varlen bookkeeping without device→host synchronisation: cumulative sequence lengths from host-known shapes, dense unpad / pad as views.

Contract. A flash-attention varlen kernel takes ``cu_seqlens`` (int32, ``[batch + 1]``) and ``max_seqlen`` (a host int). Deriving them
from a device mask (``mask.sum().item()``, ``nonzero``) synchronises the stream once per layer per forward; for a batch whose lengths
the host already knows (every row one full-length sequence, or lengths carried from tokenisation) the same numbers are arithmetic.
:func:`dense_offsets` / :func:`offsets_from_lengths` are that arithmetic (pure Python); :func:`dense_cu_seqlens` holds the result as
an int32 tensor built from host ints and cached per ``(batch, seqlen, device)`` (a bounded set: the shapes a route serves), so a
steady-state dense forward allocates and synchronises nothing; :func:`cu_seqlens_from_lengths` builds the tensor per call, uncached (ragged
length tuples do not repeat). :func:`unpad_dense` / :func:`pad_dense` are the dense case of unpad / pad: reshapes (a view when the layout
is contiguous — the dense case — else a copy with the same values), no gather, no scatter. The kernel and its arguments are the engine's own: the values equal what the engine's mask-derived
path computes for the same batch, so the attention output is unchanged. torch is imported on first tensor use; without it the tensor
functions raise :class:`opt_core.seq.numerics.TorchUnavailable` through ``numerics.require_torch()`` (the sub-package's one torch refusal) and the pure functions still work.
``census()`` / ``line_fields()`` / ``line()`` record what was served (``varlen=<dense|lengths|both|none> cu_built=<n> cu_hits=<n>
cu_lengths=<n>`` — the word is derived from the counters, not assumed).
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List, Sequence, Tuple

from ..report import kv

MAX_DENSE_SHAPES = 64                                        # distinct (batch, seqlen, device) keys held; beyond it tensors are built uncached (never evicted: a graph may read a cached one)
_CU = {}                                                     # (batch, seqlen, device) -> int32 tensor
COUNTERS = {"cu_built": 0, "cu_hits": 0, "cu_uncached": 0, "cu_lengths": 0, "unpad_dense": 0, "pad_dense": 0}


def _torch():
    from .numerics import require_torch                      # the sub-package's one torch import-or-refuse (raises numerics.TorchUnavailable)
    return require_torch()


def dense_offsets(batch: int, seqlen: int) -> List[int]:
    """``[0, L, 2L, ..., B*L]`` — the cumulative lengths of ``batch`` rows of ``seqlen`` tokens each."""
    b, n = int(batch), int(seqlen)
    if b < 0 or n < 0:
        raise ValueError(f"dense_offsets: batch={batch} seqlen={seqlen}")
    return [i * n for i in range(b + 1)]


def offsets_from_lengths(lengths: Sequence[int]) -> Tuple[List[int], int]:
    """``(cu, max_len)`` for host-known per-row lengths: ``cu[0] = 0``, ``cu[i+1] = cu[i] + lengths[i]``. Lengths must be host ints (a
    device tensor here would be the synchronisation this module exists to avoid — refused by type)."""
    cu = [0]
    mx = 0
    for x in lengths:
        if type(x) is not int:
            if hasattr(x, "device") or hasattr(x, "data_ptr"):
                raise TypeError("offsets_from_lengths: lengths must be host ints, not device tensors (pass the tokeniser's lengths)")
            x = operator.index(x)                            # numpy / python integers; a float is refused by type
        if x < 0:
            raise ValueError(f"offsets_from_lengths: negative length {x}")
        cu.append(cu[-1] + x)
        mx = x if x > mx else mx
    return cu, mx


def _int32(values: List[int], device: Any):
    torch = _torch()
    return torch.tensor(values, dtype=torch.int32, device=device)


def dense_cu_seqlens(batch: int, seqlen: int, device: Any):
    """int32 ``[batch + 1]`` cu_seqlens of a dense batch on ``device``, cached per ``(batch, seqlen, device)`` (the same tensor object on
    every later call — safe under graph capture). Past :data:`MAX_DENSE_SHAPES` distinct keys new shapes are built per call, uncached and
    counted (``cu_uncached``); nothing is ever evicted."""
    k = (int(batch), int(seqlen), str(device))
    t = _CU.get(k)
    if t is not None:
        COUNTERS["cu_hits"] += 1
        return t
    t = _int32(dense_offsets(batch, seqlen), device)
    if len(_CU) < MAX_DENSE_SHAPES:
        _CU[k] = t
        COUNTERS["cu_built"] += 1
    else:
        COUNTERS["cu_uncached"] += 1
    return t


def cu_seqlens_from_lengths(lengths: Sequence[int], device: Any) -> Tuple[Any, int]:
    """``(cu_seqlens int32 [len + 1] on device, max_seqlen)`` for host-known lengths — built per call from host ints (no cache: ragged
    length tuples do not repeat; no device→host sync either way)."""
    cu, mx = offsets_from_lengths(lengths)
    COUNTERS["cu_lengths"] += 1
    return _int32(cu, device), int(mx)


def unpad_dense(*xs: Any) -> Tuple:
    """``(B, L, ...)`` → ``(B*L, ...)`` for each tensor (``reshape``: a view when contiguous, else a copy with the same values) — what
    unpad's gather returns for an all-true mask, minus the gather."""
    out = []
    for x in xs:
        b, n = int(x.shape[0]), int(x.shape[1])
        out.append(x.reshape(b * n, *tuple(x.shape[2:])))
    COUNTERS["unpad_dense"] += 1
    return tuple(out)


def pad_dense(h: Any, batch: int, seqlen: int) -> Any:
    """``(B*L, ...)`` → ``(B, L, ...)``: the dense case of pad_input (a reshape, no scatter). A first dimension that is not ``B*L`` is a
    caller error (the batch was not dense) and raises."""
    b, n = int(batch), int(seqlen)
    if int(h.shape[0]) != b * n:
        raise ValueError(f"pad_dense: {int(h.shape[0])} rows is not batch*seqlen={b}*{n} — the batch is not dense; use the engine's pad_input")
    COUNTERS["pad_dense"] += 1
    return h.reshape(b, n, *tuple(h.shape[1:]))


def clear() -> None:
    """Forget the cached tensors — the kit's disengage only: a CUDA graph that captured a cached tensor reads it by address, so never
    clear while such a graph is alive."""
    _CU.clear()


def census() -> dict:
    return dict(COUNTERS, cached=len(_CU))


def served() -> str:
    """``dense`` | ``lengths`` | ``both`` | ``none`` — which cu_seqlens path this process actually served, from the counters."""
    d = (COUNTERS["cu_built"] + COUNTERS["cu_hits"] + COUNTERS["cu_uncached"]) > 0
    g = COUNTERS["cu_lengths"] > 0
    return "both" if d and g else "dense" if d else "lengths" if g else "none"


def line_fields() -> Dict[str, str]:
    """``{"varlen": served(), "cu_built": n, "cu_hits": n, "cu_lengths": n}`` (values as str) — the activation-evidence fields for the
    kit's ACTIVE line; ``cu_uncached`` is added when non-zero."""
    out = {"varlen": served(), "cu_built": str(COUNTERS["cu_built"]), "cu_hits": str(COUNTERS["cu_hits"]), "cu_lengths": str(COUNTERS["cu_lengths"])}
    if COUNTERS["cu_uncached"]:
        out["cu_uncached"] = str(COUNTERS["cu_uncached"])
    return out


def line() -> str:
    """``varlen=<dense|lengths|both|none> cu_built=<n> cu_hits=<n> cu_lengths=<n>``."""
    return kv(*line_fields().items())
