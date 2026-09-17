"""Row-blocked pairwise statements over ATOMS for the confidence tail — the pieces of an AF3-style engine's post-processing whose stock
form allocates ``[N_atom, N_atom(, 3)]`` (46 GB at 62k atoms for the frame-atom search; a chain-pair ``cdist`` matrix for the clash rule)
although each result is a per-row reduction (top-k of a row, a count). The same elementwise statements run on ROW BLOCKS of the query
rows against all keys: per element identical operands and operations, top-k / count per row on an identical row -> identical results
(indices bit-exact; counts exact). REPLICATED by design: every rank evaluates these on the replicated atom tensors (no communication) — they
are memory levers of the tail, not sharded statements, so they take no Layout and run at any P (including a kit's single-device path
when the kit names the lever).

API::

    frame_rows(n_keys, rows=None, bytes_per_pair=40, budget_bytes=None)   ``(rows, source)`` by :func:`.shard.choose_block_rows`: ``rows`` ->
                                             ``given``; ``ROWPAIR_FRAME_ROWS`` (row count) -> ``env:…``; ``budget_bytes`` -> ``budget``;
                                             ``ROWPAIR_FRAME_BUDGET_MB`` -> ``env:…``; else 1024 MiB -> ``default`` (largest block with
                                             ``rows * n_keys * bytes_per_pair`` within the budget). Recorded (``frame_rows``, ``frame_rows_source``).
                                             No collective depends on it (replicated statement), so a byte budget is safe.
    nearest_atoms_rows(x, query_index, atom_mask, atom_group, k=3, eps=1e-8, inf=1e9, rows=None)   for each query atom ``q = query_index[t]``:
                                             ``d[t, j] = (eps + |x_q - x_j|^2).sum(-1) ** 0.5`` masked by ``mask_q * mask_j * (group_q == group_j)``
                                             (``d * m + inf * (1 - m)``), then ``topk(d, k, largest=False)`` -> indices ``[*, n_q, k]`` (long).
                                             The AF3 token-frame search (query = each token's start atom; ``[..., 1]`` / ``[..., 2]`` are the
                                             frame's a / c atoms) without the all-pairs tensors.
    count_pairs_within(a, b, threshold, rows=None, max_pair_bytes=None)   ``int((cdist(a, b) < threshold).sum())`` — whole when the
                                             ``[n_a, n_b]`` fp32 matrix fits ``max_pair_bytes`` (``ROWPAIR_CLASH_MAX_GB``, default 8 GB), else summed
                                             over row blocks of ``rows`` (``ROWPAIR_CLASH_ROWS``, default 8192) query rows (integer counts: exact,
                                             up to ``cdist``'s own last-bit behaviour for pairs AT the threshold).
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import env_int

__all__ = ["ENV_FRAME_ROWS", "ENV_FRAME_BUDGET_MB", "ENV_CLASH_MAX_GB", "ENV_CLASH_ROWS", "frame_rows", "nearest_atoms_rows", "count_pairs_within"]

ENV_FRAME_ROWS = "ROWPAIR_FRAME_ROWS"
ENV_FRAME_BUDGET_MB = "ROWPAIR_FRAME_BUDGET_MB"
ENV_CLASH_MAX_GB = "ROWPAIR_CLASH_MAX_GB"
ENV_CLASH_ROWS = "ROWPAIR_CLASH_ROWS"


def frame_rows(n_keys: int, rows: Optional[int] = None, bytes_per_pair: int = 40, budget_bytes: Optional[int] = None,
               *, record: bool = True) -> Tuple[int, str]:
    """``(rows, source)`` — the query-row block of :func:`nearest_atoms_rows`, by :func:`.shard.choose_block_rows`: ``rows`` -> ``given``;
    env ``ROWPAIR_FRAME_ROWS`` (a row count) -> ``env:ROWPAIR_FRAME_ROWS``; ``budget_bytes`` -> ``budget``; env ``ROWPAIR_FRAME_BUDGET_MB`` ->
    ``env:…``; else 1024 MiB -> ``default`` (one row = ``n_keys * bytes_per_pair`` bytes). Recorded as ``frame_rows`` / ``frame_rows_source``."""
    from .shard import choose_block_rows
    env_rows = env_int(ENV_FRAME_ROWS, 0)
    given = int(rows) if (rows is not None and int(rows) > 0) else (env_rows if env_rows > 0 else None)
    r, src = choose_block_rows(rows=given, budget_bytes=budget_bytes, env=ENV_FRAME_BUDGET_MB, default_mb=1024,
                               bytes_per_row=max(1, int(n_keys) * int(bytes_per_pair)), cap_elems=None)
    if given is not None and not (rows is not None and int(rows) > 0):
        src = src.replace("given", "env:" + ENV_FRAME_ROWS, 1)
    if record:
        from .evidence import record_schedule
        record_schedule(frame_rows=r, frame_rows_source=src)
    return r, src


def nearest_atoms_rows(x, query_index, atom_mask, atom_group, k: int = 3, eps: float = 1e-8, inf: float = 1e9,
                       rows: Optional[int] = None):
    """Row-blocked masked nearest-atom search. ``x [*, N_atom, 3]``, ``query_index [*, n_q]`` (long; the query atoms), ``atom_mask [*, N_atom]``,
    ``atom_group [*, N_atom]`` (only same-group pairs are candidates; the AF3 statement uses the atom's chain id). Returns the ``topk``
    indices ``[*, n_q, k]`` of the k smallest masked distances per query row — the stock all-pairs statements evaluated on the query rows
    only, ``rows`` rows at a time (transient ``rows x N_atom x 3``)."""
    lead = tuple(x.shape[:-2])
    n_atom = int(x.shape[-2])
    query_index = query_index.long().expand(*lead, query_index.shape[-1])
    n_q = int(query_index.shape[-1])
    rb, _ = frame_rows(n_atom, rows)
    mask_k = atom_mask.expand(*lead, n_atom)
    group_k = atom_group.expand(*lead, n_atom)
    out = torch.empty(lead + (n_q, int(k)), dtype=torch.long, device=x.device)
    for t0 in range(0, n_q, rb):
        t1 = min(n_q, t0 + rb)
        q = query_index[..., t0:t1]                                                              # [*, r]
        xq = torch.gather(x, dim=-2, index=q.unsqueeze(-1).expand(*lead, t1 - t0, 3))            # [*, r, 3]
        mq = torch.gather(mask_k, dim=-1, index=q)                                               # [*, r]
        gq = torch.gather(group_k, dim=-1, index=q)                                              # [*, r]
        # ---- the stock statements, rows = the r query atoms, columns = all atoms
        pair_mask = mq[..., None] * mask_k[..., None, :]
        same_group = gq[..., None] == group_k[..., None, :]
        pair_mask = pair_mask * same_group
        d = torch.sum(eps + (xq[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5        # [*, r, N_atom]
        d = d * pair_mask + inf * (1 - pair_mask)
        _, idx = torch.topk(d, k=int(k), dim=-1, largest=False)                                  # [*, r, k]
        out[..., t0:t1, :] = idx
        del xq, mq, gq, pair_mask, same_group, d, idx
    return out


def count_pairs_within(a, b, threshold: float, rows: Optional[int] = None, max_pair_bytes: Optional[float] = None) -> int:
    """``int((torch.cdist(a, b) < threshold).sum())`` for point sets ``a [n_a, 3]``, ``b [n_b, 3]`` without an ``[n_a, n_b]`` matrix larger than
    ``max_pair_bytes``: above it the count is summed over row blocks of ``rows`` rows of ``a``."""
    if a.dim() != 2 or b.dim() != 2:
        raise RowpairRefused(f"count_pairs_within: a, b must be [n, 3]; got {tuple(a.shape)}, {tuple(b.shape)}")
    cap = float(max_pair_bytes) if max_pair_bytes is not None else float(env_int(ENV_CLASH_MAX_GB, 8)) * 1e9
    rb = int(rows) if rows else env_int(ENV_CLASH_ROWS, 8192)
    if int(a.shape[0]) * int(b.shape[0]) * 4 <= cap:
        return int((torch.cdist(a, b, p=2) < threshold).sum().item())
    n = 0
    for r0 in range(0, int(a.shape[0]), rb):
        n += int((torch.cdist(a[r0:r0 + rb], b, p=2) < threshold).sum().item())
    return n
